from typing import List, Dict, Any, Optional
from collections import defaultdict
from pathlib import Path

from codewiki.src.be.dependency_analyzer.models.core import Node
from codewiki.core import Settings, MAX_TOKEN_PER_MODULE
from codewiki.core.llm import ResilientLLMClient
from codewiki.core.llm.tokenizers import TokenCounter
from codewiki.core.logging import CodeWikiLogger
from codewiki.core.utils.hashing import ModuleHasher
from codewiki.src.be.prompt_template import format_cluster_prompt


def _enrich_module_metadata(
    module_info: Dict[str, Any],
    components: Dict[str, Node],
    repo_path: str,
    logger: CodeWikiLogger
) -> None:
    """
    Enrich module with MCP-required metadata: path, type, hash.

    Args:
        module_info: Module dict to enrich (modified in-place)
        components: All code components for path inference
        repo_path: Repository root path for filesystem checks
        logger: Logger instance
    """
    # 1. Preserve or infer path
    if "path" not in module_info or not module_info["path"]:
        # Fallback: Infer from first component's directory
        module_components = module_info.get("components", [])
        path_inferred = False

        if module_components:
            # Skip invalid components and use first valid one
            for component_id in module_components:
                if component_id in components:
                    first_component = components[component_id]
                    # Extract directory from relative_path (e.g., "src/be/utils.py" -> "src/be")
                    parent_path = Path(first_component.relative_path).parent
                    # Convert "." to empty string for root directory
                    module_info["path"] = "" if str(parent_path) == "." else str(parent_path)
                    path_inferred = True
                    break

        if not path_inferred:
            # No components or all invalid: use empty path (virtual module)
            module_info["path"] = ""
            logger.debug("Module has no valid path, marking as virtual")

    # 2. Determine type: "package" (real directory) or "virtual" (logical grouping)
    # Resolve paths to prevent directory traversal attacks (e.g., "../../etc/passwd")
    if module_info["path"]:
        repo_root = Path(repo_path).resolve()
        full_path = (repo_root / module_info["path"]).resolve()

        # Security check: ensure resolved path is inside repository root
        try:
            full_path.relative_to(repo_root)
            if full_path.is_dir():
                module_info["type"] = "package"
            elif full_path.is_file():
                # LLM returned a file path - normalize to parent directory
                parent_path = full_path.parent
                relative_parent = parent_path.relative_to(repo_root)
                # Convert to string, use empty string for root directory
                module_info["path"] = "" if str(relative_parent) == "." else str(relative_parent)
                module_info["type"] = "package"
                logger.debug(
                    f"Normalized file path to parent directory: '{module_info['path']}'"
                )
            else:
                # Path exists in repo but is neither file nor directory
                logger.warning(
                    f"Module path '{module_info['path']}' exists but is neither file nor directory, "
                    f"marking as virtual"
                )
                module_info["type"] = "virtual"
        except ValueError:
            # Path escapes repo root - treat as virtual and log warning
            logger.warning(
                f"Module path '{module_info['path']}' resolves outside repository root, "
                f"marking as virtual"
            )
            module_info["type"] = "virtual"
    else:
        # Empty path: check if this is a root module with components or a virtual module
        # If there are valid components in the root, it's a package; otherwise virtual
        module_components = module_info.get("components", [])
        has_valid_components = any(comp_id in components for comp_id in module_components)

        if has_valid_components:
            # Module has components in root directory
            repo_root = Path(repo_path).resolve()
            module_info["type"] = "package" if repo_root.is_dir() else "virtual"
        else:
            # No valid components - this is a virtual/logical grouping
            module_info["type"] = "virtual"

    # 3. Initialize hash (empty for now, will be computed by compute_module_tree_hashes)
    # Only set empty hash if not already set (preserve existing non-empty hashes)
    if "hash" not in module_info or not module_info["hash"]:
        module_info["hash"] = ""


def compute_module_tree_hashes(
    module_tree: Dict[str, Any],
    components: Dict[str, Node],
    logger: CodeWikiLogger
) -> None:
    """
    Compute and populate hash values for all modules in the tree using bottom-up traversal.

    This function implements Merkle-style parent-child hashing where:
    - Leaf modules get hashes based on their component content hashes
    - Parent modules get hashes based on their component hashes + child module hashes

    Args:
        module_tree: The complete module tree to populate with hashes (modified in-place)
        components: All code components with their Node data (containing source_code)
        logger: Logger instance for warnings/errors
    """
    def compute_hashes_recursive(tree: Dict[str, Any]) -> None:
        """Recursively compute hashes bottom-up (children first, then parents)."""
        for module_name, module_info in tree.items():
            # First, recursively process children (bottom-up traversal)
            children = module_info.get("children", {})
            if children and isinstance(children, dict):
                compute_hashes_recursive(children)

            # Compute component hashes from source code
            component_hashes: List[str] = []
            module_components = module_info.get("components", [])
            for component_id in module_components:
                if component_id in components:
                    node = components[component_id]
                    if node.source_code:
                        content_hash = ModuleHasher.compute_content_hash(node.source_code)
                        component_hashes.append(content_hash)
                    else:
                        logger.debug(
                            f"Component '{component_id}' in module '{module_name}' has no source code"
                        )
                else:
                    logger.warning(
                        f"Component '{component_id}' in module '{module_name}' not found in components dict"
                    )

            # Collect child hashes (only non-empty ones)
            child_hashes: List[str] = []
            for child_name, child_info in children.items():
                child_hash = child_info.get("hash", "")
                if child_hash:
                    child_hashes.append(child_hash)
                else:
                    logger.warning(
                        f"Child module '{child_name}' of '{module_name}' has empty hash"
                    )

            # Compute module hash using Merkle-style combination
            module_hash = ModuleHasher.compute_module_hash(component_hashes, child_hashes)
            module_info["hash"] = module_hash

            logger.debug(
                f"Computed hash for module '{module_name}': {module_hash[:16]}... "
                f"(from {len(component_hashes)} components, {len(child_hashes)} children)"
            )

    # Start recursive hash computation from the root of the tree
    compute_hashes_recursive(module_tree)


def format_potential_core_components(
    leaf_nodes: List[str], components: Dict[str, Node], logger: CodeWikiLogger
) -> tuple[str, str]:
    """
    Format the potential core components into a string that can be used in the prompt.
    """
    # Filter out any invalid leaf nodes that don't exist in components
    valid_leaf_nodes = []
    for leaf_node in leaf_nodes:
        if leaf_node in components:
            valid_leaf_nodes.append(leaf_node)
        else:
            logger.warning(f"Skipping invalid leaf node '{leaf_node}' - not found in components")
    
    #group leaf nodes by file
    leaf_nodes_by_file = defaultdict(list)
    for leaf_node in valid_leaf_nodes:
        leaf_nodes_by_file[components[leaf_node].relative_path].append(leaf_node)

    potential_core_components = ""
    potential_core_components_with_code = ""
    for file, leaf_nodes in dict(sorted(leaf_nodes_by_file.items())).items():
        potential_core_components += f"# {file}\n"
        potential_core_components_with_code += f"# {file}\n"
        for leaf_node in leaf_nodes:
            potential_core_components += f"\t{leaf_node}\n"
            potential_core_components_with_code += f"\t{leaf_node}\n"
            potential_core_components_with_code += f"{components[leaf_node].source_code}\n"

    return potential_core_components, potential_core_components_with_code


async def cluster_modules(
    leaf_nodes: List[str],
    components: Dict[str, Node],
    settings: Settings,
    resilient_client: ResilientLLMClient,
    logger: CodeWikiLogger,
    current_module_tree: dict[str, Any] = {},
    current_module_name: str = None,
    current_module_path: List[str] = []
) -> Optional[Dict[str, Any]]:
    """
    Cluster the potential core components into modules.

    Returns:
        A dict representing the module tree if clustering was performed,
        or None if no clustering occurred (e.g., token count below threshold,
        invalid LLM response, or resulting module tree too small). Callers
        should check for None to avoid overwriting existing module structures.
    """
    potential_core_components, potential_core_components_with_code = format_potential_core_components(
        leaf_nodes, components, logger
    )

    token_counter = TokenCounter()
    model = settings.cluster_model or settings.main_model
    token_count = token_counter.count(potential_core_components_with_code, model)

    if token_count <= MAX_TOKEN_PER_MODULE:
        logger.debug(
            f"Skipping clustering for {current_module_name} because the potential core components "
            f"are too few: {token_count} tokens"
        )
        return None

    prompt = format_cluster_prompt(potential_core_components, current_module_tree, current_module_name)

    llm_response = await resilient_client.complete(
        messages=[{"role": "user", "content": prompt}],
        model=model,
        temperature=0.0,
        max_tokens=32768
    )
    response = llm_response.content

    # parse the response
    try:
        if "<GROUPED_COMPONENTS>" not in response or "</GROUPED_COMPONENTS>" not in response:
            logger.error(f"Invalid LLM response format - missing component tags: {response[:200]}...")
            return None

        response_content = response.split("<GROUPED_COMPONENTS>")[1].split("</GROUPED_COMPONENTS>")[0]
        # Use ast.literal_eval for safe parsing (no arbitrary code execution)
        import ast
        module_tree = ast.literal_eval(response_content)
        
        if not isinstance(module_tree, dict):
            logger.error(f"Invalid module tree format - expected dict, got {type(module_tree)}")
            return None

    except Exception as e:
        logger.error(f"Failed to parse LLM response: {e}. Response: {response[:200]}...")
        return None

    # check if the module tree is valid
    if len(module_tree) <= 1:
        logger.debug(f"Skipping clustering for {current_module_name} because the module tree is too small: {len(module_tree)} modules")
        return None

    # Enrich all modules with MCP metadata (path, type, hash)
    for module_name, module_info in module_tree.items():
        _enrich_module_metadata(module_info, components, settings.repo_path, logger)

    if current_module_tree == {}:
        current_module_tree = module_tree
    else:
        value = current_module_tree
        for key in current_module_path:
            value = value[key]["children"]
        for module_name, module_info in module_tree.items():
            # Note: path is now preserved (not deleted) for MCP server use
            value[module_name] = module_info

    for module_name, module_info in module_tree.items():
        sub_leaf_nodes = module_info.get("components", [])
        
        # Filter sub_leaf_nodes to ensure they exist in components
        valid_sub_leaf_nodes = []
        for node in sub_leaf_nodes:
            if node in components:
                valid_sub_leaf_nodes.append(node)
            else:
                logger.warning(f"Skipping invalid sub leaf node '{node}' in module '{module_name}' - not found in components")
        
        current_module_path.append(module_name)
        module_info["children"] = {}
        sub_module_tree = await cluster_modules(
            valid_sub_leaf_nodes,
            components,
            settings,
            resilient_client,
            logger,
            current_module_tree,
            module_name,
            current_module_path
        )
        # Only update children if clustering produced a result
        # Note: Child modules are already enriched by recursive cluster_modules() call
        # at line 187 (MCP metadata enrichment happens at the start of each recursion)
        if sub_module_tree is not None:
            module_info["children"] = sub_module_tree
        current_module_path.pop()

    return module_tree