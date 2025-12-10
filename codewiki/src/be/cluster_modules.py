from typing import List, Dict, Any, Optional
from collections import defaultdict

from codewiki.src.be.dependency_analyzer.models.core import Node
from codewiki.core import Settings, MAX_TOKEN_PER_MODULE
from codewiki.core.llm import ResilientLLMClient
from codewiki.core.llm.tokenizers import TokenCounter
from codewiki.core.logging import CodeWikiLogger
from codewiki.src.be.prompt_template import format_cluster_prompt


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

    if current_module_tree == {}:
        current_module_tree = module_tree
    else:
        value = current_module_tree
        for key in current_module_path:
            value = value[key]["children"]
        for module_name, module_info in module_tree.items():
            del module_info["path"]
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
        if sub_module_tree is not None:
            module_info["children"] = sub_module_tree
        current_module_path.pop()

    return module_tree