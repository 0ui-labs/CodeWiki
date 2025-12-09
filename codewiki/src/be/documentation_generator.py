import os
import json
from typing import Dict, List, Any
from copy import deepcopy
import traceback

# Local imports
from codewiki.src.be.dependency_analyzer import DependencyGraphBuilder
from codewiki.src.be.prompt_template import (
    REPO_OVERVIEW_PROMPT,
    MODULE_OVERVIEW_PROMPT,
)
from codewiki.src.be.cluster_modules import cluster_modules
from codewiki.src.config import (
    Config,
    FIRST_MODULE_TREE_FILENAME,
    MODULE_TREE_FILENAME,
    OVERVIEW_FILENAME
)
from codewiki.src.utils import file_manager
from codewiki.src.be.agent_orchestrator import AgentOrchestrator

# Core imports
from codewiki.core import Settings, ParallelModuleProcessor
from codewiki.core.llm import ResilientLLMClient, LLMClient, RetryConfig
from codewiki.core.logging import CodeWikiLogger, get_logger


class DocumentationGenerator:
    """Main documentation generation orchestrator."""
    
    def __init__(self, config: Config, commit_id: str = None):
        self.config = config
        self.commit_id = commit_id

        # Convert old Config to new Settings
        self.settings = config.to_core_settings()
        self.logger = get_logger(self.settings)

        # Initialize LLM clients
        base_client = LLMClient(self.settings, logger=self.logger)
        retry_config = RetryConfig(
            max_retries=self.settings.retry_attempts,
            base_delay=self.settings.retry_base_delay,
            fallback_models=self.settings.fallback_models
        )
        self.resilient_client = ResilientLLMClient(base_client, retry_config, self.logger)

        # Initialize parallel processor
        self.processor = ParallelModuleProcessor(
            max_concurrency=self.settings.max_concurrent_modules,
            logger=self.logger
        )

        # Keep existing components
        self.graph_builder = DependencyGraphBuilder(config)
        self.agent_orchestrator = AgentOrchestrator(
            settings=self.settings,
            repo_path=config.repo_path,
            max_depth=config.max_depth
        )
    
    def create_documentation_metadata(self, working_dir: str, components: Dict[str, Any], num_leaf_nodes: int):
        """Create a metadata file with documentation generation information."""
        from datetime import datetime
        
        metadata = {
            "generation_info": {
                "timestamp": datetime.now().isoformat(),
                "main_model": self.config.main_model,
                "generator_version": "1.0.0",
                "repo_path": self.config.repo_path,
                "commit_id": self.commit_id
            },
            "statistics": {
                "total_components": len(components),
                "leaf_nodes": num_leaf_nodes,
                "max_depth": self.config.max_depth
            },
            "files_generated": [
                "overview.md",
                "module_tree.json",
                "first_module_tree.json"
            ]
        }
        
        # Add generated markdown files to the metadata
        try:
            for file_path in os.listdir(working_dir):
                if file_path.endswith('.md') and file_path not in metadata["files_generated"]:
                    metadata["files_generated"].append(file_path)
        except Exception as e:
            self.logger.warning(f"Could not list generated files: {e}")
        
        metadata_path = os.path.join(working_dir, "metadata.json")
        file_manager.save_json(metadata, metadata_path)

    
    def get_processing_order(self, module_tree: Dict[str, Any], parent_path: List[str] = []) -> List[tuple[List[str], str]]:
        """Get the processing order using topological sort (leaf modules first)."""
        processing_order = []
        
        def collect_modules(tree: Dict[str, Any], path: List[str]):
            for module_name, module_info in tree.items():
                current_path = path + [module_name]
                
                # If this module has children, process them first
                if module_info.get("children") and isinstance(module_info["children"], dict) and module_info["children"]:
                    collect_modules(module_info["children"], current_path)
                    # Add this parent module after its children
                    processing_order.append((current_path, module_name))
                else:
                    # This is a leaf module, add it immediately
                    processing_order.append((current_path, module_name))
        
        collect_modules(module_tree, parent_path)
        return processing_order

    def is_leaf_module(self, module_info: Dict[str, Any]) -> bool:
        """Check if a module is a leaf module (has no children or empty children)."""
        children = module_info.get("children", {})
        return not children or (isinstance(children, dict) and len(children) == 0)

    def _build_dependency_graph(self, module_tree: dict) -> dict[str, list[str]]:
        """Build dependency graph where parents depend on their children.

        Returns:
            dict mapping module_name -> list of module names it depends on.
            Parent modules depend on their children (must wait for children to complete).
            Leaf modules have empty dependency lists (can run immediately).
        """
        graph: dict[str, list[str]] = {}

        def traverse(node: dict, parent_path: str = "") -> None:
            for module_name, module_data in node.items():
                full_path = f"{parent_path}/{module_name}" if parent_path else module_name

                children = module_data.get("children", {})
                child_names: list[str] = []

                if children and isinstance(children, dict):
                    for child_name in children.keys():
                        child_full_path = f"{full_path}/{child_name}"
                        child_names.append(child_full_path)
                    traverse(children, full_path)

                graph[full_path] = child_names

        traverse(module_tree)
        self._validate_no_cycles(graph)
        return graph

    def _validate_no_cycles(self, graph: dict[str, list[str]]) -> None:
        """Validate that the dependency graph has no cycles.

        Uses DFS with three-color marking to detect cycles.

        Args:
            graph: Dependency graph to validate

        Raises:
            ValueError: If a cycle is detected
        """
        WHITE, GRAY, BLACK = 0, 1, 2
        color: dict[str, int] = {node: WHITE for node in graph}

        def dfs(node: str, path: list[str]) -> None:
            if color[node] == GRAY:
                cycle = path[path.index(node):] + [node]
                raise ValueError(f"Cycle detected in dependency graph: {' -> '.join(cycle)}")
            if color[node] == BLACK:
                return

            color[node] = GRAY
            path.append(node)
            for dep in graph.get(node, []):
                if dep in graph:
                    dfs(dep, path)
            path.pop()
            color[node] = BLACK

        for node in graph:
            if color[node] == WHITE:
                dfs(node, [])

    def build_overview_structure(self, module_tree: Dict[str, Any], module_path: List[str],
                                 working_dir: str) -> Dict[str, Any]:
        """Build structure for overview generation with 1-depth children docs and target indicator."""
        
        processed_module_tree = deepcopy(module_tree)
        module_info = processed_module_tree
        for path_part in module_path:
            module_info = module_info[path_part]
            if path_part != module_path[-1]:
                module_info = module_info.get("children", {})
            else:
                module_info["is_target_for_overview_generation"] = True

        if "children" in module_info:
            module_info = module_info["children"]

        for child_name, child_info in module_info.items():
            if os.path.exists(os.path.join(working_dir, f"{child_name}.md")):
                child_info["docs"] = file_manager.load_text(os.path.join(working_dir, f"{child_name}.md"))
            else:
                self.logger.warning(f"Module docs not found at {os.path.join(working_dir, f"{child_name}.md")}")
                child_info["docs"] = ""

        return processed_module_tree

    def _collect_modules_for_processing(
        self,
        module_tree: Dict[str, Any],
        components: Dict[str, Any],
        working_dir: str,
    ) -> List[Dict[str, Any]]:
        """Collect all modules into a flat list with metadata for parallel processing.

        Args:
            module_tree: The hierarchical module tree
            components: All code components
            working_dir: Output directory for documentation

        Returns:
            List of module dicts with 'name', 'path', 'is_leaf', 'components', 'working_dir'
        """
        modules: List[Dict[str, Any]] = []

        def traverse(node: Dict[str, Any], path: List[str]) -> None:
            for module_name, module_info in node.items():
                current_path = path + [module_name]
                module_key = "/".join(current_path)

                is_leaf = self.is_leaf_module(module_info)

                modules.append({
                    "name": module_key,  # Used as key by ParallelModuleProcessor
                    "module_name": module_name,
                    "path": current_path,
                    "is_leaf": is_leaf,
                    "components": module_info.get("components", []),
                    "working_dir": working_dir,
                    "all_components": components,
                })

                # Recursively process children
                children = module_info.get("children", {})
                if children and isinstance(children, dict):
                    traverse(children, current_path)

        traverse(module_tree, [])
        return modules

    async def generate_module_documentation(self, components: Dict[str, Any], leaf_nodes: List[str]) -> str:
        """Generate documentation for all modules using parallel processing.

        Uses ParallelModuleProcessor to process independent modules concurrently
        while respecting dependencies (parents wait for children).
        """
        # Prepare output directory
        working_dir = os.path.abspath(self.config.docs_dir)
        file_manager.ensure_directory(working_dir)

        module_tree_path = os.path.join(working_dir, MODULE_TREE_FILENAME)
        first_module_tree_path = os.path.join(working_dir, FIRST_MODULE_TREE_FILENAME)
        module_tree = file_manager.load_json(module_tree_path)
        first_module_tree = file_manager.load_json(first_module_tree_path)

        if len(module_tree) > 0:
            # Build dependency graph for parallel processing
            dep_graph = self._build_dependency_graph(first_module_tree)

            # Collect modules for parallel processing
            modules = self._collect_modules_for_processing(module_tree, components, working_dir)

            # Define the async processing function for each module
            async def process_single_module(module_info: Dict[str, Any]) -> Dict[str, Any]:
                """Process a single module (leaf or parent)."""
                module_key = module_info["name"]
                module_name = module_info["module_name"]
                module_path = module_info["path"]
                is_leaf = module_info["is_leaf"]

                if is_leaf:
                    return await self.agent_orchestrator.process_module(
                        module_name,
                        module_info["all_components"],
                        module_info["components"],
                        module_path,
                        module_info["working_dir"],
                    )
                else:
                    return await self.generate_parent_module_docs(
                        module_path,
                        module_info["working_dir"],
                    )

            # Process all modules in parallel (respecting dependencies)
            self.logger.info(f"Processing {len(modules)} modules in parallel (max concurrency: {self.processor.max_concurrency})")
            try:
                results = await self.processor.process_modules(
                    modules=modules,
                    dependency_graph=dep_graph,
                    process_fn=process_single_module,
                )
                self.logger.success(f"Completed processing {len(results)} modules")
            except ExceptionGroup as eg:
                # Log failed modules but continue with successful ones
                for exc in eg.exceptions:
                    self.logger.error(f"Module processing failed: {exc}")
                # Some modules may have succeeded despite the group failure

            # Generate repo overview (depends on all modules, so process last)
            self.logger.info("Generating repository overview")
            await self.generate_parent_module_docs([], working_dir)

        else:
            self.logger.info("Processing whole repo because repo can fit in the context window")
            repo_name = os.path.basename(os.path.normpath(self.config.repo_path))
            await self.agent_orchestrator.process_module(
                repo_name, components, leaf_nodes, [], working_dir
            )

            # save module_tree to module_tree.json
            file_manager.save_json(module_tree, os.path.join(working_dir, MODULE_TREE_FILENAME))

            # rename repo_name.md to overview.md
            repo_overview_path = os.path.join(working_dir, f"{repo_name}.md")
            if os.path.exists(repo_overview_path):
                os.rename(repo_overview_path, os.path.join(working_dir, OVERVIEW_FILENAME))

        return working_dir

    async def generate_parent_module_docs(self, module_path: List[str],
                                        working_dir: str) -> Dict[str, Any]:
        """Generate documentation for a parent module based on its children's documentation."""
        module_name = module_path[-1] if len(module_path) >= 1 else os.path.basename(os.path.normpath(self.config.repo_path))

        self.logger.info(f"Generating parent documentation for: {module_name}")

        # Load module tree
        module_tree_path = os.path.join(working_dir, MODULE_TREE_FILENAME)
        module_tree = file_manager.load_json(module_tree_path)

        # check if overview docs already exists
        overview_docs_path = os.path.join(working_dir, OVERVIEW_FILENAME)
        if os.path.exists(overview_docs_path):
            self.logger.info(f"Overview docs already exists at {overview_docs_path}")
            return module_tree

        # check if parent docs already exists
        parent_docs_path = os.path.join(working_dir, f"{module_name if len(module_path) >= 1 else OVERVIEW_FILENAME.replace('.md', '')}.md")
        if os.path.exists(parent_docs_path):
            self.logger.info(f"Parent docs already exists at {parent_docs_path}")
            return module_tree

        # Create repo structure with 1-depth children docs and target indicator
        repo_structure = self.build_overview_structure(module_tree, module_path, working_dir)

        prompt = MODULE_OVERVIEW_PROMPT.format(
            module_name=module_name,
            repo_structure=json.dumps(repo_structure, indent=4)
        ) if len(module_path) >= 1 else REPO_OVERVIEW_PROMPT.format(
            repo_name=module_name,
            repo_structure=json.dumps(repo_structure, indent=4)
        )

        try:
            # Use resilient LLM client
            llm_response = await self.resilient_client.complete(
                messages=[{"role": "user", "content": prompt}],
                model=self.settings.main_model,
                temperature=0.0,
                max_tokens=32768
            )
            parent_docs = llm_response.content

            # Parse and save parent documentation
            parent_content = parent_docs.split("<OVERVIEW>")[1].split("</OVERVIEW>")[0].strip()
            # parent_content = prompt
            file_manager.save_text(parent_content, parent_docs_path)

            self.logger.debug(f"Successfully generated parent documentation for: {module_name}")
            return module_tree

        except Exception as e:
            self.logger.error(f"Error generating parent documentation for {module_name}: {str(e)}")
            raise
    
    async def run(self) -> None:
        """Run the complete documentation generation process using dynamic programming."""
        try:
            # Build dependency graph
            components, leaf_nodes = self.graph_builder.build_dependency_graph()

            self.logger.debug(f"Found {len(leaf_nodes)} leaf nodes")
            # self.logger.debug(f"Leaf nodes:\n{'\n'.join(sorted(leaf_nodes)[:200])}")
            # exit()

            # Cluster modules
            working_dir = os.path.abspath(self.config.docs_dir)
            file_manager.ensure_directory(working_dir)
            first_module_tree_path = os.path.join(working_dir, FIRST_MODULE_TREE_FILENAME)
            module_tree_path = os.path.join(working_dir, MODULE_TREE_FILENAME)

            # Check if module tree exists
            if os.path.exists(first_module_tree_path):
                self.logger.debug(f"Module tree found at {first_module_tree_path}")
                module_tree = file_manager.load_json(first_module_tree_path)
            else:
                self.logger.debug(f"Module tree not found at {module_tree_path}, clustering modules")
                module_tree = await cluster_modules(
                    leaf_nodes, components, self.settings,
                    self.resilient_client, self.logger
                )
                file_manager.save_json(module_tree, first_module_tree_path)

            file_manager.save_json(module_tree, module_tree_path)

            self.logger.debug(f"Grouped components into {len(module_tree)} modules")
            
            # Generate module documentation using dynamic programming approach
            # This processes leaf modules first, then parent modules
            working_dir = await self.generate_module_documentation(components, leaf_nodes)
            
            # Create documentation metadata
            self.create_documentation_metadata(working_dir, components, len(leaf_nodes))

            self.logger.debug(f"Documentation generation completed successfully using dynamic programming!")
            self.logger.debug(f"Processing order: leaf modules → parent modules → repository overview")
            self.logger.debug(f"Documentation saved to: {working_dir}")

        except Exception as e:
            self.logger.error(f"Documentation generation failed: {str(e)}")
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            raise