I have created the following plan after thorough exploration and analysis of the codebase. Follow the below plan verbatim. Trust the files and references. Do not re-verify what's written in the plan. Explore only when absolutely necessary. First implement all the proposed file changes and then I'll review all the changes together at the end.

# Cleanup Legacy Config: Remove `src/config.py` and Migrate to `core.Settings`

## Observations

The codebase has successfully migrated to the new `core.Settings` architecture, but legacy adapter code remains. The old `codewiki/src/config.py` contains:
- A `Config` dataclass used as a bridge between CLI/web app and backend
- Constants (`MODULE_TREE_FILENAME`, `MAX_TOKEN_PER_MODULE`, etc.) imported by 8+ files
- Adapter methods (`to_core_settings()`, `from_core_settings()`) for backward compatibility
- Dead code: `set_cli_context()` and `is_cli_context()` functions (defined but never used)

Current state:
- `DocumentationGenerator` accepts `Config` but immediately converts to `Settings` via `config.to_core_settings()`
- `cli/config_manager.py` already has `load_settings_for_backend()` that creates `Settings` directly—no `Config` dependency
- `src/fe/config.py` is independent (only contains `WebAppConfig` for paths/server settings)
- `DependencyGraphBuilder` uses `Config` only for `repo_path` and `dependency_graph_dir`

## Approach

**Three-phase migration** to eliminate `Config` while preserving all functionality:

1. **Extract constants** to `core/constants.py` and update all imports (8 files)
2. **Refactor backend classes** to accept `Settings` + explicit parameters instead of `Config`
3. **Delete legacy code** after verifying no remaining dependencies

This approach avoids breaking changes by:
- Keeping `Settings` interface unchanged (already validated in REFACTOR-7)
- Using explicit parameters for path-related config (clearer than nested objects)
- Maintaining CLI keyring integration via existing `ConfigManager.load_settings_for_backend()`

Trade-offs:
- `DocumentationGenerator` signature changes from `(config: Config)` to `(settings: Settings, repo_path: str, output_dir: str, dependency_graph_dir: str)` (more verbose but explicit)
- `DependencyGraphBuilder` gets similar treatment
- No functional changes—purely structural cleanup

## Implementation Steps

### Step 1: Create `core/constants.py` and Export from `core/__init__.py`

**File: `codewiki/core/constants.py` (NEW)**

Create new file with all constants from `src/config.py`:

```python
"""Constants for CodeWiki documentation generation."""

# Output directory structure
OUTPUT_BASE_DIR = 'output'
DEPENDENCY_GRAPHS_DIR = 'dependency_graphs'
DOCS_DIR = 'docs'

# Module tree filenames
FIRST_MODULE_TREE_FILENAME = 'first_module_tree.json'
MODULE_TREE_FILENAME = 'module_tree.json'
OVERVIEW_FILENAME = 'overview.md'

# Processing limits
MAX_DEPTH = 2
MAX_TOKEN_PER_MODULE = 36_369
MAX_TOKEN_PER_LEAF_MODULE = 16_000
```

**File: `codewiki/core/__init__.py`**

Add constants to exports:

```python
# Add import at top
from codewiki.core.constants import (
    FIRST_MODULE_TREE_FILENAME,
    MODULE_TREE_FILENAME,
    OVERVIEW_FILENAME,
    MAX_TOKEN_PER_MODULE,
    MAX_TOKEN_PER_LEAF_MODULE,
    OUTPUT_BASE_DIR,
    DEPENDENCY_GRAPHS_DIR,
    DOCS_DIR,
    MAX_DEPTH,
)

# Add to __all__ list
__all__ = [
    # ... existing exports ...
    "FIRST_MODULE_TREE_FILENAME",
    "MODULE_TREE_FILENAME",
    "OVERVIEW_FILENAME",
    "MAX_TOKEN_PER_MODULE",
    "MAX_TOKEN_PER_LEAF_MODULE",
    "OUTPUT_BASE_DIR",
    "DEPENDENCY_GRAPHS_DIR",
    "DOCS_DIR",
    "MAX_DEPTH",
]
```

### Step 2: Update All Constant Imports (8 Files)

Replace `from codewiki.src.config import <CONSTANT>` with `from codewiki.core import <CONSTANT>` in:

**Files to update:**
1. `codewiki/src/be/cluster_modules.py` (line 13)
2. `codewiki/src/be/agent_orchestrator.py` (lines 24-28)
3. `codewiki/src/be/documentation_generator.py` (lines 16-19)
4. `codewiki/src/be/agent_tools/generate_sub_module_documentations.py` (line 14)
5. `codewiki/cli/adapters/doc_generator.py` (line 204)

**Example change:**
```python
# Before
from codewiki.src.config import MAX_TOKEN_PER_MODULE, MODULE_TREE_FILENAME

# After
from codewiki.core import MAX_TOKEN_PER_MODULE, MODULE_TREE_FILENAME
```

### Step 3: Refactor `DependencyGraphBuilder` to Remove `Config` Dependency

**File: `codewiki/src/be/dependency_analyzer/dependency_graphs_builder.py`**

Change constructor to accept explicit parameters:

```python
# Remove Config import (line 3)
# from codewiki.src.config import Config  # DELETE

class DependencyGraphBuilder:
    """Handles dependency analysis and graph building."""
    
    def __init__(self, repo_path: str, dependency_graph_dir: str):
        """
        Initialize dependency graph builder.
        
        Args:
            repo_path: Path to repository to analyze
            dependency_graph_dir: Directory to save dependency graphs
        """
        self.repo_path = repo_path
        self.dependency_graph_dir = dependency_graph_dir
    
    def build_dependency_graph(self) -> tuple[Dict[str, Any], List[str]]:
        # Update references:
        # self.config.repo_path → self.repo_path
        # self.config.dependency_graph_dir → self.dependency_graph_dir
        
        file_manager.ensure_directory(self.dependency_graph_dir)
        
        repo_name = os.path.basename(os.path.normpath(self.repo_path))
        # ... rest of method unchanged ...
        
        parser = DependencyParser(self.repo_path)
        # ... rest unchanged ...
```

### Step 4: Refactor `DocumentationGenerator` to Accept `Settings` Directly

**File: `codewiki/src/be/documentation_generator.py`**

Update constructor and all references:

```python
# Remove Config import (line 15)
# from codewiki.src.config import Config  # DELETE

class DocumentationGenerator:
    """Main documentation generation orchestrator."""
    
    def __init__(
        self,
        settings: Settings,
        repo_path: str,
        output_dir: str,
        dependency_graph_dir: str,
        commit_id: str = None
    ):
        """
        Initialize documentation generator.
        
        Args:
            settings: Core settings with LLM config
            repo_path: Path to repository
            output_dir: Directory for generated docs
            dependency_graph_dir: Directory for dependency graphs
            commit_id: Optional commit ID for metadata
        """
        self.settings = settings
        self.repo_path = repo_path
        self.output_dir = output_dir
        self.dependency_graph_dir = dependency_graph_dir
        self.commit_id = commit_id
        
        # Remove: self.config = config
        # Remove: self.settings = config.to_core_settings()
        
        self.logger = get_logger(self.settings)
        
        # Initialize LLM clients (unchanged)
        base_client = LLMClient(self.settings, logger=self.logger)
        retry_config = RetryConfig(
            max_retries=self.settings.retry_attempts,
            base_delay=self.settings.retry_base_delay,
            fallback_models=self.settings.fallback_models
        )
        self.resilient_client = ResilientLLMClient(base_client, retry_config, self.logger)
        
        # Initialize parallel processor (unchanged)
        self.processor = ParallelModuleProcessor(
            max_concurrency=self.settings.max_concurrent_modules,
            logger=self.logger
        )
        
        # Update DependencyGraphBuilder instantiation
        self.graph_builder = DependencyGraphBuilder(
            repo_path=repo_path,
            dependency_graph_dir=dependency_graph_dir
        )
        
        # Update AgentOrchestrator instantiation
        self.agent_orchestrator = AgentOrchestrator(
            settings=self.settings,
            repo_path=repo_path,
            max_depth=self.settings.max_depth
        )
    
    # Update all methods that reference self.config:
    # - self.config.docs_dir → self.output_dir
    # - self.config.repo_path → self.repo_path
    # - self.config.max_depth → self.settings.max_depth
    
    def create_documentation_metadata(self, working_dir: str, components: Dict[str, Any], num_leaf_nodes: int):
        # ... existing code, update references:
        metadata = {
            "generation_info": {
                # ... existing fields ...
                "repo_path": self.repo_path,  # was self.settings.repo_path
                # ...
            },
            "statistics": {
                # ...
                "max_depth": self.settings.max_depth,  # unchanged
            },
            # ...
        }
        # ... rest unchanged ...
    
    async def generate_module_documentation(self, components: Dict[str, Any], leaf_nodes: List[str]) -> str:
        # Update line 291
        working_dir = os.path.abspath(self.output_dir)  # was self.config.docs_dir
        # ... rest of method unchanged ...
    
    async def generate_parent_module_docs(self, module_path: List[str], working_dir: str) -> Dict[str, Any]:
        # Update line 374
        module_name = module_path[-1] if len(module_path) >= 1 else os.path.basename(os.path.normpath(self.repo_path))
        # was: self.config.repo_path
        # ... rest unchanged ...
    
    async def run(self) -> None:
        # Update line 438
        working_dir = os.path.abspath(self.output_dir)  # was self.config.docs_dir
        # Update line 456
        repo_name = os.path.basename(os.path.normpath(self.repo_path))  # was self.config.repo_path
        # ... rest unchanged ...
```

### Step 5: Update All Callers of `DocumentationGenerator`

**File: `codewiki/src/be/main.py`**

Replace `Config.from_args()` with direct `Settings` creation:

```python
# Remove Config import (lines 25-27)
# from codewiki.src.config import Config  # DELETE

# Add Settings import
from codewiki.core import Settings
import os

async def main() -> None:
    try:
        args = parse_arguments()
        
        # Create Settings directly from environment
        settings = Settings(
            repo_path=args.repo_path,
            # Other settings loaded from env vars via CODEWIKI_ prefix
        )
        
        # Prepare output directories
        repo_name = os.path.basename(os.path.normpath(args.repo_path))
        sanitized_repo_name = ''.join(c if c.isalnum() else '_' for c in repo_name)
        output_dir = os.path.join("output", "docs", f"{sanitized_repo_name}-docs")
        dependency_graph_dir = os.path.join("output", "dependency_graphs")
        
        # Create and run documentation generator
        doc_generator = DocumentationGenerator(
            settings=settings,
            repo_path=args.repo_path,
            output_dir=output_dir,
            dependency_graph_dir=dependency_graph_dir
        )
        await doc_generator.run()
        
    except KeyboardInterrupt:
        logger.debug("Documentation generation interrupted by user")
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        raise
```

**File: `codewiki/src/fe/background_worker.py`**

Update to use `Settings` directly:

```python
# Remove Config import (line 19)
# from codewiki.src.config import Config  # DELETE

# Add Settings import
from codewiki.core import Settings

def _process_job(self, job_id: str):
    # ... existing code until line 206 ...
    
    # Replace Config.from_args() with Settings creation
    # Remove lines 207-211
    
    # Create Settings from environment variables
    settings = Settings(
        repo_path=temp_repo_dir,
        # LLM settings loaded from env vars (CODEWIKI_ANTHROPIC_API_KEY, etc.)
    )
    
    # Prepare output directories
    output_dir = os.path.join("output", "docs", f"{job_id}-docs")
    dependency_graph_dir = os.path.join("output", "dependency_graphs")
    
    job.progress = "Generating documentation..."
    
    # Create DocumentationGenerator with new signature
    doc_generator = DocumentationGenerator(
        settings=settings,
        repo_path=temp_repo_dir,
        output_dir=output_dir,
        dependency_graph_dir=dependency_graph_dir,
        commit_id=job.commit_id
    )
    
    # ... rest unchanged (lines 219-256) ...
```

**File: `codewiki/cli/adapters/doc_generator.py`**

Update to use `Settings` from `ConfigManager`:

```python
# Remove set_cli_context import (line 27)
# from codewiki.src.config import set_cli_context  # DELETE (dead code)

# In generate_documentation method (around line 135):
async def generate_documentation(self, repo_path: str) -> Path:
    # ... existing code ...
    
    try:
        # Remove set_cli_context(True) call (line 135) - dead code
        
        # Load settings from CLI config via ConfigManager (already exists, line 141)
        settings = self.config_manager.load_settings_for_backend(
            repo_path=repo_path,
            output_dir=str(self.output_dir.absolute())
        )
        
        # Prepare directories
        working_dir = str(self.output_dir.absolute())
        temp_dir = os.path.join(working_dir, "temp")
        dependency_graph_dir = os.path.join(temp_dir, "dependency_graphs")
        
        # Create DocumentationGenerator with new signature
        doc_generator = DocumentationGenerator(
            settings=settings,
            repo_path=repo_path,
            output_dir=working_dir,
            dependency_graph_dir=dependency_graph_dir
        )
        
        # Run documentation generation
        await doc_generator.run()
        
        # ... rest unchanged ...
```

### Step 6: Remove Legacy Code from `cli/models/config.py`

**File: `codewiki/cli/models/config.py`**

Delete the `to_backend_config()` method (lines 77-101):

```python
# DELETE entire method:
# def to_backend_config(self, repo_path: str, output_dir: str, api_key: str):
#     """..."""
#     from codewiki.src.config import Config
#     return Config.from_cli(...)
```

This method is no longer needed since `ConfigManager.load_settings_for_backend()` handles the conversion directly.

### Step 7: Delete `codewiki/src/config.py`

**File: `codewiki/src/config.py` (DELETE ENTIRE FILE)**

After verifying all imports are updated and tests pass, delete the entire file:

```bash
rm codewiki/src/config.py
```

### Step 8: Verification Checklist

Run these checks to ensure migration is complete:

1. **Search for remaining imports:**
   ```bash
   grep -r "from codewiki.src.config import" --include="*.py"
   grep -r "import codewiki.src.config" --include="*.py"
   ```
   Should return no results.

2. **Search for Config class usage:**
   ```bash
   grep -r "Config\\.from_args\\|Config\\.from_cli\\|Config(" --include="*.py"
   ```
   Should return no results (except in comments/docstrings).

3. **Test CLI command:**
   ```bash
   codewiki generate /path/to/repo
   ```
   Should complete successfully without errors.

4. **Check imports resolve:**
   ```python
   from codewiki.core import (
       Settings,
       MODULE_TREE_FILENAME,
       MAX_TOKEN_PER_MODULE,
       # ... all constants
   )
   ```
   Should import without errors.

## Summary of Changes

| Component | Change | Impact |
|-----------|--------|--------|
| `core/constants.py` | **NEW** - Extract 9 constants | Centralized constant management |
| `core/__init__.py` | Export constants | Public API for constants |
| `DocumentationGenerator` | Accept `Settings` + explicit paths | Clearer dependencies, no adapter |
| `DependencyGraphBuilder` | Accept explicit paths | Simpler interface |
| `main.py` | Create `Settings` directly | No `Config` dependency |
| `background_worker.py` | Create `Settings` from env | No `Config` dependency |
| `cli/adapters/doc_generator.py` | Use `ConfigManager.load_settings_for_backend()` | Already implemented |
| `cli/models/config.py` | Delete `to_backend_config()` | Remove dead code |
| `src/config.py` | **DELETE** entire file | Complete cleanup |
| 8 backend files | Update constant imports | Use `core` instead of `src.config` |

**Total files modified:** 13  
**Files deleted:** 1  
**New files:** 1  
**Lines removed:** ~210 (entire `src/config.py` + dead code)  
**Lines added:** ~50 (constants file + explicit parameters)