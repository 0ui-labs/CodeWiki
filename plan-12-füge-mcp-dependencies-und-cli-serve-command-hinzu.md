I have created the following plan after thorough exploration and analysis of the codebase. Follow the below plan verbatim. Trust the files and references. Do not re-verify what's written in the plan. Explore only when absolutely necessary. First implement all the proposed file changes and then I'll review all the changes together at the end.

## Observations

The codebase currently lacks MCP (Model Context Protocol) server functionality. The `pyproject.toml` already includes most dependencies (anthropic, google-genai, etc.) but is missing the `mcp>=1.0.0` package. The `codewiki/cli/commands/serve.py` file exists but is empty. The CLI follows a consistent pattern using Click decorators, error handling via `handle_error()`, and command registration in `main.py` via `cli.add_command()`. The documentation generator creates a structured output directory containing `module_tree.json`, `metadata.json`, `overview.md`, and module-specific markdown files—these artifacts will be served by the MCP server in subsequent phases.

## Approach

This phase establishes the CLI foundation for MCP server functionality without implementing the actual server logic (deferred to next phase). We'll add the `mcp` dependency to enable FastMCP imports, create a minimal `serve` command that validates the documentation artifacts (specifically `module_tree.json` as the critical file), and register it in the CLI entry point. The command will follow existing CLI patterns: Click-based with `--docs-dir` option (defaulting to `./output/docs`), path validation, clear error messages using the established `ConfigurationError`/`handle_error` pattern, and user-friendly output with colored messages. This approach ensures the CLI interface is stable before the complex MCP server implementation begins, allows independent testing of the command structure, and maintains consistency with existing commands like `generate` and `config`.

## Implementation Steps

### 1. Add MCP Dependency to pyproject.toml

**File:** `pyproject.toml`

**Changes:**
- Add `"mcp>=1.0.0"` to the `dependencies` list (after line 57, before the closing bracket)
- Ensure it's placed logically near other SDK dependencies (anthropic, google-genai, etc.)

**Validation:**
- Run `pip install -e .` to verify the dependency resolves correctly
- Check that `import mcp` works in a Python shell

---

### 2. Create CLI Serve Command

**File:** `codewiki/cli/commands/serve.py`

**Implementation:**

```python
"""
Serve command for MCP server.
"""

import sys
import click
from pathlib import Path

from codewiki.cli.utils.errors import (
    ConfigurationError,
    handle_error,
    EXIT_SUCCESS,
)


@click.command(name="serve")
@click.option(
    "--docs-dir",
    type=click.Path(exists=True, file_okay=False, dir_okay=True),
    default="./output/docs",
    help="Directory containing generated documentation (default: ./output/docs)",
)
def serve_command(docs_dir: str):
    """
    Start the CodeWiki MCP Server.
    
    This allows AI agents (like Claude Desktop, Cursor, or Windsurf) to connect
    to CodeWiki and query the documentation interactively via the Model Context
    Protocol (MCP).
    
    The server provides:
      • Resources: Direct access to documentation files (overview, modules)
      • Tools: Intelligent navigation and search capabilities
    
    Examples:
    
    \b
    # Start server with default docs directory
    $ codewiki serve
    
    \b
    # Start server with custom docs directory
    $ codewiki serve --docs-dir /path/to/docs
    
    \b
    # Configure in Claude Desktop (claude_desktop_config.json):
    {
      "mcpServers": {
        "codewiki": {
          "command": "codewiki",
          "args": ["serve", "--docs-dir", "/absolute/path/to/docs"]
        }
      }
    }
    """
    try:
        # Resolve and validate path
        docs_path = Path(docs_dir).expanduser().resolve()
        
        if not docs_path.exists():
            raise ConfigurationError(
                f"Documentation directory not found: {docs_path}\n\n"
                "Please run 'codewiki generate' first to create documentation."
            )
        
        # Check for required artifacts
        module_tree_path = docs_path / "module_tree.json"
        if not module_tree_path.exists():
            raise ConfigurationError(
                f"No module_tree.json found in {docs_path}\n\n"
                "The documentation appears incomplete or corrupted.\n"
                "Please run 'codewiki generate' to regenerate documentation."
            )
        
        # Display startup information
        click.echo()
        click.secho("Starting CodeWiki MCP Server...", fg="green", bold=True)
        click.echo()
        click.secho(f"📁 Documentation directory: {docs_path}", fg="blue")
        click.secho(f"✓ Module tree validated", fg="green")
        click.echo()
        click.secho("Connect this server to your AI agent via stdio.", fg="cyan")
        click.echo("See documentation for Claude Desktop/Cursor/Windsurf configuration.")
        click.echo()
        
        # Import and start server (implementation in next phase)
        try:
            from codewiki.src.be.mcp_server import run_server
            run_server(str(docs_path))
        except ImportError:
            raise ConfigurationError(
                "MCP server implementation not found.\n\n"
                "This feature is under development. The server core will be\n"
                "implemented in the next phase."
            )
        
    except ConfigurationError as e:
        click.secho(f"\n✗ {e.message}", fg="red", err=True)
        sys.exit(e.exit_code)
    except KeyboardInterrupt:
        click.echo("\n\nServer stopped by user")
        sys.exit(EXIT_SUCCESS)
    except Exception as e:
        sys.exit(handle_error(e))
```

**Key Design Decisions:**
- **Path validation:** Uses `Path.expanduser().resolve()` for robust path handling (supports `~`, relative paths, symlinks)
- **Artifact validation:** Only checks `module_tree.json` (critical file), not all artifacts, to keep validation fast
- **Error messages:** Provides actionable guidance (e.g., "run codewiki generate")
- **Graceful degradation:** Catches `ImportError` for `run_server` to allow testing before server implementation
- **User experience:** Clear startup messages with colored output, configuration examples in help text

---

### 3. Register Serve Command in CLI Main

**File:** `codewiki/cli/main.py`

**Changes:**

1. Add import after line 35 (with other command imports):
```python
from codewiki.cli.commands.serve import serve_command
```

2. Add command registration after line 39 (with other `cli.add_command()` calls):
```python
cli.add_command(serve_command, name="serve")
```

**Result:** The `serve` command becomes available via `codewiki serve`

---

### 4. Testing and Validation

**Manual Tests:**

1. **Help text:**
   ```bash
   codewiki serve --help
   ```
   - Verify description, options, and examples display correctly
   - Check formatting is consistent with other commands

2. **Error handling (missing directory):**
   ```bash
   codewiki serve --docs-dir /nonexistent/path
   ```
   - Should show clear error: "Documentation directory not found"
   - Exit code should be 2 (EXIT_CONFIG_ERROR)

3. **Error handling (missing module_tree.json):**
   ```bash
   mkdir -p /tmp/empty_docs
   codewiki serve --docs-dir /tmp/empty_docs
   ```
   - Should show error: "No module_tree.json found"
   - Suggests running `codewiki generate`

4. **Valid directory (before server implementation):**
   ```bash
   codewiki generate  # Create docs first
   codewiki serve
   ```
   - Should validate successfully
   - Show startup messages
   - Fail gracefully with ImportError message (expected until next phase)

5. **CLI integration:**
   ```bash
   codewiki --help
   ```
   - Verify `serve` appears in command list

**Automated Tests (optional, for future):**
- Unit test for path validation logic
- Unit test for `module_tree.json` check
- Integration test with mock `run_server`

---

## Summary

This implementation adds the MCP CLI foundation:
- **Dependency:** `mcp>=1.0.0` in `pyproject.toml` (~1 line)
- **Command:** `codewiki/cli/commands/serve.py` with validation and error handling (~100 lines)
- **Registration:** Import and register in `main.py` (~2 lines)

**Total:** ~103 lines of new code, 0 files modified (only additions)

**Benefits:**
- Stable CLI interface before complex server logic
- Consistent with existing command patterns
- Clear error messages guide users
- Ready for server implementation in next phase

**Next Phase Preview:**
The subsequent phase will implement `codewiki/src/be/mcp_server.py` with FastMCP, resources (`codewiki://overview`, `codewiki://module/{path}`), and tools (`list_modules`, `search_modules`). The `serve_command` will seamlessly integrate by calling `run_server()`.