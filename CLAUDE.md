# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

CodeWiki is an AI-powered framework that generates comprehensive repository-level documentation for codebases in Python, Java, JavaScript, TypeScript, C, C++, and C#. It uses Tree-sitter for AST parsing, LLM-based module clustering, and multi-agent documentation generation.

## Commands

```bash
# Install for development
pip install -e .

# Run the CLI
codewiki config set --api-key <key> --main-model <model>
codewiki config show
codewiki generate --output ./docs --verbose

# Linting & formatting
black --line-length 100 .
ruff check --line-length 100 .
mypy codewiki/

# Testing
pytest                                    # Run all tests with coverage
pytest tests/test_specific.py             # Run single test file
pytest tests/test_file.py::test_name -v   # Run single test
```

## Architecture

### Three-Stage Pipeline

1. **Code Analysis** (`codewiki/src/be/dependency_analyzer/`): Tree-sitter parses source files, builds dependency graphs, extracts components (functions, classes, modules), and performs topological sorting.

2. **Module Clustering** (`codewiki/src/be/cluster_modules.py`): LLM groups leaf components into feature-oriented modules, creating a hierarchical module tree (max depth: 2).

3. **Documentation Generation** (`codewiki/src/be/agent_orchestrator.py`): Multi-agent system processes modules leaf-first. Complex modules use full agents with sub-module tools; leaf modules use lightweight agents.

### Key Directories

- `codewiki/cli/` - Click-based CLI with commands in `commands/`, config in `config_manager.py`
- `codewiki/src/be/` - Backend: `documentation_generator.py` (orchestrator), `llm_services.py` (LiteLLM abstraction), `agent_tools/` (Pydantic AI tools)
- `codewiki/src/be/dependency_analyzer/analyzers/` - Language-specific analyzers (one file per language)
- `codewiki/src/fe/` - FastAPI web app with background workers
- `codewiki/templates/github_pages/` - Jinja2 templates for HTML output

### Configuration

- User config stored at `~/.codewiki/config.json`
- API keys stored securely in system keyring
- LLM integration via LiteLLM (supports Anthropic, OpenAI, etc.)

### Adding a New Language

1. Create analyzer in `codewiki/src/be/dependency_analyzer/analyzers/new_lang.py` extending `BaseAnalyzer`
2. Register in `codewiki/src/be/dependency_analyzer/ast_parser.py` `LANGUAGE_ANALYZERS` dict
3. Add file extensions to configuration
4. Add corresponding tests

## External Dependencies

- **Node.js**: Required for Mermaid diagram validation
- **Git**: Required for branch creation and repo analysis

## Known Architectural Limitations

### LLM Integration requires OpenAI-compatible Proxy

The code in `llm_services.py` uses `OpenAI` client and `pydantic_ai.providers.openai.OpenAIProvider` for all LLM calls. Direct connection to Anthropic API does not work.

**Workaround:** Run LiteLLM as proxy server:
```bash
litellm --model claude-3-5-sonnet --port 4000
```
Then set `LLM_BASE_URL=http://127.0.0.1:4000`

### Token Counting Inaccuracy

`codewiki/src/be/utils.py:29` uses `tiktoken.encoding_for_model("gpt-4")` for all models. Token counts will be inaccurate for Claude models, potentially causing context window overflows.

### Sync LLM Calls in `call_llm()`

The `call_llm()` function in `llm_services.py` is synchronous despite being called from async contexts. The Agent orchestration via `pydantic_ai.Agent` handles async properly, but standalone `call_llm()` (used for clustering) blocks.

### Default Model Name

`config.py:34` defaults to `claude-sonnet-4` which is not a valid model identifier. Override with actual model name (e.g., `claude-sonnet-4-20250514`).
