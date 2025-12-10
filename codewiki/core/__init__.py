"""Core configuration, logging, error handling, and async utilities for CodeWiki.

This module provides:
- Settings: Central configuration using Pydantic Settings
- CodeWikiLogger: Unified logging with Rich console and JSON file output
- get_logger: Factory function for creating loggers
- ParallelModuleProcessor: Async module processor with dependency management
- Exception hierarchy for LLM-specific error handling
- Constants: Documentation generation constants
- ModuleHasher: Deterministic hashing for module content and hierarchies
- ModelPricing, calculate_cost: LLM cost calculation utilities
"""

from codewiki.core.async_utils import ParallelModuleProcessor
from codewiki.core.llm.costs import ModelPricing, calculate_cost
from codewiki.core.utils.hashing import ModuleHasher
from codewiki.core.config import Settings
from codewiki.core.constants import (
    DEPENDENCY_GRAPHS_DIR,
    DOCS_DIR,
    FIRST_MODULE_TREE_FILENAME,
    MAX_DEPTH,
    MAX_TOKEN_PER_LEAF_MODULE,
    MAX_TOKEN_PER_MODULE,
    MODULE_TREE_FILENAME,
    OUTPUT_BASE_DIR,
    OVERVIEW_FILENAME,
)
from codewiki.core.errors import (
    AuthenticationError,
    CodeWikiError,
    ContextLengthError,
    DependencyFailedError,
    LLMError,
    ProviderUnavailableError,
    RateLimitError,
)
from codewiki.core.logging import CodeWikiLogger, get_logger

__all__ = [
    "Settings",
    "CodeWikiLogger",
    "get_logger",
    "ParallelModuleProcessor",
    "CodeWikiError",
    "DependencyFailedError",
    "LLMError",
    "RateLimitError",
    "ContextLengthError",
    "ProviderUnavailableError",
    "AuthenticationError",
    "FIRST_MODULE_TREE_FILENAME",
    "MODULE_TREE_FILENAME",
    "OVERVIEW_FILENAME",
    "MAX_TOKEN_PER_MODULE",
    "MAX_TOKEN_PER_LEAF_MODULE",
    "OUTPUT_BASE_DIR",
    "DEPENDENCY_GRAPHS_DIR",
    "DOCS_DIR",
    "MAX_DEPTH",
    "ModuleHasher",
    "ModelPricing",
    "calculate_cost",
]
