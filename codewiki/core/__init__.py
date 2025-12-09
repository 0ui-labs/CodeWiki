"""Core configuration, logging, error handling, and async utilities for CodeWiki.

This module provides:
- Settings: Central configuration using Pydantic Settings
- CodeWikiLogger: Unified logging with Rich console and JSON file output
- get_logger: Factory function for creating loggers
- ParallelModuleProcessor: Async module processor with dependency management
- Exception hierarchy for LLM-specific error handling
"""

from codewiki.core.async_utils import ParallelModuleProcessor
from codewiki.core.config import Settings
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
]
