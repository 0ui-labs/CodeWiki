"""Core configuration, logging, and error handling for CodeWiki.

This module provides:
- Settings: Central configuration using Pydantic Settings
- CodeWikiLogger: Unified logging with Rich console and JSON file output
- get_logger: Factory function for creating loggers
- Exception hierarchy for LLM-specific error handling
"""

from codewiki.core.config import Settings
from codewiki.core.errors import (
    AuthenticationError,
    CodeWikiError,
    ContextLengthError,
    LLMError,
    ProviderUnavailableError,
    RateLimitError,
)
from codewiki.core.logging import CodeWikiLogger, get_logger

__all__ = [
    "Settings",
    "CodeWikiLogger",
    "get_logger",
    "CodeWikiError",
    "LLMError",
    "RateLimitError",
    "ContextLengthError",
    "ProviderUnavailableError",
    "AuthenticationError",
]
