"""Core configuration and error handling for CodeWiki.

This module provides:
- Settings: Central configuration using Pydantic Settings
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

__all__ = [
    "Settings",
    "CodeWikiError",
    "LLMError",
    "RateLimitError",
    "ContextLengthError",
    "ProviderUnavailableError",
    "AuthenticationError",
]
