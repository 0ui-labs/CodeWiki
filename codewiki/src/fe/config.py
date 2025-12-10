#!/usr/bin/env python3
"""
Configuration settings for the CodeWiki web application.
"""

import os
import warnings
from pathlib import Path
from typing import Optional

from codewiki.core import Settings
from codewiki.core.llm.utils import detect_provider


# Legacy environment variable mappings for backward compatibility
_LEGACY_ENV_MAPPINGS = {
    # Old name -> (new name, provider)
    # Note: LLM_API_KEY is handled specially via _detect_provider_from_model()
    "OPENAI_API_KEY": ("CODEWIKI_OPENAI_API_KEY", "openai"),
    "GOOGLE_API_KEY": ("CODEWIKI_GOOGLE_API_KEY", "google"),
    "GROQ_API_KEY": ("CODEWIKI_GROQ_API_KEY", "groq"),
    "CEREBRAS_API_KEY": ("CODEWIKI_CEREBRAS_API_KEY", "cerebras"),
    "MAIN_MODEL": ("CODEWIKI_MAIN_MODEL", None),
    "CLUSTER_MODEL": ("CODEWIKI_CLUSTER_MODEL", None),
    "LOG_LEVEL": ("CODEWIKI_LOG_LEVEL", None),
}

# Default model used when none is specified
_DEFAULT_MAIN_MODEL = "claude-sonnet-4-20250514"


def _detect_provider_from_model(model: str) -> str:
    """
    Detect the LLM provider from the model name.

    Uses the centralized detect_provider function from codewiki.core.llm.utils.
    Falls back to 'anthropic' for unknown models (backward compatibility
    since the default model is claude-sonnet-4-20250514).

    Args:
        model: The model name (e.g., 'claude-sonnet-4-20250514', 'gpt-4o')

    Returns:
        Provider name: 'anthropic', 'openai', 'google', 'groq', or 'cerebras'
    """
    provider = detect_provider(model)
    # Default to anthropic for unknown models (backward compatibility)
    return provider if provider != "unknown" else "anthropic"


def _get_env_with_fallback(
    new_name: str, legacy_name: Optional[str] = None, default: Optional[str] = None
) -> Optional[str]:
    """
    Get environment variable with fallback to legacy name.

    Args:
        new_name: New CODEWIKI_* prefixed environment variable name
        legacy_name: Optional legacy environment variable name
        default: Default value if neither is set

    Returns:
        Environment variable value or default
    """
    # Try new name first
    value = os.getenv(new_name)
    if value:
        return value

    # Fall back to legacy name
    if legacy_name:
        legacy_value = os.getenv(legacy_name)
        if legacy_value:
            warnings.warn(
                f"Environment variable '{legacy_name}' is deprecated. "
                f"Please use '{new_name}' instead. "
                f"Support for '{legacy_name}' will be removed in a future version.",
                DeprecationWarning,
                stacklevel=3,
            )
            return legacy_value

    return default


def get_web_app_settings() -> Settings:
    """
    Load web app settings from environment variables with legacy fallback.

    Environment variables should use CODEWIKI_ prefix:
    - CODEWIKI_ANTHROPIC_API_KEY
    - CODEWIKI_OPENAI_API_KEY
    - CODEWIKI_GOOGLE_API_KEY
    - CODEWIKI_GROQ_API_KEY
    - CODEWIKI_CEREBRAS_API_KEY
    - CODEWIKI_MAIN_MODEL (default: claude-sonnet-4-20250514)
    - CODEWIKI_OUTPUT_DIR (default: ./output/docs)
    - CODEWIKI_LOG_LEVEL (default: INFO)
    - CODEWIKI_LOG_FILE (optional)

    Legacy environment variables are still supported with deprecation warnings:
    - LLM_API_KEY -> Mapped to the correct provider based on MAIN_MODEL
    - OPENAI_API_KEY -> CODEWIKI_OPENAI_API_KEY
    - GOOGLE_API_KEY -> CODEWIKI_GOOGLE_API_KEY
    - GROQ_API_KEY -> CODEWIKI_GROQ_API_KEY
    - CEREBRAS_API_KEY -> CODEWIKI_CEREBRAS_API_KEY
    - MAIN_MODEL -> CODEWIKI_MAIN_MODEL
    - CLUSTER_MODEL -> CODEWIKI_CLUSTER_MODEL

    LLM_API_KEY Interpretation:
        The legacy LLM_API_KEY variable is mapped to the appropriate provider
        based on the configured MAIN_MODEL (or CODEWIKI_MAIN_MODEL):
        - claude-* models -> anthropic_api_key
        - gpt-*, o1-*, o3-* models -> openai_api_key
        - gemini-* models -> google_api_key
        - groq/* models -> groq_api_key
        - cerebras/* models -> cerebras_api_key

        If no MAIN_MODEL is set, LLM_API_KEY defaults to anthropic_api_key
        (since the default model is claude-sonnet-4-20250514).

    Returns:
        Settings instance configured for web app
    """
    # Build settings dict with fallback support
    settings_kwargs = {}

    # Get model configuration first (needed for LLM_API_KEY provider detection)
    main_model = _get_env_with_fallback("CODEWIKI_MAIN_MODEL", "MAIN_MODEL")
    if main_model:
        settings_kwargs["main_model"] = main_model

    # Determine the effective model for provider detection
    effective_model = main_model or _DEFAULT_MAIN_MODEL

    # Provider-specific API keys (new CODEWIKI_* format)
    anthropic_key = os.getenv("CODEWIKI_ANTHROPIC_API_KEY")
    if anthropic_key:
        settings_kwargs["anthropic_api_key"] = anthropic_key

    openai_key = _get_env_with_fallback("CODEWIKI_OPENAI_API_KEY", "OPENAI_API_KEY")
    if openai_key:
        settings_kwargs["openai_api_key"] = openai_key

    google_key = _get_env_with_fallback("CODEWIKI_GOOGLE_API_KEY", "GOOGLE_API_KEY")
    if google_key:
        settings_kwargs["google_api_key"] = google_key

    groq_key = _get_env_with_fallback("CODEWIKI_GROQ_API_KEY", "GROQ_API_KEY")
    if groq_key:
        settings_kwargs["groq_api_key"] = groq_key

    cerebras_key = _get_env_with_fallback("CODEWIKI_CEREBRAS_API_KEY", "CEREBRAS_API_KEY")
    if cerebras_key:
        settings_kwargs["cerebras_api_key"] = cerebras_key

    # Handle legacy LLM_API_KEY with provider detection
    # Only use LLM_API_KEY if no provider-specific key is set for the detected provider
    legacy_api_key = os.getenv("LLM_API_KEY")
    if legacy_api_key:
        provider = _detect_provider_from_model(effective_model)
        provider_key_field = f"{provider}_api_key"

        # Only apply LLM_API_KEY if the provider-specific key isn't already set
        if provider_key_field not in settings_kwargs:
            warnings.warn(
                f"Environment variable 'LLM_API_KEY' is deprecated. "
                f"Based on model '{effective_model}', it is being mapped to "
                f"'{provider_key_field}'. Please use 'CODEWIKI_{provider.upper()}_API_KEY' "
                f"instead. Support for 'LLM_API_KEY' will be removed in a future version.",
                DeprecationWarning,
                stacklevel=2,
            )
            settings_kwargs[provider_key_field] = legacy_api_key

    cluster_model = _get_env_with_fallback("CODEWIKI_CLUSTER_MODEL", "CLUSTER_MODEL")
    if cluster_model:
        settings_kwargs["cluster_model"] = cluster_model

    # Output directory
    output_dir = os.getenv("CODEWIKI_OUTPUT_DIR", "./output/docs")
    settings_kwargs["output_dir"] = output_dir

    # Logging
    log_level = _get_env_with_fallback("CODEWIKI_LOG_LEVEL", "LOG_LEVEL", "INFO")
    settings_kwargs["log_level"] = log_level

    log_file = os.getenv("CODEWIKI_LOG_FILE")
    if log_file:
        settings_kwargs["log_file"] = log_file

    return Settings(**settings_kwargs)


# Legacy constants for backward compatibility (deprecated)
class WebAppConfig:
    """DEPRECATED: Use get_web_app_settings() instead."""

    CACHE_DIR = "./output/cache"
    TEMP_DIR = "./output/temp"
    OUTPUT_DIR = "./output"
    QUEUE_SIZE = 100
    CACHE_EXPIRY_DAYS = 365
    JOB_CLEANUP_HOURS = 24000
    RETRY_COOLDOWN_MINUTES = 3
    DEFAULT_HOST = "127.0.0.1"
    DEFAULT_PORT = 8000
    CLONE_TIMEOUT = 300
    CLONE_DEPTH = 1

    @classmethod
    def ensure_directories(cls):
        """Ensure all required directories exist."""
        for directory in [cls.CACHE_DIR, cls.TEMP_DIR, cls.OUTPUT_DIR]:
            Path(directory).mkdir(parents=True, exist_ok=True)

    @classmethod
    def get_absolute_path(cls, path: str) -> str:
        """Get absolute path for a given relative path."""
        return os.path.abspath(path)
