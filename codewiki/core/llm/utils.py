"""Utility functions for LLM operations."""

from typing import Optional


def detect_provider(model: str, base_url: Optional[str] = None) -> str:
    """
    Detect LLM provider from model name and optional base URL.

    This is the centralized provider detection function used throughout CodeWiki.
    It determines which API provider to use based on model naming conventions
    and base URL patterns.

    Args:
        model: Model name (e.g., 'claude-sonnet-4-20250514', 'gpt-4o', 'groq/llama-3')
        base_url: Optional API base URL for provider detection via URL patterns

    Returns:
        Provider name: 'anthropic', 'openai', 'google', 'groq', 'cerebras', or 'unknown'

    Examples:
        >>> detect_provider("claude-sonnet-4-20250514")
        'anthropic'
        >>> detect_provider("gpt-4o")
        'openai'
        >>> detect_provider("gemini-2.5-pro")
        'google'
        >>> detect_provider("groq/llama-3-70b")
        'groq'
        >>> detect_provider("llama-3", base_url="https://api.groq.com/openai/v1")
        'groq'
    """
    model_lower = model.lower()

    # Check model name prefixes first (most common case)
    if model_lower.startswith("claude"):
        return "anthropic"
    elif model_lower.startswith(("gpt", "o1", "o3")):
        return "openai"
    elif model_lower.startswith("gemini"):
        return "google"
    elif model_lower.startswith("groq/"):
        return "groq"
    elif model_lower.startswith("cerebras/"):
        return "cerebras"

    # Check base_url patterns if provided (for custom deployments)
    if base_url:
        base_url_lower = base_url.lower()
        if "groq" in base_url_lower:
            return "groq"
        elif "cerebras" in base_url_lower:
            return "cerebras"
        elif "anthropic" in base_url_lower:
            return "anthropic"
        elif "openai" in base_url_lower:
            return "openai"
        elif "google" in base_url_lower or "generativelanguage" in base_url_lower:
            return "google"

    # Unknown provider
    return "unknown"
