"""LLM pricing module for cost tracking.

This module provides pricing information for various LLM models and calculates
the cost of API calls based on token usage.

All prices are per 1 million tokens (as of December 2025).
"""

from typing import Dict

# Default pricing for unknown models (per 1M tokens)
DEFAULT_PRICING = {
    "input": 5.00,
    "output": 15.00,
}

# Model pricing per 1 million tokens
# Prices are current as of December 2025
MODEL_PRICING: Dict[str, Dict[str, float]] = {
    # Anthropic Claude models
    "claude-opus-4-5-20251101": {"input": 5.00, "output": 25.00},
    "claude-sonnet-4-5-20251101": {"input": 3.00, "output": 15.00},
    "claude-sonnet-4-20250514": {"input": 3.00, "output": 15.00},
    "claude-opus-4-20250514": {"input": 15.00, "output": 75.00},
    "claude-3-5-sonnet-20241022": {"input": 3.00, "output": 15.00},
    "claude-3-5-haiku-20241022": {"input": 0.80, "output": 4.00},
    "claude-3-opus-20240229": {"input": 15.00, "output": 75.00},
    "claude-3-sonnet-20240229": {"input": 3.00, "output": 15.00},
    "claude-3-haiku-20240307": {"input": 0.25, "output": 1.25},
    # OpenAI GPT models
    "gpt-4o": {"input": 5.00, "output": 20.00},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-4.1": {"input": 2.00, "output": 8.00},
    "gpt-4.1-mini": {"input": 0.40, "output": 1.60},
    "gpt-4.1-nano": {"input": 0.10, "output": 0.40},
    "gpt-4-turbo": {"input": 10.00, "output": 30.00},
    "gpt-4": {"input": 30.00, "output": 60.00},
    "gpt-3.5-turbo": {"input": 0.50, "output": 1.50},
    # OpenAI reasoning models
    "o1": {"input": 15.00, "output": 60.00},
    "o1-mini": {"input": 3.00, "output": 12.00},
    "o1-preview": {"input": 15.00, "output": 60.00},
    "o3": {"input": 1.00, "output": 4.00},
    "o3-mini": {"input": 1.10, "output": 4.40},
    # Google Gemini models
    "gemini-2.5-pro": {"input": 1.25, "output": 10.00},
    "gemini-2.5-flash": {"input": 0.30, "output": 2.50},
    "gemini-2.5-flash-lite": {"input": 0.10, "output": 0.40},
    "gemini-2.0-flash": {"input": 0.10, "output": 0.40},
    "gemini-2.0-flash-lite": {"input": 0.02, "output": 0.10},
    "gemini-1.5-pro": {"input": 1.25, "output": 5.00},
    "gemini-1.5-flash": {"input": 0.075, "output": 0.30},
    "gemini-pro": {"input": 0.50, "output": 1.50},
    # Groq models
    "groq/llama-3.3-70b-versatile": {"input": 0.59, "output": 0.79},
    "groq/llama-3.1-70b-versatile": {"input": 0.59, "output": 0.79},
    "groq/llama-3.1-8b-instant": {"input": 0.05, "output": 0.08},
    "groq/llama3-70b-8192": {"input": 0.59, "output": 0.79},
    "groq/llama3-8b-8192": {"input": 0.05, "output": 0.08},
    "groq/mixtral-8x7b-32768": {"input": 0.24, "output": 0.24},
    "groq/gemma2-9b-it": {"input": 0.20, "output": 0.20},
    # Cerebras models
    "cerebras/llama3.1-8b": {"input": 0.10, "output": 0.10},
    "cerebras/llama3.1-70b": {"input": 0.60, "output": 0.60},
    "cerebras/llama3.1-405b": {"input": 6.00, "output": 12.00},
}


def get_model_pricing(model: str) -> Dict[str, float]:
    """
    Get pricing information for a specific model.

    Args:
        model: Model identifier (e.g., 'gpt-4o', 'claude-sonnet-4-20250514')

    Returns:
        Dictionary with 'input' and 'output' pricing per 1M tokens.
        Returns default pricing for unknown models.
    """
    pricing = MODEL_PRICING.get(model, DEFAULT_PRICING)
    # Return a copy to prevent accidental modification
    return pricing.copy()


def calculate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """
    Calculate the cost of an LLM API call.

    Args:
        model: Model identifier (e.g., 'gpt-4o', 'claude-sonnet-4-20250514')
        input_tokens: Number of input/prompt tokens
        output_tokens: Number of output/completion tokens

    Returns:
        Total cost in USD

    Raises:
        ValueError: If token counts are negative

    Example:
        >>> calculate_cost("gpt-4o-mini", 1_000_000, 500_000)
        0.45
        >>> calculate_cost("claude-sonnet-4-20250514", 1000, 500)
        0.0105
    """
    if input_tokens < 0 or output_tokens < 0:
        raise ValueError("Token counts cannot be negative")

    pricing = get_model_pricing(model)

    # Calculate cost based on pricing per 1M tokens
    input_cost = (input_tokens / 1_000_000) * pricing["input"]
    output_cost = (output_tokens / 1_000_000) * pricing["output"]

    return input_cost + output_cost
