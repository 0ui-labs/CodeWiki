"""LLM cost calculation module with typed pricing data.

This module provides a structured approach to LLM pricing using dataclasses,
enabling type-safe cost calculations for various model providers.

All prices are per 1 million tokens (as of December 2025).

Usage:
    from codewiki.core.llm.costs import calculate_cost, PRICING, ModelPricing

    # Calculate cost for a specific model
    cost = calculate_cost("gpt-4o", input_tokens=1000, output_tokens=500)

    # Access pricing directly
    pricing = PRICING.get("claude-sonnet-4-20250514")
    if pricing:
        print(f"Input: ${pricing.input_per_million}/M tokens")
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from codewiki.core.logging import CodeWikiLogger
    from codewiki.core.llm.pricing_provider import PricingProvider


@dataclass(frozen=True, slots=True)
class ModelPricing:
    """Pricing information for an LLM model.

    Attributes:
        input_per_million: Cost in USD per 1 million input/prompt tokens.
        output_per_million: Cost in USD per 1 million output/completion tokens.
    """

    input_per_million: float
    output_per_million: float


# Model pricing per 1 million tokens
# Prices are current as of December 2025
PRICING: dict[str, ModelPricing] = {
    # Anthropic Claude models
    "claude-opus-4-5-20251101": ModelPricing(input_per_million=5.00, output_per_million=25.00),
    "claude-sonnet-4-5-20251101": ModelPricing(input_per_million=3.00, output_per_million=15.00),
    "claude-sonnet-4-20250514": ModelPricing(input_per_million=3.00, output_per_million=15.00),
    "claude-opus-4-20250514": ModelPricing(input_per_million=15.00, output_per_million=75.00),
    "claude-3-5-sonnet-20241022": ModelPricing(input_per_million=3.00, output_per_million=15.00),
    "claude-3-5-haiku-20241022": ModelPricing(input_per_million=0.80, output_per_million=4.00),
    "claude-3-opus-20240229": ModelPricing(input_per_million=15.00, output_per_million=75.00),
    "claude-3-sonnet-20240229": ModelPricing(input_per_million=3.00, output_per_million=15.00),
    "claude-3-haiku-20240307": ModelPricing(input_per_million=0.25, output_per_million=1.25),
    # OpenAI GPT models
    "gpt-4o": ModelPricing(input_per_million=5.00, output_per_million=20.00),
    "gpt-4o-mini": ModelPricing(input_per_million=0.15, output_per_million=0.60),
    "gpt-4.1": ModelPricing(input_per_million=2.00, output_per_million=8.00),
    "gpt-4.1-mini": ModelPricing(input_per_million=0.40, output_per_million=1.60),
    "gpt-4.1-nano": ModelPricing(input_per_million=0.10, output_per_million=0.40),
    "gpt-4-turbo": ModelPricing(input_per_million=10.00, output_per_million=30.00),
    "gpt-4": ModelPricing(input_per_million=30.00, output_per_million=60.00),
    "gpt-3.5-turbo": ModelPricing(input_per_million=0.50, output_per_million=1.50),
    # OpenAI reasoning models
    "o1": ModelPricing(input_per_million=15.00, output_per_million=60.00),
    "o1-mini": ModelPricing(input_per_million=3.00, output_per_million=12.00),
    "o1-preview": ModelPricing(input_per_million=15.00, output_per_million=60.00),
    "o3": ModelPricing(input_per_million=1.00, output_per_million=4.00),
    "o3-mini": ModelPricing(input_per_million=1.10, output_per_million=4.40),
    # Google Gemini models
    "gemini-2.5-pro": ModelPricing(input_per_million=1.25, output_per_million=10.00),
    "gemini-2.5-flash": ModelPricing(input_per_million=0.30, output_per_million=2.50),
    "gemini-2.5-flash-lite": ModelPricing(input_per_million=0.10, output_per_million=0.40),
    "gemini-2.0-flash": ModelPricing(input_per_million=0.10, output_per_million=0.40),
    "gemini-2.0-flash-lite": ModelPricing(input_per_million=0.02, output_per_million=0.10),
    "gemini-1.5-pro": ModelPricing(input_per_million=1.25, output_per_million=5.00),
    "gemini-1.5-flash": ModelPricing(input_per_million=0.075, output_per_million=0.30),
    "gemini-pro": ModelPricing(input_per_million=0.50, output_per_million=1.50),
    # Groq models
    "groq/llama-3.3-70b-versatile": ModelPricing(input_per_million=0.59, output_per_million=0.79),
    "groq/llama-3.1-70b-versatile": ModelPricing(input_per_million=0.59, output_per_million=0.79),
    "groq/llama-3.1-8b-instant": ModelPricing(input_per_million=0.05, output_per_million=0.08),
    "groq/llama3-70b-8192": ModelPricing(input_per_million=0.59, output_per_million=0.79),
    "groq/llama3-8b-8192": ModelPricing(input_per_million=0.05, output_per_million=0.08),
    "groq/mixtral-8x7b-32768": ModelPricing(input_per_million=0.24, output_per_million=0.24),
    "groq/gemma2-9b-it": ModelPricing(input_per_million=0.20, output_per_million=0.20),
    # Cerebras models
    "cerebras/llama3.1-8b": ModelPricing(input_per_million=0.10, output_per_million=0.10),
    "cerebras/llama3.1-70b": ModelPricing(input_per_million=0.60, output_per_million=0.60),
    "cerebras/llama3.1-405b": ModelPricing(input_per_million=6.00, output_per_million=12.00),
}


def calculate_cost(
    model: str,
    input_tokens: int,
    output_tokens: int,
    logger: CodeWikiLogger | None = None,
    pricing_provider: PricingProvider | None = None,
) -> float:
    """Calculate the cost of an LLM API call.

    Args:
        model: Model identifier (e.g., 'gpt-4o', 'claude-sonnet-4-20250514')
        input_tokens: Number of input/prompt tokens
        output_tokens: Number of output/completion tokens
        logger: Optional logger for warnings about unknown models
        pricing_provider: Optional PricingProvider for fetched pricing data.
            If provided, checks provider first, then falls back to static PRICING.

    Returns:
        Total cost in USD. Returns 0.0 for unknown models.

    Example:
        >>> calculate_cost("gpt-4o-mini", 1_000_000, 500_000)
        0.45
        >>> calculate_cost("claude-sonnet-4-20250514", 1000, 500)
        0.0105
    """
    # Try pricing provider first, then fall back to static PRICING
    pricing = None
    if pricing_provider is not None:
        pricing = pricing_provider.get_pricing(model)
    if pricing is None:
        pricing = PRICING.get(model)

    if pricing is None:
        if logger is not None:
            logger.warning(
                f"Unknown model '{model}' - cannot calculate cost",
                model=model,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
            )
        return 0.0

    # Calculate cost based on pricing per 1M tokens
    input_cost = (input_tokens / 1_000_000) * pricing.input_per_million
    output_cost = (output_tokens / 1_000_000) * pricing.output_per_million

    return input_cost + output_cost
