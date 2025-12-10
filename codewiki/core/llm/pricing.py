"""Legacy LLM pricing module - thin wrapper around costs.py.

This module provides backwards-compatible exports for existing code that imports
from ``codewiki.core.llm.pricing``. New code should import from
``codewiki.core.llm.costs`` directly.

All pricing data is defined in ``costs.py`` as the single source of truth.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Dict

from .costs import PRICING, calculate_cost as _calculate_cost_new

if TYPE_CHECKING:
    from codewiki.core.logging import CodeWikiLogger

# Module-level logger for fallback when no CodeWikiLogger is available
_logger = logging.getLogger(__name__)

# Default pricing for unknown models (per 1M tokens)
# Used for backwards compatibility with code expecting these constants
DEFAULT_PRICING: Dict[str, float] = {
    "input": 5.00,
    "output": 15.00,
}

# Backwards-compatible MODEL_PRICING dict built from costs.py PRICING
# This converts the typed ModelPricing dataclass to the legacy dict format
MODEL_PRICING: Dict[str, Dict[str, float]] = {
    model: {"input": pricing.input_per_million, "output": pricing.output_per_million}
    for model, pricing in PRICING.items()
}


def get_model_pricing(model: str, logger: "CodeWikiLogger | None" = None) -> Dict[str, float]:
    """Get pricing information for a specific model.

    Args:
        model: Model identifier (e.g., 'gpt-4o', 'claude-sonnet-4-20250514')
        logger: Optional CodeWikiLogger for warning about unknown models

    Returns:
        Dictionary with 'input' and 'output' pricing per 1M tokens.
        Returns default pricing for unknown models.
    """
    pricing = PRICING.get(model)

    if pricing is None:
        # Log warning about unknown model
        warning_msg = (
            f"Unknown model '{model}' - using default pricing "
            f"(${DEFAULT_PRICING['input']:.2f}/${DEFAULT_PRICING['output']:.2f} per 1M tokens)"
        )
        if logger is not None:
            logger.warning(warning_msg, model=model)
        else:
            _logger.warning(warning_msg)

        return DEFAULT_PRICING.copy()

    return {"input": pricing.input_per_million, "output": pricing.output_per_million}


def calculate_cost(
    model: str,
    input_tokens: int,
    output_tokens: int,
    logger: "CodeWikiLogger | None" = None,
) -> float:
    """Calculate the cost of an LLM API call.

    Args:
        model: Model identifier (e.g., 'gpt-4o', 'claude-sonnet-4-20250514')
        input_tokens: Number of input/prompt tokens
        output_tokens: Number of output/completion tokens
        logger: Optional logger for warnings about unknown models

    Returns:
        Total cost in USD. Returns 0.0 for unknown models.

    Example:
        >>> calculate_cost("gpt-4o-mini", 1_000_000, 500_000)
        0.45
        >>> calculate_cost("claude-sonnet-4-20250514", 1000, 500)
        0.0105
    """
    pricing = PRICING.get(model)

    if pricing is None:
        # Log warning about unknown model - matches costs.py behavior
        if logger is not None:
            logger.warning(
                f"Unknown model '{model}' - cannot calculate cost",
                model=model,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
            )
        else:
            _logger.warning(
                "Unknown model '%s' - cannot calculate cost (input_tokens=%d, output_tokens=%d)",
                model,
                input_tokens,
                output_tokens,
            )
        return 0.0

    # Delegate to costs.py for known models
    return _calculate_cost_new(model, input_tokens, output_tokens, logger)
