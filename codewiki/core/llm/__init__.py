"""LLM client and tokenization utilities for multi-provider support."""

from .client import LLMClient, LLMResponse, MODEL_ALIASES, ToolCall
from .tokenizers import TokenCounter, MODEL_CONTEXT_LIMITS
from .retry import ResilientLLMClient, RetryConfig, LoggerProtocol
from .utils import detect_provider

# Cost calculation (costs.py is the source of truth for pricing data)
from .costs import ModelPricing, PRICING, calculate_cost
from .pricing_provider import PricingProvider

# Backwards-compatible exports (pricing.py wraps costs.py)
from .pricing import MODEL_PRICING, DEFAULT_PRICING, get_model_pricing

__all__ = [
    # Client
    "LLMClient",
    "LLMResponse",
    "ToolCall",
    "MODEL_ALIASES",
    # Tokenization
    "TokenCounter",
    "MODEL_CONTEXT_LIMITS",
    # Retry/resilience
    "ResilientLLMClient",
    "RetryConfig",
    "LoggerProtocol",
    # Cost calculation (preferred API)
    "ModelPricing",
    "PRICING",
    "PricingProvider",
    "calculate_cost",
    # Legacy pricing exports (backwards compatibility)
    "MODEL_PRICING",
    "DEFAULT_PRICING",
    "get_model_pricing",
    # Utilities
    "detect_provider",
]
