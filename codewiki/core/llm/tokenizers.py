"""Token counting functionality for various LLM providers."""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

try:
    import tiktoken
except ImportError:
    tiktoken = None  # type: ignore

try:
    import anthropic
except ImportError:
    anthropic = None  # type: ignore

try:
    import google.generativeai as genai
except ImportError:
    genai = None  # type: ignore

from codewiki.core.errors import ContextLengthError

if TYPE_CHECKING:
    from codewiki.core.config import Settings


# Default context window limits per provider (in tokens).
# Used when a specific model is not found in MODEL_CONTEXT_LIMITS.
# These are conservative defaults to avoid context overflow.
PROVIDER_DEFAULT_LIMITS: dict[str, int] = {
    "anthropic": 200_000,  # Claude models typically support 200K
    "openai": 128_000,  # GPT-4o and newer support 128K
    "google": 1_000_000,  # Gemini models support up to 1M
    "groq": 32_000,  # Conservative default for Groq-hosted models
    "cerebras": 32_000,  # Conservative default for Cerebras-hosted models
    "unknown": 8_000,  # Very conservative default for unknown providers
}

# Model-specific context window limits (in tokens).
# This map takes precedence over provider defaults. Models not listed here
# will fall back to their provider's default limit from PROVIDER_DEFAULT_LIMITS.
#
# To add custom limits, use Settings.custom_context_limits which will be merged
# with this map at runtime, allowing per-deployment configuration.
#
# Last updated: December 2025
MODEL_CONTEXT_LIMITS: dict[str, int] = {
    # Anthropic Claude models
    "claude-opus-4-5-20251101": 200_000,
    "claude-sonnet-4-5-20251101": 200_000,
    "claude-sonnet-4-20250514": 200_000,
    "claude-opus-4-20250514": 200_000,
    "claude-3-5-sonnet-20241022": 200_000,
    "claude-3-5-haiku-20241022": 200_000,
    "claude-3-opus-20240229": 200_000,
    "claude-3-sonnet-20240229": 200_000,
    "claude-3-haiku-20240307": 200_000,
    # OpenAI GPT models
    "gpt-4o": 128_000,
    "gpt-4o-2024-11-20": 128_000,
    "gpt-4o-2024-08-06": 128_000,
    "gpt-4o-2024-05-13": 128_000,
    "gpt-4o-mini": 128_000,
    "gpt-4o-mini-2024-07-18": 128_000,
    "gpt-4.1": 1_000_000,
    "gpt-4.1-mini": 1_000_000,
    "gpt-4.1-nano": 1_000_000,
    "gpt-4-turbo": 128_000,
    "gpt-4-turbo-2024-04-09": 128_000,
    "gpt-4-turbo-preview": 128_000,
    "gpt-4-1106-preview": 128_000,
    "gpt-4": 8_192,
    "gpt-4-32k": 32_768,
    "gpt-3.5-turbo": 16_385,
    # OpenAI o1/o3 reasoning models
    "o1": 200_000,
    "o1-2024-12-17": 200_000,
    "o1-preview": 128_000,
    "o1-preview-2024-09-12": 128_000,
    "o1-mini": 128_000,
    "o1-mini-2024-09-12": 128_000,
    "o3": 200_000,
    "o3-mini": 200_000,
    "o3-mini-2025-01-31": 200_000,
    # Google Gemini models
    "gemini-2.5-pro": 1_000_000,
    "gemini-2.5-flash": 1_000_000,
    "gemini-2.5-flash-lite": 1_000_000,
    "gemini-2.0-flash": 1_000_000,
    "gemini-2.0-flash-lite": 1_000_000,
    "gemini-2.0-flash-exp": 1_000_000,
    "gemini-1.5-pro": 1_000_000,
    "gemini-1.5-pro-latest": 1_000_000,
    "gemini-1.5-flash": 1_000_000,
    "gemini-1.5-flash-latest": 1_000_000,
    "gemini-pro": 32_000,
    # Groq-hosted models
    "groq/llama-3.3-70b-versatile": 128_000,
    "groq/llama-3.1-70b-versatile": 128_000,
    "groq/llama-3.1-8b-instant": 128_000,
    "groq/llama3-70b-8192": 8_192,
    "groq/llama3-8b-8192": 8_192,
    "groq/mixtral-8x7b-32768": 32_768,
    "groq/gemma2-9b-it": 8_192,
    # Cerebras-hosted models
    "cerebras/llama3.1-70b": 128_000,
    "cerebras/llama3.1-8b": 128_000,
    "cerebras/llama3.1-405b": 128_000,
}


class TokenCounter:
    """Multi-provider token counter with accurate model-specific counting."""

    def __init__(self, settings: Optional[Settings] = None):
        """
        Initialize TokenCounter with lazy-loaded clients.

        Args:
            settings: Application settings with API keys. If None, creates a new Settings instance.
        """
        if settings is None:
            from codewiki.core.config import Settings

            settings = Settings()
        self.settings = settings
        self._tiktoken_enc: Optional[tiktoken.Encoding] = None
        self._anthropic_client: Optional[anthropic.Anthropic] = None
        self._google_models: dict[str, object] = {}  # Cache GenerativeModel per model name
        self._google_configured: bool = False

    def count(self, text: str, model: str) -> int:
        """
        Count tokens for given text and model.

        Args:
            text: Text to count tokens for
            model: Model name (e.g., 'gpt-4o', 'claude-3-5-sonnet-20241022')

        Returns:
            Number of tokens
        """
        provider = self._detect_provider(model)

        if provider == "anthropic":
            return self._count_anthropic(text, model)
        elif provider == "google":
            return self._count_google(text, model)
        elif provider in ("openai", "groq", "cerebras"):
            return self._count_tiktoken(text)
        else:
            # Unknown provider - use fallback
            return self._count_fallback(text)

    def _detect_provider(self, model: str) -> str:
        """
        Detect provider from model name.

        Args:
            model: Model name

        Returns:
            Provider name: 'anthropic', 'openai', 'google', 'groq', 'cerebras', or 'unknown'
        """
        model_lower = model.lower()

        if model_lower.startswith("claude-"):
            return "anthropic"
        elif model_lower.startswith(("gpt-", "o1-", "o3-")):
            return "openai"
        elif model_lower.startswith("gemini-"):
            return "google"
        elif model_lower.startswith("groq/"):
            return "groq"
        elif model_lower.startswith("cerebras/"):
            return "cerebras"
        else:
            return "unknown"

    def _count_anthropic(self, text: str, model: str) -> int:
        """
        Count tokens using Anthropic's messages.count_tokens API.

        Uses the same authentication context as LLMClient._call_anthropic() to ensure
        consistent credentials between context limit checks and completion calls.

        Args:
            text: Text to count
            model: Model name for accurate tokenization

        Returns:
            Token count
        """
        if anthropic is None:
            raise ImportError("anthropic package required for Claude models")

        if self._anthropic_client is None:
            api_key = self.settings.get_api_key("anthropic")
            self._anthropic_client = anthropic.Anthropic(api_key=api_key)

        # Use the official messages.count_tokens API (same format as messages.create)
        # Wrap text in a user message for token counting
        response = self._anthropic_client.messages.count_tokens(
            model=model,
            messages=[{"role": "user", "content": text}],
        )
        return response.input_tokens

    def _count_google(self, text: str, model: str) -> int:
        """
        Count tokens using Google's Gemini API.

        Uses the same configuration pattern as LLMClient._call_google() to ensure
        consistent credentials between context limit checks and completion calls.

        Args:
            text: Text to count
            model: Model name

        Returns:
            Token count
        """
        if genai is None:
            raise ImportError("google-generativeai package required for Gemini models")

        # Configure API key globally (same pattern as LLMClient._call_google)
        if not self._google_configured:
            api_key = self.settings.get_api_key("google")
            genai.configure(api_key=api_key)
            self._google_configured = True

        # Get or create GenerativeModel for this model (same pattern as LLMClient)
        if model not in self._google_models:
            self._google_models[model] = genai.GenerativeModel(model_name=model)

        google_model = self._google_models[model]
        response = google_model.count_tokens(text)
        return response.total_tokens

    def _count_tiktoken(self, text: str) -> int:
        """
        Count tokens using tiktoken (for OpenAI, Groq, Cerebras).

        Args:
            text: Text to count

        Returns:
            Token count
        """
        if tiktoken is None:
            raise ImportError("tiktoken package required for token counting")

        if self._tiktoken_enc is None:
            self._tiktoken_enc = tiktoken.get_encoding("cl100k_base")

        return len(self._tiktoken_enc.encode(text))

    def _count_fallback(self, text: str) -> int:
        """
        Fallback token counting with 10% buffer.

        Args:
            text: Text to count

        Returns:
            Token count with 10% buffer
        """
        if tiktoken is None:
            raise ImportError("tiktoken package required for token counting")

        if self._tiktoken_enc is None:
            self._tiktoken_enc = tiktoken.get_encoding("cl100k_base")

        base_count = len(self._tiktoken_enc.encode(text))
        return int(base_count * 1.1)

    def get_context_limit(self, model: str) -> int:
        """
        Get the context window limit for a model.

        Resolution order:
        1. Custom limits from Settings.custom_context_limits (highest priority)
        2. Built-in MODEL_CONTEXT_LIMITS for known models
        3. Provider-specific default from PROVIDER_DEFAULT_LIMITS
        4. Very conservative fallback for unknown providers (8,000 tokens)

        Args:
            model: Model name

        Returns:
            Context window limit in tokens

        Note:
            Unknown models will fall back to their provider's default limit,
            or 8,000 tokens if the provider is also unknown. This conservative
            default helps prevent context overflow errors at the cost of
            potentially underutilizing some models' full capacity. Add custom
            limits via Settings.custom_context_limits for unlisted models.
        """
        # Priority 1: Custom limits from settings
        if model in self.settings.custom_context_limits:
            return self.settings.custom_context_limits[model]

        # Priority 2: Built-in model-specific limits
        if model in MODEL_CONTEXT_LIMITS:
            return MODEL_CONTEXT_LIMITS[model]

        # Priority 3: Provider-specific default
        provider = self._detect_provider(model)
        return PROVIDER_DEFAULT_LIMITS.get(provider, PROVIDER_DEFAULT_LIMITS["unknown"])

    def check_context_limit(self, text: str, model: str, threshold: float = 0.95) -> None:
        """
        Check if text exceeds model's context limit.

        Uses get_context_limit() to determine the appropriate limit, which checks
        custom limits, model-specific limits, and provider defaults in order.

        Args:
            text: Text to check
            model: Model name
            threshold: Percentage of limit to check against (default: 0.95)

        Raises:
            ContextLengthError: If token count exceeds threshold * context_limit

        Note:
            Unknown models fall back to provider-specific defaults or a very
            conservative 8,000 token limit. See get_context_limit() for details.
        """
        token_count = self.count(text, model)
        context_limit = self.get_context_limit(model)
        max_tokens = int(context_limit * threshold)

        if token_count > max_tokens:
            provider = self._detect_provider(model)
            raise ContextLengthError(
                provider=provider,
                model=model,
                message=f"Token count {token_count} exceeds limit {context_limit} "
                f"(threshold: {threshold:.0%})",
                max_tokens=context_limit,
                actual_tokens=token_count,
            )
