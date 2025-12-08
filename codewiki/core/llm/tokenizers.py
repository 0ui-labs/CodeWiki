"""Token counting functionality for various LLM providers."""

from typing import Optional

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


# Model context window limits (in tokens)
MODEL_CONTEXT_LIMITS = {
    "claude-sonnet-4-20250514": 200_000,
    "claude-3-5-sonnet-20241022": 200_000,
    "gpt-4o": 128_000,
    "gpt-4o-mini": 128_000,
    "gemini-1.5-pro": 1_000_000,
    "gemini-2.0-flash-exp": 1_000_000,
    "groq/llama-3.3-70b-versatile": 128_000,
    "cerebras/llama3.1-70b": 128_000,
}


class TokenCounter:
    """Multi-provider token counter with accurate model-specific counting."""

    def __init__(self):
        """Initialize TokenCounter with lazy-loaded clients."""
        self._tiktoken_enc: Optional[tiktoken.Encoding] = None
        self._anthropic_client: Optional[anthropic.Anthropic] = None
        self._google_client: Optional[genai.Client] = None

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
        Count tokens using Anthropic's API.

        Args:
            text: Text to count
            model: Model name (unused, Anthropic uses same tokenizer for all models)

        Returns:
            Token count
        """
        if anthropic is None:
            raise ImportError("anthropic package required for Claude models")

        if self._anthropic_client is None:
            self._anthropic_client = anthropic.Anthropic()

        # Anthropic's count_tokens is a method on the client that takes the text directly
        return self._anthropic_client.count_tokens(text)

    def _count_google(self, text: str, model: str) -> int:
        """
        Count tokens using Google's Gemini API.

        Args:
            text: Text to count
            model: Model name

        Returns:
            Token count
        """
        if genai is None:
            raise ImportError("google-generativeai package required for Gemini models")

        if self._google_client is None:
            self._google_client = genai.Client()

        response = self._google_client.models.count_tokens(model=model, contents=text)
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

    def check_context_limit(self, text: str, model: str, threshold: float = 0.95) -> None:
        """
        Check if text exceeds model's context limit.

        Args:
            text: Text to check
            model: Model name
            threshold: Percentage of limit to check against (default: 0.95)

        Raises:
            ContextLengthError: If token count exceeds threshold * context_limit
        """
        token_count = self.count(text, model)
        context_limit = MODEL_CONTEXT_LIMITS.get(model, 128_000)  # Default fallback
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
