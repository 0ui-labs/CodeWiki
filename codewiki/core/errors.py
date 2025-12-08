"""Exception hierarchy for CodeWiki Core."""


class CodeWikiError(Exception):
    """Base exception for all CodeWiki Core errors."""

    pass


class LLMError(CodeWikiError):
    """Base exception for all LLM-related errors."""

    def __init__(self, provider: str, model: str, message: str):
        """
        Initialize LLMError.

        Args:
            provider: The LLM provider name (e.g., "openai", "anthropic")
            model: The model identifier
            message: Error message
        """
        self.provider = provider
        self.model = model
        self.message = message
        super().__init__(message)

    def __str__(self) -> str:
        """Return a string representation including provider and model info."""
        return f"LLM Error [{self.provider}/{self.model}]: {self.message}"


class RateLimitError(LLMError):
    """Exception raised when API rate limit is exceeded."""

    def __init__(self, provider: str, model: str, message: str, retry_after: float | None = None):
        """
        Initialize RateLimitError.

        Args:
            provider: The LLM provider name
            model: The model identifier
            message: Error message
            retry_after: Optional seconds to wait before retrying
        """
        super().__init__(provider, model, message)
        self.retry_after = retry_after

    def __str__(self) -> str:
        """Return a string representation including retry_after if available."""
        base = f"Rate Limit Error [{self.provider}/{self.model}]: {self.message}"
        if self.retry_after is not None:
            base += f" (retry after {self.retry_after}s)"
        return base


class ContextLengthError(LLMError):
    """Exception raised when context window is exceeded."""

    def __init__(
        self, provider: str, model: str, message: str, max_tokens: int, actual_tokens: int
    ):
        """
        Initialize ContextLengthError.

        Args:
            provider: The LLM provider name
            model: The model identifier
            message: Error message
            max_tokens: Maximum allowed tokens
            actual_tokens: Actual number of tokens provided
        """
        super().__init__(provider, model, message)
        self.max_tokens = max_tokens
        self.actual_tokens = actual_tokens

    def __str__(self) -> str:
        """Return a string representation including token information."""
        return (
            f"Context Length Error [{self.provider}/{self.model}]: {self.message} "
            f"(max: {self.max_tokens}, actual: {self.actual_tokens})"
        )


class ProviderUnavailableError(LLMError):
    """Exception raised when the LLM provider API is not reachable."""

    def __str__(self) -> str:
        """Return a string representation indicating provider unavailability."""
        return f"Provider Unavailable [{self.provider}/{self.model}]: {self.message}"


class AuthenticationError(LLMError):
    """Exception raised when API key is invalid or authentication fails."""

    def __str__(self) -> str:
        """Return a string representation indicating authentication failure."""
        return f"Authentication Error [{self.provider}/{self.model}]: {self.message}"
