"""Exception hierarchy for CodeWiki Core."""

from abc import ABC
from dataclasses import dataclass


class CodeWikiError(Exception):
    """Base exception for all CodeWiki Core errors."""

    pass


@dataclass
class LLMError(CodeWikiError, ABC):
    """Abstract base exception for all LLM-related errors."""

    provider: str
    model: str
    message: str

    def __post_init__(self) -> None:
        """Initialize the Exception base class with the message."""
        super().__init__(self.message)

    def __str__(self) -> str:
        """Return a string representation including provider and model info."""
        return f"LLM Error [{self.provider}/{self.model}]: {self.message}"


@dataclass
class RateLimitError(LLMError):
    """Exception raised when API rate limit is exceeded."""

    retry_after: float | None = None

    def __str__(self) -> str:
        """Return a string representation including retry_after if available."""
        base = f"Rate Limit Error [{self.provider}/{self.model}]: {self.message}"
        if self.retry_after is not None:
            base += f" (retry after {self.retry_after}s)"
        return base


@dataclass
class ContextLengthError(LLMError):
    """Exception raised when context window is exceeded."""

    max_tokens: int
    actual_tokens: int

    def __str__(self) -> str:
        """Return a string representation including token information."""
        return (
            f"Context Length Error [{self.provider}/{self.model}]: {self.message} "
            f"(max: {self.max_tokens}, actual: {self.actual_tokens})"
        )


@dataclass
class ProviderUnavailableError(LLMError):
    """Exception raised when the LLM provider API is not reachable."""

    def __str__(self) -> str:
        """Return a string representation indicating provider unavailability."""
        return f"Provider Unavailable [{self.provider}/{self.model}]: {self.message}"


@dataclass
class AuthenticationError(LLMError):
    """Exception raised when API key is invalid or authentication fails."""

    def __str__(self) -> str:
        """Return a string representation indicating authentication failure."""
        return f"Authentication Error [{self.provider}/{self.model}]: {self.message}"


@dataclass
class LLMTimeoutError(LLMError):
    """Exception raised when API request times out.

    Note: Named LLMTimeoutError to avoid shadowing Python's built-in TimeoutError.
    """

    timeout: float

    def __str__(self) -> str:
        """Return a string representation including timeout duration."""
        return f"Timeout Error [{self.provider}/{self.model}]: {self.message} (timeout: {self.timeout}s)"


@dataclass
class InvalidModelError(LLMError):
    """Exception raised when model name is invalid or not supported."""

    suggestions: list[str] | None = None

    def __str__(self) -> str:
        """Return a string representation including suggestions if available."""
        base = f"Invalid Model Error [{self.provider}/{self.model}]: {self.message}"
        if self.suggestions:
            base += f" (suggestions: {', '.join(self.suggestions)})"
        return base
