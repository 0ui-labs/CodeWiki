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


class DependencyFailedError(CodeWikiError):
    """Exception raised when a module cannot be processed due to a failed dependency.

    This error is used by ParallelModuleProcessor to signal that a module's
    dependency has failed, preventing the module from being processed.
    This prevents deadlocks by allowing dependent modules to fail fast
    instead of waiting indefinitely.

    Attributes:
        module: The name of the module that cannot be processed
        failed_dependency: The name of the dependency that failed
    """

    def __init__(self, module: str, failed_dependency: str) -> None:
        self.module = module
        self.failed_dependency = failed_dependency
        super().__init__(
            f"Module '{module}' cannot be processed: dependency '{failed_dependency}' failed"
        )


class InvalidDependencyError(CodeWikiError):
    """Exception raised when a dependency graph references an unknown module.

    This error is raised during dependency graph validation when a module
    declares a dependency on another module that doesn't exist in the
    module list.

    Attributes:
        module: The name of the module with the invalid dependency
        unknown_dependency: The name of the dependency that doesn't exist
    """

    def __init__(self, module: str, unknown_dependency: str) -> None:
        self.module = module
        self.unknown_dependency = unknown_dependency
        super().__init__(
            f"Module '{module}' depends on unknown module '{unknown_dependency}'"
        )


class CircularDependencyError(CodeWikiError):
    """Exception raised when a circular dependency is detected in the dependency graph.

    This error is raised during dependency graph validation when a cycle
    is detected, which would cause an infinite loop during processing.

    Attributes:
        cycle: List of module names forming the cycle (first and last are the same)
    """

    def __init__(self, cycle: list[str]) -> None:
        self.cycle = cycle
        cycle_str = " -> ".join(cycle)
        super().__init__(f"Circular dependency detected: {cycle_str}")
