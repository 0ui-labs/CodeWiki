"""Resilient LLM client with retry logic and fallback support.

This module provides a wrapper around LLMClient that adds:
- Exponential backoff with jitter for transient errors
- Automatic fallback to alternative models
- Proper classification of retryable vs fail-fast errors

Error Handling Strategy:
    - RateLimitError: Retry with exponential backoff (respects retry_after)
    - ProviderUnavailableError: Retry, then fallback to next model
    - ContextLengthError: Fail fast (input won't fit, retry is pointless)
    - AuthenticationError: Fail fast (wrong API key won't become right)

Example:
    >>> from codewiki.core.llm import LLMClient, ResilientLLMClient, RetryConfig
    >>> client = LLMClient(settings)
    >>> config = RetryConfig(max_retries=3, fallback_models=["gpt-4o"])
    >>> resilient = ResilientLLMClient(client, config, logger)
    >>> response = await resilient.complete(messages, "claude-sonnet-4-20250514")
"""

import asyncio
import random
from dataclasses import dataclass, field
from typing import Protocol

from codewiki.core.llm.client import LLMClient, LLMResponse
from codewiki.core.errors import (
    RateLimitError,
    ContextLengthError,
    ProviderUnavailableError,
    AuthenticationError,
    LLMError,
)


class LoggerProtocol(Protocol):
    """Minimal logger interface for compatibility with stdlib and future Rich logger.

    This protocol defines the minimum interface required for logging in the
    ResilientLLMClient. It is compatible with:
    - Standard library logging.Logger
    - Future CodeWikiLogger from REFACTOR-4
    - Any custom logger implementing these methods
    """

    def info(self, msg: str, *args, **kwargs) -> None:
        """Log informational message."""
        ...

    def warning(self, msg: str, *args, **kwargs) -> None:
        """Log warning message."""
        ...

    def error(self, msg: str, *args, **kwargs) -> None:
        """Log error message."""
        ...


@dataclass
class RetryConfig:
    """Configuration for retry and fallback behavior.

    Attributes:
        max_retries: Maximum retry attempts per model before trying fallback.
            Default is 3, meaning up to 3 attempts per model.
        base_delay: Base delay in seconds for exponential backoff.
            Actual delay is base_delay * 2^attempt + jitter.
        fallback_models: Ordered list of fallback model names to try if
            the primary model fails. Empty list means no fallbacks.

    Example:
        >>> config = RetryConfig(
        ...     max_retries=3,
        ...     base_delay=1.0,
        ...     fallback_models=["gpt-4o", "gpt-4o-mini"]
        ... )
    """

    max_retries: int = 3
    base_delay: float = 1.0
    fallback_models: list[str] = field(default_factory=list)


class ResilientLLMClient:
    """LLM client wrapper with retry logic and fallback support.

    This class wraps an LLMClient and adds resilience features:
    - Automatic retry with exponential backoff for transient errors
    - Fallback to alternative models when primary model fails
    - Proper error classification (fail-fast vs retryable)
    - User notification via logger when fallbacks are used

    The retry strategy uses nested loops:
    - Outer loop: Iterates through models (primary, then fallbacks)
    - Inner loop: Retry attempts for each model with exponential backoff

    Attributes:
        client: The underlying LLMClient instance
        config: RetryConfig with retry and fallback settings
        logger: Logger implementing LoggerProtocol

    Example:
        >>> client = LLMClient(settings)
        >>> config = RetryConfig(max_retries=3, fallback_models=["gpt-4o"])
        >>> resilient = ResilientLLMClient(client, config, logger)
        >>> response = await resilient.complete(
        ...     messages=[{"role": "user", "content": "Hello"}],
        ...     model="claude-sonnet-4-20250514"
        ... )
    """

    def __init__(
        self,
        client: LLMClient,
        config: RetryConfig,
        logger: LoggerProtocol,
    ) -> None:
        """Initialize ResilientLLMClient.

        Args:
            client: LLMClient instance to wrap
            config: RetryConfig with retry and fallback settings
            logger: Logger implementing info(), warning(), error() methods
        """
        self.client = client
        self.config = config
        self.logger = logger

    async def complete(
        self,
        messages: list[dict],
        model: str,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        **kwargs,
    ) -> LLMResponse:
        """Generate completion with automatic retry and fallback.

        This method attempts to get a completion from the primary model,
        retrying on transient errors with exponential backoff. If all
        retries are exhausted, it falls back to alternative models.

        Args:
            messages: List of message dicts with 'role' and 'content'
            model: Primary model identifier (e.g., 'claude-sonnet-4-20250514')
            temperature: Sampling temperature (default: 0.0)
            max_tokens: Maximum tokens to generate (default: 4096)
            **kwargs: Additional provider-specific parameters

        Returns:
            LLMResponse with content and token usage. The 'model' field
            reflects the actual model used (may differ if fallback was used).

        Raises:
            ContextLengthError: If input exceeds context window (fail-fast)
            AuthenticationError: If API key is invalid (fail-fast)
            LLMError: If all models and retries are exhausted
        """
        # Build model chain: primary model + fallbacks
        models_to_try = [model] + self.config.fallback_models
        last_error: Exception | None = None

        # Outer loop: try each model in chain
        for model_index, current_model in enumerate(models_to_try):
            is_fallback = model_index > 0

            if is_fallback:
                self.logger.warning(
                    f"Trying fallback model: {current_model} "
                    f"(after {models_to_try[model_index - 1]} failed)"
                )

            # Inner loop: retry attempts for current model
            for attempt in range(self.config.max_retries):
                try:
                    response = await self.client.complete(
                        messages=messages,
                        model=current_model,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        **kwargs,
                    )

                    # Success! Log if we used a fallback
                    if is_fallback:
                        self.logger.warning(
                            f"Successfully used fallback model: {current_model} "
                            f"(original request was for {model})"
                        )

                    return response

                except ContextLengthError:
                    # Fail fast: input too long, retry won't help
                    self.logger.error(
                        f"Context length exceeded for {current_model} - failing immediately"
                    )
                    raise

                except AuthenticationError:
                    # Fail fast: wrong API key won't become right
                    self.logger.error(
                        f"Authentication failed for {current_model} - failing immediately"
                    )
                    raise

                except RateLimitError as e:
                    # Retry with backoff (respect retry_after if provided)
                    last_error = e
                    delay = self._calculate_backoff(attempt, e.retry_after)

                    self.logger.warning(
                        f"Rate limit hit for {current_model} "
                        f"(attempt {attempt + 1}/{self.config.max_retries}), "
                        f"waiting {delay:.2f}s"
                    )

                    await asyncio.sleep(delay)
                    # Continue to next attempt

                except ProviderUnavailableError as e:
                    # Retry with backoff for transient provider errors
                    last_error = e
                    delay = self._calculate_backoff(attempt)

                    self.logger.warning(
                        f"Error from {current_model} "
                        f"(attempt {attempt + 1}/{self.config.max_retries}): {e}, "
                        f"waiting {delay:.2f}s"
                    )

                    await asyncio.sleep(delay)
                    # Continue to next attempt

            # All retries exhausted for this model
            self.logger.warning(
                f"Model {current_model} exhausted after {self.config.max_retries} attempts"
            )

        # All models failed
        fallback_info = (
            f" -> {', '.join(self.config.fallback_models)}"
            if self.config.fallback_models
            else ""
        )

        error_msg = (
            f"All models failed ({model}{fallback_info}). "
            f"Last error: {last_error}"
        )

        self.logger.error(error_msg)

        raise LLMError(
            provider="resilient",
            model=model,
            message=error_msg,
        )

    def _calculate_backoff(
        self,
        attempt: int,
        retry_after: float | None = None,
    ) -> float:
        """Calculate delay with exponential backoff and jitter.

        Uses exponential backoff with a small random jitter to prevent
        the "thundering herd" problem when multiple parallel tasks
        hit rate limits simultaneously.

        Args:
            attempt: Current retry attempt (0-indexed)
            retry_after: Optional delay from RateLimitError header.
                If provided, this value is used directly.

        Returns:
            Delay in seconds before next retry attempt
        """
        if retry_after is not None:
            return retry_after

        base_backoff = self.config.base_delay * (2**attempt)
        jitter = random.uniform(0, 0.5)  # 0-500ms random variation
        return base_backoff + jitter
