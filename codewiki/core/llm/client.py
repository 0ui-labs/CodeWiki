"""LLM client implementation with multi-provider support."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

try:
    import httpx
except ImportError:
    httpx = None  # type: ignore

try:
    from anthropic import AsyncAnthropic
except ImportError:
    AsyncAnthropic = None  # type: ignore

try:
    from openai import AsyncOpenAI
except ImportError:
    AsyncOpenAI = None  # type: ignore

try:
    import google.generativeai as genai
except ImportError:
    genai = None  # type: ignore

try:
    from google.api_core import exceptions as google_exceptions
except ImportError:
    google_exceptions = None  # type: ignore

try:
    import anthropic
except ImportError:
    anthropic = None  # type: ignore

try:
    import openai
except ImportError:
    openai = None  # type: ignore

from codewiki.core.config import Settings
from codewiki.core.llm.tokenizers import TokenCounter, MODEL_CONTEXT_LIMITS
from codewiki.core.llm.costs import calculate_cost
from codewiki.core.llm.pricing_provider import PricingProvider
from codewiki.core.llm.utils import detect_provider
from codewiki.core.errors import (
    LLMError,
    RateLimitError,
    AuthenticationError,
    ProviderUnavailableError,
    LLMTimeoutError,
    InvalidModelError,
)

if TYPE_CHECKING:
    from codewiki.core.logging import CodeWikiLogger

# Common model name aliases mapped to their correct identifiers.
# Used by validate_model() to auto-correct user-provided model names.
# Keys are lowercase for case-insensitive lookup.
MODEL_ALIASES: dict[str, str] = {
    # Claude aliases (missing date suffixes)
    "claude-opus-4.5": "claude-opus-4-5-20251101",
    "claude-sonnet-4.5": "claude-sonnet-4-5-20251101",
    "claude-sonnet-4": "claude-sonnet-4-20250514",
    "claude-opus-4": "claude-opus-4-20250514",
    "claude-3-5-sonnet": "claude-3-5-sonnet-20241022",
    "claude-3-5-haiku": "claude-3-5-haiku-20241022",
    "claude-3-opus": "claude-3-opus-20240229",
    "claude-3-sonnet": "claude-3-sonnet-20240229",
    "claude-3-haiku": "claude-3-haiku-20240307",
    # OpenAI aliases
    "gpt-4o-latest": "gpt-4o",
    "gpt-4-turbo-latest": "gpt-4-turbo",
    # o1/o3 aliases
    "o1-latest": "o1",
    "o3-latest": "o3",
    "o3-mini-latest": "o3-mini",
    # Gemini aliases
    "gemini-pro": "gemini-2.5-pro",
    "gemini-flash": "gemini-2.5-flash",
    "gemini-2-flash": "gemini-2.0-flash",
    "gemini-2.5": "gemini-2.5-pro",
    "gemini-2.0": "gemini-2.0-flash",
}

# Pre-computed case-insensitive mapping for MODEL_CONTEXT_LIMITS.
# Maps lowercase model names to their canonical cased versions.
# Created at module load for O(1) lookup performance.
_MODEL_CONTEXT_LIMITS_LOWER: dict[str, str] = {
    k.lower(): k for k in MODEL_CONTEXT_LIMITS
}


@dataclass
class ToolCall:
    """Represents a tool/function call from the LLM.

    Attributes:
        tool_name: Name of the tool being called
        args: Arguments to pass to the tool (as dict)
        tool_call_id: Unique identifier for this tool call
    """

    tool_name: str
    args: dict
    tool_call_id: str


@dataclass
class LLMResponse:
    """Response from LLM provider with token usage statistics and cost.

    Attributes:
        content: The text content returned by the LLM
        input_tokens: Number of tokens in the input/prompt
        output_tokens: Number of tokens in the output/completion
        model: Model identifier used for the request
        provider: Provider name (e.g., 'anthropic', 'openai')
        cost: Total cost in USD for this API call
        tool_calls: List of tool calls requested by the LLM (if any)
    """

    content: str
    input_tokens: int
    output_tokens: int
    model: str
    provider: str
    cost: float
    tool_calls: list[ToolCall] = field(default_factory=list)


class LLMClient:
    """Multi-provider LLM client with unified interface."""

    def __init__(
        self,
        settings: Settings,
        token_counter: Optional[TokenCounter] = None,
        logger: Optional[CodeWikiLogger] = None,
    ):
        """
        Initialize LLMClient with settings and optional token counter.

        Args:
            settings: Application settings with API keys and model config
            token_counter: Optional TokenCounter instance (creates new if None)
            logger: Optional CodeWikiLogger for structured logging.
                    If provided, logs model corrections and validation warnings.
                    If None, no logging is performed.
        """
        self.settings = settings
        self.token_counter = token_counter if token_counter is not None else TokenCounter(settings)
        self._logger = logger

        # Initialize pricing provider (fetches from LiteLLM if cache is stale)
        self._pricing_provider = PricingProvider(logger=logger)

        # Lazy-loaded provider clients
        self._anthropic: Optional[AsyncAnthropic] = None
        self._openai: Optional[AsyncOpenAI] = None
        self._groq: Optional[AsyncOpenAI] = None
        self._cerebras: Optional[AsyncOpenAI] = None
        self._google: Optional[object] = None  # Google uses GenerativeModel

    async def close(self) -> None:
        """
        Close all initialized async clients and release resources.

        This method is idempotent and can be called multiple times safely.
        Errors during cleanup are collected and raised after all cleanup attempts.
        """
        errors = []

        # Close Anthropic client
        if self._anthropic is not None:
            try:
                await self._anthropic.close()
            except Exception as e:
                errors.append(("anthropic", e))
            finally:
                self._anthropic = None

        # Close OpenAI client
        if self._openai is not None:
            try:
                await self._openai.close()
            except Exception as e:
                errors.append(("openai", e))
            finally:
                self._openai = None

        # Close Groq client (OpenAI-compatible)
        if self._groq is not None:
            try:
                await self._groq.close()
            except Exception as e:
                errors.append(("groq", e))
            finally:
                self._groq = None

        # Close Cerebras client (OpenAI-compatible)
        if self._cerebras is not None:
            try:
                await self._cerebras.close()
            except Exception as e:
                errors.append(("cerebras", e))
            finally:
                self._cerebras = None

        # Google GenerativeModel doesn't need explicit closing
        if self._google is not None:
            self._google = None

        if errors:
            error_msg = "; ".join(f"{client}: {str(err)}" for client, err in errors)
            raise RuntimeError(f"Errors during client cleanup: {error_msg}")

    async def __aenter__(self):
        """Async context manager entry - returns self."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit - closes all clients."""
        await self.close()
        return False  # Don't suppress exceptions

    async def complete(
        self,
        messages: list[dict],
        model: str,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        timeout: float = 60.0,
        **kwargs,
    ) -> LLMResponse:
        """
        Generate completion from LLM provider.

        Args:
            messages: List of message dicts with 'role' and 'content'
            model: Model identifier (e.g., 'gpt-4o', 'claude-sonnet-4-20250514')
            temperature: Sampling temperature (default: 0.0)
            max_tokens: Maximum tokens to generate (default: 4096)
            timeout: Request timeout in seconds (default: 60.0)
            **kwargs: Additional provider-specific parameters, including:
                - tools: List of tool definitions for function calling.
                  Format depends on provider:
                  - Anthropic: [{"name": str, "description": str, "input_schema": dict}]
                  - OpenAI/Groq/Cerebras: [{"type": "function", "function": {...}}]
                  If provided, the LLM may return tool_calls in the response.
                  Google Gemini does not currently support tools via this interface.

        Returns:
            LLMResponse with content, token usage, and tool_calls (if any)

        Raises:
            ContextLengthError: If input exceeds model's context window
            RateLimitError: If rate limit is exceeded
            AuthenticationError: If API key is invalid
            ProviderUnavailableError: If provider API is unreachable
            LLMTimeoutError: If request times out
            InvalidModelError: If model name is invalid (in strict mode)
        """
        # Validate and potentially auto-correct model name
        model = self.validate_model(model, strict=False)

        # Check context limit before making API call
        text = self._messages_to_text(messages)
        self.token_counter.check_context_limit(text, model)

        # Detect provider and delegate
        provider = self._detect_provider(model)

        match provider:
            case "anthropic":
                return await self._call_anthropic(
                    messages, model, temperature, max_tokens, timeout, **kwargs
                )
            case "openai":
                return await self._call_openai(messages, model, temperature, max_tokens, timeout, **kwargs)
            case "google":
                return await self._call_google(messages, model, temperature, max_tokens, timeout, **kwargs)
            case "groq":
                return await self._call_groq(messages, model, temperature, max_tokens, timeout, **kwargs)
            case "cerebras":
                return await self._call_cerebras(messages, model, temperature, max_tokens, timeout, **kwargs)
            case _:
                raise ValueError(f"Unsupported provider for model: {model}")

    def _detect_provider(self, model: str) -> str:
        """
        Detect provider from model name.

        Uses the centralized detect_provider function from codewiki.core.llm.utils.

        Args:
            model: Model name

        Returns:
            Provider name: 'anthropic', 'openai', 'google', 'groq', 'cerebras', or 'unknown'
        """
        return detect_provider(model)

    def _messages_to_text(self, messages: list[dict]) -> str:
        """
        Convert messages list to concatenated text.

        Args:
            messages: List of message dicts with 'content' field

        Returns:
            Concatenated message contents
        """
        return "\n".join(msg.get("content", "") for msg in messages)

    def validate_model(self, model: str, strict: bool = False) -> str:
        """
        Validate model name and auto-correct common aliases.

        Args:
            model: Model name to validate
            strict: If True, raise error for unknown models. If False, warn and allow.

        Returns:
            Validated (and potentially corrected) model name

        Raises:
            InvalidModelError: If model is invalid in strict mode

        Note:
            - Known aliases are auto-corrected (e.g., "claude-sonnet-4" -> "claude-sonnet-4-20250514")
            - Known models pass through unchanged
            - Unknown models: warn but allow in non-strict mode, raise in strict mode
        """
        model_lower = model.lower()

        # Check if it's a known alias (case-insensitive)
        if model_lower in MODEL_ALIASES:
            corrected = MODEL_ALIASES[model_lower]
            if self._logger:
                self._logger.info(
                    f"Auto-correcting model alias '{model}' to '{corrected}'",
                    original_model=model,
                    corrected_model=corrected,
                )
            return corrected

        # Check if it's a known model (case-insensitive)
        # This handles inputs like "GPT-4O" -> "gpt-4o"
        if model_lower in _MODEL_CONTEXT_LIMITS_LOWER:
            canonical = _MODEL_CONTEXT_LIMITS_LOWER[model_lower]
            if canonical != model and self._logger:
                self._logger.debug(
                    f"Normalized model case '{model}' to '{canonical}'",
                    original_model=model,
                    canonical_model=canonical,
                )
            return canonical

        # Unknown model
        if strict:
            # In strict mode, raise error with suggestions
            provider = self._detect_provider(model)
            suggestions = self._get_model_suggestions(model, provider)
            raise InvalidModelError(
                provider=provider,
                model=model,
                message=f"Model '{model}' is not recognized. This may cause API errors.",
                suggestions=suggestions,
            )
        else:
            # In non-strict mode, warn and allow
            if self._logger:
                self._logger.warning(
                    f"Model '{model}' is not in the known models list. "
                    f"This may be a custom model or could cause API errors if invalid.",
                    model=model,
                )
            return model

    def _get_model_suggestions(self, invalid_model: str, provider: str) -> list[str] | None:
        """
        Get suggestions for an invalid model name.

        Args:
            invalid_model: The invalid model name
            provider: Detected provider

        Returns:
            List of suggested model names, or None if no suggestions found
        """
        suggestions = []
        invalid_lower = invalid_model.lower()

        # Check aliases first
        if invalid_lower in MODEL_ALIASES:
            suggestions.append(MODEL_ALIASES[invalid_lower])

        # Find similar models from the known list
        for known_model in MODEL_CONTEXT_LIMITS.keys():
            known_lower = known_model.lower()
            # Match by provider
            if self._detect_provider(known_model) == provider:
                # Check if invalid model is a prefix of known model
                if known_lower.startswith(invalid_lower):
                    if known_model not in suggestions:
                        suggestions.append(known_model)
                # Check if they share common prefix (e.g., "claude-sonnet")
                elif invalid_lower.split("-")[0:2] == known_lower.split("-")[0:2]:
                    # Same family (e.g., both are claude-sonnet)
                    if known_model not in suggestions:
                        suggestions.append(known_model)

        # Limit to top 5 suggestions and return None if empty
        return suggestions[:5] if suggestions else None

    async def _call_anthropic(
        self,
        messages: list[dict],
        model: str,
        temperature: float,
        max_tokens: int,
        timeout: float,
        **kwargs,
    ) -> LLMResponse:
        """Call Anthropic API."""
        if AsyncAnthropic is None:
            raise ImportError("anthropic package required for Claude models")

        # Lazy-init client without timeout (timeout is passed per-call)
        if self._anthropic is None:
            api_key = self.settings.get_api_key("anthropic")
            self._anthropic = AsyncAnthropic(api_key=api_key)

        try:
            response = await self._anthropic.messages.create(
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                messages=messages,
                timeout=timeout,
                **kwargs,
            )

            # Parse response content - can contain both text and tool calls
            content = ""
            tool_calls = []

            for block in response.content:
                if hasattr(block, "text"):
                    # Text content block
                    content += block.text
                elif hasattr(block, "type") and block.type == "tool_use":
                    # Anthropic tool use block
                    tool_calls.append(
                        ToolCall(
                            tool_name=block.name,
                            args=block.input if isinstance(block.input, dict) else {},
                            tool_call_id=block.id,
                        )
                    )

            input_tokens = response.usage.input_tokens
            output_tokens = response.usage.output_tokens
            cost = calculate_cost(model, input_tokens, output_tokens, pricing_provider=self._pricing_provider)

            return LLMResponse(
                content=content,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                model=model,
                provider="anthropic",
                cost=cost,
                tool_calls=tool_calls,
            )
        except Exception as e:
            # Wrap exceptions in LLMError subclasses
            self._handle_exception(e, "anthropic", model, timeout)

    async def _call_openai(
        self,
        messages: list[dict],
        model: str,
        temperature: float,
        max_tokens: int,
        timeout: float,
        **kwargs,
    ) -> LLMResponse:
        """Call OpenAI API."""
        if AsyncOpenAI is None:
            raise ImportError("openai package required for OpenAI models")

        # Lazy-init client without timeout (timeout is passed per-call)
        if self._openai is None:
            api_key = self.settings.get_api_key("openai")
            self._openai = AsyncOpenAI(api_key=api_key)

        try:
            response = await self._openai.chat.completions.create(
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                messages=messages,
                timeout=timeout,
                **kwargs,
            )

            # Parse response message - can contain both text and tool calls
            message = response.choices[0].message
            content = message.content or ""
            tool_calls = []

            if message.tool_calls:
                import json

                for tc in message.tool_calls:
                    # Parse arguments (may be JSON string)
                    args = {}
                    if tc.function.arguments:
                        try:
                            args = json.loads(tc.function.arguments)
                        except json.JSONDecodeError:
                            # If parsing fails, store as empty dict
                            args = {}

                    tool_calls.append(
                        ToolCall(
                            tool_name=tc.function.name,
                            args=args,
                            tool_call_id=tc.id,
                        )
                    )

            input_tokens = response.usage.prompt_tokens
            output_tokens = response.usage.completion_tokens
            cost = calculate_cost(model, input_tokens, output_tokens, pricing_provider=self._pricing_provider)

            return LLMResponse(
                content=content,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                model=model,
                provider="openai",
                cost=cost,
                tool_calls=tool_calls,
            )
        except Exception as e:
            self._handle_exception(e, "openai", model, timeout)

    async def _call_google(
        self,
        messages: list[dict],
        model: str,
        temperature: float,
        max_tokens: int,
        timeout: float,
        **kwargs,
    ) -> LLMResponse:
        """Call Google Gemini API."""
        if genai is None:
            raise ImportError("google-generativeai package required for Gemini models")

        # Configure API key
        api_key = self.settings.get_api_key("google")
        genai.configure(api_key=api_key)

        # Lazy-init model
        if self._google is None:
            self._google = genai.GenerativeModel(model_name=model)

        # Convert messages to Google format
        contents = []

        for msg in messages:
            if msg.get("role") == "system":
                # System messages are not used in Google format (could be added to first user message)
                pass
            elif msg.get("role") in ("user", "assistant"):
                # Google uses "model" instead of "assistant"
                role = "model" if msg["role"] == "assistant" else "user"
                contents.append({"role": role, "parts": [msg["content"]]})

        try:
            generation_config = genai.GenerationConfig(
                temperature=temperature, max_output_tokens=max_tokens
            )

            # Google SDK uses request_options for timeout
            request_options = kwargs.pop("request_options", {})
            request_options["timeout"] = timeout

            response = await self._google.generate_content_async(
                contents=contents,
                generation_config=generation_config,
                request_options=request_options,
                **kwargs,
            )

            input_tokens = response.usage_metadata.prompt_token_count
            output_tokens = response.usage_metadata.candidates_token_count
            cost = calculate_cost(model, input_tokens, output_tokens, pricing_provider=self._pricing_provider)

            return LLMResponse(
                content=response.text,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                model=model,
                provider="google",
                cost=cost,
            )
        except Exception as e:
            self._handle_exception(e, "google", model, timeout)

    async def _call_groq(
        self,
        messages: list[dict],
        model: str,
        temperature: float,
        max_tokens: int,
        timeout: float,
        **kwargs,
    ) -> LLMResponse:
        """Call Groq API (OpenAI-compatible)."""
        if AsyncOpenAI is None:
            raise ImportError("openai package required for Groq models")

        # Lazy-init client with Groq base URL (timeout is passed per-call)
        if self._groq is None:
            api_key = self.settings.get_api_key("groq")
            self._groq = AsyncOpenAI(
                base_url="https://api.groq.com/openai/v1",
                api_key=api_key,
            )

        # Remove 'groq/' prefix from model name
        model_name = model.replace("groq/", "")

        try:
            response = await self._groq.chat.completions.create(
                model=model_name,
                max_tokens=max_tokens,
                temperature=temperature,
                messages=messages,
                timeout=timeout,
                **kwargs,
            )

            # Parse response message - can contain both text and tool calls
            message = response.choices[0].message
            content = message.content or ""
            tool_calls = []

            if message.tool_calls:
                import json

                for tc in message.tool_calls:
                    # Parse arguments (may be JSON string)
                    args = {}
                    if tc.function.arguments:
                        try:
                            args = json.loads(tc.function.arguments)
                        except json.JSONDecodeError:
                            args = {}

                    tool_calls.append(
                        ToolCall(
                            tool_name=tc.function.name,
                            args=args,
                            tool_call_id=tc.id,
                        )
                    )

            input_tokens = response.usage.prompt_tokens
            output_tokens = response.usage.completion_tokens
            cost = calculate_cost(model, input_tokens, output_tokens, pricing_provider=self._pricing_provider)

            return LLMResponse(
                content=content,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                model=model,
                provider="groq",
                cost=cost,
                tool_calls=tool_calls,
            )
        except Exception as e:
            self._handle_exception(e, "groq", model, timeout)

    async def _call_cerebras(
        self,
        messages: list[dict],
        model: str,
        temperature: float,
        max_tokens: int,
        timeout: float,
        **kwargs,
    ) -> LLMResponse:
        """Call Cerebras API (OpenAI-compatible)."""
        if AsyncOpenAI is None:
            raise ImportError("openai package required for Cerebras models")

        # Lazy-init client with Cerebras base URL (timeout is passed per-call)
        if self._cerebras is None:
            api_key = self.settings.get_api_key("cerebras")
            self._cerebras = AsyncOpenAI(
                base_url="https://api.cerebras.ai/v1",
                api_key=api_key,
            )

        # Remove 'cerebras/' prefix from model name
        model_name = model.replace("cerebras/", "")

        try:
            response = await self._cerebras.chat.completions.create(
                model=model_name,
                max_tokens=max_tokens,
                temperature=temperature,
                messages=messages,
                timeout=timeout,
                **kwargs,
            )

            # Parse response message - can contain both text and tool calls
            message = response.choices[0].message
            content = message.content or ""
            tool_calls = []

            if message.tool_calls:
                import json

                for tc in message.tool_calls:
                    # Parse arguments (may be JSON string)
                    args = {}
                    if tc.function.arguments:
                        try:
                            args = json.loads(tc.function.arguments)
                        except json.JSONDecodeError:
                            args = {}

                    tool_calls.append(
                        ToolCall(
                            tool_name=tc.function.name,
                            args=args,
                            tool_call_id=tc.id,
                        )
                    )

            input_tokens = response.usage.prompt_tokens
            output_tokens = response.usage.completion_tokens
            cost = calculate_cost(model, input_tokens, output_tokens, pricing_provider=self._pricing_provider)

            return LLMResponse(
                content=content,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                model=model,
                provider="cerebras",
                cost=cost,
                tool_calls=tool_calls,
            )
        except Exception as e:
            self._handle_exception(e, "cerebras", model, timeout)

    def _handle_exception(self, e: Exception, provider: str, model: str, timeout: float) -> None:
        """
        Handle and wrap provider exceptions.

        Args:
            e: Original exception
            provider: Provider name
            model: Model name
            timeout: Timeout value used for the request

        Raises:
            Appropriate LLMError subclass
        """
        # Re-raise if already an LLMError
        if isinstance(e, LLMError):
            raise e

        # Handle timeout exceptions from httpx (used by Anthropic and OpenAI SDKs)
        if httpx and isinstance(e, httpx.TimeoutException):
            raise LLMTimeoutError(
                provider=provider,
                model=model,
                message=f"Request timed out after {timeout}s",
                timeout=timeout
            )

        # Map Anthropic exceptions
        if anthropic:
            if isinstance(e, anthropic.RateLimitError):
                retry_after = getattr(e, "retry_after", None)
                raise RateLimitError(
                    provider=provider, model=model, message=str(e), retry_after=retry_after
                )
            if isinstance(e, anthropic.AuthenticationError):
                raise AuthenticationError(provider=provider, model=model, message=str(e))
            if isinstance(e, anthropic.APIError):
                raise ProviderUnavailableError(provider=provider, model=model, message=str(e))

        # Map OpenAI exceptions (also used by Groq/Cerebras)
        if openai:
            if isinstance(e, openai.RateLimitError):
                raise RateLimitError(provider=provider, model=model, message=str(e))
            if isinstance(e, openai.AuthenticationError):
                raise AuthenticationError(provider=provider, model=model, message=str(e))
            if isinstance(e, openai.APIError):
                raise ProviderUnavailableError(provider=provider, model=model, message=str(e))

        # Map Google exceptions
        if google_exceptions:
            # Check for deadline/timeout first
            if hasattr(google_exceptions, 'DeadlineExceeded') and isinstance(e, google_exceptions.DeadlineExceeded):
                raise LLMTimeoutError(
                    provider=provider,
                    model=model,
                    message=f"Request timed out after {timeout}s",
                    timeout=timeout
                )
            if hasattr(google_exceptions, 'ResourceExhausted') and isinstance(e, google_exceptions.ResourceExhausted):
                raise RateLimitError(provider=provider, model=model, message=str(e))
            if hasattr(google_exceptions, 'Unauthenticated') and isinstance(e, google_exceptions.Unauthenticated):
                raise AuthenticationError(provider=provider, model=model, message=str(e))
            if hasattr(google_exceptions, 'GoogleAPIError') and isinstance(e, google_exceptions.GoogleAPIError):
                raise ProviderUnavailableError(provider=provider, model=model, message=str(e))

        # Fallback for unknown exceptions
        raise ProviderUnavailableError(
            provider=provider, model=model, message=f"Unexpected error: {str(e)}"
        ) from e
