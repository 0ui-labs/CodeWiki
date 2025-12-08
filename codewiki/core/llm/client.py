"""LLM client implementation with multi-provider support."""

from dataclasses import dataclass
from typing import Optional

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
from codewiki.core.llm.tokenizers import TokenCounter
from codewiki.core.errors import (
    LLMError,
    RateLimitError,
    AuthenticationError,
    ProviderUnavailableError,
)


@dataclass
class LLMResponse:
    """Response from LLM provider with token usage statistics.

    Attributes:
        content: The text content returned by the LLM
        input_tokens: Number of tokens in the input/prompt
        output_tokens: Number of tokens in the output/completion
        model: Model identifier used for the request
        provider: Provider name (e.g., 'anthropic', 'openai')
    """

    content: str
    input_tokens: int
    output_tokens: int
    model: str
    provider: str


class LLMClient:
    """Multi-provider LLM client with unified interface."""

    def __init__(self, settings: Settings, token_counter: Optional[TokenCounter] = None):
        """
        Initialize LLMClient with settings and optional token counter.

        Args:
            settings: Application settings with API keys and model config
            token_counter: Optional TokenCounter instance (creates new if None)
        """
        self.settings = settings
        self.token_counter = token_counter if token_counter is not None else TokenCounter()

        # Lazy-loaded provider clients
        self._anthropic: Optional[AsyncAnthropic] = None
        self._openai: Optional[AsyncOpenAI] = None
        self._groq: Optional[AsyncOpenAI] = None
        self._cerebras: Optional[AsyncOpenAI] = None
        self._google: Optional[object] = None  # Google uses GenerativeModel

    async def complete(
        self,
        messages: list[dict],
        model: str,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        **kwargs,
    ) -> LLMResponse:
        """
        Generate completion from LLM provider.

        Args:
            messages: List of message dicts with 'role' and 'content'
            model: Model identifier (e.g., 'gpt-4o', 'claude-sonnet-4-20250514')
            temperature: Sampling temperature (default: 0.0)
            max_tokens: Maximum tokens to generate (default: 4096)
            **kwargs: Additional provider-specific parameters

        Returns:
            LLMResponse with content and token usage

        Raises:
            ContextLengthError: If input exceeds model's context window
            RateLimitError: If rate limit is exceeded
            AuthenticationError: If API key is invalid
            ProviderUnavailableError: If provider API is unreachable
        """
        # Check context limit before making API call
        text = self._messages_to_text(messages)
        self.token_counter.check_context_limit(text, model)

        # Detect provider and delegate
        provider = self._detect_provider(model)

        match provider:
            case "anthropic":
                return await self._call_anthropic(
                    messages, model, temperature, max_tokens, **kwargs
                )
            case "openai":
                return await self._call_openai(messages, model, temperature, max_tokens, **kwargs)
            case "google":
                return await self._call_google(messages, model, temperature, max_tokens, **kwargs)
            case "groq":
                return await self._call_groq(messages, model, temperature, max_tokens, **kwargs)
            case "cerebras":
                return await self._call_cerebras(messages, model, temperature, max_tokens, **kwargs)
            case _:
                raise ValueError(f"Unsupported provider for model: {model}")

    def _detect_provider(self, model: str) -> str:
        """
        Detect provider from model name.

        Args:
            model: Model name

        Returns:
            Provider name: 'anthropic', 'openai', 'google', 'groq', 'cerebras'
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

    def _messages_to_text(self, messages: list[dict]) -> str:
        """
        Convert messages list to concatenated text.

        Args:
            messages: List of message dicts with 'content' field

        Returns:
            Concatenated message contents
        """
        return "\n".join(msg.get("content", "") for msg in messages)

    async def _call_anthropic(
        self,
        messages: list[dict],
        model: str,
        temperature: float,
        max_tokens: int,
        **kwargs,
    ) -> LLMResponse:
        """Call Anthropic API."""
        if AsyncAnthropic is None:
            raise ImportError("anthropic package required for Claude models")

        # Lazy-init client
        if self._anthropic is None:
            api_key = self.settings.get_api_key("anthropic")
            self._anthropic = AsyncAnthropic(api_key=api_key)

        try:
            response = await self._anthropic.messages.create(
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                messages=messages,
                **kwargs,
            )

            return LLMResponse(
                content=response.content[0].text,
                input_tokens=response.usage.input_tokens,
                output_tokens=response.usage.output_tokens,
                model=model,
                provider="anthropic",
            )
        except Exception as e:
            # Wrap exceptions in LLMError subclasses
            self._handle_exception(e, "anthropic", model)

    async def _call_openai(
        self,
        messages: list[dict],
        model: str,
        temperature: float,
        max_tokens: int,
        **kwargs,
    ) -> LLMResponse:
        """Call OpenAI API."""
        if AsyncOpenAI is None:
            raise ImportError("openai package required for OpenAI models")

        # Lazy-init client
        if self._openai is None:
            api_key = self.settings.get_api_key("openai")
            self._openai = AsyncOpenAI(api_key=api_key)

        try:
            response = await self._openai.chat.completions.create(
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                messages=messages,
                **kwargs,
            )

            return LLMResponse(
                content=response.choices[0].message.content,
                input_tokens=response.usage.prompt_tokens,
                output_tokens=response.usage.completion_tokens,
                model=model,
                provider="openai",
            )
        except Exception as e:
            self._handle_exception(e, "openai", model)

    async def _call_google(
        self,
        messages: list[dict],
        model: str,
        temperature: float,
        max_tokens: int,
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

            response = await self._google.generate_content_async(
                contents=contents,
                generation_config=generation_config,
                **kwargs,
            )

            return LLMResponse(
                content=response.text,
                input_tokens=response.usage_metadata.prompt_token_count,
                output_tokens=response.usage_metadata.candidates_token_count,
                model=model,
                provider="google",
            )
        except Exception as e:
            self._handle_exception(e, "google", model)

    async def _call_groq(
        self,
        messages: list[dict],
        model: str,
        temperature: float,
        max_tokens: int,
        **kwargs,
    ) -> LLMResponse:
        """Call Groq API (OpenAI-compatible)."""
        if AsyncOpenAI is None:
            raise ImportError("openai package required for Groq models")

        # Lazy-init client with Groq base URL
        if self._groq is None:
            api_key = self.settings.get_api_key("groq")
            self._groq = AsyncOpenAI(base_url="https://api.groq.com/openai/v1", api_key=api_key)

        # Remove 'groq/' prefix from model name
        model_name = model.replace("groq/", "")

        try:
            response = await self._groq.chat.completions.create(
                model=model_name,
                max_tokens=max_tokens,
                temperature=temperature,
                messages=messages,
                **kwargs,
            )

            return LLMResponse(
                content=response.choices[0].message.content,
                input_tokens=response.usage.prompt_tokens,
                output_tokens=response.usage.completion_tokens,
                model=model,
                provider="groq",
            )
        except Exception as e:
            self._handle_exception(e, "groq", model)

    async def _call_cerebras(
        self,
        messages: list[dict],
        model: str,
        temperature: float,
        max_tokens: int,
        **kwargs,
    ) -> LLMResponse:
        """Call Cerebras API (OpenAI-compatible)."""
        if AsyncOpenAI is None:
            raise ImportError("openai package required for Cerebras models")

        # Lazy-init client with Cerebras base URL
        if self._cerebras is None:
            api_key = self.settings.get_api_key("cerebras")
            self._cerebras = AsyncOpenAI(base_url="https://api.cerebras.ai/v1", api_key=api_key)

        # Remove 'cerebras/' prefix from model name
        model_name = model.replace("cerebras/", "")

        try:
            response = await self._cerebras.chat.completions.create(
                model=model_name,
                max_tokens=max_tokens,
                temperature=temperature,
                messages=messages,
                **kwargs,
            )

            return LLMResponse(
                content=response.choices[0].message.content,
                input_tokens=response.usage.prompt_tokens,
                output_tokens=response.usage.completion_tokens,
                model=model,
                provider="cerebras",
            )
        except Exception as e:
            self._handle_exception(e, "cerebras", model)

    def _handle_exception(self, e: Exception, provider: str, model: str) -> None:
        """
        Handle and wrap provider exceptions.

        Args:
            e: Original exception
            provider: Provider name
            model: Model name

        Raises:
            Appropriate LLMError subclass
        """
        # Re-raise if already an LLMError
        if isinstance(e, LLMError):
            raise e

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
            if isinstance(e, google_exceptions.ResourceExhausted):
                raise RateLimitError(provider=provider, model=model, message=str(e))
            if isinstance(e, google_exceptions.Unauthenticated):
                raise AuthenticationError(provider=provider, model=model, message=str(e))
            if isinstance(e, google_exceptions.GoogleAPIError):
                raise ProviderUnavailableError(provider=provider, model=model, message=str(e))

        # Fallback for unknown exceptions
        raise ProviderUnavailableError(
            provider=provider, model=model, message=f"Unexpected error: {str(e)}"
        ) from e
