"""pydantic_ai Model wrapper for CodeWiki's ResilientLLMClient.

This module provides a custom pydantic_ai Model implementation that integrates
with CodeWiki's LLM infrastructure:
- Uses ResilientLLMClient for retry logic and fallback support
- Supports tool calls for pydantic_ai Agents
- Provides proper message conversion between pydantic_ai and core formats
- Integrates with CodeWiki's logging and error handling

Example:
    >>> from codewiki.core import Settings, get_logger
    >>> from codewiki.src.be.llm_adapter import CodeWikiModel
    >>>
    >>> settings = Settings()
    >>> logger = get_logger(settings)
    >>> model = CodeWikiModel(settings, logger)
    >>>
    >>> # Use with pydantic_ai Agent
    >>> from pydantic_ai import Agent
    >>> agent = Agent(model, system_prompt="You are a helpful assistant")
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from pydantic_ai.models import (
    Model,
    ModelRequestParameters,
    ModelSettings,
    RequestUsage,
)
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    SystemPromptPart,
    UserPromptPart,
    ToolReturnPart,
    TextPart,
    ToolCallPart,
)

from codewiki.core.config import Settings
from codewiki.core.llm import ResilientLLMClient, LLMClient, RetryConfig
from codewiki.core.errors import LLMError

if TYPE_CHECKING:
    from codewiki.core.logging import CodeWikiLogger


class CodeWikiModel(Model):
    """pydantic_ai Model wrapper for ResilientLLMClient.

    This model implementation provides:
    - Integration with CodeWiki's LLM infrastructure
    - Automatic retry and fallback support
    - Tool call support for pydantic_ai Agents
    - Proper error handling and logging

    Note: This implementation currently does NOT support tool calls from the LLM.
    Tool calls require native provider SDK integration (Anthropic/OpenAI format)
    which is not yet available in ResilientLLMClient. The LLMClient only returns
    text content.

    To add tool call support in the future:
    1. Extend LLMClient._call_anthropic and _call_openai to parse tool_use blocks
    2. Update LLMResponse to include optional tool_calls field
    3. Update this adapter to convert tool calls to ToolCallPart objects

    Attributes:
        settings: Application settings with API keys and model config
        logger: CodeWikiLogger for structured logging
        client: ResilientLLMClient instance for making LLM calls
    """

    def __init__(
        self,
        codewiki_settings: Settings,
        codewiki_logger: CodeWikiLogger,
        *,
        settings: ModelSettings | None = None,
        profile: Any = None,
    ):
        """Initialize the CodeWikiModel.

        Args:
            codewiki_settings: Application settings with API keys and model config
            codewiki_logger: CodeWikiLogger for structured logging
            settings: Optional pydantic_ai ModelSettings (passed to parent)
            profile: Optional pydantic_ai model profile (passed to parent)
        """
        super().__init__(settings=settings, profile=profile)
        self.codewiki_settings = codewiki_settings
        self.codewiki_logger = codewiki_logger

        # Initialize resilient LLM client
        base_client = LLMClient(codewiki_settings, logger=codewiki_logger)
        retry_config = RetryConfig(
            max_retries=codewiki_settings.retry_attempts,
            base_delay=codewiki_settings.retry_base_delay,
            fallback_models=codewiki_settings.fallback_models,
        )
        self.client = ResilientLLMClient(base_client, retry_config, codewiki_logger)

    @property
    def model_name(self) -> str:
        """Return the model name for OpenTelemetry semantic conventions.

        This property is required by pydantic_ai's Model interface and is used
        for telemetry and logging purposes.
        """
        return self.codewiki_settings.main_model

    @property
    def system(self) -> str:
        """Return the system identifier for OpenTelemetry semantic conventions.

        This property is required by pydantic_ai's Model interface and indicates
        the underlying LLM provider/system being used.

        Returns the provider name based on the main model, e.g.:
        - "anthropic" for Claude models
        - "openai" for GPT models
        - "google" for Gemini models
        """
        # Detect provider from model name
        model_lower = self.codewiki_settings.main_model.lower()
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

    def name(self) -> str:
        """Return model identifier for logging and debugging."""
        return f"codewiki:{self.codewiki_settings.main_model}"

    async def request(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> ModelResponse:
        """Process a model request and return response with tool calls.

        This method:
        1. Converts pydantic_ai messages to core LLM format
        2. Calls ResilientLLMClient.complete() with retry/fallback support
        3. Converts LLM response to pydantic_ai ModelResponse format
        4. Handles tool calls if present (currently NOT supported, see class docstring)

        Args:
            messages: List of pydantic_ai ModelMessage objects
            model_settings: Optional temperature/max_tokens overrides
            model_request_parameters: Tool definitions and output modes

        Returns:
            ModelResponse with content, usage, and tool calls (if any)

        Raises:
            LLMError: If all retries and fallbacks are exhausted
        """
        # Convert pydantic_ai messages to core format
        core_messages = self._convert_messages_to_core(messages)

        # Extract settings - try both dict-like and attribute access
        temperature = 0.0
        max_tokens = 4096
        if model_settings:
            # Handle both dict-like and object-like settings
            if isinstance(model_settings, dict):
                temperature = model_settings.get("temperature", 0.0)
                max_tokens = model_settings.get("max_tokens", 4096)
            else:
                temperature = getattr(model_settings, "temperature", 0.0) or 0.0
                max_tokens = getattr(model_settings, "max_tokens", 4096) or 4096

        # Call LLM with retry/fallback support
        llm_response = await self.client.complete(
            messages=core_messages,
            model=self.codewiki_settings.main_model,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        # Convert to pydantic_ai format
        # Note: LLMResponse only contains text content, no tool calls yet
        parts = [TextPart(content=llm_response.content)]

        # Build usage stats
        usage = RequestUsage(
            input_tokens=llm_response.input_tokens,
            output_tokens=llm_response.output_tokens,
            details={
                "model": llm_response.model,
                "provider": llm_response.provider,
                "cost_usd": llm_response.cost,
            },
        )

        return ModelResponse(
            parts=parts,
            usage=usage,
            model_name=llm_response.model,
            timestamp=datetime.now(timezone.utc),
        )

    def _convert_messages_to_core(self, messages: list[ModelMessage]) -> list[dict]:
        """Convert pydantic_ai messages to core LLM format.

        Handles:
        - ModelRequest with system prompt and user parts
        - Mixed message types (SystemPromptPart, UserPromptPart, ToolReturnPart, etc.)
        - Multi-part content (lists of parts)

        Args:
            messages: List of pydantic_ai ModelMessage objects

        Returns:
            List of message dicts with 'role' and 'content' keys
        """
        core_messages = []

        for msg in messages:
            if isinstance(msg, ModelRequest):
                # Handle ModelRequest with system prompt and parts
                if msg.instructions:
                    core_messages.append({"role": "system", "content": msg.instructions})

                # Convert request parts to user message
                if msg.parts:
                    content = self._convert_parts_to_text(msg.parts)
                    if content:
                        core_messages.append({"role": "user", "content": content})

            elif isinstance(msg, ModelResponse):
                # Handle ModelResponse (assistant messages)
                content = self._convert_parts_to_text(msg.parts)
                if content:
                    core_messages.append({"role": "assistant", "content": content})

            else:
                # Unknown message type - try to extract content
                self.codewiki_logger.warning(f"Unknown message type: {type(msg)}")

        return core_messages

    def _convert_parts_to_text(self, parts: list) -> str:
        """Convert message parts to text content.

        Handles:
        - TextPart: Extract text content
        - UserPromptPart: Extract text
        - SystemPromptPart: Extract text
        - ToolReturnPart: Convert tool return to text description
        - ToolCallPart: Convert tool call to text description

        Args:
            parts: List of message parts

        Returns:
            Concatenated text content
        """
        text_parts = []

        for part in parts:
            if isinstance(part, TextPart):
                text_parts.append(part.content)
            elif isinstance(part, UserPromptPart):
                text_parts.append(part.content)
            elif isinstance(part, SystemPromptPart):
                text_parts.append(part.content)
            elif isinstance(part, ToolReturnPart):
                # Convert tool return to text format
                tool_result = f"[Tool {part.tool_name} returned: {part.content}]"
                text_parts.append(tool_result)
            elif isinstance(part, ToolCallPart):
                # Convert tool call to text format
                tool_call = f"[Calling tool {part.tool_name} with args: {part.args}]"
                text_parts.append(tool_call)
            else:
                # Unknown part type - try to get string representation
                self.codewiki_logger.warning(f"Unknown part type: {type(part)}")
                text_parts.append(str(part))

        return "\n".join(text_parts)
