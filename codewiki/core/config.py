"""Configuration management using Pydantic Settings."""

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables with CODEWIKI_ prefix."""

    model_config = SettingsConfigDict(
        env_prefix="CODEWIKI_", env_file=".env", env_file_encoding="utf-8"
    )

    # LLM Provider
    main_model: str = "claude-sonnet-4-20250514"
    fallback_models: list[str] = Field(default_factory=lambda: ["gpt-4o"])
    cluster_model: str | None = None

    # API Keys
    anthropic_api_key: str | None = None
    openai_api_key: str | None = None
    google_api_key: str | None = None
    groq_api_key: str | None = None
    cerebras_api_key: str | None = None

    # Performance
    max_concurrent_modules: int = Field(default=5, ge=1, le=20)
    retry_attempts: int = Field(default=3, ge=1, le=10)
    retry_base_delay: float = Field(default=1.0, ge=0.1)

    # Token Limits
    max_tokens_per_module: int = 36_369
    max_tokens_per_leaf: int = 16_000

    # Custom context limits for models not in the default map.
    # Format: {"model-name": context_limit_in_tokens}
    # Example: {"my-custom-model": 32000, "local/llama": 8192}
    # These will be merged with MODEL_CONTEXT_LIMITS, with custom values taking precedence.
    custom_context_limits: dict[str, int] = Field(default_factory=dict)

    # Output
    output_dir: str = "./docs"
    log_level: str = "INFO"
    log_file: str | None = None

    @field_validator("main_model")
    @classmethod
    def validate_model(cls, v: str) -> str:
        """
        Validate that the model name starts with a valid prefix.

        Args:
            v: The model name to validate

        Returns:
            The validated model name

        Raises:
            ValueError: If the model name doesn't start with a valid prefix
        """
        valid_prefixes = ("claude", "gpt", "o1", "o3", "gemini", "groq/", "cerebras/")
        if not any(v.startswith(prefix) for prefix in valid_prefixes):
            raise ValueError(
                f"Invalid model name '{v}'. Must start with one of: {', '.join(valid_prefixes)}"
            )
        return v

    @model_validator(mode="after")
    def set_cluster_model_default(self) -> "Settings":
        """
        Set cluster_model to main_model if not explicitly provided.

        This ensures that when no cluster_model is specified, the main_model
        is used for clustering operations.

        Returns:
            The Settings instance with cluster_model defaulted if needed
        """
        if self.cluster_model is None:
            self.cluster_model = self.main_model
        return self

    def get_api_key(self, provider: str) -> str:
        """
        Get the API key for a specific provider.

        Args:
            provider: The provider name (anthropic, openai, google, groq, cerebras)

        Returns:
            The API key for the provider

        Raises:
            ValueError: If the provider name is not supported
            AuthenticationError: If the API key is missing for the provider
        """
        from codewiki.core.errors import AuthenticationError

        api_key_mapping = {
            "anthropic": self.anthropic_api_key,
            "openai": self.openai_api_key,
            "google": self.google_api_key,
            "groq": self.groq_api_key,
            "cerebras": self.cerebras_api_key,
        }

        if provider not in api_key_mapping:
            valid_providers = ", ".join(sorted(api_key_mapping.keys()))
            raise ValueError(
                f"Unknown provider '{provider}'. Supported providers: {valid_providers}"
            )

        api_key = api_key_mapping[provider]
        if api_key is None:
            raise AuthenticationError(
                provider=provider,
                model="",
                message=f"API key for provider '{provider}' is not configured",
            )

        return api_key
