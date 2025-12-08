"""Configuration management using Pydantic Settings."""

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables with CODEWIKI_ prefix."""

    model_config = SettingsConfigDict(
        env_prefix="CODEWIKI_", env_file=".env", env_file_encoding="utf-8"
    )

    # LLM Provider
    main_model: str = "claude-sonnet-4-20250514"
    fallback_models: list[str] = ["gpt-4o"]
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

    def get_api_key(self, provider: str) -> str:
        """
        Get the API key for a specific provider.

        Args:
            provider: The provider name (anthropic, openai, google, groq, cerebras)

        Returns:
            The API key for the provider

        Raises:
            AuthenticationError: If the API key is missing for the provider
            KeyError: If the provider name is invalid
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
            raise KeyError(f"Invalid provider '{provider}'")

        api_key = api_key_mapping[provider]
        if api_key is None:
            raise AuthenticationError(
                provider=provider,
                model="",
                message=f"API key for provider '{provider}' is not configured",
            )

        return api_key
