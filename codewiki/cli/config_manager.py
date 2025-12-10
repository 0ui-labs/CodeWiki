"""
Configuration manager with keyring integration for secure credential storage.
"""

import json
from pathlib import Path
from typing import Any, Optional, TYPE_CHECKING
import keyring
from keyring.errors import KeyringError

from codewiki.cli.models.config import Configuration
from codewiki.cli.utils.errors import ConfigurationError, FileSystemError
from codewiki.cli.utils.fs import ensure_directory, safe_write, safe_read
from codewiki.core.llm.utils import detect_provider

if TYPE_CHECKING:
    from codewiki.core import Settings


# Keyring configuration
KEYRING_SERVICE = "codewiki"
KEYRING_API_KEY_ACCOUNT = "api_key"

# Configuration file location
CONFIG_DIR = Path.home() / ".codewiki"
CONFIG_FILE = CONFIG_DIR / "config.json"
CONFIG_VERSION = "1.0"


class ConfigManager:
    """
    Manages CodeWiki configuration with secure keyring storage for API keys.

    Storage:
        - API key: System keychain via keyring (macOS Keychain, Windows Credential Manager,
                  Linux Secret Service)
        - Other settings: ~/.codewiki/config.json
    """

    def __init__(self):
        """Initialize the configuration manager."""
        self._api_key: Optional[str] = None
        self._config: Optional[Configuration] = None
        self._keyring_available = self._check_keyring_available()

    def _check_keyring_available(self) -> bool:
        """Check if system keyring is available."""
        try:
            # Try to get/set a test value
            keyring.get_password(KEYRING_SERVICE, "__test__")
            return True
        except KeyringError:
            return False

    def load(self) -> bool:
        """
        Load configuration from file and keyring.

        Returns:
            True if configuration exists, False otherwise
        """
        # Load from JSON file
        if not CONFIG_FILE.exists():
            return False

        try:
            content = safe_read(CONFIG_FILE)
            data = json.loads(content)

            # Validate version
            if data.get("version") != CONFIG_VERSION:
                # Could implement migration here
                pass

            self._config = Configuration.from_dict(data)

            # Load API key from keyring
            try:
                self._api_key = keyring.get_password(KEYRING_SERVICE, KEYRING_API_KEY_ACCOUNT)
            except KeyringError:
                # Keyring unavailable, API key will be None
                pass

            return True
        except (json.JSONDecodeError, FileSystemError) as e:
            raise ConfigurationError(f"Failed to load configuration: {e}")

    def save(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        main_model: Optional[str] = None,
        cluster_model: Optional[str] = None,
        default_output: Optional[str] = None,
    ):
        """
        Save configuration to file and keyring.

        Args:
            api_key: API key (stored in keyring)
            base_url: LLM API base URL
            main_model: Primary model
            cluster_model: Clustering model
            default_output: Default output directory
        """
        # Ensure config directory exists
        try:
            ensure_directory(CONFIG_DIR)
        except FileSystemError as e:
            raise ConfigurationError(f"Cannot create config directory: {e}")

        # Load existing config or create new
        if self._config is None:
            if CONFIG_FILE.exists():
                self.load()
            else:
                self._config = Configuration(
                    base_url="", main_model="", cluster_model="", default_output="docs"
                )

        # Update fields if provided
        if base_url is not None:
            self._config.base_url = base_url
        if main_model is not None:
            self._config.main_model = main_model
        if cluster_model is not None:
            self._config.cluster_model = cluster_model
        if default_output is not None:
            self._config.default_output = default_output

        # Validate configuration
        self._config.validate()

        # Save API key to keyring
        if api_key is not None:
            self._api_key = api_key
            try:
                keyring.set_password(KEYRING_SERVICE, KEYRING_API_KEY_ACCOUNT, api_key)
            except KeyringError as e:
                # Fallback: warn about keyring unavailability
                raise ConfigurationError(
                    f"System keychain unavailable: {e}\n"
                    f"Please ensure your system keychain is properly configured."
                )

        # Save non-sensitive config to JSON
        config_data = {"version": CONFIG_VERSION, **self._config.to_dict()}

        try:
            safe_write(CONFIG_FILE, json.dumps(config_data, indent=2))
        except FileSystemError as e:
            raise ConfigurationError(f"Failed to save configuration: {e}")

    def get_api_key(self) -> Optional[str]:
        """
        Get API key from keyring.

        Returns:
            API key or None if not set
        """
        if self._api_key is None:
            try:
                self._api_key = keyring.get_password(KEYRING_SERVICE, KEYRING_API_KEY_ACCOUNT)
            except KeyringError:
                pass

        return self._api_key

    def get_config(self) -> Optional[Configuration]:
        """
        Get current configuration.

        Returns:
            Configuration object or None if not loaded
        """
        return self._config

    def is_configured(self) -> bool:
        """
        Check if configuration is complete and valid.

        Returns:
            True if configured, False otherwise
        """
        if self._config is None:
            return False

        # Check if API key is set
        if self.get_api_key() is None:
            return False

        # Check if config is complete
        return self._config.is_complete()

    def delete_api_key(self):
        """Delete API key from keyring."""
        try:
            keyring.delete_password(KEYRING_SERVICE, KEYRING_API_KEY_ACCOUNT)
            self._api_key = None
        except KeyringError:
            pass

    def clear(self):
        """Clear all configuration (file and keyring)."""
        # Delete API key from keyring
        self.delete_api_key()

        # Delete config file
        if CONFIG_FILE.exists():
            CONFIG_FILE.unlink()

        self._config = None
        self._api_key = None

    @property
    def keyring_available(self) -> bool:
        """Check if keyring is available."""
        return self._keyring_available

    @property
    def config_file_path(self) -> Path:
        """Get configuration file path."""
        return CONFIG_FILE

    def load_settings_for_backend(self, repo_path: str, output_dir: str) -> "Settings":
        """
        Load CLI configuration and convert to core.Settings for backend.

        Args:
            repo_path: Repository path (for context, not stored in Settings)
            output_dir: Output directory for docs

        Returns:
            Settings instance ready for backend use

        Raises:
            ConfigurationError: If configuration incomplete or invalid
        """
        from codewiki.core import Settings

        if not self.is_configured():
            raise ConfigurationError(
                "Configuration incomplete. Run 'codewiki config validate' to check."
            )

        config = self.get_config()
        api_key = self.get_api_key()

        # After is_configured() check, config and api_key must be non-None
        assert config is not None, "Config must be set after is_configured()"
        assert api_key is not None, "API key must be set after is_configured()"

        # Map CLI config to Settings fields
        # Detect provider from base_url or model name
        provider = self._detect_provider(config.main_model, config.base_url)

        settings_dict: dict[str, Any] = {
            "main_model": config.main_model,
            "cluster_model": config.cluster_model or config.main_model,
            "output_dir": output_dir,
            "log_level": "INFO",  # CLI default
            "repo_path": repo_path,
        }

        # Set API key for detected provider
        if provider == "anthropic":
            settings_dict["anthropic_api_key"] = api_key
        elif provider == "openai":
            settings_dict["openai_api_key"] = api_key
        elif provider == "google":
            settings_dict["google_api_key"] = api_key
        elif provider == "groq":
            settings_dict["groq_api_key"] = api_key
        elif provider == "cerebras":
            settings_dict["cerebras_api_key"] = api_key

        return Settings(**settings_dict)

    def _detect_provider(self, model: str, base_url: str) -> str:
        """Detect provider from model name or base URL.

        Uses the centralized detect_provider function from codewiki.core.llm.utils.
        Falls back to 'openai' for unknown providers (e.g., custom base URLs).
        """
        provider = detect_provider(model, base_url if base_url else None)
        # Default to OpenAI for unknown providers (backward compatibility for custom base URLs)
        return provider if provider != "unknown" else "openai"
