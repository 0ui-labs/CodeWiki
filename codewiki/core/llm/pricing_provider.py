"""LiteLLM pricing provider with automatic fetching and caching.

This module fetches LLM pricing data from LiteLLM's community-maintained
pricing database and caches it locally for offline use.

Usage:
    from codewiki.core.llm.pricing_provider import PricingProvider

    provider = PricingProvider()
    pricing = provider.get_pricing("claude-sonnet-4-20250514")
    if pricing:
        print(f"Input: ${pricing.input_per_million}/M tokens")
"""

from __future__ import annotations

import json
import urllib.request
import urllib.error
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypedDict

from codewiki.core.llm.costs import ModelPricing, PRICING as FALLBACK_PRICING


class CacheData(TypedDict):
    """Type definition for cache file structure."""

    fetched_at: str
    source: str
    models: dict[str, ModelPricing]

if TYPE_CHECKING:
    from codewiki.core.logging import CodeWikiLogger

# LiteLLM pricing data URL (raw GitHub content)
LITELLM_PRICING_URL = (
    "https://raw.githubusercontent.com/BerriAI/litellm/"
    "main/model_prices_and_context_window.json"
)

# Fetch timeout in seconds
FETCH_TIMEOUT_SECONDS = 10

# Default cache TTL in days
DEFAULT_TTL_DAYS = 7

# Cache filename
CACHE_FILENAME = "pricing_cache.json"


class PricingProvider:
    """Provider for LLM pricing data with automatic fetching and caching.

    Fetches pricing data from LiteLLM's GitHub repository and caches it
    locally. Falls back to cached data (even if expired) or built-in
    pricing data if fetching fails.

    Attributes:
        cache_dir: Directory for cache file storage
        ttl_days: Cache time-to-live in days
    """

    def __init__(
        self,
        cache_dir: Path | None = None,
        ttl_days: int = DEFAULT_TTL_DAYS,
        logger: CodeWikiLogger | None = None,
    ) -> None:
        """Initialize the pricing provider.

        Args:
            cache_dir: Directory for cache storage. Defaults to ~/.codewiki
            ttl_days: Cache TTL in days. Defaults to 7.
            logger: Optional logger for warnings and debug messages.
        """
        self._cache_dir = cache_dir or Path.home() / ".codewiki"
        self._ttl_days = ttl_days
        self._logger = logger
        self._pricing: dict[str, ModelPricing] | None = None
        self._initialized = False

    @property
    def _cache_path(self) -> Path:
        """Path to the cache file."""
        return self._cache_dir / CACHE_FILENAME

    def _ensure_initialized(self) -> None:
        """Ensure pricing data is loaded (lazy initialization)."""
        if self._initialized:
            return

        self._initialized = True
        self._pricing = self._load_pricing()

    def _load_pricing(self) -> dict[str, ModelPricing]:
        """Load pricing data using the fallback chain.

        Returns:
            Dictionary mapping model names to ModelPricing instances.
        """
        # Check if cache exists and is fresh
        cache_data = self._read_cache()
        if cache_data is not None:
            age_days = self._get_cache_age_days(cache_data)
            if age_days is not None and age_days < self._ttl_days:
                # Cache is fresh, use it
                if self._logger:
                    self._logger.debug(
                        f"Using cached pricing data ({age_days:.1f} days old)",
                        cache_age_days=age_days,
                    )
                return cache_data["models"]

        # Cache is missing or expired, try to fetch
        fetched = self._fetch_and_cache()
        if fetched is not None:
            return fetched

        # Fetch failed, try to use expired cache
        if cache_data is not None:
            age_days = self._get_cache_age_days(cache_data)
            if self._logger:
                self._logger.warning(
                    f"Pricing cache expired ({age_days:.1f} days old), using stale data",
                    cache_age_days=age_days,
                )
            return cache_data["models"]

        # No cache available, use built-in fallback
        if self._logger:
            self._logger.warning(
                "Could not fetch pricing data and no cache available, using built-in fallback"
            )
        return dict(FALLBACK_PRICING)

    def _read_cache(self) -> CacheData | None:
        """Read and parse the cache file.

        Returns:
            Parsed cache data or None if cache doesn't exist or is invalid.
        """
        if not self._cache_path.exists():
            return None

        try:
            with open(self._cache_path, "r", encoding="utf-8") as f:
                raw_data: Any = json.load(f)

            # Validate structure
            if "fetched_at" not in raw_data or "models" not in raw_data:
                return None

            # Convert models dict back to ModelPricing instances
            models: dict[str, ModelPricing] = {}
            for model, pricing in raw_data["models"].items():
                models[model] = ModelPricing(
                    input_per_million=pricing["input_per_million"],
                    output_per_million=pricing["output_per_million"],
                )

            return CacheData(
                fetched_at=raw_data["fetched_at"],
                source=raw_data.get("source", LITELLM_PRICING_URL),
                models=models,
            )

        except (json.JSONDecodeError, KeyError, TypeError, OSError):
            return None

    def _get_cache_age_days(self, cache_data: CacheData) -> float | None:
        """Calculate the age of cache data in days.

        Args:
            cache_data: Parsed cache data with 'fetched_at' field.

        Returns:
            Age in days or None if timestamp is invalid.
        """
        try:
            fetched_at = datetime.fromisoformat(cache_data["fetched_at"])
            now = datetime.now(timezone.utc)
            age = now - fetched_at
            return age.total_seconds() / (24 * 60 * 60)
        except (KeyError, ValueError):
            return None

    def _fetch_and_cache(self) -> dict[str, ModelPricing] | None:
        """Fetch pricing data from LiteLLM and update cache.

        Returns:
            Dictionary mapping model names to ModelPricing, or None if fetch failed.
        """
        try:
            # Fetch from GitHub
            request = urllib.request.Request(
                LITELLM_PRICING_URL,
                headers={"User-Agent": "CodeWiki/1.0"},
            )
            with urllib.request.urlopen(request, timeout=FETCH_TIMEOUT_SECONDS) as response:
                raw_data = json.loads(response.read().decode("utf-8"))

            # Parse LiteLLM format
            models = self._parse_litellm_pricing(raw_data)

            if not models:
                if self._logger:
                    self._logger.warning("Fetched pricing data contained no valid models")
                return None

            # Write cache
            self._write_cache(models)

            if self._logger:
                self._logger.debug(
                    f"Fetched pricing for {len(models)} models from LiteLLM",
                    model_count=len(models),
                )

            return models

        except (urllib.error.URLError, urllib.error.HTTPError, json.JSONDecodeError, OSError) as e:
            if self._logger:
                self._logger.warning(
                    f"Failed to fetch pricing data: {e}",
                    error=str(e),
                )
            return None

    def _parse_litellm_pricing(self, raw: dict) -> dict[str, ModelPricing]:
        """Parse LiteLLM pricing format into ModelPricing instances.

        LiteLLM stores costs per token, we convert to per million tokens.

        Args:
            raw: Raw JSON data from LiteLLM.

        Returns:
            Dictionary mapping model names to ModelPricing instances.
        """
        result = {}
        for model, data in raw.items():
            # Skip non-dict entries (metadata fields)
            if not isinstance(data, dict):
                continue

            input_cost = data.get("input_cost_per_token")
            output_cost = data.get("output_cost_per_token")

            # Skip models without pricing data
            if input_cost is None or output_cost is None:
                continue

            # Skip invalid values
            try:
                input_cost = float(input_cost)
                output_cost = float(output_cost)
            except (TypeError, ValueError):
                continue

            # Convert from per-token to per-million
            result[model] = ModelPricing(
                input_per_million=input_cost * 1_000_000,
                output_per_million=output_cost * 1_000_000,
            )

        return result

    def _write_cache(self, models: dict[str, ModelPricing]) -> None:
        """Write pricing data to cache file.

        Args:
            models: Dictionary mapping model names to ModelPricing instances.
        """
        try:
            # Ensure cache directory exists
            self._cache_dir.mkdir(parents=True, exist_ok=True)

            # Convert ModelPricing to serializable dict
            models_dict = {
                model: {
                    "input_per_million": pricing.input_per_million,
                    "output_per_million": pricing.output_per_million,
                }
                for model, pricing in models.items()
            }

            cache_data = {
                "fetched_at": datetime.now(timezone.utc).isoformat(),
                "source": LITELLM_PRICING_URL,
                "models": models_dict,
            }

            with open(self._cache_path, "w", encoding="utf-8") as f:
                json.dump(cache_data, f, indent=2)

        except OSError as e:
            if self._logger:
                self._logger.warning(
                    f"Failed to write pricing cache: {e}",
                    error=str(e),
                )

    def get_pricing(self, model: str) -> ModelPricing | None:
        """Get pricing for a specific model.

        Args:
            model: Model identifier (e.g., 'claude-sonnet-4-20250514')

        Returns:
            ModelPricing instance or None if model is unknown.
        """
        self._ensure_initialized()
        if self._pricing is None:
            return None
        return self._pricing.get(model)

    def refresh(self, force: bool = False) -> bool:
        """Refresh pricing data from LiteLLM.

        Args:
            force: If True, fetch even if cache is fresh.

        Returns:
            True if fetch succeeded, False otherwise.
        """
        if not force:
            # Check if cache is still fresh
            cache_data = self._read_cache()
            if cache_data is not None:
                age_days = self._get_cache_age_days(cache_data)
                if age_days is not None and age_days < self._ttl_days:
                    return True  # Cache is fresh, no need to refresh

        fetched = self._fetch_and_cache()
        if fetched is not None:
            self._pricing = fetched
            return True
        return False

    @property
    def model_count(self) -> int:
        """Number of models with pricing data."""
        self._ensure_initialized()
        return len(self._pricing) if self._pricing else 0
