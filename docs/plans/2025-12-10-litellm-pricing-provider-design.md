# LiteLLM Pricing Provider Design

## Übersicht

Automatisches Fetchen und Cachen von LLM-Preisdaten aus LiteLLMs Community-gepflegter Preisdatenbank.

## Entscheidungen

| Aspekt | Entscheidung |
|--------|--------------|
| Fetch-Zeitpunkt | Beim Start (LLMClient init) |
| Caching | Datei-Cache mit 7 Tagen TTL |
| Fallback | Stille Fallback-Kette, kein Abbruch |

## Architektur

```
┌─────────────────────────────────────────────────────────┐
│                     LLMClient                           │
│                         │                               │
│                         ▼                               │
│               PricingProvider.get_pricing()             │
│                         │                               │
│         ┌───────────────┼───────────────┐               │
│         ▼               ▼               ▼               │
│   ┌──────────┐   ┌──────────┐   ┌──────────────┐       │
│   │  Cache   │   │  Fetch   │   │  Fallback    │       │
│   │ (< 7d)   │   │ LiteLLM  │   │  costs.py    │       │
│   └──────────┘   └──────────┘   └──────────────┘       │
└─────────────────────────────────────────────────────────┘
```

## Datenstrukturen

### Cache-Datei (`~/.codewiki/pricing_cache.json`)

```json
{
  "fetched_at": "2025-12-10T14:30:00Z",
  "source": "https://raw.githubusercontent.com/BerriAI/litellm/main/model_prices_and_context_window.json",
  "models": {
    "claude-sonnet-4-20250514": {
      "input_per_million": 3.0,
      "output_per_million": 15.0
    }
  }
}
```

### PricingProvider API

```python
class PricingProvider:
    def __init__(
        self,
        cache_dir: Path = Path.home() / ".codewiki",
        ttl_days: int = 7,
        logger: CodeWikiLogger | None = None,
    ): ...

    def get_pricing(self, model: str) -> ModelPricing | None:
        """Hole Pricing für ein Modell. None wenn unbekannt."""

    def refresh(self, force: bool = False) -> bool:
        """Aktualisiere Cache. Returns True bei Erfolg."""
```

## Fetch & Parsing

### LiteLLM Quell-URL

```
https://raw.githubusercontent.com/BerriAI/litellm/main/model_prices_and_context_window.json
```

### Format-Transformation

LiteLLM speichert Preise pro Token, wir brauchen pro Million:

```python
def _parse_litellm_pricing(raw: dict) -> dict[str, ModelPricing]:
    result = {}
    for model, data in raw.items():
        input_cost = data.get("input_cost_per_token")
        output_cost = data.get("output_cost_per_token")

        if input_cost is None or output_cost is None:
            continue

        result[model] = ModelPricing(
            input_per_million=input_cost * 1_000_000,
            output_per_million=output_cost * 1_000_000,
        )
    return result
```

## Fallback-Kette

Priorität bei `get_pricing()`:

1. **Frischer Cache** (< 7 Tage) → Nutzen, kein Fetch
2. **Erfolgreicher Fetch** → Nutzen, Cache aktualisieren
3. **Abgelaufener Cache** → Nutzen + Warning loggen
4. **Kein Cache, Fetch fehlschlägt** → `costs.py` PRICING + Warning

Keine Exceptions nach außen - Dokumentationsgenerierung läuft immer weiter.

## Logging

```python
# Abgelaufener Cache:
logger.warning(f"Pricing cache expired ({age_days} days old), using stale data")

# Fallback auf costs.py:
logger.warning("Could not fetch pricing data, using built-in fallback")

# Erfolgreicher Fetch (debug):
logger.debug(f"Fetched pricing for {len(models)} models from LiteLLM")
```

## Dateiänderungen

### Neue Dateien

- `codewiki/core/llm/pricing_provider.py` - Hauptlogik

### Geänderte Dateien

- `codewiki/core/llm/costs.py` - Erweiterte `calculate_cost()` Signatur
- `codewiki/core/llm/client.py` - Provider-Instanz erstellen
- `codewiki/core/llm/__init__.py` - Export `PricingProvider`

## Keine neuen Dependencies

Nutzt nur `urllib.request` (stdlib) für HTTP-Requests.
