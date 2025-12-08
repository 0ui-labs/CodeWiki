# CodeWiki Refactoring Design

**Datum:** 2025-12-08
**Ziel:** Robustheit, Performance, exzellentes Error-Handling
**Scope:** CLI + Web-App

---

## Entscheidungen

| Bereich | Entscheidung |
|---------|--------------|
| **Provider** | OpenAI, Anthropic, Gemini, Groq, Cerebras – direkt ohne Abstraktion |
| **Error Handling** | Retry + Fallback + User-Notification |
| **Token Counting** | Native Tokenizer pro Provider |
| **Async** | Vollständig async + parallele Modul-Verarbeitung |
| **Config** | Pydantic Settings zentral, CLI + Keyring Support |
| **Logging** | Rich Console + Structured JSON File |

---

## Neue Dateistruktur

```
codewiki/
├── core/                        # NEU
│   ├── __init__.py
│   ├── config.py                # Pydantic Settings
│   ├── errors.py                # Exception Hierarchy
│   ├── logging.py               # Rich + File Logger
│   ├── llm/
│   │   ├── __init__.py
│   │   ├── client.py            # LLMClient (direkt)
│   │   ├── retry.py             # ResilientLLMClient
│   │   └── tokenizers.py        # TokenCounter
│   └── async_utils.py           # ParallelModuleProcessor
├── cli/                         # Angepasst
│   ├── config_loader.py         # NEU: Keyring + JSON laden
│   └── ...                      # Rest bleibt
├── src/be/                      # Angepasst
│   ├── agent_orchestrator.py    # Nutzt core/, async parallel
│   ├── llm_services.py          # ENTFERNT (ersetzt durch core/llm/)
│   └── ...                      # Rest bleibt
```

---

## Komponenten-Design

### 1. LLM Client (Direkter Ansatz)

Ein `LLMClient` ohne Vererbungs-Hierarchie. Provider-Erkennung via Model-Name.

```python
# core/llm/client.py
from anthropic import AsyncAnthropic
from openai import AsyncOpenAI
from google import genai

@dataclass
class LLMResponse:
    content: str
    input_tokens: int
    output_tokens: int
    model: str
    provider: str

class LLMClient:
    def __init__(self, settings: Settings):
        self.settings = settings
        # Clients lazy initialisiert wenn gebraucht
        self._anthropic: AsyncAnthropic | None = None
        self._openai: AsyncOpenAI | None = None
        self._groq: AsyncOpenAI | None = None
        self._cerebras: AsyncOpenAI | None = None
        self._google: genai.Client | None = None

    async def complete(
        self,
        messages: list[dict],
        model: str,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        **kwargs  # Provider-spezifische Optionen
    ) -> LLMResponse:
        provider = self._detect_provider(model)

        match provider:
            case "anthropic":
                return await self._call_anthropic(messages, model, temperature, max_tokens, **kwargs)
            case "openai":
                return await self._call_openai(messages, model, temperature, max_tokens, **kwargs)
            case "google":
                return await self._call_google(messages, model, temperature, max_tokens, **kwargs)
            case "groq":
                return await self._call_groq(messages, model, temperature, max_tokens, **kwargs)
            case "cerebras":
                return await self._call_cerebras(messages, model, temperature, max_tokens, **kwargs)
            case _:
                raise ValueError(f"Unknown provider for model: {model}")

    def _detect_provider(self, model: str) -> str:
        if model.startswith("claude"):
            return "anthropic"
        elif model.startswith(("gpt", "o1", "o3")):
            return "openai"
        elif model.startswith("gemini"):
            return "google"
        elif "groq/" in model:
            return "groq"
        elif "cerebras/" in model:
            return "cerebras"
        raise ValueError(f"Cannot detect provider for: {model}")
```

**Vorteile:**
- ~200 Zeilen statt ~400 mit Abstraktion
- Provider-spezifische Features via `**kwargs` durchreichbar
- Lazy Client-Initialisierung (nur was gebraucht wird)
- Explizit und debugbar

---

### 2. Error Handling & Exceptions

```python
# core/errors.py
class CodeWikiError(Exception):
    """Base exception for all CodeWiki errors."""
    pass

class LLMError(CodeWikiError):
    """Base for LLM-related errors."""
    provider: str
    model: str

class RateLimitError(LLMError):
    """Provider rate limit hit."""
    retry_after: float | None = None

class ContextLengthError(LLMError):
    """Input exceeded model's context window."""
    max_tokens: int
    actual_tokens: int

class ProviderUnavailableError(LLMError):
    """Provider API is down or unreachable."""
    pass

class AuthenticationError(LLMError):
    """Invalid API key or auth failure."""
    pass
```

---

### 3. Retry & Fallback Logic

```python
# core/llm/retry.py
@dataclass
class RetryConfig:
    max_retries: int = 3
    base_delay: float = 1.0  # Exponential backoff
    fallback_models: list[str] = field(default_factory=list)

class ResilientLLMClient:
    def __init__(self, client: LLMClient, config: RetryConfig, logger: Logger):
        self.client = client
        self.config = config
        self.logger = logger

    async def complete(self, messages, model, **kwargs) -> LLMResponse:
        models_to_try = [model] + self.config.fallback_models
        last_error = None

        for current_model in models_to_try:
            for attempt in range(self.config.max_retries):
                try:
                    response = await self.client.complete(messages, current_model, **kwargs)

                    # User-Notification bei Fallback
                    if current_model != model:
                        self.logger.warning(f"Used fallback model: {current_model}")

                    return response

                except RateLimitError as e:
                    delay = e.retry_after or (self.config.base_delay * 2 ** attempt)
                    self.logger.warning(f"Rate limit, retry in {delay}s...")
                    await asyncio.sleep(delay)

                except ContextLengthError:
                    raise  # Nicht retry-bar, sofort hochwerfen

                except (ProviderUnavailableError, Exception) as e:
                    last_error = e
                    self.logger.warning(f"Attempt {attempt+1} failed: {e}")
                    await asyncio.sleep(self.config.base_delay * 2 ** attempt)

            # Alle Retries für dieses Model fehlgeschlagen → nächstes Model
            self.logger.warning(f"Model {current_model} exhausted, trying fallback...")

        raise LLMError(f"All models failed. Last error: {last_error}")
```

**Verhalten:**
1. Versuche Main-Model (z.B. `claude-sonnet-4`) mit Retries
2. Bei dauerhaftem Fehler → Fallback-Model (z.B. `gpt-4o`)
3. User wird über Console informiert wenn Fallback genutzt wird
4. `ContextLengthError` wird sofort geworfen (nicht retry-bar)

---

### 4. Token Counting per Provider

```python
# core/llm/tokenizers.py
import tiktoken
import anthropic

class TokenCounter:
    def __init__(self):
        self._tiktoken_enc: tiktoken.Encoding | None = None
        self._anthropic_client: anthropic.Anthropic | None = None

    def count(self, text: str, model: str) -> int:
        """Count tokens for text with model-appropriate tokenizer."""
        provider = self._detect_provider(model)

        match provider:
            case "anthropic":
                return self._count_anthropic(text, model)
            case "google":
                return self._count_google(text, model)
            case "openai" | "groq" | "cerebras":
                return self._count_tiktoken(text, model)
            case _:
                return int(self._count_tiktoken(text, "gpt-4") * 1.1)

    def _count_anthropic(self, text: str, model: str) -> int:
        if not self._anthropic_client:
            self._anthropic_client = anthropic.Anthropic()
        return self._anthropic_client.count_tokens(text)

    def _count_google(self, text: str, model: str) -> int:
        from google import genai
        client = genai.Client()
        response = client.models.count_tokens(model=model, contents=text)
        return response.total_tokens

    def _count_tiktoken(self, text: str, model: str) -> int:
        if not self._tiktoken_enc:
            self._tiktoken_enc = tiktoken.get_encoding("cl100k_base")
        return len(self._tiktoken_enc.encode(text))
```

**Context-Limit Check vor API-Call:**

```python
MODEL_CONTEXT_LIMITS = {
    "claude-sonnet-4-20250514": 200_000,
    "claude-3-5-sonnet": 200_000,
    "gpt-4o": 128_000,
    "gemini-1.5-pro": 1_000_000,
}

async def complete(self, messages, model, **kwargs):
    total_tokens = self.token_counter.count(self._messages_to_text(messages), model)
    limit = MODEL_CONTEXT_LIMITS.get(model, 100_000)

    if total_tokens > limit * 0.95:
        raise ContextLengthError(
            f"Input ({total_tokens} tokens) exceeds model limit ({limit})",
            max_tokens=limit,
            actual_tokens=total_tokens
        )

    return await self._call_provider(messages, model, **kwargs)
```

---

### 5. Async & Parallelisierung

```python
# core/async_utils.py
import asyncio

class ParallelModuleProcessor:
    def __init__(self, max_concurrency: int = 5):
        self.semaphore = asyncio.Semaphore(max_concurrency)
        self.completed: dict[str, Any] = {}
        self.logger = get_logger()

    async def process_modules(
        self,
        modules: list[Module],
        dependency_graph: dict[str, list[str]],
        process_fn: Callable[[Module], Awaitable[Any]]
    ) -> dict[str, Any]:
        """Process modules respecting dependencies, parallelizing where possible."""

        pending = set(m.name for m in modules)
        module_map = {m.name: m for m in modules}

        async def process_when_ready(module_name: str):
            deps = dependency_graph.get(module_name, [])
            while not all(d in self.completed for d in deps):
                await asyncio.sleep(0.1)

            async with self.semaphore:
                self.logger.info(f"Processing: {module_name}")
                result = await process_fn(module_map[module_name])
                self.completed[module_name] = result
                self.logger.success(f"Completed: {module_name}")

        tasks = [process_when_ready(m.name) for m in modules]
        await asyncio.gather(*tasks)

        return self.completed
```

**Performance-Gewinn:**

Bei Dependency-Tree mit 5 Modulen auf 3 Ebenen:
- Sequentiell: 5 × 30s = **150s**
- Parallel: 3 Ebenen × 30s = **90s** (40% schneller)

---

### 6. Konfiguration (Pydantic Settings)

```python
# core/config.py
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field, field_validator

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="CODEWIKI_",
        env_file=".env",
        env_file_encoding="utf-8",
    )

    # LLM Provider
    main_model: str = "claude-sonnet-4-20250514"
    fallback_models: list[str] = Field(default=["gpt-4o"])

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
        valid_prefixes = ("claude", "gpt", "o1", "o3", "gemini", "groq/", "cerebras/")
        if not v.startswith(valid_prefixes):
            raise ValueError(f"Unknown model: {v}")
        return v

    def get_api_key(self, provider: str) -> str:
        key_map = {
            "anthropic": self.anthropic_api_key,
            "openai": self.openai_api_key,
            "google": self.google_api_key,
            "groq": self.groq_api_key,
            "cerebras": self.cerebras_api_key,
        }
        key = key_map.get(provider)
        if not key:
            raise AuthenticationError(f"No API key configured for {provider}")
        return key
```

**CLI-Integration mit Keyring:**

```python
# cli/config_loader.py
import keyring

def load_settings_for_cli() -> Settings:
    settings_dict = {}

    # ~/.codewiki/config.json
    config_file = Path.home() / ".codewiki" / "config.json"
    if config_file.exists():
        with open(config_file) as f:
            settings_dict.update(json.load(f))

    # API Keys aus Keyring
    for provider in ["anthropic", "openai", "google", "groq", "cerebras"]:
        key = keyring.get_password("codewiki", f"{provider}_api_key")
        if key:
            settings_dict[f"{provider}_api_key"] = key

    return Settings(**settings_dict)
```

---

### 7. Logging (Rich + Structured File)

```python
# core/logging.py
import logging
import json
from datetime import datetime
from pathlib import Path
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn

console = Console()

class StructuredFileHandler(logging.Handler):
    def __init__(self, filepath: Path):
        super().__init__()
        self.filepath = filepath
        self.filepath.parent.mkdir(parents=True, exist_ok=True)

    def emit(self, record: logging.LogRecord):
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "message": record.getMessage(),
            "module": record.module,
            **getattr(record, "extra", {})
        }
        with open(self.filepath, "a") as f:
            f.write(json.dumps(log_entry) + "\n")


class CodeWikiLogger:
    def __init__(self, settings: Settings):
        self.console = console
        self.file_handler = None

        if settings.log_file:
            self.file_handler = StructuredFileHandler(Path(settings.log_file))

    def info(self, msg: str, **extra):
        self.console.print(f"[blue]ℹ[/blue] {msg}")
        self._log_to_file("INFO", msg, extra)

    def success(self, msg: str, **extra):
        self.console.print(f"[green]✓[/green] {msg}")
        self._log_to_file("INFO", msg, extra)

    def warning(self, msg: str, **extra):
        self.console.print(f"[yellow]⚠[/yellow] {msg}")
        self._log_to_file("WARNING", msg, extra)

    def error(self, msg: str, **extra):
        self.console.print(f"[red]✗[/red] {msg}")
        self._log_to_file("ERROR", msg, extra)

    def _log_to_file(self, level: str, msg: str, extra: dict):
        if self.file_handler:
            record = logging.LogRecord(
                name="codewiki", level=getattr(logging, level),
                pathname="", lineno=0, msg=msg, args=(), exc_info=None
            )
            record.extra = extra
            self.file_handler.emit(record)
```

**Output-Beispiel:**

```
Console:
ℹ Processing: auth
✓ Completed: auth (1,247 tokens, 12.3s)
⚠ Rate limit on database, retrying...
✓ Completed: database (892 tokens, 18.1s)

File (codewiki.log):
{"timestamp": "2025-12-08T10:23:45", "level": "INFO", "message": "Completed: auth", "tokens": 1247, "duration": 12.3, "provider": "anthropic"}
```

---

## Migrations-Reihenfolge

Inkrementelle Migration, jede Phase einzeln testbar:

| Phase | Komponente | Beschreibung |
|-------|------------|--------------|
| 1 | `core/config.py`, `core/errors.py` | Grundstruktur + Settings + Exceptions |
| 2 | `core/llm/client.py`, `core/llm/tokenizers.py` | LLM Client + Token Counting |
| 3 | `core/llm/retry.py` | ResilientLLMClient mit Fallback |
| 4 | `core/logging.py` | Rich + Structured Logging |
| 5 | `core/async_utils.py` | ParallelModuleProcessor |
| 6 | `src/be/agent_orchestrator.py` | Refactor auf neue core/ Module |
| 7 | Cleanup | Alte `llm_services.py` entfernen, Tests anpassen |

---

## Bekannte Einschränkungen (aktueller Code)

Diese werden durch das Refactoring behoben:

- **OpenAI-only Client:** Erfordert aktuell Proxy für Anthropic
- **tiktoken für alle Modelle:** Ungenaue Token-Counts für Claude/Gemini
- **Sync `call_llm()`:** Blockiert Web-App
- **Verstreute Config:** Env-Vars in mehreren Dateien
- **Ungültiger Default-Modellname:** `claude-sonnet-4` existiert nicht
