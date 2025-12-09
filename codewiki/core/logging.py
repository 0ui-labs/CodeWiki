"""Unified logging system for CodeWiki.

Provides Rich-based console output with colored icons and structured JSON file logging
with log rotation, thread safety, and secret sanitization.

Console Output Example:
    ℹ Processing: auth
    ✓ Completed: auth (1,247 tokens, 12.3s)
    ⚠ Rate limit on database, retrying...
    ✗ Failed to process module

File Output Example (codewiki.log):
    {"timestamp": "2025-12-08T10:23:45.123456+00:00", "level": "INFO", "message": "Processing: auth"}
    {"timestamp": "2025-12-08T10:23:57.456789+00:00", "level": "INFO", "message": "Completed: auth", "tokens": 1247, "duration": 12.3}

Usage:
    from codewiki.core import Settings, get_logger

    settings = Settings(log_level="INFO", log_file="codewiki.log")
    logger = get_logger(settings)

    logger.info("Starting process")
    logger.success("Task completed", tokens=1500, duration=8.2)
    logger.warning("Rate limit approaching")
    logger.error("Operation failed", error_code="E001")
    logger.debug("Detailed info", context={"key": "value"})
"""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import TYPE_CHECKING, Any

from rich.console import Console

if TYPE_CHECKING:
    from codewiki.core.config import Settings

# Pattern for detecting sensitive field names
# Uses word boundaries to avoid false positives (e.g., "tokens" should not match)
_SENSITIVE_KEY_PATTERN = re.compile(
    r"(api[_-]?key|(?<![a-z])token(?![s])|secret|password|credential|auth[_-]?key|access[_-]?key)",
    re.IGNORECASE,
)


def _sanitize(data: Any) -> Any:
    """Recursively sanitize sensitive data in log entries.

    Replaces values of keys matching sensitive patterns (api_key, token, secret, etc.)
    with '***REDACTED***'.

    Args:
        data: The data to sanitize (can be dict, list, or primitive)

    Returns:
        Sanitized copy of the data with sensitive values redacted
    """
    if isinstance(data, dict):
        result = {}
        for key, value in data.items():
            if isinstance(key, str) and _SENSITIVE_KEY_PATTERN.search(key):
                result[key] = "***REDACTED***"
            else:
                result[key] = _sanitize(value)
        return result
    elif isinstance(data, list):
        return [_sanitize(item) for item in data]
    else:
        return data


class StructuredFileHandler(RotatingFileHandler):
    """File handler that writes JSON Lines format with log rotation and thread safety.

    Inherits from RotatingFileHandler to provide:
    - Automatic log rotation (default: 10MB max, 3 backups)
    - Thread-safe writes via inherited locking mechanism
    - JSON Lines format for easy parsing and streaming
    - UTC timestamps for consistency across timezones
    - Secret sanitization for API keys, tokens, etc.

    JSON Structure:
        Core structural fields (always at top level, controlled by handler):
        - timestamp: ISO 8601 UTC timestamp
        - level: Log level name (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        - message: Log message text
        - module: Source module name (if available)
        - logger: Logger name (if available)

        User metadata fields are merged at top level if they don't conflict
        with reserved keys. Any conflicting keys are namespaced under "data".

    Args:
        filepath: Path to the log file
        max_bytes: Maximum file size before rotation (default: 10MB)
        backup_count: Number of backup files to keep (default: 3)
    """

    # Reserved keys that the handler controls - user extra data cannot override these
    # Includes core structural fields and common LogRecord attributes
    _RESERVED_KEYS: frozenset[str] = frozenset({
        # Core structural fields
        "timestamp",
        "level",
        "message",
        "module",
        "logger",
        # LogRecord standard attributes that could leak through
        "name",
        "levelname",
        "levelno",
        "pathname",
        "filename",
        "lineno",
        "funcName",
        "created",
        "msecs",
        "relativeCreated",
        "thread",
        "threadName",
        "process",
        "processName",
        "args",
        "msg",
        "exc_info",
        "exc_text",
        "stack_info",
        # Our custom attribute
        "extra",
    })

    def __init__(
        self,
        filepath: Path | str,
        max_bytes: int = 10 * 1024 * 1024,  # 10MB
        backup_count: int = 3,
    ) -> None:
        filepath = Path(filepath)
        # Ensure parent directory exists
        filepath.parent.mkdir(parents=True, exist_ok=True)

        super().__init__(
            filename=str(filepath),
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding="utf-8",
        )

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON line.

        Overrides the parent's format method to produce JSON output.

        User-provided extra fields are merged at the top level if they don't
        conflict with reserved structural keys. Any conflicting keys are
        collected into a "data" sub-dictionary to preserve the information
        while maintaining structural integrity.

        Args:
            record: The log record to format

        Returns:
            JSON string representation of the log entry
        """
        # Build the log entry with core structural fields
        log_entry: dict[str, Any] = {
            "timestamp": datetime.now(tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "message": record.getMessage(),
        }

        # Add module info if available
        if record.module:
            log_entry["module"] = record.module

        # Add logger name if meaningful (not empty)
        if record.name and record.name != "root":
            log_entry["logger"] = record.name

        # Merge extra fields (if present)
        extra = getattr(record, "extra", {})
        if extra:
            # Sanitize extra data before processing
            sanitized_extra = _sanitize(extra)

            # Separate conflicting keys from safe keys
            conflicting: dict[str, Any] = {}
            for key, value in sanitized_extra.items():
                if key in self._RESERVED_KEYS:
                    # Collect conflicting keys for namespacing
                    conflicting[key] = value
                else:
                    # Safe to merge at top level
                    log_entry[key] = value

            # Namespace conflicting keys under "data" to preserve them
            if conflicting:
                if "data" in log_entry:
                    existing_data = log_entry["data"]
                    if isinstance(existing_data, dict):
                        # Merge conflicting keys into existing dict
                        existing_data.update(conflicting)
                    else:
                        # Preserve non-dict value under "_original" key
                        log_entry["data"] = {"_original": existing_data, **conflicting}
                else:
                    log_entry["data"] = conflicting

        # Sanitize the entire entry (catches any remaining sensitive data)
        log_entry = _sanitize(log_entry)

        return json.dumps(log_entry)

    def emit(self, record: logging.LogRecord) -> None:
        """Emit a log record as a JSON line.

        Thread safety is provided by the parent class's acquire()/release() locking.

        Args:
            record: The log record to emit
        """
        try:
            msg = self.format(record)
            # Use parent's stream handling with proper locking
            self.acquire()
            try:
                if self.shouldRollover(record):
                    self.doRollover()
                if self.stream:
                    self.stream.write(msg + self.terminator)
                    self.flush()
            finally:
                self.release()
        except Exception:
            # Silent fail to avoid breaking application
            self.handleError(record)


class CodeWikiLogger:
    """Unified logger with Rich console output and structured JSON file logging.

    Provides semantic log methods with colored icons for console output:
    - info(): Blue ℹ icon
    - success(): Green ✓ icon (INFO level)
    - warning(): Yellow ⚠ icon
    - error(): Red ✗ icon
    - debug(): Dim 🔍 icon (only shown at DEBUG level)

    Log Level Behavior:
        Both console and file outputs share a single log level threshold from
        ``settings.log_level``. Messages below this threshold are suppressed
        for both sinks. This simplified model ensures consistent behavior
        across outputs.

        For example, with ``log_level="WARNING"``:
        - debug(), info(), success() messages are suppressed everywhere
        - warning(), error() messages appear in both console and file

        If separate thresholds per sink are needed in the future, the Settings
        class can be extended with ``console_log_level`` and ``file_log_level``.

    Args:
        settings: Settings object with the following attributes:
            - log_level (str): Minimum log level threshold for both console and
              file output. One of: "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL".
              Defaults to "INFO" if not specified or invalid.
            - log_file (str | None): Path to JSON log file. If None or not set,
              file logging is disabled and only console output is produced.

    Example:
        from codewiki.core import Settings, get_logger

        settings = Settings(log_level="DEBUG", log_file="app.log")
        logger = get_logger(settings)

        logger.info("Processing module", module_name="auth")
        logger.success("Completed", tokens=1247, duration=12.3)
        logger.warning("Rate limit approaching")
        logger.error("Failed to process", error="timeout")
        logger.debug("Detailed debug info", context={"key": "value"})

        logger.close()  # Clean up file handler
    """

    # Log level mapping
    _LEVEL_MAP = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
        "CRITICAL": logging.CRITICAL,
    }

    def __init__(self, settings: Settings) -> None:
        self.console = Console()

        # Parse log level with fallback to INFO
        level_name = getattr(settings, "log_level", "INFO").upper()
        self._level = self._LEVEL_MAP.get(level_name, logging.INFO)

        # Initialize file handler if log_file is configured
        log_file = getattr(settings, "log_file", None)
        if log_file:
            self.file_handler: StructuredFileHandler | None = StructuredFileHandler(Path(log_file))
            self.file_handler.setLevel(self._level)
        else:
            self.file_handler = None

    def _should_log(self, level: int) -> bool:
        """Check if a message at the given level should be logged.

        Uses a single threshold for both console and file output.
        See class docstring for details on log level behavior.

        Args:
            level: Python logging level (e.g., logging.INFO, logging.DEBUG)

        Returns:
            True if the message should be logged to both console and file
        """
        return level >= self._level

    def _log_to_file(self, level: int, msg: str, extra: dict[str, Any]) -> None:
        """Write a log entry to the file handler.

        Args:
            level: Python logging level (e.g., logging.INFO)
            msg: Log message
            extra: Additional fields to include in the log entry
        """
        if self.file_handler is None:
            return

        record = logging.LogRecord(
            name="codewiki",
            level=level,
            pathname="",
            lineno=0,
            msg=msg,
            args=(),
            exc_info=None,
        )
        record.extra = extra
        self.file_handler.emit(record)

    def debug(self, msg: str, **extra: Any) -> None:
        """Log a debug message with dim 🔍 icon.

        Only shown when log level is DEBUG.

        Args:
            msg: The message to log
            **extra: Additional fields to include in file log
        """
        if self._should_log(logging.DEBUG):
            self.console.print(f"[dim]🔍 {msg}[/dim]")
            self._log_to_file(logging.DEBUG, msg, extra)

    def info(self, msg: str, **extra: Any) -> None:
        """Log an info message with blue ℹ icon.

        Args:
            msg: The message to log
            **extra: Additional fields to include in file log
        """
        if self._should_log(logging.INFO):
            self.console.print(f"[blue]ℹ[/blue] {msg}")
            self._log_to_file(logging.INFO, msg, extra)

    def success(self, msg: str, **extra: Any) -> None:
        """Log a success message with green ✓ icon.

        Success is a semantic variant of INFO level.

        Args:
            msg: The message to log
            **extra: Additional fields to include in file log
        """
        if self._should_log(logging.INFO):
            self.console.print(f"[green]✓[/green] {msg}")
            self._log_to_file(logging.INFO, msg, extra)

    def warning(self, msg: str, **extra: Any) -> None:
        """Log a warning message with yellow ⚠ icon.

        Args:
            msg: The message to log
            **extra: Additional fields to include in file log
        """
        if self._should_log(logging.WARNING):
            self.console.print(f"[yellow]⚠[/yellow] {msg}")
            self._log_to_file(logging.WARNING, msg, extra)

    def error(self, msg: str, **extra: Any) -> None:
        """Log an error message with red ✗ icon.

        Args:
            msg: The message to log
            **extra: Additional fields to include in file log
        """
        if self._should_log(logging.ERROR):
            self.console.print(f"[red]✗[/red] {msg}")
            self._log_to_file(logging.ERROR, msg, extra)

    def close(self) -> None:
        """Close the file handler and release resources.

        Safe to call multiple times.
        """
        if self.file_handler is not None:
            try:
                self.file_handler.close()
            except Exception:
                pass  # Ignore errors during cleanup


def get_logger(settings: Settings) -> CodeWikiLogger:
    """Create a CodeWikiLogger instance.

    Factory function that creates a logger configured from settings.

    Args:
        settings: Settings object with log_level and log_file attributes

    Returns:
        Configured CodeWikiLogger instance

    Example:
        from codewiki.core import Settings, get_logger

        settings = Settings(log_level="INFO", log_file="codewiki.log")
        logger = get_logger(settings)
        logger.info("Hello, world!")
    """
    return CodeWikiLogger(settings)
