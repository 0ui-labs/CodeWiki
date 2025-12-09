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

    Args:
        filepath: Path to the log file
        max_bytes: Maximum file size before rotation (default: 10MB)
        backup_count: Number of backup files to keep (default: 3)
    """

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

        Args:
            record: The log record to format

        Returns:
            JSON string representation of the log entry
        """
        # Build the log entry
        log_entry: dict[str, Any] = {
            "timestamp": datetime.now(tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "message": record.getMessage(),
        }

        # Add module info if available
        if record.module:
            log_entry["module"] = record.module

        # Merge extra fields (if present)
        extra = getattr(record, "extra", {})
        if extra:
            # Sanitize extra data before adding
            sanitized_extra = _sanitize(extra)
            # Extra fields should not override core fields
            for key, value in sanitized_extra.items():
                if key not in ("timestamp", "level", "message"):
                    log_entry[key] = value

        # Sanitize the entire entry
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
    - success(): Green ✓ icon
    - warning(): Yellow ⚠ icon
    - error(): Red ✗ icon
    - debug(): Dim 🔍 icon (only shown at DEBUG level)

    Args:
        settings: Settings object with log_level and log_file attributes.
                  Uses duck typing - any object with these attributes works.

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
        """Check if a message at the given level should be logged."""
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
