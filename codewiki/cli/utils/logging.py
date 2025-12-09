"""
Logging utilities for CLI with colored output and progress tracking.

This module provides a unified logging approach for CLI commands:

- CLILogger: Primary CLI logger with step tracking, elapsed time, and optional
  file logging via CodeWikiLogger integration
- create_logger: Factory for CLILogger instances (existing behavior preserved)
- get_core_logger: Factory for CodeWikiLogger instances for non-CLI contexts

Migration Strategy:
    CLILogger now optionally wraps CodeWikiLogger internally, allowing:
    - Consistent Rich-based console output across CLI and core
    - Structured JSON file logging when configured
    - CLI-specific features (step tracking, elapsed time) preserved
    - Gradual migration path: enable core_logger integration via Settings

    Current phase: CLILogger uses click.secho by default for backward compatibility.
    When a Settings object is provided, it delegates to CodeWikiLogger for
    console output while adding CLI-specific methods on top.
"""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Any, Optional

import click

if TYPE_CHECKING:
    from codewiki.core.config import Settings
    from codewiki.core.logging import CodeWikiLogger


class CLILogger:
    """Logger for CLI with support for verbose mode, step tracking, and elapsed time.

    This logger provides CLI-specific features while optionally integrating with
    CodeWikiLogger for unified logging behavior and file output.

    Operating Modes:
        1. Standalone mode (default): Uses click.secho for colored output.
           Best for simple CLI scripts or when Settings is not available.

        2. Integrated mode (with Settings): Delegates to CodeWikiLogger for
           console output and optional JSON file logging. Provides consistent
           Rich-based output matching core module logging.

    CLI-Specific Features (both modes):
        - step(): Log processing steps with [step/total] prefixes
        - elapsed_time(): Get formatted elapsed time since logger creation
        - Verbose mode: Show/hide debug messages

    Args:
        verbose: Enable verbose output (shows debug messages)
        settings: Optional Settings object to enable CodeWikiLogger integration.
                  When provided, console output uses Rich and file logging is
                  enabled if settings.log_file is configured.

    Example (standalone mode - backward compatible):
        logger = CLILogger(verbose=True)
        logger.step("Processing", 1, 3)
        logger.info("Working...")
        logger.success("Done!")
        print(f"Completed in {logger.elapsed_time()}")

    Example (integrated mode - with file logging):
        from codewiki.core import Settings

        settings = Settings(log_level="DEBUG", log_file="codewiki.log")
        logger = CLILogger(verbose=True, settings=settings)
        logger.step("Processing", 1, 3)
        logger.info("Working...")  # Also written to JSON log file
        logger.success("Done!")
        logger.close()  # Clean up file handler
    """

    def __init__(self, verbose: bool = False, settings: Optional[Settings] = None):
        """
        Initialize the CLI logger.

        Args:
            verbose: Enable verbose output (debug messages visible)
            settings: Optional Settings for CodeWikiLogger integration
        """
        self.verbose = verbose
        self.start_time = datetime.now()

        # Initialize core logger if Settings provided
        self._core_logger: Optional[CodeWikiLogger] = None
        if settings is not None:
            from codewiki.core.logging import get_logger

            self._core_logger = get_logger(settings)

    @property
    def _use_core_logger(self) -> bool:
        """Check if we should delegate to CodeWikiLogger."""
        return self._core_logger is not None

    def debug(self, message: str, **extra: Any) -> None:
        """Log debug message (only in verbose mode).

        Args:
            message: The message to log
            **extra: Additional fields for file logging (ignored in standalone mode)
        """
        if not self.verbose:
            return

        if self._use_core_logger:
            self._core_logger.debug(message, **extra)  # type: ignore[union-attr]
        else:
            timestamp = datetime.now().strftime("%H:%M:%S")
            click.secho(f"[{timestamp}] {message}", fg="cyan", dim=True)

    def info(self, message: str, **extra: Any) -> None:
        """Log info message.

        Args:
            message: The message to log
            **extra: Additional fields for file logging (ignored in standalone mode)
        """
        if self._use_core_logger:
            self._core_logger.info(message, **extra)  # type: ignore[union-attr]
        else:
            click.echo(message)

    def success(self, message: str, **extra: Any) -> None:
        """Log success message in green.

        Args:
            message: The message to log
            **extra: Additional fields for file logging (ignored in standalone mode)
        """
        if self._use_core_logger:
            self._core_logger.success(message, **extra)  # type: ignore[union-attr]
        else:
            click.secho(f"✓ {message}", fg="green")

    def warning(self, message: str, **extra: Any) -> None:
        """Log warning message in yellow.

        Args:
            message: The message to log
            **extra: Additional fields for file logging (ignored in standalone mode)
        """
        if self._use_core_logger:
            self._core_logger.warning(message, **extra)  # type: ignore[union-attr]
        else:
            click.secho(f"⚠️  {message}", fg="yellow")

    def error(self, message: str, **extra: Any) -> None:
        """Log error message in red.

        Args:
            message: The message to log
            **extra: Additional fields for file logging (ignored in standalone mode)
        """
        if self._use_core_logger:
            self._core_logger.error(message, **extra)  # type: ignore[union-attr]
        else:
            click.secho(f"✗ {message}", fg="red", err=True)

    def step(self, message: str, step: Optional[int] = None, total: Optional[int] = None) -> None:
        """Log a processing step with optional progress indicator.

        This is a CLI-specific method not available in CodeWikiLogger.
        Uses click.secho directly for consistent step formatting.

        Args:
            message: Step description
            step: Current step number (e.g., 1)
            total: Total number of steps (e.g., 5)

        Example:
            logger.step("Validating configuration...", 1, 4)
            # Output: [1/4] Validating configuration...

            logger.step("Processing files...")
            # Output: → Processing files...
        """
        if step is not None and total is not None:
            prefix = f"[{step}/{total}]"
        else:
            prefix = "→"

        click.secho(f"{prefix} {message}", fg="blue", bold=True)

        # Also log to file if core logger is available (as INFO level)
        if self._use_core_logger:
            self._core_logger._log_to_file(  # type: ignore[union-attr]
                level=20,  # logging.INFO
                msg=f"{prefix} {message}",
                extra={"step": step, "total": total} if step is not None else {},
            )

    def elapsed_time(self) -> str:
        """Get formatted elapsed time since logger was created.

        This is a CLI-specific method for tracking operation duration.

        Returns:
            Formatted time string (e.g., "45s" or "2m 15s")

        Example:
            logger = CLILogger()
            # ... do work ...
            logger.success(f"Completed in {logger.elapsed_time()}")
        """
        elapsed = datetime.now() - self.start_time
        minutes = int(elapsed.total_seconds() // 60)
        seconds = int(elapsed.total_seconds() % 60)

        if minutes > 0:
            return f"{minutes}m {seconds}s"
        else:
            return f"{seconds}s"

    def close(self) -> None:
        """Close the underlying file handler and release resources.

        Safe to call multiple times. No-op in standalone mode.
        """
        if self._core_logger is not None:
            self._core_logger.close()


def create_logger(verbose: bool = False, settings: Optional[Settings] = None) -> CLILogger:
    """
    Create and return a CLI logger.

    This is the primary factory for CLI logging. Returns a CLILogger configured
    for CLI-specific features (step tracking, elapsed time).

    Args:
        verbose: Enable verbose output (debug messages visible)
        settings: Optional Settings for CodeWikiLogger integration.
                  When provided, enables Rich-based console output and
                  optional JSON file logging.

    Returns:
        Configured CLILogger instance

    Example (backward compatible):
        logger = create_logger(verbose=True)
        logger.step("Processing", 1, 3)
        logger.success("Done!")

    Example (with file logging):
        from codewiki.core import Settings

        settings = Settings(log_level="INFO", log_file="codewiki.log")
        logger = create_logger(verbose=True, settings=settings)
        logger.info("This goes to console AND file")
        logger.close()
    """
    return CLILogger(verbose=verbose, settings=settings)


def get_core_logger(settings: Settings) -> CodeWikiLogger:
    """
    Create a CodeWikiLogger instance from core module.

    Use this for backend and core modules where CLI-specific features
    (step tracking, elapsed time) are not needed. For CLI commands,
    prefer create_logger() with settings parameter instead.

    This factory provides access to the unified logging system with:
    - Rich-based colored console output
    - JSON Lines file logging with rotation
    - Secret sanitization (API keys, tokens)

    Args:
        settings: Settings object with log_level and log_file attributes.
                  Settings can be created from environment variables or
                  constructed directly:

                  # From environment (reads CODEWIKI_* vars)
                  from codewiki.core import Settings
                  settings = Settings()

                  # Direct construction
                  settings = Settings(log_level="DEBUG", log_file="codewiki.log")

    Returns:
        Configured CodeWikiLogger instance

    Example:
        from codewiki.core import Settings
        from codewiki.cli.utils.logging import get_core_logger

        settings = Settings(log_level="INFO", log_file="codewiki.log")
        logger = get_core_logger(settings)

        logger.info("Processing module", module_name="auth")
        logger.success("Completed", tokens=1247, duration=12.3)
        logger.warning("Rate limit approaching")
        logger.error("Failed to process", error="timeout")

        logger.close()  # Clean up file handler
    """
    from codewiki.core.logging import get_logger

    return get_logger(settings)
