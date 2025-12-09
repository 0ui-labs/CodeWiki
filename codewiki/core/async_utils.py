"""Async utilities for parallel module processing with dependency management.

Provides ParallelModuleProcessor for concurrent processing of modules while
respecting dependency relationships and controlling concurrency with semaphores.

Uses Python 3.12 asyncio best practices:
- TaskGroup for structured concurrency and clean exception handling
- Explicit failure tracking to prevent deadlocks
- Semaphore acquisition only after dependencies are satisfied

Example:
    from codewiki.core import get_logger, Settings
    from codewiki.core.async_utils import ParallelModuleProcessor

    settings = Settings(log_level="INFO")
    logger = get_logger(settings)
    processor = ParallelModuleProcessor(max_concurrency=5, logger=logger)

    async def process_module(module: dict) -> str:
        # Process the module...
        return f"Processed {module['name']}"

    results = await processor.process_modules(
        modules=[{"name": "auth"}, {"name": "db"}],
        dependency_graph={"db": ["auth"]},  # db depends on auth
        process_fn=process_module,
    )
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Any, Awaitable, Callable, Protocol

from codewiki.core.errors import (
    CircularDependencyError,
    DependencyFailedError,
    InvalidDependencyError,
)

if TYPE_CHECKING:
    pass


class LoggerProtocol(Protocol):
    """Protocol for duck-typed logger compatibility.

    Defines the minimal interface required for logging in ParallelModuleProcessor.
    Both CodeWikiLogger and _StdlibLoggerAdapter implement this protocol.
    """

    def info(self, msg: str, **kwargs: Any) -> None:
        """Log an info message."""
        ...

    def success(self, msg: str, **kwargs: Any) -> None:
        """Log a success message (semantic variant of info)."""
        ...

    def warning(self, msg: str, **kwargs: Any) -> None:
        """Log a warning message."""
        ...

    def error(self, msg: str, **kwargs: Any) -> None:
        """Log an error message."""
        ...


class _StdlibLoggerAdapter:
    """Adapter to add success() method to stdlib logger.

    Wraps extra kwargs in a nested dict to avoid LogRecord attribute conflicts.
    """

    def __init__(self, logger: logging.Logger) -> None:
        self._logger = logger

    def _safe_extra(self, kwargs: dict[str, Any]) -> dict[str, Any] | None:
        """Wrap extra kwargs to avoid LogRecord attribute conflicts."""
        if not kwargs:
            return None
        # Namespace user kwargs under 'extra_data' to avoid conflicts
        # with LogRecord's built-in attributes like 'module'
        return {"extra_data": kwargs}

    def info(self, msg: str, **kwargs: Any) -> None:
        self._logger.info(msg, extra=self._safe_extra(kwargs))

    def success(self, msg: str, **kwargs: Any) -> None:
        # stdlib has no success level, use info
        self._logger.info(msg, extra=self._safe_extra(kwargs))

    def warning(self, msg: str, **kwargs: Any) -> None:
        self._logger.warning(msg, extra=self._safe_extra(kwargs))

    def error(self, msg: str, **kwargs: Any) -> None:
        self._logger.error(msg, extra=self._safe_extra(kwargs))


class ParallelModuleProcessor:
    """Process modules in parallel while respecting dependencies.

    Uses asyncio.TaskGroup for structured concurrency with clean exception
    handling and cancellation. Tracks both completed and failed modules to
    prevent deadlocks when dependencies fail.

    Features:
    - Semaphore-based concurrency control (configurable max_concurrency)
    - Dependency-aware scheduling (waits for dependencies before processing)
    - Deadlock prevention (fails fast when dependencies fail)
    - Structured logging with configurable logger

    Attributes:
        max_concurrency: Maximum number of modules processed simultaneously
        logger: Logger instance for progress and error logging

    Example:
        processor = ParallelModuleProcessor(max_concurrency=5)

        async def process_module(module: dict) -> str:
            # Your processing logic here
            return f"Processed {module['name']}"

        results = await processor.process_modules(
            modules=[{"name": "auth"}, {"name": "db"}],
            dependency_graph={"db": ["auth"]},  # db depends on auth
            process_fn=process_module,
        )
    """

    def __init__(
        self,
        max_concurrency: int = 5,
        logger: LoggerProtocol | None = None,
    ) -> None:
        """Initialize the parallel module processor.

        Args:
            max_concurrency: Maximum number of modules to process in parallel.
                Default is 5 to avoid API rate limits and memory pressure.
            logger: Optional logger instance conforming to LoggerProtocol.
                Must have info(), success(), warning(), and error() methods.
                If None, uses stdlib logger wrapped with _StdlibLoggerAdapter.
        """
        self.max_concurrency = max_concurrency
        self._semaphore: asyncio.Semaphore | None = None

        # Set up logger - use provided logger or create default adapter
        if logger is None:
            stdlib_logger = logging.getLogger(__name__)
            self.logger: LoggerProtocol = _StdlibLoggerAdapter(stdlib_logger)
        else:
            self.logger = logger

        # Tracking state (reset on each process_modules call)
        self._completed: dict[str, Any] = {}
        self._failed: set[str] = set()

    def _validate_dependencies(
        self,
        module_names: set[str],
        dependency_graph: dict[str, list[str]],
    ) -> None:
        """Validate the dependency graph before processing.

        Checks for:
        1. Unknown dependencies: modules referencing dependencies that don't exist
        2. Circular dependencies: cycles that would cause infinite loops

        Args:
            module_names: Set of all valid module names
            dependency_graph: Maps module name to list of dependency names

        Raises:
            InvalidDependencyError: If a module depends on an unknown module
            CircularDependencyError: If a circular dependency is detected
        """
        # Check for unknown dependencies
        for module, deps in dependency_graph.items():
            for dep in deps:
                if dep not in module_names:
                    raise InvalidDependencyError(module=module, unknown_dependency=dep)

        # Check for circular dependencies using DFS with three-color marking
        # WHITE (not visited), GRAY (in current path), BLACK (fully processed)
        WHITE, GRAY, BLACK = 0, 1, 2
        color: dict[str, int] = {name: WHITE for name in module_names}
        path: list[str] = []

        def dfs(node: str) -> None:
            """Depth-first search to detect cycles."""
            if color[node] == BLACK:
                return
            if color[node] == GRAY:
                # Found a cycle - extract the cycle from the path
                cycle_start = path.index(node)
                cycle = path[cycle_start:] + [node]
                raise CircularDependencyError(cycle=cycle)

            color[node] = GRAY
            path.append(node)

            for dep in dependency_graph.get(node, []):
                if dep in module_names:  # Only check valid dependencies
                    dfs(dep)

            path.pop()
            color[node] = BLACK

        # Run DFS from all nodes to catch disconnected cycles
        for module in module_names:
            if color[module] == WHITE:
                dfs(module)

    async def process_modules(
        self,
        modules: list[dict[str, Any]],
        dependency_graph: dict[str, list[str]],
        process_fn: Callable[[dict[str, Any]], Awaitable[Any]],
    ) -> dict[str, Any]:
        """Process modules in parallel while respecting dependencies.

        Modules are processed as soon as their dependencies are complete.
        Uses TaskGroup for structured concurrency with automatic cancellation
        when any task fails.

        Args:
            modules: List of module dicts. Each must have a 'name' key.
            dependency_graph: Maps module name to list of dependency names.
                Modules not in the graph are treated as having no dependencies.
            process_fn: Async function that processes a single module and
                returns the result.

        Returns:
            Dict mapping module names to their processing results.

        Raises:
            InvalidDependencyError: If a module depends on an unknown module.
            CircularDependencyError: If a circular dependency is detected.
            ExceptionGroup: Contains exceptions from failed modules. When a
                module fails, dependent modules may raise either
                DependencyFailedError (if they detect the failure before
                cancellation) or CancelledError (if TaskGroup cancels them
                first). Callers should handle the ExceptionGroup without
                relying on a specific exception type for each module.

        Example:
            processor = ParallelModuleProcessor(max_concurrency=5)

            async def process_module(module: dict) -> str:
                return f"Processed {module['name']}"

            results = await processor.process_modules(
                modules=[{"name": "auth"}, {"name": "db"}],
                dependency_graph={"db": ["auth"]},
                process_fn=process_module,
            )
            # results = {"auth": "Processed auth", "db": "Processed db"}
        """
        if not modules:
            return {}

        # Reset state for this processing run
        self._completed = {}
        self._failed = set()
        self._semaphore = asyncio.Semaphore(self.max_concurrency)

        # Validate modules and build lookup map
        module_map: dict[str, dict[str, Any]] = {}
        for idx, module in enumerate(modules):
            if not isinstance(module, dict):
                raise ValueError(
                    f"Module at index {idx} is not a dict: {type(module).__name__}"
                )
            if "name" not in module:
                raise ValueError(
                    f"Module at index {idx} missing required 'name' key: {module}"
                )
            name = module["name"]
            if not isinstance(name, str) or not name.strip():
                raise ValueError(
                    f"Module at index {idx} has invalid 'name' (must be non-empty string): {name!r}"
                )
            if name in module_map:
                raise ValueError(
                    f"Duplicate module name '{name}' at index {idx}"
                )
            module_map[name] = module

        # Validate dependency graph before processing
        # This catches unknown dependencies and cycles early, preventing infinite loops
        self._validate_dependencies(set(module_map.keys()), dependency_graph)

        async def process_when_ready(module_name: str) -> None:
            """Wait for dependencies, then process the module."""
            deps = dependency_graph.get(module_name, [])

            # Log if waiting for dependencies
            if deps:
                self.logger.info(f"Waiting for dependencies: {deps}", module=module_name)

            # Wait for all dependencies to complete or fail
            # Poll every 50ms (balance between responsiveness and CPU usage)
            while True:
                # Check if any dependency has failed
                failed_dep = next((d for d in deps if d in self._failed), None)
                if failed_dep:
                    self.logger.warning(
                        f"Skipping due to failed dependency: {failed_dep}",
                        module=module_name,
                    )
                    self._failed.add(module_name)
                    raise DependencyFailedError(
                        module=module_name,
                        failed_dependency=failed_dep,
                    )

                # Check if all dependencies are complete
                if all(d in self._completed for d in deps):
                    break

                await asyncio.sleep(0.05)

            # Acquire semaphore AFTER dependencies are satisfied
            # This ensures waiting modules don't block execution slots
            assert self._semaphore is not None  # Set in process_modules
            async with self._semaphore:
                self.logger.info(f"Processing: {module_name}", module=module_name)
                try:
                    result = await process_fn(module_map[module_name])
                    self._completed[module_name] = result
                    self.logger.success(f"Completed: {module_name}", module=module_name)
                except Exception as e:
                    self._failed.add(module_name)
                    self.logger.error(
                        f"Failed: {module_name} - {e}",
                        module=module_name,
                        error=str(e),
                    )
                    raise

        # Use TaskGroup for structured concurrency (Python 3.11+)
        # This provides clean exception handling and automatic cancellation
        async with asyncio.TaskGroup() as tg:
            for module in modules:
                tg.create_task(process_when_ready(module["name"]))

        return self._completed
