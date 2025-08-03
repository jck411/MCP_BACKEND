"""
Centralized logging and error handling utilities for MCP Platform.

This module provides decorators and helper functions to standardize logging
and error handling patterns across the codebase, reducing boilerplate and
ensuring consistent error reporting.

Features:
- Structured logging with contextual information
- MCP-specific error handling decorators
- Automatic error type detection and classification
- Performance timing and metrics
- Context-aware error messages
"""

from __future__ import annotations

import functools
import time
from collections.abc import Awaitable, Callable
from contextlib import asynccontextmanager
from typing import Any, ParamSpec, TypeVar

import structlog
from mcp import McpError, types
from pydantic import ValidationError

# Configure structured logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.dev.ConsoleRenderer(colors=True),
    ],
    wrapper_class=structlog.stdlib.BoundLogger,
    logger_factory=structlog.stdlib.LoggerFactory(),
    cache_logger_on_first_use=True,
)

# Type variables for generic decorators
P = ParamSpec("P")
T = TypeVar("T")
AsyncCallable = Callable[P, Awaitable[T]]

logger = structlog.get_logger(__name__)


class MCPErrorHandler:
    """Centralized MCP error handling with structured logging."""

    @staticmethod
    def classify_error(error: Exception) -> tuple[int, str]:
        """
        Classify an error and return appropriate MCP error code and category.

        Args:
            error: The exception to classify

        Returns:
            Tuple of (mcp_error_code, error_category)
        """
        if isinstance(error, McpError):
            return error.error.code, "mcp_error"
        if isinstance(error, ValidationError):
            return types.INVALID_PARAMS, "validation_error"
        if isinstance(error, TimeoutError):
            return types.INTERNAL_ERROR, "timeout_error"
        if isinstance(error, ConnectionError | OSError):
            return types.INTERNAL_ERROR, "connection_error"
        if isinstance(error, ValueError | TypeError):
            return types.INVALID_PARAMS, "parameter_error"
        return types.INTERNAL_ERROR, "unknown_error"

    @staticmethod
    def create_mcp_error(
        error: Exception,
        operation: str,
        context: dict[str, Any] | None = None,
        custom_message: str | None = None,
    ) -> McpError:
        """
        Create a standardized McpError with structured logging.

        Args:
            error: Original exception
            operation: Description of the operation that failed
            context: Additional context for logging and error data
            custom_message: Override the default error message

        Returns:
            McpError with structured error data
        """
        error_code, error_category = MCPErrorHandler.classify_error(error)
        context = context or {}

        # Create structured error message
        if custom_message:
            message = custom_message
        elif isinstance(error, McpError):
            message = error.error.message
        else:
            message = f"{operation} failed: {error!s}"

        # Log the error with context
        logger.error(
            "Operation failed",
            operation=operation,
            error_type=type(error).__name__,
            error_category=error_category,
            error_code=error_code,
            error_message=str(error),
            **context,
        )

        # Create error data with context
        error_data = {
            "operation": operation,
            "error_category": error_category,
            "original_error_type": type(error).__name__,
            **context,
        }

        return McpError(
            error=types.ErrorData(
                code=error_code,
                message=message,
                data=error_data,
            )
        )


def log_operation(
    operation: str,
    *,
    log_args: bool = False,
    log_result: bool = False,
    log_timing: bool = True,
    context: dict[str, Any] | None = None,
) -> Callable[[AsyncCallable[P, T]], AsyncCallable[P, T]]:
    """
    Decorator for logging async operations with structured context.

    Args:
        operation: Description of the operation being performed
        log_args: Whether to log function arguments
        log_result: Whether to log function result
        log_timing: Whether to log execution timing
        context: Additional context to include in logs

    Returns:
        Decorated function with logging
    """
    def decorator(func: AsyncCallable[P, T]) -> AsyncCallable[P, T]:
        @functools.wraps(func)
        async def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            operation_logger = logger.bind(
                operation=operation,
                function=func.__name__,
                **(context or {}),
            )

            # Log function start
            log_data = {}
            if log_args:
                log_data.update({
                    "args": args[1:] if args else [],  # Skip 'self' if present
                    "kwargs": kwargs,
                })

            operation_logger.info("Operation started", **log_data)

            start_time = time.perf_counter() if log_timing else None

            try:
                result = await func(*args, **kwargs)

                # Log success
                end_log_data: dict[str, Any] = {}
                if log_timing and start_time is not None:
                    duration = round((time.perf_counter() - start_time) * 1000, 2)
                    end_log_data["duration_ms"] = duration
                if log_result:
                    end_log_data["result"] = result

                operation_logger.info(
                    "Operation completed successfully", **end_log_data
                )
                return result

            except Exception as e:
                # Log failure with timing
                error_log_data: dict[str, Any] = {
                    "error_type": type(e).__name__,
                    "error_message": str(e),
                }
                if log_timing and start_time is not None:
                    duration = round((time.perf_counter() - start_time) * 1000, 2)
                    error_log_data["duration_ms"] = duration

                operation_logger.error("Operation failed", **error_log_data)
                raise

        return wrapper
    return decorator


def handle_mcp_errors(
    operation: str,
    *,
    context: dict[str, Any] | None = None,
    custom_message: str | None = None,
    reraise_mcp_errors: bool = True,
) -> Callable[[AsyncCallable[P, T]], AsyncCallable[P, T]]:
    """
    Decorator for standardized MCP error handling.

    Args:
        operation: Description of the operation for error context
        context: Additional context to include in error data
        custom_message: Custom error message template
        reraise_mcp_errors: Whether to re-raise McpError instances as-is

    Returns:
        Decorated function with MCP error handling
    """
    def decorator(func: AsyncCallable[P, T]) -> AsyncCallable[P, T]:
        @functools.wraps(func)
        async def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            try:
                return await func(*args, **kwargs)
            except McpError as e:
                if reraise_mcp_errors:
                    # Re-log MCP errors with operation context
                    logger.error(
                        "MCP error in operation",
                        operation=operation,
                        mcp_error_code=e.error.code,
                        mcp_error_message=e.error.message,
                        **(context or {}),
                    )
                    raise
                # Wrap MCP error with operation context
                raise MCPErrorHandler.create_mcp_error(
                    e, operation, context, custom_message
                ) from e
            except Exception as e:
                # Convert non-MCP errors to MCP errors
                raise MCPErrorHandler.create_mcp_error(
                    e, operation, context, custom_message
                ) from e

        return wrapper
    return decorator


@asynccontextmanager
async def operation_context(
    operation: str,
    *,
    context: dict[str, Any] | None = None,
    log_timing: bool = True,
):
    """
    Async context manager for operation logging and error handling.

    Args:
        operation: Description of the operation
        context: Additional context for logging
        log_timing: Whether to log operation timing

    Yields:
        Bound logger for the operation
    """
    operation_logger = logger.bind(
        operation=operation,
        **(context or {}),
    )

    operation_logger.info("Operation started")
    start_time = time.perf_counter() if log_timing else None

    try:
        yield operation_logger

        # Log success
        log_data: dict[str, Any] = {}
        if log_timing and start_time is not None:
            duration = round((time.perf_counter() - start_time) * 1000, 2)
            log_data["duration_ms"] = duration

        operation_logger.info("Operation completed successfully", **log_data)

    except Exception as e:
        # Log failure
        error_log_data: dict[str, Any] = {
            "error_type": type(e).__name__,
            "error_message": str(e),
        }
        if log_timing and start_time is not None:
            duration = round((time.perf_counter() - start_time) * 1000, 2)
            error_log_data["duration_ms"] = duration

        operation_logger.error("Operation failed", **error_log_data)
        raise


class ContextualLogger:
    """Logger that maintains context across related operations."""

    def __init__(self, base_context: dict[str, Any] | None = None):
        self.base_context = base_context or {}
        self._logger = logger.bind(**self.base_context)

    def bind(self, **context: Any) -> ContextualLogger:
        """Create a new logger with additional context."""
        merged_context = {**self.base_context, **context}
        return ContextualLogger(merged_context)

    def info(self, message: str, **context: Any) -> None:
        """Log info message with context."""
        self._logger.info(message, **context)

    def warning(self, message: str, **context: Any) -> None:
        """Log warning message with context."""
        self._logger.warning(message, **context)

    def error(self, message: str, **context: Any) -> None:
        """Log error message with context."""
        self._logger.error(message, **context)

    def debug(self, message: str, **context: Any) -> None:
        """Log debug message with context."""
        self._logger.debug(message, **context)


# Convenience functions for common patterns
def log_mcp_operation(operation: str, **kwargs) -> Callable:
    """Combined logging and MCP error handling decorator."""
    # Split kwargs between the two decorators
    log_kwargs = {
        k: v for k, v in kwargs.items()
        if k in ["log_args", "log_result", "log_timing", "context"]
    }
    error_kwargs = {
        k: v for k, v in kwargs.items()
        if k in ["context", "custom_message", "reraise_mcp_errors"]
    }

    def decorator(func):
        return handle_mcp_errors(operation, **error_kwargs)(
            log_operation(operation, **log_kwargs)(func)
        )
    return decorator


async def safe_mcp_call[T](
    operation: str,
    coro: Awaitable[T],
    *,
    context: dict[str, Any] | None = None,
    custom_message: str | None = None,
) -> T:
    """
    Safely execute an MCP operation with standardized error handling.

    Args:
        operation: Description of the operation
        coro: Coroutine to execute
        context: Additional context for logging
        custom_message: Custom error message

    Returns:
        Result of the coroutine

    Raises:
        McpError: Standardized MCP error with context
    """
    try:
        async with operation_context(operation, context=context):
            return await coro
    except Exception as e:
        raise MCPErrorHandler.create_mcp_error(
            e, operation, context, custom_message
        ) from e
