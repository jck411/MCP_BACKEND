"""
Modern error handling for LLM operations.

This module provides comprehensive error handling with rich context:
- Provider-specific error information
- Retry guidance for rate limits
- Token limit detection
- Circuit breaker support
"""

from __future__ import annotations


class LLMError(Exception):
    """Base LLM error with rich context."""

    def __init__(
        self,
        message: str,
        provider: str,
        model: str,
        status_code: int | None = None,
        response_data: dict | None = None,
    ):
        super().__init__(message)
        self.provider = provider
        self.model = model
        self.status_code = status_code
        self.response_data = response_data or {}


class RateLimitError(LLMError):
    """Rate limit error with retry information."""

    def __init__(
        self,
        message: str,
        retry_after: float | None = None,
        **kwargs,
    ):
        super().__init__(message, **kwargs)
        self.retry_after = retry_after


class TokenLimitError(LLMError):
    """Token limit exceeded error."""
    pass


class CircuitBreakerError(LLMError):
    """Circuit breaker is open."""
    pass


class StreamingError(LLMError):
    """Streaming-specific errors."""

    def __init__(
        self,
        message: str,
        provider: str = "unknown",
        model: str = "unknown",
        **kwargs,
    ):
        super().__init__(message, provider, model, **kwargs)


class ProviderError(LLMError):
    """Provider-specific configuration or setup errors."""
    pass
