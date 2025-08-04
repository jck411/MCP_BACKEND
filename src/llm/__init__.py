"""
Modern LLM integration with dataclass-based architecture.

This package provides enhanced LLM API integration with:
- Type-safe dataclass models
- Provider-specific optimizations (OpenAI, OpenRouter, Groq)
- Modern streaming with proper error handling
- Rate limiting and cost tracking
- Clean modular architecture
"""

from __future__ import annotations

from .client import ModernLLMClient
from .exceptions import LLMError, ProviderError, RateLimitError, TokenLimitError
from .models import (
    FinishReason,
    LLMChoice,
    LLMMessage,
    LLMRequest,
    LLMResponse,
    MessageRole,
    ProviderConfig,
    ProviderType,
    TokenUsage,
    Tool,
    ToolCall,
    ToolFunction,
)

__all__ = [
    # Core models
    "FinishReason",
    "LLMChoice",
    # Exceptions
    "LLMError",
    "LLMMessage",
    "LLMRequest",
    "LLMResponse",
    "MessageRole",
    # Client
    "ModernLLMClient",
    "ProviderConfig",
    "ProviderError",
    "ProviderType",
    "RateLimitError",
    "TokenLimitError",
    "TokenUsage",
    "Tool",
    "ToolCall",
    "ToolFunction",
]
