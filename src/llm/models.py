"""
Core LLM dataclasses with comprehensive type safety and validation.

This module provides the foundational dataclasses for LLM interactions:
- Provider configurations
- Message structures
- Request/response models
- Tool calling support
- Token usage tracking
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Literal


class ProviderType(Enum):
    """Supported LLM providers."""
    OPENAI = "openai"
    OPENROUTER = "openrouter"
    GROQ = "groq"


class MessageRole(Enum):
    """OpenAI-compatible message roles."""
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class FinishReason(Enum):
    """OpenAI-compatible finish reasons."""
    STOP = "stop"
    LENGTH = "length"
    TOOL_CALLS = "tool_calls"
    CONTENT_FILTER = "content_filter"


@dataclass(frozen=True)
class ToolFunction:
    """Tool function definition."""
    name: str
    arguments: str


@dataclass(frozen=True)
class ToolCall:
    """OpenAI-compatible tool call structure."""
    id: str
    type: Literal["function"] = "function"
    function: ToolFunction = field(default_factory=lambda: ToolFunction("", ""))


@dataclass(frozen=True)
class LLMMessage:
    """OpenAI-compatible message structure."""
    role: MessageRole
    content: str | list[dict[str, Any]] | None = None
    name: str | None = None
    tool_calls: list[ToolCall] | None = None
    tool_call_id: str | None = None


@dataclass(frozen=True)
class Tool:
    """OpenAI-compatible tool definition."""
    type: Literal["function"] = "function"
    function: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TokenUsage:
    """Token usage statistics."""
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


@dataclass(frozen=True)
class LLMChoice:
    """OpenAI-compatible choice structure."""
    index: int
    message: LLMMessage
    finish_reason: FinishReason


@dataclass(frozen=True)
class LLMResponse:
    """Complete LLM response structure."""
    id: str
    object: str
    created: int
    model: str
    choices: list[LLMChoice]
    usage: TokenUsage
    system_fingerprint: str | None = None

    # Cost tracking (added by our system)
    estimated_cost: float | None = None
    actual_cost: float | None = None


@dataclass
class LLMRequest:
    """Complete LLM request structure."""
    model: str
    messages: list[LLMMessage]
    temperature: float = 0.7
    max_tokens: int = 4096
    top_p: float = 1.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    tools: list[Tool] | None = None
    tool_choice: str | dict | None = None
    stream: bool = False

    # Provider-specific fields
    response_format: dict[str, str] | None = None
    seed: int | None = None
    stop: list[str] | None = None

    # OpenRouter-specific
    route: str | None = None
    models: list[str] | None = None
    transforms: list[str] | None = None
    provider: dict[str, Any] | None = None


@dataclass(frozen=True)
class ProviderConfig:
    """Provider configuration."""
    provider: ProviderType
    base_url: str
    model: str
    api_key: str

    # Connection settings
    max_connections: int = 100
    max_keepalive: int = 20
    keepalive_expiry: float = 5.0
    connect_timeout: float = 10.0
    read_timeout: float = 60.0
    write_timeout: float = 10.0
    pool_timeout: float = 10.0

    # Rate limiting
    requests_per_minute: int = 3500
    tokens_per_minute: int = 200000

    # Cost tracking
    enable_cost_tracking: bool = True
    cost_warning_threshold: float = 1.0
    daily_cost_limit: float = 50.0

    # Provider-specific settings
    app_name: str | None = None
    app_url: str | None = None
    enable_fallbacks: bool = False
    fallback_models: list[str] = field(default_factory=list)
