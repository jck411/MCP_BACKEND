"""
Streaming-specific dataclasses for Phase 2 implementation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from ..models import ToolCall


class StreamChunkType(Enum):
    """Types of streaming chunks."""
    CONTENT = "content"
    TOOL_CALLS = "tool_calls"
    COMPLETION = "completion"
    ERROR = "error"


class SSEEventType(Enum):
    """Server-Sent Event types."""
    CHUNK = "chunk"
    COMPLETION = "completion"
    ERROR = "error"
    HEARTBEAT = "heartbeat"


@dataclass(frozen=True)
class RawSSEChunk:
    """Raw SSE chunk from HTTP response."""
    event_type: SSEEventType
    data: dict[str, Any] | None
    raw_data: str
    error: str | None = None
    timestamp: float = field(default_factory=lambda: __import__('time').time())


@dataclass(frozen=True)
class StreamChunk:
    """Processed streaming chunk with accumulated state."""
    chunk_type: StreamChunkType
    content: str | None
    accumulated_content: str
    tool_calls: list[ToolCall]
    finish_reason: str | None = None
    error: str | None = None
    timestamp: float = field(default_factory=lambda: __import__('time').time())

    # Metadata for advanced features
    delta_tokens: int = 0
    cumulative_tokens: int = 0
    processing_latency: float = 0.0


@dataclass
class AccumulatorState:
    """Mutable state for chunk accumulation with enhanced tracking."""
    content_buffer: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    total_tokens: int = 0
    chunk_count: int = 0
    first_chunk_time: float | None = None
    last_chunk_time: float | None = None

    def update_timing(self, timestamp: float) -> None:
        """Update timing information for latency tracking."""
        if self.first_chunk_time is None:
            self.first_chunk_time = timestamp
        self.last_chunk_time = timestamp
        self.chunk_count += 1

    @property
    def streaming_duration(self) -> float:
        """Calculate total streaming duration."""
        if self.first_chunk_time is None or self.last_chunk_time is None:
            return 0.0
        return self.last_chunk_time - self.first_chunk_time


@dataclass(frozen=True)
class StreamingStats:
    """Statistics for streaming performance analysis."""
    total_chunks: int
    content_chunks: int
    tool_call_chunks: int
    error_chunks: int
    total_duration: float
    average_chunk_latency: float
    tokens_per_second: float
    first_token_latency: float
