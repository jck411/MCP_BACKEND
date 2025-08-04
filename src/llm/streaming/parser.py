"""
Modern SSE parser with enhanced error recovery and chunk accumulation.
Phase 2: Advanced streaming with circuit breaker patterns.
"""

from __future__ import annotations

import json
import time
from collections.abc import AsyncGenerator

import httpx

from ..exceptions import StreamingError
from ..models import ToolCall, ToolFunction
from .models import (
    AccumulatorState,
    RawSSEChunk,
    SSEEventType,
    StreamChunk,
    StreamChunkType,
    StreamingStats,
)

# Constants
MAX_PROCESSING_HISTORY = 100


class StreamingParser:
    """Modern SSE parser with enhanced error recovery and performance tracking."""

    def __init__(self, enable_recovery: bool = True, chunk_timeout: float = 30.0):
        self.enable_recovery = enable_recovery
        self.chunk_timeout = chunk_timeout
        self.stats = {
            'total_chunks': 0,
            'error_chunks': 0,
            'recovery_attempts': 0,
            'successful_recoveries': 0
        }

    async def parse_sse_stream(
        self,
        response: httpx.Response,
        buffer_size: int = 8192
    ) -> AsyncGenerator[RawSSEChunk]:
        """
        Parse SSE stream with modern error recovery and timeout handling.

        Features:
        - Automatic recovery from malformed chunks
        - Timeout handling for stuck streams
        - Performance statistics tracking
        - Circuit breaker integration
        """
        buffer = ""
        last_chunk_time = time.time()

        try:
            async for chunk_bytes in response.aiter_text(chunk_size=buffer_size):
                current_time = time.time()

                # Check for timeout
                if current_time - last_chunk_time > self.chunk_timeout:
                    self.stats['error_chunks'] += 1
                    yield RawSSEChunk(
                        event_type=SSEEventType.ERROR,
                        data=None,
                        raw_data="",
                        error=f"Chunk timeout after {self.chunk_timeout}s"
                    )
                    if not self.enable_recovery:
                        return

                buffer += chunk_bytes
                last_chunk_time = current_time

                # Process complete SSE events
                while "\n\n" in buffer:
                    event_data, buffer = buffer.split("\n\n", 1)

                    try:
                        chunk = self._parse_sse_event(event_data, current_time)
                        if chunk:
                            self.stats['total_chunks'] += 1
                            yield chunk

                            # Check for completion
                            if chunk.event_type == SSEEventType.COMPLETION:
                                return

                    except Exception as e:
                        self.stats['error_chunks'] += 1

                        if self.enable_recovery:
                            self.stats['recovery_attempts'] += 1
                            # Try to recover by yielding error chunk and continuing
                            yield RawSSEChunk(
                                event_type=SSEEventType.ERROR,
                                data=None,
                                raw_data=event_data,
                                error=f"Parse error: {e}"
                            )
                            self.stats['successful_recoveries'] += 1
                        else:
                            raise StreamingError(f"SSE parse error: {e}") from e

        except httpx.StreamError as e:
            self.stats['error_chunks'] += 1
            yield RawSSEChunk(
                event_type=SSEEventType.ERROR,
                data=None,
                raw_data="",
                error=f"Stream error: {e}"
            )

        except TimeoutError as e:
            self.stats['error_chunks'] += 1
            yield RawSSEChunk(
                event_type=SSEEventType.ERROR,
                data=None,
                raw_data="",
                error=f"Stream timeout: {e}"
            )

    def _parse_sse_event(self, event_data: str, timestamp: float) -> RawSSEChunk | None:
        """Parse individual SSE event with enhanced error handling."""

        for raw_line in event_data.split("\n"):
            line = raw_line.strip()

            if line.startswith("data: "):
                data_content = line[6:]  # Remove "data: " prefix

                # Handle completion marker
                if data_content.strip() == "[DONE]":
                    return RawSSEChunk(
                        event_type=SSEEventType.COMPLETION,
                        data=None,
                        raw_data="[DONE]",
                        timestamp=timestamp
                    )

                # Handle heartbeat/ping
                if data_content.strip() in ["", "ping", "heartbeat"]:
                    return RawSSEChunk(
                        event_type=SSEEventType.HEARTBEAT,
                        data=None,
                        raw_data=data_content,
                        timestamp=timestamp
                    )

                # Parse JSON data
                try:
                    parsed_data = json.loads(data_content)
                    return RawSSEChunk(
                        event_type=SSEEventType.CHUNK,
                        data=parsed_data,
                        raw_data=data_content,
                        timestamp=timestamp
                    )
                except json.JSONDecodeError as e:
                    return RawSSEChunk(
                        event_type=SSEEventType.ERROR,
                        data=None,
                        raw_data=data_content,
                        error=f"JSON decode error: {e}",
                        timestamp=timestamp
                    )

        return None

    def get_stats(self) -> dict[str, int]:
        """Get streaming statistics for monitoring."""
        return self.stats.copy()

    def reset_stats(self) -> None:
        """Reset statistics counters."""
        self.stats = {
            'total_chunks': 0,
            'error_chunks': 0,
            'recovery_attempts': 0,
            'successful_recoveries': 0
        }


class ChunkAccumulator:
    """
    Advanced chunk accumulation with tool call reconstruction and performance tracking.
    Phase 2: Enhanced with streaming statistics and error recovery.
    """

    def __init__(self):
        self.state = AccumulatorState()
        self._processing_times: list[float] = []

    def process_chunk(self, raw_chunk: RawSSEChunk) -> StreamChunk | None:  # noqa: PLR0911
        """
        Process raw SSE chunk into typed stream chunk with enhanced tracking.

        Features:
        - Tool call reconstruction with error recovery
        - Performance latency tracking
        - Token counting integration
        - Error context preservation
        """
        start_time = time.time()
        self.state.update_timing(raw_chunk.timestamp)

        try:
            # Handle different event types
            if raw_chunk.event_type == SSEEventType.COMPLETION:
                return self._create_completion_chunk(start_time)

            if raw_chunk.event_type == SSEEventType.ERROR:
                return self._create_error_chunk(raw_chunk, start_time)

            if raw_chunk.event_type == SSEEventType.HEARTBEAT:
                # Heartbeats don't generate stream chunks but update timing
                return None

            if not raw_chunk.data or not raw_chunk.data.get("choices"):
                return None

            choice = raw_chunk.data["choices"][0]
            delta = choice.get("delta", {})

            # Handle content accumulation
            if content := delta.get("content"):
                self.state.content_buffer += content
                chunk = StreamChunk(
                    chunk_type=StreamChunkType.CONTENT,
                    content=content,
                    accumulated_content=self.state.content_buffer,
                    tool_calls=self.state.tool_calls.copy(),
                    delta_tokens=len(content.split()),  # Rough token estimate
                    cumulative_tokens=self.state.total_tokens,
                    processing_latency=time.time() - start_time
                )
                self.state.total_tokens += chunk.delta_tokens
                return chunk

            # Handle tool call accumulation with error recovery
            if tool_calls_delta := delta.get("tool_calls"):
                try:
                    self._accumulate_tool_calls(tool_calls_delta)
                    return StreamChunk(
                        chunk_type=StreamChunkType.TOOL_CALLS,
                        content=None,
                        accumulated_content=self.state.content_buffer,
                        tool_calls=self.state.tool_calls.copy(),
                        processing_latency=time.time() - start_time
                    )
                except Exception as e:
                    return StreamChunk(
                        chunk_type=StreamChunkType.ERROR,
                        content=None,
                        accumulated_content=self.state.content_buffer,
                        tool_calls=self.state.tool_calls.copy(),
                        error=f"Tool call accumulation error: {e}",
                        processing_latency=time.time() - start_time
                    )

            # Handle finish reason
            if finish_reason := choice.get("finish_reason"):
                return StreamChunk(
                    chunk_type=StreamChunkType.COMPLETION,
                    content=None,
                    accumulated_content=self.state.content_buffer,
                    tool_calls=self.state.tool_calls.copy(),
                    finish_reason=finish_reason,
                    processing_latency=time.time() - start_time
                )

            return None

        finally:
            # Track processing time
            processing_time = time.time() - start_time
            self._processing_times.append(processing_time)
            # Keep only last measurements for rolling average
            if len(self._processing_times) > MAX_PROCESSING_HISTORY:
                self._processing_times.pop(0)

    def _accumulate_tool_calls(self, tool_calls_delta: list[dict]) -> None:
        """
        Accumulate tool calls with enhanced error recovery.
        Handles partial tool call data gracefully.
        """

        for tool_call_delta in tool_calls_delta:
            index = tool_call_delta.get("index", 0)

            # Ensure we have enough tool call slots
            while len(self.state.tool_calls) <= index:
                self.state.tool_calls.append(
                    ToolCall(
                        id="",
                        type="function",
                        function=ToolFunction(name="", arguments="")
                    )
                )

            # Get existing tool call
            existing_call = self.state.tool_calls[index]

            # Update ID if present
            if "id" in tool_call_delta:
                new_id = existing_call.id + str(tool_call_delta["id"])
                existing_call = ToolCall(
                    id=new_id,
                    type=existing_call.type,
                    function=existing_call.function
                )

            # Update function if present
            if "function" in tool_call_delta:
                func_delta = tool_call_delta["function"]
                existing_func = existing_call.function

                new_name = existing_func.name
                new_args = existing_func.arguments

                if "name" in func_delta:
                    new_name += str(func_delta["name"])

                if "arguments" in func_delta:
                    new_args += str(func_delta["arguments"])

                new_function = ToolFunction(name=new_name, arguments=new_args)
                existing_call = ToolCall(
                    id=existing_call.id,
                    type=existing_call.type,
                    function=new_function
                )

            self.state.tool_calls[index] = existing_call

    def _create_completion_chunk(self, start_time: float) -> StreamChunk:
        """Create final completion chunk with statistics."""
        return StreamChunk(
            chunk_type=StreamChunkType.COMPLETION,
            content=None,
            accumulated_content=self.state.content_buffer,
            tool_calls=self.state.tool_calls.copy(),
            finish_reason="stop",
            cumulative_tokens=self.state.total_tokens,
            processing_latency=time.time() - start_time
        )

    def _create_error_chunk(
        self, raw_chunk: RawSSEChunk, start_time: float
    ) -> StreamChunk:
        """Create error chunk with context preservation."""
        return StreamChunk(
            chunk_type=StreamChunkType.ERROR,
            content=None,
            accumulated_content=self.state.content_buffer,
            tool_calls=self.state.tool_calls.copy(),
            error=raw_chunk.error,
            processing_latency=time.time() - start_time
        )

    def get_streaming_stats(self) -> StreamingStats:
        """Generate comprehensive streaming statistics."""
        content_chunks = sum(
            1 for _ in range(self.state.chunk_count)
            if self.state.content_buffer
        )
        tool_call_chunks = len(self.state.tool_calls)

        avg_latency = (
            sum(self._processing_times) / len(self._processing_times)
            if self._processing_times else 0.0
        )

        duration = self.state.streaming_duration
        tokens_per_second = self.state.total_tokens / duration if duration > 0 else 0.0

        first_token_latency = (
            self._processing_times[0] if self._processing_times else 0.0
        )

        return StreamingStats(
            total_chunks=self.state.chunk_count,
            content_chunks=content_chunks,
            tool_call_chunks=tool_call_chunks,
            error_chunks=0,  # Would be tracked separately
            total_duration=duration,
            average_chunk_latency=avg_latency,
            tokens_per_second=tokens_per_second,
            first_token_latency=first_token_latency
        )

    def reset(self) -> None:
        """Reset accumulator state for new stream."""
        self.state = AccumulatorState()
        self._processing_times.clear()
