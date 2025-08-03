# Hybrid Response Streaming Implementation Plan

## Problem Statement

Currently, the websocket server filters out final response dictionaries containing complete message state, which can lead to missing content in hybrid responses (responses with both text content and tool calls). This happens because:

1. Chat service streams text chunks as `ChatMessage` objects
2. Chat service yields final complete state as raw dictionary
3. Websocket server ignores dictionaries, only processes `ChatMessage` objects
4. Missing content if streaming chunks don't contain complete message

## Solution: Dataclass-Based Streaming Messages

Based on industry best practices (Anthropic, OpenAI), implement an event-based streaming system using dataclasses for optimal performance.

## Implementation Plan

### Phase 1: Create New Streaming Message Types

#### 1.1 Create `src/streaming_types.py`
```python
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

@dataclass
class StreamingMessage:
    """
    High-performance streaming message using dataclass.
    
    Designed for internal message passing between chat service and websocket server.
    Optimized for frequent creation during streaming without validation overhead.
    """
    type: str  # "text", "tool_execution", "final_state" 
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)
    
    # Final state fields (only used for type="final_state")
    complete_content: str | None = None
    tool_calls: list[dict[str, Any]] | None = None
    finish_reason: str | None = None
    
    def is_final_state(self) -> bool:
        """Check if this is a final state message."""
        return self.type == "final_state"
    
    def has_missing_content(self, streamed_content: str) -> bool:
        """Check if final state has content missing from streaming."""
        if not self.is_final_state() or not self.complete_content:
            return False
        return self.complete_content != streamed_content

@dataclass 
class ContentBlock:
    """Represents a content block for structured streaming."""
    index: int
    type: str  # "text", "tool_use"
    content: str = ""
    tool_call: dict[str, Any] | None = None
    complete: bool = False
```

### Phase 2: Update Chat Service

#### 2.1 Modify `src/chat_service.py`

**Import new types:**
```python
from src.streaming_types import StreamingMessage, ContentBlock
```

**Replace ChatMessage with StreamingMessage in method signatures:**
```python
async def process_message(
    self,
    conversation_id: str, 
    user_msg: str,
    request_id: str,
) -> AsyncGenerator[StreamingMessage]:  # Changed from ChatMessage
```

**Update `_stream_llm_response_with_deltas` to yield StreamingMessage:**
```python
# Replace current ChatMessage yields
yield StreamingMessage(
    type="text",
    content=content,
    metadata={"type": "delta", "hop": hop_number}
)

# Replace current dictionary yield with final state
yield StreamingMessage(
    type="final_state",
    content="",  # Empty for final state
    complete_content=message_buffer or None,
    tool_calls=current_tool_calls if current_tool_calls else None,
    finish_reason=finish_reason,
    metadata={"hop": hop_number}
)
```

**Update tool execution messages:**
```python
yield StreamingMessage(
    type="tool_execution",
    content=f"Executing {len(current_tool_calls)} tool(s)...",
    metadata={"tool_count": len(current_tool_calls), "hop": hop_number}
)
```

### Phase 3: Update WebSocket Server

#### 3.1 Modify `src/websocket_server.py`

**Import new types:**
```python
from src.streaming_types import StreamingMessage
```

**Update method signatures:**
```python
async def _send_chat_response(
    self, websocket: WebSocket, request_id: str, message: StreamingMessage
):
```

**Enhance `_handle_streaming_chat` with content tracking:**
```python
async def _handle_streaming_chat(
    self,
    websocket: WebSocket,
    request_id: str,
    conversation_id: str,
    user_message: str,
):
    """Handle streaming chat response with hybrid content support."""
    streamed_content = ""
    
    async for message in self.chat_service.process_message(
        conversation_id, user_message, request_id
    ):
        if message.type == "text":
            streamed_content += message.content
            await self._send_chat_response(websocket, request_id, message)
        elif message.type == "tool_execution":
            await self._send_chat_response(websocket, request_id, message)
        elif message.type == "final_state":
            await self._handle_final_state(
                websocket, request_id, message, streamed_content
            )

    # Send completion signal
    await websocket.send_text(
        json.dumps({
            "request_id": request_id,
            "status": "complete", 
            "chunk": {},
        })
    )
```

**Add final state handler:**
```python
async def _handle_final_state(
    self,
    websocket: WebSocket,
    request_id: str,
    final_message: StreamingMessage,
    streamed_content: str,
):
    """Handle final state message, sending any missing content."""
    if final_message.has_missing_content(streamed_content):
        missing_content = self._calculate_missing_content(
            final_message.complete_content, streamed_content
        )
        
        if missing_content:
            await websocket.send_text(
                json.dumps({
                    "request_id": request_id,
                    "status": "chunk",
                    "chunk": {
                        "type": "text",
                        "data": missing_content,
                        "metadata": {
                            "source": "final_state_completion",
                            "reason": "hybrid_response_gap"
                        },
                    },
                })
            )
    
    # Log final state for debugging
    if final_message.tool_calls:
        logger.debug(
            f"Final state: {len(final_message.tool_calls)} tool calls, "
            f"finish_reason: {final_message.finish_reason}"
        )

def _calculate_missing_content(
    self, complete_content: str | None, streamed_content: str
) -> str:
    """Calculate missing content between complete and streamed content."""
    if not complete_content:
        return ""
    
    if not streamed_content:
        return complete_content
    
    if complete_content.startswith(streamed_content):
        return complete_content[len(streamed_content):]
    
    # Log complex cases for debugging
    logger.warning(
        f"Complex content mismatch: streamed='{streamed_content}', "
        f"complete='{complete_content}'"
    )
    return ""
```

### Phase 4: Backward Compatibility & Migration

#### 4.1 Deprecation Strategy
1. Keep existing `ChatMessage` Pydantic model for external APIs
2. Use `StreamingMessage` dataclass for internal streaming
3. Add conversion utilities if needed

#### 4.2 Convert existing ChatMessage usages
```python
# In websocket_server.py - convert for external response
def _streaming_message_to_response(self, message: StreamingMessage) -> dict:
    """Convert internal StreamingMessage to external response format."""
    return {
        "type": message.type,
        "data": message.content,
        "metadata": message.metadata
    }
```

### Phase 5: Testing & Validation

#### 5.1 Create test scenarios
```python
# test_hybrid_streaming.py
async def test_perfect_streaming():
    """Test case where all content is streamed correctly."""
    
async def test_missing_content():
    """Test case where final state has more content than streamed."""
    
async def test_no_streaming():
    """Test case where no content was streamed."""
    
async def test_tool_calls_only():
    """Test case with only tool calls, no text content."""

async def test_complex_hybrid():
    """Test case with multiple tool calls and text content."""
```

#### 5.2 Performance benchmarks
- Compare dataclass vs Pydantic creation time
- Memory usage comparison
- Streaming latency measurement

### Phase 6: Configuration & Monitoring

#### 6.1 Add configuration options
```yaml
# config.yaml
chat:
  service:
    streaming:
      hybrid_content_detection: true
      missing_content_logging: true
      content_mismatch_threshold: 0.95
```

#### 6.2 Add monitoring
- Track hybrid response frequency
- Monitor missing content events
- Alert on content mismatches

## Migration Steps

### Step 1: Create streaming_types.py
- No breaking changes
- Pure addition

### Step 2: Update chat_service.py imports and yields
- Replace ChatMessage with StreamingMessage in streaming methods
- Keep ChatMessage for non-streaming methods
- Update type hints

### Step 3: Update websocket_server.py
- Add final state handling
- Enhance content tracking
- Update type hints

### Step 4: Test thoroughly
- Unit tests for each component
- Integration tests for full streaming flow
- Performance benchmarks

### Step 5: Deploy and monitor
- Monitor for missing content events
- Validate performance improvements
- Collect metrics on hybrid responses

## Benefits

1. **Performance**: Dataclass ~2-3x faster than Pydantic for frequent creation
2. **Industry Alignment**: Matches patterns used by Anthropic, OpenAI
3. **Reliability**: Guarantees no content loss in hybrid responses
4. **Maintainability**: Clear separation of concerns, type-safe
5. **Monitoring**: Better visibility into streaming issues

## Risks & Mitigation

1. **Risk**: Breaking existing functionality
   - **Mitigation**: Gradual migration, extensive testing

2. **Risk**: Performance regression
   - **Mitigation**: Benchmark before/after, dataclass should be faster

3. **Risk**: Complex edge cases in content matching
   - **Mitigation**: Comprehensive logging, conservative matching algorithm

## Success Metrics

- Zero reported missing content issues
- Performance improvement in streaming latency
- Reduced memory usage during streaming
- Increased test coverage for hybrid responses

## Timeline

- **Week 1**: Implement Phase 1-2 (streaming types, chat service)
- **Week 2**: Implement Phase 3 (websocket server updates)  
- **Week 3**: Testing and validation (Phase 5)
- **Week 4**: Deployment and monitoring (Phase 6)
