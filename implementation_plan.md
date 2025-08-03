# Hybrid Response Streaming Implementation Plan - REVISED

## Problem Statement

Currently, the websocket server filters out final response dictionaries containing complete message state, which can lead to missing content in hybrid responses (responses with both text content and tool calls). This happens because:

1. Chat service streams text chunks as `ChatMessage` objects
2. Chat service yields final complete state as raw dictionary  
3. Websocket server ignores dictionaries, only processes `ChatMessage` objects
4. Missing content if streaming chunks don't contain complete message

## Codebase Analysis

### Current Architecture (Must Preserve)
- **ChatService.process_message()**: `AsyncGenerator[ChatMessage]` - core contract
- **ChatMessage**: Pydantic model used throughout websocket layer
- **ChatEvent**: Database persistence model with sophisticated token counting
- **AsyncSqlRepo**: High-performance SQLite with conversation locking
- **Configuration**: YAML-based config system with streaming controls

### Key Dependencies
- WebSocket server expects `ChatMessage` objects only
- Database persistence requires `ChatEvent` objects  
- Token counting and conversation history management
- Concurrency safety with per-conversation locks

## Solution: Minimal Impact Hybrid Support

**Strategy**: Extend existing `ChatMessage` model to support final state, maintain all existing contracts.

## Implementation Plan

### Phase 1: Extend ChatMessage Model (Zero Breaking Changes)

#### 1.1 Update `src/chat_service.py` - ChatMessage Definition
```python
class ChatMessage(BaseModel):
    """
    Represents a chat message with metadata.
    Pydantic model for proper validation and serialization.
    """
    type: str  # "text", "tool_execution", "final_state"
    content: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    
    # NEW: Optional fields for final state (backward compatible)
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
```

### Phase 2: Update Chat Service Streaming (Preserve Existing Contract)

#### 2.1 Modify `_stream_llm_response_with_deltas` in `src/chat_service.py`

**BEFORE (current):**
```python
yield {
    "content": message_buffer or None,
    "tool_calls": current_tool_calls if current_tool_calls else None,
    "finish_reason": finish_reason,
}
```

**AFTER (new):**
```python
# Replace dictionary yield with ChatMessage final state
yield ChatMessage(
    type="final_state",
    content="",  # Empty for final state type
    complete_content=message_buffer or None,
    tool_calls=current_tool_calls if current_tool_calls else None,
    finish_reason=finish_reason,
    metadata={"hop": hop_number, "source": "llm_completion"}
)
```

**Key Point**: `AsyncGenerator[ChatMessage]` contract preserved - no breaking changes!

### Phase 3: Update WebSocket Server (Minimal Changes)

#### 3.1 Enhance `_handle_streaming_chat` in `src/websocket_server.py`

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
        # Track streamed text content
        if message.type == "text":
            streamed_content += message.content
            await self._send_chat_response(websocket, request_id, message)
        elif message.type == "tool_execution":
            await self._send_chat_response(websocket, request_id, message)
        elif message.type == "final_state":
            # NEW: Handle final state to catch missing content
            await self._handle_final_state_message(
                websocket, request_id, message, streamed_content
            )
        else:
            # Preserve existing behavior for unknown types
            await self._send_chat_response(websocket, request_id, message)

    # Send completion signal (unchanged)
    await websocket.send_text(json.dumps({
        "request_id": request_id,
        "status": "complete",
        "chunk": {},
    }))
```

#### 3.2 Add Final State Handler

```python
async def _handle_final_state_message(
    self,
    websocket: WebSocket,
    request_id: str, 
    final_message: ChatMessage,
    streamed_content: str,
):
    """Handle final state message, sending any missing content."""
    if final_message.has_missing_content(streamed_content):
        missing_content = self._calculate_missing_content(
            final_message.complete_content, streamed_content
        )
        
        if missing_content:
            # Send missing content using existing response format
            await websocket.send_text(json.dumps({
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
            }))
            logger.info(f"Sent missing content: '{missing_content[:50]}...'")
    
    # Log final state metadata (preserve existing logging patterns)
    if final_message.tool_calls:
        logger.debug(
            f"Final state: {len(final_message.tool_calls)} tool calls, "
            f"finish_reason: {final_message.finish_reason}" 
        )

def _calculate_missing_content(
    self, complete_content: str | None, streamed_content: str
) -> str:
    """Calculate missing content with conservative matching."""
    if not complete_content:
        return ""
    
    if not streamed_content:
        return complete_content
        
    if complete_content.startswith(streamed_content):
        return complete_content[len(streamed_content):]
    
    # Log mismatches for debugging (preserve existing logging style)
    logger.warning(
        f"Content mismatch detected - may indicate streaming issue. "
        f"Streamed: '{streamed_content[:50]}...', "
        f"Complete: '{complete_content[:50]}...'"
    )
    return ""
```

### Phase 4: Database & Persistence Integration

#### 4.1 Ensure ChatEvent Compatibility

The existing `ChatEvent` model already handles:
- Tool calls via `tool_calls: list[ToolCall] | None`
- Content tracking with `content: str | list[Part] | None`  
- Usage tracking with `usage: Usage | None`

**No changes needed** - existing delta persistence and compaction handle this correctly.

#### 4.2 Verify Repository Integration

The existing repository patterns remain unchanged:
- `await self.repo.compact_deltas()` - handles final content correctly
- `build_conversation_with_token_limit()` - works with existing events
- Per-conversation locking - preserved

### Phase 5: Configuration Integration

#### 5.1 Add Configuration Options to `config.yaml`

```yaml
chat:
  service:
    streaming:
      enabled: true
      hybrid_content_detection: true  # NEW
      missing_content_logging: true   # NEW
      content_mismatch_warning: true  # NEW
```

#### 5.2 Update `src/config.py` Methods

```python
def get_hybrid_streaming_config(self) -> dict[str, Any]:
    """Get hybrid streaming configuration."""
    service_config = self._config.get("chat", {}).get("service", {})
    streaming_config = service_config.get("streaming", {})
    
    return {
        "hybrid_detection": streaming_config.get("hybrid_content_detection", True),
        "missing_content_logging": streaming_config.get("missing_content_logging", True),
        "content_mismatch_warning": streaming_config.get("content_mismatch_warning", True),
    }
```

### Phase 6: Testing Strategy

#### 6.1 Unit Tests (Preserve Existing Patterns)
```python
# test_hybrid_streaming.py
class TestHybridStreaming:
    async def test_perfect_streaming_no_missing_content(self):
        """Test normal case - no missing content."""
        
    async def test_missing_content_detection(self):
        """Test missing content is detected and sent."""
        
    async def test_final_state_message_format(self):
        """Test ChatMessage final_state format."""
        
    async def test_backward_compatibility(self):
        """Test existing streaming still works."""
```

#### 6.2 Integration Tests
- Full WebSocket flow with hybrid responses
- Database persistence of final states
- Configuration loading and application

## Migration Steps (Zero Downtime)

### Step 1: Extend ChatMessage (Backward Compatible)
- Add optional fields to ChatMessage
- No breaking changes - all existing code works

### Step 2: Update Chat Service Yields  
- Replace dict yield with ChatMessage final_state
- Maintain AsyncGenerator[ChatMessage] contract

### Step 3: Update WebSocket Final State Handling
- Add final_state message type handling
- Preserve all existing message type behavior

### Step 4: Add Configuration Support
- Add hybrid streaming config options
- Default values preserve existing behavior

### Step 5: Testing & Deployment
- Comprehensive testing of hybrid scenarios
- Monitor for missing content events

## Benefits of Revised Plan

1. **Zero Breaking Changes**: All existing contracts preserved
2. **Minimal Code Changes**: <50 lines of new code total
3. **Preserves Architecture**: Respects existing patterns and systems  
4. **Database Compatible**: Works with existing ChatEvent/repo system
5. **Configuration Integrated**: Uses existing YAML config patterns
6. **Type Safe**: Maintains Pydantic validation throughout

## Risks & Mitigation (Revised)

1. **Risk**: ChatMessage model changes
   - **Mitigation**: Optional fields, backward compatible

2. **Risk**: Performance impact of new fields
   - **Mitigation**: Optional fields have minimal overhead

3. **Risk**: Complex content matching edge cases
   - **Mitigation**: Conservative matching, comprehensive logging

## Success Metrics

- Zero reported missing content in hybrid responses
- No performance regression in streaming
- All existing tests continue to pass
- New hybrid content detection works correctly

## Timeline (Revised)

- **Day 1**: Extend ChatMessage model (backward compatible)
- **Day 2**: Update chat service final state yield
- **Day 3**: Update websocket server final state handling  
- **Day 4**: Add configuration options and testing
- **Day 5**: Integration testing and deployment

**Total Effort**: ~2-3 days instead of 4 weeks, with zero breaking changes.
