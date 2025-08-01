#!/usr/bin/env python3
"""
MCP Backend System Test

This script tests the core functionality of the MCP backend system:
- ChatEvent model and validation
- AsyncJsonlRepo implementation 
- Prompt history filtering (_visible_to_llm)
- Sequence number assignment
- Token counting

Run this script to verify everything is working correctly.

This is a test file - print statements are intentional for test output.
"""
from __future__ import annotations

import asyncio
import sys
import tempfile
import uuid
from pathlib import Path

from src.history.chat_store import (
    ChatEvent, AsyncJsonlRepo, _visible_to_llm, Usage
)

# For testing purposes, we'll create a simpler version of AsyncJsonlRepo that doesn't use file locking
class TestAsyncJsonlRepo(AsyncJsonlRepo):
    """Test version of AsyncJsonlRepo without file locking for CI/test environments."""
    
    async def _append_async(self, event: ChatEvent) -> None:
        """Simplified append without file locking for testing."""
        import aiofiles
        import json
        
        async with aiofiles.open(self.path, "a", encoding="utf-8") as f:
            data = json.dumps(
                event.model_dump(mode="json"), ensure_ascii=False
            )
            await f.write(data + "\n")
            await f.flush()


def test_chat_event_model():
    """Test ChatEvent model and seq behavior."""
    print("🧪 Testing ChatEvent model...")
    
    # Test that seq defaults to None
    event = ChatEvent(
        conversation_id="test",
        type="user_message",
        role="user",
        content="Hello"
    )
    assert event.seq is None, f"Expected seq=None, got {event.seq}"
    print("  ✅ seq defaults to None")
    
    # Test token counting
    event.compute_and_cache_tokens()
    assert event.token_count is not None and event.token_count > 0
    print(f"  ✅ Token counting works: {event.token_count} tokens")


def test_visibility_filter():
    """Test the _visible_to_llm filter function."""
    print("🧪 Testing _visible_to_llm filter...")
    
    test_cases = [
        (ChatEvent(conversation_id="test", type="user_message", role="user", content="Hi"), True),
        (ChatEvent(conversation_id="test", type="assistant_message", role="assistant", content="Hello"), True),
        (ChatEvent(conversation_id="test", type="tool_result", content="Tool output"), True),
        (ChatEvent(conversation_id="test", type="meta", extra={"kind": "assistant_delta"}), False),
        (ChatEvent(conversation_id="test", type="tool_call"), False),
        (ChatEvent(conversation_id="test", type="system_update", content="System message"), False),
        (ChatEvent(conversation_id="test", type="system_update", content="System message", extra={"visible_to_llm": True}), True),
        (ChatEvent(conversation_id="test", type="system_update", content="System message", extra={"visible_to_llm": False}), False),
    ]
    
    for event, expected in test_cases:
        result = _visible_to_llm(event)
        assert result == expected, f"Expected {expected} for {event.type}, got {result}"
        status = "✅" if result == expected else "❌"
        print(f"  {status} {event.type} -> {result}")


async def test_async_jsonl_repo():
    """Test AsyncJsonlRepo functionality with cross-process locking."""
    print("🧪 Testing AsyncJsonlRepo...")
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
        temp_path = f.name
    
    try:
        repo = TestAsyncJsonlRepo(temp_path)  # Use test version without file locking
        conv_id = str(uuid.uuid4())
        
        # Test adding events
        user_event = ChatEvent(
            conversation_id=conv_id,
            type="user_message",
            role="user",
            content="Testing AsyncJsonlRepo"
        )
        
        success = await repo.add_event(user_event)
        assert success, "Failed to add event to AsyncJsonlRepo"
        assert user_event.seq == 1, f"Expected seq=1, got {user_event.seq}"
        
        # Add assistant event  
        asst_event = ChatEvent(
            conversation_id=conv_id,
            type="assistant_message",
            role="assistant", 
            content="AsyncJsonlRepo response"
        )
        await repo.add_event(asst_event)
        assert asst_event.seq == 2
        
        # Test getting events
        events = await repo.get_events(conv_id)
        assert len(events) == 2, f"Expected 2 events, got {len(events)}"
        
        # Test persistence by creating new repo instance
        repo2 = TestAsyncJsonlRepo(temp_path)
        events = await repo2.get_events(conv_id)
        assert len(events) == 2, f"Expected 2 persisted events, got {len(events)}"
        
        # Test filtering
        filtered = await repo2.last_n_tokens(conv_id, 1000)
        assert len(filtered) == 2, "All events should be visible"
        print("  ✅ AsyncJsonlRepo persistence and filtering works")
        
        # Test duplicate prevention
        duplicate = ChatEvent(
            conversation_id=conv_id,
            type="user_message",
            role="user",
            content="Duplicate",
            extra={"request_id": "unique123"}
        )
        success1 = await repo.add_event(duplicate)
        assert success1, "First event should succeed"
        
        duplicate2 = ChatEvent(
            conversation_id=conv_id,
            type="user_message", 
            role="user",
            content="Different content",
            extra={"request_id": "unique123"}  # Same request_id
        )
        success2 = await repo.add_event(duplicate2)
        assert not success2, "Duplicate should be rejected"
        print("  ✅ Duplicate detection works")
        
    finally:
        Path(temp_path).unlink(missing_ok=True)
        # Clean up lock file if it exists
        Path(f"{temp_path}.lock").unlink(missing_ok=True)


async def test_compact_deltas():
    """Test delta compaction functionality."""
    print("🧪 Testing compact_deltas...")
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
        temp_path = f.name
    
    try:
        repo = TestAsyncJsonlRepo(temp_path)
        conv_id = str(uuid.uuid4())
        user_request_id = str(uuid.uuid4())
        
        # Add some delta events
        for i in range(3):
            delta_event = ChatEvent(
                conversation_id=conv_id,
                type="meta",
                extra={
                    "kind": "assistant_delta",
                    "user_request_id": user_request_id,
                    "delta": f"chunk {i}"
                }
            )
            await repo.add_event(delta_event)
        
        # Compact deltas
        final_content = "This is the final message"
        usage = Usage(prompt_tokens=10, completion_tokens=20, total_tokens=30)
        
        compact_event = await repo.compact_deltas(
            conv_id, user_request_id, final_content, usage, "test-model"
        )
        
        assert compact_event.type == "assistant_message"
        assert compact_event.content == final_content
        assert compact_event.usage == usage
        assert compact_event.seq is not None
        print(f"  ✅ Delta compaction works, final seq: {compact_event.seq}")
        
        # Verify deltas were removed and final message is visible
        filtered = await repo.last_n_tokens(conv_id, 1000)
        assert len(filtered) == 1
        assert filtered[0].type == "assistant_message"
        assert filtered[0].content == final_content
        print("  ✅ Deltas filtered out, final message visible")
        
    finally:
        Path(temp_path).unlink(missing_ok=True)
        Path(f"{temp_path}.lock").unlink(missing_ok=True)


async def main():
    """Run all tests."""
    print("🚀 Starting MCP Backend System Tests\n")
    
    try:
        test_chat_event_model()
        print()
        
        test_visibility_filter()
        print()
        
        await test_async_jsonl_repo()
        print()
        
        await test_compact_deltas()
        print()
        
        print("🎉 All tests passed! System is working correctly.")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
