#!/usr/bin/env python3
"""
MCP Backend Basic System Test

This script tests the core functionality without stress testing concurrent writes.
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


async def test_async_jsonl_repo_basic():
    """Test basic AsyncJsonlRepo functionality."""
    print("🧪 Testing AsyncJsonlRepo basic functionality...")
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
        temp_path = f.name
    
    try:
        repo = AsyncJsonlRepo(temp_path)
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
        repo2 = AsyncJsonlRepo(temp_path)
        events = await repo2.get_events(conv_id)
        assert len(events) == 2, f"Expected 2 persisted events, got {len(events)}"
        
        # Test filtering
        filtered = await repo2.last_n_tokens(conv_id, 1000)
        assert len(filtered) == 2, "All events should be visible"
        print("  ✅ AsyncJsonlRepo basic functionality works")
        
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


async def main():
    """Run all tests."""
    print("🚀 Starting MCP Backend Basic Tests\n")
    
    try:
        test_chat_event_model()
        print()
        
        await test_async_jsonl_repo_basic()
        print()
        
        print("🎉 All basic tests passed! System is working correctly.")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
