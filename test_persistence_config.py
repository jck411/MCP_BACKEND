#!/usr/bin/env python3
"""Test script for persistence configuration."""

import asyncio
import tempfile
from pathlib import Path

from src.config import Configuration
from src.history.models import ChatEvent
from src.history.repositories.sql_repo import AsyncSqlRepo


async def test_persistence_enabled():
    """Test that events are stored when persistence is enabled."""
    print("🧪 Testing persistence enabled...")
    
    persistence_config = {
        "enabled": True,
        "retention_policy": "token_limit",
        "max_tokens_per_conversation": 1000,
        "max_messages_per_conversation": 10,
        "retention_days": 30,
        "clear_on_startup": False
    }
    
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
        db_path = f.name
    
    try:
        repo = AsyncSqlRepo(db_path, persistence_config)
        
        # Add an event
        event = ChatEvent(
            conversation_id="test-conv",
            type="user_message",
            role="user",
            content="Hello, persistent world!",
            extra={"request_id": "test-req-1"}
        )
        event.compute_and_cache_tokens()
        
        success = await repo.add_event(event)
        assert success, "Failed to add event with persistence enabled"
        
        # Retrieve events
        events = await repo.get_events("test-conv")
        assert len(events) == 1, f"Expected 1 event, got {len(events)}"
        assert events[0].content == "Hello, persistent world!"
        
        print("  ✅ Persistence enabled works correctly")
        
    finally:
        Path(db_path).unlink(missing_ok=True)


async def test_persistence_disabled():
    """Test that events are not stored when persistence is disabled."""
    print("🧪 Testing persistence disabled...")
    
    persistence_config = {
        "enabled": False,
        "retention_policy": "unlimited",
        "max_tokens_per_conversation": 1000,
        "max_messages_per_conversation": 10,
        "retention_days": 30,
        "clear_on_startup": False
    }
    
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
        db_path = f.name
    
    try:
        repo = AsyncSqlRepo(db_path, persistence_config)
        
        # Add an event
        event = ChatEvent(
            conversation_id="test-conv",
            type="user_message",
            role="user",
            content="Hello, non-persistent world!",
            extra={"request_id": "test-req-1"}
        )
        event.compute_and_cache_tokens()
        
        success = await repo.add_event(event)
        assert success, "add_event should return True even when persistence disabled"
        
        # Retrieve events - should be empty
        events = await repo.get_events("test-conv")
        assert len(events) == 0, f"Expected 0 events with persistence disabled, got {len(events)}"
        
        print("  ✅ Persistence disabled works correctly")
        
    finally:
        Path(db_path).unlink(missing_ok=True)


async def test_token_limit_retention():
    """Test token limit retention policy."""
    print("🧪 Testing token limit retention...")
    
    persistence_config = {
        "enabled": True,
        "retention_policy": "token_limit",
        "max_tokens_per_conversation": 50,  # Very low limit for testing
        "max_messages_per_conversation": 100,
        "retention_days": 30,
        "clear_on_startup": False
    }
    
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
        db_path = f.name
    
    try:
        repo = AsyncSqlRepo(db_path, persistence_config)
        
        # Add multiple events with different token counts
        events_to_add = [
            ("First message", 20),  # Should be kept
            ("Second message", 25),  # Should be kept (total: 45)
            ("Third message", 30),   # Should be removed (would make total: 75)
        ]
        
        for i, (content, token_count) in enumerate(events_to_add):
            event = ChatEvent(
                conversation_id="test-conv",
                type="user_message", 
                role="user",
                content=content,
                token_count=token_count,
                extra={"request_id": f"test-req-{i}"}
            )
            await repo.add_event(event)
        
        # Apply retention (simulates startup behavior)
        await repo._apply_retention_policies()
        
        # Check remaining events
        events = await repo.get_events("test-conv")
        print(f"  Events after retention: {len(events)}")
        for event in events:
            print(f"    - {event.content} ({event.token_count} tokens)")
        
        # Should keep the 2 most recent events within token limit
        assert len(events) <= 2, f"Expected <= 2 events after retention, got {len(events)}"
        
        print("  ✅ Token limit retention works correctly")
        
    finally:
        Path(db_path).unlink(missing_ok=True)


async def main():
    """Run all persistence tests."""
    print("🔬 Testing Persistence Configuration...")
    print("====================================")
    
    await test_persistence_enabled()
    await test_persistence_disabled()
    await test_token_limit_retention()
    
    print("\n✅ All persistence tests passed!")


if __name__ == "__main__":
    asyncio.run(main())
