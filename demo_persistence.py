#!/usr/bin/env python3
"""
Demonstration script showing different persistence configurations.
"""

import asyncio
import tempfile
from pathlib import Path

from src.history.models import ChatEvent
from src.history.repositories.sql_repo import AsyncSqlRepo


async def demo_persistence_modes():
    """Demonstrate different persistence configuration modes."""
    print("🔧 MCP Backend Persistence Configuration Demo")
    print("=" * 50)
    
    # Demo 1: Full persistence with token limit
    print("\n📝 Demo 1: Full Persistence with Token Limit")
    print("-" * 40)
    
    config1 = {
        "enabled": True,
        "retention_policy": "token_limit",
        "max_tokens_per_conversation": 100,
        "clear_on_startup": False
    }
    
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
        db_path1 = f.name
    
    try:
        repo1 = AsyncSqlRepo(db_path1, config1)
        
        # Add some messages
        messages = [
            "Hello, this is a test message",  # ~7 tokens
            "This is another message with more content to test token counting",  # ~12 tokens
            "Yet another message for testing purposes",  # ~7 tokens
            "Final message to exceed token limit and trigger retention"  # ~9 tokens
        ]
        
        for i, content in enumerate(messages):
            event = ChatEvent(
                conversation_id="demo-conv",
                type="user_message",
                role="user",
                content=content,
                extra={"request_id": f"demo-req-{i}"}
            )
            event.compute_and_cache_tokens()
            await repo1.add_event(event)
            print(f"  Added: '{content[:30]}...' ({event.token_count} tokens)")
        
        # Show what's retained
        events = await repo1.get_events("demo-conv")
        total_tokens = sum(e.token_count or 0 for e in events)
        print(f"  Retained: {len(events)} messages, {total_tokens} total tokens")
        
    finally:
        Path(db_path1).unlink(missing_ok=True)
    
    # Demo 2: No persistence (session-only)
    print("\n🔄 Demo 2: No Persistence (Session Only)")
    print("-" * 40)
    
    config2 = {
        "enabled": False,
        "retention_policy": "unlimited",
        "clear_on_startup": False
    }
    
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
        db_path2 = f.name
    
    try:
        repo2 = AsyncSqlRepo(db_path2, config2)
        
        # Add a message
        event = ChatEvent(
            conversation_id="temp-conv",
            type="user_message",
            role="user",
            content="This message won't be persisted",
            extra={"request_id": "temp-req"}
        )
        
        success = await repo2.add_event(event)
        print(f"  Add event returned: {success}")
        
        # Try to retrieve - should be empty
        events = await repo2.get_events("temp-conv")
        print(f"  Retrieved events: {len(events)} (expected: 0)")
        
    finally:
        Path(db_path2).unlink(missing_ok=True)
    
    # Demo 3: Message count limit
    print("\n📊 Demo 3: Message Count Limit")
    print("-" * 40)
    
    config3 = {
        "enabled": True,
        "retention_policy": "message_count",
        "max_messages_per_conversation": 3,
        "clear_on_startup": False
    }
    
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
        db_path3 = f.name
    
    try:
        repo3 = AsyncSqlRepo(db_path3, config3)
        
        # Add 5 messages
        for i in range(5):
            event = ChatEvent(
                conversation_id="count-conv",
                type="user_message",
                role="user",
                content=f"Message number {i + 1}",
                extra={"request_id": f"count-req-{i}"}
            )
            await repo3.add_event(event)
            print(f"  Added message {i + 1}")
        
        # Apply retention policy
        await repo3._apply_retention_to_conversation("count-conv", "message_count")
        
        # Show retained messages
        events = await repo3.get_events("count-conv")
        print(f"  Retained: {len(events)} most recent messages (limit: 3)")
        for event in events:
            print(f"    - {event.content}")
        
    finally:
        Path(db_path3).unlink(missing_ok=True)
    
    print("\n✅ Demo complete! Check your config.yaml to set your preferred persistence mode.")


if __name__ == "__main__":
    asyncio.run(demo_persistence_modes())
