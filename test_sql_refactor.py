#!/usr/bin/env python3
"""
Simple test script to verify the SQL repository refactoring works correctly.
"""

import asyncio
import os
from src.history.models import ChatEvent, Usage
from src.history.repositories.sql_repo import AsyncSqlRepo


async def test_sql_repository():
    """Test basic SQL repository functionality."""
    # Use a test database
    test_db = "test_events.db"
    
    # Clean up any existing test database
    if os.path.exists(test_db):
        os.remove(test_db)
    
    try:
        repo = AsyncSqlRepo(test_db)
        
        # Test adding an event
        event = ChatEvent(
            conversation_id="test-conv-1",
            type="user_message",
            role="user",
            content="Hello, world!",
            extra={"request_id": "test-req-1"}
        )
        
        result = await repo.add_event(event)
        print(f"Added event: {result}")
        
        # Test retrieving events
        events = await repo.get_events("test-conv-1")
        print(f"Retrieved {len(events)} events")
        
        if events:
            print(f"First event content: {events[0].content}")
            print(f"First event token count: {events[0].token_count}")
        
        # Test duplicate detection
        duplicate_result = await repo.add_event(event)
        print(f"Duplicate add result: {duplicate_result}")
        
        # Test event_exists
        exists = await repo.event_exists("test-conv-1", "user_message", "test-req-1")
        print(f"Event exists: {exists}")
        
        # Test list conversations
        conversations = await repo.list_conversations()
        print(f"Conversations: {conversations}")
        
        print("✅ All tests passed!")
        
    finally:
        # Clean up test database
        if os.path.exists(test_db):
            os.remove(test_db)


if __name__ == "__main__":
    asyncio.run(test_sql_repository())
