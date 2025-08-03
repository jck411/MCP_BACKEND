#!/usr/bin/env python3
"""
Simple test to verify persistent connection functionality
"""
import asyncio
import tempfile
import os
from datetime import datetime
from src.history.models import ChatEvent
from src.history.repositories.sql_repo import AsyncSqlRepo


async def test_persistent_connection():
    """Test that the repository uses persistent connections efficiently."""
    # Use a temporary database file
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp:
        db_path = tmp.name
    
    try:
        # Create repository with persistent connection
        repo = AsyncSqlRepo(db_path=db_path)
        
        # Verify connection is None initially
        assert repo._connection is None, "Connection should be None initially"
        
        # Add an event - this should initialize the connection
        event1 = ChatEvent(
            conversation_id="test-conv",
            type="user_message",
            role="user",
            content="Hello",
            extra={"request_id": "req1"}
        )
        
        await repo.add_event(event1)
        
        # Verify connection is now established
        assert repo._connection is not None, "Connection should be established after first operation"
        connection_id = id(repo._connection)
        
        # Add another event - should reuse the same connection
        event2 = ChatEvent(
            conversation_id="test-conv",
            type="assistant_message", 
            role="assistant",
            content="Hi there!",
            extra={"request_id": "req2"}
        )
        
        await repo.add_event(event2)
        
        # Verify same connection is being reused
        assert repo._connection is not None, "Connection should still exist"
        assert id(repo._connection) == connection_id, "Should reuse the same connection"
        
        # Test retrieving events - should still use same connection
        events = await repo.get_events("test-conv")
        assert len(events) == 2, f"Expected 2 events, got {len(events)}"
        assert id(repo._connection) == connection_id, "Should still reuse the same connection"
        
        # Test other operations
        conversations = await repo.list_conversations()
        assert "test-conv" in conversations, "Should find the test conversation"
        assert id(repo._connection) == connection_id, "Should still reuse the same connection"
        
        # Test close functionality
        await repo.close()
        assert repo._connection is None, "Connection should be None after close"
        assert repo._initialized is False, "Should be uninitialized after close"
        
        print("✅ All persistent connection tests passed!")
        
    finally:
        # Clean up temp file
        if os.path.exists(db_path):
            os.unlink(db_path)


if __name__ == "__main__":
    asyncio.run(test_persistent_connection())
