#!/usr/bin/env python3
"""
Simple test to verify persistent connection functionality
"""
import asyncio
import os
import tempfile

from src.history.models import ChatEvent
from src.history.repositories.sql_repo import AsyncSqlRepo


async def test_persistent_connection():
    """Test that the repository works with persistent connections."""
    # Use a temporary database file
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp:
        db_path = tmp.name

    try:
        # Create repository with persistent connection
        repo = AsyncSqlRepo(db_path=db_path)

        # Add an event
        event1 = ChatEvent(
            conversation_id="test-conv",
            type="user_message",
            role="user",
            content="Hello",
            extra={"request_id": "req1"}
        )

        await repo.add_event(event1)

        # Add another event
        event2 = ChatEvent(
            conversation_id="test-conv",
            type="assistant_message",
            role="assistant",
            content="Hi there!",
            extra={"request_id": "req2"}
        )

        await repo.add_event(event2)

        # Test retrieving events
        events = await repo.get_events("test-conv")
        expected_events = 2
        assert len(events) == expected_events, f"Expected {expected_events} events, got {len(events)}"

        # Test other operations
        conversations = await repo.list_conversations()
        assert "test-conv" in conversations, "Should find the test conversation"

        # Test get event by request ID
        found_event = await repo.get_event_by_request_id("test-conv", "req1")
        assert found_event is not None, "Should find event by request ID"
        assert found_event.content == "Hello", "Should have correct content"

        # Test close functionality
        await repo.close()

        # Verify we can reinitialize and still access data
        repo2 = AsyncSqlRepo(db_path=db_path)
        events_again = await repo2.get_events("test-conv")
        assert len(events_again) == expected_events, "Should still have events after reinit"
        await repo2.close()

        print("✅ All persistent connection tests passed!")

    finally:
        # Clean up temp file
        if os.path.exists(db_path):
            os.unlink(db_path)


if __name__ == "__main__":
    asyncio.run(test_persistent_connection())
