#!/usr/bin/env python3
"""
Test async context manager functionality
"""
import asyncio
import os
import tempfile

from src.history.models import ChatEvent
from src.history.repositories.sql_repo import AsyncSqlRepo


async def test_context_manager():
    """Test that the repository works as an async context manager."""
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp:
        db_path = tmp.name

    try:
        # Test using async context manager
        async with AsyncSqlRepo(db_path=db_path) as repo:
            # Add an event
            event = ChatEvent(
                conversation_id="test-conv",
                type="user_message",
                role="user",
                content="Hello from context manager",
                extra={"request_id": "req1"}
            )

            await repo.add_event(event)

            # Retrieve events
            events = await repo.get_events("test-conv")
            assert len(events) == 1, f"Expected 1 event, got {len(events)}"

        # Connection should be closed automatically after exiting context
        print("✅ Async context manager test passed!")

        # Test that data persists after context manager closes
        async with AsyncSqlRepo(db_path=db_path) as repo2:
            events = await repo2.get_events("test-conv")
            assert len(events) == 1, "Data should persist after context close"
            assert events[0].content == "Hello from context manager"

        print("✅ Data persistence after context close test passed!")

    finally:
        if os.path.exists(db_path):
            os.unlink(db_path)


if __name__ == "__main__":
    asyncio.run(test_context_manager())
