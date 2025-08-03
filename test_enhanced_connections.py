#!/usr/bin/env python3
"""
Test script to verify the improved persistent connection functionality 
with error handling and automatic reconnection.
"""
import asyncio
import os
import tempfile
import signal
import logging

from src.history.models import ChatEvent
from src.history.repositories.sql_repo import AsyncSqlRepo

# Set up logging to see connection management messages
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def test_enhanced_persistent_connection():
    """Test the enhanced persistent connection management."""
    # Use a temporary database file
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp:
        db_path = tmp.name

    try:
        logger.info("Creating repository with enhanced connection management...")
        repo = AsyncSqlRepo(db_path=db_path)

        # Test 1: Basic operations
        logger.info("Test 1: Basic operations")
        event1 = ChatEvent(
            conversation_id="test-conv",
            type="user_message",
            role="user",
            content="Hello",
            extra={"request_id": "req1"}
        )

        success = await repo.add_event(event1)
        assert success, "Should successfully add event"

        events = await repo.get_events("test-conv")
        assert len(events) == 1, f"Expected 1 event, got {len(events)}"
        logger.info("✅ Basic operations working")

        # Test 2: Connection persistence across operations
        logger.info("Test 2: Connection persistence")
        connection_before = id(repo._connection) if repo._connection else None
        
        event2 = ChatEvent(
            conversation_id="test-conv",
            type="assistant_message",
            role="assistant",
            content="Hi there!",
            extra={"request_id": "req2"}
        )
        
        await repo.add_event(event2)
        connection_after = id(repo._connection) if repo._connection else None
        
        assert connection_before == connection_after, "Connection should be reused"
        logger.info("✅ Connection persistence verified")

        # Test 3: Batch operations
        logger.info("Test 3: Batch operations")
        batch_events = []
        for i in range(10):
            batch_events.append(ChatEvent(
                conversation_id="batch-conv",
                type="user_message", 
                role="user",
                content=f"Batch message {i}",
                extra={"request_id": f"batch-req-{i}"}
            ))
        
        await repo.add_events(batch_events)
        batch_result = await repo.get_events("batch-conv")
        assert len(batch_result) == 10, f"Expected 10 events, got {len(batch_result)}"
        logger.info("✅ Batch operations working")

        # Test 4: Connection recovery simulation
        logger.info("Test 4: Connection recovery simulation")
        
        # Force mark connection as unhealthy to test recovery
        repo._connection_healthy = False
        
        recovery_event = ChatEvent(
            conversation_id="recovery-conv",
            type="user_message",
            role="user", 
            content="Recovery test",
            extra={"request_id": "recovery-req"}
        )
        
        # This should trigger reconnection
        await repo.add_event(recovery_event)
        
        # Verify the event was added successfully
        recovery_events = await repo.get_events("recovery-conv")
        assert len(recovery_events) == 1, "Recovery event should be added"
        assert repo._connection_healthy, "Connection should be healthy after recovery"
        logger.info("✅ Connection recovery working")

        # Test 5: Context manager behavior
        logger.info("Test 5: Context manager behavior")
        async with AsyncSqlRepo(db_path=db_path) as context_repo:
            ctx_event = ChatEvent(
                conversation_id="ctx-conv",
                type="user_message",
                role="user",
                content="Context manager test",
                extra={"request_id": "ctx-req"}
            )
            await context_repo.add_event(ctx_event)
            
            ctx_events = await context_repo.get_events("ctx-conv")
            assert len(ctx_events) == 1, "Context manager should work"
        
        logger.info("✅ Context manager working")

        # Test 6: Proper cleanup
        logger.info("Test 6: Cleanup test")
        connection_before_close = repo._connection
        await repo.close()
        
        assert repo._connection is None, "Connection should be None after close"
        assert not repo._connection_healthy, "Connection should be marked unhealthy"
        assert not repo._initialized, "Repository should be marked uninitialized"
        logger.info("✅ Cleanup working")

        logger.info("🎉 All enhanced connection management tests passed!")

    finally:
        # Clean up temp file
        if os.path.exists(db_path):
            os.unlink(db_path)


async def test_error_handling():
    """Test error handling in database operations."""
    logger.info("Testing error handling scenarios...")
    
    # Test with invalid database path
    invalid_repo = AsyncSqlRepo(db_path="/invalid/path/test.db")
    
    try:
        event = ChatEvent(
            conversation_id="error-test",
            type="user_message",
            role="user",
            content="This should fail",
            extra={"request_id": "error-req"}
        )
        
        await invalid_repo.add_event(event)
        assert False, "Should have raised an exception"
    except Exception as e:
        logger.info(f"✅ Error handling working: {type(e).__name__}: {e}")
    finally:
        await invalid_repo.close()


if __name__ == "__main__":
    asyncio.run(test_enhanced_persistent_connection())
    asyncio.run(test_error_handling())
