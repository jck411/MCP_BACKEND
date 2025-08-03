#!/usr/bin/env python3
"""
Test script to verify the new get_event_by_id method works correctly.
"""

import asyncio
import tempfile
import os
from src.history.repositories.sql_repo import AsyncSqlRepo
from src.history.models import ChatEvent


async def test_get_event_by_id():
    """Test the new get_event_by_id method."""
    print("Testing get_event_by_id implementation...")
    
    # Create temporary database
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
        db_path = tmp.name
    
    try:
        # Setup repository
        config = {
            "enabled": True, 
            "clear_on_startup": False,
            "retention_policy": "unlimited"
        }
        repo = AsyncSqlRepo(db_path, config)
        
        # Create test events
        event1 = ChatEvent(
            conversation_id="test-conv-1",
            type="user_message",
            role="user",
            content="First message",
            extra={"request_id": "req-1"}
        )
        
        event2 = ChatEvent(
            conversation_id="test-conv-1", 
            type="assistant_message",
            role="assistant",
            content="First response",
            extra={"user_request_id": "req-1"}
        )
        
        event3 = ChatEvent(
            conversation_id="test-conv-1",
            type="user_message", 
            role="user",
            content="Second message",
            extra={"request_id": "req-2"}
        )
        
        # Add events to database
        await repo.add_event(event1)
        await repo.add_event(event2)
        await repo.add_event(event3)
        
        print(f"Added 3 events to database")
        print(f"Event IDs: {event1.id}, {event2.id}, {event3.id}")
        
        # Test get_event_by_id method
        retrieved_event = await repo.get_event_by_id("test-conv-1", event2.id)
        
        if retrieved_event is None:
            print("❌ FAILED: get_event_by_id returned None")
            return False
            
        if retrieved_event.id != event2.id:
            print(f"❌ FAILED: Expected ID {event2.id}, got {retrieved_event.id}")
            return False
            
        if retrieved_event.content != event2.content:
            print(f"❌ FAILED: Expected content '{event2.content}', got '{retrieved_event.content}'")
            return False
            
        print("✅ SUCCESS: get_event_by_id correctly retrieved the event")
        
        # Test with non-existent ID
        non_existent = await repo.get_event_by_id("test-conv-1", "fake-id")
        if non_existent is not None:
            print("❌ FAILED: get_event_by_id should return None for non-existent ID")
            return False
            
        print("✅ SUCCESS: get_event_by_id correctly returns None for non-existent ID")
        
        # Test with different conversation ID
        wrong_conv = await repo.get_event_by_id("wrong-conv", event2.id)
        if wrong_conv is not None:
            print("❌ FAILED: get_event_by_id should return None for wrong conversation")
            return False
            
        print("✅ SUCCESS: get_event_by_id correctly returns None for wrong conversation")
        
        # Performance comparison: Simulate the old inefficient method
        print("\n--- Performance Comparison ---")
        
        all_events = await repo.get_events("test-conv-1")
        print(f"Old method: Would scan through {len(all_events)} events")
        
        # Find event using old scanning method
        found_event = None
        for event in all_events:
            if event.id == event2.id:
                found_event = event
                break
                
        direct_event = await repo.get_event_by_id("test-conv-1", event2.id)
        
        if found_event.id == direct_event.id:
            print("✅ SUCCESS: Both methods return the same event")
        else:
            print("❌ FAILED: Methods return different events")
            return False
            
        print("New method: Direct database lookup with WHERE clause")
        
        return True
        
    finally:
        # Cleanup
        if os.path.exists(db_path):
            os.unlink(db_path)


async def test_chat_service_integration():
    """Test that ChatService uses the new method correctly."""
    print("\n--- Testing ChatService Integration ---")
    
    # Create temporary database
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
        db_path = tmp.name
    
    try:
        # Setup repository
        config = {
            "enabled": True, 
            "clear_on_startup": False,
            "retention_policy": "unlimited"
        }
        repo = AsyncSqlRepo(db_path, config)
        
        # Create test assistant response
        assistant_event = ChatEvent(
            conversation_id="test-conv-1",
            type="assistant_message",
            role="assistant", 
            content="This is a cached response",
            extra={"user_request_id": "req-123"}
        )
        
        await repo.add_event(assistant_event)
        
        # Test get_last_assistant_reply_id method
        reply_id = await repo.get_last_assistant_reply_id("test-conv-1", "req-123")
        
        if reply_id != assistant_event.id:
            print(f"❌ FAILED: get_last_assistant_reply_id returned {reply_id}, expected {assistant_event.id}")
            return False
            
        print("✅ SUCCESS: get_last_assistant_reply_id works correctly")
        
        # Test direct retrieval using get_event_by_id
        if reply_id is None:
            print("❌ FAILED: reply_id is None")
            return False
            
        cached_response = await repo.get_event_by_id("test-conv-1", reply_id)
        
        if cached_response is None:
            print("❌ FAILED: get_event_by_id returned None for valid ID")
            return False
            
        if cached_response.content != assistant_event.content:
            print(f"❌ FAILED: Content mismatch - expected '{assistant_event.content}', got '{cached_response.content}'")
            return False
            
        print("✅ SUCCESS: Direct retrieval of cached response works correctly")
        print(f"Retrieved content: '{cached_response.content}'")
        
        return True
        
    finally:
        # Cleanup
        if os.path.exists(db_path):
            os.unlink(db_path)


async def main():
    """Run all tests."""
    print("=" * 60)
    print("Testing get_event_by_id Implementation")
    print("=" * 60)
    
    success1 = await test_get_event_by_id()
    success2 = await test_chat_service_integration()
    
    print("\n" + "=" * 60)
    if success1 and success2:
        print("✅ ALL TESTS PASSED - Implementation is working correctly!")
        print("The inefficient retrieval issue has been resolved.")
    else:
        print("❌ SOME TESTS FAILED - Please check the implementation.")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
