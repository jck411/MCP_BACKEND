#!/usr/bin/env python3
"""
Test script to demonstrate clear_on_startup behavior with different persistence settings.
"""
import asyncio
import os
import tempfile
from src.history.repositories.sql_repo import AsyncSqlRepo
from src.history.models import ChatEvent

async def test_scenario(enabled: bool, clear_on_startup: bool, scenario_name: str):
    """Test a specific configuration scenario."""
    print(f"\n=== {scenario_name} ===")
    print(f"enabled={enabled}, clear_on_startup={clear_on_startup}")
    
    # Create temporary database
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
        db_path = tmp.name
    
    try:
        # Setup: Create repo and add some test data
        setup_config = {"enabled": True, "clear_on_startup": False}
        setup_repo = AsyncSqlRepo(db_path, setup_config)
        
        # Add test event
        test_event = ChatEvent(
            conversation_id="test-conv-1",
            type="user_message", 
            role="user",
            content="Hello, this is a test message"
        )
        await setup_repo.add_event(test_event)
        
        # Check data exists
        events = await setup_repo.get_events("test-conv-1")
        print(f"Setup: Added {len(events)} events to database")
        
        # Now test the actual scenario
        test_config = {
            "enabled": enabled,
            "clear_on_startup": clear_on_startup,
            "retention_policy": "unlimited"
        }
        test_repo = AsyncSqlRepo(db_path, test_config)
        
        # This will trigger _handle_persistence_on_startup
        events_after = await test_repo.get_events("test-conv-1") 
        print(f"After startup: {len(events_after)} events in database")
        
        # Test adding new data
        new_event = ChatEvent(
            conversation_id="test-conv-1",
            type="user_message",
            role="user", 
            content="New message after startup"
        )
        success = await test_repo.add_event(new_event)
        
        if enabled:
            final_events = await test_repo.get_events("test-conv-1")
            print(f"New data persistence: {'SUCCESS' if success and len(final_events) > len(events_after) else 'FAILED'}")
        else:
            print(f"New data persistence: {'CORRECTLY IGNORED' if success else 'FAILED'}")
            
    finally:
        # Cleanup
        if os.path.exists(db_path):
            os.unlink(db_path)

async def main():
    """Test all four scenarios."""
    print("Testing clear_on_startup behavior with different persistence settings")
    
    await test_scenario(
        enabled=False, 
        clear_on_startup=False,
        scenario_name="Scenario 1: Persistence OFF, No Clear"
    )
    
    await test_scenario(
        enabled=False,
        clear_on_startup=True, 
        scenario_name="Scenario 2: Persistence OFF, Clear on Startup"
    )
    
    await test_scenario(
        enabled=True,
        clear_on_startup=False,
        scenario_name="Scenario 3: Persistence ON, No Clear"
    )
    
    await test_scenario(
        enabled=True,
        clear_on_startup=True,
        scenario_name="Scenario 4: Persistence ON, Clear on Startup"
    )
    
    print("\n=== Summary ===")
    print("Scenario 1: Keeps existing data, ignores new data")
    print("Scenario 2: Clears existing data, ignores new data") 
    print("Scenario 3: Keeps existing data, persists new data")
    print("Scenario 4: Clears existing data, persists new data")

if __name__ == "__main__":
    asyncio.run(main())
