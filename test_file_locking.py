#!/usr/bin/env python3
"""
Simple test to verify that the async file locking mechanism works.
"""
import asyncio
import tempfile
from pathlib import Path

from src.history.chat_store import async_file_lock


async def test_async_file_lock():
    """Test that the async file lock works correctly."""
    print("🧪 Testing async file locking...")
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.test', delete=False) as f:
        temp_path = f.name
    
    try:
        # Test that we can acquire and release the lock
        async with async_file_lock(temp_path):
            # Write some test data while holding the lock
            with open(temp_path, 'w') as f:
                f.write("test data\n")
                
        # Verify the data was written
        with open(temp_path) as f:
            content = f.read()
            assert content == "test data\n", f"Expected 'test data\\n', got {repr(content)}"
            
        print("  ✅ File lock acquired and released successfully")
        print("  ✅ Data written correctly while holding lock")
        
        # Test concurrent lock attempts
        lock_acquired_count = 0
        
        async def try_acquire_lock():
            nonlocal lock_acquired_count
            async with async_file_lock(temp_path):
                lock_acquired_count += 1
                await asyncio.sleep(0.1)  # Hold lock briefly
                
        # Run multiple concurrent lock attempts
        tasks = [try_acquire_lock() for _ in range(3)]
        await asyncio.gather(*tasks)
        
        assert lock_acquired_count == 3, f"Expected 3 locks acquired, got {lock_acquired_count}"
        print("  ✅ Concurrent lock acquisition handled correctly")
        
    finally:
        Path(temp_path).unlink(missing_ok=True)
        Path(f"{temp_path}.lock").unlink(missing_ok=True)


if __name__ == "__main__":
    asyncio.run(test_async_file_lock())
