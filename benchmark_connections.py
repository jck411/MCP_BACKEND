#!/usr/bin/env python3
"""
Benchmark script to measure connection overhead in the current AsyncSqlRepo implementation.
"""
import asyncio
import time
import tempfile
import os
from contextlib import asynccontextmanager

from src.history.models import ChatEvent
from src.history.repositories.sql_repo import AsyncSqlRepo


async def benchmark_current_implementation():
    """Benchmark the current persistent connection implementation."""
    # Use a temporary database file
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp:
        db_path = tmp.name

    try:
        repo = AsyncSqlRepo(db_path=db_path)
        
        # Warmup - initialize the connection
        await repo._initialize()
        
        num_operations = 100
        events = []
        
        # Create test events
        for i in range(num_operations):
            event = ChatEvent(
                conversation_id="bench-conv",
                type="user_message",
                role="user",
                content=f"Message {i}",
                extra={"request_id": f"req-{i}"}
            )
            events.append(event)
        
        # Benchmark individual add_event operations
        print("Benchmarking individual add_event operations...")
        start_time = time.perf_counter()
        
        for event in events:
            await repo.add_event(event)
        
        individual_time = time.perf_counter() - start_time
        print(f"Individual operations: {individual_time:.4f}s for {num_operations} ops")
        print(f"Avg per operation: {individual_time/num_operations*1000:.2f}ms")
        
        # Clear events for next test
        await repo.clear_all_conversations()
        
        # Benchmark batch add_events operation
        print("\nBenchmarking batch add_events operation...")
        start_time = time.perf_counter()
        
        await repo.add_events(events)
        
        batch_time = time.perf_counter() - start_time
        print(f"Batch operation: {batch_time:.4f}s for {num_operations} ops")
        print(f"Avg per operation: {batch_time/num_operations*1000:.2f}ms")
        
        # Benchmark read operations
        print("\nBenchmarking read operations...")
        start_time = time.perf_counter()
        
        for i in range(50):  # Fewer reads as they should be faster
            await repo.get_events("bench-conv", limit=10)
        
        read_time = time.perf_counter() - start_time
        print(f"Read operations: {read_time:.4f}s for 50 reads")
        print(f"Avg per read: {read_time/50*1000:.2f}ms")
        
        # Test connection persistence
        print("\nTesting connection persistence...")
        assert repo._connection is not None, "Connection should be persistent"
        connection_id = id(repo._connection)
        
        # Perform more operations and check if connection is the same
        await repo.add_event(ChatEvent(
            conversation_id="test-persistence",
            type="user_message", 
            role="user",
            content="Test persistence",
            extra={"request_id": "persistence-test"}
        ))
        
        assert repo._connection is not None, "Connection should still exist"
        assert id(repo._connection) == connection_id, "Connection should be the same object"
        print("✅ Connection persistence verified")
        
        # Cleanup
        await repo.close()
        
    finally:
        # Clean up temp file
        if os.path.exists(db_path):
            os.unlink(db_path)


async def benchmark_connection_churn():
    """Benchmark what happens if we recreate the repository each time (connection churn)."""
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp:
        db_path = tmp.name

    try:
        num_operations = 50  # Fewer operations since this is slower
        
        print(f"\nBenchmarking connection churn ({num_operations} operations)...")
        start_time = time.perf_counter()
        
        for i in range(num_operations):
            # Create new repo instance each time (simulating connection churn)
            repo = AsyncSqlRepo(db_path=db_path)
            
            event = ChatEvent(
                conversation_id="churn-conv",
                type="user_message",
                role="user", 
                content=f"Message {i}",
                extra={"request_id": f"churn-req-{i}"}
            )
            
            await repo.add_event(event)
            await repo.close()  # Close each time
        
        churn_time = time.perf_counter() - start_time
        print(f"Connection churn: {churn_time:.4f}s for {num_operations} ops")
        print(f"Avg per operation: {churn_time/num_operations*1000:.2f}ms")
        
    finally:
        if os.path.exists(db_path):
            os.unlink(db_path)


if __name__ == "__main__":
    asyncio.run(benchmark_current_implementation())
    asyncio.run(benchmark_connection_churn())
