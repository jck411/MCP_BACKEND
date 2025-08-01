#!/usr/bin/env python3
"""
Test script to verify cross-process file locking in AsyncJsonlRepo.

This script demonstrates that the new filelock-based implementation provides
robust cross-process safety by simulating concurrent file access from multiple
processes.
"""
import asyncio
import multiprocessing
import tempfile
import uuid
from pathlib import Path

from src.history.chat_store import AsyncJsonlRepo, ChatEvent


async def write_events_concurrently(repo_path: str, process_id: int, event_count: int):
    """Write events concurrently from a single process."""
    repo = AsyncJsonlRepo(repo_path)
    
    for i in range(event_count):
        event = ChatEvent(
            conversation_id=f"conv-{process_id}",
            type="user_message",
            role="user",
            content=f"Message {i} from process {process_id}",
            extra={"request_id": f"req-{process_id}-{i}"}
        )
        success = await repo.add_event(event)
        assert success, f"Failed to add event {i} from process {process_id}"
        
        # Small delay to increase chance of race conditions
        await asyncio.sleep(0.001)
    
    print(f"Process {process_id} completed writing {event_count} events")


def run_async_writer(repo_path: str, process_id: int, event_count: int):
    """Wrapper to run async function in a separate process."""
    asyncio.run(write_events_concurrently(repo_path, process_id, event_count))


async def test_cross_process_locking():
    """Test that cross-process file locking prevents corruption."""
    print("🧪 Testing cross-process file locking...")
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
        temp_path = f.name
    
    try:
        # Start multiple processes that will write concurrently
        num_processes = 4
        events_per_process = 10
        
        processes = []
        for i in range(num_processes):
            p = multiprocessing.Process(
                target=run_async_writer,
                args=(temp_path, i, events_per_process)
            )
            processes.append(p)
            p.start()
        
        # Wait for all processes to complete
        for p in processes:
            p.join()
            assert p.exitcode == 0, f"Process failed with exit code {p.exitcode}"
        
        # Verify that all events were written correctly
        repo = AsyncJsonlRepo(temp_path)
        
        total_expected_events = num_processes * events_per_process
        actual_events = []
        
        for process_id in range(num_processes):
            conv_events = await repo.get_events(f"conv-{process_id}")
            actual_events.extend(conv_events)
        
        print(f"Expected {total_expected_events} events, got {len(actual_events)}")
        assert len(actual_events) == total_expected_events, (
            f"Expected {total_expected_events} events, got {len(actual_events)}"
        )
        
        # Verify file integrity by checking that each line is valid JSON
        with open(temp_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            assert len(lines) == total_expected_events, (
                f"File should have {total_expected_events} lines, got {len(lines)}"
            )
            
            # Each line should be valid JSON
            import json
            for i, line in enumerate(lines):
                try:
                    json.loads(line.strip())
                except json.JSONDecodeError as e:
                    raise AssertionError(f"Line {i+1} is not valid JSON: {e}")
        
        print("  ✅ Cross-process locking prevented file corruption")
        print("  ✅ All events were written correctly")
        print("  ✅ File integrity maintained across processes")
        
    finally:
        Path(temp_path).unlink(missing_ok=True)
        # Clean up lock file if it exists
        Path(f"{temp_path}.lock").unlink(missing_ok=True)


if __name__ == "__main__":
    asyncio.run(test_cross_process_locking())
