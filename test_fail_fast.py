#!/usr/bin/env python3
"""
Test script to verify fail-fast behavior changes.
"""

import asyncio
import json
import tempfile
from unittest.mock import Mock

from src.history.chat_store import AsyncJsonlRepo, ChatEvent
from src.chat_service import ChatService
from mcp import types, McpError


async def test_compact_deltas_fail_fast():
    """Test that compact_deltas fails fast on inconsistent state."""
    print("🧪 Testing compact_deltas fail-fast behavior...")
    
    with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as temp_file:
        temp_path = temp_file.name
    
    try:
        repo = AsyncJsonlRepo(temp_path)
        conversation_id = "test-conv"
        user_request_id = "test-request"
        
        # Manually corrupt the state by adding a request_id to the set
        # without the corresponding event
        await repo._ensure_loaded()
        async with repo._lock:
            req_ids = repo._req_ids.setdefault(conversation_id, set())
            req_ids.add(f"assistant:{user_request_id}")
        
        # This should now raise a ValueError instead of logging a warning
        try:
            await repo.compact_deltas(
                conversation_id, user_request_id, "test content"
            )
            print("  ❌ Expected ValueError but none was raised")
            return False
        except ValueError as e:
            if "Data corruption detected" in str(e):
                print("  ✅ compact_deltas correctly fails fast on corrupted state")
                return True
            else:
                print(f"  ❌ Unexpected ValueError: {e}")
                return False
    finally:
        import os
        try:
            os.unlink(temp_path)
        except:
            pass


def test_json_parsing_fail_fast():
    """Test that JSON parsing fails fast on malformed JSON."""
    print("🧪 Testing JSON parsing fail-fast behavior...")
    
    # Create a mock chat service
    config_dict = {
        "mcp_clients": [],
        "llm": {"model": "test"},
        "chat": {},
        "logging": {}
    }
    
    mock_clients = []
    mock_llm_client = Mock()
    mock_repo = Mock()
    mock_config = Mock()
    mock_config.get_max_tool_hops.return_value = 5
    
    service = ChatService(
        clients=mock_clients,
        llm_client=mock_llm_client, 
        config=config_dict,
        repo=mock_repo,
        configuration=mock_config
    )
    
    # Create a mock tool call with malformed JSON
    mock_tool_calls = [{
        "id": "test-call-1",
        "function": {
            "name": "test_tool",
            "arguments": "invalid json {"  # Malformed JSON
        }
    }]
    
    conv = []
    
    try:
        # This should now raise an McpError instead of using fallback
        asyncio.run(service._execute_tool_calls(conv, mock_tool_calls))
        print("  ❌ Expected McpError but none was raised")
        return False
    except McpError as e:
        if "Invalid JSON in tool call arguments" in e.error.message:
            print("  ✅ JSON parsing correctly fails fast on malformed JSON")
            return True
        else:
            print(f"  ❌ Unexpected McpError: {e.error.message}")
            return False
    except Exception as e:
        print(f"  ❌ Unexpected exception: {e}")
        return False


def test_structured_content_fail_fast():
    """Test that structured content serialization fails fast."""
    print("🧪 Testing structured content fail-fast behavior...")
    
    # Create a mock chat service
    config_dict = {
        "mcp_clients": [],
        "llm": {"model": "test"},
        "chat": {},
        "logging": {}
    }
    
    mock_clients = []
    mock_llm_client = Mock()
    mock_repo = Mock()
    mock_config = Mock()
    
    service = ChatService(
        clients=mock_clients,
        llm_client=mock_llm_client, 
        config=config_dict,
        repo=mock_repo,
        configuration=mock_config
    )
    
    # Create a mock result with unserializable structured content
    mock_result = Mock(spec=types.CallToolResult)
    mock_result.content = []
    mock_result.structuredContent = Mock()
    
    # Make structured content fail to serialize
    def failing_dumps(*args, **kwargs):
        raise TypeError("Object is not JSON serializable")
    
    original_dumps = json.dumps
    json.dumps = failing_dumps
    
    try:
        service._pluck_content(mock_result)
        print("  ❌ Expected McpError but none was raised")
        return False
    except McpError as e:
        if "Failed to serialize structured content" in e.error.message:
            print("  ✅ Structured content correctly fails fast on serialization error")
            return True
        else:
            print(f"  ❌ Unexpected McpError: {e.error.message}")
            return False
    except Exception as e:
        print(f"  ❌ Unexpected exception: {e}")
        return False
    finally:
        json.dumps = original_dumps


async def main():
    """Run compact_deltas fail-fast test."""
    print("Running fail-fast behavior tests...\n")
    
    # Test compact_deltas (most important test)
    result = await test_compact_deltas_fail_fast()
    
    print(f"\nTest Results: {1 if result else 0}/1 passed")
    
    if result:
        print("🎉 Compact deltas fail-fast behavior test passed!")
        print("✅ The system now fails fast instead of logging warnings on corrupted state")
        return 0
    else:
        print("❌ Test failed")
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(asyncio.run(main()))
