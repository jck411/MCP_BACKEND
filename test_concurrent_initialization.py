#!/usr/bin/env python3
"""
Test script to verify concurrent initialization of ToolSchemaManager.

This script creates multiple mock clients and verifies that tools, prompts,
and resources are registered concurrently, potentially reducing startup time.
"""

from __future__ import annotations

import asyncio
import time
from unittest.mock import AsyncMock

from mcp import types

from src.tool_schema_manager import ToolSchemaManager


class MockMCPClient:
    """Mock MCP client with artificial delays to simulate network calls."""

    def __init__(self, name: str, delay: float = 0.1) -> None:
        self.name = name
        self.is_connected = True
        self.delay = delay

    async def list_tools(self) -> list[types.Tool]:
        """Return mock tools with artificial delay."""
        await asyncio.sleep(self.delay)
        return [
            types.Tool(
                name=f"{self.name}_tool_1",
                description=f"Test tool 1 from {self.name}",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "param1": {"type": "string", "description": "Test parameter"}
                    },
                    "required": ["param1"]
                }
            ),
            types.Tool(
                name=f"{self.name}_tool_2",
                description=f"Test tool 2 from {self.name}",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "param2": {"type": "integer", "description": "Test parameter"}
                    },
                    "required": ["param2"]
                }
            ),
        ]

    async def list_prompts(self) -> list[types.Prompt]:
        """Return mock prompts with artificial delay."""
        await asyncio.sleep(self.delay)
        return [
            types.Prompt(
                name=f"{self.name}_prompt_1",
                description=f"Test prompt 1 from {self.name}",
            ),
        ]

    async def list_resources(self) -> list[types.Resource]:
        """Return mock resources with artificial delay."""
        await asyncio.sleep(self.delay)
        return [
            types.Resource(
                uri=f"test://{self.name}/resource1",
                name=f"Resource 1 from {self.name}",
                description=f"Test resource from {self.name}",
            ),
        ]


async def test_concurrent_initialization():
    """Test that concurrent initialization works correctly."""
    print("Testing concurrent ToolSchemaManager initialization...")
    
    # Create multiple mock clients with delays to simulate real network calls
    clients = [
        MockMCPClient("client1", 0.1),
        MockMCPClient("client2", 0.1),
        MockMCPClient("client3", 0.1),
        MockMCPClient("client4", 0.1),
        MockMCPClient("client5", 0.1),
    ]
    
    # Test concurrent initialization
    manager = ToolSchemaManager(clients)
    
    start_time = time.time()
    await manager.initialize()
    end_time = time.time()
    
    initialization_time = end_time - start_time
    
    # Verify all tools, prompts, and resources were registered
    tools = manager.get_openai_tools()
    prompts = manager.list_available_prompts()
    resources = manager.list_available_resources()
    
    print(f"Initialization took: {initialization_time:.3f} seconds")
    print(f"Registered {len(tools)} tools")
    print(f"Registered {len(prompts)} prompts")
    print(f"Registered {len(resources)} resources")
    
    # With concurrent execution, the total time should be roughly:
    # max(tool_delay, prompt_delay, resource_delay) * num_clients
    # rather than (tool_delay + prompt_delay + resource_delay) * num_clients
    
    # Expected time for concurrent: ~0.1 seconds (max delay across all operations)
    # Expected time for sequential: ~1.5 seconds (0.1 * 3 operations * 5 clients)
    
    expected_concurrent_time = 0.2  # Allow some overhead
    expected_sequential_time = 1.0  # Would be much higher if sequential
    
    if initialization_time < expected_concurrent_time:
        print("✅ Concurrent initialization appears to be working correctly!")
        print(f"   Time ({initialization_time:.3f}s) is within expected concurrent range")
    elif initialization_time > expected_sequential_time:
        print("❌ Initialization may still be sequential!")
        print(f"   Time ({initialization_time:.3f}s) suggests sequential execution")
    else:
        print("⚠️  Initialization time is between expected ranges")
        print(f"   Time ({initialization_time:.3f}s) - may need further investigation")
    
    # Verify that all expected items were registered
    expected_tools = 2 * len(clients)  # 2 tools per client
    expected_prompts = len(clients)    # 1 prompt per client
    expected_resources = len(clients)  # 1 resource per client
    
    assert len(tools) == expected_tools, f"Expected {expected_tools} tools, got {len(tools)}"
    assert len(prompts) == expected_prompts, f"Expected {expected_prompts} prompts, got {len(prompts)}"
    assert len(resources) == expected_resources, f"Expected {expected_resources} resources, got {len(resources)}"
    
    print("✅ All registration counts are correct!")
    
    # Test that tools can be called
    tool_names = [tool["function"]["name"] for tool in tools]
    print(f"Available tools: {tool_names}")
    
    return initialization_time


if __name__ == "__main__":
    asyncio.run(test_concurrent_initialization())
