#!/usr/bin/env python3
"""
Migration example: Refactoring main.py MCPClient methods to use centralized logging.

This script shows before/after comparisons for actual methods from the codebase.
"""

from __future__ import annotations

import asyncio
from typing import Any
from mcp import McpError, types
from src.logging_utils import log_mcp_operation, ContextualLogger


class RefactoredMCPClient:
    """Example of MCPClient with refactored error handling using logging utilities."""
    
    def __init__(self, name: str):
        self.name = name
        self.session = None  # Mock session
        # Create contextual logger for this client
        self.logger = ContextualLogger({"client_name": name})
    
    # BEFORE (Original pattern from main.py - 22 lines)
    async def old_list_tools_pattern(self) -> list[types.Tool]:
        """Original repetitive error handling pattern from main.py."""
        import logging
        
        if not self.session:
            raise McpError(
                error=types.ErrorData(
                    code=types.INTERNAL_ERROR,
                    message=f"Client {self.name} not connected",
                )
            )

        try:
            result = await self._mock_session_call("list_tools")
            return result.get("tools", [])
        except McpError as e:
            logging.error(
                f"MCP error listing tools from {self.name}: {e.error.message}"
            )
            raise
        except Exception as e:
            logging.error(f"Error listing tools from {self.name}: {e}")
            raise McpError(
                error=types.ErrorData(
                    code=types.INTERNAL_ERROR,
                    message=f"Failed to list tools: {e!s}",
                )
            ) from e
    
    # AFTER (Refactored with centralized logging - 8 lines)
    @log_mcp_operation("list_tools", log_timing=True, context={"client": "filesystem"})
    async def list_tools(self) -> list[types.Tool]:
        """Refactored method using centralized logging and error handling."""
        if not self.session:
            raise McpError(
                error=types.ErrorData(
                    code=types.INTERNAL_ERROR,
                    message=f"Client {self.name} not connected",
                )
            )
        
        result = await self._mock_session_call("list_tools")
        return result.get("tools", [])
    
    # BEFORE (Original call_tool pattern - 25 lines)
    async def old_call_tool_pattern(
        self, name: str, arguments: dict[str, Any] | None = None
    ) -> types.CallToolResult:
        """Original call_tool method with repetitive error handling."""
        import logging
        
        if not self.session:
            raise McpError(
                error=types.ErrorData(
                    code=types.INTERNAL_ERROR,
                    message=f"Client {self.name} not connected",
                )
            )

        try:
            logging.info(f"Calling tool '{name}' on client '{self.name}'")
            result = await self._mock_session_call("call_tool", name, arguments)
            logging.info(f"Tool '{name}' executed successfully")
            return result
        except McpError as e:
            logging.error(f"MCP error calling tool '{name}': {e.error.message}")
            raise
        except Exception as e:
            logging.error(f"Error calling tool '{name}': {e}")
            raise McpError(
                error=types.ErrorData(
                    code=types.INTERNAL_ERROR,
                    message=f"Failed to call tool '{name}': {e!s}",
                )
            ) from e
    
    # AFTER (Refactored call_tool - 8 lines)
    @log_mcp_operation("call_tool", log_timing=True)
    async def call_tool(
        self, name: str, arguments: dict[str, Any] | None = None
    ) -> types.CallToolResult:
        """Refactored call_tool method."""
        if not self.session:
            raise McpError(
                error=types.ErrorData(
                    code=types.INTERNAL_ERROR,
                    message=f"Client {self.name} not connected",
                )
            )
        
        # The decorator handles all the logging and error wrapping
        return await self._mock_session_call("call_tool", name, arguments)
    
    async def _mock_session_call(self, method: str, *args, **kwargs):
        """Mock session call for demonstration."""
        import random
        
        # Simulate random failures for demonstration
        if random.random() < 0.3:  # 30% failure rate
            if method == "list_tools":
                raise ConnectionError("Failed to connect to MCP server")
            elif method == "call_tool":
                raise ValueError(f"Tool '{args[0]}' not found")
        
        # Simulate success responses
        if method == "list_tools":
            return {"tools": [{"name": "search"}, {"name": "calculator"}]}
        elif method == "call_tool":
            return types.CallToolResult(
                content=[types.TextContent(type="text", text="Tool executed successfully")]
            )


async def demonstrate_migration():
    """Demonstrate the migration from old to new patterns."""
    print("🔄 MCPClient Migration Example")
    print("=" * 50)
    
    client = RefactoredMCPClient("demo_client")
    client.session = "mock_session"  # Simulate connected session
    
    print("\n1. Testing old vs new list_tools patterns:")
    
    # Test old pattern
    print("\n   📜 Old Pattern (22 lines of boilerplate):")
    try:
        result = await client.old_list_tools_pattern()
        print(f"   ✅ Success: Found {len(result)} tools")
    except Exception as e:
        print(f"   ❌ Error: {type(e).__name__}: {e}")
    
    # Test new pattern
    print("\n   ✨ New Pattern (8 lines, structured logging):")
    try:
        result = await client.list_tools()
        print(f"   ✅ Success: Found {len(result)} tools")
    except Exception as e:
        print(f"   ❌ Error: {type(e).__name__}: {e}")
        if hasattr(e, 'error') and hasattr(e.error, 'data'):
            print(f"   📊 Context: {e.error.data}")
    
    print("\n2. Testing call_tool patterns:")
    
    # Test old call_tool pattern
    print("\n   📜 Old Pattern (25 lines of boilerplate):")
    try:
        result = await client.old_call_tool_pattern("search", {"query": "test"})
        print(f"   ✅ Success: {result}")
    except Exception as e:
        print(f"   ❌ Error: {type(e).__name__}: {e}")
    
    # Test new call_tool pattern
    print("\n   ✨ New Pattern (8 lines, automatic logging):")
    try:
        result = await client.call_tool("search", {"query": "test"})
        print(f"   ✅ Success: {result}")
    except Exception as e:
        print(f"   ❌ Error: {type(e).__name__}: {e}")
        if hasattr(e, 'error') and hasattr(e.error, 'data'):
            print(f"   📊 Context: {e.error.data}")
    
    print("\n" + "=" * 50)
    print("📊 Migration Benefits:")
    print("   • Code reduction: 22 lines → 8 lines (64% less)")
    print("   • Consistent error handling across all methods")
    print("   • Structured logging with timing and context")
    print("   • Automatic error classification and MCP formatting")
    print("   • Built-in performance monitoring")
    print("   • Type-safe error codes")
    print("\n🎯 Migration Strategy:")
    print("   1. Start with high-frequency methods (list_tools, call_tool)")
    print("   2. Apply @log_mcp_operation decorator")
    print("   3. Remove manual try/except/logging boilerplate")
    print("   4. Test error paths to ensure proper handling")
    print("   5. Update all similar methods using the same pattern")


if __name__ == "__main__":
    asyncio.run(demonstrate_migration())
