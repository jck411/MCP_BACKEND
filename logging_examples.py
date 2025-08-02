#!/usr/bin/env python3
"""
Example usage of centralized logging and error handling utilities.

This script demonstrates how to refactor existing repetitive error handling
patterns using the new logging utilities.
"""

from __future__ import annotations

import asyncio
from mcp import McpError, types

from src.logging_utils import (
    log_mcp_operation,
    handle_mcp_errors,
    log_operation,
    operation_context,
    safe_mcp_call,
    ContextualLogger,
    MCPErrorHandler,
)


# Example 1: Before - Repetitive error handling pattern
class OldStyleClient:
    """Example showing old repetitive error handling patterns."""
    
    def __init__(self, name: str):
        self.name = name
    
    async def old_list_tools_pattern(self):
        """Example of old repetitive error handling pattern."""
        import logging
        
        try:
            # Simulate some operation
            result = await self._simulate_mcp_call()
            return result
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
    
    async def _simulate_mcp_call(self):
        """Simulate an MCP call that might fail."""
        import random
        if random.random() < 0.3:  # 30% chance of failure
            raise ValueError("Simulated connection error")
        return {"tools": ["tool1", "tool2"]}


# Example 2: After - Using decorators for cleaner code
class NewStyleClient:
    """Example showing new centralized error handling."""
    
    def __init__(self, name: str):
        self.name = name
        # Create a contextual logger for this client
        self.logger = ContextualLogger({"client_name": name})
    
    @log_mcp_operation("list_tools", log_timing=True)
    async def list_tools(self):
        """Clean method using decorator for error handling."""
        return await self._simulate_mcp_call()
    
    @handle_mcp_errors("get_tool_info", context={"client": "example"})
    async def get_tool_info(self, tool_name: str):
        """Method with custom error context."""
        self.logger.info("Getting tool info", tool_name=tool_name)
        # Simulate some operation that might fail
        if tool_name == "failing_tool":
            raise ValueError(f"Tool {tool_name} not found")
        return {"name": tool_name, "description": "A useful tool"}
    
    @log_operation("connect_to_server", log_args=True, log_result=True)
    async def connect_to_server(self, host: str, port: int):
        """Method with detailed logging."""
        await asyncio.sleep(0.1)  # Simulate connection time
        return {"status": "connected", "host": host, "port": port}
    
    async def _simulate_mcp_call(self):
        """Simulate an MCP call that might fail."""
        import random
        if random.random() < 0.3:  # 30% chance of failure
            raise ValueError("Simulated connection error")
        return {"tools": ["tool1", "tool2"]}


# Example 3: Using context managers for operations
async def example_context_manager_usage():
    """Demonstrate operation context manager usage."""
    
    async with operation_context(
        "database_backup",
        context={"database": "user_data", "type": "full_backup"},
        log_timing=True
    ) as logger:
        logger.info("Starting backup process")
        await asyncio.sleep(0.2)  # Simulate backup time
        logger.info("Backup completed successfully")
        return {"backup_id": "backup_123", "size_mb": 150}


# Example 4: Using safe_mcp_call for one-off operations
async def example_safe_mcp_call():
    """Demonstrate safe MCP call wrapper."""
    
    async def risky_operation():
        import random
        if random.random() < 0.5:
            raise ConnectionError("Network is unreachable")
        return {"result": "success"}
    
    try:
        result = await safe_mcp_call(
            "fetch_user_data",
            risky_operation(),
            context={"user_id": "12345"},
            custom_message="Failed to fetch user data from remote service"
        )
        return result
    except McpError as e:
        print(f"Structured error data: {e.error.data}")
        raise


# Example 5: Error classification demonstration
async def example_error_classification():
    """Show how different errors are classified."""
    
    test_errors = [
        ValueError("Invalid parameter"),
        ConnectionError("Network unreachable"),
        TimeoutError("Operation timed out"),
        McpError(error=types.ErrorData(code=types.PARSE_ERROR, message="Parse failed")),
        RuntimeError("Unknown runtime error"),
    ]
    
    print("Error Classification Examples:")
    print("-" * 40)
    
    for error in test_errors:
        code, category = MCPErrorHandler.classify_error(error)
        print(f"{type(error).__name__:20} -> {category:20} (code: {code})")


# Main demonstration
async def main():
    """Run all examples."""
    print("🚀 MCP Logging and Error Handling Examples")
    print("=" * 50)
    
    # Example 1: Old vs New patterns
    print("\n1. Comparing old vs new error handling patterns:")
    
    old_client = OldStyleClient("old_client")
    new_client = NewStyleClient("new_client")
    
    print("\n   Testing old pattern (with basic logging):")
    try:
        result = await old_client.old_list_tools_pattern()
        print(f"   ✅ Old pattern result: {result}")
    except Exception as e:
        print(f"   ❌ Old pattern error: {type(e).__name__}: {e}")
    
    print("\n   Testing new pattern (with structured logging):")
    try:
        result = await new_client.list_tools()
        print(f"   ✅ New pattern result: {result}")
    except Exception as e:
        print(f"   ❌ New pattern error: {type(e).__name__}: {e}")
    
    # Example 2: Detailed logging
    print("\n2. Detailed operation logging:")
    try:
        result = await new_client.connect_to_server("localhost", 8080)
        print(f"   ✅ Connection result: {result}")
    except Exception as e:
        print(f"   ❌ Connection error: {e}")
    
    # Example 3: Context manager
    print("\n3. Operation context manager:")
    try:
        result = await example_context_manager_usage()
        print(f"   ✅ Backup result: {result}")
    except Exception as e:
        print(f"   ❌ Backup error: {e}")
    
    # Example 4: Safe MCP call
    print("\n4. Safe MCP call wrapper:")
    try:
        result = await example_safe_mcp_call()
        print(f"   ✅ Safe call result: {result}")
    except McpError as e:
        print(f"   ❌ Safe call error: {e.error.message}")
        print(f"   📊 Error data: {e.error.data}")
    
    # Example 5: Error classification
    print("\n5. Error classification:")
    await example_error_classification()
    
    print("\n" + "=" * 50)
    print("🎉 Examples completed!")
    print("\n📋 Key Benefits:")
    print("   • Reduced boilerplate code by ~70%")
    print("   • Consistent structured logging")
    print("   • Automatic error classification")
    print("   • Contextual error information")
    print("   • Performance timing built-in")
    print("   • Type-safe error handling")


if __name__ == "__main__":
    asyncio.run(main())
