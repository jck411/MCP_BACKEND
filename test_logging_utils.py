#!/usr/bin/env python3
"""
Test script for logging utilities.

This validates that the centralized logging and error handling works correctly.
"""

import asyncio
import pytest
from unittest.mock import Mock, patch
from mcp import McpError, types
from pydantic import ValidationError

from src.logging_utils import (
    log_mcp_operation,
    handle_mcp_errors,
    log_operation,
    operation_context,
    safe_mcp_call,
    MCPErrorHandler,
    ContextualLogger,
)


class TestMCPErrorHandler:
    """Test the MCPErrorHandler class."""
    
    def test_classify_mcp_error(self):
        """Test classification of McpError."""
        mcp_error = McpError(
            error=types.ErrorData(code=types.PARSE_ERROR, message="Parse failed")
        )
        code, category = MCPErrorHandler.classify_error(mcp_error)
        assert code == types.PARSE_ERROR
        assert category == "mcp_error"
    
    def test_classify_validation_error(self):
        """Test classification of ValidationError."""
        validation_error = ValidationError.from_exception_data(
            "ValidationError", [{"type": "missing", "loc": ("field",), "msg": "Field required"}]
        )
        code, category = MCPErrorHandler.classify_error(validation_error)
        assert code == types.INVALID_PARAMS
        assert category == "validation_error"
    
    def test_classify_connection_error(self):
        """Test classification of ConnectionError."""
        conn_error = ConnectionError("Network unreachable")
        code, category = MCPErrorHandler.classify_error(conn_error)
        assert code == types.INTERNAL_ERROR
        assert category == "connection_error"
    
    def test_classify_value_error(self):
        """Test classification of ValueError."""
        value_error = ValueError("Invalid parameter")
        code, category = MCPErrorHandler.classify_error(value_error)
        assert code == types.INVALID_PARAMS
        assert category == "parameter_error"
    
    def test_classify_unknown_error(self):
        """Test classification of unknown error type."""
        runtime_error = RuntimeError("Unknown error")
        code, category = MCPErrorHandler.classify_error(runtime_error)
        assert code == types.INTERNAL_ERROR
        assert category == "unknown_error"
    
    def test_create_mcp_error_with_context(self):
        """Test creating MCP error with context."""
        original_error = ValueError("Test error")
        context = {"user_id": "123", "action": "test"}
        
        mcp_error = MCPErrorHandler.create_mcp_error(
            original_error, "test_operation", context, "Custom message"
        )
        
        assert isinstance(mcp_error, McpError)
        assert mcp_error.error.code == types.INVALID_PARAMS
        assert mcp_error.error.message == "Custom message"
        assert mcp_error.error.data["operation"] == "test_operation"
        assert mcp_error.error.data["user_id"] == "123"
        assert mcp_error.error.data["action"] == "test"


class TestDecorators:
    """Test logging and error handling decorators."""
    
    @pytest.mark.asyncio
    async def test_log_operation_success(self):
        """Test log_operation decorator with successful function."""
        
        @log_operation("test_operation", log_timing=True)
        async def successful_function():
            return "success"
        
        result = await successful_function()
        assert result == "success"
    
    @pytest.mark.asyncio
    async def test_log_operation_with_error(self):
        """Test log_operation decorator with function that raises error."""
        
        @log_operation("test_operation", log_timing=True)
        async def failing_function():
            raise ValueError("Test error")
        
        with pytest.raises(ValueError, match="Test error"):
            await failing_function()
    
    @pytest.mark.asyncio
    async def test_handle_mcp_errors_converts_exception(self):
        """Test handle_mcp_errors converts regular exceptions to McpError."""
        
        @handle_mcp_errors("test_operation")
        async def failing_function():
            raise ValueError("Test error")
        
        with pytest.raises(McpError) as exc_info:
            await failing_function()
        
        assert exc_info.value.error.code == types.INVALID_PARAMS
        assert "test_operation" in exc_info.value.error.data["operation"]
    
    @pytest.mark.asyncio
    async def test_handle_mcp_errors_preserves_mcp_error(self):
        """Test handle_mcp_errors preserves existing McpError."""
        original_error = McpError(
            error=types.ErrorData(code=types.PARSE_ERROR, message="Parse failed")
        )
        
        @handle_mcp_errors("test_operation", reraise_mcp_errors=True)
        async def failing_function():
            raise original_error
        
        with pytest.raises(McpError) as exc_info:
            await failing_function()
        
        # Should be the same error, not wrapped
        assert exc_info.value.error.code == types.PARSE_ERROR
        assert exc_info.value.error.message == "Parse failed"
    
    @pytest.mark.asyncio
    async def test_log_mcp_operation_combined(self):
        """Test log_mcp_operation combines logging and error handling."""
        
        @log_mcp_operation("test_operation", log_timing=True)
        async def failing_function():
            raise ValueError("Test error")
        
        with pytest.raises(McpError) as exc_info:
            await failing_function()
        
        assert exc_info.value.error.code == types.INVALID_PARAMS
        assert "test_operation" in exc_info.value.error.data["operation"]


class TestContextManager:
    """Test operation context manager."""
    
    @pytest.mark.asyncio
    async def test_operation_context_success(self):
        """Test operation context manager with successful operation."""
        
        async with operation_context("test_operation", log_timing=True) as logger:
            assert logger is not None
            # Should not raise any exception
        
        # Test passes if no exception is raised
    
    @pytest.mark.asyncio
    async def test_operation_context_with_error(self):
        """Test operation context manager with failing operation."""
        
        with pytest.raises(ValueError, match="Test error"):
            async with operation_context("test_operation", log_timing=True):
                raise ValueError("Test error")


class TestSafeMCPCall:
    """Test safe_mcp_call utility function."""
    
    @pytest.mark.asyncio
    async def test_safe_mcp_call_success(self):
        """Test safe_mcp_call with successful coroutine."""
        
        async def successful_coro():
            return "success"
        
        result = await safe_mcp_call("test_operation", successful_coro())
        assert result == "success"
    
    @pytest.mark.asyncio
    async def test_safe_mcp_call_with_error(self):
        """Test safe_mcp_call with failing coroutine."""
        
        async def failing_coro():
            raise ValueError("Test error")
        
        with pytest.raises(McpError) as exc_info:
            await safe_mcp_call("test_operation", failing_coro())
        
        assert exc_info.value.error.code == types.INVALID_PARAMS
        assert "test_operation" in exc_info.value.error.data["operation"]


class TestContextualLogger:
    """Test ContextualLogger class."""
    
    def test_contextual_logger_initialization(self):
        """Test ContextualLogger initialization."""
        context = {"client": "test_client", "session": "123"}
        logger = ContextualLogger(context)
        assert logger.base_context == context
    
    def test_contextual_logger_bind(self):
        """Test ContextualLogger bind method."""
        base_context = {"client": "test_client"}
        logger = ContextualLogger(base_context)
        
        bound_logger = logger.bind(session="123", user="test_user")
        expected_context = {"client": "test_client", "session": "123", "user": "test_user"}
        assert bound_logger.base_context == expected_context
    
    def test_contextual_logger_methods(self):
        """Test ContextualLogger logging methods don't raise exceptions."""
        logger = ContextualLogger({"client": "test"})
        
        # These should not raise exceptions
        logger.info("Test info message", extra="data")
        logger.warning("Test warning message", extra="data")
        logger.error("Test error message", extra="data")
        logger.debug("Test debug message", extra="data")


async def run_manual_tests():
    """Run manual tests for demonstration."""
    print("🧪 Running Logging Utils Tests")
    print("=" * 40)
    
    # Test error classification
    print("\n1. Testing error classification:")
    test_errors = [
        ValueError("Invalid parameter"),
        ConnectionError("Network unreachable"),
        McpError(error=types.ErrorData(code=types.PARSE_ERROR, message="Parse failed")),
    ]
    
    for error in test_errors:
        code, category = MCPErrorHandler.classify_error(error)
        print(f"   {type(error).__name__} → {category} (code: {code})")
    
    # Test decorator functionality
    print("\n2. Testing decorators:")
    
    @log_mcp_operation("test_decorated_function")
    async def test_function():
        return "success"
    
    result = await test_function()
    print(f"   ✅ Decorated function result: {result}")
    
    # Test error handling
    print("\n3. Testing error handling:")
    
    @handle_mcp_errors("test_error_handling")
    async def failing_function():
        raise ValueError("Intentional test error")
    
    try:
        await failing_function()
    except McpError as e:
        print(f"   ✅ Error properly converted to McpError")
        print(f"      Code: {e.error.code}, Category: {e.error.data.get('error_category')}")
    
    print("\n" + "=" * 40)
    print("✅ All manual tests completed successfully!")


if __name__ == "__main__":
    # Run manual tests
    asyncio.run(run_manual_tests())
    
    # Run pytest if available
    try:
        pytest.main([__file__, "-v"])
    except ImportError:
        print("\n📝 Note: Install pytest to run automated tests: uv add pytest")
