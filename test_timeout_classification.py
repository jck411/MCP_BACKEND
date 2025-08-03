#!/usr/bin/env python3
"""
Test script to verify TimeoutError classification fix.
This file will be deleted after verification.
"""

import asyncio
from src.logging_utils import MCPErrorHandler


def test_timeout_error_classification():
    """Test that TimeoutError is correctly classified as timeout_error."""
    timeout_error = TimeoutError("Connection timed out")
    error_code, error_category = MCPErrorHandler.classify_error(timeout_error)
    
    print(f"TimeoutError classification:")
    print(f"  Error: {timeout_error}")
    print(f"  Category: {error_category}")
    print(f"  Code: {error_code}")
    
    assert error_category == "timeout_error", f"Expected 'timeout_error', got '{error_category}'"
    print("✓ TimeoutError correctly classified as timeout_error")


def test_oserror_classification():
    """Test that OSError is still classified as connection_error."""
    os_error = OSError("Network unreachable")
    error_code, error_category = MCPErrorHandler.classify_error(os_error)
    
    print(f"\nOSError classification:")
    print(f"  Error: {os_error}")
    print(f"  Category: {error_category}")
    print(f"  Code: {error_code}")
    
    assert error_category == "connection_error", f"Expected 'connection_error', got '{error_category}'"
    print("✓ OSError correctly classified as connection_error")


def test_connection_error_classification():
    """Test that ConnectionError is still classified as connection_error."""
    conn_error = ConnectionError("Connection refused")
    error_code, error_category = MCPErrorHandler.classify_error(conn_error)
    
    print(f"\nConnectionError classification:")
    print(f"  Error: {conn_error}")
    print(f"  Category: {error_category}")
    print(f"  Code: {error_code}")
    
    assert error_category == "connection_error", f"Expected 'connection_error', got '{error_category}'"
    print("✓ ConnectionError correctly classified as connection_error")


if __name__ == "__main__":
    print("Testing error classification fix...")
    test_timeout_error_classification()
    test_oserror_classification() 
    test_connection_error_classification()
    print("\n🎉 All tests passed! TimeoutError classification fix is working correctly.")
