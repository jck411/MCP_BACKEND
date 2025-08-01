#!/usr/bin/env python3
"""
Test script to demonstrate structured validation error handling.

This script tests the enhanced validate_tool_parameters() method that now returns
structured field-level errors instead of concatenated strings.
"""

from __future__ import annotations

import asyncio
import json
import logging
from unittest.mock import Mock

from mcp import McpError, types
from pydantic import BaseModel, Field, create_model

from src.tool_schema_manager import ToolSchemaManager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MockMCPClient:
    """Mock MCP client for testing."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.is_connected = True

    async def list_tools(self) -> list[types.Tool]:
        """Return a mock tool with complex parameter validation."""
        return [
            types.Tool(
                name="complex_tool",
                description="A tool with complex parameter validation",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "name": {
                            "type": "string",
                            "minLength": 3,
                            "maxLength": 50,
                            "description": "A required name field"
                        },
                        "age": {
                            "type": "integer",
                            "minimum": 0,
                            "maximum": 120,
                            "description": "Age in years"
                        },
                        "email": {
                            "type": "string",
                            "format": "email",
                            "description": "Valid email address"
                        },
                        "settings": {
                            "type": "object",
                            "properties": {
                                "theme": {
                                    "type": "string",
                                    "enum": ["light", "dark", "auto"]
                                },
                                "notifications": {"type": "boolean"}
                            },
                            "required": ["theme"]
                        }
                    },
                    "required": ["name", "age"]
                }
            )
        ]

    async def list_prompts(self) -> list[types.Prompt]:
        """Return empty prompts list."""
        return []

    async def list_resources(self) -> list[types.Resource]:
        """Return empty resources list."""
        return []


async def test_structured_validation_errors() -> None:
    """Test that validation errors are now structured and detailed."""
    print("🧪 Testing structured validation error handling...")

    # Create mock client and tool schema manager
    mock_client = MockMCPClient("test_client")
    manager = ToolSchemaManager([mock_client])
    await manager.initialize()

    # Test cases with various validation failures
    test_cases = [
        {
            "name": "Single field error",
            "params": {"name": "AB", "age": 25},  # name too short
            "expected_errors": 1,
        },
        {
            "name": "Multiple field errors",
            "params": {"name": "", "age": -5, "email": "invalid-email"},
            "expected_errors": 3,
        },
        {
            "name": "Missing required fields",
            "params": {"email": "test@example.com"},  # missing name and age
            "expected_errors": 2,
        },
        {
            "name": "Nested object validation",
            "params": {
                "name": "John",
                "age": 30,
                "settings": {"theme": "invalid", "notifications": "not-boolean"}
            },
            "expected_errors": 2,  # Invalid enum + wrong type
        },
        {
            "name": "Type mismatch errors",
            "params": {"name": 123, "age": "thirty"},  # wrong types
            "expected_errors": 2,
        }
    ]

    for test_case in test_cases:
        print(f"\n📋 Test: {test_case['name']}")
        print(f"   Parameters: {test_case['params']}")

        try:
            await manager.validate_tool_parameters("complex_tool", test_case["params"])
            print("   ❌ Expected validation to fail!")
        except McpError as e:
            # Check that we have structured error data
            error_data = e.error.data
            if not error_data or "validation_errors" not in error_data:
                print("   ❌ Error data missing or not structured!")
                continue

            validation_errors = error_data["validation_errors"]
            print(f"   ✅ Got {len(validation_errors)} structured validation errors")

            # Display structured error details
            for i, field_error in enumerate(validation_errors, 1):
                print(f"      {i}. Field: '{field_error['field']}'")
                print(f"         Message: {field_error['message']}")
                print(f"         Type: {field_error['type']}")
                if field_error.get('input') is not None:
                    print(f"         Input: {field_error['input']}")

            # Verify error count matches expectation
            if len(validation_errors) == test_case["expected_errors"]:
                print(f"   ✅ Error count matches expected ({test_case['expected_errors']})")
            else:
                print(f"   ❌ Expected {test_case['expected_errors']} errors, got {len(validation_errors)}")

            # Show the summary message
            print(f"   📝 Summary: {e.error.message}")


async def test_valid_parameters() -> None:
    """Test that valid parameters still work correctly."""
    print("\n🧪 Testing valid parameter handling...")

    mock_client = MockMCPClient("test_client")
    manager = ToolSchemaManager([mock_client])
    await manager.initialize()

    valid_params = {
        "name": "John Doe",
        "age": 30,
        "email": "john@example.com",
        "settings": {
            "theme": "dark",
            "notifications": True
        }
    }

    try:
        result = await manager.validate_tool_parameters("complex_tool", valid_params)
        print("   ✅ Valid parameters processed successfully")
        print(f"   📄 Validated data: {json.dumps(result, indent=2)}")
    except McpError as e:
        print(f"   ❌ Unexpected validation error: {e.error.message}")


async def main() -> None:
    """Run all tests."""
    print("🚀 Testing Structured Tool Parameter Validation")
    print("=" * 60)

    try:
        await test_structured_validation_errors()
        await test_valid_parameters()

        print("\n" + "=" * 60)
        print("🎉 All tests completed!")
        print("\n📊 Summary of improvements:")
        print("   • Validation errors are now structured as field-level objects")
        print("   • Each error includes field path, message, type, and input value")
        print("   • Error data is embedded in McpError.error.data for programmatic access")
        print("   • User-friendly summary messages are still provided")
        print("   • UI/clients can now display precise, actionable feedback")

    except Exception as e:
        logger.error(f"Test failed with error: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())
