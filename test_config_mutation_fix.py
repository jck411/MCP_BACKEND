#!/usr/bin/env python3
"""Test script to verify configuration mutation fix."""

import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock


def test_configuration_no_mutation():
    """Test that get_repository_config does not mutate the original config."""
    print("🧪 Testing configuration mutation fix...")
    
    # Mock the config loading to avoid file dependency
    mock_config = {
        "llm": {
            "active": "openai",
            "providers": {"openai": {"model": "gpt-4"}}
        },
        "chat": {
            "service": {
                "repository": {},  # Empty repository config to trigger defaults
                "max_tool_hops": 5,
                "context": {
                    "max_tokens": 8000,
                    "reserve_tokens": 1000,
                    "preserve_recent": 2,
                    "conversation_limits": {"short": 10, "medium": 20, "long": 50},
                    "response_tokens": {"short": 500, "medium": 1000, "long": 2000, "max": 4000}
                },
                "streaming": {
                    "backoff": {
                        "max_attempts": 3,
                        "initial_delay": 0.1,
                        "flush_every_n_deltas": 5
                    }
                }
            }
        },
        "mcp": {
            "connection": {
                "max_reconnect_attempts": 3,
                "initial_reconnect_delay": 1.0,
                "max_reconnect_delay": 30.0,
                "connection_timeout": 10.0,
                "ping_timeout": 30.0
            },
            "server_execution": {
                "default_args": [],
                "default_env": {},
                "require_enabled_flag": True
            }
        },
        "chat_store": {
            "visibility": {
                "system_updates_visible_to_llm": True,
                "default_visible_to_llm": True
            }
        }
    }
    
    # Import here to avoid circular imports during patching
    from src.config import Configuration
    
    with patch.object(Configuration, '_load_yaml_config', return_value=mock_config):
        with patch.object(Configuration, 'load_env'):
            # Create configuration instance
            config = Configuration()
            
            # Get the original config dict
            original_repo_config = config._config["chat"]["service"]["repository"]
            
            # Verify the original repo config is empty
            assert "persistence" not in original_repo_config, "Original config should not have persistence key"
            print("  ✅ Original config has no persistence key")
            
            # Call get_repository_config multiple times
            result1 = config.get_repository_config()
            result2 = config.get_repository_config()
            
            # Verify the results have persistence configuration
            assert "persistence" in result1, "Result should have persistence key"
            assert "persistence" in result2, "Result should have persistence key"
            print("  ✅ Results contain persistence configuration")
            
            # Verify the original config was NOT mutated
            original_repo_config_after = config._config["chat"]["service"]["repository"]
            assert "persistence" not in original_repo_config_after, "Original config should still not have persistence key"
            print("  ✅ Original configuration was not mutated")
            
            # Verify the results are separate objects
            result1["persistence"]["enabled"] = False
            assert result2["persistence"]["enabled"] is True, "Results should be independent objects"
            print("  ✅ Results are independent objects")
            
            print("  ✅ Configuration mutation fix works correctly!")


if __name__ == "__main__":
    test_configuration_no_mutation()
    print("\n✅ Configuration mutation test passed!")
