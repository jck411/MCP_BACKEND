#!/usr/bin/env python3
"""
Test explicit configuration requirements.
"""

import pytest
import tempfile
import yaml
from src.config import Configuration


def test_max_tool_hops_requires_explicit_config():
    """Test that max_tool_hops requires explicit configuration."""
    
    # Create a config without max_tool_hops
    config_without_hops = {
        "chat": {
            "service": {
                "context": {
                    "max_tokens": 4000,
                    "reserve_tokens": 500,
                    "conversation_limits": {
                        "short": 100,
                        "medium": 500,
                        "long": 1500
                    },
                    "response_tokens": {
                        "short": 150,
                        "medium": 300,
                        "long": 500,
                        "max": 800
                    },
                    "preserve_recent": 5
                }
            }
        }
    }
    
    # Create temporary config file without max_tool_hops
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(config_without_hops, f)
        temp_config_path = f.name
    
    # Temporarily replace the config path in Configuration class
    original_init = Configuration.__init__
    
    def mock_init(self):
        Configuration.load_env()
        with open(temp_config_path) as file:
            self._config = yaml.safe_load(file)
    
    Configuration.__init__ = mock_init
    
    try:
        config = Configuration()
        
        # This should raise ValueError because max_tool_hops is not configured
        with pytest.raises(ValueError, match="max_tool_hops must be explicitly configured"):
            config.get_max_tool_hops()
            
    finally:
        # Restore original __init__
        Configuration.__init__ = original_init
        import os
        os.unlink(temp_config_path)


def test_mcp_connection_config_requires_explicit_config():
    """Test that MCP connection config requires all parameters."""
    
    # Create a config missing some MCP connection parameters
    config_missing_params = {
        "mcp": {
            "connection": {
                "max_reconnect_attempts": 5,
                # Missing other required parameters
            }
        }
    }
    
    # Create temporary config file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(config_missing_params, f)
        temp_config_path = f.name
    
    # Temporarily replace the config path in Configuration class
    original_init = Configuration.__init__
    
    def mock_init(self):
        Configuration.load_env()
        with open(temp_config_path) as file:
            self._config = yaml.safe_load(file)
    
    Configuration.__init__ = mock_init
    
    try:
        config = Configuration()
        
        # This should raise ValueError because required parameters are missing
        with pytest.raises(ValueError, match="must be explicitly configured"):
            config.get_mcp_connection_config()
            
    finally:
        # Restore original __init__
        Configuration.__init__ = original_init
        import os
        os.unlink(temp_config_path)


def test_streaming_backoff_config_requires_explicit_config():
    """Test that streaming backoff config requires all parameters."""
    
    # Create a config missing backoff parameters
    config_missing_backoff = {
        "chat": {
            "service": {
                "streaming": {
                    # Missing backoff section
                }
            }
        }
    }
    
    # Create temporary config file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(config_missing_backoff, f)
        temp_config_path = f.name
    
    # Temporarily replace the config path in Configuration class
    original_init = Configuration.__init__
    
    def mock_init(self):
        Configuration.load_env()
        with open(temp_config_path) as file:
            self._config = yaml.safe_load(file)
    
    Configuration.__init__ = mock_init
    
    try:
        config = Configuration()
        
        # This should raise ValueError because backoff config is missing
        with pytest.raises(ValueError, match="streaming.backoff.* must be explicitly configured"):
            config.get_streaming_backoff_config()
            
    finally:
        # Restore original __init__
        Configuration.__init__ = original_init
        import os
        os.unlink(temp_config_path)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
