# Explicit Configuration Changes Summary

This document summarizes all the changes made to remove implicit defaults and require explicit configuration throughout the MCP Backend system.

## Overview

The changes enforce a **"no fallbacks"** policy where all configuration values must be explicitly specified in `config.yaml` or configuration files. This prevents unexpected behavior in different environments and ensures predictable configuration across all deployments.

## Configuration File Changes

### config.yaml Additions

Added the following new configuration sections:

#### 1. Streaming Backoff Configuration
```yaml
chat:
  service:
    streaming:
      backoff:
        max_attempts: 5        # Maximum number of retry attempts
        initial_delay: 0.05    # Initial backoff delay in seconds
        flush_every_n_deltas: 25  # Flush frequency for streaming deltas
```

#### 2. LLM Provider Response Parsing
```yaml
llm:
  providers:
    openai:
      # ... existing config ...
      choice_index: 0  # Default choice index for response parsing
    # Same for all other providers
```

#### 3. MCP Server Execution Configuration
```yaml
mcp:
  server_execution:
    default_args: []              # Default arguments for server commands
    default_env: {}               # Default environment variables
    require_enabled_flag: true    # Require explicit enabled flag
```

#### 4. Chat Event Store Configuration
```yaml
chat_store:
  visibility:
    system_updates_visible_to_llm: false  # System update visibility
    default_visible_to_llm: false         # Default visibility for new event types
```

## Code Changes

### 1. Configuration Class (`src/config.py`)

#### Removed Implicit Defaults:
- `get_max_tool_hops()` - Now requires explicit configuration
- `get_mcp_connection_config()` - Validates all required parameters are present
- `get_context_config()` - Requires all nested configuration sections

#### Added New Methods:
- `get_streaming_backoff_config()` - Retrieves streaming backoff parameters
- `get_server_execution_config()` - Retrieves server execution defaults
- `get_chat_store_config()` - Retrieves chat store visibility configuration

### 2. MCPClient Class (`src/main.py`)

#### Changes:
- Constructor now requires `connection_config` parameter - no defaults
- Added `server_execution_config` parameter for explicit server execution configuration
- Removed all `.get()` calls with default values in connection parameter setup
- Server args and env now use explicit configuration or raise ValueError

#### New Validation:
```python
# Before
self._max_reconnect_attempts: int = conn_config.get("max_reconnect_attempts", 5)

# After  
if not connection_config:
    raise ValueError("connection_config must be provided - no default connection parameters allowed")
```

### 3. LLMClient Class (`src/main.py`)

#### Changes:
- Constructor validates all required LLM parameters are present
- Removed all `.get()` calls with defaults for temperature, max_tokens, top_p
- Added validation for `choice_index` parameter in response parsing

#### New Validation:
```python
# Before
"temperature": self.config.get("temperature", 0.7)

# After
required_keys = ["base_url", "model", "temperature", "max_tokens", "top_p"]
for key in required_keys:
    if key not in config:
        raise ValueError(f"Required LLM configuration parameter '{key}' not found")
```

### 4. ChatService Class (`src/chat_service.py`)

#### Changes:
- Streaming backoff parameters now loaded from configuration
- Updated `_write_with_backoff()` function to accept configuration parameter
- All calls to `_write_with_backoff()` now pass `self.backoff_config`

#### Removed Constants:
```python
# Before (hardcoded)
_MAX_BACKOFF_ATTEMPTS = 5
_INITIAL_BACKOFF = 0.05
_FLUSH_EVERY_N_DELTAS = 25

# After (from config)
backoff_config = self.configuration.get_streaming_backoff_config()
```

### 5. WebSocket Server (`src/websocket_server.py`)

#### Changes:
- Host and port configuration now require explicit values
- Removed `.get()` calls with defaults

#### New Validation:
```python
# Before
host = self.config.get("websocket", {}).get("host", "localhost")

# After
if "host" not in websocket_config:
    raise ValueError("WebSocket host must be explicitly configured")
```

### 6. Chat Store (`src/history/chat_store.py`)

#### Changes:
- `_visible_to_llm()` function now accepts optional visibility configuration
- `_get_last_n_token_events()` function updated to pass visibility config
- Event visibility rules now use explicit configuration

## Server Configuration Changes

### servers_config.json Updates

Added explicit `env` configuration to all server definitions:

```json
{
  "demo": {
    "enabled": true,
    "command": "uv",
    "args": ["run", "python", "Servers/demo_server.py"],
    "cwd": "/home/jack/MCP_PLATFORM_FastMCP",
    "env": {}  // Added explicit empty environment
  }
}
```

## Error Messages

The system now provides clear error messages when configuration is missing:

```
ValueError: max_tool_hops must be explicitly configured in config.yaml under chat.service
ValueError: Required LLM configuration parameter 'temperature' not found. All LLM parameters must be explicitly configured.
ValueError: WebSocket host must be explicitly configured in config.yaml under chat.websocket.host
```

## Benefits

1. **Predictability**: No hidden defaults that vary between environments
2. **Explicitness**: All configuration values are visible in config files
3. **Validation**: Early detection of missing configuration
4. **Documentation**: Configuration requirements are self-documenting through error messages
5. **Maintainability**: No scattered default values throughout the codebase

## Migration Guide

When deploying these changes:

1. **Update config.yaml** with all new required sections
2. **Add explicit env configuration** to all MCP servers in servers_config.json
3. **Verify all LLM provider configurations** include required parameters
4. **Test configuration loading** before deploying to production

## Testing

Added `test_explicit_config.py` with tests that verify:
- Configuration loading fails when required parameters are missing
- Error messages are clear and actionable
- All new configuration methods work correctly

The changes maintain backward compatibility where possible while enforcing explicit configuration for new deployments.
