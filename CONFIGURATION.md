# Configuration Reference

This document describes all configurable options in the MCP Backend system. After making changes to `src/config.yaml`, restart the server for changes to take effect.

## Overview

The system now exposes many previously hardcoded values through the configuration system, allowing users to tune behavior without editing code. All configurations have sensible defaults.

## Configuration Sections

### Context Management (`chat.service.context`)

Controls how conversations are managed within token limits.

```yaml
chat:
  service:
    context:
      # Maximum context window size in tokens
      max_tokens: 4000
      
      # Tokens to reserve for LLM response generation
      reserve_tokens: 500
      
      # Conversation length thresholds for response estimation
      conversation_limits:
        short: 100     # Short conversation threshold
        medium: 500    # Medium conversation threshold  
        long: 1500     # Long conversation threshold
      
      # Expected response token counts by conversation length
      response_tokens:
        short: 150     # Expected tokens for short responses
        medium: 300    # Expected tokens for medium responses
        long: 500      # Expected tokens for long responses
        max: 800       # Maximum response tokens for very long conversations
      
      # Conversation optimization settings
      preserve_recent: 5  # Number of recent messages to always preserve
```

**Usage Examples:**
- **Increase context window**: Set `max_tokens: 8000` for longer conversations
- **Reduce response overhead**: Set `reserve_tokens: 300` for more history
- **Adjust response estimation**: Modify `response_tokens` for your use case

### Streaming Configuration (`chat.service.streaming`)

Controls real-time message streaming behavior.

```yaml
chat:
  service:
    streaming:
      enabled: true  # Enable streaming by default (required setting)
```

**Important:** This setting is **required** and must be explicitly set. The system will fail fast if not configured.

### Tool Execution Configuration (`chat.service`)

Controls tool call behavior and limits.

```yaml
chat:
  service:
    # Maximum number of recursive tool calls to prevent infinite loops
    max_tool_hops: 8

    # Tool execution notifications
    tool_notifications:
      enabled: true
      show_args: true
      icon: "🔧"
      format: "{icon} Executing tool: {tool_name}"
      # Available placeholders: {icon}, {tool_name}, {tool_args}
```

**Usage Examples:**
- **Increase tool depth**: Set `max_tool_hops: 15` for complex workflows
- **Disable notifications**: Set `tool_notifications.enabled: false`
- **Custom notification format**: Modify `format` string with placeholders

### Logging Configuration (`chat.service.logging`)

Controls what gets logged during chat operations.

```yaml
chat:
  service:
    logging:
      tool_execution: true    # Log when tools are executed
      tool_results: true      # Log tool execution results
      result_truncate_length: 200  # Truncate tool results in logs
      system_prompt: true     # Log the generated system prompt during initialization
      llm_replies: true       # Log ALL LLM replies including internal ones
      llm_reply_truncate_length: 500  # Truncate length for LLM reply logs
```

**Usage Examples:**
- **Reduce log verbosity**: Set `llm_replies: false` and `tool_execution: false`
- **Increase log detail**: Set `result_truncate_length: 1000`
- **Debug system prompts**: Ensure `system_prompt: true`

### MCP Connection Configuration (`mcp.connection`)

Controls how the system connects to and manages MCP servers.

```yaml
mcp:
  connection:
    # Maximum number of reconnection attempts per server
    max_reconnect_attempts: 5
    
    # Initial delay between reconnection attempts (seconds)
    # Uses exponential backoff up to max_reconnect_delay
    initial_reconnect_delay: 1.0
    
    # Maximum delay between reconnection attempts (seconds)
    max_reconnect_delay: 30.0
    
    # Connection timeout for initial server connection (seconds)
    connection_timeout: 30.0
    
    # Ping timeout for connection health checks (seconds)
    ping_timeout: 10.0
```

**Usage Examples:**
- **More resilient connections**: Set `max_reconnect_attempts: 10`
- **Faster reconnection**: Set `initial_reconnect_delay: 0.5`
- **Longer timeouts**: Set `connection_timeout: 60.0` for slow servers

## Configuration Validation

The system validates all configuration values on startup:

- **Token counts** must be positive integers
- **Conversation limits** must be in ascending order (short < medium < long)
- **Response tokens** must be in non-decreasing order
- **Connection timeouts** must be positive
- **Reconnect delays** must have max >= initial

Invalid configurations will cause the system to fail fast with clear error messages.

## Required Configuration Values

All configuration values must be explicitly set in `src/config.yaml`. The system will not start with missing configuration values. Here are the required values with their purposes:

| Configuration Path | Purpose | Example |
|-------------------|---------|---------|
| `context.max_tokens` | Context window size | 4000 |
| `context.reserve_tokens` | Reserved tokens for responses | 500 |
| `context.conversation_limits.short` | Short conversation threshold | 100 |
| `context.conversation_limits.medium` | Medium conversation threshold | 500 |
| `context.conversation_limits.long` | Long conversation threshold | 1500 |
| `context.preserve_recent` | Recent messages to preserve | 5 |
| `context.response_tokens.short` | Short response token estimate | 150 |
| `context.response_tokens.medium` | Medium response token estimate | 300 |
| `context.response_tokens.long` | Long response token estimate | 500 |
| `context.response_tokens.max` | Maximum response token estimate | 800 |

## Examples

### High-Throughput Configuration

For handling many concurrent requests with smaller contexts:

```yaml
chat:
  service:
    context:
      max_tokens: 2000
      reserve_tokens: 200
      preserve_recent: 3
    max_tool_hops: 5
    logging:
      llm_replies: false
      tool_execution: false
```

### Deep Analysis Configuration

For complex analysis tasks requiring longer contexts:

```yaml
chat:
  service:
    context:
      max_tokens: 8000
      reserve_tokens: 800
      preserve_recent: 10
      response_tokens:
        max: 1200
    max_tool_hops: 15
```

### Development/Debug Configuration

For development with verbose logging:

```yaml
chat:
  service:
    logging:
      system_prompt: true
      llm_replies: true
      tool_execution: true
      tool_results: true
      result_truncate_length: 1000
      llm_reply_truncate_length: 1000
```

## Testing Your Configuration

Use the included demo script to verify your configuration:

```bash
uv run python config_demo.py
```

This will display all current configuration values and test that they work correctly.
