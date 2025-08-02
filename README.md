# MCP Backend Platform

Production-ready Model Context Protocol (MCP) platform with comprehensive support for **tools**, **prompts**, and **resources**. Built with fail-fast principles, explicit configuration, and robust error handling.

## ✨ Key Features

- **Full MCP Protocol Support**: Tools, prompts, resources with multi-server connections
- **Production Ready**: Cross-process safety, graceful shutdowns, resource leak prevention
- **Explicit Configuration**: Fail-fast validation with no implicit defaults
- **Structured Logging**: JSON logs with performance timing and error classification
- **Real-time Streaming**: WebSocket API with configurable streaming modes
- **Type Safety**: Comprehensive Pydantic validation throughout
- **Token Tracking**: Accurate cost monitoring with tiktoken integration

## 🚀 Quick Start

```bash
# Install dependencies
uv sync

# Set API key
export GROQ_API_KEY="your_key_here"

# Run system tests (recommended)
./run_tests.sh

# Start the platform
./run.sh
```

## 🧪 System Validation

Comprehensive test suite validates all core functionality:

```bash
./run_tests.sh
```

**Tests Include:**
- ChatEvent model with timezone-aware UTC timestamps
- Cross-process file locking for multi-deployment safety
- Structured validation errors with field-level feedback
- Repository implementations (InMemory and AsyncJsonl)
- Delta compaction and token counting accuracy
- Graceful shutdown and resource cleanup

## 📡 WebSocket API

**Connect:** `ws://localhost:8000/ws/chat`

**Message Format:**
```json
{
  "action": "chat",
  "request_id": "unique-request-id",  // REQUIRED - prevents duplicate processing
  "payload": {
    "text": "Your message here",
    "model": "llama-3.1-70b-versatile",  // Optional - uses config default
    "streaming": true                    // Optional - overrides config setting
  }
}
```

**Response Types:**
```json
// Processing started
{"request_id": "...", "status": "processing"}

// Streaming chunk
{"request_id": "...", "status": "chunk", "chunk": {"type": "text", "data": "..."}}

// Completion
{"request_id": "...", "status": "complete", "metadata": {"tokens": 42}}

// Structured error
{"request_id": "...", "status": "error", "error": {"code": 400, "message": "..."}}
```

## ⚙️ Configuration

### Required Configuration (`src/config.yaml`)

**All values must be explicitly configured - no defaults provided:**

```yaml
# LLM Provider (REQUIRED)
llm:
  active: "groq"  # "groq" | "openai" | "anthropic"
  providers:
    groq:
      api_key: "env:GROQ_API_KEY"
      model: "llama-3.1-70b-versatile"
      temperature: 0.7
      max_tokens: 4000
      top_p: 0.9

# Chat Service (REQUIRED)
chat:
  service:
    streaming:
      enabled: true               # REQUIRED: true/false - no fallback
    context:
      max_tokens: 4000           # Context window size
      reserve_tokens: 500        # Reserved for responses
      preserve_recent: 5         # Always keep recent messages
    max_tool_hops: 8            # Prevent infinite tool loops
    
# WebSocket Server (REQUIRED)
websocket:
  host: "localhost"             # Must be explicit
  port: 8000                    # Must be explicit

# MCP Connection (REQUIRED)
mcp:
  connection:
    max_reconnect_attempts: 5
    connection_timeout: 30.0
    initial_reconnect_delay: 1.0
```

### MCP Servers (`src/servers_config.json`)
```json
{
  "mcpServers": {
    "demo": {
      "enabled": true,
      "command": "uv",
      "args": ["run", "python", "Servers/demo_server.py"],
      "cwd": "/absolute/path/to/project",
      "env": {}
    }
  }
}
```

### Environment Variables
```bash
# API Keys (choose your provider)
export GROQ_API_KEY="your_groq_key"
export OPENAI_API_KEY="your_openai_key"
export ANTHROPIC_API_KEY="your_anthropic_key"

# Optional: Custom paths
export MCP_CONFIG_PATH="/path/to/config.yaml"
export MCP_SERVERS_CONFIG_PATH="/path/to/servers_config.json"
```

## 🛠️ Development Commands

```bash
# Start platform
./run.sh                              # Production start
uv run python src/main.py             # Direct start
uv run mcp-platform                   # Alternative entry point

# System management
uv sync --upgrade                     # Update dependencies
./run_tests.sh                        # Run comprehensive tests
DEBUG_TOKENS=1 uv run python src/main.py  # Debug with token logging

# Server development
uv run python Servers/demo_server.py  # Test individual server
cat events.jsonl | jq '.usage'        # Monitor token usage
cat events.jsonl | jq '.timestamp'    # View UTC timestamps
```

## � Error Handling

### Fail-Fast Philosophy
- **No Silent Failures**: All errors explicitly handled or propagated
- **Configuration Validation**: Startup fails with missing required config
- **Structured Errors**: Field-level validation with actionable messages
- **Clear Error Codes**: MCP-compliant error responses

### Example Error Response
```json
{
  "status": "error",
  "error": {
    "code": 400,
    "message": "Parameter validation failed",
    "data": {
      "validation_errors": [
        {"field": "temperature", "message": "Must be between 0 and 2", "type": "value_error"}
      ]
    }
  }
}
```

## � Production Features

### Cross-Process Safety
- **File Locking**: Prevents corruption in multi-process deployments
- **Atomic Operations**: Complete writes or rollback (no partial corruption)
- **Resource Management**: Proper cleanup of connections and file handles

### Graceful Shutdown
- **Signal Handling**: Proper SIGTERM/SIGINT handling for containers
- **Resource Cleanup**: All HTTP clients, WebSocket connections closed
- **Timeout Handling**: 5-second graceful shutdown with fallback

### Structured Logging
- **JSON Format**: Machine-readable logs for aggregation tools
- **Performance Timing**: Built-in operation timing and metrics
- **Error Classification**: Automatic categorization for monitoring

## 📁 Project Structure

```
src/
├── main.py                  # MCP client and entry point
├── chat_service.py          # Chat orchestration and LLM integration
├── websocket_server.py      # WebSocket communication layer
├── config.py               # Configuration management with validation
├── tool_schema_manager.py   # MCP to OpenAI schema conversion
├── logging_utils.py         # Centralized logging and error handling
└── history/
    ├── chat_store.py        # Event storage with cross-process locking
    ├── conversation_utils.py # Token optimization and context management
    └── token_counter.py     # Accurate token counting with tiktoken

Servers/                     # Your MCP server implementations
events.jsonl                 # Chat history with UTC timestamps
```

---

**Requirements:** Python 3.13+, explicit configuration required, `request_id` mandatory in WebSocket messages.

**Philosophy:** Fail-fast validation, explicit configuration, structured error handling, production-ready resource management.

