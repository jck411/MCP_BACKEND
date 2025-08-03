# MCP Backend Platform

Production-ready Model Context Protocol (MCP) chatbot platform with persistent conversation memory using optimized SQLite storage.

## ✨ Key Features

- **Full MCP Protocol Support**: Tools, prompts, resources with multi-server connections
- **Persistent Memory**: High-performance SQLite database with WAL mode and lock-free operations
- **Real-time Streaming**: WebSocket API with configurable streaming modes
- **Production Ready**: Graceful shutdowns, automatic error recovery, structured logging
- **Type Safety**: Comprehensive Pydantic validation throughout
- **Token Tracking**: Usage monitoring with LRU-cached tiktoken integration

## � Quick Start

```bash
# Install dependencies
uv sync

# Set API key (choose one)
export GROQ_API_KEY="your_key_here"
export OPENAI_API_KEY="your_key_here"

# Start the platform
./run.sh
```

## �💾 Persistent Memory Configuration

Conversations are stored in `events.db` SQLite database with flexible retention policies:

### Memory Persistence Options
```yaml
# Repository configuration (SQLite database)
repository:
  type: "sql"
  path: "events.db"
  
  # Persistent memory configuration
  persistence:
    enabled: true                    # Enable/disable persistent memory
    retention_policy: "token_limit"  # How to limit memory retention
    
    # Token-based retention (recommended)
    max_tokens_per_conversation: 8000  # Max tokens to keep per conversation
    
    # Alternative retention policies
    max_messages_per_conversation: 100  # Max messages (when policy: "message_count")
    retention_days: 30                  # Days to keep (when policy: "time_based")
    
    # Session behavior
    clear_on_startup: false  # Clear all data on startup when persistence disabled
```

### Retention Policies
- **`token_limit`**: Keep recent messages within token budget (recommended)
- **`message_count`**: Keep only the N most recent messages
- **`time_based`**: Keep messages within specified days
- **`unlimited`**: Keep all messages (no cleanup)

### Session Behavior
- **`enabled: true`**: Conversations persist between restarts with retention policies applied
- **`enabled: false`**: Each session starts fresh (no memory between restarts)
- **`clear_on_startup: true`**: Force clear all data on startup regardless of persistence setting

### Performance Optimizations
- **SQLite WAL Mode**: Enables concurrent reads during writes
- **Lock-Free Operations**: Eliminated connection locks for 53-64% performance improvement
- **Automatic Reconnection**: Robust error recovery with exponential backoff
- **LRU Token Cache**: Bounded memory usage with 1024-entry cache
- **Connection Persistence**: Single connection per repository instance

## 📡 WebSocket API

**Connect:** `ws://localhost:8000/ws/chat`

**Message Format:**
```json
{
  "action": "chat",
  "request_id": "unique-request-id",
  "payload": {
    "text": "Your message here",
    "conversation_id": "optional-conversation-id"
  }
}
```

**Response Types:**
```json
// Streaming chunk
{"request_id": "...", "status": "chunk", "chunk": {"type": "text", "data": "..."}}

// Completion with usage
{"request_id": "...", "status": "complete", "metadata": {"tokens": 42}}
```

## ⚙️ Configuration

### Required Configuration (`src/config.yaml`)

```yaml
# LLM Provider
llm:
  active: "groq"  # "groq" | "openai" | "openrouter" | "anthropic"
  providers:
    groq:
      model: "llama-3.3-70b-versatile"
      temperature: 0.7
      max_tokens: 4096

# Chat Service
chat:
  service:
    streaming:
      enabled: true
    context:
      max_tokens: 4000      # Context window size
      reserve_tokens: 500   # Reserved for responses
    max_tool_hops: 8        # Tool call limit

# WebSocket Server
chat:
  websocket:
    host: "localhost"
    port: 8000
```

### MCP Servers (`src/servers_config.json`)
```json
{
  "mcpServers": {
    "filesystem": {
      "enabled": true,
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-filesystem", "/path/to/directory"]
    }
  }
}
```

## 🛠️ Development Commands

```bash
# Start platform
./run.sh                              # Production start
uv run python src/main.py             # Direct start

# System management
uv sync --upgrade                     # Update dependencies

# Database inspection
sqlite3 events.db "SELECT conversation_id, COUNT(*) FROM chat_events GROUP BY conversation_id;"
```

## 📁 Architecture

```
src/
├── main.py                  # MCP client initialization and entry point
├── chat_service.py          # Chat orchestration and LLM integration
├── websocket_server.py      # WebSocket communication layer
├── config.py               # Configuration management
├── tool_schema_manager.py   # MCP to OpenAI schema conversion
├── logging_utils.py         # Centralized error handling and logging
└── history/
    ├── models.py           # ChatEvent, Usage, ToolCall models
    ├── repositories/       # Storage backends
    │   ├── base.py        # Repository protocol
    │   └── sql_repo.py    # Optimized SQLite implementation
    ├── conversation_utils.py # Context management with token limits
    └── token_counter.py    # LRU-cached token counting

events.db                   # SQLite conversation storage
```

### Key Architectural Components

#### Storage Layer
- **AsyncSqlRepo**: High-performance SQLite implementation with WAL mode
- **ChatRepository Protocol**: Clean interface for storage backends
- **ChatEvent Model**: Unified event model for all chat interactions

#### Data Flow
1. **WebSocket Input**: Receives chat messages via WebSocket API
2. **Chat Service**: Orchestrates LLM interactions and tool calls
3. **SQL Storage**: Persists all events to optimized SQLite database
4. **Context Retrieval**: Loads conversation history within token limits

#### Database Schema
```sql
CREATE TABLE chat_events (
    id TEXT PRIMARY KEY,
    conversation_id TEXT NOT NULL,
    seq INTEGER,
    timestamp TEXT,
    type TEXT NOT NULL,
    role TEXT,
    content TEXT,
    tool_calls TEXT,
    tool_call_id TEXT,
    model TEXT,
    usage TEXT,
    extra TEXT,
    token_count INTEGER,
    request_id TEXT
);
```

## 🔧 Recent Optimizations

- **Connection Performance**: Removed connection locks, achieving 53-64% performance improvement
- **Memory Management**: Implemented LRU cache for token counting with bounded memory usage
- **Error Recovery**: Added automatic reconnection with exponential backoff
- **Code Cleanup**: Removed all legacy JSONL code and deprecated patterns

---

**Requirements:** Python 3.13+, WebSocket client for API access

**Philosophy:** Persistent memory, clean architecture, production-ready resource management.

## 📡 WebSocket APIsistent MemoryModel Context Protocol (MCP) chatbot platform with persistent conversation memory using SQLite storage.

## ✨ Key Features

- **Full MCP Protocol Support**: Tools, prompts, resources with multi-server connections
- **Persistent Memory**: SQLite database stores all conversations across sessions
- **Real-time Streaming**: WebSocket API with configurable streaming modes
- **Production Ready**: Graceful shutdowns, resource management, structured logging
- **Type Safety**: Comprehensive Pydantic validation throughout
- **Token Tracking**: Usage monitoring with tiktoken integration

## 🚀 Quick Start

```bash
# Install dependencies
uv sync

# Set API key (choose one)
export GROQ_API_KEY="your_key_here"
export OPENAI_API_KEY="your_key_here"

# Start the platform
./run.sh
```

## 🧪 System Validation

```bash
./run_tests.sh
```

**Tests Include:**
- ChatEvent model with timezone-aware UTC timestamps
- SQL repository with persistent storage
- Structured validation and error handling
- Token counting accuracy
- Graceful shutdown and resource cleanup

## � Persistent Memory

Conversations are automatically stored in `events.db` SQLite database:
- **Cross-Session Memory**: Conversations persist between restarts
- **Token-Aware Context**: Retrieves recent messages within token limits
- **Usage Tracking**: Monitors LLM costs per conversation
- **Concurrent Safety**: SQLite WAL mode supports multiple connections

## �📡 WebSocket API

**Connect:** `ws://localhost:8000/ws/chat`

**Message Format:**
```json
{
  "action": "chat",
  "request_id": "unique-request-id",
  "payload": {
    "text": "Your message here",
    "conversation_id": "optional-conversation-id"
  }
}
```

**Response Types:**
```json
// Streaming chunk
{"request_id": "...", "status": "chunk", "chunk": {"type": "text", "data": "..."}}

// Completion with usage
{"request_id": "...", "status": "complete", "metadata": {"tokens": 42}}
```

## ⚙️ Configuration

### Required Configuration (`src/config.yaml`)

```yaml
# LLM Provider
llm:
  active: "groq"  # "groq" | "openai" | "openrouter" | "anthropic"
  providers:
    groq:
      model: "llama-3.3-70b-versatile"
      temperature: 0.7
      max_tokens: 4096

# Chat Service
chat:
  service:
    streaming:
      enabled: true
    context:
      max_tokens: 4000      # Context window size
      reserve_tokens: 500   # Reserved for responses
    max_tool_hops: 8        # Tool call limit

# WebSocket Server
chat:
  websocket:
    host: "localhost"
    port: 8000
```

### MCP Servers (`src/servers_config.json`)
```json
{
  "mcpServers": {
    "demo": {
      "enabled": true,
      "command": "uv",
      "args": ["run", "python", "Servers/demo_server.py"]
    }
  }
}
```

## 🛠️ Development Commands

```bash
# Start platform
./run.sh                              # Production start
uv run python src/main.py             # Direct start

# System management
uv sync --upgrade                     # Update dependencies
./run_tests.sh                        # Run tests

# Database inspection
sqlite3 events.db "SELECT conversation_id, COUNT(*) FROM chat_events GROUP BY conversation_id;"
```

## 📁 Architecture

```
src/
├── main.py                  # MCP client and entry point
├── chat_service.py          # Chat orchestration and LLM integration
├── websocket_server.py      # WebSocket communication layer
├── config.py               # Configuration management
├── tool_schema_manager.py   # MCP to OpenAI schema conversion
└── history/
    ├── models.py           # ChatEvent, Usage, ToolCall models
    ├── repositories/       # Storage backends
    │   ├── base.py        # Repository protocol
    │   └── sql_repo.py    # SQLite implementation
    ├── conversation_utils.py # Context management
    └── token_counter.py    # Token counting

events.db                   # SQLite conversation storage
```

---

**Requirements:** Python 3.13+, WebSocket client for API access

**Philosophy:** Persistent memory, clean architecture, production-ready resource management.

