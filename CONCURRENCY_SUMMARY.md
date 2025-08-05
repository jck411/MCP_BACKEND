# FastAPI/Starlette + WebSockets Concurrency Improvements

## Summary

Implemented comprehensive concurrency limits to prevent stalling when many SSE or WebSocket clients connect simultaneously. The changes address the key areas you mentioned: **uvicorn --backlog** and **connection-pool limits in httpx**.

## Changes Made

### 1. WebSocket Server Configuration (`src/config.yaml`)

**Added concurrency configuration section:**
```yaml
chat:
  websocket:
    concurrency:
      max_connections: 100              # Prevent resource exhaustion
      connection_queue_size: 500        # Queue size before rejection
      uvicorn:
        backlog: 2048                   # 🔥 KEY: Socket backlog for uvicorn
        workers: 1                      # Single worker for WebSocket state
        access_log: false               # Performance optimization
        keepalive_timeout: 5            # Connection keep-alive settings
        max_keepalive_requests: 1000    # Max requests per connection
        # Additional HTTP limits for resource protection
```

### 2. HTTP Client Connection Pooling (`src/config.yaml`)

**Added httpx connection pool limits for each LLM provider:**
```yaml
llm:
  providers:
    openai:
      http_client:
        max_connections: 50             # 🔥 KEY: Connection pool limit
        max_keepalive: 20              # Keep-alive connections
        keepalive_expiry: 30.0         # Connection expiry time
        connect_timeout: 10.0          # Connection timeouts
        read_timeout: 60.0             
        write_timeout: 10.0            
        pool_timeout: 5.0              # Wait time for pool connection
        concurrent_requests: 25         # Max concurrent requests
        # Rate limiting and retry configuration
```

### 3. Configuration Management (`src/config.py`)

**Added new configuration methods:**
- `get_websocket_concurrency_config()` - WebSocket limits with validation
- `get_http_client_config()` - HTTP client pool configuration
- Comprehensive validation of all concurrency parameters
- Fail-fast approach for missing configuration

### 4. WebSocket Server Implementation (`src/websocket_server.py`)

**Enhanced connection handling:**
- **Connection limiting**: Reject connections when at max capacity
- **Graceful rejection**: Send code 1013 "Try again later"
- **Uvicorn configuration**: Use backlog and performance settings
- **Connection monitoring**: Log active connections vs. limits

### 5. HTTP Client Implementation (`src/main.py`)

**Enhanced LLMClient with connection pooling:**
- **httpx.Limits**: Configure max_connections, max_keepalive, keepalive_expiry
- **httpx.Timeout**: Separate timeouts for connect, read, write, pool
- **Semaphore limiting**: Prevent too many concurrent API requests
- **Per-provider settings**: Different limits for OpenAI, Groq, OpenRouter

## Key Benefits

### 🚀 Performance Improvements

1. **Uvicorn Backlog**: 2048 socket queue prevents connection drops
2. **Connection Pooling**: Reuse HTTP connections instead of creating new ones
3. **Concurrent Limiting**: Prevent overwhelming LLM APIs
4. **Resource Protection**: Graceful degradation under load

### 🛡️ Reliability Improvements

1. **Graceful Connection Rejection**: Clients get clear "try again" signal
2. **Timeout Protection**: Prevent hung connections
3. **Pool Exhaustion Handling**: Queue requests when pool is full
4. **Provider-Specific Limits**: Respect different API rate limits

### 📊 Monitoring & Observability

1. **Connection Metrics**: Log active connections vs. limits
2. **Pool Utilization**: Track HTTP connection pool usage
3. **Performance Logging**: Configuration values logged at startup
4. **Failure Detection**: Clear error messages for limit breaches

## Provider-Specific Configuration

### OpenAI
- Higher connection limits (50 connections)
- Faster timeouts (optimized for fast responses)
- Higher concurrent requests (25)

### Groq
- Lower connection limits (30 connections) 
- Higher read timeout (90s for larger models)
- Conservative concurrent requests (15)

### OpenRouter
- Medium connection limits (40 connections)
- Higher timeouts (variable provider performance)
- Balanced concurrent requests (20)

## Testing

**Syntax validated:**
- ✅ Python syntax check passed
- ✅ YAML configuration validated
- ✅ All imports and dependencies confirmed

**Ready for testing with:**
```bash
# Use production script with concurrency limits
./run_production.sh

# Or standard application
uv run mcp-platform
```

## Files Modified

1. `src/config.yaml` - Added concurrency configuration
2. `src/config.py` - Added configuration methods
3. `src/websocket_server.py` - Enhanced connection handling
4. `src/main.py` - Enhanced HTTP client with pooling
5. `run_production.sh` - Production deployment script (new)
6. `CONCURRENCY_IMPROVEMENTS.md` - Comprehensive documentation (new)

The implementation provides a robust foundation for handling high-concurrency WebSocket and HTTP traffic without stalling.
