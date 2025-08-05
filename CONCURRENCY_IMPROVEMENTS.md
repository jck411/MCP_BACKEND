# Concurrency and Performance Improvements

This document outlines the comprehensive concurrency limits and performance improvements implemented to prevent stalling when many SSE or WebSocket clients connect simultaneously.

## Overview

The FastAPI/Starlette + WebSockets setup has been enhanced with proper concurrency controls at multiple levels:

1. **Uvicorn Server Level** - Socket backlog configuration
2. **WebSocket Connection Level** - Maximum concurrent connections
3. **HTTP Client Level** - Connection pool limits for outbound requests
4. **Application Level** - Request concurrency limiting

## Configuration

### WebSocket Server Concurrency

Located in `src/config.yaml` under `chat.websocket.concurrency`:

```yaml
concurrency:
  # Maximum concurrent WebSocket connections
  max_connections: 100
  
  # Connection queue settings
  connection_queue_size: 500  # Pending connections before rejection
  
  # Uvicorn server configuration
  uvicorn:
    backlog: 2048  # Socket backlog for accepting connections
    workers: 1     # Single worker for WebSocket state management
    access_log: false  # Disable access log for performance
    
    # Keep-alive settings for WebSocket connections
    keepalive_timeout: 5   # Seconds to keep idle connections
    max_keepalive_requests: 1000  # Max requests per connection
    
    # Server limits to prevent resource exhaustion
    h11_max_incomplete_event_size: 16384  # Max incomplete HTTP event size
    h11_max_request_line_size: 8192       # Max HTTP request line size
    h11_max_header_size: 32768            # Max HTTP header size
```

### HTTP Client Connection Pooling

Located in `src/config.yaml` under each LLM provider's `http_client` section:

```yaml
http_client:
  # Connection pool limits for outbound HTTP requests
  max_connections: 50      # Total connection pool size
  max_keepalive: 20        # Connections to keep alive in pool
  keepalive_expiry: 30.0   # Seconds before idle connections expire
  
  # Timeout configuration for HTTP requests
  connect_timeout: 10.0    # Connection establishment timeout
  read_timeout: 60.0       # Response read timeout
  write_timeout: 10.0      # Request write timeout
  pool_timeout: 5.0        # Wait time for connection from pool
  
  # Rate limiting to prevent API overwhelming
  requests_per_minute: 3500    # Max requests per minute
  concurrent_requests: 25      # Max concurrent requests
  
  # Retry configuration for failed requests
  max_retries: 3               # Max retry attempts
  backoff_factor: 0.5          # Exponential backoff multiplier
```

## Implementation Details

### 1. Uvicorn Backlog Configuration

The `backlog` parameter controls how many pending connections the socket can queue before rejecting new ones:

- **Problem**: Default backlog is usually 128, causing connection drops under load
- **Solution**: Configured to 2048 for high-concurrency scenarios
- **Location**: `WebSocketServer._create_uvicorn_server()`

### 2. WebSocket Connection Limits

Application-level connection limiting prevents resource exhaustion:

- **Problem**: Unlimited WebSocket connections can exhaust server memory
- **Solution**: Configurable max connections with graceful rejection
- **Location**: `WebSocketServer._connect_websocket()`

```python
# Check connection limits before accepting
if len(self.active_connections) >= self.max_connections:
    await websocket.close(code=1013, reason="Server at maximum capacity")
    return
```

### 3. HTTP Client Connection Pooling

HTTPX client is configured with proper connection pooling:

- **Problem**: Default HTTPX creates new connections for each request
- **Solution**: Connection pool with limits and keep-alive
- **Location**: `LLMClient.__init__()`

```python
limits = httpx.Limits(
    max_connections=http_config["max_connections"],
    max_keepalive_connections=http_config["max_keepalive"],
    keepalive_expiry=http_config["keepalive_expiry"],
)

timeout = httpx.Timeout(
    connect=http_config["connect_timeout"],
    read=http_config["read_timeout"],
    write=http_config["write_timeout"],
    pool=http_config["pool_timeout"],
)
```

### 4. Request Concurrency Limiting

Semaphore-based limiting for outbound HTTP requests:

- **Problem**: Too many concurrent API requests can overwhelm LLM providers
- **Solution**: Semaphore limiting concurrent requests
- **Location**: `LLMClient.get_response_with_tools()`

```python
# Initialize semaphore for concurrent request limiting
self._request_semaphore = asyncio.Semaphore(http_config["concurrent_requests"])

# Use semaphore in request methods
async with self._request_semaphore:
    response = await self.client.post("/chat/completions", json=payload)
```

## Performance Tuning Guidelines

### WebSocket Connections

1. **max_connections**: Set based on expected concurrent users
   - Small deployment: 50-100
   - Medium deployment: 100-500
   - Large deployment: 500-2000

2. **connection_queue_size**: Should be 2-5x max_connections

3. **uvicorn.backlog**: Should be >= connection_queue_size

### HTTP Client Pool

1. **max_connections**: Based on LLM API rate limits
   - OpenAI: 50-100
   - Groq: 20-30 (stricter limits)
   - OpenRouter: 30-40

2. **concurrent_requests**: Typically 25-50% of max_connections

3. **Timeouts**: Adjust based on provider characteristics
   - OpenAI: Lower timeouts (fast responses)
   - OpenRouter: Higher timeouts (variable providers)

## Monitoring and Observability

The implementation includes comprehensive logging:

```
WebSocket server configured with max_connections=100, queue_size=500, uvicorn_backlog=2048
LLM HTTP client configured with max_connections=50, concurrent_requests=25, pool_timeout=5.0s
WebSocket connection established. Active connections: 5/100
```

### Key Metrics to Monitor

1. **Active WebSocket connections** vs max_connections
2. **HTTP connection pool utilization**
3. **Request queue wait times**
4. **Connection rejection rates**

## Failure Modes and Recovery

### Connection Limit Reached

- **Behavior**: New WebSocket connections receive code 1013 (Try again later)
- **Client Action**: Should implement exponential backoff retry
- **Server Action**: Logs warning and continues serving existing connections

### HTTP Pool Exhaustion

- **Behavior**: Requests wait for available connection (up to pool_timeout)
- **Fallback**: Request fails with timeout error after pool_timeout
- **Recovery**: Pool automatically recovers as requests complete

### Provider Rate Limiting

- **Protection**: Semaphore prevents overwhelming provider APIs
- **Behavior**: Requests queue until semaphore slot available
- **Monitoring**: Track semaphore utilization and queue times

## Deployment Considerations

### Production Settings

For production deployments, consider:

1. **Increase limits** based on actual load testing
2. **Monitor resource usage** (memory, CPU, file descriptors)
3. **Set up alerts** for connection limit approaches
4. **Use load balancers** for horizontal scaling

### Development Settings

For development, use conservative limits:

1. **Lower max_connections** (10-20)
2. **Smaller HTTP pools** (5-10 connections)
3. **Enable verbose logging** for debugging

## Testing Concurrency Limits

To test the concurrency improvements:

1. **WebSocket Load Testing**:
   ```bash
   # Use websocket load testing tools
   artillery run websocket-load-test.yml
   ```

2. **HTTP Pool Testing**:
   ```bash
   # Monitor connection pool metrics
   curl -s http://localhost:8000/health
   ```

3. **Monitor Logs**:
   ```bash
   # Watch for connection limit messages
   tail -f application.log | grep "max_connections\|pool_timeout"
   ```

This comprehensive approach ensures the MCP platform can handle high concurrency loads without stalling or resource exhaustion.
