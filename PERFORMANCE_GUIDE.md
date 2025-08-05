# MCP Platform Performance Guide

## Overview

This guide covers the comprehensive performance optimizations implemented in the MCP platform, including concurrency limits, database optimizations, and configuration tuning for high-load scenarios.

## Current Performance Features

### ✅ Implemented Optimizations

1. **High-Concurrency WebSocket Support**
   - 1000 concurrent WebSocket connections
   - Optimized connection pooling and queuing
   - Uvicorn backlog configuration for load handling

2. **Database Connection Pooling**
   - SQLite WAL mode with 20-connection pool
   - 16 readers + 4 writers for optimal read/write separation
   - Connection health monitoring and recovery

3. **HTTP Client Optimization**
   - Provider-specific connection pool tuning
   - Rate limiting and concurrent request management
   - Automatic retry with exponential backoff

4. **Memory and Processing Optimizations**
   - LRU caching for JSON parsing and schema conversion
   - Efficient string building for large tool calls
   - Parallel resource loading and tool registration

## Configuration

### WebSocket Server (High Load)
```yaml
chat:
  websocket:
    concurrency:
      max_connections: 1000        # Concurrent WebSocket connections
      connection_queue_size: 1000  # Pending connections queue
      uvicorn:
        backlog: 1200              # Socket backlog for load handling
        workers: 1                 # Single worker for state management
        keepalive_timeout: 120     # Extended keep-alive for efficiency
        max_keepalive_requests: 10000  # High connection reuse
```

### Database Connection Pool
```yaml
chat:
  service:
    repository:
      connection_pool:
        enable_pooling: true       # Enable high-concurrency pooling
        max_connections: 20        # Total pool size
        max_readers: 16            # Optimized for read-heavy workloads
        max_writers: 4             # Sufficient write throughput
        connection_timeout: 30.0   # Connection acquisition timeout
        health_check_interval: 30.0  # Connection health monitoring
```

### HTTP Client Pools (Per Provider)

**OpenAI (High Performance)**:
```yaml
llm:
  providers:
    openai:
      http_client:
        max_connections: 50        # Large pool for high throughput
        max_keepalive: 20          # Connection reuse
        concurrent_requests: 25    # Parallel request limit
        requests_per_minute: 3500  # Rate limiting
```

**Groq (Conservative)**:
```yaml
groq:
  http_client:
    max_connections: 30          # Moderate pool size
    concurrent_requests: 15      # Conservative concurrency
    requests_per_minute: 600     # Groq-specific limits
    read_timeout: 90.0           # Extended timeout for large models
```

**OpenRouter (Balanced)**:
```yaml
openrouter:
  http_client:
    max_connections: 20          # Efficient pool size
    concurrent_requests: 20      # Balanced concurrency
    requests_per_minute: 2000    # Provider limits
```

## Performance Tuning Guidelines

### Deployment Sizing

**Small Deployment (≤50 users)**:
- WebSocket connections: 100-200
- Database pool: 6-10 connections (5 readers, 1-2 writers)
- HTTP connections: 20-30 per provider

**Medium Deployment (50-500 users)**:
- WebSocket connections: 500-1000
- Database pool: 15-20 connections (12-16 readers, 3-4 writers)
- HTTP connections: 30-50 per provider

**Large Deployment (500+ users)**:
- WebSocket connections: 1000+ (current config)
- Database pool: 20+ connections (16+ readers, 4+ writers)
- HTTP connections: 50+ per provider

### Monitoring Key Metrics

1. **WebSocket Performance**
   - Active connections vs. max_connections
   - Connection queue utilization
   - Connection rejection rates

2. **Database Performance**
   - Pool utilization (readers/writers)
   - Connection acquisition wait times
   - Query response times

3. **HTTP Client Performance**
   - Pool utilization per provider
   - Request queue wait times
   - Rate limit hit rates

4. **Memory and CPU**
   - Cache hit rates (JSON/schema caching)
   - String builder efficiency
   - Resource loading times

## Provider-Specific Optimizations

### OpenAI
- **Headers**: OpenAI-Beta for latest API features
- **Connections**: Aggressive pooling (50 connections)
- **Timeouts**: Fast timeouts for quick responses
- **Rate Limits**: High (3500 RPM)

### Groq
- **Connections**: Conservative pooling (30 connections)
- **Timeouts**: Extended read timeout (90s) for large models
- **Rate Limits**: Moderate (600 RPM)
- **Concurrency**: Limited to 15 concurrent requests

### OpenRouter
- **Headers**: HTTP-Referer and X-Title for better service ranking
- **Connections**: Balanced pooling (20 connections)
- **Timeouts**: Variable based on underlying providers
- **Rate Limits**: Generous (2000 RPM)

## Database Architecture

### SQLite WAL Mode Benefits
- **Concurrent Reads**: Multiple readers don't block each other
- **Read/Write Separation**: Dedicated connection pools
- **Performance**: Eliminates "database is locked" errors under load
- **Reliability**: Connection health monitoring and auto-recovery

### Connection Pool Strategy
```
┌─────────────────────────────────────────┐
│              PooledSqlRepo              │
├─────────────────────────────────────────┤
│  Read Pool (16)    │   Write Pool (4)   │
│  ┌─────────────┐   │   ┌─────────────┐  │
│  │ Reader #1   │   │   │ Writer #1   │  │
│  │ Reader #2   │   │   │ Writer #2   │  │
│  │   ...       │   │   │ Writer #3   │  │
│  │ Reader #16  │   │   │ Writer #4   │  │
│  └─────────────┘   │   └─────────────┘  │
└─────────────────────────────────────────┘
                      │
                      ▼
                ┌─────────────┐
                │ SQLite WAL  │
                │  Database   │
                └─────────────┘
```

## Performance Benchmarks

### Expected Improvements vs. Single Connection
- **Concurrent Reads**: 8-16x improvement
- **Read Latency**: 5-10x reduction
- **Write Throughput**: 3-5x improvement
- **Database Lock Errors**: 90%+ reduction
- **WebSocket Throughput**: 10x improvement (100 → 1000 connections)

## Troubleshooting

### High Load Issues
1. **WebSocket Connection Rejections**
   - Check: Active connections vs. max_connections
   - Solution: Increase max_connections or implement load balancing

2. **Database Lock Errors**
   - Check: Connection pool utilization
   - Solution: Increase pool size or optimize queries

3. **HTTP Timeout Errors**
   - Check: Provider-specific rate limits
   - Solution: Adjust concurrent_requests or implement backoff

4. **Memory Issues**
   - Check: Cache sizes and resource loading
   - Solution: Tune cache limits or implement streaming

### Monitoring Commands
```bash
# Check WebSocket server status
curl -s http://localhost:8000/health | jq '.websocket_stats'

# Monitor database pool utilization
curl -s http://localhost:8000/health | jq '.repository_stats'

# Check HTTP client pool status
curl -s http://localhost:8000/health | jq '.llm_client_stats'
```

## Security Considerations

### Connection Limits
- Prevent DoS attacks through connection exhaustion
- Graceful degradation under load
- Clear error messages for client retry logic

### Resource Protection
- Memory usage limits through caching bounds
- CPU protection through request concurrency limits
- Network protection through rate limiting

## Deployment Checklist

### Production Deployment
- [ ] Enable connection pooling (`enable_pooling: true`)
- [ ] Set appropriate connection limits for expected load
- [ ] Configure provider-specific HTTP client settings
- [ ] Enable monitoring and health checks
- [ ] Test failover and recovery procedures

### Performance Validation
- [ ] Load test WebSocket connections up to configured limits
- [ ] Verify database pool utilization under load
- [ ] Test provider rate limit handling
- [ ] Monitor memory usage during peak load
- [ ] Validate connection recovery after failures

## Future Optimizations

### Potential Enhancements
- **Horizontal Scaling**: Load balancer with multiple instances
- **Caching Layer**: Redis for shared state and caching
- **Message Queuing**: Background task processing
- **Advanced Monitoring**: Metrics collection and alerting

### Configuration Evolution
As usage patterns change, monitor and adjust:
- Connection pool sizes based on actual utilization
- Rate limits based on provider feedback
- Timeout values based on response patterns
- Cache sizes based on hit rates

This performance guide reflects the current production configuration and provides guidance for scaling and optimization based on actual usage patterns.
