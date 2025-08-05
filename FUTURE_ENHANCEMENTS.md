# Future Enhancement Reference - LLM Client

## Cost Tracking Implementation

### Real-time Cost Monitoring
```python
class CostTracker:
    """Real-time cost tracking with 2025 pricing."""
    
    def __init__(self, provider: str, model: str):
        self.provider = provider
        self.model = model
        
        # 2025 pricing
        self.pricing = {
            "openai": {
                "gpt-4o": {"input": 0.005, "output": 0.015},
                "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
                "gpt-4": {"input": 0.03, "output": 0.06},
                "gpt-3.5-turbo": {"input": 0.001, "output": 0.002}
            },
            "openrouter": {
                # OpenRouter provides actual cost in response
                "actual_cost_tracking": True
            }
        }
        
    def estimate_cost(self, input_tokens: int, output_tokens: int = 0) -> float:
        """Estimate cost based on 2025 pricing."""
        if self.provider == "openrouter":
            return 0.0  # Will be updated from response
            
        pricing = self.pricing.get(self.provider, {}).get(self.model)
        if not pricing:
            return 0.0
            
        input_cost = (input_tokens / 1000) * pricing["input"]
        output_cost = (output_tokens / 1000) * pricing["output"]
        
        return input_cost + output_cost
```

## Advanced Configuration Features

### Budget Controls
```yaml
llm:
  providers:
    openai:
      # Cost management
      enable_cost_tracking: true
      cost_warning_threshold: 1.00  # USD per request
      daily_cost_limit: 50.00       # USD per day
      max_cost_per_request: 0.10    # USD
      
      # Performance monitoring
      enable_metrics: true
      log_costs: true
      log_token_usage: true
      
    openrouter:
      # OpenRouter-specific features
      enable_fallbacks: true
      fallback_models:
        - "openai/gpt-3.5-turbo"
        - "anthropic/claude-3-haiku"
      
      # Provider routing preferences
      provider_preferences:
        allow_fallbacks: true
        data_collection: "deny"
      
      # App identification (for better service ranking)
      app_name: "MCP Platform"
      app_url: "https://localhost:8000"
```


### Connection Pool Optimization
```python
# Provider-specific connection limits
limits = httpx.Limits(
    max_connections=config.get("max_connections", 100),
    max_keepalive_connections=config.get("max_keepalive", 20),
    keepalive_expiry=config.get("keepalive_expiry", 5.0)
)

timeout = httpx.Timeout(
    connect=config.get("connect_timeout", 10.0),
    read=config.get("read_timeout", 60.0),
    write=config.get("write_timeout", 10.0),
    pool=config.get("pool_timeout", 10.0)
)
```

## Performance Monitoring

### Metrics Collection
```python
class PerformanceMonitor:
    """Track LLM performance metrics."""
    
    def __init__(self):
        self.request_times = []
        self.token_usage = []
        self.costs = []
        self.errors = []
    
    def record_request(self, duration: float, tokens: int, cost: float):
        """Record request metrics."""
        self.request_times.append(duration)
        self.token_usage.append(tokens)
        self.costs.append(cost)
    
    def get_statistics(self) -> dict:
        """Get performance statistics."""
        return {
            "avg_response_time": sum(self.request_times) / len(self.request_times),
            "total_tokens": sum(self.token_usage),
            "total_cost": sum(self.costs),
            "success_rate": 1 - (len(self.errors) / len(self.request_times))
        }
```

## Advanced Rate Limiting Patterns

### Multi-dimensional Rate Limiting
```python
class AdvancedRateLimiter:
    """Multi-dimensional rate limiting (RPM, TPM, daily limits)."""
    
    async def check_rate_limit(self, estimated_tokens: int) -> RateLimitResult:
        """Check all rate limit dimensions."""
        now = datetime.now()
        
        # Check RPM, TPM, and daily limits simultaneously
        rpm_result = self._check_requests_per_minute(now)
        tpm_result = self._check_tokens_per_minute(now, estimated_tokens)
        daily_result = self._check_daily_limits(now, estimated_tokens)
        
        # Return most restrictive limit
        results = [r for r in [rpm_result, tpm_result, daily_result] if not r.allowed]
        return max(results, key=lambda r: r.wait_time) if results else allow_result
```

## Circuit Breaker Implementation

### Resilience Patterns
```python
class CircuitBreaker:
    """Three-state circuit breaker with exponential backoff."""
    
    async def call(self, func, *args, **kwargs):
        if self.state.state == CircuitBreakerStateEnum.OPEN:
            if self._should_attempt_reset():
                self.state.state = CircuitBreakerStateEnum.HALF_OPEN
            else:
                raise CircuitBreakerError("Circuit breaker is OPEN")
        
        try:
            result = await func(*args, **kwargs)
            self._record_success()
            return result
        except Exception as e:
            self._record_failure()
            raise
```

## Token Management Enhancement

### Advanced Token Estimation
```python
# Leverage existing TokenCounter but add cost awareness
class TokenCostEstimator:
    """Combine token counting with cost estimation."""
    
    def __init__(self, token_counter: TokenCounter, cost_tracker: CostTracker):
        self.token_counter = token_counter
        self.cost_tracker = cost_tracker
    
    def estimate_request_cost(self, messages: list[dict], max_tokens: int) -> dict:
        """Estimate both tokens and cost for a request."""
        input_tokens = self.token_counter.count_messages(messages)
        
        return {
            "input_tokens": input_tokens,
            "estimated_output_tokens": max_tokens,
            "estimated_cost": self.cost_tracker.estimate_cost(input_tokens, max_tokens),
            "within_budget": self.cost_tracker.check_budget(input_tokens + max_tokens)
        }
```

---

**Status**: 
- ✅ **Provider-Specific Headers**: IMPLEMENTED in production
- ✅ **Connection Pool Optimization**: IMPLEMENTED (see PERFORMANCE_GUIDE.md)
- ✅ **Basic Rate Limiting**: IMPLEMENTED in production configuration
- 🔄 **Cost Tracking**: Ready for implementation when needed
- 🔄 **Advanced Performance Monitoring**: Ready for implementation when needed
- 🔄 **Circuit Breaker**: Ready for implementation when needed

**Dependencies**: Current production implementation provides the foundation  
**Integration**: All patterns designed to work with existing `src/llm/` architecture

**Note**: Basic performance optimizations are complete. See `PERFORMANCE_GUIDE.md` for current configuration.

## Implementation Notes

### Current Production Features
- ✅ **Provider-specific headers** automatically applied based on base URL
- ✅ **OpenRouter optimization** for better service ranking and availability
- ✅ **OpenAI API v2** access with latest features
- ✅ **Professional identification** across all providers

### Next Phase Ready
All remaining features are designed to integrate seamlessly with the current architecture and can be implemented independently as needed.
## Provider-Specific Optimizations

### ✅ IMPLEMENTED: OpenRouter Headers
```python
def _get_provider_headers(self, api_key: str, base_url: str) -> dict[str, str]:
    """Get provider-specific headers for optimal API performance."""
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "User-Agent": "MCP-Platform/2025"
    }
    
    if "openrouter.ai" in base_url:
        # OpenRouter-specific headers for better service ranking
        headers.update({
            "HTTP-Referer": "https://localhost:8000",
            "X-Title": "MCP Platform",
        })
    elif "api.openai.com" in base_url:
        # OpenAI-specific optimizations
        headers["OpenAI-Beta"] = "assistants=v2"
        
    return headers
```

**Status**: ✅ **IMPLEMENTED** in `src/main.py` - LLMClient class
**Benefits**: 
- OpenRouter: Better service ranking, model availability, lower latency
- OpenAI: Access to latest Assistants API v2 features
- All providers: Professional identification for better support

### ✅ IMPLEMENTED: Connection Pool Optimization
All connection pool optimizations are implemented and documented in `PERFORMANCE_GUIDE.md`:
- Provider-specific connection limits and timeouts
- SQLite WAL connection pooling (16 readers + 4 writers)
- WebSocket concurrency limits (1000 concurrent connections)
- HTTP client pool management with health monitoring

**Status**: ✅ **IMPLEMENTED** - See `PERFORMANCE_GUIDE.md` for configuration details

### 🔄 READY: Advanced Cost Tracking
```python
def _get_provider_limits(self, config: dict[str, Any]) -> httpx.Limits:
    """Get provider-specific connection limits for optimal performance."""
    if "openrouter.ai" in config["base_url"]:
        # OpenRouter benefits from conservative connection pooling
        return httpx.Limits(
            max_connections=config.get("max_connections", 50),
            max_keepalive_connections=config.get("max_keepalive", 10),
            keepalive_expiry=config.get("keepalive_expiry", 10.0)
        )
    elif "api.openai.com" in config["base_url"]:
        # OpenAI can handle aggressive connection pooling
        return httpx.Limits(
            max_connections=config.get("max_connections", 100),
            max_keepalive_connections=config.get("max_keepalive", 20),
            keepalive_expiry=config.get("keepalive_expiry", 5.0)
        )
    # ... additional providers
```

**Status**: 🔄 **Ready for implementation** - Foundation in place
**Benefits**: Optimized connection pooling per provider characteristics
