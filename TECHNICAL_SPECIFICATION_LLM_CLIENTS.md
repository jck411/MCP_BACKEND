# Technical Specification: Modern LLM Client Architecture

## Overview

This document provides detailed technical specifications for implementing modern LLM API best practices in the MCP Backend, focusing specifically on OpenAI and OpenRouter integration patterns used in production systems as of 2025.

## Current Architecture Analysis

### File Structure Analysis
```
src/
├── main.py                 # Contains LLMClient class (lines 541-719)
├── config.py              # Configuration management
├── config.yaml            # LLM provider configurations
├── chat_service.py        # Orchestrates LLM interactions
└── websocket_server.py    # WebSocket interface
```

### Current LLMClient Implementation Issues

**Location**: `src/main.py:541-719`

#### Issues Identified:
1. **Not using official OpenAI SDK** despite having `openai>=1.12.0` dependency
2. **Manual SSE parsing** instead of leveraging SDK streaming
3. **No token estimation** despite `tiktoken>=0.7.0` being available
4. **Basic error handling** without provider-specific error types
5. **No connection pooling** optimization
6. **Missing rate limiting** awareness
7. **No cost tracking** or usage optimization

#### Current Code Pattern:
```python
# Line 554-557: Basic HTTP client creation
self.client: httpx.AsyncClient = httpx.AsyncClient(
    base_url=config["base_url"],
    headers={"Authorization": f"Bearer {api_key}"},
    timeout=30.0,
)
```

## Recommended Architecture

### 1. Client Factory Pattern

**Create**: `src/llm_clients/factory.py`

```python
from typing import Any
from .base import BaseLLMClient, LLMClientConfig
from .openai_client import OpenAIClient
from .openrouter_client import OpenRouterClient

def create_llm_client(provider: str, config: dict[str, Any], api_key: str) -> BaseLLMClient:
    """Factory function to create appropriate LLM client."""
    
    # Convert dict config to Pydantic model with validation
    client_config = LLMClientConfig(
        base_url=config["base_url"],
        model=config["model"],
        api_key=api_key,
        **config
    )
    
    if provider == "openai":
        return OpenAIClient(client_config)
    elif provider == "openrouter":
        return OpenRouterClient(client_config)
    else:
        raise ValueError(f"Unsupported provider: {provider}")
```

### 2. Base Client Interface

**Create**: `src/llm_clients/base.py`

#### Key Components:

```python
from pydantic import BaseModel, Field
from typing import Any, AsyncGenerator
from abc import ABC, abstractmethod

class LLMClientConfig(BaseModel):
    """Validated configuration for LLM clients."""
    
    # Core settings
    base_url: str
    model: str
    api_key: str
    
    # Generation parameters with validation
    temperature: float = Field(ge=0.0, le=2.0, default=0.7)
    max_tokens: int = Field(gt=0, default=4096)
    top_p: float = Field(ge=0.0, le=1.0, default=1.0)
    frequency_penalty: float = Field(ge=-2.0, le=2.0, default=0.0)
    presence_penalty: float = Field(ge=-2.0, le=2.0, default=0.0)
    
    # Client behavior
    timeout: float = Field(gt=0, default=60.0)
    max_retries: int = Field(ge=0, default=3)
    
    # Connection pooling
    max_connections: int = Field(gt=0, default=100)
    max_keepalive_connections: int = Field(gt=0, default=20)
    
    # Provider-specific
    extra_headers: dict[str, str] = Field(default_factory=dict)
    extra_params: dict[str, Any] = Field(default_factory=dict)

class Usage(BaseModel):
    """Standardized usage tracking."""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    total_cost: float | None = None

class LLMResponse(BaseModel):
    """Standardized response format."""
    message: dict[str, Any]
    finish_reason: str | None = None
    usage: Usage | None = None
    model: str
    provider: str
    response_time_ms: float | None = None
```

### 3. OpenAI Client Implementation

**Create**: `src/llm_clients/openai_client.py`

#### Key Features:

```python
from openai import AsyncOpenAI, RateLimitError, AuthenticationError
import tiktoken
import time
from typing import AsyncGenerator

class OpenAIClient(BaseLLMClient):
    """Production-ready OpenAI client with 2025 best practices."""
    
    def __init__(self, config: LLMClientConfig):
        super().__init__(config)
        self._client: AsyncOpenAI | None = None
        self._tokenizer: tiktoken.Encoding | None = None
        self._context_length: int = 4096
        
    async def initialize(self) -> None:
        """Initialize with connection pooling and tokenizer."""
        
        # Create HTTP client with optimized settings
        import httpx
        http_client = httpx.AsyncClient(
            limits=httpx.Limits(
                max_connections=self.config.max_connections,
                max_keepalive_connections=self.config.max_keepalive_connections,
                keepalive_expiry=5.0
            ),
            timeout=httpx.Timeout(self.config.timeout)
        )
        
        # Initialize official OpenAI client
        self._client = AsyncOpenAI(
            api_key=self.config.api_key,
            base_url=self.config.base_url,
            timeout=self.config.timeout,
            max_retries=self.config.max_retries,
            http_client=http_client,
            default_headers=self.config.extra_headers
        )
        
        # Initialize tokenizer for accurate counting
        self._tokenizer = tiktoken.get_encoding("cl100k_base")
        self._context_length = self._get_model_context_length()
        
    def _get_model_context_length(self) -> int:
        """Get model-specific context window size."""
        model_contexts = {
            "gpt-4o": 128000,
            "gpt-4o-mini": 128000,
            "gpt-4-turbo": 128000,
            "gpt-4": 8192,
            "gpt-3.5-turbo": 16385,
        }
        
        for model_name, context_size in model_contexts.items():
            if model_name in self.config.model:
                return context_size
        return 4096  # Default fallback
        
    async def estimate_tokens(self, messages: list[dict[str, Any]]) -> int:
        """Accurate token estimation using tiktoken."""
        if not self._tokenizer:
            # Fallback estimation
            return sum(len(str(msg.get("content", ""))) for msg in messages) // 4
            
        tokens = 0
        for message in messages:
            tokens += 4  # Message formatting overhead
            content = message.get("content", "")
            tokens += len(self._tokenizer.encode(str(content)))
            tokens += len(self._tokenizer.encode(message.get("role", "")))
        tokens += 2  # Conversation overhead
        return tokens
        
    async def get_response(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        **kwargs: Any
    ) -> LLMResponse:
        """Get response with comprehensive error handling."""
        
        start_time = time.time()
        
        # Pre-flight token validation
        estimated_tokens = await self.estimate_tokens(messages)
        if estimated_tokens > self._context_length:
            raise TokenLimitError(
                f"Messages ({estimated_tokens} tokens) exceed context limit ({self._context_length})"
            )
            
        try:
            params = {
                "model": self.config.model,
                "messages": messages,
                "temperature": self.config.temperature,
                "max_tokens": self.config.max_tokens,
                "top_p": self.config.top_p,
                "frequency_penalty": self.config.frequency_penalty,
                "presence_penalty": self.config.presence_penalty,
                **kwargs
            }
            
            if tools:
                params["tools"] = tools
            params.update(self.config.extra_params)
            
            response = await self._client.chat.completions.create(**params)
            
            return LLMResponse(
                message=response.choices[0].message.model_dump(),
                finish_reason=response.choices[0].finish_reason,
                usage=self._extract_usage(response.usage),
                model=response.model,
                provider="openai",
                response_time_ms=(time.time() - start_time) * 1000
            )
            
        except RateLimitError as e:
            raise LLMRateLimitError(f"OpenAI rate limit: {e}") from e
        except AuthenticationError as e:
            raise LLMAuthError(f"OpenAI auth error: {e}") from e
        except Exception as e:
            raise LLMClientError(f"OpenAI error: {e}") from e
```

### 4. OpenRouter Client Implementation

**Create**: `src/llm_clients/openrouter_client.py`

#### OpenRouter-Specific Optimizations:

```python
from openai import AsyncOpenAI  # OpenRouter is OpenAI-compatible
import httpx

class OpenRouterClient(BaseLLMClient):
    """OpenRouter client with cost tracking and model fallbacks."""
    
    async def initialize(self) -> None:
        """Initialize with OpenRouter-specific headers."""
        
        # OpenRouter-specific headers for better service
        headers = {
            "HTTP-Referer": "https://localhost:8000",  # From config
            "X-Title": "MCP Platform",  # From config
            **self.config.extra_headers
        }
        
        http_client = httpx.AsyncClient(
            limits=httpx.Limits(
                max_connections=self.config.max_connections,
                max_keepalive_connections=self.config.max_keepalive_connections
            ),
            headers=headers
        )
        
        self._client = AsyncOpenAI(
            api_key=self.config.api_key,
            base_url="https://openrouter.ai/api/v1",
            http_client=http_client
        )
        
    async def get_response_with_cost_tracking(self, **kwargs) -> dict[str, Any]:
        """OpenRouter provides cost information in responses."""
        response = await self.get_response(**kwargs)
        
        # Extract OpenRouter cost information
        cost_info = self._extract_cost_info(response.usage)
        
        return {
            "response": response,
            "cost_info": cost_info
        }
        
    def _extract_cost_info(self, usage: Any) -> dict[str, Any]:
        """Extract cost information from OpenRouter usage data."""
        if not usage:
            return {}
            
        return {
            "total_cost": getattr(usage, 'total_cost', None),
            "prompt_cost": getattr(usage, 'prompt_cost', None),
            "completion_cost": getattr(usage, 'completion_cost', None),
        }
```

### 5. Integration with Existing Chat Service

**Modify**: `src/chat_service.py`

#### Integration Points:

```python
# Line 18: Add import
from src.llm_clients.factory import create_llm_client

# In ChatService.__init__ method:
class ChatService:
    def __init__(self, service_config: ChatServiceConfig):
        # ... existing code ...
        
        # Replace direct LLMClient usage with factory
        provider = self.configuration.get_llm_config()["active"]
        llm_config = self.configuration.get_llm_config()["providers"][provider]
        api_key = self.configuration.llm_api_key
        
        self.llm_client = create_llm_client(provider, llm_config, api_key)
        
    async def initialize(self) -> None:
        """Initialize chat service with modern LLM client."""
        # ... existing MCP client initialization ...
        
        # Initialize modern LLM client
        await self.llm_client.initialize()
        
        # ... rest of existing code ...
```

### 6. Enhanced Configuration Schema

**Modify**: `src/config.yaml`

#### Enhanced Provider Configurations:

```yaml
llm:
  active: "openrouter"  # or "openai"
  
  providers:
    openai:
      base_url: "https://api.openai.com/v1"
      model: "gpt-4o-mini"
      
      # Generation parameters
      temperature: 0.7
      max_tokens: 4096
      top_p: 1.0
      frequency_penalty: 0.0
      presence_penalty: 0.0
      
      # Client optimization
      timeout: 60.0
      max_retries: 3
      max_connections: 100
      max_keepalive_connections: 20
      
      # Advanced features
      enable_logprobs: false
      enable_json_mode: false
      
      # Rate limiting awareness (based on your OpenAI tier)
      requests_per_minute: 3500  # Tier 2 default
      tokens_per_minute: 200000  # Tier 2 default
      
      # Cost monitoring
      enable_cost_tracking: true
      cost_warning_threshold: 1.00  # USD per request
      
      extra_headers: {}
      extra_params: {}
      
    openrouter:
      base_url: "https://openrouter.ai/api/v1"
      model: "openai/gpt-4o-mini"
      
      # Same generation parameters as OpenAI
      temperature: 0.7
      max_tokens: 4096
      top_p: 1.0
      
      # OpenRouter-specific settings
      enable_cost_tracking: true
      enable_fallback_models: true
      fallback_models:
        - "openai/gpt-3.5-turbo"
        - "anthropic/claude-3-haiku"
        
      # OpenRouter headers for better service
      extra_headers:
        HTTP-Referer: "https://localhost:8000"
        X-Title: "MCP Platform"
        
      # Cost limits
      max_cost_per_request: 0.10  # USD
      daily_cost_limit: 10.00     # USD
```

### 7. Error Handling Enhancement

**Create**: `src/llm_clients/exceptions.py`

```python
"""Modern LLM client exceptions with provider-specific handling."""

class LLMClientError(Exception):
    """Base exception for LLM client errors."""
    pass

class LLMRateLimitError(LLMClientError):
    """Rate limit exceeded."""
    def __init__(self, message: str, retry_after: float | None = None):
        super().__init__(message)
        self.retry_after = retry_after

class LLMAuthError(LLMClientError):
    """Authentication failed."""
    pass

class LLMTokenLimitError(LLMClientError):
    """Token limit exceeded."""
    pass

class LLMModelNotFoundError(LLMClientError):
    """Model not available."""
    pass

class LLMStreamingError(LLMClientError):
    """Streaming-specific errors."""
    pass

class LLMCostLimitError(LLMClientError):
    """Cost limit exceeded."""
    pass
```

### 8. Monitoring and Observability

**Create**: `src/llm_clients/monitoring.py`

```python
"""Monitoring and metrics collection for LLM clients."""

import logging
import time
from typing import Any
from collections import defaultdict, deque
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class LLMMetrics:
    """Collect and track LLM usage metrics."""
    
    def __init__(self):
        self.request_count = defaultdict(int)
        self.token_usage = defaultdict(int)
        self.cost_tracking = defaultdict(float)
        self.response_times = defaultdict(list)
        self.error_count = defaultdict(int)
        
    def record_request(
        self,
        provider: str,
        model: str,
        tokens: int,
        cost: float | None,
        response_time_ms: float,
        error: str | None = None
    ):
        """Record metrics for a single request."""
        
        key = f"{provider}:{model}"
        
        self.request_count[key] += 1
        self.token_usage[key] += tokens
        
        if cost:
            self.cost_tracking[key] += cost
            
        self.response_times[key].append(response_time_ms)
        
        # Keep only recent response times (last 100)
        if len(self.response_times[key]) > 100:
            self.response_times[key] = self.response_times[key][-100:]
            
        if error:
            self.error_count[f"{key}:{error}"] += 1
            
        # Log high-level metrics
        logger.info(
            f"LLM Request - {provider}:{model} | "
            f"Tokens: {tokens} | "
            f"Cost: ${cost:.4f}" if cost else "Cost: Unknown" f" | "
            f"Time: {response_time_ms:.0f}ms"
        )
        
    def get_summary(self) -> dict[str, Any]:
        """Get usage summary for monitoring."""
        return {
            "total_requests": sum(self.request_count.values()),
            "total_tokens": sum(self.token_usage.values()),
            "total_cost": sum(self.cost_tracking.values()),
            "avg_response_time": self._calculate_avg_response_time(),
            "error_rate": self._calculate_error_rate(),
            "by_model": dict(self.request_count),
        }
        
    def _calculate_avg_response_time(self) -> float:
        """Calculate average response time across all models."""
        all_times = []
        for times in self.response_times.values():
            all_times.extend(times)
        return sum(all_times) / len(all_times) if all_times else 0.0
        
    def _calculate_error_rate(self) -> float:
        """Calculate error rate across all requests."""
        total_requests = sum(self.request_count.values())
        total_errors = sum(self.error_count.values())
        return total_errors / total_requests if total_requests > 0 else 0.0
```

## Migration Strategy

### Phase 1: Create New Client Structure
1. Create `src/llm_clients/` directory
2. Implement base classes and interfaces
3. Create OpenAI and OpenRouter clients
4. Add comprehensive testing

### Phase 2: Update Configuration
1. Enhance `config.yaml` with new parameters
2. Update `config.py` to handle new settings
3. Add validation for new configuration options

### Phase 3: Integrate with Chat Service
1. Update `chat_service.py` to use factory pattern
2. Replace direct LLMClient usage
3. Add monitoring and metrics collection
4. Implement gradual rollout with feature flags

### Phase 4: Deprecate Old Implementation
1. Mark old LLMClient as deprecated
2. Add migration warnings
3. Remove old implementation after validation
4. Update documentation

## Testing Strategy

### Unit Tests
- Test token estimation accuracy
- Test error handling scenarios
- Test configuration validation
- Test rate limiting logic

### Integration Tests
- Test full API workflows
- Test streaming functionality
- Test cost tracking accuracy
- Test failover scenarios

### Performance Tests
- Benchmark connection pooling benefits
- Test rate limiting effectiveness
- Measure response time improvements
- Validate memory usage optimization

## Monitoring and Rollback

### Key Metrics to Monitor
- Response time percentiles (p50, p95, p99)
- Error rates by provider and model
- Token usage and cost trends
- Rate limit hit frequency
- Connection pool utilization

### Rollback Triggers
- Error rate increase > 5%
- Response time increase > 20%
- Cost increase > 15%
- Any authentication failures

### Rollback Procedure
1. Disable new client via feature flag
2. Revert to old LLMClient implementation
3. Monitor metrics for stabilization
4. Investigate issues before re-enabling

This technical specification provides the detailed implementation guidance needed to modernize the MCP Backend's LLM integration with production-ready best practices for 2025.
