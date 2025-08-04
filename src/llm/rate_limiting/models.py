"""
Rate limiting models and dataclasses for Phase 2 implementation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


class RateLimitType(Enum):
    """Types of rate limits."""
    REQUESTS_PER_MINUTE = "rpm"
    TOKENS_PER_MINUTE = "tpm"
    REQUESTS_PER_DAY = "rpd"
    COST_PER_DAY = "cpd"


class CircuitBreakerStateEnum(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Blocking requests
    HALF_OPEN = "half_open"  # Testing recovery


@dataclass(frozen=True)
class RateLimitConfig:
    """Configuration for rate limiting."""
    requests_per_minute: int = 3500
    tokens_per_minute: int = 200000
    requests_per_day: int = 100000
    cost_per_day: float = 50.0

    # Burst handling
    burst_allowance: float = 1.2  # 20% burst above limit
    burst_duration: float = 10.0  # seconds

    # Queue management
    max_queue_size: int = 100
    queue_timeout: float = 30.0


@dataclass
class RateLimitState:
    """Current rate limiting state."""
    request_timestamps: list[datetime] = field(default_factory=list)
    token_usage: list[tuple[datetime, int]] = field(default_factory=list)
    daily_requests: int = 0
    daily_cost: float = 0.0
    last_reset: datetime = field(default_factory=datetime.now)

    # Queue state
    queued_requests: int = 0
    average_wait_time: float = 0.0


@dataclass(frozen=True)
class CircuitBreakerConfig:
    """Configuration for circuit breaker."""
    failure_threshold: int = 5
    recovery_timeout: float = 60.0
    success_threshold: int = 3  # For half-open -> closed
    timeout_duration: float = 30.0


@dataclass
class CircuitBreakerState:
    """Current circuit breaker state."""
    state: CircuitBreakerStateEnum = CircuitBreakerStateEnum.CLOSED
    failure_count: int = 0
    success_count: int = 0
    last_failure_time: datetime | None = None
    next_attempt_time: datetime | None = None

    # Statistics
    total_requests: int = 0
    total_failures: int = 0
    total_timeouts: int = 0
    state_changes: int = 0


@dataclass(frozen=True)
class RateLimitResult:
    """Result of rate limit check."""
    allowed: bool
    wait_time: float  # seconds to wait
    limit_type: RateLimitType
    current_usage: int | float
    limit_value: int | float
    reset_time: datetime | None = None


@dataclass(frozen=True)
class TokenBucket:
    """Token bucket for rate limiting algorithm."""
    capacity: int
    refill_rate: float  # tokens per second
    current_tokens: float
    last_refill: datetime

    def can_consume(self, tokens: int) -> bool:
        """Check if we can consume tokens."""
        return self.current_tokens >= tokens

    def consume(self, tokens: int) -> bool:
        """Attempt to consume tokens."""
        self.refill()
        return self.current_tokens >= tokens

    def refill(self) -> None:
        """Refill tokens based on time elapsed."""
        # This would need special handling due to frozen dataclass
        pass


@dataclass(frozen=True)
class QueuedRequest:
    """A queued request waiting for rate limit clearance."""
    request_id: str
    estimated_tokens: int
    priority: int = 0
    queued_at: datetime = field(default_factory=datetime.now)
    timeout_at: datetime | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
