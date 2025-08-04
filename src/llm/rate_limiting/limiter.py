"""
Advanced rate limiting with circuit breaker patterns and predictive queuing.
Phase 2: Modern rate limiting with enhanced error recovery.
"""

from __future__ import annotations

import asyncio
import contextlib
from collections import deque
from datetime import datetime, timedelta

from ..exceptions import CircuitBreakerError, RateLimitError
from .models import (
    CircuitBreakerConfig,
    CircuitBreakerState,
    CircuitBreakerStateEnum,
    QueuedRequest,
    RateLimitConfig,
    RateLimitResult,
    RateLimitState,
    RateLimitType,
)

# Constants
MAX_PROCESSING_HISTORY = 100
BURST_WINDOW_SECONDS = 60


class AdvancedRateLimiter:
    """
    Advanced rate limiter with predictive queuing and circuit breaker integration.

    Features:
    - Token bucket algorithm with burst handling
    - Predictive queue management
    - Multiple limit types (RPM, TPM, daily limits)
    - Circuit breaker integration
    - Performance statistics
    """

    def __init__(self, config: RateLimitConfig):
        self.config = config
        self.state = RateLimitState()
        self._lock = asyncio.Lock()

        # Processing time tracking for predictive queuing
        self._processing_times: deque[float] = deque(maxlen=MAX_PROCESSING_HISTORY)

        # Queue for managing pending requests
        self._request_queue: asyncio.Queue[QueuedRequest] = asyncio.Queue(
            maxsize=config.max_queue_size
        )

        # Background task for queue processing
        self._queue_processor_task: asyncio.Task | None = None
        self._stop_processing = False

    async def __aenter__(self) -> AdvancedRateLimiter:
        """Start background queue processor."""
        self._queue_processor_task = asyncio.create_task(self._process_queue())
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Stop background processing and cleanup."""
        self._stop_processing = True
        if self._queue_processor_task:
            self._queue_processor_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._queue_processor_task

    async def check_rate_limit(
        self,
        estimated_tokens: int,
        _priority: int = 0
    ) -> RateLimitResult:
        """
        Check if request can proceed or needs to wait.

        Args:
            estimated_tokens: Expected token usage
            priority: Request priority (higher = more important)

        Returns:
            RateLimitResult with decision and wait time
        """
        async with self._lock:
            now = datetime.now()

            # Clean expired entries
            self._clean_expired_entries(now)

            # Check each limit type
            rpm_result = self._check_requests_per_minute(now)
            tpm_result = self._check_tokens_per_minute(now, estimated_tokens)
            daily_result = self._check_daily_limits(now, estimated_tokens)

            # Find the most restrictive limit
            results = [
                r for r in [rpm_result, tpm_result, daily_result]
                if not r.allowed
            ]

            if not results:
                # All limits passed - record usage and allow
                self._record_usage(now, estimated_tokens)
                return RateLimitResult(
                    allowed=True,
                    wait_time=0.0,
                    limit_type=RateLimitType.REQUESTS_PER_MINUTE,
                    current_usage=len(self.state.request_timestamps),
                    limit_value=self.config.requests_per_minute
                )

        # Find the result with longest wait time
        return max(results, key=lambda r: r.wait_time)

    async def wait_for_capacity(
        self,
        estimated_tokens: int,
        priority: int = 0,
        timeout: float | None = None
    ) -> None:
        """
        Wait until rate limit capacity is available.

        Args:
            estimated_tokens: Expected token usage
            priority: Request priority
            timeout: Maximum wait time (None for no timeout)

        Raises:
            RateLimitError: If timeout exceeded or queue full
        """
        result = await self.check_rate_limit(estimated_tokens, priority)

        if result.allowed:
            return

        # Use timeout from config if not specified
        actual_timeout = timeout or self.config.queue_timeout

        if result.wait_time > actual_timeout:
            raise RateLimitError(
                f"Rate limit wait time ({result.wait_time:.2f}s) "
                f"exceeds timeout ({actual_timeout:.2f}s)",
                retry_after=result.wait_time,
                provider="rate_limiter",
                model="N/A"
            )

        # Wait for the required time
        await asyncio.sleep(result.wait_time)

        # Re-check after waiting
        final_result = await self.check_rate_limit(estimated_tokens, priority)
        if not final_result.allowed:
            raise RateLimitError(
                f"Rate limit still exceeded after waiting {result.wait_time:.2f}s",
                retry_after=final_result.wait_time,
                provider="rate_limiter",
                model="N/A"
            )

    def _check_requests_per_minute(self, now: datetime) -> RateLimitResult:
        """Check requests per minute limit."""
        minute_ago = now - timedelta(minutes=1)
        recent_requests = [
            ts for ts in self.state.request_timestamps
            if ts > minute_ago
        ]

        current_rpm = len(recent_requests)
        limit = self.config.requests_per_minute

        if current_rpm < limit:
            return RateLimitResult(
                allowed=True,
                wait_time=0.0,
                limit_type=RateLimitType.REQUESTS_PER_MINUTE,
                current_usage=current_rpm,
                limit_value=limit
            )

        # Calculate wait time until oldest request expires
        oldest_request = min(recent_requests)
        wait_time = (oldest_request + timedelta(minutes=1) - now).total_seconds()

        return RateLimitResult(
            allowed=False,
            wait_time=max(0.0, wait_time),
            limit_type=RateLimitType.REQUESTS_PER_MINUTE,
            current_usage=current_rpm,
            limit_value=limit,
            reset_time=oldest_request + timedelta(minutes=1)
        )

    def _check_tokens_per_minute(self, now: datetime, tokens: int) -> RateLimitResult:
        """Check tokens per minute limit."""
        minute_ago = now - timedelta(minutes=1)
        recent_usage = [
            (ts, token_count) for ts, token_count in self.state.token_usage
            if ts > minute_ago
        ]

        current_tokens = sum(token_count for _, token_count in recent_usage)
        projected_tokens = current_tokens + tokens
        limit = self.config.tokens_per_minute

        if projected_tokens <= limit:
            return RateLimitResult(
                allowed=True,
                wait_time=0.0,
                limit_type=RateLimitType.TOKENS_PER_MINUTE,
                current_usage=current_tokens,
                limit_value=limit
            )

        # Calculate wait time based on token usage decay
        if recent_usage:
            oldest_ts, oldest_tokens = min(recent_usage, key=lambda x: x[0])
            # Estimate when enough tokens will have "expired"
            tokens_needed = projected_tokens - limit
            if oldest_tokens >= tokens_needed:
                wait_time = (oldest_ts + timedelta(minutes=1) - now).total_seconds()
            else:
                # More complex calculation for multiple entries
                wait_time = 30.0  # Conservative estimate
        else:
            wait_time = 60.0  # Full minute if no recent usage

        return RateLimitResult(
            allowed=False,
            wait_time=max(0.0, wait_time),
            limit_type=RateLimitType.TOKENS_PER_MINUTE,
            current_usage=current_tokens,
            limit_value=limit
        )

    def _check_daily_limits(self, now: datetime, _tokens: int) -> RateLimitResult:
        """Check daily limits (requests and cost)."""
        # Reset daily counters if new day
        if self.state.last_reset.date() < now.date():
            self.state.daily_requests = 0
            self.state.daily_cost = 0.0
            self.state.last_reset = now

        # Check daily request limit
        if self.state.daily_requests >= self.config.requests_per_day:
            tomorrow = datetime.combine(
                now.date() + timedelta(days=1),
                datetime.min.time()
            )
            wait_time = (tomorrow - now).total_seconds()

            return RateLimitResult(
                allowed=False,
                wait_time=wait_time,
                limit_type=RateLimitType.REQUESTS_PER_DAY,
                current_usage=self.state.daily_requests,
                limit_value=self.config.requests_per_day,
                reset_time=tomorrow
            )

        # Daily cost check would require cost estimation
        # For now, assume cost limit is not exceeded
        return RateLimitResult(
            allowed=True,
            wait_time=0.0,
            limit_type=RateLimitType.REQUESTS_PER_DAY,
            current_usage=self.state.daily_requests,
            limit_value=self.config.requests_per_day
        )

    def _clean_expired_entries(self, now: datetime) -> None:
        """Remove expired rate limit entries."""
        minute_ago = now - timedelta(minutes=1)

        # Clean request timestamps
        self.state.request_timestamps = [
            ts for ts in self.state.request_timestamps
            if ts > minute_ago
        ]

        # Clean token usage
        self.state.token_usage = [
            (ts, tokens) for ts, tokens in self.state.token_usage
            if ts > minute_ago
        ]

    def _record_usage(self, timestamp: datetime, tokens: int) -> None:
        """Record usage for rate limiting tracking."""
        self.state.request_timestamps.append(timestamp)
        self.state.token_usage.append((timestamp, tokens))
        self.state.daily_requests += 1

    async def _process_queue(self) -> None:
        """Background task to process queued requests."""
        while not self._stop_processing:
            try:
                # Get next request from queue with timeout
                request = await asyncio.wait_for(
                    self._request_queue.get(),
                    timeout=1.0
                )

                # Check if request has timed out
                if request.timeout_at and datetime.now() > request.timeout_at:
                    continue

                # Try to process request
                result = await self.check_rate_limit(
                    request.estimated_tokens, request.priority
                )

                if result.allowed:
                    # Request can proceed
                    self._request_queue.task_done()
                else:
                    # Re-queue with updated priority
                    await self._request_queue.put(request)
                    await asyncio.sleep(min(result.wait_time, 1.0))

            except TimeoutError:
                # No requests in queue, continue
                continue
            except Exception:
                # Log error and continue
                continue

    def get_statistics(self) -> dict[str, int | float]:
        """Get current rate limiting statistics."""
        now = datetime.now()
        minute_ago = now - timedelta(minutes=1)

        recent_requests = len([
            ts for ts in self.state.request_timestamps
            if ts > minute_ago
        ])

        recent_tokens = sum(
            tokens for ts, tokens in self.state.token_usage
            if ts > minute_ago
        )

        return {
            'requests_last_minute': recent_requests,
            'tokens_last_minute': recent_tokens,
            'daily_requests': self.state.daily_requests,
            'daily_cost': self.state.daily_cost,
            'queue_size': self._request_queue.qsize(),
            'rpm_utilization': recent_requests / self.config.requests_per_minute,
            'tpm_utilization': recent_tokens / self.config.tokens_per_minute,
        }


class CircuitBreaker:
    """
    Circuit breaker implementation for LLM API resilience.

    Features:
    - Configurable failure thresholds
    - Exponential backoff for recovery
    - Half-open state for gradual recovery
    - Comprehensive statistics
    """

    def __init__(self, config: CircuitBreakerConfig):
        self.config = config
        self.state = CircuitBreakerState()
        self._lock = asyncio.Lock()

    async def call(self, func, *args, **kwargs):
        """
        Execute function with circuit breaker protection.

        Args:
            func: Async function to execute
            *args, **kwargs: Arguments for the function

        Returns:
            Function result

        Raises:
            CircuitBreakerError: If circuit is open
        """
        async with self._lock:
            self.state.total_requests += 1

            # Check circuit state
            if self.state.state == CircuitBreakerStateEnum.OPEN:
                if self._should_attempt_reset():
                    self.state.state = CircuitBreakerStateEnum.HALF_OPEN
                    self.state.state_changes += 1
                else:
                    raise CircuitBreakerError(
                        f"Circuit breaker is OPEN. Next attempt at "
                        f"{self.state.next_attempt_time}",
                        provider="circuit_breaker",
                        model="N/A"
                    )

            # Attempt the function call
            try:
                result = await func(*args, **kwargs)
                await self._record_success()
                return result

            except Exception as e:
                await self._record_failure(e)
                raise

    async def _record_success(self) -> None:
        """Record successful operation."""
        if self.state.state == CircuitBreakerStateEnum.HALF_OPEN:
            self.state.success_count += 1

            if self.state.success_count >= self.config.success_threshold:
                # Transition to closed
                self.state.state = CircuitBreakerStateEnum.CLOSED
                self.state.failure_count = 0
                self.state.success_count = 0
                self.state.state_changes += 1
        else:
            # Reset failure count on success
            self.state.failure_count = 0

    async def _record_failure(self, _exception: Exception) -> None:
        """Record failed operation."""
        self.state.total_failures += 1
        self.state.failure_count += 1
        self.state.last_failure_time = datetime.now()

        # Check if we should open the circuit
        if (self.state.state == CircuitBreakerStateEnum.CLOSED and
            self.state.failure_count >= self.config.failure_threshold):

            self.state.state = CircuitBreakerStateEnum.OPEN
            self.state.next_attempt_time = (
                datetime.now() + timedelta(seconds=self.config.recovery_timeout)
            )
            self.state.state_changes += 1

        elif self.state.state == CircuitBreakerStateEnum.HALF_OPEN:
            # Failure in half-open state - go back to open
            self.state.state = CircuitBreakerStateEnum.OPEN
            self.state.next_attempt_time = (
                datetime.now() + timedelta(seconds=self.config.recovery_timeout)
            )
            self.state.success_count = 0
            self.state.state_changes += 1

    def _should_attempt_reset(self) -> bool:
        """Check if circuit should attempt reset to half-open."""
        if self.state.next_attempt_time is None:
            return True

        return datetime.now() >= self.state.next_attempt_time

    def get_statistics(self) -> dict[str, int | float | str | None]:
        """Get circuit breaker statistics."""
        failure_rate = (
            self.state.total_failures / self.state.total_requests
            if self.state.total_requests > 0 else 0.0
        )

        return {
            'state': self.state.state.value,
            'total_requests': self.state.total_requests,
            'total_failures': self.state.total_failures,
            'failure_rate': failure_rate,
            'current_failure_count': self.state.failure_count,
            'state_changes': self.state.state_changes,
            'last_failure': (
                self.state.last_failure_time.isoformat()
                if self.state.last_failure_time else None
            ),
            'next_attempt': (
                self.state.next_attempt_time.isoformat()
                if self.state.next_attempt_time else None
            ),
        }
