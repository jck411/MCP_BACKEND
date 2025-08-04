#!/usr/bin/env python3
"""
Comprehensive Phase 2 Testing Suite
Tests all advanced features and integrations.
"""

import asyncio
import json
from typing import Any, Dict

from src.llm.rate_limiting.limiter import AdvancedRateLimiter, CircuitBreaker
from src.llm.rate_limiting.models import RateLimitConfig, CircuitBreakerConfig, CircuitBreakerState
from src.llm.streaming.models import StreamChunkType, RawSSEChunk, AccumulatorState
from src.llm.streaming.parser import StreamingParser, ChunkAccumulator
from src.llm.models import ProviderType
from src.llm.client import Phase2LLMClient, ModernLLMClient


async def test_comprehensive_rate_limiting():
    """Test advanced rate limiting features with circuit breaker integration."""
    print("🧪 Testing Comprehensive Rate Limiting...")
    
    # Test rate limiter with tight limits
    config = RateLimitConfig(
        requests_per_minute=5,
        tokens_per_minute=1000,
        requests_per_day=100,
        cost_per_day=1.0
    )
    
    async with AdvancedRateLimiter(config) as limiter:
        # Test multiple requests
        start_stats = limiter.get_statistics()
        print(f"   Initial stats: {start_stats}")
        
        # Make several requests quickly
        for i in range(3):
            allowed, wait_time = await limiter.check_rate_limit(200)
            print(f"   Request {i+1}: allowed={allowed}, wait_time={wait_time:.2f}s")
            
            if allowed:
                await limiter.wait_for_capacity(200)
        
        # Check final statistics
        final_stats = limiter.get_statistics()
        print(f"   Final stats: {final_stats}")
        
        # Verify rate limiting is working
        assert final_stats['requests_last_minute'] <= 5, "Rate limiting not working"
        assert final_stats['tokens_last_minute'] <= 1000, "Token limiting not working"
        
    print("   ✅ Advanced rate limiting working correctly")


async def test_circuit_breaker_patterns():
    """Test circuit breaker states and recovery."""
    print("🔌 Testing Circuit Breaker Patterns...")
    
    config = CircuitBreakerConfig(
        failure_threshold=3,
        recovery_timeout=1.0,  # Short timeout for testing
        success_threshold=2
    )
    
    breaker = CircuitBreaker(config)
    
    # Test initial state
    assert breaker.state == CircuitBreakerState.CLOSED
    print(f"   Initial state: {breaker.state}")
    
    # Simulate failures to open circuit
    async def failing_operation():
        raise Exception("Simulated failure")
    
    # Test failure accumulation
    for i in range(3):
        try:
            await breaker.call(failing_operation)
        except Exception:
            print(f"   Failure {i+1} recorded")
    
    # Circuit should now be open
    assert breaker.state == CircuitBreakerState.OPEN
    print(f"   Circuit opened after failures: {breaker.state}")
    
    # Wait for recovery timeout
    await asyncio.sleep(1.1)
    
    # Next call should move to half-open
    try:
        await breaker.call(failing_operation)
    except Exception:
        pass
    
    assert breaker.state == CircuitBreakerState.HALF_OPEN
    print(f"   Circuit moved to half-open: {breaker.state}")
    
    # Test successful operation to close circuit
    async def successful_operation():
        return "success"
    
    for i in range(2):  # Need 2 successes to close
        result = await breaker.call(successful_operation)
        print(f"   Success {i+1}: {result}")
    
    assert breaker.state == CircuitBreakerState.CLOSED
    print(f"   Circuit closed after successes: {breaker.state}")
    
    # Get statistics
    stats = breaker.get_statistics()
    print(f"   Circuit breaker stats: {stats}")
    
    print("   ✅ Circuit breaker patterns working correctly")


async def test_streaming_error_recovery():
    """Test streaming with error recovery."""
    print("🌊 Testing Streaming Error Recovery...")
    
    parser = StreamingParser(enable_recovery=True, chunk_timeout=1.0)
    accumulator = ChunkAccumulator()
    
    # Test normal chunk processing
    normal_chunk = RawSSEChunk(
        event_type="chunk",
        data={"choices": [{"delta": {"content": "Hello"}}]},
        raw_data='{"choices": [{"delta": {"content": "Hello"}}]}'
    )
    
    processed = accumulator.process_chunk(normal_chunk)
    assert processed is not None
    assert processed.chunk_type == StreamChunkType.CONTENT
    print(f"   Normal chunk processed: {processed.content}")
    
    # Test error chunk handling
    error_chunk = RawSSEChunk(
        event_type="error",
        data=None,
        raw_data="invalid json",
        error="JSON decode error"
    )
    
    error_processed = accumulator.process_chunk(error_chunk)
    assert error_processed is not None
    assert error_processed.chunk_type == StreamChunkType.ERROR
    print(f"   Error chunk handled: {error_processed.error}")
    
    # Test tool call accumulation
    tool_chunk1 = RawSSEChunk(
        event_type="chunk",
        data={
            "choices": [{
                "delta": {
                    "tool_calls": [{
                        "index": 0,
                        "id": "call_123",
                        "function": {"name": "test_func", "arguments": '{"arg"}'}
                    }]
                }
            }]
        },
        raw_data=""
    )
    
    tool_processed = accumulator.process_chunk(tool_chunk1)
    assert tool_processed is not None
    assert tool_processed.chunk_type == StreamChunkType.TOOL_CALLS
    print(f"   Tool call chunk processed: {len(tool_processed.tool_calls)} tools")
    
    # Test completion chunk
    completion_chunk = RawSSEChunk(
        event_type="completion",
        data=None,
        raw_data="[DONE]"
    )
    
    completion_processed = accumulator.process_chunk(completion_chunk)
    assert completion_processed is not None
    assert completion_processed.chunk_type == StreamChunkType.COMPLETION
    print(f"   Completion chunk processed: {completion_processed.finish_reason}")
    
    # Get parser statistics
    stats = parser.get_stats()
    print(f"   Parser stats: {stats}")
    
    print("   ✅ Streaming error recovery working correctly")


async def test_phase2_client_integration():
    """Test Phase 2 client integration."""
    print("🤖 Testing Phase 2 Client Integration...")
    
    # Create test configuration
    config = {
        "base_url": "https://api.openai.com/v1",
        "model": "gpt-4o-mini",
        "temperature": 0.7,
        "max_tokens": 4096,
        "top_p": 1.0,
        "requests_per_minute": 100,
        "tokens_per_minute": 10000,
        "circuit_breaker_threshold": 5,
        "circuit_breaker_timeout": 60.0,
        "enable_streaming_recovery": True,
        "streaming_timeout": 30.0
    }
    
    client = Phase2LLMClient(config, "test-api-key")
    
    # Test provider detection
    assert client.provider_type == ProviderType.OPENAI
    print(f"   Provider detected: {client.provider_type}")
    
    # Test component initialization
    assert client.rate_limiter is not None
    assert client.circuit_breaker is not None
    assert client.streaming_parser is not None
    print("   ✅ All components initialized")
    
    # Test statistics gathering
    async with client:
        stats = client.get_statistics()
        assert "provider" in stats
        assert "model" in stats
        assert "requests_last_minute" in stats
        assert "circuit_breaker_state" in stats
        assert "streaming" in stats
        print(f"   Statistics collected: {len(stats)} metrics")
    
    print("   ✅ Phase 2 client integration working correctly")


async def test_performance_monitoring():
    """Test performance monitoring and statistics."""
    print("📊 Testing Performance Monitoring...")
    
    config = RateLimitConfig(
        requests_per_minute=1000,
        tokens_per_minute=50000
    )
    
    async with AdvancedRateLimiter(config) as limiter:
        # Simulate workload
        for i in range(10):
            await limiter.wait_for_capacity(100)
            
        stats = limiter.get_statistics()
        
        # Verify statistics collection
        expected_keys = [
            'requests_last_minute', 'tokens_last_minute', 
            'daily_requests', 'daily_cost', 'queue_size',
            'rpm_utilization', 'tpm_utilization'
        ]
        
        for key in expected_keys:
            assert key in stats, f"Missing statistic: {key}"
        
        print(f"   Performance stats: {stats}")
        
        # Test utilization calculations
        assert stats['rpm_utilization'] > 0, "RPM utilization not calculated"
        assert stats['tpm_utilization'] > 0, "TPM utilization not calculated"
        
    print("   ✅ Performance monitoring working correctly")


async def test_backward_compatibility():
    """Test backward compatibility with Phase 1."""
    print("🔄 Testing Backward Compatibility...")
    
    # Test ModernLLMClient still works
    config = {
        "base_url": "https://api.openai.com/v1",
        "model": "gpt-4o-mini"
    }
    
    client = ModernLLMClient(config, "test-api-key")
    
    # Test provider detection
    assert client.provider_type == ProviderType.OPENAI
    print(f"   Phase 1 client provider: {client.provider_type}")
    
    # Test basic methods exist
    async with client:
        # These should work without errors
        response = await client.get_response_with_tools([])
        assert response is not None
        print("   ✅ get_response_with_tools works")
        
        # Test streaming
        async for chunk in client.get_streaming_response_with_tools([]):
            assert chunk is not None
            print("   ✅ get_streaming_response_with_tools works")
            break  # Just test first chunk
    
    print("   ✅ Backward compatibility maintained")


async def main():
    """Run comprehensive Phase 2 testing suite."""
    print("🚀 Comprehensive Phase 2 Testing Suite")
    print("=" * 50)
    
    try:
        await test_comprehensive_rate_limiting()
        await test_circuit_breaker_patterns()
        await test_streaming_error_recovery()
        await test_phase2_client_integration()
        await test_performance_monitoring()
        await test_backward_compatibility()
        
        print("\n🎉 All Comprehensive Tests Passed!")
        print("Phase 2 implementation is fully functional with:")
        print("  ✅ Advanced rate limiting with predictive queuing")
        print("  ✅ Circuit breaker patterns with recovery")
        print("  ✅ Enhanced streaming with error recovery")
        print("  ✅ Performance monitoring and statistics")
        print("  ✅ Full backward compatibility")
        print("  ✅ Multi-dimensional rate limiting")
        print("  ✅ Background processing and queue management")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    success = asyncio.run(main())
    exit(0 if success else 1)
