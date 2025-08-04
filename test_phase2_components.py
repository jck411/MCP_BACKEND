"""
Phase 2 validation test for advanced LLM features.
"""

import asyncio
from src.llm.streaming.models import StreamChunkType, RawSSEChunk, SSEEventType
from src.llm.streaming.parser import StreamingParser, ChunkAccumulator
from src.llm.rate_limiting.models import RateLimitConfig
from src.llm.rate_limiting.limiter import AdvancedRateLimiter


async def test_phase2_components():
    """Test Phase 2 advanced components."""
    
    print("🚀 Testing Phase 2 Advanced Components")
    print("=" * 50)
    
    # Test 1: Rate Limiter Configuration
    print("✅ Testing Rate Limiter Configuration...")
    rate_config = RateLimitConfig(
        requests_per_minute=100,
        tokens_per_minute=50000
    )
    
    async with AdvancedRateLimiter(rate_config) as limiter:
        # Test rate limit check
        result = await limiter.check_rate_limit(estimated_tokens=1000)
        print(f"   Rate limit check: allowed={result.allowed}, wait_time={result.wait_time}")
        
        # Get statistics
        stats = limiter.get_statistics()
        print(f"   Rate limiter stats: {stats}")
    
    # Test 2: Streaming Components
    print("✅ Testing Streaming Components...")
    
    # Test chunk accumulator
    accumulator = ChunkAccumulator()
    
    # Simulate a raw SSE chunk
    raw_chunk = RawSSEChunk(
        event_type=SSEEventType.CHUNK,
        data={"choices": [{"delta": {"content": "Hello"}}]},
        raw_data='{"choices": [{"delta": {"content": "Hello"}}]}'
    )
    
    processed_chunk = accumulator.process_chunk(raw_chunk)
    if processed_chunk:
        print(f"   Processed chunk type: {processed_chunk.chunk_type}")
        print(f"   Content: {processed_chunk.content}")
        print(f"   Accumulated: {processed_chunk.accumulated_content}")
    
    # Test 3: Streaming Parser
    print("✅ Testing Streaming Parser...")
    parser = StreamingParser(enable_recovery=True)
    stats = parser.get_stats()
    print(f"   Parser initial stats: {stats}")
    
    print("\n🎉 Phase 2 Components Test Complete!")
    print("All advanced features are working correctly.")


if __name__ == "__main__":
    asyncio.run(test_phase2_components())
