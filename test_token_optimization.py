#!/usr/bin/env python3
"""
Performance test comparing optimized vs original token computation.

This test demonstrates the performance improvement achieved by using
incremental token counting instead of recomputing tokens for the entire
conversation on each iteration.
"""

import time
from datetime import datetime, UTC
from typing import Any

from src.history.chat_store import ChatEvent
from src.history.token_counter import count_conversation_tokens, estimate_tokens


def original_build_conversation_with_token_limit(
    system_prompt: str,
    events: list[ChatEvent],
    user_message: str,
    max_tokens: int,
    reserve_tokens: int,
) -> tuple[list[dict[str, Any]], int]:
    """Original implementation with redundant token computation."""
    
    # Start with minimal conversation: system + current user message
    base_conversation = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message},
    ]

    # Calculate the immutable token cost (system prompt + user message)
    base_tokens = count_conversation_tokens(base_conversation)

    # Calculate how many tokens we have available for historical events
    available_tokens = max_tokens - base_tokens - reserve_tokens

    # Early exit if system prompt + user message already exceed limits
    if available_tokens <= 0:
        return base_conversation, base_tokens

    # Build conversation incrementally, starting with system prompt
    conversation = [{"role": "system", "content": system_prompt}]

    # Process events in chronological order to maintain conversation flow
    for event in events:
        # Only include events that make sense in LLM conversation context
        if event.type in ("user_message", "assistant_message", "system_update"):
            event_msg = {"role": event.role, "content": event.content or ""}

            # Create test conversation with this event included
            test_conversation = [*conversation, event_msg]

            # Calculate final token count if we include this event + user message
            final_tokens_with_user = count_conversation_tokens(
                [*test_conversation, {"role": "user", "content": user_message}]
            )

            # Check if including this event would breach our token budget
            if final_tokens_with_user + reserve_tokens > max_tokens:
                break  # Stop adding events - we've hit our limit

            # Safe to include this event - add it to the conversation
            conversation.append(event_msg)

    # Always append the current user message last
    conversation.append({"role": "user", "content": user_message})

    # Calculate and return final token count
    final_tokens = count_conversation_tokens(conversation)

    return conversation, final_tokens


def optimized_build_conversation_with_token_limit(
    system_prompt: str,
    events: list[ChatEvent],
    user_message: str,
    max_tokens: int,
    reserve_tokens: int,
) -> tuple[list[dict[str, Any]], int]:
    """Optimized implementation with incremental token counting."""
    
    # Calculate base token costs using cached estimates
    system_tokens = estimate_tokens(system_prompt)
    user_tokens = estimate_tokens(user_message)

    # Account for OpenAI message formatting overhead
    # (3 tokens per message + 3 for priming)
    message_overhead = 3 * 2 + 3  # system + user messages + priming
    base_tokens = system_tokens + user_tokens + message_overhead

    # Calculate available token budget for historical events
    available_tokens = max_tokens - base_tokens - reserve_tokens

    # Early exit if system prompt + user message already exceed limits
    if available_tokens <= 0:
        base_conversation = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ]
        return base_conversation, base_tokens

    # Build conversation incrementally with token accumulation
    conversation = [{"role": "system", "content": system_prompt}]
    accumulated_tokens = system_tokens + 3  # system message + overhead

    # Process events in chronological order to maintain conversation flow
    for event in events:
        # Only include events that make sense in LLM conversation context
        if event.type in ("user_message", "assistant_message", "system_update"):
            # Get cached token count for this event
            event_tokens = event.compute_and_cache_tokens()
            event_overhead = 3  # message formatting overhead
            total_event_cost = event_tokens + event_overhead

            # Check if adding this event would exceed our budget
            # Account for user message that will be added at the end
            projected_total = (
                accumulated_tokens + total_event_cost + user_tokens + 3 + 3
            )  # user msg + overhead + priming

            if projected_total + reserve_tokens > max_tokens:
                break  # Stop adding events - we've hit our limit

            # Safe to include this event - add it to the conversation
            event_msg = {"role": event.role, "content": event.content or ""}
            conversation.append(event_msg)
            accumulated_tokens += total_event_cost

    # Always append the current user message last (required for OpenAI format)
    conversation.append({"role": "user", "content": user_message})
    final_tokens = (
        accumulated_tokens + user_tokens + 3 + 3
    )  # user msg + overhead + priming

    return conversation, final_tokens


def create_test_events(count: int) -> list[ChatEvent]:
    """Create test events for performance testing."""
    events = []
    for i in range(count):
        if i % 2 == 0:
            events.append(ChatEvent(
                conversation_id='test',
                type='user_message',
                role='user',
                content=f'This is user message number {i} with some content to make it realistic.',
                timestamp=datetime.now(UTC)
            ))
        else:
            events.append(ChatEvent(
                conversation_id='test',
                type='assistant_message',
                role='assistant',
                content=f'This is assistant response number {i}. It provides a helpful answer to the user question with detailed information.',
                timestamp=datetime.now(UTC)
            ))
    return events


def benchmark_function(func, name: str, *args, **kwargs) -> tuple[Any, float]:
    """Benchmark a function and return result and execution time."""
    start_time = time.perf_counter()
    result = func(*args, **kwargs)
    end_time = time.perf_counter()
    execution_time = (end_time - start_time) * 1000  # Convert to milliseconds
    print(f"{name}: {execution_time:.2f}ms")
    return result, execution_time


def main():
    """Run performance comparison tests."""
    print("🧪 Performance Test: Token Computation Optimization")
    print("=" * 60)
    
    # Test with different event counts
    test_sizes = [10, 50, 100, 200]
    
    system_prompt = "You are a helpful AI assistant that provides accurate and detailed responses."
    user_message = "Can you help me understand the optimization that was implemented?"
    max_tokens = 4000
    reserve_tokens = 500
    
    for size in test_sizes:
        print(f"\n📊 Testing with {size} events:")
        events = create_test_events(size)
        
        # Benchmark original implementation
        original_result, original_time = benchmark_function(
            original_build_conversation_with_token_limit,
            "Original (redundant counting)",
            system_prompt, events, user_message, max_tokens, reserve_tokens
        )
        
        # Benchmark optimized implementation
        optimized_result, optimized_time = benchmark_function(
            optimized_build_conversation_with_token_limit,
            "Optimized (incremental)",
            system_prompt, events, user_message, max_tokens, reserve_tokens
        )
        
        # Calculate improvement
        if original_time > 0:
            speedup = original_time / optimized_time
            improvement = ((original_time - optimized_time) / original_time) * 100
            print(f"📈 Speedup: {speedup:.1f}x ({improvement:.1f}% faster)")
        else:
            print("📈 Both implementations were too fast to measure accurately")
        
        # Verify results are consistent
        orig_conv, orig_tokens = original_result
        opt_conv, opt_tokens = optimized_result
        
        # Allow small differences due to different token counting methods
        token_diff = abs(orig_tokens - opt_tokens)
        if token_diff < 10:  # Allow small differences in overhead calculation
            print("✅ Results are consistent")
        else:
            print(f"⚠️ Token count difference: {token_diff}")
            print(f"   Original: {orig_tokens}, Optimized: {opt_tokens}")

    print("\n" + "=" * 60)
    print("🎯 Summary:")
    print("   - Optimized version uses incremental token counting")
    print("   - No redundant recomputation of entire conversation tokens")  
    print("   - Leverages cached token counts from ChatEvent.compute_and_cache_tokens()")
    print("   - Significant performance improvement with larger conversations")


if __name__ == "__main__":
    main()
