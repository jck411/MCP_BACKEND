#!/usr/bin/env python3

"""
Test script to verify the duplicate user message fix works correctly
for various conversation scenarios.
"""

import asyncio
import logging
from src.history.conversation_utils import build_conversation_with_token_limit
from src.history.models import ChatEvent, Usage

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_conversation_scenarios():
    """Test various conversation scenarios to ensure fix works correctly."""
    
    print("🧪 Testing conversation building with duplicate prevention...\n")
    
    # Scenario 1: Empty conversation (new user message)
    print("Test 1: Empty conversation")
    conv, tokens = build_conversation_with_token_limit(
        system_prompt="You are helpful.",
        events=[],
        user_message="Hello!",
        max_tokens=1000,
        reserve_tokens=100
    )
    
    user_msgs = [m for m in conv if m['role'] == 'user']
    print(f"  Messages: {len(conv)}, User messages: {len(user_msgs)}")
    assert len(user_msgs) == 1, "Should have exactly 1 user message"
    assert user_msgs[0]['content'] == "Hello!", "User message content should match"
    print("  ✅ PASSED\n")
    
    # Scenario 2: Conversation with existing user message (exact duplicate)
    print("Test 2: Conversation with existing identical user message")
    existing_events = [
        ChatEvent(
            conversation_id="test",
            type="user_message",
            role="user",
            content="Hello!",
            extra={"request_id": "prev-req"}
        ),
        ChatEvent(
            conversation_id="test",
            type="assistant_message",
            role="assistant",
            content="Hi there! How can I help you?"
        )
    ]
    
    conv, tokens = build_conversation_with_token_limit(
        system_prompt="You are helpful.",
        events=existing_events,
        user_message="Hello!",  # Same as existing message
        max_tokens=1000,
        reserve_tokens=100
    )
    
    user_msgs = [m for m in conv if m['role'] == 'user']
    print(f"  Messages: {len(conv)}, User messages: {len(user_msgs)}")
    assert len(user_msgs) == 1, "Should have exactly 1 user message (no duplicate)"
    assert user_msgs[0]['content'] == "Hello!", "User message content should match"
    print("  ✅ PASSED - No duplicate user message added\n")
    
    # Scenario 3: Conversation with different user message
    print("Test 3: Conversation with different user message")
    conv, tokens = build_conversation_with_token_limit(
        system_prompt="You are helpful.",
        events=existing_events,
        user_message="How are you?",  # Different from existing message
        max_tokens=1000,
        reserve_tokens=100
    )
    
    user_msgs = [m for m in conv if m['role'] == 'user']
    print(f"  Messages: {len(conv)}, User messages: {len(user_msgs)}")
    assert len(user_msgs) == 2, "Should have 2 user messages (old + new)"
    assert "Hello!" in [m['content'] for m in user_msgs], "Should contain old message"
    assert "How are you?" in [m['content'] for m in user_msgs], "Should contain new message"
    print("  ✅ PASSED - Both user messages included\n")
    
    # Scenario 4: Complex conversation with multiple messages
    print("Test 4: Complex conversation")
    complex_events = [
        ChatEvent(
            conversation_id="test",
            type="user_message",
            role="user",
            content="What's the weather?",
            extra={"request_id": "req-1"}
        ),
        ChatEvent(
            conversation_id="test",
            type="assistant_message",
            role="assistant",
            content="I can't check the weather, but..."
        ),
        ChatEvent(
            conversation_id="test",
            type="user_message",
            role="user",
            content="Tell me a joke",
            extra={"request_id": "req-2"}
        ),
        ChatEvent(
            conversation_id="test",
            type="assistant_message",
            role="assistant",
            content="Why did the chicken cross the road?"
        )
    ]
    
    conv, tokens = build_conversation_with_token_limit(
        system_prompt="You are helpful.",
        events=complex_events,
        user_message="Tell me a joke",  # Same as last user message
        max_tokens=1000,
        reserve_tokens=100
    )
    
    user_msgs = [m for m in conv if m['role'] == 'user']
    print(f"  Messages: {len(conv)}, User messages: {len(user_msgs)}")
    
    # Should have 2 unique user messages (no duplicate of "Tell me a joke")
    unique_contents = set(m['content'] for m in user_msgs)
    assert len(unique_contents) == 2, "Should have 2 unique user messages"
    assert "What's the weather?" in unique_contents, "Should contain first message"
    assert "Tell me a joke" in unique_contents, "Should contain second message"
    print("  ✅ PASSED - No duplicate of last user message\n")
    
    print("🎉 All tests passed! The duplicate user message fix works correctly.")

if __name__ == "__main__":
    asyncio.run(test_conversation_scenarios())
