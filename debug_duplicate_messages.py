#!/usr/bin/env python3

"""
Debug script to reproduce and analyze the duplicate user message issue.

This script will trace through the conversation building flow to identify
where user messages are being duplicated when using OpenRouter vs OpenAI.
"""

import asyncio
import json
import logging
from typing import Any

# Configure logging to show detailed trace
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Mock the key components to isolate the issue
class MockRepo:
    def __init__(self):
        self.events = []
    
    async def event_exists(self, conv_id: str, event_type: str, request_id: str) -> bool:
        return any(
            e["conversation_id"] == conv_id 
            and e["type"] == event_type 
            and e.get("extra", {}).get("request_id") == request_id
            for e in self.events
        )
    
    async def add_event(self, event):
        logger.debug(f"Adding event: {event}")
        # Convert to dict for storage
        event_dict = {
            "conversation_id": event.conversation_id,
            "type": event.type,
            "role": event.role,
            "content": event.content,
            "extra": event.extra
        }
        self.events.append(event_dict)
        return True
    
    async def last_n_tokens(self, conv_id: str, max_tokens: int):
        # Return events for this conversation
        conv_events = [e for e in self.events if e["conversation_id"] == conv_id]
        logger.debug(f"Retrieved {len(conv_events)} events for conversation {conv_id}")
        
        # Convert back to ChatEvent-like objects
        from src.history.models import ChatEvent, Usage
        result = []
        for e in conv_events:
            result.append(ChatEvent(
                conversation_id=e["conversation_id"],
                type=e["type"],
                role=e["role"],
                content=e["content"],
                extra=e.get("extra", {}),
                usage=Usage()
            ))
        return result

# Mock ChatEvent
class MockChatEvent:
    def __init__(self, conversation_id: str, type: str, role: str, content: str, extra: dict):
        self.conversation_id = conversation_id
        self.type = type
        self.role = role
        self.content = content
        self.extra = extra
    
    def compute_and_cache_tokens(self):
        return len(self.content.split()) * 1.3  # Rough token estimate

async def debug_conversation_flow():
    """Debug the conversation building flow to identify duplicate messages."""
    
    # Import the actual modules
    from src.history.conversation_utils import build_conversation_with_token_limit
    from src.history.models import ChatEvent, Usage
    
    # Test scenario
    conversation_id = "test-conv-123"
    user_message = "Hello, how are you?"
    request_id = "req-456"
    system_prompt = "You are a helpful assistant."
    
    # Create mock repo
    repo = MockRepo()
    
    print("\n=== DEBUGGING DUPLICATE USER MESSAGE ISSUE ===\n")
    
    # Step 1: Simulate user message persistence (what happens in _handle_user_message_persistence)
    print("Step 1: Persisting user message to database...")
    user_event = ChatEvent(
        conversation_id=conversation_id,
        seq=0,
        type="user_message", 
        role="user",
        content=user_message,
        extra={"request_id": request_id}
    )
    user_event.compute_and_cache_tokens()
    await repo.add_event(user_event)
    
    # Step 2: Retrieve events from database (what happens in _build_conversation_history)
    print("\nStep 2: Retrieving events from database...")
    events = await repo.last_n_tokens(conversation_id, 4000)
    print(f"Retrieved events: {len(events)}")
    for i, event in enumerate(events):
        print(f"  Event {i}: type={event.type}, role={event.role}, content='{event.content}'")
    
    # Step 3: Build conversation with token limit (what happens in build_conversation_with_token_limit)
    print("\nStep 3: Building conversation with token limit...")
    conv, tokens = build_conversation_with_token_limit(
        system_prompt=system_prompt,
        events=events,  # This includes the user message we just persisted!
        user_message=user_message,  # This is added again!
        max_tokens=4000,
        reserve_tokens=500
    )
    
    print(f"Final conversation ({tokens} tokens):")
    for i, msg in enumerate(conv):
        print(f"  Message {i}: role={msg['role']}, content='{msg['content']}'")
    
    # Check for duplicates
    print("\n=== ANALYSIS ===")
    user_messages = [msg for msg in conv if msg['role'] == 'user']
    print(f"Found {len(user_messages)} user messages in conversation:")
    for i, msg in enumerate(user_messages):
        print(f"  User message {i}: '{msg['content']}'")
    
    if len(user_messages) > 1:
        print("\n❌ DUPLICATE USER MESSAGES DETECTED!")
        print("This indicates the fix has not resolved the issue.")
    else:
        print("\n✅ No duplicate user messages found - fix successful!")
        print("The conversation building logic now correctly avoids adding duplicate user messages.")

if __name__ == "__main__":
    asyncio.run(debug_conversation_flow())
