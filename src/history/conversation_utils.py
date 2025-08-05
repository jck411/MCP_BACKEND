"""
Conversation utilities for token-aware context management.

This module provides utilities for building conversations with proper
token accounting and context window management.
"""

from __future__ import annotations

import logging
from typing import Any

from src.history.models import ChatEvent
from src.history.token_counter import count_conversation_tokens, estimate_tokens

logger = logging.getLogger(__name__)


def build_conversation_with_token_limit(
    system_prompt: str,
    events: list[ChatEvent],
    user_message: str,
    max_tokens: int,
    reserve_tokens: int,
) -> tuple[list[dict[str, Any]], int]:
    """
    Build a conversation list with token limit enforcement.

    This function builds an OpenAI-style conversation while respecting
    token limits and reserving space for the response. Uses incremental
    token counting to avoid redundant computation.

    Args:
        system_prompt: The system prompt to include
        events: List of chat events to include
        user_message: The current user message
        max_tokens: Maximum tokens for the conversation
        reserve_tokens: Tokens to reserve for the response

    Returns:
        Tuple of (conversation_list, total_tokens_used)
    """

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
        logger.warning(
            f"System prompt and user message exceed token limit. "
            f"Base: {base_tokens}, Max: {max_tokens}, Reserve: {reserve_tokens}"
        )
        base_conversation = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ]
        return base_conversation, base_tokens

    # Build conversation incrementally with token accumulation
    conversation = [{"role": "system", "content": system_prompt}]
    accumulated_tokens = system_tokens + 3  # system message + overhead

    # Process events in chronological order to maintain conversation flow
    user_message_already_included = False
    seen_user_messages = set()  # Track user message content to prevent duplicates

    for event in events:
        # Only include events that make sense in LLM conversation context
        if event.type in ("user_message", "assistant_message", "system_update"):
            # For user messages, implement deduplication
            if event.type == "user_message":
                # Check if this is a duplicate user message
                if event.content and event.content in seen_user_messages:
                    content_str = str(event.content)
                    content_preview = (content_str[:50] + "...")
                    logger.debug(f"Skipping duplicate user message: {content_preview}")
                    continue

                # Track this user message content
                if event.content:
                    seen_user_messages.add(event.content)

                # Check if this matches the current user message
                if event.content == user_message:
                    user_message_already_included = True

            # Get cached token count for this event
            event_tokens = event.compute_and_cache_tokens()
            event_overhead = 3  # message formatting overhead
            total_event_cost = event_tokens + event_overhead

            # Check if adding this event would exceed our budget
            # Account for user message that will be added (if not already included)
            user_msg_cost = 0 if user_message_already_included else (user_tokens + 3)
            projected_total = (
                accumulated_tokens + total_event_cost + user_msg_cost + 3
            )  # overhead + priming

            if projected_total + reserve_tokens > max_tokens:
                logger.debug(
                    f"Stopping conversation build at {len(conversation)} messages. "
                    f"Would use {projected_total} + {reserve_tokens} tokens."
                )
                break  # Stop adding events - we've hit our limit

            # Safe to include this event - add it to the conversation
            event_msg = {"role": event.role, "content": event.content or ""}
            conversation.append(event_msg)
            accumulated_tokens += total_event_cost

    # Append the current user message only if it wasn't already included from events
    if not user_message_already_included:
        conversation.append({"role": "user", "content": user_message})
        final_tokens = (
            accumulated_tokens + user_tokens + 3 + 3
        )  # user msg + overhead + priming
    else:
        final_tokens = (
            accumulated_tokens + 3
        )  # just priming tokens

    return conversation, final_tokens


def estimate_response_tokens(
    conversation: list[dict[str, Any]],
    conversation_limits: dict[str, int],
    response_tokens: dict[str, int],
) -> int:
    """
    Estimate how many tokens a response might need based on conversation.

    This is a heuristic based on typical response patterns.

    Args:
        conversation: The conversation to analyze
        conversation_limits: Dict with 'short', 'medium', 'long' limits
        response_tokens: Dict with 'short', 'medium', 'long', 'max' token counts
    """
    conversation_tokens = count_conversation_tokens(conversation)

    # Heuristics for response size based on configuration
    if conversation_tokens < conversation_limits["short"]:
        return response_tokens["short"]  # Short responses for short conversations
    if conversation_tokens < conversation_limits["medium"]:
        return response_tokens["medium"]  # Medium responses
    if conversation_tokens < conversation_limits["long"]:
        return response_tokens["long"]  # Longer responses for complex conversations
    return response_tokens["max"]  # Max response for very long conversations


def get_conversation_token_distribution(events: list[ChatEvent]) -> dict[str, int]:
    """
    Analyze token distribution across different event types.

    Returns a dictionary with token counts by event type.
    """
    distribution: dict[str, int] = {}

    for event in events:
        tokens = event.compute_and_cache_tokens()
        event_type = event.type
        distribution[event_type] = distribution.get(event_type, 0) + tokens

    return distribution


def optimize_conversation_for_tokens(
    events: list[ChatEvent],
    target_tokens: int,
    preserve_recent: int,
) -> list[ChatEvent]:
    """
    Optimize a conversation to fit within a token budget.

    This function intelligently removes older messages while preserving
    important context and recent messages.

    Args:
        events: List of chat events
        target_tokens: Target token count
        preserve_recent: Number of recent messages to always preserve

    Returns:
        Optimized list of chat events
    """
    if not events:
        return events

    # Always preserve recent messages
    recent_events = events[-preserve_recent:] if preserve_recent > 0 else []
    older_events = events[:-preserve_recent] if preserve_recent > 0 else events

    # Calculate tokens for recent events
    recent_tokens = sum(event.compute_and_cache_tokens() for event in recent_events)

    if recent_tokens >= target_tokens:
        logger.warning(
            f"Recent {preserve_recent} messages already use {recent_tokens} tokens, "
            f"exceeding target of {target_tokens}"
        )
        return recent_events

    # Add older events until we hit the token limit
    remaining_tokens = target_tokens - recent_tokens
    selected_older = []

    for event in reversed(older_events):
        event_tokens = event.compute_and_cache_tokens()
        if remaining_tokens >= event_tokens:
            selected_older.insert(0, event)  # Insert at beginning to maintain order
            remaining_tokens -= event_tokens
        else:
            break

    return selected_older + recent_events
