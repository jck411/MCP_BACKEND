# src/history/repositories/base.py
from __future__ import annotations

from typing import Protocol

from src.history.models import ChatEvent, Usage


class ChatRepository(Protocol):
    """
    Interface for storing and retrieving chat history.
    """

    async def add_event(self, event: ChatEvent) -> bool:
        """
        Store a single chat event.  Returns True if added, False if a duplicate
        (same request_id) was detected.
        """
        ...

    async def add_events(self, events: list[ChatEvent]) -> bool:
        """
        Store multiple chat events atomically.
        """
        ...

    async def get_events(
        self, conversation_id: str, limit: int | None = None
    ) -> list[ChatEvent]:
        """
        Return chat events for a conversation in chronological order.
        If limit is provided, only the most recent `limit` events are returned.
        """
        ...

    async def last_n_tokens(
        self, conversation_id: str, max_tokens: int
    ) -> list[ChatEvent]:
        """
        Return events whose cumulative token_count does not exceed `max_tokens`,
        preserving chronological order.
        """
        ...

    async def list_conversations(self) -> list[str]:
        """
        Return a list of all conversation IDs.
        """
        ...

    async def get_event_by_request_id(
        self, conversation_id: str, request_id: str
    ) -> ChatEvent | None:
        """
        Retrieve an event by its request_id for idempotent operations.
        """
        ...

    async def event_exists(
        self, conversation_id: str, event_type: str, request_id: str
    ) -> bool:
        """
        Check if an event exists for a specific request.
        """
        ...

    async def get_last_assistant_reply_id(
        self, conversation_id: str, user_request_id: str
    ) -> str | None:
        """
        Return the ID of the last assistant reply for a given user_request_id.
        """
        ...

    async def get_event_by_id(
        self, conversation_id: str, event_id: str
    ) -> ChatEvent | None:
        """
        Retrieve a specific event by its ID within a conversation.
        """
        ...

    async def compact_deltas(
        self,
        conversation_id: str,
        user_request_id: str,
        final_content: str,
        usage: Usage | None = None,
        model: str | None = None,
    ) -> ChatEvent:
        """
        Combine delta events into a single assistant_message.  Used to collapse
        streaming partial responses into one final event.
        """
        ...
