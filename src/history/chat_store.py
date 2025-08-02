#!/usr/bin/env python3
"""
Chat History Storage Module

This module provides a high-performance async chat event storage system for the
MCP Platform. It manages conversation history, token counting, and persistent
storage with robust cross-process file locking.

Key Components:
- ChatEvent: Pydantic model representing individual chat messages, tool calls,
  and system events
- ChatRepository: Protocol defining the interface for chat storage backends
- AsyncJsonlRepo: High-performance async JSONL file storage with cross-process
  locking

Features:
- Comprehensive event tracking (user messages, assistant responses, tool calls,
  system updates)
- Token counting and usage tracking for cost monitoring
- Duplicate detection via request IDs
- Conversation-based organization
- Cross-process file locking for multi-process deployments
- Full async/await support

The module follows the MCP Platform's standards with:
- Pydantic v2 for data validation and serialization
- Type hints on all functions and methods
- Fail-fast error handling
- Modern Python syntax (union types with |)
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import uuid
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager, suppress
from datetime import UTC, datetime
from typing import Any, Literal, Protocol

import aiofiles
from filelock import FileLock
from pydantic import BaseModel, Field

from src.history.token_counter import estimate_tokens

logger = logging.getLogger(__name__)


@asynccontextmanager
async def async_file_lock(
    file_path: str, timeout: float = 30.0
) -> AsyncGenerator[None]:
    """
    Async context manager for cross-process file locking with timeout.

    This function provides robust cross-process file locking using the filelock
    library in an async-compatible way. It creates a lock file alongside the
    target file and ensures exclusive access across multiple processes.

    Args:
        file_path: Path to the file that needs to be locked
        timeout: Maximum time to wait for the lock (seconds)

    Yields:
        None: The lock is held during the context manager's execution

    Raises:
        TimeoutError: If the lock cannot be acquired within the timeout period

    Example:
        async with async_file_lock("events.jsonl"):
            # Perform file operations with exclusive access
            async with aiofiles.open("events.jsonl", "a") as f:
                await f.write("data\n")
    """
    lock_path = f"{file_path}.lock"
    file_lock = FileLock(lock_path, timeout=timeout)

    # Use run_in_executor to make the blocking lock acquisition async-compatible
    loop = asyncio.get_event_loop()

    try:
        await loop.run_in_executor(None, file_lock.acquire)
        yield
    except Exception as e:
        # Convert filelock timeout to asyncio TimeoutError
        if "timeout" in str(e).lower():
            raise TimeoutError(f"Failed to acquire file lock within {timeout}s") from e
        raise
    finally:
        # Release the lock in executor as well to ensure it's non-blocking
        with suppress(Exception):
            # Ignore release errors - lock will be cleaned up by system
            await loop.run_in_executor(None, file_lock.release)


# ---------- Canonical models (Pydantic v2) ----------

Role = Literal["system", "user", "assistant", "tool"]

class TextPart(BaseModel):
    type: Literal["text"] = "text"
    text: str

Part = TextPart  # extend later

class ToolCall(BaseModel):
    id: str
    name: str
    arguments: dict[str, Any]

class Usage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

class ChatEvent(BaseModel):
    """
    Unified event model for all chat interactions.

    This model represents every type of event that can occur during a conversation,
    from user messages to tool calls to system updates. The flexible design allows
    for future extensibility while maintaining type safety.
    """
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    conversation_id: str
    seq: int | None = None  # Auto-assigned by repository
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(UTC)
    )

    # Event classification
    type: Literal[
        "user_message", "assistant_message", "tool_call", "tool_result",
        "system_update", "meta"
    ]

    # Message content (for user_message, assistant_message, system_update)
    role: Role | None = None
    content: str | list[Part] | None = None

    # Tool interaction (for tool_call, tool_result)
    tool_calls: list[ToolCall] | None = None
    tool_call_id: str | None = None

    # LLM metadata
    model: str | None = None
    usage: Usage | None = None

    # Extensible metadata
    extra: dict[str, Any] = Field(default_factory=dict)

    # Cached token count (computed on demand)
    token_count: int | None = None

    def compute_and_cache_tokens(self) -> int:
        """Compute and cache token count for this event."""
        if self.token_count is None:
            # Convert event content to text for token counting
            text_content = ""
            if isinstance(self.content, str):
                text_content = self.content
            elif isinstance(self.content, list):
                # Handle list of Parts (like TextPart)
                text_content = " ".join(
                    part.text if hasattr(part, 'text') else str(part)
                    for part in self.content
                )
            elif self.content is not None:
                text_content = str(self.content)

            self.token_count = estimate_tokens(text_content)
        return self.token_count

    def model_post_init(self, __context: Any) -> None:
        """Post-initialization to compute tokens if content exists."""
        if self.content and self.token_count is None:
            self.compute_and_cache_tokens()


def _visible_to_llm(event: ChatEvent) -> bool:
    """
    Determine if an event should be visible to the LLM in context.

    This function implements the filtering logic for what events should be
    included when building context for LLM requests. It excludes internal
    system events and metadata that would confuse the model.

    Args:
        event: ChatEvent to evaluate for LLM visibility

    Returns:
        bool: True if the event should be included in LLM context
    """
    if event.type in ("user_message", "assistant_message", "tool_result"):
        return True
    if event.type == "system_update":
        # Only show system updates explicitly marked as visible
        return event.extra.get("visible_to_llm", False)
    # Hide meta events, tool_calls, and other internal events
    return False


def _get_last_n_token_events(
    events: list[ChatEvent], max_tokens: int
) -> list[ChatEvent]:
    """
    Get the last N tokens worth of events from a conversation.

    This function implements intelligent context window management by working
    backwards from the most recent events and including events until the token
    limit is reached. Only events visible to the LLM are considered.

    Args:
        events: List of ChatEvents in chronological order
        max_tokens: Maximum number of tokens to include

    Returns:
        list[ChatEvent]: Events that fit within the token limit, in chronological order
    """
    if not events:
        return []

    # Filter to only LLM-visible events and work backwards
    visible_events = [e for e in events if _visible_to_llm(e)]
    if not visible_events:
        return []

    # Calculate tokens for each event (compute if not cached)
    for event in visible_events:
        if event.token_count is None:
            event.compute_and_cache_tokens()

    # Work backwards to fit within token limit
    selected = []
    total_tokens = 0

    for event in reversed(visible_events):
        event_tokens = event.token_count or 0
        if total_tokens + event_tokens <= max_tokens:
            selected.append(event)
            total_tokens += event_tokens
        else:
            break

    # Return in chronological order
    return list(reversed(selected))


# ---------- Repository Protocol ----------

class ChatRepository(Protocol):
    """
    Protocol defining the interface for chat storage backends.

    This protocol ensures all repository implementations provide the same
    interface, allowing for easy swapping between in-memory, file-based,
    or database storage solutions.
    """

    async def add_event(self, event: ChatEvent) -> bool:
        """
        Add a new event to the repository.

        Args:
            event: ChatEvent to store

        Returns:
            bool: True if event was added, False if duplicate detected
        """
        ...

    async def add_events(self, events: list[ChatEvent]) -> bool:
        """
        Add multiple events to the repository in a batch operation.

        Args:
            events: List of ChatEvents to store

        Returns:
            bool: True if all events were added successfully
        """
        ...

    async def event_exists(
        self, conversation_id: str, event_type: str, request_id: str
    ) -> bool:
        """
        Check if an event exists for a specific request.

        Args:
            conversation_id: ID of the conversation
            event_type: Type of event to check for (e.g., "user_message")
            request_id: The request ID to check

        Returns:
            bool: True if the event exists, False otherwise
        """
        ...

    async def get_events(
        self, conversation_id: str, limit: int | None = None
    ) -> list[ChatEvent]:
        """
        Get events for a conversation.

        Args:
            conversation_id: ID of the conversation
            limit: Maximum number of events to return (most recent)

        Returns:
            list[ChatEvent]: Events in chronological order
        """
        ...

    async def last_n_tokens(
        self, conversation_id: str, max_tokens: int
    ) -> list[ChatEvent]:
        """
        Get the last N tokens worth of events for LLM context.

        Args:
            conversation_id: ID of the conversation
            max_tokens: Maximum tokens to include

        Returns:
            list[ChatEvent]: Events that fit within token limit
        """
        ...

    async def list_conversations(self) -> list[str]:
        """
        List all conversation IDs.

        Returns:
            list[str]: All conversation IDs in the repository
        """
        ...

    async def get_event_by_request_id(
        self, conversation_id: str, request_id: str
    ) -> ChatEvent | None:
        """
        Get an event by its request_id for O(1) retrieval.

        Args:
            conversation_id: ID of the conversation
            request_id: The request_id to look up

        Returns:
            ChatEvent | None: The event if found, None otherwise
        """
        ...

    async def get_last_assistant_reply_id(
        self, conversation_id: str, user_request_id: str
    ) -> str | None:
        """
        Get the ID of the last assistant reply for a user request.

        Args:
            conversation_id: ID of the conversation
            user_request_id: The user request ID to look up

        Returns:
            str | None: Assistant event ID if found, None otherwise
        """
        ...

    async def compact_deltas(
        self, conversation_id: str, user_request_id: str, final_content: str,
        usage: Usage | None = None, model: str | None = None
    ) -> ChatEvent:
        """
        Compact delta events into a single assistant_message.

        Args:
            conversation_id: ID of the conversation
            user_request_id: The user request that generated deltas
            final_content: The final assembled content
            usage: Token usage information
            model: Model that generated the response

        Returns:
            ChatEvent: The compacted assistant message event
        """
        ...


# ---------- Async JSONL Repository (Production Implementation) ----------

class AsyncJsonlRepo(ChatRepository):
    """
    High-performance async file I/O version with robust cross-process locking.

    This implementation uses aiofiles for truly asynchronous file operations and the
    filelock library for cross-process safety, eliminating both thread pool overhead
    and single-process locking limitations. Key improvements:

    - Native async file I/O using aiofiles instead of threading.Thread
    - Cross-process file locking using FileLock for multi-process safety
    - Async-compatible lock acquisition via run_in_executor
    - Memory-efficient streaming for large file loads
    - Full compatibility with ChatRepository protocol
    - Optional fsync for maximum durability in crash-sensitive environments

    Performance Benefits:
    - No thread pool context switching overhead for file operations
    - Better scalability under high concurrent load
    - Reduced memory pressure from thread pools
    - Native async/await integration throughout

    Cross-Process Safety:
    - Uses FileLock for true cross-process file locking (not just asyncio.Lock)
    - Safe for concurrent access from multiple processes/containers
    - Prevents file corruption in distributed deployments
    - Maintains data consistency across process boundaries

    Durability Options:
    - fsync_enabled: When True, forces data to persistent storage for crash safety
    - When False, relies on OS buffer flushing for better performance
    """

    def __init__(self, path: str = "events.jsonl", fsync_enabled: bool = True):
        self.path = path
        self.fsync_enabled = fsync_enabled
        self._lock = asyncio.Lock()
        self._by_conv: dict[str, list[ChatEvent]] = {}
        self._seq_counters: dict[str, int] = {}
        self._req_ids: dict[str, set[str]] = {}
        self._req_id_to_event: dict[str, dict[str, ChatEvent]] = {}
        self._last_assistant_ids: dict[str, dict[str, str]] = {}
        # Initialize on first access to avoid sync operations in __init__
        self._loaded = False

    async def _ensure_loaded(self) -> None:
        """Ensure data is loaded from file. Called automatically on first access."""
        if self._loaded:
            return

        async with self._lock:
            if self._loaded:  # Double-check with lock held
                return
            await self._load_async()
            self._loaded = True

    async def _load_async(self) -> None:
        """
        Asynchronously load and parse events from JSONL file during initialization.

        This method uses async file operations for better performance and
        non-blocking I/O.
        """
        try:
            async with aiofiles.open(self.path, encoding="utf-8") as f:
                async for file_line in f:
                    stripped_line = file_line.strip()
                    if not stripped_line:
                        continue

                    try:
                        data = json.loads(stripped_line)
                        ev = ChatEvent.model_validate(data)
                        self._by_conv.setdefault(ev.conversation_id, []).append(ev)

                        # Track highest sequence number per conversation
                        if ev.seq is not None:
                            current_max = self._seq_counters.get(ev.conversation_id, 0)
                            self._seq_counters[ev.conversation_id] = max(
                                current_max, ev.seq
                            )

                        # Build request_id sets for O(1) duplicate detection
                        request_id = ev.extra.get("request_id")
                        if request_id:
                            req_ids = self._req_ids.setdefault(
                                ev.conversation_id, set()
                            )
                            req_ids.add(request_id)
                            # Build request_id -> event mapping for O(1) retrieval
                            req_map = self._req_id_to_event.setdefault(
                                ev.conversation_id, {}
                            )
                            req_map[request_id] = ev

                        # Cache assistant IDs for O(1) lookup
                        if (ev.type == "assistant_message" and
                            ev.extra.get("user_request_id")):
                            conv_id = ev.conversation_id
                            asst_cache = self._last_assistant_ids.setdefault(
                                conv_id, {}
                            )
                            asst_cache[ev.extra["user_request_id"]] = ev.id

                    except (json.JSONDecodeError, ValueError) as e:
                        logger.warning(f"Skipping invalid line in {self.path}: {e}")
                        continue

        except FileNotFoundError:
            # File doesn't exist yet - this is fine for new repositories
            pass

    def _get_next_seq(self, conversation_id: str) -> int:
        """Get next sequence number for conversation. Must be called with lock held."""
        current = self._seq_counters.get(conversation_id, 0)
        next_seq = current + 1
        self._seq_counters[conversation_id] = next_seq
        return next_seq

    async def _append_async(self, event: ChatEvent) -> None:
        """
        Asynchronously append a single event to the JSONL file with crash safety.

        This method provides robust cross-process file locking using the filelock
        library, ensuring safe concurrent access from multiple processes. Key features:

        - Cross-process file locking via FileLock (not just single-process asyncio.Lock)
        - Async-compatible lock acquisition using run_in_executor
        - Atomic write operations (complete line or nothing)
        - File flushing and optional fsync for maximum durability
        - Exception handling with proper lock cleanup

        Args:
            event: ChatEvent to persist to disk

        Performance Notes:
        - Uses dedicated async file locking for true cross-process safety
        - Maintains durability guarantees while being async-compatible
        - More robust than relying solely on asyncio.Lock for multi-process scenarios
        - Optional fsync ensures data reaches persistent storage for crash safety
        """
        async with (
            async_file_lock(self.path),
            aiofiles.open(self.path, "a", encoding="utf-8") as f,
        ):
            data = json.dumps(
                event.model_dump(mode="json"), ensure_ascii=False
            )
            await f.write(data + "\n")
            await f.flush()
            # Force data to persistent storage for maximum durability
            if self.fsync_enabled:
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(None, os.fsync, f.fileno())

    async def add_event(self, event: ChatEvent) -> bool:
        """Add a new event to the repository with async file I/O."""
        await self._ensure_loaded()

        async with self._lock:
            events = self._by_conv.setdefault(event.conversation_id, [])

            # Check for duplicate by request_id if present - O(1) lookup
            request_id = event.extra.get("request_id")
            if request_id:
                req_ids = self._req_ids.setdefault(event.conversation_id, set())
                if request_id in req_ids:
                    return False  # Duplicate found, don't add

            # Assign sequence number atomically
            event.seq = self._get_next_seq(event.conversation_id)
            events.append(event)
            await self._append_async(event)

            # Add request_id to fast lookup structures
            if request_id:
                req_ids = self._req_ids.setdefault(event.conversation_id, set())
                req_ids.add(request_id)
                # Add to request_id -> event mapping for O(1) retrieval
                req_map = self._req_id_to_event.setdefault(event.conversation_id, {})
                req_map[request_id] = event

            # Cache assistant IDs for O(1) lookup
            if (event.type == "assistant_message" and
                event.extra.get("user_request_id")):
                asst_cache = self._last_assistant_ids.setdefault(
                    event.conversation_id, {}
                )
                asst_cache[event.extra["user_request_id"]] = event.id

            return True  # Successfully added

    async def get_events(
        self, conversation_id: str, limit: int | None = None
    ) -> list[ChatEvent]:
        """Get events for a conversation with async data loading."""
        await self._ensure_loaded()

        async with self._lock:
            events = self._by_conv.get(conversation_id, [])
            return events[-limit:] if limit is not None else list(events)

    async def last_n_tokens(
        self, conversation_id: str, max_tokens: int
    ) -> list[ChatEvent]:
        """Get last N tokens worth of events with async data loading."""
        await self._ensure_loaded()

        async with self._lock:
            events = self._by_conv.get(conversation_id, [])
            return _get_last_n_token_events(events, max_tokens)

    async def list_conversations(self) -> list[str]:
        """List all conversation IDs with async data loading."""
        await self._ensure_loaded()

        async with self._lock:
            return list(self._by_conv.keys())

    async def get_event_by_request_id(
        self, conversation_id: str, request_id: str
    ) -> ChatEvent | None:
        """Get event by request_id for O(1) retrieval with async data loading."""
        await self._ensure_loaded()

        async with self._lock:
            req_map = self._req_id_to_event.get(conversation_id, {})
            return req_map.get(request_id)

    async def get_last_assistant_reply_id(
        self, conversation_id: str, user_request_id: str
    ) -> str | None:
        """Get last assistant reply ID with async data loading."""
        await self._ensure_loaded()

        async with self._lock:
            asst_cache = self._last_assistant_ids.get(conversation_id, {})
            return asst_cache.get(user_request_id)

    async def compact_deltas(
        self, conversation_id: str, user_request_id: str, final_content: str,
        usage: Usage | None = None, model: str | None = None
    ) -> ChatEvent:
        """Compact delta events into a single assistant_message with async ops."""
        await self._ensure_loaded()

        async with self._lock:
            events = self._by_conv.get(conversation_id, [])

            # Check if assistant message already exists for this user_request_id
            req_ids = self._req_ids.setdefault(conversation_id, set())
            assistant_req_id = f"assistant:{user_request_id}"
            if assistant_req_id in req_ids:
                # Find and return existing assistant message
                for event in events:
                    if (event.type == "assistant_message" and
                        event.extra.get("user_request_id") == user_request_id):
                        return event
                # FAIL FAST: Inconsistent state detected
                raise ValueError(
                    f"Data corruption detected: Assistant request_id "
                    f"{assistant_req_id} exists in index but corresponding event "
                    f"not found in conversation {conversation_id}. This indicates "
                    f"a corrupted repository state that must be fixed."
                )

            # Remove all delta events for this user_request_id
            original_count = len(events)
            events[:] = [
                e for e in events
                if not (e.type == "meta" and
                       e.extra.get("kind") == "assistant_delta" and
                       e.extra.get("user_request_id") == user_request_id)
            ]
            removed_count = original_count - len(events)
            logger.info(f"Removed {removed_count} delta events for {user_request_id}")

            # Create single assistant_message event
            assistant_event = ChatEvent(
                conversation_id=conversation_id,
                type="assistant_message",
                role="assistant",
                content=final_content,
                usage=usage,
                model=model,
                extra={"user_request_id": user_request_id}
            )
            assistant_event.seq = self._get_next_seq(conversation_id)
            assistant_event.compute_and_cache_tokens()

            events.append(assistant_event)
            await self._append_async(assistant_event)

            # Add to request_id set to prevent duplicate compaction
            req_ids.add(assistant_req_id)
            # Add to request_id -> event mapping for O(1) retrieval
            req_map = self._req_id_to_event.setdefault(conversation_id, {})
            req_map[assistant_req_id] = assistant_event

            # Cache assistant ID for O(1) lookup
            asst_cache = self._last_assistant_ids.setdefault(conversation_id, {})
            asst_cache[user_request_id] = assistant_event.id

            return assistant_event

    async def add_events(self, events: list[ChatEvent]) -> bool:
        """
        Add multiple events to the repository in a batch operation.

        This method provides efficient bulk insertion with cross-process locking
        and maintains consistency with the single event add operation.

        Args:
            events: List of ChatEvents to store

        Returns:
            bool: True if all events were added successfully
        """
        await self._ensure_loaded()

        if not events:
            return True

        async with self._lock:
            # Process each event similar to add_event but without individual writes
            processed_events = []

            for event in events:
                # Check for duplicate by request_id if present
                request_id = event.extra.get("request_id")
                if request_id:
                    req_ids = self._req_ids.setdefault(event.conversation_id, set())
                    if request_id in req_ids:
                        continue  # Skip duplicate

                # Assign sequence number
                event.seq = self._get_next_seq(event.conversation_id)

                # Add to in-memory structures
                conv_events = self._by_conv.setdefault(event.conversation_id, [])
                conv_events.append(event)
                processed_events.append(event)

                # Update request_id tracking
                if request_id:
                    req_ids.add(request_id)
                    req_map = self._req_id_to_event.setdefault(
                        event.conversation_id, {}
                    )
                    req_map[request_id] = event

                # Cache assistant IDs for O(1) lookup
                if (event.type == "assistant_message" and
                    event.extra.get("user_request_id")):
                    asst_cache = self._last_assistant_ids.setdefault(
                        event.conversation_id, {}
                    )
                    asst_cache[event.extra["user_request_id"]] = event.id

            # Batch write to file
            if processed_events:
                await self._append_events_async(processed_events)

            return True

    async def _append_events_async(self, events: list[ChatEvent]) -> None:
        """
        Asynchronously append multiple events to the JSONL file with crash safety.

        Args:
            events: List of ChatEvents to persist to disk
        """
        async with (
            async_file_lock(self.path),
            aiofiles.open(self.path, "a", encoding="utf-8") as f,
        ):
            for event in events:
                data = json.dumps(
                    event.model_dump(mode="json"), ensure_ascii=False
                )
                await f.write(data + "\n")
            await f.flush()
            # Force data to persistent storage for maximum durability
            if self.fsync_enabled:
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(None, os.fsync, f.fileno())

    async def event_exists(
        self, conversation_id: str, event_type: str, request_id: str  # noqa: ARG002
    ) -> bool:
        """
        Check if an event exists for a specific request.

        Args:
            conversation_id: ID of the conversation
            event_type: Type of event to check for (e.g., "user_message")
            request_id: The request ID to check

        Returns:
            bool: True if the event exists, False otherwise

        Note:
            Currently uses request_id for O(1) lookup regardless of event_type.
            Future implementations may filter by event_type for more specificity.
        """
        await self._ensure_loaded()

        async with self._lock:
            req_ids = self._req_ids.get(conversation_id, set())
            return request_id in req_ids


# ---------- Demo Script ----------

if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    async def main():
        """Demo script showing AsyncJsonlRepo usage with fsync options."""
        # Demo with fsync enabled for maximum durability
        repo = AsyncJsonlRepo("demo_events.jsonl", fsync_enabled=True)
        conv_id = str(uuid.uuid4())

        # Add a user message
        user_event = ChatEvent(
            conversation_id=conv_id,
            type="user_message",
            role="user",
            content="Hello, world!"
        )
        await repo.add_event(user_event)
        logger.info(f"Added user event with seq: {user_event.seq} (fsync enabled)")

        # Add an assistant response
        assistant_event = ChatEvent(
            conversation_id=conv_id,
            type="assistant_message",
            role="assistant",
            content="Hi there! How can I help you today?",
            usage=Usage(prompt_tokens=10, completion_tokens=15, total_tokens=25)
        )
        await repo.add_event(assistant_event)
        logger.info(
            f"Added assistant event with seq: {assistant_event.seq} (fsync enabled)"
        )

        # Demo with fsync disabled for higher performance
        fast_repo = AsyncJsonlRepo("demo_events_fast.jsonl", fsync_enabled=False)
        await fast_repo.add_event(ChatEvent(
            conversation_id=conv_id,
            type="user_message",
            role="user",
            content="Fast write without fsync"
        ))
        logger.info("Added event with fsync disabled (higher performance)")

        # Retrieve events
        events = await repo.get_events(conv_id)
        logger.info(f"Retrieved {len(events)} events")

        # Test token-based retrieval
        token_events = await repo.last_n_tokens(conv_id, 1000)
        logger.info(f"Token-filtered events: {len(token_events)}")

        logger.info("Demo completed successfully!")
        logger.info(
            "fsync_enabled=True: Maximum durability for crash-sensitive environments"
        )
        logger.info(
            "fsync_enabled=False: Higher performance when OS-level caching "
            "is sufficient"
        )

    asyncio.run(main())
