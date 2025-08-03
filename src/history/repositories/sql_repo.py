# src/history/repositories/sql_repo.py
from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Any

import aiosqlite

from src.history.models import ChatEvent, ToolCall, Usage
from src.history.repositories.base import ChatRepository

logger = logging.getLogger(__name__)


class AsyncSqlRepo(ChatRepository):
    """
    SQL implementation of ChatRepository with configurable persistence.
    Uses SQLite for simplicity; swap aiosqlite with asyncpg or other drivers
    when moving to another database.
    """

    def __init__(
        self,
        db_path: str = "events.db",
        persistence_config: dict[str, Any] | None = None
    ):
        self.db_path = db_path
        self._init_lock = asyncio.Lock()
        self._initialized = False
        self._connection_lock = asyncio.Lock()  # Serialize database access
        self._connection: aiosqlite.Connection | None = None

        # Persistence configuration
        self.persistence_config = persistence_config or {
            "enabled": True,
            "retention_policy": "token_limit",
            "max_tokens_per_conversation": 8000,
            "max_messages_per_conversation": 100,
            "retention_days": 30,
            "clear_on_startup": False
        }

    async def _initialize(self) -> None:
        """
        Lazily create tables and indices on first use.
        """
        if self._initialized:
            return
        async with self._init_lock:
            if self._initialized:
                return

            # Create persistent connection
            self._connection = await aiosqlite.connect(self.db_path)

            # Configure SQLite for better concurrency
            await self._connection.execute("PRAGMA journal_mode=WAL")
            await self._connection.execute("PRAGMA synchronous=NORMAL")
            await self._connection.execute("PRAGMA cache_size=10000")
            await self._connection.execute("PRAGMA temp_store=memory")
            # 30 second timeout
            await self._connection.execute("PRAGMA busy_timeout=30000")

            await self._connection.execute("""
                CREATE TABLE IF NOT EXISTS chat_events (
                    id TEXT PRIMARY KEY,
                    conversation_id TEXT NOT NULL,
                    seq INTEGER,
                    timestamp TEXT,
                    type TEXT NOT NULL,
                    role TEXT,
                    content TEXT,
                    tool_calls TEXT,
                    tool_call_id TEXT,
                    model TEXT,
                    usage TEXT,
                    extra TEXT,
                    token_count INTEGER,
                    request_id TEXT
                )
            """)
            await self._connection.execute("""
                CREATE UNIQUE INDEX IF NOT EXISTS idx_request
                ON chat_events(conversation_id, request_id)
            """)
            await self._connection.execute("""
                CREATE INDEX IF NOT EXISTS idx_conversation_seq
                ON chat_events(conversation_id, seq)
            """)
            await self._connection.commit()

            # Handle persistence settings
            await self._handle_persistence_on_startup()
            self._initialized = True

    async def close(self) -> None:
        """
        Close the persistent database connection.
        """
        if self._connection:
            await self._connection.close()
            self._connection = None
            self._initialized = False

    async def __aenter__(self) -> AsyncSqlRepo:
        """Async context manager entry."""
        await self._initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        await self.close()

    async def _handle_persistence_on_startup(self) -> None:
        """Handle persistence settings during initialization."""
        if self.persistence_config["clear_on_startup"]:
            # Clear data regardless of persistence enabled/disabled
            if self.persistence_config["enabled"]:
                logger.info(
                    "Persistence enabled with clear_on_startup=True, "
                    "clearing all conversations but will persist new data"
                )
            else:
                logger.info(
                    "Persistence disabled with clear_on_startup=True, "
                    "clearing all conversations"
                )
            await self.clear_all_conversations()
        elif self.persistence_config["enabled"]:
            # Persistence enabled, no clearing - apply retention policies
            logger.info(
                "Persistence enabled, applying retention policies to existing data"
            )
            await self._apply_retention_policies()
        else:
            # Persistence disabled, no clearing - just keep existing data
            logger.info(
                "Persistence disabled but clear_on_startup=False, "
                "keeping existing data"
            )

    async def clear_all_conversations(self) -> None:
        """Clear all conversation data from the database."""
        if not self._connection:
            raise RuntimeError("Database connection not available")

        async with self._connection_lock:
            await self._connection.execute("DELETE FROM chat_events")
            await self._connection.commit()
            logger.info("Cleared all conversation history")

    async def _apply_retention_policies(self) -> None:
        """Apply retention policies to existing conversations."""
        retention_policy = self.persistence_config["retention_policy"]

        if retention_policy == "unlimited":
            return  # No cleanup needed

        if not self._connection:
            raise RuntimeError("Database connection not available")

        # Get conversations directly to avoid circular dependency
        async with self._connection_lock:
            cursor = await self._connection.execute(
                "SELECT DISTINCT conversation_id FROM chat_events"
            )
            rows = await cursor.fetchall()
            conversations = [row[0] for row in rows]

        for conv_id in conversations:
            await self._apply_retention_to_conversation(conv_id, retention_policy)

    async def _apply_retention_to_conversation(
        self, conversation_id: str, policy: str
    ) -> None:
        """Apply retention policy to a specific conversation."""
        if not self._connection:
            raise RuntimeError("Database connection not available")

        async with self._connection_lock:
            if policy == "token_limit":
                max_tokens = self.persistence_config["max_tokens_per_conversation"]
                # Keep only events that fit within token limit (from most recent)
                cursor = await self._connection.execute("""
                    SELECT id, token_count FROM chat_events
                    WHERE conversation_id = ?
                    ORDER BY timestamp DESC
                """, (conversation_id,))
                rows = list(await cursor.fetchall())

                total_tokens = 0
                keep_ids = []
                for event_id, token_count in rows:
                    if total_tokens + (token_count or 0) <= max_tokens:
                        keep_ids.append(event_id)
                        total_tokens += (token_count or 0)
                    else:
                        break

                if keep_ids and len(keep_ids) < len(rows):
                    # Delete events not in keep_ids
                    placeholders = ",".join(["?" for _ in keep_ids])
                    await self._connection.execute(f"""
                        DELETE FROM chat_events
                        WHERE conversation_id = ? AND id NOT IN ({placeholders})
                    """, [conversation_id, *keep_ids])
                    deleted_count = len(rows) - len(keep_ids)
                    conv_short = conversation_id[:8]
                    logger.info(
                        f"Removed {deleted_count} old events from "
                        f"conversation {conv_short}... (token limit)"
                    )

            elif policy == "message_count":
                max_messages = self.persistence_config["max_messages_per_conversation"]
                # Keep only the most recent N messages
                await self._connection.execute("""
                    DELETE FROM chat_events
                    WHERE conversation_id = ? AND id NOT IN (
                        SELECT id FROM chat_events
                        WHERE conversation_id = ?
                        ORDER BY timestamp DESC
                        LIMIT ?
                    )
                """, (conversation_id, conversation_id, max_messages))

            elif policy == "time_based":
                retention_days = self.persistence_config["retention_days"]
                cutoff_date = (
                    datetime.now().replace(microsecond=0) -
                    timedelta(days=retention_days)
                )
                await self._connection.execute("""
                    DELETE FROM chat_events
                    WHERE conversation_id = ? AND timestamp < ?
                """, (conversation_id, cutoff_date.isoformat()))

            await self._connection.commit()

    async def add_event(self, event: ChatEvent) -> bool:
        await self._initialize()

        # If persistence is disabled, don't store events
        if not self.persistence_config["enabled"]:
            return True  # Pretend we stored it successfully

        if not self._connection:
            raise RuntimeError("Database connection not available")

        # Extract request_id if present; stored in extra
        request_id = event.extra.get("request_id")

        async with self._connection_lock:
            try:
                # Auto-assign sequence number if not set
                if event.seq is None or event.seq == 0:
                    cursor = await self._connection.execute("""
                        SELECT COALESCE(MAX(seq), 0) + 1
                        FROM chat_events
                        WHERE conversation_id = ?
                    """, (event.conversation_id,))
                    result = await cursor.fetchone()
                    await cursor.close()
                    next_seq = result[0] if result else 1
                    event.seq = next_seq

                content_str = self._serialize_content(event.content)
                tool_calls_str = json.dumps([
                    tc.model_dump() for tc in (event.tool_calls or [])
                ])
                usage_str = json.dumps(
                    event.usage.model_dump() if event.usage else None
                )

                await self._connection.execute("""
                    INSERT INTO chat_events (
                        id, conversation_id, seq, timestamp, type, role, content,
                        tool_calls, tool_call_id, model, usage, extra,
                        token_count, request_id
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    event.id,
                    event.conversation_id,
                    event.seq,
                    event.timestamp.isoformat(),
                    event.type,
                    event.role,
                    content_str,
                    tool_calls_str,
                    event.tool_call_id,
                    event.model,
                    usage_str,
                    json.dumps(event.extra),
                    event.token_count,
                    request_id,
                ))
                await self._connection.commit()
                return True
            except aiosqlite.IntegrityError:
                # duplicate request_id
                return False

    async def add_events(self, events: list[ChatEvent]) -> bool:
        await self._initialize()

        if not self._connection:
            raise RuntimeError("Database connection not available")

        async with self._connection_lock:
            try:
                # Group events by conversation for sequence assignment
                conv_events: dict[str, list[ChatEvent]] = {}
                for event in events:
                    conv_events.setdefault(event.conversation_id, []).append(event)

                # Assign sequence numbers per conversation
                for conv_id, conv_event_list in conv_events.items():
                    # Get current max sequence for this conversation
                    cursor = await self._connection.execute("""
                        SELECT COALESCE(MAX(seq), 0)
                        FROM chat_events
                        WHERE conversation_id = ?
                    """, (conv_id,))
                    result = await cursor.fetchone()
                    await cursor.close()
                    current_max = result[0] if result else 0

                    # Assign sequential numbers
                    for i, event in enumerate(conv_event_list):
                        if event.seq is None or event.seq == 0:
                            event.seq = current_max + i + 1

                # Insert all events
                for event in events:
                    request_id = event.extra.get("request_id")
                    content_str = self._serialize_content(event.content)
                    tool_calls_str = json.dumps([
                        tc.model_dump() for tc in (event.tool_calls or [])
                    ])
                    usage_str = json.dumps(
                        event.usage.model_dump() if event.usage else None
                    )

                    await self._connection.execute("""
                        INSERT INTO chat_events (
                            id, conversation_id, seq, timestamp, type, role,
                            content, tool_calls, tool_call_id, model, usage,
                            extra, token_count, request_id
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        event.id,
                        event.conversation_id,
                        event.seq,
                        event.timestamp.isoformat(),
                        event.type,
                        event.role,
                        content_str,
                        tool_calls_str,
                        event.tool_call_id,
                        event.model,
                        usage_str,
                        json.dumps(event.extra),
                        event.token_count,
                        request_id,
                    ))
                await self._connection.commit()
                return True
            except aiosqlite.IntegrityError:
                return False

    async def get_events(
        self, conversation_id: str, limit: int | None = None
    ) -> list[ChatEvent]:
        await self._initialize()

        # If persistence is disabled, return empty list
        if not self.persistence_config["enabled"]:
            return []

        if not self._connection:
            raise RuntimeError("Database connection not available")

        async with self._connection_lock:
            if limit is not None:
                query = (
                    "SELECT * FROM chat_events WHERE conversation_id = ? "
                    "ORDER BY seq LIMIT ?"
                )
                params = (conversation_id, limit)
            else:
                query = (
                    "SELECT * FROM chat_events WHERE conversation_id = ? "
                    "ORDER BY seq"
                )
                params = (conversation_id,)

            cursor = await self._connection.execute(query, params)
            rows = await cursor.fetchall()
            await cursor.close()
        return [self._row_to_event(row) for row in rows]

    async def last_n_tokens(
        self, conversation_id: str, max_tokens: int
    ) -> list[ChatEvent]:
        await self._initialize()

        # If persistence is disabled, return empty list
        if not self.persistence_config["enabled"]:
            return []

        if not self._connection:
            raise RuntimeError("Database connection not available")

        # Fetch events in reverse order and accumulate tokens until limit
        async with self._connection_lock:
            cursor = await self._connection.execute(
                "SELECT * FROM chat_events WHERE conversation_id = ? "
                "ORDER BY seq DESC",
                (conversation_id,)
            )
            events: list[ChatEvent] = []
            total = 0
            async for row in cursor:
                event = self._row_to_event(row)
                tokens = event.token_count or 0
                if total + tokens > max_tokens:
                    break
                events.insert(0, event)  # prepend to maintain chronological order
                total += tokens
            await cursor.close()
        return events

    async def list_conversations(self) -> list[str]:
        await self._initialize()

        if not self._connection:
            raise RuntimeError("Database connection not available")

        async with self._connection_lock:
            cursor = await self._connection.execute(
                "SELECT DISTINCT conversation_id FROM chat_events"
            )
            rows = await cursor.fetchall()
            await cursor.close()
        return [row[0] for row in rows]

    async def get_event_by_request_id(
        self, conversation_id: str, request_id: str
    ) -> ChatEvent | None:
        await self._initialize()

        if not self._connection:
            raise RuntimeError("Database connection not available")

        async with self._connection_lock:
            cursor = await self._connection.execute(
                "SELECT * FROM chat_events WHERE conversation_id = ? "
                "AND request_id = ?",
                (conversation_id, request_id)
            )
            row = await cursor.fetchone()
            await cursor.close()
        return self._row_to_event(row) if row else None

    async def event_exists(
        self, conversation_id: str, event_type: str, request_id: str
    ) -> bool:
        """
        Check if an event exists for a specific request.
        """
        await self._initialize()

        if not self._connection:
            raise RuntimeError("Database connection not available")

        async with self._connection_lock:
            cursor = await self._connection.execute(
                "SELECT 1 FROM chat_events WHERE conversation_id = ? "
                "AND request_id = ? AND type = ?",
                (conversation_id, request_id, event_type)
            )
            row = await cursor.fetchone()
            await cursor.close()
        return row is not None

    async def get_last_assistant_reply_id(
        self, conversation_id: str, user_request_id: str
    ) -> str | None:
        await self._initialize()

        if not self._connection:
            raise RuntimeError("Database connection not available")

        async with self._connection_lock:
            cursor = await self._connection.execute("""
                SELECT id, extra FROM chat_events
                WHERE conversation_id = ? AND type = 'assistant_message'
                ORDER BY seq DESC
            """, (conversation_id,))
            async for row in cursor:
                extra = json.loads(row[1] or "{}")
                if extra.get("user_request_id") == user_request_id:
                    await cursor.close()
                    return row[0]
            await cursor.close()
        return None

    async def get_event_by_id(
        self, conversation_id: str, event_id: str
    ) -> ChatEvent | None:
        """
        Retrieve a specific event by its ID within a conversation.

        This method provides direct access to a single event without needing
        to scan through all events in the conversation, significantly improving
        performance for cached response retrieval.
        """
        await self._initialize()

        # If persistence is disabled, return None
        if not self.persistence_config["enabled"]:
            return None

        if not self._connection:
            raise RuntimeError("Database connection not available")

        async with self._connection_lock:
            cursor = await self._connection.execute(
                "SELECT * FROM chat_events WHERE conversation_id = ? AND id = ?",
                (conversation_id, event_id)
            )
            row = await cursor.fetchone()
            await cursor.close()
        return self._row_to_event(row) if row else None

    async def compact_deltas(
        self,
        conversation_id: str,
        user_request_id: str,
        final_content: str,
        usage: Usage | None = None,
        model: str | None = None,
    ) -> ChatEvent:
        # remove meta delta events and insert one assistant_message
        await self._initialize()

        if not self._connection:
            raise RuntimeError("Database connection not available")

        async with self._connection_lock:
            # Delete meta events for this user_request_id
            await self._connection.execute("""
                DELETE FROM chat_events
                WHERE conversation_id = ? AND type = 'meta'
                AND json_extract(extra, '$.kind') = 'assistant_delta'
                AND json_extract(extra, '$.user_request_id') = ?
            """, (conversation_id, user_request_id))
            await self._connection.commit()

        # Insert a new assistant_message
        new_event = ChatEvent(
            conversation_id=conversation_id,
            type="assistant_message",
            role="assistant",
            content=final_content,
            usage=usage,
            model=model,
            extra={"user_request_id": user_request_id},
        )
        await self.add_event(new_event)
        return new_event

    def _serialize_content(self, content: str | list | None) -> str | None:
        """Serialize content for database storage."""
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            return json.dumps([p.model_dump() for p in content])
        return None

    def _row_to_event(self, row: Any) -> ChatEvent:
        """
        Convert a database row (Row or tuple) into a ChatEvent instance.
        """
        # Convert Row to tuple for consistent access
        if hasattr(row, '__getitem__'):
            # Both Row and tuple support indexing
            (
                id_,
                conversation_id,
                seq,
                timestamp,
                type_,
                role,
                content,
                tool_calls,
                tool_call_id,
                model,
                usage,
                extra,
                token_count,
                request_id,
            ) = (row[i] for i in range(14))
        else:
            # Fallback for direct tuple unpacking
            (
                id_,
                conversation_id,
                seq,
                timestamp,
                type_,
                role,
                content,
                tool_calls,
                tool_call_id,
                model,
                usage,
                extra,
                token_count,
                request_id,
            ) = row
        return ChatEvent(
            id=id_,
            conversation_id=conversation_id,
            seq=seq,
            timestamp=datetime.fromisoformat(timestamp),
            type=type_,
            role=role,
            content=(
                json.loads(content)
                if content and content.startswith('[')
                else content
            ),
            tool_calls=[
                ToolCall.model_validate(tc)
                for tc in json.loads(tool_calls or "[]")
            ],
            tool_call_id=tool_call_id,
            model=model,
            usage=(
                Usage.model_validate(json.loads(usage))
                if usage and usage != "null"
                else None
            ),
            extra=json.loads(extra or "{}"),
            token_count=token_count,
        )
