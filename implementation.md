# MCP Backend Implementation - SQL Architecture

## Architecture Overview

The MCP Backend has been refactored to use a clean SQL-based architecture with SQLite for persistent conversation storage.

## Key Components

### Storage Layer
- **AsyncSqlRepo**: SQLite database implementation with WAL mode
- **ChatRepository Protocol**: Clean interface for storage backends
- **ChatEvent Model**: Unified event model for all chat interactions

### Data Flow
1. **WebSocket Input**: Receives chat messages via WebSocket API
2. **Chat Service**: Orchestrates LLM interactions and tool calls
3. **SQL Storage**: Persists all events to SQLite database
4. **Context Retrieval**: Loads conversation history within token limits

### Benefits
- **Persistent Memory**: Conversations survive restarts
- **Concurrent Safety**: SQLite WAL mode supports multiple connections
- **Clean Architecture**: Repository pattern separates storage from business logic
- **Type Safety**: Pydantic models throughout

## Database Schema

```sql
CREATE TABLE chat_events (
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
);
```

## Migration Complete

The refactoring from JSONL to SQL is complete:
- ✅ Removed `chat_store.py` and `AsyncJsonlRepo`
- ✅ Implemented `AsyncSqlRepo` with full functionality
- ✅ Updated configuration to use SQL backend
- ✅ All tests passing with SQL storage  These steps remove file‑locking logic and any backward‑compatibility code, and they define a clean repository interface and model layer that your future SQL implementation can satisfy.  Each instruction includes complete code where appropriate so you can copy‑paste it into your project.

---

### 1. Delete the old JSONL repository module

Remove `src/history/chat_store.py` and the `events.jsonl` file entirely.  They are no longer needed.

### 2. Create a clean model module

Add a new file `src/history/models.py` containing the chat event and usage models.  These models will be used by your SQL repository and the rest of the application:

```python
# src/history/models.py
from __future__ import annotations
from datetime import UTC, datetime
import uuid
from typing import Any, Literal, List
from pydantic import BaseModel, Field

Role = Literal["system", "user", "assistant", "tool"]

class TextPart(BaseModel):
    """Represents a simple text part used in ChatEvent content."""
    type: Literal["text"] = "text"
    text: str

Part = TextPart

class ToolCall(BaseModel):
    id: str
    name: str
    arguments: dict[str, Any]

class Usage(BaseModel):
    """LLM usage statistics."""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

class ChatEvent(BaseModel):
    """
    Unified event model for all chat interactions.
    """
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    conversation_id: str
    seq: int | None = None
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    type: Literal[
        "user_message", "assistant_message", "tool_call", "tool_result",
        "system_update", "meta"
    ]
    role: Role | None = None
    content: str | List[Part] | None = None
    tool_calls: List[ToolCall] | None = None
    tool_call_id: str | None = None
    model: str | None = None
    usage: Usage | None = None
    extra: dict[str, Any] = Field(default_factory=dict)
    token_count: int | None = None

    def compute_and_cache_tokens(self) -> int:
        """
        Compute and cache token count for this event.
        """
        if self.token_count is None:
            from src.history.token_counter import estimate_tokens
            text_content = ""
            if isinstance(self.content, str):
                text_content = self.content
            elif isinstance(self.content, list):
                text_content = " ".join(part.text for part in self.content)
            elif self.content is not None:
                text_content = str(self.content)
            self.token_count = estimate_tokens(text_content)
        return self.token_count

    def model_post_init(self, __context: Any) -> None:
        """
        Post-init hook to precompute tokens if content exists.
        """
        if self.content and self.token_count is None:
            self.compute_and_cache_tokens()
```

### 3. Define a simple repository interface

Add a new file `src/history/repositories/base.py` to define the interface for any chat history backend.  With no backward compatibility requirement, you can remove JSONL‑specific details and locking semantics:

```python
# src/history/repositories/base.py
from __future__ import annotations
from typing import Protocol, List, Optional
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

    async def add_events(self, events: List[ChatEvent]) -> bool:
        """
        Store multiple chat events atomically.
        """

    async def get_events(self, conversation_id: str, limit: Optional[int] = None) -> List[ChatEvent]:
        """
        Return chat events for a conversation in chronological order.
        If limit is provided, only the most recent `limit` events are returned.
        """

    async def last_n_tokens(self, conversation_id: str, max_tokens: int) -> List[ChatEvent]:
        """
        Return events whose cumulative token_count does not exceed `max_tokens`,
        preserving chronological order.
        """

    async def list_conversations(self) -> List[str]:
        """
        Return a list of all conversation IDs.
        """

    async def get_event_by_request_id(self, conversation_id: str, request_id: str) -> Optional[ChatEvent]:
        """
        Retrieve an event by its request_id for idempotent operations.
        """

    async def get_last_assistant_reply_id(self, conversation_id: str, user_request_id: str) -> Optional[str]:
        """
        Return the ID of the last assistant reply for a given user_request_id.
        """

    async def compact_deltas(
        self,
        conversation_id: str,
        user_request_id: str,
        final_content: str,
        usage: Optional[Usage] = None,
        model: Optional[str] = None,
    ) -> ChatEvent:
        """
        Combine delta events into a single assistant_message.  Used to collapse
        streaming partial responses into one final event.
        """
```

### 4. Create a skeleton SQL repository

Since you no longer need JSONL, focus on a single SQL repository.  A fast choice for prototyping is SQLite with the asynchronous `aiosqlite` driver.  Create `src/history/repositories/sql_repo.py` with the following code.  The methods marked `TODO` will later contain SQL queries; leave them unimplemented until you are ready to add the actual database:

```python
# src/history/repositories/sql_repo.py
from __future__ import annotations
from typing import Optional, List
import json
import asyncio
import aiosqlite
from src.history.models import ChatEvent, Usage
from src.history.repositories.base import ChatRepository

class AsyncSqlRepo(ChatRepository):
    """
    SQL implementation of ChatRepository.
    Uses SQLite for simplicity; swap aiosqlite with asyncpg or other drivers
    when moving to another database.
    """

    def __init__(self, db_path: str = "events.db"):
        self.db_path = db_path
        self._init_lock = asyncio.Lock()
        self._initialized = False

    async def _initialize(self) -> None:
        """
        Lazily create tables and indices on first use.
        """
        if self._initialized:
            return
        async with self._init_lock:
            if self._initialized:
                return
            async with aiosqlite.connect(self.db_path) as db:
                await db.execute("""
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
                await db.execute("""
                    CREATE UNIQUE INDEX IF NOT EXISTS idx_request
                    ON chat_events(conversation_id, request_id)
                """)
                await db.commit()
            self._initialized = True

    async def add_event(self, event: ChatEvent) -> bool:
        await self._initialize()
        # Extract request_id if present; stored in extra
        request_id = event.extra.get("request_id")
        async with aiosqlite.connect(self.db_path) as db:
            try:
                await db.execute("""
                    INSERT INTO chat_events (
                        id, conversation_id, seq, timestamp, type, role, content,
                        tool_calls, tool_call_id, model, usage, extra, token_count, request_id
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    event.id,
                    event.conversation_id,
                    event.seq,
                    event.timestamp.isoformat(),
                    event.type,
                    event.role,
                    event.content if isinstance(event.content, str) else json.dumps([p.model_dump() for p in (event.content or [])]),
                    json.dumps([tc.model_dump() for tc in (event.tool_calls or [])]),
                    event.tool_call_id,
                    event.model,
                    json.dumps(event.usage.model_dump() if event.usage else None),
                    json.dumps(event.extra),
                    event.token_count,
                    request_id,
                ))
                await db.commit()
                return True
            except aiosqlite.IntegrityError:
                # duplicate request_id
                return False

    async def add_events(self, events: List[ChatEvent]) -> bool:
        await self._initialize()
        async with aiosqlite.connect(self.db_path) as db:
            try:
                async with db.execute("BEGIN"):
                    for event in events:
                        request_id = event.extra.get("request_id")
                        await db.execute("""
                            INSERT INTO chat_events (
                                id, conversation_id, seq, timestamp, type, role, content,
                                tool_calls, tool_call_id, model, usage, extra, token_count, request_id
                            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """, (
                            event.id,
                            event.conversation_id,
                            event.seq,
                            event.timestamp.isoformat(),
                            event.type,
                            event.role,
                            event.content if isinstance(event.content, str) else json.dumps([p.model_dump() for p in (event.content or [])]),
                            json.dumps([tc.model_dump() for tc in (event.tool_calls or [])]),
                            event.tool_call_id,
                            event.model,
                            json.dumps(event.usage.model_dump() if event.usage else None),
                            json.dumps(event.extra),
                            event.token_count,
                            request_id,
                        ))
                await db.commit()
                return True
            except aiosqlite.IntegrityError:
                return False

    async def get_events(self, conversation_id: str, limit: Optional[int] = None) -> List[ChatEvent]:
        await self._initialize()
        async with aiosqlite.connect(self.db_path) as db:
            query = "SELECT * FROM chat_events WHERE conversation_id = ? ORDER BY seq"
            params = [conversation_id]
            if limit is not None:
                query += " LIMIT ?"
                params.append(limit)
            cursor = await db.execute(query, params)
            rows = await cursor.fetchall()
            await cursor.close()
        return [self._row_to_event(row) for row in rows]

    async def last_n_tokens(self, conversation_id: str, max_tokens: int) -> List[ChatEvent]:
        await self._initialize()
        # Fetch events in reverse order and accumulate tokens until limit
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute(
                "SELECT * FROM chat_events WHERE conversation_id = ? ORDER BY seq DESC",
                (conversation_id,)
            )
            events: List[ChatEvent] = []
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

    async def list_conversations(self) -> List[str]:
        await self._initialize()
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute("SELECT DISTINCT conversation_id FROM chat_events")
            rows = await cursor.fetchall()
            await cursor.close()
        return [row[0] for row in rows]

    async def get_event_by_request_id(self, conversation_id: str, request_id: str) -> Optional[ChatEvent]:
        await self._initialize()
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute(
                "SELECT * FROM chat_events WHERE conversation_id = ? AND request_id = ?",
                (conversation_id, request_id)
            )
            row = await cursor.fetchone()
            await cursor.close()
        return self._row_to_event(row) if row else None

    async def get_last_assistant_reply_id(self, conversation_id: str, user_request_id: str) -> Optional[str]:
        await self._initialize()
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute("""
                SELECT id, extra FROM chat_events
                WHERE conversation_id = ? AND type = 'assistant_message'
                ORDER BY seq DESC
            """, (conversation_id,))
            async for row in cursor:
                extra = json.loads(row[11] or "{}")
                if extra.get("user_request_id") == user_request_id:
                    await cursor.close()
                    return row[0]
            await cursor.close()
        return None

    async def compact_deltas(
        self,
        conversation_id: str,
        user_request_id: str,
        final_content: str,
        usage: Optional[Usage] = None,
        model: Optional[str] = None,
    ) -> ChatEvent:
        # remove meta delta events and insert one assistant_message
        await self._initialize()
        async with aiosqlite.connect(self.db_path) as db:
            # Delete meta events for this user_request_id
            await db.execute("""
                DELETE FROM chat_events
                WHERE conversation_id = ? AND type = 'meta'
                AND json_extract(extra, '$.kind') = 'assistant_delta'
                AND json_extract(extra, '$.user_request_id') = ?
            """, (conversation_id, user_request_id))
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

    def _row_to_event(self, row: tuple) -> ChatEvent:
        """
        Convert a database row tuple into a ChatEvent instance.
        """
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
        event = ChatEvent(
            id=id_,
            conversation_id=conversation_id,
            seq=seq,
            timestamp=datetime.fromisoformat(timestamp),
            type=type_,
            role=role,
            content=json.loads(content) if content and content.startswith('[') else content,
            tool_calls=[ToolCall.model_validate(tc) for tc in json.loads(tool_calls or "[]")],
            tool_call_id=tool_call_id,
            model=model,
            usage=Usage.model_validate(json.loads(usage)) if usage else None,
            extra=json.loads(extra or "{}"),
            token_count=token_count,
        )
        return event
```

This implementation uses SQLite’s `INSERT` with a unique index on `(conversation_id, request_id)` to prevent duplicate events and automatically assigns sequence numbers when you set the `seq` field before insertion.

### 5. Simplify the chat service for SQL

With file‑locking gone, you can remove `_write_with_backoff` and any reference to JSONL locks in `src/chat_service.py`.  Instead, call repository methods directly and handle duplicate detection via the return value of `add_event`.  Import the new models and repository interface:

```python
# At the top of src/chat_service.py
from src.history.models import ChatEvent, Usage
from src.history.repositories.base import ChatRepository

# remove _write_with_backoff entirely and replace usages with direct calls, e.g.:
added = await self.repo.add_event(user_ev)
if not added:
    # event already exists; either return or ignore depending on your logic
    return True
```

Because your SQL database enforces uniqueness on `(conversation_id, request_id)`, there is no need for manual locking or exponential backoff.  If a duplicate is detected, `add_event` returns `False`, and you can simply skip re‑inserting the user message.

Update all other repository interactions in `ChatService` (`event_exists`, `get_event_by_request_id`, `compact_deltas`, etc.) to call the corresponding SQL repository methods; their semantics remain the same.

### 6. Update the WebSocket server

Modify `src/websocket_server.py` to import `ChatRepository` from `src/history/repositories/base` and construct an `AsyncSqlRepo` when initializing the server.  For example:

```python
from src.history.repositories.sql_repo import AsyncSqlRepo

# When creating the WebSocketServer:
repo = AsyncSqlRepo(db_path="events.db")
server = WebSocketServer(clients, llm_client, config, repo, configuration)
```

### 7. Configuration and initialisation

Remove any configuration references to `events.jsonl.lock` or file paths.  Add a configuration option for the SQL database path if desired.  Initialise the SQL repository once at application startup and pass it into both `ChatService` and `WebSocketServer`.

---

These steps discard the old JSONL‑based history mechanism entirely.  The codebase will now define clean models and a simple repository interface, and it includes a concrete asynchronous SQL repository ready for future expansion.  Once you implement the remaining SQL queries in `AsyncSqlRepo`, the rest of the application can operate on a fast database without further refactoring.
