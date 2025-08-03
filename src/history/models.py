# src/history/models.py
from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Any, Literal

from pydantic import BaseModel, Field

from src.history.token_counter import estimate_tokens

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
    content: str | list[Part] | None = None
    tool_calls: list[ToolCall] | None = None
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
