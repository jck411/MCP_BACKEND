# Timezone-Aware Timestamps

## Overview
The MCP Platform now uses timezone-aware timestamps for all ChatEvent records to ensure consistency across distributed deployments in different timezones.

## Changes Made
- **ChatEvent Model**: Updated `timestamp` field to use `datetime.now(UTC)` instead of naive `datetime.now()`
- **Storage**: All timestamps are stored in UTC for consistency
- **Serialization**: Pydantic handles timezone-aware datetime serialization automatically

## Benefits

### ✅ Cross-Timezone Consistency
All events are timestamped in UTC regardless of the server's local timezone, eliminating timezone confusion in distributed deployments.

### ✅ Proper Chronological Ordering
Events can be reliably sorted chronologically across servers in different timezones.

### ✅ Local Display Conversion
UTC timestamps can be easily converted to any local timezone for display purposes.

### ✅ Backward Compatibility
Existing naive datetime timestamps in stored data continue to work without issues.

## Implementation Details

```python
# Before (naive datetime)
timestamp: datetime = Field(default_factory=datetime.now)

# After (timezone-aware UTC)
timestamp: datetime = Field(
    default_factory=lambda: datetime.now(UTC)
)
```

## Best Practices for Distributed Systems

1. **Always use UTC**: All system timestamps are stored in UTC
2. **Convert for display**: Convert to local timezone only when displaying to users
3. **Consistent comparison**: UTC timestamps can be reliably compared across servers
4. **ISO format**: Timestamps serialize to ISO 8601 format with 'Z' suffix (UTC indicator)

## Example Usage

```python
from src.history.chat_store import ChatEvent
import uuid

# Create event with timezone-aware timestamp
event = ChatEvent(
    conversation_id=str(uuid.uuid4()),
    type="user_message",
    role="user",
    content="Hello world"
)

print(event.timestamp)  # 2025-08-01 09:30:00.123456+00:00 (UTC)
print(event.timestamp.isoformat())  # 2025-08-01T09:30:00.123456Z
```

## Migration Notes
- No migration required for existing data
- New events automatically use timezone-aware timestamps
- Legacy events with naive timestamps continue to work
- All serialization/deserialization handles both formats transparently
