"""Python client for superfluid's native session API over the `--web` transport."""

from ._client import (
    ASSISTANT,
    REASONING,
    SYSTEM,
    TEXT,
    TOOL,
    TOOL_CALL,
    USER,
    Chunk,
    Client,
    Event,
    Reply,
    Sampling,
    Session,
    SessionBusy,
    Stream,
    SuperfluidError,
    ToolCall,
    drop_events,
    replace_events,
)

__all__ = [
    "Client",
    "Session",
    "Stream",
    "Sampling",
    "Event",
    "Chunk",
    "Reply",
    "ToolCall",
    "SuperfluidError",
    "SessionBusy",
    "drop_events",
    "replace_events",
    "SYSTEM",
    "USER",
    "ASSISTANT",
    "TOOL",
    "TEXT",
    "REASONING",
    "TOOL_CALL",
]

__version__ = "0.1.0"
