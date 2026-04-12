"""
In-process pub/sub event system for agent lifecycle observability.

Event types
-----------
Agent lifecycle:    AgentStarted, AgentCompleted, AgentError, IterationLimitReached
LLM generation:     GenerationCompleted
Tool calls:         ToolCallStarted, ToolCallCompleted

Usage::

    from time import time
    from uuid import uuid4
    from common.events import (
        AgentEventEmitter, EventContext,
        AgentStarted, AgentCompleted,
        GenerationCompleted,
        ToolCallStarted, ToolCallCompleted,
        subscribe_event_logger, LangfuseSubscriber,
    )

    emitter = AgentEventEmitter()
    subscribe_event_logger(emitter)
    LangfuseSubscriber().attach(emitter)

    ctx = EventContext(
        trace_id=str(uuid4()), session_id="s1",
        agent_id="my_agent", root_agent_id="my_agent",
        depth=0, timestamp=time() * 1000,
    )

    emitter.emit(AgentStarted(type="agent.started", ctx=ctx))
    # ... per step: GenerationCompleted, ToolCallStarted/ToolCallCompleted ...
    emitter.emit(AgentCompleted(type="agent.completed", ctx=ctx, duration_ms=..., result=...))
"""

from .event_types import (
    EventContext,
    # agent lifecycle
    AgentStarted,
    AgentCompleted,
    AgentError,
    IterationLimitReached,
    # LLM generation
    GenerationCompleted,
    # tool calls
    ToolCallStarted,
    ToolCallCompleted,
)
from .emitter import AgentEventEmitter
from .subscriber import subscribe_event_logger
from .langfuse_subscriber import LangfuseSubscriber

__all__ = [
    "EventContext",
    # agent lifecycle
    "AgentStarted",
    "AgentCompleted",
    "AgentError",
    "IterationLimitReached",
    # LLM generation
    "GenerationCompleted",
    # tool calls
    "ToolCallStarted",
    "ToolCallCompleted",
    # infrastructure
    "AgentEventEmitter",
    "subscribe_event_logger",
    "LangfuseSubscriber",
]
