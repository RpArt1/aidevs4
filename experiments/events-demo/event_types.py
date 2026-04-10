# Layer 1 — event definitions.
# This module is the shared vocabulary of the system.
# Both the emitter (AgentEventEmitter) and all subscribers (loggers, observers)
# import from here. Nobody else owns these shapes.

from dataclasses import dataclass


# Carried by every event — the "passport".
# Lets any subscriber correlate events (same trace_id = same run)
# without needing access to the agent or runner state.
@dataclass
class EventContext:
    trace_id: str
    timestamp: float   # epoch ms
    session_id: str
    agent_id: str
    root_agent_id: str
    depth: int
    parent_agent_id: str | None = None


# Emitted once when the agent starts its run.
# Subscribers use this to open a span / start a timer.
@dataclass
class AgentStarted:
    type: str
    ctx: EventContext


# Emitted after the LLM responds.
# Self-contained: carries the output so subscribers need no extra state.
@dataclass
class GenerationCompleted:
    type: str
    ctx: EventContext
    output: str | None = None

@dataclass
class AgentCompleted:
    type: str
    ctx: EventContext
    duration_ms: float
    result: str | None = None