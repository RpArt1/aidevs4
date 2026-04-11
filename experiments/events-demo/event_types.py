"""
event_types.py — Shared vocabulary (Layer 1 of 3)
==================================================

PATTERN: Observer / Pub-Sub
This module is the *only* place where event shapes are declared.
Both the producer (runner) and all consumers (subscribers) import from here —
neither side imports the other. This is what decouples them.

  Layer 1: event_types.py        ← shared vocabulary (this file)
  Layer 2: agent_event_emitter.py ← event bus (routes events to handlers)
  Layer 3: events_demo.py + subscriber.py ← producer + consumers

Each event dataclass captures one moment in the agent lifecycle and carries
all the data a subscriber needs to react — subscribers never call back into
the agent to fetch more information.
"""

from dataclasses import dataclass


@dataclass
class EventContext:
    """
    Correlation metadata stamped on every event.

    Allows any subscriber to link events from the same run (same trace_id),
    the same agent (agent_id), or reconstruct a multi-agent call tree
    (parent_agent_id, depth, root_agent_id) — without accessing the agent object.
    """
    trace_id: str
    timestamp: float        # epoch ms — when the run started
    session_id: str
    agent_id: str
    root_agent_id: str
    depth: int              # 0 = root agent; 1+ = nested sub-agent
    parent_agent_id: str | None = None


@dataclass
class AgentStarted:
    """Emitted once at the start of a run. Subscribers use it to open a trace span."""
    type: str               # always "agent.started"
    ctx: EventContext


@dataclass
class GenerationCompleted:
    """
    Emitted after each LLM call. In a multi-step loop this fires once per
    reasoning step. Carries everything a subscriber needs — output, model,
    the original input messages, token counts, and wall-clock duration —
    so subscribers need no extra state and no callback into the runner.
    """
    type: str               # always "generation.completed"
    ctx: EventContext
    output: str | None = None
    model: str | None = None
    input: list | None = None   # the messages list sent to the LLM
    input_tokens: int = 0
    output_tokens: int = 0
    duration_ms: float = 0.0


@dataclass
class AgentCompleted:
    """
    Emitted once at the end of a run. Mirror of AgentStarted — together they
    bracket the full lifecycle. duration_ms is pre-computed by the runner so
    subscribers don't need their own timers.
    """
    type: str               # always "agent.completed"
    ctx: EventContext
    duration_ms: float
    result: str | None = None
