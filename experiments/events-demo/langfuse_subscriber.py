"""
langfuse_subscriber.py — Langfuse tracing subscriber (Layer 3a of 3)
=====================================================================

PATTERN: Observer / Pub-Sub — consumer side
Mirrors subscriber.py in role: registers handlers on the event bus and reacts
to agent lifecycle events. Implemented as a class (vs. the plain function in
subscriber.py) because it holds state — open observation spans that must be
kept alive between the agent.started and agent.completed events.

NESTING WITHOUT CONTEXT MANAGERS
---------------------------------
The usual `with langfuse.start_as_current_observation(...)` keeps a span open
for the duration of a `with` block. That does not work across event boundaries
where start and end arrive in separate callbacks.

Langfuse v4 exposes an imperative API for exactly this use-case:

  lf.start_observation(name, as_type, trace_context=TraceContext(...))
      → creates a root-level span pinned to an explicit trace ID
  parent_obs.start_observation(name, as_type, ...)
      → creates a child span (auto-parented; no trace_context needed)
  obs.update(...) / obs.end()
      → update fields and close the span when the matching event arrives

HOW EVENTS MAP TO LANGFUSE OBSERVATIONS
-----------------------------------------
  agent.started       → open "agent" span, store in _agent_obs by agent_id
  generation.completed → open child "generation" span, update with output +
                         token usage, close immediately (all data in the event)
  agent.completed     → update + close the "agent" span

TRACE IDS
----------
Langfuse v4 requires 32-char lowercase hex IDs. EventContext.trace_id is a
UUID string. lf.create_trace_id(seed=ctx.trace_id) converts it
deterministically — same run always maps to the same Langfuse trace.
"""

import os
from typing import Any

from agent_event_emitter import AgentEventEmitter


class LangfuseSubscriber:
    """
    Langfuse tracing subscriber.

    Lifecycle:
      sub = LangfuseSubscriber()   ← initialise (no-op if env vars absent)
      sub.attach(emitter)           ← register on the event bus
      ... agent run ...             ← events flow in; spans are opened/closed
      sub.flush()                   ← flush buffered data before process exit
    """

    def __init__(self) -> None:
        self._lf: Any = None
        # Open agent observations keyed by agent_id.
        # Nested sub-agents each get their own entry so they don't clobber parents.
        self._agent_obs: dict[str, Any] = {}

        if self._is_enabled():
            self._lf = self._init_client()

    # ── Public interface ─────────────────────────────────────────────────────

    def attach(self, emitter: AgentEventEmitter) -> None:
        """Register the wildcard handler on the event bus.

        No-op when tracing is disabled (LANGFUSE_* env vars absent).
        Must be called before the agent run starts so no events are missed.
        """
        if self._lf is None:
            return
        emitter.on_any(self._handle)

    def flush(self) -> None:
        """Flush all buffered Langfuse events before the process exits.

        Equivalent to shutdown_tracing() in common/langfuse_tracing.py.
        Safe to call when tracing is disabled — becomes a no-op.
        """
        if self._lf is None:
            return
        try:
            self._lf.flush()
        except Exception:
            pass

    # ── Private helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _is_enabled() -> bool:
        return bool(os.getenv("LANGFUSE_SECRET_KEY") and os.getenv("LANGFUSE_PUBLIC_KEY"))

    @staticmethod
    def _init_client() -> Any:
        from langfuse import get_client
        return get_client()

    def _handle(self, event) -> None:
        match event.type:
            case "agent.started":
                self._on_agent_started(event)
            case "generation.completed":
                self._on_generation_completed(event)
            case "agent.completed":
                self._on_agent_completed(event)
            case _:
                pass  # unknown event types are silently ignored

    def _on_agent_started(self, event) -> None:
        from langfuse.types import TraceContext

        # Convert UUID trace_id → valid 32-char hex Langfuse trace ID.
        # seed= makes it deterministic: same EventContext always → same trace.
        langfuse_trace_id = self._lf.create_trace_id(seed=event.ctx.trace_id)

        obs = self._lf.start_observation(
            name=event.ctx.agent_id,
            as_type="agent",
            trace_context=TraceContext(trace_id=langfuse_trace_id),
            metadata={
                "agent_id": event.ctx.agent_id,
                "session_id": event.ctx.session_id,
                "depth": event.ctx.depth,
            },
        )

        self._agent_obs[event.ctx.agent_id] = obs

    def _on_generation_completed(self, event) -> None:
        agent_obs = self._agent_obs.get(event.ctx.agent_id)
        if agent_obs is None:
            return

        # Stamp trace-level input now that we have the messages.
        # Can't do this at agent.started because the messages don't exist yet.
        if event.ctx.depth == 0:
            agent_obs.set_trace_io(input=event.input)

        # Child span is auto-parented to agent_obs — no trace_context needed.
        gen = agent_obs.start_observation(
            name="generation",
            as_type="generation",
            model=event.model,
            input=event.input,
        )
        gen.update(
            output=event.output,
            usage_details=(
                {
                    "input": event.input_tokens,
                    "output": event.output_tokens,
                    "total": event.input_tokens + event.output_tokens,
                }
                if event.input_tokens or event.output_tokens
                else None
            ),
        )
        gen.end()

    def _on_agent_completed(self, event) -> None:
        obs = self._agent_obs.pop(event.ctx.agent_id, None)
        if obs is None:
            return

        obs.update(output=event.result)
        if event.ctx.depth == 0:
            obs.set_trace_io(output=event.result)
        obs.end()
