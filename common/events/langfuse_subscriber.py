"""Langfuse tracing subscriber for the agent event bus."""

import os
from typing import Any

from .emitter import AgentEventEmitter


class LangfuseSubscriber:
    """
    Langfuse tracing subscriber.

    Implemented as a class rather than a plain factory function because it holds
    state — open observation spans that must survive between ``agent.started``
    and ``agent.completed`` events.

    Becomes a no-op when ``LANGFUSE_SECRET_KEY`` / ``LANGFUSE_PUBLIC_KEY`` are
    absent from the environment, so assignments that don't use Langfuse are
    unaffected.

    Usage::

        sub = LangfuseSubscriber()
        sub.attach(emitter)
        # ... agent run ...
        sub.flush()
    """

    def __init__(self, tags: list[str] | None = None) -> None:
        self._lf: Any = None
        self._tags = tags or []
        self._agent_obs: dict[str, Any] = {}   # agent_id → open agent span
        self._tool_obs: dict[str, Any] = {}    # call_id  → open tool span
        self._trace_input_set: set[str] = set()  # agent_ids whose trace input is already set

        if self._is_enabled():
            self._lf = self._init_client()

    def attach(self, emitter: AgentEventEmitter) -> None:
        """Register on the event bus. Must be called before the run starts."""
        if self._lf is None:
            return
        emitter.on_any(self._handle)

    def flush(self) -> None:
        """Flush buffered Langfuse events. Safe to call when tracing is disabled."""
        if self._lf is None:
            return
        try:
            self._lf.flush()
        except Exception:
            pass

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
            case "agent.completed":
                self._on_agent_completed(event)
            case "agent.error":
                self._on_agent_error(event)
            case "agent.iteration_limit":
                self._on_iteration_limit(event)
            case "generation.completed":
                self._on_generation_completed(event)
            case "tool.started":
                self._on_tool_started(event)
            case "tool.completed":
                self._on_tool_completed(event)

    # ── Agent lifecycle ───────────────────────────────────────────────────────

    def _on_agent_started(self, event) -> None:
        from langfuse.types import TraceContext

        trace_id = self._lf.create_trace_id(seed=event.ctx.trace_id)
        obs = self._lf.start_observation(
            name=event.ctx.agent_id,
            as_type="agent",
            trace_context=TraceContext(trace_id=trace_id),
            metadata={
                "agent_id": event.ctx.agent_id,
                "session_id": event.ctx.session_id,
                "depth": event.ctx.depth,
            },
        )
        if self._tags:
            self._lf._create_trace_tags_via_ingestion(trace_id=trace_id, tags=self._tags)
        self._agent_obs[event.ctx.agent_id] = obs

    def _on_agent_completed(self, event) -> None:
        agent_id = event.ctx.agent_id
        obs = self._agent_obs.pop(agent_id, None)
        self._trace_input_set.discard(agent_id)
        if obs is None:
            return
        obs.update(output=event.result)
        if event.ctx.depth == 0:
            obs.set_trace_io(output=event.result)
        obs.end()

    def _on_agent_error(self, event) -> None:
        obs = self._agent_obs.get(event.ctx.agent_id)
        if obs is None:
            return
        obs.update(
            level="ERROR",
            status_message=event.message,
            metadata={"error_type": event.error_type, "step": event.step, "tool_name": event.tool_name},
        )

    def _on_iteration_limit(self, event) -> None:
        obs = self._agent_obs.get(event.ctx.agent_id)
        if obs is None:
            return
        obs.update(
            level="WARNING",
            status_message=f"Iteration limit reached: {event.step}/{event.max_iterations}",
        )

    # ── LLM generation ────────────────────────────────────────────────────────

    def _on_generation_completed(self, event) -> None:
        agent_obs = self._agent_obs.get(event.ctx.agent_id)
        if agent_obs is None:
            return

        agent_id = event.ctx.agent_id
        if agent_id not in self._trace_input_set:
            self._trace_input_set.add(agent_id)
            user_message = next(
                (m["content"] for m in (event.input or []) if m.get("role") == "user"),
                None,
            )
            agent_obs.set_trace_io(input=user_message)

        step_label = f"step_{event.step}" if event.step is not None else "generation"
        gen = agent_obs.start_observation(
            name=step_label,
            as_type="generation",
            model=event.model,
            input=event.input,
        )
        usage = (
            {"input": event.input_tokens, "output": event.output_tokens,
             "total": event.input_tokens + event.output_tokens}
            if event.input_tokens or event.output_tokens
            else None
        )
        gen.update(output=event.output, usage_details=usage)
        gen.end()

    # ── Tool calls ────────────────────────────────────────────────────────────

    def _on_tool_started(self, event) -> None:
        agent_obs = self._agent_obs.get(event.ctx.agent_id)
        if agent_obs is None:
            return
        span = agent_obs.start_observation(
            name=event.tool_name,
            as_type="tool",
            input=event.arguments,
            metadata={"call_id": event.call_id, "step": event.step},
        )
        self._tool_obs[event.call_id] = span

    def _on_tool_completed(self, event) -> None:
        span = self._tool_obs.pop(event.call_id, None)
        if span is None:
            return
        span.update(
            output=event.result,
            level="ERROR" if not event.success else "DEFAULT",
            metadata={"duration_ms": event.duration_ms},
        )
        span.end()
