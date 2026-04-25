"""Typed convenience wrappers around `AgentEventEmitter`.

`AgentEventReporter` binds an emitter to a single `EventContext` so callers
no longer need to thread `ctx` through every method that emits an event.
The reporter knows the wire format of each event dataclass; the emitter
stays a generic pub/sub bus.
"""

from __future__ import annotations

from typing import Any

from .emitter import AgentEventEmitter
from .event_types import (
    AgentCompleted,
    AgentError,
    AgentStarted,
    EventContext,
    GenerationCompleted,
    IterationLimitReached,
    ToolCallCompleted,
    ToolCallStarted,
)


class AgentEventReporter:
    """Context-bound facade over `AgentEventEmitter`.

    One instance per agent. Holds the `EventContext` so emission sites do
    not have to repeat it, and exposes one method per event type so the
    event dataclass schema is referenced in exactly one place.
    """

    def __init__(self, emitter: AgentEventEmitter, ctx: EventContext) -> None:
        self._emitter = emitter
        self._ctx = ctx

    @property
    def ctx(self) -> EventContext:
        return self._ctx

    def for_ctx(self, ctx: EventContext) -> "AgentEventReporter":
        """Return a sibling reporter bound to a different context.

        Useful when an agent needs to emit on behalf of a child context
        (e.g. a spawn event) without mutating its own bound context.
        """
        return AgentEventReporter(self._emitter, ctx)

    # ── agent lifecycle ────────────────────────────────────────────────────

    def agent_started(self) -> None:
        self._emitter.emit(AgentStarted(type="agent.started", ctx=self._ctx))

    def agent_completed(self, *, duration_ms: float, result: str | None) -> None:
        self._emitter.emit(AgentCompleted(
            type="agent.completed",
            ctx=self._ctx,
            duration_ms=duration_ms,
            result=result,
        ))

    def agent_error(
        self,
        *,
        error_type: str,
        message: str,
        step: int | None = None,
        tool_name: str | None = None,
    ) -> None:
        self._emitter.emit(AgentError(
            type="agent.error",
            ctx=self._ctx,
            error_type=error_type,
            message=message,
            step=step,
            tool_name=tool_name,
        ))

    def iteration_limit(self, *, max_iterations: int, step: int) -> None:
        self._emitter.emit(IterationLimitReached(
            type="agent.iteration_limit",
            ctx=self._ctx,
            max_iterations=max_iterations,
            step=step,
        ))

    # ── LLM generation ─────────────────────────────────────────────────────

    def generation_completed(
        self,
        *,
        output: str | None,
        model: str | None,
        input: list[dict[str, Any]] | None,
        input_tokens: int,
        output_tokens: int,
        duration_ms: float,
        step: int,
    ) -> None:
        self._emitter.emit(GenerationCompleted(
            type="generation.completed",
            ctx=self._ctx,
            output=output,
            model=model,
            input=input,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            duration_ms=duration_ms,
            step=step,
        ))

    # ── tool calls ─────────────────────────────────────────────────────────

    def tool_started(
        self,
        *,
        call_id: str,
        tool_name: str,
        arguments: dict,
        step: int,
    ) -> None:
        self._emitter.emit(ToolCallStarted(
            type="tool.started",
            ctx=self._ctx,
            call_id=call_id,
            tool_name=tool_name,
            arguments=arguments,
            step=step,
        ))

    def tool_completed(
        self,
        *,
        call_id: str,
        tool_name: str,
        result: str,
        duration_ms: float,
        success: bool,
        step: int,
    ) -> None:
        self._emitter.emit(ToolCallCompleted(
            type="tool.completed",
            ctx=self._ctx,
            call_id=call_id,
            tool_name=tool_name,
            result=result,
            duration_ms=duration_ms,
            success=success,
            step=step,
        ))
