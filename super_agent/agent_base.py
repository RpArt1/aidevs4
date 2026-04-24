"""
Local abstract base class shared by Orchestrator, Planner, and Solver.

Standalone — does **not** inherit from `assignments.assignment.Assignment` and
imports only from `common/`. The two packages (`super_agent/` and
`assignments/`) stay fully decoupled. The ReAct tool-call plumbing here is a
re-implementation of the pattern used in `assignments/assignment.py`, tailored
to super-agent needs:

- budget-aware (raises `BudgetExceeded` when iteration / wall-clock caps are hit
  between tool calls)
- emits events through this agent's own `AgentEventEmitter` instance
- no `AssignmentService` coupling — flag submission lives in a Solver tool
- defines `run() -> dict` so the orchestrator's spawn-dispatcher can invoke any
  sub-agent uniformly and act on a compact result dict
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from pathlib import Path
from time import time
from typing import Any, Callable

from common import LLMService, get_logger
from common.events import (
    AgentEventEmitter,
    AgentError,
    EventContext,
    ToolCallCompleted,
    ToolCallStarted,
)


class BudgetExceeded(Exception):
    """
    Raised by `SuperAgentBase._process_tool_calls` (and may be raised by
    subclasses' run loops) when this agent has exhausted its iteration count or
    wall-clock budget. The spawning agent catches it and converts it into a
    structured `{"outcome": "max_iter", ...}` result.
    """

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


class SuperAgentBase(ABC):
    """
    Abstract base for super-agent roles (Orchestrator, Planner, Solver).

    Holds shared per-run state and provides:
      * `self.ctx` — this agent's own `EventContext` (nested under `parent_ctx`
        when present, fresh root context otherwise)
      * `child_ctx(child_agent_id)` — factory for a sub-agent's context
      * `_process_tool_calls(...)` — budget-aware ReAct tool-call helper
      * abstract `run() -> dict` — uniform entry point for the dispatcher
    """

    def __init__(
        self,
        *,
        agent_id: str,
        run_id: str,
        workspace: Path,
        emitter: AgentEventEmitter,
        llm: LLMService,
        max_iterations: int,
        wall_clock_s: int,
        parent_ctx: EventContext | None = None,
        session_id: str | None = None,
    ) -> None:
        self.agent_id = agent_id
        self.run_id = run_id
        self.workspace = workspace
        self.parent_ctx = parent_ctx
        self._emitter = emitter
        self.llm = llm
        self.max_iterations = max_iterations
        self.wall_clock_s = wall_clock_s
        self.log = get_logger(f"super_agent.{agent_id}")

        self._started_at: float | None = None

        now_ms = time() * 1000
        if parent_ctx is None:
            # Root agent (typically the Orchestrator). Use run_id as trace_id so
            # the entire run is a single Langfuse trace; session_id falls back
            # to run_id when not explicitly supplied.
            self.ctx = EventContext(
                trace_id=run_id,
                session_id=session_id or run_id,
                agent_id=agent_id,
                root_agent_id=agent_id,
                depth=0,
                timestamp=now_ms,
                parent_agent_id=None,
            )
        else:
            self.ctx = EventContext(
                trace_id=parent_ctx.trace_id,
                session_id=parent_ctx.session_id,
                agent_id=agent_id,
                root_agent_id=parent_ctx.root_agent_id,
                depth=parent_ctx.depth + 1,
                timestamp=now_ms,
                parent_agent_id=parent_ctx.agent_id,
            )

    def child_ctx(self, child_agent_id: str) -> EventContext:
        """
        Build a nested `EventContext` for a sub-agent this agent is about to
        spawn. Inherits `trace_id`, `session_id`, and `root_agent_id`; bumps
        `depth` by 1; sets `parent_agent_id` to this agent's id.
        """
        return EventContext(
            trace_id=self.ctx.trace_id,
            session_id=self.ctx.session_id,
            agent_id=child_agent_id,
            root_agent_id=self.ctx.root_agent_id,
            depth=self.ctx.depth + 1,
            timestamp=time() * 1000,
            parent_agent_id=self.ctx.agent_id,
        )

    def mark_started(self) -> None:
        """
        Record the wall-clock start of this agent's run loop. Subclasses must
        call this once at the top of `run()` before entering the loop, so that
        `_budget_exceeded` can enforce the wall-clock cap.
        """
        self._started_at = time()

    def _budget_exceeded(self, step: int) -> str | None:
        """
        Return a human-readable reason string when this agent has exhausted its
        iteration count or wall-clock budget; otherwise `None`.

        Iteration cap is checked against `step` (1-indexed). Wall-clock cap is
        only enforced once `mark_started()` has been called.
        """
        if step > self.max_iterations:
            return f"max_iterations={self.max_iterations} exceeded at step={step}"
        if self._started_at is not None:
            elapsed = time() - self._started_at
            if elapsed > self.wall_clock_s:
                return (
                    f"wall_clock_s={self.wall_clock_s} exceeded "
                    f"(elapsed={elapsed:.1f}s) at step={step}"
                )
        return None

    def _process_tool_calls(
        self,
        messages: list[dict],
        tool_calls: list,
        ctx: EventContext,
        step: int,
        execute_tool: Callable[[str, dict], str],
    ) -> bool:
        """
        Append the assistant's `tool_calls` and each tool's result to `messages`
        in place, emitting `ToolCallStarted`/`ToolCallCompleted`/`AgentError`
        events along the way.

        Returns `True` when any tool result carries `{"fatal": true}`, signalling
        the outer agent loop should stop immediately.

        Raises `BudgetExceeded` if the agent runs out of iteration or wall-clock
        budget between tool calls (re-checked before each invocation so a
        runaway batch of parallel tool calls cannot blow past the cap).
        """
        messages.append({
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in tool_calls
            ],
        })

        emitter = self._emitter
        fatal = False

        for tc in tool_calls:
            reason = self._budget_exceeded(step)
            if reason:
                raise BudgetExceeded(reason)

            try:
                args = json.loads(tc.function.arguments)
            except json.JSONDecodeError as exc:
                args = {}
                if emitter:
                    emitter.emit(AgentError(
                        type="agent.error",
                        ctx=ctx,
                        error_type="json_decode",
                        message=str(exc),
                        step=step,
                        tool_name=tc.function.name,
                    ))

            if emitter:
                emitter.emit(ToolCallStarted(
                    type="tool.started",
                    ctx=ctx,
                    call_id=tc.id,
                    tool_name=tc.function.name,
                    arguments=args,
                    step=step,
                ))

            t0 = time()
            try:
                tool_result = execute_tool(tc.function.name, args)
                success = True
            except Exception as exc:
                tool_result = json.dumps({"error": str(exc)})
                success = False
                if emitter:
                    emitter.emit(AgentError(
                        type="agent.error",
                        ctx=ctx,
                        error_type="tool_dispatch",
                        message=str(exc),
                        step=step,
                        tool_name=tc.function.name,
                    ))

            if emitter:
                emitter.emit(ToolCallCompleted(
                    type="tool.completed",
                    ctx=ctx,
                    call_id=tc.id,
                    tool_name=tc.function.name,
                    result=tool_result,
                    duration_ms=(time() - t0) * 1000,
                    success=success,
                    step=step,
                ))

            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": tool_result,
            })

            try:
                parsed: Any = json.loads(tool_result)
                if isinstance(parsed, dict) and parsed.get("fatal"):
                    self.log.error(
                        "fatal tool result at step=%d tool=%s — stopping agent loop",
                        step, tc.function.name,
                    )
                    fatal = True
            except (json.JSONDecodeError, TypeError):
                pass

        return fatal

    @abstractmethod
    def run(self) -> dict:
        """
        Execute this agent to a terminal outcome and return a compact result
        dict the spawner can act on. Concrete subclasses define the exact keys
        (e.g. `{"outcome": "flag", "flag": "..."}` for the Solver,
        `{"plan_path": "...", "task_family": "..."}` for the Planner).
        """
        ...
