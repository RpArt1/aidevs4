"""
Local abstract base class shared by Orchestrator, Planner, and Solver.

- budget-aware via composed `BudgetGuard` (raises `BudgetExceeded` when
  iteration / wall-clock caps are hit between tool calls)
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
    AgentEventReporter,
    EventContext,
)

from .agent_helper import BudgetGuard


class SuperAgentBase(ABC):
    """
    Abstract base for super-agent roles (Orchestrator, Planner, Solver).

    Holds shared per-run state and provides:
      * `self.ctx` — this agent's own `EventContext` (nested under `parent_ctx`
        when present, fresh root context otherwise)
      * `self.events` — `AgentEventReporter` bound to `self.ctx`; typed
        wrapper around the shared `AgentEventEmitter`
      * `self.budget` — `BudgetGuard` enforcing iteration + wall-clock caps
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
        self.budget = BudgetGuard(max_iterations=max_iterations, wall_clock_s=wall_clock_s)
        self.log = get_logger(f"super_agent.{agent_id}")

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

        self.events = AgentEventReporter(emitter, self.ctx)

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

    def _process_tool_calls(
        self,
        messages: list[dict],
        tool_calls: list,
        step: int,
        execute_tool: Callable[[str, dict], str],
    ) -> bool:
        """Execute a batch of tool calls and append their results to messages.

        The assistant turn that carried ``tool_calls`` is appended first,
        followed by one ``role="tool"`` reply per call. Lifecycle events
        (``ToolCallStarted`` / ``ToolCallCompleted`` / ``AgentError``) are
        emitted around each invocation through ``self.events``.

        Args:
            messages: Chat history. Mutated in place.
            tool_calls: Tool-call objects from the LLM's assistant message.
            step: 1-indexed iteration number, recorded on events.
            execute_tool: Dispatcher mapping ``(name, args) -> result_str``.

        Returns:
            True when any tool result carries ``{"fatal": true}`` (signalling
            the outer agent loop should stop immediately); False otherwise.

        Raises:
            BudgetExceeded: When the iteration or wall-clock budget is
                exhausted between tool calls. Re-checked before each
                invocation so a runaway batch of parallel tool calls cannot
                blow past the cap.
        """
        messages.append(self._assistant_tool_calls_msg(tool_calls))

        fatal = False
        for tc in tool_calls:
            self.budget.raise_if_exceeded(step)

            tool_msg, is_fatal = self._invoke_tool_call(tc, step, execute_tool)
            messages.append(tool_msg)
            fatal = fatal or is_fatal

        return fatal

    @staticmethod
    def _assistant_tool_calls_msg(tool_calls: list) -> dict[str, Any]:
        """Serialise the model's assistant turn carrying ``tool_calls``."""
        return {
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
        }

    def _invoke_tool_call(
        self,
        tc: Any,
        step: int,
        execute_tool: Callable[[str, dict], str],
    ) -> tuple[dict, bool]:
        """Run one tool call end-to-end, emitting its lifecycle events.

        Returns:
            Tuple of ``(tool_message_to_append, fatal_flag)``. The tool
            message is the ``role="tool"`` chat entry the caller appends to
            the conversation; ``fatal_flag`` is True when the result carries
            the ``{"fatal": true}`` sentinel.
        """
        name = tc.function.name
        args = self._parse_tool_args(tc, step)

        self.events.tool_started(call_id=tc.id, tool_name=name, arguments=args, step=step)
        t0 = time()
        result, success = self._safe_execute(execute_tool, name, args, step)
        self.events.tool_completed(
            call_id=tc.id,
            tool_name=name,
            result=result,
            success=success,
            duration_ms=(time() - t0) * 1000,
            step=step,
        )

        fatal = self._is_fatal_result(result)
        if fatal:
            self.log.error(
                "fatal tool result at step=%d tool=%s — stopping agent loop",
                step, name,
            )

        return {"role": "tool", "tool_call_id": tc.id, "content": result}, fatal

    def _parse_tool_args(self, tc: Any, step: int) -> dict:
        """Parse a tool call's JSON arguments; emit ``AgentError`` on failure.

        Returns an empty dict on parse failure so the dispatcher still gets a
        well-typed payload (matching the model's intent of "call this tool").
        """
        try:
            return json.loads(tc.function.arguments)
        except json.JSONDecodeError as exc:
            self.events.agent_error(
                error_type="json_decode",
                message=str(exc),
                step=step,
                tool_name=tc.function.name,
            )
            return {}

    def _safe_execute(
        self,
        execute_tool: Callable[[str, dict], str],
        name: str,
        args: dict,
        step: int,
    ) -> tuple[str, bool]:
        """Invoke the dispatcher, converting exceptions into a JSON error.

        Returns:
            Tuple of ``(result_str, success_flag)``. On exception the result
            is a ``{"error": "..."}`` JSON payload and ``success=False``.
        """
        try:
            return execute_tool(name, args), True
        except Exception as exc:
            self.events.agent_error(
                error_type="tool_dispatch",
                message=str(exc),
                step=step,
                tool_name=name,
            )
            return json.dumps({"error": str(exc)}), False

    @staticmethod
    def _is_fatal_result(tool_result: str) -> bool:
        """Return True iff ``tool_result`` is JSON of shape ``{"fatal": true}``."""
        try:
            parsed = json.loads(tool_result)
        except (json.JSONDecodeError, TypeError):
            return False
        return isinstance(parsed, dict) and bool(parsed.get("fatal"))

    @abstractmethod
    def run(self) -> dict:
        """
        Execute this agent to a terminal outcome and return a compact result
        dict the spawner can act on. Concrete subclasses define the exact keys
        (e.g. `{"outcome": "flag", "flag": "..."}` for the Solver,
        `{"plan_path": "...", "task_family": "..."}` for the Planner).
        """
        ...
