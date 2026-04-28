"""Mock Solver agent for early end-to-end orchestration smoke tests.

This is intentionally not the real ReAct Solver from the design plan. It is a
contract-compatible placeholder that lets us verify:

* Docker / CLI startup reaches the orchestrator.
* The orchestrator can spawn Planner, then Solver.
* Solver output can drive the orchestrator to call `finish`.

The real solver will replace this mock with a ReAct loop and solver tools.
"""

from __future__ import annotations

import json
from pathlib import Path
from time import time
from typing import Any

from common import LLMService
from common.events import (
    AgentCompleted,
    AgentEventEmitter,
    AgentError,
    AgentStarted,
    EventContext,
)

from .agent_base import SuperAgentBase

DEFAULT_MAX_ITERATIONS = 1
DEFAULT_WALL_CLOCK_S = 60
MOCK_FLAG = "MOCK_FLAG"


class SolverAgent(SuperAgentBase):
    """Temporary mock solver that returns a fake flag outcome.

    Args:
        plan_path: Path to the plan JSON produced by `PlannerAgent`.
        feedback: Optional orchestrator feedback for retry attempts.
        run_id: Shared super-agent run id.
        workspace: Per-run workspace directory.
        emitter: Shared event emitter.
        llm: Shared LLM service. Unused by the mock, kept for constructor
            compatibility with the future real solver.
        agent_id: Unique solver agent id for this spawn.
        max_iterations: Budget cap retained for base-class compatibility.
        wall_clock_s: Wall-clock cap retained for base-class compatibility.
        parent_ctx: Parent orchestrator event context.
        session_id: Optional tracing session id.
    """

    def __init__(
        self,
        *,
        plan_path: Path,
        run_id: str,
        workspace: Path,
        emitter: AgentEventEmitter,
        llm: LLMService,
        feedback: str | None = None,
        agent_id: str = "solver",
        max_iterations: int = DEFAULT_MAX_ITERATIONS,
        wall_clock_s: int = DEFAULT_WALL_CLOCK_S,
        parent_ctx: EventContext | None = None,
        session_id: str | None = None,
    ) -> None:
        super().__init__(
            agent_id=agent_id,
            run_id=run_id,
            workspace=workspace,
            emitter=emitter,
            llm=llm,
            max_iterations=max_iterations,
            wall_clock_s=wall_clock_s,
            parent_ctx=parent_ctx,
            session_id=session_id,
        )
        self.plan_path = plan_path
        self.feedback = feedback

    def run(self) -> dict[str, Any]:
        """Return a fake flag after validating that the plan artifact exists.

        Returns:
            On success, a compact solver outcome of shape
            `{"outcome": "flag", "flag": "MOCK_FLAG", ...}`. If the plan
            cannot be read, returns `{"outcome": "error", "error_summary": ...}`.
        """
        run_t0 = time()
        self.budget.mark_started()
        self._emitter.emit(AgentStarted(type="agent.started", ctx=self.ctx))

        try:
            plan = self._read_plan()
            result = self._mock_flag_result(plan)
        except Exception as exc:
            result = self._on_error(exc)

        return self._finalize(result, run_t0)

    def _read_plan(self) -> dict[str, Any]:
        """Read and decode the planner artifact.

        Raises:
            FileNotFoundError: If `plan_path` does not exist.
            json.JSONDecodeError: If the plan file is not valid JSON.
        """
        return json.loads(self.plan_path.read_text(encoding="utf-8"))

    def _mock_flag_result(self, plan: dict[str, Any]) -> dict[str, Any]:
        """Build the contract-compatible fake solver success result."""
        self.log.info(
            "mock solver returning fake flag for plan=%s task_family=%s",
            self.plan_path,
            plan.get("task_family"),
        )
        return {
            "outcome": "flag",
            "flag": MOCK_FLAG,
            "mock": True,
            "plan_path": str(self.plan_path),
            "task_family": plan.get("task_family"),
            "verify_task_name": plan.get("verify_task_name"),
            "feedback_seen": self.feedback is not None,
            "tool_calls_made": 0,
            "submit_responses": [],
        }

    def _on_error(self, exc: Exception) -> dict[str, Any]:
        """Convert mock-solver failures into structured solver errors."""
        message = f"{type(exc).__name__}: {exc}"
        self.log.exception("mock solver failed: %s", exc)
        self._emitter.emit(AgentError(
            type="agent.error",
            ctx=self.ctx,
            error_type="mock_solver_failure",
            message=message,
            step=1,
        ))
        return {
            "outcome": "error",
            "error_summary": message,
            "mock": True,
            "plan_path": str(self.plan_path),
        }

    def _finalize(self, result: dict[str, Any], run_t0: float) -> dict[str, Any]:
        """Emit completion event and return the result unchanged."""
        self._emitter.emit(AgentCompleted(
            type="agent.completed",
            ctx=self.ctx,
            duration_ms=(time() - run_t0) * 1000,
            result=result.get("outcome"),
        ))
        return result
