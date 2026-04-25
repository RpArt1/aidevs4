"""Tool schemas and dispatcher for `OrchestratorAgent`.

The orchestrator itself is an LLM ReAct loop. This module is the small Python
bridge that turns its tool calls into concrete sub-agent runs:

* `spawn_planner` runs `PlannerAgent` and records a compact plan summary.
* `spawn_solver` runs `SolverAgent` and records a compact solver outcome.
* `finish` stores the terminal result and returns the fatal sentinel consumed
  by `SuperAgentBase._process_tool_calls`.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

from .planner_agent import PlannerAgent

if TYPE_CHECKING:
    from .orchestrator import OrchestratorAgent


ToolDispatcher = Callable[[str, dict], str]


TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "spawn_planner",
            "description": (
                "Run the Planner sub-agent to create or revise a plan.json "
                "artifact from the plain-text task."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "task_text": {
                        "type": "string",
                        "description": (
                            "Plain-text task description to plan from. Usually "
                            "the original task text from the user message."
                        ),
                    },
                    "critique": {
                        "type": "string",
                        "description": (
                            "Optional critique of a previous plan. Use only "
                            "when re-planning after a solver failure."
                        ),
                    },
                },
                "required": ["task_text"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "spawn_solver",
            "description": (
                "Run the Solver sub-agent against an existing plan artifact. "
                "The solver returns a compact terminal outcome."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "plan_path": {
                        "type": "string",
                        "description": "Path to the plan JSON produced by spawn_planner.",
                    },
                    "feedback": {
                        "type": "string",
                        "description": (
                            "Optional execution feedback for a retry, based on "
                            "a previous solver failure."
                        ),
                    },
                },
                "required": ["plan_path"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "finish",
            "description": "Finish the orchestrator run with success or give_up.",
            "parameters": {
                "type": "object",
                "properties": {
                    "status": {
                        "type": "string",
                        "enum": ["success", "give_up"],
                        "description": "Terminal status for the whole run.",
                    },
                    "flag": {
                        "type": "string",
                        "description": "Flag returned by the solver on success.",
                    },
                    "reason": {
                        "type": "string",
                        "description": "Reason for give_up, or extra context.",
                    },
                },
                "required": ["status"],
                "additionalProperties": False,
            },
        },
    },
]


def make_dispatcher(orchestrator: "OrchestratorAgent") -> ToolDispatcher:
    """Build a tool dispatcher bound to one orchestrator instance.

    Args:
        orchestrator: Running orchestrator whose budgets, history, workspace,
            LLM, and emitter are shared with spawned sub-agents.

    Returns:
        Callable matching `SuperAgentBase._process_tool_calls`.
    """

    def dispatch(name: str, args: dict) -> str:
        if name == "spawn_planner":
            return _spawn_planner(orchestrator, args)
        if name == "spawn_solver":
            return _spawn_solver(orchestrator, args)
        if name == "finish":
            return _finish(orchestrator, args)
        return _json({"error": f"unknown orchestrator tool: {name}"})

    return dispatch


def _spawn_planner(orchestrator: "OrchestratorAgent", args: dict) -> str:
    """Run `PlannerAgent`, record its compact result, and return JSON."""
    if orchestrator.planner_spawns_remaining <= 0:
        return _json({"outcome": "error", "error_summary": "planner spawn budget exhausted"})

    orchestrator.planner_spawns_remaining -= 1
    spawn_index = len(orchestrator.plans) + 1
    output_path = _plan_output_path(orchestrator.workspace, spawn_index)

    planner = PlannerAgent(
        task_text=str(args.get("task_text") or orchestrator.task_text),
        output_path=output_path,
        run_id=orchestrator.run_id,
        workspace=orchestrator.workspace,
        emitter=orchestrator._emitter,
        llm=orchestrator.llm,
        critique=args.get("critique"),
        public_webhook_url=orchestrator.public_webhook_url,
        verify_task_name_override=orchestrator.verify_task_name_override,
        agent_id=f"planner_{spawn_index}",
        parent_ctx=orchestrator.ctx,
        session_id=orchestrator.ctx.session_id,
    )
    result = planner.run()
    result["spawn_index"] = spawn_index
    orchestrator.plans.append(result)
    return _json(result)


def _spawn_solver(orchestrator: "OrchestratorAgent", args: dict) -> str:
    """Run `SolverAgent`, record its compact result, and return JSON.

    `SolverAgent` is imported lazily so the orchestrator and planner can be
    loaded while the solver implementation is still being developed.
    """
    if orchestrator.solver_spawns_remaining <= 0:
        return _json({"outcome": "error", "error_summary": "solver spawn budget exhausted"})

    from .solver_agent import SolverAgent

    orchestrator.solver_spawns_remaining -= 1
    spawn_index = len(orchestrator.solver_runs) + 1

    solver = SolverAgent(
        plan_path=Path(str(args["plan_path"])),
        feedback=args.get("feedback"),
        run_id=orchestrator.run_id,
        workspace=orchestrator.workspace,
        emitter=orchestrator._emitter,
        llm=orchestrator.llm,
        agent_id=f"solver_{spawn_index}",
        parent_ctx=orchestrator.ctx,
        session_id=orchestrator.ctx.session_id,
    )
    result = solver.run()
    result["spawn_index"] = spawn_index
    orchestrator.solver_runs.append(result)
    return _json(result)


def _finish(orchestrator: "OrchestratorAgent", args: dict) -> str:
    """Store the terminal result and return the fatal sentinel."""
    status = str(args.get("status") or "give_up")
    result: dict[str, Any] = {"status": status}

    if args.get("flag"):
        result["flag"] = args["flag"]
    if args.get("reason"):
        result["reason"] = args["reason"]

    if status == "success" and "flag" not in result:
        result = {
            "status": "give_up",
            "reason": "finish(status='success') called without a flag",
        }

    orchestrator._final_result = result
    return _json({"fatal": True, **result})


def _plan_output_path(workspace: Path, spawn_index: int) -> Path:
    """Return the planner output path for the given spawn number."""
    if spawn_index == 1:
        return workspace / "plan.json"
    return workspace / f"plan.v{spawn_index}.json"


def _json(payload: dict[str, Any]) -> str:
    """Serialize a compact tool result payload."""
    return json.dumps(payload, ensure_ascii=False)
