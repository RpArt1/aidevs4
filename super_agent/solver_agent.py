"""ReAct Solver agent.

Reads plan.json produced by PlannerAgent, then drives a ReAct loop
(LLM + execute_python + submit_answer) until the flag is captured or the
budget is exhausted.
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
    GenerationCompleted,
    IterationLimitReached,
)

from .agent_base import SuperAgentBase
from .agent_helper import BudgetExceeded
from .solver_tools import SOLVER_TOOLS, make_solver_dispatcher

DEFAULT_MAX_ITERATIONS = 15
DEFAULT_WALL_CLOCK_S = 300

PROMPTS_DIR = Path(__file__).parent / "prompts"

FALLBACK_SYSTEM_PROMPT = """\
You are the Solver. You receive a structured plan and execute it by writing Python code \
until you have the correct answer, then submit it.

Tools:
- execute_python(code, timeout?): run Python in a subprocess; returns {stdout, stderr, returncode}.
- submit_answer(task, answer): submit to aidevs; returns fatal sentinel with flag on success,
  or {outcome: "incorrect", hint: "..."} on failure — read the hint and retry.

Rules:
1. Before every tool call, write one sentence explaining why.
2. Follow plan steps in order; debug failures from stderr.
3. Only call submit_answer when confident.
4. Never fabricate a flag (FLG:...).
"""


class SolverAgent(SuperAgentBase):
    """ReAct solver: write Python code → execute → submit until flag is captured."""

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
        self._final_result: dict[str, Any] | None = None
        self._script_counter: int = 0
        self._last_step: int = 0

    def run(self) -> dict[str, Any]:
        run_t0 = time()
        self.budget.mark_started()
        self._emitter.emit(AgentStarted(type="agent.started", ctx=self.ctx))

        try:
            plan = self._read_plan()
            messages = self._initial_messages(plan)
            execute_tool = make_solver_dispatcher(self)
            result = self._loop(messages, SOLVER_TOOLS, execute_tool)
        except BudgetExceeded as exc:
            result = self._on_budget_exceeded(exc)
        except Exception as exc:
            result = self._on_error(exc)

        return self._finalize(result, run_t0)

    # ── ReAct loop ──────────────────────────────────────────────────────────

    def _loop(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict],
        execute_tool: Any,
    ) -> dict[str, Any]:
        for step in range(1, self.budget.max_iterations + 1):
            self._last_step = step
            self.budget.raise_if_exceeded(step)
            self.log.info(
                "solver step=%d/%d scripts_written=%d",
                step, self.budget.max_iterations, self._script_counter,
            )
            terminal = self._step(messages, tools, execute_tool, step)
            if terminal is not None:
                return terminal
        return self._on_max_iter()

    def _step(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict],
        execute_tool: Any,
        step: int,
    ) -> dict[str, Any] | None:
        message = self._chat(messages, tools, step)

        if not message.tool_calls:
            return self._on_no_tool_calls(step)

        fatal = self._process_tool_calls(
            messages=messages,
            tool_calls=message.tool_calls,
            step=step,
            execute_tool=execute_tool,
        )
        if fatal:
            return self._on_fatal_tool(step)

        return None

    def _chat(self, messages: list[dict[str, Any]], tools: list[dict], step: int):
        t0 = time()
        message = self.llm.chat_with_tools(messages=messages, tools=tools)
        self._emitter.emit(GenerationCompleted(
            type="generation.completed",
            ctx=self.ctx,
            output=message.content,
            model=self.llm.model,
            input=messages,
            input_tokens=self.llm.last_usage.input_tokens,
            output_tokens=self.llm.last_usage.output_tokens,
            duration_ms=(time() - t0) * 1000,
            step=step,
        ))
        return message

    # ── Terminal state handlers ─────────────────────────────────────────────

    def _on_fatal_tool(self, step: int) -> dict[str, Any]:
        if self._final_result is None:
            self.log.error("fatal tool result but _final_result is None; dispatcher contract violated")
            return {"outcome": "error", "error_summary": "fatal tool result with no recorded final result", "steps": step}
        return {**self._final_result, "steps": step}

    def _on_no_tool_calls(self, step: int) -> dict[str, Any]:
        self.log.warning("solver produced no tool_calls at step=%d; giving up", step)
        return {"outcome": "error", "error_summary": "solver gave up without submitting", "steps": step}

    def _on_max_iter(self) -> dict[str, Any]:
        step = self._last_step
        self.log.warning("solver reached max_iterations=%d", self.budget.max_iterations)
        self._emitter.emit(IterationLimitReached(
            type="agent.iteration_limit",
            ctx=self.ctx,
            max_iterations=self.budget.max_iterations,
            step=step,
        ))
        return {"outcome": "error", "error_summary": f"max_iterations={self.budget.max_iterations} reached", "steps": step}

    def _on_budget_exceeded(self, exc: BudgetExceeded) -> dict[str, Any]:
        step = self._last_step
        self.log.warning("solver budget exceeded: %s", exc.reason)
        self._emitter.emit(IterationLimitReached(
            type="agent.iteration_limit",
            ctx=self.ctx,
            max_iterations=self.budget.max_iterations,
            step=step,
        ))
        return {"outcome": "error", "error_summary": f"budget exceeded: {exc.reason}", "steps": step}

    def _on_error(self, exc: Exception) -> dict[str, Any]:
        step = self._last_step
        message = f"{type(exc).__name__}: {exc}"
        self.log.exception("solver crashed: %s", exc)
        self._emitter.emit(AgentError(
            type="agent.error",
            ctx=self.ctx,
            error_type="solver_crash",
            message=message,
            step=step,
        ))
        return {"outcome": "error", "error_summary": message, "steps": step}

    # ── Setup helpers ───────────────────────────────────────────────────────

    def _read_plan(self) -> dict[str, Any]:
        return json.loads(self.plan_path.read_text(encoding="utf-8"))

    def _initial_messages(self, plan: dict[str, Any]) -> list[dict[str, Any]]:
        return [
            {"role": "system", "content": self._load_system_prompt()},
            {"role": "user", "content": self._build_user_message(plan)},
        ]

    def _load_system_prompt(self) -> str:
        prompt_file = PROMPTS_DIR / "solver.md"
        if prompt_file.is_file():
            return prompt_file.read_text(encoding="utf-8").replace("{workspace}", str(self.workspace))
        self.log.debug("solver prompt file missing (%s); using inline default", prompt_file)
        return FALLBACK_SYSTEM_PROMPT.replace("{workspace}", str(self.workspace))

    def _build_user_message(self, plan: dict[str, Any]) -> str:
        parts = [
            "# Plan",
            "",
            f"**Goal**: {plan.get('goal', '')}",
            f"**Task family**: {plan.get('task_family', '')}",
            f"**Verify task name**: {plan.get('verify_task_name', '')}",
            "",
        ]

        if plan.get("required_env"):
            parts.append("**Required env vars**: " + ", ".join(plan["required_env"]))
            parts.append("")

        if plan.get("input_data"):
            parts.append("**Input data**:")
            for item in plan["input_data"]:
                parts.append(f"  - {item.get('path')}: {item.get('description')}")
            parts.append("")

        parts.append("**Steps**:")
        for i, step in enumerate(plan.get("steps", []), 1):
            parts.append(f"  {i}. {step}")
        parts.append("")

        if plan.get("hints"):
            parts.append("**Hints**:")
            for hint in plan["hints"]:
                parts.append(f"  - {hint}")
            parts.append("")

        if plan.get("success_check"):
            parts.append(f"**Success check**: {plan['success_check']}")
            parts.append("")

        if self.feedback:
            parts += [
                "# Feedback from previous attempt",
                self.feedback.strip(),
                "",
            ]

        parts += [
            f"# Workspace",
            f"Your workspace directory: {self.workspace}",
            "",
            "# Instruction",
            "Execute the plan step by step. Write Python code with execute_python. "
            "When confident about the answer, call submit_answer. "
            "Always explain WHY you are calling each tool before calling it.",
        ]
        return "\n".join(parts)

    def _finalize(self, result: dict[str, Any], run_t0: float) -> dict[str, Any]:
        self._emitter.emit(AgentCompleted(
            type="agent.completed",
            ctx=self.ctx,
            duration_ms=(time() - run_t0) * 1000,
            result=result.get("outcome"),
        ))
        return result
