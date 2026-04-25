"""
Orchestrator agent — the LLM ReAct supervisor.

`OrchestratorAgent(SuperAgentBase)` bootstraps the run and then hands all
control-flow decisions (plan / solve / re-plan / retry / give up) to the LLM
via three tools: ``spawn_planner``, ``spawn_solver``, ``finish``. The tool
schemas and dispatcher live in :mod:`super_agent.orchestrator_tools`; this
module owns only the ReAct loop, run-level state, and lifecycle events.

The orchestrator is deliberately thin: it does not retry or re-plan in Python.
It just exposes state (`task_text`, spawn budgets, plan/solver run history,
`_final_result` slot) that the dispatcher reads and writes while the LLM is
reasoning.

Budgets (per ``super agent design v2``):
    * orchestrator iterations: 8
    * wall-clock: 15 min
    * planner spawns per run: 2  (initial + one re-plan)
    * solver spawns per run: 3
"""

from __future__ import annotations

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


DEFAULT_MAX_ITERATIONS = 8
DEFAULT_WALL_CLOCK_S = 15 * 60
DEFAULT_MAX_PLANNER_SPAWNS = 2
DEFAULT_MAX_SOLVER_SPAWNS = 3

PROMPTS_DIR = Path(__file__).parent / "prompts"

FALLBACK_SYSTEM_PROMPT = """\
You are the Orchestrator, the top-level supervisor of a super-agent that \
solves aidevs tasks. You do NOT write code or call task APIs yourself. You \
coordinate two sub-agents via tool calls:

- spawn_planner(task_text, critique?) -> writes plan.json; returns compact plan \
summary (goal, task_family, verify_task_name, required_env).
- spawn_solver(plan_path, feedback?) -> runs the Solver ReAct loop against the \
plan; returns outcome: "flag", "error", or "max_iter".
- finish(status, flag?, reason?) -> terminal; call this exactly once when done.

## Decision rules
1. If no plan exists yet, call spawn_planner with the task_text as-is.
2. After a plan, call spawn_solver(plan_path).
3. On solver outcome="flag", call finish(status="success", flag=<flag>) and stop.
4. On solver outcome="error" or "max_iter", read the returned error_summary and \
decide:
   - Bad plan (wrong verify_task_name, missing step, wrong task_family) -> \
spawn_planner again with a crisp critique citing the specific defect.
   - Bad execution (code bug, wrong approach within a valid plan) -> \
spawn_solver again passing a short feedback string that points at the fix.
   - Out of budget or repeated identical failures -> finish(status="give_up", \
reason=...).
5. Respect your spawn budgets. The dispatcher will refuse spawns past the caps.

Be concise. Do not narrate; act through tool calls.\
"""


class OrchestratorAgent(SuperAgentBase):
    """LLM ReAct supervisor over ``spawn_planner``/``spawn_solver``/``finish``."""

    def __init__(
        self,
        *,
        task_text: str,
        run_id: str,
        workspace: Path,
        emitter: AgentEventEmitter,
        llm: LLMService,
        public_webhook_url: str | None = None,
        verify_task_name_override: str | None = None,
        agent_id: str = "orchestrator",
        max_iterations: int = DEFAULT_MAX_ITERATIONS,
        wall_clock_s: int = DEFAULT_WALL_CLOCK_S,
        max_planner_spawns: int = DEFAULT_MAX_PLANNER_SPAWNS,
        max_solver_spawns: int = DEFAULT_MAX_SOLVER_SPAWNS,
        session_id: str | None = None,
        parent_ctx: EventContext | None = None,
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

        self.task_text = task_text
        self.public_webhook_url = public_webhook_url
        self.verify_task_name_override = verify_task_name_override

        # Spawn budgets — decremented by the dispatcher in orchestrator_tools.
        self.planner_spawns_remaining = max_planner_spawns
        self.solver_spawns_remaining = max_solver_spawns

        # Run history — the dispatcher appends to these after each spawn so
        # the orchestrator's LLM context can reference plan/run numbers by
        # index without the full artifact being re-embedded in messages.
        self.plans: list[dict[str, Any]] = []
        self.solver_runs: list[dict[str, Any]] = []

        # Terminal result slot. `finish` tool populates this; the loop sees
        # the fatal sentinel on the tool result and exits cleanly.
        self._final_result: dict[str, Any] | None = None

        # Tracks the last step the loop entered, so exception handlers can
        # tag IterationLimitReached / AgentError with the right step number
        # without threading `step` through every helper.
        self._last_step: int = 0

    # ── Public entry point ──────────────────────────────────────────────────

    def run(self) -> dict[str, Any]:
        # Lazy import: orchestrator_tools pulls in planner_agent / solver_agent,
        # which are separate todos in the build plan. Importing here keeps
        # this module importable (and unit-testable) before those land.
        from .orchestrator_tools import TOOLS, make_dispatcher

        run_t0 = time()
        self.budget.mark_started()
        self._emitter.emit(AgentStarted(type="agent.started", ctx=self.ctx))

        messages = self._initial_messages()
        execute_tool = make_dispatcher(self)

        try:
            result = self._loop(messages, TOOLS, execute_tool)
        except BudgetExceeded as exc:
            result = self._on_budget_exceeded(exc)
        except Exception as exc:
            result = self._on_crash(exc)

        return self._finalize(result, run_t0)

    # ── ReAct loop ──────────────────────────────────────────────────────────

    def _loop(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict],
        execute_tool: Any,
    ) -> dict[str, Any]:
        """Run the ReAct loop. Returns the terminal result dict."""
        for step in range(1, self.budget.max_iterations + 1):
            self._last_step = step

            self.budget.raise_if_exceeded(step)

            self._log_step_header(step)
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
        """Run one ReAct step. Returns a terminal result dict, or None to continue."""
        message = self._chat(messages, tools, step)

        if not message.tool_calls:
            return self._on_no_tool_calls(message, step)

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
        """Run one LLM call and emit the matching GenerationCompleted event."""
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

    # ── Terminal-state handlers (return result dict) ────────────────────────

    def _on_fatal_tool(self, step: int) -> dict[str, Any]:
        """`finish` fired: the dispatcher should have populated _final_result."""
        if self._final_result is None:
            self.log.error(
                "fatal tool result but _final_result is None; dispatcher contract violated",
            )
            return self._give_up("fatal tool result with no recorded final result", step)
        return {**self._final_result, "steps": step}

    def _on_no_tool_calls(self, message, step: int) -> dict[str, Any]:
        self.log.warning(
            "orchestrator produced no tool_calls at step=%d; treating as give_up", step,
        )
        return {
            **self._give_up(
                "orchestrator returned a final message without calling finish()", step,
            ),
            "last_message": message.content or "",
        }

    def _on_max_iter(self) -> dict[str, Any]:
        step = self._last_step
        self.log.warning("orchestrator reached max_iterations=%d without finishing", self.budget.max_iterations)
        self._emit_iter_limit(step)
        return self._give_up(f"max_iterations={self.budget.max_iterations} reached", step)

    def _on_budget_exceeded(self, exc: BudgetExceeded) -> dict[str, Any]:
        step = self._last_step
        self.log.warning("orchestrator budget exceeded: %s", exc.reason)
        self._emit_iter_limit(step)
        return self._give_up(f"budget exceeded: {exc.reason}", step)

    def _on_crash(self, exc: Exception) -> dict[str, Any]:
        step = self._last_step
        self.log.exception("orchestrator crashed: %s", exc)
        self._emitter.emit(AgentError(
            type="agent.error",
            ctx=self.ctx,
            error_type="orchestrator_crash",
            message=f"{type(exc).__name__}: {exc}",
            step=step,
        ))
        return self._give_up(f"orchestrator crashed: {type(exc).__name__}: {exc}", step)

    # ── Small helpers ───────────────────────────────────────────────────────

    def _initial_messages(self) -> list[dict[str, Any]]:
        return [
            {"role": "system", "content": self._load_system_prompt()},
            {"role": "user", "content": self._build_initial_user_message()},
        ]

    def _load_system_prompt(self) -> str:
        prompt_file = PROMPTS_DIR / "orchestrator.md"
        if prompt_file.is_file():
            return prompt_file.read_text(encoding="utf-8")
        self.log.debug("orchestrator prompt file missing (%s); using inline default", prompt_file)
        return FALLBACK_SYSTEM_PROMPT

    def _build_initial_user_message(self) -> str:
        parts = ["# Task (plain text, as given to the human)", "", self.task_text.strip()]
        env_hints: list[str] = []
        if self.public_webhook_url:
            env_hints.append(f"PUBLIC_WEBHOOK_URL={self.public_webhook_url}")
        if self.verify_task_name_override:
            env_hints.append(f"verify_task_name (override)={self.verify_task_name_override}")
        if env_hints:
            parts += ["", "# Run-level context", *(f"- {h}" for h in env_hints)]
        parts += [
            "",
            "# Instruction",
            "Plan and solve. Call spawn_planner first. End with exactly one finish() call.",
        ]
        return "\n".join(parts)

    def _log_step_header(self, step: int) -> None:
        self.log.info(
            "orchestrator step=%d/%d planner_left=%d solver_left=%d",
            step,
            self.budget.max_iterations,
            self.planner_spawns_remaining,
            self.solver_spawns_remaining,
        )

    def _emit_iter_limit(self, step: int) -> None:
        self._emitter.emit(IterationLimitReached(
            type="agent.iteration_limit",
            ctx=self.ctx,
            max_iterations=self.budget.max_iterations,
            step=step,
        ))

    @staticmethod
    def _give_up(reason: str, step: int) -> dict[str, Any]:
        return {"status": "give_up", "reason": reason, "steps": step}

    def _finalize(self, result: dict[str, Any], run_t0: float) -> dict[str, Any]:
        result.setdefault("plans_spawned", len(self.plans))
        result.setdefault("solver_runs", len(self.solver_runs))
        self._emitter.emit(AgentCompleted(
            type="agent.completed",
            ctx=self.ctx,
            duration_ms=(time() - run_t0) * 1000,
            result=result.get("status"),
        ))
        return result
