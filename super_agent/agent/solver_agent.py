"""ReAct Solver agent.

Reads plan.json produced by PlannerAgent, then drives a ReAct loop
(LLM + execute_python + submit_answer) until the flag is captured or the
budget is exhausted.
"""

from __future__ import annotations

import json
import re
from collections import deque
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
from .tools.solver_tools import SOLVER_TOOLS, make_solver_dispatcher

DEFAULT_MAX_ITERATIONS = 15
DEFAULT_WALL_CLOCK_S = 300


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
        # Rolling window of error fingerprints from recent execute_python calls.
        # Used to detect "stuck in a loop" patterns (same error twice in a row).
        self._stderr_window: deque[str] = deque(maxlen=3)

    def _system_prompt_basename(self) -> str:
        return "solver.md"

    def _transform_system_prompt(self, raw: str) -> str:
        return raw.replace("{workspace}", str(self.workspace))

    def run(self) -> dict[str, Any]:
        run_t0 = time()
        self.budget.mark_started()
        self._emitter.emit(AgentStarted(type="agent.started", ctx=self.ctx))

        try:
            plan = self._read_plan()
            messages = self.(plan)
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

        self._maybe_inject_loop_reminder(messages)
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

    # ── Loop detection ─────────────────────────────────────────────────────

    def record_execution_stderr(self, stderr: str) -> None:
        """Called by the execute_python tool after each run to track errors.

        Args:
            stderr: The stderr string captured from the subprocess.
        """
        fingerprint = self._extract_error_fingerprint(stderr)
        self._stderr_window.append(fingerprint or "")

    @staticmethod
    def _extract_error_fingerprint(stderr: str) -> str | None:
        """Return a short, stable label for the dominant error in stderr.

        Walks the stderr lines in reverse to find the last traceback
        summary line (``ExceptionClass: message``). Falls back to the
        first non-empty line so that non-traceback errors are still
        tracked.

        Args:
            stderr: Raw stderr text from the subprocess.

        Returns:
            A fingerprint string, or None when stderr is empty/whitespace.
        """
        if not stderr.strip():
            return None

        # Python tracebacks end with "SomeClass: message" on the final line.
        # Walk backwards to find the first such line (= the exception summary).
        for line in reversed(stderr.splitlines()):
            stripped = line.strip()
            if not stripped:
                continue
            # Match "word.word.ClassName: ..." — must not start with whitespace
            # (indented lines are code context, not the exception class).
            m = re.match(r'^([\w][\w.]*)\s*:', stripped)
            if m:
                return m.group(1)

        # No traceback-style line found; use first non-empty line as fallback.
        for line in stderr.splitlines():
            stripped = line.strip()
            if stripped:
                return stripped[:120]

        return None

    def _maybe_inject_loop_reminder(self, messages: list[dict[str, Any]]) -> None:
        """Append a corrective user message when the same error repeats.

        Checks the last two entries in the stderr window. If they share
        the same non-empty fingerprint the solver is stuck, and a reminder
        is injected so the LLM sees it before its next generation step.

        Args:
            messages: Mutable chat history. A ``role="user"`` reminder is
                appended when a loop is detected.
        """
        window = list(self._stderr_window)
        if len(window) < 2:
            return

        last = window[-1]
        if not last or last != window[-2]:
            return

        reminder = (
            f"IMPORTANT — you have failed with the same error ({last!r}) at least "
            "twice in a row. Re-read the execution environment section of your system "
            "prompt carefully before calling execute_python again. Common causes: "
            "using openai<1.0 API syntax (openai.ChatCompletion.create does not exist "
            "— use the v1 client pattern shown in the prompt), wrong import names, "
            "missing env variables, or incorrect file paths. Change your approach."
        )
        messages.append({"role": "user", "content": reminder})
        self.log.warning(
            "loop-detection: injecting corrective reminder fingerprint=%r", last
        )

    # ── Setup helpers ───────────────────────────────────────────────────────

    def _read_plan(self) -> dict[str, Any]:
        return json.loads(self.plan_path.read_text(encoding="utf-8"))

    def _initial_messages(self, plan: dict[str, Any]) -> list[dict[str, Any]]:
        return [
            {"role": "system", "content": self._load_system_prompt()},
            {"role": "user", "content": self._build_user_message(plan)},
        ]

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

        resources = plan.get("extracted_resources")
        if isinstance(resources, dict) and any(resources.get(k) for k in (
            "urls", "exact_strings", "expected_formats",
        )):
            parts.append("**Extracted resources**:")
            for label, key in (
                ("URLs", "urls"),
                ("Exact strings", "exact_strings"),
                ("Expected formats", "expected_formats"),
            ):
                vals = resources.get(key) if isinstance(resources.get(key), list) else []
                for entry in vals:
                    parts.append(f"  - [{label}] {entry}")
            parts.append("")

        if plan.get("input_data"):
            parts.append("**Input data**:")
            for item in plan["input_data"]:
                if not isinstance(item, dict):
                    continue
                loc = item.get("location") or item.get("path") or ""
                desc = item.get("description") or ""
                st = item.get("source_type") or ""
                parts.append(f"  - ({st}) {loc}: {desc}")
            parts.append("")

        if plan.get("expected_output"):
            parts.append("**Expected output (verify payload / answer shape)**:")
            parts.append(f"  {plan['expected_output']}")
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
