"""
Planner agent — single structured-LLM call that turns plain-text task into
``plan.json``.

`PlannerAgent(SuperAgentBase)` is intentionally the simplest of the three
super-agent roles: no tools, no ReAct loop, one shot at structured output via
``LLMService.chat_structured``. The orchestrator's ``spawn_planner`` tool
constructs it, runs it, and feeds the compact summary back into the
orchestrator's ReAct loop.

Why a separate agent at all?
    * The plan is a persisted, replayable artifact (``plan.json``).
    * Failure attribution is easier — orchestrator can tell "bad plan" apart
      from "bad execution" because each lives behind its own agent boundary.
    * Independently swappable for a stronger/cheaper model later without
      touching the Solver loop.

The output schema mirrors ``super_agent/prompts/planner.md`` and the v2 design
note (``goal`` / ``task_family`` / ``verify_task_name`` /
``extracted_resources`` / ``required_env`` / ``input_data`` / ``expected_output``
/ ``steps`` / ``hints`` / ``success_check``).
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
)

from .agent_base import SuperAgentBase


# Single-shot agent — these caps mostly exist to satisfy the base class
# contract and to backstop a stuck LLM call.
DEFAULT_MAX_ITERATIONS = 1
DEFAULT_WALL_CLOCK_S = 120

VALID_TASK_FAMILIES = ("data_structured", "tool_react", "long_running_webhook")

PLAN_SCHEMA: dict[str, Any] = {
    "name": "task_plan",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "goal": {
                "type": "string",
                "description": "One-sentence restatement of what the task asks for.",
            },
            "task_family": {
                "type": "string",
                "enum": list(VALID_TASK_FAMILIES),
                "description": (
                    "Coarse-grained shape: 'data_structured' for static-data + "
                    "structured-LLM tasks, 'tool_react' for tool-using ReAct "
                    "tasks, 'long_running_webhook' for tasks where the aidevs "
                    "server calls back into a server we expose."
                ),
            },
            "verify_task_name": {
                "type": "string",
                "description": (
                    "Short slug used by AssignmentService.send(task, answer); "
                    "should be extracted from the task text (e.g. 'nazwa zadania to: \"people\"')."
                ),
            },
            "extracted_resources": {
                "type": "object",
                "description": (
                    "Concrete URLs, exact strings, and payload shapes taken from the task text."
                ),
                "properties": {
                    "urls": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "URLs, API endpoints, or webhooks from the task.",
                    },
                    "exact_strings": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": (
                            "Passwords, labels, regex fragments, or other verbatim strings."
                        ),
                    },
                    "expected_formats": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": (
                            "How the final answer must be shaped (e.g. mock JSON for verify API)."
                        ),
                    },
                },
                "required": ["urls", "exact_strings", "expected_formats"],
                "additionalProperties": False,
            },
            "required_env": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "Names of environment variables the Solver's generated "
                    "code will need (e.g. AIDEVS_API_KEY, PUBLIC_WEBHOOK_URL)."
                ),
            },
            "input_data": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "source_type": {
                            "type": "string",
                            "enum": ["url", "local_file", "api"],
                            "description": "Where this input comes from.",
                        },
                        "location": {
                            "type": "string",
                            "description": "Path, URL, or API identifier for the input.",
                        },
                        "description": {
                            "type": "string",
                            "description": "What the Solver will find and in what format.",
                        },
                    },
                    "required": ["source_type", "location", "description"],
                    "additionalProperties": False,
                },
                "description": (
                    "Inputs the Solver should expect (URLs, files in the "
                    "workspace, etc.). Empty array if none."
                ),
            },
            "expected_output": {
                "type": "string",
                "description": (
                    "Expected JSON or structure for the verify POST body / answer field "
                    "(from the task example), so the Solver does not guess."
                ),
            },
            "steps": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Natural-language step-by-step approach for the Solver.",
            },
            "hints": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Gotchas / invariants the Solver must respect.",
            },
            "success_check": {
                "type": "string",
                "description": "What the final submission response should look like.",
            },
        },
        "required": [
            "goal",
            "task_family",
            "verify_task_name",
            "extracted_resources",
            "required_env",
            "input_data",
            "expected_output",
            "steps",
            "hints",
            "success_check",
        ],
        "additionalProperties": False,
    },
}


class PlannerAgent(SuperAgentBase):
    """
    Single-shot planner: ``chat_structured`` once, write ``plan.json``, return
    a compact summary dict for the spawning orchestrator.

    Args (kw-only):
        task_text: Plain-text task description (as given to the human).
        output_path: Absolute path where the produced plan JSON is written.
            The orchestrator's spawn dispatcher chooses this (typically
            ``workspaces/<run_id>/plan.json`` or ``plan.vN.json`` on re-plan).
        critique: Optional critique from the orchestrator about a previous
            plan attempt. Prepended to the user message when present.
        public_webhook_url: Optional run-level hint surfaced to the LLM when
            the task may need an externally reachable webhook.
        verify_task_name_override: Optional human-supplied override for the
            ``verify_task_name`` slug; surfaced as a strong hint.

    The remaining kwargs (`run_id`, `workspace`, `emitter`, `llm`,
    `max_iterations`, `wall_clock_s`, `parent_ctx`, `session_id`, `agent_id`)
    follow the standard ``SuperAgentBase`` contract.
    """

    def __init__(
        self,
        *,
        task_text: str,
        output_path: Path,
        run_id: str,
        workspace: Path,
        emitter: AgentEventEmitter,
        llm: LLMService,
        critique: str | None = None,
        public_webhook_url: str | None = None,
        verify_task_name_override: str | None = None,
        agent_id: str = "planner",
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

        self.task_text = task_text
        self.output_path = output_path
        self.critique = critique
        self.public_webhook_url = public_webhook_url
        self.verify_task_name_override = verify_task_name_override

    def _system_prompt_basename(self) -> str:
        return "planner.md"

    # ── Public entry point ──────────────────────────────────────────────────

    def run(self) -> dict[str, Any]:
        """Produce the plan, persist it, and return a compact summary.

        Returns:
            On success::

                {
                    "outcome": "plan",
                    "plan_path": "<absolute path to plan.json>",
                    "task_family": "...",
                    "goal": "...",
                    "verify_task_name": "...",
                    "required_env": [...],
                    "summary": "<short human-readable digest>",
                }

            On failure (LLM error, schema violation, write error)::

                {
                    "outcome": "error",
                    "error_summary": "...",
                }
        """
        run_t0 = time()
        self.budget.mark_started()
        self._emitter.emit(AgentStarted(type="agent.started", ctx=self.ctx))

        try:
            plan = self._chat_for_plan()
            self._validate_plan(plan)
            plan_path = self._write_plan(plan)
            result = self._build_summary(plan, plan_path)
        except Exception as exc:
            result = self._on_error(exc)

        return self._finalize(result, run_t0)

    # ── LLM call ────────────────────────────────────────────────────────────

    def _chat_for_plan(self) -> dict[str, Any]:
        """Single ``chat_structured`` call. Emits a ``GenerationCompleted`` event."""
        system_prompt = self._load_system_prompt()
        user_message = self._build_user_message()
        messages = [{"role": "user", "content": user_message}]

        # Emit step=1 so the event stream stays consistent with multi-step
        # agents (Orchestrator, Solver) even though we only ever take one.
        step = 1
        t0 = time()
        plan = self.llm.chat_structured(
            messages=messages,
            schema=PLAN_SCHEMA,
            system_prompt=system_prompt,
        )
        duration_ms = (time() - t0) * 1000

        self._emitter.emit(GenerationCompleted(
            type="generation.completed",
            ctx=self.ctx,
            output=json.dumps(plan, ensure_ascii=False),
            model=self.llm.model,
            input=[{"role": "system", "content": system_prompt}, *messages],
            input_tokens=self.llm.last_usage.input_tokens,
            output_tokens=self.llm.last_usage.output_tokens,
            duration_ms=duration_ms,
            step=step,
        ))
        return plan

    # ── Prompt assembly ─────────────────────────────────────────────────────

    def _build_user_message(self) -> str:
        """Compose the user turn: task text + run-level hints + optional critique."""
        parts = ["# Task (plain text, as given to the human)", "", self.task_text.strip()]

        hints: list[str] = []
        if self.public_webhook_url:
            hints.append(
                f"PUBLIC_WEBHOOK_URL is available at runtime: {self.public_webhook_url} "
                "(treat as a strong signal for long_running_webhook tasks).",
            )
        if self.verify_task_name_override:
            hints.append(
                f"Use verify_task_name='{self.verify_task_name_override}' "
                "(human-supplied override; do not change it).",
            )
        if hints:
            parts += ["", "# Run-level context", *(f"- {h}" for h in hints)]

        if self.critique:
            parts += [
                "",
                "# Critique of previous plan (orchestrator)",
                self.critique.strip(),
                "",
                "Address every point above. Do NOT repeat the prior mistake.",
            ]

        parts += [
            "",
            "# Instruction",
            "Produce the plan JSON now, conforming exactly to the schema.",
        ]
        return "\n".join(parts)

    # ── Plan validation ────────────────────────────────────────────────────

    def _validate_plan(self, plan: dict[str, Any]) -> None:
        """Light post-hoc checks beyond what ``strict`` schema already enforces.

        ``chat_structured`` (with ``strict=True``) already enforces field
        presence, types, and the ``task_family`` enum. We additionally guard
        against trivially broken plans the schema can't express:

        Raises:
            ValueError: When ``steps`` is empty (a plan with no steps is
                useless to the Solver) or when ``verify_task_name`` is blank.
        """
        if not plan.get("steps"):
            raise ValueError("plan.steps must not be empty")
        if not plan.get("verify_task_name", "").strip():
            raise ValueError("plan.verify_task_name must not be blank")

    # ── Persistence ────────────────────────────────────────────────────────

    def _write_plan(self, plan: dict[str, Any]) -> Path:
        """Write the validated plan to ``self.output_path`` and return it.

        Creates parent directories as needed so the orchestrator dispatcher
        can pass a path inside a per-run workspace without pre-creating it.
        """
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.output_path.write_text(
            json.dumps(plan, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        self.log.info("planner wrote plan to %s", self.output_path)
        return self.output_path

    # ── Result shaping ─────────────────────────────────────────────────────

    def _build_summary(self, plan: dict[str, Any], plan_path: Path) -> dict[str, Any]:
        """Build the compact dict returned to the orchestrator's dispatcher.

        Deliberately omits ``steps``/``hints``/``input_data``/``extracted_resources``/
        ``expected_output``/``success_check``:
        those live on disk in ``plan.json`` and are read by the Solver, not by
        the Orchestrator. Keeping the orchestrator's context lean is the whole
        point of the spawn-then-summarize pattern.
        """
        return {
            "outcome": "plan",
            "plan_path": str(plan_path),
            "task_family": plan["task_family"],
            "goal": plan["goal"],
            "verify_task_name": plan["verify_task_name"],
            "required_env": list(plan["required_env"]),
            "summary": self._render_summary_line(plan),
        }

    @staticmethod
    def _render_summary_line(plan: dict[str, Any]) -> str:
        """One-line human-readable digest for logs and the orchestrator's tool result."""
        return (
            f"goal={plan['goal']!r} "
            f"task_family={plan['task_family']} "
            f"verify_task_name={plan['verify_task_name']} "
            f"steps={len(plan.get('steps', []))} "
            f"required_env={plan.get('required_env', [])}"
        )

    # ── Error / completion bookkeeping ─────────────────────────────────────

    def _on_error(self, exc: Exception) -> dict[str, Any]:
        """Convert any failure into a structured ``{outcome: 'error'}`` result.

        Catches both LLM-side issues (network, schema enforcement) and our own
        post-hoc ``_validate_plan`` violations. The orchestrator decides what
        to do with the error_summary (re-plan with critique, or give up).
        """
        self.log.exception("planner failed: %s", exc)
        message = f"{type(exc).__name__}: {exc}"
        self._emitter.emit(AgentError(
            type="agent.error",
            ctx=self.ctx,
            error_type="planner_failure",
            message=message,
            step=1,
        ))
        return {"outcome": "error", "error_summary": message}

    def _finalize(self, result: dict[str, Any], run_t0: float) -> dict[str, Any]:
        """Emit ``AgentCompleted`` and return the result unchanged."""
        self._emitter.emit(AgentCompleted(
            type="agent.completed",
            ctx=self.ctx,
            duration_ms=(time() - run_t0) * 1000,
            result=result.get("outcome"),
        ))
        return result
