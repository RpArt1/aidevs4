"""Tool schemas and dispatcher for SolverAgent.

Two tools:
- execute_python: write and run a Python script as a subprocess in the workspace
- submit_answer: POST answer to aidevs verify API; returns fatal sentinel on flag
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from typing import TYPE_CHECKING, Any, Callable
import debugpy


from common.assignment_service import AssignmentService
from common.logger import get_logger

if TYPE_CHECKING:
    from ..solver_agent import SolverAgent


ToolDispatcher = Callable[[str, dict], str]

log = get_logger(__name__)

_FLAG_RE = re.compile(r"FLG:[A-Za-z0-9_]+")

SOLVER_TOOLS: list[dict[str, Any]] = [
  {
        "type": "function",
        "function": {
            "name": "execute_python",
            "description": (
                "Executes a standalone Python script in the workspace to fulfill the current step of your plan. "
                "Use this to fetch data, process files, call APIs, or perform computations. "
                "IMPORTANT: The script runs as a subprocess. You will only see what the script explicitly outputs to stdout/stderr. "
                "Therefore, you MUST include print() statements to capture the data you need. "
                "The environment includes WORKSPACE (the per-run directory) and standard API keys (AIDEVS_API_KEY, OPENROUTER_API_KEY)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "reason": {
                        "type": "string",
                        "description": (
                            "One or two sentences explaining WHY this script is needed "
                            "and WHAT it will produce. Written before the code is run."
                        ),
                    },
                    "code": {
                        "type": "string",
                        "description": "The complete, executable Python source code. It must include all necessary imports and print the final results. Base this code strictly on your established plan.",
                    },
                    "timeout": {
                        "type": "integer",
                        "description": "Maximum seconds to allow the script to run (capped at 120). Default 60.",
                        "default": 60,
                    },
                },
                "required": ["reason", "code"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "submit_answer",
            "description": (
                "Submit the final answer to the aidevs verification API. "
                "Call this only when you are confident the answer is correct. "
                "On success returns a flag (FLG:...). "
                "On failure the response includes a hint — read it and correct your approach before retrying. "
                "Always explain why you believe the answer is correct before calling this tool."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "task": {
                        "type": "string",
                        "description": "The task slug / verify_task_name from the plan.",
                    },
                    "answer": {
                        "description": "The answer to submit. Can be a string, list, dict, or number.",
                    },
                },
                "required": ["task", "answer"],
                "additionalProperties": False,
            },
        },
    },
]


def make_solver_dispatcher(solver: "SolverAgent") -> ToolDispatcher:
    """Build a tool dispatcher bound to one solver instance."""

    def dispatch(name: str, args: dict) -> str:
        if name == "execute_python":
            return _execute_python(solver, args)
        if name == "submit_answer":
            return _submit_answer(solver, args)
        return json.dumps({"error": f"unknown solver tool: {name}"})

    return dispatch


def _execute_python(solver: "SolverAgent", args: dict) -> str:
    code = str(args.get("code") or "")
    timeout = min(int(args.get("timeout") or 60), 120)

    script_path = solver.workspace / f"script_{solver._script_counter}.py"
    solver._script_counter += 1
    script_path.write_text(code, encoding="utf-8")

    try:
        proc = subprocess.run(
            [sys.executable, str(script_path)],
            capture_output=True,
            timeout=timeout,
            text=True,
            env={**os.environ, "WORKSPACE": str(solver.workspace)},
        )
        stdout = proc.stdout
        if len(stdout) > 8000:
            stdout = stdout[-8000:]
        stderr = proc.stderr
        if len(stderr) > 4000:
            stderr = stderr[-4000:]
        return json.dumps({"stdout": stdout, "stderr": stderr, "returncode": proc.returncode})
    except subprocess.TimeoutExpired:
        return json.dumps({"error": f"script timed out after {timeout}s"})


def _submit_answer(solver: "SolverAgent", args: dict) -> str:
    log.info(f"\n\n..... Submitting answer .........\n\n")
    # debugpy.listen(("0.0.0.0", 5678))
    # debugpy.wait_for_client()
    task = str(args.get("task") or "")
    answer = args.get("answer")

    try:
        assignment_service = AssignmentService()
        response = assignment_service.send(task=task, answer=answer)
        log.info(f"Response: {response}")
    except Exception as exc:
        log.error("submit_answer request failed task=%s: %s", task, exc)
        return json.dumps({"error": f"submission failed: {exc}"})

    response_str = json.dumps(response, ensure_ascii=False)
    match = _FLAG_RE.search(response_str)
    if match:
        flag = match.group(0)
        solver._final_result = {"outcome": "flag", "flag": flag, "submit_response": response}
        return json.dumps({"fatal": True, "flag": flag, "response": response})

    hint = response.get("message", "") if isinstance(response, dict) else ""
    log.warning(
        "submit_answer rejected task=%s hint=%s full_response=%s",
        task,
        hint or "(none)",
        response_str,
    )
    return json.dumps({"outcome": "incorrect", "response": response, "hint": hint})
