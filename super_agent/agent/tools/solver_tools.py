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
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

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
    step = solver._last_step

    script_path = solver.workspace / f"script_{solver._script_counter}.py"
    script_name = script_path.name
    solver._script_counter += 1
    script_path.write_text(code, encoding="utf-8")

    solver.log.info(
        "script.start  step=%d  script=%s  timeout=%ds  lines=%d",
        step,
        script_name,
        timeout,
        code.count("\n") + 1,
    )

    log_path = script_path.with_suffix(".log")

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
        solver.record_execution_stderr(stderr)
        _log_script_done(solver, step, script_name, proc.returncode, stdout, stderr)
        _write_script_log(log_path, stdout=stdout, stderr=stderr, returncode=proc.returncode)

        preflight_result = _check_preflight_sentinel(solver, stdout)
        if preflight_result is not None:
            return preflight_result

        return json.dumps({"stdout": stdout, "stderr": stderr, "returncode": proc.returncode})
    except subprocess.TimeoutExpired:
        solver.record_execution_stderr("")
        solver.log.warning(
            "script.done   step=%d  script=%s  timed_out after=%ds",
            step,
            script_name,
            timeout,
        )
        _write_script_log(log_path, error=f"script timed out after {timeout}s")
        return json.dumps({"error": f"script timed out after {timeout}s"})


def _check_preflight_sentinel(solver: "SolverAgent", stdout: str) -> str | None:
    """Scan stdout for a PREFLIGHT FAILED sentinel and hard-stop the run if found.

    The solver prompt instructs the LLM to print ``PREFLIGHT FAILED: <reason>``
    and call ``sys.exit(1)`` when any preflight check fails.  This function
    provides a mechanical second layer of enforcement: even if the LLM forgets
    ``sys.exit(1)``, any ``PREFLIGHT FAILED:`` line in stdout terminates the run
    immediately so the solver cannot silently continue with fabricated data.

    Args:
        solver: The running SolverAgent instance.
        stdout: Captured standard output from the just-executed script.

    Returns:
        A fatal JSON string when the sentinel is found, otherwise None.
    """
    for line in stdout.splitlines():
        if "PREFLIGHT FAILED:" in line:
            reason = line.split("PREFLIGHT FAILED:", 1)[1].strip()
            solver.log.error("preflight.failed  reason=%s", reason)
            solver._final_result = {
                "outcome": "preflight_failed",
                "error_summary": f"Preflight check failed: {reason}",
            }
            return json.dumps({
                "fatal": True,
                "outcome": "preflight_failed",
                "reason": reason,
            })
    return None


def _write_script_log(
    log_path: Path,
    *,
    stdout: str = "",
    stderr: str = "",
    returncode: int | None = None,
    error: str | None = None,
) -> None:
    """Write a human-readable .log file next to the executed script.

    Args:
        log_path: Destination path for the log file (e.g. script_0.log).
        stdout: Captured standard output of the script.
        stderr: Captured standard error of the script.
        returncode: Process exit code, or None when execution never finished.
        error: High-level error message (e.g. timeout) when no returncode is available.
    """
    lines: list[str] = []

    if error is not None:
        lines += [f"ERROR: {error}", ""]
    else:
        lines += [f"returncode: {returncode}", ""]

    lines += ["=== STDOUT ===", stdout or "(empty)", ""]
    lines += ["=== STDERR ===", stderr or "(empty)", ""]

    log_path.write_text("\n".join(lines), encoding="utf-8")


def _log_script_done(
    solver: "SolverAgent",
    step: int,
    script_name: str,
    returncode: int,
    stdout: str,
    stderr: str,
) -> None:
    """Emit completion logs so script runs are easy to correlate in app.log."""
    solver.log.info(
        "script.done   step=%d  script=%s  returncode=%d  stdout_bytes=%d  stderr_bytes=%d",
        step,
        script_name,
        returncode,
        len(stdout),
        len(stderr),
    )
    if returncode != 0 and stderr.strip():
        solver.log.warning(
            "script.stderr  step=%d  script=%s  returncode=%d\n%s",
            step,
            script_name,
            returncode,
            stderr[-2000:],
        )


def _submit_answer(solver: "SolverAgent", args: dict) -> str:
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
