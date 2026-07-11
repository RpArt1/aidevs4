"""Tool schemas and dispatcher for SolverAgent.

Two tools:
- execute_python: write and run a Python script as a subprocess in the workspace
- submit_answer: POST answer to aidevs verify API; returns fatal sentinel on flag
"""
from __future__ import annotations

import json
import os
import queue
import re
import subprocess
import sys
import threading
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from time import time
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


@dataclass
class CapturedLine:
    """One line of output captured in real time from a running subprocess."""
    timestamp: datetime
    stream: str   # "stdout" | "stderr"
    text: str


def _run_interleaved(
    script_path: Path,
    timeout: int,
    env: dict[str, str],
) -> tuple[int | None, list[CapturedLine], str | None]:
    """Run a script and capture stdout/stderr as chronologically ordered lines.

    Uses Popen with two reader threads so lines from both streams are
    timestamped at arrival and interleaved in the order they were produced.

    Args:
        script_path: Python script to execute.
        timeout: Wall-clock seconds before the process is killed.
        env: Full environment for the subprocess.

    Returns:
        (returncode, lines, error_message).
        returncode is None on timeout; error_message describes the failure.
    """
    captured: list[CapturedLine] = []
    line_queue: queue.Queue[CapturedLine | None] = queue.Queue()

    proc = subprocess.Popen(
        [sys.executable, str(script_path)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
        bufsize=1,
    )

    def _reader(stream: Any, name: str) -> None:
        try:
            for raw in stream:
                line_queue.put(CapturedLine(datetime.now(), name, raw.rstrip()))
        finally:
            line_queue.put(None)

    threads = [
        threading.Thread(target=_reader, args=(proc.stdout, "stdout"), daemon=True),
        threading.Thread(target=_reader, args=(proc.stderr, "stderr"), daemon=True),
    ]
    for t in threads:
        t.start()

    done = 0
    deadline = time() + timeout

    while done < 2:
        remaining = deadline - time()
        if remaining <= 0:
            proc.kill()
            # Drain whatever the readers still have so the threads can exit
            while done < 2:
                try:
                    item = line_queue.get(timeout=0.5)
                    if item is None:
                        done += 1
                    else:
                        captured.append(item)
                except queue.Empty:
                    break
            for t in threads:
                t.join(timeout=1.0)
            return None, captured, f"script timed out after {timeout}s"
        try:
            item = line_queue.get(timeout=min(remaining, 0.2))
            if item is None:
                done += 1
            else:
                captured.append(item)
        except queue.Empty:
            pass

    for t in threads:
        t.join(timeout=1.0)
    proc.wait()
    return proc.returncode, captured, None


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
        step, script_name, timeout, code.count("\n") + 1,
    )

    log_path = script_path.with_suffix(".log")
    env = {**os.environ, "WORKSPACE": str(solver.workspace)}
    returncode, captured_lines, timeout_error = _run_interleaved(script_path, timeout, env)

    stdout = "\n".join(l.text for l in captured_lines if l.stream == "stdout")
    stderr = "\n".join(l.text for l in captured_lines if l.stream == "stderr")

    if timeout_error is not None:
        solver.record_execution_stderr("")
        solver.log.warning(
            "script.timeout  step=%d  script=%s  after=%ds",
            step, script_name, timeout,
        )
        _write_script_log(log_path, captured_lines=captured_lines, error=timeout_error)
        return json.dumps({"error": timeout_error})

    solver.record_execution_stderr(stderr)
    _log_script_done(solver, step, script_name, returncode, stdout, stderr, log_path.name)
    _write_script_log(log_path, captured_lines=captured_lines, returncode=returncode)

    preflight_result = _check_preflight_sentinel(solver, stdout)
    if preflight_result is not None:
        return preflight_result

    return json.dumps({"stdout": stdout, "stderr": stderr, "returncode": returncode})


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


_LOG_INLINE_LINES = 30   # max lines shown inline in app.log; full output always in .log


def _inline(text: str, log_name: str) -> str:
    """Return first N lines of text with an overflow note pointing to the .log file."""
    lines = text.splitlines()
    if len(lines) <= _LOG_INLINE_LINES:
        return text
    head = "\n".join(lines[:_LOG_INLINE_LINES])
    return f"{head}\n… (+{len(lines) - _LOG_INLINE_LINES} lines — see {log_name})"


def _write_script_log(
    log_path: Path,
    *,
    captured_lines: list[CapturedLine],
    returncode: int | None = None,
    error: str | None = None,
) -> None:
    """Write a chronological, timestamped .log file next to the executed script.

    All output lines appear in the order they were produced. The exit status
    is appended at the bottom — that is when it becomes known.

    Args:
        log_path: Destination path for the log file (e.g. script_0.log).
        captured_lines: Interleaved lines from _run_interleaved().
        returncode: Process exit code (None on timeout).
        error: Error message when execution was cut short (e.g. timeout).
    """
    rows: list[str] = []

    if not captured_lines:
        rows.append("(no output)")
    else:
        for cl in captured_lines:
            ts = cl.timestamp.strftime("%H:%M:%S.%f")[:-3]  # HH:MM:SS.mmm
            rows.append(f"[{ts}] [{cl.stream:6}] {cl.text}")

    rows.append("")
    rows.append(f"ERROR: {error}" if error is not None else f"returncode: {returncode}")

    log_path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def _log_script_done(
    solver: "SolverAgent",
    step: int,
    script_name: str,
    returncode: int,
    stdout: str,
    stderr: str,
    log_name: str,
) -> None:
    """Emit script completion to app.log.

    stdout is always surfaced (it is the script's intentional signal output).
    stderr is surfaced only on failure — on success it is library noise
    and its full content is preserved in the .log file.
    """
    solver.log.info(
        "script.done   step=%d  script=%s  returncode=%d  stdout_bytes=%d  stderr_bytes=%d",
        step, script_name, returncode, len(stdout), len(stderr),
    )

    if stdout.strip():
        solver.log.info(
            "script.stdout  step=%d  script=%s\n%s",
            step, script_name, _inline(stdout, log_name),
        )
    else:
        solver.log.info("script.stdout  step=%d  script=%s  (empty)", step, script_name)

    if returncode != 0 and stderr.strip():
        solver.log.warning(
            "script.stderr  step=%d  script=%s  returncode=%d\n%s",
            step, script_name, returncode, _inline(stderr, log_name),
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
