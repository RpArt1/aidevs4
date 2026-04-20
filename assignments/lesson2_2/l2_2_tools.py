"""Tool schemas, handler stubs, and executor for the lesson 2_2 electricity-puzzle agent.

Skeleton only — real board math, vision, and hub I/O land in follow-up todos
(`board.py`, `vision.py`, `hub_client.py`). Every stub returns ``"fatal": true``
so the agent loop exits cleanly during skeleton runs.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

from dotenv import load_dotenv

from common import get_logger

load_dotenv()
load_dotenv(Path(__file__).parent / "secrets.env")

log = get_logger(__name__)

_api_key = os.environ["AIDEVS_API_KEY"]
_verify_url = os.environ["AIDEVS_VERIFY_URL"]
_image_url = os.environ["ELECTRICITY_IMAGE_URL"]
_solved_image_url = os.environ["SOLVED_ELECTRICITY_IMAGE_URL"]
_task_name = os.environ["ELECTRICITY_TASK"]

MAX_ROTATIONS = 20
_CELL_RE = re.compile(r"^[1-3]x[1-3]$")

_STUB_HINT = (
    "Skeleton stub — board.py / vision.py / hub_client.py are not implemented yet. "
    "This is a fatal stub response: stop the agent loop and report that the "
    "lesson2_2 skeleton is in place but real tool bodies are pending."
)


# ── Tool implementations (stubs) ──────────────────────────────────────────────

def read_board() -> str:
    """Stub: return an empty board snapshot with a fatal flag so the loop aborts."""
    log.info("read_board called (stub)")
    return json.dumps({
        "current": {},
        "target": {},
        "plan": [],
        "rotations_used": 0,
        "rotations_budget": MAX_ROTATIONS,
        "fatal": True,
        "hints": _STUB_HINT,
    })


def rotate(cell: str) -> str:
    """Stub: validate the cell address, then return a fatal stub response."""
    log.info("rotate called cell=%r (stub)", cell)

    if not isinstance(cell, str) or not _CELL_RE.match(cell):
        log.warning("rotate invalid cell=%r", cell)
        return json.dumps({
            "error": f"Invalid cell address: {cell!r}",
            "recoveryHints": (
                "cell must match the regex ^[1-3]x[1-3]$ — e.g. '1x1', '2x3', '3x2'. "
                "Retry with a valid address."
            ),
        })

    return json.dumps({
        "cell": cell,
        "response": None,
        "fatal": True,
        "hints": _STUB_HINT,
    })


def reset_board() -> str:
    """Stub: return a fatal stub response so the loop aborts."""
    log.info("reset_board called (stub)")
    return json.dumps({
        "response": None,
        "fatal": True,
        "hints": _STUB_HINT,
    })


# ── Tool schemas ──────────────────────────────────────────────────────────────

tool_read_board = {
    "type": "function",
    "function": {
        "name": "read_board",
        "description": (
            "Read the current 3x3 electricity puzzle board, compare against the cached "
            "target, and return an ordered rotation plan. Free of network side-effects "
            "beyond fetching the puzzle image."
        ),
        "parameters": {
            "type": "object",
            "properties": {},
            "additionalProperties": False,
        },
    },
}

tool_rotate = {
    "type": "function",
    "function": {
        "name": "rotate",
        "description": (
            "Rotate a single tile 90° clockwise on the hub-side board. "
            "One call = one 90° rotation. A tile that needs N rotations requires N calls."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "cell": {
                    "type": "string",
                    "description": "Cell address in the form 'AxB' where A, B ∈ {1,2,3} (e.g. '2x3').",
                    "pattern": r"^[1-3]x[1-3]$",
                },
            },
            "required": ["cell"],
            "additionalProperties": False,
        },
    },
}

tool_reset_board = {
    "type": "function",
    "function": {
        "name": "reset_board",
        "description": (
            "Reset the puzzle to its initial state on the hub. Use only as an emergency "
            "recovery when the current plan has become unreachable."
        ),
        "parameters": {
            "type": "object",
            "properties": {},
            "additionalProperties": False,
        },
    },
}

TOOLS: list[dict] = [tool_read_board, tool_rotate, tool_reset_board]

_REGISTRY: dict[str, callable] = {
    "read_board": read_board,
    "rotate": rotate,
    "reset_board": reset_board,
}


# ── Dispatcher ────────────────────────────────────────────────────────────────

def execute_tool(name: str, arguments: dict) -> str:
    """Dispatch a tool call by name and return a JSON string result."""
    log.info("execute_tool name=%s arguments=%s", name, arguments)
    handler = _REGISTRY.get(name)
    if handler is None:
        log.warning("execute_tool unknown tool: %s", name)
        return json.dumps({
            "error": f"Unknown tool: {name}",
            "recoveryHints": (
                f"Available tools: {', '.join(_REGISTRY)}. Use one of these exact names."
            ),
        })
    return handler(**arguments)
