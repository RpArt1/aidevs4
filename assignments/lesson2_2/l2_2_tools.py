"""Tool schemas, handlers, and executor for the lesson 2_2 electricity-puzzle agent.

Three tools are exposed to the planner:

* ``read_current_board(reset=False)`` — fetches the live puzzle PNG from the hub,
  runs it through :class:`BoardVision`, and returns the structured 3x3
  description. Passing ``reset=True`` asks the hub to reset the puzzle first.
* ``read_target_board()`` — same shape as above, but for the static solved
  reference image. The target is immutable, so the description is cached
  in-memory after the first successful call.
* ``rotate(cell)`` — POSTs one 90° clockwise rotation to the hub.

All handlers follow the project's ``hints`` / ``recoveryHints`` rule
(see ``.cursor/rules/agent-tool-hints.mdc``).
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

import requests
from dotenv import load_dotenv

from common import get_logger
from assignments.lesson2_2.board_vision import BoardVision
from assignments.lesson2_2.electricity_api_client import ElectricityApiClient

load_dotenv()
load_dotenv(Path(__file__).parent / "secrets.env")

log = get_logger(__name__)

_api_key = os.environ["AIDEVS_API_KEY"]
_verify_url = os.environ["AIDEVS_VERIFY_URL"]
_image_url = os.environ["ELECTRICITY_IMAGE_URL"]
_target_image_url = os.environ["SOLVED_ELECTRICITY_IMAGE_URL"]
_task_name = os.environ["ELECTRICITY_TASK"]

_client = ElectricityApiClient(
    api_key=_api_key,
    verify_url=_verify_url,
    image_url=_image_url,
    target_image_url=_target_image_url,
    task_name=_task_name,
)

_vision = BoardVision()

# Target board never changes — cache the structured description after the
# first successful read.
_target_board_cache: dict[str, Any] | None = None

# Current board is cached too, but only between rotations. Any successful
# rotate() call (or a reset=True read) invalidates this cache. This keeps the
# planner from spending iterations re-reading a board it already knows hasn't
# changed, and nudges it to act instead of re-observing.
_current_board_cache: dict[str, Any] | None = None

_CELL_RE = re.compile(r"^[1-3]x[1-3]$")

_CW_ROTATION_HINT = (
    "Rotation semantics: 90° clockwise maps top→right→bottom→left→top. "
    "N rotations on a tile = N separate rotate() calls. Minimise total calls."
)

_VERIFY_HINT = (
    "Vision can hallucinate — after a small batch of rotations call "
    "read_current_board again to verify the real state before issuing more. "
    "Never trust the previous board description as ground truth once you've rotated."
)


# ── Tool implementations ──────────────────────────────────────────────────────

def read_current_board(reset: bool = False) -> str:
    """Fetch the live puzzle PNG and return its structured 3x3 description.

    When ``reset`` is True, asks the hub to reset the puzzle to its initial
    state before the image is produced. Resets are expensive — use sparingly.

    A successful read is cached in-process and re-served on subsequent calls
    until the next successful rotate() (or a reset=True read) invalidates it.
    The cached response carries a hint telling the planner that no rotations
    have happened, so re-reading is uninformative and it should rotate next.
    """
    global _current_board_cache
    log.info("read_current_board called reset=%s", reset)

    if reset:
        _current_board_cache = None
    elif _current_board_cache is not None:
        cached = dict(_current_board_cache)
        cached["cached"] = True
        cached["reset_requested"] = False
        cached["hints"] = (
            "Returning the cached board snapshot — no rotate() has succeeded "
            "since the last read, so the board state CANNOT have changed. "
            "Re-reading is wasted budget; pick the next rotate() now (or stop "
            "if the board already matches the target). "
            + _CW_ROTATION_HINT
        )
        log.info("read_current_board cache hit — returning cached snapshot")
        return json.dumps(cached, ensure_ascii=False)

    try:
        png_bytes = _client.fetch_board_png(reset=reset)
    except requests.HTTPError as exc:
        status = exc.response.status_code if exc.response is not None else None
        log.error("read_current_board fetch failed status=%s err=%s", status, exc)
        return json.dumps({
            "error": f"Failed to fetch current board PNG (HTTP {status}).",
            "recoveryHints": (
                "The hub rejected or failed the board image request. "
                "If this is transient, retry once; if it persists, the board URL "
                "or task name may be misconfigured and the run should be aborted."
            ),
        })
    except requests.RequestException as exc:
        log.error("read_current_board network error: %s", exc)
        return json.dumps({
            "error": f"Network error fetching current board PNG: {exc}",
            "recoveryHints": (
                "Transient network failure. Retry read_current_board once. "
                "If it still fails, stop and report the issue."
            ),
        })

    try:
        description = _vision.describe_board(png_bytes)
    except Exception as exc:  # noqa: BLE001 — surface vision errors to the agent
        log.exception("read_current_board vision failed")
        return json.dumps({
            "error": f"Vision subagent failed to describe the board: {exc}",
            "recoveryHints": (
                "The vision model could not parse the tiles. Retry read_current_board "
                "once — transient model errors can happen. If it fails again, stop "
                "and report a vision-pipeline problem."
            ),
        })

    _current_board_cache = dict(description)

    description["reset_requested"] = reset
    description["cached"] = False
    description["hints"] = (
        "Board read successfully. Diff this 'grid' against the cached "
        "read_target_board() result tile-by-tile: a tile is correct iff its "
        "four edges (top/right/bottom/left) match the target's. For each "
        "mismatched tile, count the minimum number of 90° clockwise rotations "
        "needed to align current edges with target edges (0–3), and issue "
        "that many rotate() calls for that cell. "
        + _CW_ROTATION_HINT + " " + _VERIFY_HINT
    )
    log.info(
        "read_current_board success board_size=%s tile_size=%s reset=%s",
        description["board_size"], description["tile_size"], reset,
    )
    return json.dumps(description, ensure_ascii=False)


def read_target_board() -> str:
    """Return the structured description of the static solved board (cached)."""
    global _target_board_cache
    log.info("read_target_board called cached=%s", _target_board_cache is not None)

    if _target_board_cache is not None:
        cached = dict(_target_board_cache)
        cached["cached"] = True
        cached["hints"] = (
            "Returning cached target description. This is the desired end state — "
            "compare current tiles against these edges to decide rotations."
        )
        return json.dumps(cached, ensure_ascii=False)

    try:
        png_bytes = _client.fetch_target_png()
    except requests.HTTPError as exc:
        status = exc.response.status_code if exc.response is not None else None
        log.error("read_target_board fetch failed status=%s err=%s", status, exc)
        return json.dumps({
            "error": f"Failed to fetch target board PNG (HTTP {status}).",
            "recoveryHints": (
                "The solved/target image URL could not be retrieved. "
                "Retry once; if it still fails, the SOLVED_ELECTRICITY_IMAGE_URL "
                "is likely misconfigured — stop and report."
            ),
        })
    except requests.RequestException as exc:
        log.error("read_target_board network error: %s", exc)
        return json.dumps({
            "error": f"Network error fetching target board PNG: {exc}",
            "recoveryHints": (
                "Transient network failure. Retry read_target_board once. "
                "If it still fails, stop and report the issue."
            ),
        })

    try:
        description = _vision.describe_board(png_bytes)
    except Exception as exc:  # noqa: BLE001
        log.exception("read_target_board vision failed")
        return json.dumps({
            "error": f"Vision subagent failed to describe the target board: {exc}",
            "recoveryHints": (
                "The vision model could not parse the target tiles. Retry "
                "read_target_board once. If it fails again, stop and report."
            ),
        })

    _target_board_cache = description
    response = dict(description)
    response["cached"] = False
    response["hints"] = (
        "Target board described and cached for the rest of this run. This is the "
        "immutable goal state — every tile's edges here define what the current "
        "board must look like to solve the puzzle. Call read_current_board next "
        "and diff edges tile-by-tile to derive rotations."
    )
    log.info(
        "read_target_board success board_size=%s tile_size=%s (cached now)",
        description["board_size"], description["tile_size"],
    )
    return json.dumps(response, ensure_ascii=False)


def rotate(cell: str) -> str:
    """Send one 90° CW rotation for ``cell`` to the hub and return the response.

    Non-fatal: hub errors are recoverable (the agent can retry a different cell
    or re-read the board). Only validation failures short-circuit without
    touching the network.

    On success the cached current-board snapshot is invalidated so the next
    read_current_board() call goes back to the hub instead of serving stale
    state.
    """
    global _current_board_cache
    log.info("rotate called cell=%r", cell)

    if not isinstance(cell, str) or not _CELL_RE.match(cell):
        log.warning("rotate invalid cell=%r", cell)
        return json.dumps({
            "error": f"Invalid cell address: {cell!r}",
            "recoveryHints": (
                "cell must match the regex ^[1-3]x[1-3]$ — e.g. '1x1', '2x3', '3x2'. "
                "Retry with a valid address."
            ),
        })

    result = _client.rotate(cell)
    result["cell"] = cell

    if result["ok"]:
        _current_board_cache = None
        result["hints"] = (
            "Rotation applied. Inspect 'body' for a flag {FLG:...} — if present, "
            "the puzzle is solved and you should stop and report it. " + _CW_ROTATION_HINT
            + " " + _VERIFY_HINT
        )
        log.info("rotate success cell=%s status=%d", cell, result["status"])
    else:
        result["recoveryHints"] = (
            f"The hub returned an error (HTTP {result['status']}). Read 'body' for the "
            "exact reason. Do not retry with identical arguments — if the cell looks "
            "correct, call read_current_board to re-sync before trying again."
        )
        log.error(
            "rotate error cell=%s status=%d body=%.200s",
            cell, result["status"], result["body"],
        )

    return json.dumps(result, ensure_ascii=False)


# ── Tool schemas ──────────────────────────────────────────────────────────────

tool_read_current_board = {
    "type": "function",
    "function": {
        "name": "read_current_board",
        "description": (
            "Fetch the live 3x3 electricity puzzle PNG from the hub and return "
            "a structured description of each tile (open edges top/right/bottom/left, "
            "power-plant marker, label). The planner never sees raw pixels. "
            "Set reset=true to ask the hub to reset the puzzle to its initial state "
            "before the image is produced — use sparingly, resets are expensive."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "reset": {
                    "type": "boolean",
                    "description": (
                        "If true, request the hub to reset the puzzle before fetching. "
                        "Defaults to false."
                    ),
                    "default": False,
                },
            },
            "required": [],
            "additionalProperties": False,
        },
    },
}

tool_read_target_board = {
    "type": "function",
    "function": {
        "name": "read_target_board",
        "description": (
            "Return the structured description of the solved / target board image. "
            "The target is immutable and the result is cached in-memory after the "
            "first successful call, so subsequent calls are free. Use this once at "
            "the start of a run to learn the desired end state, then diff against "
            "read_current_board() to plan rotations."
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
            "One call = one 90° rotation. A tile that needs N rotations requires N calls. "
            "Returns the hub's response verbatim in 'body'; if the puzzle is complete, "
            "'body' will contain a flag of the form {FLG:...}."
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

TOOLS: list[dict] = [tool_read_current_board, tool_read_target_board, tool_rotate]

_REGISTRY: dict[str, callable] = {
    "read_current_board": read_current_board,
    "read_target_board": read_target_board,
    "rotate": rotate,
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
