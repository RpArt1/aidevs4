"""Tool schemas and dispatcher for `PlannerAgent` (Tier 1: input grounding).

These tools are registered only with the Planner LLM (`PLANNER_TOOLS`), not with
the orchestrator. They mirror the orchestrator pattern: JSON tool results and a
single ``make_planner_dispatcher`` callable compatible with
``SuperAgentBase._process_tool_calls``.

* ``list_workspace`` — list files under the run workspace (optional subpath).
* ``read_workspace_file`` — read a bounded byte slice from a workspace file.
* ``http_preview`` — bounded HTTP(S) probe (HEAD metadata + small GET sample).
"""

from __future__ import annotations

import base64
import json
import ssl
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

PlannerToolDispatcher = Callable[[str, dict], str]

DEFAULT_READ_MAX_BYTES = 64 * 1024
DEFAULT_LIST_MAX_ENTRIES = 500
DEFAULT_HTTP_TIMEOUT_S = 15.0
DEFAULT_HTTP_BODY_SAMPLE_BYTES = 16 * 1024

_BLOCKED_HOSTNAMES = frozenset({"localhost", "localhost.localdomain"})


def _json(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False)


@dataclass(frozen=True)
class _HttpSsrfGuard:
    """Minimal SSRF hygiene: scheme allowlist + optional hostname allowlist."""

    allowed_hosts: frozenset[str] | None

    def check_url(self, raw_url: str) -> urllib.parse.ParseResult:
        parsed = urllib.parse.urlparse(raw_url.strip())
        if parsed.scheme not in ("http", "https"):
            raise ValueError("only http and https URLs are allowed")
        if not parsed.netloc:
            raise ValueError("URL must include a host")
        host = parsed.hostname
        if host is None:
            raise ValueError("URL must include a resolvable hostname")
        lowered = host.lower()
        if lowered in _BLOCKED_HOSTNAMES:
            raise ValueError("host localhost is not allowed")
        if lowered.startswith("127."):
            raise ValueError("loopback hosts are not allowed")
        if host == "::1":
            raise ValueError("loopback hosts are not allowed")
        if self.allowed_hosts is not None and lowered not in self.allowed_hosts:
            allowed = ", ".join(sorted(self.allowed_hosts))
            raise ValueError(f"host not in allowlist: {allowed}")
        return parsed


class _WorkspaceSandbox:
    """Path confinement for all workspace-relative tool paths."""

    def __init__(self, workspace: Path) -> None:
        self._root = workspace.resolve()

    @property
    def root(self) -> Path:
        return self._root

    def resolve_file(self, rel_path: str) -> Path:
        """Return a resolved path under ``workspace`` (files only)."""
        path = self._resolve_under_root(rel_path)
        if not path.is_file():
            raise ValueError("not a file or does not exist")
        return path

    def resolve_existing_under_root(self, rel_subpath: str) -> Path:
        """Directory or file that must exist."""
        path = self._resolve_under_root(rel_subpath)
        if not path.exists():
            raise ValueError("path does not exist")
        return path

    def _resolve_under_root(self, rel_path: str) -> Path:
        text = (rel_path or "").strip().replace("\\", "/")
        if not text or text == ".":
            candidate = self._root
        else:
            p = Path(text)
            if p.is_absolute():
                raise ValueError("path must be relative to the workspace root")
            candidate = (self._root / text).resolve()
        try:
            candidate.relative_to(self._root)
        except ValueError as exc:
            raise ValueError("path escapes workspace boundary") from exc
        return candidate


PLANNER_TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "list_workspace",
            "description": (
                "List files and subdirectories under the task workspace root, "
                "or under an optional subdirectory. Returns paths relative to "
                "the workspace, kind (file/dir), and size for files."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "subpath": {
                        "type": "string",
                        "description": (
                            "Optional path relative to the workspace root "
                            "(use empty string or omit for the root)."
                        ),
                    },
                    "max_entries": {
                        "type": "integer",
                        "description": (
                            "Optional cap on returned entries (default 500)."
                        ),
                    },
                },
                "required": [],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_workspace_file",
            "description": (
                "Read a slice of a text or binary file under the workspace. "
                "Returns base64 for non-UTF8 content; UTF-8 text otherwise."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "File path relative to the workspace root.",
                    },
                    "offset": {
                        "type": "integer",
                        "description": "Byte offset to start reading (default 0).",
                    },
                    "max_bytes": {
                        "type": "integer",
                        "description": (
                            f"Maximum bytes to read (default {DEFAULT_READ_MAX_BYTES})."
                        ),
                    },
                },
                "required": ["path"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "http_preview",
            "description": (
                "Preview a public HTTP(S) URL from the task: status line from "
                "HEAD when possible, Content-Type, and a bounded raw body sample "
                "from GET. Use to ground URLs mentioned in the task."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "http or https URL to probe.",
                    },
                    "max_body_bytes": {
                        "type": "integer",
                        "description": (
                            "Max bytes of response body to include in the sample "
                            f"(default {DEFAULT_HTTP_BODY_SAMPLE_BYTES})."
                        ),
                    },
                },
                "required": ["url"],
                "additionalProperties": False,
            },
        },
    },
]


def make_planner_dispatcher(
    workspace: Path,
    *,
    allowed_http_hosts: frozenset[str] | None = None,
    list_max_entries: int = DEFAULT_LIST_MAX_ENTRIES,
    http_timeout_s: float = DEFAULT_HTTP_TIMEOUT_S,
) -> PlannerToolDispatcher:
    """Build a tool dispatcher confined to ``workspace``.

    Args:
        workspace: Run workspace root (absolute path recommended).
        allowed_http_hosts: If set, ``http_preview`` permits only these hostnames
            (lowercase compared). If ``None``, any host is allowed except blocked
            loopback aliases.
        list_max_entries: Hard cap for ``list_workspace``.
        http_timeout_s: Socket timeout for HTTP operations.

    Returns:
        Callable ``(name, args) -> json_str`` for ``_process_tool_calls``.
    """

    sandbox = _WorkspaceSandbox(workspace)
    ssrf = _HttpSsrfGuard(allowed_hosts=allowed_http_hosts)
    ssl_ctx = ssl.create_default_context()

    def dispatch(name: str, args: dict[str, Any]) -> str:
        if name == "list_workspace":
            return _tool_list_workspace(sandbox, args, list_max_entries)
        if name == "read_workspace_file":
            return _tool_read_workspace_file(sandbox, args)
        if name == "http_preview":
            return _tool_http_preview(ssrf, args, http_timeout_s, ssl_ctx)
        return _json({"error": f"unknown planner tool: {name}"})

    return dispatch


def _tool_list_workspace(
    sandbox: _WorkspaceSandbox,
    args: dict[str, Any],
    default_max: int,
) -> str:
    sub = str(args.get("subpath") or "").strip()
    max_entries = int(args.get("max_entries") or default_max)
    if max_entries < 1:
        return _json({"error": "max_entries must be at least 1"})
    max_entries = min(max_entries, DEFAULT_LIST_MAX_ENTRIES)

    try:
        base = sandbox.resolve_existing_under_root(sub)
    except ValueError as exc:
        return _json({"error": str(exc)})

    entries: list[dict[str, Any]] = []
    if base.is_file():
        rel = base.relative_to(sandbox.root)
        entries.append(
            {
                "path": str(rel).replace("\\", "/"),
                "kind": "file",
                "size": base.stat().st_size,
            },
        )
        return _json(
            {
                "workspace_root": str(sandbox.root),
                "entries": entries,
                "truncated": False,
            },
        )

    try:
        children = sorted(base.iterdir(), key=lambda p: p.name.lower())
    except OSError as exc:
        return _json({"error": str(exc)})
    truncated = len(children) > max_entries
    for child in children[:max_entries]:
        try:
            rel = child.relative_to(sandbox.root)
        except ValueError:
            continue
        kind = "dir" if child.is_dir() else "file"
        item: dict[str, Any] = {
            "path": str(rel).replace("\\", "/"),
            "kind": kind,
        }
        if child.is_file():
            item["size"] = child.stat().st_size
        entries.append(item)

    return _json(
        {
            "workspace_root": str(sandbox.root),
            "entries": entries,
            "truncated": truncated,
        },
    )


def _tool_read_workspace_file(sandbox: _WorkspaceSandbox, args: dict[str, Any]) -> str:
    rel = str(args.get("path") or "")
    offset = int(args.get("offset") or 0)
    max_bytes = int(args.get("max_bytes") or DEFAULT_READ_MAX_BYTES)
    if offset < 0:
        return _json({"error": "offset must be non-negative"})
    if max_bytes < 1:
        return _json({"error": "max_bytes must be at least 1"})
    max_bytes = min(max_bytes, DEFAULT_READ_MAX_BYTES)

    try:
        path = sandbox.resolve_file(rel)
    except ValueError as exc:
        return _json({"error": str(exc)})

    try:
        with path.open("rb") as handle:
            if offset:
                handle.seek(offset)
            raw = handle.read(max_bytes)
    except OSError as exc:
        return _json({"error": f"read failed: {exc}"})

    rel_out = path.relative_to(sandbox.root)
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        return _json(
            {
                "path": str(rel_out).replace("\\", "/"),
                "encoding": "base64",
                "content": base64.standard_b64encode(raw).decode("ascii"),
                "byte_length": len(raw),
                "offset": offset,
            },
        )

    return _json(
        {
            "path": str(rel_out).replace("\\", "/"),
            "encoding": "utf-8",
            "content": text,
            "byte_length": len(raw),
            "offset": offset,
        },
    )


@dataclass(frozen=True)
class _HttpBodyFetch:
    """Bounded GET result for ``http_preview`` (body slice + optional transport error)."""

    body: bytes
    body_note: str
    get_error: str | None = None


def _http_preview_max_body_or_error(args: dict[str, Any]) -> tuple[int, str | None]:
    """Return capped ``max_body`` or ``(0, error_message)`` when args are invalid."""
    max_body = int(args.get("max_body_bytes") or DEFAULT_HTTP_BODY_SAMPLE_BYTES)
    if max_body < 0:
        return 0, "max_body_bytes must be non-negative"
    capped = min(max_body, DEFAULT_HTTP_BODY_SAMPLE_BYTES * 4)
    return capped, None


def _http_response_meta(resp: Any) -> dict[str, Any]:
    return {
        "status": getattr(resp, "status", None) or resp.getcode(),
        "content_type": resp.headers.get("Content-Type") if resp.headers else None,
    }


def _http_request_head(
    url: str,
    timeout_s: float,
    ssl_ctx: ssl.SSLContext,
) -> dict[str, Any]:
    try:
        head_req = urllib.request.Request(url, method="HEAD")
        with urllib.request.urlopen(head_req, timeout=timeout_s, context=ssl_ctx) as resp:
            return _http_response_meta(resp)
    except urllib.error.HTTPError as resp:
        return {
            "status": resp.code,
            "content_type": resp.headers.get("Content-Type"),
        }
    except (urllib.error.URLError, OSError, TimeoutError) as exc:
        return {"head_error": str(exc)}


def _http_request_body_sample(
    url: str,
    max_body: int,
    timeout_s: float,
    ssl_ctx: ssl.SSLContext,
) -> _HttpBodyFetch:
    try:
        get_req = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(get_req, timeout=timeout_s, context=ssl_ctx) as resp:
            raw = resp.read(max_body + 1)[:max_body]
            note = "sample may be truncated" if len(raw) == max_body else ""
            return _HttpBodyFetch(body=raw, body_note=note, get_error=None)
    except urllib.error.HTTPError as resp:
        raw = resp.read(max_body + 1)[:max_body] if resp.fp else b""
        return _HttpBodyFetch(body=raw, body_note="", get_error=None)
    except (urllib.error.URLError, OSError, TimeoutError) as exc:
        return _HttpBodyFetch(body=b"", body_note="", get_error=str(exc))


def _http_preview_payload(
    url: str,
    head_info: dict[str, Any],
    fetch: _HttpBodyFetch,
) -> dict[str, Any]:
    if fetch.get_error is not None:
        return {"url": url, "head": head_info, "get_error": fetch.get_error}
    preview: dict[str, Any] = {
        "url": url,
        "head": head_info,
        "body_sample_bytes": len(fetch.body),
    }
    if fetch.body_note:
        preview["body_note"] = fetch.body_note
    _attach_utf8_or_base64_body(preview, fetch.body)
    return preview


def _attach_utf8_or_base64_body(preview: dict[str, Any], body: bytes) -> None:
    try:
        preview["body_text"] = body.decode("utf-8")
        preview["body_encoding"] = "utf-8"
    except UnicodeDecodeError:
        preview["body_encoding"] = "base64"
        preview["body_base64"] = base64.standard_b64encode(body).decode("ascii")


def _tool_http_preview(
    ssrf: _HttpSsrfGuard,
    args: dict[str, Any],
    timeout_s: float,
    ssl_ctx: ssl.SSLContext,
) -> str:
    max_body, arg_err = _http_preview_max_body_or_error(args)
    if arg_err is not None:
        return _json({"error": arg_err})

    raw_url = str(args.get("url") or "")
    try:
        parsed = ssrf.check_url(raw_url)
    except ValueError as exc:
        return _json({"error": str(exc)})

    url = urllib.parse.urlunparse(parsed)
    head_info = _http_request_head(url, timeout_s, ssl_ctx)
    fetch = _http_request_body_sample(url, max_body, timeout_s, ssl_ctx)
    return _json(_http_preview_payload(url, head_info, fetch))
