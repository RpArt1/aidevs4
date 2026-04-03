"""Persistent conversation history per session (JSON files under /tmp/sessions)."""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

_DEFAULT_DIR = Path("/tmp/sessions")
_SAFE_SESSION_ID = re.compile(r"^[A-Za-z0-9._-]+$")


class SessionPersistenceError(OSError):
    """Raised when history cannot be read or written reliably."""


class SessionManager:
    """Stores OpenAI/Anthropic-style message lists as one JSON file per session."""

    def __init__(self, base_dir: str | Path | None = None) -> None:
        self._dir = Path(base_dir) if base_dir is not None else _DEFAULT_DIR
        try:
            self._dir.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            raise SessionPersistenceError(
                f"Cannot create sessions directory {self._dir!s}"
            ) from e

    def _path_for(self, session_id: str) -> Path:
        sid = session_id.strip()
        if not sid or not _SAFE_SESSION_ID.fullmatch(sid):
            raise ValueError(
                "session_id must be non-empty and contain only "
                "letters, digits, '.', '_' or '-'"
            )
        return self._dir / f"{sid}.json"

    def get_history(self, session_id: str) -> list[dict[str, Any]]:
        path = self._path_for(session_id)
        if not path.is_file():
            return []
        try:
            text = path.read_text(encoding="utf-8")
        except OSError as e:
            log.warning("Cannot read session file %s: %s", path, e)
            return []
        try:
            data = json.loads(text)
        except json.JSONDecodeError as e:
            log.warning("Invalid JSON in session file %s: %s", path, e)
            return []
        if not isinstance(data, list) or not all(isinstance(m, dict) for m in data):
            log.warning("Session file %s must be a JSON array of objects; ignoring", path)
            return []
        return data

    def save_history(self, session_id: str, messages: list[dict[str, Any]]) -> None:
        path = self._path_for(session_id)
        tmp = path.with_suffix(".json.tmp")
        try:
            with tmp.open("w", encoding="utf-8") as f:
                json.dump(messages, f, ensure_ascii=False)
                f.write("\n")
            tmp.replace(path)
        except OSError as e:
            try:
                if tmp.is_file():
                    tmp.unlink()
            except OSError:
                pass
            raise SessionPersistenceError(
                f"Cannot save session {session_id!r} to {path!s}"
            ) from e

    def add_message(self, session_id: str, role: str, content: str) -> None:
        history = self.get_history(session_id)
        history.append({"role": role, "content": content})
        self.save_history(session_id, history)
