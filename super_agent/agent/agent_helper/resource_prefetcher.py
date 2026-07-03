"""Pre-fetch GET-able URLs from task text and produce a structured preview block.

`ResourcePreFetcher` is called once inside `PlannerAgent._build_user_message()`
before the single `chat_structured` call. It enriches the planner prompt with
actual field names, types, and sample values from external data sources so the
plan's steps can reference concrete field names rather than guessing.

Design constraints (from plan spec):
- Only GET requests; POST endpoints and auth-blocked resources are silently skipped.
- 5 s per-URL timeout; failure is never propagated — the planner always runs.
- Max 8 KB per response body, truncated before snapshot to keep context lean.
- No new third-party dependencies: uses stdlib ``urllib`` only.
"""

from __future__ import annotations

import json
import logging
import re
import urllib.error
import urllib.request
from typing import Any

_MAX_BODY_BYTES = 8 * 1024
_FETCH_TIMEOUT_S = 5
_TEXT_PREVIEW_CHARS = 300

# Regex for bare URL extraction from free text.
_URL_RE = re.compile(r"https?://[^\s\"'<>)\]]+")

# Only fetch URLs whose path ends with one of these extensions.
# API endpoints, HTML pages, and Wikipedia links are intentionally excluded.
_DATA_EXTENSIONS: frozenset[str] = frozenset({
    ".json", ".jsonl", ".ndjson",
    ".csv", ".tsv",
    ".xml",
    ".yaml", ".yml",
    ".txt",
})

# Placeholder patterns that signal an unresolved secret inside a URL.
# Order matters: ${VAR} first, then named English marker, then Polish marker.
_PLACEHOLDER_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\$\{([A-Z_][A-Z0-9_]*)\}"), "dollar_brace"),
    (re.compile(r"<YOUR_API_KEY>", re.IGNORECASE), "angle_bracket"),
    (re.compile(r"tutaj-tw[oó]j-klucz", re.IGNORECASE), "polish_placeholder"),
]


class ResourcePreFetcher:
    """Fetch URL previews from task text and format them as a markdown block.

    Args:
        log: Logger instance shared with the calling agent so fetch activity
            appears in the existing planner log stream.
    """

    def __init__(self, log: logging.Logger) -> None:
        self._log = log

    # ── Public API ───────────────────────────────────────────────────────────

    def fetch_previews(self, task_text: str, env_vars: dict[str, str]) -> str:
        """Fetch all resolvable URLs from *task_text* and return a markdown block.

        Args:
            task_text: Plain-text task description passed to the planner.
            env_vars: Mapping of env-var name to its current value; used to
                substitute ``${VAR}`` and similar placeholder patterns in URLs.

        Returns:
            A ``# Resource Previews`` markdown block when at least one URL
            was successfully fetched; empty string otherwise.
        """
        urls = self._extract_urls(task_text)
        if not urls:
            return ""

        sections: list[str] = []
        for raw_url in urls:
            resolved = self._substitute_placeholders(raw_url, env_vars)
            if resolved is None:
                self._log.info("[prefetcher] skipping unresolvable URL: %s", raw_url)
                continue

            body = self._fetch_one(resolved)
            if body is None:
                continue

            snapshot = self._snapshot(body)
            sections.append(self._format_section(resolved, snapshot))

        if not sections:
            return ""

        header = (
            "# Resource Previews\n"
            "(Fetched before planning. Use structure info to write precise steps.)\n"
        )
        return header + "\n".join(sections)

    # ── URL extraction ───────────────────────────────────────────────────────

    def _extract_urls(self, text: str) -> list[str]:
        """Extract unique data-file URLs from *text*, preserving first-seen order.

        Only URLs whose path ends with a known data-file extension (JSON, CSV,
        XML, …) are kept. API endpoints, HTML pages, and documentation links are
        intentionally skipped — they produce noisy, unhelpful previews.
        """
        seen: set[str] = set()
        result: list[str] = []
        for url in _URL_RE.findall(text):
            url = url.rstrip(".,;:!?)")
            if not self._is_data_url(url):
                self._log.debug("[prefetcher] skipping non-data URL: %s", url)
                continue
            if url not in seen:
                seen.add(url)
                result.append(url)
        return result

    @staticmethod
    def _is_data_url(url: str) -> bool:
        """Return True only when the URL path ends with a known data-file extension."""
        from urllib.parse import urlparse
        path = urlparse(url).path.lower()
        return any(path.endswith(ext) for ext in _DATA_EXTENSIONS)

    # ── Placeholder substitution ─────────────────────────────────────────────

    def _substitute_placeholders(
        self, url: str, env_vars: dict[str, str]
    ) -> str | None:
        """Replace placeholder tokens in *url* with resolved values from *env_vars*.

        Returns:
            The substituted URL when all placeholders could be resolved,
            or ``None`` when a placeholder had no matching env var value.
        """
        result = url

        # Named ${VAR} placeholders: resolve from env_vars by name.
        for match in _PLACEHOLDER_PATTERNS[0][0].finditer(url):
            var_name = match.group(1)
            value = env_vars.get(var_name, "")
            if not value:
                self._log.info(
                    "[prefetcher] env var %r unset; skipping URL: %s", var_name, url
                )
                return None
            result = result.replace(match.group(0), value)

        # Generic opaque placeholders: require at least one env_var value to
        # substitute. When multiple values exist, try each in order and keep the
        # first non-empty one.
        for pattern, label in _PLACEHOLDER_PATTERNS[1:]:
            if pattern.search(result):
                replacement = next((v for v in env_vars.values() if v), None)
                if not replacement:
                    self._log.info(
                        "[prefetcher] %s placeholder with no env value; skipping URL: %s",
                        label,
                        url,
                    )
                    return None
                result = pattern.sub(replacement, result)

        return result

    # ── HTTP fetch ───────────────────────────────────────────────────────────

    def _fetch_one(self, url: str) -> bytes | None:
        """GET *url* with a 5 s timeout; return at most 8 KB of body or ``None``.

        Never raises — any network or HTTP error is logged and silently skipped
        so a bad URL never blocks plan generation.
        """
        self._log.info("[prefetcher] GET %s", url)
        try:
            with urllib.request.urlopen(url, timeout=_FETCH_TIMEOUT_S) as resp:
                body: bytes = resp.read(_MAX_BODY_BYTES)
            self._log.info("[prefetcher] %s → %d bytes", url, len(body))
            return body
        except urllib.error.HTTPError as exc:
            self._log.info("[prefetcher] HTTP %s for %s; skipping", exc.code, url)
        except urllib.error.URLError as exc:
            self._log.info("[prefetcher] URL error for %s: %s; skipping", url, exc.reason)
        except Exception as exc:  # noqa: BLE001
            self._log.info("[prefetcher] unexpected error for %s: %s; skipping", url, exc)
        return None

    # ── Structure snapshot ───────────────────────────────────────────────────

    def _snapshot(self, raw_bytes: bytes) -> dict[str, Any]:
        """Parse *raw_bytes* and return a lightweight structure descriptor.

        Returns:
            For a JSON list::

                {"type": "list", "length_hint": N, "item_sample": <first element>}

            For a JSON dict::

                {"type": "dict", "keys": [...], **{key: <first_value_truncated>}}

            For non-JSON text::

                {"type": "text", "preview": "<first 300 chars>"}
        """
        text = raw_bytes.decode("utf-8", errors="replace")
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            return {"type": "text", "preview": text[:_TEXT_PREVIEW_CHARS]}

        if isinstance(parsed, list):
            return {
                "type": "list",
                "length_hint": len(parsed),
                "item_sample": parsed[0] if parsed else None,
            }

        if isinstance(parsed, dict):
            keys = list(parsed.keys())
            sample = {k: self._truncate_value(parsed[k]) for k in keys[:5]}
            return {"type": "dict", "keys": keys, **sample}

        # Scalar JSON (number, string, bool, null) — treat as text.
        return {"type": "text", "preview": str(parsed)[:_TEXT_PREVIEW_CHARS]}

    @staticmethod
    def _truncate_value(value: Any) -> Any:
        """Return a compact representation of *value* safe for prompt injection."""
        if isinstance(value, str) and len(value) > 120:
            return value[:120] + "…"
        if isinstance(value, list):
            return value[:1]
        return value

    # ── Formatting ───────────────────────────────────────────────────────────

    @staticmethod
    def _format_section(url: str, snapshot: dict[str, Any]) -> str:
        """Render one URL's snapshot as a markdown sub-section."""
        lines = [f"## {url}"]
        kind = snapshot.get("type")

        if kind == "list":
            lines.append(f"type: list")
            lines.append(f"length_hint: {snapshot.get('length_hint')}")
            sample = snapshot.get("item_sample")
            if sample is not None:
                lines.append(f"item_sample: {json.dumps(sample, ensure_ascii=False)}")

        elif kind == "dict":
            lines.append(f"type: dict")
            keys = snapshot.get("keys", [])
            lines.append(f"keys: {json.dumps(keys, ensure_ascii=False)}")
            for k in keys[:5]:
                if k in snapshot and k not in ("type", "keys"):
                    lines.append(f"{k}: {json.dumps(snapshot[k], ensure_ascii=False)}")

        else:
            lines.append(f"type: text")
            lines.append(f"preview: {snapshot.get('preview', '')}")

        lines.append("")
        
        return "\n".join(lines)
