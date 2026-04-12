from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Any

import requests

from common import get_logger


class RailwayApiClient:
    def __init__(
        self,
        api_key: str,
        verify_url: str,
        max_503_retries: int = 10,
        base_backoff_s: float = 2.0,
    ) -> None:
        self._api_key = api_key
        self._verify_url = verify_url
        self._max_503_retries = max_503_retries
        self._base_backoff_s = base_backoff_s
        self.log = get_logger(__name__)

    def call(self, action: str, extra_params: dict[str, Any] | None = None) -> dict:
        payload = {
            "apikey": self._api_key,
            "task": "railway",
            "answer": {"action": action, **(extra_params or {})},
        }

        backoff = self._base_backoff_s
        for attempt in range(self._max_503_retries + 1):
            response = requests.post(self._verify_url, json=payload, timeout=30)

            remaining = self._parse_int_header(response, "X-RateLimit-Remaining")
            reset_at = self._parse_reset_header(response)

            self.log.info(
                "action=%s status=%d remaining=%s reset_at=%s body=%.500s",
                action,
                response.status_code,
                remaining,
                reset_at,
                response.text,
            )

            if response.status_code == 503:
                if attempt >= self._max_503_retries:
                    self.log.error("503 retries exhausted after %d attempts", attempt + 1)
                    break
                sleep_s = min(backoff, 60.0)
                self.log.warning(
                    "503 received attempt=%d sleeping=%.1fs", attempt + 1, sleep_s
                )
                time.sleep(sleep_s)
                backoff *= 2
                continue

            self._maybe_sleep_rate_limit(remaining, reset_at)

            try:
                body: dict | str = response.json()
            except ValueError:
                body = response.text

            return {
                "ok": response.ok,
                "status": response.status_code,
                "body": body,
                "rate_limit": {
                    "remaining": remaining,
                    "reset_at": reset_at,
                },
            }

        return {
            "ok": False,
            "status": 503,
            "body": "503 retries exhausted",
            "rate_limit": {"remaining": None, "reset_at": None},
        }

    def _parse_int_header(self, response: requests.Response, header: str) -> int | None:
        value = response.headers.get(header)
        if value is None:
            return None
        try:
            return int(value)
        except (ValueError, TypeError):
            return None

    def _parse_reset_header(self, response: requests.Response) -> str | None:
        value = response.headers.get("X-RateLimit-Reset")
        if value is None:
            return None
        # Value may be a Unix timestamp (int) or an ISO string — normalise to ISO.
        try:
            ts = int(value)
            return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
        except (ValueError, TypeError):
            return value

    def _maybe_sleep_rate_limit(self, remaining: int | None, reset_at: str | None) -> None:
        if remaining != 0 or reset_at is None:
            return
        try:
            reset_dt = datetime.fromisoformat(reset_at)
        except ValueError:
            self.log.warning("Could not parse reset_at=%s, skipping rate-limit sleep", reset_at)
            return

        now = datetime.now(tz=timezone.utc)
        sleep_s = max(0.0, (reset_dt - now).total_seconds() + 0.5)
        if sleep_s > 0:
            self.log.info("rate limit exhausted, sleeping=%.1fs until %s", sleep_s, reset_at)
            time.sleep(sleep_s)
