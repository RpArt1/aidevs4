from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Any, Callable

import requests

from common import get_logger


class ElectricityApiClient:
    """Thin client for the electricity puzzle endpoints.

    Three operations:
      * ``fetch_board_png(reset=False)`` — GET the current puzzle PNG.
        When ``reset`` is True, appends ``?reset=1`` to the image URL to
        ask the hub to reset the puzzle to its initial state.
      * ``fetch_target_png()`` — GET the solved / target puzzle PNG
        (static reference image used to derive the desired end state).
      * ``rotate(cell)`` — POST a single 90° clockwise rotation for the
        given cell (format ``AxB`` where ``A,B ∈ {1,2,3}``).

    All operations reuse the same 503/429 backoff pattern used by
    :class:`assignments.lesson5_1.railway_api_client.RailwayApiClient` so
    transient hub errors and rate limits are handled uniformly.
    """

    def __init__(
        self,
        api_key: str,
        verify_url: str,
        image_url: str,
        target_image_url: str,
        task_name: str = "electricity",
        max_retries: int = 10,
        base_backoff_s: float = 2.0,
        request_timeout_s: float = 30.0,
    ) -> None:
        self._api_key = api_key
        self._verify_url = verify_url
        self._image_url = image_url.replace("{apikey}", api_key)
        self._target_image_url = target_image_url.replace("{apikey}", api_key)
        self._task_name = task_name
        self._max_retries = max_retries
        self._base_backoff_s = base_backoff_s
        self._request_timeout_s = request_timeout_s
        self.log = get_logger(__name__)

    def fetch_board_png(self, reset: bool = False) -> bytes:
        """Fetch the current puzzle PNG as raw bytes.

        If ``reset`` is True, request the hub to reset the puzzle first
        by appending ``?reset=1`` to the image URL.

        Raises ``requests.HTTPError`` if the final response (after retries)
        is not successful, so the caller can surface a fatal error.
        """
        url = self._image_url
        if reset:
            sep = "&" if "?" in url else "?"
            url = f"{url}{sep}reset=1"

        label = f"fetch_board_png reset={reset}"
        response = self._with_retry(
            label=label,
            make_request=lambda: requests.get(url, timeout=self._request_timeout_s),
        )
        response.raise_for_status()
        return response.content

    def fetch_target_png(self) -> bytes:
        """Fetch the solved / target puzzle PNG as raw bytes.

        The target image is immutable, so callers are expected to cache the
        result. Raises ``requests.HTTPError`` on final failure.
        """
        label = "fetch_target_png"
        response = self._with_retry(
            label=label,
            make_request=lambda: requests.get(
                self._target_image_url, timeout=self._request_timeout_s
            ),
        )
        response.raise_for_status()
        return response.content

    def rotate(self, cell: str) -> dict[str, Any]:
        """POST a single 90° CW rotation for ``cell`` to the hub verify URL.

        Returns a structured dict mirroring ``RailwayApiClient.call``:
            {
                "ok": bool,
                "status": int,
                "body": dict | str,
                "rate_limit": {"remaining": int|None, "reset_at": str|None},
            }
        """
        payload = {
            "apikey": self._api_key,
            "task": self._task_name,
            "answer": {"rotate": cell},
        }

        label = f"rotate cell={cell}"
        response = self._with_retry(
            label=label,
            make_request=lambda: requests.post(
                self._verify_url, json=payload, timeout=self._request_timeout_s
            ),
        )

        remaining = self._parse_int_header(response, "X-RateLimit-Remaining")
        reset_at = self._parse_reset_header(response)
        try:
            body: dict | str = response.json()
        except ValueError:
            body = response.text

        return {
            "ok": response.ok,
            "status": response.status_code,
            "body": body,
            "rate_limit": {"remaining": remaining, "reset_at": reset_at},
        }

    def _with_retry(
        self,
        label: str,
        make_request: Callable[[], requests.Response],
    ) -> requests.Response:
        """Run ``make_request`` with 503/429 backoff, returning the final response.

        On 503: exponential backoff (capped at 60s).
        On 429: sleep until ``X-RateLimit-Reset`` (or 30s fallback).
        On any other status (including 2xx/4xx/5xx besides 503/429): return
        the response unchanged — the caller decides how to handle it.
        If retries are exhausted, the last response is returned as-is.
        """
        backoff = self._base_backoff_s
        last_response: requests.Response | None = None

        for attempt in range(self._max_retries + 1):
            response = make_request()
            last_response = response

            remaining = self._parse_int_header(response, "X-RateLimit-Remaining")
            reset_at = self._parse_reset_header(response)

            self.log.info(
                "%s status=%d remaining=%s reset_at=%s",
                label,
                response.status_code,
                remaining,
                reset_at,
            )

            if response.status_code == 503:
                if attempt >= self._max_retries:
                    self.log.error(
                        "%s 503 retries exhausted after %d attempts", label, attempt + 1
                    )
                    break
                sleep_s = min(backoff, 60.0)
                self.log.warning(
                    "%s 503 attempt=%d sleeping=%.1fs", label, attempt + 1, sleep_s
                )
                time.sleep(sleep_s)
                backoff *= 2
                continue

            if response.status_code == 429:
                if attempt >= self._max_retries:
                    self.log.error(
                        "%s 429 retries exhausted after %d attempts", label, attempt + 1
                    )
                    break
                sleep_s = self._rate_limit_sleep_s(remaining, reset_at)
                self.log.warning(
                    "%s 429 attempt=%d sleeping=%.1fs", label, attempt + 1, sleep_s
                )
                time.sleep(sleep_s)
                continue

            return response

        assert last_response is not None  # loop always runs at least once
        return last_response

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
        try:
            ts = int(value)
            return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
        except (ValueError, TypeError):
            return value

    def _rate_limit_sleep_s(self, remaining: int | None, reset_at: str | None) -> float:
        """Return seconds to sleep before retrying a rate-limited request."""
        if reset_at is not None:
            try:
                reset_dt = datetime.fromisoformat(reset_at)
                now = datetime.now(tz=timezone.utc)
                return max(1.0, (reset_dt - now).total_seconds() + 0.5)
            except ValueError:
                self.log.warning(
                    "Could not parse reset_at=%s, using default backoff", reset_at
                )
        return 30.0
