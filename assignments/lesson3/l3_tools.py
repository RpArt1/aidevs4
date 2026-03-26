"""Tool definitions and executor for the lesson 3 package proxy assistant."""

from __future__ import annotations

import json
import os

import httpx
from dotenv import load_dotenv

from common import get_logger

load_dotenv()

log = get_logger(__name__)

PACKAGES_API_URL = "AIDEVS_PACKAGES_API_URL_ENV"

TOOLS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "check_package",
            "description": "Check the current status and location of a package.",
            "parameters": {
                "type": "object",
                "properties": {
                    "packageid": {
                        "type": "string",
                        "description": "The package identifier, e.g. PKG12345678",
                    }
                },
                "required": ["packageid"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "redirect_package",
            "description": (
                "Redirect a package to a new destination facility. "
                "Returns a confirmation code on success."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "packageid": {
                        "type": "string",
                        "description": "The package identifier",
                    },
                    "destination": {
                        "type": "string",
                        "description": "Destination facility code",
                    },
                    "code": {
                        "type": "string",
                        "description": "Security code provided by the operator",
                    },
                },
                "required": ["packageid", "destination", "code"],
                "additionalProperties": False,
            },
        },
    },
]


def _api_key() -> str | None:
    return os.getenv("AIDEVS_API_KEY")


def _post(payload: dict) -> dict:
    """Send a POST request to the packages API and return parsed JSON."""
    with httpx.Client(timeout=15.0) as client:
        response = client.post(PACKAGES_API_URL, json=payload)
        response.raise_for_status()
        return response.json()


def check_package(packageid: str) -> str:
    """Check the status and location of a package. Returns JSON string."""
    log.info("check_package called packageid=%s", packageid)
    api_key = _api_key()
    if not api_key:
        log.error("check_package failed: AIDEVS_API_KEY not configured")
        return json.dumps({"error": "AIDEVS_API_KEY not configured"})

    payload = {"apikey": api_key, "action": "check", "packageid": packageid}
    try:
        result = _post(payload)
    except httpx.HTTPStatusError as e:
        log.error(
            "check_package HTTP error packageid=%s status=%s",
            packageid, e.response.status_code,
        )
        return json.dumps({"error": f"API error {e.response.status_code}"})
    except Exception as e:
        log.error("check_package failed packageid=%s error=%s", packageid, e)
        return json.dumps({"error": str(e)})

    log.info("check_package success packageid=%s result=%s", packageid, result)
    return json.dumps(result, ensure_ascii=False)


def redirect_package(packageid: str, destination: str, code: str) -> str:
    """Redirect a package to a new destination. Returns JSON string with confirmation."""
    log.info(
        "redirect_package called packageid=%s destination=%s code=%s",
        packageid, destination, code,
    )
    api_key = _api_key()
    if not api_key:
        log.error("redirect_package failed: AIDEVS_API_KEY not configured")
        return json.dumps({"error": "AIDEVS_API_KEY not configured"})

    payload = {
        "apikey": api_key,
        "action": "redirect",
        "packageid": packageid,
        "destination": destination,
        "code": code,
    }
    try:
        result = _post(payload)
    except httpx.HTTPStatusError as e:
        log.error(
            "redirect_package HTTP error packageid=%s destination=%s status=%s",
            packageid, destination, e.response.status_code,
        )
        return json.dumps({"error": f"API error {e.response.status_code}"})
    except Exception as e:
        log.error(
            "redirect_package failed packageid=%s destination=%s error=%s",
            packageid, destination, e,
        )
        return json.dumps({"error": str(e)})

    log.info(
        "redirect_package success packageid=%s destination=%s result=%s",
        packageid, destination, result,
    )
    return json.dumps(result, ensure_ascii=False)


_REGISTRY: dict[str, callable] = {
    "check_package": lambda args: check_package(args["packageid"]),
    "redirect_package": lambda args: redirect_package(
        args["packageid"], args["destination"], args["code"]
    ),
}


def execute_tool(name: str, arguments: dict) -> str:
    """Dispatch a tool call by name. Returns JSON string result."""
    log.info("execute_tool dispatching name=%s arguments=%s", name, arguments)
    handler = _REGISTRY.get(name)
    if handler is None:
        log.warning("execute_tool unknown tool name=%s", name)
        return json.dumps({"error": f"Unknown tool: {name}"})
    return handler(arguments)
