"""Tool definitions and executor for the lesson 3 package proxy assistant."""

from __future__ import annotations

import json
import os

import httpx
from dotenv import load_dotenv

from common import get_logger

load_dotenv()

log = get_logger(__name__)

PACKAGES_API_URL = os.getenv("AIDEVS_PACKAGES_API_URL", "")

TOOLS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "check_package",
            "description": "Check the current status and location of a package.",
            "strict": True,
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
            "strict": True,
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

def _post(payload: dict) -> dict:
    """Send a POST request to the packages API and return parsed JSON."""
    with httpx.Client(timeout=15.0) as client:
        response = client.post(PACKAGES_API_URL, json=payload)
        response.raise_for_status()
        return response.json()


def check_package(packageid: str) -> str:
    """Check the status and location of a package. Returns JSON string."""
    log.info("check_package called packageid=%s", packageid)
    api_key = os.getenv("AIDEVS_API_KEY")
    if not api_key:
        log.error("check_package failed: AIDEVS_API_KEY not configured")
        return json.dumps({
            "error": "AIDEVS_API_KEY not configured",
            "recoveryHints": "The API key is missing from the environment. The operation cannot proceed without it.",
        })

    payload = {"apikey": api_key, "action": "check", "packageid": packageid}
    try:
        result = _post(payload)
    except httpx.HTTPStatusError as e:
        log.error(
            "check_package HTTP error packageid=%s status=%s",
            packageid, e.response.status_code,
        )
        return json.dumps({
            "error": f"API error {e.response.status_code}",
            "recoveryHints": (
                "The packages API returned an error. "
                "Verify the packageid format (e.g. PKG12345678) and ask the operator to confirm it."
            ),
        })
    except Exception as e:
        log.error("check_package failed packageid=%s error=%s", packageid, e)
        return json.dumps({
            "error": str(e),
            "recoveryHints": (
                "An unexpected error occurred while checking the package. "
                "Ask the operator to re-confirm the package ID and try again."
            ),
        })

    log.info("check_package success packageid=%s result=%s", packageid, result)
    result["hints"] = (
        "Package status retrieved. "
        "If the package needs to be redirected, call redirect_package with the packageid, "
        "the destination facility code, and the security code provided by the operator."
    )
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
        return json.dumps({
            "error": "AIDEVS_API_KEY not configured",
            "recoveryHints": "The API key is missing from the environment. The operation cannot proceed without it.",
        })

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
        return json.dumps({
            "error": f"API error {e.response.status_code}",
            "recoveryHints": (
                "The redirect request was rejected by the API. "
                "Check that the security code is exactly as provided by the operator "
                "and that the destination facility code is valid."
            ),
        })
    except Exception as e:
        log.error(
            "redirect_package failed packageid=%s destination=%s error=%s",
            packageid, destination, e,
        )
        return json.dumps({
            "error": str(e),
            "recoveryHints": (
                "An unexpected error occurred during redirection. "
                "Verify the packageid, destination code, and security code, then try again."
            ),
        })

    log.info(
        "redirect_package success packageid=%s destination=%s result=%s",
        packageid, destination, result,
    )
    result["hints"] = (
        "Package successfully redirected. "
        "A confirmation code has been returned — relay it to the operator as proof of the completed redirect."
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
