"""Tool schema and executor for the lesson 5_1 railway API agent."""

from __future__ import annotations

import json
import os

from common import get_logger
from assignments.lesson5_1.railway_api_client import RailwayApiClient

log = get_logger(__name__)

_client = RailwayApiClient(
    api_key=os.environ["AIDEVS_API_KEY"],
    verify_url=os.environ["AIDEVS_VERIFY_URL"],
)

tool_call_railway_api = {
    "type": "function",
    "function": {
        "name": "call_railway_api",
        "description": (
            "Call an action on the railway API. "
            "Start with action='help' to get full documentation. "
            "Use only actions and parameter names returned by help."
        ),
        #"strict": True,
        "parameters": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "description": "Action name (e.g. 'help', or any action from the help response)",
                },
                "api_params": {
                    "type": "object",
                    "description": (
                        "Key-value parameters for the API action as documented by help. "
                        "Use exactly the parameter names from the help response. "
                        "Pass {} for actions that need none."
                    ),
                    "additionalProperties": {"type": "string"},
                },
            },
            "required": ["action", "api_params"],
            "additionalProperties": False,
        },
    },
}

TOOLS = [tool_call_railway_api]


def call_railway_api(action: str, api_params: dict | None = None) -> str:
    """Execute an action on the railway API and return a JSON string with hints."""
    api_params = api_params or {}
    log.info("call_railway_api action=%s api_params=%s", action, api_params)
    result = _client.call(action, extra_params=api_params)

    if result["ok"]:
        result["hints"] = (
            "Action succeeded. Read the response body carefully — it may contain "
            "the next required action name and its parameters, or a flag {FLG:...} "
            "indicating the task is complete. Never guess action names or parameters; "
            "always derive them from the API response."
        )
        log.info("call_railway_api success action=%s status=%d", action, result["status"])
    else:
        result["recoveryHints"] = (
            f"The API returned an error (HTTP {result['status']}). "
            "Read the 'body' field — it contains the exact reason for failure. "
            "If the action name is wrong, call action='help' again to see valid actions. "
            "If a parameter is wrong, correct only that parameter and retry. "
            "Do not retry with identical arguments."
        )
        log.error(
            "call_railway_api error action=%s status=%d body=%.200s",
            action, result["status"], result["body"],
        )

    return json.dumps(result, ensure_ascii=False)


_REGISTRY: dict[str, callable] = {
    "call_railway_api": call_railway_api,
}


_CALL_RAILWAY_KNOWN_KEYS = {"action", "api_params", "parameters"}


def execute_tool(name: str, arguments: dict) -> str:
    """Dispatch a tool call by name. Returns a JSON string result."""
    log.info("execute_tool name=%s arguments=%s", name, arguments)
    handler = _REGISTRY.get(name)
    if handler is None:
        log.warning("execute_tool unknown tool name=%s", name)
        available = ", ".join(_REGISTRY.keys())
        return json.dumps({
            "error": f"Unknown tool: {name}",
            "recoveryHints": f"Available tools are: {available}. Use one of these exact names.",
        })
    if name == "call_railway_api":
        # Rescue: if the model flattened API params to the top level (e.g. sent
        # {action, route} instead of {action, api_params: {route}}), collect the
        # stray keys and move them into api_params.
        extra = {k: v for k, v in arguments.items() if k not in _CALL_RAILWAY_KNOWN_KEYS}
        if extra:
            log.warning("execute_tool rescued flattened api_params keys=%s", list(extra))
            base = dict(arguments.get("api_params") or arguments.get("parameters") or {})
            arguments = {"action": arguments["action"], "api_params": {**base, **extra}}
    return handler(**arguments)
