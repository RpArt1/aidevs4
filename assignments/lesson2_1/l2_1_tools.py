"""Tool schemas, handler implementations, and executor for the lesson 2_1 classification agent."""

from __future__ import annotations

import csv
import io
import json
import os
from pathlib import Path

import requests
from dotenv import load_dotenv

from common import get_logger

load_dotenv()
load_dotenv(Path(__file__).parent / "secrets.env")

log = get_logger(__name__)

_api_key = os.environ["AIDEVS_API_KEY"]
_verify_url = os.environ["AIDEVS_VERIFY_URL"]
_items_url = os.environ["ITEMS_URL"]
_task_name = os.environ["CLASSIFICATION_TASK"]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _fetch_items() -> list[dict]:
    """Fetch the items list from the hub and return [{id, description}, ...]."""
    resp = requests.get(_items_url, timeout=30)
    log.info("fetch_items response status=%d", resp.status_code)
    log.info("url=%s", _items_url)
    log.info("response=%s", resp.text)
    resp.raise_for_status()
    content_type = resp.headers.get("content-type", "")
    if "json" in content_type:
        data = resp.json()
        if isinstance(data, list):
            return [{"id": str(item["id"]), "description": str(item["description"])} for item in data]
        return data
    reader = csv.DictReader(io.StringIO(resp.text))
    id_key = "id" if "id" in reader.fieldnames else "code"
    return [{"id": str(row[id_key]), "description": str(row["description"])} for row in reader]


def _post_to_hub(answer: dict | str) -> requests.Response:
    """POST a single answer payload to the hub and return the response."""
    payload = {"apikey": _api_key, "task": _task_name, "answer": answer}
    resp = requests.post(_verify_url, json=payload, timeout=30)
    return resp


# ── Tool implementations ──────────────────────────────────────────────────────

def preview_items() -> str:
    """Fetch and return the current item list without triggering a classification cycle."""
    log.info("preview_items called")
    try:
        items = _fetch_items()
        result: dict = {
            "items": items,
            "count": len(items),
            "hints": (
                "Items fetched. Design a prompt_template that includes {id} and {description} "
                "placeholders (≤100 tokens total) and call run_classification_cycle to test it. "
                "Remember: reactor-related items must always be labelled NEU."
            ),
        }
        log.info("preview_items count=%d", len(items))
        return json.dumps(result, ensure_ascii=False)
    except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as exc:
        log.error("preview_items fatal network error: %s", exc)
        return json.dumps({
            "error": str(exc),
            "fatal": True,
            "recoveryHints": (
                "The items URL is unreachable (connection/timeout error). "
                "This is a fatal, non-recoverable error — do NOT retry. "
                "Stop immediately and report the failure to the user."
            ),
        })
    except Exception as exc:
        log.error("preview_items error: %s", exc)
        return json.dumps({
            "error": str(exc),
            "fatal": True,
            "recoveryHints": (
                "Failed to fetch items — this error cannot be resolved by retrying. "
                "Stop immediately and report the failure to the user."
            ),
        })


def run_classification_cycle(prompt_template: str) -> str:
    """Reset the hub budget, then submit each item's filled prompt to the hub for classification."""
    log.info("run_classification_cycle template=%r", prompt_template)

    if "{id}" not in prompt_template or "{description}" not in prompt_template:
        return json.dumps({
            "error": "prompt_template must contain both {id} and {description} placeholders",
            "recoveryHints": "Add {id} and {description} to your prompt and retry.",
        })

    # Step 1: reset the hub budget
    try:
        reset_resp = _post_to_hub({"prompt": "reset"})
        log.info("reset response status=%d body=%.200s", reset_resp.status_code, reset_resp.text[:200])
    except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as exc:
        log.error("run_classification_cycle reset fatal network error: %s", exc)
        return json.dumps({
            "error": f"Failed to reset hub budget: {exc}",
            "fatal": True,
            "recoveryHints": (
                "The hub URL is unreachable (connection/timeout error). "
                "This is a fatal, non-recoverable error — do NOT retry. "
                "Stop immediately and report the failure to the user."
            ),
        })
    except Exception as exc:
        log.error("run_classification_cycle reset error: %s", exc)
        return json.dumps({
            "error": f"Failed to reset hub budget: {exc}",
            "fatal": True,
            "recoveryHints": (
                "Failed to reset hub budget — this error cannot be resolved by retrying. "
                "Stop immediately and report the failure to the user."
            ),
        })

    # Step 2: fetch fresh items
    try:
        items = _fetch_items()
    except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as exc:
        log.error("run_classification_cycle fetch fatal network error: %s", exc)
        return json.dumps({
            "error": f"Failed to fetch items: {exc}",
            "fatal": True,
            "recoveryHints": (
                "The items URL is unreachable (connection/timeout error). "
                "This is a fatal, non-recoverable error — do NOT retry. "
                "Stop immediately and report the failure to the user."
            ),
        })
    except Exception as exc:
        log.error("run_classification_cycle fetch error: %s", exc)
        return json.dumps({
            "error": f"Failed to fetch items: {exc}",
            "fatal": True,
            "recoveryHints": (
                "Failed to fetch items — this error cannot be resolved by retrying. "
                "Stop immediately and report the failure to the user."
            ),
        })

    # Step 3: submit each item's filled prompt to the hub and collect responses
    item_results: list[dict] = []
    all_response_text = ""
    for item in items:
        filled = prompt_template.replace("{id}", item["id"]).replace("{description}", item["description"])
        try:
            resp = _post_to_hub({"prompt": filled})
            hub_body = resp.json()
            body_text = json.dumps(hub_body)
            all_response_text += body_text
            log.info("item id=%s status=%d body=%.200s", item["id"], resp.status_code, body_text[:200])
            item_results.append({
                "id": item["id"],
                "prompt": filled,
                "hub_response": hub_body,
                "http_status": resp.status_code,
            })
        except Exception as exc:
            log.error("run_classification_cycle item id=%s error: %s", item["id"], exc)
            item_results.append({
                "id": item["id"],
                "prompt": filled,
                "error": str(exc),
            })

    result: dict = {"item_results": item_results}

    if "FLG:" in all_response_text:
        result["hints"] = "The hub returned a flag — task complete! Report it to the user."
    else:
        result["hints"] = (
            "All items submitted. Inspect each hub_response to see how the hub classified each prompt. "
            "Adjust prompt_template to fix any mis-classifications and call run_classification_cycle again."
        )

    return json.dumps(result, ensure_ascii=False)


# ── Tool schemas ──────────────────────────────────────────────────────────────

tool_preview_items = {
    "type": "function",
    "function": {
        "name": "preview_items",
        "description": "Preview the items in the inventory (free — no budget cost)",
        "parameters": {
            "type": "object",
            "properties": {},
            "additionalProperties": False,
        },
    },
}

tool_run_classification_cycle = {
    "type": "function",
    "function": {
        "name": "run_classification_cycle",
        "description": (
            "Reset the hub budget, fetch a fresh item list, then POST each item's filled prompt "
            "to the hub — the hub runs its own model and returns DNG or NEU per item. "
            "Returns all hub responses so you can evaluate the prompt's accuracy."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "prompt_template": {
                    "type": "string",
                    "description": (
                        "Classification prompt with {id} and {description} placeholders. "
                        "Must fit in ≤100 tokens. Place static instructions before the placeholders."
                    ),
                },
            },
            "required": ["prompt_template"],
            "additionalProperties": False,
        },
    },
}

TOOLS: list[dict] = [tool_preview_items, tool_run_classification_cycle]

_REGISTRY: dict[str, callable] = {
    "preview_items": preview_items,
    "run_classification_cycle": run_classification_cycle,
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
