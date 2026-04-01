"""Tool definitions and executors for the lesson 4 SPK declaration agent."""

from __future__ import annotations

import base64
import json

import httpx

from common import get_logger
from common.assignment_service import AssignmentService
from common.llm_service import LLMService

log = get_logger(__name__)
_llm = LLMService(model="openai/gpt-4o")
_assignment = AssignmentService()

tool_fetch_text_from_url = {
    "type": "function",
    "function": {
        "name": "fetch_text_from_url",
        "description": "Fetches decoded text content from a URL.",
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "The URL to fetch text from.",
                },
            },
            "required": ["url"],
            "additionalProperties": False,
        },
    },
}

tool_fetch_image_and_analyze = {
    "type": "function",
    "function": {
        "name": "fetch_image_and_analyze",
        "description": (
            "Downloads an image from a URL and answers a question about it "
            "using a vision-capable model."
        ),
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "The URL of the image to download and analyze.",
                },
                "question": {
                    "type": "string",
                    "description": "The question to answer about the image.",
                },
            },
            "required": ["url", "question"],
            "additionalProperties": False,
        },
    },
}

tool_submit_declaration = {
    "type": "function",
    "function": {
        "name": "submit_declaration",
        "description": (
            "Submits the completed SPK declaration to the hub and returns "
            "the hub response (flag if correct, error message if not)."
        ),
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": {
                "declaration": {
                    "type": "string",
                    "description": "The full, formatted SPK declaration text to submit.",
                },
            },
            "required": ["declaration"],
            "additionalProperties": False,
        },
    },
}

TOOLS = [tool_fetch_text_from_url, tool_fetch_image_and_analyze, tool_submit_declaration]


def fetch_text_from_url(url: str) -> str:
    """Fetch decoded text from a URL. Returns the text body or a JSON error string."""
    log.info("fetch_text_from_url called url=%s", url)
    try:
        response = httpx.get(url, follow_redirects=True, timeout=30)
        response.raise_for_status()
    except httpx.HTTPStatusError as e:
        log.error("fetch_text_from_url HTTP error url=%s status=%s", url, e.response.status_code)
        return json.dumps({
            "error": f"HTTP {e.response.status_code}",
            "url": url,
            "recoveryHints": (
                f"The server returned HTTP {e.response.status_code}. "
                "Verify the URL is correct and publicly accessible. "
                "If 404, the resource may have moved — check the base page for updated links. "
                "If 403/401, the resource requires authentication and cannot be fetched directly."
            ),
        })
    except Exception as e:
        log.error("fetch_text_from_url failed url=%s error=%s", url, e)
        return json.dumps({
            "error": str(e),
            "url": url,
            "recoveryHints": (
                "A network error occurred while fetching the URL. "
                "Check that the URL is well-formed and the host is reachable, then retry."
            ),
        })

    text = response.text
    log.info("fetch_text_from_url success url=%s chars=%d", url, len(text))
    return json.dumps({
        "content": text,
        "url": url,
        "chars": len(text),
        "hints": (
            f"Text content retrieved ({len(text)} characters). "
            "Read and analyse it to extract the information needed for the declaration. "
            "If the page contains links to additional resources (images, annexes), "
            "fetch them separately with the appropriate tool."
        ),
    }, ensure_ascii=False)


def fetch_image_and_analyze(url: str, question: str) -> str:
    """Download an image and answer a question about it using a vision-capable model."""
    log.info("fetch_image_and_analyze called url=%s question=%s", url, question)
    try:
        response = httpx.get(url, follow_redirects=True, timeout=30)
        response.raise_for_status()
    except httpx.HTTPStatusError as e:
        log.error("fetch_image_and_analyze HTTP error url=%s status=%s", url, e.response.status_code)
        return json.dumps({
            "error": f"HTTP {e.response.status_code}",
            "url": url,
            "recoveryHints": (
                f"The image could not be downloaded — server returned HTTP {e.response.status_code}. "
                "Verify the image URL is correct and publicly accessible. "
                "If the URL was extracted from a page, re-fetch that page to confirm the current link."
            ),
        })
    except Exception as e:
        log.error("fetch_image_and_analyze download failed url=%s error=%s", url, e)
        return json.dumps({
            "error": str(e),
            "url": url,
            "recoveryHints": (
                "A network error occurred while downloading the image. "
                "Check the URL is well-formed and the host is reachable, then retry."
            ),
        })

    b64 = base64.b64encode(response.content).decode()
    content_type = response.headers.get("content-type", "image/png").split(";")[0].strip()
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": f"data:{content_type};base64,{b64}"}},
                {"type": "text", "text": question},
            ],
        }
    ]
    try:
        answer = _llm.chat(messages)
    except Exception as e:
        log.error("fetch_image_and_analyze vision call failed url=%s error=%s", url, e)
        return json.dumps({
            "error": str(e),
            "url": url,
            "recoveryHints": (
                "The image was downloaded but the vision model failed to analyse it. "
                "Try rephrasing the question to be more specific, or retry once."
            ),
        })

    log.info("fetch_image_and_analyze success url=%s answer=%s", url, answer)
    return json.dumps({
        "answer": answer,
        "url": url,
        "hints": (
            "Image analysed successfully. "
            "Use the answer to fill in the relevant field(s) of the SPK declaration. "
            "If the answer is ambiguous or incomplete, call this tool again with a more specific question."
        ),
    }, ensure_ascii=False)


def submit_declaration(declaration: str) -> str:
    """Submit the completed SPK declaration to the hub. Returns hub response as JSON string."""
    log.info("submit_declaration called declaration=%s", declaration)
    try:
        result = _assignment.send("sendit", {"declaration": declaration})
    except Exception as e:
        log.error("submit_declaration failed error=%s", e)
        return json.dumps({
            "error": str(e),
            "recoveryHints": (
                "The declaration could not be submitted due to a network or service error. "
                "Do NOT modify the declaration content — retry the submission as-is. "
                "If the error persists, verify that the AssignmentService is reachable."
            ),
        })

    log.info("submit_declaration response=%s", result)
    result["hints"] = (
        "Declaration submitted to the hub. "
        "If the response contains a flag, the task is complete — report it to the user. "
        "If the response contains a validation error, correct only the specific field mentioned "
        "and resubmit."
    )
    return json.dumps(result, ensure_ascii=False)

_REGISTRY: dict[str, callable] = {
    "fetch_text_from_url": fetch_text_from_url,
    "fetch_image_and_analyze": fetch_image_and_analyze,
    "submit_declaration": submit_declaration,
}


def execute_tool(name: str, arguments: dict) -> str:
    """Dispatch a tool call by name. Returns JSON string result."""
    log.info("execute_tool dispatching name=%s arguments=%s", name, arguments)
    handler = _REGISTRY.get(name)
    if handler is None:
        log.warning("execute_tool unknown tool name=%s", name)
        available = ", ".join(_REGISTRY.keys())
        return json.dumps({
            "error": f"Unknown tool: {name}",
            "recoveryHints": f"Available tools are: {available}. Use one of these exact names.",
        })
    return handler(**arguments)