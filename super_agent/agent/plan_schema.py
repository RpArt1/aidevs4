"""JSON schema for planner structured output (``LLMService.chat_structured``).

Mirrors ``super_agent/prompts/planner.md`` field names and shapes.
"""

from __future__ import annotations

from typing import Any

VALID_TASK_FAMILIES = ("data_structured", "tool_react", "long_running_webhook")

PLAN_SCHEMA: dict[str, Any] = {
    "name": "task_plan",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "goal": {
                "type": "string",
                "description": "One-sentence restatement of what the task asks for.",
            },
            "task_family": {
                "type": "string",
                "enum": list(VALID_TASK_FAMILIES),
                "description": (
                    "Coarse-grained shape: 'data_structured' for static-data + "
                    "structured-LLM tasks, 'tool_react' for tool-using ReAct "
                    "tasks, 'long_running_webhook' for tasks where the aidevs "
                    "server calls back into a server we expose."
                ),
            },
            "verify_task_name": {
                "type": "string",
                "description": (
                    "Short slug used by AssignmentService.send(task, answer); "
                    "should be extracted from the task text (e.g. 'nazwa zadania to: \"people\"')."
                ),
            },
            "required_env": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "Names of environment variables the Solver's generated "
                    "code will need (e.g. AIDEVS_API_KEY, PUBLIC_WEBHOOK_URL)."
                ),
            },
            "input_data": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "source_type": {
                            "type": "string",
                            "enum": ["url", "local_file", "api"],
                            "description": "Where this input comes from.",
                        },
                        "location": {
                            "type": "string",
                            "description": "Path, URL, or API identifier for the input.",
                        },
                        "description": {
                            "type": "string",
                            "description": "What the Solver will find and in what format.",
                        },
                    },
                    "required": ["source_type", "location", "description"],
                    "additionalProperties": False,
                },
                "description": (
                    "Inputs the Solver should expect (URLs, files in the "
                    "workspace, etc.). Empty array if none."
                ),
            },
            "expected_output": {
                "type": "string",
                "description": (
                    "Expected JSON or structure for the verify POST body / answer field "
                    "(from the task example), so the Solver does not guess."
                ),
            },
            "steps": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Natural-language step-by-step approach for the Solver.",
            },
            "hints": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Gotchas / invariants the Solver must respect.",
            },
            "success_check": {
                "type": "string",
                "description": "What the final submission response should look like.",
            },
        },
        "required": [
            "goal",
            "task_family",
            "verify_task_name",
            "required_env",
            "input_data",
            "expected_output",
            "steps",
            "hints",
            "success_check",
        ],
        "additionalProperties": False,
    },
}
