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
                    "Names of environment variables the Solver's generated code will need. "
                    "Use OPENROUTER_API_KEY (never OPENAI_API_KEY) for LLM calls. "
                    "Other common values: AIDEVS_API_KEY, AIDEVS_VERIFY_URL, PUBLIC_WEBHOOK_URL."
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
                    "Extract the complete answer JSON example directly from the task text's "
                    "code block — every nested field, every key name, exactly as written. "
                    "Do NOT translate, rename, simplify, or invent any part of the structure. "
                    "Encode the extracted object as a compact JSON string."
                ),
            },
            "preflight_checks": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "check_type": {
                            "type": "string",
                            "enum": ["env_var", "url_reachable", "local_file"],
                            "description": "Category of the check.",
                        },
                        "target": {
                            "type": "string",
                            "description": "The var name, URL, or file path to verify.",
                        },
                        "description": {
                            "type": "string",
                            "description": "Why this resource is mandatory for the task.",
                        },
                    },
                    "required": ["check_type", "target", "description"],
                    "additionalProperties": False,
                },
                "description": (
                    "Access checks the Solver MUST run first. Each entry maps to a "
                    "required_env var (env_var), input_data URL (url_reachable), or "
                    "local file (local_file). If any check fails the whole run aborts."
                ),
            },
            "steps": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "action": {
                            "type": "string",
                            "description": (
                                "Specific, verb-led description of what to do and "
                                "broadly how (e.g. 'Download the CSV from input_data[0] "
                                "and parse it with pandas')."
                            ),
                        },
                        "output_artifact": {
                            "type": "string",
                            "description": (
                                "Workspace-relative filename this step writes "
                                "(e.g. 'people_raw.csv'), or 'none' for steps "
                                "that produce no file (e.g. final submission)."
                            ),
                        },
                    },
                    "required": ["action", "output_artifact"],
                    "additionalProperties": False,
                },
                "description": (
                    "Ordered execution plan for the Solver. Each step must declare "
                    "what it does (action) and what file it writes (output_artifact) "
                    "so data flow between steps is explicit."
                ),
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
            "preflight_checks",
            "steps",
            "success_check",
        ],
        "additionalProperties": False,
    },
}
