from __future__ import annotations

import json
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()
load_dotenv(Path(__file__).parent / "secrets.env")

from assignments.lesson4.l4_tools import TOOLS, execute_tool
from common import get_logger
from common.llm_service import LLMService

log = get_logger(__name__)
llm = LLMService(model="openai/gpt-4o")

MAX_ITERATIONS = 10
_SYSTEM_PROMPT = (Path(__file__).parent / "system_prompt.md").read_text().format(
    SPK_DOC_INDEX_URL=os.environ["SPK_DOC_INDEX_URL"],
    SPK_DECLARATION_TEMPLATE_URL=os.environ["SPK_DECLARATION_TEMPLATE_URL"],
    SPK_DECLARATION_TEMPLATE_FILENAME=os.environ["SPK_DECLARATION_TEMPLATE_FILENAME"],
    SPK_ROUTES_IMAGE_URL=os.environ["SPK_ROUTES_IMAGE_URL"],
    SPK_ROUTES_IMAGE_FILENAME=os.environ["SPK_ROUTES_IMAGE_FILENAME"],
)


def _process_tool_calls(messages: list[dict], tool_calls: list) -> None:
    messages.append({
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": tc.id,
                "type": "function",
                "function": {"name": tc.function.name, "arguments": tc.function.arguments},
            }
            for tc in tool_calls
        ],
    })
    for tc in tool_calls:
        try:
            args = json.loads(tc.function.arguments)
        except json.JSONDecodeError:
            args = {}
        result = execute_tool(tc.function.name, args)
        log.info("tool=%s result_len=%d", tc.function.name, len(result))
        messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})


def _run_iteration(messages: list[dict]) -> str | None:
    """Run one LLM step. Returns the final answer string when done, None to continue."""
    assistant_msg = llm.chat_with_tools(messages=messages, tools=TOOLS)
    tool_calls = assistant_msg.tool_calls
    if not tool_calls:
        return assistant_msg.content or ""
    _process_tool_calls(messages, tool_calls)
    return None


def run_agent() -> str:
    messages: list[dict] = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": "Rozpocznij procedurę wypełnienia i wysłania deklaracji SPK."},
    ]
    for step in range(1, MAX_ITERATIONS + 1):
        log.info("run_agent step=%d", step)
        result = _run_iteration(messages)
        if result is not None:
            log.info("run_agent done step=%d result=%s", step, result)
            return result
    log.warning("run_agent hit max_iterations=%d without finishing", MAX_ITERATIONS)
    return ""


def main() -> None:
    result = run_agent()
    print(result)


if __name__ == "__main__":
    main()
