from __future__ import annotations

import json
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()
load_dotenv(Path(__file__).parent / "secrets.env")

from assignments.assignment import Assignment
from assignments.lesson5_1.l5_tools import TOOLS, execute_tool
from common.llm_service import LLMService

MAX_ITERATIONS = 20
_SYSTEM_PROMPT = (Path(__file__).parent / "system_prompt.md").read_text()


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
        messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})


class Lesson5_1(Assignment):
    def __init__(self) -> None:
        super().__init__("Lesson 5_1", "railway")
        self.llm = LLMService(model="openai/gpt-4o-mini")

    def solve(self) -> str:
        messages: list[dict] = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": "Activate railway route X-01."},
        ]
        for step in range(1, MAX_ITERATIONS + 1):
            self.log.info("step=%d", step)
            msg = self.llm.chat_with_tools(messages=messages, tools=TOOLS)
            if not msg.tool_calls:
                self.log.info("agent done step=%d result=%s", step, msg.content)
                return msg.content or ""
            _process_tool_calls(messages, msg.tool_calls)
        self.log.warning("hit MAX_ITERATIONS=%d", MAX_ITERATIONS)
        return ""


if __name__ == "__main__":
    lesson = Lesson5_1()
    result = lesson.solve()
    print(result)
