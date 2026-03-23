import json
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from common import LLMService, get_logger
from assignments.assignment import Assignment
from assignments.lesson2.tools import TOOLS, HANDLERS

MAX_ITERATIONS = 10

SYSTEM_PROMPT = """\
You are an investigative agent. Your goal is to find which suspect was spotted \
near a nuclear power plant, determine their access level, and submit the answer.

## Steps
1. Call find_suspect_nearest_power_plant with the full suspects list. It fetches \
power plants, queries each suspect's sightings, and returns the suspect closest \
to any power plant along with the power plant code and name.
2. Call get_access_level for the returned person using their name, surname, and birthYear.
3. Call submit_answer with the suspect's name, surname, accessLevel, and powerPlant code.

## Important
- The power plant code format is like PWR0000PL — use the exact code returned by the tool.
- birthYear must be an integer (e.g. 1987), not a full date string.
- After submitting, report the verification result.
"""


class Lesson2(Assignment):
    def __init__(self):
        self.log = get_logger(__name__)
        self.llm = LLMService(model="openai/gpt-4.1-mini")

    def _load_suspects(self) -> list[dict]:
        suspects_path = Path(__file__).parent.parent / "lesson1" / "filtered_jobs.json"
        raw = json.loads(suspects_path.read_text(encoding="utf-8"))
        if not raw:
            raise ValueError("No suspects found")
        suspects = [
            {
                "name": p["name"],
                "surname": p["surname"],
                "birthYear": int(p["birthDate"][:4]),
            }
            for p in raw
        ]
        # for now limit to 1 suspect
#        return suspects[:1]
        return suspects

    def _execute_tool_calls(self, tool_calls: list) -> list[dict]:
        results: list[dict] = []
        self.log.info(f"Tool calls: {len(tool_calls)}")

        for call in tool_calls:
            name = call.function.name
            args = json.loads(call.function.arguments)
            self.log.info(f"  -> {name}({json.dumps(args, ensure_ascii=False)})")

            try:
                handler = HANDLERS.get(name)
                if handler is None:
                    raise ValueError(f"Unknown tool: {name}")
                result = handler(**args)
                self.log.info(f"     OK")
                output = json.dumps(result, ensure_ascii=False)
            except Exception as exc:
                self.log.error(f"     Error: {exc}")
                output = json.dumps({"error": str(exc)})

            results.append({
                "role": "tool",
                "tool_call_id": call.id,
                "content": output,
            })

        return results

    def solve(self):
        suspects = self._load_suspects()
        self.log.info(f"Loaded {len(suspects)} suspects")

        user_message = (
            "Here is the list of suspects (name, surname, birthYear):\n"
            + json.dumps(suspects, ensure_ascii=False, indent=2)
            + "\n\nPlease investigate which suspect was near a nuclear power plant, "
            "determine their access level, and submit the answer."
        )

        messages: list[dict] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ]

        for iteration in range(1, MAX_ITERATIONS + 1):
            self.log.info(f"--- Agent iteration {iteration}/{MAX_ITERATIONS} ---")

            message = self.llm.chat_with_tools(messages, TOOLS)
            tool_calls = message.tool_calls

            if not tool_calls:
                text = message.content or "(no response)"
                self.log.info(f"Agent final answer:\n{text}")
                return text

            messages.append({
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in tool_calls
                ],
            })
            messages.extend(self._execute_tool_calls(tool_calls))

        self.log.warning("Max iterations reached without a final answer")
        return "Max iterations reached"
