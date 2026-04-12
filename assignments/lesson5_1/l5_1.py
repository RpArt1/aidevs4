from __future__ import annotations

import json
from pathlib import Path
from time import time
from uuid import uuid4

from dotenv import load_dotenv

load_dotenv()
load_dotenv(Path(__file__).parent / "secrets.env")

from assignments.assignment import Assignment
from assignments.lesson5_1.l5_tools import TOOLS, execute_tool
from common.llm_service import LLMService
from common.events import (
    AgentEventEmitter,
    EventContext,
    AgentStarted,
    AgentCompleted,
    AgentError,
    IterationLimitReached,
    GenerationCompleted,
    ToolCallStarted,
    ToolCallCompleted,
    subscribe_event_logger,
    LangfuseSubscriber,
)

MAX_ITERATIONS = 20
_SYSTEM_PROMPT = (Path(__file__).parent / "system_prompt.md").read_text()


class Lesson5_1(Assignment):
    def __init__(self) -> None:
        super().__init__("Lesson 5_1", "railway")
        self.llm = LLMService(model="openai/gpt-4o-mini")

        self._emitter = AgentEventEmitter()
        subscribe_event_logger(self._emitter)
        self._langfuse = LangfuseSubscriber(tags=["l5_1"])
        self._langfuse.attach(self._emitter)

    def solve(self) -> str:
        t_start = time()
        ctx = EventContext(
            trace_id=str(uuid4()),
            session_id=self.name,
            agent_id="lesson5_1",
            root_agent_id="lesson5_1",
            depth=0,
            timestamp=t_start * 1000,
        )

        self._emitter.emit(AgentStarted(type="agent.started", ctx=ctx))

        messages: list[dict] = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": "Activate railway route X-01."},
        ]
        result = ""

        for step in range(1, MAX_ITERATIONS + 1):
            self.log.info("step=%d", step)
            t0 = time()
            msg = self.llm.chat_with_tools(messages=messages, tools=TOOLS)
            duration_ms = (time() - t0) * 1000

            self._emitter.emit(GenerationCompleted(
                type="generation.completed",
                ctx=ctx,
                step=step,
                output=msg.content,
                model=self.llm.model,
                input=messages,
                input_tokens=self.llm.last_usage.input_tokens,
                output_tokens=self.llm.last_usage.output_tokens,
                duration_ms=duration_ms,
            ))

            if not msg.tool_calls:
                self.log.info("agent done step=%d result=%s", step, msg.content)
                result = msg.content or ""
                break

            self._process_tool_calls(messages, msg.tool_calls, ctx, step)
        else:
            self.log.warning("hit MAX_ITERATIONS=%d", MAX_ITERATIONS)
            self._emitter.emit(IterationLimitReached(
                type="agent.iteration_limit",
                ctx=ctx,
                max_iterations=MAX_ITERATIONS,
                step=MAX_ITERATIONS,
            ))

        self._emitter.emit(AgentCompleted(
            type="agent.completed",
            ctx=ctx,
            duration_ms=(time() - t_start) * 1000,
            result=result or None,
        ))
        self._langfuse.flush()
        return result

    def _process_tool_calls(
        self,
        messages: list[dict],
        tool_calls: list,
        ctx: EventContext,
        step: int,
    ) -> None:
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
            except json.JSONDecodeError as exc:
                args = {}
                self._emitter.emit(AgentError(
                    type="agent.error",
                    ctx=ctx,
                    error_type="json_decode",
                    message=str(exc),
                    step=step,
                    tool_name=tc.function.name,
                ))

            self._emitter.emit(ToolCallStarted(
                type="tool.started",
                ctx=ctx,
                call_id=tc.id,
                tool_name=tc.function.name,
                arguments=args,
                step=step,
            ))

            t0 = time()
            try:
                tool_result = execute_tool(tc.function.name, args)
                success = True
            except Exception as exc:
                tool_result = json.dumps({"error": str(exc)})
                success = False
                self._emitter.emit(AgentError(
                    type="agent.error",
                    ctx=ctx,
                    error_type="tool_dispatch",
                    message=str(exc),
                    step=step,
                    tool_name=tc.function.name,
                ))

            self._emitter.emit(ToolCallCompleted(
                type="tool.completed",
                ctx=ctx,
                call_id=tc.id,
                tool_name=tc.function.name,
                result=tool_result,
                duration_ms=(time() - t0) * 1000,
                success=success,
                step=step,
            ))

            messages.append({"role": "tool", "tool_call_id": tc.id, "content": tool_result})


if __name__ == "__main__":
    lesson = Lesson5_1()
    result = lesson.solve()
    print(result)
