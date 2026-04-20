from __future__ import annotations

from pathlib import Path
from time import time
from uuid import uuid4

from dotenv import load_dotenv

load_dotenv()
load_dotenv(Path(__file__).parent / "secrets.env")

from assignments.assignment import Assignment
from assignments.lesson2_2.l2_2_tools import TOOLS, execute_tool
from common.llm_service import LLMService
from common.events import (
    AgentEventEmitter,
    EventContext,
    AgentStarted,
    AgentCompleted,
    IterationLimitReached,
    GenerationCompleted,
    subscribe_event_logger,
    LangfuseSubscriber,
)

MAX_ITERATIONS = 15
MAX_ROTATIONS = 20
_SYSTEM_PROMPT = (Path(__file__).parent / "system_prompt.md").read_text().format(
    MAX_ROTATIONS=MAX_ROTATIONS,
)


class Lesson2_2(Assignment):
    def __init__(self) -> None:
        super().__init__("Lesson 2_2", "lesson2_2")
        self.llm = LLMService(model="anthropic/claude-sonnet-4-6")

        self._emitter = AgentEventEmitter()
        subscribe_event_logger(self._emitter)
        self._langfuse = LangfuseSubscriber(tags=["l2_2"])
        self._langfuse.attach(self._emitter)

    def solve(self) -> str:
        t_start = time()
        ctx = EventContext(
            trace_id=str(uuid4()),
            session_id=self.name,
            agent_id="lesson2_2",
            root_agent_id="lesson2_2",
            depth=0,
            timestamp=t_start * 1000,
        )

        self._emitter.emit(AgentStarted(type="agent.started", ctx=ctx))

        messages: list[dict] = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": "Complete the task."},
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

            fatal = self._process_tool_calls(messages, msg.tool_calls, ctx, step, execute_tool)
            if fatal:
                self.log.error("fatal tool error — aborting agent loop at step=%d", step)
                result = "Task aborted: a fatal, non-recoverable error occurred (unreachable URL or similar). Check logs for details."
                break
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



if __name__ == "__main__":
    lesson = Lesson2_2()
    result = lesson.solve()
    print(result)
