from __future__ import annotations

import json
from abc import ABC, abstractmethod
from time import time
from typing import Callable

from common import LLMService, AssignmentService, get_logger, setup_logging
from common.events import (
    AgentEventEmitter,
    AgentError,
    EventContext,
    ToolCallCompleted,
    ToolCallStarted,
)


class Assignment(ABC):
    def __init__(self, name: str, description: str) -> None:
        self.name = name
        self.description = description
        self.llm = LLMService()
        self.assignment = AssignmentService()
        self.log = get_logger(__name__)
        setup_logging()
        self._emitter: AgentEventEmitter | None = None

    @abstractmethod
    def solve(self):
        pass

    def _process_tool_calls(
        self,
        messages: list[dict],
        tool_calls: list,
        ctx: EventContext,
        step: int,
        execute_tool: Callable[[str, dict], str],
    ) -> bool:
        """Process tool calls and append results to messages.

        Returns True if any tool result carries ``"fatal": true``, signalling
        that the agent loop should stop immediately.
        """
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

        emitter = self._emitter
        fatal = False

        for tc in tool_calls:
            try:
                args = json.loads(tc.function.arguments)
            except json.JSONDecodeError as exc:
                args = {}
                if emitter:
                    emitter.emit(AgentError(
                        type="agent.error",
                        ctx=ctx,
                        error_type="json_decode",
                        message=str(exc),
                        step=step,
                        tool_name=tc.function.name,
                    ))

            if emitter:
                emitter.emit(ToolCallStarted(
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
                if emitter:
                    emitter.emit(AgentError(
                        type="agent.error",
                        ctx=ctx,
                        error_type="tool_dispatch",
                        message=str(exc),
                        step=step,
                        tool_name=tc.function.name,
                    ))

            if emitter:
                emitter.emit(ToolCallCompleted(
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

            try:
                parsed = json.loads(tool_result)
                if parsed.get("fatal"):
                    self.log.error(
                        "fatal tool result at step=%d tool=%s — stopping agent loop",
                        step, tc.function.name,
                    )
                    fatal = True
            except (json.JSONDecodeError, AttributeError):
                pass

        return fatal

