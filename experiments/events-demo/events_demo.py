import sys
import os
from time import time
from uuid import uuid4

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from dotenv import load_dotenv

from common.llm_service import LLMService
from agent_event_emitter import AgentEventEmitter
from event_types import EventContext, AgentStarted, GenerationCompleted, AgentCompleted
from subscriber import subscribe_event_logger

load_dotenv()


def main() -> None:
    emitter = AgentEventEmitter()
    subscribe_event_logger(emitter)

    ctx = EventContext(
        trace_id=str(uuid4()),
        session_id="s1",
        agent_id="a1",
        root_agent_id="a1",
        depth=0,
        timestamp=time() * 1000,
    )

    emitter.emit(AgentStarted(type="agent.started", ctx=ctx))

    llm = LLMService(model="openai/gpt-4o-mini")
    messages = [{"role": "user", "content": "What is 2+2? Answer in one word."}]

    t0 = time()
    reply = llm.chat(messages)
    duration_ms = (time() - t0) * 1000

    emitter.emit(GenerationCompleted(type="generation.completed", ctx=ctx, output=reply))
    emitter.emit(AgentCompleted(type="agent.completed", ctx=ctx, duration_ms=duration_ms, result=reply))


if __name__ == "__main__":
    main()
