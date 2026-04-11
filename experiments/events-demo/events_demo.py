"""
events_demo.py — Agent runner / producer (Layer 3b of 3)
=========================================================

ROLE IN THE SYSTEM
------------------
This is the entry point and the *producer* side of the event-driven demo.
It plays the role of an "agent runner": it orchestrates one full agent
lifecycle (start → LLM call → completion) and emits lifecycle events at
each milestone.

FULL SYSTEM OVERVIEW
--------------------
The demo showcases a three-layer in-process pub/sub (publish-subscribe)
architecture:

  ┌──────────────────────────────────────────────────────────────────┐
  │ Layer 1: event_types.py                                          │
  │   Defines the shared vocabulary — dataclasses for each event.    │
  │   Both producers and consumers import from here.                 │
  ├──────────────────────────────────────────────────────────────────┤
  │ Layer 2: agent_event_emitter.py                                  │
  │   The event bus. Receives emitted events from the producer and   │
  │   forwards them to registered subscriber handlers.               │
  ├──────────────────────────────────────────────────────────────────┤
  │ Layer 3a: subscriber.py  (consumer)                              │
  │   Registers handler functions on the emitter.                    │
  │   Handlers react to events (log, trace, alert, etc.).            │
  │                                                                  │
  │ Layer 3b: events_demo.py  (producer — this file)                 │
  │   Creates the emitter, wires up subscribers, then runs the agent │
  │   lifecycle and emits events.                                    │
  └──────────────────────────────────────────────────────────────────┘

DATA FLOW FOR ONE RUN
---------------------
  main()
    │
    ├─ create AgentEventEmitter()        ← the shared bus
    ├─ subscribe_event_logger(emitter)   ← wire up consumer(s)
    │
    ├─ emitter.emit(AgentStarted)        ← signal: run begins
    │      └─ handler prints "[agent] agent.started"
    │
    ├─ llm.chat(messages)                ← actual LLM call (blocking)
    │
    ├─ emitter.emit(GenerationCompleted) ← signal: LLM replied
    │      └─ handler prints "[gen] generation.completed ..."
    │
    └─ emitter.emit(AgentCompleted)      ← signal: run ends
           └─ handler prints "[agent] agent.completed ..."

WHY EMIT EVENTS INSTEAD OF JUST PRINTING?
------------------------------------------
The agent runner (this file) knows nothing about logging, Langfuse,
cost tracking, or any other observability concern. It only knows about
the emitter. This means:
  - Adding a new subscriber (e.g. Langfuse tracing) requires NO changes here.
  - Removing all subscribers (e.g. in unit tests) still lets the runner work.
  - The runner stays focused on its single responsibility: running the agent.

HOW TO EXTEND THIS DEMO
-----------------------
  1. Add a new event type in event_types.py.
  2. Call emitter.emit(NewEvent(...)) at the right place in main().
  3. Add a case branch in subscriber.py (or create a new subscriber module).
  No changes to AgentEventEmitter needed.
"""

import sys
import os
from time import time
from uuid import uuid4

# Make the project root importable so that common/ package is on the path.
# This is a dev-mode path fix; in a properly installed package you would
# use relative imports or install the package with pip install -e .
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from dotenv import load_dotenv

from common.llm_service import LLMService
from agent_event_emitter import AgentEventEmitter
from event_types import EventContext, AgentStarted, GenerationCompleted, AgentCompleted
from subscriber import subscribe_event_logger
from langfuse_subscriber import LangfuseSubscriber

# Load API keys and other secrets from .env / secrets.env.
# Must be called before any LLMService is constructed.
load_dotenv()


def main() -> None:
    """
    Run one full agent lifecycle and emit events at each milestone.

    This function acts as the "agent runner". In a real system this logic
    would live inside an Agent class, but here it is kept as a plain function
    to keep the demo readable.

    STEPS
    -----
    1. Create the shared event bus (AgentEventEmitter).
    2. Register all subscribers (wiring the consumer side).
    3. Build an EventContext that will be stamped on every event in this run.
    4. Emit AgentStarted — signals the beginning of the agent lifecycle.
    5. Call the LLM and measure wall-clock duration.
    6. Emit GenerationCompleted — signals the LLM replied.
    7. Emit AgentCompleted — signals the end of the agent lifecycle.

    NOTE ON EVENTCONTEXT
    --------------------
    The EventContext is built once and reused for all events in this run.
    This gives every event the same trace_id, session_id, and agent_id,
    which is what allows subscribers to correlate all events from a single
    run. In a multi-agent system each agent would have its own agent_id but
    share the same trace_id.

    NOTE ON TIMING
    --------------
    We capture t0 just before the LLM call and compute duration_ms after it
    returns. This gives us the actual LLM latency. We pass duration_ms into
    AgentCompleted so that any subscriber (logger, Langfuse, SLA monitor)
    gets accurate timing without needing to do its own timing logic.
    """

    # -----------------------------------------------------------------------
    # Step 1: Create the event bus
    # -----------------------------------------------------------------------
    # One emitter per run. Holds the registry of subscriber handlers.
    # All emit() calls below will route through this object.
    emitter = AgentEventEmitter()

    # -----------------------------------------------------------------------
    # Step 2: Register subscribers
    # -----------------------------------------------------------------------
    # subscribe_event_logger registers a wildcard handler that prints each
    # event to stdout. Any other subscriber (Langfuse, cost tracker, etc.)
    # would be wired up here in the same way — before the run starts.
    subscribe_event_logger(emitter)

    langfuse = LangfuseSubscriber()
    langfuse.attach(emitter)

    # -----------------------------------------------------------------------
    # Step 3: Build the shared EventContext (run "passport")
    # -----------------------------------------------------------------------
    # uuid4() gives a globally unique trace_id so this run can be
    # distinguished from all past and future runs, even across restarts.
    # In a real system this context might be passed in from an outer
    # orchestrator so events from sub-agents share the same trace.
    ctx = EventContext(
        trace_id=str(uuid4()),   # globally unique run identifier
        session_id="s1",         # logical session (could group multiple runs)
        agent_id="a1",           # this agent's identifier
        root_agent_id="a1",      # same as agent_id since there's no parent here
        depth=0,                 # depth 0 = root agent (no parent above it)
        timestamp=time() * 1000, # epoch ms at run start — used for ordering
    )

    # -----------------------------------------------------------------------
    # Step 4: Emit AgentStarted — "opening the span"
    # -----------------------------------------------------------------------
    # Subscribers use this event to start timers, open traces, log headers.
    # After this emit() returns, all registered handlers have already fired.
    emitter.emit(AgentStarted(type="agent.started", ctx=ctx))

    # -----------------------------------------------------------------------
    # Step 5: Do the actual LLM work
    # -----------------------------------------------------------------------
    # LLMService is a thin wrapper around the OpenAI-compatible chat endpoint.
    # We use gpt-4o-mini for cost efficiency in this demo.
    # The messages list follows the OpenAI chat format:
    #   [{"role": "user"|"assistant"|"system", "content": "..."}]
    llm = LLMService(model="openai/gpt-4o-mini")
    messages = [{"role": "user", "content": "What is 9+2? Answer in one word."}]

    # Capture t0 just before the call so we measure pure LLM latency.
    t0 = time()
    reply = llm.chat(messages)
    # duration_ms is the wall-clock time the LLM call took, in milliseconds.
    duration_ms = (time() - t0) * 1000

    # -----------------------------------------------------------------------
    # Step 6: Emit GenerationCompleted — "LLM replied"
    # -----------------------------------------------------------------------
    # The reply is embedded in the event so subscribers never need to call
    # back into the LLM service or capture it from an outer scope.
    emitter.emit(GenerationCompleted(
        type="generation.completed",
        ctx=ctx,
        output=reply,
        model=llm.model,
        input=messages,
        input_tokens=llm.last_usage.input_tokens,
        output_tokens=llm.last_usage.output_tokens,
        duration_ms=duration_ms,
    ))

    # -----------------------------------------------------------------------
    # Step 7: Emit AgentCompleted — "closing the span"
    # -----------------------------------------------------------------------
    # Passes duration_ms so observability subscribers can record latency
    # without doing their own timing. result carries the final answer.
    emitter.emit(AgentCompleted(type="agent.completed", ctx=ctx, duration_ms=duration_ms, result=reply))

    langfuse.flush()


if __name__ == "__main__":
    main()
