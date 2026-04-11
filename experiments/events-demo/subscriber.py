"""
subscriber.py — Consumer / subscriber layer (Layer 3a of 3)
============================================================

PATTERN: Observer / Pub-Sub — consumer side
This module defines *subscribers*: functions that react to events.
A subscriber is wired up via a factory function that takes an emitter,
creates a handler closure, and registers it. The producer never changes.

FACTORY + CLOSURE PATTERN
--------------------------
  subscribe_event_logger(emitter)
      |-- defines handler() as a closure (can close over injected config)
      +-- calls emitter.on_any(handler)  <- wires handler to the bus

To add a new subscriber (e.g. Langfuse tracing, cost tracking), create
another factory function and call it alongside subscribe_event_logger() in
events_demo.py. Zero changes to the emitter or the runner needed.
"""

from agent_event_emitter import AgentEventEmitter


def subscribe_event_logger(emitter: AgentEventEmitter) -> None:
    """
    Subscriber factory: registers a wildcard logging handler.

    Creates an inner handler closure and registers it via emitter.on_any(),
    so it receives every event emitted during the run. Dispatches on
    event.type using match/case -- the string discriminator approach mirrors
    how the emitter routes events and avoids importing event classes here.
    """

    def handler(event) -> None:
        match event.type:
            case "agent.started":
                print(f"[agent] Event type: agent.started   agent_id={event.ctx.agent_id}")

            case "generation.completed":
                # Truncate output preview to keep the log line readable.
                preview = (event.output or "")[:60]
                print(f"[gen] Event type: generation.completed   output={preview!r}...")

            case "agent.completed":
                # duration_ms was computed by the runner and carried in the event --
                # this handler needs no timing logic of its own.
                print(f"[agent] Event type: agent.completed   duration_ms={event.duration_ms:.0f}")

            case _:
                # Wildcard handlers must handle unknown types gracefully.
                print(f"[event] Event type: {event.type}")

    emitter.on_any(handler)
