"""
agent_event_emitter.py — Event bus (Layer 2 of 3)
=================================================

PATTERN: Observer / Pub-Sub
AgentEventEmitter is the central broker between producers and consumers.
The producer (runner) calls emit(); registered handlers are called immediately.
Producer and consumers share only this object — neither knows about the other.

  events_demo.py  ── emit() ──►  AgentEventEmitter  ── handler() ──►  subscriber.py
  (producer)                    (this class / bus)                    (consumer)

TWO SUBSCRIPTION CHANNELS
--------------------------
  on(type, handler)  — typed channel: handler receives only one event type.
                       Use when a subscriber cares about a single event.
  on_any(handler)    — wildcard "*" channel: handler receives every event.
                       Use for cross-cutting concerns (logging, tracing).

SYNCHRONOUS DELIVERY
---------------------
Handlers are called inline inside emit(), in registration order.
Typed handlers fire first, wildcard handlers second.
"""

from collections import defaultdict


class AgentEventEmitter:
    """In-process pub/sub event bus. One instance per agent run."""

    def __init__(self):
        # registry: channel_key -> [callable, ...]
        # defaultdict(list) avoids KeyError when a channel has no subscribers yet.
        # "*" is the wildcard key used by on_any().
        self._listeners = defaultdict(list)

    def emit(self, event) -> None:
        """
        Publish an event to all interested subscribers.

        Fires typed handlers for event.type first, then wildcard "*" handlers.
        The producer calls this; it has no knowledge of who is listening.
        """
        for handler in self._listeners[event.type]:
            handler(event)
        for handler in self._listeners["*"]:
            handler(event)

    def on(self, event_type: str, handler) -> None:
        """Subscribe handler to a single event type (e.g. "agent.completed")."""
        self._listeners[event_type].append(handler)

    def on_any(self, handler) -> None:
        """
        Subscribe handler to ALL events (wildcard channel).

        The handler receives every event regardless of type. Use event.type
        inside the handler (match/case or if/elif) to act selectively.
        Preferred for logging, tracing, and any observability concern.
        """
        self._listeners["*"].append(handler)
