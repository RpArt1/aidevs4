# Layer 1 — event bus.
# This module is the single communication channel between the agent runner (producer)
# and any number of subscribers (consumers: loggers, observability tools, etc.).
# The runner holds one AgentEventEmitter instance and calls emit() at key lifecycle points.
# Subscribers register handlers via on() or on_any() — they never talk to the runner directly.
# Adding a new subscriber requires zero changes to the runner.

from collections import defaultdict


class AgentEventEmitter:
    def __init__(self):
        # registry: event_type -> [handler, ...]
        # "*" is the wildcard channel — receives every event regardless of type
        self._listeners = defaultdict(list)

    def emit(self, event):
        # fire handlers subscribed to this specific event type (e.g. "agent.started")
        for handler in self._listeners[event.type]:
            handler(event)
        # then fire wildcard handlers — useful for loggers, observability subscribers
        for handler in self._listeners["*"]:
            handler(event)

    def on(self, event_type, handler):
        # subscribe to a single event type
        self._listeners[event_type].append(handler)

    def on_any(self, handler):
        # subscribe to all events — shortcut for on("*", handler)
        self._listeners["*"].append(handler)
