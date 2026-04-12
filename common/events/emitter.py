"""In-process pub/sub event bus for agent lifecycle events."""

from collections import defaultdict
from typing import Callable, Any


class AgentEventEmitter:
    """
    Synchronous in-process event bus. One instance per agent run.

    Supports two subscription channels:
    - ``on(type, handler)`` — typed channel, receives one event type only.
    - ``on_any(handler)`` — wildcard channel, receives every event.

    Typed handlers fire before wildcard handlers. All handlers are called
    inline inside ``emit()``, in registration order.
    """

    def __init__(self) -> None:
        self._listeners: dict[str, list[Callable]] = defaultdict(list)

    def emit(self, event: Any) -> None:
        """Publish an event to all matching subscribers."""
        for handler in self._listeners[event.type]:
            handler(event)
        for handler in self._listeners["*"]:
            handler(event)

    def on(self, event_type: str, handler: Callable) -> None:
        """Subscribe ``handler`` to a single event type (e.g. ``"agent.completed"``)."""
        self._listeners[event_type].append(handler)

    def on_any(self, handler: Callable) -> None:
        """
        Subscribe ``handler`` to every event (wildcard channel).

        Use ``event.type`` inside the handler to act selectively.
        Preferred for cross-cutting concerns such as logging and tracing.
        """
        self._listeners["*"].append(handler)
