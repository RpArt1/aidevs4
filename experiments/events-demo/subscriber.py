"""Event subscribers for the events-demo package."""

from agent_event_emitter import AgentEventEmitter


def subscribe_event_logger(emitter: AgentEventEmitter) -> None:
    """Attach a stateless wildcard handler that prints each event to stdout."""

    def handler(event) -> None:
        match event.type:
            case "agent.started":
                print(f"[agent] Event type: agent.started   agent_id={event.ctx.agent_id}")
            case "generation.completed":
                preview = (event.output or "")[:60]
                print(f"[gen] Event type: generation.completed   output={preview!r}...")
            case "agent.completed":
                print(f"[agent] Event type: agent.completed   duration_ms={event.duration_ms:.0f}")
            case _:
                print(f"[event] Event type: {event.type}")

    emitter.on_any(handler)
