"""Built-in subscribers for the agent event bus."""

from .emitter import AgentEventEmitter


def subscribe_event_logger(emitter: AgentEventEmitter) -> None:
    """
    Register a wildcard handler that prints each lifecycle event to stdout.

    Call this before the agent run starts. Adding or removing this subscriber
    requires no changes to the emitter or the runner.
    """

    def handler(event) -> None:
        match event.type:
            case "agent.started":
                print(f"[agent] agent.started   agent_id={event.ctx.agent_id}")
            case "agent.completed":
                print(f"[agent] agent.completed   duration_ms={event.duration_ms:.0f}   result={str(event.result)[:60]!r}")
            case "agent.error":
                print(f"[agent] agent.error   error_type={event.error_type}   step={event.step}   msg={event.message}")
            case "agent.iteration_limit":
                print(f"[agent] agent.iteration_limit   step={event.step}/{event.max_iterations}")
            case "generation.completed":
                preview = (event.output or "")[:60]
                print(f"[gen]   generation.completed   step={event.step}   tokens={event.input_tokens}+{event.output_tokens}   output={preview!r}...")
            case "tool.started":
                print(f"[tool]  tool.started   step={event.step}   tool={event.tool_name}   call_id={event.call_id}")
            case "tool.completed":
                status = "ok" if event.success else "FAIL"
                print(f"[tool]  tool.completed   step={event.step}   tool={event.tool_name}   status={status}   duration_ms={event.duration_ms:.0f}")
            case _:
                print(f"[event] {event.type}")

    emitter.on_any(handler)
