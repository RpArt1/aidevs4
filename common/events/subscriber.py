"""Built-in subscribers for the agent event bus."""

import logging

from .emitter import AgentEventEmitter

logger = logging.getLogger(__name__)


def subscribe_event_logger(emitter: AgentEventEmitter) -> None:
    """
    Register a wildcard handler that logs each lifecycle event at DEBUG level.

    Call this before the agent run starts. Adding or removing this subscriber
    requires no changes to the emitter or the runner.
    """

    def handler(event) -> None:
        match event.type:
            case "agent.started":
                logger.debug("[agent] agent.started   agent_id=%s", event.ctx.agent_id)
            case "agent.completed":
                logger.info("[agent] agent.completed   duration_ms=%.0f   result=%r", event.duration_ms, event.result)
            case "agent.error":
                logger.warning("[agent] agent.error   error_type=%s   step=%s   msg=%s", event.error_type, event.step, event.message)
            case "agent.iteration_limit":
                logger.warning("[agent] agent.iteration_limit   step=%s/%s", event.step, event.max_iterations)
            case "generation.completed":
                out = event.output or ""
                logger.trace(
                    "[gen]   generation.completed   step=%s   tokens=%s+%s   output=%r",
                    event.step, event.input_tokens, event.output_tokens, out,
                )
            case "tool.started":
                logger.trace("[tool]  tool.started   step=%s   tool=%s   call_id=%s", event.step, event.tool_name, event.call_id)
            case "tool.completed":
                status = "ok" if event.success else "FAIL"
                logger.info("[tool]  tool.completed   step=%s   tool=%s   status=%s   duration_ms=%.0f", event.step, event.tool_name, status, event.duration_ms)
            case _:
                logger.debug("[event] %s", event.type)

    emitter.on_any(handler)
