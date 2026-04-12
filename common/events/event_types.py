"""Shared event vocabulary for the agent lifecycle pub/sub system."""

from dataclasses import dataclass, field


@dataclass
class EventContext:
    """
    Correlation metadata stamped on every event.

    Allows subscribers to link events from the same run (trace_id),
    the same agent (agent_id), or reconstruct a multi-agent call tree
    (parent_agent_id, depth, root_agent_id) without accessing the agent object.
    """
    trace_id: str
    timestamp: float
    session_id: str
    agent_id: str
    root_agent_id: str
    depth: int
    parent_agent_id: str | None = None


# ── Agent lifecycle ───────────────────────────────────────────────────────────

@dataclass
class AgentStarted:
    """Emitted once at the start of a run. Subscribers use it to open a trace span."""
    type: str   # "agent.started"
    ctx: EventContext


@dataclass
class AgentCompleted:
    """
    Emitted once at the end of a run. Together with AgentStarted it brackets
    the full lifecycle. duration_ms is pre-computed by the runner.
    """
    type: str   # "agent.completed"
    ctx: EventContext
    duration_ms: float
    result: str | None = None


@dataclass
class AgentError:
    """
    Emitted when a caught exception or unrecoverable error condition occurs.

    Surfaces errors to observability systems that cannot inspect log files.
    error_type is a short machine-readable label (e.g. ``"tool_dispatch"``,
    ``"json_decode"``, ``"llm_call"``). step and tool_name are optional
    context to locate where in the loop the error occurred.
    """
    type: str   # "agent.error"
    ctx: EventContext
    error_type: str
    message: str
    step: int | None = None
    tool_name: str | None = None


@dataclass
class IterationLimitReached:
    """
    Emitted when the agent exits the loop without producing a result.

    This is the primary signal that the agent failed to complete its goal
    within the allowed budget. Subscribers can use it to alert on stuck agents.
    """
    type: str   # "agent.iteration_limit"
    ctx: EventContext
    max_iterations: int
    step: int


# ── LLM generation ────────────────────────────────────────────────────────────

@dataclass
class GenerationCompleted:
    """
    Emitted after each LLM call.

    In a multi-step loop this fires once per reasoning step. Carries everything
    a subscriber needs — output, model, input messages, token counts, and
    wall-clock duration — so subscribers need no extra state or callbacks.
    """
    type: str   # "generation.completed"
    ctx: EventContext
    output: str | None = None
    model: str | None = None
    input: list | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    duration_ms: float = 0.0
    step: int | None = None


# ── Tool calls ────────────────────────────────────────────────────────────────

@dataclass
class ToolCallStarted:
    """
    Emitted immediately before a tool is executed.

    Together with ToolCallCompleted it brackets the tool call, enabling
    latency measurement and hang detection. Parallel tool calls (multiple
    calls in one LLM step) each get their own pair of events, identified
    by call_id.
    """
    type: str   # "tool.started"
    ctx: EventContext
    call_id: str
    tool_name: str
    arguments: dict = field(default_factory=dict)
    step: int | None = None


@dataclass
class ToolCallCompleted:
    """
    Emitted after a tool returns, whether successfully or not.

    ``success=False`` means the tool itself reported failure (e.g. API error),
    not that an exception was raised — exceptions produce AgentError instead.
    result carries the raw string returned by the tool (typically JSON).
    """
    type: str   # "tool.completed"
    ctx: EventContext
    call_id: str
    tool_name: str
    result: str | None = None
    duration_ms: float = 0.0
    success: bool = True
    step: int | None = None
