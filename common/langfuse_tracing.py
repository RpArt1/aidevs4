from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Any, Generator

from common.logger import get_logger

log = get_logger(__name__)

# Lazy-imported to avoid a hard dependency when tracing is disabled.
_langfuse: Any = None


def is_tracing_enabled() -> bool:
    """Return True when the required Langfuse env vars are present."""
    return bool(os.getenv("LANGFUSE_SECRET_KEY") and os.getenv("LANGFUSE_PUBLIC_KEY"))


def init_tracing() -> None:
    """Initialize the Langfuse client and store it as a module-level singleton.

    Mirrors initTracing() in the TS version. Safe to call multiple times —
    subsequent calls are no-ops if tracing is already initialized.
    """
    global _langfuse

    if not is_tracing_enabled():
        log.info("langfuse disabled — set LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY to enable")
        return

    if _langfuse is not None:
        return  # already initialized

    from langfuse import get_client
    _langfuse = get_client()

    host = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")
    log.info(f"langfuse tracing enabled (host={host})")


def shutdown_tracing() -> None:
    """Flush all pending events and shut down the client.

    Equivalent to shutdownTracing() in the TS version. Call this before
    process exit in scripts; long-running servers can skip it (flush is automatic).
    """
    global _langfuse
    if _langfuse is None:
        return
    try:
        _langfuse.flush()
        log.info("langfuse tracing flushed and shut down")
    except Exception as exc:
        log.error(f"langfuse shutdown error: {exc}")
    finally:
        _langfuse = None


# ── Observation factories ────────────────────────────────────────────────────
#
# Each function mirrors its TS counterpart (traceAgent / traceGeneration / traceTool).
# They all return a context manager that wraps the observation span.
# When tracing is disabled they yield None so callers can use the same
#   `with trace_agent(...) as obs:` pattern without any conditional checks.


@contextmanager
def trace_agent(
    name: str,
    *,
    input: Any = None,
    metadata: dict[str, Any] | None = None,
) -> Generator[Any, None, None]:
    """Open an 'agent' span — the top-level trace container for one agentic run."""
    if _langfuse is None:
        yield None
        return
    with _langfuse.start_as_current_observation(
        name=name,
        as_type="agent",
        input=input,
        metadata=metadata,
    ) as obs:
        yield obs


@contextmanager
def trace_generation(
    name: str,
    *,
    model: str,
    input: Any = None,
    model_parameters: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
) -> Generator[Any, None, None]:
    """Open a 'generation' span — represents a single LLM call.

    Always nest this inside a trace_agent span so the generation appears as
    a child in the Langfuse dashboard rather than a standalone trace.
    """
    if _langfuse is None:
        yield None
        return
    with _langfuse.start_as_current_observation(
        name=name,
        as_type="generation",
        model=model,
        input=input,
        model_parameters=model_parameters,
        metadata=metadata,
    ) as obs:
        yield obs


@contextmanager
def trace_tool(
    name: str,
    *,
    input: Any = None,
    metadata: dict[str, Any] | None = None,
) -> Generator[Any, None, None]:
    """Open a 'tool' span — represents a single tool/function call inside an agent run."""
    if _langfuse is None:
        yield None
        return
    with _langfuse.start_as_current_observation(
        name=name,
        as_type="tool",
        input=input,
        metadata=metadata,
    ) as obs:
        yield obs
