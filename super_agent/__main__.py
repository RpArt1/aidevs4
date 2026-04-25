"""Command-line entrypoint for running the Super Agent package.

This module intentionally only bootstraps a run: it resolves task text,
creates the per-run workspace, wires logging/events, and delegates control to
`OrchestratorAgent`. The orchestrator tools and solver implementation are
separate build steps.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Sequence
from uuid import uuid4

from common import LLMService, get_logger, setup_logging
from common.events import (
    AgentEventEmitter,
    LangfuseSubscriber,
    subscribe_event_logger,
)

from .orchestrator import OrchestratorAgent

DEFAULT_MODEL = "openai/gpt-4o"
WORKSPACES_DIR = Path(__file__).resolve().parent / "workspaces"


def build_parser() -> argparse.ArgumentParser:
    """Create the CLI argument parser.

    Returns:
        Configured argument parser for `python -m super_agent`.
    """
    parser = argparse.ArgumentParser(
        prog="python -m super_agent",
        description="Run the Super Agent on a plain-text aidevs task.",
    )
    input_group = parser.add_mutually_exclusive_group()
    input_group.add_argument(
        "--task-file",
        type=Path,
        help="Path to a file containing the task text.",
    )
    input_group.add_argument(
        "--stdin",
        action="store_true",
        help="Read task text from stdin.",
    )
    parser.add_argument(
        "task_text",
        nargs="*",
        help="Inline task text. Multiple words are joined with spaces.",
    )
    parser.add_argument(
        "--public-webhook-url",
        default=os.getenv("PUBLIC_WEBHOOK_URL"),
        help="Externally reachable webhook URL for long-running webhook tasks.",
    )
    parser.add_argument(
        "--verify-task-name",
        dest="verify_task_name_override",
        help="Optional human-provided override for AssignmentService task slug.",
    )
    parser.add_argument(
        "--model",
        default=os.getenv("SUPER_AGENT_MODEL", DEFAULT_MODEL),
        help=f"OpenRouter model name to use. Defaults to {DEFAULT_MODEL}.",
    )
    parser.add_argument(
        "--session-id",
        help="Optional tracing session id. Defaults to the generated run id.",
    )
    parser.add_argument(
        "--no-event-log",
        action="store_true",
        help="Disable stdout event subscriber output.",
    )
    return parser


def resolve_task_text(args: argparse.Namespace, parser: argparse.ArgumentParser) -> str:
    """Resolve task text from file, stdin, or positional words.

    Args:
        args: Parsed CLI arguments.
        parser: Parser used to report validation errors.

    Returns:
        Non-empty task text.

    Raises:
        SystemExit: Raised by `parser.error` when no non-empty task text was
            provided.
    """
    if args.task_file is not None:
        task_text = args.task_file.read_text(encoding="utf-8")
    elif args.stdin:
        task_text = sys.stdin.read()
    else:
        task_text = " ".join(args.task_text)

    task_text = task_text.strip()
    if not task_text:
        parser.error("task text is required via positional args, --task-file, or --stdin")
    return task_text


def create_workspace(run_id: str) -> Path:
    """Create and return the per-run workspace directory.

    Args:
        run_id: Unique id for this Super Agent run.

    Returns:
        Path to `super_agent/workspaces/<run_id>`.
    """
    workspace = WORKSPACES_DIR / run_id
    workspace.mkdir(parents=True, exist_ok=False)
    return workspace


def build_emitter(enable_event_log: bool) -> tuple[AgentEventEmitter, LangfuseSubscriber]:
    """Create the event emitter and attach built-in subscribers.

    Args:
        enable_event_log: Whether to print compact event logs to stdout.

    Returns:
        Tuple of the shared emitter and Langfuse subscriber. The caller owns
        flushing the subscriber after the run.
    """
    emitter = AgentEventEmitter()
    if enable_event_log:
        subscribe_event_logger(emitter)

    langfuse = LangfuseSubscriber(tags=["super_agent_v0"])
    langfuse.attach(emitter)
    return emitter, langfuse


def run_super_agent(args: argparse.Namespace, task_text: str) -> dict:
    """Bootstrap and run the orchestrator.

    Args:
        args: Parsed CLI arguments.
        task_text: Plain-text aidevs task description.

    Returns:
        Final result dictionary returned by `OrchestratorAgent.run`.
    """
    run_id = str(uuid4())
    workspace = create_workspace(run_id)
    emitter, langfuse = build_emitter(enable_event_log=not args.no_event_log)
    log = get_logger("super_agent.__main__")

    log.info("starting super-agent run_id=%s workspace=%s", run_id, workspace)
    try:
        orchestrator = OrchestratorAgent(
            task_text=task_text,
            run_id=run_id,
            workspace=workspace,
            emitter=emitter,
            llm=LLMService(model=args.model),
            public_webhook_url=args.public_webhook_url,
            verify_task_name_override=args.verify_task_name_override,
            session_id=args.session_id,
        )
        result = orchestrator.run()
        result.setdefault("run_id", run_id)
        result.setdefault("workspace", str(workspace))
        return result
    finally:
        langfuse.flush()


def main(argv: Sequence[str] | None = None) -> int:
    """Run the Super Agent CLI.

    Args:
        argv: Optional argument vector. Defaults to `sys.argv[1:]`.

    Returns:
        Process exit code.
    """
    parser = build_parser()
    args = parser.parse_args(argv)
    task_text = resolve_task_text(args, parser)

    setup_logging()
    result = run_super_agent(args, task_text)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result.get("status") == "success" else 1


if __name__ == "__main__":
    raise SystemExit(main())
