#!/usr/bin/env bash
# Super Agent container entrypoint.
#
# Resolves the task text from one of three sources (in priority order):
#   1. $TASK_TEXT env var (inline string)
#   2. $TASK_FILE env var (path inside the container)
#   3. positional CLI args, forwarded verbatim to `python -m super_agent`
#
# Anything passed as positional args is forwarded as-is, so callers can also
# do e.g. `docker run super_agent --task-file /work/task.txt` or pipe via
# `docker run -i ... super_agent --stdin < task.txt`.

set -euo pipefail

if [[ -n "${TASK_TEXT:-}" ]]; then
    exec python -m super_agent "$TASK_TEXT" "$@"
fi

if [[ -n "${TASK_FILE:-}" ]]; then
    exec python -m super_agent --task-file "$TASK_FILE" "$@"
fi

# No env-driven input — forward whatever the caller passed (may be empty,
# in which case `python -m super_agent` will print its own usage error).
exec python -m super_agent "$@"
