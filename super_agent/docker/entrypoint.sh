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

# SSH (used by tunnel tools like pinggy) requires the current UID to have an
# entry in /etc/passwd. When the container is launched with --user <host-uid>
# that UID usually has no entry, causing "No user exists for uid N".
# Fix: use libnss-wrapper to inject a temporary passwd entry at runtime.
if ! getent passwd "$(id -u)" > /dev/null 2>&1; then
    _tmp_passwd=$(mktemp)
    cp /etc/passwd "$_tmp_passwd"
    echo "user:x:$(id -u):$(id -g):User:/tmp:/bin/bash" >> "$_tmp_passwd"
    export NSS_WRAPPER_PASSWD="$_tmp_passwd"
    export NSS_WRAPPER_GROUP=/etc/group
    export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libnss_wrapper.so
fi

# Respect LOG_LEVEL env var (forwarded from run.sh); default to INFO.
LOG_LEVEL="${LOG_LEVEL:-INFO}"

if [[ -n "${TASK_TEXT:-}" ]]; then
    exec python -m super_agent --log-level "${LOG_LEVEL}" "$TASK_TEXT" "$@"
fi

if [[ -n "${TASK_FILE:-}" ]]; then
    exec python -m super_agent --log-level "${LOG_LEVEL}" --task-file "$TASK_FILE" "$@"
fi

# No env-driven input — forward whatever the caller passed (may be empty,
# in which case `python -m super_agent` will print its own usage error).
exec python -m super_agent --log-level "${LOG_LEVEL}" "$@"
