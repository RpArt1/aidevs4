#!/usr/bin/env bash
# Human-facing wrapper: build the super-agent image (if needed) and run it
# with the hardening flags from the design (cap-drop, no-new-privileges,
# pids/cpu/memory limits, port 3000 published for lesson3 webhooks).
#
# Usage:
#   super_agent/docker/run.sh "<inline task text>"
#   super_agent/docker/run.sh --task-file assignments/lesson1/task.txt
#   cat task.txt | super_agent/docker/run.sh --stdin
#   super_agent/docker/run.sh --shell
#     → opens an interactive bash inside the same image/mounts; check e.g.
#       `head -50 /app/super_agent/prompts/planner.md` to confirm prompt revision.
#
# Env knobs:
#   IMAGE              image tag to build/run (default: super_agent:dev)
#   SKIP_BUILD=1       skip `docker build` (use the existing image as-is)
#   PUBLIC_WEBHOOK_URL forwarded into the container (lesson3)
#   AIDEVS_API_KEY     forwarded into the container
#   AIDEVS_VERIFY_URL  forwarded into the container
#   OPENROUTER_API_KEY forwarded into the container
#   LANGFUSE_*         forwarded into the container if set
#   HOST_PORT          host port to publish 3000 on (default: 3000)
#   PLAN_PERSIST_DIR   host dir for run workspaces (default: /tmp/aidevs4_plan)
#   DOCKER_RUN_AS_HOST_USER run container with host uid:gid (default: 1)
#   EXTRA_DOCKER_ARGS  extra flags inserted before the image name

set -euo pipefail

IMAGE="${IMAGE:-super_agent:dev}"
HOST_PORT="${HOST_PORT:-9999}"
PLAN_PERSIST_DIR="${PLAN_PERSIST_DIR:-/tmp/aidevs4_plan}"
DOCKER_RUN_AS_HOST_USER="${DOCKER_RUN_AS_HOST_USER:-1}"

RUN_SHELL=0
if [[ "${1:-}" == "--shell" ]]; then
    RUN_SHELL=1
    shift
fi

# Resolve repo root (the build context) regardless of where this script is
# invoked from.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." >/dev/null 2>&1 && pwd)"

# Hand the repo-root .env straight to `docker run --env-file`. Note: docker's
# parser is strict — no `KEY = value` (spaces) and no surrounding quotes on
# values. Override path with ENV_FILE=/path/to/other.env.
ENV_FILE="${ENV_FILE:-${REPO_ROOT}/.env}"
env_file_args=()
if [[ -f "${ENV_FILE}" ]]; then
    env_file_args+=(--env-file "${ENV_FILE}")
fi
echo "Building context from: ${REPO_ROOT}"


if [[ "${SKIP_BUILD:-0}" == "1" ]]; then
    echo ">> SKIP_BUILD=1 — using existing ${IMAGE} (no docker build)" >&2
else
    echo ">> building ${IMAGE} from ${REPO_ROOT}" >&2
    docker build \
        -f "${SCRIPT_DIR}/Dockerfile" \
        -t "${IMAGE}" \
        --build-arg CACHEBUST="$(date +%s)" \
        "${REPO_ROOT}" </dev/null
fi

# Forward only env vars that are actually set, so we don't accidentally
# inject empty values that override .env defaults inside the container.
env_args=()
for var in \
    AIDEVS_API_KEY \
    AIDEVS_VERIFY_URL \
    OPENROUTER_API_KEY \
    PUBLIC_WEBHOOK_URL \
    LANGFUSE_PUBLIC_KEY \
    LANGFUSE_SECRET_KEY \
    LANGFUSE_HOST \
    SUPER_AGENT_WORKSPACES_DIR \
    TASK_TEXT \
    TASK_FILE; do
    if [[ -n "${!var:-}" ]]; then
        env_args+=(-e "${var}=${!var}")
    fi
done

# Ensure the container uses the mounted host directory for per-run artifacts
# unless the caller provided an explicit override.
if [[ -z "${SUPER_AGENT_WORKSPACES_DIR:-}" ]]; then
    env_args+=(-e "SUPER_AGENT_WORKSPACES_DIR=${PLAN_PERSIST_DIR}")
fi

# Allow the caller to mount a host directory at /work to expose task files
# inside the container without rebuilding.
mount_args=()
if [[ -n "${MOUNT_DIR:-}" ]]; then
    mount_args+=(-v "${MOUNT_DIR}:/work:ro")
fi

# Persist planner/solver run outputs (plan.json, generated scripts, logs) on host.
mkdir -p "${PLAN_PERSIST_DIR}"
mount_args+=(-v "${PLAN_PERSIST_DIR}:${PLAN_PERSIST_DIR}")

extra_args=()
if [[ -n "${EXTRA_DOCKER_ARGS:-}" ]]; then
    # shellcheck disable=SC2206
    extra_args=(${EXTRA_DOCKER_ARGS})
fi

user_args=()
if [[ "${DOCKER_RUN_AS_HOST_USER}" == "1" ]]; then
    user_args+=(--user "$(id -u):$(id -g)")
fi

echo ">> running ${IMAGE}" >&2

interactive_flags=(-i)
entrypoint_args=()
run_cli=("$@")
if [[ "${RUN_SHELL}" == "1" ]]; then
    interactive_flags=(-it)
    entrypoint_args=(--entrypoint /bin/bash)
    run_cli=()
    echo ">> interactive shell — verify prompts: sed -n '1,55p' /app/super_agent/prompts/planner.md" >&2
fi

exec docker run \
    --rm \
    "${interactive_flags[@]}" \
    --cap-drop=ALL \
    --security-opt=no-new-privileges \
    --pids-limit=256 \
    --cpus=2 \
    --memory=2g \
    -p "${HOST_PORT}:3000" \
    --add-host=host.docker.internal:host-gateway \
    "${user_args[@]}" \
    "${env_file_args[@]}" \
    "${env_args[@]}" \
    "${mount_args[@]}" \
    "${extra_args[@]}" \
    "${entrypoint_args[@]}" \
    "${IMAGE}" \
    "${run_cli[@]}"
