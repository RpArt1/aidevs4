#!/usr/bin/env bash
# Human-facing wrapper: build the super-agent image (if needed) and run it
# with the hardening flags from the design (cap-drop, no-new-privileges,
# pids/cpu/memory limits, port 3000 published for lesson3 webhooks).
#
# Usage:
#   super_agent/docker/run.sh "<inline task text>"
#   super_agent/docker/run.sh --task-file assignments/lesson1/task.txt
#   cat task.txt | super_agent/docker/run.sh --stdin
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
#   EXTRA_DOCKER_ARGS  extra flags inserted before the image name

set -euo pipefail

IMAGE="${IMAGE:-super_agent:dev}"
HOST_PORT="${HOST_PORT:-3000}"

# Resolve repo root (the build context) regardless of where this script is
# invoked from.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." >/dev/null 2>&1 && pwd)"

if [[ "${SKIP_BUILD:-0}" != "1" ]]; then
    echo ">> building ${IMAGE} from ${REPO_ROOT}" >&2
    docker build \
        -f "${SCRIPT_DIR}/Dockerfile" \
        -t "${IMAGE}" \
        "${REPO_ROOT}"
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
    TASK_TEXT \
    TASK_FILE; do
    if [[ -n "${!var:-}" ]]; then
        env_args+=(-e "${var}=${!var}")
    fi
done

# Allow the caller to mount a host directory at /work to expose task files
# inside the container without rebuilding.
mount_args=()
if [[ -n "${MOUNT_DIR:-}" ]]; then
    mount_args+=(-v "${MOUNT_DIR}:/work:ro")
fi

extra_args=()
if [[ -n "${EXTRA_DOCKER_ARGS:-}" ]]; then
    # shellcheck disable=SC2206
    extra_args=(${EXTRA_DOCKER_ARGS})
fi

echo ">> running ${IMAGE}" >&2
exec docker run \
    --rm \
    -i \
    --cap-drop=ALL \
    --security-opt=no-new-privileges \
    --pids-limit=256 \
    --cpus=2 \
    --memory=2g \
    -p "${HOST_PORT}:3000" \
    "${env_args[@]}" \
    "${mount_args[@]}" \
    "${extra_args[@]}" \
    "${IMAGE}" \
    "$@"
