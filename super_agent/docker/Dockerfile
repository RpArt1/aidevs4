# Super Agent — single-container sandbox.
#
# The whole super-agent (Orchestrator + Planner + Solver) runs inside this
# container. Generated code is executed in-process as subprocesses via
# `super_agent/sandbox.py` (rlimits + timeout) — no docker-in-docker.
#
# Build context = repo root (so we can COPY common/, assignments/, super_agent/
# in one shot). See `super_agent/docker/run.sh` for the wrapper that builds
# from the right context.

FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Minimal OS deps. `tini` gives us a real PID 1 that reaps zombies and
# forwards signals to the entrypoint script (important because the solver
# spawns long-running children via process_manager).
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        tini \
        ca-certificates \
        curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first so this layer caches across code changes.
COPY requirements.txt /app/requirements.txt
RUN pip install -r /app/requirements.txt

# Copy the whole repo. The super-agent reuses common/ at runtime, and the
# solver's generated code is allowed to `import common` if it wants to.
# `super_agent/workspaces/` is recreated empty on each run so we don't ship
# host scratch into the image.
COPY common/ /app/common/
COPY assignments/ /app/assignments/
COPY super_agent/ /app/super_agent/
COPY pyproject.toml /app/pyproject.toml

# Workspaces live inside the container; each run gets its own subdir.
RUN mkdir -p /app/super_agent/workspaces && rm -rf /app/super_agent/workspaces/*

# Make the package importable without a full editable install.
ENV PYTHONPATH=/app

COPY super_agent/docker/entrypoint.sh /usr/local/bin/super-agent-entrypoint
RUN chmod +x /usr/local/bin/super-agent-entrypoint

ENTRYPOINT ["/usr/bin/tini", "--", "/usr/local/bin/super-agent-entrypoint"]
CMD []
