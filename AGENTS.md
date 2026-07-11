# AGENTS.md

## Cursor Cloud specific instructions

### What this project is
A multi-agent "super-agent" (focus of development) that solves `aidevs` tasks. An
`OrchestratorAgent` (ReAct supervisor) coordinates a `PlannerAgent` and a
`SolverAgent` over OpenRouter LLM calls. Code lives in `super_agent/` and shared
infrastructure (LLM client, logging, event pub/sub, assignment API) in `common/`.
`assignments/` holds older course lessons; `experiments/` holds throwaway demos.

### Environment
- The startup update script creates/refreshes a virtualenv at `.venv` (gitignored).
  Activate it before doing anything: `source .venv/bin/activate`.
- `requirements.txt` is intentionally lean. The Docker image (and the update script)
  additionally install runtime libs that the code imports or that solver-generated
  programs may use: `debugpy` (hard import in
  `super_agent/agent/tools/solver_tools.py`), plus `pydantic numpy pandas
  beautifulsoup4 httpx`. Without `debugpy`, `import super_agent` fails.

### Required configuration
- `OPENROUTER_API_KEY` is **required** for any real run — `common/llm_service.py`
  raises `ValueError` on import-time use if it is unset. All three agents make
  OpenRouter calls (`https://openrouter.ai/api/v1`).
- Optional: `AIDEVS_API_KEY` / `AIDEVS_VERIFY_URL` (only needed to actually submit
  answers via `common/assignment_service.py`), and `LANGFUSE_PUBLIC_KEY` /
  `LANGFUSE_SECRET_KEY` / `LANGFUSE_HOST` (tracing; no-op if unset).
- Secrets are read via `python-dotenv` from a repo-root `.env`, or from real
  environment variables.

### Run the super-agent (the product)
```
source .venv/bin/activate
python -m super_agent "your plain-text task here"
# or: python -m super_agent --task-file path/to/task.txt
# or: echo "task" | python -m super_agent --stdin
```
Useful flags: `--mock-solver` (runs planner/orchestrator only, still needs an LLM
key), `--log-level DEBUG`, `--model openai/gpt-4o-mini`. Per-run artifacts
(`plan.json`, generated scripts, logs) are written under
`/tmp/aidevs4_plan/<run_id>/`; override with `SUPER_AGENT_WORKSPACES_DIR`.

There is also a Docker wrapper (`super_agent/docker/run.sh`) for a hardened
sandbox run; not needed for local development.

### Lint / test / build
- No automated test suite and no linter config are committed. "Build" for this
  package is just an editable install (`pip install -e .`), already done by the
  update script. Sanity-check with `python -m compileall super_agent common` and
  `python -m super_agent --help`.
