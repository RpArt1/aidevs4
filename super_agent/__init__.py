"""
Super Agent — multi-agent system that automates the analyze-plan-code-run-verify-fix
loop end-to-end for plain-text aidevs task descriptions.

v0 layout: an Orchestrator (LLM ReAct supervisor) spawns a Planner (single
structured-LLM call) and a Solver (LLM ReAct loop with a small tool surface)
inside a single Docker container. All three agents share `SuperAgentBase`
(this package's local abstract base class).

"""
