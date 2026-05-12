"""Tool subpackage for Super Agent dispatcher/tool definitions."""

from typing import Any

from .planner_tools import PLANNER_TOOLS, PlannerToolDispatcher, make_planner_dispatcher

__all__ = [
    "TOOLS",
    "ToolDispatcher",
    "make_dispatcher",
    "PLANNER_TOOLS",
    "PlannerToolDispatcher",
    "make_planner_dispatcher",
]


def __getattr__(name: str) -> Any:
    """Load orchestrator tool symbols on demand to break import cycles.

    Importing ``solver_tools`` must not eagerly load ``orchestrator_tools``,
    because the latter instantiates ``SolverAgent`` at runtime only.
    """
    if name == "TOOLS":
        from .orchestrator_tools import TOOLS

        return TOOLS
    if name == "ToolDispatcher":
        from .orchestrator_tools import ToolDispatcher

        return ToolDispatcher
    if name == "make_dispatcher":
        from .orchestrator_tools import make_dispatcher

        return make_dispatcher
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
