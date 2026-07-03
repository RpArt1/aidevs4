"""Reusable building blocks composed by `SuperAgentBase` and its subclasses.

Exports:
    BudgetGuard:        iteration + wall-clock budget tracker.
    BudgetExceeded:     raised when a `BudgetGuard` cap is exhausted.
    ResourcePreFetcher: pre-fetch URL previews from task text for the planner.
"""

from .budget_guard import BudgetExceeded, BudgetGuard
from .resource_prefetcher import ResourcePreFetcher

__all__ = [
    "BudgetExceeded",
    "BudgetGuard",
    "ResourcePreFetcher",
]
