"""Iteration- and wall-clock-based budget enforcement for agent run loops.

`BudgetGuard` owns the budget caps and the run start time, and decides
whether the agent has exhausted its allowance. It is deliberately
ignorant of agents, events, and LLMs so it stays trivially testable.

Typical use::

    self.budget = BudgetGuard(max_iterations=20, wall_clock_s=600)
    self.budget.mark_started()
    for step in range(1, self.budget.max_iterations + 1):
        self.budget.raise_if_exceeded(step)
        ...
"""

from __future__ import annotations

from time import time


class BudgetExceeded(Exception):
    """
    Raised by `BudgetGuard.raise_if_exceeded` when the agent has exhausted
    its iteration count or wall-clock budget. The spawning agent catches it
    and converts it into a structured ``{"outcome": "max_iter", ...}`` result.
    """

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


class BudgetGuard:
    """
    Iteration + wall-clock budget tracker for one agent run.

    Stateful: ``mark_started()`` records the start of the run loop so the
    wall-clock cap can be enforced. Before that call, only the iteration
    cap is checked.
    """

    def __init__(self, max_iterations: int, wall_clock_s: int) -> None:
        self.max_iterations = max_iterations
        self.wall_clock_s = wall_clock_s
        self._started_at: float | None = None

    def mark_started(self) -> None:
        """Record the wall-clock start of the agent's run loop.

        Call once at the top of ``run()`` before entering the loop, so that
        ``raise_if_exceeded`` can enforce the wall-clock cap.
        """
        self._started_at = time()

    def check(self, step: int) -> str | None:
        """Return a human-readable reason when the budget is exhausted.

        Iteration cap is checked against ``step`` (1-indexed). Wall-clock
        cap is only enforced once ``mark_started()`` has been called.

        Returns:
            Reason string when exhausted, ``None`` otherwise.
        """
        if step > self.max_iterations:
            return f"max_iterations={self.max_iterations} exceeded at step={step}"
        if self._started_at is not None:
            elapsed = time() - self._started_at
            if elapsed > self.wall_clock_s:
                return (
                    f"wall_clock_s={self.wall_clock_s} exceeded "
                    f"(elapsed={elapsed:.1f}s) at step={step}"
                )
        return None

    def raise_if_exceeded(self, step: int) -> None:
        """Raise `BudgetExceeded` when the budget is exhausted.

        Convenience that composes `check` + raise — the only pattern in
        use today across the agent run loops.

        Raises:
            BudgetExceeded: When the iteration or wall-clock budget has
                been exhausted at the given ``step``.
        """
        reason = self.check(step)
        if reason is not None:
            raise BudgetExceeded(reason)
