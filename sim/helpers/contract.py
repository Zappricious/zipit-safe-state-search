"""The one class every submission subclasses.

    from helpers import CostFunction, bfs_to_next

    class Cost(CostFunction):
        name = "my_team"

        def cost(self, state):
            return state.numbers_left * state.n_cells + bfs_to_next(state)

The engine expands the state with the LOWEST cost first. ``cost`` must return
a finite int or float and depend only on ``state`` (no randomness, no
counters kept on ``self``).

Optional hooks, ignored unless switched on in ``config/engine.json``:

    def prune(self, state) -> bool      # True = never put this state on the frontier
    def tiebreak(self, state) -> tuple  # among equal costs, SMALLER tuple first

With tiebreak off, equal costs are expanded newest-first (depth-first on
plateaus).
"""

from __future__ import annotations


class CostFunction:
    name: str = "unnamed"

    def cost(self, state) -> float:
        raise NotImplementedError("submission must override cost(self, state)")
