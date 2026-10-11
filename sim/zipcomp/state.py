"""Read-only view of a search state, the only thing a submission's ``cost`` sees.

A Zip state is fully determined by (set of visited cells, head cell): the
checkpoints collected so far are exactly the numbered cells in the visited
set, because numbers can only be entered in order. The engine deduplicates on
that pair, so ``State`` deliberately exposes nothing path-dependent (no parent,
no move history). Two different paths that reach the same state therefore
always get the same cost.

Cells are ``(row, col)`` tuples everywhere in the participant-facing API.
"""

from __future__ import annotations


class State:
    __slots__ = ("_node", "_board", "_cache")

    def __init__(self, node, board):
        object.__setattr__(self, "_node", node)
        object.__setattr__(self, "_board", board)
        object.__setattr__(self, "_cache", {})

    def __setattr__(self, key, value):
        raise AttributeError("State is read-only")

    # ---- board facts --------------------------------------------------------
    @property
    def rows(self) -> int:
        return self._board.height

    @property
    def cols(self) -> int:
        return self._board.width

    @property
    def n_cells(self) -> int:
        return self._board.n

    @property
    def n_numbers(self) -> int:
        """How many numbered cells (checkpoints) the board has."""
        return self._board.num_checkpoints

    # ---- progress -----------------------------------------------------------
    @property
    def head(self) -> tuple[int, int]:
        """Cell the path currently ends on."""
        return self._board.rc(self._node.head)

    @property
    def cells_visited(self) -> int:
        return self._node.depth

    @property
    def cells_left(self) -> int:
        return self._board.n - self._node.depth

    @property
    def last_number(self) -> int:
        """Highest number reached so far (1 at the start)."""
        return self._node.next_cp - 1

    @property
    def next_number(self) -> int | None:
        """Number that must be reached next, or None once the last one is reached."""
        k = self._node.next_cp
        return k if k <= self._board.num_checkpoints else None

    @property
    def numbers_left(self) -> int:
        """Numbers not reached yet (0 when solved)."""
        return self._board.num_checkpoints - self.last_number

    @property
    def on_number(self) -> bool:
        """True if the head sits on a numbered cell (always ``last_number``)."""
        return self._node.head in self._board.cp_number

    @property
    def is_solved(self) -> bool:
        return self._board.is_goal(self._node)

    # ---- lookups ------------------------------------------------------------
    def is_visited(self, cell: tuple[int, int]) -> bool:
        return self._node.is_visited(self._board.idx(*cell))

    def number_at(self, cell: tuple[int, int]) -> int | None:
        """Number printed on ``cell``, or None for a blank cell."""
        return self._board.cp_number.get(self._board.idx(*cell))

    def cell_of(self, number: int) -> tuple[int, int]:
        """Cell carrying ``number`` (1-based)."""
        return self._board.checkpoints[number - 1]

    def neighbours(self, cell: tuple[int, int]) -> list[tuple[int, int]]:
        """Orthogonal neighbours not separated by a wall (visited or not)."""
        b = self._board
        return [b.rc(d) for d in b.adj[b.idx(*cell)]]

    def __repr__(self) -> str:
        return (f"State(head={self.head}, cells_visited={self.cells_visited}/{self.n_cells}, "
                f"last_number={self.last_number}/{self.n_numbers})")
