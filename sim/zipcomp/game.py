"""Zip game rules.

A board is an H x W grid. Cells are addressed as (i, j) = (row, col) and
internally as a flat index ``idx = i * W + j``.

Rules (same as the LinkedIn "Zip" puzzle):
  * The path starts on checkpoint 1.
  * Each step moves to an orthogonally adjacent cell that is not yet visited
    and not separated from the current cell by a wall.
  * Checkpoints must be visited in increasing order (you may not step on
    checkpoint k+1 before k).
  * The path must cover every cell exactly once and end on the last checkpoint.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

# Fixed neighbour order: up, down, left, right.
DIRECTIONS = ((-1, 0), (1, 0), (0, -1), (0, 1))


class Node:
    """A search node = a partial path ("snake") on the board.

    Only the head and a parent pointer are stored; the full path is rebuilt
    on demand, so a node costs O(1) memory.
    """

    __slots__ = ("head", "parent", "visited", "next_cp", "depth", "eid")

    def __init__(self, head: int, parent: "Node | None", visited: int, next_cp: int, depth: int):
        self.head = head          # flat index of the snake head
        self.parent = parent      # previous node (None for the start node)
        self.visited = visited    # bitmask of visited cells
        self.next_cp = next_cp    # number of the next checkpoint to reach
        self.depth = depth        # number of cells in the path
        self.eid = -1             # expansion id, set by the engine when expanded

    @property
    def path(self) -> list[int]:
        out = []
        n = self
        while n is not None:
            out.append(n.head)
            n = n.parent
        out.reverse()
        return out

    def is_visited(self, idx: int) -> bool:
        return (self.visited >> idx) & 1 == 1


@dataclass
class Board:
    width: int
    height: int
    checkpoints: list[tuple[int, int]]           # ordered: checkpoints[0] is number 1
    walls: list[tuple[tuple[int, int], tuple[int, int]]] = field(default_factory=list)
    board_id: str = ""

    def __post_init__(self):
        self.n = self.width * self.height
        self.full_mask = (1 << self.n) - 1
        self.cp_idx = [self.idx(i, j) for i, j in self.checkpoints]
        self.cp_number = {c: k + 1 for k, c in enumerate(self.cp_idx)}
        self.num_checkpoints = len(self.cp_idx)
        self.start = self.cp_idx[0]
        self.goal = self.cp_idx[-1]
        wall_set = {frozenset((self.idx(*a), self.idx(*b))) for a, b in self.walls}
        self.adj: list[list[int]] = []
        for c in range(self.n):
            i, j = self.rc(c)
            nbrs = []
            for di, dj in DIRECTIONS:
                ni, nj = i + di, j + dj
                if 0 <= ni < self.height and 0 <= nj < self.width:
                    d = self.idx(ni, nj)
                    if frozenset((c, d)) not in wall_set:
                        nbrs.append(d)
            self.adj.append(nbrs)

    # ---- coordinates -------------------------------------------------------
    def idx(self, i: int, j: int) -> int:
        return i * self.width + j

    def rc(self, idx: int) -> tuple[int, int]:
        return divmod(idx, self.width)

    def manhattan(self, a: int, b: int) -> int:
        ai, aj = self.rc(a)
        bi, bj = self.rc(b)
        return abs(ai - bi) + abs(aj - bj)

    # ---- rules -------------------------------------------------------------
    def start_node(self) -> Node:
        return Node(self.start, None, 1 << self.start, 2, 1)

    def can_enter(self, node: Node, cell: int) -> bool:
        """True if the snake at ``node`` may legally step onto ``cell``."""
        if node.is_visited(cell) or node.next_cp > self.num_checkpoints:
            return False
        num = self.cp_number.get(cell)
        if num is not None and num != node.next_cp:
            return False
        # The last checkpoint can only be entered as the very last cell.
        if cell == self.goal and node.depth + 1 != self.n:
            return False
        return True

    def successors(self, node: Node) -> list[Node]:
        children = []
        for cell in self.adj[node.head]:
            if self.can_enter(node, cell):
                nxt = node.next_cp + 1 if self.cp_number.get(cell) == node.next_cp else node.next_cp
                children.append(Node(cell, node, node.visited | (1 << cell), nxt, node.depth + 1))
        return children

    def is_goal(self, node: Node) -> bool:
        return node.depth == self.n and node.head == self.goal

    # ---- helpers available to submissions ---------------------------------
    def next_checkpoint_cell(self, node: Node) -> int | None:
        k = node.next_cp
        return self.cp_idx[k - 1] if k <= self.num_checkpoints else None

    def remaining(self, node: Node) -> int:
        return self.n - node.depth

    def free_degree(self, node: Node, cell: int) -> int:
        """Number of unvisited neighbours of ``cell`` (Warnsdorff degree)."""
        return sum(1 for d in self.adj[cell] if not node.is_visited(d))

    def unvisited_connected(self, node: Node) -> bool:
        """True if all unvisited cells form one region reachable from the head."""
        if node.visited == self.full_mask:
            return True
        stack = [node.head]
        seen = node.visited
        count = 0
        while stack:
            c = stack.pop()
            for d in self.adj[c]:
                if not (seen >> d) & 1:
                    seen |= 1 << d
                    count += 1
                    stack.append(d)
        return count == self.n - node.depth

    def dead_end_cells(self, node: Node) -> int:
        """Unvisited non-goal cells with <= 1 exit (head counts as an exit); any such cell means unsolvable."""
        dead = 0
        for c in range(self.n):
            if node.is_visited(c) or c == self.goal:
                continue
            free = sum(1 for d in self.adj[c] if not node.is_visited(d) or d == node.head)
            if free <= 1:
                dead += 1
        return dead

    # ---- (de)serialisation -------------------------------------------------
    def to_dict(self) -> dict:
        return {
            "id": self.board_id,
            "width": self.width,
            "height": self.height,
            "checkpoints": [list(c) for c in self.checkpoints],
            "walls": [[list(a), list(b)] for a, b in self.walls],
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Board":
        return cls(
            width=d["width"],
            height=d["height"],
            checkpoints=[tuple(c) for c in d["checkpoints"]],
            walls=[(tuple(a), tuple(b)) for a, b in d.get("walls", [])],
            board_id=d.get("id", ""),
        )


def load_suite(path: str) -> list[Board]:
    with open(path) as f:
        data = json.load(f)
    return [Board.from_dict(b) for b in data["boards"]]


def validate_solution(board: Board, path: list[int]) -> tuple[bool, str]:
    """Independent checker for a full solution path (flat indices)."""
    if len(path) != board.n or len(set(path)) != board.n:
        return False, "path must cover every cell exactly once"
    if path[0] != board.start or path[-1] != board.goal:
        return False, "path must start at checkpoint 1 and end at the last checkpoint"
    for a, b in zip(path, path[1:]):
        if b not in board.adj[a]:
            return False, f"illegal step {board.rc(a)} -> {board.rc(b)}"
    order = [board.cp_number[c] for c in path if c in board.cp_number]
    if order != list(range(1, board.num_checkpoints + 1)):
        return False, "checkpoints visited out of order"
    return True, "ok"
