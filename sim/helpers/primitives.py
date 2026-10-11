"""Measurement primitives for ``cost(state)``. Intentionally not a Zip solver.

Every function takes the read-only ``State`` the engine hands to ``cost`` and
answers one question. Combining them (weights, bonuses, if/else) is the
submission. Cells are ``(row, col)`` tuples. Anything that needs a BFS or a
flood fill is cached on the state, so calling it twice is free.

Full reference with examples: helpful_text/HELPERS.md
"""

from __future__ import annotations

from collections import deque


def _cached(state, key, fn):
    cache = state._cache
    if key not in cache:
        cache[key] = fn()
    return cache[key]


# ---- distances ----------------------------------------------------------------
def manhattan(a: tuple[int, int], b: tuple[int, int]) -> int:
    """Grid distance between two cells, ignoring walls and visited cells."""
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def manhattan_to_number(state, number: int) -> int:
    """Manhattan distance from the head to the cell carrying ``number``."""
    return manhattan(state.head, state.cell_of(number))


def manhattan_to_next(state, ahead: int = 0) -> int:
    """Manhattan distance from the head to number ``next_number + ahead``.

    ``ahead=0`` is the next number (n+1), ``ahead=1`` the one after (n+2).
    Returns 0 if that number does not exist.
    """
    nxt = state.next_number
    if nxt is None or nxt + ahead > state.n_numbers:
        return 0
    return manhattan_to_number(state, nxt + ahead)


def bfs_to_next(state) -> int | None:
    """Shortest walk from the head to the next number through unvisited cells.

    Respects walls and treats every other unreached number as blocked (you may
    not step on it yet). 0 if there is no next number, None if it can no
    longer be reached.
    """
    def run():
        b, node = state._board, state._node
        if node.next_cp > b.num_checkpoints:
            return 0
        target = b.cp_idx[node.next_cp - 1]
        blocked = node.visited
        for c in b.cp_idx[node.next_cp:]:
            blocked |= 1 << c
        q = deque([(node.head, 0)])
        seen = blocked | (1 << node.head)
        while q:
            c, dist = q.popleft()
            for d in b.adj[c]:
                if d == target:
                    return dist + 1
                if not (seen >> d) & 1:
                    seen |= 1 << d
                    q.append((d, dist + 1))
        return None
    return _cached(state, "bfs_to_next", run)


def detour(state) -> int:
    """Extra steps the real walk needs over the straight line (BFS - Manhattan).

    0 = a straight route is open; large = walls or your own path are in the way.
    Returns 0 if the next number is unreachable (check ``bfs_to_next`` for that).
    """
    walk = bfs_to_next(state)
    return 0 if walk is None else walk - manhattan_to_next(state)


def next_gap(state) -> int:
    """Manhattan distance from the next number to the one after it (n+1 -> n+2).

    Tells you how far you will have to travel right after collecting the next
    number. 0 if there is no number after the next.
    """
    nxt = state.next_number
    if nxt is None or nxt + 1 > state.n_numbers:
        return 0
    return manhattan(state.cell_of(nxt), state.cell_of(nxt + 1))


# ---- progress -------------------------------------------------------------------
def last_number_cell(state) -> tuple[int, int]:
    """Cell of the latest number reached (``state.last_number``)."""
    return state.cell_of(state.last_number)


def cells_per_number(state) -> float:
    """Free cells still to fill per number still to reach (slack per segment)."""
    return state.cells_left / max(1, state.numbers_left)


# ---- neighbourhood ----------------------------------------------------------------
def _free(state, idx: int) -> list[int]:
    b, visited = state._board, state._node.visited
    return [d for d in b.adj[idx] if not (visited >> d) & 1]


def _idx(state, cell) -> int:
    return state._node.head if cell is None else state._board.idx(*cell)


def unvisited_neighbours(state, cell=None) -> list[tuple[int, int]]:
    """Unvisited neighbours of ``cell`` (default: head), walls respected."""
    return [state._board.rc(d) for d in _free(state, _idx(state, cell))]


def free_degree(state, cell=None) -> int:
    """Number of unvisited neighbours of ``cell`` (default: head). Warnsdorff degree."""
    return len(_free(state, _idx(state, cell)))


# ---- safety: is the rest still solvable? ---------------------------------------------
def reachable_cells(state) -> int:
    """Unvisited cells reachable from the head through unvisited cells."""
    def run():
        b, node = state._board, state._node
        stack = [node.head]
        seen = node.visited
        count = 0
        while stack:
            c = stack.pop()
            for d in b.adj[c]:
                if not (seen >> d) & 1:
                    seen |= 1 << d
                    count += 1
                    stack.append(d)
        return count
    return _cached(state, "reachable_cells", run)


def cut_off_cells(state) -> int:
    """Unvisited cells the head can no longer reach. Any > 0 means unsolvable."""
    return state.cells_left - reachable_cells(state)


def components(state) -> int:
    """Separate regions the unvisited cells fall into. More than 1 means unsolvable."""
    def run():
        b, node = state._board, state._node
        seen = node.visited
        count = 0
        for start in range(b.n):
            if (seen >> start) & 1:
                continue
            count += 1
            seen |= 1 << start
            stack = [start]
            while stack:
                c = stack.pop()
                for d in b.adj[c]:
                    if not (seen >> d) & 1:
                        seen |= 1 << d
                        stack.append(d)
        return count
    return _cached(state, "components", run)


def dead_end_cells(state) -> int:
    """Unvisited non-final cells with at most one way in/out (head counts as one).

    Such a cell can be entered but never left, so any > 0 means unsolvable.
    """
    def run():
        b, node = state._board, state._node
        dead = 0
        for c in range(b.n):
            if node.is_visited(c) or c == b.goal:
                continue
            exits = sum(1 for d in b.adj[c] if not node.is_visited(d) or d == node.head)
            if exits <= 1:
                dead += 1
        return dead
    return _cached(state, "dead_end_cells", run)


def corridor_cells(state) -> int:
    """Unvisited cells with exactly two unvisited neighbours: forced one-way passages.

    Not fatal, but many corridors means the remaining area is thin and fragile.
    """
    def run():
        b, node = state._board, state._node
        return sum(1 for c in range(b.n) if not node.is_visited(c)
                   and sum(1 for d in b.adj[c] if not node.is_visited(d)) == 2)
    return _cached(state, "corridor_cells", run)


def forced_moves(state) -> int:
    """Empty cells next to the head that must be entered on the very next move.

    Such a cell (not the last number) has exactly one other empty neighbour,
    so if the head walks away it can never be passed through. 1 = the move is
    forced; 2 or more = the state is lost.
    """
    def run():
        goal = state._board.goal
        return sum(1 for c in _free(state, state._node.head) if c != goal and len(_free(state, c)) == 1)
    return _cached(state, "forced_moves", run)


def _chokepoints(state):
    """One Tarjan pass over the empty cells plus the head -> (articulation cells, split cells, head splits)."""
    def run():
        b, node = state._board, state._node
        visited, head, adj = node.visited, node.head, b.adj
        disc = {head: 0}
        low = {head: 0}
        pieces = {}
        t = 1
        stack = [(head, -1, iter(adj[head]))]
        while stack:
            v, parent, it = stack[-1]
            for w in it:
                if (visited >> w) & 1 and w != head:
                    continue
                if w not in disc:
                    disc[w] = low[w] = t
                    t += 1
                    stack.append((w, v, iter(adj[w])))
                    break
                if w != parent:
                    low[v] = min(low[v], disc[w])
            else:
                stack.pop()
                if parent >= 0:
                    low[parent] = min(low[parent], low[v])
                    if low[v] >= disc[parent]:
                        pieces[parent] = pieces.get(parent, 0) + 1
        art = split = 0
        for v, k in pieces.items():
            if v == head:
                continue
            k += 1                       # the side the DFS came from
            if k >= 2:
                art += 1
            if k >= 3 or (v == b.goal and k >= 2):
                split += 1
        return art, split, pieces.get(head, 0) >= 2
    return _cached(state, "chokepoints", run)


def articulation_cells(state) -> int:
    """Empty cells whose removal disconnects the empty region (chokepoints).

    The path passes each cell once, so every chokepoint is a one-shot door
    between two areas. Not fatal on its own; many of them = a fragile region.
    """
    return _chokepoints(state)[0]


def split_cells(state) -> int:
    """Empty cells that are the only door to THREE or more areas (or the last
    number, if it is a door between two areas).

    The path passes such a cell once, so it can join at most two of those
    areas (one if the path ends there): any > 0 means unsolvable.
    """
    return _chokepoints(state)[1]


def head_splits_region(state) -> bool:
    """True if the empty cells around the head form two or more separate areas.

    The head can enter only one of them and never come back, so True means
    unsolvable. ``cut_off_cells`` misses this (every area still touches the head).
    """
    return _chokepoints(state)[2]


def bfs_next_to_after(state) -> int | None:
    """Shortest walk from the next number to the number after it (n+1 -> n+2),
    through empty cells, not stepping on later numbers.

    0 if there is no number after the next; None if that segment is already
    blocked (unsolvable). Ignores the cells you will use to reach n+1.
    """
    def run():
        b, node = state._board, state._node
        k = node.next_cp
        if k + 1 > b.num_checkpoints:
            return 0
        src, target = b.cp_idx[k - 1], b.cp_idx[k]
        blocked = node.visited
        for c in b.cp_idx[k + 1:]:
            blocked |= 1 << c
        q = deque([(src, 0)])
        seen = blocked | (1 << src)
        while q:
            c, dist = q.popleft()
            for d in b.adj[c]:
                if d == target:
                    return dist + 1
                if not (seen >> d) & 1:
                    seen |= 1 << d
                    q.append((d, dist + 1))
        return None
    return _cached(state, "bfs_next_to_after", run)


def local_free(state, radius: int = 2) -> int:
    """Empty cells within ``radius`` steps (Manhattan, ignoring walls) of the head.

    "Pocket left behind": a high count means the head is about to walk away
    from empty cells it will have to come back for.
    """
    (hr, hc) = state.head
    return sum(1 for r in range(max(0, hr - radius), min(state.rows, hr + radius + 1))
               for c in range(max(0, hc - radius), min(state.cols, hc + radius + 1))
               if abs(r - hr) + abs(c - hc) <= radius and (r, c) != (hr, hc)
               and not state.is_visited((r, c)))


def body_contacts(state) -> int:
    """Sides of the head touching the path already drawn (not walls or edges).

    High = the path is hugging itself, leaving no thin gaps behind.
    """
    b, node = state._board, state._node
    hr, hc = state.head
    near = [(hr + dr, hc + dc) for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1))]
    return sum(1 for r, c in near if 0 <= r < state.rows and 0 <= c < state.cols
               and node.is_visited(b.idx(r, c))) - (1 if node.parent is not None else 0)


def pace_error(state) -> float:
    """How far the path is ahead of (+) or behind (-) an even pace.

    Compares the share of cells filled with the share of numbers reached. Big
    positive = you filled lots of cells but reached few numbers (wandering);
    big negative = you rushed numbers and left cells behind.
    """
    k = state.n_numbers
    if k <= 1:
        return 0.0
    return state.cells_visited / state.n_cells - (state.last_number - 1) / (k - 1)


def is_hopeless(state) -> bool:
    """True if the state provably cannot be completed: a cell is cut off, a
    dead-end cell exists, or the next number is unreachable. Cheap necessary
    checks, not a full proof (see also ``forced_moves``)."""
    if state.is_solved:
        return False
    return cut_off_cells(state) > 0 or dead_end_cells(state) > 0 or bfs_to_next(state) is None


# ---- geometry extras ---------------------------------------------------------------
def on_border(state, cell=None) -> bool:
    """True if ``cell`` (default: head) is on the outer edge of the board."""
    r, c = state.head if cell is None else cell
    return r in (0, state.rows - 1) or c in (0, state.cols - 1)


def border_cells_left(state) -> int:
    """Unvisited cells on the outer edge. Edges are easy to strand, so clearing
    them early is a common strategy."""
    def run():
        return sum(1 for r in range(state.rows) for c in range(state.cols)
                   if (r in (0, state.rows - 1) or c in (0, state.cols - 1))
                   and not state.is_visited((r, c)))
    return _cached(state, "border_cells_left", run)


def cells_in_box(state) -> int:
    """Unvisited cells inside the rectangle spanned by the head and the next number.

    Roughly: how much you would skip past if you ran straight there.
    0 if there is no next number.
    """
    nxt = state.next_number
    if nxt is None:
        return 0
    (r1, c1), (r2, c2) = state.head, state.cell_of(nxt)
    return sum(1 for r in range(min(r1, r2), max(r1, r2) + 1)
               for c in range(min(c1, c2), max(c1, c2) + 1)
               if not state.is_visited((r, c)))
