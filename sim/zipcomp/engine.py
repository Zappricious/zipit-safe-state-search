"""Best-first search engine, lowest cost first.

The engine owns the frontier, the closed set, the game rules, and metric
counting. A participant supplies one ``Cost`` class (see ``helpers/contract.py``):

  * ``cost(state) -> number`` — **lower is expanded first**. Called once per
    generated state; it is a per-state value, not a path sum.
  * optional ``prune(state) -> bool`` and ``tiebreak(state) -> tuple``, used
    only when switched on in ``config/engine.json`` (both off by default).

Ordering on the frontier: lower cost, then (if enabled) smaller tiebreak
tuple, then most recently generated (newest-first / depth-first on ties).

Completeness: a state is (visited cells, head); next checkpoint is implied by
the visited set. Each state is expanded at most once (closed set), every
legal move is generated, and the state space is finite, so with prune off
the search finds a solution whenever one exists (given enough budget).

Metrics:
  expansions     states popped and expanded
  generated      children pushed onto the frontier
  dead_ends      expanded states that produced no child
  backtracks     expansions whose state is not a child of the previously
                 expanded state (the search jumped to another branch)
  backtrack_cells  total path cells undone by those jumps (how far back it
                 had to go: undoing 1 cell is cheap, undoing 30 is not)
  tiebreaks      pops where the best cost was shared with the next entry
  tiebreak_lifo  of those, ties the tiebreak hook could not split (decided
                 newest-first)
  duplicates     popped states already expanded via another path
"""

from __future__ import annotations

import heapq
import itertools
import json
import math
import os
import time

from .game import Board, Node
from .paths import CONFIG
from .state import State


class ContractError(Exception):
    """A submission broke the cost-function contract at run time."""


def engine_config() -> dict:
    path = os.path.join(CONFIG, "engine.json")
    if os.path.isfile(path):
        with open(path) as f:
            return json.load(f)
    return {}


def _check_cost(value, state) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ContractError(f"cost() must return a finite int/float, got {value!r} for {state!r}")
    return value


def _cells_undone(last: Node, parent: Node) -> int:
    """Cells on ``last``'s path that are not on ``parent``'s path (both share the start node)."""
    a, b = last, parent
    while a.depth > b.depth:
        a = a.parent
    while b.depth > a.depth:
        b = b.parent
    while a is not b:
        a, b = a.parent, b.parent
    return last.depth - a.depth


def best_cost_search(board: Board, cost_fn, max_expansions: int = 200_000,
                     time_limit: float = 30.0, recorder=None,
                     use_prune: bool | None = None, use_tiebreak: bool | None = None) -> dict:
    cfg = engine_config()
    if use_prune is None:
        use_prune = bool(cfg.get("prune", False))
    if use_tiebreak is None:
        use_tiebreak = bool(cfg.get("tiebreak", False))
    prune = getattr(cost_fn, "prune", None) if use_prune else None
    tiebreak = getattr(cost_fn, "tiebreak", None) if use_tiebreak else None

    seq = itertools.count()
    stats = dict(expansions=0, generated=0, pruned=0, dead_ends=0, backtracks=0,
                 backtrack_cells=0, tiebreaks=0, tiebreak_lifo=0, duplicates=0)

    def entry(node):
        state = State(node, board)
        c = _check_cost(cost_fn.cost(state), state)
        tb = tuple(tiebreak(state)) if tiebreak is not None else ()
        # heapq pops the smallest tuple; -seq makes newer entries win exact ties.
        return (c, tb, -next(seq), node)

    frontier = [entry(board.start_node())]
    closed: set[tuple[int, int]] = set()
    last: Node | None = None
    solution = None
    status = "exhausted"
    t0 = time.perf_counter()

    while frontier:
        cost, tb, _, node = heapq.heappop(frontier)
        key = (node.visited, node.head)
        if key in closed:
            stats["duplicates"] += 1
            continue
        closed.add(key)

        tie = bool(frontier) and frontier[0][0] == cost
        if tie:
            stats["tiebreaks"] += 1
            if frontier[0][1] == tb:
                stats["tiebreak_lifo"] += 1
        backtrack = last is not None and node.parent is not last
        if backtrack:
            stats["backtracks"] += 1
            stats["backtrack_cells"] += _cells_undone(last, node.parent)

        node.eid = stats["expansions"]
        stats["expansions"] += 1
        last = node
        if recorder is not None:
            recorder.expand(node, cost, tie, backtrack)

        if board.is_goal(node):
            solution = node.path
            status = "solved"
            break
        if stats["expansions"] >= max_expansions:
            status = "expansion_limit"
            break
        if (stats["expansions"] & 1023) == 0 and time.perf_counter() - t0 > time_limit:
            status = "time_limit"
            break

        pushed = 0
        for child in board.successors(node):
            if (child.visited, child.head) in closed:
                continue
            if prune is not None and prune(State(child, board)):
                stats["pruned"] += 1
                continue
            heapq.heappush(frontier, entry(child))
            pushed += 1
        stats["generated"] += pushed
        if pushed == 0:
            stats["dead_ends"] += 1

    stats["elapsed"] = time.perf_counter() - t0
    stats["status"] = status
    stats["solution"] = solution
    stats["prune"] = prune is not None
    stats["tiebreak"] = tiebreak is not None
    if recorder is not None:
        recorder.finish(status, solution, stats)
    return stats
