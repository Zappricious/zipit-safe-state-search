"""Submission contract: what ``submissions/submission_NN/`` must look like.

Rules (any violation rejects the submission):

  Folder
    * ``search.py`` is the only Python file (no ``tiebreaker.py``; tie-breaking
      is an optional method on the same class).

  search.py
    * Imports only ``helpers`` (``from helpers import ...``), ``math`` and
      ``from __future__ import annotations``.
    * Exactly one class, named ``Cost``, subclassing ``CostFunction``.
    * ``Cost`` sets ``name = "<non-empty string>"``.
    * ``Cost`` defines ``cost(self, state)``; optional ``prune(self, state)``
      and ``tiebreak(self, state)`` take exactly the same arguments.
    * No access to private attributes (``x._anything``) except on ``self``,
      and no reflection / IO builtins (``getattr``, ``eval``, ``open``, ...).

  Runtime (smoke test on a small built-in board)
    * ``Cost()`` constructs with no arguments.
    * ``cost`` returns a finite int/float, the same value on repeated calls.
    * ``prune`` returns a bool, ``tiebreak`` a tuple of numbers.
"""

from __future__ import annotations

import ast
import math
import os

ALLOWED_MODULES = {"helpers", "math", "__future__"}
FORBIDDEN_NAMES = {
    "getattr", "setattr", "delattr", "vars", "globals", "locals", "eval", "exec",
    "compile", "open", "__import__", "input", "breakpoint", "exit", "quit",
}
STATE_METHODS = ("cost", "prune", "tiebreak")


class SubmissionRejected(Exception):
    def __init__(self, where: str, problems: list[str]):
        self.where = where
        self.problems = problems
        super().__init__(f"{where} rejected:\n  - " + "\n  - ".join(problems))


def _check_args(fn: ast.FunctionDef) -> str | None:
    a = fn.args
    names = [x.arg for x in a.posonlyargs + a.args]
    if (names != ["self", "state"] or a.vararg or a.kwarg or a.kwonlyargs
            or a.defaults or a.kw_defaults):
        return f"`{fn.name}` must be declared exactly as `def {fn.name}(self, state)`"
    return None


def check_source(source: str, path: str = "search.py") -> list[str]:
    """Static checks. Returns a list of problems (empty = ok)."""
    try:
        tree = ast.parse(source, filename=path)
    except SyntaxError as e:
        return [f"syntax error: {e}"]

    problems: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name not in ALLOWED_MODULES - {"__future__", "helpers"}:
                    problems.append(f"line {node.lineno}: `import {alias.name}` not allowed "
                                    f"(use `from helpers import ...`; only `math` may be imported)")
        elif isinstance(node, ast.ImportFrom):
            if node.level or node.module not in ALLOWED_MODULES:
                problems.append(f"line {node.lineno}: `from {node.module or '.'} import ...` not allowed "
                                f"(only `helpers`, `math`, `__future__`)")
        elif isinstance(node, ast.Attribute):
            if node.attr.startswith("_") and not (isinstance(node.value, ast.Name) and node.value.id == "self"):
                problems.append(f"line {node.lineno}: private attribute `.{node.attr}` not allowed")
        elif isinstance(node, ast.Name) and node.id in FORBIDDEN_NAMES:
            problems.append(f"line {node.lineno}: `{node.id}` not allowed")

    classes = [n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]
    if len(classes) != 1:
        problems.append(f"must define exactly one class (`class Cost(CostFunction)`), found {len(classes)}")
    top = [n for n in tree.body if isinstance(n, ast.ClassDef)]
    if len(top) != 1 or top[0].name != "Cost":
        problems.append("the class must be top-level and named `Cost`")
        return problems
    cls = top[0]
    if not any(isinstance(b, ast.Name) and b.id == "CostFunction" for b in cls.bases) or len(cls.bases) != 1:
        problems.append("`Cost` must subclass `CostFunction` only (`from helpers import CostFunction`)")

    has_name = False
    methods = {}
    for item in cls.body:
        if isinstance(item, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "name" for t in item.targets):
            v = item.value
            has_name = isinstance(v, ast.Constant) and isinstance(v.value, str) and bool(v.value.strip())
        elif isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
            methods[item.name] = item
    if not has_name:
        problems.append('`Cost` must set `name = "<non-empty string>"`')
    if "cost" not in methods:
        problems.append("`Cost` must define `def cost(self, state)`")
    for m in STATE_METHODS:
        if m in methods:
            if isinstance(methods[m], ast.AsyncFunctionDef):
                problems.append(f"`{m}` must not be async")
            elif (msg := _check_args(methods[m])):
                problems.append(msg)
    return problems


def check_folder(folder: str) -> list[str]:
    problems = []
    search_py = os.path.join(folder, "search.py")
    if not os.path.isfile(search_py):
        return ["missing search.py"]
    extra = sorted(f for f in os.listdir(folder) if f.endswith(".py") and f != "search.py")
    if extra:
        problems.append(f"only search.py is allowed, found extra: {', '.join(extra)} "
                        f"(tie-breaking is now an optional `tiebreak(self, state)` method on Cost)")
    with open(search_py) as f:
        problems += check_source(f.read(), search_py)
    return problems


def _smoke_board():
    from .game import Board
    # 4x4, hand-checked solvable: snake order, one wall the snake never crosses.
    return Board(width=4, height=4, checkpoints=[(0, 0), (1, 1), (2, 3), (3, 0)],
                 walls=[((0, 1), (1, 1))], board_id="smoke")


def check_runtime(instance) -> list[str]:
    """Call the hooks on a handful of states from a tiny board."""
    from .state import State
    board = _smoke_board()
    nodes = [board.start_node()]
    frontier = list(nodes)
    while frontier and len(nodes) < 40:
        kids = board.successors(frontier.pop(0))
        nodes += kids
        frontier += kids

    problems: list[str] = []
    for node in nodes:
        st = State(node, board)
        try:
            a = instance.cost(st)
            b = instance.cost(State(node, board))
        except Exception as e:  # noqa: BLE001
            return [f"cost() raised {type(e).__name__}: {e} on {st!r}"]
        if isinstance(a, bool) or not isinstance(a, (int, float)) or not math.isfinite(a):
            return [f"cost() must return a finite int/float, got {a!r} on {st!r}"]
        if a != b:
            return [f"cost() is not deterministic: {a!r} then {b!r} on {st!r}"]
        if callable(getattr(instance, "prune", None)):
            try:
                p = instance.prune(st)
            except Exception as e:  # noqa: BLE001
                return [f"prune() raised {type(e).__name__}: {e}"]
            if not isinstance(p, bool):
                return [f"prune() must return True/False, got {p!r}"]
        if callable(getattr(instance, "tiebreak", None)):
            try:
                t = instance.tiebreak(st)
            except Exception as e:  # noqa: BLE001
                return [f"tiebreak() raised {type(e).__name__}: {e}"]
            if not isinstance(t, tuple) or not all(
                    isinstance(x, (int, float)) and not isinstance(x, bool) for x in t):
                return [f"tiebreak() must return a tuple of numbers, got {t!r}"]
    return problems
