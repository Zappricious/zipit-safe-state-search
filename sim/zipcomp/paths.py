"""Repo layout helpers. All tools resolve names like ``suite_6_4`` / ``submission_01`` here.

A suite is ``testcases/suite_N_K/``: every board is N x N with exactly K numbers.
"""

from __future__ import annotations

import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TESTCASES = os.path.join(ROOT, "testcases")
SUITES_CONFIG = os.path.join(TESTCASES, "suites.json")
SUBMISSIONS = os.path.join(ROOT, "submissions")
CONFIG = os.path.join(ROOT, "config")
RANKING = os.path.join(ROOT, "ranking")
ARENA = os.path.join(ROOT, "arena")

SUITE_RE = re.compile(r"^suite_(\d+)_(\d+)$")
SUB_RE = re.compile(r"^submission_(\d+)$")


def suite_name(n: int, k: int) -> str:
    return f"suite_{n}_{k}"


def suite_dir(name: str) -> str:
    if not SUITE_RE.match(name):
        raise ValueError(f"suite must look like suite_N_K (e.g. suite_6_4), got {name!r}")
    path = os.path.join(TESTCASES, name)
    if not os.path.isdir(path):
        raise FileNotFoundError(f"unknown suite {name} ({path}); run `python3 cli.py generate`")
    return path


def suite_boards_path(name: str) -> str:
    return os.path.join(suite_dir(name), "boards.json")


def suite_baselines_path(name: str) -> str:
    return os.path.join(suite_dir(name), "baselines.json")


def submission_dir(name: str) -> str:
    if not SUB_RE.match(name):
        raise ValueError(f"submission must look like submission_01, got {name!r}")
    path = os.path.join(SUBMISSIONS, name)
    if not os.path.isdir(path):
        raise FileNotFoundError(f"unknown submission {name} ({path})")
    return path


def list_suites() -> list[str]:
    """Generated suites, easiest first (by N, then K)."""
    if not os.path.isdir(TESTCASES):
        return []
    found = []
    for name in os.listdir(TESTCASES):
        m = SUITE_RE.match(name)
        if m and os.path.isdir(os.path.join(TESTCASES, name)):
            found.append((int(m.group(1)), int(m.group(2)), name))
    return [n for _, _, n in sorted(found)]


def list_submissions() -> list[str]:
    if not os.path.isdir(SUBMISSIONS):
        return []
    found = []
    for name in os.listdir(SUBMISSIONS):
        m = SUB_RE.match(name)
        if m and os.path.isdir(os.path.join(SUBMISSIONS, name)):
            found.append((int(m.group(1)), name))
    return [n for _, n in sorted(found)]
