# A* Search Optimization & Safe State Evaluation (ZipIT Benchmark)

Optimized heuristic evaluation, bit-parallel constraint propagation, and formal state-space verification developed for the **ZipIT Hamiltonian Pathfinding Benchmark (IIT Delhi)**.

---

## Overview

In constrained Hamiltonian pathfinding (Zip puzzles), an autonomous planner must traverse every cell of an $N \times M$ grid exactly once while visiting numbered checkpoints $1, 2, \dots, K$ in strict sequence and respecting obstacles/walls. Naive heuristic search suffers from catastrophic state-space explosion when entering unrecoverable dead-end regions (isolated components, parity violations, or degree-starved corridors).

This project implements a **bitboard-accelerated state evaluator and verifier** ([`search.py`](search.py)) that enforces **provable safety and reachability invariants** prior to state expansion, bounding worst-case search complexity and preventing runaway state exploration under adversarial board configurations.

---

## Key Technical Contributions & Safety Verification

### 1. Whole-Board Bitboard Representation
Every grid cell is mapped to a single bit in a native Python integer, allowing whole-board topological queries, neighbor lookups, and C-level visited-state scans (`bytes.translate`) to execute in $O(1)$ bitwise word operations.

### 2. Deterministic Dead-End & Safety Invariant Verification (`IMPOSSIBLE` Pruning)
Before assigning a heuristic score, [`Cost._judge`](search.py) runs a cascade of formal Necessary Conditions for Solvability, immediately rejecting states that provably violate completion invariants:

- **Bipartite Parity Verification (Checkerboard Invariant):** Tracks alternating dark/light cell counts across the remaining unvisited subgraph to reject states whose parity excess cannot bridge the current head to the terminal checkpoint.
- **Iterative Constraint Propagation (Degree & Edge-Budget Bounds):** Every intermediate unvisited cell requires degree $2$, while the current head and terminal cell require degree $1$. The evaluator iteratively propagates forced edges across the entire bitboard until convergence:
  - Immediately rejects states where any cell falls below its required edge budget (starvation).
  - Immediately rejects states where forced constraints over-saturate a cell ($\text{degree} > 2$ or endpoint $\text{degree} > 1$).
- **Premature Subtour / Cycle Detection:** Traces forced edge chains from the current head to ensure the path cannot be forced into the terminal checkpoint before all remaining live cells are visited.
- **Bit-Parallel Reachability & Cut-Off Detection:** Performs bitwise wavefront propagation from the head to the next checkpoint and continues flooding across all live edges to ensure no unvisited cells are partitioned or isolated.
- **SIMD-Style Stacked Multi-Lane BFS (`board_leg_lanes`):** Packs all future checkpoint-to-checkpoint segments ($k \to k+1$ for all $k \ge \text{goal}$) side-by-side into a single wide integer ("lanes") and executes a **single simultaneous bit-parallel flood fill** across all future legs at once. If the sum of exact shortest-path leg distances exceeds the remaining cell budget (`total > left`), the state is provably unsolvable and pruned.

### 3. Limited Discrepancy Ordering & Bounded Memoization
- **Sibling Detour Tracking:** Ranks legal moves at each parent state and penalizes branches that deviate from top-ranked transitions (`DETOUR_COST`), combining the speed of depth-first greedy commitment with the completeness of best-first search.
- **Bounded State Memoization:** Caches evaluated `(head, free_bitmask)` verdicts up to a hard memory ceiling (`MEMO_LIMIT = 1,500,000`) to guarantee bounded memory usage during adversarial search.

---

## Benchmark Results

Evaluated using the official `zipcomp` best-cost search engine (`max_expansions=10000`, `time_limit=5.0s`) across standard, hard, and adversarial $8 \times 8$ benchmark suites:

| Benchmark Suite | Solver | Puzzles Solved | Total State Expansions | Expansion Reduction | Total Time (s) | Speedup |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Standard (`benchmark_puzzles.json`)** | Baseline (`DraftCost`) | 9 / 9 (100%) | 2,347 | — | 0.2784s | 1.0x |
| **Standard (`benchmark_puzzles.json`)** | **Pathfinders (`search.py`)** | **9 / 9 (100%)** | **619** | **-73.6%** | **0.0420s** | **6.6x** |
| **Hard (`benchmark_puzzles_hard.json`)** | Baseline (`DraftCost`) | 19 / 19 (100%) | 5,964 | — | 0.7332s | 1.0x |
| **Hard (`benchmark_puzzles_hard.json`)** | **Pathfinders (`search.py`)** | **19 / 19 (100%)** | **1,756** | **-70.6%** | **0.1209s** | **6.1x** |
| **Adversarial $8 \times 8$ (`hard_8_8_boards.json`)** | Baseline (`DraftCost`) | 5 / 5 (100%) | 886 | — | 0.0655s | 1.0x |
| **Adversarial $8 \times 8$ (`hard_8_8_boards.json`)** | **Pathfinders (`search.py`)** | **5 / 5 (100%)** | **374** | **-57.8%** | **0.0188s** | **3.5x** |

---

## Repository Structure

```text
├── search.py                    # Core bitboard evaluator, constraint propagator & multi-lane BFS verifier
├── pure_judge.py                # Standalone state judge variant
├── pure_cost.py                 # Lightweight heuristic cost baseline
├── search_lb.py                 # Lower-bound admissible heuristic experiments
├── run_bench.py                 # Standard benchmark evaluation script
├── run_bench_hard.py            # Hard benchmark evaluation script
├── benchmark_puzzles.json       # Standard benchmark puzzle suite (9 boards)
├── benchmark_puzzles_hard.json  # Hard benchmark puzzle suite (19 boards)
├── hard_8_8_boards.json         # Adversarial 8x8 puzzle suite (5 boards)
└── sim/                         # ZipIT simulation engine & helper primitives
    ├── zipcomp/                 # Game engine, state representation, and best-cost search loop
    ├── helpers/                 # Graph & grid helper primitives
    └── config/                  # Engine & level configuration files
```

---

## Quick Start

Run the standard and hard benchmark evaluations (requires Python 3.8+, no external dependencies):

```bash
# Run standard benchmark suite
python3 run_bench.py

# Run hard benchmark suite
python3 run_bench_hard.py
```
