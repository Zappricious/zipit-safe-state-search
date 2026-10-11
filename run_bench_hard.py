import sys, json
sys.path.insert(0, "sim")
from zipcomp.game import Board
from zipcomp.engine import best_cost_search
from helpers import (CostFunction, manhattan, manhattan_to_next, free_degree,
                     is_hopeless, pace_error, on_border)

class DraftCost(CostFunction):
    name = "draft"
    def cost(self, state):
        cost = 0
        cost += state.cells_left * 10
        left = state.cells_left
        nxt = state.next_number
        if state.is_solved:
            return 0
        if nxt is None:
            return 10**9 if left > 0 else 0
        deg = free_degree(state)
        if deg == 0:
            return 10**9
        m_next = manhattan_to_next(state)
        if m_next > left:
            return 10**9
        if is_hopeless(state):
            return 10**9
        pe = pace_error(state)
        cost = deg * 28 + pe * 45 + m_next * 3
        if state.on_number:
            cost += pe * 80
        if not on_border(state):
            cost += 10
        return cost

def evaluate(cost_cls, max_exp=10000, time_lim=5.0):
    with open("benchmark_puzzles_hard.json") as f:
        puzzles = json.load(f)

    print(f"{'Puzzle':<15} {'Status':<16} {'Expansions':<12} {'Time (s)':<10}")
    print("-" * 55)
    total_exp = 0
    total_time = 0
    solved_count = 0
    for p in puzzles:
        b = Board.from_dict(p)
        res = best_cost_search(b, cost_cls(), max_expansions=max_exp, time_limit=time_lim)
        st = res["status"]
        exp = res["expansions"]
        el = res["elapsed"]
        if st == "solved":
            solved_count += 1
        print(f"{p['id']:<15} {st:<16} {exp:<12} {el:<10.4f}")
        total_exp += exp
        total_time += el

    print("-" * 55)
    print(f"Solved: {solved_count}/{len(puzzles)} | Total Expansions: {total_exp} | Total Time: {total_time:.4f}s")

if __name__ == "__main__":
    from search import Cost
    print("=== Optimized Solver: Pathfinders (search.py) ===")
    evaluate(Cost)
    print("\n=== Baseline Solver: DraftCost ===")
    evaluate(DraftCost)

