from __future__ import annotations
import math
from helpers import (
    CostFunction, manhattan, manhattan_to_next, bfs_to_next, bfs_next_to_after, next_gap,
    free_degree, local_free, body_contacts, forced_moves,
    components, dead_end_cells, cut_off_cells, head_splits_region, split_cells,
    unvisited_neighbours,
)

BIG = 10**9

class Cost(CostFunction):
    name = "Pathfinders"

    def cost(self, state):
        if state.is_solved:
            return 0

        left = state.cells_left
        nxt = state.next_number
        if nxt is None:
            return BIG if left > 0 else 0

        deg = free_degree(state)
        if deg == 0:
            return BIG

        fm = forced_moves(state)
        if fm >= 2:
            return BIG

        m_next = manhattan_to_next(state)
        if m_next > left:
            return BIG

        k = state.n_numbers
        tot_m = m_next
        if nxt < k:
            for cur in range(nxt, k):
                tot_m += manhattan(state.cell_of(cur), state.cell_of(cur + 1))
        if tot_m > left:
            return BIG

        b_next = bfs_to_next(state)
        if b_next is None or b_next > left:
            return BIG
        dt = b_next - m_next
        if tot_m + dt > left:
            return BIG

        b_after = bfs_next_to_after(state)
        if b_after is None:
            return BIG
        if b_after > 0:
            dt_after = b_after - next_gap(state)
            if tot_m + dt + dt_after > left:
                return BIG

        if components(state) > 1:
            return BIG
        if dead_end_cells(state) > 0:
            return BIG
        if cut_off_cells(state) > 0:
            return BIG
        if split_cells(state) > 0:
            return BIG
        if deg >= 2 and head_splits_region(state):
            return BIG

        hr, hc = state.head
        head_color = (hr + hc) & 1
        gr, gc = state.cell_of(k)
        goal_color = (gr + gc) & 1
        if (head_color + left) & 1 != goal_color:
            return BIG

        cost = 0.5 * m_next + 2.0 * dt
        cost += deg * 2.5
        cost -= body_contacts(state) * 1.5
        cost += local_free(state) * 1.0
        if fm == 1:
            cost -= 5.0

        for cell in unvisited_neighbours(state):
            cost += free_degree(state, cell) * 0.8

        return cost
