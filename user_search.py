from helpers import (
    CostFunction,
    # How far to the numbers
    manhattan, manhattan_to_number, manhattan_to_next, bfs_to_next, detour,
    next_gap, bfs_next_to_after, last_number_cell, cells_per_number, pace_error,
    # Around the head (the end of the path)
    unvisited_neighbours, free_degree, on_border, local_free, body_contacts, forced_moves,
    # Can the puzzle still be finished?
    is_hopeless, cut_off_cells, components, head_splits_region, split_cells,
    dead_end_cells, reachable_cells, corridor_cells, border_cells_left,
    cells_in_box, articulation_cells,
)

# =============================================================================
#  Zip cost function
#
#  Every cell is a bit in one Python integer, so whole-board questions
#  ("which cells still need two path edges?", "what can the head reach?")
#  are answered with a handful of integer operations.
#
#  A state is rejected (IMPOSSIBLE) when it provably cannot be finished:
#    * colour count  - the path alternates black/white squares
#    * edge budget   - head and end need 1 path edge, every other cell 2;
#                      cells with exactly enough edges force them, which can
#                      starve neighbours (repeated until nothing changes)
#    * reachability  - next number reachable, nothing cut off, and every
#                      later number-to-number leg still walkable in total
#
#  Live states are ordered by
#      (cells left + DETOUR_DEPTH * detours) * DEPTH_UNIT + tie_break
#  A "detour" is a step that did not take the best-ranked move among its
#  siblings. Charging for detours keeps the search depth-first while the
#  ranking is right, but lets it abandon a branch that needs too many
#  second choices (limited discrepancy search).
# =============================================================================

IMPOSSIBLE = 10 ** 12
DEPTH_UNIT = 1 << 20
DETOUR_DEPTH = 20                       # one detour weighs as much as 20 cells
DETOUR_COST = DETOUR_DEPTH * DEPTH_UNIT

W_OPTIONS = 4096                        # fewer real options from the head first
W_DISTANCE = 1024                       # then closer to the next number
W_CONTACT = 64                          # then hugging walls / the path
MEMO_LIMIT = 1500000


class Cost(CostFunction):
    name = "TitanV6F"

    board_size = None
    _board_id = None
    _fresh_start = False

    # -------------------------------------------------------------------------
    #  entry point
    # -------------------------------------------------------------------------
    def cost(self, state):
        visited = state.cells_visited
        if visited <= 2 or self.board_size is None:
            self._prepare(state, visited)

        left = self.board_size - visited
        if left == 0:
            return 0
        goal = state.next_number
        if goal is None:
            return IMPOSSIBLE

        r, c = state.head
        head = r * self.board_width + c
        free = self._free_cells(state)
        key = (head, free)

        verdicts = self._verdicts
        if len(verdicts) > MEMO_LIMIT:
            verdicts.clear()
            self._detours.clear()
            self._ranks.clear()
        base = verdicts.get(key)
        if base is None:
            base = verdicts[key] = self._judge(free, head, goal, left)
        if base >= IMPOSSIBLE:
            return base

        detours = self._detours.get(key)
        if detours is None:
            detours = 0 if visited == 1 else self._count_detours(
                head, free, goal, left, visited)
            self._detours[key] = detours
        return base + detours * DETOUR_COST

    # -------------------------------------------------------------------------
    #  per-board setup (once per puzzle, not once per early state)
    # -------------------------------------------------------------------------
    def _prepare(self, state, visited):
        board_id = (state.rows, state.cols, state.n_numbers,
                    state.cell_of(1), state.cell_of(state.n_numbers))
        if (self.board_size is None or board_id != self._board_id
                or (visited == 1 and self._fresh_start)):
            self._build_board(state)
            self._board_id = board_id
            self._verdicts = {}
            self._detours = {}
            self._ranks = {}
            self._fresh_start = visited == 1
        elif visited == 1:
            self._fresh_start = True

    # -------------------------------------------------------------------------
    #  board tables
    # -------------------------------------------------------------------------
    def _build_board(self, state):
        """Static bitmask tables for one puzzle, built once per board."""
        rows, cols = state.rows, state.cols
        size = rows * cols
        last = state.n_numbers
        self.board_width, self.board_size, self.board_last = cols, size, last

        # -- numbers --------------------------------------------------------
        number_at = [0] * size
        number_bit = [0] * (last + 2)
        numbers = 0
        for k in range(1, last + 1):
            r, c = state.cell_of(k)
            idx = r * cols + c
            number_at[idx] = k
            number_bit[k] = 1 << idx
            numbers |= 1 << idx
        self.board_number_at, self.board_number_bit = number_at, number_bit
        self.board_end_bit = number_bit[last]

        # -- grid: neighbours, edges, colours --------------------------------
        reach = [0] * size
        around = [None] * size
        right_edges = down_edges = dark = 0
        cell_bits = []
        for r in range(rows):
            for c in range(cols):
                idx = r * cols + c
                bit = 1 << idx
                cell_bits.append((bit, (r, c)))
                if not (r + c) & 1:
                    dark |= bit
                mask = 0
                near = []
                for rr, cc in state.neighbours((r, c)):
                    j = rr * cols + cc
                    near.append(j)
                    a, b = number_at[idx], number_at[j]
                    if a and b and a - b != 1 and b - a != 1:
                        continue                    # non-consecutive numbers never touch
                    mask |= 1 << j
                    if j == idx + 1:
                        right_edges |= bit
                    elif j == idx + cols:
                        down_edges |= bit
                reach[idx] = mask
                around[idx] = near
        self.board_reach, self.board_around, self.board_cell_bits = reach, around, cell_bits
        self.board_right_edges, self.board_down_edges = right_edges, down_edges
        self.board_dark = dark
        self.board_end_dark = bool(self.board_end_bit & dark)

        # -- legs -------------------------------------------------------------
        # walls_for[k]: other numbers block the way while heading for k
        # leg_cells[k]: cells usable on the leg k -> k+1
        self.board_walls_for = [numbers & ~number_bit[k] for k in range(last + 2)]
        leg_cells = [0] * (last + 2)
        for k in range(1, last):
            leg_cells[k] = ~(numbers & ~(number_bit[k] | number_bit[k + 1]))
        self.board_leg_cells = leg_cells

        # leg_lanes[n]: all legs n->n+1 ... last-1->last stacked side by side,
        # one board-sized lane each, so a single flood advances every leg.
        full = (1 << size) - 1
        lanes = [None] * (last + 1)
        for first in range(1, last):
            copies = starts = goals = usable = 0
            for lane, k in enumerate(range(first, last)):
                shift = lane * size
                copies |= 1 << shift
                starts |= number_bit[k] << shift
                goals |= number_bit[k + 1] << shift
                usable |= (full & leg_cells[k]) << shift
            lanes[first] = (copies, starts, goals, usable)
        self.board_leg_lanes = lanes

        # -- fast scan of visited cells (one C-level pass when possible) ------
        self.board_scan_order = [cell for _, cell in reversed(cell_bits)]
        self.board_scan_table = bytes.maketrans(b'\x00\x01', b'10')
        self.board_scan_fast = False
        try:
            quick = int(bytes(map(state.is_visited, self.board_scan_order))
                        .translate(self.board_scan_table), 2)
            self.board_scan_fast = quick == self._free_slow(state)
        except Exception:
            self.board_scan_fast = False

    def _free_slow(self, state):
        seen = state.is_visited
        free = 0
        for bit, cell in self.board_cell_bits:
            if not seen(cell):
                free |= bit
        return free

    def _free_cells(self, state):
        if self.board_scan_fast:
            return int(bytes(map(state.is_visited, self.board_scan_order))
                       .translate(self.board_scan_table), 2)
        return self._free_slow(state)

    # -------------------------------------------------------------------------
    #  detours: how often this path skipped the best-ranked sibling
    # -------------------------------------------------------------------------
    def _count_detours(self, head, free, goal, left, visited):
        here = self.board_number_at[head]
        prev_goal = here if here else goal          # what the parent was heading for
        before = free | (1 << head)                 # parent's free cells
        best = None
        for prev in self.board_around[head]:
            if (before >> prev) & 1:
                continue                            # the previous head is visited
            parent = (prev, before)
            spent = self._detours.get(parent)
            if spent is None:
                if visited == 2 and self.board_number_bit[1] == 1 << prev:
                    spent = 0                       # parent is the starting cell
                else:
                    continue
            ranks = self._ranks.get(parent)
            if ranks is None:
                ranks = self._rank_moves(prev, before, prev_goal, left + 1)
            step = ranks.get(head)
            if step is not None and (best is None or spent + step < best):
                best = spent + step
        return 0 if best is None else best

    def _rank_moves(self, prev, before, goal, left):
        """Score every legal move out of `prev`; 0 = best, 1 = second, ..."""
        moves = self.board_reach[prev] & before & ~(1 << prev) & ~self.board_walls_for[goal]
        if left > 1:
            moves &= ~self.board_end_bit
        verdicts = self._verdicts
        target = self.board_number_bit[goal]
        scored = []
        while moves:
            bit = moves & -moves
            moves ^= bit
            cell = bit.bit_length() - 1
            rest = before & ~bit & ~(1 << prev)
            if bit == target:
                nxt = goal + 1 if goal < self.board_last else None
            else:
                nxt = goal
            key = (cell, rest)
            value = verdicts.get(key)
            if value is None:
                if left == 1:
                    value = 0
                elif nxt is None:
                    value = IMPOSSIBLE
                else:
                    value = self._judge(rest, cell, nxt, left - 1)
                verdicts[key] = value
            if value < IMPOSSIBLE:
                scored.append((value, cell))
        scored.sort()
        ranks = {cell: place for place, (_, cell) in enumerate(scored)}
        self._ranks[(prev, before)] = ranks
        return ranks

    # -------------------------------------------------------------------------
    #  the verdict on one position: IMPOSSIBLE, or depth + tie-break
    # -------------------------------------------------------------------------
    def _judge(self, free, head, goal, left):
        width = self.board_width
        head_bit = 1 << head
        live = free | head_bit
        end_bit = self.board_end_bit

        # legal next steps (other numbers are walls; the end comes last)
        steps = self.board_reach[head] & free & ~self.board_walls_for[goal]
        if left > 1:
            steps &= ~end_bit
        if not steps:
            return IMPOSSIBLE

        # colour count: the remaining path alternates dark/light squares
        excess = 2 * bin(live & self.board_dark).count('1') - (left + 1)
        if head_bit & self.board_dark:
            if excess != (1 if self.board_end_dark else 0):
                return IMPOSSIBLE
        elif excess != (0 if self.board_end_dark else -1):
            return IMPOSSIBLE

        if left == 1:
            return 1

        # possible path edges among live cells; the head keeps legal ones only
        east = self.board_right_edges & live & (live >> 1)
        south = self.board_down_edges & live & (live >> width)
        banned = self.board_reach[head] & live & ~steps
        if banned:
            if banned & (head_bit << 1):
                east &= ~head_bit
            if banned & (head_bit >> 1):
                east &= ~(head_bit >> 1)
            if banned & (head_bit << width):
                south &= ~head_bit
            if banned & (head_bit >> width):
                south &= ~(head_bit >> width)

        # edge budget, propagated over the whole board at once
        need_one = head_bit | end_bit
        need_two = live & ~need_one
        while True:
            west = east << 1
            north = south << width
            one = east
            two = one & west; one |= west
            three = two & south; two |= one & south; one |= south
            three |= two & north; two |= one & north; one |= north
            if (need_two & ~two) or (need_one & ~one):
                return IMPOSSIBLE                    # a cell is short of edges
            tight = (need_two & ~three) | (need_one & ~two)
            fixed_e = east & (tight | (tight >> 1))
            fixed_s = south & (tight | (tight >> width))
            fixed_w = fixed_e << 1
            fixed_n = fixed_s << width
            one = fixed_e
            two = one & fixed_w; one |= fixed_w
            three = two & fixed_s; two |= one & fixed_s; one |= fixed_s
            three |= two & fixed_n; two |= one & fixed_n; one |= fixed_n
            if three or (need_one & two):
                return IMPOSSIBLE                    # a cell is forced too often
            full = (need_one & one) | (need_two & two)
            new_e = fixed_e | (east & ~(full | (full >> 1)))
            new_s = fixed_s | (south & ~(full | (full >> width)))
            if new_e == east and new_s == south:
                break
            east, south = new_e, new_s
        west = east << 1
        north = south << width

        # true distance to the next number
        target = self.board_number_bit[goal]
        open_ = live & ~self.board_walls_for[goal]
        wave = seen = head_bit
        dist = 0
        while not (seen & target):
            spread = (((wave & east) << 1) | ((wave & west) >> 1)
                      | ((wave & south) << width) | ((wave & north) >> width))
            wave = spread & open_ & ~seen
            if not wave:
                return IMPOSSIBLE
            seen |= wave
            dist += 1

        # nothing may be cut off (continues from what the search above reached)
        wave = seen
        while wave:
            spread = (((wave & east) << 1) | ((wave & west) >> 1)
                      | ((wave & south) << width) | ((wave & north) >> width))
            wave = spread & live & ~seen
            seen |= wave
        if seen != live:
            return IMPOSSIBLE

        # all later legs at once; their lengths must fit in the cells left
        if goal < self.board_last:
            copies, starts, goals, usable = self.board_leg_lanes[goal]
            lane_e = east * copies
            lane_s = south * copies
            lane_w = lane_e << 1
            lane_n = lane_s << width
            lane_open = (live * copies) & usable
            wave = seen = starts
            total = dist
            rounds = 0
            while (seen & goals) != goals:
                rounds += 1
                spread = (((wave & lane_e) << 1) | ((wave & lane_w) >> 1)
                          | ((wave & lane_s) << width) | ((wave & lane_n) >> width))
                wave = spread & lane_open & ~seen
                if not wave:
                    return IMPOSSIBLE
                seen |= wave
                arrived = wave & goals
                if arrived:
                    total += rounds * bin(arrived).count('1')
            if total > left:
                return IMPOSSIBLE
        elif dist > left:
            return IMPOSSIBLE

        # tie-break: fewest options, then nearest goal, then most contact
        options = (((head_bit & east) << 1) | ((head_bit & west) >> 1)
                   | ((head_bit & south) << width) | ((head_bit & north) >> width))
        contact = self.board_reach[head] & ~free
        return (left * DEPTH_UNIT
                + bin(options).count('1') * W_OPTIONS
                + dist * W_DISTANCE
                - bin(contact).count('1') * W_CONTACT)