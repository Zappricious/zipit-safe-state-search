from helpers import CostFunction


DEAD = 10**12

STEP = 1 << 20              
DSTEP = 20 * STEP           
W_MOBILITY = 4096           
W_DIST = 1024               
W_WALL = 64                 
CACHE_LIMIT = 1500000       


class Cost(CostFunction):
    name = "pathfinders"

    _t = None               
    _key = None
    _built_cv1 = False




    def _build(self, state):
        R, C, K = state.rows, state.cols, state.n_numbers
        N = R * C

        numof = [0] * N
        numbit = [0] * (K + 2)
        nummask = 0
        for k in range(1, K + 1):
            r, c = state.cell_of(k)
            i = r * C + c
            numof[i] = k
            numbit[k] = 1 << i
            nummask |= 1 << i

        nbm = [0] * N           
        HE = VE = black = 0     
        cells = []
        for r in range(R):
            for c in range(C):
                i = r * C + c
                bit = 1 << i
                cells.append((bit, (r, c)))
                if (r + c) & 1 == 0:
                    black |= bit
                m = 0
                for rr, cc in state.neighbours((r, c)):
                    j = rr * C + cc
                    a, b = numof[i], numof[j]
                    if a and b and abs(a - b) != 1:
                        continue            
                    m |= 1 << j
                    if j == i + 1:
                        HE |= bit
                    elif j == i + C:
                        VE |= bit
                nbm[i] = m

    
        block = [nummask & ~numbit[k] for k in range(K + 2)]
        pair_ok = [0] * (K + 2)
        for k in range(1, K):
            pair_ok[k] = ~(nummask & ~(numbit[k] | numbit[k + 1]))

    
        rcells = [cell for _, cell in reversed(cells)]
        tbl = bytes.maketrans(b"\x00\x01", b"10")
        fast = False
        try:
            u1 = int(bytes(map(state.is_visited, rcells)).translate(tbl), 2)
            u2 = 0
            for bit, cell in cells:
                if not state.is_visited(cell):
                    u2 |= bit
            fast = (u1 == u2)
        except Exception:
            fast = False

    
        self._numof = numof
        self._adj = [[rr * C + cc for rr, cc in state.neighbours(divmod(i, C))]
                     for i in range(N)]
        self._disc = {}         
        self._rank = {}         
        self._evalc = {}        

        endbit = numbit[K]
        self._t = (C, N, K, nbm, HE, VE, black, cells, numbit, block,
                   pair_ok, endbit, (endbit & black) != 0, rcells, tbl, fast)

    def _ensure_built(self, state, cv):
        """Rebuild the tables once per board, not once per early state."""
        key = (state.rows, state.cols, state.n_numbers,
               state.cell_of(1), state.cell_of(state.n_numbers))
        if self._t is None or key != self._key or (cv == 1 and self._built_cv1):
            self._build(state)
            self._key = key
            self._built_cv1 = (cv == 1)
        elif cv == 1:
            self._built_cv1 = True




    def cost(self, state):
        cv = state.cells_visited
        if cv <= 2 or self._t is None:
            self._ensure_built(state, cv)
        (C, N, K, nbm, HE, VE, black, cells, numbit, block,
         pair_ok, endbit, end_black, rcells, tbl, fast) = self._t

        cl = N - cv
        if cl == 0:
            return 0
        nn = state.next_number
        if nn is None:
            return DEAD

        hr, hc = state.head
        hi = hr * C + hc

    
        if fast:
            U = int(bytes(map(state.is_visited, rcells)).translate(tbl), 2)
        else:
            isv = state.is_visited
            U = 0
            for bit, cell in cells:
                if not isv(cell):
                    U |= bit

    
        key = (hi, U)
        evalc = self._evalc
        if len(evalc) > CACHE_LIMIT:
            evalc.clear()
            self._disc.clear()
            self._rank.clear()
        base = evalc.get(key)
        if base is None:
            base = self._core(U, hi, nn, cl)
            evalc[key] = base
        if base >= DEAD:
            return base

    
        disc = self._disc
        d0 = disc.get(key)
        if d0 is None:
            d0 = 0 if cv == 1 else self._discrepancy(key, hi, U, nn, cl, cv)
            disc[key] = d0
        return base + d0 * DSTEP




    def _discrepancy(self, key, hi, U, nn, cl, cv):
        """Fewest non-best moves over the possible previous heads of this state."""
        numbit = self._t[8]
        hb = 1 << hi
        k_here = self._numof[hi]
        pn = k_here if k_here else nn         
        Up = U | hb
        disc_map = self._disc
        rank_map = self._rank

        best = None
        for p in self._adj[hi]:
            if (Up >> p) & 1:
                continue                      
            pkey = (p, Up)
            pd = disc_map.get(pkey)
            if pd is None:
                if cv == 2 and numbit[1] == (1 << p):
                    pd = 0
                else:
                    continue
            rk = rank_map.get(pkey)
            if rk is None:
                rk = self._rank_children(p, Up, pn, cl + 1)
            r = rk.get(hi)
            if r is None:
                continue
            if best is None or pd + r < best:
                best = pd + r
        return 0 if best is None else best

    def _rank_children(self, p, Up, pn, pcl):
        (C, N, K, nbm, HE, VE, black, cells, numbit, block,
         pair_ok, endbit, end_black, rcells, tbl, fast) = self._t
        evalc = self._evalc
        moves = nbm[p] & Up & ~(1 << p) & ~block[pn]
        if pcl > 1:
            moves &= ~endbit

        scored = []
        m = moves
        while m:
            wb = m & -m
            m ^= wb
            w = wb.bit_length() - 1
            U2 = Up & ~wb & ~(1 << p)
            if wb == numbit[pn]:
                n2 = pn + 1 if pn < K else None
            else:
                n2 = pn
            ck = (w, U2)
            v = evalc.get(ck)
            if v is None:
                if pcl == 1:
                    v = 0
                elif n2 is None:
                    v = DEAD
                else:
                    v = self._core(U2, w, n2, pcl - 1)
                evalc[ck] = v
            if v < DEAD:
                scored.append((v, w))

        scored.sort()
        rk = {w: i for i, (_, w) in enumerate(scored)}
        self._rank[(p, Up)] = rk
        return rk




    def _core(self, U, hi, nn, cl):
        (C, N, K, nbm, HE, VE, black, cells, numbit, block,
         pair_ok, endbit, end_black, rcells, tbl, fast) = self._t
        hb = 1 << hi
        V = U | hb

    
        diff = 2 * bin(V & black).count("1") - (cl + 1)
        if hb & black:
            if diff != (1 if end_black else 0):
                return DEAD
        elif diff != (0 if end_black else -1):
            return DEAD

    
        LEGAL = nbm[hi] & U & ~block[nn]
        if cl > 1:
            LEGAL &= ~endbit
        if not LEGAL:
            return DEAD
        if cl == 1:
            return 1

    
        H = HE & V & (V >> 1)
        Vt = VE & V & (V >> C)
        bad = nbm[hi] & V & ~LEGAL
        if bad:
            if bad & (hb << 1): H &= ~hb
            if bad & (hb >> 1): H &= ~(hb >> 1)
            if bad & (hb << C): Vt &= ~hb
            if bad & (hb >> C): Vt &= ~(hb >> C)

    
        req1 = hb | endbit
        req2 = V & ~req1
        while True:
            x2 = H << 1
            x4 = Vt << C
            s1 = H
            s2 = s1 & x2; s1 |= x2
            s3 = s2 & Vt; s2 |= s1 & Vt; s1 |= Vt
            s3 |= s2 & x4; s2 |= s1 & x4; s1 |= x4
            if (req2 & ~s2) or (req1 & ~s1):
                return DEAD                                  
            F = (req2 & ~s3) | (req1 & ~s2)                   
            FH = H & (F | (F >> 1))
            FV = Vt & (F | (F >> C))
            y2 = FH << 1
            y4 = FV << C
            f1 = FH
            f2 = f1 & y2; f1 |= y2
            f3 = f2 & FV; f2 |= f1 & FV; f1 |= FV
            f3 |= f2 & y4; f2 |= f1 & y4; f1 |= y4
            if f3 or (req1 & f2):
                return DEAD                                  
            S = (req1 & f1) | (req2 & f2)                     
            nH = FH | (H & ~(S | (S >> 1)))
            nV = FV | (Vt & ~(S | (S >> C)))
            if nH == H and nV == Vt:
                break
            H = nH
            Vt = nV

        HL = H << 1
        VU = Vt << C

    
        front = seen = hb
        while front:
            x = (((front & H) << 1) | ((front & HL) >> 1)
                 | ((front & Vt) << C) | ((front & VU) >> C))
            front = x & ~seen
            seen |= front
        if seen != V:
            return DEAD

    
        target = numbit[nn]
        allowed = V & ~block[nn]
        front = seen = hb
        dist = 0
        while not (seen & target):
            x = (((front & H) << 1) | ((front & HL) >> 1)
                 | ((front & Vt) << C) | ((front & VU) >> C))
            front = x & allowed & ~seen
            if not front:
                return DEAD
            seen |= front
            dist += 1

    
        lb = dist
        for k in range(nn, K):
            tgt = numbit[k + 1]
            al = V & pair_ok[k]
            front = seen = numbit[k]
            while not (seen & tgt):
                x = (((front & H) << 1) | ((front & HL) >> 1)
                     | ((front & Vt) << C) | ((front & VU) >> C))
                front = x & al & ~seen
                if not front:
                    return DEAD
                seen |= front
                lb += 1
        if lb > cl:
            return DEAD

    
        hn = (((hb & H) << 1) | ((hb & HL) >> 1) | ((hb & Vt) << C) | ((hb & VU) >> C))
        mob = bin(hn).count("1")
        contacts = bin(nbm[hi] & ~U).count("1")
        return cl * STEP + mob * W_MOBILITY + dist * W_DIST - contacts * W_WALL