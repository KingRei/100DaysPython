"""Day 19 - A*, and what happens when a search stops being sequential.

Run me:  python astar.py

Dijkstra orders its frontier by  g(v)  - the cost already paid.
A* orders it by                  f(v) = g(v) + h(v)  - cost paid plus a guess
                                 of what is left.

h = 0                 gives Dijkstra back, exactly.
h admissible          (never overestimates)  ->  the answer is still optimal.
h consistent          (h(u) <= w(u,v) + h(v))  ->  a vertex never needs
                      re-expanding, so "pop it once" stays valid.
h too large           ->  fewer expansions, and a silently sub-optimal path.
h only  (f = h)       ->  greedy best-first: fast, frequently wrong.

The second half is the same question asked in parallel: Dijkstra's strict
one-vertex-at-a-time order is exactly what makes it hard to parallelise, and
delta-stepping is the knob between "too strict" (Dijkstra) and "too much work"
(Bellman-Ford).
"""

import heapq
from collections import deque

# --------------------------------------------------------------------- the map

# '#' is a wall.  A digit is the cost of STEPPING ONTO that cell.
# '5' is mud: passable, but five times the price of open ground.
GRID = [
    "111111111",
    "111111111",
    "115555511",
    "111111111",
    "111111111",
    "111111111",
]
START = (2, 0)
GOAL = (2, 8)
MIN_COST = 1                      # the cheapest any single step can ever be


def parse(grid):
    rows, cols = len(grid), len(grid[0])
    cost = [[None] * cols for _ in range(rows)]
    for r in range(rows):
        for c in range(cols):
            ch = grid[r][c]
            cost[r][c] = None if ch == '#' else int(ch)
    return rows, cols, cost


ROWS, COLS, COST = parse(GRID)
STEPS = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def neighbours(cell):
    r, c = cell
    for dr, dc in STEPS:
        nr, nc = r + dr, c + dc
        if 0 <= nr < ROWS and 0 <= nc < COLS and COST[nr][nc] is not None:
            yield (nr, nc), COST[nr][nc]


def grid_adj():
    """The grid, rewritten as a plain adjacency dict - so the parallel routines
    below can run on any weighted graph, not just this map."""
    adj = {}
    for r in range(ROWS):
        for c in range(COLS):
            if COST[r][c] is not None:
                adj[(r, c)] = list(neighbours((r, c)))
    return adj


# A small weighted graph with a deliberate mix of light and heavy edges.
# The grid is nearly unweighted, so it hides what delta-stepping is for.
WEIGHTED_EDGES = [
    (0, 1, 7), (0, 3, 1), (0, 5, 3), (0, 7, 7),
    (0, 13, 1), (1, 2, 1), (1, 4, 7), (1, 10, 1),
    (2, 6, 1), (2, 11, 1), (3, 2, 1), (4, 12, 1),
    (5, 8, 7), (5, 13, 1), (7, 8, 3), (8, 9, 7),
    (10, 12, 1), (11, 10, 9), (12, 7, 9), (12, 10, 2),
    (12, 11, 3), (13, 2, 2), (13, 11, 7),
]


def weighted_adj(edges=WEIGHTED_EDGES):
    adj = {}
    for a, b, w in edges:
        adj.setdefault(a, []).append((b, w))
        adj.setdefault(b, []).append((a, w))
    return adj


def manhattan(cell, goal=GOAL):
    """|dr| + |dc|, scaled by the cheapest possible step.

    Admissible: any route to the goal needs at least this many steps, and no
    step costs less than MIN_COST, so the true remaining cost is never smaller.
    """
    return (abs(cell[0] - goal[0]) + abs(cell[1] - goal[1])) * MIN_COST


# ------------------------------------------------------- the one search function

def best_first(start, goal, h=None, weight=1.0, greedy=False):
    """Dijkstra, A*, weighted A* and greedy best-first - all the same loop.

    The only thing that changes is the priority:

        greedy      f = h(v)                 ignore what we have paid
        otherwise   f = g(v) + weight*h(v)   weight 1 and h=0 -> Dijkstra
    """
    if h is None:
        h = lambda cell: 0
    g = {start: 0}
    parent = {start: None}
    key = (lambda cell, gv: h(cell)) if greedy else (lambda cell, gv: gv + weight * h(cell))
    pq = [(key(start, 0), start)]
    closed = set()
    expanded = []                       # every vertex we pop and scan
    while pq:
        _, u = heapq.heappop(pq)
        if u in closed:
            continue
        closed.add(u)
        expanded.append(u)
        if u == goal:
            break
        for v, w in neighbours(u):
            ng = g[u] + w
            if v not in g or ng < g[v]:
                g[v] = ng
                parent[v] = u
                heapq.heappush(pq, (key(v, ng), v))
    if goal not in g:
        return None, 0, expanded
    path, cur = [], goal
    while cur is not None:
        path.append(cur)
        cur = parent[cur]
    return path[::-1], g[goal], expanded


def path_cost(path):
    return sum(COST[r][c] for r, c in path[1:])


# ---------------------------------------------------------- checking a heuristic

def is_admissible(h, true_dist):
    """h(v) <= the real remaining cost, for every reachable v."""
    return all(h(v) <= d for v, d in true_dist.items())


def is_consistent(h):
    """h(u) <= w(u,v) + h(v) for every edge - the triangle inequality.

    Consistency is the stronger property, and it is the one that licenses the
    `closed` set: pop a vertex once and its g is final.
    """
    for r in range(ROWS):
        for c in range(COLS):
            if COST[r][c] is None:
                continue
            u = (r, c)
            for v, w in neighbours(u):
                if h(u) > w + h(v):
                    return False, (u, v, h(u), w + h(v))
    return True, None


def true_distances(goal):
    """Dijkstra from the goal on the reversed graph - the exact h we could
    never afford to compute in advance."""
    dist = {goal: 0}
    pq = [(0, goal)]
    while pq:
        d, u = heapq.heappop(pq)
        if d > dist[u]:
            continue
        for v, _ in neighbours(u):
            # a step is charged for the cell it enters, so walking the graph
            # backwards we pay for the cell we are leaving
            nd = d + COST[u[0]][u[1]]
            if v not in dist or nd < dist[v]:
                dist[v] = nd
                heapq.heappush(pq, (nd, v))
    return dist


# ------------------------------------------------------------- delta-stepping

def delta_stepping(adj, start, delta):
    """Dijkstra relaxes one vertex at a time; Bellman-Ford relaxes everything.

    Delta-stepping puts tentative distances into buckets of width delta and
    settles a whole bucket at once - every vertex in it may be relaxed in
    parallel.  Light edges (w <= delta) can move a vertex within the current
    bucket, so the bucket is re-scanned until it stops changing; heavy edges
    are deferred to the end of the phase, when distances can no longer shrink.

    delta -> 0        one vertex per bucket   = Dijkstra
    delta -> infinity one bucket for the lot  = Bellman-Ford

    Returns the distances, the phases (each phase is a batch that could have
    been relaxed in parallel) and a count of edge relaxations - the price paid
    for the wider batches.
    """
    dist = {start: 0}
    buckets = {0: {start}}
    phases = []                      # (bucket index, vertices settled together)
    relaxations = 0
    b = 0
    while buckets:
        if b not in buckets:
            b = min(buckets)
        heavy = []
        while buckets.get(b):
            frontier = buckets.pop(b)
            phases.append((b, sorted(frontier)))
            for u in frontier:           # <- one parallel batch
                for v, w in adj[u]:
                    relaxations += 1
                    nd = dist[u] + w
                    if v not in dist or nd < dist[v]:
                        if w <= delta:
                            if v in dist:
                                old = dist[v] // delta
                                if old in buckets:
                                    buckets[old].discard(v)
                                    if not buckets[old]:
                                        del buckets[old]
                            dist[v] = nd
                            buckets.setdefault(nd // delta, set()).add(v)
                        else:
                            heavy.append((u, v, w))
        for u, v, w in heavy:            # heavy edges, once per phase
            relaxations += 1
            nd = dist[u] + w
            if v not in dist or nd < dist[v]:
                dist[v] = nd
                buckets.setdefault(nd // delta, set()).add(v)
        buckets = {k: st for k, st in buckets.items() if st}
        if buckets:
            b = min(buckets)
    return dist, phases, relaxations


# --------------------------------------------------- level-synchronous BFS

def parallel_bfs_levels(adj, start):
    """BFS written the way a parallel implementation has to write it.

    Not "a queue", but "a frontier": every vertex in the current level is
    independent, so the whole level can be expanded at once.  The sequential
    cost is unchanged - O(V+E) work - but the *depth* is the number of levels,
    which is what a machine with enough cores actually waits for.
    """
    seen = {start}
    frontier = [start]
    levels = []
    while frontier:
        levels.append(sorted(frontier))
        nxt = []
        for u in frontier:               # <- this loop is the parallel part
            for v, _ in adj[u]:
                if v not in seen:
                    seen.add(v)
                    nxt.append(v)
        frontier = nxt
    return levels


# ------------------------------------------- LeetCode 1091, with a heuristic

def shortest_path_binary_matrix(grid):
    """8-directional, unit cost, 0 = open.  A* with the Chebyshev distance.

    max(|dr|, |dc|) is exactly the number of diagonal-ish steps needed on an
    empty board, so it is admissible and consistent - and on an empty board it
    is exact, which means A* walks straight to the goal.
    """
    n = len(grid)
    if grid[0][0] or grid[n - 1][n - 1]:
        return -1
    if n == 1:
        return 1
    goal = (n - 1, n - 1)
    h = lambda c: max(abs(c[0] - goal[0]), abs(c[1] - goal[1]))
    g = {(0, 0): 1}
    pq = [(1 + h((0, 0)), (0, 0))]
    closed = set()
    expanded = 0
    while pq:
        _, u = heapq.heappop(pq)
        if u in closed:
            continue
        closed.add(u)
        expanded += 1
        if u == goal:
            return g[u]
        r, c = u
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                v = (r + dr, c + dc)
                if 0 <= v[0] < n and 0 <= v[1] < n and grid[v[0]][v[1]] == 0:
                    ng = g[u] + 1
                    if v not in g or ng < g[v]:
                        g[v] = ng
                        heapq.heappush(pq, (ng + h(v), v))
    return -1


def bfs_binary_matrix(grid):
    """The usual BFS answer, for cross-checking A*."""
    n = len(grid)
    if grid[0][0] or grid[n - 1][n - 1]:
        return -1
    q = deque([((0, 0), 1)])
    seen = {(0, 0)}
    while q:
        (r, c), d = q.popleft()
        if (r, c) == (n - 1, n - 1):
            return d
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                v = (r + dr, c + dc)
                if (0 <= v[0] < n and 0 <= v[1] < n and grid[v[0]][v[1]] == 0
                        and v not in seen):
                    seen.add(v)
                    q.append((v, d + 1))
    return -1


# ---------------------------------------------------------------------- output

def show(expanded=(), path=(), title=''):
    exp, pth = set(expanded), set(path)
    print(title)
    for r in range(ROWS):
        row = []
        for c in range(COLS):
            cell = (r, c)
            if COST[r][c] is None:
                row.append('##')
            elif cell == START:
                row.append(' S')
            elif cell == GOAL:
                row.append(' G')
            elif cell in pth:
                row.append(' *')
            elif cell in exp:
                row.append(' o')
            elif COST[r][c] > 1:
                row.append(' ~')
            else:
                row.append(' .')
        print('   ' + ''.join(row))


def main():
    print(__doc__.strip())
    print()

    print('1. the map')
    print('----------')
    show(title='   S start, G goal, ## wall, ~ mud (cost 5), . open (cost 1)')
    print()

    dij_path, dij_cost, dij_exp = best_first(START, GOAL)
    a_path, a_cost, a_exp = best_first(START, GOAL, manhattan)

    print('2. Dijkstra is A* with h = 0')
    print('----------------------------')
    show(dij_exp, dij_path, '   Dijkstra: expanded %d cells' % len(dij_exp))
    print()
    show(a_exp, a_path, '   A* (Manhattan): expanded %d cells' % len(a_exp))
    print()
    print('   cost      %d  vs  %d      (identical - the heuristic is admissible)'
          % (dij_cost, a_cost))
    print('   expanded  %d vs %d      (%.0f%% of the work)'
          % (len(dij_exp), len(a_exp), 100.0 * len(a_exp) / len(dij_exp)))
    print('   the guess does not change the answer, only how much of the map')
    print('   we bother to look at.')
    print()

    print('3. is the heuristic allowed?')
    print('----------------------------')
    td = true_distances(GOAL)
    print('   admissible (h <= true remaining cost):', is_admissible(manhattan, td))
    ok, bad = is_consistent(manhattan)
    print('   consistent (h(u) <= w + h(v)):        ', ok)
    print('   admissible alone guarantees an optimal answer; consistency is what')
    print('   makes the closed set safe - without it a vertex can be popped with')
    print('   a g that is not final and has to be re-opened.')
    print()

    print('4. two ways to be fast and wrong')
    print('--------------------------------')
    w_path, w_cost, w_exp = best_first(START, GOAL, manhattan, weight=5.0)
    gr_path, gr_cost, gr_exp = best_first(START, GOAL, manhattan, greedy=True)
    show(w_exp, w_path, '   weighted A*, h x 5: cost %d, expanded %d' % (w_cost, len(w_exp)))
    print()
    show(gr_exp, gr_path, '   greedy best-first (f = h): cost %d, expanded %d'
         % (gr_cost, len(gr_exp)))
    print()
    print('   weight   cost   expanded   optimal?')
    for weight in (1, 2, 3, 4, 5, 8):
        p, c, e = best_first(START, GOAL, manhattan, weight=float(weight))
        print('   %-8d %-6d %-10d %s' % (weight, c, len(e),
                                         'yes' if c == a_cost else
                                         'NO  (%.1fx optimal, bound was %dx)'
                                         % (c / a_cost, weight)))
    print('   greedy   %-6d %-10d %s' % (gr_cost, len(gr_exp),
                                         'NO  (%.1fx optimal, no bound at all)'
                                         % (gr_cost / a_cost)))
    print()
    print('   Everything here finishes. Everything returns a path. Nothing')
    print('   raises. Up to weight 4 the answer happens to stay optimal; at 5')
    print('   the search commits to the mud and never reconsiders.')
    print('   Weighted A* at least comes with a promise - the cost is at most')
    print('   w times optimal - and greedy best-first offers no promise at all.')
    print()

    print('5. delta-stepping: the knob between Dijkstra and Bellman-Ford')
    print('-------------------------------------------------------------')
    wadj = weighted_adj()
    print('   %d vertices, %d edges, weights 1..9 - the grid is too uniform to'
          % (len(wadj), len(WEIGHTED_EDGES)))
    print('   show this, because almost every step on it costs the same.')
    print()
    print('   delta   phases   largest batch   edge relaxations')
    ref, rows = None, []
    for delta in (1, 2, 4, 9, 99):
        dist, phases, relax = delta_stepping(wadj, 0, delta)
        ref = ref or dist
        assert dist == ref, 'delta must not change the answer'
        widths = [len(v) for _, v in phases]
        rows.append((delta, len(phases), max(widths), relax))
        print('   %-7d %-8d %-15d %d' % rows[-1])
    print()
    lo, hi = rows[0], rows[-1]
    print('   Every row computes the same distances. What changes is the shape of')
    print('   the work: delta = %d is almost Dijkstra - %d phases, largest batch %d,'
          % (lo[0], lo[1], lo[2]))
    print('   nothing to do in parallel. delta = %d is almost Bellman-Ford - %d'
          % (hi[0], hi[1]))
    print('   phases, batches of %d, and %.0f%% more edge relaxations to pay for it.'
          % (hi[2], 100.0 * (hi[3] - lo[3]) / lo[3]))
    print('   Fewer, fatter phases is what a parallel machine wants - right up to')
    print('   the point where the redundant work eats the win.')
    print()

    print('6. BFS as a frontier, not a queue')
    print('---------------------------------')
    gadj = grid_adj()
    levels = parallel_bfs_levels(gadj, START)
    widths = [len(l) for l in levels]
    print('   %d levels, widths %s' % (len(levels), widths))
    print('   work  = O(V+E), unchanged from the sequential version')
    print('   depth = %d, the number of levels - that is what a parallel machine' % len(levels))
    print('           waits for, and why a long thin graph parallelises badly.')
    print('   widest frontier = %d vertices that are all independent.' % max(widths))
    print()

    print('7. LeetCode 1091 - shortest path in a binary matrix')
    print('---------------------------------------------------')
    boards = [
        [[0, 1], [1, 0]],
        [[0, 0, 0], [1, 1, 0], [1, 1, 0]],
        [[1, 0, 0], [1, 1, 0], [1, 1, 0]],
    ]
    for b in boards:
        print('   %-34s A* %-3s BFS %s' % (b, shortest_path_binary_matrix(b),
                                           bfs_binary_matrix(b)))
    print('   8-directional movement means the admissible guess is the Chebyshev')
    print('   distance max(|dr|, |dc|), not Manhattan - Manhattan would over-')
    print('   estimate on a diagonal and could return a path that is too long.')
    print()

    print('tests')
    print('-----')
    assert dij_cost == a_cost, 'an admissible heuristic must not change the cost'
    assert len(a_exp) < len(dij_exp), 'A* should look at less of the map'
    assert is_admissible(manhattan, td)
    assert is_consistent(manhattan)[0]
    assert w_cost >= a_cost and gr_cost >= a_cost
    assert path_cost(a_path) == a_cost
    exact = delta_stepping(gadj, START, 1)[0]
    for delta in (2, 5, 99):
        assert delta_stepping(gadj, START, delta)[0] == exact, 'delta changed the answer'
    assert exact[GOAL] == a_cost
    assert sum(len(l) for l in levels) == sum(
        1 for r in range(ROWS) for c in range(COLS) if COST[r][c] is not None)
    for b in boards:
        assert shortest_path_binary_matrix(b) == bfs_binary_matrix(b)
    print('all assertions passed')


if __name__ == '__main__':
    main()
