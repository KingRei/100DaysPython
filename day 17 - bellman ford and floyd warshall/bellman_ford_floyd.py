"""Day 17 - negative weights and all pairs: Bellman-Ford, SPFA, Floyd-Warshall.

Run me:  python bellman_ford_floyd.py

Dijkstra settles a vertex and never looks at it again, which is only safe when no edge can
make a path cheaper later - i.e. when every weight is non-negative.  Drop that assumption
and the whole family of algorithms below appears: relax everything V-1 times instead of
settling anything (Bellman-Ford), only re-relax what actually changed (SPFA), or run a DP
over "which vertices may be used in the middle" for every pair at once (Floyd-Warshall).
"""

from collections import deque
from itertools import product

INF = float('inf')

# CLRS's classic example: negative edges, but no negative cycle.
NAMES = 'ABCDE'
N = 5
EDGES = [(0, 1, 6), (0, 3, 7),          # A->B 6   A->D 7
         (1, 2, 5), (1, 3, 8), (1, 4, -4),
         (2, 1, -2),
         (3, 2, -3), (3, 4, 9),
         (4, 0, 2), (4, 2, 7)]


def build_adj(n, edges):
    adj = [[] for _ in range(n)]
    for u, v, w in edges:
        adj[u].append((v, w))           # directed
    return adj


# --------------------------------------------------------------------- 1. Bellman-Ford
def bellman_ford(n, edges, src):
    """Returns (dist, parent, negative_cycle_reachable)."""
    dist = [INF] * n
    parent = [-1] * n
    dist[src] = 0
    for _ in range(n - 1):              # a shortest path uses at most n-1 edges
        changed = False
        for u, v, w in edges:
            if dist[u] + w < dist[v]:   # relax
                dist[v] = dist[u] + w
                parent[v] = u
                changed = True
        if not changed:                 # nothing moved: we are done early
            break
    # one extra round: anything that still improves is fed by a negative cycle
    for u, v, w in edges:
        if dist[u] + w < dist[v]:
            return dist, parent, True
    return dist, parent, False


def rounds_of_bellman_ford(n, edges, src):
    """The dist array after each round - shows the wavefront spreading one edge per round."""
    dist = [INF] * n
    dist[src] = 0
    snapshots = [dist[:]]
    for _ in range(n - 1):
        for u, v, w in edges:
            if dist[u] + w < dist[v]:
                dist[v] = dist[u] + w
        snapshots.append(dist[:])
    return snapshots


def find_negative_cycle(n, edges):
    """Return one negative cycle as a list of vertices, or None."""
    dist = [0] * n                      # 0 everywhere = a virtual source into every vertex
    parent = [-1] * n
    x = -1
    for _ in range(n):
        x = -1
        for u, v, w in edges:
            if dist[u] + w < dist[v]:
                dist[v] = dist[u] + w
                parent[v] = u
                x = v
    if x == -1:
        return None
    for _ in range(n):                  # walk back into the cycle itself
        x = parent[x]
    cycle, v = [], x
    while True:
        cycle.append(v)
        v = parent[v]
        if v == x:
            break
    cycle.append(x)
    cycle.reverse()
    return cycle


# --------------------------------------------------------------------- 2. SPFA
def spfa(n, adj, src):
    """Bellman-Ford that only re-relaxes vertices whose dist actually changed."""
    dist = [INF] * n
    dist[src] = 0
    inq = [False] * n
    relaxed = [0] * n                   # how many times each vertex entered the queue
    q = deque([src])
    inq[src] = True
    pops = 0
    while q:
        u = q.popleft()
        inq[u] = False
        pops += 1
        for v, w in adj[u]:
            if dist[u] + w < dist[v]:
                dist[v] = dist[u] + w
                if not inq[v]:
                    relaxed[v] += 1
                    if relaxed[v] >= n:  # a vertex cannot improve n times without a cycle
                        return dist, pops, True
                    q.append(v)
                    inq[v] = True
    return dist, pops, False


# --------------------------------------------------------------------- 3. Floyd-Warshall
def floyd_warshall(n, edges):
    """All pairs. dist[i][j] = shortest i->j; nxt[i][j] = first hop, for path recovery."""
    dist = [[INF] * n for _ in range(n)]
    nxt = [[-1] * n for _ in range(n)]
    for i in range(n):
        dist[i][i] = 0
    for u, v, w in edges:
        if w < dist[u][v]:
            dist[u][v] = w
            nxt[u][v] = v
    for k in range(n):                  # k MUST be the outer loop
        for i in range(n):
            if dist[i][k] == INF:
                continue
            for j in range(n):
                if dist[i][k] + dist[k][j] < dist[i][j]:
                    dist[i][j] = dist[i][k] + dist[k][j]
                    nxt[i][j] = nxt[i][k]
    return dist, nxt


def floyd_path(nxt, i, j):
    if nxt[i][j] == -1:
        return []
    path = [i]
    while i != j:
        i = nxt[i][j]
        path.append(i)
    return path


def floyd_wrong_loop_order(n, edges):
    """The same three loops with k innermost - a very common and very silent bug."""
    dist = [[INF] * n for _ in range(n)]
    for i in range(n):
        dist[i][i] = 0
    for u, v, w in edges:
        dist[u][v] = min(dist[u][v], w)
    for i in range(n):
        for j in range(n):
            for k in range(n):
                if dist[i][k] + dist[k][j] < dist[i][j]:
                    dist[i][j] = dist[i][k] + dist[k][j]
    return dist


# --------------------------------------------------------------------- 4. LeetCode 787
def find_cheapest_price(n, flights, src, dst, k):
    """At most k stops = at most k + 1 edges = exactly k + 1 rounds of Bellman-Ford."""
    dist = [INF] * n
    dist[src] = 0
    for _ in range(k + 1):
        prev = dist[:]                  # relax from LAST round's values only
        for u, v, w in flights:
            if prev[u] + w < dist[v]:
                dist[v] = prev[u] + w
    return dist[dst] if dist[dst] < INF else -1


def find_cheapest_price_buggy(n, flights, src, dst, k):
    """Same code without the copy: one round can chain several edges together."""
    dist = [INF] * n
    dist[src] = 0
    for _ in range(k + 1):
        for u, v, w in flights:
            if dist[u] + w < dist[v]:
                dist[v] = dist[u] + w
    return dist[dst] if dist[dst] < INF else -1


# --------------------------------------------------------------------- demo
def show(title):
    print('\n' + '=' * 68)
    print(title)
    print('=' * 68)


def fmt(dist, names=NAMES):
    return '  '.join('%s=%s' % (names[i], '-' if d == INF else d) for i, d in enumerate(dist))


def main():
    adj = build_adj(N, EDGES)

    show('1. the graph (directed, negative edges, no negative cycle)')
    for u, v, w in EDGES:
        print('   %s -> %s   %3d' % (NAMES[u], NAMES[v], w))

    show('2. Bellman-Ford from A, round by round')
    snaps = rounds_of_bellman_ford(N, EDGES, 0)
    for r, d in enumerate(snaps):
        print('   after round %d :  %s' % (r, fmt(d)))
    print('   each round lets the answer travel one more edge, so n-1 rounds always suffice')

    dist, parent, neg = bellman_ford(N, EDGES, 0)
    print('   final        :  %s   negative cycle: %s' % (fmt(dist), neg))

    show('3. SPFA - the same answer, fewer relaxations')
    sdist, pops, sneg = spfa(N, adj, 0)
    print('   dist  :  %s' % fmt(sdist))
    print('   queue pops: %d   (plain Bellman-Ford scans all %d edges %d times = %d)'
          % (pops, len(EDGES), N - 1, len(EDGES) * (N - 1)))
    print('   SPFA is Bellman-Ford + "only re-check what changed"; same O(VE) worst case')

    show('4. a negative cycle: change E->C from 7 to 1')
    neg_edges = [(u, v, 1 if (u, v) == (4, 2) else w) for u, v, w in EDGES]
    ndist, nparent, has_neg = bellman_ford(N, neg_edges, 0)
    _, _, spfa_neg = spfa(N, build_adj(N, neg_edges), 0)
    cyc = find_negative_cycle(N, neg_edges)
    print('   B -> E -> C -> B  =  -4 + 1 + (-2)  =  -5   (go round again, pay less)')
    print('   Bellman-Ford extra round still improves -> negative cycle: %s' % has_neg)
    print('   SPFA: some vertex entered the queue n times -> negative cycle: %s' % spfa_neg)
    print('   cycle found: %s' % ' -> '.join(NAMES[v] for v in cyc))
    print('   "shortest path" is undefined here - the answer is minus infinity')

    show('5. Floyd-Warshall - every pair at once')
    fdist, nxt = floyd_warshall(N, EDGES)
    print('        ' + '  '.join('%4s' % c for c in NAMES))
    for i in range(N):
        print('   %s   ' % NAMES[i] + '  '.join(
            '%4s' % ('-' if fdist[i][j] == INF else fdist[i][j]) for j in range(N)))
    print('   row A matches Bellman-Ford from A: %s' % (fdist[0] == dist))
    print('   C -> D path: %s' % ' -> '.join(NAMES[v] for v in floyd_path(nxt, 2, 3)))

    show('6. why k has to be the OUTER loop')
    bad = floyd_wrong_loop_order(N, EDGES)
    diff = [(NAMES[i], NAMES[j], bad[i][j], fdist[i][j])
            for i, j in product(range(N), repeat=2) if bad[i][j] != fdist[i][j]]
    print('   k innermost means "i to j via k" is computed before k itself is finished.')
    print('   pairs that come out wrong: %s' % (diff if diff else 'none on this graph'))
    print('   it disagrees on %d of %d pairs - and it fails silently' % (len(diff), N * N))

    show('7. LeetCode 787 - cheapest flights within K stops')
    flights = [(0, 1, 100), (1, 2, 100), (2, 0, 100), (1, 3, 600), (2, 3, 200)]
    for k in (1, 2):
        good = find_cheapest_price(4, flights, 0, 3, k)
        bad_ = find_cheapest_price_buggy(4, flights, 0, 3, k)
        print('   k=%d stops:  with the copy -> %s      without the copy -> %s'
              % (k, good, bad_))
    print('   k+1 rounds of Bellman-Ford = "use at most k+1 edges", which is exactly the')
    print('   constraint the problem adds. Relaxing in place lets one round chain 0->1->2->3,')
    print('   so the buggy version answers as if more stops were allowed.')

    # ------------------------------------------------------------------ asserts
    assert dist == [0, 2, 4, 7, -2]
    assert sdist == dist
    assert not neg and not sneg
    assert has_neg and spfa_neg
    assert set(cyc) == {1, 2, 4}
    assert fdist[0] == dist
    assert fdist[2][3] == 3 and floyd_path(nxt, 2, 3) == [2, 1, 4, 0, 3]
    assert len(diff) > 0
    assert find_cheapest_price(4, flights, 0, 3, 1) == 700
    assert find_cheapest_price(4, flights, 0, 3, 2) == 400
    assert find_cheapest_price_buggy(4, flights, 0, 3, 1) == 400   # wrong on purpose
    assert find_cheapest_price(3, [(0, 1, 100), (1, 2, 100)], 0, 2, 0) == -1
    print('\nall assertions passed')


if __name__ == '__main__':
    main()
