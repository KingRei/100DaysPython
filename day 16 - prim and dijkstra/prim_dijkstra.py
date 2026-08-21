"""Day 16 - Prim and Dijkstra: one heap-greedy skeleton, two algorithms.

Run me:  python prim_dijkstra.py
"""
from heapq import heappush, heappop
from itertools import combinations
INF = float('inf')

NAMES = 'ABCDEFG'
# same graph as day 15, so Prim can be checked against Kruskal's answer
EDGES = [(0, 1, 7), (0, 3, 5), (1, 2, 8), (1, 3, 9), (1, 4, 7),
         (2, 4, 5), (3, 4, 15), (3, 5, 6), (4, 5, 8), (4, 6, 9), (5, 6, 11)]


def build_adj(n, edges, directed=False):
    """adjacency list: adj[u] = [(v, w), ...]"""
    adj = [[] for _ in range(n)]
    for u, v, w in edges:
        adj[u].append((v, w))
        if not directed:
            adj[v].append((u, w))
    return adj


# ------------------------------------------------------------------ the skeleton
def heap_greedy(n, adj, src, key):
    """Grow a tree out of `src`, always settling the cheapest unsettled vertex.

    key(d, w) turns the distance of the settled vertex `u` and the weight of the
    edge u->v into the priority of v.  That single function is the whole
    difference between the two algorithms:

        Prim      key = lambda d, w: w        # cost of the edge itself
        Dijkstra  key = lambda d, w: d + w    # cost of the whole path

    Returns (best, parent, order) where best[v] is that priority once v is
    settled, parent[v] is the vertex it was reached from, and order is the
    sequence in which vertices were settled.
    """
    best = [INF] * n
    parent = [-1] * n
    done = [False] * n
    order = []
    best[src] = 0
    pq = [(0, src)]
    while pq:
        d, u = heappop(pq)
        if done[u]:                     # a stale copy, left behind by a decrease-key
            continue
        done[u] = True
        order.append(u)
        for v, w in adj[u]:
            nd = key(d, w)
            if not done[v] and nd < best[v]:
                best[v] = nd
                parent[v] = u
                heappush(pq, (nd, v))   # lazy deletion: push, never delete
    return best, parent, order


def prim(n, adj, src=0):
    best, parent, order = heap_greedy(n, adj, src, lambda d, w: w)
    tree = [(best[v], parent[v], v) for v in range(n) if parent[v] != -1]
    total = sum(w for w, _, _ in tree)
    reached = sum(1 for v in range(n) if best[v] < INF)
    return sorted(tree), total, reached, order


def dijkstra(n, adj, src=0):
    dist, parent, order = heap_greedy(n, adj, src, lambda d, w: d + w)
    return dist, parent, order


def tree_path_cost(tree_edges, u, v):
    """Cost of the unique u->v path inside a tree, by BFS over the tree."""
    adj = {}
    for a, b, w in tree_edges:
        adj.setdefault(a, []).append((b, w))
        adj.setdefault(b, []).append((a, w))
    stack, seen = [(u, 0)], {u}
    while stack:
        x, acc = stack.pop()
        if x == v:
            return acc
        for y, w in adj.get(x, []):
            if y not in seen:
                seen.add(y)
                stack.append((y, acc + w))
    return INF


def path_to(parent, v):
    out = []
    while v != -1:
        out.append(v)
        v = parent[v]
    return out[::-1]


# ------------------------------------------------------------------ references
def brute_force_mst(n, edges):
    """Cheapest spanning subset of exactly n-1 edges, checked by brute force."""
    best = INF
    for pick in combinations(edges, n - 1):
        seen, comp = {i: i for i in range(n)}, 0

        def find(x):
            while seen[x] != x:
                seen[x] = seen[seen[x]]
                x = seen[x]
            return x
        ok = True
        for u, v, _ in pick:
            ru, rv = find(u), find(v)
            if ru == rv:
                ok = False
                break
            seen[rv] = ru
        if ok:
            best = min(best, sum(w for _, _, w in pick))
    return best


def bellman_ford(n, edges, src, directed=True):
    """Reference shortest paths that also work with negative weights."""
    dist = [INF] * n
    dist[src] = 0
    arcs = [(u, v, w) for u, v, w in edges]
    if not directed:
        arcs += [(v, u, w) for u, v, w in edges]
    for _ in range(n - 1):
        for u, v, w in arcs:
            if dist[u] + w < dist[v]:
                dist[v] = dist[u] + w
    return dist


# ------------------------------------------------------------------ LeetCode 743
def network_delay_time(times, n, k):
    """times: [[u, v, w], ...] 1-indexed directed edges. Returns -1 if unreachable."""
    adj = build_adj(n, [(u - 1, v - 1, w) for u, v, w in times], directed=True)
    dist, _, _ = dijkstra(n, adj, k - 1)
    slowest = max(dist)
    return -1 if slowest == INF else slowest


def show(title):
    print()
    print('=' * 68)
    print(title)
    print('=' * 68)


def main():
    n = len(NAMES)
    adj = build_adj(n, EDGES)

    show('1. the same skeleton, twice')
    print('key = lambda d, w: w        -> Prim      (cost of the edge)')
    print('key = lambda d, w: d + w    -> Dijkstra  (cost of the path)')

    show('2. Prim from A')
    tree, total, reached, order = prim(n, adj)
    print('settle order :', ' '.join(NAMES[u] for u in order))
    for w, p, v in tree:
        print('   took %s%s (%d)' % (NAMES[p], NAMES[v], w))
    print('vertices reached : %d / %d' % (reached, n))
    print('total weight     : %d' % total)
    bf = brute_force_mst(n, EDGES)
    print('brute force over every spanning subset : %d  -> %s'
          % (bf, 'match' if bf == total else 'MISMATCH'))
    print("day 15's Kruskal on this graph also returned 39: different order, same tree weight")

    show('3. Dijkstra from A')
    dist, parent, order = dijkstra(n, adj)
    print('settle order :', ' '.join(NAMES[u] for u in order))
    for v in range(n):
        print('   %s : %-3d  via %s' % (NAMES[v], dist[v],
                                        ' -> '.join(NAMES[x] for x in path_to(parent, v))))
    ref = bellman_ford(n, EDGES, 0, directed=False)
    print('Bellman-Ford agrees :', dist == ref)

    show('4. the two trees are NOT the same tree')
    mst_edges = {tuple(sorted((p, v))) for _, p, v in tree}
    sp_edges = {tuple(sorted((parent[v], v))) for v in range(n) if parent[v] != -1}
    fmt = lambda s: ' '.join(NAMES[a] + NAMES[b] for a, b in sorted(s))
    print('Prim     :', fmt(mst_edges))
    print('Dijkstra :', fmt(sp_edges))
    print('only in the shortest-path tree :', fmt(sp_edges - mst_edges))
    print('only in the MST                :', fmt(mst_edges - sp_edges))
    mst_t = [(p, v, w) for w, p, v in tree]
    for t in (2, 6):
        print('A->%s costs %d if you walk the MST, but the real shortest path is %d'
              % (NAMES[t], tree_path_cost(mst_t, 0, t), dist[t]))
    sp_total = sum(w for u, v, w in EDGES if tuple(sorted((u, v))) in sp_edges)
    print('total weight: MST %d, shortest-path tree %d' % (total, sp_total))
    print('an MST minimises the *total* weight of the tree,')
    print('a shortest-path tree minimises each distance *from the source* - different jobs')

    show('5. a negative edge breaks Dijkstra (and not Prim)')
    #   S->A 5, S->B 2, A->B -10 : B looks cheap, gets settled, and the -10 arrives too late
    neg = [(0, 1, 5), (0, 2, 2), (1, 2, -10)]
    nadj = build_adj(3, neg, directed=True)
    dneg, _, _ = dijkstra(3, nadj, 0)
    ref_neg = bellman_ford(3, neg, 0)
    print('Dijkstra     :', dneg, ' <- wrong')
    print('Bellman-Ford :', ref_neg, ' <- right')
    print('B is settled at 2, and by the time A is settled the -10 edge is ignored')
    print('because B is already marked done;')
    print('"settled" only means "final" when no edge can make a path cheaper.')
    ptree, ptotal, _, _ = prim(3, build_adj(3, neg))
    print('Prim on the same weights is fine, total %d: it never adds distances up'
          % ptotal)

    show('6. LeetCode 743 - Network Delay Time')
    cases = [([[2, 1, 1], [2, 3, 1], [3, 4, 1]], 4, 2, 2),
             ([[1, 2, 1]], 2, 1, 1),
             ([[1, 2, 1]], 2, 2, -1)]
    for times, nn, k, want in cases:
        got = network_delay_time(times, nn, k)
        print('%-34s k=%d -> %-3d (expected %d)' % (str(times), k, got, want))
        assert got == want

    assert total == bf == 39
    assert reached == n
    assert dist == ref
    assert dist[2] == 15 and dist[6] == 22
    assert tree_path_cost(mst_t, 0, 6) == 23 > dist[6]
    assert mst_edges != sp_edges and sp_total > total
    assert dneg[2] == 2 and ref_neg[2] == -5
    print()
    print('all assertions passed')


if __name__ == '__main__':
    main()
