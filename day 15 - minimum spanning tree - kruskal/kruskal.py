"""Day 15 - Minimum spanning tree with Kruskal's algorithm.

Kruskal is the whole of Day 14's Union-Find put to work: sort every edge by
weight, walk the sorted list, and keep an edge only when its two endpoints are
still in different components.

Run it:  python3 kruskal.py
"""
from itertools import combinations


# --------------------------------------------------------------------------
# Union-Find (Day 14), trimmed to what Kruskal needs
# --------------------------------------------------------------------------
class UnionFind:
    """Disjoint set union with path compression and union by rank."""

    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n
        self.count = n                     # number of components

    def find(self, x):
        root = x
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[x] != root:      # path compression
            self.parent[x], x = root, self.parent[x]
        return root

    def union(self, x, y):
        """Merge the two sets. Return False when x and y were already together."""
        rx, ry = self.find(x), self.find(y)
        if rx == ry:
            return False                   # adding this edge would close a cycle
        if self.rank[rx] < self.rank[ry]:
            rx, ry = ry, rx
        self.parent[ry] = rx
        if self.rank[rx] == self.rank[ry]:
            self.rank[rx] += 1
        self.count -= 1
        return True


# --------------------------------------------------------------------------
# Kruskal
# --------------------------------------------------------------------------
def kruskal(n, edges):
    """Minimum spanning tree (or forest) of an undirected weighted graph.

    n      - number of vertices, labelled 0..n-1
    edges  - list of (weight, u, v)

    Returns (chosen_edges, total_weight, trace) where trace records, in order,
    every edge the scan looked at, what happened to it, how many components are
    left, and how many edges had already been chosen at that moment.
    """
    uf = UnionFind(n)
    chosen, total, trace = [], 0, []

    for w, u, v in sorted(edges):          # cheapest first - the whole idea
        if uf.union(u, v):                 # different components -> safe to take
            chosen.append((w, u, v))
            total += w
            trace.append(('take', w, u, v, uf.count, len(chosen)))
            if len(chosen) == n - 1:       # a spanning tree has exactly n-1 edges
                trace.append(('stop', w, u, v, uf.count, len(chosen)))
                break
        else:                              # same component -> would close a cycle
            trace.append(('skip', w, u, v, uf.count, len(chosen)))

    return chosen, total, trace


def is_connected(n, edges):
    """True when the edge list connects all n vertices - used to sanity-check."""
    uf = UnionFind(n)
    for w, u, v in edges:
        uf.union(u, v)
    return uf.count == 1


# --------------------------------------------------------------------------
# brute force, only used to prove Kruskal right on a small graph
# --------------------------------------------------------------------------
def brute_force_mst(n, edges):
    """Try every subset of n-1 edges and keep the cheapest spanning one."""
    best, best_set = None, None
    for combo in combinations(edges, n - 1):
        if is_connected(n, combo):
            w = sum(e[0] for e in combo)
            if best is None or w < best:
                best, best_set = w, combo
    return best, best_set


# --------------------------------------------------------------------------
# LeetCode 1584 - Min Cost to Connect All Points
# --------------------------------------------------------------------------
def min_cost_connect_points(points):
    """Manhattan distance between every pair of points, then Kruskal.

    The graph is complete, so there are n(n-1)/2 edges; sorting them dominates
    at O(n^2 log n). Prim with a heap is the better fit for dense graphs, but
    Kruskal is the shorter answer and passes comfortably at n <= 1000.
    """
    n = len(points)
    edges = []
    for i, j in combinations(range(n), 2):
        (x1, y1), (x2, y2) = points[i], points[j]
        edges.append((abs(x1 - x2) + abs(y1 - y2), i, j))
    _, total, _ = kruskal(n, edges)
    return total


# --------------------------------------------------------------------------
# demo
# --------------------------------------------------------------------------
NAMES = 'ABCDEFG'
GRAPH = [(7, 0, 1), (5, 0, 3), (8, 1, 2), (9, 1, 3), (7, 1, 4),
         (5, 2, 4), (15, 3, 4), (6, 3, 5), (8, 4, 5), (9, 4, 6), (11, 5, 6)]


def name(u):
    return NAMES[u]


def show(edges):
    return ' '.join('%s%s(%d)' % (name(u), name(v), w) for w, u, v in edges)


def main():
    n = len(NAMES)

    print('=' * 68)
    print('1. the graph')
    print('=' * 68)
    print('%d vertices, %d edges' % (n, len(GRAPH)))
    print(show(sorted(GRAPH)))
    print()

    print('=' * 68)
    print('2. Kruskal, edge by edge')
    print('=' * 68)
    chosen, total, trace = kruskal(n, GRAPH)
    print('%-6s %-8s %-6s %s' % ('action', 'edge', 'weight', 'components left'))
    for what, w, u, v, comp, _ in trace:
        if what == 'stop':
            print('       n-1 = %d edges chosen, the rest of the list is dead weight' % (n - 1))
            continue
        print('%-6s %-8s %-6d %d' % (what, name(u) + '-' + name(v), w, comp))
    print()
    print('MST edges : %s' % show(chosen))
    print('total     : %d' % total)
    print()

    print('=' * 68)
    print('3. is it really the minimum? brute force says:')
    print('=' * 68)
    best, best_set = brute_force_mst(n, GRAPH)
    print('cheapest of all %d-edge spanning subsets: %d' % (n - 1, best))
    print('one such tree: %s' % show(sorted(best_set)))
    print('Kruskal total: %d  -> %s' % (total, 'match' if best == total else 'MISMATCH'))
    print()

    print('=' * 68)
    print('4. a disconnected graph gives a spanning *forest*')
    print('=' * 68)
    broken = [e for e in GRAPH if 6 not in (e[1], e[2])]   # cut G loose
    f_edges, f_total, _ = kruskal(n, broken)
    print('edges kept: %d (a tree would need %d)' % (len(f_edges), n - 1))
    print('total     : %d, so G stays on its own island' % f_total)
    print()

    print('=' * 68)
    print('5. LeetCode 1584 - Min Cost to Connect All Points')
    print('=' * 68)
    cases = [([[0, 0], [2, 2], [3, 10], [5, 2], [7, 0]], 20),
             ([[3, 12], [-2, 5], [-4, 1]], 18),
             ([[0, 0]], 0)]
    for pts, want in cases:
        got = min_cost_connect_points(pts)
        print('%-42s -> %-3d (expected %d)' % (str(pts), got, want))
    print()

    # ---------------------------------------------------------------- tests
    assert total == 39 and best == 39
    assert len(chosen) == n - 1
    assert is_connected(n, chosen)
    # every edge Kruskal skipped really would have closed a cycle
    for what, w, u, v, _, k in trace:
        if what == 'skip':
            # at that moment the first k chosen edges already joined u and v,
            # so taking this one really would have closed a cycle
            assert same_component(chosen[:k], u, v), (u, v)
    # the forest keeps one edge fewer per extra component
    assert len(f_edges) == n - 2
    for pts, want in cases:
        assert min_cost_connect_points(pts) == want
    print('all assertions passed')


def same_component(tree_edges, u, v):
    """True when u and v are already joined by these edges."""
    uf = UnionFind(len(NAMES))
    for w, a, b in tree_edges:
        uf.union(a, b)
    return uf.find(u) == uf.find(v)


if __name__ == '__main__':
    main()
