"""
Day 13 - Topological Sort
=========================

Run me directly:

    python topological_sort.py

A topological order of a directed graph is an ordering of its vertices such that
every edge u -> v puts u somewhere before v.  It exists if and only if the graph
is a DAG - a directed graph with no cycle - so "sort it" and "prove there is no
cycle" are the same computation.

Two standard ways to get one:

* Kahn's algorithm (BFS flavour) - repeatedly take a vertex with in-degree 0.
* DFS post-order - a vertex is emitted only after all of its successors are,
  so reversing the finish order gives a topological order.

Both are O(V + E).
"""

from collections import defaultdict, deque
import heapq


# --------------------------------------------------------------------------
# graph helpers
# --------------------------------------------------------------------------
def build(n, edges):
    """Adjacency list + in-degree array for `n` vertices labelled 0..n-1.

    `edges` is a list of (u, v) meaning "u must come before v".
    """
    adj = defaultdict(list)
    indeg = [0] * n
    for u, v in edges:
        adj[u].append(v)
        indeg[v] += 1
    return adj, indeg


# --------------------------------------------------------------------------
# 1. Kahn's algorithm
# --------------------------------------------------------------------------
def kahn(n, edges):
    """Topological order via repeated removal of in-degree-0 vertices.

    Returns the order, or None if the graph has a cycle.
    """
    adj, indeg = build(n, edges)
    queue = deque(v for v in range(n) if indeg[v] == 0)
    order = []

    while queue:
        u = queue.popleft()
        order.append(u)
        for v in adj[u]:
            indeg[v] -= 1          # u is placed, so v has one fewer blocker
            if indeg[v] == 0:      # v is now free to go
                queue.append(v)

    # every vertex inside a cycle keeps in-degree >= 1 forever
    return order if len(order) == n else None


def kahn_levels(n, edges):
    """Same algorithm, but grouped into rounds.

    Everything inside one round is mutually independent, so it is exactly the
    set of tasks you could run in parallel - the shape a build system or a
    multi-agent scheduler cares about.
    """
    adj, indeg = build(n, edges)
    frontier = [v for v in range(n) if indeg[v] == 0]
    levels, seen = [], 0

    while frontier:
        levels.append(sorted(frontier))
        seen += len(frontier)
        nxt = []
        for u in frontier:
            for v in adj[u]:
                indeg[v] -= 1
                if indeg[v] == 0:
                    nxt.append(v)
        frontier = nxt

    return levels if seen == n else None


def kahn_smallest(n, edges):
    """The lexicographically smallest topological order.

    Only change: a min-heap instead of a queue, so among the currently free
    vertices we always take the smallest.  (Day 05's heap, earning its keep.)
    """
    adj, indeg = build(n, edges)
    heap = [v for v in range(n) if indeg[v] == 0]
    heapq.heapify(heap)
    order = []

    while heap:
        u = heapq.heappop(heap)
        order.append(u)
        for v in adj[u]:
            indeg[v] -= 1
            if indeg[v] == 0:
                heapq.heappush(heap, v)

    return order if len(order) == n else None


# --------------------------------------------------------------------------
# 2. DFS post-order
# --------------------------------------------------------------------------
WHITE, GREY, BLACK = 0, 1, 2       # unseen / on the current stack / finished


def dfs_topo(n, edges):
    """Topological order via DFS finish times.  Returns None on a cycle.

    The three colours are what makes cycle detection work: meeting a GREY
    vertex means we walked back onto our own path, i.e. a back edge.
    """
    adj, _ = build(n, edges)
    color = [WHITE] * n
    finished = []

    def visit(u):
        color[u] = GREY
        for v in adj[u]:
            if color[v] == GREY:       # back edge -> cycle
                return False
            if color[v] == WHITE and not visit(v):
                return False
        color[u] = BLACK
        finished.append(u)             # all successors are already out
        return True

    for s in range(n):
        if color[s] == WHITE and not visit(s):
            return None

    return finished[::-1]              # reverse finish order


def dfs_topo_iterative(n, edges):
    """The same thing without recursion - Python's stack is only ~1000 deep."""
    adj, _ = build(n, edges)
    color = [WHITE] * n
    finished = []

    for s in range(n):
        if color[s] != WHITE:
            continue
        stack = [(s, iter(adj[s]))]
        color[s] = GREY
        while stack:
            u, it = stack[-1]
            for v in it:
                if color[v] == GREY:
                    return None
                if color[v] == WHITE:
                    color[v] = GREY
                    stack.append((v, iter(adj[v])))
                    break
            else:                      # no break: u has no unexplored edge left
                color[u] = BLACK
                finished.append(u)
                stack.pop()

    return finished[::-1]


def find_cycle(n, edges):
    """Return one cycle as a list of vertices, or None if the graph is a DAG.

    'It has a cycle' is rarely a useful error message; 'A -> B -> C -> A' is.
    """
    adj, _ = build(n, edges)
    color = [WHITE] * n
    parent = [-1] * n
    cycle = []

    def visit(u):
        color[u] = GREY
        for v in adj[u]:
            if color[v] == GREY:
                x = u
                cycle.append(v)
                while x != v:
                    cycle.append(x)
                    x = parent[x]
                cycle.append(v)
                cycle.reverse()
                return False
            if color[v] == WHITE:
                parent[v] = u
                if not visit(v):
                    return False
        color[u] = BLACK
        return True

    for s in range(n):
        if color[s] == WHITE and not visit(s):
            return cycle
    return None


# --------------------------------------------------------------------------
# 3. Validation
# --------------------------------------------------------------------------
def is_topological(order, n, edges):
    """Check an order directly against the definition."""
    if order is None or sorted(order) != list(range(n)):
        return False
    pos = {v: i for i, v in enumerate(order)}
    return all(pos[u] < pos[v] for u, v in edges)


# --------------------------------------------------------------------------
# 4. LeetCode 207 / 210 - course schedule
# --------------------------------------------------------------------------
def can_finish(num_courses, prerequisites):
    """LC 207.  prerequisites[i] = [a, b] means: take b before a."""
    edges = [(b, a) for a, b in prerequisites]
    return kahn(num_courses, edges) is not None


def find_order(num_courses, prerequisites):
    """LC 210.  Return any valid order, or [] if impossible."""
    edges = [(b, a) for a, b in prerequisites]
    return kahn(num_courses, edges) or []


# --------------------------------------------------------------------------
# demo
# --------------------------------------------------------------------------
COURSE_NAMES = {
    0: 'intro programming', 1: 'discrete math', 2: 'data structures',
    3: 'algorithms', 4: 'linear algebra', 5: 'databases',
    6: 'machine learning', 7: 'capstone',
}
N = 8
EDGES = [(0, 2), (0, 3), (1, 3), (1, 4), (2, 5), (3, 5),
         (3, 6), (4, 6), (5, 7), (6, 7)]


def section(title):
    print('\n' + '=' * 66)
    print(title)
    print('=' * 66)


def main():
    section('the DAG')
    adj, indeg = build(N, EDGES)
    for u in range(N):
        print('  %d %-20s in-degree %d   ->  %s'
              % (u, COURSE_NAMES[u], indeg[u], sorted(adj[u]) or '-'))

    section("Kahn's algorithm")
    order = kahn(N, EDGES)
    print('  order      ', order)
    print('  as courses ', ' -> '.join(COURSE_NAMES[v] for v in order))
    print('  valid?     ', is_topological(order, N, EDGES))

    section('Kahn by rounds - what could run in parallel')
    for i, level in enumerate(kahn_levels(N, EDGES)):
        print('  round %d: %s' % (i, [COURSE_NAMES[v] for v in level]))
    print('  4 rounds, so 8 courses need at least 4 semesters')

    section('smallest order (heap instead of queue)')
    print('  ', kahn_smallest(N, EDGES))

    section('DFS post-order')
    rec, it = dfs_topo(N, EDGES), dfs_topo_iterative(N, EDGES)
    print('  recursive  ', rec)
    print('  iterative  ', it)
    print('  same?      ', rec == it)
    print('  valid?     ', is_topological(rec, N, EDGES))
    print('  note: a different but equally correct order than Kahn gave')

    section('a cycle breaks everything')
    bad = EDGES + [(7, 0)]           # capstone becomes a prerequisite of intro
    print('  added edge 7 -> 0 (capstone before intro programming)')
    print('  kahn      ', kahn(N, bad))
    print('  dfs       ', dfs_topo(N, bad))
    print('  the cycle ', ' -> '.join(str(v) for v in find_cycle(N, bad)))

    section('LeetCode 207 / 210 - course schedule')
    print('  canFinish(2, [[1,0]])        ->', can_finish(2, [[1, 0]]))
    print('  canFinish(2, [[1,0],[0,1]])  ->', can_finish(2, [[1, 0], [0, 1]]))
    print('  findOrder(4, [[1,0],[2,0],[3,1],[3,2]]) ->',
          find_order(4, [[1, 0], [2, 0], [3, 1], [3, 2]]))
    print('  findOrder(2, [[1,0],[0,1]])  ->', find_order(2, [[1, 0], [0, 1]]))

    section('tests')
    assert is_topological(kahn(N, EDGES), N, EDGES)
    assert is_topological(dfs_topo(N, EDGES), N, EDGES)
    assert is_topological(dfs_topo_iterative(N, EDGES), N, EDGES)
    assert kahn_smallest(N, EDGES) == sorted(range(N)) or \
        is_topological(kahn_smallest(N, EDGES), N, EDGES)
    assert kahn(N, EDGES + [(7, 0)]) is None
    assert dfs_topo(N, EDGES + [(7, 0)]) is None
    assert dfs_topo_iterative(N, EDGES + [(7, 0)]) is None
    assert find_cycle(N, EDGES) is None
    assert find_cycle(3, [(0, 1), (1, 2), (2, 0)])[0] == \
        find_cycle(3, [(0, 1), (1, 2), (2, 0)])[-1]
    assert kahn(3, []) == [0, 1, 2]                     # no edges at all
    assert kahn(1, []) == [0]
    assert can_finish(2, [[1, 0]]) and not can_finish(2, [[1, 0], [0, 1]])
    assert find_order(2, [[1, 0], [0, 1]]) == []

    # a random DAG: shuffle labels, keep edges pointing forward
    import random
    random.seed(13)
    for _ in range(200):
        n = random.randint(1, 12)
        perm = list(range(n))
        random.shuffle(perm)
        e = [(perm[i], perm[j])
             for i in range(n) for j in range(i + 1, n) if random.random() < .3]
        assert is_topological(kahn(n, e), n, e)
        assert is_topological(dfs_topo(n, e), n, e)
        assert is_topological(dfs_topo_iterative(n, e), n, e)
        assert kahn_levels(n, e) is not None
    print('  all assertions passed (including 200 random DAGs)')


if __name__ == '__main__':
    main()
