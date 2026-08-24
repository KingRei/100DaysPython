"""Day 18 - one DFS with timestamps: SCCs, bridges, articulation points.

Run me:  python tarjan.py

Everything here is the same depth-first search, instrumented with two numbers per vertex:

    disc[u]  when the DFS first reached u   (a clock that only ever increases)
    low[u]   the smallest disc reachable from u's subtree using tree edges
             plus at most one edge that goes "backwards"

`low` answers a single question - "can this subtree get above me without going through
me?" - and the three classic algorithms are three readings of that answer:

    low[v] >  disc[u]   the tree edge u-v is a bridge            (undirected)
    low[v] >= disc[u]   u is an articulation point               (undirected)
    low[u] == disc[u]   u is the root of a strongly connected     (directed)
                        component, sitting on top of it in a stack
"""

import sys
from collections import defaultdict

# --------------------------------------------------------------------- example graphs

# Undirected.  Two triangles glued at vertex 2, then a single edge across to a third
# triangle.  Articulation points: 2, 4, 5.  Bridge: 4-5 only.
# Vertex 2 is the interesting one - it cuts the graph but lies on no bridge at all.
UN_N = 8
UN_EDGES = [(0, 1), (1, 2), (2, 0),
            (2, 3), (3, 4), (4, 2),
            (4, 5),
            (5, 6), (6, 7), (7, 5)]

# Directed.  SCCs are {1,2,3}, {4,5} and {0} on its own.
# The edge 5->2 is the trap: it points into an SCC that has already been finished.
DI_N = 6
DI_EDGES = [(0, 1), (0, 4),
            (1, 2), (2, 3), (3, 1),
            (4, 5), (5, 4),
            (5, 2)]


def undirected_adj(n, edges):
    """Adjacency list of (neighbour, edge_id) - the edge id is what makes the
    parent test correct even when the same pair is connected twice."""
    adj = [[] for _ in range(n)]
    for i, (u, v) in enumerate(edges):
        adj[u].append((v, i))
        adj[v].append((u, i))
    return adj


def directed_adj(n, edges):
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
    return adj


# --------------------------------------------------------------------- 1. bridges
def find_bridges(n, edges):
    """Edges whose removal disconnects the graph.  Returns (bridges, disc, low)."""
    adj = undirected_adj(n, edges)
    disc = [-1] * n
    low = [0] * n
    clock = 0
    bridges = []

    def dfs(u, in_edge):
        nonlocal clock
        disc[u] = low[u] = clock
        clock += 1
        for v, eid in adj[u]:
            if eid == in_edge:          # skip the edge we arrived on, NOT the vertex
                continue
            if disc[v] == -1:           # tree edge
                dfs(v, eid)
                low[u] = min(low[u], low[v])
                if low[v] > disc[u]:    # v's subtree cannot reach u or above
                    bridges.append((u, v))
            else:                       # back edge: v is an ancestor still on the path
                low[u] = min(low[u], disc[v])

    for s in range(n):
        if disc[s] == -1:
            dfs(s, -1)
    return bridges, disc, low


def find_bridges_by_parent_vertex(n, edges):
    """The common buggy variant: skip the parent *vertex* instead of the parent *edge*.
    Identical on simple graphs, wrong the moment two vertices are joined twice."""
    adj = undirected_adj(n, edges)
    disc = [-1] * n
    low = [0] * n
    clock = 0
    bridges = []

    def dfs(u, parent):
        nonlocal clock
        disc[u] = low[u] = clock
        clock += 1
        for v, _eid in adj[u]:
            if v == parent:             # <-- the bug
                continue
            if disc[v] == -1:
                dfs(v, u)
                low[u] = min(low[u], low[v])
                if low[v] > disc[u]:
                    bridges.append((u, v))
            else:
                low[u] = min(low[u], disc[v])

    for s in range(n):
        if disc[s] == -1:
            dfs(s, -1)
    return bridges


# --------------------------------------------------------- 2. articulation points
def find_articulation_points(n, edges):
    """Vertices whose removal disconnects the graph.  Same DFS, two changes:
    `>=` instead of `>`, and the root needs its own rule."""
    adj = undirected_adj(n, edges)
    disc = [-1] * n
    low = [0] * n
    clock = 0
    cut = set()

    def dfs(u, in_edge, root):
        nonlocal clock
        disc[u] = low[u] = clock
        clock += 1
        children = 0
        for v, eid in adj[u]:
            if eid == in_edge:
                continue
            if disc[v] == -1:
                children += 1
                dfs(v, eid, root)
                low[u] = min(low[u], low[v])
                if u != root and low[v] >= disc[u]:
                    cut.add(u)          # v's subtree can reach u, but no higher
            else:
                low[u] = min(low[u], disc[v])
        if u == root and children > 1:  # the root cuts iff it starts two subtrees
            cut.add(u)

    for s in range(n):
        if disc[s] == -1:
            dfs(s, -1, s)
    return sorted(cut)


# ------------------------------------------------------------------------- 3. SCC
def tarjan_scc(n, edges):
    """Strongly connected components of a directed graph, one DFS, O(V+E).
    Returns components in reverse topological order of the condensation."""
    adj = directed_adj(n, edges)
    disc = [-1] * n
    low = [0] * n
    on_stack = [False] * n
    stack = []
    comps = []
    clock = 0

    def dfs(u):
        nonlocal clock
        disc[u] = low[u] = clock
        clock += 1
        stack.append(u)
        on_stack[u] = True
        for v in adj[u]:
            if disc[v] == -1:                   # tree edge
                dfs(v)
                low[u] = min(low[u], low[v])
            elif on_stack[v]:                   # back / forward edge inside this SCC
                low[u] = min(low[u], disc[v])
            # else: v belongs to a component that is already finished - ignore it
        if low[u] == disc[u]:                   # u is the root of an SCC
            comp = []
            while True:
                w = stack.pop()
                on_stack[w] = False
                comp.append(w)
                if w == u:
                    break
            comps.append(sorted(comp))

    for s in range(n):
        if disc[s] == -1:
            dfs(s)
    return comps, disc, low


def tarjan_scc_without_onstack(n, edges):
    """The bug that merges components: relaxing `low` against every visited vertex
    instead of only the ones still on the stack."""
    adj = directed_adj(n, edges)
    disc = [-1] * n
    low = [0] * n
    stack = []
    comps = []
    clock = 0

    def dfs(u):
        nonlocal clock
        disc[u] = low[u] = clock
        clock += 1
        stack.append(u)
        for v in adj[u]:
            if disc[v] == -1:
                dfs(v)
                low[u] = min(low[u], low[v])
            else:
                low[u] = min(low[u], disc[v])    # <-- no on_stack test
        if low[u] == disc[u]:
            comp = []
            while True:
                w = stack.pop()
                comp.append(w)
                if w == u:
                    break
            comps.append(sorted(comp))

    for s in range(n):
        if disc[s] == -1:
            dfs(s)
    return comps


def condensation(n, edges, comps):
    """Collapse each SCC into one vertex.  The result is always a DAG - which is why
    'topological order' is a well-defined thing to ask for on any directed graph."""
    cid = [0] * n
    for i, comp in enumerate(comps):
        for u in comp:
            cid[u] = i
    dag = defaultdict(set)
    for u, v in edges:
        if cid[u] != cid[v]:
            dag[cid[u]].add(cid[v])
    return cid, {k: sorted(v) for k, v in dag.items()}


# ------------------------------------------------------- 4. LeetCode 1192 (bridges)
def critical_connections(n, connections):
    """LC 1192 - Critical Connections in a Network.  The answer is exactly the set of
    bridges.  Written iteratively because n goes up to 10^5 and CPython's default
    recursion limit is 1000: the recursive version above is correct but blows the stack.
    """
    adj = [[] for _ in range(n)]
    for i, (u, v) in enumerate(connections):
        adj[u].append((v, i))
        adj[v].append((u, i))

    disc = [-1] * n
    low = [0] * n
    clock = 0
    out = []

    for s in range(n):
        if disc[s] != -1:
            continue
        disc[s] = low[s] = clock
        clock += 1
        stack = [(s, -1, iter(adj[s]))]     # (vertex, edge we came in on, cursor)
        while stack:
            u, in_edge, it = stack[-1]
            descended = False
            for v, eid in it:               # the cursor remembers where we stopped
                if eid == in_edge:
                    continue
                if disc[v] == -1:
                    disc[v] = low[v] = clock
                    clock += 1
                    stack.append((v, eid, iter(adj[v])))
                    descended = True
                    break                   # go deeper, resume this loop later
                low[u] = min(low[u], disc[v])
            if not descended:               # u is finished - this is the "return"
                stack.pop()
                if stack:
                    p = stack[-1][0]
                    low[p] = min(low[p], low[u])
                    if low[u] > disc[p]:
                        out.append([p, u])
    return out


# ------------------------------------------------------------------------- walkthrough
def line(title):
    print('\n' + title)
    print('-' * len(title))


def show_times(disc, low, names=None):
    n = len(disc)
    names = names or [str(i) for i in range(n)]
    print('vertex ' + ' '.join('%4s' % names[i] for i in range(n)))
    print('disc   ' + ' '.join('%4d' % disc[i] for i in range(n)))
    print('low    ' + ' '.join('%4d' % low[i] for i in range(n)))


def main():
    line('the undirected example')
    print('edges:', UN_EDGES)
    print('two triangles glued at vertex 2, a single edge 4-5, a third triangle 5-6-7')

    line('1. bridges  (low[v] > disc[u])')
    bridges, disc, low = find_bridges(UN_N, UN_EDGES)
    show_times(disc, low)
    print('bridges:', bridges)
    print('only 4-5: every other edge sits on a cycle, so its subtree can climb past')
    print('the vertex above it without using the tree edge.')

    line('2. articulation points  (low[v] >= disc[u], plus a rule for the root)')
    cuts = find_articulation_points(UN_N, UN_EDGES)
    print('cut vertices:', cuts)
    print('4 and 5 are the endpoints of the bridge, but 2 is not on any bridge at all:')
    print('removing the edge 1-2 or 2-3 leaves the graph connected, removing the')
    print('vertex 2 does not.  That single character, > vs >=, is the whole difference.')

    line('3. two cables between the same pair of routers')
    doubled = UN_EDGES + [(4, 5)]           # a second, parallel 4-5 link
    ok, _, _ = find_bridges(UN_N, doubled)
    bad = find_bridges_by_parent_vertex(UN_N, doubled)
    print('edge-id version  :', ok, '  <- correct, the link is now redundant')
    print('parent-vertex ver:', bad, '  <- still calls 4-5 critical')
    print('skipping the parent vertex also skips the second cable, so the DFS never')
    print('sees the alternative route it just walked past.')

    line('4. strongly connected components  (low[u] == disc[u])')
    print('edges:', DI_EDGES)
    comps, ddisc, dlow = tarjan_scc(DI_N, DI_EDGES)
    show_times(ddisc, dlow)
    print('components (in reverse topological order):', comps)

    line('5. the same code without the on-stack test')
    print('buggy:', tarjan_scc_without_onstack(DI_N, DI_EDGES))
    print('the edge 5->2 points into {1,2,3}, which was finished long ago.  Its disc')
    print('is small, so relaxing low against it drags 4, 5 and even 0 into one blob.')
    print('"still on the stack" is what distinguishes an ancestor from a stranger.')

    line('6. condensation: every directed graph is a DAG of its SCCs')
    cid, dag = condensation(DI_N, DI_EDGES, comps)
    print('component id per vertex:', cid)
    print('edges between components:', dict(dag))
    print('collapsing each SCC leaves an acyclic graph, which is what makes')
    print('"topological order" meaningful for a graph that contains cycles.')

    line('7. LeetCode 1192 - critical connections')
    conns = [[0, 1], [1, 2], [2, 0], [1, 3]]
    print('n = 4, connections =', conns)
    print('answer:', critical_connections(4, conns))
    print('identical to find_bridges, but iterative: n <= 10^5 and CPython recurses')
    print('only 1000 deep, so a path graph would raise RecursionError.')

    line('tests')
    assert sorted(tuple(sorted(b)) for b in bridges) == [(4, 5)]
    assert cuts == [2, 4, 5]
    assert find_bridges(UN_N, doubled)[0] == []
    assert sorted(tuple(sorted(b)) for b in bad) == [(4, 5)]
    assert sorted(comps) == [[0], [1, 2, 3], [4, 5]]
    assert comps[0] == [1, 2, 3]                    # deepest component finishes first
    assert tarjan_scc_without_onstack(DI_N, DI_EDGES) == [[1, 2, 3], [0, 4, 5]]
    assert dag == {2: [0, 1], 1: [0]}
    assert critical_connections(4, conns) == [[1, 3]]
    assert sorted(tuple(sorted(e)) for e in
                  critical_connections(UN_N, [list(e) for e in UN_EDGES])) == [(4, 5)]
    path = [[i, i + 1] for i in range(3000)]        # would break the recursive version
    assert len(critical_connections(3001, path)) == 3000
    print('all assertions passed')


if __name__ == '__main__':
    sys.setrecursionlimit(10000)
    main()
