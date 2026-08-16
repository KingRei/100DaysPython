"""
100 Days of Python - Day 12: Graph Traversal (BFS & DFS)

Run me:   python graph_traversal.py

Prints a full walkthrough: the adjacency list, BFS layers and shortest paths, DFS
(recursive and iterative), connected components, and LeetCode 200 - Number of Islands.
"""

from collections import defaultdict, deque


# ---------------------------------------------------------------------------
# the graph
# ---------------------------------------------------------------------------
class Graph:
    """Undirected graph stored as an adjacency list."""

    def __init__(self):
        self.adj = defaultdict(list)

    def add_edge(self, u, v):
        self.adj[u].append(v)
        self.adj[v].append(u)      # drop this line for a directed graph

    def neighbors(self, u):
        return self.adj[u]

    def vertices(self):
        return sorted(self.adj)

    def __repr__(self):
        return '\n'.join(f'  {v}: {sorted(self.adj[v])}' for v in self.vertices())


# ---------------------------------------------------------------------------
# breadth-first search
# ---------------------------------------------------------------------------
def bfs(graph, start):
    """Return (visit order, distance from start, parent pointers)."""
    visited = {start}
    dist = {start: 0}
    parent = {start: None}
    order = []
    queue = deque([start])

    while queue:
        u = queue.popleft()
        order.append(u)
        for v in sorted(graph.neighbors(u)):
            if v not in visited:
                visited.add(v)              # mark on push, not on pop
                dist[v] = dist[u] + 1
                parent[v] = u
                queue.append(v)
    return order, dist, parent


def shortest_path(graph, start, goal):
    """Fewest-edges path on an unweighted graph, or None if unreachable."""
    _, _, parent = bfs(graph, start)
    if goal not in parent:
        return None
    path, node = [], goal
    while node is not None:
        path.append(node)
        node = parent[node]
    return path[::-1]


# ---------------------------------------------------------------------------
# depth-first search
# ---------------------------------------------------------------------------
def dfs_recursive(graph, start, visited=None, order=None):
    if visited is None:
        visited, order = set(), []
    visited.add(start)
    order.append(start)
    for v in sorted(graph.neighbors(start)):
        if v not in visited:
            dfs_recursive(graph, v, visited, order)
    return order


def dfs_iterative(graph, start):
    """Same traversal without recursion - safer on deep graphs."""
    visited, order = set(), []
    stack = [start]
    while stack:
        u = stack.pop()
        if u in visited:
            continue
        visited.add(u)
        order.append(u)
        # reversed so the smallest neighbour comes off the stack first
        for v in sorted(graph.neighbors(u), reverse=True):
            if v not in visited:
                stack.append(v)
    return order


def connected_components(graph):
    seen, comps = set(), []
    for v in graph.vertices():
        if v not in seen:
            comp = dfs_recursive(graph, v)
            seen |= set(comp)
            comps.append(comp)
    return comps


# ---------------------------------------------------------------------------
# LeetCode 200 - Number of Islands
# ---------------------------------------------------------------------------
def num_islands_bfs(grid):
    if not grid or not grid[0]:
        return 0
    rows, cols = len(grid), len(grid[0])
    grid = [list(row) for row in grid]      # work on a mutable copy
    count = 0

    for r in range(rows):
        for c in range(cols):
            if grid[r][c] != '1':
                continue
            count += 1                      # found a fresh island
            queue = deque([(r, c)])
            grid[r][c] = '0'                # sink it: the grid is our visited set
            while queue:
                i, j = queue.popleft()
                for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    ni, nj = i + di, j + dj
                    if 0 <= ni < rows and 0 <= nj < cols and grid[ni][nj] == '1':
                        grid[ni][nj] = '0'
                        queue.append((ni, nj))
    return count


def num_islands_dfs(grid):
    """Recursive flood fill - watch the recursion depth on big grids."""
    if not grid or not grid[0]:
        return 0
    rows, cols = len(grid), len(grid[0])
    grid = [list(row) for row in grid]

    def sink(i, j):
        if not (0 <= i < rows and 0 <= j < cols) or grid[i][j] != '1':
            return
        grid[i][j] = '0'
        sink(i + 1, j); sink(i - 1, j); sink(i, j + 1); sink(i, j - 1)

    count = 0
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == '1':
                count += 1
                sink(r, c)
    return count


# ---------------------------------------------------------------------------
# demo
# ---------------------------------------------------------------------------
def section(title):
    print('\n' + title)
    print('-' * len(title))


def main():
    g = Graph()
    for u, v in [(1, 2), (1, 3), (2, 4), (2, 5), (3, 5), (3, 6),
                 (4, 7), (5, 7), (6, 8), (7, 8)]:
        g.add_edge(u, v)

    section('the graph (adjacency list)')
    print(g)

    section('BFS from vertex 1')
    order, dist, _ = bfs(g, 1)
    print('visit order :', order)
    layers = defaultdict(list)
    for v, d in dist.items():
        layers[d].append(v)
    for d in sorted(layers):
        print(f'  layer {d} (distance {d}): {sorted(layers[d])}')

    section('shortest paths - unweighted, so BFS is enough')
    for goal in (5, 7, 8):
        path = shortest_path(g, 1, goal)
        print(f'  1 -> {goal}: {path}  ({len(path) - 1} edges)')

    section('DFS from vertex 1')
    print('recursive   :', dfs_recursive(g, 1))
    print('iterative   :', dfs_iterative(g, 1))

    section('connected components of a disconnected graph')
    h = Graph()
    for u, v in [(1, 2), (2, 3), (10, 11), (20, 21), (21, 22)]:
        h.add_edge(u, v)
    for i, comp in enumerate(connected_components(h), 1):
        print(f'  component {i}: {comp}')

    section('LeetCode 200 - Number of Islands')
    tests = [
        (['11110', '11010', '11000', '00000'], 1),
        (['11000', '11000', '00100', '00011'], 3),
        ([], 0),
        (['000'], 0),
    ]
    for grid, expected in tests:
        got_bfs, got_dfs = num_islands_bfs(grid), num_islands_dfs(grid)
        assert got_bfs == got_dfs == expected, (grid, got_bfs, got_dfs, expected)
        for row in grid:
            print('   ', row.replace('1', '#').replace('0', '.'))
        print(f'    -> {got_bfs} island(s)   ok\n')

    print('all assertions passed')


if __name__ == '__main__':
    main()
