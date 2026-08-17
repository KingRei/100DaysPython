# Graph Traversal - BFS and DFS

More details in:
https://medium.com/100-days-of-python/day-12-data-structure-graph-traversal-bfs-and-dfs-96580ae83b0e

Interactive walkthrough: [demo.html](demo.html) - a single self-contained page
(no build step, works offline) that animates BFS ring by ring, shortest-path
reconstruction, DFS and LeetCode 200 step by step.

![img](https://github.com/KingRei/100DaysPython/blob/master/day%2012%20-%20graph%20traversal/imgs/day12_1.png?raw=true)

A graph you cannot walk through is just a pile of edges. Graph traversal is the process
of visiting every vertex reachable from a starting point, exactly once, in some
well-defined order. There are two classic orders, and they differ only in the container
used to hold the vertices that have been discovered but not yet expanded.

**Breadth-first search (BFS)** uses a queue. It expands the start vertex, then all of
its neighbours, then all of *their* unseen neighbours, sweeping outwards in rings. Since
vertices leave the queue in non-decreasing order of distance, BFS gives the shortest
path in edges on an unweighted graph for free.

**Depth-first search (DFS)** uses a stack - usually the call stack, via recursion. It
follows one branch as far as it goes, then backtracks to the most recent vertex with an
unexplored neighbour. The edges DFS walks form a *DFS tree*, and classifying the
remaining edges against that tree is the basis of cycle detection, bridges, articulation
points and strongly connected components.

Both traversals visit every vertex once and inspect every edge once from each endpoint,
so both run in O(V + E) time.

![img](https://github.com/KingRei/100DaysPython/blob/master/day%2012%20-%20graph%20traversal/imgs/day12_2.png?raw=true)

## Implementation notes

- **Mark a vertex visited when you push it, not when you pop it.** Marking on pop lets a
  vertex with two parents enter the queue twice.
- Keeping a `parent` dictionary during BFS costs nothing and lets you reconstruct the
  actual shortest path, not just its length.
- A graph need not be connected. A full traversal loops over every vertex and starts a
  new search from any vertex not yet seen - that loop is exactly a connected-components
  count.
- Prefer the iterative DFS when the graph may be deep; CPython's default recursion limit
  is 1000 frames.

## BFS or DFS?

| | BFS | DFS |
|---|---|---|
| container | queue | stack / recursion |
| finds | shortest path on an unweighted graph | some path, not necessarily shortest |
| memory | O(width of the graph) | O(depth of the graph) |
| natural for | distances, levels, "closest" | cycles, components, ordering, backtracking |

## Complexity

| Operation | Time | Space |
|---|---|---|
| BFS / DFS over the whole graph | O(V + E) | O(V) |
| Shortest path on an unweighted graph | O(V + E) | O(V) |
| Connected components | O(V + E) | O(V) |
| LeetCode 200 on an m x n grid | O(m * n) | O(min(m, n)) BFS, O(m * n) DFS |

## LeetCode 200 - Number of Islands

![img](https://github.com/KingRei/100DaysPython/blob/master/day%2012%20-%20graph%20traversal/imgs/day12_3.png?raw=true)

Given an `m x n` binary grid where `'1'` is land and `'0'` is water, count the islands.
Every land cell is a vertex and orthogonally adjacent land cells share an edge, so
counting islands is counting connected components. Scan the grid; each time an unvisited
`'1'` appears, increment the counter and flood fill the island away. Sinking the land as
you go means the grid doubles as the visited set.

## Run it

```sh
python graph_traversal.py     # standalone walkthrough, no Jupyter required
```

Or open `demo.html` in a browser for the animated version (中文 / English toggle).

## References

- [Wiki - Breadth-first search](https://en.wikipedia.org/wiki/Breadth-first_search)
- [Wiki - Depth-first search](https://en.wikipedia.org/wiki/Depth-first_search)
- [LeetCode 200 - Number of Islands](https://leetcode.com/problems/number-of-islands/)
- [YouTube - BFS and DFS](https://www.youtube.com/watch?v=zaBhtODEL0w)
