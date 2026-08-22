# Prim and Dijkstra - one heap-greedy skeleton

More details in:
https://medium.com/100-days-of-python

Interactive walkthrough: [demo.html](demo.html) - a single self-contained page
(no build step, works offline, 中文 / English toggle) that animates the shared loop with
both keys side by side, shows why the minimum spanning tree and the shortest-path tree are
genuinely different trees, breaks Dijkstra with a single negative edge, and walks through
LeetCode 743.

![img](https://github.com/KingRei/100DaysPython/blob/master/day%2016%20-%20prim%20and%20dijkstra/imgs/day16_1.png?raw=true)

Yesterday's Kruskal sorted every edge once and filtered the list with Union-Find. Prim
attacks the same problem from the other end: instead of collecting edges from all over the
graph, it grows **one** tree outwards from a starting vertex, and at every step it buys the
cheapest edge that leaves the tree. Dijkstra's shortest-path algorithm does something that
sounds unrelated - it computes the distance from a source to every other vertex - yet the
loop is the same loop.

Both algorithms repeat exactly two actions: pop the cheapest unsettled vertex from a heap,
and then relax its neighbours. The only difference is what the heap is ordered by. Prim
stores the weight of the single edge that would pull a vertex in; Dijkstra stores the total
length of the path from the source. One `key` function apart.

![img](https://github.com/KingRei/100DaysPython/blob/master/day%2016%20-%20prim%20and%20dijkstra/imgs/day16_2.png?raw=true)

## The shared skeleton

```python
def heap_greedy(n, adj, src, key):
    best, parent, done = [INF] * n, [-1] * n, [False] * n
    best[src] = 0
    pq = [(0, src)]
    while pq:
        d, u = heappop(pq)
        if done[u]:                       # a stale copy: lazy deletion
            continue
        done[u] = True
        for v, w in adj[u]:
            nd = key(d, w)                # Prim: w      Dijkstra: d + w
            if not done[v] and nd < best[v]:
                best[v], parent[v] = nd, u
                heappush(pq, (nd, v))
    return best, parent

prim     = lambda n, adj, s: heap_greedy(n, adj, s, lambda d, w: w)
dijkstra = lambda n, adj, s: heap_greedy(n, adj, s, lambda d, w: d + w)
```

`heapq` has no decrease-key, so instead of editing an entry in place (which would mean
scanning the heap, O(n)) we push a second, better copy and let the old one rot. The
`if done[u]: continue` line throws the stale copies away when they surface. The heap can
therefore hold up to *E* entries, which is what makes the bound `O(E log V)` rather than
`O(E log E)` - the same thing, since `log E < 2 log V`.

## Why settling a vertex is safe

Prim inherits Kruskal's cut property: at every step the frontier splits the graph into
"the tree" and "everything else", and the cheapest edge crossing that cut belongs to some
minimum spanning tree. Every edge Prim buys is therefore final - it is already part of
the minimum spanning tree, not a provisional choice. Dijkstra needs a different argument: any competing route to `u` must
leave through a vertex still sitting in the heap, whose tentative distance is already at
least `d`, and non-negative edges can only make it longer. That argument is the entire
reason **Dijkstra requires non-negative weights** - and Prim, which never accumulates, does
not care.

![img](https://github.com/KingRei/100DaysPython/blob/master/day%2016%20-%20prim%20and%20dijkstra/imgs/day16_4.png?raw=true)

## Minimum spanning tree is not shortest-path tree

On the example graph Prim returns a tree of weight 39 - the same 39 Kruskal found - while
Dijkstra's `parent` array forms a spanning tree of weight 44. Walking from A to G inside
the MST costs 23; the true shortest path costs 22 and uses an edge the MST does not
contain. Minimising the sum of the edges and minimising every individual path are
different objectives, and they conflict.

![img](https://github.com/KingRei/100DaysPython/blob/master/day%2016%20-%20prim%20and%20dijkstra/imgs/day16_3.png?raw=true)

## LeetCode 743 - Network Delay Time

A signal starts at node `k` and travels along directed edges; the answer is when the last
node hears it, i.e. `max(dist)`, or `-1` if some node never does. Three traps: the edges
are directed (do not add both directions out of habit), the answer is a maximum over
shortest paths rather than a single shortest path, and the unreachable case must be checked
explicitly.

## Complexity

| Algorithm | Time | Space | Notes |
|---|---|---|---|
| Prim (binary heap) | O(E log V) | O(V + E) | dense graphs favour the O(V²) array version |
| Dijkstra (binary heap) | O(E log V) | O(V + E) | non-negative weights only |
| Dijkstra (Fibonacci heap) | O(E + V log V) | O(V + E) | better in theory, rarely in practice |

## Run it

```sh
python prim_dijkstra.py     # standalone walkthrough, no Jupyter required
```

The script brute-forces every spanning tree of the example graph to confirm 39, checks
Dijkstra against Bellman-Ford, and demonstrates the negative-edge failure with an assert,
so a regression cannot pass quietly.

Or open `demo.html` in a browser for the animated version (中文 / English toggle).

## References
- [Wikipedia - Prim's algorithm](https://en.wikipedia.org/wiki/Prim%27s_algorithm)
- [Wikipedia - Dijkstra's algorithm](https://en.wikipedia.org/wiki/Dijkstra%27s_algorithm)
- [LeetCode 743 - Network Delay Time](https://leetcode.com/problems/network-delay-time/)
