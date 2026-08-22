# Bellman-Ford, SPFA and Floyd-Warshall - shortest paths without the non-negative assumption

More details in:
https://medium.com/100-days-of-python

Interactive walkthrough: [demo.html](demo.html) - a single self-contained page
(no build step, works offline, 中文 / English toggle) that animates Bellman-Ford round by
round, contrasts it with SPFA's work list, detects a negative cycle with one extra round,
runs Floyd-Warshall stage by stage (including what happens when the loops are nested in the
wrong order), and solves LeetCode 787 both correctly and incorrectly.

![img](https://github.com/KingRei/100DaysPython/blob/master/day%2017%20-%20bellman%20ford%20and%20floyd%20warshall/imgs/day17_1.png?raw=true)

Yesterday's `heap_greedy` skeleton settles one vertex per iteration and never revisits it.
That shortcut is only sound when no edge can shorten a path after the fact - in other words,
when every weight is non-negative. Exchange rates, energies, profit and loss, and any edge
that represents a reward rather than a cost all break the assumption, and Dijkstra does not
complain: it simply returns a wrong number.

Bellman-Ford gives up the shortcut. It settles nothing, and instead relaxes **every** edge,
`V-1` times over. The single line of relaxation, `if dist[u] + w < dist[v]`, is the same line
Dijkstra uses; the difference is that Bellman-Ford refuses to commit to an order, so an
answer written in round 1 can still be improved in round 2.

## Bellman-Ford

```python
def bellman_ford(n, edges, src):
    dist, parent = [INF] * n, [-1] * n
    dist[src] = 0
    for _ in range(n - 1):
        changed = False
        for u, v, w in edges:
            if dist[u] + w < dist[v]:
                dist[v] = dist[u] + w
                parent[v] = u
                changed = True
        if not changed:                 # converged early
            break
    for u, v, w in edges:               # one extra round
        if dist[u] + w < dist[v]:
            return dist, parent, True   # negative cycle
    return dist, parent, False
```

After round `r`, every shortest path that uses at most `r` edges is correct. Without a
negative cycle a shortest path never repeats a vertex, so it uses at most `V-1` edges, and
`V-1` rounds always suffice. The loop count is not an estimate - it is a proof.

## Negative cycles

![img](https://github.com/KingRei/100DaysPython/blob/master/day%2017%20-%20bellman%20ford%20and%20floyd%20warshall/imgs/day17_2.png?raw=true)

Change one weight in the example graph and `B -> E -> C -> B` sums to `-5`. Going round the
loop again is always cheaper, so no shortest path exists; the answer is minus infinity. The
detector is one extra round: `V-1` rounds already cover every path that repeats no vertex, so
anything that still improves must be revisiting one. Walking back `V` steps along `parent` is
guaranteed to land inside the cycle, which is how the cycle itself is recovered.

This "failure" is frequently the goal. Take `-log` of exchange rates as edge weights and a
negative cycle is an arbitrage loop.

## SPFA

Most edges cannot relax: only an edge whose tail just improved has any chance. SPFA keeps a
queue of vertices whose distance actually changed and scans only those - Bellman-Ford with a
work list. It reaches the same answer on the example graph in 7 queue pops instead of 40 edge
scans, and detects negative cycles for free (a vertex cannot be enqueued `V` times otherwise).
The worst case is still `O(VE)`, so it is a constant-factor win, not a better algorithm.

## Floyd-Warshall

![img](https://github.com/KingRei/100DaysPython/blob/master/day%2017%20-%20bellman%20ford%20and%20floyd%20warshall/imgs/day17_3.png?raw=true)

For all pairs at once, the DP is over *which vertices may appear in the middle of a path*.
After stage `k`, `dist[i][j]` is the cheapest `i -> j` path whose intermediate vertices all
come from `{0..k}`. Adding vertex `k` offers exactly two options - route through it or not:

```python
for k in range(n):          # k MUST be outermost: it is the DP stage
    for i in range(n):
        for j in range(n):
            if dist[i][k] + dist[k][j] < dist[i][j]:
                dist[i][j] = dist[i][k] + dist[k][j]
```

Write the loops as `i, j, k` instead and the program still runs, still fills the table, and
silently leaves some entries too large, because `dist[i][k]` gets read before stage `k`
finished computing it. On the example graph three of the twenty-five pairs come out wrong.

## LeetCode 787 - Cheapest Flights Within K Stops

![img](https://github.com/KingRei/100DaysPython/blob/master/day%2017%20-%20bellman%20ford%20and%20floyd%20warshall/imgs/day17_4.png?raw=true)

"At most `k` stops" is "at most `k+1` edges", and `k+1` rounds of Bellman-Ford is exactly the
set of paths using at most `k+1` edges. The constraint that looks like an extra complication
*is* the loop counter.

```python
def find_cheapest_price(n, flights, src, dst, k):
    dist = [INF] * n
    dist[src] = 0
    for _ in range(k + 1):
        prev = dist[:]                  # relax from LAST round's values only
        for u, v, w in flights:
            if prev[u] + w < dist[v]:
                dist[v] = prev[u] + w
    return dist[dst] if dist[dst] < INF else -1
```

Drop the `prev = dist[:]` snapshot and a single round can chain `0 -> 1 -> 2 -> 3`, quietly
granting more stops than the problem allows: 400 instead of 700. Nothing crashes, nothing
slows down, and the wrong answer is the smaller-looking one.

## Complexity

| Algorithm | Time | Space | Handles |
|---|---|---|---|
| Dijkstra (day 16) | O(E log V) | O(V) | non-negative weights only |
| Bellman-Ford | O(VE) | O(V) | negative weights, detects negative cycles |
| SPFA | O(VE) worst case, much less in practice | O(V) | same as Bellman-Ford |
| Floyd-Warshall | O(V^3) | O(V^2) | all pairs, negative weights |
| LC 787 | O(kE) | O(V) | k+1 rounds of Bellman-Ford |

## Run it

```bash
python bellman_ford_floyd.py
```

Prints the graph, Bellman-Ford round by round, the SPFA comparison, the negative cycle it
recovers, the full Floyd-Warshall table, the three pairs the wrong loop order gets wrong, and
both LeetCode 787 versions - ending in asserts.

## References

- [Bellman-Ford algorithm](https://en.wikipedia.org/wiki/Bellman%E2%80%93Ford_algorithm)
- [Shortest Path Faster Algorithm](https://en.wikipedia.org/wiki/Shortest_path_faster_algorithm)
- [Floyd-Warshall algorithm](https://en.wikipedia.org/wiki/Floyd%E2%80%93Warshall_algorithm)
- [LeetCode 787 - Cheapest Flights Within K Stops](https://leetcode.com/problems/cheapest-flights-within-k-stops/)
