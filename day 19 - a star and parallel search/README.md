# A* - Dijkstra with a guess, and what happens when the search stops being sequential

More details in:
https://medium.com/100-days-of-python

Interactive walkthrough: [demo.html](demo.html) - a single self-contained page
(no build step, works offline, 中文 / English toggle) that runs the *same* implementation
with h = 0 and with a Manhattan heuristic side by side, shows an inflated heuristic walking
confidently into the expensive part of the map, animates delta-stepping at delta = 1, 4 and 99
so the phases-versus-work trade is visible, expands a level-synchronous BFS one frontier at a
time, and solves LeetCode 1091 with and without the heuristic.

![img](https://github.com/KingRei/100DaysPython/blob/master/day%2019%20-%20a%20star%20and%20parallel%20search/imgs/day19_1.png?raw=true)

Dijkstra orders its frontier by `g(v)` - the cost already paid. A* orders it by
`f(v) = g(v) + h(v)` - the cost already paid plus a guess of what is left. That is the
entire difference, and it is one line of code.

```python
key = lambda cell, g: g + h(cell)     # h = lambda cell: 0  gives Dijkstra back, exactly
```

The guess is not free advice. Three properties decide what you get back:

| property of h | meaning | what it buys |
|---|---|---|
| `h = 0` | no guess at all | Dijkstra |
| admissible | never overestimates the true remaining cost | the answer is still optimal |
| consistent | `h(u) <= w(u,v) + h(v)` | popped means final - no vertex is ever re-expanded |
| overestimating | the guess exceeds the truth | a path is still returned, just not the cheapest one |

## The map

The example is a 6x9 grid. Every cell costs 1 to step onto except a strip of mud in row 2
that costs 5, and it lies directly on the straight line between start and goal. Manhattan
distance points straight at it, which is what makes the map interesting: a good heuristic has
to be talked out of the obvious route by the arithmetic.

![img](https://github.com/KingRei/100DaysPython/blob/master/day%2019%20-%20a%20star%20and%20parallel%20search/imgs/day19_2.png?raw=true)

Dijkstra expands 49 cells. A* with Manhattan expands 13 and returns the identical path of
cost 10 - 27% of the work for the same answer. Nothing about correctness changed; the
heuristic only decided which parts of the map were never worth looking at.

## An admissible heuristic, checked rather than assumed

`h(start) = 8` while the true remaining cost is 10, so the guess is below the truth. The
script verifies both properties by brute force rather than asserting them in prose: it runs a
reverse Dijkstra from the goal for the true distances, compares every cell against `h`, and
checks the triangle inequality on every edge.

```python
def is_consistent(h):
    return all(h(u) <= w + h(v) for u in cells for v, w in neighbours(u))
```

Consistency is the property that makes the `closed` set safe. Without it a vertex can be
popped, marked final, and then reached again more cheaply - so the "if u in closed: continue"
line silently discards the better path.

## Inflating the guess

![img](https://github.com/KingRei/100DaysPython/blob/master/day%2019%20-%20a%20star%20and%20parallel%20search/imgs/day19_3.png?raw=true)

Weighted A* orders by `g + w*h`. Raising `w` shrinks the search and, past a point, starts
returning worse paths - and the failure is silent, because a search that overestimates simply
has a different opinion about what is worth exploring.

| ordering | cost | cells expanded | optimal? |
|---|---|---|---|
| `f = g` (Dijkstra) | 10 | 49 | yes |
| `f = g + h` (A*) | 10 | 13 | yes |
| `f = g + 5h` | 28 | 9 | no |
| `f = h` (greedy) | 28 | 9 | no |

The damage is bounded for weighted A*: with weight `w` the path costs at most `w` times the
optimum, which makes it a legitimate tool as long as the bound is stated. Greedy best-first
throws `g` away entirely and has no bound at all - on this map it returns a path 2.8 times
the optimum after walking straight into the mud.

## Delta-stepping

Dijkstra settles exactly one vertex per iteration, and that strict order is precisely what
makes it hard to parallelise. Delta-stepping relaxes a whole bucket of tentative distances at
once: bucket `i` holds distances in `[i*delta, (i+1)*delta)`, light edges (`w <= delta`) may
move a vertex back into the current bucket so it is re-scanned, and heavy edges are deferred
to the end of the phase where they can no longer shrink anything.

![img](https://github.com/KingRei/100DaysPython/blob/master/day%2019%20-%20a%20star%20and%20parallel%20search/imgs/day19_4.png?raw=true)

On the 14-vertex weighted graph in the script:

| delta | phases | largest batch | edge relaxations |
|---|---|---|---|
| 1 | 10 | 3 | 54 |
| 4 | 8 | 3 | 61 |
| 99 | 5 | 8 | 82 |

`delta -> 0` is Dijkstra, `delta -> infinity` is Bellman-Ford, and everything in between is
the same answer bought with a different mix of rounds and work. Halving the number of phases
costs 52% more edge relaxations, because a vertex settled in a wide bucket may not be final
yet and has to be relaxed again.

## Parallel BFS is a frontier, not a queue

![img](https://github.com/KingRei/100DaysPython/blob/master/day%2019%20-%20a%20star%20and%20parallel%20search/imgs/day19_5.png?raw=true)

The same idea with unit weights is much simpler: everything at distance `d` is independent,
so a level can be expanded all at once and the only shared state is the `seen` set.

```python
while frontier:
    levels.append(frontier)
    nxt = []
    for u in frontier:               # <- this loop is the parallel part
        for v, _ in adj[u]:
            if v not in seen:        # the only shared state
                seen.add(v); nxt.append(v)
    frontier = nxt
```

The work stays `O(V+E)`; what changes is the *depth* - the number of levels you must wait
through. On this grid that is 12 levels whose widest is 6 cells, and that 6 is the ceiling on
how many cores could ever be busy. Total work and critical-path depth are two separate costs,
and Dijkstra is hard to parallelise precisely because its depth is `V`.

## LeetCode 1091 - Shortest Path in Binary Matrix

Eight directions, and every cell on the path counts as one step. That changes which
heuristic is legal: a single diagonal move cuts the row gap and the column gap at the same
time, so the admissible guess is the Chebyshev distance, not the Manhattan one.

```python
h = lambda r, c: max(n - 1 - r, n - 1 - c)      # Manhattan would OVERESTIMATE here
```

Manhattan counts that diagonal step twice, exceeds the true remaining distance, and quietly
takes the optimality guarantee with it. The interview question is rarely "can you write A*";
it is whether you noticed that.

## Complexity

| Operation | Time | Space |
|---|---|---|
| A* / Dijkstra (binary heap) | O((V + E) log V) | O(V) |
| A* with a perfect h | O(path length) expansions | O(V) |
| delta-stepping (sequential) | O(V + E + phases) | O(V) |
| delta-stepping (parallel) | depth = number of phases | O(V) |
| level-synchronous BFS | O(V + E) work, depth = levels | O(V) |
| LC 1091 | O(n^2 log n) | O(n^2) |

## Run it

```bash
python astar.py
```

Prints the map, Dijkstra and A* footprints side by side, the admissibility and consistency
checks, the weight sweep from 1 to 8 plus greedy best-first, delta-stepping at five values of
delta with the phase batches printed out, the BFS level widths, and LeetCode 1091 - ending in
asserts.

## References

- [A* search algorithm](https://en.wikipedia.org/wiki/A*_search_algorithm)
- [Consistent heuristic](https://en.wikipedia.org/wiki/Consistent_heuristic)
- [Parallel single-source shortest path (delta-stepping)](https://en.wikipedia.org/wiki/Parallel_single-source_shortest_path_algorithm)
- [Parallel breadth-first search](https://en.wikipedia.org/wiki/Parallel_breadth-first_search)
- [LeetCode 1091 - Shortest Path in Binary Matrix](https://leetcode.com/problems/shortest-path-in-binary-matrix/)
