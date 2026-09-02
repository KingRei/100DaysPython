# Minimum Spanning Tree - Kruskal

More details in:
https://medium.com/100-days-of-python/day-15-%E6%9C%80%E5%B0%8F%E7%94%9F%E6%88%90%E6%A8%B9-kruskal-cae4865723fa

Interactive walkthrough: [demo.html](demo.html) - a single self-contained page
(no build step, works offline, 中文 / English toggle) that animates the sorted edge scan,
what happens when the graph is disconnected, the cut and cycle properties that make the
greedy choice safe, and LeetCode 1584 step by step.

![img](https://github.com/KingRei/100DaysPython/blob/master/day%2015%20-%20minimum%20spanning%20tree%20-%20kruskal/imgs/day15_1.png?raw=true)

A network of seven sites has to be wired together. Every possible cable has a price, and
the only requirement is that any site can reach any other; nobody cares how many hops it
takes. Buying every cable works and is wasteful, so the question is which subset of cables
to buy.

The answer is always a tree. Connecting *n* sites needs at least *n - 1* cables, and any
extra cable closes a cycle, meaning one edge of that cycle could be removed without
disconnecting anything. A **spanning tree** is a subset of edges that touches every vertex
and contains no cycle; a **minimum spanning tree** is the spanning tree whose total weight
is smallest.

**Kruskal's algorithm** finds one in two lines of real work: sort every edge by weight,
then walk the list from cheapest to most expensive and keep an edge whenever its two
endpoints are not yet connected. That last test is exactly the question
[Day 14's Union-Find](../day%2014%20-%20union%20find) answers in near-constant time, which
is why the two days sit next to each other.

![img](https://github.com/KingRei/100DaysPython/blob/master/day%2015%20-%20minimum%20spanning%20tree%20-%20kruskal/imgs/day15_2.png?raw=true)

```python
def kruskal(n, edges):
    uf, chosen, total = UnionFind(n), [], 0
    for w, u, v in sorted(edges):
        if uf.union(u, v):          # False when u and v already share a root
            chosen.append((w, u, v))
            total += w
            if len(chosen) == n - 1:
                break
    return chosen, total
```

`union` returns `False` when the two endpoints already have the same root, and that is
precisely the case where adding the edge would close a cycle. No cycle detection code is
needed - the disjoint set structure *is* the cycle detector.

## Why greedy is safe here

Greedy algorithms are usually wrong, so it is worth knowing why this one is not.

![img](https://github.com/KingRei/100DaysPython/blob/master/day%2015%20-%20minimum%20spanning%20tree%20-%20kruskal/imgs/day15_3.png?raw=true)

The **cut property**: split the vertices into two non-empty groups any way you like. The
cheapest edge with one endpoint in each group belongs to some minimum spanning tree.
Suppose a minimum tree *T* omitted that edge *e*. Adding *e* to *T* creates exactly one
cycle, and that cycle has to cross back over the split, so it contains another crossing
edge *f* with weight at least that of *e*. Swap *f* out for *e* and the result is still a
spanning tree, and no heavier. Every edge Kruskal accepts is the cheapest one crossing the
cut between the component it is growing and everything else, so every accepted edge is
safe.

The **cycle property** is the mirror image: the heaviest edge of any cycle is in no minimum
spanning tree, because deleting it from a tree that contained it splits the tree in two and
the rest of the cycle reconnects the halves more cheaply. Every edge Kruskal rejects is the
heaviest edge of the cycle it would have closed - it is scanned last of all the edges on
that cycle - so every rejection is safe too.

## When the graph is disconnected

Kruskal never asks whether the graph is connected. If it is not, the scan simply ends with
fewer than *n - 1* edges and what comes out is a **spanning forest**: a minimum spanning
tree of each connected component. Detach vertex G from the example graph and the run keeps
5 edges totalling 30 instead of 6 totalling 39. Checking `len(chosen) == n - 1` afterwards
is how you tell the two cases apart.

## LeetCode 1584 - Min Cost to Connect All Points

The input is a list of points and the cost of joining two of them is their Manhattan
distance. There is no edge list, so build one: every pair of points is an edge, which gives
a complete graph with `n(n-1)/2` of them, and hand that to Kruskal unchanged.

The caveat worth saying out loud in an interview is the complexity. Sorting `n²` edges costs
`O(n² log n)`, and for a dense graph like this Prim's algorithm - which grows a single tree
and only ever looks at edges touching it - is the better fit. Kruskal shines on sparse
graphs, where *E* is close to *V* and the sort is cheap.

## Complexity

| Step | Time | Space |
|---|---|---|
| sort the edges | O(E log E) | O(E) |
| scan with Union-Find | O(E α(V)) | O(V) |
| total | **O(E log E)** = O(E log V) | O(V + E) |

The sort dominates. If the weights are small integers, or the edges arrive already sorted,
the scan alone is effectively linear.

## Run it

```sh
python kruskal.py     # standalone walkthrough, no Jupyter required
```

The script also brute-forces every 6-edge subset of the example graph and confirms that 39
really is the minimum, so a regression cannot pass quietly.

Or open `demo.html` in a browser for the animated version (中文 / English toggle).

## References
- [Wikipedia - Kruskal's algorithm](https://en.wikipedia.org/wiki/Kruskal%27s_algorithm)
- [Wikipedia - Minimum spanning tree](https://en.wikipedia.org/wiki/Minimum_spanning_tree)
- [LeetCode 1584 - Min Cost to Connect All Points](https://leetcode.com/problems/min-cost-to-connect-all-points/)
