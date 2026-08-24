# Tarjan - one DFS with timestamps: bridges, articulation points, strongly connected components

More details in:
https://medium.com/100-days-of-python

Interactive walkthrough: [demo.html](demo.html) - a single self-contained page
(no build step, works offline, 中文 / English toggle) that animates the `disc`/`low`
bookkeeping edge by edge, shows how the same run answers bridges and articulation points,
demonstrates the parallel-edge trap that makes the textbook `if v == parent` version report a
bridge that does not exist, runs Tarjan's SCC algorithm with and without the `on_stack` test,
and solves LeetCode 1192 with the recursion rewritten as an explicit stack.

![img](https://github.com/KingRei/100DaysPython/blob/master/day%2018%20-%20tarjan%20scc%20and%20bridges/imgs/day18_1.png?raw=true)

A depth-first search already knows more than it usually admits. Record two numbers per
vertex while it runs - the time it was first reached, and the earliest time its subtree can
climb back to - and three classic graph structures fall out of the same traversal.

```
disc[u]   the clock value when the DFS first arrived at u
low[u]    the smallest disc reachable from u's subtree via tree edges
          plus at most one back edge
```

`low` answers one question: *can this subtree get above me without going through me?* The
three algorithms below are three readings of that single answer.

| test | meaning | graph |
|---|---|---|
| `low[v] > disc[u]` | the tree edge `u-v` is a bridge | undirected |
| `low[v] >= disc[u]` | `u` is an articulation point | undirected |
| `low[u] == disc[u]` | `u` is the root of an SCC | directed |

## Bridges

```python
def dfs(u, in_edge):
    disc[u] = low[u] = clock; clock += 1
    for v, eid in adj[u]:
        if eid == in_edge:          # skip the EDGE, not the vertex
            continue
        if disc[v] == -1:
            dfs(v, eid)
            low[u] = min(low[u], low[v])
            if low[v] > disc[u]:
                bridges.append((u, v))
        else:
            low[u] = min(low[u], disc[v])
```

A back edge contributes `disc[v]`, not `low[v]`: the question is which timestamp we can jump
back to, not where that vertex's own subtree reaches. On the example graph - two triangles
glued at vertex 2, joined by a single edge to a third - the only bridge is `4-5`.

## The parallel-edge trap

Most write-ups skip the parent with `if v == parent: continue`, and it passes almost every
test case. Run two cables between the same pair of routers and it breaks: both copies get
skipped, the second one never registers as a back edge, and the code reports a bridge that
physically is not there. Tracking the edge id costs one extra field and removes the whole
class of bug.

## Articulation points

![img](https://github.com/KingRei/100DaysPython/blob/master/day%2018%20-%20tarjan%20scc%20and%20bridges/imgs/day18_2.png?raw=true)

Deleting a *vertex* removes `u` itself as an escape route, so the subtree has to reach
strictly past `u` to survive - `>` becomes `>=`. The DFS root needs its own rule, because it
has no parent to climb back to: it is an articulation point exactly when it has more than one
child subtree. In the example, `2` and `4` are cut vertices but lie on no bridge at all -
removing the edge `2-4` keeps the graph connected, removing the vertex `2` does not.

## Strongly connected components

![img](https://github.com/KingRei/100DaysPython/blob/master/day%2018%20-%20tarjan%20scc%20and%20bridges/imgs/day18_3.png?raw=true)

In a directed graph, "u can reach v" no longer implies "v can reach u", so `low` needs a
guard: only a vertex still on the stack counts as an ancestor.

```python
for v in adj[u]:
    if disc[v] == -1:
        dfs(v)
        low[u] = min(low[u], low[v])
    elif on_stack[v]:               # only an unfinished ancestor counts
        low[u] = min(low[u], disc[v])
    # else: v belongs to a component that is already closed - ignore it
if low[u] == disc[u]:               # u is the root of an SCC
    pop the stack down to u
```

![img](https://github.com/KingRei/100DaysPython/blob/master/day%2018%20-%20tarjan%20scc%20and%20bridges/imgs/day18_4.png?raw=true)

Drop the `elif` and one edge - `5->2`, pointing into a component that was finished three
steps earlier - merges `{4,5}` with `{0}`, producing a component whose first vertex has no
incoming edges at all. Nothing crashes; the answer is simply wrong.

Contract each component to a single node and the result is always a DAG, and Tarjan emits the
components in reverse topological order of that DAG for free. That is day 13's topological
sort on the other side of the same traversal, and it is why 2-SAT, dead-code elimination and
circular-import detection are all built on this one function.

## LeetCode 1192 - Critical Connections in a Network

The problem is "find the bridges" in disguise, with one practical wrinkle: `n` goes up to
10^5 while CPython stops recursing at depth 1000. Rewriting the recursion means storing, per
frame, the thing the language normally remembers for us - a cursor into the neighbour list:

```python
stack = [(0, -1, iter(adj[0]))]
while stack:
    u, in_edge, it = stack[-1]
    for v, eid in it:               # the iterator resumes where it stopped
        ...
        stack.append((v, eid, iter(adj[v])))
        break                       # "call"
    else-branch: stack.pop()        # "return" - do the low[] merge here
```

## Complexity

| Operation | Time | Space |
|---|---|---|
| bridges | O(V + E) | O(V) |
| articulation points | O(V + E) | O(V) |
| Tarjan SCC | O(V + E) | O(V) |
| condensation DAG | O(V + E) | O(V + E) |
| LC 1192 (iterative) | O(V + E) | O(V) |

## Run it

```bash
python tarjan.py
```

Prints the timestamps vertex by vertex, the bridges and articulation points, the parallel-edge
failure side by side with the correct version, Tarjan's components with and without the
`on_stack` test, the condensation DAG, and LeetCode 1192 on a 3000-server chain that would
crash a recursive solution - ending in asserts.

## References

- [Bridge (graph theory)](https://en.wikipedia.org/wiki/Bridge_(graph_theory))
- [Biconnected component](https://en.wikipedia.org/wiki/Biconnected_component)
- [Tarjan's strongly connected components algorithm](https://en.wikipedia.org/wiki/Tarjan%27s_strongly_connected_components_algorithm)
- [LeetCode 1192 - Critical Connections in a Network](https://leetcode.com/problems/critical-connections-in-a-network/)
