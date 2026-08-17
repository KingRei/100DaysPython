# Topological Sort - Kahn and DFS

More details in:
https://medium.com/100-days-of-python/day-13-%E6%8B%93%E6%A8%B8%E6%8E%92%E5%BA%8F-kahn-dfs-%E4%BB%A5%E5%8F%8A%E4%BD%A0%E7%9A%84-build-%E7%82%BA%E4%BB%80%E9%BA%BC%E6%9C%83%E5%8D%A1%E4%BD%8F-b31f3edc4870

Interactive walkthrough: [demo.html](demo.html) - a single self-contained page
(no build step, works offline) that animates Kahn, the DFS post-order, cycle detection
and LeetCode 207/210 step by step.

![img](https://github.com/KingRei/100DaysPython/blob/master/day%2013%20-%20topological%20sort/imgs/day13_1.png?raw=true)

A topological order of a directed graph is a linear ordering of its vertices such that
for every edge `u -> v`, `u` appears before `v`. It is the answer to any question of the
form "in what order may I do these things, given that some of them depend on others":
build targets, course prerequisites, spreadsheet recalculation, package installation,
task graphs in a scheduler.

Such an ordering exists **if and only if** the graph is a directed acyclic graph (DAG).
A cycle means a set of items that each transitively depend on themselves, and no linear
order can satisfy them all. Sorting the graph and proving it acyclic are therefore the
same computation, which is why both algorithms below double as cycle detectors.

A DAG usually has many valid topological orders; the algorithms differ in which one they
happen to produce, and in what else they hand you along the way.

## Kahn's algorithm

![img](https://github.com/KingRei/100DaysPython/blob/master/day%2013%20-%20topological%20sort/imgs/day13_2.png?raw=true)

Compute the in-degree of every vertex and put the zero-in-degree vertices in a queue.
Pop one, append it to the output, and decrement the in-degree of each of its successors;
any successor whose in-degree reaches zero joins the queue. If fewer than `V` vertices
come out, the remainder lie in or downstream of a cycle.

Taking the entire frontier each iteration instead of one vertex groups the output into
levels. Everything within a level is mutually independent, so the levels are exactly the
rounds a parallel scheduler can dispatch, and the number of levels is the length of the
longest path through the graph.

Replacing the queue with a min-heap yields the lexicographically smallest topological
order, at O(V log V + E).

## DFS post-order

![img](https://github.com/KingRei/100DaysPython/blob/master/day%2013%20-%20topological%20sort/imgs/day13_3.png?raw=true)

Run a depth-first search and append a vertex to a list at the moment it finishes, i.e.
once no unexplored outgoing edge remains. Everything reachable from that vertex is
already in the list, so reversing the finish order produces a topological order.

Cycle detection uses three colours: white (unvisited), grey (on the current recursion
path), black (finished). Encountering a grey vertex means a back edge onto the current
path, hence a cycle. Colouring is what distinguishes this from plain Day 12 DFS, where a
single visited set suffices.

## Cycles

![img](https://github.com/KingRei/100DaysPython/blob/master/day%2013%20-%20topological%20sort/imgs/day13_4.png?raw=true)

`None` is a poor diagnostic for a build system. Recording each vertex's parent during the
DFS lets the back edge be turned into the actual cycle by walking the parent chain back
to the grey vertex, which is what a scheduler should report to the user.

## Complexity

| Operation | Time | Space |
|---|---|---|
| Kahn's algorithm | O(V + E) | O(V) |
| DFS post-order | O(V + E) | O(V) |
| Levels / parallel rounds | O(V + E) | O(V) |
| Lexicographically smallest order | O(V log V + E) | O(V) |
| Cycle extraction | O(V + E) | O(V) |

## LeetCode 207 - Course Schedule

`prerequisites[i] = [a, b]` means course `b` must be taken before course `a`, so the edge
runs `b -> a`; reversing it is the single most common mistake on this problem. The
question "can all courses be finished" is then "is this graph a DAG", one call to Kahn's
algorithm. LeetCode 210 asks for the order itself and is the same code returning the
array instead of a boolean.

## Run it

```sh
python topological_sort.py     # standalone walkthrough, no Jupyter required
```

Or open `demo.html` in a browser for the animated version (中文 / English toggle).

## References

- [Wiki - Topological sorting](https://en.wikipedia.org/wiki/Topological_sorting)
- [Wiki - Directed acyclic graph](https://en.wikipedia.org/wiki/Directed_acyclic_graph)
- [LeetCode 207 - Course Schedule](https://leetcode.com/problems/course-schedule/)
- [LeetCode 210 - Course Schedule II](https://leetcode.com/problems/course-schedule-ii/)
