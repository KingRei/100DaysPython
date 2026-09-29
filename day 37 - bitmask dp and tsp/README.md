# Bitmask DP and Held-Karp: indexing a DP table by a set of visited cities

More details in:
https://medium.com/100-days-of-python

Interactive walkthrough: [demo.html](demo.html) - a single self-contained page
(no build step, works offline, 中文 / English toggle) that collapses 24 route prefixes
into 12 `(mask, last)` states and then into 1 mask-only state so the missing field
becomes visible, fills the whole 16x5 Held-Karp table row by row and reads the optimal
tour back out of the parent pointers, runs the `dp[mask]`-only recurrence until it has
quietly grown a tree instead of a cycle, walks nearest neighbour into the inland trap
that costs it 106 km, and then solves LeetCode 943 with the very same table by swapping
`min` for `max`.

![img](https://github.com/KingRei/100DaysPython/blob/master/day%2037%20-%20bitmask%20dp%20and%20tsp/imgs/day37_1.png?raw=true)

From Day 33 to Day 36 the index of the DP table kept changing shape: a prefix length, a
remaining capacity, an interval, a subtree. Today it becomes a **set**. A set of up to
about twenty elements fits in a single machine integer, one bit per element, which means
it can be used directly as an array index - and once a subproblem may be keyed by "which
things are already done", a whole family of problems that look hopelessly permutational
becomes a table fill.

The example is the travelling salesman problem over eight stops in Taiwan. There are
5,040 possible itineraries; Held-Karp finds the shortest one, 689 km, by filling 2^8 x 8
cells instead of enumerating them. The interesting part is not the speed-up. It is that
the two natural ways to get this wrong both return a plausible number and never raise
anything.

## A set is an integer

![img](https://github.com/KingRei/100DaysPython/blob/master/day%2037%20-%20bitmask%20dp%20and%20tsp/imgs/day37_2.png?raw=true)

Five operations carry the whole day. Bit `i` of `mask` stands for element `i`:

```python
def has(mask, i):    return (mask >> i) & 1
def add(mask, i):    return mask | (1 << i)
def remove(mask, i): return mask & ~(1 << i)
def members(mask):   return [i for i in range(mask.bit_length()) if mask >> i & 1]
def popcount(mask):  return bin(mask).count('1')
```

`members` iterates only up to `bit_length()`, so it costs nothing on sparse masks.
`popcount` is `int.bit_count()` from Python 3.10 onward; the `bin(...).count('1')` form
is kept here because it works everywhere and is not the bottleneck.

Two precedence notes, because both are worth saying out loud. In C, `mask & 1 == 0`
parses as `mask & (1 == 0)` and is a classic bug; in Python `&` binds *tighter* than
`==`, so the expression means what it looks like. And `~mask` in Python is negative
(`~0b0101 == -6`), which is fine as a value but dangerous as an index into a Python
list - `dp[~mask]` silently reads from the other end. For a table of exactly `2**n`
entries, `mask ^ full` is the complement you actually want.

## Enumerating submasks

Iterating over every subset of a mask is a two-line idiom that looks like a typo:

```python
sub = mask
while True:
    yield sub
    if sub == 0:
        break
    sub = (sub - 1) & mask
```

Subtracting one borrows through the low zero bits, and the `& mask` throws away the
borrowed bits that do not belong to the set. Written as `while sub:` instead of the
do-while above, it skips the empty subset - which is exactly the subset that usually
carries the base case.

The total work of "for every mask, for every submask" is not `4^n`. Every element is in
the submask, in the mask but not the submask, or in neither, so the sum is `3^n`: 59,049
for `n = 10`, not 1,048,576. That factor of about eighteen is what makes subset-partition
DPs practical at all.

## The fill order is free this time

Day 36's whole difficulty was the order the table was filled in: an interval DP has to
go by increasing length, and the instinctive `for i: for j:` loops read cells that have
not been written yet. Bitmask DP gets that for free.

`mask | (1 << nxt)` is always numerically **larger** than `mask`, because `nxt` was not
in `mask`. So a plain `for mask in range(1 << n)` never reads an unfilled cell, whatever
the transition is. The module checks this directly - across all masks and all supersets
reachable in one step, the number of subsets that are numerically larger than their
superset is zero.

## Held-Karp

The state is a pair: which cities have been visited, and which one I am standing on.

```python
dp[1 << 0][0] = 0
for mask in range(1 << n):
    for last in members(mask):
        if dp[mask][last] == INF:
            continue
        for nxt in range(n):
            if has(mask, nxt):
                continue
            nm = add(mask, nxt)
            cand = dp[mask][last] + D[last][nxt]
            if cand < dp[nm][nxt]:
                dp[nm][nxt] = cand
                parent[nm][nxt] = last
best = min(dp[full][last] + D[last][0] for last in range(1, n))
```

The `last` field is the entire point. Two prefixes that have visited the same set of
cities and finish at the same city are interchangeable for everything that follows, so
only the cheaper one needs to survive. Twenty-four prefixes of length four over five
cities collapse to twelve states; at `n` cities it is `(n-1)!` orderings against
`2^n * n` states.

`parent` is what turns a number back into an itinerary. Nothing in the recurrence needs
it, which is why it is so often left out and so painful to add back later.

There is also a top-down version with `functools.lru_cache` on `(mask, last)`, which
reaches 449 memoised states for the eight-city instance instead of the full 2,048 - it
only touches the states that are actually reachable, at the cost of Python's recursion
and hashing overhead.

## The silent failure: dropping `last`

![img](https://github.com/KingRei/100DaysPython/blob/master/day%2037%20-%20bitmask%20dp%20and%20tsp/imgs/day37_3.png?raw=true)

Suppose the state is only `dp[mask]`. The recurrence still compiles, still terminates,
still returns a number. But without knowing where the previous prefix ended, the only
cost that can be written for the next hop is the distance to the *nearest already visited
city*:

```python
hop = min(D[i][nxt] for i in members(mask))
dp[mask | 1 << nxt] = min(dp[mask | 1 << nxt], dp[mask] + hop)
```

That is not a broken recurrence. It is a correct recurrence for a different problem: it
attaches each new city to whichever visited city is closest, which grows a **tree**, not
a tour. Nobody ever comes home. The number it returns for the eight-city instance is 502
km - and the minimum spanning tree of the same instance also weighs 502 km, because that
is precisely what has been computed. A classic TSP lower bound, always smaller than the
true optimum of 689 km, always plausible, never flagged.

## Nearest neighbour: right shape, wrong number

![img](https://github.com/KingRei/100DaysPython/blob/master/day%2037%20-%20bitmask%20dp%20and%20tsp/imgs/day37_5.png?raw=true)

The other way to be wrong is to skip the DP entirely and hop to the closest unvisited
city every time. This one produces a genuine closed tour, in `O(n^2)`, and the answer
*looks* like a reasonable itinerary: 795 km against the optimal 689, 15.4% over.

The damage is done at one step. Standing at Sun Moon Lake, the closest unvisited city is
Hualien at 74 km, nearer than Tainan at 116 km. Taking it strands the entire south -
Tainan, Kaohsiung, Taitung - to be collected later, and that detour costs far more than
the 42 km just saved. Greedy cannot see it, because its state has no record of which
cities are left.

The two failures bracket the right answer, which is why they are worth remembering
together: the tree is always below the optimum, the greedy tour always above it, and
neither raises an exception. "It ran" is not evidence that a TSP implementation is
correct.

## Two exponentials

![img](https://github.com/KingRei/100DaysPython/blob/master/day%2037%20-%20bitmask%20dp%20and%20tsp/imgs/day37_4.png?raw=true)

`2^n * n^2` is still exponential - Held-Karp does not make the TSP tractable, it moves
the wall. Measured in this module, brute force takes 8.6 ms at `n = 8`, 98.8 ms at 9, and
890 ms at 10; Held-Karp takes 1.2 ms at 8 and 614 ms at 15. Extrapolating the factorial
curve, brute force at `n = 15` would need roughly 42 hours.

| n | (n-1)! | 2^n · n² |
|---|---|---|
| 8 | 5,040 | 16,384 |
| 10 | 362,880 | 102,400 |
| 13 | 479,001,600 | 1,384,448 |
| 15 | 87,178,291,200 | 7,372,800 |
| 20 | 121,645,100,408,832,000 | 419,430,400 |

Past roughly twenty cities the table stops fitting in memory and the answer has to come
from somewhere else: a heuristic that gives up on optimality, such as a genetic algorithm
or simulated annealing, or a solver that prunes the search with the very lower bound the
broken recurrence above accidentally computes.

## LeetCode 847 - Shortest Path Visiting All Nodes

**The task.** Given a connected undirected graph, find the length of the shortest walk
that visits every node at least once. The walk may start at any node, end at any node,
and revisit nodes and edges as often as it likes.

**Input / output.** `graph` is an adjacency list: `graph[i]` lists the neighbours of node
`i`. The return value is the number of edges in the shortest such walk.

**Example.** For `graph = [[1,2,3],[0],[0],[0]]` the answer is 4. Node 0 is a hub joined
to three leaves, so a walk such as 1 - 0 - 2 - 0 - 3 uses four edges, and the hub has to
be passed through twice.

**Constraints.** `n` is at most 12, which is the whole hint: `2^12 * 12` states is small,
`12!` is not. Revisiting is allowed, so this is not a Hamiltonian path problem and the
cost of a step is always one edge.

Because every edge costs the same, this is a BFS rather than a table fill, but the state
is exactly today's: `(mask of visited nodes, current node)`. Seed the queue with `(1 << i,
i)` for every `i` - any node may be the start - and stop the first time `mask` is full.
The single thing to get right is that a state is marked as seen per `(mask, node)` pair,
not per node; marking per node is what makes an implementation return a too-large answer
on graphs that require backtracking.

[leetcode.com/problems/shortest-path-visiting-all-nodes](https://leetcode.com/problems/shortest-path-visiting-all-nodes/)

## LeetCode 943 - Find the Shortest Superstring

**The task.** Given a list of strings, arrange all of them into one string that contains
every input string as a substring, and make that string as short as possible. Any one of
the shortest answers may be returned.

**Input / output.** `words` is a list of strings; the return value is a single string.

**Example.** For `words = ["catg","ctaagt","gcta","ttca","atgcatc"]` one shortest answer
is `gctaagttcatgcatc`, 16 characters. The five inputs total 25 characters, and the order
`gcta, ctaagt, ttca, catg, atgcatc` overlaps away 9 of them.

**Constraints.** At most 12 words, each at most 20 characters, and no word is a substring
of another - that last clause is what makes "concatenate in some order" the whole search
space, and 12 is again the bitmask hint.

This is the TSP with two substitutions: distance becomes overlap, `min` becomes `max`.
Precompute `ov[a][b]`, the number of characters of `b`'s head that `a`'s tail already
supplies, then run the same `dp[mask][last]` fill maximising total overlap. Minimising
the final length and maximising the overlap are the same objective, because the sum of
the input lengths is a constant.

The trap is the seeding. In a closed tour any city may be called the start, so seeding
only `dp[1][0]` is free; here the answer is an open chain and the first word is a real
decision. Seeding one start still returns a valid superstring - on this input,
`catgcatcgctaagttca`, 18 characters instead of 16. Longer, legal, and easy to miss on
small test cases.

[leetcode.com/problems/find-the-shortest-superstring](https://leetcode.com/problems/find-the-shortest-superstring/)

## LeetCode 1125 - Smallest Sufficient Team

**The task.** A project needs a list of required skills. Each candidate person masters
some subset of them. Pick the smallest team of people that together covers every required
skill.

**Input / output.** `req_skills` is a list of distinct skill names; `people[i]` is the
list of skills person `i` has. Return the indices of a smallest sufficient team, in any
order; any one of the smallest teams is accepted.

**Example.** With `req_skills = ["java","nodejs","reactjs"]` and
`people = [["java"],["nodejs"],["nodejs","reactjs"]]` the answer is `[0, 2]`: person 0
brings java, person 2 brings both nodejs and reactjs, and no single person covers
everything.

**Constraints.** At most 16 required skills and at most 60 people, and the problem
guarantees a sufficient team exists. Sixteen skills means `2^16` masks, so the state is
the *set of covered skills* and people are iterated over that - not the other way round.

`dp[covered]` holds the smallest team reaching that coverage. For each person, `new =
covered | skill_mask[p]`, and the entry is replaced when the team gets shorter. Note that
this is a set-cover problem and set cover is NP-hard in general; the 16-skill bound is
what makes the exhaustive table legitimate here, and greedily taking the person who adds
the most new skills gives a team that is sometimes larger than optimal.

[leetcode.com/problems/smallest-sufficient-team](https://leetcode.com/problems/smallest-sufficient-team/)

## LeetCode 526 - Beautiful Arrangement

**The task.** Count the permutations of `1..n` in which every element is *compatible*
with its position: for each position `i` (1-based), either the number placed there
divides `i`, or `i` divides the number.

**Input / output.** An integer `n`; return the number of such permutations.

**Example.** For `n = 2` the answer is 2: `[1,2]` works because 1 divides 1 and 2 divides
2, and `[2,1]` works because 2 is divisible by position 1 and 1 divides position 2.

**Constraints.** `1 <= n <= 15`. Fifteen factorial is 1.3 trillion, `2^15 * 15` is about
half a million - the gap between those two numbers is the problem.

The state is only the mask: `popcount(mask)` already says which position is being filled,
so no second field is needed. `dp[mask] = sum(dp[mask without i])` over every `i` that is
compatible with position `popcount(mask)`. That is the cleanest example in the set of
*not* needing a `last` field - the position is recoverable from the mask itself, which is
the same argument that makes the TSP's missing field so easy to overlook.

[leetcode.com/problems/beautiful-arrangement](https://leetcode.com/problems/beautiful-arrangement/)

## LeetCode 698 - Partition to K Equal Sum Subsets

**The task.** Decide whether an array of positive integers can be split into exactly `k`
non-empty groups whose sums are all equal. Every element must land in exactly one group.

**Input / output.** An integer array `nums` and an integer `k`; return `True` or `False`.

**Example.** `nums = [4,3,2,3,5,2,1]`, `k = 4` is `True`: the total is 20, so each group
must sum to 5, and `[5]`, `[1,4]`, `[2,3]`, `[2,3]` works. `nums = [1,2,3,4]`, `k = 3` is
`False`, because 10 is not divisible by 3.

**Constraints.** At most 16 elements, each at most `10^4`, and `k` at most `nums.length`.
Sixteen elements is the bitmask bound again.

The neat formulation carries a single number per mask: `used[mask]` is the sum already
placed in the *current, partially filled* group, with completed groups implied by the
rest. A new element may join only if it does not overflow the target, and when the running
sum hits the target exactly it wraps back to zero and a new group starts. Two cheap
prefilters do most of the work in practice: the total must be divisible by `k`, and no
single element may exceed the target.

[leetcode.com/problems/partition-to-k-equal-sum-subsets](https://leetcode.com/problems/partition-to-k-equal-sum-subsets/)

## Complexity

| Operation | Time | Space |
|---|---|---|
| `has` / `add` / `remove` / `popcount` | O(1) on machine-word masks | O(1) |
| enumerate all submasks of every mask | O(3^n) | O(1) |
| Held-Karp TSP, table + parents | O(2^n · n²) | O(2^n · n) |
| brute-force TSP | O(n!) | O(n) |
| nearest neighbour | O(n²) | O(n) |
| LeetCode 847, BFS over `(mask, node)` | O(2^n · n²) | O(2^n · n) |
| LeetCode 943, shortest superstring | O(2^n · n² + n²·L) | O(2^n · n) |
| LeetCode 1125, smallest sufficient team | O(2^s · p) | O(2^s · p) |
| LeetCode 526, beautiful arrangement | O(2^n · n) | O(2^n) |
| LeetCode 698, k equal subsets | O(2^n · n) | O(2^n) |

## References

- [Travelling salesman problem - Wikipedia](https://en.wikipedia.org/wiki/Travelling_salesman_problem)
- [Held-Karp algorithm - Wikipedia](https://en.wikipedia.org/wiki/Held%E2%80%93Karp_algorithm)
- Michael Held and Richard M. Karp, "A Dynamic Programming Approach to Sequencing Problems", 1962
- [Bit manipulation - Wikipedia](https://en.wikipedia.org/wiki/Bit_manipulation)
- [Set cover problem - Wikipedia](https://en.wikipedia.org/wiki/Set_cover_problem)
