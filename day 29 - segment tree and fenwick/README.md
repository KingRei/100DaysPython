# Segment tree and Fenwick tree: range queries when the data keeps changing

More details in:
https://medium.com/100-days-of-python

Interactive walkthrough: [demo.html](demo.html) - a single self-contained page
(no build step, works offline, 中文 / English toggle) that watches a prefix-sum array go stale
one cell at a time, cuts a query range into whole subtrees on an iterative segment tree, replays
a lazy range-add against the eager version that walks to every leaf, unfolds the `i & -i`
staircase behind a Fenwick tree and both of its walks, shows a prefix-min BIT quietly returning
the wrong answer, and finishes with LeetCode 307, 315 and 370.

![img](https://github.com/KingRei/100DaysPython/blob/master/day%2029%20-%20segment%20tree%20and%20fenwick/imgs/day29_1.png?raw=true)

Two operations, one array: write a single cell, and ask for the sum of a slice. Each one has an
obvious data structure, and the two structures are mirror images. A plain array updates in
`O(1)` and answers a range sum in `O(n)`. A prefix-sum array answers in `O(1)` and needs `O(n)`
to absorb one write, because changing `a[2]` invalidates every prefix from index 2 onward. On a
million elements that is a million adds per query, or a million rewrites per update. Everything
below exists to refuse that choice.

Both structures reach `O(log n)` for both operations, and they are usually presented as
alternatives with a shrug about constant factors. They are not interchangeable, and the reason
is not speed - it is at the bottom of this page.

## Segment tree: cut the range into whole subtrees

```python
def query(self, l, r):              # half open [l, r)
    res, l, r = 0, l + self.n, r + self.n
    while l < r:
        if l & 1: res += self.t[l]; l += 1     # l is a right child: take it
        if r & 1: r -= 1; res += self.t[r]     # r is a right child: take r-1
        l >>= 1; r >>= 1
    return res
```

The iterative form stores the tree in a flat array of size `2n`: leaf `i` lives at `t[n + i]`
and the parent of `k` is `k >> 1`. No node objects, no recursion, no pointers - and updating a
leaf is just a walk up the parent chain, `t[k] = t[2k] + t[2k+1]`, which is `log2(n)` additions.

The query is the interesting half. It converts the range to leaf positions and climbs, harvesting
a node only when that node is entirely inside the range. `canonical_cover(16, 3, 13)` returns
`[5, 6, 19, 28]`: **four stored sums answer a ten-element query**. The bound is `2·log2(n)`, and
the reason is worth internalising - on each level at most one node can be a partial overhang on
the left and one on the right, so at most two nodes per level are ever taken. Measured over
random ranges: `n = 1024` averages 7.29 nodes against a bound of 20, `n = 65536` averages 13.25
against 32, `n = 1000000` averages 17.03 against 38.

Because the query only ever *combines disjoint pieces*, the merge can be any associative
operation. Sum, min, max, gcd, matrix product, "the largest sum of a contiguous run" - all the
same tree with a different two-line merge.

![img](https://github.com/KingRei/100DaysPython/blob/master/day%2029%20-%20segment%20tree%20and%20fenwick/imgs/day29_2.png?raw=true)

## Lazy propagation: the same idea, applied to updates

Adding `delta` to every element of a range by touching each leaf is `O(n)` and throws away the
tree entirely. But the query already knows how to cut a range into whole subtrees, and a node
that is *fully covered* does not need its children's help: its sum increases by exactly
`delta * width`. So apply it there, leave a note - `lazy[node] += delta` - and stop.

```python
def _push(self, k, lo, hi):
    if self.lazy[k]:
        mid = (lo + hi) // 2
        self._apply(2*k,   self.lazy[k], mid - lo + 1)
        self._apply(2*k+1, self.lazy[k], hi - mid)
        self.lazy[k] = 0
```

The note is only cashed in when somebody actually descends past that node, which is what `_push`
does on the way down. It is deferred work, not skipped work, and the accounting is exact: a
node's stored sum is *always* correct, while its children may be out of date by whatever is
recorded in `lazy`.

On 100000 elements and 2000 random range-adds the eager version makes 100775266 node visits and
the lazy version 116963, with 42681 pushes - **862 times fewer**. This is also the pattern behind
`std::deque`-style bulk operations, interval scheduling and every "apply an operator to a whole
subtree" trick further up the stack; the tag does not have to be a number, it just has to be
composable.

![img](https://github.com/KingRei/100DaysPython/blob/master/day%2029%20-%20segment%20tree%20and%20fenwick/imgs/day29_3.png?raw=true)

## Fenwick tree: throw the tree away, keep the bits

A Fenwick tree - Peter Fenwick's 1994 paper calls it a **binary indexed tree**, and the two names
are the same structure - stores `n + 1` numbers and no links at all. The entire design is one
expression, `i & -i`, the lowest set bit of `i`:

```python
def prefix(self, i):                 # sum of a[0:i]
    s = 0
    while i > 0:
        s += self.t[i]
        i -= i & -i                  # strip the lowest set bit
    return s

def add(self, i, delta):             # a[i] += delta
    i += 1
    while i <= self.n:
        self.t[i] += delta
        i += i & -i                  # climb to the next responsible cell
```

`t[i]` holds the sum of the `i & -i` elements *ending at* `i`. So `t[8]` covers `a[0:8]`, `t[12]`
covers `a[8:12]`, `t[6]` covers `a[4:6]`, and `t[7]` covers just `a[6:7]`. The staircase is not a
convention someone chose; it falls out of binary counting. `prefix(13)` peels the set bits of
`1101`: `13 → 12 → 8 → 0`, three reads, one per set bit. `add` climbs the complementary chain.
Both loops run at most `log2(n)` times, and in practice fewer - the average `prefix` on
`n = 1000000` takes 9.88 steps against a worst case of 19.

The payoff is memory and constants. On a million 64-bit cells: an iterative segment tree needs
`2n`, a recursive one `4n`, a lazy one `8n`, and a Fenwick tree `n + 1`. On 20000 elements with
2000 update-plus-query pairs, the Fenwick tree finished in 22.5 ms against 44.0 ms for the
segment tree, 87.4 ms for the plain array and 9068.1 ms for prefix sums. Six lines of code, half
the memory of the simplest tree, twice its speed.

![img](https://github.com/KingRei/100DaysPython/blob/master/day%2029%20-%20segment%20tree%20and%20fenwick/imgs/day29_4.png?raw=true)

It also does more than prefix sums. Running one BIT over the *difference* array turns a range add
into two point updates and a point query into one prefix sum. Running **two** BITs recovers full
range-add plus range-sum, with
`prefix(i) = B1.prefix(i) * i - B2.prefix(i)` - the same asymptotics as a lazy segment tree in a
quarter of the memory. And `lower_bound` - the smallest index whose prefix reaches a target - is
a binary lift over the same array in `O(log n)`, with no extra structure, which is how a Fenwick
tree becomes a weighted sampler.

## What a Fenwick tree cannot do

```python
a = [5, 3, 8, 1]; a[3] = 9
real min          = 3
prefix-min BIT    = 1     # wrong - it never forgot the 1
min segment tree  = 3
```

Every BIT range query is `range(l, r) = prefix(r) - prefix(l)`. That subtraction is not an
implementation detail, it is a **requirement on the operation**: there has to be an inverse.
Addition has one, xor has one, multiplication over a field has one. `min` does not. Once a 1 has
been folded into a cell there is no operation that takes it back out, so the structure can only
ever let values decrease, and when they increase it returns a plausible, wrong number without
complaining.

A segment tree never subtracts - it combines *disjoint* pieces - so associativity is the only
requirement, and it survives `min`, `max`, `gcd` and matrix products. That is the real dividing
line between the two structures. Not the constant factor, not the memory: **can this operation
be undone?**

![img](https://github.com/KingRei/100DaysPython/blob/master/day%2029%20-%20segment%20tree%20and%20fenwick/imgs/day29_5.png?raw=true)

## The LeetCode problems

LC 307 (Range Sum Query - Mutable) is the canonical problem for the whole topic, and it is
constructed so that neither naive answer passes. Both structures solve it in a few lines, but the
BIT version has a detail interviewers look for: **a BIT stores deltas, not values**, so
"set `a[i]` to `val`" has to be written `add(i, val - a[i])`, which means keeping the old array
alongside the tree. Miss it and the tree drifts silently. The other easy loss is the interval
convention - LeetCode's `sumRange` is inclusive, the half-open `range_sum` needs `r + 1`.

LC 315 (Count of Smaller Numbers After Self) is the reduction worth remembering: compress the
values to ranks, scan **right to left** so that "after me" becomes "already seen", and use the
BIT as a histogram. Each element then costs one `prefix(rank)` and one `add`. At `n = 4000` that
is around 30 times faster than the double loop, and the same move - turn a counting question into
a prefix query - powers inversion counts, sliding-window ranks and offline range statistics.

LC 370 (Range Addition) is the deliberate anticlimax. All the updates arrive before any query, so
a difference array in `O(n + k)` is optimal and no tree helps at all. That is the question to ask
first, straight from the problem statement: **are the updates and the queries interleaved?** If
they are not, use prefix sums. If they are and the operation has an inverse, a Fenwick tree is
six lines. If they are and it does not, build the segment tree.

![img](https://github.com/KingRei/100DaysPython/blob/master/day%2029%20-%20segment%20tree%20and%20fenwick/imgs/day29_6.png?raw=true)

## Complexity

`n` = number of elements, `k` = number of updates.

| Operation | Time | Space |
|---|---|---|
| plain array: point update / range sum | O(1) / O(n) | n |
| prefix sums: point update / range sum | O(n) / O(1) | n |
| segment tree build (iterative, bottom up) | O(n) | 2n |
| segment tree point update / range query | O(log n), <= 2·log2(n) nodes | O(1) |
| lazy segment tree range update / range query | O(log n) | 2·(2n) with the tags |
| eager range update (descend to every leaf) | O(n) | 2n |
| Fenwick build from an array | O(n) | n + 1 |
| Fenwick `add` / `prefix` | O(log n), one step per set bit | O(1) |
| Fenwick `lower_bound` (binary lifting) | O(log n) | O(1) |
| one BIT over a difference array: range add + point query | O(log n) | n + 1 |
| two BITs: range add + range sum | O(log n) | 2(n + 1) |
| min / max / gcd segment tree query | O(log n) | 2n |
| min via a BIT | not possible - no inverse | - |
| LC 307 (both structures) | O(log n) per operation | O(n) |
| LC 315 `count_smaller` (BIT over ranks) | O(n log n) | O(n) |
| LC 370 `get_modified_array` (difference array) | O(n + k) | O(n) |

## References
- [Segment tree - Wikipedia](https://en.wikipedia.org/wiki/Segment_tree)
- [Fenwick tree - Wikipedia](https://en.wikipedia.org/wiki/Fenwick_tree)
- [A new data structure for cumulative frequency tables (Fenwick, 1994)](https://doi.org/10.1002/spe.4380240306)
- [Efficient and easy segment trees (Codeforces, Al.Cash)](https://codeforces.com/blog/entry/18051)
- [Segment tree - cp-algorithms](https://cp-algorithms.com/data_structures/segment_tree.html)
- [Fenwick tree - cp-algorithms](https://cp-algorithms.com/data_structures/fenwick.html)
- [LeetCode 307 · Range Sum Query - Mutable](https://leetcode.com/problems/range-sum-query-mutable/)
- [LeetCode 315 · Count of Smaller Numbers After Self](https://leetcode.com/problems/count-of-smaller-numbers-after-self/)
- [LeetCode 370 · Range Addition](https://leetcode.com/problems/range-addition/)
