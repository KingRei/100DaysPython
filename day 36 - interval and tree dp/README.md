# Interval DP and tree DP: choosing the split point instead of the next item

More details in:
https://medium.com/100-days-of-python

Interactive walkthrough: [demo.html](demo.html) - a single self-contained page
(no build step, works offline, 中文 / English toggle) that fills the matrix-chain
triangle by interval length and then re-fills it with the instinctive i-then-j loops so
the answer silently drops from 26,000 to 20,000, pops the balloons of LeetCode 312 by
"which one is popped last" and then by "which one is popped first" so the same code
returns 167 and 98, runs LeetCode 337 with a `(take, skip)` pair per node and again with
a single number per node so a parent and its child end up robbed together, and merges a
tree knapsack child by child while counting the inner-loop work until it lands exactly
on C(6, 2).

![img](https://github.com/KingRei/100DaysPython/blob/master/day%2036%20-%20interval%20and%20tree%20dp/imgs/day36_1.png?raw=true)

Every DP so far has walked forwards: Day 33 asked "what do I do with element `i`", Day 34
asked "do I take this item", Day 35 asked "does the subsequence end here". Today the
subproblem stops being a prefix. An interval DP asks "where does this range split", and a
tree DP asks "what does this subtree report to its parent". The recurrences are short; the
two things that are easy to get wrong are the order the table is filled in and the number
of values each subproblem has to report.

## Matrix chain multiplication

Matrix multiplication is associative, so `(AB)C` and `A(BC)` compute the same matrix at
wildly different prices. For `40x20`, `20x30`, `30x10`, `10x30`, the bracketing `((A(BC))D)`
costs 26,000 scalar multiplications and the worst one costs 69,000. The number of
bracketings is a Catalan number, so enumeration is hopeless; `dp[i][j]` instead asks a
single question - where does the **topmost** multiplication split the chain?

```python
for length in range(2, n + 1):            # the outer loop is length
    for i in range(n - length + 1):
        j = i + length - 1
        for k in range(i, j):
            cost = dp[i][k] + dp[k + 1][j] + dims[i] * dims[k + 1] * dims[j + 1]
```

## The loop order is the algorithm

`dp[i][j]` reads `dp[k+1][j]`, a row *below* it, which the instinctive `for i: for j:` pair
of loops has not filled yet. Those cells are still `0`, so several multiplications are
priced at zero and the function returns 20,000 - smaller than the true minimum, matching no
bracketing at all, with no exception and no warning. Filling by interval length guarantees
both sub-intervals are already done, because both are strictly shorter.

## LeetCode 312 - Burst Balloons

**The task.** A row of balloons each carries a number. Pop them one at a time; popping a
balloon earns the product of its own number and the numbers of its two current neighbours,
and after it pops, its neighbours become adjacent. A missing neighbour off the end of the
row counts as 1. Maximise the total earned by popping every balloon.

**Input / output.** A list of integers; return the maximum number of coins.

**Example.** `[3, 1, 5, 8]` gives 167, popping 1, then 5, then 3, then 8:
`3*1*5 = 15`, `3*5*8 = 120`, `1*3*8 = 24`, `1*8*1 = 8`.

**Constraints.** Up to 300 balloons and values up to 100. 300 rules out any factorial
enumeration - `4! = 24` orders is fine to brute-force in the example, `300!` is not - and it
comfortably allows the `O(n^3)` interval DP.

[leetcode.com/problems/burst-balloons](https://leetcode.com/problems/burst-balloons/)

The instinctive recurrence picks which balloon pops **first**, and it is wrong: once that
balloon is gone the left half's right-hand neighbour is a balloon belonging to the right
half, so the two sides are not independent. It returns 98 instead of 167 and raises nothing.
Choosing the balloon popped **last** inside the open interval `(i, j)` fixes it: by the time
`k` pops, everything strictly between `i` and `j` is already gone, so its neighbours are
exactly the walls `i` and `j`, which never move.

```python
for k in range(i + 1, j):
    cand = dp[i][k] + dp[k][j] + a[i] * a[k] * a[j]
```

## LeetCode 1547 - Minimum Cost to Cut a Stick

**The task.** A wooden stick of length `n` must be cut at a given set of positions. Cutting
a piece costs the length of that piece, and the cuts may be performed in any order.
Minimise the total cost.

**Input / output.** The stick length `n` and a list of cut positions; return the minimum
total cost.

**Example.** `n = 7`, cuts `[1, 3, 4, 5]`. Performing them in the given order costs 20; the
best order costs 16.

**Constraints.** `n` up to 10^6, but at most 100 cuts. The DP is over the *cut positions*,
not the stick's length - which is why an `O(m^3)` table with `m <= 102` is fine while
anything indexed by `n` would not be.

[leetcode.com/problems/minimum-cost-to-cut-a-stick](https://leetcode.com/problems/minimum-cost-to-cut-a-stick/)

This is the mirror image of the balloons. Here the *first* cut inside a piece is the
separable one: it costs the whole length of that piece and splits it into two pieces that
never interact again. Add the two ends as fake cuts, sort, and it is matrix chain with a
different cost formula.

## LeetCode 516 - Longest Palindromic Subsequence

**The task.** Given a string, find the length of the longest subsequence of it that reads
the same forwards and backwards. Characters may be skipped; the ones kept must stay in
order.

**Input / output.** A string; return the length as an integer.

**Example.** `'character'` gives 5, for the subsequence `'carac'`.

**Constraints.** Length up to 1000, lower-case letters only. 1000 allows an `O(n^2)` table
of a million cells but rules out anything exponential over subsets.

[leetcode.com/problems/longest-palindromic-subsequence](https://leetcode.com/problems/longest-palindromic-subsequence/)

As an interval DP, `dp[i][j]` is the answer for `s[i..j]`: equal ends contribute 2 and
recurse inwards, otherwise drop one end or the other. But there is a one-liner too - the LPS
of `s` is the LCS of `s` and `s` reversed, which is yesterday's table with no new code at
all.

## LeetCode 337 - House Robber III

**The task.** Houses are arranged in a binary tree and each holds an amount of money. No two
directly connected houses may both be robbed. Maximise the money taken.

**Input / output.** The root of a binary tree of integers; return the maximum total.

**Example.** The tree with root 3, left child 2 whose right child is 3, and right child 3
whose right child is 1, gives 7 - the two 3s at the bottom plus the 1.

**Constraints.** Up to 10^4 nodes, values from 0 to 10^4. Ten thousand nodes is well past
Python's default recursion limit of 1000 if the tree degenerates into a path, which is the
practical trap the constraint hides.

[leetcode.com/problems/house-robber-iii](https://leetcode.com/problems/house-robber-iii/)

The mistake here is not the recurrence, it is the state. If a node reports one number -
"the best this subtree can give" - the parent cannot tell whether that best used the child
itself, which is exactly what it needs to know. With positive values `max` always prefers
taking the node, so every node gets taken, parents and children together, and the answer is
12: the sum of everything, wearing a DP costume. Reporting a pair fixes it.

```python
robbed  = node.val + ls + rs           # take me: children must be skipped
skipped = max(lr, ls) + max(rr, rs)    # skip me: children are free
```

The same "what a node returns is not what a node scores" split shows up in LeetCode 124,
maximum path sum, where a node *returns* the best single downward arm but *scores* the best
path bending through it, and in tree diameter, which is the unweighted version of the same
pair.

## Tree knapsack, and why the double loop is not cubic

Choosing courses where a course may only be taken together with its prerequisite turns the
items into a tree. `dp[v][t]` is the best value from `v`'s subtree taking exactly `t` items,
and children are merged one at a time like a small knapsack. The merge looks like an `O(n^3)`
disaster, but the loop bounds `min(size[v], budget)` and `min(size[c], budget - t)` make the
work at a node exactly `size(child) * size(merged so far)` - the number of node pairs whose
lowest common ancestor is that node. Summed over the tree every pair is charged once, so the
total is `C(n, 2)`: 15 for six nodes, which the code counts and asserts.

## The depth trap

A tree DP is a post-order traversal, and a recursive post-order dies on a 60,000-node path
graph long before it finishes - `RecursionError`, not a wrong answer. Pushing nodes onto a
stack produces parents before children; reversing that order gives children before parents,
which is exactly what the DP needs, with no call stack involved.

```python
order, stack = [], [root]
while stack:
    v = stack.pop()
    order.append(v)
    stack.extend(children[v])
for v in reversed(order):
    total[v] = value[v] + sum(total[c] for c in children[v])
```

## Complexity
| Problem | Time | Space |
| --- | --- | --- |
| Matrix chain | O(n^3) | O(n^2) |
| Burst balloons (LC 312) | O(n^3) | O(n^2) |
| Cut a stick (LC 1547) | O(m^3) | O(m^2) |
| Longest palindromic subsequence (LC 516) | O(n^2) | O(n^2) |
| House robber III (LC 337) | O(n) | O(height) |
| Tree knapsack | O(n * budget) | O(n * budget) |

## References
- [Matrix chain multiplication](https://en.wikipedia.org/wiki/Matrix_chain_multiplication)
- [Dynamic programming](https://en.wikipedia.org/wiki/Dynamic_programming)
- [Tree (graph theory)](https://en.wikipedia.org/wiki/Tree_(graph_theory))
