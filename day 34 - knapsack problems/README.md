# The knapsack family: 0/1, unbounded and bounded

More details in:
https://medium.com/100-days-of-python

Interactive walkthrough: [demo.html](demo.html) - a single self-contained page
(no build step, works offline, 中文 / English toggle) that fills the full two-dimensional
table cell by cell and then walks back up it to recover *which* items were taken, collapses
that table into one rolling row and runs the identical update line downwards and upwards so
the answer flips between 130 and 150, bundles thirteen copies of an item into four binary
packages and reconstructs every count from 0 to 13 out of them, lets the greedy fill the bag
with and without scissors so the 10-point loss becomes visible, and finishes on LeetCode 416
where `max` becomes `or` and the whole boolean row collapses into a single Python integer.

![img](https://github.com/KingRei/100DaysPython/blob/master/day%2034%20-%20knapsack%20problems/imgs/day34_1.png?raw=true)

The knapsack problem is the one every DP course reaches for, and for a reason: it is the
smallest problem where the recurrence is obvious and the *order* of evaluation decides which
question you actually answered. Day 33 ended on the observation that swapping two `for`
statements turns "count the combinations" into "count the permutations". Today the knife is
smaller still. Not the order of the loops - the **direction** of the inner one.

Four items, one bag with capacity 10:

| item | weight | value |
|---|---|---|
| rope | 3 | 50 |
| book | 4 | 40 |
| pan | 5 | 70 |
| tent | 6 | 80 |

Run the same eight lines with the inner loop descending and the answer is **130**. Run it
ascending and the answer is **150**. Both are correct; they answer different questions, and
Python raises nothing either way.

## The two-dimensional table

`dp[i][c]` is the best value obtainable from the first `i` items with capacity `c`. Each item
has exactly two fates - leave it, and the cell inherits the row above; take it, and the cell is
the row above at capacity `c - w` plus `v`:

```python
def knap01_table(weights, values, cap):
    n = len(weights)
    dp = [[0] * (cap + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        w, v = weights[i - 1], values[i - 1]
        for c in range(cap + 1):
            dp[i][c] = dp[i - 1][c]                       # leave it
            if w <= c and dp[i - 1][c - w] + v > dp[i][c]:
                dp[i][c] = dp[i - 1][c - w] + v           # take it
    return dp[n][cap]
```

Both readings live in row `i - 1`, so the table fills top to bottom and never reads unfinished
work. The finished table:

```
none  0 0 0  0  0  0  0  0   0   0   0
rope  0 0 0 50 50 50 50 50  50  50  50
book  0 0 0 50 50 50 50 90  90  90  90
pan   0 0 0 50 50 70 70 90 120 120 120
tent  0 0 0 50 50 70 80 90 120 130 130
```

A finished table gives you a number, not a solution. To recover the items you walk back up:
any cell that differs from the one directly above it marks an item that was taken.
`knap01_items` returns `(130, [0, 3])` - rope and tent, total weight 9. Note what that means:
the optimum deliberately leaves one unit of capacity empty. Rope + pan packs to exactly 8 and
is worth only 120, so any heuristic that optimises "space used" loses right here.

## One row, and the direction that decides everything

![img](https://github.com/KingRei/100DaysPython/blob/master/day%2034%20-%20knapsack%20problems/imgs/day34_2.png?raw=true)

Row `i` only ever reads row `i - 1`, so the table is unnecessary - one row will do, as long as
the row still *looks like* the previous one at the moment it is read.

```python
def knap01_rolling(weights, values, cap):
    dp = [0] * (cap + 1)
    for w, v in zip(weights, values):
        for c in range(cap, w - 1, -1):          # descending: read the old row
            if dp[c - w] + v > dp[c]:
                dp[c] = dp[c - w] + v
    return dp[cap]
```

Going downwards, `dp[c - w]` sits at a *lower* index than the cell being written, and lower
indices have not been touched yet this round. So the value read is still last round's, each
item is offered exactly once, and the rolling rows reproduce the two-dimensional table row for
row. 130.

## The same code, upwards

![img](https://github.com/KingRei/100DaysPython/blob/master/day%2034%20-%20knapsack%20problems/imgs/day34_3.png?raw=true)

```python
def knap_unbounded(weights, values, cap):
    dp = [0] * (cap + 1)
    for w, v in zip(weights, values):
        for c in range(w, cap + 1):              # ascending: read the new row
            if dp[c - w] + v > dp[c]:
                dp[c] = dp[c - w] + v
    return dp[cap]
```

One `reversed()` apart. Now `dp[c - w]` was already rewritten this round and may already
contain a copy of this very item, so it gets taken again, and again:

```
+rope  0 0 0 50 50 50 100 100 100 150 150
+book  0 0 0 50 50 50 100 100 100 150 150
+pan   0 0 0 50 50 70 100 100 120 150 150
+tent  0 0 0 50 50 70 100 100 120 150 150
```

150, and `unbounded_counts` says why: three ropes. That is the *complete* knapsack - unlimited
copies of every item - and it is a genuinely useful algorithm. The danger is that it is also
exactly what you get when you write the 0/1 loop and forget which way it runs. The direction of
the inner loop **is** the answer to "may this item be taken more than once".

## Bounded: k copies, and binary splitting

![img](https://github.com/KingRei/100DaysPython/blob/master/day%2034%20-%20knapsack%20problems/imgs/day34_4.png?raw=true)

Real supply is usually neither one nor infinite but `k`. With counts `[2, 1, 3, 1]` the same
four items are worth 140. Expanding `k` copies into `k` separate 0/1 items is correct, but its
cost is proportional to the *value* of `k`. Binary splitting bundles the copies into
1, 2, 4, 8 … plus a remainder:

```python
def binary_split(k):
    parts, p = [], 1
    while p <= k:
        parts.append(p)
        k -= p
        p *= 2
    if k:
        parts.append(k)
    return parts
```

`binary_split(13)` is `[1, 2, 4, 6]`, and every count from 0 to 13 is exactly one subset of
those four bundles - `binary_split_reaches` proves it by exhaustion. So `log k` 0/1 items
replace `k` of them, exactly, not approximately, and what runs afterwards is the identical
descending loop. `binary_split(1000)` is ten parts. On weights `[3, 4, 5, 6, 7]` with counts
`[300, 500, 800, 1200, 2000]` and capacity 4000 both routes return 58,400, but the naive
expansion builds 4,800 items and takes 3,441 ms while the split builds 50 and takes 22 ms.

## Why greedy is not allowed here

![img](https://github.com/KingRei/100DaysPython/blob/master/day%2034%20-%20knapsack%20problems/imgs/day34_5.png?raw=true)

Sort by value per unit of weight - rope 16.67, pan 14.00, tent 13.33, book 10.00 - and take
greedily. If items may be cut, this is provably optimal by an exchange argument: the bag ends
exactly full at 146.67, taking all of the rope, all of the pan and a third of the tent. Put the
scissors away and the identical rule on the identical items returns 120, because after rope and
pan there are 2 units left and the tent needs 6. The DP returns 130.

The gap is only 10, and the size is not the point. The point is that the greedy does not
report that it lost. Plausibility is not evidence of correctness: the fractional version has a
proof and the 0/1 version does not, and that is the whole difference between them.

## Same row, different operator

`or` instead of `max` turns the table into subset sum: forget the values and ask only which
totals are reachable. LeetCode 416 asks whether an array splits into two piles of equal sum,
which is the question "is there a subset summing to `sum/2`" - a 0/1 knapsack with capacity
`sum/2`, still scanning downwards so no number is used twice. `[1, 5, 11, 5]` splits;
`[2, 2, 3, 5]` has an even total and still does not.

And a boolean row is a bit vector, so Python's arbitrary-precision integers can hold the whole
thing:

```python
def subset_sum_bits(nums, target):
    bits = 1                       # bit 0 set: the empty subset sums to 0
    for x in nums:
        bits |= bits << x          # one shift offers x to every reachable sum at once
    return (bits >> target) & 1 == 1
```

The shift reads a snapshot of the old value, so the "must run downwards" worry disappears as
well. On 400 numbers with target 9545 the list version takes about 190 ms and this takes about
0.2 ms - roughly a thousandfold, from moving the inner loop into the C-level big-integer
routines. The same trick solves 494, 1049, 474 and 279, which are all the same row with a
different operator.

## Pseudo-polynomial: what "efficient" is hiding

![img](https://github.com/KingRei/100DaysPython/blob/master/day%2034%20-%20knapsack%20problems/imgs/day34_6.png?raw=true)

`O(n · cap)` looks polynomial and is not. Multiply every weight and the capacity by 10 and the
answer is unchanged at 130, but the table grows from 44 cells to 404, then 4,004, then 40,004 -
while the *input* grows from 15 bits to 30, 47 and 64. The runtime is polynomial in the
magnitude of the numbers, not in their length, which is why 0/1 knapsack is NP-complete and
this table is not a counterexample. It also tells you exactly when to reach for it: small
capacities, small integer weights. Floating-point or astronomically large capacities need
branch and bound or an approximation scheme instead.

## The problems, stated in full

Restated in my own words - what is being asked, what goes in and comes out, one worked
example, and the constraints that actually change which algorithm is allowed.

### LeetCode 416 - Partition Equal Subset Sum

**The task.** Given an array of positive integers, decide whether it can be split into two
subsets whose sums are equal. Every element must go into exactly one of the two subsets.

**Input / output.** Input is `nums`; output is a boolean.

**Example.** `nums = [1,5,11,5]` → `True`, as `[1,5,5]` and `[11]` both sum to `11`.
`nums = [1,2,3,5]` → `False`, the total `11` is odd so it cannot even be halved.
`nums = [2,2,3,5]` → `False`: the total `12` is even, but no subset reaches `6`.

**Constraints.** `1 <= len(nums) <= 200`, `1 <= nums[i] <= 100`, so the total is at most
`20000`. Halve the total (bail out if it is odd) and the question becomes "is there a subset
summing to exactly half?" - a 0/1 knapsack where the weight *is* the value and you only care
whether a cell is reachable, which is why a bitset can answer it 64 numbers at a time.

[leetcode.com/problems/partition-equal-subset-sum](https://leetcode.com/problems/partition-equal-subset-sum/)

### LeetCode 494 - Target Sum

**The task.** Put a `+` or a `-` in front of every number in the array and concatenate them
into an expression. Count how many of the `2^n` sign assignments evaluate to `target`.

**Input / output.** Input is `nums` and `target`; output is an integer count.

**Example.** `nums = [1,1,1,1,1]`, `target = 3` → `5`: flip exactly one of the five ones to
`-` and the rest to `+`, and there are five ways to choose which.

**Constraints.** `1 <= len(nums) <= 20`, `0 <= nums[i] <= 1000`, `0 <= sum(nums) <= 1000`,
`-1000 <= target <= 1000`. Let `P` be the numbers given a `+`; then
`P - (total - P) = target`, so `P = (total + target) / 2`. If that is negative or odd there
are zero ways; otherwise the problem is exactly "count the subsets summing to `P`" - the
same table as LeetCode 416, counting instead of testing reachability.

[leetcode.com/problems/target-sum](https://leetcode.com/problems/target-sum/)

### LeetCode 1049 - Last Stone Weight II

**The task.** You repeatedly pick any two stones `x <= y`, smash them together, and they
become a single stone of weight `y - x` (both vanish if they are equal). Return the smallest
possible weight of the stone left at the end, or `0` if nothing is left.

**Input / output.** Input is `stones`; output is an integer.

**Example.** `stones = [2,7,4,1,8,1]` → `1`. `stones = [31,26,33,21,40]` → `5`.

**Constraints.** `1 <= len(stones) <= 30`, `1 <= stones[i] <= 100`, so the total is at most
`3000`. The reformulation is the whole problem: every smashing schedule amounts to giving
each stone a `+` or a `-` sign, so the surviving weight is `|sum(A) - sum(B)|` over a split
into two groups. Minimising that means finding the subset sum closest to `total / 2` from
below - a 0/1 knapsack with capacity `total // 2`. Note this is *not* the same as the greedy
LeetCode 1046, where you must always smash the two heaviest.

[leetcode.com/problems/last-stone-weight-ii](https://leetcode.com/problems/last-stone-weight-ii/)

### LeetCode 474 - Ones and Zeroes

**The task.** Given an array of binary strings and two budgets `m` and `n`, return the size
of the largest subset of the strings that uses at most `m` zeros and at most `n` ones in
total.

**Input / output.** Input is `strs`, `m`, `n`; output is an integer.

**Example.** `strs = ["10","0001","111001","1","0"]`, `m = 5`, `n = 3` → `4`, the subset
`{"10","0001","1","0"}`, which uses five `0`s and three `1`s exactly. With `m = 1`, `n = 1`
the answer is `2`, namely `{"0","1"}`.

**Constraints.** `1 <= len(strs) <= 600`, each string at most `100` characters of `0`/`1`,
`1 <= m, n <= 100`. It is a 0/1 knapsack with **two** capacities instead of one: the table
is `dp[zeros][ones]`, and both loops run downwards for exactly the same reason the one-
dimensional 0/1 loop does - so that each string is taken at most once.

[leetcode.com/problems/ones-and-zeroes](https://leetcode.com/problems/ones-and-zeroes/)

### LeetCode 279 - Perfect Squares

**The task.** Given an integer `n`, return the least number of perfect squares
(`1, 4, 9, 16, ...`) that sum to exactly `n`. Squares may be repeated.

**Input / output.** One integer in, one integer out.

**Example.** `n = 12` → `3` (`4 + 4 + 4`). `n = 13` → `2` (`4 + 9`).

**Constraints.** `1 <= n <= 10^4`. It is an unbounded knapsack whose "coins" are the squares
below `n`, so `O(n * sqrt(n))` falls straight out. Lagrange's four-square theorem says the
answer is never more than `4`, and Legendre's three-square theorem pins down exactly when
it is `4`, which gives an `O(sqrt(n))` mathematical answer - a nice reminder that DP is a
general hammer, not always the sharpest tool.

[leetcode.com/problems/perfect-squares](https://leetcode.com/problems/perfect-squares/)

## Complexity

| Operation | Time | Space |
|---|---|---|
| `knap01_table` | `O(n · cap)` | `O(n · cap)` |
| `knap01_items` (with reconstruction) | `O(n · cap)` | `O(n · cap)` |
| `knap01_rolling` | `O(n · cap)` | `O(cap)` |
| `knap_unbounded` | `O(n · cap)` | `O(cap)` |
| `knap_bounded_naive` | `O(cap · Σk)` - 4,800 items, 3,441 ms | `O(cap)` |
| `knap_bounded_binary` | `O(cap · Σ log k)` - 50 items, 22 ms | `O(cap)` |
| `fractional_greedy` | `O(n log n)` - optimal, with a proof | `O(n)` |
| `greedy_01` | `O(n log n)` - **not** optimal, 120 against 130 | `O(n)` |
| `subset_sum` | `O(n · target)` | `O(target)` |
| `subset_sum_bits` | `O(n · target / 64)` words - about 1000x | `O(target / 64)` |

## References
- [Knapsack problem](https://en.wikipedia.org/wiki/Knapsack_problem)
- [Continuous (fractional) knapsack, and the exchange argument](https://en.wikipedia.org/wiki/Continuous_knapsack_problem)
- [Subset sum problem](https://en.wikipedia.org/wiki/Subset_sum_problem)
- [Pseudo-polynomial time](https://en.wikipedia.org/wiki/Pseudo-polynomial_time)
- [Pisinger, "Where are the hard knapsack problems?" (2005)](http://hjemmesider.diku.dk/~pisinger/94-27.pdf)
- [LeetCode 416 - Partition Equal Subset Sum](https://leetcode.com/problems/partition-equal-subset-sum/)
- [LeetCode 494 - Target Sum](https://leetcode.com/problems/target-sum/)
- [LeetCode 1049 - Last Stone Weight II](https://leetcode.com/problems/last-stone-weight-ii/)
- [LeetCode 474 - Ones and Zeroes](https://leetcode.com/problems/ones-and-zeroes/)
- [LeetCode 279 - Perfect Squares](https://leetcode.com/problems/perfect-squares/)
