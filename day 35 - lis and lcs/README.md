# LIS and LCS: the longest subsequence you keep, and the table you throw away

More details in:
https://medium.com/100-days-of-python

Interactive walkthrough: [demo.html](demo.html) - a single self-contained page
(no build step, works offline, 中文 / English toggle) that deals the input into patience
piles and shows `tails` being overwritten in place, then replays an array whose finished
`tails` is not a subsequence at all so the parent pointers become visibly necessary,
flips `bisect_left` to `bisect_right` to turn "strictly increasing" into "non-decreasing",
sorts envelopes with the height tie-break ascending and descending so the answer moves
between 5 and 3, fills the LCS table and walks the backtrack path back out of it, and
finally maps one permutation through the other so an `O(n·m)` table collapses into a
single `O(n log n)` LIS.

![img](https://github.com/KingRei/100DaysPython/blob/master/day%2035%20-%20lis%20and%20lcs/imgs/day35_1.png?raw=true)

Day 34 ended on a table you fill and then walk back up to recover *which* items were taken.
Today is the same shape with a different lesson: the longest increasing subsequence has a
fast algorithm whose working array **is not the answer**, and the longest common subsequence
has a table whose only purpose is to be thrown away. Both are DP, both are five lines, and
both have a trap that Python will not report.

A subsequence is not a substring. `[2, 3, 7, 18]` is a subsequence of
`[10, 9, 2, 5, 3, 7, 101, 18]` even though those four numbers are scattered: order is kept,
adjacency is not required. That single relaxation is what makes diff, ROUGE-L, `git blame`
and sequence alignment possible.

## LIS the obvious way

`dp[i]` is the length of the longest increasing subsequence **ending at** `i`. Every earlier
smaller element is a candidate predecessor, so the recurrence is a scan:

```python
def lis_dp(a):
    n = len(a)
    dp = [1] * n
    parent = [-1] * n
    for i in range(n):
        for j in range(i):
            if a[j] < a[i] and dp[j] + 1 > dp[i]:
                dp[i] = dp[j] + 1
                parent[i] = j
    best = max(range(n), key=lambda i: dp[i])
    out = []
    while best != -1:
        out.append(a[best])
        best = parent[best]
    out.reverse()
    return max(dp), out
```

On `[10, 9, 2, 5, 3, 7, 101, 18]` that is `4` and `[2, 5, 7, 101]`. Note the "ending at `i`"
framing: `dp[i]` is not the answer for the prefix, it is the answer for the prefix *that is
forced to use* `a[i]`. Without that constraint the subproblems do not compose, because you
cannot tell whether the next element may be appended.

`O(n²)`, and the `parent` array makes the reconstruction free.

## Patience sorting, and the array that lies

Deal the numbers as a game of patience: each card goes on the leftmost pile whose top card
is `>= ` it, or starts a new pile on the right. The number of piles is the length of the LIS.
Keep only the top cards and they are sorted, so the pile can be found by binary search:

```python
from bisect import bisect_left

def lis_patience(a):
    tails = []
    for x in a:
        k = bisect_left(tails, x)
        if k == len(tails):
            tails.append(x)
        else:
            tails[k] = x
    return len(tails), tails
```

`tails[k]` is the smallest value that can end an increasing run of length `k + 1` seen so far.
Overwriting is never a loss: a smaller ending is at least as extensible as a larger one, and
the run length it represents is unchanged. `O(n log n)`.

![img](https://github.com/KingRei/100DaysPython/blob/master/day%2035%20-%20lis%20and%20lcs/imgs/day35_2.png?raw=true)

Now the trap. On `[10, 9, 2, 5, 3, 7, 101, 18]` the finished `tails` is `[2, 3, 7, 18]`, which
happens to be a real LIS, and that coincidence teaches the wrong lesson. Run it on
`[2, 6, 8, 3, 4, 5, 1]`:

```
tails = [1, 3, 4, 5]        real LIS = [2, 3, 4, 5]
```

`1` is the **last** element of the input, so `[1, 3, 4, 5]` is not a subsequence of the array
at all - it is a scoreboard, not a route. The length is always right; the contents usually are
not. To get the actual subsequence, record where each element landed and who was in front of
it:

```python
def lis_patience_reconstruct(a):
    tails, idx, parent = [], [], [-1] * len(a)
    for i, x in enumerate(a):
        k = bisect_left(tails, x)
        if k:
            parent[i] = idx[k - 1]
        if k == len(tails):
            tails.append(x); idx.append(i)
        else:
            tails[k] = x; idx[k] = i
    out, i = [], idx[-1]
    while i != -1:
        out.append(i); i = parent[i]
    out.reverse()
    return len(tails), out, [a[i] for i in out]
```

`parent[i]` is frozen at insertion time, so it points at whoever was ending a length-`k` run
*at that moment* - which is exactly the predecessor that was valid then, even if `tails[k-1]`
is later overwritten by something smaller.

## One character: strict or not

```python
find = bisect_left if strict else bisect_right
```

`bisect_left` places `x` on the pile whose top equals `x`, so equal values cannot chain;
`bisect_right` puts it on the next pile, so they can. On `[1, 3, 3, 3, 5]` that is `3` against
`5`. "Increasing" is ambiguous in English and precise in code - ask which one the problem
means before writing anything.

## LeetCode 300 - Longest Increasing Subsequence

**The task.** Given an integer array, return the length of its longest strictly increasing
subsequence.

**Input / output.** Input is `nums`; output is one integer.

**Example.** `nums = [10,9,2,5,3,7,101,18]` → `4`, from `[2,3,7,101]` (also `[2,3,7,18]`).
`nums = [0,1,0,3,2,3]` → `4`. `nums = [7,7,7,7,7]` → `1`, because strictly increasing rules
out repeats.

**Constraints.** `1 <= len(nums) <= 2500`, `-10^4 <= nums[i] <= 10^4`. At 2500 the `O(n²)`
DP passes comfortably, which is why the problem's follow-up - "can you do it in
`O(n log n)`?" - is where the interview actually happens.

[leetcode.com/problems/longest-increasing-subsequence](https://leetcode.com/problems/longest-increasing-subsequence/)

```python
def length_of_lis(nums):
    tails = []
    for x in nums:
        k = bisect_left(tails, x)
        if k == len(tails):
            tails.append(x)
        else:
            tails[k] = x
    return len(tails)
```

## The tie-break is the algorithm

### LeetCode 354 - Russian Doll Envelopes

**The task.** Each envelope is a `[width, height]` pair. One envelope fits inside another only
if **both** its width and its height are strictly smaller. Return the largest number of
envelopes that can be nested one inside the next.

**Input / output.** Input is `envelopes`; output is one integer.

**Example.** `envelopes = [[5,4],[6,4],[6,7],[2,3]]` → `3`, nesting `[2,3] → [5,4] → [6,7]`.
`envelopes = [[1,1],[1,1],[1,1]]` → `1`: equal widths are not *strictly* smaller, so nothing
nests.

**Constraints.** `1 <= len(envelopes) <= 10^5`, `1 <= w, h <= 10^5`. The `10^5` is the whole
hint - an `O(n²)` pairwise check is 10^10 operations and will not run, so the intended
solution must be the `O(n log n)` LIS.

[leetcode.com/problems/russian-doll-envelopes](https://leetcode.com/problems/russian-doll-envelopes/)

Sort by width and the second dimension becomes a plain LIS on heights. But equal widths must
not be allowed to chain, and that is handled entirely by the sort key:

```python
def max_envelopes(envelopes):
    envelopes.sort(key=lambda e: (e[0], -e[1]))      # width asc, height DESC
    return length_of_lis([h for _, h in envelopes])
```

Sorting heights **descending** within a width makes them a decreasing run, and a decreasing
run can contribute at most one element to an increasing subsequence. Flip that minus sign and
`[[1,1],[1,2],[1,3]]` returns `3` - three envelopes of the same width, all supposedly nested.
No exception, no warning, just a wrong number.

### LeetCode 673 - Number of Longest Increasing Subsequence

**The task.** Given an integer array, return *how many* distinct longest strictly increasing
subsequences it has. Two subsequences count as different if they use different positions.

**Input / output.** Input is `nums`; output is one integer count.

**Example.** `nums = [1,3,5,4,7]` → `2`: the LIS length is `4`, reached by `[1,3,5,7]` and
`[1,3,4,7]`. `nums = [2,2,2,2,2]` → `5`: the LIS length is `1`, and each of the five positions
is its own subsequence.

**Constraints.** `1 <= len(nums) <= 2000`, `-10^6 <= nums[i] <= 10^6`, and the answer fits in
a 32-bit integer. `2000` allows the `O(n²)` DP, which matters because the patience version has
nowhere to keep the counts.

[leetcode.com/problems/number-of-longest-increasing-subsequence](https://leetcode.com/problems/number-of-longest-increasing-subsequence/)

Carry a second array beside `dp`:

```python
def count_lis(a):
    n = len(a)
    dp, cnt = [1] * n, [1] * n
    for i in range(n):
        for j in range(i):
            if a[j] < a[i]:
                if dp[j] + 1 > dp[i]:
                    dp[i] = dp[j] + 1
                    cnt[i] = cnt[j]          # a longer route: replace
                elif dp[j] + 1 == dp[i]:
                    cnt[i] += cnt[j]         # an equally long route: add
    best = max(dp)
    return sum(c for d, c in zip(dp, cnt) if d == best)
```

`=` when the length improves, `+=` when it ties. Getting those two lines the wrong way round is
the single most common way to fail this problem, and both versions return a plausible integer.

## LCS: the table, and the path

![img](https://github.com/KingRei/100DaysPython/blob/master/day%2035%20-%20lis%20and%20lcs/imgs/day35_3.png?raw=true)

`dp[i][j]` is the LCS length of `x[:i]` and `y[:j]`. Matching characters extend the diagonal;
otherwise the best of dropping one character from either side wins:

```python
def lcs_table(x, y):
    dp = [[0] * (len(y) + 1) for _ in range(len(x) + 1)]
    for i in range(1, len(x) + 1):
        for j in range(1, len(y) + 1):
            if x[i - 1] == y[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    return dp
```

`AGGTAB` against `GXTXAYB` gives `4`, and walking back from the bottom-right corner -
diagonally on a match, otherwise towards the larger neighbour - spells `GTAB`.

Two rows are enough for the number, since row `i` only reads row `i - 1`. Exactly as on Day 34,
though, **the rolling version cannot reconstruct the path**: the cells the backtrack would have
walked through are gone. Number or path, pick one - or use Hirschberg's divide and conquer,
which recovers the path in linear space at the cost of running the DP twice per split.

### LeetCode 1035 - Uncrossed Lines

**The task.** Two rows of integers are written down, one above the other. Draw straight
connecting lines between equal numbers, one in each row, such that no two lines cross and no
number is used twice. Return the maximum number of lines.

**Input / output.** Input is `nums1` and `nums2`; output is one integer.

**Example.** `nums1 = [1,4,2]`, `nums2 = [1,2,4]` → `2`: connect the two `1`s and then either
the `4`s or the `2`s, but not both, because those two lines would cross.
`nums1 = [2,5,1,2,5]`, `nums2 = [10,5,2,1,5,2]` → `3`.

**Constraints.** `1 <= len(nums1), len(nums2) <= 500`, `1 <= nums[i] <= 2000`. The `500 × 500`
table is the point: it is exactly the size the `O(n·m)` DP wants.

[leetcode.com/problems/uncrossed-lines](https://leetcode.com/problems/uncrossed-lines/)

"No lines cross" means the matched positions increase in both rows, which is the definition of
a common subsequence. It is LCS with the words changed, and it is worth recognising that in an
interview rather than deriving a fresh recurrence.

## When LCS *is* LIS

![img](https://github.com/KingRei/100DaysPython/blob/master/day%2035%20-%20lis%20and%20lcs/imgs/day35_4.png?raw=true)

If both sequences are permutations of the same set - every element distinct, both sides drawn
from the same alphabet - the table is unnecessary. Replace each element of `q` by its index in
`p`; a common subsequence is then a set of positions increasing in `p` and appearing in order
in `q`, which is precisely an increasing subsequence of the mapped array:

```python
def lcs_of_permutations(p, q):
    pos = {v: i for i, v in enumerate(p)}
    mapped = [pos[v] for v in q if v in pos]
    length, seq, _ = lis_patience_reconstruct(mapped)
    return length, [p[i] for i in seq]
```

`p = [1,2,3,4,5,6]`, `q = [2,4,1,5,6,3]` maps to `[1,3,0,4,5,2]`, whose LIS `[1,3,4,5]` reads
back as `[2,4,5,6]`. On two random permutations of size 1200 the `O(n·m)` table takes 1.300 s
and this takes 0.0018 s for the same answer of 67 - about 700x. This is the core of the
Hunt-Szymanski algorithm, which is why `diff` on files of mostly-unique lines is fast.

## What it is actually for

LCS is the engine inside `diff`: the common subsequence is the set of unchanged lines, and
everything else is a deletion or an insertion. `git diff`, `patch`, code review and merge
conflict detection all rest on it.

```
     def f(x):
   -     y = x + 1
   +     # bump
   +     y = x + 2
         return y
```

ROUGE-L, the standard summarisation metric, is the F-measure of the LCS length between a
candidate and a reference, so it rewards words appearing in the right order without demanding
adjacency. `"the cat sat on the mat"` against `"the cat was sitting on the mat"` scores
`0.769`. Sequence alignment in bioinformatics (Needleman-Wunsch) is the same table with a
scoring matrix instead of a `+1`.

## Complexity

| Operation | Time | Space |
|---|---|---|
| `lis_dp` (with reconstruction) | `O(n²)` | `O(n)` |
| `lis_patience` | `O(n log n)` | `O(n)` - length only |
| `lis_patience_reconstruct` | `O(n log n)` | `O(n)` - real subsequence |
| `count_lis` (LC 673) | `O(n²)` | `O(n)` |
| `max_envelopes` (LC 354) | `O(n log n)` | `O(n)` |
| `lcs_table` + `lcs_backtrack` | `O(n·m)` | `O(n·m)` |
| `lcs_rolling` | `O(n·m)` | `O(min(n, m))` - no path |
| `lcs_of_permutations` | `O(n log n)` | `O(n)` - 1.300 s → 0.0018 s at n = 1200 |

## References
- [Longest increasing subsequence](https://en.wikipedia.org/wiki/Longest_increasing_subsequence)
- [Patience sorting](https://en.wikipedia.org/wiki/Patience_sorting)
- [Longest common subsequence](https://en.wikipedia.org/wiki/Longest_common_subsequence)
- [Hunt-Szymanski algorithm](https://en.wikipedia.org/wiki/Hunt%E2%80%93Szymanski_algorithm)
- [Hirschberg's algorithm](https://en.wikipedia.org/wiki/Hirschberg%27s_algorithm)
- [ROUGE (metric)](https://en.wikipedia.org/wiki/ROUGE_(metric))
- [LeetCode 300 - Longest Increasing Subsequence](https://leetcode.com/problems/longest-increasing-subsequence/)
- [LeetCode 354 - Russian Doll Envelopes](https://leetcode.com/problems/russian-doll-envelopes/)
- [LeetCode 673 - Number of Longest Increasing Subsequence](https://leetcode.com/problems/number-of-longest-increasing-subsequence/)
- [LeetCode 1035 - Uncrossed Lines](https://leetcode.com/problems/uncrossed-lines/)
