# Dynamic programming from scratch: climbing stairs and coin change

More details in:
https://medium.com/100-days-of-python

Interactive walkthrough: [demo.html](demo.html) - a single self-contained page
(no build step, works offline, 中文 / English toggle) that expands the recursion tree of
`f(5)` and paints every subproblem it computes twice, folds that tree into a row of cells
filled left to right, slides a two-variable window across the same row, fills the coin-change
table and then walks back along the choices to rebuild `6 + 5`, watches greedy take the 9 and
end up one coin worse, runs the *same* update line under both loop orders so the answer flips
between 4 and 9, and finishes on LeetCode 139 where `or` becomes `+` and the identical table
starts counting instead of deciding.

![img](https://github.com/KingRei/100DaysPython/blob/master/day%2033%20-%20dynamic%20programming%20basics/imgs/day33_1.png?raw=true)

Dynamic programming is not an algorithm. It is a rewrite you apply to a recursion that keeps
asking the same question, and the rewrite has exactly two ingredients: a recurrence, and an
order to evaluate it in. The recurrence is usually the part people can produce in thirty
seconds. The order is the part that quietly breaks things.

Climbing stairs is the smallest honest example. You can go up one or two steps at a time, so
the number of ways to reach step `n` is `f(n) = f(n-1) + f(n-2)`. Write that literally as a
recursive function and it is correct and useless: `stairs_naive(30)` makes 4,356,617 calls to
answer a question that has only 31 distinct subproblems, a factor of 140,536 wasted. Inside
that one call `f(1)` is evaluated 832,040 times and `f(0)` 1,346,269 times, because nothing in
the program remembers an answer it has already found. The reason is visible in the tree: the
number of leaves *is* the answer, so an exponential answer costs exponential time.

Two rewrites fix it and they are the same fix seen from two ends. Memoisation keeps the
recursion and adds a dictionary, so each argument is expanded once. Tabulation throws the
recursion away, lays the subproblems out in an array, and fills it in an order that guarantees
every cell it reads is already written. On this machine the three versions answer `stairs(28)`
in 252.74 ms, 0.022 ms and 0.0072 ms.

## The table, and the order that fills it

![img](https://github.com/KingRei/100DaysPython/blob/master/day%2033%20-%20dynamic%20programming%20basics/imgs/day33_2.png?raw=true)

```python
def stairs_table(n):
    dp = [0] * (n + 1)
    dp[0] = dp[1] = 1
    for i in range(2, n + 1):
        dp[i] = dp[i - 1] + dp[i - 2]
    return dp[n]
```

`dp[7] = dp[6] + dp[5] = 13 + 8 = 21`, and the row for `n = 10` reads
`[1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89]` - Fibonacci, shifted by one. Left-to-right is a valid
order for one reason only: `dp[i]` depends on smaller indices, and smaller indices are already
done. That sentence is the entire rule of DP. Since each cell reads only the two before it, the
array itself is unnecessary; two variables give the same answer in `O(1)` space.

## Coin change: swap `+` for `min`

![img](https://github.com/KingRei/100DaysPython/blob/master/day%2033%20-%20dynamic%20programming%20basics/imgs/day33_3.png?raw=true)

Keep the skeleton, change what a cell holds. `dp[a]` is now the fewest coins that sum to `a`,
and the combine step is `min` instead of `+`:

```python
def coin_change_min(coins, amount):
    dp = [0] + [INF] * amount
    for a in range(1, amount + 1):
        for c in coins:
            if c <= a and dp[a - c] + 1 < dp[a]:
                dp[a] = dp[a - c] + 1
    return -1 if dp[amount] == INF else dp[amount]
```

Filling `dp[a]` asks one question - what was the last coin? - and every candidate leaves a
smaller amount that is already solved. With `coins = [1, 5, 6, 9]` and `amount = 11` the row is
`0 1 2 3 4 1 1 2 3 1 2 2` and the answer is two coins, `6 + 5`. Greedy, taking the biggest coin
that fits, produces `9 + 1 + 1` - three coins, 50% worse, and no error message. Greedy is safe
only on *canonical* coin systems; real currencies are designed to be canonical, arbitrary coin
sets are not. Amounts that cannot be made stay at infinity, which is how `coin_change_min([5, 7], 3)`
correctly returns `-1`.

## The silent failure: combinations vs permutations

![img](https://github.com/KingRei/100DaysPython/blob/master/day%2033%20-%20dynamic%20programming%20basics/imgs/day33_4.png?raw=true)

This is the bug worth memorising, because it never announces itself.

```python
# LC 518 - combinations
dp = [1] + [0] * amount
for c in coins:                     # coin loop OUTSIDE
    for a in range(c, amount + 1):
        dp[a] += dp[a - c]

# LC 377 - permutations
dp = [1] + [0] * amount
for a in range(1, amount + 1):      # amount loop OUTSIDE
    for c in coins:
        if c <= a:
            dp[a] += dp[a - c]
```

Same array, same update line, same number of updates. Only the two `for` statements swap
places. With `coins = [1, 2, 5]` and `amount = 5` the first returns 4 and the second returns 9.
With the coin loop outside, coin `5` is offered once and for all, so `1+2+2` can be built in
exactly one order and you are counting multisets. With the amount loop outside, every coin gets
a turn at being the *last* one added, so `1+2+2`, `2+1+2` and `2+2+1` are three separate fills
and you are counting sequences. The gap grows fast: at amount 30 it is 58 against 5,508,222.

Which also explains a pretty coincidence. Climbing stairs is coin change with `coins = {1, 2}`,
counted as permutations - steps are ordered, one-then-two is a different climb from
two-then-one. The permutation column matches `stairs(n)` for every `n`; the combination column
does not.

## One table, three questions

![img](https://github.com/KingRei/100DaysPython/blob/master/day%2033%20-%20dynamic%20programming%20basics/imgs/day33_5.png?raw=true)

The word break from Day 08 turns out to be this same table with the cells sitting at the cut
points of the string: `dp[i]` asks whether `s[:i]` can be segmented, and the transition looks
for a `j` where `dp[j]` holds and `s[j:i]` is a dictionary word. Change the operator and the
question changes with it - `+` counts the segmentations, `min` returns the fewest words, `or`
decides feasibility. `word_break('applepenapple', ['apple', 'pen'])` is `True` with exactly one
segmentation; `'catsandog'` matches `cat`, `cats`, `sand` and `and` along the way and still
fails, which is why a greedy left-to-right scanner gets it wrong.

## Memoisation or tabulation

![img](https://github.com/KingRei/100DaysPython/blob/master/day%2033%20-%20dynamic%20programming%20basics/imgs/day33_6.png?raw=true)

They compute the same thing, so the choice is about which states you actually touch. With
`coins = [100, 250]` and `amount = 10000` only multiples of 50 are ever reachable, so top-down
visits 203 states while a table allocates and fills 10,001 cells - 49.3x more work for the same
answer of 40 coins. With `coins = [1, 2, 5]` and `amount = 1000` every amount is reachable, the
counts coincide (1005 states against 1001 cells), and tabulation wins on the constants: no call
overhead, and no 1000-deep call stack. The recursive version there needs
`sys.setrecursionlimit`, because the chain 1000, 999, 998, … is exactly Python's default limit.

## Complexity

| Operation | Time | Space |
|---|---|---|
| `stairs_naive(n)` | `O(φ^n)` - 4,356,617 calls at n = 30 | `O(n)` stack |
| `stairs_memo(n)` | `O(n)` | `O(n)` memo + `O(n)` stack |
| `stairs_table(n)` | `O(n)` | `O(n)` |
| `stairs_rolling(n)` | `O(n)` | `O(1)` |
| `coin_change_min` | `O(amount · len(coins))` | `O(amount)` |
| `coin_ways_combinations` / `_permutations` | `O(amount · len(coins))` | `O(amount)` |
| `word_break(s, words)` | `O(len(s)² · cost of slicing)` | `O(len(s))` |

## References
- [Dynamic programming](https://en.wikipedia.org/wiki/Dynamic_programming)
- [Bellman, "The theory of dynamic programming" (1954)](https://www.ams.org/journals/bull/1954-60-06/S0002-9904-1954-09848-8/)
- [Memoization](https://en.wikipedia.org/wiki/Memoization)
- [Change-making problem, and when greedy is optimal](https://en.wikipedia.org/wiki/Change-making_problem)
- [Pearson, "A polynomial-time algorithm for the change-making problem" (2005)](https://graal.ens-lyon.fr/~abenoit/algo09/coins2.pdf)
- [LeetCode 70 - Climbing Stairs](https://leetcode.com/problems/climbing-stairs/)
- [LeetCode 322 - Coin Change](https://leetcode.com/problems/coin-change/)
- [LeetCode 518 - Coin Change II](https://leetcode.com/problems/coin-change-ii/)
- [LeetCode 377 - Combination Sum IV](https://leetcode.com/problems/combination-sum-iv/)
- [LeetCode 139 - Word Break](https://leetcode.com/problems/word-break/)
