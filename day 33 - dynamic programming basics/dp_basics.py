"""Day 33 - Dynamic programming from the ground up: climbing stairs and coin change.

Run me:  python3 dp_basics.py

Dynamic programming is not an algorithm, it is a rewrite: a recursion whose call
tree repeats itself gets folded into a table that is filled once.  Everything in
this file is the same one-dimensional table; only the combine step changes
(+ for counting, min for optimising, or for feasibility).

Sections
  1  climbing stairs: the exponential recursion and what is actually repeated
  2  three shapes of the same table: memo, tabulation, rolling variables
  3  coin change, minimum coins (LC 322) and reconstructing the actual coins
  4  the silent bug: combinations vs permutations (LC 518 vs LC 377)
  5  climbing stairs IS coin change with coins {1, 2}
  6  Day 08 word break, re-read as the same table
  7  when memoisation beats tabulation (and when it does not)
  8  LeetCode round-up with asserts
"""

from functools import lru_cache
import sys
import time


# --------------------------------------------------------------------------
# 1  Climbing stairs
# --------------------------------------------------------------------------

CALLS = {'n': 0}


def stairs_naive(n):
    """Count ways to climb n steps taking 1 or 2 at a time.  Exponential."""
    CALLS['n'] += 1
    if n < 0:
        return 0
    if n == 0:
        return 1                       # one way to stand still: take nothing
    return stairs_naive(n - 1) + stairs_naive(n - 2)


def stairs_memo(n):
    """Same recursion, but every distinct n is computed once."""
    memo = {}

    def go(k):
        if k < 0:
            return 0
        if k == 0:
            return 1
        if k not in memo:
            memo[k] = go(k - 1) + go(k - 2)
        return memo[k]

    return go(n), memo


def stairs_table(n):
    """Bottom-up: fill dp[0..n] left to right.  dp[i] = ways to reach step i."""
    dp = [0] * (n + 1)
    dp[0] = 1                          # the empty climb
    for i in range(1, n + 1):
        dp[i] = dp[i - 1]              # arrive with a 1-step
        if i >= 2:
            dp[i] += dp[i - 2]         # arrive with a 2-step
    return dp


def stairs_rolling(n):
    """Only the last two cells are ever read, so keep two variables."""
    prev, cur = 1, 1                   # dp[0], dp[1]
    for _ in range(2, n + 1):
        prev, cur = cur, cur + prev
    return cur if n >= 1 else 1


def count_subproblems(n):
    """How many calls the naive version makes vs how many distinct values exist."""
    CALLS['n'] = 0
    stairs_naive(n)
    return CALLS['n'], n + 1


# --------------------------------------------------------------------------
# 3  Coin change - minimum number of coins (LeetCode 322)
# --------------------------------------------------------------------------

INF = float('inf')


def coin_change_min(coins, amount):
    """Fewest coins summing to amount, or -1.  dp[a] = best for amount a."""
    dp = [0] + [INF] * amount
    for a in range(1, amount + 1):
        for c in coins:
            if c <= a and dp[a - c] + 1 < dp[a]:
                dp[a] = dp[a - c] + 1
    return -1 if dp[amount] == INF else dp[amount]


def coin_change_min_trace(coins, amount):
    """Same table, plus the coin that produced each cell, so we can rebuild it."""
    dp = [0] + [INF] * amount
    pick = [None] * (amount + 1)       # pick[a] = last coin used to reach a
    for a in range(1, amount + 1):
        for c in coins:
            if c <= a and dp[a - c] + 1 < dp[a]:
                dp[a] = dp[a - c] + 1
                pick[a] = c
    if dp[amount] == INF:
        return -1, [], dp
    out, a = [], amount
    while a > 0:                       # walk the parent pointers backwards
        out.append(pick[a])
        a -= pick[a]
    return dp[amount], sorted(out, reverse=True), dp


def coin_change_min_greedy(coins, amount):
    """Take the biggest coin that fits, repeatedly.  Wrong in general."""
    left, used = amount, []
    for c in sorted(coins, reverse=True):
        while c <= left:
            left -= c
            used.append(c)
    return (-1, []) if left else (len(used), used)


# --------------------------------------------------------------------------
# 4  The silent bug: combinations vs permutations
# --------------------------------------------------------------------------

def coin_ways_combinations(coins, amount):
    """LC 518.  Coin loop OUTSIDE: each coin is offered once, so order is fixed."""
    dp = [1] + [0] * amount
    for c in coins:                    # <- outer
        for a in range(c, amount + 1):
            dp[a] += dp[a - c]
    return dp[amount]


def coin_ways_permutations(coins, amount):
    """LC 377.  Amount loop OUTSIDE: every coin may be the *last* one added."""
    dp = [1] + [0] * amount
    for a in range(1, amount + 1):     # <- outer
        for c in coins:
            if c <= a:
                dp[a] += dp[a - c]
    return dp[amount]


def enumerate_combinations(coins, amount):
    """Brute force multisets, for proof."""
    out = []

    def go(i, left, cur):
        if left == 0:
            out.append(tuple(cur))
            return
        if i == len(coins):
            return
        go(i + 1, left, cur)                       # skip this coin
        if coins[i] <= left:
            cur.append(coins[i])
            go(i, left - coins[i], cur)            # reuse allowed
            cur.pop()

    go(0, amount, [])
    return out


def enumerate_permutations(coins, amount):
    """Brute force ordered sequences, for proof."""
    out = []

    def go(left, cur):
        if left == 0:
            out.append(tuple(cur))
            return
        for c in coins:
            if c <= left:
                cur.append(c)
                go(left - c, cur)
                cur.pop()

    go(amount, [])
    return out


# --------------------------------------------------------------------------
# 6  Word break, re-read (Day 08)
# --------------------------------------------------------------------------

def word_break(s, words):
    """dp[i] = can s[:i] be segmented?  Same table, combine step is `or`."""
    wordset = set(words)
    dp = [True] + [False] * len(s)
    for i in range(1, len(s) + 1):
        for j in range(i):
            if dp[j] and s[j:i] in wordset:
                dp[i] = True
                break
    return dp[len(s)]


def word_break_count(s, words):
    """Swap `or` for `+` and the same table counts the segmentations."""
    wordset = set(words)
    dp = [1] + [0] * len(s)
    for i in range(1, len(s) + 1):
        for j in range(i):
            if dp[j] and s[j:i] in wordset:
                dp[i] += dp[j]
    return dp[len(s)]


# --------------------------------------------------------------------------
# 7  Memoisation vs tabulation
# --------------------------------------------------------------------------

def coin_change_min_memo(coins, amount):
    """Top-down.  Only the amounts actually reachable get computed."""
    visited = set()

    @lru_cache(maxsize=None)
    def best(a):
        visited.add(a)
        if a == 0:
            return 0
        if a < 0:
            return INF
        return min((best(a - c) + 1 for c in coins), default=INF)

    sys.setrecursionlimit(100000)
    r = best(amount)
    best.cache_clear()
    return (-1 if r == INF else r), len(visited)


def sparse_state_report(coins, amount):
    """Tabulation touches every amount; memoisation touches only the useful ones."""
    r_memo, states = coin_change_min_memo(coins, amount)
    r_tab = coin_change_min(coins, amount)
    return r_tab, r_memo, states, amount + 1


# --------------------------------------------------------------------------
# 8  LeetCode
# --------------------------------------------------------------------------

def lc70_climb_stairs(n):
    a, b = 1, 1
    for _ in range(n - 1):
        a, b = b, a + b
    return b


def lc322_coin_change(coins, amount):
    return coin_change_min(coins, amount)


def lc518_change(amount, coins):
    return coin_ways_combinations(coins, amount)


def lc377_combination_sum4(nums, target):
    return coin_ways_permutations(nums, target)


def lc139_word_break(s, word_dict):
    return word_break(s, word_dict)


# --------------------------------------------------------------------------
# printing helpers
# --------------------------------------------------------------------------

def section(title):
    print()
    print('=' * 74)
    print(title)
    print('=' * 74)


def show_table(dp, width=5, limit=16, fmt=str):
    cells = dp[:limit]
    idx = ''.join(str(i).rjust(width) for i in range(len(cells)))
    row = ''.join(fmt(v).rjust(width) for v in cells)
    print('  index ' + idx)
    print('  dp    ' + row)


def money(v):
    return '-' if v == INF else str(v)


def main():
    section('1  Climbing stairs: what the naive recursion actually repeats')
    for n in (10, 20, 25, 30):
        calls, distinct = count_subproblems(n)
        print(f'  n = {n:>2}   naive calls {calls:>9,}   distinct subproblems {distinct:>3}'
              f'   waste {calls / distinct:>9,.0f}x')
    print()
    print('  The recursion is not wrong, it is redundant. Inside stairs_naive(30)')
    print('  the call f(1) is evaluated 832,040 times and f(0) 1,346,269 times,')
    print('  because nothing remembers an answer that was already found.')

    section('2  Three shapes of one table')
    n = 10
    val, memo = stairs_memo(n)
    dp = stairs_table(n)
    print(f'  memoised     stairs({n}) = {val}, memo holds {len(memo)} entries')
    print(f'  tabulated    dp = {dp}')
    print(f'  rolling      stairs({n}) = {stairs_rolling(n)}  (two variables, O(1) space)')
    show_table(dp)
    print()
    print('  dp[i] = dp[i-1] + dp[i-2] is Fibonacci, which is the giveaway that')
    print('  "count the ways" problems add, they do not minimise.')
    assert val == dp[n] == stairs_rolling(n) == 89

    section('3  Coin change: fewest coins (LC 322)')
    coins = [1, 5, 6, 9]
    amount = 11
    best, used, dp = coin_change_min_trace(coins, amount)
    print(f'  coins = {coins}, amount = {amount}')
    show_table(dp, fmt=money, limit=amount + 1)
    print(f'  dp[{amount}] = {best} coins: {used}')
    g_count, g_used = coin_change_min_greedy(coins, amount)
    print(f'  greedy (biggest coin first) = {g_count} coins: {g_used}')
    print()
    print('  Greedy takes the 9 and is then stuck with 1+1; DP takes 5+6.')
    print('  Greedy is only safe on canonical coin systems - a currency is designed')
    print('  to make it safe, an arbitrary coin set is not.')
    assert best == 2 and used == [6, 5]
    assert g_count == 3

    print()
    print('  unreachable amounts stay at infinity and come back as -1:')
    print(f'  coin_change_min([5, 7], 3) = {coin_change_min([5, 7], 3)}')
    assert coin_change_min([5, 7], 3) == -1

    section('4  The silent bug: combinations vs permutations')
    coins = [1, 2, 5]
    amount = 5
    comb = coin_ways_combinations(coins, amount)
    perm = coin_ways_permutations(coins, amount)
    print(f'  coins = {coins}, amount = {amount}')
    print(f'  coin loop outside  (LC 518, combinations) = {comb}')
    print(f'  amount loop outside(LC 377, permutations) = {perm}')
    print()
    print('  brute-force multisets :', [' + '.join(map(str, c)) for c in
                                        sorted(enumerate_combinations(coins, amount))])
    print('  brute-force sequences :', len(enumerate_permutations(coins, amount)))
    print()
    print('  The two functions differ by which loop is outer. Nothing raises,')
    print('  nothing looks suspicious, and one of them answers a different question.')
    assert comb == len(enumerate_combinations(coins, amount)) == 4
    assert perm == len(enumerate_permutations(coins, amount)) == 9

    print()
    print('  the gap grows fast:')
    print('   amount  combinations  permutations')
    for a in (5, 10, 20, 30):
        print(f'   {a:>6}  {coin_ways_combinations([1, 2, 5], a):>12,}'
              f'  {coin_ways_permutations([1, 2, 5], a):>13,}')
    print()
    print('  Reading it out loud fixes it: with the coin loop outside, coin 5 is')
    print('  offered once and for all, so "1+2+2" is built in exactly one order.')
    print('  With the amount loop outside, every coin gets a turn at being the')
    print('  last one added, so 1+2+2, 2+1+2 and 2+2+1 are three different fills.')

    section('5  Climbing stairs IS coin change with coins {1, 2}')
    for n in range(1, 11):
        a = stairs_rolling(n)
        b = coin_ways_permutations([1, 2], n)
        c = coin_ways_combinations([1, 2], n)
        print(f'  n = {n:>2}   stairs {a:>3}   permutations {b:>3}   combinations {c:>3}')
        assert a == b
    print()
    print('  Steps are ordered - 1 then 2 is a different climb from 2 then 1 - so')
    print('  stairs matches the permutation table exactly. If you ever write the')
    print('  stairs recurrence with the step loop outside, you have quietly started')
    print('  counting how many 2-steps you took, not how you climbed.')

    section('6  Day 08 word break, re-read as the same table')
    s = 'applepenapple'
    words = ['apple', 'pen']
    print(f'  s = {s!r}, words = {words}')
    print(f'  word_break        -> {word_break(s, words)}')
    print(f'  word_break_count  -> {word_break_count(s, words)} segmentation(s)')
    bad = 'catsandog'
    print(f'  s = {bad!r} with {["cats", "dog", "sand", "and", "cat"]}'
          f' -> {word_break(bad, ["cats", "dog", "sand", "and", "cat"])}')
    print()
    print('  Three problems, one table:')
    print('    stairs        dp[i] = sum over steps      combine with +')
    print('    coin change   dp[a] = min over coins      combine with min')
    print('    word break    dp[i] = any over cut points combine with or')
    print('  The loop structure is identical; the operator is the whole difference.')
    assert word_break(s, words) is True
    assert word_break_count(s, words) == 1
    assert word_break(bad, ['cats', 'dog', 'sand', 'and', 'cat']) is False

    section('7  Memoisation vs tabulation: who touches fewer states')
    for coins, amount in (([100, 250], 10000), ([1, 2, 5], 1000)):
        r_tab, r_memo, states, cells = sparse_state_report(coins, amount)
        print(f'  coins {str(coins):<10} amount {amount}   answer {r_tab}'
              f'   memo states {states:>5}   table cells {cells:>5}'
              f'   ratio {cells / states:>5.1f}x')
        assert r_tab == r_memo
    print()
    print('  With coins {100, 250} only multiples of 50 are ever asked about, so')
    print('  top-down visits 203 states out of 10,001 cells - the table is mostly')
    print('  zeros that tabulation fills anyway. With {1, 2, 5} every amount is')
    print('  reachable, the two coincide, and tabulation wins instead: no call')
    print('  overhead, and no 1000-deep call stack. Top-down on amount 1000 needs')
    print('  sys.setrecursionlimit or it dies with RecursionError - the default')
    print('  limit of 1000 is exactly the depth of the chain 1000, 999, 998, ...')

    section('8  LeetCode')
    print(f'  70  climbStairs(10)                     = {lc70_climb_stairs(10)}')
    print(f'  322 coinChange([1,2,5], 11)             = {lc322_coin_change([1, 2, 5], 11)}')
    print(f'  322 coinChange([2], 3)                  = {lc322_coin_change([2], 3)}')
    print(f'  518 change(5, [1,2,5])                  = {lc518_change(5, [1, 2, 5])}')
    print(f'  377 combinationSum4([1,2,3], 4)         = {lc377_combination_sum4([1, 2, 3], 4)}')
    print(f'  139 wordBreak("leetcode", ["leet","code"]) = '
          f'{lc139_word_break("leetcode", ["leet", "code"])}')
    assert lc70_climb_stairs(10) == 89
    assert lc322_coin_change([1, 2, 5], 11) == 3
    assert lc322_coin_change([2], 3) == -1
    assert lc518_change(5, [1, 2, 5]) == 4
    assert lc377_combination_sum4([1, 2, 3], 4) == 7
    assert lc139_word_break('leetcode', ['leet', 'code']) is True

    print()
    print('  LC 377 is called "Combination Sum IV" and asks for permutations.')
    print('  The name is wrong; the examples are not. Read the examples.')

    section('timing: the same answer, three ways')
    t0 = time.perf_counter(); stairs_naive(28); t1 = time.perf_counter()
    stairs_memo(28); t2 = time.perf_counter()
    stairs_rolling(28); t3 = time.perf_counter()
    print(f'  naive    {(t1 - t0) * 1000:>8.2f} ms')
    print(f'  memoised {(t2 - t1) * 1000:>8.3f} ms')
    print(f'  rolling  {(t3 - t2) * 1000:>8.4f} ms')

    print()
    print('all assertions passed')


if __name__ == '__main__':
    main()
