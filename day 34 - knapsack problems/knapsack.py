"""Day 34 - the knapsack family: 0/1, unbounded and bounded.

Every variant here is the same table with the same recurrence.  What changes is
the *order* the table is filled in, and that order is the whole answer to the
question "may I take this item more than once?".

Run me:  python knapsack.py
"""

from itertools import product
import time

# --------------------------------------------------------------------------
# the running example
# --------------------------------------------------------------------------
# four items, a knapsack that holds 10 units of weight
NAMES = ['rope', 'book', 'pan', 'tent']
WEIGHTS = [3, 4, 5, 6]
VALUES = [50, 40, 70, 80]
CAP = 10


# --------------------------------------------------------------------------
# 1. 0/1 knapsack, the honest two-dimensional table
# --------------------------------------------------------------------------
def knap01_table(weights, values, cap):
    """dp[i][c] = best value using the first i items with capacity c.

    Two choices for item i-1: leave it (inherit the row above) or take it
    (add its value to dp[i-1][c - w], the row above at a smaller capacity).
    Both readings come from row i-1, which is why the rows can be filled
    top to bottom without ever looking at unfinished work.
    """
    n = len(weights)
    dp = [[0] * (cap + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        w, v = weights[i - 1], values[i - 1]
        for c in range(cap + 1):
            dp[i][c] = dp[i - 1][c]                       # leave it
            if w <= c and dp[i - 1][c - w] + v > dp[i][c]:
                dp[i][c] = dp[i - 1][c - w] + v           # take it
    return dp


def knap01_items(weights, values, cap):
    """Walk the finished table backwards to recover *which* items were taken."""
    dp = knap01_table(weights, values, cap)
    chosen, c = [], cap
    for i in range(len(weights), 0, -1):
        if dp[i][c] != dp[i - 1][c]:       # the row changed => item i-1 was taken
            chosen.append(i - 1)
            c -= weights[i - 1]
    chosen.reverse()
    return dp[len(weights)][cap], chosen


# --------------------------------------------------------------------------
# 2. the same thing in one row - and the direction that makes it 0/1
# --------------------------------------------------------------------------
def knap01_rolling(weights, values, cap):
    """One row, inner loop **descending**.

    dp[c] is about to become row i; dp[c - w] must still be row i-1.  Walking
    down means the cells we read on the left have not been touched this round,
    so each item is offered exactly once.
    """
    dp = [0] * (cap + 1)
    for w, v in zip(weights, values):
        for c in range(cap, w - 1, -1):          # descending: read the old row
            if dp[c - w] + v > dp[c]:
                dp[c] = dp[c - w] + v
    return dp[cap]


def knap_unbounded(weights, values, cap):
    """Same code, inner loop **ascending** - and now items are unlimited.

    dp[c - w] has already been updated this round, so it may already contain a
    copy of this very item.  Taking it again is exactly what "unbounded" means.
    """
    dp = [0] * (cap + 1)
    for w, v in zip(weights, values):
        for c in range(w, cap + 1):              # ascending: read the new row
            if dp[c - w] + v > dp[c]:
                dp[c] = dp[c - w] + v
    return dp[cap]


def knap_rolling_rows(weights, values, cap, descending=True):
    """Record the row after each item, for the figures and the demo page."""
    dp = [0] * (cap + 1)
    rows = [dp[:]]
    for w, v in zip(weights, values):
        rng = range(cap, w - 1, -1) if descending else range(w, cap + 1)
        for c in rng:
            if dp[c - w] + v > dp[c]:
                dp[c] = dp[c - w] + v
        rows.append(dp[:])
    return rows


def unbounded_counts(weights, values, cap):
    """How many copies of each item the ascending loop actually used."""
    dp = [0] * (cap + 1)
    pick = [-1] * (cap + 1)
    for i, (w, v) in enumerate(zip(weights, values)):
        for c in range(w, cap + 1):
            if dp[c - w] + v > dp[c]:
                dp[c] = dp[c - w] + v
                pick[c] = i
    counts = [0] * len(weights)
    c = cap
    while c > 0 and pick[c] >= 0:
        i = pick[c]
        counts[i] += 1
        c -= weights[i]
    return dp[cap], counts


# --------------------------------------------------------------------------
# 3. bounded ("multiple") knapsack - k copies of each item
# --------------------------------------------------------------------------
def knap_bounded_naive(weights, values, counts, cap):
    """Expand every item into `count` separate items and run plain 0/1.

    Correct, and the cost is O(cap * sum(counts)) - which is linear in the
    *values* of the counts, not in the number of digits it takes to write them.
    """
    w2, v2 = [], []
    for w, v, k in zip(weights, values, counts):
        w2 += [w] * k
        v2 += [v] * k
    return knap01_rolling(w2, v2, cap), len(w2)


def binary_split(k):
    """Split k into 1, 2, 4, ... and a remainder.

    Any number from 0 to k can be written as a subset sum of these parts, so a
    0/1 choice over the parts reproduces every "how many copies" choice - with
    O(log k) items instead of k.
    """
    parts, p = [], 1
    while p <= k:
        parts.append(p)
        k -= p
        p *= 2
    if k:
        parts.append(k)
    return parts


def binary_split_reaches(k):
    """Proof by exhaustion that the parts really do cover 0..k."""
    parts = binary_split(k)
    seen = set()
    for mask in range(1 << len(parts)):
        seen.add(sum(p for j, p in enumerate(parts) if mask >> j & 1))
    return seen == set(range(k + 1)), parts


def knap_bounded_binary(weights, values, counts, cap):
    """Bounded knapsack via binary splitting: O(cap * sum(log k))."""
    w2, v2 = [], []
    for w, v, k in zip(weights, values, counts):
        for p in binary_split(k):
            w2.append(w * p)
            v2.append(v * p)
    return knap01_rolling(w2, v2, cap), len(w2)


# --------------------------------------------------------------------------
# 4. the greedy that is exactly right for one problem and wrong for the other
# --------------------------------------------------------------------------
def fractional_greedy(weights, values, cap):
    """Best value/weight first, allowed to cut the last item.

    This is optimal for the *fractional* knapsack - an exchange argument proves
    it - and it is the one greedy in this article that is not a trap.
    """
    order = sorted(range(len(weights)), key=lambda i: values[i] / weights[i],
                   reverse=True)
    total, left, taken = 0.0, cap, []
    for i in order:
        if left <= 0:
            break
        take = min(1.0, left / weights[i])
        total += take * values[i]
        left -= take * weights[i]
        taken.append((i, take))
    return total, taken


def greedy_01(weights, values, cap):
    """The same rule with no cutting allowed - now it can lose, silently."""
    order = sorted(range(len(weights)), key=lambda i: values[i] / weights[i],
                   reverse=True)
    total, left, taken = 0, cap, []
    for i in order:
        if weights[i] <= left:
            total += values[i]
            left -= weights[i]
            taken.append(i)
    return total, taken


# --------------------------------------------------------------------------
# 5. subset sum: the same table with `or` instead of `max`
# --------------------------------------------------------------------------
def subset_sum(nums, target):
    """Reachability, not optimisation - Day 33's third operator again."""
    dp = [False] * (target + 1)
    dp[0] = True
    for x in nums:
        for c in range(target, x - 1, -1):
            if dp[c - x]:
                dp[c] = True
    return dp[target]


def subset_sum_bits(nums, target):
    """The whole boolean row packed into one Python integer.

    `bits <<= x` shifts every reachable sum up by x at once, and `|` merges the
    two branches.  One big-integer shift replaces the inner loop, and the
    descending-order worry disappears because the shift reads a snapshot.
    """
    bits = 1
    for x in nums:
        bits |= bits << x
    return bool(bits >> target & 1)


# --------------------------------------------------------------------------
# 6. LeetCode - five problems that are all this one table
# --------------------------------------------------------------------------
def lc416_can_partition(nums):
    """416. Partition Equal Subset Sum - 0/1 subset sum to total/2."""
    total = sum(nums)
    if total % 2:
        return False
    return subset_sum_bits(nums, total // 2)


def lc494_find_target_sum_ways(nums, target):
    """494. Target Sum - signs become a subset: P - N = target, P + N = sum,
    so the positive side must sum to (sum + target) / 2.  Counting subsets is
    the `+` version of the same row."""
    total = sum(nums)
    if abs(target) > total or (total + target) % 2:
        return 0
    s = (total + target) // 2
    dp = [0] * (s + 1)
    dp[0] = 1
    for x in nums:
        for c in range(s, x - 1, -1):
            dp[c] += dp[c - x]
    return dp[s]


def lc1049_last_stone_weight_ii(stones):
    """1049. Last Stone Weight II - smashing stones is really splitting them
    into two piles; the answer is total - 2 * (best pile <= total/2)."""
    total = sum(stones)
    half = total // 2
    dp = [0] * (half + 1)
    for x in stones:
        for c in range(half, x - 1, -1):
            if dp[c - x] + x > dp[c]:
                dp[c] = dp[c - x] + x
    return total - 2 * dp[half]


def lc474_find_max_form(strs, m, n):
    """474. Ones and Zeroes - a 0/1 knapsack with two capacities, so the
    rolling row becomes a rolling *grid* and both loops run backwards."""
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for s in strs:
        z, o = s.count('0'), s.count('1')
        for i in range(m, z - 1, -1):
            for j in range(n, o - 1, -1):
                if dp[i - z][j - o] + 1 > dp[i][j]:
                    dp[i][j] = dp[i - z][j - o] + 1
    return dp[m][n]


def lc279_num_squares(n):
    """279. Perfect Squares - unbounded knapsack minimising the count, so the
    inner loop runs forwards on purpose."""
    squares = []
    k = 1
    while k * k <= n:
        squares.append(k * k)
        k += 1
    INF = float('inf')
    dp = [0] + [INF] * n
    for s in squares:
        for c in range(s, n + 1):
            if dp[c - s] + 1 < dp[c]:
                dp[c] = dp[c - s] + 1
    return dp[n]


# --------------------------------------------------------------------------
# 7. pseudo-polynomial: why O(n * cap) is not polynomial
# --------------------------------------------------------------------------
def scale_report(weights, values, cap, factors=(1, 10, 100, 1000)):
    """Multiply every weight and the capacity by f.  The answer never changes,
    the input grows by log2(f) bits per number, and the table grows by f."""
    rows = []
    for f in factors:
        w = [x * f for x in weights]
        best = knap01_rolling(w, values, cap * f)
        cells = len(weights) * (cap * f + 1)
        bits = sum(max(1, x.bit_length()) for x in w) + (cap * f).bit_length()
        rows.append((f, best, cells, bits))
    return rows


def timing(n_items=200, cap=5000, seed=34):
    import random
    rnd = random.Random(seed)
    w = [rnd.randint(1, 100) for _ in range(n_items)]
    v = [rnd.randint(1, 1000) for _ in range(n_items)]
    out = {}
    t = time.perf_counter()
    a = knap01_table(w, v, cap)[n_items][cap]
    out['2d table'] = (time.perf_counter() - t) * 1000
    t = time.perf_counter()
    b = knap01_rolling(w, v, cap)
    out['rolling row'] = (time.perf_counter() - t) * 1000
    assert a == b
    return out, a


# --------------------------------------------------------------------------
# walkthrough
# --------------------------------------------------------------------------
def _row(vals, width=5):
    return ''.join(str(x).rjust(width) for x in vals)


def main():
    print('=' * 68)
    print('1. 0/1 knapsack - the two-dimensional table')
    print('=' * 68)
    print(f'capacity {CAP}')
    for nm, w, v in zip(NAMES, WEIGHTS, VALUES):
        print(f'  {nm:<5} weight {w}  value {v:>3}  ratio {v / w:.2f}')
    dp = knap01_table(WEIGHTS, VALUES, CAP)
    print('\n      c =' + _row(range(CAP + 1)))
    print('  {} '.format('-' * 4) + _row(['-' * 4] * (CAP + 1)))
    for i, row in enumerate(dp):
        tag = 'none' if i == 0 else NAMES[i - 1]
        print(f'  {tag:<5}' + _row(row))
    best, chosen = knap01_items(WEIGHTS, VALUES, CAP)
    names = ' + '.join(NAMES[i] for i in chosen)
    load = sum(WEIGHTS[i] for i in chosen)
    print(f'\nbest = {best} using {names} (weight {load} of {CAP})')
    print('each cell reads only the row above, so rows fill top to bottom')

    print()
    print('=' * 68)
    print('2. one row instead of n+1 - and the direction that keeps it 0/1')
    print('=' * 68)
    rows = knap_rolling_rows(WEIGHTS, VALUES, CAP, descending=True)
    print('      c =' + _row(range(CAP + 1)))
    for i, r in enumerate(rows):
        tag = 'start' if i == 0 else '+' + NAMES[i - 1]
        print(f'  {tag:<5}' + _row(r))
    roll = knap01_rolling(WEIGHTS, VALUES, CAP)
    print(f'\nrolling row -> {roll}   (2-D table -> {best})')
    assert roll == best
    print('descending: dp[c-w] has not been touched yet this round, so it is')
    print('still row i-1 - the item is offered exactly once')

    print()
    print('=' * 68)
    print('3. THE SILENT FAILURE: turn the inner loop around')
    print('=' * 68)
    up = knap_unbounded(WEIGHTS, VALUES, CAP)
    rows_up = knap_rolling_rows(WEIGHTS, VALUES, CAP, descending=False)
    print('      c =' + _row(range(CAP + 1)))
    for i, r in enumerate(rows_up):
        tag = 'start' if i == 0 else '+' + NAMES[i - 1]
        print(f'  {tag:<5}' + _row(r))
    print(f'\ndescending -> {roll}      ascending -> {up}')
    tot, counts = unbounded_counts(WEIGHTS, VALUES, CAP)
    used = ', '.join(f'{c}x {nm}' for nm, c in zip(NAMES, counts) if c)
    print(f'the ascending answer packs {used} = weight '
          f'{sum(c * w for c, w in zip(counts, WEIGHTS))}')
    print('no exception, no warning: `reversed()` IS the 0/1 constraint')
    assert up > roll

    print()
    print('=' * 68)
    print('4. bounded knapsack: k copies, and the log-k trick')
    print('=' * 68)
    counts_b = [2, 1, 3, 1]
    for nm, k in zip(NAMES, counts_b):
        print(f'  {nm:<5} up to {k} copies')
    naive, n_naive = knap_bounded_naive(WEIGHTS, VALUES, counts_b, CAP)
    fast, n_fast = knap_bounded_binary(WEIGHTS, VALUES, counts_b, CAP)
    print(f'\nexpanded to {n_naive} items -> {naive}')
    print(f'binary split into {n_fast} items -> {fast}')
    assert naive == fast
    for k in (1, 3, 7, 13, 100, 1000):
        ok, parts = binary_split_reaches(k) if k <= 100 else (None, binary_split(k))
        shown = parts if len(parts) <= 8 else parts[:7] + ['...']
        extra = '' if ok is None else f'  covers 0..{k}: {ok}'
        print(f'  k = {k:>4} -> {len(parts)} parts {shown}{extra}')
    ok13, parts13 = binary_split_reaches(13)
    assert ok13 and parts13 == [1, 2, 4, 6]
    bw, bv = [3, 4, 5, 6, 7], [50, 40, 70, 80, 95]
    bk, bcap = [300, 500, 800, 1200, 2000], 4000
    t = time.perf_counter()
    big_naive, n1 = knap_bounded_naive(bw, bv, bk, bcap)
    t1 = (time.perf_counter() - t) * 1000
    t = time.perf_counter()
    big_fast, n2 = knap_bounded_binary(bw, bv, bk, bcap)
    t2 = (time.perf_counter() - t) * 1000
    assert big_naive == big_fast
    print(f'\ncounts {bk}, capacity {bcap} -> {big_fast}')
    print(f'  expanded  {n1:>5} items  {t1:8.1f} ms')
    print(f'  split     {n2:>5} items  {t2:8.1f} ms   ({t1 / t2:.0f}x)')

    print()
    print('=' * 68)
    print('5. greedy: right answer for one problem, wrong for the other')
    print('=' * 68)
    gw, gv, gcap = WEIGHTS, VALUES, CAP
    print('same four items, ranked by value per unit of weight:')
    for i in sorted(range(len(gw)), key=lambda j: gv[j] / gw[j], reverse=True):
        print(f'  {NAMES[i]:<5} {gv[i]:>3} / {gw[i]} = {gv[i] / gw[i]:.2f}')
    frac, ftaken = fractional_greedy(gw, gv, gcap)
    g01, gtaken = greedy_01(gw, gv, gcap)
    opt = knap01_rolling(gw, gv, gcap)
    print(f'\nfractional greedy = {frac:.2f}   (optimal, cutting allowed)')
    print('   took ' + ', '.join(f'{NAMES[i]} x{t:.2f}' for i, t in ftaken))
    print(f'0/1 greedy        = {g01}   took '
          + ' + '.join(NAMES[i] for i in gtaken))
    print(f'0/1 DP            = {opt}   <- greedy loses {opt - g01} and says nothing')
    assert g01 < opt == best

    print()
    print('=' * 68)
    print('6. subset sum: same row, `or` instead of `max`')
    print('=' * 68)
    nums = [3, 34, 4, 12, 5, 2]
    for t in (9, 30, 11):
        print(f'  target {t:>3}: list dp {subset_sum(nums, t)}, '
              f'bitset {subset_sum_bits(nums, t)}')
        assert subset_sum(nums, t) == subset_sum_bits(nums, t)
    big = [i % 97 + 1 for i in range(400)]
    tgt = sum(big) // 2
    def _best(fn, reps):
        out = float('inf')
        for _ in range(reps):
            t = time.perf_counter()
            r = fn(big, tgt)
            out = min(out, time.perf_counter() - t)
        return out, r
    slow, r1 = _best(subset_sum, 3)
    fast_t, r2 = _best(subset_sum_bits, 30)
    assert r1 == r2
    print(f'\n400 numbers, target {tgt} (best of several runs):')
    print(f'  list of bools {slow * 1000:8.1f} ms')
    print(f'  one big int   {fast_t * 1000:8.2f} ms   ({slow / fast_t:.0f}x)')

    print()
    print('=' * 68)
    print('7. LeetCode')
    print('=' * 68)
    print(f'  416 [1,5,11,5]              -> {lc416_can_partition([1, 5, 11, 5])}')
    print(f'  416 [1,2,3,5]               -> {lc416_can_partition([1, 2, 3, 5])}')
    print(f'  416 [2,2,3,5]               -> {lc416_can_partition([2, 2, 3, 5])}')
    print(f'  494 [1,1,1,1,1] target 3    -> {lc494_find_target_sum_ways([1] * 5, 3)}')
    print(f'  1049 [2,7,4,1,8,1]          -> {lc1049_last_stone_weight_ii([2, 7, 4, 1, 8, 1])}')
    strs = ['10', '0001', '111001', '1', '0']
    print(f'  474 m=5 n=3                 -> {lc474_find_max_form(strs, 5, 3)}')
    print(f'  279 n=12                    -> {lc279_num_squares(12)}')
    print(f'  279 n=13                    -> {lc279_num_squares(13)}')
    assert lc416_can_partition([1, 5, 11, 5]) is True
    assert lc416_can_partition([1, 2, 3, 5]) is False
    assert lc416_can_partition([2, 2, 3, 5]) is False   # even total, still unreachable
    assert lc494_find_target_sum_ways([1] * 5, 3) == 5
    assert lc1049_last_stone_weight_ii([2, 7, 4, 1, 8, 1]) == 1
    assert lc474_find_max_form(strs, 5, 3) == 4
    assert lc279_num_squares(12) == 3 and lc279_num_squares(13) == 2

    print()
    print('=' * 68)
    print('8. pseudo-polynomial: the table grows with the *value* of W')
    print('=' * 68)
    print('  factor    best   table cells   input bits')
    for f, b, cells, bits in scale_report(WEIGHTS, VALUES, CAP):
        print(f'  {f:>6}  {b:>6}   {cells:>11}   {bits:>10}')
    print('  same answer every time; 4 more bits of input, 1000x the work')

    t, best_t = timing()
    print(f'\n200 items, capacity 5000 (best {best_t}):')
    for k, ms in t.items():
        print(f'  {k:<12} {ms:8.2f} ms')

    print('\nall assertions passed')


if __name__ == '__main__':
    main()
