"""Day 35 - LIS and LCS: the subsequence you keep, and the table you throw away.

Two "longest subsequence" problems that look like twins and behave nothing alike.
LIS has a quadratic DP that can be deleted outright: patience sorting plus the
`lower_bound` from day 23 answers it in O(n log n).  LCS has no such trick in
general - unless the two inputs are permutations of one another, in which case
LCS *is* LIS and inherits the speedup.

Run me:  python lis_lcs.py
"""

from bisect import bisect_left, bisect_right
import random
import time

# --------------------------------------------------------------------------
# the running example
# --------------------------------------------------------------------------
A = [10, 9, 2, 5, 3, 7, 101, 18]        # LIS length 4:  2, 3, 7, 18 / 2, 3, 7, 101
B = [3, 4, 5, 10]                        # a second, friendlier sequence


# --------------------------------------------------------------------------
# 1. LIS, the honest quadratic DP
# --------------------------------------------------------------------------
def lis_dp(a):
    """dp[i] = length of the longest increasing subsequence *ending at* i.

    "Ending at i" is the whole design decision.  The natural phrasing - "the
    LIS of the first i elements" - is not enough state: to extend a
    subsequence you have to know what its last element was, and that phrasing
    has thrown it away.
    """
    n = len(a)
    if n == 0:
        return 0, []
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


# --------------------------------------------------------------------------
# 2. patience sorting: one array of tails, one binary search per element
# --------------------------------------------------------------------------
def lis_patience(a, strict=True):
    """tails[k] = the smallest value that can end an increasing subsequence
    of length k + 1 seen so far.

    tails is sorted by construction, so the position of the incoming value is
    a `lower_bound` (day 23).  `bisect_left` refuses to sit after an equal
    value and therefore builds a *strictly* increasing subsequence;
    `bisect_right` allows it and builds a non-decreasing one.  One function
    name is the entire difference between the two problems.
    """
    find = bisect_left if strict else bisect_right
    tails = []
    for x in a:
        k = find(tails, x)
        if k == len(tails):
            tails.append(x)          # x extends the longest run so far
        else:
            tails[k] = x             # x is a cheaper ending for length k+1
    return len(tails), tails


def lis_patience_reconstruct(a):
    """The same walk, but recording who came before whom.

    This is the trap the tails array sets: `tails` has the right *length* and
    is almost never a subsequence of the input.  The actual answer has to be
    recovered from parent pointers taken at insert time.
    """
    tails, idx, parent = [], [], [-1] * len(a)
    for i, x in enumerate(a):
        k = bisect_left(tails, x)
        if k == len(tails):
            tails.append(x)
            idx.append(i)
        else:
            tails[k] = x
            idx[k] = i
        parent[i] = idx[k - 1] if k else -1
    out, i = [], (idx[-1] if idx else -1)
    while i != -1:
        out.append(a[i])
        i = parent[i]
    out.reverse()
    return len(tails), out, tails


def patience_piles(a):
    """The card game the algorithm is named after: each pile's top card is a
    tails entry, and the number of piles is the answer."""
    piles = []
    for x in a:
        k = bisect_left([p[-1] for p in piles], x)
        if k == len(piles):
            piles.append([x])
        else:
            piles[k].append(x)
    return piles


# --------------------------------------------------------------------------
# 3. LeetCode 354 - Russian Doll Envelopes: sort, then LIS
# --------------------------------------------------------------------------
def max_envelopes(envelopes):
    """Sort width ascending, height *descending* inside equal widths, then
    take the LIS of the heights.

    The descending tie-break is the whole problem.  Equal widths cannot nest,
    and a descending height order guarantees that no two envelopes of the same
    width can both appear in an increasing run of heights.  Sort both keys
    ascending and [(1,1),(1,2)] silently becomes a valid pair of dolls.
    """
    envelopes.sort(key=lambda e: (e[0], -e[1]))
    return lis_patience([h for _, h in envelopes])[0]


# --------------------------------------------------------------------------
# 4. LeetCode 673 - Number of Longest Increasing Subsequence
# --------------------------------------------------------------------------
def count_lis(a):
    """Carry a second table: cnt[i] = how many LIS of length dp[i] end at i.

    When a longer subsequence is found the count is *replaced*; when an
    equally long one is found the counts are *added*.  Mixing those two up is
    the classic off-by-a-lot in this problem.
    """
    n = len(a)
    if n == 0:
        return 0
    dp = [1] * n
    cnt = [1] * n
    for i in range(n):
        for j in range(i):
            if a[j] < a[i]:
                if dp[j] + 1 > dp[i]:
                    dp[i] = dp[j] + 1
                    cnt[i] = cnt[j]
                elif dp[j] + 1 == dp[i]:
                    cnt[i] += cnt[j]
    longest = max(dp)
    return sum(c for l, c in zip(dp, cnt) if l == longest)


# --------------------------------------------------------------------------
# 5. LCS - the table everyone draws
# --------------------------------------------------------------------------
def lcs_table(x, y):
    """dp[i][j] = LCS length of x[:i] and y[:j].

    Two cases only.  If the last characters match, they *must* be usable
    together - no optimal solution is hurt by taking them - so the answer is
    1 + the diagonal.  If they do not match, one of the two last characters is
    useless, and we do not know which, so we take the better of dropping each.
    """
    n, m = len(x), len(y)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if x[i - 1] == y[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    return dp


def lcs_backtrack(x, y, dp=None):
    """Walk the finished table backwards from the bottom-right corner."""
    dp = dp or lcs_table(x, y)
    i, j, out = len(x), len(y), []
    while i and j:
        if x[i - 1] == y[j - 1]:
            out.append(x[i - 1])
            i -= 1
            j -= 1
        elif dp[i - 1][j] >= dp[i][j - 1]:
            i -= 1
        else:
            j -= 1
    out.reverse()
    return ''.join(out) if isinstance(x, str) else out


def lcs_rolling(x, y):
    """Two rows instead of n+1 - O(min(n, m)) space.

    The length survives; the path does not.  Backtracking needs the whole
    table, so the rolling-row version can only ever answer "how long".  Same
    trade as the 1-D knapsack on day 34.
    """
    if len(y) > len(x):
        x, y = y, x
    prev = [0] * (len(y) + 1)
    for i in range(1, len(x) + 1):
        cur = [0] * (len(y) + 1)
        for j in range(1, len(y) + 1):
            cur[j] = prev[j - 1] + 1 if x[i - 1] == y[j - 1] else max(prev[j], cur[j - 1])
        prev = cur
    return prev[-1]


# --------------------------------------------------------------------------
# 6. LCS of two permutations IS an LIS problem (Hunt-Szymanski)
# --------------------------------------------------------------------------
def lcs_of_permutations(p, q):
    """When every element is distinct and both lists hold the same set,
    relabel q by "where does this value sit in p".

    A common subsequence of p and q is then exactly an increasing run of those
    labels, so the answer is an LIS - O(n log n) instead of O(n*m).  This is
    the small case of the Hunt-Szymanski algorithm, which does the same trick
    with match lists when values repeat.
    """
    pos = {v: i for i, v in enumerate(p)}
    mapped = [pos[v] for v in q if v in pos]
    length, seq, _ = lis_patience_reconstruct(mapped)
    return length, [p[i] for i in seq]


# --------------------------------------------------------------------------
# 7. LeetCode 1035 - Uncrossed Lines, which is LCS wearing a hat
# --------------------------------------------------------------------------
def max_uncrossed_lines(a, b):
    """Non-crossing pairs of equal numbers = a common subsequence.  The
    picture is different; the recurrence is character for character the LCS
    one."""
    return lcs_rolling(a, b)


# --------------------------------------------------------------------------
# 8. the application: a diff, and ROUGE-L
# --------------------------------------------------------------------------
def diff(old, new):
    """A minimal line diff: everything outside the LCS is a delete or an add.

    This is what `git diff` is doing underneath (with a better algorithm -
    Myers - but the same definition of "unchanged").
    """
    dp = lcs_table(old, new)
    i, j, out = len(old), len(new), []
    while i or j:
        if i and j and old[i - 1] == new[j - 1]:
            out.append(('  ', old[i - 1])); i -= 1; j -= 1
        elif j and (i == 0 or dp[i][j - 1] >= dp[i - 1][j]):
            out.append(('+ ', new[j - 1])); j -= 1
        else:
            out.append(('- ', old[i - 1])); i -= 1
    out.reverse()
    return out


def rouge_l(candidate, reference):
    """The summarisation metric: F-measure over the LCS length.

    Every time a model summary is scored with ROUGE-L, this table is what runs.
    """
    c, r = candidate.split(), reference.split()
    l = lcs_rolling(c, r)
    if not l:
        return 0.0
    prec, rec = l / len(c), l / len(r)
    return 2 * prec * rec / (prec + rec)


# --------------------------------------------------------------------------
# 9. LeetCode 300 - the plain question, both ways
# --------------------------------------------------------------------------
def length_of_lis(nums):
    return lis_patience(nums)[0]


# --------------------------------------------------------------------------
# walkthrough
# --------------------------------------------------------------------------
def rule(title):
    print('\n' + '=' * 68)
    print(title)
    print('=' * 68)


def main():
    rule('1. LIS the slow way - O(n^2) DP')
    n, seq = lis_dp(A)
    print(f'  a        = {A}')
    print(f'  length   = {n}')
    print(f'  one LIS  = {seq}')
    assert n == 4

    rule('2. patience sorting - O(n log n)')
    ln, tails = lis_patience(A)
    print(f'  tails    = {tails}   (length {ln})')
    print('  piles    :')
    for k, pile in enumerate(patience_piles(A)):
        print(f'    pile {k}: {pile}')
    trap = [2, 6, 8, 3, 4, 5, 1]
    t = lis_patience(trap)[1]
    print('\n  NOTE: the length is always right, the array itself usually is not.')
    print(f'        a     = {trap}')
    print(f'        tails = {t}   <- 1 is the *last* element of a, so this is not')
    print(f'        even a subsequence.  The real answer is '
          f'{lis_patience_reconstruct(trap)[1]}.')
    assert ln == 4

    rule('3. reconstructing the real subsequence')
    ln, real, tails = lis_patience_reconstruct(A)
    print(f'  tails    = {tails}')
    print(f'  real LIS = {real}')
    assert len(real) == ln == 4
    assert all(real[i] < real[i + 1] for i in range(len(real) - 1))

    rule('4. strict vs non-decreasing - one function name')
    dup = [1, 3, 3, 3, 5]
    print(f'  a                        = {dup}')
    print(f'  bisect_left  (strict)    = {lis_patience(dup, strict=True)[0]}')
    print(f'  bisect_right (allows =)  = {lis_patience(dup, strict=False)[0]}')
    assert lis_patience(dup, True)[0] == 3 and lis_patience(dup, False)[0] == 5

    rule('5. LeetCode 354 - Russian Doll Envelopes')
    env = [[5, 4], [6, 4], [6, 7], [2, 3]]
    print(f'  envelopes                 = {env}')
    print(f'  sorted (w asc, h desc)    = {sorted(env, key=lambda e: (e[0], -e[1]))}')
    print(f'  answer                    = {max_envelopes(env)}')
    same = [[1, 1], [1, 2], [1, 3]]
    print(f'  all width 1               = {same} -> {max_envelopes(same)}  (correct: cannot nest)')
    assert max_envelopes(env) == 3 and max_envelopes(same) == 1

    rule('6. LeetCode 673 - counting the longest ones')
    for arr, want in ([1, 3, 5, 4, 7], 2), ([2, 2, 2, 2, 2], 5):
        print(f'  {str(arr):18s} -> {count_lis(arr)}')
        assert count_lis(arr) == want

    rule('7. LCS - table and backtrack')
    x, y = 'AGGTAB', 'GXTXAYB'
    dp = lcs_table(x, y)
    print(f'  x = {x}   y = {y}')
    print('      ' + ' '.join(f'{c:>3s}' for c in '-' + y))
    for i, row in enumerate(dp):
        head = '-' if i == 0 else x[i - 1]
        print(f'   {head}  ' + ' '.join(f'{v:3d}' for v in row))
    print(f'\n  length   = {dp[-1][-1]}')
    print(f'  LCS      = {lcs_backtrack(x, y, dp)}')
    print(f'  rolling  = {lcs_rolling(x, y)}  (same number, no path)')
    assert lcs_backtrack(x, y, dp) == 'GTAB' and lcs_rolling(x, y) == 4

    rule('8. LCS of two permutations = LIS (Hunt-Szymanski)')
    p = [1, 2, 3, 4, 5, 6]
    q = [2, 4, 1, 5, 6, 3]
    ln, sub = lcs_of_permutations(p, q)
    print(f'  p = {p}')
    print(f'  q = {q}')
    print(f'  positions of q in p      = {[p.index(v) for v in q]}')
    print(f'  LCS via LIS              = {sub} (length {ln})')
    assert ln == lcs_rolling(p, q)

    random.seed(35)
    sz = 1200
    pp = list(range(sz)); random.shuffle(pp)
    qq = list(range(sz)); random.shuffle(qq)
    t0 = time.perf_counter(); slow = lcs_rolling(pp, qq); t1 = time.perf_counter()
    fast = lcs_of_permutations(pp, qq)[0]; t2 = time.perf_counter()
    print(f'\n  n = {sz} random permutations')
    print(f'    O(n*m) table   : {slow}  in {t1 - t0:.3f}s')
    print(f'    O(n log n) LIS : {fast}  in {t2 - t1:.4f}s   ({(t1 - t0) / (t2 - t1):.0f}x faster)')
    assert slow == fast

    rule('9. LeetCode 1035 - Uncrossed Lines')
    a1, b1 = [1, 4, 2], [1, 2, 4]
    print(f'  A = {a1}  B = {b1}  -> {max_uncrossed_lines(a1, b1)}')
    assert max_uncrossed_lines(a1, b1) == 2

    rule('10. what it is actually for: diff and ROUGE-L')
    old = ['def f(x):', '    y = x + 1', '    return y']
    new = ['def f(x):', '    # bump', '    y = x + 2', '    return y']
    for tag, line in diff(old, new):
        print('   ' + tag + line)
    score = rouge_l('the cat sat on the mat', 'the cat was sitting on the mat')
    print(f'\n  ROUGE-L("the cat sat on the mat", "the cat was sitting on the mat") = {score:.3f}')
    assert 0.7 < score < 0.9

    rule('11. LeetCode 300 - the plain question')
    print(f'  lengthOfLIS({A}) = {length_of_lis(A)}')
    assert length_of_lis(A) == 4 and length_of_lis([7, 7, 7, 7]) == 1

    print('\nall checks passed.')


if __name__ == '__main__':
    main()
