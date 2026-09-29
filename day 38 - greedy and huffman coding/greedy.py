"""Day 38 - Greedy algorithms: interval scheduling and Huffman coding.

Run me:  python3 greedy.py

A greedy algorithm commits to the locally best-looking choice and never
reconsiders.  Writing one is easy.  The whole difficulty is proving that the
key you sort by is the right one - because a wrong key still produces a
perfectly legal answer, just a worse one, and nothing in the program will
tell you.
"""

from __future__ import annotations

import heapq
from bisect import bisect_right
from collections import Counter
from itertools import combinations

# ---------------------------------------------------------------- utilities

def rule(title: str) -> None:
    print()
    print('=' * 72)
    print(title)
    print('=' * 72)


T0 = 9 * 60          # the day starts at 09:00
UNIT = 15            # one slot is 15 minutes


def clock(u: int) -> str:
    """Slot index -> wall clock string."""
    m = T0 + u * UNIT
    return f'{m // 60:02d}:{m % 60:02d}'


def span(iv: tuple[int, int]) -> str:
    return f'{clock(iv[0])}-{clock(iv[1])}'


# ------------------------------------------------------- the meeting requests
# Eight rooms requests for one meeting room.  (name, start slot, end slot)

MEETINGS: list[tuple[str, int, int]] = [
    ('A', 1, 9),      # 09:15-11:15   the long design review
    ('B', 6, 7),      # 10:30-10:45   standup
    ('C', 7, 8),      # 10:45-11:00   1:1
    ('D', 8, 12),     # 11:00-12:00   sprint planning
    ('E', 11, 13),    # 11:45-12:15   vendor call
    ('F', 12, 18),    # 12:00-13:30   lunch and learn
    ('G', 18, 23),    # 13:30-14:45   retro
    ('H', 18, 25),    # 13:30-15:15   customer demo
]

NAME = {(s, e): n for n, s, e in MEETINGS}
IV: list[tuple[int, int]] = [(s, e) for _, s, e in MEETINGS]


def names(chosen) -> str:
    return ' '.join(NAME[iv] for iv in chosen)


def compatible(a, b) -> bool:
    """Two half-open intervals do not clash if one ends before the other starts."""
    return a[1] <= b[0] or b[1] <= a[0]


def is_feasible(chosen) -> bool:
    return all(compatible(x, y) for x, y in combinations(chosen, 2))


# ------------------------------------------------------- the greedy, 3 keys

def select_by_end(intervals):
    """Interval scheduling done right: always take the one that frees the room first."""
    out, last_end = [], float('-inf')
    for s, e in sorted(intervals, key=lambda iv: iv[1]):
        if s >= last_end:            # no clash with what we already took
            out.append((s, e))
            last_end = e
        # else: skip it - it overlaps something we already committed to
    return out


def select_by_start(intervals):
    """Plausible and wrong: whoever asked for the earliest slot goes first."""
    out, last_end = [], float('-inf')
    for s, e in sorted(intervals, key=lambda iv: iv[0]):
        if s >= last_end:
            out.append((s, e))
            last_end = e
    return out


def select_by_duration(intervals):
    """Also plausible and also wrong: pack the shortest meetings first."""
    out = []
    for s, e in sorted(intervals, key=lambda iv: (iv[1] - iv[0], iv[1])):
        if all(compatible((s, e), c) for c in out):
            out.append((s, e))
    return sorted(out)


def select_bruteforce(intervals):
    """Ground truth: the largest clash-free subset, found by trying every subset."""
    best: list = []
    n = len(intervals)
    for mask in range(1 << n):
        pick = [intervals[i] for i in range(n) if mask >> i & 1]
        if len(pick) > len(best) and is_feasible(pick):
            best = pick
    return sorted(best)


def exchange_argument(intervals) -> tuple[tuple[int, int], int, int]:
    """The proof, run as an experiment.

    Claim: the interval that finishes first is in *some* optimal schedule.
    So we compare the best schedule overall with the best schedule that is
    forced to contain the earliest-finishing interval.  If the claim holds
    the two have the same size, and the greedy choice costs nothing.
    """
    first = min(intervals, key=lambda iv: iv[1])
    unrestricted = len(select_bruteforce(intervals))
    rest = [iv for iv in intervals if compatible(iv, first) and iv != first]
    forced = 1 + len(select_bruteforce(rest))
    return first, unrestricted, forced


# ------------------------------------------------- where greedy stops working

# Same meetings, but now each one is worth money and we want the richest
# schedule rather than the fullest one.
VALUES = {
    (1, 9): 8, (6, 7): 1, (7, 8): 1, (8, 12): 3,
    (11, 13): 2, (12, 18): 4, (18, 23): 3, (18, 25): 5,
}


def greedy_weighted(intervals, value):
    """Run the same by-end greedy and just add up what it happens to collect."""
    chosen = select_by_end(intervals)
    return sum(value[iv] for iv in chosen), chosen


def dp_weighted(intervals, value):
    """Weighted interval scheduling needs a DP table, not a sort key.

    Sort by end time, and for each interval binary-search the last interval
    that finishes before it starts (Day 23's bisect on Day 33's DP).
    """
    iv = sorted(intervals, key=lambda t: t[1])
    ends = [e for _, e in iv]
    best = [0] * (len(iv) + 1)
    take = [False] * len(iv)
    for i, (s, e) in enumerate(iv):
        j = bisect_right(ends, s, 0, i)      # last compatible interval, 0 if none
        with_i = best[j] + value[(s, e)]
        best[i + 1] = max(best[i], with_i)
        take[i] = with_i > best[i]
    # walk the decisions backwards to recover the schedule itself
    out, i = [], len(iv) - 1
    while i >= 0:
        s, e = iv[i]
        if take[i] and best[i + 1] != best[i]:
            out.append((s, e))
            i = bisect_right(ends, s, 0, i) - 1
        else:
            i -= 1
    return best[-1], sorted(out)


# ------------------------------------------------------------ Huffman coding

TEXT = 'greedy algorithms are easy to write and hard to prove'


def huffman(freqs: dict[str, int]):
    """Repeatedly merge the two rarest symbols.  Returns {symbol: codeword}.

    The heap entry is (weight, tie, node); `tie` is a counter that keeps the
    ordering deterministic when two weights are equal, so the output does not
    depend on how Python happens to order tuples of lists.
    """
    heap = [(w, i, sym) for i, (sym, w) in enumerate(sorted(freqs.items()))]
    heapq.heapify(heap)
    tie = len(heap)
    if len(heap) == 1:                    # a one-symbol alphabet still needs one bit
        return {heap[0][2]: '0'}, [(heap[0][2], heap[0][0])]
    merges = []
    while len(heap) > 1:
        w1, _, n1 = heapq.heappop(heap)
        w2, _, n2 = heapq.heappop(heap)
        merges.append((n1, n2, w1 + w2))
        heapq.heappush(heap, (w1 + w2, tie, (n1, n2)))
        tie += 1
    root = heap[0][2]
    codes: dict[str, str] = {}

    def walk(node, prefix):
        if isinstance(node, str):
            codes[node] = prefix or '0'
            return
        walk(node[0], prefix + '0')
        walk(node[1], prefix + '1')

    walk(root, '')
    return codes, merges


def huffman_no_heap(freqs: dict[str, int]):
    """The bug: sort once by frequency and merge left to right.

    Every merge still joins two subtrees, so the result is still a legal
    prefix code and still decodes perfectly.  What it loses is the *re-sort*:
    a merged node whose weight has grown past its neighbours should sink back
    down the queue, and here it never does.
    """
    items = sorted(freqs.items(), key=lambda kv: (kv[1], kv[0]))
    node, weight = items[0][0], items[0][1]
    for sym, w in items[1:]:
        node, weight = ((node, sym), weight + w)
    codes: dict[str, str] = {}

    def walk(n, prefix):
        if isinstance(n, str):
            codes[n] = prefix or '0'
            return
        walk(n[0], prefix + '0')
        walk(n[1], prefix + '1')

    walk(node, '')
    return codes


def encoded_bits(freqs: dict[str, int], codes: dict[str, str]) -> int:
    return sum(freqs[s] * len(codes[s]) for s in freqs)


def fixed_width_bits(freqs: dict[str, int]) -> tuple[int, int]:
    """What a fixed-length code costs: ceil(log2(alphabet)) bits per symbol."""
    n = len(freqs)
    width = max(1, (n - 1).bit_length())
    return width, width * sum(freqs.values())


def encode(text: str, codes: dict[str, str]) -> str:
    return ''.join(codes[c] for c in text)


def decode(bits: str, codes: dict[str, str]) -> str:
    """Decoding needs no separators - that is what 'prefix code' buys."""
    back = {v: k for k, v in codes.items()}
    out, cur = [], ''
    for b in bits:
        cur += b
        if cur in back:
            out.append(back[cur])
            cur = ''
    assert cur == '', 'trailing bits: this was not a prefix code'
    return ''.join(out)


def is_prefix_free(codes: dict[str, str]) -> bool:
    words = sorted(codes.values())
    return all(not b.startswith(a) for a, b in zip(words, words[1:]))


def kraft_sum(codes: dict[str, str]) -> float:
    """Sum of 2^-len.  Exactly 1.0 means the code tree has no wasted branch."""
    return sum(2.0 ** -len(c) for c in codes.values())


# ------------------------------------------------ the coin counterexample

COINS = [1, 3, 4]


def coin_greedy(coins, amount):
    left, used = amount, []
    for c in sorted(coins, reverse=True):
        while left >= c:
            left -= c
            used.append(c)
    return (used if left == 0 else None)


def coin_dp(coins, amount):
    best = [0] + [None] * amount
    for a in range(1, amount + 1):
        for c in coins:
            if c <= a and best[a - c] is not None:
                cand = best[a - c] + 1
                if best[a] is None or cand < best[a]:
                    best[a] = cand
    return best[amount]


# ------------------------------------------------------------------ LeetCode
# These are the exact bodies that appear in the article.  On LeetCode the
# class is called `Solution`; here each one is suffixed with its problem
# number so that all five can live in one file and be tested together.

from typing import List


class Solution435:
    """LeetCode 435 - Non-overlapping Intervals."""

    def eraseOverlapIntervals(self, intervals: List[List[int]]) -> int:
        intervals.sort(key=lambda iv: iv[1])        # earliest finishing first
        kept, last_end = 0, float('-inf')
        for s, e in intervals:
            if s >= last_end:                       # fits after what we kept
                kept += 1
                last_end = e
        return len(intervals) - kept


class Solution452:
    """LeetCode 452 - Minimum Number of Arrows to Burst Balloons."""

    def findMinArrowShots(self, points: List[List[int]]) -> int:
        points.sort(key=lambda p: p[1])
        arrows, last = 0, float('-inf')
        for s, e in points:
            if s > last:                            # note: > not >=, touching counts as hit
                arrows += 1
                last = e
        return arrows


class Solution621:
    """LeetCode 1834-style scheduling: Task Scheduler."""

    def leastInterval(self, tasks: List[str], n: int) -> int:
        counts = Counter(tasks)
        peak = max(counts.values())
        tied = sum(1 for c in counts.values() if c == peak)
        skeleton = (peak - 1) * (n + 1) + tied
        return max(skeleton, len(tasks))


class Solution1046:
    """LeetCode 1046 - Last Stone Weight."""

    def lastStoneWeight(self, stones: List[int]) -> int:
        heap = [-s for s in stones]                 # heapq is a min-heap, so negate
        heapq.heapify(heap)
        while len(heap) > 1:
            a = -heapq.heappop(heap)
            b = -heapq.heappop(heap)
            if a != b:
                heapq.heappush(heap, -(a - b))
        return -heap[0] if heap else 0


class Solution134:
    """LeetCode 134 - Gas Station."""

    def canCompleteCircuit(self, gas: List[int], cost: List[int]) -> int:
        if sum(gas) < sum(cost):
            return -1                               # not enough fuel in the whole loop
        start, tank = 0, 0
        for i in range(len(gas)):
            tank += gas[i] - cost[i]
            if tank < 0:                            # i is not reachable from start
                start, tank = i + 1, 0
        return start


def least_interval_simulate(tasks, n):
    """A slow but obvious simulation, used only to check the formula above."""
    counts = Counter(tasks)
    heap = [(-c, t) for t, c in counts.items()]
    heapq.heapify(heap)
    time, cooling = 0, []                           # cooling: (ready_time, -count, task)
    while heap or cooling:
        time += 1
        while cooling and cooling[0][0] <= time:
            _, c, t = heapq.heappop(cooling)
            heapq.heappush(heap, (c, t))
        if heap:
            c, t = heapq.heappop(heap)
            if c + 1 < 0:
                heapq.heappush(cooling, (time + n + 1, c + 1, t))
    return time


def can_complete_circuit_brute(gas, cost):
    """Try every start and drive the loop; O(n^2) ground truth for LC 134."""
    n = len(gas)
    for start in range(n):
        tank, ok = 0, True
        for k in range(n):
            i = (start + k) % n
            tank += gas[i] - cost[i]
            if tank < 0:
                ok = False
                break
        if ok:
            return start
    return -1


# ---------------------------------------------------------------------- main

def main() -> None:
    rule('1. eight people want the same meeting room')
    for n, s, e in MEETINGS:
        print(f'  {n}  {span((s, e))}  ({(e - s) * UNIT:>3d} min)')

    rule('2. the greedy: always take the meeting that frees the room first')
    picked = select_by_end(IV)
    for iv in picked:
        print(f'  take {NAME[iv]}  {span(iv)}')
    print(f'  -> {len(picked)} meetings: {names(picked)}')

    rule('3. two other sort keys that look just as reasonable')
    best = select_bruteforce(IV)
    by_s = select_by_start(IV)
    by_d = select_by_duration(IV)
    print(f'  by end time   {len(picked)} meetings   {names(picked)}')
    print(f'  by start time {len(by_s)} meetings   {names(by_s)}'
          f'   <- A alone blocks 09:15-11:15')
    print(f'  by duration   {len(by_d)} meetings   {names(by_d)}'
          f'   <- the two 15-minute meetings look cheap')
    print(f'  brute force   {len(best)} meetings   {names(best)}  (all 2^8 subsets)')
    print('  every one of the three is a legal, clash-free schedule.')
    print('  nothing raises, nothing warns - the wrong keys just seat fewer people.')
    assert is_feasible(by_s) and is_feasible(by_d)

    rule('4. why the end time is the right key (the exchange argument)')
    first, unrestricted, forced = exchange_argument(IV)
    print(f'  earliest finishing meeting: {NAME[first]} {span(first)}')
    print(f'  best schedule overall                     : {unrestricted}')
    print(f'  best schedule forced to contain {NAME[first]}        : {forced}')
    print('  the two are equal, so taking it first costs nothing;')
    print('  the same argument then applies to what is left.')

    rule('5. change the question slightly and greedy dies')
    g_val, g_pick = greedy_weighted(IV, VALUES)
    d_val, d_pick = dp_weighted(IV, VALUES)
    print('  now each meeting is worth money:')
    print('   ', '  '.join(f'{NAME[iv]}={VALUES[iv]}' for iv in IV))
    print(f'  greedy by end time : {g_val:>3d}  {names(g_pick)}')
    print(f'  DP (Day 33 + Day 23 bisect): {d_val:>3d}  {names(d_pick)}')
    print(f'  greedy leaves {d_val - g_val} on the table and still returns a real schedule.')

    rule('6. Huffman: the same idea, and here it is provably optimal')
    freqs = dict(Counter(TEXT))
    codes, merges = huffman(freqs)
    width, fixed = fixed_width_bits(freqs)
    bits = encoded_bits(freqs, codes)
    print(f'  text        : {TEXT!r}')
    print(f'  {len(TEXT)} characters, {len(freqs)} distinct symbols')
    print(f'  fixed-length: {width} bits each -> {fixed} bits')
    print(f'  Huffman     : {bits} bits  ({100 * (fixed - bits) / fixed:.1f}% smaller)')
    print('  first four merges (always the two rarest subtrees):')
    for n1, n2, w in merges[:4]:
        print(f'    {str(n1):>28}  +  {str(n2):<28} = {w}')
    common = sorted(freqs.items(), key=lambda kv: -kv[1])[:5]
    print('  the five most common symbols get the shortest codewords:')
    for sym, f in common:
        print(f'    {sym!r:>5} x{f:<3d} -> {codes[sym]}')

    rule('7. the silent failure: merging without re-sorting')
    naive = huffman_no_heap(freqs)
    nbits = encoded_bits(freqs, naive)
    print(f'  Huffman with a heap              : {bits} bits')
    print(f'  sorted once, merged left to right : {nbits} bits'
          f'  (+{nbits - bits}, {100 * (nbits - bits) / bits:.0f}% worse)')
    print(f'  plain 5-bit fixed-length code    : {fixed} bits'
          f'  <- the broken version is worse than no compression at all')
    print(f'  prefix-free? heap={is_prefix_free(codes)}  naive={is_prefix_free(naive)}')
    print(f'  round-trips?  heap={decode(encode(TEXT, codes), codes) == TEXT}'
          f'   naive={decode(encode(TEXT, naive), naive) == TEXT}')
    print(f'  Kraft sum     heap={kraft_sum(codes):.3f}   naive={kraft_sum(naive):.3f}')
    print('  both decode perfectly.  the broken one is just a bigger file.')
    longest = max(naive.values(), key=len)
    print(f'  the naive tree is a ladder: its longest codeword is {len(longest)} bits'
          f' vs {max(len(c) for c in codes.values())}')

    rule('8. the classic counterexample: coins 1, 3, 4')
    amount = 6
    g = coin_greedy(COINS, amount)
    print(f'  pay {amount}: greedy grabs the biggest coin first -> {g},'
          f' {len(g)} coins')
    print(f'            the best you can do is {coin_dp(COINS, amount)} coins (3 + 3)')
    print('  with 1/5/10/25 the same greedy is always right, which is exactly')
    print('  why nobody checks: the answer is a real way to pay, just not the fewest.')

    rule('9. LeetCode 435 - Non-overlapping Intervals')
    ex = [[1, 2], [2, 3], [3, 4], [1, 3]]
    print(f'  {ex} -> {Solution435().eraseOverlapIntervals([list(x) for x in ex])}')
    room = [[s, e] for _, s, e in MEETINGS]
    print(f'  the meeting room above -> remove'
          f' {Solution435().eraseOverlapIntervals(room)} of {len(MEETINGS)}')

    rule('10. LeetCode 452 - Minimum Number of Arrows')
    balloons = [[10, 16], [2, 8], [1, 6], [7, 12]]
    print(f'  {balloons} -> {Solution452().findMinArrowShots([list(b) for b in balloons])} arrows')
    touching = [[1, 2], [2, 3], [3, 4], [4, 5]]
    print(f'  {touching} -> {Solution452().findMinArrowShots([list(b) for b in touching])}'
          f' arrows (touching ends count as a hit: the test is > not >=)')

    rule('11. LeetCode 621 - Task Scheduler')
    for tasks, n in ((list('AAABBB'), 2), (list('AAABBB'), 0), (list('AAAAABCDEF'), 2)):
        f = Solution621().leastInterval(tasks, n)
        s = least_interval_simulate(tasks, n)
        print(f'  {"".join(tasks):<12} n={n} -> formula {f:>2d}, simulation {s:>2d}')

    rule('12. LeetCode 1046 and 134')
    print(f'  stones [2,7,4,1,8,1] -> {Solution1046().lastStoneWeight([2, 7, 4, 1, 8, 1])}')
    gas, cost = [1, 2, 3, 4, 5], [3, 4, 5, 1, 2]
    print(f'  gas {gas} cost {cost} -> start at'
          f' {Solution134().canCompleteCircuit(gas, cost)}'
          f' (brute force says {can_complete_circuit_brute(gas, cost)})')
    gas2, cost2 = [2, 3, 4], [3, 4, 3]
    print(f'  gas {gas2} cost {cost2} -> {Solution134().canCompleteCircuit(gas2, cost2)}'
          f' (brute force says {can_complete_circuit_brute(gas2, cost2)})')

    rule('13. assertions')
    assert len(select_by_end(IV)) == len(select_bruteforce(IV)) == 5
    assert len(select_by_start(IV)) == 3
    assert len(select_by_duration(IV)) == 4
    assert is_feasible(select_by_end(IV))
    assert exchange_argument(IV)[1] == exchange_argument(IV)[2]
    assert dp_weighted(IV, VALUES)[0] > greedy_weighted(IV, VALUES)[0]
    assert is_feasible(dp_weighted(IV, VALUES)[1])
    assert decode(encode(TEXT, codes), codes) == TEXT
    assert decode(encode(TEXT, naive), naive) == TEXT
    assert is_prefix_free(codes) and is_prefix_free(naive)
    assert abs(kraft_sum(codes) - 1.0) < 1e-9
    assert bits < fixed < nbits   # the broken tree is worse than not compressing
    assert len(coin_greedy(COINS, 6)) == 3 and coin_dp(COINS, 6) == 2
    assert Solution435().eraseOverlapIntervals([[1, 2], [2, 3], [3, 4], [1, 3]]) == 1
    assert Solution452().findMinArrowShots([[10, 16], [2, 8], [1, 6], [7, 12]]) == 2
    assert Solution452().findMinArrowShots([[1, 2], [3, 4], [5, 6], [7, 8]]) == 4
    assert Solution1046().lastStoneWeight([2, 7, 4, 1, 8, 1]) == 1
    assert Solution134().canCompleteCircuit([1, 2, 3, 4, 5], [3, 4, 5, 1, 2]) == 3
    assert Solution134().canCompleteCircuit([2, 3, 4], [3, 4, 3]) == -1
    for tasks, n in ((list('AAABBB'), 2), (list('AAABBB'), 0), (list('AAAAABCDEF'), 2),
                     (list('ABCDEFG'), 3), (list('AAABBBCCC'), 2)):
        assert Solution621().leastInterval(tasks, n) == least_interval_simulate(tasks, n)
    print('all assertions passed')


if __name__ == '__main__':
    main()
