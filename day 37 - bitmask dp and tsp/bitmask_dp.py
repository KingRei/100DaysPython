"""Day 37 - state-compression (bitmask) DP, and Held-Karp for the TSP.

Run it:  python bitmask_dp.py

The one idea:  a *set* of at most ~20 things fits in one machine integer, so a
subset can be a DP index.  Everything else in this file is a consequence of
that, including the ways it silently goes wrong.
"""

import itertools
import math
import time
from functools import lru_cache

INF = float('inf')

# ---------------------------------------------------------------------------
# 1. The vocabulary: a set as an integer
# ---------------------------------------------------------------------------


def has(mask, i):
    """Is element i in the set?"""
    return (mask >> i) & 1 == 1


def add(mask, i):
    """The set with i added (masks are immutable ints - this returns a new one)."""
    return mask | (1 << i)


def remove(mask, i):
    """The set with i removed."""
    return mask & ~(1 << i)


def members(mask):
    """The elements of the set, ascending."""
    i = 0
    while mask:
        if mask & 1:
            yield i
        mask >>= 1
        i += 1


def popcount(mask):
    """How many elements.  int.bit_count() exists from Python 3.10."""
    return bin(mask).count('1')


def subsets_of(mask):
    """Every subset of mask, including mask itself and the empty set.

    The trick is `sub = (sub - 1) & mask`: subtracting one borrows through the
    low zeros, and the & puts back only the bits that belong to mask.  It has
    to be a do-while, because the empty set is a legal subset and the loop
    condition would skip it.
    """
    sub = mask
    while True:
        yield sub
        if sub == 0:
            break
        sub = (sub - 1) & mask


def subsets_of_buggy(mask):
    """The same loop written as a plain `while sub:` - drops the empty set."""
    sub = mask
    while sub:
        yield sub
        sub = (sub - 1) & mask


def fill_order_is_free(n=4):
    """Why `for mask in range(1 << n)` needs no sorting.

    Every transition of a subset DP adds elements, and adding a bit can only
    make the integer larger.  So numeric order on the masks is already a
    topological order on the subset lattice - the thing that had to be worked
    out by hand for interval DP comes for free here.
    """
    bad = [(mask, sub) for mask in range(1 << n)
           for sub in subsets_of(mask) if sub > mask]
    return len(bad)


# ---------------------------------------------------------------------------
# 2. The instance: eight stops on a round-the-island trip
# ---------------------------------------------------------------------------

CITIES = [
    ('Taipei', 157.6, 336.3),
    ('Hsinchu', 98.0, 312.0),
    ('Taichung', 68.7, 238.7),
    ('Sun Moon Lake', 90.9, 203.1),
    ('Tainan', 21.2, 109.9),
    ('Kaohsiung', 30.3, 69.9),
    ('Taitung', 115.1, 84.4),
    ('Hualien', 162.6, 220.9),
]

SHORT = ['TPE', 'HSZ', 'TXG', 'SML', 'TNN', 'KHH', 'TTT', 'HUN']


def distances(cities=CITIES):
    """Rounded straight-line distances, in km."""
    n = len(cities)
    return [[round(math.hypot(cities[i][1] - cities[j][1],
                              cities[i][2] - cities[j][2]))
             for j in range(n)] for i in range(n)]


DIST = distances()


def tour_cost(D, tour):
    """Cost of a closed tour given as a list of city indices."""
    return sum(D[tour[k]][tour[(k + 1) % len(tour)]] for k in range(len(tour)))


def names(tour, short=False):
    table = SHORT if short else [c[0] for c in CITIES]
    return [table[i] for i in tour]


# ---------------------------------------------------------------------------
# 3. Brute force - the baseline that is provably right and hopelessly slow
# ---------------------------------------------------------------------------


def tsp_brute(D):
    """Try every ordering of cities 1..n-1.  (n-1)! of them."""
    n = len(D)
    best, best_tour, tried = INF, None, 0
    for perm in itertools.permutations(range(1, n)):
        tried += 1
        cost = tour_cost(D, (0,) + perm)
        if cost < best:
            best, best_tour = cost, [0] + list(perm)
    return best, best_tour, tried


# ---------------------------------------------------------------------------
# 4. Held-Karp: the state is (set visited, where I am standing)
# ---------------------------------------------------------------------------


def held_karp(D):
    """Exact TSP in O(2^n * n^2) time and O(2^n * n) space.

    dp[mask][last] = cheapest way to start at city 0, visit exactly the cities
    in mask, and be standing on `last` right now.  Both halves matter: the set
    says what is already spent, `last` says what the next edge will cost.
    """
    n = len(D)
    size = 1 << n
    dp = [[INF] * n for _ in range(size)]
    parent = [[-1] * n for _ in range(size)]
    dp[1][0] = 0                                   # start at city 0, nothing else seen

    # Plain increasing integer order is already a legal fill order: every
    # subset of a mask is numerically smaller than the mask.
    for mask in range(size):
        if not has(mask, 0):
            continue
        for last in range(n):
            cur = dp[mask][last]
            if cur == INF or not has(mask, last):
                continue
            for nxt in range(n):
                if has(mask, nxt):
                    continue
                nmask = add(mask, nxt)
                cand = cur + D[last][nxt]
                if cand < dp[nmask][nxt]:
                    dp[nmask][nxt] = cand
                    parent[nmask][nxt] = last

    full = size - 1
    best, end = INF, -1
    for last in range(1, n):                       # close the loop back to 0
        cand = dp[full][last] + D[last][0]
        if cand < best:
            best, end = cand, last

    tour, mask, cur = [], full, end                # walk the parent pointers back
    while cur != -1:
        tour.append(cur)
        prev = parent[mask][cur]
        mask = remove(mask, cur)
        cur = prev
    tour.reverse()
    return best, tour, dp


def held_karp_memo(D):
    """The same recurrence written top-down, for the people who prefer it."""
    n = len(D)
    full = (1 << n) - 1

    @lru_cache(maxsize=None)
    def go(mask, last):
        if mask == full:
            return D[last][0]
        best = INF
        for nxt in range(n):
            if not has(mask, nxt):
                best = min(best, D[last][nxt] + go(add(mask, nxt), nxt))
        return best

    ans = go(1, 0)
    return ans, go.cache_info().currsize


# ---------------------------------------------------------------------------
# 5. The silent failure: dropping `last` from the state
# ---------------------------------------------------------------------------


def tsp_mask_only(D):
    """dp[mask] alone - the version that returns a number and not a tour.

    Without `last` there is no way to price the next edge, so the only thing
    that can be written is "connect the new city to whichever visited city is
    nearest".  That is a legal recurrence.  It just answers a different
    question: it builds a *tree*, not a tour.
    """
    n = len(D)
    dp = [INF] * (1 << n)
    dp[1] = 0
    for mask in range(1 << n):
        if dp[mask] == INF:
            continue
        for nxt in range(n):
            if has(mask, nxt):
                continue
            hop = min(D[i][nxt] for i in members(mask))
            nmask = add(mask, nxt)
            if dp[mask] + hop < dp[nmask]:
                dp[nmask] = dp[mask] + hop
    return dp[(1 << n) - 1]


def mst_weight(D):
    """Prim's algorithm, for comparison with tsp_mask_only."""
    n = len(D)
    seen = {0}
    total = 0
    while len(seen) < n:
        best = min(((D[i][j], j) for i in seen for j in range(n) if j not in seen))
        total += best[0]
        seen.add(best[1])
    return total


# ---------------------------------------------------------------------------
# 6. The other tempting shortcut: nearest neighbour
# ---------------------------------------------------------------------------


def nearest_neighbour(D, start=0):
    """Always drive to the closest place you have not been to yet."""
    n = len(D)
    cur, seen, tour, cost = start, {start}, [start], 0
    while len(seen) < n:
        nxt = min((j for j in range(n) if j not in seen), key=lambda j: (D[cur][j], j))
        cost += D[cur][nxt]
        cur = nxt
        seen.add(nxt)
        tour.append(nxt)
    return cost + D[cur][start], tour


# ---------------------------------------------------------------------------
# 7. How the two exponentials compare
# ---------------------------------------------------------------------------


def work_table(sizes=(5, 8, 10, 13, 15, 20)):
    """(n, (n-1)! permutations, 2^n * n^2 DP cells-times-transitions)."""
    return [(n, math.factorial(n - 1), (1 << n) * n * n) for n in sizes]


def same_cycle(a, b):
    """A tour and its mirror image cost the same; treat them as one answer."""
    n = len(a)
    rot = a.index(b[0])
    fwd = a[rot:] + a[:rot]
    bwd = [fwd[0]] + fwd[1:][::-1]
    return b in (fwd, bwd)


def ring(n, seed=7):
    """A deterministic n-city instance for timing, no random module needed."""
    cs = []
    for i in range(n):
        ang = 2 * math.pi * ((i * seed) % n) / n
        r = 100 + 37 * ((i * i * seed) % 11)
        cs.append((f'c{i}', r * math.cos(ang), r * math.sin(ang)))
    return distances(cs)


def crossover(brute_upto=10, hk_upto=15):
    """Measured milliseconds for both exact methods as n grows."""
    rows = []
    for n in range(8, hk_upto + 1):
        D = ring(n)
        _, hms = timed(lambda d: held_karp(d), D)
        bms = None
        if n <= brute_upto:
            _, bms = timed(lambda d: tsp_brute(d), D)
        rows.append((n, bms, hms))
    return rows


def timed(fn, D):
    t0 = time.perf_counter()
    out = fn(D)
    return out, (time.perf_counter() - t0) * 1000


def subset_pair_count(n):
    """sum over masks of 2^popcount(mask) - the cost of 'for every subset of

    every mask'.  Each element is in sub, in mask-but-not-sub, or outside:
    three choices, so the total is exactly 3^n, not 4^n.
    """
    return sum(1 << popcount(m) for m in range(1 << n))


# ---------------------------------------------------------------------------
# 8. LeetCode 847 - Shortest Path Visiting All Nodes
# ---------------------------------------------------------------------------


def shortest_path_length(graph):
    """Fewest edges in a walk that touches every node.  Revisits allowed.

    Because every edge costs the same 1, this is BFS, not DP - but over the
    same (mask, node) state space.  Revisits are free to allow: the state
    already records them as 'same mask, different node'.
    """
    from collections import deque
    n = len(graph)
    full = (1 << n) - 1
    if n == 1:
        return 0
    seen = {(1 << i, i) for i in range(n)}
    q = deque(((1 << i, i, 0) for i in range(n)))
    while q:
        mask, node, d = q.popleft()
        for nb in graph[node]:
            nmask = mask | (1 << nb)
            if nmask == full:
                return d + 1
            if (nmask, nb) not in seen:
                seen.add((nmask, nb))
                q.append((nmask, nb, d + 1))
    return 0


# ---------------------------------------------------------------------------
# 9. LeetCode 943 - Find the Shortest Superstring
# ---------------------------------------------------------------------------


def overlap(a, b):
    """Length of the longest suffix of a that is also a prefix of b."""
    for k in range(min(len(a), len(b)), 0, -1):
        if a.endswith(b[:k]):
            return k
    return 0


def shortest_superstring(words):
    """Shortest string containing every word - an open TSP in disguise.

    Maximise total overlap instead of minimising distance; the state is the
    same (set of words used, which word is last).
    """
    n = len(words)
    ov = [[overlap(words[i], words[j]) if i != j else 0 for j in range(n)]
          for i in range(n)]
    dp = [[-1] * n for _ in range(1 << n)]
    parent = [[-1] * n for _ in range(1 << n)]
    for i in range(n):
        dp[1 << i][i] = 0
    for mask in range(1 << n):
        for last in range(n):
            if dp[mask][last] < 0 or not has(mask, last):
                continue
            for nxt in range(n):
                if has(mask, nxt):
                    continue
                nmask = add(mask, nxt)
                cand = dp[mask][last] + ov[last][nxt]
                if cand > dp[nmask][nxt]:
                    dp[nmask][nxt] = cand
                    parent[nmask][nxt] = last
    full = (1 << n) - 1
    end = max(range(n), key=lambda i: dp[full][i])
    order, mask, cur = [], full, end
    while cur != -1:
        order.append(cur)
        prev = parent[mask][cur]
        mask = remove(mask, cur)
        cur = prev
    order.reverse()
    out = words[order[0]]
    for k in range(1, len(order)):
        out += words[order[k]][ov[order[k - 1]][order[k]]:]
    return out, order, dp[full][end]


def shortest_superstring_fixed_start(words):
    """The habit carried over from TSP, applied where it does not hold.

    In a closed tour any city can be called the start, so seeding only
    dp[1][0] is free.  A superstring is an open path: the first word is a real
    decision.  Seeding one start still produces a valid superstring - just a
    longer one - so nothing complains.
    """
    n = len(words)
    ov = [[overlap(words[i], words[j]) if i != j else 0 for j in range(n)]
          for i in range(n)]
    dp = [[-1] * n for _ in range(1 << n)]
    parent = [[-1] * n for _ in range(1 << n)]
    dp[1][0] = 0                                  # <- only word 0 may go first
    for mask in range(1 << n):
        for last in range(n):
            if dp[mask][last] < 0 or not has(mask, last):
                continue
            for nxt in range(n):
                if has(mask, nxt):
                    continue
                nmask = add(mask, nxt)
                cand = dp[mask][last] + ov[last][nxt]
                if cand > dp[nmask][nxt]:
                    dp[nmask][nxt] = cand
                    parent[nmask][nxt] = last
    full = (1 << n) - 1
    end = max(range(n), key=lambda i: dp[full][i])
    order, mask, cur = [], full, end
    while cur != -1:
        order.append(cur)
        prev = parent[mask][cur]
        mask = remove(mask, cur)
        cur = prev
    order.reverse()
    out = words[order[0]]
    for k in range(1, len(order)):
        out += words[order[k]][ov[order[k - 1]][order[k]]:]
    return out, order


# ---------------------------------------------------------------------------
# 10. LeetCode 1125 - Smallest Sufficient Team
# ---------------------------------------------------------------------------


def smallest_sufficient_team(req_skills, people):
    """The mask indexes the *requirement*, not the items.

    dp[covered] = the smallest team achieving exactly that coverage, so the
    table has 2^(number of skills) entries and the people are the transitions.
    """
    idx = {s: i for i, s in enumerate(req_skills)}
    m = len(req_skills)
    pmask = [sum(1 << idx[s] for s in p if s in idx) for p in people]
    full = (1 << m) - 1
    dp = {0: []}
    for i, pm in enumerate(pmask):
        if pm == 0:
            continue
        for covered, team in list(dp.items()):
            nc = covered | pm
            if nc == covered:
                continue
            if nc not in dp or len(dp[nc]) > len(team) + 1:
                dp[nc] = team + [i]
    return dp[full]


# ---------------------------------------------------------------------------
# 11. LeetCode 526 - Beautiful Arrangement
# ---------------------------------------------------------------------------


def count_arrangement(n):
    """The mask already knows the position: pos = popcount(mask) + 1.

    So the state is one integer even though the problem is about permutations.
    """
    full = (1 << n) - 1
    dp = [0] * (1 << n)
    dp[0] = 1
    for mask in range(1 << n):
        if dp[mask] == 0:
            continue
        pos = popcount(mask) + 1
        for v in range(1, n + 1):
            if has(mask, v - 1):
                continue
            if v % pos == 0 or pos % v == 0:
                dp[add(mask, v - 1)] += dp[mask]
    return dp[full]


def count_arrangement_brute(n):
    return sum(1 for p in itertools.permutations(range(1, n + 1))
               if all(v % (i + 1) == 0 or (i + 1) % v == 0 for i, v in enumerate(p)))


# ---------------------------------------------------------------------------
# 12. LeetCode 698 - Partition to K Equal Sum Subsets
# ---------------------------------------------------------------------------


def can_partition_k_subsets(nums, k):
    """Again the mask carries the extra state for free.

    Once the chosen set is fixed, its sum is fixed, so sum(mask) % target tells
    you how full the bucket in progress is - no need to store a bucket index or
    a running total alongside the mask.
    """
    total = sum(nums)
    if total % k:
        return False
    target = total // k
    if max(nums) > target:
        return False
    n = len(nums)
    sums = [0] * (1 << n)
    for mask in range(1, 1 << n):
        low = mask & -mask                 # lowest set bit
        sums[mask] = sums[mask ^ low] + nums[low.bit_length() - 1]
    ok = [False] * (1 << n)
    ok[0] = True
    for mask in range(1 << n):
        if not ok[mask]:
            continue
        room = target - sums[mask] % target
        for i in range(n):
            if has(mask, i):
                continue
            if nums[i] <= room:
                ok[add(mask, i)] = True
    return ok[(1 << n) - 1]


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def rule(title):
    print()
    print('=' * 68)
    print(title)
    print('=' * 68)


def main():
    rule('1. a set is an integer')
    m = 0
    for i in (0, 2, 3):
        m = add(m, i)
    print(f'  add 0,2,3      -> {m:5d}  = 0b{m:08b}  members {list(members(m))}')
    print(f'  remove 2       -> {remove(m, 2):5d}  = 0b{remove(m, 2):08b}')
    print(f'  popcount       -> {popcount(m)}')
    print(f'  subsets of 0b1011: {[format(s, "04b") for s in subsets_of(0b1011)]}')
    print(f'  same loop as `while sub:` misses {len(list(subsets_of(0b1011))) - len(list(subsets_of_buggy(0b1011)))}'
          f' subset (the empty one)')

    rule('2. increasing integer order is already a legal fill order')
    print(f'  subsets that are numerically larger than their superset: {fill_order_is_free(8)}')
    print('  so `for mask in range(1 << n)` never reads an unfilled cell,')
    print('  no matter which transition the recurrence uses.')
    print(f'  every subset of every mask, n=10: {subset_pair_count(10):,}'
          f'  = 3^10 = {3 ** 10:,}  (not 4^10 = {4 ** 10:,}),')
    print('  because each element is in sub, in mask but not sub, or outside.')

    rule('3. eight stops, one closed tour')
    for i, (nm, x, y) in enumerate(CITIES):
        print(f'  {i}  {SHORT[i]}  {nm:<14} ({x:6.1f}, {y:6.1f})')
    print('\n  distance matrix (km)')
    print('        ' + ' '.join(f'{s:>5}' for s in SHORT))
    for i, row in enumerate(DIST):
        print(f'  {SHORT[i]:>5} ' + ' '.join(f'{v:5d}' for v in row))

    rule('4. brute force vs Held-Karp')
    (bcost, btour, tried), bms = timed(lambda D: tsp_brute(D), DIST)
    (hcost, htour, _), hms = timed(lambda D: held_karp(D), DIST)
    print(f'  brute force  {bcost} km  after {tried} permutations   {bms:8.2f} ms')
    print(f'               {" -> ".join(names(btour, short=True))}')
    print(f'  Held-Karp    {hcost} km                         {hms:8.2f} ms')
    print(f'               {" -> ".join(names(htour, short=True))}')
    memo_cost, memo_states = held_karp_memo(DIST)
    print(f'  top-down     {memo_cost} km  from {memo_states} memoised states')
    print(f'  agree: {bcost == hcost == memo_cost}')

    rule('5. the silent failure: dropping `last` from the state')
    wrong = tsp_mask_only(DIST)
    print(f'  dp[mask] only        {wrong} km   <- smaller than the true optimum')
    print(f'  minimum spanning tree {mst_weight(DIST)} km   <- the same number')
    print('  the recurrence is not broken, it answers a different question:')
    print('  it grows a tree, and a tree is not a tour (nobody ever comes home).')
    print(f'  true optimum         {hcost} km')

    rule('6. nearest neighbour: right shape, wrong number')
    ncost, ntour = nearest_neighbour(DIST)
    print(f'  greedy   {ncost} km  {" -> ".join(names(ntour, short=True))}')
    print(f'  exact    {hcost} km  {" -> ".join(names(htour, short=True))}')
    print(f'  {ncost - hcost} km worse, {100 * (ncost - hcost) / hcost:.1f}% over,'
          f' and it looks like a perfectly good itinerary')

    rule('7. two exponentials')
    print('     n                    (n-1)!        2^n * n^2            ratio')
    for n, f, d in work_table():
        print(f'  {n:4d}  {f:24,d}   {d:14,d}   {f / d:14.1f}x')
    print('\n  measured, milliseconds')
    print('     n     brute force     Held-Karp')
    for n, bms, hms in crossover():
        bs = f'{bms:12.1f}' if bms is not None else '           -'
        print(f'  {n:4d}  {bs}  {hms:12.1f}')

    rule('8. LeetCode 847 - shortest path visiting all nodes')
    g = [[1, 2, 3], [0], [0], [0]]
    print(f'  graph {g}  -> {shortest_path_length(g)} edges (the hub is walked through twice)')
    g2 = [[1], [0, 2, 4], [1, 3, 4], [2], [1, 2]]
    print(f'  graph {g2}  -> {shortest_path_length(g2)} edges')

    rule('9. LeetCode 943 - shortest superstring')
    words = ['catg', 'ctaagt', 'gcta', 'ttca', 'atgcatc']
    sup, order, gained = shortest_superstring(words)
    print(f'  words     {words}')
    print(f'  order     {[words[i] for i in order]}')
    print(f'  overlap saved {gained} characters:'
          f' {sum(len(w) for w in words)} -> {len(sup)}')
    print(f'  answer    {sup}')
    bad, bad_order = shortest_superstring_fixed_start(words)
    print(f'  seeding only dp[1][0] - the habit borrowed from the tour, where any')
    print(f'  city may be called the start - gives a real superstring, {len(bad)} long:')
    print(f'            {" ".join(words[i] for i in bad_order)}')
    print(f'            {bad}')

    rule('10. LeetCode 1125 - smallest sufficient team')
    skills = ['java', 'nodejs', 'reactjs']
    people = [['java'], ['nodejs'], ['nodejs', 'reactjs']]
    team = smallest_sufficient_team(skills, people)
    print(f'  skills {skills}')
    print(f'  people {people}')
    print(f'  team   {team}  -> {[people[i] for i in team]}')

    rule('11. LeetCode 526 - beautiful arrangement')
    for n in (2, 3, 4, 8):
        print(f'  n = {n:2d}  ->  {count_arrangement(n):5d} arrangements'
              f'   (brute force agrees: {count_arrangement(n) == count_arrangement_brute(n)})')

    rule('12. LeetCode 698 - partition to k equal sum subsets')
    for nums, k in (([4, 3, 2, 3, 5, 2, 1], 4), ([1, 2, 3, 4], 3), ([2, 2, 2, 2, 3, 4, 5], 4)):
        print(f'  {str(nums):<26} k={k}  ->  {can_partition_k_subsets(nums, k)}')

    rule('checks')
    assert bcost == hcost == memo_cost == 689
    assert same_cycle(btour, htour)
    assert wrong == mst_weight(DIST) < hcost
    assert ncost == 795 and hcost == 689
    assert shortest_path_length(g) == 4 and shortest_path_length(g2) == 4
    assert len(sup) == 16 and all(w in sup for w in words)
    assert len(bad) == 18 and all(w in bad for w in words)
    assert sorted(team) == [0, 2]
    assert count_arrangement(8) == count_arrangement_brute(8) == 132
    assert can_partition_k_subsets([4, 3, 2, 3, 5, 2, 1], 4)
    assert not can_partition_k_subsets([1, 2, 3, 4], 3)
    assert subset_pair_count(10) == 3 ** 10
    print('  all assertions passed')


if __name__ == '__main__':
    main()
