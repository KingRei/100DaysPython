"""Day 20 - divide and conquer, twice: merge sort and quick sort.

Run me:  python quick_merge_sort.py

Both algorithms split the problem in half and recurse.  The difference is
*where the work happens*:

    merge sort   trivial split, all the work is in the merge      (bottom-up)
    quick sort   all the work is in the partition, trivial join   (top-down)

Everything else follows from that one sentence: merge sort is stable, has a
guaranteed O(n log n), needs O(n) scratch space, and is what you use when the
data does not fit in memory.  Quick sort sorts in place with a much smaller
constant, and pays for it with a worst case of O(n^2) that a hostile - or
merely sorted - input can trigger.

The last third asks the parallel question.  Splitting is exactly what a
parallel machine wants, but the naive parallel merge sort is disappointing:
the merges are sequential, so the *depth* stays O(n) no matter how many cores
you own.  Fixing that needs a merge that is itself divide and conquer.
"""

import random
from bisect import bisect_left
from heapq import merge as heap_merge

# ------------------------------------------------------------------ merge sort

def merge(left, right, stats=None):
    """Merge two sorted lists.  `<=` is what makes the whole sort stable."""
    out = []
    i = j = 0
    while i < len(left) and j < len(right):
        if stats is not None:
            stats['cmp'] += 1
        if left[i] <= right[j]:            # <= keeps equal elements in order
            out.append(left[i]); i += 1
        else:
            out.append(right[j]); j += 1
    out.extend(left[i:])
    out.extend(right[j:])
    return out


def merge_sort(a, stats=None, depth=0):
    if stats is not None:
        stats['depth'] = max(stats['depth'], depth)
    if len(a) <= 1:
        return list(a)
    mid = len(a) // 2
    left = merge_sort(a[:mid], stats, depth + 1)
    right = merge_sort(a[mid:], stats, depth + 1)
    return merge(left, right, stats)


def count_inversions(a):
    """Merge sort's free by-product: how far the input is from sorted.

    Every time an element of the right half is emitted, it jumps ahead of
    everything still left in the left half - and each of those is an inversion.
    """
    def go(a):
        if len(a) <= 1:
            return list(a), 0
        mid = len(a) // 2
        left, x = go(a[:mid])
        right, y = go(a[mid:])
        out, inv = [], x + y
        i = j = 0
        while i < len(left) and j < len(right):
            if left[i] <= right[j]:
                out.append(left[i]); i += 1
            else:
                inv += len(left) - i        # <- the whole trick
                out.append(right[j]); j += 1
        out.extend(left[i:]); out.extend(right[j:])
        return out, inv
    return go(a)[1]


# ------------------------------------------------------------------ quick sort

def quicksort_lomuto(a, stats=None, pivot='last', rng=None, lo=0, hi=None,
                     depth=0):
    """In-place quick sort, Lomuto partition, two-way.

    pivot='last'   textbook version - O(n^2) on already sorted input
    pivot='random' the one-line fix
    """
    if hi is None:
        hi = len(a) - 1
    if stats is not None:
        stats['depth'] = max(stats['depth'], depth)
    if lo >= hi:
        return a
    if pivot == 'random':
        k = rng.randint(lo, hi)
        a[k], a[hi] = a[hi], a[k]
    p = a[hi]
    i = lo
    for j in range(lo, hi):
        if stats is not None:
            stats['cmp'] += 1
        if a[j] < p:
            a[i], a[j] = a[j], a[i]
            i += 1
    a[i], a[hi] = a[hi], a[i]
    quicksort_lomuto(a, stats, pivot, rng, lo, i - 1, depth + 1)
    quicksort_lomuto(a, stats, pivot, rng, i + 1, hi, depth + 1)
    return a


def three_way_partition(a, lo, hi, p, stats=None):
    """Dutch national flag: everything equal to the pivot lands in the middle
    and is never looked at again.  This is the fix for duplicate-heavy data."""
    lt, i, gt = lo, lo, hi
    while i <= gt:
        if stats is not None:
            stats['cmp'] += 1
        if a[i] < p:
            a[lt], a[i] = a[i], a[lt]; lt += 1; i += 1
        elif a[i] > p:
            a[i], a[gt] = a[gt], a[i]; gt -= 1
        else:
            i += 1
    return lt, gt


def quicksort_3way(a, stats=None, rng=None, lo=0, hi=None, depth=0):
    if hi is None:
        hi = len(a) - 1
    if stats is not None:
        stats['depth'] = max(stats['depth'], depth)
    if lo >= hi:
        return a
    k = rng.randint(lo, hi) if rng else (lo + hi) // 2
    lt, gt = three_way_partition(a, lo, hi, a[k], stats)
    quicksort_3way(a, stats, rng, lo, lt - 1, depth + 1)
    quicksort_3way(a, stats, rng, gt + 1, hi, depth + 1)
    return a


def new_stats():
    return {'cmp': 0, 'depth': 0}


# ------------------------------------------------------- external (k-way) sort

def external_sort(data, chunk):
    """Sort more data than fits in memory.

    Read `chunk` items at a time, sort that much in RAM, spill it to a run,
    then merge all the runs with a heap - which only ever holds one element
    per run.  Peak memory is chunk + number of runs, not n.
    """
    runs = [sorted(data[i:i + chunk]) for i in range(0, len(data), chunk)]
    peak = max(chunk, len(runs))
    return list(heap_merge(*runs)), len(runs), peak


# ------------------------------------------------- work and depth, measured

def merge_seq(a, b):
    """The ordinary merge: n steps, and every one of them waits for the last."""
    out = merge(a, b)
    return out, len(a) + len(b), len(a) + len(b)


def merge_par(a, b):
    """Divide-and-conquer merge.

    Take the median of the longer list, binary-search for its place in the
    other, and the two sides of that split can be merged independently.  The
    work barely changes; the depth collapses from n to log^2 n.
    """
    if len(a) < len(b):
        a, b = b, a
    if not a:
        return list(b), max(len(b), 1), 1
    if not b:
        return list(a), len(a), 1
    if len(a) == 1:
        return merge(a, b), 2, 1
    mid = len(a) // 2
    piv = a[mid]
    cost = max(1, len(b).bit_length())          # the binary search
    j = bisect_left(b, piv)
    left, wl, dl = merge_par(a[:mid], b[:j])
    right, wr, dr = merge_par(a[mid + 1:], b[j:])
    return left + [piv] + right, wl + wr + cost + 1, max(dl, dr) + cost


WORKDEPTH_N = 1024


def workdepth_input(m=WORKDEPTH_N):
    """A fixed permutation of 0..m-1.

    Work and depth depend on the actual values, so the script, the figure and
    the demo page all measure this same array - otherwise the three artefacts
    would quote three slightly different numbers for the same claim.
    """
    return [(i * 37) % m for i in range(m)]


def msort_cost(a, merger, parallel_recursion):
    """Run merge sort while accumulating (work, depth).

    Sequential composition adds depths; parallel composition takes the max.
    That single choice is the whole difference between the three rows of the
    table this prints.
    """
    if len(a) <= 1:
        return list(a), 0, 0
    mid = len(a) // 2
    l, wl, dl = msort_cost(a[:mid], merger, parallel_recursion)
    r, wr, dr = msort_cost(a[mid:], merger, parallel_recursion)
    out, wm, dm = merger(l, r)
    below = max(dl, dr) if parallel_recursion else dl + dr
    return out, wl + wr + wm, below + dm


# --------------------------------------------------------------- sample sort

def sample_sort(data, p, oversample, rng):
    """How a sort is done across p machines.

    Pick p*oversample samples, sort them, take every oversample-th as a
    splitter, and every machine keeps the bucket it owns.  One all-to-all
    exchange, then each machine sorts locally.  The only thing that can go
    wrong is load imbalance - which is exactly what oversampling buys down.
    """
    s = rng.sample(data, min(len(data), p * oversample))
    s.sort()
    splitters = [s[i * oversample] for i in range(1, p)]
    buckets = [[] for _ in range(p)]
    for x in data:
        buckets[bisect_left(splitters, x)].append(x)
    out = []
    for b in buckets:
        out.extend(sorted(b))                    # each machine, in parallel
    sizes = [len(b) for b in buckets]
    return out, sizes, max(sizes) / (len(data) / p)


# ------------------------------------------------- LeetCode 912, Sort an Array

def sort_an_array(nums):
    """LC 912 asks you to sort without the library sort - it is a quick sort
    test in disguise, and the test data contains long runs of equal values and
    sorted stretches, precisely the two inputs the textbook version dies on."""
    rng = random.Random(912)
    a = list(nums)

    def go(lo, hi):
        while lo < hi:
            if hi - lo < 16:                     # small runs: insertion sort
                for i in range(lo + 1, hi + 1):
                    v = a[i]; j = i - 1
                    while j >= lo and a[j] > v:
                        a[j + 1] = a[j]; j -= 1
                    a[j + 1] = v
                return
            k = rng.randint(lo, hi)
            lt, gt = three_way_partition(a, lo, hi, a[k])
            if lt - lo < hi - gt:                # recurse on the smaller side,
                go(lo, lt - 1)                   # loop on the larger one:
                lo = gt + 1                      # depth stays O(log n)
            else:
                go(gt + 1, hi)
                hi = lt - 1
    go(0, len(a) - 1)
    return a


# ----------------------------------------------------------------------- demo

def bar(n, width=44, scale=1.0):
    return '#' * max(1, int(round(n * width * scale)))


def main():
    rng = random.Random(20)
    data = [rng.randint(0, 999) for _ in range(64)]

    print('=' * 72)
    print('1.  merge sort - the work is in the merge')
    print('=' * 72)
    small = [5, 2, 9, 1, 5, 6]
    st = new_stats()
    print('   input        ', small)
    print('   merge_sort   ', merge_sort(small, st))
    print('   comparisons  ', st['cmp'], ' recursion depth', st['depth'])
    pairs = [(1, 'a'), (0, 'b'), (1, 'c'), (0, 'd')]
    keyed = merge_sort([(k, t) for k, t in pairs])
    print()
    print('   stability: sorting', pairs)
    print('   gives     ', keyed, '- a before c, b before d')
    print('   `<=` in the merge is the entire reason.  Change it to `<` and')
    print('   equal elements swap places.')
    print()
    perm = [2, 4, 1, 3, 5]
    print('   inversions in', perm, '=', count_inversions(perm),
          '(counted for free while sorting)')
    print('   inversions in the 64 random values =', count_inversions(data))
    print('   sorted input  ->', count_inversions(sorted(data)),
          ' reversed input ->', count_inversions(sorted(data, reverse=True)))

    print()
    print('=' * 72)
    print('2.  quick sort - the work is in the partition')
    print('=' * 72)
    demo = [7, 2, 9, 4, 7, 1, 7, 3]
    a = list(demo); st = new_stats()
    quicksort_lomuto(a, st, 'last')
    print('   input          ', demo)
    print('   lomuto         ', a, ' comparisons', st['cmp'],
          ' depth', st['depth'])
    a = list(demo); st = new_stats()
    quicksort_3way(a, st, random.Random(1))
    print('   three-way      ', a, ' comparisons', st['cmp'],
          ' depth', st['depth'])
    print()
    print('   three-way partition of', demo, 'around pivot 7:')
    b = list(demo)
    lt, gt = three_way_partition(b, 0, len(b) - 1, 7)
    print('      ', b[:lt], b[lt:gt + 1], b[gt + 1:],
          '   <7 | ==7 | >7')
    print('   the equal block is finished - neither recursive call touches it.')

    print()
    print('=' * 72)
    print('3.  the two inputs that break the textbook version')
    print('=' * 72)
    n = 200
    rows = []
    for name, arr in (('already sorted', list(range(n))),
                      ('all equal    ', [7] * n),
                      ('random       ', [rng.randint(0, 999) for _ in range(n)])):
        st = new_stats(); quicksort_lomuto(list(arr), st, 'last')
        st2 = new_stats(); quicksort_lomuto(list(arr), st2, 'random',
                                            random.Random(7))
        st3 = new_stats(); quicksort_3way(list(arr), st3, random.Random(7))
        st4 = new_stats(); merge_sort(list(arr), st4)
        rows.append((name, st, st2, st3, st4))
    print('   n = %d      lomuto/last   lomuto/random   three-way    merge sort'
          % n)
    for name, st, st2, st3, st4 in rows:
        print('   %s  cmp %7d %10d %12d %12d' %
              (name, st['cmp'], st2['cmp'], st3['cmp'], st4['cmp']))
        print('   %s  dep %7d %10d %12d %12d' %
              (' ' * len(name), st['depth'], st2['depth'], st3['depth'],
               st4['depth']))
    print()
    print('   sorted input + last-element pivot = one element per level:')
    print('   %d levels of recursion, and CPython gives up at 1000.' %
          rows[0][1]['depth'])
    print('   a random pivot fixes the sorted case; only the three-way split')
    print('   fixes the all-equal case, because it stops re-partitioning')
    print('   values it has already placed.')

    print()
    print('=' * 72)
    print('4.  external sort - when the data does not fit in memory')
    print('=' * 72)
    big = [rng.randint(0, 9999) for _ in range(1000)]
    out, runs, peak = external_sort(big, chunk=100)
    print('   %d values, memory for %d at a time' % (len(big), 100))
    print('   -> %d sorted runs, merged with a heap of %d entries' %
          (runs, runs))
    print('   peak items held in memory: %d  (%.0f%% of the data)' %
          (peak, 100 * peak / len(big)))
    print('   sorted correctly:', out == sorted(big))
    print('   quick sort cannot do this: it needs random access to everything.')

    print()
    print('=' * 72)
    print('5.  work vs depth - why the obvious parallel merge sort is not fast')
    print('=' * 72)
    m = WORKDEPTH_N
    arr = workdepth_input(m)
    confs = [('sequential                    ', merge_seq, False),
             ('parallel recursion, seq merge ', merge_seq, True),
             ('parallel recursion, par merge ', merge_par, True)]
    print('   n = %d' % m)
    print('   %-32s %10s %8s %12s' % ('', 'work', 'depth', 'parallelism'))
    costs = []
    for name, mg, par in confs:
        out, w, d = msort_cost(arr, mg, par)
        assert out == sorted(arr)
        costs.append((name, w, d))
        print('   %-32s %10d %8d %11.1fx' % (name, w, d, w / d))
    print()
    print('   The middle row is the trap: every core is busy at the bottom of')
    print('   the recursion and idle at the top, because the final merge of')
    print('   %d elements is one sequential loop.  Depth stays O(n).' % m)
    print('   The parallel merge splits on the median and binary-searches the')
    print('   other half, so depth drops from %d to %d for %.1f%% more work.' %
          (costs[1][2], costs[2][2],
           100 * (costs[2][1] - costs[1][1]) / costs[1][1]))

    print()
    print('=' * 72)
    print('6.  sample sort - the same idea across machines')
    print('=' * 72)
    prng = random.Random(99)
    payload = [prng.randint(0, 10 ** 6) for _ in range(20000)]
    print('   20000 values, p = 8.  Sampling is a lottery, so each row is')
    print('   200 different random samples of the same data:')
    print()
    print('   oversample   mean imbalance   worst seen')
    means = {}
    for over in (1, 4, 32):
        imbs = []
        for t in range(200):
            out, sizes, imb = sample_sort(payload, 8, over, random.Random(t))
            imbs.append(imb)
        assert out == sorted(payload)
        means[over] = sum(imbs) / len(imbs)
        print('   %10d %14.2fx %11.2fx' % (over, means[over], max(imbs)))
    print()
    _, s1, i1 = sample_sort(payload, 8, 1, random.Random(5))
    _, s32, i32 = sample_sort(payload, 8, 32, random.Random(5))
    print('   one run, oversample  1:      ', s1, '-> %.2fx' % i1)
    print('   the same run, oversample 32:  ', s32, '-> %.2fx' % i32)
    print()
    print('   the sort finishes when the *slowest* bucket finishes, so the')
    print('   mean column is the number that matters - and one splitter per')
    print('   machine leaves it %.0f%% above a fair share.'
          % (100 * (means[1] - 1)))

    print()
    print('=' * 72)
    print('7.  LeetCode 912 - Sort an Array')
    print('=' * 72)
    cases = [[5, 2, 3, 1], [5, 1, 1, 2, 0, 0], list(range(300, 0, -1)),
             [4] * 100]
    for c in cases:
        got = sort_an_array(c)
        tag = c if len(c) <= 8 else '%d values (%s)' % (
            len(c), 'reversed' if c[0] > c[-1] else 'all equal')
        print('   %-28s -> %s' % (tag, got if len(got) <= 8 else 'ok'))
        assert got == sorted(c)
    st = new_stats()
    quicksort_lomuto([7] * 400, st, 'last')
    print()
    print('   400 equal values, textbook quick sort: %d comparisons, depth %d'
          % (st['cmp'], st['depth']))
    st = new_stats()
    quicksort_3way([7] * 400, st, random.Random(3))
    print('   400 equal values, three-way          : %d comparisons, depth %d'
          % (st['cmp'], st['depth']))
    print('   the three-way version is the reason LC 912 accepts and the')
    print('   textbook one times out.')

    print()
    print('tests')
    print('-----')
    for trial in range(60):
        r = random.Random(trial)
        arr = [r.randint(0, 20) for _ in range(r.randint(0, 40))]
        exp = sorted(arr)
        assert merge_sort(arr) == exp
        assert quicksort_lomuto(list(arr), None, 'random', r) == exp
        assert quicksort_3way(list(arr), None, random.Random(trial)) == exp
        assert sort_an_array(arr) == exp
        assert external_sort(arr, 7)[0] == exp
        assert msort_cost(arr, merge_par, True)[0] == exp
        assert count_inversions(arr) == sum(
            1 for i in range(len(arr)) for j in range(i + 1, len(arr))
            if arr[i] > arr[j])
    assert merge_sort([(1, 'a'), (0, 'b'), (1, 'c'), (0, 'd')]) == \
        [(0, 'b'), (0, 'd'), (1, 'a'), (1, 'c')]
    print('all assertions passed')


if __name__ == '__main__':
    main()
