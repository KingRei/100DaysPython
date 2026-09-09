"""Day 29 - range structures: segment tree (with lazy propagation) and the
Fenwick tree / binary indexed tree.

Run me:  python range_structures.py

The question of the day is the one LeetCode 307 asks: keep an array, answer
"sum of a[l:r]" and "set a[i] = v", both fast.  Either naive answer is O(1)
for one operation and O(n) for the other, and the whole subject exists to
kill that asymmetry.
"""

import bisect
import random
import time


# =====================================================================
# 0.  The two naive answers, and why they are not enough
# =====================================================================

class PlainArray:
    """O(1) update, O(n) query."""

    def __init__(self, data):
        self.a = list(data)
        self.query_work = 0

    def update(self, i, value):
        self.a[i] = value

    def range_sum(self, l, r):                       # sum of a[l:r]
        self.query_work += r - l
        return sum(self.a[l:r])


class PrefixSum:
    """O(1) query, O(n) update - the mirror image."""

    def __init__(self, data):
        self.a = list(data)
        self.update_work = 0
        self._rebuild()

    def _rebuild(self):
        p = [0] * (len(self.a) + 1)
        for i, x in enumerate(self.a):
            p[i + 1] = p[i] + x
        self.p = p

    def update(self, i, value):
        self.a[i] = value
        self.update_work += len(self.a)              # every later prefix moves
        self._rebuild()

    def range_sum(self, l, r):
        return self.p[r] - self.p[l]


# =====================================================================
# 1.  Segment tree, iterative bottom-up form
# =====================================================================

class SegmentTree:
    """Point update, range query, over any associative combine().

    Layout: a flat array of size 2n.  Leaf i lives at t[n + i], and the
    parent of node k is k >> 1.  No recursion, no 4n allocation, and the
    tree is built in one backwards pass.
    """

    def __init__(self, data, combine=None, identity=0):
        self.n = len(data)
        self.combine = combine or (lambda a, b: a + b)
        self.identity = identity
        self.t = [identity] * (2 * self.n)
        self.t[self.n:] = list(data)                 # leaves
        for k in range(self.n - 1, 0, -1):           # internal nodes, bottom up
            self.t[k] = self.combine(self.t[2 * k], self.t[2 * k + 1])
        self.nodes_touched = 0

    def update(self, i, value):
        """Set a[i] = value and repair the O(log n) ancestors."""
        k = i + self.n
        self.t[k] = value
        k >>= 1
        while k:
            self.nodes_touched += 1
            self.t[k] = self.combine(self.t[2 * k], self.t[2 * k + 1])
            k >>= 1

    def query(self, l, r):
        """Combine over a[l:r] - half open, like a Python slice."""
        res_l, res_r = self.identity, self.identity
        l += self.n
        r += self.n
        while l < r:
            self.nodes_touched += 1
            if l & 1:                                # l is a right child: take it
                res_l = self.combine(res_l, self.t[l])
                l += 1
            if r & 1:                                # r is a right child: take r-1
                r -= 1
                res_r = self.combine(self.t[r], res_r)
            l >>= 1
            r >>= 1
        return self.combine(res_l, res_r)


def canonical_cover(n, l, r):
    """The exact set of tree nodes an iterative query [l, r) visits.

    This is the thing worth seeing: an arbitrary range is cut into at most
    2*ceil(log2(n)) whole subtrees, and never more.
    """
    out = []
    l += n
    r += n
    while l < r:
        if l & 1:
            out.append(l)
            l += 1
        if r & 1:
            r -= 1
            out.append(r)
        l >>= 1
        r >>= 1
    return sorted(out)


# =====================================================================
# 2.  Segment tree with lazy propagation: range update, range query
# =====================================================================

class LazySegmentTree:
    """Range add, range sum.  Recursive, 4n nodes, one pending value per node.

    The idea: when a whole node is covered by the update, do NOT descend.
    Apply the change to that node's aggregate and leave a note ("lazy") on
    it.  The note is pushed down only if someone later needs to look inside.
    """

    def __init__(self, data):
        self.n = len(data)
        self.sum = [0] * (4 * self.n)
        self.lazy = [0] * (4 * self.n)
        self.pushes = 0
        self.visits = 0
        if self.n:
            self._build(1, 0, self.n - 1, list(data))

    def _build(self, node, lo, hi, data):
        if lo == hi:
            self.sum[node] = data[lo]
            return
        mid = (lo + hi) // 2
        self._build(2 * node, lo, mid, data)
        self._build(2 * node + 1, mid + 1, hi, data)
        self.sum[node] = self.sum[2 * node] + self.sum[2 * node + 1]

    def _apply(self, node, lo, hi, delta):
        self.sum[node] += delta * (hi - lo + 1)      # +delta on each element
        self.lazy[node] += delta                     # note for the children

    def _push(self, node, lo, hi):
        if self.lazy[node]:
            mid = (lo + hi) // 2
            self._apply(2 * node, lo, mid, self.lazy[node])
            self._apply(2 * node + 1, mid + 1, hi, self.lazy[node])
            self.lazy[node] = 0
            self.pushes += 1

    def range_add(self, l, r, delta):                # inclusive [l, r]
        self._range_add(1, 0, self.n - 1, l, r, delta)

    def _range_add(self, node, lo, hi, l, r, delta):
        self.visits += 1
        if r < lo or hi < l:                         # disjoint
            return
        if l <= lo and hi <= r:                      # fully covered: stop here
            self._apply(node, lo, hi, delta)
            return
        self._push(node, lo, hi)
        mid = (lo + hi) // 2
        self._range_add(2 * node, lo, mid, l, r, delta)
        self._range_add(2 * node + 1, mid + 1, hi, l, r, delta)
        self.sum[node] = self.sum[2 * node] + self.sum[2 * node + 1]

    def range_sum(self, l, r):                       # inclusive [l, r]
        return self._range_sum(1, 0, self.n - 1, l, r)

    def _range_sum(self, node, lo, hi, l, r):
        self.visits += 1
        if r < lo or hi < l:
            return 0
        if l <= lo and hi <= r:
            return self.sum[node]
        self._push(node, lo, hi)
        mid = (lo + hi) // 2
        return (self._range_sum(2 * node, lo, mid, l, r)
                + self._range_sum(2 * node + 1, mid + 1, hi, l, r))


class EagerRangeAddTree:
    """The same range-add, but without lazy: descend all the way to the leaves.

    Kept only to measure how much lazy propagation actually saves.
    """

    def __init__(self, data):
        self.n = len(data)
        self.sum = [0] * (4 * self.n)
        self.visits = 0
        if self.n:
            self._build(1, 0, self.n - 1, list(data))

    def _build(self, node, lo, hi, data):
        if lo == hi:
            self.sum[node] = data[lo]
            return
        mid = (lo + hi) // 2
        self._build(2 * node, lo, mid, data)
        self._build(2 * node + 1, mid + 1, hi, data)
        self.sum[node] = self.sum[2 * node] + self.sum[2 * node + 1]

    def range_add(self, l, r, delta):
        self._add(1, 0, self.n - 1, l, r, delta)

    def _add(self, node, lo, hi, l, r, delta):
        self.visits += 1
        if r < lo or hi < l:
            return
        if lo == hi:
            self.sum[node] += delta
            return
        mid = (lo + hi) // 2
        self._add(2 * node, lo, mid, l, r, delta)
        self._add(2 * node + 1, mid + 1, hi, l, r, delta)
        self.sum[node] = self.sum[2 * node] + self.sum[2 * node + 1]

    def range_sum(self, l, r):
        return self._sum(1, 0, self.n - 1, l, r)

    def _sum(self, node, lo, hi, l, r):
        if r < lo or hi < l:
            return 0
        if l <= lo and hi <= r:
            return self.sum[node]
        mid = (lo + hi) // 2
        return (self._sum(2 * node, lo, mid, l, r)
                + self._sum(2 * node + 1, mid + 1, hi, l, r))


# =====================================================================
# 3.  Fenwick tree (binary indexed tree)
# =====================================================================
#
# Peter Fenwick, 1994, "A new data structure for cumulative frequency
# tables".  Same structure, two names: "Fenwick tree" and "binary indexed
# tree" (BIT) are the same thing.
#
# The whole design is one identity:
#
#     i & -i   ==   the lowest set bit of i
#
# Index i (1-based) stores the sum of the  i & -i  elements ending at i.
# So tree[12] (binary 1100, lowest bit 4) covers a[9..12].  Walking a
# prefix means peeling off one set bit at a time - and a number below n
# has at most log2(n) set bits, which is where the O(log n) comes from.
# There is no tree in memory: the shape is implied by the bits of the index.

class Fenwick:
    """1-based BIT.  Point add, prefix sum, and a binary-lifting search."""

    def __init__(self, n_or_data):
        if isinstance(n_or_data, int):
            self.n = n_or_data
            self.t = [0] * (self.n + 1)
        else:
            data = list(n_or_data)
            self.n = len(data)
            self.t = [0] + data                      # t[i] starts as a[i-1]
            # O(n) build: push each cell into its parent once.
            for i in range(1, self.n + 1):
                j = i + (i & -i)
                if j <= self.n:
                    self.t[j] += self.t[i]
        self.steps = 0

    def add(self, i, delta):
        """a[i] += delta, with i 0-based on the outside."""
        i += 1
        while i <= self.n:
            self.steps += 1
            self.t[i] += delta
            i += i & -i                              # go to the parent

    def prefix(self, i):
        """Sum of a[0:i] - i elements, 0-based half-open."""
        s = 0
        while i > 0:
            self.steps += 1
            s += self.t[i]
            i -= i & -i                              # drop the lowest set bit
        return s

    def range_sum(self, l, r):                       # a[l:r]
        return self.prefix(r) - self.prefix(l)

    def lower_bound(self, target):
        """Smallest 0-based i with prefix(i+1) >= target, for non-negative a.

        Binary lifting over the tree, in O(log n) - and it needs no extra
        memory, because the powers of two ARE the tree.  This is the trick
        that makes a BIT a usable ordered multiset over a small key space.
        """
        pos = 0
        bit = 1 << (self.n.bit_length())
        while bit:
            nxt = pos + bit
            if nxt <= self.n and self.t[nxt] < target:
                pos = nxt
                target -= self.t[nxt]
            bit >>= 1
        return pos                                   # 0-based index

    def covered_by(self, i):
        """Which elements t[i] holds - purely for the explanation."""
        return (i - (i & -i), i)                     # half-open, 0-based


class FenwickRangeUpdate:
    """Range add + point query, using one BIT over the difference array.

    add(l, r, v) becomes d[l] += v, d[r+1] -= v, and a[i] is prefix(d, i).
    """

    def __init__(self, n):
        self.n = n
        self.d = Fenwick(n + 1)

    def range_add(self, l, r, v):                    # inclusive [l, r]
        self.d.add(l, v)
        self.d.add(r + 1, -v)

    def point_query(self, i):
        return self.d.prefix(i + 1)


class FenwickRangeRange:
    """Range add AND range sum, with two BITs.

    prefix(i) of the updated array is  B1.prefix(i)*i - B2.prefix(i),
    which is the standard two-BIT trick.  Half the memory of a lazy
    segment tree and a much shorter inner loop, but it only works because
    addition has an inverse.
    """

    def __init__(self, n):
        self.n = n
        self.b1 = Fenwick(n + 2)
        self.b2 = Fenwick(n + 2)

    def range_add(self, l, r, v):                    # inclusive [l, r]
        self.b1.add(l, v)
        self.b1.add(r + 1, -v)
        self.b2.add(l, v * l)                    # v * (L-1) with L = l+1
        self.b2.add(r + 1, -v * (r + 1))         # -v * R  with R = r+1

    def _prefix(self, i):                            # sum of a[0:i]
        return self.b1.prefix(i) * i - self.b2.prefix(i)

    def range_sum(self, l, r):                       # inclusive [l, r]
        return self._prefix(r + 1) - self._prefix(l)


# =====================================================================
# 4.  What a Fenwick tree cannot do
# =====================================================================
#
# A BIT answers range(l, r) as prefix(r) - prefix(l).  That subtraction is
# the catch: the operation must have an inverse.  Sum and xor do.  min and
# max do not - once you have folded a value into a prefix minimum there is
# no way to remove it again.  A segment tree never subtracts, it only
# combines disjoint pieces, so it works for ANY associative operation.

class MinSegmentTree(SegmentTree):
    """The same class with a different combine - that is the entire change."""

    def __init__(self, data):
        super().__init__(data, combine=min, identity=float('inf'))


class BrokenMinBIT:
    """A prefix-min BIT.  It works right up until a value increases.

    Kept in the module deliberately: this is the bug people actually write,
    and it passes every test where values only ever go down.
    """

    def __init__(self, n):
        self.n = n
        self.t = [float('inf')] * (n + 1)

    def update(self, i, value):                      # "a[i] = value"
        i += 1
        while i <= self.n:
            self.t[i] = min(self.t[i], value)        # can only ever shrink
            i += i & -i

    def prefix_min(self, i):                         # min of a[0:i]
        m = float('inf')
        while i > 0:
            m = min(m, self.t[i])
            i -= i & -i
        return m


def min_bit_failure():
    """Show the failure concretely.  Returns (true, broken BIT, segment tree)."""
    a = [5, 3, 8, 1]
    bit = BrokenMinBIT(len(a))
    for i, x in enumerate(a):
        bit.update(i, x)
    a[3] = 9                                          # the 1 goes away
    bit.update(3, 9)
    seg = MinSegmentTree([5, 3, 8, 1])
    seg.update(3, 9)
    return min(a), bit.prefix_min(4), seg.query(0, 4)


# =====================================================================
# 5.  LeetCode
# =====================================================================

class NumArrayBIT:
    """LC 307 - Range Sum Query, Mutable.  The Fenwick answer."""

    def __init__(self, nums):
        self.a = list(nums)
        self.bit = Fenwick(self.a)                   # O(n) build

    def update(self, index, val):
        self.bit.add(index, val - self.a[index])     # BITs add, they do not set
        self.a[index] = val

    def sumRange(self, left, right):                 # inclusive
        return self.bit.range_sum(left, right + 1)


class NumArraySeg:
    """LC 307 - the segment tree answer.  Same complexity, more memory."""

    def __init__(self, nums):
        self.tree = SegmentTree(nums)

    def update(self, index, val):
        self.tree.update(index, val)

    def sumRange(self, left, right):
        return self.tree.query(left, right + 1)


def count_smaller(nums):
    """LC 315 - Count of Smaller Numbers After Self.

    Scan right to left; for each value ask "how many already-seen values are
    strictly smaller".  That is a prefix sum over a frequency array, and the
    frequency array is being updated as we go - exactly a BIT.  Coordinate
    compression first, so the key space is O(n) rather than 10^9.
    """
    ranks = {v: i for i, v in enumerate(sorted(set(nums)))}
    bit = Fenwick(len(ranks))
    out = [0] * len(nums)
    for i in range(len(nums) - 1, -1, -1):
        r = ranks[nums[i]]
        out[i] = bit.prefix(r)                       # strictly smaller ranks
        bit.add(r, 1)
    return out


def count_smaller_bruteforce(nums):
    return [sum(1 for x in nums[i + 1:] if x < nums[i]) for i in range(len(nums))]


def get_modified_array(length, updates):
    """LC 370 - Range Addition.  The punchline: you need neither structure.

    All the updates arrive before any query, so a difference array does the
    whole job in O(n + k) with no tree at all.  A segment tree here is a
    correct answer to a question nobody asked.
    """
    d = [0] * (length + 1)
    for l, r, v in updates:
        d[l] += v
        d[r + 1] -= v
    out, run = [], 0
    for i in range(length):
        run += d[i]
        out.append(run)
    return out


# =====================================================================
# 6.  Measurements
# =====================================================================

def popcount(x):
    return bin(x).count('1')


def query_cost_profile(n, trials=20000, seed=29):
    """Average and worst node count for a segment tree query on [0, n)."""
    rng = random.Random(seed)
    tot = worst = 0
    for _ in range(trials):
        l = rng.randrange(n)
        r = rng.randrange(l + 1, n + 1)
        c = len(canonical_cover(n, l, r))
        tot += c
        worst = max(worst, c)
    return tot / trials, worst


def fenwick_cost_profile(n):
    """Steps a BIT prefix(i) takes = number of set bits in i."""
    tot = sum(popcount(i) for i in range(1, n + 1))
    worst = max(popcount(i) for i in range(1, n + 1))
    return tot / n, worst


def lazy_vs_eager(n=100000, ops=2000, seed=29):
    rng = random.Random(seed)
    data = [0] * n
    lz = LazySegmentTree(data)
    eg = EagerRangeAddTree(data)
    for _ in range(ops):
        l = rng.randrange(n)
        r = rng.randrange(l, n)
        v = rng.randint(1, 9)
        lz.range_add(l, r, v)
        eg.range_add(l, r, v)
    assert lz.range_sum(0, n - 1) == eg.range_sum(0, n - 1)
    return lz.visits, eg.visits, lz.pushes


def timing(n=20000, ops=2000, seed=29):
    """Wall clock for a mixed update/query workload.  Ratios, not absolutes."""
    rng = random.Random(seed)
    data = [rng.randint(0, 99) for _ in range(n)]
    plan = []
    for _ in range(ops):
        i = rng.randrange(n)
        v = rng.randint(0, 99)
        l = rng.randrange(n)
        r = rng.randrange(l + 1, n + 1)
        plan.append((i, v, l, r))

    out = {}

    a = PlainArray(data)
    t0 = time.perf_counter()
    for i, v, l, r in plan:
        a.update(i, v)
        a.range_sum(l, r)
    out['plain array'] = time.perf_counter() - t0

    p = PrefixSum(data)
    t0 = time.perf_counter()
    for i, v, l, r in plan:
        p.update(i, v)
        p.range_sum(l, r)
    out['prefix sum'] = time.perf_counter() - t0

    s = SegmentTree(data)
    t0 = time.perf_counter()
    for i, v, l, r in plan:
        s.update(i, v)
        s.query(l, r)
    out['segment tree'] = time.perf_counter() - t0

    cur = list(data)
    f = Fenwick(data)
    t0 = time.perf_counter()
    for i, v, l, r in plan:
        f.add(i, v - cur[i])
        cur[i] = v
        f.range_sum(l, r)
    out['fenwick'] = time.perf_counter() - t0
    return out


def show(title):
    print()
    print('=' * 68)
    print(title)
    print('=' * 68)


def main():
    show('1. The two naive answers')
    n = 1000000
    print('  n = %d, and we want BOTH update and range-sum to be fast' % n)
    print('    plain array   update O(1)      range_sum O(n)   -> %d adds per query' % n)
    print('    prefix sums   update O(n)      range_sum O(1)   -> %d rewrites per update' % n)
    print('  each is perfect at one job and useless at the other; the whole')
    print('  subject is refusing to choose.')

    show('2. Segment tree: cut the range into whole subtrees')
    cover = canonical_cover(16, 3, 13)
    print('  n = 16, query [3, 13) touches nodes %s' % cover)
    print('    that is %d nodes for a 10-element range' % len(cover))
    print('    -> at most 2 nodes per level, because at most one node per')
    print('       level can be a partial "left edge" and one a "right edge"')
    for n_ in (1024, 65536, 1000000):
        avg, worst = query_cost_profile(n_, trials=5000)
        print('    n = %-8d  average %.2f nodes, worst seen %d  (bound 2*log2(n) = %.0f)'
              % (n_, avg, worst, 2 * (n_.bit_length() - 1)))
    st = SegmentTree([1, 3, 5, 7, 9, 11])
    print('  SegmentTree([1,3,5,7,9,11]).query(1, 5) = %d  (3+5+7+9)' % st.query(1, 5))
    st.update(2, 100)
    print('  after update(2, 100): query(1, 5) = %d' % st.query(1, 5))

    show('3. Lazy propagation: stop at the covered node')
    lv, ev, pushes = lazy_vs_eager()
    print('  n = 100000, 2000 random range-adds')
    print('    eager (descend to every leaf) : %d node visits' % ev)
    print('    lazy  (stop when covered)     : %d node visits' % lv)
    print('    speedup %.0fx, with %d pushes down' % (ev / lv, pushes))
    print('  the note left behind is the whole trick: an update is only')
    print('  applied to a child when somebody actually looks at that child.')

    show('4. Fenwick tree: the shape is in the bits')
    print('  i & -i is the lowest set bit, and tree[i] covers that many')
    print('  elements ending at i:')
    f = Fenwick(list(range(1, 17)))
    for i in (1, 2, 4, 6, 8, 12, 16):
        lo_, hi_ = f.covered_by(i)
        print('    tree[%2d]  binary %6s  lowbit %2d  covers a[%2d:%2d]'
              % (i, bin(i)[2:], i & -i, lo_, hi_))
    for n_ in (1024, 65536, 1000000):
        avg, worst = fenwick_cost_profile(n_)
        print('    n = %-8d  prefix() takes %.2f steps on average, %d worst'
              % (n_, avg, worst))
    a16 = list(range(1, 17))
    f2 = Fenwick(a16)
    print('  Fenwick(1..16).range_sum(3, 11) = %d  (sum of 4..11 = %d)'
          % (f2.range_sum(3, 11), sum(a16[3:11])))
    print('  lower_bound: smallest i with prefix(i+1) >= 40 is %d (prefix = %d)'
          % (f2.lower_bound(40), f2.prefix(f2.lower_bound(40) + 1)))

    show('5. Range update with BITs instead of lazy')
    ru = FenwickRangeUpdate(10)
    ru.range_add(2, 6, 5)
    ru.range_add(4, 8, 3)
    print('  range add + point query (one BIT over the difference array):')
    print('    %s' % [ru.point_query(i) for i in range(10)])
    rr = FenwickRangeRange(10)
    rr.range_add(2, 6, 5)
    rr.range_add(4, 8, 3)
    print('  range add + range sum (two BITs):')
    print('    sum[0:10] = %d, sum[3:7] = %d' % (rr.range_sum(0, 9), rr.range_sum(3, 6)))
    print('  same asymptotics as a lazy segment tree, a quarter of the memory,')
    print('  and it only works because addition can be undone.')

    show('6. What a BIT cannot do')
    true_min, bit_min, seg_min = min_bit_failure()
    print('  a = [5, 3, 8, 1], then a[3] = 9')
    print('    real min           = %d' % true_min)
    print('    prefix-min BIT     = %s   <- wrong, it never forgot the 1' % bit_min)
    print('    min segment tree   = %d' % seg_min)
    print('  range(l, r) = prefix(r) - prefix(l) needs an INVERSE.  min has')
    print('  none, so a BIT can only ever let values decrease.  A segment')
    print('  tree combines disjoint pieces and never subtracts, so any')
    print('  associative operation works - sum, min, max, gcd, matrices.')

    show('7. Memory and wall clock')
    n_ = 1000000
    print('  n = %d, 64-bit cells' % n_)
    print('    segment tree (iterative) 2n = %8d cells' % (2 * n_))
    print('    segment tree (recursive) 4n = %8d cells' % (4 * n_))
    print('    lazy segment tree      2*4n = %8d cells' % (8 * n_))
    print('    fenwick tree            n+1 = %8d cells' % (n_ + 1))
    t = timing()
    base = t['fenwick']
    print('  20000 elements, 2000 (update + range query) pairs:')
    for k in ('plain array', 'prefix sum', 'segment tree', 'fenwick'):
        print('    %-14s %7.1f ms   %5.1fx fenwick' % (k, t[k] * 1000, t[k] / base))

    show('8. LeetCode')
    nums = [1, 3, 5]
    na = NumArrayBIT(nums)
    nb = NumArraySeg(nums)
    print('  307  NumArray([1,3,5]).sumRange(0, 2) = %d / %d'
          % (na.sumRange(0, 2), nb.sumRange(0, 2)))
    na.update(1, 2)
    nb.update(1, 2)
    print('       after update(1, 2)              = %d / %d'
          % (na.sumRange(0, 2), nb.sumRange(0, 2)))
    print('       a BIT stores deltas, so "set" becomes add(i, new - old)')
    demo = [5, 2, 6, 1]
    print('  315  count_smaller(%s) = %s' % (demo, count_smaller(demo)))
    rng = random.Random(315)
    big = [rng.randint(-10000, 10000) for _ in range(4000)]
    t0 = time.perf_counter()
    fast = count_smaller(big)
    t1 = time.perf_counter()
    slow = count_smaller_bruteforce(big)
    t2 = time.perf_counter()
    assert fast == slow
    print('       n = 4000: BIT %.1f ms vs brute force %.1f ms (%.0fx), same answer'
          % ((t1 - t0) * 1000, (t2 - t1) * 1000, (t2 - t1) / (t1 - t0)))
    print('  370  get_modified_array(5, [[1,3,2],[2,4,3],[0,2,-2]]) = %s'
          % get_modified_array(5, [[1, 3, 2], [2, 4, 3], [0, 2, -2]]))
    print('       all updates before any query -> a difference array, O(n + k),')
    print('       no tree at all.  Reach for a segment tree only when the')
    print('       updates and the queries are interleaved.')

    show('9. Tests')
    rng = random.Random(2929)
    for _ in range(200):
        m = rng.randint(1, 40)
        arr = [rng.randint(-9, 9) for _ in range(m)]
        seg = SegmentTree(list(arr))
        fen = Fenwick(list(arr))
        cur = list(arr)
        for _ in range(20):
            l = rng.randrange(m + 1)
            r = rng.randint(l, m)
            assert seg.query(l, r) == sum(cur[l:r])
            assert fen.range_sum(l, r) == sum(cur[l:r])
            i = rng.randrange(m)
            v = rng.randint(-9, 9)
            seg.update(i, v)
            fen.add(i, v - cur[i])
            cur[i] = v
        lz = LazySegmentTree(list(arr))
        ref = list(arr)
        for _ in range(20):
            l = rng.randrange(m)
            r = rng.randint(l, m - 1)
            v = rng.randint(-5, 5)
            lz.range_add(l, r, v)
            for k in range(l, r + 1):
                ref[k] += v
            q1 = rng.randrange(m)
            q2 = rng.randint(q1, m - 1)
            assert lz.range_sum(q1, q2) == sum(ref[q1:q2 + 1])
        rr = FenwickRangeRange(m)
        ru = FenwickRangeUpdate(m)
        ref = [0] * m
        for _ in range(20):
            l = rng.randrange(m)
            r = rng.randint(l, m - 1)
            v = rng.randint(-5, 5)
            rr.range_add(l, r, v)
            ru.range_add(l, r, v)
            for k in range(l, r + 1):
                ref[k] += v
            q1 = rng.randrange(m)
            q2 = rng.randint(q1, m - 1)
            assert rr.range_sum(q1, q2) == sum(ref[q1:q2 + 1])
            assert ru.point_query(q1) == ref[q1]
        pos = [rng.randint(0, 5) for _ in range(m)]
        fb = Fenwick(pos)
        run = 0
        pre = [0]
        for x in pos:
            run += x
            pre.append(run)
        for target in range(1, sum(pos) + 2):
            exp = next((i for i in range(m) if pre[i + 1] >= target), m)
            assert fb.lower_bound(target) == exp
        assert count_smaller(arr) == count_smaller_bruteforce(arr)
    mn = MinSegmentTree([4, 2, 9, 7, 1])
    assert mn.query(0, 5) == 1 and mn.query(2, 4) == 7
    mn.update(4, 100)
    assert mn.query(0, 5) == 2
    assert get_modified_array(5, [[1, 3, 2], [2, 4, 3], [0, 2, -2]]) == [-2, 0, 3, 5, 3]
    print('  segment tree, lazy, fenwick, two-BIT range/range, lower_bound,')
    print('  min tree, LC 307 / 315 / 370 - all checked against brute force')
    print()
    print('all assertions passed')


if __name__ == '__main__':
    main()
