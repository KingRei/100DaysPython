"""Day 21 - selection: you only want the k-th, not the whole order.

Run me:  python selection.py

Sorting answers "what is the order of everything".  Selection answers a much
smaller question - "what are the k biggest" - and the whole point of today is
that the smaller question deserves a cheaper algorithm.  Sorting to get the
top-8 of a million scores does about 20 million comparisons to throw away
999,992 of them.

The day is in two halves.

The first half is the classic single-machine toolkit, quickly: a size-k heap,
quickselect, median-of-medians, and reservoir sampling for when the data is a
stream you only see once.

The second half is the one that actually runs a few thousand times a second
inside every LLM you have ever used.  Sparse attention picks the top-k of a
long attention-score row; sampling masks all but the top-k logits.  Those
kernels do not use a heap and they do not use quickselect - they use *radix
select*, and they parallelise it in two quite different ways.  We build both,
including the grid-wide barrier, and including the deadlock that barrier is
famous for.
"""

import heapq
import random
import struct
import threading
import time

# ============================================================== float <-> key
#
# Radix select needs to look at the *bits* of a float and have integer order
# agree with float order.  IEEE-754 almost gives you that for free:
#
#   * positive floats already compare correctly as unsigned integers
#   * negative floats compare backwards, because they are sign-magnitude
#
# So: flip the sign bit of a positive, invert every bit of a negative.  The
# map is a bijection, so nothing is lost - you can go back.

def to_ordered(x):
    """float32 -> uint32 such that a < b  <=>  to_ordered(a) < to_ordered(b)."""
    bits = struct.unpack('<I', struct.pack('<f', x))[0]
    if bits >> 31:                       # negative: reverse the whole range
        return (~bits) & 0xFFFFFFFF
    return bits | 0x80000000             # positive: lift above every negative


def from_ordered(u):
    """The inverse of to_ordered - the map loses nothing."""
    bits = (u ^ 0x80000000) if (u >> 31) else ((~u) & 0xFFFFFFFF)
    return struct.unpack('<f', struct.pack('<I', bits))[0]


# One honest wrinkle: -0.0 and +0.0 are equal as floats but get *different*
# ordered keys (-0.0 lands just below +0.0).  Every radix select shares this;
# it only matters if you feed it signed zeros and expect them interchangeable.

def f32(x):
    """Round a Python float to float32, so the bit tricks are exact."""
    return struct.unpack('<f', struct.pack('<f', x))[0]


# ------------------------------------------------------- the lossy coarse key
#
# The GPU kernels do NOT histogram the top 8 bits of the float32 key.  They
# first narrow the value to float16 and histogram *that*.  The reason is
# distribution: the top byte of a float32 is the sign bit plus 7 of its 8
# exponent bits, and attention logits all live in a couple of exponents, so a
# raw-float32 histogram piles the entire row into three or four bins out of
# 256 and the first round buys you almost nothing.  float16 has a 5-bit
# exponent, so its top byte spreads the same values across the full 256.
#
# float32 -> float16 is monotone non-decreasing: it can merge two distinct
# values into one bin, but it can never swap their order.  That is the only
# property the algorithm needs.

def to_fp16_bits(x):
    """float32 -> the 16 raw bits of float16, round-to-nearest-even."""
    b = struct.unpack('<I', struct.pack('<f', f32(x)))[0]
    sign = (b >> 16) & 0x8000
    exp = (b >> 23) & 0xFF
    man = b & 0x7FFFFF
    if exp == 0xFF:                                  # inf / nan
        return sign | 0x7C00 | (0x200 if man else 0)
    e = exp - 127 + 15                               # rebias 127 -> 15
    if e >= 0x1F:                                    # overflows fp16 -> inf
        return sign | 0x7C00
    if e <= 0:                                       # subnormal or zero
        if e < -10:
            return sign
        man |= 0x800000                              # restore implicit 1
        shift = 14 - e
        half = man >> shift
        rem = man & ((1 << shift) - 1)
        tie = 1 << (shift - 1)
        if rem > tie or (rem == tie and (half & 1)):
            half += 1
        return sign | half
    half = (e << 10) | (man >> 13)
    rem = man & 0x1FFF                               # the 13 dropped bits
    if rem > 0x1000 or (rem == 0x1000 and (half & 1)):
        half += 1                                    # may carry into exponent
    return sign | half


def ordered16(x):
    """The same sign trick, on the 16 bits of float16."""
    b = to_fp16_bits(x)
    return ((~b) & 0xFFFF) if (b >> 15) else (b | 0x8000)


def coarse_key(x, bits=8):
    """The first-round bucket id: the top `bits` of the ordered float16 key."""
    return ordered16(x) >> (16 - bits)


# ====================================================== 1. the classic toolkit
#
# Four ways to answer "give me the k largest" on one machine.  Each one is a
# different trade: how much memory, how many passes, worst case or average.

def sort_topk(a, k):
    """The obvious one.  O(n log n) to answer an O(n) question."""
    return sorted(a, reverse=True)[:k]


def heap_topk(a, k, stats=None):
    """Keep a min-heap of size k.  The root is the *weakest* survivor, so one
    comparison per new element decides whether it is worth keeping.

    O(n log k) time and - the part that matters - O(k) memory.  This is the
    only algorithm here that never needs the whole input at once, so it is
    what you reach for when `a` arrives over a network."""
    h = []
    for x in a:
        if stats is not None:
            stats['cmp'] += 1
        if len(h) < k:
            heapq.heappush(h, x)
        elif x > h[0]:                   # beats the weakest survivor
            heapq.heapreplace(h, x)
        # else: discarded without ever entering the heap
    return sorted(h, reverse=True)


def heap_sort(a, stats=None):
    """Heap sort, for contrast: build a max-heap in place, then repeatedly
    swap the root to the end and sift down.  O(n log n) worst case with O(1)
    extra memory - the only sort we have seen with both.

    Written on a max-heap so it sorts ascending in place."""
    a = list(a)
    n = len(a)

    def sift(root, end):
        while True:
            child = 2 * root + 1
            if child >= end:
                return
            if child + 1 < end:
                if stats is not None:
                    stats['cmp'] += 1
                if a[child + 1] > a[child]:
                    child += 1           # always compare against the LARGER child
            if stats is not None:
                stats['cmp'] += 1
            if a[root] >= a[child]:
                return
            a[root], a[child] = a[child], a[root]
            root = child

    for i in range(n // 2 - 1, -1, -1):  # build: O(n), not O(n log n)
        sift(i, n)
    for end in range(n - 1, 0, -1):
        a[0], a[end] = a[end], a[0]      # largest goes to its final slot
        sift(0, end)
    return a


def quickselect(a, k, rng=None, stats=None):
    """The k largest, by partitioning and recursing into ONE side.

    Quick sort recurses into both halves and pays O(n log n).  Selection only
    ever needs one of them, and n + n/2 + n/4 + ... = 2n, so the average cost
    collapses to O(n).  That halving is the entire idea.

    Three-way partition, exactly as on day 20: without it a row of equal
    scores - which is what a masked attention row looks like - degrades to
    O(n^2)."""
    rng = rng or random
    a = list(a)
    lo, hi = 0, len(a) - 1
    while lo <= hi:
        p = a[rng.randint(lo, hi)]       # random pivot: no input is special
        lt, i, gt = lo, lo, hi
        while i <= gt:                   # Dutch national flag, descending
            if stats is not None:
                stats['cmp'] += 1
            if a[i] > p:
                a[lt], a[i] = a[i], a[lt]; lt += 1; i += 1
            elif a[i] < p:
                a[i], a[gt] = a[gt], a[i]; gt -= 1     # do NOT advance i
            else:
                i += 1
        # a[lo:lt] > p,  a[lt:gt+1] == p,  a[gt+1:hi+1] < p
        if k <= lt - lo:
            hi = lt - 1                  # answer lives entirely in the > side
        elif k <= gt + 1 - lo:
            break                        # the boundary falls inside the ties
        else:
            k -= gt + 1 - lo             # discard > and ==, keep hunting
            lo = gt + 1
    return a


def median_of_medians(a, k, stats=None):
    """Quickselect with a *guaranteed* good pivot: O(n) worst case.

    Split into groups of 5, take each group's median, then recursively take
    the median of those.  That pivot is provably better than 30% and worse
    than 30% of the input, so the recursion always throws away at least three
    tenths - no adversary can build a bad input.

    It is a genuine theoretical result and almost nobody ships it: the
    constant factor is several times a random pivot, and a random pivot is
    already unbeatable in practice.  Worth knowing exactly because it shows
    the difference between 'no bad input exists' and 'bad inputs are too
    unlikely to care about'."""
    a = list(a)

    def select(items, j):                # j-th smallest, 0-indexed
        if len(items) <= 5:
            return sorted(items)[j]
        medians = []
        for i in range(0, len(items), 5):
            g = sorted(items[i:i + 5])
            medians.append(g[len(g) // 2])
        pivot = select(medians, len(medians) // 2)
        lo = [x for x in items if x < pivot]
        eq = [x for x in items if x == pivot]
        hi = [x for x in items if x > pivot]
        if stats is not None:
            stats['cmp'] += len(items)
        if j < len(lo):
            return select(lo, j)
        if j < len(lo) + len(eq):
            return pivot
        return select(hi, j - len(lo) - len(eq))

    kth = select(a, len(a) - k)          # k-th largest = (n-k)-th smallest
    return kth


def reservoir_sample(stream, k, rng=None):
    """A uniform sample of k items from a stream of unknown length, in O(k).

    Keep the first k.  For the i-th item (0-indexed), keep it with probability
    k/(i+1), evicting a uniformly random survivor.  The invariant is that
    after seeing n items every item is present with probability exactly k/n -
    which is what 'uniform' means, and it holds without ever knowing n.

    This is the selection problem's sibling: same 'one pass, bounded memory'
    shape as heap_topk, except the score is random instead of given."""
    rng = rng or random
    res = []
    for i, x in enumerate(stream):
        if i < k:
            res.append(x)
        else:
            j = rng.randint(0, i)        # inclusive
            if j < k:
                res[j] = x
    return res


# ================================================== 2. radix select, sequential
#
# Every algorithm above compares elements against each other.  Radix select
# never does.  It bins elements by a slice of their bits, counts the bins, and
# uses the counts alone to decide which bin the k-th element lives in.  Then it
# recurses into that one bin with the next slice of bits.
#
# The consequence that makes it the GPU algorithm of choice: a round moves
# 256 counters, not n elements.  Quickselect has to physically swap elements
# to partition them; radix select never moves the data at all.

RADIX_BITS = 8
RADIX = 1 << RADIX_BITS


def radix_select(values, k, bits=RADIX_BITS, trace=None):
    """Find the pivot - the ordered key of the k-th largest - MSB first.

    Returns (pivot, above, equal) where `above` is the list of indices
    strictly greater than the pivot and `equal` is the list tied with it.
    The answer is `above` plus (k - len(above)) of `equal`; which ones, see
    section 4.
    """
    ordered = [to_ordered(v) for v in values]
    rounds = (32 + bits - 1) // bits
    radix = 1 << bits
    prefix = 0                            # the high bits decided so far
    remaining = k                         # how many of the top-k live in here
    for r in range(rounds):
        shift = 32 - (r + 1) * bits
        mask = (0xFFFFFFFF << (shift + bits)) & 0xFFFFFFFF   # the decided bits
        hist = [0] * radix
        for u in ordered:
            if (u & mask) == prefix:      # only elements still in the running
                hist[(u >> shift) & (radix - 1)] += 1
        # suffix sum: ge[b] = how many candidates have bucket >= b
        ge = [0] * (radix + 1)
        for b in range(radix - 1, -1, -1):
            ge[b] = ge[b + 1] + hist[b]
        # exactly one bucket straddles the boundary
        bucket = next(b for b in range(radix)
                      if ge[b] >= remaining and ge[b + 1] < remaining)
        if trace is not None:
            trace.append({'round': r, 'shift': shift, 'hist': hist,
                          'bucket': bucket, 'candidates': ge[0],
                          'gt': ge[bucket + 1], 'eq': hist[bucket],
                          'remaining': remaining, 'prefix': prefix})
        prefix |= bucket << shift
        remaining -= ge[bucket + 1]       # those are already guaranteed winners
    pivot = prefix
    above = [i for i, u in enumerate(ordered) if u > pivot]
    equal = [i for i, u in enumerate(ordered) if u == pivot]
    return pivot, above, equal


def radix_topk(values, k, bits=RADIX_BITS, trace=None):
    """The k largest indices, via radix select.  Ties resolved smallest-index."""
    if k >= len(values):
        return list(range(len(values)))
    pivot, above, equal = radix_select(values, k, bits, trace)
    return above + equal[:k - len(above)]


# ------------------------------------------------- projecting a bin back to fp32
#
# The second pass has to ask "is this element above the threshold bin, in it,
# or below it?".  The obvious way is to recompute coarse_key for every element
# - a float32->float16 conversion plus bit twiddling, per element, in the
# innermost loop of a bandwidth-bound kernel.
#
# The trick sglang uses instead: compute the bin's boundaries ONCE, as plain
# float32 numbers, and then the whole classification is two float compares
# against a loop-invariant register.  Since float32->float16 rounds to
# nearest, the boundary between two adjacent float16 values is their midpoint.

def fp16_bits_to_float(b):
    """The 16 bits of a float16 -> the float32 it represents."""
    sign = -1.0 if (b >> 15) else 1.0
    exp = (b >> 10) & 0x1F
    man = b & 0x3FF
    if exp == 0x1F:
        return sign * (float('inf') if man == 0 else float('nan'))
    if exp == 0:
        return sign * man * 2.0 ** -24                     # subnormal
    return sign * (1.0 + man / 1024.0) * 2.0 ** (exp - 15)


def unordered16(key):
    """Inverse of ordered16: the ordered uint16 back to raw float16 bits."""
    return (key & 0x7FFF) if (key >> 15) else ((~key) & 0xFFFF)


def coarse_bin_lower_bound(b, bits=8):
    """The smallest float32 v for which coarse_key(v, bits) >= b.

    So `v >= coarse_bin_lower_bound(t)` replaces `coarse_key(v) >= t`, and
    two of these bracket the threshold bin exactly."""
    if b <= 0:
        return float('-inf')
    if b >= (1 << bits):
        return float('inf')
    key = b << (16 - bits)                # smallest ordered16 key inside bin b
    hi = fp16_bits_to_float(unordered16(key))
    lo = fp16_bits_to_float(unordered16(key - 1))
    # The ends of the key space are inf and NaN, and a midpoint against those
    # is meaningless.  Getting this wrong is not academic: a NaN bound makes
    # BOTH of the collect pass's compares fail, the row comes back with fewer
    # than k entries, and whatever consumes those indices reads garbage.
    if hi != hi:                          # bin sits in a NaN region
        # NaN lives at BOTH ends of the key space - negative NaNs below every
        # finite value, positive NaNs above every finite value - so which
        # infinity to answer depends on which end we are at.
        return float('inf') if key >= 0x8000 else float('-inf')
    if hi == float('-inf'):
        return float('-inf')              # every finite value clears it
    if hi == float('inf'):
        # No midpoint available.  The boundary is half an ulp above the
        # largest finite float16, because that is where rounding tips to inf.
        prev = fp16_bits_to_float(unordered16(key - 2))
        return f32(lo + 0.5 * (lo - prev))
    if lo != lo or lo == float('-inf'):   # nothing finite below this bin
        return hi
    return f32(0.5 * (lo + hi))           # the round-to-nearest boundary


# ==================================== 3. parallel top-k, version A: one block
#
# This is flashinfer's FilteredTopK and sglang's TopKRegister: one GPU block
# owns one row and the row never leaves the chip.  The shape is
#
#     one coarse histogram pass  ->  one filter pass  ->  bounded refinement
#
# and the word that matters is *bounded*.  After the filter, the surviving
# candidates fit in a fixed shared-memory buffer, so every later round costs
# the same no matter whether the row was 4 thousand or 4 million long.
#
# The dtype table is not decoration.  It is the reason fp32 costs four
# refinement rounds and bf16 costs one.

DTYPES = {
    # name        exact_bits  num_rounds  first_shift  coarse_is_prefix
    'float32':   (32,         4,          24,          False),
    'float16':   (16,         1,          0,           True),
    'bfloat16':  (16,         1,          0,           True),
}

# For fp16 and bf16 the coarse key IS the top byte of the exact key, so the
# coarse pass already decided 8 of the 16 bits and one round of 8 finishes it.
#
# For fp32 the coarse key lives in a DIFFERENT key space - the value was
# narrowed to fp16 first - so it tells you which *bin* you are in but not a
# single bit of the fp32 key.  Refinement has to resolve all 32 bits from the
# top: shifts 24, 16, 8, 0.  Four rounds.  You buy a well-spread histogram
# with the coarse pass and you pay for it with rounds you cannot skip.

FILTER_CAPACITY = 16384       # candidate slots in shared memory (flashinfer's)


def exact_key(v, dtype='float32'):
    """The lossless sortable key for the refinement rounds."""
    if dtype == 'float32':
        return to_ordered(v)
    if dtype == 'float16':
        return ordered16(v)
    if dtype == 'bfloat16':                     # bf16 = the top 16 bits of fp32
        u = to_ordered(v)
        return u >> 16
    raise ValueError(dtype)


def filtered_topk(values, k, dtype='float32', capacity=FILTER_CAPACITY,
                  coarse_bits=8, stats=None):
    """Top-k for one row, the way a single GPU block does it.

    Returns the selected indices.  `stats` collects the numbers that make the
    argument: how many elements survived the filter, and how many rounds ran.
    """
    n = len(values)
    if k >= n:
        return list(range(n))
    exact_bits, num_rounds, first_shift, coarse_is_prefix = DTYPES[dtype]
    nbins = 1 << coarse_bits

    # How the coarse bin is derived is exactly what splits the two cases.
    # 16-bit inputs: the bin is literally the top byte of the exact key, so
    # the coarse pass has already decided half the bits.  fp32: the bin comes
    # from a narrowed copy and decides nothing about the fp32 key.
    if coarse_is_prefix:
        def bin_of(v):
            return exact_key(v, dtype) >> (16 - coarse_bits)
    else:
        def bin_of(v):
            return coarse_key(v, coarse_bits)

    # -- pass 1: the coarse histogram.  One streaming read of the row. --------
    hist = [0] * nbins
    for v in values:
        hist[bin_of(v)] += 1
    ge = [0] * (nbins + 1)
    for b in range(nbins - 1, -1, -1):
        ge[b] = ge[b + 1] + hist[b]
    threshold_bin = next(b for b in range(nbins)
                         if ge[b] >= k and ge[b + 1] < k)
    remaining = k - ge[threshold_bin + 1]       # still owed from inside the bin

    # -- the boundary trick: two floats replace a per-element key computation -
    #
    # Pass 2 is compute-bound, so recomputing fp32->fp16->bits for every
    # element just to recover its bin is real money.  Instead project the
    # threshold bin's two edges back into fp32 ONCE, and the whole
    # classification becomes two compares against loop-invariant registers.
    if coarse_is_prefix:
        v_hi = ((threshold_bin + 1) << (16 - coarse_bits))
        v_lo = (threshold_bin << (16 - coarse_bits))

        def rank(v):
            return exact_key(v, dtype)
    else:
        v_hi = coarse_bin_lower_bound(threshold_bin + 1, coarse_bits)
        v_lo = coarse_bin_lower_bound(threshold_bin, coarse_bits)

        def rank(v):
            return v                            # a bare float compare

    # -- pass 2: filter.  Three-way, and the only pass that touches the row
    #    again.  Everything above the threshold bin is already a winner; we
    #    never have to look at it or at anything below the bin ever again. ----
    winners, candidates = [], []
    for i, v in enumerate(values):
        rv = rank(v)
        if rv >= v_hi:
            winners.append(i)                   # guaranteed, no further work
        elif rv >= v_lo:
            candidates.append(i)                # needs the exact key to rank
    overflow = len(candidates) > capacity

    if stats is not None:
        stats.update(n=n, k=k, dtype=dtype, threshold_bin=threshold_bin,
                     v_lo=v_lo, v_hi=v_hi, winners=len(winners),
                     candidates=len(candidates), remaining=remaining,
                     overflow=overflow, coarse_bins_used=sum(1 for h in hist if h),
                     rounds=0)

    # A subtlety worth pausing on: `remaining` was computed from the fp16
    # histogram, but `winners` was computed from the fp32 boundary compare.
    # Rounding can move a boundary element between the two sets.  The kernels
    # trust the COLLECT counts, not the histogram counts - one classification
    # has to be the ground truth and it has to be the one that actually wrote
    # the output.
    remaining = k - len(winners)
    if remaining <= 0:
        return winners[:k]

    # -- refinement: radix select, but only over the candidate buffer --------
    keys = [(exact_key(values[i], dtype), i) for i in candidates]
    prefix, need = 0, remaining
    live = keys
    for r in range(num_rounds):
        shift = first_shift - 8 * r
        h = [0] * 256
        for u, _ in live:
            h[(u >> shift) & 0xFF] += 1
        g = [0] * 257
        for b in range(255, -1, -1):
            g[b] = g[b + 1] + h[b]
        bucket = next(b for b in range(256) if g[b] >= need and g[b + 1] < need)
        if stats is not None:
            stats['rounds'] = r + 1
        prefix |= bucket << shift
        need -= g[bucket + 1]
        nxt = [(u, i) for u, i in live if ((u >> shift) & 0xFF) == bucket]
        winners.extend(i for u, i in live if ((u >> shift) & 0xFF) > bucket)
        live = nxt
        if need == 0:
            break
    # whatever is left is tied with the pivot; take `need` of them
    winners.extend(i for _, i in live[:need])
    return winners


# =============================== 4. parallel top-k, version B: many blocks
#
# Version A caps out at one block per row.  If you have four rows and a
# thousand cores, most of the machine is idle.  Version B splits ONE row
# across several blocks (CTAs), which buys parallelism and costs you the one
# thing a single block never needed: a way for the blocks to agree.
#
# The algorithm barely changes.  Each CTA histograms its own chunk, the
# histograms are summed, and every CTA reads the same total and independently
# derives the same threshold bucket.  Note what does NOT move between CTAs:
# the data.  Only 256 counters cross the block boundary, once per round.
#
# What is genuinely new is the barrier.


class CountingBarrier:
    """The barrier a cooperative kernel actually uses, warts included.

    There is no `__syncthreads()` across blocks, so it is built by hand out of
    one shared counter.  Two details are load-bearing:

    * The counter is NEVER reset between rounds.  Each CTA keeps a private
      phase number and waits for `counter >= (phase+1) * num_ctas`.  A
      sense-reversing barrier would need a second variable and a second race.

    * The wait is `>=`, not `==`.  A CTA that is descheduled for a moment can
      wake up to find the counter has already sailed past its target; with
      `==` it would wait forever for an edge that already happened.
    """

    def __init__(self, num_ctas):
        self.num_ctas = num_ctas
        self.counter = 0
        self.lock = threading.Lock()
        self.max_spins = 0

    def arrive_and_wait(self, phase, deadline, spin_hook=None):
        with self.lock:
            self.counter += 1
        target = (phase + 1) * self.num_ctas
        spins = 0
        while True:
            if spin_hook is not None:
                spin_hook(spins)
            with self.lock:
                seen = self.counter
            if seen >= target:                  # ">=", not "=="
                self.max_spins = max(self.max_spins, spins)
                return True
            spins += 1
            if time.monotonic() > deadline:
                return False                    # caller decides: this is a hang
            # A real GPU spins on a cache line and costs nothing.  Python
            # threads share one interpreter lock, so an unyielding spin would
            # starve the very CTAs we are waiting for.  This sleep is an
            # artefact of the simulation, not of the algorithm.
            time.sleep(0.0002)


def multi_cta_topk(values, k, num_ctas=4, bits=8, reset_policy='last',
                   slow_cta=None, timeout=10.0, log=None):
    """Top-k for one row, split across `num_ctas` cooperating blocks.

    `reset_policy` picks who zeroes the shared counter so the next launch
    starts clean:

        'last'  - only the CTA that provably arrived last.  Correct.
        'first' - whoever finishes the final barrier first.  Deadlocks.

    Returns (indices, info).  If the barrier hangs, indices is None.
    """
    n = len(values)
    ordered = [to_ordered(v) for v in values]
    radix = 1 << bits
    rounds = (32 + bits - 1) // bits
    chunk = (n + num_ctas - 1) // num_ctas

    barrier = CountingBarrier(num_ctas)
    # the shared state that lives in global memory, one copy for the whole row
    # Three histogram buffers, not one and not two.  See the loop below.
    shared = {'hist': [[0] * radix for _ in range(3)],
              'out': [], 'out_lock': threading.Lock(),
              'hist_lock': threading.Lock(), 'deadlocked': False,
              'exit_arrivals': 0, 'reset_by': None}
    deadline = time.monotonic() + timeout
    per_cta = [None] * num_ctas

    def cta(c):
        lo, hi = c * chunk, min((c + 1) * chunk, n)
        mine = list(range(lo, hi))
        prefix, remaining, phase = 0, k, 0
        for r in range(rounds):
            shift = 32 - (r + 1) * bits
            mask = (0xFFFFFFFF << (shift + bits)) & 0xFFFFFFFF
            cur, nxt = r % 3, (r + 1) % 3
            local = [0] * radix
            for i in mine:
                if (ordered[i] & mask) == prefix:
                    local[(ordered[i] >> shift) & (radix - 1)] += 1
            with shared['hist_lock']:           # the only cross-CTA traffic:
                for b in range(radix):          # 256 counters, once per round
                    shared['hist'][cur][b] += local[b]
                if c == 0:
                    # Clear the NEXT round's buffer now, while everyone is
                    # still adding into this one.  That way the single barrier
                    # below proves two things at once - all adds for round r
                    # have landed, AND round r+1 starts from zero.
                    shared['hist'][nxt] = [0] * radix
            if not barrier.arrive_and_wait(phase, deadline):
                shared['deadlocked'] = True
                return
            phase += 1
            # Why three buffers and not two: a CTA that is slow to leave the
            # barrier is still reading round r's counts below, while the
            # others may already be adding into r+1 and clearing r+2.  With
            # only two buffers, "clear r+1" and "straggler still reading r-1"
            # are the same memory.  The third buffer is the slack.
            #
            # Every CTA now reads the SAME totals and does the SAME arithmetic,
            # so they all reach the same bucket without anyone broadcasting it.
            hist = list(shared['hist'][cur])
            ge = [0] * (radix + 1)
            for b in range(radix - 1, -1, -1):
                ge[b] = ge[b + 1] + hist[b]
            bucket = next(b for b in range(radix)
                          if ge[b] >= remaining and ge[b + 1] < remaining)
            prefix |= bucket << shift
            remaining -= ge[bucket + 1]
        pivot = prefix
        # -- collect.  Two passes, and the barrier between them is required. --
        gt = [i for i in mine if ordered[i] > pivot]
        with shared['out_lock']:                # reserve a contiguous run
            shared['out'].extend(gt)
        if not barrier.arrive_and_wait(phase, deadline):
            shared['deadlocked'] = True
            return
        phase += 1
        # Without that barrier a fast CTA's ties would eat output slots that a
        # slow CTA's guaranteed winners still need.
        eq = [i for i in mine if ordered[i] == pivot]
        with shared['out_lock']:
            room = k - len(shared['out'])
            shared['out'].extend(eq[:max(0, room)])
        per_cta[c] = {'chunk': (lo, hi), 'gt': len(gt), 'eq': len(eq)}
        # -- the exit protocol: who is allowed to reset the counter? ---------
        if reset_policy == 'first':
            # WRONG.  This CTA finished, but a peer may still be spinning
            # inside its final wait.  Zeroing the counter now means that peer
            # reads 0, which is less than its target, and it spins forever.
            if c == 0:
                with barrier.lock:
                    barrier.counter = 0
                shared['reset_by'] = c
        else:
            with barrier.lock:
                barrier.counter += 1
                mine_now = barrier.counter
            exit_target = (phase + 1) * num_ctas
            if mine_now == exit_target:         # exactly one CTA sees this,
                with barrier.lock:              # and it is provably the last
                    barrier.counter = 0
                shared['reset_by'] = c

    def hook_for(c):
        if slow_cta is None or c != slow_cta:
            return None

        def hook(spins):
            if spins == 0:
                time.sleep(0.05)                # deschedule at the worst moment
        return hook

    threads = []
    for c in range(num_ctas):
        h = hook_for(c)
        if h is None:
            t = threading.Thread(target=cta, args=(c,))
        else:
            def slow(cc=c, hh=h):
                orig = barrier.arrive_and_wait

                def patched(phase, dl, spin_hook=None):
                    return orig(phase, dl, hh)
                barrier.arrive_and_wait = patched
                try:
                    cta(cc)
                finally:
                    barrier.arrive_and_wait = orig
            t = threading.Thread(target=slow)
        threads.append(t)
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout + 1.0)

    info = {'num_ctas': num_ctas, 'chunk': chunk, 'per_cta': per_cta,
            'deadlocked': shared['deadlocked'], 'reset_by': shared['reset_by'],
            'counter_after': barrier.counter}
    if shared['deadlocked']:
        return None, info
    return shared['out'][:k], info


# =========================================== 5. ties, and two different knobs
#
# Radix select does not find "the k-th element".  It finds the k-th *value* -
# the pivot.  If the pivot occurs 50 times and only 7 slots are left, 43 of
# those elements have to lose, and nothing in the algorithm says which.
#
# On a GPU the answer is decided by whichever block wins a race on an atomic
# counter, so the same input can return a different - equally correct - answer
# on every run.  For attention that is usually fine.  For a regression test,
# or for anyone trying to reproduce a training run, it is not.
#
# The fix is two knobs that people constantly confuse:
#
#   deterministic  fixes the ORDER the selected indices appear in the output
#   tie_break      fixes WHICH tied elements get selected at all
#
# You can have a perfectly reproducible ordering of a set that still changes
# membership between runs.  Only tie_break settles membership, and to do that
# it needs a row-global rule - which is why it costs more than the cheap
# thread-strided ordering that `deterministic` alone can use.

def tie_topk(values, k, num_ctas=4, policy='race', rng=None):
    """Select k, and resolve ties at the pivot according to `policy`.

    'race'    whichever CTA gets to the atomic first  (what you get by default)
    'det'     fixed by CTA index, then position within the chunk
    'small'   prefer the smallest original index      (tie_break = Small)
    'large'   prefer the largest original index       (tie_break = Large)
    """
    rng = rng or random
    n = len(values)
    pivot, above, equal = radix_select(values, k)
    need = k - len(above)
    if need <= 0:
        return sorted(above[:k])
    # a grid-stride loop: CTA c owns elements c, c + num_ctas, c + 2*num_ctas, ...
    # This is how real kernels walk a row, and it is why the launch shape leaks
    # into the answer.
    lanes = [[i for i in equal if i % num_ctas == c] for c in range(num_ctas)]
    if policy == 'race':
        # CTAs reach the atomic in an unpredictable order; each dumps its ties
        order = list(range(num_ctas))
        rng.shuffle(order)
        pool = [i for c in order for i in lanes[c]]
    elif policy == 'det':
        # reproducible for a fixed launch - but the LANE ASSIGNMENT is baked
        # into the answer, so change num_ctas and the membership changes too
        pool = [i for c in range(num_ctas) for i in lanes[c]]
    elif policy == 'small':
        pool = sorted(equal)                     # row-global rule
    elif policy == 'large':
        pool = sorted(equal, reverse=True)
    else:
        raise ValueError(policy)
    return sorted(above + pool[:need])


# ================================================================ 6. LeetCode
#
# LC 215 - Kth Largest Element in an Array.  The interview answer is
# quickselect, and the interview *follow-up* is the one people miss.

def find_kth_largest(nums, k):
    """LC 215.  Quickselect with a random pivot and a three-way partition.

    The two traps, both of which turn O(n) into O(n^2):

      * a fixed pivot (first or last element) on already-sorted input
      * a two-way partition on input where every value is the same

    The second one is the nastier trap because sorted input at least looks
    suspicious in a test case, whereas [1,1,1,...,1] looks harmless."""
    a = quickselect(nums, k, rng=random.Random(0xC0FFEE))
    return a[k - 1]


# ================================================ 7. which kernel do you want?
#
# There is no single best top-k.  A library ships several and picks one at
# launch time from the shape of the problem, because the thing that dominates
# changes completely between "k=8 out of 4096, one row" and "k=4096 out of two
# million, ten thousand rows".

def should_use_filtered(n, k, dtype='float32', smem_bytes=16 * 1024,
                        need_tie_break=False, cuda_graph=False,
                        num_rows=1, num_sms=132):
    """A stand-in for the dispatch logic in front of a real top-k.

    Returns (use_filtered, reason).  The filtered kernel is the fast path: one
    block owns a whole row, keeps the candidates in shared memory, and never
    talks to another block - no grid barrier, no second launch, no workspace.
    All the conditions below are about whether that is *possible*, not whether
    it is quick."""
    elem = 4 if dtype == 'float32' else 2
    capacity = smem_bytes // elem
    if k > capacity:
        return False, f'k={k} exceeds the {capacity}-element shared-memory budget'
    if k > 2048:
        return False, f'k={k} is past the point where a per-block sort stops paying'
    if need_tie_break:
        return False, 'tie_break needs a row-global rule, which needs the multi-CTA pass'
    if n > 64 * capacity:
        return False, f'n={n} makes the filter pass too likely to overflow'
    # One block per row means the grid is num_rows blocks wide.  A single
    # enormous row is the worst case: correct, fits, and uses 1 of 132 SMs.
    if num_rows * 4 < num_sms and n > 32_768:
        return False, (f'{num_rows} row(s) x {n} elements would leave '
                       f'{num_sms - num_rows} of {num_sms} SMs idle')
    # A CUDA-graph capture cannot branch on a device-side count, so a kernel
    # that might need "one more round" has to be sized for the worst case.
    if cuda_graph:
        return True, 'fits, and the round count is bounded so the graph is safe'
    return True, 'fits in shared memory, single block, no grid barrier'


# ==================================================================== 8. main

def rule(title):
    print()
    print('=' * 74)
    print(title)
    print('=' * 74)


def main():
    rng = random.Random(20260821)

    rule('1.  selection vs sorting - the k largest of 200,000')
    n, k = 200_000, 8
    data = [rng.random() for _ in range(n)]
    truth = sorted(data, reverse=True)[:k]
    st_heap = {'cmp': 0}
    st_sort = {'cmp': 0}
    st_qs = {'cmp': 0}
    st_mom = {'cmp': 0}
    got_heap = heap_topk(data, k, st_heap)
    sorted_all = heap_sort(list(data), st_sort)
    got_qs = quickselect(data, k, random.Random(7), st_qs)[:k]
    got_mom = median_of_medians(data, k, st_mom)
    assert sorted_all == sorted(data)
    print(f'  n = {n:,}   k = {k}')
    print(f'  heap sort the lot    {st_sort["cmp"]:>12,} comparisons, and')
    print(f'                       {n - k:>12,} of the {n:,} results get thrown away')
    print(f'  size-k min-heap      {st_heap["cmp"]:>12,} comparisons, {k} floats of memory')
    print(f'  quickselect          {st_qs["cmp"]:>12,} comparisons  (n + n/2 + n/4 + ... = 2n)')
    print(f'  median-of-medians    {st_mom["cmp"]:>12,} comparisons  (O(n) *worst case*, and it shows)')
    print(f'  all four agree on the top-{k}: '
          f'{got_heap == truth and sorted(got_qs, reverse=True) == truth and got_mom == truth[-1]}')
    assert got_heap == truth
    assert sorted(got_qs, reverse=True) == truth
    assert got_mom == truth[-1]

    print()
    print('  reservoir sampling - the same one-pass, O(k)-memory shape, but for')
    print('  a uniform sample instead of the largest, and without knowing n:')
    hits = [0] * 10
    for t in range(20_000):
        for x in reservoir_sample(range(10), 3, random.Random(t)):
            hits[x] += 1
    print('    each of 10 items, sampled k=3, over 20,000 draws (expect ~6000 each):')
    print('    ' + ' '.join(f'{h:5d}' for h in hits))
    assert all(5400 < h < 6600 for h in hits)

    rule('2.  the order-preserving key - why bits can stand in for floats')
    probe = [-float('inf'), -3.5, -1.0, -0.0, 0.0, 1.0, 3.5, float('inf')]
    print('     value          float32 bits    ordered key')
    for v in probe:
        b = struct.unpack('<I', struct.pack('<f', f32(v)))[0]
        print(f'  {v:>10}      0x{b:08x}     0x{to_ordered(v):08x}')
    keys = [to_ordered(v) for v in probe]
    print(f'  monotone across the whole probe: {keys == sorted(keys)}')
    assert keys == sorted(keys)
    print('  positive floats already compare correctly as unsigned, so we only')
    print('  lift them above the negatives; negatives compare backwards, so we')
    print('  invert them wholesale.  Both directions are bijective - nothing is')
    print('  lost, which is what lets a radix pass replace a comparison.')

    rule('3.  radix select - counting instead of comparing')
    row = [f32(rng.gauss(0, 2)) for _ in range(4000)]
    trace = []
    pivot, above, equal = radix_select(row, 32, trace=trace)
    print('  4000 attention-like scores, k = 32.  Each round looks at 8 bits:')
    print('    round  shift  bucket   candidates left   still to find')
    for t in trace:
        print(f'      {t["round"]}     {t["shift"]:>2}     {t["bucket"]:>3}     '
              f'{t["candidates"]:>12,}     {t["remaining"]:>10}')
    print(f'  pivot = {from_ordered(pivot):.6f}, '
          f'{len(above)} strictly above it, {len(equal)} tied with it')
    print('  Note what shrinks: the candidate list.  Note what never moves:')
    print('  the data.  A round ships 256 counters, not 4000 floats.')
    assert sorted(radix_topk(row, 32), key=lambda i: -row[i])[:1] == \
        [max(range(len(row)), key=lambda i: row[i])]
    assert sorted(row[i] for i in radix_topk(row, 32)) == sorted(row, reverse=True)[:32][::-1]

    rule('4.  parallel top-k, version A: one block owns one row')
    wide = [f32(rng.gauss(0, 2)) for _ in range(20_000)]
    st = {}
    sel = filtered_topk(wide, 64, 'float32', stats=st)
    assert sorted(wide[i] for i in sel) == sorted(wide, reverse=True)[:64][::-1]
    print(f'  n = {len(wide):,}, k = 64, coarse histogram over {st["coarse_bins_used"]} occupied bins')
    print(f'  threshold bin {st["threshold_bin"]}, projected back to float32 as')
    print(f'    v_hi = {st["v_hi"]:.6f}   (everything above this is a guaranteed winner)')
    print(f'    v_lo = {st["v_lo"]:.6f}   (everything below this is out)')
    print(f'  guaranteed winners       {st["winners"]:>8,}')
    print(f'  candidates kept          {st["candidates"]:>8,}   '
          f'({100.0 * st["candidates"] / len(wide):.2f}% of the row)')
    print(f'  overflowed the buffer    {str(st["overflow"]):>8}')
    print('  The whole point: one pass over global memory narrows 20,000 elements')
    print('  down to a few hundred that fit in shared memory.  Everything after')
    print('  that is on-chip, so the second pass never touches DRAM again.')
    print()
    print('  The two float compares against v_lo / v_hi replace a per-element')
    print('  float32->float16 conversion in the innermost loop.  Because the')
    print('  narrowing rounds to nearest, the true boundary is the MIDPOINT of')
    print('  two adjacent float16 values - not the bin edge itself.')

    rule('5.  why float32 needs four refinement rounds and bfloat16 needs one')
    print('    dtype      exact bits   rounds   first shift   coarse key is a prefix?')
    for name, (bits_, nr, fs, pref) in DTYPES.items():
        print(f'    {name:<10} {bits_:>8}   {nr:>6}   {fs:>11}   {"yes" if pref else "no":>10}')
    print()
    print('  The table is not a tuning choice, it falls out of one fact: for a')
    print('  16-bit dtype the coarse bin IS the top byte of the exact key, so')
    print('  the coarse pass has already decided 8 of 16 bits and one round at')
    print('  shift 0 finishes the job.  For float32 the coarse key lives in a')
    print('  DIFFERENT key space - it came from a narrowed copy - so it decides')
    print('  zero bits of the float32 key, and refinement starts from the top.')
    print()
    print('  What that costs, measured on the same 20,000 values:')
    print('    dtype        coarse bins used   candidates after filter   rounds')
    for name in ('float32', 'float16', 'bfloat16'):
        s = {}
        got = filtered_topk(wide, 64, name, stats=s)
        assert len(got) == 64
        print(f'    {name:<10} {s["coarse_bins_used"]:>14}   '
              f'{s["candidates"]:>21,}   {s["rounds"]:>6}')
    print()
    print('  bfloat16 keeps far more candidates than float16 for the same row,')
    print('  and that is the honest reason nobody histograms the raw top byte of')
    print('  a float32: its high byte is sign plus 7 exponent bits, so a row of')
    print('  attention logits piles into a handful of bins.  Narrowing first')
    print('  spreads the same values across the full 256 - a narrowing is')
    print('  monotone non-decreasing, so it can merge values but never reorder')
    print('  them, and merging is exactly what the refinement rounds undo.')

    rule('6.  parallel top-k, version B: several blocks share one row')
    print('  Splitting a row across CTAs costs one thing a single block never')
    print('  needed: a grid-wide barrier.  There is no __syncthreads() across')
    print('  blocks, so we build one from a single counter.')
    print()
    print('    num_ctas   chunk   correct   who reset the counter   counter after')
    for nc in (1, 2, 4, 8, 16):
        ok = 0
        for t in range(5):
            got, info = multi_cta_topk(wide, 64, num_ctas=nc)
            if got is not None and \
               sorted(wide[i] for i in got) == sorted(wide, reverse=True)[:64][::-1]:
                ok += 1
        print(f'    {nc:>8}   {info["chunk"]:>5}     {ok}/5     '
              f'{"CTA" + str(info["reset_by"]):>18}   {info["counter_after"]:>12}')
        assert ok == 5
    print()
    print('  Three details in that barrier are load-bearing:')
    print('   * the counter is never reset between rounds; each CTA keeps a')
    print('     private phase and waits for counter >= (phase+1)*num_ctas.')
    print('   * the wait is >=, not ==.  A CTA that gets descheduled can wake')
    print('     up after the counter has already run past its target.')
    print('   * the histogram is TRIPLE buffered: round r accumulates into')
    print('     hist[r%3] while CTA 0 clears hist[(r+1)%3], so one barrier')
    print('     proves both "all the adds landed" and "the next buffer is')
    print('     clean".  The third buffer is slack for a straggler still')
    print('     reading round r.  With one buffer this deadlocks; with two it')
    print('     races.')

    rule('7.  the failure case - flashinfer issue #3610')
    print('  At the end of the kernel someone has to zero the counter so the')
    print('  next launch starts clean.  The obvious choice - whoever finishes')
    print('  first - is a deadlock, because a peer may still be spinning inside')
    print('  its final wait, and it reads 0 < target and never leaves.')
    print()
    print('    reset policy   slow CTA   deadlocked   who reset')
    for pol in ('last', 'first'):
        for trial in range(3):
            got, info = multi_cta_topk(wide, 64, num_ctas=4,
                                       reset_policy=pol, slow_cta=3, timeout=1.5)
            dead = info['deadlocked']
            print(f'    {pol:<12}   {"CTA3":>8}   {str(dead):>10}   '
                  f'{("CTA" + str(info["reset_by"])) if info["reset_by"] is not None else "-":>9}')
            assert dead == (pol == 'first')
            if pol == 'last':
                assert info['reset_by'] == 3
    print()
    print('  The fix is an exit barrier that doubles as leader election: every')
    print('  CTA increments once more, and the single CTA that sees its own')
    print('  increment land exactly on the target is provably the last one out,')
    print('  so every peer has already stopped reading.  Above, that is always')
    print('  CTA3 - the one we deliberately made slow.')

    rule('8.  ties - two knobs people keep confusing')
    tied = [f32(v) for v in ([9.0] * 3 + [5.0] * 12 + [1.0] * 10)]
    kt = 8
    print('  A row with three 9s, twelve 5s and ten 1s, k = 8.  Three elements')
    print('  beat the pivot; five of the twelve tied 5s get the remaining slots,')
    print('  and the algorithm does not say which five.')
    print()
    r = random.Random(3)
    seen = {tuple(tie_topk(tied, kt, policy='race', rng=r)) for _ in range(8)}
    print(f'    race (the default)   {len(seen)} different answers over 8 runs, '
          f'all of them correct')
    assert len(seen) > 1
    dets = {}
    for nc in (2, 4, 8):
        s = {tuple(tie_topk(tied, kt, num_ctas=nc, policy='det')) for _ in range(6)}
        assert len(s) == 1
        dets[nc] = sorted(s)[0]
        print(f'    deterministic, {nc} CTAs   stable over 6 runs -> {dets[nc]}')
    assert len(set(dets.values())) == 3
    print('    ...but that is three DIFFERENT stable answers.  `deterministic`')
    print('    pins the output ordering for one launch shape; it does not pin')
    print('    membership, because a grid-stride loop bakes num_ctas into which')
    print('    CTA sees which tie.')
    print()
    for pol in ('small', 'large'):
        s = {tuple(tie_topk(tied, kt, num_ctas=nc, policy=pol))
             for nc in (2, 4, 8) for _ in range(6)}
        assert len(s) == 1
        print(f'    tie_break = {pol:<6}    one answer across every launch shape '
              f'-> {sorted(s)[0]}')
    print('  Only a row-global rule - smallest index wins, or largest - makes')
    print('  membership reproducible, and that is why it costs more than the')
    print('  cheap ordering fix.')

    rule('9.  which kernel does a library actually launch?')
    cases = [
        (4096, 8, 'float32', False, 512),
        (200_000, 64, 'float32', False, 1),
        (4096, 8192, 'float32', False, 512),
        (4096, 64, 'float32', True, 512),
        (4096, 3000, 'float16', False, 512),
    ]
    for n_, k_, dt, tb, rows in cases:
        use, why = should_use_filtered(n_, k_, dt, need_tie_break=tb,
                                       num_rows=rows)
        print(f'  n={n_:>7,} k={k_:>5} {dt:<8} rows={rows:<4} '
              f'tie_break={str(tb):<5}')
        print(f'      -> {"filtered" if use else "multi-CTA":<10} : {why}')
    print('  Most of those conditions are about whether the single-block path')
    print('  is POSSIBLE at all - shared memory, tie_break - and when it is, it')
    print('  wins by default: no grid barrier, no second launch, no workspace.')
    print('  The odd one out is the second row: it fits perfectly well, it is')
    print('  just that one block per row means one block, and 131 idle SMs.')

    rule('10.  LeetCode 215 - Kth Largest Element in an Array')
    print('  nums = [3,2,3,1,2,4,5,5,6], k = 4  ->', find_kth_largest([3, 2, 3, 1, 2, 4, 5, 5, 6], 4))
    print('  nums = [3,2,1,5,6,4],       k = 2  ->', find_kth_largest([3, 2, 1, 5, 6, 4], 2))
    assert find_kth_largest([3, 2, 3, 1, 2, 4, 5, 5, 6], 4) == 4
    assert find_kth_largest([3, 2, 1, 5, 6, 4], 2) == 5
    big = [rng.randint(0, 50) for _ in range(50_000)]
    assert find_kth_largest(big, 137) == sorted(big, reverse=True)[136]
    flat = [7] * 50_000
    t0 = time.time()
    assert find_kth_largest(flat, 25_000) == 7
    print(f'  50,000 identical values, k = 25,000 -> 7, in {time.time() - t0:.3f}s')
    print('  That last case is the interview follow-up.  A two-way partition')
    print('  splits [7,7,7,...] into one empty side and one side of n-1, which')
    print('  is O(n^2) - about a billion operations here.  The three-way split')
    print('  drops every tie in a single pass, and it is the same reason the')
    print('  GPU kernels above have to think about ties at all.')

    print()
    print('all assertions passed.')


if __name__ == '__main__':
    main()
