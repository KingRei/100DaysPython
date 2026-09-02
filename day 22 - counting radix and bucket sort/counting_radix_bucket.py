"""Day 22 - counting sort, radix sort, bucket sort.

Run me:  python counting_radix_bucket.py

Everything a comparison sort does, it does by asking "is a < b?".  That single
restriction is where the n log n lower bound comes from, and stepping outside
it is the whole subject of this file.  Counting sort never compares two
elements at all; it uses the key as an address.  Radix sort is counting sort
applied one digit at a time.  Bucket sort is counting sort that gives up on
exactness and delegates the last bit of ordering to somebody else.

The parts worth staying for are the conditions, not the algorithms:

  * O(n) is really O(n + k), and k is the size of the key space
  * radix sort is only correct because every pass is STABLE
  * the digit width r trades passes against buckets, and the optimum is
    narrow (4-8 bits) for exactly the same reason on CPU and GPU
  * the parallel skeleton of counting sort is a prefix sum, and the two
    classic scans differ in work, not in depth
  * a real one lives in every MoE model: sglang's moe_align_block_size is a
    counting sort over expert ids, with the buckets rounded up to a GEMM
    block size

Sections 1-8 are the algorithms and the scan, section 9 is the kernel, and
sections 10-11 are the LeetCode pair.
"""

import math
import random
import struct
from collections import Counter


# ============================================================ 1. the bound
#
# A comparison sort is a decision tree: every internal node is one "a < b?"
# and every leaf is one permutation.  A tree of depth d has at most 2**d
# leaves, and we need at least n! of them, so d >= log2(n!).  That is the
# whole proof, and it says nothing about algorithms - only about questions.
#
# Counting sort answers zero of those questions.  It is not a counterexample
# to the theorem; it simply is not in the model.

def comparison_lower_bound(n):
    """log2(n!), the minimum number of comparisons any comparison sort makes
    on the worst input of length n."""
    return math.lgamma(n + 1) / math.log(2.0)


def merge_sort_comparisons(a, stats):
    """Merge sort, instrumented.  Merge sort is close to optimal in the
    comparison model, so its count is a fair stand-in for "the best a
    comparison sort can do"."""
    if len(a) <= 1:
        return list(a)
    mid = len(a) // 2
    left = merge_sort_comparisons(a[:mid], stats)
    right = merge_sort_comparisons(a[mid:], stats)
    out = []
    i = j = 0
    while i < len(left) and j < len(right):
        stats['cmp'] += 1
        if left[i] <= right[j]:
            out.append(left[i]); i += 1
        else:
            out.append(right[j]); j += 1
    out.extend(left[i:]); out.extend(right[j:])
    return out


# ================================================== 2. counting sort proper
#
# Three phases, and they are the same three phases in every parallel version
# of this that anyone has ever shipped:
#
#   count    histogram the keys
#   scan     exclusive prefix sum over the histogram -> where each bucket starts
#   scatter  walk the input and drop each element at its bucket's cursor
#
# The scan is the interesting one.  After it, count[v] is not "how many v are
# there" but "how many elements are strictly less than v", which is exactly
# the index where the first v belongs.  The histogram has become an address
# table.

def counting_sort(keys, k, payload=None, stable=True, trace=None):
    """Sort `keys` (integers in [0, k)) with no comparisons at all.

    `payload` rides along - counting sort is a permutation, so anything
    indexed the same way as `keys` can be permuted with it.  That is not a
    detail: the GPU kernel in section 8 carries the routing weights next to
    the token ids for exactly this reason.

    `stable=False` scatters each bucket back to front, which still sorts
    correctly and is still O(n + k).  It only becomes wrong when something
    downstream depends on ties keeping their input order - see section 4."""
    n = len(keys)
    counts = [0] * k
    for v in keys:
        counts[v] += 1
    if trace is not None:
        trace.append(('count', list(counts)))

    # exclusive prefix sum: counts[v] becomes the start of bucket v
    total = 0
    starts = [0] * k
    for v in range(k):
        starts[v] = total
        total += counts[v]
    if trace is not None:
        trace.append(('scan', list(starts)))

    out_keys = [0] * n
    out_payload = [None] * n if payload is not None else None
    if stable:
        # Walk the input forward, each bucket cursor moving forward too.  Equal
        # keys therefore land in input order.
        cursor = list(starts)
        order = range(n)
    else:
        # Same buckets, filled from the back.  Equal keys come out reversed.
        cursor = [starts[v] + counts[v] - 1 for v in range(k)]
        order = range(n)
    for i in order:
        v = keys[i]
        pos = cursor[v]
        out_keys[pos] = v
        if payload is not None:
            out_payload[pos] = payload[i]
        cursor[v] += 1 if stable else -1
    if trace is not None:
        trace.append(('scatter', list(out_keys)))
    return (out_keys, out_payload) if payload is not None else out_keys


def counting_sort_ops(n, k):
    """Operation count, the honest version: O(n + k), not O(n).

    n touches to histogram, k to scan, n to scatter.  The k term is invisible
    when k is small and is the entire cost when k is large."""
    return {'count': n, 'scan': k, 'scatter': n, 'total': 2 * n + k,
            'memory_slots': k}


# =============================================== 3. what "O(n)" is hiding
#
# "Counting sort is linear" is true and useless on its own.  It is linear in
# n + k, so it is linear in n only when k = O(n).  Two ways that breaks:
# time (the scan walks every bucket, empty or not) and space (you allocate
# every bucket, empty or not).

def crossover_table(n, k_values):
    """For a fixed n, watch the k term take over."""
    rows = []
    bound = comparison_lower_bound(n)
    for k in k_values:
        ops = counting_sort_ops(n, k)
        rows.append({'k': k, 'total': ops['total'], 'slots': ops['memory_slots'],
                     'vs_bound': ops['total'] / bound})
    return rows


# ============================================================= 4. LSD radix
#
# Radix sort is counting sort run once per digit, least significant first,
# and it is correct for exactly one reason: every pass is stable, so the
# order established by the previous (less significant) passes survives.
#
# Drop stability and the algorithm does not degrade - it breaks.  Pass d
# scrambles the work of pass d-1 and the output is simply not sorted.

def digits_of(x, r, passes):
    """The `passes` digits of x in base 2**r, least significant first."""
    mask = (1 << r) - 1
    return [(x >> (r * p)) & mask for p in range(passes)]


def radix_sort(values, bits=32, r=8, stable=True, payload=None, trace=None):
    """LSD radix sort over non-negative integers of `bits` bits.

    `r` is the digit width: 2**r buckets per pass, ceil(bits/r) passes."""
    passes = (bits + r - 1) // r
    k = 1 << r
    mask = k - 1
    cur = list(values)
    cur_payload = list(payload) if payload is not None else None
    for p in range(passes):
        keys = [(v >> (r * p)) & mask for v in cur]
        # counting_sort permutes; we need the permutation applied to `cur`,
        # so `cur` itself is the payload of this pass.
        rider = list(zip(cur, cur_payload)) if cur_payload is not None else cur
        _, permuted = counting_sort(keys, k, payload=rider, stable=stable)
        if cur_payload is not None:
            cur = [x for x, _ in permuted]
            cur_payload = [y for _, y in permuted]
        else:
            cur = list(permuted)
        if trace is not None:
            trace.append({'pass': p, 'digit_bits': (r * p, r * p + r - 1),
                          'result': list(cur)})
    return (cur, cur_payload) if payload is not None else cur


def radix_cost(n, bits, r, tile=1024):
    """Work model for LSD radix sort with a digit of r bits.

    The textbook model is `passes * (2n + 2**r)` and it is misleading,
    because it charges for the histogram once per pass.  Every real
    implementation - CPU cache-blocked or GPU - histograms per TILE, so that
    the counters stay in fast memory: the array is cut into n/tile chunks and
    each chunk clears, fills and scans its own 2**r counters.

        passes = ceil(bits / r)
        work   = passes * (2n + ceil(n / tile) * 2**r)

    Now both ends are punished.  Small r means many passes.  Large r means
    every tile drags a histogram bigger than the tile itself, mostly full of
    zeros.  The minimum is broad and lands where production radix sorts
    actually sit."""
    passes = (bits + r - 1) // r
    buckets = 1 << r
    tiles = (n + tile - 1) // tile
    return {'r': r, 'passes': passes, 'buckets': buckets, 'tiles': tiles,
            'work': passes * (2 * n + tiles * buckets),
            'naive_work': passes * (2 * n + buckets),
            'hist_bytes': buckets * 4}


# ============================================================ 5. bucket sort
#
# Bucket sort stops pretending the key is an address and treats it as an
# estimate: scale the value into one of m buckets, sort each bucket with
# whatever you like, concatenate.  Linear *on average*, and only if the input
# is close to uniform over the range.  The distribution is not a footnote;
# it is the precondition.

def bucket_sort(values, m=None, stats=None):
    """Bucket sort over floats.  Returns the sorted list; `stats` collects
    the occupancy, which is the thing that actually decides the runtime."""
    n = len(values)
    if n == 0:
        return []
    m = m or n
    lo, hi = min(values), max(values)
    span = hi - lo
    buckets = [[] for _ in range(m)]
    for v in values:
        idx = 0 if span == 0 else min(m - 1, int((v - lo) / span * m))
        buckets[idx].append(v)
    inner = 0
    for b in buckets:
        # insertion sort, so the cost of a fat bucket is visibly quadratic
        for i in range(1, len(b)):
            x = b[i]
            j = i - 1
            while j >= 0 and b[j] > x:
                inner += 1
                b[j + 1] = b[j]
                j -= 1
            b[j + 1] = x
    if stats is not None:
        occ = [len(b) for b in buckets]
        stats['occupancy'] = occ
        stats['max_bucket'] = max(occ)
        stats['empty'] = sum(1 for c in occ if c == 0)
        stats['inner_moves'] = inner
    out = []
    for b in buckets:
        out.extend(b)
    return out


# ================================== 6. floats, negatives, and ordered keys
#
# Radix sort reads raw bits, and raw IEEE-754 bits are not ordered the way
# the numbers are: the sign bit makes every negative float look enormous, and
# among negatives the magnitude runs backwards.  This is the same
# order-preserving key from day 21 - flip the sign bit for positives, flip
# every bit for negatives - and it is what makes "sort floats by radix" legal.

def f32_bits(x):
    return struct.unpack('<I', struct.pack('<f', x))[0]


def bits_f32(b):
    return struct.unpack('<f', struct.pack('<I', b & 0xFFFFFFFF))[0]


def to_ordered(x):
    """IEEE-754 float32 -> uint32 whose unsigned order matches float order."""
    b = f32_bits(x)
    return (b ^ 0xFFFFFFFF) if (b >> 31) else (b | 0x80000000)


def from_ordered(u):
    b = (u & 0x7FFFFFFF) if (u >> 31) else (u ^ 0xFFFFFFFF)
    return bits_f32(b)


# ============================================ 7. the scan in the middle
#
# The count and the scatter parallelise on sight: one thread per element,
# atomics into the histogram, atomics into the cursors.  The scan does not.
# `starts[v] = starts[v-1] + counts[v-1]` is a loop where every iteration
# reads what the previous one wrote, which is the definition of sequential.
#
# Two classic ways out, and they differ in WORK, not in depth.

def sequential_scan(a, stats=None):
    """The baseline: n-1 adds, n-1 steps deep."""
    out, total = [], 0
    for v in a:
        out.append(total)
        total += v
    if stats is not None:
        stats.update({'name': 'sequential', 'work': max(len(a) - 1, 0),
                      'depth': max(len(a) - 1, 0)})
    return out, total


def hillis_steele_scan(a, stats=None):
    """Inclusive scan by doubling strides.  Every element adds the element
    `off` to its left, for off = 1, 2, 4, ...

    Depth log2(n), which is the point.  Work n log2(n), which is the price:
    it does more additions than the sequential version it replaces.  On a
    machine with idle lanes that is a bargain; as a serial algorithm it is
    strictly worse."""
    buf = list(a)
    n = len(buf)
    work = 0
    depth = 0
    off = 1
    while off < n:
        nxt = list(buf)
        for i in range(off, n):
            nxt[i] = buf[i] + buf[i - off]
            work += 1
        buf = nxt
        depth += 1
        off <<= 1
    if stats is not None:
        stats.update({'name': 'hillis-steele', 'work': work, 'depth': depth})
    return buf


def blelloch_scan(a, stats=None):
    """Work-efficient exclusive scan: up-sweep builds a reduction tree in
    place, then down-sweep pushes partial sums back down.

    2(n-1) adds and 2 log2(n) rounds - the same depth class as Hillis-Steele
    but linear work.  This is the scan sglang's MoE kernel runs on the HIP
    path, which is why it insists on a power-of-two buffer and zero-pads the
    expert histogram up to it."""
    n = len(a)
    assert n & (n - 1) == 0, 'Blelloch wants a power-of-two length'
    buf = list(a)
    work = depth = 0
    off = 1
    d = n >> 1
    while d > 0:                                   # up-sweep
        for i in range(d):
            ai = off * (2 * i + 1) - 1
            bi = off * (2 * i + 2) - 1
            buf[bi] += buf[ai]
            work += 1
        off <<= 1
        depth += 1
        d >>= 1
    total = buf[n - 1]
    buf[n - 1] = 0
    d = 1
    while d < n:                                   # down-sweep
        off >>= 1
        for i in range(d):
            ai = off * (2 * i + 1) - 1
            bi = off * (2 * i + 2) - 1
            buf[ai], buf[bi] = buf[bi], buf[bi] + buf[ai]
            work += 1
        d <<= 1
        depth += 1
    if stats is not None:
        stats.update({'name': 'blelloch', 'work': work, 'depth': depth})
    return buf, total


# ============================ 8. the kernel: moe_align_block_size
#
# Every MoE forward pass has to solve this: the router hands you, for each
# token, the ids of the experts that should process it.  The GEMM wants each
# expert's tokens contiguous, in blocks of a fixed size.  So you sort the
# tokens by expert id - and since expert ids are small integers in a known
# range, you sort them by counting.
#
# sgl-kernel/csrc/moe/moe_align_kernel.cu is that counting sort, with two
# production-shaped modifications:
#
#   * each bucket is rounded UP to a multiple of block_size before the scan,
#     so every expert's region starts on a block boundary.  The buckets have
#     holes, pre-filled with the sentinel id `numel` (a token index that does
#     not exist), and the GEMM masks them out.
#   * the answer "which expert owns block i" is recovered by BINARY SEARCH
#     over the same prefix array - the scan output is reused as an index.
#
# Read it next to counting_sort() above and the three phases line up exactly.

def moe_align_block_size(topk_ids, num_experts, block_size, stable=True):
    """Pure-python moe_align_block_size.

    topk_ids: [tokens][top_k] expert ids.  Returns
    (sorted_token_ids, expert_ids, num_tokens_post_pad).

    Matches the worked example in the docstring of
    sglang/srt/layers/moe/moe_runner/triton_utils/moe_align_block_size.py."""
    flat = [e for row in topk_ids for e in row]
    numel = len(flat)

    # ---- count ------------------------------------------------------------
    counts = [0] * num_experts
    for e in flat:
        counts[e] += 1

    # ---- pad, then scan ---------------------------------------------------
    # This is the one line that separates it from the textbook: the histogram
    # is rounded up before the prefix sum, not after.
    padded = [((c + block_size - 1) // block_size) * block_size for c in counts]
    prefix = [0] * (num_experts + 1)
    for e in range(num_experts):
        prefix[e + 1] = prefix[e] + padded[e]
    num_tokens_post_pad = prefix[num_experts]

    # ---- scatter ----------------------------------------------------------
    # `numel` is the sentinel: a token id one past the end, pointing at a row
    # the GEMM is told to ignore.  The real kernel fills this with a whole
    # separate thread block while the histogram is still running.
    sorted_token_ids = [numel] * num_tokens_post_pad
    cursor = list(prefix[:num_experts])
    order = range(numel) if stable else reversed(range(numel))
    for i in order:
        e = flat[i]
        sorted_token_ids[cursor[e]] = i
        cursor[e] += 1

    # ---- which expert owns each block, by binary search over the scan -----
    expert_ids = []
    for b in range(num_tokens_post_pad // block_size):
        block_start = b * block_size
        left, right = 0, num_experts
        while left < right:
            mid = (left + right) >> 1
            if prefix[mid] <= block_start:
                left = mid + 1
            else:
                right = mid
        expert_ids.append(left - 1)
    return sorted_token_ids, expert_ids, num_tokens_post_pad


def moe_scatter_atomic(topk_ids, num_experts, block_size, seed=0):
    """The big-batch kernel's scatter, simulated.

        rank = atomicAdd(&cumsum[expert_id], 1);
        sorted_token_ids[rank] = i;

    Every thread claims a slot with one atomic.  There is no cursor per
    thread, so the order within an expert is whatever order the atomics
    happened to retire in.  The result is a correct counting sort that is not
    stable and not reproducible.

    The small-batch kernel does the opposite: per-thread histograms, a scan
    across threads, then a private cursor per thread - deterministic.  Same
    function, two kernels, and the dispatch condition in the .cu is
    `topk_ids.numel() < 1024 && num_experts <= 64`."""
    flat = [e for row in topk_ids for e in row]
    numel = len(flat)
    counts = [0] * num_experts
    for e in flat:
        counts[e] += 1
    padded = [((c + block_size - 1) // block_size) * block_size for c in counts]
    prefix, total = [], 0
    for c in padded:
        prefix.append(total)
        total += c
    out = [numel] * total
    cursor = list(prefix)
    rng = random.Random(seed)
    arrival = list(range(numel))
    rng.shuffle(arrival)                 # the warps retire in some order
    for i in arrival:
        e = flat[i]
        out[cursor[e]] = i
        cursor[e] += 1
    return out


# ==================================================== 9. the LeetCode pair
#
# LC 75 is counting sort with k = 3, and its follow-up is the one-pass
# version.  LC 164 is the interesting one: it asks for the largest gap
# between consecutive elements of the sorted array, in O(n) time - which is
# a promise you cannot keep by sorting, so the shape of the answer is forced.

def sort_colors_counting(nums):
    """Two passes, k = 3.  Counting sort with the buckets written back in
    place instead of scattered - when the payload is nothing but the key,
    you do not need the permutation, only the histogram."""
    counts = [0, 0, 0]
    for v in nums:
        counts[v] += 1
    i = 0
    for v in range(3):
        for _ in range(counts[v]):
            nums[i] = v
            i += 1
    return nums


def sort_colors_dutch(nums):
    """One pass, the follow-up answer.  Three regions grow from both ends:
    [0, lo) is 0s, [lo, mid) is 1s, (hi, n) is 2s, and [mid, hi] is unknown.

    Note the asymmetry: after swapping with `lo` we advance `mid`, because
    what came back is known to be a 1; after swapping with `hi` we do not,
    because what came back has never been looked at."""
    lo, mid, hi = 0, 0, len(nums) - 1
    while mid <= hi:
        if nums[mid] == 0:
            nums[lo], nums[mid] = nums[mid], nums[lo]
            lo += 1; mid += 1
        elif nums[mid] == 2:
            nums[mid], nums[hi] = nums[hi], nums[mid]
            hi -= 1
        else:
            mid += 1
    return nums


def maximum_gap(nums, stats=None):
    """LC 164, in O(n) time and O(n) space.

    The trick is pigeonhole, not sorting.  Spread n numbers over n-1 buckets
    of width w = ceil((max - min) / (n - 1)); at least one bucket must be
    empty, so the largest gap is at least w, so it CANNOT lie inside a
    bucket.  Only the boundaries matter, and for those we need one min and
    one max per bucket - never the contents.  That is the whole reason this
    is allowed to be linear: we throw away the ordering we cannot afford to
    compute and keep only the two values per bucket that can possibly be part
    of the answer."""
    n = len(nums)
    if n < 2:
        return 0
    lo, hi = min(nums), max(nums)
    if lo == hi:
        return 0
    width = max(1, -(-(hi - lo) // (n - 1)))          # ceil division
    nbuckets = (hi - lo) // width + 1
    bmin = [None] * nbuckets
    bmax = [None] * nbuckets
    for v in nums:
        b = (v - lo) // width
        bmin[b] = v if bmin[b] is None else min(bmin[b], v)
        bmax[b] = v if bmax[b] is None else max(bmax[b], v)
    best = 0
    prev = bmax[0]
    empties = 0
    for b in range(1, nbuckets):
        if bmin[b] is None:
            empties += 1
            continue
        best = max(best, bmin[b] - prev)
        prev = bmax[b]
    if stats is not None:
        stats.update({'width': width, 'buckets': nbuckets, 'empty': empties})
    return best


def rule(title):
    print()
    print('=' * 74)
    print(f'  {title}')
    print('=' * 74)


def main():
    rule('1.  the n log n bound is about questions, not about sorting')
    for n in (16, 1_000, 100_000):
        st = {'cmp': 0}
        rng = random.Random(20260831 + n)
        a = [rng.randrange(1_000_000) for _ in range(n)]
        merge_sort_comparisons(a, st)
        b = comparison_lower_bound(n)
        print(f'  n = {n:>7,}:  log2(n!) = {b:12,.0f}   '
              f'merge sort used {st["cmp"]:12,}  ({st["cmp"] / b:.2f}x)')
    print('  Merge sort is already within a few percent of the floor, so no')
    print('  comparison sort is going to be meaningfully faster.  The only way')
    print('  down is to stop comparing.')

    rule('2.  counting sort: histogram, scan, scatter')
    keys = [4, 1, 3, 1, 0, 4, 1, 2]
    names = list('abcdefgh')
    tr = []
    sk, sp = counting_sort(keys, 5, payload=names, stable=True, trace=tr)
    print(f'  input keys     {keys}')
    print(f'  payload        {names}')
    for phase, data in tr:
        print(f'  {phase:<8}       {data}')
    print(f'  payload after  {sp}')
    print('  Read the scan line again: after it, entry v is not "how many v"')
    print('  but "how many elements come before the first v" - the histogram')
    print('  has turned into an address table, and no two elements were ever')
    print('  compared.')
    _, unstable = counting_sort(keys, 5, payload=names, stable=False)
    print(f'  the same sort, buckets filled back to front: {unstable}')
    print("  Still sorted.  The 1s are now 'g','d','b' instead of 'b','d','g'.")

    rule('3.  "O(n)" is O(n + k), and k is the key space')
    n = 10_000
    print(f'  n = {n:,} elements, cost model 2n + k, comparison floor '
          f'= {comparison_lower_bound(n):,.0f} comparisons')
    for row in crossover_table(n, [10, 1_000, 100_000, 10_000_000, 2 ** 32]):
        print(f'      k = {row["k"]:>14,}   ops = {row["total"]:>14,}   '
              f'buckets to allocate = {row["slots"]:>14,}')
    print('  Sorting 10,000 32-bit integers by counting means four billion')
    print('  buckets, almost all of them zero.  That is the failure mode radix')
    print('  sort exists to fix: keep k small by sorting a few bits at a time.')

    rule('4.  LSD radix - stability is the correctness condition')
    r4 = random.Random(4242)
    vals = [r4.randrange(1 << 16) for _ in range(12)]
    tr = []
    got = radix_sort(vals, bits=16, r=4, stable=True, trace=tr)
    assert got == sorted(vals)
    print(f'  input  {vals}')
    for t in tr:
        a, b = t['digit_bits']
        print(f'  pass {t["pass"]} (bits {a:>2}-{b:>2}): {t["result"]}')
    print('  Each pass only looks at four bits, and yet the array is fully')
    print('  sorted at the end, because every pass preserved the order the')
    print('  previous ones established.')
    bad = radix_sort(vals, bits=16, r=4, stable=False)
    print()
    print(f'  the same radix sort on an unstable counting sort:')
    print(f'      {bad}')
    print(f'      sorted? {bad == sorted(vals)}')
    print('  Not "slower" - wrong.  Pass 1 reversed ties that pass 0 had just')
    print('  ordered, and there is no later pass that can put them back.')

    rule('5.  how wide should a digit be?')
    nbig = 1_000_000
    TILE = 1024
    LDS = 64 * 1024
    print(f'  n = {nbig:,}, 32-bit keys, tiles of {TILE:,} elements.')
    print('  work = passes * (2n + tiles * 2**r), i.e. every tile clears, fills')
    print('  and scans its own histogram - which is what keeps the counters in')
    print(f'  fast memory (a thread block gets {LDS // 1024} KB of it).')
    print()
    best = best_naive = None
    for r in (1, 2, 4, 8, 11, 16, 24, 32):
        c = radix_cost(nbig, 32, r, tile=TILE)
        if best is None or c['work'] < best['work']:
            best = c
        if best_naive is None or c['naive_work'] < best_naive['naive_work']:
            best_naive = c
        fits = 'fits' if c['hist_bytes'] <= LDS else 'DOES NOT FIT'
        print(f'      r = {r:>2} bits: {c["passes"]:>2} passes x '
              f'{c["buckets"]:>12,} buckets  ->  work = {c["work"]:>16,}   '
              f'histogram {c["hist_bytes"] / 1024:>10,.0f} KB  {fits}')
    print()
    print(f'  The textbook model (one histogram per pass, not per tile) would')
    print(f'  pick r = {best_naive["r"]}, because it charges nothing for a bucket array of')
    print(f'  {best_naive["hist_bytes"] // 1024:,} KB.  Charge for it once per tile, as every real')
    print(f'  implementation must, and the answer moves to r = {best["r"]} with r = 4 a')
    print('  close second - which is exactly the window production radix sorts')
    print('  live in.  A 4-bit digit is 16 counters: small enough that a thread')
    print('  block, sometimes a single thread, keeps a private copy and skips')
    print('  the atomics entirely.')

    rule('6.  bucket sort, and the assumption it rests on')
    r5 = random.Random(77)
    nb = 2_000
    uniform = [r5.random() for _ in range(nb)]
    # same count, same range, but almost everything in one place
    clustered = [r5.gauss(0.5, 0.003) for _ in range(nb - 2)] + [0.0, 1.0]
    for label, data in (('uniform  ', uniform), ('clustered', clustered)):
        st = {}
        out = bucket_sort(data, m=nb, stats=st)
        assert out == sorted(data)
        print(f'  {label}: fullest bucket {st["max_bucket"]:>5,} of {nb:,}   '
              f'empty buckets {st["empty"]:>5,}   '
              f'inner-sort moves {st["inner_moves"]:>10,}')
    print('  Same n, same number of buckets, same code.  The clustered input')
    print('  puts almost every element in one bucket, the inner sort sees')
    print('  something close to n**2, and the "linear" sort is not linear.')
    print('  Bucket sort does not sort the data; it bets on the distribution.')

    rule('7.  radix over floats needs an order-preserving key')
    floats = [3.5, -1.0, 0.0, -0.0, 2.25, -7.5, 1e-4, -3.5]
    raw = sorted(floats, key=f32_bits)
    ordered = sorted(floats, key=to_ordered)
    print(f'  values                 {floats}')
    print(f'  sorted by RAW bits     {raw}')
    print(f'  sorted by ordered key  {ordered}')
    assert ordered == sorted(floats)
    assert all(from_ordered(to_ordered(v)) == bits_f32(f32_bits(v)) for v in floats)
    print('  Raw IEEE-754 bits sort every negative float above every positive')
    print('  one, and sort the negatives backwards among themselves - the sign')
    print('  bit is a flag, not a most significant digit.  Flip the sign bit on')
    print('  positives, flip all 32 bits on negatives, and unsigned order now')
    print('  matches numeric order.  Same key as day 21 used for radix select:')
    print('  once comparisons become counting, everything has to be an integer.')

    rule('8.  the scan in the middle: work vs depth')
    hist = [3, 1, 7, 0, 4, 1, 6, 3, 2, 8, 0, 5, 1, 2, 9, 4]
    ref, tot = sequential_scan(hist)
    s1, s2, s3 = {}, {}, {}
    sequential_scan(hist, s1)
    hs = hillis_steele_scan(hist, s2)
    bl, bl_total = blelloch_scan(hist, s3)
    assert bl == ref and bl_total == tot
    assert [x - hist[i] for i, x in enumerate(hs)] == ref
    print(f'  histogram      {hist}')
    print(f'  exclusive scan {ref}   (total {tot})')
    print()
    for st in (s1, s2, s3):
        print(f'      {st["name"]:<14} work {st["work"]:>4}   depth {st["depth"]:>4}')
    print('  Hillis-Steele buys depth with work: log2(n) rounds instead of n,')
    print('  but n log2(n) additions instead of n.  Blelloch gets the same')
    print('  depth class for linear work by building a reduction tree on the')
    print('  way up and pushing partial sums back down.  Nothing here is')
    print('  faster in serial - both do MORE additions than the loop they')
    print('  replace.  They are faster only because the lanes were idle.')
    n2 = 1024
    big = [1] * n2
    a1, a2, a3 = {}, {}, {}
    sequential_scan(big, a1); hillis_steele_scan(big, a2); blelloch_scan(big, a3)
    print(f'  at n = {n2:,}:  sequential {a1["work"]:>6,} adds / {a1["depth"]:>5,} deep   '
          f'hillis-steele {a2["work"]:>6,} / {a2["depth"]:>2}   '
          f'blelloch {a3["work"]:>6,} / {a3["depth"]:>2}')

    rule('9.  a real one: moe_align_block_size is a counting sort')
    topk_ids = [[2, 3, 4], [1, 2, 4], [1, 3, 4], [1, 2, 3]]
    block_size, num_experts = 4, 5      # ids 0..4, expert 0 unused
    sorted_ids, expert_ids, post_pad = moe_align_block_size(
        topk_ids, num_experts, block_size)
    print(f'  topk_ids         {topk_ids}      block_size = {block_size}')
    print(f'  flattened        {[e for row in topk_ids for e in row]}')
    print(f'  sorted_token_ids {sorted_ids}')
    print(f'  expert_ids       {expert_ids}   (one per block of {block_size})')
    print(f'  tokens after pad {post_pad}')
    expected = [3, 6, 9, 12, 0, 4, 10, 12, 1, 7, 11, 12, 2, 5, 8, 12]
    assert sorted_ids == expected, sorted_ids
    assert expert_ids == [1, 2, 3, 4]
    print('  That is exactly the worked example in sglang\'s docstring, and it')
    print('  is counting_sort() from section 2 with two changes: each bucket')
    print('  is rounded up to block_size BEFORE the scan, and the holes are')
    print('  pre-filled with token id 12 - one past the last real token - so')
    print('  the GEMM can read a full block and mask the padding out.')
    print()
    print('  Which expert owns block b?  Binary search the scan output:')
    counts = [0] * num_experts
    for e in (x for row in topk_ids for x in row):
        counts[e] += 1
    padded = [((c + block_size - 1) // block_size) * block_size for c in counts]
    pref, t = [], 0
    for c in padded:
        pref.append(t); t += c
    print(f'      counts   {counts}')
    print(f'      padded   {padded}')
    print(f'      prefix   {pref + [t]}')
    print('  The prefix array is used three times: as the bucket starts for')
    print('  the scatter, as the moving cursors during it, and as the search')
    print('  key for expert_ids.  One scan, three jobs.')

    print()
    print('  stable vs atomic scatter - the same non-determinism as day 21:')
    for seed in range(3):
        out = moe_scatter_atomic(topk_ids, num_experts, block_size, seed=seed)
        print(f'      atomicAdd order, run {seed}: {out}')
    print(f'      per-thread cursors       : {sorted_ids}')
    print('  All four are correct counting sorts and all four feed the same')
    print('  GEMM, because rows of a GEMM do not care what order they are in.')
    print('  But only the last one is reproducible, and sgl-kernel ships both:')
    print('  a deterministic per-thread-histogram kernel for small batches and')
    print('  an atomicAdd kernel for large ones, switched on')
    print('  `topk_ids.numel() < 1024 && num_experts <= 64`.')

    print()
    print('  at MoE scale (4,096 tokens, top-8 of 256 experts, block 64):')
    r6 = random.Random(2231)
    big_ids = [[e for e in r6.sample(range(256), 8)] for _ in range(4096)]
    s_ids, e_ids, pp = moe_align_block_size(big_ids, 256, 64)
    real = sum(1 for x in s_ids if x != 4096 * 8)
    print(f'      {4096 * 8:,} routed tokens -> {pp:,} padded slots '
          f'({len(e_ids):,} blocks, {pp - real:,} slots of padding, '
          f'{(pp - real) / pp:.1%})')
    print('  The padding is the price of a fixed block size, and it is what')
    print('  the counting sort is arranging the data to make possible.')

    rule('10.  LeetCode 75 - Sort Colors')
    for case in ([2, 0, 2, 1, 1, 0], [2, 0, 1], [0], [2, 2, 1, 1, 0, 0]):
        a, b = list(case), list(case)
        assert sort_colors_counting(a) == sorted(case)
        assert sort_colors_dutch(b) == sorted(case)
        print(f'  {str(case):<26} -> {b}')
    print('  Two passes with a histogram, or one pass with three pointers.')
    print('  The follow-up is asking for the one-pass version, and the reason')
    print('  it is harder is the asymmetry: a value swapped down from the top')
    print('  has never been examined, so `mid` must not advance.')

    rule('11.  LeetCode 164 - Maximum Gap, in linear time')
    cases = [([3, 6, 9, 1], 3), ([10], 0), ([1, 1, 1], 0),
             ([1, 10000000], 9999999), ([15, 3, 8, 1, 24, 22], 7)]
    for nums, want in cases:
        st = {}
        got = maximum_gap(nums, st)
        assert got == want, (nums, got, want)
        extra = f'   width {st["width"]}, {st["buckets"]} buckets, {st["empty"]} empty' if st else ''
        print(f'  {str(nums):<30} -> {got}{extra}')
    r7 = random.Random(164)
    rnd = [r7.randrange(10 ** 9) for _ in range(5000)]
    srt = sorted(rnd)
    brute = max(srt[i + 1] - srt[i] for i in range(len(srt) - 1))
    assert maximum_gap(rnd) == brute
    print(f'  5,000 random values up to 1e9: {brute:,} (matches the sorted answer)')
    print('  Note what is NOT computed: the order inside a bucket.  With n-1')
    print('  buckets at least one is empty, so the largest gap is at least one')
    print('  bucket wide and can never sit inside a bucket.  Keeping one min')
    print('  and one max per bucket is enough, and that is what buys O(n).')
    print('  "Sort it and look" is the O(n log n) answer; the linear answer')
    print('  comes from noticing which part of the sort you can skip.')

    print()
    print('=' * 74)
    print('  all assertions passed')
    print('=' * 74)


if __name__ == '__main__':
    main()
