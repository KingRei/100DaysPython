"""Day 23 - Binary search: three boundary conventions, and searching an answer space.

Run me:  python3 binary_search.py

Everything printed below is computed, not typed by hand.
"""

from bisect import bisect_left, bisect_right


# ---------------------------------------------------------------------------
# 1. The two boundaries.  Every correct binary search I write is one of these.
# ---------------------------------------------------------------------------
#
# Both keep the same invariant on the HALF-OPEN interval [lo, hi):
#
#     everything in a[:lo]  is already known to be too small
#     everything in a[hi:]  is already known to be big enough
#     the answer is somewhere in [lo, hi)
#
# The loop shrinks that window until it is empty, and then lo == hi IS the
# answer.  There is no "found it" early exit, so there is no separate case to
# get wrong.

def lower_bound(a, x):
    """First index i with a[i] >= x  (== len(a) if x is bigger than everything)."""
    lo, hi = 0, len(a)
    while lo < hi:
        mid = (lo + hi) // 2
        if a[mid] < x:
            lo = mid + 1          # a[mid] is too small -> discard it too
        else:
            hi = mid              # a[mid] might BE the answer -> keep it
    return lo


def upper_bound(a, x):
    """First index i with a[i] > x.  One character different from lower_bound."""
    lo, hi = 0, len(a)
    while lo < hi:
        mid = (lo + hi) // 2
        if a[mid] <= x:           # <= instead of <
            lo = mid + 1
        else:
            hi = mid
    return lo


def find(a, x):
    """Classic 'does x exist' search, built on lower_bound instead of open-coded."""
    i = lower_bound(a, x)
    return i if i < len(a) and a[i] == x else -1


def count_of(a, x):
    return upper_bound(a, x) - lower_bound(a, x)


def last_le(a, x):
    """Largest index i with a[i] <= x, or -1.  This is upper_bound - 1."""
    return upper_bound(a, x) - 1


def section_boundaries():
    print("=" * 72)
    print("1. lower_bound / upper_bound on an array with duplicates")
    print("=" * 72)
    a = [2, 3, 3, 3, 5, 8, 8, 13]
    print(f"  a = {a}")
    print(f"  {'x':>3} | {'lower':>5} {'upper':>5} {'count':>5} {'find':>5} {'last<=':>6}")
    print("  " + "-" * 40)
    for x in [1, 2, 3, 4, 8, 13, 20]:
        lb, ub = lower_bound(a, x), upper_bound(a, x)
        assert (lb, ub) == (bisect_left(a, x), bisect_right(a, x)), x
        print(f"  {x:>3} | {lb:>5} {ub:>5} {count_of(a, x):>5} {find(a, x):>5} {last_le(a, x):>6}")
    print()
    print("  Read the table, not the code:")
    print("    - lower_bound and upper_bound differ only for values that EXIST (x=3, 8).")
    print("    - for a missing value they collapse to the same insertion point (x=1, 4, 20).")
    print("    - lower_bound is Python's bisect_left; upper_bound is bisect_right.")
    print("    - 'the largest element <= x' is upper_bound(x) - 1, and the -1 is not")
    print("      cosmetic: it is what turns a lookup into a bucket lookup.")
    print()


# ---------------------------------------------------------------------------
# 2. Three ways to write the loop wrong.  All three still look like a binary
#    search, which is exactly the problem.
# ---------------------------------------------------------------------------

STEP_CAP = 100


def broken_never_shrinks(a, x):
    """Closed interval [lo, hi] but assigns hi = mid.  Hangs when lo == hi."""
    lo, hi = 0, len(a) - 1
    steps = 0
    while lo <= hi:
        steps += 1
        if steps > STEP_CAP:
            return None, steps           # gave up: infinite loop
        mid = (lo + hi) // 2
        if a[mid] < x:
            lo = mid + 1
        else:
            hi = mid                     # <-- hi never moves past mid
    return lo, steps


def broken_lo_equals_mid(a, x):
    """Half-open but assigns lo = mid.  Hangs when hi - lo == 1, because
    (lo + hi) // 2 == lo, so lo = mid is lo = lo."""
    lo, hi = 0, len(a)
    steps = 0
    while lo < hi:
        steps += 1
        if steps > STEP_CAP:
            return None, steps
        mid = (lo + hi) // 2
        if a[mid] < x:
            lo = mid                     # <-- should be mid + 1
        else:
            hi = mid
    return lo, steps


def broken_skips_answer(a, x):
    """Half-open but assigns hi = mid - 1.  Terminates, and quietly throws away
    the very element it was supposed to keep."""
    lo, hi = 0, len(a)
    while lo < hi:
        mid = (lo + hi) // 2
        if a[mid] < x:
            lo = mid + 1
        else:
            hi = mid - 1                 # <-- discards a[mid], which may be the answer
    return lo


def add_int32(x, y):
    """Two's-complement 32-bit add, so we can watch (lo + hi) overflow the way it
    does in C++ / CUDA."""
    s = (x + y) & 0xFFFFFFFF
    return s - (1 << 32) if s >= (1 << 31) else s


def section_broken():
    print("=" * 72)
    print("2. Three loops that look right")
    print("=" * 72)
    a = [1, 2, 3, 4, 5, 6, 7]
    print(f"  a = {a}")

    got, steps = broken_never_shrinks(a, 4)
    print(f"  (a) while lo <= hi with hi = mid      -> {'HANGS' if got is None else got}"
          f" (stopped after {steps} steps)")
    assert got is None

    got, steps = broken_lo_equals_mid(a, 4)
    print(f"  (b) lo = mid instead of lo = mid + 1  -> {'HANGS' if got is None else got}"
          f" (stopped after {steps} steps)")
    assert got is None

    good = lower_bound(a, 4)
    bad = broken_skips_answer(a, 4)
    print(f"  (c) hi = mid - 1                      -> returns {bad}, correct is {good}"
          f"  ({'WRONG' if bad != good else 'ok'}, and it terminates - no crash, no hang)")
    assert bad != good

    lo, hi = 2_000_000_000, 2_100_000_000
    mid_ok = lo + (hi - lo) // 2
    mid_bad = add_int32(lo, hi) // 2
    print(f"  (d) mid = (lo + hi) // 2 in int32     -> lo={lo}, hi={hi}")
    print(f"      (lo + hi) overflows to {add_int32(lo, hi)}, so mid = {mid_bad}"
          f"  (negative index)")
    print(f"      lo + (hi - lo) // 2 gives {mid_ok}")
    assert mid_bad < 0 < mid_ok
    print()
    print("  (a) and (b) hang, which at least gets noticed.  (c) is the dangerous one:")
    print("  it returns a plausible number.  (d) cannot happen in Python - ints are")
    print("  arbitrary precision - but the same loop inside a CUDA kernel indexes")
    print("  int32 offsets, and that is the 2006 JDK binarySearch bug.")
    print()

# ---------------------------------------------------------------------------
# 3. Binary search on the answer.  There is no array.
# ---------------------------------------------------------------------------
#
# The only thing binary search actually needs is a MONOTONE PREDICATE: a
# function ok(x) that is False, False, ..., False, True, True, ..., True over
# the search range.  A sorted array is just the special case ok(i) = a[i] >= x.
#
# So the recipe becomes:
#   1. guess what the answer is a number of  (hours? speed? bytes? qps?)
#   2. write ok(candidate) - usually a simulation, not a comparison
#   3. argue that ok is monotone
#   4. run the same lower_bound loop over the candidate range
#
# Step 3 is the whole job.  If ok is not monotone, the loop still terminates
# and still returns a number, and that number is meaningless.

def search_first_true(lo, hi, ok):
    """Smallest x in [lo, hi) with ok(x) True.  Same loop as lower_bound."""
    probes = []
    while lo < hi:
        mid = lo + (hi - lo) // 2
        good = ok(mid)
        probes.append((mid, good))
        if good:
            hi = mid
        else:
            lo = mid + 1
    return lo, probes


def koko_hours(piles, speed):
    """LC 875: hours needed to finish every pile at `speed` bananas/hour."""
    return sum(-(-p // speed) for p in piles)     # ceil division


def min_eating_speed(piles, h):
    ok = lambda s: koko_hours(piles, s) <= h
    ans, probes = search_first_true(1, max(piles) + 1, ok)
    return ans, probes


def can_split(nums, m, cap):
    """LC 410: can nums be cut into <= m contiguous chunks, each summing <= cap?"""
    chunks, cur = 1, 0
    for v in nums:
        if cur + v > cap:
            chunks += 1
            cur = 0
        cur += v
    return chunks <= m


def split_array_min_largest(nums, m):
    ok = lambda cap: can_split(nums, m, cap)
    return search_first_true(max(nums), sum(nums) + 1, ok)


def section_answer_space():
    print("=" * 72)
    print("3. Binary search on the answer (LC 875, LC 410)")
    print("=" * 72)
    piles, h = [30, 11, 23, 4, 20], 6
    ans, probes = min_eating_speed(piles, h)
    print(f"  piles = {piles}, h = {h}")
    print(f"  answer: {ans} bananas/hour  ({len(probes)} probes over a range of {max(piles)})")
    print()
    print("  The array we never built (True = finishes in time):")
    row_s = "   speed  "
    row_h = "   hours  "
    row_o = "   ok     "
    for s in range(1, 31, 2):
        row_s += f"{s:>4}"
        row_h += f"{koko_hours(piles, s):>4}"
        row_o += f"{('T' if koko_hours(piles, s) <= h else '.'):>4}"
    print(row_s)
    print(row_h)
    print(row_o)
    print()
    print("  hours(speed) is non-increasing, so ok(speed) is  . . . . T T T T  -")
    print("  monotone, and lower_bound finds the first T.  Probes:")
    for mid, good in probes:
        print(f"    speed={mid:>3} -> {koko_hours(piles, mid):>3} hours"
              f" -> {'ok, try slower' if good else 'too slow, speed up'}")
    assert ans == 23 and koko_hours(piles, 23) <= h < koko_hours(piles, 22)

    nums, m = [7, 2, 5, 10, 8], 2
    cap, probes2 = split_array_min_largest(nums, m)
    print()
    print(f"  LC 410: nums = {nums}, m = {m} -> smallest possible largest sum = {cap}"
          f" ({len(probes2)} probes)")
    print(f"    lower end of the range is max(nums) = {max(nums)} (one chunk must hold it)")
    print(f"    upper end is sum(nums) = {sum(nums)} (one chunk holds everything)")
    assert cap == 18 and can_split(nums, m, 18) and not can_split(nums, m, 17)
    print()


# ---------------------------------------------------------------------------
# 4. When the predicate is expensive, and lies.
# ---------------------------------------------------------------------------
#
# SGLang's autotuner (python/sglang/auto_benchmark_lib.py) binary searches the
# request rate: it asks "does the server still meet the SLA at this qps?", and
# each answer costs a full benchmark run.  Two things change once the predicate
# is a measurement rather than a comparison.

def bisect_float(lo, hi, ok, tolerance, max_rounds=64):
    """Float bisection.  Terminates on tolerance and on a round budget, not on
    lo < hi - a float interval can always be halved again."""
    rounds = 0
    best = None
    while hi - lo > tolerance and rounds < max_rounds:
        mid = round((lo + hi) / 2, 4)
        if mid <= lo or mid >= hi:          # midpoint collapsed onto an endpoint
            break
        if ok(mid):
            lo = mid
            best = mid
        else:
            hi = mid
        rounds += 1
    return best, lo, hi, rounds


def make_sla_probe(true_capacity, jitter=0.0, seed=0):
    """ok(qps) = 'the run met the SLA'.  With jitter > 0 the measurement is noisy
    near the capacity, so the predicate is no longer exactly monotone."""
    import random
    rng = random.Random(seed)
    calls = []

    def ok(qps):
        measured = true_capacity + (rng.uniform(-jitter, jitter) if jitter else 0.0)
        verdict = qps <= measured
        calls.append((qps, round(measured, 3), verdict))
        return verdict

    return ok, calls


def section_noisy_predicate():
    print("=" * 72)
    print("4. An expensive, noisy predicate: searching for max sustainable qps")
    print("=" * 72)
    cap = 13.7
    ok, calls = make_sla_probe(cap)
    best, lo, hi, rounds = bisect_float(1.0, 64.0, ok, tolerance=0.1)
    print(f"  clean predicate, true capacity {cap} qps")
    print(f"    {rounds} benchmark runs -> best passing qps {best}, bracket [{lo}, {hi}]")
    print(f"    every probe: {[c[0] for c in calls]}")
    assert best is not None and abs(best - cap) < 0.4

    ok_n, calls_n = make_sla_probe(cap, jitter=0.6, seed=23)
    best_n, lo_n, hi_n, rounds_n = bisect_float(1.0, 64.0, ok_n, tolerance=0.1)
    print(f"  noisy predicate (+-0.6 qps run-to-run)")
    print(f"    {rounds_n} runs -> best passing qps {best_n}, bracket [{lo_n}, {hi_n}]")
    flips = [c for c in calls_n if (c[0] <= cap) != c[2]]
    print(f"    {len(flips)} of {len(calls_n)} probes disagreed with the true capacity:")
    for qps, measured, verdict in flips:
        print(f"      qps={qps:>7} measured capacity {measured:>6}"
              f" -> said {'PASS' if verdict else 'FAIL'}")
    print()
    print("  A wrong answer near the middle of the search is unrecoverable: binary")
    print("  search never revisits a discarded half.  That is why the real loop")
    print("  carries a round budget (max_rounds) and keeps the passing RECORD, not")
    print("  just the number - and why the midpoint is guarded with")
    print("  'if qps <= lower or qps >= upper: break', since a rounded midpoint can")
    print("  land on an endpoint and stall the loop forever.")
    print()

# ---------------------------------------------------------------------------
# 5. Galloping: binary search when you expect the answer to be near the front.
# ---------------------------------------------------------------------------
#
# SGLang's radix cache has to answer "how long a prefix do these two token
# sequences share?" on every incoming request.  A per-token Python loop is out
# of the question, so RadixKey.match (srt/mem_cache/radix_cache.py) does an
# exponential search: compare slices of length 1, 2, 4, 8, ... until one of
# them differs, then binary search inside that one window.

def match_linear(t0, t1):
    """The obvious version.  Returns (matched, slice_compares, tokens_touched)."""
    n = min(len(t0), len(t1))
    i = 0
    while i < n and t0[i] == t1[i]:
        i += 1
    return i, i + (0 if i == n else 1), i + (0 if i == n else 1)


def match_binary(t0, t1):
    """Plain binary search over the whole range.  Each probe compares a slice, so
    the number of PROBES is log2(n) but the number of TOKENS touched is not."""
    n = min(len(t0), len(t1))
    probes = tokens = 0
    lo, hi = 0, n
    while lo < hi:
        mid = lo + (hi - lo) // 2
        probes += 1
        tokens += mid + 1
        if t0[: mid + 1] == t1[: mid + 1]:
            lo = mid + 1
        else:
            hi = mid
    return lo, probes, tokens


def match_gallop(t0, t1):
    """Exponential search, then binary search inside the window that broke.
    Mirrors RadixKey.match: `lo` is always a known-equal prefix length, so every
    slice compared is a fresh window rather than the whole prefix again."""
    n = min(len(t0), len(t1))
    probes = tokens = 0
    matched = n
    lo, step = 0, 1
    while lo < n:
        hi = lo + step if lo + step < n else n
        probes += 1
        tokens += hi - lo
        if t0[lo:hi] != t1[lo:hi]:
            # the divergence is inside [lo, hi); note the loop condition is
            # hi - lo > 1, so mid is strictly greater than lo and `lo = mid`
            # is safe here - the window shrinks even though lo moves to mid.
            while hi - lo > 1:
                mid = (lo + hi) // 2
                probes += 1
                tokens += mid - lo
                if t0[lo:mid] == t1[lo:mid]:
                    lo = mid
                else:
                    hi = mid
            matched = lo
            break
        lo = hi
        step *= 2
    return matched, probes, tokens


def section_gallop():
    print("=" * 72)
    print("5. Galloping prefix match (SGLang radix cache)")
    print("=" * 72)
    n = 4096
    base = list(range(n))
    print(f"  two token sequences of length {n}, sharing a prefix of length p")
    print(f"  {'p':>6} | {'linear':>17} | {'binary':>17} | {'gallop':>17}")
    print(f"  {'':>6} | {'probes':>8}{'tokens':>9} | {'probes':>8}{'tokens':>9}"
          f" | {'probes':>8}{'tokens':>9}")
    print("  " + "-" * 66)
    for p in [0, 3, 17, 250, 2048, 4095, 4096]:
        other = base[:p] + [-1] * (n - p)
        cells = []
        for fn in (match_linear, match_binary, match_gallop):
            got, probes, tokens = fn(base, other)
            assert got == min(p, n), (fn.__name__, p, got)
            cells.append(f"{probes:>8}{tokens:>9}")
        print(f"  {p:>6} | " + " | ".join(cells))
    print()
    print("  Two cost models, two different winners, and the honest one is the")
    print("  right-hand column of each pair:")
    print("    - counting PROBES, plain binary search looks unbeatable: 12 or 13")
    print("      whatever p is.  But a probe compares a prefix, not an element, so")
    print("      each probe costs O(mid) - the probes are not free and not equal.")
    print("    - counting TOKENS touched, binary search does ~n log n work while")
    print("      galloping does O(p): the windows double, so their lengths sum to")
    print("      about 2p, and the binary phase only searches the last window.")
    print("  Galloping is not uniformly better - at p ~ n/2 it makes about twice as")
    print("  many probes as binary search.  It is the right bet for a prefix cache")
    print("  because that workload is bimodal: a new conversation shares almost")
    print("  nothing, a follow-up turn shares almost everything, and hardly anything")
    print("  lands in the middle.")
    print("  (Linear touches the fewest tokens of all.  Its problem is that it touches")
    print("  them one at a time from Python, while the other two hand whole slices to")
    print("  the C-level compare - the same reason a 3x larger token count still wins.)")
    print()


# ---------------------------------------------------------------------------
# 6. The inverse of a prefix sum is a binary search.
# ---------------------------------------------------------------------------
#
# Yesterday's counting sort turned a histogram into an offset table with an
# exclusive scan.  Every batched kernel in an inference server carries that
# table as `cu_seqlens`, and every kernel that works one token at a time has to
# undo it: given a flat token index, which request does it belong to?
#
#     seq_of = torch.searchsorted(cu_seqlens, tok, right=True) - 1
#
# right=True is not a style choice.  It is the difference between correct and
# silently wrong at every sequence boundary - and every sequence has one.

def seq_of_tokens(cu_seqlens, n_tokens, right=True):
    ub = bisect_right if right else bisect_left
    return [ub(cu_seqlens, tok) - 1 for tok in range(n_tokens)]


def section_cu_seqlens():
    print("=" * 72)
    print("6. Undoing a prefix sum: flat token index -> request id")
    print("=" * 72)
    seq_lens = [3, 5, 2]
    cu = [0]
    for L in seq_lens:
        cu.append(cu[-1] + L)
    total = cu[-1]
    truth = [i for i, L in enumerate(seq_lens) for _ in range(L)]
    right_t = seq_of_tokens(cu, total, right=True)
    right_f = seq_of_tokens(cu, total, right=False)
    print(f"  seq_lens   = {seq_lens}")
    print(f"  cu_seqlens = {cu}")
    print("  token       " + "".join(f"{t:>4}" for t in range(total)))
    print("  truth       " + "".join(f"{v:>4}" for v in truth))
    print("  right=True  " + "".join(f"{v:>4}" for v in right_t))
    print("  right=False " + "".join(f"{v:>4}" for v in right_f))
    bad = [t for t in range(total) if right_f[t] != truth[t]]
    print(f"  wrong with right=False: tokens {bad}"
          f" - exactly the first token of every request")
    print(f"  token 0 gets seq id {right_f[0]}, which then indexes cu_seqlens[-1]"
          f" and reads another request's state.")
    assert right_t == truth and right_f != truth
    print()
    print("  bisect_right(cu, tok) - 1 = 'the last start that is <= tok', which is")
    print("  the definition of 'which bucket am I in'.  bisect_left asks for the")
    print("  first start >= tok, and at a boundary that is the NEXT request.")
    print()

# ---------------------------------------------------------------------------
# 7. Rounding up to a bucket: the other half of yesterday's padding story.
# ---------------------------------------------------------------------------
#
# A CUDA graph is captured for a fixed batch size, so a decode step of 37
# requests has to be replayed on the smallest captured shape that can hold it.
# SGLang does exactly one line for this
# (srt/model_executor/runner/base_cuda_graph_runner.py):
#
#     index = bisect.bisect_left(buckets, raw_size)
#     return buckets[index]
#
# bisect_left, not bisect_right.  If raw_size is already a captured size,
# bisect_left returns that size and nothing is padded; bisect_right would step
# to the NEXT bucket and pay for a whole graph's worth of empty rows.

# The default decode capture list (server_args._generate_decode_cuda_graph_batch_sizes)
NORMAL_BS = [1, 2, 4, 8, 12] + list(range(16, 257, 8))
SPEC_BS = (list(range(1, 9)) + list(range(10, 33, 2))
           + list(range(40, 65, 4)) + list(range(72, 257, 8)))


def pad_to_bucket(raw_size, buckets, right=False):
    assert raw_size <= buckets[-1], f"{raw_size} exceeds max captured bucket"
    index = (bisect_right if right else bisect_left)(buckets, raw_size)
    return buckets[index]


def padding_waste(buckets, hi=256):
    """Mean fraction of each replayed graph that is padding."""
    total = 0.0
    for bs in range(1, hi + 1):
        padded = pad_to_bucket(bs, buckets)
        total += (padded - bs) / padded
    return total / hi


def section_buckets():
    print("=" * 72)
    print("7. bisect_left as a capacity rounder (CUDA graph batch buckets)")
    print("=" * 72)
    print(f"  buckets = {NORMAL_BS[:8]} ... {NORMAL_BS[-3:]}  ({len(NORMAL_BS)} shapes)")
    print(f"  {'batch':>6} | {'bisect_left':>11} {'waste':>7} | {'bisect_right':>12} {'waste':>7}")
    print("  " + "-" * 52)
    for bs in [1, 3, 12, 16, 17, 24, 100, 249]:
        L = pad_to_bucket(bs, NORMAL_BS)
        R = pad_to_bucket(bs, NORMAL_BS, right=True)
        print(f"  {bs:>6} | {L:>11} {(L - bs) / L:>6.1%} | {R:>12} {(R - bs) / R:>6.1%}"
              + ("   <- exactly a captured size" if bs in NORMAL_BS else ""))
    print()
    wl = padding_waste(NORMAL_BS, 255)
    wr = sum((pad_to_bucket(b, NORMAL_BS, right=True) - b)
             / pad_to_bucket(b, NORMAL_BS, right=True) for b in range(1, 256)) / 255
    print(f"  averaged over batch sizes 1..255:")
    print(f"    bisect_left  {wl:.2%} of every replayed graph is padding")
    print(f"    bisect_right {wr:.2%}  (the same list, one character different)")
    print(f"    spec-decode bucket list, bisect_left: {padding_waste(SPEC_BS, 255):.2%}")
    try:
        pad_to_bucket(256, NORMAL_BS, right=True)
        raise SystemExit("expected bisect_right to run off the end")
    except IndexError:
        print(f"    and at the largest batch (256) bisect_right returns index"
              f" {len(NORMAL_BS)} -> IndexError,")
        print(f"    which is the failure the assert in _pad_to_bucket exists to name.")
    print()
    print("  The spec-decoding list is denser below 64 for a reason: with a draft")
    print("  model every batch is multiplied by the number of speculated tokens, so")
    print("  a fat bucket at small batch sizes is where the waste actually lands.")
    print("  Same lesson as yesterday's block padding - you cannot pick the bucket")
    print("  boundaries without knowing the distribution of what you are bucketing.")
    print()


# ---------------------------------------------------------------------------
# 8. The interview versions.
# ---------------------------------------------------------------------------

def search_range(nums, target):
    """LC 34: first and last position of target.  Two boundaries, no special
    cases - and the emptiness test is lo == hi, not a sentinel."""
    lo = lower_bound(nums, target)
    hi = upper_bound(nums, target)
    return [lo, hi - 1] if lo < hi else [-1, -1]


def search_rotated(nums, target):
    """LC 33: sorted, then rotated.  The array is not monotone, but at every mid
    ONE of the two halves is, and 'is target in that half' is decidable - so the
    same loop still discards half the array per step."""
    lo, hi = 0, len(nums) - 1
    while lo <= hi:
        mid = lo + (hi - lo) // 2
        if nums[mid] == target:
            return mid
        if nums[lo] <= nums[mid]:                       # left half is sorted
            if nums[lo] <= target < nums[mid]:
                hi = mid - 1
            else:
                lo = mid + 1
        else:                                           # right half is sorted
            if nums[mid] < target <= nums[hi]:
                lo = mid + 1
            else:
                hi = mid - 1
    return -1


def section_leetcode():
    print("=" * 72)
    print("8. LC 34 (first and last position) and LC 33 (rotated array)")
    print("=" * 72)
    nums = [5, 7, 7, 8, 8, 8, 10]
    for t in [8, 7, 6, 5, 10]:
        got = search_range(nums, t)
        exp = ([nums.index(t), len(nums) - 1 - nums[::-1].index(t)]
               if t in nums else [-1, -1])
        assert got == exp, (t, got, exp)
        print(f"  search_range({nums}, {t}) = {got}")
    print()
    rot = [4, 5, 6, 7, 0, 1, 2]
    for t in [0, 4, 2, 3, 7]:
        got = search_rotated(rot, t)
        exp = rot.index(t) if t in rot else -1
        assert got == exp, (t, got, exp)
        print(f"  search_rotated({rot}, {t}) = {got:>2}"
              f"   ({'found' if got >= 0 else 'absent'})")
    print()
    print("  LC 34 is the whole point of having two boundaries: with lower_bound")
    print("  and upper_bound it is four tokens of logic, and the 'not found' case")
    print("  falls out of lo == hi instead of needing its own branch.")
    print("  LC 33 is the reminder that 'sorted' is stronger than binary search")
    print("  needs.  All the loop requires is that each step can rule out one side.")
    print()


def main():
    section_boundaries()
    section_broken()
    section_answer_space()
    section_noisy_predicate()
    section_gallop()
    section_cu_seqlens()
    section_buckets()
    section_leetcode()
    print("=" * 72)
    print("all assertions passed")
    print("=" * 72)


if __name__ == "__main__":
    main()
