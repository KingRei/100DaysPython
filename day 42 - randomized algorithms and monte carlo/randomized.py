# -*- coding: utf-8 -*-
"""Day 42 - Randomized algorithms and Monte Carlo.

Run it:  python randomized.py

Throw a million random darts at a square and count how many land inside the
quarter circle: four times that fraction is pi to about three decimals.  That
is Monte Carlo - an answer that is only approximately right, whose error
shrinks like 1 / sqrt(N): every extra correct digit costs a hundred times
more darts.  Its sibling, Las Vegas, is always right but takes a random amount
of time (quicksort with a random pivot).

Randomized code has one nasty property: when it is wrong it does not crash.
It still returns a random-looking answer - just drawn from the wrong
distribution.  Every bug below passes the examples:

* swapping each slot with a random slot anywhere ("naive shuffle") makes some
  orders of [1, 2, 3] appear 5 times in 27 and others 4 times in 27;
* writing randrange(i) instead of randrange(i + 1) in Fisher-Yates never
  leaves any element where it was - only 2 of the 6 orders of 3 elements;
* the same off-by-one in reservoir sampling (LeetCode 382) never returns the
  first node;
* (rand7 index) % 10 without rejection (LeetCode 470) gives 1 a 4/49 chance
  and the other nine digits 5/49;
* randint(0, total) instead of randrange(total) in LeetCode 528 doubles the
  chance of a weight-1 item next to a weight-99 one.

The only way to test randomness is to count, so section 3 builds a tool that
runs an algorithm once for every possible sequence of random choices and adds
up the exact probabilities as fractions.

Sections
  1. Monte Carlo: estimating pi, and the 1 / sqrt(N) law
  2. Las Vegas vs Monte Carlo: random-pivot quicksort; Fermat vs Miller-Rabin
  3. Testing randomness exactly: enumerate every random choice
  4. Shuffling: naive, the randrange(i) slip, Fisher-Yates; LeetCode 384
  5. Reservoir sampling: LeetCode 382, LeetCode 398
  6. Rejection sampling: LeetCode 470
  7. Weighted pick by prefix sums: LeetCode 528
"""
from __future__ import annotations

import math
import random
import time
from bisect import bisect_left, bisect_right
from collections import Counter, defaultdict
from fractions import Fraction
from itertools import accumulate, permutations


def header(title: str) -> None:
    print('\n' + '=' * 72 + f'\n{title}\n' + '=' * 72)


# ------------------------------------------------------------ 1. Monte Carlo pi
def estimate_pi(n: int, rng: random.Random) -> float:
    """Throw n darts at the unit square; the quarter circle covers pi/4 of it."""
    inside = 0
    for _ in range(n):
        x, y = rng.random(), rng.random()
        if x * x + y * y < 1.0:
            inside += 1
    return 4 * inside / n


P_IN = math.pi / 4
SIGMA_PI = 4 * math.sqrt(P_IN * (1 - P_IN))       # std of ONE dart's 4*[inside]


def pi_std(n: int) -> float:
    """Standard error of estimate_pi(n): 1.642 / sqrt(n)."""
    return SIGMA_PI / math.sqrt(n)


def darts_for_error(err: float) -> float:
    """Darts needed so that one standard error equals err."""
    return (SIGMA_PI / err) ** 2


# --------------------------------------------- 2a. Las Vegas: random pivots
def quicksort_comparisons(a: list, rng: random.Random | None) -> tuple[list, int]:
    """Lomuto quicksort with an explicit stack.  rng=None -> first element as
    pivot; otherwise a random pivot.  Returns (sorted copy, comparisons)."""
    a = list(a)
    comps = 0
    stack = [(0, len(a) - 1)]
    while stack:
        lo, hi = stack.pop()
        if lo >= hi:
            continue
        if rng is not None:
            k = rng.randint(lo, hi)
            a[lo], a[k] = a[k], a[lo]
        pivot, i = a[lo], lo
        for j in range(lo + 1, hi + 1):
            comps += 1
            if a[j] < pivot:
                i += 1
                a[i], a[j] = a[j], a[i]
        a[lo], a[i] = a[i], a[lo]
        stack.append((lo, i - 1))
        stack.append((i + 1, hi))
    return a, comps


def expected_quicksort_comparisons(n: int) -> float:
    """Exact mean for a uniformly random pivot: 2(n+1)H_n - 4n."""
    h = sum(1 / k for k in range(1, n + 1))
    return 2 * (n + 1) * h - 4 * n


# ------------------------------------------- 2b. Monte Carlo: primality tests
def fermat_passes(n: int, a: int) -> bool:
    """Fermat: if n is prime then a^(n-1) = 1 (mod n).  The converse is false."""
    return pow(a, n - 1, n) == 1


def miller_rabin_passes(n: int, a: int) -> bool:
    """One Miller-Rabin round with base a.  False means n is certainly composite."""
    d, s = n - 1, 0
    while d % 2 == 0:
        d //= 2
        s += 1
    x = pow(a, d, n)
    if x == 1 or x == n - 1:
        return True
    for _ in range(s - 1):
        x = x * x % n
        if x == n - 1:
            return True
    return False                 # a square root of 1 other than +-1: composite


def is_probable_prime(n: int, rounds: int, rng: random.Random) -> bool:
    """Monte Carlo: always says True for a prime; wrong on a composite with
    probability at most (1/4)^rounds."""
    if n < 4:
        return n in (2, 3)
    if n % 2 == 0:
        return False
    return all(miller_rabin_passes(n, rng.randrange(2, n - 1)) for _ in range(rounds))


SMALL_PRIMES = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37)


def is_prime(n: int) -> bool:
    """Deterministic for n < 3.18 * 10**23 (every 64-bit integer): with the twelve
    primes 2..37 as bases no composite below that bound passes."""
    if n < 2:
        return False
    for p in SMALL_PRIMES:
        if n % p == 0:
            return n == p
    return all(miller_rabin_passes(n, a) for a in SMALL_PRIMES)


def sieve(n: int) -> bytearray:
    s = bytearray([1]) * n
    s[:2] = b'\x00\x00'
    for p in range(2, math.isqrt(n - 1) + 1):
        if s[p]:
            s[p * p::p] = bytes(len(range(p * p, n, p)))
    return s


def liars(n: int) -> tuple[int, int, int]:
    """(bases tried, Fermat liars, Miller-Rabin liars) over bases 2..n-2."""
    bases = range(2, n - 1)
    return (len(bases), sum(fermat_passes(n, a) for a in bases),
            sum(miller_rabin_passes(n, a) for a in bases))


def worst_miller_rabin(limit: int) -> tuple[float, int]:
    """(largest fraction of lying bases, n) over the odd composites below limit."""
    best = (0.0, 0)
    for m in range(9, limit, 2):
        if not is_prime(m):
            tried, _, mr = liars(m)
            best = max(best, (mr / tried, m))
    return best


# ------------------------------------------ 3. exact distribution of a sampler
def exact_distribution(run) -> dict:
    """run(randrange) must call randrange(k) for its random choices and return
    a hashable result.  Every possible sequence of choices is replayed once
    (an odometer over the choices) and the probabilities are added up as exact
    fractions - no sampling, no noise, so a bias of 1/27 cannot hide."""
    dist: dict = defaultdict(Fraction)
    prefix: list[int] = []
    while True:
        radices: list[int] = []

        def randrange(k: int) -> int:
            pos = len(radices)
            radices.append(k)
            if pos == len(prefix):
                prefix.append(0)
            return prefix[pos]

        out = run(randrange)
        p = Fraction(1)
        for k in radices:
            p /= k
        dist[out] += p
        del prefix[len(radices):]
        while prefix and prefix[-1] + 1 == radices[len(prefix) - 1]:
            prefix.pop()
        if not prefix:
            return dict(dist)
        prefix[-1] += 1


def frac_str(f: Fraction) -> str:
    return '0' if f == 0 else f'{f.numerator}/{f.denominator}'


# --------------------------------------------------------------- 4. shuffling
def shuffle_naive(a: list, randrange) -> list:
    """Swap every slot with a random slot ANYWHERE: n**n equally likely runs."""
    a = list(a)
    n = len(a)
    for i in range(n):
        j = randrange(n)
        a[i], a[j] = a[j], a[i]
    return a


def shuffle_off_by_one(a: list, randrange) -> list:
    """Fisher-Yates with randrange(i) instead of randrange(i + 1): j < i, so the
    element at i can never stay - this is Sattolo's algorithm (one big cycle)."""
    a = list(a)
    for i in range(len(a) - 1, 0, -1):
        j = randrange(i)
        a[i], a[j] = a[j], a[i]
    return a


def fisher_yates(a: list, randrange) -> list:
    """Fill slots from the back; slot i gets a uniform pick of the i+1 still free."""
    a = list(a)
    for i in range(len(a) - 1, 0, -1):
        j = randrange(i + 1)           # 0..i inclusive - i itself is allowed
        a[i], a[j] = a[j], a[i]
    return a


def shuffle_report(fn, n: int) -> tuple[Counter, int]:
    """How many of the equally likely random runs produce each order."""
    dist = exact_distribution(lambda rr: tuple(fn(range(n), rr)))
    runs = math.lcm(*(f.denominator for f in dist.values()))
    return Counter({p: int(f * runs) for p, f in dist.items()}), runs


class Solution384:
    """LeetCode 384 - Shuffle an Array."""

    def __init__(self, nums: list[int]):
        self.original = list(nums)
        self.rng = random.Random()

    def reset(self) -> list[int]:
        return list(self.original)

    def shuffle(self) -> list[int]:
        a = list(self.original)
        for i in range(len(a) - 1, 0, -1):
            j = self.rng.randrange(i + 1)
            a[i], a[j] = a[j], a[i]
        return a


# ------------------------------------------------------ 5. reservoir sampling
def reservoir_one(stream, randrange):
    """Keep the i-th item (1-based) with probability 1/i.  One pass, O(1) memory,
    and the length of the stream is never needed."""
    chosen = None
    for i, x in enumerate(stream, 1):
        if randrange(i) == 0:          # 1 chance in i
            chosen = x
    return chosen


def reservoir_off_by_one(stream, randrange):
    """Same loop with a 0-based i: item i is kept with probability 1/i, so the
    first item (i = 0) is skipped and the second (i = 1) ALWAYS replaces it."""
    chosen = None
    for i, x in enumerate(stream):
        if i == 0:
            chosen = x
            continue
        if randrange(i) == 0:          # should be randrange(i + 1)
            chosen = x
    return chosen


def reservoir_k(stream, k: int, randrange) -> list:
    """Algorithm R: keep k items; item i (0-based, i >= k) enters with chance k/(i+1)."""
    res = []
    for i, x in enumerate(stream):
        if i < k:
            res.append(x)
        else:
            j = randrange(i + 1)
            if j < k:
                res[j] = x
    return res


class ListNode:
    def __init__(self, val: int = 0, next: 'ListNode | None' = None):
        self.val, self.next = val, next


def build_list(vals) -> ListNode | None:
    head = None
    for v in reversed(vals):
        head = ListNode(v, head)
    return head


class Solution382:
    """LeetCode 382 - Linked List Random Node (reservoir, no length needed)."""

    def __init__(self, head: ListNode | None):
        self.head = head
        self.rng = random.Random()

    def getRandom(self) -> int:
        chosen, i, node = 0, 0, self.head
        while node:
            i += 1
            if self.rng.randrange(i) == 0:
                chosen = node.val
            node = node.next
        return chosen


class Solution398:
    """LeetCode 398 - Random Pick Index: reservoir over the matching positions."""

    def __init__(self, nums: list[int]):
        self.nums = nums
        self.rng = random.Random()

    def pick(self, target: int) -> int:
        chosen, seen = -1, 0
        for i, v in enumerate(self.nums):
            if v == target:
                seen += 1
                if self.rng.randrange(seen) == 0:
                    chosen = i
        return chosen


# ------------------------------------------------------ 6. rejection sampling
def rand10_from_rand7(rand7) -> tuple[int, int]:
    """Two rand7 calls give 49 equally likely cells; keep the first 40 (four
    copies of 1..10) and throw the other 9 away.  Returns (digit, calls)."""
    calls = 0
    while True:
        idx = (rand7() - 1) * 7 + rand7()       # 1..49, uniform
        calls += 2
        if idx <= 40:
            return (idx - 1) % 10 + 1, calls


def rand10_mod_bias(rand7) -> int:
    """No rejection: 49 is not a multiple of 10, so one digit is short a cell."""
    idx = (rand7() - 1) * 7 + rand7()
    return idx % 10 + 1


def rand10_reuse(rand7) -> tuple[int, int]:
    """Keep the leftover randomness of a rejected draw instead of discarding it."""
    calls = 0
    while True:
        a = (rand7() - 1) * 7 + rand7()          # 1..49
        calls += 2
        if a <= 40:
            return (a - 1) % 10 + 1, calls
        a = (a - 40 - 1) * 7 + rand7()           # 1..63
        calls += 1
        if a <= 60:
            return (a - 1) % 10 + 1, calls
        a = (a - 60 - 1) * 7 + rand7()           # 1..21
        calls += 1
        if a <= 20:
            return (a - 1) % 10 + 1, calls


def rand10_one_round(randrange) -> int | None:
    """One round of the rejection sampler, for exact_distribution (None = reject)."""
    idx = randrange(7) * 7 + randrange(7) + 1
    return (idx - 1) % 10 + 1 if idx <= 40 else None


class Solution470:
    """LeetCode 470 - Implement Rand10() Using Rand7()."""

    def __init__(self, rand7):
        self.rand7 = rand7

    def rand10(self) -> int:
        while True:
            idx = (self.rand7() - 1) * 7 + self.rand7()
            if idx <= 40:
                return (idx - 1) % 10 + 1


# ---------------------------------------------- 7. weighted pick, prefix sums
class Solution528:
    """LeetCode 528 - Random Pick with Weight.  O(n) build, O(log n) per pick."""

    def __init__(self, w: list[int], rng: random.Random | None = None):
        self.prefix = list(accumulate(w))        # w=[1,3] -> [1, 4]
        self.rng = rng or random.Random()

    def pickIndex(self) -> int:
        target = self.rng.randrange(self.prefix[-1])   # 0..total-1
        return bisect_right(self.prefix, target)       # first prefix > target


def pick_index_exact(w, randrange) -> int:
    prefix = list(accumulate(w))
    return bisect_right(prefix, randrange(prefix[-1]))


def pick_index_off_by_one(w, randrange) -> int:
    """randint(0, total) is total + 1 values - one too many, and the extra one
    (target 0) lands on index 0."""
    prefix = list(accumulate(w))
    return bisect_left(prefix, randrange(prefix[-1] + 1))


# ======================================================================= main
def main() -> None:
    rng = random.Random(42)

    header('1. Monte Carlo: estimating pi')
    for n in (100, 10_000, 1_000_000):
        est = estimate_pi(n, rng)
        print(f'  {n:>9,} darts: pi ~ {est:.6f}   error {abs(est - math.pi):.6f}'
              f'   one standard error = {pi_std(n):.6f}')
    print('error over many runs (root-mean-square) vs the 1.642 / sqrt(N) law:')
    for n, trials in ((100, 400), (1_000, 200), (10_000, 100), (100_000, 20)):
        errs = [estimate_pi(n, rng) - math.pi for _ in range(trials)]
        rms = math.sqrt(sum(e * e for e in errs) / trials)
        print(f'  N = {n:>7,}  ({trials:>3} runs): rms error {rms:.5f}   law {pi_std(n):.5f}')
    print(f'  100x the darts -> 10x smaller error.  6 correct decimals (error 5e-7) needs '
          f'{darts_for_error(5e-7):.2e} darts')
    r1, r2 = random.Random(7), random.Random(7)
    print(f'  same seed, same darts: {estimate_pi(1000, r1)} == {estimate_pi(1000, r2)}')

    header('2. Las Vegas vs Monte Carlo')
    n = 2000
    data = list(range(n))
    _, first = quicksort_comparisons(data, None)
    runs = [quicksort_comparisons(data, random.Random(s))[1] for s in range(200)]
    print(f'quicksort on an already sorted list of {n:,}:')
    print(f'  first element as pivot: {first:,} comparisons (= n(n-1)/2 = {n*(n-1)//2:,})')
    print(f'  random pivot, 200 runs: mean {sum(runs)/len(runs):,.0f}, min {min(runs):,}, '
          f'max {max(runs):,}   exact mean {expected_quicksort_comparisons(n):,.0f}')
    print('  Las Vegas: the output is always sorted, only the running time is random')
    tried, f_liars, mr_liars = liars(561)
    print(f'561 = 3 x 11 x 17 (a Carmichael number), bases 2..559:')
    print(f'  Fermat says "prime" for {f_liars} of {tried} bases ({f_liars/tried:.0%})')
    print(f'  Miller-Rabin says "prime" for {mr_liars} of {tried} ({mr_liars/tried:.1%}); '
          f'the bound is 1/4')
    worst = worst_miller_rabin(2000)
    print(f'  worst odd composite below 2000 for Miller-Rabin: n = {worst[1]}, '
          f'{worst[0]:.1%} of bases lie')
    s = sieve(100_000)
    pseudo2 = [m for m in range(3, 100_000, 2) if not s[m] and fermat_passes(m, 2)]
    print(f'below 100,000: {len(pseudo2)} odd composites pass Fermat with base 2 '
          f'(first: {pseudo2[:5]})')
    mr = random.Random(1)   # its own stream, so the count does not depend on earlier sections
    wrong1 = sum(is_probable_prime(m, 1, mr) != bool(s[m]) for m in range(100_000))
    wrong5 = sum(is_probable_prime(m, 5, mr) != bool(s[m]) for m in range(100_000))
    print(f'  Miller-Rabin with ONE random base: {wrong1} wrong answers below 100,000; '
          f'5 bases: {wrong5}')
    big = 2**61 - 1
    print(f'  is_prime(2**61 - 1) = {is_prime(big)}  ({len(str(big))} digits, 12 fixed bases, '
          f'no factoring)')

    header('3. Testing randomness exactly')
    coin = exact_distribution(lambda rr: rr(2) + rr(2))
    print(f'two coin flips, heads counted: {{{", ".join(f"{k}: {frac_str(v)}" for k, v in sorted(coin.items()))}}}')
    print('  every possible sequence of random choices is replayed once and weighted by')
    print('  its probability - exact fractions, so a 1/27 bias cannot hide in noise')

    header('4. Shuffling')
    for name, fn in (('naive (swap with any slot)', shuffle_naive),
                     ('randrange(i) slip', shuffle_off_by_one),
                     ('Fisher-Yates', fisher_yates)):
        cnt, runs = shuffle_report(fn, 3)
        row = '  '.join(f"{''.join('abc'[i] for i in p)}:{cnt.get(p, 0)}"
                        for p in permutations(range(3)))
        print(f'  {name:<27} {runs:>2} runs -> {row}')
    cnt, runs = shuffle_report(shuffle_naive, 4)
    print(f'  naive on 4 items: {runs} runs over 24 orders -> each order {min(cnt.values())} to '
          f'{max(cnt.values())} runs ({max(cnt.values())/min(cnt.values()):.2f}x); '
          f'{runs} / 24 is not a whole number')
    cnt, runs = shuffle_report(shuffle_off_by_one, 4)
    fixed = sum(1 for p in cnt if any(p[i] == i for i in range(4)))
    print(f'  randrange(i) slip on 4 items: reaches {len(cnt)} of 24 orders, '
          f'{fixed} with any item left in place')
    big = Counter()
    sim = random.Random(1)
    for _ in range(60_000):
        big[tuple(shuffle_naive(range(3), sim.randrange))] += 1
    print('  naive, 60,000 real shuffles: ' + '  '.join(
        f"{''.join('abc'[i] for i in p)}:{big[p]}" for p in permutations(range(3))))
    s = Solution384([1, 2, 3])
    s.rng.seed(3)
    print(f'LC 384: shuffle {s.shuffle()}, reset {s.reset()}, shuffle {s.shuffle()}')

    header('5. Reservoir sampling')
    items = list('abcde')
    for name, fn in (('reservoir 1/i', reservoir_one), ('randrange(i) slip', reservoir_off_by_one)):
        d = exact_distribution(lambda rr, fn=fn: fn(items, rr))
        print(f'  {name:<18} ' + '  '.join(f'{x}:{frac_str(d.get(x, Fraction(0)))}' for x in items))
    d = exact_distribution(lambda rr: frozenset(reservoir_k(items, 2, rr)))
    print(f'  k = 2 from 5: {len(d)} pairs, each {frac_str(next(iter(set(d.values()))))}'
          f' (= 1 / C(5, 2))')
    s = Solution382(build_list([1, 2, 3]))
    s.rng.seed(5)
    got = Counter(s.getRandom() for _ in range(30_000))
    print(f'LC 382 [1, 2, 3], 30,000 calls: {dict(sorted(got.items()))}')
    slip = random.Random(5)
    got = Counter(reservoir_off_by_one([1, 2, 3], slip.randrange) for _ in range(30_000))
    print(f'       the randrange(i) slip:    {dict(sorted(got.items()))}  <- 1 never comes back')
    s = Solution398([1, 2, 3, 3, 3])
    s.rng.seed(9)
    got = Counter(s.pick(3) for _ in range(30_000))
    print(f'LC 398 [1, 2, 3, 3, 3] pick(3), 30,000 calls: {dict(sorted(got.items()))}, pick(1) = {s.pick(1)}')

    header('6. Rejection sampling')
    one = exact_distribution(rand10_one_round)
    acc = sum(v for k, v in one.items() if k is not None)
    print(f'one round of 49 cells: accept {frac_str(acc)}, each digit '
          f'{frac_str(one[1] / acc)} once accepted')
    bias = exact_distribution(lambda rr: rand10_mod_bias(lambda: rr(7) + 1))
    print('no rejection, idx % 10 + 1: ' + '  '.join(f'{k}:{frac_str(v)}' for k, v in sorted(bias.items())))
    print(f'expected rand7 calls: 2 x 49/40 = {2*49/40}')
    sim = random.Random(11)
    r7 = lambda: sim.randint(1, 7)
    c1 = [rand10_from_rand7(r7)[1] for _ in range(200_000)]
    c2 = [rand10_reuse(r7)[1] for _ in range(200_000)]
    print(f'  measured over 200,000 calls: {sum(c1)/len(c1):.3f};  reusing the rejected '
          f'leftovers: {sum(c2)/len(c2):.3f}')
    s = Solution470(r7)
    got = Counter(s.rand10() for _ in range(100_000))
    print(f'LC 470, 100,000 calls: min count {min(got.values()):,}, max {max(got.values()):,} '
          f'(expect 10,000 each)')

    header('7. Weighted pick: prefix sums + binary search')
    for w in ([1, 3], [1, 99]):
        good = exact_distribution(lambda rr, w=w: pick_index_exact(w, rr))
        bad = exact_distribution(lambda rr, w=w: pick_index_off_by_one(w, rr))
        print(f'  w = {w}: prefix {list(accumulate(w))}  correct ' +
              '  '.join(frac_str(good[i]) for i in range(len(w))) +
              '   randint(0, total) + bisect_left ' +
              '  '.join(frac_str(bad[i]) for i in range(len(w))))
    s = Solution528([1, 3], random.Random(13))
    got = Counter(s.pickIndex() for _ in range(40_000))
    print(f'LC 528 [1, 3], 40,000 picks: {dict(sorted(got.items()))}')
    vocab = {'the': 2.0, 'a': 1.0, 'cat': 0.1}
    print('an LLM picks its next token with exactly this prefix-sum + bisect step, over')
    print('softmax(logits / temperature); logits the 2.0, a 1.0, cat 0.1:')
    for temp in (0.5, 1.0, 2.0):
        z = [math.exp(v / temp) for v in vocab.values()]
        print(f'  temperature {temp}: ' + '  '.join(
            f'{t} {x / sum(z):.3f}' for t, x in zip(vocab, z)))

    run_tests()
    print('\nall assertions passed')


# ====================================================================== tests
def run_tests() -> None:
    rng = random.Random(0)
    for n in range(1, 7):
        cnt, runs = shuffle_report(fisher_yates, n)
        assert len(cnt) == math.factorial(n) and set(cnt.values()) == {1}
        cnt, _ = shuffle_report(shuffle_off_by_one, n)
        assert len(cnt) == math.factorial(n - 1)
        assert all(all(p[i] != i for i in range(n)) for p in cnt) or n == 1
    cnt, runs = shuffle_report(shuffle_naive, 3)
    assert runs == 27 and sorted(cnt.values()) == [4, 4, 4, 5, 5, 5]
    for n in range(1, 7):
        d = exact_distribution(lambda rr: reservoir_one(range(n), rr))
        assert all(v == Fraction(1, n) for v in d.values()) and len(d) == n
    d = exact_distribution(lambda rr: reservoir_off_by_one(range(5), rr))
    assert 0 not in d and all(v == Fraction(1, 4) for v in d.values())
    for n in range(2, 7):
        for k in range(1, n + 1):
            d = exact_distribution(lambda rr: frozenset(reservoir_k(range(n), k, rr)))
            assert len(d) == math.comb(n, k) and set(d.values()) == {Fraction(1, math.comb(n, k))}
    one = exact_distribution(rand10_one_round)
    assert one[None] == Fraction(9, 49) and all(one[k] == Fraction(4, 49) for k in range(1, 11))
    bias = exact_distribution(lambda rr: rand10_mod_bias(lambda: rr(7) + 1))
    assert bias[1] == Fraction(4, 49) and all(bias[k] == Fraction(5, 49) for k in range(2, 11))
    for w in ([1], [1, 3], [2, 5, 3], [1, 99], [4, 4, 4]):
        d = exact_distribution(lambda rr: pick_index_exact(w, rr))
        assert all(d.get(i, 0) == Fraction(w[i], sum(w)) for i in range(len(w)))
    d = exact_distribution(lambda rr: pick_index_off_by_one([1, 99], rr))
    assert d[0] == Fraction(2, 101)
    s = sieve(50_000)
    assert all(is_prime(m) == bool(s[m]) for m in range(50_000))
    assert is_prime(2**61 - 1) and not is_prime(2**61 + 1) and is_prime(2**89 - 1)
    assert all(fermat_passes(561, a) for a in range(2, 560) if math.gcd(a, 561) == 1)
    for m in range(9, 600, 2):
        if not s[m]:
            t, _, mr = liars(m)
            assert mr * 4 <= t + 2         # Rabin's bound, bases 2..n-2
    data = list(range(300))
    rng.shuffle(data)
    for r in (None, random.Random(1)):
        out, c = quicksort_comparisons(data, r)
        assert out == sorted(data)
    assert quicksort_comparisons(list(range(100)), None)[1] == 100 * 99 // 2
    s = Solution384([1, 2, 3, 4])
    assert sorted(s.shuffle()) == [1, 2, 3, 4] and s.reset() == [1, 2, 3, 4]
    s = Solution382(build_list([7]))
    assert s.getRandom() == 7
    s = Solution398([1, 2, 3, 3, 3])
    assert s.pick(1) == 0 and s.pick(3) in (2, 3, 4)
    s = Solution470(lambda: rng.randint(1, 7))
    assert all(1 <= s.rand10() <= 10 for _ in range(1000))
    s = Solution528([1, 3])
    assert all(s.pickIndex() in (0, 1) for _ in range(100))


if __name__ == '__main__':
    main()
