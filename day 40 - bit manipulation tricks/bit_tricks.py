# -*- coding: utf-8 -*-
"""Day 40 - Bit manipulation: masks, popcount, and integers that have no width.

Run it:  python bit_tricks.py

A Python int is a two's-complement number with infinitely many bits: a negative
number is a 1 in every position from some point upward, forever.  That one fact
splits the classic bit tricks into two groups:

* &  |  ^  ~  work column by column, so they give the same answer at every
  width - x & -x, x & (x - 1), XOR-ing out pairs, the ones/twos state machine.
* anything that asks "how many bits?" or "which one is the top bit?" needs a
  width, and Python never stores one - popcount of a negative number, a carry
  that should fall off bit 31, a per-column count that rebuilds bit 31.

Those second ones do not crash.  bin(-5).count('1') is 2 where C and Java say 31,
LeetCode 371's classic loop returns 4294967294 for -1 + -1 once it is masked and
never returns at all before that, and LeetCode 137's bit-count answer passes
both examples and fails every test whose answer is negative.  Each one is below,
next to the one line that pins the width.

Sections
  1. Python integers have no width (u32 / s32)
  2. The toolkit on x = 180, and a precedence trap
  3. popcount five ways - and why the clever one loses in Python
  4. popcount of a negative number
  5. LeetCode 338 - counting bits for 0..n without popcount
  6. A Python int as a set of a million members
  7. Gosper's hack - every k-subset in order
  8. LeetCode 136 / 137 - what XOR can and cannot cancel
  9. LeetCode 371 - addition without +, and the carry with nowhere to fall off
 10. LeetCode 191 / 231
"""
from __future__ import annotations

import math
import random
import sys
import time
from functools import reduce
from operator import xor

W32 = 32
M32 = (1 << 32) - 1            # 0xFFFFFFFF: the 32 bits a C int would keep


# ------------------------------------------------------------------ 1. width
def bits(x: int, width: int = 8) -> str:
    """The low `width` bits of x as a string - what a width-bit register holds."""
    return format(x & ((1 << width) - 1), f'0{width}b')


def u32(x: int) -> int:
    """Keep the low 32 bits: the unsigned value a C uint32_t would hold."""
    return x & M32


def s32(u: int) -> int:
    """Read 32 bits as a signed int: bit 31 is worth -2**31, not +2**31."""
    u &= M32
    return u - (1 << 32) if u >> 31 else u


# ---------------------------------------------------------------- 2. toolkit
def test_bit(x: int, k: int) -> int:
    return (x >> k) & 1


def set_bit(x: int, k: int) -> int:
    return x | (1 << k)


def clear_bit(x: int, k: int) -> int:
    return x & ~(1 << k)


def toggle_bit(x: int, k: int) -> int:
    return x ^ (1 << k)


def lowbit(x: int) -> int:
    """The lowest set bit alone (Fenwick trees, Day 39's bitmask N-Queens)."""
    return x & -x


def clear_lowest(x: int) -> int:
    """x with its lowest set bit turned off."""
    return x & (x - 1)


def set_lowest_zero(x: int) -> int:
    """x with its lowest 0 bit turned on."""
    return x | (x + 1)


def trailing_zeros(x: int) -> int:
    """How many 0s sit below the lowest 1 (x > 0)."""
    return (x & -x).bit_length() - 1


def gray(i: int) -> int:
    """Gray code: consecutive values differ in exactly one bit."""
    return i ^ (i >> 1)


TOOLKIT = [
    ('x >> 2 & 1',   'test bit 2',                lambda x: test_bit(x, 2)),
    ('x | 1 << 3',   'set bit 3',                 lambda x: set_bit(x, 3)),
    ('x & ~(1 << 4)', 'clear bit 4',              lambda x: clear_bit(x, 4)),
    ('x ^ 1 << 7',   'toggle bit 7',              lambda x: toggle_bit(x, 7)),
    ('x & -x',       'keep only the lowest 1',    lowbit),
    ('x & (x - 1)',  'drop the lowest 1',         clear_lowest),
    ('x | (x + 1)',  'turn on the lowest 0',      set_lowest_zero),
    ('x ^ (x >> 1)', 'Gray code of x',            gray),
]


def full_mask_buggy(n: int) -> int:
    """Meant to be n ones. `-` binds tighter than `<<`, so this is 1 << (n - 1)."""
    return 1 << n - 1


def full_mask(n: int) -> int:
    return (1 << n) - 1


# --------------------------------------------------------------- 3. popcount
def popcount_bin(x: int) -> int:
    return bin(x).count('1')


def popcount_builtin(x: int) -> int:
    return x.bit_count()                  # Python 3.10+


def popcount_shift(x: int) -> int:
    """Look at every bit: the loop runs bit_length(x) times."""
    count = 0
    while x:
        count += x & 1
        x >>= 1
    return count


def popcount_kernighan(x: int) -> int:
    """Each x &= x - 1 removes the lowest 1: the loop runs popcount(x) times."""
    count = 0
    while x:
        x &= x - 1
        count += 1
    return count


def popcount_swar32(x: int) -> int:
    """Add the bits in parallel inside one word: pairs, then nibbles, then bytes."""
    x = x - ((x >> 1) & 0x55555555)                   # 16 two-bit counts
    x = (x & 0x33333333) + ((x >> 2) & 0x33333333)    # 8 four-bit counts
    x = (x + (x >> 4)) & 0x0F0F0F0F                   # 4 byte counts
    return ((x * 0x01010101) & M32) >> 24             # sum of the 4 bytes


BYTE_TABLE = [bin(i).count('1') for i in range(256)]


def popcount_table32(x: int) -> int:
    return (BYTE_TABLE[x & 255] + BYTE_TABLE[(x >> 8) & 255]
            + BYTE_TABLE[(x >> 16) & 255] + BYTE_TABLE[(x >> 24) & 255])


def popcount32(x: int) -> int:
    """What C's __builtin_popcount / Java's Integer.bitCount return, negatives too."""
    return (x & M32).bit_count()


POPCOUNTS = [('bin(x).count("1")', popcount_bin),
             ('x.bit_count()', popcount_builtin),
             ('shift loop', popcount_shift),
             ("Kernighan x &= x-1", popcount_kernighan),
             ('SWAR (32-bit)', popcount_swar32),
             ('byte table', popcount_table32)]


def swar_stages(x: int) -> list[int]:
    """The word after each SWAR step, for the figure and the demo."""
    a = x - ((x >> 1) & 0x55555555)
    b = (a & 0x33333333) + ((a >> 2) & 0x33333333)
    c = (b + (b >> 4)) & 0x0F0F0F0F
    return [x, a, b, c, ((c * 0x01010101) & M32) >> 24]


def kernighan_trace(x: int, cap: int = 64) -> list[int]:
    """Every value x takes under x &= x - 1, stopping at 0 or after `cap` steps."""
    out = [x]
    while x and len(out) <= cap:
        x &= x - 1
        out.append(x)
    return out


def best_time(fn, repeat: int = 5) -> float:
    best = float('inf')
    for _ in range(repeat):
        t = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t)
    return best


# ------------------------------------------------------------ 5. LeetCode 338
class Solution338:
    def countBits(self, n: int) -> list[int]:
        ans = [0] * (n + 1)
        for i in range(1, n + 1):
            ans[i] = ans[i >> 1] + (i & 1)      # i without its last bit, plus that bit
        return ans


def count_bits_lowbit(n: int) -> list[int]:
    """Same table, other recurrence: i has one more 1 than i with its lowest 1 dropped."""
    ans = [0] * (n + 1)
    for i in range(1, n + 1):
        ans[i] = ans[i & (i - 1)] + 1
    return ans


# ------------------------------------------------------------- 6. bitsets
def to_bitset(values, universe: int) -> int:
    """Set bit v for every v: fill a bytearray, convert once. Linear time."""
    buf = bytearray((universe + 7) // 8)
    for v in values:
        buf[v >> 3] |= 1 << (v & 7)
    return int.from_bytes(buf, 'little')


def to_bitset_slow(values) -> int:
    """Looks the same, is quadratic: every |= builds a brand-new big integer."""
    x = 0
    for v in values:
        x |= 1 << v
    return x


def members(x: int) -> list[int]:
    """Indices of the set bits, lowest first (fine for small or sparse x)."""
    out = []
    while x:
        low = x & -x
        out.append(low.bit_length() - 1)
        x ^= low
    return out


def jaccard_bits(a: int, b: int) -> float:
    return (a & b).bit_count() / (a | b).bit_count()


# ------------------------------------------------------------- 7. Gosper
def next_same_popcount(x: int) -> int:
    """Gosper's hack: the next larger integer with the same number of 1s."""
    c = x & -x                   # lowest 1
    r = x + c                    # carry it into the block of 1s above it
    return (((r ^ x) >> 2) // c) | r   # put the leftover 1s back at the bottom


def k_subsets(n: int, k: int) -> list[int]:
    """Every n-bit mask with exactly k ones, in increasing order."""
    if k == 0:
        return [0]
    out, x = [], (1 << k) - 1
    while x < 1 << n:
        out.append(x)
        x = next_same_popcount(x)
    return out


# ------------------------------------------------------- 8. LeetCode 136 / 137
class Solution136:
    def singleNumber(self, nums: list[int]) -> int:
        return reduce(xor, nums, 0)             # pairs cancel, column by column


class Solution137:
    def singleNumber(self, nums: list[int]) -> int:
        ones = twos = 0                         # per column: seen once / seen twice (mod 3)
        for x in nums:
            ones = (ones ^ x) & ~twos
            twos = (twos ^ x) & ~ones
        return ones


def single_number_ii_count(nums: list[int]) -> int:
    """Count each of 32 columns mod 3 - then read bit 31 as the sign."""
    res = 0
    for k in range(32):
        if sum((x >> k) & 1 for x in nums) % 3:
            res |= 1 << k
    return s32(res)


def single_number_ii_count_nosign(nums: list[int]) -> int:
    """Same loop, last line forgotten: bit 31 comes back worth +2**31."""
    res = 0
    for k in range(32):
        if sum((x >> k) & 1 for x in nums) % 3:
            res |= 1 << k
    return res


# ------------------------------------------------------------ 9. LeetCode 371
def get_sum_naive(a: int, b: int, cap: int | None = None):
    """The C answer typed into Python. Returns None if `cap` rounds pass."""
    rounds = 0
    while b:
        if cap is not None and rounds == cap:
            return None
        a, b = a ^ b, (a & b) << 1          # sum without carry, carry moved up
        rounds += 1
    return a


def get_sum_trace(a: int, b: int, cap: int = 40, mask: bool = False) -> list[tuple[int, int]]:
    out = [(a, b)]
    if mask:
        a, b = a & M32, b & M32
        out = [(a, b)]
    while b and len(out) <= cap:
        a, b = a ^ b, (a & b) << 1
        if mask:
            a, b = a & M32, b & M32
        out.append((a, b))
    return out


def get_sum_masked(a: int, b: int) -> int:
    """32-bit wrap added - but the answer is still read as unsigned."""
    a, b = a & M32, b & M32
    while b:
        a, b = (a ^ b) & M32, ((a & b) << 1) & M32
    return a


class Solution371:
    def getSum(self, a: int, b: int) -> int:
        a, b = a & M32, b & M32
        while b:
            a, b = (a ^ b) & M32, ((a & b) << 1) & M32   # the carry falls off bit 31
        return a if a < 1 << 31 else a - (1 << 32)       # read bit 31 as the sign


# ------------------------------------------------------- 10. LeetCode 191 / 231
class Solution191:
    def hammingWeight(self, n: int) -> int:
        count = 0
        while n:
            n &= n - 1
            count += 1
        return count


class Solution231:
    def isPowerOfTwo(self, n: int) -> bool:
        return n > 0 and n & (n - 1) == 0


def is_power_of_two_buggy(n: int) -> bool:
    return n & (n - 1) == 0                     # 0 has no 1 to drop either


# ================================================================= the story
def header(title: str) -> None:
    print()
    print('=' * 72)
    print(title)
    print('=' * 72)


def main(argv=None) -> None:
    rng = random.Random(40)

    header('1. Python integers have no width')
    for x in (5, -5):
        print(f'{x:>3}: bin() -> {bin(x):<7}  low 8 bits {bits(x)}  low 32 bits {bits(x, 32)}')
    print(f' ~5 = {~5}   (flip every bit of ...0000101 -> ...1111010, which is -6)')
    print(f' -5 >> 1 = {-5 >> 1}   (arithmetic shift = floor(-5 / 2); the 1s keep coming in)')
    print(f' u32(-5) = {u32(-5):,}   s32(u32(-5)) = {s32(u32(-5))}')
    print(f' s32(0x80000000) = {s32(0x80000000):,}   (bit 31 is worth -2**31)')

    header('2. The toolkit on x = 180 = 0b10110100')
    x = 180
    print(f'{"expression":<16}{"meaning":<26}{"bits":>10}{"value":>7}')
    print(f'{"x":<16}{"":<26}{bits(x):>10}{x:>7}')
    for expr, meaning, fn in TOOLKIT:
        v = fn(x)
        print(f'{expr:<16}{meaning:<26}{bits(v):>10}{v:>7}')
    print(f'trailing zeros: {trailing_zeros(x)}   bit_length: {x.bit_length()}   '
          f'popcount: {x.bit_count()}')
    print('Gray codes 0..7:', [bits(gray(i), 3) for i in range(8)],
          '- neighbours differ in one bit')
    print('\nprecedence: arithmetic binds tighter than <<, and << tighter than & ^ |')
    for n in (4, 8):
        print(f'  n = {n}: 1 << n - 1 = {full_mask_buggy(n):>3} ({bits(full_mask_buggy(n))})'
              f'   (1 << n) - 1 = {full_mask(n):>3} ({bits(full_mask(n))})')

    header('3. popcount five ways (and a lookup table)')
    xs = [rng.getrandbits(32) for _ in range(200_000)]
    ref = [v.bit_count() for v in xs]
    total = sum(ref)
    print(f'200,000 random 32-bit ints, {total:,} ones in total')
    times = {}
    for name, fn in POPCOUNTS:
        assert [fn(v) for v in xs[:2000]] == ref[:2000], name
        times[name] = best_time(lambda: [fn(v) for v in xs], repeat=3)
    base = times['x.bit_count()']
    for name, _ in POPCOUNTS:
        print(f'  {name:<22}{times[name] * 1000:>8.1f} ms  {times[name] / base:>5.1f}x')
    shift_steps = sum(v.bit_length() for v in xs)
    print(f'loop rounds: shift loop {shift_steps:,} (one per bit up to the top 1), '
          f'Kernighan {total:,} (one per 1) = {shift_steps / total:.2f}x fewer')
    print('SWAR stages for 180:', [bits(s, 8) for s in swar_stages(180)[:4]],
          '->', swar_stages(180)[4])
    big = rng.getrandbits(1_000_000)
    tb = best_time(lambda: big.bit_count())
    ts = best_time(lambda: bin(big).count('1'))
    print(f'one 1,000,000-bit integer: bit_count {tb * 1e6:.0f} us, '
          f'bin().count {ts * 1e6:.0f} us ({ts / tb:.0f}x)')

    header('4. popcount of a negative number')
    x = -5
    print(f'x = {x}; C / Java (32-bit) answer: {popcount32(x)}')
    print(f'  bin(x).count("1") -> {popcount_bin(x)}   (bin(-5) is {bin(-5)!r}: it counts |x|)')
    print(f'  x.bit_count()     -> {popcount_builtin(x)}   (documented: popcount of abs(x))')
    tr = kernighan_trace(x, cap=40)
    print(f'  Kernighan         -> never returns: {tr[:6]} ... after 40 rounds x = {tr[-1]:,}')
    print(f'  (x & M32).bit_count() -> {popcount32(x)}   pin the width first')

    header('5. LeetCode 338 - Counting Bits')
    print('n = 5  ->', Solution338().countBits(5))
    print('n = 16 ->', Solution338().countBits(16))
    t_dp = best_time(lambda: Solution338().countBits(200_000), repeat=3)
    t_pc = best_time(lambda: [bin(i).count('1') for i in range(200_001)], repeat=3)
    print(f'n = 200,000: DP {t_dp * 1000:.1f} ms, bin().count per number {t_pc * 1000:.1f} ms')

    header('6. A Python int as a set of a million members')
    rng = random.Random(40)
    U = 1_000_000
    A = set(rng.sample(range(U), 100_000))
    B = set(rng.sample(range(U), 100_000))
    a, b = to_bitset(A, U), to_bitset(B, U)
    t_set = best_time(lambda: len(A & B), repeat=10)
    t_bit = best_time(lambda: (a & b).bit_count(), repeat=10)
    print(f'two 100,000-member sets from 0..999,999: |A & B| = {len(A & B):,} = '
          f'{(a & b).bit_count():,}')
    print(f'  set intersection {t_set * 1000:.2f} ms   AND + bit_count {t_bit * 1000:.3f} ms'
          f'   ({t_set / t_bit:.0f}x)')
    print(f'  memory: set {sys.getsizeof(A):,} B   int {sys.getsizeof(a):,} B'
          f'   ({sys.getsizeof(A) / sys.getsizeof(a):.0f}x)')
    print(f'  Jaccard {jaccard_bits(a, b):.4f} = {len(A & B) / len(A | B):.4f}')
    few = rng.sample(range(U), 20_000)
    t_slow = best_time(lambda: to_bitset_slow(few), repeat=1)
    t_fast = best_time(lambda: to_bitset(few, U), repeat=3)
    print(f'building from 20,000 members: x |= 1 << v {t_slow * 1000:.0f} ms, '
          f'bytearray {t_fast * 1000:.1f} ms ({t_slow / t_fast:.0f}x) - '
          f'every |= copies the whole integer')

    header("7. Gosper's hack - every k-subset in increasing order")
    print('n = 5, k = 3:', [bits(m, 5) for m in k_subsets(5, 3)])
    n, k = 20, 3
    print(f'n = {n}, k = {k}: {len(k_subsets(n, k)):,} masks visited directly vs '
          f'{1 << n:,} checked by filtering range(1 << n)')

    header('8. LeetCode 136 / 137 - what XOR can and cannot cancel')
    print('136 [4, 1, 2, 1, 2] ->', Solution136().singleNumber([4, 1, 2, 1, 2]),
          '   [-7, 3, 3] ->', Solution136().singleNumber([-7, 3, 3]))
    for nums in ([2, 2, 3, 2], [0, 1, 0, 1, 0, 1, 99], [-2, -2, -3, -2]):
        print(f'137 {str(nums):<24} state machine {Solution137().singleNumber(nums):>4}'
              f'   count+sign {single_number_ii_count(nums):>4}'
              f'   count, no sign {single_number_ii_count_nosign(nums):>12,}')
    tests = []
    for _ in range(1000):
        pool = rng.sample(range(-2**31, 2**31), 6)
        tests.append(pool[1:] * 3 + [pool[0]])
    wrong = sum(single_number_ii_count_nosign(t) != Solution137().singleNumber(t)
                for t in tests)
    assert wrong == sum(Solution137().singleNumber(t) < 0 for t in tests)
    print(f'1,000 random tests over the full int range: no-sign version wrong on {wrong}'
          f' - exactly the ones whose answer is negative')

    header('9. LeetCode 371 - Sum of Two Integers without + or -')
    for a_, b_ in ((1, 2), (2, 3), (-1, -1), (-1, 1)):
        naive = get_sum_naive(a_, b_, cap=100)
        print(f'  {a_:>2} + {b_:>2}: naive {"no answer after 100 rounds" if naive is None else naive!s:<26}'
              f' masked {get_sum_masked(a_, b_):>12,}   fixed {Solution371().getSum(a_, b_):>3}')
    tr = get_sum_trace(-1, 1, cap=40)
    print('naive (-1, 1):', ', '.join(f'({p}, {q})' for p, q in tr[:4]),
          f'... round 40: ({tr[40][0]:,}, {tr[40][1]:,})')
    tm = get_sum_trace(-1, 1, cap=40, mask=True)
    print(f'masked (-1, 1): carry reaches bit 31 and falls off after {len(tm) - 1} rounds -> {tm[-1]}')

    header('10. LeetCode 191 / 231')
    print('191:', [Solution191().hammingWeight(v) for v in (11, 128, 4294967293)])
    for v in (1, 16, 3, 0):
        print(f'231 n = {v:>2}: n > 0 and n & (n-1) == 0 -> {Solution231().isPowerOfTwo(v)!s:<5}'
              f'  without n > 0 -> {is_power_of_two_buggy(v)}')

    run_tests()
    print('\nall assertions passed')


def run_tests() -> None:
    assert bits(-5) == '11111011' and u32(-5) == 4294967291 and s32(u32(-5)) == -5
    assert ~5 == -6 and -5 >> 1 == -3 and s32(0x80000000) == -2**31
    x = 180
    assert [fn(x) for _, _, fn in TOOLKIT] == [1, 188, 164, 52, 4, 176, 181, 238]
    assert trailing_zeros(180) == 2 and full_mask_buggy(4) == 8 and full_mask(4) == 15
    assert all(bin(gray(i) ^ gray(i + 1)).count('1') == 1 for i in range(1000))
    rng = random.Random(1)
    for _ in range(3000):
        v = rng.getrandbits(32)
        want = bin(v).count('1')
        assert (popcount_builtin(v) == popcount_shift(v) == popcount_kernighan(v)
                == popcount_swar32(v) == popcount_table32(v) == want)
    assert swar_stages(180)[-1] == 4
    assert popcount_bin(-5) == popcount_builtin(-5) == 2 and popcount32(-5) == 31
    tr = kernighan_trace(-5, cap=40)
    assert len(tr) == 41 and tr[-1] != 0
    assert Solution338().countBits(5) == [0, 1, 1, 2, 1, 2]
    assert Solution338().countBits(5000) == count_bits_lowbit(5000) == [bin(i).count('1') for i in range(5001)]
    vals = rng.sample(range(10_000), 500)
    assert to_bitset(vals, 10_000) == to_bitset_slow(vals) and members(to_bitset(vals, 10_000)) == sorted(vals)
    for n in range(1, 13):
        for k in range(0, n + 1):
            ks = k_subsets(n, k)
            assert len(ks) == math.comb(n, k) and ks == sorted(ks)
            assert all(m.bit_count() == k and m < 1 << n for m in ks)
    assert Solution136().singleNumber([4, 1, 2, 1, 2]) == 4 and Solution136().singleNumber([-7, 3, 3]) == -7
    for nums, want in (([2, 2, 3, 2], 3), ([0, 1, 0, 1, 0, 1, 99], 99), ([-2, -2, -3, -2], -3)):
        assert Solution137().singleNumber(nums) == single_number_ii_count(nums) == want
    assert single_number_ii_count_nosign([2, 2, 3, 2]) == 3
    assert single_number_ii_count_nosign([0, 1, 0, 1, 0, 1, 99]) == 99
    assert single_number_ii_count_nosign([-2, -2, -3, -2]) == 4294967293
    for a, b in ((1, 2), (2, 3), (-1, -1), (-1, 1), (-1000, 999), (2**31 - 1, -2**31), (0, 0)):
        assert Solution371().getSum(a, b) == a + b
    assert get_sum_naive(1, 2) == 3 and get_sum_naive(2, 3) == 5 and get_sum_naive(-1, 1, cap=1000) is None
    assert get_sum_masked(-1, -1) == 4294967294 and get_sum_masked(-1, 1) == 0
    assert len(get_sum_trace(-1, 1, cap=100, mask=True)) - 1 == 32
    assert [Solution191().hammingWeight(v) for v in (11, 128, 4294967293)] == [3, 1, 31]
    assert [Solution231().isPowerOfTwo(v) for v in (1, 16, 3, 0, -16)] == [True, True, False, False, False]
    assert is_power_of_two_buggy(0) is True


if __name__ == '__main__':
    main()
