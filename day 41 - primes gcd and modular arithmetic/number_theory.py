# -*- coding: utf-8 -*-
"""Day 41 - Primes, fast power, GCD and modular arithmetic.

Run it:  python number_theory.py

Modular arithmetic is the deal that keeps huge numbers small: take % m after
every + - and *, and the remainder at the end is exactly what you would have
got from the full-size number.  Python's three-argument pow(3, e, m) is that
deal plus fast power - 82 multiplications for an exponent of a billion
billion - and it answers a million-digit question in microseconds.

The deal has one hole: division.  Nothing about it crashes.

* nCk mod p computed as  fact[n] // (fact[k] * fact[n-k])  on factorials that
  were already reduced mod 1e9+7 is right for every n up to 12 and wrong from
  13 on - because 13! is the first factorial bigger than the modulus.
* the "modular inverse" pow(a, m - 2, m) is Fermat's little theorem and only
  works when m is prime.  For m = 10 it says the inverse of 3 is 1.
* LeetCode 204 asks for primes *below* n; counting primes up to n passes all
  three examples and is wrong every time n itself is prime.

Sections
  1. What survives % (and what does not)
  2. Fast power: square-and-multiply, LeetCode 50, Fibonacci by matrix power
  3. GCD: Euclid, the worst case, extended Euclid, LeetCode 365
  4. Modular inverse - and Fermat's shortcut on a composite modulus
  5. nCk mod p - the // that works until n = 13
  6. Sieve of Eratosthenes, LeetCode 204
  7. LeetCode 1922 and LeetCode 372
"""
from __future__ import annotations

import math
import random
import sys
import time
from collections import deque
from math import comb, isqrt

MOD = 10**9 + 7                 # prime - the modulus almost every judge uses


def header(title: str) -> None:
    print('\n' + '=' * 72 + f'\n{title}\n' + '=' * 72)


def best_time(fn, repeat: int = 3) -> float:
    best = float('inf')
    for _ in range(repeat):
        t = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t)
    return best


# --------------------------------------------------------------- 2. fast power
def power_mod(a: int, e: int, m: int) -> int:
    """Square-and-multiply: walk the bits of e from the lowest one upward."""
    result = 1 % m
    a %= m
    while e:
        if e & 1:                       # this bit is on: multiply its power in
            result = result * a % m
        a = a * a % m                   # a, a^2, a^4, a^8 ... one squaring per bit
        e >>= 1
    return result


def power_trace(a: int, e: int, m: int) -> list[dict]:
    """One row per bit of e: the bit, the current square, the running result."""
    rows, result, sq, k = [], 1 % m, a % m, 0
    while e:
        bit = e & 1
        if bit:
            result = result * sq % m
        rows.append({'k': k, 'bit': bit, 'square': sq, 'result': result})
        sq = sq * sq % m
        e >>= 1
        k += 1
    return rows


def multiplications(e: int) -> tuple[int, int]:
    """(naive loop, square-and-multiply) multiplication counts for a^e."""
    return e - 1 if e else 0, max(e.bit_length() - 1, 0) + max(e.bit_count() - 1, 0)


class Solution50:
    """LeetCode 50 - Pow(x, n)."""

    def myPow(self, x: float, n: int) -> float:
        if n < 0:
            x, n = 1 / x, -n            # in Python -n never overflows
        result = 1.0
        while n:
            if n & 1:
                result *= x
            x *= x
            n >>= 1
        return result


def mat_mul(A, B, m):
    return [[(A[0][0] * B[0][0] + A[0][1] * B[1][0]) % m,
             (A[0][0] * B[0][1] + A[0][1] * B[1][1]) % m],
            [(A[1][0] * B[0][0] + A[1][1] * B[1][0]) % m,
             (A[1][0] * B[0][1] + A[1][1] * B[1][1]) % m]]


def fib_mod(n: int, m: int = MOD) -> tuple[int, int]:
    """F(n) mod m from [[1,1],[1,0]]^n - same square-and-multiply, on matrices.
    Returns (F(n) mod m, number of 2x2 matrix multiplications)."""
    R, A, mults = [[1, 0], [0, 1]], [[1, 1], [1, 0]], 0
    while n:
        if n & 1:
            R = mat_mul(R, A, m)
            mults += 1
        n >>= 1
        if n:
            A = mat_mul(A, A, m)
            mults += 1
    return R[0][1], mults


def fib_loop(n: int, m: int = MOD) -> int:
    a, b = 0, 1
    for _ in range(n):
        a, b = b, (a + b) % m
    return a


# --------------------------------------------------------------------- 3. GCD
def gcd(a: int, b: int) -> int:
    """Euclid: gcd(a, b) = gcd(b, a % b), until the remainder is 0."""
    while b:
        a, b = b, a % b
    return abs(a)


def gcd_trace(a: int, b: int) -> list[tuple[int, int, int, int]]:
    """Rows (a, b, q, r) with a = q*b + r."""
    rows = []
    while b:
        q, r = divmod(a, b)
        rows.append((a, b, q, r))
        a, b = b, r
    return rows


def gcd_subtract_steps(a: int, b: int) -> int:
    """How many subtractions the 'subtract the smaller from the larger' gcd needs.
    Each Euclid step a % b replaces a // b subtractions, so the total is the sum
    of the quotients - computed here without actually doing them all."""
    steps = 0
    while b:
        steps += a // b
        a, b = b, a % b
    return steps - 1                     # the last quotient's final subtraction hits 0


def ext_gcd(a: int, b: int) -> tuple[int, int, int]:
    """(g, x, y) with a*x + b*y == g == gcd(a, b)."""
    x0, y0, x1, y1 = 1, 0, 0, 1
    while b:
        q, a, b = a // b, b, a % b
        x0, x1 = x1, x0 - q * x1
        y0, y1 = y1, y0 - q * y1
    return a, x0, y0


def ext_gcd_trace(a: int, b: int) -> list[tuple[int, int, int, int, int]]:
    """Rows (a, b, q, x, y): after each step a == a0*x + b0*y still holds."""
    a0, b0 = a, b
    x0, y0, x1, y1 = 1, 0, 0, 1
    rows = [(a, b, None, x0, y0)]
    while b:
        q, a, b = a // b, b, a % b
        x0, x1 = x1, x0 - q * x1
        y0, y1 = y1, y0 - q * y1
        rows.append((a, b, q, x0, y0))
        assert a == a0 * x0 + b0 * y0
    return rows


class Solution365:
    """LeetCode 365 - Water and Jug Problem."""

    def canMeasureWater(self, x: int, y: int, target: int) -> bool:
        return target <= x + y and target % math.gcd(x, y) == 0


def can_measure_bfs(x: int, y: int, target: int) -> bool:
    """Brute force over (jug1, jug2) states - the check for Solution365."""
    seen, q = {(0, 0)}, deque([(0, 0)])
    while q:
        a, b = q.popleft()
        if a + b == target:
            return True
        pour_ab, pour_ba = min(a, y - b), min(b, x - a)
        for s in ((x, b), (a, y), (0, b), (a, 0),
                  (a - pour_ab, b + pour_ab), (a + pour_ba, b - pour_ba)):
            if s not in seen:
                seen.add(s)
                q.append(s)
    return False


# ---------------------------------------------------------- 4. modular inverse
def inverse(a: int, m: int) -> int:
    """x with a*x % m == 1, via extended Euclid; exists only if gcd(a, m) == 1."""
    g, x, _ = ext_gcd(a % m, m)
    if g != 1:
        raise ValueError(f'{a} has no inverse mod {m}')
    return x % m


def inverse_fermat(a: int, m: int) -> int:
    """a^(m-2) mod m.  Correct only when m is prime - and nothing checks that."""
    return pow(a, m - 2, m)


def fermat_report(m: int) -> tuple[int, int]:
    """(invertible a in 1..m-1, how many of them Fermat gets wrong)."""
    inv = [a for a in range(1, m) if math.gcd(a, m) == 1]
    return len(inv), sum(a * inverse_fermat(a, m) % m != 1 for a in inv)


# ------------------------------------------------------------- 5. nCk mod p
def factorials(n: int, p: int = MOD) -> list[int]:
    f = [1] * (n + 1)
    for i in range(1, n + 1):
        f[i] = f[i - 1] * i % p
    return f


def ncr_floor_div(n: int, k: int, f: list[int], p: int = MOD) -> int:
    """The bug: integer division on numbers that have already been reduced mod p."""
    return f[n] // (f[k] * f[n - k] % p) % p


class Binomial:
    """nCk mod p for many queries: O(n) table once, O(1) per query."""

    def __init__(self, n: int, p: int = MOD):
        self.p = p
        self.f = factorials(n, p)
        self.inv_f = [1] * (n + 1)
        self.inv_f[n] = pow(self.f[n], p - 2, p)       # p is prime: Fermat is fine here
        for i in range(n, 0, -1):                      # 1/(i-1)! = i * 1/i!
            self.inv_f[i - 1] = self.inv_f[i] * i % p

    def ncr(self, n: int, k: int) -> int:
        if k < 0 or k > n:
            return 0
        return self.f[n] * self.inv_f[k] % self.p * self.inv_f[n - k] % self.p


# ------------------------------------------------------------------ 6. sieve
def is_prime_trial(n: int) -> bool:
    if n < 2:
        return False
    for d in range(2, isqrt(n) + 1):
        if n % d == 0:
            return False
    return True


def sieve(n: int) -> bytearray:
    """is_prime[i] for 0 <= i < n.  Cross out multiples of each prime p,
    starting at p*p (smaller multiples were crossed by smaller primes),
    and only for p*p < n (a composite below n has a factor <= sqrt(n))."""
    is_prime = bytearray([1]) * max(n, 2)
    is_prime[0] = is_prime[1] = 0
    for p in range(2, isqrt(n - 1) + 1 if n > 1 else 2):
        if is_prime[p]:
            is_prime[p * p::p] = bytes(len(range(p * p, n, p)))
    return is_prime[:n]


def sieve_loop(n: int, start_at_square: bool = True, stop_at_root: bool = True):
    """The same sieve as plain loops, counting every crossing-out it does."""
    is_prime = [True] * max(n, 2)
    is_prime[0] = is_prime[1] = False
    crossings = 0
    top = isqrt(n - 1) if stop_at_root else n - 1
    for p in range(2, top + 1):
        if is_prime[p]:
            for q in range(p * p if start_at_square else 2 * p, n, p):
                is_prime[q] = False
                crossings += 1
    return is_prime[:n], crossings


def sieve_trace(n: int) -> list[tuple[int, list[int]]]:
    """(p, the numbers p crosses out for the first time) for each sieving prime."""
    is_prime = [True] * n
    is_prime[0] = is_prime[1] = False
    out = []
    for p in range(2, isqrt(n - 1) + 1):
        if is_prime[p]:
            new = [q for q in range(p * p, n, p) if is_prime[q]]
            for q in range(p * p, n, p):
                is_prime[q] = False
            out.append((p, new))
    return out


class Solution204:
    """LeetCode 204 - Count Primes (strictly less than n)."""

    def countPrimes(self, n: int) -> int:
        return sum(sieve(n)) if n > 2 else 0


def count_primes_upto(n: int) -> int:
    """The bug: primes <= n instead of < n.  Passes n = 10, 0, 1."""
    return sum(sieve(n + 1)) if n >= 2 else 0


# --------------------------------------------------------------- 7. LC 1922 / 372
class Solution1922:
    """LeetCode 1922 - Count Good Numbers."""

    def countGoodNumbers(self, n: int) -> int:
        even_slots, odd_slots = (n + 1) // 2, n // 2   # indices 0, 2, 4 ... / 1, 3, 5 ...
        return pow(5, even_slots, MOD) * pow(4, odd_slots, MOD) % MOD


class Solution372:
    """LeetCode 372 - Super Pow: a^b mod 1337, b given as a list of digits."""

    def superPow(self, a: int, b: list[int]) -> int:
        result = 1
        for d in b:                      # a^(10*e + d) = (a^e)^10 * a^d
            result = pow(result, 10, 1337) * pow(a, d, 1337) % 1337
        return result


def super_pow_fermat(a: int, b: list[int]) -> int:
    """The bug: shrink the exponent mod 1336 as if 1337 were prime (1337 = 7 * 191)."""
    e = int(''.join(map(str, b)))
    return pow(a, e % 1336, 1337)


# ===================================================================== story
def main() -> None:
    header('1. What survives % (and what does not)')
    a, b, m = 123_456_789, 987_654_321, 1_000
    print(f'a = {a:,}  b = {b:,}  m = {m}')
    print(f'  (a + b) % m = {(a + b) % m:>3}   (a % m + b % m) % m = {(a % m + b % m) % m:>3}')
    print(f'  (a - b) % m = {(a - b) % m:>3}   (a % m - b % m) % m = {(a % m - b % m) % m:>3}')
    print(f'  (a * b) % m = {(a * b) % m:>3}   (a % m) * (b % m) % m = {(a % m) * (b % m) % m:>3}')
    x, y = 12, 4
    print(f'  division: ({x} // {y}) % 5 = {(x // y) % 5}   but ({x} % 5) // ({y} % 5) = {x % 5} // {y % 5} = {(x % 5) // (y % 5)}')
    print(f'  sign: Python -7 % 3 = {-7 % 3}   C / Java (truncate toward zero) gives {int(math.fmod(-7, 3))}')
    print('  in Python the remainder takes the sign of m, so (a - b) % m never goes negative')

    header('2. Fast power: square-and-multiply')
    a, e, m = 3, 13, 100
    print(f'{a}^{e} mod {m}, e = {e} = 0b{e:b}: one squaring per bit, one multiply per 1 bit')
    print('  bit k  bit  a^(2^k) mod m  result')
    for r in power_trace(a, e, m):
        print(f'  {r["k"]:>5}  {r["bit"]:>3}  {r["square"]:>13}  {r["result"]:>6}'
              + ('   <- multiply in' if r['bit'] else ''))
    print(f'  check: 3**13 = {3**13:,}, mod 100 = {3**13 % 100}')
    for e in (13, 10**9, 10**18):
        naive, fast = multiplications(e)
        print(f'  e = {e:>25,}: naive {naive:>25,} multiplications, square-and-multiply {fast:>2}')
    for e in (10**5, 10**6):
        t_big = best_time(lambda: 3**e % MOD, repeat=3)
        t_pow = best_time(lambda: pow(3, e, MOD), repeat=3)
        print(f'  3**{e:,} % MOD: {t_big * 1000:7.2f} ms (builds a {(3**e).bit_length():,}-bit number first)'
              f'   pow(3, {e:,}, MOD): {t_pow * 1e6:5.1f} us  ({t_big / t_pow:,.0f}x)')
    s = Solution50()
    print(f'LC 50: myPow(2.0, 10) = {s.myPow(2.0, 10)}   myPow(2.1, 3) = {s.myPow(2.1, 3):.5f}'
          f'   myPow(2.0, -2) = {s.myPow(2.0, -2)}')
    print(f'       n = -2**31: naive loop {2**31:,} multiplications, square-and-multiply {multiplications(2**31)[1]}')
    n = 10**18
    f, mults = fib_mod(n)
    print(f'Fibonacci: F(10**18) mod 1e9+7 = {f:,} in {mults} matrix multiplications'
          f'  (the loop would need a billion billion additions)')

    header('3. GCD: Euclid, the worst case, extended Euclid')
    print('gcd(252, 105):')
    for a, b, q, r in gcd_trace(252, 105):
        print(f'  {a:>4} = {q} x {b:>3} + {r:>3}')
    print(f'  -> {gcd(252, 105)}: anything dividing a and b divides a - q*b = r, and the other way round')
    print(f'subtraction version: gcd(252, 105) {gcd_subtract_steps(252, 105)} subtractions,'
          f' gcd(1,000,000, 1) {gcd_subtract_steps(10**6, 1):,} subtractions vs {len(gcd_trace(10**6, 1))} Euclid step')
    worst = max(((len(gcd_trace(a, b)), a, b) for a in range(1, 1001) for b in range(1, a + 1)))
    print(f'most Euclid steps for 1 <= b <= a <= 1000: {worst[0]} steps at gcd({worst[1]}, {worst[2]})'
          f' - consecutive Fibonacci numbers (every quotient is 1)')
    print(f'gcd(832,040, 514,229) = F(30), F(29): {len(gcd_trace(832040, 514229))} steps;'
          f' a 1000-digit pair of random numbers: {len(gcd_trace(random.Random(3).getrandbits(3322), random.Random(4).getrandbits(3322)))} steps')
    print('extended Euclid on (240, 46), keeping a == 240*x + 46*y on every row:')
    print('     a     b   q     x     y')
    for a, b, q, x, y in ext_gcd_trace(240, 46):
        print(f'  {a:>4}  {b:>4}  {"" if q is None else q:>2}  {x:>4}  {y:>4}')
    g, x, y = ext_gcd(240, 46)
    print(f'  -> gcd 2 = 240*({x}) + 46*({y}) = {240 * x + 46 * y}')
    s = Solution365()
    for args in ((3, 5, 4), (2, 6, 5), (1, 2, 3), (3, 5, 9)):
        print(f'LC 365 {args}: {s.canMeasureWater(*args)}   BFS over states agrees: {can_measure_bfs(*args)}')
    print('  without target <= x + y, (3, 5, 9) says True: gcd 1 divides 9, but the jugs hold 8')

    header('4. Modular inverse - and Fermat on a composite modulus')
    print(f'inverse(3, 10) = {inverse(3, 10)}  (3 x 7 = 21 = 2*10 + 1)   pow(3, -1, 10) = {pow(3, -1, 10)}')
    print(f'inverse(3, 11) = {inverse(3, 11)}  Fermat pow(3, 9, 11) = {inverse_fermat(3, 11)}  - 11 is prime, they agree')
    print(f'Fermat on m = 10: pow(3, 8, 10) = {inverse_fermat(3, 10)}  -> 3 x {inverse_fermat(3, 10)} % 10 = {3 * inverse_fermat(3, 10) % 10}, not 1')
    for m in (10, 12, 1337, 1_000_000, MOD):
        if m == MOD:
            rng = random.Random(41)
            sample = [rng.randrange(1, m) for _ in range(1000)]
            wrong = sum(a * inverse_fermat(a, m) % m != 1 for a in sample)
            print(f'  m = {m:>13,} (prime):     1,000 random a, Fermat wrong on {wrong}')
        else:
            n_inv, wrong = fermat_report(m)
            print(f'  m = {m:>13,} (composite): {n_inv:>7,} invertible a, Fermat wrong on {wrong:,}')
    try:
        pow(4, -1, 10)
    except ValueError as err:
        print(f'pow(4, -1, 10) raises: {err}   - the built-in refuses; pow(4, 8, 10) = {pow(4, 8, 10)} just answers')

    header('5. nCk mod p - the // that works until n = 13')
    f = factorials(200)
    first = next((n, k) for n in range(200) for k in range(n + 1) if ncr_floor_div(n, k, f) != comb(n, k) % MOD)
    print(f'fact[n] // (fact[k] * fact[n-k]) on factorials reduced mod 1e9+7:')
    print(f'  every n <= 12 correct; first wrong: C{first} -> {ncr_floor_div(*first, f)} instead of {comb(*first)}')
    print(f'  12! = {math.factorial(12):,} < 1e9+7 < 13! = {math.factorial(13):,}: from 13 on fact[n] is a remainder, not n!')
    bad = sum(ncr_floor_div(n, k, f) != comb(n, k) % MOD for n in range(31) for k in range(n + 1))
    print(f'  all n <= 30: {bad} of {sum(n + 1 for n in range(31))} answers wrong;'
          f' C(100, 50) -> {ncr_floor_div(100, 50, f)} instead of {comb(100, 50) % MOD:,}')
    B = Binomial(10**6)
    print(f'fix: multiply by inverse factorials - C(100, 50) = {B.ncr(100, 50):,}, C(10**6, 500,000) = {B.ncr(10**6, 500_000):,}')
    rng = random.Random(5)
    qs = [(n, rng.randrange(n + 1)) for n in (rng.randrange(1, 10_001) for _ in range(1000))]
    t_tab = best_time(lambda: [B.ncr(n, k) for n, k in qs], repeat=3)
    t_cmb = best_time(lambda: [comb(n, k) % MOD for n, k in qs], repeat=1)
    print(f'1,000 queries with n <= 10,000: table {t_tab * 1000:.1f} ms, math.comb(n, k) % p {t_cmb * 1000:.0f} ms'
          f' ({t_cmb / t_tab:,.0f}x) - comb builds the full-size number every time')

    header('6. Sieve of Eratosthenes')
    print('n = 50, each prime crosses out the numbers nobody crossed before it:')
    for p, new in sieve_trace(50):
        print(f'  p = {p}: starts at {p * p:>2}, new: {new}')
    print(f'  left standing: {[i for i, v in enumerate(sieve(50)) if v]}')
    n = 10**6
    _, c_best = sieve_loop(n)
    _, c_2p = sieve_loop(n, start_at_square=False)
    _, c_all = sieve_loop(n, start_at_square=False, stop_at_root=False)
    print(f'crossings for n = 1,000,000: start at p*p {c_best:,}; start at 2p {c_2p:,}; '
          f'and sieve with every prime {c_all:,}')
    t_trial = best_time(lambda: sum(is_prime_trial(i) for i in range(n)), repeat=1)
    t_loop = best_time(lambda: sieve_loop(n), repeat=1)
    t_slice = best_time(lambda: sum(sieve(n)), repeat=3)
    print(f'primes below 1,000,000 = {sum(sieve(n)):,}:  trial division {t_trial:.2f} s,'
          f'  sieve with loops {t_loop * 1000:.0f} ms,  sieve with bytearray slices {t_slice * 1000:.1f} ms')
    print(f'  ({t_trial / t_slice:,.0f}x and {t_loop / t_slice:,.0f}x slower than the slices)')
    t7 = best_time(lambda: sum(sieve(10**7)), repeat=1)
    print(f'primes below 10,000,000 = {sum(sieve(10**7)):,} in {t7 * 1000:.0f} ms')
    s = Solution204()
    print('LC 204 (primes strictly below n) vs the same code counting primes up to n:')
    for n in (10, 0, 1, 2, 3, 5, 12, 13):
        print(f'  n = {n:>2}: below {s.countPrimes(n)}   up to {count_primes_upto(n)}'
              + ('   <- n is prime' if s.countPrimes(n) != count_primes_upto(n) else ''))
    wrong = sum(s.countPrimes(n) != count_primes_upto(n) for n in range(5001))
    print(f'  n = 0..5000: wrong on {wrong} values = exactly the {sum(sieve(5001)):,} primes; the three examples are 10, 0, 1')

    header('7. LeetCode 1922 and LeetCode 372')
    s = Solution1922()
    print(f'1922: n = 1 -> {s.countGoodNumbers(1)}, n = 4 -> {s.countGoodNumbers(4)}, n = 50 -> {s.countGoodNumbers(50):,},'
          f' n = 10**15 -> {s.countGoodNumbers(10**15):,}')
    s = Solution372()
    for a, b in ((2, [3]), (2, [1, 0]), (1, [4, 3, 3, 8, 5, 2]), (2147483647, [2, 0, 0])):
        print(f'372: a = {a:<10} b = {str(b):<20} -> {s.superPow(a, b):>4}   exponent mod 1336 -> {super_pow_fermat(a, b):>4}')
    rng = random.Random(41)
    tests = []
    for _ in range(2000):
        b = [rng.randrange(1, 10)] + [rng.randrange(10) for _ in range(rng.randrange(0, 6))]
        tests.append((rng.randrange(1, 2**31), b))
    wrong = sum(super_pow_fermat(a, b) != s.superPow(a, b) for a, b in tests)
    print(f'  2,000 random tests: "exponent mod 1336" wrong on {wrong:,} - 1337 = 7 x 191 is not prime')

    run_tests()
    print('\nall assertions passed')


def run_tests() -> None:
    assert (-7) % 3 == 2 and 12 // 4 % 5 == 3 and (12 % 5) // (4 % 5) == 0
    rng = random.Random(1)
    for _ in range(500):
        a, e, m = rng.randrange(10**9), rng.randrange(10**6), rng.randrange(1, 10**9)
        assert power_mod(a, e, m) == pow(a, e, m)
    assert power_mod(5, 0, 1) == 0 and power_trace(3, 13, 100)[-1]['result'] == 23
    assert multiplications(13) == (12, 5)
    s = Solution50()
    assert s.myPow(2.0, 10) == 1024.0 and abs(s.myPow(2.1, 3) - 9.261) < 1e-9 and s.myPow(2.0, -2) == 0.25
    assert all(fib_mod(n)[0] == fib_loop(n) for n in range(300))
    for _ in range(500):
        a, b = rng.randrange(10**12), rng.randrange(10**12)
        g, x, y = ext_gcd(a, b)
        assert g == math.gcd(a, b) == gcd(a, b) and a * x + b * y == g
    assert gcd(252, 105) == 21 and len(gcd_trace(832040, 514229)) == 28
    for a in range(1, 60):
        for b in range(1, 60):
            x, y, steps = a, b, 0
            while x != y:
                x, y = (x - y, y) if x > y else (x, y - x)
                steps += 1
            assert gcd_subtract_steps(max(a, b), min(a, b)) == steps
    s = Solution365()
    for x in range(1, 9):
        for y in range(1, 9):
            for t in range(0, 20):
                assert s.canMeasureWater(x, y, t) == can_measure_bfs(x, y, t)
    assert inverse(3, 10) == 7 and inverse_fermat(3, 10) == 1
    for m in (7, 11, 97, 101):
        assert all(inverse_fermat(a, m) == inverse(a, m) == pow(a, -1, m) for a in range(1, m))
    assert fermat_report(10) == (4, 3)
    f = factorials(60)
    assert all(ncr_floor_div(n, k, f) == comb(n, k) for n in range(13) for k in range(n + 1))
    assert ncr_floor_div(13, 1, f) != 13
    B = Binomial(500)
    assert all(B.ncr(n, k) == comb(n, k) % MOD for n in range(200) for k in range(n + 1))
    assert sum(sieve(10**6)) == 78498 and sum(sieve(100)) == 25
    assert all(bool(v) == is_prime_trial(i) for i, v in enumerate(sieve(5000)))
    assert sieve_loop(1000)[0] == [bool(v) for v in sieve(1000)]
    s = Solution204()
    assert [s.countPrimes(n) for n in (10, 0, 1)] == [4, 0, 0]
    assert [count_primes_upto(n) for n in (10, 0, 1)] == [4, 0, 0] and count_primes_upto(3) == 2
    s = Solution1922()
    assert [s.countGoodNumbers(n) for n in (1, 4, 50)] == [5, 400, 564908303]
    s = Solution372()
    assert s.superPow(2, [3]) == 8 and s.superPow(2, [1, 0]) == 1024
    assert s.superPow(1, [4, 3, 3, 8, 5, 2]) == 1 and s.superPow(2147483647, [2, 0, 0]) == 1198
    for _ in range(300):
        a = rng.randrange(1, 2**31)
        b = [rng.randrange(1, 10)] + [rng.randrange(10) for _ in range(rng.randrange(0, 8))]
        assert s.superPow(a, b) == pow(a, int(''.join(map(str, b))), 1337)


if __name__ == '__main__':
    if hasattr(sys, 'set_int_max_str_digits'):
        sys.set_int_max_str_digits(0)
    main()
