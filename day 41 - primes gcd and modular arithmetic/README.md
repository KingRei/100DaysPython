# Modular arithmetic, fast power, GCD and the prime sieve

More details in:
https://medium.com/100-days-of-python

Interactive walkthrough: [demo.html](demo.html) - a single self-contained page
(no build step, works offline, 中文 / English toggle) that sieves the numbers below 50
prime by prime (and shows where "up to n" breaks LeetCode 204), runs square-and-multiply
on 3^13 mod 100 one bit at a time, walks Euclid and extended Euclid on 240 and 46 into
LeetCode 365, checks Fermat's inverse against every number mod 11 and mod 10, and watches
the `//` version of nCk mod p hit the wall at n = 13.

![img](https://github.com/KingRei/100DaysPython/blob/master/day%2041%20-%20primes%20gcd%20and%20modular%20arithmetic/imgs/day41_1.png?raw=true)

`pow(3, 10**18, 10**9 + 7)` raises 3 to the power of a billion billion and hands back the remainder in about two microseconds. It needs 82 multiplications to do it. Asking instead for `3**1_000_000 % (10**9 + 7)` - an exponent a million million times *smaller* - first builds a number with 1.58 million bits and takes close to a tenth of a second, tens of thousands of times slower. In the same toolbox, the sieve of Eratosthenes finds all 78,498 primes below a million in about 13 milliseconds, and Euclid's algorithm finds the greatest common divisor of two thousand-digit numbers in under two thousand steps.

All of it rests on one deal: take `% m` after every `+`, `-` and `*`, and the remainder you end with is exactly the one the full-size number would have had. The deal has one hole, which is division, and none of the ways through that hole raise an error:

- `fact[n] // (fact[k] * fact[n-k])` on factorials already reduced mod 1,000,000,007 is right for every `n` up to 12 and wrong from 13 on.
- The popular "modular inverse" `pow(a, m - 2, m)` only works when `m` is prime. For `m = 10` it says the inverse of 3 is 1.
- LeetCode 204 asks for primes *below* `n`. Counting primes *up to* `n` passes all three examples and is wrong every time `n` is itself prime.

Yesterday was about the bits of an integer (Day 40). Today the bits of the *exponent* turn out to be the whole trick behind fast power.

Run `python number_theory.py` to print every number on this page.

## What survives %

Take `a = 123,456,789`, `b = 987,654,321`, `m = 1000`. Reduce first or reduce at the end - the answers agree:

```text
(a + b) % m = 110     (a % m + b % m) % m = 110
(a - b) % m = 468     (a % m - b % m) % m = 468
(a * b) % m = 269     (a % m) * (b % m) % m = 269
```

The reason is that `a` and `a % m` differ by a multiple of `m`, and adding, subtracting or multiplying by a multiple of `m` only ever adds more multiples of `m`. Division does not have that property: `(12 // 4) % 5` is 3, but `(12 % 5) // (4 % 5)` is `2 // 4 = 0`. Remember that `0`; it comes back as a wrong binomial coefficient.

One Python-specific detail: `%` takes the sign of `m`, so `-7 % 3` is 2 where C and Java give -1. `(a - b) % m` therefore never goes negative in Python, and the `+ m` that C code adds before reducing is unnecessary here.

## Fast power: square-and-multiply

13 in binary is `1101`, which is 8 + 4 + 1. So 3¹³ = 3⁸ · 3⁴ · 3¹, and 3¹, 3², 3⁴, 3⁸ are each the square of the one before. Walk the bits of the exponent from the lowest: square once per bit, and multiply into the result only on the bits that are 1.

```python
def power_mod(a, e, m):
    result = 1 % m
    a %= m
    while e:
        if e & 1:                       # this bit is on: multiply its power in
            result = result * a % m
        a = a * a % m                   # a, a^2, a^4, a^8 ... one squaring per bit
        e >>= 1
    return result
```

On `3^13 mod 100` the loop runs four times:

```text
bit k  bit  3^(2^k) mod 100  result
    0    1                3       3   <- multiply in
    1    0                9       3
    2    1               81      43   <- multiply in
    3    1               61      23   <- multiply in
```

And 3¹³ is 1,594,323, which does end in 23. The count of multiplications is one squaring per bit plus one multiply per 1 bit, minus the free ones at the start:

| exponent | naive loop | square-and-multiply |
|---|---|---|
| 13 | 12 | 5 |
| 1,000,000,000 | 999,999,999 | 41 |
| 1,000,000,000,000,000,000 | 999,999,999,999,999,999 | 82 |

Python's built-in `pow(a, e, m)` is this loop written in C. Use it; the function above is here so you can see what it does. What you should *not* do is `a ** e % m`: Python will happily compute the full-size power first.

### LeetCode 50 - Pow(x, n)

**The task.** Raise a floating-point number to a whole-number power, without calling the library's power function.

**Input / output.** A float `x` and an integer `n`; return `x` to the power `n` as a float.

**Example.** `x = 2.0, n = 10` gives `1024.0`. `x = 2.0, n = -2` gives `0.25`: a negative power is one over the positive power, `1 / 2² = 1/4`.

**Constraints.** `-100 < x < 100`, and `n` can be anything in the signed 32-bit range, from -2,147,483,648 to 2,147,483,647. Two billion multiplications is far too many, so the problem is really asking for square-and-multiply. The bottom of the range is the trap in C and Java: `-(-2^31)` does not fit in 32 bits. The result is guaranteed to stay between -10,000 and 10,000.

[leetcode.com/problems/powx-n](https://leetcode.com/problems/powx-n/)

```python
class Solution:
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
```

For `n = -2**31` this does 31 squarings instead of 2,147,483,648 multiplications. In Python the negation is safe; in C++ you would widen `n` to a 64-bit `long long` first.

The same loop works on anything you can multiply, including matrices. Day 33's climbing-stairs answer turned out to be the Fibonacci sequence, filled in one addition at a time. `[[1, 1], [1, 0]]` raised to the power `n` holds F(n) in its corner, so square-and-multiply on 2x2 matrices gives F(10¹⁸) mod 1,000,000,007 = 209,783,453 in 83 matrix multiplications. The loop would need a billion billion additions.

## Euclid's algorithm for the greatest common divisor

![img](https://github.com/KingRei/100DaysPython/blob/master/day%2041%20-%20primes%20gcd%20and%20modular%20arithmetic/imgs/day41_2.png?raw=true)

Any number that divides both `a` and `b` also divides `a - q·b` for every `q`, and in particular divides the remainder `a % b`. The reverse is true as well. So `gcd(a, b) = gcd(b, a % b)`, and the numbers shrink until the remainder is 0:

```text
252 = 2 x 105 + 42
105 = 2 x 42  + 21
 42 = 2 x 21  + 0      -> gcd = 21
```

The figure shows it as tiling: cut the largest squares you can from a 252 by 105 rectangle, then from what is left over, and the last square that tiles the rest exactly has side 21.

```python
def gcd(a, b):
    while b:
        a, b = b, a % b
    return abs(a)
```

The version that subtracts the smaller number from the larger one is the same idea done slowly: `gcd(1,000,000, 1)` takes 999,999 subtractions, while `%` does it in one step. The slowest input for Euclid is a pair of consecutive Fibonacci numbers, because every quotient is 1: among all pairs up to 1000, the worst is `gcd(987, 610)` at 14 steps, and `gcd(832,040, 514,229)` takes 28. The step count grows with the number of *digits*, not the size: two random thousand-digit numbers took 1,875 steps.

**Extended Euclid** carries two extra numbers along so that every row stays a combination of the two inputs. On `(240, 46)` it keeps `a == 240·x + 46·y` true on every row:

```text
   a     b   q     x     y
 240    46         1     0
  46    10   5     0     1
  10     6   4     1    -5
   6     4   1    -4    21
   4     2   1     5   -26
   2     0   2    -9    47
```

So `2 = 240·(-9) + 46·47`. This is Bezout's identity: the gcd can always be written as a whole-number combination of the inputs, and the amounts you can build from `a` and `b` are exactly the multiples of their gcd.

```python
def ext_gcd(a, b):
    x0, y0, x1, y1 = 1, 0, 0, 1
    while b:
        q, a, b = a // b, b, a % b
        x0, x1 = x1, x0 - q * x1
        y0, y1 = y1, y0 - q * y1
    return a, x0, y0                    # a*x0 + b*y0 == gcd
```

### LeetCode 365 - Water and Jug Problem

**The task.** You have two jugs holding `x` and `y` litres and an unlimited tap. Each move fills one jug to the top, empties one jug, or pours one jug into the other until the first is empty or the second is full. Decide whether you can end up with exactly `target` litres in the two jugs together.

**Input / output.** Three integers `x`, `y`, `target`; return `True` or `False`.

**Example.** `x = 3, y = 5, target = 4` is `True`: fill the 5, pour it into the 3 (2 left), empty the 3, pour the 2 across, fill the 5 again, top up the 3 with one litre, and empty the 3 - exactly 4 litres are left, all in the big jug. `x = 2, y = 6, target = 5` is `False`: every amount you can make is even.

**Constraints.** `1 <= x, y, target <= 1000`. A breadth-first search over the (jug 1, jug 2) states visits at most about a million states and is fine at this size; Bezout answers it with one gcd.

[leetcode.com/problems/water-and-jug-problem](https://leetcode.com/problems/water-and-jug-problem/)

```python
class Solution:
    def canMeasureWater(self, x: int, y: int, target: int) -> bool:
        return target <= x + y and target % math.gcd(x, y) == 0
```

The first condition is the one people drop. Without it, `(3, 5, 9)` says `True` - gcd 1 divides 9 - but two jugs of 3 and 5 hold 8 litres at most. The BFS in the repo agrees with this one-liner on all four examples.

## The modular inverse, and Fermat's shortcut

![img](https://github.com/KingRei/100DaysPython/blob/master/day%2041%20-%20primes%20gcd%20and%20modular%20arithmetic/imgs/day41_4.png?raw=true)

You cannot divide under `% m`, but you can multiply by an *inverse*: a number `x` with `a·x % m == 1`. The inverse of 3 mod 10 is 7, because 3 · 7 = 21. Extended Euclid finds it - run it on `(a, m)`, and if the gcd is 1 then `a·x + m·y = 1`, so `a·x` leaves remainder 1. If the gcd is not 1, no inverse exists; 4 has none mod 10.

```python
def inverse(a, m):
    g, x, _ = ext_gcd(a % m, m)
    if g != 1:
        raise ValueError(f'{a} has no inverse mod {m}')
    return x % m
```

Since Python 3.8 the built-in does this too: `pow(3, -1, 10)` is 7, and `pow(4, -1, 10)` raises `ValueError: base is not invertible for the given modulus`.

The shortcut you see in most competitive code is `pow(a, m - 2, m)`. It comes from Fermat's little theorem: when `m` is a prime, `a^(m-1) % m == 1`, so `a^(m-2)` is the inverse. When `m` is not a prime, nothing checks, and the answer is simply wrong. For `m = 10` it gives `pow(3, 8, 10) = 1`, and 3 · 1 is not 1 mod 10. Counting every number that does have an inverse:

| m | numbers with an inverse | Fermat wrong on |
|---|---|---|
| 10 | 4 | 3 |
| 12 | 4 | 3 |
| 1,337 | 1,140 | 1,136 |
| 1,000,000 | 400,000 | 399,999 |
| 1,000,000,007 (prime) | 1,000 random a | 0 |

For the prime it is right every time; for the others it is right almost never. `pow(4, 8, 10)` returns 6 without complaint, where `pow(4, -1, 10)` refuses. When the modulus is 1,000,000,007 or 998,244,353 the shortcut is fine. For any other modulus, use `pow(a, -1, m)`.

## nCk mod p: the // that works until 13

![img](https://github.com/KingRei/100DaysPython/blob/master/day%2041%20-%20primes%20gcd%20and%20modular%20arithmetic/imgs/day41_5.png?raw=true)

Problems that ask for "the number of ways, modulo 10⁹ + 7" usually need binomial coefficients `C(n, k) = n! / (k! (n-k)!)`. The natural first attempt builds a table of factorials mod p and divides:

```python
f[n] // (f[k] * f[n - k] % p) % p      # the bug
```

It gives the right answer for every `n` up to 12. The first wrong one is `C(13, 1)`, which comes back as 0 instead of 13. The reason is one line of arithmetic: 12! = 479,001,600 is smaller than 1,000,000,007 and 13! = 6,227,020,800 is larger. Up to 12 the "factorials mod p" are the real factorials, so `//` is real division. From 13 on, `f[n]` is a remainder, and dividing remainders is the `2 // 4 = 0` from the start of this article. For `n` up to 30, 369 of the 496 answers are wrong, and `C(100, 50)` comes back as 0 instead of 538,992,043. Small hand-made tests all pass.

The fix is to multiply by inverse factorials. Because 1/(i-1)! = i · 1/i!, one `pow` call for the largest factorial gives all the others:

```python
class Binomial:
    def __init__(self, n, p=MOD):
        self.p = p
        self.f = factorials(n, p)
        self.inv_f = [1] * (n + 1)
        self.inv_f[n] = pow(self.f[n], p - 2, p)       # p is prime: Fermat is fine here
        for i in range(n, 0, -1):                      # 1/(i-1)! = i * 1/i!
            self.inv_f[i - 1] = self.inv_f[i] * i % p

    def ncr(self, n, k):
        if k < 0 or k > n:
            return 0
        return self.f[n] * self.inv_f[k] % self.p * self.inv_f[n - k] % self.p
```

`C(1,000,000, 500,000) mod p` is 996,692,777. A thousand random queries with `n` up to 10,000 take under a millisecond with the table and around one and a half seconds with `math.comb(n, k) % p`, which builds the full-size number each time.

## The sieve of Eratosthenes

![img](https://github.com/KingRei/100DaysPython/blob/master/day%2041%20-%20primes%20gcd%20and%20modular%20arithmetic/imgs/day41_3.png?raw=true)

Instead of testing each number for divisors, let each prime cross out its own multiples. On the numbers below 50:

```text
p = 2 crosses out 4, 6, 8, ... 48     (23 numbers)
p = 3 crosses out 9, 15, 21, 27, 33, 39, 45
p = 5 crosses out 25, 35
p = 7 crosses out 49
```

Two details make it fast. Each prime starts at `p·p`, because the smaller multiples (`3·5 = 15` for p = 5) have already been crossed out by a smaller prime. And the outer loop stops once `p·p` passes `n`: 8 · 8 = 64 is past 49, so nothing after 7 has anything left to do. The 15 numbers still standing are the primes. For n = 1,000,000, starting at `p·p` does 2,122,046 crossings, starting at `2p` does 2,197,837, and letting every prime up to `n` sieve does 2,775,208.

The bigger speed-up in Python is not doing the inner loop in Python at all. Assigning zeros to a slice of a `bytearray` crosses out every multiple in one C-level operation:

```python
def sieve(n):
    is_prime = bytearray([1]) * max(n, 2)
    is_prime[0] = is_prime[1] = 0
    for p in range(2, isqrt(n - 1) + 1 if n > 1 else 2):
        if is_prime[p]:
            is_prime[p * p::p] = bytes(len(range(p * p, n, p)))
    return is_prime[:n]
```

Finding the 78,498 primes below a million took about 4.6 seconds by trial division, a quarter of a second with the sieve written as loops, and about 13 milliseconds with slices. Below ten million there are 664,579 primes, found in about a third of a second.

### LeetCode 204 - Count Primes

**The task.** Count how many prime numbers are strictly smaller than `n`.

**Input / output.** One integer `n`; return the count.

**Example.** `n = 10` gives 4: the primes below 10 are 2, 3, 5 and 7. `n = 0` and `n = 1` both give 0.

**Constraints.** `0 <= n <= 5,000,000`. Testing each number by trial division is too slow at five million in Python, which is why this is a sieve problem. The word "strictly" is the part that matters.

[leetcode.com/problems/count-primes](https://leetcode.com/problems/count-primes/)

```python
class Solution:
    def countPrimes(self, n: int) -> int:
        return sum(sieve(n)) if n > 2 else 0
```

Write `sieve(n + 1)` - "primes up to n" - and all three examples still pass, because 10, 0 and 1 are not prime. It is wrong exactly when `n` itself is prime: for every `n` from 0 to 5000 it is wrong 669 times, and there are exactly 669 primes up to 5000.

## LeetCode 1922 and LeetCode 372: exponents too large to loop over

### LeetCode 1922 - Count Good Numbers

**The task.** A string of digits is "good" when every digit at an even position (counting from 0) is even and every digit at an odd position is a prime digit - 2, 3, 5 or 7. Count the good strings of length `n`, modulo 1,000,000,007. Leading zeros are allowed.

**Input / output.** One integer `n`; return the count mod 1,000,000,007.

**Example.** `n = 1` gives 5: the strings "0", "2", "4", "6", "8". `n = 4` gives 400: positions 0 and 2 have 5 choices each, positions 1 and 3 have 4 each, and 5 · 4 · 5 · 4 = 400.

**Constraints.** `1 <= n <= 10^15`. A thousand million million positions: no loop over them can finish, so the answer has to be two calls to fast power.

[leetcode.com/problems/count-good-numbers](https://leetcode.com/problems/count-good-numbers/)

```python
class Solution:
    def countGoodNumbers(self, n: int) -> int:
        even_slots, odd_slots = (n + 1) // 2, n // 2   # indices 0, 2, 4 ... / 1, 3, 5 ...
        return pow(5, even_slots, MOD) * pow(4, odd_slots, MOD) % MOD
```

`n = 50` gives 564,908,303 and `n = 10^15` gives 711,414,395, each in microseconds.

### LeetCode 372 - Super Pow

**The task.** Compute `a` to the power `b`, modulo 1337, where `b` is so large that it is given as a list of its decimal digits.

**Input / output.** An integer `a` and a list of digits `b`, most significant first; return `a^b mod 1337`.

**Example.** `a = 2, b = [1, 0]` means 2¹⁰ and gives 1024. `a = 2147483647, b = [2, 0, 0]` means that number to the 200th power and gives 1198.

**Constraints.** `1 <= a <= 2^31 - 1`, `b` has up to 2000 digits and no leading zeros. In Python you can join the digits into one integer and call `pow` - it is a 2000-digit exponent, which `pow` handles easily. The digit-by-digit version below is the one that works in any language.

[leetcode.com/problems/super-pow](https://leetcode.com/problems/super-pow/)

```python
class Solution:
    def superPow(self, a: int, b: list[int]) -> int:
        result = 1
        for d in b:                      # a^(10*e + d) = (a^e)^10 * a^d
            result = pow(result, 10, 1337) * pow(a, d, 1337) % 1337
        return result
```

The trap: some solutions shrink the exponent with `% 1336` first, borrowing Fermat's "the exponent repeats every m - 1 steps". That needs 1337 to be prime, and 1337 = 7 × 191. The shortcut passes all four official examples, and on 2,000 random tests it was wrong 961 times.

## Complexity

| Operation | Time | Space |
|---|---|---|
| `pow(a, e, m)` | O(log e) multiplications | O(1) |
| Fibonacci by 2x2 matrix power | O(log n) matrix products | O(1) |
| gcd / extended gcd | O(log min(a, b)) steps | O(1) |
| modular inverse (extended gcd or `pow(a, -1, m)`) | O(log m) | O(1) |
| Fermat inverse `pow(a, m - 2, m)`, m prime only | O(log m) | O(1) |
| Binomial table up to n, then each query | O(n) once, O(1) per query | O(n) |
| Sieve of Eratosthenes below n | O(n log log n) | O(n) bytes |
| Count Primes (LeetCode 204) | O(n log log n) | O(n) |
| Count Good Numbers (LeetCode 1922) | O(log n) | O(1) |
| Super Pow (LeetCode 372), d digits | O(d) | O(1) |

These counts assume numbers that fit in a machine word, which is what `% m` guarantees:
every intermediate value stays below `m²`. Without the `%`, Python's integers grow and each
multiplication costs time proportional to their length - which is why `3**1_000_000 % m`
is tens of thousands of times slower than `pow`.

## References

- [Modular arithmetic - Wikipedia](https://en.wikipedia.org/wiki/Modular_arithmetic)
- [Exponentiation by squaring - Wikipedia](https://en.wikipedia.org/wiki/Exponentiation_by_squaring)
- [Euclidean algorithm - Wikipedia](https://en.wikipedia.org/wiki/Euclidean_algorithm) and [Extended Euclidean algorithm](https://en.wikipedia.org/wiki/Extended_Euclidean_algorithm)
- [Bézout's identity - Wikipedia](https://en.wikipedia.org/wiki/B%C3%A9zout%27s_identity)
- [Modular multiplicative inverse - Wikipedia](https://en.wikipedia.org/wiki/Modular_multiplicative_inverse) and [Fermat's little theorem](https://en.wikipedia.org/wiki/Fermat%27s_little_theorem)
- [Sieve of Eratosthenes - Wikipedia](https://en.wikipedia.org/wiki/Sieve_of_Eratosthenes)
- [Python docs: `pow()`](https://docs.python.org/3/library/functions.html#pow) - three-argument form and negative exponents (3.8+)
- Donald Knuth, *The Art of Computer Programming*, Vol. 2, sections 4.5.2 (the greatest common divisor) and 4.6.3 (evaluation of powers)
