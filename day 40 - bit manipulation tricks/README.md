# Bit manipulation: masks, popcount, and integers that have no width

More details in:
https://medium.com/100-days-of-python

Interactive walkthrough: [demo.html](demo.html) - a single self-contained page
(no build step, works offline, 中文 / English toggle) that runs the lowest-bit tricks on
`x = 180` one bit row at a time, counts the ones in 180 with Kernighan's loop and with the
shift loop side by side, fills the LeetCode 338 table from `ans[i >> 1]`, adds `-1 + 1`
without `+` and watches the carry either fall off bit 31 or march left forever, and
counts LeetCode 137's columns mod 3 with and without the line that reads bit 31 as the
sign.

![img](https://github.com/KingRei/100DaysPython/blob/master/day%2040%20-%20bit%20manipulation%20tricks/imgs/day40_1.png?raw=true)

A Python integer used as a set of a million numbers answers "how many members do these two
sets share?" about sixty times faster than two `set` objects and takes a thirty-first of
the memory. Counting its ones with `int.bit_count()` beats every clever popcount formula
by four to thirty-five times. Both are bit manipulation, and both are safe.

What is not safe is carrying a C or Java bit trick into Python unchanged. A Python `int`
behaves like a two's-complement number with infinitely many bits: a negative number is a
1 in every position from some point upward, forever. The tricks that look at each column
on its own - `&`, `|`, `^`, `~`, `x & -x`, `x & (x - 1)` - give the same answer at every
width and carry over untouched. The tricks that ask "how many bits?" or rely on a carry
falling off the top need a width, and Python never stores one. They do not crash:
`bin(-5).count('1')` is 2 where C says 31, LeetCode 371's classic loop never returns on
`-1 + 1`, and LeetCode 137's column-count answer passes both examples and fails every
test whose answer is negative. Two tiny functions fix all of them:

```python
M32 = (1 << 32) - 1

def u32(x):           # keep the low 32 bits - a C uint32_t
    return x & M32

def s32(u):           # read 32 bits as signed: bit 31 is worth -2**31
    u &= M32
    return u - (1 << 32) if u >> 31 else u
```

Run `python bit_tricks.py` to print every number on this page.

## The toolkit on x = 180

![img](https://github.com/KingRei/100DaysPython/blob/master/day%2040%20-%20bit%20manipulation%20tricks/imgs/day40_2.png?raw=true)

| expression | meaning | 180 = `10110100` becomes |
|---|---|---|
| `x >> k & 1` | test bit k | bit 2 -> 1 |
| `x \| 1 << k` | set bit k | bit 3 -> 188 |
| `x & ~(1 << k)` | clear bit k | bit 4 -> 164 |
| `x ^ 1 << k` | toggle bit k | bit 7 -> 52 |
| `x & -x` | keep only the lowest 1 | 4 |
| `x & (x - 1)` | drop the lowest 1 | 176 |
| `x \| (x + 1)` | turn on the lowest 0 | 181 |
| `x ^ (x >> 1)` | Gray code | 238 |

The lowest-bit three work because subtracting 1 turns the lowest 1 into a 0 and every 0
below it into a 1, and `-x` is `~x + 1`. `x & -x` is the step a Fenwick tree takes (Day 29)
and how Day 39's bitmask N-Queens picked its next free column. `(x & -x).bit_length() - 1`
is the number of trailing zeros.

**Precedence trap.** Arithmetic binds tighter than `<<`, so `1 << n - 1` is a single bit
(`n = 4` gives 8), not `n` ones (15). Write `(1 << n) - 1`. Python's `x & 1 == 0` is
fine - Day 37 covered why C's version of that trap does not exist here.

## popcount five ways

![img](https://github.com/KingRei/100DaysPython/blob/master/day%2040%20-%20bit%20manipulation%20tricks/imgs/day40_3.png?raw=true)

```python
def popcount_shift(x):          # one round per bit up to the top 1
    count = 0
    while x:
        count += x & 1
        x >>= 1
    return count

def popcount_kernighan(x):      # one round per 1: x &= x - 1 removes the lowest 1
    count = 0
    while x:
        x &= x - 1
        count += 1
    return count

def popcount_swar32(x):         # add all 32 bits in parallel inside one word
    x = x - ((x >> 1) & 0x55555555)                   # 16 two-bit counts
    x = (x & 0x33333333) + ((x >> 2) & 0x33333333)    # 8 four-bit counts
    x = (x + (x >> 4)) & 0x0F0F0F0F                   # 4 byte counts
    return ((x * 0x01010101) & 0xFFFFFFFF) >> 24      # sum of the 4 bytes
```

On 200,000 random 32-bit integers, Kernighan's loop runs 3,199,046 rounds against the
shift loop's 6,200,129 - half, because a random 32-bit number has about 16 ones and about
31 bits up to its top 1.

![img](https://github.com/KingRei/100DaysPython/blob/master/day%2040%20-%20bit%20manipulation%20tricks/imgs/day40_4.png?raw=true)

In C, SWAR is the fast one. In Python it loses to the builtin by five times, because every
`&`, `>>` and `+` is a separate trip through the interpreter; the method with the fewest
Python-level operations wins, and `int.bit_count()` (Python 3.10+) runs its loop in C.
On one 1,000,000-bit integer, `bit_count()` takes about 70 microseconds and
`bin(x).count('1')` about 6 milliseconds, because `bin()` first builds a million-character
string.

## Silent failure: popcount of a negative number

| method | -5 gives |
|---|---|
| C `__builtin_popcount`, Java `Integer.bitCount` | 31 |
| `bin(-5).count('1')` (`bin(-5)` is `'-0b101'`) | 2 |
| `(-5).bit_count()` (documented as popcount of `abs(x)`) | 2 |
| Kernighan or the shift loop | never returns |
| `(-5 & M32).bit_count()` | 31 |

`x & (x - 1)` of a negative number is still negative - the 1s above never run out - so
Kernighan's loop goes `-5, -6, -8, -16, -32, ...` forever. Pin the width first.

## LeetCode 338 - Counting Bits

**The task.** For every whole number from 0 up to `n`, report how many 1s its binary form
contains.

**Input / output.** One integer `n`; return a list of `n + 1` integers where entry `i` is
the number of 1 bits in `i`.

**Example.** `n = 5` gives `[0, 1, 1, 2, 1, 2]`: 0 is `0`, 1 is `1`, 2 is `10`, 3 is `11`,
4 is `100`, 5 is `101`.

**Constraints.** `0 <= n <= 100,000`. Calling popcount on each number is already fast
enough; the follow-up asks for one linear pass with no builtin popcount, which is what
rules out the obvious answer and makes this a DP problem.

[leetcode.com/problems/counting-bits](https://leetcode.com/problems/counting-bits/)

```python
class Solution:
    def countBits(self, n: int) -> list[int]:
        ans = [0] * (n + 1)
        for i in range(1, n + 1):
            ans[i] = ans[i >> 1] + (i & 1)      # i without its last bit, plus that bit
        return ans
```

`i >> 1` is smaller than `i`, so its entry is always filled already. Dropping the lowest 1
instead gives the same table: `ans[i] = ans[i & (i - 1)] + 1`.

## A Python int as a set of a million members

Bit `v` set means "v is in the set". Intersection is `&`, union is `|`, size is
`bit_count()`, and each is one C loop over machine words. Two random 100,000-member sets
drawn from 0..999,999 share 9,955 members; `len(A & B)` on two `set` objects takes about
5 ms, `(a & b).bit_count()` about 0.08 ms. The `set` takes 4,194,520 bytes, the integer
133,360. Day 34's subset-sum `bits |= bits << x`, Day 32's Bloom filter and Day 27's Jaccard
similarity all run on this.

**Silent slowdown: building it one member at a time.** `x |= 1 << v` looks constant-time
but creates a brand-new integer every time, as long as the whole set. Adding 20,000
members to a million-bit set takes about 220 ms that way, and about 3 ms by filling a
`bytearray` and converting once:

```python
def to_bitset(values, universe):
    buf = bytearray((universe + 7) // 8)
    for v in values:
        buf[v >> 3] |= 1 << (v & 7)
    return int.from_bytes(buf, 'little')
```

## Gosper's hack: every k-subset in increasing order

```python
def next_same_popcount(x):
    c = x & -x                          # lowest 1
    r = x + c                           # carry it into the block of 1s above
    return (((r ^ x) >> 2) // c) | r    # put the leftover 1s back at the bottom
```

Starting from `00111` it visits `01011 01101 01110 10011 10101 10110 11001 11010 11100`.
Choosing 3 of 20 visits 1,140 masks instead of filtering all 1,048,576.

## LeetCode 136 / 137 - what XOR can and cannot cancel

### LeetCode 136 - Single Number

**The task.** In a list where every value shows up exactly twice except one, find the one
that shows up once.

**Input / output.** A non-empty list of integers `nums`; return the lone value.

**Example.** `[4, 1, 2, 1, 2]` gives 4: the 1s pair up, the 2s pair up, 4 is left over.

**Constraints.** Up to 30,000 values between -30,000 and 30,000, and the problem demands
linear time *and* constant extra space. The space rule is what forbids a `Counter` or a
`set` and forces a bit trick.

[leetcode.com/problems/single-number](https://leetcode.com/problems/single-number/)

### LeetCode 137 - Single Number II

**The task.** Same idea, but every other value shows up exactly three times.

**Input / output.** A non-empty list of integers `nums`; return the value that appears
once.

**Example.** `[0, 1, 0, 1, 0, 1, 99]` gives 99: 0 and 1 each appear three times.

**Constraints.** Up to 30,000 values anywhere in the signed 32-bit range, linear time and
constant extra space again. The full signed range is the trap: the answer can be negative,
and in Python that decides whether the bit-counting solution is right.

[leetcode.com/problems/single-number-ii](https://leetcode.com/problems/single-number-ii/)

```python
class Solution136:
    def singleNumber(self, nums):
        return reduce(xor, nums, 0)             # pairs cancel, column by column

class Solution137:
    def singleNumber(self, nums):
        ones = twos = 0                         # per column: seen once / seen twice (mod 3)
        for x in nums:
            ones = (ones ^ x) & ~twos
            twos = (twos ^ x) & ~ones
        return ones
```

XOR cancels pairs, not triples. Both of these work column by column, so they need no width
and return negatives correctly. The other standard answer to 137 counts each of the 32
columns mod 3 - and without `s32` at the end bit 31 comes back worth +2**31:

![img](https://github.com/KingRei/100DaysPython/blob/master/day%2040%20-%20bit%20manipulation%20tricks/imgs/day40_6.png?raw=true)

| nums | ones/twos | count + `s32` | count, no `s32` |
|---|---|---|---|
| `[2, 2, 3, 2]` | 3 | 3 | 3 |
| `[0, 1, 0, 1, 0, 1, 99]` | 99 | 99 | 99 |
| `[-2, -2, -3, -2]` | -3 | -3 | 4,294,967,293 |

On 1,000 random tests over the full range the no-`s32` version is wrong on 502 - exactly
the ones whose answer is negative, and none of the examples.

## LeetCode 371 - Sum of Two Integers

**The task.** Add two integers without using the `+` or `-` operators.

**Input / output.** Two integers `a` and `b`; return their sum.

**Example.** `a = 2, b = 3` gives 5: `2 ^ 3 = 1` is the sum ignoring carries, `(2 & 3) << 1
= 4` is the carry, and `1 ^ 4 = 5` with no carry left.

**Constraints.** `-1000 <= a, b <= 1000`. Small, but negative - which is the whole problem
in Python, because the carry loop only terminates when the carry falls off a fixed-width
word.

[leetcode.com/problems/sum-of-two-integers](https://leetcode.com/problems/sum-of-two-integers/)

![img](https://github.com/KingRei/100DaysPython/blob/master/day%2040%20-%20bit%20manipulation%20tricks/imgs/day40_5.png?raw=true)

```python
class Solution:
    def getSum(self, a: int, b: int) -> int:
        a, b = a & M32, b & M32
        while b:
            a, b = (a ^ b) & M32, ((a & b) << 1) & M32   # the carry falls off bit 31
        return a if a < 1 << 31 else a - (1 << 32)       # read bit 31 as the sign
```

| a + b | loop with no mask | mask, no sign fix | the version above |
|---|---|---|---|
| 1 + 2 | 3 | 3 | 3 |
| 2 + 3 | 5 | 5 | 5 |
| -1 + -1 | -2 | 4,294,967,294 | -2 |
| -1 + 1 | never returns | 0 | 0 |

Without the mask, `(-1, 1)` becomes `(-2, 2)`, `(-4, 4)`, `(-8, 8)`, ... - after 40 rounds
the carry is 1,099,511,627,776 and still growing. With the mask the carry reaches bit 31
and falls off after 32 rounds. Both LeetCode examples pass with no mask at all.

## LeetCode 191 / 231

### LeetCode 191 - Number of 1 Bits

**The task.** Count the 1s in the binary form of a number (its Hamming weight).

**Input / output.** A positive integer `n`; return the count.

**Example.** `n = 11` is `1011`, so the answer is 3. `128` is `10000000`, answer 1.

**Constraints.** `1 <= n <= 2**31 - 1` in the current version; the older version handed
over an unsigned 32-bit value such as 4,294,967,293 (31 ones), which is exactly the case
where a signed language needs a mask. The follow-up - "called many times, how would you
speed it up?" - is asking for Kernighan's loop or a byte lookup table.

[leetcode.com/problems/number-of-1-bits](https://leetcode.com/problems/number-of-1-bits/)

### LeetCode 231 - Power of Two

**The task.** Decide whether an integer is a power of two.

**Input / output.** An integer `n`; return `True` if `n` equals 2 to some whole-number
power.

**Example.** 1 (two to the zero) and 16 are powers of two, 3 is not.

**Constraints.** `n` ranges over the whole signed 32-bit range, zero and negatives
included. The follow-up asks for no loops or recursion, which is what `n & (n - 1)` is for
- and the zero and negatives are what make the `n > 0` guard mandatory.

[leetcode.com/problems/power-of-two](https://leetcode.com/problems/power-of-two/)

```python
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
```

A power of two has exactly one 1, so dropping it leaves 0. But 0 has no 1 to drop either:
without `n > 0`, `0` returns `True`, and all three examples still pass.

## Complexity

| Operation | Time | Space |
|---|---|---|
| test / set / clear / toggle / lowbit, fixed width | O(1) | O(1) |
| popcount, shift loop | O(bit length) | O(1) |
| popcount, Kernighan | O(number of 1s) | O(1) |
| popcount, SWAR (32-bit) | O(1), 13 operations | O(1) |
| `int.bit_count()`, n-bit int | O(n / 64) machine words | O(1) |
| Counting Bits (LeetCode 338) | O(n) | O(n) output |
| bitset `&`, `\|`, `bit_count` over a universe of U | O(U / 64) | O(U / 8) bytes |
| bitset built by `x \|= 1 << v`, m members | O(m · U / 64) | O(U / 8) |
| bitset built by `bytearray` + `from_bytes` | O(m + U / 8) | O(U / 8) |
| Gosper's hack, all k-subsets of n | O(C(n, k)) | O(1) per step |
| Single Number I / II (LeetCode 136 / 137) | O(n) | O(1) |
| Sum of Two Integers (LeetCode 371), 32-bit | O(32) rounds | O(1) |

Every "O(1)" in the first rows assumes a fixed width. Python's integers are arbitrary
precision, so on a million-bit integer each of those operations costs time proportional
to its length - which is why building a bitset one bit at a time is quadratic.

## References

- [Bitwise operation - Wikipedia](https://en.wikipedia.org/wiki/Bitwise_operation)
- [Hamming weight - Wikipedia](https://en.wikipedia.org/wiki/Hamming_weight) - SWAR popcount and the lookup-table method
- [Two's complement - Wikipedia](https://en.wikipedia.org/wiki/Two%27s_complement)
- [Python docs: `int.bit_count()`](https://docs.python.org/3/library/stdtypes.html#int.bit_count) and [bitwise operations on integer types](https://docs.python.org/3/library/stdtypes.html#bitwise-operations-on-integer-types)
- Henry S. Warren Jr., *Hacker's Delight*, 2nd ed., 2012 - chapters 2 and 5
- Sean Eron Anderson, [Bit Twiddling Hacks](https://graphics.stanford.edu/~seander/bithacks.html)
- Donald Knuth, *The Art of Computer Programming*, Vol. 4A, section 7.1.3 - bitwise tricks, including Gosper's hack
