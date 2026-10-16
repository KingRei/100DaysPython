# Randomized algorithms and Monte Carlo: estimating with random numbers, and testing code that uses them

More details in:
https://medium.com/100-days-of-python

Interactive walkthrough: [demo.html](demo.html) - a single self-contained page
(no build step, works offline, 中文 / English toggle) that throws darts at a quarter circle
and watches the estimate of pi settle, replays every possible run of three shuffles (the
naive one, the `randrange(i)` slip and Fisher-Yates) and counts where each lands, walks
reservoir sampling down a stream with and without the off-by-one, fills the 7 x 7 grid
behind LeetCode 470, and slices `0..total` into weighted ranges for LeetCode 528.

![img](https://github.com/KingRei/100DaysPython/blob/master/day%2042%20-%20randomized%20algorithms%20and%20monte%20carlo/imgs/day42_1.png?raw=true)

Throw a million random points at a square and count how many land inside the quarter circle drawn from one corner. Four times that fraction came out as 3.140828 - pi to three decimals, off by 0.000765. Getting six correct decimals the same way would take about ten trillion points, because the error of a Monte Carlo estimate shrinks with the square root of the work: a hundred times more points buys only ten times less error.

That is one half of today. The other half is that randomized code has an unpleasant property: when it is wrong, it does not crash. A broken shuffle still returns a shuffled-looking list; a broken sampler still returns a plausible item. It is just drawn from the wrong distribution, and every example in the problem statement still passes:

- Swapping each slot with a random slot *anywhere* in the list makes some orders of three items come up 5 times in 27 and others 4 times in 27.
- Writing `randrange(i)` instead of `randrange(i + 1)` in Fisher-Yates reaches only 2 of the 6 orders of three items - it never leaves any item where it was.
- The same off-by-one in reservoir sampling (LeetCode 382) never returns the first node.
- `rand7`-based `rand10` without rejection (LeetCode 470) gives the digit 1 a chance of 4/49 and the other nine 5/49.
- `randint(0, total)` instead of `randrange(total)` in LeetCode 528 doubles the chance of a weight-1 item next to a weight-99 one.

The only test for randomness is to count. So this day builds a tool that runs an algorithm once for *every* possible sequence of random choices and adds up the exact probabilities as fractions - no sampling noise, so a bias of 1/27 has nowhere to hide.

Yesterday's `pow(a, e, m)` and Fermat's little theorem (Day 41) come back here as a primality test that is allowed to be wrong, and how often it is wrong is something we can measure.

Run `python randomized.py` to print every number on this page.

## Monte Carlo: pi from random points

A point dropped uniformly in the unit square lands inside the quarter circle `x² + y² < 1` with probability equal to the circle's area, pi/4. Drop `n` points, count the hits, and `4 * hits / n` estimates pi.

```python
def estimate_pi(n, rng):
    inside = 0
    for _ in range(n):
        x, y = rng.random(), rng.random()
        if x * x + y * y < 1.0:
            inside += 1
    return 4 * inside / n
```

With `random.Random(42)`:

```text
      100 points: pi ~ 3.120000   error 0.021593   one standard error = 0.164218
   10,000 points: pi ~ 3.125200   error 0.016393   one standard error = 0.016422
1,000,000 points: pi ~ 3.140828   error 0.000765   one standard error = 0.001642
```

One point's contribution `4 * [inside]` is either 0 or 4, and its spread (standard deviation) is `4 * sqrt(p(1 - p))` with `p = pi/4`, which is 1.642. Averaging `n` independent points divides the spread by `sqrt(n)`, so the typical error is `1.642 / sqrt(n)`. Repeating each experiment many times and measuring the root-mean-square error confirms it:

| points N | runs | measured error | 1.642 / sqrt(N) |
|---|---|---|---|
| 100 | 400 | 0.16296 | 0.16422 |
| 1,000 | 200 | 0.05454 | 0.05193 |
| 10,000 | 100 | 0.01690 | 0.01642 |
| 100,000 | 20 | 0.00389 | 0.00519 |

![img](https://github.com/KingRei/100DaysPython/blob/master/day%2042%20-%20randomized%20algorithms%20and%20monte%20carlo/imgs/day42_2.png?raw=true)

The last row is noisier because it only has 20 runs. The law is the point: one more correct decimal costs a hundred times more points, and an error of 0.0000005 (six decimals) needs 1.08 × 10¹³ points - about ten trillion. Monte Carlo is the wrong way to compute pi. It is the right way when the quantity has no formula at all: the value of a board position, the average return of a policy, an integral in a hundred dimensions. Those cost the same `1 / sqrt(N)`, no matter how many dimensions there are.

One more practical detail: seed your generator. `random.Random(7)` twice gives the same points and the same estimate (3.152 both times), which is what makes a randomized bug reproducible.

## Las Vegas vs Monte Carlo

Randomized algorithms come in two kinds, and it matters which one you are holding.

**Las Vegas: always right, random running time.** Quicksort with the first element as pivot does `n(n-1)/2` comparisons on an already sorted list - 1,999,000 for 2,000 items. Pick the pivot at random and no input is bad any more; only unlucky dice are, and they are rare:

```text
random pivot, 200 runs: mean 24,795, min 22,449, max 31,811   exact mean 24,730
```

The exact mean is `2(n+1)H(n) - 4n` with `H(n) = 1 + 1/2 + ... + 1/n`; for large `n` it grows like `1.39 n log₂ n`. The output is sorted every single time. This is the quicksort of Day 20 and the quickselect of Day 21.

**Monte Carlo: fixed running time, small chance of a wrong answer.** Day 41's Fermat's little theorem says that for a prime `n`, `a^(n-1) mod n` is 1. So if it is *not* 1, `n` is certainly composite. The converse fails, and badly for the Carmichael numbers. 561 = 3 × 11 × 17, and among the bases 2 to 559:

```text
Fermat says "prime" for 318 of 558 bases (57%)
Miller-Rabin says "prime" for 8 of 558 (1.4%); the bound is 1/4
```

Miller-Rabin writes `n - 1 = d · 2^s` with `d` odd, computes `a^d mod n` and squares it `s` times. If `n` is prime, the only numbers whose square is 1 mod `n` are 1 and `n - 1`, so the chain must hit `n - 1` right before it hits 1 (or start at 1). A different square root of 1 proves `n` composite.

```python
def miller_rabin_passes(n, a):
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
```

Rabin proved that at most a quarter of the bases lie for any odd composite; the worst below 2000 is 1891, with 23.7%. So `k` random bases are fooled with probability at most `(1/4)^k`. Below 100,000 there are 78 odd composites that pass Fermat with base 2 (341, 561, 645, 1105, 1387, ...); Miller-Rabin with one random base made 38 mistakes on the same range, and with 5 bases it made none. With the twelve primes 2 to 37 as fixed bases the test has been checked to be correct for every number below about 3.18 × 10²³, which covers every 64-bit integer: `is_prime(2**61 - 1)` says `True` for that 19-digit number without trying a single divisor beyond 37.

## Testing randomness exactly

Here is the tool the rest of the day relies on. Write the algorithm so it gets its random numbers from a function `randrange(k)` passed in. The tester supplies a fake `randrange` that answers 0 the first time, and then walks every combination of answers like a car odometer, rolling over each digit when it reaches `k`. Every run is weighted by `1/k₁ · 1/k₂ · ...` and the weights are added up as exact `Fraction`s.

```python
def exact_distribution(run):
    dist = defaultdict(Fraction)
    prefix = []
    while True:
        radices = []

        def randrange(k):
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
```

Two coin flips, counting heads: `exact_distribution(lambda rr: rr(2) + rr(2))` returns `{0: 1/4, 1: 1/2, 2: 1/4}`. Because the runs can have different lengths (a rejection loop that draws again, a stream that is cut short) the odometer only spins the digits a run actually used. It works for anything with a small number of choices, which is exactly where off-by-one bugs live.

## Shuffling: LeetCode 384

### LeetCode 384 - Shuffle an Array

**The task.** Build a class around an integer array that can return a random shuffle of it, where every ordering is equally likely, and can also reset the array to its original order.

**Input / output.** The constructor gets `nums`. `shuffle()` returns a random permutation; `reset()` returns the original array.

**Example.** For `nums = [1, 2, 3]`, `shuffle()` might return `[3, 1, 2]`; any of the 6 orders is fine, but each must come up with chance 1/6. `reset()` then returns `[1, 2, 3]`, and the next `shuffle()` is again any of the 6.

**Constraints.** Up to 50 elements, all distinct; up to 10,000 calls in total. The size is tiny - the real constraint is "equally likely", which the judge can only check statistically, so a slightly biased shuffle can pass.

[leetcode.com/problems/shuffle-an-array](https://leetcode.com/problems/shuffle-an-array/)

Three ways to write it:

```python
def shuffle_naive(a, randrange):       # swap with ANY slot
    n = len(a)
    for i in range(n):
        j = randrange(n)
        a[i], a[j] = a[j], a[i]

def shuffle_off_by_one(a, randrange):  # Fisher-Yates, one character off
    for i in range(len(a) - 1, 0, -1):
        j = randrange(i)
        a[i], a[j] = a[j], a[i]

def fisher_yates(a, randrange):
    for i in range(len(a) - 1, 0, -1):
        j = randrange(i + 1)           # 0..i inclusive - i itself is allowed
        a[i], a[j] = a[j], a[i]
```

Every possible run counted exactly, on three items `a b c`:

| version | runs | abc | acb | bac | bca | cab | cba |
|---|---|---|---|---|---|---|---|
| swap with any slot | 27 | 4 | 5 | 5 | 5 | 4 | 4 |
| `randrange(i)` slip | 2 | 0 | 0 | 0 | 1 | 1 | 0 |
| Fisher-Yates | 6 | 1 | 1 | 1 | 1 | 1 | 1 |

![img](https://github.com/KingRei/100DaysPython/blob/master/day%2042%20-%20randomized%20algorithms%20and%20monte%20carlo/imgs/day42_3.png?raw=true)

The naive shuffle makes `n^n` equally likely runs - 27 for three items - and they have to land on `n!` = 6 orders. 27 does not divide by 6, so it cannot be fair, and no amount of extra looping fixes that. On four items it is 256 runs over 24 orders, each order coming up between 8 and 15 times: the most common is 1.88 times as likely as the rarest. Sampled for real, 60,000 naive shuffles of three items gave `abc 8927, acb 11040, bac 11235, bca 10915, cab 8929, cba 8954` - visible if you look, invisible if you only check that the output is a permutation.

The `randrange(i)` slip is subtler. Since `j < i`, the item at position `i` always moves away, so nothing ever stays where it started. That is a known algorithm with a different purpose, Sattolo's algorithm: it produces only the `(n-1)!` orders that form one big cycle - 2 of 6 for three items, 6 of 24 for four.

Fisher-Yates fills the slots from the back. Slot `i` gets a uniform pick from the `i + 1` items not yet placed, so there are exactly `n!` runs and each gives a different order.

```python
class Solution:
    def __init__(self, nums: list[int]):
        self.original = list(nums)

    def reset(self) -> list[int]:
        return list(self.original)

    def shuffle(self) -> list[int]:
        a = list(self.original)
        for i in range(len(a) - 1, 0, -1):
            j = random.randrange(i + 1)
            a[i], a[j] = a[j], a[i]
        return a
```

(`random.shuffle` is Fisher-Yates; use it in real code.)

## Reservoir sampling: LeetCode 382 and LeetCode 398

### LeetCode 382 - Linked List Random Node

**The task.** Given a singly linked list, build a class whose `getRandom()` returns the value of a node chosen uniformly at random.

**Input / output.** The constructor gets the head of the list. `getRandom()` returns one node value; every node must have the same chance.

**Example.** For the list `1 -> 2 -> 3`, each call to `getRandom()` returns 1, 2 or 3, each with chance 1/3.

**Constraints.** Between 1 and 10,000 nodes, up to 10,000 calls. Copying the list into an array and indexing it is accepted. The follow-up is the interesting part: what if the list is too long to store, or its length is unknown? Then you get one pass and constant memory.

[leetcode.com/problems/linked-list-random-node](https://leetcode.com/problems/linked-list-random-node/)

### LeetCode 398 - Random Pick Index

**The task.** Given an integer array that may contain duplicates, build a class whose `pick(target)` returns a random index `i` with `nums[i] == target`, each such index equally likely.

**Input / output.** The constructor gets `nums`; `pick(target)` returns an index. The target is guaranteed to be in the array.

**Example.** For `nums = [1, 2, 3, 3, 3]`, `pick(3)` returns 2, 3 or 4, each with chance 1/3; `pick(1)` always returns 0.

**Constraints.** Up to 20,000 elements, up to 10,000 calls. A dictionary from value to its list of indices is the fast answer; the one-pass, no-extra-memory answer is reservoir sampling over the matching positions.

[leetcode.com/problems/random-pick-index](https://leetcode.com/problems/random-pick-index/)

Walk the stream once and keep the `i`-th item (counting from 1) with chance `1/i`. Item `j` is taken with chance `1/j` and then survives every later step: `1/j · j/(j+1) · (j+1)/(j+2) · ... · (n-1)/n = 1/n`. The length is never needed.

```python
def reservoir_one(stream, randrange):
    chosen = None
    for i, x in enumerate(stream, 1):
        if randrange(i) == 0:          # 1 chance in i
            chosen = x
    return chosen
```

Drop the `1` from `enumerate` and the second item is kept with chance `1/1`. It always replaces the first, so the first never comes back:

```text
reservoir 1/i       a:1/5  b:1/5  c:1/5  d:1/5  e:1/5
randrange(i) slip   a:0    b:1/4  c:1/4  d:1/4  e:1/4
```

![img](https://github.com/KingRei/100DaysPython/blob/master/day%2042%20-%20randomized%20algorithms%20and%20monte%20carlo/imgs/day42_4.png?raw=true)

On LeetCode 382 with `[1, 2, 3]` and 30,000 calls, the correct version gave `{1: 9977, 2: 9958, 3: 10065}`; the slip gave `{2: 15042, 3: 14958}`. Note that the slip passes the example "returns 1, 2 or 3" - it returns only 2 or 3, and no single call can prove that wrong. To keep `k` items instead of one (Algorithm R), item `i` (from 0) replaces a random slot with chance `k/(i+1)`; on `k = 2` out of 5 items, each of the 10 pairs comes out with exactly 1/10.

```python
class Solution:                      # LeetCode 382
    def __init__(self, head: Optional[ListNode]):
        self.head = head

    def getRandom(self) -> int:
        chosen, i, node = 0, 0, self.head
        while node:
            i += 1
            if random.randrange(i) == 0:
                chosen = node.val
            node = node.next
        return chosen
```

LeetCode 398 is the same loop, counting only the positions that match:

```python
class Solution:                      # LeetCode 398
    def __init__(self, nums: list[int]):
        self.nums = nums

    def pick(self, target: int) -> int:
        chosen, seen = -1, 0
        for i, v in enumerate(self.nums):
            if v == target:
                seen += 1
                if random.randrange(seen) == 0:
                    chosen = i
        return chosen
```

For `[1, 2, 3, 3, 3]`, 30,000 calls of `pick(3)` gave `{2: 9919, 3: 10010, 4: 10071}`.

## Rejection sampling: LeetCode 470

### LeetCode 470 - Implement Rand10() Using Rand7()

**The task.** You are given `rand7()`, which returns a uniform random integer from 1 to 7. Using only it, write `rand10()`, which returns a uniform random integer from 1 to 10.

**Input / output.** No arguments; `rand10()` returns an integer in 1..10. You may not call any other random function.

**Example.** Three calls to `rand10()` might return `[2, 8, 10]`; the judge only checks the range and the overall distribution.

**Constraints.** Up to 100,000 calls. The follow-ups ask for the expected number of `rand7()` calls per `rand10()` and how to lower it - which is what separates a correct answer from a good one.

[leetcode.com/problems/implement-rand10-using-rand7](https://leetcode.com/problems/implement-rand10-using-rand7/)

Two `rand7` calls give 7 × 7 = 49 equally likely cells, numbered 1 to 49. 49 does not divide by 10, so take the first 40 - four copies of 1..10 - and if the cell is one of the last 9, throw it away and draw again.

![img](https://github.com/KingRei/100DaysPython/blob/master/day%2042%20-%20randomized%20algorithms%20and%20monte%20carlo/imgs/day42_5.png?raw=true)

```python
class Solution:
    def rand10(self) -> int:
        while True:
            idx = (rand7() - 1) * 7 + rand7()    # 1..49, uniform
            if idx <= 40:
                return (idx - 1) % 10 + 1
```

One round accepts with chance 40/49, and once accepted each digit has exactly 1/10. Skipping the rejection and returning `idx % 10 + 1` gives the digit 1 only four cells (4/49) and every other digit five (5/49). The expected cost is `2 × 49/40 = 2.45` calls of `rand7`; over 200,000 calls it measured 2.456. The follow-up trick is to not throw the rejected cell away: a rejected `idx` in 41..49 is itself a uniform number from 1 to 9, which together with one more `rand7` gives 63 cells, of which 60 are usable, and so on. That brought it down to 2.193 calls. On 100,000 calls of the plain version, every digit came out between 9,882 and 10,254 times.

Rejection sampling is the general tool: to sample from an awkward shape, sample from a simple one that covers it and discard what falls outside. The dartboard for pi was the same move.

## Weighted pick by prefix sums: LeetCode 528

### LeetCode 528 - Random Pick with Weight

**The task.** Given a list of positive weights `w`, build a class whose `pickIndex()` returns index `i` with probability `w[i] / sum(w)`.

**Input / output.** The constructor gets `w`; `pickIndex()` returns an index.

**Example.** `w = [1, 3]`: `pickIndex()` should return 0 a quarter of the time and 1 three quarters of the time.

**Constraints.** Up to 10,000 weights, each up to 100,000, and up to 10,000 calls. With that many calls, scanning all weights on every call is slow; build prefix sums once and binary search each pick in `O(log n)`.

[leetcode.com/problems/random-pick-with-weight](https://leetcode.com/problems/random-pick-with-weight/)

Prefix sums (Day 22) cut the whole numbers `0..total-1` into consecutive slices, one per index, each as wide as its weight. For `w = [1, 3]`, the prefix is `[1, 4]`: target 0 belongs to index 0, targets 1, 2, 3 to index 1. Draw a target and binary search for the first prefix that is *greater* than it:

```python
class Solution:
    def __init__(self, w: list[int]):
        self.prefix = list(accumulate(w))        # [1, 3] -> [1, 4]

    def pickIndex(self) -> int:
        target = random.randrange(self.prefix[-1])   # 0..total-1
        return bisect_right(self.prefix, target)     # first prefix > target
```

40,000 picks on `[1, 3]` gave `{0: 10011, 1: 29989}`.

![img](https://github.com/KingRei/100DaysPython/blob/master/day%2042%20-%20randomized%20algorithms%20and%20monte%20carlo/imgs/day42_6.png?raw=true)

The common bug mixes two conventions: `random.randint(0, total)` (which includes `total`, so `total + 1` values) with `bisect_left`. Target 1 now lands on index 0 too, so index 0 gets two of five values:

| w | correct | `randint(0, total)` + `bisect_left` |
|---|---|---|
| [1, 3] | 1/4, 3/4 | 2/5, 3/5 |
| [1, 99] | 1/100, 99/100 | 2/101, 99/101 |

The rare item is now twice as likely. Use either `randrange(total)` with `bisect_right`, or `randint(1, total)` with `bisect_left`; both are correct, but they do not mix.

This exact step is how a language model picks its next word. It turns its scores ("logits") into probabilities with `softmax(logits / temperature)` and then does the prefix-sum-and-binary-search pick above. For three words with logits `the 2.0, a 1.0, cat 0.1`:

| temperature | the | a | cat |
|---|---|---|---|
| 0.5 | 0.864 | 0.117 | 0.019 |
| 1.0 | 0.659 | 0.242 | 0.099 |
| 2.0 | 0.502 | 0.304 | 0.194 |

A low temperature sharpens the distribution toward the top word; a high one flattens it. A bug like the one above would quietly double the chance of a rare token.

## Complexity

| Operation | Time | Space |
|---|---|---|
| Monte Carlo estimate with N samples | O(N), error about 1/sqrt(N) | O(1) |
| Random-pivot quicksort | O(n log n) expected, O(n²) worst | O(log n) expected |
| Miller-Rabin, k bases | O(k log n) multiplications | O(1) |
| Exact distribution of a sampler | one run per sequence of choices | O(choices) |
| Fisher-Yates shuffle (LeetCode 384) | O(n) | O(n) for the copy |
| Reservoir sampling, one item (LeetCode 382 / 398) | O(n) per pick, one pass | O(1) |
| Algorithm R, k items | O(n) | O(k) |
| rand10 from rand7 (LeetCode 470) | 2.45 rand7 calls expected | O(1) |
| Weighted pick (LeetCode 528) | O(n) build, O(log n) per pick | O(n) |

## References

- [Monte Carlo method - Wikipedia](https://en.wikipedia.org/wiki/Monte_Carlo_method) and [Las Vegas algorithm](https://en.wikipedia.org/wiki/Las_Vegas_algorithm)
- [Miller-Rabin primality test - Wikipedia](https://en.wikipedia.org/wiki/Miller%E2%80%93Rabin_primality_test) and [Carmichael number](https://en.wikipedia.org/wiki/Carmichael_number)
- [Fisher-Yates shuffle - Wikipedia](https://en.wikipedia.org/wiki/Fisher%E2%80%93Yates_shuffle), including Sattolo's algorithm and the naive-shuffle bias
- [Reservoir sampling - Wikipedia](https://en.wikipedia.org/wiki/Reservoir_sampling)
- [Rejection sampling - Wikipedia](https://en.wikipedia.org/wiki/Rejection_sampling)
- [Python docs: `random`](https://docs.python.org/3/library/random.html) - `randrange` excludes the stop value, `randint` includes it
- Donald Knuth, *The Art of Computer Programming*, Vol. 2, sections 3.4.2 (random sampling and shuffling) and 4.5.4 (factoring into primes)
- Motwani and Raghavan, *Randomized Algorithms*, chapters 1 and 14
