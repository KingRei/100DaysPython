# Bloom filter, HyperLogLog and Count-Min sketch: answering set questions in kilobytes

More details in:
https://medium.com/100-days-of-python

Interactive walkthrough: [demo.html](demo.html) - a single self-contained page
(no build step, works offline, 中文 / English toggle) that switches on the bits of a Bloom
filter and then catches it saying yes to a word it never saw, clears three bits to delete one
item and watches a different item disappear, splits a hash into a register index and a run of
zeros to count distinct values, reads the same key out of four Count-Min rows and takes the
minimum, and finishes on LeetCode 347 where a sketch one size too small lets a word seen seven
times climb into the top three.

![img](https://github.com/KingRei/100DaysPython/blob/master/day%2032%20-%20probabilistic%20structures/imgs/day32_1.png?raw=true)

Every structure in this series so far has been exact. A hash table either has the key or it does
not; a Fenwick tree returns the prefix sum, not approximately the prefix sum. Today's three
structures give that up on purpose, and get something specific in return: their memory stops
depending on how much data goes in. A Bloom filter for ten thousand items is 11,982 bytes, and a
Python `set` holding the same ten thousand strings is 1,103,394 bytes - 92 times larger. A
HyperLogLog is 16,384 bytes whether it has counted a thousand distinct users or five hundred
thousand.

The interesting part is not that they are wrong. It is that each one is wrong in exactly one
direction, and that direction is what decides whether you can use it. A Bloom filter only ever
says yes too often, never no too often. A Count-Min sketch only ever over-counts, never under.
HyperLogLog is the awkward one: it can miss in either direction, which is why its unions are
free and its intersections are worthless. Knowing the direction of the error matters more than
knowing its size, because the size always comes with a knob and the direction never does.

## Bloom filter: k bits switched on, and no way back

```python
def add(self, item):
    h1, h2 = hash_pair(item)
    for i in range(self.k):
        self.bits[(h1 + i * h2 + i * i) % self.m] = 1

def __contains__(self, item):
    h1, h2 = hash_pair(item)
    return all(self.bits[(h1 + i * h2 + i * i) % self.m]
               for i in range(self.k))
```

An item is turned into `k` positions in an `m`-bit array, and those bits are set. A query
recomputes the same positions: a single zero proves the item was never added, and all ones means
it was *probably* added. `k` independent hash functions are not needed - the
Kirsch-Mitzenmacher trick derives them all from two, with `g_i = h1 + i·h2 + i²`, and one
BLAKE2b digest split in half provides both.

Two formulas size the thing: `m = -n ln p / (ln 2)²` bits and `k = (m/n) ln 2` hashes. For
10,000 items at 1%, that is 9.59 bits per item and 7 hashes - and note the units, *bits per
item*, independent of how long the items are. Measured over 100,000 lookups of items that were
never added, the false positive rate comes out at 1.03% against a 1.00% design. False negatives
over the 10,000 members: zero, by construction.

That 1% is a promise about a load, not about the structure. Keep adding past the design size and
the rate does not degrade gently:

| load | items | fill | expected FPR | measured |
|---|---|---|---|---|
| 0.5x | 500 | 30.7% | 0.03% | 0.01% |
| 1.0x | 1,000 | 52.5% | 1.00% | 1.26% |
| 2.0x | 2,000 | 77.1% | 15.74% | 16.19% |
| 4.0x | 4,000 | 94.8% | 67.86% | 69.18% |
| 8.0x | 8,000 | 99.7% | 97.99% | 97.50% |

At eight times the design load the filter says yes to everything, which is not a filter at all.
The number of hashes has an optimum for the same reason: more hashes give a query more chances
to find a zero, and also fill the array faster. At 10 bits per item the two effects cross at
`k = 7`, exactly where `(m/n) ln 2` says they should.

## Delete is not a missing feature, it is an impossible one

![img](https://github.com/KingRei/100DaysPython/blob/master/day%2032%20-%20probabilistic%20structures/imgs/day32_3.png?raw=true)

A bit set by `alpha` may also have been set by `bravo`. Clearing the three bits of `alpha` in a
64-bit filter holding 14 words takes `bravo` and `india` down with it: two false negatives, in a
structure whose entire contract is that false negatives cannot happen. Nothing raises an error.
The query just starts answering no about things that are plainly in the set.

A counting Bloom filter replaces each bit with a 4-bit counter, increments on add and decrements
on delete, so a shared slot drops from 2 to 1 instead of to 0 - zero false negatives on the same
test. The price is four times the memory and saturation: once a counter reaches 15 it stops
counting up, and every later decrement is wrong.

The more expensive mistake is not deleting, it is trusting a yes. Deduplicating 20,000 documents
with a 1% filter keeps 13,932 of the 13,954 distinct ones: 22 documents were declared
"already seen" and thrown away. The filter behaved exactly as specified. As a pre-filter in
front of a real lookup, a false positive costs one wasted read; as the final answer, it costs
data that never comes back.

## HyperLogLog: a maximum that counts

![img](https://github.com/KingRei/100DaysPython/blob/master/day%2032%20-%20probabilistic%20structures/imgs/day32_4.png?raw=true)

Flip a fair coin repeatedly and a run of `r` heads turns up about once in `2^r` tries. So the
longest run you have seen is evidence of how many tries you have made - and a maximum needs one
number of storage, not a list. Hash each item, use the low `p` bits to pick one of `m = 2^p`
registers, count the run of zeros in the remaining bits, and keep the maximum per register.
Because it is a maximum, feeding the same item in a thousand times changes nothing: the sketch
deduplicates by construction.

```python
def add(self, item):
    h = hash64(item)
    idx = h & (self.m - 1)                    # low p bits pick a register
    rho = trailing_zeros(h >> self.p) + 1     # the run of zeros
    self.reg[idx] = max(self.reg[idx], rho)
```

One register is far too noisy, so the count is a harmonic mean across all `m` of them, scaled by
a constant `alpha_m` - harmonic because it damps the registers that got a freakishly long run.
Below `2.5m` the estimator is biased and linear counting takes over, using the number of
registers still at zero. With `p = 14`, that is 16,384 one-byte registers, 16 KB, and a
theoretical error of `1.04/√m` = 0.81%:

| true | estimate | error | HLL bytes | exact set bytes | ratio |
|---|---|---|---|---|---|
| 1,000 | 1,005 | 0.52% | 16,384 | 64,984 | 4x |
| 10,000 | 9,930 | -0.70% | 16,384 | 844,504 | 52x |
| 100,000 | 99,924 | -0.08% | 16,384 | 7,394,520 | 451x |
| 500,000 | 504,141 | 0.83% | 16,384 | 32,777,432 | 2,001x |

The memory column does not move. That is the entire point.

The property that makes HLL the one that actually ships is that merging is a register-wise
maximum, and it is *exact*: two shards merged give byte-identical registers to one machine
scanning everything. Store one sketch per hour and any range of hours can be rolled up
afterwards with no accumulated error.

Intersections are the trap. `|A ∩ B| = |A| + |B| - |A ∪ B|` is algebraically true and
numerically hopeless: with `|A| = |B| = 200,000` and a true intersection of 2,000, the union is
estimated to within 0.84% and the intersection comes out at 1,681 - a 16% error. Three relative
errors on numbers around 200,000 became one absolute error the size of the answer. Inclusion-
exclusion subtracts big numbers to get a small one, which is precisely what an estimator with
relative error cannot survive.

## Count-Min sketch: every row over-counts, so take the smallest

![img](https://github.com/KingRei/100DaysPython/blob/master/day%2032%20-%20probabilistic%20structures/imgs/day32_6.png?raw=true)

```python
def add(self, item, count=1):
    for i, row in enumerate(self.rows):
        row[self._hash(item, i) % self.w] += count

def query(self, item):
    return min(self.rows[i][self._hash(item, i) % self.w]
               for i in range(self.d))
```

`d` rows of `w` counters, each row a complete tally of the stream with its own collisions folded
in. A collision can only push a counter up, so every row over-counts and the smallest row is the
least polluted - and the answer is never below the truth. Sizing is `w = ⌈e/ε⌉` and
`d = ⌈ln(1/δ)⌉`, which gives an error of at most `ε·N` with probability `1 - δ`.

On 200,000 events over 4,934 distinct keys, a 5 × 5,437 sketch (108,740 bytes, against 443,608
for a `Counter`) has a guaranteed bound of +100, a worst observed error of 21, a mean of 0.3,
and zero under-counts. But look at where the error lands: +0.01% relative on the top ten keys,
+150% on the bottom ten. The bound is *absolute*, so the same 21-count error is rounding noise
on a key seen 31,648 times and the entire answer on a key seen once. A Count-Min sketch is
excellent at "which keys are big" and useless at "did this key appear exactly once".

The conservative update - only raise the counters that are currently at the minimum - drops the
mean error from 0.3 to 0.1 for free, and keeps the no-under-count guarantee. It makes delete
impossible, but delete was never on the menu.

## LeetCode 347, with and without enough memory

```python
def top_k_frequent(nums, k):
    counts = Counter(nums)
    return heapq.nlargest(k, counts, key=counts.get)
```

The textbook answer is a `Counter` plus a k-sized heap: `O(n)` time, and memory proportional to
the number of distinct keys. In an interview that is the answer, and the sketch version is the
follow-up when the interviewer says "now the array is a stream and there are a billion keys".

The streaming version adds each item to a Count-Min sketch, queries it straight back, and keeps
only `k` candidates - fixed memory, no dict of every key. On a 200,000-event Zipf stream with
`k = 10` it returns the same ten keys in the same order with the same counts, out of 108,740
bytes instead of 443,608. Then make the stream nearly flat - top key seen 65 times, tenth seen
61 - and it drops `w-385` from the list. Approximate top-k only means something when the head
stands clear of the noise floor: the rule is that the k-th true count has to be comfortably
larger than `e/w · N`.

## Picking one

![img](https://github.com/KingRei/100DaysPython/blob/master/day%2032%20-%20probabilistic%20structures/imgs/day32_7.png?raw=true)

| structure | question | error direction | mergeable | bytes @ 1e6 items |
|---|---|---|---|---|
| `set` (exact) | membership + count | none | yes (union) | 60,000,000 |
| Bloom filter | have I seen x? | false positive only | yes (OR, same m and k) | 1,198,133 |
| HyperLogLog | how many distinct? | relative, both ways | yes (register max) | 16,384 |
| Count-Min | how often is x? | over-count only | yes (add the rows) | 108,740 |

Bloom filters sit in Chrome's Safe Browsing list, in Cassandra and LevelDB SSTable lookups and
in SPV wallets - all places where a cheap no skips an expensive disk or network read.
HyperLogLog is what Redis `PFCOUNT`, BigQuery `APPROX_COUNT_DISTINCT` and Presto
`approx_distinct` are doing, rolling unique visitors up from shards. Count-Min shows up in
network flow monitoring, trending topics and per-key rate limiting, wherever heavy hitters
matter and one counter per key does not fit.

## Complexity

| Operation | Time | Space |
|---|---|---|
| Bloom `add` / `query` | `O(k)` | `-n ln p / (ln 2)²` bits, independent of item size |
| Bloom false positive rate | - | `(1 - e^(-kn/m))^k` |
| Counting Bloom `delete` | `O(k)` | 4x the plain filter |
| HLL `add` | `O(1)` | `m = 2^p` bytes, `1.04/√m` error |
| HLL `merge` | `O(m)` | exact, no added error |
| CMS `add` / `query` | `O(d)` | `d·w` counters, error `≤ e/w · N` |
| exact `set` / `Counter` | `O(1)` | `O(n)`, grows forever |

## References
- [Bloom filter](https://en.wikipedia.org/wiki/Bloom_filter)
- [Bloom, "Space/Time Trade-offs in Hash Coding with Allowable Errors" (1970)](https://dl.acm.org/doi/10.1145/362686.362692)
- [Kirsch & Mitzenmacher, "Less Hashing, Same Performance" (2006)](https://www.eecs.harvard.edu/~michaelm/postscripts/rsa2008.pdf)
- [Flajolet et al., "HyperLogLog: the analysis of a near-optimal cardinality estimation algorithm" (2007)](https://algo.inria.fr/flajolet/Publications/FlFuGaMe07.pdf)
- [Heule et al., "HyperLogLog in Practice" (Google, 2013)](https://research.google/pubs/pub40671/)
- [Cormode & Muthukrishnan, "An Improved Data Stream Summary: The Count-Min Sketch" (2005)](http://dimacs.rutgers.edu/~graham/pubs/papers/cm-full.pdf)
- [Redis HyperLogLog (PFADD / PFCOUNT / PFMERGE)](https://redis.io/docs/latest/develop/data-types/probabilistic/hyperloglogs/)
- [LeetCode 347 - Top K Frequent Elements](https://leetcode.com/problems/top-k-frequent-elements/)
