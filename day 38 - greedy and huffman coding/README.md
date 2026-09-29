# Greedy algorithms: interval scheduling and Huffman coding

More details in:
https://medium.com/100-days-of-python

Interactive walkthrough: [demo.html](demo.html) - a single self-contained page
(no build step, works offline, 中文 / English toggle) that seats five of eight meetings
by always taking the one that frees the room first, replays the same loop under two
other sort keys that quietly seat three and four, checks the exchange argument against
a brute force over all 2^8 subsets, builds a Huffman tree by pulling the two lightest
subtrees off a heap, rebuilds it without ever putting the merged node back and watches
the tree fall into a ladder that still decodes perfectly, and then solves LeetCode 435
and LeetCode 134 with the same one-pass shape.

![img](https://github.com/KingRei/100DaysPython/blob/master/day%2038%20-%20greedy%20and%20huffman%20coding/imgs/day38_1.png?raw=true)

A greedy algorithm takes the choice that looks best right now and never reconsiders.
That is the entire idea, and it is why greedy code is so short: one sort, one pass, no
table. Days 33 to 37 spent five days building tables precisely because the future had to
be taken into account; today is the opposite claim - that for some problems, a single
local rule is already optimal.

The difficulty is not writing a greedy. It is knowing that the key you sorted by is the
right one. **A wrong greedy key does not crash.** It returns a schedule with no clashes,
a code that decodes perfectly, a way to pay the exact amount. It is simply worse than the
best answer, and nothing in the program says so. Every section below pairs a greedy with
a ground truth so that the gap becomes a number instead of a suspicion.

The two examples are the classics. Interval scheduling is the smallest problem where
choosing the wrong key is genuinely tempting. Huffman coding is the one where the greedy
is provably optimal, and where the proof turns on a single detail of the data structure -
the merged node has to go back into the queue.

## Interval scheduling: take whoever frees the room first

Eight people want the same meeting room. The intervals are half-open, so a meeting that
ends at 11:00 does not clash with one that starts at 11:00. Sort by finish time, walk the
list once, accept a meeting whenever it starts at or after the end of the last one
accepted:

```python
def select_by_end(intervals):
    out, last_end = [], float('-inf')
    for s, e in sorted(intervals, key=lambda iv: iv[1]):
        if s >= last_end:            # no clash with what we already took
            out.append((s, e))
            last_end = e
    return out
```

That seats five of the eight: B, C, D, F and G. The reason the finish time is the right
key and not just *a* key is that the only thing constraining everything still to come is
the clock time at which the room becomes free again. Taking the meeting that finishes
earliest leaves the room free at the earliest possible moment, so it can never close off
an option that some other choice would have kept open.

## Two other keys that look just as reasonable

![img](https://github.com/KingRei/100DaysPython/blob/master/day%2038%20-%20greedy%20and%20huffman%20coding/imgs/day38_2.png?raw=true)

"Whoever asked first goes first" is how a human actually runs a room. "Pack the short
meetings first" is how you would fit socks into a drawer. Both are defensible, both fit
in the same four lines, and both lose:

| sort key | meetings seated | what it picks |
|---|---|---|
| by finish time | **5** | B C D F G |
| by start time | 3 | A E G |
| by duration | 4 | B C E G |
| brute force over all 2^8 subsets | 5 | B C D F G |

Sorting by start time hands the room to A, a two-hour design review that blocks 09:15 to
11:15 and with it the three short meetings underneath. Sorting by duration takes the two
fifteen-minute meetings because they look cheap, and then cannot fit D.

The point is not the ranking. It is that all three outputs are legal, clash-free
schedules. No exception, no warning - the wrong keys just seat fewer people, and in a
problem with no brute force to compare against, nobody would ever find out.

## Why the finish time is the right key: the exchange argument

![img](https://github.com/KingRei/100DaysPython/blob/master/day%2038%20-%20greedy%20and%20huffman%20coding/imgs/day38_3.png?raw=true)

Proofs that a greedy is optimal almost always have the same shape. Claim: the interval
that finishes first belongs to *some* optimal schedule. Take any optimal schedule that
does not contain it and swap its first meeting for the earliest finisher. The swap cannot
create a clash, because the replacement ends no later than what it replaced, so the result
is a schedule of the same size that does contain the greedy choice. The greedy choice
therefore costs nothing, and the argument then applies again to whatever is left.

Run as an experiment rather than argued, that is a three-line check: compare the best
schedule overall with the best schedule *forced* to contain the earliest finisher.

```python
first = min(intervals, key=lambda iv: iv[1])
unrestricted = len(select_bruteforce(intervals))
rest = [iv for iv in intervals if compatible(iv, first) and iv != first]
forced = 1 + len(select_bruteforce(rest))
```

Both come back 5. This is worth doing on any greedy you are unsure about: force the first
greedy choice, brute-force the rest, and see whether the answer got worse.

## Change the question slightly and the greedy dies

Now every meeting is worth money and the goal is the most valuable schedule rather than
the fullest one. The greedy still runs and still returns a real schedule:

| | total value | schedule |
|---|---|---|
| greedy by finish time | 12 | B C D F G |
| DP over the same intervals | **17** | A F H |

No sort key can rescue this, because the value of taking a meeting now depends on what
was given up for it - which is exactly the situation a table exists to handle. Weighted
interval scheduling is Day 33's DP with Day 23's `bisect` finding, for each interval, the
last one that finishes before it starts:

```python
iv = sorted(intervals, key=lambda t: t[1])
ends = [e for _, e in iv]
best = [0] * (len(iv) + 1)
for i, (s, e) in enumerate(iv):
    j = bisect_right(ends, s, 0, i)      # last compatible interval, 0 if none
    best[i + 1] = max(best[i], best[j] + value[(s, e)])
```

Five units of value left on the table, and a perfectly valid schedule handed back. This
is the same failure mode as the wrong sort key, one step further along: the greedy is not
just using the wrong key, it is the wrong tool - and it still does not complain.

## Huffman coding

![img](https://github.com/KingRei/100DaysPython/blob/master/day%2038%20-%20greedy%20and%20huffman%20coding/imgs/day38_4.png?raw=true)

A fixed-length code spends as many bits on the rarest symbol as on the most common one.
Huffman's observation is that the two rarest symbols can always be placed as siblings at
the deepest level of the code tree, because nothing is lost by giving the longest
codewords to the symbols that are used least. That turns into an algorithm with no
lookahead at all: take the two lightest subtrees off a min-heap, merge them, push the
merged node back, repeat until one tree remains. Left branch is a `0`, right is a `1`.

```python
def huffman(freqs):
    heap = [(w, i, sym) for i, (sym, w) in enumerate(sorted(freqs.items()))]
    heapq.heapify(heap)
    tie = len(heap)
    merges = []
    while len(heap) > 1:
        w1, _, n1 = heapq.heappop(heap)
        w2, _, n2 = heapq.heappop(heap)
        merges.append((n1, n2, w1 + w2))
        heapq.heappush(heap, (w1 + w2, tie, (n1, n2)))   # back into the queue
        tie += 1
    ...
```

On the 53-character sentence `'greedy algorithms are easy to write and hard to prove'`
there are 18 distinct characters, so a fixed-length code needs 5 bits each and 265 bits
in total. Huffman gets it to 204 bits, 23% smaller. The space character, which
appears nine times, gets a 3-bit codeword; the longest codeword in the tree is 6 bits
and goes to a letter that appears once.

Two details in that snippet are load-bearing. The `tie` counter keeps the ordering
deterministic when two weights are equal - without it Python falls through to comparing
the node payloads, and the output would depend on how tuples of tuples happen to order.
And the `heappush` is not a formality: a merged node is heavier than either child and
must re-enter the queue at its new position. That is the entire reason this needs a heap
rather than a sorted list, and it is what the next section breaks.

Decoding needs no separators. No codeword is a prefix of another - walking to a leaf
guarantees that - so a decoder reads bits one at a time and emits a symbol the moment
what it has read is a codeword. A cheap structural check is the Kraft sum, the sum of
2^-length over all codewords: exactly 1.0 means the code tree has no branch leading
nowhere, and anything below 1.0 means bits are being spent on a shape nothing uses.

## The silent failure: merging without re-sorting

![img](https://github.com/KingRei/100DaysPython/blob/master/day%2038%20-%20greedy%20and%20huffman%20coding/imgs/day38_5.png?raw=true)

Here is the version to remember, because it is the one that produces no error. Sort the
symbols by frequency once, then merge left to right and never put the merged node back
into the queue:

```python
items = sorted(freqs.items(), key=lambda kv: (kv[1], kv[0]))
node, weight = items[0][0], items[0][1]
for sym, w in items[1:]:
    node, weight = ((node, sym), weight + w)
```

Every merge still joins two subtrees, so the result is still a legal prefix code and still
decodes the original text exactly. What is lost is the re-sort. A merged node whose weight
has grown past its neighbours should sink back down the queue; here it never does, so it
is always one of the two lightest and the tree degenerates into a ladder hanging off one
side. The longest codeword goes from 6 bits to 17.

| code | bits for the same sentence |
|---|---|
| Huffman with a heap | 204 |
| plain 5-bit fixed-length code | 265 |
| sorted once, merged left to right | 316 |

Both codes are prefix-free. Both round-trip the text exactly. Both have a Kraft sum of
exactly 1.000, so even the structural check passes. The broken one is simply 55% bigger
than the correct one, and worse than not compressing at all - and the only way to find
out is to compare it against something.

## The classic counterexample: coins 1, 3, 4

Paying 6 with coins of 1, 3 and 4: the greedy takes the largest coin that still fits, so
4 + 1 + 1, three coins. Two coins of 3 would have done it. Greedy's answer is a real way
to pay the exact amount - it is just not the fewest coins, and the only signal is the
count.

What makes this trap durable is that with the denominations actually in a wallet - 1, 5,
10, 25 - the same greedy is always optimal. Those systems are designed so that it is. The
habit gets formed on coins that forgive it and then carried to denominations that do not.

## LeetCode 435 - Non-overlapping Intervals

**The task.** Given a list of intervals, remove as few of them as possible so that none of
the survivors overlap, and return how many were removed.

**Input / output.** A list of `[start, end]` pairs. Returns a single integer: the minimum
number of intervals to delete.

**Example.** `[[1,2], [2,3], [3,4], [1,3]]` returns `1`. Deleting `[1,3]` leaves
`[1,2]`, `[2,3]` and `[3,4]`, which meet only at their endpoints and so do not overlap.
Deleting any single one of the other three would still leave `[1,3]` clashing with two
survivors, so one deletion is both necessary and sufficient.

**Constraints.** Up to 100,000 intervals, with coordinates fitting comfortably in a signed
32-bit integer. The interval count is the constraint that decides the algorithm: anything
that compares intervals pairwise is 10^10 operations and will time out, so the only
affordable shapes are a sort plus a linear pass. The other constraint is hidden in the
wording - intervals that merely touch at an endpoint are *not* considered overlapping,
which is why the test is `s >= last_end` and not `s > last_end`.

This is the meeting room with the answer inverted: maximising what is kept is the same
problem as minimising what is thrown away, so the by-finish-time greedy solves it
unchanged.

```python
class Solution:
    def eraseOverlapIntervals(self, intervals: List[List[int]]) -> int:
        intervals.sort(key=lambda iv: iv[1])        # earliest finishing first
        kept, last_end = 0, float('-inf')
        for s, e in intervals:
            if s >= last_end:                       # fits after what we kept
                kept += 1
                last_end = e
        return len(intervals) - kept
```

[leetcode.com/problems/non-overlapping-intervals](https://leetcode.com/problems/non-overlapping-intervals/)

## LeetCode 452 - Minimum Number of Arrows to Burst Balloons

**The task.** Balloons are drawn as horizontal segments. An arrow shot straight up at
position `x` bursts every balloon whose segment covers `x`, and keeps travelling, so one
arrow can burst any number of balloons. Find the fewest arrows that burst them all.

**Input / output.** A list of `[x_start, x_end]` pairs, one per balloon. Returns the
minimum number of arrows.

**Example.** `[[10,16], [2,8], [1,6], [7,12]]` returns `2`. Sorted by right edge the
balloons are `[1,6]`, `[2,8]`, `[7,12]`, `[10,16]`. An arrow at `x = 6` bursts the first
two, because both segments still cover 6; an arrow at `x = 12` bursts the other two. One
arrow cannot do it, because `[1,6]` and `[10,16]` share no point.

**Constraints.** Up to 100,000 balloons, coordinates anywhere in the signed 32-bit range.
The coordinate range is the one that bites: the balloons are **closed** intervals, so an
arrow at `x = 2` bursts both `[1,2]` and `[2,3]`, and a new arrow is needed only when the
next balloon starts *strictly after* the last arrow's position. One character - `>`
instead of `>=` - is the entire difference between this problem and LeetCode 435.

```python
class Solution:
    def findMinArrowShots(self, points: List[List[int]]) -> int:
        points.sort(key=lambda p: p[1])
        arrows, last = 0, float('-inf')
        for s, e in points:
            if s > last:                            # touching counts as a hit
                arrows += 1
                last = e
        return arrows
```

[leetcode.com/problems/minimum-number-of-arrows-to-burst-balloons](https://leetcode.com/problems/minimum-number-of-arrows-to-burst-balloons/)

## LeetCode 621 - Task Scheduler

**The task.** A CPU runs one task per unit of time, or idles. Two runs of the *same* task
must be separated by at least `n` units of cooldown; different tasks have no such
restriction. Given the list of tasks to run, find the least total time to finish them all.

**Input / output.** A list of task labels (single uppercase letters) and an integer
cooldown `n`. Returns the minimum number of time units, counting idle units.

**Example.** `tasks = ["A","A","A","B","B","B"], n = 2` returns `8`. The schedule
`A B idle A B idle A B` takes 8 units; there is no way to do it in 7, because the three
`A`s alone already span `A _ _ A _ _ A`, which is 7 units, and the final `B` has to go
somewhere after the last `A`.

**Constraints.** Up to 10,000 tasks, but only 26 distinct labels, and `0 <= n <= 100`.
The alphabet bound is what changes the algorithm: with at most 26 counts, the answer
depends only on the largest count and on how many tasks tie with it, so no simulation is
needed at all. The most frequent task, with `peak` copies, forces `peak - 1` complete
blocks of width `n + 1`, plus one final row holding every task tied at `peak`. If there
are enough distinct tasks to fill the gaps, no idle time exists and the answer is just
`len(tasks)` - which is why the formula is wrapped in a `max`.

```python
class Solution:
    def leastInterval(self, tasks: List[str], n: int) -> int:
        counts = Counter(tasks)
        peak = max(counts.values())
        tied = sum(1 for c in counts.values() if c == peak)
        return max((peak - 1) * (n + 1) + tied, len(tasks))
```

[leetcode.com/problems/task-scheduler](https://leetcode.com/problems/task-scheduler/)

## LeetCode 1046 - Last Stone Weight

**The task.** Repeatedly take the two heaviest stones and smash them together. If they
weigh the same, both are destroyed; otherwise the heavier one survives with the
difference as its new weight. Return the weight of the stone left at the end, or 0 if
none is left.

**Input / output.** A list of positive integer weights. Returns a single integer.

**Example.** `[2,7,4,1,8,1]` returns `1`. Smashing 8 against 7 leaves a 1, giving
`[2,4,1,1,1]`; 4 against 2 leaves a 2, giving `[2,1,1,1]`; 2 against 1 leaves a 1,
giving `[1,1,1]`; then 1 against 1 destroys both, leaving `[1]`.

**Constraints.** At most 30 stones, each at most 1000. The bounds are tiny - re-sorting
the whole list after every smash would pass easily - so the heap here is about the shape
of the code rather than about the time limit. The shape is worth noticing: this is
Huffman's loop upside down. Huffman pops the two *lightest* and pushes their sum; this
pops the two *heaviest* and pushes their difference. `heapq` only builds min-heaps, so
the weights go in negated.

```python
class Solution:
    def lastStoneWeight(self, stones: List[int]) -> int:
        heap = [-s for s in stones]                 # heapq is a min-heap, so negate
        heapq.heapify(heap)
        while len(heap) > 1:
            a = -heapq.heappop(heap)
            b = -heapq.heappop(heap)
            if a != b:
                heapq.heappush(heap, -(a - b))
        return -heap[0] if heap else 0
```

[leetcode.com/problems/last-stone-weight](https://leetcode.com/problems/last-stone-weight/)

## LeetCode 134 - Gas Station

**The task.** `n` gas stations sit in a circle. Station `i` holds `gas[i]` units of fuel,
and driving from station `i` to the next one costs `cost[i]` units. Starting with an
empty tank at a station of your choice, return an index you can start from and complete
the full loop once, or `-1` if no such start exists.

**Input / output.** Two integer lists of the same length. Returns the starting index, or
`-1`. The problem guarantees the answer is unique when one exists.

**Example.** `gas = [1,2,3,4,5]`, `cost = [3,4,5,1,2]` returns `3`. Starting at station 3
the tank goes 4, then 4 - 1 + 5 = 8, then 8 - 2 + 1 = 7, then 7 - 3 + 2 = 6, then
6 - 4 + 3 = 5, and the loop closes. The fuel and the cost sum to 15 each, so the tank
ends exactly where it started - the circuit is only just possible, and starting anywhere
else runs the tank negative before getting back around.

**Constraints.** Up to 100,000 stations, with each `gas[i]` and `cost[i]` at most 10,000.
The station count rules out the obvious solution of trying every start and driving the
loop, which is 10^10 steps. The uniqueness guarantee matters too: it is what makes it
safe to return the single-pass candidate without going back and verifying it.

This is the one greedy in the set whose correctness argument is not an exchange argument.
If the total fuel covers the total cost then a valid start must exist. And if the tank
goes negative somewhere between `start` and station `i`, then no station in that stretch
can be the answer either - each of them begins the same stretch with even less in the
tank - so the search jumps straight past all of them to `i + 1`. Every station is visited
once, and the whole thing is one pass.

```python
class Solution:
    def canCompleteCircuit(self, gas: List[int], cost: List[int]) -> int:
        if sum(gas) < sum(cost):
            return -1                               # not enough fuel in the whole loop
        start, tank = 0, 0
        for i in range(len(gas)):
            tank += gas[i] - cost[i]
            if tank < 0:                            # i is not reachable from start
                start, tank = i + 1, 0
        return start
```

[leetcode.com/problems/gas-station](https://leetcode.com/problems/gas-station/)

## Complexity

| Operation | Time | Space |
|---|---|---|
| interval scheduling, sort + one pass | O(n log n) | O(n) |
| brute force over all subsets (ground truth only) | O(2^n · n²) | O(n) |
| weighted interval scheduling, DP + bisect | O(n log n) | O(n) |
| Huffman, build the tree | O(k log k) for k symbols | O(k) |
| Huffman, encode / decode a text of length m | O(m) | O(k) |
| coin change, DP | O(amount · coins) | O(amount) |
| LeetCode 435, non-overlapping intervals | O(n log n) | O(1) extra |
| LeetCode 452, minimum arrows | O(n log n) | O(1) extra |
| LeetCode 621, task scheduler formula | O(n) | O(1), 26 counters |
| LeetCode 1046, last stone weight | O(n log n) | O(n) |
| LeetCode 134, gas station | O(n) | O(1) |

## References

- [Greedy algorithm - Wikipedia](https://en.wikipedia.org/wiki/Greedy_algorithm)
- [Interval scheduling - Wikipedia](https://en.wikipedia.org/wiki/Interval_scheduling)
- [Huffman coding - Wikipedia](https://en.wikipedia.org/wiki/Huffman_coding)
- David A. Huffman, "A Method for the Construction of Minimum-Redundancy Codes", 1952
- [Kraft-McMillan inequality - Wikipedia](https://en.wikipedia.org/wiki/Kraft%E2%80%93McMillan_inequality)
- [Matroid - Wikipedia](https://en.wikipedia.org/wiki/Matroid) - the structure that says *when* a greedy is guaranteed to work
