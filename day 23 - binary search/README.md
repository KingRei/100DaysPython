# Binary search - lower_bound, upper_bound and the three boundary variants

More details in:
https://medium.com/100-days-of-python

Interactive walkthrough: [demo.html](demo.html) - a single self-contained page
(no build step, works offline, 中文 / English toggle) that animates `lower_bound` and
`upper_bound` on the same array, three loops that look correct and are not - two that hang and
one that quietly returns the wrong index, binary search on an answer space with no array in
sight, the same search over a *noisy* float predicate where the measurement disagrees with the
truth, sglang's galloping prefix match in the radix cache, `searchsorted(cu_seqlens, tok,
right=True) - 1` next to the `right=False` version that is wrong at the first token of every
request, and LeetCode 34 solved by bracketing with two boundaries.

![img](https://github.com/KingRei/100DaysPython/blob/master/day%2023%20-%20binary%20search/imgs/day23_1.png?raw=true)

Binary search is the first algorithm most people learn and one of the last they get right.
The version in the textbook - halve the range until `a[mid] == x` - is almost never the version
that ships. Real code asks for a boundary: the first index at or after a key, the smallest
batch size that a captured CUDA graph can hold, the lowest eating speed that finishes in time,
the request that owns a given token. All of those are the same question, "where does false
turn into true", and none of them is answered by an equality test.

The half-open range makes that question easy to state. Keep `lo` and `hi` such that everything
in `a[:lo]` is known too small and everything in `a[hi:]` is known big enough, start from
`[0, len(a))`, and never break early. The invariant is preserved by `lo = mid + 1` when `mid`
is too small and by `hi = mid` when it might be the answer, and when the range becomes empty
the two pointers have met on the boundary. There is no not-found branch, because `lo == hi`
already says everything.

`lower_bound` and `upper_bound` differ by one character: `a[mid] < x` versus `a[mid] <= x`,
which is only the question of whether ties belong on the left. They are `bisect_left` and
`bisect_right` in the standard library, their difference is the number of occurrences of `x`,
and `upper_bound(x) - 1` answers "which bucket does x fall into" - the operation that undoes a
prefix sum.

## Three loops that look right

There are exactly two ways to get binary search wrong, and they are not equally visible. A
loop can fail to shrink the range - a closed interval `[lo, hi]` combined with `hi = mid` spins
forever once `lo == hi`, and half-open with `lo = mid` instead of `lo = mid + 1` spins as soon
as `hi - lo == 1`, because `(lo + hi) // 2` is then `lo`. Both hang, which at least gets
noticed.

The other failure mode is a range that discards the answer. Writing `hi = mid - 1` in a
half-open loop terminates, raises nothing, and returns a perfectly plausible index that is one
too small, because `a[mid]` was the element the loop had just proved might be the boundary. On
`[1, 2, 3, 4, 5, 6, 7]` searching for the first element at or after 4 it returns 2 where the
answer is 3, and outside the function there is nothing left to look at.

`mid = lo + (hi - lo) // 2` instead of `(lo + hi) // 2` is not fastidiousness either. Python
integers do not overflow, but the same loop inside a CUDA kernel indexes `int32` offsets, and
`2_000_000_000 + 2_100_000_000` wraps to `-194967296`, which halves to a negative index. That
is the bug that sat in `java.util.Arrays.binarySearch` for nine years.

## Binary search on the answer

Nothing in the loop requires an array. All it needs is a predicate `ok(x)` that is false,
false, ..., false, true, true, ..., true over the search range - a sorted array is just the
special case `ok(i) = a[i] >= x`. So the recipe becomes: work out what number the answer *is*,
write `ok(candidate)` - usually a simulation rather than a comparison - argue that it is
monotone, and run the same `lower_bound` loop.

The third step is the entire job. If `ok` is not monotone the loop still terminates and still
returns a number, and that number means nothing. LC 875 is monotone because eating faster can
never take longer; LC 410 is monotone because a larger cap can never need more chunks. Finding
the minimum eating speed for `piles = [30, 11, 23, 4, 20]` in 6 hours takes five probes over a
range of thirty, and the row of true/false values those probes are searching is never built.

## When the predicate is expensive and lies

sglang's autotuner binary searches the request rate: it asks "did the server still meet the SLA
at this qps?", and every answer costs a full benchmark run. Two things change once the
predicate is a measurement. Termination can no longer be `lo < hi`, because a float interval
can always be halved again, so the loop ends on a tolerance and a round budget - and a rounded
midpoint can land exactly on an endpoint, which is the float twin of the integer infinite loop
and needs its own guard. More importantly the predicate is no longer monotone: run-to-run noise
of ±0.6 qps around a true capacity of 13.7 makes two probes out of ten disagree with the truth,
and binary search never revisits a discarded half. A wrong answer in the middle of the search
is unrecoverable, which is why the real loop keeps the best *record* that actually passed
rather than a bracket, and why the tolerance has to be wider than the noise.

## Galloping, and what a probe actually costs

sglang's radix cache asks how long a prefix two token sequences share, on every incoming
request. It does not walk token by token, and it does not binary search the whole range either.
`RadixKey.match` compares windows of length 1, 2, 4, 8, ... until one differs, then binary
searches inside that window alone.

The reason is the cost model. Counting probes, plain binary search looks unbeatable - twelve or
thirteen of them on 4096 tokens whatever the shared prefix is. But a probe compares a *slice*,
not an element, so probes are neither free nor equal. Counting tokens touched, binary search
does about `n log n` work while galloping does `O(p)`: the windows double, so their lengths sum
to roughly `2p`, and only the last one is searched. Galloping is not uniformly better - at
`p ≈ n/2` it makes about twice as many probes - but a prefix cache is a bimodal workload, where
a new conversation shares almost nothing and a follow-up turn shares almost everything.

## Undoing a prefix sum

Yesterday's counting sort turned a histogram into an offset table with an exclusive scan. Every
batched kernel in an inference server carries that table as `cu_seqlens`, and every kernel that
works one token at a time has to undo it: given a flat token index, which request owns it?

    seq_of = torch.searchsorted(cu_seqlens, tok, right=True) - 1

Read out loud, that is "the last start that is at or before `tok`" - the definition of which
bucket a value falls into, and exactly `upper_bound(tok) - 1`. `right=False` asks for the first
start at or after `tok`, and at a boundary that start is the *next* request, so the first token
of every request gets the wrong id and token 0 gets `-1`, which then indexes the last entry of
`cu_seqlens` and reads another request's state. Nothing raises.

The mirror image of that search rounds a batch size *up*. A CUDA graph is captured for a fixed
shape, so a decode step of 37 requests replays on the smallest captured shape that can hold it,
and sglang's `_pad_to_bucket` is one `bisect.bisect_left` over the capture list. `bisect_left`,
not `bisect_right`: when the batch already is a captured size, `bisect_left` returns that size
and nothing is padded, while `bisect_right` steps to the next bucket and pays for a whole
graph's worth of empty rows - 6.07% average padding instead of 4.25%, from one character - and
runs off the end of the list at the largest batch.

## The problems, stated in full

Restated in my own words - what is being asked, what goes in and comes out, one worked
example, and the constraints that actually change which algorithm is allowed.

### LeetCode 33 - Search in Rotated Sorted Array

**The task.** An ascending array of distinct integers was rotated at some unknown pivot -
`[0,1,2,4,5,6,7]` may arrive as `[4,5,6,7,0,1,2]`. Given the rotated array and a `target`,
return the index of `target`, or `-1` if it is not there.

**Input / output.** Input is `nums` and `target`; output is an index (or `-1`).

**Example.** `nums = [4,5,6,7,0,1,2]`, `target = 0` → `4`. With `target = 3` → `-1`.

**Constraints.** `1 <= len(nums) <= 5000`, values in `[-10^4, 10^4]`, **all distinct**, and
the array is guaranteed to be a rotation of a sorted array. The runtime must be `O(log n)`,
so a linear scan is not an accepted answer.

[leetcode.com/problems/search-in-rotated-sorted-array](https://leetcode.com/problems/search-in-rotated-sorted-array/)

### LeetCode 34 - Find First and Last Position of Element in Sorted Array

**The task.** Given a non-decreasing array and a `target`, return the first and the last
index at which `target` appears, as `[first, last]`. If the target is absent return
`[-1, -1]`.

**Input / output.** Input is `nums` and `target`; output is a two-element list.

**Example.** `nums = [5,7,7,8,8,10]`, `target = 8` → `[3,4]`. With `target = 6` → `[-1,-1]`.

**Constraints.** `0 <= len(nums) <= 10^5`, values in `[-10^9, 10^9]`, the array is sorted
non-decreasing (duplicates allowed). Again `O(log n)` is required - which is exactly the
"find the boundary, not the element" flavour of binary search.

[leetcode.com/problems/find-first-and-last-position-of-element-in-sorted-array](https://leetcode.com/problems/find-first-and-last-position-of-element-in-sorted-array/)

### LeetCode 410 - Split Array Largest Sum

**The task.** Cut an array of non-negative integers into exactly `k` non-empty
**contiguous** pieces. Each cut has a cost: the largest piece sum. Return the smallest
largest-piece-sum achievable over all ways of cutting.

**Input / output.** Input is `nums` and `k`; output is one integer.

**Example.** `nums = [7,2,5,10,8]`, `k = 2`. The best cut is `[7,2,5] | [10,8]` with sums
`14` and `18`, so the answer is `18` - every other cut has a larger maximum.

**Constraints.** `1 <= len(nums) <= 1000`, `0 <= nums[i] <= 10^6`,
`1 <= k <= min(50, len(nums))`. The answer lives between `max(nums)` and `sum(nums)`, and
"can I do it with a budget of `x`?" is monotone in `x` - which is what makes this a binary
search *on the answer* rather than on an array.

[leetcode.com/problems/split-array-largest-sum](https://leetcode.com/problems/split-array-largest-sum/)

### LeetCode 875 - Koko Eating Bananas

**The task.** There are `n` piles of bananas and `h` hours before the guards come back.
Koko picks an eating speed `k` (bananas per hour). Each hour she chooses one pile and eats
`k` bananas from it; if the pile has fewer than `k` left she eats the whole pile and then
*stops for that hour* - she never moves on to a second pile within the same hour. Return
the smallest integer `k` that lets her finish every pile within `h` hours.

**Input / output.** Input is `piles` and `h`; output is the minimum speed `k`.

**Example.** `piles = [3,6,7,11]`, `h = 8` → `4`. At speed `4` the hours needed are
`1 + 2 + 2 + 3 = 8`; at speed `3` they are `1 + 2 + 3 + 4 = 10`, which is too slow.

**Constraints.** `1 <= len(piles) <= 10^4`, `len(piles) <= h <= 10^9`,
`1 <= piles[i] <= 10^9`. Note the search space is the *value* `k` up to `10^9`, not the
array - another binary search on the answer.

[leetcode.com/problems/koko-eating-bananas](https://leetcode.com/problems/koko-eating-bananas/)

## Complexity

| Operation | Probes | Elements touched | Space |
|---|---|---|---|
| `lower_bound` / `upper_bound` | O(log n) | O(log n) | O(1) |
| Binary search on an answer in `[lo, hi)` | O(log(hi - lo)) | one `ok()` call each | O(1) |
| Float bisection to tolerance `t` | O(log((hi - lo) / t)) | one `ok()` call each | O(1) |
| Galloping prefix match, shared prefix `p` | O(log p) | O(p) | O(1) |
| Plain binary search over prefixes | O(log n) | O(n log n) | O(1) |
| Linear prefix match | O(p) | O(p) | O(1) |

## References
- [Binary search algorithm - Wikipedia](https://en.wikipedia.org/wiki/Binary_search_algorithm)
- [Exponential search - Wikipedia](https://en.wikipedia.org/wiki/Exponential_search)
- [Extra, extra - read all about it: nearly all binary searches and mergesorts are broken](https://research.google/blog/extra-extra-read-all-about-it-nearly-all-binary-searches-and-mergesorts-are-broken/)
- [`bisect` - Array bisection algorithm](https://docs.python.org/3/library/bisect.html)
- [`torch.searchsorted`](https://pytorch.org/docs/stable/generated/torch.searchsorted.html)
- [LeetCode 34 · Find First and Last Position of Element in Sorted Array](https://leetcode.com/problems/find-first-and-last-position-of-element-in-sorted-array/)
- [LeetCode 33 · Search in Rotated Sorted Array](https://leetcode.com/problems/search-in-rotated-sorted-array/)
- [LeetCode 875 · Koko Eating Bananas](https://leetcode.com/problems/koko-eating-bananas/)
- [LeetCode 410 · Split Array Largest Sum](https://leetcode.com/problems/split-array-largest-sum/)
