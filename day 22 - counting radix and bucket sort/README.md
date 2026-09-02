# Counting, radix and bucket sort - sorting without comparing

More details in:
https://medium.com/100-days-of-python

Interactive walkthrough: [demo.html](demo.html) - a single self-contained page
(no build step, works offline, 中文 / English toggle) that animates a counting sort turning a
histogram into an address table, an LSD radix sort next to the same radix sort with stability
removed so you can watch it produce a wrong answer, one bucket sort degrading from linear to
quadratic on nothing but a change of input distribution, Hillis-Steele and Blelloch scans with
live work and depth counters, sglang's `moe_align_block_size` sorting routed tokens into
block-aligned expert regions, and LeetCode 164 answered without ever sorting anything.

![img](https://github.com/KingRei/100DaysPython/blob/master/day%2022%20-%20counting%20radix%20and%20bucket%20sort/imgs/day22_1.png?raw=true)

Every sort so far has asked one question - is `a` less than `b`? - and a decision tree with
`n!` leaves needs `log2(n!)` of those questions, which is `n log n`. Merge sort at n = 100,000
uses 1,536,501 comparisons against a floor of 1,516,704, about one percent above it. There is
no clever comparison sort left to find; the only way down is to stop asking that question.

Counting sort stops. If a key is an integer in `0..k-1`, it is already an address. Count how
many of each key there are, run an exclusive prefix sum over the histogram so that entry `v`
becomes "how many elements come before the first `v`", then write each element to
`cursor[key]`. Three passes, no comparison, and the result is stable because the cursors and
the input both advance left to right.

The cost is not O(n). It is O(n + k), and k is the size of the key space rather than the
number of elements. Sorting 10,000 32-bit integers this way means allocating four billion
buckets, almost all of them zero. Radix sort is the fix: sort a few bits at a time, so k stays
small and the number of passes grows instead. Bucket sort is a different trade again - it
treats the key as an estimate rather than an address, and it is linear only in expectation,
and only if the input cooperates.

## Stability is the correctness condition

LSD radix sort runs one stable counting sort per digit, starting from the least significant.
It is correct only because each pass leaves the previous pass's order alone: elements sharing
a high digit keep the relative order the low digits gave them. Fill each bucket back to front
instead - still a perfectly good sort on its own - and the second pass reverses ties the first
pass had just ordered. No later pass can recover them. The array comes out unsorted, not
slower. This is the same property that makes GPU implementations pay for per-thread cursors
instead of letting `atomicAdd` decide who gets which slot.

## How wide should a digit be?

The textbook cost model charges one histogram per pass and concludes that 16-bit digits are
best. Real implementations build a histogram *per tile*, because that is what keeps the
counters in fast memory - a thread block gets 64 KB of shared memory, and a 65,536-entry
histogram needs 256 KB. Charging honestly, `work = passes * (2n + tiles * 2**r)`, the answer
for a million 32-bit keys moves to r = 8 (9,000,448 units of work) with r = 4 close behind
(16,125,056), and r = 16 becomes the *worst* option that still fits nowhere. That window,
4 to 8 bits, is exactly where production radix sorts live.

## Bucket sort bets on the distribution

Same code, same n, same 2,000 buckets. Uniform input: fullest bucket 6, inner-sort cost 551
moves. Clustered input: fullest bucket 132, inner-sort cost 47,038 moves. Nothing raises an
exception - the "linear" sort just quietly stops being linear. The distribution is not a
footnote in the analysis, it is a precondition on the input.

## The scan in the middle

Counting is one thread per element. Scattering is one thread per element. The prefix sum
between them is not: each entry reads what the previous one just wrote. Hillis-Steele buys
depth with work - at n = 1,024 it does 9,217 additions in 10 rounds where the loop does 1,023
in 1,023. Blelloch reaches the same depth class for linear work, 2,046 additions in 20 rounds,
by building a reduction tree on the way up and pushing partial sums back down. Neither is
faster in serial; both do *more* additions than the loop they replace. They win only because
the lanes were idle, which is the whole shape of the work-depth trade.

Blelloch also wants a power-of-two length, which is why sglang's MoE kernel zero-pads the
expert histogram before scanning on the HIP path. An algorithmic constraint grows out into a
kernel parameter.

## A real one: moe_align_block_size

Every decode step of every MoE model runs this. The router says which experts each token goes
to; the GEMM requires each expert's tokens to be contiguous and aligned to a fixed block size.
Expert ids are small integers, so this is a counting sort, with two additions: each bucket is
rounded up to `block_size` *before* the scan, and the holes are pre-filled with a sentinel one
past the last real token so the GEMM can read a full block and mask the padding out.

Then `expert_ids` - which expert owns block `b` - is recovered by binary searching the same
prefix array. One scan does three jobs: the bucket starts for the scatter, the moving cursors
during it, and the search key afterwards.

`sgl-kernel` ships two kernels for this one function. The `atomicAdd` version is faster and
non-deterministic; the per-thread-histogram version is reproducible. Both are correct counting
sorts and both feed the same GEMM, because rows of a GEMM do not care what order they arrive
in. The switch is hard-coded: `topk_ids.numel() < 1024 && num_experts <= 64`. At 4,096 tokens
with top-8 of 256 experts and block 64, the padding is 7,936 of 40,704 slots - 19.5%, the
price of a fixed block size, and what the counting sort is arranging the data to make payable.

## LeetCode 75 - Sort Colors

A histogram of three counters and a rewrite, or the one-pass three-pointer version the
follow-up is fishing for. The asymmetry is the whole difficulty: a value swapped down from the
high end has never been examined, so `mid` must not advance after that swap.

## LeetCode 164 - Maximum Gap

Asking for O(n) is the same as forbidding a sort. Drop n numbers into n-1 buckets and
pigeonhole guarantees at least one bucket is empty, so the largest gap is at least one bucket
wide and cannot lie inside a bucket. Keep one min and one max per bucket, never compute the
order within a bucket, sweep once. The linear answer is not a cleverer sort - it comes from
noticing which part of the sort can be skipped.

## Complexity
| Operation | Time | Space |
|---|---|---|
| Comparison lower bound | Omega(n log n) | - |
| Counting sort | O(n + k) | O(n + k) |
| LSD radix sort | O(ceil(bits / r) * (n + 2**r)) | O(n + 2**r) |
| Bucket sort, uniform input | O(n + m) expected | O(n + m) |
| Bucket sort, adversarial input | O(n^2) | O(n + m) |
| Sequential scan | O(n) work, O(n) depth | O(1) |
| Hillis-Steele scan | O(n log n) work, O(log n) depth | O(n) |
| Blelloch scan | O(n) work, O(log n) depth | O(n) |
| moe_align_block_size | O(n + E) + O(blocks * log E) | O(n + E * block) |
| Maximum Gap (LC 164) | O(n) | O(n) |

## References
- [Counting sort](https://en.wikipedia.org/wiki/Counting_sort)
- [Radix sort](https://en.wikipedia.org/wiki/Radix_sort)
- [Bucket sort](https://en.wikipedia.org/wiki/Bucket_sort)
- [Comparison sort - lower bound](https://en.wikipedia.org/wiki/Comparison_sort#Number_of_comparisons_required_to_sort_a_list)
- [Prefix sum](https://en.wikipedia.org/wiki/Prefix_sum)
- [Blelloch, Prefix Sums and Their Applications](https://www.cs.cmu.edu/~guyb/papers/Ble93.pdf)
- [SGLang](https://github.com/sgl-project/sglang) - `sgl-kernel/csrc/moe/moe_align_kernel.cu`
- [SGLang](https://github.com/sgl-project/sglang) - `srt/layers/moe/moe_runner/triton_utils/moe_align_block_size.py`
