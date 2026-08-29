# Selection and top-k - you never had to sort

More details in:
https://medium.com/100-days-of-python

Interactive walkthrough: [demo.html](demo.html) - a single self-contained page
(no build step, works offline, 中文 / English toggle) that animates a three-way quickselect
throwing half the array away each round, a radix select narrowing a row with nothing but
counters, the filtered single-block kernel squeezing 14 scores into 4 slots of shared memory,
four CTAs meeting at a grid barrier with a triple-buffered histogram, the real flashinfer
deadlock that happens when the first CTA out resets the barrier counter, the same tied
input producing four different "correct" answers, a secant search closing in on a threshold
without ever comparing two elements, and four CTAs agreeing on a count through each other's
shared memory instead of through global memory.

![img](https://github.com/KingRei/100DaysPython/blob/master/day%2021%20-%20selection%20and%20top-k/imgs/day21_1.png?raw=true)

Sorting works out the order of every pair of elements. Selection answers one question: who
are the k largest? Those are not the same job, and the gap is measurable - on 200,000 floats
with k = 8, a full heap sort costs 6,439,409 comparisons and then throws 199,992 of its
results away; quickselect costs 244,798.

The reason is that quickselect recurses into one side only, so the work is
n + n/2 + n/4 + ... = 2n. A size-k min-heap gets there differently, in 200,000 comparisons
and 8 floats of memory, and it is the only variant that never needs the whole input at once -
which is what makes it the right answer for a stream. Median-of-medians guarantees O(n) in
the worst case and costs 653,431 comparisons doing it, which is why almost nothing ships it.

The second half of this day is the same problem on a GPU, where the algorithm stops being the
hard part. Every sampling step of an LLM runs a top-k over the vocabulary, so these kernels
run more often than almost anything else in the stack - and the interesting failures are not
about selection at all. They are about who resets a counter and who wins a tie.

## The order-preserving key

Flip the sign bit of a positive float, invert every bit of a negative one, and unsigned
integer order matches float order exactly. The map is bijective - nothing is lost - which is
what lets a radix pass stand in for a comparison. Note that `-0.0` and `+0.0` get different
keys, and that is correct: they are different bit patterns being ordered, not equal numbers.

## Radix select

Bin the keys by a slice of bits, count how many land in each bin, accumulate from the top
until the running total crosses k, and recurse into the bin that straddles it. A round moves
256 counters, not n elements. On 4,000 attention-like scores with k = 32 the candidate list
goes 4000 → 611 → 2 → 1 in four rounds, and no two elements are ever compared.

## One block per row - the filtered kernel

If k is small, do not launch a grid. Pass one histograms a coarse key; pass two projects the
straddling bin's two edges back into floats as `v_lo` and `v_hi`, so classifying each element
is two float compares against loop-invariant registers. On a row of 20,000 values with
k = 64 that leaves 28 guaranteed winners and 98 candidates - 0.49% of the row - and everything
after the first pass stays on chip.

The coarse key has to be taken from a *narrowed* copy of the value, not from the raw float32.
The top byte of an fp32 is the sign plus seven exponent bits, so a row of attention logits
piles into a handful of bins. Measured on the same 20,000 values: an fp16 coarse key leaves
98 candidates, a bf16 one leaves 3,197.

That also explains the refinement round counts. For a 16-bit dtype the coarse bin *is* the
top byte of the exact key, so the coarse pass has already decided 8 of 16 bits and one round
at shift 0 finishes. For fp32 the coarse key lives in a different key space, decides zero
bits of the real key, and refinement starts from the top: shifts 24, 16, 8, 0.

## Many CTAs per row - the grid barrier

Split a row across blocks and you need something CUDA does not give you: a barrier across the
grid. Build it from one counter that is *never reset*; each CTA keeps a private phase and
waits for `counter >= (phase + 1) * num_ctas`. The comparison is `>=` and not `==` because a
descheduled CTA can wake up after the counter has already run past its target.

The histogram is triple buffered: round r accumulates into `hist[r % 3]` while CTA 0 clears
`hist[(r + 1) % 3]`, so a single barrier proves both "all the adds landed" and "the next
buffer is clean". With one buffer it deadlocks; with two it races.

## The failure case - flashinfer issue #3610

Someone has to zero the counter for the next launch. The obvious choice - whoever finishes
first - hangs the kernel, because a peer still spinning reads 0 < target and never leaves. In
the simulation, `reset_policy='first'` deadlocks on every run and `'last'` never does. The fix
is an exit barrier that doubles as leader election: exactly one CTA sees its own increment
land on the target, and that one is provably last out.

## Ties

"The top k" is not unique when values tie, and the two knobs people confuse are orthogonal.
`deterministic` pins the output *ordering* for one launch shape; it does not pin *membership*,
because a grid-stride loop bakes `num_ctas` into which CTA sees which tie. Measured on a row
with twelve tied values: the racing default gives 6 different answers over 8 runs; the
deterministic path is stable per launch but gives three *different* stable answers for 2, 4
and 8 CTAs. Only a row-global rule - `tie_break` = smallest index, or largest - gives one
answer across every launch shape, and that is why it costs more.

## Guess, verify, refine - solving for the threshold

Quickselect and radix select both shrink the data. flashinfer's GVR kernel
(`flashinfer/topk_varlen/kernels/gvr_topk_decode.py`, written for Blackwell) goes the other
way: it guesses the cut-off value, counts how many elements clear it, and solves for a better
guess. Four phases - P1 produces an initial threshold, P2 refines it with a secant search that
only ever counts, P3 collects the survivors into shared memory, P4 runs one exact histogram
over that much smaller set.

P1 is the part worth stealing. It does not scan the row at all. It reads the index list this
same kernel emitted on the *previous* decode step, shifted by one token, and takes the min,
max and mean of today's logits at those positions - k loads instead of n. The bet is that the
top k of token t sits roughly where the top k of token t-1 sat, and inside decode that bet
almost always pays. The two counts that go with the bracket are not measured at all: P1 fills
in `cnt_lo = k + k/4` and `cnt_hi = 1` and lets the first real count correct them.

P2 then assumes the count falls linearly with the threshold and solves for the next guess
rather than halving the bracket. Anything landing inside the accept window `[k, kC]` is good
enough, because P4 does the exact selection anyway - so a bad threshold costs time, never
correctness.

That last sentence is what makes the failure mode interesting. On a 20,000-element row with
k = 512, the cold start converges in 3 refinements, while the *better informed* warm start
runs 15 and gives up. The cause is one constant: `kFTarget` is 384, below the accept window's
own floor of 512, so the secant converges on a count the loop is required to reject; and
because P1's fabricated `cnt_lo = 640` happens to sit close to the true count of 629, the
secant fraction pins at its 0.95 ceiling and each pass shaves 5% off the bracket instead of
jumping across it. Upstream names this in its own tuning table - `kFTarget = kK` "to avoid
upper-clamp saturation on tight-sigma layers (1.5-2.2x fewer P2 iters on swe-bench)". Over 200
random rows here, moving that one number takes the mean from 15.00 refinements (200/200 gave
up) to 8.38 (18/200), a factor of 1.8 - inside the range upstream measured. Every row returns
the exact top-k either way.

## Agreeing inside a cluster

The grid barrier above lives in global memory because two arbitrary blocks are not co-resident
and cannot wait for each other any other way. A thread block cluster is co-resident by
definition, so the CTAs sharing a row can read each other's shared memory directly -
`mapa.shared::cluster` maps a local address into a peer's shared memory and
`ld.shared::cluster` reads it. The per-iteration count aggregation never leaves the GPC: zero
global atomics, zero round trips. GVR's P2 aggregates a count on every iteration, so this
saves one round trip per refinement, which is why "fewer iterations" and "cheaper iterations"
are the same optimisation seen from two sides.

The price is portability. Clusters are a CUDA feature; the equivalent path in SGLang's topk v2
is compiled out on CDNA, which has no distributed shared memory. There is also a load-balanced
variant that classifies rows by sequence length before the main launch, hands a whole cluster
to each long row and a single CTA to each short one - break-even around a 64K scan length. The
counter saying how many rows are long is written by the device and read by the launch, which
is exactly the dependency a CUDA graph cannot capture, so that variant runs uncaptured.

## LeetCode 215 - Kth Largest Element in an Array

Quickselect with a random pivot and a three-way partition. The follow-up is the point: on
50,000 identical values a two-way partition peels off one element per round and turns into
O(n^2), while the three-way split retires every tie in a single pass - 0.006s here. It is the
same reason the GPU kernels above have to think about ties at all.

## Complexity
| Operation | Time | Space |
|---|---|---|
| Full sort, then take k | O(n log n) | O(1) - O(n) |
| Quickselect | O(n) expected, O(n^2) worst | O(1) |
| Median-of-medians | O(n) worst case | O(n) |
| Size-k min-heap (streaming) | O(n log k) | O(k) |
| Reservoir sampling | O(n) | O(k) |
| Radix select | O(n * rounds) | O(bins) |
| Filtered top-k (1 block) | O(n) over global memory, twice | O(candidates) on chip |
| Multi-CTA radix select | O(n / p) per round + barrier | O(bins) shared |
| GVR (guess-verify-refine) | O(n) per refinement, ~1-5 refinements | O(k) seed + O(kC) candidates |
| GVR P1 (warm start) | O(k) - reads the previous step's indices | O(k) |

## References
- [Selection algorithm](https://en.wikipedia.org/wiki/Selection_algorithm)
- [Quickselect](https://en.wikipedia.org/wiki/Quickselect)
- [Median of medians](https://en.wikipedia.org/wiki/Median_of_medians)
- [Reservoir sampling](https://en.wikipedia.org/wiki/Reservoir_sampling)
- [Radix sort](https://en.wikipedia.org/wiki/Radix_sort)
- [flashinfer](https://github.com/flashinfer-ai/flashinfer) - `include/flashinfer/topk.cuh`
- [SGLang](https://github.com/sgl-project/sglang) - `deepseek_v4/topk_impl.cuh`
- [flashinfer](https://github.com/flashinfer-ai/flashinfer) - `flashinfer/topk_varlen/kernels/gvr_topk_decode.py`
- [Secant method](https://en.wikipedia.org/wiki/Secant_method)
- [Thread block clusters](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#thread-block-clusters)
