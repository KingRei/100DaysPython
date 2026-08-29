# Selection and top-k - you never had to sort

More details in:
https://medium.com/100-days-of-python

Interactive walkthrough: [demo.html](demo.html) - a single self-contained page
(no build step, works offline, 中文 / English toggle) that animates a three-way quickselect
throwing half the array away each round, a radix select narrowing a row with nothing but
counters, the filtered single-block kernel squeezing 14 scores into 4 slots of shared memory,
four CTAs meeting at a grid barrier with a triple-buffered histogram, the real flashinfer
deadlock that happens when the first CTA out resets the barrier counter, and the same tied
input producing four different "correct" answers.

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

## References
- [Selection algorithm](https://en.wikipedia.org/wiki/Selection_algorithm)
- [Quickselect](https://en.wikipedia.org/wiki/Quickselect)
- [Median of medians](https://en.wikipedia.org/wiki/Median_of_medians)
- [Reservoir sampling](https://en.wikipedia.org/wiki/Reservoir_sampling)
- [Radix sort](https://en.wikipedia.org/wiki/Radix_sort)
- [flashinfer](https://github.com/flashinfer-ai/flashinfer) - `include/flashinfer/topk.cuh`
- [SGLang](https://github.com/sgl-project/sglang) - `deepseek_v4/topk_impl.cuh`
