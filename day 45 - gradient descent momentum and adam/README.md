# The gradient descent family: SGD, momentum and Adam, and the exact point where a learning rate stops working

More details in:
https://medium.com/100-days-of-python

Interactive walkthrough: [demo.html](demo.html) - a single self-contained page
(no build step, works offline, 中文 / English toggle) that walks gradient descent down a
ravine and past its exact learning-rate ceiling, races plain GD against momentum, steps
Adam's two running averages one at a time, shows what the missing bias correction and the
misplaced epsilon do, splits weight decay into Adam + L2 against AdamW, watches a
mini-batch run settle on its noise floor, and solves LeetCode 462 and LeetCode 852.

![img](https://github.com/KingRei/100DaysPython/blob/master/day%2045%20-%20gradient%20descent%20momentum%20and%20adam/imgs/day45_1.png?raw=true)

On a quadratic bowl, the learning rate has an exact ceiling. For `f(w) = 0.5(w0² + 20w1²)` it is `2 / 20 = 0.1`: at `lr = 0.0999` gradient descent converges, at `lr = 0.1001` it runs off to infinity. Nothing raises an exception on the way. After 400 steps at `lr = 0.11` my parameter vector was `[5.7e-21, 4.7e+31]` - one coordinate perfectly solved, the other somewhere past the end of the universe.

Day 44 ended with a training loop whose last line was `w -= lr * w.grad`. Everything today replaces that one line and nothing else:

```python
g = grad(w)
w = opt.step(w, g)
```

Seven update rules go through that interface: SGD, SGD with momentum, Nesterov, AdaGrad, RMSProp, Adam and AdamW. The interesting part is not the formulas, it is which of their details are load-bearing. Four that are, and that never report an error when you get them wrong:

- Adam without bias correction takes steps that are **too large**, not too small as the folklore says - 3.2x too large on the first step, 6.6x at step 12, still 1.26x at step 1,000.
- `sqrt(v) + eps` and `sqrt(v + eps)` differ by 99x on a coordinate whose `v` is `1e-12`.
- L2 regularisation inside Adam decays the weights with the largest gradients **10x less** than the others, which is backwards.
- A batch of one needs a learning rate 8.5x smaller than full batch on the same data, because it has to respect the curvature of its own row rather than the average.

Run `python optimizers.py` to print every number here (timings aside, they are all exact).

## The ravine, and where the ceiling comes from

`f(w) = 0.5(w0² + 20w1²)` is a bowl that is 20 times more curved across than along - a ravine. That ratio, largest curvature over smallest, is the **condition number**, and it is the single number that decides how hard a bowl is to descend.

Plain gradient descent on a quadratic does something very simple: it multiplies each coordinate by `(1 - lr × curvature)` every step. That is the whole algorithm, and it hands you the convergence condition for free - every multiplier has to be smaller than 1 in absolute value:

![img](https://github.com/KingRei/100DaysPython/blob/master/day%2045%20-%20gradient%20descent%20momentum%20and%20adam/imgs/day45_3.png?raw=true)

```text
lr = 0.05    largest multiplier 0.95000  converges
lr = 0.0999  largest multiplier 0.99800  converges
lr = 0.1001  largest multiplier 1.00200  diverges
lr = 0.11    largest multiplier 1.20000  diverges
```

The sharpest direction sets the limit, so `lr < 2 / largest curvature`. Above it the flat direction still converges beautifully while the sharp one explodes - which is why a diverged run often has some parameters that look perfectly reasonable.

Below the ceiling you get the other half of the problem. At `lr = 0.09` the sharp direction's multiplier is `-0.80`: it overshoots and flips sign every step, which is the zigzag in the first figure. Meanwhile the flat direction shrinks by `0.91` per step, which is a crawl. The step size you are allowed is set by the steepest direction; the progress you make is set by the flattest one. That gap is the condition number, and it is why this bowl takes 147 steps instead of a handful.

## Momentum turns kappa into sqrt(kappa)

Momentum keeps a running sum of past gradients and steps along that instead:

```python
self.v = self.momentum * self.v + g
return w - self.lr * self.v
```

Along the zigzag direction the alternating gradients cancel in the sum; along the valley floor they agree and accumulate. The theory says the error multiplier per step improves from `(kappa - 1)/(kappa + 1)` to `(sqrt(kappa) - 1)/(sqrt(kappa) + 1)`, which for `kappa = 20` is 0.905 against 0.635. Measured against theory:

![img](https://github.com/KingRei/100DaysPython/blob/master/day%2045%20-%20gradient%20descent%20momentum%20and%20adam/imgs/day45_2.png?raw=true)

```text
plain GD   lr 0.0952  rate 0.9048  theory 138.0 steps  measured 139
momentum   lr 0.1336  beta 0.4026  rate 0.6345  theory  30.4 steps  measured  39
```

The plain-GD prediction lands within one step of the measurement, which is a good sign that the model of what is happening is right. Momentum's measured 39 against a predicted 30 is the asymptotic rate ignoring the transient, and it is still 3.6x fewer steps.

One thing worth measuring rather than assuming: `beta = 0.9` is not a constant of nature. The best beta for this bowl is `0.4026`, and at a fixed `lr = 0.04`:

```text
beta 0       -> 338 steps
beta 0.4026  -> 190 steps
beta 0.9     -> 245 steps
```

Copying 0.9 from a blog post made things *worse* than the tuned value, and on some learning rates worse than no momentum at all. Beta is under-damping or over-damping an oscillator; it has a right answer that depends on the problem.

## On a real problem, with every learning rate tuned

A least-squares fit on 400 rows and 3 features, with one feature scaled up 8x so the condition number is 77. Comparing optimisers at one shared learning rate would be meaningless - AdaGrad, RMSProp and Adam divide the gradient by its own size, so their `lr` is a different quantity - so each one got its own sweep:

![img](https://github.com/KingRei/100DaysPython/blob/master/day%2045%20-%20gradient%20descent%20momentum%20and%20adam/imgs/day45_4.png?raw=true)

```text
SGD                  best lr  0.0126    400 steps
SGD + momentum 0.9   best lr  0.0032    181 steps
SGD + Nesterov 0.9   best lr  0.0079    163 steps
AdaGrad              best lr  1.5849     10 steps
RMSProp              best lr  0.1585     26 steps
Adam                 best lr  0.6310    153 steps
```

AdaGrad winning by 40x deserves an explanation rather than a celebration. AdaGrad divides by the square root of *every squared gradient ever seen*, so its effective step size only ever shrinks. On a small, fixed, noiseless quadratic that is close to ideal - it is approximating the curvature and then annealing. The same property is what makes it stall on a long training run: the denominator keeps growing, the steps keep shrinking, and eventually nothing moves. RMSProp is AdaGrad with a decay factor on that sum so the step size can recover, and Adam is RMSProp plus momentum on the gradient as well.

## Adam, and the two places it is usually written wrong

Adam keeps two running averages: `m` of the gradient and `v` of the gradient squared. The step is `lr * m / sqrt(v)`, so each coordinate moves by roughly `lr` regardless of its gradient's scale.

Both averages start at zero, which biases them low early on. The standard fix divides `m` by `1 - b1^t` and `v` by `1 - b2^t`. The folklore is that skipping this makes the first steps too small. It does not - it makes them too **large**, because `v` sits under a square root and the two biases do not cancel:

![img](https://github.com/KingRei/100DaysPython/blob/master/day%2045%20-%20gradient%20descent%20momentum%20and%20adam/imgs/day45_5.png?raw=true)

```text
uncorrected / corrected = (1 - 0.9^t) / sqrt(1 - 0.999^t)
  step     1: 3.162x        step   100: 3.241x
  step    12: 6.569x        step  1000: 1.258x
```

Measured on the real problem at `lr = 0.05`, the first three steps are `0.087, 0.086, 0.086` with correction and `0.274, 0.363, 0.404` without. The uncorrected version spends its first few hundred steps taking a learning rate several times larger than the one you asked for, which is exactly when a model is most fragile.

The second detail is where the epsilon goes. `sqrt(v) + eps` and `sqrt(v + eps)` both prevent a division by zero, but `eps = 1e-8` has `sqrt(eps) = 1e-4`:

```text
v = 1e-12   sqrt(v) + eps = 1.010e-06   sqrt(v + eps) = 1.000e-04   ratio    99.0x
v = 1e-16   sqrt(v) + eps = 2.000e-08   sqrt(v + eps) = 1.000e-04   ratio  5000.0x
```

On any coordinate whose gradients are small - a rarely-used embedding row, a nearly-dead unit - the second form quietly divides the step by a hundred or more. PyTorch uses `sqrt(v) + eps`.

## Adam + L2 is not AdamW

L2 regularisation is usually implemented by adding `wd * w` to the gradient. Inside Adam, that sum then gets divided by `sqrt(v)` - so the decay applied to each weight is scaled by `1 / sqrt(v)`, and the weights with the largest gradients get the least decay:

![img](https://github.com/KingRei/100DaysPython/blob/master/day%2045%20-%20gradient%20descent%20momentum%20and%20adam/imgs/day45_6.png?raw=true)

After 400 Adam steps on the least-squares problem, `sqrt(v)` per weight is `[0.41, 0.15, 4.06]` (measured on a run without the penalty; adding it moves `sqrt(v)` by at most 2.5%). Normalising so the average decay strength is 1, Adam + L2 applies `[0.77, 2.15, 0.08]` while AdamW applies `[1, 1, 1]`. The weight on the 8x-scaled feature - the one with by far the largest gradients - is decayed 10x less than weight 0 and 28x less than weight 1. AdamW fixes it by not routing the decay through the gradient at all:

```python
w = w - self.lr * mh / denom
if self.weight_decay and self.decoupled:
    w = w - self.lr * self.weight_decay * w    # decay applied straight to w
```

Those two lines are the entire difference between `torch.optim.Adam(weight_decay=...)` and `torch.optim.AdamW`, and they are why AdamW is the default for transformer training.

## Mini-batches move the ceiling

Everything above used the full dataset per step. Real training uses mini-batches, and the first thing that changes is the learning-rate ceiling. The ceiling comes from the sharpest curvature a step has to survive, and a batch of one sees its own row's curvature, not the average:

```text
ceiling from the average over all 400 rows   0.0140
ceiling from the sharpest single row         0.00165   (8.5x smaller)

lr 0.01, batch 1     after 30 epochs: blew up
lr 0.01, batch 8     after 30 epochs: 4.6e-04
lr 0.01, full batch  after 30 epochs: 1.5e+00
```

Below that ceiling, small batches are enormously more efficient per pass over the data - and then they stop:

![img](https://github.com/KingRei/100DaysPython/blob/master/day%2045%20-%20gradient%20descent%20momentum%20and%20adam/imgs/day45_7.png?raw=true)

```text
after  10 epochs   batch 8 8.813e-02   full batch 4.626e+00
after  30 epochs   batch 8 2.545e-03   full batch 3.911e+00
after 100 epochs   batch 8 5.396e-03   full batch 2.210e+00
after 300 epochs   batch 8 2.484e-03   full batch 4.352e-01
```

Batch 8 gets three orders of magnitude further in 30 epochs, then sits on a noise floor around `2.5e-3` for the next 270. Each mini-batch gradient points in a slightly wrong direction, so with a fixed step size the parameters circle the optimum at a radius set by the noise and the learning rate. Shrinking the learning rate over time is what shrinks that radius - which is what a learning-rate schedule is for.

Shuffling is worth measuring rather than taking on faith. On rows that are already in random order, reshuffling every epoch made no difference I could detect (1.6e-3 against 1.5e-3, median of five seeds). On the *same data sorted by target* it was 12x better. Sorted data makes every batch unrepresentative in the same way every epoch, so the identical biased sequence of steps repeats - that is the failure shuffling actually protects against.

## LeetCode 462 - Minimum Moves to Equal Array Elements II

**The task.** Given an integer array, find the smallest number of moves that makes every element equal, where one move adds 1 to a single element or subtracts 1 from it.

**Input / output.** `nums`, a list of integers. Return one integer: the total number of moves.

**Example.** `[1, 2, 3]` needs 2 moves: raise the 1 and lower the 3 until both reach 2. `[1, 10, 2, 9]` needs 16: pick 9 (or any value between 2 and 9), giving `8 + 1 + 7 + 0`.

**Constraints.** Up to 100,000 elements, each between -10⁹ and 10⁹. The large range rules out counting into an array indexed by value, and the size means `O(n log n)` sorting is fine. The real constraint is hidden in the word "moves": the cost is the sum of absolute differences, not squared differences, and those two have different minimisers.

Choosing a target `c` costs `sum|x - c|`. The derivative of `|x - c|` with respect to `c` is the *sign* of `c - x`, so the total derivative is "how many points are below me, minus how many are above me". That is zero exactly at the median:

```python
class Solution:
    def minMoves2(self, nums: list[int]) -> int:
        nums = sorted(nums)
        mid = nums[len(nums) // 2]
        return sum(abs(x - mid) for x in nums)
```

The trap is reaching for the mean, which minimises *squared* error - the thing Day 44's `2/n Xᵀ(Xw - y)` was built on. On `[1, 2, 3]` and `[1, 10, 2, 9]` the mean and the median give the same answer, so a quick test passes. On `[1, 2, 3, 100]` the median gives 100 and the mean gives 146: a plausible, larger, wrong number.

Running today's gradient descent on `sum|x - c|` is a nice check. Starting from `c = 0` with a `1/sqrt(k)` schedule it lands at `2.354`, which costs exactly 100 - because with an even count every `c` between the two middle values is optimal, and the minimiser is an interval rather than a point. Note the schedule is doing real work: with a fixed step size the sign gradient never shrinks near the optimum, so the iterate would oscillate forever. That is the same noise-floor mechanism as the mini-batch section, in its purest form.

[leetcode.com/problems/minimum-moves-to-equal-array-elements-ii](https://leetcode.com/problems/minimum-moves-to-equal-array-elements-ii/)

## LeetCode 852 - Peak Index in a Mountain Array

**The task.** An array is a mountain: it strictly increases up to one index and strictly decreases after it. Find that index.

**Input / output.** `arr`, a list of integers guaranteed to form a mountain. Return the integer index of the largest element.

**Example.** `[0, 10, 5, 2]` returns 1: `arr[1] = 10` is larger than both neighbours, and everything after it decreases.

**Constraints.** Between 3 and 100,000 elements, and the problem explicitly asks for `O(log n)`. That requirement is the whole problem - a linear scan finds the answer but is rejected.

The slope of a discrete array between `mid` and `mid + 1` is just the comparison `arr[mid] < arr[mid + 1]`. On a mountain that comparison is true for every index before the peak and false for every index at or after it, so it is a sorted boolean array and Day 23's binary search applies:

```python
class Solution:
    def peakIndexInMountainArray(self, arr: list[int]) -> int:
        lo, hi = 0, len(arr) - 1
        while lo < hi:
            mid = (lo + hi) // 2
            if arr[mid] < arr[mid + 1]:    # still going up: the peak is to the right
                lo = mid + 1
            else:                          # already coming down: mid could be the peak
                hi = mid
        return lo
```

On a 4,001-element mountain this looks at 24 elements. It is gradient ascent in the sense that the only information used is the sign of the slope - but instead of taking a step proportional to it, each probe deletes half of the remaining space. The continuous twin is ternary search: evaluate a unimodal `f` at two interior points, throw away the third that cannot contain the minimum, and converge without ever computing a derivative. On `(x - 1.234567)² + 3` over `[-10, 10]` it recovers `1.234567`.

[leetcode.com/problems/peak-index-in-a-mountain-array](https://leetcode.com/problems/peak-index-in-a-mountain-array/)

The reason gradient descent exists despite being slower per unit of information is dimensionality. Binary and ternary search own one-dimensional unimodal problems; neither survives contact with a million parameters, where the gradient gives a direction in all million at once.

## Complexity

| Operation | Time | Space |
|---|---|---|
| One SGD step on d parameters | O(d) | O(1) |
| One momentum / RMSProp / AdaGrad step | O(d) | O(d) |
| One Adam / AdamW step | O(d) | O(2d) for m and v |
| GD steps to a given accuracy, condition number k | proportional to k | - |
| Momentum steps, best beta | proportional to sqrt(k) | - |
| LeetCode 462 | O(n log n) sorting, O(n) with quickselect | O(1) extra |
| LeetCode 852 | O(log n) | O(1) |
| Ternary search to a tolerance | O(log(range / tol)) | O(1) |

## References

- [Gradient descent - Wikipedia](https://en.wikipedia.org/wiki/Gradient_descent) and [condition number](https://en.wikipedia.org/wiki/Condition_number)
- Boris Polyak, *Some methods of speeding up the convergence of iteration methods*, 1964 - the heavy-ball method
- Yurii Nesterov, *A method of solving a convex programming problem with convergence rate O(1/k²)*, 1983
- Duchi, Hazan and Singer, *Adaptive Subgradient Methods* (AdaGrad), JMLR 2011
- Kingma and Ba, [*Adam: A Method for Stochastic Optimization*](https://arxiv.org/abs/1412.6980), ICLR 2015
- Loshchilov and Hutter, [*Decoupled Weight Decay Regularization*](https://arxiv.org/abs/1711.05101) (AdamW), ICLR 2019
- [PyTorch docs: torch.optim](https://pytorch.org/docs/stable/optim.html) - the reference implementations of every rule here
