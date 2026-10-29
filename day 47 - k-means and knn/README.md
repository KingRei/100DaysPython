# k-means and k-nearest neighbours: clustering and classifying with nothing but a distance

More details in:
https://medium.com/100-days-of-python

Interactive walkthrough: [demo.html](demo.html) - a single self-contained page
(no build step, works offline, 中文 / English toggle) that runs Lloyd's algorithm one
half-step at a time from a good start and from an unlucky one, watches the inertia fall
for every extra k on real groups and on pure noise, empties a centroid and lets the
resulting nan swallow the whole data set, clusters the same requests under three
different distance functions, takes k-nearest neighbours in one at a time at k = 1, 5
and 13, shows distance concentration from 2 to 1024 dimensions, and solves LeetCode 973
and LeetCode 658 - including the one character that breaks the tie rule.

![img](https://github.com/KingRei/100DaysPython/blob/master/day%2047%20-%20k-means%20and%20knn/imgs/day47_1.png?raw=true)

I ran k-means on 180 points of uniform random noise and asked for three clusters. It gave me three: sizes 54, 75 and 51, each with a sensible-looking centroid, converged in a handful of rounds, with an objective value of 449.0 that looks exactly like every other objective value I have ever seen. There is no structure in that data at all. Nothing in the output said so.

That is the shape of this whole day. k-means and k-nearest neighbours are the two algorithms almost everyone meets first, and between them they contain about six lines of logic:

```
k-means   : repeat { assign each point to the nearest centroid
                     move each centroid to the mean of its members }
k-nearest : for a query, take the k nearest training points and let them vote
```

Neither of those rules is where the bugs are. Both algorithms are **a distance function plus a rule**, the rule is almost impossible to get wrong, and the distance function never raises. It just quietly answers a different question than the one you asked.

## Lloyd's algorithm, and the two reasons it stops

Every run below uses the same distance matrix, built the way Day 44 built everything: `|x - c|² = |x|² - 2 x·c + |c|²` turns a double loop into one matrix product.

```python
def sq_dists(X, C):
    x2 = np.einsum('ij,ij->i', X, X)[:, None]
    c2 = np.einsum('ij,ij->i', C, C)[None, :]
    return np.maximum(x2 - 2.0 * X @ C.T + c2, 0.0)
```

The `np.maximum` is not a tidy-up. The expansion subtracts two large numbers, so for the point `[1000000.5, 48000000.25]` it evaluates the distance from that point to itself as **-1.0**, and the square root of that is `nan`. Clamping at zero is part of the algorithm.

The loop itself is two lines:

```python
labels = np.argmin(sq_dists(X, C), axis=1)                      # assign
C = np.array([X[labels == j].mean(axis=0) for j in range(k)])   # update
```

On 180 points in three blobs this converges in six rounds, and the objective - the sum of squared distances to the assigned centroid, usually called the inertia - goes 1639.0, 1158.5, 1029.8, 447.9, 240.0, 239.0. It never goes up, and that is not an accident. **The two halves lower it for two completely different reasons.** Assignment lowers it because every point picks its own cheapest centroid, so nobody can end up worse off. The update lowers it because the mean is the minimiser of squared distance. A finite number of possible partitions plus an objective that only falls means the loop has to stop.

Hold on to the second reason. A few sections from now it gets taken away.

## The objective can never tell you k

Inertia falls every time you add a centroid, and at `k = n` it is exactly zero. So "minimise the objective" always answers `k = n`, and it has no power whatsoever to tell you how many groups are in the data.

![img](https://github.com/KingRei/100DaysPython/blob/master/day%2047%20-%20k-means%20and%20knn/imgs/day47_2.png?raw=true)

The elbow method is the usual answer, and on data that really does have three groups it works: the step from k=2 to k=3 removes 78.7% of the inertia and the next step only 9.3%, a factor of 8.4. But look at the second curve. Those are the 180 uniform random points from the opening paragraph. 32.7% is followed by 40.0% - the drop gets *bigger* - and there is no elbow anywhere, because there is nothing to find.

The important part is that the elbow is something you see, not something the algorithm computes. The usual responses both decline to ask how small the inertia is. The gap statistic literally compares the inertia curve against the same algorithm run on structureless data. The silhouette score takes a different route to the same scepticism: it scores each point by how much nearer it is to its own cluster than to the next nearest one, which has nowhere to go but zero when the clusters are arbitrary.

## The answer depends on the random start

Lloyd converges to *a* local minimum. On three clearly separated blobs - the easiest data set anyone could construct - 23 random starts out of 200 end more than 1% above the best found, and the worst gives an inertia of 1742.1 against 367.0, with cluster sizes `[21, 120, 39]` instead of `[60, 60, 60]`. Two centroids landed in the same blob, so that blob was split in half and the other two were merged into one.

Both runs are "the output of k-means". Neither prints a warning. The only way to tell them apart is to compare the inertia, which is exactly why the standard practice is to restart several times and keep the best, and why scikit-learn has an `n_init` parameter at all. Its `'auto'` default is itself a statement about what the next paragraph buys: one start when you initialise with k-means++, ten when you initialise randomly.

k-means++ is the cheap improvement: pick the first centre uniformly, then pick each further centre with probability proportional to its squared distance to the nearest centre so far. That is Day 42's weighted random pick, and it takes the failure rate from 23 in 200 down to 1 in 200. It is still random; it has just made the bad draw rare.

## A centroid can lose all of its points

```python
C[j] = X[labels == j].mean(axis=0)   # empty -> nan
```

The mean of an empty set is `nan`, and numpy says so with one `RuntimeWarning` on stderr. A warning is easy to miss, but that is not the interesting part.

The interesting part is `np.argmin([0.246, nan, 50.08])`, which returns **1**. It does not skip the `nan`; it returns its index. So on the next round every distance to the dead centroid is `nan`, `nan` wins every comparison, and the assignment becomes `[1, 1, 1, 1, 1]`: the entire data set joins the centroid that has no position. One more update and the model is `[[nan, nan], [2.36, 2.02], [nan, nan]]` - one cluster holding everything, plus two that do not exist.

What comes out is a perfectly shaped `(k, 2)` array, and `len(C)` is still `k`, so any sanity check that counts clusters passes. Swapping in `np.nanargmin` only hides this one level deeper. The real fix is what scikit-learn does: move the orphaned centroid to the point that is furthest from any centroid.

## The units decide the clustering

Here is a data set shaped like something you might actually have: 360 requests, two columns, `prompt_tokens` between 200 and 4000 and a cache hit rate between 0 and 1. The three real populations differ *only* in hit rate - 0.15, 0.50 and 0.85 - while the token count is drawn from the same wide range for all three.

![img](https://github.com/KingRei/100DaysPython/blob/master/day%2047%20-%20k-means%20and%20knn/imgs/day47_3.png?raw=true)

| version | purity | what the three centroids learned |
|---|---|---|
| raw units | 0.358 | hit rates 0.487, 0.492, 0.509 |
| standardised | 0.669 | hit rates 0.302, 0.311, 0.828 |
| hit rate only | 1.000 | hit rates 0.143, 0.492, 0.854 |

Purity is the fraction of points whose cluster's majority label is their own; guessing scores 0.333. The raw-units run scored 0.358, which is guessing. The token axis spans about 4000 times more than the hit-rate axis, so squared it contributes about sixteen million times more distance, and k-means cut three vertical token bands without ever looking at the column the answer was in. The inertia it reported, 46140168, is a perfectly good number for the question it answered.

Standardising is normally called the fix for this. It is not a fix, it is a **different arbitrary choice**: "every column matters equally". Here that still leaves one column of pure noise casting exactly as many votes as the answer on every distance, which is why purity stops at 0.669 instead of 1.000. Scaling decides what the distance means, and no amount of it can tell you which columns are relevant. That part is your knowledge about the problem, not a property of the data.

## The update rule has to match the distance

Day 45 spent a section on LeetCode 462 proving that the minimiser of `sum |x - c|` is the median, while the minimiser of `sum (x - c)²` is the mean. On `[1, 2, 3, 100]` the mean is 26.5 and costs 147.0 in absolute distance; the median is 2.5 and costs 100.0.

That is not trivia. It is exactly the fact Lloyd's termination proof rests on. So what happens if you keep the Lloyd loop, switch the assignment to L1 distance because you want something robust to outliers, and leave the update as a mean?

![img](https://github.com/KingRei/100DaysPython/blob/master/day%2047%20-%20k-means%20and%20knn/imgs/day47_4.png?raw=true)

On 96 points - three blobs plus six scattered outliers - the L1 objective of the mean-updating version goes 562.5, 340.2, **342.3**, 329.4, 244.1, 196.6, 196.5, 196.5. It went *up* at step 1. The update step no longer minimises the thing the assignment step is measuring, so there is no longer any argument that it cannot make things worse, and the termination proof is gone. Over 600 (data set, seed) pairs, 150 runs are non-monotone and 555 finish worse than the correct median update.

It still converged here, and it will usually converge, which is the problem: a loop that is convergent by luck looks exactly like a loop that is convergent by proof.

## Squared distance only knows one shape

Squared distance to a single centre has circular level sets, and a point joins whichever centre is nearest, so the border between two clusters is the perpendicular bisector of the line joining their centroids. Every cluster k-means can express is therefore a convex cell with straight sides.

| data | purity | cluster sizes |
|---|---|---|
| two long thin bars | 0.523 | 145, 155 |
| two moons | 0.757 | 145, 155 |
| one blob of 200, one of 20 | 0.909 | 105, 115 |

The last row is the one worth staring at. A purity of 0.909 and two clusters of 105 and 115 look like a healthy result. They are not: the 20-point group was recovered **0 times out of 20**. k-means halved the big blob instead, because halving 200 points removes far more squared distance than isolating 20 does - and 0.909 is exactly the score you get for calling everything the majority class.

## k-nearest neighbours: the data is the model

There is no training step. You store the data; the cost all moves to query time.

```python
d2 = ((Xtrain - q) ** 2).sum(axis=1)
nn = np.argsort(d2)[:k]
return np.bincount(ytrain[nn]).argmax()
```

`k` is the smoothing knob: small k follows every piece of noise, `k = n` flattens the model into a constant. On 240 points from two overlapping moons, 160 train and 80 test:

| k | train accuracy | train, self excluded | test accuracy |
|---|---|---|---|
| 1 | 1.000 | 0.956 | 0.887 |
| 3 | 0.981 | 0.981 | 0.938 |
| 5 | 0.988 | 0.969 | 0.963 |
| 11 | 0.975 | 0.969 | **0.975** |
| 25 | 0.956 | 0.956 | 0.975 |
| 61 | 0.875 | 0.869 | 0.912 |
| 160 | 0.500 | 0.500 | 0.500 |

![img](https://github.com/KingRei/100DaysPython/blob/master/day%2047%20-%20k-means%20and%20knn/imgs/day47_5.png?raw=true)

Look at the first column at k = 1. It is 1.000, and it is 1.000 for any data set whose points are distinct, because the nearest neighbour of a training point is the point itself. It proves nothing at all, and 1.000 is the number people stop checking at. (Two identical points carrying different labels are the only way to lose, and that is a data problem rather than a model one.) Excluding the point itself drops it to 0.956, and the honest held-out number is 0.887. A validation split that leaks a single row does the same thing more quietly.

Held-out accuracy peaks at k = 11 and ties there with k = 25, which is itself worth noticing: the useful range of k is a plateau, not a point, and anything that reports a single best k is reporting the first of several equals.

Two smaller traps live here too. An even `k` can tie, and `np.bincount([0, 1]).argmax()` returns 0 - numpy breaks ties by smallest index, so the lowest-numbered class wins every tie in the data set, uniformly and invisibly. And at `k = 160`, the whole training set, every query gets the global majority label; on imbalanced data that constant model will still look accurate.

## In high dimensions, nothing is near

Both algorithms are built entirely on the word "nearest", and that word stops meaning anything as the number of columns grows.

![img](https://github.com/KingRei/100DaysPython/blob/master/day%2047%20-%20k-means%20and%20knn/imgs/day47_6.png?raw=true)

With 500 uniform random points, `(furthest - nearest) / nearest` is 81.292 at two dimensions and 0.110 at 1024: the furthest point goes from being 82 times further away to being 11% further away. Nothing broke. Every extra dimension adds one more term to a sum, the distance becomes an average over more and more coordinates, and averages concentrate.

The practical version does not look like a geometry problem at all. Keep exactly two informative columns and pad with pure noise:

| dimensions | informative columns | k = 11 test accuracy |
|---|---|---|
| 2 | 2 | 0.940 |
| 4 | 2 | 0.940 |
| 10 | 2 | 0.920 |
| 30 | 2 | 0.860 |
| 100 | 2 | 0.790 |
| 400 | 2 | 0.720 |

The two useful columns never changed. Accuracy fell from 0.940 to 0.720 because the distance function averages them with 398 columns of noise. No exception, no warning, just a model that quietly stopped working - and it is the same disease as the units section, wearing a different face.

## The cost of being lazy, and three ways out

Brute-force k-nearest is `O(n · d)` per query however it is written. The module grows the training set from 2,500 to 40,000 points at 16 dimensions and prints the cost per query alongside its ratio to the smallest run. The milliseconds depend on the machine and move between runs, so they are not quoted here; run it and watch the ratio column instead. It climbs with n, and no amount of vectorising stops it climbing - that buys a constant (Day 44), it does not bend the line.

There are exactly three escapes, and they fail in this order:

- **One dimension.** The data can be sorted, so binary search it. That is LeetCode 658 below, and it is the only case where the *search* is logarithmic rather than linear in n.
- **k out of n.** You need a set, not an order, so quickselect it instead of sorting. That is LeetCode 973 below, and Day 21's selection problem.
- **Many dimensions.** No exact spatial index survives; the usual rule of thumb is that k-d trees stop beating brute force somewhere in the low tens of dimensions, for the reason measured two sections up. So you approximate, and you pay for it with a recall below 1.0. That is the HNSW graph built on Day 31.

## LeetCode 973 - K Closest Points to Origin

**The task.** You are given a list of points on a plane and a number `k`. Return the `k` points that are closest to the origin `(0, 0)`, measured by ordinary straight-line distance. The answer is a set: you may return those `k` points in any order, and the problem guarantees the answer is unique.

**Input / output.** `points` is a list of `[x, y]` pairs of integers, and `k` is an integer with `1 <= k <= len(points)`. You return a list of `k` of those pairs.

**Example.** For `points = [[3, 3], [5, -1], [-2, 4]]` and `k = 2`, the squared distances are `18`, `26` and `20`. The two smallest are 18 and 20, so the answer is `[[3, 3], [-2, 4]]` - and `[[-2, 4], [3, 3]]` is equally correct.

**Constraints.** Up to 10⁴ points, each coordinate up to 10⁴ in absolute value. A full sort passes comfortably at that size, so the interesting question is not whether your solution runs but whether it does more work than was asked for. The coordinate bound matters a second time: `x² + y²` reaches about 2 × 10⁸, which is well inside the range of integers a float64 represents exactly, so the comparison stays exact in both forms - the square root is simply work you never needed to do.

Two details are worth keeping. Never take the square root: it is monotone, so it cannot change the ordering, and it turns an integer comparison into a floating-point one for nothing. And the answer being a set is what lets you skip the sort - a max-heap of size `k` makes each step `O(log k)` instead of `O(log n)`.

```python
import heapq

class Solution:
    def kClosest(self, points: list[list[int]], k: int) -> list[list[int]]:
        heap: list[tuple[int, int, int]] = []
        for x, y in points:
            heapq.heappush(heap, (-(x * x + y * y), x, y))
            if len(heap) > k:
                heapq.heappop(heap)
        return [[x, y] for _, x, y in heap]
```

The key is negated because Python only has a min-heap and the thing to discard is the *worst* candidate so far. Day 21's quickselect gets the same answer in `O(n)` expected time by partitioning around a random pivot and recursing into whichever side still contains the boundary; on 4,000 random points with k = 50 all three approaches return the identical set.

[leetcode.com/problems/k-closest-points-to-origin](https://leetcode.com/problems/k-closest-points-to-origin/)

## LeetCode 658 - Find K Closest Elements

**The task.** Given an array of integers sorted in ascending order, an integer `k`, and a value `x`, return the `k` elements of the array that are closest to `x`, themselves sorted in ascending order. Closeness is `|a - x|`, and when two candidates are equally far from `x`, the smaller one is considered closer.

**Input / output.** `arr` is sorted ascending and may contain duplicates, `k` satisfies `1 <= k <= len(arr)`, and `x` may lie anywhere - including outside the range of the array. You return a sorted list of `k` elements.

**Example.** For `arr = [1, 2, 3, 4, 5]`, `k = 4` and `x = 3`, the distances are 2, 1, 0, 1, 2. Taking the four closest means taking everything except one of the two elements at distance 2, and the tie rule says the smaller value wins, so `5` is dropped and the answer is `[1, 2, 3, 4]`. For `arr = [1, 1, 2, 3, 4, 5]`, `k = 4` and `x = -1`, everything is to the right of `x`, so the answer is simply the first four: `[1, 1, 2, 3]`.

**Constraints.** Up to 10⁴ elements, and - this is the constraint that changes the algorithm - the array is already sorted. That is what makes the answer a contiguous window, and it is also what makes the sort-by-distance solution wasteful: it throws away the ordering it was handed and pays `O(n log n)` to rebuild it.

Because the array is sorted, the `k` values closest to `x` are contiguous, so the answer is a window and what you search for is its left edge. Window `[lo, lo+k)` beats `[lo+1, lo+k+1)` exactly when dropping `arr[lo]` costs more than gaining `arr[lo+k]` saves - and both of those are just distances to an endpoint, so none of the values inside the window are ever examined.

```python
class Solution:
    def findClosestElements(self, arr: list[int], k: int, x: int) -> list[int]:
        lo, hi = 0, len(arr) - k
        while lo < hi:
            mid = (lo + hi) // 2
            if x - arr[mid] > arr[mid + k] - x:
                lo = mid + 1
            else:
                hi = mid
        return arr[lo:lo + k]
```

The `>` is the whole tie rule, and it must not be `>=`. On a tie, the problem wants the smaller value, which means keeping the left window. Swapping that one character still passes all three worked examples, and differs from the reference on 104 of 3,000 random inputs - 3.5%. On `arr = [1, 2, 3, 4, 5]`, `k = 2`, `x = 3`, where 2 and 4 are both one away, `>` gives `[2, 3]` and `>=` gives `[3, 4]`.

This is the one-dimensional case of k-nearest, and the only one in the entire day that escapes `O(n · d)`. The reason it can is entirely that one dimension admits a sort. In two dimensions there is no such order, and that is where the whole problem starts.

[leetcode.com/problems/find-k-closest-elements](https://leetcode.com/problems/find-k-closest-elements/)

## Complexity

| Operation | Time | Space |
|---|---|---|
| One k-means round, n points, k centroids, d dims | O(n · k · d) | O((n + k) · d) data + O(n · k) distance matrix |
| k-means, i rounds | O(i · n · k · d) | same as one round |
| k-means++ initialisation, as written here | O(n · k² · d) | O((n + k) · d) |
| k-means++ keeping a running minimum instead | O(n · k · d) | O(n) extra |
| k-nearest, one query, brute force | O(n · d + n log n) to sort, O(n · d + n log k) with a heap | O(n) |
| k-nearest, training | O(1) - there is no training | O(n · d) to keep the data |
| LeetCode 973, heap | O(n log k) | O(k) |
| LeetCode 973, quickselect | O(n) expected, O(n²) worst case | O(n) |
| LeetCode 658 | O(log(n − k) + k) | O(k) output |
| LeetCode 658, sort by distance | O(n log n) | O(n) |

## References

- Lloyd, *Least squares quantization in PCM*, IEEE Transactions on Information Theory 1982 (written at Bell Labs in 1957)
- Arthur and Vassilvitskii, [*k-means++: The Advantages of Careful Seeding*](https://theory.stanford.edu/~sergei/papers/kMeansPP-soda.pdf), SODA 2007
- Cover and Hart, *Nearest neighbor pattern classification*, IEEE Transactions on Information Theory 1967
- Beyer, Goldstein, Ramakrishnan and Shaft, [*When Is "Nearest Neighbor" Meaningful?*](https://members.loria.fr/MOBerger/Enseignement/Master2/Exposes/beyer.pdf), ICDT 1999 - the distance concentration result measured here
- [k-means clustering](https://en.wikipedia.org/wiki/K-means_clustering) and [k-nearest neighbors algorithm](https://en.wikipedia.org/wiki/K-nearest_neighbors_algorithm) on Wikipedia
- [scikit-learn: clustering](https://scikit-learn.org/stable/modules/clustering.html) - including what it does with an empty cluster, and [KMeans](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html) for what `n_init="auto"` resolves to
