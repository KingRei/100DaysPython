# Quick sort and merge sort - the same idea, and the one line that separates them

More details in:
https://medium.com/100-days-of-python

Interactive walkthrough: [demo.html](demo.html) - a single self-contained page
(no build step, works offline, 中文 / English toggle) that animates a merge sort counting
its inversions as it goes, a Lomuto partition placing a pivot for good, the two inputs that
turn quick sort into an n^2 algorithm, the three-way partition that fixes the one a random
pivot cannot, and a parallel merge splitting on medians while the work and depth are counted
on screen.

![img](https://github.com/KingRei/100DaysPython/blob/master/day%2020%20-%20quick%20sort%20and%20merge%20sort/imgs/day20_1.png?raw=true)

Both algorithms are divide and conquer, both are O(n log n), and both split the array in
half. The only difference is *where the work happens*. Merge sort splits by index - no
comparisons, no thinking - and pays for everything in the join. Quick sort pays everything
in the split, arranging the array around a pivot so thoroughly that the join is a no-op.

Every practical difference falls out of that one sentence. Merge sort needs O(n) scratch
space because a merge cannot be done in place; quick sort sorts in place. Merge sort is
stable and has no bad input; quick sort is neither. Merge sort can sort a file larger than
memory; quick sort cannot, because it needs random access to everything.

## Merge sort - the work is in the join

```python
def merge(left, right):
    out, i, j = [], 0, 0
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:      # <= is the entire reason it is stable
            out.append(left[i]); i += 1
        else:
            out.append(right[j]); j += 1
    return out + left[i:] + right[j:]
```

Two fingers walk forward and every comparison fixes one output slot. Because the split never
looks at the data, the recursion tree has the same shape for every input - sorted, reversed
and all-equal data all cost exactly the same. That is what "no worst case" means.

Change `<=` to `<` and equal elements swap places. Sorting `[(1,'a'), (0,'b'), (1,'c'),
(0,'d')]` by the first field gives `[(0,'b'), (0,'d'), (1,'a'), (1,'c')]` - `a` still before
`c`, `b` still before `d`. One character.

![img](https://github.com/KingRei/100DaysPython/blob/master/day%2020%20-%20quick%20sort%20and%20merge%20sort/imgs/day20_2.png?raw=true)

The merge also answers a question nobody asked it. When an element is emitted from the right
half, every value still unemitted on the left is larger than it, and every one of those pairs
is an inversion:

```python
inv += len(left) - i        # one line, inside the else branch
```

`[2, 4, 1, 3, 5]` has 3 inversions; 64 random values have 1230; the reverse of them has 2012.
Counting inversions by brute force is O(n^2); this is free.

## Quick sort - the work is in the split

Lomuto's partition holds one invariant: everything in `a[lo:i]` is smaller than the pivot.
After the scan the pivot swaps into position `i`, and that slot is **final** - no later call
ever touches it. There is nothing to merge because the two sides are already in the right
places.

## The two inputs that break the textbook version

Measured at n = 200, the same code, four configurations:

| input | lomuto / last pivot | lomuto / random pivot | three-way | merge sort |
|---|---|---|---|---|
| already sorted | 19900 cmp, depth 199 | 1502, depth 15 | 1494, depth 12 | 732, depth 8 |
| all equal | 19900 cmp, depth 199 | 19900, depth 199 | 200, depth 1 | 732, depth 8 |
| random | 1509, depth 14 | 1636, depth 14 | 1766, depth 16 | 1282, depth 8 |

Two separate failures hide in that table. **Sorted input with a fixed pivot** splits n-1 / 0
every time: 199 levels of recursion, and CPython gives up at 1000. One line of randomisation
fixes it. **All-equal input** defeats the random pivot as well, because `a[j] < p` is never
true no matter which element is chosen - a two-way partition simply has no "equal" region.

Neither case raises anything. They just get slow, which on an online judge reads as TLE.

## Three-way partition - the Dutch national flag

![img](https://github.com/KingRei/100DaysPython/blob/master/day%2020%20-%20quick%20sort%20and%20merge%20sort/imgs/day20_3.png?raw=true)

```python
while i <= gt:
    if a[i] < p:
        a[lt], a[i] = a[i], a[lt]; lt += 1; i += 1
    elif a[i] > p:
        a[i], a[gt] = a[gt], a[i]; gt -= 1    # i does NOT advance:
    else:                                     # the value swapped in from
        i += 1                                # the right is still unexamined
```

The middle region is the point: values equal to the pivot are already in their final places
and retire immediately. The recursion depth is then decided by the number of *distinct*
values, not the number of elements. On 400 equal values the textbook version needs 79,800
comparisons at depth 399; the three-way version needs 400 at depth 1.

The `gt` branch is the line everyone gets wrong. The value swapped in from the right has
never been looked at, so `i` must stay where it is.

## External sort - when the data does not fit

Merge sort is the only one of the two that survives here, precisely because merging is
sequential: read a chunk, sort it, write it out as a run, then merge the runs with a heap.
Sorting 1000 values with room for 100 produces 10 runs and never holds more than 100 items -
10% of the data. Quick sort cannot do this at all; partitioning needs random access to the
whole array.

## Work and depth - why the obvious parallel merge sort is not fast

![img](https://github.com/KingRei/100DaysPython/blob/master/day%2020%20-%20quick%20sort%20and%20merge%20sort/imgs/day20_4.png?raw=true)

**Work** is the total number of operations; **depth** is the longest chain of operations that
must happen in order. Sequential composition adds depths, parallel composition takes the max,
and `work / depth` is the most cores the algorithm could ever keep busy. Measured on
n = 1024:

| | work | depth | most usable cores |
|---|---|---|---|
| one core | 10,240 | 10,240 | 1x |
| parallel halves, plain merge | 10,240 | 2,046 | 5x |
| parallel halves, parallel merge | 18,857 | 235 | 80x |

The middle row is the trap, and it is what most people write when asked for a parallel merge
sort. The two recursive calls run at the same time, but the final merge of n elements is a
sequential loop - step k needs to know what step k-1 emitted - so the depth stays O(n) and
buying more cores changes nothing.

The fix is to make the merge itself divide and conquer: take the median of the longer half,
binary-search for its position in the other half, and the two sides of that cut can be merged
independently. Depth falls from 2,046 to 235 for 84% more work. Trading work for depth is the
whole game in parallel algorithms, and it is the same ledger that comes back for GPU kernels
and inference schedulers later in the series.

## Sample sort - the same idea across machines

![img](https://github.com/KingRei/100DaysPython/blob/master/day%2020%20-%20quick%20sort%20and%20merge%20sort/imgs/day20_5.png?raw=true)

On a cluster you cannot afford log n rounds of pairwise merging, so sample sort splits
*once*: sample the data, sort the samples, take p-1 splitters, do one all-to-all exchange,
and let every machine sort locally. It is quick sort's partition, scaled out - with p-1
pivots instead of one.

The only failure mode is load imbalance, and the sort finishes when the **slowest** bucket
finishes. Sampling is a lottery, so the honest measurement is a mean over many samplings -
200 of them here, on 20,000 values across 8 machines:

| samples per machine | mean imbalance | worst seen |
|---|---|---|
| 1 | 2.76x | 5.54x |
| 4 | 1.82x | 3.41x |
| 32 | 1.26x | 1.82x |

One splitter per machine leaves the busiest machine 176% above a fair share. Oversampling
costs a slightly larger sort of the samples and buys almost all of it back.

## LeetCode 912 - Sort an Array

The judge's data is duplicate-heavy and contains sorted stretches, which is exactly the pair
of inputs above. An accepted solution needs four things: a **random pivot** so no input can
predict the split, a **three-way partition** so duplicates retire instead of being
re-partitioned, insertion sort for short runs, and recursion on the *smaller* side with a
loop on the larger one so the stack stays O(log n).

```python
if lt - lo < hi - gt:
    self.sort(a, lo, lt - 1); lo = gt + 1     # loop on the big side
else:
    self.sort(a, gt + 1, hi); hi = lt - 1
```

## Complexity

| Operation | Time | Space |
|---|---|---|
| merge sort | O(n log n) always | O(n) |
| quick sort, random pivot | O(n log n) expected | O(log n) stack |
| quick sort, adversarial input | O(n^2) | O(n) stack |
| three-way quick sort, k distinct values | O(n log k) | O(log n) |
| counting inversions | O(n log n) | O(n) |
| external merge sort | O(n log n), one pass per merge level | O(chunk + runs) |
| parallel merge sort | O(n log n) work, O(log^2 n) depth per merge | O(n) |
| sample sort | O(n log n) work, one all-to-all round | O(n / p) per machine |

## Run it

```bash
python quick_merge_sort.py
```

Prints the merge with its comparison count and inversion count, the stability demonstration,
both partition schemes, the n = 200 table above, an external sort with its peak memory, the
work/depth ledger, the sample-sort imbalance study, and LeetCode 912 - ending in asserts.

## References

- [Merge sort](https://en.wikipedia.org/wiki/Merge_sort)
- [Quicksort](https://en.wikipedia.org/wiki/Quicksort)
- [Dutch national flag problem](https://en.wikipedia.org/wiki/Dutch_national_flag_problem)
- [External sorting](https://en.wikipedia.org/wiki/External_sorting)
- [Merge sort - parallel merge](https://en.wikipedia.org/wiki/Merge_sort#Parallel_merge)
- [Samplesort](https://en.wikipedia.org/wiki/Samplesort)
- [Analysis of parallel algorithms (work and depth)](https://en.wikipedia.org/wiki/Analysis_of_parallel_algorithms)
- [LeetCode 912 - Sort an Array](https://leetcode.com/problems/sort-an-array/)
