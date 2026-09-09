# Edit distance and similarity: Levenshtein, Jaccard, MinHash

More details in:
https://medium.com/100-days-of-python

Interactive walkthrough: [demo.html](demo.html) - a single self-contained page
(no build step, works offline, 中文 / English toggle) that fills the Levenshtein table cell by
cell and shows which of the three predecessors won, walks the backtrace that turns the table
back into an edit script, collapses the table to two rows and then to one, shrinks it again to
a diagonal band once a threshold `k` is known, re-runs the same recurrence over word tokens to
get a WER breakdown and over an LCS to get a `git diff`, then switches to sets - shingles,
Jaccard, the n² wall, MinHash signatures, LSH banding and its S-curve - and finishes with
LeetCode 72, 1143 and 161.

![img](https://github.com/KingRei/100DaysPython/blob/master/day%2027%20-%20edit%20distance%20and%20similarity/imgs/day27_1.png?raw=true)

Days 24 to 26 all asked exact questions: does this pattern occur, at which positions, what is
repeated verbatim. Real text does not cooperate. A user types `recieve`, an ASR model drops a
word, two crawled pages differ only by a cookie banner - each of those is a *near* match that a
trie, a KMP automaton or a suffix array reports as no match at all. Today is the two ways to
put a number on "near", and they are not variants of one idea; they sit at opposite ends of a
scale/exactness trade.

**Levenshtein distance** is the exact one. The distance between two strings is the smallest
number of single-character deletions, insertions and substitutions that turns one into the
other, and `d[i][j]`, the distance between the two prefixes `a[:i]` and `b[:j]`, has exactly
three predecessors:

```python
d[i][j] = min(d[i-1][j]   + 1,                    # delete a[i-1]
              d[i][j-1]   + 1,                    # insert b[j-1]
              d[i-1][j-1] + (a[i-1] != b[j-1]))   # substitute, or match for free
```

The base cases are the first row and column - turning a prefix into the empty string costs one
deletion per character - and the answer is the bottom-right corner. `kitten` to `sitting` is 3.

## The table is not just the number

`min` throws away *which* predecessor won, so the number alone is often the less useful half of
the output. Walking backwards from the corner and re-testing the three candidates recovers the
**edit script**: substitute `k` with `s`, substitute `e` with `i`, insert `g`. That is what a
spell checker needs to explain itself, what a diff tool prints, and what an ASR error analysis
is actually made of.

If the script is not needed, the table is wasteful. Row `i` only ever reads row `i-1`, so two
rows suffice and the memory drops from `O(nm)` to `O(min(n, m))` - swap the arguments so the
shorter string indexes the row. The cost is exactly the thing above: with the table gone there
is nothing to walk backwards through. Hirschberg's algorithm buys the script back in `O(n)`
space by divide-and-conquer, for roughly twice the time; the honest summary is that you pick
two of {distance, script, linear space} and pay for the third.

![img](https://github.com/KingRei/100DaysPython/blob/master/day%2027%20-%20edit%20distance%20and%20similarity/imgs/day27_2.png?raw=true)

## Bounded k: most of the table can never matter

The far more valuable optimisation is a *question* change. Spelling correction, fuzzy joins and
dedup filters almost never ask "how far apart are these"; they ask "are these within `k`". Once
`k` is fixed, a path that strays more than `k` cells from the main diagonal has already spent
more than `k` operations, so only the band `max(0, i-k) <= j <= min(m, i+k)` can hold the
answer - width `2k+1`, so the work is `O(k · n)` instead of `O(nm)`.

Two guards make it fast in the cases that matter. `|n - m| > k` is answered before a single
cell is filled, because length difference is a lower bound on the distance. And if every cell
in a row exceeds `k`, no later row can come back under it, so the loop exits early. On the
module's 132-character pair that differs by 3 substitutions, `k = 3` fills 912 cells out of
17424 - 5.2% of the table - `k = 1` gives up after 152 cells, and a pair 5 characters apart in
length is rejected with 0 cells.

![img](https://github.com/KingRei/100DaysPython/blob/master/day%2027%20-%20edit%20distance%20and%20similarity/imgs/day27_3.png?raw=true)

## The same recurrence over words, and without substitution

Nothing in the recurrence knows it is comparing characters. Feed it word tokens and the
distance becomes **word error rate**, the standard ASR metric: WER is the edit distance divided
by the length of the reference, and the backtrace splits it into substitutions, deletions and
insertions. The module's example scores `(2S + 1D + 0I) / 8 = 0.375`, and the breakdown is the
part an error analysis lives on - a system that mostly deletes has a different problem from one
that mostly substitutes.

Delete one branch and you get a different classic. Forbid substitution, and the cheapest way to
align two sequences is to keep their **longest common subsequence** and edit everything else:
`distance = n + m - 2 · LCS`. That is why `diff` shows a changed line as a deletion plus an
insertion rather than as a modification - substitution is not in its vocabulary. The same
`O(nm)` table with two predecessors instead of three yields both the LCS length and, by
backtracing, the `+`/`-` script.

## Jaccard: when the pair count is the problem

All of the above is `O(nm)` **per pair**, and pairs are the real enemy. Deduplicating a corpus
of 10 million documents is 5 × 10¹³ pairs; at a microsecond each that is over a year, and the
per-pair cost is irrelevant to that conclusion. Scaling needs a similarity that can be
*indexed*, not merely computed quickly.

Jaccard similarity is that. Cut each document into overlapping `k`-word **shingles**, treat it
as the set of them, and define `J(A, B) = |A ∩ B| / |A ∪ B|`. Order inside a shingle is
preserved, order between shingles is thrown away, and a document becomes a set - which is the
whole point, because sets have cheap sketches and strings do not. On the module's corpus a
phrase swap scores 0.773, an appended clause 0.830, an unrelated document 0.000.

![img](https://github.com/KingRei/100DaysPython/blob/master/day%2027%20-%20edit%20distance%20and%20similarity/imgs/day27_4.png?raw=true)

## MinHash: a fixed-size sketch whose collisions are the answer

Hash every element of a set and keep the minimum. For two sets, the smallest hash in `A ∪ B`
lies in `A ∩ B` with probability exactly `|A ∩ B| / |A ∪ B|`, and their min-hashes agree
precisely in that case - so `P(equal) = J`. Repeat with `K` independent hash functions and the
fraction of agreeing slots is an unbiased estimate of `J`, with standard error about
`sqrt(J(1-J)/K)`: a document of any length is now `K` integers, and comparison is `O(K)`
regardless of how long the documents were.

`K` buys accuracy at the usual quarter-rate: on one pair with true `J = 0.773`, `K = 16` gives
0.688, `K = 64` gives 0.828, `K = 256` gives 0.766, `K = 1024` gives 0.771, and the mean error
across all pairs falls 0.083 → 0.038 → 0.017 as `K` goes 16 → 64 → 256.

One trap is worth stating plainly, because it survives every test on one machine and then
breaks: **Python's built-in `hash()` on `str` is randomised per process**. Signatures written
today and compared tomorrow will disagree, and no test that runs in a single process will show
it. The module uses seeded `hashlib.blake2b` instead.

![img](https://github.com/KingRei/100DaysPython/blob/master/day%2027%20-%20edit%20distance%20and%20similarity/imgs/day27_5.png?raw=true)

## LSH: turning "compare all pairs" into "look in a bucket"

MinHash made each comparison `O(K)` but there are still `n(n-1)/2` of them. Locality-sensitive
hashing removes the pairs themselves. Split the `K` slots into `b` bands of `r` rows, hash each
band, and let two documents be candidates if **any** band matches exactly. A band matches with
probability `J^r`, so

```
P(candidate) = 1 - (1 - J^r)^b
```

which is an S-curve with its steep part near `(1/b)^(1/r)`. That knob is the design: `b = 32,
r = 4` puts the threshold near 0.42, `b = 16, r = 8` near 0.71, `b = 8, r = 16` near 0.88. Pick
it just under the similarity you care about and the curve does the filtering - `P = 0.001` at
`J = 0.3` and `0.947` at `J = 0.8` for the middle setting.

The last point is the one that makes the whole thing safe to deploy: **LSH is a filter, not an
answer**. It proposes candidates; the exact Jaccard is then computed on those few pairs, so
false positives cost a little work and nothing else. On the module's 8-document corpus the
pipeline verifies 4 pairs instead of 28 - 14.3% of the work - and returns exactly what brute
force returns. At corpus scale that ratio is the difference between a job that finishes and one
that does not, and it is why MinHash-LSH is the standard dedup pass in dataset pipelines like
C4 and RefinedWeb.

![img](https://github.com/KingRei/100DaysPython/blob/master/day%2027%20-%20edit%20distance%20and%20similarity/imgs/day27_6.png?raw=true)

## The LeetCode problems

LC 72 (Edit Distance) is the table, unchanged - the interview value is in stating the three
predecessors as *operations* and then offering the rolling-row version and the reason it loses
the script.

LC 1143 (Longest Common Subsequence) is the same table with substitution removed, and worth
solving right after 72 because the pair makes the recurrence's structure visible: one branch
deleted, one different algorithm.

LC 161 (One Edit Distance) is the interesting one, because the right answer builds **no table
at all**. If the lengths differ by more than one, or the strings are equal, it is already
decided; otherwise scan to the first difference and check whether skipping one character on the
longer side (or both, when the lengths match) makes the tails equal. `O(n)` time, `O(1)` space -
the general algorithm is the wrong tool once `k` is nailed down to 1, which is the bounded-band
idea taken to its limit.

## Complexity

`n`, `m` = lengths of the two inputs; `k` = distance threshold; `K` = MinHash signature length;
`b`, `r` = LSH bands and rows (`b · r = K`); `N` = number of documents.

| Operation | Time | Space |
|---|---|---|
| `edit_table` / `edit_distance` | O(n·m) | O(n·m) |
| `edit_ops` (backtrace, table already built) | O(n + m) | O(n + m) |
| `edit_distance_rows` (rolling row) | O(n·m) | O(min(n, m)) |
| Hirschberg (script in linear space) | O(n·m) | O(min(n, m)) |
| `edit_distance_bounded(a, b, k)` | O(k·n) | O(min(n, m)) |
| Length gate `abs(n - m) > k` | O(1) | O(1) |
| `lcs_length` / `diff_script` | O(n·m) | O(n·m) |
| `word_error_rate` (over `w` tokens) | O(w²) | O(w²) |
| `shingles` | O(n) | O(n) |
| `jaccard` (two sets) | O(n + m) | O(n + m) |
| `all_pairs_jaccard` | O(N²·n) | O(N·n) |
| `minhash_signature` | O(K·n) | O(K) |
| `estimate_jaccard` (two signatures) | O(K) | O(1) |
| `band_buckets` + `candidate_pairs` | O(N·K + candidates) | O(N·K) |
| `near_duplicates` (LSH then exact verify) | O(N·K + candidates·n) | O(N·K) |
| LC 72 / LC 1143 | O(n·m) | O(min(n, m)) for the value only |
| LC 161 `is_one_edit_distance` | O(n) | O(1) |

## References
- [Levenshtein distance - Wikipedia](https://en.wikipedia.org/wiki/Levenshtein_distance)
- [Binary codes capable of correcting deletions, insertions, and reversals (Levenshtein, 1966)](https://nymity.ch/sybilhunting/pdf/Levenshtein1966a.pdf)
- [A linear space algorithm for computing maximal common subsequences (Hirschberg, 1975)](https://doi.org/10.1145/360825.360861)
- [Word error rate - Wikipedia](https://en.wikipedia.org/wiki/Word_error_rate)
- [Longest common subsequence - Wikipedia](https://en.wikipedia.org/wiki/Longest_common_subsequence)
- [Jaccard index - Wikipedia](https://en.wikipedia.org/wiki/Jaccard_index)
- [MinHash - Wikipedia](https://en.wikipedia.org/wiki/MinHash)
- [On the resemblance and containment of documents (Broder, 1997)](https://doi.org/10.1109/SEQUEN.1997.666900)
- [Locality-sensitive hashing - Wikipedia](https://en.wikipedia.org/wiki/Locality-sensitive_hashing)
- [Mining of Massive Datasets, chapter 3 (Leskovec, Rajaraman, Ullman)](http://www.mmds.org/)
- [Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer (C4 dedup, Raffel et al., 2019)](https://arxiv.org/abs/1910.10683)
- [The RefinedWeb Dataset for Falcon LLM (Penedo et al., 2023)](https://arxiv.org/abs/2306.01116)
- [LeetCode 72 · Edit Distance](https://leetcode.com/problems/edit-distance/)
- [LeetCode 1143 · Longest Common Subsequence](https://leetcode.com/problems/longest-common-subsequence/)
- [LeetCode 161 · One Edit Distance](https://leetcode.com/problems/one-edit-distance/)
