# Suffix array + LCP, and Aho-Corasick

More details in:
https://medium.com/100-days-of-python

Interactive walkthrough: [demo.html](demo.html) - a single self-contained page
(no build step, works offline, 中文 / English toggle) that animates prefix doubling turning
characters into ranks, Kasai's `h` sliding down by at most one per step, the two binary searches
that turn a pattern into a contiguous block of the array, a deduplication report drowning in
shifted copies until the left-diversity test is added, Aho-Corasick's fail links being built by
BFS and then matching `ushers` in one pass, a streaming server holding back exactly
`depth[state]` characters for three stop strings at once, and LeetCode 1032 and 1044.

![img](https://github.com/KingRei/100DaysPython/blob/master/day%2026%20-%20suffix%20array%20and%20aho%20corasick/imgs/day26_1.png?raw=true)

Day 25 answered "where does this pattern occur in this text". Both of its algorithms preprocess
the **pattern** and then read the text, which is the right shape when the text is a stream and
the pattern is known. Today's two structures invert each of those assumptions. A suffix array
preprocesses the **text**, so questions can be asked afterwards - including questions nobody
could turn into a pattern, such as "what does this corpus repeat". Aho-Corasick preprocesses
**all the patterns at once**, so the text is read a single time no matter how many there are.

A suffix array is the sorted order of every suffix of a string, stored as starting positions.
For `banana` it is `[5, 3, 1, 0, 4, 2]`, six integers rather than six substrings, so the memory
is `O(n)`. Writing it as `sorted(range(n), key=lambda i: s[i:])` is the definition and is also
`O(n² log n)`, because each comparison can read the whole suffix. **Prefix doubling** removes
that: replace each character by its rank, and the first `2k` characters of suffix `i` become the
pair `(rank[i], rank[i+k])`. Comparing a pair of small integers is `O(1)` however long the
suffixes are, `k` doubles every round, and `banana` is fully ordered after three rounds. With a
comparison sort that is `O(n log² n)`; with day 22's counting sort it is `O(n log n)`.

## LCP, and why Kasai is linear

The array alone answers "where". The array that makes it *useful* is `lcp[i]`: how long a prefix
the neighbours `sa[i-1]` and `sa[i]` share - `[0, 1, 3, 0, 0, 2]` for `banana`. Computing it pair
by pair is `O(n²)`. Kasai's algorithm walks the suffixes in **text** order instead of array
order and keeps a running `h`, because suffix `i+1` is suffix `i` with its first character
removed and therefore already shares at least `h - 1` with its own neighbour. `h` falls by at
most one per step and rises at most `n` times in total, so the whole loop is `O(n)` - the same
amortised argument that bounded KMP's fall-backs on day 25.

Two consequences follow immediately. Since the array is sorted, the common prefix of any two
suffixes is the minimum `lcp` between them, so the **largest entry is the longest repeated
substring** - `ana` in `banana`, `issi` in `mississippi`. And a pattern query is two binary
searches (day 23): every suffix beginning with `ana` must be contiguous, so `sa_range` returns a
block, the occurrence count is `end - start` in `O(1)`, and each of the `O(log n)` probes reads
only `m` characters. The trade against KMP is explicit: KMP re-reads the text for every new
pattern, a suffix array reads it once and never again.

## Deduplicating a corpus, and the line that makes the numbers real

The reason to build this rather than a hash index is the class of questions where **there is no
pattern to look for**. Training-corpus deduplication is the standard one: two documents share a
licence header and a boilerplate function body, and you want every block that is at least
`min_len` long and occurs more than once, without knowing in advance what those blocks are. A
run of `lcp >= min_len` in the array is exactly such a block, and its length is the run's
minimum.

Written that way the function is also useless. On the module's two-document example (`min_len = 12`) it
reports 68 spans and claims a saving of 984% of a 214-character input, because a block and every
one of its shifted copies are all reported separately - `he cat sat.` is `the cat sat.` with the
first character removed, and both are counted. The fix is one line - keep a run only if its occurrences are **not all preceded by the
same character**:

```python
left = {text[p - 1] if p else None for p in group}
if len(left) > 1:                      # left-diverse: a maximal repeat
    out.append((len(group), text[group[0]:group[0] + run]))
```

A span whose occurrences all share a left neighbour can always grow one character to the left,
so it is not a maximal repeat, only somebody else's tail. With the test the same input reports
two spans and a saving of 42.1%, and the whole pipeline is still two linear passes over the
text. This is the structure behind Lee et al.'s work on deduplicating training data, where
removing repeated passages lowered both loss and memorisation.

## Aho-Corasick is KMP's failure function on a trie

The other inversion is many patterns at once. Running KMP `k` times costs `O(k · n)` and reads
the text `k` times, which is the wrong shape for a guardrail list or a banned-phrase filter.
Aho-Corasick puts every pattern in a trie - shared prefixes stored once - and then adds the same
object day 25 built: a **fail link** from each node to the node spelling the longest proper
suffix of it that is still in the trie. Built BFS by depth, so the target's fail link is always
already known. The check that these really are one idea: feed the trie a single pattern, the
trie degenerates into a chain, and the fail depths equal `kmp_failure(pattern)` exactly.

One detail is easy to omit and silently loses matches. Arriving at the state for `she` also
completes `he`, so `out[v]` must inherit `out[fail[v]]`; without that line, `find_all('ushers')`
returns `she` and `hers` but not the `he` inside `she`. With it, matching is one pass in which
the cursor never moves backwards: on the module's 126-character sample with five patterns, five
KMP passes spend 675 character comparisons and one Aho-Corasick pass spends 159 transitions over
a 15-node automaton, with identical results. The cost is `O(n + total pattern length + matches)`
and **does not depend on `k`**.

## Streaming stop strings, and when not to use this

Day 25's streaming problem returns with `k` stop strings instead of one. A server may not emit a
character that could turn out to be the opening of a stop string, so it holds text back, and
with one pattern the hold-back was "how much of the pattern currently matches". With `k`
patterns the naive version asks that of each pattern and takes the maximum. In the automaton the
number is already there: it is `depth[state]`, because the state spells the longest prefix of
*any* pattern that the tail currently matches. Three stop strings cost the same per character as
one, and `MultiStopMatcher` emits `The answer is 42.` from chunks that split `</s>` three ways,
having held back one character, then two, then cut cleanly.

The honest conclusion is that sglang does not do this, and is right not to. It clips the tail it
re-checks to `stop_str_max_len + 1` characters and runs a plain substring search per stop string,
`O(k · L²)` per step - roughly 162 character comparisons for two stop strings, which is noise
beside a forward pass. Bounding the input instead of improving the algorithm is a legitimate
engineering choice whenever the bound holds. Aho-Corasick earns its keep at the other end of
`k`: PII, guardrail and banned-phrase lists with thousands of entries, where re-scanning per
entry is no longer free.

## The LeetCode problems

LC 1032 (Stream Checker) is the automaton with the buffer deleted: letters arrive forever, and
the object keeps **one integer**, stepping it per query and returning whether `out[state]` is
non-empty. The tempting solution - a trie of reversed words plus a growing buffer that is
re-walked on each query - stores history the state already summarises.

LC 1044 (Longest Duplicate Substring) is a deliberate rerun. Day 25 solved it with binary search
over the length plus a rolling hash, a solution that can in principle return a wrong answer and
is kept honest only by its verification step. Here it is the largest `lcp` entry: exact, with no
hashing and no randomness, because sorting has already placed every repeat next to its twin. LC
616 / 758 (Add Bold Tag) is the multi-pattern case - mark the covered intervals in one
Aho-Corasick pass, then merge them.

## Complexity

`n` = text length, `m` = pattern length, `k` = number of patterns, `L` = total pattern length.

| Operation | Time | Space |
|---|---|---|
| `suffix_array_naive` (the definition) | O(n² log n) | O(n²) |
| `suffix_array` (prefix doubling) | O(n log² n) | O(n) |
| Prefix doubling with a counting sort | O(n log n) | O(n) |
| `lcp_kasai` | O(n) | O(n) |
| `lcp_naive` | O(n²) | O(n) |
| `sa_range` / `sa_count` (one pattern) | O(m log n) | O(1) |
| `longest_repeated_substring` | O(n log² n) | O(n) |
| `longest_common_substring` (two texts) | O(n log² n) | O(n) |
| `duplicate_spans` / `dedup_report` | O(n log² n) | O(n) |
| Aho-Corasick build | O(L) | O(L · sigma) |
| Aho-Corasick search | O(n + matches) | O(L · sigma) |
| `k` separate KMP passes | O(k · n + L) | O(L) |
| Streaming multi-stop, per character | O(1) amortised | O(L) |
| sglang's clipped per-stop scan, per step | O(k · L²) on a clipped tail | O(1) |
| LC 1032 `query` | O(1) amortised | O(L · sigma) |

## References
- [Suffix array - Wikipedia](https://en.wikipedia.org/wiki/Suffix_array)
- [Suffix Arrays: A New Method for On-Line String Searches (Manber & Myers, 1990)](https://doi.org/10.1137/0222058)
- [Linear-Time Longest-Common-Prefix Computation in Suffix Arrays (Kasai et al., 2001)](https://doi.org/10.1007/3-540-48194-X_17)
- [LCP array - Wikipedia](https://en.wikipedia.org/wiki/LCP_array)
- [Aho-Corasick algorithm - Wikipedia](https://en.wikipedia.org/wiki/Aho%E2%80%93Corasick_algorithm)
- [Efficient String Matching: An Aid to Bibliographic Search (Aho & Corasick, 1975)](https://doi.org/10.1145/360825.360855)
- [Deduplicating Training Data Makes Language Models Better (Lee et al., 2021)](https://arxiv.org/abs/2107.06499)
- [`sglang/srt/managers/schedule_batch.py`](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/managers/schedule_batch.py)
- [LeetCode 1032 · Stream of Characters](https://leetcode.com/problems/stream-of-characters/)
- [LeetCode 1044 · Longest Duplicate Substring](https://leetcode.com/problems/longest-duplicate-substring/)
- [LeetCode 616 · Add Bold Tag in String](https://leetcode.com/problems/add-bold-tag-in-string/)
