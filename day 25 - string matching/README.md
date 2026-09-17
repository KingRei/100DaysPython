# String matching: KMP and Rabin-Karp

More details in:
https://medium.com/100-days-of-python

Interactive walkthrough: [demo.html](demo.html) - a single self-contained page
(no build step, works offline, 中文 / English toggle) that animates the naive matcher throwing
away everything it just learned, the failure function being built one border at a time, KMP
searching without ever moving the text index backwards, a stop string arriving split across
four streaming chunks, a speculative-decoding batch jumping clean over that stop string, the
rolling hash rolling and colliding, the chained SHA-256 that names a KV cache page, and
LeetCode 459 and 214 solved out of the border table alone.

![img](https://github.com/KingRei/100DaysPython/blob/master/day%2025%20-%20string%20matching/imgs/day25_1.png?raw=true)

Finding a pattern inside a text looks like the most elementary problem in this series, and the
naive solution is four lines: line the pattern up at every position and compare. What makes it
worth a day is the reason those four lines are slow. On `'a' * 40 + 'aaab'` searched for
`'aaab'`, the naive matcher spends 164 character comparisons and KMP spends 84, and the gap is
not cleverness - it is that the naive version *forgets*. At the moment a comparison fails at
offset `j`, it has just proved that the last `j` characters of the text are the first `j`
characters of the pattern. It throws that proof away, slides one position right and starts
again from zero.

Knuth-Morris-Pratt keeps the proof. Before the search it builds `fail`, where `fail[i]` is the
length of the longest **border** of `pat[:i+1]` - the longest prefix that is also a suffix. For
`ababaca` the table is `[0, 0, 1, 2, 3, 0, 1]`, which says that after matching `ababa` and then
failing, the only alignment still worth trying is the one that keeps the `aba` already matched.
So on a mismatch the pattern index falls back to `fail[j-1]` and the **text index never moves
backwards at all**. Each character of the text is consumed exactly once; `k` rises by at most
one per step, so the total number of fall-backs is bounded by the number of rises, and both
the table and the search are `O(m)` and `O(n)`.

That "never rewind" property is not a micro-optimisation. It is what makes the algorithm usable
on a stream - and an LLM inference server is a machine that produces exactly one kind of stream.

## The stop string that arrives in pieces

A generation request carries stop strings: `</s>`, `\n\nHuman:`, whatever the application uses
to end a turn. The server has to detect them while it is *still streaming text out to the
client*, which forbids both of the obvious implementations. Waiting for generation to finish is
not streaming, and re-scanning the whole output on every token is quadratic in the output
length.

One KMP state per stop string solves it, and the module's `StreamStopMatcher` is that: an
integer `j` per pattern, updated per character, with no memory of the text at all. The
consequence worth internalising is the **hold-back**. When `j` is 3, the last three characters
might turn out to be the start of a stop string, and text already sent to the client cannot be
recalled - so those three characters are withheld until the next chunk resolves them. Feeding
`['The answer is 4', '2.<', '/s', '> and then some more']` emits `The answer is 4`, then `2.`
while holding `<`, then nothing at all while holding `<`, `/`, `s`, and finally stops. The
amount held back is exactly the longest partial match across all stop strings.

sglang solves the same problem in `schedule_batch.py`, and it is instructive that it does *not*
use KMP: `check_match_stop_str_prefix` tries every suffix length and compares slices, which is
`O(m²)`. That is a perfectly good decision, because the tail it examines is first clipped to
`stop_str_max_len + 1` characters. Bounding the input is a legitimate alternative to a better
algorithm, provided you can prove the bound - and this is where it gets interesting, because
that bound turned out to be wrong.

## Tokens are not characters

A stop string is written in characters. A model emits tokens, and token boundaries have nothing
to do with the strings a user typed: in the module's toy vocabulary the stop string `]]` has its
first `]` buried inside the token `" 42]"` and its second as the token `"]"`. So each step has
to decode the last few tokens back into text and search there. The question is how few is safe.

With one token accepted per step, a window of `stop_max_len + 1` characters is enough, because
a stop string that just completed must end on the token that was just appended. Speculative
decoding breaks that assumption: several tokens land at once, and a stop string can complete
*in the middle* of the accepted batch, sitting entirely behind a fixed-size window. In the
module's example, one token per step detects `]]` at token 5, and four tokens per step with the
same fixed window **never detects it at all**. sglang's `_stop_match_tail_len` is the fix:

```python
min(stop_str_max_len + 1 + max(new_accepted_len - 1, 0), len(output_ids))
```

The extension by `accepted - 1` pulls the whole batch back into view. Using a character count as
a token count is safe in the conservative direction because a token decodes to at least one
character. Detecting the stop is then only half the answer - the request's finish length is
measured in tokens, so `_locate_str_stop_finished_len` decodes growing prefixes of the window
until the stop appears, which reports "finish after 5 tokens, keep `The answer is 42`". The
match lives in character space, the bookkeeping lives in token space, and decoding is the only
bridge between them.

## Rabin-Karp, and the verify step that is not optional

Rabin-Karp asks a different question: instead of comparing strings, compare **fingerprints**.
Read the current window as a base-256 number modulo a large prime, and sliding one place right
costs one multiply-add - subtract the departing character times `base^(m-1)`, multiply by the
base, add the arriving character. Unequal fingerprints mean *definitely not equal*; equal
fingerprints mean *maybe*, so every fingerprint hit is followed by a real comparison.

That verify line is part of the algorithm, not defensive programming. With `mod = 101` the
strings `aah` and `aca` share fingerprint 41, and searching for `aca` inside `xaahy` produces
one fingerprint hit, one verified rejection, and zero reported matches. Delete the verify and
nothing crashes: the function silently returns a position where the pattern is not. Against a
KMP-shaped problem Rabin-Karp wins nothing; where it wins is many patterns of the same length in
one pass, since a set of fingerprints costs the same to test as one.

## The other kind of fingerprint: KV cache page ids

sglang names each KV cache page with a chained SHA-256 (`mem_cache/utils.py`, `get_hash_str`):
the digest of a page folds in the digest of the page before it, so an id stands for *the entire
prefix up to here*, not for the four tokens it contains. Two requests sharing a system prompt
therefore produce identical ids for the shared pages and diverge the moment their token streams
do, which is how RadixAttention decides what can be reused. Content alone is deliberately not
enough: a page holding the same four tokens after a different history gets a different id,
because attention keys and values depend on the whole prefix.

The contrast with Rabin-Karp is the point of putting them side by side. **There is no verify
step here, and there cannot be one** - the tokens you would compare against were evicted long
ago, and at hit time the server holds nothing but the id. All the correctness rests on SHA-256
not colliding. Swap in a cheap hash and a collision does not make the server slow, it makes one
request read another request's KV cache. When choosing a hash, "can I check afterwards?" matters
more than "is it fast?".

## The LeetCode problems

LC 28 (`strStr`) is the algorithm itself. LC 459 and LC 214 are the interesting pair, because
neither is a search: both use the border quantity directly. For 459, if the longest border of
`s` has length `b` then sliding `s` right by `n - b` lands it on itself, so `n - b` is a period,
and `s` is a repetition iff that period divides `n` - `b > 0 and n % (n - b) == 0`, with the
`b > 0` guard stopping `abc` from counting as its own repetition. For 214, the longest
palindromic prefix of `s` is the longest prefix of `s` that is also a suffix of `reversed(s)`,
so building the failure function of `s + separator + reversed(s)` and reading the last entry
answers it in one pass; the separator must appear in neither half, or the border can straddle
the join and over-count. LC 1044 (longest duplicate substring) is the Rabin-Karp side: binary
search on the length, and a verified rolling hash at each candidate.

## The problems, stated in full

Restated in my own words - what is being asked, what goes in and comes out, one worked
example, and the constraints that actually change which algorithm is allowed.

### LeetCode 28 - Find the Index of the First Occurrence in a String

**The task.** Given two strings `haystack` and `needle`, return the index of the first
occurrence of `needle` inside `haystack`, or `-1` if it does not occur.

**Input / output.** Two strings in, one integer out.

**Example.** `haystack = "sadbutsad"`, `needle = "sad"` → `0` (it also occurs at `6`, but
we want the first). `haystack = "leetcode"`, `needle = "leeto"` → `-1`.

**Constraints.** `1 <= len(haystack), len(needle) <= 10^4`, lowercase English letters. The
naive double loop passes at this size, which is exactly why it is the right place to
*compare* it against KMP rather than to be forced into KMP.

[leetcode.com/problems/find-the-index-of-the-first-occurrence-in-a-string](https://leetcode.com/problems/find-the-index-of-the-first-occurrence-in-a-string/)

### LeetCode 214 - Shortest Palindrome

**The task.** You may only add characters **in front of** the given string. Return the
shortest palindrome you can produce that way.

**Input / output.** One string in, one string out.

**Example.** `s = "aacecaaa"` → `"aaacecaaa"` (one `a` added). `s = "abcd"` → `"dcbabcd"`
(three characters added).

**Constraints.** `0 <= len(s) <= 5 * 10^4`, lowercase English letters. The reformulation is
the trick: adding as little as possible in front means finding the *longest palindromic
prefix* of `s`, which is a prefix-function question in disguise once you run KMP over
`s + '#' + reversed(s)`.

[leetcode.com/problems/shortest-palindrome](https://leetcode.com/problems/shortest-palindrome/)

### LeetCode 459 - Repeated Substring Pattern

**The task.** Decide whether the string can be built by taking some proper substring and
concatenating two or more copies of it.

**Input / output.** One string in, a boolean out.

**Example.** `"abab"` → `True` (`"ab"` twice). `"aba"` → `False`.
`"abcabcabcabc"` → `True` (`"abc"` four times, or `"abcabc"` twice).

**Constraints.** `1 <= len(s) <= 10^4`, lowercase English letters. The one-line solution
`s in (s + s)[1:-1]` is cute, but the KMP reading is the one that generalises: the string
is periodic exactly when `n - failure[n-1]` divides `n` and is smaller than `n`.

[leetcode.com/problems/repeated-substring-pattern](https://leetcode.com/problems/repeated-substring-pattern/)

### LeetCode 1044 - Longest Duplicate Substring

**The task.** Find a longest substring that occurs **at least twice** in `s`; the two
occurrences are allowed to overlap. Return any one of them, or the empty string if no
substring repeats.

**Input / output.** One string in, one string out.

**Example.** `s = "banana"` → `"ana"` (it occurs at index `1` and index `3`, overlapping).
`s = "abcd"` → `""`.

**Constraints.** `2 <= len(s) <= 3 * 10^4`, lowercase English letters. "Does a duplicate of
length `L` exist?" is monotone in `L`, so the standard solution binary-searches `L` and
answers each question with Rabin-Karp rolling hashes - or, without any randomness, reads
the answer straight off a suffix array plus its LCP array.

[leetcode.com/problems/longest-duplicate-substring](https://leetcode.com/problems/longest-duplicate-substring/)

## Complexity

`n` = text length, `m` = pattern length, `sigma` = alphabet size, `k` = number of patterns.

| Operation | Time | Space |
|---|---|---|
| Naive search | O(n · m) worst case | O(1) |
| `build_failure` | O(m) | O(m) |
| KMP search | O(n + m) | O(m) |
| KMP as a DFA (`kmp_automaton`) | O(n) search, O(m · sigma) build | O(m · sigma) |
| Suffix/prefix overlap via borders | O(n + m) | O(n + m) |
| sglang's `check_match_stop_str_prefix` | O(m²) on a clipped tail | O(1) |
| Streaming stop match, per character | O(1) amortised | O(sum of stop lengths) |
| Rabin-Karp, expected | O(n + m) | O(1) |
| Rabin-Karp, worst case (all collisions) | O(n · m) | O(1) |
| Rabin-Karp, `k` patterns of equal length | O(n + k · m) expected | O(k) |
| LC 1044, longest duplicate substring | O(n log n) expected | O(n) |
| Chained page hashing, `p` pages | O(p · page_size) | O(p) |

## References
- [Knuth-Morris-Pratt algorithm - Wikipedia](https://en.wikipedia.org/wiki/Knuth%E2%80%93Morris%E2%80%93Pratt_algorithm)
- [Fast Pattern Matching in Strings (Knuth, Morris, Pratt, 1977)](https://doi.org/10.1137/0206024)
- [Rabin-Karp algorithm - Wikipedia](https://en.wikipedia.org/wiki/Rabin%E2%80%93Karp_algorithm)
- [Rolling hash - Wikipedia](https://en.wikipedia.org/wiki/Rolling_hash)
- [`sglang/srt/managers/schedule_batch.py`](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/managers/schedule_batch.py)
- [`sglang/srt/mem_cache/utils.py`](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/mem_cache/utils.py)
- [LeetCode 28 · Find the Index of the First Occurrence in a String](https://leetcode.com/problems/find-the-index-of-the-first-occurrence-in-a-string/)
- [LeetCode 459 · Repeated Substring Pattern](https://leetcode.com/problems/repeated-substring-pattern/)
- [LeetCode 214 · Shortest Palindrome](https://leetcode.com/problems/shortest-palindrome/)
- [LeetCode 1044 · Longest Duplicate Substring](https://leetcode.com/problems/longest-duplicate-substring/)
