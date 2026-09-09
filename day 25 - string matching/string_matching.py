"""Day 25 - string matching: KMP and Rabin-Karp.

Run me:  python3 string_matching.py

The two classic linear-time matchers, and the two places an inference server
actually needs them:

  * a stop string arriving in pieces, where the pattern may straddle a chunk
    boundary - which is exactly the question KMP's failure function answers;
  * a hash over a sliding window, and what happens when you skip the
    verification step that Rabin-Karp is defined with.

Everything here is standard library only.
"""

from __future__ import annotations

import hashlib
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple


# ===========================================================================
# 1. The naive matcher, instrumented
# ===========================================================================

def naive_search(text: str, pat: str) -> Tuple[List[int], int]:
    """Every alignment, every character.  Returns (hits, comparisons).

    The cost is not the number of alignments (n - m + 1) but the number of
    *character comparisons*, and those blow up exactly when the pattern keeps
    almost matching: on 'aaaa...a' vs 'aaab' every alignment gets to the last
    character before it fails, which is O(n * m).
    """
    n, m = len(text), len(pat)
    hits, cmps = [], 0
    if m == 0:
        return list(range(n + 1)), 0
    for i in range(n - m + 1):
        j = 0
        while j < m:
            cmps += 1
            if text[i + j] != pat[j]:
                break
            j += 1
        if j == m:
            hits.append(i)
    return hits, cmps


# ===========================================================================
# 2. KMP - the failure function is a table of borders
# ===========================================================================

def build_failure(pat: str) -> List[int]:
    """fail[i] = length of the longest proper border of pat[:i + 1].

    A *border* is a string that is both a proper prefix and a proper suffix.
    'ababa' has border 'aba' (length 3); 'abcd' has none (length 0).

    Why that is the right table: when pat[:j] matched and pat[j] did not, the
    text ends with pat[:j].  Any shorter alignment that could still work must
    have its prefix equal to a suffix of pat[:j] - that is the definition of a
    border.  So the next state to try is the longest border, and everything
    between the current alignment and that one is provably a mismatch.

    The build is the same idea applied to the pattern against itself, which is
    why it is O(m): `k` only ever increases once per character.
    """
    m = len(pat)
    fail = [0] * m
    k = 0                       # length of the current border
    for i in range(1, m):
        while k > 0 and pat[i] != pat[k]:
            k = fail[k - 1]     # fall back to the next shorter border
        if pat[i] == pat[k]:
            k += 1
        fail[i] = k
    return fail


def kmp_search(text: str, pat: str) -> Tuple[List[int], int]:
    """Returns (hits, comparisons).  `i` never moves backwards."""
    if not pat:
        return list(range(len(text) + 1)), 0
    fail = build_failure(pat)
    hits, cmps, j = [], 0, 0
    for i, ch in enumerate(text):
        while j > 0 and ch != pat[j]:
            cmps += 1
            j = fail[j - 1]
        cmps += 1
        if ch == pat[j]:
            j += 1
        if j == len(pat):
            hits.append(i - j + 1)
            j = fail[j - 1]     # keep going: overlapping matches count
    return hits, cmps


def kmp_automaton(pat: str, alphabet: Sequence[str]) -> List[Dict[str, int]]:
    """The same thing as an explicit DFA: delta[state][char] -> state.

    KMP *is* an automaton whose states are 'how many characters of the pattern
    I have matched'.  Precomputing every transition removes the fallback loop
    entirely - one table lookup per input character, no branches, no backtrack.
    That is the form the matcher takes when it has to run on a stream.
    """
    m = len(pat)
    fail = build_failure(pat)
    delta: List[Dict[str, int]] = []
    for state in range(m + 1):
        row = {}
        for ch in alphabet:
            if state < m and ch == pat[state]:
                row[ch] = state + 1
            elif state == 0:
                row[ch] = 0
            else:
                # Same fallback as the loop, resolved once at build time.
                row[ch] = delta[fail[state - 1]][ch]
        delta.append(row)
    return delta


def border_of(s: str) -> int:
    """Longest proper border of the whole string - one table entry."""
    return build_failure(s)[-1] if s else 0


# ===========================================================================
# 3. A stop string arriving in pieces
# ===========================================================================
#
# An inference server streams tokens out as they are produced, but it must not
# emit anything that might turn out to be part of a stop string.  So on every
# step it asks two questions:
#
#   (a) does the output contain the stop string?           -> finish now
#   (b) does a *suffix* of the output match a *prefix* of   -> hold the tail
#       the stop string?                                       back, do not
#                                                               stream it yet
#
# (b) is the border question again, across two different strings.

def suffix_prefix_overlap(tail: str, pat: str) -> int:
    """Longest k such that tail[-k:] == pat[:k].  O(len(tail) + len(pat)).

    The trick: borders of `pat + sep + tail` where `sep` occurs in neither.
    The final failure-function entry cannot reach across the separator, so it
    is exactly the overlap we want.
    """
    if not tail or not pat:
        return 0
    sep = "\x00"
    return build_failure(pat + sep + tail)[-1]


def overlap_quadratic(tail: str, pat: str) -> int:
    """What sglang actually does (schedule_batch.py, check_match_stop_str_prefix):
    try every k from 1 upwards and compare the slices.

    O(len(tail) * len(pat)), and completely fine - because the tail it looks at
    is clipped to `stop_str_max_len + 1` characters first.  Bounding the input
    is a legitimate alternative to a better algorithm, as long as you can prove
    the bound.
    """
    best = 0
    for k in range(1, min(len(tail), len(pat)) + 1):
        if tail[-k:] == pat[:k]:
            best = k
    return best


class StreamStopMatcher:
    """Incremental multi-pattern stop-string matcher, one KMP state per pattern.

    Feed it whatever arrives.  It answers, in O(1) amortised per character and
    with no memory of the text at all:
      * did a stop string just complete (and which one),
      * how many trailing characters must be withheld because they might be
        the start of one.
    """

    def __init__(self, stops: Sequence[str]) -> None:
        self.stops = [s for s in stops if s]
        self.fails = [build_failure(s) for s in self.stops]
        self.state = [0] * len(self.stops)     # matched length, per pattern
        self.buf = ""                          # withheld tail, not emitted yet
        self.matched: Optional[str] = None

    def feed(self, chunk: str) -> Tuple[str, Optional[str]]:
        """Returns (text safe to emit now, stop string that completed or None).

        The state survives the call, which is the entire point: a stop string
        split across two chunks is caught without ever looking back at text
        that has already been streamed out.
        """
        self.buf += chunk
        for i, ch in enumerate(chunk):
            for p, pat in enumerate(self.stops):
                j = self.state[p]
                while j > 0 and ch != pat[j]:
                    j = self.fails[p][j - 1]
                if ch == pat[j]:
                    j += 1
                self.state[p] = j
                if j == len(pat):
                    self.matched = pat
                    end = len(self.buf) - (len(chunk) - i - 1)
                    out = self.buf[: end - len(pat)]
                    self.buf = ""
                    return out, pat
        hold = max(self.state) if self.state else 0
        cut = len(self.buf) - hold
        out, self.buf = self.buf[:cut], self.buf[cut:]
        return out, None

    def pending(self) -> int:
        """Characters currently held back because they may start a stop string."""
        return len(self.buf)


# ---------------------------------------------------------------------------
# 3b. The pattern is in characters; the stream arrives in tokens
# ---------------------------------------------------------------------------

class ToyTokenizer:
    """Just enough of a tokenizer to show the boundary problem: token text is
    not aligned with anything the user typed."""

    def __init__(self, vocab: Dict[int, str]) -> None:
        self.vocab = vocab

    def decode(self, ids: Sequence[int]) -> str:
        return "".join(self.vocab[i] for i in ids)


def stop_match_tail_len(stop_max_len: int, new_accepted_len: int, n_out: int) -> int:
    """sglang's window (schedule_batch.py, _stop_match_tail_len).

    `stop_max_len + 1` characters would be enough if exactly one token were
    appended per step.  Speculative decoding accepts several at once, and a
    stop string that finished *inside* that batch would fall out of the window
    - hence the `new_accepted_len - 1` extension.  The bug this guards against
    only appears when speculation is on, which is why it was found late.
    """
    return min(stop_max_len + 1 + max(new_accepted_len - 1, 0), n_out)


def locate_stop_token_count(tok: ToyTokenizer, out_ids: Sequence[int],
                            stop: str, new_accepted_len: int,
                            stop_max_len: int) -> Tuple[int, str]:
    """Which token prefix first contains the stop string, and the text to keep.

    Mirrors sglang's _locate_str_stop_finished_len: re-decode growing prefixes
    of the tail window until the stop string appears.  It has to work in token
    counts, because that is what the KV cache and the output length are
    measured in - but the *match* only exists in character space, so the two
    have to be reconciled by decoding.
    """
    tail_len = stop_match_tail_len(stop_max_len, new_accepted_len, len(out_ids))
    start = len(out_ids) - tail_len
    window = list(out_ids[start:])
    for count in range(max(1, len(window) - new_accepted_len + 1), len(window) + 1):
        text = tok.decode(window[:count])
        if stop in text:
            full = tok.decode(out_ids[: start + count])
            return start + count, full[: full.index(stop)]
    return len(out_ids), tok.decode(out_ids)


# ===========================================================================
# 4. Rabin-Karp - compare fingerprints, then verify
# ===========================================================================

def rabin_karp(text: str, pat: str, base: int = 256,
               mod: int = 1_000_003) -> Tuple[List[int], int, int]:
    """Returns (hits, verifications, false_positives).

    One multiply-add per position keeps a fingerprint of the current window.
    Equal fingerprints mean *maybe* equal strings, so every hit is verified
    with a real comparison.  Skipping the verify step turns a probabilistic
    filter into a wrong answer, and the failure is silent.
    """
    n, m = len(text), len(pat)
    if m == 0 or m > n:
        return [], 0, 0
    high = pow(base, m - 1, mod)
    hp = ht = 0
    for i in range(m):
        hp = (hp * base + ord(pat[i])) % mod
        ht = (ht * base + ord(text[i])) % mod
    hits, verifications, false_pos = [], 0, 0
    for i in range(n - m + 1):
        if ht == hp:
            verifications += 1
            if text[i:i + m] == pat:
                hits.append(i)
            else:
                false_pos += 1          # a collision, caught by the verify
        if i < n - m:
            # roll: drop the leading character, shift, add the new one
            ht = ((ht - ord(text[i]) * high) * base + ord(text[i + m])) % mod
    return hits, verifications, false_pos


def rabin_karp_multi(text: str, pats: Iterable[str],
                     base: int = 256, mod: int = (1 << 61) - 1) -> Dict[str, List[int]]:
    """Many patterns of the same length in one pass - the case where Rabin-Karp
    beats KMP outright, because the window hash is shared by every pattern and
    the lookup is one set membership test."""
    pats = list(pats)
    assert pats and len({len(p) for p in pats}) == 1, "same length required"
    m = len(pats[0])
    table: Dict[int, List[str]] = {}
    for p in pats:
        h = 0
        for ch in p:
            h = (h * base + ord(ch)) % mod
        table.setdefault(h, []).append(p)
    out: Dict[str, List[int]] = {p: [] for p in pats}
    if m > len(text):
        return out
    high = pow(base, m - 1, mod)
    h = 0
    for i in range(m):
        h = (h * base + ord(text[i])) % mod
    for i in range(len(text) - m + 1):
        for cand in table.get(h, ()):           # still verify
            if text[i:i + m] == cand:
                out[cand].append(i)
        if i < len(text) - m:
            h = ((h - ord(text[i]) * high) * base + ord(text[i + m])) % mod
    return out


def find_collision(mod: int, base: int = 256, length: int = 3,
                   alphabet: str = "abcdefgh") -> Tuple[str, str]:
    """Two different strings with the same fingerprint, by pigeonhole.

    With a small modulus this takes milliseconds, which is the point: a
    fingerprint is not an identity, and any code that treats it as one is one
    unlucky input away from being wrong.
    """
    seen: Dict[int, str] = {}
    import itertools
    for tup in itertools.product(alphabet, repeat=length):
        s = "".join(tup)
        h = 0
        for ch in s:
            h = (h * base + ord(ch)) % mod
        if h in seen and seen[h] != s:
            return seen[h], s
        seen[h] = s
    raise RuntimeError("no collision found")


def page_hashes(tokens: Sequence[int], page_size: int,
                prior: Optional[str] = None) -> List[str]:
    """Position-aware block ids, the way sglang labels KV pages.

    Each page's id folds in the *previous* page's id, so a page is identified
    by its whole history, not just its contents.  Two requests get the same id
    for page k only if they agree on every token up to page k - which is
    exactly the condition for the cached KV to be reusable.

    Note what is missing compared with Rabin-Karp: there is no verify step.
    The id is used to fetch someone else's KV blocks, possibly across a
    network, and the tokens themselves are not there to compare against.  That
    is why this hash is SHA-256 and not a rolling multiply-add.
    """
    out: List[str] = []
    n_pages = len(tokens) // page_size
    for p in range(n_pages):
        page = tokens[p * page_size:(p + 1) * page_size]
        h = hashlib.sha256()
        if prior is not None:
            h.update(bytes.fromhex(prior))
        h.update(",".join(map(str, page)).encode())
        prior = h.hexdigest()
        out.append(prior)
    return out


# ===========================================================================
# 5. LeetCode
# ===========================================================================

def str_str(haystack: str, needle: str) -> int:
    """LC 28 - Find the Index of the First Occurrence in a String."""
    if not needle:
        return 0
    hits, _ = kmp_search(haystack, needle)
    return hits[0] if hits else -1


def repeated_substring_pattern(s: str) -> bool:
    """LC 459 - one line, once you have the failure function.

    `n - border` is the smallest period of the string.  The string is a whole
    number of copies of that period exactly when the period divides n, and it
    is a *proper* repeat when the border is non-empty.
    """
    n = len(s)
    b = build_failure(s)[-1] if s else 0
    return b > 0 and n % (n - b) == 0


def shortest_palindrome(s: str) -> str:
    """LC 214 - the same border trick as the stop-string overlap.

    The answer needs the longest palindromic *prefix*, which is the longest
    prefix of `s` that is also a suffix of `reversed(s)` - a border of
    `s + sep + reversed(s)`.
    """
    if not s:
        return s
    rev = s[::-1]
    k = build_failure(s + "\x00" + rev)[-1]
    return rev[: len(s) - k] + s


def longest_duplicate_substring(s: str) -> str:
    """LC 1044 - Rabin-Karp inside a binary search on the answer.

    Monotone: if some substring of length L appears twice, so does one of
    length L - 1 (take a prefix of it).  So binary search L, and at each L ask
    "is there a repeated window" with one rolling-hash pass and a dict of
    candidate positions - verified, because a collision would invent a match.
    """
    n = len(s)
    base, mod = 256, (1 << 61) - 1

    def dup_of_length(m: int) -> Optional[str]:
        if m == 0:
            return ""
        high = pow(base, m - 1, mod)
        h = 0
        for i in range(m):
            h = (h * base + ord(s[i])) % mod
        seen: Dict[int, List[int]] = {h: [0]}
        for i in range(1, n - m + 1):
            h = ((h - ord(s[i - 1]) * high) * base + ord(s[i + m - 1])) % mod
            for j in seen.get(h, ()):
                if s[j:j + m] == s[i:i + m]:       # verify
                    return s[i:i + m]
            seen.setdefault(h, []).append(i)
        return None

    lo, hi, best = 1, n - 1, ""                     # lower_bound on "no dup"
    while lo <= hi:
        mid = (lo + hi) // 2
        got = dup_of_length(mid)
        if got:
            best, lo = got, mid + 1
        else:
            hi = mid - 1
    return best


# ===========================================================================
# demo
# ===========================================================================

def head(title: str) -> None:
    print()
    print("=" * 68)
    print(title)
    print("=" * 68)


def main() -> None:
    head("1. Naive vs KMP - where the comparisons go")
    text = "a" * 40 + "aaab"
    pat = "aaab"
    n_hits, n_cmp = naive_search(text, pat)
    k_hits, k_cmp = kmp_search(text, pat)
    print("text = 'a'*40 + 'aaab'   pattern = 'aaab'")
    print("naive: %3d comparisons, hits %s" % (n_cmp, n_hits))
    print("KMP:   %3d comparisons, hits %s" % (k_cmp, k_hits))
    print("same answer: %s" % (n_hits == k_hits))
    prose = ("the quick brown fox jumps over the lazy dog, "
             "and the dog barks at the fox")
    pn, cn = naive_search(prose, "the fox")
    pk, ck = kmp_search(prose, "the fox")
    print("on ordinary prose the gap almost vanishes: naive %d vs KMP %d comparisons"
          % (cn, ck))
    assert pn == pk

    head("2. The failure function is a table of borders")
    p = "ababaca"
    fail = build_failure(p)
    print("pattern:  " + "  ".join(p))
    print("fail:     " + "  ".join(str(x) for x in fail))
    for i in range(len(p)):
        b = fail[i]
        if b:
            print("  pat[:%d] = %-8s border %-4s (prefix == suffix)"
                  % (i + 1, repr(p[:i + 1]), repr(p[:b])))
    delta = kmp_automaton("aba", "ab")
    print("the same matcher as a DFA over {a, b}, delta[state][char]:")
    for st, row in enumerate(delta):
        print("  state %d: %s" % (st, {k: row[k] for k in "ab"}))
    print("no fallback loop left - one lookup per character, which is the form")
    print("you want when the text is a stream you cannot rewind")

    head("3. A stop string that arrives in pieces")
    stops = ["</s>", "\n\nHuman:"]
    m = StreamStopMatcher(stops)
    chunks = ["The answer is 4", "2.<", "/s", "> and then some more"]
    for c in chunks:
        emit, hit = m.feed(c)
        print("chunk %-22s -> emit %-20s held %d  %s"
              % (repr(c), repr(emit), m.pending(), ("STOP " + repr(hit)) if hit else ""))
        if hit:
            break
    print("the '<' at the end of chunk 2 was withheld, not streamed - because at")
    print("that moment it was a 1-character match of '</s>' and nobody knew yet")

    print()
    tail = "The answer is 42.<"
    print("overlap(tail=%s, stop='</s>')" % repr(tail))
    print("  KMP border trick : %d" % suffix_prefix_overlap(tail, "</s>"))
    print("  sglang's loop    : %d  (O(len*len), on a window clipped to"
          " stop_str_max_len + 1)" % overlap_quadratic(tail, "</s>"))
    assert suffix_prefix_overlap(tail, "</s>") == overlap_quadratic(tail, "</s>")

    head("4. Tokens are not characters")
    vocab = {1: "The", 2: " answer", 3: " is", 4: " 42]", 5: "]", 6: " so",
             7: " far", 8: " and", 9: " more"}
    tok = ToyTokenizer(vocab)
    out_ids = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    stop = "]]"
    print("decoded output: %s" % repr(tok.decode(out_ids)))
    print("stop string:    %s  (it starts inside token 4 and ends in token 5)"
          % repr(stop))

    def first_detection(step: int, window_fn) -> Optional[int]:
        """Walk the stream `step` tokens at a time and report the first step at
        which the stop string is visible inside the tail window."""
        t = 0
        while t < len(out_ids):
            t = min(t + step, len(out_ids))
            w = window_fn(step, t)
            if stop in tok.decode(out_ids[t - w:t]):
                return t
        return None

    naive_window = lambda step, t: min(len(stop) + 1, t)
    sglang_window = lambda step, t: stop_match_tail_len(len(stop), step, t)
    for step in (1, 4):
        got_naive = first_detection(step, naive_window)
        got_sglang = first_detection(step, sglang_window)
        fmt = lambda v: "never" if v is None else "token %d" % v
        print("  %d token(s) per step: window = stop_max_len+1 -> %-8s |"
              " window extended by accepted-1 -> %s"
              % (step, fmt(got_naive), fmt(got_sglang)))
    count, keep = locate_stop_token_count(tok, out_ids[:8], stop, 4, len(stop))
    print("the step that catches it accepted tokens 5-8 at once; locating the stop")
    print("inside that batch: finish after %d tokens, keep %s" % (count, repr(keep)))
    assert (count, keep) == (5, "The answer is 42")
    print("the window is measured in characters and then used as a token count,")
    print("which is safe only because a token decodes to at least one character;")
    print("the match itself lives in character space, so the two are reconciled")
    print("by decoding growing prefixes until the stop string shows up")
    assert first_detection(1, naive_window) == 5
    assert first_detection(4, naive_window) is None      # the batch jumped over it
    assert first_detection(4, sglang_window) == 8

    head("5. Rabin-Karp - fingerprints, and why you still compare")
    dna = "ACGTACGTGACGTACGTTACGTACGTG"
    hits, ver, fp = rabin_karp(dna, "ACGTACGTG")
    print("text %s" % dna)
    print("pattern 'ACGTACGTG' -> hits %s, %d verifications, %d false positives"
          % (hits, ver, fp))
    a, b = find_collision(mod=101, length=3)
    print("with a 101-element modulus, %s and %s share a fingerprint"
          % (repr(a), repr(b)))
    hits2, ver2, fp2 = rabin_karp("x" + a + "y", b, mod=101)
    print("searching for %s in %s: %d verification(s), %d false positive(s),"
          " reported hits %s"
          % (repr(b), repr("x" + a + "y"), ver2, fp2, hits2))
    print("without the verify line that would have been a match")

    pats = ["ACGT", "GACG", "TTAC"]
    found = rabin_karp_multi(dna, pats)
    print("three patterns of the same length, one pass: %s" % found)

    head("6. The other kind of hash: position-aware KV block ids")
    system = list(range(1000, 1008))
    a_ids = system + [1, 2, 3, 4]
    b_ids = system + [1, 2, 9, 9]
    c_ids = [7, 7, 7, 7] + [1, 2, 3, 4]
    ha, hb, hc = (page_hashes(x, 4) for x in (a_ids, b_ids, c_ids))
    print("req A pages: %s" % [h[:8] for h in ha])
    print("req B pages: %s" % [h[:8] for h in hb])
    print("req C pages: %s" % [h[:8] for h in hc])
    print("A and B agree on the first %d page(s) - exactly their shared prefix"
          % sum(1 for x, y in zip(ha, hb) if x == y))
    print("C's last page holds the same four tokens as A's last page, and gets a")
    print("different id, because the id folds in every page before it")
    assert ha[:2] == hb[:2] and ha[2] != hb[2]
    assert ha[-1] != hc[-1]

    head("7. LeetCode")
    print("28  strStr('sadbutsad', 'sad')        = %d" % str_str("sadbutsad", "sad"))
    print("28  strStr('leetcode', 'leeto')       = %d" % str_str("leetcode", "leeto"))
    print("459 'abab'    -> %s   (period %d)"
          % (repeated_substring_pattern("abab"), 4 - build_failure("abab")[-1]))
    print("459 'aba'     -> %s" % repeated_substring_pattern("aba"))
    print("459 'abcabcabcabc' -> %s" % repeated_substring_pattern("abcabcabcabc"))
    print("214 shortest_palindrome('aacecaaa') = %s"
          % repr(shortest_palindrome("aacecaaa")))
    print("1044 longest_duplicate_substring('banana') = %s"
          % repr(longest_duplicate_substring("banana")))
    print("1044 longest_duplicate_substring('abcd')   = %s"
          % repr(longest_duplicate_substring("abcd")))
    print("     (binary search on the length, Rabin-Karp for each probe -")
    print("      the answer space is monotone, so the same loop as any other")
    print("      'largest L that still works' search)")

    # ---- asserts -------------------------------------------------------
    import random
    rng = random.Random(25)
    for _ in range(300):
        t = "".join(rng.choice("abc") for _ in range(rng.randint(0, 30)))
        q = "".join(rng.choice("abc") for _ in range(rng.randint(1, 5)))
        exp = [i for i in range(len(t) - len(q) + 1) if t[i:i + len(q)] == q]
        assert kmp_search(t, q)[0] == exp
        assert naive_search(t, q)[0] == exp
        assert rabin_karp(t, q)[0] == exp
        assert suffix_prefix_overlap(t, q) == overlap_quadratic(t, q)
        assert str_str(t, q) == (exp[0] if exp else -1)
    for _ in range(200):
        t = "".join(rng.choice("ab") for _ in range(rng.randint(2, 14)))
        exp = max((t[i:j] for i in range(len(t)) for j in range(i + 1, len(t) + 1)
                   if t.count(t[i:j]) > 1 or t.find(t[i:j], i + 1) >= 0), key=len,
                  default="")
        assert len(longest_duplicate_substring(t)) == len(exp), (t, exp)
    for s in ["abab", "aba", "abcabcabcabc", "a", "aa", "abaababaab"]:
        n = len(s)
        brute = any(s == s[:k] * (n // k) for k in range(1, n) if n % k == 0)
        assert repeated_substring_pattern(s) == brute, s
    print()
    print("all assertions passed")


if __name__ == "__main__":
    main()
