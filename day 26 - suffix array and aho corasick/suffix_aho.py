"""Day 26 - suffix array + LCP, and Aho-Corasick.

Run me:  python3 suffix_aho.py

Day 25 answered "where does this one pattern occur?".  Two questions it could
not answer:

  * "what does this text repeat?" - nobody hands you the pattern, so there is
    no failure function to build.  Sort every suffix once and the repeats fall
    out of the array of adjacent common prefixes.  This is how large training
    corpora are deduplicated.
  * "which of these ten thousand patterns just occurred?" - running KMP k
    times costs k passes over the text.  Aho-Corasick is KMP's failure
    function generalised to a trie, and costs one.

Standard library only.
"""

from __future__ import annotations

from collections import deque
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


# ===========================================================================
# 1. The suffix array
# ===========================================================================

def suffix_array_naive(s: str) -> List[int]:
    """Sort the n suffixes and return their starting positions.

    Correct, and the definition itself - but every comparison can touch O(n)
    characters and Python's sort makes O(n log n) of them, so this is
    O(n^2 log n) and it materialises n substrings on the way.  Use it as the
    reference implementation to check the fast one against.
    """
    return sorted(range(len(s)), key=lambda i: s[i:])


def suffix_array(s: str) -> List[int]:
    """Prefix doubling: sort by the first 1, 2, 4, 8 ... characters.

    The trick is that after sorting by the first k characters you do not need
    the characters any more - you need the *rank* of each suffix among them.
    Then the first 2k characters of suffix i are the pair
    (rank_k[i], rank_k[i + k]), and a pair of small integers compares in O(1)
    no matter how long 2k is.  log2(n) rounds is enough because k doubles.

    O(n log^2 n) with a comparison sort of tuples; replacing the sort with a
    two-pass counting sort (day 22) gets it to O(n log n).
    """
    n = len(s)
    if n == 0:
        return []
    rank = [ord(c) for c in s]
    sa = list(range(n))
    k = 1
    while True:
        key = lambda i: (rank[i], rank[i + k] if i + k < n else -1)
        sa.sort(key=key)
        new = [0] * n
        for a, b in zip(sa, sa[1:]):
            new[b] = new[a] + (1 if key(a) < key(b) else 0)
        rank = new
        if rank[sa[-1]] == n - 1:      # all ranks distinct - order is final
            return sa
        k *= 2


def rank_rounds(s: str) -> List[Tuple[int, List[int]]]:
    """The intermediate rank arrays, for the diagram and the notebook."""
    n = len(s)
    rank = [ord(c) for c in s]
    # normalise round 0 to small integers so it prints nicely
    order = {c: i for i, c in enumerate(sorted(set(s)))}
    rank = [order[c] for c in s]
    out = [(1, rank[:])]
    sa, k = list(range(n)), 1
    while max(rank) < n - 1:
        key = lambda i: (rank[i], rank[i + k] if i + k < n else -1)
        sa.sort(key=key)
        new = [0] * n
        for a, b in zip(sa, sa[1:]):
            new[b] = new[a] + (1 if key(a) < key(b) else 0)
        rank = new
        k *= 2
        out.append((k, rank[:]))
    return out


# ===========================================================================
# 2. Kasai's LCP array - the whole point of the suffix array
# ===========================================================================

def lcp_kasai(s: str, sa: Sequence[int]) -> List[int]:
    """lcp[i] = common prefix length of sa[i - 1] and sa[i]; lcp[0] = 0.

    Computing each entry from scratch is O(n) per pair.  Kasai's observation:
    walk the suffixes in *text* order, not array order.  If suffix i shares h
    characters with its neighbour, then suffix i + 1 - which is suffix i minus
    its first character - shares at least h - 1 with *its* neighbour.  So h
    drops by at most one per step and rises by at most n in total: O(n).
    """
    n = len(s)
    rank = [0] * n
    for i, p in enumerate(sa):
        rank[p] = i
    lcp = [0] * n
    h = 0
    for i in range(n):
        if rank[i] == 0:
            h = 0
            continue
        j = sa[rank[i] - 1]
        while i + h < n and j + h < n and s[i + h] == s[j + h]:
            h += 1
        lcp[rank[i]] = h
        if h:
            h -= 1
    return lcp


def lcp_naive(s: str, sa: Sequence[int]) -> List[int]:
    """The O(n^2) reference: compare each adjacent pair character by character."""
    if not sa:
        return []
    out = [0]
    for a, b in zip(sa, sa[1:]):
        h = 0
        while a + h < len(s) and b + h < len(s) and s[a + h] == s[b + h]:
            h += 1
        out.append(h)
    return out


# ===========================================================================
# 3. Searching a suffix array: two binary searches (day 23)
# ===========================================================================

def sa_range(s: str, sa: Sequence[int], pat: str) -> Tuple[int, int, int]:
    """Half-open range of sa[] whose suffixes start with pat, and #probes.

    Every occurrence of pat is the start of a suffix beginning with pat, and
    the suffix array is sorted, so those suffixes are *contiguous*.  Finding
    them is the lower-bound / upper-bound pair from day 23, with the twist
    that the comparison is a bounded prefix comparison, not '<' on integers:
    each probe costs O(m), so the whole search is O(m log n).

    Note what this buys over KMP: the text is preprocessed once and then any
    pattern is answered in log time.  KMP preprocesses the *pattern* and has
    to re-read the whole text for every query.
    """
    m, probes = len(pat), 0
    lo, hi = 0, len(sa)
    while lo < hi:                              # lower bound
        mid = (lo + hi) // 2
        probes += 1
        if s[sa[mid]:sa[mid] + m] < pat:
            lo = mid + 1
        else:
            hi = mid
    start = lo
    hi = len(sa)
    while lo < hi:                              # upper bound
        mid = (lo + hi) // 2
        probes += 1
        if s[sa[mid]:sa[mid] + m] <= pat:
            lo = mid + 1
        else:
            hi = mid
    return start, lo, probes


def sa_count(s: str, sa: Sequence[int], pat: str) -> int:
    a, b, _ = sa_range(s, sa, pat)
    return b - a


def sa_occurrences(s: str, sa: Sequence[int], pat: str) -> List[int]:
    a, b, _ = sa_range(s, sa, pat)
    return sorted(sa[a:b])


# ===========================================================================
# 4. What the LCP array is for
# ===========================================================================

def longest_repeated_substring(s: str) -> str:
    """The largest LCP entry is the longest substring that occurs twice.

    Why: any substring occurring twice is a common prefix of two suffixes, and
    in a sorted array the longest common prefix of *any* two entries is the
    minimum of the lcp values between them - so it is maximised by an adjacent
    pair.  One scan of lcp finds it.
    """
    sa = suffix_array(s)
    lcp = lcp_kasai(s, sa)
    if not lcp:
        return ""
    k = max(range(len(lcp)), key=lambda i: lcp[i])
    return s[sa[k]:sa[k] + lcp[k]]


def longest_common_substring(a: str, b: str, sep: str = "\x01") -> str:
    """Concatenate with a separator that appears in neither string.

    In the combined suffix array, an adjacent pair whose two suffixes come
    from *different* sides gives a substring common to both; the largest such
    lcp is the answer.  The separator must be unique, otherwise a "common"
    substring could run across the join and be reported for text that neither
    input contains.
    """
    assert sep not in a and sep not in b
    t = a + sep + b
    sa = suffix_array(t)
    lcp = lcp_kasai(t, sa)
    cut = len(a)
    best, best_at = 0, 0
    for i in range(1, len(sa)):
        left, right = sa[i - 1], sa[i]
        if (left < cut) == (right < cut):
            continue                            # same side - not evidence
        if lcp[i] > best:
            best, best_at = lcp[i], sa[i]
    return t[best_at:best_at + best]


def duplicate_spans(text: str, min_len: int) -> List[Tuple[int, str]]:
    """Maximal repeated substrings of length >= min_len, as (count, substring).

    A run of lcp values >= min_len is a group of suffixes sharing that much
    prefix, and the block's length is the smallest lcp inside the run.  One
    filter matters: report the block only if it is **left-diverse**, i.e. its
    occurrences are not all preceded by the same character.  Without it a
    single repeated paragraph is reported once per starting offset - the
    paragraph, the paragraph minus its first character, minus two, and so on -
    because every one of those shifted copies genuinely occurs twice.  They
    are the same duplicate seen from different places, and left-diversity is
    what says "this is where the repeat actually begins".
    """
    sa = suffix_array(text)
    lcp = lcp_kasai(text, sa)
    out: List[Tuple[int, str]] = []
    i, n = 1, len(sa)
    while i < n:
        if lcp[i] < min_len:
            i += 1
            continue
        j, run = i, lcp[i]
        while j < n and lcp[j] >= min_len:
            run = min(run, lcp[j])
            j += 1
        group = sa[i - 1:j]
        left = {text[p - 1] if p > 0 else None for p in group}
        if len(left) > 1:                       # left-diverse: a real block
            out.append((len(group), text[group[0]:group[0] + run]))
        i = j
    return sorted(out, key=lambda t: -len(t[1]))


def dedup_report(docs: Sequence[str], min_len: int, sep: str = "\x00") -> Dict[str, object]:
    """Bytes a corpus would save by dropping repeated blocks of >= min_len."""
    text = sep.join(docs)
    spans = [(c, sub) for c, sub in duplicate_spans(text, min_len) if sep not in sub]
    saved = sum((c - 1) * len(sub) for c, sub in spans)
    return {
        "chars": len(text),
        "spans": spans,
        "saved": saved,
        "pct": 100.0 * saved / max(len(text), 1),
    }

# ===========================================================================
# 5. Aho-Corasick - KMP's failure function, on a trie
# ===========================================================================

class AhoCorasick:
    """Multi-pattern matcher: one pass over the text, all patterns at once.

    The trie (day 24) says where you are if everything matched.  The missing
    piece is the same one KMP needed: when the next character does not fit,
    where do you fall back to *without rewinding the text*?  KMP's answer was
    'the longest border of what matched'.  Here the state is a trie node
    spelling some string u, and the fail link points at the node spelling the
    longest proper suffix of u that is still a node in the trie.  Identical
    idea, one dimension wider - and for a single pattern the fail links are
    literally KMP's failure function (see fail_depths()).

    out[v] must also collect the patterns ending at every node reachable by
    following fail links from v, because a shorter pattern can end inside a
    longer one ('he' ends inside 'she') and the state only remembers the
    longest.
    """

    def __init__(self, words: Iterable[str] = ()):
        self.goto: List[Dict[str, int]] = [{}]
        self.fail: List[int] = [0]
        self.out: List[List[str]] = [[]]
        self.depth: List[int] = [0]
        self.parent_char: List[str] = [""]
        self.built = False
        self.transitions = 0
        for w in words:
            self.add(w)
        if words:
            self.build()

    # -- construction -------------------------------------------------------
    def add(self, word: str) -> None:
        v = 0
        for ch in word:
            if ch not in self.goto[v]:
                self.goto[v][ch] = len(self.goto)
                self.goto.append({})
                self.fail.append(0)
                self.out.append([])
                self.depth.append(self.depth[v] + 1)
                self.parent_char.append(ch)
            v = self.goto[v][ch]
        self.out[v].append(word)
        self.built = False

    def build(self) -> "AhoCorasick":
        """BFS by depth, so a node's fail target is finished before it is used.

        Depth 1 fails to the root: a single character has no proper suffix
        left.  Deeper, the fail target of child (v, ch) is step(fail[v], ch) -
        follow the parent's fallback and try the same character, exactly the
        'j = fail[j - 1]' loop from day 25 with the loop hidden inside step().
        """
        q = deque()
        for ch, u in self.goto[0].items():
            self.fail[u] = 0
            q.append(u)
        while q:
            v = q.popleft()
            self.out[v] = self.out[v] + self.out[self.fail[v]]
            for ch, u in self.goto[v].items():
                f = self.fail[v]
                while f and ch not in self.goto[f]:
                    f = self.fail[f]
                self.fail[u] = self.goto[f].get(ch, 0)
                q.append(u)
        self.built = True
        return self

    # -- matching -----------------------------------------------------------
    def step(self, state: int, ch: str) -> int:
        """One character: follow fail links until the character fits."""
        while state and ch not in self.goto[state]:
            state = self.fail[state]
            self.transitions += 1
        self.transitions += 1
        return self.goto[state].get(ch, 0)

    def find_all(self, text: str) -> List[Tuple[int, str]]:
        """All (end_index, pattern) hits, in one left-to-right pass."""
        if not self.built:
            self.build()
        state, hits = 0, []
        for i, ch in enumerate(text):
            state = self.step(state, ch)
            for w in self.out[state]:
                hits.append((i, w))
        return hits

    def fail_depths(self) -> List[int]:
        """For a single-pattern automaton: KMP's failure function, verbatim."""
        if not self.built:
            self.build()
        v, out = 0, []
        while self.goto[v]:
            v = next(iter(self.goto[v].values()))
            out.append(self.depth[self.fail[v]])
        return out

    def n_nodes(self) -> int:
        return len(self.goto)


def kmp_failure(pat: str) -> List[int]:
    """Day 25's build_failure, repeated here so the file stands alone."""
    fail = [0] * len(pat)
    k = 0
    for i in range(1, len(pat)):
        while k and pat[i] != pat[k]:
            k = fail[k - 1]
        if pat[i] == pat[k]:
            k += 1
        fail[i] = k
    return fail


def kmp_search_cost(text: str, pat: str) -> Tuple[List[int], int]:
    """KMP hits plus the number of character comparisons it spent."""
    fail = kmp_failure(pat)
    hits, j, cost = [], 0, 0
    for i, ch in enumerate(text):
        while j and ch != pat[j]:
            j = fail[j - 1]
            cost += 1
        cost += 1
        if ch == pat[j]:
            j += 1
        if j == len(pat):
            hits.append(i - j + 1)
            j = fail[j - 1]
    return hits, cost


def compare_scans(text: str, pats: Sequence[str]) -> Dict[str, object]:
    """k separate KMP passes vs one Aho-Corasick pass, counted honestly."""
    kmp_cost, kmp_hits = 0, []
    for p in pats:
        hits, c = kmp_search_cost(text, p)
        kmp_cost += c
        kmp_hits += [(h + len(p) - 1, p) for h in hits]
    ac = AhoCorasick(pats)
    ac.transitions = 0
    ac_hits = ac.find_all(text)
    return {
        "kmp_cost": kmp_cost,
        "kmp_passes": len(pats),
        "ac_cost": ac.transitions,
        "ac_nodes": ac.n_nodes(),
        "same": sorted(kmp_hits) == sorted(ac_hits),
        "hits": sorted(ac_hits),
    }


# ===========================================================================
# 6. Streaming: every stop string at once, and one number to hold back
# ===========================================================================

class MultiStopMatcher:
    """Day 25's streaming stop-string problem, with k stop strings.

    A server may not emit a character that could turn out to be the first
    character of a stop string, so it holds text back.  With one pattern the
    hold-back was 'how much of the pattern currently matches'.  With k
    patterns the naive version has to ask that of every pattern separately -
    but in the automaton the answer is a single number that is already there:
    **the depth of the current state**, because the state spells exactly the
    longest prefix of any pattern that the tail currently matches.  k stop
    strings therefore cost the same per character as one.
    """

    def __init__(self, stops: Sequence[str]):
        self.ac = AhoCorasick(stops)
        self.state = 0
        self.buf = ""          # characters not yet safe to emit
        self.emitted = ""
        self.hit: Optional[str] = None

    def feed(self, chunk: str) -> str:
        """Consume a chunk, return the text that is now safe to emit."""
        if self.hit is not None:
            return ""
        safe = ""
        for ch in chunk:
            self.buf += ch
            self.state = self.ac.step(self.state, ch)
            if self.ac.out[self.state]:
                self.hit = max(self.ac.out[self.state], key=len)
                keep = len(self.buf) - len(self.hit)
                safe += self.buf[:max(keep, 0)]
                self.buf = ""
                self.emitted += safe
                return safe
            hold = self.ac.depth[self.state]     # <- the whole trick
            if len(self.buf) > hold:
                safe += self.buf[:len(self.buf) - hold]
                self.buf = self.buf[len(self.buf) - hold:]
        self.emitted += safe
        return safe

    def pending(self) -> str:
        return self.buf


def stream_trace(stops: Sequence[str], chunks: Sequence[str]) -> List[Dict[str, object]]:
    """Per-character trace of the matcher, for the notebook and the demo."""
    m = MultiStopMatcher(stops)
    rows = []
    for c, chunk in enumerate(chunks):
        for ch in chunk:
            out = m.feed(ch)
            rows.append({"chunk": c, "ch": ch, "state": m.state,
                         "hold": m.ac.depth[m.state], "emit": out,
                         "buf": m.pending(), "hit": m.hit})
            if m.hit:
                return rows
    return rows


def per_char_cost(stops: Sequence[str], tail_len: int) -> Dict[str, object]:
    """Why an inference server can get away with the naive loop.

    sglang clips the tail it re-checks to stop_str_max_len + 1 characters and
    then runs a plain substring search per stop string, which is O(k * L^2)
    per step with L = that clipped length.  With k = 2 stop strings of ~8
    characters that is a few hundred character comparisons per *token* - noise
    next to a forward pass.  Aho-Corasick is O(1) amortised per character no
    matter what k is, and it earns its keep exactly when k stops being small:
    a banned-phrase or PII guardrail list with thousands of entries.
    """
    k, L = len(stops), tail_len
    return {"k": k, "tail": L, "naive_per_step": k * L * L, "ac_per_step": L}


# ===========================================================================
# 7. LeetCode
# ===========================================================================

class StreamChecker:
    """LC 1032 - queries arrive one character at a time, forever.

    The tempting answer is a trie of the reversed words plus a growing buffer,
    which re-walks the buffer on every query.  Aho-Corasick keeps one integer:
    the automaton state after the characters seen so far.  Each query is one
    step, and out[state] is non-empty exactly when some word just ended.
    """

    def __init__(self, words: List[str]):
        self.ac = AhoCorasick(words)
        self.state = 0

    def query(self, letter: str) -> bool:
        self.state = self.ac.step(self.state, letter)
        return bool(self.ac.out[self.state])


def longest_dup_substring(s: str) -> str:
    """LC 1044 - day 25 solved this with binary search + a rolling hash.

    That solution can in principle return a wrong answer, because two distinct
    substrings can share a hash and the verification step is what keeps it
    honest.  The suffix array answers the same question exactly and without
    randomness: build it once, take the largest LCP entry.
    """
    return longest_repeated_substring(s)


def add_bold_tag(s: str, words: List[str]) -> str:
    """LC 616 / 758 - wrap every occurrence of any word, merging overlaps.

    Two independent halves: find the occurrences (one Aho-Corasick pass), then
    merge the intervals.  Marking a boolean per character makes the merge
    trivial and is the part interviewers actually look at.
    """
    if not words:
        return s
    mark = [False] * len(s)
    for end, w in AhoCorasick(words).find_all(s):
        for i in range(end - len(w) + 1, end + 1):
            mark[i] = True
    out, i = [], 0
    while i < len(s):
        if not mark[i]:
            out.append(s[i])
            i += 1
            continue
        j = i
        while j < len(s) and mark[j]:
            j += 1
        out.append("<b>" + s[i:j] + "</b>")
        i = j
    return "".join(out)


# ===========================================================================
# 8. Walkthrough
# ===========================================================================

DOCS = [
    "# Copyright 2026 Acme Inc. Licensed under Apache 2.0.\n"
    "def train(model, data):\n    return model.fit(data)",
    "# Copyright 2026 Acme Inc. Licensed under Apache 2.0.\n"
    "def evaluate(model, data):\n    return model.score(data)",
]

STOPS = ["</s>", "\n\nHuman:", "[END]"]
CHUNKS = ["The answer", " is 42.<", "/", "s> and more"]


def head(title: str) -> None:
    print()
    print("=" * 74)
    print(title)
    print("=" * 74)


def main() -> None:
    head("1. the suffix array of 'banana'")
    s = "banana"
    sa = suffix_array(s)
    lcp = lcp_kasai(s, sa)
    print("  i  sa[i]  lcp[i]  suffix")
    for i, p in enumerate(sa):
        print("  %d   %d      %d      %s" % (i, p, lcp[i], s[p:]))
    print()
    print("agrees with the naive sort:", sa == suffix_array_naive(s))
    print("Kasai agrees with the O(n^2) reference:", lcp == lcp_naive(s, sa))

    head("2. prefix doubling: sort by 1, then 2, then 4 characters")
    for k, rank in rank_rounds(s):
        print("  k=%-2d ranks %s" % (k, rank))
    print("  ranks are distinct -> the order is final, %d rounds for n=%d"
          % (len(rank_rounds(s)), len(s)))

    head("3. a query is two binary searches (day 23)")
    for pat in ["ana", "na", "ba", "x"]:
        a, b, probes = sa_range(s, sa, pat)
        print("  %-4s -> sa[%d:%d] = %-9s %d occurrence(s), %d probes"
              % (repr(pat), a, b, sa_occurrences(s, sa, pat), b - a, probes))
    print("  the text was preprocessed once; KMP would re-read all %d characters per query"
          % len(s))

    head("4. what the LCP array knows")
    for t in ["banana", "mississippi", "abcdefg"]:
        print("  longest repeated substring of %-13s = %r" % (repr(t), longest_repeated_substring(t)))
    print("  longest common substring of 'ababc' and 'abcdaba' = %r"
          % longest_common_substring("ababc", "abcdaba"))

    head("5. deduplicating a corpus")
    rep = dedup_report(DOCS, min_len=20)
    print("  %d characters over %d documents" % (rep["chars"], len(DOCS)))
    for count, sub in rep["spans"]:
        print("    x%d  %r" % (count, sub))
    print("  dropping the repeats saves %d characters (%.1f%%)" % (rep["saved"], rep["pct"]))

    head("6. Aho-Corasick over {he, she, his, hers}")
    ac = AhoCorasick(["he", "she", "his", "hers"])
    print("  node  depth  spells   fail -> spells   outputs")
    spell = [""] * ac.n_nodes()
    for v in range(ac.n_nodes()):
        for ch, u in ac.goto[v].items():
            spell[u] = spell[v] + ch
    for v in range(ac.n_nodes()):
        print("   %-4d %-6d %-8s %-4d %-8s %s"
              % (v, ac.depth[v], repr(spell[v]), ac.fail[v], repr(spell[ac.fail[v]]),
                 ac.out[v] or ""))
    print()
    print("  find_all('ushers') ->", ac.find_all("ushers"))
    print("  'he' is found inside 'she' only because out[] follows fail links")

    head("7. for a single pattern, the fail links ARE KMP's failure function")
    pat = "ababaca"
    print("  pattern       ", " ".join(pat))
    print("  fail depths   ", " ".join(map(str, AhoCorasick([pat]).fail_depths())))
    print("  kmp_failure   ", " ".join(map(str, kmp_failure(pat))))

    head("8. k passes vs one pass")
    text = "the sheriff said he is his own usher here " * 3
    pats = ["he", "she", "his", "hers", "usher"]
    cmp = compare_scans(text, pats)
    print("  text %d chars, %d patterns" % (len(text), len(pats)))
    print("  %d KMP passes: %d character comparisons"
          % (cmp["kmp_passes"], cmp["kmp_cost"]))
    print("  1 Aho-Corasick pass: %d transitions over a %d-node automaton"
          % (cmp["ac_cost"], cmp["ac_nodes"]))
    print("  same hits:", cmp["same"], " (%d of them)" % len(cmp["hits"]))

    head("9. three stop strings, one hold-back number")
    print("  stops   :", STOPS)
    print("  chunks  :", CHUNKS)
    print()
    print("  ch  state  hold  emit      buffer")
    for r in stream_trace(STOPS, CHUNKS):
        print("  %-3s %-6d %-5d %-9s %s"
              % (repr(r["ch"]), r["state"], r["hold"], repr(r["emit"]), repr(r["buf"])))
        if r["hit"]:
            print("  -> %r completed here, so nothing after it may be emitted" % r["hit"])
            break
    m = MultiStopMatcher(STOPS)
    for c in CHUNKS:
        m.feed(c)
    print("  emitted %r, held back %r, stopped on %r" % (m.emitted, m.pending(), m.hit))
    print("  the hold-back is depth[state] - one integer, whatever k is")

    head("10. why an inference server can still get away with the naive loop")
    cost = per_char_cost(["</s>", "\n\nHuman:"], tail_len=9)
    print("  k=%d stop strings, clipped tail of %d characters" % (cost["k"], cost["tail"]))
    print("  naive re-scan  : ~%d character comparisons per step" % cost["naive_per_step"])
    print("  Aho-Corasick   : ~%d transitions per step" % cost["ac_per_step"])
    print("  both are noise next to a forward pass - until k is a guardrail list of thousands")

    head("11. LeetCode")
    sc = StreamChecker(["cd", "f", "kl"])
    print("  1032 stream :", [sc.query(c) for c in "abcdefghijkl"])
    print("  1044 exact  : longest_dup_substring('banana') = %r  (day 25 got the same"
          " answer with a rolling hash - and a collision risk)" % longest_dup_substring("banana"))
    print("  616  bold   :", add_bold_tag("aaabbcc", ["aaa", "aab", "bc"]))

    head("12. asserts")
    for t in ["banana", "mississippi", "abracadabra", "aaaa", "a", ""]:
        sa_t = suffix_array(t)
        assert sa_t == suffix_array_naive(t), t
        assert lcp_kasai(t, sa_t) == lcp_naive(t, sa_t), t
    sa_b = suffix_array("banana")
    assert sa_occurrences("banana", sa_b, "ana") == [1, 3]
    assert sa_count("banana", sa_b, "na") == 2 and sa_count("banana", sa_b, "x") == 0
    assert longest_repeated_substring("mississippi") == "issi"
    assert longest_repeated_substring("abcdefg") == ""
    assert longest_common_substring("ababc", "abcdaba") == "aba"
    assert dedup_report(DOCS, 20)["saved"] == 90
    assert [c for c, _ in duplicate_spans("banana", 2)] == [2]
    assert AhoCorasick(["ababaca"]).fail_depths() == kmp_failure("ababaca")
    assert sorted(AhoCorasick(["he", "she", "his", "hers"]).find_all("ushers")) == \
        [(3, "he"), (3, "she"), (5, "hers")]
    assert compare_scans("the sheriff said he is his own usher here" * 3,
                         ["he", "she", "his", "hers", "usher"])["same"]
    mm = MultiStopMatcher(STOPS)
    assert "".join(mm.feed(c) for c in CHUNKS) == "The answer is 42."
    assert mm.hit == "</s>" and mm.pending() == ""
    sc2 = StreamChecker(["cd", "f", "kl"])
    assert [sc2.query(c) for c in "abcdefghijkl"][3] and not [StreamChecker(["cd"]).query(c) for c in "ab"][0]
    assert add_bold_tag("aaabbcc", ["aaa", "aab", "bc"]) == "<b>aaabbc</b>c"
    print("all assertions passed")


if __name__ == "__main__":
    main()
