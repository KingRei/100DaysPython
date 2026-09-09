"""Day 27 - edit distance and similarity: Levenshtein, Jaccard, MinHash.

Run me:  python3 edit_similarity.py

Days 24-26 all asked exact questions: does this pattern occur, where, what is
repeated verbatim.  Real text is messier.  A user types "recieve", an ASR model
drops a word, two web pages differ by a cookie banner - and every one of those
is a *near* match that an exact structure reports as no match at all.

Two families of answer, and the gap between them is the point of today:

  * **Edit distance** measures how far apart two strings are, exactly, by
    dynamic programming.  It costs O(n*m) per pair, which is fine for a
    spelling correction and hopeless for a corpus.
  * **Jaccard + MinHash + LSH** trade exactness for scale: a fixed-size
    signature estimates set overlap, and banding turns "compare all pairs"
    into "look in the same bucket", which is what actually deduplicates a
    training corpus of a billion documents.

Standard library only.
"""

from __future__ import annotations

import hashlib
import random
from typing import Dict, Iterable, List, Sequence, Set, Tuple


# ===========================================================================
# 1. Levenshtein distance - the table, and the path through it
# ===========================================================================

def edit_table(a: str, b: str) -> List[List[int]]:
    """Full (len(a)+1) x (len(b)+1) table of edit distances between prefixes.

    d[i][j] is the distance between a[:i] and b[:j].  The recurrence has
    exactly three predecessors, and each one is an operation:

        d[i-1][j]   + 1          delete a[i-1]
        d[i][j-1]   + 1          insert b[j-1]
        d[i-1][j-1] + (a != b)   substitute, or match for free

    The first row and column are the base cases: turning a prefix into the
    empty string costs one deletion per character.
    """
    n, m = len(a), len(b)
    d = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        d[i][0] = i
    for j in range(m + 1):
        d[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            d[i][j] = min(d[i - 1][j] + 1,          # delete
                          d[i][j - 1] + 1,          # insert
                          d[i - 1][j - 1] + cost)   # substitute / match
    return d


def edit_distance(a: str, b: str) -> int:
    """The bottom-right corner of the table."""
    return edit_table(a, b)[len(a)][len(b)]


def edit_ops(a: str, b: str) -> List[Tuple[str, int, str, str]]:
    """Walk backwards through the table to recover one optimal script.

    The distance alone is rarely what you want - a spell checker wants the
    correction, a diff wants the hunks, an ASR report wants how many words
    were substituted rather than inserted.  Backtracking is free: at each cell
    ask which of the three predecessors the value came from.

    Returns tuples (op, position_in_a, from_char, to_char) in left-to-right
    order, where op is one of match / substitute / delete / insert.
    """
    d = edit_table(a, b)
    i, j, ops = len(a), len(b), []
    while i > 0 or j > 0:
        if i > 0 and j > 0:
            cost = 0 if a[i - 1] == b[j - 1] else 1
            if d[i][j] == d[i - 1][j - 1] + cost:
                ops.append(("match" if cost == 0 else "substitute",
                            i - 1, a[i - 1], b[j - 1]))
                i, j = i - 1, j - 1
                continue
        if i > 0 and d[i][j] == d[i - 1][j] + 1:
            ops.append(("delete", i - 1, a[i - 1], ""))
            i -= 1
            continue
        ops.append(("insert", i, "", b[j - 1]))
        j -= 1
    ops.reverse()
    return ops


def edit_distance_rows(a: str, b: str) -> int:
    """Same answer in O(min(n, m)) memory: only the previous row is needed.

    Every cell reads d[i-1][j], d[i][j-1] and d[i-1][j-1], all of which live
    in the current row or the one above it.  The table is only needed if you
    want to backtrack afterwards - distance alone never needs it.
    """
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i] + [0] * len(b)
        for j, cb in enumerate(b, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1,
                         prev[j - 1] + (ca != cb))
        prev = cur
    return prev[len(b)]


def similarity_ratio(a: str, b: str) -> float:
    """Normalised into [0, 1] so it can be compared across lengths."""
    if not a and not b:
        return 1.0
    return 1.0 - edit_distance(a, b) / max(len(a), len(b))


# ===========================================================================
# 2. Bounded distance - the only version that is fast enough in practice
# ===========================================================================

def edit_distance_bounded(a: str, b: str, k: int) -> Tuple[int, int]:
    """Distance if it is <= k, otherwise k + 1.  Returns (distance, cells).

    Almost every real caller has a threshold: "did the user mistype this by at
    most 2", "is this a near-duplicate".  Under that threshold only a diagonal
    band of width 2k+1 can matter, because leaving the diagonal by one cell
    already costs one operation.  The band makes the cost O(k * min(n, m))
    instead of O(n * m), and the length difference alone can answer some
    queries with no table at all.
    """
    n, m = len(a), len(b)
    if abs(n - m) > k:              # every extra character costs an indel
        return k + 1, 0
    INF = k + 1
    prev = [INF] * (m + 1)
    for j in range(min(k, m) + 1):
        prev[j] = j
    cells = 0
    for i in range(1, n + 1):
        cur = [INF] * (m + 1)
        lo, hi = max(1, i - k), min(m, i + k)
        if lo == 1:
            cur[0] = i
        for j in range(lo, hi + 1):
            cells += 1
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1,
                         prev[j - 1] + (a[i - 1] != b[j - 1]))
        if min(cur[lo:hi + 1] or [INF]) > k:   # the whole band exceeded k
            return k + 1, cells
        prev = cur
    return (prev[m], cells) if prev[m] <= k else (k + 1, cells)


def full_cells(a: str, b: str) -> int:
    """How many cells the unbounded table fills, for comparison."""
    return len(a) * len(b)


# ===========================================================================
# 3. Relatives: LCS (what diff uses) and word error rate
# ===========================================================================

def lcs_length(a: Sequence, b: Sequence) -> int:
    """Edit distance with substitution removed: only insert and delete.

    That single change turns the metric into "longest common subsequence",
    which is what a line-based diff computes: a line cannot be *edited*, only
    added or removed, so the cheapest script is the one that keeps the most
    common lines.
    """
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0]
        for j, y in enumerate(b, 1):
            cur.append(prev[j - 1] + 1 if x == y else max(prev[j], cur[j - 1]))
        prev = cur
    return prev[len(b)]


def diff_script(a: Sequence, b: Sequence) -> List[Tuple[str, object]]:
    """A minimal keep/delete/insert script - the shape of `diff` output."""
    n, m = len(a), len(b)
    d = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            d[i][j] = (d[i - 1][j - 1] + 1 if a[i - 1] == b[j - 1]
                       else max(d[i - 1][j], d[i][j - 1]))
    i, j, out = n, m, []
    while i > 0 and j > 0:
        if a[i - 1] == b[j - 1]:
            out.append(("keep", a[i - 1])); i, j = i - 1, j - 1
        elif d[i - 1][j] >= d[i][j - 1]:
            out.append(("delete", a[i - 1])); i -= 1
        else:
            out.append(("insert", b[j - 1])); j -= 1
    while i > 0:
        out.append(("delete", a[i - 1])); i -= 1
    while j > 0:
        out.append(("insert", b[j - 1])); j -= 1
    out.reverse()
    return out


def word_error_rate(reference: str, hypothesis: str) -> Dict[str, object]:
    """WER = (S + D + I) / N, the standard ASR metric - edit distance on words.

    Nothing changes except the alphabet: the "characters" are now whole words.
    The breakdown matters more than the number, because substitutions,
    deletions and insertions have very different causes in a speech system.
    """
    ref, hyp = reference.split(), hypothesis.split()
    ops = edit_ops_seq(ref, hyp)
    counts = {"substitute": 0, "delete": 0, "insert": 0, "match": 0}
    for op, *_ in ops:
        counts[op] += 1
    errors = counts["substitute"] + counts["delete"] + counts["insert"]
    return {"wer": errors / max(len(ref), 1), "errors": errors,
            "n_ref": len(ref), "ops": ops, **counts}


def edit_ops_seq(a: Sequence, b: Sequence) -> List[Tuple[str, int, object, object]]:
    """edit_ops, but over any sequence of tokens rather than characters."""
    n, m = len(a), len(b)
    d = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        d[i][0] = i
    for j in range(m + 1):
        d[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1,
                          d[i - 1][j - 1] + (a[i - 1] != b[j - 1]))
    i, j, ops = n, m, []
    while i > 0 or j > 0:
        if i > 0 and j > 0:
            cost = 0 if a[i - 1] == b[j - 1] else 1
            if d[i][j] == d[i - 1][j - 1] + cost:
                ops.append(("match" if cost == 0 else "substitute",
                            i - 1, a[i - 1], b[j - 1]))
                i, j = i - 1, j - 1
                continue
        if i > 0 and d[i][j] == d[i - 1][j] + 1:
            ops.append(("delete", i - 1, a[i - 1], None)); i -= 1
            continue
        ops.append(("insert", i, None, b[j - 1])); j -= 1
    ops.reverse()
    return ops


# ===========================================================================
# 4. Jaccard on shingles - similarity that scales to documents
# ===========================================================================

def shingles(text: str, k: int = 5) -> Set[str]:
    """The set of k-grams of *words*.  A document becomes a set.

    Edit distance on two 5 kB documents is 25 million cells.  Shingling throws
    away the ordering above the k-gram level and keeps a set, and sets have a
    similarity measure that costs nothing: how much they overlap.  k = 5 words
    is the usual choice for near-duplicate detection - short enough that
    genuine copies share thousands of shingles, long enough that unrelated
    English text shares almost none.
    """
    words = text.lower().split()
    if len(words) < k:
        return {" ".join(words)} if words else set()
    return {" ".join(words[i:i + k]) for i in range(len(words) - k + 1)}


def jaccard(a: Set[str], b: Set[str]) -> float:
    """|A n B| / |A u B| - 1.0 for identical sets, 0.0 for disjoint ones."""
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def all_pairs_jaccard(docs: Sequence[str], k: int = 5,
                      threshold: float = 0.5) -> Tuple[List[Tuple[int, int, float]], int]:
    """Exact answer by brute force.  Returns (pairs above threshold, comparisons).

    This is the baseline every approximate method is measured against, and the
    reason they exist: n documents means n*(n-1)/2 comparisons.  Ten thousand
    documents is 50 million pairs; ten million documents is 5 * 10^13.
    """
    sets = [shingles(d, k) for d in docs]
    hits, comparisons = [], 0
    for i in range(len(docs)):
        for j in range(i + 1, len(docs)):
            comparisons += 1
            s = jaccard(sets[i], sets[j])
            if s >= threshold:
                hits.append((i, j, s))
    return hits, comparisons


# ===========================================================================
# 5. MinHash - a fixed-size sketch whose collisions *are* the Jaccard
# ===========================================================================

def _hash(item: str, seed: int) -> int:
    """A seeded 64-bit hash.

    Python's built-in hash() on str is randomised per process, so a signature
    built with it would differ between runs and nothing here would reproduce.
    blake2b takes a key, which gives an independent hash function per seed and
    the same numbers on every machine.
    """
    h = hashlib.blake2b(item.encode("utf-8"), digest_size=8,
                        key=seed.to_bytes(8, "little"))
    return int.from_bytes(h.digest(), "little")


def minhash_signature(s: Set[str], num_hashes: int = 128) -> List[int]:
    """One number per hash function: the smallest hash value in the set.

    The whole trick in one line.  Under a random permutation of the universe,
    the odds that two sets have the same minimum element are exactly the odds
    that the minimum of their union happens to lie in their intersection -
    which is |A n B| / |A u B|, the Jaccard similarity.  So each hash function
    is one Bernoulli trial with success probability J, and the fraction of
    matching slots estimates J with no bias.
    """
    if not s:
        return [0] * num_hashes
    return [min(_hash(x, seed) for x in s) for seed in range(num_hashes)]


def estimate_jaccard(sig_a: Sequence[int], sig_b: Sequence[int]) -> float:
    """Fraction of signature slots that agree."""
    same = sum(1 for x, y in zip(sig_a, sig_b) if x == y)
    return same / len(sig_a)


def signature_error(docs: Sequence[str], num_hashes: int, k: int = 5,
                    min_true: float = 0.05) -> float:
    """Mean |estimate - truth| over the overlapping pairs, watching 1/sqrt(K).

    The standard error of the estimate is sqrt(J(1-J)/K), so ten times more
    hashes buys about three times less error.  That is the whole cost model:
    accuracy is bought in square roots, but the signature is a fixed size no
    matter how long the document is.
    """
    sets = [shingles(d, k) for d in docs]
    sigs = [minhash_signature(s, num_hashes) for s in sets]
    diffs = []
    for i in range(len(docs)):
        for j in range(i + 1, len(docs)):
            truth = jaccard(sets[i], sets[j])
            if truth < min_true:        # disjoint pairs are free to get right
                continue
            diffs.append(abs(estimate_jaccard(sigs[i], sigs[j]) - truth))
    return sum(diffs) / len(diffs)


# ===========================================================================
# 6. LSH banding - stop comparing all pairs at all
# ===========================================================================

def band_buckets(signatures: Sequence[Sequence[int]], bands: int,
                 rows: int) -> Dict[Tuple[int, Tuple[int, ...]], List[int]]:
    """Split each signature into `bands` strips of `rows` and hash each strip.

    Two documents become candidates if *any one* strip matches exactly.  A
    single band of r rows matches with probability J^r, so the chance of
    missing a pair entirely is (1 - J^r)^b - and 1 minus that is an S-curve in
    J, steep around (1/b)^(1/r).  Choosing b and r *is* choosing the threshold.
    """
    buckets: Dict[Tuple[int, Tuple[int, ...]], List[int]] = {}
    for idx, sig in enumerate(signatures):
        for b in range(bands):
            strip = tuple(sig[b * rows:(b + 1) * rows])
            buckets.setdefault((b, strip), []).append(idx)
    return buckets


def candidate_pairs(signatures: Sequence[Sequence[int]], bands: int,
                    rows: int) -> Set[Tuple[int, int]]:
    """Every pair that shares at least one band bucket."""
    pairs: Set[Tuple[int, int]] = set()
    for members in band_buckets(signatures, bands, rows).values():
        if len(members) > 1:
            for a in range(len(members)):
                for b in range(a + 1, len(members)):
                    pairs.add((members[a], members[b]))
    return pairs


def lsh_threshold(bands: int, rows: int) -> float:
    """The steep point of the S-curve, approximately (1/b)^(1/r)."""
    return (1.0 / bands) ** (1.0 / rows)


def prob_candidate(j: float, bands: int, rows: int) -> float:
    """P(at least one band matches) = 1 - (1 - J^r)^b."""
    return 1.0 - (1.0 - j ** rows) ** bands


def near_duplicates(docs: Sequence[str], k: int = 5, num_hashes: int = 128,
                    bands: int = 32, rows: int = 4,
                    threshold: float = 0.5) -> Dict[str, object]:
    """The full pipeline: shingle, sketch, bucket, verify only the candidates.

    The verification step still computes the exact Jaccard, but only on pairs
    that survived banding - which is the point.  LSH is a filter, not an
    answer, and it is allowed to be wrong in both directions as long as the
    curve is steep enough that it rarely is.
    """
    sets = [shingles(d, k) for d in docs]
    sigs = [minhash_signature(s, num_hashes) for s in sets]
    cands = candidate_pairs(sigs, bands, rows)
    verified = sorted((i, j, jaccard(sets[i], sets[j]))
                      for i, j in cands if jaccard(sets[i], sets[j]) >= threshold)
    exact, comparisons = all_pairs_jaccard(docs, k, threshold)
    return {"candidates": len(cands), "all_pairs": comparisons,
            "found": verified, "exact": sorted(exact),
            "threshold_curve": lsh_threshold(bands, rows)}


# ===========================================================================
# 7. LeetCode 72 / 1143 / 161
# ===========================================================================

def min_distance(word1: str, word2: str) -> int:
    """LC 72 - Edit Distance.  The rolling-row version, O(min(n, m)) space.

    Interview caveat: the interviewer usually wants the space optimisation,
    but ask before doing it - if the follow-up is "now print the edits" you
    need the full table back, and rewriting it under pressure is worse than
    having kept it.
    """
    return edit_distance_rows(word1, word2)


def longest_common_subsequence(text1: str, text2: str) -> int:
    """LC 1143 - the same table with substitution deleted from the menu."""
    return lcs_length(text1, text2)


def is_one_edit_distance(s: str, t: str) -> bool:
    """LC 161 - distance exactly 1, in O(n) time and O(1) space.

    The special case that motivates the band: when k = 1 the table collapses
    to a single scan.  Find the first mismatch, then check whether the rest
    lines up after skipping one character on the longer side.  Same idea as
    edit_distance_bounded, taken to its narrowest possible band.
    """
    if len(s) > len(t):
        s, t = t, s
    if len(t) - len(s) > 1 or s == t:
        return False
    for i in range(len(s)):
        if s[i] != t[i]:
            if len(s) == len(t):
                return s[i + 1:] == t[i + 1:]     # substitute
            return s[i:] == t[i + 1:]             # insert into s
    return len(t) == len(s) + 1                   # t has one extra at the end


# ===========================================================================
# Demo
# ===========================================================================

A = ("machine learning models are trained on large corpora of text collected "
     "from the public web and the same paragraph often appears in thousands of "
     "pages so a training set that is not deduplicated will show the model the "
     "same sentences again and again")

B = ("a suffix array sorts every suffix of a string so that any pattern can be "
     "located with two binary searches over the sorted order and the lcp array "
     "records how much neighbouring suffixes share which is enough to find "
     "every repeated substring in the text")

DOCS = [
    A,                                                    # 0
    A.replace("public web", "open web"),                  # 1: one phrase changed
    A + " copyright notices and cookie banners make this worse",   # 2: boilerplate
    B,                                                    # 3
    B.replace("every suffix", "all suffixes")             # 4: two edits
     .replace("neighbouring", "adjacent"),
    ("the scheduler admits requests into the running batch until the batch is "
     "full and then runs one forward pass over the whole batch so that the "
     "weights are read from memory once instead of once per request which is "
     "where the throughput comes from"),                  # 5
    ("bananas grow in tropical climates and are harvested green before they are "
     "shipped in refrigerated containers and then ripened in a controlled room "
     "with ethylene gas a few days before they reach the shelves of a "
     "supermarket"),                                      # 6
    ("edit distance counts the smallest number of insertions deletions and "
     "substitutions that turn one string into another which makes it the right "
     "measure for a spelling corrector but far too slow to run over every pair "
     "of documents in a corpus"),                         # 7
]


def _rule(title: str) -> None:
    print("\n" + "=" * 68)
    print(title)
    print("=" * 68)


def main() -> None:
    _rule("1. Levenshtein: kitten -> sitting")
    a, b = "kitten", "sitting"
    table = edit_table(a, b)
    print("     " + "  ".join(f"{c:>2}" for c in " " + b))
    for i, row in enumerate(table):
        head = " " if i == 0 else a[i - 1]
        print(f"  {head}  " + "  ".join(f"{v:>2}" for v in row))
    print(f"\n  distance          = {edit_distance(a, b)}")
    print(f"  rolling row       = {edit_distance_rows(a, b)}")
    print(f"  similarity        = {similarity_ratio(a, b):.3f}")
    print("  script:")
    for op, pos, frm, to in edit_ops(a, b):
        if op == "match":
            continue
        print(f"    {op:<10} at {pos}: {frm or '-'} -> {to or '-'}")

    _rule("2. Bounded distance: only a diagonal band can matter")
    for k in (1, 2, 3):
        d, cells = edit_distance_bounded(a, b, k)
        verdict = f"{d}" if d <= k else f"> {k}"
        print(f"  k = {k}:  distance {verdict:<4} cells filled {cells:>3} "
              f"(full table {full_cells(a, b)})")
    long_a = "the quick brown fox jumps over the lazy dog " * 3
    long_b = long_a.replace("quick", "quack")       # 3 substitutions, same length
    full = full_cells(long_a, long_b)
    d3, c3 = edit_distance_bounded(long_a, long_b, 3)
    print(f"  {len(long_a)}-char pair, 3 substitutions apart:")
    print(f"    k = 3: distance {d3}, {c3} cells vs {full} for the full table "
          f"({c3 / full:.1%})")
    d1, c1 = edit_distance_bounded(long_a, long_b, 1)
    print(f"    k = 1: distance > 1 after {c1} cells - the whole band exceeded "
          f"k, so it stopped early")
    pad = long_a + "xxxxx"
    print(f"    length gate: |n - m| = 5 > k = 3, answered with "
          f"{edit_distance_bounded(long_a, pad, 3)[1]} cells")

    _rule("3. Same table, different alphabet: word error rate")
    ref = "the model streams tokens back to the client"
    hyp = "the model streamed tokens to a client"
    r = word_error_rate(ref, hyp)
    print(f"  reference : {ref}")
    print(f"  hypothesis: {hyp}")
    print(f"  WER = ({r['substitute']}S + {r['delete']}D + {r['insert']}I)"
          f" / {r['n_ref']} = {r['wer']:.3f}")
    for op, pos, frm, to in r["ops"]:
        if op != "match":
            print(f"    {op:<10} {str(frm or '-'):<10} -> {to or '-'}")
    print("\n  LCS (edit distance without substitution) drives diff:")
    old = ["import os", "def main():", "    run()", "    return 0"]
    new = ["import os", "import sys", "def main():", "    return 0"]
    for op, line in diff_script(old, new):
        mark = {"keep": " ", "delete": "-", "insert": "+"}[op]
        print(f"    {mark} {line}")

    _rule("4. Jaccard on 5-word shingles")
    s0, s1, s2, s3 = (shingles(DOCS[0]), shingles(DOCS[1]),
                      shingles(DOCS[2]), shingles(DOCS[3]))
    print(f"  doc0 has {len(s0)} shingles, doc2 has {len(s2)}")
    print(f"  J(doc0, doc1) = {jaccard(s0, s1):.3f}   (one phrase swapped)")
    print(f"  J(doc0, doc2) = {jaccard(s0, s2):.3f}   (a clause appended)")
    print(f"  J(doc0, doc3) = {jaccard(s0, s3):.3f}   (unrelated topic)")
    hits, comparisons = all_pairs_jaccard(DOCS, threshold=0.5)
    print(f"  brute force over {len(DOCS)} docs = {comparisons} comparisons, "
          f"{len(hits)} pairs above 0.5:")
    for i, j, s in hits:
        print(f"    doc{i} ~ doc{j}   J = {s:.3f}")

    _rule("5. MinHash: collisions estimate the Jaccard")
    truth = jaccard(s0, s1)
    print(f"  true J(doc0, doc1) = {truth:.3f}")
    for K in (16, 64, 256, 1024):
        est = estimate_jaccard(minhash_signature(s0, K), minhash_signature(s1, K))
        print(f"    K = {K:>4}: estimate {est:.3f}   error {abs(est - truth):.3f}"
              f"   (predicted +-{(truth * (1 - truth) / K) ** .5:.3f})")
    print("\n  mean error over all pairs, as K grows:")
    for K in (16, 64, 256):
        print(f"    K = {K:>3}: {signature_error(DOCS, K):.4f}")

    _rule("6. LSH banding: candidates instead of all pairs")
    for bands, rows in ((32, 4), (16, 8), (8, 16)):
        print(f"  b = {bands:>2}, r = {rows:>2}: threshold ~ "
              f"{lsh_threshold(bands, rows):.2f}   "
              f"P(candidate | J=0.3) = {prob_candidate(0.3, bands, rows):.3f}   "
              f"P(candidate | J=0.8) = {prob_candidate(0.8, bands, rows):.3f}")
    res = near_duplicates(DOCS, num_hashes=128, bands=32, rows=4, threshold=0.5)
    print(f"\n  pipeline: {res['candidates']} candidate pairs verified "
          f"instead of {res['all_pairs']} ("
          f"{res['candidates'] / res['all_pairs']:.1%} of the work)")
    for i, j, s in res["found"]:
        print(f"    doc{i} ~ doc{j}   J = {s:.3f}")
    print(f"  matches brute force exactly: "
          f"{[(i, j) for i, j, _ in res['found']] == [(i, j) for i, j, _ in res['exact']]}")

    _rule("7. LeetCode")
    print(f"  72   min_distance('horse', 'ros')             = "
          f"{min_distance('horse', 'ros')}")
    print(f"  72   min_distance('intention', 'execution')   = "
          f"{min_distance('intention', 'execution')}")
    print(f"  1143 LCS('abcde', 'ace')                      = "
          f"{longest_common_subsequence('abcde', 'ace')}")
    print(f"  161  is_one_edit_distance('ab', 'acb')        = "
          f"{is_one_edit_distance('ab', 'acb')}")
    print(f"  161  is_one_edit_distance('cab', 'ad')        = "
          f"{is_one_edit_distance('cab', 'ad')}")

    # ---- asserts ----
    assert edit_distance("kitten", "sitting") == 3
    assert edit_distance_rows("kitten", "sitting") == 3
    assert edit_distance("", "abc") == 3 and edit_distance("abc", "abc") == 0
    ops = edit_ops("kitten", "sitting")
    assert sum(1 for o in ops if o[0] != "match") == 3
    rebuilt = []
    for op, _, frm, to in ops:
        if op in ("match", "substitute", "insert"):
            rebuilt.append(to)
    assert "".join(rebuilt) == "sitting"
    assert edit_distance_bounded("kitten", "sitting", 1)[0] == 2      # k+1
    assert edit_distance_bounded("kitten", "sitting", 3)[0] == 3
    assert edit_distance_bounded("abc", "abcdef", 2)[0] == 3          # length gate
    assert edit_distance_bounded("kitten", "sitting", 3)[1] < full_cells(a, b)
    assert lcs_length("abcde", "ace") == 3
    assert lcs_length("abc", "abc") == 3 and lcs_length("abc", "def") == 0
    assert abs(word_error_rate("a b c d", "a b c d")["wer"]) < 1e-9
    assert jaccard(s0, s0) == 1.0 and jaccard(s0, set()) == 0.0
    assert minhash_signature(s0, 32) == minhash_signature(s0, 32)     # deterministic
    assert estimate_jaccard(minhash_signature(s0, 64),
                            minhash_signature(s0, 64)) == 1.0
    assert abs(estimate_jaccard(minhash_signature(s0, 512),
                                minhash_signature(s1, 512)) - truth) < 0.08
    assert res["candidates"] < res["all_pairs"]
    assert [(i, j) for i, j, _ in res["found"]] == [(i, j) for i, j, _ in res["exact"]]
    assert is_one_edit_distance("ab", "acb") and not is_one_edit_distance("ab", "ab")
    assert min_distance("horse", "ros") == 3
    print("\nall assertions passed")


if __name__ == "__main__":
    main()
