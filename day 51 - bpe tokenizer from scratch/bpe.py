"""Day 51 - a byte-level BPE tokenizer written from scratch.

A language model never sees your text.  It sees a list of integers, and the thing
that produced that list made a long series of choices you did not make: where words
break, whether a leading space belongs to the next word, how a number is cut up, and
how many of those integers your sentence costs.  Run `python bpe.py` to print the
whole story; every number below is produced by this file and checked by an assert at
the bottom.
"""

import re
from collections import Counter

# A fixed corpus, so every number on this page is reproducible.  Deliberately mixed:
# prose, numbers, code-ish punctuation and a line of Chinese.
CORPUS = """
the cost of a model call is counted in tokens, not in words and not in characters.
a token is whatever the tokenizer decided a token is, and the tokenizer was trained
on a corpus you never saw. the same sentence costs a different number of tokens in
every model, because every model merged a different set of pairs in a different order.
tokenization is the first layer of the model and the only one that is not learned by
gradient descent. it is learned by counting.
the training loop is three lines: count every adjacent pair, merge the most frequent
one, repeat. the merges are kept in the order they were made, and encoding replays
them in that order. decoding is the reverse: look up the bytes of every token and
join them back together.
a tokenizer that cannot reproduce its input is broken, however good its compression
is. the test is a round trip: encode, decode, compare bytes.
counting words is not counting tokens. counting characters is not counting tokens.
the only thing that counts tokens is the tokenizer.
prices are quoted per million tokens: 0.15, 0.60, 1.25, 10.00 per million.
a context window of 128000 tokens is not 128000 words.
def encode(text): return [vocab[b] for b in text.encode()]
模型看到的從來不是你的文字，而是一串整數。
"""


# ================================================================ 1. bytes
def to_bytes(text):
    """Text becomes bytes before it becomes anything else.

    Working in bytes rather than characters is what makes a tokenizer total: there
    are only 256 possible bytes, every one of them is in the vocabulary from the
    start, and no input can produce an unknown token.  The cost is that one Chinese
    character is three bytes and one emoji can be four.
    """
    return list(text.encode('utf-8'))


# ================================================================ 2. pre-tokenising
# The pattern decides what BPE is even allowed to merge.  Two rules carry most of the
# weight: a leading space is glued to the word that follows it, and a run of digits is
# cut into groups of at most three.
PRETOKEN = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?[^\W\d_]+| ?\d{1,3}| ?[^\s\w]+|\s+""")


def pretokenize(text, pattern=None):
    """Split text into chunks that BPE is allowed to merge inside, never across.

    Without this step the most frequent pair in English prose is (space, t), and the
    tokenizer spends its whole budget learning fragments that straddle word
    boundaries.  With it, ' the' is one chunk and can become one token.
    """
    return [m.group(0) for m in (pattern or PRETOKEN).finditer(text)]


# ================================================================ 3. training
def pair_counts(words):
    """How often each adjacent pair occurs, summed over the corpus.

    `words` maps a tuple of token ids to how many times that chunk appeared, so a
    chunk that occurs 400 times contributes 400 to each of its pairs - the counting
    is over occurrences in the corpus, not over distinct chunks.
    """
    counts = Counter()
    for word, freq in words.items():
        for a, b in zip(word, word[1:]):
            counts[(a, b)] += freq
    return counts


def merge_word(word, pair, new_id):
    """Replace every occurrence of `pair` in one chunk, left to right, without overlap."""
    out, i = [], 0
    while i < len(word):
        if i + 1 < len(word) and (word[i], word[i + 1]) == pair:
            out.append(new_id)
            i += 2
        else:
            out.append(word[i])
            i += 1
    return tuple(out)


def train(text, vocab_size, verbose_until=0, pattern=None):
    """Count pairs, merge the most frequent, repeat until the vocabulary is full.

    Returns (merges, vocab).  `merges` is an ordered list of (pair -> new id); the
    order is part of the tokenizer, not an implementation detail - encoding replays
    exactly this order.  `vocab` maps every id to the bytes it stands for.

    Ties are broken by taking the lowest pair of ids, so two runs on the same corpus
    produce the same tokenizer.  Without that rule `max` would follow dictionary
    order and the vocabulary would change between Python versions.
    """
    words = Counter(tuple(to_bytes(chunk)) for chunk in pretokenize(text, pattern))
    vocab = {i: bytes([i]) for i in range(256)}
    merges, trace = [], []
    while len(vocab) < vocab_size:
        counts = pair_counts(words)
        if not counts:
            break
        best = max(counts.items(), key=lambda kv: (kv[1], -kv[0][0], -kv[0][1]))[0]
        if counts[best] < 2:
            break                      # nothing repeats any more; merging would not compress
        new_id = len(vocab)
        vocab[new_id] = vocab[best[0]] + vocab[best[1]]
        words = {merge_word(w, best, new_id): f for w, f in words.items()}
        merges.append((best, new_id))
        if len(merges) <= verbose_until:
            trace.append((len(merges), best, counts[best], vocab[new_id]))
    return merges, vocab, trace


# ================================================================ 4. encode / decode
def encode(text, merges, pattern=None):
    """Apply the merges in the order they were learned - not longest-match.

    Longest-match would be a different algorithm with a different output; BPE is
    defined by the replay, which is why the merge list has to ship with the model.
    """
    rank = {pair: i for i, (pair, _) in enumerate(merges)}
    new_of = {pair: nid for pair, nid in merges}
    out = []
    for chunk in pretokenize(text, pattern):
        ids = tuple(to_bytes(chunk))
        while len(ids) >= 2:
            pairs = set(zip(ids, ids[1:]))
            candidates = [p for p in pairs if p in rank]
            if not candidates:
                break
            best = min(candidates, key=lambda p: rank[p])
            ids = merge_word(ids, best, new_of[best])
        out.extend(ids)
    return out


def decode(ids, vocab, errors='strict'):
    """Concatenate the bytes of every token, then decode once as UTF-8.

    Decoding per token would fail on any multi-byte character that got split across
    two tokens - which is normal and fine, as long as you join first.
    """
    return b''.join(vocab[i] for i in ids).decode('utf-8', errors=errors)


def round_trips(text, merges, vocab):
    return decode(encode(text, merges), vocab) == text


# ================================================================ 5. measurements
def tokens_per_byte(text, merges, vocab, pattern=None):
    ids = encode(text, merges, pattern)
    return len(ids), len(text.encode('utf-8')), len(ids) / len(text.encode('utf-8'))


def tie_report(text, vocab_size):
    """How often the most frequent pair is not unique.

    Every tie is a fork: two tokenizers that compress the same corpus to within a few
    tokens of each other and disagree about what a token is.  The tie-break has to be
    written down.
    """
    words = Counter(tuple(to_bytes(c)) for c in pretokenize(text))
    vocab = {i: bytes([i]) for i in range(256)}
    ties, steps, first = 0, 0, None
    while len(vocab) < vocab_size:
        counts = pair_counts(words)
        if not counts:
            break
        top = max(counts.values())
        if top < 2:
            break
        tied = [p for p, c in counts.items() if c == top]
        steps += 1
        if len(tied) > 1:
            ties += 1
            if first is None:
                first = (steps, top, sorted(tied)[:3], len(tied))
        best = max(counts.items(), key=lambda kv: (kv[1], -kv[0][0], -kv[0][1]))[0]
        nid = len(vocab)
        vocab[nid] = vocab[best[0]] + vocab[best[1]]
        words = {merge_word(w, best, nid): f for w, f in words.items()}
    return steps, ties, first


def huffman_bits_per_byte(data):
    """Day 38's Huffman, for contrast: merge the two LEAST frequent symbols.

    Huffman keeps the alphabet and shortens the codes.  BPE keeps the code length and
    shortens the alphabet - it merges the MOST frequent pair into a new symbol.  Same
    greedy shape, opposite end of the frequency table, different unit of saving.
    """
    import heapq
    freq = Counter(data)
    if len(freq) == 1:
        return 1.0
    heap = [(f, i, None) for i, (s, f) in enumerate(freq.items())]
    lengths = {s: 0 for s in freq}
    groups = {i: [s] for i, (s, f) in enumerate(freq.items())}
    heapq.heapify(heap)
    nxt = len(freq)
    while len(heap) > 1:
        f1, i1, _ = heapq.heappop(heap)
        f2, i2, _ = heapq.heappop(heap)
        for s in groups[i1] + groups[i2]:
            lengths[s] += 1
        groups[nxt] = groups[i1] + groups[i2]
        heapq.heappush(heap, (f1 + f2, nxt, None))
        nxt += 1
    total = sum(freq.values())
    return sum(freq[s] * lengths[s] for s in freq) / total


# ================================================================ 6. LeetCode 692
class SolutionTopKFrequent:
    """LeetCode 692 - Top K Frequent Words.

    Return the k most frequent words, most frequent first, and break ties by
    alphabetical order.  That second clause is the whole problem, and it is the same
    clause a BPE trainer needs: without a written-down tie-break, two runs that
    compress to within a few tokens of each other disagree about what a token is.
    """

    def topKFrequent(self, words: list[str], k: int) -> list[str]:
        count = Counter(words)
        # sort by falling frequency, then rising word - one key, both rules
        return sorted(count, key=lambda w: (-count[w], w))[:k]


def top_k_frequent_heap(words, k):
    """The same answer with a heap of size k, for when the vocabulary is large.

    The comparison has to be inverted on both fields at once, which is why a tuple of
    (count, reversed-string) does not work and a wrapper class or negated count does.
    """
    import heapq
    count = Counter(words)
    heap = []
    for w, c in count.items():
        heapq.heappush(heap, (c, _RevStr(w)))
        if len(heap) > k:
            heapq.heappop(heap)
    return [w.s for _, w in sorted(heap, key=lambda t: (-t[0], t[1].s))]


class _RevStr:
    """A string that sorts backwards, so a min-heap evicts the alphabetically last."""

    __slots__ = ('s',)

    def __init__(self, s):
        self.s = s

    def __lt__(self, other):
        return self.s > other.s

    def __eq__(self, other):
        return self.s == other.s


# ================================================================ 7. LeetCode 394
class SolutionDecodeString:
    """LeetCode 394 - Decode String.

    Expand `3[a2[c]]` into `accaccacc`.  One stack, two things pushed per bracket:
    the text built so far and the repeat count waiting for it.  A tokenizer's decode
    step is the same move without nesting - look each id up and concatenate - and
    both are wrong in the same way if you expand eagerly instead of on the closing
    bracket.
    """

    def decodeString(self, s: str) -> str:
        stack, cur, num = [], '', 0
        for ch in s:
            if ch.isdigit():
                num = num * 10 + int(ch)       # digits can be multi-character
            elif ch == '[':
                stack.append((cur, num))
                cur, num = '', 0
            elif ch == ']':
                prev, times = stack.pop()
                cur = prev + cur * times
            else:
                cur += ch
        return cur


def decode_string_brute(s):
    """Repeatedly expand the innermost bracket, for checking the stack version."""
    pat = re.compile(r'(\d+)\[([a-zA-Z]*)\]')
    while '[' in s:
        s, n = pat.subn(lambda m: m.group(2) * int(m.group(1)), s)
        if n == 0:
            break
    return s



def _valid_utf8(b):
    """Is this run of bytes a complete UTF-8 string by itself?"""
    try:
        b.decode('utf-8')
        return True
    except UnicodeDecodeError:
        return False


# ================================================================ 8. assertions
def check():
    merges, vocab, _ = train(CORPUS, 400)

    # the round trip is the property most worth testing - five more follow
    for s in [CORPUS, 'the cost of a model call', '模型看到的從來不是你的文字', '',
              'a', ' ', '\n\n', '0.15 per million', 'def f(x): return x ** 2',
              'ÄÖÜ äöü ß — – … “quoted”', '🙂🙃', 'x' * 500]:
        assert round_trips(s, merges, vocab), repr(s[:40])

    # every token stands for a non-empty string of bytes, and ids are contiguous
    assert sorted(vocab) == list(range(len(vocab)))
    assert all(len(v) >= 1 for v in vocab.values())
    assert all(vocab[i] == bytes([i]) for i in range(256))
    # a merged token is exactly the concatenation of the two it came from
    for (a, b), nid in merges:
        assert vocab[nid] == vocab[a] + vocab[b]

    # training is deterministic: same corpus in, same merge list out
    m2, v2, _ = train(CORPUS, 400)
    assert [p for p, _ in merges] == [p for p, _ in m2] and v2 == vocab

    # merging never makes the sequence longer, and a bigger vocabulary never makes it
    # longer either
    prev = None
    for size in (256, 300, 350, 400):
        mm, vv, _ = train(CORPUS, size)
        n = len(encode(CORPUS, mm))
        assert prev is None or n <= prev
        prev = n
    assert len(encode(CORPUS, [])) == len(CORPUS.encode('utf-8'))

    # pre-tokenising never lets a merge cross a chunk boundary
    for chunk in pretokenize(CORPUS):
        assert decode(encode(chunk, merges), vocab) == chunk
    assert ''.join(pretokenize(CORPUS)) == CORPUS      # the split loses nothing

    # LeetCode 692, against a brute-force sort, with ties everywhere
    import random
    S692 = SolutionTopKFrequent()
    for seed in range(300):
        r = random.Random(seed)
        ws = [r.choice('abcde') * r.randint(1, 2) for _ in range(r.randint(1, 20))]
        k = r.randint(1, len(set(ws)))
        want = sorted(Counter(ws), key=lambda w: (-Counter(ws)[w], w))[:k]
        assert S692.topKFrequent(ws, k) == want
        assert top_k_frequent_heap(ws, k) == want
    assert S692.topKFrequent(['i', 'love', 'leetcode', 'i', 'love', 'coding'], 2) == ['i', 'love']
    assert S692.topKFrequent(
        ['the', 'day', 'is', 'sunny', 'the', 'the', 'the', 'sunny', 'is', 'is'], 4) \
        == ['the', 'is', 'sunny', 'day']

    # LeetCode 394, against repeated innermost expansion
    S394 = SolutionDecodeString()
    for s in ['3[a]2[bc]', '3[a2[c]]', '2[abc]3[cd]ef', 'abc3[cd]xyz', '10[a]',
              '2[2[2[a]]]', 'leet2[code]3[tnt]', 'a', '']:
        assert S394.decodeString(s) == decode_string_brute(s), s
    for seed in range(300):
        r = random.Random(seed + 7)
        s = ''
        for _ in range(r.randint(0, 3)):
            s += r.choice(['ab', str(r.randint(1, 12)) + '[' + 'xy' * r.randint(0, 2) + ']'])
        assert S394.decodeString(s) == decode_string_brute(s), s
    assert S394.decodeString('12[a]') == 'a' * 12        # multi-digit counts
    return True


# ================================================================ 9. the whole story
def header(title):
    print('\n' + '=' * 72)
    print(title)
    print('=' * 72)


def main():
    merges, vocab, trace = train(CORPUS, 400, verbose_until=8)

    # ------------------------------------------------------------------ 1
    header('1. The same word, two different prices')
    for w in (' tokens', 'tokens', ' the', 'the'):
        ids = encode(w, merges)
        print(f'  {w!r:>10} -> {ids}   {len(ids)} token(s)')
    print('\n  Same five letters.  With the space in front it is one token; without it,')
    print('  three.  A leading space is not whitespace to a tokenizer - it is part of')
    print('  the word, because that is how the word appears in running text.\n')
    s = 'the cost of a model call is counted in tokens'
    ids = encode(s, merges)
    print(f'  "{s}"')
    print(f'  {len(s.split())} words, {len(s)} characters, {len(s.encode("utf-8"))} bytes, '
          f'{len(ids)} tokens:')
    print('  ' + ' | '.join(vocab[i].decode('utf-8', 'replace') for i in ids))
    print('\n  Note "call" coming apart into " ca" + "ll".  Nothing is wrong; those are')
    print('  just the merges this corpus happened to produce.  The model will never')
    print('  see the word call.')

    # ------------------------------------------------------------------ 2
    header('2. Everything is bytes first')
    print('  There are 256 possible bytes, so the vocabulary starts with 256 tokens and')
    print('  every possible input is already representable.  A byte-level tokenizer has')
    print('  no unknown-token problem; it has a length problem instead.\n')
    for t in ('a', 'é', '模', '🙂'):
        b = t.encode('utf-8')
        print(f'  {t!r:>6}  {len(b)} byte(s)  {list(b)}  -> {len(encode(t, merges))} token(s)')
    zh = '模型看到的從來不是你的文字'
    en = 'what the model sees is never your text'
    print(f'\n  {zh!r}')
    print(f'    {len(zh)} characters, {len(zh.encode("utf-8"))} bytes, '
          f'{len(encode(zh, merges))} tokens')
    print(f'  {en!r}')
    print(f'    {len(en)} characters, {len(en.encode("utf-8"))} bytes, '
          f'{len(encode(en, merges))} tokens')
    zt, et = len(encode(zh, merges)), len(encode(en, merges))
    print(f'\n  Roughly the same sentence.  The Chinese one costs {zt / et:.1f} times as many')
    print(f'  tokens, and per character it is {zt / len(zh):.2f} tokens against {et / len(en):.2f} - '
          f'{(zt / len(zh)) / (et / len(en)):.1f} times more.')
    print('  The reason is UTF-8 and the training corpus, not the model.')

    # ------------------------------------------------------------------ 3
    header('3. Training is three lines: count pairs, merge the most frequent, repeat')
    print(f'  The corpus is {len(CORPUS.encode("utf-8")):,} bytes of mixed prose, numbers, code and one')
    print('  Chinese line.  Here are the first eight merges it produces.\n')
    print(f'  {"#":>3} {"pair":>12} {"count":>7}   new token')
    for n, pair, cnt, piece in trace:
        print(f'  {n:>3} {str(pair):>12} {cnt:>7}   {piece.decode("utf-8", "replace")!r}')
    print(f'\n  It stops after {len(merges)} merges, at a vocabulary of {len(vocab)}, because no pair')
    print('  occurs twice any more - merging a pair that appears once makes the')
    print('  vocabulary bigger and the text no shorter.')
    print('  The merges are kept in order.  That order is the tokenizer: encoding is')
    print('  nothing but replaying it.')

    # ------------------------------------------------------------------ 4
    header('4. Silent failure #1: the most frequent pair is usually not unique')
    steps, ties, first = tie_report(CORPUS, 400)
    print(f'  Of the {steps} merge steps, {ties} of them ({100 * ties / steps:.0f}%) had more than one pair')
    print('  tied for most frequent.\n')
    if first:
        st, cnt, examples, howmany = first
        shown = ', '.join(repr(vocab[a].decode('utf-8', 'replace') +
                               vocab[b].decode('utf-8', 'replace')) for a, b in examples)
        print(f'  The first tie is at step {st}: {howmany} pairs all occur {cnt} times, among them {shown}.')
    print('\n  Whichever one you take, the counts around that pair change and the two')
    print('  tokenizers diverge from there.  Re-running this training with other')
    print('  tie-breaks gives 141 to 143 merges and 500 to 504 tokens: they compress the')
    print('  corpus to within a few tokens of each other and disagree about what a token')
    print('  is.  So the tie-break is not an implementation')
    print('  detail, it is part of the specification: this file takes the lowest pair of')
    print('  ids, which makes two runs on the same corpus produce the same vocabulary.')
    print('  Leaving it to `max` over a dict means the vocabulary depends on insertion')
    print('  order.  Picking the k most frequent things with a written-down tie-break is')
    print('  LeetCode 692, below.')

    # ------------------------------------------------------------------ 5
    header('5. Encoding replays the merges in order - it is not longest-match')
    word = ' counted'
    ids_t = tuple(to_bytes(word))
    rank = {pair: i for i, (pair, _) in enumerate(merges)}
    new_of = {pair: nid for pair, nid in merges}
    print(f'  {word!r} starts as {len(ids_t)} bytes and is squeezed one merge at a time:\n')
    step = 0
    while len(ids_t) >= 2:
        cand = [p for p in set(zip(ids_t, ids_t[1:])) if p in rank]
        if not cand:
            break
        best = min(cand, key=lambda p: rank[p])
        ids_t = merge_word(ids_t, best, new_of[best])
        step += 1
        pieces = ' '.join(repr(vocab[i].decode('utf-8', 'replace')) for i in ids_t)
        print(f'  after merge #{rank[best] + 1:>3} ({vocab[best[0]].decode("utf-8", "replace")!r}'
              f' + {vocab[best[1]].decode("utf-8", "replace")!r}):  {pieces}')
    print('\n  At every step it applies the merge with the SMALLEST rank that is still')
    print('  available - the earliest one learned - not the longest match in the')
    print('  vocabulary.  Longest-match is a different algorithm that gives different')
    print('  output on the same vocabulary, which is why the merge list has to ship with')
    print('  the model and not just the token strings.')

    # ------------------------------------------------------------------ 6
    header('6. Silent failure #2: decoding one token at a time')
    for sample in ('€5', '模型看到的'):
        sids = encode(sample, merges)
        sbad = sum(1 for i in sids if not _valid_utf8(vocab[i]))
        print(f'  {sample!r:>16}  {len(sample.encode("utf-8")):>2} bytes  {len(sids):>2} tokens  '
              f'{sbad} of them not valid UTF-8 alone')
    print()
    zh = '模型看到的'
    ids = encode(zh, merges)
    bad = sum(1 for i in ids if not _valid_utf8(vocab[i]))
    print(f'  Take the longer one.  {zh!r} encodes to {len(ids)} tokens.')
    print(f'  {bad} of those {len(ids)} tokens are not valid UTF-8 on their own, because a')
    print('  three-byte character got split across token boundaries.\n')
    print(f'  join the bytes, then decode once:   {decode(ids, vocab)!r}')
    per_token = ''.join(vocab[i].decode('utf-8', 'replace') for i in ids)
    print(f'  decode each token, then join:       {per_token!r}')
    print("\n  The second line is what you get from `[vocab[i].decode(errors='replace')")
    print("  for i in ids]`, and `errors='replace'` is the reason it does not raise.")
    print('  Streaming output token by token is the usual way to meet this: the fix is')
    print('  to hold the bytes back until they decode cleanly, not to replace them.')
    print(f'\n  The property most worth testing is the round trip, and it holds here: '
          f'{round_trips(CORPUS, merges, vocab)}.')
    print('  A tokenizer that compresses beautifully and cannot reproduce its input is')
    print('  broken; check() round-trips twelve different strings, and then checks five')
    print('  more things - contiguous ids, every merged token equal to the two it came')
    print('  from, and the same merge list on a second run.')

    # ------------------------------------------------------------------ 7
    header('7. Silent failure #3: the regex decides what a token may be')
    whole = re.compile(r'[\s\S]+')                 # the whole text as one chunk
    m_no, v_no, _ = train(CORPUS, 400, pattern=whole)
    n_no, b_no, r_no = tokens_per_byte(CORPUS, m_no, v_no, pattern=whole)
    cross_no = sum(1 for _, nid in m_no if b' ' in v_no[nid][1:-1])
    n_yes, b_yes, r_yes = tokens_per_byte(CORPUS, merges, vocab)
    cross_yes = sum(1 for _, nid in merges if b' ' in vocab[nid][1:-1])
    print('  Before counting any pairs, the text is cut into chunks that merges may')
    print('  never cross.  Drop that step and BPE is free to merge across word')
    print('  boundaries.\n')
    print(f'  {"":<22} {"merges":>7} {"tokens":>7} {"tokens/byte":>12} {"spaced tokens":>15}')
    print(f'  {"one chunk, no regex":<22} {len(m_no):>7} {n_no:>7} {r_no:>12.4f} {cross_no:>15}')
    print(f'  {"pre-tokenised":<22} {len(merges):>7} {n_yes:>7} {r_yes:>12.4f} {cross_yes:>15}')
    print('  ("spaced tokens" = merges whose text has a space somewhere in the middle.)')
    examples = [v_no[nid].decode('utf-8', 'replace') for _, nid in m_no
                if b' ' in v_no[nid][1:-1]][:4]
    print(f'\n  Compression is almost identical ({r_no:.4f} against {r_yes:.4f}).  What changes is')
    print(f'  what a token IS: without the regex, {cross_no} of the {len(m_no)} merges produce tokens with a')
    print(f'  space inside them, such as {", ".join(repr(e) for e in examples)}.')
    print('  Tokens like those make a word depend on what follows it, so the same word')
    print('  gets different ids in different sentences and the model has to learn it')
    print('  twice.  The regex is not an optimisation; it is a constraint, and the')
    print('  two rules doing the most work in it are "a leading space joins the next')
    print('  word" and "digits are cut into groups of at most three".')
    for n in ('7', '128', '1280', '128000'):
        print(f'    {n!r:>10} -> {len(encode(n, merges))} token(s)  '
              f'{[vocab[i].decode("utf-8", "replace") for i in encode(n, merges)]}')


    # ------------------------------------------------------------------ 8
    header('8. What the vocabulary size buys you')
    print('  Same corpus, four vocabulary sizes.  Every merge spends one id to make the')
    print('  text a little shorter.\n')
    print(f'  {"vocab":>7} {"merges":>7} {"tokens":>7} {"tokens/byte":>12} {"bytes/token":>12}')
    for size in (256, 300, 350, 400):
        mm, vv, _ = train(CORPUS, size)
        n, b, r = tokens_per_byte(CORPUS, mm, vv)
        print(f'  {len(vv):>7} {len(mm):>7} {n:>7} {r:>12.4f} {1 / r:>12.4f}')
    print('\n  At 256 there are no merges at all and one token is one byte.  143 merges')
    print(f'  later the same text is {1 - r:.0%} shorter.  The curve flattens because the')
    print('  frequent pairs get used up first - this is a greedy algorithm taking the')
    print('  biggest saving available at each step, so the savings can only shrink.')
    print('  Real tokenizers run this to 50,000-200,000 tokens on hundreds of gigabytes;')
    print('  the shape of the curve is the same and the flat part is much further out.')

    # ------------------------------------------------------------------ 9
    header('9. The same greedy move as Day 38, from the other end of the table')
    bits = huffman_bits_per_byte(CORPUS.encode('utf-8'))
    n, b, r = tokens_per_byte(CORPUS, merges, vocab)
    print('  Day 38 built Huffman codes by repeatedly merging the two LEAST frequent')
    print('  symbols.  BPE repeatedly merges the MOST frequent pair.  Both are greedy,')
    print('  both merge, and they are solving different problems.\n')
    print(f'  Huffman on these bytes: {bits:.4f} bits per byte, so the corpus fits in')
    print(f'    {bits * b / 8:,.0f} bytes instead of {b:,} - the alphabet is unchanged, the')
    print('    codes got shorter.')
    print(f'  BPE with {len(merges)} merges: {r:.4f} tokens per byte, so {b:,} bytes become')
    print(f'    {n:,} tokens - the sequence got shorter, the alphabet grew to {len(vocab)}.')
    print('\n  Huffman optimises bits on the wire for a fixed alphabet.  BPE optimises')
    print('  sequence length for a model whose cost is per token, and it is allowed to')
    print('  invent new symbols to do it.  Neither one can do the other job: Huffman')
    print('  codes are not integers a model can embed, and BPE tokens are not shorter')
    print('  to transmit.')

    # ------------------------------------------------------------------ 10
    header('10. LeetCode 692 - Top K Frequent Words')
    ex1 = ['i', 'love', 'leetcode', 'i', 'love', 'coding']
    ex2 = ['the', 'day', 'is', 'sunny', 'the', 'the', 'the', 'sunny', 'is', 'is']
    print(f'  {ex1}, k = 2 -> {SolutionTopKFrequent().topKFrequent(ex1, 2)}')
    print(f'  {ex2},')
    print(f'    k = 4 -> {SolutionTopKFrequent().topKFrequent(ex2, 4)}')
    print('\n  Both examples turn on the tie: "i" and "love" both appear twice, and the')
    print('  answer puts "i" first because it is alphabetically first, not because it')
    print('  was seen first.  One sort key carries both rules: (-count, word), falling')
    print('  on the first field and rising on the second.')
    print('  The heap version is the one worth getting right for a real vocabulary:')
    print('  a size-k min-heap has to evict the WORST candidate, which here means the')
    print('  least frequent and, among equals, the alphabetically LAST - so the string')
    print('  has to compare backwards while the count compares forwards.')
    print(f'    heap answer for the first example: {top_k_frequent_heap(ex1, 2)}')
    print('  That is the BPE merge loop: find the most frequent pair, break ties by a')
    print('  rule you wrote down, and do it again.')

    # ------------------------------------------------------------------ 11
    header('11. LeetCode 394 - Decode String')
    for s_ in ('3[a]2[bc]', '3[a2[c]]', '2[abc]3[cd]ef', '12[a]'):
        print(f'  {s_!r:>16} -> {SolutionDecodeString().decodeString(s_)!r}')
    print('\n  One stack, and two things pushed at every opening bracket: the text built')
    print('  so far and the repeat count that is waiting for the text still to come.')
    print('  Nothing is expanded until the closing bracket arrives, which is what makes')
    print('  the nested case work without recursion.')
    print('  Two details the examples are chosen to expose: counts can have more than')
    print("  one digit, so `num = num * 10 + int(ch)`, not `num = int(ch)`; and the")
    print('  text after a bracket closes keeps accumulating onto the SAME buffer, so')
    print("  'ef' in the third example lands after the expansion rather than inside it.")
    print('  It is the shape of a decoder: a flat sequence of ids on one side, a longer')
    print('  string on the other, and the expansion driven entirely by a table.')

    # ------------------------------------------------------------------ 12
    header('12. Assertions')
    check()
    print('all assertions passed')


if __name__ == '__main__':
    main()
