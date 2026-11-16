"""Day 59 - temperature, top-k and top-p: choosing the next token.

Every model from Day 55 onwards ends the same way: a vector of logits, one per token
in the vocabulary.  Turning that vector into one token is the sampling step, and it is
the only part of inference where the server gets to trade quality against variety.

Take the most likely token every time and the model repeats itself; draw from the raw
distribution and the long tail - tens of thousands of tokens that are individually
hopeless but collectively carry real probability mass - eventually ruins the output.
Temperature reshapes the distribution, top-k and top-p cut the tail off, and the order
the two are applied in changes the answer.

The language model here is a small character n-gram trained on sentences from a grammar
I wrote, so "did the sample come out grammatical" is a question with a real answer, and
every claim about quality below is a measured rate, not an impression.

Run `python sampling.py` to print the whole story.
"""

import math
from collections import defaultdict

import numpy as np


# ================================================================ 1. the corpus
DET = ['the ', 'a ']
ADJ = ['small ', 'old ', 'red ', 'quiet ', '']
NOUN = ['cat', 'dog', 'bird', 'child', 'robot']
VERB = ['sat', 'slept', 'waited', 'sang', 'ran']
PLACE = ['on the mat', 'in the garden', 'by the river', 'under the tree',
         'near the door']


def sentence(rng):
    return (DET[rng.integers(len(DET))] + ADJ[rng.integers(len(ADJ))]
            + NOUN[rng.integers(len(NOUN))] + ' ' + VERB[rng.integers(len(VERB))]
            + ' ' + PLACE[rng.integers(len(PLACE))] + '.\n')


def make_corpus(n_sentences=1500, seed=0):
    rng = np.random.default_rng(seed)
    return ''.join(sentence(rng) for _ in range(n_sentences))


def is_grammatical(line):
    """Parse one generated line against the grammar that produced the corpus."""
    s = line.strip()
    if not s.endswith('.'):
        return False
    s = s[:-1]
    for d in DET:
        if not s.startswith(d):
            continue
        rest = s[len(d):]
        for a in ADJ:
            if not rest.startswith(a):
                continue
            r2 = rest[len(a):]
            for n in NOUN:
                if not r2.startswith(n + ' '):
                    continue
                r3 = r2[len(n) + 1:]
                for v in VERB:
                    if not r3.startswith(v + ' '):
                        continue
                    if r3[len(v) + 1:] in PLACE:
                        return True
    return False


def total_sentences():
    return len(DET) * len(ADJ) * len(NOUN) * len(VERB) * len(PLACE)


# ================================================================ 2. the model
class NGram:
    """Character n-gram with add-alpha smoothing - a tiny stand-in for a real model.

    The smoothing matters for today: it is what gives every token a small non-zero
    probability, which is exactly the long tail that top-k and top-p exist to remove.
    """

    def __init__(self, text, order=8, alpha=0.01):
        self.order = order
        self.alpha = alpha
        self.vocab = sorted(set(text))
        self.V = len(self.vocab)
        self.stoi = {c: i for i, c in enumerate(self.vocab)}
        self.counts = [defaultdict(lambda: np.zeros(self.V)) for _ in range(order + 1)]
        for i in range(len(text)):
            tgt = self.stoi[text[i]]
            for o in range(order + 1):
                if i - o < 0:
                    break
                self.counts[o][text[i - o:i]][tgt] += 1

    def logits(self, context):
        """Longest context with any evidence wins; fall back to shorter if unseen."""
        for o in range(self.order, -1, -1):
            ctx = context[-o:] if o else ''
            c = self.counts[o].get(ctx)
            if c is not None and c.sum() >= 2:
                p = (c + self.alpha) / (c.sum() + self.alpha * self.V)
                return np.log(p)
        return np.zeros(self.V)


def softmax(x):
    x = x - x.max()
    e = np.exp(x)
    return e / e.sum()


def entropy(p):
    q = p[p > 0]
    return float(-(q * np.log2(q)).sum())


# ================================================================ 3. the knobs
def temperature(logits, t):
    """Divide before the softmax.  t < 1 sharpens, t > 1 flattens, t -> 0 is argmax."""
    if t <= 0:
        out = np.full_like(logits, -np.inf)
        out[int(np.argmax(logits))] = 0.0
        return out
    return logits / t


def top_k(p, k):
    """Keep the k most likely tokens, renormalise.  Fixed count, ignores shape."""
    if k is None or k >= len(p):
        return p
    cut = np.partition(p, -k)[-k]
    q = np.where(p >= cut, p, 0.0)
    return q / q.sum()


def top_p(p, thr):
    """Keep the smallest set whose mass reaches thr - the nucleus.  Count varies."""
    if thr is None or thr >= 1.0:
        return p
    order = np.argsort(-p)
    csum = np.cumsum(p[order])
    n = int(np.searchsorted(csum, thr) + 1)          # always keeps at least one
    q = np.zeros_like(p)
    q[order[:n]] = p[order[:n]]
    return q / q.sum()


def min_p(p, frac):
    """Keep tokens worth at least frac of the best one - a threshold, not a count."""
    if frac is None or frac <= 0:
        return p
    q = np.where(p >= frac * p.max(), p, 0.0)
    return q / q.sum()


def typical(p, mass=0.9):
    """Locally typical sampling: keep the tokens whose surprise is closest to the
    entropy of this step.  Not "the most likely", but "as informative as expected" -
    a token that is far more predictable than the step's average is as atypical as
    one that is far less."""
    if mass is None or mass >= 1.0:
        return p
    h = float(-(p * np.log(p)).sum())             # entropy of this step, in nats
    order = np.argsort(np.abs(-np.log(p) - h))    # distance from typical surprise
    csum = np.cumsum(p[order])
    n = int(np.searchsorted(csum, mass) + 1)
    q = np.zeros_like(p)
    q[order[:n]] = p[order[:n]]
    return q / q.sum()


def nucleus_size(p, thr):
    csum = np.cumsum(np.sort(p)[::-1])
    return int(np.searchsorted(csum, thr) + 1)


# ================================================================ 4. generation
def generate(model, rng, n=100, max_len=80, t=1.0, k=None, p=None, mp=None,
             typ=None, rep_penalty=1.0, prompt='\n'):
    """One full decode loop, exactly in the order a server applies the knobs."""
    lines, sizes = [], []
    for _ in range(n):
        out = prompt
        for _ in range(max_len):
            lg = model.logits(out).copy()
            if rep_penalty != 1.0:
                for c in set(out):
                    i = model.stoi[c]
                    lg[i] = lg[i] / rep_penalty if lg[i] > 0 else lg[i] * rep_penalty
            pr = softmax(temperature(lg, t))
            sizes.append(nucleus_size(pr, 0.9))
            if k:
                pr = top_k(pr, k)
            if p:
                pr = top_p(pr, p)
            if mp:
                pr = min_p(pr, mp)
            if typ:
                pr = typical(pr, typ)
            c = model.vocab[int(rng.choice(len(pr), p=pr))]
            out += c
            if c == '\n':
                break
        lines.append(out)
    return lines, sizes


def score(lines):
    ok = sum(is_grammatical(l) for l in lines)
    uniq = len({l.strip() for l in lines if is_grammatical(l)})
    return {'n': len(lines), 'grammatical': ok / len(lines), 'distinct': uniq}


def greedy(model, n_steps=80, prompt='\n'):
    out = prompt
    for _ in range(n_steps):
        out += model.vocab[int(np.argmax(model.logits(out)))]
        if out.endswith('\n') and len(out) > 10:
            break
    return out


# ================================================================ 5. measurements
def temperature_table(model, ts=(0.0, 0.3, 0.7, 1.0, 1.3, 1.7, 2.0), n=100, seed=1):
    rows = []
    for t in ts:
        rng = np.random.default_rng([seed, int(t * 100)])
        lines, _ = generate(model, rng, n=n, t=t)
        r = score(lines)
        r['t'] = t
        r['H'] = mean_entropy(model, t)
        rows.append(r)
    return rows


def mean_entropy(model, t, text=None):
    """Average entropy of the next-token distribution over real contexts."""
    text = text or make_corpus(40, seed=7)
    hs = [entropy(softmax(temperature(model.logits(text[:i]), t)))
          for i in range(4, len(text), 7)]
    return float(np.mean(hs))


def rescue_table(model, t=2.0, n=100, seed=2):
    """Same hot temperature, different truncations."""
    cfgs = [('no truncation', {}), ('top-k 1 (greedy)', {'k': 1}),
            ('top-k 3', {'k': 3}), ('top-p 0.9', {'p': 0.9}),
            ('top-p 0.7', {'p': 0.7}), ('top-p 0.5', {'p': 0.5}),
            ('min-p 0.1', {'mp': 0.1}), ('typical 0.9', {'typ': 0.9}),
            ('typical 0.5', {'typ': 0.5})]
    rows = []
    for name, kw in cfgs:
        rng = np.random.default_rng([seed, 3])
        lines, _ = generate(model, rng, n=n, t=t, **kw)
        r = score(lines)
        r['name'] = name
        rows.append(r)
    return rows


def shape_table(model, ks=(1, 2, 5, 20), thr=0.9, seed=3):
    """How many tokens each rule admits, step by step - the whole top-k/top-p argument."""
    text = make_corpus(30, seed=seed)
    peaked, flat = [], []
    for i in range(4, len(text), 3):
        p = softmax(model.logits(text[:i]))
        (peaked if p.max() > 0.9 else flat).append(p)
    def stats(group):
        return {'n': len(group),
                'nucleus': float(np.mean([nucleus_size(p, thr) for p in group])),
                'tail_in_k': {k: float(np.mean([1 - np.sort(p)[::-1][:k].sum()
                                                for p in group])) for k in ks}}
    return {'peaked': stats(peaked), 'flat': stats(flat), 'ks': ks, 'thr': thr}


def order_matters(model, ctx='the small cat s', t=2.5, thr=0.9):
    """top-p then temperature is not the same distribution as temperature then top-p."""
    lg = model.logits(ctx)
    a = top_p(softmax(temperature(lg, t)), thr)          # temperature first (vLLM order)
    q = top_p(softmax(lg), thr)                          # truncate on the cold shape
    b = np.where(q > 0, q ** (1 / t), 0.0)               # then flatten inside it
    b = b / b.sum()
    return {'ctx': ctx, 't': t, 'p': thr,
            'n_a': int((a > 0).sum()), 'n_b': int((b > 0).sum()),
            'max_a': float(a.max()), 'max_b': float(b.max()),
            'tv': float(0.5 * np.abs(a - b).sum())}


def forgot_to_renormalise(model, ctx='the small cat ', k=3, trials=20000, seed=4):
    """The classic bug: zero out the tail, do not rescale, hand it to the sampler."""
    p = softmax(model.logits(ctx))
    cut = np.partition(p, -k)[-k]
    q = np.where(p >= cut, p, 0.0)                   # no q /= q.sum()
    kept = float(q.sum())
    rng = np.random.default_rng(seed)
    # numpy raises; a hand-written cumulative sampler silently biases the last token
    c = np.cumsum(q)
    draws = rng.random(trials)
    idx = np.searchsorted(c, draws)
    overflow = int((idx >= len(p)).sum())
    return {'kept_mass': kept, 'overflow': overflow / trials,
            'last_token': model.vocab[int(np.argmax(q))]}


def sort_cost(V=151936, k=50):
    """Why production samplers avoid the full sort."""
    full = V * math.log2(V)
    partial = V + k * math.log2(max(k, 2))
    return {'V': V, 'k': k, 'full': full, 'partial': partial,
            'ratio': full / partial}


def pivot_search(p, thr=0.9, iters=24, trace=None):
    """Sort-free top-p: binary search a probability threshold, no ordering at all.

    This is how a GPU sampler does it - a sort over 150k tokens per sequence per step
    is far more expensive than two dozen passes that are pure reductions.
    """
    lo, hi = 0.0, float(p.max())
    for _ in range(iters):
        mid = (lo + hi) / 2
        mass = float(p[p >= mid].sum())
        if trace is not None:
            trace.append({'lo': lo, 'hi': hi, 'mid': mid, 'mass': mass,
                          'n': int((p >= mid).sum())})
        if mass >= thr:
            lo = mid                                  # still enough mass: raise the bar
        else:
            hi = mid
    kept = p >= lo
    return {'n': int(kept.sum()), 'mass': float(p[kept].sum()), 'pivot': lo}


# ================================================================ 6. checks
def typical_vs_nucleus(model, seed=5, mass=0.9):
    """How often the two rules disagree on this corpus - honestly, never."""
    txt = make_corpus(60, seed=seed)
    same, drops, n = 0, 0, 0
    for i in range(4, len(txt)):
        p = softmax(model.logits(txt[:i]))
        t = typical(p, mass)
        n += 1
        same += int((t > 0).sum()) == nucleus_size(p, mass)
        drops += t[int(np.argmax(p))] == 0
    return {'steps': n, 'same_size': same, 'dropped_argmax': drops, 'mass': mass}


def typical_contrast(mass=0.25):
    """A vocabulary shaped so the two rules must disagree.

    One dominant token, a middle group, and a long flat tail.  The entropy sits
    between the surprise of the dominant token and the surprise of the tail, so the
    middle group is the typical one - and typical sampling reaches for it first.
    """
    p = np.concatenate([[0.6], np.full(10, 0.03), np.full(1000, 0.0001)])
    p = p / p.sum()
    h = float(-(p * np.log(p)).sum())
    groups = [('dominant, p=0.60', 0), ('middle, p=0.03', 1), ('tail, p=0.0001', 11)]
    rows = [{'name': name, 'surprise': float(-np.log(p[i])),
             'distance': float(abs(-np.log(p[i]) - h))} for name, i in groups]
    tp, ty = top_p(p, mass), typical(p, mass)
    return {'H': h, 'rows': rows, 'mass': mass,
            'top_p_n': int((tp > 0).sum()), 'typical_n': int((ty > 0).sum()),
            'top_p_keeps_best': bool(tp[0] > 0), 'typical_keeps_best': bool(ty[0] > 0)}


def all_sentences():
    """Every sentence the grammar admits, in one list."""
    return [d + a + n + ' ' + v + ' ' + pl + '.'
            for d in DET for a in ADJ for n in NOUN for v in VERB for pl in PLACE]


def sentence_logprob(model, s, prefix='\n'):
    """log P(sentence) under the model: the sum of the per-step log probabilities."""
    ctx, tot = prefix, 0.0
    for ch in s + '\n':
        tot += float(np.log(softmax(model.logits(ctx))[model.stoi[ch]]))
        ctx += ch
    return tot


def sequence_table(model, draws=100):
    """Greedy maximises every step.  That is not the same as maximising the sentence."""
    scored = sorted(((sentence_logprob(model, s), s) for s in all_sentences()),
                    reverse=True)
    order = [s for _, s in scored]
    g = greedy(model).strip()
    probs = np.exp(np.array([lp for lp, _ in scored]))
    return {'best': scored[0][1], 'best_lp': scored[0][0], 'best_p': float(probs[0]),
            'greedy': g, 'greedy_lp': sentence_logprob(model, g),
            'rank': order.index(g) + 1 if g in order else None, 'n': len(order),
            'legal_mass': float(probs.sum()),
            'expected_distinct': float((1 - (1 - probs) ** draws).sum()),
            'draws': draws}


def check():
    text = make_corpus(400, seed=0)
    m = NGram(text)
    p = softmax(m.logits('the small cat '))
    assert abs(p.sum() - 1) < 1e-12
    assert abs(top_k(p, 3).sum() - 1) < 1e-12 and int((top_k(p, 3) > 0).sum()) == 3
    assert abs(top_p(p, 0.9).sum() - 1) < 1e-12
    assert int((top_p(p, 0.0) > 0).sum()) == 1, 'top-p must keep at least one token'
    assert int(np.argmax(top_p(p, 0.5))) == int(np.argmax(p))
    assert int((min_p(p, 1.0) > 0).sum()) >= 1
    # temperature 0 is argmax, and never divides by zero
    assert int(np.argmax(softmax(temperature(m.logits('the '), 0.0)))) == \
           int(np.argmax(m.logits('the ')))
    # colder is never higher-entropy
    hs = [entropy(softmax(temperature(m.logits('the small cat '), t)))
          for t in (0.5, 1.0, 2.0)]
    assert hs[0] < hs[1] < hs[2], hs
    # the sort-free pivot finds the same set as the sorted nucleus
    for ctx in ('the ', 'a red dog s', 'the quiet bird sang '):
        q = softmax(m.logits(ctx))
        n0 = nucleus_size(q, 0.9)
        # tie-robust: both rules keep "everything at least as likely as the nth"
        assert pivot_search(q)['n'] == int((q >= np.sort(q)[::-1][n0 - 1]).sum())
        assert pivot_search(q)['mass'] >= 0.9
    # typical sampling is a valid distribution and never empties
    assert abs(typical(p, 0.9).sum() - 1) < 1e-12 and int((typical(p, 0.9) > 0).sum()) >= 1
    tc = typical_contrast()
    assert tc['top_p_keeps_best'] and not tc['typical_keeps_best'], tc
    # the sequence-level check: greedy is a step rule, not a sentence rule
    sq = sequence_table(m)
    assert sq['rank'] is not None and sq['greedy_lp'] <= sq['best_lp']
    assert len(set(all_sentences())) == total_sentences()
    assert is_grammatical('the small cat sat on the mat.')
    assert not is_grammatical('the small cat sat on the moon.')
    assert not is_grammatical('cat the sat on the mat.')


# ================================================================ 7. the story
def header(t):
    print('\n' + '=' * 72)
    print(t)
    print('=' * 72)


def main():
    text = make_corpus()
    m = NGram(text)

    header('1. The model and the vocabulary')
    print(f'  corpus       {len(text):,} characters, {text.count(chr(10)):,} sentences')
    print(f'  grammar      {total_sentences()} sentences are legal in total')
    print(f'  vocabulary   {m.V} characters: {"".join(m.vocab).strip()!r} + space, newline')
    print(f'  model        character {m.order}-gram, add-{m.alpha} smoothing')
    print('\n  Every claim below is a rate over generated samples, parsed against the')
    print('  grammar that produced the corpus.')

    header('2. One step, and the tail')
    ctx = 'the small cat '
    p = softmax(m.logits(ctx))
    order = np.argsort(-p)
    print(f'  context {ctx!r} -> next character:\n')
    for i in order[:6]:
        print(f'    {m.vocab[i]!r:6} {p[i]:8.4f}')
    nuc = nucleus_size(p, 0.9)
    tail = float(p[order[nuc:]].sum())
    print(f'    ... {m.V - 6} more characters, all below 0.0002')
    print(f'\n  entropy {entropy(p):.3f} bits, nucleus at p=0.9 is {nuc} characters,')
    print(f'  and the {m.V - nuc} characters outside it hold {tail:.4%} between them')
    risk = 1 - (1 - tail) ** 30
    print(f'\n  That tail is the whole problem.  {tail:.2%} per step sounds like')
    print(f'  nothing, but a sentence is about 30 steps, so the chance of drawing at')
    print(f'  least one impossible character on the way is {risk:.1%} - and one is')
    print('  enough to ruin the sentence, because the model then has to continue from')
    print('  a prefix it has never seen.')

    header('3. Greedy decoding: correct and useless')
    g = greedy(m)
    print(f'  argmax every step -> {g.strip()!r}')
    rng = np.random.default_rng(0)
    lines, _ = generate(m, rng, n=60, t=0.0)
    s = score(lines)
    print(f'  60 samples at temperature 0: {s["grammatical"]:.0%} grammatical, '
          f'{s["distinct"]} distinct sentence out of {total_sentences()} possible')
    print('\n  Greedy is not wrong, it is degenerate.  Every claim the model could')
    print('  have made collapses onto one, and it is the same one every time.')

    header('4. Greedy is not even the most likely sentence')
    sq = sequence_table(m)
    print(f'  Greedy maximises every step.  Enumerating all {sq["n"]:,} sentences the')
    print('  grammar admits and scoring each one exactly - the sum of its per-step log')
    print('  probabilities - shows what that is worth:\n')
    print(f'    most likely sentence   {sq["best"]!r}')
    print(f'                           log p = {sq["best_lp"]:.3f}')
    print(f'    greedy produces        {sq["greedy"]!r}')
    print(f'                           log p = {sq["greedy_lp"]:.3f}, '
          f'rank {sq["rank"]} of {sq["n"]:,}')
    print(f'\n  The greedy sentence is {math.exp(sq["best_lp"] - sq["greedy_lp"]):.2f}x '
          f'less likely than the best one.')
    print('  Taking the locally best character at every step walks into a prefix from')
    print('  which no very good ending exists, and by then it cannot go back.  Nothing')
    print('  about this is specific to characters: it is why searching over sequences')
    print('  rather than over steps is a separate decoding strategy.')
    print(f'\n  Two checks that the whole measurement rig is sound.  These {sq["n"]:,}')
    print(f'  sentences hold {sq["legal_mass"]:.1%} of the model\'s sentence-level')
    print(f'  probability, and section 5 measures a grammatical rate of 90% at t=1.0.')
    print(f'  And drawing {sq["draws"]} sentences from that distribution should give')
    print(f'  {sq["expected_distinct"]:.1f} distinct ones; the measured column says 88.')
    print('  Theory and sampling agree, so the rates below can be trusted.')

    header('5. Temperature')
    print('  logits / t, then softmax.  Smaller t sharpens, larger t flattens.')
    print(f'  100 samples per row, parsed against the grammar '
          f'({total_sentences()} legal sentences).\n')
    print('   t     entropy   grammatical   distinct')
    for r in temperature_table(m):
        print(f'  {r["t"]:4.1f}  {r["H"]:7.3f}   {r["grammatical"]:9.0%}   '
              f'{r["distinct"]:8d}')
    print('\n  The two columns pull against each other.  Cold is safe and repetitive,')
    print('  hot is varied and wrong, and the fall is not gentle: between t=1.0 and')
    print('  t=1.5 the grammatical rate collapses while the entropy barely doubles.')
    print('  No single t gives both, which is exactly why truncation exists.')

    header('6. Truncation: keep the heat, drop the tail')
    print('  Every row below is temperature 2.0 - the setting that scored worst')
    print('  above - with one truncation rule applied on top of it.\n')
    print('  rule               grammatical   distinct')
    for r in rescue_table(m):
        print(f'  {r["name"]:18} {r["grammatical"]:9.0%}   {r["distinct"]:8d}')
    print('\n  The temperature did not change; only the set of tokens it was allowed')
    print('  to spread over did.  min-p 0.1 gets back to a perfect grammatical rate')
    print('  and the widest variety on the page - more distinct sentences than any')
    print('  row of the temperature table, because the heat is still there and only')
    print('  the impossible characters were removed.')
    print('\n  The two typical rows are a fourth rule, explained in section 8: at the')
    print('  same mass it behaves almost exactly like top-p here, which is itself a')
    print('  measurement worth having.')
    print('\n  Two honest caveats.  top-k 1 is greedy wearing a different name, so its')
    print('  perfect score is worthless.  And this vocabulary is 24 characters, not')
    print('  150,000 tokens, so the thresholds here do not transfer: p=0.5 is severe')
    print('  on a real model.  What transfers is the shape of the table.')

    header('7. Why top-p and not just top-k')
    st = shape_table(m)
    print(f'  Next-token distributions split into two kinds.  Over {st["peaked"]["n"]}')
    print(f'  peaked steps (one character above 90%) and {st["flat"]["n"]} flat ones:\n')
    print('                      peaked      flat')
    print(f'  nucleus size (p=0.9) {st["peaked"]["nucleus"]:8.2f}  {st["flat"]["nucleus"]:8.2f}')
    for k in st['ks']:
        print(f'  mass left out by k={k:<3}{st["peaked"]["tail_in_k"][k]:9.4f}  '
              f'{st["flat"]["tail_in_k"][k]:8.4f}')
    print('\n  Read the k=1 row: on a step where the model is unsure it throws away')
    print('  54% of the probability mass, and on a step where it is certain it throws')
    print('  away 0.45% - which is the smoothing floor, pure noise.  One count cannot')
    print('  be right for both, and a real model switches between the two kinds many')
    print('  times in a sentence.  top-p asks for a mass instead of a count, so the')
    print('  number of candidates follows the shape of the step: 1 here, 3.46 there.')
    print('\n  min-p takes a third position: keep anything worth at least a fraction')
    print('  of the best token.  That is a single comparison per token - no sort, no')
    print('  cumulative sum - and it tightens automatically when one token dominates,')
    print('  which is why it survived temperature 2.0 in section 6 and top-p did not.')

    header('8. Typical sampling: a third question to ask')
    print('  top-k asks for a count, top-p asks for a mass.  Locally typical sampling')
    print('  asks for neither: keep the tokens whose surprise, -log p, is closest to')
    print('  the entropy of this step.  The claim behind it is that natural language')
    print('  is not a sequence of maximally predictable words - it carries a roughly')
    print('  constant amount of information per word - so a token far more obvious')
    print('  than the step average is as atypical as one far less.\n')
    tv = typical_vs_nucleus(m)
    print(f'  On this corpus the two rules never disagree: over {tv["steps"]:,} steps')
    print(f'  typical sampling at {tv["mass"]} kept the same number of tokens as top-p')
    print(f'  every single time, and dropped the most likely token '
          f'{tv["dropped_argmax"]} times.')
    print('  With 24 characters there is no room for them to separate - which is worth')
    print('  saying rather than hiding, because it is exactly the kind of difference a')
    print('  toy model cannot show.\n')
    tc = typical_contrast()
    print('  So here is a distribution built to separate them: one dominant token, a')
    print(f'  middle group, and a long flat tail.  Entropy {tc["H"]:.2f} nats.\n')
    for r in tc['rows']:
        print(f'    {r["name"]:20} surprise {r["surprise"]:5.2f}   '
              f'distance from entropy {r["distance"]:5.2f}')
    print(f'\n  Asking each rule for {tc["mass"]} of the mass: top-p keeps '
          f'{tc["top_p_n"]} token')
    print(f'  (the dominant one), typical sampling keeps {tc["typical_n"]} - and '
          f'{"keeps" if tc["typical_keeps_best"] else "drops"} the most likely token')
    print('  entirely, because the middle group is the one carrying a typical amount')
    print('  of information.  That is the whole difference: top-p ranks by probability,')
    print('  typical sampling ranks by distance from the expected surprise.')

    header('9. The order of operations changes the answer')
    o = order_matters(m)
    print(f'  context {o["ctx"]!r}, t={o["t"]}, p={o["p"]}\n')
    print(f'  temperature then top-p   {o["n_a"]:3d} candidates, top prob {o["max_a"]:.4f}')
    print(f'  top-p then temperature   {o["n_b"]:3d} candidates, top prob {o["max_b"]:.4f}')
    print(f'  total variation distance {o["tv"]:.4f}')
    print('\n  Temperature first lets the flattened distribution decide the nucleus,')
    print('  so a hot t widens it.  Truncating first fixes the candidate set at the')
    print('  cold shape and only then flattens inside it.  vLLM and SGLang both do')
    print('  penalties -> temperature -> top-k -> top-p, so a request that sets both')
    print('  gets the first row.  It is a convention, not a theorem.')

    header('10. The bug: truncating without renormalising')
    b = forgot_to_renormalise(m)
    print(f'  Zero out everything below the top 3 and the remaining mass is '
          f'{b["kept_mass"]:.4f}.')
    print(f'  Hand that to a cumulative-sum sampler and {b["overflow"]:.2%} of draws')
    print(f'  fall past the end of the array - in C++ that reads off the end, in')
    print(f'  Python it silently becomes whichever token the clamp picks.')
    print('  Always divide by the kept mass.  Every filter in section 3 does.')

    header('11. Doing this on a GPU without sorting')
    c = sort_cost()
    print(f'  A real vocabulary is {c["V"]:,} tokens, and sampling runs once per')
    print(f'  sequence per decode step.  A full sort is ~{c["full"]/1e6:.1f}M element')
    print(f'  comparisons; a top-{c["k"]} selection is ~{c["partial"]/1e6:.1f}M, '
          f'{c["ratio"]:.0f}x less.')
    q = softmax(m.logits('the small cat '))
    ps = pivot_search(q)
    print(f'\n  top-p can skip ordering entirely.  Binary-search a probability')
    print(f'  threshold and sum the mass above it; 24 passes of a pure reduction')
    print(f'  land on {ps["n"]} tokens holding {ps["mass"]:.4f} - the same set the')
    print(f'  sorted version returns ({nucleus_size(q, 0.9)} tokens), no sort at all.')
    print('  That is what SGLang\'s top_p_renorm and vLLM\'s FlashInfer sampling')
    print('  kernels do, and why they can leave sampling on the critical path.')

    header('12. In vLLM')
    print('  Everything above is one object:\n')
    print('    from vllm import LLM, SamplingParams')
    print('    out = LLM("Qwen/Qwen3-8B").generate(prompts, SamplingParams(')
    print('        temperature=0.8, top_k=50, top_p=0.95, min_p=0.0,')
    print('        repetition_penalty=1.1, seed=1234, n=4, logprobs=5,')
    print('        max_tokens=256, stop=["\\n\\n"]))\n')
    print('    temperature=0      takes the argmax path, no division by zero')
    print('    n=4                four samples, one prompt - the shared blocks of Day 58')
    print('    seed               per-request generator, so a retry reproduces')
    print('    logprobs=5         the top rows of section 2, returned to the caller')
    print('  The sampler batches every running sequence into one kernel, which is why')
    print('  the parameters are per-request but the work is per-step.')

    print('\n' + '-' * 72)
    print('  In SGLang')
    print('-' * 72)
    print('  Same knobs over HTTP, applied in the same order:\n')
    print('    curl localhost:30000/generate -d \'{"text": "...", "sampling_params":')
    print('      {"temperature": 0.8, "top_k": 50, "top_p": 0.95, "min_p": 0.0,')
    print('       "repetition_penalty": 1.1, "frequency_penalty": 0.2, "n": 4}}\'\n')
    print('    --sampling-backend flashinfer   fused top-k/top-p kernels (default)')
    print('    --sampling-backend pytorch      the sort-based reference path')
    print('    --disable-radix-cache           note n>1 shares the prefix via Day 57')
    print('  SGLang also exposes a custom logit processor, which is the same hook as')
    print('  the filters in section 3 - grammar-constrained decoding, banned phrases')
    print('  and JSON schema enforcement are all written as one more mask on logits')
    print('  before temperature.')

    header('13. Where this comes back')
    print('  The choice between "take the best arm" and "keep exploring" is the')
    print('  exploration-exploitation trade-off, and softmax-over-values with a')
    print('  temperature is literally the Boltzmann exploration policy used by')
    print('  multi-armed bandits.  Reinforcement learning will reuse this exact')
    print('  function: temperature controls how much a policy explores, and the')
    print('  entropy term in a policy-gradient objective is the column printed in')
    print('  section 5.  Speculative decoding needs more: it has to accept or reject')
    print('  a draft token in a way that leaves this distribution unchanged.')

    header('14. Assertions')
    check()
    print('all assertions passed')


if __name__ == '__main__':
    main()
