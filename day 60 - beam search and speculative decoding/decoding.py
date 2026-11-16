"""Day 60 - beam search and speculative decoding: searching, and cheating.

The previous day stopped at a result worth taking seriously: greedy decoding, which
takes the highest-probability character at every step, produces only the 44th most
likely sentence out of the 1,250 the grammar admits.  Picking the best token at each
step is not the same as picking the best sentence, and no amount of tuning temperature
or top-p fixes that - it is a search problem, not a sampling problem.

Beam search is the search.  Speculative decoding is the opposite kind of idea: do not
change what comes out at all, just produce it with fewer calls to the expensive model
by letting a cheap one guess ahead.

Both are measured here against ground truth, because this grammar is small enough to
enumerate: 1,250 sentences, each scored exactly.  So "did beam search find the best
sentence" and "did speculative decoding change the distribution" both have real
answers rather than plausible-sounding ones.

Run `python decoding.py` to print the whole story.
"""

import math
from collections import defaultdict

import numpy as np


# ================================================================ 1. the model
# Same grammar and the same character 8-gram as the previous day, kept here so this
# file runs on its own.
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


def all_sentences():
    return [d + a + n + ' ' + v + ' ' + pl + '.'
            for d in DET for a in ADJ for n in NOUN for v in VERB for pl in PLACE]


def is_grammatical(line):
    s = line.strip()
    return s in set(all_sentences())


class NGram:
    """Character n-gram with add-alpha smoothing; order controls how much it knows."""

    def __init__(self, text, order=8, alpha=0.01):
        self.order, self.alpha = order, alpha
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
        self._cache = {}

    def logits(self, context):
        key = context[-self.order:]
        hit = self._cache.get(key)
        if hit is not None:
            return hit
        for o in range(self.order, -1, -1):
            c = self.counts[o].get(context[-o:] if o else '')
            if c is not None and c.sum() >= 2:
                out = np.log((c + self.alpha) / (c.sum() + self.alpha * self.V))
                break
        else:
            out = np.zeros(self.V)
        self._cache[key] = out
        return out

    def probs(self, context):
        return softmax(self.logits(context))


def softmax(x):
    e = np.exp(x - x.max())
    return e / e.sum()


def logprob(model, s, prefix='\n'):
    """log P(sentence): the sum of its per-step log probabilities."""
    ctx, tot = prefix, 0.0
    for ch in s + '\n':
        tot += float(np.log(model.probs(ctx)[model.stoi[ch]]))
        ctx += ch
    return tot


def exact_ranking(model):
    """Ground truth: every sentence the grammar admits, scored and sorted."""
    return sorted(((logprob(model, s), s) for s in all_sentences()), reverse=True)


# ================================================================ 2. the two searches
def greedy(model, max_len=80, prefix='\n'):
    out = prefix
    for _ in range(max_len):
        out += model.vocab[int(np.argmax(model.logits(out)))]
        if out.endswith('\n') and len(out) > 2:
            break
    return out.strip()


def beam_search(model, width=4, max_len=80, prefix='\n', alpha=0.0, n_best=1):
    """Keep the `width` best prefixes at every step instead of only the best one.

    alpha is the length penalty exponent: the score becomes logp / len**alpha, so
    alpha=0 scores raw probability and alpha=1 scores the per-character average.
    """
    beams = [(0.0, prefix, 0)]                     # (logp, text, length)
    done = []
    for _ in range(max_len):
        cand = []
        for lp, text, n in beams:
            lg = np.log(model.probs(text))
            # stable sort: exact ties happen in this model, and an unstable sort
            # would make a width-1 beam disagree with greedy for no real reason
            for i in np.argsort(-lg, kind='stable')[:width]:
                ch = model.vocab[i]
                if ch == '\n':
                    if len(text) > 1:              # not the empty "sentence"
                        done.append((lp + float(lg[i]), text[1:], n + 1))
                else:
                    cand.append((lp + float(lg[i]), text + ch, n + 1))
        if not cand:
            break
        key = lambda b: b[0] / (b[2] ** alpha if alpha else 1.0)
        beams = sorted(cand, key=key, reverse=True)[:width]
        if len(done) >= n_best and \
                max(key(d) for d in done) >= max(key(b) for b in beams):
            break                                   # no live beam can still win
    done.sort(key=lambda b: b[0] / (b[2] ** alpha if alpha else 1.0), reverse=True)
    out = [(lp, text) for lp, text, _ in done[:n_best]]
    return out[0] if n_best == 1 else out


def beam_table(model, widths=(1, 2, 4, 8, 16, 32)):
    rank_of = {s: i + 1 for i, (_, s) in enumerate(exact_ranking(model))}
    rows = []
    for w in widths:
        lp, text = beam_search(model, width=w)
        rows.append({'width': w, 'text': text, 'lp': lp,
                     'rank': rank_of.get(text), 'legal': is_grammatical(text)})
    return rows


def beam_monotonicity(model, widths=range(1, 17)):
    """Does a wider beam always give a better answer?  No.  Find where it does not."""
    rows = []
    for w in widths:
        lp, text = beam_search(model, width=w)
        rows.append({'width': w, 'lp': lp, 'text': text})
    drops = [(a, b) for a, b in zip(rows, rows[1:]) if b['lp'] < a['lp'] - 1e-9]
    return {'rows': rows, 'drops': drops}


def length_bias(model, top=10):
    """Probability falls with every character, so any total-probability search is
    also, silently, a search for short outputs.  Measured on the exact ranking."""
    scored = [(lp, s) for lp, s in exact_ranking(model)]
    by_total = sorted(scored, reverse=True)[:top]
    by_char = sorted(((lp / (len(s) + 1), s) for lp, s in scored), reverse=True)[:top]
    lens = np.array([len(s) for _, s in scored])
    lps = np.array([lp for lp, _ in scored])
    return {'mean_len': float(lens.mean()), 'shortest': int(lens.min()),
            'longest': int(lens.max()),
            'corr': float(np.corrcoef(lens, lps)[0, 1]),
            'top_total_len': float(np.mean([len(s) for _, s in by_total])),
            'top_char_len': float(np.mean([len(s) for _, s in by_char])),
            'total_best': by_total[0][1], 'char_best': by_char[0][1],
            'top': top,
            'alphas': [{'alpha': a,
                        'text': beam_search(model, width=8, alpha=a)[1]}
                       for a in (0.0, 1.0)]}


def parse(s):
    """Split a legal sentence back into its five grammar slots."""
    s = s.strip().rstrip('.')
    for d in DET:
        if not s.startswith(d):
            continue
        r = s[len(d):]
        for a in ADJ:
            if not r.startswith(a):
                continue
            r2 = r[len(a):]
            for n in NOUN:
                if not r2.startswith(n + ' '):
                    continue
                r3 = r2[len(n) + 1:]
                for v in VERB:
                    if r3.startswith(v + ' ') and r3[len(v) + 1:] in PLACE:
                        return (d, a, n, v, r3[len(v) + 1:])
    return None


def slot_coverage(texts):
    """How similar a set of sentences is to itself, slot by slot."""
    parsed = [p for p in (parse(t) for t in texts) if p]
    if len(parsed) < 2:
        return {'n': len(parsed), 'subjects': 0, 'overlap': 0.0, 'legal': len(parsed)}
    pairs = [(a, b) for i, a in enumerate(parsed) for b in parsed[i + 1:]]
    overlap = float(np.mean([sum(x == y for x, y in zip(a, b)) for a, b in pairs]))
    return {'n': len(parsed), 'legal': len(parsed),
            'subjects': len({p[:3] for p in parsed}),
            'overlap': overlap}


def sample_lines(model, rng, n=8, t=1.0, max_len=80, prefix='\n'):
    out = []
    for _ in range(n):
        s = prefix
        for _ in range(max_len):
            p = softmax(model.logits(s) / t)
            c = model.vocab[int(rng.choice(len(p), p=p))]
            s += c
            if c == '\n':
                break
        out.append(s.strip())
    return out


def beam_diversity(model, width=48, n_best=16, seeds=12):
    """The n best sentences against n drawn at random - the same model, both times.

    The sampled side is averaged over several draws so the comparison is not one
    lucky set.
    """
    beams = [t for _, t in beam_search(model, width=width, n_best=n_best)]
    sets = [sample_lines(model, np.random.default_rng(s), n=n_best)
            for s in range(seeds)]
    covs = [slot_coverage(x) for x in sets]
    return {'beam': beams, 'sample': sets[0], 'n': n_best,
            'beam_cov': slot_coverage(beams),
            'sample_cov': {'subjects': float(np.mean([c['subjects'] for c in covs])),
                           'overlap': float(np.mean([c['overlap'] for c in covs])),
                           'legal': float(np.mean([c['legal'] for c in covs])),
                           'sets': seeds}}


# ================================================================ 3. speculative
def speculative_distribution(p, q):
    """The exact output distribution of one accept/reject step - not an estimate.

    The rule is: accept the draft token x with probability min(1, p(x)/q(x)); if it is
    rejected, draw from the normalised positive part of (p - q).  Summing both routes
    gives back p exactly, which is why this is "free" rather than "usually fine".
    """
    acc = np.minimum(1.0, p / np.maximum(q, 1e-300))
    accepted = q * acc                                   # drew x and kept it
    reject_mass = float(1.0 - accepted.sum())
    resid = np.maximum(p - q, 0.0)
    resid = resid / resid.sum() if resid.sum() > 0 else np.zeros_like(p)
    return accepted + reject_mass * resid, float(accepted.sum())


def always_accept_distribution(p, q):
    """Shortcut 1: trust the draft.  The output is simply the draft's distribution."""
    return q.copy(), 1.0


def deterministic_accept_distribution(p, q):
    """Shortcut 2: accept when the target likes the token at least as much as the
    draft did.  It sounds like the same rule without the coin flip.  It is not: the
    coin flip is what makes the accepted mass come out as min(p, q) exactly."""
    keep = p >= q
    out = np.where(keep, q, 0.0)
    rest = float(1.0 - out.sum())
    resid = np.maximum(p - q, 0.0)
    resid = resid / resid.sum() if resid.sum() > 0 else np.zeros_like(p)
    return out + rest * resid, float(out.sum())


def speculative_step(model, draft, ctx, gamma, rng):
    """One verification round: draft gamma tokens, verify them all, emit what sticks."""
    proposals, qs = [], []
    c = ctx
    for _ in range(gamma):
        q = draft.probs(c)
        i = int(rng.choice(len(q), p=q))
        proposals.append(i)
        qs.append(q)
        c += model.vocab[i]
    emitted, accepted = [], 0
    c = ctx
    for i, q in zip(proposals, qs):
        p = model.probs(c)                       # in a real server: one batched pass
        if rng.random() < min(1.0, p[i] / max(q[i], 1e-300)):
            emitted.append(i)
            accepted += 1
            c += model.vocab[i]
        else:
            resid = np.maximum(p - q, 0.0)
            resid = resid / resid.sum()
            j = int(rng.choice(len(p), p=resid))
            emitted.append(j)
            c += model.vocab[j]
            break
    else:
        p = model.probs(c)                       # all accepted: one free bonus token
        emitted.append(int(rng.choice(len(p), p=p)))
    return emitted, accepted


def speculative_generate(model, draft, rng, n=100, gamma=4, max_len=80, prefix='\n'):
    lines, acc, drafted, rounds, tokens = [], 0, 0, 0, 0
    for _ in range(n):
        out = prefix
        while len(out) < max_len:
            ids, a = speculative_step(model, draft, out, gamma, rng)
            rounds += 1
            acc += a
            drafted += gamma
            tokens += len(ids)
            text = ''.join(model.vocab[i] for i in ids)
            nl = text.find('\n')
            out += text if nl < 0 else text[:nl + 1]
            if nl >= 0:
                break
        lines.append(out)
    return lines, {'accept_rate': acc / drafted, 'rounds': rounds,
                   'tokens': tokens, 'per_round': tokens / rounds, 'gamma': gamma}


def draft_table(model, text, orders=(1, 2, 3, 5), gamma=4, n=40, seed=3):
    rows = []
    for o in orders:
        d = NGram(text, order=o)
        rng = np.random.default_rng([seed, o])
        lines, st = speculative_generate(model, d, rng, n=n, gamma=gamma)
        ok = sum(is_grammatical(l) for l in lines) / len(lines)
        rows.append({'order': o, 'accept': st['accept_rate'], 'per_round': st['per_round'],
                     'grammatical': ok, 'distinct': len({l.strip() for l in lines})})
    return rows


def gamma_table(model, draft, gammas=(1, 2, 4, 8), n=40, seed=4, draft_cost=0.1):
    """Longer guesses win more when they land and waste more when they do not."""
    rows = []
    for g in gammas:
        rng = np.random.default_rng([seed, g])
        _, st = speculative_generate(model, draft, rng, n=n, gamma=g)
        a = st['accept_rate']
        theory = (1 - a ** (g + 1)) / (1 - a) if a < 1 else g + 1
        cost = 1 + g * draft_cost                 # one target pass + g draft passes
        rows.append({'gamma': g, 'accept': a, 'per_round': st['per_round'],
                     'theory': theory, 'speedup': st['per_round'] / cost})
    return rows


# ================================================================ 4. checks
def check():
    text = make_corpus(400, seed=0)
    m = NGram(text)
    rank = exact_ranking(m)
    assert len(rank) == 1250 and rank[0][0] > rank[-1][0]
    # a beam of width 1 is greedy, by construction
    assert beam_search(m, width=1)[1] == greedy(m)
    # a beam never beats the enumerated optimum...
    lps = [beam_search(m, width=w)[0] for w in (1, 2, 4, 8)]
    assert max(lps) <= rank[0][0] + 1e-9
    # ...but it is NOT monotone in the width: this model has a width where
    # widening the beam makes the answer worse, which is the point of section 3
    assert not all(b >= a - 1e-9 for a, b in zip(lps, lps[1:])), lps
    # speculative sampling returns exactly the target distribution
    d = NGram(text, order=2)
    worst = {'trust': 0.0, 'nocoin': 0.0}
    for ctx in ('\n', '\nthe ', '\nthe small cat ', '\na red dog s'):
        p, q = m.probs(ctx), d.probs(ctx)
        out, acc = speculative_distribution(p, q)
        assert abs(out.sum() - 1) < 1e-12
        assert float(np.abs(out - p).max()) < 1e-12, float(np.abs(out - p).max())
        assert 0 <= acc <= 1
        # the plausible shortcut does not
        for name, wrong in (('trust', always_accept_distribution),
                            ('nocoin', deterministic_accept_distribution)):
            bad, _ = wrong(p, q)
            worst[name] = max(worst[name], float(np.abs(bad - p).max()))
    # both shortcuts move the distribution somewhere, the correct rule never does
    assert worst['trust'] > 1e-3 and worst['nocoin'] > 1e-3, worst
    # identical draft and target: everything is accepted, nothing changes
    out, acc = speculative_distribution(m.probs('\nthe '), m.probs('\nthe '))
    assert abs(acc - 1) < 1e-12 and float(np.abs(out - m.probs('\nthe ')).max()) < 1e-12
    assert is_grammatical('the small cat sat on the mat.')
    assert not is_grammatical('the small cat sat on the moon.')


# ================================================================ 5. the story
def header(t):
    print('\n' + '=' * 72)
    print(t)
    print('=' * 72)


def main():
    text = make_corpus()
    m = NGram(text)
    rank = exact_ranking(m)
    rank_of = {s: i + 1 for i, (_, s) in enumerate(rank)}

    header('1. Ground truth')
    print(f'  {len(rank):,} legal sentences, each scored exactly by summing its')
    print('  per-step log probabilities.  Nothing below is an estimate.\n')
    for lp, s in rank[:3]:
        print(f'    {lp:8.3f}  {s!r}')
    print(f'    ...')
    print(f'    {rank[-1][0]:8.3f}  {rank[-1][1]!r}')
    g = greedy(m)
    print(f'\n  greedy produces {g!r}, rank {rank_of[g]} of {len(rank):,}.')
    print('  Picking the best character at every step is not picking the best')
    print('  sentence, and that gap is what search is for.')

    header('2. Beam search')
    print('  Keep the `width` best prefixes at every step instead of only the best.\n')
    print('  width   rank   log p   sentence')
    rows = beam_table(m)
    for r in rows:
        print(f'  {r["width"]:5d}  {str(r["rank"]):>5}  {r["lp"]:7.3f}   {r["text"]!r}')
    best = min(r['rank'] for r in rows)
    print(f'\n  Width 1 is greedy by definition (and reproduces its sentence exactly,')
    print('  once the sort is made stable - this model has exact ties, and an unstable')
    print('  sort alone is enough to make the two disagree).')
    print(f'  Widening the beam climbs from rank {rows[0]["rank"]} to rank {best}, and')
    print('  then stops: the beam prunes a prefix that only pays off later, so it')
    print('  never reaches rank 1 at any width here.  Beam search is a heuristic, not')
    print('  a maximiser - the asserts only check that a wider beam never scores worse')
    print('  and that no beam beats the enumerated optimum.')
    print('\n  The cost is linear in the width: a width-8 beam runs the model on 8')
    print('  sequences at every step, which on a server is 8 times the KV cache of')
    print('  the previous days and 8 times the attention work.')

    header('3. A wider beam is not always a better answer')
    small = NGram(make_corpus(400, seed=0))
    bm = beam_monotonicity(small)
    print('  On the model trained on a smaller corpus, scanning the width one step')
    print('  at a time:\n')
    seen = None
    for r in bm['rows'][:8]:
        mark = ''
        if seen is not None and r['lp'] < seen - 1e-9:
            mark = '   <- worse than the narrower beam'
        seen = r['lp']
        print(f'    width {r["width"]:2d}   log p {r["lp"]:7.3f}{mark}')
    print(f'\n  {len(bm["drops"])} width(s) in 1-16 where widening the beam made the')
    print('  answer less likely.  This is not a bug in the implementation: a wider')
    print('  beam keeps more prefixes alive, and a prefix that looks mediocre early')
    print('  can survive to crowd out the one that would have won later.  The search')
    print('  is a heuristic, so nothing forbids it.  Translation papers call the')
    print('  version of this that shows up in quality scores the beam search curse:')
    print('  past a width of roughly 5 to 10, BLEU goes down while model probability')
    print('  goes up, which also says something uncomfortable about the model.')

    header('4. Beam search prefers short sentences, and that is a bug')
    lb = length_bias(m)
    print('  Every character multiplies the score by something below 1, so a longer')
    print('  sentence is almost always less likely than a shorter one.  Across all')
    print(f'  {len(rank):,} sentences the correlation between length and log')
    print(f'  probability is {lb["corr"]:.3f}: a total-probability search is also, in')
    print('  part, a search for short outputs.\n')
    print(f'    mean length of all sentences            {lb["mean_len"]:.1f}')
    print(f'    mean length of the top {lb["top"]} by total log p   {lb["top_total_len"]:.1f}')
    print(f'    mean length of the top {lb["top"]} by log p / len   {lb["top_char_len"]:.1f}')
    print(f'\n    best by total      {lb["total_best"]!r}')
    print(f'    best per character {lb["char_best"]!r}')
    print('\n  Dividing by length**alpha is the standard patch - alpha=1 scores the')
    print('  per-character average instead of the total.  Here the two agree on the')
    print(f'  winner ({lb["alphas"][0]["text"]!r} at both alpha=0 and alpha=1), because')
    print(f'  this grammar only spans {lb["shortest"]}-{lb["longest"]} characters.  In')
    print('  translation, where candidates differ by tens of tokens, the uncorrected')
    print('  beam truncates - and alpha is a heuristic with no derivation behind it,')
    print('  which is why papers report the value they used as a hyperparameter.')

    header('5. What beam search is bad at')
    bd = beam_diversity(m)
    print(f'  The first 6 of the {bd["n"]} best sentences a wide beam returns:\n')
    for t in bd['beam'][:6]:
        print(f'    {t!r}')
    print(f'\n  And 6 sentences sampled from the same model at t=1.0:\n')
    for t in bd['sample'][:6]:
        print(f'    {t!r}')
    bc, sc = bd['beam_cov'], bd['sample_cov']
    print(f'\n  Both sets are {bd["n"]} different sentences, so counting distinct')
    print('  strings says nothing.  Counting how alike they are does - a subject is')
    print('  the determiner, adjective and noun together:\n')
    print(f'    beam      {bc["subjects"]:4.1f} distinct subjects, '
          f'{bc["overlap"]:.2f} of 5 slots shared by the average pair')
    print(f'    sampled   {sc["subjects"]:4.1f} distinct subjects, '
          f'{sc["overlap"]:.2f} shared  (mean over {sc["sets"]} independent draws)')
    print('\n  The beam\'s answers are variations on one sentence, and they have to be:')
    print('  a beam searches for the most probable output, and the most probable')
    print('  outputs resemble each other by construction.  Asking it for n answers')
    print('  gives n neighbours, not n alternatives.  That is fatal for a chat model')
    print('  and exactly right where one answer is correct - translation, speech')
    print('  recognition, constrained extraction - which is where beam search is')
    print('  still the default.')

    header('6. Speculative decoding: the opposite idea')
    print('  Beam search changes the output and costs more.  Speculative decoding')
    print('  changes nothing about the output and costs less.\n')
    print('  A cheap draft model proposes gamma tokens.  The expensive model scores')
    print('  all of them in ONE pass - that is the whole trick, because verifying')
    print('  gamma tokens is a single forward pass over gamma positions, while')
    print('  generating them would have been gamma passes.  Then each proposal is')
    print('  accepted with probability min(1, p(x)/q(x)), and the first rejection is')
    print('  replaced by a draw from the normalised positive part of (p - q).')

    header('7. It is exactly the target distribution, not approximately')
    d2 = NGram(text, order=2)
    print('  Taking the accept route and the reject route together:\n')
    print('    P(emit x) = q(x)*min(1, p/q) + P(reject) * norm(p - q)+ (x) = p(x)\n')
    print('                          accept    max |out - p| for three rules')
    print('  context                   rate    correct   trust draft   no coin flip')
    for ctx in ('\nthe ', '\nthe small cat ', '\na red dog s'):
        p, q = m.probs(ctx), d2.probs(ctx)
        out, acc = speculative_distribution(p, q)
        a, _ = always_accept_distribution(p, q)
        b, _ = deterministic_accept_distribution(p, q)
        print(f'  {ctx.strip()!r:22}  {acc:6.3f}   {np.abs(out - p).max():8.1e}   '
              f'{np.abs(a - p).max():11.3f}   {np.abs(b - p).max():12.3f}')
    print('\n  The correct column is 1e-16: floating point, not approximation.  The')
    print('  model you deployed is the model that answers, exactly, at any draft')
    print('  quality - which is why this is a systems optimisation and not a quality')
    print('  trade-off.')
    print('\n  The other two columns are shortcuts that sound equivalent.  Trusting')
    print('  the draft is obviously wrong once written down.  Dropping the coin flip')
    print('  and accepting whenever the target likes the token at least as much as')
    print('  the draft did is not obviously wrong, and it still moves the')
    print('  distribution: the randomised accept is what makes the accepted mass come')
    print('  out as min(p, q) at every token, and the residual draw is what puts the')
    print('  rest back.  Both halves are load-bearing.')

    header('8. How good the draft has to be')
    print('  The draft is the same n-gram at a lower order: cheaper and worse.\n')
    print('  draft order   accept rate   tokens/round   grammatical   distinct')
    for r in draft_table(m, text):
        print(f'  {r["order"]:11d}   {r["accept"]:11.3f}   {r["per_round"]:12.2f}   '
              f'{r["grammatical"]:11.0%}   {r["distinct"]:8d}')
    print('\n  Read the last two columns first: they barely move.  They are not')
    print('  supposed to - the output distribution is identical no matter how bad the')
    print('  draft is, and that is the point.  A terrible draft costs speed, never')
    print('  quality.  Only the middle columns move, and they move a lot.')

    header('9. How far ahead to guess')
    d3 = NGram(text, order=3)
    print('  gamma   accept   tokens/round   theory   speedup vs one-at-a-time')
    for r in gamma_table(m, d3):
        print(f'  {r["gamma"]:5d}   {r["accept"]:6.3f}   {r["per_round"]:12.2f}   '
              f'{r["theory"]:6.2f}   {r["speedup"]:10.2f}x')
    print('\n  The theory column is (1 - a**(gamma+1)) / (1 - a), the textbook')
    print('  expected number of tokens per round when every proposal is accepted')
    print('  independently with probability a.  Look at gamma=8: the measurement is')
    print('  7.08 against a predicted 3.82.  The formula is not wrong, its assumption')
    print('  is: acceptances are not independent.  The draft is right for a whole')
    print('  predictable stretch ("...in the gar|den.") and wrong at the branch')
    print('  points, so long runs of acceptance are much more common than a coin with')
    print('  the same average would produce.  The same effect is why real deployments')
    print('  report acceptance *lengths* rather than acceptance rates.')
    print('  The speedup column charges one target pass plus gamma draft passes at a')
    print('  tenth of the cost each.  Guessing further ahead stops paying once the')
    print('  acceptance rate makes the extra proposals likely to be thrown away: every')
    print('  rejected token was drafted, verified, and discarded.')

    header('10. In vLLM')
    print('  Beam search is no longer a sampling parameter - it was moved out of')
    print('  SamplingParams into a separate `LLM.beam_search()` API, precisely')
    print('  because it is a different kind of thing: a search over sequences that')
    print('  needs its own KV-cache bookkeeping (section 2), not a per-token rule.\n')
    print('    llm.beam_search(prompts, BeamSearchParams(beam_width=4, max_tokens=64))\n')
    print('  Speculative decoding is configured on the server:\n')
    print('    vllm serve meta-llama/Llama-3.1-70B-Instruct \\')
    print('        --speculative-config \'{"model": "meta-llama/Llama-3.1-8B-Instruct",')
    print('                                "num_speculative_tokens": 5}\'\n')
    print('    --speculative-config \'{"method": "ngram",      # no draft model at all:')
    print('                            "num_speculative_tokens": 4,')
    print('                            "prompt_lookup_max": 4}\'   # copy from the prompt\n')
    print('  The n-gram method is this file\'s draft model, in production: it proposes')
    print('  continuations copied from the prompt itself.  It costs nothing and works')
    print('  startlingly well on summarisation, code editing and RAG, where the answer')
    print('  repeats long spans of the input.  EAGLE and Medusa replace the draft with')
    print('  a small head trained on the target model\'s own hidden states, which pushes')
    print('  the acceptance rate up without a second full model.')

    header('11. In SGLang')
    print('  Same idea, more of the machinery exposed:\n')
    print('    python -m sglang.launch_server --model ... \\')
    print('        --speculative-algorithm EAGLE3 \\')
    print('        --speculative-draft-model-path <eagle head> \\')
    print('        --speculative-num-steps 3 \\        # draft depth')
    print('        --speculative-eagle-topk 4 \\       # branches kept per step')
    print('        --speculative-num-draft-tokens 8   # nodes verified per round\n')
    print('  The difference worth understanding: `--speculative-eagle-topk` means the')
    print('  draft is a *tree*, not a chain.  Instead of one guess of length gamma it')
    print('  proposes several branches and verifies them all in one pass with a mask')
    print('  that lets every node attend only to its own ancestors.  More candidates')
    print('  per round, one target pass - the same trade as section 9 with a better')
    print('  shape.  Models that ship their own extra prediction head (MTP in')
    print('  DeepSeek-V3 and GLM) are the same mechanism with the draft built in:')
    print('  `--speculative-algorithm NEXTN`.')
    print('\n  And the measurement that matters on a server is not the acceptance rate')
    print('  but the accepted length per round against the batch size: speculation')
    print('  spends compute to save memory bandwidth, so it wins at small batch sizes')
    print('  (where decode is bandwidth-bound, the previous days) and can lose at')
    print('  large ones, where the GPU was already busy and the wasted drafts cost')
    print('  more than the saved passes.')

    header('12. Assertions')
    check()
    print('all assertions passed')


if __name__ == '__main__':
    main()
