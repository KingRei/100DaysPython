"""Day 55 - a character-level mini-GPT, trained from scratch in numpy.

Everything from the last four days in one file: a tokeniser (here just "one character is
one token"), multi-head attention, a transformer block, learned positions, a training
loop with Adam, and sampling.  No framework, no autograd - the backward pass is written
out by hand and checked against finite differences.

The corpus is generated from a small grammar defined in this file, which buys something
a pile of scraped text cannot: the true entropy of the source is known exactly, so
"the model learned something" is a number rather than an impression.

Run `python mini_gpt.py` to train it and print the whole story (about a minute).
"""

import math

import numpy as np


# ================================================================ 1. the corpus
DET = ['the ', 'a ']
ADJ = ['small ', 'old ', 'red ', 'quiet ', '']
NOUN = ['cat', 'dog', 'bird', 'child', 'robot']
VERB = ['sat', 'slept', 'waited', 'sang', 'ran']
PLACE = ['on the mat', 'in the garden', 'by the river', 'under the tree', 'near the door']


def sentence(rng):
    return (rng.choice(DET) + rng.choice(ADJ) + rng.choice(NOUN) + ' ' +
            rng.choice(VERB) + ' ' + rng.choice(PLACE) + '.\n')


def make_corpus(n_sentences=1200, seed=0):
    rng = np.random.default_rng(seed)
    return ''.join(sentence(rng) for _ in range(n_sentences))


def source_entropy():
    """Bits per character of the generator itself - the floor no model can beat.

    Every sentence is five independent choices, so its information content is the sum
    of the five log2 counts; dividing by the average sentence length gives bits/char.
    """
    bits = sum(math.log2(len(xs)) for xs in (DET, ADJ, NOUN, VERB, PLACE))
    rng = np.random.default_rng(1)
    lengths = [len(sentence(rng)) for _ in range(4000)]
    return bits / float(np.mean(lengths)), bits, float(np.mean(lengths))


def is_grammatical(line):
    """Could the generator have produced this line?"""
    if not line.endswith('.'):
        return False
    body = line[:-1]
    for d in DET:
        if not body.startswith(d):
            continue
        rest = body[len(d):]
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


class CharTokenizer:
    """One character, one token - the smallest tokeniser there is."""

    def __init__(self, text):
        self.chars = sorted(set(text))
        self.stoi = {c: i for i, c in enumerate(self.chars)}
        self.itos = {i: c for c, i in self.stoi.items()}

    @property
    def vocab_size(self):
        return len(self.chars)

    def encode(self, s):
        return np.array([self.stoi[c] for c in s], dtype=np.int64)

    def decode(self, ids):
        return ''.join(self.itos[int(i)] for i in ids)


# ================================================================ 2. the model
def init_params(vocab, d_model=64, n_layer=2, n_head=4, ctx=32, seed=0):
    rng = np.random.default_rng(seed)
    s = 0.02
    p = {'tok': rng.standard_normal((vocab, d_model)) * s,
         'pos': rng.standard_normal((ctx, d_model)) * s,
         'g_f': np.ones(d_model), 'b_f': np.zeros(d_model),
         'head': rng.standard_normal((d_model, vocab)) * s,
         'head_b': np.zeros(vocab)}
    for l in range(n_layer):
        p[f'g1_{l}'] = np.ones(d_model)
        p[f'b1_{l}'] = np.zeros(d_model)
        p[f'g2_{l}'] = np.ones(d_model)
        p[f'b2_{l}'] = np.zeros(d_model)
        p[f'Wqkv_{l}'] = rng.standard_normal((d_model, 3 * d_model)) * s
        p[f'Wo_{l}'] = rng.standard_normal((d_model, d_model)) * s / math.sqrt(2 * n_layer)
        p[f'W1_{l}'] = rng.standard_normal((d_model, 4 * d_model)) * s
        p[f'b1f_{l}'] = np.zeros(4 * d_model)
        p[f'W2_{l}'] = rng.standard_normal((4 * d_model, d_model)) * s / math.sqrt(2 * n_layer)
        p[f'b2f_{l}'] = np.zeros(d_model)
    cfg = {'d_model': d_model, 'n_layer': n_layer, 'n_head': n_head, 'ctx': ctx,
           'vocab': vocab}
    return p, cfg


def param_count(p):
    return sum(v.size for v in p.values())


def layer_norm(x, g, b, eps=1e-5):
    mu = x.mean(-1, keepdims=True)
    xc = x - mu
    var = (xc ** 2).mean(-1, keepdims=True)
    inv = 1.0 / np.sqrt(var + eps)
    xhat = xc * inv
    return xhat * g + b, (xhat, inv, g)


def layer_norm_backward(dy, cache):
    xhat, inv, g = cache
    d = xhat.shape[-1]
    dg = (dy * xhat).reshape(-1, d).sum(0)
    db = dy.reshape(-1, d).sum(0)
    dxhat = dy * g
    dx = inv / d * (d * dxhat - dxhat.sum(-1, keepdims=True)
                    - xhat * (dxhat * xhat).sum(-1, keepdims=True))
    return dx, dg, db


def gelu(x):
    return 0.5 * x * (1 + np.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * x ** 3)))


def gelu_backward(x, dy):
    c = math.sqrt(2 / math.pi)
    u = c * (x + 0.044715 * x ** 3)
    t = np.tanh(u)
    du = c * (1 + 3 * 0.044715 * x ** 2)
    return dy * (0.5 * (1 + t) + 0.5 * x * (1 - t ** 2) * du)


def softmax(x, axis=-1):
    m = np.max(x, axis=axis, keepdims=True)
    e = np.exp(x - m)
    return e / e.sum(axis=axis, keepdims=True)


def forward(p, cfg, idx, causal=True):
    """idx is (B, T) of token ids.  Returns logits (B, T, vocab) and a cache."""
    B, T = idx.shape
    H, d = cfg['n_head'], cfg['d_model']
    dh = d // H
    x = p['tok'][idx] + p['pos'][:T]
    cache = {'idx': idx, 'T': T, 'causal': causal, 'layers': []}
    mask = np.tril(np.ones((T, T), dtype=bool)) if causal else None
    for l in range(cfg['n_layer']):
        h, c1 = layer_norm(x, p[f'g1_{l}'], p[f'b1_{l}'])
        qkv = h @ p[f'Wqkv_{l}']
        q, k, v = np.split(qkv, 3, axis=-1)
        q = q.reshape(B, T, H, dh).transpose(0, 2, 1, 3)
        k = k.reshape(B, T, H, dh).transpose(0, 2, 1, 3)
        v = v.reshape(B, T, H, dh).transpose(0, 2, 1, 3)
        att = q @ k.transpose(0, 1, 3, 2) / math.sqrt(dh)
        if mask is not None:
            att = np.where(mask, att, -np.inf)
        w = softmax(att, axis=-1)
        o = (w @ v).transpose(0, 2, 1, 3).reshape(B, T, d)
        a = o @ p[f'Wo_{l}']
        x = x + a
        h2, c2 = layer_norm(x, p[f'g2_{l}'], p[f'b2_{l}'])
        f1 = h2 @ p[f'W1_{l}'] + p[f'b1f_{l}']
        f2 = gelu(f1)
        f3 = f2 @ p[f'W2_{l}'] + p[f'b2f_{l}']
        x = x + f3
        cache['layers'].append({'c1': c1, 'h': h, 'q': q, 'k': k, 'v': v, 'w': w,
                                'o': o, 'c2': c2, 'h2': h2, 'f1': f1, 'f2': f2})
    xf, cf = layer_norm(x, p['g_f'], p['b_f'])
    logits = xf @ p['head'] + p['head_b']
    cache['cf'] = cf
    cache['xf'] = xf
    return logits, cache


def cross_entropy(logits, targets):
    B, T, V = logits.shape
    flat = logits.reshape(-1, V)
    probs = softmax(flat, axis=-1)
    n = flat.shape[0]
    tgt = targets.reshape(-1)
    loss = -np.log(probs[np.arange(n), tgt] + 1e-12).mean()
    dlogits = probs
    dlogits[np.arange(n), tgt] -= 1.0
    return loss, (dlogits / n).reshape(B, T, V)


def backward(p, cfg, cache, dlogits):
    """The hand-written backward pass: one dictionary of gradients, same keys as p."""
    B, T = cache['idx'].shape
    H, d = cfg['n_head'], cfg['d_model']
    dh = d // H
    g = {k: np.zeros_like(v) for k, v in p.items()}
    g['head'] = cache['xf'].reshape(-1, d).T @ dlogits.reshape(-1, cfg['vocab'])
    g['head_b'] = dlogits.reshape(-1, cfg['vocab']).sum(0)
    dxf = dlogits @ p['head'].T
    dx, g['g_f'], g['b_f'] = layer_norm_backward(dxf, cache['cf'])
    for l in reversed(range(cfg['n_layer'])):
        c = cache['layers'][l]
        # --- MLP branch
        df3 = dx
        g[f'W2_{l}'] = c['f2'].reshape(-1, 4 * d).T @ df3.reshape(-1, d)
        g[f'b2f_{l}'] = df3.reshape(-1, d).sum(0)
        df2 = df3 @ p[f'W2_{l}'].T
        df1 = gelu_backward(c['f1'], df2)
        g[f'W1_{l}'] = c['h2'].reshape(-1, d).T @ df1.reshape(-1, 4 * d)
        g[f'b1f_{l}'] = df1.reshape(-1, 4 * d).sum(0)
        dh2 = df1 @ p[f'W1_{l}'].T
        dx2, g[f'g2_{l}'], g[f'b2_{l}'] = layer_norm_backward(dh2, c['c2'])
        dx = dx + dx2
        # --- attention branch
        da = dx
        g[f'Wo_{l}'] = c['o'].reshape(-1, d).T @ da.reshape(-1, d)
        do = da @ p[f'Wo_{l}'].T
        do = do.reshape(B, T, H, dh).transpose(0, 2, 1, 3)
        dw = do @ c['v'].transpose(0, 1, 3, 2)
        dv = c['w'].transpose(0, 1, 3, 2) @ do
        # softmax backward, row by row
        datt = c['w'] * (dw - (dw * c['w']).sum(-1, keepdims=True))
        datt = datt / math.sqrt(dh)
        dq = datt @ c['k']
        dk = datt.transpose(0, 1, 3, 2) @ c['q']
        def merge(t):
            return t.transpose(0, 2, 1, 3).reshape(B, T, d)
        dqkv = np.concatenate([merge(dq), merge(dk), merge(dv)], axis=-1)
        g[f'Wqkv_{l}'] = c['h'].reshape(-1, d).T @ dqkv.reshape(-1, 3 * d)
        dh1 = dqkv @ p[f'Wqkv_{l}'].T
        dx1, g[f'g1_{l}'], g[f'b1_{l}'] = layer_norm_backward(dh1, c['c1'])
        dx = dx + dx1
    np.add.at(g['tok'], cache['idx'].reshape(-1), dx.reshape(-1, d))
    g['pos'][:T] = dx.sum(0)
    return g


# ================================================================ 3. training
def get_batch(data, ctx, batch, rng):
    i = rng.integers(0, len(data) - ctx - 1, batch)
    x = np.stack([data[j:j + ctx] for j in i])
    y = np.stack([data[j + 1:j + ctx + 1] for j in i])
    return x, y


def train(p, cfg, data, steps=500, batch=16, lr=3e-3, seed=0, causal=True,
          shift=1, log_every=None):
    """Adam, no schedule, no clipping - the smallest loop that works.

    `shift=1` means "predict the next character".  Section 7 sets it to 0.
    """
    rng = np.random.default_rng(seed)
    m = {k: np.zeros_like(v) for k, v in p.items()}
    v = {k: np.zeros_like(x) for k, x in p.items()}
    b1, b2, eps = 0.9, 0.95, 1e-8
    hist = []
    for t in range(1, steps + 1):
        i = rng.integers(0, len(data) - cfg['ctx'] - 1, batch)
        x = np.stack([data[j:j + cfg['ctx']] for j in i])
        y = np.stack([data[j + shift:j + cfg['ctx'] + shift] for j in i])
        logits, cache = forward(p, cfg, x, causal=causal)
        loss, dlogits = cross_entropy(logits, y)
        grads = backward(p, cfg, cache, dlogits)
        for k in p:
            m[k] = b1 * m[k] + (1 - b1) * grads[k]
            v[k] = b2 * v[k] + (1 - b2) * grads[k] ** 2
            mh = m[k] / (1 - b1 ** t)
            vh = v[k] / (1 - b2 ** t)
            p[k] -= lr * mh / (np.sqrt(vh) + eps)
        hist.append(loss)
        if log_every and (t == 1 or t % log_every == 0):
            print(f'  step {t:>5}   loss {loss:.4f}   bits/char {loss / math.log(2):.4f}')
    return hist


def evaluate(p, cfg, data, rng, batches=20, batch=32):
    losses = []
    for _ in range(batches):
        x, y = get_batch(data, cfg['ctx'], batch, rng)
        logits, _ = forward(p, cfg, x)
        losses.append(cross_entropy(logits, y)[0])
    return float(np.mean(losses))


# ================================================================ 4. sampling
def sample(p, cfg, tok, prompt='the ', n=200, temperature=1.0, top_k=None, seed=0):
    rng = np.random.default_rng(seed)
    ids = list(tok.encode(prompt))
    for _ in range(n):
        window = np.array(ids[-cfg['ctx']:])[None, :]
        logits, _ = forward(p, cfg, window)
        z = logits[0, -1] / max(temperature, 1e-6)
        if top_k is not None:
            cut = np.sort(z)[-top_k]
            z = np.where(z >= cut, z, -np.inf)
        probs = softmax(z)
        ids.append(int(rng.choice(len(probs), p=probs)))
    return tok.decode(ids)


def grammatical_fraction(text):
    """Of the complete lines in a sample, how many could the generator have made?"""
    lines = [l for l in text.split('\n')[1:-1] if l]
    if not lines:
        return 0.0, 0
    ok = sum(is_grammatical(l) for l in lines)
    return ok / len(lines), len(lines)


# ================================================================ 5. gradient check
def gradcheck(seed=0, n=6, eps=1e-5):
    """Finite differences against the hand-written backward pass."""
    text = make_corpus(40, seed=2)
    tok = CharTokenizer(text)
    data = tok.encode(text)
    p, cfg = init_params(tok.vocab_size, d_model=16, n_layer=2, n_head=2, ctx=8, seed=1)
    rng = np.random.default_rng(seed)
    x, y = get_batch(data, cfg['ctx'], 4, rng)
    logits, cache = forward(p, cfg, x)
    loss, dlogits = cross_entropy(logits, y)
    grads = backward(p, cfg, cache, dlogits)
    worst = 0.0
    for k in ['tok', 'pos', 'Wqkv_0', 'Wo_0', 'W1_1', 'g1_0', 'b2f_1', 'head']:
        flat = p[k].ravel()
        gflat = grads[k].ravel()
        for j in rng.integers(0, flat.size, n):
            old = flat[j]
            flat[j] = old + eps
            lp = cross_entropy(forward(p, cfg, x)[0], y)[0]
            flat[j] = old - eps
            lm = cross_entropy(forward(p, cfg, x)[0], y)[0]
            flat[j] = old
            num = (lp - lm) / (2 * eps)
            ana = gflat[j]
            denom = max(1e-8, abs(num) + abs(ana))
            worst = max(worst, abs(num - ana) / denom)
    return worst


# ================================================================ 6. assertions
def check():
    # the grammar checker accepts what the grammar makes and rejects small edits
    rng = np.random.default_rng(5)
    for _ in range(50):
        assert is_grammatical(sentence(rng).strip())
    assert not is_grammatical('the cat sat on the moon.')
    assert not is_grammatical('cat the sat on the mat.')
    assert not is_grammatical('the cat sat on the mat')

    # the tokeniser round-trips
    text = make_corpus(20, seed=3)
    tok = CharTokenizer(text)
    assert tok.decode(tok.encode(text)) == text

    # the hand-written backward pass agrees with finite differences
    assert gradcheck() < 2e-5

    # causal masking really does hide the future: changing a later token cannot
    # change an earlier prediction
    data = tok.encode(text)
    p, cfg = init_params(tok.vocab_size, d_model=16, n_layer=1, n_head=2, ctx=8, seed=4)
    x, _ = get_batch(data, cfg['ctx'], 2, np.random.default_rng(0))
    base, _ = forward(p, cfg, x)
    x2 = x.copy()
    x2[:, -1] = (x2[:, -1] + 1) % cfg['vocab']
    changed, _ = forward(p, cfg, x2)
    assert np.abs(base[:, :-1] - changed[:, :-1]).max() < 1e-12
    nomask, _ = forward(p, cfg, x, causal=False)
    nomask2, _ = forward(p, cfg, x2, causal=False)
    assert np.abs(nomask[:, :-1] - nomask2[:, :-1]).max() > 1e-6
    return True


# ================================================================ 7. the whole story
def header(title):
    print('\n' + '=' * 72)
    print(title)
    print('=' * 72)


def main():
    np.set_printoptions(suppress=True, precision=4, legacy='1.25')

    # ------------------------------------------------------------------ 1
    header('1. The corpus, and why it is generated rather than scraped')
    text = make_corpus()
    tok = CharTokenizer(text)
    data = tok.encode(text)
    ent, bits, avg_len = source_entropy()
    print('  The text comes from a five-slot grammar defined in this file:\n')
    print('    ' + text[:120].replace('\n', '\n    '))
    print(f'  {len(text):,} characters, {tok.vocab_size} distinct ones, so one token is')
    print(f'  one character and the whole vocabulary is {tok.vocab_size} entries.\n')
    print(f'  Because the generator is known, so is the entropy of the source:')
    print(f'  every sentence is five independent choices worth {bits:.3f} bits in total,')
    print(f'  spread over {avg_len:.2f} characters on average = {ent:.4f} bits per character.')
    print('  No model can go below that, and a model that goes far below it is')
    print('  memorising rather than learning.  Scraped text gives you no such line.')

    # ------------------------------------------------------------------ 2
    header('2. The model')
    p, cfg = init_params(tok.vocab_size)
    print(f'  d_model {cfg["d_model"]}, {cfg["n_layer"]} blocks, {cfg["n_head"]} heads, '
          f'context {cfg["ctx"]}, vocab {cfg["vocab"]}')
    print(f'  {param_count(p):,} parameters - about the size of a photograph.\n')
    print('  The pieces are the last four days: character ids -> embedding + learned')
    print('  position -> pre-LN blocks of (multi-head attention, MLP) -> final norm ->')
    print('  one logit per vocabulary entry.  The backward pass is written out by hand.')

    # ------------------------------------------------------------------ 3
    header('3. Does the hand-written backward pass agree with the derivative?')
    worst = gradcheck()
    print(f'  Worst relative disagreement with central differences: {worst:.2e}')
    print('  over eight parameter tensors, six entries each.  This is the only check')
    print('  that distinguishes "the loss went down" from "the gradients are right":')
    print('  a wrong gradient still trains, just to somewhere else.')

    # ------------------------------------------------------------------ 4
    header('4. Training')
    print(f'  Adam, lr 3e-3, batch 16, 500 steps.  The starting loss should be about')
    print(f'  ln({tok.vocab_size}) = {math.log(tok.vocab_size):.4f}, which is what "no idea at all" costs.\n')
    hist = train(p, cfg, data, steps=500, log_every=100)
    final = float(np.mean(hist[-50:]))
    val_text = make_corpus(200, seed=99)
    val = evaluate(p, cfg, tok.encode(val_text), np.random.default_rng(0))
    print(f'\n  final training loss  {final:.4f} = {final / math.log(2):.4f} bits/char')
    print(f'  held-out loss        {val:.4f} = {val / math.log(2):.4f} bits/char')
    print(f'  source entropy       {ent:.4f} bits/char')
    print(f'  gap to the floor     {val / math.log(2) - ent:+.4f} bits/char')
    print('\n  The held-out text comes from the same grammar with a different seed, so')
    print('  train and held-out losses agreeing means the model learned the grammar')
    print('  rather than the particular 38,000 characters it was shown.')

    # ------------------------------------------------------------------ 5
    header('5. What it writes')
    for temp, top_k in ((1.0, None), (0.8, 5), (0.2, None)):
        out = sample(p, cfg, tok, n=240, temperature=temp, top_k=top_k, seed=7)
        frac, n_lines = grammatical_fraction(out)
        label = f'temperature {temp}' + (f', top-k {top_k}' if top_k else '')
        print(f'\n  --- {label}: {frac * 100:.0f}% of {n_lines} complete lines are grammatical')
        print('    ' + out[:200].replace('\n', '\n    '))
    print('\n  "Grammatical" here means the five-slot generator could have produced that')
    print('  line - a check this file can actually run, not a judgement about whether')
    print('  the text reads nicely.')

    # ------------------------------------------------------------------ 6
    header('6. Silent failure #1: training without the causal mask')
    p2, cfg2 = init_params(tok.vocab_size, seed=0)
    hist2 = train(p2, cfg2, data, steps=250, causal=False)
    print(f'  250 steps without the mask: loss {np.mean(hist2[-50:]):.4f}')
    print(f'  250 steps with the mask:    loss {np.mean(hist[200:250]):.4f}')
    print('\n  The unmasked run looks better by a wide margin, and it is cheating: every')
    print('  position can see the character it is being asked to predict.  Nothing about')
    print('  the shapes, the loss curve or the gradients says so.  It only shows up when')
    print('  you generate, where there is no future to look at:\n')
    out2 = sample(p2, cfg2, tok, n=240, temperature=0.8, seed=7)
    f2, n2 = grammatical_fraction(out2)
    print(f'    {f2 * 100:.0f}% of {n2} lines grammatical, from the model with the lower loss')
    print('    ' + out2[:160].replace('\n', '\n    '))

    # ------------------------------------------------------------------ 7
    header('7. Silent failure #2: predicting the character you were just given')
    p4, cfg4 = init_params(tok.vocab_size, seed=0)
    hist4 = train(p4, cfg4, data, steps=150, shift=0)
    print(f'  150 steps with targets shifted by 0 instead of 1: loss '
          f'{np.mean(hist4[-20:]):.4f}')
    print(f'  (the same 150 steps done properly: {np.mean(hist[130:150]):.4f})')
    print('\n  A loss heading for zero is the tell, and it is easy to read as success.')
    print('  The model is learning the identity function: copy the input through.  It')
    print('  cannot generate anything, because at generation time the character it is')
    print('  being asked to copy is the one that does not exist yet.')
    out4 = sample(p4, cfg4, tok, n=120, temperature=0.8, seed=7)
    print('    ' + out4[:120].replace('\n', '\n    '))

    # ------------------------------------------------------------------ 8
    header('8. Assertions')
    check()
    print('all assertions passed')


if __name__ == '__main__':
    main()
