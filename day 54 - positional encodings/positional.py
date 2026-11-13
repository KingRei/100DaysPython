"""Day 54 - positional encodings: sinusoidal, RoPE, ALiBi, and the FlexAttention idea.

Attention has no idea what order its inputs are in.  Shuffle the tokens and the output
shuffles with them, to the last bit - so every transformer has to put the order back by
hand, and the three ways of doing it differ in what happens past the lengths they were
trained on.

Run `python positional.py` to print the whole story; every number below is produced by
this file and checked by the asserts at the bottom.
"""

import math

import numpy as np


# ================================================================ 1. attention again
def softmax(x, axis=-1):
    x = np.asarray(x, dtype=float)
    m = np.max(x, axis=axis, keepdims=True)
    e = np.exp(x - m)
    return e / e.sum(axis=axis, keepdims=True)


def attention(Q, K, V, mask=None, bias=None):
    """softmax(QK^T/sqrt(d) + bias) V, with bias the hook ALiBi needs."""
    d = Q.shape[-1]
    s = Q @ K.T / math.sqrt(d)
    if bias is not None:
        s = s + bias
    if mask is not None:
        s = np.where(mask, s, -np.inf)
    w = softmax(s, axis=-1)
    return w @ V, w


def permutation_test(n=8, d=16, seed=0):
    """Attention is permutation-equivariant: permute the input, permute the output.

    Nothing in softmax(QK^T)V refers to an index, so the model literally cannot tell
    "the cat sat" from "sat cat the" until something puts the order back.
    """
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, d))
    perm = rng.permutation(n)
    out, _ = attention(X, X, X)
    out_perm, _ = attention(X[perm], X[perm], X[perm])
    return float(np.abs(out[perm] - out_perm).max())


# ================================================================ 2. sinusoidal
def sinusoidal(n, d, base=10000.0):
    """The 2017 table: pos / base^(2i/d) through sin and cos, one row per position."""
    pos = np.arange(n)[:, None]
    i = np.arange(0, d, 2)[None, :]
    ang = pos / (base ** (i / d))
    pe = np.zeros((n, d))
    pe[:, 0::2] = np.sin(ang)
    pe[:, 1::2] = np.cos(ang)
    return pe


def sinusoidal_shift_is_linear(n=64, d=32, k=5):
    """Shifting by k is a fixed rotation of each (sin, cos) pair - the same matrix for
    every position, which is what makes relative offsets learnable at all."""
    pe = sinusoidal(n, d)
    i = np.arange(0, d, 2)
    theta = k / (10000.0 ** (i / d))
    rot = np.zeros((n - k, d))
    s, c = np.sin(theta), np.cos(theta)
    rot[:, 0::2] = pe[:-k, 0::2] * c + pe[:-k, 1::2] * s
    rot[:, 1::2] = pe[:-k, 1::2] * c - pe[:-k, 0::2] * s
    return float(np.abs(rot - pe[k:]).max())


def sinusoidal_similarity(n=64, d=64):
    """Dot product between the encoding of position 0 and every other position."""
    pe = sinusoidal(n, d)
    return (pe @ pe[0]) / d


# ================================================================ 3. RoPE
def rope_angles(d, pos, base=10000.0):
    i = np.arange(0, d, 2)
    return np.outer(np.asarray(pos, dtype=float), 1.0 / (base ** (i / d)))


def rope(x, pos, base=10000.0):
    """Rotate each (even, odd) pair of channels by an angle proportional to position.

    No vector is added to anything: the position is applied to q and k as a rotation,
    so it changes the direction of the query and the key, not their length.
    """
    x = np.atleast_2d(np.asarray(x, dtype=float))
    ang = rope_angles(x.shape[-1], pos, base)
    c, s = np.cos(ang), np.sin(ang)
    out = np.empty_like(x)
    out[..., 0::2] = x[..., 0::2] * c - x[..., 1::2] * s
    out[..., 1::2] = x[..., 1::2] * c + x[..., 0::2] * s
    return out


def rope_is_relative(d=64, m=3, n=11, shift=97, seed=0):
    """<RoPE(q, m), RoPE(k, n)> depends only on n - m, not on m and n themselves."""
    rng = np.random.default_rng(seed)
    q, k = rng.standard_normal(d), rng.standard_normal(d)
    a = float(rope(q, [m])[0] @ rope(k, [n])[0])
    b = float(rope(q, [m + shift])[0] @ rope(k, [n + shift])[0])
    return a, b, abs(a - b)


def rope_wavelengths(d=128, base=10000.0):
    """How many tokens each channel pair takes to complete one full turn."""
    i = np.arange(0, d, 2)
    freq = 1.0 / (base ** (i / d))
    return 2 * math.pi / freq


def rope_context_scaling(d=128, base=10000.0, train=4096, target=32768):
    """Two ways to make a model trained to `train` work at `target`.

    Position interpolation squeezes the positions; raising the base stretches the
    wavelengths.  Both aim at the same thing: no channel should see an angle it never
    saw in training.
    """
    wl = rope_wavelengths(d, base)
    slowest = wl.max()
    need = base * (target / train) ** (d / (d - 2))
    return {'slowest_wavelength': float(slowest),
            'turns_at_train': train / slowest,
            'turns_at_target': target / slowest,
            'interp_factor': train / target,
            'ntk_base': float(need)}


# ================================================================ 4. ALiBi
def alibi_slopes(h):
    """The published geometric sequence: 2^(-8/h), 2^(-16/h), ... for h heads."""
    start = 2.0 ** (-8.0 / h)
    return np.array([start ** (i + 1) for i in range(h)])


def alibi_bias(n, slope):
    """-slope * (distance), added to the scores before the softmax."""
    pos = np.arange(n)
    dist = pos[None, :] - pos[:, None]          # key - query, negative into the past
    return slope * dist


def alibi_effective_window(n=512, h=8, tail=0.01):
    """How far back each head still puts `tail` of its weight, with flat scores.

    With no content signal the weights are exp(-slope * distance) normalised, which is
    a geometric decay: each head is a soft window of its own width.
    """
    rows = []
    for sl in alibi_slopes(h):
        b = alibi_bias(n, sl)[-1]
        w = softmax(np.where(np.arange(n) <= n - 1, b, -np.inf))
        cum = np.cumsum(w[::-1])                # from the newest token backwards
        k = int(np.searchsorted(cum, 1 - tail) + 1)
        rows.append({'slope': float(sl), 'window': k})
    return rows


# ================================================================ 5. FlexAttention
def flex_attention(Q, K, V, score_mod=None, mask_mod=None):
    """The FlexAttention idea in nine lines of numpy.

    One function edits the score, one function says which pairs are allowed, and every
    positional scheme in this file becomes an argument instead of a kernel.
    """
    n, d = Q.shape
    m = K.shape[0]
    s = Q @ K.T / math.sqrt(d)
    qi = np.arange(n)[:, None]
    kj = np.arange(m)[None, :]
    if score_mod is not None:
        s = score_mod(s, qi, kj)
    if mask_mod is not None:
        s = np.where(mask_mod(qi, kj), s, -np.inf)
    w = softmax(s, axis=-1)
    return w @ V, w


def mod_causal():
    return lambda qi, kj: kj <= qi


def mod_sliding(window):
    return lambda qi, kj: (kj <= qi) & (qi - kj < window)


def mod_alibi(slope):
    return lambda s, qi, kj: s + slope * (kj - qi)


def mod_softcap(cap):
    return lambda s, qi, kj: cap * np.tanh(s / cap)


def block_sparsity(n=1024, block=128, window=256):
    """What a block mask buys: whole tiles that are entirely masked are never computed.

    FlexAttention's BlockMask is this table - the fraction of (query block, key block)
    tiles with nothing allowed inside them.
    """
    nb = n // block
    qi = np.arange(n)[:, None]
    kj = np.arange(n)[None, :]
    rows = []
    for name, allowed in (('full', np.ones((n, n), dtype=bool)),
                          ('causal', mod_causal()(qi, kj)),
                          (f'sliding {window}', mod_sliding(window)(qi, kj))):
        live = 0
        for a in range(nb):
            for b in range(nb):
                if allowed[a * block:(a + 1) * block, b * block:(b + 1) * block].any():
                    live += 1
        rows.append({'name': name, 'live': live, 'total': nb * nb,
                     'skipped': 1 - live / (nb * nb)})
    return rows


# ================================================================ 6. seeded tables
def far_token_wins(train=32, far=512, d=64, trials=200, seed=0):
    """How often a token beyond the trained length out-scores everything inside it.

    No model is trained here; this is a property of the encoding itself.  For each
    random (q, k) pair, compare the best score at a distance the model has seen with
    the best score at a distance it has not.
    """
    rng = np.random.default_rng(seed)
    dist = np.arange(1, far + 1)
    near, far_m = dist <= train, dist > train
    pe = sinusoidal(far + 1, d)
    slope = alibi_slopes(8)[3]
    out = {'sinusoidal': 0, 'rope': 0, 'alibi': 0}
    for _ in range(trials):
        q, k = rng.standard_normal(d), rng.standard_normal(d)
        s_sin = np.array([(q + pe[t]) @ (k + pe[0]) for t in dist]) / math.sqrt(d)
        s_rope = np.array([rope(q, [t])[0] @ rope(k, [0])[0] for t in dist]) / math.sqrt(d)
        s_ali = (q @ k) / math.sqrt(d) - slope * dist
        for name, sc in (('sinusoidal', s_sin), ('rope', s_rope), ('alibi', s_ali)):
            if sc[far_m].max() > sc[near].max():
                out[name] += 1
    return {k2: v / trials for k2, v in out.items()}


def decode_offset_bug(d=64, n=6, seed=1):
    """The cache bug: RoPE applied at the wrong position when a token is appended.

    During decoding the new token's absolute position is len(cache); using 0 instead
    (because the new query is "the first row of this step") is a one-line mistake that
    produces finite, plausible numbers.
    """
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, d))
    pos = np.arange(n)
    Kr = rope(X, pos)
    q = rng.standard_normal(d)
    right = rope(q[None, :], [n - 1])[0] @ Kr.T / math.sqrt(d)
    wrong = rope(q[None, :], [0])[0] @ Kr.T / math.sqrt(d)
    return (softmax(right), softmax(wrong),
            float(np.abs(softmax(right) - softmax(wrong)).max()))


# ================================================================ 7. assertions
def check():
    # attention cannot see order at all
    assert permutation_test() < 1e-12

    # sinusoidal: shifting by k is one fixed rotation, independent of position
    assert sinusoidal_shift_is_linear() < 1e-12

    # RoPE: the score depends only on the distance
    a, b, gap = rope_is_relative()
    assert gap < 1e-9 and abs(a) > 1e-6
    # ... and it is a rotation, so it does not change lengths
    rng = np.random.default_rng(3)
    v = rng.standard_normal(64)
    assert abs(np.linalg.norm(rope(v, [17])[0]) - np.linalg.norm(v)) < 1e-12

    # ALiBi: a bigger slope means a shorter window
    rows = alibi_effective_window()
    assert rows[0]['window'] < rows[-1]['window']   # steeper slope, shorter window

    # FlexAttention reproduces the hand-written versions exactly
    rng = np.random.default_rng(4)
    Q, K, V = (rng.standard_normal((12, 16)), rng.standard_normal((12, 16)),
               rng.standard_normal((12, 8)))
    ref, _ = attention(Q, K, V, mask=np.tril(np.ones((12, 12), dtype=bool)))
    got, _ = flex_attention(Q, K, V, mask_mod=mod_causal())
    assert np.abs(ref - got).max() == 0.0
    sl = alibi_slopes(8)[0]
    ref2, _ = attention(Q, K, V, mask=np.tril(np.ones((12, 12), dtype=bool)),
                        bias=alibi_bias(12, sl))
    got2, _ = flex_attention(Q, K, V, score_mod=mod_alibi(sl), mask_mod=mod_causal())
    assert np.abs(ref2 - got2).max() == 0.0

    # ALiBi can never let a far token win; the oscillating schemes can
    fw = far_token_wins(trials=40)
    assert fw['alibi'] == 0.0 and fw['rope'] > 0.0

    # a block mask skips strictly more as the window gets smaller
    bs = block_sparsity()
    assert bs[0]['skipped'] == 0 and bs[1]['skipped'] < bs[2]['skipped']

    # the decode offset bug is silent: finite, normalised, and wrong
    r, w, gap2 = decode_offset_bug()
    assert np.isfinite(w).all() and abs(w.sum() - 1) < 1e-12 and gap2 > 0.05
    return True


# ================================================================ 8. the whole story
def header(title):
    print('\n' + '=' * 72)
    print(title)
    print('=' * 72)


def main():
    np.set_printoptions(suppress=True, precision=4, legacy='1.25')

    # ------------------------------------------------------------------ 1
    header('1. Attention cannot see order at all')
    gap = permutation_test()
    print('  Shuffle the input rows and every output row moves with them - the largest')
    print(f'  difference between "attend then permute" and "permute then attend" is')
    print(f'  {gap:.1e}, which is floating-point noise.\n')
    print('  Nothing in softmax(QK^T / sqrt(d)) V mentions an index.  A transformer is')
    print('  a set function until something puts the order back, and that something is')
    print('  the entire subject of this file.  Three ways to do it:')
    print('    sinusoidal   add a fixed vector to the input embedding')
    print('    RoPE         rotate q and k by an angle proportional to position')
    print('    ALiBi        add -slope * distance straight to the scores')

    # ------------------------------------------------------------------ 2
    header('2. Sinusoidal: a table of sines, and why sines')
    pe = sinusoidal(8, 8)
    print('  positions 0..7, d = 8 (each row is one position):')
    print(pe)
    print(f'\n  Shifting by k is a FIXED rotation of each (sin, cos) pair: the same')
    print('  matrix works at every position, so "five tokens back" is one linear map')
    print(f'  rather than a different relationship at every index.  Measured: {sinusoidal_shift_is_linear():.1e}')
    sim = sinusoidal_similarity()
    print(f'\n  Similarity to position 0 at distances 1, 2, 4, 8, 16, 32:')
    print('   ', ' '.join(f'{sim[t]:+.3f}' for t in (1, 2, 4, 8, 16, 32)))
    print('  It falls away with distance, but not monotonically - the sum of cosines')
    print('  wobbles.  That is a property of the table, not something anyone asked for.')

    # ------------------------------------------------------------------ 3
    header('3. RoPE: the position is a rotation, not an addition')
    a, b, g = rope_is_relative()
    print('  Rotate each (even, odd) channel pair of q and k by an angle proportional')
    print('  to the position.  The dot product then depends only on the DISTANCE:\n')
    print(f'    <RoPE(q, 3),  RoPE(k, 11)>  = {a:+.6f}')
    print(f'    <RoPE(q, 100), RoPE(k, 108)> = {b:+.6f}')
    print(f'    difference                   = {g:.1e}')
    print('\n  Both pairs are eight apart, and that is all the score can tell.  Two')
    print('  consequences: a rotation does not change the length of q or k (so it')
    print('  cannot blow anything up), and nothing is added to V - the values stay')
    print('  exactly what the layer computed.')
    wl = rope_wavelengths()
    print(f'\n  Wavelengths, d = 128: fastest channel pair turns every {wl.min():.2f} tokens,')
    print(f'  slowest every {wl.max():,.0f}.  The slow channels are the ones that carry')
    print('  long-range position - and the ones that have never completed a turn.')
    sc = rope_context_scaling()
    print(f'\n  Trained at 4,096 the slowest pair gets through {sc["turns_at_train"]:.2f} of a turn;')
    print(f'  asking for 32,768 would need {sc["turns_at_target"]:.2f} turns, i.e. angles it has')
    print('  never seen.  The two standard fixes aim at exactly that: position')
    print(f'  interpolation multiplies positions by {sc["interp_factor"]:.3f}, and raising the base')
    print(f'  from 10,000 to about {sc["ntk_base"]:,.0f} stretches the wavelengths instead.')

    # ------------------------------------------------------------------ 4
    header('4. ALiBi: no vectors, no rotations, just a penalty on distance')
    print('  Add -slope * (query position - key position) to the scores before the')
    print('  softmax, with one slope per head.  There is nothing to embed and nothing')
    print('  to train.\n')
    print(f'  {"head":>6} {"slope":>10} {"tokens holding 99% of the weight":>34}')
    for i, r in enumerate(alibi_effective_window()):
        print(f'  {i:>6} {r["slope"]:>10.4f} {r["window"]:>34,}')
    print('\n  (The last row is clipped by the 512-token probe, not by the slope.)')
    print('\n  With no content signal each head is a soft window of its own width, and')
    print('  the set of heads covers several scales at once.  Nothing changes when the')
    print('  sequence gets longer: the penalty is a function of distance, so a model')
    print('  that never saw position 10,000 has still seen "400 tokens back".')

    # ------------------------------------------------------------------ 5
    header('5. What survives past the trained length')
    fw = far_token_wins()
    print('  All three are formulas, so all three can be EVALUATED at a position the')
    print('  model never trained on - unlike a learned table, where row 4,097 of a')
    print('  4,096-row table simply does not exist.  Being evaluable is not the same as')
    print('  being meaningful.  Over 200 random (q, k) pairs, trained length 32, how')
    print('  often does some token between 33 and 512 away out-score everything inside')
    print('  the trained range?\n')
    print(f'  {"sinusoidal":>14} {fw["sinusoidal"] * 100:>6.1f}%')
    print(f'  {"RoPE":>14} {fw["rope"] * 100:>6.1f}%')
    print(f'  {"ALiBi":>14} {fw["alibi"] * 100:>6.1f}%')
    print('\n  Sinusoidal and RoPE keep oscillating forever, so a token 400 positions')
    print('  away can look more attractive than its neighbour - the model has to have')
    print('  learned to ignore that, and it only learned it for distances it saw.')
    print('  ALiBi cannot do it by construction: the penalty grows with distance, so')
    print('  "further" always means "weaker", at any length.  That is what people mean')
    print('  when they say ALiBi extrapolates for free - and it is also exactly what')
    print('  they give up, because a hard recency bias is not always what you want.')

    # ------------------------------------------------------------------ 6
    header('6. FlexAttention: the position scheme is an argument, not a kernel')
    print('  Every scheme above touches the same two things: the value of a score, and')
    print('  whether a pair is allowed at all.  Give attention those two as functions:\n')
    print('      s = Q @ K.T / sqrt(d)')
    print('      s = score_mod(s, q_index, k_index)')
    print('      s = where(mask_mod(q_index, k_index), s, -inf)\n')
    rng = np.random.default_rng(4)
    Q, K, V = (rng.standard_normal((12, 16)), rng.standard_normal((12, 16)),
               rng.standard_normal((12, 8)))
    msk = np.tril(np.ones((12, 12), dtype=bool))
    ref, _ = attention(Q, K, V, mask=msk)
    got, _ = flex_attention(Q, K, V, mask_mod=mod_causal())
    sl = alibi_slopes(8)[0]
    ref2, _ = attention(Q, K, V, mask=msk, bias=alibi_bias(12, sl))
    got2, _ = flex_attention(Q, K, V, score_mod=mod_alibi(sl), mask_mod=mod_causal())
    print(f'  causal, hand-written vs flex:        max difference {np.abs(ref - got).max():.1e}')
    print(f'  causal + ALiBi, hand vs flex:        max difference {np.abs(ref2 - got2).max():.1e}')
    print('  sliding window, soft-capping, prefix-LM, document masking: same two hooks.')
    print('\n  The point is not elegance, it is that the fused kernel survives.  Writing')
    print('  the mask by hand means materialising an n x n matrix; writing it as a')
    print('  function lets the kernel skip whole tiles that contain nothing:\n')
    print(f'  {"mask":>14} {"live blocks":>13} {"of":>5} {"skipped":>10}')
    for r in block_sparsity():
        print(f'  {r["name"]:>14} {r["live"]:>13} {r["total"]:>5} {r["skipped"] * 100:>9.1f}%')
    print('\n  At 1,024 tokens with 128-wide tiles, causal masking leaves 36 of 64 tiles')
    print('  to compute and a 256-token sliding window leaves 21.  That skipped work is')
    print('  the whole reason the API exists.')

    # ------------------------------------------------------------------ 7
    header('7. Silent failure: RoPE at the wrong position while decoding')
    right, wrong, g2 = decode_offset_bug()
    print('  Appending one token to a cache of 5, the new query sits at position 5.')
    print('  Rotating it at position 0 instead - because it is the first row of this')
    print('  step - is a one-line mistake.\n')
    print(f'  attention weights, correct position: {np.round(right, 4)}')
    print(f'  attention weights, position 0:       {np.round(wrong, 4)}')
    print(f'  largest difference: {g2:.4f}')
    print('\n  Both rows are finite, both sum to 1, both look like attention weights.')
    print('  Nothing raises, and the generated text is merely worse - which is why the')
    print('  test to write is "prefill then decode must equal prefill alone", comparing')
    print('  the two paths token by token rather than eyeballing the output.')

    # ------------------------------------------------------------------ 8
    header('8. Assertions')
    check()
    print('all assertions passed')


if __name__ == '__main__':
    main()
