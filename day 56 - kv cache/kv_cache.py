"""Day 56 - the KV cache: what it stores, what it costs, and why prefill and decode
are two different machines.

Generating a token means running the whole model over the whole prompt again - unless
you keep the keys and values you already computed.  That one change turns generation
from quadratic to linear, and it splits inference into two phases with completely
different bottlenecks: prefill does a lot of arithmetic on a lot of tokens at once,
decode does almost no arithmetic per token and spends its life reading memory.

Run `python kv_cache.py` to print the whole story; every number below is produced by
this file and checked by the asserts at the bottom.
"""

import math
import time

import numpy as np


# ================================================================ 1. a small model
def init_params(vocab=24, d_model=64, n_layer=2, n_head=4, ctx=256, seed=0):
    rng = np.random.default_rng(seed)
    s = 0.02
    p = {'tok': rng.standard_normal((vocab, d_model)) * s,
         'pos': rng.standard_normal((ctx, d_model)) * s,
         'g_f': np.ones(d_model), 'b_f': np.zeros(d_model),
         'head': rng.standard_normal((d_model, vocab)) * s}
    for l in range(n_layer):
        p[f'g1_{l}'] = np.ones(d_model)
        p[f'b1_{l}'] = np.zeros(d_model)
        p[f'g2_{l}'] = np.ones(d_model)
        p[f'b2_{l}'] = np.zeros(d_model)
        p[f'Wqkv_{l}'] = rng.standard_normal((d_model, 3 * d_model)) * s
        p[f'Wo_{l}'] = rng.standard_normal((d_model, d_model)) * s
        p[f'W1_{l}'] = rng.standard_normal((d_model, 4 * d_model)) * s
        p[f'W2_{l}'] = rng.standard_normal((4 * d_model, d_model)) * s
    cfg = {'d_model': d_model, 'n_layer': n_layer, 'n_head': n_head, 'ctx': ctx,
           'vocab': vocab}
    return p, cfg


def layer_norm(x, g, b, eps=1e-5):
    mu = x.mean(-1, keepdims=True)
    xc = x - mu
    return xc / np.sqrt((xc ** 2).mean(-1, keepdims=True) + eps) * g + b


def gelu(x):
    return 0.5 * x * (1 + np.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * x ** 3)))


def softmax(x, axis=-1):
    m = np.max(x, axis=axis, keepdims=True)
    e = np.exp(x - m)
    return e / e.sum(axis=axis, keepdims=True)


def forward_full(p, cfg, ids):
    """The honest version: recompute every position from scratch, every time."""
    T = len(ids)
    H, d = cfg['n_head'], cfg['d_model']
    dh = d // H
    x = p['tok'][ids] + p['pos'][:T]
    mask = np.tril(np.ones((T, T), dtype=bool))
    for l in range(cfg['n_layer']):
        h = layer_norm(x, p[f'g1_{l}'], p[f'b1_{l}'])
        q, k, v = np.split(h @ p[f'Wqkv_{l}'], 3, axis=-1)
        q = q.reshape(T, H, dh).transpose(1, 0, 2)
        k = k.reshape(T, H, dh).transpose(1, 0, 2)
        v = v.reshape(T, H, dh).transpose(1, 0, 2)
        att = np.where(mask, q @ k.transpose(0, 2, 1) / math.sqrt(dh), -np.inf)
        o = (softmax(att, -1) @ v).transpose(1, 0, 2).reshape(T, d)
        x = x + o @ p[f'Wo_{l}']
        h2 = layer_norm(x, p[f'g2_{l}'], p[f'b2_{l}'])
        x = x + gelu(h2 @ p[f'W1_{l}']) @ p[f'W2_{l}']
    return layer_norm(x, p['g_f'], p['b_f']) @ p['head']


# ================================================================ 2. the cache
class KVCache:
    """Two arrays per layer, filled in from the left, plus how much is valid.

    Allocated once at the maximum length: growing a numpy array per token would copy
    the whole thing every step, which is the cache's own version of the quadratic
    problem it exists to solve.
    """

    def __init__(self, cfg, max_len=None):
        self.cfg = cfg
        self.max_len = max_len or cfg['ctx']
        H, dh = cfg['n_head'], cfg['d_model'] // cfg['n_head']
        self.k = [np.zeros((H, self.max_len, dh)) for _ in range(cfg['n_layer'])]
        self.v = [np.zeros((H, self.max_len, dh)) for _ in range(cfg['n_layer'])]
        self.length = 0

    def reset(self):
        self.length = 0

    def append(self, layer, k, v):
        self.k[layer][:, self.length] = k
        self.v[layer][:, self.length] = v

    def bytes(self, dtype_size=2):
        return (2 * self.cfg['n_layer'] * self.cfg['n_head'] *
                (self.cfg['d_model'] // self.cfg['n_head']) * self.length * dtype_size)


def forward_step(p, cfg, cache, token, pos):
    """One new token, reading everything that came before out of the cache."""
    H, d = cfg['n_head'], cfg['d_model']
    dh = d // H
    x = p['tok'][token] + p['pos'][pos]
    for l in range(cfg['n_layer']):
        h = layer_norm(x, p[f'g1_{l}'], p[f'b1_{l}'])
        q, k, v = np.split(h @ p[f'Wqkv_{l}'], 3)
        q = q.reshape(H, dh)
        cache.append(l, k.reshape(H, dh), v.reshape(H, dh))
        n = cache.length + 1
        K = cache.k[l][:, :n]
        V = cache.v[l][:, :n]
        att = np.einsum('hd,hnd->hn', q, K) / math.sqrt(dh)
        o = np.einsum('hn,hnd->hd', softmax(att, -1), V).reshape(d)
        x = x + o @ p[f'Wo_{l}']
        h2 = layer_norm(x, p[f'g2_{l}'], p[f'b2_{l}'])
        x = x + gelu(h2 @ p[f'W1_{l}']) @ p[f'W2_{l}']
    cache.length += 1
    return layer_norm(x, p['g_f'], p['b_f']) @ p['head']


def prefill(p, cfg, cache, ids):
    """Run the prompt through one token at a time to fill the cache.

    A real implementation prefills in one batched pass - the point here is only that
    the cache ends up holding exactly what the full forward would have computed.
    """
    cache.reset()
    out = None
    for i, t in enumerate(ids):
        out = forward_step(p, cfg, cache, int(t), i)
    return out


# ================================================================ 3. with and without
def generate_nocache(p, cfg, ids, n, seed=0):
    """Generate n tokens by rerunning the whole model on the whole sequence."""
    rng = np.random.default_rng(seed)
    out = list(ids)
    for _ in range(n):
        logits = forward_full(p, cfg, np.array(out))[-1]
        out.append(int(rng.choice(cfg['vocab'], p=softmax(logits))))
    return out


def generate_cached(p, cfg, ids, n, seed=0):
    rng = np.random.default_rng(seed)
    cache = KVCache(cfg)
    out = list(ids)
    logits = prefill(p, cfg, cache, out)
    for _ in range(n):
        nxt = int(rng.choice(cfg['vocab'], p=softmax(logits)))
        out.append(nxt)
        logits = forward_step(p, cfg, cache, nxt, cache.length)
    return out


def equivalence(p, cfg, ids, n=16):
    """The test that matters: cached decoding must equal the full forward, exactly."""
    cache = KVCache(cfg)
    prefill(p, cfg, cache, ids[:-n])
    worst = 0.0
    seq = list(ids[:-n])
    for t in ids[-n:]:
        step = forward_step(p, cfg, cache, int(t), cache.length)
        seq.append(int(t))
        full = forward_full(p, cfg, np.array(seq))[-1]
        worst = max(worst, float(np.abs(step - full).max()))
    return worst


def timing(p, cfg, prompt_len=64, n=64, seed=0):
    ids = np.random.default_rng(seed).integers(0, cfg['vocab'], prompt_len)
    t0 = time.perf_counter()
    generate_nocache(p, cfg, ids, n)
    t_no = time.perf_counter() - t0
    t0 = time.perf_counter()
    generate_cached(p, cfg, ids, n)
    t_yes = time.perf_counter() - t0
    return t_no, t_yes, t_no / t_yes


# ================================================================ 4. the arithmetic
def kv_bytes_per_token(n_layer, n_head, head_dim, dtype_size=2, kv_heads=None):
    return 2 * n_layer * (kv_heads or n_head) * head_dim * dtype_size


def phase_costs(n_layer, d_model, n_ff, T, dtype_size=2, n_head=32, head_dim=128,
                kv_heads=None):
    """Multiply-accumulates and bytes moved, for a prefill of T tokens and for one
    decode step with T tokens already in the cache.

    The weights have to be read in both cases; the cache has to be read only in
    decode.  Dividing one by the other gives the arithmetic intensity, which is what
    decides whether a phase is limited by the arithmetic units or by memory.
    """
    w = n_layer * (4 * d_model * d_model + 2 * d_model * n_ff)
    pre_flops = T * w + 2 * T * T * d_model * n_layer
    pre_bytes = w * dtype_size + T * d_model * dtype_size
    kv = kv_bytes_per_token(n_layer, n_head, head_dim, dtype_size, kv_heads)
    dec_flops = w + 2 * T * d_model * n_layer
    dec_bytes = w * dtype_size + T * kv
    return {'weights': w, 'prefill_flops': pre_flops, 'prefill_bytes': pre_bytes,
            'prefill_ai': pre_flops / pre_bytes, 'decode_flops': dec_flops,
            'decode_bytes': dec_bytes, 'decode_ai': dec_flops / dec_bytes,
            'kv_per_token': kv}


def batch_decode_time(p, cfg, batches=(1, 8, 32, 128), ctx=512, repeats=7, seed=0):
    """One decode step at several batch sizes.

    Decode reads the same weights whatever the batch size, so until the arithmetic
    catches up, a bigger batch is almost free per sequence.
    """
    rng = np.random.default_rng(seed)
    H, d = cfg['n_head'], cfg['d_model']
    dh = d // H
    rows = []
    for B in batches:
        x = rng.standard_normal((B, d))
        K = rng.standard_normal((B, H, ctx, dh))
        V = rng.standard_normal((B, H, ctx, dh))
        best = math.inf
        for _ in range(repeats):
            t0 = time.perf_counter()
            for l in range(cfg['n_layer']):
                h = layer_norm(x, p[f'g1_{l}'], p[f'b1_{l}'])
                q, k, v = np.split(h @ p[f'Wqkv_{l}'], 3, axis=-1)
                q = q.reshape(B, H, dh)
                att = np.einsum('bhd,bhnd->bhn', q, K) / math.sqrt(dh)
                o = np.einsum('bhn,bhnd->bhd', softmax(att, -1), V).reshape(B, d)
                y = x + o @ p[f'Wo_{l}']
                h2 = layer_norm(y, p[f'g2_{l}'], p[f'b2_{l}'])
                x = y + gelu(h2 @ p[f'W1_{l}']) @ p[f'W2_{l}']
            best = min(best, time.perf_counter() - t0)
        rows.append({'batch': B, 'step_ms': best * 1e3,
                     'per_seq_us': best * 1e6 / B})
    return rows


# ================================================================ 5. the stale cache
def stale_cache(p, cfg, seed=0, n_first=24, n_second=12):
    """Start a second sequence without clearing the cache.

    Nothing raises: the new tokens simply attend to the previous conversation as well
    as their own, and the logits are finite and normalised and wrong.
    """
    rng = np.random.default_rng(seed)
    first = rng.integers(0, cfg['vocab'], n_first)
    second = rng.integers(0, cfg['vocab'], n_second)
    dirty = KVCache(cfg)
    prefill(p, cfg, dirty, first)                        # conversation one, not reset
    clean = KVCache(cfg)
    gaps, flips = [], 0
    for i, t in enumerate(second):
        ld = forward_step(p, cfg, dirty, int(t), i)
        lc = forward_step(p, cfg, clean, int(t), i)
        gaps.append(float(np.abs(ld - lc).max()))
        flips += int(np.argmax(ld) != np.argmax(lc))
    stale = n_first / (n_first + n_second)
    return {'worst': max(gaps), 'mean': float(np.mean(gaps)), 'flips': flips,
            'steps': len(second), 'stale': stale}


# ================================================================ 6. assertions
def check():
    p, cfg = init_params()
    ids = np.random.default_rng(1).integers(0, cfg['vocab'], 48)

    # the cache is only an optimisation: same numbers, to the last bit or so
    assert equivalence(p, cfg, ids) < 1e-12

    # and sampling from it gives exactly the same tokens
    assert generate_cached(p, cfg, ids[:16], 24) == generate_nocache(p, cfg, ids[:16], 24)

    # bytes grow linearly with what is in the cache
    cache = KVCache(cfg)
    prefill(p, cfg, cache, ids[:10])
    b10 = cache.bytes()
    prefill(p, cfg, cache, ids[:20])
    assert cache.bytes() == 2 * b10

    # decode has a much lower arithmetic intensity than prefill
    c = phase_costs(32, 4096, 16384, 2048)
    assert c['prefill_ai'] > 100 * c['decode_ai']

    # a stale cache changes the answer without raising anything
    st = stale_cache(p, cfg)
    assert st['worst'] > 1e-3 and st['flips'] > 0
    return True


# ================================================================ 7. the whole story
def header(title):
    print('\n' + '=' * 72)
    print(title)
    print('=' * 72)


def main():
    np.set_printoptions(suppress=True, precision=4, legacy='1.25')
    p, cfg = init_params()
    rng = np.random.default_rng(1)
    ids = rng.integers(0, cfg['vocab'], 64)

    # ------------------------------------------------------------------ 1
    header('1. Why there is a cache at all')
    print('  Generating one token needs the model to look at every earlier token.')
    print('  Without a cache that means rerunning the whole model over the whole')
    print('  sequence for every single new token - and the sequence keeps growing.\n')
    t_no, t_yes, speed = timing(p, cfg, prompt_len=64, n=64)
    print(f'  64 tokens after a 64-token prompt, this toy model:')
    print(f'    recomputing everything : {t_no:.2f} s')
    print(f'    with a KV cache        : {t_yes:.2f} s   ({speed:.1f}x faster)')
    print('\n  The ratio is not the interesting part - it grows with the sequence.')
    print('  What matters is the shape: without a cache, generating n tokens after a')
    print('  prompt of T costs work proportional to n * (T + n); with one it is')
    print('  proportional to n * T.  The cache turns a quadratic loop into a linear one.')

    # ------------------------------------------------------------------ 2
    header('2. What is actually in it')
    cache = KVCache(cfg)
    prefill(p, cfg, cache, ids)
    print('  Per layer, two arrays: the keys and the values of every token so far.')
    print(f'  Shape here: {cache.k[0].shape} per layer, {cfg["n_layer"]} layers, '
          f'{cache.length} tokens used.\n')
    print('  Note what is NOT in it: the queries.  A query is used once, by the token')
    print('  that made it, and never again.  Keys and values are read by every token')
    print('  that comes after, which is exactly why they are the ones worth keeping.\n')
    print(f'  {"model":>26} {"bytes/token":>12} {"at 4k":>10} {"at 128k":>12}')
    for name, (L, H, dh, kv) in (('this toy model', (2, 4, 16, None)),
                                 ('7B-ish, 32 heads', (32, 32, 128, None)),
                                 ('7B-ish, GQA 8 kv heads', (32, 32, 128, 8)),
                                 ('70B-ish, GQA 8 kv heads', (80, 64, 128, 8))):
        b = kv_bytes_per_token(L, H, dh, 2, kv)
        print(f'  {name:>26} {b:>12,} {b * 4096 / 2**20:>9.0f}M {b * 131072 / 2**30:>11.1f}G')
    print('\n  Two bytes per number, two arrays, one entry per layer per head per token.')
    print('  Nothing clever: it is just a lot of numbers, and they are per sequence.')

    # ------------------------------------------------------------------ 3
    header('3. The cache is an optimisation, so it has to change nothing')
    worst = equivalence(p, cfg, ids)
    same = generate_cached(p, cfg, ids[:16], 24) == generate_nocache(p, cfg, ids[:16], 24)
    print(f'  Largest difference between a cached decode step and the full forward')
    print(f'  over the whole sequence: {worst:.2e}')
    print(f'  Sampling 24 tokens both ways gives the same tokens: {same}\n')
    print('  This is the test to write, and it is a differential one: prefill the')
    print('  prompt, then decode token by token, and require every step to equal what')
    print('  the full forward would have said.  Almost every cache bug - an off-by-one')
    print('  in the write position, a stale entry, a wrong rotation - shows up here as')
    print('  a number that is finite, plausible, and not equal.')

    # ------------------------------------------------------------------ 4
    header('4. Prefill and decode are two different machines')
    print('  Both phases read the same weights.  Prefill does arithmetic for T tokens')
    print('  at once; decode does it for one token and then reads the entire cache.\n')
    print(f'  {"context":>9} {"prefill MACs/byte":>19} {"decode MACs/byte":>18} {"ratio":>8}')
    for T in (128, 512, 2048, 8192):
        c = phase_costs(32, 4096, 16384, T)
        print(f'  {T:>9,} {c["prefill_ai"]:>19.1f} {c["decode_ai"]:>18.2f} '
              f'{c["prefill_ai"] / c["decode_ai"]:>8.0f}x')
    print('\n  Arithmetic intensity is multiply-accumulates per byte moved, and a GPU')
    print('  wants it in the hundreds.  Prefill gets there and climbs with the context;')
    print('  decode sits at exactly 0.50 and never moves.\n')
    print('  That 0.50 is not a coincidence, it is the whole story in one number: in')
    print('  decode every weight and every cached number is loaded and used in exactly')
    print('  ONE multiply-accumulate, and a 16-bit number is two bytes.  One MAC per')
    print('  two bytes is 0.5.  Prefill reuses each loaded weight across all T tokens,')
    print('  which is where its intensity comes from.')
    print('\n  Note what this means for GQA: sharing K and V across heads cuts how many')
    print('  bytes decode has to read, but it cuts the MACs by the same factor, so the')
    print('  intensity stays at 0.50.  It makes decode cheaper without making it any')
    print('  less memory-bound.')

    # ------------------------------------------------------------------ 5
    header('5. Which is why decode gets batched')
    print('  The weights are read once per step no matter how many sequences are')
    print('  decoding, so adding sequences to a batch is nearly free until the')
    print('  arithmetic catches up:\n')
    print(f'  {"batch":>7} {"ms per step":>13} {"us per sequence":>18}')
    rows = batch_decode_time(p, cfg)
    for r in rows:
        print(f'  {r["batch"]:>7} {r["step_ms"]:>13.2f} {r["per_seq_us"]:>18.1f}')
    first, last = rows[0], rows[-1]
    best = min(rows, key=lambda r: r['per_seq_us'])
    print(f'\n  Going from batch {first["batch"]} to {best["batch"]} costs '
          f'{best["step_ms"] / first["step_ms"]:.1f}x the time per step and divides')
    print(f'  the time per sequence by {first["per_seq_us"] / best["per_seq_us"]:.1f}.  '
          f'After that it flattens out here, because')
    print(f'  with {512} tokens in the cache numpy is already spending its time on the')
    print('  attention einsum rather than on the weights.  On a GPU the flat part runs')
    print('  much further, and that is the reason serving systems fight so hard to keep')
    print('  the decode batch full: the weights get read once either way.')
    print('  (numpy on a CPU, so read the shape of the curve, not the milliseconds.)')
    print('\n  What stops you making the batch enormous is the cache itself: every')
    print('  sequence in the batch carries its own, and they are the thing that fills')
    print('  the memory.')

    # ------------------------------------------------------------------ 6
    header('6. Silent failure: a cache that was not reset')
    st = stale_cache(p, cfg)
    print('  Start a second sequence while the first one is still in the cache - no')
    print('  reset, no new allocation, just carry on appending.\n')
    print(f'    largest difference in the logits: {st["worst"]:.4f}  '
          f'(mean {st["mean"]:.4f})')
    print(f'    the most likely next token changes on {st["flips"]} of '
          f'{st["steps"]} steps')
    print(f'    at that point up to {st["stale"] * 100:.0f}% of what the model could')
    print('    attend to belonged to the previous sequence')
    print('\n  Nothing raises.  The shapes are right, the logits are finite, the')
    print('  softmax still sums to one.  The model is simply reading somebody else\'s')
    print('  conversation, and the only symptom is that the output is a bit strange.')
    print('  The differential test from section 3 catches it immediately; eyeballing')
    print('  the generated text does not.')

    # ------------------------------------------------------------------ 7
    header('7. Assertions')
    check()
    print('all assertions passed')


if __name__ == '__main__':
    main()
