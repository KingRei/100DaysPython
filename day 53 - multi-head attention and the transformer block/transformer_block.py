"""Day 53 - multi-head attention and the transformer block, from scratch.

One head mixes the sequence one way.  A block with h heads mixes it h ways at the same
parameter count, adds a position-wise MLP, and wraps both in residual connections and a
normalisation layer.  That is the whole architecture - and almost every way of getting
it slightly wrong produces an array of exactly the right shape.

Run `python transformer_block.py` to print the whole story; every number below is
produced by this file, and the implementations are checked by the asserts at the bottom.
"""

import math

import numpy as np


# ================================================================ 1. attention
def softmax(x, axis=-1):
    x = np.asarray(x, dtype=float)
    m = np.max(x, axis=axis, keepdims=True)
    e = np.exp(x - m)
    return e / e.sum(axis=axis, keepdims=True)


def attention(Q, K, V, mask=None):
    """Day 52's four lines, now applied per head."""
    d = Q.shape[-1]
    s = Q @ np.swapaxes(K, -1, -2) / math.sqrt(d)
    if mask is not None:
        s = np.where(mask, s, -np.inf)
    w = softmax(s, axis=-1)
    return w @ V, w


# ================================================================ 2. splitting heads
def split_heads(X, h):
    """(n, d_model) -> (h, n, d_head), the correct way.

    Head j owns the contiguous slice X[:, j*d_head:(j+1)*d_head].  Reshaping to
    (n, h, d_head) groups the feature axis first and then swaps it to the front, so each
    head keeps its own block of features for every position.
    """
    n, d = X.shape
    return X.reshape(n, h, d // h).transpose(1, 0, 2)


def split_heads_wrong(X, h):
    """(n, d_model) -> (h, n, d_head) by reshaping the wrong axis first.

    This hands head 0 the first n/h POSITIONS instead of the first d/h FEATURES.  The
    shape is identical, every later line runs, and the gradients flow.
    """
    n, d = X.shape
    return X.reshape(h, n, d // h)


def merge_heads(H):
    """(h, n, d_head) -> (n, d_model), the inverse of split_heads."""
    h, n, dh = H.shape
    return H.transpose(1, 0, 2).reshape(n, h * dh)


def multi_head_attention(X, Wq, Wk, Wv, Wo, h, mask=None, split=split_heads):
    """The layer: project, split into h heads, attend per head, concatenate, project."""
    Q, K, V = split(X @ Wq, h), split(X @ Wk, h), split(X @ Wv, h)
    out, w = attention(Q, K, V, mask=mask)
    return merge_heads(out) @ Wo, w


def head_params(d_model, h):
    """Four d_model x d_model matrices, whatever h is - h only changes how they split."""
    return {'per_head_dim': d_model // h,
            'Wq': d_model * d_model, 'total': 4 * d_model * d_model,
            'score_entries_per_layer': h}


# ================================================================ 3. what h heads buy
def two_pattern_target(n=8, d=8, seed=0):
    """A target this block has to produce: one half of the channels wants the previous
    token, the other half wants the first token.

    Both are things attention can do.  The question is whether ONE head can do both at
    once, because a head applies the same weight matrix to every channel it carries.
    """
    rng = np.random.default_rng(seed)
    V = rng.standard_normal((n, d))
    prev = np.eye(n, k=-1)
    prev[0, 0] = 1.0                       # position 0 has no previous token
    first = np.zeros((n, n))
    first[:, 0] = 1.0
    target = np.empty_like(V)
    target[:, :d // 2] = (prev @ V)[:, :d // 2]
    target[:, d // 2:] = (first @ V)[:, d // 2:]
    return V, prev, first, target


def best_single_head(m=12, n=8, d=8, steps=4000, lr=0.5, seed=0):
    """The best a single head could do, given everything a real head has.

    A real head computes A (X Wv) Wo = A X M: one attention matrix A, and one channel
    matrix M that can route features freely.  This fits both - M shared across m
    different inputs, and A chosen per input, which is more than a real head gets
    (its A has to come out of Q and K).  The one thing it cannot do is use a different
    A for different channels, and that is the whole point.
    """
    rng = np.random.default_rng(seed)
    Vs, Ts, Ls = [], [], []
    prev = np.eye(n, k=-1)
    prev[0, 0] = 1.0
    first = np.zeros((n, n))
    first[:, 0] = 1.0
    for t in range(m):
        V = rng.standard_normal((n, d))
        T = np.empty_like(V)
        T[:, :d // 2] = (prev @ V)[:, :d // 2]
        T[:, d // 2:] = (first @ V)[:, d // 2:]
        Vs.append(V)
        Ts.append(T)
        Ls.append(rng.standard_normal((n, n)) * 0.1)
    M = np.eye(d)
    size = m * n * d
    for _ in range(steps):
        dM = np.zeros_like(M)
        grads = []
        for V, T, L in zip(Vs, Ts, Ls):
            A = softmax(L, axis=-1)
            Z = A @ V
            R = Z @ M - T
            dM += 2 * Z.T @ R / size
            dZ = 2 * R @ M.T / size
            dA = dZ @ V.T
            grads.append(A * (dA - (dA * A).sum(axis=-1, keepdims=True)))
        for L, g in zip(Ls, grads):
            L -= lr * g
        M -= lr * dM
    err = 0.0
    for V, T, L in zip(Vs, Ts, Ls):
        err += float(np.mean((softmax(L, axis=-1) @ V @ M - T) ** 2))
    return softmax(Ls[0], axis=-1), M, err / m


def one_input_exact(n=8, d=8, seed=0):
    """Why the comparison above needs more than one input.

    With a single input and n == d, the matrix A @ V is square and invertible, so
    solving (A V) M = target for M alone fits the target exactly for essentially any
    attention matrix A.  A result like that is about the toy, not about heads.
    """
    V, prev, first, target = two_pattern_target(n, d, seed)
    A = softmax(np.random.default_rng(seed + 1).standard_normal((n, n)), axis=-1)
    M = np.linalg.solve(A @ V, target)
    return float(np.mean((A @ V @ M - target) ** 2))


def two_head_error_many(m=12, n=8, d=8, seed=0):
    """The same m inputs through two heads with fixed patterns, measured the same way."""
    rng = np.random.default_rng(seed)
    prev = np.eye(n, k=-1)
    prev[0, 0] = 1.0
    first = np.zeros((n, n))
    first[:, 0] = 1.0
    err = 0.0
    for _ in range(m):
        V = rng.standard_normal((n, d))
        T = np.empty_like(V)
        T[:, :d // 2] = (prev @ V)[:, :d // 2]
        T[:, d // 2:] = (first @ V)[:, d // 2:]
        out = np.empty_like(V)
        out[:, :d // 2] = (prev @ V)[:, :d // 2]
        out[:, d // 2:] = (first @ V)[:, d // 2:]
        err += float(np.mean((out - T) ** 2))
    return err / m


def target_scale(m=12, n=8, d=8, seed=0):
    """What predicting zero everywhere would score on the same m inputs."""
    rng = np.random.default_rng(seed)
    prev = np.eye(n, k=-1)
    prev[0, 0] = 1.0
    first = np.zeros((n, n))
    first[:, 0] = 1.0
    err = 0.0
    for _ in range(m):
        V = rng.standard_normal((n, d))
        T = np.empty_like(V)
        T[:, :d // 2] = (prev @ V)[:, :d // 2]
        T[:, d // 2:] = (first @ V)[:, d // 2:]
        err += float(np.mean(T ** 2))
    return err / m


def two_head_error(V, prev, first, target):
    """Two heads, each carrying half the channels, reproduce the target exactly."""
    out = np.empty_like(V)
    half = V.shape[1] // 2
    out[:, :half] = (prev @ V)[:, :half]
    out[:, half:] = (first @ V)[:, half:]
    return float(np.mean((out - target) ** 2))


# ================================================================ 4. normalisation
def layer_norm(x, gamma=None, beta=None, eps=1e-5):
    """Centre, scale to unit variance, then rescale with learned gamma and beta."""
    mu = x.mean(axis=-1, keepdims=True)
    var = ((x - mu) ** 2).mean(axis=-1, keepdims=True)
    y = (x - mu) / np.sqrt(var + eps)
    if gamma is not None:
        y = y * gamma
    if beta is not None:
        y = y + beta
    return y


def layer_norm_naive_var(x, eps=1e-5):
    """The same layer, with the variance computed as E[x^2] - E[x]^2.

    One pass instead of two, which is why it keeps being written - and two large numbers
    subtracted to get a small one, which is why it stops working.
    """
    mu = x.mean(axis=-1, keepdims=True)
    var = (x ** 2).mean(axis=-1, keepdims=True) - mu ** 2
    return (x - mu) / np.sqrt(var + eps), var


def rms_norm(x, gamma=None, eps=1e-5):
    """RMSNorm: divide by the root mean square, never subtract the mean."""
    r = np.sqrt((x ** 2).mean(axis=-1, keepdims=True) + eps)
    y = x / r
    return y * gamma if gamma is not None else y


def cancellation_table(offsets=(0.0, 1e2, 1e3, 1e4, 1e5, 1e6), n=512, seed=0, dtype=np.float32):
    """How a constant offset - exactly what a residual stream accumulates - destroys the
    one-pass variance while leaving the two-pass one alone."""
    rng = np.random.default_rng(seed)
    base = rng.standard_normal(n).astype(dtype)
    rows = []
    for off in offsets:
        x = (base + dtype(off)).astype(dtype)
        two = float(((x - x.mean()) ** 2).mean())
        one = float((x.astype(dtype) ** 2).mean() - x.mean() ** 2)
        rows.append({'offset': off, 'two_pass': two, 'one_pass': one,
                     'distinct': int(len(np.unique(x))),
                     'rel_err': abs(one - two) / two})
    return rows


# ================================================================ 5. the block
def gelu(x):
    """The tanh approximation, which is what the original implementations shipped."""
    return 0.5 * x * (1 + np.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * x ** 3)))


def relu(x):
    return np.maximum(x, 0.0)


def ffn(x, W1, b1, W2, b2, act=gelu):
    """Position-wise MLP: expand by 4x, non-linearity, project back."""
    return act(x @ W1 + b1) @ W2 + b2


def block_pre_ln(x, p, mask=None):
    """Pre-LN: normalise the branch input, leave the residual path untouched."""
    a, _ = multi_head_attention(layer_norm(x, p['g1'], p['b1']), p['Wq'], p['Wk'],
                                p['Wv'], p['Wo'], p['h'], mask=mask)
    x = x + a
    return x + ffn(layer_norm(x, p['g2'], p['b2']), p['W1'], p['bb1'], p['W2'], p['bb2'])


def block_post_ln(x, p, mask=None):
    """Post-LN: normalise after the add, so the residual path goes through the norm."""
    a, _ = multi_head_attention(x, p['Wq'], p['Wk'], p['Wv'], p['Wo'], p['h'], mask=mask)
    x = layer_norm(x + a, p['g1'], p['b1'])
    return layer_norm(x + ffn(x, p['W1'], p['bb1'], p['W2'], p['bb2']), p['g2'], p['b2'])


def make_params(d_model=64, h=8, seed=0, scale=None):
    """He-ish initialisation: every projection scaled by 1/sqrt(fan_in)."""
    rng = np.random.default_rng(seed)
    s = (1 / math.sqrt(d_model)) if scale is None else scale
    d_ff = 4 * d_model
    return {'h': h,
            'Wq': rng.standard_normal((d_model, d_model)) * s,
            'Wk': rng.standard_normal((d_model, d_model)) * s,
            'Wv': rng.standard_normal((d_model, d_model)) * s,
            'Wo': rng.standard_normal((d_model, d_model)) * s,
            'W1': rng.standard_normal((d_model, d_ff)) * s,
            'bb1': np.zeros(d_ff),
            'W2': rng.standard_normal((d_ff, d_model)) / math.sqrt(d_ff),
            'bb2': np.zeros(d_model),
            'g1': np.ones(d_model), 'b1': np.zeros(d_model),
            'g2': np.ones(d_model), 'b2': np.zeros(d_model)}


def depth_signal(layers=32, n=16, d_model=64, h=8, seed=0, eps=1e-3):
    """Push a tiny perturbation through L blocks and watch how much of it survives.

    How far a change at the input still moves the output is the forward-mode mirror of
    how much gradient gets back to the first layer, and it needs no autograd.
    """
    rng = np.random.default_rng(seed)
    x0 = rng.standard_normal((n, d_model))
    dx = rng.standard_normal((n, d_model))
    dx = eps * dx / np.linalg.norm(dx)
    ps = [make_params(d_model, h, seed=100 + i) for i in range(layers)]
    out = {}
    for name, fn in (('pre', block_pre_ln), ('post', block_post_ln)):
        a, b = x0.copy(), x0 + dx
        trace = []
        for p in ps:
            a, b = fn(a, p), fn(b, p)
            amp = float(np.linalg.norm(b - a) / np.linalg.norm(dx))
            trace.append({'norm': float(np.linalg.norm(a)), 'signal': amp,
                          'relative': amp / float(np.linalg.norm(a))})
        out[name] = trace
    return out


def residual_growth(layers=32, n=16, d_model=64, h=8, seed=0, branch_scale=None):
    """The norm of the residual stream before the first block and after each one,
    with and without the 1/sqrt(2L) scaling applied to what each branch writes."""
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n, d_model))
    x = x / np.linalg.norm(x) * math.sqrt(n * d_model)
    ps = [make_params(d_model, h, seed=200 + i) for i in range(layers)]
    s = 1.0 if branch_scale is None else branch_scale
    start = float(np.linalg.norm(x))
    norms = []
    for p in ps:
        a, _ = multi_head_attention(layer_norm(x), p['Wq'], p['Wk'], p['Wv'], p['Wo'], p['h'])
        x = x + s * a
        x = x + s * ffn(layer_norm(x), p['W1'], p['bb1'], p['W2'], p['bb2'])
        norms.append(float(np.linalg.norm(x)))
    return {'start': start, 'norms': norms}


# ================================================================ 6. the budget
def block_params(d_model, d_ff=None, h=8):
    """Parameters in one block, split between attention and the MLP."""
    d_ff = 4 * d_model if d_ff is None else d_ff
    attn = 4 * d_model * d_model
    mlp = 2 * d_model * d_ff
    return {'attn': attn, 'mlp': mlp, 'total': attn + mlp,
            'mlp_share': mlp / (attn + mlp), 'd_ff': d_ff, 'h': h}


def block_flops(n, d_model, d_ff=None):
    """Multiply-accumulates for one block on a sequence of n tokens.

    The projections and the MLP are linear in n.  The two attention matmuls (scores and
    the weighted sum) are quadratic, and they do not depend on the number of heads.
    """
    d_ff = 4 * d_model if d_ff is None else d_ff
    linear = n * (4 * d_model * d_model + 2 * d_model * d_ff)
    quad = 2 * n * n * d_model
    return {'n': n, 'linear': linear, 'quad': quad, 'total': linear + quad,
            'quad_share': quad / (linear + quad)}


def flop_crossover(d_model, d_ff=None):
    """The context length at which the quadratic part catches the linear part."""
    d_ff = 4 * d_model if d_ff is None else d_ff
    return (4 * d_model * d_model + 2 * d_model * d_ff) / (2 * d_model)


# ================================================================ 7. LeetCode 189
class SolutionRotateArray:
    """LeetCode 189 - Rotate Array.

    Reverse the whole array, then reverse each of the two pieces.  No extra array, and
    exactly the kind of in-place index surgery that splitting heads is: the bytes do not
    move for a reshape, they do move for a transpose, and knowing which is which is the
    difference between a free operation and a copy.
    """

    def rotate(self, nums: list[int], k: int) -> None:
        n = len(nums)
        k %= n                                  # k can exceed n
        self._rev(nums, 0, n - 1)
        self._rev(nums, 0, k - 1)
        self._rev(nums, k, n - 1)

    @staticmethod
    def _rev(a, i, j):
        while i < j:
            a[i], a[j] = a[j], a[i]
            i, j = i + 1, j - 1


def rotate_cycles(nums, k):
    """The other in-place method: walk the permutation cycles.

    Correct, and it needs gcd(n, k) separate starting points - forget that and you move
    one cycle and leave the rest untouched.
    """
    n = len(nums)
    k %= n
    if k == 0:
        return nums
    moved = 0
    start = 0
    while moved < n:
        cur, val = start, nums[start]
        while True:
            nxt = (cur + k) % n
            nums[nxt], val = val, nums[nxt]
            cur = nxt
            moved += 1
            if cur == start:
                break
        start += 1
    return nums


def rotate_one_cycle(nums, k):
    """The same walk started only at index 0: right for some k, silently partial for
    others, because it visits only one of the gcd(n, k) cycles."""
    n = len(nums)
    k %= n
    if k == 0:
        return nums
    cur, val = 0, nums[0]
    while True:
        nxt = (cur + k) % n
        nums[nxt], val = val, nums[nxt]
        cur = nxt
        if cur == 0:
            break
    return nums


# ================================================================ 8. LeetCode 303
class NumArray:
    """LeetCode 303 - Range Sum Query (immutable).

    One prefix-sum pass up front turns every later range query into one subtraction.
    """

    def __init__(self, nums: list[int]):
        self.pre = [0]
        for v in nums:
            self.pre.append(self.pre[-1] + v)

    def sumRange(self, left: int, right: int) -> int:
        return self.pre[right + 1] - self.pre[left]


def window_variance_prefix(x, lo, hi):
    """Variance of a window from prefix sums of x and x^2 - one subtraction each.

    It is the same identity LayerNorm's one-pass variance uses, and it fails the same
    way: two large numbers differenced to produce a small one.
    """
    x = np.asarray(x, dtype=np.float32)
    p1 = np.concatenate([[0], np.cumsum(x, dtype=np.float32)])
    p2 = np.concatenate([[0], np.cumsum(x * x, dtype=np.float32)])
    m = hi - lo
    s1 = p1[hi] - p1[lo]
    s2 = p2[hi] - p2[lo]
    return float(s2 / m - (s1 / m) ** 2)


# ================================================================ 9. seeded tables
def head_split_demo(n=4, d=8, h=2, seed=3):
    """One small input split both ways, so the difference is visible element by element."""
    rng = np.random.default_rng(seed)
    X = np.round(rng.standard_normal((n, d)), 2)
    return X, split_heads(X, h), split_heads_wrong(X, h)


def split_output_gap(n=16, d_model=64, h=8, seed=5):
    """How far apart the two splittings put the layer's output."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, d_model))
    p = make_params(d_model, h, seed=seed)
    good, _ = multi_head_attention(X, p['Wq'], p['Wk'], p['Wv'], p['Wo'], h)
    bad, _ = multi_head_attention(X, p['Wq'], p['Wk'], p['Wv'], p['Wo'], h,
                                  split=split_heads_wrong)
    return (float(np.abs(good - bad).max()),
            float(np.linalg.norm(good - bad) / np.linalg.norm(good)),
            good.shape == bad.shape)


def head_count_table(d_model=64, hs=(1, 2, 4, 8, 16, 64)):
    return [dict(h=h, **head_params(d_model, h)) for h in hs]


def budget_table(configs=((768, 12), (2048, 16), (4096, 32), (8192, 64))):
    rows = []
    for d, h in configs:
        p = block_params(d, h=h)
        rows.append({'d_model': d, 'h': h, 'attn': p['attn'], 'mlp': p['mlp'],
                     'total': p['total'], 'mlp_share': p['mlp_share'],
                     'crossover': flop_crossover(d)})
    return rows


def flop_table(d_model=4096, ns=(512, 2048, 8192, 32768)):
    return [block_flops(n, d_model) for n in ns]


# ================================================================ 10. assertions
def check():
    rng = np.random.default_rng(0)

    # splitting and merging are inverses, and the correct split keeps feature blocks
    X = rng.standard_normal((6, 12))
    assert np.allclose(merge_heads(split_heads(X, 4)), X)
    assert np.allclose(split_heads(X, 4)[2], X[:, 6:9])          # head 2 owns features 6..8
    assert split_heads_wrong(X, 4).shape == split_heads(X, 4).shape
    assert not np.allclose(split_heads_wrong(X, 4), split_heads(X, 4))

    # one head of the full width is plain single-head attention
    d = 16
    p = make_params(d, 1, seed=1)
    Y = rng.standard_normal((5, d))
    one, w = multi_head_attention(Y, p['Wq'], p['Wk'], p['Wv'], p['Wo'], 1)
    ref, wref = attention(Y @ p['Wq'], Y @ p['Wk'], Y @ p['Wv'])
    assert np.allclose(one, ref @ p['Wo']) and np.allclose(w[0], wref)

    # layer norm really does centre and scale; RMSNorm does not centre
    Z = rng.standard_normal((7, 32)) * 3 + 5
    ln = layer_norm(Z)
    assert np.allclose(ln.mean(axis=-1), 0, atol=1e-8)
    assert np.allclose(ln.std(axis=-1), 1, atol=1e-4)
    assert abs(rms_norm(Z).mean()) > 1e-3
    # both are scale invariant, only layer norm is shift invariant
    assert np.allclose(layer_norm(Z), layer_norm(2 * Z), atol=1e-6)
    assert np.allclose(layer_norm(Z), layer_norm(Z + 100), atol=1e-4)
    assert np.allclose(rms_norm(Z), rms_norm(2 * Z), atol=1e-6)
    assert not np.allclose(rms_norm(Z), rms_norm(Z + 100), atol=1e-2)

    # the one-pass variance goes negative once the offset is large enough
    rows = cancellation_table()
    assert rows[0]['rel_err'] < 1e-9                 # identical with no offset
    assert any(r['one_pass'] < 0 for r in rows)      # and negative once it is large
    assert all(abs(r['two_pass'] - rows[0]['two_pass']) < 1e-2 for r in rows)

    # two heads reach the two-pattern target exactly, one head cannot - and the single
    # head is given a channel matrix as well, which is the only extra a real head has
    V, prev, first, target = two_pattern_target()
    assert two_head_error(V, prev, first, target) < 1e-20
    assert two_head_error_many() < 1e-20
    _, _, single = best_single_head(steps=400)
    assert single > 1e-2 and single < target_scale()
    assert one_input_exact() < 1e-20                   # one input proves nothing

    # a block preserves shape, and the FLOP crossover is where the two halves are equal
    pb = make_params(32, 4, seed=2)
    xb = rng.standard_normal((9, 32))
    assert block_pre_ln(xb, pb).shape == xb.shape
    assert block_post_ln(xb, pb).shape == xb.shape
    nc = flop_crossover(4096)
    f = block_flops(int(round(nc)), 4096)
    assert abs(f['linear'] - f['quad']) / f['linear'] < 1e-6

    # LeetCode 189 against the obvious slice, including k > n
    for seed in range(300):
        r = np.random.default_rng(seed)
        nums = r.integers(-50, 50, int(r.integers(1, 12))).tolist()
        k = int(r.integers(0, 25))
        want = nums[-(k % len(nums)):] + nums[:-(k % len(nums))] if k % len(nums) else list(nums)
        got = list(nums)
        SolutionRotateArray().rotate(got, k)
        assert got == want
        assert rotate_cycles(list(nums), k) == want
    a = [1, 2, 3, 4, 5, 6]
    assert rotate_one_cycle(list(a), 2) != [5, 6, 1, 2, 3, 4]    # gcd(6, 2) = 2 cycles
    assert rotate_one_cycle(list(a), 1) == [6, 1, 2, 3, 4, 5]    # gcd(6, 1) = 1 cycle

    # LeetCode 303 against the obvious sum
    for seed in range(300):
        r = np.random.default_rng(seed + 7)
        nums = r.integers(-100, 100, int(r.integers(1, 20))).tolist()
        na = NumArray(nums)
        for _ in range(5):
            i = int(r.integers(0, len(nums)))
            j = int(r.integers(i, len(nums)))
            assert na.sumRange(i, j) == sum(nums[i:j + 1])
    return True


# ================================================================ 11. the whole story
def header(title):
    print('\n' + '=' * 72)
    print(title)
    print('=' * 72)


def main():
    np.set_printoptions(suppress=True, precision=4, legacy='1.25')

    # ------------------------------------------------------------------ 1
    header('1. A block is two mixers: one across positions, one across features')
    print('      x = x + attention(norm(x))      mixes positions, every channel the same way')
    print('      x = x + mlp(norm(x))            mixes channels, every position the same way\n')
    d_model, h, n = 64, 8, 16
    p = make_params(d_model, h, seed=0)
    rng = np.random.default_rng(1)
    X = rng.standard_normal((n, d_model))
    y = block_pre_ln(X, p)
    print(f'  d_model {d_model}, {h} heads, {n} tokens.')
    print(f'  input {X.shape} -> block -> {y.shape}: a block never changes the shape of')
    print('  the residual stream, which is why you can stack 32 of them.')
    a_only, _ = multi_head_attention(layer_norm(X, p['g1'], p['b1']), p['Wq'], p['Wk'],
                                     p['Wv'], p['Wo'], h)
    print(f'  the two branches add into that stream rather than replacing it: the')
    print(f'  attention branch is {np.linalg.norm(a_only) / np.linalg.norm(X):.3f} of the '
          f'stream it was read from, and the whole')
    print(f'  block moves it by {np.linalg.norm(y - X) / np.linalg.norm(X):.3f} '
          f'in relative norm.')

    # ------------------------------------------------------------------ 2
    header('2. h heads cost nothing extra: the same four matrices, cut differently')
    print(f'  {"heads":>6} {"d_head":>8} {"params in Wq,Wk,Wv,Wo":>24} {"weight matrices":>17}')
    for r in head_count_table(d_model):
        print(f'  {r["h"]:>6} {r["per_head_dim"]:>8} {r["total"]:>24,} '
              f'{r["score_entries_per_layer"]:>17}')
    print('\n  Nothing in that table changes except the last two columns: the head count')
    print('  splits the SAME d_model x d_model matrices into h independent subspaces.')
    print('  Each head then produces its own n x n weight matrix, so the layer mixes the')
    print('  sequence h different ways instead of one, at identical parameter cost.')

    # ------------------------------------------------------------------ 3
    header('3. What the extra heads actually buy')
    print('  A target that needs two different mixings at once: half the channels want')
    print('  the previous token, the other half want the first token.  Measured over 12')
    print('  different inputs, because one input alone is too easy to fit.\n')
    two = two_head_error_many()
    _, _, single = best_single_head()
    zero = target_scale()
    print(f'  two heads, one pattern each:            MSE {two:.2e}')
    print(f'  one head, with a channel matrix too:    MSE {single:.4f}')
    print(f'  (predicting zero everywhere would be    MSE {zero:.4f})')
    print('\n  The single-head run gets everything a real head has and more: its own')
    print('  d x d channel matrix M - which is what Wv and Wo amount to - plus a free')
    print('  choice of attention matrix per input, which a real head does not get,')
    print('  because its weights have to come out of Q and K.  Without the channel')
    print(f'  matrix this comparison is rigged: with one input and n = d = 8, solving')
    print(f'  (A V) M = target for M alone fits it to MSE {one_input_exact():.1e} for a')
    print('  RANDOM attention matrix A, which says nothing about heads at all.')
    print('  What one head still cannot do is use a DIFFERENT attention matrix for')
    print('  different channels, and that is exactly what the target asks for.  Heads')
    print('  are how a layer does several things at once; width alone cannot buy that.')

    # ------------------------------------------------------------------ 4
    header('4. Silent failure #1: splitting the heads along the wrong axis')
    Xs, good, bad = head_split_demo()
    print('  A 4 x 8 input, 2 heads.  Head 0 should own features 0..3 of every token.\n')
    print(f'  correct  split_heads(X, 2)[0] =\n{good[0]}')
    print(f'\n  wrong    X.reshape(2, 4, 4)[0] =\n{bad[0]}')
    mx, rel, same_shape = split_output_gap()
    print(f'\n  The wrong one hands head 0 the first two POSITIONS instead of the first')
    print('  four FEATURES, and then mixes the two halves of the sequence separately.')
    print(f'  Same shape: {same_shape}.  Difference in the layer output: max {mx:.3f},')
    print(f'  {rel * 100:.0f}% of the output norm.  No exception, no shape error, and')
    print('  a model built on it still trains - just worse, for no visible reason.')

    # ------------------------------------------------------------------ 5
    header('5. Silent failure #2: the one-pass variance inside LayerNorm')
    print('  var = E[x^2] - E[x]^2 needs one pass instead of two, so it keeps getting')
    print('  written.  A residual stream accumulates a mean offset, and that identity')
    print('  subtracts two large numbers to get a small one.  In float32, 512 values:\n')
    print(f'  {"offset":>12} {"two-pass var":>14} {"one-pass var":>16} '
          f'{"relative error":>16} {"distinct values":>17}')
    for r in cancellation_table():
        print(f'  {r["offset"]:>12,.0f} {r["two_pass"]:>14.6f} {r["one_pass"]:>16.6f} '
              f'{r["rel_err"]:>16.3e} {r["distinct"]:>17}')
    rows = cancellation_table()
    bad = next(r for r in rows if r['one_pass'] < 0)
    print(f'\n  At an offset of {bad["offset"]:,.0f} the one-pass variance is '
          f'{bad["one_pass"]:.1f} - negative, so the')
    print('  square root is nan - while the two-pass one still reports '
          f'{bad["two_pass"]:.6f}')
    print(f'  from the same array, which still holds {bad["distinct"]} distinct float32')
    print('  values.  The data is fine; the identity is not.  And below that offset the')
    print('  damage is quiet: the eps inside the square root keeps it finite and the')
    print('  layer simply returns the wrong scale.  Two passes over 4,096 numbers cost')
    print('  nothing next to the matmuls around them.')

    # ------------------------------------------------------------------ 6
    header('6. LayerNorm vs RMSNorm: which invariance you are paying for')
    Z = np.random.default_rng(4).standard_normal((3, 64)) * 2 + 7
    print(f'  a row with mean {Z[0].mean():.3f} and std {Z[0].std():.3f}:')
    print(f'    layer_norm -> mean {layer_norm(Z)[0].mean():+.2e}, std {layer_norm(Z)[0].std():.4f}')
    print(f'    rms_norm   -> mean {rms_norm(Z)[0].mean():+.4f}, std {rms_norm(Z)[0].std():.4f}')
    print(f'  shift the row by +100:')
    print(f'    layer_norm changes by {np.abs(layer_norm(Z) - layer_norm(Z + 100)).max():.2e}')
    print(f'    rms_norm   changes by {np.abs(rms_norm(Z) - rms_norm(Z + 100)).max():.4f}')
    print('\n  RMSNorm drops the mean subtraction and the beta parameter: fewer reductions,')
    print('  fewer parameters, and it gives up shift invariance.  Both are invariant to')
    print('  scale, which is the one the residual stream actually threatens.')

    # ------------------------------------------------------------------ 7
    header('7. Pre-LN vs post-LN: which path through the stack stays unnormalised')
    tr = depth_signal(layers=32)
    print('  Push a tiny change into the input of 32 stacked blocks and measure how')
    print('  much of it is still there at the top, and how big the stream itself got.\n')
    print(f'  {"after block":>12} {"||x|| pre-LN":>14} {"||x|| post-LN":>15} '
          f'{"influence pre":>14} {"post":>8}')
    for i in (0, 3, 7, 15, 31):
        a, b = tr['pre'][i], tr['post'][i]
        print(f'  {i + 1:>12} {a["norm"]:>14.1f} {b["norm"]:>15.1f} '
              f'{a["signal"]:>14.3f} {b["signal"]:>8.3f}')
    pre, post = tr['pre'][-1], tr['post'][-1]
    print(f'\n  Post-LN pins the stream at {post["norm"]:.1f} after every single block, '
          f'because the last')
    print('  thing a post-LN block does is normalise (with gamma = 1 and beta = 0 as')
    print('  initialised here; learned gamma and beta move that number, but the norm')
    print('  still stops depending on the depth).  Pre-LN leaves the residual')
    print(f'  path alone, so it grows from {tr["pre"][0]["norm"]:.1f} to '
          f'{pre["norm"]:.1f} over 32 blocks.')
    print(f'  The perturbation grows with it - {pre["signal"]:.2f}x against '
          f'{post["signal"]:.2f}x in absolute')
    print(f'  terms - but as a fraction of the stream the two are nearly the same')
    print(f'  ({pre["relative"]:.4f} vs {post["relative"]:.4f}), so this measurement does')
    print('  NOT show pre-LN carrying more signal.  What it shows is the structural')
    print('  difference: pre-LN leaves one unnormalised path from the input all the way')
    print('  to the output, post-LN puts a normalisation on that path in every block.')
    print('  That is the thing the training-stability results in the literature are')
    print('  about (Xiong et al. 2020), and it is why a pre-LN stack needs one final')
    print('  LayerNorm before the output head: nothing else bounds what arrives there.')

    # ------------------------------------------------------------------ 8
    header('8. The residual stream grows, and the init has to pay for it')
    g = residual_growth(layers=32)
    gs = residual_growth(layers=32, branch_scale=1 / math.sqrt(2 * 32))
    plain, scaled = g['norms'], gs['norms']
    print(f'  {"after block":>12} {"||x|| unscaled":>16} {"||x|| with 1/sqrt(2L)":>23}')
    print(f'  {"0 (input)":>12} {g["start"]:>16.1f} {gs["start"]:>23.1f}')
    for i in (0, 7, 15, 31):
        print(f'  {i + 1:>12} {plain[i]:>16.1f} {scaled[i]:>23.1f}')
    print(f'\n  Every branch adds into the same stream, so without the scaling the norm')
    print(f'  grows {plain[-1] / g["start"]:.1f}x over 32 blocks - which is the sqrt(32) = '
          f'{math.sqrt(32):.2f} you')
    print('  would expect from summing 32 independent pieces into one running sum.')
    print(f'  Scaling what each branch writes by 1/sqrt(2L) holds it at '
          f'{scaled[-1] / gs["start"]:.2f}x.  That is')
    print('  the whole content of "scale the residual branches at initialisation".')

    # ------------------------------------------------------------------ 9
    header('9. Where the parameters and the FLOPs actually are')
    print(f'  {"d_model":>8} {"heads":>6} {"attention":>14} {"MLP":>14} {"MLP share":>11} '
          f'{"FLOP crossover":>15}')
    for r in budget_table():
        print(f'  {r["d_model"]:>8} {r["h"]:>6} {r["attn"]:>14,} {r["mlp"]:>14,} '
              f'{r["mlp_share"] * 100:>10.0f}% {r["crossover"]:>15,.0f}')
    print('\n  Two thirds of every block is the MLP, not the attention - the part people')
    print('  talk about least.  And with d_ff = 4 d_model the crossover is always 6')
    print('  d_model: below that context length the block is dominated by the linear')
    print('  part, above it by the quadratic one.\n')
    print(f'  d_model 4096, one block:')
    print(f'  {"context":>10} {"linear MACs":>18} {"quadratic MACs":>18} {"quadratic share":>17}')
    for r in flop_table():
        print(f'  {r["n"]:>10,} {r["linear"]:>18,} {r["quad"]:>18,} '
              f'{r["quad_share"] * 100:>16.1f}%')

    # ------------------------------------------------------------------ 10
    header('10. LeetCode 189 - Rotate Array')
    a = [1, 2, 3, 4, 5, 6, 7]
    b = list(a)
    SolutionRotateArray().rotate(b, 3)
    print(f'  {a}, k = 3 -> {b}')
    print('  Reverse the whole thing, then reverse the two pieces: the last k elements')
    print('  come to the front in order, and nothing is copied.\n')
    c = [1, 2, 3, 4, 5, 6]
    print(f'  the cycle method on {c}:')
    print(f'    all gcd(n, k) starting points, k = 2 -> {rotate_cycles(list(c), 2)}')
    print(f'    starting only at index 0,      k = 2 -> {rotate_one_cycle(list(c), 2)}')
    print(f'    starting only at index 0,      k = 1 -> {rotate_one_cycle(list(c), 1)}')
    print('  With k = 1 there is one cycle and the shortcut is right; with k = 2 there')
    print('  are gcd(6, 2) = 2 of them and it moves half the array.  Same shape, same')
    print('  type, half the answer - which is the shape of the head-splitting bug too.')

    # ------------------------------------------------------------------ 11
    header('11. LeetCode 303 - Range Sum Query (immutable)')
    nums = [-2, 0, 3, -5, 2, -1]
    na = NumArray(nums)
    print(f'  {nums}')
    print(f'  sumRange(0, 2) = {na.sumRange(0, 2)}, sumRange(2, 5) = {na.sumRange(2, 5)}, '
          f'sumRange(0, 5) = {na.sumRange(0, 5)}')
    print('  One pass builds the prefix sums, every query is then one subtraction.\n')
    x = np.full(512, 1e4, dtype=np.float32) + np.random.default_rng(0).standard_normal(512).astype(np.float32)
    good = float(np.var(x[100:300].astype(np.float64)))
    viapre = window_variance_prefix(x, 100, 300)
    print('  The same trick extended to variance is where it bites: prefix sums of x and')
    print('  of x^2, one subtraction each, exactly the identity from section 5.')
    print(f'    window variance, computed directly:    {good:.6f}')
    print(f'    from float32 prefix sums:              {viapre:.6f}')
    print('  Prefix sums answer any range in O(1) and that is worth having; for sums of')
    print('  squares the running total gets large while the answer stays small, which is')
    print('  the same cancellation LayerNorm has to avoid.')

    # ------------------------------------------------------------------ 12
    header('12. Assertions')
    check()
    print('all assertions passed')


if __name__ == '__main__':
    main()
