"""Day 52 - scaled dot-product attention from scratch, and the four things
everyone changes about it.

Attention is a weighted average.  The weights come from a softmax over similarity
scores, and almost everything that is hard about it - overflow, the sqrt(d), the mask,
the memory bill, FlashAttention, paged KV, MQA/GQA/MLA, linear attention - is about
how you compute that average without building the thing it is an average over.

Run `python attention.py` to print the whole story; every number below is produced by
this file and checked by an assert at the bottom.
"""

import math

import numpy as np


# ================================================================ 1. softmax
def softmax_naive(x, axis=-1):
    """The formula as written.  exp() of a large number is inf, and inf / inf is nan."""
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


def softmax(x, axis=-1):
    """Subtract the row max first.

    softmax(x) == softmax(x - c) for any constant c, because the c cancels between
    numerator and denominator.  Choosing c = max(x) makes the largest exponent exactly
    exp(0) = 1, so nothing can overflow and the largest term is never lost.
    """
    x = np.asarray(x, dtype=float)
    m = np.max(x, axis=axis, keepdims=True)
    e = np.exp(x - m)
    return e / e.sum(axis=axis, keepdims=True)


def overflow_threshold(dtype=np.float32):
    """The smallest integer whose exp() is inf in this dtype."""
    info = np.finfo(dtype)
    k = 1
    with np.errstate(over='ignore'):
        while np.isfinite(np.exp(np.array(k, dtype=dtype))):
            k += 1
    return k, float(info.max)


# ================================================================ 2. attention
def attention(Q, K, V, mask=None, scale=None):
    """softmax(QK^T / sqrt(d)) V - four lines, and the rest of the file is about
    what each line costs.

    Q is (n, d), K and V are (m, d).  The score matrix is (n, m): one number for
    every pair of positions, which is where the quadratic memory comes from.
    """
    d = Q.shape[-1]
    scores = Q @ K.T / (scale if scale is not None else math.sqrt(d))
    if mask is not None:
        scores = np.where(mask, scores, -np.inf)
    w = softmax(scores, axis=-1)
    return w @ V, w


def causal_mask(n, m=None):
    """True where query row i is allowed to look at key j.

    For the square case this is j <= i.  When there are more keys than queries (the
    decode case, where the queries are the newest m - n positions) the diagonal is
    shifted by m - n, so row i may look at j <= i + m - n.
    """
    m = n if m is None else m
    return np.tril(np.ones((n, m), dtype=bool), k=m - n)


def attention_mask_after_softmax(Q, K, V, mask):
    """The bug: softmax first, then zero the future.

    The rows still sum to 1 before the zeroing, so afterwards they do not, and every
    position is scaled down by however much future it was not allowed to see.  No
    exception, no nan - just a quietly wrong weighted average.
    """
    d = Q.shape[-1]
    w = softmax(Q @ K.T / math.sqrt(d), axis=-1)
    w = np.where(mask, w, 0.0)
    return w @ V, w


# ================================================================ 3. online softmax
def attention_online(Q, K, V, block=32, mask=None):
    """FlashAttention's identity, written plainly: one pass over K and V in blocks,
    carrying a running max and a running sum, and rescaling what you already have
    whenever the max moves.

    The n x m score matrix is never materialised - only one block of it at a time -
    and the result is the same number, not an approximation.
    """
    n, d = Q.shape
    m = K.shape[0]
    out = np.zeros((n, V.shape[1]))
    run_max = np.full(n, -np.inf)
    run_sum = np.zeros(n)
    for s in range(0, m, block):
        e = min(s + block, m)
        s_blk = Q @ K[s:e].T / math.sqrt(d)
        if mask is not None:
            s_blk = np.where(mask[:, s:e], s_blk, -np.inf)
        blk_max = np.max(s_blk, axis=-1)
        new_max = np.maximum(run_max, blk_max)
        new_max = np.where(np.isfinite(new_max), new_max, 0.0)
        # rescale what we already accumulated to the new maximum
        alpha = np.exp(run_max - new_max)
        p = np.exp(s_blk - new_max[:, None])
        p = np.where(np.isfinite(p), p, 0.0)
        run_sum = run_sum * alpha + p.sum(axis=-1)
        out = out * alpha[:, None] + p @ V[s:e]
        run_max = new_max
    return out / np.where(run_sum == 0, 1.0, run_sum)[:, None]


# ================================================================ 4. the memory bill
def score_matrix_bytes(n, heads, bytes_per_elem=2):
    """How big the n x n score matrix is, per layer, if you build it."""
    return n * n * heads * bytes_per_elem


def kv_cache_bytes_per_token(layers=32, heads=32, head_dim=128, kv_heads=None,
                             latent=None, rope_dim=0, bytes_per_elem=2):
    """Bytes of KV cache one token costs, for the four common layouts.

    MHA   : kv_heads == heads          - one K and one V per head
    MQA   : kv_heads == 1              - all heads share one K and one V
    GQA   : 1 < kv_heads < heads       - groups of heads share
    MLA   : latent is not None         - cache one low-rank vector per token instead
                                         of K and V, plus the part that carries RoPE
    """
    if latent is not None:
        return layers * (latent + rope_dim) * bytes_per_elem
    kv_heads = heads if kv_heads is None else kv_heads
    return layers * 2 * kv_heads * head_dim * bytes_per_elem


def linear_state_bytes(layers=32, heads=32, head_dim=128, bytes_per_elem=2):
    """A delta-rule / linear-attention layer carries a head_dim x head_dim matrix per
    head instead of a growing cache.  It does not depend on the sequence length."""
    return layers * heads * head_dim * head_dim * bytes_per_elem


# ================================================================ 5. linear attention
def delta_rule(K, V, beta, state=None):
    """The delta rule: a fixed-size memory that is written to, not appended to.

    Softmax attention keeps every key and value and re-reads them all at every step.
    A delta-rule layer keeps one d x d matrix S and updates it in place:

        S <- S (I - beta_t k_t k_t^T) + beta_t v_t k_t^T

    The first term erases whatever S currently associates with k_t, the second writes
    v_t there instead - which is why it is called the delta rule and not just a sum.
    A real layer reads with its own query projection, o_t = S_t q_t; this function
    probes with the key itself, because what it is here to measure is what was stored.

    Gated variants add a forget factor so old associations decay instead of persisting
    until overwritten: Gated DeltaNet uses one scalar per head, and Kimi's KDA makes it
    per channel (a diagonal gate).  The shape of the state, and the fact that it does
    not grow with the sequence, are the same.
    """
    n, d = K.shape
    dv = V.shape[1]
    S = np.zeros((dv, d)) if state is None else state
    out = np.zeros((n, dv))
    for t in range(n):
        k = K[t] / (np.linalg.norm(K[t]) + 1e-12)
        b = beta[t]
        out[t] = S @ k
        S = S - b * np.outer(S @ k, k) + b * np.outer(V[t], k)
    return out, S


def delta_rule_recall(d=16, n=8, seed=0, orthogonal=True):
    """Write n key-value pairs into one d x d state, then read all n of them back.

    With orthogonal keys the write of one pair does not disturb the others, so the
    state holds exactly d associations perfectly.  Ask for the d+1-th and something
    has to be overwritten.  With ordinary random keys nothing is orthogonal and every
    write smears the ones already there, so the error starts above zero immediately.
    """
    rng = np.random.default_rng(seed)
    if orthogonal:
        basis = np.linalg.qr(rng.standard_normal((d, d)))[0]
        K = np.array([basis[t % d] for t in range(n)])
    else:
        K = rng.standard_normal((n, d))
    V = rng.standard_normal((n, d))
    _, S = delta_rule(K, V, np.ones(n))
    errs = []
    for t in range(n):
        k = K[t] / np.linalg.norm(K[t])
        errs.append(float(np.linalg.norm(S @ k - V[t]) / np.linalg.norm(V[t])))
    return np.array(errs)


# ================================================================ 6. paged KV
def contiguous_waste(lengths, capacity):
    """Reserve `capacity` slots per sequence up front, the way a contiguous cache must.

    Every sequence is allocated for the longest it might get, so the waste is the gap
    between what was reserved and what was used.
    """
    reserved = len(lengths) * capacity
    used = sum(lengths)
    return reserved, used, (reserved - used) / reserved


def paged_waste(lengths, page_size):
    """Hand out fixed-size pages on demand instead.

    The only waste left is the unused tail of each sequence's last page, which is at
    most page_size - 1 slots per sequence no matter how long the sequence is.
    """
    pages = sum(math.ceil(n / page_size) for n in lengths)
    reserved = pages * page_size
    used = sum(lengths)
    return pages, reserved, used, (reserved - used) / reserved


# ================================================================ 7. LeetCode 239
class SolutionSlidingWindowMaximum:
    """LeetCode 239 - Sliding Window Maximum.

    A deque holding indices whose values are strictly decreasing.  The front is the
    maximum of the current window; anything smaller than the value arriving is dead,
    because the newcomer is both larger and newer, so it outlives them all.

    Online softmax carries the same kind of running maximum - one number instead of
    the whole row - and rescales everything already accumulated whenever it changes.
    """

    def maxSlidingWindow(self, nums: list[int], k: int) -> list[int]:
        from collections import deque
        dq, out = deque(), []
        for i, v in enumerate(nums):
            while dq and nums[dq[-1]] <= v:
                dq.pop()                      # smaller AND older: never the max again
            dq.append(i)
            if dq[0] <= i - k:
                dq.popleft()                  # the front fell out of the window
            if i >= k - 1:
                out.append(nums[dq[0]])
        return out


def max_sliding_window_brute(nums, k):
    return [max(nums[i:i + k]) for i in range(len(nums) - k + 1)]


# ================================================================ 8. LeetCode 703
class KthLargest:
    """LeetCode 703 - Kth Largest Element in a Stream.

    Keep a min-heap of exactly k elements.  Its root is the k-th largest seen so far,
    and everything smaller is discarded the moment it arrives, so the memory is k and
    not the length of the stream.

    That is the shape FlashAttention and linear attention both have: a stream you only
    get one pass over, and a bounded amount of state that has to be enough.
    """

    def __init__(self, k: int, nums: list[int]):
        import heapq
        self.k = k
        self.heap = list(nums)
        heapq.heapify(self.heap)
        while len(self.heap) > k:
            heapq.heappop(self.heap)

    def add(self, val: int) -> int:
        import heapq
        if len(self.heap) < self.k:
            heapq.heappush(self.heap, val)
        elif val > self.heap[0]:
            heapq.heapreplace(self.heap, val)
        return self.heap[0]


def kth_largest_brute(k, nums, adds):
    seen, out = list(nums), []
    for v in adds:
        seen.append(v)
        out.append(sorted(seen, reverse=True)[k - 1])
    return out


# ================================================================ 9. seeded tables
# Every table in main() comes from one of these, so the figures, the notebook and the
# demo page can reproduce the exact same numbers by importing this file.
def toy_attention(n=4, m=6, d=8, dv=3, seed=10):
    """One small attention call: the weight matrix and the values it averages."""
    rng = np.random.default_rng(seed)
    Q, K, V = rng.standard_normal((n, d)), rng.standard_normal((m, d)), rng.standard_normal((m, dv))
    out, w = attention(Q, K, V)
    return Q, K, V, out, w


def scale_table(ds=(8, 64, 512, 4096), n=256, seed=11):
    """How sharp the softmax gets as the head widens, with and without the 1/sqrt(d)."""
    rows = []
    for d in ds:
        # seeded per d, so a table of four rows and a curve of eleven agree on
        # the rows they share
        rng = np.random.default_rng([seed, d])
        q, k = rng.standard_normal((n, d)), rng.standard_normal((n, d))
        raw = q @ k.T
        rows.append({'d': d,
                     'std': float(np.std(raw)),
                     'sqrt': math.sqrt(d),
                     'raw': float(softmax(raw, axis=-1).max(axis=-1).mean()),
                     'scaled': float(softmax(raw / math.sqrt(d), axis=-1).max(axis=-1).mean())})
    return rows


def causal_demo(n=6, d=8, dv=3, seed=13):
    """The same causal attention masked before and after the softmax."""
    rng = np.random.default_rng(seed)
    Q, K, V = rng.standard_normal((n, d)), rng.standard_normal((n, d)), rng.standard_normal((n, dv))
    mask = causal_mask(n)
    _, w_free = attention(Q, K, V)
    _, w_ok = attention(Q, K, V, mask=mask)
    _, w_bad = attention_mask_after_softmax(Q, K, V, mask)
    return mask, w_free, w_ok, w_bad


def flash_table(blocks=(1, 8, 32, 128, 256), nq=64, nk=256, d=64, dv=32, seed=12):
    """The online (FlashAttention) result against the full matrix, per block size."""
    rng = np.random.default_rng(seed)
    Q, K, V = (rng.standard_normal((nq, d)), rng.standard_normal((nk, d)),
               rng.standard_normal((nk, dv)))
    ref, _ = attention(Q, K, V)
    return [{'blk': b, 'err': float(np.abs(attention_online(Q, K, V, block=b) - ref).max())}
            for b in blocks]


# ================================================================ 10. assertions
def check():
    rng = np.random.default_rng(0)

    # softmax: rows sum to 1, and the max trick changes nothing but the overflow
    for _ in range(50):
        x = rng.standard_normal((4, 7)) * 3
        assert np.allclose(softmax(x).sum(axis=-1), 1.0)
        assert np.allclose(softmax(x), softmax_naive(x), atol=1e-12)
        # shifting the input by any constant leaves softmax unchanged
        assert np.allclose(softmax(x), softmax(x + 1000.0), atol=1e-12)
    with np.errstate(over='ignore', invalid='ignore'):
        big = np.array([[1000.0, 1001.0, 999.0]])
        assert np.isnan(softmax_naive(big)).any()      # the naive one dies
    assert np.isfinite(softmax(big)).all()             # the shifted one does not

    # attention: weights are a distribution, output is inside the convex hull of V
    for _ in range(30):
        n, m, d, dv = 5, 9, 8, 4
        Q, K, V = (rng.standard_normal((n, d)), rng.standard_normal((m, d)),
                   rng.standard_normal((m, dv)))
        out, w = attention(Q, K, V)
        assert np.allclose(w.sum(axis=-1), 1.0) and (w >= 0).all()
        assert (out.max(axis=0) <= V.max(axis=0) + 1e-9).all()
        assert (out.min(axis=0) >= V.min(axis=0) - 1e-9).all()
        # one pass in blocks gives the same number, not an approximation
        for blk in (1, 2, 3, 4, 16):
            assert np.allclose(out, attention_online(Q, K, V, block=blk), atol=1e-12)
        msk = causal_mask(n, m)
        out_c, w_c = attention(Q, K, V, mask=msk)
        assert np.allclose(w_c.sum(axis=-1), 1.0)
        assert (w_c[~msk] == 0).all()                  # nothing leaks from the future
        for blk in (1, 3, 16):
            assert np.allclose(out_c, attention_online(Q, K, V, block=blk, mask=msk),
                               atol=1e-12)
    # masking after the softmax leaves rows that do not sum to 1
    msk = causal_mask(5, 5)
    Q, K, V = rng.standard_normal((5, 8)), rng.standard_normal((5, 8)), rng.standard_normal((5, 3))
    _, w_bad = attention_mask_after_softmax(Q, K, V, msk)
    assert not np.allclose(w_bad.sum(axis=-1), 1.0)
    assert abs(w_bad.sum(axis=-1)[0]) < 1.0            # the first row loses the most

    # scaling: without 1/sqrt(d) the logits grow with d and the softmax saturates
    sharp = []
    for d in (8, 64, 512):
        q, k = rng.standard_normal((200, d)), rng.standard_normal((200, d))
        sharp.append(float(np.std(q @ k.T)))
    assert sharp[0] < sharp[1] < sharp[2]              # std grows like sqrt(d)

    # KV cache arithmetic
    mha = kv_cache_bytes_per_token()
    mqa = kv_cache_bytes_per_token(kv_heads=1)
    gqa = kv_cache_bytes_per_token(kv_heads=8)
    mla = kv_cache_bytes_per_token(latent=512, rope_dim=64)
    assert mha == 32 * 2 * 32 * 128 * 2 == 524288
    assert mqa * 32 == mha and gqa * 4 == mha
    assert mla == 32 * (512 + 64) * 2 == 36864
    assert mqa < mla < gqa < mha
    # a linear-attention state does not depend on how long the sequence is
    assert linear_state_bytes() == 32 * 32 * 128 * 128 * 2

    # paging: never more than page_size - 1 wasted slots per sequence
    lens = [1, 15, 16, 17, 300, 4096]
    for page in (8, 16, 64):
        pages, reserved, used, _ = paged_waste(lens, page)
        assert reserved >= used and reserved - used <= len(lens) * (page - 1)

    # the delta rule writes in place: d orthogonal associations fit exactly
    assert delta_rule_recall(d=16, n=16).max() < 1e-9
    over = delta_rule_recall(d=16, n=17)               # one more than the state holds
    assert over.max() > 0.5 and over[-1] < 1e-9        # the newest is perfect, one is gone
    smear = delta_rule_recall(d=16, n=8, orthogonal=False)
    assert smear.max() > 1e-3                          # non-orthogonal keys interfere

    # LeetCode 239 against brute force
    for seed in range(300):
        r = np.random.default_rng(seed)
        nums = r.integers(-50, 50, int(r.integers(1, 20))).tolist()
        k = int(r.integers(1, len(nums) + 1))
        assert SolutionSlidingWindowMaximum().maxSlidingWindow(nums, k) == \
            max_sliding_window_brute(nums, k)
    assert SolutionSlidingWindowMaximum().maxSlidingWindow([1, 3, -1, -3, 5, 3, 6, 7], 3) \
        == [3, 3, 5, 5, 6, 7]
    assert SolutionSlidingWindowMaximum().maxSlidingWindow([1], 1) == [1]

    # LeetCode 703 against brute force
    for seed in range(300):
        r = np.random.default_rng(seed + 99)
        k = int(r.integers(1, 6))
        init = r.integers(-20, 20, int(r.integers(0, 6))).tolist()
        adds = r.integers(-20, 20, int(r.integers(1, 8))).tolist()
        if len(init) + 1 < k:
            continue                                   # 703 guarantees k-th exists
        kl = KthLargest(k, list(init))
        got = [kl.add(v) for v in adds]
        assert got == kth_largest_brute(k, init, adds)
    kl = KthLargest(3, [4, 5, 8, 2])
    assert [kl.add(v) for v in [3, 5, 10, 9, 4]] == [4, 5, 5, 8, 8]
    return True


# ================================================================ 11. the whole story
def header(title):
    print('\n' + '=' * 72)
    print(title)
    print('=' * 72)


def main():
    np.set_printoptions(suppress=True, precision=4, legacy='1.25')

    # ------------------------------------------------------------------ 1
    header('1. Attention is a weighted average, and the weights come from a softmax')
    print('      scores = Q @ K.T / sqrt(d)      one number per pair of positions')
    print('      scores = where(mask, scores, -inf)')
    print('      w      = softmax(scores)        each row sums to 1')
    print('      out    = w @ V                  a weighted average of the values\n')
    n, m, d, dv = 4, 6, 8, 3
    Q, K, V, out, w = toy_attention(n, m, d, dv)
    print(f'  {n} queries, {m} keys, head dimension {d}.')
    print(f'  the weight matrix is {w.shape} and its rows sum to {np.round(w.sum(axis=-1), 6)}')
    print(f'  row 0 of the weights: {np.round(w[0], 4)}')
    print(f'\n  Every output row is a convex combination of the {m} value rows, so it can')
    print('  never leave the range of V.  Attention cannot invent a value; it can only')
    print('  decide how much of each one to use.')
    print(f'    V ranges from {np.round(V.min(axis=0), 3)} to {np.round(V.max(axis=0), 3)}')
    print(f'    out ranges from {np.round(out.min(axis=0), 3)} to {np.round(out.max(axis=0), 3)}')

    # ------------------------------------------------------------------ 2
    header('2. Silent failure #1: exp() of a large number')
    k32, max32 = overflow_threshold(np.float32)
    k16, max16 = overflow_threshold(np.float16)
    print(f'  float32 holds up to {max32:.3e}, and exp({k32}) is already inf.')
    print(f'  float16 holds up to {max16:.0f},        and exp({k16}) is already inf.\n')
    big = np.array([[1000.0, 1001.0, 999.0]])
    with np.errstate(over='ignore', invalid='ignore'):
        naive = softmax_naive(big)
    print(f'  softmax_naive([1000, 1001, 999]) = {naive}   (inf / inf)')
    print(f'  softmax      ([1000, 1001, 999]) = {np.round(softmax(big), 6)}')
    print('\n  The fix is one line and it is exact, not an approximation: softmax(x) is')
    print('  equal to softmax(x - c) for ANY constant c, because c cancels between the')
    print('  numerator and the denominator.  Taking c = max(x) makes the largest')
    print('  exponent exp(0) = 1, so nothing can overflow.')
    print('  Attention logits reach that range easily: they are dot products of two')
    print('  d-dimensional vectors, and in fp16 you only need a logit of 12.')

    # ------------------------------------------------------------------ 3
    header('3. Why the 1 / sqrt(d), measured')
    print('  A score is a dot product of d terms.  With unit-variance inputs its')
    print('  standard deviation grows like sqrt(d), so the logits spread out as the')
    print('  head gets wider and the softmax turns into an argmax.\n')
    print(f'  {"d":>6} {"std of raw scores":>18} {"sqrt(d)":>9} {"max weight, unscaled":>21} {"scaled":>9}')
    for r in scale_table():
        print(f"  {r['d']:>6} {r['std']:>18.3f} {r['sqrt']:>9.2f} {r['raw']:>21.4f} {r['scaled']:>9.4f}")
    print('\n  Unscaled, a 4096-wide head puts almost all of its weight on one key: the')
    print('  average row already has a maximum weight near 1, which is an argmax with')
    print('  no gradient to speak of.  Dividing by sqrt(d) keeps the logits at roughly')
    print('  unit scale whatever d is, and the weights stay spread out.')

    # ------------------------------------------------------------------ 4
    header('4. Silent failure #2: masking after the softmax instead of before')
    nn = 6
    msk, w_free, w_ok, w_bad = causal_demo(nn)
    print('  Causal attention has to stop position i from reading position j > i.')
    print('  Put -inf in the scores and the softmax gives those keys exactly zero')
    print('  weight, and the row still sums to 1.  Zero them AFTER the softmax and the')
    print('  row no longer sums to anything in particular.\n')
    print(f'  row sums, masked before softmax: {np.round(w_ok.sum(axis=-1), 6)}')
    print(f'  row sums, masked after  softmax: {np.round(w_bad.sum(axis=-1), 6)}')
    print(f'\n  Row 0 can only see itself, so it should be a weight of 1.0 on one key.')
    print(f'  Masked afterwards it carries {w_bad.sum(axis=-1)[0]:.4f} instead - every early')
    print('  position is scaled down by however much future it was not allowed to see,')
    print('  and the earlier the position the worse it is.  Nothing raises, nothing is')
    print('  nan, and the loss still goes down.')

    # ------------------------------------------------------------------ 5
    header('5. FlashAttention: the same number, in one pass, without the matrix')
    nq, nk = 64, 256
    print('  Walk the keys in blocks, carrying two numbers per query row: the largest')
    print('  score seen so far, and the running sum of exp(score - that max).  When a')
    print('  block contains a bigger score, rescale everything already accumulated by')
    print('  exp(old_max - new_max) and carry on.  That rescaling is the whole trick.\n')
    print(f'  {"block size":>12} {"max abs difference from the full matrix":>42}')
    for r in flash_table():
        print(f"  {r['blk']:>12} {r['err']:>42.3e}")
    print('\n  Agreeing to the last bit or two, at every block size.  This is an exact')
    print('  rewrite of the same sum, not an approximation of it - which is why')
    print('  FlashAttention changes the memory and the speed and not the model.')
    print('  What it buys is the next section.')

    # ------------------------------------------------------------------ 6
    header('6. The bill the quadratic term sends')
    print('  The score matrix has one entry per pair of positions, per head, per layer.')
    print('  In fp16, for 32 heads:\n')
    print(f'  {"context":>10} {"score matrix per layer":>24} {"with 32 layers":>18}')
    for ctx in (1024, 4096, 16384, 131072):
        b = score_matrix_bytes(ctx, 32)
        print(f'  {ctx:>10,} {b / 2**30:>21.2f} GB {32 * b / 2**30:>15.1f} GB')
    gb = score_matrix_bytes(131072, 32) / 2**30
    print(f'\n  At a context of 131,072 the score matrix alone is {gb:,.0f} GB per layer, which')
    print('  is not a thing you can allocate.  FlashAttention never allocates it: it')
    print('  holds one block at a time, so the memory becomes linear in the context')
    print('  while the arithmetic stays quadratic.  The quadratic work does not go')
    print('  away - only the quadratic storage does.')

    # ------------------------------------------------------------------ 7
    header('7. Paged attention: the KV cache is a memory allocator problem')
    lengths = [12, 300, 48, 900, 7, 2048, 64, 130]
    cap = 4096
    reserved, used, waste = contiguous_waste(lengths, cap)
    print(f'  {len(lengths)} sequences of lengths {lengths},')
    print(f'  each allowed to grow to {cap:,} tokens.\n')
    print(f'  one contiguous block per sequence: {reserved:,} slots, {used:,} used, '
          f'{100 * waste:.1f}% wasted')
    for page in (16, 64, 256):
        pages, res, usd, w = paged_waste(lengths, page)
        print(f'  pages of {page:>4}:              {res:,} slots, {usd:,} used, '
              f'{100 * w:.1f}% wasted ({pages} pages)')
    print('\n  A contiguous cache has to reserve for the longest the sequence might get,')
    print('  because it cannot move afterwards.  Pages are handed out as the sequence')
    print('  grows, so the only waste left is the unused tail of the last page - at')
    print('  most page_size - 1 slots per sequence, however long the sequence gets.')
    print('  Attention then reads through a page table instead of a flat array, which')
    print('  is the same indirection an operating system uses for exactly the same')
    print('  reason, and it is what lets two sequences share a prefix page.')

    # ------------------------------------------------------------------ 8
    header('8. MHA, MQA, GQA, MLA: four answers to "what has to be cached"')
    layers, heads, hd = 32, 32, 128
    rows = [('MHA  (32 kv heads)', kv_cache_bytes_per_token(layers, heads, hd)),
            ('GQA  (8 kv heads)', kv_cache_bytes_per_token(layers, heads, hd, kv_heads=8)),
            ('MLA  (512 latent + 64 rope)', kv_cache_bytes_per_token(layers, heads, hd,
                                                                     latent=512, rope_dim=64)),
            ('MQA  (1 kv head)', kv_cache_bytes_per_token(layers, heads, hd, kv_heads=1))]
    print(f'  {layers} layers, {heads} query heads, head dimension {hd}, fp16.\n')
    print(f'  {"layout":>27} {"bytes/token":>12} {"KB":>6} {"128k ctx":>12} {"vs MHA":>8}')
    base = rows[0][1]
    for name, b in rows:
        print(f'  {name:>27} {b:>12,} {b / 1024:>6.0f} {131072 * b / 2**30:>9.1f} GB '
              f'{base / b:>7.1f}x')
    st = linear_state_bytes(layers, heads, hd)
    print(f'\n  All four still grow with the context - the cache is per token.  A')
    print(f'  delta-rule layer does not: all {layers} layers together hold {st / 2**20:.0f} MB of state,')
    print('  and it is the same 32 MB whether the context is 1,000 tokens or 1,000,000.')
    print('\n  What each one gives up: MQA makes all 32 heads read the same K and V, so')
    print('  the heads stop being able to attend to different things.  GQA is the same')
    print('  trade in groups.  MLA keeps one low-rank vector per token and reconstructs')
    print('  per-head K and V from it, so the heads stay distinct and the cache still')
    print(f'  shrinks {base / rows[2][1]:.1f}x - at the cost of extra arithmetic at read time and a')
    print('  separate small path for the part that carries position information.')

    # ------------------------------------------------------------------ 9
    header('9. Linear attention and the delta rule: a state you write to, not append to')
    print('  Softmax attention keeps every key and value and re-reads all of them at')
    print('  every step - that is what makes it exact and what makes the cache grow.')
    print('  A delta-rule layer keeps one d x d matrix and edits it in place:\n')
    print('      S <- S (I - beta k k^T) + beta v k^T')
    print('      read: S @ q    (a real layer has its own query projection)\n')
    print('  The first term erases whatever S currently associates with k, the second')
    print('  writes v there instead.  How much fits:\n')
    print(f'  {"pairs written":>14} {"state":>10} {"worst recall error":>20} {"associations lost":>19}')
    for npairs in (8, 16, 17, 24, 32):
        e = delta_rule_recall(d=16, n=npairs)
        print(f'  {npairs:>14} {"16 x 16":>10} {e.max():>20.2e} {int((e > 0.5).sum()):>19}')
    print('\n  Sixteen orthogonal associations fit a 16 x 16 state: the 1e-12 error is the')
    print('  epsilon this file adds when it normalises the keys, not interference.  The')
    print('  seventeenth overwrites exactly one of them, and thirty-two overwrite')
    print(f'  sixteen.  The newest write is always perfect '
          f'({delta_rule_recall(d=16, n=32)[-1]:.1e}); it is the old ones that go.')
    smear = delta_rule_recall(d=16, n=8, orthogonal=False)
    print(f'\n  And that is with orthogonal keys.  With ordinary random keys, eight pairs')
    print(f'  into the same state already give a worst error of {smear.max():.2f}, because')
    print('  every write smears the ones already there.')
    print('\n  So the trade is not "linear attention is worse", it is: constant memory,')
    print('  bounded recall.  The gated delta-rule family adds a forget factor so old')
    print('  associations decay instead of persisting until something overwrites them -')
    print('  Gated DeltaNet uses one scalar per head, Kimi\'s KDA one per channel - which')
    print('  makes the bound behave better; the state is still a fixed-size matrix.')
    print('  That is why these layers get interleaved with a few full-attention ones')
    print('  rather than replacing them: the full layers do the exact long-range')
    print('  recall, while the linear layers carry everything else at constant cost')
    print('  per token.')

    # ------------------------------------------------------------------ 10
    header('10. LeetCode 239 - Sliding Window Maximum')
    nums, k = [1, 3, -1, -3, 5, 3, 6, 7], 3
    print(f'  {nums}, k = {k} -> {SolutionSlidingWindowMaximum().maxSlidingWindow(nums, k)}')
    print(f'  brute force agrees: {max_sliding_window_brute(nums, k)}\n')
    print('  A deque of indices whose values are strictly decreasing.  When a value')
    print('  arrives, everything smaller than it at the back is dead for good - the')
    print('  newcomer is both larger AND newer, so it outlives all of them and is')
    print('  always the better answer.  The front is the maximum of the window, and it')
    print('  gets dropped only when it falls out of the window by index.')
    print('  Every index is pushed once and popped once, so the whole sweep is O(n)')
    print('  even though it looks like a nested loop.')
    print('\n  This is the running maximum from section 5.  Online softmax carries one')
    print('  number instead of the whole row, and rescales what it has whenever that')
    print('  number changes - the same move, with a smaller window and an extra sum.')

    # ------------------------------------------------------------------ 11
    header('11. LeetCode 703 - Kth Largest Element in a Stream')
    kl = KthLargest(3, [4, 5, 8, 2])
    print(f'  KthLargest(3, [4, 5, 8, 2]); add 3, 5, 10, 9, 4')
    print(f'  -> {[kl.add(v) for v in [3, 5, 10, 9, 4]]}\n')
    print('  A min-heap of exactly k elements.  Its root is the k-th largest so far,')
    print('  and anything smaller than the root is discarded the moment it arrives, so')
    print('  the memory is k and not the length of the stream.')
    print('  The trap is which heap: a MAX-heap of everything also answers the query,')
    print('  in O(n) memory and O(k log n) per call.  The min-heap of size k is the')
    print('  one that makes the memory independent of how much has gone past.')
    print('\n  Which is the shape of this whole day: one pass over something you cannot')
    print('  store, and a bounded amount of state that has to be enough.  FlashAttention')
    print('  bounds it at two numbers per row and stays exact; a delta-rule layer bounds')
    print('  it at one matrix and gives up exactness.  Both are answers to the same')
    print('  question, and which one is right depends on what you are willing to lose.')

    # ------------------------------------------------------------------ 12
    header('12. Assertions')
    check()
    print('all assertions passed')


if __name__ == '__main__':
    main()
