"""Day 62 - Mixture of Experts: growing the parameters without growing the cost per token.

Day 53 measured where a transformer's parameters live: two thirds of them are in the
feed-forward block, and every token pays for all of them.  That is the observation the
whole of MoE is built on.  Replace the one FFN with E copies, send each token to the
best k of them, and the parameter count multiplies by E while the arithmetic per token
multiplies by k/1.  At E=64 and k=2 that is 32x the capacity for 2x the FLOPs.

Nothing is free.  The router has to learn which expert to use without being told; if it
is not pushed, it collapses onto a handful of experts and the rest are dead weight.  And
when the experts live on different GPUs, every token has to be shipped to its expert and
shipped back - two all-to-all collectives per layer, which is what expert parallelism
and DeepEP exist to make fast.

This file builds the whole thing in numpy: a dense FFN, a routed MoE trained from
scratch, the balancing methods, and a simulation of expert parallelism that shows where
the time actually goes.

Run `python moe.py` to print the whole story.
"""

import math

import numpy as np


# ================================================================ 1. the accounting
def ffn_vs_moe(d_model=4096, d_ff=14336, n_experts=64, k=2, shared=1, n_layers=61):
    """Parameters and arithmetic, per token, for one layer and for a whole model."""
    dense = 3 * d_model * d_ff                      # gate, up, down in a SwiGLU FFN
    routed = n_experts * dense
    active = (k + shared) * dense
    return {'d_model': d_model, 'd_ff': d_ff, 'experts': n_experts, 'k': k,
            'shared': shared,
            'dense_params': dense, 'moe_params': routed + shared * dense,
            'active_params': active,
            'param_ratio': (routed + shared * dense) / dense,
            'flop_ratio': active / dense,
            'capacity_per_flop': (routed + shared * dense) / active,
            'model_total': n_layers * (routed + shared * dense) / 1e9,
            'model_active': n_layers * active / 1e9}


def width_equivalent(d_model=4096, d_ff=14336, n_experts=64, k=2):
    """How wide a dense FFN with the same parameter count would have to be."""
    return {'dense_d_ff': d_ff, 'moe_total_d_ff': n_experts * d_ff,
            'active_d_ff': k * d_ff,
            'note': 'the same weights, but only k of them touched per token'}


# ================================================================ 2. the router
def softmax(x, axis=-1):
    e = np.exp(x - x.max(axis=axis, keepdims=True))
    return e / e.sum(axis=axis, keepdims=True)


def route(X, Wg, k=2, bias=None, renorm=True):
    """The gate: score every expert, keep the top k, renormalise the weights.

    `bias` is added to the scores for selection only - never to the weights that
    multiply the expert outputs.  That asymmetry is what makes loss-free balancing
    possible: it moves tokens without distorting the gradient the expert receives.
    """
    logits = X @ Wg                                   # (T, E)
    probs = softmax(logits)
    score = probs if bias is None else probs + bias
    idx = np.argsort(-score, axis=1, kind='stable')[:, :k]
    w = np.take_along_axis(probs, idx, axis=1)
    if renorm:
        w = w / w.sum(axis=1, keepdims=True)
    return idx, w, probs


def expert_loads(idx, n_experts):
    counts = np.bincount(idx.reshape(-1), minlength=n_experts)
    return counts


def balance_stats(counts):
    """Everything about load balance in three numbers."""
    f = counts / max(counts.sum(), 1)
    ideal = 1.0 / len(counts)
    p = f[f > 0]
    return {'max_over_mean': float(counts.max() / counts.mean()),
            'min_over_mean': float(counts.min() / counts.mean()),
            'dead': int((counts == 0).sum()),
            'entropy': float(-(p * np.log(p)).sum() / math.log(len(counts))),
            'cv': float(counts.std() / counts.mean()), 'ideal_share': ideal}


# ================================================================ 3. a trained MoE
def make_task(T=2048, d=24, n_clusters=8, seed=0):
    """Tokens whose right answer depends on which region of space they are in.

    The regime is a function of the token itself - which of n_clusters random
    directions it points most along - and each regime applies a different linear map.
    That is exactly the structure MoE is supposed to discover, and a single dense
    layer cannot represent it: it has to average over all the regimes at once.
    """
    rng = np.random.default_rng(seed)
    G = rng.standard_normal((d, n_clusters))          # the regime boundaries
    maps = rng.standard_normal((n_clusters, d, d)) / math.sqrt(d)
    X = rng.standard_normal((T, d)) / math.sqrt(d)
    which = np.argmax(X @ G, axis=1)
    Y = np.einsum('td,tde->te', X, maps[which])
    return {'X': X, 'Y': Y, 'which': which, 'clusters': n_clusters, 'd': d}


def aux_loss(probs, idx, n_experts):
    """The Switch/GShard balancing loss: E * sum_i (fraction routed) * (mean prob).

    Both factors matter.  The fraction is what actually happened and has no gradient;
    the mean probability is differentiable.  Multiplying them pushes probability away
    from experts that are already busy.
    """
    T = probs.shape[0]
    counts = expert_loads(idx, n_experts) / (T * idx.shape[1])
    mean_p = probs.mean(axis=0)
    return float(n_experts * (counts * mean_p).sum()), counts, mean_p


def train_moe(task, n_experts=16, k=2, steps=600, lr=4.0, aux=0.0, bias_rate=0.0,
              seed=1):
    """Train a router and E linear experts jointly, by hand.

    aux        > 0 adds the balancing loss (Switch-style)
    bias_rate  > 0 instead nudges a per-expert selection bias towards balance,
                 which is the loss-free scheme DeepSeek-V3 uses

    The expert weights here are the raw gate probabilities rather than renormalised
    ones, which is what GShard and Switch do and keeps the gradient exact.
    """
    rng = np.random.default_rng(seed)
    X, Y, d = task['X'], task['Y'], task['d']
    T = X.shape[0]
    Wg = rng.standard_normal((d, n_experts)) * 0.1
    We = rng.standard_normal((n_experts, d, d)) / math.sqrt(d)
    bias = np.zeros(n_experts)
    rows = np.arange(T)
    hist = []
    for t in range(steps):
        idx, w, probs = route(X, Wg, k, bias if bias_rate else None, renorm=False)
        gate = np.zeros((T, n_experts))               # weight per (token, expert)
        np.put_along_axis(gate, idx, w, axis=1)
        outs = np.stack([X @ We[e] for e in range(n_experts)])        # (E, T, d)
        out = np.einsum('te,etd->td', gate, outs)
        err = out - Y
        hist.append(float((err ** 2).mean()))
        # expert gradients: only the tokens that were routed there
        gWe = np.zeros_like(We)
        for e in range(n_experts):
            sel = gate[:, e] > 0
            if sel.any():
                gWe[e] = X[sel].T @ (err[sel] * gate[sel, e:e + 1]) * 2 / T
        # router gradient: through the gate weights, then through the softmax
        gw = np.einsum('td,etd->te', err, outs) * 2 / T
        gw = np.where(gate > 0, gw, 0.0)
        gl = probs * (gw - (gw * probs).sum(axis=1, keepdims=True))
        if aux:
            counts = expert_loads(idx, n_experts) / (T * k)
            gaux = aux * n_experts * counts / T
            gl += probs * (gaux - (gaux * probs).sum(axis=1, keepdims=True))
        Wg -= lr * (X.T @ gl)
        We -= lr * gWe
        if bias_rate:                                  # loss-free balancing
            counts = expert_loads(idx, n_experts)
            bias -= bias_rate * np.sign(counts - counts.mean())
    idx, w, probs = route(X, Wg, k, bias if bias_rate else None, renorm=False)
    counts = expert_loads(idx, n_experts)
    gate = np.zeros((T, n_experts))
    np.put_along_axis(gate, idx, w, axis=1)
    outs = np.stack([X @ We[e] for e in range(n_experts)])
    loss = float((((np.einsum('te,etd->td', gate, outs)) - Y) ** 2).mean())
    return {'loss': loss, 'hist': hist[::10], 'counts': counts,
            'balance': balance_stats(counts), 'Wg': Wg, 'We': We, 'bias': bias,
            'idx': idx, 'w': w, 'probs': probs,
            'purity': cluster_purity(idx[:, 0], task['which'], n_experts)}


def cluster_purity(top1, which, n_experts):
    """Did the router discover the structure?  Fraction of tokens whose expert is
    that expert's most common cluster."""
    hits = 0
    for e in range(n_experts):
        m = top1 == e
        if m.sum():
            hits += np.bincount(which[m]).max()
    return float(hits / len(top1))


def dense_baseline(task, steps=600, lr=4.0, seed=1):
    """One linear map for everything - the dense FFN this is replacing."""
    rng = np.random.default_rng(seed)
    X, Y, d = task['X'], task['Y'], task['d']
    W = rng.standard_normal((d, d)) / math.sqrt(d)
    for _ in range(steps):
        err = X @ W - Y
        W -= lr * (X.T @ err * 2 / X.shape[0])
    return float(((X @ W - Y) ** 2).mean())


# ================================================================ 4. capacity
def capacity_drop(counts, T, k, n_experts, factor=1.0):
    """Fixed buffers mean tokens past the limit are dropped, not queued."""
    cap = math.ceil(factor * k * T / n_experts)
    kept = np.minimum(counts, cap).sum()
    return {'factor': factor, 'capacity': cap, 'dropped': int(counts.sum() - kept),
            'drop_rate': float(1 - kept / counts.sum()),
            'padding': float(cap * n_experts / max(counts.sum(), 1) - 1)}


# ================================================================ 5. expert parallelism
def ep_simulation(counts, n_gpu=8, d_model=4096, bytes_per=2, k=2, dispatch_bytes=None):
    """Experts split across GPUs: every token travels to its expert and back.

    With E experts on N GPUs, each GPU owns E/N of them, and a token whose expert is
    elsewhere must be sent there (dispatch) and its result returned (combine).  Two
    all-to-all collectives per MoE layer, and they are *imbalanced* exactly as much as
    the router is: the slowest GPU sets the pace for everyone.
    """
    E = len(counts)
    per = E // n_gpu
    gpu = counts.reshape(n_gpu, per).sum(axis=1)
    tokens = counts.sum() / k                         # distinct tokens
    db = dispatch_bytes if dispatch_bytes else bytes_per
    dispatch = counts.sum() * d_model * db
    combine = counts.sum() * d_model * bytes_per
    local = tokens / n_gpu * k                        # if routing were uniform
    return {'gpus': n_gpu, 'experts_per_gpu': per, 'gpu_tokens': gpu,
            'max_over_mean': float(gpu.max() / gpu.mean()),
            'idle_share': float(1 - gpu.mean() / gpu.max()),
            'dispatch_bytes': dispatch, 'combine_bytes': combine,
            'total_mb': (dispatch + combine) / 1e6,
            'fp8_total_mb': (counts.sum() * d_model * 1 + combine) / 1e6,
            'ideal_tokens_per_gpu': local}


def ep_vs_tp(d_model=4096, d_ff=14336, n_experts=64, k=2, tokens=4096, n_gpu=8,
             bytes_per=2):
    """Why MoE layers use expert parallelism instead of tensor parallelism.

    Tensor parallelism splits every expert across all GPUs and pays an all-reduce of
    the full activation per layer.  Expert parallelism keeps each expert whole on one
    GPU and pays two all-to-alls of only the tokens that move.
    """
    tp_bytes = 2 * tokens * d_model * bytes_per * (n_gpu - 1) / n_gpu   # ring all-reduce
    ep_bytes = 2 * tokens * k * d_model * bytes_per * (n_gpu - 1) / n_gpu
    tp_weights = n_experts * 3 * d_model * d_ff / n_gpu
    ep_weights = n_experts * 3 * d_model * d_ff / n_gpu
    return {'tp_mb': tp_bytes / 1e6, 'ep_mb': ep_bytes / 1e6,
            'ratio': ep_bytes / tp_bytes, 'k': k,
            'weights_per_gpu_gb': tp_weights * 2 / 1e9,
            'note': 'same weights per GPU; the difference is what moves per token'}


def deepep_model(tokens=4096, d_model=7168, k=8, n_gpu=64, nvlink_gbs=160.0,
                 rdma_gbs=50.0, intranode=8):
    """Where the all-to-all time goes, and what the two DeepEP modes are for.

    Intranode links are several times faster than the network, so a dispatch that
    stays inside a node is cheap and one that crosses nodes is not.  DeepEP's
    high-throughput kernels forward tokens NVLink-first and then over RDMA once per
    destination node; its low-latency kernels use pure RDMA with no forwarding, which
    costs bandwidth but removes the hop that decode cannot afford.
    """
    moved = tokens * k
    bytes_bf16 = moved * d_model * 2
    bytes_fp8 = moved * d_model * 1
    cross = (n_gpu - intranode) / n_gpu               # share that leaves the node
    t_nv = bytes_bf16 * (1 - cross) / (nvlink_gbs * 1e9)
    t_rdma = bytes_bf16 * cross / (rdma_gbs * 1e9)
    t_fp8 = (bytes_fp8 * (1 - cross) / (nvlink_gbs * 1e9)
             + bytes_fp8 * cross / (rdma_gbs * 1e9))
    return {'tokens': tokens, 'k': k, 'moved': moved,
            'mb_bf16': bytes_bf16 / 1e6, 'mb_fp8': bytes_fp8 / 1e6,
            'cross_share': cross, 'ms_bf16': (t_nv + t_rdma) * 1e3,
            'ms_fp8': t_fp8 * 1e3, 'speedup': (t_nv + t_rdma) / t_fp8}


# ================================================================ 6. checks
def check():
    acc = ffn_vs_moe()
    assert acc['param_ratio'] > 10 * acc['flop_ratio']        # the whole point
    assert abs(acc['flop_ratio'] - 3.0) < 1e-9                # k=2 plus 1 shared
    task = make_task(T=1024, d=16, n_clusters=6, seed=2)
    dense = dense_baseline(task, steps=200)
    plain = train_moe(task, n_experts=16, steps=200, aux=0.0)
    bal = train_moe(task, n_experts=16, steps=200, aux=0.3)
    free = train_moe(task, n_experts=16, steps=200, bias_rate=0.001)
    # the MoE beats one dense map on data that has several regimes
    assert plain['loss'] < dense, (plain['loss'], dense)
    # both balancing methods spread the tokens more than no balancing at all
    assert bal['balance']['max_over_mean'] < plain['balance']['max_over_mean']
    assert free['balance']['max_over_mean'] < plain['balance']['max_over_mean']
    assert free['balance']['dead'] == 0
    # a bias only moves selection, never the weight that multiplies an expert
    idx0, w0, p0 = route(task['X'], plain['Wg'], 2, None)
    idx1, w1, p1 = route(task['X'], plain['Wg'], 2, np.full(16, 0.3))
    assert np.allclose(p0, p1) and np.allclose(idx0, idx1)    # uniform bias: no change
    # capacity: a tighter factor drops more
    c = plain['counts']
    d1 = capacity_drop(c, 1024, 2, 16, 1.0)
    d2 = capacity_drop(c, 1024, 2, 16, 1.5)
    assert d1['drop_rate'] >= d2['drop_rate']
    # EP: the slowest GPU is set by the router, and fp8 dispatch moves fewer bytes
    ep = ep_simulation(c[:8], n_gpu=4, d_model=512)
    assert ep['max_over_mean'] >= 1.0 and ep['fp8_total_mb'] < ep['total_mb']
    dp = deepep_model()
    assert dp['ms_fp8'] < dp['ms_bf16'] and abs(dp['speedup'] - 2.0) < 1e-9


# ================================================================ 7. the story
def header(t):
    print('\n' + '=' * 72)
    print(t)
    print('=' * 72)


def main():
    header('1. Where this comes from: the FFN is most of the model')
    acc = ffn_vs_moe()
    w = width_equivalent()
    print(f'  A transformer block spends two thirds of its parameters in the')
    print(f'  feed-forward network, and every token pays for all of them.\n')
    print(f'    d_model {acc["d_model"]}, d_ff {acc["d_ff"]}')
    print(f'    one dense FFN           {acc["dense_params"] / 1e6:8.1f} M parameters')
    print(f'    {acc["experts"]} experts + {acc["shared"]} shared   '
          f'{acc["moe_params"] / 1e6:8.1f} M parameters')
    print(f'    active per token        {acc["active_params"] / 1e6:8.1f} M '
          f'(top-{acc["k"]} plus the shared one)\n')
    print(f'  So the parameters go up {acc["param_ratio"]:.0f}x and the arithmetic per')
    print(f'  token only {acc["flop_ratio"]:.0f}x: {acc["capacity_per_flop"]:.1f} times')
    print('  more capacity for the same cost per token.  Across a 61-layer model that')
    print(f'  is {acc["model_total"]:.0f} B parameters of which {acc["model_active"]:.0f} B '
          f'run for any given token -')
    print('  the shape of every large open MoE released in the last two years.')
    print(f'\n  Another way to say it: the MoE layer is a dense FFN of width')
    print(f'  {w["moe_total_d_ff"]:,} that only ever activates {w["active_d_ff"]:,} '
          f'of it per token.')

    header('2. The router')
    print('  A linear layer scores the experts, softmax turns the scores into')
    print('  probabilities, the top k are kept and their weights renormalised:\n')
    print('    logits = x @ Wg            # (tokens, experts)')
    print('    probs  = softmax(logits)')
    print('    idx    = argsort(-probs)[:k]')
    print('    w      = probs[idx] / probs[idx].sum()')
    print('    y      = sum_j w[j] * expert[idx[j]](x)\n')
    print('  Two things make this harder than it looks.  The selection is discrete, so')
    print('  no gradient flows through it - the router only learns through the weights')
    print('  w of the experts it already chose, which means an expert that is never')
    print('  chosen never improves and never gets chosen.  And the whole layer is only')
    print('  efficient if the tokens spread out, which nothing in the loss asks for.')

    header('3. A routed MoE, trained from scratch')
    task = make_task()
    dense = dense_baseline(task)
    plain = train_moe(task, aux=0.0)
    n_e = len(plain['counts'])
    print(f'  {task["X"].shape[0]} tokens from {task["clusters"]} regimes, each with '
          f'its own linear map.\n')
    print(f'    one dense map (the FFN being replaced)    loss {dense:.4f}')
    print(f'    {n_e} experts, top-2, no balancing            loss {plain["loss"]:.4f}'
          f'   ({(1 - plain["loss"] / dense):.0%} better)')
    b = plain['balance']
    print(f'\n  The MoE wins on the loss, and then look at the load:\n')
    print(f'    tokens per expert     {sorted(plain["counts"].tolist(), reverse=True)}')
    print(f'    busiest / average     {b["max_over_mean"]:.2f}')
    print(f'    quietest / average    {b["min_over_mean"]:.2f}')
    print(f'    dead experts          {b["dead"]} of {len(plain["counts"])}')
    print(f'    routing entropy       {b["entropy"]:.3f} of 1.0')
    print('\n  That is expert collapse, and it is self-reinforcing: the experts that')
    print('  win early get more gradient, get better, and win more.  The parameters')
    print('  in the dead experts are not merely useless, they were paid for in memory')
    print('  and in the all-to-all that still has to carry their shape.')

    header('4. Two ways to make the tokens spread out')
    bal = train_moe(task, aux=0.3)
    free = train_moe(task, bias_rate=0.001)
    print('  (a) An auxiliary loss.  Switch and GShard add')
    print('      E * sum_i (fraction of tokens routed to i) * (mean probability of i),')
    print('      which is minimised when both are flat.  The fraction has no gradient;')
    print('      the probability does, so the push lands on the router.\n')
    print('  (b) No loss at all.  DeepSeek-V3 adds a per-expert bias to the *selection*')
    print('      score only, and nudges it by the sign of that expert\'s load error')
    print('      after every step.  The bias moves tokens; it never touches the weight')
    print('      that multiplies the expert output, so it adds no gradient of its own')
    print('      and cannot fight the task loss.\n')
    rows = [('no balancing', plain), ('auxiliary loss', bal), ('bias, no loss', free)]
    print('  method            loss     busiest/avg   dead   entropy   router purity')
    for name, r in rows:
        bb = r['balance']
        print(f'  {name:16} {r["loss"]:.4f}   {bb["max_over_mean"]:11.2f}   '
              f'{bb["dead"]:4d}   {bb["entropy"]:7.3f}   {r["purity"]:13.3f}')
    print('\n  Purity is the fraction of tokens whose expert is that expert\'s most')
    print('  common regime - did the router find the structure that is really there.')
    print('\n  Read this table honestly.  Balancing costs a little loss here: the')
    print(f'  auxiliary loss gives up {(bal["loss"] / plain["loss"] - 1):.0%} and the bias '
          f'scheme {(free["loss"] / plain["loss"] - 1):.0%}, because both are pushing')
    print('  tokens away from the expert the router would have picked.  What they buy')
    print('  is the load column: no dead experts, and a busiest-to-average that falls')
    print(f'  from {plain["balance"]["max_over_mean"]:.2f} to '
          f'{free["balance"]["max_over_mean"]:.2f}.  On a toy that looks like a bad')
    print('  trade.  The next three sections are why it is not: at scale that ratio')
    print('  is a capacity factor, a dropped-token rate, and the fraction of a GPU')
    print('  fleet that sits idle waiting for one rank.')
    print('\n  Note also which method wins which column.  The auxiliary loss keeps')
    print('  the specialisation the router already found; the bias scheme flattens')
    print('  the load much harder and costs more purity.  That is the real difference')
    print('  between them - one is a gradient that negotiates with the task loss, the')
    print('  other is a thumb on the scale that does not.')

    header('5. Capacity: what happens to the overflow')
    print('  Kernels want fixed shapes, so each expert gets a buffer of')
    print('  capacity = factor * k * tokens / experts, and tokens past it are dropped')
    print('  (they skip the layer through the residual connection).\n')
    n_e = len(plain['counts'])
    print('  factor   capacity   dropped   drop rate   padding waste')
    for f in (1.0, 1.25, 1.5, 2.0):
        d = capacity_drop(plain['counts'], task['X'].shape[0], 2, n_e, f)
        print(f'  {d["factor"]:6.2f}   {d["capacity"]:8d}   {d["dropped"]:7d}   '
              f'{d["drop_rate"]:9.2%}   {d["padding"]:13.1%}')
    print('\n  With the flattest router from the previous section (bias, '
          f'busiest/avg {free["balance"]["max_over_mean"]:.2f}):\n')
    print('  factor   dropped   drop rate')
    for f in (1.0, 1.25, 1.5):
        d = capacity_drop(free['counts'], task['X'].shape[0], 2, n_e, f)
        print(f'  {d["factor"]:6.2f}   {d["dropped"]:7d}   {d["drop_rate"]:9.2%}')
    print('\n  Note what the capacity factor is really buying: at factor 1.0 the')
    print('  buffers hold exactly the average load, so every token above average is')
    print('  dropped and every token below it leaves the buffer half empty.  Raising')
    print('  the factor pads every expert to make room for the busiest one, so the')
    print('  padding column is wasted arithmetic on *all* the experts.  Balance is')
    print('  what lets you run a small factor without dropping anything.')
    print('\n  Dropping is a training-time compromise; at inference nobody drops')
    print('  tokens, so the buffers are sized for the worst case instead and the')
    print('  imbalance turns into wasted compute rather than lost tokens.')

    header('6. Expert parallelism: the experts do not fit on one GPU')
    ep = ep_simulation(plain['counts'], n_gpu=4, d_model=4096)
    epb = ep_simulation(free['counts'], n_gpu=4, d_model=4096)
    print(f'  {len(plain["counts"])} experts over {ep["gpus"]} GPUs is '
          f'{ep["experts_per_gpu"]} experts each.  A token whose expert')
    print('  lives elsewhere has to be sent there and its result sent back: two')
    print('  all-to-all collectives per MoE layer, every layer, every step.\n')
    print(f'    tokens per GPU, unbalanced router   {ep["gpu_tokens"].tolist()}')
    print(f'    busiest / average                   {ep["max_over_mean"]:.2f}')
    print(f'    tokens per GPU, balanced router     {epb["gpu_tokens"].tolist()}')
    print(f'    busiest / average                   {epb["max_over_mean"]:.2f}')
    print('\n  This is the real reason load balancing is a systems problem and not an')
    print('  accuracy nicety: every GPU waits for the slowest one at the collective,')
    print(f'  so an imbalance of {ep["max_over_mean"]:.2f} means the other GPUs idle for')
    print(f'  {ep["idle_share"]:.0%} of the expert computation.')
    tp = ep_vs_tp()
    print(f'\n  And why expert parallelism rather than splitting each expert across all')
    print('  GPUs (tensor parallelism)?  Both hold the same weights per GPU; the')
    print('  difference is what moves per token:\n')
    print(f'    tensor parallel: all-reduce of every activation   '
          f'{tp["tp_mb"]:.1f} MB per layer')
    print(f'    expert parallel: all-to-all of the routed tokens  '
          f'{tp["ep_mb"]:.1f} MB per layer')
    print(f'\n  At k={tp["k"]} the all-to-all moves {tp["ratio"]:.0f}x the bytes of the')
    print('  all-reduce - but it is a different shape of traffic: sparse, uneven, and')
    print('  it overlaps with compute, which the all-reduce at the end of a layer')
    print('  cannot.  Real systems use both, expert parallel across the experts and')
    print('  tensor parallel inside the attention.')

    header('7. DeepEP: making the all-to-all affordable')
    dp = deepep_model()
    print('  DeepEP is the communication library written for exactly this collective.')
    print('  Three things in it are worth understanding even if you never call it.\n')
    print(f'  (a) Two modes.  With {dp["tokens"]} tokens, top-{dp["k"]} routing and')
    print(f'      d_model {7168}, one dispatch moves {dp["mb_bf16"]:.0f} MB.  Inside a')
    print('      node that is NVLink; between nodes it is RDMA at a fraction of the')
    print('      speed.  The high-throughput kernels forward a token over NVLink and')
    print('      then send it once per destination node over RDMA, which is the right')
    print('      trade for prefill and training.  The low-latency kernels skip the')
    print('      forwarding entirely and use pure RDMA, because in decode the hop')
    print('      costs more than the bandwidth it saves.\n')
    print(f'  (b) FP8 dispatch.  Sending the tokens in FP8 instead of bf16 halves the')
    print(f'      bytes: {dp["mb_bf16"]:.0f} MB becomes {dp["mb_fp8"]:.0f} MB, '
          f'{dp["ms_bf16"]:.2f} ms becomes {dp["ms_fp8"]:.2f} ms in this model.')
    print('      The combine stays in bf16, because summing k expert outputs in FP8')
    print('      loses too much - the previous day, applied to a network link.\n')
    print('  (c) Overlap.  The low-latency kernels expose a hook so the receive is')
    print('      not a barrier: a layer can compute attention while the MoE tokens')
    print('      are still in flight, and nothing is burned on SM cycles spinning on')
    print('      the network.  That is what makes the two all-to-alls per layer')
    print('      survivable at all.')

    header('8. Writing the layer in PyTorch')
    print('  The whole layer is about fifty lines, and the interesting part is that')
    print('  the loop over experts is the thing you are trying to avoid.\n')
    print('    class MoE(nn.Module):')
    print('        def __init__(self, d_model, d_ff, n_experts, k):')
    print('            self.gate = nn.Linear(d_model, n_experts, bias=False)')
    print('            # all the experts as ONE parameter, not a ModuleList:')
    print('            self.w1 = nn.Parameter(torch.empty(n_experts, d_model, d_ff))')
    print('            self.w2 = nn.Parameter(torch.empty(n_experts, d_ff, d_model))')
    print('            self.k = k\n')
    print('        def forward(self, x):                      # x: (tokens, d_model)')
    print('            probs = self.gate(x).softmax(-1)')
    print('            w, idx = probs.topk(self.k, dim=-1)    # (tokens, k)')
    print('            w = w / w.sum(-1, keepdim=True)\n')
    print('            flat = idx.reshape(-1)                 # k entries per token')
    print('            order = flat.argsort()                 # group by expert')
    print('            counts = torch.bincount(flat, minlength=self.w1.shape[0])')
    print('            xs = x.repeat_interleave(self.k, 0)[order]\n')
    print('            # one grouped matmul instead of a Python loop over experts:')
    print('            h = torch._grouped_mm(xs, self.w1, counts)   # or a for-loop')
    print('            h = F.silu(h)')
    print('            ys = torch._grouped_mm(h, self.w2, counts)\n')
    print('            out = torch.zeros_like(x)')
    print('            ys = ys[order.argsort()] * w.reshape(-1, 1)')
    print('            out.index_add_(0, torch.arange(len(x)).repeat_interleave(self.k),')
    print('                           ys)')
    print('            return out\n')
    print('  Four design decisions are hiding in there, and they are the same four in')
    print('  every real implementation:\n')
    print('  1. The experts are ONE tensor of shape (E, d_model, d_ff), not a')
    print('     ModuleList of E Linears.  A list means E separate kernel launches and')
    print('     E separate weight loads; one tensor means the expert index is just an')
    print('     index, which is what lets the next three things work.\n')
    print('  2. Sort, do not mask.  The obvious implementation multiplies by a one-hot')
    print('     mask and runs every expert on every token - correct, and it throws')
    print('     away the entire point, because it does E times the arithmetic.  Real')
    print('     layers sort the tokens by expert id, run one grouped matmul over the')
    print('     resulting variable-sized groups (torch._grouped_mm, or the Triton')
    print('     kernels in Megatron-LM and vLLM), and unsort at the end.\n')
    print('  3. Scatter back with index_add_, because a token appears k times and its')
    print('     k expert outputs have to be summed.  This is also where the gradient')
    print('     of the gate weights comes from, so it has to be an add and not a')
    print('     write.\n')
    print('  4. The balancing loss is returned from forward, not computed in the loss')
    print('     function: it needs `probs` and the routing counts, which only exist')
    print('     inside the layer.  Every MoE codebase has a module-level list that')
    print('     the layers append their aux losses to, and the training loop sums it.')
    print('\n  For the distributed version, the dispatch is literally one collective:')
    print('     torch.distributed.all_to_all_single(recv, send, out_splits, in_splits)')
    print('  where the splits are the per-rank token counts from the sort.  That is')
    print('  the line that makes an imbalanced router cost real time, because the')
    print('  splits are uneven and everyone waits.')

    header('9. Writing the layer in JAX')
    print('  Same algorithm, and the code looks completely different, because JAX')
    print('  wants a fixed shape and a sharding rule rather than a gather:\n')
    print('    def moe(x, wg, w1, w2, k, capacity):')
    print('        probs = jax.nn.softmax(x @ wg)              # (T, E)')
    print('        w, idx = jax.lax.top_k(probs, k)')
    print('        # one-hot dispatch into FIXED buffers: (T, E, capacity)')
    print('        mask = jax.nn.one_hot(idx, n_experts).sum(1)')
    print('        pos = jnp.cumsum(mask, axis=0) * mask - 1   # slot inside the buffer')
    print('        disp = mask[..., None] * jax.nn.one_hot(pos, capacity)')
    print('        xe = jnp.einsum("te,tec,td->ecd", mask, disp, x)   # (E, cap, d)')
    print('        he = jax.nn.silu(jnp.einsum("ecd,edf->ecf", xe, w1))')
    print('        ye = jnp.einsum("ecf,efd->ecd", he, w2)')
    print('        return jnp.einsum("ecd,tec,te->td", ye, disp, w)\n')
    print('  The einsum chain *is* the dispatch and the combine.  Nothing is sorted,')
    print('  nothing is variable length, and the capacity appears explicitly as an')
    print('  array dimension - which is exactly why the capacity factor of section 5')
    print('  was invented here: with fixed buffers, the overflow has nowhere to go.')
    print('\n  And the parallelism is a decorator rather than a collective:\n')
    print('    mesh = Mesh(devices, ("data", "expert"))')
    print('    w1 = jax.device_put(w1, NamedSharding(mesh, P("expert", None, None)))')
    print('    moe = shard_map(moe, mesh, in_specs=..., out_specs=...)\n')
    print('  Sharding the expert axis of w1 *is* expert parallelism; XLA inserts the')
    print('  all-to-all because the einsum needs data that lives on another device.')
    print('  Nobody writes the collective, which is the real difference between the')
    print('  two ecosystems: in PyTorch you call all_to_all_single and own the splits;')
    print('  in JAX you annotate the axis and the compiler emits the communication.')
    print('\n  The trade is the one you would expect.  The gather-and-sort version')
    print('  does arithmetic proportional to the tokens actually routed, and needs')
    print('  custom kernels to be fast; the one-hot version is pure dense einsum that')
    print('  any compiler can fuse, and pays for every empty slot in the buffers.')

    header('10. In the serving stacks, briefly')
    print('  Both servers expose the same two decisions.  Where the experts live:')
    print('  SGLang `--ep-size 8`, vLLM `--enable-expert-parallel`.  And what carries')
    print('  the all-to-all: SGLang `--enable-deepep-moe` with `--deepep-mode')
    print('  normal|low_latency|auto` (section 7), vLLM picking DeepEP or')
    print('  pplx-kernels underneath.  Both end up running data-parallel attention')
    print('  next to expert-parallel experts, because attention has no experts to')
    print('  spread and the MoE weights do not fit otherwise.')
    print('\n  The flag worth knowing beyond that is SGLang\'s `--enable-eplb` with')
    print('  `--ep-num-redundant-experts`: a balancing loss flattens the *average*,')
    print('  and real traffic is not the average, so the server measures the load per')
    print('  expert, moves experts between ranks, and replicates the hottest ones.')
    print('  Section 4 reappearing as an operations problem.')

    header('10. Assertions')
    check()
    print('all assertions passed')


if __name__ == '__main__':
    main()
