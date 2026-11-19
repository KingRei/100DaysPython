"""Day 63 - parallelism and collective communication.

Every way of splitting a model across GPUs is really a choice of which
collective runs, how often, and on how many bytes.  This module builds the
collectives from scratch on a simulated ring of ranks, counts the bytes and
the steps, and then re-derives data, tensor, expert, context and pipeline
parallelism as different call sites for those same five primitives.

Run it:  python parallelism.py
"""

import numpy as np

BYTES = {'fp32': 4, 'bf16': 2, 'fp8': 1}


# ----------------------------------------------------------------- the wire
class Wire:
    """A cost model for one collective: steps x latency + bytes / bandwidth.

    `alpha_us` is the per-hop latency and `beta_gbs` the per-link bandwidth.
    Defaults are the order-of-magnitude figures for an in-node link; pass
    the cross-node pair to see what changes.
    """

    def __init__(self, alpha_us=2.0, beta_gbs=400.0, name='in-node link'):
        self.alpha_us, self.beta_gbs, self.name = alpha_us, beta_gbs, name
        self.reset()

    def reset(self):
        self.steps = 0
        self.bytes_per_rank = 0

    def hop(self, nbytes):
        """One communication step in which every rank ships `nbytes`."""
        self.steps += 1
        self.bytes_per_rank += nbytes

    @property
    def micros(self):
        return self.steps * self.alpha_us + self.bytes_per_rank / (self.beta_gbs * 1e3)


def time_us(steps, nbytes, alpha_us=2.0, beta_gbs=400.0):
    """The same model without running anything: hops x latency + bytes / bw.

    `nbytes` is what crosses the critical path, i.e. the sum over steps of
    what one rank ships in that step.
    """
    return steps * alpha_us + nbytes / (beta_gbs * 1e3)


def ring_us(nbytes, p=8, **kw):
    return time_us(2 * (p - 1), 2 * (p - 1) / p * nbytes, **kw)


def tree_us(nbytes, p=8, **kw):
    """A tree moves the *whole* tensor on every level, up and back down."""
    levels = 2 * np.log2(p)
    return time_us(levels, levels * nbytes, **kw)


# ----------------------------------------------------------- the collectives
def ring_reduce_scatter(bufs, wire=None):
    """Each rank ends up owning one fully reduced chunk.

    Step s: rank r sends the chunk it finished last round to r+1 and adds the
    one arriving from r-1.  After p-1 steps rank r owns chunk (r+1) % p.
    """
    p = len(bufs)
    chunks = [list(np.array_split(b.astype(np.float32).copy(), p)) for b in bufs]
    item = bufs[0].dtype.itemsize
    for s in range(p - 1):
        incoming = {}
        for r in range(p):
            idx = (r - s) % p                      # what rank r ships this step
            incoming[((r + 1) % p, idx)] = chunks[r][idx]
        for (dst, idx), payload in incoming.items():
            chunks[dst][idx] = chunks[dst][idx] + payload
        if wire:
            wire.hop(bufs[0].size // p * item)
    return [chunks[r][(r + 1) % p] for r in range(p)]


def ring_all_gather(owned, wire=None, item=4):
    """Rank r starts with chunk (r+1) % p and ends with the whole vector."""
    p = len(owned)
    have = [{(r + 1) % p: owned[r].copy()} for r in range(p)]
    for s in range(p - 1):
        incoming = {}
        for r in range(p):
            idx = (r + 1 - s) % p                  # the chunk r just received
            incoming[(r + 1) % p] = (idx, have[r][idx])
        for dst, (idx, payload) in incoming.items():
            have[dst][idx] = payload.copy()
        if wire:
            wire.hop(owned[0].size * item)
    return [np.concatenate([have[r][i] for i in range(p)]) for r in range(p)]


def ring_all_reduce(bufs, wire=None):
    """all-reduce = reduce-scatter + all-gather, the standard ring version."""
    item = bufs[0].dtype.itemsize
    owned = ring_reduce_scatter(bufs, wire)
    return ring_all_gather(owned, wire, item)


def tree_all_reduce(bufs, wire=None):
    """Reduce up a binary tree and broadcast down it: 2*log2(p) steps."""
    p = len(bufs)
    item = bufs[0].dtype.itemsize
    cur = [b.astype(np.float32).copy() for b in bufs]
    span = 1
    while span < p:
        for r in range(0, p, 2 * span):
            if r + span < p:
                cur[r] = cur[r] + cur[r + span]
        if wire:
            wire.hop(bufs[0].size * item)
        span *= 2
    span //= 2
    while span >= 1:
        for r in range(0, p, 2 * span):
            if r + span < p:
                cur[r + span] = cur[r].copy()
        if wire:
            wire.hop(bufs[0].size * item)
        span //= 2
    return cur


def all_to_all(bufs, wire=None):
    """Rank r hands shard j to rank j and keeps what everybody sent to r.

    This is the one MoE needs: it is a transpose of the rank x shard matrix.
    """
    p = len(bufs)
    item = bufs[0].dtype.itemsize
    shards = [list(np.array_split(b, p)) for b in bufs]
    out = [np.concatenate([shards[src][r] for src in range(p)]) for r in range(p)]
    if wire:
        for _ in range(p - 1):
            wire.hop(bufs[0].size // p * item)
    return out


def broadcast(buf, p, wire=None):
    """Tree broadcast: log2(p) steps, every rank sends at most once."""
    item = buf.dtype.itemsize
    out = [None] * p
    out[0] = buf.copy()
    span = 1
    while span < p:
        have = [r for r in range(p) if out[r] is not None]
        for r in have:
            if r + span < p and out[r + span] is None:
                out[r + span] = out[r].copy()
        if wire:
            wire.hop(buf.size * item)
        span *= 2
    return out


COLLECTIVES = [
    # name, what it does, bytes each rank ships, steps (ring/tree)
    ('broadcast', 'one rank has it, everybody needs it', 'N', 'log2(p)'),
    ('reduce', 'everybody has a piece of the sum, one rank needs it', 'N', 'log2(p)'),
    ('all-reduce', 'everybody has a partial sum, everybody needs the total',
     '2(p-1)/p * N', '2(p-1)'),
    ('reduce-scatter', 'everybody has a partial sum, each keeps 1/p of it',
     '(p-1)/p * N', 'p-1'),
    ('all-gather', 'each rank has 1/p, everybody needs the whole thing',
     '(p-1)/p * N', 'p-1'),
    ('all-to-all', 'rank r holds a shard addressed to every other rank',
     '(p-1)/p * N', 'p-1'),
    ('point-to-point', 'rank r hands a tensor to rank r+1 and nobody else cares',
     'N', '1'),
]


def collective_costs(p=8, n_bytes=256 * 1024 ** 2):
    """The bytes-on-the-wire table, as formulas and as megabytes."""
    rows = []
    for name, what, fb, fs in COLLECTIVES:
        per = {'broadcast': n_bytes, 'reduce': n_bytes,
               'all-reduce': 2 * (p - 1) / p * n_bytes,
               'reduce-scatter': (p - 1) / p * n_bytes,
               'all-gather': (p - 1) / p * n_bytes,
               'all-to-all': (p - 1) / p * n_bytes,
               'point-to-point': n_bytes}[name]
        steps = {'broadcast': np.log2(p), 'reduce': np.log2(p),
                 'all-reduce': 2 * (p - 1), 'reduce-scatter': p - 1,
                 'all-gather': p - 1, 'all-to-all': p - 1,
                 'point-to-point': 1}[name]
        on_path = {'broadcast': np.log2(p) * n_bytes, 'reduce': np.log2(p) * n_bytes,
                   'point-to-point': n_bytes}.get(name, per)
        rows.append({'name': name, 'what': what, 'bytes_formula': fb,
                     'steps_formula': fs, 'mb_per_rank': per / 1024 ** 2,
                     'steps': steps, 'us': time_us(steps, on_path)})
    return rows


# --------------------------------------------------------- a model to split
MODEL = {'name': 'a 70B-class dense transformer', 'params': 70e9, 'd_model': 8192,
         'd_ff': 28672, 'layers': 80, 'heads': 64, 'kv_heads': 8, 'head_dim': 128}


def weight_bytes(params=MODEL['params'], dtype='bf16'):
    return params * BYTES[dtype]


# ------------------------------------------------------------ data parallel
def toy_data(n=512, d=16, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n, d)).astype(np.float32)
    w_true = rng.standard_normal((d, 1)).astype(np.float32)
    y = np.tanh(x @ w_true) + 0.05 * rng.standard_normal((n, 1)).astype(np.float32)
    return x, y.astype(np.float32)


def init_params(d=16, h=32, seed=1):
    rng = np.random.default_rng(seed)
    return [(rng.standard_normal((d, h)) / np.sqrt(d)).astype(np.float32),
            (rng.standard_normal((h, 1)) / np.sqrt(h)).astype(np.float32)]


def loss_and_grads(params, x, y):
    w1, w2 = params
    h = np.tanh(x @ w1)
    pred = h @ w2
    err = pred - y
    loss = float((err ** 2).mean())
    n = x.shape[0]
    g2 = h.T @ (2 * err / n)
    gh = (2 * err / n) @ w2.T * (1 - h ** 2)
    g1 = x.T @ gh
    return loss, [g1.astype(np.float32), g2.astype(np.float32)]


def train_single(steps=60, lr=0.5, seed=1):
    x, y = toy_data()
    params = init_params(seed=seed)
    for _ in range(steps):
        loss, grads = loss_and_grads(params, x, y)
        params = [p - lr * g for p, g in zip(params, grads)]
    return float(loss_and_grads(params, x, y)[0]), params


def train_ddp(p=4, steps=60, lr=0.5, seed=1, divide_by_world=True, wire=None):
    """Every rank keeps a full replica and sees 1/p of the batch.

    The gradient all-reduce is what makes the replicas stay identical: the
    sum of per-rank gradients divided by p is exactly the full-batch gradient,
    because the loss is a mean over the batch.
    """
    x, y = toy_data()
    shards = [(x[r::p], y[r::p]) for r in range(p)]
    params = [init_params(seed=seed) for _ in range(p)]
    for _ in range(steps):
        grads = [loss_and_grads(params[r], *shards[r])[1] for r in range(p)]
        for t in range(len(grads[0])):
            flat = [g[t].ravel() for g in grads]
            summed = ring_all_reduce(flat, wire)
            for r in range(p):
                avg = summed[r] / (p if divide_by_world else 1)
                grads[r][t] = avg.reshape(grads[r][t].shape)
        params = [[q - lr * g for q, g in zip(params[r], grads[r])] for r in range(p)]
    return float(loss_and_grads(params[0], x, y)[0]), params[0]


def ddp_overlap(buckets=(1, 4, 25), params_gb=1.0, p=8, backward_ms=100.0,
                alpha_us=2.0, beta_gbs=400.0):
    """Why DDP groups gradients into buckets instead of one giant all-reduce.

    Gradients become ready back to front, so a bucket can start flying while
    the rest of the backward pass is still running.  Only the last bucket is
    exposed; everything earlier hides under compute.
    """
    rows = []
    total_bytes = params_gb * 1024 ** 3
    for nb in buckets:
        per = total_bytes / nb
        comm_ms = time_us(2 * (p - 1), 2 * (p - 1) / p * per, alpha_us, beta_gbs) / 1e3
        exposed = comm_ms                       # the final bucket cannot overlap
        rows.append({'buckets': nb, 'bucket_mb': per / 1024 ** 2,
                     'comm_ms_each': comm_ms, 'all_comm_ms': comm_ms * nb,
                     'step_ms': backward_ms + exposed, 'exposed_ms': exposed})
    return rows


# -------------------------------------------------- ZeRO / FSDP accounting
def optimizer_state_bytes(mixed_precision=True):
    """Adam on one parameter, in bytes."""
    if mixed_precision:
        return {'param (bf16)': 2, 'grad (bf16)': 2, 'master param (fp32)': 4,
                'adam m (fp32)': 4, 'adam v (fp32)': 4}
    return {'param (fp32)': 4, 'grad (fp32)': 4, 'adam m (fp32)': 4, 'adam v (fp32)': 4}


def zero_table(params=MODEL['params'], p=8):
    """Memory per rank and bytes on the wire for each sharding stage."""
    s = optimizer_state_bytes()
    pbytes, gbytes = s['param (bf16)'], s['grad (bf16)']
    obytes = s['master param (fp32)'] + s['adam m (fp32)'] + s['adam v (fp32)']
    n = params * pbytes                                  # one bf16 copy, in bytes
    ar = 2 * (p - 1) / p * n                             # ring all-reduce volume
    half = (p - 1) / p * n                               # one-directional volume
    rows = [
        ('DDP (no sharding)', pbytes + gbytes + obytes, ar, 'all-reduce(grad)'),
        ('ZeRO-1 (optimizer)', pbytes + gbytes + obytes / p, half * 2,
         'reduce-scatter(grad) + all-gather(param)'),
        ('ZeRO-2 (+ gradients)', pbytes + (gbytes + obytes) / p, half * 2,
         'reduce-scatter(grad) + all-gather(param)'),
        ('ZeRO-3 / FSDP (+ parameters)', (pbytes + gbytes + obytes) / p, half * 3,
         'all-gather(param) x2 + reduce-scatter(grad)'),
    ]
    out = []
    for name, per_param, vol, how in rows:
        out.append({'stage': name, 'bytes_per_param': per_param,
                    'gb_per_rank': per_param * params / 1024 ** 3,
                    'comm_gb': vol / 1024 ** 3, 'comm_vs_ddp': vol / ar, 'how': how})
    return out


# ---------------------------------------------------------- tensor parallel
def mlp_block(x, w1, w2):
    return np.maximum(x @ w1, 0) @ w2


def tensor_parallel_mlp(x, w1, w2, p=4, wire=None, skip_all_reduce=False):
    """Megatron's pair: split w1 by columns, w2 by rows, add the pieces back.

    Column-split means each rank owns whole hidden units, so the elementwise
    non-linearity needs no communication.  Row-split means each rank produces a
    *partial* output of the full width, so one all-reduce closes the block.
    """
    w1s = np.array_split(w1, p, axis=1)
    w2s = np.array_split(w2, p, axis=0)
    partial = [mlp_block(x, w1s[r], w2s[r]) for r in range(p)]
    if skip_all_reduce:
        return partial[0]
    flat = [q.ravel().astype(np.float32) for q in partial]
    summed = ring_all_reduce(flat, wire)
    return summed[0].reshape(x.shape[0], w2.shape[1])


def tp_traffic(tokens, d_model=MODEL['d_model'], layers=MODEL['layers'], p=8,
               dtype='bf16', blocks_per_layer=2):
    """Bytes on the wire for tensor parallelism, per forward pass.

    Attention and MLP each end in one all-reduce, so a layer pays twice.
    The payload is the activation - it scales with tokens, not with weights.
    """
    act = tokens * d_model * BYTES[dtype]
    per_ar = 2 * (p - 1) / p * act
    return {'act_mb': act / 1024 ** 2, 'per_all_reduce_mb': per_ar / 1024 ** 2,
            'all_reduces': layers * blocks_per_layer,
            'total_gb': per_ar * layers * blocks_per_layer / 1024 ** 3,
            'us_per_all_reduce': time_us(2 * (p - 1), per_ar)}


def sequence_parallel_bytes(tokens, d_model=MODEL['d_model'], p=8, dtype='bf16'):
    """all-reduce == reduce-scatter + all-gather, so the volume is unchanged.

    What changes is that the norm and dropout between the two halves run on
    1/p of the sequence, so their activations are sharded too.
    """
    act = tokens * d_model * BYTES[dtype]
    rs = (p - 1) / p * act
    return {'all_reduce_mb': 2 * (p - 1) / p * act / 1024 ** 2,
            'rs_plus_ag_mb': 2 * rs / 1024 ** 2,
            'act_mem_per_rank_mb': act / p / 1024 ** 2,
            'act_mem_dp_mb': act / 1024 ** 2}


# ---------------------------------------------------------- expert parallel
def moe_reference(x, experts, route):
    out = np.zeros_like(x)
    for t, e in enumerate(route):
        out[t] = mlp_block(x[t:t + 1], *experts[e])[0]
    return out


def expert_parallel(x, experts, route, p=4, wire=None):
    """Two all-to-alls per layer: dispatch tokens to owners, combine results.

    The weights never move.  The tokens do - which is why EP's bill is set by
    how many tokens were routed off-rank, not by how big the experts are.
    """
    n_exp = len(experts)
    per_rank = n_exp // p
    owner = [e // per_rank for e in range(n_exp)]
    dispatched = {r: [] for r in range(p)}
    for t, e in enumerate(route):
        dispatched[owner[e]].append(t)
    if wire:
        moved = sum(len(v) for r, v in dispatched.items()) * x.shape[1] * 2
        wire.hop((p - 1) / p * moved)                 # all-to-all dispatch
    out = np.zeros_like(x)
    for r in range(p):
        for t in dispatched[r]:
            out[t] = mlp_block(x[t:t + 1], *experts[route[t]])[0]
    if wire:
        wire.hop((p - 1) / p * moved)                 # all-to-all combine
    return out


def ep_traffic(tokens, d_model=MODEL['d_model'], top_k=8, layers=MODEL['layers'],
               p=8, dtype='bf16'):
    payload = tokens * top_k * d_model * BYTES[dtype]
    one = (p - 1) / p * payload
    return {'dispatch_mb': one / 1024 ** 2, 'combine_mb': one / 1024 ** 2,
            'total_gb': 2 * one * layers / 1024 ** 3,
            'fp8_total_gb': 2 * one * layers / 2 / 1024 ** 3}


# ------------------------------------------- context parallel (split the KV)
def attention(q, k, v):
    s = q @ k.T / np.sqrt(q.shape[-1])
    s = s - s.max(axis=-1, keepdims=True)
    w = np.exp(s)
    return (w @ v) / w.sum(axis=-1, keepdims=True)


def context_parallel_attention(q, k, v, p=4, wire=None):
    """Each rank owns a slice of the keys, computes a partial attention and
    its log-sum-exp, and the ranks merge with the same identity Day 52 used
    for FlashAttention - only here the two halves live on different GPUs.
    """
    ks, vs = np.array_split(k, p), np.array_split(v, p)
    nums, maxes, lses = [], [], []
    for r in range(p):
        s = q @ ks[r].T / np.sqrt(q.shape[-1])
        m = s.max(axis=-1, keepdims=True)
        w = np.exp(s - m)
        nums.append(w @ vs[r])                              # local numerator
        maxes.append(m[:, 0])
        lses.append(m[:, 0] + np.log(w.sum(axis=-1)))       # local log-sum-exp
    lse, mx = np.array(lses), np.array(maxes)               # both (p, tokens)
    gmax = lse.max(axis=0)                                  # a shift that is safe
    num = sum(nums[r] * np.exp(mx[r] - gmax)[:, None] for r in range(p))
    den = np.exp(lse - gmax).sum(axis=0)
    if wire:
        item = 4
        payload = q.shape[0] * (q.shape[1] + 1) * item      # partial out + lse
        wire.hop((p - 1) / p * payload)
    return num / den[:, None]


def kv_cache_gb(seq=131072, batch=64, layers=MODEL['layers'],
                kv_heads=MODEL['kv_heads'], head_dim=MODEL['head_dim'], dtype='bf16'):
    per_token = 2 * layers * kv_heads * head_dim * BYTES[dtype]
    return batch * seq * per_token / 1024 ** 3


def kv_layouts(p=8, **kw):
    full = kv_cache_gb(**kw)
    return [('TP, MHA/GQA (KV split by head)', full / p, 'heads divide by p'),
            ('TP, MLA (one latent per token)', full, 'the latent cannot be split by head'),
            ('DP attention (requests split)', full / p, 'each rank owns whole requests'),
            ('context parallel (sequence split)', full / p, 'each rank owns a key range')]


# -------------------------------------------------------- pipeline parallel
def pipeline_bubble(stages=8, micro=(1, 2, 4, 8, 16, 32, 64)):
    """1F1B: the bubble is (stages-1) / (micro + stages - 1) of the step."""
    rows = []
    for m in micro:
        frac = (stages - 1) / (m + stages - 1)
        rows.append({'micro': m, 'bubble': frac, 'util': 1 - frac,
                     'slots': (m + stages - 1) * stages, 'useful': m * stages})
    return rows


def pipeline_traffic(tokens, d_model=MODEL['d_model'], stages=8, dtype='bf16'):
    """Only the boundary activation crosses, once per stage boundary."""
    act = tokens * d_model * BYTES[dtype]
    return {'per_boundary_mb': act / 1024 ** 2, 'boundaries': stages - 1,
            'total_mb': act * (stages - 1) / 1024 ** 2}


# ------------------------------------------------------- where each axis goes
def placement_table(tokens=8192, p=8, layers=MODEL['layers'], top_k=8,
                    in_node=(2.0, 400.0), cross_node=(10.0, 50.0)):
    """Bytes one rank ships per step on each axis, and what the link costs.

    The ordering of this table is the whole reason real deployments put
    tensor and expert parallelism inside a node and data and pipeline
    parallelism across nodes.
    """
    d = MODEL['d_model']
    act = tokens * d * BYTES['bf16']
    rows = [
        ('tensor parallel (TP)', 2 * (p - 1) / p * act * layers * 2, 2 * (p - 1) * layers * 2,
         'all-reduce after attention and after MLP, every layer'),
        ('expert parallel (EP)', 2 * (p - 1) / p * tokens * top_k * d * BYTES['bf16'] * layers,
         2 * (p - 1) * layers, 'all-to-all to dispatch tokens and to combine them'),
        ('context parallel (CP)', (p - 1) / p * tokens * MODEL['heads'] *
         (MODEL['head_dim'] + 1) * 4 * layers, (p - 1) * layers,
         'all-gather of partial outputs and their log-sum-exp'),
        ('pipeline parallel (PP)', act, 2,
         'one point-to-point handoff per stage boundary'),
        ('data parallel (DDP)', 2 * (p - 1) / p * weight_bytes(), 2 * (p - 1),
         'all-reduce of the gradients, once per step, not per layer'),
    ]
    out = []
    for name, nbytes, steps, how in rows:
        fast = time_us(steps, nbytes, *in_node) / 1e3
        slow = time_us(steps, nbytes, *cross_node) / 1e3
        out.append({'axis': name, 'gb': nbytes / 1024 ** 3, 'steps': steps,
                    'in_node_ms': fast, 'cross_node_ms': slow,
                    'penalty': slow / fast, 'how': how})
    return out


# ------------------------------------------------------- five silent failures
def forgot_to_average(steps=(1, 3, 5, 10), p=4, lr=0.5):
    """Summing gradients instead of averaging them multiplies the learning
    rate by the world size.  Nothing raises; the loss just leaves."""
    rows = []
    for n in steps:
        ok = train_ddp(p=p, steps=n, lr=lr, divide_by_world=True)[0]
        bad = train_ddp(p=p, steps=n, lr=lr, divide_by_world=False)[0]
        rows.append({'steps': n, 'averaged': ok, 'summed': bad})
    return rows


def mismatched_order(program_a=('grad', 'loss'), program_b=('loss', 'grad'), p=2):
    """Collectives are matched by call order, not by name.

    Two ranks that take different branches - 'this rank had no tokens for any
    expert, so skip the all-reduce' - meet at different collectives and wait
    for each other forever.  There is no error message; the job just stops.
    """
    pcs = [0] * p
    programs = [list(program_a if r % 2 == 0 else program_b) for r in range(p)]
    log = []
    for tick in range(10):
        at = [programs[r][pcs[r]] if pcs[r] < len(programs[r]) else None
              for r in range(p)]
        if all(a is None for a in at):
            return {'finished': True, 'ticks': tick, 'log': log}
        if len(set(at)) == 1:
            log.append((tick, at[0], 'all ranks agree, the collective runs'))
            pcs = [pc + 1 for pc in pcs]
        else:
            log.append((tick, tuple(at), 'ranks are at different collectives'))
            return {'finished': False, 'ticks': tick, 'log': log,
                    'waiting_on': tuple(at)}
    return {'finished': False, 'ticks': 10, 'log': log}


def reduction_order(n=1 << 16, p=8, seed=3):
    """The same sum, three schedules, three different last bits."""
    rng = np.random.default_rng(seed)
    scale = 10.0 ** rng.integers(-3, 4, size=(p, n))
    bufs = [(rng.standard_normal(n) * scale[r]).astype(np.float32) for r in range(p)]
    ring = ring_all_reduce(bufs)[0]
    tree = tree_all_reduce(bufs)[0]
    naive = np.zeros(n, dtype=np.float32)
    for b in bufs:
        naive = naive + b
    ring2 = ring_all_reduce(bufs)[0]
    return {'ring_vs_tree': float(np.abs(ring - tree).max()),
            'ring_vs_naive': float(np.abs(ring - naive).max()),
            'ring_vs_tree_rel': float(np.abs((ring - tree) / (np.abs(naive) + 1e-30)).max()),
            'differing': int((ring != tree).sum()), 'n': n,
            'ring_repeatable': bool((ring == ring2).all())}


def straggler(p=(8, 64, 512, 4096), jitter=0.05, hiccup_p=0.01, hiccup=0.5,
              trials=4000, seed=5):
    """A collective is a barrier, so every rank waits for the slowest one.

    Small symmetric jitter barely matters.  A rare hiccup does: if one rank in
    a hundred stalls on any given step, then at 512 ranks almost every step
    has a stalled rank in it, and the whole job runs at the stalled speed.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for n in p:
        t = 1.0 + rng.uniform(0, jitter, size=(trials, n))
        t = t + hiccup * (rng.random((trials, n)) < hiccup_p)
        step = t.max(axis=1)
        rows.append({'ranks': n, 'mean_rank_ms': float(t.mean()),
                     'mean_step_ms': float(step.mean()),
                     'any_hiccup_pct': float(100 * (1 - (1 - hiccup_p) ** n)),
                     'loss_pct': float((step.mean() / t.mean() - 1) * 100)})
    return rows


def missing_all_reduce(tokens=8, d=32, h=64, p=4, seed=0):
    """Dropping the all-reduce in a row-parallel layer does not crash."""
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((tokens, d)).astype(np.float32)
    w1 = (rng.standard_normal((d, h)) / 5).astype(np.float32)
    w2 = (rng.standard_normal((h, d)) / 8).astype(np.float32)
    ref = mlp_block(x, w1, w2)
    good = tensor_parallel_mlp(x, w1, w2, p)
    bad = tensor_parallel_mlp(x, w1, w2, p, skip_all_reduce=True)
    rel = np.abs(bad - ref).max() / np.abs(ref).max()
    return {'max_abs_correct': float(np.abs(good - ref).max()),
            'max_abs_dropped': float(np.abs(bad - ref).max()),
            'rel_dropped': float(rel), 'shape_ok': bad.shape == ref.shape}



# ------------------------------------------- what a training run looks like
def training_mesh(gpus=64, tp=8, pp=4, tokens=8192, layers=MODEL['layers'],
                  params=MODEL['params'], dtype='bf16'):
    """A real run stacks the axes: one mesh dimension per kind of parallelism.

    Tensor and pipeline parallelism already shard the parameters, so what is
    left for ZeRO to shard is only what the data-parallel axis replicates.
    """
    dp = gpus // (tp * pp)
    d = MODEL['d_model']
    act = tokens * d * BYTES[dtype]
    per_stage = layers // pp
    params_per_gpu = params / (tp * pp)
    adam = sum(optimizer_state_bytes().values())
    return {
        'gpus': gpus, 'tp': tp, 'pp': pp, 'dp': dp,
        'layers_per_stage': per_stage,
        'params_per_gpu_b': params_per_gpu / 1e9,
        'mem_ddp_gb': params_per_gpu * adam / 1024 ** 3,
        'mem_zero3_gb': params_per_gpu * adam / dp / 1024 ** 3,
        'tp_gb': 2 * (tp - 1) / tp * act * per_stage * 2 / 1024 ** 3,
        'pp_gb': act / 1024 ** 3,
        'dp_gb': 2 * (dp - 1) / dp * params_per_gpu * BYTES[dtype] / 1024 ** 3,
    }


# ------------------------------------------- what changes at inference time
def decode_tp_cost(requests=64, p=8, layers=MODEL['layers'], dtype='bf16',
                   alpha_us=2.0, beta_gbs=400.0):
    """The same tensor-parallel all-reduce, in decode and in prefill.

    Decode produces one token per request, so the payload collapses and what
    is left is the hop count - the collective becomes a latency problem.
    """
    rows = []
    for name, tokens in (('decode (1 token x %d requests)' % requests, requests),
                         ('prefill (8,192 tokens)', 8192)):
        act = tokens * MODEL['d_model'] * BYTES[dtype]
        per = 2 * (p - 1) / p * act
        lat = 2 * (p - 1) * alpha_us
        bw = per / (beta_gbs * 1e3)
        rows.append({'phase': name, 'mb': per / 1024 ** 2, 'latency_us': lat,
                     'bandwidth_us': bw, 'total_us': lat + bw,
                     'latency_share': lat / (lat + bw),
                     'per_step_ms': (lat + bw) * layers * 2 / 1e3})
    return rows


def dp_attention(requests=64, p=8, **kw):
    """Serving replicas: no gradient, so data parallelism costs only memory.

    Under plain tensor parallelism with MLA every rank keeps the whole KV
    cache; under DP attention each rank owns whole requests instead.
    """
    full = kv_cache_gb(batch=requests, **kw)
    return {'kv_total_gb': full, 'tp_mla_per_rank_gb': full,
            'dp_attention_per_rank_gb': full / p,
            'requests_per_rank': requests // p,
            'collective_per_step': 'none for the gradients (there are none)'}

# ----------------------------------------------------------------- printing
def head(n, title):
    print('\n' + '=' * 72)
    print(f'{n}. {title}')
    print('=' * 72 + '\n')


def main():
    print('Day 63 - parallelism and collective communication')
    print('Splitting a model across GPUs is choosing which collective runs,')
    print('how often, and on how many bytes.  Everything below is that sentence.')

    # ------------------------------------------------------------ 1
    head(1, 'The short version')
    z = zero_table(p=8)
    tp = tp_traffic(8192, p=8)
    print(f"  {MODEL['name']} with Adam needs {z[0]['gb_per_rank']:,.0f} GB per GPU if")
    print(f"  every GPU keeps a full replica.  Shard the optimizer, the gradients")
    print(f"  and the parameters across 8 GPUs and that becomes "
          f"{z[3]['gb_per_rank']:,.0f} GB - at the")
    print(f"  cost of {z[3]['comm_vs_ddp']:.1f}x the bytes on the wire.\n")
    print(f"  Serving the same model 8-way tensor parallel moves "
          f"{tp['total_gb']:.0f} GB per forward")
    print(f"  pass of {8192:,} tokens: {tp['all_reduces']} all-reduces of "
          f"{tp['per_all_reduce_mb']:.0f} MB each, one after")
    print('  attention and one after the MLP in all 80 layers.\n')
    print('  Those two numbers - how much memory a split saves and how many bytes')
    print('  it costs - are what the rest of this file computes, for every kind of')
    print('  parallelism, from five primitives that we build first.\n')
    print('  The file is in two halves, because the two jobs are genuinely')
    print('  different.  Training has gradients and optimizer state, so its')
    print('  problem is that the *weights* do not fit: DDP, ZeRO/FSDP, tensor and')
    print('  pipeline parallelism.  Inference has neither, so its problems are')
    print('  that the *KV cache* does not fit and that every collective happens on')
    print('  a latency budget: DP attention, expert parallelism, context')
    print('  parallelism.  The collectives underneath are the same five.')

    # ------------------------------------------------------------ 2
    head(2, 'The five collectives everything is made of')
    p = 8
    for r in collective_costs(p=p, n_bytes=256 * 1024 ** 2):
        print(f"    {r['name']:16} {r['bytes_formula']:>14}  "
              f"{r['steps_formula']:>9} steps   {r['mb_per_rank']:7.1f} MB/rank")
    print(f'\n  (N is the size of the tensor, p = {p} ranks, and the formulas are for')
    print('  the ring or tree implementation every library actually uses.)\n')
    rng = np.random.default_rng(0)
    bufs = [rng.standard_normal(1 << 14).astype(np.float32) for _ in range(p)]
    ref = np.sum(bufs, axis=0)
    wire = Wire()
    out = ring_all_reduce(bufs, wire)
    n_bytes = bufs[0].nbytes
    print('  A ring all-reduce, run here on 8 simulated ranks:')
    print(f'    every rank ends with the same vector   : '
          f'{all(np.allclose(o, ref, atol=1e-4) for o in out)}')
    print(f'    steps                                  : {wire.steps} '
          f'(= 2(p-1), {p - 1} to reduce-scatter and {p - 1} to all-gather)')
    print(f'    bytes each rank shipped                : {wire.bytes_per_rank:,} '
          f'(= 2(p-1)/p x {n_bytes:,})')
    print(f'    the naive version - everyone sends to rank 0 and rank 0 sends back -')
    print(f'    would ship {2 * n_bytes * (p - 1):,} bytes through one link instead.\n')
    print('  The ring is the reason all-reduce costs about 2N and not pN: no rank')
    print('  ever holds more than one chunk of the sum at a time, so the volume')
    print('  stops growing with the world size and only the step count does.')

    # ------------------------------------------------------------ 3
    head(3, 'Ring or tree: who wins depends on the message size')
    print('    bytes        ring (2(p-1) steps)   tree (2log2(p) steps)   winner')
    for nb in (4 * 1024, 64 * 1024, 1024 * 1024, 64 * 1024 ** 2, 1024 ** 3):
        ring, tree = ring_us(nb, p), tree_us(nb, p)
        win = 'ring' if ring < tree else 'tree'
        print(f'    {nb:12,}   {ring:12.1f} us     {tree:14.1f} us      {win}')
    cross = next(nb for nb in (1 << i for i in range(10, 31))
                 if ring_us(nb, p) < tree_us(nb, p))
    print(f'\n  The crossover on this link is around {cross / 1024:,.0f} KB.  Below it the')
    print('  message is latency-bound and all that matters is the number of hops:')
    print('  a tree needs 2log2(p) of them, the ring needs 2(p-1).  Above it the')
    print('  message is bandwidth-bound, and the ring wins because every rank only')
    print('  ever ships 1/p of the tensor per step while the tree pushes the whole')
    print('  thing up every level.  Libraries pick by size at runtime - which is')
    print('  also why a small all-reduce inside a decode step is not free: it costs')
    print('  hops even when it costs almost no bytes.')

    print('\n' + '#' * 72)
    print('# PART ONE - TRAINING: the model does not fit, and the gradients have to agree')
    print('#' * 72)

    # ------------------------------------------------------------ 4
    head(4, 'DDP: the same model on every GPU, a different slice of the batch')
    single_loss, single_params = train_single()
    ddp_wire = Wire()
    ddp_loss, ddp_params = train_ddp(p=4, wire=ddp_wire)
    diff = max(float(np.abs(a - b).max()) for a, b in zip(single_params, ddp_params))
    print('  DDP - torch.nn.parallel.DistributedDataParallel - is the simplest')
    print('  thing that works: every GPU keeps a *complete* copy of the model and')
    print('  reads a different slice of the batch.  One process per GPU, one')
    print('  gradient all-reduce per step, and nothing else is shared.\n')
    print('  Trained the same tiny network twice, 60 steps:')
    print(f'    one device, full batch         : loss {single_loss:.6f}')
    print(f'    4 ranks, quarter batch each    : loss {ddp_loss:.6f}')
    print(f'    largest weight difference      : {diff:.2e}\n')
    print('  They agree because the loss is a mean: the average of the per-rank')
    print('  gradients is exactly the full-batch gradient.  That identity is the')
    print('  whole of data parallelism, and the all-reduce is what enforces it.')
    print('  Nothing else about the model is split, so the memory per GPU does')
    print('  not move - which is why DDP alone stops working above a few billion')
    print('  parameters.\n')
    print('  Two DDP details are worth knowing, because they are how that')
    print('  all-reduce stops being visible.  The first is bucketing: gradients')
    print('  become ready back to front, so the all-reduce is cut into chunks that')
    print('  fly while the backward pass is still running:')
    print('    buckets   bucket size   all-reduce each   total comm   exposed')
    for r in ddp_overlap():
        print(f"    {r['buckets']:7}   {r['bucket_mb']:8.0f} MB   "
              f"{r['comm_ms_each']:12.2f} ms   {r['all_comm_ms']:8.2f} ms   "
              f"{r['exposed_ms']:5.2f} ms")
    print('\n  More buckets means slightly more total traffic (each one pays the')
    print('  latency again) but much less of it exposed at the end of the step.\n')
    print('  The second is gradient accumulation.  Inside `with model.no_sync():`')
    print('  DDP skips the all-reduce entirely and lets gradients pile up locally,')
    print('  so N micro-steps cost one all-reduce instead of N.  That is how a')
    print('  large effective batch is reached on few GPUs - and it is the one')
    print('  place where skipping a collective is safe, because *every* rank skips')
    print('  it, which is exactly the condition that failure 2 below breaks.')

    # ------------------------------------------------------------ 5
    head(5, 'ZeRO and FSDP: stop replicating what you can shard')
    s = optimizer_state_bytes()
    print('  Adam in mixed precision keeps, per parameter:')
    for k, v in s.items():
        print(f'    {k:22} {v} bytes')
    print(f"    total                  {sum(s.values())} bytes - "
          f"{sum(s.values()) / 2:.0f}x the weights themselves\n")
    print(f"  For {MODEL['name']} ({MODEL['params'] / 1e9:.0f}B parameters) on 8 GPUs:\n")
    print('    stage                           GB/rank   wire GB   vs DDP   collectives')
    for r in zero_table(p=8):
        print(f"    {r['stage']:30} {r['gb_per_rank']:7.0f}   {r['comm_gb']:7.1f}   "
              f"{r['comm_vs_ddp']:5.2f}x   {r['how']}")
    print('\n  FSDP is ZeRO-3: the parameters themselves are sharded, so each layer')
    print('  is all-gathered just before it runs and thrown away right after.')
    print('  That is 1.5x the communication of DDP for 1/p of the memory, and it is')
    print('  the same trade the whole series keeps making - Day 62.5 called it')
    print('  recompute versus store, here it is refetch versus store.')

    # ------------------------------------------------------------ 6
    head(6, 'Tensor parallelism: one layer, split across GPUs')
    rng = np.random.default_rng(1)
    x = rng.standard_normal((64, 256)).astype(np.float32)
    w1 = (rng.standard_normal((256, 1024)) / 16).astype(np.float32)
    w2 = (rng.standard_normal((1024, 256)) / 32).astype(np.float32)
    ref_y = mlp_block(x, w1, w2)
    tp_wire = Wire()
    tp_y = tensor_parallel_mlp(x, w1, w2, p=4, wire=tp_wire)
    print('  Column-split the first matrix, row-split the second, add the pieces:')
    print(f'    4-way split matches one device   : '
          f'{np.abs(ref_y - tp_y).max():.2e} max difference')
    print(f'    collectives per block            : 1 all-reduce forward, 1 backward\n')
    print('  The split is chosen so the non-linearity needs no communication: each')
    print('  rank owns whole hidden units, so ReLU or SwiGLU is local.  Only after')
    print('  the second matrix is each rank holding a *partial* sum of the full')
    print('  width, and that is what the all-reduce fixes.\n')
    t = tp_traffic(8192, p=8)
    print(f"  At 8,192 tokens and 8-way TP: activation {t['act_mb']:.0f} MB, "
          f"{t['per_all_reduce_mb']:.0f} MB per")
    print(f"  all-reduce, {t['all_reduces']} of them per forward pass = "
          f"{t['total_gb']:.0f} GB per rank per pass.")
    print('  Notice what that scales with: tokens, not parameters - the opposite')
    print('  of DDP, whose bill is the weights and ignores the batch.  That is why')
    print('  this is the axis that lets a long sequence train at all, and why it')
    print('  has to stay inside a node.  What the same all-reduce costs when there')
    print('  is only one token per request is section 9.\n')
    sp = sequence_parallel_bytes(8192, p=8)
    print('  Sequence parallelism splits the leftover norms and residuals too:')
    print(f"    one all-reduce                    : {sp['all_reduce_mb']:.0f} MB")
    print(f"    reduce-scatter + all-gather       : {sp['rs_plus_ag_mb']:.0f} MB "
          f"(identical - an all-reduce *is* those two)")
    print(f"    activation kept per rank          : "
          f"{sp['act_mem_per_rank_mb']:.0f} MB instead of {sp['act_mem_dp_mb']:.0f} MB")
    print('  Same bytes, same math, 1/p of the activation memory: the free half of')
    print('  the trade is real, and it comes from noticing that an all-reduce was')
    print('  two collectives glued together and the middle was worth keeping.')

    # ------------------------------------------------------------ 7
    head(7, 'Pipeline parallelism: cheap on the wire, expensive in idle time')
    pt = pipeline_traffic(8192, stages=8)
    print(f"  Only the boundary activation crosses: {pt['per_boundary_mb']:.0f} MB x "
          f"{pt['boundaries']} boundaries")
    print(f"  = {pt['total_mb']:.0f} MB per forward pass, point to point, no collective.")
    print('  That is three orders of magnitude less than tensor parallelism moves,')
    print('  which is why pipeline stages can sit on different nodes.\n')
    print('  The cost is the bubble: stage 8 has nothing to do until the first')
    print('  micro-batch reaches it.')
    print('    micro-batches   bubble   utilisation')
    for r in pipeline_bubble(stages=8):
        print(f"    {r['micro']:13}   {r['bubble']:6.1%}   {r['util']:11.1%}")
    print('\n  bubble = (stages - 1) / (micro-batches + stages - 1), so the fix is')
    print('  always more micro-batches - until they get small enough that the')
    print('  matmuls stop filling the GPU, which is the Day 62.5 roofline again.')

    # ------------------------------------------------------------ 8
    head(8, 'A real training run stacks all four at once')
    tm = training_mesh(gpus=64, tp=8, pp=4)
    print(f"  {tm['gpus']} GPUs as a mesh: tensor {tm['tp']} x pipeline {tm['pp']} x "
          f"data {tm['dp']}, so each GPU")
    print(f"  holds {tm['layers_per_stage']} of the 80 layers and "
          f"{tm['params_per_gpu_b']:.2f} B of the {MODEL['params'] / 1e9:.0f} B parameters.\n")
    print(f"    optimizer state per GPU, no sharding : {tm['mem_ddp_gb']:.1f} GB")
    print(f"    the same with ZeRO-3 on the dp axis  : {tm['mem_zero3_gb']:.1f} GB")
    print(f"    tensor-parallel traffic per step     : {tm['tp_gb']:.2f} GB  (inside the node)")
    print(f"    pipeline hand-offs per step          : {tm['pp_gb']:.2f} GB  (point to point)")
    print(f"    gradient all-reduce per step         : {tm['dp_gb']:.2f} GB  (once, overlapped)\n")
    print('  The ordering is the lesson.  Tensor and pipeline parallelism already')
    print('  shard the parameters, so the 1,043 GB from section 5 is down to')
    print(f"  {tm['mem_ddp_gb']:.1f} GB before ZeRO does anything at all - ZeRO only shards what the")
    print('  data-parallel axis is still replicating.  And the three traffic rows')
    print('  are three different links: tensor parallelism on the fastest one,')
    print('  pipeline on the slowest, data parallelism anywhere because it is')
    print('  paid once per step rather than once per layer.')

    print('\n' + '#' * 72)
    print('# PART TWO - INFERENCE: no gradients, tiny activations, enormous caches')
    print('#' * 72)

    # ------------------------------------------------------------ 9
    head(9, 'What inference changes: no gradients, one token per request')
    da = dp_attention(requests=64, p=8)
    print('  Two things disappear and one thing takes over.\n')
    print('  First, there are no gradients, so the largest collective in the whole')
    print('  training half - the gradient all-reduce - is simply not called.  Data')
    print('  parallelism in a serving stack means independent replicas with their')
    print('  own schedulers, and its only cost is memory:\n')
    print(f"    KV cache for {64} requests x 128K tokens  : "
          f"{da['kv_total_gb']:,.0f} GB")
    print(f"    tensor parallel with MLA, per rank      : "
          f"{da['tp_mla_per_rank_gb']:,.0f} GB  (a full copy on every rank)")
    print(f"    DP attention, per rank                  : "
          f"{da['dp_attention_per_rank_gb']:,.0f} GB  "
          f"({da['requests_per_rank']} whole requests each)\n")
    print('  Second, decode produces one token per request, so the activation that')
    print('  tensor parallelism was shipping collapses - and what is left is the')
    print('  hop count:\n')
    print('    phase                            payload    latency   bandwidth   total')
    for r in decode_tp_cost(requests=64, p=8):
        print(f"    {r['phase']:30} {r['mb']:7.2f} MB  {r['latency_us']:6.1f} us  "
              f"{r['bandwidth_us']:7.1f} us  {r['total_us']:7.1f} us")
    d, pre = decode_tp_cost(requests=64, p=8)
    print(f"\n  In prefill the all-reduce is {pre['latency_share']:.0%} latency and the rest is")
    print(f"  bandwidth; in decode it is {d['latency_share']:.0%} latency.  Same collective, same")
    print(f"  2(p-1) hops, {pre['mb'] / d['mb']:.0f}x less data.  Across 160 of them that is still")
    print(f"  {d['per_step_ms']:.2f} ms per decode step, which is why serving stacks fuse the")
    print('  all-reduce into the next norm, prefer fewer and fatter collectives,')
    print('  and why tensor parallelism stops helping decode long before it runs')
    print('  out of bandwidth.  Day 62.5 said decode is memory-bound; this is the')
    print('  same sentence with the interconnect in place of HBM.')

    # ------------------------------------------------------------ 10
    head(10, 'Expert parallelism: the tokens move, the weights stay')
    rng = np.random.default_rng(2)
    xm = rng.standard_normal((64, 128)).astype(np.float32)
    experts = [((rng.standard_normal((128, 256)) / 11).astype(np.float32),
                (rng.standard_normal((256, 128)) / 16).astype(np.float32))
               for _ in range(8)]
    route = rng.integers(0, 8, size=64)
    ep_wire = Wire()
    ep_y = expert_parallel(xm, experts, route, p=4, wire=ep_wire)
    print(f'  8 experts on 4 ranks, 64 tokens routed by the Day 62 router:')
    print(f'    matches the single-device MoE    : '
          f'{np.abs(moe_reference(xm, experts, route) - ep_y).max():.2e}')
    print(f'    collectives per layer            : 2 all-to-all (dispatch, combine)\n')
    e = ep_traffic(8192, p=8)
    print(f"  At 8,192 tokens, top-8 routing, 8-way EP: "
          f"{e['dispatch_mb']:.0f} MB out and {e['combine_mb']:.0f} MB")
    print(f"  back per layer, {e['total_gb']:.0f} GB per forward pass - "
          f"{e['total_gb'] / t['total_gb']:.0f}x what tensor parallelism")
    print(f"  pays for the same tokens, because every token is sent to k experts.")
    print(f"  Dispatching in fp8 halves it to {e['fp8_total_gb']:.0f} GB, which is "
          f"exactly why DeepEP")
    print('  ships a quantized dispatch path and why MoE serving lives or dies on')
    print('  the interconnect rather than on the GPUs.\n')
    print('  Training a MoE uses the same two all-to-alls and two more in the')
    print('  backward pass, so the expert axis belongs inside a node for exactly')
    print('  the reason section 12 gives.')

    # ------------------------------------------------------------ 11
    head(11, 'Context parallelism (DCP): split the keys, merge with the log-sum-exp')
    rng = np.random.default_rng(3)
    q = rng.standard_normal((32, 128)).astype(np.float32)
    k = rng.standard_normal((8192, 128)).astype(np.float32)
    v = rng.standard_normal((8192, 128)).astype(np.float32)
    cp_wire = Wire()
    cp_y = context_parallel_attention(q, k, v, p=4, wire=cp_wire)
    print(f'  4 ranks, each holding a quarter of the keys and values:')
    print(f'    matches full attention           : '
          f'{np.abs(attention(q, k, v) - cp_y).max():.2e}')
    print('    what crosses the wire            : the partial output and one')
    print('                                       log-sum-exp per query token\n')
    print('  This is the Day 52 identity again - two partial softmaxes combine if')
    print('  you carry their running maximum and sum - except the two halves are')
    print('  now on different GPUs instead of in different registers.  The same')
    print('  algebra that made FlashAttention possible is what makes a million-')
    print('  token context splittable.\n')
    print('  Who needs this, and who does not.  Splitting the keys works for any')
    print('  attention - that is Ring Attention - but it is only the cheapest')
    print('  option when there is nothing else left to split.  With MHA or GQA the')
    print('  KV cache divides neatly by head, so tensor parallelism already shards')
    print('  it for free and nobody reaches for context parallelism.  With MLA')
    print('  there is one latent per token and no head dimension at all, so every')
    print('  tensor-parallel rank keeps a full copy - and that is the case the')
    print('  serving feature called DCP exists for.  Read DCP as "for the MLA')
    print('  models", not as a general-purpose axis.\n')
    print('  Why anybody bothers: the KV cache, at 64 requests x 128K tokens,')
    print(f'  is {kv_cache_gb():,.0f} GB, which no single GPU has.')
    print('    layout                             KV GB per rank   why')
    for name, gb, why in kv_layouts(p=8):
        print(f'    {name:34} {gb:9,.0f}   {why}')
    print('\n  The MLA row is the one that surprises people, and it is the whole')
    print('  reason this axis exists in a serving stack: compressing the KV into')
    print('  one latent per token removes the head dimension that tensor')
    print('  parallelism was splitting along, so every rank keeps a full copy and')
    print('  the only ways left to shard it are by request (section 9) or by')
    print('  position (this one).')

    print('\n' + '#' * 72)
    print('# BOTH HALVES AT ONCE')
    print('#' * 72)

    # ------------------------------------------------------------ 12
    head(12, 'Putting them together: which axis goes where')
    print('  One rank, one step, 8,192 tokens, 8-way on each axis:\n')
    print('    axis                     GB     steps   in-node   cross-node  penalty')
    for r in placement_table():
        print(f"    {r['axis']:22} {r['gb']:7.2f}   {r['steps']:5}   "
              f"{r['in_node_ms']:7.1f}ms   {r['cross_node_ms']:8.1f}ms   "
              f"{r['penalty']:5.1f}x")
    print('\n  Read the GB column and the deployment writes itself: tensor and')
    print('  expert parallelism move the most and must stay inside a node, on the')
    print('  fast links; pipeline parallelism moves almost nothing and is happy')
    print('  crossing the slow network; data parallelism moves a lot but only once')
    print('  per step, so it can be overlapped with the backward pass and also')
    print('  survives the slow link.  Real clusters stack all of them at once -')
    print('  a mesh with one axis per kind - and the mesh shape is chosen exactly')
    print('  by this table.')

    # ------------------------------------------------------------ 13
    head(13, 'Five ways this goes wrong without an error message')
    print('  1. Summing the gradients instead of averaging them.')
    print('     The learning rate is silently multiplied by the world size:')
    print('       steps   averaged     summed')
    for r in forgot_to_average(steps=(1, 3, 5)):
        print(f"       {r['steps']:5}   {r['averaged']:8.4f}   {r['summed']:,.1f}")
    bad10 = train_ddp(p=4, steps=10, divide_by_world=False)[0]
    print(f'     By step 10 the loss is {bad10:.3g}; nothing raised.\n')
    m = mismatched_order()
    print('  2. Ranks that take different branches meet different collectives.')
    print(f"     rank 0 is at '{m['waiting_on'][0]}', rank 1 is at "
          f"'{m['waiting_on'][1]}' - both wait forever.")
    print('     Collectives are matched by call order, not by name, so "skip the')
    print('     all-reduce when this rank has no tokens" is a hang, not a speedup.\n')
    ro = reduction_order()
    print('  3. The reduction order is part of the answer.')
    print(f"     ring vs tree all-reduce of the same data: "
          f"{ro['differing']:,} of {ro['n']:,} values differ,")
    print(f"     max absolute difference {ro['ring_vs_tree']:.2e} "
          f"(relative {ro['ring_vs_tree_rel']:.1e}).")
    print(f"     Re-running the ring gives identical bits ({ro['ring_repeatable']}), so this")
    print('     is not randomness - it is that float addition is not associative')
    print('     and the schedule chose a different order.  Same seed, different')
    print('     world size, different loss curve.\n')
    print('  4. One slow rank is the step time, because a collective is a barrier.')
    print('       ranks   any rank hiccuped   mean step   slower than the mean rank')
    for r in straggler():
        print(f"       {r['ranks']:5}   {r['any_hiccup_pct']:16.1f}%   "
              f"{r['mean_step_ms']:8.3f}   {r['loss_pct']:8.1f}%")
    print('     A 1% chance of a hiccup per rank is a 99% chance per step at 512')
    print('     ranks.  Scaling out multiplies the number of chances, not the')
    print('     size of each one.\n')
    ma = missing_all_reduce()
    print('  5. Dropping the all-reduce in a row-parallel layer.')
    print(f"     shape still correct              : {ma['shape_ok']}")
    print(f"     max error with the all-reduce    : {ma['max_abs_correct']:.2e}")
    print(f"     max error without it             : {ma['max_abs_dropped']:.2f} "
          f"({ma['rel_dropped']:.0%} of the output scale)")
    print('     The model keeps generating text; it just generates worse text.')
    print('     Every parallelism bug in this list looks like a quality problem,')
    print('     not a crash, which is why the first test for any of them is')
    print('     "does one rank still match the single-GPU answer".')

    # ------------------------------------------------------------ 14
    head(14, 'In PyTorch, briefly')
    print('    from torch.distributed.device_mesh import init_device_mesh')
    print('    mesh = init_device_mesh("cuda", (2, 4), mesh_dim_names=("dp", "tp"))\n')
    print('  One mesh, one axis per kind of parallelism, and every wrapper takes')
    print('  the sub-mesh it belongs on:\n')
    print('    DistributedDataParallel(model)            # all-reduce(grad), bucketed')
    print('    fully_shard(model, mesh=mesh["dp"])       # FSDP2 = ZeRO-3')
    print('    parallelize_module(block, mesh["tp"], {')
    print('        "w1": ColwiseParallel(), "w2": RowwiseParallel()})\n')
    print('  ColwiseParallel and RowwiseParallel are literally section 6: the')
    print('  first marks the output as sharded, the second marks it as partial,')
    print('  and DTensor inserts the all-reduce when a partial tensor is asked to')
    print('  become replicated.  The useful mental model is that you annotate')
    print('  placements and the framework derives the collectives - so when you')
    print('  see an unexpected all-reduce in the profile, look for a placement')
    print('  that forced it.\n')
    print('    dist.all_reduce / reduce_scatter_tensor / all_gather_into_tensor')
    print('    dist.all_to_all_single(out, inp, out_splits, in_splits)   # MoE')
    print('    with model.no_sync():    # skip the all-reduce while accumulating')
    print('  and torch.profiler shows each one as an nccl kernel, so the bytes')
    print('  from section 10 are directly checkable against a trace.')

    # ------------------------------------------------------------ 15
    head(15, 'In JAX, briefly')
    print('    mesh = jax.make_mesh((2, 4), ("dp", "tp"))')
    print('    x = jax.device_put(x, NamedSharding(mesh, P("dp", None)))')
    print('    w = jax.device_put(w, NamedSharding(mesh, P(None, "tp")))\n')
    print('  You do not call a collective at all: you say where the arrays live,')
    print('  and XLA inserts the all-reduce, all-gather or all-to-all needed to')
    print('  make the program type-check across devices.  Data parallelism is a')
    print('  sharded batch axis; tensor parallelism is a sharded feature axis;')
    print('  FSDP is a sharded parameter axis plus jax.remat - the same three')
    print('  decisions, expressed as annotations instead of wrappers.\n')
    print('    shard_map(f, mesh, in_specs=..., out_specs=...)   # write it by hand')
    print('    jax.lax.psum(x, "tp")        # this is the all-reduce')
    print('    jax.lax.all_to_all(x, "ep", 0, 0)')
    print('    print(jax.jit(f).lower(x).compile().as_text())    # see the inserted ops\n')
    print('  The last line is the one worth keeping: the compiled HLO names every')
    print('  collective it added, so a parallelism review in JAX is reading that')
    print('  text, while in PyTorch it is reading a profiler trace.')

    # ------------------------------------------------------------ 16
    head(16, 'In the serving stacks, briefly')
    print('    --tensor-parallel-size 8        # section 6, inside one node')
    print('    --expert-parallel-size 8        # section 7, for MoE layers')
    print('    --data-parallel-size 2          # whole replicas, separate schedulers')
    print('    --pipeline-parallel-size 2      # section 9, across nodes')
    print('    --context-parallel-size 2       # section 8, long context\n')
    print('  Inference changes the arithmetic in two ways.  There is no gradient,')
    print('  so the DDP all-reduce disappears entirely and data parallelism costs')
    print('  nothing but memory.  And the activations are tiny during decode - one')
    print('  token per request - so tensor parallelism that looked expensive in')
    print('  section 6 becomes latency-dominated instead: the same 2(p-1) hops,')
    print('  almost no bytes.  That is why decode TP is limited by how fast the')
    print('  links can turn around a small message, and why the stacks fuse the')
    print('  all-reduce with the next norm when they can.\n')
    print('  The MoE path is the mirror image: all-to-all volume scales with')
    print('  tokens x top-k, so a serving stack adds a dispatch in fp8 and tries')
    print('  to overlap it with the expert matmuls - which is the Day 62 DeepEP')
    print('  story told from the collective side.\n')
    print('  "DP attention" is the layout from section 8: with MLA the KV cache')
    print('  cannot be split by head, so each rank serves whole requests and keeps')
    print('  their KV, instead of all ranks keeping one copy of everything.')

    # ------------------------------------------------------------ 17
    head(17, 'Assertions')
    check()
    print('all assertions passed')
def check():
    rng = np.random.default_rng(7)
    p = 8
    bufs = [rng.standard_normal(4096).astype(np.float32) for _ in range(p)]
    ref = np.sum(bufs, axis=0)

    # the ring really all-reduces, and the bytes follow the formula
    w = Wire()
    out = ring_all_reduce(bufs, w)
    assert all(np.allclose(o, ref, atol=1e-4) for o in out)
    assert w.steps == 2 * (p - 1)
    assert abs(w.bytes_per_rank - 2 * (p - 1) / p * bufs[0].nbytes) < 1e-6

    # small messages favour the tree, large ones the ring
    assert tree_us(4 * 1024, p) < ring_us(4 * 1024, p)
    assert ring_us(1024 ** 3, p) < tree_us(1024 ** 3, p)

    # the tree agrees with the ring up to float ordering, in fewer steps
    w2 = Wire()
    tre = tree_all_reduce(bufs, w2)
    assert all(np.allclose(o, ref, atol=1e-4) for o in tre)
    assert w2.steps == 2 * np.log2(p)

    # reduce-scatter + all-gather == all-reduce
    owned = ring_reduce_scatter(bufs)
    gathered = ring_all_gather(owned)
    assert all(np.allclose(g, ref, atol=1e-4) for g in gathered)

    # all-to-all is a transpose: doing it twice is the identity
    mats = [rng.standard_normal(64).astype(np.float32) for _ in range(p)]
    assert all(np.allclose(a, b) for a, b in zip(all_to_all(all_to_all(mats)), mats))

    # broadcast reaches every rank in log2(p) steps
    wb = Wire()
    bc = broadcast(bufs[0], p, wb)
    assert all((x == bufs[0]).all() for x in bc) and wb.steps == np.log2(p)

    # data parallel with the average == one device with the full batch
    s_loss, s_par = train_single(steps=30)
    d_loss, d_par = train_ddp(p=4, steps=30)
    assert abs(s_loss - d_loss) < 1e-5
    assert max(float(np.abs(a - b).max()) for a, b in zip(s_par, d_par)) < 1e-4
    # and without the average it is a different (much worse) model
    assert train_ddp(p=4, steps=10, divide_by_world=False)[0] > 10 * s_loss

    # FSDP: 1/p of the memory for 1.5x the volume
    z = zero_table(p=8)
    assert z[3]['gb_per_rank'] < z[0]['gb_per_rank'] / 7
    assert abs(z[3]['comm_vs_ddp'] - 1.5) < 1e-9
    assert abs(z[1]['comm_vs_ddp'] - 1.0) < 1e-9

    # tensor parallelism reproduces the single-device layer, and only with the
    # all-reduce
    x = rng.standard_normal((32, 128)).astype(np.float32)
    w1 = (rng.standard_normal((128, 512)) / 11).astype(np.float32)
    w2 = (rng.standard_normal((512, 128)) / 22).astype(np.float32)
    y = mlp_block(x, w1, w2)
    assert np.abs(tensor_parallel_mlp(x, w1, w2, 4) - y).max() < 1e-4
    assert np.abs(tensor_parallel_mlp(x, w1, w2, 4, skip_all_reduce=True) - y).max() > 0.1

    # an all-reduce is a reduce-scatter plus an all-gather, so sequence
    # parallelism moves exactly the same bytes
    sp = sequence_parallel_bytes(4096, p=8)
    assert abs(sp['all_reduce_mb'] - sp['rs_plus_ag_mb']) < 1e-9
    assert abs(sp['act_mem_per_rank_mb'] * 8 - sp['act_mem_dp_mb']) < 1e-9

    # expert parallelism moves tokens, not weights, and changes no answer
    xm = rng.standard_normal((32, 64)).astype(np.float32)
    experts = [((rng.standard_normal((64, 128)) / 8).astype(np.float32),
                (rng.standard_normal((128, 64)) / 11).astype(np.float32))
               for _ in range(8)]
    route = rng.integers(0, 8, size=32)
    assert np.abs(expert_parallel(xm, experts, route, 4)
                  - moe_reference(xm, experts, route)).max() == 0.0

    # context parallelism reproduces attention exactly (the Day 52 identity)
    q = rng.standard_normal((16, 64)).astype(np.float32)
    k = rng.standard_normal((2048, 64)).astype(np.float32)
    v = rng.standard_normal((2048, 64)).astype(np.float32)
    full = attention(q, k, v)
    for parts in (2, 4, 8):
        assert np.abs(context_parallel_attention(q, k, v, parts) - full).max() < 1e-6

    # pipeline bubble formula, and the boundary traffic is the small one
    pb = {r['micro']: r['bubble'] for r in pipeline_bubble(stages=8)}
    assert abs(pb[1] - 7 / 8) < 1e-12 and abs(pb[64] - 7 / 71) < 1e-12
    pl = placement_table()
    by = {r['axis'].split(' (')[0]: r['gb'] for r in pl}
    assert by['pipeline parallel'] < by['tensor parallel'] / 100
    assert by['expert parallel'] > by['tensor parallel']

    # the mesh: tensor and pipeline shard the parameters before ZeRO does
    tm = training_mesh(gpus=64, tp=8, pp=4)
    assert tm['dp'] == 2 and tm['layers_per_stage'] == 20
    assert tm['mem_ddp_gb'] < zero_table(p=8)[0]['gb_per_rank'] / 30
    assert abs(tm['mem_zero3_gb'] * tm['dp'] - tm['mem_ddp_gb']) < 1e-6

    # decode keeps the hops and loses the bytes
    dec, pre = decode_tp_cost(requests=64, p=8)
    assert dec['latency_us'] == pre['latency_us']
    assert dec['latency_share'] > 0.8 and pre['latency_share'] < 0.1

    # no gradient at inference: data parallelism is a memory layout, not traffic
    da = dp_attention(requests=64, p=8)
    assert abs(da['tp_mla_per_rank_gb'] / da['dp_attention_per_rank_gb'] - 8) < 1e-9

    # the five failures
    assert mismatched_order()['finished'] is False
    ro = reduction_order()
    assert ro['differing'] > 0 and ro['ring_repeatable'] is True
    st = straggler()
    assert st[-1]['loss_pct'] > st[0]['loss_pct'] * 5
    assert missing_all_reduce()['rel_dropped'] > 0.1


if __name__ == '__main__':
    main()
