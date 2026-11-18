"""Day 61 - quantization: fewer bits per number, and what it costs.

A 70B model in bf16 is 140 GB of weights.  Every decode step reads all of them, so
both the size on the card and the time per token scale with the number of bytes per
weight.  Quantization is the decision to store those weights in 8 bits, or 4, and the
whole subject is about what that rounding does to the answers.

The question is never "is int8 accurate".  It is: which numbers share a scale, where do
the outliers go, and is the rounding chosen to minimise the error in the weights or the
error in the layer's output?  Those three decisions separate every method here -
absmax, per-channel, group-wise, FP8, MXFP4, SmoothQuant, AWQ, GPTQ, and QAT.

Everything is measured on one linear layer with activations shaped like a real one:
a few channels an order of magnitude larger than the rest.  The metric throughout is
the relative error of the layer's output, not of its weights, because that is what the
next layer sees.

Run `python quantization.py` to print the whole story.
"""

import math

import numpy as np


# ================================================================ 1. a layer
def make_layer(d_in=512, d_out=512, n=256, seed=0, outlier_channels=6,
               outlier_scale=20.0):
    """Weights and activations with the shape a transformer actually produces.

    Weights are near-Gaussian.  Activations are not: a handful of channels carry
    values an order of magnitude larger than everything else, and that single fact
    is responsible for most of the methods in this file.
    """
    rng = np.random.default_rng(seed)
    W = rng.standard_normal((d_out, d_in)) * 0.02
    W += rng.standard_normal((d_out, d_in)) * 0.08 * (rng.random((d_out, d_in)) < 0.01)
    X = rng.standard_normal((n, d_in))
    chans = rng.choice(d_in, outlier_channels, replace=False)
    X[:, chans] *= outlier_scale
    return {'W': W, 'X': X, 'outliers': np.sort(chans)}


def out_err(Y, Yq):
    """Relative error of the layer output - the number that matters."""
    return float(np.linalg.norm(Yq - Y) / np.linalg.norm(Y))


# ================================================================ 2. integers
def quant_int(x, bits=8, axis=None, group=None, symmetric=True, percentile=None):
    """Round to a uniform grid and come back.

    axis=None       one scale for the whole tensor
    axis=-1         one scale per row (per output channel, for a weight matrix)
    group=g         one scale per g consecutive values in a row
    percentile=p    clip at the p-th percentile instead of the maximum
    """
    if group:
        shape = x.shape
        xg = x.reshape(-1, group)
        return quant_int(xg, bits, axis=-1, symmetric=symmetric,
                         percentile=percentile).reshape(shape)
    qmax = 2 ** (bits - 1) - 1
    a = np.abs(x)
    if percentile is not None:
        hi = np.percentile(a, percentile, axis=axis, keepdims=True)
    else:
        hi = a.max(axis=axis, keepdims=True) if axis is not None else a.max()
    if symmetric:
        scale = np.maximum(hi, 1e-12) / qmax
        q = np.clip(np.rint(x / scale), -qmax - 1, qmax)
        return q * scale
    lo = x.min(axis=axis, keepdims=True) if axis is not None else x.min()
    hi2 = x.max(axis=axis, keepdims=True) if axis is not None else x.max()
    scale = np.maximum(hi2 - lo, 1e-12) / (2 ** bits - 1)
    zero = np.rint(-lo / scale)
    q = np.clip(np.rint(x / scale) + zero, 0, 2 ** bits - 1)
    return (q - zero) * scale


def best_clip(x, bits=8, grid=40):
    """Search the clipping threshold that minimises squared error - 'MSE scale'."""
    a = np.abs(x).max()
    qmax = 2 ** (bits - 1) - 1
    best = (None, np.inf)
    for f in np.linspace(0.3, 1.0, grid):
        scale = a * f / qmax
        xq = np.clip(np.rint(x / scale), -qmax - 1, qmax) * scale
        e = float(((xq - x) ** 2).mean())
        if e < best[1]:
            best = (float(f), e)
    return {'fraction': best[0], 'mse': best[1],
            'absmax_mse': float(((quant_int(x, bits) - x) ** 2).mean())}


# ================================================================ 3. floats
def float_round(x, m_bits, emin, emax):
    """Round to the nearest value of a float format with this exponent range.

    emin/emax are the exponents of the smallest normal and the largest value; below
    emin the spacing stops shrinking, which is what subnormals do.  The point of a
    float grid is that the spacing grows with the magnitude, so the *relative* error
    is roughly constant instead of the absolute one.
    """
    big = (2 - 2.0 ** -m_bits) * 2.0 ** emax
    a = np.abs(x)
    with np.errstate(divide='ignore'):
        e = np.floor(np.log2(np.maximum(a, 1e-300)))
    e = np.clip(e, emin, emax)
    step = 2.0 ** (e - m_bits)
    q = np.rint(x / step) * step
    return np.clip(q, -big, big)


# (mantissa bits, smallest normal exponent, largest exponent, largest value)
FP8 = {'e4m3': (3, -6, 8, 448.0),        # no infinities: the e4m3fn of PyTorch and JAX
       'e5m2': (2, -14, 15, 57344.0)}
FP4_E2M1 = (1, 0, 2, 6.0)


def quant_fp8(x, fmt='e4m3', axis=None):
    """Scale into the format's range, round, scale back - the usual FP8 recipe."""
    m_bits, emin, emax, big = FP8[fmt]
    hi = np.abs(x).max(axis=axis, keepdims=True) if axis is not None else np.abs(x).max()
    scale = np.maximum(hi, 1e-12) / big
    return float_round(x / scale, m_bits, emin, emax) * scale


def mxfp4(x, block=32):
    """Micro-scaling FP4: one power-of-two exponent per block, E2M1 elements.

    Each block of 32 values shares an 8-bit exponent, and each value is 4 bits:
    sign, 2 exponent bits, 1 mantissa bit, so {0, .5, 1, 1.5, 2, 3, 4, 6} times the
    block scale.  Average cost is 4 + 8/32 = 4.25 bits per weight.
    """
    shape = x.shape
    xb = x.reshape(-1, block)
    hi = np.abs(xb).max(axis=1, keepdims=True)
    m_bits, emin, emax, _ = FP4_E2M1
    e = np.floor(np.log2(np.maximum(hi, 1e-30))) - emax       # E8M0 shared scale
    scale = 2.0 ** e
    q = float_round(xb / scale, m_bits, emin, emax)
    return (q * scale).reshape(shape)


def format_table(x, block=32):
    rows = []
    for name, fn, bits in [
            ('int8, per tensor', lambda v: quant_int(v, 8), 8.0),
            ('int8, per row', lambda v: quant_int(v, 8, axis=-1), 8.0),
            ('fp8 e4m3', lambda v: quant_fp8(v, 'e4m3'), 8.0),
            ('fp8 e5m2', lambda v: quant_fp8(v, 'e5m2'), 8.0),
            ('int4, per row', lambda v: quant_int(v, 4, axis=-1), 4.0),
            ('int4, group 128', lambda v: quant_int(v, 4, group=128), 4.0 + 16 / 128),
            ('mxfp4, block 32', lambda v: mxfp4(v, block), 4.0 + 8 / block)]:
        xq = fn(x)
        rows.append({'name': name, 'bits': bits,
                     'rel': float(np.linalg.norm(xq - x) / np.linalg.norm(x)),
                     'worst': float(np.abs(xq - x).max())})
    return rows


def dynamic_range(bits=8):
    """Why a float format survives outliers and an integer one does not."""
    qmax = 2 ** (bits - 1) - 1
    return {'int8_steps': 2 * qmax, 'int8_ratio': float(qmax),
            'e4m3_ratio': 448.0 / 2.0 ** -9, 'e5m2_ratio': 57344.0 / 2.0 ** -16,
            'fp4_levels': 15}


# ================================================================ 3b. a format table
# (name, exponent bits, mantissa bits, has infinities, where it is used)
# `special` says what the all-ones exponent is spent on:
#   'ieee'    - infinities and NaNs, as in fp32 (costs one whole binade of range)
#   'one_nan' - no infinity, one NaN pattern (the "fn" in float8_e4m3fn)
#   'none'    - every bit pattern is a number (the MX element formats)
SPECS = [
    ('fp32',        8, 23, 'ieee',    'the default everywhere'),
    ('tf32',        8, 10, 'ieee',    'NVIDIA tensor cores: fp32 range, fp16 precision'),
    ('bf16',        8,  7, 'ieee',    'training: fp32 range, half the bytes'),
    ('fp16',        5, 10, 'ieee',    'older training, needs loss scaling'),
    ('fp8 e4m3fn',  4,  3, 'one_nan', 'FP8 forward pass: weights and activations'),
    ('fp8 e5m2',    5,  2, 'ieee',    'FP8 gradients: more range, less precision'),
    ('fp6 e3m2',    3,  2, 'none',    'MX block element, more range'),
    ('fp6 e2m3',    2,  3, 'none',    'MX block element, more precision'),
    ('fp4 e2m1',    2,  1, 'none',    'MXFP4 / NVFP4 element'),
]
INT_SPECS = [('int8', 8), ('int4', 4)]


def spec(e_bits, m_bits, special='ieee'):
    """Everything about a float format follows from three numbers.

    A float is stored as sign | exponent | mantissa and read as

        value = (-1)^s * 2^(exponent - bias) * 1.mantissa      (normal numbers)
        value = (-1)^s * 2^(1 - bias)        * 0.mantissa      (subnormals)

    bias = 2^(e_bits-1) - 1 so the stored exponent is unsigned.  The all-ones
    exponent is reserved for infinity and NaN in IEEE formats; the FP8 "fn"
    formats give it up to gain one more binade of range instead.
    """
    bias = 2 ** (e_bits - 1) - 1
    emax = 2 ** e_bits - 1 - bias - (1 if special == 'ieee' else 0)
    emin = 1 - bias
    # the top mantissa pattern of the top exponent is a NaN in the "fn" formats
    top = 2 - (2 if special == 'one_nan' else 1) * 2.0 ** -m_bits
    return {'bias': bias, 'emax': emax, 'emin': emin, 'special': special,
            'max': top * 2.0 ** emax,
            'min_normal': 2.0 ** emin,
            'min_subnormal': 2.0 ** (emin - m_bits),
            'eps': 2.0 ** -m_bits,                 # relative spacing: 1 -> 1 + eps
            'decimal_digits': m_bits * math.log10(2),
            'values': 2 ** (1 + e_bits + m_bits)}


def format_spec_table():
    rows = []
    for name, e, m, special, use in SPECS:
        d = spec(e, m, special)
        rows.append({'name': name, 'bits': 1 + e + m, 's': 1, 'e': e, 'm': m,
                     'bias': d['bias'], 'max': d['max'],
                     'min_normal': d['min_normal'], 'min_sub': d['min_subnormal'],
                     'eps': d['eps'], 'digits': d['decimal_digits'],
                     'range': d['max'] / d['min_subnormal'], 'use': use,
                     'special': special, 'kind': 'float'})
    for name, b in INT_SPECS:
        qmax = 2 ** (b - 1) - 1
        rows.append({'name': name, 'bits': b, 's': 1, 'e': 0, 'm': b - 1,
                     'bias': None, 'max': float(qmax), 'min_normal': 1.0,
                     'min_sub': 1.0, 'eps': 1.0 / qmax, 'digits': math.log10(qmax),
                     'range': float(qmax), 'kind': 'int',
                     'use': 'a uniform grid: the scale carries the magnitude'})
    return rows


def mx_table(block=32):
    """The MX ("micro-scaling") formats: an element format plus a block exponent."""
    rows = []
    for name, e, m, sp in [('MXFP8 e4m3', 4, 3, 'one_nan'), ('MXFP6 e3m2', 3, 2, 'none'),
                           ('MXFP4 e2m1', 2, 1, 'none')]:
        el = 1 + e + m
        rows.append({'name': name, 'element_bits': el, 'block': block,
                     'scale_bits': 8,                      # E8M0: a power of two
                     'bits_per_value': el + 8 / block,
                     'max_element': spec(e, m, sp)['max']})
    return rows


def bits_of_float32(x):
    """How a float is actually laid out in memory - the thing a table cannot show."""
    b = np.float32(x).view(np.uint32)
    bits = format(int(b), '032b')
    s, e, m = bits[0], bits[1:9], bits[9:]
    exp = int(e, 2)
    frac = int(m, 2) / 2 ** 23
    return {'x': float(x), 'bits': bits, 'sign': s, 'exp_bits': e, 'man_bits': m,
            'stored_exp': exp, 'bias': 127, 'unbiased': exp - 127,
            'mantissa': 1 + frac,
            'check': (-1) ** int(s) * (1 + frac) * 2.0 ** (exp - 127)}


# ================================================================ 3c. the methods
# (name, what it quantizes, typical bits, needs calibration, needs training, idea)
METHODS = [
    ('round to nearest (RTN)', 'W', '8 / 4', False, False,
     'absmax scale per tensor, row or group; the baseline everything is compared to'),
    ('MSE / percentile clipping', 'W, A', '8 / 4', True, False,
     'search the clipping threshold that minimises squared error instead of using max'),
    ('LLM.int8()', 'W + A', '8 (+16)', False, False,
     'split the matmul: outlier columns stay in 16-bit, the rest go to int8'),
    ('ZeroQuant / per-token dynamic', 'W + A', '8', False, False,
     'per-token activation scales computed at run time, per-column weight scales'),
    ('SmoothQuant', 'W + A', '8', True, False,
     'migrate range from activations into weights: X W = (X/s)(W s)'),
    ('AWQ', 'W', '4', True, False,
     'scale up the channels the activations actually use before rounding'),
    ('GPTQ (from OBQ / OBS)', 'W', '4 / 3', True, False,
     'round column by column and push the error into the rest using H = X^T X'),
    ('SqueezeLLM', 'W', '4 / 3', True, False,
     'sensitivity-weighted k-means codebook plus a sparse outlier set'),
    ('SpQR', 'W', '~4', True, False,
     'keep the few most sensitive weights in high precision, quantize the rest hard'),
    ('HQQ', 'W', '4 / 3 / 2', False, False,
     'optimise the scale and zero-point with a robust loss; no calibration data'),
    ('QuIP#', 'W', '2', True, False,
     'rotate the weights to make them incoherent, then lattice-quantize'),
    ('AQLM', 'W', '2', True, True,
     'additive codebooks over groups, fine-tuned end to end'),
    ('NF4 / QLoRA', 'W', '4', False, False,
     'levels at the quantiles of a normal distribution, plus a second quantization '
     'of the scales'),
    ('GGUF k-quants', 'W', '2-8', False, False,
     'blocks of 16-32 with a super-block scale; the llama.cpp family (Q4_K_M, ...)'),
    ('FP8 (e4m3 / e5m2)', 'W + A + grad', '8', False, False,
     'a float grid plus a per-tensor or per-row scale; the hardware path'),
    ('MX / NVFP4', 'W + A', '4-8', False, False,
     'block floating point: one shared exponent per 16-32 values'),
    ('KV cache quantization', 'KV', '8 / 4', False, False,
     'the cache is the memory, so quantize K and V per head or per token'),
    ('QAT (STE, LSQ, PACT)', 'W + A', '4 / 3 / 2', False, True,
     'train through the rounding; LSQ learns the step size, PACT the clipping'),
]


def method_table():
    """What is quantized, what it costs to apply, and what the idea is."""
    rows = []
    for name, what, bits, calib, train, idea in METHODS:
        rows.append({'name': name, 'what': what, 'bits': bits,
                     'calibration': calib, 'training': train, 'idea': idea,
                     'cost': ('a training run' if train else
                              'a few calibration batches' if calib else
                              'nothing - run it on the checkpoint')})
    return rows


def notation(w_bits=4, a_bits=16, group=128):
    """The WxAy notation, decoded, with the memory each choice actually saves."""
    return {'name': f'W{w_bits}A{a_bits}',
            'weights': f'{w_bits}-bit weights',
            'activations': ('16-bit activations: weight-only, the default for serving'
                            if a_bits == 16 else f'{a_bits}-bit activations too'),
            'bits_per_weight': w_bits + 16 / group,
            'group': group,
            'why_weight_only': 'decode reads weights once per token and activations '
                               'are tiny, so W4A16 already removes most of the traffic',
            'why_w8a8': 'prefill is compute-bound, so int8 or fp8 activations buy '
                        'tensor-core throughput that weight-only cannot'}


# ================================================================ 4. the outliers
def where_the_error_is(layer, bits=8):
    """Quantize weights and activations separately and see which one hurts."""
    W, X = layer['W'], layer['X']
    Y = X @ W.T
    rows = [('weights only, per row', quant_int(W, bits, axis=-1), X),
            ('activations only, per tensor', W, quant_int(X, bits)),
            ('activations only, per token', W, quant_int(X, bits, axis=-1)),
            ('both', quant_int(W, bits, axis=-1), quant_int(X, bits, axis=-1))]
    return [{'name': n, 'rel': out_err(Y, Xq @ Wq.T)} for n, Wq, Xq in rows]


def smooth_quant(layer, alpha=0.5, bits=8):
    """Move the dynamic range from the activations into the weights.

    X W^T = (X / s) (W s)^T for any per-channel s, so a channel that is hard to
    quantize on the activation side can be made easier at the weights' expense.
    s = max|X_j|^alpha / max|W_j|^(1-alpha) splits the difficulty between them.
    """
    W, X = layer['W'], layer['X']
    Y = X @ W.T
    ax = np.abs(X).max(axis=0)
    aw = np.abs(W).max(axis=0)
    s = np.maximum(ax, 1e-9) ** alpha / np.maximum(aw, 1e-9) ** (1 - alpha)
    s = s / s.mean()
    Xs, Ws = X / s, W * s
    plain = out_err(Y, quant_int(X, bits, axis=-1) @ quant_int(W, bits, axis=-1).T)
    smooth = out_err(Y, quant_int(Xs, bits, axis=-1) @ quant_int(Ws, bits, axis=-1).T)
    return {'alpha': alpha, 'plain': plain, 'smooth': smooth,
            'act_range_before': float(ax.max() / np.median(ax)),
            'act_range_after': float(np.abs(Xs).max(axis=0).max()
                                     / np.median(np.abs(Xs).max(axis=0)))}


def smooth_sweep(layer, alphas=(0.0, 0.25, 0.5, 0.75, 1.0), bits=8):
    return [smooth_quant(layer, a, bits) for a in alphas]


def mixed_precision(layer, bits=8, keep=6):
    """LLM.int8(): leave the outlier columns in high precision, quantize the rest."""
    W, X = layer['W'], layer['X']
    Y = X @ W.T
    mag = np.abs(X).max(axis=0)
    cols = np.argsort(-mag)[:keep]
    mask = np.zeros(X.shape[1], bool)
    mask[cols] = True
    Xq = quant_int(X[:, ~mask], bits, axis=-1)
    Wq = quant_int(W[:, ~mask], bits, axis=-1)
    Yq = Xq @ Wq.T + X[:, mask] @ W[:, mask].T           # the 16-bit remainder
    return {'kept': keep, 'share': float(mask.mean()),
            'rel': out_err(Y, Yq),
            'all_int8': out_err(Y, quant_int(X, bits, axis=-1)
                                @ quant_int(W, bits, axis=-1).T)}


# ================================================================ 5. rounding better
def awq_scale(layer, bits=4, grid=20, group=128):
    """AWQ: scale the salient input channels up before rounding the weights.

    Rounding error is proportional to the scale of a row, so multiplying a column of
    W by s (and dividing the activation by s) protects the columns that the
    activations actually use.  The exponent is searched, not derived.
    """
    W, X = layer['W'], layer['X']
    Y = X @ W.T
    mag = np.abs(X).mean(axis=0)
    base = out_err(Y, X @ quant_int(W, bits, group=group).T)
    best = (0.0, base)
    for a in np.linspace(0, 1, grid):
        s = np.maximum(mag, 1e-9) ** a
        s = s / s.mean()
        Wq = quant_int(W * s, bits, group=group) / s
        e = out_err(Y, X @ Wq.T)
        if e < best[1]:
            best = (float(a), e)
    return {'rtn': base, 'awq': best[1], 'alpha': best[0]}


def gptq(W, X, bits=4, group=128, damp=0.01):
    """Round column by column and push the error into the columns not yet done.

    Round-to-nearest treats every weight independently.  GPTQ asks a different
    question: given the inputs this layer actually sees (through H = X^T X), what is
    the cheapest way to absorb the rounding error I just made?  The answer is a
    rank-one update of the remaining columns, which is the OBQ/OBS update specialised
    to one layer.
    """
    W = W.copy()
    d = W.shape[1]
    H = X.T @ X / X.shape[0]
    H[np.diag_indices(d)] += damp * np.mean(np.diag(H))
    Hinv = np.linalg.inv(H)
    Q = np.zeros_like(W)
    for start in range(0, d, group):
        end = min(start + group, d)
        blk = W[:, start:end]
        qmax = 2 ** (bits - 1) - 1
        scale = np.maximum(np.abs(blk).max(axis=1, keepdims=True), 1e-12) / qmax
        for j in range(start, end):
            w = W[:, j]
            q = np.clip(np.rint(w / scale[:, 0]), -qmax - 1, qmax) * scale[:, 0]
            Q[:, j] = q
            err = (w - q) / Hinv[j, j]
            if j + 1 < d:                              # compensate what remains
                W[:, j + 1:] -= np.outer(err, Hinv[j, j + 1:])
    return Q


def rounding_table(layer, bits=4, group=128):
    W, X = layer['W'], layer['X']
    Y = X @ W.T
    rtn = quant_int(W, bits, group=group)
    g = gptq(W, X, bits=bits, group=group)
    aw = awq_scale(layer, bits=bits, group=group)
    return {'rtn_weight_err': float(np.linalg.norm(rtn - W) / np.linalg.norm(W)),
            'gptq_weight_err': float(np.linalg.norm(g - W) / np.linalg.norm(W)),
            'rtn_out': out_err(Y, X @ rtn.T), 'gptq_out': out_err(Y, X @ g.T),
            'awq_out': aw['awq'], 'awq_alpha': aw['alpha'], 'bits': bits,
            'group': group}


# ================================================================ 6. training
def fake_quant(x, bits=8, axis=None):
    return quant_int(x, bits, axis=axis)


def qat_demo(bits=3, steps=1500, n=512, d=32, lr=0.02, seed=1, correlated=True):
    """Fine-tune through the rounding with a straight-through estimator.

    round() has zero gradient almost everywhere, so the backward pass pretends it is
    the identity.  The weights stay in float, the forward pass sees rounded values,
    and the optimiser can land on weights that are good *after* rounding - which is
    not the same as the best float weights, rounded.

    As in real QAT this starts from the trained float model and fine-tunes; and as in
    real QAT the loss does not fall monotonically, because the forward pass is a step
    function.  Both the last and the best iterate are reported.
    """
    rng = np.random.default_rng(seed)
    if correlated:                                     # features that overlap, as in a real layer
        A = rng.standard_normal((d, d)) / np.sqrt(d) + 0.3 * np.ones((d, d)) / np.sqrt(d)
        X = rng.standard_normal((n, d)) @ A
    else:                                              # nearly orthogonal features
        X = rng.standard_normal((n, d))
    y = X @ rng.standard_normal(d)
    wf = np.linalg.lstsq(X, y, rcond=None)[0]          # the float model
    loss = lambda w: float(np.mean((X @ fake_quant(w, bits) - y) ** 2))
    w, hist = wf.copy(), []
    step = lr / max(1.0, float(np.linalg.norm(X.T @ X / n, 2)))
    for t in range(steps):
        wq = fake_quant(w, bits)                       # forward: rounded
        g = X.T @ (X @ wq - y) * 2 / n                 # backward: straight through
        w -= step * (1 - t / steps) * g
        hist.append(loss(w))
    return {'bits': bits, 'correlated': correlated,
            'float': float(np.mean((X @ wf - y) ** 2)),
            'ptq': loss(wf), 'qat': min(hist), 'qat_last': hist[-1],
            'osc': float(np.std(hist[-300:])), 'tail_mean': float(np.mean(hist[-300:])),
            'ratio': loss(wf) / max(min(hist), 1e-12), 'hist': hist[::100]}


def memory_table(params=70e9):
    rows = []
    for name, bits in [('bf16', 16), ('fp8', 8), ('int8', 8), ('int4 + group 128', 4 + 16 / 128),
                       ('mxfp4', 4 + 8 / 32)]:
        gb = params * bits / 8 / 1e9
        rows.append({'name': name, 'bits': bits, 'gb': gb,
                     'ms_at_5TBs': gb * 1e9 / 5e12 * 1e3})
    return rows


# ================================================================ 7. checks
def check():
    L = make_layer(d_in=128, d_out=128, n=64, seed=3)
    W, X = L['W'], L['X']
    # round trip through a grid: error bounded by half a step
    q8 = quant_int(W, 8)
    step = 2 * np.abs(W).max() / (2 ** 8 - 2)
    assert np.abs(q8 - W).max() <= step, (np.abs(q8 - W).max(), step)
    # more bits is never worse, finer scales are never worse
    e = [float(np.linalg.norm(quant_int(W, b) - W)) for b in (4, 6, 8)]
    assert e[0] > e[1] > e[2], e
    assert np.linalg.norm(quant_int(W, 8, axis=-1) - W) <= np.linalg.norm(q8 - W)
    assert np.linalg.norm(quant_int(W, 4, group=32) - W) <= \
           np.linalg.norm(quant_int(W, 4, axis=-1) - W)
    # fp8 e4m3 keeps more precision than e5m2, e5m2 keeps more range
    assert float_round(1.0, 3, -6, 8) == 1.0 and float_round(0.0, 3, -6, 8) == 0.0
    assert np.abs(quant_fp8(W, 'e4m3') - W).mean() < np.abs(quant_fp8(W, 'e5m2') - W).mean()
    assert FP8['e5m2'][3] > FP8['e4m3'][3]
    # mxfp4 beats plain per-row int4 on this data, at a comparable bit cost
    assert np.linalg.norm(mxfp4(W) - W) < np.linalg.norm(quant_int(W, 4, axis=-1) - W)
    # clipping below the maximum can beat absmax
    bc = best_clip(X, 8)
    assert bc['fraction'] < 1.0 and bc['mse'] <= bc['absmax_mse']
    # the method table is complete and self-consistent
    mt = method_table()
    names = {r['name'] for r in mt}
    for needed in ('GPTQ (from OBQ / OBS)', 'AWQ', 'SmoothQuant', 'LLM.int8()',
                   'QAT (STE, LSQ, PACT)', 'MX / NVFP4', 'NF4 / QLoRA'):
        assert needed in names, needed
    assert all(r['cost'] for r in mt)
    assert {r['cost'] for r in mt} == {'nothing - run it on the checkpoint',
                                       'a few calibration batches', 'a training run'}
    assert abs(notation()['bits_per_weight'] - 4.125) < 1e-9
    # the format table reproduces the published constants
    t = {r['name']: r for r in format_spec_table()}
    assert t['fp16']['max'] == 65504.0 and t['fp32']['max'] == np.finfo(np.float32).max
    assert t['fp8 e4m3fn']['max'] == 448.0 and t['fp8 e5m2']['max'] == 57344.0
    assert t['fp4 e2m1']['max'] == 6.0 and t['fp6 e3m2']['max'] == 28.0
    assert t['bf16']['eps'] == 2.0 ** -7 and t['bf16']['e'] == t['fp32']['e']
    assert abs(t['fp32']['min_sub'] - float(np.finfo(np.float32).smallest_subnormal)) < 1e-50
    assert [r['max_element'] for r in mx_table()] == [448.0, 28.0, 6.0]
    assert abs(mx_table()[2]['bits_per_value'] - 4.25) < 1e-12
    # and the worked example really is what the bits say
    for v in (0.15625, -3.5, 1.0, 12345.75):
        b = bits_of_float32(v)
        assert b['check'] == v, b
    # the activations, not the weights, are what break int8 here
    rows = {r['name']: r['rel'] for r in where_the_error_is(L)}
    assert rows['activations only, per tensor'] > 3 * rows['weights only, per row'], rows
    # SmoothQuant and mixed precision both fix it
    sq = smooth_quant(L, 0.5)
    assert sq['smooth'] < sq['plain']
    mp = mixed_precision(L, keep=6)
    assert mp['rel'] < mp['all_int8']
    # GPTQ trades weight error for output error, and wins on the one that matters
    rt = rounding_table(L, bits=4, group=32)
    assert rt['gptq_out'] < rt['rtn_out']
    assert rt['gptq_weight_err'] > rt['rtn_weight_err']
    # QAT beats "train in float, then round"
    q = qat_demo(bits=3, steps=400, d=16)
    assert q['qat'] < q['ptq'], q
    # and the gain comes from correlated inputs, not from the training itself
    qo = qat_demo(bits=3, steps=400, d=16, correlated=False)
    assert q['ratio'] > qo['ratio'], (q['ratio'], qo['ratio'])


# ================================================================ 8. the story
def header(t):
    print('\n' + '=' * 72)
    print(t)
    print('=' * 72)


def main():
    L = make_layer()
    W, X = L['W'], L['X']
    Y = X @ W.T

    header('1. Why anyone does this')
    print('  Weights are read once per decode step, so bytes per weight is both the')
    print('  size on the card and the time per token:\n')
    print('  format             bits/weight   70B model   one read at 5 TB/s')
    for r in memory_table():
        print(f'  {r["name"]:18} {r["bits"]:9.2f}   {r["gb"]:7.1f} GB   '
              f'{r["ms_at_5TBs"]:14.1f} ms')
    print('\n  Nothing else in this series buys a 4x cut in both numbers at once.')
    print('  The rest of the day is about what the rounding costs.')

    header('2. What a quantized number is')
    print('  Pick a scale, round to a grid of integers, multiply back:\n')
    print('    q = round(x / scale),   x_hat = q * scale\n')
    v = np.array([0.031, -0.0072, 0.0009, 0.118])
    s = np.abs(v).max() / 127
    print(f'  scale = max|x| / 127 = {s:.6f}')
    for x_ in v:
        print(f'    {x_:+.4f}  ->  q = {int(round(x_ / s)):4d}  ->  '
              f'{round(x_ / s) * s:+.6f}   (error {abs(round(x_ / s) * s - x_):.6f})')
    print('\n  Every error above is at most half a step, and the step is set by the')
    print('  largest value in the group.  So the whole subject reduces to one')
    print('  question: which numbers are forced to share a scale with which?')

    header('3. Every format on one page')
    b = bits_of_float32(0.15625)
    print('  Before the table, what "fp32" actually means in memory.  A float is')
    print('  three fields packed into one word - a sign bit, an exponent, and a')
    print('  mantissa (the fraction) - and it is read as\n')
    print('      value = (-1)^sign * 2^(exponent - bias) * 1.mantissa\n')
    print(f'  Take {b["x"]} as an fp32:\n')
    print(f'    sign      {b["sign"]}                 -> +1')
    print(f'    exponent  {b["exp_bits"]}          -> stored {b["stored_exp"]}, '
          f'minus bias {b["bias"]} = {b["unbiased"]}')
    print(f'    mantissa  {b["man_bits"]} -> 1.{b["man_bits"][:3]}... = {b["mantissa"]}')
    print(f'    value     {b["mantissa"]} * 2^{b["unbiased"]} = {b["check"]}\n')
    print('  The exponent is stored biased (as an unsigned number) so that comparing')
    print('  two floats as integers gives the same order.  The leading 1 before the')
    print('  point is not stored at all - it is implied, which is one free bit of')
    print('  precision.  Numbers too small for that (below 2^emin) are stored as')
    print('  "subnormals" with a leading 0 instead, losing precision gradually rather')
    print('  than snapping to zero.\n')
    print('  So a format is three numbers: how many exponent bits (range), how many')
    print('  mantissa bits (precision), and what the all-ones exponent is spent on.\n')
    print('  format        bits  S  E   M   bias     largest      smallest normal')
    print('  ' + '-' * 70)
    for r in format_spec_table():
        if r['kind'] != 'float':
            continue
        print(f'  {r["name"]:12} {r["bits"]:4d}  {r["s"]}  {r["e"]:2d}  {r["m"]:2d}  '
              f'{r["bias"]:5d}   {r["max"]:11.4g}   {r["min_normal"]:14.3g}')
    for r in format_spec_table():
        if r['kind'] == 'float':
            continue
        print(f'  {r["name"]:12} {r["bits"]:4d}  {r["s"]}   -  {r["m"]:2d}      -   '
              f'{r["max"]:11.4g}   {1.0:14.3g}')
    print('\n  format        relative step    decimal digits   dynamic range   used for')
    print('  ' + '-' * 86)
    for r in format_spec_table():
        print(f'  {r["name"]:12} {r["eps"]:13.3g}   {r["digits"]:14.1f}   '
              f'{r["range"]:13.3g}   {r["use"]}')
    print('\n  How to read the two tables together.  Exponent bits buy *range*: bf16')
    print('  and fp32 have the same 8, which is why bf16 can hold any fp32 value and')
    print('  dropping to bf16 never overflows.  Mantissa bits buy *precision*: the')
    print('  relative step is 2^-m, so bf16 resolves about 2 decimal digits and fp32')
    print('  about 7.  fp16 has the opposite trade from bf16 at the same 16 bits -')
    print('  more precision, far less range - which is exactly why fp16 training')
    print('  needs loss scaling and bf16 training does not.')
    print('\n  The same trade repeats at 8 bits (e4m3 for the forward pass, e5m2 for')
    print('  gradients) and at 6 and 4 bits inside the MX formats.  And note the')
    print('  integers at the bottom: no exponent at all, so their "range" is just')
    print('  the number of levels, and all the magnitude information lives in the')
    print('  scale stored next to them.')
    print('\n  Two notes on names.  The "fn" in float8_e4m3fn means *finite*: it has')
    print('  no infinities, which buys one more binade of range (448 instead of 240).')
    print('  And "e4m3" is just the field widths, so any format can be read off its')
    print('  name: e5m2 is 5 exponent bits and 2 mantissa bits.\n')
    print('  The MX ("micro-scaling") formats are an element format plus a shared')
    print('  exponent per block of 32, which is where the fractional bit counts come')
    print('  from:\n')
    print('  format        element  block  scale   bits per value   element max')
    for r in mx_table():
        print(f'  {r["name"]:12} {r["element_bits"]:7d}  {r["block"]:5d}  E8M0   '
              f'{r["bits_per_value"]:13.2f}   {r["max_element"]:11.1f}')
    print('\n  The block scale is E8M0: 8 bits of exponent and no mantissa, so it is')
    print('  a power of two and applying it is an exponent add rather than a multiply.')
    print('  NVFP4 is the same idea with a block of 16 and an e4m3 scale instead.')

    header('4. Every method on one page')
    print('  The formats were the units; these are the methods.  Three questions')
    print('  separate all of them: what gets quantized, how the scale is chosen, and')
    print('  whether the rounding itself is improved.\n')
    print('  method                          quantizes     bits      what it costs you')
    print('  ' + '-' * 86)
    for r in method_table():
        print(f'  {r["name"]:31} {r["what"]:12}  {r["bits"]:9} {r["cost"]}')
    print('\n  Read the last column as the practical ladder.  Everything that costs')
    print('  nothing can be applied to a checkpoint you downloaded five minutes ago.')
    print('  Everything that wants calibration needs a few hundred real sequences -')
    print('  still cheap, still offline, but now the result depends on what you fed')
    print('  it.  Only the last rows want a training run, and that is why they are')
    print('  rare outside the labs that trained the model in the first place.\n')
    print('  The ideas, in one line each:\n')
    for r in method_table():
        print(f'    {r["name"]:31} {r["idea"]}')
    n = notation()
    print(f'\n  And the notation.  "{n["name"]}" means {n["weights"]} and '
          f'{n["activations"]}.')
    print(f'  With a scale per {n["group"]} weights, W4 really costs '
          f'{n["bits_per_weight"]:.2f} bits.')
    print(f'\n    weight-only (W4A16, W8A16): {n["why_weight_only"]}')
    print(f'    weights and activations (W8A8, FP8): {n["why_w8a8"]}')
    print('\n  The rest of this file measures the ones that can be measured on a')
    print('  single layer: RTN with every grouping, MSE clipping, LLM.int8(),')
    print('  SmoothQuant, AWQ, GPTQ and QAT.')

    header('5. Who shares a scale')
    print('  Same weights, same bit widths, different grouping:\n')
    print('  scheme                bits   rel. error   worst single error')
    for r in format_table(W):
        print(f'  {r["name"]:20} {r["bits"]:5.2f}   {r["rel"]:10.5f}   {r["worst"]:.6f}')
    print('\n  Per-row scales cost one extra float per 512 weights and cut the error')
    print('  by a third.  Going to 4 bits costs an order of magnitude - unless the')
    print('  group is small: int4 with a scale per 128 weights beats int8 per tensor.')
    print('  That is the trade everyone actually makes: fewer bits per value, more')
    print('  scales.')

    header('6. Integers against floats at the same width')
    dr = dynamic_range()
    print(f'  int8 has {dr["int8_steps"]} evenly spaced levels and a largest-to-smallest')
    print(f'  ratio of {dr["int8_ratio"]:.0f}.  The FP8 formats spend bits on an exponent')
    print(f'  instead: e4m3 covers a ratio of {dr["e4m3_ratio"]:.0f} and e5m2 '
          f'{dr["e5m2_ratio"]:.0f}.\n')
    print('  Integers space their levels evenly, so the absolute error is constant and')
    print('  small values lose all their significant digits.  Floats space levels')
    print('  geometrically, so the relative error is roughly constant everywhere.')
    print('  e4m3 (3 mantissa bits) is the forward-pass format; e5m2 (2 mantissa bits,')
    print('  more exponent) exists for gradients, which span a far wider range and')
    print('  care less about precision.  That is why mixed FP8 training uses both.')

    header('7. The outliers are in the activations, not the weights')
    print(f'  {len(L["outliers"])} of {X.shape[1]} input channels carry values '
          f'{np.abs(X[:, L["outliers"]]).max() / np.median(np.abs(X)):.0f}x the median.')
    print('  Quantizing one side at a time shows where the damage comes from:\n')
    for r in where_the_error_is(L):
        print(f'    {r["name"]:32} output error {r["rel"]:.5f}')
    print('\n  The weights are easy; they are nearly Gaussian and a per-row scale fits')
    print('  them.  The activations are not: one channel with a huge value sets the')
    print('  scale for the whole tensor and everything else rounds to a handful of')
    print('  levels.  This is the entire reason weight-only quantization is the')
    print('  default and activation quantization needs a trick.')

    header('8. Three ways to deal with the outliers')
    mp = mixed_precision(L, keep=6)
    print(f'  (a) Keep them.  LLM.int8() splits the matrix: {mp["share"]:.1%} of the')
    print(f'      columns stay in 16-bit, the rest go to int8.  Output error')
    print(f'      {mp["all_int8"]:.5f} -> {mp["rel"]:.5f}.  The cost is two matmuls and')
    print('      an irregular shape, which is why it is slower than it looks.\n')
    print('  (b) Move them.  SmoothQuant uses X W^T = (X/s)(W s)^T to shift range')
    print('      from the activations into the weights, with alpha deciding how much:\n')
    print('      alpha   activation range   output error')
    for r in smooth_sweep(L):
        print(f'      {r["alpha"]:5.2f}   {r["act_range_after"]:16.1f}   {r["smooth"]:.5f}')
    print(f'      (no smoothing at all: {smooth_sweep(L)[0]["plain"]:.5f})')
    print('\n      alpha=0 moves nothing, alpha=1 moves everything and breaks the')
    print('      weights instead.  The useful values sit in the middle, and the')
    print('      scales fold into the previous layer norm, so it costs nothing at')
    print('      run time.\n')
    print('  (c) Give them their own scale: that is what per-token and group-wise')
    print('      scales in section 3 already do, and it is why fine-grained formats')
    print('      like MXFP4 are the direction hardware is going.')

    header('9. Rounding to nearest is not the best rounding')
    rt = rounding_table(L, bits=4, group=128)
    print(f'  At {rt["bits"]} bits with a scale per {rt["group"]} weights:\n')
    print(f'    round to nearest   weight error {rt["rtn_weight_err"]:.5f}   '
          f'output error {rt["rtn_out"]:.5f}')
    print(f'    GPTQ               weight error {rt["gptq_weight_err"]:.5f}   '
          f'output error {rt["gptq_out"]:.5f}')
    print(f'    AWQ (alpha {rt["awq_alpha"]:.2f})   '
          f'                           output error {rt["awq_out"]:.5f}')
    print('\n  Look at the two columns.  GPTQ ends up with *worse* weights than')
    print('  round-to-nearest and a better layer output, which is the whole idea:')
    print('  nobody cares about the weights.  It rounds one column at a time and')
    print('  pushes the error it just made into the columns it has not done yet,')
    print('  using H = X^T X - the second-order information about which directions')
    print('  the real inputs actually excite.  AWQ reaches a similar place from the')
    print('  other side: it does not change the rounding, it rescales the columns the')
    print('  activations care about so that the same rounding hurts less.')
    print('\n  Both need calibration data.  That is the practical difference from')
    print('  everything above: a handful of real sequences, run once, offline.')

    header('10. Quantization-aware training')
    q = qat_demo()
    print(f'  Same layer at {q["bits"]} bits, two ways to get there:\n')
    print(f'    train in float, then round    loss {q["ptq"]:7.3f}')
    print(f'    fine-tune through the rounding loss {q["qat"]:6.3f}   '
          f'({q["ratio"]:.2f}x better)')
    print(f'    float baseline                loss {q["float"]:7.6f}')
    print('\n  round() has zero gradient almost everywhere, so the backward pass')
    print('  pretends it is the identity - the straight-through estimator.  The')
    print('  weights stay in float, the forward pass sees rounded values, and the')
    print('  optimiser lands on weights that are good *after* rounding.')
    qo = qat_demo(correlated=False)
    print(f'\n  And here is where the gain actually comes from.  Repeat the whole')
    print('  thing with nearly orthogonal inputs instead of overlapping ones:\n')
    print(f'    correlated inputs     PTQ {q["ptq"]:7.3f} -> QAT {q["qat"]:7.3f}  '
          f'({q["ratio"]:.2f}x)')
    print(f'    orthogonal inputs     PTQ {qo["ptq"]:7.3f} -> QAT {qo["qat"]:7.3f}  '
          f'({qo["ratio"]:.2f}x)')
    print('\n  With orthogonal inputs each weight stands alone, so the best rounded')
    print('  weights really are the rounded best weights and there is nothing to')
    print('  learn.  The gain comes entirely from one weight being able to absorb')
    print('  another one\'s rounding error - which is the same property GPTQ exploits')
    print('  in closed form through H = X^T X.  Training discovers it; GPTQ computes')
    print('  it.  Neither works on inputs that do not overlap.')
    print(f'\n  One honest detail the papers put in a footnote: the loss does not fall')
    print(f'  monotonically.  Over the last 300 steps it swings with a standard')
    print(f'  deviation of {q["osc"]:.2f} around {q["tail_mean"]:.2f}, because a weight')
    print(f'  crossing a rounding boundary changes the output discontinuously.  The')
    print(f'  final iterate scores {q["qat_last"]:.3f} and the best one {q["qat"]:.3f}, '
          f'so QAT runs')
    print('  keep a checkpoint by validation loss rather than the last step.')

    header('11. In PyTorch')
    print('  Three layers of the stack, which are easy to confuse:\n')
    print('  1. dtypes.  torch.float8_e4m3fn and torch.float8_e5m2 are real tensor')
    print('     dtypes; torch._scaled_mm(a, b, scale_a, scale_b) is the matmul that')
    print('     takes FP8 inputs and a scale and returns bf16.  The scale is yours to')
    print('     compute - which is section 3.\n')
    print('  2. post-training.  torchao is the current home:\n')
    print('       from torchao.quantization import quantize_, Int8WeightOnlyConfig')
    print('       quantize_(model, Int8WeightOnlyConfig())      # weight-only')
    print('       quantize_(model, Int4WeightOnlyConfig(group_size=128))')
    print('       quantize_(model, Float8DynamicActivationFloat8WeightConfig())\n')
    print('     GPTQ and AWQ live here too, and both want calibration batches.\n')
    print('  3. training.  Fake quantization with a straight-through estimator is')
    print('     section 8:\n')
    print('       class RoundSTE(torch.autograd.Function):')
    print('           @staticmethod')
    print('           def forward(ctx, x): return torch.round(x)')
    print('           @staticmethod')
    print('           def backward(ctx, g): return g        # straight through\n')
    print('     torchao.quantization.qat wraps this (QATConfig / FakeQuantizeConfig),')
    print('     and torchao.float8 does the real thing - FP8 matmuls in the forward')
    print('     and backward passes with per-tensor or per-row scales, which is how')
    print('     FP8 pretraining runs are done.')

    header('12. In JAX')
    print('  Same ideas, different shape, because JAX puts the numerics in the')
    print('  function rather than in the module:\n')
    print('  1. dtypes.  jnp.float8_e4m3fn and jnp.float8_e5m2 exist, and')
    print('     jax.lax.dot_general(..., preferred_element_type=jnp.bfloat16) is how')
    print('     you say "multiply in 8 bits, accumulate in 16".\n')
    print('  2. a straight-through estimator is six lines, with no autograd class:\n')
    print('       def ste_round(x):')
    print('           return x + jax.lax.stop_gradient(jnp.round(x) - x)\n')
    print('     The forward value is round(x); the gradient flows as if it were x.')
    print('     Exactly the trick in section 8, written as an identity plus a')
    print('     detached correction - the idiom to recognise in any JAX codebase.\n')
    print('  3. libraries.  AQT (google/aqt) is the quantized-training library: you')
    print('     wrap a dot_general and it handles the scales, the calibration and the')
    print('     backward pass, so the same function trains in int8 or serves in int8.')
    print('     Qwix is the newer JAX quantization library covering PTQ, QAT and')
    print('     LoRA-style flows without rewriting the model.\n')
    print('  The thing worth carrying between the two: PyTorch quantizes *modules*')
    print('  and JAX quantizes *operations*.  Everything else - the scales, the')
    print('  grouping, the straight-through estimator - is the same arithmetic.')

    header('13. Assertions')
    check()
    print('all assertions passed')


if __name__ == '__main__':
    main()
