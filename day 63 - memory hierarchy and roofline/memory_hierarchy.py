"""Day 63 - the memory hierarchy: why some kernels never get faster.

Every performance number in this series has turned out to be a memory number.  Decode
is slow because it reads the weights once per token.  The KV cache is the capacity
limit.  MoE is a communication problem.  Quantization helps because fewer bits means
fewer bytes.  None of those are statements about arithmetic.

This file is the model underneath all of them.  It measures a real memory hierarchy on
this machine, writes down the roofline that decides whether a kernel is limited by
arithmetic or by memory, and then applies it to the kernels from the earlier days -
including FlashAttention, which is tiling for attention and belongs here rather than in
its own day.

Numbers measured on this machine are marked "measured"; GPU numbers are vendor
specifications and are marked as such, because this VM has no GPU to measure.

Run `python memory_hierarchy.py` to print the whole story.
"""

import math
import time

import numpy as np


# ================================================================ 1. the hierarchy
# Vendor specifications, not measurements.  Capacity per unit, bandwidth aggregate.
GPU_LEVELS = [
    ('registers', 256 * 1024, 100_000.0, 1),
    ('shared memory / LDS', 160 * 1024, 20_000.0, 20),
    ('L2 cache', 50 * 1024 * 1024, 7_000.0, 200),
    ('HBM', 192 * 1024 ** 3, 5_300.0, 600),
    ('host memory over PCIe', 1024 * 1024 ** 3, 64.0, 2_000),
    ('another node over RDMA', 0, 50.0, 10_000),
]


def hierarchy_table():
    """Capacity, bandwidth and latency, with each level relative to the next."""
    rows = []
    for i, (name, cap, bw, lat) in enumerate(GPU_LEVELS):
        prev = GPU_LEVELS[i - 1][2] if i else None
        rows.append({'name': name, 'bytes': cap, 'gbs': bw, 'ns': lat,
                     'slower_than_above': bw and prev and prev / bw,
                     'bytes_per_flop_time': bw})
    return rows


def measure_sequential(sizes_kb=(64, 256, 1024, 4096, 16384, 65536, 262144),
                       work_bytes=2e8):
    """Measured: streaming read bandwidth against working-set size.

    Summing an array reads 4 bytes per element and does one add, so it is
    bandwidth-bound at every size.  Hardware prefetchers see the pattern coming, so
    this curve is much flatter than the hierarchy would suggest - which is the point.
    """
    rows = []
    for kb in sizes_kb:
        n = kb * 1024 // 4
        a = np.ones(n, dtype=np.float32)
        reps = max(3, int(work_bytes / (n * 4)))
        a.sum()
        t0 = time.perf_counter()
        for _ in range(reps):
            a.sum()
        dt = (time.perf_counter() - t0) / reps
        rows.append({'kb': kb, 'seconds': dt, 'gbs': n * 4 / dt / 1e9})
    return rows


def measure_random(sizes_kb=(16, 64, 256, 1024, 4096, 16384, 65536),
                   max_probes=1 << 20, seed=0):
    """Measured: the cost of one access that the prefetcher cannot predict.

    Same machine, same arrays, reading in a random order instead of in order.  Now
    every level of the hierarchy shows up, because each access waits for memory
    instead of for a prefetch that already happened.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for kb in sizes_kb:
        n = kb * 1024 // 4
        a = np.ones(n, dtype=np.float32)
        idx = rng.permutation(n)[:min(n, max_probes)].astype(np.int64)
        a[idx].sum()
        reps = max(3, int(2e7 / len(idx)))
        t0 = time.perf_counter()
        for _ in range(reps):
            a[idx].sum()
        dt = (time.perf_counter() - t0) / reps
        rows.append({'kb': kb, 'ns_per_access': dt / len(idx) * 1e9,
                     'probes': len(idx)})
    return rows


# ================================================================ 2. the roofline
class Machine:
    """A machine is two numbers: how fast it computes and how fast it reads."""

    def __init__(self, tflops=1300.0, bandwidth_gbs=5300.0, name='an MI355X-class GPU'):
        self.flops = tflops * 1e12
        self.bw = bandwidth_gbs * 1e9
        self.name = name

    @property
    def balance(self):
        """FLOPs per byte at which the two limits meet - the ridge point."""
        return self.flops / self.bw

    def time_for(self, flops, bytes_moved):
        return max(flops / self.flops, bytes_moved / self.bw)

    def bound(self, flops, bytes_moved):
        return 'compute' if flops / self.flops > bytes_moved / self.bw else 'memory'


def kernel_costs(d_model=4096, d_ff=14336, n_heads=32, d_head=128, seq=4096,
                 batch=1, bytes_per=2, params=70e9):
    """FLOPs and bytes for the kernels this series has already met."""
    k = {}
    # one token through one linear layer: a matrix-vector product
    k['decode: one linear (GEMV)'] = (2 * d_model * d_ff * batch,
                                      d_model * d_ff * bytes_per)
    # the same layer during prefill: a matrix-matrix product
    k['prefill: one linear (GEMM)'] = (2 * seq * d_model * d_ff,
                                       (d_model * d_ff + seq * d_model) * bytes_per)
    # attention over a long context, one query position
    kv = 2 * seq * n_heads * d_head * bytes_per
    k['decode: attention over KV'] = (2 * 2 * seq * n_heads * d_head * batch, kv)
    k['prefill: attention (no tiling)'] = (4 * seq * seq * n_heads * d_head,
                                           2 * seq * seq * n_heads * bytes_per)
    # the pointwise things everyone forgets
    n = seq * d_model
    k['RMSNorm'] = (4 * n, 2 * n * bytes_per)
    k['residual add'] = (n, 3 * n * bytes_per)
    k['softmax over the vocabulary'] = (5 * 151936 * batch, 2 * 151936 * batch * bytes_per)
    # a whole decode step of a 70B model
    k['decode: the whole model'] = (2 * params * batch, params * bytes_per)
    return k


def roofline_table(machine=None, **kw):
    m = machine or Machine()
    rows = []
    for name, (flops, bytes_moved) in kernel_costs(**kw).items():
        ai = flops / bytes_moved
        rows.append({'name': name, 'flops': flops, 'bytes': bytes_moved,
                     'intensity': ai, 'bound': m.bound(flops, bytes_moved),
                     'ms': m.time_for(flops, bytes_moved) * 1e3,
                     'achieved_frac': min(1.0, ai / m.balance)})
    return rows, m


def batch_crossover(machine=None, d_model=4096, d_ff=14336, bytes_per=2,
                    batches=(1, 4, 16, 64, 256, 1024)):
    """The one knob that moves a kernel along the roofline: how many rows it has."""
    m = machine or Machine()
    rows = []
    for b in batches:
        flops = 2 * b * d_model * d_ff
        bytes_moved = d_model * d_ff * bytes_per + 2 * b * d_model * bytes_per
        rows.append({'batch': b, 'intensity': flops / bytes_moved,
                     'bound': m.bound(flops, bytes_moved),
                     'ms': m.time_for(flops, bytes_moved) * 1e3,
                     'per_token_us': m.time_for(flops, bytes_moved) / b * 1e6})
    return rows, m.balance


# ================================================================ 3. tiling
def matmul_traffic(n=4096, tile=None, cache_floats=32 * 1024, bytes_per=4):
    """How many bytes a matrix multiply moves, with and without tiling.

    Without tiling, every element of B is re-read for every row of A: the traffic is
    O(n^3).  With square tiles of size T that fit in cache together, each tile of A
    and B is read n/T times, so the traffic falls by a factor of T.  The optimum T is
    set by the cache: 3 T^2 floats must fit.
    """
    if tile is None:
        tile = int(math.sqrt(cache_floats / 3))
    naive = (2 * n ** 3 + n ** 2) * bytes_per          # A row re-read per column block
    tiled = (2 * n ** 3 / tile + n ** 2) * bytes_per
    flops = 2 * n ** 3
    return {'n': n, 'tile': tile, 'naive_bytes': naive, 'tiled_bytes': tiled,
            'ratio': naive / tiled, 'naive_ai': flops / naive,
            'tiled_ai': flops / tiled,
            'max_tile_for_cache': int(math.sqrt(cache_floats / 3))}


def tile_sweep(n=4096, cache_floats=32 * 1024, tiles=(8, 16, 32, 64, 128, 256, 512)):
    rows = []
    for t in tiles:
        r = matmul_traffic(n, t, cache_floats)
        rows.append({'tile': t, 'fits': 3 * t * t <= cache_floats,
                     'gb': r['tiled_bytes'] / 1e9, 'ai': r['tiled_ai']})
    return rows


def flash_io(seq=4096, d_head=128, n_heads=32, sram_bytes=160 * 1024, bytes_per=2,
             standard_passes=4):
    """FlashAttention, read as an IO problem rather than a softmax trick.

    Standard attention materialises the seq x seq score matrix in HBM and then walks
    over it several times - write S, read it to softmax, write P, read it for P@V,
    plus any mask or dropout pass.  `standard_passes` is how many of those there are.

    Tiled attention keeps blocks of Q, K, V and O in SRAM and never writes the matrix.
    For each block row of Q it reads all of K and V, so its traffic is
    (Q + O) + (K + V) * ceil(seq / Br), where Br is set by what fits in SRAM.
    """
    per_head = seq * d_head * bytes_per
    qo = 2 * per_head
    kv = 2 * per_head
    scores = seq * seq * bytes_per                       # per head
    block = max(1, int(sram_bytes / (4 * d_head * bytes_per)))
    passes = max(1, math.ceil(seq / block))
    standard = n_heads * (qo + kv + standard_passes * scores)
    tiled = n_heads * (qo + kv * passes)
    flops = 4 * seq * seq * n_heads * d_head
    return {'seq': seq, 'block': block, 'passes': passes,
            'standard_gb': standard / 1e9, 'tiled_gb': tiled / 1e9,
            'ratio': standard / tiled,
            'standard_ai': flops / standard, 'tiled_ai': flops / tiled,
            'score_matrix_gb': n_heads * scores / 1e9,
            'resident_kb': sram_bytes / 1024,
            'memory_ratio': n_heads * scores / sram_bytes,
            'standard_passes': standard_passes}


def flash_scaling(seqs=(512, 2048, 8192, 32768), **kw):
    return [flash_io(seq=s, **kw) for s in seqs]


# ================================================================ 4. latency
def little(bandwidth_gbs=5300.0, latency_ns=600.0, bytes_per_request=128):
    """Little's law: to keep a pipe full you need bandwidth x latency in flight.

    This is why a GPU kernel that issues one load per thread and waits is slow even
    when nothing is wrong with it: the hardware needs thousands of requests in the
    air at once, and that number is a property of the machine, not of the code.
    """
    in_flight_bytes = bandwidth_gbs * 1e9 * latency_ns * 1e-9
    return {'bandwidth_gbs': bandwidth_gbs, 'latency_ns': latency_ns,
            'bytes_in_flight': in_flight_bytes,
            'requests_in_flight': in_flight_bytes / bytes_per_request,
            'bytes_per_request': bytes_per_request}


def strided_cost(n=1 << 22, strides=(1, 2, 4, 8, 16, 32), bytes_per=4, line=64):
    """Measured: the same number of useful bytes, read with growing stride.

    A cache line is fetched whole.  Reading every 16th float still fetches every line,
    so the useful fraction falls while the time does not - which is why layout and
    coalescing are performance decisions and not style ones.
    """
    rows = []
    a = np.ones(n, dtype=np.float32)
    for s in strides:
        v = a[::s]
        v.sum()
        t0 = time.perf_counter()
        for _ in range(20):
            v.sum()
        dt = (time.perf_counter() - t0) / 20
        useful = v.nbytes
        per_line = line // bytes_per                   # 16 floats in a 64-byte line
        # a line is fetched whole: with stride <= 16 every line is touched
        lines = math.ceil(n / per_line) if s <= per_line else len(v)
        fetched = lines * line
        rows.append({'stride': s, 'useful_mb': useful / 1e6, 'seconds': dt,
                     'useful_gbs': useful / dt / 1e9,
                     'fetched_mb': fetched / 1e6, 'efficiency': useful / fetched})
    return rows


# ================================================================ 5. checks
def check():
    m = Machine()
    # the ridge point is where the two limits cross, by construction
    assert abs(m.balance - m.flops / m.bw) < 1e-9
    assert m.bound(m.balance * 1e6 * 1.1, 1e6) == 'compute'
    assert m.bound(m.balance * 1e6 * 0.9, 1e6) == 'memory'
    rows, _ = roofline_table()
    by = {r['name']: r for r in rows}
    # decode is memory bound and prefill is not - the whole shape of inference
    assert by['decode: one linear (GEMV)']['bound'] == 'memory'
    assert by['prefill: one linear (GEMM)']['bound'] == 'compute'
    assert by['decode: the whole model']['bound'] == 'memory'
    # the pointwise kernels are always memory bound, by a lot
    assert by['RMSNorm']['intensity'] < 2 and by['residual add']['intensity'] < 1
    # batching moves a kernel along the roofline until it crosses the ridge
    bc, balance = batch_crossover()
    assert bc[0]['bound'] == 'memory' and bc[-1]['bound'] == 'compute'
    assert bc[0]['intensity'] < bc[-1]['intensity']
    assert bc[-1]['per_token_us'] < bc[0]['per_token_us']
    # tiling raises arithmetic intensity by exactly the tile size
    t = matmul_traffic(n=1024, tile=64)
    assert abs(t['ratio'] - 64) < 3.0, t['ratio']      # the tile size, up to the n^2 term
    assert t['tiled_ai'] > t['naive_ai']
    sweep = tile_sweep(n=1024)
    assert all(a['gb'] >= b['gb'] for a, b in zip(sweep, sweep[1:]))
    assert any(not r['fits'] for r in sweep)           # the cache sets the limit
    # FlashAttention: a modest traffic win, and an enormous capacity one
    small, big = flash_io(seq=1024), flash_io(seq=16384)
    assert big['ratio'] > small['ratio'] > 1.0
    assert big['tiled_ai'] > big['standard_ai']
    assert big['memory_ratio'] > 1000           # the matrix never fits in SRAM
    # fewer passes over the score matrix narrows the gap, as it should
    assert flash_io(standard_passes=2)['ratio'] < flash_io(standard_passes=6)['ratio']
    # measured: a random access costs more once the working set leaves the cache
    rnd = measure_random(sizes_kb=(64, 65536))
    assert rnd[-1]['ns_per_access'] > 2 * rnd[0]['ns_per_access'], rnd
    seq = measure_sequential(sizes_kb=(256, 65536))
    assert all(r['gbs'] > 1 for r in seq)
    # Little's law
    L = little()
    assert L['requests_in_flight'] > 1000


# ================================================================ 6. the story
def header(t):
    print('\n' + '=' * 72)
    print(t)
    print('=' * 72)


def main():
    header('1. The hierarchy, and what it costs to go one level down')
    print('  Vendor specifications for a current data-centre GPU, not measurements:\n')
    print('  level                      capacity        bandwidth      latency')
    for r in hierarchy_table():
        cap = ('-' if not r['bytes'] else
               f'{r["bytes"] / 1024:.0f} KB' if r['bytes'] < 1 << 20 else
               f'{r["bytes"] / 1024 ** 2:.0f} MB' if r['bytes'] < 1 << 30 else
               f'{r["bytes"] / 1024 ** 3:.0f} GB')
        slower = '' if not r['slower_than_above'] else f'   {r["slower_than_above"]:.0f}x slower'
        print(f'  {r["name"]:26} {cap:>9}   {r["gbs"]:9,.0f} GB/s   '
              f'{r["ns"]:6,.0f} ns{slower}')
    print('\n  Two things to notice.  Each level is roughly an order of magnitude')
    print('  slower and two orders larger than the one above it, and the drop from')
    print('  shared memory to HBM is the one every kernel in this series has been')
    print('  fighting.  The last two rows are the previous days: the host memory row')
    print('  is why HiCache tiers exist, and the RDMA row is why DeepEP has two')
    print('  different sets of kernels.')

    header('2. The same hierarchy, measured here')
    print('  No GPU in this VM, but the effect is not exotic - this machine has the')
    print('  same shape of hierarchy.  Two measurements, and the difference between')
    print('  them is the lesson.\n')
    print('  (a) Reading in order.  Summing an array is bandwidth-bound at every')
    print('      size, so this should fall off a cliff when the array stops fitting')
    print('      in cache:\n')
    seq = measure_sequential()
    peak = max(r['gbs'] for r in seq)
    for r in seq:
        bar = '#' * max(1, int(28 * r['gbs'] / peak))
        size = f'{r["kb"]} KB' if r['kb'] < 1024 else f'{r["kb"] // 1024} MB'
        print(f'      {size:>7}   {r["gbs"]:6.1f} GB/s  {bar}')
    drop = peak / min(r['gbs'] for r in seq)
    print(f'\n      It barely does - only {drop:.1f}x from the best size to the')
    print('      worst.  The hardware prefetcher sees a sequential read coming and')
    print('      starts it early, so most of the latency never happens.\n')
    print('  (b) Reading in a random order.  Same arrays, same bytes, an order the')
    print('      prefetcher cannot predict:\n')
    rnd = measure_random()
    worst = max(r['ns_per_access'] for r in rnd)
    for r in rnd:
        bar = '#' * max(1, int(28 * r['ns_per_access'] / worst))
        size = f'{r["kb"]} KB' if r['kb'] < 1024 else f'{r["kb"] // 1024} MB'
        print(f'      {size:>7}   {r["ns_per_access"]:6.2f} ns/access  {bar}')
    best = min(r['ns_per_access'] for r in rnd)
    print(f'\n      Now the hierarchy is visible: {best:.2f} ns per access while the')
    print(f'      working set fits in cache, {worst:.2f} ns when it does not, a')
    print(f'      factor of {worst / best:.0f}.  Measured on this CPU.\n')
    print('  That gap between (a) and (b) is most of practical performance work.')
    print('  Bandwidth is what the specification sells you; locality is what decides')
    print('  whether you get it.  Every structure in this series that looks like a')
    print('  pointer chase - a radix tree, a block table, a gather of expert weights')
    print('  - is paying column (b) prices, and every kernel that streams contiguous')
    print('  rows is paying column (a) prices.')

    header('3. The roofline: which side of the machine is the limit')
    rows, m = roofline_table()
    print(f'  A machine is two numbers: {m.flops / 1e12:.0f} TFLOP/s and '
          f'{m.bw / 1e9:.0f} GB/s.')
    print(f'  Their ratio is the machine balance - {m.balance:.0f} FLOPs per byte.')
    print('  A kernel that does fewer FLOPs per byte than that cannot be')
    print('  compute-bound no matter how it is written.\n')
    print('  kernel                              FLOPs/byte   bound     time')
    for r in rows:
        print(f'  {r["name"]:36} {r["intensity"]:10.2f}   {r["bound"]:8} '
              f'{r["ms"]:7.2f} ms')
    print('\n  Read the first two rows together.  The same linear layer, the same')
    print('  weights: as a matrix-vector product in decode it is memory-bound and as')
    print('  a matrix-matrix product in prefill it is compute-bound.  Nothing about')
    print('  the kernel changed; the number of rows did.')
    print('\n  And the pointwise rows are the ones people under-estimate: RMSNorm and')
    print('  a residual add do almost no arithmetic, so their time is entirely the')
    print('  traffic.  That is why fusing them into their neighbours - which removes')
    print('  a read and a write, not a multiply - is worth so much.')

    header('4. Batching is how you move along the roofline')
    bc, balance = batch_crossover()
    print(f'  One linear layer, machine balance {balance:.0f} FLOPs/byte:\n')
    print('  batch   FLOPs/byte   bound     layer time   per token')
    for r in bc:
        print(f'  {r["batch"]:5d}   {r["intensity"]:10.2f}   {r["bound"]:8}  '
              f'{r["ms"]:8.3f} ms  {r["per_token_us"]:8.1f} us')
    print('\n  The weights are read once whatever the batch size, so the arithmetic')
    print('  per byte grows linearly with the batch until it crosses the ridge.  That')
    print('  is the whole economics of serving: batch 1 wastes the machine, and the')
    print('  cost per token falls by two orders of magnitude on the way up.  It is')
    print('  also why speculative decoding stops paying at large batch - the day on')
    print('  decoding measured the same curve from the other side.')

    header('5. Tiling: the one way to raise arithmetic intensity')
    t = matmul_traffic()
    print('  A kernel cannot choose its FLOPs - the algorithm fixes those.  It can')
    print('  choose how many times it re-reads the same bytes.\n')
    print(f'  A {t["n"]}x{t["n"]} matmul does {2 * t["n"] ** 3 / 1e9:.0f} GFLOP either way:\n')
    print(f'    no tiling      {t["naive_bytes"] / 1e9:8.1f} GB moved   '
          f'{t["naive_ai"]:6.2f} FLOPs/byte')
    print(f'    tiles of {t["tile"]:3d}   {t["tiled_bytes"] / 1e9:8.1f} GB moved   '
          f'{t["tiled_ai"]:6.2f} FLOPs/byte   {t["ratio"]:.0f}x less traffic')
    print('\n  The factor is exactly the tile size, because each tile of B is read')
    print('  once per tile-row of A instead of once per row.  And the tile size is')
    print('  not a free parameter - three tiles have to fit in the cache at once:\n')
    print('  tile   fits in cache   bytes moved   FLOPs/byte')
    for r in tile_sweep():
        print(f'  {r["tile"]:4d}   {"yes" if r["fits"] else "NO":>13}   '
              f'{r["gb"]:8.1f} GB   {r["ai"]:10.2f}')
    print(f'\n  So the optimum is the largest tile that fits, and that is the whole')
    print(f'  design rule: T = sqrt(cache / 3) = {t["max_tile_for_cache"]}.  Every')
    print('  blocked kernel in every library is this calculation plus the details.')

    header('6. FlashAttention is this argument, applied to attention')
    print('  Day 52 showed the online-softmax identity: the same answer in one pass')
    print('  over the keys, agreeing to the last bit.  What it did not say is why')
    print('  that is worth doing, and the answer is the table above.\n')
    f = flash_io()
    print(f'  seq {f["seq"]}, 32 heads of 128, {f["standard_passes"]} passes over the')
    print('  score matrix in the untiled version:\n')
    print(f'    the score matrix itself         {f["score_matrix_gb"]:7.2f} GB   '
          f'allocated in HBM')
    print(f'    standard attention moves        {f["standard_gb"]:7.2f} GB   '
          f'{f["standard_ai"]:7.1f} FLOPs/byte')
    print(f'    tiled (FlashAttention) moves    {f["tiled_gb"]:7.2f} GB   '
          f'{f["tiled_ai"]:7.1f} FLOPs/byte')
    print(f'    traffic ratio                   {f["ratio"]:7.1f}x')
    print('\n  Two honest observations about that number.\n')
    print('  First, it is not a factor of ten.  At d_head 128 and 160 KB of shared')
    print(f'  memory the block is {f["block"]} rows, so K and V get re-read '
          f'{f["passes"]} times, and')
    print('  the tiled traffic is within a small factor of the untiled one.  The')
    print('  traffic win depends on d_head^2 / SRAM, which is why it was larger in')
    print('  the original paper (d_head 64) than it is on a model with 128.\n')
    print('  Second, the traffic was never the main prize.  What tiling actually buys')
    print(f'  is that the {f["score_matrix_gb"]:.2f} GB in the first row is never')
    print(f'  allocated at all - {f["memory_ratio"]:,.0f} times the shared memory it')
    print('  runs in.  That is a capacity argument, not a bandwidth one, and it is')
    print('  the reason long contexts are servable: standard attention needs memory')
    print('  that grows with seq squared, and tiled attention does not.\n')
    print('  The backward pass is the same trick again.  It does not store the score')
    print('  matrix either: it keeps the log-sum-exp per row - one number instead of')
    print('  seq numbers - and recomputes the scores from Q and K.  Recomputation is')
    print('  cheaper than the round trip, and the roofline table is why.\n')
    print('  How it scales with the context:\n')
    print('  seq      score matrix   standard GB   tiled GB   ratio')
    for r in flash_scaling():
        print(f'  {r["seq"]:6d}   {r["score_matrix_gb"]:10.2f} GB   '
              f'{r["standard_gb"]:10.2f}   {r["tiled_gb"]:8.2f}   {r["ratio"]:5.1f}x')
    print('\n  Both columns grow with seq squared - the arithmetic does, so the')
    print('  traffic must.  What changes is that one of them has to be *resident* and')
    print('  the other does not.')

    header('7. Latency: bandwidth is not enough on its own')
    L = little()
    print('  A memory system has a second number that specifications rarely put next')
    print('  to the first: how long one access takes.  To keep the pipe full you need')
    print('  bandwidth x latency bytes in flight at all times.\n')
    print(f'    {L["bandwidth_gbs"]:,.0f} GB/s x {L["latency_ns"]:.0f} ns = '
          f'{L["bytes_in_flight"] / 1024:,.0f} KB in flight')
    print(f'    at {L["bytes_per_request"]} bytes per request, that is '
          f'{L["requests_in_flight"]:,.0f} requests at once')
    print('\n  Which is why a kernel that issues one load and waits is slow even when')
    print('  it is perfectly coalesced and does no redundant work: the hardware needs')
    print('  thousands of outstanding requests, and only many threads - or one thread')
    print('  unrolled over many loads - can supply them.')
    print('\n  And the other half of using bandwidth well is that a cache line is')
    print('  fetched whole.  Measured here, reading with a growing stride:\n')
    print('  stride   useful MB   fetched MB   useful %   time       useful GB/s')
    for r in strided_cost():
        print(f'  {r["stride"]:6d}   {r["useful_mb"]:9.2f}   {r["fetched_mb"]:10.2f}   '
              f'{r["efficiency"]:7.1%}   {r["seconds"] * 1e3:5.2f} ms   '
              f'{r["useful_gbs"]:11.1f}')
    print('\n  Up to stride 16 the fetched column does not move at all: every cache')
    print('  line is touched whatever the stride, so the useful fraction collapses')
    print('  while the work does not - and the time even goes up, because the same')
    print('  16 MB is fetched for a sixteenth of the payload.  That is the whole')
    print('  content of "coalescing" on a GPU, and on a CPU it is why an array of')
    print('  structs is slower than a struct of arrays.  Same hardware fact.')

    header('8. In PyTorch')
    print('  The profiler answers the only question that matters here - how many')
    print('  bytes, not how many FLOPs:\n')
    print('    with torch.profiler.profile(activities=[CUDA], with_stack=True) as p:')
    print('        model(x)')
    print('    print(p.key_averages().table(sort_by="self_cuda_time_total"))\n')
    print('    torch.cuda.max_memory_allocated()       # the peak you actually hit')
    print('    torch.cuda.memory._record_memory_history()   # and where it went\n')
    print('  And the single highest-leverage tool is fusion, because section 3 says')
    print('  the pointwise kernels are pure traffic:\n')
    print('    torch.compile(model)        # fuses norm + add + activation chains')
    print('    F.scaled_dot_product_attention(q, k, v)   # dispatches to a tiled kernel')
    print('    with sdpa_kernel(SDPBackend.FLASH_ATTENTION): ...\n')
    print('  Reading a profile with the roofline in hand is a different experience:')
    print('  a kernel at 20% of peak FLOPs is not necessarily badly written, it may')
    print('  be at 95% of peak bandwidth, and the fix is to give it more arithmetic')
    print('  per byte rather than more threads.')

    header('9. In JAX')
    print('  XLA fuses by default, which removes most of the pointwise traffic')
    print('  without being asked, and it will tell you what it decided:\n')
    print('    compiled = jax.jit(f).lower(x).compile()')
    print('    compiled.memory_analysis()      # bytes in, out, and temporaries')
    print('    compiled.cost_analysis()        # the FLOP count to divide by\n')
    print('  Those two calls are a roofline for your function, computed by the')
    print('  compiler before you run anything.\n')
    print('    jax.jit(f, donate_argnums=(0,))   # reuse the input buffer: fewer bytes')
    print('    jax.profiler.trace("/tmp/tb")     # timeline and memory profile\n')
    print('  The deeper difference from PyTorch is where the tiling lives.  In')
    print('  PyTorch you call a kernel somebody tiled for you, or you write one in')
    print('  Triton.  In JAX the compiler tiles the fused region itself, and when')
    print('  that is not good enough you drop to Pallas - which is the same Triton')
    print('  model: you choose the block sizes, and the constraint you are choosing')
    print('  against is the one in section 5.')

    header('10. Assertions')
    check()
    print('all assertions passed')


if __name__ == '__main__':
    main()
