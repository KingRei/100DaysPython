"""Day 62.5 - the memory hierarchy: why some kernels never get faster.

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
# Everything here is public.  Each row is tagged with where its numbers come from:
#
#   'spec'     - published by the vendor: product pages, datasheets, the CDNA 3 and
#                Hopper architecture whitepapers, the PCIe and Ethernet standards.
#   'estimate' - my own order-of-magnitude figure, derived from unit counts and
#                clocks or from published third-party microbenchmarks.  On-chip
#                bandwidths and access latencies are almost never in a datasheet, so
#                every one of those is an estimate and should be read as "about".
MACHINES = {
    'MI300X': {
        'vendor': 'AMD, CDNA 3',
        'tflops_bf16': 1307.0,                       # spec
        'levels': [
            # (name, capacity, GB/s, ns, capacity source, bandwidth source)
            ('registers (VGPR per CU)', 512 * 1024, 100_000.0, 1, 'spec', 'estimate'),
            ('LDS / shared memory per CU', 64 * 1024, 20_000.0, 20, 'spec', 'estimate'),
            ('L2 per XCD (8 of them)', 4 * 1024 * 1024, 10_000.0, 200, 'spec', 'estimate'),
            ('Infinity Cache (MALL)', 256 * 1024 * 1024, 17_000.0, 300, 'spec', 'spec'),
            ('HBM3', 192 * 1024 ** 3, 5_300.0, 600, 'spec', 'spec'),
            ('host memory over PCIe 5 x16', 0, 64.0, 2_000, '-', 'spec'),
            ('another node over RDMA', 0, 50.0, 10_000, '-', 'spec'),
        ],
    },
    'H200 SXM': {
        'vendor': 'NVIDIA, Hopper',
        'tflops_bf16': 990.0,                        # spec, dense; same die as H100
        'levels': [
            ('registers (per SM)', 256 * 1024, 100_000.0, 1, 'spec', 'estimate'),
            ('shared memory per SM', 228 * 1024, 20_000.0, 20, 'spec', 'estimate'),
            ('L2 cache', 50 * 1024 * 1024, 7_000.0, 200, 'spec', 'estimate'),
            ('HBM3e', 141 * 1024 ** 3, 4_800.0, 600, 'spec', 'spec'),
            ('host memory over PCIe 5 x16', 0, 64.0, 2_000, '-', 'spec'),
            ('another node over RDMA', 0, 50.0, 10_000, '-', 'spec'),
        ],
    },
    'H100 SXM': {
        'vendor': 'NVIDIA, Hopper',
        'tflops_bf16': 990.0,                        # spec, dense
        'levels': [
            ('registers (per SM)', 256 * 1024, 100_000.0, 1, 'spec', 'estimate'),
            ('shared memory per SM', 228 * 1024, 20_000.0, 20, 'spec', 'estimate'),
            ('L2 cache', 50 * 1024 * 1024, 7_000.0, 200, 'spec', 'estimate'),
            ('HBM3', 80 * 1024 ** 3, 3_350.0, 600, 'spec', 'spec'),
            ('host memory over PCIe 5 x16', 0, 64.0, 2_000, '-', 'spec'),
            ('another node over RDMA', 0, 50.0, 10_000, '-', 'spec'),
        ],
    },
}

SOURCES = [
    'AMD Instinct MI300X product page and data sheet (192 GB HBM3, 5.3 TB/s, '
    '1307 TFLOP/s bf16, 256 MB Infinity Cache at ~17 TB/s)',
    'AMD CDNA 3 architecture whitepaper and the CDNA 3 ISA guide '
    '(512 KB VGPR and 64 KB LDS per CU, 4 MB L2 per XCD, 8 XCDs)',
    'NVIDIA H100 and H200 data sheets and the Hopper architecture whitepaper '
    '(H100 80 GB HBM3 at 3.35 TB/s, H200 141 GB HBM3e at 4.8 TB/s, the same die at '
    '989 TFLOP/s bf16 dense, 50 MB L2, 228 KB shared memory per SM)',
    'PCIe 5.0 x16 is 64 GB/s each way by the standard; 50 GB/s is a 400 Gb/s NIC',
    'latencies and on-chip bandwidths: my estimates, in line with published '
    'third-party microbenchmarks - treat them as magnitudes, not measurements',
]


def hierarchy_table(machine='MI300X'):
    """Capacity, bandwidth and latency, with each level relative to the one above."""
    rows = []
    levels = MACHINES[machine]['levels']
    for i, (name, cap, bw, lat, cap_src, bw_src) in enumerate(levels):
        prev = levels[i - 1][2] if i else None
        rows.append({'machine': machine, 'name': name, 'bytes': cap, 'gbs': bw,
                     'ns': lat, 'capacity_source': cap_src, 'bandwidth_source': bw_src,
                     'slower_than_above': bw and prev and prev / bw})
    return rows


def hbm(machine='MI300X'):
    """The level everything in this series is actually fighting."""
    for name, cap, bw, lat, _c, _b in MACHINES[machine]['levels']:
        if name.startswith('HBM'):
            return {'machine': machine, 'bytes': cap, 'gbs': bw, 'ns': lat}
    raise KeyError(machine)


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
    """A machine is two numbers: how fast it computes and how fast it reads.

    Both come from the table above, so the roofline below is the same specification
    sheet seen from a different angle rather than a second set of numbers.
    """

    def __init__(self, name='MI300X', tflops=None, bandwidth_gbs=None):
        spec = MACHINES.get(name)
        self.flops = (tflops if tflops is not None else spec['tflops_bf16']) * 1e12
        self.bw = (bandwidth_gbs if bandwidth_gbs is not None else hbm(name)['gbs']) * 1e9
        self.name = name
        self.vendor = spec['vendor'] if spec else ''  

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


def flash_io(seq=4096, d_head=128, n_heads=32, sram_bytes=64 * 1024, bytes_per=2,
             standard_passes=4, cache_bytes=256 * 1024 ** 2):
    """FlashAttention, read as an IO problem rather than a softmax trick.

    Standard attention materialises the seq x seq score matrix in HBM and walks over
    it several times - write S, read it to softmax, write P, read it for P@V, plus any
    mask or dropout pass.  `standard_passes` counts those.

    Tiled attention keeps blocks of Q, K, V and O in SRAM and never writes the matrix.
    For each block row of Q it re-reads K and V, so the traffic to the next level down
    is (Q + O) + (K + V) * ceil(seq / Br), with Br set by what fits in SRAM.  Whether
    that next level is HBM or a cache is what `cache_bytes` decides: if K and V for a
    head fit in the L2 or Infinity Cache, the re-reads never reach HBM at all.
    """
    per_head = seq * d_head * bytes_per
    qo = 2 * per_head
    kv = 2 * per_head
    scores = seq * seq * bytes_per                       # per head
    block = max(1, int(sram_bytes / (4 * d_head * bytes_per)))
    passes = max(1, math.ceil(seq / block))
    standard = n_heads * (qo + kv + standard_passes * scores)
    tiled = n_heads * (qo + kv * passes)
    fits = kv <= cache_bytes
    tiled_hbm = n_heads * (qo + kv) if fits else tiled   # re-reads caught by the cache
    flops = 4 * seq * seq * n_heads * d_head
    return {'seq': seq, 'block': block, 'passes': passes,
            'standard_gb': standard / 1e9, 'tiled_gb': tiled / 1e9,
            'tiled_hbm_gb': tiled_hbm / 1e9,
            'ratio': standard / tiled, 'ratio_hbm': standard / tiled_hbm,
            'standard_ai': flops / standard, 'tiled_ai': flops / tiled_hbm,
            'score_matrix_gb': n_heads * scores / 1e9,
            'kv_per_head_mb': kv / 1e6, 'kv_fits_cache': fits,
            'cache_mb': cache_bytes / 1e6, 'sram_kb': sram_bytes / 1024,
            'memory_ratio': n_heads * scores / sram_bytes,
            'standard_passes': standard_passes}


def flash_scaling(seqs=(512, 2048, 8192, 32768), **kw):
    return [flash_io(seq=s, **kw) for s in seqs]


# ================================================================ 4. latency
def little(bandwidth_gbs=None, latency_ns=600.0, bytes_per_request=128,
           machine='MI300X'):
    """Little's law: to keep a pipe full you need bandwidth x latency in flight.

    This is why a GPU kernel that issues one load per thread and waits is slow even
    when nothing is wrong with it: the hardware needs thousands of requests in the
    air at once, and that number is a property of the machine, not of the code.
    """
    bandwidth_gbs = bandwidth_gbs if bandwidth_gbs is not None else hbm(machine)['gbs']
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
    # every row declares where its numbers came from, and nothing is unsourced
    for name in MACHINES:
        for r in hierarchy_table(name):
            assert r['capacity_source'] in ('spec', '-')
            assert r['bandwidth_source'] in ('spec', 'estimate')
    assert any('MI300X' in s for s in SOURCES) and any('H200' in s for s in SOURCES)
    # H200 is the same die as H100 with more memory: same peak, lower balance
    assert Machine('H200 SXM').flops == Machine('H100 SXM').flops
    assert Machine('H200 SXM').balance < Machine('MI300X').balance < Machine('H100 SXM').balance
    assert list(MACHINES)[0] == 'MI300X'
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
    big = flash_io(seq=16384)
    assert big['memory_ratio'] > 1000              # the matrix never fits in SRAM
    assert big['ratio_hbm'] >= big['ratio']        # the cache absorbs the re-reads
    assert flash_io(sram_bytes=228 * 1024)['ratio'] > flash_io(sram_bytes=64 * 1024)['ratio']
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
    print('  Two named parts, from public sources only.  Each number is tagged with')
    print('  where it comes from: "spec" is published by the vendor, "est." is my own')
    print('  order-of-magnitude figure.  On-chip bandwidths and latencies are almost')
    print('  never in a datasheet, so those are all estimates.\n')
    for name in MACHINES:
        spec = MACHINES[name]
        print(f'  {name} ({spec["vendor"]}, {spec["tflops_bf16"]:.0f} TFLOP/s bf16)\n')
        print('    level                            capacity        bandwidth         latency')
        for r in hierarchy_table(name):
            cap = ('-' if not r['bytes'] else
                   f'{r["bytes"] / 1024:.0f} KB' if r['bytes'] < 1 << 20 else
                   f'{r["bytes"] / 1024 ** 2:.0f} MB' if r['bytes'] < 1 << 30 else
                   f'{r["bytes"] / 1024 ** 3:.0f} GB')
            cs = '' if r['capacity_source'] == '-' else f' ({r["capacity_source"][:4]})'
            bs = ' (est.)' if r['bandwidth_source'] == 'estimate' else ' (spec)'
            print(f'    {r["name"]:30} {cap:>9}{cs:<7} {r["gbs"]:9,.0f} GB/s{bs:<8}'
                  f'{r["ns"]:7,.0f} ns (est.)')
        print()
    print('  Three things to notice.  Each level is roughly an order of magnitude')
    print('  slower and two orders larger than the one above it, and the drop from')
    print('  shared memory to HBM is the one every kernel in this series has been')
    print('  fighting.\n')
    print('  The parts differ in ways that matter later.  The MI300X has the most HBM')
    print('  at the most bandwidth, and a 256 MB Infinity Cache between L2 and HBM')
    print('  that the Hopper parts do not have.  Both NVIDIA parts have a much larger')
    print('  L2 (50 MB) and more shared memory per SM (228 KB against 64 KB) - and')
    print('  shared memory is exactly what sets the block size in the tiling sections')
    print('  below.  H200 is the same die as H100 with more and faster memory, which')
    print('  is why it has the same peak arithmetic and a different balance:\n')
    for name in MACHINES:
        mm = Machine(name)
        print(f'    {name:10} {mm.flops / 1e12:6.0f} TFLOP/s   '
              f'{mm.bw / 1e9:6,.0f} GB/s   balance {mm.balance:5.0f} FLOPs/byte')
    print('\n  Lower is better for everything memory-bound, so on that column the')
    print('  order is H200, then MI300X, then H100 - which is the whole argument for')
    print('  the refresh parts: the arithmetic did not move, the memory did.')
    print('\n  The last two rows of each are the previous days: the host-memory row is')
    print('  why HiCache has tiers, and the RDMA row is why DeepEP ships two different')
    print('  sets of kernels.  Everything below uses the MI300X numbers unless it says')
    print('  otherwise.')
    print('\n  Where these came from:\n')
    for src in SOURCES:
        print(f'    - {src}')

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
    print(f'  A machine is two numbers.  The {m.name}: {m.flops / 1e12:.0f} TFLOP/s and '
          f'{m.bw / 1e9:.0f} GB/s.')
    print(f'  Their ratio is the machine balance - {m.balance:.0f} FLOPs per byte.')
    print('  (' + ', '.join(f'{n} {Machine(n).balance:.0f}' for n in MACHINES) +
          ' - the same shape on all three,')
    print('  and close enough that nothing below changes qualitatively.)')
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
    print('  that is worth doing, and the answer is the table above.')
    f = flash_io()
    print(f'\n  seq {f["seq"]}, 32 heads of 128, {f["standard_passes"]} passes over the '
          f'score matrix in the')
    print(f'  untiled version, {f["sram_kb"]:.0f} KB of LDS:\n')
    print(f'    the score matrix itself         {f["score_matrix_gb"]:7.2f} GB   '
          f'allocated in HBM')
    print(f'    standard attention moves        {f["standard_gb"]:7.2f} GB   '
          f'{f["standard_ai"]:7.1f} FLOPs/byte')
    print(f'    tiled, counting every level     {f["tiled_gb"]:7.2f} GB   '
          f'{f["ratio"]:7.1f}x less')
    print(f'    tiled, what reaches HBM         {f["tiled_hbm_gb"]:7.2f} GB   '
          f'{f["ratio_hbm"]:7.1f}x less')
    print('\n  Three honest observations about those numbers.\n')
    print('  First, the traffic win is not a factor of ten - counted naively it is')
    print(f'  barely anything.  With {f["sram_kb"]:.0f} KB of LDS and d_head 128 the '
          f'block is {f["block"]} rows,')
    print(f'  so K and V are re-read {f["passes"]} times and the tiled traffic lands on '
          f'top of the')
    print('  untiled one.  The saving scales with SRAM over d_head squared, which is')
    print('  why it looked larger in the original paper (d_head 64) than it does on a')
    print('  model with 128.\n')
    print('  Second, those re-reads mostly never reach HBM.  K and V for one head are')
    print(f'  {f["kv_per_head_mb"]:.1f} MB against a {f["cache_mb"]:.0f} MB Infinity '
          f'Cache, so the repeated passes are')
    print(f'  cache hits, and the traffic that actually crosses to HBM is '
          f'{f["tiled_hbm_gb"]:.2f} GB -')
    print(f'  {f["ratio_hbm"]:.1f}x less than untiled.  A model that counts only one '
          f'level will')
    print('  mispredict, which is the lesson of this whole day applied to the day.\n')
    print('  Third, and most important: the traffic was never the main prize.  What')
    print(f'  tiling buys is that the {f["score_matrix_gb"]:.2f} GB in the first row is '
          f'never allocated at')
    print(f'  all - {f["memory_ratio"]:,.0f} times the LDS the kernel runs in.  That is '
          f'a capacity')
    print('  argument, and it is why long contexts are servable: untiled attention')
    print('  needs memory that grows with seq squared, and tiled attention does not.\n')
    print('  The backward pass is the same trick again: it stores the log-sum-exp per')
    print('  row - one number instead of seq numbers - and recomputes the scores from')
    print('  Q and K, because recomputation is cheaper than the round trip.\n')
    print('  How it scales with the context:\n')
    print('  seq      score matrix   standard GB   tiled to HBM   ratio   KV in cache')
    for r in flash_scaling():
        print(f'  {r["seq"]:6d}   {r["score_matrix_gb"]:10.2f} GB   '
              f'{r["standard_gb"]:10.2f}   {r["tiled_hbm_gb"]:12.2f}   '
              f'{r["ratio_hbm"]:5.1f}x   {"yes" if r["kv_fits_cache"] else "no":>11}')
    print('\n  Watch the last column.  While K and V for a head fit in the Infinity')
    print('  Cache the re-reads are absorbed and tiling looks excellent; once they')
    print('  stop fitting, the re-reads reach HBM and the block size stops being a')
    print('  detail.  That is the whole day in one table: the same code, a different')
    print('  level of the hierarchy, a different answer.')


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
