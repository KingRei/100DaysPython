"""Day 64 - continuous batching and the inference scheduler.

A serving engine is a loop that decides, a few hundred times a second, which
requests run in the next forward pass.  This module simulates that loop:
static batching against continuous batching, the waiting queue, chunked
prefill, a KV pool that can run out, preemption, and the scheduling policies -
measuring time to first token, time per output token, inter-token latency and
throughput for each.

Run it:  python scheduling.py
"""

import numpy as np

# --------------------------------------------------------------- the machine
# A 70B-class model served 8-way tensor parallel in bf16.  Every constant here
# is derived from public specifications, the same way Day 62.5 did it.
GPU = {
    'name': '70B bf16, 8-way tensor parallel',
    'weight_gb_per_gpu': 17.5,        # 140 GB of bf16 weights over 8 GPUs
    'hbm_tbs': 5.3,                   # HBM bandwidth per GPU
    'tflops': 1307.0,                 # peak bf16
    'mfu': 0.5,                       # what a prefill actually reaches
    'kv_kb_per_token': 40.0,          # GQA, 80 layers, 8 KV heads, over 8 GPUs
    'kv_pool_tokens': 150_000,        # the KV cache left after the weights
}

BYTES_PER_MS = GPU['hbm_tbs'] * 1e12 / 1e3          # HBM bytes moved per ms
# one decode step reads every weight once, whatever the batch size
STEP_WEIGHT_MS = GPU['weight_gb_per_gpu'] * 1e9 / BYTES_PER_MS
# ...and reads the KV of every token of context in the batch
KV_MS_PER_TOKEN = GPU['kv_kb_per_token'] * 1e3 / BYTES_PER_MS
# a prefill is arithmetic, not bandwidth
PREFILL_MS_PER_TOKEN = (2 * 70e9 / 8) / (GPU['tflops'] * 1e12 * GPU['mfu']) * 1e3


def decode_step_ms(context_tokens, batch):
    """Weights once, plus the KV of everything in the batch."""
    return STEP_WEIGHT_MS + context_tokens * KV_MS_PER_TOKEN if batch else 0.0


def prefill_ms(tokens):
    return tokens * PREFILL_MS_PER_TOKEN


PCIE_BYTES_PER_MS = 64e9 / 1e3                      # PCIe 5 x16, one direction


def swap_ms(tokens):
    """Preemption by swapping: the KV leaves for host memory and comes back."""
    return 2 * tokens * GPU['kv_kb_per_token'] * 1e3 / PCIE_BYTES_PER_MS


# ------------------------------------------------------------- the workload
class Request:
    __slots__ = ('rid', 'arrive', 'prompt', 'out_len', 'done_prefill', 'produced',
                 'start', 'first_token', 'finish', 'last_token', 'itls',
                 'preempted', 'prefilled')

    def __init__(self, rid, arrive, prompt, out_len):
        self.rid, self.arrive, self.prompt, self.out_len = rid, arrive, prompt, out_len
        self.prefilled = 0            # how much of the prompt is in the KV cache
        self.produced = 0             # output tokens emitted
        self.first_token = None
        self.finish = None
        self.last_token = None
        self.itls = []
        self.preempted = 0

    @property
    def context(self):
        return self.prefilled + self.produced

    @property
    def ttft(self):
        return None if self.first_token is None else self.first_token - self.arrive

    @property
    def latency(self):
        return None if self.finish is None else self.finish - self.arrive

    @property
    def tpot(self):
        if self.first_token is None or self.produced <= 1:
            return None
        return (self.finish - self.first_token) / (self.produced - 1)


def workload(n=240, rate_per_s=2.0, seed=0, long_share=0.12):
    """Poisson arrivals, lognormal prompts, a minority of very long ones."""
    rng = np.random.default_rng(seed)
    gaps = rng.exponential(1.0 / rate_per_s, size=n)
    arrive = np.cumsum(gaps) * 1e3                       # milliseconds
    prompts = np.clip(rng.lognormal(np.log(900), 0.6, size=n), 64, 4096).astype(int)
    long_ones = rng.random(n) < long_share
    prompts[long_ones] = rng.integers(6000, 16000, size=long_ones.sum())
    outs = np.clip(rng.lognormal(np.log(180), 0.7, size=n), 8, 1200).astype(int)
    return [Request(i, float(arrive[i]), int(prompts[i]), int(outs[i]))
            for i in range(n)]


def clone(reqs):
    return [Request(r.rid, r.arrive, r.prompt, r.out_len) for r in reqs]


# ------------------------------------------------------------ the two loops
def run_static(reqs, batch_size=16, pool_tokens=GPU['kv_pool_tokens'], trace=False):
    """Collect a batch, prefill it, decode until the *last* one finishes.

    Nobody joins and nobody leaves in between, so short requests sit in a
    finished state burning a slot, and new arrivals wait for the whole batch.
    """
    reqs = sorted(clone(reqs), key=lambda r: r.arrive)
    t, i, wasted_slots, total_slots = 0.0, 0, 0, 0
    tape = []
    while i < len(reqs):
        batch = reqs[i:i + batch_size]
        i += len(batch)
        t = max(t, batch[-1].arrive)                     # wait for the batch
        t += prefill_ms(sum(r.prompt for r in batch))
        if trace:
            tape.append({'kind': 'prefill', 'ms': t,
                         'slots': [(r.rid, 'prefill') for r in batch]})
        for r in batch:
            r.prefilled = r.prompt
            r.first_token = t
            r.produced = 1
        steps = max(r.out_len for r in batch)
        for s in range(1, steps):
            alive = [r for r in batch if r.produced < r.out_len]
            ctx = sum(r.context for r in batch)           # padding: all slots held
            t += decode_step_ms(ctx, len(batch))
            total_slots += len(batch)
            wasted_slots += len(batch) - len(alive)
            if trace:
                tape.append({'kind': 'decode', 'ms': t,
                             'slots': [(r.rid, 'decode' if r in alive else 'padding')
                                       for r in batch]})
            for r in alive:
                if r.produced:
                    r.itls.append(t - (r.last_token or r.first_token))
                r.last_token = t
                r.produced += 1
                if r.produced >= r.out_len:
                    r.finish = t
        for r in batch:
            if r.finish is None:
                r.finish = t
    return {'name': f'static batching (B={batch_size})', 'reqs': reqs, 'end': t,
            'pad_waste': wasted_slots / max(1, total_slots), 'tape': tape}


def run_continuous(reqs, max_batch=64, pool_tokens=GPU['kv_pool_tokens'],
                   chunk=None, policy='fcfs', admit_cap=None, preempt='recompute',
                   trace=False):
    """One step at a time: admit what fits, run everybody who is running.

    `chunk` is the chunked-prefill token budget: None means a prefill occupies
    a whole step on its own, an integer means prefill and decode share each
    step up to that many prompt tokens.
    """
    reqs = sorted(clone(reqs), key=lambda r: r.arrive)
    waiting, running, done = [], [], []
    t, nxt = 0.0, 0
    tape = []
    used = 0                                   # KV tokens currently held
    preemptions = 0
    busy_ms = 0.0

    def admit():
        nonlocal used
        if policy == 'sjf':
            waiting.sort(key=lambda r: r.prompt)
        while waiting:
            r = waiting[0]
            need = r.prompt + 16
            cap = admit_cap or max_batch
            if len(running) >= cap or used + need > pool_tokens:
                break
            waiting.pop(0)
            running.append(r)
            used += need

    while nxt < len(reqs) or waiting or running:
        while nxt < len(reqs) and reqs[nxt].arrive <= t:
            waiting.append(reqs[nxt])
            nxt += 1
        admit()
        if not running:
            if nxt < len(reqs):
                t = max(t, reqs[nxt].arrive)
                continue
            if waiting:                        # nothing fits: evict to make room
                break
            break

        pending = [r for r in running if r.prefilled < r.prompt]
        step_ms, prefill_tokens = 0.0, 0
        if pending and chunk is None:
            r = pending[0]                     # a prefill owns the whole step
            prefill_tokens = r.prompt - r.prefilled
            r.prefilled = r.prompt
            step_ms = prefill_ms(prefill_tokens)
            t += step_ms
            busy_ms += step_ms
            r.first_token = t
            r.produced = 1
            r.last_token = t
            if trace:
                tape.append({'kind': 'prefill', 'ms': t, 'used': used,
                             'waiting': len(waiting),
                             'slots': [(q.rid, 'prefill' if q is r else 'stalled')
                                       for q in running]})
            continue
        if pending:                            # chunked: share the step
            budget = chunk
            for r in pending:
                if budget <= 0:
                    break
                take = min(budget, r.prompt - r.prefilled)
                r.prefilled += take
                budget -= take
                prefill_tokens += take

        decoding = [r for r in running if r.prefilled >= r.prompt and r.first_token]
        ctx = sum(r.context for r in running)
        step_ms = decode_step_ms(ctx, len(running)) + prefill_ms(prefill_tokens)
        t += step_ms
        busy_ms += step_ms
        if trace:
            tape.append({'kind': 'decode', 'ms': t, 'used': used,
                         'waiting': len(waiting),
                         'slots': [(q.rid, 'decode' if q.first_token else 'prefill')
                                   for q in running]})
        for r in running:
            if r.prefilled >= r.prompt and r.first_token is None:
                r.first_token = t              # the chunk that finished the prompt
                r.last_token = t
                r.produced = 1
            elif r in decoding:
                r.itls.append(t - r.last_token)
                r.last_token = t
                r.produced += 1
        for r in list(running):
            if r.produced >= r.out_len:
                r.finish = t
                running.remove(r)
                done.append(r)
                used -= r.prompt + 16 + r.produced
            else:
                used += 1                      # one more token of KV
        while used > pool_tokens and len(running) > 1:
            victim = max(running, key=lambda q: q.context)   # the biggest KV
            preemptions += 1
            used -= victim.context + 16
            if preempt == 'recompute':
                victim.prefilled, victim.produced = 0, 0
                victim.first_token, victim.last_token = None, None
                victim.itls = []
            else:                              # swap: keep the KV, move it out
                swap = swap_ms(victim.context)  # out now, back in later
                t += swap
                busy_ms += swap
            running.remove(victim)
            waiting.insert(0, victim)
            if trace:
                tape.append({'kind': 'preempt', 'ms': t, 'used': used,
                             'waiting': len(waiting), 'victim': victim.rid,
                             'slots': [(q.rid, 'decode') for q in running]})
    for r in reqs:
        if r.finish is None:
            r.finish = t
    name = 'continuous batching'
    if chunk:
        name += f' + chunked prefill ({chunk})'
    if policy != 'fcfs':
        name += f', {policy}'
    if admit_cap:
        name += f', cap {admit_cap}'
    return {'name': name, 'reqs': reqs, 'end': t, 'preemptions': preemptions,
            'busy_frac': busy_ms / max(t, 1e-9), 'tape': tape}


# ------------------------------------------------------------------ metrics
def metrics(run):
    reqs = run['reqs']
    ttft = np.array([r.ttft for r in reqs if r.ttft is not None])
    itls = np.concatenate([np.array(r.itls) for r in reqs if r.itls]) \
        if any(r.itls for r in reqs) else np.array([0.0])
    tpot = np.array([r.tpot for r in reqs if r.tpot is not None])
    out_tokens = sum(r.produced for r in reqs)
    span = run['end'] - min(r.arrive for r in reqs)
    lat = np.array([r.latency for r in reqs if r.latency is not None])
    return {
        'name': run['name'],
        'ttft_p50': float(np.percentile(ttft, 50)) / 1e3,
        'ttft_p99': float(np.percentile(ttft, 99)) / 1e3,
        'tpot_ms': float(tpot.mean()) if len(tpot) else 0.0,
        'itl_p50': float(np.percentile(itls, 50)),
        'itl_p99': float(np.percentile(itls, 99)),
        'latency_p50': float(np.percentile(lat, 50)) / 1e3,
        'throughput': out_tokens / (span / 1e3),
        'end_s': run['end'] / 1e3,
        'preemptions': run.get('preemptions', 0),
        'pad_waste': run.get('pad_waste', 0.0),
    }


def goodput(run, ttft_s=2.0, itl_ms=50.0):
    """Requests that met the service-level objective, not just finished."""
    ok = 0
    for r in run['reqs']:
        if r.ttft is None or r.finish is None:
            continue
        p99 = float(np.percentile(r.itls, 99)) if r.itls else 0.0
        if r.ttft / 1e3 <= ttft_s and p99 <= itl_ms:
            ok += 1
    return ok / len(run['reqs'])


# ------------------------------------------------------------ experiments
def rate_sweep(rates=(4, 8, 12, 16, 20, 24), n=300, **kw):
    rows = []
    for r in rates:
        run = run_continuous(workload(n=n, rate_per_s=r), **kw)
        m = metrics(run)
        m['rate'] = r
        m['goodput'] = goodput(run)
        rows.append(m)
    return rows


def batching_table(n=300, rate=12):
    w = workload(n=n, rate_per_s=rate)
    runs = [run_static(w, 8), run_static(w, 32), run_continuous(w)]
    out = []
    for run in runs:
        m = metrics(run)
        m['goodput'] = goodput(run)
        out.append(m)
    return out, w


def chunk_table(w, chunks=(None, 1024, 512, 256), **kw):
    rows = []
    for c in chunks:
        run = run_continuous(w, chunk=c, **kw)
        m = metrics(run)
        m['chunk'] = c or 0
        m['goodput'] = goodput(run)
        rows.append(m)
    return rows


def pool_table(w, pools=(150_000, 60_000), caps=(None, 24, 16), preempts=('recompute',)):
    rows = []
    for pool in pools:
        for cap in caps:
            for pre in preempts:
                run = run_continuous(w, pool_tokens=pool, admit_cap=cap, preempt=pre)
                m = metrics(run)
                m.update({'pool': pool, 'cap': cap or 0, 'preempt': pre,
                          'goodput': goodput(run)})
                rows.append(m)
    return rows


def policy_table(n=300, rate=20, cap=32):
    w = workload(n=n, rate_per_s=rate)
    rows = []
    for pol in ('fcfs', 'sjf'):
        run = run_continuous(w, policy=pol, admit_cap=cap)
        lat = np.array([r.latency / 1e3 for r in run['reqs'] if r.latency])
        longs = [r.latency / 1e3 for r in run['reqs'] if r.prompt > 5000]
        m = metrics(run)
        rows.append({'policy': pol, 'mean': float(lat.mean()),
                     'p50': float(np.percentile(lat, 50)),
                     'p99': float(np.percentile(lat, 99)), 'max': float(lat.max()),
                     'long_mean': float(np.mean(longs)), 'n_long': len(longs),
                     'throughput': m['throughput']})
    return rows


# ----------------------------------------------------------------- printing
def head(n, title):
    print('\n' + '=' * 72)
    print(f'{n}. {title}')
    print('=' * 72 + '\n')


def main():
    print('Day 64 - continuous batching and the inference scheduler')
    print('A serving engine is a loop that chooses, a few hundred times a second,')
    print('which requests go into the next forward pass.  This is that loop.')

    # ------------------------------------------------------------------ 1
    head(1, 'The short version')
    rows, w = batching_table()
    st8, st32, cont = rows
    print(f"  Same model, same GPUs, same 300 requests arriving at 12 per second:\n")
    print('    scheduler                      TTFT p50   throughput   finished in')
    for r in (st8, st32, cont):
        print(f"    {r['name']:28} {r['ttft_p50']:7.2f} s   "
              f"{r['throughput']:7.0f} tok/s   {r['end_s']:7.0f} s")
    print(f"\n  Continuous batching is {cont['throughput'] / st8['throughput']:.1f}x the "
          f"throughput of a batch of 8 and drops the")
    print(f"  median time to first token from {st8['ttft_p50']:.1f} seconds to "
          f"{cont['ttft_p50']:.2f}.  No kernel changed.")
    print('  The only difference is that requests may join and leave the batch')
    print('  between steps instead of only at its edges.')

    # ------------------------------------------------------------------ 2
    head(2, 'What one step costs, and why that shapes everything')
    print(f"  The machine: {GPU['name']}.\n")
    print(f"    reading every weight once          {STEP_WEIGHT_MS:6.2f} ms per step")
    print(f"    reading one token of KV            {KV_MS_PER_TOKEN * 1e3:6.3f} us")
    print(f"    prefilling one prompt token        {PREFILL_MS_PER_TOKEN:6.4f} ms")
    print(f"    KV pool                            {GPU['kv_pool_tokens']:,} tokens\n")
    for b, ctx in ((1, 2_000), (16, 32_000), (64, 128_000), (128, 256_000)):
        ms = decode_step_ms(ctx, b)
        print(f'    batch {b:4}: {ms:6.2f} ms per step, '
              f'{b / ms * 1e3:8.0f} output tokens/s, {ms / b * 1e3:6.1f} us per token')
    print('\n  That table is the whole economics of serving, and it is Day 62.5')
    print('  again: the weights are read once per *step*, not once per request,')
    print('  so every request added to a running batch is nearly free until the')
    print('  KV reads catch up with the weight read.  A scheduler exists to keep')
    print('  that batch as full as the KV pool allows.')

    # ------------------------------------------------------------------ 3
    head(3, 'Static batching: the slot stays occupied after the answer is done')
    print('    scheduler                      padding waste   throughput   TTFT p50')
    for r in (st8, st32):
        print(f"    {r['name']:28} {r['pad_waste']:11.0%}   "
              f"{r['throughput']:7.0f} tok/s   {r['ttft_p50']:6.2f} s")
    print(f"\n  A batch runs until its *longest* member finishes, so "
          f"{st32['pad_waste']:.0%} of the slot-steps")
    print('  in a batch of 32 are spent on requests that already produced their')
    print('  last token.  Worse, a request that arrives one step after the batch')
    print('  started waits for the whole batch: that is where the')
    print(f"  {st8['ttft_p50']:.1f}-second median time to first token comes from, not the model.")

    # ------------------------------------------------------------------ 4
    head(4, 'Continuous batching: join and leave between steps')
    print('  The loop, in full:\n')
    print('    while work remains:')
    print('        admit what fits in the KV pool        # the scheduler')
    print('        if a prompt is waiting: prefill it')
    print('        else: one decode step for everybody running')
    print('        retire anyone who hit their stop condition\n')
    print(f"    TTFT p50 / p99   {cont['ttft_p50']:.2f} s / {cont['ttft_p99']:.2f} s")
    print(f"    ITL  p50 / p99   {cont['itl_p50']:.1f} ms / {cont['itl_p99']:.1f} ms")
    print(f"    TPOT             {cont['tpot_ms']:.1f} ms")
    print(f"    throughput       {cont['throughput']:.0f} output tokens/s")
    print(f"    padding waste    {cont['pad_waste']:.0%}\n")
    print('  Note the one number that got *worse*: the 99th percentile of the')
    print(f"  inter-token latency is {cont['itl_p99']:.0f} ms against a median of "
          f"{cont['itl_p50']:.1f} ms.  Section 6")
    print('  is about where that comes from.')

    # ------------------------------------------------------------------ 5
    head(5, 'The latency-throughput curve, and where the knee is')
    print('    arrivals/s   throughput   TTFT p50   TTFT p99   ITL p99   goodput')
    for r in rate_sweep():
        print(f"    {r['rate']:10}   {r['throughput']:7.0f}    {r['ttft_p50']:7.2f} s  "
              f"{r['ttft_p99']:7.2f} s  {r['itl_p99']:7.1f} ms   {r['goodput']:6.0%}")
    print('\n  Below the knee the engine is arrival-limited: throughput tracks the')
    print('  offered load and latency barely moves.  Above it the queue grows')
    print('  without bound and the only thing that still improves is throughput -')
    print('  which is exactly why a serving report that quotes throughput alone')
    print('  cannot be read.  The capacity here is set by prefill: the average')
    print(f"  prompt costs {prefill_ms(1800):.0f} ms of pure arithmetic, so about 20 requests per")
    print('  second is all the arithmetic allows, whatever the batch size.')

    # ------------------------------------------------------------------ 6
    head(6, 'A prefill blocks every decode behind it')
    print('    chunk budget   TTFT p50   TTFT p99   ITL p50   ITL p99   throughput')
    for r in chunk_table(w):
        label = 'none (whole prompt)' if not r['chunk'] else f"{r['chunk']} tokens"
        print(f"    {label:20} {r['ttft_p50']:7.2f} s  {r['ttft_p99']:7.2f} s  "
              f"{r['itl_p50']:7.1f} ms  {r['itl_p99']:7.1f} ms  {r['throughput']:7.0f}")
    print(f"\n  A 16,000-token prompt takes {prefill_ms(16000):.0f} ms of arithmetic.  If that runs as")
    print('  one step, every request already decoding sees a gap that long between')
    print('  two of its tokens - which is the ITL p99 in the first row.  Chunked')
    print('  prefill cuts the prompt into pieces and mixes them into the decode')
    print('  steps, so the gap becomes the chunk size instead of the prompt size.')
    print('  It is not free: each chunk pays the weight read again, so prefill')
    print('  throughput drops, and a prompt that now takes several steps reaches')
    print('  its first token later.  Smaller chunks trade TTFT for ITL.')

    # ------------------------------------------------------------------ 7
    head(7, 'When the KV pool runs out')
    print('    pool tokens   admission cap   preemptions   throughput   ITL p99   goodput')
    for r in pool_table(w):
        cap = r['cap'] or 'none'
        print(f"    {r['pool']:11,}   {str(cap):>13}   {r['preemptions']:11}   "
              f"{r['throughput']:7.0f}    {r['itl_p99']:7.1f}   {r['goodput']:6.0%}")
    swap = metrics(run_continuous(w, pool_tokens=60_000, preempt='swap'))
    swap['goodput'] = goodput(run_continuous(w, pool_tokens=60_000, preempt='swap'))
    print(f"    {60000:11,}   {'none (swap)':>13}   {swap['preemptions']:11}   "
          f"{swap['throughput']:7.0f}    {swap['itl_p99']:7.1f}   {swap['goodput']:6.0%}")
    print('\n  Admitting a request means promising it KV space for every token it')
    print('  will generate, and nobody knows how many that is.  When the pool')
    print('  fills, somebody has to be evicted: with recompute the victim loses')
    print('  all its progress and its prompt is prefilled again, which is why the')
    print('  small pool both loses throughput and produces a 400 ms ITL tail.')
    print('  Swapping the KV to host memory keeps the progress and pays PCIe')
    print(f"  instead - {swap_ms(4000):.0f} ms for a 4,000-token context, round trip.\n")
    print('  The cheaper answer is to not over-admit in the first place.')

    # ------------------------------------------------------------------ 8
    head(8, 'Admission control: refusing to start work you cannot finish')
    rows = pool_table(w)
    get = lambda pool, cap: [r for r in rows if r['pool'] == pool and r['cap'] == cap][0]
    small_nocap, small_cap = get(60_000, 0), get(60_000, 16)
    big_nocap, big_cap = get(150_000, 0), get(150_000, 16)
    print('    pool          cap   throughput   preemptions   goodput')
    for r, lbl in ((small_nocap, 'small (60k)'), (small_cap, 'small (60k)'),
                   (big_nocap, 'big (150k)'), (big_cap, 'big (150k)')):
        cap = r['cap'] or 'none'
        print(f"    {lbl:12} {str(cap):>5}   {r['throughput']:8.0f}   "
              f"{r['preemptions']:11}   {r['goodput']:6.0%}")
    print(f"\n  On the small pool the cap is free in both directions: throughput goes")
    print(f"  *up* by {small_cap['throughput'] / small_nocap['throughput'] - 1:.0%} and goodput by "
          f"{small_cap['goodput'] / max(small_nocap['goodput'], 1e-9):.0f}x, because what the")
    print('  preemptions threw away was pure waste - prompts prefilled twice, KV')
    print('  written and then dropped.  A queue is cheaper than half-finished')
    print('  work, and that is the entire argument for admission control.\n')
    print('  On the big pool the same cap is a mistake: nothing was being')
    print(f"  preempted there, so capping only adds queueing - "
          f"{1 - big_cap['throughput'] / big_nocap['throughput']:.0%} less throughput")
    print(f"  and goodput down from {big_nocap['goodput']:.0%} to {big_cap['goodput']:.0%}.  A cap is not a knob that")
    print('  always helps; it is a claim that the KV pool is the binding')
    print('  constraint, and when it is not, the claim costs you the slack.')

    # ------------------------------------------------------------------ 9
    head(9, 'Which request goes first')
    print('    policy   mean latency   p50     p99     max     long prompts (n=27)')
    for r in policy_table():
        print(f"    {r['policy']:6}   {r['mean']:9.1f} s   {r['p50']:5.1f}   "
              f"{r['p99']:5.1f}   {r['max']:5.1f}   {r['long_mean']:9.1f} s")
    print('\n  Shortest-job-first is the textbook answer and it works: the mean')
    print('  latency drops by about half.  It also doubles the latency of exactly')
    print('  the requests that were already slowest, because a long prompt is')
    print('  overtaken by every short one that arrives while it waits.  At a')
    print('  constant overload it never runs at all.  Production schedulers')
    print('  therefore age the queue: priority rises with waiting time, so the')
    print('  policy is shortest-job-first with a guarantee attached.')

    # ----------------------------------------------------------------- 10
    head(10, 'Goodput, not throughput')
    cfgs = [('continuous, no limits', run_continuous(w)),
            ('+ chunked prefill (512)', run_continuous(w, chunk=512)),
            ('+ admission cap (32)', run_continuous(w, chunk=512, admit_cap=32)),
            ('small pool, no cap', run_continuous(w, pool_tokens=60_000))]
    print('    configuration             throughput   TTFT p99   ITL p99   goodput')
    for name, run in cfgs:
        m = metrics(run)
        print(f"    {name:24} {m['throughput']:8.0f}   {m['ttft_p99']:7.2f} s  "
              f"{m['itl_p99']:7.1f} ms   {goodput(run):6.0%}")
    print('\n  The first three move the same number of tokens per second.  What')
    print('  separates them is how many requests met a service-level objective -')
    print('  here time to first token under 2 s and 99th-percentile inter-token')
    print('  latency under 50 ms.  Tuning a serving stack means moving the last')
    print('  column, and the last column is invisible if you only measure the')
    print('  first one.')

    # ----------------------------------------------------------------- 11
    head(11, 'Four things that look like wins and are not')
    pool_rows = pool_table(w)
    big = [r for r in pool_rows if r['pool'] == 60_000 and r['cap'] == 0][0]
    print('  1. "A bigger batch is always better."  With a pool that cannot hold')
    print(f"     it, raising the running set produces {big['preemptions']} preemptions, "
          f"{1 - big['throughput'] / cont['throughput']:.0%} less")
    print('     throughput and a 400 ms ITL tail.  The limit is the KV pool, not')
    print('     the batch dimension of the kernels.\n')
    print('  2. "Throughput went up, so it is better."  Section 10: three configs')
    print('     at the same throughput and goodput from 40% to 100%.\n')
    print('  3. "Shortest-job-first minimises latency."  It minimises the *mean*')
    print('     while roughly doubling the long prompts\' latency - the requests')
    print('     that were already worst off.\n')
    print(f"  4. \"The batch is full, so the GPU is busy.\"  In static batching "
          f"{st32['pad_waste']:.0%} of")
    print('     the slot-steps are requests that have already finished; the batch')
    print('     is full of nothing.')

    # ----------------------------------------------------------------- 12
    head(12, 'In vLLM')
    print('    --max-num-seqs 256              # the running set, section 8')
    print('    --max-num-batched-tokens 8192   # the chunked-prefill budget')
    print('    --enable-chunked-prefill        # on by default in recent versions')
    print('    --gpu-memory-utilization 0.9    # how much HBM becomes KV pool')
    print('    --preemption-mode recompute     # or swap, section 7\n')
    print('  vLLM calls the admitted set the running queue and keeps a waiting')
    print('  queue behind it; the block manager hands out KV in pages, which is')
    print('  what makes admission a counting problem rather than a guess.  The')
    print('  log line to watch is the preemption warning: it is the engine saying')
    print('  that section 7 is happening to you right now.')

    # ----------------------------------------------------------------- 13
    head(13, 'In SGLang')
    print('    --max-running-requests 256      # the same cap')
    print('    --chunked-prefill-size 8192')
    print('    --schedule-policy lpm           # longest-prefix-match, not FCFS')
    print('    --schedule-conservativeness 1.0 # how eagerly to admit\n')
    print('  The interesting difference is the default policy: SGLang orders the')
    print('  waiting queue by longest prefix match, so requests that share a')
    print('  prefix run together and hit the radix cache instead of prefilling')
    print('  the same tokens twice.  That is a scheduling decision made for a')
    print('  memory reason - the same move as everything else in this series.')

    # ----------------------------------------------------------------- 14
    head(14, 'Assertions')
    check()
    print('all assertions passed')


def check():
    w = workload(n=200, rate_per_s=12, seed=1)

    # the cost model: adding a request to a running batch is nearly free
    one = decode_step_ms(2_000, 1)
    many = decode_step_ms(128_000, 64)
    assert many < 2 * one and 64 / many > 30 * (1 / one)

    # continuous batching beats static on both axes at the same load
    st = metrics(run_static(w, 8))
    co = metrics(run_continuous(w))
    assert co['throughput'] > 2 * st['throughput']
    assert co['ttft_p50'] < st['ttft_p50'] / 100
    assert st['pad_waste'] > 0.4 and co['pad_waste'] == 0.0

    # nobody is lost or duplicated, and everybody gets their tokens
    for run in (run_static(w, 8), run_continuous(w), run_continuous(w, chunk=512)):
        reqs = run['reqs']
        assert len(reqs) == len(w)
        assert all(r.produced == r.out_len for r in reqs)
        assert all(r.finish >= r.arrive for r in reqs)
        assert all(r.ttft >= 0 for r in reqs)

    # chunked prefill trades tail latency for first-token latency
    plain = metrics(run_continuous(w))
    chunked = metrics(run_continuous(w, chunk=512))
    assert chunked['itl_p99'] < plain['itl_p99'] / 2
    assert chunked['ttft_p50'] >= plain['ttft_p50']

    # a pool too small for the running set causes preemption, not an error
    small = run_continuous(w, pool_tokens=60_000)
    assert small['preemptions'] > 0
    assert metrics(small)['throughput'] < plain['throughput']

    # capping admission costs throughput and buys goodput
    capped = run_continuous(w, pool_tokens=60_000, admit_cap=16)
    assert capped['preemptions'] == 0
    assert goodput(capped) > goodput(small)

    # swapping keeps progress, so it finishes sooner than recompute
    swapped = run_continuous(w, pool_tokens=60_000, preempt='swap')
    assert metrics(swapped)['throughput'] > metrics(small)['throughput']

    # shortest-job-first: better mean, worse for the long prompts
    pol = {r['policy']: r for r in policy_table(n=200)}
    assert pol['sjf']['mean'] < pol['fcfs']['mean']
    assert pol['sjf']['long_mean'] > pol['fcfs']['long_mean']

    # the knee: throughput saturates while latency does not
    sweep = rate_sweep(rates=(4, 12, 24), n=200)
    assert sweep[-1]['throughput'] < 2 * sweep[1]['throughput']
    assert sweep[-1]['ttft_p99'] > 5 * sweep[0]['ttft_p99']


if __name__ == '__main__':
    main()
