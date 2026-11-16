"""Day 58 - PagedAttention as what it really is: a memory allocator for the KV cache.

The attention kernel change is small - read K and V through a block table instead of
from one flat array.  Everything else is an operating systems problem that an inference
server has to solve: handing out fixed-size blocks, refusing work when the pool is full,
sharing blocks between samples of the same prompt, and taking blocks back from a running
request when something more urgent arrives.

This file simulates the allocator, measures what paging actually buys, and then shows
the knobs that vLLM and SGLang expose for each piece.

Run `python paged_memory.py` to print the whole story; every number below is produced by
this file and checked by the asserts at the bottom.
"""

import math

import numpy as np


# ================================================================ 1. a workload
def make_requests(n=240, seed=0, pool_tokens=8192):
    """Arrivals with a prompt and an answer length, the way a server sees them."""
    rng = np.random.default_rng(seed)
    reqs = []
    t = 0
    for i in range(n):
        t += int(rng.integers(0, 3))
        prompt = int(rng.integers(64, 1024))
        out = int(rng.integers(32, 512))
        reqs.append({'id': i, 'arrive': t, 'prompt': prompt, 'out': out,
                     'total': prompt + out})
    return reqs


# ================================================================ 2. two allocators
class ContiguousPool:
    """One contiguous range per request, reserved for the longest it might get.

    This is how you would write it if the KV cache were a single array per sequence:
    the range cannot move once handed out, so it has to be big enough from the start.
    """

    def __init__(self, capacity, reserve):
        self.capacity = capacity
        self.reserve = reserve          # tokens reserved per request
        self.free = [(0, capacity)]     # (start, length), kept sorted
        self.held = {}

    def allocate(self, rid):
        need = self.reserve
        best, bi = None, -1
        for i, (s, ln) in enumerate(self.free):      # best fit
            if ln >= need and (best is None or ln < best[1]):
                best, bi = (s, ln), i
        if best is None:
            return False
        s, ln = self.free.pop(bi)
        self.held[rid] = (s, need)
        if ln > need:
            self.free.insert(bi, (s + need, ln - need))
        return True

    def release(self, rid):
        s, ln = self.held.pop(rid)
        self.free.append((s, ln))
        self.free.sort()
        merged = []                                   # coalesce neighbours
        for seg in self.free:
            if merged and merged[-1][0] + merged[-1][1] == seg[0]:
                merged[-1] = (merged[-1][0], merged[-1][1] + seg[1])
            else:
                merged.append(seg)
        self.free = merged

    def free_tokens(self):
        return sum(ln for _, ln in self.free)

    def largest_hole(self):
        return max((ln for _, ln in self.free), default=0)


class PagedPool:
    """Fixed-size blocks and a block table per request - a page table, in other words."""

    def __init__(self, capacity, block=16):
        self.block = block
        self.n_blocks = capacity // block
        self.free = list(range(self.n_blocks))
        self.table = {}                  # rid -> [block ids]
        self.refs = {}                   # block id -> how many requests point at it

    def blocks_for(self, tokens):
        return math.ceil(tokens / self.block)

    def allocate(self, rid, tokens):
        need = self.blocks_for(tokens) - len(self.table.get(rid, []))
        if need <= 0:
            return True
        if need > len(self.free):
            return False
        got = [self.free.pop() for _ in range(need)]
        for b in got:
            self.refs[b] = 1
        self.table.setdefault(rid, []).extend(got)
        return True

    def share(self, src, dst, tokens):
        """Point a second request at the same blocks: parallel sampling, one prompt."""
        shared = self.table[src][:self.blocks_for(tokens)]
        for b in shared:
            self.refs[b] += 1
        self.table[dst] = list(shared)
        return len(shared)

    def release(self, rid):
        for b in self.table.pop(rid, []):
            self.refs[b] -= 1
            if self.refs[b] == 0:
                self.free.append(b)

    def used_blocks(self):
        return self.n_blocks - len(self.free)

    def free_tokens(self):
        return len(self.free) * self.block


# ================================================================ 3. what paging buys
def fragmentation(reqs, capacity=16384, block=16):
    """Serve the same arrivals both ways and count who gets turned away.

    The contiguous pool here is given the best possible case: it is told the exact
    final length of every request, so it never over-reserves.  It still refuses work,
    because the free space it has is in the wrong shape - that is external
    fragmentation, and it is the thing fixed-size blocks remove by construction.
    """
    cont = ContiguousPool(capacity, 0)
    paged = PagedPool(capacity, block)
    running_c, running_p = {}, {}
    rej_c = rej_p = rej_c_frag = 0
    use_c, use_p = [], []
    clock = 0
    pending = list(reqs)
    while pending or running_c or running_p:
        clock += 1
        for rid, (end, _) in list(running_c.items()):
            if end <= clock:
                cont.release(rid)
                del running_c[rid]
        for rid, (end, _) in list(running_p.items()):
            if end <= clock:
                paged.release(rid)
                del running_p[rid]
        while pending and pending[0]['arrive'] <= clock:
            r = pending.pop(0)
            life = clock + r['total'] // 64 + 1
            cont.reserve = r['total']                 # exact size: the kindest case
            if cont.allocate(r['id']):
                running_c[r['id']] = (life, r['total'])
            else:
                rej_c += 1
                if cont.free_tokens() >= r['total']:  # room, but not in one piece
                    rej_c_frag += 1
            if paged.allocate(r['id'], r['total']):
                running_p[r['id']] = (life, r['total'])
            else:
                rej_p += 1
        use_c.append(sum(t for _, t in running_c.values()))
        use_p.append(sum(t for _, t in running_p.values()))
    return {'rejected_contiguous': rej_c, 'rejected_fragmentation': rej_c_frag,
            'rejected_paged': rej_p, 'util_contiguous': float(np.mean(use_c)) / capacity,
            'util_paged': float(np.mean(use_p)) / capacity, 'steps': clock,
            'requests': len(reqs)}


def internal_waste(reqs, blocks=(1, 8, 16, 64, 256)):
    """What paging costs: the unused tail of every request's last block."""
    rows = []
    for b in blocks:
        used = sum(math.ceil(r['total'] / b) * b for r in reqs)
        real = sum(r['total'] for r in reqs)
        rows.append({'block': b, 'slots': used, 'waste': (used - real) / used,
                     'tables': sum(math.ceil(r['total'] / b) for r in reqs)})
    return rows


# ================================================================ 4. sharing
def parallel_sampling(prompt=900, n_samples=4, out=200, block=16, capacity=65536):
    """n answers to one prompt: the prompt's blocks are shared, not copied.

    The last prompt block is the interesting one: it is only partly full, so the moment
    two samples write their own next token into it, it has to be copied.  That is
    copy-on-write, and it is one block per sample, not one prompt per sample.
    """
    pool = PagedPool(capacity, block)
    pool.allocate('s0', prompt)
    shared = 0
    for i in range(1, n_samples):
        shared += pool.share('s0', f's{i}', prompt)
    naive = n_samples * math.ceil(prompt / block)
    cow = 1 if prompt % block else 0           # the partial block each sample copies
    actual = pool.used_blocks() + cow * (n_samples - 1)
    for i in range(n_samples):
        pool.allocate(f's{i}', prompt + out)
    return {'prompt_blocks': math.ceil(prompt / block), 'naive': naive,
            'shared_blocks': shared, 'with_sharing': actual,
            'saved': 1 - actual / naive, 'cow_blocks': cow * (n_samples - 1),
            'after_decode': pool.used_blocks()}


# ================================================================ 5. preemption
def preemption(capacity=4096, block=16, n=60, seed=1, policy='recompute',
               max_steps=20000):
    """When the pool runs out, a running request has to give its blocks back.

    Two ways to do it: drop them and recompute the prefix later, or copy them to host
    memory and copy them back.  The first costs compute, the second costs bandwidth -
    the same trade as the day before, one level up.

    Newest-first is the victim rule, because the newest request has the least work
    invested in it; it is also what vLLM does.
    """
    rng = np.random.default_rng(seed)
    pool = PagedPool(capacity, block)
    waiting = [{'id': i, 'need': int(rng.integers(200, 900)), 'done': 0}
               for i in range(n)]
    running = []
    recomputed = swapped = steps = preemptions = 0
    while (waiting or running) and steps < max_steps:
        steps += 1
        while waiting and pool.allocate(waiting[0]['id'], waiting[0]['need']):
            running.append(waiting.pop(0))
        for r in list(running):                       # one decode step of work
            r['done'] += 64
            if r['done'] >= r['need']:
                pool.release(r['id'])
                running.remove(r)
        if waiting and running and pool.free_tokens() < waiting[0]['need']:
            victim = running.pop()                    # newest first
            pool.release(victim['id'])
            preemptions += 1
            if policy == 'recompute':
                recomputed += victim['done']          # work thrown away
                victim['done'] = 0
            else:
                swapped += victim['done']             # copied out and back
            waiting.append(victim)
    return {'preemptions': preemptions, 'recomputed_tokens': recomputed,
            'swapped_tokens': swapped, 'steps': steps,
            'finished': n - len(waiting) - len(running)}


# ================================================================ 6. two silent failures
def leaked_blocks(n=200, block=16, capacity=65536, fail_rate=0.1, seed=0):
    """A request that errors out and never frees its blocks.

    Nothing raises: the pool just gets smaller, and the server slowly stops accepting
    work for no visible reason.
    """
    rng = np.random.default_rng(seed)
    pool = PagedPool(capacity, block)
    leaked = 0
    for i in range(n):
        tokens = int(rng.integers(200, 800))
        pool.allocate(i, tokens)
        if rng.random() < fail_rate:
            leaked += pool.blocks_for(tokens)      # the except: branch forgot to free
        else:
            pool.release(i)
    return {'leaked_blocks': leaked, 'leaked_tokens': leaked * block,
            'free_left': len(pool.free) * block, 'capacity': capacity}


def block_table_off_by_one(tokens=70, block=16):
    """The last block is partly full, and the kernel has to be told how full.

    Round the length up to the block and attention reads the slots after the last real
    token - uninitialised memory, not a crash.
    """
    n_blocks = math.ceil(tokens / block)
    padded = n_blocks * block
    return {'tokens': tokens, 'blocks': n_blocks, 'padded': padded,
            'garbage_slots': padded - tokens,
            'share': (padded - tokens) / padded}


# ================================================================ 7. assertions
def check():
    reqs = make_requests()

    # paging turns away far fewer requests than a contiguous pool of the same size
    f = fragmentation(reqs)
    assert f['rejected_paged'] < f['rejected_contiguous']
    assert f['rejected_fragmentation'] == f['rejected_contiguous'] > 0
    assert f['util_paged'] > f['util_contiguous']

    # internal waste falls as the block gets smaller, and the table gets longer
    rows = internal_waste(reqs)
    assert rows[0]['waste'] == 0 and rows[-1]['waste'] > rows[1]['waste']
    assert rows[0]['tables'] > rows[-1]['tables']

    # sharing one prompt between samples saves most of the blocks
    ps = parallel_sampling()
    assert ps['with_sharing'] < ps['naive'] and ps['saved'] > 0.5

    # both preemption policies move the same work, in different currencies
    a = preemption(policy='recompute')
    b = preemption(policy='swap')
    assert a['preemptions'] > 0 and b['preemptions'] > 0
    assert a['recomputed_tokens'] > 0 and b['swapped_tokens'] > 0
    assert a['finished'] == b['finished'] == 60

    # a leak is silent: the pool shrinks and nothing raises
    lk = leaked_blocks()
    assert lk['leaked_tokens'] > 0 and lk['free_left'] < lk['capacity']

    # the last block is partly garbage unless the kernel is told the real length
    ob = block_table_off_by_one()
    assert ob['garbage_slots'] > 0
    return True


# ================================================================ 8. the whole story
def header(title):
    print('\n' + '=' * 72)
    print(title)
    print('=' * 72)


def main():
    np.set_printoptions(suppress=True, precision=4, legacy='1.25')
    reqs = make_requests()

    # ------------------------------------------------------------------ 1
    header('1. The attention change is small; the allocator is the work')
    print('  Paged attention reads K and V through a block table instead of from one')
    print('  flat array per sequence.  That is a few lines in the kernel.  What it')
    print('  enables is everything else in this file: handing out fixed-size blocks,')
    print('  refusing work when the pool is full, sharing blocks between samples, and')
    print('  taking blocks back from a running request.  It is an operating system,')
    print('  not an attention variant.\n')
    f = fragmentation(reqs)
    print(f'  {f["requests"]} requests against a pool of 16,384 token slots.  The')
    print('  contiguous allocator is given the kindest possible case: it is told the')
    print('  exact final length of every request, so it never over-reserves.\n')
    print(f'    contiguous, one range per request : {f["rejected_contiguous"]} refused')
    print(f'      of those, with enough free      : {f["rejected_fragmentation"]}')
    print(f'    paged, 16 tokens per block        : {f["rejected_paged"]} refused')
    print(f'    occupancy, averaged over time     : '
          f'{f["util_contiguous"] * 100:.1f}% vs {f["util_paged"] * 100:.1f}%')
    print('\n  The second line is the whole argument: every request the contiguous')
    print('  pool turned away was turned away while the pool had room for it.  The')
    print('  space existed, in pieces too small to use.  That is external')
    print('  fragmentation, and fixed-size blocks remove it by construction - any free')
    print('  block is as good as any other, so there is no such thing as the wrong')
    print('  shape of free memory.')

    # ------------------------------------------------------------------ 2
    header('2. What paging costs instead')
    print('  Internal fragmentation: the unused tail of each request\'s last block.\n')
    print(f'  {"block size":>12} {"slots used":>12} {"wasted":>9} {"block table entries":>21}')
    for r in internal_waste(reqs):
        print(f'  {r["block"]:>12} {r["slots"]:>12,} {r["waste"] * 100:>8.1f}% '
              f'{r["tables"]:>21,}')
    print('\n  One token per block wastes nothing and needs a table entry per token;')
    print('  256 tokens per block needs 224 times fewer entries and throws away an')
    print('  eighth of the pool.')
    print('  Real servers sit at 16 or 32, and the reason is not only the waste: a')
    print('  larger block is also the unit that prefix sharing has to align to.')

    # ------------------------------------------------------------------ 3
    header('3. Sharing: n samples of one prompt')
    ps = parallel_sampling()
    print(f'  A 900-token prompt, 4 samples, 16 tokens per block:\n')
    print(f'    one copy per sample : {ps["naive"]} blocks')
    print(f'    sharing the prompt  : {ps["with_sharing"]} blocks  '
          f'({ps["saved"] * 100:.0f}% less)')
    print(f'    of which copied     : {ps["cow_blocks"]} (the last, partly-full block)')
    print('\n  The samples point at the same blocks and a reference count keeps them')
    print('  alive.  Only the final block of the prompt is a problem: it is half empty,')
    print('  and the moment two samples write their own next token into it they need')
    print('  their own copy.  Copying one block instead of the whole prompt is what')
    print('  copy-on-write means here - the same mechanism a process fork uses.')

    # ------------------------------------------------------------------ 4
    header('4. Preemption: taking blocks back from a running request')
    a = preemption(policy='recompute')
    b = preemption(policy='swap')
    print(f'  60 requests against a pool of 4,096 slots, so the pool runs out:\n')
    print(f'    recompute policy : {a["preemptions"]} preemptions, '
          f'{a["recomputed_tokens"]:,} tokens prefilled again')
    print(f'    swap policy      : {b["preemptions"]} preemptions, '
          f'{b["swapped_tokens"]:,} tokens copied out and back')
    print(f'    both finished all {a["finished"]} requests, in {a["steps"]} and '
          f'{b["steps"]} steps')
    print('\n  A similar number of interruptions, two different currencies: compute or')
    print('  bandwidth.  Recompute is simpler and wins for short prefixes; swapping')
    print('  wins when the prefix is long enough that prefilling it again costs more')
    print('  than moving it.  Both are preferable to the third option, which is to let')
    print('  the pool fill up and deadlock with every request half-finished.')

    # ------------------------------------------------------------------ 5
    header('5. Silent failure #1: blocks that are never freed')
    lk = leaked_blocks()
    print(f'  200 requests, one in ten failing on a path that forgets to free:\n')
    print(f'    leaked blocks : {lk["leaked_blocks"]}')
    print(f'    leaked slots  : {lk["leaked_tokens"]:,} of {lk["capacity"]:,}')
    print(f'    pool left     : {lk["free_left"]:,} slots')
    print('\n  Nothing raises.  The server keeps working, accepts fewer and fewer')
    print('  requests, and the symptom reported is "throughput degrades after a few')
    print('  hours".  The test that finds it is an invariant, checked when the server')
    print('  is idle: free blocks plus blocks in block tables must equal the pool.')

    # ------------------------------------------------------------------ 6
    header('6. Silent failure #2: forgetting how full the last block is')
    ob = block_table_off_by_one()
    print(f'  A sequence of {ob["tokens"]} tokens with 16-token blocks occupies '
          f'{ob["blocks"]} blocks =')
    print(f'  {ob["padded"]} slots, so {ob["garbage_slots"]} slots at the end hold '
          f'whatever was there before.')
    print(f'  Attend over the padded length and {ob["share"] * 100:.0f}% of the last '
          f'block is noise.')
    print('\n  The kernel needs both the block table and the true sequence length, and')
    print('  the masking has to use the second.  Get it wrong and nothing crashes: the')
    print('  rows still sum to one, and the model is attending to stale KV from some')
    print('  other request that used those slots an hour ago.')

    # ------------------------------------------------------------------ 7
    header('7. In vLLM')
    print('  vLLM is where PagedAttention comes from, and its flags map one to one')
    print('  onto this file:\n')
    print('    --block-size 16                 tokens per block (section 2)')
    print('    --gpu-memory-utilization 0.90   how much HBM the pool may take')
    print('    --num-gpu-blocks-override N     set the pool size directly')
    print('    --max-num-seqs 256              cap on concurrent sequences')
    print('    --swap-space 4                  GiB of host memory for preemption')
    print('    --enable-prefix-caching         reuse blocks across requests\n')
    print('  At startup it runs a profiling pass, measures what is left after the')
    print('  weights and activations, and divides by the per-token size to get the')
    print('  number of blocks - the same division as the previous day.  When the pool')
    print('  runs out it logs "Sequence group ... is preempted by PreemptionMode.')
    print('  RECOMPUTE" and that line, in volume, is the signal that the pool is too')
    print('  small for the concurrency you asked for.')

    # ------------------------------------------------------------------ 8
    header('8. In SGLang')
    print('  SGLang solves the same problem with the same pieces under other names:\n')
    print('    --mem-fraction-static 0.9       share of HBM for the KV pool')
    print('    --page-size 1                   tokens per page (1 by default)')
    print('    --max-running-requests 256      cap on the decode batch')
    print('    --chunked-prefill-size 8192     split long prefills')
    print('    --max-total-tokens N            pool size directly, in tokens\n')
    print('  Two differences worth knowing.  Its default page size is 1 - one token per')
    print('  page - which costs more block-table entries and wastes nothing, and it')
    print('  makes prefix sharing exact at any boundary instead of block-aligned (the')
    print('  previous day).  And when the pool runs out it retracts requests rather')
    print('  than swapping: the same recompute policy as above, chosen by the same')
    print('  argument - with the radix tree holding the prefix, recomputing is cheap.')
    print('\n  The two systems disagree about the defaults and agree completely about')
    print('  the structure: a fixed pool of slots, a table per request, reference')
    print('  counts for sharing, and a policy for what to do when it runs out.')

    # ------------------------------------------------------------------ 9
    header('9. Assertions')
    check()
    print('all assertions passed')


if __name__ == '__main__':
    main()
