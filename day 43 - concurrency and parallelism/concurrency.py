"""Day 43 - Concurrency and parallelism: the GIL, races, deadlocks, asyncio,
work and depth, and map-reduce.

Run `python concurrency.py` to print the whole story.  Timings depend on the
machine (this file was written on a 2-core VM); every count that is not a
timing is exact and checked by an assert at the bottom.
"""

import asyncio
import hashlib
import math
import multiprocessing as mp
import os
import sys
import threading
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache, reduce
from itertools import accumulate
from math import comb
import random


# ---------------------------------------------------------------- 1. the GIL
def burn(n: int) -> int:
    """Pure-Python CPU work: holds the GIL the whole time."""
    s = 0
    for i in range(n):
        s += i * i
    return s


def best_of(fn, repeat=3) -> float:
    best = float('inf')
    for _ in range(repeat):
        t = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t)
    return best


def run_threads(target, args_list):
    ts = [threading.Thread(target=target, args=a) for a in args_list]
    for t in ts:
        t.start()
    for t in ts:
        t.join()


def gil_timings(n=2_000_000, jobs=2, nap=0.05, naps=20):
    """CPU-bound and I/O-bound jobs, run serially, in threads and in processes."""
    out = {}
    out['cpu serial'] = best_of(lambda: [burn(n) for _ in range(jobs)])
    out['cpu threads'] = best_of(lambda: run_threads(burn, [(n,)] * jobs))
    with mp.Pool(jobs) as pool:
        pool.map(burn, [10] * jobs)                       # warm the workers up
        out['cpu processes'] = best_of(lambda: pool.map(burn, [n] * jobs))
    out['io serial'] = best_of(lambda: [time.sleep(nap) for _ in range(naps)], 1)
    out['io threads'] = best_of(lambda: run_threads(time.sleep, [(nap,)] * naps), 1)
    return out


# ---------------------------------------------------------------- 2. races, for real
class Box:
    def __init__(self):
        self.v = 0


def _noop():
    pass


def bump_inline(box):
    box.v += 1                      # read, add, write - one line


def bump_with_call(box):
    v = box.v                       # read
    _noop()                         # any function call: logging, validation, ...
    box.v = v + 1                   # write the value read *before* the call


def hammer(bump, threads=4, each=200_000, switch=None, lock=None):
    """`threads` threads each call bump(box) `each` times; return the final count."""
    old = sys.getswitchinterval()
    if switch is not None:
        sys.setswitchinterval(switch)
    box = Box()

    def work():
        for _ in range(each):
            if lock is None:
                bump(box)
            else:
                with lock:
                    bump(box)
    try:
        run_threads(work, [()] * threads)
    finally:
        sys.setswitchinterval(old)
    return box.v


# ---------------------------------------------------------------- 3. an interleaving checker
# A tiny model of threads: each thread is a list of atomic steps.  The checker runs
# EVERY order in which the steps of different threads can interleave (each thread's
# own steps stay in order) - the same idea as Day 42's "replay every random choice".
#   ('read',  var, reg)        reg <- shared[var]
#   ('write', var, reg, add)   shared[var] <- reg + add
#   ('acquire', lock)          only possible while nobody holds the lock
#   ('release', lock)
#   ('wait', event)            only possible once the event is set
#   ('set', event)
#   ('emit', text)             append text to the output
DEADLOCK = 'DEADLOCK'


def _start(threads, shared):
    return (tuple(0 for _ in threads), tuple(sorted(shared.items())),
            tuple(() for _ in threads), (), frozenset(), '')


def _enabled(threads, state, i):
    pcs, shared, regs, held, events, out = state
    if pcs[i] >= len(threads[i]):
        return False
    op = threads[i][pcs[i]]
    if op[0] == 'acquire':
        return all(l != op[1] for l, _ in held)
    if op[0] == 'wait':
        return op[1] in events
    return True


def _step(threads, state, i):
    pcs, shared, regs, held, events, out = state
    op = threads[i][pcs[i]]
    sh, rg = dict(shared), dict(regs[i])
    held, events = list(held), set(events)
    kind = op[0]
    if kind == 'read':
        rg[op[2]] = sh[op[1]]
    elif kind == 'write':
        sh[op[1]] = rg[op[2]] + op[3]
    elif kind == 'acquire':
        held.append((op[1], i))
    elif kind == 'release':
        held.remove((op[1], i))
    elif kind == 'set':
        events.add(op[1])
    elif kind == 'emit':
        out += op[1]
    pcs = pcs[:i] + (pcs[i] + 1,) + pcs[i + 1:]
    regs = regs[:i] + (tuple(sorted(rg.items())),) + regs[i + 1:]
    return (pcs, tuple(sorted(sh.items())), regs, tuple(sorted(held)), frozenset(events), out)


def _outcome(threads, state):
    pcs, shared, regs, held, events, out = state
    if all(pcs[i] == len(t) for i, t in enumerate(threads)):
        return (shared, out)
    return DEADLOCK


def every_schedule(threads, shared):
    """Yield (schedule, outcome) for every interleaving.  Small cases only."""
    def go(state, sched):
        nxt = [i for i in range(len(threads)) if _enabled(threads, state, i)]
        if not nxt:
            yield tuple(sched), _outcome(threads, state)
            return
        for i in nxt:
            sched.append(i)
            yield from go(_step(threads, state, i), sched)
            sched.pop()
    yield from go(_start(threads, shared), [])


def count_schedules(threads, shared):
    """Counter outcome -> number of interleavings reaching it (memoised on the state,
    so it copes with billions of interleavings)."""
    @lru_cache(maxsize=None)
    def go(state):
        nxt = [i for i in range(len(threads)) if _enabled(threads, state, i)]
        if not nxt:
            return ((_outcome(threads, state), 1),)
        total = Counter()
        for i in nxt:
            for k, v in go(_step(threads, state, i)):
                total[k] += v
        return tuple(total.items())
    return Counter(dict(go(_start(threads, shared))))


def reachable_states(threads, shared):
    """(number of distinct states, number of them that are deadlocks)."""
    seen, stack, dead = set(), [_start(threads, shared)], 0
    while stack:
        s = stack.pop()
        if s in seen:
            continue
        seen.add(s)
        nxt = [i for i in range(len(threads)) if _enabled(threads, s, i)]
        if not nxt and _outcome(threads, s) == DEADLOCK:
            dead += 1
        stack.extend(_step(threads, s, i) for i in nxt)
    return len(seen), dead


def incrementer(locked=False, times=1):
    body = [('read', 'x', 'r'), ('write', 'x', 'r', 1)]
    if locked:
        body = [('acquire', 'L')] + body + [('release', 'L')]
    return body * times


def final_x(outcome):
    return DEADLOCK if outcome == DEADLOCK else dict(outcome[0])['x']


# ---------------------------------------------------------------- 4. deadlock
def two_lock_threads(same_order: bool):
    t1 = [('acquire', 'A'), ('acquire', 'B'), ('release', 'B'), ('release', 'A')]
    first, second = ('A', 'B') if same_order else ('B', 'A')
    t2 = [('acquire', first), ('acquire', second), ('release', second), ('release', first)]
    return [t1, t2]


def philosophers(n=5, ordered=False):
    """Philosopher i needs fork i (left) and fork (i+1) % n (right)."""
    ts = []
    for i in range(n):
        a, b = i, (i + 1) % n
        if ordered and a > b:
            a, b = b, a                  # always pick up the lower-numbered fork first
        ts.append([('acquire', f'f{a}'), ('acquire', f'f{b}'), ('emit', str(i)),
                   ('release', f'f{b}'), ('release', f'f{a}')])
    return ts


def real_deadlock(n=5, nap=0.05, patience=0.5, ordered=False):
    """Real threads and locks.
    Left fork first: everybody picks up a fork, waits until all have one (a Barrier),
    then looks at the fork they need next.  Returns who found it taken - that moment
    is the deadlock, caught without hanging the program.
    Lower-numbered fork first: everybody picks a fork, naps, then reaches for the
    other one with a timeout.  Returns who had to give up."""
    forks = [threading.Lock() for _ in range(n)]
    stuck = []
    all_hold_one = threading.Barrier(n)

    def phil(i):
        a, b = i, (i + 1) % n
        if ordered and a > b:
            a, b = b, a
        with forks[a]:
            if not ordered:
                all_hold_one.wait()
                if forks[b].locked():
                    stuck.append(i)
                all_hold_one.wait()           # nobody lets go before everybody looked
                return
            time.sleep(nap)
            if forks[b].acquire(timeout=patience):
                forks[b].release()
            else:
                stuck.append(i)
    run_threads(phil, [(i,) for i in range(n)])
    return sorted(stuck)


class DiningPhilosophers:
    """LeetCode 1226: forks are taken lowest-number first, so no cycle can form."""

    def __init__(self, n=5):
        self.n = n
        self.forks = [threading.Lock() for _ in range(n)]

    def wantsToEat(self, philosopher, pickLeftFork, pickRightFork, eat,
                   putLeftFork, putRightFork):
        left, right = philosopher, (philosopher + 1) % self.n
        lo, hi = min(left, right), max(left, right)
        with self.forks[lo], self.forks[hi]:
            pickLeftFork()
            pickRightFork()
            eat()
            putLeftFork()
            putRightFork()


# ---------------------------------------------------------------- 5. LeetCode 1114
def print_in_order_model(synced: bool):
    names = ['first', 'second', 'third']
    if not synced:
        return [[('emit', w[0])] for w in names]
    return [[('emit', 'f'), ('set', 'e1')],
            [('wait', 'e1'), ('emit', 's'), ('set', 'e2')],
            [('wait', 'e2'), ('emit', 't')]]


class Foo:
    """LeetCode 1114: second() waits for first(), third() waits for second()."""

    def __init__(self):
        self.done1 = threading.Event()
        self.done2 = threading.Event()

    def first(self, printFirst):
        printFirst()
        self.done1.set()

    def second(self, printSecond):
        self.done1.wait()
        printSecond()
        self.done2.set()

    def third(self, printThird):
        self.done2.wait()
        printThird()


# ---------------------------------------------------------------- 6. asyncio
async def fetch_await(i, delay, log, t0):
    log.append((i, 'start', time.perf_counter() - t0))
    await asyncio.sleep(delay)          # hands control back to the event loop
    log.append((i, 'end', time.perf_counter() - t0))
    return i


async def fetch_blocking(i, delay, log, t0):
    log.append((i, 'start', time.perf_counter() - t0))
    time.sleep(delay)                   # blocks the only thread the loop has
    log.append((i, 'end', time.perf_counter() - t0))
    return i


def run_coro(coro):
    """asyncio.run(coro) - also from inside Jupyter, which already runs an event loop
    (asyncio.run refuses to start a second loop in the same thread)."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    with ThreadPoolExecutor(1) as ex:
        return ex.submit(asyncio.run, coro).result()


def run_async(fetch, n=10, delay=0.1, limit=None):
    """Run n fetches with asyncio.gather (at most `limit` at a time).
    Returns (seconds, results, log)."""
    log = []

    async def main():
        t0 = time.perf_counter()
        sem = asyncio.Semaphore(limit or n)

        async def one(i):
            async with sem:
                return await fetch(i, delay, log, t0)
        res = await asyncio.gather(*(one(i) for i in range(n)))
        return time.perf_counter() - t0, res
    secs, res = run_coro(main())
    return secs, res, log


# ---------------------------------------------------------------- 7. LeetCode 1242
def hostname(url: str) -> str:
    return url.split('/')[2]


def make_web():
    """A small site: news.yahoo.com pages link to each other and to other hosts."""
    h = 'http://news.yahoo.com'
    links = {
        f'{h}/': [f'{h}/world', f'{h}/sport', f'{h}/tech', 'http://news.google.com/'],
        f'{h}/world': [f'{h}/world/asia', f'{h}/world/europe', f'{h}/'],
        f'{h}/sport': [f'{h}/sport/f1', f'{h}/sport/tennis', 'http://yahoo.com/sport'],
        f'{h}/tech': [f'{h}/tech/ai', f'{h}/tech/chips', f'{h}/world'],
        f'{h}/world/asia': [f'{h}/world/asia/japan', f'{h}/world/asia/taiwan'],
        f'{h}/world/europe': [f'{h}/world/europe/france'],
        f'{h}/sport/f1': [f'{h}/sport/f1/monaco'],
        f'{h}/sport/tennis': [],
        f'{h}/tech/ai': [f'{h}/tech/ai/llm', 'http://news.yahoo.com.evil.example/ai'],
        f'{h}/tech/chips': [f'{h}/tech/ai'],
        f'{h}/world/asia/japan': [],
        f'{h}/world/asia/taiwan': [f'{h}/'],
        f'{h}/world/europe/france': [],
        f'{h}/sport/f1/monaco': [],
        f'{h}/tech/ai/llm': [f'{h}/tech/ai/llm/agents'],
        f'{h}/tech/ai/llm/agents': [],
        'http://news.google.com/': ['http://news.google.com/a'],
        'http://yahoo.com/sport': [],
        'http://news.yahoo.com.evil.example/ai': [],
    }
    return links


class HtmlParser:
    """LeetCode's interface; each call costs `latency` seconds, like a real fetch."""

    def __init__(self, links, latency=0.05):
        self.links, self.latency, self.calls = links, latency, 0
        self._lock = threading.Lock()

    def getUrls(self, url):
        with self._lock:
            self.calls += 1
        time.sleep(self.latency)
        return list(self.links.get(url, []))


def crawl_serial(startUrl, parser):
    host, seen, stack = hostname(startUrl), {startUrl}, [startUrl]
    while stack:
        for v in parser.getUrls(stack.pop()):
            if hostname(v) == host and v not in seen:
                seen.add(v)
                stack.append(v)
    return seen


def crawl_substring_bug(startUrl, parser):
    """Checks `host in url` instead of comparing host names exactly."""
    host, seen, stack = hostname(startUrl), {startUrl}, [startUrl]
    while stack:
        for v in parser.getUrls(stack.pop()):
            if host in v and v not in seen:
                seen.add(v)
                stack.append(v)
    return seen


class Solution1242:
    """Breadth-first by waves: every page of the current wave is fetched at once."""

    def crawl(self, startUrl, htmlParser, workers=16):
        hostname = lambda url: url.split('/')[2]     # self-contained, as LeetCode needs
        host, seen, frontier, waves = hostname(startUrl), {startUrl}, [startUrl], 0
        with ThreadPoolExecutor(workers) as pool:
            while frontier:
                waves += 1
                nxt = []
                for urls in pool.map(htmlParser.getUrls, frontier):
                    for v in urls:
                        if hostname(v) == host and v not in seen:
                            seen.add(v)
                            nxt.append(v)
                frontier = nxt
        self.waves = waves
        return list(seen)


Solution = Solution1242          # the class name LeetCode expects


def bfs_depth(links, start):
    """Longest shortest-path (in links) from start inside the start host + 1 = waves."""
    host, dist, q = hostname(start), {start: 1}, [start]
    for u in q:
        for v in links.get(u, []):
            if hostname(v) == host and v not in dist:
                dist[v] = dist[u] + 1
                q.append(v)
    return max(dist.values())


# ---------------------------------------------------------------- 8. work and depth
class Dag:
    """A computation as a graph: every node is one addition, its inputs are earlier
    nodes or input values.  work = number of additions, depth = longest chain."""

    def __init__(self, n_inputs):
        self.n_inputs = n_inputs
        self.deps = []                      # deps[k] for addition node k
        self.level = []                     # longest chain ending at node k

    def inp(self, i):
        return ('in', i)

    def add(self, a, b):
        lv = 1 + max(self.level[x[1]] if x[0] == 'op' else 0 for x in (a, b))
        self.deps.append((a, b))
        self.level.append(lv)
        return ('op', len(self.deps) - 1)

    @property
    def work(self):
        return len(self.deps)

    @property
    def depth(self):
        return max(self.level, default=0)

    def evaluate(self, values, outputs):
        val = []
        get = lambda x: values[x[1]] if x[0] == 'in' else val[x[1]]
        for a, b in self.deps:
            val.append(get(a) + get(b))
        return [get(o) if o is not None else 0 for o in outputs]

    def schedule(self, p):
        """Greedy: at every tick run up to p additions whose inputs are ready
        (lowest node id first).  Returns the list of ticks, each a list of node ids."""
        import heapq
        waiting = [sum(x[0] == 'op' for x in d) for d in self.deps]
        users = defaultdict(list)
        for k, d in enumerate(self.deps):
            for x in d:
                if x[0] == 'op':
                    users[x[1]].append(k)
        ready = [k for k in range(self.work) if waiting[k] == 0]
        heapq.heapify(ready)
        ticks = []
        while ready:
            run = [heapq.heappop(ready) for _ in range(min(p, len(ready)))]
            ticks.append(run)
            for k in run:
                for u in users[k]:
                    waiting[u] -= 1
                    if waiting[u] == 0:
                        heapq.heappush(ready, u)
        return ticks


def sum_chain(n):
    g = Dag(n)
    acc = g.inp(0)
    for i in range(1, n):
        acc = g.add(acc, g.inp(i))
    return g, [acc]


def sum_tree(n):
    g = Dag(n)
    level = [g.inp(i) for i in range(n)]
    while len(level) > 1:
        nxt = [g.add(level[i], level[i + 1]) for i in range(0, len(level) - 1, 2)]
        if len(level) % 2:
            nxt.append(level[-1])
        level = nxt
    return g, level


def scan_serial(n):
    g = Dag(n)
    out = [g.inp(0)]
    for i in range(1, n):
        out.append(g.add(out[-1], g.inp(i)))
    return g, out


def scan_hillis_steele(n):
    """Inclusive scan: at round d every position i >= 2^d adds the value 2^d back."""
    g = Dag(n)
    cur = [g.inp(i) for i in range(n)]
    d = 1
    while d < n:
        cur = [cur[i] if i < d else g.add(cur[i - d], cur[i]) for i in range(n)]
        d *= 2
    return g, cur


def scan_blelloch(n):
    """Exclusive scan (n a power of two): an up-sweep that builds partial sums in a
    tree, then a down-sweep that hands each subtree the sum of everything left of it."""
    g = Dag(n)
    cur = [g.inp(i) for i in range(n)]
    d = 1
    while d < n:
        for i in range(2 * d - 1, n, 2 * d):
            cur[i] = g.add(cur[i - d], cur[i])
        d *= 2
    cur[n - 1] = None                               # the identity, 0
    d = n // 2
    while d >= 1:
        for i in range(2 * d - 1, n, 2 * d):
            left = cur[i - d]
            cur[i - d] = cur[i]
            cur[i] = left if cur[i] is None else g.add(left, cur[i])
        d //= 2
    return g, cur


def brent_bounds(work, depth, p):
    return max(math.ceil(work / p), depth), math.ceil(work / p) + depth


def amdahl(serial_fraction, p):
    return 1 / (serial_fraction + (1 - serial_fraction) / p)


# ---------------------------------------------------------------- 9. map-reduce
VOCAB = ('the of and to in a is that for it as was with be by on not he this are '
         'or his from at which but have an they you were her she there one all we '
         'their been has would when who will more if no out so said what up its '
         'about into than them can only other new some could time these two may').split()


def make_corpus(docs=40, words=400, seed=43):
    rng = random.Random(seed)
    weights = [1 / (r + 1) for r in range(len(VOCAB))]       # Zipf: a few words dominate
    return [rng.choices(VOCAB, weights, k=words) for _ in range(docs)]


def stable_bucket(word, reducers, salt=b''):
    """Same word -> same reducer, in every process (blake2b, not hash())."""
    h = hashlib.blake2b(word.encode(), digest_size=8, salt=salt.ljust(16, b'\0')[:16])
    return int.from_bytes(h.digest(), 'big') % reducers


def map_doc(doc, reducers=4, salt=b''):
    """Map + combine: count inside the document, then split the counts by reducer."""
    counts = Counter(doc)
    parts = [Counter() for _ in range(reducers)]
    for w, c in counts.items():
        parts[stable_bucket(w, reducers, salt)][w] += c
    return parts


def map_doc_builtin_hash(doc, reducers=4):
    """The same mapper with Python's hash(): its value is randomised per process."""
    parts = [Counter() for _ in range(reducers)]
    for w, c in Counter(doc).items():
        parts[hash(w) % reducers][w] += c
    return parts


def reduce_parts(all_parts, reducers=4):
    """Shuffle (reducer r gets part r of every mapper) + reduce (sum), then write the
    reducers' outputs one after another into one table, as a real job's output files."""
    out = {}
    for r in range(reducers):
        total = Counter()
        for parts in all_parts:
            total.update(parts[r])
        out.update(total)                # a word seen by two reducers: last one wins
    return out


def mapreduce(docs, reducers=4, workers=2, salts=None, ctx=None, mapper=None):
    if salts is not None:                # simulate mappers that disagree on the bucket
        all_parts = [map_doc(d, reducers, salts[i % len(salts)]) for i, d in enumerate(docs)]
    else:
        with (ctx or mp).Pool(workers) as pool:
            all_parts = pool.starmap(mapper or map_doc, [(d, reducers) for d in docs])
    return reduce_parts(all_parts, reducers)


def mean_of_means(chunks):
    return sum(sum(c) / len(c) for c in chunks) / len(chunks)


def mean_by_sum_count(chunks):
    s, n = reduce(lambda a, b: (a[0] + b[0], a[1] + b[1]), ((sum(c), len(c)) for c in chunks))
    return s / n


def tree_reduce(xs, op):
    while len(xs) > 1:
        xs = [op(xs[i], xs[i + 1]) if i + 1 < len(xs) else xs[i] for i in range(0, len(xs), 2)]
    return xs[0]


def square(x):
    return x * x


def lc1226_run(n=5, meals=3):
    """Every philosopher eats `meals` times; returns (meals eaten, action log)."""
    table, log = DiningPhilosophers(n), []

    def phil(i):
        left, right = i, (i + 1) % n
        for _ in range(meals):
            table.wantsToEat(i, lambda: log.append((i, 'pick', left)),
                             lambda: log.append((i, 'pick', right)),
                             lambda: log.append((i, 'eat', None)),
                             lambda: log.append((i, 'put', left)),
                             lambda: log.append((i, 'put', right)))
    run_threads(phil, [(i,) for i in range(n)])
    return sum(1 for _, a, _ in log if a == 'eat'), log


def forks_never_shared(log) -> bool:
    holder = {}
    for who, action, fork in log:
        if action == 'pick':
            if fork in holder:
                return False
            holder[fork] = who
        elif action == 'put':
            if holder.pop(fork, None) != who:
                return False
    return not holder


def lc1114_all_orders():
    from itertools import permutations
    outs = []
    for order in permutations(range(3)):
        foo, out = Foo(), []
        calls = [lambda: foo.first(lambda: out.append('first')),
                 lambda: foo.second(lambda: out.append('second')),
                 lambda: foo.third(lambda: out.append('third'))]
        ts = [threading.Thread(target=calls[k]) for k in order]
        for t in ts:
            t.start()
            time.sleep(0.005)            # give the early starters a head start
        for t in ts:
            t.join()
        outs.append(''.join(out))
    return outs


def check() -> None:
    # exact model-checker counts
    c2 = xcount(count_schedules([incrementer()] * 2, {'x': 0}))
    assert c2 == Counter({1: 4, 2: 2})
    c3 = xcount(count_schedules([incrementer()] * 3, {'x': 0}))
    assert c3 == Counter({1: 48, 2: 36, 3: 6})
    assert xcount(count_schedules([incrementer(locked=True)] * 3, {'x': 0})) == Counter({3: 6})
    assert count_schedules(two_lock_threads(False), {})[DEADLOCK] == 2
    assert count_schedules(two_lock_threads(True), {})[DEADLOCK] == 0
    assert reachable_states(philosophers(5), {}) == (9302, 1)
    assert reachable_states(philosophers(5, ordered=True), {})[1] == 0
    c = count_schedules(philosophers(5), {})
    assert c[DEADLOCK] == 120 and sum(c.values()) == 2_430_842_120
    assert len(count_schedules(print_in_order_model(False), {})) == 6
    assert [o[1] for o in count_schedules(print_in_order_model(True), {})] == ['fst']
    # real threads where the answer must not depend on timing
    assert hammer(bump_with_call, lock=threading.Lock()) == 800_000
    assert set(lc1114_all_orders()) == {'firstsecondthird'}
    meals, log = lc1226_run()
    assert meals == 15 and forks_never_shared(log)
    assert real_deadlock() == list(range(5)) and real_deadlock(ordered=True) == []
    # crawler
    web, start = make_web(), 'http://news.yahoo.com/'
    want = crawl_serial(start, HtmlParser(web, 0))
    assert len(want) == 16 and all(hostname(u) == 'news.yahoo.com' for u in want)
    sol = Solution1242()
    assert set(sol.crawl(start, HtmlParser(web, 0))) == want
    assert sol.waves == bfs_depth(web, start) == 5
    assert len(crawl_substring_bug(start, HtmlParser(web, 0))) == 17
    # work / depth: the DAGs compute the right answers
    from itertools import accumulate as acc
    for n in (8, 16, 64):
        vals = list(range(3, 3 + n))
        assert sum_chain(n)[0].evaluate(vals, sum_chain(n)[1]) == [sum(vals)]
        g, o = sum_tree(n)
        assert g.evaluate(vals, o) == [sum(vals)]
        for b in (scan_serial, scan_hillis_steele):
            g, o = b(n)
            assert g.evaluate(vals, o) == list(acc(vals))
        g, o = scan_blelloch(n)
        assert g.evaluate(vals, o) == [0] + list(acc(vals))[:-1]
    for b in (sum_chain, sum_tree, scan_serial, scan_hillis_steele, scan_blelloch):
        g, _ = b(1024)
        for p in (1, 4, 64, 1024):
            lo, hi = brent_bounds(g.work, g.depth, p)
            assert lo <= len(g.schedule(p)) <= hi
    assert (sum_tree(1024)[0].work, sum_tree(1024)[0].depth) == (1023, 10)
    assert (scan_hillis_steele(1024)[0].work, scan_blelloch(1024)[0].work) == (9217, 2036)
    # map-reduce
    docs = make_corpus()
    truth = Counter(w for d in docs for w in d)
    assert reduce_parts([map_doc(d) for d in docs]) == dict(truth)
    assert mean_by_sum_count([[1, 2], [3]]) == 2 and mean_of_means([[1, 2], [3]]) == 2.25
    assert tree_reduce([0.1] * 10, lambda a, b: a + b) != sum([0.1] * 10)


# ---------------------------------------------------------------- the walkthrough
def header(t):
    print('\n' + '=' * 72 + '\n' + t + '\n' + '=' * 72)


def xcount(counter):
    """Counter keyed by the final x (or DEADLOCK)."""
    out = Counter()
    for k, v in counter.items():
        out[final_x(k)] += v
    return out


def main() -> None:
    header('1. The GIL: threads overlap waiting, not computing')
    print(f'this machine: {os.cpu_count()} cores, Python {sys.version.split()[0]}, '
          f'GIL switch interval {sys.getswitchinterval() * 1000:.0f} ms')
    t = gil_timings()
    for k in ('cpu serial', 'cpu threads', 'cpu processes'):
        print(f'  2 CPU jobs, {k[4:]:<9}: {t[k]:.3f} s   speed-up {t["cpu serial"] / t[k]:.2f}x')
    for k in ('io serial', 'io threads'):
        print(f'  20 waits of 50 ms, {k[3:]:<7}: {t[k]:.3f} s   speed-up {t["io serial"] / t[k]:.1f}x')

    header('2. A race you can run')
    print('  4 threads x 200,000 increments, so the right answer is 800,000:')
    print(f'  box.v += 1                      -> {hammer(bump_inline):,}')
    lost = hammer(bump_with_call)
    print(f'  read, call a function, write    -> {lost:,}  ({1 - lost / 800_000:.0%} of the updates lost)')
    print(f'  same, inside `with lock:`       -> {hammer(bump_with_call, lock=threading.Lock()):,}')
    print('  (in this CPython the thread switch only happens at loop jumps and calls,')
    print('   so `+=` is never split here - an accident of the version, not a promise)')

    header('3. Every interleaving, counted')
    two = list(every_schedule([incrementer(), incrementer()], {'x': 0}))
    names = {0: 'A', 1: 'B'}
    ops = ['read', 'write']
    for sched, out in two:
        pos, line = [0, 0], []
        for i in sched:
            line.append(f'{names[i]}.{ops[pos[i]]}')
            pos[i] += 1
        print(f'  {"  ".join(line):<36} x = {final_x(out)}')
    for k in (2, 3, 4):
        c = xcount(count_schedules([incrementer()] * k, {'x': 0}))
        tot = sum(c.values())
        print(f'  {k} threads, one x += 1 each: {tot:>5} interleavings, correct ({k}) in '
              f'{c[k]:>3} = {c[k] / tot:.1%};  outcomes {dict(sorted(c.items()))}')
    c = xcount(count_schedules([incrementer(times=2)] * 2, {'x': 0}))
    print(f'  2 threads, two x += 1 each: {sum(c.values())} interleavings, x ends as {dict(sorted(c.items()))}')
    c = xcount(count_schedules([incrementer(locked=True)] * 3, {'x': 0}))
    print(f'  3 threads with a lock: {sum(c.values())} interleavings possible, all give {list(c)}')

    header('4. Deadlock')
    for same in (False, True):
        c = count_schedules(two_lock_threads(same), {})
        print(f'  two threads, two locks, {"same" if same else "opposite"} order: '
              f'{sum(c.values())} interleavings, {c[DEADLOCK]} deadlock')
    print('  dining philosophers, everybody takes the left fork first:')
    for n in (2, 3, 4, 5):
        st, dead = reachable_states(philosophers(n), {})
        c = count_schedules(philosophers(n), {})
        tot = sum(c.values())
        print(f'    n = {n}: {st:>5,} states, {dead} deadlocked; {c[DEADLOCK]:,} of {tot:,} '
              f'interleavings deadlock (1 in {tot // c[DEADLOCK]:,})')
    st, dead = reachable_states(philosophers(5, ordered=True), {})
    print(f'    n = 5, lower-numbered fork first: {st:,} states, {dead} deadlocked')
    print(f'  real threads, left fork first: who holds one fork and finds the other taken '
          f'{real_deadlock()}')
    print(f'  real threads, lower fork first (50 ms nap, 0.5 s patience): who had to give up '
          f'{real_deadlock(ordered=True)}')
    meals, log = lc1226_run()
    print(f'LC 1226: 5 philosophers x 3 meals each -> {meals} meals, every fork used by one at a time: '
          f'{forks_never_shared(log)}')

    header('5. LeetCode 1114 - Print in Order')
    for synced in (False, True):
        c = count_schedules(print_in_order_model(synced), {})
        outs = sorted(o[1] for o in c)
        print(f'  {"with events" if synced else "no sync":<11}: possible outputs {outs}')
    print(f'  real threads, all 6 start orders: {sorted(set(lc1114_all_orders()))}')

    header('6. asyncio: one thread, many waits')
    for name, fetch, limit in (('await asyncio.sleep', fetch_await, None),
                               ('time.sleep inside async def', fetch_blocking, None),
                               ('await, Semaphore(3)', fetch_await, 3)):
        secs, res, log = run_async(fetch, limit=limit)
        starts = [round(t, 1) for i, kind, t in log if kind == 'start']
        print(f'  10 x 0.1 s, {name:<28}: {secs:.2f} s   start times {starts}')

    header('7. LeetCode 1242 - Web Crawler Multithreaded')
    web, start = make_web(), 'http://news.yahoo.com/'
    p = HtmlParser(web)
    t0 = time.perf_counter()
    got = crawl_serial(start, p)
    ts = time.perf_counter() - t0
    print(f'  serial:  {len(got)} pages, {p.calls} fetches, {ts:.2f} s')
    p = HtmlParser(web)
    sol = Solution1242()
    t0 = time.perf_counter()
    got2 = sol.crawl(start, p)
    tt = time.perf_counter() - t0
    print(f'  threads: {len(got2)} pages, {p.calls} fetches in {sol.waves} waves, {tt:.2f} s '
          f'(= waves x 50 ms; the longest link chain is {bfs_depth(web, start)} pages)')
    bad = crawl_substring_bug(start, HtmlParser(web, 0))
    print(f'  `host in url` instead of ==: {len(bad)} pages, extra {sorted(set(bad) - set(got))}')

    header('8. Work and depth')
    builders = (('sum, loop', sum_chain), ('sum, tree', sum_tree), ('scan, loop', scan_serial),
                ('scan, Hillis-Steele', scan_hillis_steele), ('scan, Blelloch', scan_blelloch))
    for n, ps in ((16, (1, 2, 4, 8, 16)), (1024, (1, 4, 16, 64, 256, 1024))):
        print(f'  n = {n}: work, depth, and ticks T_p with p workers')
        for name, b in builders:
            g, _ = b(n)
            tp = '  '.join(f'T_{p}={len(g.schedule(p))}' for p in ps)
            print(f'    {name:<20} W={g.work:<5} D={g.depth:<5} {tp}')
    hs, _ = scan_hillis_steele(1024)
    bl, _ = scan_blelloch(1024)
    lo, hi = 1, 1024                     # binary search: the first p where it wins
    while lo < hi:
        mid = (lo + hi) // 2
        if len(hs.schedule(mid)) < len(bl.schedule(mid)):
            hi = mid
        else:
            lo = mid + 1
    cross = lo
    print(f'  Hillis-Steele beats Blelloch only from p = {cross} workers')
    print('  Brent: max(W/p, D) <= T_p <= W/p + D;  Amdahl with a serial share s:')
    for s in (0.05, 0.10):
        print(f'    s = {s:.0%}: ' + '  '.join(f'p={p}: {amdahl(s, p):.1f}x' for p in (2, 4, 8, 16, 64))
              + f'   limit {1 / s:.0f}x')

    header('9. Map-reduce')
    docs = make_corpus()
    truth = Counter(w for d in docs for w in d)
    print(f'  {len(docs)} documents, {sum(truth.values()):,} words, {len(truth)} distinct; '
          f'top 3 {truth.most_common(3)}')
    mr = mapreduce(docs)
    print(f'  map-reduce on 2 processes, 4 reducers == Counter: {mr == dict(truth)}')
    sim = mapreduce(docs, salts=[b'mapper-0', b'mapper-1'])
    wrong = sorted((w for w in truth if sim[w] != truth[w]), key=lambda w: -truth[w])
    print(f'  two mappers that bucket words differently: {len(wrong)} of {len(truth)} words wrong, '
          f'e.g. "{wrong[0]}" {sim[wrong[0]]} instead of {truth[wrong[0]]}')
    spawn = mapreduce(docs, ctx=mp.get_context('spawn'), mapper=map_doc_builtin_hash)
    fork = mapreduce(docs, ctx=mp.get_context('fork'), mapper=map_doc_builtin_hash)
    print(f'  hash() buckets, fork workers: '
          f'{sum(fork.get(w) != truth[w] for w in truth)} words wrong')
    print(f'  hash() buckets, spawn workers (default on Windows, macOS): '
          f'{sum(spawn.get(w) != truth[w] for w in truth)} words wrong (this run)')
    chunks = [[120, 80, 100], [400], [90, 110, 100, 95, 105, 100]]
    print(f'  latency chunks of {[len(c) for c in chunks]}: mean of means {mean_of_means(chunks):.2f}, '
          f'true mean {mean_by_sum_count(chunks):.2f}')
    tens = [0.1] * 10
    print(f'  ten 0.1s: loop {sum(tens)!r}, tree {tree_reduce(tens, lambda a, b: a + b)!r}, '
          f'math.fsum {math.fsum(tens)!r}')
    nums = list(range(50_000))
    with mp.Pool(2) as pool:
        pool.map(square, range(10))
        t_ser = best_of(lambda: [square(x) for x in nums])
        t_c1 = best_of(lambda: pool.map(square, nums, chunksize=1), 1)
        t_cd = best_of(lambda: pool.map(square, nums))
    print(f'  squaring 50,000 numbers: loop {t_ser * 1000:.0f} ms, Pool chunksize=1 '
          f'{t_c1 * 1000:.0f} ms, Pool default chunks {t_cd * 1000:.0f} ms')

    check()
    print('\nall assertions passed')


if __name__ == '__main__':
    main()
