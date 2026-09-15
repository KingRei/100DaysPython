"""Day 32 - Bloom filter, HyperLogLog and Count-Min sketch.

Three structures that answer set questions in a fixed amount of memory, no
matter how much data goes in.  None of them stores the data; each one keeps a
few bits per item and pays for it with a bounded, *one-sided* error:

    Bloom filter     "have I seen x?"          may say yes when the answer is no
    HyperLogLog      "how many distinct?"      relative error, either direction
    Count-Min sketch "how often was x?"        never under-counts, may over-count

Run me:  python probabilistic.py
"""

import hashlib
import heapq
import math
import random
import sys
from collections import Counter


# ====================================================================
# hashing
# ====================================================================

def hash64(item, seed=0):
    """A stable 64-bit hash.

    Python's built-in hash() is randomised per process for str and bytes
    (PYTHONHASHSEED), so a sketch built in one process would disagree with the
    same sketch built in another.  Every structure on this page needs a hash
    that is a pure function of the bytes, so they all come through here.
    """
    h = hashlib.blake2b(repr(item).encode(), digest_size=16,
                        salt=seed.to_bytes(2, 'little').ljust(16, b'\0')[:16])
    return int.from_bytes(h.digest()[:8], 'little')


def hash_pair(item):
    """Two independent 64-bit hashes out of one digest.

    Kirsch-Mitzenmacher: k hash functions can be simulated with two, using
    g_i(x) = h1(x) + i*h2(x) + i*i, with no measurable loss in false positive
    rate.  One digest instead of k is the whole reason Bloom filters are fast.
    """
    d = hashlib.blake2b(repr(item).encode(), digest_size=16).digest()
    return int.from_bytes(d[:8], 'little'), int.from_bytes(d[8:], 'little')


# ====================================================================
# 1. Bloom filter
# ====================================================================

def optimal_bits(n, fpr):
    """m = -n ln(p) / (ln 2)^2 - the bits needed for n items at rate p."""
    return max(8, int(math.ceil(-n * math.log(fpr) / (math.log(2) ** 2))))


def optimal_hashes(m, n):
    """k = (m/n) ln 2 - too few hashes underuses the bits, too many fills them."""
    return max(1, int(round(m / n * math.log(2))))


class BloomFilter:
    """A bit array plus k hash functions.  Membership, never the members."""

    def __init__(self, n=1000, fpr=0.01, m=None, k=None):
        self.m = m if m is not None else optimal_bits(n, fpr)
        self.k = k if k is not None else optimal_hashes(self.m, n)
        self.bits = bytearray((self.m + 7) // 8)
        self.n = 0

    def indices(self, item):
        h1, h2 = hash_pair(item)
        return [(h1 + i * h2 + i * i) % self.m for i in range(self.k)]

    def add(self, item):
        for i in self.indices(item):
            self.bits[i >> 3] |= 1 << (i & 7)
        self.n += 1

    def __contains__(self, item):
        return all(self.bits[i >> 3] >> (i & 7) & 1 for i in self.indices(item))

    def bits_set(self):
        return sum(bin(b).count('1') for b in self.bits)

    def fill_ratio(self):
        return self.bits_set() / self.m

    def expected_fpr(self):
        """(1 - e^(-kn/m))^k - what the filter promises at its current load."""
        return (1 - math.exp(-self.k * self.n / self.m)) ** self.k

    def estimated_items(self):
        """Count the items back out of the fill ratio - Swamidass & Baldi."""
        f = self.fill_ratio()
        if f >= 1.0:
            return float('inf')
        return -self.m / self.k * math.log(1 - f)

    def nbytes(self):
        return len(self.bits)


def bloom_report(n=10000, fpr=0.01, trials=100000, seed=1):
    """Build a filter at its design load and measure the promise."""
    rnd = random.Random(seed)
    bf = BloomFilter(n=n, fpr=fpr)
    present = ['item-%d' % i for i in range(n)]
    for x in present:
        bf.add(x)
    misses = sum(('absent-%d-%d' % (i, rnd.randrange(1 << 30))) in bf
                 for i in range(trials))
    exact = set(present)
    return {
        'n': n, 'm': bf.m, 'k': bf.k, 'bytes': bf.nbytes(),
        'bits_per_item': bf.m / n,
        'fill': bf.fill_ratio(),
        'design_fpr': fpr,
        'expected_fpr': bf.expected_fpr(),
        'measured_fpr': misses / trials,
        'false_negatives': sum(x not in bf for x in present),
        'estimated_items': bf.estimated_items(),
        'exact_set_bytes': sys.getsizeof(exact)
                           + sum(sys.getsizeof(s) for s in present),
    }


def overfill_report(design_n=1000, fpr=0.01, trials=20000, seed=2):
    """The rate is a promise about a load, not about the structure."""
    rnd = random.Random(seed)
    bf = BloomFilter(n=design_n, fpr=fpr)
    rows, added = [], 0
    for mult in (0.5, 1, 2, 4, 8):
        while added < design_n * mult:
            bf.add('x-%d' % added)
            added += 1
        miss = sum(('q-%d-%d' % (i, rnd.randrange(1 << 30))) in bf
                   for i in range(trials))
        rows.append({'load': mult, 'items': added, 'fill': bf.fill_ratio(),
                     'expected': bf.expected_fpr(), 'measured': miss / trials})
    return rows


def k_sweep(n=2000, bits_per_item=10, trials=50000, seed=3):
    """Why k = (m/n) ln 2: too few hashes and too many are both worse."""
    rnd = random.Random(seed)
    m = n * bits_per_item
    best = optimal_hashes(m, n)
    rows = []
    for k in range(1, 13):
        bf = BloomFilter(m=m, k=k)
        for i in range(n):
            bf.add('k-%d' % i)
        miss = sum(('z-%d-%d' % (i, rnd.randrange(1 << 30))) in bf
                   for i in range(trials))
        rows.append({'k': k, 'fill': bf.fill_ratio(),
                     'theory': bf.expected_fpr(), 'measured': miss / trials,
                     'best': k == best})
    return rows, best


class CountingBloom(BloomFilter):
    """Counters instead of bits, so a delete can undo exactly one add."""

    def __init__(self, n=1000, fpr=0.01, m=None, k=None, width=4):
        super().__init__(n=n, fpr=fpr, m=m, k=k)
        self.counters = [0] * self.m
        self.cap = (1 << width) - 1
        self.saturated = 0

    def add(self, item):
        for i in self.indices(item):
            if self.counters[i] < self.cap:
                self.counters[i] += 1
            else:
                self.saturated += 1
        self.n += 1

    def delete(self, item):
        if item not in self:
            return False
        for i in self.indices(item):
            if 0 < self.counters[i] < self.cap:
                self.counters[i] -= 1
        self.n -= 1
        return True

    def __contains__(self, item):
        return all(self.counters[i] for i in self.indices(item))


def deletion_report(seed=4):
    """Clearing the bits of one item can delete a different one.

    A Bloom filter has no delete, and the reason is worth seeing rather than
    being told: the bits are shared, so zeroing the bits of 'y' can flip a bit
    that 'x' was relying on - and the filter's one guarantee, no false
    negatives, is gone.
    """
    words = ['alpha', 'bravo', 'charlie', 'delta', 'echo', 'foxtrot', 'golf',
             'hotel', 'india', 'juliet', 'kilo', 'lima', 'mike', 'november']
    bf = BloomFilter(m=64, k=3)
    for w in words:
        bf.add(w)
    victim = collide = None
    for y in words:
        ys = set(bf.indices(y))
        for x in words:
            if x != y and set(bf.indices(x)) & ys:
                victim, collide = x, y
                break
        if victim:
            break
    for i in bf.indices(collide):                 # the naive "delete"
        bf.bits[i >> 3] &= ~(1 << (i & 7)) & 0xff
    broken = [w for w in words if w != collide and w not in bf]

    cb = CountingBloom(m=64, k=3)
    for w in words:
        cb.add(w)
    cb.delete(collide)
    still = [w for w in words if w != collide and w not in cb]
    return {
        'deleted': collide, 'shared_with': victim,
        'shared_index': sorted(set(bf.indices(victim)) & set(bf.indices(collide))),
        'false_negatives': broken,
        'counting_false_negatives': still,
        'counting_still_finds_deleted': collide in cb,
    }


def dedup_report(docs=20000, dup_rate=0.3, fpr=0.01, seed=5):
    """Dedup is the misuse case: a false positive silently drops a document."""
    rnd = random.Random(seed)
    stream, uniq = [], 0
    for i in range(docs):
        if stream and rnd.random() < dup_rate:
            stream.append(rnd.choice(stream))
        else:
            stream.append('doc-%d' % uniq)
            uniq += 1
    bf = BloomFilter(n=uniq, fpr=fpr)
    kept_bloom, seen = [], set()
    for d in stream:
        if d not in bf:
            bf.add(d)
            kept_bloom.append(d)
    kept_exact = [d for d in stream if not (d in seen or seen.add(d))]
    lost = set(kept_exact) - set(kept_bloom)
    return {
        'documents': docs, 'unique': uniq,
        'kept_exact': len(kept_exact), 'kept_bloom': len(kept_bloom),
        'lost': len(lost), 'lost_pct': 100 * len(lost) / uniq,
        'fpr': fpr, 'bloom_bytes': bf.nbytes(),
        'exact_bytes': sys.getsizeof(seen) + sum(sys.getsizeof(s) for s in seen),
    }


# ====================================================================
# 2. HyperLogLog
# ====================================================================

def leading_zeros_intuition(trials=2000, seed=7):
    """Why a maximum can count: flip coins, remember the longest head run.

    If n people each flip until tails, the longest run of heads seen is about
    log2(n).  So 2**(longest run) is an estimate of n - built from one number,
    not from a list of who flipped what.
    """
    rnd = random.Random(seed)
    rows = []
    for n in (8, 64, 512, 4096):
        runs = []
        for _ in range(trials // 8):
            longest = 0
            for _ in range(n):
                r = 0
                while rnd.random() < 0.5:
                    r += 1
                longest = max(longest, r)
            runs.append(longest)
        avg = sum(runs) / len(runs)
        rows.append({'n': n, 'log2n': math.log2(n), 'avg_longest_run': avg,
                     'estimate': 2 ** avg})
    return rows


class HyperLogLog:
    """m registers, each holding the longest run of leading zeros it has seen.

    The estimate is the harmonic mean of 2**register over all registers, which
    is what tames the variance: one lucky item with 20 leading zeros would
    dominate an arithmetic mean, but barely moves a harmonic one.
    """

    def __init__(self, p=14):
        self.p = p
        self.m = 1 << p
        self.reg = bytearray(self.m)
        self.alpha = {4: 0.673, 5: 0.697, 6: 0.709}.get(
            p, 0.7213 / (1 + 1.079 / self.m))

    def add(self, item):
        x = hash64(item)
        idx = x & (self.m - 1)                    # which register
        w = x >> self.p                           # the rest decides the run
        rho = 1
        while rho <= 64 - self.p and not (w >> (rho - 1)) & 1:
            rho += 1
        if rho > self.reg[idx]:
            self.reg[idx] = rho

    def count(self):
        z = sum(2.0 ** -r for r in self.reg)
        est = self.alpha * self.m * self.m / z
        zeros = self.reg.count(0)
        if est <= 2.5 * self.m and zeros:
            return self.m * math.log(self.m / zeros)   # linear counting
        return est

    def merge(self, other):
        """Union of two sets, from the two sketches, with no re-reading.

        Register-wise max, and that is the whole operation.  It is exact in the
        sense that the merged sketch equals the sketch you would have built by
        feeding it both streams - which is why HLLs can be summed across
        shards, days, or machines.
        """
        assert self.p == other.p
        out = HyperLogLog(self.p)
        out.reg = bytearray(max(a, b) for a, b in zip(self.reg, other.reg))
        return out

    def nbytes(self):
        return self.m                              # 1 byte/register here


def hll_report(p=14, sizes=(1000, 10000, 100000, 500000), seed=8):
    hll_bytes = 1 << p
    rows = []
    for n in sizes:
        h = HyperLogLog(p)
        for i in range(n):
            h.add('u-%d-%d' % (seed, i))
        est = h.count()
        exact = sys.getsizeof(set(range(n))) + 32 * n   # 32B per small str, low
        rows.append({'n': n, 'estimate': est,
                     'error_pct': 100 * (est - n) / n,
                     'hll_bytes': hll_bytes, 'exact_bytes': exact,
                     'ratio': exact / hll_bytes})
    return rows, 1.04 / math.sqrt(1 << p) * 100


def hll_intersection_report(p=14, n=200000, overlap=2000, seed=9):
    """Union is free; intersection is where the error eats the answer.

    |A and B| = |A| + |B| - |A or B| subtracts three numbers each carrying
    ~0.8% relative error.  When the sets are big and the overlap is small, the
    absolute errors are the same size as the answer.
    """
    a, b = HyperLogLog(p), HyperLogLog(p)
    for i in range(n):
        a.add('a-%d' % i)
    for i in range(n - overlap, 2 * n - overlap):
        b.add('a-%d' % i if i < n else 'b-%d' % i)
    ca, cb = a.count(), b.count()
    cu = a.merge(b).count()
    inter = ca + cb - cu
    return {'n': n, 'true_union': 2 * n - overlap, 'true_intersection': overlap,
            'est_a': ca, 'est_b': cb, 'est_union': cu,
            'union_error_pct': 100 * (cu - (2 * n - overlap)) / (2 * n - overlap),
            'est_intersection': inter,
            'intersection_error_pct': 100 * (inter - overlap) / overlap}


# ====================================================================
# 3. Count-Min sketch
# ====================================================================

class CountMinSketch:
    """d rows of w counters; every row is a complete, lossy count of the stream.

    Every row counts *everything* - the whole stream is added to every row -
    so every row is an over-estimate, and taking the minimum picks the row that
    happened to give this item the fewest collisions.
    """

    def __init__(self, epsilon=0.001, delta=0.01, w=None, d=None,
                 conservative=False):
        self.w = w if w is not None else int(math.ceil(math.e / epsilon))
        self.d = d if d is not None else int(math.ceil(math.log(1 / delta)))
        self.rows = [[0] * self.w for _ in range(self.d)]
        self.conservative = conservative
        self.total = 0

    def _cols(self, item):
        h1, h2 = hash_pair(item)
        return [(h1 + r * h2 + r * r) % self.w for r in range(self.d)]

    def add(self, item, count=1):
        cols = self._cols(item)
        if self.conservative:
            # only raise the counters that are currently at the minimum -
            # the others already over-estimate, so raising them adds error
            cur = min(self.rows[r][c] for r, c in enumerate(cols))
            for r, c in enumerate(cols):
                self.rows[r][c] = max(self.rows[r][c], cur + count)
        else:
            for r, c in enumerate(cols):
                self.rows[r][c] += count
        self.total += count

    def query(self, item):
        return min(self.rows[r][c] for r, c in enumerate(self._cols(item)))

    def nbytes(self):
        return self.w * self.d * 4                 # 32-bit counters

    def error_bound(self):
        """The guarantee: estimate <= truth + eps*N, with probability 1-delta."""
        return math.e / self.w * self.total


def zipf_stream(n_items=5000, length=200000, skew=1.1, seed=11):
    rnd = random.Random(seed)
    weights = [1.0 / (i + 1) ** skew for i in range(n_items)]
    total = sum(weights)
    cum, acc = [], 0.0
    for w in weights:
        acc += w / total
        cum.append(acc)
    from bisect import bisect_left
    return ['w-%d' % bisect_left(cum, rnd.random()) for _ in range(length)]


def cms_report(stream=None, epsilon=0.0005, delta=0.01, seed=11):
    stream = stream if stream is not None else zipf_stream(seed=seed)
    truth = Counter(stream)
    plain = CountMinSketch(epsilon, delta)
    cons = CountMinSketch(epsilon, delta, conservative=True)
    for x in stream:
        plain.add(x)
        cons.add(x)
    items = sorted(truth, key=lambda k: -truth[k])
    heavy, tail = items[:10], items[-10:]
    err = [plain.query(k) - truth[k] for k in truth]
    under = sum(plain.query(k) < truth[k] for k in truth)
    return {
        'distinct': len(truth), 'length': len(stream),
        'w': plain.w, 'd': plain.d, 'bytes': plain.nbytes(),
        'exact_bytes': sys.getsizeof(truth) + 60 * len(truth),
        'bound': plain.error_bound(),
        'max_error': max(err), 'mean_error': sum(err) / len(err),
        'under_counts': under,
        'heavy': [(k, truth[k], plain.query(k), cons.query(k)) for k in heavy[:5]],
        'tail': [(k, truth[k], plain.query(k), cons.query(k)) for k in tail[:5]],
        'heavy_rel': 100 * sum(plain.query(k) / truth[k] - 1 for k in heavy) / 10,
        'tail_rel': 100 * sum(plain.query(k) / truth[k] - 1 for k in tail) / 10,
        'cons_mean_error': sum(cons.query(k) - truth[k] for k in truth) / len(truth),
    }


# ====================================================================
# 4. LeetCode 347 - top k frequent elements, exact and streaming
# ====================================================================

def top_k_frequent(nums, k):
    """LC 347, the exact answer: count everything, then a k-sized heap.

    O(n) time and O(distinct) space - and that second term is the one that
    stops being affordable when 'nums' is a firehose rather than a list.
    """
    freq = Counter(nums)
    return [x for x, _ in heapq.nlargest(k, freq.items(), key=lambda p: p[1])]


def top_k_streaming(stream, k, epsilon=0.0005, delta=0.01, conservative=True):
    """The same question with a sketch plus a k-sized heap of candidates.

    Memory is w*d counters plus k entries, independent of how many distinct
    items go by.  The sketch never under-counts, so a genuine heavy hitter can
    never be pushed out by a lighter one - what can happen is that a rare item
    is inflated enough to sneak in.
    """
    cms = CountMinSketch(epsilon, delta, conservative=conservative)
    heap, inside = [], {}
    for x in stream:
        cms.add(x)
        c = cms.query(x)
        if x in inside:
            inside[x] = c
        elif len(heap) < k:
            inside[x] = c
        else:
            lo = min(inside, key=lambda t: inside[t])
            if c > inside[lo]:
                del inside[lo]
                inside[x] = c
    heap = sorted(inside, key=lambda t: -inside[t])
    return heap[:k], cms


def lc347_report(k=10, seed=11):
    stream = zipf_stream(seed=seed)
    exact = top_k_frequent(stream, k)
    approx, cms = top_k_streaming(stream, k)
    truth = Counter(stream)
    flat = zipf_stream(n_items=5000, length=200000, skew=0.05, seed=seed)
    exact_f = top_k_frequent(flat, k)
    approx_f, _ = top_k_streaming(flat, k)
    tf = Counter(flat)
    return {
        'k': k,
        'exact': exact, 'approx': approx,
        'same_set': set(exact) == set(approx),
        'same_order': exact == approx,
        'exact_counts': [truth[x] for x in exact],
        'approx_counts': [cms.query(x) for x in approx],
        'cms_bytes': cms.nbytes(),
        'counter_bytes': sys.getsizeof(truth) + 60 * len(truth),
        'flat_same_set': set(exact_f) == set(approx_f),
        'flat_gap': tf[exact_f[0]] - tf[exact_f[-1]],
        'flat_top': tf[exact_f[0]], 'flat_kth': tf[exact_f[k - 1]],
        'flat_missed': sorted(set(exact_f) - set(approx_f)),
    }


# ====================================================================
# 5. Choosing between them
# ====================================================================

def comparison_table(n=1000000):
    """Same n distinct items through all three, so the bytes are comparable."""
    bloom = BloomFilter(n=n, fpr=0.01)
    hll = HyperLogLog(p=14)
    cms = CountMinSketch(epsilon=0.0005, delta=0.01)
    return [
        ('set (exact)', 'membership + count', 'none',
         'yes (union)', 60 * n),
        ('Bloom filter', 'have I seen x?', 'false positive only',
         'yes (OR, same m,k)', bloom.nbytes()),
        ('HyperLogLog', 'how many distinct?', 'relative, both ways',
         'yes (register max)', hll.nbytes()),
        ('Count-Min', 'how often is x?', 'over-count only',
         'yes (add the rows)', cms.nbytes()),
    ]


def merge_demo(p=14, n=100000, shared=20000, seed=12):
    """HLL merges are exact: two shards merged == one HLL over everything.

    This is the property that makes HLL the one people actually deploy.  Each
    machine keeps 16 KB; the daily, weekly and monthly rollups are a max over
    registers, with no re-scan of the raw events.
    """
    a, b, whole = HyperLogLog(p), HyperLogLog(p), HyperLogLog(p)
    rnd = random.Random(seed)
    left = ['u-%d' % rnd.randrange(10 ** 9) for _ in range(n)]
    right = left[:shared] + ['v-%d' % rnd.randrange(10 ** 9)
                             for _ in range(n - shared)]
    for x in left:
        a.add(x)
        whole.add(x)
    for x in right:
        b.add(x)
        whole.add(x)
    merged = a.merge(b)
    return {
        'truth': len(set(left) | set(right)),
        'merged': merged.count(),
        'single_pass': whole.count(),
        'identical': merged.reg == whole.reg,
        'shard_bytes': a.nbytes(),
    }


def where_they_ship():
    return [
        ('Bloom filter',
         'Chrome Safe Browsing, Cassandra / LevelDB SSTable lookups, SPV wallets',
         'a cheap NO that skips an expensive disk or network lookup'),
        ('HyperLogLog',
         'Redis PFCOUNT, BigQuery APPROX_COUNT_DISTINCT, Presto approx_distinct',
         'unique visitors per hour / day / month, rolled up from shards'),
        ('Count-Min',
         'network flow monitoring, trending topics, per-key rate limiting',
         'heavy hitters without one counter per key'),
    ]


# ====================================================================
# main
# ====================================================================

def section(title):
    print()
    print('=' * 70)
    print(title)
    print('=' * 70)


def main():
    section('1. Bloom filter: the design formulas')
    r = bloom_report()
    print('asking for %d items at a %.0f%% false positive rate:'
          % (r['n'], 100 * r['design_fpr']))
    print('  m = %d bits (%.2f bits/item), k = %d hashes'
          % (r['m'], r['bits_per_item'], r['k']))
    print('  filter    %8d bytes' % r['bytes'])
    print('  exact set %8d bytes  (%.0fx bigger)'
          % (r['exact_set_bytes'], r['exact_set_bytes'] / r['bytes']))
    print('  bits set: %.1f%% of the array' % (100 * r['fill']))
    print('  design %.2f%%   expected %.2f%%   measured %.2f%%'
          % (100 * r['design_fpr'], 100 * r['expected_fpr'],
             100 * r['measured_fpr']))
    print('  false NEGATIVES over the %d members: %d  <- zero by construction'
          % (r['n'], r['false_negatives']))
    print('  cardinality read back out of the bit array: %.0f (true %d)'
          % (r['estimated_items'], r['n']))

    section('2. The rate is a promise about a load, not about the structure')
    print('%6s %8s %9s %11s %11s' % ('load', 'items', 'fill', 'expected',
                                     'measured'))
    for row in overfill_report():
        print('%5.1fx %8d %8.1f%% %10.2f%% %10.2f%%'
              % (row['load'], row['items'], 100 * row['fill'],
                 100 * row['expected'], 100 * row['measured']))
    print('an overfilled filter does not degrade gently - it degrades like a')
    print('power: every extra item multiplies the miss rate')

    section('3. How many hashes? (10 bits per item)')
    rows, best = k_sweep()
    print('%3s %8s %11s %11s' % ('k', 'fill', 'theory', 'measured'))
    for row in rows:
        print('%3d %7.1f%% %10.2f%% %10.2f%% %s'
              % (row['k'], 100 * row['fill'], 100 * row['theory'],
                 100 * row['measured'], '<- k = (m/n) ln 2' if row['best'] else ''))
    print('optimum at k = %d: more hashes give a query more chances to find a' % best)
    print('zero, but also fill the array faster - the two effects cross here')

    section('4. Deleting is the thing that breaks it')
    d = deletion_report()
    print('a deliberately tiny filter (m = 64 bits, k = 3, 14 words in it)')
    print('  %r and %r share bit(s) %s'
          % (d['deleted'], d['shared_with'], d['shared_index']))
    print('  clearing the 3 bits of %r ->' % d['deleted'])
    print('    plain Bloom     %d false negative(s): %s'
          % (len(d['false_negatives']), d['false_negatives']))
    print('    counting Bloom  %d false negative(s): %s'
          % (len(d['counting_false_negatives']), d['counting_false_negatives']))
    print('  the counting version still reports %r as absent: %s'
          % (d['deleted'], not d['counting_still_finds_deleted']))
    print('  price of delete: 4-bit counters = 4x the memory, and counters that')
    print('  saturate can never be decremented again')

    section('5. The misuse: Bloom dedup silently drops documents')
    g = dedup_report()
    print('%d documents, %d of them distinct, filter designed for %.0f%%'
          % (g['documents'], g['unique'], 100 * g['fpr']))
    print('  exact dedup keeps    %6d' % g['kept_exact'])
    print('  Bloom dedup keeps    %6d' % g['kept_bloom'])
    print('  DISTINCT DOCS LOST   %6d   (%.2f%% of the distinct ones)'
          % (g['lost'], g['lost_pct']))
    print('  bloom %d bytes vs exact set %d bytes'
          % (g['bloom_bytes'], g['exact_bytes']))
    print('a false positive means "already seen".  As a pre-filter in front of a')
    print('real lookup that costs one wasted read; as the final answer it costs')
    print('data that never comes back.  Same structure, same error rate, and the')
    print('difference is entirely in what happens after the yes')

    section('6. HyperLogLog: why a maximum can count')
    print('%6s %9s %16s %12s' % ('n', 'log2(n)', 'avg longest run', '2**run'))
    for row in leading_zeros_intuition():
        print('%6d %9.2f %16.2f %12.0f'
              % (row['n'], row['log2n'], row['avg_longest_run'], row['estimate']))
    print('a run of r heads turns up about once in 2^r tries, so the longest run')
    print('you have seen estimates log2(how many you tried) - and a maximum needs')
    print('one number of storage, not a list')

    section('7. HyperLogLog: 16384 registers, one byte each')
    hrows, theo = hll_report()
    print('p = 14 -> m = 16384 registers, theoretical error 1.04/sqrt(m) = %.2f%%'
          % theo)
    print('%9s %12s %9s %11s %13s %8s'
          % ('true', 'estimate', 'error', 'hll bytes', 'exact bytes', 'ratio'))
    for row in hrows:
        print('%9d %12.0f %8.2f%% %11d %13d %7.0fx'
              % (row['n'], row['estimate'], row['error_pct'], row['hll_bytes'],
                 row['exact_bytes'], row['ratio']))
    print('the memory column does not move - that is the entire point')

    section('8. Merging is exact - this is why HLL is the one that ships')
    mg = merge_demo()
    print('two shards, merged register-wise with max:')
    print('  true union    %9d' % mg['truth'])
    print('  merged shards %9.0f' % mg['merged'])
    print('  single pass   %9.0f' % mg['single_pass'])
    print('  register arrays identical: %s' % mg['identical'])
    print('  each shard ships %d bytes, whatever it counted' % mg['shard_bytes'])

    section('9. ...but intersection is not')
    i = hll_intersection_report()
    print('|A| = %d, |B| = %d, true intersection = %d'
          % (i['n'], i['n'], i['true_intersection']))
    print('  HLL says |A| = %.0f, |B| = %.0f, |A u B| = %.0f  (union error %+.2f%%)'
          % (i['est_a'], i['est_b'], i['est_union'], i['union_error_pct']))
    print('  |A| + |B| - |A u B| = %.0f   (true %d, error %+.0f%%)'
          % (i['est_intersection'], i['true_intersection'],
             i['intersection_error_pct']))
    print('inclusion-exclusion subtracts big numbers to get a small one, so three')
    print('*relative* errors become one absolute error the size of the answer')

    section('10. Count-Min sketch: every row over-counts, so take the minimum')
    c = cms_report()
    print('%d distinct keys, %d events, %d rows x %d counters = %d bytes'
          % (c['distinct'], c['length'], c['d'], c['w'], c['bytes']))
    print('an exact Counter would be roughly %d bytes' % c['exact_bytes'])
    print('  guaranteed bound  e/w * N = %.0f' % c['bound'])
    print('  worst error %d, mean error %.1f, under-counts %d'
          % (c['max_error'], c['mean_error'], c['under_counts']))
    print('  conservative update drops the mean error to %.1f' % c['cons_mean_error'])
    print()
    print('%10s %9s %9s %14s' % ('key', 'true', 'CMS', 'conservative'))
    for k, t, q, cq in c['heavy']:
        print('%10s %9d %9d %14d' % (k, t, q, cq))
    print('   ...the tail...')
    for k, t, q, cq in c['tail']:
        print('%10s %9d %9d %14d' % (k, t, q, cq))
    print('relative error on the top 10: %+.2f%%; on the bottom 10: %+.0f%%'
          % (c['heavy_rel'], c['tail_rel']))
    print('the error is *absolute*, so it is rounding noise on a heavy hitter and')
    print('the entire answer on a rare key - which is exactly the shape you want')
    print('when the question is "which keys are big?"')

    section('11. LeetCode 347 - top k frequent elements')
    print('exact: Counter + a k-sized heap, O(n) time, O(distinct) space')
    print('  nums = [1,1,1,2,2,3], k = 2 ->', top_k_frequent([1, 1, 1, 2, 2, 3], 2))
    print('  nums = [1], k = 1 ->', top_k_frequent([1], 1))
    lc = lc347_report()
    print()
    print('the same question on a 200k-event Zipf stream, k = %d:' % lc['k'])
    print('  exact  %s' % lc['exact'])
    print('  sketch %s' % lc['approx'])
    print('  same set: %s    same order: %s' % (lc['same_set'], lc['same_order']))
    print('  true counts %s' % lc['exact_counts'])
    print('  cms counts  %s' % lc['approx_counts'])
    print('  sketch %d bytes vs Counter %d bytes'
          % (lc['cms_bytes'], lc['counter_bytes']))
    print()
    print('a nearly flat stream (top = %d, kth = %d, gap = %d):'
          % (lc['flat_top'], lc['flat_kth'], lc['flat_gap']))
    print('  same set: %s   missed: %s' % (lc['flat_same_set'], lc['flat_missed']))
    print('approximate top-k only means something when the head stands clear of')
    print('the noise floor.  In the interview, the answer is the exact heap')

    section('12. Picking one')
    print('%-13s %-20s %-22s %-20s %12s'
          % ('structure', 'question', 'error direction', 'mergeable', 'bytes @1e6'))
    for name, q, err, mrg, b in comparison_table():
        print('%-13s %-20s %-22s %-20s %12d' % (name, q, err, mrg, b))
    print()
    for name, where, why in where_they_ship():
        print('%s' % name)
        print('    %s' % where)
        print('    -> %s' % why)
    print()
    print('The size of the error is the easy part - all three come with a knob.')
    print('The part that decides whether you can use one is which *direction* it')
    print('leans, and what the code after the answer does with a wrong one.')

    section('asserts')
    assert r['false_negatives'] == 0, 'a Bloom filter cannot have false negatives'
    assert r['measured_fpr'] < 3 * r['design_fpr']
    assert abs(r['estimated_items'] - r['n']) / r['n'] < 0.05
    assert len(d['false_negatives']) > 0, 'naive deletion must break membership'
    assert len(d['counting_false_negatives']) == 0
    assert not d['counting_still_finds_deleted']
    assert g['lost'] > 0, 'Bloom dedup must lose distinct documents'
    assert g['kept_bloom'] < g['kept_exact']
    assert all(abs(row['error_pct']) < 2.5 for row in hrows)
    assert mg['identical'] and mg['merged'] == mg['single_pass']
    assert abs(mg['merged'] - mg['truth']) / mg['truth'] < 0.03
    assert abs(i['union_error_pct']) < 2.5
    assert abs(i['intersection_error_pct']) > 10, 'HLL intersections are unusable'
    assert c['under_counts'] == 0, 'Count-Min must never under-count'
    assert c['max_error'] <= c['bound']
    assert c['cons_mean_error'] <= c['mean_error']
    assert top_k_frequent([1, 1, 1, 2, 2, 3], 2) == [1, 2]
    assert lc['same_set'], 'a clear Zipf head should survive the sketch'
    print('all assertions passed')


if __name__ == '__main__':
    main()
