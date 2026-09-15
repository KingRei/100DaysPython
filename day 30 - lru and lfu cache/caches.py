"""Day 30 - LRU and LFU caches: what to throw away when memory runs out.

A cache is a bounded map plus a policy for choosing a victim.  The map part is
Day 09's hash table; everything interesting is in the policy.

Run me:  python caches.py
"""

import heapq
import random
import time
from collections import OrderedDict, defaultdict


# ====================================================================
# 1. The ceiling: Belady's optimal offline policy
# ====================================================================

def belady_hits(trace, capacity):
    """Evict the item whose NEXT use is furthest away (or never).

    Provably optimal, and impossible online: it needs the future.  We use it
    only as a yardstick - no online policy can beat this number.
    """
    # next_use[i] = the next index > i where the same key appears
    nxt = [len(trace)] * len(trace)
    last_seen = {}
    for i in range(len(trace) - 1, -1, -1):
        nxt[i] = last_seen.get(trace[i], len(trace))
        last_seen[trace[i]] = i

    cache = {}                       # key -> next use index
    hits = 0
    for i, key in enumerate(trace):
        if key in cache:
            hits += 1
        elif len(cache) < capacity:
            pass
        else:
            victim = max(cache, key=lambda k: cache[k])
            del cache[victim]
        cache[key] = nxt[i]
    return hits


# ====================================================================
# 2. LRU, three ways
# ====================================================================

class Node:
    """Doubly linked list node.  __slots__ because there is one per entry."""
    __slots__ = ('key', 'val', 'prev', 'next')

    def __init__(self, key=None, val=None):
        self.key, self.val = key, val
        self.prev = self.next = None


class LRUCache:
    """LeetCode 146.  Hash map for lookup, doubly linked list for order.

    The list is kept most-recent-first between two sentinels, so no branch
    anywhere has to ask "am I at the end?".
    """

    def __init__(self, capacity):
        self.capacity = capacity
        self.map = {}
        self.head = Node()            # sentinel: most recent side
        self.tail = Node()            # sentinel: least recent side
        self.head.next = self.tail
        self.tail.prev = self.head
        self.hits = self.misses = self.evictions = 0

    # --- list surgery, both O(1) ---
    def _unlink(self, node):
        node.prev.next = node.next
        node.next.prev = node.prev

    def _push_front(self, node):
        node.prev = self.head
        node.next = self.head.next
        self.head.next.prev = node
        self.head.next = node

    def get(self, key):
        node = self.map.get(key)
        if node is None:
            self.misses += 1
            return -1
        self.hits += 1
        self._unlink(node)            # a READ is a use: refresh the order
        self._push_front(node)
        return node.val

    def put(self, key, value):
        node = self.map.get(key)
        if node is not None:
            node.val = value
            self._unlink(node)
            self._push_front(node)
            return
        if len(self.map) >= self.capacity:
            victim = self.tail.prev   # the sentinel makes this one line
            self._unlink(victim)
            del self.map[victim.key]
            self.evictions += 1
        node = Node(key, value)
        self.map[key] = node
        self._push_front(node)

    def order(self):
        """Most recent first - for the diagrams and the tests."""
        out, cur = [], self.head.next
        while cur is not self.tail:
            out.append(cur.key)
            cur = cur.next
        return out


class LRUCacheOrderedDict:
    """The same policy in six lines, because OrderedDict IS that linked list."""

    def __init__(self, capacity):
        self.capacity = capacity
        self.d = OrderedDict()
        self.hits = self.misses = self.evictions = 0

    def get(self, key):
        if key not in self.d:
            self.misses += 1
            return -1
        self.hits += 1
        self.d.move_to_end(key)
        return self.d[key]

    def put(self, key, value):
        if key in self.d:
            self.d.move_to_end(key)
        elif len(self.d) >= self.capacity:
            self.d.popitem(last=False)
            self.evictions += 1
        self.d[key] = value

    def order(self):
        return list(reversed(self.d))


class LRUCacheList:
    """The version people write first: a plain list, move-to-front on use.

    Same eviction decisions as LRUCache, but every hit scans the list, so it
    is O(n) per operation instead of O(1).  Correct, and quadratic.
    """

    def __init__(self, capacity):
        self.capacity = capacity
        self.order_ = []              # most recent first
        self.val = {}
        self.hits = self.misses = self.evictions = 0
        self.scanned = 0              # cells touched by list.remove / index

    def get(self, key):
        if key not in self.val:
            self.misses += 1
            return -1
        self.hits += 1
        self.scanned += self.order_.index(key) + 1
        self.order_.remove(key)
        self.order_.insert(0, key)
        return self.val[key]

    def put(self, key, value):
        if key in self.val:
            self.scanned += self.order_.index(key) + 1
            self.order_.remove(key)
        elif len(self.val) >= self.capacity:
            victim = self.order_.pop()
            del self.val[victim]
            self.evictions += 1
        self.order_.insert(0, key)
        self.val[key] = value

    def order(self):
        return list(self.order_)


class FIFOCache:
    """LRU with the refresh on `get` deleted - which is exactly FIFO.

    This is the single most common LRU bug: the eviction order stops tracking
    use and starts tracking arrival.  Nothing crashes; the hit rate just drops.
    """

    def __init__(self, capacity):
        self.capacity = capacity
        self.d = OrderedDict()
        self.hits = self.misses = self.evictions = 0

    def get(self, key):
        if key not in self.d:
            self.misses += 1
            return -1
        self.hits += 1
        return self.d[key]            # <- no move_to_end

    def put(self, key, value):
        if key not in self.d and len(self.d) >= self.capacity:
            self.d.popitem(last=False)
            self.evictions += 1
        self.d[key] = value

    def order(self):
        return list(reversed(self.d))


# ====================================================================
# 3. LFU in O(1)  (LeetCode 460)
# ====================================================================

class LFUCache:
    """Count uses, evict the least used, break ties by LRU.

    The trick that makes it O(1): one bucket per frequency, each bucket an
    ordered dict.  `min_freq` is the only global, and it can only ever move
    up by one (on a hit) or reset to 1 (on an insert), so it never searches.
    """

    def __init__(self, capacity):
        self.capacity = capacity
        self.val = {}                          # key -> value
        self.freq = {}                         # key -> use count
        self.buckets = defaultdict(OrderedDict)  # count -> keys, LRU first
        self.min_freq = 0
        self.hits = self.misses = self.evictions = 0

    def _touch(self, key):
        f = self.freq[key]
        del self.buckets[f][key]
        if not self.buckets[f]:
            del self.buckets[f]
            if self.min_freq == f:
                self.min_freq = f + 1          # the ONLY way min_freq grows
        self.freq[key] = f + 1
        self.buckets[f + 1][key] = None

    def get(self, key):
        if key not in self.val:
            self.misses += 1
            return -1
        self.hits += 1
        self._touch(key)
        return self.val[key]

    def put(self, key, value):
        if self.capacity <= 0:
            return
        if key in self.val:
            self.val[key] = value
            self._touch(key)
            return
        if len(self.val) >= self.capacity:
            victim, _ = self.buckets[self.min_freq].popitem(last=False)
            if not self.buckets[self.min_freq]:
                del self.buckets[self.min_freq]
            del self.val[victim], self.freq[victim]
            self.evictions += 1
        self.val[key] = value
        self.freq[key] = 1
        self.buckets[1][key] = None
        self.min_freq = 1                      # a new key resets the floor

    def snapshot(self):
        """{frequency: [keys, least recent first]} - for the diagrams."""
        return {f: list(keys) for f, keys in sorted(self.buckets.items())}


class LFUNoTieBreak:
    """LFU that keeps only counts and drops the within-bucket order.

    It still evicts a minimum-frequency key, so it is "an LFU" - but which
    one it picks is arbitrary, and LeetCode 460 checks exactly that.
    """

    def __init__(self, capacity):
        self.capacity = capacity
        self.val, self.freq = {}, {}
        self.hits = self.misses = self.evictions = 0

    def get(self, key):
        if key not in self.val:
            self.misses += 1
            return -1
        self.hits += 1
        self.freq[key] += 1
        return self.val[key]

    def put(self, key, value):
        if key in self.val:
            self.val[key] = value
            self.freq[key] += 1
            return
        if len(self.val) >= self.capacity:
            lo = min(self.freq.values())
            # "any key with the minimum count" - here the first inserted one
            victim = next(k for k in self.val if self.freq[k] == lo)
            del self.val[victim], self.freq[victim]
            self.evictions += 1
        self.val[key] = value
        self.freq[key] = 1


# ====================================================================
# 4. One heap, five policies - the sglang shape
# ====================================================================
#
# sglang's radix cache does not implement five caches.  Every tree node
# carries last_access_time, hit_count and creation_time, and a policy object
# turns a node into a sort key:
#
#     class LRUStrategy:  get_priority(node) -> node.last_access_time
#     class LFUStrategy:  get_priority(node) -> (node.hit_count, node.last_access_time)
#     class FIFOStrategy: get_priority(node) -> node.creation_time
#     class MRUStrategy:  get_priority(node) -> -node.last_access_time
#     class SLRUStrategy: get_priority(node) -> (hit_count >= threshold, last_access)
#
# Eviction is then one heap over the candidates, and "which policy" is a
# command-line flag.  We reproduce that here on a flat cache.

class Entry:
    __slots__ = ('key', 'val', 'last_access', 'hit_count', 'created')

    def __init__(self, key, val, clock):
        self.key, self.val = key, val
        self.last_access = self.created = clock
        self.hit_count = 0


def priority_lru(e):
    return e.last_access


def priority_lfu(e):
    return (e.hit_count, e.last_access)


def priority_fifo(e):
    return e.created


def priority_mru(e):
    return -e.last_access


def make_priority_slru(threshold=2):
    """Segmented LRU: one bit of protection on top of LRU.

    Entries that have been hit `threshold` times move to the protected
    segment and are evicted only after every probationary entry is gone.
    That single bit is what stops a scan from wiping the cache.
    """
    def priority(e):
        protected = 1 if e.hit_count >= threshold else 0
        return (protected, e.last_access)
    return priority


POLICIES = {
    'lru': priority_lru,
    'lfu': priority_lfu,
    'fifo': priority_fifo,
    'mru': priority_mru,
    'slru': make_priority_slru(2),
}


class PolicyCache:
    """A cache whose eviction rule is a function from entry to sort key."""

    def __init__(self, capacity, priority):
        self.capacity = capacity
        self.priority = priority
        self.map = {}
        self.clock = 0
        self.hits = self.misses = self.evictions = 0

    def _tick(self):
        self.clock += 1
        return self.clock

    def get(self, key):
        e = self.map.get(key)
        if e is None:
            self.misses += 1
            return -1
        self.hits += 1
        e.hit_count += 1
        e.last_access = self._tick()
        return e.val

    def put(self, key, value):
        e = self.map.get(key)
        if e is not None:
            e.val = value
            e.hit_count += 1
            e.last_access = self._tick()
            return
        if len(self.map) >= self.capacity:
            # the same heapify-the-candidates shape sglang uses
            victim = heapq.nsmallest(
                1, self.map.values(), key=self.priority)[0]
            del self.map[victim.key]
            self.evictions += 1
        self.map[key] = Entry(key, value, self._tick())


# ====================================================================
# 5. Workloads - the policies only differ when the access pattern does
# ====================================================================

def zipf_trace(n_keys=200, length=4000, alpha=1.1, seed=0):
    """Skewed popularity: a few keys get most of the requests."""
    rng = random.Random(seed)
    weights = [1.0 / (i ** alpha) for i in range(1, n_keys + 1)]
    total = sum(weights)
    cum, acc = [], 0.0
    for w in weights:
        acc += w / total
        cum.append(acc)
    out = []
    for _ in range(length):
        x = rng.random()
        lo, hi = 0, n_keys - 1
        while lo < hi:                      # Day 23, still earning its keep
            mid = (lo + hi) // 2
            if cum[mid] < x:
                lo = mid + 1
            else:
                hi = mid
        out.append(lo)
    return out


def scan_trace(n_keys=33, rounds=120):
    """A loop over one more key than fits.  The classic LRU killer."""
    return [i % n_keys for i in range(n_keys * rounds)]


def shifting_trace(n_hot=16, hot_len=300, phases=12, n_keys=400, seed=1):
    """A hot set that moves.  Yesterday's stars are today's dead weight."""
    rng = random.Random(seed)
    out = []
    for p in range(phases):
        hot = [rng.randrange(n_keys) for _ in range(n_hot)]
        for _ in range(hot_len):
            out.append(rng.choice(hot))
    return out


def mixed_trace(n_hot=24, n_cold=400, length=6000, scan_every=40, seed=2):
    """A hot working set, interrupted by a long scan every so often."""
    rng = random.Random(seed)
    out, cold = [], 0
    for i in range(length):
        if i % scan_every == 0:
            for _ in range(12):             # a burst of never-reused keys
                out.append(1000 + cold)
                cold = (cold + 1) % n_cold
        out.append(rng.randrange(n_hot))
    return out


def run_trace(cache, trace):
    """A read-through cache: miss -> fetch -> insert."""
    for key in trace:
        if cache.get(key) == -1:
            cache.put(key, key)
    return cache.hits / len(trace)


def compare_policies(trace, capacity):
    """Hit rate of every policy, plus the offline optimum."""
    rows = {}
    for name, prio in POLICIES.items():
        rows[name] = run_trace(PolicyCache(capacity, prio), trace)
    rows['opt'] = belady_hits(trace, capacity) / len(trace)
    return rows


WORKLOADS = {
    'zipf (skewed popularity)': (zipf_trace(), 32),
    'scan (loop of 33 keys)': (scan_trace(), 32),
    'shifting hot set': (shifting_trace(), 32),
    'hot set + cold scans': (mixed_trace(), 32),
}


# ====================================================================
# 6. LeetCode 432 - All O(1), the LFU bucket trick on its own
# ====================================================================

class AllOne:
    """inc / dec / getMaxKey / getMinKey, all O(1).

    Same insight as LFU: keys with equal counts live together in a bucket,
    and the buckets themselves form a sorted doubly linked list, so a count
    only ever moves a key to the neighbouring bucket.
    """

    class Bucket:
        __slots__ = ('count', 'keys', 'prev', 'next')

        def __init__(self, count=0):
            self.count = count
            self.keys = set()
            self.prev = self.next = None

    def __init__(self):
        self.head = self.Bucket()              # sentinels, counts -inf / +inf
        self.tail = self.Bucket()
        self.head.next, self.tail.prev = self.tail, self.head
        self.where = {}                        # key -> bucket

    def _insert_after(self, node, count):
        b = self.Bucket(count)
        b.prev, b.next = node, node.next
        node.next.prev = b
        node.next = b
        return b

    def _drop(self, b):
        b.prev.next = b.next
        b.next.prev = b.prev

    def inc(self, key):
        if key not in self.where:
            first = self.head.next
            if first is self.tail or first.count != 1:
                first = self._insert_after(self.head, 1)
            first.keys.add(key)
            self.where[key] = first
            return
        cur = self.where[key]
        nxt = cur.next
        if nxt is self.tail or nxt.count != cur.count + 1:
            nxt = self._insert_after(cur, cur.count + 1)
        nxt.keys.add(key)
        self.where[key] = nxt
        cur.keys.discard(key)
        if not cur.keys:
            self._drop(cur)

    def dec(self, key):
        if key not in self.where:
            return
        cur = self.where[key]
        if cur.count == 1:
            del self.where[key]
        else:
            prv = cur.prev
            if prv is self.head or prv.count != cur.count - 1:
                prv = self._insert_after(cur.prev, cur.count - 1)
            prv.keys.add(key)
            self.where[key] = prv
        cur.keys.discard(key)
        if not cur.keys:
            self._drop(cur)

    def getMaxKey(self):
        return '' if self.tail.prev is self.head else next(iter(self.tail.prev.keys))

    def getMinKey(self):
        return '' if self.head.next is self.tail else next(iter(self.head.next.keys))

    def buckets(self):
        out, cur = [], self.head.next
        while cur is not self.tail:
            out.append((cur.count, sorted(cur.keys)))
            cur = cur.next
        return out


# ====================================================================
# 7. Timing - the linked list is not a flourish
# ====================================================================

def timing(capacity=2000, ops=40000, seed=7):
    """Same decisions, two data structures, very different clocks."""
    rng = random.Random(seed)
    trace = [rng.randrange(capacity * 2) for _ in range(ops)]

    out = {}
    for name, cache in (('linked list (O(1))', LRUCache(capacity)),
                        ('python list (O(n))', LRUCacheList(capacity))):
        t0 = time.perf_counter()
        run_trace(cache, trace)
        out[name] = time.perf_counter() - t0
    scan_probe = LRUCacheList(capacity)
    run_trace(scan_probe, trace)
    out['cells scanned by the O(n) version'] = scan_probe.scanned
    return out


# ====================================================================
# 8. LeetCode
# ====================================================================

def lc146_demo():
    """LeetCode 146 - LRU Cache.  The judge's own example."""
    c = LRUCache(2)
    log = []
    c.put(1, 1)
    c.put(2, 2)
    log.append(('get(1)', c.get(1)))       # 1, and 1 becomes most recent
    c.put(3, 3)                            # evicts 2, not 1
    log.append(('get(2)', c.get(2)))       # -1
    c.put(4, 4)                            # evicts 1
    log.append(('get(1)', c.get(1)))       # -1
    log.append(('get(3)', c.get(3)))       # 3
    log.append(('get(4)', c.get(4)))       # 4
    return log


def lc146_wrong_demo():
    """The same script through FIFOCache - one missing line, wrong answers."""
    c = FIFOCache(2)
    log = []
    c.put(1, 1)
    c.put(2, 2)
    log.append(('get(1)', c.get(1)))       # 1 - the read looks fine
    c.put(3, 3)                            # but it evicted 1, not 2
    log.append(('get(2)', c.get(2)))       # 2  <- should be -1
    log.append(('get(1)', c.get(1)))       # -1 <- should be 1
    return log


def lc460_demo():
    """LeetCode 460 - LFU Cache.  The tie-break is the whole test."""
    c = LFUCache(2)
    log = []
    c.put(1, 1)
    c.put(2, 2)
    log.append(('get(1)', c.get(1)))       # 1 -> freq(1)=2, freq(2)=1
    c.put(3, 3)                            # evicts 2 (lowest count)
    log.append(('get(2)', c.get(2)))       # -1
    log.append(('get(3)', c.get(3)))       # 3 -> freq(3)=2
    c.put(4, 4)                            # 1 and 3 tie at 2; evict 1 (older)
    log.append(('get(1)', c.get(1)))       # -1
    log.append(('get(3)', c.get(3)))       # 3
    log.append(('get(4)', c.get(4)))       # 4
    return log


def lc460_tie_break_check():
    """A case where "least frequently used" has two right-looking answers.

    put(1) put(2) get(2) get(1) leaves both keys at count 2, but key 2 is
    the one that has not been touched in longest.  LC 460 wants key 2.
    Tie-breaking by insertion order instead picks key 1 - and every other
    test case still passes.
    """
    def script(cache):
        cache.put(1, 1)
        cache.put(2, 2)
        cache.get(2)
        cache.get(1)         # counts now tie at 2; key 2 is the older use
        cache.put(3, 3)
        return cache.get(1), cache.get(2)
    return {'LFUCache (LRU tie-break)': script(LFUCache(2)),
            'LFUNoTieBreak (insertion order)': script(LFUNoTieBreak(2))}


def lc432_demo():
    c = AllOne()
    for word in ('hello', 'hello', 'world', 'leet', 'leet', 'leet'):
        c.inc(word)
    before = (c.getMaxKey(), c.getMinKey(), c.buckets())
    c.dec('world')
    c.dec('leet')
    after = (c.getMaxKey(), c.getMinKey(), c.buckets())
    return before, after


# ====================================================================
# 9. The story, printed
# ====================================================================

def rule(title):
    print('\n' + '=' * 66)
    print(title)
    print('=' * 66)


def pct(x):
    return '%5.1f%%' % (100 * x)


def main():
    rule('1. LeetCode 146 - LRU Cache')
    c = LRUCache(2)
    c.put(1, 1)
    c.put(2, 2)
    print('   capacity 2, after put(1,1) put(2,2):', c.order(), '(most recent first)')
    c.get(1)
    print('   after get(1):                       ', c.order(),
          '<- a READ counts as a use')
    c.put(3, 3)
    print('   after put(3,3):                     ', c.order(),
          '<- 2 was the victim, not 1')
    for call, got in lc146_demo():
        print('   %-8s -> %s' % (call, got))

    rule('2. Forget one line and LRU silently becomes FIFO')
    print('   LRUCache : ', lc146_demo()[:3])
    print('   FIFOCache: ', lc146_wrong_demo())
    print('   Nothing raises.  get(2) returns 2 where the judge wants -1,')
    print('   because without the refresh on `get`, the order tracks arrival')
    print('   instead of use.')

    rule('3. OrderedDict is that linked list')
    a, b = LRUCache(3), LRUCacheOrderedDict(3)
    for k in (1, 2, 3, 1, 4, 2, 5, 1):
        if a.get(k) == -1:
            a.put(k, k)
        if b.get(k) == -1:
            b.put(k, k)
    print('   hand-rolled :', a.order())
    print('   OrderedDict :', b.order())
    assert a.order() == b.order()
    print('   move_to_end() and popitem(last=False) are the two list edits,')
    print('   already written in C.')

    rule('4. ...but the plain list version is quadratic')
    t = timing()
    print('   linked list (O(1)) : %.1f ms' % (1000 * t['linked list (O(1))']))
    print('   python list (O(n)) : %.1f ms' % (1000 * t['python list (O(n))']))
    print('   ratio              : %.0fx' % (t['python list (O(n))'] /
                                             t['linked list (O(1))']))
    print('   list cells walked  : %s' % f"{t['cells scanned by the O(n) version']:,}")

    rule('5. LeetCode 460 - LFU Cache, and why the tie-break is the test')
    for call, got in lc460_demo():
        print('   %-8s -> %s' % (call, got))
    f = LFUCache(3)
    for k in (1, 1, 1, 2, 2, 3):
        if f.get(k) == -1:
            f.put(k, k)
    print('   frequency buckets  :', f.snapshot(), ' min_freq =', f.min_freq)
    print('   tie-break check    :')
    for name, result in lc460_tie_break_check().items():
        print('     %-32s get(1), get(2) = %s' % (name, result))
    print('   Both are "least frequently used".  Only the first one is LC 460.')

    rule('6. One heap, five policies (the sglang shape)')
    names = ['opt'] + sorted(POLICIES)
    print('   %-26s %s' % ('workload', ' '.join('%6s' % n for n in names)))
    for label, (trace, cap) in WORKLOADS.items():
        rows = compare_policies(trace, cap)
        print('   %-26s %s' % (label, ' '.join(pct(rows[n]) for n in names)))
    print()
    print('   No policy wins every row.  That is why sglang ships a flag')
    print('   (--radix-eviction-policy) instead of a winner: every strategy')
    print('   is one get_priority(node) function feeding the same heap.')

    rule('7. Scan pollution - the failure mode LRU cannot see coming')
    cap = 32
    loop = scan_trace(n_keys=cap + 1, rounds=120)
    for name in ('lru', 'lfu', 'slru', 'mru'):
        r = run_trace(PolicyCache(cap, POLICIES[name]), loop)
        print('   %-5s on a %d-key loop through a %d-slot cache: %s'
              % (name, cap + 1, cap, pct(r)))
    print('   LRU evicts exactly the key it is about to ask for, every time,')
    print('   and LFU and SLRU do no better: on a pure scan nothing is ever')
    print('   hit twice, so no counter rises and no entry earns protection.')
    print('   Only MRU - throw away what you just used - survives a loop.')
    mix, cap2 = WORKLOADS['hot set + cold scans']
    print('   Where SLRU pays off is the realistic mix of a hot set plus')
    print('   cold scans: lru %s vs slru %s on that workload.'
          % (pct(run_trace(PolicyCache(cap2, POLICIES['lru']), mix)),
             pct(run_trace(PolicyCache(cap2, POLICIES['slru']), mix))))
    print('   The bug is not in the code; it is in the policy.')

    rule('8. LFU has the opposite problem: it never forgets')
    trace = shifting_trace()
    rows = compare_policies(trace, 32)
    print('   hot set moves every 300 requests, capacity 32')
    for n in ('opt', 'lru', 'lfu', 'slru'):
        print('   %-5s %s' % (n, pct(rows[n])))
    print('   Yesterday\'s hot keys keep their counts and squat in the cache.')
    print('   SLRU inherits the same stickiness: a protection bit that never')
    print('   expires is a small LFU.  Real LFU implementations age the')
    print('   counters; sglang softens it by pairing hit_count with')
    print('   last_access_time in the sort key, so a stale favourite still')
    print('   loses to a fresh one at equal count.')

    rule('9. LeetCode 432 - All O(1), the same bucket trick alone')
    before, after = lc432_demo()
    print('   after inc hello x2, world, leet x3:')
    print('     max = %-6s min = %-6s buckets = %s' % before)
    print('   after dec world, dec leet:')
    print('     max = %-6s min = %-6s buckets = %s' % after)

    rule('10. Assertions')
    assert [v for _, v in lc146_demo()] == [1, -1, -1, 3, 4]
    assert [v for _, v in lc460_demo()] == [1, -1, 3, -1, 3, 4]
    tb = lc460_tie_break_check()
    assert tb['LFUCache (LRU tie-break)'] == (1, -1)
    assert tb['LFUNoTieBreak (insertion order)'] == (-1, 2)
    assert LRUCache(3).capacity == 3
    scan = scan_trace(n_keys=9, rounds=30)
    assert run_trace(PolicyCache(8, POLICIES['lru']), scan) == 0.0
    assert run_trace(PolicyCache(8, POLICIES['mru']), scan) > 0.5
    zipf = zipf_trace(length=1500)
    opt = belady_hits(zipf, 16) / len(zipf)
    for name in POLICIES:
        assert run_trace(PolicyCache(16, POLICIES[name]), zipf) <= opt + 1e-12
    o = AllOne()
    for w in ('a', 'a', 'b'):
        o.inc(w)
    assert (o.getMaxKey(), o.getMinKey()) == ('a', 'b')
    o.dec('a')
    assert o.buckets() == [(1, ['a', 'b'])]
    print('   all assertions passed')


if __name__ == '__main__':
    main()
