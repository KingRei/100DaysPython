"""Day 31 - Skip lists: a sorted structure that replaces balancing with coin flips.

Run me:  python3 skip_list.py

Sections
    1. Why a sorted linked list cannot be binary searched
    2. Express lanes: the deterministic version, and why it cannot survive inserts
    3. The skip list: search / insert / delete, and the update[] array
    4. Spans: turning "how many nodes did I jump over" into rank (Redis ZRANK)
    5. The silent bug: forgetting the spans of the levels you did not touch
    6. Choosing p - Redis picks 0.25, and pays for it in traversal steps
    7. LeetCode 1206 - Design Skiplist
    8. LeetCode 315 revisited - the same answer from spans instead of a Fenwick tree
    9. The same idea in metric space: a toy HNSW
"""

import math
import random
from bisect import bisect_left, insort


# ====================================================================
# 1. A sorted linked list: correct, ordered, and unsearchable
# ====================================================================

class SortedLinkedList:
    """Sorted singly linked list. Insert is O(n) *because* the search is."""

    class Node:
        __slots__ = ('key', 'next')

        def __init__(self, key, nxt=None):
            self.key, self.next = key, nxt

    def __init__(self, keys=()):
        self.head = None
        self.steps = 0
        for k in sorted(keys, reverse=True):
            self.head = SortedLinkedList.Node(k, self.head)

    def search(self, key):
        """Walk until we meet or pass the key. No random access, no bisection."""
        node, steps = self.head, 0
        while node is not None and node.key < key:
            node = node.next
            steps += 1
        self.steps = steps
        return node is not None and node.key == key

    def keys(self):
        out, node = [], self.head
        while node is not None:
            out.append(node.key)
            node = node.next
        return out


def binary_search_needs_random_access(sorted_keys, key):
    """The same search on an array, counted in probes.

    A linked list can do everything this does except the one thing that matters:
    jump to the middle. Hence the express lanes below.
    """
    lo, hi, probes = 0, len(sorted_keys), 0
    while lo < hi:
        mid = (lo + hi) // 2
        probes += 1
        if sorted_keys[mid] < key:
            lo = mid + 1
        else:
            hi = mid
    return (lo < len(sorted_keys) and sorted_keys[lo] == key), probes


# ====================================================================
# 2. Express lanes - the deterministic skip list nobody can maintain
# ====================================================================

def perfect_lanes(keys):
    """Every 2nd node on level 1, every 4th on level 2, ... - a static index.

    Returns a list of levels, level 0 being the full sorted list. Searching this
    is binary search wearing pointers. Inserting one key at the front shifts the
    parity of every node after it, so the whole index has to be rebuilt: O(n).
    """
    levels, cur = [list(keys)], list(keys)
    while len(cur) > 1:
        cur = cur[::2]
        levels.append(cur)
    return levels


def rebuild_cost(keys, new_key):
    """How many nodes change level when one key is inserted into perfect lanes."""
    before = perfect_lanes(keys)
    after = perfect_lanes(sorted(list(keys) + [new_key]))

    def level_of(levels, k):
        return sum(1 for lv in levels if k in lv)

    changed = sum(1 for k in keys if level_of(before, k) != level_of(after, k))
    return changed, len(keys)


# ====================================================================
# 3. The skip list
# ====================================================================

class SkipNode:
    __slots__ = ('key', 'val', 'forward', 'span')

    def __init__(self, key, val, level):
        self.key, self.val = key, val
        self.forward = [None] * level      # forward[i] = next node on level i
        self.span = [0] * level            # how many level-0 nodes that hop covers


class SkipList:
    """A sorted map. Search, insert and delete are all O(log n) expected.

    No rotations, no colours, no height bookkeeping: the shape is decided by a
    coin, once, when a node is born, and never touched again.
    """

    MAX_LEVEL = 32                         # Redis: ZSKIPLIST_MAXLEVEL
    P = 0.25                               # Redis: ZSKIPLIST_P

    def __init__(self, p=None, max_level=None, seed=0):
        self.p = self.P if p is None else p
        self.max_level = self.MAX_LEVEL if max_level is None else max_level
        self.rng = random.Random(seed)
        self.head = SkipNode(None, None, self.max_level)
        self.level = 1                     # highest level currently in use
        self.length = 0
        self.steps = 0                     # forward hops taken by the last search
        self.coin_flips = 0

    # ---- the coin ---------------------------------------------------
    def random_level(self):
        """Redis' zslRandomLevel: keep flipping while the coin says 'go higher'.

        P(level >= k) = p**(k-1), so the expected number of forward pointers per
        node is 1/(1-p) - 2.00 at p=0.5, 1.33 at p=0.25.
        """
        lvl = 1
        while self.rng.random() < self.p and lvl < self.max_level:
            lvl += 1
            self.coin_flips += 1
        return lvl

    # ---- the one traversal every operation shares -------------------
    def _descend(self, key):
        """Walk down the staircase, recording the last node visited per level.

        update[i] is the node on level i whose forward pointer we may have to
        rewrite, and rank[i] is its position, which is what makes spans cheap.
        Insert and delete both need exactly this, which is why the search is
        written once and returns it.
        """
        update = [None] * self.max_level
        rank = [0] * self.max_level
        node, steps = self.head, 0
        for i in range(self.level - 1, -1, -1):
            rank[i] = 0 if i == self.level - 1 else rank[i + 1]
            while node.forward[i] is not None and node.forward[i].key < key:
                rank[i] += node.span[i]
                node = node.forward[i]
                steps += 1
            update[i] = node
        self.steps = steps
        return update, rank, node.forward[0]

    def search(self, key):
        _, _, cand = self._descend(key)
        return cand.val if cand is not None and cand.key == key else None

    def insert(self, key, val):
        update, rank, cand = self._descend(key)
        if cand is not None and cand.key == key:
            cand.val = val
            return
        lvl = self.random_level()
        if lvl > self.level:               # the new node is taller than the list
            for i in range(self.level, lvl):
                update[i] = self.head
                rank[i] = 0
                self.head.span[i] = self.length
            self.level = lvl
        node = SkipNode(key, val, lvl)
        for i in range(lvl):
            node.forward[i] = update[i].forward[i]
            update[i].forward[i] = node
            # split update[i]'s old span at the insertion point
            node.span[i] = update[i].span[i] - (rank[0] - rank[i])
            update[i].span[i] = (rank[0] - rank[i]) + 1
        for i in range(lvl, self.level):
            # levels the new node does NOT reach still jumped over it
            update[i].span[i] += 1
        self.length += 1

    def delete(self, key):
        update, _, cand = self._descend(key)
        if cand is None or cand.key != key:
            return False
        for i in range(self.level):
            if update[i].forward[i] is cand:
                update[i].span[i] += cand.span[i] - 1
                update[i].forward[i] = cand.forward[i]
            else:
                update[i].span[i] -= 1
        while self.level > 1 and self.head.forward[self.level - 1] is None:
            self.level -= 1
        self.length -= 1
        return True

    # ---- ordered-collection operations, free from level 0 -----------
    def keys(self):
        out, node = [], self.head.forward[0]
        while node is not None:
            out.append(node.key)
            node = node.forward[0]
        return out

    def range(self, lo, hi):
        """All keys in [lo, hi) - find the start in O(log n), then walk."""
        _, _, node = self._descend(lo)
        out = []
        while node is not None and node.key < hi:
            out.append(node.key)
            node = node.forward[0]
        return out

    # ---- 4. spans -> rank -------------------------------------------
    def rank(self, key):
        """0-based rank, or -1 if absent. Redis' ZRANK, built out of spans."""
        node, r = self.head, 0
        for i in range(self.level - 1, -1, -1):
            while node.forward[i] is not None and node.forward[i].key <= key:
                r += node.span[i]
                node = node.forward[i]
        return r - 1 if node is not self.head and node.key == key else -1

    def select(self, idx):
        """The key at rank idx, in O(log n). Redis' ZRANGE by index."""
        node, traversed = self.head, 0
        for i in range(self.level - 1, -1, -1):
            while node.forward[i] is not None and traversed + node.span[i] <= idx + 1:
                traversed += node.span[i]
                node = node.forward[i]
            if traversed == idx + 1 and node is not self.head:
                return node.key
        return None

    def count_less(self, key):
        """How many stored keys are < key. Spans again, no counting loop."""
        node, r = self.head, 0
        for i in range(self.level - 1, -1, -1):
            while node.forward[i] is not None and node.forward[i].key < key:
                r += node.span[i]
                node = node.forward[i]
        return r

    # ---- introspection used by the figures and the demo -------------
    def levels(self):
        """[[keys on level 0], [keys on level 1], ...]"""
        out = []
        for i in range(self.level):
            row, node = [], self.head.forward[i]
            while node is not None:
                row.append(node.key)
                node = node.forward[i]
            out.append(row)
        return out

    def spans(self):
        out = []
        for i in range(self.level):
            row, node = [], self.head
            while node is not None:
                row.append((node.key, node.span[i]))
                node = node.forward[i]
            out.append(row)
        return out

    def search_path(self, key):
        """[(level, key_of_node_we_stood_on), ...] - the staircase, for drawing."""
        path, node = [], self.head
        for i in range(self.level - 1, -1, -1):
            path.append((i, node.key))     # we arrive at this level standing here
            while node.forward[i] is not None and node.forward[i].key < key:
                node = node.forward[i]
                path.append((i, node.key))
        return path

    def pointer_count(self):
        total, node = 0, self.head.forward[0]
        while node is not None:
            total += len(node.forward)
            node = node.forward[0]
        return total


# ====================================================================
# 5. The silent bug: the spans of the levels the new node does not reach
# ====================================================================

class SkipListBadSpan(SkipList):
    """Identical to SkipList except for four missing lines in insert().

    A node born with level 2 is still *jumped over* by every level-3 and level-4
    pointer that straddles it, so those spans must grow by one. Forget that and
    the list is still perfectly sorted: keys(), search(), range() - every test
    an ordered-set test would write - keep passing. Only rank() drifts, and only
    once a tall pointer happens to straddle a short node. In Redis terms:
    ZRANGEBYSCORE stays right while ZRANK silently lies.
    """

    def insert(self, key, val):
        update, rank, cand = self._descend(key)
        if cand is not None and cand.key == key:
            cand.val = val
            return
        lvl = self.random_level()
        if lvl > self.level:
            for i in range(self.level, lvl):
                update[i] = self.head
                rank[i] = 0
                self.head.span[i] = self.length
            self.level = lvl
        node = SkipNode(key, val, lvl)
        for i in range(lvl):
            node.forward[i] = update[i].forward[i]
            update[i].forward[i] = node
            node.span[i] = update[i].span[i] - (rank[0] - rank[i])
            update[i].span[i] = (rank[0] - rank[i]) + 1
        # MISSING: for i in range(lvl, self.level): update[i].span[i] += 1
        self.length += 1


def span_bug_report(n=200, seed=7):
    """Run both versions on the same keys and the same coin, and compare."""
    good, bad = SkipList(seed=seed), SkipListBadSpan(seed=seed)
    keys = list(range(n))
    random.Random(seed).shuffle(keys)
    for k in keys:
        good.insert(k, k)
        bad.insert(k, k)
    same_order = good.keys() == bad.keys() == sorted(keys)
    same_search = all((good.search(k) is not None) == (bad.search(k) is not None)
                      for k in range(n + 5))
    wrong_ranks = [k for k in range(n) if bad.rank(k) != k]
    first = wrong_ranks[0] if wrong_ranks else None
    return {
        'order_identical': same_order,
        'search_identical': same_search,
        'wrong_rank_count': len(wrong_ranks),
        'first_wrong_key': first,
        'expected_rank': first,
        'buggy_rank': bad.rank(first) if first is not None else None,
    }


# ====================================================================
# 6. Choosing p
# ====================================================================

def level_distribution(p, n, seed=0, max_level=32):
    sl = SkipList(p=p, max_level=max_level, seed=seed)
    for k in range(n):
        sl.insert(k, k)
    counts = {}
    node = sl.head.forward[0]
    while node is not None:
        counts[len(node.forward)] = counts.get(len(node.forward), 0) + 1
        node = node.forward[0]
    return sl, counts


def p_tradeoff(n=20000, probes=2000, seed=1):
    """Pointers per node vs search steps, for three values of p."""
    rows = []
    for p in (0.5, 0.25, 0.1):
        sl, counts = level_distribution(p, n, seed=seed)
        rnd = random.Random(seed)
        total = 0
        for _ in range(probes):
            sl.search(rnd.randrange(n))
            total += sl.steps
        rows.append({
            'p': p,
            'ptr_per_node': sl.pointer_count() / n,
            'theory_ptr': 1 / (1 - p),
            'top_level': sl.level,
            'avg_steps': total / probes,
            'levels_touched': counts,
        })
    return rows


# ====================================================================
# 7. LeetCode 1206 - Design Skiplist
# ====================================================================

class Skiplist:
    """add / erase / search on a multiset of ints, 0 <= num <= 20000.

    The interview twist: duplicates are allowed, so erase() removes *one*
    occurrence and add() must not dedupe. A dict-backed skip list would have to
    carry counts; here the nodes themselves are the multiset, which is why the
    key comparison below is strictly `<` and never `<=`.
    """

    MAXL, P = 16, 0.5

    class N:
        __slots__ = ('val', 'fw')

        def __init__(self, val, level):
            self.val, self.fw = val, [None] * level

    def __init__(self):
        self.head = Skiplist.N(-1, self.MAXL)
        self.level = 1
        self.rng = random.Random(31)

    def _update(self, target):
        upd, node = [self.head] * self.MAXL, self.head
        for i in range(self.level - 1, -1, -1):
            while node.fw[i] and node.fw[i].val < target:
                node = node.fw[i]
            upd[i] = node
        return upd, node.fw[0]

    def search(self, target: int) -> bool:
        _, cand = self._update(target)
        return cand is not None and cand.val == target

    def add(self, num: int) -> None:
        upd, _ = self._update(num)
        lvl = 1
        while self.rng.random() < self.P and lvl < self.MAXL:
            lvl += 1
        self.level = max(self.level, lvl)
        node = Skiplist.N(num, lvl)
        for i in range(lvl):
            node.fw[i] = upd[i].fw[i]
            upd[i].fw[i] = node

    def erase(self, num: int) -> bool:
        upd, cand = self._update(num)
        if cand is None or cand.val != num:
            return False
        for i in range(self.level):
            if upd[i].fw[i] is cand:
                upd[i].fw[i] = cand.fw[i]
        while self.level > 1 and self.head.fw[self.level - 1] is None:
            self.level -= 1
        return True

    def to_list(self):
        out, node = [], self.head.fw[0]
        while node:
            out.append(node.val)
            node = node.fw[0]
        return out


# ====================================================================
# 8. LeetCode 315 revisited - count of smaller numbers after self
# ====================================================================

def count_smaller_skiplist(nums):
    """Walk right to left, asking the skip list how many keys are already < x.

    Day 29 answered this with a Fenwick tree over compressed values. A skip list
    answers it with count_less(), which is the same prefix sum expressed as
    spans - and unlike the Fenwick tree it needs no coordinate compression and
    supports deletion for free.
    """
    sl, out = SkipList(seed=315), []
    for i, x in enumerate(reversed(nums)):
        # keys are (value, arrival) tuples so duplicates get distinct keys;
        # (x, -1) sorts before every stored copy of x, so count_less counts
        # exactly the strictly-smaller values.
        out.append(sl.count_less((x, -1)))
        sl.insert((x, i), x)
    return out[::-1]


def count_smaller_fenwick(nums):
    """The Day 29 answer, kept here so the two can be asserted equal."""
    order = {v: i for i, v in enumerate(sorted(set(nums)))}
    tree = [0] * (len(order) + 1)

    def upd(i):
        i += 1
        while i <= len(order):
            tree[i] += 1
            i += i & -i

    def qry(i):
        s = 0
        while i > 0:
            s += tree[i]
            i -= i & -i
        return s

    out = []
    for x in reversed(nums):
        out.append(qry(order[x]))
        upd(order[x])
    return out[::-1]


# ====================================================================
# 9. The same trick in metric space: a toy HNSW
# ====================================================================

def _dist(a, b):
    return math.dist(a, b)


class TinyHNSW:
    """Hierarchical Navigable Small World: a skip list where "next" means "near".

    A skip list can only go right because integers live on a line. Vectors do
    not, so the forward pointer becomes a neighbour list: at each layer a point
    keeps its M nearest known neighbours, and the search is the same descent -
    from the top layer, hop to a neighbour closer to the query until none is,
    then drop a layer. The level of a point is drawn the same way as a skip
    list's, just written with the exponential in closed form:

        level = floor(-ln(U) * mL),   mL = 1 / ln(M)

    which is exactly the geometric distribution the coin flips were sampling.
    The one extra ingredient is ef: on the bottom layer we keep the ef best
    candidates instead of a single current node, because in a metric space a
    purely greedy walk can get stuck in a local minimum - something a sorted
    line can never do.
    """

    def __init__(self, M=8, ef=24, seed=0):
        self.M, self.ef = M, ef
        self.mL = 1.0 / math.log(M)
        self.rng = random.Random(seed)
        self.points = []
        self.layers = []                   # layers[l][idx] = [neighbour idx, ...]
        self.entry = None
        self.entry_level = -1
        self.dist_calls = 0

    def random_level(self):
        return int(-math.log(self.rng.random()) * self.mL)

    def _dq(self, q, b):
        self.dist_calls += 1
        return _dist(q, self.points[b])

    def _greedy(self, q, entry, layer):
        """Plain hill climbing - used on the upper, sparse layers."""
        cur, best = entry, self._dq(q, entry)
        moved = True
        while moved:
            moved = False
            for nb in self.layers[layer].get(cur, ()):
                d = self._dq(q, nb)
                if d < best:
                    cur, best, moved = nb, d, True
        return cur

    def _search_layer(self, q, entry, layer, ef):
        """Best-first search keeping the ef closest nodes seen so far."""
        visited = {entry}
        cand = [(self._dq(q, entry), entry)]
        best = list(cand)
        while cand:
            cand.sort()
            d, node = cand.pop(0)
            if d > best[-1][0] and len(best) >= ef:
                break
            for nb in self.layers[layer].get(node, ()):
                if nb in visited:
                    continue
                visited.add(nb)
                dn = self._dq(q, nb)
                if len(best) < ef or dn < best[-1][0]:
                    cand.append((dn, nb))
                    best.append((dn, nb))
                    best.sort()
                    del best[ef:]
        return best

    def add(self, point):
        idx = len(self.points)
        self.points.append(point)
        lvl = self.random_level()
        while len(self.layers) <= lvl:
            self.layers.append({})
        if self.entry is None:
            for l in range(lvl + 1):
                self.layers[l][idx] = []
            self.entry, self.entry_level = idx, lvl
            return
        cur = self.entry
        for l in range(self.entry_level, lvl, -1):
            cur = self._greedy(point, cur, l)
        for l in range(min(lvl, self.entry_level), -1, -1):
            found = self._search_layer(point, cur, l, self.ef)
            cur = found[0][1]
            self.layers[l][idx] = [j for _, j in found[:self.M]]
            for j in self.layers[l][idx]:
                nbrs = self.layers[l][j]
                nbrs.append(idx)
                if len(nbrs) > self.M:
                    nbrs.sort(key=lambda k: self._dq(self.points[j], k))
                    del nbrs[self.M:]
        for l in range(self.entry_level + 1, lvl + 1):
            self.layers[l][idx] = []
        if lvl > self.entry_level:
            self.entry, self.entry_level = idx, lvl

    def search(self, q):
        cur = self.entry
        for l in range(self.entry_level, 0, -1):
            cur = self._greedy(q, cur, l)
        return self._search_layer(q, cur, 0, self.ef)[0][1]


def hnsw_report(n=600, dim=4, queries=100, seed=3):
    rnd = random.Random(seed)
    pts = [tuple(rnd.random() for _ in range(dim)) for _ in range(n)]
    idx = TinyHNSW(M=8, seed=seed)
    for p in pts:
        idx.add(p)
    idx.dist_calls = 0
    hits, brute = 0, 0
    for _ in range(queries):
        q = tuple(rnd.random() for _ in range(dim))
        got = idx.search(q)
        truth = min(range(n), key=lambda i: _dist(q, pts[i]))
        brute += n
        hits += (got == truth)
    return {
        'n': n, 'queries': queries,
        'recall@1': hits / queries,
        'hnsw_dist_calls': idx.dist_calls,
        'brute_dist_calls': brute,
        'speedup': brute / idx.dist_calls,
        'layers': len(idx.layers),
    }


# ====================================================================
# main
# ====================================================================

def section(t):
    print('\n' + '=' * 66)
    print(t)
    print('=' * 66)


def main():
    section('1. A sorted linked list cannot be binary searched')
    keys = list(range(0, 200, 2))
    ll = SortedLinkedList(keys)
    ll.search(198)
    print(f'linked list, searching 198 among {len(keys)} keys: {ll.steps} hops')
    found, probes = binary_search_needs_random_access(keys, 198)
    print(f'array,       searching 198 among {len(keys)} keys: {probes} probes '
          f'(found={found})')
    print('The list has the order but not the random access, so it cannot bisect.')

    section('2. Perfect express lanes: great to search, impossible to maintain')
    small = list(range(1, 17))
    for i, lv in enumerate(perfect_lanes(small)):
        print(f'  level {i}: {lv}')
    changed, total = rebuild_cost(small, 0)
    print(f'insert one key at the front -> {changed}/{total} nodes change level')
    print('That is why real skip lists roll the level instead of computing it.')

    section('3. A random skip list over the same 16 keys')
    sl = SkipList(p=0.5, seed=5)
    for k in small:
        sl.insert(k, k * k)
    for i, row in reversed(list(enumerate(sl.levels()))):
        print(f'  level {i}: {row}')
    print(f'search(13) -> {sl.search(13)} in {sl.steps} hops')
    print(f'search(99) -> {sl.search(99)} in {sl.steps} hops')
    print('path:', sl.search_path(13))
    print('range(5, 11) ->', sl.range(5, 11))
    sl.delete(8)
    print('after delete(8):', sl.keys())

    section('4. Spans turn the same traversal into ZRANK')
    for i, row in reversed(list(enumerate(sl.spans()))):
        print(f'  level {i}: ' + '  '.join(f'{k if k is not None else "H"}:{s}'
                                           for k, s in row))
    for k in (1, 7, 16, 8):
        print(f'  rank({k:2d}) = {sl.rank(k):2d}   select({sl.rank(k)}) = '
              f'{sl.select(sl.rank(k))}')
    print('count_less(10) =', sl.count_less(10), ' (keys strictly below 10)')

    section('5. The silent span bug')
    rep = span_bug_report()
    print(f'order identical to a correct list : {rep["order_identical"]}')
    print(f'search identical to a correct list: {rep["search_identical"]}')
    print(f'ranks wrong                       : {rep["wrong_rank_count"]}/200')
    print(f'first disagreement                : key {rep["first_wrong_key"]} '
          f'-> expected rank {rep["expected_rank"]}, buggy list says '
          f'{rep["buggy_rank"]}')
    print('Every ordered-set test passes. ZRANK is the only witness - and once')
    print('one span is short, every rank past it inherits the error.')

    section('6. Choosing p (n = 20000, 2000 random searches)')
    print(f'{"p":>6} {"ptr/node":>9} {"theory":>7} {"top lvl":>8} {"avg steps":>10}')
    for r in p_tradeoff():
        print(f'{r["p"]:>6} {r["ptr_per_node"]:>9.2f} {r["theory_ptr"]:>7.2f} '
              f'{r["top_level"]:>8} {r["avg_steps"]:>10.1f}')
    print('Redis picks p = 0.25: a third of the pointers of a coin-flip list,')
    print('bought with a longer walk on each level.')

    section('7. LeetCode 1206 - Design Skiplist')
    sk = Skiplist()
    for v in (1, 2, 3, 3, 3):
        sk.add(v)
    print('after add 1,2,3,3,3 ->', sk.to_list())
    print('search(0) =', sk.search(0), ' search(3) =', sk.search(3))
    print('erase(3)  =', sk.erase(3), '->', sk.to_list())
    print('erase(9)  =', sk.erase(9), '->', sk.to_list())
    print('Duplicates are why the comparison is `<` and never `<=`.')

    section('8. LeetCode 315 - the Day 29 answer, rebuilt from spans')
    for nums in ([5, 2, 6, 1], [-1, -1], [2, 0, 1]):
        a, b = count_smaller_skiplist(nums), count_smaller_fenwick(nums)
        print(f'{str(nums):>14} -> skiplist {a}   fenwick {b}')

    section('9. The same layered idea in metric space: a toy HNSW')
    print(f'{"n":>6} {"layers":>7} {"recall@1":>9} {"HNSW dists":>11} '
          f'{"brute":>8} {"speedup":>8}')
    for n in (500, 2000, 8000):
        h = hnsw_report(n=n, queries=100)
        print(f'{h["n"]:>6} {h["layers"]:>7} {h["recall@1"]:>9.2f} '
              f'{h["hnsw_dist_calls"]:>11} {h["brute_dist_calls"]:>8} '
              f'{h["speedup"]:>7.1f}x')
    print('16x the data, 1.1x the work: the distance count barely moves, which')
    print('is the skip list result (O(log n) hops) carried into metric space.')

    section('10. Assertions')
    ref = SkipList(seed=5)
    model = []
    rnd = random.Random(5)
    for _ in range(3000):
        k = rnd.randrange(500)
        if rnd.random() < 0.65:
            ref.insert(k, k)
            if k not in model:
                insort(model, k)
        else:
            hit = ref.delete(k)
            i = bisect_left(model, k)
            inside = i < len(model) and model[i] == k
            assert hit == inside, (k, hit, inside)
            if inside:
                model.pop(i)
        assert ref.length == len(model)
    assert ref.keys() == model
    for k in model:
        assert ref.search(k) == k
        assert ref.rank(k) == model.index(k)
        assert ref.select(model.index(k)) == k
    assert ref.range(100, 140) == [k for k in model if 100 <= k < 140]
    assert count_smaller_skiplist([5, 2, 6, 1]) == [2, 1, 1, 0]
    assert count_smaller_skiplist([-1, -1]) == [0, 0]
    big = [rnd.randrange(50) for _ in range(400)]
    assert count_smaller_skiplist(big) == count_smaller_fenwick(big)
    assert span_bug_report()['wrong_rank_count'] > 0
    s2 = Skiplist()
    for v in (1, 2, 3, 3):
        s2.add(v)
    assert s2.to_list() == [1, 2, 3, 3]
    assert s2.erase(3) and s2.to_list() == [1, 2, 3]
    assert not s2.erase(7)
    assert hnsw_report(n=200, queries=40)['recall@1'] == 1.0
    print('all assertions passed')


if __name__ == '__main__':
    main()
