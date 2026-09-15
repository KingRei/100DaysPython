# LRU and LFU caches: choosing what to throw away

More details in:
https://medium.com/100-days-of-python

Interactive walkthrough: [demo.html](demo.html) - a single self-contained page
(no build step, works offline, 中文 / English toggle) that splices nodes through a doubly linked
list on LeetCode 146, deletes the one refresh line that silently turns that LRU into a FIFO,
walks LFU's frequency buckets and the two-case rule that keeps `min_freq` from ever searching,
runs a loop of six keys through five slots until LRU's hit rate settles at exactly zero, swaps
sglang's five eviction policies through one shared heap, and finishes with LeetCode 432.

![img](https://github.com/KingRei/100DaysPython/blob/master/day%2030%20-%20lru%20and%20lfu%20cache/imgs/day30_1.png?raw=true)

A cache is a bounded map. Two of its three operations are the hash table from day 09 and nothing
more; the third one is the interesting one, and it only runs when the cache is full: *which
entry do we throw away?* Every answer to that question is a bet about the future, and the bet is
what separates a cache that earns its memory from one that merely occupies it.

The offline optimum is known and useless: Bélády's MIN evicts the entry whose next use is
furthest away, which requires knowing the whole request sequence in advance. It is worth
implementing anyway, because it turns "is our policy good?" from an opinion into a number - on
the skewed workload below, the best online policy gets 72% where MIN gets 79%, and no amount of
cleverness will close the rest.

The two workhorses guess from the past instead. LRU bets that what was used recently will be
used again, and tracks a single fact per entry: when it was last touched. LFU bets that what has
been used often will be used often, and tracks a count. Both reach `O(1)` per operation, and
both have a workload that reduces them to zero.

## LRU: a hash map welded to a doubly linked list

```python
def get(self, key):
    node = self.map.get(key)
    if node is None:
        return -1
    self._unlink(node)              # a read IS a use
    self._push_front(node)
    return node.val
```

The map answers "where is this key" in `O(1)`; the list answers "who is least recent" in `O(1)`
by keeping the victim at `tail.prev`. Neither structure can do the other's job, which is why
both are there. Two sentinel nodes at the ends remove every null check from `_unlink`, and
`OrderedDict` *is* this structure - `move_to_end` and `popitem(last=False)` are the same two
pointer dances with the loop already written in C.

Delete those two refresh lines inside `get` and nothing breaks. No exception, no warning, every
test that only checks values still passes. But the list order now records *arrival* rather than
*use*, so the cache has silently become a FIFO, and on the LeetCode 146 script it starts
evicting different keys. Running the same calls through both is the whole point of the first
demo tab.

The list version is worth writing once, badly, to feel the difference: searching a Python list
for the key and moving it to the front is correct and quadratic. Over 40,000 operations on a
2,000-slot cache it scans 19 million cells and runs about 9.5x slower than the linked list,
which is the entire justification for carrying two data structures instead of one.

## LFU: one bucket per count, and a minimum that never searches

```python
self.buckets = defaultdict(OrderedDict)     # count -> keys, least recent first
self.min_freq = 0
```

Evicting the least frequently used entry sounds like it needs a heap or a scan. It needs
neither, because of a small observation: a count only ever increases by one. Keep an
`OrderedDict` per count and `min_freq` changes in exactly two situations - it becomes `f + 1`
when a hit empties the bucket that *was* the minimum, and it resets to `1` when a new key is
inserted, because a newcomer has earned nothing. There is no third case, so `min_freq` is never
recomputed and never searched for.

Inside a bucket the order matters: keys tied on count are broken by recency, oldest use first.
LeetCode 460 tests this, and a version that breaks ties by insertion order passes every
value-level check while returning different keys. The module ships both so the divergence can be
printed rather than argued about.

Strip the values away and what is left - a sorted chain of count buckets, each holding a set of
keys - is LeetCode 432 on its own: `inc`, `dec`, `getMaxKey` and `getMinKey`, all `O(1)`,
because the destination bucket is always the neighbour and the extremes are the two ends of the
chain.

## The two ways these policies fail

Measured over four synthetic workloads, 32 slots, `opt` being Bélády's offline optimum:

| workload | opt | lru | lfu | fifo | mru | slru |
|---|---|---|---|---|---|---|
| zipf (skewed popularity) | 79.0% | 63.2% | 71.6% | 57.6% | 22.2% | 69.5% |
| scan (loop of 33 keys) | 96.1% | 0.0% | 0.0% | 0.0% | 96.1% | 0.0% |
| shifting hot set | 95.7% | 94.9% | 26.8% | 94.8% | 27.3% | 30.5% |
| hot set + cold scans | 77.0% | 64.1% | 76.6% | 55.6% | 16.7% | 76.4% |

Two rows are worth staring at.

**Scan pollution.** A loop over 33 keys through 32 slots makes LRU evict, every single time,
precisely the key it is about to request. Not a bad hit rate - exactly `0.0%`. Nothing is
broken; the policy is simply in lockstep with the workload and one step behind it. Note that
LFU and SLRU score zero too, and for a different reason: no key is ever touched twice, so no
counter ever rises and no entry ever earns protection. Only MRU, which throws away the entry it
just used, survives - by sacrificing the same slot over and over so the rest of the working set
lives.

**LFU stickiness.** When the hot set moves - a different set of keys is hot in each phase - LFU
collapses to 26.8% against LRU's 94.9%. The counters are a record of a popularity contest that
ended, and yesterday's winners hold the cache against today's. This is why real LFU
implementations age their counters, or pair the count with recency instead of trusting it alone.

## Which is why sglang ships five of them

`python/sglang/srt/mem_cache/evict_policy.py` does not pick a winner. Five strategies are five
`get_priority(node)` key functions over one shared heap:

```python
LRUStrategy   ->  node.last_access_time
LFUStrategy   ->  (node.hit_count, node.last_access_time)
FIFOStrategy  ->  node.creation_time
MRUStrategy   ->  -node.last_access_time
SLRUStrategy  ->  (node.hit_count >= threshold, node.last_access_time)

victim = heapq.nsmallest(1, entries, key=strategy.get_priority)[0]
```

Selected at launch by `--radix-eviction-policy {lru,lfu,fifo,mru,slru}`. The policy is expressed
as a comparison, not as five separate data structures - so adding one is a lambda, and the
scan-pollution disaster above is a flag rather than a rewrite. SLRU is the compromise that shows
up in the last row: an entry is protected once it has been hit twice, which buys +12 points over
LRU on the mixed workload without LFU's stickiness on the shifting one.

This eviction machinery sits under the prefix cache: the KV cache keeps the attention keys and
values of every prompt prefix already computed, the radix tree from day 24 lets a new request
find the longest prefix it shares with an old one, and when GPU memory fills up, something has
to go. The victim is chosen by exactly the policies on this page, over tree nodes rather than
plain keys - an eviction there throws away compute, not just bytes.

## Complexity

| Operation | Time | Space |
|---|---|---|
| LRU `get` / `put` | `O(1)` | `O(capacity)` |
| LRU on a plain list | `O(n)` | `O(capacity)` |
| LFU `get` / `put` | `O(1)` amortised | `O(capacity)` |
| LC 432 `inc` / `dec` / `getMaxKey` / `getMinKey` | `O(1)` | `O(keys)` |
| Bélády's MIN (offline) | `O(n log k)` | `O(n)` |
| heap-based policy eviction | `O(n)` per eviction over n candidates | `O(1)` |

## References
- [Cache replacement policies](https://en.wikipedia.org/wiki/Cache_replacement_policies)
- [Bélády's anomaly and the MIN algorithm](https://en.wikipedia.org/wiki/B%C3%A9l%C3%A1dy%27s_algorithm)
- [LeetCode 146 - LRU Cache](https://leetcode.com/problems/lru-cache/)
- [LeetCode 460 - LFU Cache](https://leetcode.com/problems/lfu-cache/)
- [LeetCode 432 - All O`one Data Structure](https://leetcode.com/problems/all-oone-data-structure/)
- [sglang eviction policies](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/mem_cache/evict_policy.py)
