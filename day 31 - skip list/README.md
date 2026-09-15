# Skip list: an ordered map built out of coin flips

More details in:
https://medium.com/100-days-of-python

Interactive walkthrough: [demo.html](demo.html) - a single self-contained page
(no build step, works offline, 中文 / English toggle) that searches the same key through a skip
list and through the flat sorted list underneath it, flips a coin for a new node's height and
splices it level by level, adds up spans to answer `ZRANK`, comments out one line so that the
order stays perfect while every rank quietly goes wrong, and finishes on LeetCode 1206 where a
`<` written as `<=` walks straight past all three copies of a duplicate key.

![img](https://github.com/KingRei/100DaysPython/blob/master/day%2031%20-%20skip%20list/imgs/day31_1.png?raw=true)

A sorted array can be binary searched because it has random access. A sorted linked list has the
same order and cannot be binary searched at all: there is no way to land in the middle without
walking there first. Searching 198 among 100 sorted keys costs 7 probes in the array and 99 hops
in the list.

A skip list buys the missing jumps back. It keeps the sorted linked list exactly as it is, and
stacks a few sparser copies of it on top as express lanes. Search starts on the highest lane,
moves right while the next key is still small enough, and drops a level when it is not. Every
drop lands on a node already known to be below the target, so nothing is ever re-walked: the
whole search is a staircase down and to the right.

The interesting part is not the lanes, it is how their heights are chosen. Perfectly regular
lanes - every second node on level 1, every fourth on level 2 - give a beautiful `O(log n)`
search and are unmaintainable: inserting a single key at the front shifts every other key's
parity, and in the 16-key example all 16 nodes change level. Instead, each node flips a coin
when it is born: keep flipping while heads, stop at the first tail. The height is decided once,
at random, and never changes again. No rotations, no rebalancing, no parent pointers - the
balance of day 28's AVL and red-black trees, handed over to probability.

## Search: right until you cannot, then down

```python
def _descend(self, key):
    update, rank = [None] * self.max_level, [0] * self.max_level
    node = self.head
    for i in range(self.level - 1, -1, -1):
        rank[i] = 0 if i == self.level - 1 else rank[i + 1]
        while node.forward[i] is not None and node.forward[i].key < key:
            rank[i] += node.span[i]       # count what this hop flew over
            node = node.forward[i]
        update[i] = node                  # the pointer that may be rewritten
    return update, rank, node.forward[0]
```

One traversal serves search, insert and delete. It returns `update[i]`, the last node on level
`i` that is still strictly below the key - exactly the node whose forward pointer would have to
be rewritten if we inserted here - and `rank[i]`, how far along level 0 that node sits. On the
16-key list in the module, `search(13)` reaches its answer in 2 hops and a miss like
`search(99)` costs 3.

Expected search cost is `O(log n)` and, unlike a balanced tree, that is a statement about the
random heights rather than about the data: an adversary who picks the keys cannot make a skip
list degenerate, because the shape does not depend on the keys at all.

## Spans: the same walk, now answering "what rank is this?"

```python
for i in range(lvl):
    node.span[i] = update[i].span[i] - (rank[0] - rank[i])
    update[i].span[i] = (rank[0] - rank[i]) + 1
for i in range(lvl, self.level):
    update[i].span[i] += 1        # the line everyone forgets
```

Give every pointer a second number - how many level-0 nodes it flies over - and the search walk
starts returning positions for free. Summing the spans of the hops taken to reach key 13 gives
`10 + 2 + 1 = 13`, so its 0-based rank is 12. Run the sums the other way and the same structure
answers `select(k)`: descend while the accumulated span stays below `k`. Those two operations
are Redis's `ZRANK` and index-based `ZRANGE`.

The second loop is where the bug lives. Levels *above* the new node's height need no pointer
change at all, which makes them easy to skip - but each of them now flies over one more node, so
its span must grow by one. Delete that loop and the structure stays perfectly ordered: `keys()`
matches a correct list, `search()` matches a correct list, `range()` matches a correct list.
Only `rank()` disagrees, on 197 of 200 keys, starting at key 3, which reports rank 0 instead of
3. Every test you would write for an ordered set still passes while `ZRANK` lies.

![img](https://github.com/KingRei/100DaysPython/blob/master/day%2031%20-%20skip%20list/imgs/day31_5.png?raw=true)

## Why Redis picks p = 0.25

The coin does not have to be fair. With probability `p` of promoting a node one more level, the
expected number of forward pointers per node is `1 / (1 - p)`, and the expected number of levels
to walk grows like `log n / log(1/p)`. Measured over 20,000 keys and 2,000 searches:

| p | pointers/node | theory | top level | avg steps |
|---|---|---|---|---|
| 0.5 | 2.00 | 2.00 | 15 | 14.1 |
| 0.25 | 1.34 | 1.33 | 8 | 19.1 |
| 0.1 | 1.11 | 1.11 | 6 | 29.2 |

Redis's `t_zset.c` uses `ZSKIPLIST_P 0.25` and `ZSKIPLIST_MAXLEVEL 32`: compared with a fair
coin it keeps a third of the *optional* pointers - 0.34 per node instead of 1.00, on top of the
one every node must have - and pays 35% more hops for them. Pointers are memory that every node
pays; hops are cache misses that only a query pays, and there are fewer queries than nodes in a
sorted set of any size.

The rest of the Redis structure follows from what a sorted set has to do. Each node also carries
a backward pointer on level 0, because `ZREVRANGE` has to walk the other way. Nodes are ordered
by score and then by member, which is what makes `ZRANGEBYLEX` meaningful when every score is
equal. And below `zset-max-listpack-entries` (128 by default) there is no skip list at all - a
flat listpack is scanned instead, because for a handful of elements the pointers cost more than
the walk saves.

## LeetCode 1206, and the `<` that cannot be `<=`

Design Skiplist allows duplicate values, and `erase` must remove exactly one copy. That single
requirement pins down the comparison in the descent: `while node.forward[i].val < target` stops
in front of the *first* copy, which is the one `erase` unlinks. Write `<=` and the walk sails
past all three copies of a 3 and reports that 3 is not in the list. Nothing crashes, nothing is
out of order - the same shape of failure as the missing span bump.

LeetCode 315, counting smaller numbers after self, also falls out of spans: insert the array
from the right and ask `count_less(x)` before each insert. It agrees with the Fenwick tree
answer from day 29, which is the point - a Fenwick tree counts prefixes over value indices, a
skip list counts them over the values themselves, and neither needs the values compressed into
a range first.

## The same layered idea in metric space

The lanes are not really about keys. They are about approaching a target with big strides and
finishing with small ones, and that idea survives when "sorted order" no longer exists. HNSW -
the index behind most vector databases - draws each node's layer from `floor(-ln(U) * mL)`,
which is the same geometric distribution, greedily descends the sparse upper layers, and then
runs a beam of width `ef` on the bottom one. The beam is the one real difference: a line cannot
trap a greedy walk, but a metric space can, so the bottom layer needs several candidates alive
at once. The toy implementation in the module keeps recall at 1.00 while the distance count
barely moves:

| n | layers | recall@1 | HNSW distances | brute force | speedup |
|---|---|---|---|---|---|
| 500 | 4 | 1.00 | 10,146 | 50,000 | 4.9x |
| 2,000 | 5 | 1.00 | 12,316 | 200,000 | 16.2x |
| 8,000 | 5 | 1.00 | 13,655 | 800,000 | 58.6x |

16x the data, 1.1x the work.

## Complexity

| Operation | Time | Space |
|---|---|---|
| `search` / `insert` / `delete` | `O(log n)` expected | `O(n)` expected, `O(n log n)` worst |
| `rank` / `select` / `count_less` | `O(log n)` expected | - |
| `range(lo, hi)` | `O(log n + k)` | - |
| forward pointers per node | `1 / (1 - p)` expected | - |
| sorted linked list, same search | `O(n)` | `O(n)` |

## References
- [Skip list](https://en.wikipedia.org/wiki/Skip_list)
- [Pugh, "Skip Lists: A Probabilistic Alternative to Balanced Trees" (1990)](https://15721.courses.cs.cmu.edu/spring2018/papers/08-oltpindexes1/pugh-skiplists-cacm1990.pdf)
- [Redis t_zset.c](https://github.com/redis/redis/blob/unstable/src/t_zset.c)
- [Malkov & Yashunin, "Efficient and robust approximate nearest neighbor search using HNSW" (2016)](https://arxiv.org/abs/1603.09320)
- [LeetCode 1206 - Design Skiplist](https://leetcode.com/problems/design-skiplist/)
- [LeetCode 315 - Count of Smaller Numbers After Self](https://leetcode.com/problems/count-of-smaller-numbers-after-self/)
