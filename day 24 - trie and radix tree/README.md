# Tries and radix trees: storing a shared prefix exactly once

More details in:
https://medium.com/100-days-of-python

Interactive walkthrough: [demo.html](demo.html) - a single self-contained page
(no build step, works offline, 中文 / English toggle) that animates a trie growing one
character at a time, the radix tree splitting an edge when a new word diverges in the middle
of it, a routing table answering a longest-prefix-match query in a bounded number of steps
whatever order the rules were added in, an inference server's KV cache reusing a shared prompt
prefix, the page alignment that quietly strands the tail of a match, LRU eviction that can only
touch leaves, and LeetCode 211's `'.'` wildcard turning a lookup into a bounded DFS.

![img](https://github.com/KingRei/100DaysPython/blob/master/day%2024%20-%20trie%20and%20radix%20tree/imgs/day24_1.png?raw=true)

A hash set answers "is this string in the set". A trie answers a different question: "what do
these strings have in common". It stores one node per character, so every prefix of every
stored word is a real node in the tree, and walking down from the root spells the string out.
Because a prefix is a node, "does anything start with `rubic`" costs the same as "is `rubic` a
word", and the two are told apart by a single boolean on the node - `is_word`. Insertion costs
`O(len(word))` no matter how many words are already stored, and there is no hash, no resizing,
no collision, and no comparison against any other key.

The price is space. Storing `romane`, `romanus`, `romulus`, `rubens`, `ruber`, `rubicon` and
`rubicundus` takes 28 nodes, and most of them are single-child chains that carry no information
- once you are at `rubicund` there is nowhere else to go, so the seven nodes that spell out
`undus` are seven pointer hops standing in for a `strcmp`. A radix tree (Morrison's PATRICIA
tree) collapses each of those chains into one edge labelled with the whole substring. The same
seven words take 14 nodes, half as many; on a 32-word English list it is 115 nodes down to 43,
a 63% cut. Nodes are now branch points, and strings live on the edges.

Keying each node's children by the *first* character of the edge label keeps edge selection
`O(1)`, so a lookup never scans siblings. The one operation that gets harder is insertion, and
it is worth doing by hand once.

## The split

Inserting into a radix tree walks down matching edge labels, and eventually one of three things
happens: the label is consumed and we recurse, the query is consumed and we mark the node, or
the two disagree halfway through a label. That last case is the split. The shared head of the
label becomes a new internal node, the old edge keeps its tail and reparents under it, and the
remainder of the new word hangs off as a sibling. Nothing is copied and nothing else in the
tree moves - a split is a local, `O(1)` pointer rearrangement, and the seven-word tree above
needs exactly six of them.

Splitting is why a radix tree is not just a compressed trie you build once. It is the operation
that lets the structure stay compressed while it is being written to, which is precisely what an
online cache needs.

## Longest prefix match

Every IP packet leaving a machine has to be matched against a routing table, and the rule is
"most specific wins": `10.20.30.0/24` beats `10.20.0.0/16` beats `10.0.0.0/8` beats the default
route. Put the prefixes in a binary trie over the address bits and "most specific wins" becomes
"the deepest marked node on the path from the root", which falls out of the walk for free -
remember the last marked node you passed, and when the walk runs out of tree, that is the
answer.

Two properties matter more than the constant factor. Rule order is irrelevant, because
specificity is depth and depth is not a function of insertion order, so the table can be edited
without re-sorting anything. And the walk is bounded by the *address*, not the table: at most 33
nodes for IPv4 whether the table holds the five routes in the demo or the roughly one million a
core router carries, while a linear scan touches every rule. That is the whole reason hardware
routing tables are tries.

## The same tree, one abstraction up

An LLM inference server has a structurally identical problem. Two requests that share a system
prompt share the KV cache entries for those tokens - the attention state for token `i` depends
only on tokens `0..i`, so an identical prefix has an identical cache. sglang's `RadixCache` is a
radix tree whose "characters" are token ids and whose edge labels are runs of tokens. In the
module's worked example three requests ask for 160 tokens in total and occupy 68 KV slots, a
57% saving, and a fourth request that diverges in the middle of a cached run triggers exactly
the split described above.

Two details are where the theory meets the hardware. First, the cache is paged: KV memory is
allocated in blocks, so a match is rounded down to a whole number of pages. Ten tokens in
common with `page_size=4` yields eight reusable tokens - the last two are identical and still
get recomputed, because they live in a page that also holds a token the two requests disagree
about. Second, eviction is LRU over *leaves only*. An interior node is a prefix that something
else is still using, so it cannot be freed even if it is old; the shared system prompt near the
root outlives the request tails hanging off it, which is exactly the behaviour you want and
falls out of the tree shape rather than any policy. Freeing 16 tokens in the demo takes the
tree from 80 nodes to 64 and never touches the root path. This is the machinery behind
RadixAttention, and it is a radix tree with an eviction heap bolted on.

## The LeetCode problems

LC 208 is the trie itself, and the interesting line is `search` versus `startsWith`: both walk
the same path, and only the final `is_word` check differs. LC 211 adds a `'.'` wildcard, and it
is the problem that proves a trie is not a compressed hash map. A `'.'` means there is no key to
look up, so the walk forks into every child and the lookup becomes a DFS - bounded, because the
depth is still the pattern length, but no longer a single path. LC 648 (replace words) is
longest prefix match on English: walk the trie until you hit a marked node and stop, which is
the routing table again with roots instead of subnets.

## Complexity

`L` = key length, `n` = number of keys, `T` = total characters stored, `sigma` = alphabet size.

| Operation | Trie | Radix tree | Space |
|---|---|---|---|
| `insert` / `search` / `starts_with` | O(L) | O(L) | - |
| `autocomplete(prefix)`, `k` results | O(L + k) | O(L + k) | O(k) |
| Nodes needed | O(T) | O(n) | O(T) vs O(n) |
| Longest prefix match, `b`-bit key | O(b) | O(b) | O(1) |
| Linear scan of the same table | O(n · L) | - | O(1) |
| Wildcard search, `d` dots (LC 211) | O(sigma^d · L) | - | O(L) |
| Split on insert | - | O(1) | O(1) |

## References
- [Trie - Wikipedia](https://en.wikipedia.org/wiki/Trie)
- [Radix tree - Wikipedia](https://en.wikipedia.org/wiki/Radix_tree)
- [PATRICIA - Practical Algorithm To Retrieve Information Coded in Alphanumeric (Morrison, 1968)](https://dl.acm.org/doi/10.1145/321479.321481)
- [Longest prefix match - Wikipedia](https://en.wikipedia.org/wiki/Longest_prefix_match)
- [SGLang: Efficient Execution of Structured Language Model Programs (RadixAttention)](https://arxiv.org/abs/2312.07104)
- [`sglang/srt/mem_cache/radix_cache.py`](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/mem_cache/radix_cache.py)
- [LeetCode 208 · Implement Trie (Prefix Tree)](https://leetcode.com/problems/implement-trie-prefix-tree/)
- [LeetCode 211 · Design Add and Search Words Data Structure](https://leetcode.com/problems/design-add-and-search-words-data-structure/)
- [LeetCode 648 · Replace Words](https://leetcode.com/problems/replace-words/)
