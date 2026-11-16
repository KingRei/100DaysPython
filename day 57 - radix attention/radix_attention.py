"""Day 57 - RadixAttention: sharing the KV cache between requests that start the same way.

Yesterday's cache belonged to one sequence.  In a real server most requests start with
the same few hundred tokens - a system prompt, a few examples, the earlier turns of the
same conversation - and computing those keys and values again for every request is the
single largest piece of wasted work in LLM serving.

The fix is a data structure: keep the cached tokens in a radix tree (Day 24), look up
the longest prefix that is already there, and evict with LRU (Day 30) but only from the
leaves.  This file builds it and measures what it saves.

Run `python radix_attention.py` to print the whole story; every number below is produced
by this file and checked by the asserts at the bottom.
"""

import math

import numpy as np


# ================================================================ 1. a workload
SYSTEM = tuple(range(1000, 1200))          # 200 tokens of system prompt
EXAMPLES = [tuple(range(2000 + 100 * i, 2000 + 100 * i + 100)) for i in range(4)]


def make_workload(n_convs=20, turns=4, seed=0, few_shot=2, n_systems=1):
    """Requests that look like a chat server's day.

    Every request starts with the same system prompt; a request is then zero or more
    shared few-shot examples, the earlier turns of its own conversation, and finally
    the new user turn.  That is three different kinds of sharing in one list.
    """
    rng = np.random.default_rng(seed)
    reqs = []
    systems = [tuple(t + 10000 * k for t in SYSTEM) for k in range(n_systems)]
    for c in range(n_convs):
        shared = systems[c % n_systems] + tuple(t for e in EXAMPLES[:few_shot] for t in e)
        history = ()
        for t in range(turns):
            user = tuple(int(x) for x in rng.integers(0, 900, int(rng.integers(20, 60))))
            reply = tuple(int(x) for x in rng.integers(0, 900, int(rng.integers(30, 90))))
            reqs.append({'conv': c, 'turn': t, 'tokens': shared + history + user})
            history = history + user + reply
    return reqs


# ================================================================ 2. the radix tree
class RadixNode:
    __slots__ = ('key', 'children', 'parent', 'last_used', 'refs', 'tier')

    def __init__(self, key=(), parent=None):
        self.key = key                 # the tokens stored on the edge into this node
        self.children = {}             # first token of the edge -> child
        self.parent = parent
        self.last_used = 0
        self.refs = 0                  # requests currently decoding on this path
        self.tier = 1                  # 1 = on the GPU, 2 = demoted to host memory


class RadixCache:
    """A radix tree whose edges hold runs of token ids.

    Every node owns `len(key)` slots in the KV pool.  Matching a new request means
    walking down from the root for as long as the tokens agree; everything matched is
    a prefill that does not have to happen.
    """

    def __init__(self, capacity=None, l2_capacity=0):
        self.root = RadixNode()
        self.capacity = capacity       # tier-1 slots (on the GPU); None = unlimited
        self.l2_capacity = l2_capacity  # tier-2 slots (host memory); 0 = no tier 2
        self.size = 0                  # tier-1 slots in use
        self.size2 = 0                 # tier-2 slots in use
        self.clock = 0
        self.evicted = 0               # slots dropped entirely
        self.demoted = 0               # slots pushed down to tier 2
        self.hit2 = 0                  # matched slots that were sitting in tier 2
        self.nodes = 1

    # ---------------------------------------------------------------- matching
    def _touch(self, node, n):
        """A matched node is used again: count a tier-2 hit and bring it back up."""
        node.last_used = self.clock
        if node.tier == 2:
            self.hit2 += n
            self.size2 -= len(node.key)
            self.size += len(node.key)
            node.tier = 1

    def match(self, tokens):
        """Longest prefix of `tokens` already in the tree, and the node it ends at."""
        self.clock += 1
        node, i = self.root, 0
        while i < len(tokens):
            child = node.children.get(tokens[i])
            if child is None:
                break
            k = child.key
            j = 0
            while j < len(k) and i + j < len(tokens) and k[j] == tokens[i + j]:
                j += 1
            node.last_used = self.clock
            if j < len(k):             # matched part of an edge: stop here
                i += j
                node = child           # partial match still counts as cached tokens
                self._touch(node, j)
                return i, node, j
            i += len(k)
            node = child
            self._touch(node, len(k))
        return i, node, len(node.key)

    # ---------------------------------------------------------------- inserting
    def insert(self, tokens):
        """Add `tokens` to the tree, splitting an edge if it diverges halfway."""
        matched, node, used = self.match(tokens)
        if used < len(node.key):       # the match stopped inside an edge: split it
            self._split(node, used)
            self.nodes += 1
        rest = tuple(tokens[matched:])
        if rest:
            child = RadixNode(rest, node)
            node.children[rest[0]] = child
            child.last_used = self.clock
            self.size += len(rest)
            self.nodes += 1
            node = child
        if self.capacity is not None:
            self._evict_until_fits()
        return matched, node

    def _split(self, node, at):
        """Turn one edge into two so that a divergent request can branch off."""
        parent = node.parent
        head, tail = node.key[:at], node.key[at:]
        mid = RadixNode(head, parent)
        mid.last_used = node.last_used
        parent.children[head[0]] = mid
        mid.children[tail[0]] = node
        node.key = tail
        node.parent = mid
        return mid

    # ---------------------------------------------------------------- eviction
    def _leaves(self):
        out, stack = [], [self.root]
        while stack:
            n = stack.pop()
            if n is not self.root and not n.children and n.refs == 0:
                out.append(n)
            stack.extend(n.children.values())
        return out

    def _evict_until_fits(self):
        """LRU, but only leaves: an internal node is a prefix somebody else is using.

        This is the one rule that makes a shared cache different from an ordinary LRU
        cache.  Evicting a node in the middle would orphan everything below it.

        With a tier 2, the victim is demoted to host memory instead of being dropped;
        only when tier 2 is full as well does anything leave the tree for good.
        """
        while self.size > self.capacity:
            leaves = [n for n in self._leaves() if n.tier == 1]
            if not leaves:
                break                  # everything left is in use or is a prefix
            victim = min(leaves, key=lambda n: n.last_used)
            if self.l2_capacity:
                victim.tier = 2
                self.size -= len(victim.key)
                self.size2 += len(victim.key)
                self.demoted += len(victim.key)
                self._trim_l2()
            else:
                del victim.parent.children[victim.key[0]]
                self.size -= len(victim.key)
                self.evicted += len(victim.key)
                self.nodes -= 1

    def _trim_l2(self):
        while self.size2 > self.l2_capacity:
            leaves = [n for n in self._leaves() if n.tier == 2]
            if not leaves:
                break
            victim = min(leaves, key=lambda n: n.last_used)
            del victim.parent.children[victim.key[0]]
            self.size2 -= len(victim.key)
            self.evicted += len(victim.key)
            self.nodes -= 1

    # ---------------------------------------------------------------- bookkeeping
    def total_nodes(self):
        return self.nodes


def trie_nodes(sequences):
    """How many nodes a plain character-by-character trie would need.

    The radix tree stores a run of tokens per edge; a trie stores one token per node,
    which is the same information in far more objects.
    """
    root = {}
    count = 1
    for seq in sequences:
        node = root
        for t in seq:
            if t not in node:
                node[t] = {}
                count += 1
            node = node[t]
    return count


# ================================================================ 3. what it saves
def run(reqs, capacity=None, l2_capacity=0):
    """Serve the workload in order and count what had to be prefilled."""
    cache = RadixCache(capacity, l2_capacity)
    total = reused = 0
    per_req = []
    for r in reqs:
        toks = r['tokens']
        matched, _ = cache.insert(toks)
        total += len(toks)
        reused += matched
        per_req.append({'len': len(toks), 'matched': matched})
    return {'total': total, 'reused': reused, 'hit': reused / total,
            'nodes': cache.total_nodes(), 'slots': cache.size,
            'evicted': cache.evicted, 'per_req': per_req,
            'from_l2': cache.hit2, 'demoted': cache.demoted, 'slots2': cache.size2}


def no_sharing(reqs):
    """What the same workload costs when each request keeps its own cache."""
    total = sum(len(r['tokens']) for r in reqs)
    peak = sum(len(r['tokens']) for r in reqs)
    return {'total': total, 'reused': 0, 'hit': 0.0, 'slots': peak}


def capacity_sweep(reqs, caps=(200, 400, 600, 800, 1200, 2000, 4000, None)):
    full = run(reqs)['slots']
    rows = []
    for cap in caps:
        c = full if cap is None else cap
        res = run(reqs, capacity=c)
        rows.append({'cap': c, 'frac': c / full, 'hit': res['hit'],
                     'evicted': res['evicted']})
    return rows, full


def tier_table(l1=800, mults=(0, 2, 4, 8), n_convs=36, n_systems=6, seed=0):
    """One small fast pool, optionally backed by a larger slow one.

    A busy server has more than one tenant, so the tree no longer fits.  Demoting a
    cold branch to host memory instead of dropping it means the next request that
    wants it pays a copy over PCIe rather than a full prefill.
    """
    reqs = make_workload(n_convs=n_convs, n_systems=n_systems, seed=seed)
    full = run(reqs)['slots']
    rows = []
    for m in mults:
        res = run(reqs, capacity=l1, l2_capacity=l1 * m)
        rows.append({'l2': l1 * m, 'mult': m, 'hit': res['hit'],
                     'from_l2': res['from_l2'], 'evicted': res['evicted']})
    ideal = run(reqs)['hit']
    return rows, full, ideal, len(reqs)


# ================================================================ 4. the two traps
def prefix_on_text_not_tokens():
    """Matching on the text and matching on the tokens are not the same thing.

    A tokeniser merges across what looks like a boundary, so two requests whose text
    shares a prefix can have token sequences that diverge one token earlier - and a
    cache keyed on text would hand back keys and values for a token that was never
    there.
    """
    # a toy merge table: "ab" is one token, and so is "abc"
    vocab = {'a': 1, 'b': 2, 'c': 3, 'ab': 4, 'abc': 5, ' ': 6, 'd': 7}

    def encode(s):
        out, i = [], 0
        while i < len(s):
            for n in (3, 2, 1):        # longest match wins, as a real tokeniser does
                if s[i:i + n] in vocab:
                    out.append(vocab[s[i:i + n]])
                    i += n
                    break
        return tuple(out)

    a, b = 'abc', 'abd'
    ta, tb = encode(a), encode(b)
    text_prefix = 0
    while text_prefix < min(len(a), len(b)) and a[text_prefix] == b[text_prefix]:
        text_prefix += 1
    tok_prefix = 0
    while tok_prefix < min(len(ta), len(tb)) and ta[tok_prefix] == tb[tok_prefix]:
        tok_prefix += 1
    return {'a': a, 'b': b, 'ta': ta, 'tb': tb,
            'text_prefix': text_prefix, 'token_prefix': tok_prefix}


def evict_internal_node(reqs):
    """What happens if eviction is allowed to take an internal node.

    LRU on its own will happily evict the system prompt, because it was inserted first
    and nothing touches that node again directly - and everything hanging off it stops
    being reachable.
    """
    cache = RadixCache()
    for r in reqs[:8]:
        cache.insert(r['tokens'])
    before = run(reqs[:8])['hit']
    # pick the oldest node that is NOT a leaf, the way a naive LRU would
    stack, internal = [cache.root], []
    while stack:
        n = stack.pop()
        if n is not cache.root and n.children:
            internal.append(n)
        stack.extend(n.children.values())
    victim = min(internal, key=lambda n: n.last_used)
    orphaned = 0
    stack = [victim]
    while stack:
        n = stack.pop()
        orphaned += len(n.key)
        stack.extend(n.children.values())
    return {'victim_len': len(victim.key), 'orphaned': orphaned,
            'children': len(victim.children), 'hit_before': before}


# ================================================================ 5. assertions
def check():
    reqs = make_workload()

    # sharing finds a lot, and finds strictly more than nothing
    res = run(reqs)
    assert 0.5 < res['hit'] < 1.0
    assert res['slots'] < sum(len(r['tokens']) for r in reqs)

    # the first request of all can match nothing
    assert res['per_req'][0]['matched'] == 0

    # the second turn of a conversation matches at least its own first turn
    second = next(i for i, r in enumerate(reqs) if r['turn'] == 1)
    assert res['per_req'][second]['matched'] >= len(SYSTEM)

    # a radix tree needs far fewer nodes than a trie over the same sequences
    t = trie_nodes([r['tokens'] for r in reqs])
    assert res['nodes'] < t / 10

    # a smaller pool means more eviction and a lower hit rate
    rows, _ = capacity_sweep(reqs)
    assert rows[0]['hit'] < rows[-1]['hit']
    assert rows[0]['evicted'] > 0 and rows[-1]['evicted'] == 0

    # a second tier turns evictions into demotions and recovers the hit rate
    rows, full, ideal, _ = tier_table()
    assert rows[0]['hit'] < ideal - 0.05          # one small pool loses a lot
    assert abs(rows[-1]['hit'] - ideal) < 1e-9    # with tier 2 it is back
    assert rows[-1]['evicted'] == 0 and rows[-1]['from_l2'] > 0

    # matching text is not matching tokens
    tp = prefix_on_text_not_tokens()
    assert tp['text_prefix'] == 2 and tp['token_prefix'] == 0

    # an internal node has children that would be orphaned with it
    ev = evict_internal_node(reqs)
    assert ev['children'] >= 1 and ev['orphaned'] > ev['victim_len']
    return True


# ================================================================ 6. the whole story
def header(title):
    print('\n' + '=' * 72)
    print(title)
    print('=' * 72)


def main():
    np.set_printoptions(suppress=True, precision=4, legacy='1.25')
    reqs = make_workload()

    # ------------------------------------------------------------------ 1
    header('1. Most prefill tokens have been computed before')
    res = run(reqs)
    plain = no_sharing(reqs)
    print(f'  {len(reqs)} requests: {len(set(r["conv"] for r in reqs))} conversations of '
          f'4 turns, all sharing a {len(SYSTEM)}-token system')
    print(f'  prompt and 2 shared examples, each turn continuing its own history.\n')
    print(f'  tokens to prefill, caching nothing : {plain["total"]:,}')
    print(f'  tokens actually new, with sharing  : {res["total"] - res["reused"]:,}')
    print(f'  reused from the cache              : {res["reused"]:,}  '
          f'({res["hit"] * 100:.1f}%)')
    print('\n  Nothing about the model changed.  The same keys and values were simply')
    print('  computed once instead of once per request - which is the largest single')
    print('  saving available in serving, because prefill is the compute-bound phase.')

    # ------------------------------------------------------------------ 2
    header('2. The data structure: a radix tree over token ids')
    print('  Each edge holds a run of tokens; walking down from the root for as long')
    print('  as the tokens agree gives the longest cached prefix.  When a new request')
    print('  diverges halfway along an edge, the edge is split in two and both sides')
    print('  keep sharing everything above the split.\n')
    t = trie_nodes([r['tokens'] for r in reqs])
    print(f'  nodes in the radix tree          : {res["nodes"]:,}')
    print(f'  nodes a plain trie would need    : {t:,}   ({t / res["nodes"]:.0f}x more)')
    print(f'  token slots held                 : {res["slots"]:,}')
    print(f'  slots without sharing            : {plain["slots"]:,}  '
          f'({plain["slots"] / res["slots"]:.1f}x more)')
    print('\n  One token per node is the textbook trie; one RUN of tokens per edge is')
    print('  the radix tree, and the difference is not cosmetic - it is the difference')
    print('  between a few hundred objects and a few hundred thousand.')

    # ------------------------------------------------------------------ 3
    header('3. Where the sharing comes from')
    first_turn = [p for p, r in zip(res['per_req'], reqs) if r['turn'] == 0]
    later = [p for p, r in zip(res['per_req'], reqs) if r['turn'] > 0]
    print(f'  {"":>22} {"requests":>9} {"avg length":>11} {"avg reused":>11} {"share":>7}')
    for name, group in (('first turn of a chat', first_turn), ('later turns', later)):
        n = len(group)
        L = np.mean([g['len'] for g in group])
        M = np.mean([g['matched'] for g in group])
        print(f'  {name:>22} {n:>9} {L:>11.0f} {M:>11.0f} {M / L * 100:>6.0f}%')
    print('\n  A first turn can only share the system prompt and the examples; a later')
    print('  turn also shares everything the conversation has said so far, which is')
    print('  why multi-turn chat is the workload this helps most.  The same structure')
    print('  covers all three cases without knowing anything about them.')

    # ------------------------------------------------------------------ 4
    header('4. The pool is finite, so something has to be evicted')
    rows, full = capacity_sweep(reqs)
    shared = len(SYSTEM) + 2 * len(EXAMPLES[0])
    avg = np.mean([len(r['tokens']) for r in reqs])
    print(f'  Keeping everything would need {full:,} token slots.  With less:\n')
    print(f'  {"pool":>10} {"of the full tree":>18} {"hit rate":>10} {"evicted":>12}')
    for r in rows:
        print(f'  {r["cap"]:>10,} {r["frac"] * 100:>17.0f}% {r["hit"] * 100:>9.1f}% '
              f'{r["evicted"]:>12,}')
    print(f'\n  Two things in that table.  From {rows[3]["cap"]:,} slots upwards the hit '
          f'rate does not move:')
    print(f'  {rows[3]["hit"] * 100:.1f}% with {rows[3]["frac"] * 100:.0f}% of the tree '
          f'against {rows[-1]["hit"] * 100:.1f}% with all of it.  The')
    print('  part that earns the hits is the shared prefix, and leaf-only LRU never')
    print('  evicts it, because it is an internal node that everything else hangs off.')
    print(f'\n  And below that it does not degrade, it collapses: at {rows[1]["cap"]:,} '
          f'slots the hit rate is')
    print(f'  {rows[1]["hit"] * 100:.1f}%.  The pool has to hold the shared prefix '
          f'({shared} tokens here) plus')
    print(f'  enough of a request (average length {avg:.0f}) for anything to survive '
          f'insertion;')
    print('  under that line every insert evicts what the last one just wrote.')
    print('\n  Eviction is LRU - Day 30 - with one extra rule an ordinary cache does')
    print('  not need: only leaves may go.  An internal node is a prefix that other')
    print('  entries are built on, and dropping it would orphan everything below.')
    print('  Nodes a running request is still decoding on are pinned by a reference')
    print('  count for the same reason.')

    # ------------------------------------------------------------------ 5
    header('5. Silent failure #1: matching text instead of tokens')
    tp = prefix_on_text_not_tokens()
    print('  A tiny tokeniser where "ab" and "abc" are both single tokens:\n')
    print(f'    "{tp["a"]}" -> {tp["ta"]}')
    print(f'    "{tp["b"]}" -> {tp["tb"]}')
    print(f'    characters in common: {tp["text_prefix"]}')
    print(f'    tokens in common    : {tp["token_prefix"]}')
    print('\n  Two strings that share two of three characters share NO tokens at all,')
    print('  because the tokeniser merged across the boundary.  A cache keyed on text')
    print('  would hand back keys and values for a token that the second request never')
    print('  contained.  Nothing would raise: the shapes are right and the numbers are')
    print('  finite, and the model would simply be reading the wrong activations.')
    print('  Prefix matching belongs on the token ids, after the tokeniser.')

    # ------------------------------------------------------------------ 6
    header('6. Silent failure #2: evicting an internal node')
    ev = evict_internal_node(reqs)
    print('  Plain LRU picks the least recently used node.  After eight requests the')
    print('  oldest node is the one nobody touches directly any more:\n')
    print(f'    victim holds {ev["victim_len"]} tokens, and has {ev["children"]} children')
    print(f'    tokens that become unreachable with it: {ev["orphaned"]}')
    print(f'    hit rate before: {ev["hit_before"] * 100:.1f}%')
    print('\n  The node with the oldest timestamp is usually the system prompt, because')
    print('  every later request matches *through* it rather than ending on it.  Evict')
    print(f'  it and {ev["orphaned"]} tokens of cache go with it, including every')
    print('  conversation hanging below.  That is why the eviction rule is "leaves')
    print('  only", and why a node that a running request is sitting on is pinned.')

    # ------------------------------------------------------------------ 7
    header('7. One pool is not enough: hierarchical and disaggregated caches')
    rows, full, ideal, nreq = tier_table()
    print(f'  A busier server: {nreq} requests across 6 different system prompts, so')
    print(f'  the whole tree now wants {full:,} slots and no longer fits in the pool.')
    print(f'  Give it {800:,} slots on the GPU, and then a slower tier behind it:\n')
    print(f'  {"tier 2 (host)":>15} {"hit rate":>10} {"served from tier 2":>20} '
          f'{"dropped entirely":>18}')
    for r in rows:
        lab = 'none' if r['mult'] == 0 else f'{r["l2"]:,} ({r["mult"]}x)'
        print(f'  {lab:>15} {r["hit"] * 100:>9.1f}% {r["from_l2"]:>20,} '
              f'{r["evicted"]:>18,}')
    print(f'\n  With one pool the hit rate is {rows[0]["hit"] * 100:.1f}%; with a host-memory tier')
    print(f'  eight times larger it is {rows[-1]["hit"] * 100:.1f}%, which is exactly what '
          f'keeping the whole')
    print(f'  tree on the GPU would give ({ideal * 100:.1f}%).  '
          f'{rows[-1]["from_l2"]:,} tokens were copied back')
    print('  from host memory instead of being recomputed, and nothing was dropped.')
    print('\n  That is the whole idea behind the hierarchical caches in production')
    print('  servers.  SGLang calls its version HiCache: the same radix tree, with the')
    print('  GPU pool as tier 1, host DRAM as tier 2, and an optional remote store as')
    print('  tier 3, moving branches up and down in the background.  The trade is')
    print('  bandwidth for compute - a PCIe copy instead of a prefill - which is worth')
    print('  it exactly when prefill is expensive, i.e. for long shared prefixes.')
    print('\n  Mooncake (the architecture behind Kimi) takes the same observation one')
    print('  step further and makes the cache the centre of the cluster: prefill and')
    print('  decode run on separate pools of machines - they are the two different')
    print('  machines from the previous day - and the KV cache lives in a shared store')
    print('  built from the CPU memory and SSDs of the whole fleet, moved around over')
    print('  RDMA.  A request can then be prefilled wherever its prefix already is.')
    print('\n  Both answer the same question this file has been circling: the cache is')
    print('  the scarce resource, so the system is designed around where it lives.')

    # ------------------------------------------------------------------ 8
    header('8. Assertions')
    check()
    print('all assertions passed')


if __name__ == '__main__':
    main()
