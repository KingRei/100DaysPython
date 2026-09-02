"""Day 24 - Trie, radix (Patricia) tree, and the prefix cache in an LLM server.

Run me:  python trie_radix.py

A hash map answers "have I seen this exact string?".  A trie answers a different
question - "what do I know about strings that *start* like this?" - and that one
question is behind autocomplete, IP routing tables, and the prefix cache that lets
an LLM server skip recomputing a system prompt it has already seen.

Sections
    1. Trie: one node per character
    2. Radix / Patricia tree: collapse the single-child chains
    3. Longest prefix match: how a router picks a route
    4. A token-level radix cache, the way sglang does it
    5. LeetCode 208 / 211 / 648
"""

from __future__ import annotations

import time
from typing import Dict, Iterator, List, Optional, Tuple


# --------------------------------------------------------------------------
# 1. Trie - one node per character
# --------------------------------------------------------------------------

class TrieNode:
    __slots__ = ("children", "is_word")

    def __init__(self) -> None:
        self.children: Dict[str, "TrieNode"] = {}
        self.is_word = False


class Trie:
    """Classic character trie.  Lookup costs O(len(word)) - the number of words
    stored never enters the complexity, which is the whole point."""

    def __init__(self) -> None:
        self.root = TrieNode()
        self.n_words = 0

    def insert(self, word: str) -> None:
        node = self.root
        for ch in word:
            nxt = node.children.get(ch)
            if nxt is None:
                nxt = TrieNode()
                node.children[ch] = nxt
            node = nxt
        if not node.is_word:
            node.is_word = True
            self.n_words += 1

    def _walk(self, prefix: str) -> Optional[TrieNode]:
        """Follow `prefix` from the root; None if the path falls off the tree."""
        node = self.root
        for ch in prefix:
            node = node.children.get(ch)
            if node is None:
                return None
        return node

    def search(self, word: str) -> bool:
        node = self._walk(word)
        return node is not None and node.is_word

    def starts_with(self, prefix: str) -> bool:
        return self._walk(prefix) is not None

    def autocomplete(self, prefix: str, limit: Optional[int] = None) -> List[str]:
        """Every stored word under `prefix`, in lexicographic order.

        Two phases, and only the first one depends on the prefix length:
        walk down to the prefix node, then collect the subtree below it.
        """
        node = self._walk(prefix)
        if node is None:
            return []
        out: List[str] = []
        self._collect(node, prefix, out, limit)
        return out

    def _collect(self, node: TrieNode, path: str, out: List[str],
                 limit: Optional[int]) -> None:
        if limit is not None and len(out) >= limit:
            return
        if node.is_word:
            out.append(path)
        for ch in sorted(node.children):
            self._collect(node.children[ch], path + ch, out, limit)
            if limit is not None and len(out) >= limit:
                return

    def shortest_root(self, word: str) -> Optional[str]:
        """The shortest stored word that is a prefix of `word` (LeetCode 648)."""
        node = self.root
        for i, ch in enumerate(word):
            node = node.children.get(ch)
            if node is None:
                return None
            if node.is_word:
                return word[: i + 1]
        return None

    def count_nodes(self) -> int:
        n = 0
        stack = [self.root]
        while stack:
            node = stack.pop()
            n += 1
            stack.extend(node.children.values())
        return n


# --------------------------------------------------------------------------
# 2. Radix / Patricia tree - collapse the single-child chains
# --------------------------------------------------------------------------

def common_prefix_len(a: str, b: str) -> int:
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


class RadixNode:
    __slots__ = ("children", "is_word")

    def __init__(self, is_word: bool = False) -> None:
        # first character of the edge label -> (label, child)
        # Keying on the first character is what keeps the child lookup O(1):
        # at most one child can possibly match, and only then do we compare
        # the whole label.
        self.children: Dict[str, Tuple[str, "RadixNode"]] = {}
        self.is_word = is_word


class RadixTree:
    """A trie whose edges carry strings instead of single characters.

    Every node in a plain trie that has exactly one child and stores no word is
    pure overhead - it exists only to be walked through.  Merging those chains
    into one labelled edge is the entire idea; the price is that insertion has
    to be able to *split* an edge when a new word diverges in the middle of it.
    """

    def __init__(self) -> None:
        self.root = RadixNode()
        self.n_words = 0
        self.n_splits = 0

    def insert(self, word: str) -> None:
        node = self.root
        i = 0
        while i < len(word):
            first = word[i]
            edge = node.children.get(first)
            if edge is None:
                # Nothing shares even one character: hang the whole rest here.
                node.children[first] = (word[i:], RadixNode(is_word=True))
                self.n_words += 1
                return
            label, child = edge
            k = common_prefix_len(label, word[i:])
            if k == len(label):
                # The edge is fully consumed - descend and keep going.
                node, i = child, i + k
                continue
            # Diverged inside the label: split the edge at k.
            self.n_splits += 1
            mid = RadixNode()
            mid.children[label[k]] = (label[k:], child)
            node.children[first] = (label[:k], mid)
            i += k
            if i == len(word):
                mid.is_word = True
            else:
                mid.children[word[i]] = (word[i:], RadixNode(is_word=True))
            self.n_words += 1
            return
        if not node.is_word:
            node.is_word = True
            self.n_words += 1

    def _walk(self, s: str) -> Optional[Tuple[RadixNode, int]]:
        """Return (node, consumed) where the walk ended, or None if `s` runs off
        the tree.  `consumed` may exceed len(s) when we stop mid-edge."""
        node = self.root
        i = 0
        while i < len(s):
            edge = node.children.get(s[i])
            if edge is None:
                return None
            label, child = edge
            rest = s[i:]
            k = common_prefix_len(label, rest)
            if k == len(rest):          # prefix ends inside (or at) this edge
                return child, i + len(label)
            if k < len(label):          # mismatch inside the label
                return None
            node, i = child, i + k
        return node, i

    def search(self, word: str) -> bool:
        hit = self._walk(word)
        if hit is None:
            return False
        node, consumed = hit
        return consumed == len(word) and node.is_word

    def starts_with(self, prefix: str) -> bool:
        return self._walk(prefix) is not None

    def words(self) -> List[str]:
        out: List[str] = []
        self._collect(self.root, "", out)
        return out

    def _collect(self, node: RadixNode, path: str, out: List[str]) -> None:
        if node.is_word:
            out.append(path)
        for first in sorted(node.children):
            label, child = node.children[first]
            self._collect(child, path + label, out)

    def count_nodes(self) -> int:
        n = 0
        stack = [self.root]
        while stack:
            node = stack.pop()
            n += 1
            stack.extend(child for _, child in node.children.values())
        return n

    def dump(self) -> List[str]:
        """Human-readable edge list, one line per edge, indented by depth."""
        lines: List[str] = []

        def rec(node: RadixNode, depth: int) -> None:
            for first in sorted(node.children):
                label, child = node.children[first]
                mark = " *" if child.is_word else ""
                lines.append("  " * depth + "-" + label + mark)
                rec(child, depth + 1)

        rec(self.root, 0)
        return lines


# --------------------------------------------------------------------------
# 3. Longest prefix match - how a router picks a route
# --------------------------------------------------------------------------

def ip_to_int(ip: str) -> int:
    a, b, c, d = (int(x) for x in ip.split("."))
    return (a << 24) | (b << 16) | (c << 8) | d


def bits_of(value: int, length: int, width: int = 32) -> str:
    """The first `length` bits of a `width`-bit value, as '0'/'1' characters."""
    return format(value, "0%db" % width)[:length]


class BinaryTrieRouter:
    """A routing table as a binary trie: one level per bit of the address.

    A route is not a key, it is a *set* of keys - 10.20.0.0/16 covers 65536
    addresses.  Rules overlap on purpose, and the rule that wins is the most
    specific one, i.e. the longest matching prefix.  In a trie that is free:
    walk down the address bit by bit and remember the deepest node that carried
    a next hop.  Insertion order does not matter, which is exactly what an
    ordered list of firewall-style rules cannot promise.
    """

    def __init__(self) -> None:
        self.root = TrieNode()          # children keyed by '0' / '1'
        self.hop: Dict[int, str] = {}   # id(node) -> next hop
        self.plen: Dict[int, int] = {}  # id(node) -> prefix length

    def add(self, cidr: str, next_hop: str) -> None:
        net, length = cidr.split("/")
        length = int(length)
        node = self.root
        for bit in bits_of(ip_to_int(net), length):
            nxt = node.children.get(bit)
            if nxt is None:
                nxt = TrieNode()
                node.children[bit] = nxt
            node = nxt
        node.is_word = True
        self.hop[id(node)] = next_hop
        self.plen[id(node)] = length

    def lookup(self, ip: str) -> Tuple[Optional[str], int, int]:
        """Return (next hop, matched prefix length, nodes visited)."""
        node = self.root
        best_hop = self.hop.get(id(self.root))
        best_len = self.plen.get(id(self.root), 0)
        visited = 1
        for bit in bits_of(ip_to_int(ip), 32):
            node = node.children.get(bit)
            if node is None:
                break
            visited += 1
            if node.is_word:
                best_hop = self.hop[id(node)]
                best_len = self.plen[id(node)]
        return best_hop, best_len, visited


def linear_lookup(table: List[Tuple[str, str]], ip: str) -> Tuple[Optional[str], int]:
    """The obvious alternative: scan every rule, keep the most specific hit.
    Correct, but it touches the whole table for every packet."""
    target = ip_to_int(ip)
    best_hop, best_len = None, -1
    for cidr, hop in table:
        net, length = cidr.split("/")
        length = int(length)
        mask = ((1 << length) - 1) << (32 - length) if length else 0
        if (target & mask) == (ip_to_int(net) & mask) and length > best_len:
            best_hop, best_len = hop, length
    return best_hop, len(table)


# --------------------------------------------------------------------------
# 4. A token-level radix cache, the way an LLM server does it
# --------------------------------------------------------------------------

class KVNode:
    """One node = one contiguous run of tokens plus the KV slots holding them."""

    __slots__ = ("key", "value", "children", "parent", "last_access", "lock_ref")

    def __init__(self, key: Tuple[int, ...] = (), value: Tuple[int, ...] = (),
                 parent: Optional["KVNode"] = None) -> None:
        self.key = key
        self.value = value
        self.children: Dict[Tuple[int, ...], "KVNode"] = {}
        self.parent = parent
        self.last_access = time.monotonic()
        self.lock_ref = 0


class TokenRadixCache:
    """A radix tree over token ids instead of characters.

    Three things change once the alphabet is a 150k-token vocabulary and the
    values are GPU memory:

    * children are stored in a dict keyed by the first *page* of the edge, so
      picking the candidate child is O(1) even with a huge branching factor;
    * matches are rounded down to a whole number of pages, because KV cache is
      allocated a page at a time and half a page is not shareable;
    * eviction is LRU over *leaves only* - freeing an interior node would
      orphan everything under it.
    """

    def __init__(self, page_size: int = 1) -> None:
        self.page_size = page_size
        self.root = KVNode()
        self.next_slot = 0
        self.n_splits = 0
        self.allocated = 0      # KV slots actually handed out
        self.requested = 0      # tokens asked for across all requests

    # -- helpers ----------------------------------------------------------
    def _child_key(self, key: Tuple[int, ...]) -> Tuple[int, ...]:
        return key[: self.page_size]

    def _match_len(self, a: Tuple[int, ...], b: Tuple[int, ...]) -> int:
        n = min(len(a), len(b))
        i = 0
        while i < n and a[i] == b[i]:
            i += 1
        return i // self.page_size * self.page_size

    def _split(self, node: KVNode, at: int) -> KVNode:
        """Cut `node` into a prefix node and a suffix node; return the prefix."""
        self.n_splits += 1
        mid = KVNode(node.key[:at], node.value[:at], node.parent)
        node.parent.children[self._child_key(node.key)] = mid
        node.key, node.value = node.key[at:], node.value[at:]
        node.parent = mid
        mid.children[self._child_key(node.key)] = node
        mid.last_access = node.last_access
        return mid

    # -- public API -------------------------------------------------------
    def match_prefix(self, tokens: Tuple[int, ...]) -> Tuple[int, KVNode]:
        """Longest cached prefix of `tokens`.  Splits a node if the match ends
        in the middle of one, so the shared part becomes a real boundary."""
        tokens = tokens[: len(tokens) // self.page_size * self.page_size]
        node, matched = self.root, 0
        now = time.monotonic()
        node.last_access = now
        while tokens:
            child = node.children.get(self._child_key(tokens))
            if child is None:
                break
            k = self._match_len(child.key, tokens)
            child.last_access = now
            if k < len(child.key):
                node = self._split(child, k)
                matched += k
                break
            node = child
            matched += k
            tokens = tokens[k:]
        return matched, node

    def insert(self, tokens: Tuple[int, ...]) -> Tuple[int, int]:
        """Cache `tokens`; return (reused, allocated) token counts."""
        self.requested += len(tokens)
        matched, node = self.match_prefix(tokens)
        rest = tokens[matched:]
        rest = rest[: len(rest) // self.page_size * self.page_size]
        if rest:
            slots = tuple(range(self.next_slot, self.next_slot + len(rest)))
            self.next_slot += len(rest)
            self.allocated += len(rest)
            child = KVNode(rest, slots, node)
            node.children[self._child_key(rest)] = child
            node = child
        return matched, len(rest)

    def leaves(self) -> List[KVNode]:
        out: List[KVNode] = []
        stack = [self.root]
        while stack:
            n = stack.pop()
            kids = list(n.children.values())
            if not kids and n is not self.root and n.lock_ref == 0:
                out.append(n)
            stack.extend(kids)
        return out

    def evict(self, n_tokens: int) -> int:
        """Free at least `n_tokens` slots, oldest leaf first.

        After a leaf goes, its parent may itself have become a leaf - that is
        how a whole cold branch peels away one node at a time, while a hot
        shared prefix nearer the root survives.
        """
        freed = 0
        while freed < n_tokens:
            cands = self.leaves()
            if not cands:
                break
            victim = min(cands, key=lambda n: n.last_access)
            del victim.parent.children[self._child_key(victim.key)]
            freed += len(victim.key)
        return freed

    def size(self) -> int:
        total, stack = 0, [self.root]
        while stack:
            n = stack.pop()
            total += len(n.key)
            stack.extend(n.children.values())
        return total

    def dump(self) -> List[str]:
        """One line per node: how many tokens it holds and which ones."""
        lines: List[str] = []

        def rec(node: KVNode, depth: int) -> None:
            for child in node.children.values():
                k = child.key
                span = ("%d..%d" % (k[0], k[-1])) if len(k) > 2 else str(list(k))
                lines.append("  " * depth + "- %2d tokens  %s" % (len(k), span))
                rec(child, depth + 1)

        rec(self.root, 0)
        return lines


# --------------------------------------------------------------------------
# 5. LeetCode
# --------------------------------------------------------------------------

class WordDictionary:
    """LeetCode 211 - add and search words, where '.' matches any letter.

    This is the problem that shows a trie is not just a compressed hash map.
    With a real wildcard the key is not known, so there is nothing to hash;
    a '.' turns the walk into a branch over every child, and the search
    becomes a bounded DFS.  Cost is O(26^k * m) in the worst case for k dots.
    """

    def __init__(self) -> None:
        self.root = TrieNode()

    def addWord(self, word: str) -> None:
        node = self.root
        for ch in word:
            node = node.children.setdefault(ch, TrieNode())
        node.is_word = True

    def search(self, word: str) -> bool:
        return self._dfs(self.root, word, 0)

    def _dfs(self, node: TrieNode, word: str, i: int) -> bool:
        if i == len(word):
            return node.is_word
        ch = word[i]
        if ch != ".":
            nxt = node.children.get(ch)
            return nxt is not None and self._dfs(nxt, word, i + 1)
        return any(self._dfs(child, word, i + 1) for child in node.children.values())


def replace_words(dictionary: List[str], sentence: str) -> str:
    """LeetCode 648 - replace every word by the shortest root that prefixes it.

    'Shortest root' means: stop at the first node on the way down that is
    marked as a word.  One walk per word, no scanning of the dictionary.
    """
    trie = Trie()
    for root in dictionary:
        trie.insert(root)
    return " ".join(trie.shortest_root(w) or w for w in sentence.split())


# --------------------------------------------------------------------------
# Demo
# --------------------------------------------------------------------------

ROMAN = ["romane", "romanus", "romulus", "rubens", "ruber", "rubicon", "rubicundus"]

WORDS = [
    "import", "imports", "important", "impossible", "improve",
    "in", "inner", "input", "inputs", "insert", "inspect", "install",
    "instance", "int", "integer", "interface", "internal", "interpreter",
    "print", "printer", "prints", "priority", "private", "probe", "process",
    "profile", "program", "project", "prompt", "property", "protocol", "prune",
]


def head(title: str) -> None:
    print()
    print("=" * 68)
    print(title)
    print("=" * 68)


def main() -> None:
    head("1. Trie - one node per character")
    trie = Trie()
    for w in ROMAN:
        trie.insert(w)
    print("stored:", ", ".join(ROMAN))
    print("search('rubicon')   ->", trie.search("rubicon"))
    print("search('rubic')     ->", trie.search("rubic"), "(a prefix is not a word)")
    print("starts_with('rubic')->", trie.starts_with("rubic"))
    print("autocomplete('rom') ->", trie.autocomplete("rom"))
    print("autocomplete('ru')  ->", trie.autocomplete("ru"))
    print("nodes:", trie.count_nodes())

    head("2. Radix tree - collapse the single-child chains")
    radix = RadixTree()
    for w in ROMAN:
        radix.insert(w)
    print("edges (indented by depth, * = a stored word):")
    for line in radix.dump():
        print("   ", line)
    print("splits performed while inserting:", radix.n_splits)
    print("trie nodes: %d   radix nodes: %d   -> %.0f%% fewer"
          % (trie.count_nodes(), radix.count_nodes(),
             100 * (1 - radix.count_nodes() / trie.count_nodes())))
    assert sorted(radix.words()) == sorted(ROMAN)

    t2, r2 = Trie(), RadixTree()
    for w in WORDS:
        t2.insert(w)
        r2.insert(w)
    print()
    print("on a %d-word list: trie %d nodes, radix %d nodes -> %.0f%% fewer"
          % (len(WORDS), t2.count_nodes(), r2.count_nodes(),
             100 * (1 - r2.count_nodes() / t2.count_nodes())))
    print("autocomplete('int') ->", t2.autocomplete("int"))
    print("both agree on every stored word:", sorted(r2.words()) == sorted(WORDS))

    head("3. Longest prefix match - a routing table")
    table = [("0.0.0.0/0", "isp-default"), ("10.0.0.0/8", "corp-core"),
             ("10.20.0.0/16", "site-b"), ("10.20.30.0/24", "rack-7"),
             ("192.168.0.0/16", "lab")]
    router = BinaryTrieRouter()
    for cidr, hop in table:
        router.add(cidr, hop)
    for ip in ["10.20.30.5", "10.20.99.1", "10.99.0.1", "8.8.8.8"]:
        hop, plen, visited = router.lookup(ip)
        lin_hop, lin_touched = linear_lookup(table, ip)
        print("%-12s -> %-12s /%-2d  trie visited %2d nodes, "
              "linear scan touched %d rules   (agree: %s)"
              % (ip, hop, plen, visited, lin_touched, hop == lin_hop))
    print("the deepest marked node on the path wins, so rule order is irrelevant")
    print("the trie visits at most 33 nodes whether the table holds 5 routes or "
          "the ~1M a core router carries; the linear scan touches all of them")

    head("4. A token-level radix cache")
    page = 4
    cache = TokenRadixCache(page_size=page)
    system = tuple(range(1000, 1040))            # 40-token system prompt
    reqs = {
        "req-A": system + tuple(range(1, 13)),
        "req-B": system + tuple(range(50, 58)),
        "req-C": system + tuple(range(1, 13)) + tuple(range(70, 78)),
    }
    for name, toks in reqs.items():
        reused, allocated = cache.insert(toks)
        print("%s: %3d tokens -> %3d reused from cache, %3d newly allocated"
              % (name, len(toks), reused, allocated))
    print("splits so far:", cache.n_splits)
    total = sum(len(t) for t in reqs.values())
    print("across the three requests: %d tokens asked for, %d KV slots used "
          "(%.0f%% saved)" % (total, cache.allocated,
                              100 * (1 - cache.allocated / total)))

    print()
    print("a request that diverges *inside* a cached run forces a split:")
    reused, allocated = cache.insert(system[:20] + tuple(range(900, 912)))
    print("req-D: %3d reused, %3d allocated, splits now %d"
          % (reused, allocated, cache.n_splits))
    print("tree (each line is one node's token run):")
    for line in cache.dump():
        print("   ", line)

    print()
    print("page alignment truncates a match to a whole number of pages:")
    aligned = TokenRadixCache(page_size=page)
    aligned.insert(tuple(range(100, 116)))
    reused, _ = aligned.insert(tuple(range(100, 110)) + (999,))
    print("  10 tokens in common, page_size=%d -> %d reusable "
          "(the tail of a page cannot be shared)" % (page, reused))

    print()
    print("eviction is LRU over leaves only:")
    before = cache.size()
    freed = cache.evict(12)
    print("  freed %d tokens, tree %d -> %d; the shared prefix near the root "
          "outlives its children" % (freed, before, cache.size()))

    head("5. LeetCode")
    lc208 = Trie()
    for w in ["apple"]:
        lc208.insert(w)
    print("208: search('apple') =", lc208.search("apple"),
          " search('app') =", lc208.search("app"),
          " startsWith('app') =", lc208.starts_with("app"))
    lc208.insert("app")
    print("     after insert('app'), search('app') =", lc208.search("app"))

    wd = WordDictionary()
    for w in ["bad", "dad", "mad"]:
        wd.addWord(w)
    print("211: search('pad') =", wd.search("pad"),
          " search('bad') =", wd.search("bad"),
          " search('.ad') =", wd.search(".ad"),
          " search('b..') =", wd.search("b.."))

    print("648:", replace_words(["cat", "bat", "rat"],
                                "the cattle was rattled by the battery"))

    # ---- assertions -----------------------------------------------------
    assert trie.search("rubicon") and not trie.search("rubic")
    assert trie.autocomplete("rom") == ["romane", "romanus", "romulus"]
    assert radix.count_nodes() < trie.count_nodes()
    assert sorted(r2.words()) == sorted(WORDS)
    assert all(r2.search(w) for w in WORDS)
    assert not r2.search("inte") and r2.starts_with("inte")
    assert router.lookup("10.20.30.5")[0] == "rack-7"
    assert router.lookup("10.20.99.1")[0] == "site-b"
    assert router.lookup("8.8.8.8")[0] == "isp-default"
    for ip in ["10.20.30.5", "10.20.99.1", "10.99.0.1", "8.8.8.8"]:
        assert router.lookup(ip)[0] == linear_lookup(table, ip)[0]
    assert cache.allocated < total
    assert reused % page == 0
    assert wd.search(".ad") and not wd.search("pad")
    assert replace_words(["cat", "bat", "rat"],
                         "the cattle was rattled by the battery") == \
        "the cat was rat by the bat"
    print()
    print("all assertions passed")


if __name__ == "__main__":
    main()
