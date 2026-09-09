"""Day 28 - balanced trees: AVL, red-black and B-tree.

Run me:  python3 balanced_trees.py

Day 04 built a binary search tree and left one thing unsaid: nothing in
`insert` looks at the shape of the tree.  Feed it sorted keys - which is
exactly what a database import, a log replay or a range of ids does - and the
tree degenerates into a linked list.  Every operation is still "correct" and
every operation is O(n).

Three answers, and they are three different bargains, not three versions of
one idea:

  * **AVL** keeps the two subtree heights within 1.  The strictest of the
    three, so it is the shortest, so lookups are the cheapest - and it pays
    for that with the most rebalancing work on writes.
  * **Red-black** keeps a weaker invariant (no path is more than twice any
    other), so the tree is taller but insert and delete finish after at most
    a couple of rotations plus some recolouring.  This is why it is the one
    inside std::map, Java's TreeMap and the Linux scheduler.
  * **B-tree** changes the cost model instead of the invariant.  When a node
    is a disk page, the thing to minimise is not comparisons but *node
    visits*, so make each node hold hundreds of keys and the height collapses
    to three or four.  Every database index on earth is this.

Standard library only.
"""

from __future__ import annotations

import random
from typing import Dict, List, Optional, Tuple


# ===========================================================================
# 1. The problem: a plain BST has no shape guarantee
# ===========================================================================

class BSTNode:
    """The day 04 node, unchanged."""

    __slots__ = ('key', 'left', 'right')

    def __init__(self, key: int) -> None:
        self.key = key
        self.left: Optional['BSTNode'] = None
        self.right: Optional['BSTNode'] = None


def bst_insert(root: Optional[BSTNode], key: int) -> BSTNode:
    """Ordinary BST insert - notice that nothing here inspects the shape.

    Written iteratively on purpose: the recursive version blows Python's stack
    on exactly the input that motivates this whole day, a sorted key sequence.
    """
    if root is None:
        return BSTNode(key)
    cur = root
    while True:
        if key < cur.key:
            if cur.left is None:
                cur.left = BSTNode(key)
                return root
            cur = cur.left
        elif key > cur.key:
            if cur.right is None:
                cur.right = BSTNode(key)
                return root
            cur = cur.right
        else:
            return root


def height(node) -> int:
    """Height in edges; an empty tree is -1, a single node is 0.

    Iterative post-order, for the same stack reason as above.
    """
    if node is None:
        return -1
    depth = {id(None): -1}
    stack = [(node, False)]
    while stack:
        cur, ready = stack.pop()
        if cur is None:
            continue
        if ready:
            depth[id(cur)] = 1 + max(depth.get(id(cur.left), -1),
                                     depth.get(id(cur.right), -1))
        else:
            stack.append((cur, True))
            stack.append((cur.left, False))
            stack.append((cur.right, False))
    return depth[id(node)]


def size(node) -> int:
    n, stack = 0, [node]
    while stack:
        cur = stack.pop()
        if cur is not None:
            n += 1
            stack.append(cur.left)
            stack.append(cur.right)
    return n


def inorder(node) -> List[int]:
    """In-order walk - the sorted sequence every one of today's trees keeps."""
    out: List[int] = []
    stack, cur = [], node
    while stack or cur is not None:
        while cur is not None:
            stack.append(cur)
            cur = cur.left
        cur = stack.pop()
        out.append(cur.key)
        cur = cur.right
    return out


def search_cost(root, key: int) -> int:
    """Number of key comparisons a lookup performs - the metric that matters."""
    n = 0
    cur = root
    while cur is not None:
        n += 1
        if key == cur.key:
            return n
        cur = cur.left if key < cur.key else cur.right
    return n


def mean_search_cost(root, keys: List[int]) -> float:
    return sum(search_cost(root, k) for k in keys) / len(keys)


# ===========================================================================
# 2. Rotations - the one primitive all of today's trees share
# ===========================================================================
#
# A rotation moves a node up and its parent down while keeping the in-order
# sequence identical.  That last clause is the whole reason it is legal: the
# tree is still a BST afterwards, so a rotation can never change a lookup's
# answer, only its cost.
#
#        y                               x
#       / \      right rotate y         / \
#      x   C     ------------->        A   y
#     / \        <-------------           / \
#    A   B        left rotate x           B   C
#
#   in-order both sides:  A x B y C

def rotate_right(y: 'AVLNode') -> 'AVLNode':
    """y goes down, its left child x comes up.  B changes parent, nothing else."""
    x = y.left
    y.left = x.right          # B moves from x's right to y's left
    x.right = y
    update_height(y)          # y is now lower, so recompute it first
    update_height(x)
    return x                  # the new subtree root


def rotate_left(x: 'AVLNode') -> 'AVLNode':
    """Mirror image of rotate_right."""
    y = x.right
    x.right = y.left
    y.left = x
    update_height(x)
    update_height(y)
    return y


# ===========================================================================
# 3. AVL - keep the two subtree heights within one
# ===========================================================================

class AVLNode:
    """Every node caches its own height, so balance is an O(1) test."""

    __slots__ = ('key', 'left', 'right', 'h')

    def __init__(self, key: int) -> None:
        self.key = key
        self.left: Optional['AVLNode'] = None
        self.right: Optional['AVLNode'] = None
        self.h = 0            # height in edges of the subtree rooted here


def node_height(n: Optional[AVLNode]) -> int:
    return -1 if n is None else n.h


def update_height(n: AVLNode) -> None:
    n.h = 1 + max(node_height(n.left), node_height(n.right))


def balance_factor(n: AVLNode) -> int:
    """left height minus right height; AVL requires this to stay in {-1,0,1}."""
    return node_height(n.left) - node_height(n.right)


def avl_rebalance(n: AVLNode, stats: Optional[Dict[str, int]] = None) -> AVLNode:
    """Restore |balance_factor| <= 1 at n, assuming both children are AVL.

    There are exactly four shapes, and the two "outside" ones need a single
    rotation while the two "inside" ones need two.  The tell is the sign of
    the *child's* balance factor: if it leans the same way as the parent the
    node is on the outside, if it leans the other way it is a zig-zag and one
    rotation would only move the problem to the other side.
    """
    update_height(n)
    bf = balance_factor(n)

    if bf > 1:                                    # left heavy
        if balance_factor(n.left) < 0:            # LR: zig-zag
            n.left = rotate_left(n.left)
            _bump(stats, 'double')
        else:                                     # LL: straight line
            _bump(stats, 'single')
        return rotate_right(n)

    if bf < -1:                                   # right heavy
        if balance_factor(n.right) > 0:           # RL: zig-zag
            n.right = rotate_right(n.right)
            _bump(stats, 'double')
        else:                                     # RR
            _bump(stats, 'single')
        return rotate_left(n)

    return n


def _bump(stats: Optional[Dict[str, int]], key: str, by: int = 1) -> None:
    if stats is not None:
        stats[key] = stats.get(key, 0) + by


def avl_insert(root: Optional[AVLNode], key: int,
               stats: Optional[Dict[str, int]] = None) -> AVLNode:
    """Insert, then rebalance every node on the way back up.

    Written with an explicit stack rather than recursion so that inserting a
    million sorted keys does not hit the recursion limit.  The important part
    is the shape of the loop, not the stack: descend, insert, and repair on
    the way back to the root.
    """
    if root is None:
        return AVLNode(key)

    path: List[Tuple[AVLNode, bool]] = []         # (node, went_left)
    cur: Optional[AVLNode] = root
    while cur is not None:
        if key < cur.key:
            path.append((cur, True))
            cur = cur.left
        elif key > cur.key:
            path.append((cur, False))
            cur = cur.right
        else:
            return root                           # duplicate: nothing to do

    child: AVLNode = AVLNode(key)
    while path:
        parent, went_left = path.pop()
        if went_left:
            parent.left = child
        else:
            parent.right = child
        child = avl_rebalance(parent, stats)
    return child


def avl_min(n: AVLNode) -> AVLNode:
    while n.left is not None:
        n = n.left
    return n


def avl_delete(root: Optional[AVLNode], key: int,
               stats: Optional[Dict[str, int]] = None) -> Optional[AVLNode]:
    """Delete is where AVL and red-black really diverge.

    A single insert needs at most one (single or double) rotation, because
    fixing the lowest offender also restores the subtree's original height.
    A delete has no such luck: the subtree gets *shorter*, so its parent can
    go out of balance too, and the repairs can cascade all the way to the
    root - O(log n) rotations rather than O(1).
    """
    if root is None:
        return None
    if key < root.key:
        root.left = avl_delete(root.left, key, stats)
    elif key > root.key:
        root.right = avl_delete(root.right, key, stats)
    else:
        if root.left is None:
            return root.right
        if root.right is None:
            return root.left
        succ = avl_min(root.right)                # in-order successor
        root.key = succ.key
        root.right = avl_delete(root.right, succ.key, stats)
    return avl_rebalance(root, stats)


def avl_check(n: Optional[AVLNode],
              lo: float = float('-inf'), hi: float = float('inf')) -> int:
    """Assert BST order, cached heights and the balance invariant; return height."""
    if n is None:
        return -1
    assert lo < n.key < hi, 'BST order violated at %r' % n.key
    lh = avl_check(n.left, lo, n.key)
    rh = avl_check(n.right, n.key, hi)
    assert n.h == 1 + max(lh, rh), 'stale cached height at %r' % n.key
    assert abs(lh - rh) <= 1, 'AVL balance violated at %r' % n.key
    return n.h


# ===========================================================================
# 4. Red-black - a weaker invariant, bought with a colour bit
# ===========================================================================
#
# The five properties, and only the last two do any work:
#
#   1. every node is red or black
#   2. the root is black
#   3. every leaf (the NIL sentinel) is black
#   4. a red node's children are both black    <- no two reds in a row
#   5. every path from a node down to a NIL has the same number of blacks
#
# Property 5 fixes a "black height"; property 4 says reds can at most double
# a path relative to the all-black one.  So the longest root-to-leaf path is
# at most twice the shortest, giving height <= 2*log2(n+1).  That is looser
# than AVL's 1.44*log2(n), and the looseness is exactly what buys cheap
# writes: an insert needs at most two rotations, a delete at most three, no
# matter how large the tree is.

RED, BLACK = 'R', 'B'


class RBNode:
    __slots__ = ('key', 'left', 'right', 'parent', 'color')

    def __init__(self, key, color=RED) -> None:
        self.key = key
        self.left: 'RBNode' = NIL
        self.right: 'RBNode' = NIL
        self.parent: 'RBNode' = NIL
        self.color = color


# One shared sentinel for every leaf, so `x.left.color` is always safe - and
# so the fixup code never needs a None check.  It has to be built without
# calling __init__, because __init__ itself wants to point at NIL.
NIL = RBNode.__new__(RBNode)
NIL.key = None
NIL.color = BLACK
NIL.left = NIL.right = NIL.parent = NIL


class RedBlackTree:
    """CLRS red-black tree, instrumented to count rotations and recolourings."""

    def __init__(self) -> None:
        self.root: RBNode = NIL
        self.stats: Dict[str, int] = {'rotations': 0, 'recolors': 0}

    # ---- rotations (same primitive as AVL, but with parent pointers) ------

    def _rotate_left(self, x: RBNode) -> None:
        y = x.right
        x.right = y.left
        if y.left is not NIL:
            y.left.parent = x
        y.parent = x.parent
        if x.parent is NIL:
            self.root = y
        elif x is x.parent.left:
            x.parent.left = y
        else:
            x.parent.right = y
        y.left = x
        x.parent = y
        self.stats['rotations'] += 1

    def _rotate_right(self, y: RBNode) -> None:
        x = y.left
        y.left = x.right
        if x.right is not NIL:
            x.right.parent = y
        x.parent = y.parent
        if y.parent is NIL:
            self.root = x
        elif y is y.parent.right:
            y.parent.right = x
        else:
            y.parent.left = x
        x.right = y
        y.parent = x
        self.stats['rotations'] += 1

    # ---- insert ----------------------------------------------------------

    def insert(self, key) -> None:
        """Insert red, then repair the only property a red insert can break."""
        parent, cur = NIL, self.root
        while cur is not NIL:
            parent = cur
            if key < cur.key:
                cur = cur.left
            elif key > cur.key:
                cur = cur.right
            else:
                return
        z = RBNode(key, RED)
        z.parent = parent
        if parent is NIL:
            self.root = z
        elif key < parent.key:
            parent.left = z
        else:
            parent.right = z
        self._insert_fixup(z)

    def _insert_fixup(self, z: RBNode) -> None:
        """A new node is red, so property 5 is safe and only 4 can break.

        Three cases, and the first one is the reason writes are cheap:

          case 1  the uncle is red     -> recolour parent, uncle and
                  grandparent, then move the problem two levels up.
                  No rotation at all.  This is the common case.
          case 2  uncle black, z is
                  the "inside" child   -> one rotation to turn it into case 3
          case 3  uncle black, z is
                  the "outside" child  -> recolour and one rotation, and the
                  loop *ends* - which is why an insert never does more than
                  two rotations however far the recolouring travelled.
        """
        while z.parent.color == RED:
            gp = z.parent.parent
            if z.parent is gp.left:
                uncle = gp.right
                if uncle.color == RED:                       # case 1
                    z.parent.color = BLACK
                    uncle.color = BLACK
                    gp.color = RED
                    self.stats['recolors'] += 3
                    z = gp
                else:
                    if z is z.parent.right:                  # case 2
                        z = z.parent
                        self._rotate_left(z)
                    z.parent.color = BLACK                   # case 3
                    z.parent.parent.color = RED
                    self.stats['recolors'] += 2
                    self._rotate_right(z.parent.parent)
            else:                                            # mirror image
                uncle = gp.left
                if uncle.color == RED:
                    z.parent.color = BLACK
                    uncle.color = BLACK
                    gp.color = RED
                    self.stats['recolors'] += 3
                    z = gp
                else:
                    if z is z.parent.left:
                        z = z.parent
                        self._rotate_right(z)
                    z.parent.color = BLACK
                    z.parent.parent.color = RED
                    self.stats['recolors'] += 2
                    self._rotate_left(z.parent.parent)
        if self.root.color != BLACK:
            self.root.color = BLACK                          # property 2
            self.stats['recolors'] += 1

    # ---- delete ----------------------------------------------------------

    def _transplant(self, u: RBNode, v: RBNode) -> None:
        if u.parent is NIL:
            self.root = v
        elif u is u.parent.left:
            u.parent.left = v
        else:
            u.parent.right = v
        v.parent = u.parent

    def _minimum(self, x: RBNode) -> RBNode:
        while x.left is not NIL:
            x = x.left
        return x

    def _find(self, key) -> RBNode:
        cur = self.root
        while cur is not NIL and cur.key != key:
            cur = cur.left if key < cur.key else cur.right
        return cur

    def delete(self, key) -> None:
        """CLRS delete: splice out a node, then repair the black height.

        The bound that matters is in `_delete_fixup`: it performs at most
        three rotations however tall the tree is, because case 2 - the one
        that loops - only ever recolours.
        """
        z = self._find(key)
        if z is NIL:
            return
        y = z
        y_original_color = y.color
        if z.left is NIL:
            x = z.right
            self._transplant(z, z.right)
        elif z.right is NIL:
            x = z.left
            self._transplant(z, z.left)
        else:
            y = self._minimum(z.right)
            y_original_color = y.color
            x = y.right
            if y.parent is z:
                x.parent = y
            else:
                self._transplant(y, y.right)
                y.right = z.right
                y.right.parent = y
            self._transplant(z, y)
            y.left = z.left
            y.left.parent = y
            y.color = z.color
            self.stats['recolors'] += 1
        if y_original_color == BLACK:
            self._delete_fixup(x)

    def _delete_fixup(self, x: RBNode) -> None:
        """x carries an 'extra black' that has to be pushed away or absorbed.

          case 1  red sibling      -> one rotation, turns into 2/3/4
          case 2  black sibling,
                  black nephews    -> recolour only, move the extra black up
                                      (this is the loop, and it is free)
          case 3  black sibling,
                  near nephew red  -> one rotation, turns into case 4
          case 4  far nephew red   -> one rotation and the loop *terminates*

        Cases 1, 3 and 4 each happen at most once per delete, so a delete
        does at most three rotations - regardless of n.
        """
        while x is not self.root and x.color == BLACK:
            if x is x.parent.left:
                w = x.parent.right
                if w.color == RED:                          # case 1
                    w.color = BLACK
                    x.parent.color = RED
                    self.stats['recolors'] += 2
                    self._rotate_left(x.parent)
                    w = x.parent.right
                if w.left.color == BLACK and w.right.color == BLACK:
                    w.color = RED                           # case 2
                    self.stats['recolors'] += 1
                    x = x.parent
                else:
                    if w.right.color == BLACK:              # case 3
                        w.left.color = BLACK
                        w.color = RED
                        self.stats['recolors'] += 2
                        self._rotate_right(w)
                        w = x.parent.right
                    w.color = x.parent.color                # case 4
                    x.parent.color = BLACK
                    w.right.color = BLACK
                    self.stats['recolors'] += 3
                    self._rotate_left(x.parent)
                    x = self.root
            else:                                           # mirror image
                w = x.parent.left
                if w.color == RED:
                    w.color = BLACK
                    x.parent.color = RED
                    self.stats['recolors'] += 2
                    self._rotate_right(x.parent)
                    w = x.parent.left
                if w.right.color == BLACK and w.left.color == BLACK:
                    w.color = RED
                    self.stats['recolors'] += 1
                    x = x.parent
                else:
                    if w.left.color == BLACK:
                        w.right.color = BLACK
                        w.color = RED
                        self.stats['recolors'] += 2
                        self._rotate_left(w)
                        w = x.parent.left
                    w.color = x.parent.color
                    x.parent.color = BLACK
                    w.left.color = BLACK
                    self.stats['recolors'] += 3
                    self._rotate_right(x.parent)
                    x = self.root
        if x.color != BLACK:
            x.color = BLACK
            self.stats['recolors'] += 1

    # ---- queries ---------------------------------------------------------

    def search_cost(self, key) -> int:
        n, cur = 0, self.root
        while cur is not NIL:
            n += 1
            if key == cur.key:
                return n
            cur = cur.left if key < cur.key else cur.right
        return n

    def inorder(self) -> List:
        out, stack, cur = [], [], self.root
        while stack or cur is not NIL:
            while cur is not NIL:
                stack.append(cur)
                cur = cur.left
            cur = stack.pop()
            out.append(cur.key)
            cur = cur.right
        return out

    def height(self) -> int:
        """Height in edges, counting real nodes only."""
        best, stack = -1, [(self.root, 0)]
        while stack:
            node, d = stack.pop()
            if node is NIL:
                best = max(best, d - 1)
                continue
            stack.append((node.left, d + 1))
            stack.append((node.right, d + 1))
        return best

    def black_height(self) -> int:
        n, cur = 0, self.root
        while cur is not NIL:
            if cur.color == BLACK:
                n += 1
            cur = cur.left
        return n

    def check(self) -> int:
        """Verify all five properties; return the black height."""
        assert self.root.color == BLACK, 'property 2: root must be black'
        assert NIL.color == BLACK, 'property 3: the sentinel must be black'

        def walk(n, lo, hi):
            if n is NIL:
                return 1
            assert lo < n.key < hi, 'BST order violated at %r' % n.key
            if n.color == RED:
                assert n.left.color == BLACK and n.right.color == BLACK, \
                    'property 4: red node %r has a red child' % n.key
            lb = walk(n.left, lo, n.key)
            rb = walk(n.right, n.key, hi)
            assert lb == rb, 'property 5: black height differs under %r' % n.key
            return lb + (1 if n.color == BLACK else 0)

        return walk(self.root, float('-inf'), float('inf'))


# ===========================================================================
# 5. B-tree - keep the invariant, change the cost model
# ===========================================================================
#
# AVL and red-black both minimise comparisons, because they assume a node
# costs the same as a comparison.  On disk - or on any storage with a page
# size - that assumption is wrong by four orders of magnitude: reading a node
# is a 4 KB page fetch, and comparing two integers inside a page you already
# hold is free.  So the quantity to minimise is *node visits*, i.e. height.
#
# A B-tree of minimum degree t holds between t-1 and 2t-1 keys per node, so
# its height is about log_t(n) rather than log_2(n).  With 4 KB pages and
# 16-byte (key, pointer) entries, t is around 128, and a million keys fit in
# a tree of height 2.  The invariant is the strongest of the three: *all*
# leaves are at the same depth, which is achievable only because a node can
# absorb a new key without growing taller.

class BTreeNode:
    __slots__ = ('keys', 'children', 'leaf')

    def __init__(self, leaf: bool = True) -> None:
        self.keys: List[int] = []
        self.children: List['BTreeNode'] = []
        self.leaf = leaf


class BTree:
    """A B-tree with minimum degree t, counting the node visits it makes.

    `visits` is the interesting counter: on real storage it is the number of
    page reads, and it is what the whole design is optimising.
    """

    def __init__(self, t: int = 3) -> None:
        assert t >= 2, 'minimum degree must be at least 2'
        self.t = t
        self.root = BTreeNode(leaf=True)
        self.visits = 0
        self.splits = 0

    # ---- search ----------------------------------------------------------

    def search(self, key: int) -> bool:
        node, self.visits = self.root, 0
        while True:
            self.visits += 1
            i = 0
            while i < len(node.keys) and key > node.keys[i]:
                i += 1
            if i < len(node.keys) and key == node.keys[i]:
                return True
            if node.leaf:
                return False
            node = node.children[i]

    # ---- insert ----------------------------------------------------------

    def insert(self, key: int) -> None:
        """Split full nodes on the way *down*, so the recursion never backs up.

        The trick that makes B-trees simple: before descending into a child,
        make sure it is not full.  Then a leaf insert can always succeed, and
        the tree only ever grows taller at the root - which is why every leaf
        stays at the same depth.
        """
        root = self.root
        if len(root.keys) == 2 * self.t - 1:
            new_root = BTreeNode(leaf=False)
            new_root.children.append(root)
            self._split_child(new_root, 0)
            self.root = new_root
            root = new_root
        self._insert_nonfull(root, key)

    def _split_child(self, parent: BTreeNode, i: int) -> None:
        """Split parent.children[i], pushing its median key up into parent."""
        t = self.t
        full = parent.children[i]
        right = BTreeNode(leaf=full.leaf)
        median = full.keys[t - 1]

        right.keys = full.keys[t:]
        full.keys = full.keys[:t - 1]
        if not full.leaf:
            right.children = full.children[t:]
            full.children = full.children[:t]

        parent.keys.insert(i, median)
        parent.children.insert(i + 1, right)
        self.splits += 1

    def _insert_nonfull(self, node: BTreeNode, key: int) -> None:
        i = len(node.keys) - 1
        if node.leaf:
            node.keys.append(None)
            while i >= 0 and key < node.keys[i]:
                node.keys[i + 1] = node.keys[i]
                i -= 1
            node.keys[i + 1] = key
            return
        while i >= 0 and key < node.keys[i]:
            i -= 1
        i += 1
        if len(node.children[i].keys) == 2 * self.t - 1:
            self._split_child(node, i)
            if key > node.keys[i]:
                i += 1
        self._insert_nonfull(node.children[i], key)

    # ---- queries ---------------------------------------------------------

    def height(self) -> int:
        h, node = 0, self.root
        while not node.leaf:
            node = node.children[0]
            h += 1
        return h

    def node_count(self) -> int:
        n, stack = 0, [self.root]
        while stack:
            node = stack.pop()
            n += 1
            stack.extend(node.children)
        return n

    def inorder(self) -> List[int]:
        out: List[int] = []

        def walk(node: BTreeNode) -> None:
            for i, k in enumerate(node.keys):
                if not node.leaf:
                    walk(node.children[i])
                out.append(k)
            if not node.leaf:
                walk(node.children[-1])

        walk(self.root)
        return out

    def check(self) -> None:
        """Every leaf at the same depth, every node within its key bounds."""
        t, depths = self.t, set()

        def walk(node: BTreeNode, depth: int, lo: float, hi: float) -> None:
            assert node.keys == sorted(node.keys), 'keys out of order'
            assert all(lo < k < hi for k in node.keys), 'key outside subtree range'
            assert len(node.keys) <= 2 * t - 1, 'node overfull'
            if node is not self.root:
                assert len(node.keys) >= t - 1, 'node underfull'
            if node.leaf:
                depths.add(depth)
                assert not node.children
                return
            assert len(node.children) == len(node.keys) + 1, 'arity mismatch'
            bounds = [lo] + node.keys + [hi]
            for i, child in enumerate(node.children):
                walk(child, depth + 1, bounds[i], bounds[i + 1])

        walk(self.root, 0, float('-inf'), float('inf'))
        assert len(depths) == 1, 'leaves at differing depths: %s' % sorted(depths)


def degree_for_page(page_bytes: int = 4096, entry_bytes: int = 16) -> int:
    """How many keys fit in one page, expressed as a minimum degree t.

    A node of degree t holds up to 2t-1 keys and 2t child pointers, so an
    entry is one key plus one pointer and 2t entries fill the page.
    """
    return max(2, page_bytes // entry_bytes // 2)


# ===========================================================================
# 6. LeetCode
# ===========================================================================

def is_balanced(root: Optional[BSTNode]) -> bool:
    """LC 110 - Balanced Binary Tree, height-balanced in the AVL sense.

    The naive solution calls `height` at every node and is O(n log n) at best
    and O(n^2) on a degenerate tree.  Returning the height *and* the verdict
    from one post-order walk makes it O(n): -1 is the sentinel for "already
    unbalanced somewhere below", and it short-circuits everything above it.
    """

    def check(n) -> int:
        if n is None:
            return -1
        lh = check(n.left)
        if lh == -2:
            return -2
        rh = check(n.right)
        if rh == -2 or abs(lh - rh) > 1:
            return -2
        return 1 + max(lh, rh)

    return check(root) != -2


def sorted_array_to_bst(nums: List[int]) -> Optional[BSTNode]:
    """LC 108 - build a height-balanced BST from a sorted array.

    Picking the middle element as the root is the whole algorithm, and it is
    the cheapest balancing there is: O(n), no rotations, because the input
    already told us the answer.
    """
    def build(lo: int, hi: int) -> Optional[BSTNode]:
        if lo > hi:
            return None
        mid = (lo + hi) // 2
        node = BSTNode(nums[mid])
        node.left = build(lo, mid - 1)
        node.right = build(mid + 1, hi)
        return node

    return build(0, len(nums) - 1)


def balance_bst(root: Optional[BSTNode]) -> Optional[BSTNode]:
    """LC 1382 - Balance a Binary Search Tree.

    In-order walk to get the sorted keys, then LC 108.  O(n) time and O(n)
    space, and worth contrasting with the in-place O(1)-space alternative,
    Day-Stout-Warren: flatten the tree into a right-leaning vine with
    rotations, then rotate every other node up log2(n) times.  DSW is what a
    library uses when it cannot afford the array; the array version is what
    you should write in an interview unless asked otherwise.
    """
    return sorted_array_to_bst(inorder(root))


# ===========================================================================
# 7. Measurement harness
# ===========================================================================

def build_all(keys: List[int]) -> Dict[str, object]:
    """Insert the same keys into both balanced trees and record what each did."""
    avl_stats: Dict[str, int] = {'single': 0, 'double': 0}
    avl = None
    for k in keys:
        avl = avl_insert(avl, k, avl_stats)

    rb = RedBlackTree()
    for k in keys:
        rb.insert(k)

    return {'avl': avl, 'avl_stats': avl_stats, 'rb': rb}


def report(keys: List[int], probes: List[int], label: str) -> Dict[str, object]:
    built = build_all(keys)
    avl, rb = built['avl'], built['rb']
    st = built['avl_stats']
    rot = st['single'] + 2 * st['double']

    print('  %s (n = %d)' % (label, len(keys)))
    print('    AVL          height %5d   mean lookup %8.2f   rotations %d'
          ' (%d single + %d double)'
          % (height(avl), mean_search_cost(avl, probes), rot,
             st['single'], st['double']))
    print('    red-black    height %5d   mean lookup %8.2f   rotations %d'
          '   recolourings %d'
          % (rb.height(),
             sum(rb.search_cost(k) for k in probes) / len(probes),
             rb.stats['rotations'], rb.stats['recolors']))
    return built


def show(node, indent: str = '', side: str = '') -> None:
    """Tiny pretty-printer for the worked examples (AVL nodes)."""
    if node is None:
        return
    show(node.right, indent + '      ', '/')
    print('%s%s%s' % (indent, side, node.key))
    show(node.left, indent + '      ', '\\')


def main() -> None:
    random.seed(28)

    print('=' * 68)
    print('1. Why balance at all: the same keys, two orders')
    print('=' * 68)
    n = 2047
    sorted_keys = list(range(1, n + 1))
    shuffled = sorted_keys[:]
    random.shuffle(shuffled)

    a = b = None
    for k in sorted_keys:
        a = bst_insert(a, k)
    for k in shuffled:
        b = bst_insert(b, k)
    print('  sorted insert   : height %4d   mean lookup %7.2f comparisons'
          % (height(a), mean_search_cost(a, sorted_keys)))
    print('  shuffled insert : height %4d   mean lookup %7.2f comparisons'
          % (height(b), mean_search_cost(b, sorted_keys)))
    print('  perfectly balanced would be height %d, and log2(%d) = %.1f'
          % (height(sorted_array_to_bst(sorted_keys)), n, (n + 1).bit_length() - 1))
    print('  the sorted case is the common one: ids, timestamps, a restored dump.')

    print()
    print('=' * 68)
    print('2. AVL: four shapes, two of them need a double rotation')
    print('=' * 68)
    cases = [('LL  (left-left)   ', [30, 20, 10]),
             ('RR  (right-right) ', [10, 20, 30]),
             ('LR  (left-right)  ', [30, 10, 20]),
             ('RL  (right-left)  ', [10, 30, 20])]
    for name, ks in cases:
        st: Dict[str, int] = {}
        t = None
        for k in ks:
            t = avl_insert(t, k, st)
        kind = 'double' if st.get('double') else 'single'
        print('  %s insert %-14s -> root %d, %s rotation, height %d'
              % (name, str(ks), t.key, kind, height(t)))
    print('  in every case the in-order walk is unchanged - a rotation cannot')
    print('  change what the tree *contains*, only how deep it is.')

    print()
    print('  the same thing on a real insert: 1..7 in order')
    t = None
    st = {'single': 0, 'double': 0}
    for k in range(1, 8):
        t = avl_insert(t, k, st)
    show(t)
    print('  %d rotations turned a 7-node chain into a perfect tree of height %d'
          % (st['single'] + st['double'], height(t)))

    print()
    print('=' * 68)
    print('3. AVL vs red-black: shorter tree, or cheaper writes')
    print('=' * 68)
    n = 100_000
    keys_sorted = list(range(n))
    keys_random = keys_sorted[:]
    random.shuffle(keys_random)
    probes = random.sample(keys_sorted, 2000)
    report(keys_sorted, probes, 'sorted keys  ')
    print()
    report(keys_random, probes, 'shuffled keys')
    print()
    print('  (the plain BST is left out here on purpose: on the sorted keys it')
    print('   is a 100000-node chain, and timing it would take longer than the')
    print('   rest of this script put together.)')
    print()
    print('  AVL is never taller, and on the sorted keys it is half the height,')
    print('  so every lookup is cheaper.  Note what red-black spends instead:')
    print('  five recolourings for every rotation on the sorted input - it')
    print('  moves colour, which is a bit in a node you already hold, rather')
    print('  than pointers.  Read-mostly index -> AVL.  General-purpose ordered')
    print('  map -> red-black, which is what std::map, Java TreeMap and the')
    print('  Linux CFS runqueue all use, and section 4 is the reason why.')

    print()
    print('=' * 68)
    print('4. Deletion is where the two invariants really differ')
    print('=' * 68)
    keys = list(range(4095))
    random.shuffle(keys)

    avl = None
    ins_events, ins_max = 0, 0
    for k in keys:
        st = {'single': 0, 'double': 0}
        avl = avl_insert(avl, k, st)
        r = st['single'] + 2 * st['double']
        ins_events += r
        ins_max = max(ins_max, r)

    victims = random.sample(keys, 2000)
    del_events, del_max = 0, 0
    for k in victims:
        st = {'single': 0, 'double': 0}
        avl = avl_delete(avl, k, st)
        r = st['single'] + 2 * st['double']
        del_events += r
        del_max = max(del_max, r)
    avl_check(avl)

    rb = RedBlackTree()
    rb_ins_max = 0
    for k in keys:
        before = rb.stats['rotations']
        rb.insert(k)
        rb_ins_max = max(rb_ins_max, rb.stats['rotations'] - before)
    rb_ins_total = rb.stats['rotations']
    rb_del_max, rb_del_total = 0, 0
    for k in victims:
        before = rb.stats['rotations']
        rb.delete(k)
        d = rb.stats['rotations'] - before
        rb_del_total += d
        rb_del_max = max(rb_del_max, d)
    rb.check()
    assert rb.inorder() == inorder(avl), 'the two trees disagree after deletion'

    print('  same 4095 random inserts, then the same 2000 random deletes')
    print('    AVL        insert: %5d rotations, worst single insert %d'
          % (ins_events, ins_max))
    print('               delete: %5d rotations, worst single delete %d'
          % (del_events, del_max))
    print('    red-black  insert: %5d rotations, worst single insert %d'
          % (rb_ins_total, rb_ins_max))
    print('               delete: %5d rotations, worst single delete %d'
          % (rb_del_total, rb_del_max))
    print()
    print('  the averages are close - it is the *worst case per operation*')
    print('  that separates them.  An AVL insert stops after one rebalance,')
    print('  because the rotation restores the subtree to its original height;')
    print('  an AVL delete makes a subtree shorter, so the repair can cascade')
    print('  to the root.  Red-black is bounded at both ends: at most 2')
    print('  rotations per insert and 3 per delete, because the case that')
    print('  loops only ever changes colours.  That bound, not the average,')
    print('  is why latency-sensitive kernels pick red-black.')

    print()
    print('=' * 68)
    print('5. B-tree: minimise node visits, not comparisons')
    print('=' * 68)
    small = BTree(t=2)
    for k in range(1, 11):
        small.insert(k)
    small.check()
    print('  t = 2 (a 2-3-4 tree), keys 1..10 inserted in order')
    print('    root keys %s, height %d, %d nodes, %d splits'
          % (small.root.keys, small.height(), small.node_count(), small.splits))
    print('    every leaf is at depth %d - the tree grew at the *root*, not'
          ' at the leaves' % small.height())

    t_page = degree_for_page(4096, 16)
    n = 1_000_000
    big = BTree(t=t_page)
    for k in range(n):
        big.insert(k)
    big.search(n // 2)
    print()
    print('  4 KB page, 16-byte entries -> minimum degree t = %d'
          ' (up to %d keys per node)' % (t_page, 2 * t_page - 1))
    print('    %d keys: height %d, %d nodes, one lookup touches %d of them'
          % (n, big.height(), big.node_count(), big.visits))
    avl_h = 0
    m = 1
    while m <= n:                      # AVL height bound 1.44*log2(n+2)-0.33
        m *= 2
        avl_h += 1
    print('    a balanced binary tree of the same %d keys is about %d levels'
          ' deep,' % (n, avl_h))
    print('    so it would be ~%d page reads instead of %d.  Same big-O,'
          % (avl_h, big.visits))
    print('    %.0fx fewer round trips - which is the entire reason every'
          % (avl_h / big.visits))
    print('    database index, filesystem and key-value store is a B-tree.')

    print()
    print('=' * 68)
    print('6. LeetCode')
    print('=' * 68)
    chain = None
    for k in [1, 2, 3, 4]:
        chain = bst_insert(chain, k)
    print('  110  is_balanced(chain 1-2-3-4)          = %s'
          % is_balanced(chain))
    print('  110  is_balanced(sorted_array_to_bst)    = %s'
          % is_balanced(sorted_array_to_bst(list(range(15)))))
    print('  108  sorted_array_to_bst(0..14) height   = %d (optimal)'
          % height(sorted_array_to_bst(list(range(15)))))
    print('  1382 balance_bst(chain of 63) height     = %d -> %d'
          % (height(_chain(63)), height(balance_bst(_chain(63)))))
    print('       and the keys are untouched: %s'
          % (inorder(balance_bst(_chain(63))) == list(range(63))))

    # ---- assertions --------------------------------------------------------
    keys = random.sample(range(100_000), 5000)
    avl = None
    for k in keys:
        avl = avl_insert(avl, k)
    avl_check(avl)
    assert inorder(avl) == sorted(keys)

    rb = RedBlackTree()
    for k in keys:
        rb.insert(k)
    rb.check()
    assert rb.inorder() == sorted(keys)
    assert rb.height() <= 2 * ((len(keys) + 1).bit_length()), \
        'red-black height bound violated'

    bt = BTree(t=3)
    for k in keys:
        bt.insert(k)
    bt.check()
    assert bt.inorder() == sorted(keys)
    assert all(bt.search(k) for k in keys[:200])
    assert not bt.search(-1)

    # a rotation must never change the in-order sequence
    node = None
    for k in [50, 25, 75, 10, 30]:
        node = avl_insert(node, k)
    before = inorder(node)
    assert inorder(rotate_right(node)) == before
    node = None
    for k in [50, 25, 75, 10, 30]:
        node = avl_insert(node, k)
    assert inorder(rotate_left(node)) == before

    assert is_balanced(sorted_array_to_bst(list(range(1000))))
    assert not is_balanced(_chain(10))
    assert inorder(balance_bst(_chain(100))) == list(range(100))
    assert height(balance_bst(_chain(100))) == 6
    assert degree_for_page(4096, 16) == 128

    # the two published per-operation bounds, checked rather than quoted
    rb2 = RedBlackTree()
    worst_ins = 0
    ks = random.sample(range(50_000), 3000)
    for k in ks:
        before = rb2.stats['rotations']
        rb2.insert(k)
        worst_ins = max(worst_ins, rb2.stats['rotations'] - before)
    worst_del = 0
    for k in random.sample(ks, 1500):
        before = rb2.stats['rotations']
        rb2.delete(k)
        worst_del = max(worst_del, rb2.stats['rotations'] - before)
    rb2.check()
    assert worst_ins <= 2, 'red-black insert did %d rotations' % worst_ins
    assert worst_del <= 3, 'red-black delete did %d rotations' % worst_del
    assert min_distance_free()

    print()
    print('all assertions passed')


def _chain(n: int) -> Optional[BSTNode]:
    """A degenerate BST: 0..n-1 inserted in order, i.e. a linked list."""
    root = None
    for k in range(n):
        root = bst_insert(root, k)
    return root


def min_distance_free() -> bool:
    """Sanity check that all three structures agree on membership."""
    keys = list(range(200))
    random.shuffle(keys)
    avl = None
    for k in keys:
        avl = avl_insert(avl, k)
    rb = RedBlackTree()
    for k in keys:
        rb.insert(k)
    bt = BTree(t=4)
    for k in keys:
        bt.insert(k)
    return (inorder(avl) == rb.inorder() == bt.inorder() == list(range(200)))


if __name__ == '__main__':
    main()
