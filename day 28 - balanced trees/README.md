# Balanced trees: AVL, red-black and the B-tree

More details in:
https://medium.com/100-days-of-python

Interactive walkthrough: [demo.html](demo.html) - a single self-contained page
(no build step, works offline, 中文 / English toggle) that degenerates a BST into a linked list,
isolates the rotation and shows that it changes the height without touching the in-order
sequence, replays AVL rebalancing (including the LR case where one rotation is not enough),
steps through red-black insert fixup case by case with the colours drawn, splits a B-tree with
`t = 2` and watches the median move up into the parent, and finishes with LeetCode 110 and 1382.

![img](https://github.com/KingRei/100DaysPython/blob/master/day%2028%20-%20balanced%20trees/imgs/day28_1.png?raw=true)

A binary search tree gives `O(log n)` lookups only if it happens to be short, and nothing in the
insert procedure enforces that. Insert `1, 2, 3, …` in order and every key goes to the right of
the previous one: the tree is a linked list with extra pointers. On 2047 keys the difference is
not subtle - sorted input gives height 2046 and 1024 comparisons for an average lookup, a
shuffled order gives height 22 and 12.62 comparisons, and a perfectly balanced build gives
height 10 and exactly 10.0. The random case is fine on average, which is precisely the problem:
the input that breaks it - already-sorted data - is the most common input there is.

A balanced tree is a BST plus an **invariant** that is re-established after every write. All
three structures here do the same three things: keep the search property, define a rule that
bounds the height, and repair the rule locally after each insert or delete. They differ only in
which rule they pick, and every difference downstream follows from that choice.

## The rotation is the whole toolkit

```python
def rotate_right(y):
    x = y.left
    y.left = x.right      # B changes parent - the only subtree that moves
    x.right = y
    update_height(y)      # y is now the lower node: update it FIRST
    update_height(x)
    return x
```

Read the shape as `A x B y C` in order. Before the rotation `y` is the root with `x` and `A`,
`B` beneath it; afterwards `x` is the root with `y`, `B`, `C` beneath it - and the in-order
sequence is still `A x B y C`. That is the point: **a rotation changes the height but not the
sorted order**, so the search property survives for free and the depths on one side drop by one
while the other side gains one. Only one subtree, `B`, changes parent, so the whole operation
is three pointer writes and `O(1)` time.

The one detail that is easy to get wrong is the order of the two `update_height` calls. `y` ends
up *below* `x`, so `y`'s height must be recomputed before `x` reads it; swap the lines and every
height above the rotation is silently wrong, which produces a tree that is still a valid BST and
quietly stops being balanced.

![img](https://github.com/KingRei/100DaysPython/blob/master/day%2028%20-%20balanced%20trees/imgs/day28_2.png?raw=true)

## AVL: the strict rule

An AVL tree stores a height in every node and requires `|height(left) - height(right)| <= 1`
everywhere. After an insert, walk back up the path; the first node that violates the rule is
repaired with one or two rotations, and there are exactly four shapes. `LL` and `RR` are
straight paths and need a single rotation. `LR` and `RL` bend, and a single rotation only
produces the mirror image of the same problem - so the child is rotated first to straighten the
path, and then the parent is rotated. **The sign of the child's balance factor is what decides
single versus double**: same sign as the parent means straight, opposite sign means bent.

Deletion uses the same four cases but is not symmetric with insertion in cost. An insert
rebalances at most one node, because the rotation restores the subtree to its pre-insert height
and nothing above it changes. A delete can shorten a subtree, which can unbalance its parent,
which can unbalance *its* parent - so the repairs can cascade all the way to the root.

![img](https://github.com/KingRei/100DaysPython/blob/master/day%2028%20-%20balanced%20trees/imgs/day28_3.png?raw=true)

## Red-black: the loose rule

A red-black tree never measures a height. It colours nodes and maintains five properties: every
node is red or black, the root is black, the leaves (NIL sentinels) are black, a red node has
two black children, and every path from a node down to a leaf passes through the same number of
black nodes. The last two together do all the work: black nodes are equally distributed, and red
nodes cannot be adjacent, so the longest path is at most twice the shortest and
`height <= 2·log2(n + 1)`.

Insertion paints the new node red - which cannot break the black-height property, only the
no-two-reds one - and then repairs it in three cases. **Case 1**, a red uncle, is recolouring
only: parent and uncle go black, grandparent goes red, and the same violation reappears two
levels up. That is the case that loops. **Cases 2 and 3** have a black uncle: case 2 rotates
once to straighten a zig-zag into case 3, and case 3 recolours and rotates once more, after
which the subtree root is black and **the loop ends**. So an insert costs at most 2 rotations,
and a delete at most 3, no matter how large the tree is. Measured over 4095 random inserts and
2000 random deletes, the worst single operation used 2 and 3 rotations for red-black, against 2
and 5 for AVL.

![img](https://github.com/KingRei/100DaysPython/blob/master/day%2028%20-%20balanced%20trees/imgs/day28_4.png?raw=true)

## Which one, and why

On 100000 sorted keys, AVL ends up 16 levels deep with a mean lookup of 15.72 comparisons -
`log2(100000)` is 16.6, so it is essentially at the floor - after 99983 rotations. Red-black on
the same input is 30 levels deep with a mean lookup of 16.09, after 99969 rotations and 499846
recolourings. On shuffled input the two converge: height 19 for both, 15.95 against 16.03
comparisons, 69892 against 58398 rotations.

So AVL is shorter and reads faster, and red-black does fewer structural writes with a hard
per-operation bound; the difference is in constants and worst cases, not in the asymptotics.
That bound is why the Linux kernel's CFS scheduler, `epoll`, the ext3 directory index, `std::map`
and Java's `TreeMap` are all red-black rather than AVL: a scheduler cares much more about no
single operation ever being slow than about one extra comparison per lookup. Choose AVL when
reads dominate and writes are rare; choose red-black when writes are frequent and latency must
be predictable.

![img](https://github.com/KingRei/100DaysPython/blob/master/day%2028%20-%20balanced%20trees/imgs/day28_5.png?raw=true)

## The B-tree changes the question

Both structures above assume comparisons are expensive and pointer hops are free. On disk that
is backwards: a random read costs tens of thousands of cycles, and reading 16 bytes costs the
same as reading a whole 4096-byte page. If a page must be paid for anyway, fill it with keys. A
B-tree node *is* a page: with minimum degree `t`, every node except the root holds between
`t - 1` and `2t - 1` keys, and `degree_for_page(4096, 16)` gives `t = 128`, so a node holds up to
255 keys and has up to 256 children.

The shape follows immediately. One million keys fit in a tree of height 2 with 7874 nodes, and a
lookup reads 3 pages - against roughly 20 levels, and therefore up to 20 random reads, for a
binary tree. Insertion never rotates: descending toward a leaf, any full child is **split on the
way down**, its median key moving up into the parent. Because the parent was made non-full
before the descent, a split can never propagate upward, so one downward pass suffices. And when
the root itself is full it splits into a new root, which is the only way a B-tree ever grows -
**upward from the root, never downward at the leaves** - which is exactly why all leaves stay at
the same depth. With `t = 2` and keys 1 through 10 the result is a root of `[4]`, height 2, 8
nodes and 5 splits; sorted input, the worst case for a plain BST, is a non-event here.

![img](https://github.com/KingRei/100DaysPython/blob/master/day%2028%20-%20balanced%20trees/imgs/day28_6.png?raw=true)

## The LeetCode problems

LC 110 (Balanced Binary Tree) asks whether a tree satisfies the AVL invariant. The naive answer
calls a `height()` helper at every node and is `O(n²)`; the good answer makes one post-order
recursion return **both** the height and the verdict, using `-1` as a failure sentinel that
every level above simply forwards. One pass, `O(n)` time, `O(h)` stack - and that `O(h)` is
itself the argument for caring about height, since a degenerate tree does not merely search
slowly, it overflows the stack.

LC 108 (Convert Sorted Array to Binary Search Tree) is the rebuild half: take the middle element
as the root and recurse on the two halves, which differ in size by at most one, so the height is
forced to `ceil(log2(n + 1))`.

LC 1382 (Balance a Binary Search Tree) puts the two together - flatten with an in-order walk,
which is sorted by definition, then rebuild with the LC 108 procedure. A chain of 63 nodes goes
from height 62 to height 5 with the key set untouched. The cost is `O(n)` extra space for the
array; when that is not available, the **Day-Stout-Warren** algorithm rotates the entire tree
into a right-leaning vine and then back into balance in `O(1)` space, built out of the same
rotation primitive from the top of this page.

## The problems, stated in full

Restated in my own words - what is being asked, what goes in and comes out, one worked
example, and the constraints that actually change which algorithm is allowed.

### LeetCode 108 - Convert Sorted Array to Binary Search Tree

**The task.** Given an array sorted in ascending order, build a **height-balanced** binary
search tree from it. Any valid answer is accepted.

**Input / output.** Input is `nums`; output is the root of the tree.

**Example.** `nums = [-10,-3,0,5,9]` → one valid answer is the tree rooted at `0` with left
child `-3` (whose left child is `-10`) and right child `9` (whose left child is `5`).

**Constraints.** `1 <= len(nums) <= 10^4`, values in `[-10^4, 10^4]`, strictly increasing.
The recursive "take the middle element as the root, recurse on both halves" is the whole
solution, and it is the cleanest possible demonstration that balance is a property you can
*construct* rather than repair.

[leetcode.com/problems/convert-sorted-array-to-binary-search-tree](https://leetcode.com/problems/convert-sorted-array-to-binary-search-tree/)

### LeetCode 110 - Balanced Binary Tree

**The task.** Decide whether a binary tree is height-balanced, meaning that for **every**
node the heights of its two subtrees differ by at most one.

**Input / output.** Input is the root; output is a boolean.

**Example.** The tree `[3,9,20,null,null,15,7]` is balanced → `True`. The tree
`[1,2,2,3,3,null,null,4,4]` is not → `False`. An empty tree is balanced.

**Constraints.** `0 <= number of nodes <= 5000`, values in `[-10^4, 10^4]`. Computing the
height at every node separately is `O(n^2)`; returning "height, or a sentinel meaning
unbalanced" from one post-order pass makes it `O(n)` - the AVL invariant, checked instead
of maintained.

[leetcode.com/problems/balanced-binary-tree](https://leetcode.com/problems/balanced-binary-tree/)

### LeetCode 1382 - Balance a Binary Search Tree

**The task.** Given the root of a binary search tree, return **a** balanced binary search
tree holding exactly the same values. Balanced here means the depth of the two subtrees of
every node never differs by more than one. Any valid answer is accepted.

**Input / output.** Input is the root; output is the root of the rebuilt tree.

**Example.** The degenerate chain `1 -> 2 -> 3 -> 4` (each value the right child of the
previous) can be returned as the tree rooted at `2` with children `1` and `3`, and `4` as
the right child of `3`.

**Constraints.** `1 <= number of nodes <= 10^4`, values in `[1, 10^5]` and unique. The
two-line solution - in-order traversal into a sorted array, then LeetCode 108 - is a
perfectly good `O(n)` rebuild, and it is worth contrasting with what an AVL or red-black
tree does instead: pay a little on *every* insert so this rebuild is never needed.

[leetcode.com/problems/balance-a-binary-search-tree](https://leetcode.com/problems/balance-a-binary-search-tree/)

## Complexity

`n` = number of keys, `h` = height, `t` = B-tree minimum degree, `B` = page size.

| Operation | Time | Space |
|---|---|---|
| BST search / insert / delete (no balancing) | O(h), h up to n | O(1) iterative |
| `rotate_left` / `rotate_right` | O(1) | O(1) |
| AVL search | O(log n), height <= 1.44·log2(n) | O(1) |
| AVL insert (<= 1 rebalance) | O(log n) | O(log n) recursion |
| AVL delete (rebalance may cascade to the root) | O(log n) | O(log n) |
| Red-black search | O(log n), height <= 2·log2(n+1) | O(1) |
| Red-black insert (<= 2 rotations) | O(log n) | O(1) iterative fixup |
| Red-black delete (<= 3 rotations) | O(log n) | O(1) |
| B-tree search | O(log_t n) node visits, O(t·log_t n) comparisons | O(1) |
| B-tree insert (split on the way down) | O(t·log_t n) | O(1), one downward pass |
| B-tree page reads per lookup | O(log_t n) = height + 1 | - |
| `degree_for_page(B, entry)` | O(1) | O(1) |
| LC 110 `is_balanced` | O(n) | O(h) |
| LC 108 `sorted_array_to_bst` | O(n) | O(n) |
| LC 1382 `balance_bst` (flatten + rebuild) | O(n) | O(n) |
| Day-Stout-Warren (in-place rebalance) | O(n) | O(1) |

## References
- [AVL tree - Wikipedia](https://en.wikipedia.org/wiki/AVL_tree)
- [An algorithm for the organization of information (Adelson-Velsky and Landis, 1962)](https://zhjwpku.com/assets/pdf/AED2-10-avl-paper.pdf)
- [Red-black tree - Wikipedia](https://en.wikipedia.org/wiki/Red%E2%80%93black_tree)
- [A dichromatic framework for balanced trees (Guibas and Sedgewick, 1978)](https://doi.org/10.1109/SFCS.1978.3)
- [Introduction to Algorithms, chapters 13 and 18 (CLRS)](https://mitpress.mit.edu/9780262046305/introduction-to-algorithms/)
- [B-tree - Wikipedia](https://en.wikipedia.org/wiki/B-tree)
- [Organization and maintenance of large ordered indexes (Bayer and McCreight, 1972)](https://doi.org/10.1007/BF00288683)
- [The Ubiquitous B-Tree (Comer, 1979)](https://doi.org/10.1145/356770.356776)
- [Linux kernel red-black trees (Documentation/core-api/rbtree.rst)](https://www.kernel.org/doc/html/latest/core-api/rbtree.html)
- [Tree rebalancing in optimal time and space (Stout and Warren, 1986)](https://doi.org/10.1145/6592.6599)
- [LeetCode 110 · Balanced Binary Tree](https://leetcode.com/problems/balanced-binary-tree/)
- [LeetCode 108 · Convert Sorted Array to Binary Search Tree](https://leetcode.com/problems/convert-sorted-array-to-binary-search-tree/)
- [LeetCode 1382 · Balance a Binary Search Tree](https://leetcode.com/problems/balance-a-binary-search-tree/)
