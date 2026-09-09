"""Day 28 figures - balanced trees: AVL, red-black, B-tree.

Regenerate with:  python3 make_figs.py
Every number on these figures is computed by importing balanced_trees.py.
"""

import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(HERE, '..', '..', 'tools'))
sys.path.append(os.path.join(HERE, '..'))

from diagram_style import *          # noqa: E402,F403
import balanced_trees as B           # noqa: E402


def tree_layout(node, depth=0, x0=0.0, slot=None):
    """Assign (x, y) to every node of a small tree by in-order position."""
    if slot is None:
        slot = [0]
    pos = {}
    if node is None:
        return pos
    pos.update(tree_layout(node.left, depth + 1, x0, slot))
    pos[id(node)] = (slot[0], depth, node)
    slot[0] += 1
    pos.update(tree_layout(node.right, depth + 1, x0, slot))
    return pos


def draw_tree(ax, root, ox, oy, dx, dy, r=0.30, size=12,
              color_of=None, edge_lw=1.6):
    """Draw a small binary tree; returns {id(node): (x, y)}."""
    pos = tree_layout(root)
    xy = {k: (ox + s * dx, oy - d * dy) for k, (s, d, _) in pos.items()}
    for k, (s, d, n) in pos.items():
        for child in (n.left, n.right):
            if child is not None and id(child) in xy:
                ax.plot([xy[k][0], xy[id(child)][0]],
                        [xy[k][1], xy[id(child)][1]],
                        color=TEAL, lw=edge_lw, zorder=2)
    for k, (s, d, n) in pos.items():
        fill, edg = NAVY, TEAL
        if color_of is not None:
            fill, edg = color_of(n)
        node(ax, xy[k][0], xy[k][1], str(n.key), r=r, fill=fill, edge=edg,
             size=size)
    return xy


# --------------------------------------------------------------------------
# 1. Why balance: the same keys, two insertion orders
# --------------------------------------------------------------------------
def fig1():
    fig, ax = canvas(10.4, 6.8, xlim=(0, 10.4), ylim=(0, 6.8))

    text(ax, 2.30, 6.52, 'inserted 1, 2, 3, 4, 5 in order', color=TEAL_L,
         size=15)
    chain = None
    for k in [1, 2, 3, 4, 5]:
        chain = B.bst_insert(chain, k)
    draw_tree(ax, chain, 1.00, 6.00, 0.62, 0.66, r=0.26, size=12)
    text(ax, 2.05, 2.86, 'height 4  -  a linked list', color=RED, size=15)

    text(ax, 7.75, 6.52, 'the same five keys, balanced', color=TEAL_L,
         size=15)
    bal = B.sorted_array_to_bst([1, 2, 3, 4, 5])
    draw_tree(ax, bal, 6.55, 6.00, 0.62, 0.66, r=0.26, size=12)
    text(ax, 7.75, 3.75, 'height 2', color=CYAN, size=15)

    n = 2047
    keys = list(range(1, n + 1))
    a = None
    for k in keys:
        a = B.bst_insert(a, k)
    shuffled = keys[:]
    random.Random(28).shuffle(shuffled)
    b = None
    for k in shuffled:
        b = B.bst_insert(b, k)
    c = B.sorted_array_to_bst(keys)

    box(ax, 0.35, 0.62, 9.70, 2.05, fill='none', edge=CYAN, lw=1.7)
    text(ax, 5.20, 2.34, '2047 keys, three insertion orders', color=CYAN,
         size=15, glowing=False)
    rows = [('sorted', a, RED), ('shuffled', b, TEAL), ('balanced', c, CYAN)]
    text(ax, 0.75, 1.88, 'order', color=TEAL_L, size=13, ha='left',
         glowing=False)
    text(ax, 4.60, 1.88, 'height', color=TEAL_L, size=13, ha='right',
         glowing=False)
    text(ax, 8.00, 1.88, 'mean comparisons per lookup', color=TEAL_L, size=13,
         ha='right', glowing=False)
    for i, (what, tree, col) in enumerate(rows):
        y = 1.52 - i * 0.36
        text(ax, 0.75, y, what, color=col, size=13, ha='left', glowing=False)
        text(ax, 4.60, y, '%d' % B.height(tree), color=col, size=13,
             ha='right', glowing=False)
        text(ax, 8.00, y, '%.1f' % B.mean_search_cost(tree, keys), color=col,
             size=13, ha='right', glowing=False)

    text(ax, 5.20, 0.28,
         'sorted input is the normal case: ids, timestamps, a restored dump',
         color=TEAL_L, size=13, glowing=False)
    save(fig, 'day28_1.png')


# --------------------------------------------------------------------------
# 2. The rotation primitive
# --------------------------------------------------------------------------
def fig2():
    fig, ax = canvas(9.6, 5.4, xlim=(0, 9.6), ylim=(0, 5.4))

    def side(ox, top, bot, sub_left, sub_mid, sub_right):
        node(ax, ox + 1.20, 4.35, top, r=0.34, size=15)
        node(ax, ox + 0.60, 3.35, bot, r=0.34, size=15)
        ax.plot([ox + 1.20, ox + 0.60], [4.35, 3.35], color=TEAL, lw=1.7,
                zorder=2)
        tri(ox + 0.05, 2.30, sub_left)
        tri(ox + 1.05, 2.30, sub_mid)
        tri(ox + 1.75, 3.30, sub_right)
        ax.plot([ox + 0.60, ox + 0.35], [3.35, 2.78], color=TEAL, lw=1.7,
                zorder=2)
        ax.plot([ox + 0.60, ox + 1.35], [3.35, 2.78], color=TEAL, lw=1.7,
                zorder=2)
        ax.plot([ox + 1.20, ox + 2.05], [4.35, 3.78], color=TEAL, lw=1.7,
                zorder=2)

    def tri(x, y, label):
        ax.plot([x, x + 0.60, x + 0.30, x],
                [y, y, y + 0.48, y], color=TEAL_L, lw=1.5, zorder=2)
        text(ax, x + 0.30, y + 0.17, label, color=TEAL_L, size=13,
             glowing=False)

    # left: y on top, x below  -> rotate right
    side(0.55, 'y', 'x', 'A', 'B', 'C')
    # right: x on top, y below (mirrored) -> after rotate right
    node(ax, 6.95, 4.35, 'x', r=0.34, size=15)
    node(ax, 7.65, 3.35, 'y', r=0.34, size=15)
    ax.plot([6.95, 7.65], [4.35, 3.35], color=TEAL, lw=1.7, zorder=2)
    ax.plot([6.95, 6.20], [4.35, 3.78], color=TEAL, lw=1.7, zorder=2)
    ax.plot([7.65, 7.35], [3.35, 2.78], color=TEAL, lw=1.7, zorder=2)
    ax.plot([7.65, 8.35], [3.35, 2.78], color=TEAL, lw=1.7, zorder=2)
    tri(5.90, 3.30, 'A')
    tri(7.05, 2.30, 'B')
    tri(8.05, 2.30, 'C')

    arrow(ax, (3.35, 3.60), (5.15, 3.60), color=AMBER, lw=2.0)
    text(ax, 4.25, 4.05, 'rotate_right(y)', color=AMBER, size=14)
    text(ax, 4.25, 3.20, 'rotate_left(x)', color=AMBER, size=14)
    arrow(ax, (5.15, 2.95), (3.35, 2.95), color=AMBER, lw=2.0)

    box(ax, 1.35, 0.35, 6.90, 1.15, fill='none', edge=CYAN, lw=1.7)
    text(ax, 4.80, 1.15, 'in-order reading is identical on both sides',
         color=CYAN, size=15, glowing=False)
    text(ax, 4.80, 0.68, 'A   x   B   y   C', color=TEAL_L, size=17,
         glowing=False)
    save(fig, 'day28_2.png')


# --------------------------------------------------------------------------
# 3. AVL's four shapes
# --------------------------------------------------------------------------
def fig3():
    fig, ax = canvas(9.8, 6.0, xlim=(0, 9.8), ylim=(-0.45, 5.55))

    cases = [
        ('LL', [30, 20, 10], 'LL  -  left left', 'single: rotate_right', 0.30),
        ('LR', [30, 10, 20], 'LR  -  left right', 'double: left, then right', 2.65),
        ('RR', [10, 20, 30], 'RR  -  right right', 'single: rotate_left', 5.00),
        ('RL', [10, 30, 20], 'RL  -  right left', 'double: right, then left', 7.35),
    ]

    for name, keys, shape, fixname, ox in cases:
        # unbalanced shape, drawn by hand from the insertion order
        a, b, c = keys
        pts = {}
        node(ax, ox + 1.10, 5.05, str(a), r=0.30, size=13)
        if name.startswith('L'):
            node(ax, ox + 0.60, 4.25, str(b), r=0.30, size=13)
            ax.plot([ox + 1.10, ox + 0.60], [5.05, 4.25], color=TEAL, lw=1.6)
            x3 = ox + 0.20 if name == 'LL' else ox + 1.00
            node(ax, x3, 3.45, str(c), r=0.30, size=13)
            ax.plot([ox + 0.60, x3], [4.25, 3.45], color=TEAL, lw=1.6)
        else:
            node(ax, ox + 1.60, 4.25, str(b), r=0.30, size=13)
            ax.plot([ox + 1.10, ox + 1.60], [5.05, 4.25], color=TEAL, lw=1.6)
            x3 = ox + 2.00 if name == 'RR' else ox + 1.20
            node(ax, x3, 3.45, str(c), r=0.30, size=13)
            ax.plot([ox + 1.60, x3], [4.25, 3.45], color=TEAL, lw=1.6)

        text(ax, ox + 1.10, 2.92, shape, color=AMBER, size=13,
             glowing=False)
        arrow(ax, (ox + 1.10, 2.62), (ox + 1.10, 2.05), color=AMBER, lw=1.8)

        # rebalanced: always 20 on top with 10 and 30 as leaves
        node(ax, ox + 1.10, 1.72, '20', r=0.30, size=13)
        node(ax, ox + 0.60, 0.95, '10', r=0.30, size=13)
        node(ax, ox + 1.60, 0.95, '30', r=0.30, size=13)
        ax.plot([ox + 1.10, ox + 0.60], [1.72, 0.95], color=TEAL, lw=1.6)
        ax.plot([ox + 1.10, ox + 1.60], [1.72, 0.95], color=TEAL, lw=1.6)
        text(ax, ox + 1.10, 0.42, fixname, color=TEAL_L, size=12,
             glowing=False)

    text(ax, 4.90, -0.28,
         'the sign of the balance factor of the child decides '
         'single vs double',
         color=TEAL_L, size=13, glowing=False)
    save(fig, 'day28_3.png')


# --------------------------------------------------------------------------
# 4. Red-black: the rules, and the case that costs nothing
# --------------------------------------------------------------------------
def fig4():
    fig, ax = canvas(10.4, 5.8, xlim=(0, 10.4), ylim=(0, 5.8))

    rules = ['1. every node is red or black',
             '2. the root is black',
             '3. every leaf (NIL) is black',
             '4. a red node has two black',
             '     children',
             '5. every root-to-leaf path has',
             '     the same number of blacks']
    box(ax, 0.25, 2.20, 3.85, 3.35, fill='none', edge=TEAL, lw=1.7)
    text(ax, 2.17, 5.22, 'five properties', color=CYAN, size=15)
    for i, r in enumerate(rules):
        text(ax, 0.45, 4.78 - i * 0.40, r, color=TEAL_L, size=13, ha='left',
             glowing=False)

    box(ax, 0.25, 0.40, 3.85, 1.55, fill='none', edge=CYAN, lw=1.7)
    text(ax, 2.17, 1.60, 'property 4 + property 5', color=CYAN, size=14,
         glowing=False)
    text(ax, 2.17, 1.15, 'no path is more than twice', color=TEAL_L, size=13,
         glowing=False)
    text(ax, 2.17, 0.75, 'the length of any other', color=TEAL_L, size=13,
         glowing=False)

    RED_F, BLACK_F = '#5a1414', NAVY

    def rb(x, y, lab, red, r=0.28):
        node(ax, x, y, lab, r=r, fill=RED_F if red else BLACK_F,
             edge=RED if red else TEAL, size=12)

    text(ax, 7.30, 5.42, 'insert fixup, case 1: the uncle is red',
         color=TEAL_L, size=14, glowing=False)

    def triple(ox, gr, pr, ur):
        ax.plot([ox + 0.95, ox + 0.45], [4.75, 4.05], color=TEAL, lw=1.5)
        ax.plot([ox + 0.95, ox + 1.45], [4.75, 4.05], color=TEAL, lw=1.5)
        ax.plot([ox + 0.45, ox + 0.10], [4.05, 3.38], color=TEAL, lw=1.5)
        rb(ox + 0.95, 4.75, 'G', gr)
        rb(ox + 0.45, 4.05, 'P', pr)
        rb(ox + 1.45, 4.05, 'U', ur)
        rb(ox + 0.10, 3.38, 'N', True)

    triple(4.75, False, True, True)
    text(ax, 4.85, 2.98, 'new', color=AMBER, size=12)
    triple(8.10, True, False, False)

    arrow(ax, (6.55, 4.10), (7.55, 4.10), color=AMBER, lw=1.9)
    text(ax, 7.05, 4.48, 'recolour', color=AMBER, size=13)

    box(ax, 4.35, 0.40, 5.80, 2.05, fill='none', edge=CYAN, lw=1.7)
    text(ax, 7.25, 1.98, 'no pointer moved  -  three colour bits',
         color=CYAN, size=14, glowing=False)
    for i, line in enumerate([
            'the violation moves two levels up and repeats;',
            'the cases that do rotate (2 and 3) end the loop',
            'at once, so an insert costs at most 2 rotations']):
        text(ax, 7.25, 1.50 - i * 0.40, line, color=TEAL_L, size=13,
             glowing=False)
    save(fig, 'day28_4.png')


# --------------------------------------------------------------------------
# 5. Measured: AVL is shorter, red-black is bounded
# --------------------------------------------------------------------------
def fig5():
    n = 100000
    keys = list(range(n))
    shuffled = keys[:]
    random.Random(28).shuffle(shuffled)
    probes = random.Random(2828).sample(keys, 2000)

    s = B.build_all(keys)
    r = B.build_all(shuffled)

    def rot(built):
        st = built['avl_stats']
        return st['single'] + 2 * st['double']

    fig, ax = canvas(9.8, 5.4, xlim=(0, 9.8), ylim=(0, 5.4))

    cols = [0.32, 3.15, 4.85, 6.75, 8.55]
    text(ax, 4.00, 5.22, 'sorted keys', color=TEAL_L, size=14, glowing=False)
    text(ax, 7.65, 5.22, 'shuffled keys', color=TEAL_L, size=14,
         glowing=False)
    for x, h in zip(cols, ['n = 100000', 'AVL', 'red-black', 'AVL',
                           'red-black']):
        text(ax, x, 4.76, h, color=CYAN, size=13,
             ha='left' if x == cols[0] else 'center', glowing=False)

    data = [
        ('height', '%d' % B.height(s['avl']), '%d' % s['rb'].height(),
         '%d' % B.height(r['avl']), '%d' % r['rb'].height()),
        ('mean lookup', '%.2f' % B.mean_search_cost(s['avl'], probes),
         '%.2f' % (sum(s['rb'].search_cost(k) for k in probes) / len(probes)),
         '%.2f' % B.mean_search_cost(r['avl'], probes),
         '%.2f' % (sum(r['rb'].search_cost(k) for k in probes) / len(probes))),
        ('rotations', '%d' % rot(s), '%d' % s['rb'].stats['rotations'],
         '%d' % rot(r), '%d' % r['rb'].stats['rotations']),
        ('recolourings', '-', '%d' % s['rb'].stats['recolors'],
         '-', '%d' % r['rb'].stats['recolors']),
    ]
    for i, row in enumerate(data):
        y = 4.26 - i * 0.46
        for x, cell in zip(cols, row):
            text(ax, x, y, cell, color=TEAL_L if x == cols[0] else TEAL,
                 size=13, ha='left' if x == cols[0] else 'center',
                 glowing=False)

    rng = random.Random(4095)
    wkeys = rng.sample(range(1000000), 4095)
    victims = rng.sample(wkeys, 2000)
    avl, a_ins, a_del = None, 0, 0
    for k in wkeys:
        st = {'single': 0, 'double': 0}
        avl = B.avl_insert(avl, k, st)
        a_ins = max(a_ins, st['single'] + 2 * st['double'])
    for k in victims:
        st = {'single': 0, 'double': 0}
        avl = B.avl_delete(avl, k, st)
        a_del = max(a_del, st['single'] + 2 * st['double'])
    rb = B.RedBlackTree()
    r_ins, r_del = 0, 0
    for k in wkeys:
        before = rb.stats['rotations']
        rb.insert(k)
        r_ins = max(r_ins, rb.stats['rotations'] - before)
    for k in victims:
        before = rb.stats['rotations']
        rb.delete(k)
        r_del = max(r_del, rb.stats['rotations'] - before)

    box(ax, 0.25, 0.32, 9.30, 1.92, fill='none', edge=AMBER, lw=1.8)
    text(ax, 4.90, 1.98, 'worst rotations in a single operation',
         color=AMBER, size=15, glowing=False)
    text(ax, 4.90, 1.52, 'AVL          insert %d      delete %d'
         % (a_ins, a_del), color=TEAL, size=15, glowing=False)
    text(ax, 4.90, 1.08, 'red-black    insert %d      delete %d'
         % (r_ins, r_del), color=TEAL, size=15, glowing=False)
    text(ax, 4.90, 0.58,
         'AVL wins the average lookup; red-black wins the per-operation '
         'guarantee',
         color=TEAL_L, size=13, glowing=False)
    save(fig, 'day28_5.png')


# --------------------------------------------------------------------------
# 6. B-tree: the cost model changes
# --------------------------------------------------------------------------
def fig6():
    fig, ax = canvas(9.8, 5.6, xlim=(0, 9.8), ylim=(0, 5.6))

    t = B.degree_for_page(4096, 16)
    N = 1000000
    bt = B.BTree(t=t)
    for k in range(N):
        bt.insert(k)

    text(ax, 4.90, 5.28, 'one node = one 4 KB disk page', color=CYAN, size=15)

    box(ax, 2.55, 4.15, 4.70, 0.70, fill=NAVY, edge=TEAL, lw=1.8)
    text(ax, 4.90, 4.50, 'up to %d keys in one node' % (2 * t - 1),
         color=TEAL_L, size=14, glowing=False)
    text(ax, 1.90, 4.50, 'root', color=TEAL_L, size=13, glowing=False)

    xs = [0.55, 2.70, 4.85, 7.00]
    labs = ['keys < a', 'a .. b', 'b .. c', 'c .. d']
    for x, lab in zip(xs, labs):
        box(ax, x, 2.80, 1.95, 0.60, fill=NAVY, edge=TEAL, lw=1.6,
            label=lab)
        arrow(ax, (4.90, 4.11), (x + 0.97, 3.46), color=TEAL, lw=1.4)
        box(ax, x, 1.62, 1.95, 0.54, fill=GREY, edge=TEAL_L, lw=1.4)
        arrow(ax, (x + 0.97, 2.76), (x + 0.97, 2.20), color=TEAL, lw=1.4)
    text(ax, 9.20, 3.10, '. . .', color=TEAL_L, size=15, glowing=False)
    text(ax, 9.20, 1.89, '. . .', color=TEAL_L, size=15, glowing=False)

    box(ax, 0.28, 0.28, 9.24, 1.05, fill='none', edge=AMBER, lw=1.8)
    text(ax, 4.90, 0.94,
         '%s keys,  t = %d   ->   height %d,  %s nodes'
         % ('{:,}'.format(N), t, bt.height(), '{:,}'.format(bt.node_count())),
         color=TEAL, size=14, glowing=False)
    text(ax, 4.90, 0.52,
         'a lookup reads %d pages; a binary tree would be ~20 levels deep'
         % (bt.height() + 1),
         color=AMBER, size=13, glowing=False)
    save(fig, 'day28_6.png')


if __name__ == '__main__':
    os.chdir(HERE)
    fig1()
    fig2()
    fig3()
    fig4()
    fig5()
    fig6()
    print('wrote day28_1..6.png')
