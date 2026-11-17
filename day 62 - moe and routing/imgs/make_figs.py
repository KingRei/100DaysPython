"""Figures for Day 62 - every number is computed by importing moe.py."""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(HERE, '..', '..', 'tools'))
sys.path.append(os.path.join(HERE, '..'))

from diagram_style import *            # noqa: E402
import moe as M                        # noqa: E402

TASK = M.make_task()
PLAIN = M.train_moe(TASK)
BAL = M.train_moe(TASK, aux=0.3)
FREE = M.train_moe(TASK, bias_rate=0.001)
ACC = M.ffn_vs_moe()


def fig0():
    """The lead figure: one FFN becomes many, and only k of them run."""
    fig, ax = canvas(10.6, 6.8, xlim=(0, 10.6), ylim=(0, 6.8))
    text(ax, 5.3, 6.5, 'from one FFN to a mixture of experts', size=17)

    # ---- dense
    text(ax, 2.3, 5.95, 'dense: every token, every weight', size=13, color=TEAL_L)
    node(ax, 0.75, 4.3, 'x', r=0.3)
    box(ax, 1.5, 3.6, 1.7, 1.4, label='FFN\nd_ff', size=12)
    node(ax, 4.0, 4.3, 'y', r=0.3)
    arrow(ax, (1.1, 4.3), (1.45, 4.3)); arrow(ax, (3.25, 4.3), (3.65, 4.3))
    text(ax, 2.35, 3.25, f'{ACC["dense_params"] / 1e6:.0f} M parameters,\n'
                         f'all of them active', size=11, color=GREY_L)

    # ---- moe
    text(ax, 7.6, 5.95, 'MoE: every token, two experts', size=13, color=TEAL_L)
    node(ax, 5.3, 4.3, 'x', r=0.3)
    box(ax, 5.95, 4.0, 0.95, 0.6, label='router', size=11, fill=GREY)
    arrow(ax, (5.65, 4.3), (5.9, 4.3))
    picked = [1, 3]
    for i in range(5):
        y = 5.35 - i * 0.62
        on = i in picked
        box(ax, 7.3, y - 0.22, 1.25, 0.44, label=f'E{i + 1}', size=11,
            fill=NAVY if on else GREY, edge=TEAL if on else GREY_L,
            lw=1.8 if on else 0.9)
        if on:
            arrow(ax, (6.95, 4.3), (7.25, y), rad=0.15, lw=1.4)
            arrow(ax, (8.6, y), (8.95, 4.3), rad=0.15, lw=1.4)
    text(ax, 7.9, 2.35, '...', size=13, color=GREY_L)
    node(ax, 9.3, 4.3, 'y', r=0.3)
    text(ax, 7.6, 1.85, f'{ACC["moe_params"] / 1e9:.1f} B parameters,\n'
                        f'{ACC["active_params"] / 1e6:.0f} M active per token',
         size=11, color=GREY_L)

    text(ax, 5.3, 1.0, f'{ACC["param_ratio"]:.0f}x the parameters for '
                       f'{ACC["flop_ratio"]:.0f}x the arithmetic per token',
         size=14, color=CYAN)
    text(ax, 5.3, 0.45, 'the price: the router has to spread the tokens, and the '
                        'experts live on different GPUs', size=12, color=GREY_L)
    save(fig, os.path.join(HERE, 'day62_0.png'))


def fig1():
    """What the router does with the tokens, with and without balancing."""
    fig, ax = canvas(10.4, 5.8, xlim=(0, 10.4), ylim=(0, 5.8))
    text(ax, 5.2, 5.5, 'tokens per expert, sorted - the same model, three routers',
         size=16)
    runs = [('no balancing', PLAIN, RED), ('auxiliary loss', BAL, TEAL),
            ('bias, no loss', FREE, AMBER)]
    top = max(r[1]['counts'].max() for r in runs)
    w, gap = 0.155, 0.04
    for i, (name, r, col) in enumerate(runs):
        x0 = 0.75 + i * 3.25
        counts = sorted(r['counts'].tolist(), reverse=True)
        mean = np.mean(counts)
        for j, c in enumerate(counts):
            h = 2.4 * c / top
            ax.add_patch(plt.Rectangle((x0 + j * (w + gap), 1.75), w, max(h, 0.02),
                                       facecolor=col, edgecolor=col))
        ax.plot([x0 - 0.05, x0 + len(counts) * (w + gap)],
                [1.75 + 2.4 * mean / top] * 2, color=GREY_L, lw=1.0, ls='--')
        text(ax, x0 + 1.35, 4.55, name, size=13, color=col, glowing=False)
        b = r['balance']
        text(ax, x0 + 1.35, 1.35,
             f"busiest/avg {b['max_over_mean']:.2f}\n{b['dead']} dead  "
             f"loss {r['loss']:.4f}", size=11, color=GREY_L)
    text(ax, 5.2, 0.5, 'dashed line: a perfectly flat load', size=12, color=GREY_L)
    save(fig, os.path.join(HERE, 'day62_1.png'))


def fig2():
    """Expert parallelism: the tokens have to travel."""
    ep = M.ep_simulation(PLAIN['counts'], n_gpu=4, d_model=4096)
    epb = M.ep_simulation(FREE['counts'], n_gpu=4, d_model=4096)
    fig, ax = canvas(10.4, 6.0, xlim=(0, 10.4), ylim=(0, 6.0))
    text(ax, 5.2, 5.7, 'expert parallelism: two all-to-alls per layer', size=16)
    for g in range(4):
        x = 0.8 + g * 2.4
        box(ax, x, 4.3, 1.9, 0.75, label=f'GPU {g}\n4 experts', size=11)
        box(ax, x, 1.5, 1.9, 0.75, label='tokens', size=11, fill=GREY)
        arrow(ax, (x + 0.95, 4.25), (x + 0.95, 2.35), lw=1.4)
    text(ax, 5.2, 3.35, 'dispatch: each token to the GPU that owns its expert',
         size=12, color=GREY_L)
    top = ep['gpu_tokens'].max()
    for g, c in enumerate(ep['gpu_tokens'].tolist()):
        x = 0.8 + g * 2.4
        ax.add_patch(plt.Rectangle((x, 0.6), 1.9 * c / top, 0.3,
                                   facecolor=RED, edgecolor=RED))
        text(ax, x + 1.95, 0.75, str(c), size=11, ha='left', color=RED, glowing=False)
    text(ax, 5.2, 0.2, f'busiest/average {ep["max_over_mean"]:.2f} unbalanced, '
                       f'{epb["max_over_mean"]:.2f} balanced - everyone waits for the '
                       f'slowest', size=12, color=CYAN)
    save(fig, os.path.join(HERE, 'day62_2.png'))


def fig3():
    """Capacity factor: drops against padding."""
    rows = [M.capacity_drop(PLAIN['counts'], TASK['X'].shape[0], 2, 16, f)
            for f in (1.0, 1.25, 1.5, 2.0)]
    rows_b = [M.capacity_drop(FREE['counts'], TASK['X'].shape[0], 2, 16, f)
              for f in (1.0, 1.25, 1.5, 2.0)]
    fig, ax = canvas(10.4, 5.2, xlim=(0, 10.4), ylim=(0, 5.2))
    text(ax, 5.2, 4.9, 'the capacity factor: dropped tokens against wasted buffer',
         size=16)
    for i, (r, rb) in enumerate(zip(rows, rows_b)):
        x = 1.1 + i * 2.3
        ax.add_patch(plt.Rectangle((x, 2.3), 0.75, 2.0 * r['drop_rate'],
                                   facecolor=RED, edgecolor=RED))
        ax.add_patch(plt.Rectangle((x + 0.85, 2.3), 0.75, 2.0 * rb['drop_rate'],
                                   facecolor=AMBER, edgecolor=AMBER))
        text(ax, x + 0.8, 2.0, f"factor {r['factor']:g}", size=12, glowing=False)
        text(ax, x + 0.38, 2.35 + 2.0 * r['drop_rate'] + 0.18,
             f"{r['drop_rate']:.0%}", size=11, color=RED, glowing=False)
        text(ax, x + 1.23, 2.35 + 2.0 * rb['drop_rate'] + 0.18,
             f"{rb['drop_rate']:.0%}", size=11, color=AMBER, glowing=False)
        text(ax, x + 0.8, 1.5, f"padding +{r['padding']:.0%}", size=11,
             color=GREY_L, glowing=False)
    legend(ax, 0.9, 1.0, [(RED, 'dropped, unbalanced router'),
                          (AMBER, 'dropped, balanced router')])
    save(fig, os.path.join(HERE, 'day62_3.png'))


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    globals()['plt'] = plt
    fig0(); fig1(); fig2(); fig3()
    print('wrote day62_0 day62_1 day62_2 day62_3')
