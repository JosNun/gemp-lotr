"""The combined network's July agreement by epoch, one panel per decision, against the current bot (sel10-net6: trees, plus the first network at Fellowship and Shadow plays; dashed).

usage (from nn/):  uv run python plot_epochs.py <net2.log> <out.png> [--assign <epoch -> legal-decode % json>]
"""
import argparse, json, re

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# sel10-net6 on 30% of July (HumanReplay --counterfactual, labels/hl/hl-sel10-net6.tsv); the win model's AUC from
# BOT-NOTES (13 terms, scripts/aggregate-value.weights)
TREES = {'fellowship': 68.6, 'shadow': 64.1, 'skirmish': 81.1, 'maneuver': 83.7, 'archery': 70.8, 'regroup': 81.5,
         'optional': 90.3, 'reconcile': 52.4, 'move': 82.1, 'assign fp': 54.1}
NAMES = {'fellowship': 'Fellowship plays', 'shadow': 'Shadow plays', 'skirmish': 'Skirmish plays',
         'maneuver': 'Maneuver plays', 'archery': 'Archery plays', 'regroup': 'Regroup plays',
         'optional': 'Optional responses', 'reconcile': 'Reconcile', 'move': 'Move again?',
         'assign fp': 'Assignment (FP, legal)'}
WIN_BASELINE = 0.670


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('log')
    ap.add_argument('out')
    ap.add_argument('--assign', default=None)
    args = ap.parse_args()
    epochs, acc, auc = [], {k: [] for k in TREES}, []
    for line in open(args.log):
        m = re.match(r'epoch (\d+):', line)
        if not m:
            continue
        epochs.append(int(m.group(1)))
        for k in TREES:
            v = re.search(rf'[;,(] {re.escape(k)} ([\d.]+)%', line)
            acc[k].append(float(v.group(1)) if v else None)
        auc.append(float(re.search(r'win AUC ([\d.]+)', line).group(1)))
    if args.assign:
        legal = json.load(open(args.assign))
        acc['assign fp'] = [legal.get(str(e)) for e in epochs]
    fig, axes = plt.subplots(3, 4, figsize=(14, 9), constrained_layout=True)
    panels = list(TREES) + ['win']
    for ax, k in zip(axes.flat, panels):
        if k == 'win':
            ax.plot(epochs, auc, 'o-', color='tab:purple', label='network')
            ax.axhline(WIN_BASELINE, ls='--', color='grey', label='current judge')
            ax.set_title('Win prediction (AUC)')
            lo, hi = min(auc + [WIN_BASELINE]), max(auc + [WIN_BASELINE])
        else:
            ys = acc[k]
            xs = [e for e, y in zip(epochs, ys) if y is not None]
            ys = [y for y in ys if y is not None]
            ahead = ys and ys[-1] >= TREES[k]
            ax.plot(xs, ys, 'o-', color='tab:green' if ahead else 'tab:blue', label='network')
            ax.axhline(TREES[k], ls='--', color='grey', label='current bot (sel10-net6)')
            ax.set_title(NAMES[k])
            lo, hi = min(ys + [TREES[k]]), max(ys + [TREES[k]])
            if ys:
                ax.annotate(f'{ys[-1]:.1f}%', (xs[-1], ys[-1]), textcoords='offset points', xytext=(4, 4), fontsize=9)
            ax.annotate(f'{TREES[k]:.1f}%', (1, TREES[k]), textcoords='offset points', xytext=(0, -12), fontsize=8,
                        color='grey')
        pad = (hi - lo) * 0.3 + (0.5 if k != 'win' else 0.005)
        ax.set_ylim(lo - pad, hi + pad)
        ax.set_xlim(0.7, max(8, max(epochs)) + 0.3)
        ax.set_xticks(range(1, max(8, max(epochs)) + 1))
        ax.grid(alpha=0.3)
    axes.flat[-1].axis('off')
    axes.flat[0].legend(loc='lower right', fontsize=8)
    axes.flat[len(panels) - 1].legend(loc='lower right', fontsize=8)
    fig.suptitle(f'Combined network vs the current bot (sel10-net6) on held-out July -- agreement with people by epoch '
                 f'(green: at or above it)', fontsize=12)
    fig.savefig(args.out, dpi=110)


if __name__ == '__main__':
    main()
