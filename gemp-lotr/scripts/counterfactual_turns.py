"""Where in the game the bot loses to people: HumanReplay --counterfactual rows (with turn and phase, columns 14-15)
summed by turn, side and phase.

    python3 scripts/counterfactual_turns.py cf.tsv [--min 30]
"""
import collections, math, sys

argv = sys.argv[1:]
MIN = 30
if '--min' in argv:
    i = argv.index('--min'); MIN = int(argv[i + 1]); del argv[i:i + 2]
rows = []
for f in argv:
    for line in open(f, encoding='utf-8'):
        p = line.rstrip('\n').split('\t')
        if len(p) < 15:
            continue
        rows.append(dict(kind=p[2], h=int(p[6]), b=int(p[7]), n=int(p[8]), side=p[9], turn=int(p[13]), phase=p[14]))


def show(title, key):
    g = collections.defaultdict(lambda: {'asked': 0, 'agree': 0, 'd': []})
    for r in rows:
        k = key(r)
        g[k]['asked'] += 1
        if r['n'] == 0:
            g[k]['agree'] += 1
        elif r['n'] > 0:
            g[k]['d'].append((r['b'] - r['h']) / r['n'])
    print(title)
    print(f"  {'':34s} {'asked':>6s} {'agree':>6s} {'played':>6s} {'bot-human':>14s} {'net':>7s}")
    for k in sorted(g, key=lambda k: sum(g[k]['d'])):
        v = g[k]
        if v['asked'] < MIN:
            continue
        d = v['d']
        m = sum(d) / len(d) if d else 0
        se = math.sqrt(sum((x - m) ** 2 for x in d) / max(1, len(d) - 1) / max(1, len(d))) if len(d) > 1 else 0
        print(f"  {str(k)[:34]:34s} {v['asked']:6d} {100 * v['agree'] / v['asked']:5.1f}% {len(d):6d} "
              f"{100 * m:+6.1f} +-{196 * se:4.1f} {sum(d):+7.1f}")


turn = lambda r: 'turn ' + (str(r['turn']) if r['turn'] <= 4 else '5+')
show('by turn and side:', lambda r: (turn(r), r['side']))
show('by phase and side:', lambda r: (r['phase'], r['side']))
show('turns 1-2, by kind:', lambda r: r['kind'] if r['turn'] <= 2 else None)
