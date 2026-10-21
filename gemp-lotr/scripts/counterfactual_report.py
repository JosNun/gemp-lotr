"""Where does the bot decide worse than people? Reads HumanReplay --cf-out files (Counterfactual).

    python3 scripts/counterfactual_report.py cf.tsv [more.tsv ...] [--min 30]

Per kind of decision: how often the bot's answer was the human's; where it was not, the deciding player's win rate
after the human's answer and after the bot's (each played out by the bot for both players, on the same futures), and
the difference with a 95% interval. "Net" is the difference summed over the decisions -- where the most is lost.
"""
import collections, math, sys

argv = sys.argv[1:]
MIN = 30
PLAYERS = None
if '--players' in argv:
    i = argv.index('--players'); PLAYERS = argv[i + 1]; del argv[i:i + 2]
if '--min' in argv:
    i = argv.index('--min'); MIN = int(argv[i + 1]); del argv[i:i + 2]
args = argv
rows = []
for f in args:
    for line in open(f, encoding='utf-8'):
        p = line.rstrip('\n').split('\t')
        if len(p) < 9: continue
        rows.append((p[2], int(p[6]), int(p[7]), int(p[8]), p[1]))

def group(key):
    g = collections.defaultdict(lambda: {'asked': 0, 'agree': 0, 'diffs': [], 'h': 0.0, 'b': 0.0})
    for bucket, h, b, n, *rest in rows:
        k = key(bucket) if not rest or key.__code__.co_argcount == 1 else key(bucket, rest[0])
        g[k]['asked'] += 1
        if n == 0:
            g[k]['agree'] += 1
        else:
            g[k]['diffs'].append((b - h) / n)
            g[k]['h'] += h / n
            g[k]['b'] += b / n
    return g

def show(title, g):
    print(title)
    print(f"  {'decision':48s} {'asked':>6s} {'agree':>6s} {'played':>6s} {'human':>6s} {'bot':>6s} {'bot-human':>14s} {'net':>7s}")
    items = []
    for k, v in g.items():
        d = v['diffs']
        if v['asked'] < MIN: continue
        m = sum(d) / len(d) if d else 0
        se = math.sqrt(sum((x - m) ** 2 for x in d) / max(1, len(d) - 1) / max(1, len(d))) if len(d) > 1 else 0
        items.append((sum(d), k, v, m, se))
    for net, k, v, m, se in sorted(items):
        nd = max(1, len(v['diffs']))
        print(f"  {k[:48]:48s} {v['asked']:6d} {100*v['agree']/v['asked']:5.1f}% {len(v['diffs']):6d} "
              f"{100*v['h']/nd:5.1f}% {100*v['b']/nd:5.1f}% {100*m:+6.1f} +-{196*se:4.1f} {net:+7.1f}")

all_d = [(b - h) / n for _, h, b, n, _ in rows if n > 0]
hw = sum(h / n for _, h, b, n, _ in rows if n > 0) / max(1, len(all_d))
bw = sum(b / n for _, h, b, n, _ in rows if n > 0) / max(1, len(all_d))
m = sum(all_d) / max(1, len(all_d))
se = math.sqrt(sum((x - m) ** 2 for x in all_d) / max(1, len(all_d) - 1) / max(1, len(all_d)))
print(f"{len(rows)} decisions asked, bot agreed on {sum(1 for r in rows if r[3] == 0)} "
      f"({100*sum(1 for r in rows if r[3] == 0)/max(1,len(rows)):.1f}%); {len(all_d)} disagreements played out:")
print(f"  deciding player's win rate after the human's answer {100*hw:.1f}%, after the bot's {100*bw:.1f}%: "
      f"bot - human {100*m:+.1f} +- {196*se:.1f} points\n")
show('by kind of decision (worst for the bot first):', group(lambda b: b))
print()
show('by decision type:', group(lambda b: b.split(':')[0].split(' (')[0]))

if PLAYERS:
    # the human's skill: their 2022 Movie Block win rate (20+ games), in tiers
    import json
    rec = json.load(open(PLAYERS))
    def tier(player):
        w, g = rec.get(player, [0, 0])
        if g < 20: return 'unrated (<20 games)'
        r = w / g
        return 'strong (>=60% wins)' if r >= 0.6 else 'middling (45-60%)' if r >= 0.45 else 'weak (<45%)'
    print()
    show("by the human's skill:", group(lambda b, pl: tier(pl)))
