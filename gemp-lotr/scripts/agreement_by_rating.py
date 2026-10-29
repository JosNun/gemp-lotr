#!/usr/bin/env python3
"""Agreement with people by the deciding player's rating tier, from HumanReplay --cf-rollouts 0 files.

usage: agreement_by_rating.py ratings.tsv cf.tsv... [--only REGEX]
Tiers by rating among players with 20+ games (quartiles), and "few games" for the rest.
"""
import csv, re, sys, collections
args = sys.argv[1:]
only = None
if '--only' in args:
    i = args.index('--only'); only = re.compile(args[i + 1]); args = args[:i] + args[i + 2:]
ratings = {r['player']: (float(r['rating']), int(r['games'])) for r in csv.DictReader(open(args[0]), delimiter='\t')}
reg = sorted(v[0] for v in ratings.values() if v[1] >= 20)
q = [reg[int(len(reg) * x)] for x in (0.25, 0.5, 0.75)]
def tier(p):
    r, n = ratings.get(p, (1500, 0))
    if n < 20: return '0 few games'
    return '1 bottom quarter' if r < q[0] else '2 second' if r < q[1] else '3 third' if r < q[2] else '4 top quarter'
for path in args[1:]:
    t = collections.Counter(); a = collections.Counter()
    for line in open(path, encoding='utf-8'):
        f = line.rstrip('\n').split('\t')
        if len(f) < 9 or (only and not only.search(f[2])): continue
        k = tier(f[1]); t[k] += 1; a[k] += f[8] == '0'
    print(path.split('/')[-1], ' '.join(f'{k[2:]}: {a[k]/t[k]:.1%} of {t[k]}' for k in sorted(t)))
