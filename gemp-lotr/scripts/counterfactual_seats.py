#!/usr/bin/env python3
"""HumanReplay --cf-deaths rows split by seat: where the bot differs from the human, its answer minus the human's in
Free Peoples companions and Ring-bearers killed that turn, by side, by whether the deciding player went first, and by
kind of decision.

usage: counterfactual_seats.py cf.tsv manifests-dir

Free Peoples rows: positive is the bot losing more of its own companions than the human did. Shadow rows: positive is
the bot killing more.
"""
import collections, glob, json, math, os, sys

first = {}
for f in glob.glob(os.path.join(sys.argv[2], '*.json')):
    g = json.load(open(f))
    first[os.path.basename(f)[:-5]] = g.get('first')
diff = collections.defaultdict(lambda: ([], []))
asked = collections.Counter()
for line in open(sys.argv[1], encoding='utf-8'):
    p = line.rstrip('\n').split('\t')
    if len(p) < 12 or p[0] not in first:
        continue
    seat = 'first' if first[p[0]] == p[1] else 'second'
    n = int(p[8])
    for key in [(p[9], seat, 'all'), (p[9], seat, p[2][:44])]:
        asked[key] += 1
        if n > 0:
            diff[key][0].append((int(p[7]) - int(p[6])) / n)
            diff[key][1].append((int(p[11]) - int(p[10])) / n)

def mean_ci(v):
    m = sum(v) / len(v)
    return m, 1.96 * math.sqrt(sum((x - m) ** 2 for x in v) / max(1, len(v) - 1) / len(v))

print('bot minus human where they differ, companions / Ring-bearers killed that turn (95% intervals)')
for key in sorted(asked, key=lambda k: (k[0], k[2] != 'all', k[2], k[1])):
    a, b = diff[key]
    if len(a) < 30:
        continue
    ma, ca = mean_ci(a)
    mb, cb = mean_ci(b)
    total = sum(a)
    print(f'  {key[0]:6} {key[1]:6} {key[2]:44} asked {asked[key]:6d} differ {len(a):5d}  companions {ma:+.3f} +- {ca:.3f}'
          f' (sum {total:+.0f})  Ring-bearer {mb:+.3f} +- {cb:.3f}')
