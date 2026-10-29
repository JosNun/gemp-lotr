#!/usr/bin/env python3
"""Elo ratings of 2022's Movie Block players from their results, for weighting and judging by skill.

usage: player_ratings.py out.tsv

Games are the replay manifests (~/dev/gemp-replays/manifests-2022-*/movie), in the order they started (StartTime from
the season's summaries zip); a concession is a loss. K = 24, everyone starts at 1500. Writes player, games, wins and
final rating.
"""
import glob, json, os, sys, zipfile, collections

summaries = os.path.expanduser('~/dev/gemp-replays/dl-2022/GEMP-LOTR-Replay_Summaries_2022.zip')
start = {}
with zipfile.ZipFile(summaries) as z:
    for name in z.namelist():
        if '/movie' in name.lower().replace(' ', '') or 'Movie' in name:
            if name.endswith('.json'):
                try:
                    start[os.path.basename(name)] = json.loads(z.read(name)).get('StartTime', '')
                except ValueError:
                    pass
games = []
for f in glob.glob(os.path.expanduser('~/dev/gemp-replays/manifests-2022-*/movie/*.json')):
    g = json.load(open(f))
    if g.get('winner') not in g['players']:
        continue
    games.append((start.get(g.get('summary', ''), ''), f[-60:], g['players'], g['winner']))
games.sort()
rating = collections.defaultdict(lambda: 1500.0)
played, won = collections.Counter(), collections.Counter()
for _, _, players, winner in games:
    a, b = players
    ea = 1 / (1 + 10 ** ((rating[b] - rating[a]) / 400))
    sa = 1.0 if winner == a else 0.0
    rating[a] += 24 * (sa - ea)
    rating[b] += 24 * ((1 - sa) - (1 - ea))
    for p in players:
        played[p] += 1
    won[winner] += 1
with open(sys.argv[1], 'w') as out:
    out.write('player\tgames\twins\trating\n')
    for p in sorted(rating, key=lambda p: -rating[p]):
        out.write(f'{p}\t{played[p]}\t{won[p]}\t{rating[p]:.0f}\n')
timed = sum(1 for t, *_ in games if t)
print(f'{len(games)} games ({timed} with a start time), {len(rating)} players')
