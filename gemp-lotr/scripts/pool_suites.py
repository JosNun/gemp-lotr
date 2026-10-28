#!/usr/bin/env python3
"""Suites for testing against the pool of style bots: for each style, 50 matchups of a test-suite deck (for the bot
being tested, deck 1) against a deck of the kind that style's people played (for the pool bot, deck 2). Run them with
-Dbvb.fixedDecks=true so the pool bot keeps its deck.

usage: pool_suites.py  -> decks/real/pool-<style>.txt

A deck's kind: its main Free Peoples culture, main Shadow culture and the mean twilight cost of its minions
(scripts/play_styles_cluster.py's profiles: racer half Elven; wide fellowship 63% Shire; builder Gondor and Gandalf;
big minions cost 5.0 against 4.1; swarm-grinder 3.3).
"""
import collections, csv, glob, random, re

info = {}
for f in glob.glob('gemp-lotr-cards/src/main/resources/cards/official/**/*.hjson', recursive=True):
    cur = None
    for line in open(f, encoding='utf-8'):
        m = re.match(r'^\t(\d+_\d+)\s*:\s*\{', line)
        if m:
            cur = m.group(1); info[cur] = {}
            continue
        if cur:
            m = re.match(r'^\t\t(culture|side|type|twilight):\s*(.+)', line)
            if m:
                info[cur][m.group(1)] = m.group(2).strip().lower()

games = {r['id']: int(r['games']) for r in csv.DictReader(open('decks/real/movie.csv'))}
kinds = {}
for path in glob.glob('decks/real/movie/*.txt'):
    did = path.split('/')[-1][:-4]
    parts = open(path).read().strip().split('|')
    cards = [re.sub(r'[*T]+$', '', c) for c in parts[-1].split(',') if c]
    fp = collections.Counter(info.get(c, {}).get('culture') for c in cards if info.get(c, {}).get('side') == 'free peoples')
    sh = collections.Counter(info.get(c, {}).get('culture') for c in cards if info.get(c, {}).get('side') == 'shadow')
    costs = [int(info[c]['twilight']) for c in cards if info.get(c, {}).get('type') == 'minion'
             and info[c].get('twilight', '').isdigit()]
    if not fp or not sh or len(costs) < 6:
        continue
    kinds[did] = (fp.most_common(1)[0][0], sh.most_common(1)[0][0], sum(costs) / len(costs), path)

styles = {
    'racer': lambda k: k[0] == 'elven',
    'wide-fellowship': lambda k: k[0] == 'shire',
    'builder': lambda k: k[0] in ('gondor', 'gandalf'),
    'big-minions': lambda k: k[2] >= 4.8 and k[1] in ('sauron', 'wraith', 'raider'),
    'swarm-grinder': lambda k: k[2] <= 3.6 and k[0] in ('dwarven', 'elven', 'gondor', 'rohan'),
}
test = []
for line in open('decks/real/suite-test.txt'):
    if line.startswith('#') or '|' not in line:
        continue
    test.append(line.split('|')[1].strip())
rng = random.Random(7)
for style, ok in styles.items():
    pool = sorted((d for d, k in kinds.items() if ok(k) and games.get(d, 0) >= 3), key=lambda d: -games.get(d, 0))[:60]
    cands = rng.sample(test, 50)
    with open(f'decks/real/pool-{style}.txt', 'w') as out:
        out.write(f'# {style}: test decks (bot under test) against {len(pool)} {style} decks (pool bot); -Dbvb.fixedDecks\n')
        for i, c in enumerate(cands):
            out.write(f'movie | {c} | {kinds[pool[i % len(pool)]][3]}\n')
    mix = collections.Counter(kinds[d][:2] for d in pool)
    print(f'{style:16} {len(pool)} decks, mean minion cost {sum(kinds[d][2] for d in pool) / len(pool):.1f}, '
          f'{", ".join(f"{a}|{b} {n}" for (a, b), n in mix.most_common(4))}')
