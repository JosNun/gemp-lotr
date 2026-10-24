#!/usr/bin/env python3
"""Whole-game positions from replay manifests' event streams, for race_value.py: at the start of every turn, from
each player's side, its fellowship's site and the opponent's, companions in play on each side, and each
Ring-bearer's resistance left; labelled with whether that player won.

usage: race_positions.py out.csv manifests-dir...

HumanReplay --values only reaches the part of a game the engine re-creates (31% of decisions, half of the positions
from turns 1-3), so a win model fit on it barely sees the late race. The event stream covers every turn: PP (a
fellowship's site), PCIP/RCFP (cards into and out of play), AT/RT on BURDEN tokens, GPC (phase changes), TC (whose turn).
"""
import glob, json, os, re, sys

info = {}
for f in glob.glob('gemp-lotr-cards/src/main/resources/cards/official/**/*.hjson', recursive=True):
    cur = None
    for line in open(f, encoding='utf-8'):
        m = re.match(r'^\t(\d+_\d+)\s*:\s*\{', line)
        if m:
            cur = m.group(1); info[cur] = {}
            continue
        if cur:
            m = re.match(r'^\t\t(type|resistance):\s*(.+)', line)
            if m:
                info[cur][m.group(1)] = m.group(2).strip()

def base(bp):
    return re.sub(r'[*T]+$', '', bp or '')

out = open(sys.argv[1], 'w')
rows = games = 0
for d in sys.argv[2:]:
    for f in glob.glob(os.path.join(d, '*.json')):
        g = json.load(open(f))
        if g.get('conceded') or g.get('winner') not in g['players']:
            continue
        players = g['players']
        stream = next(iter(g['events'].values()))  # either player's stream carries the public events
        pos = {p: 1 for p in players}
        burdens = {p: 0 for p in players}
        rb_res = {p: int(info.get(base(g['decks'][p].get('rb')), {}).get('resistance', 10) or 10) for p in players}
        rb_card = {}
        inplay = {}  # card id -> (owner, is companion)
        current, turn, started = None, 0, False
        for kind, a, _ in stream:
            if kind == 'TC':
                current = a.get('participantId')
            elif kind == 'PP':
                pos[a['participantId']] = int(a['index'])
            elif kind == 'PCIP' and a.get('zone') == 'FREE_CHARACTERS':
                t = info.get(base(a.get('blueprintId')), {}).get('type', '')
                inplay[a['cardId']] = (a['participantId'], t.lower() == 'companion')
            elif kind == 'RCFP':
                for cid in str(a.get('otherCardIds', '')).split(','):
                    inplay.pop(cid, None)
            elif kind in ('AT', 'RT') and a.get('token') == 'BURDEN':
                n = int(a.get('count', 0) or 0)
                burdens[a['participantId']] += n if kind == 'AT' else -n
            elif kind == 'GPC' and a.get('phase') == 'Fellowship' and current in players:
                turn += 1
                comps = {p: sum(1 for o, c in inplay.values() if o == p and c) for p in players}
                for me in players:
                    opp = players[1] if me == players[0] else players[0]
                    out.write(','.join(str(v) for v in (
                        1 if g['winner'] == me else 0, turn, 1 if current == me else 0, pos[me], pos[opp],
                        comps[me], comps[opp], rb_res[me] - burdens[me], rb_res[opp] - burdens[opp])) + '\n')
                    rows += 1
        games += 1
print(f'{games} games, {rows} positions -> {sys.argv[1]}')
