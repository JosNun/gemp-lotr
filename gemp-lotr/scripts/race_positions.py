#!/usr/bin/env python3
"""Whole-game positions from replay manifests' event streams, for race_value.py: at the start of every turn, from
each player's side, its fellowship's site and the opponent's, companions in play on each side, each Ring-bearer's
resistance left, cards in hand and in the draw deck on each side, and wounds on each side's companions; then the
rest of the board, per owner (columns 16 on); labelled with whether that player won.

usage: race_positions.py out.csv manifests-dir...

HumanReplay --values only reaches the part of a game the engine re-creates (31% of decisions, half of the positions
from turns 1-3), so a win model fit on it barely sees the late race. The event stream covers every turn: PP (a
fellowship's site), PCIP/RCFP (cards into and out of play), MCIP (a card moved within play: a transferred
possession, a site changing controller), AT/RT on BURDEN, WOUND and culture tokens, GS (zone sizes), GPC (phase
changes), TC (whose turn), and M (log lines: the only place threats show up).

Columns, in order (every "my"/"opp" pair is from the row's player's side; a row is written for each player):
  won turn iAmFp myPosition oppPosition myCompanions oppCompanions myRbResistance oppRbResistance myHand oppHand
  myDeck oppDeck myWounds oppWounds                         -- the original 15
  myBurdens oppBurdens                                      -- burden tokens on each Ring-bearer
  myCompanionVitalityLeft oppCompanionVitalityLeft          -- sum over own companions of printed vitality - wounds
  myExhausted oppExhausted                                  -- own companions with printed vitality - wounds <= 1
  myAllies oppAllies                                        -- allies in play, by owner
  myFpPossessions oppFpPossessions                          -- Free Peoples possessions + artifacts, by owner
  myFpConditions oppFpConditions                            -- Free Peoples conditions, by owner
  myShadowPossessions oppShadowPossessions                  -- Shadow possessions + artifacts, by owner
  myShadowConditions oppShadowConditions                    -- Shadow conditions, by owner
  myMinions oppMinions                                      -- minions in play (left over from the last turn)
  myDead oppDead myDiscard oppDiscard                       -- GS zone sizes DEAD and DISCARD
  myThreats oppThreats                                      -- each player's threats (GameState keeps one count per
                                                               player; getThreats() is the current player's)
  myCultureTokens oppCultureTokens                          -- culture tokens on cards each player owns
Counts are of every card in play whatever its side, attached or in the support area, active or not (stacked cards
are not in play; the One Ring and sites are not counted). The live GeneralStateFeatures counts fpPossessions,
fpConditions, shadowConditions... only over *active* cards -- the current player's Free Peoples cards and the other
player's Shadow cards -- so race_value.py builds those from these per-owner columns, not the other way round.

Threats come from log lines, since there is no token for them: "<card> adds N(out of M) threat(s)" and
"<card> removed N threat(s)" change the current player's count (AddThreatsEffect/RemoveThreatsEffect both use
getCurrentPlayerId), and "<player> assigns N threat damage" (ThreatRule, after a companion or ally is killed) removes
N -- RemoveThreatsEffect with no source logs nothing of its own. Nothing resets them between turns.
Vitality is printed vitality less wounds; the engine's (ModifiersQuerying) also carries modifiers, e.g. +1 from
armour, which the stream does not say.
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
            m = re.match(r'^\t\t(type|side|resistance|vitality):\s*(.+)', line)
            if m:
                info[cur][m.group(1)] = m.group(2).strip().lower()

def base(bp):
    return re.sub(r'[*T]+$', '', bp or '')

def num(s, default=0):
    try:
        return int(s)
    except (TypeError, ValueError):
        return default

IN_PLAY = {'FREE_CHARACTERS', 'SHADOW_CHARACTERS', 'SUPPORT', 'ATTACHED'}
ADD_THREAT = re.compile(r' adds (\d+)(?:\(out of \d+\))? threats?$')
REMOVE_THREAT = re.compile(r' removed (\d+)(?:\(out of \d+\))? threats?$')
THREAT_DAMAGE = re.compile(r' assigns (\d+) threat damage$')
KINDS = ('comp', 'ally', 'minion', 'fpPoss', 'fpCond', 'shPoss', 'shCond')

def kind_of(t, s):
    if t == 'companion': return 'comp'
    if t == 'ally': return 'ally'
    if t == 'minion': return 'minion'
    if t in ('possession', 'artifact'): return 'fpPoss' if s == 'free peoples' else 'shPoss'
    if t == 'condition': return 'fpCond' if s == 'free peoples' else 'shCond'
    return None

out = open(sys.argv[1], 'w')
rows = games = unresolved = seen = 0
for d in sys.argv[2:]:
    for f in sorted(glob.glob(os.path.join(d, '*.json'))):
        g = json.load(open(f))
        if g.get('conceded') or g.get('winner') not in g['players']:
            continue
        players = g['players']
        stream = next(iter(g['events'].values()))  # either player's stream carries the public events
        pos = {p: 1 for p in players}
        burdens = {p: 0 for p in players}
        threats = {p: 0 for p in players}
        rb_res = {p: num(info.get(base(g['decks'][p].get('rb')), {}).get('resistance', 10), 10) or 10 for p in players}
        inplay = {}  # card id -> {owner, kind, printed vitality, card it is attached to}
        wounds = {}  # card id -> wounds on it
        culture = {}  # card id -> culture tokens on it
        zones = {p: {} for p in players}
        current, turn = None, 0
        for kind, a, _ in stream:
            if kind == 'TC':
                current = a.get('participantId')
            elif kind == 'PP':
                pos[a['participantId']] = int(a['index'])
            elif kind == 'PCIP' and a.get('zone') in IN_PLAY:
                ci = info.get(base(a.get('blueprintId')))
                seen += 1
                if ci is None:
                    unresolved += 1
                    continue
                k = kind_of(ci.get('type', ''), ci.get('side', ''))
                if k:
                    inplay[a['cardId']] = {'owner': a['participantId'], 'kind': k, 'vit': num(ci.get('vitality')),
                                           'target': a.get('targetCardId')}
            elif kind == 'MCIP' and a.get('cardId') in inplay:
                inplay[a['cardId']]['target'] = a.get('targetCardId')
            elif kind == 'RCFP':
                gone = set(str(a.get('otherCardIds', '')).split(','))
                # what was attached to a card leaving play goes with it (the stream nearly always says so too)
                gone |= {cid for cid, c in inplay.items() if c['target'] in gone}
                for cid in gone:
                    inplay.pop(cid, None)
                    wounds.pop(cid, None)
                    culture.pop(cid, None)
            elif kind in ('AT', 'RT'):
                n = num(a.get('count'))
                n = n if kind == 'AT' else -n
                tok = a.get('token')
                if tok == 'WOUND':
                    wounds[a['cardId']] = max(0, wounds.get(a['cardId'], 0) + n)
                elif tok == 'BURDEN':
                    burdens[a['participantId']] += n
                elif tok:
                    culture[a['cardId']] = max(0, culture.get(a['cardId'], 0) + n)
            elif kind == 'GS':
                try:
                    z = json.loads(a.get('zones') or '{}')
                except ValueError:
                    z = {}
                for p in players:
                    if p in z:
                        zones[p] = z[p]
            elif kind == 'M' and current in players:
                msg = a.get('message', '')
                if 'threat' in msg:
                    m = ADD_THREAT.search(msg)
                    if m:
                        threats[current] += int(m.group(1))
                    m = REMOVE_THREAT.search(msg) or THREAT_DAMAGE.search(msg)
                    if m:
                        threats[current] = max(0, threats[current] - int(m.group(1)))
            elif kind == 'GPC' and a.get('phase') == 'Fellowship' and current in players:
                turn += 1
                cnt = {p: dict.fromkeys(KINDS + ('hurt', 'vit', 'exh', 'cult'), 0) for p in players}
                for cid, c in inplay.items():
                    o = cnt.get(c['owner'])
                    if o is None:
                        continue
                    o[c['kind']] += 1
                    if c['kind'] == 'comp':
                        w = wounds.get(cid, 0)
                        left = max(0, c['vit'] - w)
                        o['hurt'] += w
                        o['vit'] += left
                        o['exh'] += left <= 1
                    o['cult'] += culture.get(cid, 0)
                z = {p: {k: num(zones[p].get(k)) for k in ('HAND', 'DECK', 'DEAD', 'DISCARD')} for p in players}
                for me in players:
                    opp = players[1] if me == players[0] else players[0]
                    m, o = cnt[me], cnt[opp]
                    vals = [1 if g['winner'] == me else 0, turn, 1 if current == me else 0, pos[me], pos[opp],
                            m['comp'], o['comp'], rb_res[me] - burdens[me], rb_res[opp] - burdens[opp],
                            z[me]['HAND'], z[opp]['HAND'], z[me]['DECK'], z[opp]['DECK'], m['hurt'], o['hurt'],
                            burdens[me], burdens[opp], m['vit'], o['vit'], m['exh'], o['exh'], m['ally'], o['ally'],
                            m['fpPoss'], o['fpPoss'], m['fpCond'], o['fpCond'], m['shPoss'], o['shPoss'],
                            m['shCond'], o['shCond'], m['minion'], o['minion'], z[me]['DEAD'], z[opp]['DEAD'],
                            z[me]['DISCARD'], z[opp]['DISCARD'], threats[me], threats[opp], m['cult'], o['cult']]
                    out.write(','.join(str(int(v)) for v in vals) + '\n')
                    rows += 1
        games += 1
print(f'{games} games, {rows} positions -> {sys.argv[1]}; {unresolved} of {seen} cards put into play not in the '
      f'card definitions')
