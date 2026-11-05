"""How people played, game by game, in the shape BotVsBot's -Dbvb.gameLog writes for the bots (GameLog.java), so the
two can be set side by side (scripts/style_compare.py).

usage: python3 -I scripts/people_style.py <out.jsonl> <manifest dir>...

From each game's event stream (the first player's: what both saw): who went first and who won, how (the end
message), turns, and per player the site reached, companions dead, "move again?" asked and taken (taken: the player
moved again before the turn ended), Shadow turns and minions played on them (a minion entering play on the
opponent's turn), the twilight left when each Shadow phase ended, and the cards played by phase. Passes are not here:
a replay records the prompt, not the answer.
"""
import collections, glob, json, os, re, sys

MOVE = 'Do you want to make another move?'
PLAY_ZONES = {'FREE_CHARACTERS', 'SHADOW_CHARACTERS', 'SUPPORT', 'ATTACHED'}


def route(reason):
    r = (reason or '').lower()
    if 'surviving to regroup' in r or 'site 9' in r or 'site 6' in r:
        return 'site9'
    if 'corrupt' in r:
        return 'corrupt'
    if 'ring-bearer is dead' in r or 'ring-bearer dead' in r:
        return 'kill'
    return 'other'


def game(m):
    players = m['players']
    p0 = players[0]
    stream = m['events'][p0]
    pos = {p: 1 for p in players}
    st = {p: dict(moveAsked=0, moveYes=0, shadowTurns=0, minions=0, unspent=[], at_end=[], played=collections.Counter(),
                  deadCompanions=0) for p in players}
    in_play = {}  # minion card id -> owner, while it is in play
    minion_bps = set()
    turn, current, phase = 0, None, None
    pending_move = None  # the player asked "move again?" this turn, until they move or the turn ends
    reason = None
    twilight = 0
    for kind, at, params in stream:
        if kind == 'TC':
            # setup (bids, starting fellowships, mulligans) changes the current player ~6 times before turn 1: a
            # turn is counted when its Fellowship phase starts, as the engine's turn number counts real turns only
            current = at.get('participantId')
            pending_move = None
        elif kind == 'GPC':
            new = at.get('phase', '')
            if new == 'Fellowship':
                turn += 1
                for p in players:
                    if p != current:
                        st[p]['shadowTurns'] += 1
            if phase == 'Shadow' and new != 'Shadow':
                shadow = next((p for p in players if p != current), None)
                if shadow:
                    st[shadow]['unspent'].append(twilight)
                    st[shadow]['at_end'].append(sum(1 for o in in_play.values() if o == shadow))
            phase = new
        elif kind == 'TP':
            try:
                twilight = int(at.get('count', 0))
            except ValueError:
                pass
        elif kind == 'PP':
            p = at.get('participantId')
            if p in pos:
                if pending_move == p:
                    st[p]['moveYes'] += 1
                    pending_move = None
                pos[p] = int(at.get('index', pos[p]))
        elif kind == 'D' and at.get('text') == MOVE:
            # in the asking player's own stream only: the first player's asks here; the other's below
            if at.get('participantId') in st:
                st[at['participantId']]['moveAsked'] += 1
                pending_move = at['participantId']
        elif kind == 'RCFP':
            for cid in at.get('otherCardIds', '').split(','):
                in_play.pop(cid, None)
        elif kind == 'PCIP':
            p, zone = at.get('participantId'), at.get('zone')
            cid = at.get('cardId')
            if zone == 'SHADOW_CHARACTERS' and p in st:
                in_play[cid] = p
            elif cid in in_play:  # put somewhere else (discard, hand): no longer in play
                in_play.pop(cid)
            if p in st and zone in PLAY_ZONES and turn > 0 and phase not in (None, 'Play starting fellowship'):
                st[p]['played'][(phase or '?').split()[0].lower()] += 1
                if zone == 'SHADOW_CHARACTERS' and p != current:
                    st[p]['minions'] += 1
        elif kind == 'GS':
            try:
                z = json.loads(at['zones'])
                for p in players:
                    st[p]['deadCompanions'] = int(z.get(p, {}).get('DEAD', st[p]['deadCompanions']))
            except (KeyError, ValueError):
                pass
        elif kind == 'M':
            msg = at.get('message', '')
            mm = re.match(r'(.+) lost due to: (.+)$', msg) or re.match(r'(.+) is the winner due to: (.+)$', msg)
            if mm and reason is None:
                reason = mm.group(2)
    # the second player's "move again?" prompts are only in their own stream
    p1 = players[1]
    asked = yes = 0
    pend = False
    for kind, at, _ in m['events'].get(p1, []):
        if kind == 'TC':
            pend = False
        elif kind == 'D' and at.get('text') == MOVE and at.get('participantId') == p1:
            asked += 1
            pend = True
        elif kind == 'PP' and at.get('participantId') == p1 and pend:
            yes += 1
            pend = False
    st[p1]['moveAsked'], st[p1]['moveYes'] = asked, yes
    out = {'winner': m.get('winner'), 'first': m.get('first'), 'reason': reason, 'route': route(reason),
           'turns': turn, 'conceded': bool(m.get('conceded')), 'players': {}}
    for p in players:
        s = st[p]
        out['players'][p] = {'site': pos[p], 'deadCompanions': s['deadCompanions'], 'moveAsked': s['moveAsked'],
                             'moveYes': s['moveYes'], 'shadowTurns': s['shadowTurns'], 'minions': s['minions'],
                             'unspentTwilight': sum(s['unspent']) / len(s['unspent']) if s['unspent'] else None,
                             'minionsAtShadowEnd': sum(s['at_end']) / len(s['at_end']) if s['at_end'] else None,
                             'played': dict(s['played'])}
    return out


def main():
    out, dirs = sys.argv[1], sys.argv[2:]
    n = 0
    with open(out, 'w') as f:
        for d in dirs:
            for path in sorted(glob.glob(os.path.join(d, '*.json'))):
                try:
                    g = game(json.load(open(path)))
                except (KeyError, ValueError, IndexError, TypeError):
                    continue
                f.write(json.dumps(g) + '\n')
                n += 1
    print(f'{n} games', file=sys.stderr)


if __name__ == '__main__':
    main()
