#!/usr/bin/env python3
"""Shadow response dataset: one row per Shadow phase in the replay manifests -- what the Shadow player did with the
twilight the Free Peoples player handed them, next to what was knowable when the phase began.

usage: python3 -I scripts/shadow_response_data.py out.csv manifests-dir...   (e.g. ~/dev/gemp-replays/manifests-2022-*/movie)

Each game carries one event stream per player. Public events (PCIP into play, CAC, TP, GPC, PP, AT/RT, RCFP) are
in both; a player's own hand (PCIP zone HAND) and own discard pile (PCIP zone DISCARD) only in that player's. So:
  * everything public, the targets, and the "revealed" opponent profile are read from the FREE PEOPLES player's
    stream -- i.e. only what the FP player could have seen;
  * the Shadow player's actual hand at the start of the phase (a ceiling, not knowable to FP) is read from the
    SHADOW player's stream; the two are joined on the ordinal of the Shadow phase in the game (GPC phase 'Shadow'),
    and the twilight pool read from each must agree (it does; mismatching games are dropped and counted).

Event semantics used (verified by printing turns; see shadow_response_fit.py's docstring for the spot checks):
  TC participantId       -- the current (FP) player from here on
  PP index participantId -- a fellowship moved to site `index`; PPs of the current player after the turn's
                            'Fellowship' GPC are this turn's moves (setup PPs come before the first Fellowship GPC)
  TP count               -- twilight pool after a change (the last TP before GPC 'Shadow' is the pool at its start)
  GPC phase              -- 'Between turns','Fellowship','Shadow','Maneuver','Archery','Assignment','Skirmish',
                            'Regroup' (+ 'Fierce Assignment','Fierce Skirmish', 'Play starting fellowship')
  PCIP                   -- card put in a zone: blueprintId, cardId, participantId (owner), zone, targetCardId
  CAC                    -- a card's action was used / an event was played: blueprintId, cardId, participantId, zone
                            (VOID_FROM_HAND / VOID = events); public in both streams
  RCFP otherCardIds      -- cards leaving a zone (play, or a hand when a card is played/discarded)
  AT/RT token WOUND      -- wounds placed/removed; AT BURDEN on the Ring-bearer
  GS zones               -- zone sizes per player (HAND, DECK, DEAD, DISCARD)

Columns (one row per Shadow phase; fp = current player, sh = the other):
  id/meta   month game turn ordinal fp sh shCulture(the Shadow player's most-played Shadow culture over the game,
            for reporting only -- hindsight)
  base      twilight (pool at GPC 'Shadow'), site (fp's position = site number after moving), moves (fp's PPs this
            turn), fpComps, fpCompStr (printed strength of FP companions in play), fpCompVit (printed vitality - wounds),
            shHand (GS HAND of sh), carryMinions, carryStr (sh's minions already in play at the start of the phase)
  revealed  rv* -- distinct cards (by cardId) of sh's SHADOW side seen in public before the phase: PCIP by sh into
            SHADOW_CHARACTERS/SUPPORT/ATTACHED/FREE_CHARACTERS/DEAD/STACKED, or CAC/EP by sh (events played, abilities used).
            rvShadow (count), rvMinions, rvMinStr (sum printed strength), rvMinStrMax, rvMinCost (sum twilight cost),
            rvArcher, rvFierce, rvDamage (minions with Damage+N), rvEvents, rvConditions, rvPossessions,
            rv_<CULTURE> (Shadow cards of that culture), rvTurnsAsShadow (Shadow phases sh has already had)
  hand      hd* -- sh's hand at the phase start from sh's own stream (ceiling): hdMinions, hdMinStr, hdMinCost,
            hdShadow, hdAffStr (greedy: strength of the strongest-per-cost minions affordable within twilight +
            roaming ignored), hdAffMinions
  deck      dk* -- sh's decklist as far as the manifest knows it (decks[sh].deck non-null + fill: every card sh drew
            or showed all game -- ~51% of ids; hindsight): dkShadow, dkMinions, dkMinStrMean, dkMinCostMean,
            dkMinFrac (minions / shadow cards), dk_<CULTURE> fraction of shadow cards
  targets   yMinions (minions sh put into SHADOW_CHARACTERS during the phase), yStr (printed strength of all sh
            minions in play at the end of the phase), yStrPlayed (printed strength of minions played this phase),
            yStrPoss (yStr + printed strength of Shadow possessions attached to them), yTwilightEnd (pool at the
            phase's end), yKilled (FP companions put in DEAD from this phase until the next 'Regroup'/'Shadow'),
            yWounds (wounds placed on FP characters in that window, exertions included), yWoundsDealt (the same
            less wounds whose preceding log line says 'exert'), yBurdens (burdens added in that window)
"""
import glob, json, os, re, sys
from collections import Counter

CARD_DIR = 'gemp-lotr-cards/src/main/resources/cards/official'
SHADOW_CULTURES = ['DUNLAND', 'ISENGARD', 'MEN', 'MORIA', 'ORC', 'RAIDER', 'SAURON', 'URUK_HAI', 'WRAITH', 'GOLLUM']


def load_cards(root=CARD_DIR):
    """blueprint -> {type, side, culture, twilight, strength, vitality, keywords:set} from the HJSON definitions."""
    info = {}
    for f in glob.glob(os.path.join(root, '**/*.hjson'), recursive=True):
        cur = None
        in_kw = False
        for line in open(f, encoding='utf-8'):
            m = re.match(r'^\t(\d+_\d+)\s*:\s*\{', line)
            if m:
                cur = m.group(1)
                info[cur] = {'keywords': set()}
                in_kw = False
                continue
            if not cur:
                continue
            if in_kw:
                if re.match(r'^\t\t\]', line):
                    in_kw = False
                else:
                    kw = line.strip()
                    if kw:
                        info[cur]['keywords'].add(kw.lower())
                continue
            m = re.match(r'^\t\t(type|side|culture|twilight|strength|vitality|title):\s*(.+)', line)
            if m:
                info[cur][m.group(1)] = m.group(2).strip().lower()
                continue
            m = re.match(r'^\t\tkeywords:\s*(.*)', line)
            if m:
                v = m.group(1).strip()
                if v == '[':
                    in_kw = True
                elif v:
                    info[cur]['keywords'].add(v.lower())
    for c in info.values():
        c['culture'] = (c.get('culture') or '').upper().replace('-', '_')
        for k in ('twilight', 'strength', 'vitality'):
            c[k] = num(c.get(k))
    return info


def num(s, default=0):
    try:
        return int(s)
    except (TypeError, ValueError):
        return default


def base(bp):
    return re.sub(r'[*T]+$', '', bp or '')


def has_kw(c, prefix):
    return any(k.startswith(prefix) for k in c['keywords'])


PUBLIC_PCIP = {'SHADOW_CHARACTERS', 'SUPPORT', 'ATTACHED', 'FREE_CHARACTERS', 'DEAD', 'STACKED'}
IN_PLAY = {'FREE_CHARACTERS', 'SHADOW_CHARACTERS', 'SUPPORT', 'ATTACHED'}


def hand_snapshots(stream, me, info):
    """Ordinal of each Shadow phase where `me` is the Shadow player -> (hand composition dict, twilight).
    Walks `me`'s own stream: PCIP zone HAND (owner me) adds a card to the hand, RCFP / a PCIP elsewhere removes it."""
    hand = {}
    current, tw, ordinal = None, 0, 0
    zones = {}
    out = {}
    for kind, a, _ in stream:
        if kind == 'TC':
            current = a.get('participantId')
        elif kind == 'TP':
            tw = num(a.get('count'))
        elif kind == 'GS':
            try:
                zones = json.loads(a.get('zones') or '{}') or zones
            except ValueError:
                pass
        elif kind == 'PCIP':
            cid = a.get('cardId')
            if a.get('zone') == 'HAND' and a.get('participantId') == me:
                hand[cid] = base(a.get('blueprintId'))
            else:
                hand.pop(cid, None)
        elif kind == 'RCFP':
            for cid in str(a.get('otherCardIds', '')).split(','):
                hand.pop(cid, None)
        elif kind == 'GPC' and a.get('phase') == 'Shadow':
            ordinal += 1
            if current == me:
                continue
            h = dict(hdMinions=0, hdMinStr=0, hdMinCost=0, hdShadow=0, hdAffStr=0, hdAffMinions=0, hdSize=len(hand),
                     hdGs=num((zones.get(me) or {}).get('HAND')))
            mins = []
            for bp in hand.values():
                c = info.get(bp)
                if not c or c.get('side') != 'shadow':
                    continue
                h['hdShadow'] += 1
                if c.get('type') == 'minion':
                    h['hdMinions'] += 1
                    h['hdMinStr'] += c['strength']
                    h['hdMinCost'] += c['twilight']
                    mins.append((c['strength'], c['twilight']))
            budget = tw
            for s, t in sorted(mins, key=lambda x: -x[0] / max(1, x[1])):
                if t <= budget:
                    budget -= t
                    h['hdAffStr'] += s
                    h['hdAffMinions'] += 1
            out[ordinal] = (h, tw)
    return out


def deck_profile(deck, info):
    cards = [base(b) for b in (deck.get('deck') or []) if b] + [base(b) for b in (deck.get('fill') or [])]
    sh = [info[b] for b in cards if b in info and info[b].get('side') == 'shadow']
    mins = [c for c in sh if c.get('type') == 'minion']
    p = dict(dkShadow=len(sh), dkMinions=len(mins),
             dkMinStrMean=sum(c['strength'] for c in mins) / len(mins) if mins else 0,
             dkMinCostMean=sum(c['twilight'] for c in mins) / len(mins) if mins else 0,
             dkMinFrac=len(mins) / len(sh) if sh else 0)
    for cu in SHADOW_CULTURES:
        p['dk_' + cu] = sum(c['culture'] == cu for c in sh) / len(sh) if sh else 0
    return p


def fp_rows(stream, me, other, info, game, month, hands, deckp, sh_culture, stats):
    """Rows for the Shadow phases in which `me` is the FP player, from `me`'s stream (public info only)."""
    rows = []
    pos = {me: 1, other: 1}
    tw = 0
    zones = {}
    inplay = {}   # cardId -> dict(owner, c, target)
    wounds = {}
    revealed = {}  # cardId -> card info, Shadow-side cards of `other`
    current, turn, moves, ordinal, sh_turns = None, 0, 0, 0, 0
    last_msg = ''
    open_row = None   # row being filled during its Shadow phase
    window = None     # row whose skirmish window is open (until next Regroup/Shadow)
    for kind, a, _ in stream:
        if kind == 'M':
            last_msg = a.get('message', '')
        elif kind == 'TC':
            current = a.get('participantId')
        elif kind == 'PP':
            pos[a['participantId']] = num(a.get('index'), 1)
            if a['participantId'] == current and turn:
                moves += 1
        elif kind == 'TP':
            tw = num(a.get('count'))
        elif kind == 'GS':
            try:
                z = json.loads(a.get('zones') or '{}')
            except ValueError:
                z = {}
            if z:
                zones = z
        elif kind in ('PCIP', 'CAC', 'EP'):
            bp = base(a.get('blueprintId'))
            c = info.get(bp)
            owner = a.get('participantId')
            zone = a.get('zone')
            cid = a.get('cardId')
            if kind == 'PCIP' and zone not in ('HAND', 'DECK', 'DISCARD', 'ADVENTURE_PATH'):
                stats['seen'] += 1
                if c is None:
                    stats['unresolved'] += 1
            if c is None:
                continue
            if owner == other and c.get('side') == 'shadow' and (kind != 'PCIP' or zone in PUBLIC_PCIP):
                revealed.setdefault(cid, c)
            if kind == 'PCIP' and zone in IN_PLAY:
                inplay[cid] = {'owner': owner, 'c': c, 'target': a.get('targetCardId'), 'zone': zone}
                if open_row is not None and owner == other and zone == 'SHADOW_CHARACTERS' \
                        and c.get('type') == 'minion':
                    open_row['yMinions'] += 1
                    open_row['yStrPlayed'] += c['strength']
            if kind == 'PCIP' and zone == 'DEAD' and window is not None and owner == me \
                    and c.get('type') == 'companion':
                window['yKilled'] += 1
        elif kind == 'MCIP' and a.get('cardId') in inplay:
            inplay[a['cardId']]['target'] = a.get('targetCardId')
        elif kind == 'RCFP':
            gone = set(str(a.get('otherCardIds', '')).split(','))
            gone |= {cid for cid, x in inplay.items() if x['target'] in gone}
            for cid in gone:
                inplay.pop(cid, None)
                wounds.pop(cid, None)
        elif kind in ('AT', 'RT'):
            n = num(a.get('count'))
            tok = a.get('token')
            if tok == 'WOUND':
                cid = a.get('cardId')
                wounds[cid] = max(0, wounds.get(cid, 0) + (n if kind == 'AT' else -n))
                if kind == 'AT' and window is not None and a.get('participantId') == me:
                    window['yWounds'] += n
                    if ' exert' not in last_msg:
                        window['yWoundsDealt'] += n
            elif tok == 'BURDEN' and kind == 'AT' and window is not None:
                window['yBurdens'] += n
        elif kind == 'GPC':
            ph = a.get('phase')
            if open_row is not None:   # the Shadow phase just ended
                r = open_row
                r['yTwilightEnd'] = tw
                for cid, x in inplay.items():
                    if x['owner'] == other and x['zone'] == 'SHADOW_CHARACTERS' and x['c'].get('type') == 'minion':
                        r['yStr'] += x['c']['strength']
                        r['yStrPoss'] += x['c']['strength']
                for cid, x in inplay.items():
                    if x['owner'] == other and x['zone'] == 'ATTACHED' and x['c'].get('side') == 'shadow' \
                            and x['c'].get('type') in ('possession', 'artifact') and x['target'] in inplay \
                            and inplay[x['target']]['c'].get('type') == 'minion':
                        r['yStrPoss'] += x['c']['strength']
                open_row = None
                window = r   # the skirmish window runs from here to the next Regroup / Shadow
            if ph in ('Regroup', 'Between turns', 'Fellowship', 'Shadow'):
                window = None
            if ph == 'Fellowship':
                turn += 1
                moves = 0
            elif ph == 'Shadow':
                ordinal += 1
                if current != me:
                    continue
                stats['phases'] += 1
                hs = hands.get(ordinal)
                if hs is None:
                    stats['no_hand'] += 1
                    continue
                if hs[1] != tw:
                    stats['tw_mismatch'] += 1
                    continue
                r = dict(month=month, game=game, turn=turn, ordinal=ordinal, fp=me, sh=other, shCulture=sh_culture,
                         twilight=tw, site=pos[me], moves=moves)
                comps = [(cid, x) for cid, x in inplay.items() if x['owner'] == me and x['c'].get('type') == 'companion']
                r['fpComps'] = len(comps)
                r['fpCompStr'] = sum(x['c']['strength'] for _, x in comps)
                r['fpCompVit'] = sum(max(0, x['c']['vitality'] - wounds.get(cid, 0)) for cid, x in comps)
                r['shHand'] = num((zones.get(other) or {}).get('HAND'))
                carry = [x for x in inplay.values() if x['owner'] == other and x['zone'] == 'SHADOW_CHARACTERS'
                         and x['c'].get('type') == 'minion']
                r['carryMinions'] = len(carry)
                r['carryStr'] = sum(x['c']['strength'] for x in carry)
                rv = list(revealed.values())
                rm = [c for c in rv if c.get('type') == 'minion']
                r.update(rvShadow=len(rv), rvMinions=len(rm), rvMinStr=sum(c['strength'] for c in rm),
                         rvMinStrMax=max([c['strength'] for c in rm], default=0),
                         rvMinCost=sum(c['twilight'] for c in rm),
                         rvArcher=sum(has_kw(c, 'archer') for c in rm), rvFierce=sum(has_kw(c, 'fierce') for c in rm),
                         rvDamage=sum(has_kw(c, 'damage') for c in rm),
                         rvEvents=sum(c.get('type') == 'event' for c in rv),
                         rvConditions=sum(c.get('type') == 'condition' for c in rv),
                         rvPossessions=sum(c.get('type') in ('possession', 'artifact') for c in rv),
                         rvTurnsAsShadow=sh_turns)
                for cu in SHADOW_CULTURES:
                    r['rv_' + cu] = sum(c['culture'] == cu for c in rv)
                r.update(hs[0])
                r.update(deckp)
                r.update(yMinions=0, yStr=0, yStrPlayed=0, yStrPoss=0, yTwilightEnd=0, yKilled=0, yWounds=0, yWoundsDealt=0,
                         yBurdens=0)
                sh_turns += 1
                rows.append(r)
                open_row = r
    return rows


def main():
    info = load_cards()
    out_path, dirs = sys.argv[1], sys.argv[2:]
    stats = Counter()
    rows = []
    for d in dirs:
        month = re.search(r'2022-(\d\d)', d).group(1)
        for f in sorted(glob.glob(os.path.join(d, '*.json'))):
            g = json.load(open(f))
            players = g['players']
            if len(players) != 2 or any(p not in g['events'] for p in players):
                stats['skipped_games'] += 1
                continue
            stats['games'] += 1
            # the Shadow player's main culture over the whole game (for reporting by culture only)
            shc = {}
            for p in players:
                cnt = Counter()
                for kind, a, _ in g['events'][p]:
                    if kind in ('PCIP', 'CAC', 'EP') and a.get('participantId') == p:
                        c = info.get(base(a.get('blueprintId')))
                        if c and c.get('side') == 'shadow' and c['culture']:
                            cnt[c['culture']] += 1
                shc[p] = cnt.most_common(1)[0][0] if cnt else 'NONE'
            for me in players:
                other = players[1] if me == players[0] else players[0]
                hands = hand_snapshots(g['events'][other], other, info)
                deckp = deck_profile((g.get('decks') or {}).get(other) or {}, info)
                rows += fp_rows(g['events'][me], me, other, info, os.path.basename(f), month, hands, deckp,
                                shc[other], stats)
    cols = list(rows[0].keys())
    with open(out_path, 'w') as o:
        o.write(','.join(cols) + '\n')
        for r in rows:
            o.write(','.join(str(round(r[c], 4)) if isinstance(r[c], float) else str(r[c]) for c in cols) + '\n')
    print(f'{len(rows)} rows -> {out_path}; {dict(stats)}')


if __name__ == '__main__':
    main()
