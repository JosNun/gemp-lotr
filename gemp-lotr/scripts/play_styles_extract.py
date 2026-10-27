#!/usr/bin/env python3
"""Per-(game, player) play-style metrics from replay manifests' event streams, for play_styles_cluster.py.

usage: play_styles_extract.py out.csv [manifests-root]   (root defaults to ~/dev/gemp-replays; reads
       manifests-2022-MM/movie/*.json for MM in 01..11; July (07) is the held-out month, flagged in column `heldout`)

One row per player per game. A player is the Free Peoples (FP) side on their own turns (TC names the current
player) and the Shadow side on the opponent's. Event kinds used (see race_positions.py): TC whose turn, GPC phase
changes (a turn starts at GPC Fellowship; a turn has one Shadow phase per move), PP a fellowship's site, PCIP card
into a zone (owner = participantId; zone DEAD = killed), TP twilight pool, AT/RT tokens (WOUND, BURDEN), EP an event
played, D a decision (each player's own stream carries that player's own decisions -- the count is the number of
training decisions a player-game gives), M log lines ("<P> plays <card> from hand", "Archery fire: fellowship - N,
minion - N", "<P> lost due to: ...", "<P> bid N", "Ring-bearer puts on The One Ring").

Columns are grouped:
  meta   game month heldout player opp won first conceded endReason (site9|rbDead|corrupted|concession|timeout|
         other, always the game's) concedeKind (for conceded games, the state at the end: race|corruption|attrition|
         unclear) turns myTurns oppTurns decisions decFp decSh fpCulture fpCulture2 shCulture shCulture2 deckKey
  fp_*   my FP turns: moves per turn, rate of saying yes to "another move", pace (sites per own turn), site by the
         end of own turn 4 (final site if the game ended first), companions at the start of each Fellowship phase
         (mean, max), starting companions, cards played per Fellowship phase, FP possessions/conditions/events/allies/
         companions played per own turn, twilight the FP added in Fellowship phases per turn (heavy fellowship),
         twilight in the pool when moving, healing (wound tokens removed from own characters) per turn, burdens
         taken per turn, bid, ring put on, FP archery per turn, own companions killed, mulligan
  sh_*   the opponent's turns: minions played per Shadow phase, printed strength / twilight cost of those minions,
         twilight available at the start of a Shadow phase and left at its end, fraction spent, Shadow conditions /
         possessions / events per Shadow phase, minions in play at Assignment, minion archery per turn and share of
         turns with any, opponent companions killed per opponent turn, wounds dealt, burdens added, FP cards
         discarded from play
Card facts (type, side, culture, strength, twilight, keywords) come from the official HJSON definitions.
"""
import glob, json, os, re, sys
from collections import Counter, defaultdict
from multiprocessing import Pool

ROOT = os.path.expanduser('~/dev/gemp-replays')
CARDS = 'gemp-lotr-cards/src/main/resources/cards/official'


def load_cards():
    info = {}
    for f in glob.glob(CARDS + '/**/*.hjson', recursive=True):
        cur, in_kw = None, False
        for line in open(f, encoding='utf-8'):
            m = re.match(r'^\t(\d+_\d+)\s*:\s*\{', line)
            if m:
                cur = m.group(1); info[cur] = {'kw': set()}; in_kw = False
                continue
            if not cur:
                continue
            if in_kw:
                if re.match(r'^\t\t\]', line):
                    in_kw = False
                else:
                    info[cur]['kw'].add(line.strip().split()[0].lower() if line.strip() else '')
                continue
            m = re.match(r'^\t\t(type|side|culture|strength|twilight|vitality|resistance|title):\s*(.+)', line)
            if m:
                info[cur][m.group(1)] = m.group(2).strip().lower()
                continue
            m = re.match(r'^\t\tkeywords?:\s*(.*)', line)
            if m:
                v = m.group(1).strip()
                if v == '[':
                    in_kw = True
                else:
                    info[cur]['kw'].add(v.split()[0].lower() if v else '')
    return info


INFO = load_cards()
IN_PLAY = {'FREE_CHARACTERS', 'SHADOW_CHARACTERS', 'SUPPORT', 'ATTACHED'}
HINT = re.compile(r"<div class='cardHint' value='([^']*)'>")
ARCHERY = re.compile(r'^Archery fire: fellowship - (\d+), minion - (\d+)')


def base(bp):
    return re.sub(r'[*T]+$', '', bp or '')


def num(s, d=0):
    try:
        return int(s)
    except (TypeError, ValueError):
        return d


def ci(bp):
    return INFO.get(base(bp))


def cultures(g, p, played):
    """Dominant FP culture(s) and Shadow culture(s): deck list as recorded plus every card seen played by p."""
    d = g['decks'].get(p) or {}
    bps = [b for b in (d.get('deck') or []) if b] + [b for b in (d.get('fromDeck') or []) if b] + played
    if d.get('rb'):
        bps.append(d['rb'])
    fp, sh = Counter(), Counter()
    for b in bps:
        c = ci(b)
        if not c or c.get('type') in ('site', 'the one ring') or 'culture' not in c:
            continue
        (fp if c.get('side') == 'free peoples' else sh)[c['culture']] += 1
    f = [k for k, _ in fp.most_common(2)] + ['', '']
    s = [k for k, _ in sh.most_common(2)] + ['', '']
    # a second culture only counts when it is a real share of the side (>= 25%)
    if f[1] and fp[f[1]] < 0.25 * sum(fp.values()):
        f[1] = ''
    if s[1] and sh[s[1]] < 0.25 * sum(sh.values()):
        s[1] = ''
    return f[0], f[1], s[0], s[1]


def end_reason(msgs):
    for m in reversed(msgs[-40:]):
        if ' lost due to: ' in m or ' is the winner due to: ' in m:
            r = m.split(' due to: ', 1)[1]
            if r.startswith('Surviving to Regroup phase on site 9'): return 'site9'
            if 'Ring-Bearer is dead' in r: return 'rbDead'
            if 'corrupted' in r: return 'corrupted'
            if r.startswith('Concession'): return 'concession'
            if 'timed-out' in r: return 'timeout'
            if r.startswith('Last remaining'): continue
            return 'other'
    return 'other'


def game_rows(path):
    try:
        g = json.load(open(path))
    except Exception:
        return []
    players = g.get('players') or []
    if len(players) != 2 or g.get('winner') not in players:
        return []
    gid = os.path.basename(path)[:-5]
    month = re.search(r'manifests-2022-(\d\d)', path).group(1)
    stream = max(g['events'].values(), key=len)
    P = players
    other = {P[0]: P[1], P[1]: P[0]}

    S = {p: defaultdict(float) for p in P}  # accumulators
    lst = {p: defaultdict(list) for p in P}
    owner = {}       # cardId -> (owner, info) for cards in play
    pos = {p: 1 for p in P}
    burd = {p: 0 for p in P}
    rbres = {p: num((ci((g['decks'].get(p) or {}).get('rb')) or {}).get('resistance'), 10) or 10 for p in P}
    played = {p: [] for p in P}
    msgs = []
    current, phase, turn = None, None, 0
    turns = Counter()
    site_t4 = {}
    last_msg = ''
    shadow_start_tp = None
    tp = 0
    another_pending = {p: False for p in P}
    for kind, a, x in stream:
        if kind == 'TC':
            current = a.get('participantId')
        elif kind == 'GPC':
            ph = a.get('phase')
            if phase == 'Shadow' and current in P and shadow_start_tp is not None:
                sp = other[current]
                lst[sp]['twAvail'].append(shadow_start_tp)
                lst[sp]['twLeft'].append(tp)
                shadow_start_tp = None
            if ph == 'Fellowship' and current in P:
                turns[current] += 1; turn += 1
                me = current
                comps = sum(1 for o, c in owner.values() if o == me and c.get('type') == 'companion')
                lst[me]['comps'].append(comps)
                if turns[me] == 1:
                    S[me]['startComps'] = comps
                S[me]['fellTpStart'] = tp
            if ph == 'Shadow' and current in P:
                S[other[current]]['shPhases'] += 1
                shadow_start_tp = tp
            if ph == 'Assignment' and current in P:
                sp = other[current]
                lst[sp]['minionsAtAssign'].append(
                    sum(1 for o, c in owner.values() if o == sp and c.get('type') == 'minion'))
            if ph == 'Between turns' and current in P:
                another_pending[current] = False
            phase = ph
        elif kind == 'TP':
            tp = num(a.get('count'))
        elif kind == 'PP':
            p = a.get('participantId')
            idx = num(a.get('index'), 1)
            if p in P and p == current and phase not in (None, 'Play starting fellowship') and turns[p] > 0:
                S[p]['moves'] += 1
                lst[p]['tpAtMove'].append(tp)
                if another_pending[p]:
                    S[p]['anotherYes'] += 1
                    another_pending[p] = False
                if phase == 'Fellowship':
                    S[p]['fellTwAdded'] += max(0, tp - S[p].get('fellTpStart', 0))
            if p in P:
                pos[p] = idx
                if turns[p] <= 4:
                    site_t4[p] = idx
        elif kind == 'D':
            # a stream carries its own player's decisions only; the other player's are counted below
            p = a.get('participantId')
            if p in P and a.get('text') == 'Do you want to make another move?':
                S[p]['anotherAsked'] += 1
                another_pending[p] = True
        elif kind == 'PCIP':
            z = a.get('zone'); p = a.get('participantId'); c = ci(a.get('blueprintId'))
            if z in IN_PLAY and c is not None and p in P:
                owner[a.get('cardId')] = (p, c)
                t, side = c.get('type'), c.get('side')
                if current in P and turns[current] > 0 and phase != 'Play starting fellowship':
                    if p == current:  # FP on own turn
                        if t in ('possession', 'artifact') and side == 'free peoples': S[p]['fpPoss'] += 1
                        elif t == 'condition' and side == 'free peoples': S[p]['fpCond'] += 1
                        elif t == 'companion': S[p]['fpCompsPlayed'] += 1
                        elif t == 'ally': S[p]['fpAllies'] += 1
                    else:  # Shadow on the opponent's turn
                        if t == 'minion':
                            S[p]['minions'] += 1
                            if phase == 'Shadow': S[p]['minionsInShadow'] += 1
                            lst[p]['minStr'].append(num(c.get('strength')))
                            lst[p]['minCost'].append(num(c.get('twilight')))
                            if 'archer' in c.get('kw', ()): S[p]['archerMinions'] += 1
                        elif t in ('possession', 'artifact') and side == 'shadow': S[p]['shPoss'] += 1
                        elif t == 'condition' and side == 'shadow': S[p]['shCond'] += 1
            elif z == 'DEAD' and p in P and c is not None and c.get('type') == 'companion':
                S[p]['compsLost'] += 1
                if current == p:
                    S[other[p]]['kills'] += 1
            if p in P and z in IN_PLAY | {'DEAD', 'DISCARD'} and c is not None and c.get('type') != 'site':
                played[p].append(a.get('blueprintId'))
        elif kind == 'RCFP':
            for cid in str(a.get('otherCardIds', '')).split(','):
                owner.pop(cid, None)
        elif kind == 'EP':
            p = a.get('participantId'); c = ci(a.get('blueprintId')) or {}
            if p in P and current in P and turns[current] > 0:
                if c.get('side') == 'shadow' and p != current: S[p]['shEvents'] += 1
                elif c.get('side') == 'free peoples' and p == current:
                    S[p]['fpEvents'] += 1
                    if phase == 'Skirmish': S[p]['fpSkirmishEvents'] += 1
        elif kind in ('AT', 'RT'):
            tok = a.get('token'); n = num(a.get('count')); cid = a.get('cardId'); p = a.get('participantId')
            if tok == 'BURDEN' and p in P:
                if kind == 'AT':
                    burd[p] += n
                    if turn > 0:  # the bid's burdens land during setup
                        S[p]['burdensTaken'] += n
                else:
                    burd[p] = max(0, burd[p] - n)
                    S[p]['burdensRemoved'] += n
            elif tok == 'WOUND' and p in P and current in P and turns[current] > 0:
                if kind == 'RT' and p == current and a.get('zone') == 'FREE_CHARACTERS' and 'gets killed' not in last_msg:
                    S[p]['healed'] += n
                elif kind == 'AT' and p == current and a.get('zone') == 'FREE_CHARACTERS' and ' wounded by ' in last_msg:
                    S[other[p]]['woundsDealt'] += n
        elif kind == 'M':
            m = a.get('message', '')
            msgs.append(m)
            last_msg = m
            if current in P and turns[current] > 0:
                mm = ARCHERY.match(m)
                if mm:
                    S[current]['fpArchery'] += int(mm.group(1))
                    sp = other[current]
                    S[sp]['shArchery'] += int(mm.group(2))
                    S[sp]['shArcheryAny'] += int(mm.group(2)) > 0
                    S[sp]['archeryPhases'] += 1
                for p in P:
                    if m.startswith(p + ' plays ') and ' from ' in m:
                        if p == current and phase == 'Fellowship':
                            S[p]['fellPlays'] += 1
                        elif p != current:
                            S[p]['shPlays'] += 1
                    elif m.startswith(p + ' discards ') and ' from play using ' in m and p != current:
                        S[p]['shDiscards'] += len(HINT.findall(m.split(' from play using ')[0]))
                if m == 'Ring-bearer puts on The One Ring':
                    S[current]['ringOn'] += 1
            else:
                for p in P:
                    if m.startswith(p + ' bid '):
                        S[p]['bid'] = num(m.split(' bid ')[1])
                    elif m == p + ' mulligans':
                        S[p]['mulligan'] = 1
                    elif m.startswith(p + ' has chosen to go '):
                        S[p]['wonBid'] = 1

    for p in P:  # decisions from each player's own stream, split by whose turn it was
        cur = None
        for kind, a, x in g['events'].get(p, []):
            if kind == 'TC':
                cur = a.get('participantId')
            elif kind == 'D' and a.get('participantId') == p:
                S[p]['decisions'] += 1
                S[p]['decFp' if cur == p else 'decSh'] += 1
                if stream is not g['events'][p] and a.get('text') == 'Do you want to make another move?':
                    S[p]['anotherAsked'] += 1
        if stream is not g['events'][p]:
            # "another move" yes/no for the player whose stream was not walked: every move past the first of a
            # turn followed such a question
            S[p]['anotherYes'] = max(0, S[p]['moves'] - turns[p])
    reason = end_reason(msgs)
    loser = other[g['winner']]
    conceded = bool(g.get('conceded')) or reason in ('concession', 'timeout')
    ckind = ''
    if conceded:
        w = g['winner']
        if pos[w] >= 8 or pos[w] >= pos[loser] + 2:
            ckind = 'race'
        elif rbres[loser] - burd[loser] <= 2:
            ckind = 'corruption'
        elif S[loser]['compsLost'] >= 3:
            ckind = 'attrition'
        else:
            ckind = 'unclear'
    rows = []
    for p in P:
        o = other[p]
        s, l = S[p], lst[p]
        mt, ot = max(turns[p], 1), max(turns[o], 1)
        shp = max(s['shPhases'], 1)
        avail, left = sum(l['twAvail']), sum(l['twLeft'])
        mean = lambda v: sum(v) / len(v) if v else float('nan')
        f1, f2, s1, s2 = cultures(g, p, played[p])
        r = dict(game=gid, month=month, heldout=int(month == '07'), player=p, opp=o, won=int(g['winner'] == p),
                 first=int(g.get('first') == p), conceded=int(conceded), endReason=reason, concedeKind=ckind,
                 turns=turn, myTurns=turns[p], oppTurns=turns[o], decisions=int(s['decisions']),
                 decFp=int(s['decFp']), decSh=int(s['decSh']),
                 fpCulture=f1, fpCulture2=f2, shCulture=s1, shCulture2=s2,
                 deckKey=f1 + '|' + '+'.join(sorted(x for x in (s1, s2) if x)),
                 fp_movesPerTurn=s['moves'] / mt,
                 fp_moveAgainRate=s['anotherYes'] / s['anotherAsked'] if s['anotherAsked'] else float('nan'),
                 fp_pace=(pos[p] - 1) / mt, fp_finalSite=pos[p], fp_siteT4=site_t4.get(p, pos[p]),
                 fp_compsMean=mean(l['comps']), fp_compsMax=max(l['comps']) if l['comps'] else 0,
                 fp_startComps=s['startComps'], fp_fellPlays=s['fellPlays'] / mt,
                 fp_poss=s['fpPoss'] / mt, fp_cond=s['fpCond'] / mt, fp_events=s['fpEvents'] / mt,
                 fp_skirmishEvents=s['fpSkirmishEvents'] / mt, fp_compsPlayed=s['fpCompsPlayed'] / mt,
                 fp_allies=s['fpAllies'] / mt, fp_fellTwAdded=s['fellTwAdded'] / mt, fp_tpAtMove=mean(l['tpAtMove']),
                 fp_healed=s['healed'] / mt, fp_burdensPerTurn=s['burdensTaken'] / mt,
                 fp_burdensRemoved=s['burdensRemoved'] / mt, fp_bid=s['bid'], fp_wonBid=s['wonBid'],
                 fp_ringOn=s['ringOn'] / mt, fp_archery=s['fpArchery'] / mt, fp_compsLost=s['compsLost'],
                 fp_mulligan=s['mulligan'],
                 sh_minionsPerPhase=s['minionsInShadow'] / shp, sh_minionsPerTurn=s['minions'] / ot,
                 sh_minStr=mean(l['minStr']), sh_minCost=mean(l['minCost']),
                 sh_twAvail=mean(l['twAvail']), sh_twLeft=mean(l['twLeft']),
                 sh_spentFrac=1 - left / avail if avail else float('nan'),
                 sh_cond=s['shCond'] / shp, sh_poss=s['shPoss'] / shp, sh_events=s['shEvents'] / ot,
                 sh_plays=s['shPlays'] / ot, sh_minionsAtAssign=mean(l['minionsAtAssign']),
                 sh_archery=s['shArchery'] / ot,
                 sh_archeryAny=s['shArcheryAny'] / s['archeryPhases'] if s['archeryPhases'] else float('nan'),
                 sh_archerMinions=s['archerMinions'] / ot,
                 sh_kills=s['kills'] / ot, sh_woundsDealt=s['woundsDealt'] / ot,
                 sh_discards=s['shDiscards'] / ot)
        # burdens the opponent took during my Shadow turns are my corruption pressure: approximate with theirs/turn
        r['sh_burdensAdded'] = S[o]['burdensTaken'] / ot
        rows.append(r)
    return rows


def main():
    out = sys.argv[1]
    root = sys.argv[2] if len(sys.argv) > 2 else ROOT
    files = sorted(f for mm in range(1, 12)
                   for f in glob.glob(f'{root}/manifests-2022-{mm:02d}/movie/*.json'))
    rows = []
    with Pool(max(1, os.cpu_count() - 2)) as pool:
        for i, rs in enumerate(pool.imap_unordered(game_rows, files, chunksize=16)):
            rows.extend(rs)
            if i % 2000 == 0:
                print(i, len(files), file=sys.stderr)
    cols = list(rows[0].keys())
    with open(out, 'w') as f:
        f.write('\t'.join(cols) + '\n')
        for r in rows:
            f.write('\t'.join(str(r[c]) if not isinstance(r[c], float) else f'{r[c]:.4f}' for c in cols) + '\n')
    print(f'{len(files)} files, {len(rows)} player-games -> {out}')


if __name__ == '__main__':
    main()
