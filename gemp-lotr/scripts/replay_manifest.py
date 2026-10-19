"""Turn GEMP's 2020-2022 replay files into per-game manifests for HumanReplay (Java).

    python3 -I scripts/replay_manifest.py <extract dir> <out dir> [--format "Fellowship block"] [--limit N]

<extract dir> holds `replays/<player>_<recordingId>.xml` and `<yyyy>/<mm>/summaries/<format>/*.json`, as the
GEMP-LOTR-Replays_<month>.zip and GEMP-LOTR-Replay_Summaries_<year>.zip unpack. The files are untrusted: they are
only parsed (ElementTree, json), never executed; run with -I.

A manifest is one JSON per game:
  players      -- the two player names in card-id order (the first one's cards were created first)
  bids, first  -- each player's bid and who went first
  decks        -- per player: idBase, sites[9], deck[N] (blueprint per original card id, null where never seen),
                  rb, ring, fromDeck (cards played straight from the deck), fill (blueprints the unseen ids must
                  include: plays from the deck and deck-search listings no seen id accounts for)
  events       -- per player, the replay's events with GS (stats) dropped: [type, {attrs}, [[param, value]...]]
  departures   -- per player, [event index, card id] for every card seen leaving that player's draw deck
                  (draws, mostly), in order: what the engine's shuffles have to put on top

How card ids work in the 2020 server, which the engine still does: every card gets an id as it is created, player
by player (HashMap order of the names), sites first, then the draw deck in deck-list order, then the Ring-bearer and
the Ring; a card gets a fresh id each time it enters play. So a card's original id is its place in its owner's deck
list, and a card that is drawn keeps that id in hand -- the owner's replay shows it (PCIP zone HAND).
"""
import glob
import json
import os
import re
import sys
import xml.etree.ElementTree as ET

HINT = re.compile(r"value='([^']+)'")


def load_events(path):
    out = []
    for e in ET.parse(path).getroot():
        t = e.get('type')
        if t == 'GS':
            # keep the zone sizes of the first few: they give each deck's size before anything was drawn
            zones = {z.get('name'): dict(z.attrib) for z in e.findall('playerZones')}
            out.append(['GS', {'zones': json.dumps(zones)}, []])
            continue
        attrs = {k: v for k, v in e.attrib.items() if k != 'type'}
        params = [[p.get('name'), p.get('value')] for p in e.findall('parameter')]
        out.append([t, attrs, params])
    return out


def first_zones(events):
    for t, a, _ in events:
        if t == 'GS':
            z = json.loads(a['zones'])
            if z:
                return z
    return None


def build(summary_path, replay_dir):
    s = json.load(open(summary_path))
    names = list(s['Players'].values())
    if len(names) != 2 or s.get('Winner') not in names:
        return None, 'not two players'
    winner = s['Winner']
    loser = [n for n in names if n != winner][0]
    files = {winner: os.path.join(replay_dir, f"{winner}_{s['WinnerReplayID']}.xml"),
             loser: os.path.join(replay_dir, f"{loser}_{s['LoserReplayID']}.xml")}
    if not all(os.path.exists(f) for f in files.values()):
        return None, 'replay file missing'
    events = {p: load_events(f) for p, f in files.items()}
    zones = first_zones(events[winner])
    if not zones or any(p not in zones for p in names):
        return None, 'no zone sizes'

    msgs = [a.get('message', '') for t, a, _ in events[winner] if t == 'M']
    order = None
    for m in msgs:
        mm = re.match(r'Players in the game are: (.+), (.+)$', m)
        if mm:
            order = [mm.group(1), mm.group(2)]
            break
    bids, first = {}, None
    for m in msgs:
        mm = re.match(r'(.+) bid (\d+)$', m)
        if mm and mm.group(1) in names:
            bids[mm.group(1)] = int(mm.group(2))
        mm = re.match(r'(.+) has chosen to go (first|second)$', m)
        if mm and mm.group(1) in names:
            first = mm.group(1) if mm.group(2) == 'first' else [n for n in names if n != mm.group(1)][0]
    if first is None or len(bids) != 2:
        return None, 'no bids or seating'

    sizes = {p: (int(zones[p]['ADVENTURE_DECK']), int(zones[p]['DECK'])) for p in names}

    # Which player's cards were created first? Their own hand ids tell: try both orders, keep the consistent one.
    def layout(first_p):
        second_p = [n for n in names if n != first_p][0]
        a1, n1 = sizes[first_p]
        a2, n2 = sizes[second_p]
        b2 = a1 + n1 + 2
        return {first_p: 0, second_p: b2}, b2 + a2 + n2 + 2

    def consistent(bases):
        for p in names:
            a, n = sizes[p]
            lo, hi = bases[p] + a, bases[p] + a + n
            for t, at, _ in events[p]:
                if t == 'PCIP' and at.get('zone') == 'HAND' and at.get('participantId') == p:
                    cid = int(at['cardId'])
                    if cid < total and not (lo <= cid < hi):
                        return False
        return True

    chosen = None
    for fp in (order or names):
        bases, total = layout(fp)
        if consistent(bases):
            chosen = (fp, bases, total)
            break
    if chosen is None:
        return None, 'card ids fit neither creation order'
    fp, bases, total = chosen
    id_order = [fp, [n for n in names if n != fp][0]]

    decks = {}
    departures = {}
    for p in names:
        a, n = sizes[p]
        base = bases[p]
        sites = [None] * a
        deck = [None] * n
        rb = ring = None
        # blueprints of original ids, from either file
        for q in names:
            evs = events[q]
            for i, (t, at, _) in enumerate(evs):
                if t == 'PCIP' and at.get('participantId') == p and 'cardId' in at:
                    cid = int(at['cardId'])
                    if base + a <= cid < base + a + n:
                        deck[cid - base - a] = at['blueprintId']
                if t == 'RCFP' and q == p:
                    # a site leaving the adventure deck, then played: its original id, then its blueprint
                    ids = [int(x) for x in at.get('otherCardIds', '').split(',') if x]
                    if len(ids) == 1 and base <= ids[0] < base + a and i + 1 < len(evs):
                        t2, at2, _ = evs[i + 1]
                        if t2 == 'PCIP' and at2.get('zone') == 'ADVENTURE_PATH':
                            sites[ids[0] - base] = at2['blueprintId']
        # Ring-bearer and Ring: the first character and attachment the player puts out, before the starting
        # fellowship phase
        for t, at, _ in events[p]:
            if t == 'GPC':
                break
            if t == 'PCIP' and at.get('participantId') == p:
                if at.get('zone') == 'FREE_CHARACTERS' and rb is None:
                    rb = at['blueprintId']
                elif at.get('zone') == 'ATTACHED' and ring is None:
                    ring = at['blueprintId']
        # Which cards were never seen under their original id but must be in the deck: cards put into play straight
        # from it (they get a new id at once) and cards a deck search listed (by blueprint, under temp ids). Walk the
        # player's own events in order, keeping the known ids that are in the deck, and add an unseen copy whenever
        # the known ones cannot explain a play from the deck or a search's list.
        evs = events[p]
        bp_of = {}
        for t, at, _ in evs:
            if t == 'PCIP' and at.get('participantId') == p and 'cardId' in at:
                bp_of.setdefault(int(at['cardId']), at['blueprintId'])
        for j, b in enumerate(deck):
            if b is not None:
                bp_of[base + a + j] = b
        dep = []
        zone = {}
        for i, (t, at, _) in enumerate(evs):
            if t == 'PCIP' and at.get('participantId') == p and 'cardId' in at:
                cid = int(at['cardId'])
                z = zone.get(cid)
                in_range = base + a <= cid < base + a + n
                if (z is None and in_range) or z == 'deck':
                    dep.append([i, cid])
                zone[cid] = at.get('zone')
            elif t == 'RCFP':
                # a mulligan's RCFP comes after its message: the hand is already back in the deck
                for x in at.get('otherCardIds', '').split(','):
                    if x and zone.get(int(x)) != 'deck':
                        zone[int(x)] = 'gone'
            elif t == 'M' and at.get('message') == p + ' mulligans':
                for cid, z in list(zone.items()):
                    if z == 'HAND':
                        zone[cid] = 'deck'
            elif t == 'D':
                # anything removed and not put anywhere visible by the next decision may have gone to the deck
                for cid, z in list(zone.items()):
                    if z == 'gone':
                        zone[cid] = 'deck'
        departures[p] = dep
        dep_at = {}
        for i, cid in dep:
            dep_at.setdefault(cid, []).append(i)

        in_deck = {base + a + j for j, b in enumerate(deck) if b is not None}
        zone = {}
        extra, extra_left, from_deck = {}, {}, []
        for i, (t, at, params) in enumerate(evs):
            if t == 'PCIP' and at.get('participantId') == p and 'cardId' in at:
                in_deck.discard(int(at['cardId']))
                zone[int(at['cardId'])] = at.get('zone')
            elif t == 'RCFP':
                # a mulligan's RCFP comes after its message: the hand is already back in the deck
                for x in at.get('otherCardIds', '').split(','):
                    if x and zone.get(int(x)) != 'deck':
                        zone[int(x)] = 'gone'
            elif t == 'M' and at.get('message') == p + ' mulligans':
                for cid, z in list(zone.items()):
                    if z == 'HAND' and cid in bp_of:
                        in_deck.add(cid)
            elif t == 'M' and at.get('message', '').startswith(p + ' plays ') and ' from deck' in at['message']:
                h = HINT.search(at['message'])
                if not h:
                    continue
                b = h.group(1)
                from_deck.append(b)
                known = [cid for cid in in_deck if bp_of.get(cid) == b and not any(j > i for j in dep_at.get(cid, []))]
                if known:
                    in_deck.discard(min(known))
                elif extra_left.get(b, 0) > 0:
                    extra_left[b] -= 1
                else:
                    extra[b] = extra.get(b, 0) + 1
            elif t == 'D':
                for cid, z in list(zone.items()):
                    if z == 'gone':
                        zone[cid] = 'deck'
                        if cid in bp_of:
                            in_deck.add(cid)
                text = at.get('text', '')
                if at.get('decisionType') == 'ARBITRARY_CARDS' and (
                        'deck' in text.lower() and 'discard' not in text.lower() or text.startswith('Starting fellowship')):
                    listed = {}
                    for k, v in params:
                        if k == 'blueprintId':
                            listed[v] = listed.get(v, 0) + 1
                    for b, c in listed.items():
                        have = sum(1 for cid in in_deck if bp_of.get(cid) == b) + extra_left.get(b, 0)
                        if c > have:
                            extra[b] = extra.get(b, 0) + c - have
                            extra_left[b] = extra_left.get(b, 0) + c - have
        fill = [b for b, c in extra.items() for _ in range(c)]
        decks[p] = {'idBase': base, 'sites': sites, 'deck': deck, 'rb': rb, 'ring': ring,
                    'fromDeck': from_deck, 'fill': fill}

    return {
        'summary': os.path.basename(summary_path),
        'format': s['Format'],
        'players': id_order,
        'seating': order,
        'bids': bids,
        'first': first,
        'winner': winner,
        'conceded': s.get('Conceded'),
        'decks': decks,
        'events': events,
        'departures': departures,
    }, None


def main():
    args = sys.argv[1:]
    fmt, limit = 'Fellowship block', None
    if '--format' in args:
        i = args.index('--format')
        fmt = args[i + 1]
        del args[i:i + 2]
    if '--limit' in args:
        i = args.index('--limit')
        limit = int(args[i + 1])
        del args[i:i + 2]
    src, out = args
    replay_dir = os.path.join(src, 'replays')
    summaries = sorted(glob.glob(os.path.join(src, '*', '*', 'summaries', fmt, '*.json')))
    if limit:
        summaries = summaries[:limit]
    os.makedirs(out, exist_ok=True)
    reasons = {}
    written = 0
    for sp in summaries:
        try:
            m, why = build(sp, replay_dir)
        except (ET.ParseError, ValueError, KeyError) as e:
            m, why = None, 'parse error: ' + type(e).__name__
        if m is None:
            reasons[why] = reasons.get(why, 0) + 1
            continue
        name = os.path.splitext(os.path.basename(sp))[0]
        with open(os.path.join(out, name + '.json'), 'w') as f:
            json.dump(m, f, separators=(',', ':'))
        written += 1
    print(f'{len(summaries)} summaries, {written} manifests written', file=sys.stderr)
    for k, v in sorted(reasons.items(), key=lambda kv: -kv[1]):
        print(f'  skipped {v}: {k}', file=sys.stderr)


if __name__ == '__main__':
    main()
