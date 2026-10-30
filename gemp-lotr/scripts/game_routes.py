"""How each replayed game was won -- site 9, killing the Ring-bearer, or corrupting it -- with concessions inferred.

usage: python3 -I scripts/game_routes.py <out.jsonl> <manifest dir>...

44% of 2022's Fellowship-block games end in a concession, which says who won but not how they were winning. The
route is read off the position instead: each game's event stream is walked to the end, keeping both players' site,
burdens and wounds on their Ring-bearer, characters in play and dead pile, snapshotted at the start of every turn. A
softmax regression learns route from the snapshot at the start of the last turn of games that ended on the board
(fit on 80% of them, accuracy reported on the rest), and gives a concession (or a time-out) its route only where it is
confident (--min-conf, default 0.7); the rest are left without one.

Writes one line per game: {game, winner, route: site9|kill|corrupt|null, how: board|inferred|unsure, conf}.
"""
import argparse, glob, json, os, re

import numpy as np

ROUTES = ['site9', 'kill', 'corrupt']


def walk(m):
    """Snapshots (one per turn start, then the end) of both players' positions, and how the game ended."""
    players = m['players']
    stream = m['events'][players[0]]
    rb_bp = {p: m['decks'][p].get('rb') for p in players}
    rb = {}
    pos = {p: 1 for p in players}
    burdens = {p: 0 for p in players}
    wounds = {}  # card id -> wounds
    chars = {p: set() for p in players}
    dead = {p: 0 for p in players}
    turn = 0
    current = None
    snaps = []
    lost, won = {}, {}

    def snap():
        return {p: dict(pos=pos[p], burdens=burdens[p], rbw=wounds.get(rb.get(p), 0), chars=len(chars[p]),
                        dead=dead[p]) for p in players} | {'turn': turn, 'current': current}

    for kind, at, _ in stream:
        if kind == 'PP':
            pos[at['participantId']] = int(at['index'])
        elif kind == 'PCIP':
            p, cid = at.get('participantId'), at.get('cardId')
            if at.get('zone') == 'FREE_CHARACTERS' and p in chars:
                chars[p].add(cid)
                if at.get('blueprintId') == rb_bp.get(p) and p not in rb:
                    rb[p] = cid
        elif kind == 'RCFP':
            for cid in at.get('otherCardIds', '').split(','):
                for s in chars.values():
                    s.discard(cid)
        elif kind in ('AT', 'RT'):
            n = int(at.get('count', 0)) * (1 if kind == 'AT' else -1)
            cid, tok = at.get('cardId'), at.get('token')
            if tok == 'BURDEN':
                for p in players:
                    if rb.get(p) == cid:
                        burdens[p] += n
            elif tok == 'WOUND':
                wounds[cid] = wounds.get(cid, 0) + n
        elif kind == 'GS':
            try:
                z = json.loads(at['zones'])
                for p in players:
                    dead[p] = int(z.get(p, {}).get('DEAD', dead[p]))
            except (KeyError, ValueError):
                pass
        elif kind == 'TC':
            turn += 1
            current = at.get('participantId')
            snaps.append(snap())
        elif kind == 'M':
            msg = at.get('message', '')
            mm = re.match(r'(.+) lost due to: (.+)$', msg)
            if mm:
                lost[mm.group(1)] = mm.group(2).lower()
            mm = re.match(r'(.+) is the winner due to: (.+)$', msg)
            if mm:
                won[mm.group(1)] = mm.group(2).lower()
    snaps.append(snap())
    return snaps, lost, won


def route_of(lost, won):
    """The route from the end messages, or 'infer' (a concession or time-out), or None."""
    reasons = ' '.join(list(lost.values()) + list(won.values()))
    if 'surviving to regroup' in reasons:
        return 'site9'
    if 'corrupt' in reasons:
        return 'corrupt'
    if 'ring-bearer is dead' in reasons:
        return 'kill'
    if 'concession' in reasons or 'timed-out' in reasons or 'out of time' in reasons:
        return 'infer'
    return None


def features(s, w, l):
    """The snapshot from the winner's side: positions, burdens, Ring-bearer wounds, characters, dead -- both players."""
    a, b = s[w], s[l]
    mine = float(s['current'] == w)  # the winner's own turn: site 9 is won on it, a kill on the other's
    return [a['pos'], b['pos'], a['pos'] - b['pos'], float(a['pos'] >= 8), float(b['pos'] >= 8), float(a['pos'] == 9),
            a['burdens'], b['burdens'], a['rbw'], b['rbw'], a['chars'], b['chars'], a['dead'], b['dead'],
            float(b['chars'] <= 1), float(b['chars'] <= 2), float(b['burdens'] >= 6), float(b['burdens'] >= 8),
            s['turn'], mine, mine * a['pos'], mine * float(a['pos'] >= 8), (1 - mine) * b['chars'],
            (1 - mine) * b['rbw'], (1 - mine) * b['burdens']]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('out')
    ap.add_argument('dirs', nargs='+')
    ap.add_argument('--min-conf', type=float, default=0.7)
    args = ap.parse_args()
    cache = args.out + '.walk.json'
    cached = os.path.exists(cache)
    games = [tuple(g) for g in json.load(open(cache))] if cached else []
    for d in ([] if cached else args.dirs):
        for path in sorted(glob.glob(os.path.join(d, '*.json'))):
            m = json.load(open(path))
            w = m.get('winner')
            if w not in m['players']:
                continue
            l = [p for p in m['players'] if p != w][0]
            snaps, lost, won = walk(m)
            r = route_of(lost, won)
            if r is None:
                continue
            last = snaps[-2] if len(snaps) >= 2 else snaps[-1]  # the start of the last turn
            games.append((os.path.basename(path)[:-5], w, r, features(last, w, l)))
    if not os.path.exists(cache):
        json.dump(games, open(cache, 'w'))
    decided = [g for g in games if g[2] in ROUTES]
    X = np.array([g[3] for g in decided], np.float64)
    y = np.array([ROUTES.index(g[2]) for g in decided])
    mu, sd = X.mean(0), X.std(0) + 1e-9
    rng = np.random.default_rng(0)
    idx = rng.permutation(len(y))
    cut = int(len(y) * 0.8)
    tr, te = idx[:cut], idx[cut:]

    def fit(rows):
        Z = (X[rows] - mu) / sd
        W = np.zeros((Z.shape[1] + 1, len(ROUTES)))
        Zb = np.hstack([Z, np.ones((len(Z), 1))])
        Y = np.eye(len(ROUTES))[y[rows]]
        for _ in range(3000):
            P = softmax(Zb @ W)
            W -= 0.5 * (Zb.T @ (P - Y) / len(Z) + 1e-3 * W)
        return W

    def predict(W, F):
        Z = (np.asarray(F, np.float64) - mu) / sd
        return softmax(np.hstack([Z, np.ones((len(Z), 1))]) @ W)

    W = fit(tr)
    P = predict(W, X[te])
    acc = (P.argmax(1) == y[te]).mean()
    conf = P.max(1) >= args.min_conf
    print(f'{len(decided)} games ended on the board ({", ".join(f"{r} {np.mean(y == i):.0%}" for i, r in enumerate(ROUTES))}); '
          f'route from the start of the last turn, held out: {acc:.1%}; where confident (>= {args.min_conf}): '
          f'{(P.argmax(1)[conf] == y[te][conf]).mean():.1%} of {conf.mean():.0%}')
    for t in (0.6, 0.7, 0.8, 0.85, 0.9, 0.95):
        c = P.max(1) >= t
        print(f'  confidence >= {t}: {(P.argmax(1)[c] == y[te][c]).mean():.1%} right on {c.mean():.0%} of held-out games')
    W = fit(np.arange(len(y)))
    rest = [g for g in games if g[2] == 'infer']
    P = predict(W, [g[3] for g in rest]) if rest else np.zeros((0, len(ROUTES)))
    n_sure = 0
    with open(args.out, 'w') as f:
        for g in decided:
            f.write(json.dumps({'game': g[0], 'winner': g[1], 'route': g[2], 'how': 'board', 'conf': 1.0}) + '\n')
        for g, p in zip(rest, P):
            sure = p.max() >= args.min_conf
            n_sure += sure
            f.write(json.dumps({'game': g[0], 'winner': g[1], 'route': ROUTES[int(p.argmax())] if sure else None,
                                'how': 'inferred' if sure else 'unsure', 'conf': round(float(p.max()), 3)}) + '\n')
    inferred = [ROUTES[int(p.argmax())] for p in P if p.max() >= args.min_conf]
    print(f'{len(rest)} concessions and time-outs: {n_sure} given a route ('
          + ', '.join(f'{r} {inferred.count(r)}' for r in ROUTES) + f'), {len(rest) - n_sure} left without one')


def softmax(z):
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


if __name__ == '__main__':
    main()
