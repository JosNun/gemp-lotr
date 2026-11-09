"""Bot-vs-bot win rates scored by pair (2026-11-09): BotVsBot gives bot A deck 1 on even games and deck 2 on odd ones,
so games (2j, 2j+1) of a matchup are a pair in which the decks swap and, with --seed, the shuffles do not. The decks
decide most games; scored by pair, their effect cancels, and the interval is the spread of pair scores rather than of
coin flips.

usage: python3 -I scripts/paired.py <game-log.jsonl>             A's win rate, by pair and binomial, and the split
       python3 -I scripts/paired.py <x.jsonl> <y.jsonl>          X - Y, where X and Y each played the same B on the
                                                                 same --seed: matched game by game

A game log is what -Dbvb.gameLog writes, one JSON object a game. Two logs are matched on (deckA, deckB, seed, which
game of its pair), so both must come from runs on the same base seed; games without a partner are left out and
counted. Logs written before seeds were logged still score alone, by (matchup, game / 2).

Found: on two 6,330-game unseeded logs, by pair gives +-1.06-1.08 where binomial gives +-1.23 -- the swap alone is
worth a quarter fewer games. Seeded, a pair's two games are the same deal, and the decks' part goes almost entirely:
sp-m3 vs sp-m2, first 150 matchups of suite-train x 4 games, gave +-1.94 by pair (265 of 300 pairs split) where the
same run unseeded gave +-3.35 (195 split) and binomial +-4.0 -- about 4x fewer games for one interval. Matching two
runs on one seed (sp-m3 vs sp-m2 and sp-m2-t25 vs sp-m2) put X - Y at +-2.09, against +-3.12 for the same two
by-pair intervals unmatched and +-5.66 binomial.
"""
import json, math, sys


def load(path):
    games = []
    for line in open(path):
        line = line.strip()
        if line:
            g = json.loads(line)
            if g.get('winner') in ('A', 'B') and not g.get('error'):
                games.append(g)
    return games


def pair_of(g):
    return g['matchup'], g.get('pair', g['game'] // 2)


def mean_sd(xs):
    m = sum(xs) / len(xs)
    return m, math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1)) if len(xs) > 1 else float('nan')


def one(path):
    games = load(path)
    if not games:
        sys.exit(path + ': no finished games')
    n = len(games)
    p = sum(g['winner'] == 'A' for g in games) / n
    pairs = {}
    for g in games:
        pairs.setdefault(pair_of(g), []).append(g['winner'] == 'A')
    whole = [v for v in pairs.values() if len(v) == 2]
    print('%s: %d games, A (%s) %.1f%% +- %.2f binomial' % (path, n, games[0].get('botA', '?'), 100 * p,
                                                            196 * math.sqrt(p * (1 - p) / n)))
    if len(whole) < 2:
        print('  fewer than 2 complete pairs: no pair-based interval')
        return
    _, sd = mean_sd([sum(v) / 2 for v in whole])
    both = sum(all(v) for v in whole)
    neither = sum(not any(v) for v in whole)
    print('  by pair: +- %.2f over %d complete pairs; A won both %d, split %d, B won both %d' % (
        196 * sd / math.sqrt(len(whole)), len(whole), both, len(whole) - both - neither, neither))


def two(px, py):
    def keyed(path):
        out = {}
        for g in load(path):
            if 'seed' not in g:
                sys.exit(path + ': no seeds in this log -- matching needs a run with --seed')
            out[(g['deckA'], g['deckB'], g['seed'], g['game'] % 2)] = g
        return out

    x, y = keyed(px), keyed(py)
    common = sorted(set(x) & set(y), key=str)
    if not common:
        sys.exit('no game in both logs: were they played on the same --seed?')
    if x[common[0]].get('botB') != y[common[0]].get('botB'):
        print('warning: the opponents differ (%s, %s)' % (x[common[0]].get('botB'), y[common[0]].get('botB')))
    wx = [x[k]['winner'] == 'A' for k in common]
    wy = [y[k]['winner'] == 'A' for k in common]
    n = len(common)
    pxr, pyr = sum(wx) / n, sum(wy) / n
    # the pair: both games of a deck pairing on one seed, whichever deck A held
    pairs = {}
    for k, a, b in zip(common, wx, wy):
        pairs.setdefault((tuple(sorted(k[:2])), k[2]), []).append((a, b))
    whole = [v for v in pairs.values() if len(v) == 2]
    print('X %s: %.1f%%   Y %s: %.1f%%   over %d games in both logs (%d only in X, %d only in Y)' % (
        x[common[0]].get('botA'), 100 * pxr, y[common[0]].get('botA'), 100 * pyr, n, len(x) - n, len(y) - n))
    unpaired = 196 * math.sqrt(pxr * (1 - pxr) / n + pyr * (1 - pyr) / n)
    print('  X - Y = %+.1f points; unpaired (binomial) +- %.2f' % (100 * (pxr - pyr), unpaired))
    if len(whole) < 2:
        print('  fewer than 2 complete pairs in both logs: nothing matched to score')
        return
    _, sx = mean_sd([(v[0][0] + v[1][0]) / 2 for v in whole])
    _, sy = mean_sd([(v[0][1] + v[1][1]) / 2 for v in whole])
    m, sd = mean_sd([(v[0][0] + v[1][0] - v[0][1] - v[1][1]) / 2 for v in whole])
    r = math.sqrt(len(whole))
    print('  over %d pairs in both: X - Y = %+.1f; matched by pair +- %.2f; each by pair, unmatched +- %.2f' % (
        len(whole), 100 * m, 196 * sd / r, 196 * math.sqrt(sx ** 2 + sy ** 2) / r))
    same = sum(v[0][0] == v[0][1] and v[1][0] == v[1][1] for v in whole)
    print('  pairs where X and Y won exactly the same games: %d of %d' % (same, len(whole)))


if __name__ == '__main__':
    if len(sys.argv) == 2:
        one(sys.argv[1])
    elif len(sys.argv) == 3:
        two(sys.argv[1], sys.argv[2])
    else:
        sys.exit(__doc__)
