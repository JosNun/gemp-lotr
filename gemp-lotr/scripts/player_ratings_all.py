"""Players' strength from their results over 2020-22 (2026-11-07): a Bradley-Terry fit over every replayed Movie Block
game in the routes files (game ids name both players; "winner" says who won), with a prior of one win and one loss
against an average player so a few lucky games make no star. scripts/player_ratings.py (October) rated 2022 alone.

usage: python3 -I scripts/player_ratings_all.py <out.json> <routes.jsonl>... [--top 0.33] [--min-games 20] [--top-out f]

Writes {player: {"rating", "games", "wins"}}; with --top-out, the top share of players with --min-games or more games,
one a line -- what board_net2 --players keeps. A rating from hundreds of games is mostly skill; one game's result is
mostly luck and deck.
"""
import collections, json, math, sys


def parse(game):
    a, rest = game.split('_vs_', 1)
    return a, rest.rsplit('_', 2)[0]


def main():
    args = sys.argv[1:]
    top_share, min_games, top_out = 0.33, 20, None
    for flag in ('--top', '--min-games', '--top-out'):
        if flag in args:
            i = args.index(flag)
            v = args[i + 1]
            del args[i:i + 2]
            if flag == '--top':
                top_share = float(v)
            elif flag == '--min-games':
                min_games = int(v)
            else:
                top_out = v
    out, files = args[0], args[1:]
    games = []
    for f in files:
        for line in open(f):
            r = json.loads(line)
            if '_vs_' not in r['game'] or not r.get('winner'):
                continue
            a, b = parse(r['game'])
            w = r['winner']
            if w in (a, b) and a != b:
                games.append((w, b if w == a else a))
    players = sorted({p for g in games for p in g})
    idx = {p: i for i, p in enumerate(players)}
    wins = collections.Counter(w for w, _ in games)
    n = collections.Counter(p for g in games for p in g)
    s = [1.0] * len(players)
    for _ in range(200):  # minorise-maximise for Bradley-Terry
        denom = [2.0 / (x + 1.0) for x in s]  # the prior's two games against strength 1
        for w, l in games:
            d = 1.0 / (s[idx[w]] + s[idx[l]])
            denom[idx[w]] += d
            denom[idx[l]] += d
        s = [(wins[p] + 1.0) / denom[idx[p]] for p in players]
        g = math.exp(sum(math.log(x) for x in s) / len(s))
        s = [x / g for x in s]
    rating = {p: 400 * math.log10(s[idx[p]]) + 1500 for p in players}
    json.dump({p: {'rating': round(rating[p], 1), 'games': n[p], 'wins': wins[p]} for p in players}, open(out, 'w'))
    ranked = sorted((p for p in players if n[p] >= min_games), key=lambda p: -rating[p])
    top = ranked[:int(len(ranked) * top_share)]
    print(f'{len(games)} games, {len(players)} players, {len(ranked)} with {min_games}+ games; the top {len(top)} '
          f'(rating {rating[top[-1]]:.0f}+) played {sum(n[p] for p in top)} of {2 * len(games)} game-sides', file=sys.stderr)
    if top_out:
        open(top_out, 'w').write('\n'.join(top) + '\n')


if __name__ == '__main__':
    main()
