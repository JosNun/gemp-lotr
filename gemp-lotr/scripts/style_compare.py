"""Bots' style against people's, side by side: BotVsBot -Dbvb.gameLog lines and scripts/people_style.py lines.

usage: python3 -I scripts/style_compare.py <name>=<file.jsonl>[:A|:B] ...    (:A -- a head-to-head's seat A only)

Per source: games; the first player's win rate (all games with a winner, and those that ended on the board); how
board-ended games ended; their median length; and per player, split by whether they went first -- "move again?"
taken when asked, sites gained per own turn, minions per Shadow turn, twilight left when the Shadow phase ended,
companions dead at the end.
"""
import json, statistics, sys

BOARD = ('site9', 'kill', 'corrupt')


def own_turns(turns, first):
    return turns // 2 + (1 if first and turns % 2 else 0)


def summarise(spec):
    path, seat = (spec[:-2], spec[-1]) if spec[-2:] in (':A', ':B') else (spec, None)
    games = [json.loads(l) for l in open(path)]
    games = [g for g in games if g.get('winner') and g.get('players')]
    board = [g for g in games if g.get('route') in BOARD]
    s = {'games': len(games), 'board': len(board)}
    s['first wins (all)'] = 100 * sum(g['winner'] == g['first'] for g in games) / max(1, len(games))
    s['first wins (board)'] = 100 * sum(g['winner'] == g['first'] for g in board) / max(1, len(board))
    for r in BOARD:
        s[f'ends: {r}'] = 100 * sum(g['route'] == r for g in board) / max(1, len(board))
    s['median turns (board)'] = statistics.median(g['turns'] for g in board) if board else None
    for side in ('first', 'second'):
        asked = yes = st = mi = 0
        un, pace, dead, won, ate = [], [], [], [], []
        for g in board:
            for name, p in g['players'].items():
                is_first = name == g['first']
                if (side == 'first') != is_first or (seat and name != seat):
                    continue
                asked += p.get('moveAsked', 0); yes += p.get('moveYes', 0)
                st += p.get('shadowTurns', 0); mi += p.get('minions', 0)
                if p.get('unspentTwilight') is not None:
                    un.append(p['unspentTwilight'])
                if p.get('minionsAtShadowEnd') is not None:
                    ate.append(p['minionsAtShadowEnd'])
                ot = own_turns(g['turns'], is_first)
                if ot:
                    pace.append((p.get('site', 1) - 1) / ot)
                dead.append(p.get('deadCompanions', 0))
                won.append(g['winner'] == name)
        s[f'{side}: wins (board)'] = 100 * sum(won) / max(1, len(won))
        s[f'{side}: move again'] = 100 * yes / max(1, asked)
        s[f'{side}: sites a turn'] = statistics.mean(pace) if pace else None
        s[f'{side}: minions a Shadow turn'] = mi / max(1, st)
        s[f'{side}: twilight left'] = statistics.mean(un) if un else None
        s[f'{side}: minions in play, Shadow end'] = statistics.mean(ate) if ate else None
        s[f'{side}: companions dead'] = statistics.mean(dead) if dead else None
    return s


def main():
    cols = [(a.split('=', 1)[0], summarise(a.split('=', 1)[1])) for a in sys.argv[1:]]
    keys = list(cols[0][1])
    print('| | ' + ' | '.join(n for n, _ in cols) + ' |')
    print('|---|' + '---|' * len(cols))
    for k in keys:
        cells = []
        for _, s in cols:
            v = s.get(k)
            cells.append('' if v is None else f'{v:.0f}' if k in ('games', 'board') else f'{v:.2f}' if abs(v) < 10 else f'{v:.1f}')
        print(f'| {k} | ' + ' | '.join(cells) + ' |')


if __name__ == '__main__':
    main()
