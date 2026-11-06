"""Where a bot that moves again more often loses: companions lost in its own turns with and without a second move, and
win rate by how many second moves it made (BotVsBot -Dbvb.gameLog with movedAgainOn and deadByTurn, 2026-11-05).

usage: python3 -I scripts/double_moves.py <gameLog.jsonl>

Turn numbers are the engine's, one per player turn: the first player's are the odd ones. A player's companions die on
its own turns (its skirmishes), so its deaths are read on those.
"""
import collections, json, re, statistics, sys


def main():
    # deadByTurn was first written with bare number keys ({3:1}), which is not JSON: quoted before reading
    games = [json.loads(re.sub(r'([{,])(\d+):', r'\1"\2":', l)) for l in open(sys.argv[1])]
    games = [g for g in games if g.get('winner') and g.get('players')]
    for seat in ('A', 'B'):
        dead = {True: [], False: []}  # companions lost in an own turn, by whether it moved again that turn
        by_moves = collections.defaultdict(list)  # second moves in the game -> won?
        late = collections.defaultdict(lambda: [0, 0])  # moved again on turn t -> companions lost on own turn t+2
        for g in games:
            p = g['players'].get(seat)
            if not p or 'deadByTurn' not in p:
                continue
            first = g['first'] == seat
            own = [t for t in range(1, g['turns'] + 1) if (t % 2 == 1) == first]
            moved = set(p.get('movedAgainOn', []))
            d = {int(k): v for k, v in p['deadByTurn'].items()}
            for t in own:
                dead[t in moved].append(d.get(t, 0))
                if t in moved:
                    late[True][0] += d.get(t + 2, 0); late[True][1] += 1
                else:
                    late[False][0] += d.get(t + 2, 0); late[False][1] += 1
            by_moves[min(len(moved), 4)].append(g['winner'] == seat)
        print(f'seat {seat}:')
        for k in (False, True):
            v = dead[k]
            if v:
                print(f'  own turns {"with" if k else "without"} a second move: {len(v)}, companions lost that turn '
                      f'{statistics.mean(v):.3f}, the own turn after {late[k][0] / max(1, late[k][1]):.3f}')
        for n in sorted(by_moves):
            w = by_moves[n]
            print(f'  games with {n}{"+" if n == 4 else ""} second moves: {len(w)}, won {100 * statistics.mean(w):.1f}%')


if __name__ == '__main__':
    main()
