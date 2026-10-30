"""Each replayed game's decklists, as a player knows their own deck, from the same deck's other games.

usage: python3 -I scripts/decklists.py <out.jsonl> <manifest dir>...

The summaries list only the cards a game showed (about 60% of a deck), but a card's original id is its slot in the
decklist, and a deck keeps its slot order from game to game: of 2022's same-size game pairs by one player, 156,397
agree on every slot both saw. So one player's games are grouped where they agree on every shared slot (10 or more),
and each game is given the union of the slots its group's *other* games saw. Never its own: a deck seen once would
otherwise list exactly the cards that game was about to draw. The slots no game ever shows are the cards played
straight from the deck (the starting fellowship, mostly), which are out of the deck by the first decision anyway.

Writes one line per game and player: {game, player, deck: [blueprints of known slots], size, known}.
"""
import collections, glob, json, os, sys


def main():
    out, dirs = sys.argv[1], sys.argv[2:]
    games = collections.defaultdict(list)  # player -> [(game, deck slots)]
    for d in dirs:
        for path in sorted(glob.glob(os.path.join(d, '*.json'))):
            m = json.load(open(path))
            name = os.path.basename(path)[:-5]
            for p, x in m['decks'].items():
                games[p].append((name, x['deck']))
    n = full = 0
    known = []
    with open(out, 'w') as f:
        for p, gs in games.items():
            groups = []  # [union, [(game, slots)]]
            for name, deck in gs:
                for g in groups:
                    if len(g[0]) != len(deck):
                        continue
                    both = [(a, b) for a, b in zip(g[0], deck) if a and b]
                    if len(both) >= 10 and all(a == b for a, b in both):
                        g[0] = [a or b for a, b in zip(g[0], deck)]
                        g[1].append((name, deck))
                        break
                else:
                    groups.append([list(deck), [(name, deck)]])
            for _, members in groups:
                for i, (name, deck) in enumerate(members):
                    slots = [None] * len(deck)
                    for j, (_, other) in enumerate(members):
                        if j != i:
                            slots = [a or b for a, b in zip(slots, other)]
                    cards = [b for b in slots if b]
                    k = len(cards) / len(deck) if deck else 0
                    known.append(k)
                    full += k >= 0.9
                    n += 1
                    f.write(json.dumps({'game': name, 'player': p, 'deck': cards, 'size': len(deck),
                                        'known': round(k, 3)}) + '\n')
    known.sort()
    print(f'{n} decklists; known slots from the deck\'s other games: median {known[n // 2]:.0%}, '
          f'quartiles {known[n // 4]:.0%} / {known[3 * n // 4]:.0%}; 90% or more known: {full / n:.0%}')


if __name__ == '__main__':
    main()
