"""Does the bot rank real decks the way their human results do? Run from the repo root.

    python3 scripts/deck_fidelity.py make decks/real/movie-human-games.txt
    ./bot-vs-bot.sh --suite decks/real/movie-human-games.txt --a safe@<models> --b safe@<models> --games 4 > bot.log
    python3 scripts/deck_fidelity.py score bot.log

`make` writes every human Movie Block game in decks/real/games.csv that has a deck of 20+ human games as a
matchup (4,476 games, 135 decks). `score` takes each such deck's bot win rate over the matchups it actually
played and correlates it with its human win rate. Human win rates split in half agree at r 0.62 (0.77 for
the full record), so that is about the ceiling -- and part of it is the pilot, not the deck. A
16-deck round robin swung from 0.44 to -0.04 between bot settings that play equally well; this is the
check to trust. Plain bots run it in a few minutes.
"""
import csv, os, re, sys, collections, math
games = [r for r in csv.reader(open('decks/real/games.csv')) if r[0] == 'movie' and r[3] in (r[1], r[2])]
games = [r for r in games if os.path.exists(f'decks/real/movie/{r[1]}.txt') and os.path.exists(f'decks/real/movie/{r[2]}.txt')]
count = collections.Counter(d for r in games for d in r[1:3])
MIN = 20
if sys.argv[1] == 'make':
    keep = [r for r in games if count[r[1]] >= MIN or count[r[2]] >= MIN]
    with open(sys.argv[2], 'w') as f:
        f.write(f'# every human Movie Block game (games.csv) with a deck of {MIN}+ games: the bot replays the matchup\n')
        for r in keep:
            f.write(f'movie | decks/real/movie/{r[1]}.txt | decks/real/movie/{r[2]}.txt\n')
    print(len(keep), 'matchups,', sum(1 for d in count if count[d] >= MIN), 'decks with', MIN, '+ games')
else:
    bot = collections.defaultdict(lambda: [0, 0]); hum = collections.defaultdict(lambda: [0, 0])
    pairs = []
    for line in open(sys.argv[2]):
        m = re.match(r'decks/real/movie/(\w+)\.txt vs decks/real/movie/(\w+)\.txt \[movie\]\s+A\s+\d+/\d+\s+deck1\s+(\d+)/(\d+)', line)
        if m: pairs.append((m.group(1), m.group(2), int(m.group(3)), int(m.group(4))))
    res = collections.defaultdict(lambda: [0, 0])
    for a, b, w1, n in pairs:
        res[(a, b)][0] += w1; res[(a, b)][1] += n
    keep = [r for r in games if count[r[1]] >= MIN or count[r[2]] >= MIN]
    # matchups appear once per suite line, in order
    for r in keep:
        a, b = r[1], r[2]
        w1, n = res[(a, b)]
        if n == 0: continue
        for d, w, hw in ((a, w1, r[3] == a), (b, n - w1, r[3] == b)):
            bot[d][0] += w / n; bot[d][1] += 1; hum[d][0] += hw; hum[d][1] += 1
    ds = [d for d in hum if count[d] >= MIN and hum[d][1] >= MIN]
    x = [bot[d][0] / bot[d][1] for d in ds]; y = [hum[d][0] / hum[d][1] for d in ds]
    def rank(v):
        o = sorted(range(len(v)), key=lambda i: v[i]); r = [0] * len(v)
        for k, i in enumerate(o): r[i] = k
        return r
    def pearson(a, b):
        ma, mb = sum(a) / len(a), sum(b) / len(b)
        return sum((p - ma) * (q - mb) for p, q in zip(a, b)) / math.sqrt(sum((p - ma) ** 2 for p in a) * sum((q - mb) ** 2 for q in b))
    print(f'{len(pairs)} matchups scored; {len(ds)} decks with {MIN}+ human games: Spearman {pearson(rank(x), rank(y)):.3f}, Pearson {pearson(x, y):.3f}')
