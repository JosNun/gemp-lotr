"""Which cards does the bot use differently from people? Run from the repo root.

    JAVA_OPTS=-DcardUse=all ./bot-vs-bot.sh --suite decks/real/suite-test-movie.txt --a safe@<m> --b safe@<m> --games 8 > bot.log
    python3 scripts/card_use.py bot.log ~/dev/gemp-replays/training/movie-2022-1*.jsonl

For every card action (blueprint), the share of the times it was offered that it was taken -- by the bot (CardUse,
both seats pooled) and by people in reproduced replays (HumanReplay --steps) -- and the cards where they differ most,
by a z-score on the difference. A card the bot plays far more often than people is a candidate for a self-inflicted
wound (a burden paid for a skirmish); far less often, a card it does not know how to use.
"""
import collections, glob, json, math, os, re, sys

bot = collections.defaultdict(lambda: [0, 0])
for line in open(sys.argv[1]):
    m = re.match(r'\s+[AB]\s+(\S+)\s+[\d.]+%\s+(\d+) / (\d+)', line)
    if m and '_' in m.group(1):
        bot[m.group(1)][0] += int(m.group(2)); bot[m.group(1)][1] += int(m.group(3))

human = collections.defaultdict(lambda: [0, 0])
for f in sys.argv[2:]:
    key, seen = None, {}
    def flush():
        for bp, used in seen.items():
            human[bp][0] += used; human[bp][1] += 1
    for line in open(f):
        if '"CardActionSelectionDecision"' not in line:
            continue
        d = json.loads(line)
        k = json.dumps(d['decision'], sort_keys=True) + str(d['state'][:8])
        if k != key:
            flush(); key, seen = k, {}
        bp = d['action'].get('sourceBlueprint')
        if bp:
            seen[bp] = seen.get(bp, False) or d['reward'] > 0
    flush()

titles = {}
for f in glob.glob('gemp-lotr-cards/src/main/resources/cards/official/**/*.hjson', recursive=True):
    cur = None
    for line in open(f, encoding='utf-8'):
        m = re.match(r'\s*(\d+_\d+): \{', line)
        if m: cur = m.group(1)
        m = re.match(r'\s*title: (.+)', line)
        if m and cur and cur not in titles: titles[cur] = m.group(1).strip().strip('"')

rows = []
for bp in bot:
    bu, bn = bot[bp]; hu, hn = human.get(bp, [0, 0])
    if bn < 100 or hn < 100: continue
    pb, ph = bu / bn, hu / hn
    p = (bu + hu) / (bn + hn)
    se = math.sqrt(max(p * (1 - p), 1e-9) * (1 / bn + 1 / hn))
    rows.append(((pb - ph) / se, bp, pb, bn, ph, hn))
rows.sort()
print(f'{len(rows)} cards offered 100+ times to both. Bot uses LESS than people:')
for z, bp, pb, bn, ph, hn in rows[:15]:
    print(f'  {bp:8s} {titles.get(bp, "?")[:34]:34s} bot {100*pb:5.1f}% of {bn:6d}   people {100*ph:5.1f}% of {hn:6d}   z {z:6.1f}')
print('Bot uses MORE than people:')
for z, bp, pb, bn, ph, hn in reversed(rows[-15:]):
    print(f'  {bp:8s} {titles.get(bp, "?")[:34]:34s} bot {100*pb:5.1f}% of {bn:6d}   people {100*ph:5.1f}% of {hn:6d}   z {z:6.1f}')
