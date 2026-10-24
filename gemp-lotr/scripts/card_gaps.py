#!/usr/bin/env python3
"""Per card, how often people and a bot play it when it is offered, from a HumanReplay --cf-rollouts 0 file.

usage: card_gaps.py cf.tsv [min_offers]

At each card-action prompt (column 17 lists the options), every option counts one offer; the human's and the bot's
picks count one play each. Cards are ranked by the plays the bot gets wrong in total, |bot - people|, so a card offered
often and misjudged a little outranks a rare one misjudged badly.
"""
import sys
from collections import defaultdict

path = sys.argv[1]
floor = int(sys.argv[2]) if len(sys.argv) > 2 else 30
offers, human, bot, names = defaultdict(int), defaultdict(int), defaultdict(int), {}
prompts = defaultdict(lambda: defaultdict(int))
rows = 0
for line in open(path, encoding='utf-8'):
    f = line.rstrip('\n').split('\t')
    if len(f) < 17 or f[3] != 'CARD_ACTION_CHOICE' or not f[16]:
        continue
    rows += 1
    prompt = f[2].replace('card action: ', '')
    seen = set()
    for opt in f[16].split(' ;; '):
        key = opt.split('|', 1)
        k = (key[0], key[1].split(' ')[0] if len(key) > 1 else '')  # blueprint, verb (Play / Use / Transfer...)
        names[k] = opt
        if k in seen:
            continue
        seen.add(k)
        offers[k] += 1
        prompts[k][prompt] += 1
    for col, tally in ((10, human), (11, bot)):
        if f[col] != 'pass':
            p = f[col].split('|', 1)
            tally[(p[0], p[1].split(' ')[0] if len(p) > 1 else '')] += 1

print(f'{rows} card-action prompts')
ranked = sorted((k for k in offers if offers[k] >= floor), key=lambda k: -abs(bot[k] - human[k]))
total = sum(abs(bot[k] - human[k]) for k in offers)
print(f'{"card":44} {"offered":>7} {"people":>7} {"bot":>7}  {"gap":>5}  main prompt')
for k in ranked[:int(sys.argv[3]) if len(sys.argv) > 3 else 40]:
    o = offers[k]
    main = max(prompts[k], key=prompts[k].get)
    print(f'{names[k][:44]:44} {o:7d} {human[k] / o:7.1%} {bot[k] / o:7.1%}  {bot[k] - human[k]:+5d}  {main[:30]}')
print(f'summed |gap| over all cards: {total}; top 40 carry {sum(abs(bot[k] - human[k]) for k in ranked[:40])}')
