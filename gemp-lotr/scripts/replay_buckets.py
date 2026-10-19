"""Bucket HumanReplay's divergences: python3 -I scripts/replay_buckets.py <report.tsv> <manifest dir> [cards dir]

Reads the --report TSV, sorts each game's first divergence into a cause, and names the cards involved (card ids in
the differing events, looked up in the game's manifest; names from the card definitions).
"""
import collections
import glob
import json
import os
import re
import sys


def card_names(cards_dir):
    names = {}
    for f in glob.glob(os.path.join(cards_dir, '**', '*.hjson'), recursive=True):
        bp = None
        for line in open(f, encoding='utf-8'):
            m = re.match(r'^\t(\d+_\d+\w*): \{', line)
            if m:
                bp = m.group(1)
                continue
            m = re.match(r'^\t\ttitle: (.*)$', line)
            if m and bp and bp not in names:
                names[bp] = m.group(1).strip()
    return names


def bucket(detail, row_bucket):
    if row_bucket != 'events differ':
        return row_bucket
    m = re.match(r'(\S+): expected (.*?) got (.*?) after (.*)$', detail)
    if not m:
        return 'events differ (unparsed)'
    exp, got = m.group(2), m.group(3)
    exp_d = 'D|' in exp
    got_d = 'D|' in got
    if exp.startswith('D|') and got.startswith('D|') and ' only [' in exp:
        he, hg = exp.split(' only [')[0], got.split(' only [')[0]
        if he == hg:
            return 'same question, other options'
        return 'different question'
    if got_d and not exp_d:
        return 'engine asks a question the human was not asked'
    if exp_d and not got_d:
        return 'human was asked a question the engine did not ask'
    first_exp = exp.split(' ; ')[0].split('|')[0]
    first_got = got.split(' ; ')[0].split('|')[0]
    if first_exp == first_got:
        return f'event differs: {first_exp}'
    return f'event differs: {first_exp} expected, {first_got} instead'


def main():
    report, manifests = sys.argv[1], sys.argv[2]
    cards_dir = sys.argv[3] if len(sys.argv) > 3 else 'gemp-lotr-cards/src/main/resources/cards'
    names = card_names(cards_dir)
    rows = []
    with open(report, encoding='utf-8') as f:
        header = f.readline().rstrip('\n').split('\t')
        for line in f:
            rows.append(dict(zip(header, line.rstrip('\n').split('\t'))))
    rec = sum(int(r['recorded']) for r in rows)
    rep = sum(int(r['reproduced']) for r in rows)
    print(f'{len(rows)} games, {rec} recorded decisions, {rep} reproduced ({100 * rep / max(1, rec):.1f}%)')
    by = collections.defaultdict(list)
    for r in rows:
        by[bucket(r['detail'], r['bucket'])].append(r)
    print('\nfirst divergence, by cause (games; recorded decisions those games left unreproduced):')
    for b, rs in sorted(by.items(), key=lambda kv: -len(kv[1])):
        lost = sum(int(r['recorded']) - int(r['reproduced']) for r in rs)
        print(f'  {len(rs):5d}  {100 * len(rs) / len(rows):5.1f}%  {lost:7d}  {b}')

    # the cards named in each divergence
    cards = collections.Counter()
    games_with = collections.defaultdict(set)
    for r in rows:
        if r['bucket'] != 'events differ':
            continue
        path = os.path.join(manifests, r['game'] + '.json')
        if not os.path.exists(path):
            continue
        m = json.load(open(path))
        bp = {}
        for p in m['players']:
            for t, a, _ in m['events'][p]:
                if t == 'PCIP' and 'cardId' in a:
                    bp[a['cardId']] = a['blueprintId']
        d = r['detail']
        d = d.split(' after ')[0]
        ids = set(re.findall(r'cardId=(\d+)', d)) | set(re.findall(r'\|(\d+)(?=\||$| |;)', d))
        named = {bp[i] for i in ids if i in bp}
        for b in named:
            cards[b] += 1
            games_with[b].add(r['game'])
    print('\ncards named in the first divergence (games):')
    for b, n in cards.most_common(30):
        print(f'  {n:4d}  {b:8s} {names.get(b, "?")}')


if __name__ == '__main__':
    main()
