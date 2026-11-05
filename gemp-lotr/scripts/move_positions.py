"""Where people and bots stand when asked "move again?": the positions' features side by side, and the share saying
yes in matched positions -- whether the bots move less because they decide differently or because they get there in
worse shape (2026-11-05).

usage: python3 -I scripts/move_positions.py <name>=<move boards.jsonl[.gz]> ...

Reads BoardDump move records (people's from boards4-*, a bot's from BotVsBot -Dbvb.boards -Dbvb.boardsKinds=move);
the state's feature names come from GeneralStateFeatures.java, in its order.
"""
import collections, gzip, json, os, re, statistics, sys

JAVA = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'gemp-lotr-server', 'src', 'main', 'java', 'com',
                    'gempukku', 'lotro', 'bots', 'rl', 'fotrstarters', 'GeneralStateFeatures.java')
SHOW = ['site', 'turn', 'burdens', 'threats', 'twilight', 'rbVitalityLeft', 'rbResistanceLeft', 'fpCompanions',
        'fpCompanionVitalityLeft', 'fpMinVitalityLeft', 'fpExhausted', 'fpDeadCompanions', 'fpPossessions',
        'myHand', 'oppHand', 'oppCompanions', 'oppPosition', 'oppBurdens', 'myDeck']


def names():
    src = open(JAVA).read()
    block = re.search(r'for \(String name : List\.of\((.*?)\)\)', src, re.S).group(1)
    return re.findall(r'"([A-Za-z0-9_]+)"', re.sub(r'//[^\n]*', '', block))


def load(path, idx):
    rows = []
    with (gzip.open(path, 'rt') if path.endswith('.gz') else open(path)) as f:
        for line in f:
            if '"kind":"move"' not in line:
                continue
            r = json.loads(line)
            opts = [o['text'] for o in r.get('opts', [])]
            if 'chosen' not in r or 'Yes' not in opts:
                continue
            st = r['state']
            rows.append({'yes': opts[r['chosen']] == 'Yes', **{k: st[i] for k, i in idx.items() if i < len(st)}})
    return rows


def main():
    nm = names()
    idx = {k: nm.index(k) for k in SHOW if k in nm}
    sets = [(a.split('=', 1)[0], load(a.split('=', 1)[1], idx)) for a in sys.argv[1:]]
    print('| | ' + ' | '.join(n for n, _ in sets) + ' |')
    print('|---|' + '---|' * len(sets))
    print('| decisions | ' + ' | '.join(str(len(r)) for _, r in sets) + ' |')
    print('| said yes | ' + ' | '.join(f'{100 * statistics.mean(x["yes"] for x in r):.1f}%' for _, r in sets) + ' |')
    for k in idx:
        print(f'| {k} | ' + ' | '.join(f'{statistics.mean(x[k] for x in r):.2f}' for _, r in sets) + ' |')
    # matched: yes-share by site, and by the fellowship's vitality left (in bands), where both have 50+ decisions
    for key, band in (('site', 1), ('fpCompanionVitalityLeft', 4), ('rbVitalityLeft', 1), ('burdens', 2)):
        print(f'\nsaid yes, by {key}' + (f' (bands of {band})' if band > 1 else '') + ':')
        print('| ' + key + ' | ' + ' | '.join(f'{n} (n)' for n, _ in sets) + ' |')
        print('|---|' + '---|' * len(sets))
        buckets = sorted({int(x[key] // band) for _, r in sets for x in r})
        for b in buckets:
            cells, ok = [], False
            for _, r in sets:
                g = [x['yes'] for x in r if int(x[key] // band) == b]
                cells.append(f'{100 * statistics.mean(g):.0f}% ({len(g)})' if len(g) >= 30 else f'- ({len(g)})')
                ok = ok or len(g) >= 30
            if ok:
                lo = b * band
                print(f'| {lo}{"-" + str(lo + band - 1) if band > 1 else ""} | ' + ' | '.join(cells) + ' |')


if __name__ == '__main__':
    main()
