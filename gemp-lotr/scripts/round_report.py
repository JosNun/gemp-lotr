"""One self-play generation's numbers for the dashboard (labels/nn/rounds/gen-NN.json).

usage: python3 -I scripts/round_report.py --gen N --name <bot> --parent <bot> --log <round log> --games <game log jsonl>
           [--hl <hl-<bot>.tsv>] [--adopted yes|no]

Reads the round's log (result lines "<a> vs <b> A x% +- y", "<bot> pool mean x%", "N games in Xs", the fine-tune's
"search overrules the network on x%"), the evaluation games' GameLog lines (how they ended; the learner's style
fingerprints, seat A), and the July agreement file.
"""
import argparse, collections, json, os, re

KIND = {'card action: Play Fellowship action or Pass': 'fellowship', 'card action: Play Shadow action or Pass': 'shadow',
        'card action: Choose action to play or Pass': 'skirmish', 'assignment (FP)': 'assign fp',
        'choice: Do you want to make another move': 'move'}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gen', type=int, required=True)
    ap.add_argument('--name', required=True)
    ap.add_argument('--parent', required=True)
    ap.add_argument('--log', required=True)
    ap.add_argument('--games', default=None)
    ap.add_argument('--hl', default=None)
    ap.add_argument('--adopted', default=None)
    ap.add_argument('--out', default=os.path.expanduser('~/dev/gemp-replays/labels/nn/rounds'))
    args = ap.parse_args()
    r = {'gen': args.gen, 'name': args.name, 'parent': args.parent, 'adopted': args.adopted == 'yes'}
    text = open(args.log, errors='replace').read() if os.path.exists(args.log) else ''
    for a, b, rate, ci in re.findall(r'^(\S+) vs (\S+) .*?A ([\d.]+)% \+- ([\d.]+)', text, re.M):
        if a != args.name:
            continue
        if b == args.parent:
            r['vsPrev'], r['vsPrevCi'] = float(rate), float(ci)
        elif b == 'sel10-net6':
            r['vsNet6'], r['vsNet6Ci'] = float(rate), float(ci)
    for a, b, rate, ci in re.findall(r'^(\S+) vs (\S+) .*?A ([\d.]+)% \+- ([\d.]+)', text, re.M):
        if a == args.name + '-pure' and b == 'sel10-net6':
            r['pureVsNet6'], r['pureVsNet6Ci'] = float(rate), float(ci)
    m = re.search(rf'^{re.escape(args.name)} pool mean ([\d.]+)%', text, re.M)
    if m:
        r['pool'] = float(m.group(1))
    m = re.findall(r'search overrules the network on ([\d.]+)% of (\d+)', text)
    if m:
        r['overrule'], r['searched'] = float(m[-1][0]), int(m[-1][1])
    m = re.findall(r'(\d+) games in (\d+)s', text)
    if m:
        n, s = map(int, m[0])  # the play phase's; the evaluation's own come after
        r['gamesPerHour'] = round(n / max(s, 1) * 3600, 1)
    if args.hl and os.path.exists(args.hl):
        asked, agree = collections.Counter(), collections.Counter()
        for line in open(args.hl, encoding='utf-8'):
            p = line.rstrip('\n').split('\t')
            if len(p) < 9:
                continue
            for k in ('all', KIND.get(p[2])):
                if k:
                    asked[k] += 1
                    agree[k] += p[8] == '0'
        r['agreement'] = round(100 * agree['all'] / max(1, asked['all']), 1)
        r['agreementByKind'] = {k: round(100 * agree[k] / asked[k], 1) for k in asked if k != 'all'}
    if args.games and os.path.exists(args.games):
        ends, n = collections.Counter(), 0
        fp = collections.defaultdict(list)
        played = collections.Counter()
        for line in open(args.games):
            g = json.loads(line)
            if g.get('error') or not g.get('route'):
                continue
            n += 1
            ends[g['route']] += 1
            a = g['players'].get('A', {})
            fp['turns'].append(g['turns'])
            for k in ('site', 'deadCompanions'):
                if a.get(k) is not None:
                    fp[k].append(a[k])
            if a.get('moveAsked'):
                fp['moveAgain'].append(a['moveYes'] / a['moveAsked'])
            if a.get('shadowTurns'):
                fp['minionsPerShadowTurn'].append(a['minions'] / a['shadowTurns'])
            if a.get('unspentTwilight') is not None:
                fp['unspentTwilight'].append(a['unspentTwilight'])
            for ctx, (asked, passed) in a.get('passes', {}).items():
                fp[f'pass:{ctx}'].append(passed / asked if asked else 0)
            played.update(a.get('played', {}))
        if n:
            r['games'] = n
            r['ends'] = {k: round(100 * v / n, 1) for k, v in ends.items() if k in ('site9', 'kill', 'corrupt')}
            r['style'] = {k: round(sum(v) / len(v), 3) for k, v in fp.items() if v}
            r['moveAgain'] = round(100 * r['style'].get('moveAgain', 0), 1)
            r['topCards'] = played.most_common(25)
    os.makedirs(args.out, exist_ok=True)
    json.dump(r, open(os.path.join(args.out, f'gen-{args.gen:02d}.json'), 'w'), indent=1)
    print(json.dumps({k: v for k, v in r.items() if k not in ('topCards', 'style', 'agreementByKind')}))


if __name__ == '__main__':
    main()
