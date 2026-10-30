"""Collect what the bot work has measured into dashboard/data/data.json, for the dashboard page (dashboard/index.html).

usage: python3 -I scripts/dashboard_data.py [--every 120]     # once, or every N seconds

Reads, all under ~/dev/gemp-replays/labels (outside the repo -- the page only ever gets aggregate numbers):
- nn/net2.log, nn/judge.log, nn/net2-assign-legal.json: the combined network's and the judge's July checks;
- hl/hl-<bot>.tsv: each bot's agreement with people on July, per kind of decision (HumanReplay --counterfactual);
- nn/*.log: head-to-head ("<a> vs <b> A 49.5% +- 0.9") and pool ("<bot> pool mean 45.7%") results, from which an
  Elo ladder is fitted (Bradley-Terry, sel10-net6 = 1500);
- nn/routes-2022.jsonl: how people's games end;
- nn/rounds/*.json: one file per self-play generation (the round loop writes them), passed through as they are;
- the processes running now, and the last line of each log, for the status bar.
"""
import argparse, collections, glob, json, math, os, re, subprocess, time

L = os.path.expanduser('~/dev/gemp-replays/labels')
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'dashboard', 'data', 'data.json')
ORDER = ['sel10-f', 'sel10-gr', 'sel10-gf', 'sel10-ar1', 'sel10-big5', 'sel10-net', 'sel10-netT', 'sel10-net6', 'sel10-n2', 'sel10-n2p']
ANCHOR = 'sel10-net6'
# the bots the page shows (the rest are history; their results still go into the Elo fit). Self-play generations are
# added as they are made. Override with a file dashboard/data/main-bots.txt, one name a line.
MAIN = ['sel10-big5', 'sel10-net6', 'sel10-n2', 'sel10-n2p']
CONTEXTS = ['fellowship', 'shadow', 'maneuver', 'archery', 'skirmish', 'regroup', 'optional', 'reconcile', 'move',
            'assign fp']
KIND_NAMES = {  # the counterfactual file's decision labels -> the network's kinds
    'card action: Play Fellowship action or Pass': 'fellowship', 'card action: Play Shadow action or Pass': 'shadow',
    'card action: Choose action to play or Pass': 'skirmish', 'card action: Play Maneuver action or Pass': 'maneuver',
    'card action: Play Archery action or Pass': 'archery', 'card action: Play Regroup action or Pass': 'regroup',
    'card action: Optional responses': 'optional', 'card action: Play Assignment action or Pass': 'assignment',
    'card selection: Reconcile  choose card to discard or': 'reconcile', 'assignment (FP)': 'assign fp',
    'choice: Do you want to make another move': 'move'}


def training():
    rows = []
    legal = {}
    p = os.path.join(L, 'nn', 'net2-assign-legal.json')
    if os.path.exists(p):
        legal = json.load(open(p))
    for name, path in (('net2', 'nn/net2.log'), ('judge', 'nn/judge.log')):
        f = os.path.join(L, path)
        if not os.path.exists(f):
            continue
        for line in open(f):
            m = re.match(r'epoch ([\d.]+):', line)
            if not m:
                continue
            r = {'run': name, 'epoch': float(m.group(1))}
            for k in CONTEXTS:
                v = re.search(rf'[;,(] {re.escape(k)} ([\d.]+)%', line)
                if v:
                    r[k] = float(v.group(1))
            if name == 'net2' and str(int(r['epoch'])) in legal:
                r['assign fp'] = legal[str(int(r['epoch']))]
            a = re.search(r'win AUC ([\d.]+)', line)
            if a:
                r['auc'] = float(a.group(1))
            rows.append(r)
    return rows


def agreement():
    out = {}
    for f in sorted(glob.glob(os.path.join(L, 'hl', 'hl-*.tsv'))):
        bot = os.path.basename(f)[3:-4]
        asked, agree = collections.Counter(), collections.Counter()
        for line in open(f, encoding='utf-8'):
            p = line.rstrip('\n').split('\t')
            if len(p) < 9:
                continue
            k = KIND_NAMES.get(p[2]) or next((v for key, v in KIND_NAMES.items() if p[2].startswith(key)), None)
            for key in ('all', k):
                if key:
                    asked[key] += 1
                    agree[key] += p[8] == '0'
        out[bot] = {k: round(100 * agree[k] / asked[k], 1) for k in asked}
        out[bot]['_asked'] = asked['all']
        out[bot]['_mtime'] = os.path.getmtime(f)
    return out


RESULT = re.compile(r'^(\S+) vs (\S+) A ([\d.]+)% \+- ([\d.]+)')
POOL = re.compile(r'^(\S+) pool mean ([\d.]+)%')


def matches():
    h2h, pool = [], {}
    for f in sorted(glob.glob(os.path.join(L, 'nn', '*.log')), key=os.path.getmtime):
        for line in open(f, errors='replace'):
            line = line.strip()
            m = RESULT.match(line)
            if m:
                a, b = m.group(1), m.group(2)
                h2h.append({'a': a, 'b': b, 'rate': float(m.group(3)), 'ci': float(m.group(4)), 'source': os.path.basename(f)})
                continue
            m = POOL.match(line)
            if m:
                pool[m.group(1)] = float(m.group(2))
    # the pool's per-style lines ("<bot> vs pool-<style> A ...") are head-to-heads too; keep them per bot
    styles = collections.defaultdict(dict)
    for r in h2h:
        if r['b'].startswith('pool-'):
            styles[r['a']][r['b'][5:]] = r['rate']
    # the same pairing measured twice: keep the latest
    latest = {}
    for r in h2h:
        latest[(r['a'], r['b'])] = r
    return list(latest.values()), pool, dict(styles)


def elo(h2h):
    """Bradley-Terry by gradient ascent on the head-to-heads (not the pool's per-style lines), anchored at 1500."""
    games = [(r['a'], r['b'], r['rate'] / 100) for r in h2h if not r['b'].startswith('pool-')]
    bots = sorted({x for g in games for x in g[:2]})
    if ANCHOR not in bots or len(bots) < 2:
        return {}
    s = {b: 0.0 for b in bots}
    for _ in range(2000):
        grad = {b: 0.0 for b in bots}
        for a, b, p in games:
            q = 1 / (1 + math.exp(s[b] - s[a]))
            grad[a] += p - q
            grad[b] -= p - q
        for b in bots:
            s[b] += 0.5 * grad[b]
        shift = s[ANCHOR]
        for b in bots:
            s[b] -= shift
    # a +-x% (95%) at 50% is about 1.96 * 0.5 / sqrt(n): the Elo interval of a single comparison against the anchor
    out = {}
    for b in bots:
        ci = min((r['ci'] for r in h2h if b in (r['a'], r['b']) and not r['b'].startswith('pool-')), default=None)
        out[b] = {'elo': round(1500 + 400 / math.log(10) * s[b], 1),
                  'ci': round(400 / math.log(10) * 4 * ci / 100, 1) if ci else None}
    out[ANCHOR]['ci'] = 0.0
    return out


def people_ends():
    f = os.path.join(L, 'nn', 'routes-2022.jsonl')
    if not os.path.exists(f):
        return {}
    board, all_ = collections.Counter(), collections.Counter()
    for line in open(f):
        r = json.loads(line)
        if r['route']:
            all_[r['route']] += 1
            if r['how'] == 'board':
                board[r['route']] += 1
    norm = lambda c: {k: round(100 * v / max(1, sum(c.values())), 1) for k, v in c.items()}
    return {'decided on the board': norm(board), 'with inferred concessions': norm(all_)}


def status():
    jobs = []
    try:
        ps = subprocess.run(['ps', '-axo', 'etime=,command='], capture_output=True, text=True).stdout
    except OSError:
        ps = ''
    for line in ps.splitlines():
        et, _, cmd = line.strip().partition(' ')
        for key, label in (('board_net2.py', 'training the network'), ('BotVsBot', 'bot-vs-bot games'),
                           ('HumanReplay', 'replaying people\'s games'), ('BotTeach', 'teacher play-outs'),
                           ('selfplay', 'self-play round')):
            if key in cmd and 'grep' not in cmd and '/bin/sh -c' not in cmd and 'zsh -c' not in cmd:
                detail = ''
                if key == 'board_net2.py' and '--decision-weight 0' in cmd:
                    label = 'training the judge'
                m = re.search(r'--a general@bot-models-exp/(\S+) --b general@bot-models-exp/(\S+)', cmd)
                if m:
                    detail = f'{m.group(1)} vs {m.group(2)}'
                m = re.search(r'--counterfactual \S*bot-models-exp/(\S+)', cmd)
                if m:
                    detail = f'{m.group(1)} on July'
                jobs.append({'what': label, 'detail': detail, 'elapsed': et})
    logs = []
    for f in sorted(glob.glob(os.path.join(L, 'nn', '*.log')), key=os.path.getmtime)[-4:]:
        last = [l.strip() for l in open(f, errors='replace') if l.strip() and 'Warning' not in l and 'self.encoder' not in l]
        logs.append({'log': os.path.basename(f), 'modified': os.path.getmtime(f), 'last': last[-1][:240] if last else ''})
    seen = set()
    uniq = []
    for j in jobs:
        key = (j['what'], j['detail'])
        if key not in seen:
            seen.add(key)
            uniq.append(j)
    return {'jobs': uniq, 'logs': logs[::-1]}


def rounds():
    out = []
    for f in sorted(glob.glob(os.path.join(L, 'nn', 'rounds', '*.json'))):
        try:
            out.append(json.load(open(f)))
        except ValueError:
            pass
    return out


def collect():
    h2h, pool, styles = matches()
    agree = agreement()
    main = MAIN
    mf = os.path.join(os.path.dirname(OUT), 'main-bots.txt')
    if os.path.exists(mf):
        main = [x.strip() for x in open(mf) if x.strip()]
    main = main + [r['name'] for r in rounds() if r.get('name') and r['name'] not in main]
    ladder = {b: v for b, v in elo(h2h).items() if b in main}
    agree = {b: v for b, v in agree.items() if b in main}
    pool = {b: v for b, v in pool.items() if b in main}
    h2h = [r for r in h2h if r['a'] in main and (r['b'] in main or r['b'].startswith('pool-'))]
    bots = list(main)
    data = {'updated': time.time(), 'bots': bots, 'anchor': ANCHOR, 'training': training(), 'agreement': agree,
            'h2h': h2h, 'pool': pool, 'poolStyles': {b: v for b, v in styles.items() if b in main}, 'elo': ladder, 'peopleEnds': people_ends(),
            'status': status(), 'rounds': rounds(), 'contexts': CONTEXTS}
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    tmp = OUT + '.tmp'
    json.dump(data, open(tmp, 'w'))
    os.replace(tmp, OUT)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--every', type=int, default=0)
    args = ap.parse_args()
    while True:
        collect()
        if not args.every:
            break
        time.sleep(args.every)


if __name__ == '__main__':
    main()
