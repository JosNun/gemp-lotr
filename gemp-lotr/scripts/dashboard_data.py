"""Collect what the bot work has measured into dashboard/data/data.json, for the dashboard page (dashboard/index.html).

usage: python3 -I scripts/dashboard_data.py [--every 120]     # once, or every N seconds

Reads, under ~/dev/gemp-replays/labels (outside the repo -- the page only ever gets aggregate numbers):
- hl/hl-<bot>.tsv: each bot's agreement with people on July, per kind of decision (HumanReplay --counterfactual);
- the processes running now, and the last line of each nn/*.log, for the status bar.

The bots' cost and strength (the Pareto panel) are not collected here: they are hand-kept in dashboard/bots.json.
The self-play generations, the Elo ladder from the logs' head-to-heads, the pool and the network-training panels were
removed on 2026-11-09 with the self-play work they followed.
"""
import argparse, collections, glob, json, os, re, subprocess, time

L = os.path.expanduser('~/dev/gemp-replays/labels')
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'dashboard', 'data', 'data.json')
ANCHOR = 'sp-m3'  # the reference bot, as in dashboard/bots.json
# an hl file's name is the model directory's; the page uses the bots' names
ALIASES = {'sp-m2-t35': 'sp-m3', 'search-sp-g07': 'search@sp-g07'}
CONTEXTS = ['fellowship', 'shadow', 'maneuver', 'archery', 'skirmish', 'regroup', 'optional', 'reconcile', 'move',
            'assign fp']
KIND_NAMES = {  # the counterfactual file's decision labels -> the network's kinds
    'card action: Play Fellowship action or Pass': 'fellowship', 'card action: Play Shadow action or Pass': 'shadow',
    'card action: Choose action to play or Pass': 'skirmish', 'card action: Play Maneuver action or Pass': 'maneuver',
    'card action: Play Archery action or Pass': 'archery', 'card action: Play Regroup action or Pass': 'regroup',
    'card action: Optional responses': 'optional', 'card action: Play Assignment action or Pass': 'assignment',
    'card selection: Reconcile  choose card to discard or': 'reconcile', 'assignment (FP)': 'assign fp',
    'choice: Do you want to make another move': 'move'}


def agreement():
    out = {}
    for f in sorted(glob.glob(os.path.join(L, 'hl', 'hl-*.tsv'))):
        bot = os.path.basename(f)[3:-4]
        if bot.endswith('.strict'):
            continue
        bot = ALIASES.get(bot, bot)
        asked, agree = collections.Counter(), collections.Counter()
        strict = False
        for line in open(f, encoding='utf-8'):
            p = line.rstrip('\n').split('\t')
            if len(p) < 9:
                continue
            # files from before the order-forgiving measure (2026-11-01) have 17 columns, later ones 18; the two
            # measures differ by 4-5 points, so the older files are left out rather than shown beside the newer
            if len(p) < 18:
                strict = True
                break
            k = KIND_NAMES.get(p[2]) or next((v for key, v in KIND_NAMES.items() if p[2].startswith(key)), None)
            for key in ('all', k):
                if key:
                    asked[key] += 1
                    agree[key] += p[8] == '0'
        if strict or not asked['all']:
            continue
        out[bot] = {k: round(100 * agree[k] / asked[k], 1) for k in asked}
        out[bot]['_asked'] = asked['all']
        out[bot]['_mtime'] = os.path.getmtime(f)
    return out


def status():
    jobs = []
    try:
        ps = subprocess.run(['ps', '-axo', 'etime=,command='], capture_output=True, text=True).stdout
    except OSError:
        ps = ''
    for line in ps.splitlines():
        et, _, cmd = line.strip().partition(' ')
        for key, label in (('board_net2.py', 'training the network'), ('BotVsBot', 'bot-vs-bot games'),
                           ('HumanReplay', 'replaying people\'s games'), ('BotTeach', 'teacher play-outs')):
            if key in cmd and 'grep' not in cmd and '/bin/sh -c' not in cmd and 'zsh -c' not in cmd:
                detail = ''
                if key == 'board_net2.py' and '--decision-weight 0' in cmd:
                    label = 'training the judge'
                m = re.search(r'--a (\S+) --b (\S+)', cmd)
                if m:  # general@bot-models-exp/sp-m3 -> sp-m3, search@bot-models-exp/x -> search@x
                    short = lambda b: re.sub(r'^general@', '', b.replace('bot-models-exp/', ''))
                    detail = f'{short(m.group(1))} vs {short(m.group(2))}'
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


def collect():
    data = {'updated': time.time(), 'anchor': ANCHOR, 'agreement': agreement(), 'contexts': CONTEXTS, 'status': status()}
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
