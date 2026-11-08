"""Tune the plain bot's dials together (2026-11-08, the user's "option 6"): SPSA over the trees' share at each kind of
decision (board-net2-mix) and the move threshold, each step the bot nudged both ways at once playing itself.

usage: python3 -I scripts/tune.py <state.json> [--steps 40] [--games 2] [--suite decks/real/suite-train.txt]

Every dial was set by hand, one at a time, against sp-m3-like opponents; they interact (the trees' share at skirmishes
and the move threshold both decide how hard the bot races). SPSA: each step draws a random sign for every dial, builds
theta + c*sign and theta - c*sign as model directories (links to sp-m3's files, their own mix and threshold), plays
them against each other on the training decks (so the test decks stay for the verdict), and moves theta by
a * (A's win rate - 1/2) * sign. Dials live in [0, 1] (the threshold mapped from [0.15, 0.6]). The state file holds
theta and the step, so a stopped run resumes; each step goes to the dashboard (feed "tune").
"""
import json, os, random, re, shutil, subprocess, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE = os.path.join(ROOT, 'bot-models-exp', 'sp-m3')
DIALS = ['default', 'skirmish', 'move', 'reconcile', 'assign fp', 'threshold']
START = {'default': 0.6, 'skirmish': 1.0, 'move': 1.0, 'reconcile': 0.6, 'assign fp': 0.6, 'threshold': 0.35}
TH_LO, TH_HI = 0.15, 0.6
sys.path.insert(0, os.path.join(ROOT, 'scripts'))
from feed import emit, meta  # noqa: E402


def to_unit(d, v):
    return (v - TH_LO) / (TH_HI - TH_LO) if d == 'threshold' else v


def from_unit(d, u):
    return TH_LO + u * (TH_HI - TH_LO) if d == 'threshold' else u


def make_dir(name, theta):
    """sp-m3's files linked, with this theta's mix and move threshold."""
    path = os.path.join(ROOT, 'bot-models-exp', name)
    shutil.rmtree(path, ignore_errors=True)
    os.makedirs(path)
    for f in os.listdir(BASE):
        if f in ('board-net2-mix', 'move-threshold'):
            continue
        os.symlink(os.path.realpath(os.path.join(BASE, f)), os.path.join(path, f))
    with open(os.path.join(path, 'board-net2-mix'), 'w') as f:
        for d in DIALS[:-1]:
            f.write(f'{d} {from_unit(d, theta[d]):.4f}\n')
    with open(os.path.join(path, 'move-threshold'), 'w') as f:
        f.write(f'{from_unit("threshold", theta["threshold"]):.4f}\n')
    return path


def play(a, b, games, suite):
    out = subprocess.run(['./bot-vs-bot.sh', '--suite', suite, '--a', 'general@' + a, '--b', 'general@' + b,
                          '--games', str(games), '--max-decisions', '3000'], cwd=ROOT, capture_output=True, text=True,
                         env={**os.environ, 'JAVA_OPTS': '-Dfeed=off'}).stdout
    m = re.search(r'A \(general@[^)]*\) won (\d+), B \(general@[^)]*\) won (\d+)', out)
    if not m:
        raise RuntimeError('no result:\n' + out[-2000:])
    wa, wb = int(m.group(1)), int(m.group(2))
    return wa / max(1, wa + wb), wa + wb


def main():
    args = sys.argv[1:]
    state_path = args[0]
    opt = {'--steps': 40, '--games': 2, '--suite': 'decks/real/suite-train.txt'}
    for k in list(opt):
        if k in args:
            i = args.index(k)
            opt[k] = type(opt[k])(args[i + 1])
    state = json.load(open(state_path)) if os.path.exists(state_path) else \
        {'step': 0, 'theta': {d: to_unit(d, START[d]) for d in DIALS}, 'history': []}
    meta('tune', title='Tuning the plain bot\'s dials (SPSA)', hint='theta+ against theta- on the training decks', x='step')
    rng = random.Random(state['step'] + 7)
    while state['step'] < opt['--steps']:
        k = state['step'] + 1
        c = 0.15 / k ** 0.101  # perturbation, in unit-dial terms
        a = 0.6 / (k + 5) ** 0.602  # step size
        sign = {d: rng.choice((-1, 1)) for d in DIALS}
        th = state['theta']
        plus = {d: min(1.0, max(0.0, th[d] + c * sign[d])) for d in DIALS}
        minus = {d: min(1.0, max(0.0, th[d] - c * sign[d])) for d in DIALS}
        pa, pb = make_dir('tune-plus', plus), make_dir('tune-minus', minus)
        score, n = play(os.path.relpath(pa, ROOT), os.path.relpath(pb, ROOT), opt['--games'], opt['--suite'])
        for d in DIALS:
            move = max(-0.1, min(0.1, a * (score - 0.5) / c))  # one freak result moves no dial more than 0.1
            th[d] = min(1.0, max(0.0, th[d] + move * sign[d]))
        state['step'] = k
        rec = {'step': k, 'score': round(100 * score, 2), 'games': n,
               **{d.replace(' ', '_'): round(from_unit(d, th[d]), 3) for d in DIALS}, 'done': k, 'total': opt['--steps']}
        state['history'].append(rec)
        emit('tune', **rec)
        json.dump(state, open(state_path, 'w'), indent=1)
        print(json.dumps(rec), flush=True)


if __name__ == '__main__':
    main()
