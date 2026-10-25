#!/usr/bin/env python3
"""A small win model on people's positions (race_positions.py files): how people's results price a companion
against a site of the race.

usage: race_value.py out.weights train.csv... --test test.csv

Each row is won/lost from one player's side at the start of a turn, then turn, whether it is that player's turn,
both fellowships' sites, companions in play on each side, each Ring-bearer's resistance left, cards in hand and
in the draw deck on each side, and wounds on each side's companions. Logistic regression by
Newton's method on a few named features; prints the weights, held-out AUC, and the exchange rates. The weights file
("name weight" lines, "intercept b") is what HumanReplay --move-value reads.
"""
import re, sys
import numpy as np

COLUMNS = ['turn', 'iAmFp', 'myPosition', 'oppPosition', 'myCompanions', 'oppCompanions', 'myRbResistance',
           'oppRbResistance', 'myHand', 'oppHand', 'myDeck', 'oppDeck', 'myWounds', 'oppWounds']
# Hand size is 8 at the start of every turn (reconcile refills it); deck size comes out backwards (-0.05 of a companion
# per card left: decks that draw and cycle win), so neither prices a card spent and both are left out.
FEATURES = ['iAmFp', 'myPosition', 'oppPosition', 'myCompanions', 'oppCompanions', 'myRbResistance',
            'oppRbResistance', 'myWounds', 'oppWounds', 'cube9:myPosition', 'cube9:oppPosition']

def load(paths):
    rows = [np.loadtxt(p, delimiter=',', ndmin=2) for p in paths]
    a = np.vstack(rows)
    y, x = a[:, 0], a[:, 1:]
    return y, np.column_stack([(x[:, COLUMNS.index(f[6:])] / 9) ** 3 if f.startswith('cube9:') else x[:, COLUMNS.index(f)]
                               for f in FEATURES])

def fit(y, x, l2=1e-3):
    X = np.column_stack([np.ones(len(x)), x])
    w = np.zeros(X.shape[1])
    for _ in range(30):
        p = 1 / (1 + np.exp(-X @ w))
        g = X.T @ (y - p) - l2 * w
        H = (X * (p * (1 - p))[:, None]).T @ X + l2 * np.eye(len(w))
        step = np.linalg.solve(H, g)
        w += step
        if np.abs(step).max() < 1e-8:
            break
    return w

def auc(y, s):
    order = np.argsort(s)
    r = np.empty(len(s)); r[order] = np.arange(1, len(s) + 1)
    pos = y == 1
    return (r[pos].sum() - pos.sum() * (pos.sum() + 1) / 2) / (pos.sum() * (~pos).sum())

args = sys.argv[1:]
out = args.pop(0)
test = []
if '--test' in args:
    i = args.index('--test'); test = args[i + 1:]; args = args[:i]
y, x = load(args)
w = fit(y, x)
print(f'{len(y)} positions, {y.mean():.1%} won by the deciding player')
for n, v in zip(['intercept'] + FEATURES, w):
    print(f'  {n:16} {v:+.4f}')
if test:
    yt, xt = load(test)
    print(f'held-out AUC {auc(yt, np.column_stack([np.ones(len(xt)), xt]) @ w):.3f} on {len(yt)} positions')
c = w[1 + FEATURES.index('myCompanions')]
ix = {n: 1 + FEATURES.index(n) for n in FEATURES}
def site(p):  # what one more site is worth at site p, in companions of mine
    return (w[ix['myPosition']] + w[ix['cube9:myPosition']] * ((p + 1) ** 3 - p ** 3) / 729) / c
print('in companions of mine: one more site at ' + ', '.join(f'{p} {site(p):+.2f}' for p in range(1, 9))
      + f'; one of their companions {w[ix["oppCompanions"]] / c:+.2f}; a wound on mine {w[ix["myWounds"]] / c:+.2f}; a point of my resistance '
      f'{w[ix["myRbResistance"]] / c:+.2f}; my turn {w[ix["iAmFp"]] / c:+.2f}')
with open(out, 'w') as f:
    f.write(f'intercept {w[0]}\n')
    for n, v in zip(FEATURES, w[1:]):
        f.write(f'{n} {v}\n')
