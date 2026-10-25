#!/usr/bin/env python3
"""A small win model on people's positions (race_positions.py files): how people's results price a companion
against a site of the race, and against the rest of the board.

usage: race_value.py out.weights train.csv... --test test.csv [--features name,name,...]

Each row is won/lost from one player's side at the start of a turn, then the columns race_positions.py documents.
Logistic regression by Newton's method on a few named features; prints the weights, held-out AUC, and the exchange
rates in companions of mine. The weights file ("name weight" lines, "intercept b") is what HumanReplay --move-value
(Counterfactual.loadMoveValue) reads, over set-2 GeneralStateFeatures from the deciding player's side.

Feature names, each of which must mean the same thing as the live GeneralStateFeatures feature of that name:
  a column            myPosition, myCompanionVitalityLeft, myDead, ... read as is
  a side of the board fpPossessions, fpConditions, fpAllies, fpExhausted, fpCompanions (the current player's -- at
                      the start of a turn the Free Peoples player is the current player), shadowConditions,
                      shadowPossessions (the other player's), threats (the current player's): GeneralStateFeatures
                      counts these over *active* cards only, which is exactly the current player's Free Peoples cards
                      and the other player's Shadow cards, so they are built here from the per-owner columns.
Transforms, written "transform:name" (the Java side must compute each one from the named live feature):
  cube9:x   (x / 9)^3                         -- already read by Counterfactual.loadMoveValue
  fpsign:x  x if iAmFp else -x                -- NEW. A side-of-board count describes the bot's cards on its own
                                                 turn and the opponent's on the opponent's turn; on mirrored rows an
                                                 unsigned count can only come out at ~0. fpsign:fpPossessions is "my
                                                 Free Peoples possessions on my turn, minus theirs on theirs".
  le2:x     1 if x <= 2 else 0                -- NEW (threshold; tried for resistance, not in the recommended model)
  a*b       product of two features, each of which may itself be transformed -- NEW (tried for interactions, not in
                                                 the recommended model)
"""
import sys
import numpy as np

COLUMNS = ['turn', 'iAmFp', 'myPosition', 'oppPosition', 'myCompanions', 'oppCompanions', 'myRbResistance',
           'oppRbResistance', 'myHand', 'oppHand', 'myDeck', 'oppDeck', 'myWounds', 'oppWounds',
           'myBurdens', 'oppBurdens', 'myCompanionVitalityLeft', 'oppCompanionVitalityLeft', 'myExhausted',
           'oppExhausted', 'myAllies', 'oppAllies', 'myFpPossessions', 'oppFpPossessions', 'myFpConditions',
           'oppFpConditions', 'myShadowPossessions', 'oppShadowPossessions', 'myShadowConditions',
           'oppShadowConditions', 'myMinions', 'oppMinions', 'myDead', 'oppDead', 'myDiscard', 'oppDiscard',
           'myThreats', 'oppThreats', 'myCultureTokens', 'oppCultureTokens']
# Side-of-board features as GeneralStateFeatures computes them: (column when it is my turn, column when it is theirs).
SIDE = {'fpCompanions': ('myCompanions', 'oppCompanions'), 'fpAllies': ('myAllies', 'oppAllies'),
        'fpPossessions': ('myFpPossessions', 'oppFpPossessions'),
        'fpConditions': ('myFpConditions', 'oppFpConditions'), 'fpExhausted': ('myExhausted', 'oppExhausted'),
        'shadowConditions': ('oppShadowConditions', 'myShadowConditions'),
        'shadowPossessions': ('oppShadowPossessions', 'myShadowPossessions'),
        'threats': ('myThreats', 'oppThreats')}
# Fit on Jan-Jun + Aug-Nov 2022 (183,630 positions), held-out July AUC (15,342): the race alone 0.641; this 0.670.
# In: the five side-of-board counts (each worth 0.002-0.010 of AUC on its own, signs the same in every one of the 11
# months). Out, and why (details in the report that came with this change):
#   myHand/oppHand           8 at the start of every turn (reconcile refills it): carries nothing
#   myDeck/oppDeck           backwards (-0.05 of a companion per card left: decks that draw and cycle win)
#   myDiscard/oppDiscard     no weight (0.00 of a companion per card), no AUC; like deck size, mostly deck style
#   myPosition/oppPosition   the linear term made moving from site 1 to 2 worth -0.22 of a companion; cube9 alone is
#                            monotone and scores no worse (+0.002)
#   iAmFp                    its sign flips between specifications (it soaks up the side-of-board means); 0 AUC
#   myCompanionVitalityLeft  +0.25 of a companion per point, no AUC (it is companions and wounds again)
#   fpExhausted, myDead, myBurdens, threats, le2:RbResistance, companions x position: right signs, no AUC
# Not live (would need new GeneralStateFeatures): the same five counts by owner, active or not (+0.015), and
# culture tokens on each player's cards (+0.004).
FEATURES = ['cube9:myPosition', 'cube9:oppPosition', 'myCompanions', 'oppCompanions', 'myRbResistance',
            'oppRbResistance', 'myWounds', 'oppWounds', 'fpsign:fpPossessions', 'fpsign:fpConditions',
            'fpsign:fpAllies', 'fpsign:shadowConditions', 'fpsign:shadowPossessions']

def column(x, name):
    if name in SIDE:
        mine, theirs = SIDE[name]
        fp = x[:, COLUMNS.index('iAmFp')] == 1
        return np.where(fp, x[:, COLUMNS.index(mine)], x[:, COLUMNS.index(theirs)])
    return x[:, COLUMNS.index(name)]

def feature(x, f):
    if '*' in f:
        a, b = f.split('*', 1)
        return feature(x, a) * feature(x, b)
    if f.startswith('cube9:'):
        return (feature(x, f[6:]) / 9) ** 3
    if f.startswith('fpsign:'):
        return (2 * x[:, COLUMNS.index('iAmFp')] - 1) * feature(x, f[7:])
    if f.startswith('le2:'):
        return (feature(x, f[4:]) <= 2).astype(float)
    return column(x, f)

def load_raw(paths):
    a = np.vstack([np.loadtxt(p, delimiter=',', ndmin=2) for p in paths])
    return a[:, 0], a[:, 1:]

def design(x, features):
    return np.column_stack([feature(x, f) for f in features])

def fit(y, x, l2=1e-3):
    X = np.column_stack([np.ones(len(x)), x])
    w = np.zeros(X.shape[1])
    for _ in range(50):
        p = 1 / (1 + np.exp(-X @ w))
        g = X.T @ (y - p) - l2 * w
        H = (X * (p * (1 - p))[:, None]).T @ X + l2 * np.eye(len(w))
        step = np.linalg.solve(H, g)
        w += step
        if np.abs(step).max() < 1e-8:
            break
    return w

def auc(y, s):
    """Mann-Whitney AUC with ties given half credit (most features are small integers, so ties are common)."""
    order = np.argsort(s, kind='mergesort')
    ss = s[order]
    r = np.empty(len(s))
    # average ranks over runs of equal scores
    bounds = np.flatnonzero(np.diff(ss)) + 1
    starts = np.concatenate([[0], bounds]); ends = np.concatenate([bounds, [len(s)]])
    avg = np.repeat((starts + ends + 1) / 2, ends - starts)
    r[order] = avg
    pos = y == 1
    return (r[pos].sum() - pos.sum() * (pos.sum() + 1) / 2) / (pos.sum() * (~pos).sum())

def score(w, x):
    return np.column_stack([np.ones(len(x)), x]) @ w

def main(args):
    out = args.pop(0)
    features = FEATURES
    if '--features' in args:
        i = args.index('--features'); features = args[i + 1].split(','); args = args[:i] + args[i + 2:]
    test = []
    if '--test' in args:
        i = args.index('--test'); test = args[i + 1:]; args = args[:i]
    y, raw = load_raw(args)
    w = fit(y, design(raw, features))
    print(f'{len(y)} positions, {y.mean():.1%} won by the deciding player')
    c = w[1 + features.index('myCompanions')] if 'myCompanions' in features else None
    for n, v in zip(['intercept'] + features, w):
        rate = f'  = {v / c:+.3f} of my companions' if c and n != 'intercept' else ''
        print(f'  {n:28} {v:+.4f}{rate}')
    if test:
        yt, rt = load_raw(test)
        print(f'held-out AUC {auc(yt, score(w, design(rt, features))):.4f} on {len(yt)} positions')
    ix = {n: 1 + features.index(n) for n in features}
    if c and 'cube9:myPosition' in ix:
        lin = w[ix['myPosition']] if 'myPosition' in ix else 0
        def site(p):  # what one more site is worth at site p, in companions of mine
            return (lin + w[ix['cube9:myPosition']] * ((p + 1) ** 3 - p ** 3) / 729) / c
        print('in companions of mine: one more site at ' + ', '.join(f'{p} {site(p):+.2f}' for p in range(1, 9)))
    with open(out, 'w') as f:
        f.write(f'intercept {w[0]}\n')
        for n, v in zip(features, w[1:]):
            f.write(f'{n} {v}\n')

if __name__ == '__main__':
    main(sys.argv[1:])
