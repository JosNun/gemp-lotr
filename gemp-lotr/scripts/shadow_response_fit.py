#!/usr/bin/env python3
"""Fit and evaluate the Shadow response model on shadow_response_data.py's rows: what the Shadow player fields in a
Shadow phase, from the twilight the Free Peoples player handed over and what FP could know about the opponent.

usage: python3 -I scripts/shadow_response_fit.py /tmp/resp/shadow_phases.csv /tmp/resp

Train on every month but July 2022; report on July (held out). Least squares (tiny ridge, unpenalised intercept) for
strength; least squares and a Poisson GLM (log link, IRLS) for minion counts. Feature sets, nested:
  base      twilight and its saturation, site, moves, FP companions, Shadow hand size, minions already in play
  +revealed what FP has seen of the Shadow player's cards (KnowledgeFeatures' oppSeen* equivalents)
  +deck     the Shadow player's deck as far as the manifest knows it (hindsight: every card they drew all game)
  +hand     the Shadow player's actual hand at the phase start (hidden from FP; the real ceiling)
Writes /tmp/resp/report.txt (everything printed), /tmp/resp/response.weights (the recommended strength model),
/tmp/resp/response_minions.weights (the minions-played model, same inputs).

Event semantics were checked by printing whole turns (shadow_response_data.py), e.g. 2022-07
Allan2020_vs_GreenRouge turn 7: pool 14 at GPC 'Shadow'; Easterling Army (13), Easterling Skirmisher (5, adds 4
twilight on play), Ulaire Enquea (11) and Surging Up played; pool 4 at GPC 'Maneuver'; row says yMinions 3, yStr 29,
yTwilightEnd 4, yKilled 1 (Gandalf killed by Enquea's maneuver ability -- inside the window, as intended).
"""
import csv, os, sys
import numpy as np

CULTURES = ['SAURON', 'WRAITH', 'RAIDER', 'ISENGARD', 'MORIA', 'GOLLUM', 'DUNLAND']
KNOT = 16          # twilight beyond which fielded strength stops growing (see the bin table in the report)
STR_PRIOR = 9.0    # imputed mean revealed minion strength when none has been seen (set from train in main)


def load(path):
    rows = list(csv.DictReader(open(path)))
    cols = [c for c in rows[0] if c not in ('month', 'game', 'fp', 'sh', 'shCulture')]
    data = {c: np.array([float(r[c]) for r in rows]) for c in cols}
    data['month'] = np.array([r['month'] for r in rows])
    data['shCulture'] = np.array([r['shCulture'] for r in rows])
    # what FP has watched this Shadow player do earlier in the same game: earlier Shadow phases' strength played and
    # the twilight they started with (rows come out of shadow_response_data.py in game order, ordinal ascending)
    n = len(rows)
    hs, ht, hn = np.zeros(n), np.zeros(n), np.zeros(n)
    acc = {}
    for i, r in enumerate(rows):
        key = (r['game'], r['sh'])
        s, t, k = acc.get(key, (0.0, 0.0, 0))
        hs[i], ht[i], hn[i] = s, t, k
        acc[key] = (s + float(r['yStrPlayed']), t + float(r['twilight']), k + 1)
    data['histStr'], data['histTwilight'], data['histPhases'] = hs, ht, hn
    return data


def features(d, prior):
    """name -> column. Every name is defined so it can be recomputed live; see definitions() for the text."""
    T = d['twilight']
    f = {}
    # base
    f['twilight'] = T
    f['twilightOver16'] = np.maximum(0, T - KNOT)
    f['site'] = d['site']
    f['moves'] = d['moves']
    f['fpCompanions'] = d['fpComps']
    f['oppHand'] = d['shHand']
    f['minionStrength'] = d['carryStr']
    # revealed profile (what FP has seen)
    seen = d['rvMinions']
    mean_str = np.where(seen > 0, d['rvMinStr'] / np.maximum(1, seen), prior['str'])
    mean_cost = np.where(seen > 0, d['rvMinCost'] / np.maximum(1, seen), prior['cost'])
    shadow_seen = sum(d['rv_' + c] for c in CULTURES + ['MEN', 'ORC', 'URUK_HAI'])
    Tc = np.minimum(T, KNOT)
    f['oppSeenMinions'] = seen
    f['oppSeenNone'] = (seen == 0).astype(float)
    f['seenMinionStrMean'] = mean_str - prior['str']
    f['twilightCapped_x_seenStrMean'] = Tc * (mean_str - prior['str'])
    f['seenMinionCostMean'] = mean_cost - prior['cost']
    f['twilightCapped_x_seenStrPerCost'] = Tc * (mean_str / np.maximum(1, mean_cost) - prior['spc'])
    for c in CULTURES:
        f['seenShare_' + c] = d['rv_' + c] / np.maximum(1, shadow_seen)
        f['twilightCapped_x_seenShare_' + c] = Tc * f['seenShare_' + c]
    f['seenArchers'] = d['rvArcher']
    f['seenFierce'] = d['rvFierce']
    # history: what FP watched this Shadow player field earlier this game (not in the bot's features today)
    hk = d['histPhases']
    hratio = np.where(d['histTwilight'] > 0, d['histStr'] / np.maximum(1, d['histTwilight']), prior['ratio'])
    f['histPhases'] = hk
    f['histStrPerTwilight'] = hratio - prior['ratio']
    f['twilightCapped_x_histStrPerTwilight'] = Tc * (hratio - prior['ratio'])
    # deck (hindsight)
    dmean = np.where(d['dkMinions'] > 0, d['dkMinStrMean'], prior['str'])
    dcost = np.where(d['dkMinions'] > 0, d['dkMinCostMean'], prior['cost'])
    f['deckMinionStrMean'] = dmean - prior['str']
    f['twilightCapped_x_deckStrMean'] = Tc * (dmean - prior['str'])
    f['deckMinionCostMean'] = dcost - prior['cost']
    f['twilightCapped_x_deckStrPerCost'] = Tc * (dmean / np.maximum(1, dcost) - prior['spc'])
    f['deckMinionFrac'] = d['dkMinFrac']
    for c in CULTURES:
        f['deckShare_' + c] = d['dk_' + c]
    # hand (hidden)
    f['handMinions'] = d['hdMinions']
    f['handMinionStr'] = d['hdMinStr']
    f['handAffordableStr'] = d['hdAffStr']
    f['handAffordableMinions'] = d['hdAffMinions']
    return f


BASE = ['twilight', 'twilightOver16', 'site', 'moves', 'fpCompanions', 'oppHand', 'minionStrength']
# the recommended profile: how many minions FP has seen, and the slope of strength on twilight by seen culture mix
REVEALED = ['oppSeenMinions'] + ['twilightCapped_x_seenShare_' + c for c in CULTURES]
# everything else tried; none of it moves July R2 by more than 0.001
REVEALED_WIDE = ['oppSeenNone', 'seenMinionStrMean', 'twilightCapped_x_seenStrMean', 'seenArchers', 'seenFierce',
                 'seenMinionCostMean', 'twilightCapped_x_seenStrPerCost'] + ['seenShare_' + c for c in CULTURES]
DECK = ['deckMinionStrMean', 'twilightCapped_x_deckStrMean', 'deckMinionCostMean', 'twilightCapped_x_deckStrPerCost',
        'deckMinionFrac'] + ['deckShare_' + c for c in CULTURES]
HISTORY = ['histPhases', 'histStrPerTwilight', 'twilightCapped_x_histStrPerTwilight']
HAND = ['handMinions', 'handMinionStr', 'handAffordableStr', 'handAffordableMinions']
SETS = [('base', BASE), ('+revealed', BASE + REVEALED), ('+revealed wide', BASE + REVEALED + REVEALED_WIDE),
        ('+history (no revealed)', BASE + HISTORY), ('+revealed+history', BASE + REVEALED + HISTORY),
        ('+deck (no revealed)', BASE + DECK), ('+revealed+deck', BASE + REVEALED + DECK),
        ('+hand (no profile)', BASE + HAND), ('+revealed+deck+hand', BASE + REVEALED + DECK + HAND)]


def design(f, names):
    return np.column_stack([np.ones(len(f[names[0]]))] + [f[n] for n in names])


def ols(X, y, ridge=1e-3):
    P = ridge * len(y) * np.eye(X.shape[1])
    P[0, 0] = 0
    sd = X.std(0)
    sd[0] = 1
    sd[sd == 0] = 1
    Xs = X / sd
    w = np.linalg.solve(Xs.T @ Xs + P, Xs.T @ y)
    return w / sd


def poisson(X, y, iters=50, ridge=1e-3):
    """Poisson GLM, log link, IRLS on standardised columns with a small ridge (unpenalised intercept)."""
    mu0, sd = X.mean(0), X.std(0)
    mu0[0], sd[0] = 0, 1
    sd[sd == 0] = 1
    Xs = (X - mu0) / sd
    Xs[:, 0] = 1
    P = ridge * len(y) * np.eye(X.shape[1])
    P[0, 0] = 0
    w = np.zeros(X.shape[1])
    w[0] = np.log(max(y.mean(), 1e-6))
    for _ in range(iters):
        eta = np.clip(Xs @ w, -20, 20)
        mu = np.exp(eta)
        z = eta + (y - mu) / mu
        w_new = np.linalg.solve(Xs.T @ (Xs * mu[:, None]) + P, Xs.T @ (mu * z))
        done = np.max(np.abs(w_new - w)) < 1e-8
        w = w_new
        if done:
            break
    out = w / sd
    out[0] = w[0] - (w[1:] * mu0[1:] / sd[1:]).sum()
    return out


def r2(y, p):
    return 1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum()


def mae(y, p):
    return np.abs(y - p).mean()


def definitions():
    return {
        'intercept': 'constant 1',
        'twilight': 'twilight pool at the start of the Shadow phase (GeneralStateFeatures "twilight" read when the '
                    'Shadow phase begins; for an FP forecast: current pool + twilight still to be added by FP plays and '
                    'the move(s), i.e. the projected pool)',
        'twilightOver16': 'max(0, twilight - 16)',
        'site': 'site number the fellowship is at after moving (GeneralStateFeatures "site" in the Shadow phase; '
                '"myPosition" for the FP player; forecasting from the Fellowship phase, use the site being moved to)',
        'moves': 'moves made this turn including the one just made (GeneralStateFeatures "moves" = getMoveCount())',
        'fpCompanions': 'FP companions in play (GeneralStateFeatures "fpCompanions"; replay: printed companions in '
                        'FREE_CHARACTERS owned by FP)',
        'oppHand': 'Shadow player\'s hand size (GeneralStateFeatures "oppHand" from the FP side)',
        'minionStrength': 'strength of Shadow minions already in play at the phase start (replay: PRINTED strength; '
                          'GeneralStateFeatures "minionStrength" is modified strength -- nearly always 0 here anyway, '
                          'non-zero only on a second move)',
        'oppSeenMinions': 'distinct Shadow minions of the opponent seen in public before this phase (KnowledgeFeatures '
                          '"oppSeenMinions")',
        'twilightCapped_x_seenShare_<CULTURE>': 'min(twilight, 16) * oppSeen_<CULTURE> / max(1, sum of oppSeen_<C> '
                      'over C in DUNLAND ISENGARD MEN MORIA ORC RAIDER SAURON URUK_HAI WRAITH GOLLUM) '
                      '(KnowledgeFeatures "oppSeen_<CULTURE>"; 0 when nothing Shadow has been seen)',
    }


def main():
    path, out = sys.argv[1], sys.argv[2]
    os.makedirs(out, exist_ok=True)
    rep = open(os.path.join(out, 'report.txt'), 'w')

    def say(*a):
        s = ' '.join(str(x) for x in a)
        print(s)
        rep.write(s + '\n')

    d = load(path)
    test = d['month'] == '07'
    train = ~test
    seen = d['rvMinions'] > 0
    prior = {'str': float(np.round((d['rvMinStr'][train & seen] / d['rvMinions'][train & seen]).mean(), 1)),
             'cost': float(np.round((d['rvMinCost'][train & seen] / d['rvMinions'][train & seen]).mean(), 1))}
    prior['spc'] = float(np.round(prior['str'] / prior['cost'], 2))
    prior['ratio'] = float(np.round(d['yStrPlayed'][train].sum() / d['twilight'][train].sum(), 2))
    say(f'rows {len(test)}: train {train.sum()} (2022-01..11 except 07), test {test.sum()} (2022-07)')
    say(f'priors (train): mean seen minion strength {prior["str"]}, cost {prior["cost"]}, str/cost {prior["spc"]}')
    f = features(d, prior)

    say('\n== Fielded strength by twilight at the Shadow phase start (all rows) ==')
    say('twilight  n  mean minions played  mean strength played  mean strength in play at end')
    T = d['twilight']
    for lo in range(0, 32, 2):
        m = (T >= lo) & (T < lo + 2) if lo < 30 else T >= 30
        if m.sum():
            say(f'{lo:>3}{"+" if lo == 30 else "-" + str(lo + 1):<4}{m.sum():>7}  {d["yMinions"][m].mean():6.2f}  '
                f'{d["yStrPlayed"][m].mean():7.2f}  {d["yStr"][m].mean():7.2f}')

    results = {}
    for target in ('yStr', 'yMinions', 'yKilled'):
        y = d[target]
        say(f'\n== Target {target}: held-out July ==  (train mean {y[train].mean():.2f}, sd {y[train].std():.2f}; '
            f'predicting the train mean: MAE {mae(y[test], np.full(test.sum(), y[train].mean())):.3f})')
        say(f'{"features":<24}{"terms":>6}{"R2 train":>10}{"R2 July":>9}{"MAE July":>10}' +
            ('   Poisson: R2 July  MAE July' if target != 'yStr' else ''))
        for name, cols in SETS:
            X = design(f, cols)
            w = ols(X[train], y[train])
            p = X @ w
            line = f'{name:<24}{len(cols):>6}{r2(y[train], p[train]):>10.3f}{r2(y[test], p[test]):>9.3f}' \
                   f'{mae(y[test], p[test]):>10.3f}'
            if target != 'yStr':
                wp = poisson(X[train], y[train])
                pp = np.exp(np.clip(X @ wp, -20, 20))
                line += f'   {r2(y[test], pp[test]):>14.3f}{mae(y[test], pp[test]):>10.3f}'
            say(line)
            results[(target, name)] = (w, cols, p)

    say('\n== yStr, July, by how much FP has seen (oppSeenMinions) ==')
    say(f'{"oppSeenMinions":<16}{"n":>7}' + ''.join(f'{n:>22}' for n, _ in SETS[:5]))
    seen = d['rvMinions']
    for lo, hi in ((0, 0), (1, 2), (3, 5), (6, 9), (10, 99)):
        m = test & (seen >= lo) & (seen <= hi)
        line = f'{str(lo) + "-" + str(hi):<16}{m.sum():>7}'
        for n, _ in SETS[:5]:
            p = results[('yStr', n)][2]
            line += f'{"R2 %.3f MAE %.2f" % (r2(d["yStr"][m], p[m]), mae(d["yStr"][m], p[m])):>22}'
        say(line)

    # recommended: +revealed (what FP can know), least squares
    say('\n== Recommended model: base + revealed profile, least squares ==')
    for target, fname in (('yStr', 'response.weights'), ('yMinions', 'response_minions.weights')):
        w, cols, p = results[(target, '+revealed')]
        with open(os.path.join(out, fname), 'w') as o:
            o.write(f'# Shadow response model: {target} (see scripts/shadow_response_fit.py); fit on 2022 Movie block '
                    f'replays except July; inputs defined in /tmp/resp/report.txt\n')
            o.write(f'intercept {w[0]:.6f}\n')
            for n, v in zip(cols, w[1:]):
                o.write(f'{n} {v:.6f}\n')
        say(f'{target} -> {fname}:')
        say(f'  {"intercept":<32}{w[0]:9.4f}')
        for n, v in zip(cols, w[1:]):
            say(f'  {n:<32}{v:9.4f}   (sd {f[n][train].std():.2f})')

    say('\n== Input definitions (for recomputing in Java) ==')
    for k, v in definitions().items():
        say(f'  {k}: {v}')
    say('  <CULTURE> in ' + ' '.join(CULTURES) + ' (Shadow cultures seen in the 2022 Movie block replays)')

    # what the recommended model says strength does with twilight, per seen deck
    w, cols, _ = results[('yStr', '+revealed')]
    wm, _, _ = results[('yMinions', '+revealed')]
    say('\n== Recommended model: predicted strength (minions) vs twilight, by the culture FP has seen ==')
    say('(site 5, 1 move, 5 FP companions, Shadow hand 8, no minions in play; 5 minions seen, all of one culture; '
        '"none seen" = oppSeenMinions 0)')
    say(f'{"seen":<10}{"per tw<=16":>11}' + ''.join(f'{"tw=" + str(t):>14}' for t in (6, 9, 12, 16, 20)))
    for cu in ['none seen'] + CULTURES:
        row = {n: 0.0 for n in cols}
        row.update(site=5, moves=1, fpCompanions=5, oppHand=8, oppSeenMinions=0 if cu == 'none seen' else 5)
        line = f'{cu:<10}'
        slope = w[1 + cols.index('twilight')] + (0 if cu == 'none seen' else
                                                 w[1 + cols.index('twilightCapped_x_seenShare_' + cu)])
        line += f'{slope:>11.2f}'
        for t in (6, 9, 12, 16, 20):
            row['twilight'] = t
            row['twilightOver16'] = max(0, t - KNOT)
            for c in CULTURES:
                row['twilightCapped_x_seenShare_' + c] = min(t, KNOT) * (1.0 if c == cu else 0.0)
            x = np.array([1.0] + [row[n] for n in cols])
            line += f'{x @ w:>8.1f} ({x @ wm:.1f})'
        say(line)

    # strength per twilight, overall and by Shadow culture
    say('\n== Strength fielded per twilight point ==')
    say('ratio = sum(strength played) / sum(twilight at start); slope = d strength / d twilight for twilight <= 16 '
        '(least squares with the other base inputs); spent = sum(strength played) / sum(twilight spent), '
        'spent = start pool - end pool, floored at 0 (cards that add twilight make this an underestimate of cost)')
    say(f'{"culture":<10}{"rows":>8}{"ratio":>8}{"slope<=16":>11}{"spent":>8}{"str@10":>8}{"str@16":>8}{"str@22":>8}')
    for cu in ['ALL'] + CULTURES:
        m = np.ones(len(T), bool) if cu == 'ALL' else d['shCulture'] == cu
        sp = np.maximum(0, T - d['yTwilightEnd'])
        lo = m & (T <= KNOT)
        Xb = design(f, BASE[:1] + BASE[2:])
        wb = ols(Xb[lo], d['yStrPlayed'][lo])
        at = lambda a, b: d['yStrPlayed'][m & (T >= a) & (T <= b)].mean()
        say(f'{cu:<10}{m.sum():>8}{d["yStrPlayed"][m].sum() / T[m].sum():>8.2f}{wb[1]:>11.2f}'
            f'{d["yStrPlayed"][m].sum() / max(1, sp[m].sum()):>8.2f}{at(9, 11):>8.1f}{at(15, 17):>8.1f}'
            f'{at(21, 23):>8.1f}')
    say('(shCulture = the Shadow player\'s most-played Shadow culture over the whole game: a label, not an input)')

    # how much of the FP deaths does predicted strength minus FP strength explain
    say('\n== FP companions killed this turn (from GPC Shadow to the next Regroup) vs strength margin ==')
    pred = results[('yStr', '+revealed')][2]
    k = d['yKilled']
    for label, x in (('predicted strength - FP printed companion strength', pred - d['fpCompStr']),
                     ('actual strength - FP printed companion strength (oracle)', d['yStr'] - d['fpCompStr']),
                     ('predicted strength alone', pred)):
        X = np.column_stack([np.ones(len(x)), x, np.maximum(0, x)])
        w = ols(X[train], k[train])
        say(f'  {label:<58} R2 July {r2(k[test], (X @ w)[test]):.3f}  (linear + hinge at 0)')
    X = design(f, BASE + REVEALED)
    X = np.column_stack([X, d['fpCompStr'], d['fpCompVit']])
    w = ols(X[train], k[train])
    say(f'  {"base + revealed + FP companion strength & vitality, direct":<58} R2 July {r2(k[test], (X @ w)[test]):.3f}')
    say(f'  killed rate by margin bin (actual strength - FP strength), July:')
    x = d['yStr'] - d['fpCompStr']
    for lo, hi in ((-99, -20), (-20, -10), (-10, 0), (0, 10), (10, 20), (20, 99)):
        m = test & (x >= lo) & (x < hi)
        if m.sum():
            say(f'    [{lo:>4},{hi:>3}) n={m.sum():>5}  killed/phase {k[m].mean():.3f}  wounds dealt '
                f'{d["yWoundsDealt"][m].mean():.2f}')
    rep.close()


if __name__ == '__main__':
    main()
