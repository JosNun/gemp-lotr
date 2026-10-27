#!/usr/bin/env python3
"""Play styles in people's games: cluster the per-(game, player) rows play_styles_extract.py writes, name the
clusters by their metric profile, and write per style the (game, player) list whose decisions would train that
style's opponent bot.

usage: play_styles_cluster.py [player_games.tsv] [--k 5] [--out /tmp/styles] [--min-turns 3] [--cap N] [--seed 1]

Rows: games where the player had >= --min-turns FP turns and as many Shadow turns (shorter games are mostly setup,
and their per-turn rates are noise). July (month 07) is held out: it is clustered with the rest, so it gets labels,
but its rows go to <out>/heldout-07/<style>.txt instead of the training lists.

Method: the BEHAVIOUR metrics below (choices a player makes; outcomes such as kills, wounds and companions lost are
reported, not clustered on) are standardized (z, clipped at +-4), reduced by PCA (numpy SVD, PCS components),
and clustered with k-means (k-means++, several inits). The k-selection table (k=3..6) gives the seed-to-seed ARI,
the split-half ARI (centroids fit on one random half of the rows vs the other, both applied to all rows) and, per
cluster, how persistent a player's share of games in that cluster is (correlation of the share between two random
halves of each player's games, players with >= 10 games). Deck-vs-player share: for each metric and for cluster
membership, adjusted R^2 of deck culture alone, player alone, and deck + player (additive fixed effects); the
"player beyond deck" column is the gain of the last over deck alone.

Outputs under --out: metric_variance.tsv, k_selection.tsv, styles_k<k>.tsv (one row per (game, player) with its
style), profile_k<k>.txt (the report), players_k<k>.tsv (per-player metric means and style shares),
<style>.txt (non-July "game_id<TAB>player"), heldout-07/<style>.txt.
"""
import argparse, csv, os, sys
from collections import Counter, defaultdict
import numpy as np

FP = ['fp_moveAgainRate', 'fp_movesPerTurn', 'fp_compsMean', 'fp_startComps', 'fp_fellPlays', 'fp_poss', 'fp_cond',
      'fp_events', 'fp_skirmishEvents', 'fp_allies', 'fp_fellTwAdded', 'fp_tpAtMove', 'fp_healed',
      'fp_burdensPerTurn', 'fp_bid', 'fp_archery']
SH = ['sh_minionsPerPhase', 'sh_minStr', 'sh_minCost', 'sh_twLeft', 'sh_spentFrac', 'sh_cond', 'sh_poss',
      'sh_events', 'sh_minionsAtAssign', 'sh_archery', 'sh_discards']
FEATS = FP + SH
# reported per style but not clustered on: outcomes and context
EXTRA = ['fp_siteT4', 'fp_finalSite', 'fp_compsLost', 'fp_ringOn', 'fp_burdensRemoved', 'fp_mulligan', 'sh_twAvail',
         'sh_kills', 'sh_woundsDealt', 'sh_burdensAdded', 'sh_archerMinions', 'sh_plays']
PCS = 8
# a style is named after the metric that sets it apart most; greedy, highest z first, each name used once
# (lean, weighted down, only names a cluster nothing else describes)
SIGNATURES = [('racer', 'fp_moveAgainRate', 1), ('builder', 'fp_fellTwAdded', 1),
              ('wide-fellowship', 'fp_startComps', 1), ('big-minions', 'sh_minStr', 1),
              ('swarm-grinder', 'sh_minionsPerPhase', 1), ('passive-shadow', 'sh_spentFrac', -1),
              ('lean', 'fp_compsMean', -0.5), ('archer', 'sh_archery', 1)]


def kmeans(Z, k, seed=0, n_init=5, iters=200):
    rng = np.random.default_rng(seed)
    best = None
    for _ in range(n_init):
        C = [Z[rng.integers(len(Z))]]
        for _ in range(k - 1):
            d = np.min(((Z[:, None, :] - np.array(C)[None]) ** 2).sum(-1), 1)
            C.append(Z[rng.choice(len(Z), p=d / d.sum())])
        C = np.array(C)
        for _ in range(iters):
            d = ((Z[:, None, :] - C[None]) ** 2).sum(-1)
            a = d.argmin(1)
            Cn = np.array([Z[a == j].mean(0) if (a == j).any() else C[j] for j in range(k)])
            if np.allclose(Cn, C):
                break
            C = Cn
        inertia = d.min(1).sum()
        if best is None or inertia < best[0]:
            best = (inertia, C, a)
    return best[1], best[2]


def assign(Z, C):
    return ((Z[:, None, :] - C[None]) ** 2).sum(-1).argmin(1)


def ari(a, b):
    M = np.zeros((a.max() + 1, b.max() + 1))
    np.add.at(M, (a, b), 1)
    c2 = lambda x: (x * (x - 1) / 2).sum()
    s, sa, sb, n = c2(M), c2(M.sum(1)), c2(M.sum(0)), len(a) * (len(a) - 1) / 2
    e = sa * sb / n
    return (s - e) / ((sa + sb) / 2 - e)


def codes(keys):
    u = {k: i for i, k in enumerate(sorted(set(keys)))}
    return np.array([u[k] for k in keys]), len(u)


def gfit(y, g, k):
    s, n = np.bincount(g, y, k), np.bincount(g, None, k)
    return (s / np.maximum(n, 1))[g]


def adjr2(y, yh, df):
    n = len(y)
    ss = ((y - y.mean()) ** 2).sum()
    if ss == 0:
        return float('nan')
    return 1 - (1 - (1 - ((y - yh) ** 2).sum() / ss)) * (n - 1) / (n - df - 1)


def deck_player(y, d, kd, p, kp):
    """adjusted R^2: deck alone, player alone, deck + player (alternating projections)"""
    rd, rp = adjr2(y, gfit(y, d, kd), kd - 1), adjr2(y, gfit(y, p, kp), kp - 1)
    mu, A, B = y.mean(), np.zeros_like(y), np.zeros_like(y)
    for _ in range(40):
        A = gfit(y - mu - B, d, kd)
        B = gfit(y - mu - A, p, kp)
    return rd, rp, adjr2(y, mu + A + B, kd + kp - 2)


def name_clusters(Zc):
    """Zc: k x feats centroid z-profiles (original metric space)."""
    k = len(Zc)
    cands = []
    for nm, f, sgn in SIGNATURES:
        j = FEATS.index(f)
        for c in range(k):
            cands.append((sgn * Zc[c, j], c, nm))
    names, used = {}, set()
    for z, c, nm in sorted(cands, reverse=True):
        if c in names or nm in used or z < 0.3:
            continue
        names[c] = nm; used.add(nm)
    return [names.get(c, f'style{c}') for c in range(k)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('tsv', nargs='?', default='/tmp/styles/player_games.tsv')
    ap.add_argument('--k', type=int, default=5)
    ap.add_argument('--out', default='/tmp/styles')
    ap.add_argument('--min-turns', type=int, default=3)
    ap.add_argument('--cap', type=int, default=0, help='at most N games per player per style in the lists (0: all)')
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--skip-ksel', action='store_true')
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    allrows = list(csv.DictReader(open(args.tsv), delimiter='\t'))
    R = [r for r in allrows if int(r['myTurns']) >= args.min_turns and int(r['oppTurns']) >= args.min_turns]
    print(f'{len(allrows)} player-games, {len(R)} with >= {args.min_turns} turns on each side')

    X = np.array([[float(r[c]) for c in FEATS] for r in R])
    X = np.where(np.isnan(X), np.nanmean(X, 0), X)
    mu, sd = X.mean(0), X.std(0)
    Z = np.clip((X - mu) / sd, -4, 4)
    U, s, Vt = np.linalg.svd(Z - Z.mean(0), full_matrices=False)
    ev = s ** 2 / (s ** 2).sum()
    P = (Z - Z.mean(0)) @ Vt[:PCS].T
    print('PCA explained', np.round(ev[:PCS], 3), 'cum', round(float(ev[:PCS].sum()), 3))

    players = np.array([r['player'] for r in R])
    pcount = Counter(players)
    pl, kp = codes(list(players))
    dk, kd = codes([r['deckKey'] for r in R])
    fpc, kf = codes([r['fpCulture'] for r in R])
    shc, ks = codes([r['shCulture'] + '+' + r['shCulture2'] for r in R])

    # ---- metric variance: deck culture vs player
    with open(f'{args.out}/metric_variance.tsv', 'w') as f:
        f.write('metric\tsideCulture\tdeckKey\tplayer\tdeckKey+player\tplayerBeyondDeck\n')
        for c in FEATS + EXTRA:
            y = np.array([float(r[c]) for r in R])
            y = np.where(np.isnan(y), np.nanmean(y), y)
            g, kg = (fpc, kf) if c.startswith('fp') else (shc, ks)
            rc = adjr2(y, gfit(y, g, kg), kg - 1)
            rd, rp, rb = deck_player(y, dk, kd, pl, kp)
            f.write(f'{c}\t{rc:.3f}\t{rd:.3f}\t{rp:.3f}\t{rb:.3f}\t{rb - rd:.3f}\n')

    # ---- k selection
    big = [p for p, n in pcount.items() if n >= 10]
    rng = np.random.default_rng(args.seed)
    halves = {p: rng.permutation(np.where(players == p)[0]) for p in big}
    if not args.skip_ksel:
        idx = rng.permutation(len(P)); A, B = idx[:len(P) // 2], idx[len(P) // 2:]
        with open(f'{args.out}/k_selection.tsv', 'w') as f:
            f.write('k\tseedARI\tsplitHalfARI\tsizes\tplayerPersistence\n')
            for k in range(3, 7):
                C1, a1 = kmeans(P, k, seed=args.seed, n_init=4)
                C2, a2 = kmeans(P, k, seed=args.seed + 100, n_init=4)
                CA, _ = kmeans(P[A], k, seed=args.seed + 1, n_init=4)
                CB, _ = kmeans(P[B], k, seed=args.seed + 2, n_init=4)
                pers = []
                for j in range(k):
                    sa = [np.mean(a1[h[:len(h) // 2]] == j) for h in halves.values()]
                    sb = [np.mean(a1[h[len(h) // 2:]] == j) for h in halves.values()]
                    pers.append(round(float(np.corrcoef(sa, sb)[0, 1]), 2))
                line = (f'{k}\t{ari(a1, a2):.3f}\t{ari(assign(P, CA), assign(P, CB)):.3f}\t'
                        f'{np.bincount(a1).tolist()}\t{pers}')
                print('k-selection', line.replace('\t', '  '))
                f.write(line + '\n')

    # ---- the chosen k
    k = args.k
    C, a = kmeans(P, k, seed=args.seed, n_init=8)
    Zc = np.array([Z[a == j].mean(0) for j in range(k)])
    names = name_clusters(Zc)
    order = np.argsort([-np.sum(a == j) for j in range(k)])
    won = np.array([int(r['won']) for r in R])
    held = np.array([r['heldout'] == '1' for r in R])
    dec = np.array([int(r['decisions']) for r in R])
    Xe = np.array([[float(r[c]) for c in EXTRA] for r in R])
    lines = [f'k={k}, {len(R)} player-games ({(~held).sum()} training, {held.sum()} July held out), '
             f'{len(big)} players with >= 10 games', '']
    over = {c: np.nanmean(X[:, i]) for i, c in enumerate(FEATS)}
    over.update({c: np.nanmean(Xe[:, i]) for i, c in enumerate(EXTRA)})
    with open(f'{args.out}/styles_k{k}.tsv', 'w') as f:
        f.write('game\tplayer\tmonth\tstyle\n')
        for r, j in zip(R, a):
            f.write(f"{r['game']}\t{r['player']}\t{r['month']}\t{names[j]}\n")
    os.makedirs(f'{args.out}/heldout-07', exist_ok=True)
    for j in order:
        m = a == j
        nm = names[j]
        y = m.astype(float)
        rd, rp, rb = deck_player(y, dk, kd, pl, kp)
        sa = [np.mean(a[h[:len(h) // 2]] == j) for h in halves.values()]
        sb = [np.mean(a[h[len(h) // 2:]] == j) for h in halves.values()]
        pers = float(np.corrcoef(sa, sb)[0, 1])
        # lists
        rows_tr = [r for r, mm, hh in zip(R, m, held) if mm and not hh]
        if args.cap:
            per = Counter(); kept = []
            for r in rows_tr:
                if per[r['player']] < args.cap:
                    per[r['player']] += 1; kept.append(r)
            rows_tr = kept
        with open(f'{args.out}/{nm}.txt', 'w') as f:
            for r in rows_tr:
                f.write(f"{r['game']}\t{r['player']}\n")
        with open(f'{args.out}/heldout-07/{nm}.txt', 'w') as f:
            for r, mm, hh in zip(R, m, held):
                if mm and hh:
                    f.write(f"{r['game']}\t{r['player']}\n")
        tr_dec = sum(int(r['decisions']) for r in rows_tr)
        # outcome mix
        wins = Counter(); losses = Counter()
        for r, mm in zip(R, m):
            if not mm:
                continue
            why = r['endReason'] if r['endReason'] not in ('concession', 'timeout') else \
                f"{r['endReason']}:{r['concedeKind']}"
            (wins if r['won'] == '1' else losses)[why] += 1
        nw, nl = sum(wins.values()), sum(losses.values())
        fpcult = Counter(r['fpCulture'] for r, mm in zip(R, m) if mm)
        shcult = Counter(r['shCulture'] for r, mm in zip(R, m) if mm)
        dkeys = Counter(r['deckKey'] for r, mm in zip(R, m) if mm)
        pls = Counter(r['player'] for r, mm in zip(R, m) if mm)
        n = m.sum()
        lines.append(f'== {nm}: {n} player-games ({len(rows_tr)} in the training list, {(m & held).sum()} July), '
                     f'{tr_dec:,} training decisions, {len(pls)} players, win rate {won[m].mean():.3f}')
        lines.append(f'   membership adj R^2: deck {rd:.3f}, player {rp:.3f}, player beyond deck {rb - rd:.3f}; '
                     f'player share split-half r = {pers:.2f}')
        zs = Zc[j]
        top = np.argsort(-np.abs(zs))[:10]
        lines.append('   defining metrics (style mean vs overall, z): ' + '; '.join(
            f'{FEATS[i]} {X[m, i].mean():.2f} vs {over[FEATS[i]]:.2f} ({zs[i]:+.2f})' for i in top))
        lines.append('   outcomes/context: ' + '; '.join(
            f'{c} {np.nanmean(Xe[m, i]):.2f} vs {over[c]:.2f}' for i, c in enumerate(EXTRA)
            if c in ('fp_siteT4', 'fp_compsLost', 'sh_kills', 'sh_woundsDealt', 'sh_burdensAdded', 'sh_twAvail')))
        lines.append('   wins by: ' + ', '.join(f'{k_} {v / nw:.0%}' for k_, v in wins.most_common(6)))
        lines.append('   losses by: ' + ', '.join(f'{k_} {v / nl:.0%}' for k_, v in losses.most_common(6)))
        lines.append('   FP cultures: ' + ', '.join(f'{k_} {v / n:.0%} ({v / sum(r["fpCulture"] == k_ for r in R):.0%} of all)'
                                               for k_, v in fpcult.most_common(4)))
        lines.append('   Shadow cultures: ' + ', '.join(f'{k_} {v / n:.0%} ({v / sum(r["shCulture"] == k_ for r in R):.0%} of all)'
                                                   for k_, v in shcult.most_common(4)))
        lines.append('   decks: ' + ', '.join(f'{k_} {v}' for k_, v in dkeys.most_common(5)))
        lines.append('   top players: ' + ', '.join(f'{k_} {v} ({v / pcount[k_]:.0%} of theirs)'
                                              for k_, v in pls.most_common(5)) +
                     f'; top-3 share {sum(v for _, v in pls.most_common(3)) / n:.0%}')
        lines.append('')
    # per player: metric means and style shares
    with open(f'{args.out}/players_k{k}.tsv', 'w') as f:
        f.write('player\tgames\tdominantStyle\t' + '\t'.join(names[j] for j in order) + '\t' + '\t'.join(FEATS) + '\n')
        for p, cnt in pcount.most_common():
            m = players == p
            sh = [np.mean(a[m] == j) for j in order]
            f.write(f'{p}\t{cnt}\t{names[order[int(np.argmax(sh))]]}\t' + '\t'.join(f'{v:.2f}' for v in sh) + '\t' +
                    '\t'.join(f'{v:.3f}' for v in X[m].mean(0)) + '\n')
    txt = '\n'.join(lines)
    open(f'{args.out}/profile_k{k}.txt', 'w').write(txt + '\n')
    print(txt)


if __name__ == '__main__':
    main()
