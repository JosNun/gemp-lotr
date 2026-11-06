"""Where does play-time search pay? (approach 6, 2026-11-06) Reads Searching's records (-Dsearch.boards, with "own" --
the bot's answer without search -- and "netTop", the network's top probability) and asks how much of search's gain a
cheaper gate keeps.

usage: python3 -I scripts/search_gate.py <search-boards.jsonl>

A search's gain is the judged value of its pick minus that of the bot's own answer (0 where they agree). Ranked by
a gate's score, the searches kept at 10/25/50% and the share of the summed gain they hold: (a) the network's own
uncertainty alone (the gate in use, made stricter), (b) a small gradient-boosted model of the gain from the context,
the network's certainty, the number of options and the state -- fit on half the games, scored on the other half.
"""
import json, os, re, sys

import numpy as np

JAVA = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'gemp-lotr-server', 'src', 'main', 'java', 'com',
                    'gempukku', 'lotro', 'bots', 'rl', 'fotrstarters', 'GeneralStateFeatures.java')
STATE = ['turn', 'site', 'twilight', 'burdens', 'threats', 'fpCompanions', 'fpCompanionVitalityLeft', 'fpMinVitalityLeft',
         'minions', 'minionStrength', 'fpCompanionStrength', 'myHand', 'oppHand', 'inSkirmish', 'mySkirmishMargin',
         'skirmishFpVitalityLeft', 'rbVitalityLeft', 'iAmFp']


def names():
    src = open(JAVA).read()
    block = re.search(r'for \(String name : List\.of\((.*?)\)\)', src, re.S).group(1)
    return re.findall(r'"([A-Za-z0-9_]+)"', re.sub(r'//[^\n]*', '', block))


def main():
    nm = names()
    idx = [nm.index(k) for k in STATE]
    X, gain, game = [], [], []
    for line in open(sys.argv[1]):
        r = json.loads(line)
        vals = {int(i): v for i, v in r['searched']}
        own, pick = r.get('own'), r.get('chosen')
        if own is None or own not in vals or pick not in vals:
            continue
        g = vals[pick] - vals[own]
        ctx = r.get('ctx', '')
        st = r['state']
        X.append([1.0 if 'Choose action to play' in ctx else 0.0, r.get('netTop', 0.5), len(r.get('opts', [])) + 1,
                  len(vals), max(vals.values()) - min(vals.values())] + [st[i] if i < len(st) else 0 for i in idx])
        gain.append(g)
        game.append(r.get('game', ''))
    X, gain = np.array(X), np.array(gain)
    print(f'{len(gain)} searches; search changed the answer on {np.mean(gain != 0):.1%}; mean gain {gain.mean():.4f}; '
          f'the top 10% of gains hold {np.sort(gain)[::-1][:len(gain) // 10].sum() / max(1e-9, gain.sum()):.0%} of it')
    games = sorted(set(game))
    rng = np.random.default_rng(1)
    test_games = set(rng.choice(games, len(games) // 2, replace=False))
    te = np.array([g in test_games for g in game])

    def kept(score, mask):
        g = gain[mask]
        order = np.argsort(-score[mask])
        out = []
        for share in (0.1, 0.25, 0.5):
            k = int(len(g) * share)
            out.append(g[order[:k]].sum() / max(1e-9, g.sum()))
        return out

    # (a) the network's uncertainty: least sure first. (a2) the same within skirmishes only, Shadow only
    print('share of the gain kept, searching only the top 10% / 25% / 50% by each gate (held-out games):')
    print('  network least sure        ' + ' / '.join(f'{x:.0%}' for x in kept(-X[:, 1], te)))
    print('  value spread (oracle-ish) ' + ' / '.join(f'{x:.0%}' for x in kept(X[:, 4], te)) + '   (needs the search itself)')
    try:
        from sklearn.ensemble import HistGradientBoostingRegressor
    except ImportError:
        print('  (no scikit-learn: the learned gate is skipped)')
        return
    feats = np.delete(X, 4, axis=1)  # the spread is known only after searching
    m = HistGradientBoostingRegressor(max_iter=200, learning_rate=0.05, max_leaf_nodes=15).fit(feats[~te], gain[~te])
    print('  learned gate              ' + ' / '.join(f'{x:.0%}' for x in kept(m.predict(feats), te)))
    print('  skirmishes only (random order within): searches '
          f'{X[te, 0].mean():.0%}, gain {gain[te][X[te, 0] == 1].sum() / max(1e-9, gain[te].sum()):.0%}')


if __name__ == '__main__':
    main()
