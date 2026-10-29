"""Mix the board network with the trees: agreement with people on held-out July, all players and by rating tier.

usage (from nn/):  uv run python ensemble.py --model ~/dev/gemp-replays/labels/nn/board_net2-d256.pt \
                       --boards ~/dev/gemp-replays/labels/nn/boards3-07.jsonl --ratings ~/dev/gemp-replays/labels/ratings-2022.tsv

The boards carry the trees' scores for every option (HumanReplay --boards with -Dboards.trees): each option's
probability from the bot's listwise trees, then pass's. Both models become log-probabilities over the decision's
options -- the network's softmax, the trees' softmax over their scores' logits -- and are mixed as (1 - w) * network +
w * trees. w is chosen on half of July's games (by the game's name) and reported on the other half, so the mix is
judged on decisions it was not tuned on. Models that read the game differently make different mistakes: the network
leads on Fellowship plays, the trees on Shadow plays.
"""
import argparse, csv, collections, hashlib, json, os

import numpy as np
import torch

import board_net as bn


def tiers(path):
    r = {x['player']: (float(x['rating']), int(x['games'])) for x in csv.DictReader(open(path), delimiter='\t')}
    reg = sorted(v[0] for v in r.values() if v[1] >= 20)
    q = [reg[int(len(reg) * x)] for x in (0.25, 0.5, 0.75)]

    def tier(p):
        rating, n = r.get(p, (1500, 0))
        if n < 20:
            return 'few'
        return 'bottom' if rating < q[0] else 'second' if rating < q[1] else 'third' if rating < q[2] else 'top'
    return tier


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--boards', required=True)
    ap.add_argument('--ratings', required=True)
    ap.add_argument('--data', default=os.path.expanduser('~/dev/gemp-replays/labels/nn'))
    args = ap.parse_args()
    ck = torch.load(args.model, weights_only=False)
    a = ck['args']
    rows, table, width = bn.load_cards(os.path.join(args.data, 'cards.tsv'))
    seen = set(ck['seen'])
    data, _ = bn.encode(args.boards, rows, seen, grow=False)
    meta = []  # game, player, tree scores, options, chosen, context -- in encode's order
    for line in open(args.boards):
        r = json.loads(line)
        if r['ctx'] not in bn.CONTEXTS or len(r['opts']) > bn.MAX_OPTS:
            continue
        meta.append((r.get('game', ''), r.get('player', ''), r.get('tree'), len(r['opts'])))
    assert len(meta) == len(data['y'])
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    model = bn.BoardNet(table, data['state'].shape[1], d=a['d'], layers=a['layers'],
                        id_drop=a.get('id_drop', 0.3), dropout=a.get('dropout', 0.1)).to(device)
    model.load_state_dict(ck['model'])
    model.eval()
    mean, std = ck['mean'].to(device), ck['std'].to(device)
    nn_lp = []
    with torch.no_grad():
        for b in bn.batches(data, np.arange(len(data['y'])), 512, device, mean, std):
            nn_lp.append(torch.log_softmax(model(b), -1).cpu().numpy())
    nn_lp = np.concatenate(nn_lp)  # [N, MAX_OPTS + 1], pass last
    tier = tiers(args.ratings)
    recs = []
    for i, (game, player, tree, k) in enumerate(meta):
        if tree is None or len(tree) != k + 1:
            continue
        p = np.clip(np.array(tree, dtype=np.float64), 1e-9, 1 - 1e-9)
        s = np.log(p / (1 - p))
        t = np.full(bn.MAX_OPTS + 1, -1e9)
        t[:k] = s[:k]
        t[bn.MAX_OPTS] = s[k]
        t = t - np.logaddexp.reduce(t)
        half = int(hashlib.md5(game.encode()).hexdigest(), 16) % 2
        recs.append((half, tier(player), data['cx'][i], nn_lp[i], t, data['y'][i], data['oe'][i]))
    print(f'{len(recs)} July decisions with both models\' scores')

    def acc(sel, w):
        hit = collections.Counter(); n = collections.Counter()
        for half, tr, cx, a_, b_, y, oe in sel:
            m = (1 - w) * a_ + w * b_
            ok = int(oe[int(np.argmax(m))] == oe[y])  # the same card doing the same thing counts
            for key in ('all', tr, bn.CONTEXTS[cx].split()[1]):
                hit[key] += ok; n[key] += 1
        return {k: hit[k] / n[k] for k in n}, n

    tune = [r for r in recs if r[0] == 0]
    test = [r for r in recs if r[0] == 1]
    grid = [x / 10 for x in range(11)]
    best = max(grid, key=lambda w: acc(tune, w)[0]['all'])
    print('tuning half, all players:', ' '.join(f'w={w:.1f} {acc(tune, w)[0]["all"]:.1%}' for w in grid))
    keys = ['all', 'Fellowship', 'Shadow', 'few', 'bottom', 'second', 'third', 'top']
    for name, w in (('network', 0.0), ('trees', 1.0), (f'mix w={best:.1f}', best)):
        r, n = acc(test, w)
        print(f'{name:12} ' + '  '.join(f'{k} {r[k]:.1%}' for k in keys if k in r))
    print('(test half sizes: ' + ', '.join(f'{k} {n}' for k, n in acc(test, 0)[1].items()) + ')')
    # and the weight that suits the top quarter best, on the tuning half
    top = [r for r in tune if r[1] == 'top']
    wt = max(grid, key=lambda w: acc(top, w)[0]['all'])
    print(f'best w for the top quarter (tuning half): {wt:.1f}; on the test half\'s top quarter: '
          f'{acc([r for r in test if r[1] == "top"], wt)[0]["all"]:.1%}')


if __name__ == '__main__':
    main()
