"""The combined network's assignments as the bot would make them: the best legal one, not each minion's own pick.

usage (from nn/):  uv run python eval_assign.py --model ~/dev/gemp-replays/labels/nn/board_net2-d256.pt

board_net2's per-minion argmax can send two minions to one character, which the Free Peoples player may not do (a
defender excepted), and then counts as a miss. Here the Free Peoples assignment is decoded as a matching: each minion
to at most one character, each character taking at most one minion (two with defender +1), or the minion left
unassigned, maximising the summed log-probabilities. The Shadow player's assignment has no such limit, so it stays
per minion. Agreement is the whole assignment, card for card -- as HumanReplay counts it for the trees.
"""
import argparse, os

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

import board_net2 as bn
from board_net import load_cards

DEFENDER = 8  # BoardDump.KEYWORDS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--data', default=os.path.expanduser('~/dev/gemp-replays/labels/nn'))
    ap.add_argument('--test', default='07')
    args = ap.parse_args()
    ck = torch.load(args.model, weights_only=False, map_location='cpu')
    a = ck['args']
    rows, table, _ = load_cards(os.path.join(args.data, 'cards.tsv'))
    seen = set(ck['seen'])
    data, _ = bn.encode(os.path.join(args.data, f'{a["prefix"]}-{args.test}.jsonl'), rows, seen, False, {},
                        kinds={'assign'})
    model = bn.BoardNet2(table, data['state'].shape[1], d=a['d'], layers=a['layers'])
    model.load_state_dict(ck['model'])
    model.eval()
    hit = {'assign fp': [0, 0, 0], 'assign shadow': [0, 0, 0]}  # legal decode, per-minion argmax, asked
    with torch.no_grad():
        for b in bn.batches(data, np.arange(len(data['cx'])), 256, 'cpu', ck['mean'], ck['std']):
            _, s, _ = model(b)
            lp = torch.log_softmax(s, -1).numpy()  # [B, M, C + 1], the last "unassigned"
            for i in range(len(lp)):
                ay = b['ay'][i].numpy().astype(int)
                m = int((ay != -100).sum())
                c = int((b['af'][i].numpy() >= 0).sum())
                truth = ay[:m]
                name = bn.CONTEXTS[int(b['cx'][i])]
                greedy = lp[i, :m, :].argmax(-1)
                greedy = np.where(greedy == bn.MAX_FPC, bn.MAX_FPC, greedy)
                if name == 'assign fp':
                    # columns: each character's slots (two for a defender), then one "unassigned" per minion
                    cols, owner = [], []
                    for j in range(c):
                        fp_item = int(b['af'][i][j])
                        slots = 1 + (int(b['ck'][i][fp_item][DEFENDER]) if fp_item >= 0 else 0)
                        for _ in range(slots):
                            cols.append(lp[i, :m, j]); owner.append(j)
                    for k in range(m):
                        col = np.full(m, -1e9); col[k] = lp[i, k, bn.MAX_FPC]
                        cols.append(col); owner.append(bn.MAX_FPC)
                    cost = -np.stack(cols, 1)
                    r, cc = linear_sum_assignment(cost)
                    pick = np.array([owner[cc[list(r).index(k)]] for k in range(m)])
                else:
                    pick = greedy
                h = hit[name]
                h[0] += int((pick == truth).all())
                h[1] += int((greedy == truth).all())
                h[2] += 1
    for name, (legal, greedy, n) in hit.items():
        if n:
            print(f'{name}: {n} July decisions; whole assignment as the human made it -- legal decode {legal / n:.1%}, '
                  f'each minion its own pick {greedy / n:.1%}')


if __name__ == '__main__':
    main()
