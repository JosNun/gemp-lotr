"""How well judges tell who wins (2026-11-06): win AUC on July's people (boards4-07) and on bot positions with their
games' results (BotVsBot -Dbvb.boards / -Dbvb.outcomes) -- search judges positions bots played out, not people's.

usage (from nn/):  uv run python eval_judge.py --bots <boards.jsonl> --outcomes <outcomes.jsonl> <judge.pt> ...
"""
import argparse, os

import numpy as np
import torch

import board_net2 as bn
from board_net import load_cards


def win_auc(model, data, device, mean, std):
    model.eval()
    p, w = [], []
    with torch.no_grad():
        for b in bn.batches(data, np.arange(len(data['cx'])), 512, device, mean, std):
            o = torch.softmax(model(b)[2], -1)
            keep = b['win'] >= 0
            p.append(o[:, :3].sum(-1)[keep].cpu().numpy()); w.append(b['win'][keep].cpu().numpy())
    return bn.auc(np.concatenate(p), np.concatenate(w)), sum(len(x) for x in w)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--bots', required=True)
    ap.add_argument('--outcomes', required=True)
    ap.add_argument('judges', nargs='+')
    args = ap.parse_args()
    d = os.path.expanduser('~/dev/gemp-replays/labels/nn')
    rows, table, _ = load_cards(os.path.join(d, 'cards.tsv'))
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    for path in args.judges:
        ck = torch.load(path, weights_only=False, map_location='cpu')
        seen = set(ck['seen'])
        july, _ = bn.encode(os.path.join(d, 'boards4-07.jsonl'), rows, seen, False,
                            bn.load_routes(os.path.join(d, 'routes-2022.jsonl')))
        bots, _ = bn.encode(args.bots, rows, seen, False, bn.load_routes(args.outcomes))
        a = ck['args']
        m = bn.BoardNet2(table, july['state'].shape[1], d=a['d'], layers=a['layers'], probe=a.get('probe', False))
        m.load_state_dict(ck['model'], strict=False)
        m = m.to(device)
        mean, std = ck['mean'].to(device), ck['std'].to(device)
        aj, nj = win_auc(m, july, device, mean, std)
        ab, nb = win_auc(m, bots, device, mean, std)
        print(f'{os.path.basename(path)} (d {a["d"]}, {a["layers"]} layers): July people AUC {aj:.4f} ({nj}), '
              f'bot positions AUC {ab:.4f} ({nb})', flush=True)


if __name__ == '__main__':
    main()
