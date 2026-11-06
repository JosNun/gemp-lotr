"""July agreement of several board_net2 networks averaged (approach 7, 2026-11-06): each network alone, then their
mean log-probabilities (options and assignments) and mean outcome probabilities, through board_net2.evaluate.

usage (from nn/):  uv run python eval_ensemble.py <a.pt> <b.pt> [...]

The networks must share the state normalisation (trained from the same --init), as sp-m1's siblings do.
"""
import os, sys

import torch
import torch.nn.functional as F

import board_net2 as bn
from board_net import load_cards


class Mean(torch.nn.Module):
    def __init__(self, models):
        super().__init__()
        self.models = torch.nn.ModuleList(models)

    def forward(self, b):
        outs = [m(b) for m in self.models]
        s = torch.stack([F.log_softmax(o[0], -1) for o in outs]).mean(0)
        a = torch.stack([F.log_softmax(o[1], -1) for o in outs]).mean(0)
        w = torch.log(torch.stack([F.softmax(o[2], -1) for o in outs]).mean(0))
        return s, a, w


def main():
    data_dir = os.path.expanduser('~/dev/gemp-replays/labels/nn')
    rows, table, _ = load_cards(os.path.join(data_dir, 'cards.tsv'))
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    cks = [torch.load(p, weights_only=False, map_location='cpu') for p in sys.argv[1:]]
    mean, std = cks[0]['mean'].to(device), cks[0]['std'].to(device)
    seen = set(cks[0]['seen'])
    test, _ = bn.encode(os.path.join(data_dir, 'boards4-07.jsonl'), rows, seen, False,
                        bn.load_routes(os.path.join(data_dir, 'routes-2022.jsonl')))
    models = []
    for p, ck in zip(sys.argv[1:], cks):
        a = ck['args']
        m = bn.BoardNet2(table, test['state'].shape[1], d=a['d'], layers=a['layers'], probe=a.get('probe', False))
        m.load_state_dict(ck['model'], strict=False)
        models.append(m.to(device))
        hit, n, auc, route = bn.evaluate(m, test, device, mean, std)
        print(f'{os.path.basename(p)}: {bn.report(hit, n)}; win AUC {auc:.3f}', flush=True)
    hit, n, auc, route = bn.evaluate(Mean(models).to(device), test, device, mean, std)
    print(f'mean of {len(models)}: {bn.report(hit, n)}; win AUC {auc:.3f}')


if __name__ == '__main__':
    main()
