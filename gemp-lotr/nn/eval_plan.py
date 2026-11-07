"""July agreement of a --plan network told nothing, the true ending, or its own outcome head's likeliest win
(2026-11-06). usage (from nn/): uv run python eval_plan.py <plan.pt> [<baseline.pt>]"""
import os, sys

import torch

import board_net2 as bn
from board_net import load_cards


def main():
    d = os.path.expanduser('~/dev/gemp-replays/labels/nn')
    rows, table, _ = load_cards(os.path.join(d, 'cards.tsv'))
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    cks = [torch.load(p, weights_only=False, map_location='cpu') for p in sys.argv[1:]]
    test, _ = bn.encode(os.path.join(d, 'boards4-07.jsonl'), rows, set(cks[0]['seen']), False,
                        bn.load_routes(os.path.join(d, 'routes-2022.jsonl')))
    for p, ck in zip(sys.argv[1:], cks):
        a = ck['args']
        m = bn.BoardNet2(table, test['state'].shape[1], d=a['d'], layers=a['layers'], plan=a.get('plan', False))
        m.load_state_dict(ck['model'], strict=False)
        m = m.to(device)
        mean, std = ck['mean'].to(device), ck['std'].to(device)
        for mode in (('none', 'true', 'judge') if hasattr(m, 'plan') else ('-',)):
            if hasattr(m, 'plan'):
                m.plan_mode = mode
            hit, n, auc, route = bn.evaluate(m, test, device, mean, std)
            print(f'{os.path.basename(p)} plan {mode}: {bn.report(hit, n)}; win AUC {auc:.3f}', flush=True)


if __name__ == '__main__':
    main()
