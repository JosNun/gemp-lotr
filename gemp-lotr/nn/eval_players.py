"""July agreement of networks with all players and with a set of players only (2026-11-07: did training on the
strongest players' decisions make the network agree more with them?).

usage (from nn/):  uv run python eval_players.py <players.txt> <a.pt> ...
"""
import os, sys, zlib

import numpy as np
import torch

import board_net2 as bn
from board_net import load_cards


def main():
    d = os.path.expanduser('~/dev/gemp-replays/labels/nn')
    rows, table, _ = load_cards(os.path.join(d, 'cards.tsv'))
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    top = {zlib.crc32(p.encode()) for p in open(sys.argv[1]).read().split()}
    cks = [torch.load(p, weights_only=False, map_location='cpu') for p in sys.argv[2:]]
    test, _ = bn.encode(os.path.join(d, 'boards4-07.jsonl'), rows, set(cks[0]['seen']), False,
                        bn.load_routes(os.path.join(d, 'routes-2022.jsonl')))
    sel = np.array([h in top for h in test['ph']])
    subsets = {'all': test, 'top players': {k: v[sel] for k, v in test.items()},
               'the rest': {k: v[~sel] for k, v in test.items()}}
    print(f'July: {len(sel)} decisions, {sel.sum()} by the listed players', flush=True)
    for path, ck in zip(sys.argv[2:], cks):
        a = ck['args']
        m = bn.BoardNet2(table, test['state'].shape[1], d=a['d'], layers=a['layers']).to(device)
        m.load_state_dict(ck['model'], strict=False)
        mean, std = ck['mean'].to(device), ck['std'].to(device)
        for name, data in subsets.items():
            hit, n, _, _ = bn.evaluate(m, data, device, mean, std)
            acts = [c for c in bn.CONTEXTS[:10] if n[c]]
            print(f'{os.path.basename(path)} {name}: card plays {100 * sum(hit[c] for c in acts) / max(1, sum(n[c] for c in acts)):.2f}%, '
                  f'move {100 * hit["move"] / max(1, n["move"]):.1f}%, assign fp {100 * hit["assign fp"] / max(1, n["assign fp"]):.1f}%', flush=True)


if __name__ == '__main__':
    main()
