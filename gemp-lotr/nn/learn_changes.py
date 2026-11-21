"""Can the card-reading network learn where the search departs from the trees? (2026-11-21)

Searched boards (BotVsBot -Dsearch.boards: a card action the search played out, with each candidate's value, the
search's answer `chosen` and the trees' own `own`) are split by game, four fifths to fit from. The network starts from
a people-trained checkpoint and is fitted to the search's answer among the candidates it tried. Held out, by whether
the search kept the trees' answer or changed it: how often the network's first choice among the candidates is the
search's -- before the fit and after each pass -- beside chance (one in the number of candidates).

The trees' own refits name the search's changes at chance on wound and heal picks (BOT-NOTES, 2026-11-20). If the
network does no better here, reading the cards is not what was missing.

usage (from nn/):  uv run python learn_changes.py --boards <boards.jsonl> --init <checkpoint.pt> [--epochs 6]
"""
import argparse, json, os, zlib

import numpy as np
import torch
import torch.nn.functional as F

import board_net2 as B
from board_net import load_cards


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--boards', required=True)
    ap.add_argument('--init', required=True)
    ap.add_argument('--data', default=os.path.expanduser('~/dev/gemp-replays/labels/nn'))
    ap.add_argument('--epochs', type=int, default=6)
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--batch', type=int, default=256)
    args = ap.parse_args()
    torch.manual_seed(1); np.random.seed(1)
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    rows, table, width = load_cards(os.path.join(args.data, 'cards.tsv'))
    init = torch.load(args.init, weights_only=False, map_location='cpu')
    seen = set(init['seen'])
    # the lines encode() keeps, with the trees' own answer beside each (encode does not read it)
    lines, own = [], []
    for line in open(args.boards):
        r = json.loads(line)
        if r.get('kind', 'action') != 'action' or len(r.get('opts', [])) > B.MAX_OPTS or 'own' not in r:
            continue
        if r['own'] is None or len(r.get('searched', [])) < 2:
            continue
        lines.append(line)
        own.append(r['own'] if r['own'] >= 0 else B.MAX_OPTS)
    data, games = B.encode(lines, rows, seen, False, {})
    assert len(data['cx']) == len(own), (len(data['cx']), len(own))
    own = np.array(own)
    searched = ~np.isnan(data['sv'])
    y = data['y']                                   # the search's answer (pass: MAX_OPTS)
    ok = searched[np.arange(len(y)), y] & searched[np.arange(len(y)), own]
    # by game; boards written before the game was named go by runs of fifty lines
    held = np.array([zlib.crc32((g or f'block{i // 50}').encode()) % 5 == 0 for i, g in enumerate(games)])
    changed = y != own
    print(f'{len(y)} searched card actions, {ok.sum()} with both answers among the candidates; '
          f'held out {int((held & ok).sum())}: kept {int((held & ok & ~changed).sum())}, changed {int((held & ok & changed).sum())}; '
          f'candidates a decision {searched[ok].sum(1).mean():.2f}', flush=True)
    data['lp0'] = np.zeros((len(y), B.MAX_OPTS + 1), np.float32); data['tw'] = np.ones(len(y), np.float32)
    mean, std = init['mean'].to(device), init['std'].to(device)
    a = init['args']
    model = B.BoardNet2(table, data['state'].shape[1], d=a['d'], layers=a['layers'], id_drop=a.get('id_drop', 0.3),
                        dropout=a.get('dropout', 0.1), probe=a.get('probe', False), plan=a.get('plan', False)).to(device)
    model.load_state_dict(init['model'], strict=False)
    if hasattr(model, 'plan'):
        model.plan_mode = 'none'
    train_idx = np.where(~held & ok)[0]; test_idx = np.where(held & ok)[0]

    def check(tag):
        model.eval()
        first = []
        with torch.no_grad():
            for b in B.batches(data, test_idx, 512, device, mean, std):
                s = model(b)[0]
                s = s.masked_fill(torch.isnan(b['sv']), -1e9)   # among the candidates the search tried
                first.append(s.argmax(-1).cpu().numpy())
        model.train()
        first = np.concatenate(first)
        hit = first == y[test_idx]; ch = changed[test_idx]
        as_own = first == own[test_idx]
        chance = (1 / searched[test_idx].sum(1))
        print(f'{tag}: names the search\'s answer where it kept the trees\' {hit[~ch].mean():.1%} (chance {chance[~ch].mean():.1%}), '
              f'where it changed it {hit[ch].mean():.1%} (chance {chance[ch].mean():.1%}; the trees\' own instead {as_own[ch].mean():.1%}); '
              f'all {hit.mean():.1%}', flush=True)

    check('before')
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    for epoch in range(args.epochs):
        np.random.shuffle(train_idx)
        total, n = 0.0, 0
        for b in B.batches(data, train_idx, args.batch, device, mean, std):
            s = model(b)[0].masked_fill(torch.isnan(b['sv']), -1e9)
            loss = F.cross_entropy(s, b['y'])
            opt.zero_grad(); loss.backward(); opt.step()
            total += loss.item() * len(b['y']); n += len(b['y'])
        check(f'pass {epoch + 1} (loss {total / n:.3f})')


if __name__ == '__main__':
    main()
