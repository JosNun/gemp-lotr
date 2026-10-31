"""Play by precedent: the most similar people's decisions vote, instead of the network choosing.

usage (from nn/):  uv run python knn_policy.py --model <board_net2 .pt> [--months 01,03,05] [--k 1,5,25,100]

The user's idea: keep every decision people made, and at a decision find the most similar ones and do what they did.
"Similar" is the network's own reading of the board -- its board token after the transformer, the 256 numbers every
option is scored against -- since raw state puts boards that differ by one card far apart. Each July card play (the
option heads' decisions: every phase's card actions, reconcile, move) looks up the K nearest training boards of the same
kind of prompt (cosine similarity, brute force on the GPU), and each neighbour votes for its chosen option where the
current decision offers the same card doing the same thing (pass for pass), weighted by similarity. Agreement with
people as the network counts it, against the network's own; and with winners' decisions only, the user's variant.
"""
import argparse, collections, json, os

import numpy as np
import torch

import board_net2 as bn
from board_net import load_cards, base


def keys(path, routes):
    """Per decision the encoder keeps: each option's (card, action) key, the chosen one's, and whether the player won."""
    out = []
    for line in open(path):
        r = json.loads(line)
        kind = r.get('kind', 'action')
        if kind == 'assign':
            continue
        opts = r.get('opts', [])
        if len(opts) > bn.MAX_OPTS:
            continue
        cards = r['cards']
        ks = [((base(cards[o['src']]['bp']) if 0 <= o['src'] < len(cards) else '?'), o['text']) for o in opts]
        ch = r.get('chosen', -1)
        if kind != 'action' and not (0 <= ch < len(opts)):
            continue
        chosen = ('pass', '') if ch is None or ch < 0 else ks[ch]
        g = routes.get(r.get('game'))
        out.append((ks, chosen, None if g is None else g['winner'] == r['player'], bn.CONTEXTS.index(bn.context(r))))
    return out


def embed(model, data, dev, mean, std):
    """The board token after the transformer, for every row of data."""
    grabbed = []
    hook = model.encoder.register_forward_hook(lambda m, i, o: grabbed.append(o[:, 0].float().cpu()))
    with torch.no_grad():
        for b in bn.batches(data, np.arange(len(data['cx'])), 512, dev, mean, std):
            model(b)
    hook.remove()
    e = torch.cat(grabbed)
    return torch.nn.functional.normalize(e, dim=1).half()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--data', default=os.path.expanduser('~/dev/gemp-replays/labels/nn'))
    ap.add_argument('--months', default='01,03,05')
    ap.add_argument('--k', default='1,5,25,100')
    args = ap.parse_args()
    dev = 'mps' if torch.backends.mps.is_available() else 'cpu'
    ck = torch.load(args.model, weights_only=False, map_location='cpu')
    a = ck['args']
    rows, table, _ = load_cards(os.path.join(args.data, 'cards.tsv'))
    seen = set(ck['seen'])
    routes = bn.load_routes(os.path.join(args.data, 'routes-2022.jsonl'))
    model = bn.BoardNet2(table, ck['mean'].shape[0], d=a['d'], layers=a['layers']).to(dev)
    model.load_state_dict(ck['model'])
    model.eval()
    mean, std = ck['mean'].to(dev), ck['std'].to(dev)
    kinds = {'action', 'move', 'reconcile'}

    lib_e, lib_k = [], []
    for m in args.months.split(','):
        p = os.path.join(args.data, f'{a["prefix"]}-{m}.jsonl')
        d, _ = bn.encode(p, rows, seen, False, routes, kinds=kinds)
        k = keys(p, routes)
        assert len(k) == len(d['cx']), (len(k), len(d['cx']))
        lib_e.append(embed(model, d, dev, mean, std))
        lib_k.extend(k)
        print(f'month {m}: {len(k)} decisions in the library', flush=True)
        del d
    lib_e = torch.cat(lib_e)
    lib_cx = np.array([k[3] for k in lib_k])
    lib_won = np.array([k[2] is True for k in lib_k])

    test_path = os.path.join(args.data, f'{a["prefix"]}-07.jsonl')
    td, _ = bn.encode(test_path, rows, seen, False, routes, kinds=kinds)
    tk = keys(test_path, routes)
    assert len(tk) == len(td['cx'])
    te = embed(model, td, dev, mean, std)
    # the network's own choice, counted the same way
    with torch.no_grad():
        net = np.concatenate([model(b)[0].argmax(-1).cpu().numpy()
                              for b in bn.batches(td, np.arange(len(td['cx'])), 512, dev, mean, std)])
    print(f'{len(tk)} July decisions; library {len(lib_k)}', flush=True)

    ks = [int(x) for x in args.k.split(',')]
    kmax = max(ks)
    hits = collections.Counter()
    voted = collections.Counter()
    n = collections.Counter()
    lib_dev = lib_e.to(dev)
    for c in range(len(bn.CONTEXTS)):
        q = np.where(td['cx'] == c)[0]
        pool = np.where(lib_cx == c)[0]
        if not len(q) or not len(pool):
            continue
        pool_t = torch.from_numpy(pool).to(dev)
        for winners in (False, True):
            p_idx = pool if not winners else pool[lib_won[pool]]
            if not len(p_idx):
                continue
            keys_e = lib_dev[torch.from_numpy(p_idx).to(dev)]
            for s in range(0, len(q), 1024):
                qi = q[s:s + 1024]
                sim = te[qi].to(dev) @ keys_e.T
                top_s, top_i = sim.float().topk(min(kmax, keys_e.shape[0]), dim=1)
                top_s, top_i = top_s.cpu().numpy(), top_i.cpu().numpy()
                for row, i in enumerate(qi):
                    opts, chosen, _, _ = tk[i]
                    index = {key: j for j, key in enumerate(opts)}
                    want = chosen
                    for K in ks:
                        votes = collections.Counter()
                        for sv, li in zip(top_s[row, :K], top_i[row, :K]):
                            nk = lib_k[p_idx[li]][1]
                            if nk == ('pass', '') or nk in index:
                                votes[nk] += float(sv)
                        tag = ('winners ' if winners else 'all ') + f'K={K}'
                        n[tag] += 1
                        if votes:
                            voted[tag] += 1
                            pick = votes.most_common(1)[0][0]
                        else:
                            pick = ('pass', '')  # no precedent offered here: pass
                        hits[tag] += pick == want
                    if not winners:
                        pick = ('pass', '') if net[i] == bn.MAX_OPTS or net[i] >= len(opts) else opts[net[i]]
                        hits['network'] += pick == want
                        n['network'] += 1
    print(f'network: {100 * hits["network"] / n["network"]:.1f}% of {n["network"]}')
    for winners in (False, True):
        for K in ks:
            tag = ('winners ' if winners else 'all ') + f'K={K}'
            if n[tag]:
                print(f'{tag:14s} agreement {100 * hits[tag] / n[tag]:.1f}%  (a precedent offered on '
                      f'{100 * voted[tag] / n[tag]:.0f}% of decisions)')


if __name__ == '__main__':
    main()
