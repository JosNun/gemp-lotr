"""A neural model of people's card-action choices that sees the whole board.

The tree models see a decision as ~75 board totals plus the option's card, and fit their own training months barely
better than held-out ones (Fellowship plays 67.4% against 66.1%): what limits them is what the inputs can say. Here
every card in play and in the deciding player's hand is an item -- its printed numbers and card-text tokens, a learned
vector for cards seen often enough, and its live state (whose, zone, attached to, strength, vitality, wounds, in the
skirmish, assigned, Ring-bearer) -- read together by a small transformer with one more item for the board totals and
the prompt. Each option is scored from its card's item as read in that context; pass has a learned item of its own.
Trained on people's choices (cross-entropy over the decision's options), judged by agreement on held-out July.

usage (from nn/):  uv run python board_net.py --data ~/dev/gemp-replays/labels/nn [--epochs 4]

Data (HumanReplay --boards, BoardDump cards): boards-MM.jsonl, one decision a line; cards.tsv, the card table.
"""
import argparse, json, math, os, random, re, time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ZONES = ['FREE_CHARACTERS', 'SHADOW_CHARACTERS', 'SUPPORT', 'ATTACHED', 'HAND', 'ADVENTURE_PATH', 'STACKED',
         'DISCARD', 'DECK', 'DEAD']
KINDS = ['play', 'use', 'heal', 'transfer']
CONTEXTS = ['Play Fellowship action or Pass', 'Play Shadow action or Pass']
MAX_CARDS = 48
MAX_OPTS = 24
MAX_TOK = 20
MAX_BAG = 40
BAGS = ['myDiscard', 'oppDiscard', 'myDead', 'oppDead']
TWILIGHT = 6  # the set-2 state's twilight pool


def base(bp):
    return re.sub(r'[*T]+$', '', bp)


def load_cards(path):
    """The card table: row 0 is unknown; rows[blueprint] -> row; matrix [rows, width] of numbers and card text."""
    rows, vecs = {}, []
    with open(path) as f:
        next(f)
        for line in f:
            bp, n, t = line.rstrip('\n').split('\t')
            rows[bp] = len(vecs) + 1
            vecs.append(np.array([float(x) for x in n.split()] + [float(x) for x in t.split()], dtype=np.float32))
    width = len(vecs[0])
    return rows, np.vstack([np.zeros(width, np.float32)] + vecs), width


def encode(path, rows, seen, grow):
    """One month's boards as fixed-size arrays: each item its card-table row and live state. seen: the rows met in
    training months, which get a learned vector (others share row 0's)."""
    S, CF, CD, CI, CM, OS, OK, OM, Y, CX = [], [], [], [], [], [], [], [], [], []
    OC, TI, TV, BG = [], [], [], []
    for line in open(path):
        r = json.loads(line)
        if r['ctx'] not in CONTEXTS or len(r['opts']) > MAX_OPTS:
            continue
        cards = r['cards'][:MAX_CARDS]
        cf = np.zeros(MAX_CARDS, np.int16)
        cd = np.zeros((MAX_CARDS, 10 + len(ZONES)), np.float16)
        ci = np.zeros(MAX_CARDS, np.int16)
        cm = np.zeros(MAX_CARDS, np.bool_)
        for i, c in enumerate(cards):
            row = rows.get(c['bp'], rows.get(base(c['bp']), 0))
            cf[i] = row
            if grow and row:
                seen.add(row)
            ci[i] = row if row in seen else 0
            z = ZONES.index(c['zone']) if c['zone'] in ZONES else -1
            cd[i, :10] = [c['own'], c['str'] / 10, c['vit'] / 5, c['wnd'] / 5, c['skir'], c['asg'], c['rb'],
                          1.0 if c['att'] >= 0 else 0.0, (c['att'] + 1) / MAX_CARDS, 1.0]
            if z >= 0:
                cd[i, 10 + z] = 1
            cm[i] = True
        os_ = np.full(MAX_OPTS, -1, np.int64)
        ok = np.zeros(MAX_OPTS, np.int64)
        oc = np.zeros(MAX_OPTS, np.float32)
        ti = np.zeros((MAX_OPTS, MAX_TOK), np.int16)  # ability tokens, 1-based (0: none)
        tv = np.zeros((MAX_OPTS, MAX_TOK), np.float16)
        bg = np.zeros((len(BAGS), MAX_BAG), np.int16)
        om = np.zeros(MAX_OPTS + 1, np.bool_)
        for j, o in enumerate(r['opts']):
            os_[j] = o['src'] if 0 <= o['src'] < MAX_CARDS else -1
            t = o['text'].lower()
            ok[j] = next((k + 1 for k, kind in enumerate(KINDS) if t.startswith(kind)), 0)
            oc[j] = o.get('cost', 0)
            for k, (i, v) in enumerate(o.get('tok', [])[:MAX_TOK]):
                ti[j, k] = int(i) + 1
                tv[j, k] = v
            om[j] = True
        for k, name in enumerate(BAGS):
            for i, bp in enumerate(r.get('bags', {}).get(name, [])[:MAX_BAG]):
                bg[k, i] = rows.get(bp, rows.get(base(bp), 0)) or -1  # -1: a card not in the table, still counted
        om[MAX_OPTS] = True  # pass
        S.append(np.array(r['state'], np.float32)); CF.append(cf); CD.append(cd); CI.append(ci); CM.append(cm)
        OS.append(os_); OK.append(ok); OM.append(om); OC.append(oc); TI.append(ti); TV.append(tv); BG.append(bg)
        Y.append(r['chosen'] if r['chosen'] >= 0 else MAX_OPTS)
        CX.append(CONTEXTS.index(r['ctx']))
    return dict(state=np.stack(S), cf=np.stack(CF), cd=np.stack(CD), ci=np.stack(CI), cm=np.stack(CM),
                os=np.stack(OS), ok=np.stack(OK), om=np.stack(OM), y=np.array(Y), cx=np.array(CX),
                oc=np.stack(OC), ti=np.stack(TI), tv=np.stack(TV), bg=np.stack(BG))


class BoardNet(nn.Module):
    def __init__(self, table, state_width, d=128, layers=3, heads=4, id_drop=0.3, dropout=0.1, n_tok=541):
        super().__init__()
        self.register_buffer('table', torch.from_numpy(table))
        width, n_ids = table.shape[1], table.shape[0] - 1
        self.card = nn.Sequential(nn.Linear(width, d), nn.GELU(), nn.Linear(d, d))
        self.dyn = nn.Linear(10 + len(ZONES), d)
        self.ids = nn.Embedding(n_ids + 1, d, padding_idx=0)
        self.id_drop = id_drop
        self.state = nn.Sequential(nn.Linear(state_width + len(CONTEXTS), d), nn.GELU(), nn.Linear(d, d))
        layer = nn.TransformerEncoderLayer(d, heads, 4 * d, dropout=dropout, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, layers)
        self.kind = nn.Embedding(len(KINDS) + 1, d)
        self.tok = nn.Embedding(n_tok + 1, d, padding_idx=0)  # the ability an option fires, as card-text tokens
        self.cost = nn.Linear(3, d)  # its cost, whether the pool covers it, the pool after it
        self.bag_type = nn.Embedding(len(BAGS), d)
        self.bag_count = nn.Linear(1, d)
        self.pass_item = nn.Parameter(torch.zeros(d))
        self.score = nn.Sequential(nn.Linear(3 * d, d), nn.GELU(), nn.Linear(d, 1))

    def forward(self, b):
        ids = b['ci']
        if self.training and self.id_drop > 0:  # lean on what the card is and does, not only on which card it is
            ids = ids.masked_fill(torch.rand(ids.shape, device=ids.device) < self.id_drop, 0)
        items = self.card(self.table[b['cf'].long()]) + self.dyn(b['cd'].float()) + self.ids(ids.long())
        ctx = F.one_hot(b['cx'], len(CONTEXTS)).float()
        st = self.state(torch.cat([b['state'], ctx], -1)).unsqueeze(1)
        # the piles: each one item, the mean of its cards plus what pile it is and how big
        bg = b['bg'].long()
        bm = bg != 0
        bc = self.card(self.table[bg.clamp(min=0)])
        bags = (bc * bm.unsqueeze(-1)).sum(2) / bm.sum(2, keepdim=True).clamp(min=1)
        n = bm.sum(2, keepdim=True).float() / 20
        bags = bags + self.bag_type.weight.unsqueeze(0) + self.bag_count(n)
        x = torch.cat([st, bags, items], 1)
        nb = len(BAGS)
        mask = torch.cat([torch.ones_like(b['cm'][:, :1 + nb]), b['cm']], 1)
        h = self.encoder(x, src_key_padding_mask=~mask)
        board, cards = h[:, 0], h[:, 1 + nb:]
        src = b['os'].clamp(min=0)
        opt = torch.gather(cards, 1, src.unsqueeze(-1).expand(-1, -1, cards.size(-1)))
        tw = b['raw_tw'].unsqueeze(1)
        cost = torch.stack([b['oc'] / 10, (b['oc'] <= tw).float(), (tw - b['oc']) / 10], -1)
        tok = (self.tok(b['ti'].long()) * b['tv'].float().unsqueeze(-1)).sum(2)
        opt = opt * (b['os'] >= 0).unsqueeze(-1) + self.kind(b['ok']) + self.cost(cost) + tok
        opt = torch.cat([opt, self.pass_item.expand(opt.size(0), 1, -1)], 1)
        pooled = (cards * b['cm'].unsqueeze(-1)).sum(1) / b['cm'].sum(1, keepdim=True).clamp(min=1)
        feats = torch.cat([opt, board.unsqueeze(1).expand_as(opt), pooled.unsqueeze(1).expand_as(opt)], -1)
        s = self.score(feats).squeeze(-1)
        return s.masked_fill(~b['om'], -1e9)


def batches(data, idx, size, device, mean, std):
    for i in range(0, len(idx), size):
        j = idx[i:i + size]
        b = {k: torch.from_numpy(v[j]).to(device) for k, v in data.items()}
        b['raw_tw'] = b['state'][:, TWILIGHT].clone()
        b['state'] = (b['state'] - mean) / std
        yield b


def evaluate(model, data, device, mean, std):
    model.eval()
    hit = np.zeros(len(CONTEXTS)); n = np.zeros(len(CONTEXTS)); loss = 0.0
    with torch.no_grad():
        for b in batches(data, np.arange(len(data['y'])), 512, device, mean, std):
            s = model(b)
            loss += F.cross_entropy(s, b['y'], reduction='sum').item()
            ok = (s.argmax(-1) == b['y']).cpu().numpy()
            for c in range(len(CONTEXTS)):
                m = b['cx'].cpu().numpy() == c
                hit[c] += ok[m].sum(); n[c] += m.sum()
    model.train()
    return hit / np.maximum(n, 1), n, loss / len(data['y'])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default=os.path.expanduser('~/dev/gemp-replays/labels/nn'))
    ap.add_argument('--train', default='01,02,03,04,05,06,08,09,10,11')
    ap.add_argument('--prefix', default='boards2', help='boards (first records) or boards2 (with costs, abilities, piles)')
    ap.add_argument('--test', default='07')
    ap.add_argument('--epochs', type=int, default=4)
    ap.add_argument('--d', type=int, default=128)
    ap.add_argument('--layers', type=int, default=3)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--dropout', type=float, default=0.1)
    ap.add_argument('--wd', type=float, default=0.01)
    ap.add_argument('--id-drop', type=float, default=0.3)
    ap.add_argument('--batch', type=int, default=256)
    ap.add_argument('--save', default='board_net.pt')
    args = ap.parse_args()
    torch.manual_seed(1); random.seed(1); np.random.seed(1)
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    rows, table, width = load_cards(os.path.join(args.data, 'cards.tsv'))
    seen = set()
    t0 = time.time()
    parts = []
    for m in args.train.split(','):
        p = os.path.join(args.data, f'{args.prefix}-{m}.jsonl')
        if os.path.exists(p):
            parts.append(encode(p, rows, seen, grow=True))
    train = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    test = encode(os.path.join(args.data, f'{args.prefix}-{args.test}.jsonl'), rows, seen, grow=False)
    print(f'{len(train["y"])} training decisions, {len(test["y"])} held out, {len(seen)} cards, '
          f'{time.time() - t0:.0f}s to load', flush=True)
    mean = torch.from_numpy(train['state'].mean(0)).to(device)
    std = torch.from_numpy(train['state'].std(0) + 1e-3).to(device)
    model = BoardNet(table, train['state'].shape[1], d=args.d, layers=args.layers, id_drop=args.id_drop,
                     dropout=args.dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    steps = args.epochs * math.ceil(len(train['y']) / args.batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=steps, pct_start=0.05)
    print(f'{sum(p.numel() for p in model.parameters()):,} parameters on {device}', flush=True)
    for epoch in range(args.epochs):
        idx = np.random.permutation(len(train['y']))
        t0 = time.time(); total = 0.0
        for b in batches(train, idx, args.batch, device, mean, std):
            loss = F.cross_entropy(model(b), b['y'])
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step()
            total += loss.item() * len(b['y'])
        acc, n, tl = evaluate(model, test, device, mean, std)
        print(f'epoch {epoch + 1}: train loss {total / len(train["y"]):.4f}, July loss {tl:.4f}, '
              + ', '.join(f'{c.split()[1]} {a:.1%} of {int(k)}' for c, a, k in zip(CONTEXTS, acc, n))
              + f' ({time.time() - t0:.0f}s)', flush=True)
    torch.save({'model': model.state_dict(), 'seen': sorted(seen), 'mean': mean.cpu(), 'std': std.cpu(), 'args': vars(args)},
               args.save)


if __name__ == '__main__':
    main()
