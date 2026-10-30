"""One network for people's decisions -- card plays in every phase, assignments, "move again?", reconcile -- that also
says how the game is going to end.

The first network (board_net.py) took only Fellowship and Shadow card plays. This one reads the v4 board records
(BoardDump: every kind; live keywords and resistance per card; the player's own adventure deck; what the opponent has
shown this game; the own deck's remainder) with one transformer and answers through three heads:

- options: card plays (any phase), reconcile (keep, or discard a card) and move again (yes, no) -- each option scored
  from its card's item as read in context, its kind, cost and ability tokens; pass is an item of its own;
- assignment: per minion offered, a score for each character it could go to and for leaving it unassigned;
- outcome: six ways the game ends, from the deciding player's side -- won or lost, by site 9, by killing the
  Ring-bearer or by corrupting it (game_routes.py; a concession the position does not make clear teaches only won or
  lost, through the sum of its three).

usage (from nn/):  uv run python board_net2.py --data ~/dev/gemp-replays/labels/nn [--epochs 8]
"""
import argparse, collections, json, math, os, random, re, time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from board_net import load_cards, base

ZONES = ['FREE_CHARACTERS', 'SHADOW_CHARACTERS', 'SUPPORT', 'ATTACHED', 'HAND', 'ADVENTURE_PATH', 'STACKED',
         'DISCARD', 'DECK', 'DEAD', 'ADVENTURE_DECK', 'REMOVED']
N_KW = 16  # BoardDump.KEYWORDS
KINDS = ['other', 'play', 'use', 'heal', 'transfer', 'keep', 'discard', 'yes', 'no']
CONTEXTS = ['fellowship', 'shadow', 'maneuver', 'archery', 'assignment', 'skirmish', 'regroup', 'optional', 'required',
            'other', 'reconcile', 'move', 'assign fp', 'assign shadow']
BAGS = ['myDiscard', 'oppDiscard', 'myDead', 'oppDead', 'oppSeen', 'myDeck']
ROUTES = ['site9', 'kill', 'corrupt']
MAX_CARDS, MAX_OPTS, MAX_TOK, MAX_BAG, MAX_MIN, MAX_FPC = 64, 24, 20, 48, 10, 12
DYN = 12  # own, str, vit, res, wnd, act, skir, asg, rb, attached, att index, cost
SCALE = [1, 10, 5, 10, 5, 1, 1, 1, 1, 1, MAX_CARDS, 10]
TWILIGHT = 6  # the set-2 state's twilight pool
EXTRA = 3  # deck remainder: known, unknown, whether a decklist is known at all


def context(r):
    k = r.get('kind', 'action')
    if k in ('move', 'reconcile'):
        return k
    if k == 'assign':
        return 'assign fp' if r['fp'] else 'assign shadow'
    t = r['ctx'] or ''
    for key, name in (('Fellowship action', 'fellowship'), ('Shadow action', 'shadow'), ('Maneuver', 'maneuver'),
                      ('Archery', 'archery'), ('Assignment action', 'assignment'),
                      ('Choose action to play or Pass', 'skirmish'), ('Regroup', 'regroup'), ('Optional', 'optional'),
                      ('Required', 'required')):
        if key in t:
            return name
    return 'other'


def option_kind(r, o):
    k = r.get('kind', 'action')
    if k == 'move':
        return KINDS.index('yes' if o['text'] == 'Yes' else 'no')
    if k == 'reconcile':
        return KINDS.index(o['text'])
    t = o['text'].lower()
    return next((i for i, kind in enumerate(KINDS) if i and t.startswith(kind)), 0)


def load_routes(path):
    return {x['game']: x for x in map(json.loads, open(path))} if path and os.path.exists(path) else {}


def encode(path, rows, seen, grow, routes, kinds=None):
    """A month's boards as fixed-size arrays (int8/int16 where they fit: a million boards in ~6 GB)."""
    out = collections.defaultdict(list)
    for line in (open(path) if isinstance(path, str) else path):  # a file, or the lines themselves
        r = json.loads(line)
        kind = r.get('kind', 'action')
        if kinds and kind not in kinds:
            continue
        opts = r.get('opts', [])
        if len(opts) > MAX_OPTS:
            continue
        cards = r['cards'][:MAX_CARDS]
        cf = np.zeros(MAX_CARDS, np.int16); ci = np.zeros(MAX_CARDS, np.int16); cm = np.zeros(MAX_CARDS, np.bool_)
        cd = np.zeros((MAX_CARDS, DYN), np.int8); cz = np.zeros(MAX_CARDS, np.int8); ck = np.zeros((MAX_CARDS, N_KW), np.int8)
        for i, c in enumerate(cards):
            row = rows.get(c['bp'], rows.get(base(c['bp']), 0))
            cf[i] = row
            if grow and row:
                seen.add(row)
            ci[i] = row if row in seen else 0
            cm[i] = True
            cz[i] = ZONES.index(c['zone']) + 1 if c['zone'] in ZONES else 0
            att = c['att'] if 0 <= c['att'] < MAX_CARDS else -1
            cd[i] = [c['own'], min(c['str'], 127), min(c['vit'], 127), min(c.get('res', 0), 127), min(c['wnd'], 127),
                     c.get('act', 0), c['skir'], c['asg'], c['rb'], att >= 0, att + 1, min(c.get('cost', 0), 127)]
            for k, n in c.get('kw', []):
                ck[i, k] = min(n, 127)
        os_ = np.full(MAX_OPTS, -1, np.int8); ok = np.zeros(MAX_OPTS, np.int8); oc = np.zeros(MAX_OPTS, np.int8)
        ti = np.zeros((MAX_OPTS, MAX_TOK), np.int16); tv = np.zeros((MAX_OPTS, MAX_TOK), np.int8)
        om = np.zeros(MAX_OPTS + 1, np.bool_)
        oe = np.full(MAX_OPTS + 1, -1, np.int16); oe[MAX_OPTS] = -2
        same = {}
        for j, o in enumerate(opts):
            src = o['src'] if 0 <= o['src'] < len(cards) else -1
            os_[j] = src
            ok[j] = option_kind(r, o)
            oc[j] = min(o.get('cost', 0), 127)
            for t, (i, v) in enumerate(o.get('tok', [])[:MAX_TOK]):
                ti[j, t] = int(i) + 1; tv[j, t] = min(int(v), 127)
            om[j] = True
            # the same card doing the same thing is the same answer (two copies of a minion in hand)
            key = (base(r['cards'][o['src']]['bp']) if src >= 0 else '?' + str(j), o['text'])
            oe[j] = same.setdefault(key, len(same))
        y = -100
        am = np.full(MAX_MIN, -1, np.int8); af = np.full(MAX_FPC, -1, np.int8); ay = np.full(MAX_MIN, -100, np.int8)
        if kind == 'assign':
            fp, mins = r['asg']['fp'], r['asg']['min']
            if len(fp) > MAX_FPC or len(mins) > MAX_MIN or any(not (0 <= x < MAX_CARDS) for x in fp + mins):
                continue
            af[:len(fp)] = fp
            am[:len(mins)] = mins
            for i, t in enumerate(r['chosen']):
                ay[i] = t if t >= 0 else MAX_FPC
        elif kind == 'action':
            om[MAX_OPTS] = True  # pass
            y = r['chosen'] if r['chosen'] >= 0 else MAX_OPTS
        else:
            y = r['chosen']
            if not (0 <= y < len(opts)):
                continue
        bg = np.zeros((len(BAGS), MAX_BAG), np.int16); bc = np.zeros((len(BAGS), MAX_BAG), np.int8)
        bags = r.get('bags', {})
        for k, name in enumerate(BAGS):
            counts = collections.Counter(base(b) for b in bags.get(name, []))
            for i, (b, n) in enumerate(counts.most_common(MAX_BAG)):
                bg[k, i] = rows.get(b, 0) or -1  # -1: a card not in the table, still counted
                bc[k, i] = min(n, 127)
        known = len(bags.get('myDeck', []))
        extra = [known / 20, bags.get('myDeckUnknown', 0) / 20, 1.0 if 'myDeck' in bags else 0.0]
        sv = np.full(MAX_OPTS + 1, np.nan, np.float32)  # the search's value per option (teacher boards), pass last
        for i, v in r.get('searched', []):
            i = int(i)
            if i < len(opts):
                sv[i if i >= 0 else MAX_OPTS] = v
        g = routes.get(r.get('game'))
        win, route = -1, -1
        if g:
            win = int(g['winner'] == r['player'])
            if g['route'] in ROUTES:
                route = ROUTES.index(g['route']) + (0 if win else 3)
        for k, v in (('state', np.array(r['state'] + extra, np.float32)), ('cf', cf), ('ci', ci), ('cm', cm),
                     ('cd', cd), ('cz', cz), ('ck', ck), ('os', os_), ('ok', ok), ('oc', oc), ('ti', ti), ('tv', tv),
                     ('om', om), ('oe', oe), ('y', y), ('am', am), ('af', af), ('ay', ay), ('bg', bg), ('bc', bc),
                     ('cx', CONTEXTS.index(context(r))), ('win', win), ('route', route), ('sv', sv)):
            out[k].append(v)
        out['player'].append(r.get('player', '')); out['game'].append(r.get('game', ''))
    if not out:
        return {}, []
    data = {k: np.stack(v) if isinstance(v[0], np.ndarray) else np.array(v) for k, v in out.items()
            if k not in ('player', 'game')}
    data['y'] = data['y'].astype(np.int64); data['cx'] = data['cx'].astype(np.int64)
    data['win'] = data['win'].astype(np.int64); data['route'] = data['route'].astype(np.int64)
    return data, out['game']


class BoardNet2(nn.Module):
    def __init__(self, table, state_width, d=256, layers=4, heads=4, id_drop=0.3, dropout=0.1, n_tok=541):
        super().__init__()
        self.register_buffer('table', torch.from_numpy(table))
        self.register_buffer('scale', torch.tensor(SCALE, dtype=torch.float32))
        width, n_ids = table.shape[1], table.shape[0] - 1
        self.card = nn.Sequential(nn.Linear(width, d), nn.GELU(), nn.Linear(d, d))
        self.dyn = nn.Linear(DYN + N_KW + 2, d)
        self.zone = nn.Embedding(len(ZONES) + 1, d)
        self.ids = nn.Embedding(n_ids + 1, d, padding_idx=0)
        self.id_drop = id_drop
        self.state = nn.Sequential(nn.Linear(state_width + len(CONTEXTS), d), nn.GELU(), nn.Linear(d, d))
        layer = nn.TransformerEncoderLayer(d, heads, 4 * d, dropout=dropout, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, layers)
        self.kind = nn.Embedding(len(KINDS), d)
        self.tok = nn.Embedding(n_tok + 1, d, padding_idx=0)
        self.cost = nn.Linear(3, d)
        self.bag_type = nn.Embedding(len(BAGS), d)
        self.bag_card = nn.Linear(d + 1, d)
        self.bag_count = nn.Linear(1, d)
        self.pass_item = nn.Parameter(torch.zeros(d))
        self.score = nn.Sequential(nn.Linear(3 * d, d), nn.GELU(), nn.Linear(d, 1))
        self.pair = nn.Sequential(nn.Linear(4 * d, d), nn.GELU(), nn.Linear(d, 1))
        self.alone = nn.Sequential(nn.Linear(2 * d, d), nn.GELU(), nn.Linear(d, 1))
        self.outcome = nn.Sequential(nn.Linear(2 * d, d), nn.GELU(), nn.Linear(d, 2 * len(ROUTES)))

    def forward(self, b):
        ids = b['ci']
        if self.training and self.id_drop > 0:  # lean on what the card is and does, not only on which card it is
            ids = ids.masked_fill(torch.rand(ids.shape, device=ids.device) < self.id_drop, 0)
        tw = b['raw_tw'].unsqueeze(1)
        dyn = b['cd'].float() / self.scale
        cost = b['cd'][..., 11].float()
        in_hand = (b['cz'] == ZONES.index('HAND') + 1).float()
        dyn = torch.cat([dyn, b['ck'].float() / 3, (in_hand * (cost <= tw).float()).unsqueeze(-1),
                         (in_hand * (tw - cost) / 10).unsqueeze(-1)], -1)
        items = (self.card(self.table[b['cf'].long()]) + self.dyn(dyn) + self.zone(b['cz'].long())
                 + self.ids(ids.long()))
        ctx = F.one_hot(b['cx'], len(CONTEXTS)).float()
        st = self.state(torch.cat([b['state'], ctx], -1)).unsqueeze(1)
        # the piles: each one item, its distinct cards weighted by how many copies, plus what pile and how big
        bg = b['bg'].long()
        bm = (bg != 0).float()
        n = b['bc'].float()
        bcard = self.bag_card(torch.cat([self.card(self.table[bg.clamp(min=0)]), (n / 3).unsqueeze(-1)], -1))
        bags = (bcard * (bm * n).unsqueeze(-1)).sum(2) / 10
        size = (bm * n).sum(2, keepdim=True) / 20
        bags = bags + self.bag_type.weight.unsqueeze(0) + self.bag_count(size)
        x = torch.cat([st, bags, items], 1)
        nb = len(BAGS)
        mask = torch.cat([torch.ones_like(b['cm'][:, :1 + nb]), b['cm']], 1)
        h = self.encoder(x, src_key_padding_mask=~mask)
        board, cards = h[:, 0], h[:, 1 + nb:]
        pooled = (cards * b['cm'].unsqueeze(-1)).sum(1) / b['cm'].sum(1, keepdim=True).clamp(min=1)

        def gather(idx):
            return torch.gather(cards, 1, idx.clamp(min=0).long().unsqueeze(-1).expand(-1, -1, cards.size(-1)))

        # options
        opt = gather(b['os']) * (b['os'] >= 0).unsqueeze(-1)
        oc = b['oc'].float()
        cost = torch.stack([oc / 10, (oc <= tw).float(), (tw - oc) / 10], -1)
        tok = (self.tok(b['ti'].long()) * b['tv'].float().unsqueeze(-1)).sum(2)
        opt = opt + self.kind(b['ok'].long()) + self.cost(cost) + tok
        opt = torch.cat([opt, self.pass_item.expand(opt.size(0), 1, -1)], 1)
        feats = torch.cat([opt, board.unsqueeze(1).expand_as(opt), pooled.unsqueeze(1).expand_as(opt)], -1)
        s = self.score(feats).squeeze(-1).masked_fill(~b['om'], -1e9)
        # assignment: minion x character, and minion alone
        hm, hf = gather(b['am']), gather(b['af'])  # [B, M, d], [B, C, d]
        M, C = hm.size(1), hf.size(1)
        pm, pf = hm.unsqueeze(2).expand(-1, -1, C, -1), hf.unsqueeze(1).expand(-1, M, -1, -1)
        bd = board.unsqueeze(1).unsqueeze(1).expand(-1, M, C, -1)
        pair = self.pair(torch.cat([pm, pf, pm * pf, bd], -1)).squeeze(-1)
        alone = self.alone(torch.cat([hm, board.unsqueeze(1).expand_as(hm)], -1))
        a = torch.cat([pair.masked_fill((b['af'] < 0).unsqueeze(1), -1e9), alone], -1)  # [B, M, C + 1]
        # the outcome
        o = self.outcome(torch.cat([board, pooled], -1))
        return s, a, o


def batches(data, idx, size, device, mean, std):
    for i in range(0, len(idx), size):
        j = idx[i:i + size]
        b = {k: torch.from_numpy(v[j]).to(device) for k, v in data.items()}
        b['raw_tw'] = b['state'][:, TWILIGHT].clone()
        b['state'] = (b['state'] - mean) / std
        yield b


LAMBDA = 0.1  # the search's temperature: a value 0.1 higher multiplies an option's target by e


def losses(model, b):
    s, a, o = model(b)
    searched = ~torch.isnan(b['sv'])
    soft = searched.any(1)
    has_y = (b['y'] >= 0) & ~soft
    lo = F.cross_entropy(s, b['y'].clamp(min=0), reduction='none') * has_y
    if soft.any():
        # what people would do (the starting network, frozen), tilted toward what the search says wins, over the
        # options the search tried: target ~ p0 * exp(value / lambda); the network's log-probabilities renormalised there
        neg = torch.full_like(s, -1e9)
        tgt = F.softmax(torch.where(searched, b['lp0'] + torch.nan_to_num(b['sv']) / LAMBDA, neg), -1)
        lp = F.log_softmax(torch.where(searched, s, neg), -1)
        lo = lo + -(tgt * torch.where(searched, lp, torch.zeros_like(lp))).sum(-1) * soft * b['tw']
    la = F.cross_entropy(a.flatten(0, 1), b['ay'].long().flatten(), ignore_index=-100, reduction='none')
    la = la.view(a.size(0), -1)
    n_min = (b['ay'] != -100).sum(1)
    la = la.sum(1) / n_min.clamp(min=1)
    lp = F.log_softmax(o, -1)
    known = b['route'] >= 0
    l_route = -lp.gather(1, b['route'].clamp(min=0).unsqueeze(1)).squeeze(1)
    won = torch.logsumexp(lp[:, :3], -1)
    lost = torch.logsumexp(lp[:, 3:], -1)
    l_win = -torch.where(b['win'] == 1, won, lost)
    lw = torch.where(known, l_route, l_win) * (b['win'] >= 0)
    return lo + la, lw, (s, a, o)


def auc(scores, labels):
    order = np.argsort(scores)
    ranks = np.empty(len(scores)); ranks[order] = np.arange(1, len(scores) + 1)
    pos = labels == 1
    n1, n0 = pos.sum(), (~pos).sum()
    return (ranks[pos].sum() - n1 * (n1 + 1) / 2) / max(1, n1 * n0)


def evaluate(model, data, device, mean, std):
    model.eval()
    hit, n = collections.Counter(), collections.Counter()
    pw, wl, rh, rn = [], [], 0, 0
    with torch.no_grad():
        for b in batches(data, np.arange(len(data['cx'])), 512, device, mean, std):
            s, a, o = model(b)
            cx = b['cx'].cpu().numpy()
            pred = s.argmax(-1)
            oe = b['oe'].long()
            ok = (oe.gather(1, pred.unsqueeze(1)) == oe.gather(1, b['y'].clamp(min=0).unsqueeze(1))).squeeze(1)
            ok = ok.cpu().numpy()
            ay = b['ay'].long()
            whole = ((a.argmax(-1) == ay) | (ay == -100)).all(1).cpu().numpy()
            for i, c in enumerate(cx):
                name = CONTEXTS[c]
                if name.startswith('assign'):
                    hit[name] += whole[i]
                else:
                    if b['y'][i] < 0:
                        continue
                    hit[name] += ok[i]
                n[name] += 1
            p = F.softmax(o, -1)
            w = b['win'].cpu().numpy()
            pw.append(p[:, :3].sum(-1).cpu().numpy()[w >= 0]); wl.append(w[w >= 0])
            r = b['route'].cpu().numpy()
            rh += (p.argmax(-1).cpu().numpy() == r)[r >= 0].sum(); rn += (r >= 0).sum()
    model.train()
    pw, wl = np.concatenate(pw), np.concatenate(wl)
    return hit, n, auc(pw, wl), rh / max(rn, 1)


def report(hit, n):
    acts = [c for c in CONTEXTS[:10] if n[c]]
    a = sum(hit[c] for c in acts) / max(1, sum(n[c] for c in acts))
    fs = sum(hit[c] for c in ('fellowship', 'shadow')) / max(1, sum(n[c] for c in ('fellowship', 'shadow')))
    return (f'card plays {a:.1%} (Fellowship+Shadow {fs:.1%}); '
            + ', '.join(f'{c} {hit[c] / n[c]:.1%}' for c in CONTEXTS if n[c] and (c not in acts or n[c] > 2000)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default=os.path.expanduser('~/dev/gemp-replays/labels/nn'))
    ap.add_argument('--prefix', default='boards4')
    ap.add_argument('--train', default='01,02,03,04,05,06,08,09,10,11')
    ap.add_argument('--test', default='07')
    ap.add_argument('--routes', default=None, help='game_routes.py output (default <data>/routes-2022.jsonl)')
    ap.add_argument('--outcome-weight', type=float, default=0.3)
    ap.add_argument('--epochs', type=int, default=8)
    ap.add_argument('--d', type=int, default=256)
    ap.add_argument('--layers', type=int, default=4)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--dropout', type=float, default=0.1)
    ap.add_argument('--wd', type=float, default=0.01)
    ap.add_argument('--id-drop', type=float, default=0.3)
    ap.add_argument('--batch', type=int, default=256)
    ap.add_argument('--save', default='board_net2.pt')
    ap.add_argument('--decision-weight', type=float, default=1.0, help='0: train the outcome head alone (a judge)')
    ap.add_argument('--evals', type=int, default=1, help='July checks per epoch')
    ap.add_argument('--save-best-auc', default=None, help='also keep the checkpoint with the best July win AUC here')
    ap.add_argument('--init', default=None, help='start from this checkpoint (its card ids and normalisation too)')
    ap.add_argument('--teacher', default=None, help='searched teacher boards (BotTeach --boards), comma separated')
    ap.add_argument('--teacher-weight', type=float, default=1.0)
    ap.add_argument('--lam', type=float, default=0.1, help='the search temperature for the soft targets')
    ap.add_argument('--per-game', type=int, default=0,
                    help='each epoch, only this many random positions of each game (a judge that cannot memorise games)')
    args = ap.parse_args()
    torch.manual_seed(1); random.seed(1); np.random.seed(1)
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    global LAMBDA
    LAMBDA = args.lam
    rows, table, width = load_cards(os.path.join(args.data, 'cards.tsv'))
    routes = load_routes(args.routes or os.path.join(args.data, 'routes-2022.jsonl'))
    init = torch.load(args.init, weights_only=False, map_location='cpu') if args.init else None
    seen = set(init['seen']) if init else set()
    t0 = time.time()
    parts, games = [], []
    for m in args.train.split(','):
        p = os.path.join(args.data, f'{args.prefix}-{m}.jsonl')
        if os.path.exists(p):
            d, g = encode(p, rows, seen, init is None, routes)
            parts.append(d); games.extend(g)
    n_people = sum(len(p['cx']) for p in parts)
    for path in (args.teacher.split(',') if args.teacher else []):
        d, g = encode(path, rows, seen, False, {})
        if d:
            parts.append(d); games.extend(g)
            print(f'{len(d["cx"])} searched teacher boards from {path}', flush=True)
    for p in parts:
        p['tw'] = np.full(len(p['cx']), args.teacher_weight, np.float32)
        p['lp0'] = np.zeros((len(p['cx']), MAX_OPTS + 1), np.float32)
    train = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    test, _ = encode(os.path.join(args.data, f'{args.prefix}-{args.test}.jsonl'), rows, seen, False, routes)
    print(f'{len(train["cx"])} training decisions, {len(test["cx"])} held out, {len(seen)} cards, '
          f'{sum(v.nbytes for v in train.values()) / 1e9:.1f} GB, {time.time() - t0:.0f}s to load; by kind '
          + ', '.join(f'{c} {k}' for c, k in zip(CONTEXTS, np.bincount(train['cx'], minlength=len(CONTEXTS)))),
          flush=True)
    print(f'outcome labels: {(train["win"] >= 0).mean():.0%} with a winner, {(train["route"] >= 0).mean():.0%} with a route',
          flush=True)
    if init:
        mean, std = init['mean'].to(device), init['std'].to(device)
    else:
        mean = torch.from_numpy(train['state'].mean(0)).to(device)
        std = torch.from_numpy(train['state'].std(0) + 1e-3).to(device)
    model = BoardNet2(table, train['state'].shape[1], d=args.d, layers=args.layers, id_drop=args.id_drop,
                      dropout=args.dropout).to(device)
    if init:
        model.load_state_dict(init['model'])
        # the starting network's choices on the teacher boards: the people-like prior the soft targets tilt
        teach = np.arange(n_people, len(train['cx']))
        if len(teach):
            model.eval()
            with torch.no_grad():
                out = [torch.log_softmax(model(b)[0], -1).cpu().numpy()
                       for b in batches(train, teach, 512, device, mean, std)]
            train['lp0'][teach] = np.concatenate(out)
            model.train()
            # how often the search overrules the network: its best option is not the network's first choice among them
            sv = train['sv'][teach]
            sv_best = np.nanargmax(np.where(np.isnan(sv), -np.inf, sv), 1)
            lp_best = np.argmax(np.where(np.isnan(sv), -np.inf, train['lp0'][teach]), 1)
            spread = np.nanmax(sv, 1) - np.nanmin(sv, 1)
            print(f'search overrules the network on {np.mean(sv_best != lp_best):.1%} of {len(teach)} searched '
                  f'decisions (value spread: median {np.median(spread):.3f})', flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    by_game = collections.defaultdict(list)
    for i, g in enumerate(games):
        by_game[g].append(i)
    by_game = [np.array(v) for v in by_game.values()]

    def epoch_rows():
        if not args.per_game:
            return np.random.permutation(len(train['cx']))
        pick = [v if len(v) <= args.per_game else np.random.choice(v, args.per_game, replace=False) for v in by_game]
        return np.random.permutation(np.concatenate(pick))
    rows_per_epoch = len(epoch_rows())
    if args.per_game:
        print(f'{args.per_game} positions a game: {rows_per_epoch} rows an epoch from {len(by_game)} games', flush=True)
    steps = args.epochs * math.ceil(rows_per_epoch / args.batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=steps, pct_start=0.05)
    print(f'{sum(p.numel() for p in model.parameters()):,} parameters on {device}', flush=True)
    best_auc = 0.0
    per_epoch = math.ceil(rows_per_epoch / args.batch)
    check = set(int(per_epoch * (k + 1) / args.evals) for k in range(args.evals))

    def save(path):
        torch.save({'model': model.state_dict(), 'seen': sorted(seen), 'mean': mean.cpu(), 'std': std.cpu(),
                    'args': vars(args)}, path)

    for epoch in range(args.epochs):
        idx = epoch_rows()
        t0 = time.time(); tot_d = tot_o = 0.0; seen_rows = 0
        for step, b in enumerate(batches(train, idx, args.batch, device, mean, std), 1):
            ld, lw, _ = losses(model, b)
            loss = (args.decision_weight * ld + args.outcome_weight * lw).mean()
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step()
            tot_d += ld.sum().item(); tot_o += lw.sum().item(); seen_rows += len(b['cx'])
            if step not in check:
                continue
            hit, n, a, r = evaluate(model, test, device, mean, std)
            at = f'{epoch + step / per_epoch:.2f}' if args.evals > 1 else f'{epoch + 1}'
            print(f'epoch {at}: train decision loss {tot_d / seen_rows:.4f}, outcome loss {tot_o / seen_rows:.4f}; '
                  f'July: {report(hit, n)}; win AUC {a:.3f}, route {r:.1%} ({time.time() - t0:.0f}s)', flush=True)
            if args.save_best_auc and a > best_auc:
                best_auc = a
                save(args.save_best_auc)
        save(args.save)


if __name__ == '__main__':
    main()
