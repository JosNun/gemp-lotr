"""Export the combined network (board_net2.py) to ONNX for the bot, and write a parity file for the Java encoder.

usage (from nn/):  uv run python export_onnx2.py --model <.pt> --out <model dir> [--judge <.pt>]

Writes into <model dir>:
- board-net2.onnx: raw inputs in (the state's normalisation inside); out "opt" (log-probabilities over the options,
  pass last), "asg" (per minion, log-probabilities over the characters then "unassigned") and "win" (the six ways the
  game ends, as probabilities: won by site 9 / kill / corruption, then lost by them);
- board-net2-judge.onnx with --judge: the same, from the outcome-only run (its "win" is the judge);
- board-net2-cards.tsv and board-net2-seen.txt: the card table (whose order the inputs' rows refer to) and the rows
  with a learned vector;
- board-net2-parity.jsonl: 200 July records with what PyTorch makes of them, so `BoardNet2 parity` can check that the
  Java encoder builds exactly what board_net2.encode builds.
"""
import argparse, json, os, shutil

import numpy as np
import torch

import board_net2 as bn
from board_net import load_cards

NAMES = ['state', 'cx', 'cf', 'ci', 'cm', 'cd', 'cz', 'ck', 'os', 'ok', 'oc', 'ti', 'tv', 'om', 'am', 'af', 'bg', 'bc']
TYPES = {'state': np.float32, 'cm': np.bool_, 'om': np.bool_}
# the batch, and the cards: a board of 25 cards need not run 64 (padding is masked, so the answers are the same) --
# the Java encoder sends only the cards there are
# The cards axis cannot be left open (the attention's reshapes keep the traced length), so the model is exported again
# for smaller boards: board-net2-cK.onnx runs K cards, and the Java encoder picks the smallest K that holds the board.
AXES = {n: {0: 'batch'} for n in NAMES + ['pr', 'opt', 'asg', 'win']}
CARD_INPUTS = ('cf', 'ci', 'cm', 'cd', 'cz', 'ck')
BUCKETS = (16, 24, 32, 40, 48)


def cut(tensors, k):
    return [t[:, :k] if n in CARD_INPUTS else t for n, t in zip(NAMES + ['pr'], tensors)]


class Exported(torch.nn.Module):
    def __init__(self, model, mean, std):
        super().__init__()
        self.model = model
        self.register_buffer('mean', mean)
        self.register_buffer('std', std)

    def forward(self, *xs):
        b = dict(zip(names(self.model), xs))
        state = b['state']
        b['state'], b['raw_tw'] = (state - self.mean) / self.std, state[:, bn.TWILIGHT]
        s, a, o = self.model(b)
        return torch.log_softmax(s, -1), torch.log_softmax(a, -1), torch.softmax(o, -1)


class JudgeOnly(torch.nn.Module):
    """The judge alone (2026-11-10): "win" computed, "opt" and "asg" zeros of the shapes the bot reads -- the option and
    assignment heads were ~15% of a reading (see lean()). Every input stays in the graph (the zeros are made from the
    ones the outcome does not read), so the bot feeds it as it feeds any other."""

    def __init__(self, model, mean, std):
        super().__init__()
        self.model = model
        self.register_buffer('mean', mean)
        self.register_buffer('std', std)

    def forward(self, *xs):
        b = dict(zip(names(self.model), xs))
        state = b['state']
        b['state'], b['raw_tw'] = (state - self.mean) / self.std, state[:, bn.TWILIGHT]
        o = self.model.outcome_only(b)
        z = sum(b[k].sum() for k in ('os', 'ok', 'oc', 'ti', 'tv', 'om', 'am', 'af')).float() * 0
        opt = (state[:, :1] * 0 + z).expand(-1, bn.MAX_OPTS + 1)
        asg = opt[:, :1].unsqueeze(-1).expand(-1, bn.MAX_MIN, bn.MAX_FPC + 1)
        return opt, asg, torch.softmax(o, -1)


def lean(model):
    """A card's vector looked up, not computed: card(table[i]) depends on the card alone, and a board runs it for
    every card in play and all 288 slots of the six piles -- with the heads, most of what a reading does outside the
    encoder. The same answers to rounding (time_onnx.py: judge5b in 8-bit 12.7 -> see BOT-NOTES ms a board)."""
    with torch.no_grad():
        model.table = model.card(model.table)
        w, bias = model.bag_card.weight, model.bag_card.bias  # a pile's card: [its vector, copies / 3] through one layer
        model.register_buffer('bag_table', model.table @ w[:, :-1].T + bias)
        model.register_buffer('bag_n', w[:, -1].clone())
    model.card = torch.nn.Identity()
    return model


def names(model):
    """The inputs: a network that reads the skirmish probe takes "pr" (float, [options + pass, PROBE_N]) last."""
    return NAMES + (['pr'] if hasattr(model, 'probe') else [])


def inputs(data, idx, model=None):
    return [torch.from_numpy(np.ascontiguousarray(data[k][idx]).astype(np.float32 if k == 'pr' else TYPES.get(k, np.int64)))
            for k in (names(model) if model is not None else NAMES)]


def export(ck_path, table, data, path, buckets=True, lean_cards=False, judge_only=False):
    ck = torch.load(ck_path, weights_only=False, map_location='cpu')
    a = ck['args']
    model = bn.BoardNet2(table, data['state'].shape[1], d=a['d'], layers=a['layers'], probe=a.get('probe', False))
    model.load_state_dict(ck['model'])
    model.eval()
    if lean_cards:
        lean(model)
    ex = (JudgeOnly if judge_only else Exported)(model, ck['mean'], ck['std']).eval()
    sample = inputs(data, np.arange(4), model)
    torch.onnx.export(ex, tuple(sample), path, input_names=names(model), output_names=['opt', 'asg', 'win'],
                      dynamic_axes=AXES, opset_version=17, dynamo=False)
    if buckets:
        for k in BUCKETS:
            torch.onnx.export(ex, tuple(cut(sample, k)), path.replace('.onnx', f'-c{k}.onnx'), input_names=names(model),
                              output_names=['opt', 'asg', 'win'], dynamic_axes=AXES, opset_version=17, dynamo=False)
    return ex, ck


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--judge', default=None)
    ap.add_argument('--out', required=True)
    ap.add_argument('--data', default=os.path.expanduser('~/dev/gemp-replays/labels/nn'))
    ap.add_argument('--check', default=None)
    ap.add_argument('--lean', action='store_true',
                    help="cards' vectors precomputed, and the judge exported without the option and assignment heads")
    args = ap.parse_args()
    torch.backends.mha.set_fastpath_enabled(False)
    rows, table, _ = load_cards(os.path.join(args.data, 'cards.tsv'))
    ck = torch.load(args.model, weights_only=False, map_location='cpu')
    check = args.check or os.path.join(args.data, f'{ck["args"]["prefix"]}-07.jsonl')
    # the parity sample: every kind, one record in ~400
    lines = []
    with open(check) as f:
        for i, line in enumerate(f):
            if i % 400 == 0 and len(bn.encode([line], rows, set(ck['seen']), False, {})[1]):  # one encode keeps
                lines.append(line)
            if len(lines) == 200:
                break
    os.makedirs(args.out, exist_ok=True)
    data, _ = bn.encode(lines, rows, set(ck['seen']), False, {})
    ex, _ = export(args.model, table, data, os.path.join(args.out, 'board-net2.onnx'), lean_cards=args.lean)
    if args.judge:
        export(args.judge, table, data, os.path.join(args.out, 'board-net2-judge.onnx'), lean_cards=args.lean,
               judge_only=args.lean)
    shutil.copy(os.path.join(args.data, 'cards.tsv'), os.path.join(args.out, 'board-net2-cards.tsv'))
    with open(os.path.join(args.out, 'board-net2-seen.txt'), 'w') as f:
        f.write('\n'.join(str(r) for r in sorted(ck['seen'])) + '\n')
    import onnxruntime as ort
    sess = ort.InferenceSession(os.path.join(args.out, 'board-net2.onnx'))
    idx = np.arange(len(data['cx']))
    feeds = {n: t.numpy() for n, t in zip(names(ex.model), inputs(data, idx, ex.model))}
    got = sess.run(None, feeds)
    with torch.no_grad():
        want = [t.numpy() for t in ex(*inputs(data, idx, ex.model))]
    om = feeds['om']
    d_opt = np.abs(np.where(om, got[0] - want[0], 0)).max()
    # each smaller model on the boards that fit it: the same answers as the full one
    n_cards = feeds['cm'].sum(1)
    for k in BUCKETS:
        fit = np.where(n_cards <= k)[0]
        if not len(fit):
            continue
        sk = ort.InferenceSession(os.path.join(args.out, f'board-net2-c{k}.onnx'))
        got_k = sk.run(None, {n: (v[fit][:, :k] if n in CARD_INPUTS else v[fit]) for n, v in feeds.items()})
        dk = np.abs(np.where(om[fit], got_k[0] - got[0][fit], 0)).max()
        print(f'{k} cards: {len(fit)} boards fit, options max |diff| {dk:.2e}, outcome {np.abs(got_k[2] - got[2][fit]).max():.2e}')
    d_win = np.abs(got[2] - want[2]).max()
    print(f'exported to {args.out}; ONNX Runtime against PyTorch on {len(idx)} records: options max |diff| {d_opt:.2e}, '
          f'outcome {d_win:.2e}')
    # what the Java encoder must reproduce: per record its options' log-probabilities and the outcome
    with open(os.path.join(args.out, 'board-net2-parity.jsonl'), 'w') as f:
        for line, opt, win in zip(lines, want[0], want[2]):
            r = json.loads(line)
            n = len(r.get('opts', []))
            f.write(json.dumps({'record': r, 'opt': [float(x) for x in opt[:n]] + [float(opt[bn.MAX_OPTS])],
                                'win': [float(x) for x in win]}) + '\n')


if __name__ == '__main__':
    main()
