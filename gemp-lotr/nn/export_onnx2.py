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


class Exported(torch.nn.Module):
    def __init__(self, model, mean, std):
        super().__init__()
        self.model = model
        self.register_buffer('mean', mean)
        self.register_buffer('std', std)

    def forward(self, state, cx, cf, ci, cm, cd, cz, ck, os_, ok, oc, ti, tv, om, am, af, bg, bc):
        b = dict(state=(state - self.mean) / self.std, raw_tw=state[:, bn.TWILIGHT], cx=cx, cf=cf, ci=ci, cm=cm, cd=cd,
                 cz=cz, ck=ck, os=os_, ok=ok, oc=oc, ti=ti, tv=tv, om=om, am=am, af=af, bg=bg, bc=bc)
        s, a, o = self.model(b)
        return torch.log_softmax(s, -1), torch.log_softmax(a, -1), torch.softmax(o, -1)


def inputs(data, idx):
    return [torch.from_numpy(np.ascontiguousarray(data[k][idx]).astype(TYPES.get(k, np.int64))) for k in NAMES]


def export(ck_path, table, data, path):
    ck = torch.load(ck_path, weights_only=False, map_location='cpu')
    a = ck['args']
    model = bn.BoardNet2(table, data['state'].shape[1], d=a['d'], layers=a['layers'])
    model.load_state_dict(ck['model'])
    model.eval()
    ex = Exported(model, ck['mean'], ck['std']).eval()
    torch.onnx.export(ex, tuple(inputs(data, np.arange(4))), path, input_names=NAMES,
                      output_names=['opt', 'asg', 'win'],
                      dynamic_axes={n: {0: 'batch'} for n in NAMES + ['opt', 'asg', 'win']}, opset_version=17,
                      dynamo=False)
    return ex, ck


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--judge', default=None)
    ap.add_argument('--out', required=True)
    ap.add_argument('--data', default=os.path.expanduser('~/dev/gemp-replays/labels/nn'))
    ap.add_argument('--check', default=None)
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
    ex, _ = export(args.model, table, data, os.path.join(args.out, 'board-net2.onnx'))
    if args.judge:
        export(args.judge, table, data, os.path.join(args.out, 'board-net2-judge.onnx'))
    shutil.copy(os.path.join(args.data, 'cards.tsv'), os.path.join(args.out, 'board-net2-cards.tsv'))
    with open(os.path.join(args.out, 'board-net2-seen.txt'), 'w') as f:
        f.write('\n'.join(str(r) for r in sorted(ck['seen'])) + '\n')
    import onnxruntime as ort
    sess = ort.InferenceSession(os.path.join(args.out, 'board-net2.onnx'))
    idx = np.arange(len(data['cx']))
    feeds = {n: t.numpy() for n, t in zip(NAMES, inputs(data, idx))}
    got = sess.run(None, feeds)
    with torch.no_grad():
        want = [t.numpy() for t in ex(*inputs(data, idx))]
    om = feeds['om']
    d_opt = np.abs(np.where(om, got[0] - want[0], 0)).max()
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
