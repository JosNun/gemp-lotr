"""Export a trained board network to ONNX for the bot (ONNX Runtime from Java), and check it against PyTorch.

usage (from nn/):  uv run python export_onnx.py --model <.pt> --out <dir> [--check boards3-07.jsonl]

Writes <dir>/board-net.onnx (raw inputs in; the state's normalisation is inside; out: log-probabilities over the
options, pass last), <dir>/board-net-cards.tsv (the card table, whose order is the rows the inputs refer to) and
<dir>/board-net-seen.txt (the rows with a learned vector; others feed 0). The Java encoder must build exactly what
board_net.encode builds.
"""
import argparse, os, shutil

import numpy as np
import torch

import board_net as bn

NAMES = ['state', 'cx', 'cf', 'cd', 'ci', 'cm', 'os', 'ok', 'om', 'oc', 'ti', 'tv', 'bg']


class Exported(torch.nn.Module):
    def __init__(self, model, mean, std):
        super().__init__()
        self.model = model
        self.register_buffer('mean', mean)
        self.register_buffer('std', std)

    def forward(self, state, cx, cf, cd, ci, cm, os_, ok, om, oc, ti, tv, bg):
        b = dict(state=(state - self.mean) / self.std, raw_tw=state[:, bn.TWILIGHT], cx=cx, cf=cf, cd=cd, ci=ci,
                 cm=cm, os=os_, ok=ok, om=om, oc=oc, ti=ti, tv=tv, bg=bg)
        return torch.log_softmax(self.model(b), -1)


def inputs(data, idx):
    d = {k: data[k][idx] for k in NAMES}
    return [torch.from_numpy(np.ascontiguousarray(d['state']).astype(np.float32)),
            torch.from_numpy(d['cx'].astype(np.int64)), torch.from_numpy(d['cf'].astype(np.int64)),
            torch.from_numpy(d['cd'].astype(np.float32)), torch.from_numpy(d['ci'].astype(np.int64)),
            torch.from_numpy(d['cm'].astype(np.bool_)), torch.from_numpy(d['os'].astype(np.int64)),
            torch.from_numpy(d['ok'].astype(np.int64)), torch.from_numpy(d['om'].astype(np.bool_)),
            torch.from_numpy(d['oc'].astype(np.float32)), torch.from_numpy(d['ti'].astype(np.int64)),
            torch.from_numpy(d['tv'].astype(np.float32)), torch.from_numpy(d['bg'].astype(np.int64))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--data', default=os.path.expanduser('~/dev/gemp-replays/labels/nn'))
    ap.add_argument('--check', default=None)
    args = ap.parse_args()
    torch.backends.mha.set_fastpath_enabled(False)
    ck = torch.load(args.model, weights_only=False)
    a = ck['args']
    rows, table, width = bn.load_cards(os.path.join(args.data, 'cards.tsv'))
    check_path = args.check or os.path.join(args.data, 'boards3-07.jsonl')
    data, _ = bn.encode(check_path, rows, set(ck['seen']), grow=False)
    model = bn.BoardNet(table, data['state'].shape[1], d=a['d'], layers=a['layers'],
                        id_drop=a.get('id_drop', 0.3), dropout=a.get('dropout', 0.1))
    model.load_state_dict(ck['model'])
    model.eval()
    ex = Exported(model, ck['mean'], ck['std']).eval()
    os.makedirs(args.out, exist_ok=True)
    sample = inputs(data, np.arange(4))
    torch.onnx.export(ex, tuple(sample), os.path.join(args.out, 'board-net.onnx'), input_names=NAMES,
                      output_names=['logp'], dynamic_axes={n: {0: 'batch'} for n in NAMES + ['logp']},
                      opset_version=17, dynamo=False)
    shutil.copy(os.path.join(args.data, 'cards.tsv'), os.path.join(args.out, 'board-net-cards.tsv'))
    with open(os.path.join(args.out, 'board-net-seen.txt'), 'w') as f:
        f.write('\n'.join(str(r) for r in sorted(ck['seen'])) + '\n')
    import onnxruntime as ort
    sess = ort.InferenceSession(os.path.join(args.out, 'board-net.onnx'))
    idx = np.arange(min(2000, len(data['y'])))
    feeds = {n: t.numpy() for n, t in zip(NAMES, inputs(data, idx))}
    got = sess.run(None, feeds)[0]
    with torch.no_grad():
        want = ex(*inputs(data, idx)).numpy()
    m = feeds['om']
    diff = np.abs(np.where(m, got - want, 0)).max()
    same = (np.where(m, got, -1e9).argmax(1) == np.where(m, want, -1e9).argmax(1)).mean()
    print(f'exported to {args.out}; ONNX Runtime against PyTorch on {len(idx)} decisions: max |diff| {diff:.2e}, '
          f'same choice {same:.2%}')


if __name__ == '__main__':
    main()
