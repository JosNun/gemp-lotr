"""How long a network takes a board (2026-11-10): each ONNX file on a model directory's parity boards, one board a
call on one thread, each board through the smallest "-cK" export that holds it -- as the bot runs them.

usage (from nn/):  uv run python time_onnx.py <parity.jsonl> <model.onnx> ...   (the base file; its -cK beside it)
"""
import json, os, sys, time

import numpy as np
import onnxruntime as ort

import board_net2 as bn
from board_net import load_cards
from export_onnx2 import BUCKETS, CARD_INPUTS, NAMES, TYPES


def session(path):
    o = ort.SessionOptions()
    o.intra_op_num_threads = 1
    return ort.InferenceSession(path, o)


def main():
    parity, models = sys.argv[1], sys.argv[2:]
    d = os.path.expanduser('~/dev/gemp-replays/labels/nn')
    rows, table, _ = load_cards(os.path.join(d, 'cards.tsv'))
    seen = {int(x) for x in open(os.path.join(os.path.dirname(parity), 'board-net2-seen.txt')).read().split()}
    lines = [json.dumps(json.loads(l)['record']) for l in open(parity)]
    data, _ = bn.encode(lines, rows, seen, False, {})
    n = len(data['cx'])
    cards = data['cm'].sum(1)
    print(f'{n} boards, {cards.mean():.1f} cards each (median {int(np.median(cards))})')
    first = None
    for path in models:
        sess = {k: session(path.replace('.onnx', f'-c{k}.onnx')) for k in BUCKETS
                if os.path.exists(path.replace('.onnx', f'-c{k}.onnx'))}
        sess[bn.MAX_CARDS] = session(path)
        want = {i.name for i in sess[bn.MAX_CARDS].get_inputs()}
        feeds = []
        for i in range(n):
            k = min(b for b in sess if b >= cards[i])
            f = {}
            for name in NAMES:
                if name not in want:
                    continue
                v = np.ascontiguousarray(data[name][i:i + 1]).astype(TYPES.get(name, np.int64))
                f[name] = v[:, :k] if name in CARD_INPUTS else v
            feeds.append((k, f))
        wins = None
        best = 1e9
        for rep in range(5):
            t = time.perf_counter()
            out = [sess[k].run(None, f)[-1][0] for k, f in feeds]
            best = min(best, time.perf_counter() - t)
            wins = np.stack(out)
        first = wins if first is None else first
        print(f'{path}: {1000 * best / n:.2f} ms a board; win against the first file: max |diff| '
              f'{np.abs(wins - first).max():.2e}, mean {np.abs(wins - first).mean():.2e}')


if __name__ == '__main__':
    main()
