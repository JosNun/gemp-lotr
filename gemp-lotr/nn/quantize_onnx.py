"""8-bit transformer layers for a bot's networks (2026-11-09): the encoder's MatMuls in int8 (dynamic quantization),
everything else -- the card, state and bag layers, the output heads -- kept in float.

usage (from nn/):  uv run python quantize_onnx.py <model dir>    (rewrites its board-net2*.onnx in place)

All of a network's matrix layers in 8-bit ran 1.8-2.1x faster but moved log-probabilities by up to 0.57 and changed
the top choice on 5 of 200 parity boards; the encoder's alone, 1.4-1.6x faster, 0.07 at most, 200 of 200 the same
(BoardNet2 parity). In wide two-round search: 11.5 -> 8.6 CPU-seconds a game, 54.3 +- 1.7 against sp-m3 (~55.3 in
float). Write into a copy: the files in a directory made by copying another may be links into it.
"""
import glob, os, sys

import onnx
from onnxruntime.quantization import QuantType, quantize_dynamic


def main():
    d = sys.argv[1]
    for f in sorted(glob.glob(os.path.join(d, 'board-net2*.onnx'))):
        if os.path.islink(f):
            raise SystemExit(f'{f} is a link: quantizing it would write through into another bot')
        m = onnx.load(f)
        nodes = [n.name for n in m.graph.node if n.op_type in ('MatMul', 'Gemm') and '/encoder/layers' in n.name]
        tmp = f + '.q8'
        quantize_dynamic(f, tmp, weight_type=QuantType.QInt8, nodes_to_quantize=nodes)
        os.replace(tmp, f)
        print(f'{os.path.basename(f)}: {len(nodes)} encoder layers in 8-bit, {os.path.getsize(f) / 1e6:.1f} MB')


if __name__ == '__main__':
    main()
