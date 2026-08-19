#!/usr/bin/env python3
"""Convert the three hot EdgeTAM modules to CoreML.

    env/venv/bin/python coreml/export.py [--batch 1,2,3] [--image-size 1024]

Writes `env/coreml/<image_size>/`: one .mlpackage per (graph, batch size) plus a
manifest.json, and prints a numeric check of every graph against the PyTorch fp32
reference. One directory per tracking resolution the UI offers.

Precision and compute unit, per measurement on an M4 Pro (ms per call, static
steady-state shapes, `bench/micro_coreml.py`):

  encoder      fp16, CPU_AND_GPU  15.2 ms  (ANE 13.2, ALL 16.9; MPS fp16 ~40)
                                  ANE is 2 ms faster and measurably wrong: its
                                  fp16 accumulation put frame 147 of the
                                  reference clip at IoU 0.893 against torch,
                                  where the same graph on the GPU holds 0.958
                                  (whole-clip mean 0.993 -> 0.998)
  memory attn  fp16, CPU_AND_GPU  18.2 ms  (fp32 GPU 26.0, fp32 ALL 32.4,
                                            fp16 ANE 33.1; MPS fp16 ~43)
                                  fp16 max_abs vs fp32 torch = 5.5e-3, 1% of
                                  the output std -- the same order as the
                                  autocast path this replaces
  memory enc   fp32, CPU_AND_GPU   4.8 ms  (MPS fp16 ~18). fp16 *inputs*
                                  (its pix_feat is the encoder's fp16 output
                                  anyway) but fp32 compute: its
                                  latents *are* the memory bank, and fp16 here
                                  puts an error of 4x the signal into them.
"""
import argparse
import json
import os
import sys
import time
import warnings

warnings.filterwarnings('ignore')
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'env', 'EdgeTAM'))

import numpy as np
import torch


def patch_coremltools():
    """coremltools 9.0's torch frontend does `dtype(x.val)` on a 1-element numpy
    array; numpy >= 2 dropped that coercion, so every conversion fails with
    `TypeError: only 0-dimensional arrays can be converted to Python scalars`.
    Upstream's fix is to pin numpy < 2, which this project cannot do (the app
    needs 2.x), so patch the one function instead."""
    from coremltools.converters.mil.frontend.torch import ops as T
    from coremltools.converters.mil.mil import Builder as mb

    def _cast(context, node, dtype, dtype_name):
        inputs = T._get_inputs(context, node, expected=1)
        x = inputs[0]
        if not (len(x.shape) == 0 or np.all([d == 1 for d in x.shape])):
            raise ValueError('input to cast must be either a scalar or a length 1 tensor')
        if x.can_be_folded_to_const():
            v = x.val
            scalar = np.ravel(np.asarray(v))[0] if np.asarray(v).ndim else v
            res = x if isinstance(v, dtype) else mb.const(val=dtype(scalar), name=node.name)
        elif len(x.shape) > 0:
            sq = mb.squeeze(x=x, name=node.name + '_item')
            res = mb.cast(x=sq, dtype=dtype_name, name=node.name)
        else:
            res = mb.cast(x=x, dtype=dtype_name, name=node.name)
        context.add(res, node.name)

    T._cast = _cast
    for name in ('_int', '_float', '_bool', '_complex'):
        fn = getattr(T, name, None)
        if fn is not None and hasattr(fn, '__globals__'):
            fn.__globals__['_cast'] = _cast
    return True


def build_model(image_size=1024):
    from sam2.build_sam import build_sam2_video_predictor
    ckpt = os.path.join(ROOT, 'env', 'EdgeTAM', 'checkpoints', 'edgetam.pt')
    import edgetam_util
    m = build_sam2_video_predictor(
        'configs/edgetam.yaml', ckpt, device=torch.device('cpu'),
        hydra_overrides_extra=edgetam_util.hydra_overrides(image_size))
    edgetam_util.set_image_size(m, image_size)
    return m.eval()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--batch', default='1,2,3', help='object counts to export graphs for')
    ap.add_argument('--out', default=os.path.join(ROOT, 'env', 'coreml'))
    ap.add_argument('--image-size', type=int, default=1024,
                    help='tracker input resolution: 1024 | 768 | 512')
    ap.add_argument('--skip-encoder', action='store_true')
    args = ap.parse_args()
    batches = sorted({int(b) for b in args.batch.split(',') if b.strip()})
    args.out = os.path.join(args.out, str(args.image_size))
    os.makedirs(args.out, exist_ok=True)

    patch_coremltools()
    import coremltools as ct
    # ROOT is on sys.path from the top of this file, so this import works both
    # as `python coreml/export.py` and as `from coreml import export`
    from coreml.wrappers import StaticMemAttn, RefMemAttn, EncoderGraph, MemEncGraph

    m = build_model(args.image_size)
    NSPAT = m.num_maskmem
    NPTR = m.max_obj_ptrs_in_encoder * (m.hidden_dim // m.mem_dim)
    MEMLEN = NSPAT * 512 + NPTR
    TOK = (m.image_size // 16) ** 2          # 4096 tokens at 1024px / stride 16
    man = {'nspat': NSPAT, 'nptr': NPTR, 'memlen': MEMLEN, 'tokens': TOK,
           'hidden': m.hidden_dim, 'mem_dim': m.mem_dim, 'image_size': m.image_size,
           'batches': batches, 'graphs': {}, 'checks': {}}
    print(f'[export] size={m.image_size} nspat={NSPAT} nptr={NPTR} '
          f'memlen={MEMLEN} tokens={TOK}')

    def convert(name, traced, inputs, outputs, precision, path):
        t0 = time.perf_counter()
        ml = ct.convert(traced, inputs=inputs, outputs=outputs,
                        minimum_deployment_target=ct.target.iOS17,
                        compute_precision=precision,
                        compute_units=ct.ComputeUnit.ALL, convert_to='mlprogram')
        ml.save(path)
        print(f'[export] {name}: converted in {time.perf_counter() - t0:.1f}s -> {path}')
        return ml

    # ---------------------------------------------------------------- encoder
    if not args.skip_encoder:
        w = EncoderGraph(m).eval()
        x = torch.randn(1, 3, m.image_size, m.image_size)
        with torch.no_grad():
            ref = w(x)
            tr = torch.jit.trace(w, x, check_trace=False)
        p = os.path.join(args.out, 'encoder.mlpackage')
        convert('encoder', tr,
                [ct.TensorType(name='image', shape=tuple(x.shape), dtype=np.float16)],
                [ct.TensorType(name='f0', dtype=np.float16),
                 ct.TensorType(name='f1', dtype=np.float16),
                 ct.TensorType(name='f2', dtype=np.float16)],
                ct.precision.FLOAT16, p)
        got = ct.models.MLModel(p, compute_units=ct.ComputeUnit.CPU_AND_GPU) \
            .predict({'image': x.numpy().astype(np.float16)})
        errs = [float(np.abs(got[k] - r.numpy()).max()) for k, r in
                zip(('f0', 'f1', 'f2'), ref)]
        stds = [float(r.std()) for r in ref]
        man['graphs']['encoder'] = {'path': 'encoder.mlpackage', 'units': 'CPU_AND_GPU',
                                    'precision': 'fp16'}
        man['checks']['encoder'] = {'max_abs': errs, 'ref_std': stds}
        print(f'[export] encoder check: max_abs={["%.3e" % e for e in errs]} '
              f'ref_std={["%.3f" % s for s in stds]}')

    # -------------------------------------------------- memory attn / mem enc
    for B in batches:
        ma = StaticMemAttn(m.memory_attention, NSPAT, NPTR, m.image_size // 16).eval()
        ref_mod = RefMemAttn(m.memory_attention, NSPAT, NPTR).eval()
        torch.manual_seed(0)
        curr = torch.randn(TOK, B, m.hidden_dim)
        cpos = torch.randn(TOK, B, m.hidden_dim)
        mem = torch.randn(MEMLEN, B, m.mem_dim)
        mpos = torch.randn(MEMLEN, B, m.mem_dim)
        with torch.no_grad():
            yr = ref_mod(curr, mem, cpos, mpos)
            ys = ma(curr, mem, cpos, mpos)
        d = float((yr - ys).abs().max())
        print(f'[export] memattn b={B} torch parity: max_abs={d:.3e} ref_std={float(yr.std()):.3f}')
        with torch.no_grad():
            tr = torch.jit.trace(ma, (curr, mem, cpos, mpos), check_trace=False)
        p = os.path.join(args.out, f'memattn-b{B}.mlpackage')
        convert(f'memattn b={B}', tr,
                [ct.TensorType(name='curr', shape=tuple(curr.shape), dtype=np.float16),
                 ct.TensorType(name='memory', shape=tuple(mem.shape), dtype=np.float16),
                 ct.TensorType(name='curr_pos', shape=tuple(cpos.shape), dtype=np.float16),
                 ct.TensorType(name='memory_pos', shape=tuple(mpos.shape), dtype=np.float16)],
                [ct.TensorType(name='out', dtype=np.float16)], ct.precision.FLOAT16, p)
        got = ct.models.MLModel(p, compute_units=ct.ComputeUnit.CPU_AND_GPU).predict(
            {'curr': curr.numpy().astype(np.float16),
             'memory': mem.numpy().astype(np.float16),
             'curr_pos': cpos.numpy().astype(np.float16),
             'memory_pos': mpos.numpy().astype(np.float16)})['out']
        e = float(np.abs(got - yr.numpy()).max())
        man['graphs'][f'memattn-b{B}'] = {'path': f'memattn-b{B}.mlpackage',
                                          'units': 'CPU_AND_GPU', 'precision': 'fp16'}
        man['checks'][f'memattn-b{B}'] = {'torch_rewrite_max_abs': d,
                                          'coreml_max_abs': e, 'ref_std': float(yr.std())}
        print(f'[export] memattn b={B} coreml check: max_abs={e:.3e}')

        me = MemEncGraph(m).eval()
        g = m.image_size // 16
        pf = torch.randn(B, m.hidden_dim, g, g)
        mk = torch.rand(B, 1, m.image_size, m.image_size)
        io = torch.ones(B, 1)
        with torch.no_grad():
            rl, rp = me(pf, mk, io)
            tr = torch.jit.trace(me, (pf, mk, io), check_trace=False)
        p = os.path.join(args.out, f'memenc-b{B}.mlpackage')
        convert(f'memenc b={B}', tr,
                [ct.TensorType(name='pix_feat', shape=tuple(pf.shape), dtype=np.float16),
                 ct.TensorType(name='mask', shape=tuple(mk.shape), dtype=np.float16),
                 ct.TensorType(name='is_obj', shape=tuple(io.shape), dtype=np.float16)],
                [ct.TensorType(name='lat'), ct.TensorType(name='lpos')],
                ct.precision.FLOAT32, p)
        got = ct.models.MLModel(p, compute_units=ct.ComputeUnit.CPU_AND_GPU).predict(
            {'pix_feat': pf.numpy().astype(np.float16),
             'mask': mk.numpy().astype(np.float16),
             'is_obj': io.numpy().astype(np.float16)})
        e = float(np.abs(got['lat'] - rl.numpy()).max())
        man['graphs'][f'memenc-b{B}'] = {'path': f'memenc-b{B}.mlpackage',
                                         'units': 'CPU_AND_GPU', 'precision': 'fp32'}
        man['checks'][f'memenc-b{B}'] = {'coreml_max_abs': e, 'ref_std': float(rl.std())}
        print(f'[export] memenc b={B} coreml check: max_abs={e:.3e} ref_std={float(rl.std()):.3f}')

    with open(os.path.join(args.out, 'manifest.json'), 'w') as f:
        json.dump(man, f, indent=2)
    print('[export] wrote ' + os.path.join(args.out, 'manifest.json'))


if __name__ == '__main__':
    main()
