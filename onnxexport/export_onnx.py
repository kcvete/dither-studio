#!/usr/bin/env python3
"""Export the four EdgeTAM stages to ONNX for onnxruntime-web.

    PYTHONPATH=<pylibs> env/venv/bin/python onnx/export_onnx.py --image-size 768

Writes `<out>/`:

    encoder.onnx      image                       -> f0, f1, f2
    memattn.onnx      curr, memory, pos, mask     -> memory-conditioned features
    heads.onnx        pix_feat, f0, f1, prompts   -> 4 masks, ious, ptrs, score
    memenc.onnx       pix_feat, low-res mask      -> 512 memory latents + pos
    consts.bin        the tensors the JS loop needs outside any graph
    manifest.json     shapes, offsets, and the fp32-vs-fp16 parity numbers

Differences from the CoreML split (`coreml/export.py`), all forced by the
browser having no PyTorch to fall back to:

* the memory attention is exported once at the steady-state memory length with
  an additive key mask, instead of only at that length with a torch fallback
  for the cold-start frames;
* the SAM heads, which the CoreML build left on MPS, are a graph here;
* the memory encoder swallows the low-res -> 768px mask upsample, so the only
  mask that crosses into JS is 192x192.
"""
import argparse
import json
import os
import struct
import sys
import time
import warnings

warnings.filterwarnings('ignore')
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DV = ROOT
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'server'))
sys.path.insert(0, os.path.join(DV, 'env', 'EdgeTAM'))

import numpy as np                      # noqa: E402
import torch                            # noqa: E402

from onnxexport.wrappers_onnx import MaskedMemAttn, HeadsGraph, MemEncPlus  # noqa: E402
from coreml.wrappers import EncoderGraph                              # noqa: E402


def build_model(image_size):
    from sam2.build_sam import build_sam2_video_predictor
    import edgetam_util
    ckpt = os.path.join(DV, 'env', 'EdgeTAM', 'checkpoints', 'edgetam.pt')
    m = build_sam2_video_predictor(
        'configs/edgetam.yaml', ckpt, device=torch.device('cpu'),
        hydra_overrides_extra=edgetam_util.hydra_overrides(image_size))
    edgetam_util.set_image_size(m, image_size)
    return m.eval()


def export(mod, args, path, names_in, names_out, opset=17):
    t0 = time.perf_counter()
    with torch.no_grad():
        torch.onnx.export(mod, args, path, opset_version=opset,
                          input_names=names_in, output_names=names_out,
                          do_constant_folding=True, dynamo=False)
    mb = os.path.getsize(path) / 1e6
    print(f'[onnx] {os.path.basename(path)}: {mb:.1f} MB in '
          f'{time.perf_counter() - t0:.1f}s', flush=True)
    return mb


def to_fp16(src, dst, keep_io_fp32=False, block=()):
    from onnxruntime.transformers.float16 import convert_float_to_float16
    import onnx
    m = onnx.load(src)
    # shape inference must stay on: it is what tells the converter that
    # Resize's `scales` input is not a float32 activation to be demoted
    m16 = convert_float_to_float16(m, keep_io_types=keep_io_fp32,
                                   op_block_list=list(block) or None)
    onnx.save(m16, dst)
    return os.path.getsize(dst) / 1e6


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--image-size', type=int, default=768)
    ap.add_argument('--out', default=os.path.join(ROOT, 'web', 'models'))
    ap.add_argument('--fp16', action='store_true', default=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    m = build_model(a.image_size)
    S = m.image_size
    G = S // 16
    TOK = G * G
    NSPAT = m.num_maskmem                      # 7
    NPTR = m.max_obj_ptrs_in_encoder * (m.hidden_dim // m.mem_dim)   # 64
    MEMLEN = NSPAT * 512 + NPTR
    man = {'image_size': S, 'grid': G, 'tokens': TOK, 'nspat': NSPAT,
           'nptr': NPTR, 'memlen': MEMLEN, 'hidden': m.hidden_dim,
           'mem_dim': m.mem_dim, 'ptr_tokens': m.hidden_dim // m.mem_dim,
           'max_obj_ptrs': m.max_obj_ptrs_in_encoder,
           'sizes_mb': {}, 'checks': {}}
    print(f'[onnx] size={S} grid={G} tok={TOK} memlen={MEMLEN}', flush=True)

    # ------------------------------------------------------------- encoder
    enc = EncoderGraph(m).eval()
    img = torch.randn(1, 3, S, S)
    with torch.no_grad():
        ref_enc = enc(img)
    p = os.path.join(a.out, 'encoder.onnx')
    man['sizes_mb']['encoder_fp32'] = export(enc, (img,), p, ['image'],
                                             ['f0', 'f1', 'f2'])

    # -------------------------------------------------------- memory attn
    pe = m.image_encoder.neck.position_encoding
    with torch.no_grad():
        cpos_bchw = pe(torch.zeros(1, m.hidden_dim, G, G))
    cpos = cpos_bchw.flatten(2).permute(2, 0, 1).contiguous()   # [HW,1,C]
    ma = MaskedMemAttn(m.memory_attention, NSPAT, NPTR, G, cpos).eval()
    torch.manual_seed(0)
    feat = torch.randn(1, m.hidden_dim, G, G)
    curr = feat.reshape(1, m.hidden_dim, TOK).permute(2, 0, 1)
    mem = torch.randn(MEMLEN, 1, m.mem_dim)
    mpos = torch.randn(MEMLEN, 1, m.mem_dim)
    mmask = torch.zeros(1, 1, 1, MEMLEN)
    # parity of the rewrite against stock MemoryAttention at full memory
    from coreml.wrappers import RefMemAttn
    with torch.no_grad():
        yr = RefMemAttn(m.memory_attention, NSPAT, NPTR)(curr, mem, cpos, mpos)
        yr = yr.permute(1, 2, 0).reshape(1, m.hidden_dim, G, G)
        ys = ma(feat, mem, mpos, mmask)
    man['checks']['memattn_rewrite_max_abs'] = float((yr - ys).abs().max())
    man['checks']['memattn_ref_std'] = float(yr.std())
    print('[onnx] memattn rewrite max_abs=%.3e (std %.3f)'
          % (man['checks']['memattn_rewrite_max_abs'],
             man['checks']['memattn_ref_std']), flush=True)
    p = os.path.join(a.out, 'memattn.onnx')
    man['sizes_mb']['memattn_fp32'] = export(
        ma, (feat, mem, mpos, mmask), p,
        ['feat', 'memory', 'memory_pos', 'mem_mask'], ['out'])

    # -------------------------------------------------------------- heads
    hd = HeadsGraph(m).eval()
    pf = torch.randn(1, m.hidden_dim, G, G)
    f0 = torch.randn(1, 32, G * 4, G * 4)
    f1 = torch.randn(1, 64, G * 2, G * 2)
    hnames = ['pix_feat', 'f0', 'f1', 'point_coords', 'point_labels']
    honames = ['masks', 'ious', 'obj_ptrs', 'object_score_logits']
    # Two variants, because the prompt encoder's token count is part of the
    # decoder's sequence length: a tracking frame carries the single label -1
    # placeholder `_forward_sam_heads` invents, a prompt frame carries the
    # user's points. `heads.onnx` is the one that runs 150 times.
    p = os.path.join(a.out, 'heads.onnx')
    man['sizes_mb']['heads_fp32'] = export(
        hd, (pf, f0, f1, torch.zeros(1, 1, 2), -torch.ones(1, 1)), p,
        hnames, honames)
    p = os.path.join(a.out, 'heads_prompt.onnx')
    pc = torch.tensor([[[100., 100.], [400., 400.], [250., 250.]]])
    pl = torch.tensor([[2., 3., 1.]])
    with torch.no_grad():
        torch.onnx.export(hd, (pf, f0, f1, pc, pl, torch.ones(1)), p,
                          opset_version=17,
                          input_names=hnames + ['add_no_mem'],
                          output_names=honames,
                          dynamic_axes={'point_coords': {1: 'P'},
                                        'point_labels': {1: 'P'}},
                          do_constant_folding=True, dynamo=False)
    man['sizes_mb']['heads_prompt_fp32'] = os.path.getsize(p) / 1e6
    print('[onnx] heads_prompt.onnx: %.1f MB (dynamic point count)'
          % man['sizes_mb']['heads_prompt_fp32'], flush=True)

    # ------------------------------------------------------- memory encoder
    me = MemEncPlus(m).eval()
    lr = torch.randn(1, 1, G * 4, G * 4)
    io = torch.ones(1, 1)
    p = os.path.join(a.out, 'memenc.onnx')
    man['sizes_mb']['memenc_fp32'] = export(
        me, (pf, lr, io), p, ['pix_feat', 'low_res_mask', 'is_obj'],
        ['lat', 'lpos'])

    # The memory encoder computes in fp32 — its latents *are* the memory bank,
    # and fp16 there was measured to put 4x the signal into them as error — but
    # in the fp16 build its `pix_feat` arrives as the encoder's fp16 GPU buffer.
    # This variant takes that dtype and widens on the first op, so the encoder
    # output never has to come back to JS to be converted.
    class MemEncF16In(torch.nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.inner = inner

        def forward(self, pix_feat, low_res_mask):
            return self.inner(pix_feat.float(), low_res_mask, None)

    p = os.path.join(a.out, 'memenc.f16in.onnx')
    man['sizes_mb']['memenc_f16in'] = export(
        MemEncF16In(me).eval(), (pf.half(), lr), p,
        ['pix_feat', 'low_res_mask'], ['lat', 'lpos'])

    # ------------------------------------------------------------ fp16 pass
    # the memory encoder stays fp32: its latents *are* the memory bank, and the
    # CoreML measurements put an fp16 error of 4x the signal into them.
    if a.fp16:
        for name in ('encoder', 'memattn', 'heads', 'heads_prompt'):
            src = os.path.join(a.out, f'{name}.onnx')
            dst = os.path.join(a.out, f'{name}.fp16.onnx')
            try:
                # Resize's `scales`/`sizes` inputs are float32 by spec; the
                # converter casts them anyway and the graph fails to load.
                man['sizes_mb'][f'{name}_fp16'] = to_fp16(src, dst)
                print(f'[onnx] {name}.fp16.onnx: '
                      f'{man["sizes_mb"][f"{name}_fp16"]:.1f} MB', flush=True)
            except Exception as e:
                print(f'[onnx] fp16 {name} failed: {e}', flush=True)

    # ----------------------------------------------------------- constants
    # everything the JS tracking loop needs that is not inside a graph
    # `curr_pos` is a buffer inside memattn.onnx; what is left is the two
    # tensors the JS bookkeeping touches directly.
    blobs = {
        'no_mem_embed': m.no_mem_embed.detach().reshape(-1),    # [256]
        'maskmem_tpos_enc': m.maskmem_tpos_enc.detach(),        # [7,1,1,64]
    }
    off, entries = 0, {}
    buf = bytearray()
    for k, v in blobs.items():
        arr = v.detach().cpu().float().numpy().astype(np.float32).ravel()
        entries[k] = {'offset': off, 'count': int(arr.size),
                      'shape': list(v.shape)}
        buf += arr.tobytes()
        off += arr.nbytes
    with open(os.path.join(a.out, 'consts.bin'), 'wb') as f:
        f.write(bytes(buf))
    man['consts'] = entries
    man['consts_bytes'] = off
    man['sigmoid_scale_for_mem_enc'] = m.sigmoid_scale_for_mem_enc
    man['sigmoid_bias_for_mem_enc'] = m.sigmoid_bias_for_mem_enc
    man['mean'] = [0.485, 0.456, 0.406]
    man['std'] = [0.229, 0.224, 0.225]

    with open(os.path.join(a.out, 'manifest.json'), 'w') as f:
        json.dump(man, f, indent=2)
    print('[onnx] wrote ' + os.path.join(a.out, 'manifest.json'), flush=True)
    _ = struct, ref_enc


if __name__ == '__main__':
    main()
