"""CoreML acceleration for EdgeTAM video tracking.

`install(predictor)` swaps three module calls on a live SAM2 video predictor for
CoreML graphs and leaves everything else — the memory-bank bookkeeping, the SAM
heads, the whole propagate loop — exactly as upstream wrote it. Each shim checks
that the tensors match an exported static shape and falls back to the PyTorch
module when they don't, which is what makes the cold-start frames (where the
memory bank is still filling up) and unexported batch sizes just work.

Split, and why (per-call ms on an M4 Pro, `bench/micro_coreml.py`):

    image encoder    CoreML fp16, GPU   15.2   vs ~40 on MPS
    memory attention CoreML fp16, GPU   18.2   vs ~43 on MPS
    memory encoder   CoreML fp32, GPU    4.8   vs ~18 on MPS
    SAM heads        left on MPS         ~9    (11% of runtime, fiddliest to export)

Two things beyond "call CoreML instead":

* **fp16 tensors on the wire.** The graphs declare fp16 inputs and outputs, so
  the per-frame numpy round-trip is half the bytes it would be at fp32 — the
  encoder alone hands back 4.2 M values per frame.
* **The encoder runs a frame ahead, on a worker thread.** It is a pure function
  of the frame, so it can be computed before the rest of the step needs it;
  coremltools releases the GIL for part of `predict`, so some of it overlaps.
  The numpy -> MPS copy stays on the calling thread: two threads driving the
  Metal queue trips an `IOGPUMetalCommandBuffer` assertion.
"""
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
COREML_ROOT = os.path.join(ROOT, 'env', 'coreml')
DEFAULT_DIR = os.path.join(COREML_ROOT, '1024')


def dir_for(image_size):
    """One export directory per tracking resolution."""
    return os.path.join(COREML_ROOT, str(image_size))

_UNITS = {}


def _cu(name):
    import coremltools as ct
    if not _UNITS:
        _UNITS.update({'ALL': ct.ComputeUnit.ALL,
                       'CPU_AND_NE': ct.ComputeUnit.CPU_AND_NE,
                       'CPU_AND_GPU': ct.ComputeUnit.CPU_AND_GPU,
                       'CPU_ONLY': ct.ComputeUnit.CPU_ONLY})
    return _UNITS[name]


def available(d=DEFAULT_DIR):
    return os.path.exists(os.path.join(d, 'manifest.json'))


class Accel:
    """Loads the exported graphs and counts what actually ran where."""

    def __init__(self, d=DEFAULT_DIR):
        self.dir = d
        with open(os.path.join(d, 'manifest.json')) as f:
            self.man = json.load(f)
        self._cache = {}
        self._lock = threading.Lock()
        self.stats = {'enc_cml': 0, 'enc_torch': 0, 'mem_cml': 0, 'mem_torch': 0,
                      'me_cml': 0, 'me_torch': 0, 'load_s': 0.0}

    def model(self, key):
        with self._lock:
            if key in self._cache:
                return self._cache[key]
            g = self.man['graphs'].get(key)
            if g is None:
                self._cache[key] = None
                return None
            import coremltools as ct
            t = time.perf_counter()
            try:
                m = ct.models.MLModel(os.path.join(self.dir, g['path']),
                                      compute_units=_cu(g['units']))
            except Exception as e:   # a bad/absent package must not kill tracking
                print(f'[coreml] could not load {key}: {e}', flush=True)
                m = None
            self.stats['load_s'] += time.perf_counter() - t
            self._cache[key] = m
            return m

    def warm(self, batches=(1,)):
        """Compile everything up front so the first tracked frame isn't slow."""
        for k in ['encoder'] + [f'{n}-b{b}' for b in batches
                                for n in ('memattn', 'memenc')]:
            self.model(k)

    def summary(self):
        s = self.stats
        return {
            'encoder': f"{s['enc_cml']}/{s['enc_cml'] + s['enc_torch']} coreml",
            'memory_attention': f"{s['mem_cml']}/{s['mem_cml'] + s['mem_torch']} coreml",
            'memory_encoder': f"{s['me_cml']}/{s['me_cml'] + s['me_torch']} coreml",
            'model_load_s': round(s['load_s'], 2),
        }


def _np(t, dtype=np.float32):
    """GPU -> numpy in the dtype the graph wants.

    Casting on the device before the copy rather than after halves the bytes
    crossing the bus for every fp16 input, which at 1024 is the single largest
    non-compute cost in the loop."""
    td = torch.float16 if dtype == np.float16 else torch.float32
    return np.ascontiguousarray(t.detach().to('cpu', td).numpy())


def _in_dtype(m, name):
    """numpy dtype the compiled graph wants for input `name` (fp16 or fp32)."""
    try:
        for f in m.get_spec().description.input:
            if f.name == name:
                return np.float16 if f.type.multiArrayType.dataType == 65552 \
                    else np.float32
    except Exception:
        pass
    return np.float32


def install(predictor, accel=None, directory=DEFAULT_DIR, verbose=True,
            prefetch=True, stages=('encoder', 'memattn', 'memenc')):
    """Patch `predictor` in place. Returns the Accel (or None if unavailable)."""
    if getattr(predictor, '_coreml_accel', None) is not None:
        return predictor._coreml_accel
    if accel is None:
        if not available(directory):
            return None
        accel = Accel(directory)
    man = accel.man
    NSPAT, NPTR, TOK = man['nspat'], man['nptr'], man['tokens']
    MEMLEN, HID, MEMD = man['memlen'], man['hidden'], man['mem_dim']
    SIZE = man['image_size']
    dev = predictor.device
    tdtype = next(predictor.parameters()).dtype     # fp16 if the model is .half()

    # --------------------------------------------------------------- encoder
    orig_forward_image = predictor.forward_image
    orig_get_feat = predictor._get_image_feature
    pos_cache = {}

    def _pos_enc():
        """The FPN's position encodings are a pure function of the feature-map
        size, so build them once instead of shipping them out of the graph."""
        if 'pos' not in pos_cache:
            pe = predictor.image_encoder.neck.position_encoding
            with torch.no_grad():
                pos_cache['pos'] = [
                    pe(torch.zeros(1, HID, SIZE // s, SIZE // s, device=dev)).float()
                    for s in (4, 8, 16)]
        return pos_cache['pos']

    def _encode_np(img_np):
        """CoreML only — no torch, so this is safe to run off the main thread."""
        return accel.model('encoder').predict({'image': img_np})

    def _to_torch(o):
        return [torch.from_numpy(np.ascontiguousarray(o[k])).to(dev, tdtype)
                for k in ('f0', 'f1', 'f2')]

    def forward_image(img_batch):
        m = accel.model('encoder') if 'encoder' in stages else None
        ok = (m is not None and img_batch.shape[0] == 1
              and img_batch.shape[-1] == SIZE and img_batch.shape[-2] == SIZE)
        if not ok:
            accel.stats['enc_torch'] += 1
            return orig_forward_image(img_batch)
        accel.stats['enc_cml'] += 1
        fpn = _to_torch(_encode_np(_np(img_batch, _in_dtype(m, 'image'))))
        return {'vision_features': fpn[-1], 'vision_pos_enc': _pos_enc(),
                'backbone_fpn': fpn}

    predictor.forward_image = forward_image

    # ------------------------------------------------- one frame ahead, ANE
    if prefetch and 'encoder' in stages and accel.model('encoder') is not None:
        pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix='dv-enc')
        futures = {}
        last = {'idx': -1, 'dir': 1}
        idt = _in_dtype(accel.model('encoder'), 'image')

        def _frame_np(state, i):
            return _np(state['images'][i].unsqueeze(0), idt)

        def _get_image_feature(state, frame_idx, batch_size):
            cache = state['cached_features']
            key = (id(state), frame_idx)
            if frame_idx not in cache:
                # keyed by the inference state too: the predictor is a process
                # singleton, and a future left over from another clip would
                # otherwise be handed out for the same frame number
                fut = futures.pop(key, None)
                if fut is not None:
                    raw = fut.result()
                else:
                    m = accel.model('encoder')
                    if m is None or state['images'].shape[-1] != SIZE:
                        return orig_get_feat(state, frame_idx, batch_size)
                    raw = _encode_np(_frame_np(state, frame_idx))
                accel.stats['enc_cml'] += 1
                # the numpy -> MPS copy stays on this thread: two threads driving
                # the Metal command queue trips an IOGPUMetalCommandBuffer assert
                fpn = _to_torch(raw)
                cache.clear()
                cache[frame_idx] = (state['images'][frame_idx].unsqueeze(0),
                                    {'backbone_fpn': fpn,
                                     'vision_pos_enc': _pos_enc()})
            # queue the next frame in whatever direction we are moving.
            # The direction is latched rather than recomputed, because the
            # per-object tracking loop asks for the same frame once per subject
            # and a second look at the frame we are already on must not read as
            # "we turned around" -- that would throw away the prefetch we just
            # queued and make every reverse pass synchronous.
            if frame_idx != last['idx']:
                last['dir'] = -1 if frame_idx < last['idx'] else 1
                last['idx'] = frame_idx
            nxt = frame_idx + last['dir']
            nkey = (id(state), nxt)
            if (0 <= nxt < state['num_frames'] and nkey not in futures
                    and nxt not in cache):
                futures.clear()
                nxt_np = _frame_np(state, nxt)     # built here, on the main thread
                futures[nkey] = pool.submit(_encode_np, nxt_np)

            image, backbone_out = cache[frame_idx]
            expanded = {
                'backbone_fpn': [f.expand(batch_size, -1, -1, -1)
                                 for f in backbone_out['backbone_fpn']],
                'vision_pos_enc': [p.expand(batch_size, -1, -1, -1)
                                   for p in backbone_out['vision_pos_enc']],
            }
            return ((image.expand(batch_size, -1, -1, -1),)
                    + predictor._prepare_backbone_features(expanded))

        predictor._get_image_feature = _get_image_feature
        predictor._coreml_pool = pool

    # ---------------------------------------------------- memory attention
    ma_module = predictor.memory_attention
    orig_ma = ma_module.forward

    def memory_attention(curr, memory, curr_pos=None, memory_pos=None,
                         num_obj_ptr_tokens=0, num_spatial_mem=-1):
        c = curr[0] if isinstance(curr, list) else curr
        cp = curr_pos[0] if isinstance(curr_pos, list) else curr_pos
        B = c.shape[1]
        m = accel.model(f'memattn-b{B}') if 'memattn' in stages else None
        ok = (m is not None and num_spatial_mem == NSPAT and num_obj_ptr_tokens == NPTR
              and c.shape[0] == TOK and c.shape[2] == HID
              and memory.shape[0] == MEMLEN and memory.shape[2] == MEMD)
        if not ok:
            accel.stats['mem_torch'] += 1
            return orig_ma(curr, memory, curr_pos, memory_pos,
                           num_obj_ptr_tokens, num_spatial_mem)
        d = _in_dtype(m, 'curr')
        o = m.predict({'curr': _np(c, d), 'memory': _np(memory, d),
                       'curr_pos': _np(cp, d), 'memory_pos': _np(memory_pos, d)})['out']
        accel.stats['mem_cml'] += 1
        return torch.from_numpy(np.ascontiguousarray(o)).to(dev, tdtype)

    ma_module.forward = memory_attention

    # ------------------------------------------------------ memory encoder
    orig_encode = predictor._encode_new_memory

    def _encode_new_memory(current_vision_feats, feat_sizes, pred_masks_high_res,
                           object_score_logits, is_mask_from_pts):
        B = current_vision_feats[-1].size(1)
        m = accel.model(f'memenc-b{B}') if 'memenc' in stages else None
        H, W = feat_sizes[-1]
        ok = (m is not None and H == SIZE // 16 and W == SIZE // 16
              and pred_masks_high_res.shape[-1] == SIZE)
        if not ok:
            accel.stats['me_torch'] += 1
            return orig_encode(current_vision_feats, feat_sizes, pred_masks_high_res,
                               object_score_logits, is_mask_from_pts)
        pix_feat = current_vision_feats[-1].permute(1, 2, 0).reshape(B, HID, H, W)
        if predictor.non_overlap_masks_for_mem_enc and not predictor.training:
            pred_masks_high_res = predictor._apply_non_overlapping_constraints(
                pred_masks_high_res)
        if predictor.binarize_mask_from_pts_for_mem_enc and is_mask_from_pts \
                and not predictor.training:
            mask_for_mem = (pred_masks_high_res > 0).float()
        else:
            mask_for_mem = torch.sigmoid(pred_masks_high_res)
        if predictor.sigmoid_scale_for_mem_enc != 1.0:
            mask_for_mem = mask_for_mem * predictor.sigmoid_scale_for_mem_enc
        if predictor.sigmoid_bias_for_mem_enc != 0.0:
            mask_for_mem = mask_for_mem + predictor.sigmoid_bias_for_mem_enc
        is_obj = (object_score_logits > 0).float()
        d = _in_dtype(m, 'pix_feat')
        o = m.predict({'pix_feat': _np(pix_feat, d), 'mask': _np(mask_for_mem, d),
                       'is_obj': _np(is_obj, d)})
        accel.stats['me_cml'] += 1
        lat = torch.from_numpy(np.ascontiguousarray(o['lat'])).to(dev, tdtype)
        lpos = torch.from_numpy(np.ascontiguousarray(o['lpos'])).to(dev, tdtype)
        return lat, [lpos]

    predictor._encode_new_memory = _encode_new_memory

    predictor._coreml_accel = accel
    if verbose:
        print('[coreml] installed: encoder(fp16/GPU%s) memattn(fp16/GPU) '
              'memenc(fp32/GPU); torch fallback for cold frames'
              % (', 1 frame ahead' if prefetch else ''), flush=True)
    return accel


def uninstall(predictor):
    pool = getattr(predictor, '_coreml_pool', None)
    if pool is not None:
        pool.shutdown(wait=False)
        predictor._coreml_pool = None
