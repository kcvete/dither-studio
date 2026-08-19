"""Torch-level speed-ups for the EdgeTAM video predictor on MPS -- measured,
and none of them shipped: on this machine each is worth a few percent at most,
inside the run-to-run thermal spread. Kept because 'we tried the obvious
PyTorch-side rewrites' is only a useful statement if the code is here to
re-run: `bench/bench.py --backend torch-fast | torch-pf4 | torch-pe`.

Everything here is a behaviour-preserving rewrite of code that is slow on
Metal specifically. Nothing changes the model's weights or its algorithm.

`install(predictor, ...)` is idempotent and returns the predictor.

What it does and why:

* **Real-arithmetic RoPE.** `sam2.modeling.position_encoding.apply_rotary_enc*`
  upcasts to fp32, builds a complex tensor with `view_as_complex`, multiplies
  by `torch.polar` constants and casts back. On MPS every one of those steps is
  a full copy of a (1, 1, 4096, 128) complex tensor and the complex multiply is
  not fused. The identical rotation in real arithmetic — `a*cos - b*sin`,
  `a*sin + b*cos` — stays in the autocast dtype and touches half the memory.
  The cos/sin tables are cached per (module, grid) on the device.
* **No in-place key rotation.** Both RoPE attentions write the rotated keys
  back with `k[:, :, :n] = ...`, which on MPS is a scatter into a fresh tensor;
  a `torch.cat` of the rotated and unrotated halves is cheaper.
"""
import math

import torch
import torch.nn.functional as F

from sam2.modeling.position_encoding import compute_axial_cis
from sam2.modeling.sam.transformer import RoPEAttention, RoPEAttentionv2


# ------------------------------------------------------------ cos/sin cache
def _cos_sin(freqs_cis, device, dtype):
    r = torch.view_as_real(freqs_cis)
    return (r[..., 0].contiguous().to(device=device, dtype=dtype),
            r[..., 1].contiguous().to(device=device, dtype=dtype))


def _tables(mod, key, freqs_cis, device, dtype):
    cache = mod.__dict__.setdefault("_rope_real_cache", {})
    ck = (key, str(device), str(dtype), tuple(freqs_cis.shape))
    t = cache.get(ck)
    if t is None:
        t = _cos_sin(freqs_cis, device, dtype)
        cache[ck] = t
    return t


def _rot(x, cos, sin):
    """Rotary embedding in real arithmetic. x: (B, H, N, C), cos/sin: (N, C/2)."""
    B, H, N, C = x.shape
    xr = x.reshape(B, H, N, C // 2, 2)
    a, b = xr[..., 0], xr[..., 1]
    c = cos.reshape(1, 1, -1, C // 2)
    s = sin.reshape(1, 1, -1, C // 2)
    return torch.stack((a * c - b * s, a * s + b * c), dim=-1).reshape(B, H, N, C)


# --------------------------------------------------------------- attentions
def _rope_attention_forward(self, q, k, v, num_k_exclude_rope: int = 0):
    """RoPEAttention (memory self-attention), real-arithmetic RoPE."""
    q = self._separate_heads(self.q_proj(q), self.num_heads)
    k = self._separate_heads(self.k_proj(k), self.num_heads)
    v = self._separate_heads(self.v_proj(v), self.num_heads)

    n = q.shape[-2]
    if self.freqs_cis.shape[0] != n:
        g = int(math.sqrt(n))
        self.freqs_cis = self.compute_cis(end_x=g, end_y=g)
        self.__dict__.pop("_rope_real_cache", None)
    cos, sin = _tables(self, "sa", self.freqs_cis, q.device, q.dtype)

    q = _rot(q, cos, sin)
    n_rope = k.shape[-2] - num_k_exclude_rope
    if n_rope == k.shape[-2]:
        k = _rot(k, cos, sin)
    elif n_rope > 0:
        k = torch.cat((_rot(k[:, :, :n_rope], cos, sin), k[:, :, n_rope:]), dim=2)

    out = F.scaled_dot_product_attention(q, k, v)
    return self.out_proj(self._recombine_heads(out))


def _rope_attention_v2_forward(self, q, k, v, num_k_exclude_rope: int = 0,
                               rope_k_repeat: int = -1):
    """RoPEAttentionv2 (memory cross-attention), real-arithmetic RoPE.

    Key layout: `rope_k_repeat` spatial blocks, each of which is
    (unrotated 1-D perceiver latents .. rotated 2-D latents), followed by
    `num_k_exclude_rope` object-pointer tokens which are never rotated.
    """
    q = self._separate_heads(self.q_proj(q), self.num_heads)
    k = self._separate_heads(self.k_proj(k), self.num_heads)
    v = self._separate_heads(self.v_proj(v), self.num_heads)

    qcos, qsin = _tables(self, "q", self.freqs_cis_q, q.device, q.dtype)
    if qcos.shape[0] != q.shape[-2]:
        # image_size changed under us; rebuild for this grid
        g = int(math.sqrt(q.shape[-2]))
        self.freqs_cis_q = self.compute_cis(end_x=g, end_y=g)
        self.__dict__.pop("_rope_real_cache", None)
        qcos, qsin = _tables(self, "q", self.freqs_cis_q, q.device, q.dtype)
    q = _rot(q, qcos, qsin)

    R = rope_k_repeat
    n_rope = k.shape[-2] - num_k_exclude_rope
    if R > 0 and n_rope > 0:
        B, H, N, C = k.shape
        blk = n_rope // R                      # tokens per spatial memory frame
        rope_tok = self.freqs_cis_k.shape[0]   # rotated tail of each block
        no_rope = blk - rope_tok
        kcos, ksin = _tables(self, "k", self.freqs_cis_k, k.device, k.dtype)
        kb = k[:, :, :n_rope].reshape(B, H, R, blk, C)
        if no_rope > 0:
            k_no = kb[:, :, :, :no_rope, :]
            k_ro = kb[:, :, :, no_rope:, :].reshape(B, H, -1, C)
        else:
            k_no = None
            k_ro = kb.reshape(B, H, -1, C)
        k_ro = _rot(k_ro, kcos.repeat(R, 1), ksin.repeat(R, 1))
        k_ro = k_ro.reshape(B, H, R, rope_tok, C)
        kb = k_ro if k_no is None else torch.cat((k_no, k_ro), dim=3)
        kb = kb.reshape(B, H, n_rope, C)
        k = kb if n_rope == N else torch.cat((kb, k[:, :, n_rope:]), dim=2)

    out = F.scaled_dot_product_attention(q, k, v)
    return self.out_proj(self._recombine_heads(out))


# ------------------------------------------------------------------ install
def install(predictor, verbose=False):
    if getattr(predictor, "_fastpath", False):
        return predictor
    n = 0
    for m in predictor.modules():
        if isinstance(m, RoPEAttentionv2):
            m.forward = _rope_attention_v2_forward.__get__(m, type(m))
            n += 1
        elif isinstance(m, RoPEAttention):
            m.forward = _rope_attention_forward.__get__(m, type(m))
            n += 1
    predictor._fastpath = True
    if verbose:
        print("[fastpath] real-arithmetic RoPE on %d attention modules" % n, flush=True)
    return predictor


# ------------------------------------------------- position-encoding copies
def install_posenc_expand():
    """`PositionEmbeddingSine.forward` caches the encoding but then returns
    `cache[key][None].repeat(B, 1, 1, 1)` — a real copy. At 1024 the FPN's
    three levels are 67 + 17 + 4 MB of fp32, re-materialised on every frame,
    and the two high-resolution ones are never read by anything. `expand` is
    the same tensor without the copy. Applied to the class, once."""
    from sam2.modeling.position_encoding import PositionEmbeddingSine
    if getattr(PositionEmbeddingSine, "_dv_expand", False):
        return
    orig = PositionEmbeddingSine.forward

    def forward(self, x):
        key = (x.shape[-2], x.shape[-1])
        if key in self.cache:
            return self.cache[key][None].expand(x.shape[0], -1, -1, -1)
        out = orig(self, x)
        return self.cache[key][None].expand(x.shape[0], -1, -1, -1)

    PositionEmbeddingSine.forward = forward
    PositionEmbeddingSine._dv_expand = True


# --------------------------------------------------------- encoder prefetch
def install_prefetch(predictor, k=4, verbose=False):
    """Encode k frames per image-encoder call instead of one.

    The image encoder has no dependency on the memory bank — it is a plain
    per-frame function — but `_get_image_feature` runs it one frame at a time
    and keeps a one-entry cache, so the GPU sees a batch of 1 for the single
    most expensive module in the model. Batching is numerically identical and
    just fills the machine better.

    Direction is inferred from the request order so a backward propagate
    prefetches backwards.
    """
    if getattr(predictor, "_prefetch_k", 0) == k:
        return predictor
    predictor._prefetch_k = k
    predictor._prefetch_last = -1

    def _get_image_feature(state, frame_idx, batch_size):
        cache = state["cached_features"]
        hit = cache.get(frame_idx, (None, None))
        if hit[1] is None:
            n = state["num_frames"]
            step = -1 if frame_idx < predictor._prefetch_last else 1
            idxs = [frame_idx + step * i for i in range(k)]
            idxs = [i for i in idxs if 0 <= i < n]
            dev = state["device"]
            imgs = torch.stack([state["images"][i].to(dev, non_blocking=True).float()
                                for i in idxs])
            bo = predictor.forward_image(imgs)
            fresh = {}
            for j, i in enumerate(idxs):
                fresh[i] = (imgs[j:j + 1], {
                    "backbone_fpn": [f[j:j + 1] for f in bo["backbone_fpn"]],
                    "vision_pos_enc": [p[j:j + 1] for p in bo["vision_pos_enc"]],
                })
            state["cached_features"] = cache = fresh
            hit = cache[frame_idx]
        predictor._prefetch_last = frame_idx
        image, backbone_out = hit

        expanded_image = image.expand(batch_size, -1, -1, -1)
        expanded = {
            "backbone_fpn": [f.expand(batch_size, -1, -1, -1)
                             for f in backbone_out["backbone_fpn"]],
            "vision_pos_enc": [p.expand(batch_size, -1, -1, -1)
                               for p in backbone_out["vision_pos_enc"]],
        }
        features = predictor._prepare_backbone_features(expanded)
        return (expanded_image,) + features

    predictor._get_image_feature = _get_image_feature
    if verbose:
        print("[fastpath] image-encoder prefetch k=%d" % k, flush=True)
    return predictor
