"""ONNX-traceable wrappers for the four EdgeTAM stages.

Extends `coreml/wrappers.py`:

* `MaskedMemAttn` — the same real-arithmetic RoPE rewrite of `MemoryAttention`,
  but at a *fixed* memory length with an additive key mask, so the cold-start
  frames (memory bank not yet full) run on the same graph instead of needing a
  PyTorch fallback that a browser does not have.
* `HeadsGraph` — the SAM prompt encoder + mask decoder, which the CoreML split
  left in torch. Returns all four mask tokens so the caller picks single-mask
  (prompt frame) or best-of-three (tracking frames) without a second graph.
* `HeadsMaskPrompt` — the same stage for a *mask* prompt (lasso/polygon),
  which EdgeTAM answers with `_use_mask_as_output` rather than the decoder's
  own masks; the decoder still runs, for the object pointer only.
* `MemEncPlus` — `MemEncGraph` with the low-res -> full-res mask upsample and
  the sigmoid scale/bias folded in, so the 768x768 mask never crosses the
  JS/GPU boundary.
"""
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from coreml.wrappers import cos_sin, rot, EncoderGraph  # noqa: E402,F401

NO_OBJ_SCORE = -1024.0


def sparse_points(pe, coords, labels):
    """`PromptEncoder._embed_points(pad=True)` re-derived as arithmetic.

    Stock's boolean assignment (`point_embedding[labels == -1] = 0.0`) traces to
    NonZero + ScatterND, and NonZero is not in onnxruntime-web's WebGPU
    operator set — it would drop the decoder onto the CPU EP and make the graph
    dynamically shaped. The form below is numerically identical.
    """
    pad_c = torch.zeros(coords.shape[0], 1, 2, dtype=coords.dtype)
    pad_l = -torch.ones(labels.shape[0], 1, dtype=labels.dtype)
    pts = torch.cat([coords + 0.5, pad_c], dim=1)
    lab = torch.cat([labels, pad_l], dim=1)
    emb = pe.pe_layer.forward_with_coords(pts, pe.input_image_size)

    def m(k):
        return (lab == float(k)).to(emb.dtype).unsqueeze(-1)

    out = emb * (1.0 - m(-1)) + m(-1) * pe.not_a_point_embed.weight
    for k in range(4):
        out = out + m(k) * pe.point_embeddings[k].weight
    return out


class MaskedMemAttn(torch.nn.Module):
    """MemoryAttention at a fixed memory length with an additive cross-attention
    key mask.

    The memory is `nspat` blocks of 512 latents followed by `nptr` pointer
    tokens; unused slots are zero-filled and masked to -inf, which reproduces
    the shorter-memory attention exactly (softmax over the surviving keys).

    Takes and returns the encoder's own [1, C, H, W] layout rather than the
    (HW, B, C) one `MemoryAttention` uses internally, and carries the FPN
    position encoding as a buffer instead of an input. Both exist so the
    encoder's output tensor can be handed straight to this graph as a WebGPU
    buffer — a JS-side transpose would force a 2.4 MB readback per frame.
    """

    def __init__(self, ma, nspat=7, nptr=64, sa_grid=48, curr_pos=None):
        super().__init__()
        self.ma = ma
        self.nspat, self.nptr, self.sa_grid = nspat, nptr, sa_grid
        self.register_buffer('curr_pos', curr_pos)      # [HW,1,C]
        from sam2.modeling.position_encoding import compute_axial_cis
        for i, l in enumerate(ma.layers):
            sa = l.self_attn
            fq = compute_axial_cis(dim=sa.internal_dim // sa.num_heads,
                                   end_x=sa_grid, end_y=sa_grid, theta=10000.0)
            c, s = cos_sin(fq)
            self.register_buffer(f'sa_cos_{i}', c)
            self.register_buffer(f'sa_sin_{i}', s)
            ca = l.cross_attn_image
            cq, sq = cos_sin(ca.freqs_cis_q)
            ck, sk = cos_sin(ca.freqs_cis_k)
            self.register_buffer(f'caq_cos_{i}', cq)
            self.register_buffer(f'caq_sin_{i}', sq)
            self.register_buffer(f'cak_cos_{i}', ck.repeat(nspat, 1))
            self.register_buffer(f'cak_sin_{i}', sk.repeat(nspat, 1))

    @staticmethod
    def _heads(x, nh):
        b, n, c = x.shape
        return x.reshape(b, n, nh, c // nh).transpose(1, 2)

    @staticmethod
    def _unheads(x):
        b, h, n, c = x.shape
        return x.transpose(1, 2).reshape(b, n, h * c)

    def forward(self, feat, memory, memory_pos, mem_mask):
        NS = self.nspat
        fB, fC, fH, fW = feat.shape
        curr = feat.reshape(fB, fC, fH * fW).permute(2, 0, 1)  # [HW,B,C]
        out = curr + 0.1 * self.curr_pos
        out = out.transpose(0, 1)
        memory = memory.transpose(0, 1)
        memory_pos = memory_pos.transpose(0, 1)
        for i, l in enumerate(self.ma.layers):
            t2 = l.norm1(out)
            sa = l.self_attn
            q = self._heads(sa.q_proj(t2), sa.num_heads)
            k = self._heads(sa.k_proj(t2), sa.num_heads)
            v = self._heads(sa.v_proj(t2), sa.num_heads)
            q = rot(q, getattr(self, f'sa_cos_{i}'), getattr(self, f'sa_sin_{i}'))
            k = rot(k, getattr(self, f'sa_cos_{i}'), getattr(self, f'sa_sin_{i}'))
            out = out + sa.out_proj(self._unheads(F.scaled_dot_product_attention(q, k, v)))
            t2 = l.norm2(out)
            ca = l.cross_attn_image
            q = self._heads(ca.q_proj(t2), ca.num_heads)
            k = self._heads(ca.k_proj(memory + memory_pos), ca.num_heads)
            v = self._heads(ca.v_proj(memory), ca.num_heads)
            q = rot(q, getattr(self, f'caq_cos_{i}'), getattr(self, f'caq_sin_{i}'))
            B, H, N, C = k.shape
            kb = k[:, :, :NS * 512, :].reshape(B, H, NS, 512, C)
            k_no = kb[:, :, :, :256, :].reshape(B, H, -1, C)
            k_ro = kb[:, :, :, 256:, :].reshape(B, H, -1, C)
            k_ro = rot(k_ro, getattr(self, f'cak_cos_{i}'), getattr(self, f'cak_sin_{i}'))
            k_sp = torch.cat([k_no.reshape(B, H, NS, 256, C),
                              k_ro.reshape(B, H, NS, 256, C)], dim=3).reshape(B, H, NS * 512, C)
            k = torch.cat([k_sp, k[:, :, NS * 512:, :]], dim=2)
            att = F.scaled_dot_product_attention(q, k, v, attn_mask=mem_mask)
            out = out + ca.out_proj(self._unheads(att))
            out = out + l.linear2(l.activation(l.linear1(l.norm3(out))))
        out = self.ma.norm(out).transpose(0, 1)           # [HW,B,C]
        return out.permute(1, 2, 0).reshape(fB, fC, fH, fW)


class HeadsGraph(torch.nn.Module):
    """prompt encoder + mask decoder, all four mask tokens kept.

    `_forward_sam_heads` picks token 0 when the frame carries >1 prompt point
    and argmax-of-IoU over tokens 1..3 otherwise; both selections are one gather
    on four small tensors, so they are left to the caller and the graph stays
    single-variant.

    `_embed_points` is re-derived as arithmetic in `sparse_points` above; see
    its docstring for why.
    """

    def __init__(self, model):
        super().__init__()
        self.model = model
        self.register_buffer('image_pe',
                             model.sam_prompt_encoder.get_dense_pe().detach())

    def forward(self, pix_feat, f0, f1, point_coords, point_labels,
                add_no_mem=None):
        m = self.model
        pe = m.sam_prompt_encoder
        if add_no_mem is not None:
            # `directly_add_no_mem_embed`: an initial conditioning frame skips
            # memory attention entirely and just adds this vector. Folding it in
            # here keeps the encoder -> heads hop a pure GPU-buffer handoff on
            # the one frame that does not pass through memattn.
            pix_feat = pix_feat + add_no_mem * m.no_mem_embed.reshape(1, -1, 1, 1)
        sparse = sparse_points(pe, point_coords, point_labels)
        dense = pe.no_mask_embed.weight.reshape(1, -1, 1, 1).expand(
            point_coords.shape[0], -1, *pe.image_embedding_size)
        masks, iou, mask_tokens, osl = m.sam_mask_decoder.predict_masks(
            image_embeddings=pix_feat, image_pe=self.image_pe,
            sparse_prompt_embeddings=sparse, dense_prompt_embeddings=dense,
            repeat_image=False, high_res_features=[f0, f1])
        is_obj = (osl > 0).float()                      # [B,1]
        masks = torch.where(is_obj[:, :, None, None] > 0.5, masks,
                            torch.full_like(masks, NO_OBJ_SCORE))
        ptr = m.obj_ptr_proj(mask_tokens)               # [B,4,256]
        lam = is_obj[:, :, None]                        # soft_no_obj_ptr is False
        ptr = lam * ptr + (1 - lam) * m.no_obj_ptr      # fixed_no_obj_ptr is True
        return masks, iou, ptr, osl


class HeadsMaskPrompt(torch.nn.Module):
    """`_use_mask_as_output` as a graph: a lasso/polygon prompt frame.

    EdgeTAM's config sets `use_mask_input_as_output_without_sam: true`, so
    `add_new_mask` does *not* let the SAM decoder pick the mask. The mask the
    user drew becomes the output directly (as +/-10 logits, downsampled 4x),
    and the decoder is run only to produce the object pointer. This graph
    reproduces that, with the same output names/shapes as `HeadsGraph` so the
    caller's "pick token k" code is untouched.

    Output semantics (differs from `HeadsGraph`, where the four slots are four
    genuine candidates):

    * `masks` [1,4,192,192] - the SAME mask-derived logit map repeated into all
      four slots. There is only one answer here; the repeat exists so
      `masks[k]` works for any k.
    * `ious` [1,4] - all ones (stock returns a dummy IoU of 1).
    * `obj_ptrs` [1,4,256] - the SAME pointer repeated into all four slots. It
      comes from mask-decoder token 0 (`sam_output_tokens[:, 0]`, which is what
      `_forward_sam_heads(multimask_output=False)` uses) and carries *both*
      no-obj blends: the decoder's own `object_score_logits` blend inside
      `_forward_sam_heads`, then the mask-derived one in `_use_mask_as_output`.
    * `object_score_logits` [1,1] - `20*any(mask>0) - 10`, from the mask, not
      from the decoder.

    So the caller may use k=0 unconditionally.

    `mask_full` is [1,1,image_size,image_size]; anything in 0..1 (or 0..255)
    is accepted and thresholded at >= 0.5 in-graph, which is what
    `add_new_mask` does to a resized mask.

    Two things about the ONNX inputs. There is no `add_no_mem`: EdgeTAM's mask
    branch skips memory conditioning entirely, `no_mem_embed` included (see
    `forward`). And `f0`/`f1` are arguments here but are *pruned out of the
    exported graph*, because the object pointer comes from the transformer's
    output tokens and only `output_upscaling` — whose masks this graph throws
    away — ever touches the high-res features. The graph's real inputs are
    `pix_feat` and `mask_full`.

    The 4x downsample replaces `F.interpolate(..., antialias=True)`, which does
    not export. torch's antialiased bilinear at an exact 1/4 scale is a
    separable 8-tap triangle filter (taps at +/-0.5, 1.5, 2.5, 3.5 input pixels
    from the output centre, weights 1-|d|/4 normalised) with the kernel
    renormalised where it hangs off the edge. That is a fixed 8x8 conv with
    stride 4, pad 2, times a constant 192x192 reciprocal-normaliser -- exact,
    not an approximation (measured max-abs 9.5e-07 against torch on a real
    rasterised polygon, on logits of magnitude 10).
    """

    def __init__(self, model):
        super().__init__()
        self.model = model
        self.register_buffer('image_pe',
                             model.sam_prompt_encoder.get_dense_pe().detach())
        S = model.image_size
        n = S // 4
        d = torch.tensor([-3.5, -2.5, -1.5, -0.5, 0.5, 1.5, 2.5, 3.5])
        w = (1.0 - d.abs() / 4.0) / 4.0                  # sums to 1
        self.register_buffer('aa_k', torch.outer(w, w).reshape(1, 1, 8, 8))
        dn = torch.ones(n)
        dn[0] = w[2:].sum()                              # two taps clipped off
        dn[-1] = w[:6].sum()
        self.register_buffer('aa_norm',
                             (1.0 / torch.outer(dn, dn)).reshape(1, 1, n, n))
        # the empty point prompt `_forward_sam_heads` invents when
        # point_inputs is None, plus the prompt encoder's own padding point
        self.register_buffer('zero_coords', torch.zeros(1, 1, 2))
        self.register_buffer('neg_labels', -torch.ones(1, 1))

    def _down4(self, x):
        """antialiased 4x downsample, `F.interpolate(antialias=True)`-exact."""
        return F.conv2d(x, self.aa_k.to(x.dtype), stride=4, padding=2) \
            * self.aa_norm.to(x.dtype)

    def forward(self, pix_feat, f0, f1, mask_full):
        m = self.model
        pe = m.sam_prompt_encoder
        # NOTE: no `no_mem_embed` add here, unlike HeadsGraph. `_track_step`'s
        # mask branch never calls `_prepare_memory_conditioned_features` — it
        # hands `current_vision_feats[-1]` to `_use_mask_as_output` raw — so
        # `directly_add_no_mem_embed` does not apply to a mask prompt at all.
        # Adding it moves the object pointer by ~8e-2.

        # Binarise first, exactly as `add_new_mask` does after resizing a
        # clip-resolution mask to image_size (`(x >= 0.5).float()`). The page
        # rasterises the lasso onto a 768x768 canvas, and canvas fills are
        # antialiased, so the edge pixels arrive as fractions; this also makes
        # a 0/255 feed behave the same as a 0/1 one.
        mask_full = (mask_full >= 0.5).to(mask_full.dtype)

        # 1-2. the mask IS the output
        hi = mask_full * 20.0 - 10.0                     # [1,1,S,S]
        low = self._down4(hi)                            # [1,1,S/4,S/4]

        # 4. the decoder runs only for the pointer, with a dense mask embedding
        #    and no point prompt. `mask_downsample` is a learned Conv2d(1,1,4,4)
        #    on the model; `mask_downscaling` is the prompt encoder's separate
        #    stack. Both run, in that order -- and because mask_downsample
        #    already lands on the prompt encoder's `mask_input_size`, stock's
        #    "resize if it does not match" branch is a no-op here.
        dense = pe.mask_downscaling(m.mask_downsample(mask_full))
        sparse = sparse_points(pe, self.zero_coords.to(mask_full.dtype),
                               self.neg_labels.to(mask_full.dtype))
        _, _, mask_tokens, dec_osl = m.sam_mask_decoder.predict_masks(
            image_embeddings=pix_feat, image_pe=self.image_pe,
            sparse_prompt_embeddings=sparse, dense_prompt_embeddings=dense,
            repeat_image=False, high_res_features=[f0, f1])
        ptr = m.obj_ptr_proj(mask_tokens[:, 0])          # [1,256], token 0
        # blend #1: inside _forward_sam_heads, on the decoder's own score
        lam = (dec_osl > 0).to(ptr.dtype)                # [1,1]
        ptr = lam * ptr + (1 - lam) * m.no_obj_ptr
        # 5. blend #2: in _use_mask_as_output, on the mask's own emptiness
        any_pos = (torch.amax(mask_full.reshape(1, -1), dim=1, keepdim=True)
                   > 0).to(ptr.dtype)                    # [1,1]
        ptr = any_pos * ptr + (1 - any_pos) * m.no_obj_ptr
        osl = 20.0 * any_pos - 10.0                      # [1,1]

        masks = low.repeat(1, 4, 1, 1)
        ious = torch.ones_like(osl).repeat(1, 4)
        ptrs = ptr.reshape(1, 1, -1).repeat(1, 4, 1)
        return masks, ious, ptrs, osl


class MemEncPlus(torch.nn.Module):
    """memory encoder + no-object embedding + spatial perceiver, with the
    low-res -> image-res mask upsample and the sigmoid scale/bias folded in."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, pix_feat, low_res_mask, is_obj):
        m = self.model
        hi = F.interpolate(low_res_mask, size=(m.image_size, m.image_size),
                           mode='bilinear', align_corners=False)
        mask = torch.sigmoid(hi) * m.sigmoid_scale_for_mem_enc \
            + m.sigmoid_bias_for_mem_enc
        out = m.memory_encoder(pix_feat, mask, skip_mask_sigmoid=True)
        feats = out['vision_features']
        pos = out['vision_pos_enc'][0]
        if m.no_obj_embed_spatial is not None:
            feats = feats + (1 - is_obj[..., None, None]) \
                * m.no_obj_embed_spatial[..., None, None].expand(*feats.shape)
        lat, lpos = m.spatial_perceiver(feats, pos)
        return lat, lpos
