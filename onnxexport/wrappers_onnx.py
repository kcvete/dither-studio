"""ONNX-traceable wrappers for the four EdgeTAM stages.

Extends `coreml/wrappers.py`:

* `MaskedMemAttn` — the same real-arithmetic RoPE rewrite of `MemoryAttention`,
  but at a *fixed* memory length with an additive key mask, so the cold-start
  frames (memory bank not yet full) run on the same graph instead of needing a
  PyTorch fallback that a browser does not have.
* `HeadsGraph` — the SAM prompt encoder + mask decoder, which the CoreML split
  left in torch. Returns all four mask tokens so the caller picks single-mask
  (prompt frame) or best-of-three (tracking frames) without a second graph.
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

    `_embed_points` is re-derived here as arithmetic. Stock's boolean
    assignment (`point_embedding[labels == -1] = 0.0`) traces to
    NonZero + ScatterND, and NonZero is not in onnxruntime-web's WebGPU
    operator set — it would drop the decoder onto the CPU EP and make the
    graph dynamically shaped. The `where`-free form below is numerically
    identical.
    """

    def __init__(self, model):
        super().__init__()
        self.model = model
        self.register_buffer('image_pe',
                             model.sam_prompt_encoder.get_dense_pe().detach())

    def _sparse(self, coords, labels):
        pe = self.model.sam_prompt_encoder
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
        sparse = self._sparse(point_coords, point_labels)
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
