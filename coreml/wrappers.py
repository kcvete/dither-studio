"""Traceable, static-shape wrappers around the three EdgeTAM modules that
dominate tracking time, so coremltools can convert them.

Why wrappers at all:

* `MemoryAttention` applies rotary position embeddings with complex tensors
  (`torch.polar` / `view_as_complex`). MIL has no complex type, so conversion
  dies with `KeyError: np.int32(9)`. `StaticMemAttn` recomputes the identical
  rotation in real arithmetic (parity ~1e-6) and drops the in-place slice
  assignment into `k`, which traces as `index_put_`.
* The memory bank length varies for the first ~16 frames and is then constant at
  `num_maskmem * 512 + max_obj_ptrs * 4` tokens. These graphs are exported at
  that steady-state shape; the caller runs PyTorch until it is reached.
* The positional encodings coming out of the image encoder's neck are sinusoidal
  functions of the feature-map size only, so the encoder graph does not need to
  emit them — the runtime caches them from one warm-up pass.
"""
import torch
import torch.nn.functional as F


def cos_sin(freqs_cis):
    """complex freqs -> (cos, sin) real tensors"""
    r = torch.view_as_real(freqs_cis)
    return r[..., 0].contiguous(), r[..., 1].contiguous()


def rot(x, cos, sin):
    """Rotary embedding, real arithmetic. x: (B, H, N, C)."""
    B, H, N, C = x.shape
    xr = x.reshape(B, H, N, C // 2, 2)
    a, b = xr[..., 0], xr[..., 1]
    c = cos.reshape(1, 1, -1, C // 2)
    s = sin.reshape(1, 1, -1, C // 2)
    return torch.stack([a * c - b * s, a * s + b * c], dim=-1).reshape(B, H, N, C)


class StaticMemAttn(torch.nn.Module):
    """MemoryAttention at a fixed memory length, with real-arithmetic RoPE.

    Layout assumption, taken from `_prepare_memory_conditioned_features`: the
    memory is `nspat` spatial blocks of 512 latents followed by `nptr` object
    pointer tokens; inside each spatial block the first 256 latents are the 1-D
    perceiver latents (no rotation) and the last 256 are the 2-D ones (rotated).
    """

    def __init__(self, ma, nspat=7, nptr=64, sa_grid=64):
        super().__init__()
        self.ma = ma
        self.nspat, self.nptr, self.sa_grid = nspat, nptr, sa_grid
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

    def forward(self, curr, memory, curr_pos, memory_pos):
        NS = self.nspat
        out = curr + 0.1 * curr_pos
        out = out.transpose(0, 1)
        memory = memory.transpose(0, 1)
        memory_pos = memory_pos.transpose(0, 1)
        for i, l in enumerate(self.ma.layers):
            # ---- self attention (pos_enc_at_attn is False in edgetam.yaml)
            t2 = l.norm1(out)
            sa = l.self_attn
            q = self._heads(sa.q_proj(t2), sa.num_heads)
            k = self._heads(sa.k_proj(t2), sa.num_heads)
            v = self._heads(sa.v_proj(t2), sa.num_heads)
            q = rot(q, getattr(self, f'sa_cos_{i}'), getattr(self, f'sa_sin_{i}'))
            k = rot(k, getattr(self, f'sa_cos_{i}'), getattr(self, f'sa_sin_{i}'))
            out = out + sa.out_proj(self._unheads(F.scaled_dot_product_attention(q, k, v)))
            # ---- cross attention into the memory (queries plain, keys carry pos)
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
            out = out + ca.out_proj(self._unheads(F.scaled_dot_product_attention(q, k, v)))
            # ---- mlp
            out = out + l.linear2(l.activation(l.linear1(l.norm3(out))))
        return self.ma.norm(out).transpose(0, 1)


class EncoderGraph(torch.nn.Module):
    """image -> the three backbone_fpn levels, with the mask decoder's conv_s0 /
    conv_s1 projections already folded in (that is what forward_image does)."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, image):
        bo = self.model.image_encoder(image)
        f = bo['backbone_fpn']
        f0 = self.model.sam_mask_decoder.conv_s0(f[0])
        f1 = self.model.sam_mask_decoder.conv_s1(f[1])
        return f0, f1, f[2]


class MemEncGraph(torch.nn.Module):
    """the memory encoder, the no-object spatial embedding and the spatial
    perceiver as one graph — exactly `_encode_new_memory`'s tail."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, pix_feat, mask_for_mem, is_obj_appearing):
        m = self.model
        out = m.memory_encoder(pix_feat, mask_for_mem, skip_mask_sigmoid=True)
        feats = out['vision_features']
        pos = out['vision_pos_enc'][0]
        if m.no_obj_embed_spatial is not None:
            feats = feats + (1 - is_obj_appearing[..., None, None]) \
                * m.no_obj_embed_spatial[..., None, None].expand(*feats.shape)
        lat, lpos = m.spatial_perceiver(feats, pos)
        return lat, lpos


class RefMemAttn(torch.nn.Module):
    """stock MemoryAttention, for the parity check only"""

    def __init__(self, ma, nspat=7, nptr=64):
        super().__init__()
        self.ma, self.nspat, self.nptr = ma, nspat, nptr

    def forward(self, curr, memory, curr_pos, memory_pos):
        return self.ma(curr=curr, memory=memory, curr_pos=curr_pos, memory_pos=memory_pos,
                       num_obj_ptr_tokens=self.nptr, num_spatial_mem=self.nspat)
