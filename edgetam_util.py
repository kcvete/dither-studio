"""Model fixes that both the CoreML exporter and the server need.

Kept out of both so the two can never drift: an exported graph and the torch
model it falls back to have to agree about the input resolution.
"""

TRACK_SIZES = (512, 768, 1024)


def hydra_overrides(image_size):
    return [] if image_size == 1024 else ["++model.image_size=%d" % int(image_size)]


def set_image_size(model, image_size):
    """Finish a non-1024 build.

    `RoPEAttentionv2` (the memory cross-attention) computes its query
    frequencies once in `__init__` from `q_sizes` in the yaml — a 64x64 grid,
    i.e. 1024 px — and has no runtime fallback, so a 768 or 512 model would
    index a 4096-row table with 2304 or 1024 queries. (The self-attention
    `RoPEAttention` does recompute when the length changes, which is why only
    this one needs fixing.)
    """
    if int(image_size) == 1024:
        return model
    g = int(image_size) // 16
    for layer in model.memory_attention.layers:
        ca = layer.cross_attn_image
        if hasattr(ca, "freqs_cis_q"):
            ca.freqs_cis_q = ca.compute_cis(end_x=g, end_y=g)
            ca.__dict__.pop("_rope_real_cache", None)
    return model
