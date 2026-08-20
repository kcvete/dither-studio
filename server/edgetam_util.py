"""Model fixes that both the CoreML exporter and the server need, plus the
tracking loop the server runs when subjects were prompted on different frames.

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


# ------------------------------------------------------- per-object memory
#
# SAM2's `propagate_in_video` runs the whole batch of objects through ONE
# `output_dict`, and `propagate_in_video_preflight` consolidates *every* prompt
# frame across *every* object before the first frame is tracked. On a prompt
# frame F that belongs to object X, the objects that were not prompted there
# have no output yet, so `_consolidate_temp_output_across_obj` fills them with
# the `NO_OBJ_SCORE` placeholder -- and then runs the memory encoder over it.
# F is a *conditioning* frame, and `max_cond_frames_in_attn` is -1, so that
# "this object is not here" memory is attended to on every single frame of the
# clip, for objects that were never prompted on F. One subject prompted late
# therefore kills the tracks of the subjects prompted early: on the tennis clip
# (racket + player @ 0, ball @ 121) the player's mask went empty from frame 119
# to the end of the clip.
#
# The fix is the one upstream SAM 2.1 shipped: give every object its own memory.
# `output_dict_per_obj[i]` already exists and already holds exactly the right
# thing -- object i's own conditioning and non-conditioning outputs -- and
# `_run_single_frame_inference` already takes the output dict and the batch size
# as arguments, so the whole change is to drive the propagation loop one object
# at a time against its own dict instead of once against the shared one. No
# object ever sees a conditioning frame it was not prompted on, and there is no
# placeholder to poison anything.
#
# What does NOT change: a conditioning frame in the *future* still participates,
# because `select_closest_cond_frames` sees object i's own prompt frame whatever
# side of the current frame it is on. That is what lets a subject prompted on
# frame 48 come back non-empty from frame 38, where she actually walks in.
#
# Cost: the image encoder is untouched -- `_get_image_feature` caches per frame,
# so N objects on one frame still encode it once -- but memory attention, the
# SAM heads and the memory encoder run N times at batch 1 instead of once at
# batch N.
#
# This path is only used when the subjects were prompted on more than one
# distinct frame, which is the only case that can produce a foreign conditioning
# frame. A single prompt frame keeps the batched upstream path, bit for bit.


def _per_object_preflight(predictor, state):
    """`propagate_in_video_preflight`, without the cross-object consolidation.

    Moves each object's prompt outputs from `temp_output_dict_per_obj` into its
    own `output_dict_per_obj`, running the memory encoder on the conditioning
    ones (`add_new_points_or_box` skips it deliberately). Idempotent: the temp
    dicts are emptied on the way out, so calling it once per propagate pass --
    which is what upstream does -- is free after the first.
    """
    import torch
    state["tracking_has_started"] = True
    device = state["device"]
    for obj_idx, temp in state["temp_output_dict_per_obj"].items():
        obj_out = state["output_dict_per_obj"][obj_idx]
        for is_cond in (False, True):
            key = "cond_frame_outputs" if is_cond else "non_cond_frame_outputs"
            for frame_idx, out in temp[key].items():
                if is_cond:
                    hi = torch.nn.functional.interpolate(
                        out["pred_masks"].to(device, non_blocking=True).float(),
                        size=(predictor.image_size, predictor.image_size),
                        mode="bilinear", align_corners=False)
                    feats, pos = predictor._run_memory_encoder(
                        inference_state=state, frame_idx=frame_idx, batch_size=1,
                        high_res_masks=hi,
                        object_score_logits=out["object_score_logits"],
                        # the frame the user interacted with, same as upstream
                        is_mask_from_pts=True)
                    out = dict(out, maskmem_features=feats, maskmem_pos_enc=pos)
                obj_out[key][frame_idx] = out
            temp[key].clear()
        # a frame that became conditioning must not also be non-conditioning
        for frame_idx in obj_out["cond_frame_outputs"]:
            obj_out["non_cond_frame_outputs"].pop(frame_idx, None)


def propagate_per_object(predictor, state):
    """`propagate_in_video`, one object at a time against its own memory.

    Yields (frame_idx, obj_ids, video_res_masks) like upstream does, except that
    `obj_ids` is the subset of subjects this frame has an answer for -- which is
    what lets every subject be walked out from ITS OWN prompt frame, the way the
    browser engine does it:

        backward   its prompt frame -> 0
        forward    its prompt frame -> last frame

    Two passes over the frames cover all of that, and the image encoder is
    shared inside each one, so the encoder runs once per frame visited rather
    than once per subject per frame. Every frame ends up written for every
    subject: a frame before a subject's prompt comes out of the backward pass,
    a frame after it out of the forward one.

    Direction matters here, and not only for quality. A subject prompted on
    frame 121 that gets tracked FORWARD from frame 0 arrives there with one
    spatial memory, no object pointers and nothing else -- a cold start the
    model was never shown -- and it answers with a large false blob. Reached
    backwards out of its own prompt frame it has a full bank, and it goes empty
    where the subject genuinely is not there.
    """
    import torch
    _per_object_preflight(predictor, state)
    n_obj = predictor._get_obj_num(state)
    conds = {i: sorted(state["output_dict_per_obj"][i]["cond_frame_outputs"])
             for i in range(n_obj)}
    if not any(conds.values()):
        raise RuntimeError("No points are provided; please add points first")
    n_frames = state["num_frames"]
    ids = state["obj_ids"]
    device = state["device"]

    def run(frame_idx, obj_idx, reverse):
        obj_out = state["output_dict_per_obj"][obj_idx]
        out, pred = predictor._run_single_frame_inference(
            inference_state=state, output_dict=obj_out, frame_idx=frame_idx,
            batch_size=1, is_init_cond_frame=False, point_inputs=None,
            mask_inputs=None, reverse=reverse, run_mem_encoder=True)
        obj_out["non_cond_frame_outputs"][frame_idx] = out
        return pred

    def emit(frame_idx, picks, reverse):
        if not picks:
            return None
        masks, out_ids = [], []
        for obj_idx, cond_frame in picks:
            if cond_frame:
                stored = state["output_dict_per_obj"][obj_idx][
                    "cond_frame_outputs"][frame_idx]
                pred = stored["pred_masks"].to(device, non_blocking=True)
            else:
                pred = run(frame_idx, obj_idx, reverse)
            masks.append(pred.float())
            out_ids.append(ids[obj_idx])
        state["frames_already_tracked"][frame_idx] = {"reverse": reverse}
        _, video_res = predictor._get_orig_video_res_output(
            state, torch.cat(masks, dim=0))
        return frame_idx, out_ids, video_res

    # --- backward: every subject, from its own earliest prompt frame down to 0
    first = {i: c[0] for i, c in conds.items() if c}
    for frame_idx in range(max(first.values()) - 1, -1, -1):
        got = emit(frame_idx, [(i, False) for i in range(n_obj)
                               if frame_idx < first.get(i, 0)], True)
        if got:
            yield got

    # --- forward: prompt frames themselves, then everything after each of them
    for frame_idx in range(min(first.values()), n_frames):
        picks = []
        for i in range(n_obj):
            if i not in first:
                continue
            if frame_idx in state["output_dict_per_obj"][i]["cond_frame_outputs"]:
                picks.append((i, True))
            elif frame_idx > first[i]:
                picks.append((i, False))
        got = emit(frame_idx, picks, False)
        if got:
            yield got
