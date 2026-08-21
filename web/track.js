/* ---------------------------------------------------------------------------
   EdgeTAM video tracking, in the browser.

   The same loop as sam2's `track_step`, with the four heavy modules replaced by
   ONNX graphs (see onnxexport/export_onnx.py) and the memory bookkeeping —
   which slot holds which frame, which temporal embedding it gets, which mask
   token becomes the object pointer — living here in JS.

   Per frame:

       encoder(frame)                 -> f0, f1, f2          (stays on the GPU)
       memattn(f2, memory bank)       -> memory-conditioned features
       heads(features, f0, f1)        -> 4 masks + IoUs + 4 pointers
       memenc(f2, best mask)          -> 512 memory latents, appended to the bank

   Frame 0 skips memattn: `directly_add_no_mem_embed` is folded into
   heads_prompt.onnx, so the conditioning frame is one encoder hop plus one
   heads hop.

   The encoder -> memattn -> heads chain is wired with
   `preferredOutputLocation: 'gpu-buffer'`, so f0/f1/f2 (9.4 MB per frame at
   fp32) are never read back to JS. Only the four 192x192 mask candidates come
   down, which is what the caller wants to draw anyway.
   --------------------------------------------------------------------------- */

const NEG = -1e4;                  // additive mask for empty memory slots
const MEAN = [0.485, 0.456, 0.406];
const STD = [0.229, 0.224, 0.225];

/* ----------------------------------------------------------------- fp16 */
const HAS_F16 = typeof Float16Array !== 'undefined';

function f16from(f32) {
  if (HAS_F16) return new Uint16Array(new Float16Array(f32).buffer);
  // Chrome < 135: hand-rolled round-to-nearest-even, only used as a fallback
  const out = new Uint16Array(f32.length);
  const fb = new Float32Array(1), ib = new Uint32Array(fb.buffer);
  for (let i = 0; i < f32.length; i++) {
    fb[0] = f32[i];
    const x = ib[0], s = (x >>> 16) & 0x8000;
    let e = ((x >>> 23) & 0xff) - 112, m = x & 0x7fffff;
    if (e <= 0) { out[i] = s; continue; }
    if (e >= 31) { out[i] = s | 0x7c00; continue; }
    out[i] = s | (e << 10) | (m >>> 13);
  }
  return out;
}

function f32from(t) {
  if (t.type !== 'float16') return t.data;
  if (HAS_F16) return Float32Array.from(new Float16Array(t.data.buffer,
    t.data.byteOffset, t.data.length));
  const u = t.data, out = new Float32Array(u.length);
  for (let i = 0; i < u.length; i++) {
    const h = u[i], s = (h & 0x8000) ? -1 : 1, e = (h >> 10) & 0x1f, m = h & 0x3ff;
    out[i] = e === 0 ? s * Math.pow(2, -14) * (m / 1024)
      : e === 31 ? (m ? NaN : s * Infinity)
        : s * Math.pow(2, e - 15) * (1 + m / 1024);
  }
  return out;
}

/* ---------------------------------------------------- fetching the weights */

/** A model URL that did not answer with model bytes.
 *
 * This is the failure a static deployment actually has: a graph the release
 * tarball did not carry, a tier whose files are only half there, or one an
 * edge cache is still 404ing in the minutes after it shipped. A static host
 * answers all of those with its 404 PAGE — HTML — and `fetch` is perfectly
 * happy to hand that HTML back as an ArrayBuffer. Give it to onnxruntime and
 * the whole diagnosis a visitor gets is
 *
 *     Can't create a session. ERROR_CODE: 7, ERROR_MESSAGE: Failed to load
 *     model because protobuf parsing failed.
 *
 * which names neither the file nor the reason. So every fetch below is checked
 * before the bytes go anywhere: the status, the content type, and — for a
 * graph — the first byte, because an ONNX file is a protobuf whose first field
 * is `ir_version` and therefore starts 0x08, never with '<'.
 */
export class ModelFetchError extends Error {
  constructor(file, url, status, why) {
    super(`${file}: ${why}`);
    this.name = 'ModelFetchError';
    this.modelFetch = true;                 // what the engine branches on
    this.file = file; this.url = url; this.status = status; this.why = why;
  }
}

/* ------------------------------------------------------------- the bank */

/** `_prepare_memory_conditioned_features`, as bookkeeping only.
 *
 * Layout the graph expects: `nspat` blocks of 512 spatial latents, then `nptr`
 * pointer tokens. Slots that have no frame yet stay zero and get NEG in the
 * additive mask, which makes a half-full bank attend exactly as a shorter one
 * would — that is what lets the cold-start frames share one fixed-shape graph.
 *
 * ONE BANK IS ONE SUBJECT. `cond` holds only the frames THIS subject was
 * prompted on, because a `WebTracker` never carries more than one, and the
 * caller (web/engines/browser.js) resets it between subjects. That is why the
 * browser engine cannot have the bug the server had to be fixed for: there is
 * no batch to consolidate across, so another subject's prompt frame can never
 * enter this bank as a conditioning frame holding a "not here" placeholder.
 * Keep it that way — if a bank ever tracks several subjects at once, the
 * per-subject conditioning sets have to stay separate.
 */
export class MemoryBank {
  constructor(man, tpos) {
    this.NS = man.nspat; this.D = man.mem_dim;
    this.CH = man.ptr_tokens; this.MAXPTR = man.max_obj_ptrs;
    this.LEN = man.memlen;
    this.tpos = tpos;                       // [7][64]
    this.spatial = new Map();               // frame -> {lat, lpos}
    this.ptr = new Map();                   // frame -> Float32Array(256)
    this.cond = [];
    this.mem = new Float32Array(this.LEN * this.D);
    this.pos = new Float32Array(this.LEN * this.D);
    this.msk = new Float32Array(this.LEN);
  }

  add(t, lat, lpos, ptr, isCond) {
    this.spatial.set(t, { lat, lpos });
    this.ptr.set(t, ptr);
    if (isCond) this.cond.push(t);
    // a conditioning frame is attended to forever; everything else falls out of
    // both windows, so drop it rather than hold 150 frames of latents
    for (const k of this.spatial.keys())
      if (!this.cond.includes(k) && k < t - this.NS) this.spatial.delete(k);
    for (const k of this.ptr.keys())
      if (!this.cond.includes(k) && k < t - this.MAXPTR) this.ptr.delete(k);
  }

  build(t) {
    const { mem, pos, msk, D, NS, CH } = this;
    mem.fill(0); pos.fill(0); msk.fill(NEG);

    const picks = this.cond.map((c) => [0, c]);
    for (let tp = 1; tp < NS; tp++) {
      const prev = t - (NS - tp);
      if (this.spatial.has(prev) && !this.cond.includes(prev)) picks.push([tp, prev]);
    }
    picks.forEach(([tp, f], slot) => {
      const { lat, lpos } = this.spatial.get(f);
      const te = this.tpos[NS - tp - 1];
      const base = slot * 512 * D;
      mem.set(lat, base);
      for (let i = 0; i < 512; i++)
        for (let d = 0; d < D; d++) pos[base + i * D + d] = lpos[i * D + d] + te[d];
      msk.fill(0, slot * 512, slot * 512 + 512);
    });

    const ptrs = this.cond.filter((c) => c <= t).map((c) => this.ptr.get(c));
    for (let d = 1; d < this.MAXPTR; d++) {
      const f = t - d;
      if (f < 0) break;
      if (this.ptr.has(f) && !this.cond.includes(f)) ptrs.push(this.ptr.get(f));
    }
    const pbase = NS * 512;
    ptrs.slice(0, this.MAXPTR).forEach((p, i) => {
      mem.set(p, (pbase + i * CH) * D);
      msk.fill(0, pbase + i * CH, pbase + i * CH + CH);
    });
    return { mem, pos, msk };
  }
}

/* ------------------------------------------------------------- the model */

export class WebTracker {
  constructor(ort, opts = {}) {
    this.ort = ort;
    this.ep = opts.ep || 'webgpu';
    this.fp16 = opts.fp16 !== false;
    this.chain = opts.chain !== false && this.ep === 'webgpu';
    this.dir = opts.dir || './models/';
    this.dtype = this.fp16 ? 'float16' : 'float32';
  }

  tensor(type, data, dims) {
    if (type === 'float16') return new this.ort.Tensor('float16', f16from(data), dims);
    return new this.ort.Tensor('float32', data, dims);
  }

  /** One file out of the model directory, checked to actually BE that file.
   *
   *  `kind` is 'json', 'bin' or 'onnx'. Anything that is not what it claims to
   *  be throws a ModelFetchError naming the URL and the status, which is the
   *  difference between "heads_prompt.fp16.onnx is not in this deployment
   *  (HTTP 404)" and "protobuf parsing failed". */
  async grab(file, kind) {
    const url = this.dir + file;
    let res;
    try { res = await fetch(url); }
    catch (e) {
      throw new ModelFetchError(file, url, 0,
        `the request failed (${e && e.message ? e.message : e})`);
    }
    if (!res.ok) {
      throw new ModelFetchError(file, url, res.status, res.status === 404
        ? 'it is not in this deployment (HTTP 404)'
        : `the server answered HTTP ${res.status}`);
    }
    const ct = (res.headers.get('content-type') || '').split(';')[0].trim()
      .toLowerCase();
    if (kind === 'json') {
      const text = await res.text();
      try { return JSON.parse(text); } catch (e) {
        throw new ModelFetchError(file, url, res.status,
          `the server answered with ${ct || 'something'} rather than JSON`);
      }
    }
    const buf = await res.arrayBuffer();
    if (!buf.byteLength) {
      throw new ModelFetchError(file, url, res.status, 'the file is empty');
    }
    if (kind === 'onnx') {
      const b0 = new Uint8Array(buf, 0, 1)[0];
      if (ct === 'text/html' || b0 === 0x3c /* '<' */) {
        throw new ModelFetchError(file, url, res.status,
          `the server answered with a page, not an ONNX graph `
          + `(${buf.byteLength} bytes${ct ? ', ' + ct : ''}) \u2014 the file is not `
          + 'in this deployment, or a cache is still serving its 404');
      }
    }
    return buf;
  }

  async load(log = () => {}) {
    const t0 = performance.now();
    this.man = await this.grab('manifest.json', 'json');
    const cb = new Float32Array(await this.grab('consts.bin', 'bin'));
    const c = {};
    for (const [k, v] of Object.entries(this.man.consts))
      c[k] = cb.subarray(v.offset / 4, v.offset / 4 + v.count);
    this.noMem = c.no_mem_embed;
    this.tpos = [];
    for (let i = 0; i < this.man.nspat; i++)
      this.tpos.push(c.maskmem_tpos_enc.subarray(i * this.man.mem_dim,
        (i + 1) * this.man.mem_dim));

    const sfx = this.fp16 ? '.fp16' : '';
    const eps = [this.ep];
    const gpu = this.chain ? 'gpu-buffer' : 'cpu';
    const opt = (outs) => ({
      executionProviders: eps,
      graphOptimizationLevel: 'all',
      // ORT routes its own WARNING lines to console.error, and graph
      // optimisation emits dozens of them ("can't constant fold Sqrt") that
      // are expected on the WebGPU EP. 3 = errors only, so a page that logs a
      // console error is reporting a real one.
      logSeverityLevel: 3,
      ...(outs ? { preferredOutputLocation: outs } : {}),
    });
    // a graph that opened before a later one failed still holds its GPU
    // buffers, and nothing else has a handle on it — so keep the list
    const opened = [];
    const mk = async (file, outs) => {
      const t = performance.now();
      const buf = await this.grab(file, 'onnx');
      const s = await this.ort.InferenceSession.create(buf, opt(outs));
      opened.push(s);
      log(`${file}: ${(buf.byteLength / 1e6).toFixed(1)} MB, ` +
        `${(performance.now() - t).toFixed(0)} ms`);
      return { s, bytes: buf.byteLength };
    };

    let enc, mat, hds, hpr, hmk, mec;
    try {
      enc = await mk(`encoder${sfx}.onnx`, { f0: gpu, f1: gpu, f2: gpu });
      mat = await mk(`memattn${sfx}.onnx`, { out: gpu });
      hds = await mk(`heads${sfx}.onnx`);
      hpr = await mk(`heads_prompt${sfx}.onnx`);
      // A lasso/polygon prompt takes a different route through EdgeTAM entirely
      // (`use_mask_input_as_output_without_sam`), so it is its own graph. Older
      // model sets do not have it; the caller checks `has_mask_prompt` first.
      hmk = this.man.has_mask_prompt ? await mk(`heads_mask${sfx}.onnx`) : null;
      // the memory encoder always computes in fp32; only its pix_feat input dtype
      // follows the encoder, so a chained GPU buffer needs no conversion
      mec = await mk(this.chain && this.fp16 ? 'memenc.f16in.onnx'
        : 'memenc.onnx');
    } catch (e) {
      for (const s of opened) { try { await s.release?.(); } catch (_) { /* gone */ } }
      throw e;
    }
    this.enc = enc.s; this.mat = mat.s; this.hds = hds.s;
    this.hpr = hpr.s; this.mec = mec.s; this.hmk = hmk ? hmk.s : null;
    this.bytes = enc.bytes + mat.bytes + hds.bytes + hpr.bytes + mec.bytes
      + (hmk ? hmk.bytes : 0);
    this.loadMs = performance.now() - t0;
    return this;
  }

  /** RGBA from a canvas -> the encoder's normalised NCHW input. */
  preprocess(rgba) {
    const S = this.man.image_size, n = S * S;
    const out = new Float32Array(3 * n);
    for (let i = 0; i < n; i++) {
      out[i] = (rgba[i * 4] / 255 - MEAN[0]) / STD[0];
      out[n + i] = (rgba[i * 4 + 1] / 255 - MEAN[1]) / STD[1];
      out[2 * n + i] = (rgba[i * 4 + 2] / 255 - MEAN[2]) / STD[2];
    }
    return this.tensor(this.dtype, out, [1, 3, S, S]);
  }

  reset() {
    this.bank = new MemoryBank(this.man, this.tpos);
    this.t = 0;
  }

  /** One frame.
   *
   * `prompt` is set only on a conditioning frame, and is either
   *   {coords, labels}   a click / box prompt   -> heads_prompt
   *   {mask}             a drawn shape at S*S   -> heads_mask
   * and null on every tracking frame.
   */
  async step(rgba, prompt, timing) {
    const points = prompt && prompt.coords ? prompt : null;
    const shape = prompt && prompt.mask ? prompt : null;
    const M = this.man, G = M.grid, D = M.mem_dim, t = this.t;
    const mark = (k, t0) => { if (timing) timing[k] = (timing[k] || 0) + performance.now() - t0; };

    let t0 = performance.now();
    const image = this.preprocess(rgba);
    mark('pre', t0);

    t0 = performance.now();
    const e = await this.enc.run({ image });
    mark('enc', t0);
    image.dispose?.();

    let feats, res, k;
    if (shape) {
      // No f0/f1 and no no_mem_embed here: `_use_mask_as_output` branches
      // before the memory path and discards the high-res upscaling entirely,
      // so this graph takes the encoder's f2 and the mask, and nothing else.
      // All four token slots carry the same answer, so k = 0.
      t0 = performance.now();
      res = await this.hmk.run({
        pix_feat: e.f2,
        mask_full: this.tensor(this.dtype, shape.mask, [1, 1, M.image_size, M.image_size]),
      });
      mark('hds', t0);
      k = 0;
    } else if (points) {
      t0 = performance.now();
      res = await this.hpr.run({
        pix_feat: e.f2, f0: e.f0, f1: e.f1,
        point_coords: this.tensor(this.dtype, points.coords,
          [1, points.labels.length, 2]),
        point_labels: this.tensor(this.dtype, points.labels,
          [1, points.labels.length]),
        add_no_mem: this.tensor(this.dtype, new Float32Array([1]), [1]),
      });
      mark('hds', t0);
      k = 0;                       // >1 prompt point -> the single-mask token
    } else {
      t0 = performance.now();
      const { mem, pos, msk } = this.bank.build(t);
      const out = await this.mat.run({
        feat: e.f2,
        memory: this.tensor(this.dtype, mem, [M.memlen, 1, D]),
        memory_pos: this.tensor(this.dtype, pos, [M.memlen, 1, D]),
        mem_mask: this.tensor(this.dtype, msk, [1, 1, 1, M.memlen]),
      });
      mark('mat', t0);
      feats = out.out;
      t0 = performance.now();
      res = await this.hds.run({
        pix_feat: feats, f0: e.f0, f1: e.f1,
        point_coords: this.tensor(this.dtype, new Float32Array(2), [1, 1, 2]),
        point_labels: this.tensor(this.dtype, new Float32Array([-1]), [1, 1]),
      });
      mark('hds', t0);
      const iou = f32from(res.ious);
      k = 1 + (iou[2] > iou[1] ? (iou[3] > iou[2] ? 2 : 1) : (iou[3] > iou[1] ? 2 : 0));
    }

    const masks = f32from(res.masks);
    const ptrs = f32from(res.obj_ptrs);
    const osl = f32from(res.object_score_logits);
    const P = G * 4, area = P * P;
    const low = masks.slice(k * area, (k + 1) * area);
    const ptr = Float32Array.from(ptrs.subarray(k * M.hidden, (k + 1) * M.hidden));

    t0 = performance.now();
    const me = await this.mec.run({
      pix_feat: this.chain ? e.f2 : new this.ort.Tensor('float32',
        f32from(e.f2), [1, M.hidden, G, G]),
      low_res_mask: new this.ort.Tensor('float32', low, [1, 1, P, P]),
    });
    mark('mec', t0);

    this.bank.add(t, Float32Array.from(me.lat.data), Float32Array.from(me.lpos.data),
      ptr, !!prompt);
    for (const o of [e.f0, e.f1, e.f2, feats, res.masks, res.ious, res.obj_ptrs,
      res.object_score_logits, me.lat, me.lpos]) o?.dispose?.();
    this.t++;
    return { low, size: P, score: osl[0] };
  }
}

/** Hand the sessions back. A model set is ~50 MB of fp16 weights plus its GPU
 *  buffers, and the caller keeps one resolution alive at a time, so switching
 *  the quality chip has to actually free the one before it. */
WebTracker.prototype.release = async function release() {
  for (const s of [this.enc, this.mat, this.hds, this.hpr, this.hmk, this.mec]) {
    if (s && s.release) { try { await s.release(); } catch (e) { /* already gone */ } }
  }
  this.enc = this.mat = this.hds = this.hpr = this.hmk = this.mec = null;
  this.bank = null;
};

/** Chained sessions return before the GPU has finished, so per-stage numbers
 *  from `step` are only a split of the wall clock. This runs one stage in
 *  isolation with a CPU output, which forces a fence, N times. */
export async function benchStage(session, feeds, n = 20) {
  for (let i = 0; i < 3; i++) await session.run(feeds);
  const t0 = performance.now();
  for (let i = 0; i < n; i++) await session.run(feeds);
  return (performance.now() - t0) / n;
}
