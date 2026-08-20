/* ---------------------------------------------------------------------------
   REMOTE ENGINE — the FastAPI accelerator (server/server.py).

   Three deployments, one class:

     baseUrl ''                     the server that served this page (local)
     baseUrl 'http://…:8765'        a server on the LAN / another port
     baseUrl 'https://…' + apiKey   a rented GPU box running the same server.py

   The only thing the paid tier adds to the wire format is an
   `Authorization: Bearer <key>` header, which server.py demands when DV_API_KEY
   is set and ignores otherwise. Everything else — routes, bodies, polling —
   is identical, so there is one client and no "premium" branch anywhere.
--------------------------------------------------------------------------- */
'use strict';

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

export class RemoteEngine {
  constructor({ baseUrl = '', apiKey = '', label, sublabel } = {}) {
    this.id = 'remote';
    this.baseUrl = String(baseUrl || '').replace(/\/+$/, '');
    this.apiKey = apiKey || '';
    this.label = label || (this.baseUrl ? 'Remote server' : 'Local server');
    this.sublabel = sublabel || '';
    this.clip = null;
    this.supports = {
      maskPrompt: true,          // EdgeTAM add_new_mask, full quality
      perObjectPromptFrames: true,
      backward: true,
      multiObject: 'batched',    // one propagate pass, all subjects at once
      exportMime: 'video/mp4',
      exportExt: 'mp4',
      exportPlayable: true,
      // /api/upload_image + a one-frame /preview. Servers older than this
      // route say nothing, and the page keeps stills whole-image.
      stillSubjects: false,
      // POST /api/jobs/<id>/reextract: a different trim out of the clip the
      // server already has. Older servers say nothing and get a re-upload.
      reextract: false,
      // GET /api/extract/<ticket> while the upload POST is still running.
      extractProgress: false,
      // no 10 s / 300 frame ceiling. An old server still has one; the page
      // says so rather than pretending the estimate applies.
      uncapped: false,
      // POST /api/jobs/<id>/original: the render's own frames, re-encoded
      // without the dither. Older servers say nothing and the page hides the
      // checkbox rather than offering a button that 404s.
      original: false,
      // render / original / dots take an inclusive frame_in..frame_out window
      // over the frames already extracted, so narrowing the trim after a track
      // costs nothing. An older server ignores the fields and renders the whole
      // clip, which is why the page checks before it narrows.
      frameRange: false,
      // filled in from /api/palettes; this is what a server too old to
      // advertise formats can be assumed to do
      formats: [{ id: 'mp4', label: 'MP4 · H.264', ext: 'mp4', mime: 'video/mp4',
                  alpha: false, available: true, note: '' }],
    };
  }

  url(p) { return this.baseUrl + p; }

  headers(extra) {
    const h = Object.assign({}, extra || {});
    if (this.apiKey) h.Authorization = 'Bearer ' + this.apiKey;
    return h;
  }

  async api(path, opts = {}) {
    const r = await fetch(this.url(path), Object.assign({}, opts, {
      headers: this.headers(opts.headers),
    }));
    if (!r.ok) {
      let d = r.status + ' ' + r.statusText;
      try { d = (await r.json()).detail || d; } catch (e) { /* not json */ }
      throw new Error(d);
    }
    return r.json();
  }

  async blob(path) {
    const r = await fetch(this.url(path), { headers: this.headers() });
    if (!r.ok) throw new Error(path + ': ' + r.status);
    return r.blob();
  }

  /* ------------------------------------------------------------ metadata */
  async init() {
    this.probe = await this.api('/api/meta');
    this.supports.reextract = !!this.probe.reextract;
    this.supports.extractProgress = !!this.probe.extract_progress;
    this.supports.uncapped = !!this.probe.uncapped;
    this.supports.original = !!this.probe.original;
    this.supports.frameRange = !!this.probe.frame_range;
    return this;
  }

  async meta() {
    const m = await this.api('/api/palettes');
    this.supports.stillSubjects = !!m.segment_image;
    if (m.reextract !== undefined) this.supports.reextract = !!m.reextract;
    if (m.uncapped !== undefined) this.supports.uncapped = !!m.uncapped;
    if (m.original !== undefined) this.supports.original = !!m.original;
    if (m.frame_range !== undefined) this.supports.frameRange = !!m.frame_range;
    if (m.extract_progress !== undefined) {
      this.supports.extractProgress = !!m.extract_progress;
    }
    if (Array.isArray(m.formats) && m.formats.length) {
      this.supports.formats = m.formats.map((f) => Object.assign({
        available: true,
        note: f.alpha ? 'the flat background is keyed out; only the dots are opaque' : '',
      }, f));
    }
    return Object.assign({}, m, {
      engine: 'remote',
      // the server exports three tracker resolutions; the browser only has the
      // one it shipped models for, so this list is engine-specific
      track_sizes: m.track_sizes,
      default_track_size: m.default_track_size,
    });
  }

  async blueNoise(n, seed) {
    const j = await this.api(`/api/bluenoise?n=${n}&seed=${seed}`);
    return Float32Array.from(j.tile);
  }

  /* --------------------------------------------------------------- clip
   * Nothing is capped. The whole file goes up, ffmpeg's -ss/-t picks the
   * range, and a two-minute clip is two minutes of frames. Because that is one
   * long POST, the page polls /api/extract/<ticket> beside it: the upload
   * bytes come from XHR's own progress events, the frames from ffmpeg's.
   */
  async open(file, { trimStart = 0, trimEnd = null, fps = 30,
                     onProgress } = {}) {
    const ticket = 't' + Math.random().toString(36).slice(2) + Date.now().toString(36);
    if (onProgress) onProgress({ phase: 'upload', text: 'uploading ' + file.name + '…' });
    const fd = new FormData();
    fd.append('file', file);
    fd.append('fps', String(fps));
    fd.append('trim_start', String(trimStart || 0));
    if (trimEnd) fd.append('trim_end', String(trimEnd));
    fd.append('ticket', ticket);
    const j = await this.post('/api/upload', fd, ticket, onProgress,
                              'uploading ' + file.name);
    this.clip = { job: j.job, nFrames: j.n_frames, w: j.w, h: j.h, fps: j.fps,
                  trimStart: j.trim_start || 0, seconds: j.seconds };
    return this.clip;
  }

  /** A different range out of the clip the server already holds -- no upload.
   *  Returns a NEW job: the old one's masks belong to the old range. */
  async reopen({ trimStart = 0, trimEnd = null, fps = null,
                 onProgress } = {}) {
    if (!this.clip || !this.clip.job) throw new Error('no clip is open');
    if (!this.supports.reextract) throw new Error('this server cannot re-extract');
    const ticket = 'r' + Math.random().toString(36).slice(2) + Date.now().toString(36);
    const stop = this.watchExtract(ticket, onProgress, 're-extracting');
    try {
      const j = await this.api(`/api/jobs/${this.clip.job}/reextract`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ trim_start: trimStart || 0,
                               trim_end: trimEnd || null,
                               fps: fps || undefined, ticket }),
      });
      this.clip = { job: j.job, nFrames: j.n_frames, w: j.w, h: j.h, fps: j.fps,
                    trimStart: j.trim_start || 0, seconds: j.seconds };
      return this.clip;
    } finally { stop(); }
  }

  /** Poll the extraction ticket until told to stop. Silent on any failure --
   *  it is a progress line, not a result. */
  watchExtract(ticket, onProgress, verb) {
    let live = !!(onProgress && this.supports.extractProgress);
    const tick = async () => {
      while (live) {
        try {
          const st = await this.api('/api/extract/' + ticket);
          if (live && st && st.phase === 'extract' && st.total) {
            onProgress({ phase: 'extract', done: st.done, total: st.total,
                         text: `${verb}: ${st.done}/${st.total} frames…` });
          }
        } catch (e) { /* the POST is the one that matters */ }
        await sleep(400);
      }
    };
    tick();
    return () => { live = false; };
  }

  /** POST a FormData with real upload progress, then extraction progress. */
  post(path, fd, ticket, onProgress, verb) {
    return new Promise((ok, no) => {
      const xhr = new XMLHttpRequest();
      xhr.open('POST', this.url(path));
      const h = this.headers();
      Object.keys(h).forEach((k) => xhr.setRequestHeader(k, h[k]));
      let stop = () => {};
      xhr.upload.onprogress = (e) => {
        if (!onProgress || !e.lengthComputable) return;
        const pct = Math.round(100 * e.loaded / e.total);
        onProgress({ phase: 'upload', done: e.loaded, total: e.total,
                     text: `${verb}: ${pct}%…` });
      };
      xhr.upload.onload = () => { stop = this.watchExtract(ticket, onProgress, 'extracting'); };
      xhr.onload = () => {
        stop();
        let j = null;
        try { j = JSON.parse(xhr.responseText); } catch (e) { /* not json */ }
        if (xhr.status >= 200 && xhr.status < 300) return ok(j);
        no(new Error((j && j.detail) || (xhr.status + ' ' + xhr.statusText)));
      };
      xhr.onerror = () => { stop(); no(new Error('the server could not be reached')); };
      xhr.send(fd);
    });
  }

  /* --------------------------------------------------------------- still
   * The picture goes up once, as a job of one frame, and every click after
   * that is a /preview on it -- one image encode and the SAM heads, no
   * propagation and nothing re-uploaded. The page sends it already scaled to
   * the size it prompts at, so clicks and masks share one coordinate space. */
  async openStill(blob, { name = 'still.png', maxSide = 1600 } = {}) {
    if (!this.supports.stillSubjects) {
      throw new Error('this server has no /api/upload_image — update it, or '
        + 'switch to the browser engine for subjects in a still');
    }
    const fd = new FormData();
    fd.append('file', blob, name);
    fd.append('max_side', String(maxSide));
    const j = await this.api('/api/upload_image', {
      method: 'POST', body: fd, headers: this.headers(),
    });
    this.clip = { job: j.job, nFrames: 1, w: j.w, h: j.h, fps: 1, still: true };
    return Object.assign({ kind: 'image' }, this.clip);
  }

  /** Single-image segmentation: the one-frame preview, which is exactly what
   *  the still flow wants. Masks come back at the job's own resolution. */
  async segmentImage({ objects, imageSize }, onLog) {
    if (onLog) onLog('reading the selection…');
    return this.previewFrame({ frameIdx: 0, imageSize, objects });
  }

  jobPath(p) { return `/api/jobs/${this.clip.job}${p}`; }

  async frame(i) {
    return createImageBitmap(await this.blob(this.jobPath(`/frame/${i}`)));
  }

  /** A URL an <img> or <video> can use. Without a key that is just the route;
   *  with one it has to be a blob, because a tag cannot carry a header. */
  async frameURL(i) {
    const p = this.jobPath(`/frame/${i}`);
    if (!this.apiKey) return { url: this.url(p), revoke: false };
    return { url: URL.createObjectURL(await this.blob(p)), revoke: true };
  }

  /** Soft mask for one subject on one frame, as the server's L PNG. */
  async mask(objId, i) {
    return createImageBitmap(await this.blob(this.jobPath(`/mask/${objId}/${i}`)));
  }

  /* ------------------------------------------------------------ prompts */
  static payload(objects) {
    return objects.map((o) => {
      const base = { id: o.id, frame_idx: o.frameIdx | 0 };
      return o.mask ? Object.assign(base, { mask: o.mask })
        : Object.assign(base, { points: o.points, box: o.box });
    });
  }

  async previewFrame({ frameIdx, imageSize, objects }) {
    const r = await this.api(this.jobPath('/preview'), {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ frame_idx: frameIdx, image_size: imageSize,
                             objects: RemoteEngine.payload(objects) }),
    });
    const out = [];
    for (const o of r.objects) {
      const im = new Image();
      await new Promise((res) => { im.onload = im.onerror = res; im.src = o.mask; });
      out.push({ id: o.id, image: im, area: o.area });
    }
    return { objects: out, elapsedS: r.elapsed_s, imageSize: r.image_size,
             backend: r.backend, frameIdx: r.frame_idx };
  }

  async track({ objects, imageSize }, onProgress) {
    await this.api(this.jobPath('/track'), {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        frame_idx: objects.length ? objects[0].frameIdx | 0 : 0,
        image_size: imageSize,
        objects: RemoteEngine.payload(objects),
      }),
    });
    for (;;) {
      const st = await this.api(this.jobPath('/status'));
      if (onProgress) {
        onProgress({
          done: st.done_frames, total: st.n_frames,
          text: st.state === 'loading' ? 'loading frames…'
            : `${st.done_frames}/${st.n_frames} · ${st.fps.toFixed(1)} fps`,
        });
      }
      if (st.state === 'done') {
        return { frames: st.done_frames, elapsedS: st.elapsed_s, fps: st.fps,
                 device: (st.device || '').toUpperCase(),
                 backend: st.backend || st.precision || '', imageSize: st.image_size };
      }
      if (st.state === 'error') throw new Error(st.error);
      await sleep(350);
    }
  }

  /* ------------------------------------------------------------- export */
  async exportClip(params, onProgress) {
    const fmt = params.format || 'mp4';
    const f = (this.supports.formats || []).find((x) => x.id === fmt)
      || { id: fmt, ext: 'mp4', mime: 'video/mp4', alpha: false };
    await this.api(this.jobPath('/render'), {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(params),
    });
    for (;;) {
      const st = (await this.api(this.jobPath('/status'))).render;
      if (onProgress) {
        onProgress({ done: st.done_frames, total: st.n_frames,
                     text: `${st.done_frames}/${st.n_frames}` });
      }
      if (st.state === 'done') {
        const path = this.jobPath('/output/' + fmt);
        // Cache-bust: the same URL is re-rendered in place on every export.
        let url = this.url(path) + '?t=' + Date.now();
        if (this.apiKey) url = URL.createObjectURL(await this.blob(path));
        return { url, mime: f.mime, ext: f.ext,
                 playable: fmt !== 'gif' && fmt !== 'prores',
                 image: fmt === 'gif', alpha: !!f.alpha,
                 frames: st.done_frames, elapsedS: st.elapsed_s, fps: st.fps,
                 bytes: st.bytes || 0,
                 note: fmt === 'prores'
                   ? 'ProRes 4444 does not play in a browser — download it'
                   : (f.alpha ? 'alpha channel written' : '') };
      }
      if (st.state === 'error') throw new Error(st.error);
      await sleep(300);
    }
  }

  /* ------------------------------------------------------- original cut
   * The same frames the render just consumed, re-encoded without the dither.
   * The server has them on disk under the job id, so this ships no pixels
   * either way — one POST, one file, and a frame count it refuses to fudge.
   */
  async exportOriginal(params, onProgress) {
    if (onProgress) {
      onProgress({ done: 0, total: 1, text: 'encoding the original cut…' });
    }
    const r = await this.api(this.jobPath('/original'), {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ format: params.format || 'mp4',
                             fps: params.fps || null,
                             expect_frames: params.expect_frames || null,
                             frame_in: params.frame_in | 0,
                             frame_out: params.frame_out === undefined
                               ? null : params.frame_out }),
    });
    const path = this.jobPath('/original/' + r.format);
    let url = this.url(path) + '?t=' + Date.now();
    if (this.apiKey) url = URL.createObjectURL(await this.blob(path));
    if (onProgress) onProgress({ done: r.frames, total: r.frames, text: 'done' });
    return { url, ext: r.ext, mime: (this.supports.formats.find((f) => f.id === r.format)
                                     || {}).mime || 'video/mp4',
             frames: r.frames, bytes: r.bytes, elapsedS: r.elapsed_s,
             w: r.w, h: r.h, fps: r.fps, format: r.format, matched: !!r.matched,
             frameIn: r.frame_in | 0, frameOut: r.frame_out | 0 };
  }

  /* --------------------------------------------------------- dot data */
  /** The dots as positions, rendered server-side and handed back as the
   *  .dots.gz bytes themselves — the same file the browser engine builds. */
  async exportDots(params, onProgress) {
    if (onProgress) onProgress({ done: 0, total: 1, text: 'rendering dot positions…' });
    const r = await this.api(this.jobPath('/dots'), {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(params),
    });
    const blob = await this.blob(this.jobPath('/out.dots.gz'));
    if (onProgress) onProgress({ done: r.frames, total: r.frames, text: 'done' });
    return { bytes: new Uint8Array(await blob.arrayBuffer()), stats: r };
  }

  /** A finished sequence (dot positions, morphs already tweened in JS) ->
   *  a video. The server never re-derives the transition; it rasterises. */
  async renderSequence(bytes, format = 'mp4') {
    const fd = new FormData();
    fd.append('file', new Blob([bytes], { type: 'application/octet-stream' }),
              'sequence.dots.gz');
    fd.append('format', format);
    const r = await this.api('/api/sequence', { method: 'POST', body: fd,
                                                headers: this.headers() });
    const f = (this.supports.formats || []).find((x) => x.id === format) || {};
    let url = this.url(r.url) + '?t=' + Date.now();
    if (this.apiKey) url = URL.createObjectURL(await this.blob(r.url));
    return { url, ext: f.ext || 'mp4', mime: f.mime || 'video/mp4',
             bytes: r.bytes, frames: r.frames, elapsedS: r.elapsed_s };
  }

  /** A detached handle on the job that is open RIGHT NOW. The server keeps
   *  its frames and masks on disk under the job id, so this is just the two
   *  routes with the id already bound — which is what lets a sequence item
   *  redraw its dots at a new look after the studio has moved on. */
  snapshot() {
    const c = this.clip;
    if (!c) return null;
    const job = c.job;
    const at = (p) => `/api/jobs/${job}${p}`;
    return {
      id: 'remote', job, w: c.w, h: c.h, nFrames: c.nFrames, fps: c.fps,
      frame: async (i) => createImageBitmap(await this.blob(at(`/frame/${i}`))),
      mask: async (objId, i) =>
        createImageBitmap(await this.blob(at(`/mask/${objId}/${i}`))),
    };
  }

  dispose() { this.clip = null; }
}
