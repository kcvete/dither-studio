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
    return this;
  }

  async meta() {
    const m = await this.api('/api/palettes');
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

  /* --------------------------------------------------------------- clip */
  async open(file, { maxSeconds = 10, onProgress } = {}) {
    if (onProgress) onProgress({ phase: 'upload', text: 'uploading ' + file.name + '…' });
    const fd = new FormData();
    fd.append('file', file);
    fd.append('max_seconds', String(maxSeconds));
    const j = await this.api('/api/upload', {
      method: 'POST', body: fd, headers: this.headers(),
    });
    this.clip = { job: j.job, nFrames: j.n_frames, w: j.w, h: j.h, fps: j.fps };
    return this.clip;
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

  dispose() { this.clip = null; }
}
