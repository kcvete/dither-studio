/* ---------------------------------------------------------------------------
   GIF89a ENCODER — animated GIF export for Dither Studio.
   No dependencies, no imports, no network, no DOM.

   LOADING. A classic <script> cannot contain `export`, so this file is a plain
   classic script that publishes itself twice: on `globalThis.GifEnc` (browser
   and Node alike) and on `module.exports` (CommonJS). From an ES module load it
   side-effect style and read the global:

       await import('./vendor/gifenc.js');      // no bindings to destructure
       const bytes = globalThis.GifEnc.encode({ ... });

   From a page:  <script src="vendor/gifenc.js"></script>   then  GifEnc.encode
   From Node:    const GifEnc = require('./web/vendor/gifenc.js');

   SCOPE. One global colour table, <= 256 colours, no local per-frame palettes,
   no transparency, no interframe delta/dispose tricks — every frame is stored
   whole. That is the right trade for dithered output: 2-4 flat colours are
   almost pure run-length under LZW, and the encoder stays small enough to read.
--------------------------------------------------------------------------- */
'use strict';

const GifEnc = (() => {

/* ------------------------------------------------------------- byte sink --- */
/* Growable output buffer. Doubling, so the amortised cost is a memcpy or two
   for a whole animation instead of one array concat per frame. */
function Sink(cap) { this.b = new Uint8Array(cap || 1 << 16); this.n = 0; }
Sink.prototype.need = function (k) {
  if (this.n + k <= this.b.length) return;
  let c = this.b.length * 2;
  while (c < this.n + k) c *= 2;
  const nb = new Uint8Array(c); nb.set(this.b.subarray(0, this.n)); this.b = nb;
};
Sink.prototype.u8 = function (v) { this.need(1); this.b[this.n++] = v & 0xff; };
Sink.prototype.u16 = function (v) {                       // GIF is little-endian
  this.need(2); this.b[this.n++] = v & 0xff; this.b[this.n++] = (v >> 8) & 0xff;
};
Sink.prototype.raw = function (a) { this.need(a.length); this.b.set(a, this.n); this.n += a.length; };
Sink.prototype.ascii = function (s) {
  this.need(s.length);
  for (let i = 0; i < s.length; i++) this.b[this.n++] = s.charCodeAt(i) & 0xff;
};
Sink.prototype.done = function () { return this.b.slice(0, this.n); };

/* --------------------------------------------------------------- palette --- */
function hexRGB(h) {
  h = String(h).replace('#', '');
  if (h.length === 3) h = h[0] + h[0] + h[1] + h[1] + h[2] + h[2];
  return [parseInt(h.slice(0, 2), 16) | 0, parseInt(h.slice(2, 4), 16) | 0, parseInt(h.slice(4, 6), 16) | 0];
}
function paletteBytes(colors) {
  const p = new Uint8Array(colors.length * 3);
  for (let i = 0; i < colors.length; i++) {
    const c = hexRGB(colors[i]);
    p[i * 3] = c[0]; p[i * 3 + 1] = c[1]; p[i * 3 + 2] = c[2];
  }
  return p;
}
function nearest(pal, np, r, g, b) {
  let bi = 0, bd = Infinity;
  for (let i = 0; i < np; i++) {
    const dr = r - pal[i * 3], dg = g - pal[i * 3 + 1], db = b - pal[i * 3 + 2];
    const d = dr * dr + dg * dg + db * db;
    if (d < bd) { bd = d; bi = i; }
  }
  return bi;
}

/* RGB int -> palette index, pre-seeded with the palette itself so an exact hit
   costs one Map lookup and never runs the nearest() scan. Seeded backwards so a
   duplicated colour resolves to its lowest index. */
function newCache(pal, np) {
  const m = new Map();
  for (let i = np - 1; i >= 0; i--) m.set((pal[i * 3] << 16) | (pal[i * 3 + 1] << 8) | pal[i * 3 + 2], i);
  return m;
}

/* RGBA bytes -> one palette index per pixel. Exact RGB match first, nearest in
   squared RGB otherwise; every miss is memoised. Dithered frames carry a
   handful of distinct colours, so this is a Map hit per pixel and nothing more.
   A frame that is already width*height indices is passed through untouched. */
function indexFrame(rgba, width, height, palette, cache) {
  const n = width * height;
  if (rgba.length === n) return (rgba instanceof Uint8Array) ? rgba : Uint8Array.from(rgba);
  if (rgba.length !== n * 4) throw new Error('gifenc: frame is ' + rgba.length + ' bytes, expected ' + n + ' (indices) or ' + (n * 4) + ' (RGBA)');
  const pal = paletteBytes(palette), np = palette.length;
  const c = cache || newCache(pal, np);
  const out = new Uint8Array(n);
  for (let i = 0, p = 0; i < n; i++, p += 4) {
    const key = (rgba[p] << 16) | (rgba[p + 1] << 8) | rgba[p + 2];
    let k = c.get(key);
    if (k === undefined) { k = nearest(pal, np, rgba[p], rgba[p + 1], rgba[p + 2]); c.set(key, k); }
    out[i] = k;
  }
  return out;
}

/* ------------------------------------------------------------------ LZW ---- */
/* Dictionary as a flat Int32Array indexed by (prefix << 8 | char). 0 means
   "empty": code 0 is a raw colour index and is never handed out as a dictionary
   entry (those start at clear+2), so 0 is a safe sentinel. Allocated once and
   reused across frames — 4 MB, cleared with a memset per frame. */
let TABLE = null;

function lzw(idx, minCodeSize, out) {
  if (!TABLE) TABLE = new Int32Array(4096 * 256);
  TABLE.fill(0);

  const clear = 1 << minCodeSize, eoi = clear + 1;
  let codeSize = minCodeSize + 1, next = eoi + 1;
  let cur = 0, bits = 0;                       // bit accumulator, LSB first
  const block = new Uint8Array(255); let bn = 0;

  const flush = () => { out.u8(bn); out.raw(block.subarray(0, bn)); bn = 0; };
  const put = (v) => { block[bn++] = v; if (bn === 255) flush(); };
  const emit = (code) => {
    cur |= code << bits; bits += codeSize;     // <= 7 + 12 bits, fits an int32
    while (bits >= 8) { put(cur & 0xff); cur >>>= 8; bits -= 8; }
  };

  emit(clear);
  if (idx.length) {
    let prefix = idx[0];
    for (let i = 1; i < idx.length; i++) {
      const k = idx[i], key = (prefix << 8) | k;
      const got = TABLE[key];
      if (got !== 0) { prefix = got; continue; }
      emit(prefix);
      if (next === 4096) {
        emit(clear);                           // dictionary full: restart it.
        TABLE.fill(0);                         // the clear goes out at the OLD
        next = eoi + 1; codeSize = minCodeSize + 1;   // width, then we shrink
      } else {
        // grow before assigning, so a code is only ever emitted at a width that
        // can hold it — this is the decoder's rule too
        if (next >= (1 << codeSize) && codeSize < 12) codeSize++;
        TABLE[key] = next++;
      }
      prefix = k;
    }
    emit(prefix);
  }
  emit(eoi);
  if (bits > 0) put(cur & 0xff);               // trailing partial byte
  if (bn > 0) flush();
  out.u8(0);                                   // block terminator
}

/* --------------------------------------------------------------- encode ---- */
/* encode({ width, height, fps, frames, palette, loop }) -> Uint8Array
     palette  up to 256 '#rrggbb' strings, becomes the global colour table
     frames   array of Uint8Array/Uint8ClampedArray, either RGBA (w*h*4) or
              palette indices (w*h); the length picks which
     fps      per-frame delay = max(2, round(100 / fps)) hundredths of a second.
              The floor of 2 is not pedantry: browsers silently rewrite a delay
              of 0 or 1 to 10, so anything faster than 50 fps must clamp here.
     loop     0 (default) = forever, n > 0 = n extra passes, < 0 = play once
              (no NETSCAPE block at all) */
function encode(opt) {
  const width = opt.width | 0, height = opt.height | 0;
  const frames = opt.frames || [];
  const palette = opt.palette || [];
  if (width <= 0 || height <= 0) throw new Error('gifenc: bad dimensions');
  if (width > 65535 || height > 65535) throw new Error('gifenc: dimensions exceed the 16-bit GIF fields');
  if (!palette.length) throw new Error('gifenc: empty palette');
  if (palette.length > 256) throw new Error('gifenc: palette has ' + palette.length + ' entries, GIF allows 256');

  const fps = opt.fps > 0 ? opt.fps : 12;
  const delay = Math.max(2, Math.round(100 / fps));
  const loop = opt.loop === undefined ? 0 : opt.loop;

  const np = palette.length;
  const pal = paletteBytes(palette);
  // the colour table must be a power of two, at least 2; `field` is what goes in
  // the packed byte, where the entry count is 1 << (field + 1)
  let entries = 2, field = 0;
  while (entries < np) { entries <<= 1; field++; }
  const gct = new Uint8Array(entries * 3);     // unused tail stays zeroed
  gct.set(pal);
  const minCodeSize = Math.max(2, field + 1);  // GIF forbids a 1-bit code size

  const out = new Sink(Math.max(1 << 16, width * height));

  out.ascii('GIF89a');
  out.u16(width); out.u16(height);
  out.u8(0xf0 | field);        // GCT present | 8-bit colour resolution | size
  out.u8(0);                   // background colour index
  out.u8(0);                   // pixel aspect ratio: none given
  out.raw(gct);

  if (loop >= 0) {             // NETSCAPE2.0 application extension
    out.u8(0x21); out.u8(0xff); out.u8(0x0b);
    out.ascii('NETSCAPE2.0');
    out.u8(0x03); out.u8(0x01); out.u16(loop); out.u8(0);
  }

  const cache = newCache(pal, np);
  for (let f = 0; f < frames.length; f++) {
    const idx = indexFrame(frames[f], width, height, palette, cache);

    out.u8(0x21); out.u8(0xf9); out.u8(0x04);  // graphic control extension
    out.u8(0x04);              // disposal 1 (leave in place), no transparency
    out.u16(delay);
    out.u8(0);                 // transparent colour index (unused)
    out.u8(0);

    out.u8(0x2c);              // image descriptor
    out.u16(0); out.u16(0); out.u16(width); out.u16(height);
    out.u8(0);                 // no local colour table, not interlaced

    out.u8(minCodeSize);
    lzw(idx, minCodeSize, out);
  }

  out.u8(0x3b);                // trailer
  return out.done();
}

return { encode, indexFrame, hexRGB, paletteBytes, nearest };
})();

globalThis.GifEnc = GifEnc;
if (typeof module !== 'undefined') module.exports = GifEnc;
