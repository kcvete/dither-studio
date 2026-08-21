/* ---------------------------------------------------------------------------
   A minimal MP4/MOV demuxer — enough to hand VideoDecoder a sample table.

   Not a general container library. It reads exactly what WebCodecs needs and
   nothing else: which video track, what codec string and description, and for
   every video sample its byte range, its presentation time and whether it is a
   sync sample. Both layouts are handled — the ordinary `moov` + `stbl` table
   and the fragmented `moof` + `trun` runs a MediaRecorder-MP4 or a streaming
   export writes — because a file the user drags in can be either.

   Anything it cannot make sense of throws, and the caller falls back to the
   <video> seek path. That is the whole error strategy: never guess.

   Written for this project; no third-party demuxer is vendored (see NOTICE).
--------------------------------------------------------------------------- */
'use strict';

const typ = (v, o) => String.fromCharCode(v.getUint8(o), v.getUint8(o + 1),
                                          v.getUint8(o + 2), v.getUint8(o + 3));
const hex2 = (n) => n.toString(16).padStart(2, '0');

/** Every box between `start` and `end`, without recursing. */
function* boxes(v, start, end) {
  let p = start;
  while (p + 8 <= end) {
    let size = v.getUint32(p);
    const t = typ(v, p + 4);
    let hdr = 8;
    if (size === 1) {
      if (p + 16 > end) return;
      size = Number(v.getBigUint64(p + 8)); hdr = 16;
    } else if (size === 0) {
      size = end - p;
    }
    if (size < hdr || p + size > end) size = end - p;   // truncated tail
    if (size < hdr) return;
    yield { type: t, start: p, body: p + hdr, end: p + size };
    p += size;
  }
}
const kid = (v, b, t) => { for (const c of boxes(v, b.body, b.end)) if (c.type === t) return c; return null; };
const kids = (v, b, t) => { const o = []; for (const c of boxes(v, b.body, b.end)) if (c.type === t) o.push(c); return o; };
/** Walk a path of box types from a parent, e.g. path(v, trak, 'mdia/minf/stbl'). */
function path(v, b, p) {
  let cur = b;
  for (const t of p.split('/')) { cur = kid(v, cur, t); if (!cur) return null; }
  return cur;
}

/* ------------------------------------------------------------- codecs ---
 * The codec string has to be exact or `VideoDecoder.isConfigSupported` says no,
 * which is the signal to fall back rather than a reason to fudge it. */

function avcCodec(v, c) {                    // AVCDecoderConfigurationRecord
  return 'avc1.' + hex2(v.getUint8(c.body + 1)) + hex2(v.getUint8(c.body + 2))
    + hex2(v.getUint8(c.body + 3));
}

function hevcCodec(v, c) {                   // HEVCDecoderConfigurationRecord
  const o = c.body;
  const b1 = v.getUint8(o + 1);
  const space = (b1 >> 6) & 3, tier = (b1 >> 5) & 1, profile = b1 & 31;
  // the 32 compatibility flags are stored MSB-first and written LSB-first
  let compat = v.getUint32(o + 2), rev = 0;
  for (let i = 0; i < 32; i++) { rev = (rev << 1) | (compat & 1); compat >>>= 1; }
  const constraints = [];
  for (let i = 0; i < 6; i++) constraints.push(v.getUint8(o + 6 + i));
  while (constraints.length && constraints[constraints.length - 1] === 0) constraints.pop();
  const level = v.getUint8(o + 12);
  return 'hvc1.' + (space ? String.fromCharCode(64 + space) : '') + profile
    + '.' + (rev >>> 0).toString(16).toUpperCase()
    + '.' + (tier ? 'H' : 'L') + level
    + (constraints.length ? '.' + constraints.map(hex2).join('.') : '');
}

function av1Codec(v, c) {                    // AV1CodecConfigurationRecord
  const o = c.body;
  const b1 = v.getUint8(o + 1), b2 = v.getUint8(o + 2);
  const profile = (b1 >> 5) & 7, level = b1 & 31;
  const tier = (b2 >> 7) & 1, high = (b2 >> 6) & 1, twelve = (b2 >> 5) & 1;
  const depth = high ? (twelve ? 12 : 10) : 8;
  return `av01.${profile}.${String(level).padStart(2, '0')}${tier ? 'H' : 'M'}`
    + `.${String(depth).padStart(2, '0')}`;
}

function vpCodec(v, c) {                     // VPCodecConfigurationRecord
  const o = c.body + 4;                      // version + flags
  const profile = v.getUint8(o), level = v.getUint8(o + 1);
  const depth = v.getUint8(o + 2) >> 4;
  return `vp09.${String(profile).padStart(2, '0')}.${String(level).padStart(2, '0')}`
    + `.${String(depth).padStart(2, '0')}`;
}

/** The stsd entry for a video track -> {codec, description}. */
function sampleEntry(v, stsd, bytes) {
  const n = v.getUint32(stsd.body + 4);
  if (!n) throw new Error('mp4: the sample description is empty');
  const first = boxes(v, stsd.body + 8, stsd.end).next().value;
  if (!first) throw new Error('mp4: no sample entry');
  // VisualSampleEntry: 78 bytes of fixed fields before the child boxes
  const inner = { type: first.type, body: first.body + 78, end: first.end };
  const grab = (b) => bytes.slice(b.body, b.end);
  const cfg = (t) => kid(v, inner, t);
  switch (first.type) {
    case 'avc1': case 'avc3': {
      const c = cfg('avcC'); if (!c) throw new Error('mp4: avc1 without avcC');
      return { codec: avcCodec(v, c), description: grab(c) };
    }
    case 'hvc1': case 'hev1': {
      const c = cfg('hvcC'); if (!c) throw new Error('mp4: hvc1 without hvcC');
      return { codec: hevcCodec(v, c), description: grab(c) };
    }
    case 'av01': {
      const c = cfg('av1C'); if (!c) throw new Error('mp4: av01 without av1C');
      return { codec: av1Codec(v, c), description: grab(c) };
    }
    case 'vp08': case 'vp09': {
      const c = cfg('vpcC');
      return { codec: c ? vpCodec(v, c) : 'vp09.00.10.08', description: null };
    }
    default:
      throw new Error('mp4: ' + first.type + ' is not a codec this path decodes');
  }
}

/* ----------------------------------------------------- the sample table --- */

function stblSamples(v, stbl, timescale) {
  const stts = kid(v, stbl, 'stts'), stsc = kid(v, stbl, 'stsc');
  const stsz = kid(v, stbl, 'stsz') || kid(v, stbl, 'stz2');
  const stco = kid(v, stbl, 'stco'), co64 = kid(v, stbl, 'co64');
  const ctts = kid(v, stbl, 'ctts'), stss = kid(v, stbl, 'stss');
  if (!stts || !stsc || !stsz || !(stco || co64)) {
    throw new Error('mp4: the sample table is incomplete');
  }

  // sizes
  const sizes = [];
  const one = v.getUint32(stsz.body + 4);
  const count = v.getUint32(stsz.body + 8);
  if (stsz.type === 'stsz' && one) { for (let i = 0; i < count; i++) sizes.push(one); }
  else for (let i = 0; i < count; i++) sizes.push(v.getUint32(stsz.body + 12 + i * 4));

  // decode times
  const dts = new Float64Array(count);
  const durs = new Float64Array(count);
  {
    const n = v.getUint32(stts.body + 4);
    let s = 0, t = 0;
    for (let e = 0; e < n && s < count; e++) {
      const c = v.getUint32(stts.body + 8 + e * 8);
      const d = v.getUint32(stts.body + 12 + e * 8);
      for (let k = 0; k < c && s < count; k++) { dts[s] = t; durs[s] = d; t += d; s++; }
    }
    while (s < count) { dts[s] = t; durs[s] = durs[s - 1] || 0; t += durs[s]; s++; }
  }

  // composition offsets
  const cts = new Float64Array(count);
  if (ctts) {
    const n = v.getUint32(ctts.body + 4);
    const signed = v.getUint8(ctts.body) === 1;
    let s = 0;
    for (let e = 0; e < n && s < count; e++) {
      const c = v.getUint32(ctts.body + 8 + e * 8);
      const o = signed ? v.getInt32(ctts.body + 12 + e * 8)
        : v.getUint32(ctts.body + 12 + e * 8);
      for (let k = 0; k < c && s < count; k++) cts[s++] = o;
    }
  }

  // chunk offsets, then sample offsets through stsc
  const nChunks = v.getUint32((co64 || stco).body + 4);
  const chunkOff = new Float64Array(nChunks);
  for (let i = 0; i < nChunks; i++) {
    chunkOff[i] = co64 ? Number(v.getBigUint64(co64.body + 8 + i * 8))
      : v.getUint32(stco.body + 8 + i * 4);
  }
  const nRuns = v.getUint32(stsc.body + 4);
  const runs = [];
  for (let i = 0; i < nRuns; i++) {
    runs.push({ first: v.getUint32(stsc.body + 8 + i * 12) - 1,
                per: v.getUint32(stsc.body + 12 + i * 12) });
  }
  const offsets = new Float64Array(count);
  {
    let s = 0;
    for (let r = 0; r < runs.length && s < count; r++) {
      const last = r + 1 < runs.length ? runs[r + 1].first : nChunks;
      for (let c = runs[r].first; c < last && s < count; c++) {
        let o = chunkOff[c];
        for (let k = 0; k < runs[r].per && s < count; k++) { offsets[s] = o; o += sizes[s]; s++; }
      }
    }
    if (s < count) throw new Error('mp4: the chunk map does not cover every sample');
  }

  // sync samples: no stss means every sample is one
  const key = new Uint8Array(count);
  if (stss) {
    const n = v.getUint32(stss.body + 4);
    for (let i = 0; i < n; i++) {
      const s = v.getUint32(stss.body + 8 + i * 4) - 1;
      if (s >= 0 && s < count) key[s] = 1;
    }
  } else key.fill(1);

  // DECODE order, which is the order they must be fed to VideoDecoder in.
  // `ticks` is the presentation time; with B-frames the two disagree, and
  // sorting here is exactly how you get "Decoding error" out of a decoder.
  const out = new Array(count);
  for (let i = 0; i < count; i++) {
    out[i] = { offset: offsets[i], size: sizes[i], key: !!key[i],
               ticks: dts[i] + cts[i], dur: durs[i] };
  }
  return out;
}

/* ------------------------------------------------------ fragmented mp4 --- */

function moofSamples(v, end, trackId, defaults) {
  const out = [];
  let base = 0;
  for (const b of boxes(v, 0, end)) {
    if (b.type !== 'moof') continue;
    for (const traf of kids(v, b, 'traf')) {
      const tfhd = kid(v, traf, 'tfhd');
      if (!tfhd) continue;
      const fl = (v.getUint8(tfhd.body + 1) << 16) | v.getUint16(tfhd.body + 2);
      const id = v.getUint32(tfhd.body + 4);
      if (id !== trackId) continue;
      let p = tfhd.body + 8;
      let dataBase = b.start;                          // default-base-is-moof
      if (fl & 0x01) { dataBase = Number(v.getBigUint64(p)); p += 8; }
      if (fl & 0x02) p += 4;                            // sample-description-index
      let dDur = defaults.dur, dSize = defaults.size, dFlags = defaults.flags;
      if (fl & 0x08) { dDur = v.getUint32(p); p += 4; }
      if (fl & 0x10) { dSize = v.getUint32(p); p += 4; }
      if (fl & 0x20) { dFlags = v.getUint32(p); p += 4; }

      const tfdt = kid(v, traf, 'tfdt');
      let t = base;
      if (tfdt) {
        t = v.getUint8(tfdt.body) === 1 ? Number(v.getBigUint64(tfdt.body + 4))
          : v.getUint32(tfdt.body + 4);
      }
      for (const trun of kids(v, traf, 'trun')) {
        const tf = (v.getUint8(trun.body + 1) << 16) | v.getUint16(trun.body + 2);
        const n = v.getUint32(trun.body + 4);
        let q = trun.body + 8;
        let off = dataBase;
        if (tf & 0x0001) { off = dataBase + v.getInt32(q); q += 4; }
        let firstFlags = null;
        if (tf & 0x0004) { firstFlags = v.getUint32(q); q += 4; }
        for (let i = 0; i < n; i++) {
          let dur = dDur, size = dSize, flags = i === 0 && firstFlags !== null
            ? firstFlags : dFlags, cto = 0;
          if (tf & 0x0100) { dur = v.getUint32(q); q += 4; }
          if (tf & 0x0200) { size = v.getUint32(q); q += 4; }
          if (tf & 0x0400) { flags = v.getUint32(q); q += 4; }
          if (tf & 0x0800) { cto = v.getInt32(q); q += 4; }
          const nonSync = (flags >> 16) & 1;
          out.push({ offset: off, size, key: !nonSync, ticks: t + cto, dur });
          off += size; t += dur;
        }
      }
      base = t;
    }
  }
  return out;
}

/* ------------------------------------------------------------ edit list ---
 * `<video>` honours elst, so a demuxer that does not would index a different
 * frame as number 42 than the seek path does. Two cases matter and both are
 * common: an initial empty edit that delays the media, and a media_time offset
 * that trims the reordering lead-in off the front. */
function editShift(v, trak, mvhdTs, mdhdTs) {
  const elst = path(v, trak, 'edts/elst');
  if (!elst) return 0;
  const ver = v.getUint8(elst.body);
  const n = v.getUint32(elst.body + 4);
  let p = elst.body + 8, shift = 0;
  for (let i = 0; i < n; i++) {
    const wide = ver === 1;
    const segDur = wide ? Number(v.getBigUint64(p)) : v.getUint32(p);
    const mtime = wide ? Number(v.getBigInt64(p + 8)) : v.getInt32(p + 4);
    p += wide ? 20 : 12;
    if (mtime < 0) shift += segDur / mvhdTs;            // an empty edit: a delay
    else { shift -= mtime / mdhdTs; break; }            // the first real edit
  }
  return shift;
}

/* ================================================================ entry === */

/** `bytes` is the whole file. Returns the video track as a flat sample list
 *  with presentation times in seconds. Throws if this is not a file the
 *  WebCodecs path can take. */
export function demuxMp4(bytes) {
  const v = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const top = [...boxes(v, 0, bytes.byteLength)];
  const moov = top.find((b) => b.type === 'moov');
  if (!moov) throw new Error('mp4: no moov');
  const mvhd = kid(v, moov, 'mvhd');
  const mvhdTs = mvhd ? (v.getUint8(mvhd.body) === 1 ? v.getUint32(mvhd.body + 20)
    : v.getUint32(mvhd.body + 12)) : 1000;

  let trak = null, hdlrType = '';
  for (const t of kids(v, moov, 'trak')) {
    const hdlr = path(v, t, 'mdia/hdlr');
    if (hdlr && typ(v, hdlr.body + 8) === 'vide') { trak = t; hdlrType = 'vide'; break; }
  }
  if (!trak) throw new Error('mp4: no video track');

  const tkhd = kid(v, trak, 'tkhd');
  const tkhdV = tkhd ? v.getUint8(tkhd.body) : 0;
  const trackId = tkhd ? v.getUint32(tkhd.body + (tkhdV === 1 ? 20 : 12)) : 1;
  const mdhd = path(v, trak, 'mdia/mdhd');
  if (!mdhd) throw new Error('mp4: no mdhd');
  const mdhdV = v.getUint8(mdhd.body);
  const timescale = mdhdV === 1 ? v.getUint32(mdhd.body + 20) : v.getUint32(mdhd.body + 12);
  if (!timescale) throw new Error('mp4: zero timescale');

  const stbl = path(v, trak, 'mdia/minf/stbl');
  if (!stbl) throw new Error('mp4: no stbl');
  const { codec, description } = sampleEntry(v, kid(v, stbl, 'stsd'), bytes);

  let raw = stblSamples(v, stbl, timescale);
  if (!raw.length) {
    // fragmented: the defaults live in mvex/trex, the runs in every moof
    const trex = (kid(v, moov, 'mvex') ? kids(v, kid(v, moov, 'mvex'), 'trex') : [])
      .find((b) => v.getUint32(b.body + 4) === trackId);
    const defaults = trex
      ? { dur: v.getUint32(trex.body + 12), size: v.getUint32(trex.body + 16),
          flags: v.getUint32(trex.body + 20) }
      : { dur: 0, size: 0, flags: 0 };
    raw = moofSamples(v, bytes.byteLength, trackId, defaults);
  }
  if (!raw.length) throw new Error('mp4: the video track has no samples');

  const shift = editShift(v, trak, mvhdTs || 1000, timescale);
  const first = raw[0].ticks;
  const samples = raw.map((s) => ({
    offset: s.offset, size: s.size, key: s.key,
    pts: s.ticks / timescale + shift,
    dur: (s.dur || 0) / timescale,
  }));
  // A media timeline that starts late stays late. `<video>.currentTime` is the
  // container's own clock, not a clock rebased on the first picture, so a clip
  // whose first frame is at 0.033 s answers a seek to 0.05 s with that same
  // first frame — and rebasing here would answer with the second one. Measured
  // against docs/entry-clip.mp4, which starts at 0.033008 s exactly to make
  // this visible. Only a NEGATIVE start is moved, because there is no such
  // thing as a picture before zero.
  let base = Infinity;
  for (const s of samples) if (s.pts < base) base = s.pts;
  if (base < 0) for (const s of samples) s.pts -= base;

  const w = tkhd ? v.getUint32(tkhd.body + (tkhdV === 1 ? 88 : 76)) / 65536 : 0;
  const h = tkhd ? v.getUint32(tkhd.body + (tkhdV === 1 ? 92 : 80)) / 65536 : 0;
  return { container: 'mp4', codec, description, samples,
           width: Math.round(w), height: Math.round(h), track: hdlrType };
}

export function looksLikeMp4(bytes) {
  if (bytes.byteLength < 12) return false;
  const v = new DataView(bytes.buffer, bytes.byteOffset, 12);
  const t = typ(v, 4);
  return t === 'ftyp' || t === 'styp' || t === 'moov' || t === 'skip' || t === 'free';
}
