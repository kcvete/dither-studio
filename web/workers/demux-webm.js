/* ---------------------------------------------------------------------------
   A minimal WebM/Matroska demuxer — enough to hand VideoDecoder a sample list.

   The reason this exists is the camera. A recording made in the tab is a
   MediaRecorder WebM: VP8 or VP9, no Cues, no Duration in the header and no
   SeekHead worth following, which is exactly the file the <video> seek path is
   slowest on (every seek is a scan from the last cluster it knows about).
   Walking the clusters once, in order, is both faster and simpler.

   Scope, deliberately: EBML, Info, Tracks, and SimpleBlock/BlockGroup inside
   Clusters. No lacing (MediaRecorder writes none for video), no encryption, no
   subtitles. Anything unexpected throws and the caller falls back.

   Written for this project; no third-party demuxer is vendored (see NOTICE).
--------------------------------------------------------------------------- */
'use strict';

/* Element ids, kept as the numbers they are on the wire (marker bits and all)
 * so a read of the spec lines up with a read of this file. */
const ID = {
  EBML: 0x1A45DFA3,
  Segment: 0x18538067,
  Info: 0x1549A966,
  TimestampScale: 0x2AD7B1,
  Duration: 0x4489,
  Tracks: 0x1654AE6B,
  TrackEntry: 0xAE,
  TrackNumber: 0xD7,
  TrackType: 0x83,
  CodecID: 0x86,
  CodecPrivate: 0x63A2,
  Video: 0xE0,
  PixelWidth: 0xB0,
  PixelHeight: 0xBA,
  Cluster: 0x1F43B675,
  Timestamp: 0xE7,
  SimpleBlock: 0xA3,
  BlockGroup: 0xA0,
  Block: 0xA1,
  ReferenceBlock: 0xFB,
};
/* A MediaRecorder file is written as it is captured, so its Segment — and in
 * some builds every Cluster — carries the "unknown size" vint. Such an element
 * ends where the first element that cannot be its child begins, so the only
 * way to find its end is to know what its children are allowed to be. These
 * two are the only unknown-size masters that occur in practice. */
const CHILDREN = {
  [ID.Segment]: new Set([ID.Info, ID.Tracks, ID.Cluster,
    0x114D9B74 /* SeekHead */, 0x1C53BB6B /* Cues */, 0x1254C367 /* Tags */,
    0x1941A469 /* Attachments */, 0x1043A770 /* Chapters */,
    0xEC /* Void */, 0xBF /* CRC-32 */]),
  [ID.Cluster]: new Set([ID.Timestamp, ID.SimpleBlock, ID.BlockGroup,
    0xA7 /* Position */, 0xAB /* PrevSize */, 0xAF /* EncryptedBlock */,
    0xEC, 0xBF]),
};

/** Where an unknown-size master ends: the first child id that is not one of
 *  its own. Recurses, because an unknown-size Cluster can sit inside an
 *  unknown-size Segment. */
function unknownEnd(b, start, limit, id) {
  const allow = CHILDREN[id];
  if (!allow) return limit;
  let p = start;
  while (p < limit) {
    const [cid, idn] = vint(b, p, true);
    if (idn === 0) return limit;
    if (!allow.has(cid)) return p;
    const [size, sn] = vint(b, p + idn, false);
    if (sn === 0) return limit;
    p = size < 0 ? unknownEnd(b, p + idn + sn, limit, cid)
      : p + idn + sn + size;
  }
  return limit;
}

/** An EBML variable-length integer. `keepMarker` is what an ID needs and a
 *  size does not. Returns [value, bytesRead]. */
function vint(b, p, keepMarker) {
  const first = b[p];
  if (first === undefined) return [-1, 0];
  let len = 1;
  for (let m = 0x80; m && !(first & m); m >>= 1) len++;
  if (len > 8 || p + len > b.length) return [-1, 0];
  let v = keepMarker ? first : (first & (0xff >> len));
  let unknown = !keepMarker && (first & (0xff >> len)) === (0xff >> len);
  for (let i = 1; i < len; i++) {
    v = v * 256 + b[p + i];
    if (!keepMarker && b[p + i] !== 0xff) unknown = false;
  }
  return [unknown ? -1 : v, len];
}

const uint = (b, s, e) => { let v = 0; for (let i = s; i < e; i++) v = v * 256 + b[i]; return v; };
const flt = (b, s, e) => {
  const d = new DataView(b.buffer, b.byteOffset + s, e - s);
  return e - s === 4 ? d.getFloat32(0) : e - s === 8 ? d.getFloat64(0) : 0;
};
const str = (b, s, e) => {
  let o = '';
  for (let i = s; i < e && b[i]; i++) o += String.fromCharCode(b[i]);
  return o;
};

/** Every element between `start` and `end`, one level deep. */
function* elements(b, start, end) {
  let p = start;
  while (p < end) {
    const [id, idn] = vint(b, p, true);
    if (idn === 0) return;
    const [size, sn] = vint(b, p + idn, false);
    if (sn === 0) return;
    const body = p + idn + sn;
    const stop = size < 0 ? unknownEnd(b, body, end, id)
      : Math.min(end, body + size);
    yield { id, body, end: stop };
    p = Math.max(stop, body);
  }
}

/* ================================================================ entry === */

export function demuxWebM(bytes) {
  const b = bytes;
  let scale = 1e6;                       // ns per timestamp tick
  let track = -1, codecId = '', priv = null, width = 0, height = 0;
  const samples = [];

  const seg = [...elements(b, 0, b.length)].find((e) => e.id === ID.Segment);
  if (!seg) throw new Error('webm: no Segment');

  for (const e of elements(b, seg.body, seg.end)) {
    if (e.id === ID.Info) {
      for (const i of elements(b, e.body, e.end)) {
        if (i.id === ID.TimestampScale) scale = uint(b, i.body, i.end) || 1e6;
      }
    } else if (e.id === ID.Tracks && track < 0) {
      for (const t of elements(b, e.body, e.end)) {
        if (t.id !== ID.TrackEntry) continue;
        let num = -1, type = 0, cid = '', pv = null, w = 0, h = 0;
        for (const f of elements(b, t.body, t.end)) {
          if (f.id === ID.TrackNumber) num = uint(b, f.body, f.end);
          else if (f.id === ID.TrackType) type = uint(b, f.body, f.end);
          else if (f.id === ID.CodecID) cid = str(b, f.body, f.end);
          else if (f.id === ID.CodecPrivate) pv = b.slice(f.body, f.end);
          else if (f.id === ID.Video) {
            for (const g of elements(b, f.body, f.end)) {
              if (g.id === ID.PixelWidth) w = uint(b, g.body, g.end);
              else if (g.id === ID.PixelHeight) h = uint(b, g.body, g.end);
            }
          }
        }
        if (type === 1 && num >= 0) {
          track = num; codecId = cid; priv = pv; width = w; height = h; break;
        }
      }
    } else if (e.id === ID.Cluster) {
      let ts = 0;
      for (const c of elements(b, e.body, e.end)) {
        if (c.id === ID.Timestamp) { ts = uint(b, c.body, c.end); continue; }
        let blk = null, key = null;
        if (c.id === ID.SimpleBlock) blk = c;
        else if (c.id === ID.BlockGroup) {
          key = true;                     // no ReferenceBlock means a keyframe
          for (const g of elements(b, c.body, c.end)) {
            if (g.id === ID.Block) blk = g;
            else if (g.id === ID.ReferenceBlock) key = false;
          }
        }
        if (!blk) continue;
        const [num, n] = vint(b, blk.body, false);
        if (n === 0 || num !== track) continue;
        const rel = (b[blk.body + n] << 24 >> 16) | b[blk.body + n + 1];  // int16
        const flags = b[blk.body + n + 2];
        if (flags & 0x06) throw new Error('webm: laced video blocks');
        const off = blk.body + n + 3;
        samples.push({ offset: off, size: blk.end - off,
                       key: key === null ? !!(flags & 0x80) : key,
                       pts: (ts + rel) * scale / 1e9, dur: 0 });
      }
    }
  }
  if (track < 0) throw new Error('webm: no video track');
  if (!samples.length) throw new Error('webm: no video blocks');

  // blocks stay in the order they were written, which is decode order — a
  // sample list is fed to VideoDecoder, not displayed. Durations are measured
  // in presentation order and written back.
  const byTime = samples.map((s, i) => i).sort((a, b2) => samples[a].pts - samples[b2].pts);
  // as in the MP4 demuxer: the container's clock is the one <video> exposes,
  // so a late start is kept and only a negative one is moved
  const base = samples[byTime[0]].pts;
  if (base < 0) for (const s of samples) s.pts -= base;
  for (let k = 0; k < byTime.length; k++) {
    const s = samples[byTime[k]];
    s.dur = k + 1 < byTime.length ? samples[byTime[k + 1]].pts - s.pts : 0;
  }
  // the last block has no successor to measure against: give it the median of
  // the ones that do, so a duration derived from this table is not short
  if (byTime.length > 1) {
    const d = byTime.slice(0, -1).map((i) => samples[i].dur).sort((x, y) => x - y);
    samples[byTime[byTime.length - 1]].dur = d[d.length >> 1];
  } else samples[0].dur = 1 / 30;

  return Object.assign({ container: 'webm', samples, width, height },
                       codecOf(codecId, priv, width, height));
}

/** WebM names its codecs; WebCodecs wants the MIME-ish string. VP8 and VP9
 *  carry no configuration record, so the string is all the decoder gets — and
 *  a VP9 profile is not in the container at all, which is why this reports the
 *  8-bit 4:2:0 profile every MediaRecorder build writes. */
function codecOf(id, priv, w, h) {
  if (id === 'V_VP8') return { codec: 'vp8', description: null };
  if (id === 'V_VP9') return { codec: 'vp09.00.10.08', description: null };
  if (id === 'V_AV1') {
    if (priv && priv.length >= 4) {
      const profile = (priv[1] >> 5) & 7, level = priv[1] & 31;
      const tier = (priv[2] >> 7) & 1, high = (priv[2] >> 6) & 1,
        twelve = (priv[2] >> 5) & 1;
      const depth = high ? (twelve ? 12 : 10) : 8;
      return { codec: `av01.${profile}.${String(level).padStart(2, '0')}`
        + `${tier ? 'H' : 'M'}.${String(depth).padStart(2, '0')}`,
        description: null };
    }
    return { codec: 'av01.0.04M.08', description: null };
  }
  if (id === 'V_MPEG4/ISO/AVC') {
    if (!priv || priv.length < 4) throw new Error('webm: avc without CodecPrivate');
    const hex = (n) => n.toString(16).padStart(2, '0');
    return { codec: 'avc1.' + hex(priv[1]) + hex(priv[2]) + hex(priv[3]),
             description: priv };
  }
  throw new Error('webm: ' + id + ' is not a codec this path decodes');
}

export function looksLikeWebM(bytes) {
  return bytes.length > 4 && bytes[0] === 0x1A && bytes[1] === 0x45
    && bytes[2] === 0xDF && bytes[3] === 0xA3;
}
