/* ---------------------------------------------------------------------------
   ENGINE SELECTION.

   Dither Studio has one page and two ways to run it:

     browser   everything in the tab (ONNX + WebGPU). Free, private, deployable
               to GitHub Pages, and the only tier that exists for most people.
     remote    the FastAPI accelerator in server/. Faster where it is available
               — 20.9 fps against 12.4 on the same Mac — and the same wire
               format whether it is on 127.0.0.1 or a rented GPU box.

   The rule: if the page can see a Dither Studio server, use it; otherwise run
   in the tab. `GET /api/meta` with a short timeout is the whole probe, and it
   is deliberately relative-first, so the page served BY the server always finds
   it and a page served from Pages never blocks on a request that cannot work.

   A saved preference beats the probe, so someone on this Mac can still choose
   to exercise the browser path, and someone with a paid backend can point at it
   without editing anything.
--------------------------------------------------------------------------- */
'use strict';

import { BrowserEngine, modelsMissing } from './browser.js';
import { RemoteEngine } from './remote.js';

export { BrowserEngine, RemoteEngine, modelsMissing };

const KEY = 'dither-studio.engine';

/** {mode:'auto'|'browser'|'local'|'custom', url, key, ep?, fp16?}
 *  `ep`/`fp16` only reach the browser engine, and exist so a machine with
 *  broken WebGPU (or a verifier pinning a backend) can force 'wasm'. */
export function loadPref() {
  try {
    const p = JSON.parse(localStorage.getItem(KEY) || 'null');
    if (p && typeof p === 'object' && p.mode) return p;
  } catch (e) { /* corrupt or blocked storage */ }
  return { mode: 'auto', url: '', key: '' };
}

export function savePref(p) {
  try { localStorage.setItem(KEY, JSON.stringify(p)); } catch (e) { /* private mode */ }
}

/** GET <base>/api/meta, bounded. Resolves to the payload or null. */
export async function probeRemote(baseUrl = '', apiKey = '', timeoutMs = 1500) {
  const ac = new AbortController();
  const bail = setTimeout(() => ac.abort(), timeoutMs);
  try {
    const h = apiKey ? { Authorization: 'Bearer ' + apiKey } : {};
    const r = await fetch(String(baseUrl).replace(/\/+$/, '') + '/api/meta',
                          { signal: ac.signal, headers: h, cache: 'no-store' });
    if (!r.ok) return { error: r.status === 401 ? 'needs an API key' : 'HTTP ' + r.status };
    const j = await r.json();
    return j && j.name === 'dither-studio' ? j : { error: 'not a Dither Studio server' };
  } catch (e) {
    return { error: e.name === 'AbortError' ? 'no answer' : 'unreachable' };
  } finally {
    clearTimeout(bail);
  }
}

const isLocalHost = () => /^(localhost|127\.0\.0\.1|\[::1\])$/.test(location.hostname);

/** Human labels for a probe result. */
function remoteLabels(meta, custom) {
  const dev = (meta.device || '').toUpperCase();
  const be = meta.backend && meta.backend !== 'auto' ? ' ' + meta.backend : '';
  return {
    label: custom ? 'Remote server' : 'Local server',
    sublabel: (custom ? '' : 'faster · ') + (dev + be).trim(),
  };
}

/**
 * Build the engine the page should run with.
 *
 * Returns {engine, pref, probe, tried} — `tried` is what the auto path found,
 * so the chip can explain itself instead of just asserting.
 */
export async function chooseEngine(pref = loadPref()) {
  const tried = [];

  const mkRemote = async (url, key, custom) => {
    const p = await probeRemote(url, key);
    tried.push({ url: url || '(this origin)', ok: !p.error, why: p.error });
    if (p.error) return null;
    const e = new RemoteEngine(Object.assign({ baseUrl: url, apiKey: key },
                                             remoteLabels(p, custom)));
    await e.init();
    return { engine: e, probe: p };
  };

  const mkBrowser = () => new BrowserEngine({
    ...(pref.ep ? { ep: pref.ep } : {}),
    ...(pref.fp16 === false ? { fp16: false } : {}),
  });

  if (pref.mode === 'browser') {
    return { engine: mkBrowser(), pref, probe: null, tried };
  }
  if (pref.mode === 'custom' && pref.url) {
    const r = await mkRemote(pref.url, pref.key || '', true);
    if (r) return Object.assign(r, { pref, tried });
    return { engine: mkBrowser(), pref, probe: null, tried,
             warn: `could not reach ${pref.url} — running in the browser instead` };
  }
  if (pref.mode === 'local') {
    const r = await mkRemote('', '', false);
    if (r) return Object.assign(r, { pref, tried });
    return { engine: mkBrowser(), pref, probe: null, tried,
             warn: 'no local server on this origin — running in the browser instead' };
  }

  // --- auto -------------------------------------------------------------
  // Same-origin first: that is the case that is both free to test and the one
  // that matters (`./run.sh` opened this page). Only a page on localhost bothers
  // trying the default port, because a public deployment probing 127.0.0.1
  // would be both useless and rude.
  let r = await mkRemote('', '', false);
  if (!r && isLocalHost() && location.port !== '8765') {
    r = await mkRemote(`${location.protocol}//${location.hostname}:8765`, '', false);
  }
  if (r) return Object.assign(r, { pref, tried });
  return { engine: mkBrowser(), pref, probe: null, tried };
}
