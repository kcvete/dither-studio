/* ---------------------------------------------------------------------------
   THE WAY OUT.

   The hosted page is the browser engine and nothing else: no server to talk to,
   so tracking runs on WebGPU in the tab at roughly 12 fps. There is a faster
   tier, it is free, and it is a git clone away -- but the page had no way of
   saying so, and a visitor who never reads a README would never find out.

   This file is that one sentence, and it is deliberately a separate file with a
   separate <script> tag: it owns its own markup and its own styles, touches
   nothing app.js touches, and deleting the tag deletes the feature. It runs on
   DOMContentLoaded, which is after the deferred module has built the panel, so
   it only ever appends to a finished DOM.

   It stays quiet where it would be noise: on localhost you are already running
   from a checkout, and if the engine chip says a server answered, you already
   have the fast path.
--------------------------------------------------------------------------- */
'use strict';

(() => {
  const REPO = 'https://github.com/kcvete/dither-studio';
  /* The one place the URL is written. app.js's slow-tracking hint reads it
     from here rather than carrying a second copy -- published BEFORE the
     localhost early-return, because the hint is about the engine, not about
     where the page is served from. */
  window.DV_REPO = REPO;
  const local = /^(localhost|127\.0\.0\.1|\[::1\])$/.test(location.hostname)
    || location.protocol === 'file:';
  if (local) return;

  const css = `
.oss-out{margin:18px 0 4px;padding-top:12px;border-top:1px solid var(--line);
  font-size:10px;line-height:1.5;opacity:.45;transition:opacity .15s}
.oss-out:hover{opacity:.8}
.oss-out a{color:inherit;text-decoration:none;border-bottom:1px solid currentColor}
.oss-out b{display:block;font-size:9px;letter-spacing:.12em;text-transform:uppercase;
  font-weight:600;opacity:.8;margin-bottom:4px}
@media (max-width:820px){.oss-out{margin-bottom:84px}}`;

  const mount = () => {
    const panel = document.getElementById('panel');
    if (!panel || panel.querySelector('.oss-out')) return;

    const style = document.createElement('style');
    style.textContent = css;
    document.head.appendChild(style);

    const box = document.createElement('div');
    box.className = 'oss-out';
    box.innerHTML =
      '<b>Faster?</b>Run it on your machine — same page, a local accelerator '
      + 'behind it, about twice the tracking speed and MP4 instead of WebM. '
      + '<a href="' + REPO + '" target="_blank" rel="noopener">'
      + 'github.com/kcvete/dither-studio</a>';
    panel.appendChild(box);
  };

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', mount, { once: true });
  } else {
    mount();
  }
})();
