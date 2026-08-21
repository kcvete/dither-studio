## What changed, and why

<!-- One paragraph. The why matters more than the what — it usually belongs in
     the file's block comment too. -->

## Suites run

- [ ] `env/venv/bin/python server/parity.py` and `GATE=1 env/venv/bin/python server/parity.py`
- [ ] `env/venv/bin/python server/jobsgc_check.py`
- [ ] `npm test` (syntax over every `.js`/`.mjs`, plus the `.dots.gz` round-trip)
- [ ] `node verify-web.mjs` (browser engine — required if the tab runs any of this)
- [ ] `node verify.mjs` (server engine — required if the server, renderer or job lifecycle changed)

Paste what they printed. Zero console errors is a hard gate.

## Checklist

- [ ] No new runtime dependency, and nothing fetched from a CDN at run time
- [ ] `web/dither.js` / `server/dither.py` (and `polish.js` / `polish.py`) still agree byte for byte
- [ ] Screenshots attached for any UI change
- [ ] A feature that needs a server says so in the UI rather than hiding the button
- [ ] No model weights, `env/` or `jobs/` contents committed

<!-- Tracking-speed changes: attach a bench/bench.py run, interleaved, three rounds. -->
