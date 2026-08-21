/* ---------------------------------------------------------------------------
   SYNTAX CHECK — parses every hand-written JavaScript file in the repo without
   running any of it. No DOM, no network, no dependencies: it is the cheapest
   possible answer to "did somebody push a file that will not even parse".

   The repo mixes two module systems on purpose, and the check has to know
   which is which, because `node --check` picks its parser from the file name
   and from the nearest package.json.

     classic scripts (.js, CommonJS)   `node --check <file>`
       dither.js and polish.js are plain <script src> tags in index.html and
       publish onto `globalThis` and, under Node, `module.exports` — that is
       how server/parity.mjs reaches them, and it is the only reason the root
       package.json says "type": "commonjs". vendor/gifenc.js and
       player/dither-player.js are the same shape.

     ES modules (.js, loaded by the browser as <script type="module">)
       `node --check --input-type=module < <file>`
       app.js, canvas.js, track.js and engines/*.js use import/export. Their
       extension is .js, so --check alone parses them as scripts and chokes on
       the first `export`. Node 22 would normally sniff the source and retry as
       a module, but declaring "type" in package.json switches that sniffing
       off — an explicit type is an explicit answer — so the split below has to
       be written down rather than guessed. Feeding the source on stdin with
       --input-type=module is the one incantation that hands the module parser
       a plain .js file, and it still exits 1 on a real syntax error: verified,
       not assumed.

     .mjs files parse as modules from their extension alone.

   A missing file is a failure, not a skip: renaming a file out from under this
   list should turn CI red rather than quietly shrink the coverage.

   web/vendor/webm-muxer.js and web/vendor/mp4-muxer.js are NOT here. They are
   third-party builds vendored verbatim from npm (see NOTICE); this check is
   for code written in this repository, and a minified upstream bundle failing
   to parse would be a supply-chain problem, not a typo.
--------------------------------------------------------------------------- */
import { spawnSync } from 'node:child_process';
import { existsSync, readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import path from 'node:path';

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');

/* classic scripts — parsed as CommonJS */
const SCRIPTS = [
  'web/dither.js',
  'web/polish.js',
  'web/vendor/gifenc.js',
  'web/player/dither-player.js',
  'web/oss-link.js',
];

/* ES modules — .js files the browser loads as modules, plus real .mjs */
const MODULES = [
  'web/app.js',
  'web/canvas.js',
  'web/track.js',
  'web/engines/index.js',
  'web/engines/browser.js',
  'web/engines/decode.js',
  'web/engines/encode.js',
  'web/engines/remote.js',
  'web/workers/decode-core.js',
  'web/workers/decode-worker.js',
  'web/workers/demux-mp4.js',
  'web/workers/demux-webm.js',
  'web/player/dither-player.mjs',
  'server/parity.mjs',
];

let failed = 0;
const fail = (rel, why) => { failed++; console.error(`FAIL  ${rel}\n      ${why}`); };

for (const rel of [...SCRIPTS, ...MODULES]) {
  const abs = path.join(ROOT, rel);
  if (!existsSync(abs)) { fail(rel, 'file does not exist (renamed? update tests/syntax-check.mjs)'); continue; }

  const isModule = MODULES.includes(rel);
  const r = isModule
    ? spawnSync(process.execPath, ['--check', '--input-type=module'],
                { input: readFileSync(abs), encoding: 'utf8' })
    : spawnSync(process.execPath, ['--check', abs], { encoding: 'utf8' });

  if (r.status === 0) console.log(`ok    ${rel}  (${isModule ? 'module' : 'script'})`);
  else fail(rel, (r.stderr || r.stdout || '').trim().split('\n').slice(0, 4).join('\n      '));
}

const n = SCRIPTS.length + MODULES.length;
if (failed) {
  console.error(`\nsyntax check: ${failed} of ${n} files failed to parse`);
  process.exit(1);
}
console.log(`\nsyntax check: ${n} files parse clean (${SCRIPTS.length} scripts, ${MODULES.length} modules) on node ${process.version}`);
