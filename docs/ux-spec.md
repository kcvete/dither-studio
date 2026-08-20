# Dither Studio — UX Specification (desktop web + mobile)

*Deep UX research, 2026-08-20. Method: drove every flow of the live app at
127.0.0.1:8765 headless (desktop 1440×900 and mobile 390×844 with touch),
measured the DOM, read `web/index.html` / `web/app.js` / `style.css`
structurally, extracted the verify-suite compatibility surface from
`verify.mjs` / `verify-web.mjs`, and ran competitive research on CapCut/TikTok,
Meta SAM 2, Runway, Figma/Photoshop web, Kapwing/Veed, Dithermark, Rive/Spline.
Working screenshots: `docs/ux-research/` (d· = desktop, m· = mobile).*

This spec is written for the implementing agent. Every proposal is scoped to
keep the existing dark-glass visual language and to keep `verify.mjs` /
`verify-web.mjs` green — the exact ids, hooks and copy assertions that must
survive are in §7.

## 0 · Who this is for — three archetypes, and their first 60 seconds

**(a) Motion designer producing brand assets (the Solvd look).** Desktop,
patient, quality-driven. First 60 s: needs proof of *output quality* and
*control ceiling* — a reference-grade result on screen, evidence of alpha/
ProRes export, and a sense that parameters go deep. Current flow serves them
best of the three, but loses them at D7 (no instant proof on landing — they
bounce before donating a file) and D10 (the Solvd look is 10 slider decisions
away instead of one click). They are the audience for: look presets with a
"Custom" escape hatch, matched original cut, ProRes 4444, per-subject
palettes, the compare wipe.

**(b) Social creator on a phone.** Impatient, camera-roll-first, exports to
TikTok/Reels. First 60 s: pick a video from the roll → tap the subject → see
it styled → know an export is coming. Currently lost at second zero (D1: 94 px
stage). Needs: the mobile shell, tap-only subject picking, 9:16 named "TikTok/
Reels", share-sheet export, waiting states that survive a locked phone.

**(c) Developer embedding the `.dots` player.** Arrives via GitHub. First
60 s: what is the format, how small, how do I replay it — a working embed
snippet and a `.dots.gz` in hand. Currently served by README + `web/player/`,
but the app hides `.dots` export mid-panel (fine) with no link to the player
demo or embed docs (gap). Needs: a "For developers" corner that is quiet but
complete (P2.7) — not more prominence.

Design center of gravity: **(b) sets the layout, (a) sets the quality bar,
(c) is served by one well-made corner.**

---

## 1 · Diagnosis — ranked UX problems, with evidence

### D1 (critical) — There is no mobile layout at all
`style.css` contains **zero `@media` queries**. `#panel` is `flex: 0 0 296px`,
`body` is `overflow: hidden`. At 390×844:

- The panel takes 296 of 390 px; the whole stage — the product — gets **94 px**.
- The prompt image you are supposed to tap on measures **56×32 px**
  (`#pimg`/`#pov` bounding boxes, measured). The result canvas: 56×32. The
  sequence canvas: 56×32.
- The hero paragraph renders one word per line in a 93 px column
  (`ux-research/m00-landing.png`), the on-stage hint "CLICK WHAT YOU WANT ·
  SHIFT-CLICK WHAT YOU DON'T" wraps into a vertical word-stack
  (`m02-track-ui.png`), the tracked result is a postage stamp with a clipped
  `play` chip (`m04-tracked.png`, `m07-seq-preview.png`).
- Touch targets: chips are 28 px tall, sliders 16 px, trim handles **11 px
  wide** (Apple HIG minimum 44 pt, Material 48 dp).

Every flow technically *works* on mobile — the tap even lands a prompt point on
the 56 px canvas — but no real person can use it. **This is the single biggest
gap between the product and archetype (b), the social creator.**

### D2 (critical) — The canvas is subordinate to the panel
All interaction lives in a 296 px sidebar; the picture is passive except for
prompt clicks and canvas-drag. The Look step alone is ~1,300 px of stacked
sliders inside a 900 px viewport (`#panel` scrollHeight 1288, measured;
`d08-look-panel.png`, `d10-look-sliders.png`). Styling means scrolling a form
while glancing right at the picture. Every modern peer (CapCut, Figma, SAM 2
demo, Runway) puts the primary verbs on or directly under the canvas and
demotes fine parameters to a secondary surface.

### D3 (high) — Jargon-first vocabulary
Observed labels a first-time visitor must parse: *Prompt frame, Prompt tool,
point/box, lasso, polygon, propagation, Tracking quality "fast · prototyping ·
512 px · 27 fps", Track, Algorithm, Blue noise, Bayer, White noise, Error
diffusion ≈, Riemersma ≈, serpentine, Kernel, Matrix 4×4, Dither strength,
Cell, Fill cap, Stray, Halo band, gamma, Mask polish, .dots.gz* — plus the
mode note "organic grain, flicker-free". (`d03-track-ui.png`, `d08`, `d12`.)
The engineering-notebook tone of the notes is charming and worth keeping as
*secondary* text, but the *primary* labels assume ML + graphics literacy.
Vocabulary table with proposed renames: §2.4.

### D4 (high) — The magic moment is behind an extra button on video
On a **still**, every click re-cuts the outline live (~0.13 s, pink tint —
`d24-still-person-mask.png`; genuinely great). On a **clip**, a click paints
only a small red dot; you must find and press **"preview this frame"** to see
any mask at all (`d04-point-clicked.png` vs `d05-preview.png`). The strongest
"it understood me" moment of the product is optional and manual exactly where
the model is most impressive. SAM 2's demo shows the mask on hover/click,
always — that instant feedback is *the* pattern that made it viral.

Related: with no visible mask, a mis-click silently selects the wrong thing.
In testing, a click centred on the frame selected the *wall*, and the sky
tint in `d24` shows a click 20 px from the skater selecting the *sky*
(332,942 px). Nothing says "this looks wrong — click the thing itself, or
shift-click to carve".

### D5 (high) — All refinement affordances are keyboard-only
Every hint is modifier-based: "shift-click what you don't", "shift-drag
subtracts", "esc cancels", "double-click or enter to close" (TOOLHINT/TOOLNOTE,
app.js:1291-1304). On touch there is **no way to add a negative point, cancel
a lasso, or close a polygon**. The hint text still says "shift-click" on a
phone (`m02-track-ui.png`).

### D6 (medium-high) — Waiting states are thin for 15 s–5 min operations
Tracking: a 3 px progress bar plus microtext ("loading frames…",
`d06-tracking-progress.png`) at the bottom of the panel while the canvas sits
frozen on the prompt frame. Export: same pattern (`d13-exporting.png`). The
pre-run estimate exists and is honest (`#estline`, `DV_estimate()` returns
"tracking one subject ≈ 4.3 s at balanced") — good bones — but during the run
there is no on-canvas activity, no frames-as-they-arrive preview, no
done-notification affordance if the tab is backgrounded on a phone.

### D7 (medium-high) — First run is a blank page
`d00-landing.png`: a hero sentence and "choose a file" over an empty stage. No
sample clip, no demo content, no visual of what the output looks like, despite
the repo *shipping* `sample.mp4` (1.6 MB), built-in shapes (ring, coral — but
buried in the Sequence tab), a `.dots.gz` player that can replay a full
tracked-clip animation from a ~100–200 KB file, and dozens of gorgeous outputs
in `docs/`. On the future GitHub Pages site, a zero-context visitor must
donate their own file before seeing anything. Wow-in-30-s is impossible today.

### D8 (medium) — Sequence is a second world with its own language
"Studio"/"Sequence" tabs swap the whole panel. Sequence introduces *items,
library, captured this session, joins, morph/scatter/cut/density fade,
per-item look vs per-sequence background* — explained in three paragraphs of
panel prose (`d20-seq-view.png`, index.html #sq3 note). The strip itself
(thumbnails + MORPH chips between them, `d22-seq-join-sheet.png`) is actually
a good, direct UI — the *entry* and the *conceptual load* are the problem:
"+ to the sequence" only appears after a track completes, and nothing explains
what a sequence is *for* until you are inside it.

### D9 (medium) — Export buries the two decisions that matter
A native `<select>` of codecs ("MP4 · H.264, WebM · VP9, GIF · looping, WebM ·
VP9 + alpha, ProRes 4444 · alpha"), a "matched cut" checkbox with a paragraph,
and `.dots.gz` / `.dots.json` developer exports sit at the same visual level as
the one button 90 % of users want ("Render MP4") (`d12-export-panel.png`,
`d14-exported.png`). There is no "for TikTok / for Reels" preset that bundles
canvas 9:16 + MP4; the canvas control is in a different step (bottom of Look).
Meanwhile the storage line "storage: 2.7 GB · 257 jobs · over budget · clean
up" is visible to every user (`d03`, bottom-left) — ops detail leaking into a
creative tool.

### D10 (medium) — Dots-mode parameters are raw engine knobs
Six sliders (Dots count, Cell, Dot size, Fill cap, Stray, Halo band) with no
presets, no visual anchors, and no "the Solvd look" one-click (`d09`, `d10`).
Palettes have 18 presets; *looks* have none. The distance from "tracked" to
"looks like the reference brand work" is ~10 expert slider decisions.

### D11 (low-medium) — Reframe (canvas) is powerful but buried and blind
Canvas presets (source/16:9/9:16/1:1/4:5/custom + auto/follow/hold-still
framing + scale + drag-to-position) live at the *bottom* of the Look step
(`d25-canvas-916.png`). The drag hint is under the stage; there are no
safe-area/grid overlays for 9:16, and no platform naming (a social creator
knows "TikTok", not "9:16 · 1080×1920").

### D12 (low) — Small frictions
- Trim handles 11 px wide; no numeric time while dragging; "use this range" /
  "whole clip" is a good consent gate but easy to misread as a required choice.
- The compare wipe (`#bCmp`) is excellent and almost invisible (a small chip
  in the transport).
- The engine switch ("LOCAL SERVER · FASTER · MPS") is prominent top-of-panel
  chrome on first contact; for the public site the right default is silence.
- 18 palette chips in one flat run; no grouping (mono / retro / brand).
- Native `<select>`s for kernel & format look foreign in the dark-glass UI.

**What is already good and must not be lost:** the honest engineering voice of
the notes; the pixel-parity preview promise ("what plays here is what the
export contains"); the trim consent gate and re-cut *offer* (never a refusal);
per-subject palettes; the live still segmentation; the strip + join direct
manipulation; `.dots.gz` as a product moat; the dark-glass look itself.

---

## 2 · Information architecture

### 2.0 One model, three phases

Keep one page and one state machine. Present the flow as **three phases** that
map onto the existing five steps without moving DOM ids:

```
MAKE            STYLE            SHARE
st1 Source  →   st3 Look     →   st5 Export
st2 Subject     st4 Palette      (+ sequence as a parallel surface)
```

The five `section.step` elements (`#st1…#st5`, `data-open`, `.sh` headers)
survive untouched — verify drives them (§7). What changes is the *skin*:
grouped headers, renamed labels, preset-first ordering inside each step, and a
second, canvas-anchored surface for the primary verbs.

### 2.1 Desktop (≥ 1024 px) — keep the panel, promote the canvas

Layout stays `sidebar + stage`, dark glass unchanged. Changes, precisely:

1. **Canvas action bar** — a slim glass bar docked to the bottom edge of the
   stage (where the hint line lives today), showing the *current phase's one
   primary action* plus context:
   - Subject phase: `Keep ⊕ | Remove ⊖` toggle · subject chips (colour-dotted)
     · `Track subject — ≈ 12 s` CTA (the estimate from `clipEstimate()`).
   - Style phase: horizontal **look-preset strip** (see 2.3) — the transport
     bar stays below it.
   - Export phase: `Save video` + `9:16 · 1080×1920` frame chip.
   The panel keeps *everything* it has today; the action bar duplicates the
   3-4 highest-traffic controls next to the pixels they affect. (Pattern:
   Figma contextual toolbar; CapCut bottom bar.)
2. **Live mask on hover/click for clips** (kills D4): when the pointer is over
   `#pov` on a clip, debounce ~150 ms and run the same single-frame prediction
   the still path already runs (`predict this frame` currently behind
   `#bPrev`); paint the tint. Click commits the point. `#bPrev` stays in the
   DOM for verify but becomes the manual fallback (label: "re-check this
   frame"). The remote engine already answers single-frame predictions fast
   (0.12-0.13 s measured); on the browser engine, keep click-then-tint (no
   hover) if per-frame cost is too high — but always tint after *click*, never
   only after a separate button.
3. **Preset-first Look step**: the `#modes` chips remain (verify clicks
   `[data-mode]`), but above them insert a **look-preset row** (2.3) and move
   the six dot sliders plus Kernel/Matrix/serpentine under a single `advanced`
   disclosure (`<details>` or a chip-toggle). Tone (bright/contrast/midtones)
   stays visible — those three are universally understood.
4. **Frame (canvas) presets get platform names** and move up to sit directly
   under the mode row: `Original · 16:9 · 9:16 Story · 1:1 · 4:5`, with
   "TikTok / Reels / Shorts" as the 9:16 sublabel. When a non-source frame is
   active, draw a **safe-area overlay** on the stage (thin outline + top/bottom
   caption zones for 9:16) while dragging.
5. **Storage line** (`#gcbar`) hidden unless the engine is local/custom *and*
   usage is over budget; never on the browser engine. Ids stay.
6. **Compare** gets a labelled affordance: keep `#bCmp` but render it as
   `before / after` with an icon, and surface it in the canvas action bar
   during Style.

### 2.2 Mobile (< 768 px) — canvas-first shell, CapCut architecture

A media query + ~200 lines of CSS and a small JS shim re-parent nothing and
hide nothing that verify needs (verify runs desktop-viewport; the mobile shell
is pure additive layout).

**Screen anatomy (390×844 reference):**

```
┌──────────────────────────────┐
│ status strip (safe-area top) │  app name · engine dot · undo
│                              │
│         CANVAS               │  full-bleed, letterboxed,
│   (fills all free height)    │  pinch-zoom + drag
│                              │
│ [transport / scrubber]       │  44 px, only when a clip is open
│ ┌──────────────────────────┐ │
│ │  bottom sheet (detents:  │ │  peek 96 px · half 45 % · full 90 %
│ │  peek / half / full)     │ │  content = the ACTIVE STEP's .sb
│ └──────────────────────────┘ │
│ Media · Subject · Style · Export · Seq │  56 px tab bar (safe-area bottom)
└──────────────────────────────┘
```

- The five tabs are the five steps (`#st1…#st5` + Sequence). Tapping a tab
  sets the corresponding step's `data-open` and renders *that step's existing
  `.sb` contents* inside the sheet — implementation: on `<768px`, CSS
  re-lays-out `#panel` as the sheet (`position: fixed; bottom: 56px;
  transform: translateY(…)`) and shows only the open step; the step headers
  become the tab bar labels. No DOM surgery, no id changes.
- The sheet at **peek** shows just the step's summary line + primary CTA
  (e.g. `Track subject — ≈ 12 s`). **Half** shows the main controls. **Full**
  scrolls everything, including advanced.
- The canvas keeps ≥ 55 % of the viewport height whenever the sheet is at
  peek/half. That is the whole point.
- Sequence on mobile: the strip (`#strip2`) docks as a horizontal thumbnail
  rail directly above the tab bar; the inspector opens in the sheet.

### 2.3 Look presets (new, both platforms)

A horizontal row of **8-10 preset tiles**, each a ~72×72 live thumbnail of the
*current frame* rendered with a bundled parameter set (mode + palette + dot
params + tone), named plainly:

`Solvd · Newsprint · Game Boy · Blueprint · Ember · Ghost (alpha) · Comic ·
Terminal · Film grain · Custom`

- "Solvd" = dots mode, ~8-12k count, sage/forest palette — *the* brand look
  and the reason the tool exists; it must be one tap from landing.
- Tiles re-render on frame change (throttled) — Dithermark and CapCut both do
  live-preview pickers; static swatches lie.
- Picking a tile sets `S.P` + palette in one transaction; the sliders
  underneath update; "Custom" activates on any manual slider change.
- Implementation note: presets are pure parameter dictionaries over the
  existing engine — no new render paths.

### 2.4 Vocabulary — jargon → human

Primary labels change; the engineering voice stays in secondary `.note` text.
⚠ = copy asserted by verify, migrate test in the same commit (§7.3).

| Current (where) | Proposed | Note |
|---|---|---|
| Prompt frame (st2) | **Pick a frame** | slider stays `#sPF` |
| whole clip / track subjects (st2 chips) | **everything / just what I pick** ⚠ | or keep; if renamed, update the two `scopeLabels` assertions |
| whole image / select subjects (still) | same treatment ⚠ | assertion `verify-web.mjs:449` |
| Prompt tool · point / box · lasso · polygon | **Pick with: Tap · Draw around · Trace corners** | `data-tool` values frozen |
| "shift-click what you don't" | **"tap = keep · ⊖ mode = remove"** (touch) / keep modifier text on pointer-fine devices | dual hint via `pointer: coarse` |
| preview this frame (`#bPrev`) | **re-check this frame** | auto-tint makes it secondary |
| Track (`#bTrack`) | **Track subject — ≈ Ns** | still-mode label must keep matching `/use this selection/i` ⚠ |
| Tracking quality · fast/prototyping/512px | **Detail: Draft · Standard · Fine** (+ "≈ N s" each) | `#tq`, `[data-size]` frozen |
| Algorithm (st3) | **Style** | |
| Error diffusion ≈ / Riemersma ≈ | unchanged names, under **More styles** | `≈` tooltip: "approximate between frames" |
| Kernel / Matrix / serpentine | **Pattern (advanced)** group | `#sAlgo`, `[data-mx]` frozen |
| Dither strength | **Strength** | |
| Dots / Cell / Dot size / Fill cap / Stray / Halo band | **Density · Grid · Dot size · Ink · Scatter · Edge glow** | slider ids frozen |
| gamma | **Midtones** | |
| Mask polish | **Steady outline** | `#pollist .chip.pol` frozen |
| Background · flat / keep scene | **Background · a colour / the scene** | `data-compose` frozen |
| Canvas (st3) | **Frame** | |
| 9:16 preset | **Story 9:16 — TikTok · Reels · Shorts** | slug/id frozen |
| framing auto / follow / hold still | unchanged (already human) | |
| Export · Format select | **Save as** segmented chips (MP4 · WebM · GIF · transparent · ProRes) | keep `#sFmt` in DOM, chips proxy it |
| also save the original (matched cut) | **+ the untouched original (same cut)** | `#cOrig` frozen |
| .dots.gz / .dots.json | under **For developers** disclosure | `#bDots`, `#dotsexp` frozen |
| Sequence (view) | **Sequence** (keep) + subtitle "morph several into one" | `data-view` frozen |
| Items / library / captured this session | **Clips · captured earlier** | |
| join (between items) | **Transition** (inspector already says this) | |
| Studio / Sequence chips | unchanged | |
| storage · over budget · clean up | hidden for browser engine; "server storage" otherwise | `#gcuse`/`#bGC` stay |

---

## 3 · Pattern decisions — what we adopt, from whom, and why

Competitive pass (Aug 2026, primary sources: shipped JS bundles of the SAM 2
demo / dithermark / ditherit / remove.bg, live product pages, help centers,
app-transcript analysis of CapCut/TikTok). Decisions, not options — each maps
to a plan item in §6.

### 3.1 Subject picking

- **ADOPT — SAM 2's marker glyphs**: prompt points drawn as white-ringed
  circles, black fill + white `+` for keep, `#E6193B` fill + white `−` for
  remove (`r ≈ 8×dpr`, enlarge on hover for deletion, CVAT-style). Shape
  carries meaning, colour only reinforces; survives any footage. Replace the
  current bare red dot. → P0.2/P0.3
- **ADOPT — explicit `⊕ Add / ⊖ Remove` segmented pair** (SAM 2) on all
  pointers, as the visible state; **right-click = opposite mode** as the
  desktop shortcut (replaces shift-click as the *taught* path; shift keeps
  working).
- **ADOPT — implicit polarity as the touch default** (Roboflow Smart Select):
  tap outside the tinted mask = expand, tap inside = subtract. Zero
  toggle-state errors one-handed; the ⊕/⊖ pair stays visible as an override.
  → P0.2
- **ADOPT — click-to-delete an existing point** (SAM 2), with the marker
  enlarging under the pointer/finger first (CVAT).
- **ADOPT — the "ladder" framing** (CapCut remove-background): one entry,
  escalating rungs — tap-to-segment → more taps to refine → lasso/polygon →
  per-frame re-prompt. The current three-tool segmented control becomes the
  ladder's upper rungs, not a peer choice you face immediately.
- **INVESTIGATE — hover-preview masks** (SAM 1; dropped by SAM 2 because video
  frame embeddings can't be recomputed per hover): Dither Studio's engines
  already answer single-frame predictions in ~0.13 s on the local server. If
  the browser engine can cache the prompt-frame embedding, hover-preview is
  the single biggest available differentiator over Meta's own demo. Gate on
  measured round-trip < 250 ms. → P0.3
- **ADOPT — coaching snackbar state machine** (SAM 2, copy nearly verbatim):
  one floating pill over the canvas, fires once per moment, minimizes to a ⓘ
  chip instead of dying: *before first click* "tap any object to start";
  *after first click* "not what you expected? add a few more taps until the
  full object is selected"; *after track* "fix issues by going to the frames
  where tracking slips and adding or removing taps". This replaces the static
  hint line under the stage. → P0.7
- **ADOPT — failure-path copy** (remove.bg): when a prediction returns a
  tiny/empty mask (the observed 150 px wall-click case), say so: "that
  selection came back almost empty — try tapping the middle of the thing you
  want", and on the demo clip route back to the coach mark. Segmentation
  *will* miss; the failure branch is a routine outcome, not an exception.

### 3.2 Scrub & timeline

- **ADOPT — mask-coverage swimlane** (SAM 2): under the scrubber, one thin
  lane per subject in its colour — dim where no mask, bright where tracked,
  dots on prompted frames that jump there on tap. It answers "which frames
  are styled / where did tracking drop" at a glance and it is cheap: the data
  is already per-frame. → P1.4
- **ADOPT — whole-canvas horizontal scrub on mobile** (CapCut/IG convention):
  in preview, dragging the canvas scrubs; the transport slider stays for
  precision. Frame-step `◀ ▶` buttons ship on desktop and at the ends of the
  mobile transport — CapCut's known gap, cheap to win with a frame-indexed
  product. → P0.1
- **KEEP — the existing trim consent gate and re-cut offer** — no competitor
  handles "you asked for frames that aren't processed yet" this honestly.
  Only the hit targets and the live time readout change (P1.4).

### 3.3 AI-wait states (the 30 s track, the 1-5 min export)

- **ADOPT — streamed propagation as playback** (SAM 2, the highest-value
  single pattern in the research): during Track, masks paint onto the playing
  preview as frames complete; the advancing playhead *is* the progress bar;
  the button flips to `Cancel tracking`; on completion the finished result
  auto-plays from the top. The engines already deliver per-frame results —
  this is presentation, not pipeline work. → P0.6
- **ADOPT — labor perception** (growth.design; CapCut partial-mask preview):
  never a bare spinner — show the work. Keep the canvas alive: the background
  (which needs no mask) can start dithering immediately while the subject
  tracks; name the work ("frame 47 of 150 · 10.2 fps").
- **ADOPT — queued vs running as visibly distinct states** (Runway) for the
  remote/hosted engine: "waiting for the server" (indeterminate) must not
  look like "tracking frame 12 of 150" (determinate). Cancel is always
  available while queued.
- **ADOPT — bounded work** (CVAT's target-frame): the trim bar already bounds
  extraction; surface it in the CTA ("Track 3 s ≈ 5 s") so shortening the
  range is the obvious latency lever.
- **ADOPT — styling stays unlocked during tracking** — CapCut and SAM 2 both
  lock their UIs during AI runs; Dither Studio's dither preview needs no
  masks, so the Style step can stay live. A genuine advantage; state it in
  the progress copy ("meanwhile, pick a look —").

### 3.4 Preset-first styling

- **ADOPT — preset tiles rendered on the user's own frame; tap the active
  tile again to open its sliders** (CapCut's progressive-disclosure hinge).
  90 % of users never see a slider. → P0.4
- **ADOPT — presets compose with raw controls, never a "simple mode"**
  (Runway's "either or both"): a tile sets parameters; every slider stays
  live; touching one flips the tile to `Custom`. No mode switch, no
  graduation.
- **ADOPT — mode-first grouping with a permission-giving tooltip** (ditherit):
  lead with the 7 modes as a single friendly choice; the kernel/matrix
  taxonomy appears only after a mode that needs it is picked (already the
  case — keep). Add ditherit's one-liner: "these look quite different — try
  them out."
- **ADOPT — ◀ ▶ cycle arrows on the mode row** (dithermark): flip through
  looks like a lookbook without reading names. Nobody in the dither space
  ships per-mode live thumbnails; with a GPU preview Dither Studio can own
  that (P2.1).
- **KEEP — evocative palette names, technical algorithm names** (dithermark's
  split, already matched: "Ember/Mist/Game Boy DMG" vs "Floyd–Steinberg").
- **ADOPT — hold-to-compare on touch, wipe slider on desktop** (FX /
  ditherit): press-and-hold the canvas shows the original while held — zero
  chrome; the existing `#wipe` stays for fine pointers. → P1.7
- **ADOPT — auto-apply with a visible kill switch** (ditherit/dithermark
  "Performance"): live preview re-render on slider drag is the appeal, but on
  a 4K still or a low-power phone it becomes a hang; a "live update" toggle
  under advanced turns a bug report into a choice.

### 3.5 Export & sharing

- **ADOPT — fidelity gates, not function gates** (remove.bg/VEED framing,
  relevant for the future hosted tier): everything stays exportable free in
  the browser tier; the hosted tier sells speed and codecs (MP4/ProRes),
  which the format list already expresses with honest per-format notes —
  keep exactly that.
- **ADOPT — cost/size on the button** (ditherit `PNG ZIP (2.3 MB)`): the
  export CTA and the finished download button carry duration-derived size
  estimates; dithered output size is unpredictable, bytes-on-the-button
  removes the download-and-check loop.
- **ADOPT — a terminal share screen** (CapCut): after render, the sheet shows
  the result playing with `Share…` (native sheet, files) and `Save` — the
  emotional peak is when you ask where it should go. → P1.2
- **ADOPT — "Subject Focus" reframe naming** (Kapwing names its fit modes;
  its best one is gated behind face detection): Dither Studio already *has*
  the subject mask — name the framing options in those terms ("keep the
  subject in frame" = follow, "hold still", "fit"), strictly better than the
  category leader. → P1.3
- **ADOPT (P2, growth) — "use this look" deep links** (TikTok's `Use this
  effect`): every export can carry/copy a URL that opens the public site with
  the preset stack loaded (`?look=` params; for `.dots.gz` embeds, a link in
  the player). It converts "how did they do that?" into one tap.

### 3.6 Mobile shell & first-run

- **ADOPT — same flow, rail becomes a bottom sheet** (SAM 2 mobile does
  exactly this at <768 px; CapCut's root rail taxonomy): §2.2 architecture
  confirmed as industry-convergent. Do not gate mobile — a canvas art toy
  that works on a phone is a *praised differentiator* (HN on the ASCII toy).
- **ADOPT — "No video? Try one of these:" sample row** (remove.bg /
  Photoroom, near-verbatim): tiny looping thumbnails under the drop zone
  that skip straight into segmentation. Non-negotiable for the public site.
  → P0.5
- **ADOPT — acting dismisses the welcome** (Excalidraw WelcomeScreen): hints
  are painted in the canvas plane and disappear on first real action — no
  tour, no Next buttons, no progress dots. The §5 demo script uses one
  pulsing hint, not a modal carousel.
- **ADOPT — the landing is already running** (dither.it opens on a live
  dithered webcam feed): the hero plays a prebaked `.dots.gz`; "use your
  camera" is the second button, never auto-fired (permission prompt).
- **REJECT — onboarding checklists, tours, template marketplaces, account
  walls.** Evidence across Appcues/Linear/Duolingo teardowns: hand the user a
  real outcome, define done as something they possess (an exported clip).

---

## 4 · Mobile interaction spec

### 4.1 Breakpoints & layout

- `< 768 px` — mobile shell (§2.2): full-bleed stage, bottom sheet, tab bar.
- `768–1023 px` — tablet: sidebar returns at 296 px but sheet-style step
  accordion (one step open, sticky primary CTA); canvas action bar active.
- `≥ 1024 px` — current desktop layout + §2.1 additions.
- `viewport-fit=cover`; pad the tab bar with `env(safe-area-inset-bottom)`
  and the status strip with `env(safe-area-inset-top)`. The stage may run
  under the notch; controls may not.
- Orientation: portrait is the designed case. In landscape-phone, the sheet
  becomes a right-side panel at 320 px (CSS only, same DOM).

### 4.2 Touch targets

Minimum 44×44 px hit area on any coarse-pointer device (Apple HIG 44 pt /
Material 48 dp), delivered by padding & `::before` hit-extenders — visual
size can stay compact:

| Control | Today | Spec |
|---|---|---|
| chips (`.chip`) | 28 px tall | 36 px visual, ≥ 44 px hit |
| sliders (thumb) | 16 px track | 28 px thumb, 44 px hit, `touch-action: pan-x` on the row so vertical page-scroll still works from a slider |
| trim handles `#hIn`/`#hOut` | 11×56 px | 24 px visual, 44 px hit, magnetic ±8 px |
| step headers / tabs | 34 px | 56 px tab bar cells |
| transport buttons | 28 px | 44 px |
| subject chips `#subs .chip` | ~24 px | 36 px + colour dot ≥ 12 px |

### 4.3 Gesture grammar on the canvas (the conflict rules)

The canvas owns single-finger; the page owns nothing inside the stage
(`touch-action: none` on `#pov`, `#vcv` wrapper, `#wipe`, `#strip` — the
first three already set, keep them). Explicit grammar:

| Gesture | Subject phase | Style/preview phase | Frame ≠ source |
|---|---|---|---|
| tap | add keep-point (⊕ mode) / remove-point (⊖ mode) | play ⁄ pause | — |
| tap on existing point | select it (then `delete` chip appears) | — | — |
| drag | draw a box (point/box tool) · draw lasso (draw-around tool) | scrub (horizontal, whole canvas is a scrubber like CapCut/IG) | move the frame (existing `dx/dy` drag) |
| two-finger drag | pan (zoomed) | pan (zoomed) | pan |
| pinch | zoom the canvas (transform-only, max 4×; prompt coords unproject through the zoom) | zoom | zoom |
| long-press | ⊖ point (shortcut for one-handed use) | before/after while held (maps to `setSplit`) | — |
| double-tap | zoom to subject bbox / back to fit | fit | fit |

Rules: a gesture that starts on the canvas never scrolls the page; the sheet
is dismissed to peek by tapping the canvas, never by canvas gestures; pinch
during a drag cancels the drag (two-finger wins). Lasso/polygon on touch:
P2.4 (draw with one finger; floating ✓ / ✕ buttons replace enter/esc).

### 4.4 The sheet

- Detents: **peek** (96 px: step summary + primary CTA), **half** (45 vh),
  **full** (90 vh, internal scroll). Drag the grabber or the header; fling
  snaps. `overscroll-behavior: contain` so sheet-scroll never chains to the
  page.
- The active tab's CTA is *sticky* at the sheet bottom at every detent
  (`Track subject — ≈ 12 s`, `Save video`, …) — thumb-reachable, always.
- Keyboard (Custom URL, hex fields): sheet jumps to full and scrolls the
  focused field above the keyboard (`visualViewport` resize listener).

### 4.5 Waiting states on a phone (30 s – 5 min)

- Before: the CTA carries the honest estimate (`clipEstimate()` already
  computes it) — "Track subject — ≈ 40 s in this browser".
- During: on-canvas progress (P0.6): dimmed frame + scanline + tracked frames
  filling the scrubber; percentage + fps in the status strip; sheet drops to
  peek so the canvas shows the work.
- `navigator.wakeLock('screen')` held during track/export; released on done
  (widely supported on 2026 mobile browsers — see 4.7).
- If the tab is hidden mid-run (user answers a message): on `visibilitychange`
  back, show elapsed/remaining honestly; if the engine died in the background
  (iOS reclaims GPU), offer one-tap resume from the last tracked frame — the
  server engine survives backgrounding, the in-tab engine may not.
- Never a spinner without numbers; never block the Style sheet — styling the
  prompt frame while tracking runs is allowed and encouraged (the dither
  preview needs no masks).

### 4.6 Files in and out

- In: `<input type=file accept="image/*,video/*">` opens the iOS/Android
  photo picker directly — camera roll is the default source, keep it.
  Add `capture` only on the explicit "Use your camera" button, not the main
  picker. Registering as a **Web Share Target** (P2.5) enables
  "share from camera roll → Dither Studio".
- Out: after export, offer **`Share…`** (primary, `navigator.share({files})`)
  and **`Save`** (the existing `#dl` anchor). Platform facts in 4.7.

### 4.7 Platform reality (verified 2026)

Verified against primary sources (WebKit source & release notes, BCD/caniuse,
chromestatus, Chromium source) in Aug 2026. Confidence flagged where it matters.

**WebGPU (the browser engine's tracker):**
- iOS: **Safari 26 shipped WebGPU** (iOS 26 is the floor; iOS 18 and earlier
  have nothing). Android: Chrome 121+, but only Android 12+ on Qualcomm/ARM
  GPUs; Chrome 146 added a GL-backed Compatibility Mode
  (`requestAdapter({featureLevel:'compatibility'})`) worth requesting as a
  fallback. Firefox mobile: not a target.
- In-app browsers: iOS IG/TikTok use WKWebView, and **WebGPU is on by default
  in WKWebView on iOS 26** (confirmed from WebKit source; ~95 % confidence) —
  but only over https and not in Lockdown Mode. **Android IG/TikTok use System
  WebView where WebGPU support is genuinely unverified** (sources conflict).
  Feature-detect `navigator.gpu` + a real `requestAdapter()`; never UA-sniff.
- iOS device limits are generous for this app: `maxBufferSize` 256 MB–1 GB,
  `maxTextureDimension2D` 16384 (A13+), and `shader-f16` is always exposed —
  the existing fp16 ONNX variants can run.
- **Treat `GPUDevice.lost` as a routine code path** — iOS reclaims the GPU on
  backgrounding, Safari fires no `contextlost`, and recovery means recreating
  every resource. This is the §4.5 "resume from last tracked frame" case.

**ONNX Runtime Web on iOS — the one real landmine:**
- WebKit has an **open, unassigned memory-leak bug with Emscripten Asyncify**
  builds (bug 304810; the matching ORT issue reported 1-14 GB growth and
  crashes on iOS, closed stale). The default and `/webgpu` ORT bundles are
  Asyncify builds. The stable iOS 26 combination today is the **plain
  `/wasm` (CPU, SIMD, non-Asyncify) bundle**; the clean fix is the **JSPI
  bundle, which needs Safari 27** (shipping ~autumn 2026). Gate by
  `'Suspending' in WebAssembly`, and expect the browser engine's tracking on
  iOS 26 to run on WASM (its existing "no WebGPU → WASM, ~6x slower" fallback
  copy already covers this honestly).
- wasm threads need cross-origin isolation (COOP/COEP headers) — **GitHub
  Pages cannot set them; Cloudflare Pages can** (via `_headers`). Without
  them ORT is single-threaded; with them, iPhones still give only ~2 ORT
  threads (`hardwareConcurrency` is clamped to 4). Self-host the ORT wasm
  artifacts (already vendored in `web/ort/` — correct; COEP breaks CDN loads).
- Android in-app WebView: **no SharedArrayBuffer at all** → single-thread
  WASM there regardless of headers.
- iOS tab memory: no documented ceiling — Jetsam kills the tab without a
  catchable error. Budget frames accordingly (the existing 720p/30fps decode
  cap is the right instinct; keep export buffers streaming, never one big
  `arrayBuffer()`).

**Video I/O:**
- iOS file input hands over a **transcoded H.264 `.MOV`** (`video/quicktime`)
  from both camera and library — the current `accept` list already includes
  quicktime; never trust `file.type`, sniff the container.
- `capture` attribute opens the camera directly (iOS 10+/Android 25+); keep
  it only on the explicit camera button (§4.6).
- **MediaRecorder on iOS/Safari supports `video/mp4;codecs=avc1` natively**
  (and WebM/VP8+VP9 since Safari 18.4). Product consequence: the browser
  engine's "MP4 needs the server" is no longer true on Safari — worth a
  follow-up to offer in-tab MP4 there (verify currently asserts mp4 is
  unavailable on the browser engine; migrate that assertion when done).
  Always pass a full mime string to `isTypeSupported` (empty string returns
  a meaningless `true`).
- WebCodecs: `VideoDecoder`/`VideoEncoder` since Safari 16.4 / Chrome 94;
  on iOS only **avc1/hevc are hardware paths** (VP8/VP9 encode is software —
  thermal throttling on long clips); H.264 requires **even dimensions**
  (round canvas exports to multiples of 2); `alpha:"keep"` is unsupported —
  alpha exports stay WebM/ProRes via the current paths. Frame-accurate
  `currentTime` seeking is not guaranteed on iOS; for exact frames use
  `VideoDecoder` or play-through capture with `requestVideoFrameCallback`
  (Safari 15.4+). Offscreen extraction: `muted playsinline` + `.play()` —
  never the `autoplay` attribute (visibility-gated).
- iOS canvas cap: **4096×4096** total area per canvas — 1080×1920 previews
  are fine; never composite on a >4K scratch canvas on mobile.

**Sharing & files out (P1.2 design constraints):**
- `navigator.share({files})`: iOS 14+ / Android Chrome 76+. Must be called
  **synchronously in the tap handler** (transient activation; no `await`
  before it). Feature-detect with `canShare({files})`.
- **Name the file `clip.mp4`** — the iOS share sheet resolves the type from
  the *extension* (wrong/missing extension → the video apps silently
  disappear from the sheet). `.mov`/`video/quicktime` is also **rejected by
  Android Chrome's allowlist** — another reason the share path should always
  be MP4/WebM.
- **Android Chrome caps shared files at 50 MB** — over that, fall back to
  download with a toast. Surface the estimate on the button (§3.5).
- There is **no direct-to-TikTok/IG web API** — their share SDKs are
  native-only. The generic share sheet is the ceiling: on iOS it includes
  **"Save Video" straight to Photos** (better than `<a download>`, which
  lands in Files → Downloads); IG appears as an Android target (confirmed),
  TikTok unverified — test on device.
- Android in-app WebView has **no Web Share at all** → keep the `#dl` anchor
  fallback everywhere.

**PWA reality check (adjusts P1.8):** iOS in-app browsers run no service
workers, and Safari's ITP deletes site data (incl. OPFS/Cache) after 7 days
without interaction — treat the PWA as installable convenience for fans, not
as an offline guarantee; model weights must always be re-downloadable.

**Touch specifics (§4.2-4.3 grounding):** WCAG 2.5.5/AAA 44 px & 2.5.8/AA
24 px are the citable target minimums (Material: 48 dp). `touch-action:none`
is correct for the canvas (already set) but inhibits browser zoom — keep the
pinch-zoom + double-tap controls (§4.3) as the accessible alternative, and
set `overscroll-behavior:none` on `html` to kill pull-to-refresh around the
full-bleed canvas. `user-scalable=no` is ignored by iOS — do not rely on it.
Long-press must be a hand-rolled `pointerdown` timer (the `contextmenu`
event is not specified for touch), with `-webkit-touch-callout:none` on the
canvas. Add `viewport-fit=cover` to the existing viewport meta (currently
absent from `index.html`) or `env(safe-area-inset-*)` reads zero.

---

## 5 · First-run & demo design — wow in under 30 seconds

The public static site is the first contact for archetypes (b) and (c) and
half of (a). Today's first contact is a sentence and an empty file picker
(`d00-landing.png`). Design:

### 5.1 The landing IS the demo

- **Hero = a playing `.dots.gz`.** The repo already ships a player
  (`web/player/dither-player.mjs`) that replays a full tracked-clip dot
  animation from a ~100–200 KB file at 60 fps with trivial CPU — no model, no
  WebGPU, works in *every* browser including in-app webviews where WebGPU is
  absent. Pre-bake `demo.dots.gz` from the skater clip (the tracked subject in
  the Solvd look, morphing into the ring shape and back — a sequence export).
  It runs as the stage background behind the hero copy from byte one.
  *This is the single highest-leverage first-run asset: the actual product
  output, animating, for the cost of a small image.*
- Hero copy shrinks to one line + two buttons: `Choose a file` ·
  `Use your camera` (drop/paste hints stay on desktop; on mobile the file
  button opens the camera-roll picker natively; camera never auto-fires —
  it is a permission prompt).
- Directly under the drop zone, the remove.bg pattern near-verbatim:
  **"No video? Try one of these:"** + 4-6 tiny (40-56 px) looping
  thumbnails — the demo clip, a portrait still, a shape — each skipping
  straight past upload into the flow (§3.6). Small size signals "shortcut",
  not "main path".
- Under the hero, a **6-tile "looks" strip** (thumbnails from `docs/`
  outputs) — communicates range (Solvd dots, Game Boy, halftone, alpha cutout,
  newsprint, C64) before any interaction.

### 5.2 "Try the demo clip" path (the 30-second script)

1. Tap → `sample.mp4` (1.6 MB, shipped next to the page) loads through the
   normal `take()` path; trim bar appears pre-set to the 5 s clip. **0-3 s.**
2. The app auto-advances to Subject with one Excalidraw-style hint painted
   in the canvas plane: "tap the skater" with a pulsing marker at the
   subject's known coordinates (hardcoded for the demo clip only). Any real
   action dismisses it permanently — no modal, no Next button. **3-8 s.**
3. Tap → instant mask tint (D4 fix) → the CTA reads `Track subject — ≈ 12 s`.
   During tracking, the SAM 2 treatment (§3.3): the preview plays and masks
   stream in as frames complete; the button reads `Cancel tracking`.
   **8-25 s.**
4. On completion auto-apply the **Solvd preset** and start playback. Dithered
   skater, animating, sage-on-forest. **≈ 25-30 s. Wow delivered.**
5. The Style sheet peeks up with the preset strip; from here they are in the
   normal flow. Export is one tap away.
- On the browser engine the same script works with WASM fallback timing
  (~30-60 s track); the demo therefore also *honestly demos the wait UX*.
- If models are missing (static host without the 83 MB weights), the demo
  clip still runs whole-frame: step 2 is skipped, the Solvd preset applies to
  the full frame, and the subject step shows its existing plain-language
  missing-models note (`checkModels()`, app.js:5039).

### 5.3 Still-image micro-demo

`Try a photo` secondary path using `sample.jpg`: click → live cutout →
preset → PNG in ~10 s. Cheap to add (the still path is already instant) and it
is the flow (a)-archetype designers will test first.

### 5.4 Empty states that teach

- Sequence view, empty: instead of three paragraphs, an inline 3-frame
  storyboard graphic (clip → morph → shape) + one line: "capture things in
  Studio, then morph between them here", + the `+ ring` / `+ coral` chips
  promoted to a visible "start with a shape" row.
- After first successful track, a one-time toast: "the look is yours now —
  everything in Style updates live" pointing at the Style tab (mobile) / Look
  step (desktop).

### 5.5 Failure paths are first-run design

The demo's segmentation can still miss (a tap lands on the wall — observed in
testing). Ship remove.bg-grade failure copy: "that selection came back almost
empty — tap the middle of the thing you want", with the hint marker
re-appearing on the demo clip. A first-timer decides in that moment whether
the product is broken or their input was unusual.

### 5.6 What NOT to build

No account walls, no tour modals, no multi-step onboarding carousel, no
checklists. One in-canvas hint, one sample row, a prebaked hero. The
product's own output is the onboarding.

---

## 6 · Prioritized implementation plan

Effort scale: S ≤ ½ day · M ≈ 1-2 days · L ≈ 3-5 days. Every item ends with
both verify suites green (§7.4). Ship order within a band is the listed order.

### P0 — must (the product is broken or mute without these)

| # | Item | Effort | Notes / constraints |
|---|---|---|---|
| P0.1 | **Mobile shell**: `@media (max-width: 767px)` layout — full-bleed stage, `#panel` becomes a bottom sheet (peek/half/full via `transform` + drag), step headers become a 5-tab bar that toggles `data-open`, transport docks above the tab bar, safe-area insets (`env(safe-area-inset-*)`), `viewport-fit=cover` | **L** | CSS + ~150-line controller. Zero DOM id changes; desktop ≥1024 px pixel-identical. §2.2, §4 |
| P0.2 | **Touch subject picking**: `Keep ⊕ / Remove ⊖` toggle chips (canvas action bar + sheet), `touch-action: none` audit on `#pov`/`#strip`/`#wipe` (already present — keep), dual hint copy via `pointer: coarse`, 44 px min targets for chips/handles/sliders on coarse pointers (padding, not layout change) | **M** | Kills D5. `data-tool` values frozen. §4.2 |
| P0.3 | **Instant mask feedback on clips**: click → tint immediately (auto-run the existing single-frame prediction); hover-preview on `pointer: fine` if engine round-trip < 250 ms; `#bPrev` relabelled "re-check this frame", kept for verify | **M** | Kills D4 — the wow moment. §2.1.2 |
| P0.4 | **Look presets row** (8-10 tiles incl. "Solvd", live thumbnails, sets `S.P`+palette atomically, "Custom" on manual change); dot sliders + Kernel/Matrix under an `advanced` disclosure | **M** | Kills D10, halves D2. New DOM only; `#modes`/sliders/ids intact. §2.3 |
| P0.5 | **First-run demo**: `demo.dots.gz` hero via existing player, "No video? Try one of these:" sample row, in-canvas hint + auto-Solvd-on-complete script, looks strip | **M** | Kills D7. §5. Prebake asset in repo (`web/demo/`) |
| P0.6 | **Tracking as playback** (SAM 2 pattern, §3.3): during Track the preview plays and masks paint in as frames land (per-frame data already streams); `#bTrack` flips to `Cancel tracking`; auto-play the finished result; "frame N of M · fps" + "≈ N s left" from the pre-run estimate; same treatment for export via `#rprog`; `navigator.wakeLock` during runs on mobile; Style stays unlocked while tracking | **M-L** | Kills D6. `#tinfo` copy contract §7.3 |
| P0.7 | **Vocabulary pass** per §2.4 table (labels + notes + hints), migrating the three copy assertions in the same commit | **S** | Kills D3. §7.3 list is exhaustive |

### P1 — should (converts the fixed product into a good one)

| # | Item | Effort | Notes |
|---|---|---|---|
| P1.1 | **Canvas action bar** (desktop + mobile): phase-aware primary action + subject chips + frame chip docked to the stage | M | §2.1.1 |
| P1.2 | **Export re-order + share screen**: `Save video` primary; format as segmented chips proxying `#sFmt`; `.dots` under "For developers"; storage bar conditional; terminal share screen with `Share…` (`navigator.share({files})`, called synchronously in the tap, file named `clip.mp4`) + `Save` (`#dl` fallback — mandatory: Android in-app WebView has no Web Share) | M | Kills D9. 50 MB Android share cap → estimate on the button, fallback toast over it. §4.7 |
| P1.3 | **Frame presets promoted + safe-area overlay** for 9:16/1:1/4:5 with platform sublabels; grid + title-safe zones while dragging; `follow` framing given a one-line explainer | S-M | Kills D11 |
| P1.4 | **Trim polish**: 24 px visual / 44 px hit-area handles, live `mm:ss.f` readout while dragging, tap-preview loop of the selected range | M | `#stripcv`/`#hIn`/`#hOut` stay |
| P1.5 | **Sequence entry softening**: storyboard empty state, "+ to the sequence" renamed "use in a sequence" with a first-time popover, strip docked above mobile tab bar | M | Kills D8 surface; deep IA unchanged |
| P1.6 | **Palette grouping** (Mono · Duo · Retro · Greys) + palette tiles show their colours at true proportion; "from image" promoted | S | |
| P1.7 | **Compare wipe affordance**: labelled `before/after`, auto-invoked once after first track (500 ms peek animation) | S | |
| P1.8 | **PWA basics** for the public site: manifest + icon + offline shell for the player/demo (not the models), `display: standalone` | S-M | convenience, not an offline promise — no SW in iOS in-app browsers, ITP evicts site data after 7 idle days (§4.7) |

### P2 — nice (polish and reach)

| # | Item | Effort | Notes |
|---|---|---|---|
| P2.1 | Preset thumbnails re-render live on scrub (throttled rAF) | S | |
| P2.2 | Haptics (`navigator.vibrate`) on mask commit / track done (Android; iOS no-op) | S | |
| P2.3 | Sequence joins: tap a join opens a mini transition picker sheet with animated 24-frame previews of morph/scatter/cut/fade | M | |
| P2.4 | Lasso/polygon touch mode: one-finger draw, two-finger pan/zoom, ✓/✕ floating buttons replace enter/esc | M | completes P0.2 for the two advanced tools |
| P2.5 | `?src=<url>` + Web Share Target (PWA) so phones can "share into" Dither Studio | M | |
| P2.6 | Per-archetype landing chips ("brand assets · social clip · embed the player") that pre-set Look/Frame/Export defaults | M | |
| P2.7 | Developer surface: `/player/` demo page linked from the `.dots` export note, with copy-paste `<script>` embed snippet and the exported file pre-wired | S | archetype (c) |
| P2.9 | **"Use this look" deep links** (TikTok pattern §3.5): `?look=` URL params that open the public site with a preset stack loaded; a copy-link chip on the export screen and in the player embed | M | the growth loop |
| P2.10 | Mask-coverage swimlanes under the scrubber (one per subject, bright = tracked, dots = prompted frames, tap to jump) | M | §3.2, pairs with P1.4 |
| P2.8 | Onboarding a11y pass: focus order in the sheet, `prefers-reduced-motion` for the hero demo, captions for coach marks | M | |

### Engine follow-ups surfaced by the platform research (not UX, file separately)

- iOS 26 browser engine: prefer the non-Asyncify `/wasm` ORT bundle until
  Safari 27 ships JSPI (open WebKit Asyncify leak — §4.7); gate by
  `'Suspending' in WebAssembly`.
- Safari MediaRecorder does `video/mp4;codecs=avc1` natively — the browser
  engine could offer in-tab MP4 on Safari (migrate the verify assertion that
  mp4 is browser-unavailable when doing so).
- Host the public site where COOP/COEP headers are settable (Cloudflare
  Pages `_headers`), or browser-engine tracking runs single-threaded.
- Handle `GPUDevice.lost` as a resume path (backgrounded iOS tabs).

### Explicit non-goals of this redesign

- No visual rebrand — the dark-glass system, radii, type and palette stay.
- No framework adoption; the app stays vanilla ES modules.
- No server API changes; everything here is `web/` + copy.
- No removal of any current capability (kernels, polygon tool, custom canvas,
  `.dots.json`, engine switcher) — they move down the disclosure ladder, never
  out.

---

## 7 · Compatibility appendix — what the verify suites need to stay green

`verify.mjs` (server engine, 1898 lines) and `verify-web.mjs` (browser engine,
2353 lines) drive the real UI headless. They are the regression net for the
whole product; the redesign must not strand them. Extracted surface:

### 7.1 Element ids that MUST keep existing and keep their behavior

Clicked / read / filled directly (union of both suites):

- **Chrome / engine**: `#engine` `#engName` `#engpop` `#engUrl` `#engGo`
  `#engstat` (+ `[data-eng="browser"|"custom"]` options)
- **Source**: `#file` (file input!), `#upstat`, `#stripcv`, `#bTrim`,
  `#bTrimAll`, `#trimoffer`, `#bExtend`, `#bCam`, `#bRec`, `#bSnap`
- **Subject**: `#bTrack` `#bPrev` `#bAdd` `#bClr` `#pfui` `#sPF` `#pvinfo`
  `#tinfo` `#offframe` `#pov` (must remain the click-target canvas whose
  `getBoundingClientRect` maps to its `width`/`height` attrs — `stageXY()`
  depends on it), `#subs .chip`, `#scope .chip`, `#tq .chip`
- **Look**: `#modes .chip[data-mode]`, `#sAlgo` `#sPx` `#bCmp` `#wipe`
  `#pollist .chip.pol`, `[data-mx="8"]`, `[data-compose="cutout"|"overlay"]`
- **Palette**: `#target .chip` `#pals` `#bFromImg` `#bgui` `#swatches`
- **Export**: `#bExport` `#rinfo` `#dl` `#dlorig` `#cOrig` `#origui`
  `#orignote` `#cAlpha` `#pngalpha` `#sFmt` (may be visually replaced but must
  stay in DOM and settable), `#bDots` `#dotsexp` `#fps` `#vcv` `#bPlay`
- **Sequence**: `[data-view="sequence"]` `#bToSeq` `#seqadd` (chips by text)
  `#bSeqNew` `#shapeFile` `#seqinspect` (+ its internal contract:
  `.chip[data-mode]`, `.chip.pol`, `input[type=range]`, `input[type=color]`,
  `.lbl > span:first-child`), `#strip2` (`[data-i="0"]`…), `#bSeqPrev`
  `#bSeqDots` `#bSeqVideo` `#seqinfo` `#seqvid` `#seqdl`
- **Steps**: `#st1…#st5` as sections with `data-open` attr and a clickable
  `.sh` header (`openStep()` helper clicks `#stN .sh` and reads `data-open`).
  The mobile sheet must therefore *reuse* `data-open`, not replace it.
- **Storage**: `#gcuse` `#bGC` `#gcbar` (may be conditionally hidden — verify
  only reads them on the local engine, where they must appear).

### 7.2 `window.DV_*` hooks (all must survive verbatim)

`DV_ready` `DV_engine` `DV_switchEngine` `DV_draw` `DV_maskURL` `DV_composeAt`
`DV_originalAt` `DV_still` `DV_polish` `DV_formats` `DV_camera` `DV_trim`
`DV_limit` `DV_estimate` `DV_useRange` `DV_wholeClip` `DV_range` `DV_dots`
`DV_seq` `DV_canvas` `DV_setFormat` — plus `window.DV = S` itself with the
state fields the suites read: `kind nFrames natW natH subjects tracked scope
promptFrame view seq srcDuration recordedS awaitingChoice engine job bg
previewMasks saveOriginal trackSize`. Renaming state fields inside `S` is a
breaking change; add new fields instead.

### 7.3 Copy the suites assert on (migrate test + copy in the SAME commit)

- `#scope` chips on a still must join to `whole image/select subjects`
  (verify-web ~449) — if renamed per §2.4, update this string.
- `#bTrack` on a still must match `/use this selection/i` (verify-web ~457).
- `#tinfo` after tracking must match `/tracked|failed/` and contain
  `(N fps)` — keep "tracked … ( … fps)" in the completion line.
- `#pvinfo` must match `/subject|failed/`; `#rinfo` render lines must not
  gain the word "failed" in success states.
- Subject chips must keep the `@ <frame>` notation (`/@ 0/` asserted).
- Browser engine must keep reporting mp4/prores as unavailable-with-reason
  via `DV_formats()` (notes ≥ 10 chars).
- `#fps` remains a bare `N fps`-parseable readout.

### 7.4 Practical rule for the implementing agent

Desktop DOM = source of truth; verify runs at desktop viewport. The mobile
shell must be **CSS + a thin controller** (tab bar sets `data-open`, sheet
positions `#panel`) so that at ≥ 1024 px nothing observable changes. Any id
you must move: move the id *with* the element, never mint a duplicate id. Run
`node verify-web.mjs` and `node verify.mjs` after each P0 item, not at the end.
