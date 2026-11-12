# PhotoCraft AI

Everything this machine can do to a picture, in one panel beside the editor: **generate** an image
from a phrase, and **cut out** what is in one. Two tabs, one look — the graphite desktop palette of
PhotoCraft itself — and every result lands in PhotoCraft, not in a folder.

* **Cutout tab** — type `hair`, `face`, `the dress`, `mountain` and get the subject back as a live
  selection, an alpha cut-out (BiRefNet hair matte), a layer mask or a new layer, in the document
  PhotoCraft already has open. SAM3 over ComfyUI.
* **Generate tab** — the Qwen-Image 2.1 engine, embedded. It keeps its own server (port 7866,
  started on demand) and its own ComfyUI (port 8189); finished renders are posted straight to this
  panel and opened in PhotoCraft as a new document (or as a new layer, your choice).

    UI      http://127.0.0.1:7887
    Engine  http://127.0.0.1:7866        (the Qwen-Image studio repo; the Generate tab starts it)
    MCP     http://127.0.0.1:7887/mcp    (streamable HTTP, same process)
    Start   ./start.sh

## Setup

```bash
git clone https://github.com/mjb73/cutout-studio && cd cutout-studio
./start.sh            # first run builds .venv and installs requirements.txt
```

Then open <http://127.0.0.1:7887>. `config.json` is created on first start from
`config.example.json`; it is git-ignored and holds this machine's paths:

| Key | Meaning |
|---|---|
| `port` | the panel's port (or set `CUTOUT_PORT`) |
| `backends` | ComfyUI instances to cut out with — `host` (must answer `/system_stats`), `dir` (its folder, for models), optional `wrapper` launch script |
| `ckpt`, `node` | the SAM3 checkpoint and node name the cutout graph uses |
| `ai_engine` | where the Generate tab's engine lives (`http://127.0.0.1:7866` by default) |
| `export_dir` | where exports are written |

Cutting out needs **ComfyUI with the SAM3 nodes and the SAM3.1 checkpoint**; the Generate tab needs
the engine, [qwen21-studio](https://github.com/mjb73/qwen21-studio), which the panel starts on demand
and whose Models tab is where you point it at your ComfyUI folder and pair each model with its text
encoder and VAE. Without the editor, the panel still runs standalone in a browser — the "send to
PhotoCraft" paths just say there is no editor to send to.

## The Generate tab — how a render reaches PhotoCraft

The engine is a separate process, exactly like the ComfyUI behind it: `qwen21-studio`'s FastAPI app on
7866 talking to its own ComfyUI on 8189. It is not reimplemented here; this panel embeds it and
starts it when its tab is first opened.

    render finishes in the engine
      -> engine posts {filename, subfolder, prompt} to window.parent   (js/main.js)
      -> panel POSTs /api/ai/deliver                                   (static/app.js)
      -> this server fetches the picture from the engine's /api/view
      -> services.deliver_ai_image(): copy into ~/Pictures/photocraft-ai/, then
           mode=document  doc_open  -> PhotoCraft shows it as its own document   (default)
           mode=layer     layer.copyToDocument -> new layer named "AI · <prompt>"

`postMessage`, not `fetch`: the panel and the engine are on different ports, so a browser-side
call would need CORS on the panel. A message needs nothing. `subfolder` matters — the engine files
its outputs in dated subfolders and `/view` cannot find them without it.

Delivery survives the app going away mid-call. `doc_inspect` answers with a sentence, not JSON, both
when nothing is open (`no document open` — treated as an empty document) and when the bridge cannot
reach PhotoCraft at all (`cannot connect to 127.0.0.1:7916`), so `_pc_doc` refuses the second case
instead of returning a nameless document: that used to surface as *"AI opened in PhotoCraft as null
(null×null)"* followed by the MCP client's raw `ExceptionGroup`. After `doc_open` the delivery waits
for a document that has a name and a size, and on any bridge failure it retries once — raising the
window, and starting a drivable PhotoCraft when the control port is gone. A user's own PhotoCraft
(no `--control`) does not block that: the two are separate processes that coexist, so only a
`--control` instance whose port has died is reported as a conflict.

The panel passes `?theme=photocraft&embed=1` to the engine. The engine's own themes are CSS blocks
(`[data-theme="..."]` in its `styles.css`), so the PhotoCraft palette lives there too, `?embed=1`
hides its title bar, and the theme block also restates the studio's primitives at the editor's
density (11-12 px text, ~24 px rows, 1 px seams, pill buttons, thin grey scrollbars) — the studio is
designed at a 27 px root for a full window and would otherwise fill a 430 px column with three
oversized controls.

**I2I can take its source from the editor.** In the engine's I2I mode a checkbox appears under
`Image 1` — *"Use the layers named image 1, image 2 … in PhotoCraft"* with a Qwen-Image 2.1 model,
*"Use the layer named image 1 in PhotoCraft"* with any other model (they take one picture) — which pulls the live document
(`doc_export`, unsaved edits included) instead of asking for a file. Layer references named
`image 2`, `image 3` … travel with the same press (see *Layer references* below); the picture itself
is image 1. It is a postMessage round-trip
because the engine is a different origin: the engine asks, the panel pulls, reads the PNG from its own
`/files/sources/` and posts the bytes back (no CORS on the panel). The checkbox only exists while
embedded, it disables the file input when ticked, and **Generate re-pulls**, so it always means the
document open now. Nothing is previewed in the I2I field — the picture is not shown here at all, only
its size and aspect are read from it (the file input names the file). The engine's
`handleImageUpload()` resolves after the upload *and* the source's own size are known, because I2I
takes its dimensions from that size.

**An I2I render of a PhotoCraft picture comes back into that same document as a layer on top.** The
engine marks the render (`from_photocraft`), the panel forces `mode="layer"` for it, and
`_pc_deliver_layer` copies it in, raises it to the front and guarantees it is visible — the picture it
was derived from stays underneath, untouched. The render is made at the source's **exact pixel size**
(`dimension_mode: "Direct"` when the MP field still matches the source), so the new layer covers the
canvas instead of leaving a strip of the original showing; raising the MP field still resizes.

The engine's own **Output tab is gone**: it existed to show the finished picture inside the engine,
and that is now PhotoCraft's job — the render is announced upward and opened as a document (or layer)
there. What went with it is the engine's in-pane post-process panel (levels / colour / vignette,
`Apply & Save`), because it was only reachable from that tab; PhotoCraft's Camera Raw and adjustment
layers do the same work on the delivered picture. Progress is still visible while a job runs: the
queue strips under the engine's status field and the Generate bar's status text. The engine's tab
bar is now `Prompt | Defaults`.

API: `GET /api/ai/status` (up, port, mode) · `POST /api/ai/start` (spawn the engine wrapper, wait for
the port) · `POST /api/ai/mode {mode}` · `POST /api/ai/deliver {filename, subfolder, prompt, mode}`.

**The log is the record of the session, and it is open.** The bottom dock carries one ellipsised line
of the newest message plus a `Log` button, and under it the pane holds *everything since the server
started*, newest first: `MAX_LOG_LINES` is `Infinity`, lines wrap (`overflow-wrap: anywhere`) rather
than being cut with an ellipsis, and the pane scrolls — with a visible thin scrollbar, the one place in
this app where scrolling back is the point — up to `clamp(160px, 30vh, 380px)`. It starts open; the
`Log` button still folds it (`localStorage` `cutout-log-open`, absent = open), and folding never hides
a failure: the bar repeats the newest message with its colour and the button turns pink on an error.
One function (`setLogOpen`) owns the fold, `aria-pressed`, the title and the saved choice, and
`setPreview()` moves the whole dock between the window footer and the sidebar.

**Order of the Generate tab, top to bottom** (the sequence, not a pile of panels):

1. `Generate` | `Cutout` — Generate first; it is the tab that gets used, so it is also where a fresh
   profile lands (a remembered tab still wins).
2. `Prompt` | `Defaults` — the engine's own sub-tabs.
3. `Status`, then the CPU / RAM / GPU gauges.
4. The prompt, then `Generation Setup` and `Post-processing`, both folded by default.
5. **Generate Image**, with the queue strip directly above it — a running job is announced beside the
   button that started it, not at the top of the window.
6. `Generated images go to …` and the engine line.
7. The log.

Items 1–5 are the engine's frame, 6–7 belong to this panel, which is why the delivery choice sits
under the frame rather than inside it.

## Everyday workflow (the intended path)

The picture lives in PhotoCraft. This app is the segmentation engine behind it: it reads the
document open there, does the phrase→mask work, and puts the result back into that same
document. It is not a place you load pictures into.

1. Open **PhotoCraft from its launcher card** (Image Generation group). That card starts it with its
   control server and an automation root of `~/Pictures`; a PhotoCraft opened any other way cannot be
   driven from outside, which is the one thing that will stop this workflow. Open your picture there
   as usual.
2. Open **Cutout Studio**. It pulls the picture already open in PhotoCraft by itself — `doc_export`,
   the flattened composite, including unsaved edits — and the `Select` panel names the document it
   came from. There is **no image on screen here**: the app is the control panel, PhotoCraft is the
   picture. There is no Open / Paste / import path at all, by design: `Execute Cutout` is the only way in and
   it pulls whatever PhotoCraft is showing (again, if you switched documents). `Preview` in the
   header brings the canvas back (original / overlay / cut-out, zoom) if you want to see what will
   be selected.

   The panel is built to be a **narrow column beside PhotoCraft**: it is the editor's dock, not a
   stack of floating cards. The chrome strip, the tab strip and the `Select` pane run edge to edge
   with 1 px seams, the log is a footer status bar outside the scrolling area, and the window works
   from ~320 px wide up to anything — `body.scrollWidth == clientWidth` at every width measured
   (320/340/380/420/430/480/560/760/1200 px). Both tabs are the same width: the page never widens
   for `Generate`, and the tab buttons are equal-width halves of the strip. Below 480 px the header
   wraps to two rows, the backend badge ellipsises and `Preview` hides (a 380 px preview is useless
   — widen the window for it). The
   `Refine Edge` row and its radius only exist when the output is a mask.
3. In **Select**, type what you want (`person`, or `woman, hair` to union two phrases) or click a chip.
   `Execute Cutout` runs SAM3 and shows a green overlay of exactly what will be selected.
4. Press that one button. `Execute Cutout` segments the phrase and puts the result into the
   document it came from — there is no second "send" step and no second button. What lands there is
   decided by the single select directly above the button, `PhotoCraft gets`:
   **Selection — marching ants** (default: pixels untouched, nothing cut out, nothing hidden),
   **Cut out — hair matte** (a real per-pixel alpha layer, strand-accurate, ~10 s),
   **Layer mask** (background hidden non-destructively) or **New layer** (the subject copied onto
   its own layer, hard edge). The log reports the step count and PhotoCraft's own measured
   selection area against the mask's, so a bad match is visible rather than assumed.
5. Continue in PhotoCraft with its own tools. `Ctrl+D` deselects; one undo removes the mask or the
   layer.

Everything that is not "type a phrase and press the button" lives behind `Advanced`, folded away
by default: threshold, SAM3 refine iterations, invert, `Refine Edge` + hair radius (mask output
only), feather, background colour, and `Export files — cut-out, mask, outline`, which writes the
files into `~/Pictures/cutout-studio/<date>/` for use elsewhere. The one line above the phrase
always names what will be acted on (`PhotoCraft: my-photo.png · 978×490 · has a selection`).
The panel is deliberately one action wide: earlier it had `Find selection` (now `Execute Cutout`), `Make cutout`,
`Apply to PhotoCraft`, `Save PNG`, `Copy mask` and `Load image only`, which all read as the same
verb — the cut-out picture the preview needs is now made on demand when you switch it to `Cutout`.

The header also carries **Clear**: it drops the staged image, the mask and the PhotoCraft pin and
puts the app back to its initial state (`POST /api/reset`) — nothing in PhotoCraft is touched.

**Quick Export** sits beside Clear and takes no parameters: one press writes the picture PhotoCraft
has open to disk the way the app draws it (unsaved edits, adjustment layers, filters — the
composite, not the stale file), beside the picture it came from, with the next free version number
on the end: `<name>_v1.png`, then `<name>_v2.png`, and so on. A picture the Generate tab made is a
copy of a render in ComfyUI's output tree, so it goes back **there** — the post-processed version
lands next to the render it was made from (measured: `(2026-10-24)(18-00-07)__01_v1.png` in
`~/comfy/ComfyUI-qwen21/output/(2026-10-24)/(2026-10-24)_png/`). A document with no file of its own
falls back to `~/Pictures/photocraft-ai/`. `doc_export` may only write inside PhotoCraft's
automation root, so the composite is staged in `~/Pictures/cutout-studio/_quick-export/` and then
copied to its real home by this server — the version number is decided on the destination folder,
counting every extension so a number is never reused.

**Post-processing** sits in the Generate tab, folded away directly under Generation Setup, on **one
row**: `☐ Vibrance` and `☐ Curves`. No parameters, because the values are the document's own: the
finishing layers in the user's renders carry vibrance +30 / saturation 0 and a composite curve of
0→0, 97→92, 164→177, 255→255, so a press adds *that* look rather than an identity adjustment that
would change nothing. The layer lands above the active layer, the way `Layer > New Adjustment Layer`
does; with nothing selected the top of the stack is taken first so there is always a target.

The two halves of a row do different jobs. The **name is the button**: press it to add another
adjustment layer to the picture open now. The **checkbox is a standing choice**: ticked, every
picture the engine delivers afterwards is finished with that adjustment before the panel even
reports it; unticked, nothing is added. Ticking also applies to the open document straight away
(add if missing, show if hidden) and unticking hides the layers of that kind — nothing is ever
deleted, so the tick never lies about what is on screen. The choice lives in
`data/postprocess.json`, so it survives a reload.

Arrival uses *ensure* semantics, not blind adding: a document that already carries a layer of that
kind (an I2I render layered into a finished picture) has it switched on instead of getting a second
copy — two Vibrance layers at +30 would double the effect. `deliver_ai_image` runs
`_apply_postprocess_prefs()` after both delivery modes and reports what it added under
`adjustments`. The engine asks over `postMessage` (`photocraft-ai-postprocess`: `add`, `toggle`,
`state`), and the panel pushes `photocraft-ai-delivered` back so the section's header keeps up.

## What it does

- **Select by phrase.** SAM3 is open-vocabulary: the phrase is the prompt, no training and no
  click-through. Comma-separate to union several: `hair, face`.
- **Invert** to keep everything *except* the phrase — that is background removal in one press.
- **Cut out.** Alpha cut-out with a feather slider, optionally flattened onto a solid colour or
  a gradient. Output is a PNG at the source resolution.
- **Outline.** The mask vectorised with potrace and simplified (Douglas-Peucker), returned in
  image pixels — the form PhotoCraft's `select.lasso` takes.
- **Pull from PhotoCraft.** `From PhotoCraft` takes the document open there and makes it the
  source here — exported through the app itself, so adjustments and unsaved edits are what gets
  segmented. The document's identity is then *pinned*: results go back into that document and no
  copy of the image is ever opened beside it. This happens by itself when the page loads with no
  source yet.
- **Apply in PhotoCraft.** The outline is installed as a real lasso selection in the pinned
  document and then either left live, turned into a layer mask (`layer.layerMask.revealSelection`,
  background hidden non-destructively) or copied onto its own layer
  (`layer.new.layerViaCopy` + rename), with `select.deselect` afterwards so the picture reads
  clean. PhotoCraft's own measured selection area is read back and compared with the mask.
- **Export files.** `<slug>-cutout.png`, `-mask.png`, `-selection.json` and `<slug>-source.png`
  under `~/Pictures/cutout-studio/`, nothing pushed.

## Backends

`config.json` lists candidates; the first one that is *actually usable* wins:

| stack | port | note |
|---|---|---|
| `~/comfy/ComfyUI-qwen21` | 8189 | has `sam3.1_multiplex_fp16.safetensors` in `models/checkpoints` — the default |
| `~/comfy/ComfyUI` | 8188 | used only if the checkpoint is added there too |

"Usable" means the `SAM3_Detect` node answers **and** the checkpoint is in that stack's
checkpoint list. Nothing is assumed: a stack that would fail mid-request is skipped with the
reason shown in the header badge. If none is up, the app starts the first stack that has the
checkpoint on disk and waits up to 150 s.

SAM3 needs no sampler and no VAE — the graph is four nodes (checkpoint → CLIP text → detect →
mask image), so a phrase costs ~4 s on the 4060 Ti.

## MCP tools

Exposed at `/mcp` by the running app (FastMCP, streamable HTTP). Registered in Hermes as the
`cutout-studio` server, so tools appear as `mcp__cutout_studio__*`:

| tool | does |
|---|---|
| `cutout_status` | which stack is usable, models, phrase chips, export dir |
| `cutout_segment(image_path, phrase, threshold, refine, invert)` | mask PNG + coverage + bbox + warnings |
| `cutout_cutout(image_path, phrase, feather, background)` | alpha cut-out PNG (or on `#rrggbb` / `gradient:#a,#b`) |
| `cutout_outline(image_path, phrase, simplify)` | polygons in image pixels |
| `cutout_from_photocraft(phrase, apply, feather, threshold, refine, hair_refine, refine_radius, invert)` | the whole loop in one call: pull the document open in PhotoCraft, segment, apply there (`selection` / `mask` / `layer`), optionally through PhotoCraft's Refine Edge (`hair_refine="smart"`) |
| `cutout_to_photocraft(image_path, phrase, feather, background, mode, apply)` | push a file in instead: `cutout` / `open` / `selection` (default), `apply` as above |
| `cutout_quick_export()` | no arguments: the document PhotoCraft has open is written back beside the picture it came from, next version number (`_v1`, `_v2`, …) — a render the engine made goes back into ComfyUI's output folder |
| `cutout_adjustment(action, kind, enabled)` | `kind` = `vibrance` / `curves`; `action` = `add` (a layer above the active layer, with the document's default values) / `toggle` (remember the choice for new pictures *and* apply it now) / `pref` (remember it only) / `enable` / `disable` / `state` |

`background` accepts `transparent` (default), a colour (`#ffffff`), or `gradient:#101820,#f0f4ff`.

## HTTP API (what the browser uses)

| route | body → answer |
|---|---|
| `POST /api/upload` | multipart file → `{id, url, width, height}` — API only, no UI |
| `POST /api/open` | `{path}` → same, for a file already on disk — API only, no UI |
| `POST /api/segment` | `{id, phrase, threshold, refine, invert}` → mask + stats |
| `POST /api/cutout` | `{id, mask_id, feather, bg}` → cut-out PNG |
| `POST /api/outline` | `{mask_id, simplify}` → rings |
| `POST /api/export` | `{id, mask_id, phrase, feather, bg, mode, apply, refine, refine_radius}` → exported paths (+ PhotoCraft) |
| `POST /api/photocraft/pull` | — → `{source, doc, how, pin}`: the document open in PhotoCraft, staged as the source |
| `POST /api/quick-export` | — → `{ok, path, folder, version, doc, from, origin}`: what PhotoCraft has open, written beside its own picture with the next version number |
| `POST /api/photocraft/adjustment` | `{action, kind, enabled}` → `{doc, kind, layer, above, params, count, message, prefs}`: add or switch a Vibrance / Curves layer in the document PhotoCraft has open, and remember the choice for new pictures |
| `POST /api/reset` | — → `{ok}`: forget staged sources, masks and the PhotoCraft pin |
| `GET  /api/state` | backend/PhotoCraft status, chips |

## PhotoCraft integration

Two directions, one of which is the everyday one:

- **Pull (the front door).** `services.photocraft_pull()` / `POST /api/photocraft/pull` /
  `From PhotoCraft`: `doc_inspect` to learn which document is open, then `doc_export` to a path
  under `~/Pictures/cutout-studio/_from-photocraft/` (the app's write root), and that file becomes
  the staged source. Exporting the composite rather than reading the document's own file is
  deliberate: adjustments, masks and unsaved edits are what the user is looking at. Fallback when
  the export is refused: copy `doc["path"]`. The document's name/size are stored in `_pc_pin`.
- **Push (pinned).** `send_to_photocraft(..., apply=...)`. When the source came out of PhotoCraft
  (`_pc_pin["sid"] == sid`) nothing is ever `doc_open`ed: the outline is installed in that
  document, and if PhotoCraft has switched documents since the pull the call refuses with the
  names rather than editing the wrong picture.

| mode | what happens |
|---|---|
| `cutout` | files only: `<slug>-<stamp>-cutout.png`, `-mask.png`, `-selection.json`, plus `<slug>-<sid>-source.png` |
| `open` | export, ensure PhotoCraft, `doc_open` the **source** image (no selection) — the unpinned flow |
| `selection` | install the outline as a lasso selection in the target document, read the app's measured area back, then apply it (below) |

| apply | what the user sees in PhotoCraft |
|---|---|
| `selection` | the live lasso (marching ants), pixels untouched — **default**, what `Execute Cutout` means |
| `matte` | a real per-pixel alpha (`matte.py` + BiRefNet) carried into the document as a layer — opt-in cut-out, see "Real alpha (matte) delivery" |
| `mask` | `layer.layerMask.revealSelection` — a layer mask, background hidden non-destructively |
| `layer` | `layer.new.layerViaCopy` + `layer.renameLayer "Cutout · <phrase>"` — subject on its own layer |

Every `mask`/`layer` run ends with `select.deselect` so the picture reads clean. `mask` refuses when
the active layer already carries a mask (PhotoCraft would silently stack one) and says which layer
to clear first.

- **Back to disk (Quick Export).** `services.quick_export()` / `POST /api/quick-export` /
  `Quick Export`: `doc_export` the live composite into `_quick-export/`, then copy it beside the
  file the document came from with `<stem>_v<n><ext>` (PNG copy, or a JPEG/WebP re-encode at quality
  95 when the original is one). Where "its own file" is comes from three places, in order: the
  render recorded at delivery time (`data/ai-origins.json`, written by `deliver_ai_image` from the
  engine's own `filename` + `subfolder` — a render is *copied* into `~/Pictures/photocraft-ai/`, so
  the document's path is the copy, not the render), then the document's own `path`, then
  `~/Pictures/photocraft-ai/` for a document that has never been saved. The ComfyUI root is read
  from the engine's `config.json` (`comfy_root`), because its `output_dir` key names the *main*
  stack while its renders land under the qwen21 clone. Verified: `deliver` → `quick-export` wrote
  `(2026-10-24)(17-55-45)__01_v1.png` next to the render, pixel-identical to it (diff 0.0).

- **Post-processing adjustments.** `services.photocraft_adjustment()` / `POST
  /api/photocraft/adjustment` / `cutout_adjustment` drive `layer.newAdjustmentLayer.vibrance` and
  `.curves` through `command_run` with `ADJUSTMENT_DEFAULTS` — the values read out of the user's own
  finishing layers, so the panel's two buttons have nothing to ask for. `add` selects the top of the
  stack first when nothing is selected (the command inserts above the active layer, and a document
  with no selection would otherwise have no "above"), then finds the new layer by diffing the layer
  ids before and after — the registry command does not return it. The checkbox path uses
  `layer.showLayers` / `layer.hideLayers` with an explicit `layer` id, selecting each one first
  because those commands read the active layer. Verified end to end through the real UI: a mouse
  press added "Vibrance 2" above "Vibrance 1" at vibrance 30 / saturation 0, unticking the box hid
  both Vibrance layers and said so in the panel log, ticking it brought them back, and the Curves
  default came through as `[[0,0],[97,92],[164,177],[255,255]]`. The automation was measured with a
  real delivery: both boxes ticked, a test picture delivered as its own document arrived as
  `Curves 1 / Vibrance 1 / Background` with `adjustments: ["Vibrance", "Curves"]` reported, and the
  test document and file were removed afterwards.

### Real alpha (matte) delivery — the Affinity-grade path

`apply="matte"` — opt-in, `Send ▸ Show it in PhotoCraft as ▸ Cut out — hair matte` — stops
delegating the edge to PhotoCraft:

1. **The alpha is computed here.** `matte.py` runs **BiRefNet-portrait** (the 928 MB ONNX rembg
   ships, through `onnxruntime` in this venv) for a per-pixel alpha, then composes it with the SAM3
   mask: `alpha = 1` inside a core eroded by 6 px — the model ghosts the hair body on its own —,
   matte alpha in the rim band, strand alpha outside the polygon. `decontaminate()` then removes the
   old background's colour from partially transparent pixels (`f = (c - (1-a)·b) / a`, i.e. unpremultiply
   against the local background colour), because otherwise every wisp carries a pale halo.
   Measured: **38,378 semi-transparent px, 7.8–10.8 s on CPU**, strand retention 1.0 by construction.
2. **The PNG is written inside PhotoCraft's automation root** (`~/Pictures/cutout-studio/_matte/`),
   because PhotoCraft refuses to open documents outside it.
3. **There is no import command, so the layer is carried over.** `doc_place` does not exist and
   `file.placeEmbedded` is disabled in control mode. Instead the matte PNG is `doc_open`ed as its own
   document and its layer is copied into the target with
   `layer.copyToDocument {document: <index>, layer: <id>}`; the index is found by walking
   `document.activate {document: 0,1,2,…}` and comparing `doc_inspect`'s `name` (it answers
   `"no document at index N"` past the end and does **not** raise). The matte document is closed with
   `file.close` afterwards, and the copied layer is renamed `Cutout · <phrase>`.
4. **The layer it was cut from is hidden, not deleted.** Without that nothing appears to happen —
   the opaque original is still underneath the cut-out, exactly as in Affinity's "new layer with
   mask" output.
5. End to end through the API: **12.9–14.7 s**, verified in the live app (layer `Cutout · person`,
   `hidden_layer` reported, canvas on a checkerboard with individual strands at the bun and wisps
   across the cheek, no halo).

**`document` indices are positions, not names** — copying to index 0 while a scratch document was at
the head of the list put a layer into the user's own document during development. Locate the target
by name every time.

### A hidden layer exports as an empty PNG

`doc_export` of a document whose layers are all hidden writes a perfectly transparent image, and
then every phrase "selects 0 px" — indistinguishable from a segmentation failure (measured: a
1024² document with one `visible: false` pixel layer exported alpha = 0 everywhere, and `hair`,
`person`, `face`, `woman`, `subject` all came back empty). `photocraft_pull()` therefore checks the
exported alpha before staging it and refuses with the actual reason: *"…exports as an empty picture:
its layer is hidden — turn the eye back on in PhotoCraft's Layers panel"*.

### Fine hair: a polygon edge cannot carry wisps

SAM3 answers with a **binary** mask — the node binarises at probability 0.5
(`(refined_logit > 0) | (coarse > 0)` in `comfy_extras/nodes_sam3.py`) — and the outline is a
simplified polygon, so every wisp is cut at the silhouette. Two things were measured on a 1024²
portrait:

- **The Threshold slider does nothing for hair.** `SAM3_Detect`'s `threshold` only filters
  *detections* (box scores); the mask pixels come from the decoder crop. Phrase `person` at
  threshold 0.15 / 0.50 / 0.85 gave byte-identical masks (XOR 0 px).
- **`Refine` iterations do add edge detail**: refine 0/2/4 → 623,815 / 628,732 / 632,777 px.

**PhotoCraft cannot do this properly, and that is now measured rather than guessed.** Its own
`layer.removeBackground` is not a matting engine either: on the portrait it produced 4,547
semi-transparent px out of a million — a hard mask. A seven-way sweep of `select.refineEdge`
(radius 6 / 30, feather 0 / 0.3 / 2, bare vs brush strokes, stroke size 26 vs 10) topped out at
~39.8k semi-transparent px with only **0.75 of the true strand pixels kept**; at 6× zoom the edge is
a translucent band that also thins the hair's *body*, and small `fine` strokes scored *worse*
(0.65). That is why the default path no longer delegates the edge: it computes a matte. Refine Edge
stays available (`refine="smart"`) as the fast fallback when a layer mask is what you want.

What Refine Edge (`select.refineEdge`, the Select-and-Mask engine) does, run on our lasso:

    select.refineEdge {radius: 6, smartRadius: true, smooth: 6, feather: 0.3, contrast: 10,
                       shiftEdge: 8, decontaminate: true, amount: 60,
                       refineStrokes: [{points, size: 26, mode: "add"}, …],
                       output: "layerMask" | "selection" | "newLayerWithMask"}

`refine="smart"` in the API/UI (`Refine hair edge` checkbox, radius 1–20 px, default 6) generates
the Refine Edge brush strokes from the mask itself: `_hair_refine_strokes()` takes the mask
boundary, keeps only the parts where a `block`×`block` luminance-detail map says "strands"
(`_detail_map`, std ≥ 9), thins it to one point per ~13 px and splits it into ≤24 polylines.
Measured on the same portrait, plain vs refined:

| | area>128 | soft-edge px | gained | IoU |
|---|---|---|---|---|
| plain polygon | 632,695 | 4,584 | — | — |
| Refine Edge 6 px + 12 strokes (305 pts) | 639,711 (+1.11%) | **39,455** (8.6×) | +10,798 | 0.977 |

BiRefNet alone is not the answer either — on its own it ghosts the hair body (105,495 soft px,
including the interior). It becomes the answer once its interior is replaced by the SAM3 core and
its edge colour is decontaminated; that is `matte.py`, wired in as the default `matte` mode. See the
next section.

Mechanics that matter:

- **Launch PhotoCraft with an automation root.** `PHOTOCRAFT_AUTOMATION_READ_ROOT` and
  `_WRITE_ROOT` = `~/Pictures`. Without a root, `doc_open`/`doc_export` are refused in bridge mode
  ("automation filesystem access is not granted"). `ensure_control()` starts
  `flatpak run --share=network --command=photocraft … --control 7916 --control-token <tok>`
  with a curated env (no `LD_LIBRARY_PATH`, correct `WAYLAND_DISPLAY`, `XDG_RUNTIME_DIR`,
  DBus address) and waits for the port. Pull deliberately does **not** launch PhotoCraft: a page
  load must not open an empty editor.
- **The window has to be drawing frames.** An occluded PhotoCraft answers nothing: every engine
  call hangs until the 55 s internal timeout (measured: `doc_inspect` timed out at 55 s with the
  terminal focused, answered in 0.0 s right after `_raise_window()`). Both pull and push raise the
  window first and retry once after raising, because the failure mode is a timeout, not an error.
- **Commands are not tools.** `select.lasso` runs through `command_run` with
  `{"id": "select.lasso", "params": {"points": [[x,y],…], "mode": "replace|add|subtract"}}`;
  the first outer ring is `replace`, further outers `add`, holes `subtract`.
- **`doc_inspect` returns the document's `path`** — useful, but it is the file on disk, not what is
  on screen. Segment the export, not the path.
- **The source file has no timestamp** (`<slug>-<sid>-source.png`) so the unpinned flow reuses the
  document already open instead of piling up an identical second one.
- Tool payloads must be read **whole**: `doc_inspect` returns >2 KB of JSON, so any helper that
  truncates a tool result silently breaks the parse (the selection would land while the app
  reported a nameless document).

Verified on live runs (2026-10-24, 1024×1024 portrait, `hair-and-subject-subject-hair-<sid>-source.png`):

| run | result |
|---|---|
| pull | 2.6 s, `exported the live document`, pinned to that document |
| `person` → `selection` | 49 lasso steps, PhotoCraft 633,027 px vs mask 633,340 px — **0.05%** |
| `hair` → `selection` | 53 steps, 244,139 vs 244,091 px — **0.02%** |
| `layer` | new layer `Cutout · person` above `Layer 0`, selection cleared |
| `matte` | 14.7 s end to end — 38,378 strand px (matte 10.5 s), layer `Cutout · person` copied in, source layer hidden, matte document closed, canvas shows strands on transparency |
| `mask` + `refine="smart"` | 8 lasso steps, "a refined layer mask on Background copy 2", PhotoCraft 638,545 px vs mask 638,460 px — **0.01%** |
| `mask` | `hasMask: true` on `Layer 0`; screenshot shows the subject on transparency with the mask thumbnail |
| UI path | page load auto-pulled; typing `person` + `Execute Cutout` segmented (60.4%, 1.1 s) and installed the selection in PhotoCraft with no second press |

### Two PhotoCraft behaviours worth knowing before poking it

- **Mask commands are destructive over cycles.** Creating and deleting a layer mask repeatedly on
  the *same* document left `doc_export` returning a fully transparent composite (measured twice:
  `opaque=0`), while the identical call on a freshly opened document gave 630,459 opaque px. Test
  mask paths on a new document, not by cycling one.
- **`doc_export` only writes inside the automation root.** Anything else is refused with
  "absolute paths must be inside the automation root", so the scratch/A-B exports live under
  `~/Pictures/cutout-studio/_scratch/`.

## Traps this app is built around (all measured)

1. **ComfyUI caches an unchanged graph.** Repeating the same phrase re-executes nothing and the
   history entry has no `outputs` block at all. Every submission carries a unique SaveImage
   prefix, which also makes *this* request's file identifiable — reading by a shared prefix can
   hand back a previous request's mask.
2. **A missing phrase returns HTTP 200 with a degenerate mask** — empty, a speck, or the whole
   frame (measured: `hat` 0 px, `a blue balloon` 684 px, `spaceship` 98.5% of the frame). Below
   ~0.05% (and 400 px) it is refused with a reason; above 85% it only *warns*, because a tight
   crop of one person legitimately covers 91.3% (`person` on a torso crop).
3. **A mask PNG is opaque grayscale.** White = selected, black = not — so it must be converted
   to an alpha channel before it can mask anything. Drawing it with `destination-in` in a
   canvas does nothing on its own, which tints the whole picture instead of the selection.
   `maskAlphaCanvas()` does the luminance→alpha pass once.
4. **`np.cross` refuses 2-vectors in numpy 2** — the Douglas-Peucker distance is computed as a
   spelled-out 2-D cross product.
5. **mcp 2.x renamed FastMCP to MCPServer.** The venv pins `mcp<2` for the v1 `FastMCP` /
   `ClientSession` API.
6. **The PhotoCraft bridge needs the app's window drawing frames.** With the window occluded
   the control port accepts the connection and every engine call times out after 55 s
   (`ui.inspect timed out … the app was not drawing frames`). Bring the window forward first.
   Forcing it from a script: load a KWin script through
   `qdbus6 org.kde.KWin /Scripting org.kde.kwin.Scripting.loadScript <js>` then `.start`.
7. **`flatpak` dies with `undefined symbol: ngtcp2_conn_get_tls_early_data_rejected2`** when
   linuxbrew's `LD_LIBRARY_PATH` is set — it shadows the system libngtcp2 that
   `/usr/lib/libcurl.so.4` needs. Strip linuxbrew dirs first (`start.sh` and the launcher
   wrapper both do).
8. **PhotoCraft's MCP surface has no per-command tools.** `select.lasso` is not a tool name;
   commands run through `command_run` with `{"id": "select.lasso", "params": {...}}`.

## Verified

- `hair` on a 500×600 portrait: 26.82% coverage, 4.0 s, mask visually correct.
- Union `hair, face` on a crop with no face: per-phrase stats report `face` at 0 px, and a
  warning says the selection is the other phrases only.
- Cut-out, mask export, and outline (11 rings, main ring 97 points after simplification).
- PhotoCraft: 11 lasso steps injected, `hasSelection: true`, app-measured selection area
  **80,480 px** against the SAM3 mask's **80,464 px** — 0.02%, one main feature.

## Look — one theme, measured from the editor

There is no theme picker. The app is a pane of PhotoCraft, so it wears the editor's own palette,
sampled from a live `ui_screenshot` of PhotoCraft's window (not eyeballed):

| role | value |
|---|---|
| window chrome / dock bodies | `#535353` |
| tab strips, panel headers, inactive tabs | `#4a4a4a` (the active tab takes the body's `#535353`) |
| input wells | `#454545` on `#343434` seams |
| canvas surround (the stage) | `#282828` |
| text / muted | `#eeeeee` / `#a0a0a0` |
| accent | `#378ef0` (PhotoCraft's own blue button) |

Shape language: **buttons are pills** (`border-radius: 999px`), everything else is square with 3 px
corners, 1 px seams, no glow and no translucency. Type is 11-12.5 px Inter over JetBrains Mono for
figures. The same tokens are restated in the engine's `[data-theme="photocraft"]` block so the
embedded Generate pane is the same application, at the same density.

## Layout

    server.py       FastAPI app, routes, MCP tools, /mcp mount
    services.py     backend discovery, SAM3 graph, coverage guards, cut-out, outline, PhotoCraft
    static/         index.html, styles.css (PhotoCraft graphite — one theme, no picker), app.js
    data/           sources/ masks/ cutouts/ (served at /files)
    start.sh        venv + uvicorn
