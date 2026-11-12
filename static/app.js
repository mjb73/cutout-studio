/* Cutout Studio — UI.  One canvas, three views (original / overlay / cutout). */
'use strict';

const $ = (s) => document.querySelector(s);
const el = {
  canvas: $('#canvas'),
  stage: $('#stage'),
  empty: $('#empty'),
  phrase: $('#phrase'),
  chips: $('#chips'),
  threshold: $('#threshold'),
  thv: $('#thv'),
  refine: $('#refine'),
  rfv: $('#rfv'),
  invert: $('#invert'),
  find: $('#findbtn'),
  feather: $('#feather'),
  ftv: $('#ftv'),
  bgtype: $('#bgtype'),
  bgc1: $('#bgc1'),
  bgc2: $('#bgc2'),
  bgtwo: $('#bgtwo'),
  pcfile: $('#pcfile'),
  previewbtn: $('#previewbtn'),
  clearbtn: $('#clearbtn'),
  qe: $('#quickexportbtn'),
  pcline: $('#pcline'),
  pinline: $('#pinline'),
  applyas: $('#applyas'),
  hairrefine: $('#hairrefine'),
  hairradius: $('#hairradius'),
  hrv: $('#hrv'),
  hairfield: $('#hairfield'),
  refinerow: $('#refinerow'),
  status: $('#status'),
  logdock: $('#logdock'),
  logbar: $('#logbar'),
  loglast: $('#loglast'),
  btnlog: $('#btnLog'),
  srcname: $('#srcname'),
  metasize: $('#meta-size'),
  metasel: $('#meta-sel'),
  brandsub: $('#brandsub'),
  winname: $('#winname'),
  winclose: $('#winclose'),
  tabs: [...document.querySelectorAll('#tabs .tab')],
  genframe: $('#genframe'),
  genline: $('#genline'),
  aimode: $('#aimode'),
  tabcutout: $('#tab-cutout'),
  tabgenerate: $('#tab-generate'),
};

const state = {
  src: null,        // {id,url,width,height,name}
  img: null,        // HTMLImageElement of the source
  mask: null,       // {mask_id,url,coverage,bbox,...}
  maskImg: null,
  cutout: null,     // {url,width,height}
  cutoutImg: null,
  pin: null,        // the PhotoCraft document the source was pulled from
  liveDoc: null,    // the document PhotoCraft is showing right now
  view: 'original',
  zoom: 'fit',
  busy: false,
};

/* ------------------------------------------------------------------ helpers */

// The log is the record of the session, not a ticker: nothing is dropped. It is a scrolling pane.
const MAX_LOG_LINES = Infinity;

/* The bar always carries the newest message, so a folded log never hides a failure; the log itself
   keeps the history behind the Log button. */
function status(html, cls) {
  const line = document.createElement('div');
  line.className = 'line ' + (cls || '');
  line.innerHTML = html;
  el.status.prepend(line);
  while (el.status.children.length > MAX_LOG_LINES) el.status.lastChild.remove();
  if (el.loglast) {
    el.loglast.className = 'line ' + (cls || '');
    el.loglast.innerHTML = html;
  }
  if (el.btnlog) el.btnlog.classList.toggle('flagged', cls === 'bad');
}

const LOG_KEY = 'cutout-log-open';

/* One owner for the folded state: the attribute, aria-pressed, the title and the saved choice. */
function setLogOpen(open, remember) {
  el.status.hidden = !open;
  if (el.btnlog) {
    el.btnlog.setAttribute('aria-pressed', open ? 'true' : 'false');
    el.btnlog.classList.toggle('on', open);
    el.btnlog.title = open ? 'The generation log is showing — click to hide it'
                           : 'The generation log is hidden — click to show it';
  }
  if (remember) {
    try { localStorage.setItem(LOG_KEY, open ? '1' : '0'); } catch (e) { /* private mode */ }
  }
}

function accent() {
  return getComputedStyle(document.documentElement).getPropertyValue('--accent').trim() || '#8df574';
}

function loadImage(url) {
  return new Promise((resolve, reject) => {
    const im = new Image();
    im.onload = () => resolve(im);
    im.onerror = () => reject(new Error('could not load ' + url));
    im.src = url;
  });
}

function setBusy(on, label) {
  state.busy = on;
  el.srcname.textContent = state.pin
    ? ''                                            // the PhotoCraft line names the document
    : (state.src ? `${state.src.width}×${state.src.height}` : 'no picture pulled yet — press Execute Cutout');
  el.find.disabled = on;
  el.pcfile.disabled = on || !state.mask;
  if (el.qe) el.qe.disabled = on;
  el.find.textContent = on ? (label || 'Working…') : 'Execute Cutout';
}

async function post(url, body) {
  const r = await fetch(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
  const data = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(data.detail || ('HTTP ' + r.status));
  return data;
}

/* ------------------------------------------------------------------ drawing */

// The mask PNG is opaque grayscale (white = selected), so it must be turned into an alpha
// channel before it can mask anything - drawing it with destination-in does nothing on its
// own. Luminance becomes alpha here once, and both the tint and the cut-out build on that.
function maskAlphaCanvas(w, h) {
  const off = document.createElement('canvas');
  off.width = w; off.height = h;
  const o = off.getContext('2d');
  o.drawImage(state.maskImg, 0, 0, w, h);
  const d = o.getImageData(0, 0, w, h);
  const px = d.data;
  for (let i = 0; i < px.length; i += 4) {
    px[i + 3] = Math.round(px[i] * (px[i + 3] / 255));
  }
  o.putImageData(d, 0, 0);
  return off;
}

function tinted(maskCanvas, w, h, color) {
  const off = document.createElement('canvas');
  off.width = w; off.height = h;
  const o = off.getContext('2d');
  o.drawImage(maskCanvas, 0, 0);
  o.globalCompositeOperation = 'source-in';
  o.fillStyle = color;
  o.fillRect(0, 0, w, h);
  return off;
}

function draw() {
  const c = el.canvas;
  const im = state.img;
  if (!im) return;
  if (document.body.classList.contains('compact')) return;   // the stage is hidden; nothing to paint
  const box = el.stage.getBoundingClientRect();
  const pad = 18;
  let scale = 1;
  if (state.zoom === 'fit') {
    scale = Math.min((box.width - pad) / im.naturalWidth,
                     (box.height - pad) / im.naturalHeight, 1);
  }
  const w = Math.max(1, Math.round(im.naturalWidth * scale));
  const h = Math.max(1, Math.round(im.naturalHeight * scale));
  c.width = w;
  c.height = h;
  c.style.width = w + 'px';
  c.style.height = h + 'px';

  const ctx = c.getContext('2d');
  ctx.clearRect(0, 0, w, h);

  if (state.view === 'cutout') {
    if (state.cutoutImg) {
      ctx.drawImage(state.cutoutImg, 0, 0, w, h);
    } else if (state.maskImg) {
      // Local composite, so Cutout works before pressing Make cutout.
      const m = maskAlphaCanvas(w, h);
      const off = document.createElement('canvas');
      off.width = w; off.height = h;
      const o = off.getContext('2d');
      o.drawImage(im, 0, 0, w, h);
      o.globalCompositeOperation = 'destination-in';
      o.drawImage(m, 0, 0);
      ctx.drawImage(off, 0, 0);
    } else {
      ctx.drawImage(im, 0, 0, w, h);
    }
  } else if (state.view === 'overlay' && state.maskImg) {
    // Dim everything, then paint the selection back at full strength, then tint it.
    const m = maskAlphaCanvas(w, h);
    const lit = document.createElement('canvas');
    lit.width = w; lit.height = h;
    const lc = lit.getContext('2d');
    lc.drawImage(im, 0, 0, w, h);
    lc.globalCompositeOperation = 'destination-in';
    lc.drawImage(m, 0, 0);

    ctx.drawImage(im, 0, 0, w, h);
    ctx.save();
    ctx.globalAlpha = 0.62;
    ctx.fillStyle = '#000';
    ctx.fillRect(0, 0, w, h);
    ctx.restore();
    ctx.drawImage(lit, 0, 0);

    const t = tinted(m, w, h, accent());
    ctx.save();
    ctx.globalAlpha = 0.34;
    ctx.drawImage(t, 0, 0);
    ctx.restore();

    const bbox = state.mask && state.mask.bbox;
    if (bbox) {
      ctx.save();
      ctx.strokeStyle = accent();
      ctx.lineWidth = 1;
      ctx.setLineDash([5, 4]);
      ctx.strokeRect(bbox[0] * scale + 0.5, bbox[1] * scale + 0.5,
                     bbox[2] * scale, bbox[3] * scale);
      ctx.restore();
    }
  } else {
    ctx.drawImage(im, 0, 0, w, h);
  }

  el.stage.classList.toggle('cutout', state.view === 'cutout');
  el.empty.style.display = 'none';
  el.metasize.textContent = `${im.naturalWidth} × ${im.naturalHeight}`;
  el.metasel.textContent = state.mask ? `${state.mask.coverage.toFixed(1)}% selected` : '';
}

/* ------------------------------------------------------------------ actions */

async function adoptSource(info) {
  state.src = info;
  state.mask = null; state.maskImg = null;
  state.cutout = null; state.cutoutImg = null;
  state.view = 'original';
  setView('original');
  el.srcname.textContent = (state.pin ? state.pin.doc + ' · ' : '') +
    `${info.width}×${info.height}`;
  state.img = await loadImage(info.url + '?t=' + Date.now());
  el.find.disabled = false;
  setBusy(false);
  draw();
  status(`Loaded <span class="meta">${info.width}×${info.height}</span>. Type what to select, then Execute Cutout.`, 'ok');
}

function updatePinUi() {
  el.pinline.textContent = '';   // only the "different document open" warning belongs here
}

async function pullFromPc() {
  if (state.busy) return;
  setBusy(true, 'Reading PhotoCraft…');
  try {
    const res = await post('/api/photocraft/pull', {});
    state.pin = res.pin;
    await adoptSource(res.source);
    updatePinUi();
    const d = res.doc || {};
    status(`Pulled <span class="meta">${d.name}</span> (${d.width}×${d.height}) out of PhotoCraft`
      + ` — ${res.how}. The result goes back into that document.`, 'ok');
  } catch (e) {
    status(e.message, 'bad');
  } finally {
    setBusy(false);
    refreshPc();
  }
}

async function find() {
  if (state.busy) return;
  const phrase = el.phrase.value.trim();
  if (!phrase) { status('Type what to select first — "hair", "the dress", "mountain".', 'bad'); el.phrase.focus(); return; }
  // The picture is PhotoCraft's: with nothing pulled — or with a different document open there
  // now — Find pulls it first instead of asking for a file.
  if (!state.src || (state.pin && state.liveDoc && state.liveDoc !== state.pin.doc)) {
    await pullFromPc();
    if (!state.src) return;   // pullFromPc already explained why
  }
  let show = false;
  setBusy(true, 'Segmenting…');
  try {
    const res = await post('/api/segment', {
      id: state.src.id,
      phrase,
      threshold: parseFloat(el.threshold.value),
      refine: parseInt(el.refine.value, 10),
      invert: el.invert.checked,
    });
    state.mask = res;
    state.cutout = null; state.cutoutImg = null;
    state.maskImg = await loadImage(res.mask_url + '?t=' + Date.now());
    setView('overlay');
    draw();
    status(`<span class="meta">${res.coverage.toFixed(1)}%</span> selected in ${res.elapsed}s`
      + (res.inverted ? ' (inverted)' : '') + ` · ${res.backend}`, 'ok');
    (res.warnings || []).forEach((w) => status(w, 'warn'));
    // The work happens here and the result belongs in PhotoCraft: one press, one outcome, and the
    // mode select decides whether that outcome is ants, a cut-out, a mask or a layer.
    show = true;
  } catch (e) {
    status(e.message, 'bad');
  } finally {
    setBusy(false);
    draw();
  }
  if (show) await sendToPc('selection', el.applyas.value === 'matte'
    ? 'Cutting out — real alpha, about ten seconds…' : undefined);
}

function bgPayload() {
  const t = el.bgtype.value;
  if (t === 'solid') return { type: 'solid', color: el.bgc1.value };
  if (t === 'gradient') return { type: 'gradient', color: el.bgc1.value, color2: el.bgc2.value, angle: 90 };
  return { type: 'transparent' };
}

async function makeCutout() {
  if (!state.mask || state.busy) return;
  setBusy(true, 'Cutting…');
  try {
    const res = await post('/api/cutout', {
      id: state.src.id,
      mask_id: state.mask.mask_id,
      feather: parseFloat(el.feather.value),
      bg: bgPayload(),
    });
    state.cutout = res;
    state.cutoutImg = await loadImage(res.url + '?t=' + Date.now());
    setView('cutout');
    draw();
    status(`Cutout ready — ${res.width}×${res.height}${res.solid ? ' on background' : ' with alpha'}.`, 'ok');
  } catch (e) {
    status(e.message, 'bad');
  } finally {
    setBusy(false);
  }
}

async function sendToPc(mode, busyText) {
  if (!state.mask || state.busy) return;
  const apply = mode === 'selection' ? el.applyas.value : 'selection';
  const busyLabel = busyText || (mode === 'cutout' ? 'Exporting…'
    : apply === 'matte' ? 'Cutting out — real alpha, about ten seconds…'
    : (apply === 'selection' ? 'Selecting in PhotoCraft…' : 'Applying in PhotoCraft…'));
  setBusy(true, busyLabel);
  try {
    const res = await post('/api/export', {
      id: state.src.id,
      mask_id: state.mask.mask_id,
      phrase: el.phrase.value.trim(),
      feather: parseFloat(el.feather.value),
      bg: bgPayload(),
      mode,
      apply,
      refine: el.hairrefine.checked ? 'smart' : 'off',
      refine_radius: parseFloat(el.hairradius.value),
    });
    status((res.ok ? '<span class="ok">' : '<span class="bad">') + res.message + '</span>');
    if (res.matte_error) status('Matte failed, fell back to a selection: ' + res.matte_error, 'warn');
    if (res.doc) status('Document: <span class="meta">' + res.doc + '</span>');
    refreshPc();
    if (res.folder) status('Folder: <span class="meta">' + res.folder + '</span>');
  } catch (e) {
    status(e.message, 'bad');
  } finally {
    setBusy(false);
  }
}

/* The split column runs without a window decoration — the wrapper clears that window's border — so
   the panel's own strip is the top of the window: same colour, same edge, as the editor beside it,
   which is the whole point of stripping it. A frameless window has no title bar at all, so the strip
   takes over the two things a title bar did that a page can supply: something to drag the window by
   (the CSS app-region) and a close control. Both switch on only when the window really is frameless;
   in a browser tab outerHeight − innerHeight is the tab strip plus toolbar, tens of pixels. */
function initWindowChrome() {
  // A --app window is the only thing that reports standalone; a tab reports "browser". The size test
  // is the fallback for a build that answers differently — a decorated window is ~27 px taller than
  // its page and a few px wider, a frameless one within a pixel (measured: 430x1200 page in a
  // 435x1201 window).
  const frameless = window.matchMedia('(display-mode: standalone)').matches
    || ((window.outerHeight - window.innerHeight) < 40
        && (window.outerWidth - window.innerWidth) < 12);
  if (!frameless) return;
  document.body.classList.add('frameless');
  if (!el.winclose) return;
  el.winclose.addEventListener('click', () => {
    window.close();   // allowed for a window the browser opened, which this is
    // If it is refused, end the process that carries the --app= argument instead — its own window,
    // matched exactly as the split wrapper matches it, so the user's browser is never touched.
    setTimeout(() => { if (!window.closed) post('/api/window/close', {}).catch(() => {}); }, 300);
  });
}

function setPreview(on, remember) {
  const compact = !on;
  document.body.classList.toggle('compact', compact);
  // compact = the log is the window footer, so it is always in view and never covers a control;
  // with the preview it belongs back at the bottom of the sidebar
  // the log dock (bar + foldable log) moves as one piece
  const log = document.querySelector('#logdock');
  const app = document.querySelector('#app');
  const side = document.querySelector('#side');
  if (compact && log.parentElement !== app) app.appendChild(log);
  else if (!compact && log.parentElement !== side) side.appendChild(log);
  el.previewbtn.classList.toggle('on', on);
  if (remember) localStorage.setItem('cutout-preview', on ? '1' : '0');
  if (on) draw();
}

async function clearAll() {
  try { await post('/api/reset', {}); } catch (e) { /* the UI is reset regardless */ }
  state.src = null; state.img = null;
  state.mask = null; state.maskImg = null;
  state.cutout = null; state.cutoutImg = null;
  state.pin = null;
  state.liveDoc = null;
  state.view = 'original'; state.zoom = 'fit';
  el.phrase.value = '';
  document.querySelectorAll('#chips .chip').forEach((c) => c.classList.remove('on'));
  el.srcname.textContent = 'no picture pulled yet — press Execute Cutout';
  el.pinline.textContent = '';
  el.pinline.style.color = '';
  el.metasize.textContent = '';
  el.metasel.textContent = '';
  el.empty.style.display = '';
  const ctx = el.canvas.getContext('2d');
  ctx.clearRect(0, 0, el.canvas.width, el.canvas.height);
  el.canvas.width = 10; el.canvas.height = 10;
  updatePinUi();
  el.status.innerHTML = '<div class="line">Ready.</div>';
  setBusy(false);
  refreshPc();
}

/* One press, no parameters. Whatever PhotoCraft has open goes back beside the picture it came
   from — a picture the engine made goes back into ComfyUI's output folder, next to its render —
   with the next free version number on the end (name_v1.png, name_v2.png, …). */
async function quickExport() {
  if (state.busy) return;
  el.qe.disabled = true;
  el.qe.textContent = 'Exporting…';
  try {
    const res = await post('/api/quick-export', {});
    const where = res.origin === 'comfy' ? ' beside its ComfyUI render' : '';
    status(`<span class="meta">Quick export</span> ${res.doc} → <b>${res.path}</b>` +
           ` (v${res.version}${where})`, 'ok');
  } catch (e) {
    status('Quick export failed: ' + e.message, 'bad');
  } finally {
    el.qe.textContent = 'Quick Export';
    el.qe.disabled = state.busy;
  }
}

async function refreshPc() {
  if (!el.pcline) return;
  try {
    const s = await (await fetch('/api/photocraft/doc')).json();
    if (!s.control_up) {
      el.pcline.textContent = s.running
        ? 'PhotoCraft: open, but without its control server — quit it (Ctrl+Q) to let this app drive it'
        : 'PhotoCraft: closed — a press starts it';
      el.pcline.style.color = s.running ? 'var(--pink)' : '';
      return;
    }
    const d = s.doc;
    if (!d || !d.name) {
      el.pcline.textContent = 'PhotoCraft: empty document — open a picture there';
      return;
    }
    const sel = d.hasSelection ? ' · has a selection' : '';
    state.liveDoc = d.name;
    el.pcline.textContent = `PhotoCraft: ${d.name} · ${d.width}×${d.height}${sel}`;
    el.pcline.style.color = '';
    if (state.pin && state.pin.doc !== d.name) {
      el.pinline.textContent = `PhotoCraft now shows ${d.name} — press From PhotoCraft to switch`;
      el.pinline.style.color = 'var(--pink)';
    } else if (state.pin) {
      updatePinUi();
      el.pinline.style.color = '';
    }
    return s;
  } catch (e) {
    el.pcline.textContent = 'PhotoCraft: status unavailable';
  }
  return null;
}

// The picture already open in PhotoCraft is the picture this app works on: pick it up at boot
// instead of asking for it a second time. /api/state only asks whether the control port is up —
// reading the document itself goes through the bridge, which raises the window first.
async function autoPull() {
  if (state.src) return;
  let st;
  try { st = await (await fetch('/api/state')).json(); } catch (e) { return; }
  if (!st || !st.photocraft || !st.photocraft.control_up) { refreshPc(); return; }
  status('PhotoCraft is drivable — pulling the picture it has open.');
  await pullFromPc();
}

/* ------------------------------------------------------------------ wiring */

function setView(v) {
  state.view = v;
  document.querySelectorAll('#viewseg button').forEach((b) => b.classList.toggle('active', b.dataset.view === v));
  // The cut-out picture is only needed by the preview, so it is made when that view is asked for
  // rather than by a button of its own.
  if (v === 'cutout' && state.mask && !state.cutout && !state.busy) {
    makeCutout();
    return;
  }
  draw();
}

function initTheme() {
  // One look, no picker: the app is a pane of PhotoCraft, so it wears the editor's graphite theme.
  // (The four neon themes and the localStorage key they used are gone with the picker.)
  document.documentElement.setAttribute('data-theme', 'photocraft');
  try { localStorage.removeItem('cutout-theme'); } catch (e) { /* private mode */ }
}

function initPanels() {
  document.querySelectorAll('.panel-head').forEach((h) => {
    h.addEventListener('click', () => {
      const panel = h.parentElement;
      panel.classList.toggle('closed');
      if (!panel.classList.contains('closed') && panel.dataset.panel === 'send') refreshPc();
    });
  });
}

async function initState() {
  try {
    const st = await (await fetch('/api/state')).json();
    if (st.pin && st.pin.doc) {
      el.pinline.textContent = `PhotoCraft has ${st.pin.doc} open — press Execute Cutout to pull it`;
    }
    // The stack the panel is driving used to sit in the strip as a badge; the strip is now the app's
    // name and its buttons, so only a real problem is reported — in the log bar, which keeps the
    // newest message (a failure is what needs the room, not the name of a working engine).
    if (!st.ok) {
      status('No usable ComfyUI stack: ' + (st.backends || []).map((b) => b.name + ': ' + b.reason).join(' | '), 'bad');
    }
    el.chips.innerHTML = '';
    (st.chips || []).forEach((c) => {
      const b = document.createElement('button');
      b.className = 'chip';
      b.textContent = c;
      b.addEventListener('click', () => {
        const parts = el.phrase.value.split(',').map((s) => s.trim()).filter(Boolean);
        const at = parts.indexOf(c);
        if (at >= 0) parts.splice(at, 1); else parts.push(c);
        el.phrase.value = parts.join(', ');
        b.classList.toggle('on', at < 0);
      });
      el.chips.appendChild(b);
    });
  } catch (e) {
    status('Cannot read /api/state: ' + e.message, 'bad');
  }
}

el.phrase.addEventListener('keydown', (e) => { if (e.key === 'Enter') find(); });
el.find.addEventListener('click', find);

el.pcfile.addEventListener('click', () => sendToPc('cutout'));
el.previewbtn.addEventListener('click', () => {
  setPreview(document.body.classList.contains('compact'), true);
});
el.clearbtn.addEventListener('click', clearAll);
if (el.qe) el.qe.addEventListener('click', quickExport);

if (el.btnlog) {
  el.btnlog.addEventListener('click', () => setLogOpen(el.status.hidden, true));
  let savedLog = null;
  try { savedLog = localStorage.getItem(LOG_KEY); } catch (e) { /* private mode */ }
  // absent key = never touched = folded; present = the user's own decision
  // Shown by default: the sequence of what happened is part of the app, not a hidden drawer.
  setLogOpen(savedLog === null ? true : savedLog === '1', false);
}

el.threshold.addEventListener('input', () => { el.thv.textContent = parseFloat(el.threshold.value).toFixed(2); });
el.refine.addEventListener('input', () => { el.rfv.textContent = el.refine.value; });
el.feather.addEventListener('input', () => { el.ftv.textContent = parseFloat(el.feather.value).toFixed(1) + ' px'; });
el.hairradius.addEventListener('input', () => { el.hrv.textContent = el.hairradius.value + ' px'; });
el.hairrefine.addEventListener('change', () => {
  el.hairfield.style.display = el.hairrefine.checked ? '' : 'none';
});

// Refine Edge is a mask-mode thing: hide the whole group (and its radius) for the other outputs so
// the narrow column isn't carrying two rows of dead controls.
function updateRefineRows() {
  const isMask = el.applyas.value === 'mask';
  el.refinerow.style.display = isMask ? '' : 'none';
  el.hairfield.style.display = isMask && el.hairrefine.checked ? '' : 'none';
}
el.applyas.addEventListener('change', updateRefineRows);
el.bgtype.addEventListener('change', () => {
  el.bgtwo.style.display = el.bgtype.value === 'transparent' ? 'none' : 'flex';
});

document.querySelectorAll('#viewseg button').forEach((b) => {
  b.addEventListener('click', () => setView(b.dataset.view));
});
document.querySelectorAll('#zoomseg button').forEach((b) => {
  b.addEventListener('click', () => {
    state.zoom = b.dataset.zoom;
    document.querySelectorAll('#zoomseg button').forEach((x) => x.classList.toggle('active', x === b));
    draw();
  });
});

window.addEventListener('resize', () => draw());


// ---------------------------------------------------------------------------------------------
// Tabs: Cutout (the panel above) and Generate (the image engine, embedded). The engine is its own
// process on its own port — same as ComfyUI behind it — so this page starts it when its tab is
// first opened and then embeds it. Finished renders postMessage up and are opened in PhotoCraft.
// ---------------------------------------------------------------------------------------------
// The engine's port is the server's business (config.json ai_engine.port / PHOTOCRAFT_AI_ENGINE_PORT);
// /api/ai/status reports it, so this page never hardcodes one.
let ENGINE_PORT = 7866;

function engineUrl() {
  // same host the panel was served from, so this works over the tailnet too
  return `http://${location.hostname}:${ENGINE_PORT}/?theme=photocraft&embed=1`;
}

async function engineStatus() {
  try {
    const r = await fetch('/api/ai/status');
    const st = await r.json();
    if (st.port) ENGINE_PORT = st.port;
    el.genline.textContent = st.up
      ? 'engine: running on ' + st.port + ' — finished images open in PhotoCraft'
      : 'engine: not running — press Generate to start it';
    return st;
  } catch (e) {
    el.genline.textContent = 'engine: this panel cannot reach its own server';
    return { up: false };
  }
}

async function startEngine() {
  el.genline.textContent = 'engine: starting…';
  try {
    const r = await fetch('/api/ai/start', { method: 'POST' });
    const st = await r.json();
    el.genline.textContent = st.up
      ? 'engine: ' + (st.started ? 'started' : 'already running') + ' — finished images open in PhotoCraft'
      : 'engine: did not start';
    return st.up;
  } catch (e) {
    el.genline.textContent = 'engine: ' + e.message;
    return false;
  }
}

async function loadEngineFrame() {
  if (el.genframe.src) return;
  const up = (await engineStatus()).up || (await startEngine());
  if (!up) return;
  el.genframe.src = engineUrl();
}

function switchTab(name, remember = true) {
  const gen = name === 'generate';
  document.body.dataset.tab = gen ? 'generate' : 'cutout';
  el.tabcutout.hidden = gen;
  el.tabgenerate.hidden = !gen;
  el.tabs.forEach((b) => b.classList.toggle('active', b.dataset.tab === name));
  el.brandsub.textContent = gen ? 'generate · send to PhotoCraft' : 'make · cut out';
  if (remember) localStorage.setItem('cutout-tab', name);
  if (gen) loadEngineFrame();
  else draw();
}

// A finished render announces itself; open it in PhotoCraft the way the user chose.
window.addEventListener('message', async (ev) => {
  const d = ev.data;
  if (!d || d.type !== 'photocraft-ai-render' || !d.filename) return;
  // A render made from the picture open in PhotoCraft goes back into that same document as a new
  // layer above it, so the original stays intact underneath instead of a second document opening.
  const mode = d.from_photocraft ? 'layer' : el.aimode.value;
  status(`<span class="meta">AI</span> ${d.filename} — sending to PhotoCraft${mode === 'layer' ? ' as a new layer' : ''}…`, 'warn');
  try {
    const r = await fetch('/api/ai/deliver', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ filename: d.filename, subfolder: d.subfolder || '',
                             prompt: d.prompt || '', mode }),
    });
    const res = await r.json().catch(() => ({}));
    if (!r.ok) throw new Error(res.detail || res.message || ('HTTP ' + r.status));
    if (!res.doc) throw new Error('PhotoCraft did not report a document — press Generate again.');
    if (res.mode === 'layer') {
      const over = res.above ? ` above “${res.above}”` : '';
      status(`<span class="meta">AI</span> added as a new layer “${res.layer_name}”${over} in <b>${res.doc}</b> — the original layer is untouched.`, 'ok');
    } else {
      const size = Array.isArray(res.size) && res.size[0] ? ` (${res.size[0]}×${res.size[1]})` : '';
      status(`<span class="meta">AI</span> opened in PhotoCraft as <b>${res.doc}</b>${size}.`, 'ok');
    }
    state.liveDoc = res.doc;
    updatePinUi();
    refreshPc();
    // The Post-processing section sits in the engine's frame and its summary describes the document:
    // tell it a new picture landed (and whether the ticked adjustments came with it).
    try {
      el.genframe.contentWindow.postMessage({
        type: 'photocraft-ai-delivered', doc: res.doc, adjustments: res.adjustments || [],
      }, '*');
    } catch (e) { /* the frame is gone */ }
  } catch (e) {
    status(`<span class="meta">AI</span> could not deliver that image: ${e.message}`, 'bad');
  }
});

// The embedded engine asks for the picture open in PhotoCraft (I2I with "use PhotoCraft"):
// it is a different origin, so the panel reads the document here and posts the PNG back.
window.addEventListener('message', async (ev) => {
  const d = ev.data;
  if (!d || d.type !== 'photocraft-ai-want-source' || !ev.source) return;
  const reply = (payload, transfer) => {
    try { ev.source.postMessage({ type: 'photocraft-ai-source', id: d.id, ...payload }, '*', transfer || []); }
    catch (e) { /* the engine window went away */ }
  };
  try {
    const r = await fetch('/api/photocraft/pull', { method: 'POST' });
    const res = await r.json().catch(() => ({}));
    if (!r.ok) throw new Error(res.detail || res.message || ('HTTP ' + r.status));
    let src = res.source || {};
    if (!src.id) throw new Error('PhotoCraft gave nothing back');
    const docName = (res.doc || {}).name || 'the open document';

    // Layer references. A layer named "image 2" / "image 3" is the workflow's image_N, so the
    // prompt's "image 2" means exactly that layer. When a reference layer exists the composite is
    // no longer the picture being edited: the document's own image-1 export is, with the reference
    // layers left out of it — they are material, not something drawn on the canvas.
    //
    // Only the Qwen-Image 2.1 encoder has image_2 … image_16, and the engine says so with
    // `references: false`, which keeps every other model on a single image. Those layers are then
    // named back to the engine (from the pull's own layer list, so nothing is exported for nothing)
    // so the strip can say they are not being sent.
    const refLayerRe = /^\s*(?:image|img|picture|ref|input)\s*[_\-–]?\s*(\d{1,2})\b/i;
    const namedRefs = (res.layers || [])
      .map(l => (l && l.name) || '')
      .filter(n => {
        const m = n && n.match(refLayerRe);
        // image 1 is the picture itself, which IS handed over — only image_2 upward is left out.
        return m && parseInt(m[1], 10) > 1;
      });
    let refs = [];
    let refNote = '';
    let ignoredRefs = [];
    if (d.references === false) {
      ignoredRefs = namedRefs;
    } else {
      try {
        const rr = await fetch('/api/photocraft/refs', { method: 'POST' });
        const rj = await rr.json().catch(() => ({}));
        if (!rr.ok) throw new Error(rj.detail || rj.message || ('HTTP ' + rr.status));
        if (rj.base && rj.base.source && rj.base.source.id) src = rj.base.source;
        for (const ref of (rj.refs || [])) {
          if (!ref.source || !ref.source.id) continue;
          const got = await fetch(`/files/sources/${ref.source.id}.png?t=${Date.now()}`);
          if (!got.ok) continue;
          refs.push({ slot: ref.slot, name: ref.name, buffer: await got.arrayBuffer(), mime: 'image/png' });
        }
        if (refs.length) refNote = ' + ' + refs.map(rf => `image ${rf.slot}`).join(', ');
      } catch (e) {
        status(`<span class="meta">AI</span> could not read the reference layers: ${e.message}`, 'warn');
      }
    }

    const img = await fetch(`/files/sources/${src.id}.png?t=${Date.now()}`);
    if (!img.ok) throw new Error('the pulled picture could not be read back');
    const buf = await img.arrayBuffer();
    reply({ buffer: buf, mime: 'image/png', name: `photocraft-${src.id}.png`,
            width: src.width, height: src.height, doc: docName, refs, ignoredRefs },
          [buf, ...refs.map(rf => rf.buffer)]);
    // The result will be layered on top of that picture, so show that in the delivery choice.
    el.aimode.value = 'layer';
    status(`<span class="meta">AI</span> handed <b>${docName}</b> (${src.width}×${src.height})${refNote} to the engine as image 1${ignoredRefs.length ? ` — ${ignoredRefs.join(', ')} left out (this model takes one image)` : ''}; the render comes back as a new layer above it.`, 'ok');
    state.liveDoc = docName;
    updatePinUi();
  } catch (e) {
    reply({ error: e.message });
    status(`<span class="meta">AI</span> could not read the picture out of PhotoCraft: ${e.message}`, 'bad');
  }
});

// The embedded engine's Post-processing section adds finishing layers to the document open in
// PhotoCraft — Vibrance and Curves, above its active layer. Same request/reply shape as the source
// pull above: the engine asks, this side owns the bridge to PhotoCraft.
window.addEventListener('message', async (ev) => {
  const d = ev.data;
  if (!d || d.type !== 'photocraft-ai-postprocess' || !ev.source) return;
  const reply = (payload) => {
    try { ev.source.postMessage({ type: 'photocraft-ai-postprocess', id: d.id, ...payload }, '*'); }
    catch (e) { /* the engine window went away */ }
  };
  const kind = (d.kind || '').toLowerCase();
  try {
    const r = await fetch('/api/photocraft/adjustment', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ action: d.action || 'add', kind, enabled: d.enabled }),
    });
    const res = await r.json().catch(() => ({}));
    if (!r.ok) throw new Error(res.detail || res.message || ('HTTP ' + r.status));
    reply(res);
    if (res.message) status(`<span class="meta">POST</span> ${res.message}`, 'ok');
    refreshPc();
  } catch (e) {
    reply({ error: e.message });
    status(`<span class="meta">POST</span> ${e.message}`, 'bad');
    refreshPc();
  }
});

el.tabs.forEach((b) => b.addEventListener('click', () => switchTab(b.dataset.tab)));
el.aimode.addEventListener('change', async () => {
  try {
    await fetch('/api/ai/mode', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ mode: el.aimode.value }),
    });
    status('Generated images now go to PhotoCraft as a ' +
           (el.aimode.value === 'layer' ? 'new layer' : 'new document') + '.', 'ok');
  } catch (e) {
    status('Could not save that choice: ' + e.message, 'bad');
  }
});


/* ---- dropping a pose from the Pose Library (:7888) -------------------------------
 *
 * The library's cards carry a `text/uri-list`: the pose first, the source second when the
 * entry is a pair. That is all this needs — the server fetches the URLs (loopback only) and
 * puts the pose in the slot that makes the engine copy it: the document (image 1), with the
 * source as the `image 2` layer.
 */
function poseDropUrls(dt) {
  const raw = dt.getData("text/uri-list") || dt.getData("text/plain") || "";
  return raw.split(/\r?\n/).map(s => s.trim())
            .filter(u => /^http:\/\/(127\.0\.0\.1|localhost):\d+\//.test(u));
}

async function dropPose(ev) {
  const urls = poseDropUrls(ev.dataTransfer);
  if (!urls.length) return;
  ev.preventDefault();
  ev.stopPropagation();
  const [poseUrl, sourceUrl] = urls;
  const mode = sourceUrl ? "pair" : "image1";
  // the library's URL ends with its item id, which makes a useful document name
  const name = decodeURIComponent(poseUrl.replace(/\/+$/, "").split("/").pop() || "pose");
  status(mode === "pair" ? "Taking the pose and its source…" : "Taking the pose…");
  try {
    const r = await fetch("/api/pose/handoff", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ pose_url: poseUrl, source_url: sourceUrl || null,
                             name, mode }),
    });
    const data = await r.json().catch(() => ({}));
    if (!r.ok) throw new Error(data.detail || `HTTP ${r.status}`);
    status(`Pose opened in PhotoCraft: <span class="meta">${data.doc || "document"}</span>` +
           (data.layer_name ? ` + <span class="meta">${data.layer_name}</span>` : ""), "ok");
  } catch (e) {
    status(String(e.message || e), "bad");
  }
}

document.addEventListener("dragover", ev => {
  if (!(ev.dataTransfer?.types || []).includes("text/uri-list")) return;
  ev.preventDefault();
  ev.dataTransfer.dropEffect = "copy";
  document.body.classList.add("pose-drop");
});
document.addEventListener("dragleave", ev => {
  if (ev.relatedTarget) return;
  document.body.classList.remove("pose-drop");
});
document.addEventListener("drop", ev => {
  document.body.classList.remove("pose-drop");
  dropPose(ev);
});

initTheme();
initWindowChrome();
initPanels();
updateRefineRows();
setPreview(localStorage.getItem('cutout-preview') === '1', false);
// Generate is the first tab now, so that is where a fresh profile lands; a remembered choice wins.
switchTab(localStorage.getItem('cutout-tab') === 'cutout' ? 'cutout' : 'generate', false);
engineStatus();
initState();
autoPull();
draw();
