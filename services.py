"""Cutout Studio — phrase-driven selections and cut-outs on a local ComfyUI SAM3 stack.

Everything the browser and the MCP server do goes through this module, so there is one
implementation of "type a phrase, get the subject back".

The segmentation itself is core ComfyUI's SAM3_Detect node: checkpoint -> CLIP text ->
SAM3 -> mask image. Four nodes, no sampler, no VAE, so a phrase costs a couple of seconds
rather than a render. The checkpoint carries its own text encoder.

Traps this module exists to not repeat (both measured on this stack):
  * ComfyUI caches an unchanged graph, so an identical request re-executes nothing and the
    history entry carries no outputs at all. Every submission therefore carries a unique
    SaveImage prefix, which also makes this request's file identifiable afterwards.
  * A phrase that is not in the picture comes back as a degenerate mask - empty, a speck, or
    the whole frame - with HTTP 200. Coverage is checked, never the status code.
"""

from __future__ import annotations

import asyncio
import io
import json
import logging
import os
import re
import shutil
import subprocess
import time
import uuid
from typing import Any, Optional

import numpy as np
import requests
from PIL import Image, ImageFilter

log = logging.getLogger("cutout")

BASE = os.path.dirname(os.path.abspath(__file__))


def _p(path: str) -> str:
    return os.path.expanduser(path)


CFG_PATH = os.path.join(BASE, "config.json")
CFG_EXAMPLE = os.path.join(BASE, "config.example.json")
# config.json is git-ignored (it holds this machine's paths); a fresh clone starts from the example.
if not os.path.exists(CFG_PATH) and os.path.exists(CFG_EXAMPLE):
    import shutil
    shutil.copyfile(CFG_EXAMPLE, CFG_PATH)
    print(f"[config] created {CFG_PATH} from config.example.json — set your ComfyUI folder there or in the panel")
CFG = json.load(open(CFG_PATH, encoding="utf-8"))
DATA = os.path.join(BASE, CFG.get("data_dir", "data"))
SOURCES = os.path.join(DATA, "sources")
MASKS = os.path.join(DATA, "masks")
CUTOUTS = os.path.join(DATA, "cutouts")
EXPORT_DIR = _p(CFG.get("export_dir", "~/Pictures/cutout-studio"))
for _d in (SOURCES, MASKS, CUTOUTS):
    os.makedirs(_d, exist_ok=True)

NODE = CFG.get("node", "SAM3_Detect")
CKPT = CFG["ckpt"]

# Coverage band. SAM3 answers a phrase it cannot find with a degenerate mask rather than an
# error: measured on one portrait "hat" gave 0 px, "a blue balloon" 684 px, "spaceship" 98.5%
# of the frame, all as HTTP 200. The low end is refused; the high end is only a warning here,
# because a tight crop of one person legitimately covers more than 85% (measured: "person" on
# a torso crop = 91.3%) and refusing that would block a correct full-subject cut-out.
MIN_FRAC = 0.0005
MIN_PX = 400
BIG_FRAC = 0.85
FRAG_FRAC = 0.02
FRAG_ASPECT = 2.5


class BackendError(RuntimeError):
    """No usable ComfyUI stack, or the graph failed. The message is shown to the user."""


# ---------------------------------------------------------------- backend discovery

_probe: dict[str, Any] = {"backend": None, "at": 0.0, "reasons": [], "all": []}
_PROBE_TTL = 15.0


def _ckpt_names(backend: dict, timeout: float = 8.0) -> list[str]:
    r = requests.get(f"{backend['host']}/object_info/CheckpointLoaderSimple", timeout=timeout)
    r.raise_for_status()
    return r.json()["CheckpointLoaderSimple"]["input"]["required"]["ckpt_name"][0]


def probe(backend: dict, timeout: float = 4.0) -> tuple[bool, str]:
    """(usable, why not) for one stack — node first, then checkpoint."""
    try:
        r = requests.get(f"{backend['host']}/object_info/{NODE}", timeout=timeout)
        if r.status_code != 200:
            return False, f"not reachable on {backend['host']}"
        if not r.json().get(NODE):
            return False, f"{NODE} is not on this stack"
    except requests.exceptions.RequestException:
        return False, f"not reachable on {backend['host']}"
    try:
        if CKPT not in _ckpt_names(backend, timeout=timeout):
            return False, f"{CKPT} is not in its models/checkpoints"
    except Exception:
        return False, "cannot read its checkpoint list"
    return True, ""


def _start_backend(backend: dict) -> None:
    wrapper = _p(backend.get("wrapper") or "")
    if not wrapper or not os.path.isfile(wrapper):
        return
    log.info("starting %s via %s", backend["name"], wrapper)
    with open("/tmp/cutout-studio-comfy.log", "ab") as fh:
        subprocess.Popen(["bash", wrapper], stdout=fh, stderr=fh, start_new_session=True)


def resolve_backend(force: bool = False, start: bool = True, wait: float = 150.0) -> dict:
    """The first stack that can actually segment, starting one if none is up.

    A stack qualifies only when its SAM3 node AND the SAM3 checkpoint are both there, so this
    can never hand back a stack that fails mid-request with "node not found".
    """
    now = time.time()
    if not force and _probe["backend"] and now - _probe["at"] < _PROBE_TTL:
        return _probe["backend"]

    reasons = []
    for b in CFG["backends"]:
        ok, why = probe(b)
        if ok:
            _probe.update(backend=b, at=now, reasons=[])
            return b
        reasons.append(f"{b['name']}: {why}")

    if start:
        for b in CFG["backends"]:
            ck = os.path.join(_p(b["dir"]), "models", "checkpoints", CKPT)
            if os.path.isfile(ck):
                _start_backend(b)
                deadline = time.time() + wait
                while time.time() < deadline:
                    time.sleep(3)
                    ok, _why = probe(b, timeout=3)
                    if ok:
                        _probe.update(backend=b, at=time.time(), reasons=[])
                        return b
                reasons.append(f"{b['name']}: started, but did not come up within {int(wait)}s")
                break

    _probe.update(backend=None, at=now, reasons=reasons)
    raise BackendError(
        "No ComfyUI stack can segment by phrase. " + " | ".join(reasons)
        + f" — SAM3 needs {CKPT} in models/checkpoints."
    )


def status() -> dict:
    """What the stack can do right now, for the UI badge and the MCP status tool."""
    out = []
    for b in CFG["backends"]:
        try:
            ok, why = probe(b, timeout=2.5)
        except Exception as e:  # noqa: BLE001 - status must never raise
            ok, why = False, type(e).__name__
        out.append({"name": b["name"], "label": b["label"], "usable": ok, "reason": why})
    live = next((o for o in out if o["usable"]), None)
    return {
        "ok": bool(live),
        "backend": live["label"] if live else None,
        "node": NODE,
        "checkpoint": CKPT,
        "backends": out,
        "chips": CFG.get("chips", []),
        "export_dir": EXPORT_DIR,
    }


# ---------------------------------------------------------------- sources

def stage_source(path: Optional[str] = None, blob: Optional[bytes] = None,
                 name: Optional[str] = None) -> dict:
    """Copy an image into the app's own store and return its id.

    Stored as PNG so the pixels are exactly what the segmenter sees, whatever came in.
    """
    if blob is None:
        if not path:
            raise ValueError("give a path or bytes")
        with open(_p(path), "rb") as fh:
            blob = fh.read()
        name = name or os.path.basename(path)
    im = Image.open(io.BytesIO(blob))
    im.load()
    if im.mode not in ("RGB", "RGBA"):
        im = im.convert("RGBA" if "A" in im.mode else "RGB")
    sid = uuid.uuid4().hex[:12]
    dest = os.path.join(SOURCES, f"{sid}.png")
    im.save(dest)
    return {"id": sid, "width": im.width, "height": im.height,
            "path": dest, "name": name or f"{sid}.png",
            "url": f"/files/sources/{sid}.png"}


def source_path(sid: str) -> str:
    if os.path.sep in sid or sid.startswith("."):
        raise ValueError("bad id")
    p = os.path.join(SOURCES, f"{sid}.png")
    if not os.path.isfile(p):
        raise FileNotFoundError(sid)
    return p


def _stage_for_backend(backend: dict, sid: str) -> str:
    """The name LoadImage will read: the source copied into the stack's own input dir."""
    name = f"cutout_{sid}.png"
    dest = os.path.join(_p(backend["dir"]), "input", name)
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    shutil.copyfile(source_path(sid), dest)
    return name


# ---------------------------------------------------------------- the SAM3 graph

def build_workflow(image_name: str, phrase: str, threshold: float, refine: int,
                   prefix: str) -> dict:
    wf: dict = {}

    def add(nid: str, cls: str, **inputs: Any) -> None:
        wf[nid] = {"class_type": cls, "inputs": inputs}

    add("1", "CheckpointLoaderSimple", ckpt_name=CKPT)
    add("2", "LoadImage", image=image_name)
    # Slot 1 of the checkpoint is SAM3's own CLIP, which is what encodes the phrase.
    add("3", "CLIPTextEncode", text=phrase, clip=["1", 1])
    add("4", NODE, model=["1", 0], image=["2", 0], conditioning=["3", 0],
        threshold=float(threshold), refine_iterations=int(refine), individual_masks=False)
    # MaskToImage is what makes the result reachable on the output path; a MASK cannot be
    # saved by SaveImage on its own.
    add("5", "MaskToImage", mask=["4", 0])
    add("6", "SaveImage", images=["5", 0], filename_prefix=prefix)
    return wf


def _mask_from_outputs(backend: dict, outputs: dict, prefix: str,
                       prompt_id: str) -> Optional[str]:
    """Path of the mask this request wrote, re-reading history while the block fills in.

    Scoped to this request's own prefix, so it can never hand back another request's mask.
    """
    root = os.path.join(_p(backend["dir"]), "output")

    def find(outs: dict) -> Optional[str]:
        for out in (outs or {}).values():
            for img in (out or {}).get("images", []) or []:
                if str(img.get("filename", "")).startswith(prefix):
                    path = os.path.join(root, img.get("subfolder") or "", img["filename"])
                    if os.path.isfile(path):
                        return path
        return None

    got = find(outputs)
    for _ in range(4):
        if got:
            return got
        time.sleep(1.0)
        try:
            h = requests.get(f"{backend['host']}/history/{prompt_id}", timeout=10).json()
            if prompt_id in h:
                outputs = h[prompt_id].get("outputs", outputs)
        except (requests.exceptions.RequestException, ValueError):
            pass
        got = find(outputs)
    if got:
        return got
    # Last resort: the prefix is unique to this request, so the only possible match is ours.
    best, best_t = None, 0.0
    try:
        for name in os.listdir(root):
            if name.startswith(prefix):
                p = os.path.join(root, name)
                t = os.path.getmtime(p)
                if t > best_t:
                    best, best_t = p, t
    except OSError:
        return None
    return best


def _run_phrase(backend: dict, image_name: str, phrase: str, threshold: float,
                refine: int, timeout: float) -> np.ndarray:
    prefix = f"cutout_{uuid.uuid4().hex[:10]}"
    wf = build_workflow(image_name, phrase, threshold, refine, prefix)
    try:
        r = requests.post(f"{backend['host']}/prompt", json={"prompt": wf}, timeout=30)
    except requests.exceptions.RequestException as e:
        raise BackendError(f"ComfyUI stopped answering: {type(e).__name__}") from e
    if r.status_code != 200:
        detail = ""
        try:
            detail = json.dumps(r.json().get("node_errors", r.json()))[:300]
        except ValueError:
            detail = r.text[:300]
        raise BackendError(f"ComfyUI refused the graph: {detail}")

    prompt_id = r.json()["prompt_id"]
    deadline = time.time() + timeout
    outputs: dict = {}
    while time.time() < deadline:
        try:
            h = requests.get(f"{backend['host']}/history/{prompt_id}", timeout=15).json()
        except (requests.exceptions.RequestException, ValueError):
            time.sleep(0.5)
            continue
        if prompt_id in h:
            entry = h[prompt_id]
            st = entry.get("status", {})
            if st.get("status_str") == "error":
                msgs = [m[1].get("exception_message") for m in st.get("messages", [])
                        if m and m[0] == "execution_error" and isinstance(m[1], dict)]
                raise BackendError(f"SAM3 failed on \"{phrase}\": {(msgs or ['unknown error'])[0]}")
            outputs = entry.get("outputs") or {}
            if outputs:
                break
        time.sleep(0.4)

    path = _mask_from_outputs(backend, outputs, prefix, prompt_id)
    if not path:
        raise BackendError(
            f"SAM3 produced no mask for \"{phrase}\". That usually means the phrase named "
            f"nothing in this picture - try the object's name (\"hair\", \"top\") rather than "
            f"a description."
        )
    arr = np.array(Image.open(path).convert("L"))
    os.remove(path) if os.path.isfile(path) else None
    return arr > 127


def _slug(text: str) -> str:
    keep = [c.lower() if c.isalnum() else "-" for c in (text or "").strip()]
    out = re.sub(r"-+", "-", "".join(keep)).strip("-")
    return out[:40] or "object"


def _stats(mask: np.ndarray) -> dict:
    n = int(mask.sum())
    total = int(mask.size)
    frac = n / total if total else 0.0
    bbox = None
    if n:
        ys, xs = np.nonzero(mask)
        bbox = [int(xs.min()), int(ys.min()), int(xs.max() - xs.min() + 1),
                int(ys.max() - ys.min() + 1)]
    return {"pixels": n, "coverage": round(frac * 100, 2), "bbox": bbox}


_mask_store: dict[str, dict] = {}
_MASK_KEEP = 12


def segment(sid: str, phrase: str, threshold: float = 0.5, refine: int = 2,
            invert: bool = False, start: bool = True) -> dict:
    """Name what you want in words; get a mask back.

    Several phrases (comma separated) are run in turn and OR-ed, so "hair, face" is one
    selection. SAM3 is open-vocabulary and caps the prompt at 32 tokens, so each phrase
    should be the object's name rather than a description.
    """
    phrases = [p.strip() for p in re.split(r"[,\n]", phrase or "") if p.strip()]
    if not phrases:
        raise ValueError('Give a phrase naming what to select, e.g. "hair"')
    if len(phrase) > 400:
        raise ValueError("Keep the phrase short - SAM3 caps it at 32 tokens per phrase")

    backend = resolve_backend(start=start)
    image_name = _stage_for_backend(backend, sid)
    t0 = time.time()

    union: Optional[np.ndarray] = None
    per: list[dict] = []
    for ph in phrases:
        m = _run_phrase(backend, image_name, ph, threshold, refine,
                        timeout=240.0 if union is None else 120.0)
        st = _stats(m)
        per.append({"phrase": ph, **st})
        union = m if union is None else (union | m)

    assert union is not None
    raw = union
    if invert:
        union = ~union

    st = _stats(union)
    warnings: list[str] = []
    selected_px = st["pixels"]
    if selected_px < max(MIN_PX, int(MIN_FRAC * union.size)):
        if invert:
            raise ValueError(
                f"\"{phrase}\" selected {selected_px} px - essentially nothing - so there is "
                f"nothing to keep. Check the phrase, or turn invert off."
            )
        raise ValueError(
            f"\"{phrase}\" selected {selected_px} px - essentially nothing. SAM3 has no match "
            f"for that phrase in this picture; try the object's name."
        )
    if not invert and st["coverage"] > BIG_FRAC * 100:
        warnings.append(
            f"This selected {st['coverage']:.0f}% of the frame. Fine for a tight crop of one "
            f"subject, but check the overlay - a phrase that names nothing can return the "
            f"whole picture."
        )
    if len(per) > 1:
        for p in per:
            if p["pixels"] == 0:
                warnings.append(
                    f"\"{p['phrase']}\" matched nothing in this picture - the selection is the "
                    f"other phrases only."
                )
    if not invert and st["coverage"] < FRAG_FRAC * 100 and st["bbox"]:
        bw, bh = st["bbox"][2], st["bbox"][3]
        aspect = max(bw / max(bh, 1), bh / max(bw, 1))
        if aspect > FRAG_ASPECT:
            warnings.append(
                f"This selection is a sliver ({bw}x{bh} px, {st['coverage']:.1f}% of the "
                f"frame) rather than the whole object - widen the phrase or raise the "
                f"threshold."
            )

    mask_id = uuid.uuid4().hex[:12]
    png = os.path.join(MASKS, f"{mask_id}_{_slug(phrase)}.png")
    Image.fromarray((union.astype(np.uint8) * 255), mode="L").save(png)

    _mask_store[mask_id] = {"mask": union, "raw": raw, "sid": sid,
                            "phrase": phrase, "png": png, "invert": invert}
    while len(_mask_store) > _MASK_KEEP:
        _mask_store.pop(next(iter(_mask_store)))

    return {
        "mask_id": mask_id,
        "mask_url": f"/files/masks/{os.path.basename(png)}",
        "coverage": st["coverage"],
        "pixels": selected_px,
        "bbox": st["bbox"],
        "phrases": per,
        "inverted": invert,
        "warnings": warnings,
        "elapsed": round(time.time() - t0, 2),
        "backend": backend["label"],
    }


# ---------------------------------------------------------------- cut-out & outline

def mask_file(mask_id: str) -> Optional[str]:
    """The PNG this mask is stored as, or None if it is not on disk."""
    rec = _mask_store.get(mask_id)
    if rec and rec.get("png") and os.path.isfile(rec["png"]):
        return rec["png"]
    for name in os.listdir(MASKS):
        if name.startswith(mask_id):
            return os.path.join(MASKS, name)
    return None


def _mask_array(mask_id: str) -> np.ndarray:
    rec = _mask_store.get(mask_id)
    if rec:
        return rec["mask"]
    for name in os.listdir(MASKS):
        if name.startswith(mask_id):
            arr = np.array(Image.open(os.path.join(MASKS, name)).convert("L")) > 127
            _mask_store[mask_id] = {"mask": arr, "png": os.path.join(MASKS, name)}
            return arr
    raise KeyError(mask_id)


def _background(size: tuple[int, int], bg: Optional[dict]) -> Optional[Image.Image]:
    if not bg or bg.get("type") in (None, "transparent", "none"):
        return None
    w, h = size
    kind = bg.get("type")
    if kind == "solid":
        return Image.new("RGB", (w, h), bg.get("color") or "#ffffff")
    if kind == "gradient":
        c1, c2 = bg.get("color") or "#1b2b45", bg.get("color2") or "#f0f4ff"
        angle = float(bg.get("angle", 90))
        a, b = Image.new("RGB", (w, h), c1), Image.new("RGB", (w, h), c2)
        grad = Image.new("RGB", (w, h))
        # Linear sweep: a mask that slides across the frame at `angle`, then blended.
        theta = np.deg2rad(angle)
        yy, xx = np.mgrid[0:h, 0:w]
        t = (xx / max(w - 1, 1)) * np.cos(theta) + (yy / max(h - 1, 1)) * np.sin(theta)
        t = (t - t.min()) / max(t.max() - t.min(), 1e-6)
        arr = (np.asarray(a, dtype=np.float32) * (1 - t[..., None])
               + np.asarray(b, dtype=np.float32) * t[..., None])
        grad = Image.fromarray(arr.astype(np.uint8), mode="RGB")
        return grad
    return None


def make_cutout(sid: str, mask_id: str, feather: float = 1.0,
                bg: Optional[dict] = None, save: bool = True) -> dict:
    """The selection as pixels: source with the mask as alpha, optionally over a background."""
    src = Image.open(source_path(sid))
    if src.mode != "RGBA":
        src = src.convert("RGBA")
    mask = _mask_array(mask_id)
    if mask.shape != (src.height, src.width):
        mask_im = Image.fromarray((mask.astype(np.uint8) * 255), "L").resize(
            src.size, Image.NEAREST)
    else:
        mask_im = Image.fromarray((mask.astype(np.uint8) * 255), "L")
    if feather and feather > 0:
        mask_im = mask_im.filter(ImageFilter.GaussianBlur(float(feather)))
    src.putalpha(mask_im)

    out = src
    back = _background(src.size, bg)
    if back is not None:
        flat = back.convert("RGBA")
        flat.alpha_composite(src)
        out = flat

    path = None
    if save:
        path = os.path.join(CUTOUTS, f"{sid}_{mask_id}.png")
        out.save(path)
    buf = io.BytesIO()
    out.save(buf, format="PNG")
    return {"path": path, "png": buf.getvalue(), "width": out.width, "height": out.height,
            "url": f"/files/cutouts/{os.path.basename(path)}" if path else None,
            "solid": back is not None}


def _rdp(points: list[list[float]], eps: float) -> list[list[float]]:
    """Douglas-Peucker. Keeps the outline recognisable while dropping pixel staircases."""
    if len(points) < 3:
        return points
    start, end = np.array(points[0], dtype=float), np.array(points[-1], dtype=float)
    line = end - start
    norm = float(np.hypot(*line))
    if norm == 0:
        dists = [float(np.hypot(*(np.array(p) - start))) for p in points[1:-1]]
    else:
        # z of the 2-D cross product, spelled out: np.cross() refuses 2-vectors in numpy 2.
        dists = [abs(float(line[0] * (p[1] - start[1]) - line[1] * (p[0] - start[0]))) / norm
                 for p in points[1:-1]]
    idx = int(np.argmax(dists)) if dists else 0
    if not dists or dists[idx] <= eps:
        return [points[0], points[-1]]
    left = _rdp(points[:idx + 2], eps)
    right = _rdp(points[idx + 1:], eps)
    return left[:-1] + right


def outline(mask_id: str, simplify: float = 1.0) -> dict:
    """The mask as polygons, in image pixels - what a PhotoCraft lasso wants.

    potrace does the vectorising; its GeoJSON y axis is flipped relative to image space, so
    every point is mirrored back before it leaves here.
    """
    mask = _mask_array(mask_id)
    h, w = mask.shape
    pbm = os.path.join(MASKS, f"{mask_id}.pbm")
    packed = np.packbits(mask.astype(np.uint8), axis=1)
    with open(pbm, "wb") as fh:
        fh.write(f"P4\n{w} {h}\n".encode())
        fh.write(packed.tobytes())
    geo = os.path.join(MASKS, f"{mask_id}.geojson")
    try:
        subprocess.run(["potrace", "-b", "geojson", "-a", "1.0", "-O", "0.2",
                        pbm, "-o", geo], check=True, capture_output=True, timeout=120)
        data = json.load(open(geo, encoding="utf-8"))
    except (FileNotFoundError, subprocess.CalledProcessError) as e:
        raise RuntimeError(f"potrace could not vectorise the mask: {e}") from e
    finally:
        for p in (pbm, geo):
            try:
                os.remove(p)
            except OSError:
                pass

    rings: list[dict] = []
    for feat in data.get("features", []):
        geom = feat.get("geometry") or {}
        polys = geom.get("coordinates") or []
        if geom.get("type") == "Polygon":
            polys = [polys]
        for poly in polys:
            for i, ring in enumerate(poly):
                pts = [[round(float(x), 2), round(float(h) - float(y), 2)] for x, y in ring]
                if simplify and simplify > 0 and len(pts) > 4:
                    pts = _rdp(pts, float(simplify))
                if len(pts) >= 3:
                    rings.append({"points": pts, "hole": i > 0, "area": _ring_area(pts)})
    rings.sort(key=lambda r: r["area"], reverse=True)
    return {"rings": rings, "width": w, "height": h,
            "points": max(len(r["points"]) for r in rings) if rings else 0,
            "count": len(rings)}


def _detail_map(sid: str, block: int = 8) -> np.ndarray:
    """Local image detail per pixel (std of luminance in a `block`×`block` window).

    Hair is the high-detail part of a portrait: individual strands make the local variance jump
    while skin, cloth and a plain background stay flat. That is what decides where PhotoCraft's
    Refine Edge brush should be painted.
    """
    with Image.open(source_path(sid)) as im:
        g = np.asarray(im.convert("L"), dtype=np.float32)
    h, w = g.shape
    ph, pw = (-h) % block, (-w) % block
    if ph or pw:
        g = np.pad(g, ((0, ph), (0, pw)), mode="edge")
    H, W = g.shape
    bh, bw = H // block, W // block
    tiles = g.reshape(bh, block, bw, block).transpose(0, 2, 1, 3).reshape(bh, bw, block * block)
    std = tiles.std(axis=2)
    big = np.repeat(np.repeat(std, block, axis=0), block, axis=1)[:h, :w]
    return big


def _hair_refine_strokes(sid: str, mask_id: str, size: int = 26,
                         detail_min: float = 9.0) -> list[dict]:
    """Refine Edge brush strokes along the parts of the mask boundary that look like hair.

    PhotoCraft's Refine Edge brush is the actual hair tool — painting a stroke there makes the
    app pull strands out of the image instead of following a polygon. The strokes are generated
    from the mask: its boundary, kept only where the luminance detail says "strands", thinned to
    one point per ~`step` px and split into separate strokes so the whole head is not one path.
    """
    mask = _mask_array(mask_id)
    detail = _detail_map(sid)
    # Boundary = mask pixels with a background neighbour (4-connectivity), no scipy needed.
    m = mask.astype(bool)
    inner = m.copy()
    inner[1:, :] &= m[:-1, :]
    inner[:-1, :] &= m[1:, :]
    inner[:, 1:] &= m[:, :-1]
    inner[:, :-1] &= m[:, 1:]
    edge = m & ~inner
    ys, xs = np.nonzero(edge)
    if not len(xs):
        return []
    keep = detail[ys, xs] >= detail_min
    ys, xs = ys[keep], xs[keep]
    if not len(xs):
        return []
    # One point per grid cell keeps the payload small and the strokes smooth; cells are then
    # grouped into strokes by connectivity in that grid.
    step = max(6, size // 2)
    cells: dict[tuple[int, int], list[int]] = {}
    for y, x in zip(ys, xs):
        cells.setdefault((int(y) // step, int(x) // step), []).append(int(x))
        cells[(int(y) // step, int(x) // step)].append(int(y))
    pts = {}
    for (cy, cx), flat in cells.items():
        arr = np.array(flat, dtype=np.int32).reshape(-1, 2)
        pts[(cy, cx)] = [int(arr[:, 0].mean()), int(arr[:, 1].mean())]
    seen: set[tuple[int, int]] = set()
    strokes = []
    for start in pts:
        if start in seen:
            continue
        # flood fill over 8-connected cells
        stack, group = [start], []
        seen.add(start)
        while stack:
            c = stack.pop()
            group.append(c)
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    nb = (c[0] + dy, c[1] + dx)
                    if nb in pts and nb not in seen:
                        seen.add(nb)
                        stack.append(nb)
        if len(group) < 2:
            continue
        # order the cells as a path so the stroke is drawn along the edge, not zig-zag
        group.sort()
        path = [group[0]]
        left = set(group[1:])
        while left:
            last = path[-1]
            nxt = min(left, key=lambda c: (c[0] - last[0]) ** 2 + (c[1] - last[1]) ** 2)
            path.append(nxt)
            left.discard(nxt)
        strokes.append({"points": [pts[c] for c in path[:80]], "size": size, "mode": "add"})
    return strokes[:24]


def _ring_area(pts: list[list[float]]) -> float:
    s = 0.0
    for i in range(len(pts) - 1):
        s += pts[i][0] * pts[i + 1][1] - pts[i + 1][0] * pts[i][1]
    return abs(s) / 2.0


def export_files(sid: str, mask_id: str, phrase: str = "", feather: float = 1.0,
                 bg: Optional[dict] = None, name: Optional[str] = None,
                 with_outline: bool = True) -> dict:
    """Write the cut-out, its mask and its outline into the PhotoCraft-visible folder."""
    stamp = time.strftime("%Y%m%d-%H%M%S")
    slug = _slug(name or phrase or "selection")
    folder = os.path.join(EXPORT_DIR, time.strftime("%Y-%m-%d"))
    os.makedirs(folder, exist_ok=True)

    made = make_cutout(sid, mask_id, feather=feather, bg=bg)
    cut_path = os.path.join(folder, f"{slug}-{stamp}-cutout.png")
    with open(cut_path, "wb") as fh:
        fh.write(made["png"])

    mask_src = os.path.join(MASKS, f"{mask_id}_{_slug(phrase)}.png")
    if not os.path.isfile(mask_src):
        for n in os.listdir(MASKS):
            if n.startswith(mask_id):
                mask_src = os.path.join(MASKS, n)
                break
    mask_path = os.path.join(folder, f"{slug}-{stamp}-mask.png")
    Image.open(mask_src).save(mask_path)

    # The original, untouched, beside the cut-out: this is what PhotoCraft opens, so the
    # selection can be used for further editing instead of being baked into an alpha PNG.
    # No timestamp on this one: sending the same image twice must reuse the document
    # PhotoCraft already has open instead of piling up a second, identical one.
    src_path = os.path.join(EXPORT_DIR, f"{slug}-{sid}-source.png")
    Image.open(source_path(sid)).save(src_path)

    info: dict = {"cutout": cut_path, "mask": mask_path, "source": src_path, "folder": folder}
    if with_outline:
        try:
            rings = outline(mask_id)
            sel = os.path.join(folder, f"{slug}-{stamp}-selection.json")
            json.dump({"image": cut_path, "mask": mask_path, "phrase": phrase,
                       "rings": rings["rings"]}, open(sel, "w", encoding="utf-8"), indent=1)
            info["selection"] = sel
            info["rings"] = len(rings["rings"])
        except Exception as e:  # noqa: BLE001 - the outline is a bonus, not the export
            info["outline_error"] = str(e)
    return info


# ---------------------------------------------------------------- PhotoCraft hand-off

PHOTOCRAFT_CTRL = os.path.expanduser("~/.local/bin/photocraft-control")
PHOTOCRAFT_PORT = int(os.environ.get("PHOTOCRAFT_CONTROL_PORT", "7916"))
# PhotoCraft's flatpak sandbox only reaches granted roots. ~/Pictures is granted, so every
# hand-off is written there and the app is launched with that folder declared as its automation
# root: without the root, `doc_open` in bridge mode is refused ("automation filesystem access is
# not granted") and an image can never be loaded into a session that is already running.
PHOTOCRAFT_ROOT = os.path.expanduser(os.environ.get("PHOTOCRAFT_ROOT", "~/Pictures"))
PHOTOCRAFT_LOG = os.path.expanduser("~/.config/photocraft-mcp/gui.log")
# Where a document pulled out of PhotoCraft is exported before it is segmented. Inside the
# sandbox-granted root, because PhotoCraft's own `doc_export` can only write there.
PHOTOCRAFT_PULL_DIR = os.path.join(PHOTOCRAFT_ROOT, "cutout-studio", "_from-photocraft")
# Quick export stages the composite here first (inside the automation root, the only place
# `doc_export` may write); the finished file is then copied beside the picture it came from, which
# is usually outside the sandbox entirely.
PHOTOCRAFT_QUICK_DIR = os.path.join(PHOTOCRAFT_ROOT, "cutout-studio", "_quick-export")
# Where a delivered render came from: engine render name -> the file in ComfyUI's output tree.
AI_ORIGINS = os.path.join(DATA, "ai-origins.json")
# Which Post-processing adjustments the user left ticked: applied to every picture the engine
# delivers afterwards (the checkboxes are a standing choice, not just a switch for what is open).
POST_PREFS = os.path.join(DATA, "postprocess.json")
_QUICK_EXTS = (".png", ".jpg", ".jpeg", ".webp")

# Where a real-alpha matte is written before it is carried into PhotoCraft as a layer. Also inside the
# automation root: PhotoCraft can only open documents it is allowed to read.
PHOTOCRAFT_MATTE_DIR = os.path.join(PHOTOCRAFT_ROOT, "cutout-studio", "_matte")
_launched_by_us = False
# The PhotoCraft document the staged source came from. While it is set the hand-off is pinned:
# the result goes back into *that* document and no copy of the image is ever opened there.
_pc_pin: Optional[dict] = None


def matte_cutout(sid: str, mask_id: str) -> dict:
    """A real per-pixel alpha for the phrase, instead of a feathered polygon.

    PhotoCraft's Refine Edge can only feather, contrast and shift the polygon it is given, so its
    edge is a translucent band: measured on a portrait, 39.8k semi-transparent pixels but only 0.75 of
    the hair strands the truth has, and the band carries the old background's colour. A matte model
    (BiRefNet-portrait, run here) plus a solid core from the SAM3 mask and colour decontamination
    gives strand-accurate alpha (soft px 39k, strand retention 1.0 by construction).
    """
    import matte as engine   # the ONNX session is heavy; import only when asked for a matte

    src = Image.open(source_path(sid)).convert("RGB")
    mask = _mask_array(mask_id)
    res = engine.compose(src, mask)
    alpha = res["alpha"]
    clean = engine.decontaminate(src, alpha)
    os.makedirs(PHOTOCRAFT_MATTE_DIR, exist_ok=True)
    out = os.path.join(PHOTOCRAFT_MATTE_DIR, f"matte-{sid}-{mask_id}.png")
    rgba = clean.convert("RGBA")
    rgba.putalpha(Image.fromarray((alpha * 255).astype(np.uint8)))
    rgba.save(out)
    res.pop("alpha", None)
    res.update({"path": out,
                "soft_px": int(((alpha > 0.03) & (alpha < 0.97)).sum()),
                "opaque_px": int((alpha >= 0.97).sum())})
    return res


def _wayland_display() -> str:
    """This session's Wayland socket name.

    Guessing "wayland-0" is wrong on a machine whose compositor took wayland-1, and the
    bridge then dies with NoCompositor before it can answer a single tool call.
    """
    name = os.environ.get("WAYLAND_DISPLAY")
    if name:
        return name
    uid = os.getuid()
    try:
        socks = sorted(n for n in os.listdir(f"/run/user/{uid}") if n.startswith("wayland-"))
        socks = [s for s in socks if not s.endswith(".lock")]
        if len(socks) == 1:
            return socks[0]
    except OSError:
        pass
    return "wayland-0"


def _pc_env(with_roots: bool = False) -> dict:
    """A curated session env: no LD_LIBRARY_PATH, because linuxbrew's libngtcp2 shadows the one
    /usr/lib/libcurl.so.4 needs and flatpak dies with an undefined symbol before it starts."""
    env = {
        "PATH": os.pathsep.join([
            os.path.expanduser("~/.local/bin"), "/usr/local/bin", "/usr/bin", "/bin",
        ]),
        "HOME": os.path.expanduser("~"),
        "XDG_RUNTIME_DIR": f"/run/user/{os.getuid()}",
        "WAYLAND_DISPLAY": _wayland_display(),
        "DISPLAY": os.environ.get("DISPLAY", ":0"),
        "XDG_SESSION_TYPE": os.environ.get("XDG_SESSION_TYPE", "wayland"),
        "XDG_CURRENT_DESKTOP": os.environ.get("XDG_CURRENT_DESKTOP", "KDE"),
        "DBUS_SESSION_BUS_ADDRESS": os.environ.get(
            "DBUS_SESSION_BUS_ADDRESS", f"unix:path=/run/user/{os.getuid()}/bus"),
    }
    if with_roots:
        env["PHOTOCRAFT_AUTOMATION_READ_ROOT"] = PHOTOCRAFT_ROOT
        env["PHOTOCRAFT_AUTOMATION_WRITE_ROOT"] = PHOTOCRAFT_ROOT
    return env


def _port_up(port: int = PHOTOCRAFT_PORT) -> bool:
    try:
        out = subprocess.run(["ss", "-ltn"], capture_output=True, text=True, timeout=10).stdout
        return f"127.0.0.1:{port}" in out
    except (OSError, subprocess.SubprocessError):
        return False


def _photocraft_running() -> bool:
    """Is a PhotoCraft process up at all, reachable or not?"""
    try:
        out = subprocess.run(["pgrep", "-f", "[p]hotocraft"], capture_output=True, text=True,
                             timeout=8).stdout
        return bool(out.strip())
    except (OSError, subprocess.SubprocessError):
        return False


def _control_running() -> bool:
    """A PhotoCraft started in `--control` mode (i.e. drivable), whether or not its port answers.

    PhotoCraft has no single-instance lock: the user's own window and an automation session are
    separate processes and co-exist (verified — a plain instance from the app menu kept running
    while the control-mode one opened a document). So the presence of *a* PhotoCraft must not stop
    the panel from having a drivable one; only a control-mode instance whose port has died is a
    real conflict, because it would be replaced under the user's feet.
    """
    try:
        out = subprocess.run(["pgrep", "-af", "photocraft"], capture_output=True, text=True,
                             timeout=8).stdout
    except (OSError, subprocess.SubprocessError):
        return False
    for line in out.splitlines():
        if "photocraft" in line and "--control" in line and "pgrep" not in line:
            return True
    return False


def _token() -> str:
    path = os.path.expanduser("~/.config/photocraft-mcp/control-token")
    try:
        return open(path, encoding="utf-8").read().strip()
    except OSError as e:
        raise RuntimeError(
            f"No PhotoCraft control token at {path} — run `photocraft-control status` once to "
            f"create it ({e}).") from e


def photocraft_status() -> dict:
    return {
        "control_up": _port_up(),
        "port": PHOTOCRAFT_PORT,
        "root": PHOTOCRAFT_ROOT,
        "helper": os.path.isfile(PHOTOCRAFT_CTRL),
        "launched_by_app": _launched_by_us,
        "pinned": bool(_pc_pin),
    }


def photocraft_pin() -> Optional[dict]:
    """The PhotoCraft document the current source was pulled from, if any."""
    return dict(_pc_pin) if _pc_pin else None


def reset_state() -> dict:
    """Forget everything the app was holding: staged sources, masks, and the PhotoCraft pin.

    The UI's Clear button calls this. It does not touch the files already exported under
    ~/Pictures/cutout-studio, and it never touches the user's PhotoCraft document — a cleared app
    simply has nothing to act on until it pulls a document again.
    """
    global _pc_pin
    _pc_pin = None
    _mask_store.clear()
    return {"ok": True, "mask_store": len(_mask_store), "pin": None}


RAISE_JS = """
const wins = workspace.windowList ? workspace.windowList() : workspace.clientList();
for (const w of wins) {
  const t = (String(w.resourceClass || "") + " " + String(w.caption || "")).toLowerCase();
  if (t.indexOf("%s") >= 0) { workspace.activeWindow = w; if (w.activate) w.activate(0); }
}
"""


def _raise_window(match: str = "photocraft") -> bool:
    """Bring PhotoCraft's window to the front.

    Its control bridge only answers while the app is drawing frames: an occluded window makes
    every engine call hang until the 55 s internal timeout, so a freshly launched instance is
    useless until something raises it. kdotool refuses this KDE build, so this is a KWin script
    over DBus. Best-effort - a failure here must not sink the hand-off.
    """
    path = "/tmp/cutout-studio-raise.js"
    try:
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(RAISE_JS % match)
        subprocess.run(["qdbus6", "org.kde.KWin", "/Scripting",
                        "org.kde.kwin.Scripting.loadScript", path],
                       capture_output=True, timeout=15)
        subprocess.run(["qdbus6", "org.kde.KWin", "/Scripting",
                        "org.kde.kwin.Scripting.start"], capture_output=True, timeout=15)
        time.sleep(1.0)
        return True
    except (OSError, subprocess.SubprocessError):
        return False


def ensure_control(wait: float = 90.0) -> tuple[bool, str]:
    """A control-mode PhotoCraft that can open files from PHOTOCRAFT_ROOT.

    Launched with PHOTOCRAFT_AUTOMATION_READ_ROOT/WRITE_ROOT so `doc_open` works later; a session
    started without those roots can be reached but cannot load anything new, which is the one case
    the caller has to report instead of retry.
    """
    global _launched_by_us
    if _port_up():
        return True, "already running"
    if _control_running():
        # A control-mode instance is already up but its port is dead: starting another one would
        # silently take over from the window the user is looking at, so stop and say so.
        return False, (
            "PhotoCraft's automation session is running but no longer answering its control port. "
            "Quit that PhotoCraft window (Ctrl+Q) and press again — the app reopens it with the "
            "control server and its automation root. Save first if it holds unsaved work.")
    tok = _token()
    os.makedirs(os.path.dirname(PHOTOCRAFT_LOG), exist_ok=True)
    cmd = ["flatpak", "run", "--share=network", "--command=photocraft",
           "ai.storyteller.photocraft", "--control", str(PHOTOCRAFT_PORT),
           "--control-token", tok]
    log.info("starting PhotoCraft in control mode on %s", PHOTOCRAFT_PORT)
    with open(PHOTOCRAFT_LOG, "ab") as fh:
        subprocess.Popen(cmd, env=_pc_env(with_roots=True), stdin=subprocess.DEVNULL,
                         stdout=fh, stderr=fh, start_new_session=True)
    deadline = time.time() + wait
    while time.time() < deadline:
        time.sleep(2)
        if _port_up():
            _launched_by_us = True
            _raise_window()
            return True, "started"
    return False, (f"PhotoCraft did not open its control port within {int(wait)}s — see "
                   f"{PHOTOCRAFT_LOG}")


def _text(res: Any) -> str:
    """The whole text of a tool result — never truncated.

    It used to be cut at 600 chars for error messages, which silently broke every structured
    read: doc_inspect's JSON is >2 KB, so json.loads failed and the document reported as
    nameless while the selection had in fact landed. Callers slice for display instead.
    """
    parts = []
    for c in getattr(res, "content", []) or []:
        t = getattr(c, "text", None)
        if t:
            parts.append(str(t))
    return " ".join(parts)


def _plain_error(e: BaseException) -> str:
    """A dead bridge arrives as `ExceptionGroup: unhandled errors in a TaskGroup (1 sub-exception)`.

    That string says nothing to the person looking at the panel: the group is the MCP stdio client
    wrapping whatever went wrong underneath (usually the app's process going away mid-call). Name
    the real error, and say what the situation is.
    """
    leaf = e
    while isinstance(leaf, BaseExceptionGroup) and leaf.exceptions:
        leaf = leaf.exceptions[0]
    detail = f"{type(leaf).__name__}: {leaf}"[:200]
    if isinstance(e, BaseExceptionGroup):
        return ("PhotoCraft's control server stopped answering (its window was closed or the app "
                f"restarted mid-command) — first error {detail}")
    return f"{type(e).__name__}: {e}"[:200]


async def _pc_bridge(work):
    """Run `work(session)` against the bridge for the app's control port."""
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    if PHOTOCRAFT_PORT == 7916 and os.path.isfile(PHOTOCRAFT_CTRL):
        params = StdioServerParameters(command=PHOTOCRAFT_CTRL, args=["mcp"], env=_pc_env())
    else:
        params = StdioServerParameters(
            command="flatpak",
            args=["run", "--share=network", "--command=photocraft-cli",
                  "ai.storyteller.photocraft", "mcp", "--bridge",
                  f"127.0.0.1:{PHOTOCRAFT_PORT}", "--control-token", _token()],
            env=_pc_env())
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            return await work(session)


async def _pc_run(session, cmd: str, params: dict, timeout: float = 120.0):
    # PhotoCraft has no per-command MCP tools: registry commands go through command_run with the
    # command's id ("select.lasso") and its params doc. Calling "select.lasso" as a tool name
    # answers "tool not found".
    return await asyncio.wait_for(
        session.call_tool("command_run", {"id": cmd, "params": params}), timeout=timeout)


async def _pc_doc(session) -> dict:
    """`doc_inspect` as a dict, or a plain error when the app is not really there.

    The CLI bridge answers with a sentence rather than JSON when it cannot reach the app
    ("bridge: `engine.execute` failed: cannot connect to 127.0.0.1:7916 …"). Handing that back as
    a document is what let a failed delivery report "opened in PhotoCraft as null (null×null)" —
    every caller then worked on a dict with no name and no size. Refuse it here instead.
    """
    res = await asyncio.wait_for(session.call_tool("doc_inspect", {}), timeout=120)
    txt = _text(res)
    try:
        doc = json.loads(txt)
    except ValueError:
        if "no document open" in txt.lower():
            # The app answered, there is simply nothing open. Callers already handle an empty
            # document (the panel's "no document" line, the pull's "open the picture first").
            return {}
        raise RuntimeError(f"PhotoCraft is not answering its control port — {txt.strip()[:200]}")
    if not isinstance(doc, dict):
        raise RuntimeError(f"PhotoCraft answered with something that is not a document: {txt[:160]}")
    return doc


async def _pc_load_and_select(session, image_path: str, rings: Optional[list[dict]],
                              prefer_open_doc: bool = True,
                              pin: Optional[dict] = None, apply: str = "selection",
                              phrase: str = "", refine: str = "off",
                              strokes: Optional[list[dict]] = None,
                              refine_radius: float = 6.0) -> dict:
    """Select the outline in PhotoCraft, opening the image only if it is not already there.

    Matching is by name first, then by exact pixel size: the common case is "I already have this
    picture open in PhotoCraft and just want the subject selected in it", and opening a second
    document of the same picture would throw that away. A size match on a different document is
    reported in the answer, so a wrong guess is visible rather than silent (and one undo away).

    With `pin` (the document the source was pulled out of) nothing is ever opened: the selection
    is installed in that document, and a document switch since the pull is an error rather than
    an edit landing somewhere the user did not ask for.

    `apply` says what the user should SEE in PhotoCraft afterwards:
      "selection" the lasso alone - marching ants, pixels untouched (default);
      "mask"      the lasso becomes a layer mask, so the background is hidden non-destructively;
      "layer"     the lasso is copied onto its own layer, leaving the original intact below.
    """
    rings = rings or []
    doc = await _pc_doc(session)
    want = os.path.basename(image_path)
    if pin:
        if doc.get("name") != pin.get("doc"):
            raise RuntimeError(
                f"PhotoCraft is showing {doc.get('name')!r}, not {pin.get('doc')!r} — the document "
                "changed since the image was pulled out of it. Press From PhotoCraft to pull the "
                "one you are looking at now.")
        loaded = True
        note = "selection installed in the document the image was pulled from"
    else:
        loaded = doc.get("name") == want
        note = "already open"
        if not loaded and prefer_open_doc and doc.get("name"):
            try:
                with Image.open(image_path) as im:
                    same_size = [im.width, im.height] == [doc.get("width"), doc.get("height")]
            except OSError:
                same_size = False
            if same_size:
                loaded = True
                note = f"selected in {doc.get('name')} (open already, same {doc.get('width')}×{doc.get('height')} size)"
    if not loaded:
        res = await asyncio.wait_for(
            session.call_tool("doc_open", {"path": image_path}), timeout=180)
        msg = _text(res)[:400]
        if "automation filesystem access" in msg or "not granted" in msg:
            raise RuntimeError(
                "PhotoCraft is running without an automation root, so it cannot open new files "
                f"({msg[:160]}). Quit PhotoCraft and press the button again — the app restarts it "
                "with " + PHOTOCRAFT_ROOT + " declared as its root.")
        if "error" in msg.lower() and "name" not in msg:
            raise RuntimeError(f"PhotoCraft refused to open the image: {msg[:200]}")
        try:
            opened = json.loads(msg)
        except ValueError:
            opened = {}
        if opened.get("name") != want:
            raise RuntimeError(f"PhotoCraft opened something unexpected: {msg[:200]}")
        loaded = True
        note = "opened"

    outer = [r for r in rings if not r["hole"]]
    holes = [r for r in rings if r["hole"]]
    if not outer:
        raise RuntimeError("The mask has no outer ring to send.")
    steps = 0
    res = await _pc_run(session, "select.lasso", {"points": outer[0]["points"],
                                                 "mode": "replace"})
    steps += 1
    for extra in outer[1:]:
        await _pc_run(session, "select.lasso", {"points": extra["points"], "mode": "add"})
        steps += 1
    for hole in holes:
        await _pc_run(session, "select.lasso", {"points": hole["points"], "mode": "subtract"})
        steps += 1

    # Where PhotoCraft's own idea of the selection starts, measured while it is still live: the
    # Refine Edge call below consumes the selection, and a selection-less measure reads the whole
    # canvas (1,048,576 px on a 1024² document) — a 65% "difference" that means nothing.
    doc = await _pc_doc(session)
    area = None
    try:
        res = await _pc_run(session, "image.analysis.recordMeasurements",
                            {"source": "selection"}, timeout=60)
        rows = json.loads(_text(res)).get("rows", [])
        if rows:
            area = max(float(r["values"].get("area", 0)) for r in rows)
    except Exception:  # noqa: BLE001 - the measurement is a check, not the result
        pass

    # A polygon edge cuts every wisp off at the outline. PhotoCraft's own Refine Edge — the same
    # Select-and-Mask dialog — can pull real strands out of the image, and the refined result can
    # be handed straight to a layer mask. The Refine Edge brush is painted along the boundary
    # only where the image has hair-like detail (see _hair_refine_strokes).
    refined = ""
    refined_layer: Optional[int] = None
    if refine in ("smart", "hair"):
        params: dict = {"radius": float(refine_radius), "smartRadius": bool(refine_radius > 0),
                        "smooth": 6, "feather": 0.3, "contrast": 10, "shiftEdge": 8,
                        "decontaminate": True, "amount": 60,
                        # Accepted values: selection | layerMask | newLayer | newLayerWithMask |
                        # newDocument | newDocumentWithMask. Measured: on a flat layer every value
                        # ends in a copy of the layer ("Background copy") with the original hidden,
                        # so refineEdge is never truly in-place — the reply says which layer holds
                        # the refined mask and the caller reports that layer, not the old one.
                        "output": "layerMask" if apply == "mask" else
                                  ("newLayerWithMask" if apply == "layer" else "selection")}
        if strokes:
            params["refineStrokes"] = strokes
        res = await _pc_run(session, "select.refineEdge", params, timeout=240)
        try:
            refined_layer = json.loads(_text(res)).get("layer")
        except ValueError:
            refined_layer = None
        refined = (f"Refine Edge {refine_radius:.0f} px smart radius"
                   + (f" + {len(strokes)} hair stroke(s)" if strokes else ""))


    # What the user sees in PhotoCraft. The area above is measured with the selection still
    # live, because a mask or a copied layer would have made it unreadable.
    applied, what = "selection", "a live selection"
    if apply in ("mask", "layer"):
        active = refined_layer or doc.get("activeLayer")
        if refined_layer:
            # refineEdge writes onto its own copy, so read the document again to name the right
            # layer rather than reporting the one it was applied from.
            doc = await _pc_doc(session)
        layer = next((l for l in (doc.get("layers") or []) if l.get("id") == active), {})
        if apply == "mask":
            if refined:
                # Refine Edge already wrote the mask (output="layerMask"); adding another would
                # stack two of them.
                applied = "mask"
                what = f"a refined layer mask on {layer.get('name') or 'the active layer'}"
            else:
                if layer.get("hasMask"):
                    raise RuntimeError(
                        f"The active layer in PhotoCraft ({layer.get('name') or active}) already has a "
                        "layer mask, so the selection cannot become one. Delete that mask first "
                        "(Layer ▸ Layer Mask ▸ Delete), or use the new-layer option.")
                await _pc_run(session, "layer.layerMask.revealSelection", {"layer": active})
                applied = "mask"
                what = f"a layer mask on {layer.get('name') or 'the active layer'}"
        else:
            if not refined:
                await _pc_run(session, "layer.new.layerViaCopy", {})
            name = f"Cutout · {phrase}"[:60] if phrase else "Cutout"
            try:
                await _pc_run(session, "layer.renameLayer", {"name": name})
            except Exception:  # noqa: BLE001 - the copy is the result, the name is cosmetic
                pass
            applied = "layer"
            what = f"a new layer ({name})"
        # A clean picture: the mask/layer is the result, ants over it are just noise.
        await _pc_run(session, "select.deselect", {})
        doc = await _pc_doc(session)

    return {"doc": doc.get("name"), "size": [doc.get("width"), doc.get("height")],
            "has_selection": bool(doc.get("hasSelection")), "steps": steps,
            "selection_area": area, "how": note, "applied": applied, "shows": what,
            "refined": refined}


def photocraft_doc(timeout: float = 8.0) -> dict:
    """What PhotoCraft currently has open, for the UI's "this is what I will act on" line.

    Short leash on purpose: an occluded PhotoCraft answers nothing for 55 s, and this line must
    not freeze the panel for that long — `From PhotoCraft` raises the window and retries.
    """
    st = photocraft_status()
    if not st["control_up"]:
        st["running"] = _photocraft_running()
        st["doc"] = None
        return st

    async def _read(session):
        return await asyncio.wait_for(_pc_doc(session), timeout=timeout)

    try:
        doc = asyncio.run(_pc_bridge(_read))
    except Exception as e:  # noqa: BLE001 - the line is informational
        st["doc"] = None
        # An occluded PhotoCraft answers nothing at all; a dead or restarted one arrives wrapped in
        # the MCP client's ExceptionGroup, whose own text is meaningless to the person reading it.
        st["error"] = (_plain_error(e) if isinstance(e, BaseExceptionGroup)
                       else f"{type(e).__name__}: {str(e)[:160]}")
        return st
    if not doc or not doc.get("name"):
        # no document open at all: doc_inspect answers with an empty document rather than failing
        st["doc"] = None
        st["empty"] = True
        return st
    st["doc"] = {k: doc.get(k) for k in ("name", "width", "height", "hasSelection")} if doc else None
    st["pin"] = photocraft_pin()
    return st


def photocraft_pull() -> dict:
    """Take the picture the user has open in PhotoCraft and make it this app's source.

    PhotoCraft is the front door of the workflow: the user opens the picture there and never
    has to load it here as well. The document is read as the app itself draws it (`doc_export`,
    the flattened composite, so unsaved edits and adjustment layers are included) rather than
    from its file on disk, which would be stale the moment anything was edited. The document's
    identity is then pinned, so the result goes back into the same document instead of into a
    copy opened beside it.
    """
    global _pc_pin
    # Pull only pulls: starting PhotoCraft is not this app's business here, and launching it on
    # a page load just to find it empty would be a popup for nothing.
    if not _port_up():
        raise RuntimeError(
            "PhotoCraft is not reachable" +
            (" — it is open but without its control server, so nothing can be read out of or "
             "written into it. Quit it (Ctrl+Q) and open it from the launcher card PhotoCraft, "
             "which starts it drivable." if _photocraft_running()
             else " — open your picture in PhotoCraft first (launcher card PhotoCraft), then "
                  "press From PhotoCraft."))
    os.makedirs(PHOTOCRAFT_PULL_DIR, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    dest = os.path.join(PHOTOCRAFT_PULL_DIR, f"from-photocraft-{stamp}.png")

    async def work(session):
        doc = await _pc_doc(session)
        if not doc.get("name"):
            raise RuntimeError("PhotoCraft has no image open — open the picture there first, "
                               "then press From PhotoCraft.")
        try:
            res = await asyncio.wait_for(
                session.call_tool("doc_export", {"path": dest}), timeout=120)
            txt = _text(res)
            if "error" in txt.lower() or "not granted" in txt.lower():
                raise RuntimeError(txt[:200])
            how = "exported the live document"
        except Exception:  # noqa: BLE001 - fall back to the document's own file
            src = doc.get("path")
            if not src or not os.path.isfile(src):
                raise
            shutil.copyfile(src, dest)
            how = "read the document's own file (export refused)"
        return {"doc": {k: doc.get(k) for k in ("name", "width", "height", "hasSelection",
                                               "path")},
                "layers": [{k: l.get(k) for k in ("id", "name", "visible", "hasMask")}
                           for l in (doc.get("layers") or [])],
                "how": how}

    def _go():
        return asyncio.run(_pc_bridge(work))

    # An occluded window answers every engine call with a 55 s timeout, so pull it forward and
    # retry once - the same treatment the hand-off needs.
    _raise_window()
    time.sleep(1.0)
    try:
        got = _go()
    except Exception as first:  # noqa: BLE001
        _raise_window()
        try:
            got = _go()
        except Exception as e:  # noqa: BLE001
            raise RuntimeError(f"Could not read the PhotoCraft document — {_plain_error(e)}") from e

    if not os.path.isfile(dest):
        raise RuntimeError(f"PhotoCraft reported a successful read but wrote nothing to {dest}.")
    # A hidden layer exports as a perfectly empty PNG, and then every phrase "selects 0 px" and looks
    # like a segmentation fault. Say what is actually wrong instead.
    with Image.open(dest) as im:
        ink = np.asarray(im.convert("RGBA").split()[-1])
    if int((ink > 8).sum()) < 64:
        layers = got.get("layers") or []
        all_hidden = bool(layers) and not any(l.get("visible") for l in layers)
        why = ("its layer is hidden — turn the eye back on in PhotoCraft's Layers panel"
               if all_hidden else
               "nothing in it is opaque — its layer may be hidden or fully masked out")
        raise RuntimeError(
            f"PhotoCraft's document {got['doc']['name']!r} exports as an empty picture: {why}. "
            "Make the picture visible in PhotoCraft, then press Execute Cutout again.")
    src = stage_source(path=dest)
    doc = got["doc"]
    _pc_pin = {"doc": doc["name"], "width": doc["width"], "height": doc["height"],
               "sid": src["id"], "export": dest, "how": got["how"], "at": time.time()}
    return {"source": src, "doc": doc, "how": got["how"], "pin": photocraft_pin(),
            "layers": got.get("layers") or [], "photocraft": photocraft_status()}


# ---------------------------------------------------------------- reference layers

# A layer named "image 2" is the workflow's image_2; the number in the name IS the slot. PhotoCraft
# ships documents whose layers are called "Layer 0", "Layer 1" …, so `layer N` is deliberately not
# accepted: it would quietly turn a normal layered document into a multi-image edit.
REF_LAYER_RE = re.compile(r"^\s*(?:image|img|picture|ref|input)\s*[_\-\u2013]?\s*(\d{1,2})\b",
                          re.IGNORECASE)
PHOTOCRAFT_REF_DIR = os.path.join(PHOTOCRAFT_ROOT, "cutout-studio", "_refs")
MAX_REF_SLOT = 5          # image_1 … image_5; the engine's encoder has 16 slots, VRAM does not
REF_SNAPSHOT = os.path.join(DATA, "refs-snapshot.json")
REF_EXPORT_TTL = 24 * 3600      # the engine uploads the bytes immediately; the files are scratch
REF_SNAPSHOT_MAX_AGE = 3600     # a snapshot older than this describes a document that has moved on


def _prune_ref_exports(ttl: float = REF_EXPORT_TTL) -> None:
    """Drop reference exports older than `ttl` — one dated folder is written per pull."""
    cutoff = time.time() - ttl
    try:
        for name in os.listdir(PHOTOCRAFT_REF_DIR):
            path = os.path.join(PHOTOCRAFT_REF_DIR, name)
            try:
                if os.path.isdir(path) and os.path.getmtime(path) < cutoff:
                    shutil.rmtree(path, ignore_errors=True)
            except OSError:
                pass
    except OSError:
        pass


def _save_ref_snapshot(doc: str, snap: list) -> None:
    """Remember the document's layer visibility before it is changed, so a failure can be undone.

    Written to disk, not just held in memory: an occluded PhotoCraft times out at 55 s per call, and
    a server that dies mid-export would otherwise leave the user's document with its layers hidden
    and no record of how they were.
    """
    if not doc or not snap:
        return
    try:
        os.makedirs(DATA, exist_ok=True)
        tmp = REF_SNAPSHOT + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump({"doc": doc, "at": time.time(),
                       "snap": [list(s) for s in snap]}, fh)
        os.replace(tmp, REF_SNAPSHOT)
    except OSError as e:  # noqa: BLE001 - the export still works; only the undo record is lost
        log.warning("could not record the layer visibility of %s: %s", doc, e)


def _load_ref_snapshot() -> dict:
    """A snapshot left behind by a failed export, when it is recent enough to still describe the doc."""
    try:
        with open(REF_SNAPSHOT, encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, ValueError):
        return {}
    if not isinstance(data, dict) or not data.get("doc") or not data.get("snap"):
        return {}
    if time.time() - float(data.get("at") or 0) > REF_SNAPSHOT_MAX_AGE:
        return {}
    try:
        return {"doc": data["doc"],
                "snap": [(s[0], s[1] or "", bool(s[2])) for s in data["snap"] if len(s) == 3]}
    except (TypeError, IndexError):
        return {}


def _clear_ref_snapshot() -> None:
    for path in (REF_SNAPSHOT, REF_SNAPSHOT + ".tmp"):
        try:
            os.remove(path)
        except OSError:
            pass


def ref_slot(layer_name: str) -> Optional[int]:
    """"image 2 — red jacket" -> 2. Anything else, including "Layer 2", -> None."""
    m = REF_LAYER_RE.match(layer_name or "")
    if not m:
        return None
    slot = int(m.group(1))
    return slot if 1 <= slot <= MAX_REF_SLOT else None


def photocraft_refs() -> dict:
    """Export the open document's reference layers for the engine's multi-image edit.

    A layer named `image N` is the N-th image the Qwen-Image 2.1 encoder sees, so a prompt that
    says "image 2" points at that layer's own pixels. The composite is image 1 unless a layer is
    named `image 1`, and — because a reference layer is material, not something drawn on the
    canvas — its own pixels are excluded from that composite.

    There is no per-layer export over the control port: `file.export.layersToFiles`,
    `layer.exportAs` and `layer.quickExportAsPng` all answer "uses ambient filesystem paths and is
    disabled", so each layer is isolated with `layer.setProps {visible}` and exported with
    `doc_export`. `layer.hideLayers`/`showLayers` cannot be used for this: they TOGGLE, so hiding
    an already-hidden layer brings it back (measured: isolating the middle of three layers
    exported the top one's colour). Visibility is restored from a snapshot taken before the first
    change, and always in a `finally`.
    """
    if not _port_up():
        raise RuntimeError("PhotoCraft is not reachable — open the picture there first.")
    os.makedirs(PHOTOCRAFT_REF_DIR, exist_ok=True)
    _prune_ref_exports()
    stamp = time.strftime("%Y%m%d-%H%M%S")
    out_dir = os.path.join(PHOTOCRAFT_REF_DIR, stamp)
    os.makedirs(out_dir, exist_ok=True)

    # A snapshot left on disk by an earlier failed export is the truth about this document, not a
    # fresh read of it: the failure may be exactly what left the layers hidden. Seeded here so the
    # first attempt (and the retry after it) restore the state the user actually had.
    saved: dict = _load_ref_snapshot()

    async def work(session):
        doc = await _pc_doc(session)
        doc_name = doc.get("name") or ""
        if not doc_name:
            raise RuntimeError("PhotoCraft has no image open.")
        layers = doc.get("layers") or []
        live = [(l.get("id"), l.get("name") or "", bool(l.get("visible"))) for l in layers]
        if saved.get("snap") and saved.get("doc") == doc_name:
            snap = list(saved["snap"])
        else:
            # first attempt on this document (or the user switched): record what is on screen now,
            # before the first setProps, and keep it where a crashed server can still find it.
            snap = live
            saved["doc"] = doc_name
            saved["snap"] = snap
            _save_ref_snapshot(doc_name, snap)
        slots = {}
        for lid, name, _vis in snap:
            slot = ref_slot(name)
            if slot is not None and slot not in slots:
                slots[slot] = (lid, name)
        if not slots:
            return {"doc": {k: doc.get(k) for k in ("name", "width", "height")}, "refs": [],
                    "base": None, "slots": [], "skipped": [],
                    "note": "no layer is named image 2, image 3 … — one image only"}

        wanted = sorted(slots)
        refs_out = []
        base_info = None

        async def isolate(lid):
            for other_id, _n, _v in snap:
                await _pc_run(session, "layer.setProps",
                              {"layer": other_id, "visible": other_id == lid})

        async def export(path):
            res = await asyncio.wait_for(session.call_tool("doc_export", {"path": path}),
                                         timeout=180)
            txt = _text(res)
            if not os.path.isfile(path):
                raise RuntimeError(f"PhotoCraft would not export {os.path.basename(path)} — {txt[:160]}")
            return txt

        try:
            # image 1: the layer named image 1, else the composite with the reference layers hidden.
            if 1 in slots:
                await isolate(slots[1][0])
                base_path = os.path.join(out_dir, "image-1.png")
                await export(base_path)
                base_info = {"path": base_path, "from": f"layer {slots[1][1]!r}", "slot": 1}
            else:
                for lid, _n, _v in snap:
                    await _pc_run(session, "layer.setProps",
                                  {"layer": lid, "visible": lid not in
                                   {slots[s][0] for s in wanted if s > 1}})
                base_path = os.path.join(out_dir, "image-1-composite.png")
                await export(base_path)
                base_info = {"path": base_path, "from": "the composite without the reference layers",
                             "slot": 1}

            for slot in [s for s in wanted if s > 1]:
                lid, name = slots[slot]
                await isolate(lid)
                path = os.path.join(out_dir, f"image-{slot}.png")
                await export(path)
                refs_out.append({"slot": slot, "name": name, "path": path})
        finally:
            restored = True
            for lid, _n, vis in snap:
                try:
                    await _pc_run(session, "layer.setProps", {"layer": lid, "visible": vis})
                except Exception as e:  # noqa: BLE001 - reported, and the snapshot stays on disk
                    restored = False
                    log.error("could not restore layer %s visibility: %s", lid, _plain_error(e))
            if restored:
                _clear_ref_snapshot()
            else:
                # The next pull adopts this snapshot and restores it before exporting anything else,
                # so a half-hidden document heals instead of drifting further.
                log.warning("left the layer snapshot in %s — the next pull restores it",
                            os.path.basename(REF_SNAPSHOT))
        return {"doc": {k: doc.get(k) for k in ("name", "width", "height")},
                "refs": refs_out, "base": base_info, "slots": wanted,
                "skipped": [n for _i, n, _v in snap if ref_slot(n) is not None
                            and n not in {slots[s][1] for s in slots}]}

    def _go():
        return asyncio.run(_pc_bridge(work))

    _raise_window()   # an occluded window times every engine call out at 55 s
    time.sleep(1.0)
    try:
        got = _go()
    except Exception as first:  # noqa: BLE001
        _raise_window()
        try:
            got = _go()
        except Exception as e:  # noqa: BLE001
            raise RuntimeError(f"Could not read the reference layers — {_plain_error(e)}") from e

    for entry in ([got["base"]] if got.get("base") else []) + list(got.get("refs") or []):
        entry["source"] = stage_source(path=entry["path"])
    return got


# ---------------------------------------------------------------- quick export

def _origin_map() -> dict:
    """The render -> ComfyUI file map, or an empty one when nothing has been delivered yet."""
    try:
        with open(AI_ORIGINS, encoding="utf-8") as fh:
            data = json.load(fh)
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def remember_ai_origin(delivered: str, origin: str) -> None:
    """Record which ComfyUI file a delivered render was made from.

    A render is copied out of ComfyUI's output tree into ~/Pictures/photocraft-ai so PhotoCraft
    can open it. The document therefore points at the copy, not at the render it came from — and
    a quick export has to go back to ComfyUI's folder, so the two are tied together here while the
    engine's own answer (filename + subfolder) is still in hand.
    """
    if not delivered or not origin:
        return
    data = _origin_map()
    data[os.path.abspath(delivered)] = os.path.abspath(origin)
    data[os.path.basename(delivered)] = os.path.abspath(origin)
    tmp = AI_ORIGINS + ".tmp"
    try:
        os.makedirs(DATA, exist_ok=True)
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=1)
        os.replace(tmp, AI_ORIGINS)
    except OSError as e:  # noqa: BLE001 - the picture is already delivered; this is metadata
        # A failed replace would otherwise leave ai-origins.json.tmp behind on every delivery.
        try:
            os.remove(tmp)
        except OSError:
            pass
        log.warning("could not record where %s came from: %s", delivered, e)


def _comfy_output_root() -> str:
    """ComfyUI's output folder for the engine's stack.

    Read from the engine's own config (`comfy_root`): the engine's `output_dir` key names the main
    stack, while its renders actually land under the qwen21 clone it drives (measured:
    ~/comfy/ComfyUI-qwen21/output/(date)/(date)_png/…).
    """
    cfg = os.path.expanduser("~/personal-files/python-projects/qwen21-studio/config.json")
    root = ""
    try:
        with open(cfg, encoding="utf-8") as fh:
            root = json.load(fh).get("comfy_root") or ""
    except (OSError, ValueError):
        root = ""
    return os.path.join(os.path.expanduser(root), "output") if root else \
        os.path.expanduser("~/comfy/ComfyUI-qwen21/output")


def comfy_source_path(filename: str, subfolder: str = "") -> Optional[str]:
    """The render in ComfyUI's output tree, or None when it is not where the engine said it is."""
    if not filename:
        return None
    path = os.path.join(_comfy_output_root(), subfolder or "", filename)
    return path if os.path.isfile(path) else None


def _next_version(folder: str, stem: str, ext: str) -> tuple[str, int]:
    """The first free `<stem>_v<n><ext>` in `folder`, counting every extension.

    Numbering restarts at 1 per stem and never reuses a number, so an export can never land on a
    picture that is already there: `name_v1.png`, then `name_v2.png`, whether the earlier one was
    written as PNG or as JPEG.
    """
    pat = re.compile(r"^" + re.escape(stem) + r"_v(\d+)\.[A-Za-z0-9]+$")
    top = 0
    try:
        for name in os.listdir(folder):
            m = pat.match(name)
            if m:
                top = max(top, int(m.group(1)))
    except OSError:
        pass
    n = top + 1
    while os.path.exists(os.path.join(folder, f"{stem}_v{n}{ext}")):
        n += 1
    return os.path.join(folder, f"{stem}_v{n}{ext}"), n


def quick_export() -> dict:
    """Write the document PhotoCraft has open back beside the picture it came from.

    One press, no parameters: the composite the app itself draws (unsaved edits, adjustment
    layers, filters — everything on screen) is exported next to the file the document came from,
    with the next free version number on the end. A picture the image engine made and this panel
    delivered is a copy of a render in ComfyUI's output tree, so it goes back *there* — the
    post-processed version then sits beside the render it came from.

    `doc_export` may only write inside PhotoCraft's automation root, so the composite is staged in
    ~/Pictures/cutout-studio/_quick-export first and copied to its real home here.
    """
    if not _port_up():
        raise RuntimeError(
            "PhotoCraft is not reachable" +
            (" — it is open but without its control server, so its document cannot be exported. "
             "Quit it (Ctrl+Q) and open it from the launcher card PhotoCraft, which starts it "
             "drivable." if _photocraft_running()
             else " — open your picture in PhotoCraft first (launcher card PhotoCraft), then press "
                  "Quick Export."))
    os.makedirs(PHOTOCRAFT_QUICK_DIR, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    staged = os.path.join(PHOTOCRAFT_QUICK_DIR,
                          f"quick-export-{stamp}-{uuid.uuid4().hex[:4]}.png")

    async def work(session):
        doc = await _pc_doc(session)
        if not doc.get("name"):
            raise RuntimeError("PhotoCraft has no image open — open the picture there first, "
                               "then press Quick Export.")
        res = await asyncio.wait_for(session.call_tool("doc_export", {"path": staged}), timeout=180)
        txt = _text(res)
        if "error" in txt.lower() or "not granted" in txt.lower():
            raise RuntimeError(f"PhotoCraft would not export the document: {txt[:200]}")
        return {k: doc.get(k) for k in ("name", "width", "height", "path")}

    def _go():
        return asyncio.run(_pc_bridge(work))

    # An occluded window answers every engine call with a 55 s timeout — raise it and retry once,
    # the same treatment the pull needs.
    _raise_window()
    time.sleep(1.0)
    try:
        doc = _go()
    except Exception as first:  # noqa: BLE001
        _raise_window()
        try:
            doc = _go()
        except Exception as e:  # noqa: BLE001
            raise RuntimeError(f"Could not export the PhotoCraft document — {_plain_error(e)}") from e

    if not os.path.isfile(staged) or os.path.getsize(staged) == 0:
        raise RuntimeError(f"PhotoCraft reported a successful export but wrote nothing to {staged}.")

    # Where does this document belong? The render it was delivered from (ComfyUI's output tree),
    # else the file the document was opened from, else — a document with no file yet — the panel's
    # own picture folder.
    doc_path = doc.get("path") or ""
    origins = _origin_map()
    origin = None
    if doc_path:
        origin = origins.get(os.path.abspath(doc_path)) or origins.get(os.path.basename(doc_path))
    if origin and not os.path.isfile(origin):
        origin = None
    target = origin or doc_path
    if not target or not os.path.isdir(os.path.dirname(target)):
        folder = PHOTOCRAFT_AI_DIR
        os.makedirs(folder, exist_ok=True)
        target = os.path.join(folder, doc["name"])
    folder = os.path.dirname(target)
    stem, ext = os.path.splitext(os.path.basename(target) or doc["name"])
    stem = stem or os.path.splitext(doc["name"])[0]
    ext = ext.lower() if ext.lower() in _QUICK_EXTS else ".png"

    dest, version = _next_version(folder, stem, ext)
    try:
        if ext == ".png":
            shutil.copyfile(staged, dest)
        else:
            with Image.open(staged) as im:
                if ext in (".jpg", ".jpeg"):
                    # JPEG has no alpha, and a straight convert() would flood transparency with
                    # black; the editor flattens onto white.
                    rgb = im.convert("RGBA")
                    flat = Image.new("RGB", rgb.size, (255, 255, 255))
                    flat.paste(rgb, mask=rgb.split()[-1])
                    flat.save(dest, quality=95, subsampling=0)
                else:
                    im.save(dest)
    finally:
        try:
            os.remove(staged)
        except OSError:
            pass
    if not os.path.isfile(dest) or os.path.getsize(dest) == 0:
        raise RuntimeError(f"The export wrote nothing to {dest}.")
    return {"ok": True, "path": dest, "folder": folder, "version": version,
            "doc": doc.get("name"), "size": [doc.get("width"), doc.get("height")],
            "from": target, "origin": "comfy" if origin else ("document" if doc_path else "new"),
            "bytes": os.path.getsize(dest)}


def send_to_photocraft(sid: str, mask_id: str, rings: Optional[list[dict]] = None,
                       phrase: str = "", mode: str = "selection", feather: float = 1.0,
                       bg: Optional[dict] = None, apply: str = "selection",
                       refine: str = "off", refine_radius: float = 6.0) -> dict:
    """Hand the result to PhotoCraft.

    mode="cutout"     write the files where PhotoCraft can read them and stop there.
    mode="open"       export, start PhotoCraft in control mode if needed, and load the original
                      image into it - no selection.
    mode="selection"  export, load the original image, and install the outline as a real lasso
                      selection, then read PhotoCraft's own measured area back so the two can be
                      compared instead of assumed equal. Everything after this happens inside
                      PhotoCraft: the pixels are untouched, the selection is live.

    When the source was pulled out of PhotoCraft (`photocraft_pull`), the result goes back into
    that same document and no copy of the image is opened: the user's document is the workspace.
    `apply` chooses what appears there - "selection" (default: the live lasso, marching ants,
    pixels untouched, nothing cut out and nothing hidden), "matte" (cut out: a real per-pixel
    alpha layer computed by matte.py, hair included), "mask" (layer mask, background hidden
    non-destructively) or "layer" (the subject copied onto its own layer, hard edge).
    `refine="smart"` runs PhotoCraft's own Refine Edge first, with the hair brush painted along
    the strand-like parts of the outline; it only applies to the mask route.
    """
    if apply not in ("selection", "mask", "layer", "matte"):
        raise ValueError(f"unknown apply mode {apply!r} — selection, mask, layer or matte")
    if refine not in ("off", "smart", "hair"):
        raise ValueError(f"unknown refine mode {refine!r} — off or smart")
    info = export_files(sid, mask_id, phrase=phrase, feather=feather, bg=bg)
    info["photocraft"] = photocraft_status()
    if mode == "cutout":
        info["ok"] = True
        info["message"] = f"Exported to {info['folder']}"
        return info

    pin = _pc_pin if (_pc_pin and _pc_pin.get("sid") == sid) else None
    info["pinned"] = bool(pin)

    if mode == "selection" and rings is None:
        rings = outline(mask_id)["rings"]

    strokes: list[dict] = []
    if mode == "selection" and refine != "off":
        try:
            strokes = _hair_refine_strokes(sid, mask_id)
        except Exception as e:  # noqa: BLE001 - refinement is an upgrade, not a requirement
            log.warning("hair strokes unavailable: %s", e)
            strokes = []
        info["strokes"] = len(strokes)

    ok, why = ensure_control()
    info["photocraft"] = photocraft_status()
    if not ok:
        info["ok"] = False
        info["message"] = f"Exported, but PhotoCraft could not be started: {why}"
        return info

    matte: Optional[dict] = None
    if apply == "matte" and mode == "selection":
        if not pin:
            info["ok"] = False
            info["message"] = ("A matte cut-out has to land in the document it came from — open the "
                               "picture in PhotoCraft and press Execute Cutout so the app can pull it.")
            return info
        try:
            matte = matte_cutout(sid, mask_id)
        except Exception as e:  # noqa: BLE001 - fall back to a selection rather than fail the run
            log.warning("matte unavailable, falling back to a selection: %s", e)
            info["matte_error"] = f"{type(e).__name__}: {e}"
            apply = "selection"
        else:
            info["matte"] = {k: matte[k] for k in ("soft_px", "opaque_px", "matte_seconds",
                                                   "path")}

    def _go():
        if mode == "selection" and matte is not None:
            layer_name = f"Cutout · {phrase.strip()}" if phrase.strip() else "Cutout · matte"
            return asyncio.run(_pc_bridge(
                lambda s: _pc_deliver_matte(s, matte["path"], pin["doc"], layer_name)))
        if mode == "selection":
            return asyncio.run(_pc_bridge(
                lambda s: _pc_load_and_select(s, info["source"], rings, pin=pin, apply=apply,
                                              phrase=phrase, refine=refine, strokes=strokes,
                                              refine_radius=refine_radius)))
        return asyncio.run(_pc_bridge(lambda s: _pc_open_only(s, info["source"])))

    if mode == "selection":
        _raise_window()   # an occluded window times every engine call out at 55 s
        time.sleep(0.8)
    try:
        res = _go()
    except Exception as first:  # noqa: BLE001 - the usual cause is an occluded window
        if mode == "selection":
            _raise_window()
        try:
            res = _go()
        except Exception as e:  # noqa: BLE001 - report, never crash the request
            info["ok"] = False
            info["message"] = (
                f"Exported, but the PhotoCraft bridge failed: {type(e).__name__}: {e}"
                + ("" if str(first) == str(e) else f" (first attempt: {first})"))
            return info

    info.update(res)
    info["ok"] = True
    px = _mask_store.get(mask_id, {})
    mask_px = int(px["mask"].sum()) if px.get("mask") is not None else None
    if res.get("applied") == "matte":
        m = info.get("matte") or {}
        info["message"] = (
            f"Cut out <span class=\"meta\">{phrase or 'the subject'}</span> as a real-alpha layer "
            f"<span class=\"meta\">{res['layer_name']}</span> in {res['doc']} — "
            f"{m.get('soft_px', 0):,} semi-transparent pixels (strands), matte took "
            f"{m.get('matte_seconds', 0)} s."
            + (" The layer it was cut from is hidden, not deleted." if res.get("hidden_layer")
               else " Hide the layer underneath to see the cut-out on its own."))
        return info
    if mode == "selection":
        msg = f"{res.get('shows') or 'a live selection'} in PhotoCraft ({res['doc']}, " \
              f"{res['steps']} lasso step(s)"
        if res.get("refined"):
            msg += f", {res['refined']}"
        if res.get("applied") == "selection":
            msg += " — the pixels are untouched, keep editing there"
        msg += ")."
        if res.get("selection_area") and mask_px:
            diff = abs(res["selection_area"] - mask_px) / mask_px * 100
            msg += (f" PhotoCraft measures {res['selection_area']:.0f} px against the mask's "
                    f"{mask_px} px — {diff:.2f}% apart.")
        info["message"] = msg
    else:
        info["message"] = f"Loaded {res['doc']} in PhotoCraft."
    return info


async def _pc_deliver_matte(session, png: str, doc_name: str, layer_name: str) -> dict:
    """Carry a real-alpha layer from our matte PNG into the user's open document.

    The control port has no command that imports a file into an existing document -- `doc_place`
    does not exist and `file.placeEmbedded` is disabled in control mode -- but a document can be
    opened and its layer copied into another one with `layer.copyToDocument`, which needs the
    destination's *index* in the document list. Indices are found by walking `document.activate`
    and reading each document's name back. The matte document is closed again afterwards.
    """
    matte_doc = os.path.basename(png)
    await asyncio.wait_for(session.call_tool("doc_open", {"path": png}), timeout=180)
    target = matte = None
    # the document list is however long the user's session is — walk it until activate refuses
    source_layer = None
    for i in range(64):
        try:
            res = await _pc_run(session, "document.activate", {"document": i}, timeout=60)
        except Exception:  # noqa: BLE001 - past the last document
            break
        if "no document at index" in _text(res):
            break                      # activate reports it, it does not raise
        doc = await _pc_doc(session)
        if doc.get("name") == doc_name:
            target = i
            source_layer = doc.get("activeLayer")   # what the cut-out replaces
        elif doc.get("name") == matte_doc:
            matte = i
    if target is None or matte is None:
        raise RuntimeError(f"could not locate the documents by name (target={doc_name!r}, "
                           f"matte={matte_doc!r}) in PhotoCraft's document list")
    await _pc_run(session, "document.activate", {"document": matte}, timeout=60)
    before = await _pc_doc(session)
    res = await _pc_run(session, "layer.copyToDocument",
                        {"document": target, "layer": before.get("activeLayer")}, timeout=180)
    new_id = json.loads(_text(res)).get("layers", [None])[0]
    await _pc_run(session, "document.activate", {"document": target}, timeout=60)
    doc = await _pc_doc(session)
    await _pc_run(session, "layer.select", {"layer": new_id}, timeout=60)
    await _pc_run(session, "layer.renameLayer", {"layer": new_id, "name": layer_name}, timeout=60)
    # The cut-out only shows if the layer it replaces stops showing: Affinity's "new layer with mask"
    # has the same requirement. Non-destructive - the layer is still there, just hidden.
    hidden = None
    if source_layer and source_layer != new_id:
        try:
            await _pc_run(session, "layer.select", {"layer": source_layer}, timeout=60)
            await _pc_run(session, "layer.hideLayers", {}, timeout=60)
            hidden = source_layer
        except Exception as e:  # noqa: BLE001 - the cut-out is still delivered
            log.warning("could not hide the source layer: %s", e)
    await _pc_run(session, "layer.select", {"layer": new_id}, timeout=60)
    try:
        await _pc_run(session, "document.activate", {"document": matte}, timeout=60)
        await _pc_run(session, "file.close", {}, timeout=60)
    except Exception as e:  # noqa: BLE001 - a stray tab is cosmetic
        log.warning("could not close the matte document: %s", e)
    await _pc_run(session, "document.activate", {"document": target}, timeout=60)
    return {"doc": doc.get("name"), "layer": new_id, "layer_name": layer_name,
            "hidden_layer": hidden, "applied": "matte"}


async def _pc_open_only(session, image_path: str) -> dict:
    doc = await _pc_doc(session)
    want = os.path.basename(image_path)
    if doc.get("name") == want:
        return {"doc": want, "size": [doc.get("width"), doc.get("height")],
                "steps": 0, "how": "already open"}
    res = await asyncio.wait_for(session.call_tool("doc_open", {"path": image_path}), timeout=180)
    msg = _text(res)
    if "automation filesystem access" in msg or "not granted" in msg:
        raise RuntimeError(
            "PhotoCraft is running without an automation root, so it cannot open new files. "
            "Quit PhotoCraft and press again — the app restarts it with " + PHOTOCRAFT_ROOT +
            " declared as its root.")
    # The app answers `doc_inspect` before the new document is loaded, and after a restart it
    # answers with an empty document. Reporting that as the delivery would print "opened in
    # PhotoCraft as null (null×null)", so wait for a document that actually has a name and a size.
    doc = {}
    for _ in range(12):
        doc = await _pc_doc(session)
        if doc.get("name") and doc.get("width"):
            break
        await asyncio.sleep(0.5)
    if not doc.get("name") or not doc.get("width"):
        raise RuntimeError(
            "PhotoCraft took the file but is not showing a document — its control server has "
            "probably restarted. Press Generate again; the panel reopens PhotoCraft when its port "
            "is gone.")
    return {"doc": doc.get("name"), "size": [doc.get("width"), doc.get("height")],
            "steps": 0, "how": "opened"}


# ---------------------------------------------------------------------------------------------
# Generated pictures -> PhotoCraft
#
# A generated picture is a new picture, so the default is to open it in PhotoCraft as its own
# document; "layer" copies it in as a new layer on whatever document is in front instead. Either
# way the file has to live under PhotoCraft's automation root first - the control port refuses to
# read anything outside it.
# ---------------------------------------------------------------------------------------------
PHOTOCRAFT_AI_DIR = os.path.join(PHOTOCRAFT_ROOT, "photocraft-ai")
PHOTOCRAFT_AI_MODES = ("document", "layer")
_ai_mode = os.environ.get("PHOTOCRAFT_AI_MODE", "document")
if _ai_mode not in PHOTOCRAFT_AI_MODES:
    _ai_mode = "document"


def ai_mode(mode: Optional[str] = None) -> str:
    """How a generated picture enters PhotoCraft: "document" (default) or "layer"."""
    global _ai_mode
    if mode is not None:
        if mode not in PHOTOCRAFT_AI_MODES:
            raise ValueError(f"unknown mode {mode!r} — document or layer")
        _ai_mode = mode
    return _ai_mode


def ai_filename(prompt: str) -> str:
    """`ai-<stamp>-<slug>-<4 hex>.png` — the hex keeps two renders of one prompt apart.

    The stamp is only good to the second, and two deliveries in the same second would otherwise
    write one file: the second picture replaces the first, and the origin map (which keys on the
    delivered path) then points the earlier document at the newer render.
    """
    stamp = time.strftime("%Y%m%d-%H%M%S")
    slug = _slug(prompt or "image")[:40] or "image"
    return f"ai-{stamp}-{slug}-{uuid.uuid4().hex[:4]}.png"


def _copied_layer_id(res: Any) -> Optional[str]:
    """The layer id `layer.copyToDocument` reports, or None when the reply carries no layer.

    The bridge answers JSON for this command, but it answers in sentences for others (`doc_inspect`)
    and returns no layer at all for `layer.newAdjustmentLayer.<kind>`. A reply that is not the
    expected JSON therefore has to read as "no id here", never as an exception — the copy may have
    landed, and the caller still has to close the document it came from.
    """
    try:
        payload = json.loads(_text(res))
    except (ValueError, TypeError):
        return None
    if not isinstance(payload, dict):
        return None
    layers = payload.get("layers")
    if isinstance(layers, list) and layers:
        first = layers[0]
        if isinstance(first, dict):
            return first.get("id")
        return first if isinstance(first, str) else None
    lid = payload.get("layer")
    return lid if isinstance(lid, str) else None


async def _pc_deliver_layer(session, png: str, layer_name: str) -> dict:
    """Copy a picture in as a layer of the document PhotoCraft has in front.

    Same index walk as `_pc_deliver_matte` — `layer.copyToDocument` addresses documents by their
    position in the session list, and that position is not 0 in general.
    """
    doc = await _pc_doc(session)
    target_name = doc.get("name")
    if not target_name:
        raise RuntimeError("PhotoCraft has no document open — open a picture first, or deliver the "
                           "generated image as a document instead.")
    picture = os.path.basename(png)
    await asyncio.wait_for(session.call_tool("doc_open", {"path": png}), timeout=180)
    target = source = None
    target_ids: list = []
    for i in range(64):
        res = await _pc_run(session, "document.activate", {"document": i}, timeout=60)
        if "no document at index" in _text(res):
            break
        d = await _pc_doc(session)
        if d.get("name") == target_name:
            target = i
            # The destination's layers before the copy: the fallback for finding the new one.
            target_ids = [l.get("id") for l in (d.get("layers") or [])]
        elif d.get("name") == picture:
            source = i
    if target is None or source is None:
        raise RuntimeError(f"could not find the documents after opening the generated image "
                           f"(target={target_name!r}, new={picture!r})")
    new_id = None
    try:
        await _pc_run(session, "document.activate", {"document": source}, timeout=60)
        opened = await _pc_doc(session)
        res = await _pc_run(session, "layer.copyToDocument",
                            {"document": target, "layer": opened.get("activeLayer")}, timeout=180)
        new_id = _copied_layer_id(res)
        await _pc_run(session, "document.activate", {"document": target}, timeout=60)
        if not new_id:
            # copyToDocument answered without a layer (it can answer in words, like doc_inspect):
            # the layer it made is the one on the destination that was not there before.
            after = (await _pc_doc(session)).get("layers") or []
            fresh = [l.get("id") for l in after if l.get("id") not in target_ids]
            new_id = fresh[0] if fresh else None
        if not new_id:
            raise RuntimeError(f"PhotoCraft copied no layer into {target_name!r} — nothing was renamed "
                               f"or raised, so the document is unchanged")
        await _pc_run(session, "layer.select", {"layer": new_id}, timeout=60)
        await _pc_run(session, "layer.renameLayer", {"layer": new_id, "name": layer_name}, timeout=60)
        # Explicitly on top: the generated picture is derived from what is below it, so it must never
        # land underneath the layer it was made from (copyToDocument already inserts at the top, but
        # that is its behaviour, not a promise).
        try:
            await _pc_run(session, "layer.arrange.bringToFront", {"layer": new_id}, timeout=60)
        except Exception as e:  # noqa: BLE001 - already on top in the common case
            log.warning("could not raise the delivered layer: %s", e)
        # `copyToDocument` hands the layer over with the source layer's state, and a delivered layer
        # that arrives hidden looks exactly like a failed delivery. Make "on top and visible" explicit.
        try:
            await _pc_run(session, "layer.setProps", {"layer": new_id, "visible": True}, timeout=60)
        except Exception as e:  # noqa: BLE001
            log.warning("could not show the delivered layer: %s", e)
    finally:
        # The generated picture's own document is scratch: close it and leave the user on the
        # document the layer landed in, whatever happened above.
        try:
            await _pc_run(session, "document.activate", {"document": source}, timeout=60)
            await _pc_run(session, "file.close", {}, timeout=60)
        except Exception as e:  # noqa: BLE001 - a stray tab is cosmetic
            log.warning("could not close the generated-image document: %s", e)
        try:
            await _pc_run(session, "document.activate", {"document": target}, timeout=60)
        except Exception as e:  # noqa: BLE001
            log.warning("could not return to %s: %s", target_name, e)
    stack = (await _pc_doc(session)).get("layers") or []
    below, visible = "", None
    for i, layer in enumerate(stack):
        if layer.get("id") == new_id:
            visible = layer.get("visible")
            below = stack[i + 1].get("name", "") if i + 1 < len(stack) else ""
            break
    return {"doc": target_name, "layer": new_id, "layer_name": layer_name,
            "above": below, "visible": visible, "layers": [l.get("name") for l in stack]}



# ---------------------------------------------------------------------------------------------
# Pose library hand-off
#
# A pose render is just a picture, but which slot it lands in decides whether the engine copies a
# pose at all: the graph sizes its latent from image_1 and splices every reference in as a VAE
# latent, so the pose has to be the DOCUMENT (image 1) and the character becomes a layer named
# `image 2`. That name is not cosmetic — `ref_slot()` is what finds a reference layer, and the
# encoder numbers its slots from 1.
# ---------------------------------------------------------------------------------------------
POSE_LAYER_NAME = "image 2"
#: Where a hand-off may read a file from, besides a loopback URL.
HANDOFF_ROOTS = (os.path.expanduser("~/Pictures"),
                 os.path.expanduser("~/personal-files/python-projects/pose-library"))


def read_handoff_image(url: Optional[str] = None, path: Optional[str] = None) -> bytes:
    """The bytes of a picture another local app is offering us.

    A URL is the normal route (the pose library serves its own renders, so a drag or a POST only
    carries a link). A path is accepted for a file the user already has, but only under the
    automation root or the pose library's own directory — never an arbitrary path from a request.
    """
    if url:
        if not re.match(r"^http://(127\.0\.0\.1|localhost):\d+/", url):
            raise ValueError("only a loopback URL can be fetched")
        import urllib.request
        with urllib.request.urlopen(url, timeout=20) as r:
            data = r.read()
        if not data:
            raise RuntimeError(f"nothing came back from {url}")
        return data
    if path:
        full = os.path.realpath(os.path.expanduser(path))
        if not any(full.startswith(os.path.realpath(root) + os.sep) for root in HANDOFF_ROOTS):
            raise ValueError("that path is outside the folders this hand-off may read")
        with open(full, "rb") as fh:
            return fh.read()
    raise ValueError("a pose_url or pose_path is required")


def pose_handoff(pose_bytes: bytes, name: str = "pose", mode: str = "image1",
                 source_bytes: Optional[bytes] = None) -> dict:
    """Put a pose render into PhotoCraft for the Qwen 2.1 edit graph.

    mode="image1"  the pose becomes its own document — i.e. the engine's image 1 and the canvas
    mode="pair"    the same, then the picture the pose was made for is copied in as `image 2`
    mode="layer"   the pose is copied into the document already in front as `image 2`
    """
    if mode not in ("image1", "pair", "layer"):
        raise ValueError("mode must be image1, pair or layer")
    if mode == "pair" and not source_bytes:
        raise ValueError("mode=pair needs the source picture as well")
    os.makedirs(PHOTOCRAFT_AI_DIR, exist_ok=True)
    stem = _slug(name or "pose")
    pose_path = os.path.join(PHOTOCRAFT_AI_DIR, f"pose-{stem}.png")
    with open(pose_path, "wb") as fh:
        fh.write(pose_bytes)
    source_path = None
    if source_bytes:
        source_path = os.path.join(PHOTOCRAFT_AI_DIR, f"pose-{stem}-source.png")
        with open(source_path, "wb") as fh:
            fh.write(source_bytes)

    def _work():
        async def go(session):
            if mode == "layer":
                info = await _pc_deliver_layer(session, pose_path, POSE_LAYER_NAME)
                info["how"] = "layer in the document already open"
                return info
            info = await _pc_open_only(session, pose_path)
            if source_path:
                extra = await _pc_deliver_layer(session, source_path, POSE_LAYER_NAME)
                info.update({"source_layer": extra.get("layer"),
                             "layer_name": POSE_LAYER_NAME,
                             "source_above": extra.get("above")})
            info["how"] = "document" + (" + image 2 layer" if source_path else "")
            return info
        return asyncio.run(_pc_bridge(go))

    try:
        info = _work()
    except Exception as first:  # noqa: BLE001 - same retry story as a render delivery
        log.warning("pose hand-off failed, retrying: %s", _plain_error(first))
        ok, why = ensure_control()
        if not ok and not _port_up():
            raise RuntimeError(f"Could not open the pose in PhotoCraft: {why}") from first
        _raise_window()
        time.sleep(1.0)
        try:
            info = _work()
        except Exception as e:  # noqa: BLE001
            raise RuntimeError(f"Could not open the pose in PhotoCraft — {_plain_error(e)}") from e
    info.update({"path": pose_path, "mode": mode, "name": name})
    if source_path:
        info["source_path"] = source_path
    return info

def deliver_ai_image(data: bytes, prompt: str = "", mode: Optional[str] = None,
                     name: Optional[str] = None, origin: Optional[str] = None) -> dict:
    """Put a picture produced by the image engine into PhotoCraft and say where it went.

    `origin` is the render's own file in ComfyUI's output tree. It is remembered when the picture
    arrives as its own document, so Quick Export can put the post-processed version back beside the
    render it was made from instead of beside the copy PhotoCraft opens.
    """
    mode = ai_mode(mode)
    os.makedirs(PHOTOCRAFT_AI_DIR, exist_ok=True)
    path = os.path.join(PHOTOCRAFT_AI_DIR, name or ai_filename(prompt))
    with open(path, "wb") as fh:
        fh.write(data)
    layer_name = ("AI · " + (prompt or "generated").strip())[:60]
    _raise_window()

    def _work():
        if mode == "layer":
            return asyncio.run(_pc_bridge(lambda s: _pc_deliver_layer(s, path, layer_name)))
        return asyncio.run(_pc_bridge(lambda s: _pc_open_only(s, path)))

    # A render is the user's picture: one transient bridge failure must not throw it away. If the
    # control port went down with the app, bring a drivable PhotoCraft back and try again; if the
    # app is up but the call still fails, raise its window (an occluded window answers nothing)
    # and try once more. Only then report.
    try:
        info = _work()
    except Exception as first:  # noqa: BLE001
        log.warning("delivery failed, retrying: %s", _plain_error(first))
        ok, why = ensure_control()
        if not ok and not _port_up():
            raise RuntimeError(f"Could not open the picture in PhotoCraft: {why}") from first
        _raise_window()
        time.sleep(1.0)
        try:
            info = _work()
        except Exception as e:  # noqa: BLE001
            raise RuntimeError(f"Could not open the picture in PhotoCraft — "
                               f"{_plain_error(e)}") from e
    if mode == "layer":
        info.update({"path": path, "mode": "layer"})
        info["adjustments"] = _apply_postprocess_prefs()
        return info
    # Its own document means it can be worked on and exported as that render — remember where the
    # render lives so Quick Export writes beside it.
    remember_ai_origin(path, origin or "")
    info.update({"path": path, "mode": "document", "layer_name": layer_name, "origin": origin})
    # A picture that has just arrived is finished the way the user left the section: the ticked
    # adjustments are added above it before the panel even reports the delivery.
    info["adjustments"] = _apply_postprocess_prefs()
    return info


# ---------------------------------------------------------------------------------------------
# Post-processing: adjustment layers in the picture open in PhotoCraft
#
# The finishing moves offered beside Generation Setup. Two of them, and neither takes a parameter:
# a Vibrance and a Curves adjustment layer, added above the active layer the way Layer > New
# Adjustment Layer does. The values below are the ones the user's own finishing layers carry in
# their document — "Vibrance 1" at vibrance +30 / saturation 0, and "Curves 1" at 0→0, 97→92,
# 164→177, 255→255 on the composite — so a press reproduces their look instead of dropping an
# identity adjustment that would change nothing.
# ---------------------------------------------------------------------------------------------
ADJUSTMENT_DEFAULTS: dict[str, dict] = {
    "vibrance": {"vibrance": 30.0, "saturation": 0.0},
    "curves": {"points": [[0, 0], [97, 92], [164, 177], [255, 255]]},
}
ADJUSTMENT_LABELS = {"vibrance": "Vibrance", "curves": "Curves"}


def _adjustment_layers(doc: dict, kind: str) -> list[dict]:
    """`doc`'s adjustment layers of `kind`, top of the stack first."""
    return [l for l in (doc.get("layers") or [])
            if ((l.get("adjustment") or {}).get("kind") == kind)]


async def _pc_adjust_add(session, kind: str) -> dict:
    """Add one adjustment layer of `kind` above the active layer, with the document's defaults."""
    doc = await _pc_doc(session)
    if not doc.get("name"):
        raise RuntimeError("PhotoCraft has no document open — open the picture to finish first")
    if not doc.get("activeLayer"):
        # "Above the active layer" needs a target. With nothing selected the adjustment would land
        # at the top of the stack anyway, but selecting it first keeps the app's own dialog
        # behaviour (the new layer is inserted above the active one) rather than relying on it.
        await _pc_run(session, "layer.selectTop", {}, timeout=60)
        doc = await _pc_doc(session)
    known = {l.get("id") for l in (doc.get("layers") or [])}
    await _pc_run(session, f"layer.newAdjustmentLayer.{kind}",
                  ADJUSTMENT_DEFAULTS[kind], timeout=180)
    after = await _pc_doc(session)
    fresh = [l for l in (after.get("layers") or []) if l.get("id") not in known]
    layer = fresh[0] if fresh else {}
    rows = after.get("layers") or []
    at = next((i for i, l in enumerate(rows) if l.get("id") == layer.get("id")), None)
    under = rows[at + 1].get("name") if at is not None and at + 1 < len(rows) else ""
    name = layer.get("name") or ADJUSTMENT_LABELS[kind]
    return {"doc": after.get("name"), "kind": kind, "action": "add",
            "layer": name, "layer_id": layer.get("id"), "above": under,
            "params": ADJUSTMENT_DEFAULTS[kind],
            "count": len(_adjustment_layers(after, kind)),
            "message": f"Added “{name}”" + (f" above “{under}”" if under else "")
                       + f" in {after.get('name')}."}


async def _pc_adjust_visible(session, kind: str, visible: bool) -> dict:
    """Switch every layer of `kind` on or off — the checkbox's job, not a destructive edit."""
    doc = await _pc_doc(session)
    if not doc.get("name"):
        raise RuntimeError("PhotoCraft has no document open — open the picture to finish first")
    layers = _adjustment_layers(doc, kind)
    label = ADJUSTMENT_LABELS[kind]
    if not layers:
        return {"doc": doc.get("name"), "kind": kind, "action": "visible", "visible": visible,
                "count": 0, "changed": 0,
                "message": f"There is no {label} layer to switch — press {label} to add one."}
    targets = [l for l in layers if bool(l.get("visible")) != visible]
    cmd = "layer.showLayers" if visible else "layer.hideLayers"
    was_active = doc.get("activeLayer")
    for layer in targets:
        lid = layer.get("id")
        # The visibility commands read the active layer, so select the one being switched.
        await _pc_run(session, "layer.select", {"layer": lid}, timeout=60)
        await _pc_run(session, cmd, {"layer": lid}, timeout=60)
    after = await _pc_doc(session)
    if was_active and any(l.get("id") == was_active for l in (after.get("layers") or [])):
        # Switching a layer is not a reason to move the user's selection: put it back, or the active
        # layer after "Curves off" is the hidden Curves layer.
        try:
            await _pc_run(session, "layer.select", {"layer": was_active}, timeout=60)
        except Exception as e:  # noqa: BLE001 - the switch itself already happened
            log.warning("could not restore the active layer: %s", _plain_error(e))
    on = sum(1 for l in _adjustment_layers(after, kind) if l.get("visible"))
    word = "shown" if visible else "hidden"
    return {"doc": after.get("name"), "kind": kind, "action": "visible", "visible": visible,
            "count": len(_adjustment_layers(after, kind)), "on": on, "changed": len(targets),
            "message": (f"{len(targets)} {label} layer{'s' if len(targets) > 1 else ''} {word}."
                        if targets else
                        f"{label} is already {'on' if visible else 'off'}.")}


def photocraft_adjustment_state() -> dict:
    """Which post-processing adjustments the open document carries, for the section's checkboxes."""
    async def work(session):
        doc = await _pc_doc(session)
        kinds = {}
        for kind in ADJUSTMENT_DEFAULTS:
            layers = _adjustment_layers(doc, kind)
            kinds[kind] = {"count": len(layers),
                           "visible": any(bool(l.get("visible")) for l in layers),
                           "layer": layers[0].get("name") if layers else None}
        return {"doc": doc.get("name"), "kinds": kinds}
    out = asyncio.run(_pc_bridge(work))
    out["prefs"] = postprocess_prefs()
    return out


def postprocess_prefs() -> dict:
    """Which adjustments the user left ticked — applied to every picture that arrives afterwards."""
    try:
        with open(POST_PREFS, encoding="utf-8") as fh:
            raw = json.load(fh)
    except (OSError, ValueError):
        return {kind: False for kind in ADJUSTMENT_DEFAULTS}
    if not isinstance(raw, dict):
        return {kind: False for kind in ADJUSTMENT_DEFAULTS}
    return {kind: bool(raw.get(kind, False)) for kind in ADJUSTMENT_DEFAULTS}


def set_postprocess_pref(kind: str, enabled: bool) -> dict:
    prefs = postprocess_prefs()
    prefs[kind] = bool(enabled)
    try:
        os.makedirs(DATA, exist_ok=True)
        with open(POST_PREFS, "w", encoding="utf-8") as fh:
            json.dump(prefs, fh)
    except OSError as e:  # noqa: BLE001 - the choice still applies to this session
        log.warning("could not save the Post-processing choice: %s", e)
    return prefs


def _apply_postprocess_prefs() -> list[str]:
    """Finish a freshly delivered picture with the adjustments the user left ticked.

    `ensure` semantics, not blind adding: a picture that already carries a layer of that kind (an
    I2I render layered into a finished document) gets it switched on instead of a second copy —
    two Vibrance layers at +30 would double the effect.
    """
    wanted = [k for k, on in postprocess_prefs().items() if on]
    if not wanted:
        return []
    done: list[str] = []

    async def work(session):
        for kind in wanted:
            doc = await _pc_doc(session)
            if not doc.get("name"):
                return done
            if _adjustment_layers(doc, kind):
                await _pc_adjust_visible(session, kind, True)
            else:
                await _pc_adjust_add(session, kind)
            done.append(ADJUSTMENT_LABELS[kind])
        return done

    try:
        return asyncio.run(_pc_bridge(work))
    except Exception as e:  # noqa: BLE001 - the picture is already in PhotoCraft
        log.warning("could not add the post-processing layers: %s", _plain_error(e))
        return done


def photocraft_adjustment(action: str = "add", kind: str = "",
                          enabled: Optional[bool] = None) -> dict:
    """Add or switch a post-processing adjustment layer in the document PhotoCraft has open.

    action: "add"     — a new layer above the active one, carrying the document's default values;
            "enable"  — switch that kind on, adding a layer when the document has none;
            "disable" — switch every layer of that kind off (they stay in the stack);
            "state"   — what the document carries now, plus which adjustments are ticked for new
                        pictures;
            "toggle"  — the checkbox: remember the choice for new pictures *and* apply it to the
                        document open now;
            "pref"    — remember the choice only.
    kind:   "vibrance" or "curves".
    enabled: the new choice, for "pref"/"toggle".
    """
    if action == "state":
        return photocraft_adjustment_state()
    if action == "prefs":
        return {"prefs": postprocess_prefs()}
    kind = (kind or "").lower()
    if kind not in ADJUSTMENT_DEFAULTS:
        raise RuntimeError(f"Unknown adjustment {kind!r} — Post-processing offers "
                           + " and ".join(ADJUSTMENT_LABELS.values()) + ".")
    if action == "pref":
        return {"prefs": set_postprocess_pref(kind, bool(enabled)), "kind": kind}
    if action not in ("add", "enable", "disable", "toggle"):
        raise RuntimeError(f"Unknown action {action!r} — use add, enable, disable, toggle, pref or state.")

    async def work(session):
        if action == "add":
            return await _pc_adjust_add(session, kind)
        if action in ("enable", "disable"):
            want = action == "enable"
            if want:
                doc = await _pc_doc(session)
                if not _adjustment_layers(doc, kind):
                    return await _pc_adjust_add(session, kind)
            return await _pc_adjust_visible(session, kind, want)
        # toggle: the checkbox is both a standing choice and a switch for what is open now.
        want = bool(enabled)
        prefs = set_postprocess_pref(kind, want)
        if want:
            doc = await _pc_doc(session)
            if not _adjustment_layers(doc, kind):
                info = await _pc_adjust_add(session, kind)
                info["prefs"] = prefs
                return info
        info = await _pc_adjust_visible(session, kind, want)
        info["prefs"] = prefs
        return info

    return asyncio.run(_pc_bridge(work))
