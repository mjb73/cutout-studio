"""Cutout Studio — FastAPI app: browser UI, JSON API, and an MCP endpoint at /mcp.

Run:  ./start.sh          -> http://127.0.0.1:7887
MCP:  http://127.0.0.1:7887/mcp   (streamable HTTP; same process, same code path as the UI)

The browser and the MCP tools both call services.py, so "type a phrase, get the subject back"
has exactly one implementation.
"""

from __future__ import annotations

import asyncio
import os
import socket
import subprocess
import time
from contextlib import asynccontextmanager
from typing import Any, Optional

import httpx
import uvicorn
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel

import services as svc

BASE = os.path.dirname(os.path.abspath(__file__))
STATIC = os.path.join(BASE, "static")
PORT = int(os.environ.get("CUTOUT_PORT", svc.CFG.get("port", 7887)))


# ---------------------------------------------------------------- MCP tools
# These are the app's own MCP surface: any MCP client (Hermes, another agent, PhotoCraft's
# agent through Hermes) can name a phrase and get a mask, a cut-out and an outline back.

mcp = FastMCP(
    "cutout-studio",
    instructions=(
        "Phrase-driven selections and background removal for still images, on a local SAM3 "
        "stack. Name what to keep in words ('hair', 'the dress', 'mountain'), get a mask PNG, "
        "a cut-out PNG with alpha, and a vector outline. Results are also exported where "
        "PhotoCraft can read them (~/Pictures/cutout-studio)."
    ),
    streamable_http_path="/",
)


def _json(payload: dict) -> str:
    import json
    return json.dumps(payload, indent=1)


@mcp.tool()
def cutout_status() -> str:
    """Whether the local SAM3 stack is usable, which ComfyUI it is using, and the phrase chips."""
    return _json(svc.status())


@mcp.tool()
def cutout_segment(image_path: str, phrase: str, threshold: float = 0.5,
                   refine: int = 2, invert: bool = False) -> str:
    """Select what `phrase` names in an image and return the mask.

    phrase: the object's name ("hair", "face", "top", "mountain"); comma-separate to union
    several ("hair, face"). invert=True keeps everything EXCEPT the phrase (background
    removal). Returns mask_id, coverage %, bbox, warnings and the mask PNG path/URL.
    """
    try:
        src = svc.stage_source(path=image_path)
        out = svc.segment(src["id"], phrase, threshold=threshold, refine=refine, invert=invert)
        out["source_id"] = src["id"]
        out["source_path"] = src["path"]
        out["mask_path"] = svc.mask_file(out["mask_id"])
        return _json(out)
    except (svc.BackendError, ValueError, FileNotFoundError) as e:
        raise RuntimeError(str(e)) from e


@mcp.tool()
def cutout_cutout(image_path: str, phrase: str, feather: float = 1.0,
                  background: str = "transparent", threshold: float = 0.5,
                  refine: int = 2, invert: bool = False) -> str:
    """Select the phrase and return a cut-out PNG (alpha) plus its path.

    background: "transparent" (default), or a colour like "#ffffff" / "#1b2b45", or
    "gradient:#1b2b45,#f0f4ff" for a vertical gradient.
    """
    bg = None
    if background and background.startswith("gradient:"):
        parts = background.split(":", 1)[1].split(",")
        bg = {"type": "gradient", "color": parts[0], "color2": parts[-1], "angle": 90}
    elif background and background not in ("transparent", "none"):
        bg = {"type": "solid", "color": background}
    try:
        src = svc.stage_source(path=image_path)
        seg = svc.segment(src["id"], phrase, threshold=threshold, refine=refine, invert=invert)
        made = svc.make_cutout(src["id"], seg["mask_id"], feather=feather, bg=bg)
        return _json({"cutout_path": made["path"], "mask_path": seg["mask_url"],
                      "coverage": seg["coverage"], "warnings": seg["warnings"],
                      "elapsed": seg["elapsed"], "size": [made["width"], made["height"]]})
    except (svc.BackendError, ValueError, FileNotFoundError) as e:
        raise RuntimeError(str(e)) from e


@mcp.tool()
def cutout_outline(image_path: str, phrase: str, simplify: float = 1.0,
                   threshold: float = 0.5, invert: bool = False) -> str:
    """The selection as polygons in image pixels — what a PhotoCraft lasso selection takes."""
    try:
        src = svc.stage_source(path=image_path)
        seg = svc.segment(src["id"], phrase, threshold=threshold, invert=invert)
        rings = svc.outline(seg["mask_id"], simplify=simplify)
        return _json({"rings": rings["rings"], "points": rings["points"],
                      "count": rings["count"], "size": [rings["width"], rings["height"]],
                      "mask_url": seg["mask_url"], "coverage": seg["coverage"]})
    except (svc.BackendError, ValueError, FileNotFoundError) as e:
        raise RuntimeError(str(e)) from e


@mcp.tool()
def cutout_from_photocraft(phrase: str, apply: str = "selection", feather: float = 1.0,
                           threshold: float = 0.5, refine: int = 2,
                           hair_refine: str = "off", refine_radius: float = 6.0,
                           invert: bool = False) -> str:
    """Segment the picture the user has open in PhotoCraft and show the result there.

    PhotoCraft is the front end: this pulls the live document out of it (composite, unsaved
    edits included), runs the phrase through SAM3 here, and installs the outline back into that
    same document — no copy is opened.
    apply="selection"  (default) leave a live lasso selection — marching ants, pixels untouched,
                       nothing cut out, nothing hidden.
    apply="matte"      cut it out: compute a real per-pixel alpha here (BiRefNet matte + SAM3 core,
                       colour decontaminated) and carry it into the document as a layer with genuine
                       transparency — the Affinity-grade hair edge; the layer it was cut from is
                       hidden, not deleted.
    apply="mask"       turn it into a layer mask, so the background is hidden non-destructively.
    apply="layer"      copy the subject onto its own named layer above the original.
    hair_refine="smart" runs PhotoCraft's own Refine Edge (smart radius + hair brush strokes
                       along the strand-like parts of the outline) before applying — this is what
                       brings fine hair back, a polygon edge cannot carry wisps.
    Returns the document, PhotoCraft's own measured selection area and how far it is from the
    mask, so the two numbers can be compared instead of assumed equal.
    """
    try:
        pulled = svc.photocraft_pull()
        src = pulled["source"]
        seg = svc.segment(src["id"], phrase, threshold=threshold, refine=refine, invert=invert)
        res = svc.send_to_photocraft(src["id"], seg["mask_id"], phrase=phrase,
                                     mode="selection", feather=feather, apply=apply,
                                     refine=hair_refine, refine_radius=refine_radius)
        res["pulled_from"] = pulled["doc"]
        res["read_how"] = pulled["how"]
        res["coverage"] = seg["coverage"]
        res["warnings"] = seg["warnings"]
        return _json(res)
    except (svc.BackendError, ValueError, FileNotFoundError, RuntimeError) as e:
        raise RuntimeError(str(e)) from e


@mcp.tool()
def cutout_to_photocraft(image_path: str, phrase: str, feather: float = 1.0,
                         background: str = "transparent", mode: str = "selection",
                         apply: str = "selection", hair_refine: str = "off",
                         refine_radius: float = 6.0) -> str:
    """Segment, load the image into PhotoCraft and install the selection there.

    Every mode writes <slug>-source.png, -cutout.png, -mask.png and -selection.json into
    ~/Pictures/cutout-studio/<date>/ (a sandbox-granted root, so PhotoCraft can read them).
    mode="cutout"    files only.
    mode="open"      start PhotoCraft in control mode if needed and load the ORIGINAL image
                     into it, no selection.
    mode="selection" load the original image and install the outline as a live lasso
                     selection, leaving the pixels untouched so editing continues inside
                     PhotoCraft. Returns the document, step count and PhotoCraft's own
                     measured selection area.
    apply            what appears in PhotoCraft: "selection" (ants), "mask" (layer mask, the
                     background hidden) or "layer" (subject copied onto its own layer).
    """
    try:
        src = svc.stage_source(path=image_path)
        seg = svc.segment(src["id"], phrase)
        bg = None
        if background and background.startswith("gradient:"):
            parts = background.split(":", 1)[1].split(",")
            bg = {"type": "gradient", "color": parts[0], "color2": parts[-1], "angle": 90}
        elif background and background not in ("transparent", "none"):
            bg = {"type": "solid", "color": background}
        res = svc.send_to_photocraft(src["id"], seg["mask_id"], phrase=phrase, mode=mode,
                                     feather=feather, bg=bg, apply=apply,
                                     refine=hair_refine, refine_radius=refine_radius)
        res["coverage"] = seg["coverage"]
        res["warnings"] = seg["warnings"]
        return _json(res)
    except (svc.BackendError, ValueError, FileNotFoundError) as e:
        raise RuntimeError(str(e)) from e


@mcp.tool()
def cutout_quick_export() -> str:
    """Save the document open in PhotoCraft back beside the picture it came from.

    No parameters: the composite PhotoCraft is drawing (unsaved edits, adjustment layers, filters)
    is exported next to the file the document came from with the next free version number on the
    end — a picture the image engine made goes back to ComfyUI's output folder, beside the render
    it was made from. Returns the written path, the version number and the folder.
    """
    try:
        return _json(svc.quick_export())
    except (RuntimeError, OSError) as e:
        raise RuntimeError(str(e)) from e


@mcp.tool()
def cutout_adjustment(action: str = "add", kind: str = "vibrance",
                      enabled: Optional[bool] = None) -> str:
    """Add or switch a post-processing adjustment layer in the document PhotoCraft has open.

    action: "add" (a new Vibrance or Curves adjustment layer above the active layer, carrying the
    values the user's own finishing layers use), "enable"/"disable" (switch that kind on or off —
    non-destructive, the layers stay in the stack), "toggle" (tick/untick the box: remember the
    choice for new pictures and apply it to the document now), "state" (what the document carries
    now, plus which adjustments are ticked), or "pref" (remember the choice only).
    kind: "vibrance" or "curves". enabled: the new choice for toggle/pref.
    """
    try:
        return _json(svc.photocraft_adjustment(action, kind, enabled))
    except (RuntimeError, OSError) as e:
        raise RuntimeError(str(e)) from e


# ---------------------------------------------------------------- HTTP app

@asynccontextmanager
async def lifespan(app: FastAPI):
    # A mounted Starlette sub-app's own lifespan is never run by the parent, so the MCP
    # session manager is started here — without it /mcp accepts the connection and then
    # fails every session.
    async with mcp.session_manager.run():
        yield


app = FastAPI(title="Cutout Studio", lifespan=lifespan)


class UploadPath(BaseModel):
    path: str


class SegmentIn(BaseModel):
    id: str
    phrase: str
    threshold: float = 0.5
    refine: int = 2
    invert: bool = False


class CutoutIn(BaseModel):
    id: str
    mask_id: str
    feather: float = 1.0
    bg: Optional[dict] = None


class OutlineIn(BaseModel):
    mask_id: str
    simplify: float = 1.0


class ExportIn(BaseModel):
    id: str
    mask_id: str
    phrase: str = ""
    feather: float = 1.0
    bg: Optional[dict] = None
    mode: str = "cutout"       # "cutout" = files only, "selection" = also push to PhotoCraft
    apply: str = "selection"   # "selection" | "matte" | "mask" | "layer" — what appears in PhotoCraft
    refine: str = "off"        # "off" | "smart" — PhotoCraft's Refine Edge with hair strokes
    refine_radius: float = 6.0


@app.get("/")
async def index() -> FileResponse:
    # Never cached: the panel is edited in place, and a heuristically-cached index.html keeps loading
    # yesterday's app.js stamp — the page then runs old code while every file on disk looks right.
    return FileResponse(
        os.path.join(STATIC, "index.html"),
        headers={"Cache-Control": "no-store, must-revalidate"},
    )


@app.get("/api/state")
async def api_state() -> JSONResponse:
    st = svc.status()
    st["photocraft"] = svc.photocraft_status()
    st["pin"] = svc.photocraft_pin()
    st["port"] = PORT
    st["mcp_url"] = f"http://127.0.0.1:{PORT}/mcp"
    return JSONResponse(st)


@app.post("/api/upload")
async def api_upload(file: UploadFile = File(...)) -> JSONResponse:
    blob = await file.read()
    if not blob:
        raise HTTPException(400, "Empty upload")
    try:
        return JSONResponse(svc.stage_source(blob=blob, name=file.filename))
    except Exception as e:  # noqa: BLE001
        raise HTTPException(400, f"Cannot read that image: {e}")


@app.post("/api/open")
def api_open(body: UploadPath) -> JSONResponse:
    try:
        return JSONResponse(svc.stage_source(path=body.path))
    except FileNotFoundError:
        raise HTTPException(404, f"No such file: {body.path}")
    except Exception as e:  # noqa: BLE001
        raise HTTPException(400, f"Cannot read that image: {e}")


@app.post("/api/segment")
def api_segment(body: SegmentIn) -> JSONResponse:
    try:
        return JSONResponse(svc.segment(body.id, body.phrase, threshold=body.threshold,
                                        refine=body.refine, invert=body.invert))
    except svc.BackendError as e:
        raise HTTPException(503, str(e))
    except (ValueError, FileNotFoundError) as e:
        raise HTTPException(400, str(e))


@app.post("/api/cutout")
def api_cutout(body: CutoutIn) -> JSONResponse:
    try:
        made = svc.make_cutout(body.id, body.mask_id, feather=body.feather, bg=body.bg)
    except (ValueError, FileNotFoundError, KeyError) as e:
        raise HTTPException(400, str(e))
    made.pop("png", None)
    return JSONResponse(made)


@app.post("/api/outline")
def api_outline(body: OutlineIn) -> JSONResponse:
    try:
        return JSONResponse(svc.outline(body.mask_id, simplify=body.simplify))
    except KeyError as e:
        raise HTTPException(404, f"Unknown mask: {e}")
    except Exception as e:  # noqa: BLE001
        raise HTTPException(500, str(e))


@app.post("/api/export")
def api_export(body: ExportIn) -> JSONResponse:
    if body.mode not in ("cutout", "open", "selection"):
        raise HTTPException(400, f"Unknown mode {body.mode!r} — cutout, open or selection")
    if body.apply not in ("matte", "selection", "mask", "layer"):
        raise HTTPException(400, f"Unknown apply {body.apply!r} — matte, selection, mask or layer")
    if body.refine not in ("off", "smart", "hair"):
        raise HTTPException(400, f"Unknown refine {body.refine!r} — off or smart")
    try:
        res = svc.send_to_photocraft(body.id, body.mask_id, phrase=body.phrase,
                                     mode=body.mode, feather=body.feather, bg=body.bg,
                                     apply=body.apply, refine=body.refine,
                                     refine_radius=body.refine_radius)
        return JSONResponse(res)
    except (ValueError, FileNotFoundError, KeyError) as e:
        raise HTTPException(400, str(e))


@app.post("/api/window/close")
def api_window_close() -> JSONResponse:
    """Close this app's own window — the split column is frameless, so it has no title-bar X.

    The argument `--app=http://127.0.0.1:7887/` exists only on the dedicated-profile window the
    split wrapper opens (same handle it tiles by), so a browser tab showing this page is never
    touched: no match, nothing killed.
    """
    pids = subprocess.run(
        ["pgrep", "-f", "--", "app=http://127.0.0.1:7887/"],
        capture_output=True, text=True,
    ).stdout.split()
    closed = []
    for p in pids:
        try:
            os.kill(int(p), 15)
            closed.append(int(p))
        except (ValueError, ProcessLookupError, PermissionError):
            pass
    return JSONResponse({"closed": closed})


@app.post("/api/reset")
def api_reset() -> JSONResponse:
    """Back to the initial state: no source, no mask, no pinned PhotoCraft document."""
    return JSONResponse(svc.reset_state())


@app.post("/api/photocraft/pull")
def api_pc_pull() -> JSONResponse:
    """Make the picture open in PhotoCraft this app's source — PhotoCraft is the front door."""
    try:
        return JSONResponse(svc.photocraft_pull())
    except RuntimeError as e:
        raise HTTPException(409, str(e))
    except Exception as e:  # noqa: BLE001
        raise HTTPException(500, f"{type(e).__name__}: {e}")


@app.get("/api/photocraft/doc")
def api_pc_doc() -> JSONResponse:
    """What PhotoCraft has open right now, so the UI can say what a press will act on."""
    return JSONResponse(svc.photocraft_doc())


@app.post("/api/photocraft/refs")
def api_pc_refs() -> JSONResponse:
    """The open document's reference layers (image 2, image 3 …) as separate images for the engine."""
    try:
        return JSONResponse(svc.photocraft_refs())
    except RuntimeError as e:
        raise HTTPException(409, str(e))
    except Exception as e:  # noqa: BLE001
        raise HTTPException(500, f"{type(e).__name__}: {e}")


class AdjustmentIn(BaseModel):
    """Post-processing: which adjustment, and whether to add it or switch it (no other parameters)."""
    action: str = "add"            # add | enable | disable | toggle | pref | state
    kind: str = ""                 # vibrance | curves
    enabled: Optional[bool] = None  # the new choice, for toggle/pref


@app.post("/api/photocraft/adjustment")
def api_pc_adjustment(body: AdjustmentIn) -> JSONResponse:
    """Add or switch a post-processing adjustment layer (Vibrance, Curves) in the open document."""
    try:
        return JSONResponse(svc.photocraft_adjustment(body.action, body.kind, body.enabled))
    except RuntimeError as e:
        raise HTTPException(409, str(e))
    except Exception as e:  # noqa: BLE001
        raise HTTPException(500, f"{type(e).__name__}: {e}")


@app.post("/api/quick-export")
def api_quick_export() -> JSONResponse:
    """Export the document PhotoCraft has open back beside the picture it came from, versioned."""
    try:
        return JSONResponse(svc.quick_export())
    except RuntimeError as e:
        raise HTTPException(409, str(e))
    except Exception as e:  # noqa: BLE001
        raise HTTPException(500, f"{type(e).__name__}: {e}")


@app.get("/api/health")
async def api_health() -> JSONResponse:
    return JSONResponse({"ok": True, "port": PORT, "ts": time.time()})


# ---------------------------------------------------------------------------------------------
# PhotoCraft AI: the image engine
#
# The generator is the Qwen-Image studio, which already knows how to talk to its own ComfyUI. It is
# kept as a separate process on its own port (like ComfyUI itself) rather than reimplemented here:
# the UI embeds it, this server starts it when needed, and every finished render is handed back to
# be put into PhotoCraft.
# ---------------------------------------------------------------------------------------------
# Port and launcher come from config.json's `ai_engine` (see config.example.json) so a clone does not
# inherit this machine's launcher-dashboard layout: by default the sibling `qwen21-studio` checkout
# is started, and an empty `start` means "start it yourself, the panel only looks for it".
_AI_CFG = svc.CFG.get("ai_engine", {}) if isinstance(svc.CFG, dict) else {}
AI_PORT = int(os.environ.get("PHOTOCRAFT_AI_ENGINE_PORT", str(_AI_CFG.get("port", 7866))))


def _ai_launcher() -> str:
    """The command that starts the engine: configured launcher, or the sibling checkout."""
    configured = os.path.expanduser(str(_AI_CFG.get("start") or ""))
    if configured:
        return configured
    here = os.path.dirname(os.path.abspath(__file__))
    for candidate in ("../qwen21-studio/start.sh", "../qwen21-studio/run.sh"):
        path = os.path.abspath(os.path.join(here, candidate))
        if os.path.isfile(path):
            return path
    return ""


AI_WRAPPER = _ai_launcher()


class AiDeliverIn(BaseModel):
    filename: str
    prompt: str = ""
    subfolder: str = ""
    mode: Optional[str] = None


class AiModeIn(BaseModel):
    mode: str


class PoseHandoffIn(BaseModel):
    """What the pose library offers: a pose render, optionally the picture it was made for."""
    pose_url: Optional[str] = None
    pose_path: Optional[str] = None
    source_url: Optional[str] = None
    source_path: Optional[str] = None
    name: str = "pose"
    mode: str = "image1"          # image1 | pair | layer


def _ai_up() -> bool:
    with socket.socket() as s:
        s.settimeout(0.3)
        return s.connect_ex(("127.0.0.1", AI_PORT)) == 0


@app.get("/api/ai/status")
def api_ai_status() -> JSONResponse:
    return JSONResponse({"up": _ai_up(), "port": AI_PORT, "mode": svc.ai_mode()})


@app.post("/api/ai/start")
def api_ai_start() -> JSONResponse:
    """Bring the engine up if it is not already listening (the card that used to do this lives on)."""
    if _ai_up():
        return JSONResponse({"up": True, "started": False, "port": AI_PORT})
    if not AI_WRAPPER or not os.path.exists(AI_WRAPPER):
        raise HTTPException(500, "no image engine configured: set ai_engine.start in config.json "
                                 "to your qwen21-studio checkout (its start.sh)")
    subprocess.Popen([AI_WRAPPER], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                     stdin=subprocess.DEVNULL, start_new_session=True)
    for _ in range(120):
        if _ai_up():
            return JSONResponse({"up": True, "started": True, "port": AI_PORT})
        time.sleep(0.5)
    raise HTTPException(504, "the image engine did not start within 60 s")


@app.post("/api/ai/mode")
def api_ai_mode(body: AiModeIn) -> JSONResponse:
    try:
        return JSONResponse({"mode": svc.ai_mode(body.mode)})
    except ValueError as e:
        raise HTTPException(400, str(e))


@app.post("/api/ai/deliver")
async def api_ai_deliver(body: AiDeliverIn) -> JSONResponse:
    """A finished render -> PhotoCraft, as a new document (default) or a new layer."""
    try:
        params = {"filename": body.filename, "type": "output"}
        if body.subfolder:
            params["subfolder"] = body.subfolder
        async with httpx.AsyncClient(timeout=30.0) as client:
            r = await client.get(f"http://127.0.0.1:{AI_PORT}/api/view", params=params)
            r.raise_for_status()
            data = r.content
        if not data:
            raise RuntimeError("the engine returned an empty image")
        # Where this render actually lives in ComfyUI's output tree — Quick Export writes the
        # post-processed version back beside it, not beside the copy PhotoCraft opens.
        origin = await asyncio.to_thread(svc.comfy_source_path, body.filename, body.subfolder)
        res = await asyncio.to_thread(svc.deliver_ai_image, data, body.prompt, body.mode, None, origin)
        return JSONResponse({"ok": True, **res})
    except RuntimeError as e:
        raise HTTPException(409, str(e))
    except Exception as e:  # noqa: BLE001
        raise HTTPException(500, f"{type(e).__name__}: {e}")


@app.post("/api/pose/handoff")
async def api_pose_handoff(body: PoseHandoffIn) -> JSONResponse:
    """A pose library's render -> PhotoCraft, in the slot the pose has to occupy.

    image1: the pose opens as its own document, which is what makes it the engine's image 1 and
    the canvas the latent is sized from. pair: the picture the pose was made for is copied in on
    top as the layer `image 2`. layer: the pose joins the document already open as `image 2`.
    Delivery is the library's own button; nothing here touches the prompt or the model.
    """
    try:
        pose = await asyncio.to_thread(svc.read_handoff_image, body.pose_url, body.pose_path)
        source = None
        if body.mode == "pair":
            source = await asyncio.to_thread(svc.read_handoff_image, body.source_url,
                                             body.source_path)
        res = await asyncio.to_thread(svc.pose_handoff, pose, body.name, body.mode, source)
        return JSONResponse({"ok": True, **res})
    except ValueError as e:
        raise HTTPException(400, str(e))
    except RuntimeError as e:
        raise HTTPException(409, str(e))
    except Exception as e:  # noqa: BLE001
        raise HTTPException(500, f"{type(e).__name__}: {e}")


app.mount("/files", StaticFiles(directory=svc.DATA), name="files")
app.mount("/static", StaticFiles(directory=STATIC), name="static")
app.mount("/mcp", mcp.streamable_http_app())


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=PORT, log_level="info")
