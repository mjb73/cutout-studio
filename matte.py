"""Hair-accurate alpha, computed here instead of delegated to PhotoCraft's feather-based refine.

BiRefNet-portrait (the ONNX matte rembg ships) gives a per-pixel alpha — real partially transparent
strands. On its own it ghosts the hair body, so the SAM3 mask provides a solid core and the matte is
only trusted in the rim band and on the strands outside it. Finally the edge colour is
decontaminated (the model's alpha keeps background colour in semi-transparent pixels, which looks
like a halo once composited).
"""
import os
import time

import numpy as np
from PIL import Image

MODEL = os.path.expanduser("~/.rembg/models/birefnet-portrait/birefnet-portrait.onnx")
_session = None


def _load():
    global _session
    if _session is None:
        import onnxruntime as ort
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        opts.intra_op_num_threads = 0
        _session = ort.InferenceSession(MODEL, sess_options=opts,
                                         providers=["CPUExecutionProvider"])
    return _session


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


def raw_alpha(img: Image.Image, size: int = 1024) -> np.ndarray:
    """The model's own alpha for an RGB image, at the image's own resolution."""
    s = _load()
    w, h = img.size
    small = img.convert("RGB").resize((size, size), Image.BILINEAR)
    x = np.asarray(small, dtype=np.float32) / 255.0
    x = (x - 0.5) / 1.0
    x = x.transpose(2, 0, 1)[None, ...]
    name = s.get_inputs()[0].name
    out = s.run(None, {name: x})[0]
    a = np.asarray(out, dtype=np.float32)
    a = a.reshape(a.shape[-2], a.shape[-1]) if a.ndim == 4 else a.reshape(a.shape[-2:])
    a = _sigmoid(a)
    a = (a - a.min()) / max(a.max() - a.min(), 1e-6)
    return np.asarray(Image.fromarray((a * 255).astype(np.uint8)).resize((w, h), Image.BILINEAR),
                      dtype=np.float32) / 255.0


def _erode(m: np.ndarray, r: int) -> np.ndarray:
    out = m.copy()
    for _ in range(r):
        e = out.copy()
        e[1:, :] &= out[:-1, :]
        e[:-1, :] &= out[1:, :]
        e[:, 1:] &= out[:, :-1]
        e[:, :-1] &= out[:, 1:]
        e[0, :] = False
        e[-1, :] = False
        e[:, 0] = False
        e[:, -1] = False
        out = e
    return out


def _blur(a: np.ndarray, r: int) -> np.ndarray:
    if r < 1:
        return a
    pad = np.pad(a, r, mode="edge")
    acc = np.zeros_like(a)
    n = 0
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            acc += pad[r + dy:r + dy + a.shape[0], r + dx:r + dx + a.shape[1]]
            n += 1
    return acc / n


def compose(img: Image.Image, mask: np.ndarray, core_erode: int = 6,
            rim_scale: bool = True, outside_gain: float = 0.95) -> dict:
    """alpha = 1 inside an eroded core, matte alpha in the rim, strand alpha outside the polygon."""
    t0 = time.time()
    matte = raw_alpha(img)
    t_alpha = time.time() - t0
    mask = mask.astype(bool)
    core = _erode(mask, core_erode)
    rim = mask & ~core

    alpha = np.zeros(mask.shape, np.float32)
    scale = 1.0
    if rim.any():
        hi = float(np.percentile(matte[rim], 90)) if rim_scale else 1.0
        scale = 1.0 / max(hi, 1e-3)
        alpha[rim] = np.clip(matte[rim] * scale, 0.0, 1.0)
    alpha[core] = 1.0
    out = (~mask) & (matte > 0.15)
    alpha[out] = np.clip((matte[out] - 0.15) / 0.70, 0.0, 1.0) * outside_gain
    # a half-pixel of smoothing stops the core edge from looking like a cut-out
    alpha = np.clip(_blur(alpha, 1), 0.0, 1.0)
    alpha[core] = np.maximum(alpha[core], 0.999)

    return {"alpha": alpha, "matte_seconds": round(t_alpha, 2), "rim_scale": round(scale, 3),
            "core_px": int(core.sum()), "rim_px": int(rim.sum()),
            "strand_px": int(out.sum())}


def decontaminate(img: Image.Image, alpha: np.ndarray, bg_radius: int = 9,
                   floor: float = 0.06) -> Image.Image:
    """Remove background colour from partially transparent pixels (f = (c - (1-a)b) / a).

    Without this the strands keep the old background's colour and the result shows a halo the
    moment it is composited over anything else.
    """
    rgb = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
    a = alpha[..., None]
    # background estimate: alpha-weighted local mean of the pixels that are certainly background
    bgmask = (alpha <= floor).astype(np.float32)
    num = np.zeros_like(rgb)
    den = np.zeros(alpha.shape, np.float32)
    r = bg_radius
    pad_rgb = np.pad(rgb, ((r, r), (r, r), (0, 0)), mode="edge")
    pad_w = np.pad(bgmask, r, mode="edge")
    for dy in range(-r, r + 1, 2):
        for dx in range(-r, r + 1, 2):
            num += pad_rgb[r + dy:r + dy + rgb.shape[0], r + dx:r + dx + rgb.shape[1]] * \
                pad_w[r + dy:r + dy + alpha.shape[0], r + dx:r + dx + alpha.shape[1]][..., None]
            den += pad_w[r + dy:r + dy + alpha.shape[0], r + dx:r + dx + alpha.shape[1]]
    bg = num / np.maximum(den, 1e-6)[..., None]
    out = rgb.copy()
    partial = (alpha > floor) & (alpha < 0.995)
    fg = (rgb - (1.0 - a) * bg) / np.maximum(a, 0.08)
    out[partial] = np.clip(fg[partial], 0.0, 1.0)
    return Image.fromarray((out * 255).astype(np.uint8))
