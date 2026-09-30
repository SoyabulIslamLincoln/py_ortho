"""Per-image validity masks (ODM conventions): 255 = use the pixel, 0 = ignore it.

Sources, in priority order:
1. a user mask next to the image: ``<image stem>_mask.png`` (same size as the image);
2. ``sky_removal``: the OpenDroneMap SkyRemoval ONNX model, applied like ODM only to images that
   are not nadir (gimbal pitch more than 20 deg from straight down), refined with a guided filter;
3. ``bg_removal``: the U^2-Net ONNX model (ODM's background removal), keeps the foreground object.

The AI masks need ``pip install onnxruntime``; the models are downloaded once from the ODM
releases into ``~/.cache/pyorthomosaic/models``. Generated masks are written next to the images
as ``<stem>_mask.png`` so later runs (and other tools, e.g. ODM itself) reuse them.
Masked pixels produce no features, no depth and no orthophoto colour.
"""
from __future__ import annotations

import logging
import os
import threading
import urllib.request
import zipfile
from typing import Optional

import numpy as np
from PIL import Image

log = logging.getLogger(__name__)

SKY_MODEL = ("skyremoval", "https://github.com/OpenDroneMap/SkyRemoval/releases/download/v1.0.5/model.zip", "v1.0.5")
BG_MODEL = ("bgremoval", "https://github.com/OpenDroneMap/ODM/releases/download/v2.9.0/u2net.zip", "v2.9.0")
_lock = threading.Lock()
_sessions: dict = {}


def mask_path(image_path: str) -> str:
    return os.path.splitext(image_path)[0] + "_mask.png"


def load_mask(image_path: str, width: int, height: int) -> Optional[np.ndarray]:
    """Bool array (True = valid) of the image's mask, or None when it has no mask."""
    p = mask_path(image_path)
    if not os.path.isfile(p):
        return None
    with Image.open(p) as m:
        m = m.convert("L")
        if m.size != (width, height):
            m = m.resize((width, height), Image.NEAREST)
        return np.asarray(m) > 127


def _model(namespace: str, url: str, version: str) -> Optional[str]:
    d = os.path.join(os.path.expanduser("~/.cache/pyorthomosaic/models"), namespace, version.replace(".", "_"))
    f = os.path.join(d, "model.onnx")
    if os.path.isfile(f):
        return f
    os.makedirs(d, exist_ok=True)
    log.info("Downloading AI model %s ...", url)
    try:
        z = os.path.join(d, os.path.basename(url))
        urllib.request.urlretrieve(url, z)
        with zipfile.ZipFile(z) as zf:
            zf.extractall(d)
        os.remove(z)
    except Exception as e:  # noqa: BLE001
        log.warning("Cannot download %s: %s", url, e)
        return None
    if not os.path.isfile(f):
        found = [os.path.join(r, n) for r, _, ns in os.walk(d) for n in ns if n.endswith(".onnx")]
        if not found:
            log.warning("No .onnx model inside %s", url)
            return None
        os.replace(found[0], f)
    return f


def _session(spec):
    with _lock:
        if spec[0] not in _sessions:
            import onnxruntime as ort
            path = _model(*spec)
            if path is None:
                _sessions[spec[0]] = None
            else:
                prov = ["CUDAExecutionProvider"] if "CUDAExecutionProvider" in ort.get_available_providers() else []
                _sessions[spec[0]] = ort.InferenceSession(path, providers=prov + ["CPUExecutionProvider"])
        return _sessions[spec[0]]


def _box(a, r):
    c = np.cumsum(np.pad(a, ((r + 1, r), (0, 0)), mode="edge"), 0)
    a = c[2 * r + 1:] - c[:-2 * r - 1]
    c = np.cumsum(np.pad(a, ((0, 0), (r + 1, r)), mode="edge"), 1)
    return c[:, 2 * r + 1:] - c[:, :-2 * r - 1]


def guided_filter(img, guide, radius, eps):
    """He et al. guided filter (ODM's skyremoval/guidedfilter.py), `img` guides `guide`."""
    n = _box(np.ones_like(img), radius)
    m_i, m_g = _box(img, radius) / n, _box(guide, radius) / n
    a = (_box(img * guide, radius) / n - m_i * m_g) / (_box(img * img, radius) / n - m_i * m_i + eps)
    b = m_g - a * m_i
    return _box(a, radius) / n * img + _box(b, radius) / n


def sky_mask(rgb: np.ndarray) -> Optional[np.ndarray]:
    """uint8 mask, 0 = sky. rgb: (h, w, 3) uint8."""
    sess = _session(SKY_MODEL)
    if sess is None:
        return None
    h, w = rgb.shape[:2]
    img = rgb.astype(np.float32) / 255.0
    small = np.asarray(Image.fromarray(rgb).resize((384, 384), Image.BOX), np.float32) / 255.0
    with _lock:
        out = sess.run(None, {sess.get_inputs()[0].name: small.transpose(2, 0, 1)[None]})
    pred = np.asarray(out)[0][0].transpose(1, 2, 0)[..., 0]
    pred = np.asarray(Image.fromarray(pred.astype(np.float32)).resize((w, h), Image.LANCZOS))
    pred = np.clip(pred, 0, 1)
    refined = np.clip(guided_filter(img[..., 2].astype(np.float64), pred.astype(np.float64), 20, 0.01), 0, 1)
    return np.where((refined * 255).astype(np.uint8) > 127, 0, 255).astype(np.uint8)


def background_mask(rgb: np.ndarray) -> Optional[np.ndarray]:
    """uint8 mask, 255 = foreground object (U^2-Net salient object), 0 = background."""
    sess = _session(BG_MODEL)
    if sess is None:
        return None
    h, w = rgb.shape[:2]
    im = np.asarray(Image.fromarray(rgb).resize((320, 320), Image.BOX), np.float64)
    im = im / im.max()
    im = (im - (0.485, 0.456, 0.406)) / (0.229, 0.224, 0.225)
    with _lock:
        out = sess.run(None, {sess.get_inputs()[0].name: im.transpose(2, 0, 1)[None].astype(np.float32)})
    pred = out[0][:, 0][0]
    pred = (pred - pred.min()) / max(pred.max() - pred.min(), 1e-12)
    pred = np.asarray(Image.fromarray((pred * 255).astype(np.uint8)).resize((w, h), Image.LANCZOS))
    return np.where(pred > 127, 255, 0).astype(np.uint8)


def generate_masks(frames, sky_removal: bool = False, bg_removal: bool = False, workers: int = 4) -> int:
    """Write ``<stem>_mask.png`` for frames that have none. Returns the number written."""
    if not (sky_removal or bg_removal):
        return 0
    try:
        import onnxruntime  # noqa: F401
    except ImportError:
        log.warning("sky/background removal needs `pip install onnxruntime`; no masks generated")
        return 0
    from concurrent.futures import ThreadPoolExecutor
    from .imageio import load_rgb

    todo = []
    for f in frames:
        if os.path.isfile(mask_path(f.path)):
            continue
        if bg_removal:
            todo.append((f, background_mask))
        elif sky_removal and (f.gimbal_pitch is None or abs(f.gimbal_pitch + 90.0) > 20.0):
            todo.append((f, sky_mask))                    # ODM: only non-nadir images get sky masks

    def job(item):
        f, fn = item
        m = fn(load_rgb(f, 1.0))
        if m is not None:
            Image.fromarray(m).save(mask_path(f.path))
            return 1
        return 0

    with ThreadPoolExecutor(max(1, workers)) as ex:
        n = sum(ex.map(job, todo))
    log.info("Masks: %d generated (%s)", n, "background" if bg_removal else "sky")
    return n
