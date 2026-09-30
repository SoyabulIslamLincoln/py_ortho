"""ORB-style multi-scale features built on the Cython kernels."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import _core
from .imageio import Frame, load_rgb

PATCH_RADIUS = 15
BORDER = PATCH_RADIUS + 3


def _make_pattern(n_bits: int = 256, seed: int = 0x0B5) -> np.ndarray:
    """Fixed random BRIEF test pairs, Gaussian around the centre, inside the patch disc."""
    rng = np.random.default_rng(seed)
    out = []
    lim = PATCH_RADIUS - 0.5
    while len(out) < n_bits:
        p = np.round(rng.normal(0, PATCH_RADIUS / 2.5, 4))
        if np.hypot(p[0], p[1]) > lim or np.hypot(p[2], p[3]) > lim:
            continue
        if p[0] == p[2] and p[1] == p[3]:
            continue
        out.append(p)
    return np.ascontiguousarray(np.array(out, np.int32))


PATTERN = _make_pattern()


@dataclass
class Features:
    xy: np.ndarray       # (N, 2) float64, full-resolution pixel coordinates
    desc: np.ndarray     # (N, 32) uint8
    color: np.ndarray    # (N, 3) float32 mean RGB around each keypoint

    def __len__(self):
        return len(self.xy)


def _bucket(xs, ys, scores, w, h, budget, grid=8):
    """Keep the strongest corners while spreading them evenly over the image."""
    n = len(xs)
    if n <= budget:
        return np.arange(n)
    cx = np.minimum((xs * grid / w).astype(np.int64), grid - 1)
    cy = np.minimum((ys * grid / h).astype(np.int64), grid - 1)
    cell = cy * grid + cx
    order = np.lexsort((-scores, cell))
    sorted_cell = cell[order]
    starts = np.searchsorted(sorted_cell, np.arange(grid * grid))
    rank = np.arange(n) - starts[sorted_cell]
    per_cell = max(1, budget // (grid * grid))
    first = order[rank < per_cell]
    if len(first) >= budget:
        return first[np.argsort(-scores[first])[:budget]]
    rest = order[rank >= per_cell]
    rest = rest[np.argsort(-scores[rest])[: budget - len(first)]]
    return np.concatenate([first, rest])


def extract(frame: Frame, max_dim: int = 2000, n_features: int = 5000, levels: int = 4,
            scale_factor: float = 1.3, fast_threshold: float = 12.0) -> tuple[Features, float]:
    """Detect + describe. Returns (features, working_scale)."""
    s = min(1.0, max_dim / max(frame.width, frame.height))
    rgb = load_rgb(frame, s)
    gray = _core.rgb_to_gray(rgb)
    h0, w0 = gray.shape

    areas = np.array([scale_factor ** (-2 * l) for l in range(levels)])
    budgets = np.maximum(50, (n_features * areas / areas.sum()).astype(int))

    xy_all, desc_all = [], []
    img = gray
    for lvl in range(levels):
        if lvl > 0:
            nh = int(round(h0 / scale_factor ** lvl))
            nw = int(round(w0 / scale_factor ** lvl))
            if min(nh, nw) < 4 * BORDER:
                break
            img = _core.resize_bilinear(_core.gaussian_blur(img, 0.6), nh, nw)
        lh, lw = img.shape
        thr = fast_threshold
        xs, ys, sc = _core.detect_corners(img, thr, BORDER)
        while len(xs) < budgets[lvl] // 2 and thr > 3:  # low-texture scenes (fields, water)
            thr /= 2
            xs, ys, sc = _core.detect_corners(img, thr, BORDER)
        if len(xs) == 0:
            continue
        keep = _bucket(xs, ys, sc, lw, lh, int(budgets[lvl]))
        xs, ys = np.ascontiguousarray(xs[keep]), np.ascontiguousarray(ys[keep])
        ang = _core.orientations(img, xs, ys, PATCH_RADIUS)
        smooth = _core.gaussian_blur(img, 2.0)
        desc_all.append(_core.brief_describe(smooth, xs, ys, ang, PATTERN))
        sx, sy = w0 / lw, h0 / lh
        xy_all.append(np.stack([(xs + 0.5) * sx - 0.5, (ys + 0.5) * sy - 0.5], 1))

    if not xy_all:
        return Features(np.zeros((0, 2)), np.zeros((0, 32), np.uint8), np.zeros((0, 3), np.float32)), s

    xy = np.concatenate(xy_all).astype(np.float64)
    desc = np.concatenate(desc_all)

    # mean colour in a 5x5 window (used later for exposure compensation)
    xi = np.clip(np.round(xy[:, 0]).astype(int), 2, w0 - 3)
    yi = np.clip(np.round(xy[:, 1]).astype(int), 2, h0 - 3)
    col = np.zeros((len(xy), 3), np.float32)
    for dy in range(-2, 3):
        for dx in range(-2, 3):
            col += rgb[yi + dy, xi + dx]
    col /= 25.0

    xy_full = (xy + 0.5) * np.array([frame.width / w0, frame.height / h0]) - 0.5
    from .masks import load_mask
    m = load_mask(frame.path, frame.width, frame.height)
    if m is not None:                   # no features on masked pixels (sky, background, ...)
        xi = np.clip(np.round(xy_full[:, 0]).astype(int), 0, frame.width - 1)
        yi = np.clip(np.round(xy_full[:, 1]).astype(int), 0, frame.height - 1)
        keep = m[yi, xi]
        xy_full, desc, col = xy_full[keep], desc[keep], col[keep]
    return Features(xy_full, desc, col), s
