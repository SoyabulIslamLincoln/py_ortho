"""Point-cloud densification: per-image depth maps -> fused 3D dense cloud -> DSM.

This is the Pix4D / COLMAP / OpenMVS order of work (dense cloud first, DSM from the cloud),
unlike the ground-grid height sweep in :mod:`orthomosaic.mvs`, which estimates one height per
DSM cell directly and so can never produce wall points or a true 3D cloud.

1. **Neighbours.** For every image, the source views are chosen from the sparse tie points:
   many shared points and a useful triangulation angle (not too small, not too large).
2. **Depth map** (multi-view plane sweep). For each reference pixel, depth hypotheses
   (uniform in inverse depth, range from the image's own tie points) are back-projected with
   the full camera model, projected into the source views and scored with windowed NCC.
   The score is the mean of the best `top_k` source views, so a view that is occluded at that
   pixel does not veto it. Coarse-to-fine: full sweep at 1/4 resolution, then local refinements at 1/2 and full.
   Weakly textured pixels (reference window std below `min_texture`) get no depth rather than a
   guess.
3. **Geometric consistency + fusion.** A depth is kept only when at least `min_views - 1`
   neighbour depth maps agree: projecting into the neighbour, reading its depth and projecting
   back must land within `consistency_px` pixels and `consistency_depth` relative depth.
   Agreeing measurements are averaged into one 3D point and marked as used, so every surface
   sample is emitted once (COLMAP-style fusion). Depths are camera-axis depths throughout.
4. **DSM.** The fused cloud is rasterised to the DSM grid. Per cell, only the *top layer*
   (points within `layer_gap` of the highest point) is used and its median taken, so roof points
   are never averaged with wall or ground points below them. Cells inside the footprint of a
   measured point but without a point of their own take the lower median of their measured
   neighbours (no averaging across a height step).

Pure numpy; images are matched at `max_image_dim` (long side), i.e. about Pix4D's default
"1/2 image scale" for 20 MP cameras.
"""
from __future__ import annotations

import logging
import math
import os
import shutil
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Optional

import numpy as np

from . import _dense
try:
    from . import _fast
except ImportError:                       # source tree built before _fast existed
    _fast = None
from .imageio import load_rgb
from .masks import load_mask
from .mvs import DenseResult, _near
from .render import ImageCache

log = logging.getLogger(__name__)


@dataclass
class DepthOptions:
    max_image_dim: int = 1600         # long side of the matching images (px)
    neighbors: int = 4                # source views per depth map
    num_depths: int = 64              # hypotheses of the full sweep at 1/4 resolution
    refine_steps: int = 3             # +- steps of each refinement (half the previous spacing)
    window: int = 5                   # NCC window radius (11x11)
    top_k: int = 2                    # score = mean NCC of the best k source views
    min_ncc: float = 0.3              # photometric acceptance (geometric consistency is the main filter)
    min_texture: float = 0.5          # reject only flat windows (grey std below this): on real flights
                                      # low-contrast pixels are mostly correct (55-81 %), so geometric
                                      # consistency, not a texture threshold, decides
    min_views: int = 0                # images that must agree on a point; 0 = auto (3, or 2 for low overlap)
    consistency_px: float = 0.0       # forward-backward tolerance (px); 0 = auto from the SfM residual
    consistency_depth: float = 0.0    # relative depth tolerance; 0 = auto per image pair (baseline, depth)
    patchmatch: bool = False          # repair inconsistent pixels with slanted-plane PatchMatch (C, slow on CPU)
    pm_iters: int = 2
    pm_window: int = 4                # PatchMatch window radius (px), sampled every `pm_step`
    pm_step: int = 2
    pm_geo_weight: float = 0.5        # weight of the multi-view geometric term in the PatchMatch cost
    layer_gap: float = 1.0            # metres: points this far below the cell top are another layer
    keep_depthmaps: bool = False      # keep <out>/depthmaps/*.npy after fusion (deleted even on errors)
    point_stride: int = 2             # fuse every n-th reference pixel per axis (Pix4D "optimal" density)
    cache_mb: int = 1200              # decoded-image cache for matching (8 GB machines: keep ~1.2 GB)
    workers: int = 0


@dataclass
class DenseCloud:
    xyz: np.ndarray        # (n, 3) float64, local frame
    rgb: np.ndarray        # (n, 3) uint8
    views: np.ndarray      # (n,) uint8: images that agree on the point
    cam: np.ndarray        # (n,) int32: reference camera (index into rec.used) -> viewing direction
    spacing: float         # typical ground spacing of the points (m)


# ------------------------------------------------------------------ small image helpers
def _box(a: np.ndarray, r: int) -> np.ndarray:
    """Mean over a (2r+1)^2 window, zero padded (divide by a boxed mask for partial windows).
    float32 is enough because the images are normalised to roughly [-2, 2] (see `_norm`)."""
    if _fast is not None and a.ndim == 2 and a.dtype == np.float32:
        return _fast.box_zero_mean(np.ascontiguousarray(a), int(r))      # same float32 sums, compiled
    k = 2 * r + 1
    c = np.cumsum(np.pad(a.astype(np.float32, copy=False), ((r + 1, r), (0, 0))), 0)
    a1 = c[k:] - c[:-k]
    c = np.cumsum(np.pad(a1, ((0, 0), (r + 1, r))), 1)
    return (c[:, k:] - c[:, :-k]) * np.float32(1.0 / (k * k))


def _norm(grey: np.ndarray) -> np.ndarray:
    return ((grey - 128.0) / 64.0).astype(np.float32)


def _bilinear(img: np.ndarray, u: np.ndarray, v: np.ndarray):
    """Sample a (h, w) float32 image at pixel-centre coordinates. Returns (values, valid)."""
    h, w = img.shape
    ok = (u >= 0) & (v >= 0) & (u <= w - 1.001) & (v <= h - 1.001)
    u = np.where(ok, u, 0).astype(np.float32)
    v = np.where(ok, v, 0).astype(np.float32)
    x0 = u.astype(np.int32)
    y0 = v.astype(np.int32)
    fx, fy = u - x0, v - y0
    flat = img.ravel()
    i = y0 * w + x0
    a, b = flat[i], flat[i + 1]
    c, d = flat[i + w], flat[i + w + 1]
    val = (a * (1 - fx) + b * fx) * (1 - fy) + (c * (1 - fx) + d * fx) * fy
    return val, ok


def _half(g: np.ndarray) -> np.ndarray:
    h, w = g.shape[0] // 2 * 2, g.shape[1] // 2 * 2
    g = g[:h, :w]
    return ((g[0::2, 0::2] + g[1::2, 0::2] + g[0::2, 1::2] + g[1::2, 1::2]) * 0.25).astype(np.float32)


class _Cam:
    """Pinhole + 3-term radial model (pyOrthomosaic convention: pixel centres at integers)."""

    def __init__(self, R, C, f, k1, k2, cx, cy, w, h, k3=0.0):
        self.R, self.C = np.asarray(R, np.float64), np.asarray(C, np.float64)
        self.f, self.k1, self.k2, self.k3, self.cx, self.cy, self.w, self.h = f, k1, k2, k3, cx, cy, w, h

    def scaled(self, s: float, w: int, h: int) -> "_Cam":
        return _Cam(self.R, self.C, self.f * s, self.k1, self.k2,
                    (self.cx + 0.5) * s - 0.5, (self.cy + 0.5) * s - 0.5, w, h, self.k3)

    def rays(self, u, v):
        """Undistorted normalised camera rays (x, y, 1) for pixels (u, v)."""
        nx, ny = (u - self.cx) / self.f, (v - self.cy) / self.f
        x, y = nx.copy(), ny.copy()
        for _ in range(8):
            r2 = x * x + y * y
            d = 1 + r2 * (self.k1 + r2 * (self.k2 + r2 * self.k3))
            x, y = nx / d, ny / d
        return x, y

    def project_cam(self, xc0, xc1, xc2):
        """Camera-frame coordinates -> pixels (u, v) (NaN-free; check z > 0 separately)."""
        z = np.maximum(xc2, 1e-6)
        x, y = xc0 / z, xc1 / z
        r2 = x * x + y * y
        d = self.f * (1 + r2 * (self.k1 + r2 * (self.k2 + r2 * self.k3)))
        return d * x + self.cx, d * y + self.cy


def _exact(*cams) -> bool:
    """The compiled camera kernels (``_fast``) reproduce _Cam's float64 NumPy arithmetic; use them
    only when every parameter is a float64 scalar (NumPy would compute other types differently)."""
    if _fast is None:
        return False
    return all(isinstance(v, float) for c in cams for v in (c.f, c.k1, c.k2, c.k3, c.cx, c.cy))


def _project_neighbour(X, cj, Dj, cr, uu, vv, px_tol, rel_tol):
    """One neighbour of the fusion / consistency check (compiled, same arithmetic as the NumPy
    code it replaces; the 3x3 matrix products stay in NumPy): returns (ui, vi, ok, Xj)."""
    xc = (X - cj.C) @ cj.R.T
    ui, vi, dj, M = _fast.neighbour_sample(xc, Dj, cj.f, cj.k1, cj.k2, cj.k3, cj.cx, cj.cy, cj.w, cj.h)
    Xj = cj.C + (M @ cj.R)
    xk = (Xj - cr.C) @ cr.R.T
    ok = _fast.agreement(xk, xc, dj, uu, vv, cr.f, cr.k1, cr.k2, cr.k3, cr.cx, cr.cy, float(px_tol), float(rel_tol))
    return ui, vi, ok, Xj


# ------------------------------------------------------------------ neighbours / depth ranges
def _neighbours(rec, n_nb: int):
    """Source views per image from shared tie points and triangulation angle."""
    N = len(rec.used)
    shared = np.zeros((N, N), np.int32)
    order = np.argsort(rec.obs_pt, kind="stable")
    pts, cams = rec.obs_pt[order], rec.obs_cam[order]
    bounds = np.flatnonzero(np.diff(pts)) + 1
    for grp in np.split(cams, bounds):
        if len(grp) > 1:
            g = np.unique(grp)
            shared[np.ix_(g, g)] += 1
    np.fill_diagonal(shared, 0)
    depth = np.array([np.median((rec.R[k] @ (rec.X[rec.obs_pt[rec.obs_cam == k]] - rec.C[k]).T)[2])
                      if np.any(rec.obs_cam == k) else 1.0 for k in range(N)])
    out = []
    for k in range(N):
        base = np.linalg.norm(rec.C - rec.C[k], axis=1)
        ang = np.degrees(base / max(depth[k], 1e-3))
        w = np.clip(ang / 8.0, 0, 1) ** 2 * np.where(ang > 45, 0.2, 1.0)   # prefer >= 8 deg triangulation angle
        score = shared[k] * w
        cand = [j for j in np.argsort(-score) if score[j] > 0 and shared[k, j] >= 15][:n_nb]
        out.append(cand)
    return out


def _depth_range(rec, k):
    """Depth search range of image k from *all* tie points that project into its view (not only
    those it observed), always reaching the lowest ground of the block.

    Using only an image's own tie points fails on tall buildings flown low: a photo that mostly
    sees a roof 30 m below the camera then never searches the ground 55 m below it."""
    it = rec.intr[rec.cam_group[k]]
    xc = (rec.X - rec.C[k]) @ rec.R[k].T
    z = xc[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        u = it.f * xc[:, 0] / z + it.cx
        v = it.f * xc[:, 1] / z + it.cy
    inside = (z > 0) & (u >= 0) & (v >= 0) & (u < it.width) & (v < it.height)
    d = z[inside]
    if len(d) < 10:
        return None
    lo, hi = np.percentile(d, [1, 99])
    # the lowest ground anywhere in the block, seen along this camera's optical axis
    zlow = float(np.percentile(rec.X[:, 2], 0.5))
    axis_down = max(-float(rec.R[k][2, 2]), 0.2)        # cos of the view axis from nadir
    hi = max(hi, (float(rec.C[k][2]) - zlow) / axis_down)
    span = hi - lo
    return max(0.3 * lo, lo - 0.2 * span - 1.0), hi + 0.2 * span + 2.0


def _depth_band(rec, k, cam, shape, rng, block: int = 32, margin: float = 1.0):
    """Per-pixel inverse-depth search band (lo, hi maps at `shape`) from the tie points image k
    itself observes (OpenMVS/Pix4D seed depth maps from the sparse points the same way).

    A uniform patterned roof otherwise matches a false repeat of its pattern a few metres above
    or below (neighbouring photos agree on the same false depth, so the consistency check passes
    it). Per block of `block` px: depth range of the tie points there, widened to the 3x3 block
    neighbourhood (objects straddling blocks) and by `margin` m; blocks without tie points keep the
    image-wide range `rng`."""
    H, W = shape
    m = rec.obs_cam == k
    lo_g, hi_g = rng
    inv_lo = np.full((H, W), 1.0 / hi_g, np.float32)
    inv_hi = np.full((H, W), 1.0 / lo_g, np.float32)
    if m.sum() < 50:
        return inv_lo, inv_hi
    xc = (rec.X[rec.obs_pt[m]] - cam.C) @ cam.R.T
    ok = xc[:, 2] > 0
    u, v = cam.project_cam(xc[ok, 0], xc[ok, 1], xc[ok, 2])
    z = xc[ok, 2]
    gh, gw = -(-H // block), -(-W // block)
    bi = (np.clip(v, 0, H - 1) // block).astype(int) * gw + (np.clip(u, 0, W - 1) // block).astype(int)
    inb = (u >= 0) & (v >= 0) & (u < W) & (v < H)
    bi, z = bi[inb], z[inb]
    zlo = np.full(gh * gw, np.inf)
    zhi = np.full(gh * gw, -np.inf)
    cnt = np.bincount(bi, minlength=gh * gw)
    np.minimum.at(zlo, bi, z)
    np.maximum.at(zhi, bi, z)
    ok_b = (cnt >= 5).reshape(gh, gw)
    zlo, zhi = zlo.reshape(gh, gw), zhi.reshape(gh, gw)
    pl = np.pad(np.where(ok_b, zlo, np.inf), 1, constant_values=np.inf)
    ph = np.pad(np.where(ok_b, zhi, -np.inf), 1, constant_values=-np.inf)
    lo3 = np.min([pl[1 + dy:gh + 1 + dy, 1 + dx:gw + 1 + dx] for dy in (-1, 0, 1) for dx in (-1, 0, 1)], 0)
    hi3 = np.max([ph[1 + dy:gh + 1 + dy, 1 + dx:gw + 1 + dx] for dy in (-1, 0, 1) for dx in (-1, 0, 1)], 0)
    has = ok_b & np.isfinite(lo3) & np.isfinite(hi3)
    pad = margin + 0.05 * (hi3 - lo3)
    lo_b = np.where(has, np.clip(lo3 - pad, lo_g, hi_g), lo_g)
    hi_b = np.where(has, np.clip(hi3 + pad, lo_g, hi_g), hi_g)
    lo_px = np.repeat(np.repeat(lo_b, block, 0), block, 1)[:H, :W]
    hi_px = np.repeat(np.repeat(hi_b, block, 0), block, 1)[:H, :W]
    return (1.0 / hi_px).astype(np.float32), (1.0 / np.maximum(lo_px, 1e-6)).astype(np.float32)


def _num_depths(rec, k, nbrs_k, rng, f_coarse, base: int) -> int:
    """Coarse-sweep hypotheses: one per ~1 px of disparity at the coarse level for the median
    baseline, so a wider depth range keeps the same depth resolution (between base and 3x base)."""
    if not nbrs_k:
        return base
    B = float(np.median([np.linalg.norm(rec.C[k] - rec.C[j]) for j in nbrs_k]))
    disp = f_coarse * B * (1.0 / rng[0] - 1.0 / rng[1])
    return int(np.clip(np.ceil(disp), base, 3 * base))


# ------------------------------------------------------------------ depth map of one image
def _rays_world(shape, cam_r):
    """(3, H, W) float32 world direction of every reference pixel (camera-axis depth 1)."""
    H, W = shape
    if _exact(cam_r):
        x, y = _fast.rays_grid(H, W, cam_r.cx, cam_r.cy, cam_r.f, cam_r.k1, cam_r.k2, cam_r.k3)
    else:
        vv, uu = np.mgrid[0:H, 0:W].astype(np.float64)
        x, y = cam_r.rays(uu, vv)
    ray_c = np.stack([x, y, np.ones_like(x)])
    return np.ascontiguousarray(np.einsum("ji,jhw->ihw", cam_r.R, ray_c), np.float32)


class _Hypotheses:
    """Inverse-depth hypothesis maps generated one at a time (``make(i)``) instead of a list
    holding all of them (up to 3 x 64 coarse maps per image and worker)."""

    def __init__(self, make, n):
        self._make, self._n = make, n

    def __len__(self):
        return self._n

    def __getitem__(self, i):
        if not -self._n <= i < self._n:
            raise IndexError(i)
        return self._make(i % self._n)


def _sweep(ref, cam_r, srcs, cams_s, inv_list, opt, ray=None):
    """Winner-take-all over inverse-depth hypotheses `inv_list` (per-pixel maps, uniformly
    spaced), refined to sub-step precision with a parabola through the best score and its two
    neighbours. The warp + NCC of every source and the top-k / winner bookkeeping run in one C
    kernel per hypothesis (``_dense.sweep_hypothesis``, bit-identical to ``_dense.ncc_warp`` per
    source followed by ``_dense.combine``) that releases the GIL; memory does not grow with the
    number of hypotheses."""
    r = opt.window
    H, W = ref.shape
    if ray is None:
        ray = _rays_world(ref.shape, cam_r)
    mu = np.ascontiguousarray(_box(ref, r), np.float32)
    sd = np.ascontiguousarray(np.sqrt(np.maximum(_box(ref * ref, r) - mu * mu, 1e-6)), np.float32)
    geo = [(np.ascontiguousarray(cs.R, np.float64), np.ascontiguousarray(cs.R @ (cam_r.C - cs.C), np.float64))
           for cs in cams_s]
    best = np.full((H, W), -2.0, np.float32)
    s_prev_best = np.full((H, W), -2.0, np.float32)
    s_next_best = np.full((H, W), -2.0, np.float32)
    prev = np.full((H, W), -2.0, np.float32)
    idx = np.full((H, W), -1, np.int32)
    if 1 <= len(srcs) <= 16 and hasattr(_dense, "sweep_hypothesis"):
        G = np.array([np.r_[Rs.ravel(), b, [cs.f, cs.k1, cs.k2, cs.k3, cs.cx, cs.cy]]
                      for cs, (Rs, b) in zip(cams_s, geo)], np.float64)
        # Amortise camera rotations across long coarse sweeps without growing the
        # cache beyond 16 MiB per worker. Fine levels retain the streaming path.
        rotated = (_dense.sweep_rays(ray, G)
                   if len(inv_list) >= 16 and len(srcs) * 3 * H * W * 8 <= 16 * 1024 * 1024 else None)
        for i in range(len(inv_list)):
            inv = np.ascontiguousarray(inv_list[i], np.float32)
            _dense.sweep_hypothesis(ref, mu, sd, srcs, ray, G, inv, r, opt.top_k, i, best, prev, s_prev_best,
                                    s_next_best, idx, rotated)
    else:
        S = np.empty((len(srcs), H, W), np.float32)
        J = np.empty((H, W), np.float32)
        V = np.empty((H, W), np.float32)
        hs = np.empty((4, H, W), np.float32)
        for i in range(len(inv_list)):
            inv = np.ascontiguousarray(inv_list[i], np.float32)
            for j, (src, cs, (Rs, b)) in enumerate(zip(srcs, cams_s, geo)):
                _dense.ncc_warp(ref, mu, sd, src, ray, Rs, b, cs.f, cs.k1, cs.k2, cs.k3, cs.cx, cs.cy, inv, r, S[j],
                                J, V, hs)
            _dense.combine(S, opt.top_k, i, best, prev, s_prev_best, s_next_best, idx)
    n = len(inv_list)
    inv0 = inv_list[0]
    dinv = inv_list[1] - inv_list[0] if n > 1 else np.zeros_like(inv0)
    den = s_prev_best - 2 * best + s_next_best
    interior = (idx > 0) & (idx < n - 1) & (den < -1e-6)
    off = np.where(interior, 0.5 * (s_prev_best - s_next_best) / np.where(interior, den, -1.0), 0.0)
    inv = inv0 + (idx + np.clip(off, -0.5, 0.5)) * dinv
    return (1.0 / np.maximum(inv, 1e-9)).astype(np.float32), best, sd


def noise_sigma(grey255: np.ndarray) -> float:
    """Image noise (grey levels) from the Laplacian residual (Immerkaer 1996)."""
    g = grey255.astype(np.float32)
    lap = (g[:-2, :-2] - 2 * g[:-2, 1:-1] + g[:-2, 2:] - 2 * g[1:-1, :-2] + 4 * g[1:-1, 1:-1]
           - 2 * g[1:-1, 2:] + g[2:, :-2] - 2 * g[2:, 1:-1] + g[2:, 2:])
    lap = lap[np.isfinite(lap)]
    return float(np.sqrt(np.pi / 2) * np.mean(np.abs(lap)) / 6) if lap.size else 1.0


def depth_map(k, nbrs, rec, views, opt: DepthOptions):
    """Depth (camera-axis, metres) and NCC score for reference image `k` at matching scale."""
    rng = _depth_range(rec, k)
    if rng is None or not nbrs:
        return None
    ref, _, cam_r = views(k)
    ref_mask = ~np.isfinite(ref)
    ref = np.ascontiguousarray(np.where(ref_mask, 0.0, ref), np.float32)
    src = [views(j) for j in nbrs]
    srcs = [s[0] for s in src]
    cams_s = [s[2] for s in src]

    # pyramid: level 0 = matching resolution, level L = 1/2^L
    L = 2
    pyr_r, pyr_c, pyr_s, pyr_cs = [ref], [cam_r], [srcs], [cams_s]
    for _ in range(L):
        r_ = np.ascontiguousarray(_half(pyr_r[-1]))
        pyr_r.append(r_)
        pyr_c.append(pyr_c[-1].scaled(0.5, r_.shape[1], r_.shape[0]))
        s_ = [np.ascontiguousarray(_half(x)) for x in pyr_s[-1]]
        pyr_s.append(s_)
        pyr_cs.append([c.scaled(0.5, x.shape[1], x.shape[0]) for c, x in zip(pyr_cs[-1], s_)])

    # pixel rays once at full resolution; coarser levels by 2x2 averaging (exact for the pinhole
    # part: every ray has camera-axis component 1; distortion makes a negligible difference)
    rays = [_rays_world(ref.shape, cam_r)]
    for lvl in range(1, L + 1):
        h, w = pyr_r[lvl].shape
        r0 = rays[-1][:, :2 * h, :2 * w]
        rays.append(np.ascontiguousarray(0.25 * (r0[:, 0::2, 0::2] + r0[:, 1::2, 0::2] + r0[:, 0::2, 1::2]
                                                 + r0[:, 1::2, 1::2]), np.float32))

    # full sweep at the coarsest level, uniform in inverse depth
    nd = _num_depths(rec, k, nbrs, rng, pyr_c[L].f, opt.num_depths)
    shape = pyr_r[L].shape
    # per-pixel band seeded from this image's tie points: the same number of hypotheses spread
    # over a much narrower range where the surface is known (finer steps, no pattern aliasing)
    inv_lo, inv_hi = _depth_band(rec, k, pyr_c[L], shape, rng, block=max(8, shape[1] // 16))
    step = (inv_hi - inv_lo) / max(nd - 1, 1)
    d, _, _ = _sweep(pyr_r[L], pyr_c[L], pyr_s[L], pyr_cs[L],
                     _Hypotheses((lambda i: _fast.hyp_linear(inv_lo, step, i)) if _fast is not None else
                                 (lambda i: (inv_lo + i * step).astype(np.float32)), nd), opt, rays[L])
    dstep = step / 2
    # refine: at each finer level search +-refine_steps around the upsampled depth, half the spacing
    for lvl in range(L - 1, -1, -1):
        H, W = pyr_r[lvl].shape
        up = np.repeat(np.repeat(d, 2, 0), 2, 1)[:H, :W]
        up = np.pad(up, ((0, H - up.shape[0]), (0, W - up.shape[1])), mode="edge")
        dstep = np.repeat(np.repeat(dstep, 2, 0), 2, 1)[:H, :W]
        dstep = np.pad(dstep, ((0, H - dstep.shape[0]), (0, W - dstep.shape[1])), mode="edge")
        iu, rs_ = np.ascontiguousarray(1.0 / up), opt.refine_steps
        if _fast is not None and iu.dtype == np.float32 and dstep.dtype == np.float32:
            hyp = _Hypotheses(lambda q, iu=iu, ds=np.ascontiguousarray(dstep): _fast.hyp_refine(iu, ds, q - rs_),
                              2 * rs_ + 1)
        else:
            hyp = _Hypotheses(lambda q, iu=iu, ds=dstep: np.maximum(iu + (q - rs_) * ds, 1e-6).astype(np.float32),
                              2 * rs_ + 1)
        d, score, sd = _sweep(pyr_r[lvl], pyr_c[lvl], pyr_s[lvl], pyr_cs[lvl], hyp, opt, rays[lvl])
        dstep /= 2
    depth = d
    sd = sd * 64.0                                   # back to 0-255 grey units
    depth[ref_mask | (depth < rng[0]) | (depth > rng[1])] = np.nan
    # NCC / texture are applied after the (optional) PatchMatch repair; here only the raw map
    low_texture = sd < opt.min_texture
    return depth, score, low_texture


# ------------------------------------------------------------------ main entry
def _consistency(k, D, nbrs_k, dm, cams, px_tol, rel_tol):
    """Number of neighbour depth maps that agree with each pixel of D (forward-backward
    reprojection within px_tol and relative depth within rel_tol[j])."""
    cr = cams[k]
    cnt = np.zeros(D.shape, np.int16)
    vv, uu = (np.ascontiguousarray(a) for a in np.nonzero(np.isfinite(D)))
    if len(vv) == 0:
        return cnt
    fast = _exact(cr, *(cams[j] for j in nbrs_k if j in dm))
    if fast:
        X = cr.C + (_fast.ref_points(np.ascontiguousarray(D, np.float32), vv, uu, cr.cx, cr.cy, cr.f, cr.k1, cr.k2,
                                     cr.k3) @ cr.R)
    else:
        d = D[vv, uu].astype(np.float64)
        x, y = cr.rays(uu.astype(np.float64), vv.astype(np.float64))
        X = cr.C + (np.stack([x * d, y * d, d], 1) @ cr.R)
    c = np.zeros(len(vv), np.int16)
    for j in nbrs_k:
        if j not in dm:
            continue
        cj = cams[j]
        if fast:
            c += _project_neighbour(X, cj, np.asarray(dm[j]), cr, uu, vv, px_tol, rel_tol[j])[2]
            continue
        xc = (X - cj.C) @ cj.R.T
        u, v = cj.project_cam(xc[:, 0], xc[:, 1], xc[:, 2])
        ui, vi = np.rint(u).astype(np.int64), np.rint(v).astype(np.int64)
        ins = (xc[:, 2] > 0) & (ui >= 0) & (vi >= 0) & (ui < cj.w) & (vi < cj.h)
        dj = np.full(len(vv), np.nan)
        dj[ins] = dm[j][vi[ins], ui[ins]]
        xj, yj = cj.rays(ui.astype(np.float64), vi.astype(np.float64))
        Xj = cj.C + (np.stack([xj * dj, yj * dj, dj], 1) @ cj.R)
        xk = (Xj - cr.C) @ cr.R.T
        ub, vb = cr.project_cam(xk[:, 0], xk[:, 1], xk[:, 2])
        ok = np.isfinite(dj) & (np.hypot(ub - uu, vb - vv) <= px_tol) & (np.abs(dj - xc[:, 2]) <= rel_tol[j] * dj)
        c += ok
    cnt[vv, uu] = c
    return cnt


def _init_normals(D, ray):
    """World-frame surface normals of a depth map (local plane from 2-px central differences),
    oriented towards the camera; vertical where undefined."""
    P = ray * np.where(np.isfinite(D), D, np.nan)[None]           # point minus camera centre
    tu = np.full_like(P, np.nan)
    tv = np.full_like(P, np.nan)
    tu[:, :, 2:-2] = P[:, :, 4:] - P[:, :, :-4]
    tv[:, 2:-2, :] = P[:, 4:, :] - P[:, :-4, :]
    n = np.cross(tu, tv, axis=0)
    nn = np.linalg.norm(n, axis=0)
    bad = ~np.isfinite(nn) | (nn < 1e-12)
    n = n / np.where(bad, 1, nn)[None]
    n[:, bad] = np.array([0.0, 0.0, 1.0])[:, None]
    flip = np.sum(n * ray, 0) > 0
    n[:, flip] *= -1
    return np.ascontiguousarray(n, np.float32)


def densify(ar, rec, gains: dict, biases: Optional[dict], out_dir: str, opt: DepthOptions) -> DenseCloud:
    """Depth maps -> fused dense cloud. The per-image depth maps live in <out_dir>/depthmaps only
    while this runs: the folder is cleared first and removed afterwards, also when the run fails
    or is interrupted (unless ``opt.keep_depthmaps``)."""
    wd = os.path.join(out_dir, "depthmaps")
    shutil.rmtree(wd, ignore_errors=True)               # leftovers of an interrupted run
    os.makedirs(wd, exist_ok=True)
    try:
        return _densify(ar, rec, gains, biases, wd, opt)
    finally:
        if not opt.keep_depthmaps:
            shutil.rmtree(wd, ignore_errors=True)


def _densify(ar, rec, gains, biases, wd, opt) -> DenseCloud:
    frames = ar.frames
    N = len(rec.used)
    workers = opt.workers or ar.workers or os.cpu_count()
    t0 = time.time()

    scales, cams = {}, {}
    for k, i in enumerate(rec.used):
        fr = frames[i]
        s = min(1.0, (opt.max_image_dim or max(fr.width, fr.height)) / max(fr.width, fr.height))   # None = full resolution
        w, h = int(round(fr.width * s)), int(round(fr.height * s))
        it = rec.intr[rec.cam_group[k]]
        cams[k] = _Cam(rec.R[k], rec.C[k], it.f, it.k1, it.k2, it.cx, it.cy, fr.width, fr.height, it.k3).scaled(w / fr.width, w, h)
        scales[k] = s

    def load_view(k):
        """(normalised grey float32 with NaN = masked, uint8 RGB, camera) of image k."""
        i = rec.used[k]
        b = None if biases is None else biases.get(i)
        if _fast is not None:                          # one compiled pass, same rounding
            rgb = np.ascontiguousarray(load_rgb(frames[i], scales[k]), np.uint8)
            grey, rgb8 = _fast.depth_view(rgb, gains.get(i), b,
                                          load_mask(frames[i].path, rgb.shape[1], rgb.shape[0]))
            return grey, rgb8, cams[k]
        rgb = load_rgb(frames[i], scales[k]).astype(np.float32)
        g = gains.get(i)
        if g is not None:
            rgb *= np.asarray(g, np.float32)
        if b is not None:
            rgb += np.asarray(b, np.float32)
        grey = np.ascontiguousarray(_norm(rgb.mean(-1)))
        m = load_mask(frames[i].path, rgb.shape[1], rgb.shape[0])
        if m is not None:
            grey[~m] = np.nan                          # masked: never matched, never fused
        rgb8 = np.clip(rgb + 0.5, 0, 255).astype(np.uint8)
        return grey, rgb8, cams[k]

    # LRU by bytes; threads asking for an image that is being decoded wait for that one load
    cache = ImageCache(load_view, lambda v: v[0].nbytes + v[1].nbytes, int(opt.cache_mb) * 1024 * 1024)
    views = cache.get

    nbrs = _neighbours(rec, opt.neighbors)
    med_gsd = float(np.median([np.nanmedian((rec.R[k] @ (rec.X[rec.obs_pt[rec.obs_cam == k]] - rec.C[k]).T)[2])
                               / cams[k].f for k in range(N) if np.any(rec.obs_cam == k)]))
    log.info("Densify: %d depth maps at %.2f image scale (~%.3f m/px), %d neighbours, %d hypotheses",
             N, float(np.median(list(scales.values()))), med_gsd, opt.neighbors, opt.num_depths)

    done = 0
    step = max(1, N // 20)

    # ---- data-driven settings (no per-flight tuning)
    scale_med = float(np.median(list(scales.values())))
    px_tol = opt.consistency_px or float(np.clip(2.0 * rec.rms_px * scale_med, 1.0, 3.0))
    n_good = np.array([len(x) for x in nbrs])
    min_views = opt.min_views or (3 if np.median(n_good) >= 3 else 2)
    dmed = {}
    for k in range(N):
        m = rec.obs_cam == k
        if m.any():
            dmed[k] = float(np.median((rec.R[k] @ (rec.X[rec.obs_pt[m]] - rec.C[k]).T)[2]))

    def rel_tols(k):
        """Relative depth tolerance per neighbour: the depth change that moves the point by
        px_tol pixels between the two views (dz/z = px * z / (f * baseline))."""
        out = {}
        for j in nbrs[k]:
            B = float(np.linalg.norm(rec.C[k] - rec.C[j]))
            out[j] = (opt.consistency_depth or
                      float(np.clip(px_tol * dmed.get(k, 1.0) / max(cams[k].f * B, 1e-9), 0.01, 0.05)))
        return out

    log.info("Densify settings (auto): consistency %.2f px, min views %d, PatchMatch %s",
             px_tol, min_views, "on" if opt.patchmatch else "off")

    def job(k):
        r = depth_map(k, nbrs[k], rec, views, opt)
        if r is not None:
            np.save(os.path.join(wd, f"p{k:05d}.npy"), r[0])
            np.save(os.path.join(wd, f"s{k:05d}.npy"), r[1].astype(np.float16))
            np.save(os.path.join(wd, f"t{k:05d}.npy"), r[2])
        return k, r is not None

    have = set()
    with ThreadPoolExecutor(workers) as ex:
        for k, ok in ex.map(job, range(N)):
            done += 1
            if ok:
                have.add(k)
            if done % step == 0 or done == N:
                log.info("  dense: %d/%d depth maps", done, N)
    log.info("Depth maps done in %.1fs (%d/%d images)", time.time() - t0, len(have), N)

    # ---- pass 2: geometric check against the neighbours + PatchMatch repair + final filter
    t2 = time.time()
    raw = {k: np.load(os.path.join(wd, f"p{k:05d}.npy"), mmap_mode="r") for k in have}
    stats = {"repaired": 0, "active": 0, "pixels": 0}
    slock = threading.Lock()

    def repair(k):
        D = np.array(raw[k], np.float32)
        S = np.load(os.path.join(wd, f"s{k:05d}.npy")).astype(np.float32)
        low = np.load(os.path.join(wd, f"t{k:05d}.npy"))
        tol = rel_tols(k)
        if opt.patchmatch:
            cnt = _consistency(k, D, nbrs[k], raw, cams, px_tol, tol)
            grey, _, cr = views(k)
            active = (np.isfinite(grey) & ((cnt < min_views - 1) | (S < opt.min_ncc) | ~np.isfinite(D))).astype(np.uint8)
            src_ids = [j for j in nbrs[k] if j in raw]
            if active.any() and src_ids:
                ray = _rays_world(D.shape, cr)
                normal = _init_normals(D, ray)
                srcs = [views(j)[0] for j in src_ids]
                par = np.array([np.r_[cams[j].R.ravel(), cams[j].C, cams[j].f, cams[j].k1, cams[j].k2, cams[j].k3,
                                      cams[j].cx, cams[j].cy] for j in src_ids], np.float64)
                sdeps = [np.ascontiguousarray(raw[j], np.float32) for j in src_ids]
                rng = _depth_range(rec, k) or (float(np.nanmin(D)), float(np.nanmax(D)))
                ch = _dense.pm_refine(np.ascontiguousarray(np.where(np.isfinite(grey), grey, np.nan), np.float32),
                                      ray, np.ascontiguousarray(cr.C, np.float64), srcs, sdeps, par, D, normal, S,
                                      active, opt.pm_window, opt.pm_step, opt.top_k, opt.pm_geo_weight,
                                      float(np.median(list(tol.values()))), opt.pm_iters, rng[0], rng[1], k * 2654435761 % 2**32)
                with slock:
                    stats["repaired"] += int(ch)
                    stats["active"] += int(active.sum())
            with slock:
                stats["pixels"] += int(np.isfinite(D).sum())
        D[(S < opt.min_ncc) | low] = np.nan
        np.save(os.path.join(wd, f"{k:05d}.npy"), D)
        return k

    done = 0
    with ThreadPoolExecutor(workers) as ex:
        for _ in ex.map(repair, sorted(have)):
            done += 1
            if done % step == 0 or done == len(have):
                log.info("  repair: %d/%d depth maps", done, len(have))
    del raw
    cache.clear()
    log.info("Depth-map repair done in %.1fs (%d of %d pixels re-estimated by PatchMatch)",
             time.time() - t2, stats["active"], stats["pixels"])

    # ---- geometric consistency + fusion (sequential: pixels fused once are marked as used)
    t1 = time.time()
    log.info("Fusing depth maps (>= %d agreeing images per point)", min_views)
    dm = {k: np.load(os.path.join(wd, f"{k:05d}.npy"), mmap_mode="r") for k in have}
    used = {k: np.zeros(dm[k].shape, bool) for k in have}
    pts, cols, nv, owner = [], [], [], []
    fpool = ThreadPoolExecutor(max(1, min(workers, opt.neighbors or 1)))
    try:
        _fuse(sorted(have), dm, used, cams, nbrs, have, rel_tols, opt, px_tol, min_views, views, cache,
              pts, cols, nv, owner, step, fpool)
    finally:
        fpool.shutdown()
    del dm
    if not pts:
        raise RuntimeError("Densification produced no consistent points: too little overlap or texture "
                           "(try min_views=2 or a larger max_image_dim)")
    cloud = DenseCloud(np.concatenate(pts), np.concatenate(cols), np.concatenate(nv), np.concatenate(owner),
                       med_gsd * max(1, opt.point_stride))
    log.info("Dense cloud: %d points (median %d views/point) fused in %.1fs",
             len(cloud.xyz), int(np.median(cloud.views)), time.time() - t1)
    return cloud


def _fuse(order, dm, used, cams, nbrs, have, rel_tols, opt, px_tol, min_views, views, cache, pts, cols, nv, owner,
          step, fpool):
    """Sequential geometric-consistency fusion (pixels fused once are marked as used). The next
    reference image is decoded in the background while the current one is fused."""
    pre = ThreadPoolExecutor(1)
    try:
        _fuse_loop(order, dm, used, cams, nbrs, have, rel_tols, opt, px_tol, min_views, views, cache, pts, cols, nv,
                   owner, step, fpool, pre)
    finally:
        pre.shutdown()


def _fuse_loop(order, dm, used, cams, nbrs, have, rel_tols, opt, px_tol, min_views, views, cache, pts, cols, nv,
               owner, step, fpool, pre):
    nxt = pre.submit(views, order[0]) if order else None
    for n_done, k in enumerate(order):
        cur, nxt = nxt, (pre.submit(views, order[n_done + 1]) if n_done + 1 < len(order) else None)
        D = np.asarray(dm[k])
        cr = cams[k]
        tol_k = rel_tols(k)
        cand = np.isfinite(D) & ~used[k]
        if opt.point_stride > 1:                         # one point per stride x stride pixels
            sub = np.zeros_like(cand)
            sub[::opt.point_stride, ::opt.point_stride] = True
            cand &= sub
        vv, uu = (np.ascontiguousarray(a) for a in np.nonzero(cand))
        if len(vv) == 0:
            continue
        nb_k = [j for j in nbrs[k] if j in have]
        fast = _exact(cr, *(cams[j] for j in nb_k))
        if fast:
            X = cr.C + (_fast.ref_points(np.ascontiguousarray(D, np.float32), vv, uu, cr.cx, cr.cy, cr.f, cr.k1,
                                         cr.k2, cr.k3) @ cr.R)
        else:
            d = D[vv, uu].astype(np.float64)
            x, y = cr.rays(uu.astype(np.float64), vv.astype(np.float64))
            X = cr.C + (np.stack([x * d, y * d, d], 1) @ cr.R)             # R^T (d * ray)
        acc = X.copy()
        cnt = np.ones(len(vv), np.int32)
        marks = []
        if fast:
            # the neighbours are independent: project them in parallel, then add them up in the
            # original neighbour order (the sums are order dependent)
            res = list(fpool.map(lambda j: _project_neighbour(X, cams[j], np.asarray(dm[j]), cr, uu, vv, px_tol,
                                                              tol_k[j]), nb_k))
            for j, (ui, vi, ok, Xj) in zip(nb_k, res):
                _fast.accumulate(acc, cnt, Xj, ok.view(np.uint8))
                marks.append((j, vi[ok], ui[ok], ok))
            del res
        for j in (() if fast else nbrs[k]):
            if j not in have:
                continue
            cj = cams[j]
            xc = (X - cj.C) @ cj.R.T
            u, v = cj.project_cam(xc[:, 0], xc[:, 1], xc[:, 2])
            ui, vi = np.rint(u).astype(np.int64), np.rint(v).astype(np.int64)
            ins = (xc[:, 2] > 0) & (ui >= 0) & (vi >= 0) & (ui < cj.w) & (vi < cj.h)
            dj = np.full(len(vv), np.nan)
            dj[ins] = dm[j][vi[ins], ui[ins]]
            ok = np.isfinite(dj)
            xj, yj = cj.rays(ui.astype(np.float64), vi.astype(np.float64))
            Xj = cj.C + (np.stack([xj * dj, yj * dj, dj], 1) @ cj.R)
            xk = (Xj - cr.C) @ cr.R.T
            ub, vb = cr.project_cam(xk[:, 0], xk[:, 1], xk[:, 2])
            ok &= np.hypot(ub - uu, vb - vv) <= px_tol
            ok &= np.abs(dj - xc[:, 2]) <= tol_k[j] * dj
            acc[ok] += Xj[ok]
            cnt += ok
            marks.append((j, vi[ok], ui[ok], ok))
        keep = cnt >= min_views
        if not keep.any():
            continue
        st_ = max(1, opt.point_stride)
        for j, vi, ui, ok in marks:
            sel = keep[ok]
            # mark the stride-grid pixel that would emit this surface sample when j is the
            # reference (only those pixels are candidates); otherwise every overlapping photo
            # re-emits the same point
            used[j][(vi[sel] // st_) * st_, (ui[sel] // st_) * st_] = True
        _, rgb, _ = cur.result()
        pts.append((acc[keep] / cnt[keep, None]))
        cols.append(rgb[vv[keep], uu[keep]])
        nv.append(np.minimum(cnt[keep], 255).astype(np.uint8))
        owner.append(np.full(int(keep.sum()), k, np.int32))
        cache.clear()
        if (n_done + 1) % step == 0:
            log.info("  fusion: %d/%d", n_done + 1, len(have))


# ------------------------------------------------------------------ DSM from the cloud
def camera_count(rec, minX: float, maxY: float, W: int, H: int, gsd: float, step: int = 16,
                 Z: Optional[np.ndarray] = None) -> np.ndarray:
    """(H, W) int: number of calibrated cameras whose frame contains each cell (Pix4D's overlap
    definition, not occlusion-aware), evaluated on a coarse grid of `step` cells. Cells sit at the
    height `Z` (H, W) where given and finite, else at the typical ground height."""
    zg = float(np.percentile(rec.X[:, 2], 20)) if len(rec.X) else 0.0
    hs, ws = (H + step - 1) // step, (W + step - 1) // step
    yy, xx = np.mgrid[0:hs, 0:ws]
    z = np.full(xx.size, zg)
    if Z is not None:
        zc = Z[np.minimum(yy.ravel() * step + step // 2, H - 1), np.minimum(xx.ravel() * step + step // 2, W - 1)]
        z = np.where(np.isfinite(zc), zc, zg)
    P = np.stack([minX + (xx.ravel() * step + step / 2) * gsd, maxY - (yy.ravel() * step + step / 2) * gsd, z], 1)
    count = np.zeros(xx.size, np.int32)
    for k in range(len(rec.used)):
        it = rec.intr[rec.cam_group[k]]
        cam = _Cam(rec.R[k], rec.C[k], it.f, it.k1, it.k2, it.cx, it.cy, it.width, it.height, it.k3)
        xc = (P - cam.C) @ cam.R.T
        u, v = cam.project_cam(xc[:, 0], xc[:, 1], xc[:, 2])
        count += (xc[:, 2] > 0) & (u >= 0) & (v >= 0) & (u <= it.width - 1) & (v <= it.height - 1)
    return np.repeat(np.repeat(count.reshape(hs, ws), step, 0), step, 1)[:H, :W]


def camera_coverage(rec, minX: float, maxY: float, W: int, H: int, gsd: float, min_views: int = 2,
                    step: int = 16) -> np.ndarray:
    """(H, W) bool: cells photographed by at least `min_views` calibrated cameras."""
    return camera_count(rec, minX, maxY, W, H, gsd, step) >= min_views


def cloud_footprint(measured: np.ndarray, gsd: float, block_m: float = 0.5) -> np.ndarray:
    """(H, W) bool: area actually reconstructed (ODM crops its DEM and orthophoto to the point
    cloud's convex hull). Blocks of ~`block_m` count when at least half of their cells are measured
    and most of their 3x3 neighbourhood is too, so the thin, poorly seen fringe at the flight edge
    (single oblique views, smeared colours) is left out. The convex hull of those blocks is
    rasterised at full resolution, so the border is a straight line, not a staircase (numpy only)."""
    H, W = measured.shape
    b = max(1, int(round(block_m / gsd)))
    h, w = -(-H // b), -(-W // b)
    P = np.zeros((h * b, w * b), np.float32)
    P[:H, :W] = measured
    occ = P.reshape(h, b, w, b).mean((1, 3)) >= 0.5
    pad = np.pad(occ, 1)
    nb = sum(pad[1 + dy:h + 1 + dy, 1 + dx:w + 1 + dx] for dy in (-1, 0, 1) for dx in (-1, 0, 1))
    occ &= nb >= 6
    ys, xs = np.nonzero(occ)
    if len(xs) < 3:
        return np.ones((H, W), bool)
    # convex hull of the block corners, in full-resolution cell units (Andrew's monotone chain)
    pts = np.unique(np.concatenate([np.stack([(xs + dx) * b, (ys + dy) * b], 1)
                                    for dx in (0, 1) for dy in (0, 1)]), axis=0)

    def half(points):
        out = []
        for p in points:
            while len(out) >= 2 and ((out[-1][0] - out[-2][0]) * (p[1] - out[-2][1])
                                     - (out[-1][1] - out[-2][1]) * (p[0] - out[-2][0])) <= 0:
                out.pop()
            out.append(p)
        return out
    pl = [tuple(p) for p in pts.tolist()]
    hull = np.array(half(pl)[:-1] + half(pl[::-1])[:-1], np.float64)
    # scanline fill: for a convex polygon every row is one span [left, right]
    yc = np.arange(H) + 0.5
    left = np.full(H, np.inf)
    right = np.full(H, -np.inf)
    for (x0, y0), (x1, y1) in zip(hull, np.roll(hull, -1, 0)):
        if y0 == y1:
            continue
        lo, hi = min(y0, y1), max(y0, y1)
        r = (yc >= lo) & (yc <= hi)
        x = x0 + (yc[r] - y0) * (x1 - x0) / (y1 - y0)
        left[r] = np.minimum(left[r], x)
        right[r] = np.maximum(right[r], x)
    xc = np.arange(W) + 0.5
    return (xc[None, :] >= left[:, None]) & (xc[None, :] <= right[:, None])


def radius_steps(spacing: float, steps: int = 3, multiplier: float = 1.0) -> list:
    """ODM's DEM search radii: point spacing * multiplier, growing by sqrt(2) per step."""
    r = [spacing * multiplier]
    for _ in range(max(1, steps) - 1):
        r.append(r[-1] * math.sqrt(2))
    return r


def rasterize(cloud: DenseCloud, minX: float, maxY: float, W: int, H: int, gsd: float,
              layer_gap: float = 1.0, covered: Optional[np.ndarray] = None,
              radii: Optional[list] = None) -> DenseResult:
    """Top-layer DSM from the dense cloud: per cell the median height (mean colour) of the points within
    `layer_gap` of the cell's highest point (roof points are never mixed with wall or ground
    points below them; ``_dense.top_layer``).

    Empty cells are filled radius by radius (ODM's stepped radii, `radius_steps`): a cell within
    the current radius of measured cells takes the *lower median* of its measured neighbours, so
    no value is averaged across a height step. `score` is 1 for cells with points or within the
    first radius (the point footprint), 0.25 for cells filled at a larger radius (reported as
    interpolated), 0 elsewhere."""
    col = np.floor((cloud.xyz[:, 0] - minX) / gsd).astype(np.int64)
    row = np.floor((maxY - cloud.xyz[:, 1]) / gsd).astype(np.int64)
    ok = (col >= 0) & (col < W) & (row >= 0) & (row < H)
    cell = (row * W + col)[ok]
    # work on the occupied cells only (memory ~ points, not ~ raster): C top-layer kernel
    uc, inv = np.unique(cell, return_inverse=True)
    del cell, col, row
    Zc, RGBc, _ = _dense.top_layer(np.ascontiguousarray(inv, np.int64), np.ascontiguousarray(cloud.xyz[ok, 2]),
                                   np.ascontiguousarray(cloud.rgb[ok]), len(uc), float(layer_gap))
    del inv
    Z = np.full((H, W), np.nan, np.float32)
    Z.ravel()[uc] = Zc
    RGB = np.zeros((H, W, 3), np.uint8)
    RGB.reshape(-1, 3)[uc] = RGBc
    del uc, Zc, RGBc
    score = np.isfinite(Z).astype(np.float32)

    grown = 0
    for step, r in enumerate(radii or [cloud.spacing]):
        target = int(math.ceil(r / gsd - 0.5))
        for _ in range(max(0, target - grown)):
            filled = _dense.fill_lower_median(Z.copy(), RGB.copy(), Z, RGB, score,
                                              1.0 if step == 0 else 0.25, 3)
            if filled == 0:
                break
        grown = max(grown, target)
    cov = covered if covered is not None else _near(np.isfinite(Z), 8)
    return DenseResult(Z, score, RGB, cov, minX, maxY, gsd, stepped=(score > 0) & (score < 1))
