#changed here: new module -- Pix4D-style occlusion-aware true orthophoto.
"""Pix4D-style true orthophoto: occlusion-aware orthorectification of the final DSM.

Pix4D does not render the orthomosaic during the dense matching; it first builds the DSM and
only then orthorectifies, and while doing so it

  * keeps, for every DSM cell, only the views that actually *see* that cell (occlusion
    handling), so a building no longer smears its wall over the ground behind it;
  * favours the most nadir view (view-angle weighting), so the texture is sampled with the
    least off-nadir stretch;
  * feathers across the view seams (distance to the image border).

The colours produced inside :func:`orthomosaic.mvs.dense_reconstruct` already sample the
surface at its own height, but they blend views that are geometrically hidden and weight
oblique views the same as nadir ones.  This module reproduces the Pix4D render stage on an
already computed DSM and can therefore test occlusion against the *whole* surface, not just
the current matching tile.
"""
from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from .imageio import load_rgb
from .mvs import _View
from .render import ImageCache

log = logging.getLogger(__name__)


def _visibility(dsm, minX, maxY, gsd, C, X, Y, Z, steps, tol):
    #changed here: geometric occlusion test (Pix4D "occlusion handling").
    """Visibility mask (True = visible) for surface points (X, Y, Z) seen from camera `C`.

    Marches the ray camera -> point and marks the point hidden whenever the DSM rises above the
    ray at a sample.  `tol` (metres) absorbs DSM noise; `steps` trades accuracy for speed.
    """
    H, W = dsm.shape
    vis = np.ones(np.shape(Z), bool)
    if not (np.isfinite(dsm).any()):
        return vis
    for t in np.linspace(0.03, 0.97, int(max(2, steps))):
        sx = C[0] + t * (X - C[0])
        sy = C[1] + t * (Y - C[1])
        sz = C[2] + t * (Z - C[2])
        col = ((sx - minX) / gsd).astype(np.int64)
        row = ((maxY - sy) / gsd).astype(np.int64)
        inb = (col >= 0) & (col < W) & (row >= 0) & (row < H)
        np.clip(col, 0, W - 1, out=col)
        np.clip(row, 0, H - 1, out=row)
        zz = dsm[row, col]
        vis &= ~(inb & np.isfinite(zz) & (zz > sz + tol))
    return vis


class _Defaults:
    #changed here: fallback option object so true_orthophoto works without an Options3D.
    ortho_tile = 256
    max_views = 6
    view_angle_power = 1.5
    occlusion = True
    occlusion_tol = 0.20
    occlusion_steps = 12
    occlusion_stride = 2
    cache_mb = 1024
    workers = 0


def true_orthophoto(ar, rec, dsm, minX, maxY, gsd, gains=None, biases=None, opt=None):
    #changed here: whole function -- the Pix4D "Orthomosaic" render stage on the final DSM.
    """Render `dsm` as an occlusion-aware true orthophoto.

    `minX`/`maxY` are the world coordinates of the top-left DSM cell *corner*; `dsm` is the
    final surface (NaN = no data).  Returns ``(rgb (H, W, 3) uint8, covered (H, W) bool)`` where
    `covered` is True for cells that at least one visible view could colour.
    """
    opt = opt if opt is not None else _Defaults()
    g = lambda name: getattr(opt, name, getattr(_Defaults, name))
    backend = ar.backend
    frames = ar.frames
    dsm = np.ascontiguousarray(dsm, np.float32)
    H, W = dsm.shape
    covered = np.isfinite(dsm)
    rgb_out = np.zeros((H, W, 3), np.uint8)
    if not covered.any() or len(rec.used) == 0:
        return rgb_out, np.zeros((H, W), bool)

    gains = gains or {}
    biases = biases or {}
    zfill = float(np.nanmedian(dsm))

    def load(k):
        i = rec.used[k]
        fr = frames[i]
        rgb = load_rgb(fr, 1.0).astype(np.float32)
        gg = gains.get(i)
        if gg is not None:
            rgb *= np.asarray(gg, np.float32)
        bb = biases.get(i)
        if bb is not None:
            rgb += np.asarray(bb, np.float32)
        h, w = rgb.shape[:2]
        # 4th channel: feather weight (distance to the image border) -> smooth view seams
        fy = np.minimum(np.arange(h) + 0.5, h - 0.5 - np.arange(h)) / (0.5 * min(h, w))
        fx = np.minimum(np.arange(w) + 0.5, w - 0.5 - np.arange(w)) / (0.5 * min(h, w))
        rgb = np.dstack([rgb, np.clip(np.minimum(fy[:, None], fx[None, :]), 0, 1).astype(np.float32)])
        it = rec.intr[rec.cam_group[k]]
        sx = rgb.shape[1] / fr.width
        return _View([], backend.upload(np.ascontiguousarray(rgb)),
                     np.ascontiguousarray(rec.R[k]), np.ascontiguousarray(rec.C[k]),
                     it.f * sx, it.k1, it.k2, (it.cx + 0.5) * sx - 0.5, (it.cy + 0.5) * sx - 0.5)

    cache = ImageCache(load, lambda v: v.nbytes, int(g("cache_mb")) * 1024 * 1024)

    T = int(g("ortho_tile"))
    max_views = int(g("max_views"))
    va_power = float(g("view_angle_power"))
    use_occ = bool(g("occlusion"))
    occ_tol = float(g("occlusion_tol"))
    occ_steps = int(g("occlusion_steps"))
    occ_stride = max(1, int(g("occlusion_stride")))

    cam_xy = rec.C[:, :2]
    WSUM = np.zeros((H, W), np.float32)

    def process(tile):
        tx, ty = tile
        tw, th = min(T, W - tx), min(T, H - ty)
        sc = occ_stride
        sub = covered[ty:ty + th, tx:tx + tw]
        if not sub.any():
            return None
        X0 = minX + tx * gsd
        Y0 = maxY - ty * gsd
        Ztile = np.where(sub, dsm[ty:ty + th, tx:tx + tw], zfill).astype(np.float32)

        # candidate views: those that see the whole tile over its height range, most nadir first
        zz = Ztile[sub]
        zlo, zhi = float(zz.min()), float(zz.max())
        cxw = X0 + tw * gsd / 2.0
        cyw = Y0 - th * gsd / 2.0
        corners = np.array([[X0, Y0], [X0 + tw * gsd, Y0], [X0, Y0 - th * gsd], [X0 + tw * gsd, Y0 - th * gsd]])
        pts = np.concatenate([np.column_stack([corners, np.full(4, z)]) for z in (zlo, zhi)])
        dist = np.hypot(cam_xy[:, 0] - cxw, cam_xy[:, 1] - cyw)
        cand = []
        for k in np.argsort(dist):
            it = rec.intr[rec.cam_group[k]]
            xc = (pts - rec.C[k]) @ rec.R[k].T
            if np.any(xc[:, 2] <= 0):
                continue
            u = it.f * xc[:, 0] / xc[:, 2] + it.cx
            v = it.f * xc[:, 1] / xc[:, 2] + it.cy
            if ((u >= 0) & (u <= it.width - 1) & (v >= 0) & (v <= it.height - 1)).all():
                cand.append(int(k))
            if len(cand) >= max_views:
                break
        if not cand:
            return None
        views = [cache.get(k) for k in cand]

        # every covered cell is coloured; occlusion is ray-marched on a coarser sub-grid
        rr, cc = np.nonzero(sub)
        wx = X0 + (cc + 0.5) * gsd
        wy = Y0 - (rr + 0.5) * gsd
        wz = dsm[ty + rr, tx + cc].astype(np.float64)
        vis_h = -(-th // sc)
        vis_w = -(-tw // sc)
        idx_r = np.minimum(rr // sc, vis_h - 1)
        idx_c = np.minimum(cc // sc, vis_w - 1)

        acc = np.zeros((th, tw, 3), np.float32)
        wsum = np.zeros((th, tw), np.float32)
        for k, v in zip(cand, views):
            rv, vv = backend.sample_view(v.rgb, v.cam(0), X0, Y0, gsd, Ztile)
            rgbw = backend.to_numpy(rv)
            valid = backend.to_numpy(vv)
            C = rec.C[k].astype(np.float64)
            dz = C[2] - wz
            dx = wx - C[0]
            dy = wy - C[1]
            cos_inc = np.clip(dz / np.sqrt(dx * dx + dy * dy + dz * dz + 1e-6), 0.0, 1.0)
            if use_occ and sc > 1:
                sr, scc = np.nonzero(sub[::sc, ::sc])
                sr, scc = sr * sc, scc * sc
                vis_map = np.ones((vis_h, vis_w), bool)
                if sr.size:
                    vis_map[sr // sc, scc // sc] = _visibility(
                        dsm, minX, maxY, gsd, C,
                        X0 + (scc + 0.5) * gsd, Y0 - (sr + 0.5) * gsd,
                        dsm[ty + sr, tx + scc].astype(np.float64), occ_steps, occ_tol)
                vis = vis_map[idx_r, idx_c]
            elif use_occ:
                vis = _visibility(dsm, minX, maxY, gsd, C, wx, wy, wz, occ_steps, occ_tol)
            else:
                vis = np.ones(wz.shape, bool)
            wgt = (rgbw[rr, cc, 3] ** 2) * cos_inc ** va_power * vis * (valid[rr, cc] > 0)
            acc[rr, cc] += rgbw[rr, cc, :3] * wgt[:, None]
            wsum[rr, cc] += wgt
        return tx, ty, th, tw, acc, wsum

    def consume(res):
        if res is None:
            return
        tx, ty, th, tw, acc, wsum = res
        m = wsum > 1e-6
        if not m.any():
            return
        block = rgb_out[ty:ty + th, tx:tx + tw]
        block[m] = np.clip(acc[m] / wsum[m][:, None], 0, 255).astype(np.uint8)
        WSUM[ty:ty + th, tx:tx + tw][m] = wsum[m]

    tiles = [(tx, ty) for ty in range(0, H, T) for tx in range(0, W, T)]
    tw_n = max(1, (W + T - 1) // T)
    ordered = []
    for r in range(0, len(tiles), tw_n):     # serpentine keeps neighbouring images cached
        row = tiles[r:r + tw_n]
        ordered.extend(row if (r // tw_n) % 2 == 0 else row[::-1])

    workers = int(g("workers")) or getattr(ar, "workers", 1)
    t0 = time.time()
    done = 0
    step = max(1, len(ordered) // 10)
    if backend.parallel_blocks and workers > 1:
        with ThreadPoolExecutor(workers) as ex:
            for res in ex.map(process, ordered):
                consume(res)
                done += 1
                if done % step == 0:
                    log.info("  ortho: %d/%d tiles", done, len(ordered))
    else:
        for t in ordered:
            consume(process(t))
            done += 1
            if done % step == 0:
                log.info("  ortho: %d/%d tiles", done, len(ordered))
    log.info("True orthophoto: %d/%d tiles in %.1fs (%d image loads)",
             len(ordered), len(ordered), time.time() - t0, cache.loads)
    return rgb_out, WSUM > 0
