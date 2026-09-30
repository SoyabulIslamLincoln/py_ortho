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


def _visibility(dsm, minX, maxY, gsd, C, X, Y, Z, max_steps, tol, zmax=None):
    """Visibility mask (True = visible) of surface points (X, Y, Z) from camera centre `C`.

    Height-field line of sight: walk from the point towards the camera, but only over the
    stretch where the ray is still below the highest surface (`zmax`) -- above that nothing can
    block it. Samples are one DSM cell apart (coarser only if a ray needs more than `max_steps`),
    so a wall or roof edge right next to the point is not skipped. The point is hidden when the
    DSM rises more than `tol` metres above the ray. Heights, not depths, are compared, so there is
    no camera-axis-depth vs range ambiguity.

    Limits (2.5D): overhangs, bridges and multi-layer structures are not representable; facades
    are the steep cells between roof and ground, so views grazing a facade are judged coarsely.
    """
    H, W = dsm.shape
    X, Y, Z = (np.asarray(a, np.float64) for a in (X, Y, Z))
    vis = np.ones(Z.shape, bool)
    if zmax is None:
        zmax = float(np.nanmax(dsm)) if np.isfinite(dsm).any() else None
    if zmax is None:
        return vis
    ux, uy = C[0] - X, C[1] - Y
    dh = np.hypot(ux, uy)
    rise = (C[2] - Z) / np.maximum(dh, 1e-6)          # ray height gain per horizontal metre
    ux, uy = ux / np.maximum(dh, 1e-6), uy / np.maximum(dh, 1e-6)
    dmax = np.minimum(np.maximum(zmax + tol - Z, 0.0) / np.maximum(rise, 1e-6), dh)
    step = np.maximum(gsd, dmax / max(1, int(max_steps)))
    n = int(np.ceil(np.max(dmax / step))) if dmax.size else 0
    for k in range(1, n + 1):
        d = k * step
        act = d <= dmax
        if not act.any():
            break
        col = ((X + ux * d - minX) / gsd).astype(np.int64)
        row = ((maxY - (Y + uy * d)) / gsd).astype(np.int64)
        inb = act & (col >= 0) & (col < W) & (row >= 0) & (row < H)
        zz = dsm[np.clip(row, 0, H - 1), np.clip(col, 0, W - 1)]
        vis &= ~(inb & np.isfinite(zz) & (zz > Z + rise * d + tol))
    return vis


def _box(a, r):
    """Mean over a (2r+1)^2 window (edge-clamped), via cumulative sums."""
    if r <= 0:
        return a.astype(np.float32)
    p = np.pad(a.astype(np.float64), r + 1, mode="edge")
    c = p.cumsum(0).cumsum(1)
    k = 2 * r + 1
    s = c[k:, k:] - c[:-k, k:] - c[k:, :-k] + c[:-k, :-k]
    return (s[:a.shape[0], :a.shape[1]] / (k * k)).astype(np.float32)


def _seam_labels(cost, cols, smooth, iters):
    """Spatially coherent source labels by ICM on a 4-connected MRF.

    E(l) = sum_p D_p(l_p) + smooth * sum_{p~q, l_p != l_q} (0.1 + |I_{l_p}(p) - I_{l_q}(p)| / 64)

    The data term D is the (normalised) unsuitability of a source; the pair term charges a seam
    in proportion to how much the two sources *disagree* there, so seams avoid buildings, cars
    and misaligned edges and run through places where neighbouring images look alike.
    ICM starts from the best-score labelling and only lowers the energy (a local optimum --
    a graph-cut solver would find a stronger one, at the cost of a new dependency).
    """
    L = cost.shape[0]
    lab = np.argmin(cost, 0)
    if L < 2 or smooth <= 0:
        return lab
    for _ in range(iters):
        tot = cost.copy()
        for axis, sh in ((0, 1), (0, -1), (1, 1), (1, -1)):
            nb = np.roll(lab, sh, axis)
            edge = np.ones(lab.shape, bool)       # np.roll wraps: ignore the wrapped border line
            if axis == 0:
                edge[0 if sh == 1 else -1, :] = False
            else:
                edge[:, 0 if sh == 1 else -1] = False
            cnb = np.take_along_axis(cols, nb[None, :, :, None], 0)[0]   # neighbour's source at p
            for l in range(L):
                diff = np.abs(cols[l] - cnb).mean(-1)
                tot[l] += smooth * edge * (nb != l) * (0.1 + diff / 64.0)
        new = np.argmin(tot, 0)
        if (new == lab).all():
            break
        lab = new
    return lab


class _Defaults:
    #changed here: fallback option object so true_orthophoto works without an Options3D.
    ortho_tile = 256
    max_views = 6
    view_angle_power = 1.5
    occlusion = True
    occlusion_tol = 0.20
    occlusion_steps = 96
    occlusion_stride = 2
    cache_mb = 1024
    workers = 0
    ortho_blend = "seam"
    source_weights = (1.0, 0.5, 0.7, 0.3)   # angle, resolution, border, exposure
    seam_smoothness = 0.6
    seam_iters = 4
    seam_band = 3


def true_orthophoto(ar, rec, dsm, minX, maxY, gsd, gains=None, biases=None, opt=None):
    #changed here: whole function -- the Pix4D "Orthomosaic" render stage on the final DSM.
    """Render `dsm` as an occlusion-aware true orthophoto.

    `minX`/`maxY` are the world coordinates of the top-left DSM cell *corner*; `dsm` is the
    final surface (NaN = no data). Inverse mapping: every cell centre (X, Y, DSM(X, Y)) is
    projected into the candidate views with the full camera model (distortion applied once, on
    the original images), sampled bilinearly, and kept only where the view is valid and the
    point is visible (height-field line of sight).

    Source selection (``ortho_blend="seam"``): each visible view gets a normalised score
    ``w_angle*cos(ray, normal)^p + w_res*res + w_border*border + w_exposure*unclipped`` (weights in
    ``source_weights``); labels are made spatially coherent by :func:`_seam_labels`, then blended
    only within ``seam_band`` cells of a seam. ``ortho_blend="feather"`` keeps the old weighted
    average of all visible views (more ghosting on tall objects).

    Returns ``(rgb (H, W, 3) uint8, covered (H, W) bool, source (H, W) int32, count (H, W) uint8)``:
    `source` is the index into ``rec.used`` of the view chosen per cell (-1 = unobserved) and
    `count` the number of views that see the cell. Unobserved cells stay uncovered (NoData).
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

    # Load images at (about) the orthophoto resolution, not full resolution: sampling a 4032 px
    # image for a 6 cm ortho oversamples ~5x and, at 36 MB/image, blows past the cache so images
    # are re-decoded hundreds of times. Matching the load scale to the output GSD keeps every
    # image in cache (one decode each) with no loss of ortho detail.
    native_gsd = float(getattr(getattr(ar, "alignment", None), "gsd", gsd) or gsd)
    img_scale = float(np.clip(native_gsd / gsd, 0.1, 1.0))

    def load(k):
        i = rec.used[k]
        fr = frames[i]
        rgb = load_rgb(fr, img_scale).astype(np.float32)
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
    blend = str(g("ortho_blend"))
    w_ang, w_res, w_bord, w_exp = (float(x) for x in g("source_weights"))
    smooth = float(g("seam_smoothness"))
    s_iters = int(g("seam_iters"))
    band = max(0, int(g("seam_band")))
    pad = band + 8 if blend == "seam" else 0     # context so seams/feathering continue across tiles
    zmax = float(np.nanmax(dsm))

    # surface normals from the DSM (for the viewing-angle term)
    gy, gx = np.gradient(np.where(covered, dsm, zfill).astype(np.float32), gsd)
    nrm = np.dstack([gx, -gy, -np.ones_like(gx)])          # d/dx, d/dy(north-up rows) -> normal
    nrm /= -np.linalg.norm(nrm, axis=2, keepdims=True)
    # steep cells are facades/edge ramps of the 2.5D surface: scoring them against their tilted
    # normal would favour oblique views that look *at* the wall, so use the vertical there
    nrm[nrm[..., 2] < np.cos(np.radians(30))] = (0.0, 0.0, 1.0)

    cam_xy = rec.C[:, :2]
    SRC = np.full((H, W), -1, np.int32)                       # index into rec.used, -1 = none
    CNT = np.zeros((H, W), np.uint8)                          # number of visible, valid views

    def process(tile):
        tx, ty = tile
        tw, th = min(T, W - tx), min(T, H - ty)
        if not covered[ty:ty + th, tx:tx + tw].any():
            return None
        px0, py0 = max(0, tx - pad), max(0, ty - pad)
        px1, py1 = min(W, tx + tw + pad), min(H, ty + th + pad)
        pw, ph = px1 - px0, py1 - py0
        sub = covered[py0:py1, px0:px1]
        X0 = minX + px0 * gsd
        Y0 = maxY - py0 * gsd
        Ztile = np.where(sub, dsm[py0:py1, px0:px1], zfill).astype(np.float32)

        # candidate views: those that see the tile (at least its centre over the height range),
        # nearest first (spatial restriction: never every image for every cell). Per-cell `valid`
        # from the sampler masks the parts a view misses.
        zz = Ztile[sub]
        zlo, zhi = float(zz.min()), float(zz.max())
        cxw = X0 + pw * gsd / 2.0
        cyw = Y0 - ph * gsd / 2.0
        corners = np.array([[X0, Y0], [X0 + pw * gsd, Y0], [X0, Y0 - ph * gsd], [X0 + pw * gsd, Y0 - ph * gsd],
                            [cxw, cyw]])
        pts = np.concatenate([np.column_stack([corners, np.full(len(corners), z)]) for z in (zlo, zhi)])
        centre = np.array([[cxw, cyw, 0.5 * (zlo + zhi)]])
        dist = np.hypot(cam_xy[:, 0] - cxw, cam_xy[:, 1] - cyw)
        cand = []
        for k in np.argsort(dist):
            it = rec.intr[rec.cam_group[k]]
            xc = (pts - rec.C[k]) @ rec.R[k].T
            u = it.f * xc[:, 0] / np.maximum(xc[:, 2], 1e-6) + it.cx
            v = it.f * xc[:, 1] / np.maximum(xc[:, 2], 1e-6) + it.cy
            inside = (xc[:, 2] > 0) & (u >= 0) & (u <= it.width - 1) & (v >= 0) & (v <= it.height - 1)
            cc = (centre - rec.C[k]) @ rec.R[k].T
            cu = it.f * cc[0, 0] / max(cc[0, 2], 1e-6) + it.cx
            cv = it.f * cc[0, 1] / max(cc[0, 2], 1e-6) + it.cy
            sees_centre = cc[0, 2] > 0 and 0 <= cu <= it.width - 1 and 0 <= cv <= it.height - 1
            if sees_centre or inside.mean() >= 0.5:
                cand.append(int(k))
            if len(cand) >= max_views:
                break
        if not cand:
            return None
        views = [cache.get(k) for k in cand]

        rr, cc = np.nonzero(sub)
        wx = X0 + (cc + 0.5) * gsd                 # map coordinates of cell centres
        wy = Y0 - (rr + 0.5) * gsd
        wz = dsm[py0 + rr, px0 + cc].astype(np.float64)
        n_c = nrm[py0 + rr, px0 + cc]
        vis_h, vis_w = -(-ph // occ_stride), -(-pw // occ_stride)
        L = len(cand)
        cols = np.zeros((L, ph, pw, 3), np.float32)
        score = np.full((L, ph, pw), -1.0, np.float32)            # -1 = not usable
        inv_depth = np.zeros((L, rr.size), np.float32)
        for j, (k, v) in enumerate(zip(cand, views)):
            rv, vv = backend.sample_view(v.rgb, v.cam(0), X0, Y0, gsd, Ztile)
            rgbw = backend.to_numpy(rv)
            valid = backend.to_numpy(vv)[rr, cc] > 0
            C = rec.C[k].astype(np.float64)
            ray = np.column_stack([C[0] - wx, C[1] - wy, C[2] - wz])
            rng_ = np.linalg.norm(ray, axis=1) + 1e-9
            cos_n = np.clip(np.sum(ray * n_c, 1) / rng_, 0.0, 1.0)     # vs. surface normal
            if use_occ:
                sr, scc = np.nonzero(sub[::occ_stride, ::occ_stride])
                vis_map = np.ones((vis_h, vis_w), bool)
                if sr.size:
                    sr_, sc_ = sr * occ_stride, scc * occ_stride
                    vis_map[sr, scc] = _visibility(dsm, minX, maxY, gsd, C, X0 + (sc_ + 0.5) * gsd,
                                                   Y0 - (sr_ + 0.5) * gsd, dsm[py0 + sr_, px0 + sc_],
                                                   occ_steps, occ_tol, zmax)
                    if occ_stride > 1:   # a coarse cell is visible only if it and its 4-neighbours are
                        vm = vis_map.copy()
                        vm[1:] &= vis_map[:-1]; vm[:-1] &= vis_map[1:]
                        vm[:, 1:] &= vis_map[:, :-1]; vm[:, :-1] &= vis_map[:, 1:]
                        vis_map = vm
                vis = vis_map[rr // occ_stride, cc // occ_stride]
            else:
                vis = np.ones(rr.size, bool)
            ok = valid & vis
            px = rgbw[rr, cc, :3]
            lum = px.mean(1)
            expo = np.where((lum > 250) | (lum < 5), 0.0, 1.0)            # clipped samples
            depth = np.sum((np.column_stack([wx, wy, wz]) - C) * rec.R[k][2], 1)   # camera-axis depth
            inv_depth[j] = np.where(ok, 1.0 / np.maximum(depth, 1e-3), 0)
            s = (w_ang * cos_n ** va_power + w_bord * np.clip(rgbw[rr, cc, 3], 0, 1) + w_exp * expo)
            cols[j, rr, cc] = px
            score[j, rr, cc] = np.where(ok, s, -1.0)
        # resolution term: projected pixel size relative to the finest view at this cell
        best = inv_depth.max(0)
        res_t = np.where(best > 0, inv_depth / np.maximum(best, 1e-9), 0)
        for j in range(L):
            score[j, rr, cc] = np.where(score[j, rr, cc] >= 0, score[j, rr, cc] + w_res * res_t[j], -1.0)
        wsum_w = max(w_ang + w_res + w_bord + w_exp, 1e-6)
        usable = score >= 0
        count = usable.sum(0)

        if blend == "seam":
            cost = np.where(usable, 1.0 - score / wsum_w, 1e3).astype(np.float32)
            lab = _seam_labels(cost, cols, smooth, s_iters)
            has = count > 0
            # feather only inside a band around the seams; elsewhere one source per cell
            wts = np.zeros((L, ph, pw), np.float32)
            for j in range(L):
                wts[j] = _box((lab == j) & has, band) * usable[j]
            ws = wts.sum(0)
            rgb = (wts[..., None] * cols).sum(0) / np.maximum(ws, 1e-6)[..., None]
            src = np.where(has, np.asarray(cand)[lab], -1)
        else:   # legacy weighted average of all visible views
            w = np.where(usable, np.maximum(score, 0) ** 2, 0)
            ws = w.sum(0)
            rgb = (w[..., None] * cols).sum(0) / np.maximum(ws, 1e-6)[..., None]
            src = np.where(count > 0, np.asarray(cand)[np.argmax(score, 0)], -1)
        oy, ox = ty - py0, tx - px0
        crop = (slice(oy, oy + th), slice(ox, ox + tw))
        return tx, ty, th, tw, rgb[crop], (ws[crop] > 1e-6) & (count[crop] > 0), src[crop], count[crop]

    def consume(res):
        if res is None:
            return
        tx, ty, th, tw, rgb, m, src, count = res
        if not m.any():
            return
        block = rgb_out[ty:ty + th, tx:tx + tw]
        block[m] = np.clip(rgb[m] + 0.5, 0, 255).astype(np.uint8)
        SRC[ty:ty + th, tx:tx + tw][m] = src[m]
        CNT[ty:ty + th, tx:tx + tw] = np.minimum(count, 255)

    tiles = [(tx, ty) for ty in range(0, H, T) for tx in range(0, W, T)]
    tw_n = max(1, (W + T - 1) // T)
    ordered = []
    for r in range(0, len(tiles), tw_n):     # serpentine keeps neighbouring images cached
        row = tiles[r:r + tw_n]
        ordered.extend(row if (r // tw_n) % 2 == 0 else row[::-1])

    workers = int(g("workers")) or getattr(ar, "workers", 1)
    log.info("True orthophoto: rendering %d tiles", len(ordered))
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
    return rgb_out, SRC >= 0, SRC, CNT
