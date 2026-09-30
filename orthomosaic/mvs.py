"""Dense 2.5D reconstruction for nadir surveys: multi-view height sweep.

For every cell of a ground grid, candidate heights are swept coarse-to-fine. At each
candidate the cell is projected into the nearby views, and the photo-consistency is
the windowed NCC between the most-nadir ("reference") view and each other view,
averaged over the best half of the views (robust to occlusion). The best height per
cell is the DSM; the colours of the agreeing views give a *true* orthophoto.

All array maths is written against `xp` (numpy or cupy), so the same code runs on the
CPU (Cython sampler) and on CUDA (raw kernel sampler).
"""
from __future__ import annotations

import logging
import math
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Optional

import numpy as np

from . import _mvs
from .imageio import load_rgb
from .render import ImageCache

log = logging.getLogger(__name__)


@dataclass
class DenseOptions:
    gsd: Optional[float] = None       # DSM cell size in metres; default 2 x native GSD
    levels: int = 3                   # coarse-to-fine pyramid levels (factor 2)
    window: int = 3                   # NCC window radius in cells
    max_views: int = 6
    min_score: float = 0.5            # NCC needed to trust a height
    tile: int = 160                   # cells per tile side at full resolution (keep << image footprint)
    overlap: int = 12                 # cells of feathered height blending between tiles
    max_cells: float = 40e6           # cap on DSM size: the cell size grows if the area needs more
    refine_steps: int = 4             # +- steps searched at each finer level
    min_texture: float = 4.0          # gray variance below this is "no texture"
    full_texture: float = 8.0         # gray std at which NCC is fully trusted (weaker texture is down-weighted)
    sgm_p1: float = 0.05              # SGM penalty for a one-step height change (cost = 1 - NCC)
    sgm_p2: float = 0.6               # SGM penalty for a larger jump (building edges)
    refine_min_score: float = 0.2     # finer levels only override the coarse height above this
    #changed here: Pix4D-style true-orthophoto cue -- near-nadir views get more colour weight
    view_angle_power: float = 1.5     # cos(incidence) exponent when blending true-ortho colour
    cache_mb: int = 1536
    workers: int = 0


@dataclass
class DenseResult:
    Z: np.ndarray            # (H, W) float32 heights, NaN where unknown
    score: np.ndarray        # (H, W) float32 NCC score
    rgb: np.ndarray          # (H, W, 3) uint8 true-ortho colour
    covered: np.ndarray      # (H, W) bool: seen by >= 2 views
    minX: float
    maxY: float
    gsd: float


def _median3(xp, a):
    """3x3 median (sort-based, so it also runs on backends without a median op)."""
    p = xp.pad(a, 1, mode="edge")
    H, W = a.shape
    stack = xp.stack([p[dy:dy + H, dx:dx + W] for dy in range(3) for dx in range(3)])
    return xp.sort(stack, axis=0)[4]


class _View:
    """A camera with an image pyramid, cached on the compute device."""

    def __init__(self, gray_pyr, rgb, R, C, f, k1, k2, cx, cy):
        self.gray_pyr, self.rgb = gray_pyr, rgb
        self.R, self.C = R, C
        self.f, self.k1, self.k2, self.cx, self.cy = f, k1, k2, cx, cy

    def cam(self, level):
        """Camera parameters in the pixel units of pyramid level `level`."""
        s = 0.5 ** level
        return (self.R, self.C, self.f * s, self.k1, self.k2,
                (self.cx + 0.5) * s - 0.5, (self.cy + 0.5) * s - 0.5)

    @property
    def nbytes(self):
        return int(sum(g.nbytes for g in self.gray_pyr) + self.rgb.nbytes)


def _pyramid(gray, n):
    out = [gray]
    for _ in range(n - 1):
        g = out[-1]
        h, w = g.shape[0] // 2 * 2, g.shape[1] // 2 * 2
        g = g[:h, :w, 0]
        out.append(((g[0::2, 0::2] + g[1::2, 0::2] + g[0::2, 1::2] + g[1::2, 1::2]) * 0.25)[..., None]
                   .astype(np.float32))
    return [np.ascontiguousarray(g) for g in out]


def grid_extent(rec, gsd: float, native_gsd: float, max_cells: float = 40e6):
    """North-up DSM grid covering the camera footprints at typical ground height.

    Returns (minX, maxY, W, H, gsd, img_scale); minX/maxY are the top-left cell *corner*, the
    extent is snapped to multiples of the cell size and the cell size grows if the area would
    need more than `max_cells` cells."""
    img_scale = min(1.0, native_gsd / gsd)
    zg = float(np.percentile(rec.X[:, 2], 20)) if len(rec.X) else 0.0
    corners = []
    for k, i in enumerate(rec.used):
        it = rec.intr[rec.cam_group[k]]
        R, C = rec.R[k], rec.C[k]
        for u, v in [(0, 0), (it.width, 0), (it.width, it.height), (0, it.height)]:
            d = R.T @ np.array([(u - it.cx) / it.f, (v - it.cy) / it.f, 1.0])
            if d[2] < -1e-6:
                corners.append(C + d * (zg - C[2]) / d[2])
    corners = np.array(corners)
    # keep the area imaged by at least a couple of cameras: trim extreme corners
    minX = math.floor(np.percentile(corners[:, 0], 1) / gsd) * gsd
    maxX = math.ceil(np.percentile(corners[:, 0], 99) / gsd) * gsd
    minY = math.floor(np.percentile(corners[:, 1], 1) / gsd) * gsd
    maxY = math.ceil(np.percentile(corners[:, 1], 99) / gsd) * gsd
    if (maxX - minX) * (maxY - minY) / gsd ** 2 > max_cells:
        gsd = math.sqrt((maxX - minX) * (maxY - minY) / max_cells)
        img_scale = min(1.0, native_gsd / gsd)
        log.info("Dense: cell size raised to %.3f m to stay under %.0fM cells", gsd, max_cells / 1e6)
        minX, maxX = math.floor(minX / gsd) * gsd, math.ceil(maxX / gsd) * gsd
        minY, maxY = math.floor(minY / gsd) * gsd, math.ceil(maxY / gsd) * gsd
    W = int(round((maxX - minX) / gsd))
    H = int(round((maxY - minY) / gsd))
    return minX, maxY, W, H, gsd, img_scale


def dense_reconstruct(ar, rec, gains: dict, native_gsd: float, opt: DenseOptions,
                      biases: Optional[dict] = None) -> DenseResult:
    #changed here: `biases` (optional) carries the Pix4D colour-balancing per-image offsets.
    backend = ar.backend
    xp = backend.xp
    frames = ar.frames
    t0 = time.time()
    #changed here: Pix4D-grade resolution -- the DSM/ortho default to the native GSD (was 2x)
    gsd = opt.gsd or native_gsd
    img_scale = min(1.0, native_gsd / gsd)          # decode images so 1 px ~ 1 DSM cell
    workers = opt.workers or ar.workers

    minX, maxY, W, H, gsd, img_scale = grid_extent(rec, gsd, native_gsd, opt.max_cells)
    zg = float(np.percentile(rec.X[:, 2], 20)) if len(rec.X) else 0.0
    log.info("Dense: %d x %d cells at %.3f m, images at %.2f scale, %s backend", W, H, gsd, img_scale, backend.name)

    Zacc = np.zeros((H, W), np.float32)
    Wacc = np.zeros((H, W), np.float32)
    Sout = np.full((H, W), -1.0, np.float32)
    RGB = np.zeros((H, W, 3), np.uint8)
    COV = np.zeros((H, W), bool)

    levels = max(1, opt.levels)

    def load(k):
        i = rec.used[k]
        fr = frames[i]
        rgb = load_rgb(fr, img_scale).astype(np.float32)
        g = gains.get(i)
        if g is not None:
            rgb *= np.asarray(g, np.float32)
        #changed here: apply the Pix4D colour-balancing offset (black level) when present
        b = None if biases is None else biases.get(i)
        if b is not None:
            rgb += np.asarray(b, np.float32)
        gray = (0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2])[..., None]
        pyr = _pyramid(np.ascontiguousarray(gray, np.float32), levels)
        # 4th channel: feather weight (distance to the image border), so colours blend smoothly
        h, w = rgb.shape[:2]
        fy = np.minimum(np.arange(h) + 0.5, h - 0.5 - np.arange(h)) / (0.5 * min(h, w))
        fx = np.minimum(np.arange(w) + 0.5, w - 0.5 - np.arange(w)) / (0.5 * min(h, w))
        feather = np.clip(np.minimum(fy[:, None], fx[None, :]), 0, 1).astype(np.float32)
        rgb = np.dstack([rgb, feather])
        it = rec.intr[rec.cam_group[k]]
        sx = rgb.shape[1] / fr.width
        return _View([backend.upload(p) for p in pyr], backend.upload(np.ascontiguousarray(rgb)),
                     np.ascontiguousarray(rec.R[k]), np.ascontiguousarray(rec.C[k]),
                     it.f * sx, it.k1, it.k2, (it.cx + 0.5) * sx - 0.5, (it.cy + 0.5) * sx - 0.5)

    cache = ImageCache(load, lambda v: v.nbytes, opt.cache_mb * 1024 * 1024)

    # sparse points for per-tile height ranges
    Xs = rec.X
    if len(Xs):
        # robust height range: ignore stray sparse points, keep a margin for roof tops
        q1, q99 = np.percentile(Xs[:, 2], [1, 99])
        margin = 0.1 * (q99 - q1) + 1.0
        zlo_g, zhi_g = q1 - margin, q99 + margin
    else:
        zlo_g, zhi_g = zg - 5, zg + 30

    T = opt.tile
    pad = opt.window * (2 ** (levels - 1)) + 4
    tiles = [(tx, ty) for ty in range(0, H, T) for tx in range(0, W, T)]
    cam_xy = rec.C[:, :2]

    def tile_setup(tile):
        """Geometry of a tile and the views to use (cheap; no image access)."""
        tx, ty = tile
        tw, th = min(T, W - tx), min(T, H - ty)
        # padded tile grid (finest level)
        px0, py0 = tx - pad, ty - pad
        PW, PH = tw + 2 * pad, th + 2 * pad
        X0 = minX + px0 * gsd
        Y0 = maxY - py0 * gsd
        cxw = minX + (tx + tw / 2) * gsd
        cyw = maxY - (ty + th / 2) * gsd
        zlo, zhi = zlo_g - 1.0, zhi_g + 1.0
        # views that see the WHOLE padded tile over the tile's (local) height range, most nadir
        # first (the first one is the NCC reference, so it must cover every cell of the tile)
        m = ((Xs[:, 0] > X0) & (Xs[:, 0] < X0 + PW * gsd) & (Xs[:, 1] < Y0) & (Xs[:, 1] > Y0 - PH * gsd))
        zs_lo, zs_hi = (np.percentile(Xs[m, 2], [5, 95]) if m.sum() >= 20 else (zlo, zhi))
        corners = np.array([[X0, Y0], [X0 + PW * gsd, Y0], [X0, Y0 - PH * gsd], [X0 + PW * gsd, Y0 - PH * gsd]])
        pts = np.concatenate([np.column_stack([corners, np.full(4, z)]) for z in (zs_lo, zs_hi)])
        dist = np.hypot(cam_xy[:, 0] - cxw, cam_xy[:, 1] - cyw)
        cand, partial = [], []
        for k in np.argsort(dist):
            it = rec.intr[rec.cam_group[k]]
            xc = (pts - rec.C[k]) @ rec.R[k].T
            if np.any(xc[:, 2] <= 0):
                continue
            u = it.f * xc[:, 0] / xc[:, 2] + it.cx
            v = it.f * xc[:, 1] / xc[:, 2] + it.cy
            inside = (u >= 0) & (u <= it.width - 1) & (v >= 0) & (v <= it.height - 1)
            if inside.all():
                cand.append(int(k))
            elif inside.any():
                partial.append(int(k))
            if len(cand) >= opt.max_views:
                break
        if len(cand) < 2:          # tile at the block edge: accept partial coverage
            cand = (cand + partial)[:opt.max_views]
        if len(cand) < 2:
            return None
        return tx, ty, tw, th, px0, py0, PW, PH, X0, Y0, zlo, zhi, cand

    def process(tile, st_=None):
        st_ = tile_setup(tile) if st_ is None else st_
        if st_ is None:
            return None
        tx, ty, tw, th, px0, py0, PW, PH, X0, Y0, zlo, zhi, cand = st_
        views = [cache.get(k) for k in cand]
        topk = max(1, len(views) // 2)

        cam_c = [np.asarray(v.C, np.float64) for v in views]

        def score_at(Zmap, level, gsd_l, X0l, Y0l, want_rgb=False):
            """Photo-consistency of every cell at heights Zmap. Each cell is scored against its
            own reference view (the camera most directly above it), so results do not depend on
            how the area is tiled. Returns score (H,W) [and NCC-vs-reference per view, valids]."""
            HH, WW = Zmap.shape[-2:]            # Zmap may be a (D, HH, WW) stack of hypotheses
            V = len(views)
            full = 1.0 - 1e-6
            samples, valids = [], []
            for v in views:
                img = v.gray_pyr[min(level, len(v.gray_pyr) - 1)]
                out, valid = backend.sample_view(img, v.cam(level), X0l, Y0l, gsd_l, Zmap)
                samples.append(out[..., 0])
                valids.append(valid)
            # everything below is a handful of large, batched array operations
            S = xp.stack(samples)                                    # (V, ...)
            VA = xp.stack(valids).astype(xp.float32)
            M = backend.box(S, opt.window)
            VAR = backend.box(S * S, opt.window) - M * M
            FULL = backend.box(VA, opt.window) >= full
            # per-cell reference: nearest camera (in XY) among views that see the whole window
            cxs = backend.asarray(X0l + (np.arange(WW) + 0.5) * gsd_l, xp.float32)
            cys = backend.asarray(Y0l - (np.arange(HH) + 0.5) * gsd_l, xp.float32)
            d2 = xp.stack([(cys[:, None] - float(c[1])) ** 2 + (cxs[None, :] - float(c[0])) ** 2 for c in cam_c])
            d2 = d2.reshape((V,) + (1,) * (Zmap.ndim - 2) + (HH, WW))
            dist = xp.where(FULL, d2, xp.inf)
            ref, dmin = backend.argmin0(dist)
            # NCC of every view pair at once, then pick NCC(ref(cell), v) for each view v
            ia, ib = np.triu_indices(V, 1)
            IA, IB = backend.asarray(ia, xp.int32), backend.asarray(ib, xp.int32)
            COV = backend.box(S[IA] * S[IB], opt.window) - M[IA] * M[IB]
            NCC = COV / xp.sqrt(xp.maximum(VAR[IA], 1e-6) * xp.maximum(VAR[IB], 1e-6))
            NCC = xp.where(FULL[IA] & FULL[IB], NCC, -1.0).astype(xp.float32)
            pair = np.full((V, V), len(ia), np.int32)                 # index into NCC (+ a "self" slot)
            pair[ia, ib] = np.arange(len(ia))
            pair[ib, ia] = np.arange(len(ia))
            NCCx = xp.concatenate([NCC, xp.full((1,) + tuple(NCC.shape[1:]), -2.0, xp.float32)])
            PAIR = backend.asarray(pair, xp.int32)                     # (V_ref, V)
            sel = PAIR[ref]                                            # (..., V): pair slot per (cell, v)
            sel = xp.moveaxis(sel, -1, 0)                              # (V, ...)
            st = xp.take_along_axis(NCCx, sel, axis=0)                 # (V, ...) NCC(ref, v); -2 for v == ref
            varis = VAR
            score = backend.topk_mean(st, topk)
            varR = xp.take_along_axis(varis, ref[None], axis=0)[0]
            # NCC on near-uniform patches is noise: scale it by texture strength so such cells
            # give a flat (neutral) cost and SGM fills them from their textured surroundings
            tex_w = xp.clip(xp.sqrt(xp.maximum(varR, 0.0)) / opt.full_texture, 0.0, 1.0)
            ok = xp.isfinite(dmin) & (varR > opt.min_texture)
            score = xp.where(ok, score * tex_w, 0.0).astype(xp.float32)
            if not want_rgb:
                return score
            return score, st, valids, ref

        Zbest = None
        for level in range(levels - 1, -1, -1):
            s = 2 ** level
            HH, WW = (PH + s - 1) // s, (PW + s - 1) // s
            gsd_l = gsd * s
            if Zbest is None:
                # full height range on the coarse grid + semi-global matching, so weakly
                # textured surfaces (flat roofs, fields) inherit heights from their edges
                step = gsd_l
                cands = np.arange(zlo, zhi + step, step).astype(np.float32)
                # all height hypotheses in one batched evaluation (few, large GPU launches)
                vol = []
                for c0 in range(0, len(cands), 64):
                    zs = cands[c0:c0 + 64]
                    Zst = backend.asarray(np.broadcast_to(zs[:, None, None], (len(zs), HH, WW)), xp.float32)
                    vol.append(backend.to_numpy(score_at(Zst, level, gsd_l, X0, Y0)))
                vol = np.concatenate(vol)
                cost = np.ascontiguousarray(1.0 - np.clip(vol, -1.0, 1.0), np.float32)
                agg = _mvs.sgm(cost, opt.sgm_p1, opt.sgm_p2)
                bi = np.argmin(agg, axis=0)
                bestZ = backend.asarray(cands[bi], xp.float32)
            else:
                prev = xp.repeat(xp.repeat(Zbest, 2, axis=0), 2, axis=1)[:HH, :WW]
                if prev.shape != (HH, WW):
                    prev = xp.pad(prev, ((0, HH - prev.shape[0]), (0, WW - prev.shape[1])), mode="edge")
                step = gsd_l
                offs = np.arange(-opt.refine_steps, opt.refine_steps + 1) * step
                Zst = (prev[None] + backend.asarray(offs[:, None, None], xp.float32)).astype(xp.float32)
                scores = score_at(Zst, level, gsd_l, X0, Y0)
                bi, bestS = backend.argmax0(scores)
                weak = bestS < opt.refine_min_score    # no texture support: keep the coarse surface
                bi = xp.where(weak, opt.refine_steps, bi)
                bestS = xp.take_along_axis(scores, bi[None], axis=0)[0]
                sub = xp.zeros_like(bestS)
                if level == 0:   # parabolic sub-step refinement
                    lo = xp.clip(bi - 1, 0, len(offs) - 1)
                    hi = xp.clip(bi + 1, 0, len(offs) - 1)
                    s_lo = xp.take_along_axis(scores, lo[None], axis=0)[0]
                    s_hi = xp.take_along_axis(scores, hi[None], axis=0)[0]
                    den = s_lo - 2 * bestS + s_hi
                    ok = (den < -1e-6) & (bi > 0) & (bi < len(offs) - 1)
                    den = xp.where(ok, den, -1.0)
                    sub = xp.where(ok, xp.clip(0.5 * (s_lo - s_hi) / den, -0.5, 0.5), 0.0)
                bestZ = prev + backend.asarray(offs, xp.float32)[bi] + sub.astype(xp.float32) * float(step)
            Zbest = _median3(xp, bestZ.astype(xp.float32)).astype(xp.float32) if level > 0 else bestZ.astype(xp.float32)

        # final pass at the chosen heights: score, agreeing views, colour
        score, st, valids, ref = score_at(Zbest, 0, gsd, X0, Y0, want_rgb=True)
        #changed here: Pix4D view-angle weighting.  The true orthophoto must sample the surface
        # with the least off-nadir stretch, so near-nadir views are favoured by cos(incidence).
        cxs_w = backend.asarray((X0 + (np.arange(PW) + 0.5) * gsd).astype(np.float32), xp.float32)
        cys_w = backend.asarray((Y0 - (np.arange(PH) + 0.5) * gsd).astype(np.float32), xp.float32)
        va_power = float(getattr(opt, "view_angle_power", 1.5))
        acc = xp.zeros(Zbest.shape + (3,), xp.float32)
        wsum = xp.zeros(Zbest.shape, xp.float32)
        for vi, v in enumerate(views):
            rgbw, valid = backend.sample_view(v.rgb, v.cam(0), X0, Y0, gsd, Zbest)
            # views that disagree with the cell's reference at this height are likely occluded
            agree = xp.where(ref == vi, 1.0, xp.clip((st[vi] + 0.2) / 0.6, 0.0, 1.0))
            #changed here: cos(angle between the local up normal and the view direction)
            Cv = cam_c[vi]
            dz = float(Cv[2]) - Zbest
            dx = cxs_w[None, :] - float(Cv[0])
            dy = cys_w[:, None] - float(Cv[1])
            cos_inc = xp.clip(dz / xp.sqrt(dx * dx + dy * dy + dz * dz + 1e-6), 0.0, 1.0)
            w = rgbw[..., 3] ** 2 * agree * valid.astype(xp.float32) * cos_inc ** va_power
            acc = acc + rgbw[..., :3] * w[..., None]
            wsum = wsum + w
        color = acc / xp.maximum(wsum, 1e-6)[..., None]
        nvalid = sum(v.astype(xp.int32) for v in valids)
        to_np = backend.to_numpy
        # heights are returned with an overlap margin and blended with a linear ramp, so the
        # per-tile reference choice does not leave steps at tile borders
        ov = min(pad, opt.overlap)
        ex0, ey0 = max(0, tx - ov), max(0, ty - ov)
        ex1, ey1 = min(W, tx + tw + ov), min(H, ty + th + ov)
        zs = (slice(ey0 - py0, ey1 - py0), slice(ex0 - px0, ex1 - px0))
        ry = np.arange(ey0, ey1)
        rx = np.arange(ex0, ex1)
        wy = np.clip(np.minimum(ry - (ty - ov) + 0.5, (ty + th + ov) - ry - 0.5) / max(2 * ov, 1), 0, 1)
        wx = np.clip(np.minimum(rx - (tx - ov) + 0.5, (tx + tw + ov) - rx - 0.5) / max(2 * ov, 1), 0, 1)
        ramp = (np.minimum(wy[:, None], wx[None, :]) if ov > 0 else np.ones((len(ry), len(rx)))).astype(np.float32)
        sl = (slice(pad, pad + th), slice(pad, pad + tw))
        return (tx, ty, tw, th, (ex0, ey0, ex1, ey1), to_np(Zbest[zs]), ramp, to_np(score[sl]),
                np.clip(to_np(color[sl]), 0, 255).astype(np.uint8), to_np(nvalid[sl] >= 2))

    done = 0
    step = max(1, len(tiles) // 10)
    ordered = []
    tw_n = (W + T - 1) // T
    for r in range(0, len(tiles), tw_n):   # serpentine for cache reuse
        row = tiles[r:r + tw_n]
        ordered.extend(row if (r // tw_n) % 2 == 0 else row[::-1])

    def consume(res):
        nonlocal done
        done += 1
        if done % step == 0:
            log.info("  dense: %d/%d tiles", done, len(tiles))
        if res is None:
            return
        tx, ty, tw, th, (ex0, ey0, ex1, ey1), z, ramp, sc, col, cov = res
        Zacc[ey0:ey1, ex0:ex1] += z * ramp
        Wacc[ey0:ey1, ex0:ex1] += ramp
        Sout[ty:ty + th, tx:tx + tw] = sc
        RGB[ty:ty + th, tx:tx + tw] = col
        COV[ty:ty + th, tx:tx + tw] = cov

    if backend.parallel_blocks and workers > 1:
        with ThreadPoolExecutor(workers) as ex:
            for res in ex.map(process, ordered):
                consume(res)
    else:
        # GPU: tiles run one at a time; decode the views of the next tiles in background threads
        setups = [tile_setup(t) for t in ordered]
        prefetched = set()
        with ThreadPoolExecutor(max(2, workers)) as ex:
            for k, t in enumerate(ordered):
                for nxt in setups[k + 1:k + 3]:
                    for v in (nxt[-1] if nxt else []):
                        if v not in prefetched:
                            prefetched.add(v)
                            ex.submit(cache.get, v)
                consume(process(t, setups[k]))
    Zout = np.where(Wacc > 0, Zacc / np.maximum(Wacc, 1e-6), np.nan).astype(np.float32)
    del Zacc, Wacc
    log.info("Dense sweep done in %.1fs (%d image loads)", time.time() - t0, cache.loads)
    return DenseResult(Zout, Sout, RGB, COV, minX, maxY, gsd)


# --------------------------------------------------------------------------
# post-processing
# --------------------------------------------------------------------------

def _nanmedian_filter(Z, r=2, rows=256):
    """NaN-aware median filter, processed in row bands to bound memory."""
    import warnings
    H, W = Z.shape
    p = np.pad(Z, r, mode="constant", constant_values=np.nan)
    out = np.empty_like(Z)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for y0 in range(0, H, rows):
            y1 = min(H, y0 + rows)
            stack = np.stack([p[y0 + dy:y1 + dy, dx:dx + W] for dy in range(2 * r + 1) for dx in range(2 * r + 1)])
            out[y0:y1] = np.nanmedian(stack, axis=0)
    return out


def _upsample2(a: np.ndarray, shape) -> np.ndarray:
    """Bilinear 2x upsampling (pixel-centre aligned) to exactly `shape`."""
    H, W = shape
    h, w = a.shape

    def coords(n_out, n_in):
        x = (np.arange(n_out) + 0.5) / 2.0 - 0.5
        x = np.clip(x, 0, n_in - 1)
        i0 = np.floor(x).astype(np.int64)
        i1 = np.minimum(i0 + 1, n_in - 1)
        return i0, i1, (x - i0).astype(np.float32)

    y0, y1, fy = coords(H, h)
    x0, x1, fx = coords(W, w)
    top = a[y0][:, x0] * (1 - fx) + a[y0][:, x1] * fx
    bot = a[y1][:, x0] * (1 - fx) + a[y1][:, x1] * fx
    return top * (1 - fy)[:, None] + bot * fy[:, None]


def fill_holes(Z: np.ndarray) -> np.ndarray:
    """Push-pull interpolation of NaN cells: average valid cells down a pyramid, then fill
    holes from bilinearly upsampled coarser levels (smooth, no blocky artefacts, no scipy)."""
    levels = [Z.astype(np.float32)]
    while min(levels[-1].shape) > 4:
        z = levels[-1]
        w = np.isfinite(z).astype(np.float32)
        h, ww = (z.shape[0] + 1) // 2 * 2, (z.shape[1] + 1) // 2 * 2
        zp = np.zeros((h, ww), np.float32)
        wp = np.zeros((h, ww), np.float32)
        zp[:z.shape[0], :z.shape[1]] = np.nan_to_num(z) * w
        wp[:z.shape[0], :z.shape[1]] = w
        ws = wp[0::2, 0::2] + wp[1::2, 0::2] + wp[0::2, 1::2] + wp[1::2, 1::2]
        zs = zp[0::2, 0::2] + zp[1::2, 0::2] + zp[0::2, 1::2] + zp[1::2, 1::2]
        levels.append(np.where(ws > 0, zs / np.maximum(ws, 1e-12), np.nan).astype(np.float32))
    filled = levels[-1]
    if np.isnan(filled).any():
        filled = np.where(np.isnan(filled), np.nanmean(filled) if np.isfinite(filled).any() else 0.0, filled)
    for lvl in range(len(levels) - 2, -1, -1):
        z = levels[lvl]
        up = _upsample2(filled, z.shape)
        filled = np.where(np.isfinite(z), z, up)
    return filled.astype(np.float32)


def _near(mask: np.ndarray, cells: int) -> np.ndarray:
    """Cells within `cells` (chessboard distance) of a True cell, by repeated 3x3 dilation."""
    out = mask.copy()
    for _ in range(max(0, int(cells))):
        grown = out.copy()
        grown[1:] |= out[:-1]
        grown[:-1] |= out[1:]
        grown[:, 1:] |= out[:, :-1]
        grown[:, :-1] |= out[:, 1:]
        if (grown == out).all():
            break
        out = grown
    return out


def postprocess(d: DenseResult, min_score: float, max_dev: float = 1.0,
                smooth_range: float = 0.4, smooth_iters: int = 2, max_fill_cells: int = -1):
    """Returns (dsm, confident_mask, support).

    `support` codes every cell: 0 no data, 1 measured (confident stereo), 2 interpolated.
    Interpolated cells are never presented as measurements.

    1. Drop low-confidence cells and isolated spikes (blunders that would otherwise pit the roof).
    2. Interpolate holes inside the covered area, but only up to `max_fill_cells` from a measured
       cell (-1 = unlimited); farther cells stay NoData instead of being invented.
    3. Edge-preserving surface smoothing: flat surfaces (roofs, ground) are de-noised while the
       tall step at a building edge is kept sharp -- this is what makes roofs read flat and edges
       crisp instead of bumpy and fuzzy. `smooth_range` is the height scale (m) treated as "same
       surface"; larger values smooth more but may round off low kerbs.
    """
    conf = (d.score >= min_score) & np.isfinite(d.Z)
    Zc = np.where(conf, d.Z, np.nan).astype(np.float32)
    # progressive spike removal: coarse blunders first, then finer speckle against a wider median
    for thr, r in ((max_dev, 2), (0.6 * max_dev, 2), (0.4 * max_dev, 3)):
        med = _nanmedian_filter(Zc, r)
        spikes = np.isfinite(Zc) & (np.abs(Zc - med) > thr)
        conf &= ~spikes
        Zc[spikes] = np.nan
    filled = fill_holes(Zc)
    keep = d.covered if max_fill_cells < 0 else (d.covered & _near(conf, max_fill_cells))
    filled[~keep] = np.nan
    # edge-preserving smoothing on the covered area
    if smooth_range > 0 and smooth_iters > 0:
        vmask = np.isfinite(filled).astype(np.uint8)
        z = np.nan_to_num(filled, nan=0.0).astype(np.float32)
        for _ in range(smooth_iters):
            z = _mvs.bilateral(np.ascontiguousarray(z), vmask, 3, float(smooth_range), 2.0)
        filled = np.where(keep, z, np.nan).astype(np.float32)
    support = np.where(conf & keep, 1, np.where(np.isfinite(filled), 2, 0)).astype(np.uint8)
    return filled, conf, support
