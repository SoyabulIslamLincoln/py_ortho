"""Automatic thermal -> RGB orthophoto co-registration ("automatic GCPs").

The RGB and thermal blocks are reconstructed separately, each held only by the drone GPS, so the
two orthophotos can disagree by a smooth affine (on 050: 1.5 m shift, ~1 % scale, ~0.8 deg
rotation). Pix4D users remove this by hand: pick distinct features (HVAC units, dishes, roof
corners) in both mosaics and fit the thermal onto the RGB. This module does the same
automatically:

  1. both rasters are resampled onto a common north-up grid and turned into a modality-free
     "structure" image (gradient magnitude of the locally contrast-normalised luminance / value),
     so thermal edges and RGB edges correlate even though their intensities do not;
  2. tiles of the RGB structure are located in the thermal by zero-normalised cross-correlation,
     coarse (large search, 20 cm) then fine (5 cm, around the coarse prediction); each confident,
     distinctive match is a control-point pair (thermal position -> RGB position);
  3. a robust affine (or similarity / translation) fit of the pairs is the correction, written as
     an alignment JSON that thermal_overlay.py loads (pairs included, so they can be reviewed),
     and optionally applied to the thermal rasters.

    python -m orthomosaic.coregister rgb/orthophoto.tif thermal/orthophoto_thermal.tif \
        -o thermal/orthophoto_thermal_coreg.tif --also thermal/orthophoto.tif
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np

log = logging.getLogger(__name__)


@dataclass
class Coregistration:
    matrix: np.ndarray                  # (3, 3) thermal world (x, y) -> RGB-aligned world (x, y)
    model: str                          # affine | similarity | translation
    src: np.ndarray                     # (n, 2) control points in the thermal raster (world)
    dst: np.ndarray                     # (n, 2) the same features in the RGB raster (world)
    inliers: np.ndarray                 # (n,) bool, pairs used by the final fit
    residual_m: np.ndarray              # (n,) distance of each pair after the fit
    shift_before_m: np.ndarray          # (n,) distance of each pair before the fit
    stats: dict = field(default_factory=dict)


# --------------------------------------------------------------------------
# grids and structure images
# --------------------------------------------------------------------------

def _bounds(arr, geo):
    x0, y1 = geo["origin"]
    ps = geo["pixel_size"]
    return x0, y1 - arr.shape[0] * ps, x0 + arr.shape[1] * ps, y1


def _luminance(arr):
    """(value, valid) of a single-band (NaN = no data) or RGB(A) raster, float32."""
    if arr.ndim == 2:
        v = np.isfinite(arr)
        return np.where(v, arr, 0).astype(np.float32), v
    lum = (0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2]).astype(np.float32)
    v = arr[..., 3] > 0 if arr.shape[2] == 4 else np.ones(lum.shape, bool)
    return lum, v


def _to_grid(arr, geo, bounds, gsd):
    """Area-average a raster onto the north-up grid `bounds` with cell `gsd`: (value, valid)."""
    from PIL import Image
    lum, v = _luminance(arr)
    ox, oy = geo["origin"]
    ps = geo["pixel_size"]
    W, H = int((bounds[2] - bounds[0]) / gsd), int((bounds[3] - bounds[1]) / gsd)
    box = [(bounds[0] - ox) / ps, (oy - bounds[3]) / ps,
           (bounds[0] + W * gsd - ox) / ps, (oy - bounds[3] + H * gsd) / ps]
    vf = v.astype(np.float32)
    lum = lum * vf
    # the grid may overhang the raster (search margin): pad with no-data so the box stays inside
    pl, pt = max(0, math.ceil(-box[0])), max(0, math.ceil(-box[1]))
    pr, pb = max(0, math.ceil(box[2] - lum.shape[1])), max(0, math.ceil(box[3] - lum.shape[0]))
    if pl or pt or pr or pb:
        lum = np.pad(lum, ((pt, pb), (pl, pr)))
        vf = np.pad(vf, ((pt, pb), (pl, pr)))
        box = [box[0] + pl, box[1] + pt, box[2] + pl, box[3] + pt]
    num = np.asarray(Image.fromarray(lum, "F").resize((W, H), Image.BOX, box=box))
    den = np.asarray(Image.fromarray(vf, "F").resize((W, H), Image.BOX, box=box))
    valid = den > 0.99
    return np.where(valid, num / np.maximum(den, 1e-6), 0).astype(np.float32), valid


def _box(a, r):
    """Mean over a (2r+1)^2 window (edges replicated), via an integral image."""
    k = 2 * r + 1
    p = np.pad(a.astype(np.float64), r + 1, mode="edge")
    c = p.cumsum(0).cumsum(1)
    s = c[k:, k:] - c[:-k, k:] - c[k:, :-k] + c[:-k, :-k]
    return (s[:a.shape[0], :a.shape[1]] / (k * k)).astype(np.float32)


def structure(value, valid, gsd, norm_m: float = 1.5):
    """Modality-free edge image: gradient magnitude of the locally contrast-normalised value."""
    if not valid.any():
        return np.zeros_like(value)
    lo, hi = np.percentile(value[valid], [1, 99])
    a = np.clip((np.where(valid, value, lo) - lo) / max(hi - lo, 1e-6), 0, 1).astype(np.float32)
    r = max(2, int(round(norm_m / gsd)))
    m = _box(a, r)
    sd = np.sqrt(np.maximum(_box(a * a, r) - m * m, 0))
    a = (a - m) / (sd + 0.02)
    a = _box(a, 1)                                       # 3x3 smoothing before differentiating
    gy, gx = np.gradient(a)
    g = np.hypot(gx, gy)
    inner = _box(valid.astype(np.float32), max(2, int(round(0.5 / gsd)))) > 0.999
    g[~inner] = 0
    return g


# --------------------------------------------------------------------------
# tile matching
# --------------------------------------------------------------------------

def _zncc(window, tmpl):
    """ZNCC of `tmpl` (h, w) at every offset inside `window` (h + 2s, w + 2s): (2s+1, 2s+1)."""
    h, w = tmpl.shape
    t = tmpl - tmpl.mean()
    nt = math.sqrt(float((t * t).sum()))
    if nt < 1e-6:
        return None
    Hs, Ws = window.shape
    F = np.fft.rfft2(window, (Hs, Ws))
    T = np.fft.rfft2(t, (Hs, Ws))
    corr = np.fft.irfft2(F * np.conj(T), (Hs, Ws))[:Hs - h + 1, :Ws - w + 1]
    c1 = np.pad(window.astype(np.float64), ((1, 0), (1, 0))).cumsum(0).cumsum(1)
    c2 = np.pad(window.astype(np.float64) ** 2, ((1, 0), (1, 0))).cumsum(0).cumsum(1)

    def wsum(c):
        return c[h:, w:] - c[:-h, w:] - c[h:, :-w] + c[:-h, :-w]

    n = h * w
    var = np.maximum(wsum(c2) - wsum(c1) ** 2 / n, 0)
    return corr / (np.sqrt(var) * nt + 1e-9)


def _peak(ncc, exclude: int = 3):
    """Sub-pixel peak (dy, dx) of a correlation surface, its score and distinctiveness."""
    y, x = np.unravel_index(int(np.argmax(ncc)), ncc.shape)
    if y in (0, ncc.shape[0] - 1) or x in (0, ncc.shape[1] - 1):
        return None

    def sub(a, b, c):
        d = a - 2 * b + c
        return 0.0 if abs(d) < 1e-12 else float(0.5 * (a - c) / d)

    best = float(ncc[y, x])
    py = y + sub(ncc[y - 1, x], ncc[y, x], ncc[y + 1, x])
    px = x + sub(ncc[y, x - 1], ncc[y, x], ncc[y, x + 1])
    rest = ncc.copy()
    rest[max(0, y - exclude):y + exclude + 1, max(0, x - exclude):x + exclude + 1] = -1
    return py, px, best, best - float(rest.max())


def match_tiles(ref, mov, gsd, tile_m, search_m, predict=None, min_score=0.25, min_distinct=0.03,
                min_structure=0.2, workers=None):
    """Locate tiles of `ref` in `mov` (same grid). `predict(xy_px) -> (n, 2)` gives the expected
    displacement in pixels (col, row); the search is centred there. Returns (n, 6) rows of
    (col, row, dcol, drow, score, distinctiveness) for confident matches; displacement d means the
    feature at ref (col, row) is found at mov (col + dcol, row + drow)."""
    t = max(8, int(round(tile_m / gsd)))
    s = max(3, int(round(search_m / gsd)))
    H, W = ref.shape
    centres = [(x0 + t / 2, y0 + t / 2) for y0 in range(0, H - t + 1, t // 2) for x0 in range(0, W - t + 1, t // 2)]
    if not centres:
        return np.zeros((0, 6))
    centres = np.array(centres)
    pred = np.zeros_like(centres) if predict is None else predict(centres)
    ref_level = float(np.percentile(ref[ref > 0], 50)) if (ref > 0).any() else 0.0

    def job(k):
        cx, cy = centres[k]
        x0, y0 = int(cx - t / 2), int(cy - t / 2)
        tmpl = ref[y0:y0 + t, x0:x0 + t]
        # skip featureless or partially covered tiles: a match there is a guess
        if (tmpl > 0).mean() < 0.85 or tmpl.mean() < min_structure * ref_level:
            return None
        px, py = int(round(pred[k, 0])), int(round(pred[k, 1]))
        wx0, wy0 = x0 + px - s, y0 + py - s
        if wx0 < 0 or wy0 < 0 or wx0 + t + 2 * s > W or wy0 + t + 2 * s > H:
            return None
        win = mov[wy0:wy0 + t + 2 * s, wx0:wx0 + t + 2 * s]
        if (win > 0).mean() < 0.85:
            return None
        ncc = _zncc(win, tmpl)
        pk = None if ncc is None else _peak(ncc)
        if pk is None or pk[2] < min_score or pk[3] < min_distinct:
            return None
        return cx, cy, pk[1] - s + px, pk[0] - s + py, pk[2], pk[3]

    with ThreadPoolExecutor(workers or os.cpu_count() or 1) as ex:      # numpy FFTs release the GIL
        rows = [r for r in ex.map(job, range(len(centres))) if r is not None]
    return np.array(rows, np.float64).reshape(-1, 6)


# --------------------------------------------------------------------------
# robust transform fit
# --------------------------------------------------------------------------

def fit_transform(src, dst, model: str = "affine", iters: int = 6, floor: float = 0.05):
    """Robust least-squares src -> dst fit (3x3) with iterative MAD trimming. Returns (M, inliers)."""
    src, dst = np.asarray(src, np.float64), np.asarray(dst, np.float64)
    c = src.mean(0)
    s, d = src - c, dst - c
    keep = np.ones(len(s), bool)
    A = np.eye(3)
    for _ in range(iters):
        ss, dd = s[keep], d[keep]
        A = np.eye(3)
        if model == "translation":
            A[:2, 2] = np.median(dd - ss, 0)
        elif model == "similarity":
            zs, zd = ss[:, 0] + 1j * ss[:, 1], dd[:, 0] + 1j * dd[:, 1]
            ms, md = zs.mean(), zd.mean()
            a = np.sum((zd - md) * np.conj(zs - ms)) / max(np.sum(np.abs(zs - ms) ** 2), 1e-12)
            b = md - a * ms
            A = np.array([[a.real, -a.imag, b.real], [a.imag, a.real, b.imag], [0, 0, 1]])
        else:
            sol = np.linalg.lstsq(np.c_[ss, np.ones(len(ss))], dd, rcond=None)[0]
            A[:2] = sol.T
        e = np.linalg.norm(s @ A[:2, :2].T + A[:2, 2] - d, axis=1)
        new = e < max(3.0 * 1.4826 * float(np.median(e[keep])), floor)
        if new.sum() < {"translation": 1, "similarity": 2}.get(model, 3) or np.array_equal(new, keep):
            break
        keep = new
    T = np.array([[1, 0, c[0]], [0, 1, c[1]], [0, 0, 1]], np.float64)
    Ti = np.array([[1, 0, -c[0]], [0, 1, -c[1]], [0, 0, 1]], np.float64)
    return T @ A @ Ti, keep


def describe(M) -> dict:
    """Scale / rotation / shear / shift of a 3x3 horizontal affine, for logs and reports."""
    L = M[:2, :2]
    return dict(scale_x=float(L[0, 0]), scale_y=float(L[1, 1]), shear=[float(L[0, 1]), float(L[1, 0])],
                rotation_deg=float(math.degrees(math.atan2(L[1, 0] - L[0, 1], L[0, 0] + L[1, 1]))),
                singular_values=[float(v) for v in np.linalg.svd(L, compute_uv=False)])


# --------------------------------------------------------------------------
# driver
# --------------------------------------------------------------------------

def coregister(rgb, rgb_geo, thermal, thermal_geo, model: str = "affine", gsd: float = 0.05,
               max_shift_m: float = 8.0, tile_m: float = 6.0, search_m: float = 1.0,
               coarse_gsd: float = 0.2) -> Coregistration:
    """Fit the thermal orthophoto onto the RGB orthophoto (arrays as returned by read_geotiff)."""
    if rgb_geo.get("epsg") and thermal_geo.get("epsg") and rgb_geo["epsg"] != thermal_geo["epsg"]:
        raise ValueError(f"CRS differ: RGB EPSG:{rgb_geo['epsg']}, thermal EPSG:{thermal_geo['epsg']}")
    b1, b2 = _bounds(rgb, rgb_geo), _bounds(thermal, thermal_geo)
    m = max_shift_m
    # thermal grid padded by the search range, so features near the RGB edge can still be found
    bounds = (max(b1[0], b2[0] - m), max(b1[1], b2[1] - m), min(b1[2], b2[2] + m), min(b1[3], b2[3] + m))
    if bounds[2] - bounds[0] < 4 * tile_m or bounds[3] - bounds[1] < 4 * tile_m:
        raise ValueError("the RGB and thermal orthophotos barely overlap")

    def level(g):
        rv, rok = _to_grid(rgb, rgb_geo, bounds, g)
        tv, tok = _to_grid(thermal, thermal_geo, bounds, g)
        return structure(rv, rok, g), structure(tv, tok, g)

    def to_world(rows, g):
        """match rows on grid g -> (thermal world, rgb world) point arrays."""
        x = bounds[0] + rows[:, 0] * g
        y = bounds[3] - rows[:, 1] * g
        dst = np.c_[x, y]
        src = np.c_[x + rows[:, 2] * g, y - rows[:, 3] * g]
        return src, dst

    # 1. global shift: the whole RGB structure image (minus the search margin) inside the thermal
    ref, mov = level(coarse_gsd)
    s = int(round(max_shift_m / coarse_gsd))
    ncc = _zncc(mov, ref[s:-s, s:-s])
    pk = None if ncc is None else _peak(ncc, exclude=max(3, int(round(1.0 / coarse_gsd))))
    if pk is None:
        raise RuntimeError("no global RGB/thermal correlation (shift beyond --max-shift, or no overlap)")
    M = np.eye(3)
    M[:2, 2] = [-(pk[1] - s) * coarse_gsd, (pk[0] - s) * coarse_gsd]   # thermal -> RGB
    log.info("Co-registration global shift %.2f, %.2f m (score %.2f)", M[0, 2], M[1, 2], pk[2])

    def predictor(M, g):
        Mi = np.linalg.inv(M)

        def predict(c):
            x, y = bounds[0] + c[:, 0] * g, bounds[3] - c[:, 1] * g
            p = np.c_[x, y] @ Mi[:2, :2].T + Mi[:2, 2]           # RGB position -> thermal position
            return np.c_[(p[:, 0] - x) / g, -(p[:, 1] - y) / g]
        return predict

    # 2. medium: 2x tiles at 2x the cell, search wide enough for the scale / rotation left over
    g2 = 2 * gsd
    ref, mov = level(g2)
    rows = match_tiles(ref, mov, g2, tile_m=2 * tile_m, search_m=max(2.5, 2.5 * search_m), predict=predictor(M, g2))
    if len(rows) >= 6:
        src, dst = to_world(rows, g2)
        M, _ = fit_transform(src, dst, "affine", floor=0.15)
        log.info("Co-registration medium: %d tiles, %s", len(rows), describe(M))

    # 3. fine: small tiles, short search around where the medium fit puts each feature
    ref, mov = level(gsd)
    rows = match_tiles(ref, mov, gsd, tile_m=tile_m, search_m=search_m, predict=predictor(M, gsd))
    need = {"translation": 3, "similarity": 4}.get(model, 6)
    if len(rows) < need:
        raise RuntimeError(f"only {len(rows)} confident fine matches between RGB and thermal")
    src, dst = to_world(rows, gsd)
    M, keep = fit_transform(src, dst, model)
    res = np.linalg.norm(src @ M[:2, :2].T + M[:2, 2] - dst, axis=1)
    before = np.linalg.norm(src - dst, axis=1)
    stats = dict(pairs=int(len(src)), inliers=int(keep.sum()),
                 shift_before_median_m=float(np.median(before)), shift_before_max_m=float(before.max()),
                 residual_median_m=float(np.median(res[keep])),
                 residual_rms_m=float(np.sqrt(np.mean(res[keep] ** 2))),
                 residual_p90_m=float(np.percentile(res[keep], 90)), **describe(M))
    log.info("Co-registration (%s): %d/%d pairs, misplacement median %.3f m -> residual median %.3f m "
             "(rms %.3f m), scale %.4f / %.4f, rotation %+.3f deg", model, keep.sum(), len(src),
             stats["shift_before_median_m"], stats["residual_median_m"], stats["residual_rms_m"],
             stats["scale_x"], stats["scale_y"], stats["rotation_deg"])
    return Coregistration(M, model, src, dst, keep, res, before, stats)


def write_alignment(path, result: Coregistration, rgb_path=None, thermal_path=None, epsg=None):
    """Alignment JSON in thermal_overlay.py's format (fit_matrix = the automatic fit, the inlier
    pairs listed so they can be reviewed / pruned and re-fitted in the viewer)."""
    d = dict(
        note="matrix maps the thermal raster's georeferenced (x, y) to the corrected (x, y), CRS units; "
             "matrix = fit_matrix @ manual (manual = stretch / rotate / shift in the thermal's own frame "
             "about its coverage centre)",
        order="fit@manual", rgb=rgb_path, thermal=thermal_path, epsg=epsg,
        matrix=result.matrix.tolist(), fit_matrix=result.matrix.tolist(),
        manual=dict(dx=0.0, dy=0.0, rotation_deg=0.0, scale_x_pct=100.0, scale_y_pct=100.0),
        pairs=[dict(src=s.tolist(), dst=t.tolist())
               for s, t, k in zip(result.src, result.dst, result.inliers) if k],
        auto=dict(model=result.model, **result.stats),
    )
    with open(path, "w") as fh:
        json.dump(d, fh, indent=2)


def warp_raster(src_path, out_path, matrix, tile: int = 512):
    """Resample a (thermal) GeoTIFF through `matrix` (its world -> corrected world) onto a north-up
    grid of the same cell size; float rasters keep NaN as no-data, RGBA stays RGBA."""
    from .geotiff import GeoTIFFWriter, read_geotiff
    from .reconstruct import _warp_tile
    arr, geo = read_geotiff(src_path)
    ps = geo["pixel_size"]
    x0, y0, x1, y1 = _bounds(arr, geo)
    corners = np.array([[x0, y0], [x1, y0], [x0, y1], [x1, y1]]) @ matrix[:2, :2].T + matrix[:2, 2]
    nx0, ny1 = corners[:, 0].min(), corners[:, 1].max()
    W = int(math.ceil((corners[:, 0].max() - nx0) / ps))
    H = int(math.ceil((ny1 - corners[:, 1].min()) / ps))
    # output (col, row) -> output world -> source world -> source (col, row)
    Mi = np.linalg.inv(matrix)
    out = np.array([[ps, 0, nx0 + 0.5 * ps], [0, -ps, ny1 - 0.5 * ps], [0, 0, 1]])
    pix = np.array([[1 / ps, 0, -geo["origin"][0] / ps - 0.5], [0, -1 / ps, geo["origin"][1] / ps - 0.5], [0, 0, 1]])
    A = (pix @ Mi @ out)[:2]
    if arr.ndim == 2:
        kind, nodata, fill = "float32", float("nan"), float("nan")
        arr = arr.astype(np.float32, copy=False)
    else:
        kind, nodata, fill = "rgba", None, 0
        if arr.shape[2] == 3:
            arr = np.dstack([arr, np.full(arr.shape[:2], 255, np.uint8)])
    w = GeoTIFFWriter(out_path, W, H, tile=tile, kind=kind, nodata=nodata, epsg=geo.get("epsg"),
                      origin=(nx0, ny1), pixel_size=ps)

    def job(t):
        tx, ty = t
        blk = _warp_tile(arr, A, tx, ty, min(tile, W - tx), min(tile, H - ty), kind, fill)
        if kind == "rgba" and not blk[..., 3].any():
            return None
        if kind == "float32" and np.isnan(blk).all():
            return None
        return w.compress_tile(blk)

    tiles = [(tx, ty) for ty in range(0, H, tile) for tx in range(0, W, tile)]
    with ThreadPoolExecutor(os.cpu_count() or 1) as ex:
        for (tx, ty), data in zip(tiles, ex.map(job, tiles)):
            if data is not None:
                w.write_tile(tx // tile, ty // tile, data)
    w.close()
    return out_path


def coregister_files(rgb_path, thermal_path, output: Optional[str] = None, also: Sequence[str] = (),
                     alignment: Optional[str] = None, **kw) -> Coregistration:
    """File-level driver: fit, write the alignment JSON, warp the thermal raster(s)."""
    from .geotiff import read_geotiff
    rgb, rgb_geo = read_geotiff(rgb_path)
    th, th_geo = read_geotiff(thermal_path)
    if not rgb_geo or not th_geo:
        raise ValueError("both orthophotos must be georeferenced GeoTIFFs")
    result = coregister(rgb, rgb_geo, th, th_geo, **kw)
    del rgb, th
    stem = os.path.splitext(output or thermal_path)[0]
    alignment = alignment or (stem.removesuffix("_coreg") + "_coreg_alignment.json")
    write_alignment(alignment, result, os.path.abspath(rgb_path), os.path.abspath(thermal_path), th_geo.get("epsg"))
    log.info("Alignment -> %s", alignment)
    if output:
        warp_raster(thermal_path, output, result.matrix)
        log.info("Co-registered thermal -> %s", output)
    for p in also:
        q = os.path.splitext(p)[0] + "_coreg.tif"
        warp_raster(p, q, result.matrix)
        log.info("Co-registered %s -> %s", os.path.basename(p), q)
    return result


def main(argv=None):
    ap = argparse.ArgumentParser(prog="orthomosaic-coregister",
                                 description="Fit a thermal orthophoto onto an RGB orthophoto automatically "
                                             "(feature matching instead of hand-picked GCPs).")
    ap.add_argument("rgb", help="RGB orthophoto GeoTIFF (the reference)")
    ap.add_argument("thermal", help="thermal orthophoto GeoTIFF to move (the raw-value orthophoto_thermal.tif)")
    ap.add_argument("-o", "--output", default=None, help="co-registered thermal GeoTIFF to write")
    ap.add_argument("--also", nargs="*", default=[],
                    help="more rasters on the thermal block to move the same way (e.g. the colour orthophoto.tif); "
                         "each is written next to its source as <name>_coreg.tif")
    ap.add_argument("--alignment", default=None, help="alignment JSON (default: <output>_alignment.json)")
    ap.add_argument("--model", choices=["affine", "similarity", "translation"], default="affine")
    ap.add_argument("--gsd", type=float, default=0.05, help="matching grid in metres (default %(default)s)")
    ap.add_argument("--max-shift", type=float, default=8.0, help="largest expected misplacement in metres")
    ap.add_argument("--tile", type=float, default=6.0, help="matching tile size in metres")
    ap.add_argument("-v", "--verbose", action="store_true")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if a.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    r = coregister_files(a.rgb, a.thermal, a.output, a.also, a.alignment, model=a.model, gsd=a.gsd,
                         max_shift_m=a.max_shift, tile_m=a.tile)
    s = r.stats
    print(f"{s['inliers']}/{s['pairs']} control points; misplacement median {s['shift_before_median_m']:.3f} m "
          f"(max {s['shift_before_max_m']:.3f}) -> residual median {s['residual_median_m']:.3f} m, "
          f"p90 {s['residual_p90_m']:.3f} m; scale {s['scale_x']:.4f} / {s['scale_y']:.4f}, "
          f"rotation {s['rotation_deg']:+.3f} deg")


if __name__ == "__main__":
    main()
