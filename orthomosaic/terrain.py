"""Digital Terrain Model (bare ground) from a DSM.

Ground classification uses a progressive morphological filter (Zhang et al., 2003) on a
coarse copy of the DSM: the surface is opened (erosion then dilation) with square windows
of growing size, and a cell is flagged as an object (building, tree, car, ...) when it
rises above the opened surface by more than a threshold that grows with the window and the
allowed terrain slope. Objects wider than the largest window are treated as terrain, so set
`max_object_size` above your largest building. The ground cells are then interpolated
smoothly under the objects and refined at full resolution.

    python -m orthomosaic.terrain dsm.tif -o dtm.tif --max-object 80
"""
from __future__ import annotations

import argparse
import logging
import math
from dataclasses import dataclass

import numpy as np

log = logging.getLogger(__name__)


@dataclass
class TerrainOptions:
    max_object_size: float = 60.0   # metres: largest building/tree footprint to remove (its short side)
    slope: float = 0.3              # terrain slope tolerated inside one window (rise / run)
    dh0: float = 0.3                # metres: height above the local ground still counted as ground
    dh_max: float = 3.0             # metres: cap on the growing threshold
    cell: float = 0.0               # working grid cell (m); 0 = automatic (~0.25-1 m)
    refine_tolerance: float = 0.25  # metres: full-res cells this close to the DTM stay ground
    low_outlier: float = 1.0        # metres below the local median that count as a low blunder
    outlier_radius: float = 3.0     # metres: neighbourhood for the low-outlier test
    object_buffer: float = 1.0      # metres: grow detected objects by this before interpolating
    smooth: float = 1.0             # metres: box-smoothing window of the final DTM (0 = off)


# --------------------------------------------------------------------------
# fast running min / max (O(n log w), NaN = +inf for erosion)
# --------------------------------------------------------------------------

def _running(a: np.ndarray, w: int, axis: int, op) -> np.ndarray:
    """Centred running min/max of odd width w along an axis (edges use the partial window)."""
    if w <= 1:
        return a
    r = w // 2
    fill = np.inf if op is np.minimum else -np.inf
    pad = [(0, 0)] * a.ndim
    pad[axis] = (r, r)
    p = np.pad(a, pad, constant_values=fill)
    n = a.shape[axis]
    # doubling: m[i] = op over p[i : i + L]
    L, m = 1, p
    while 2 * L <= w:
        m = _shift_op(m, L, axis, op)
        L *= 2
    # window w = [i, i + w): combine two overlapping power-of-two windows
    first = np.take(m, np.arange(0, n), axis=axis)
    second = np.take(m, np.arange(w - L, w - L + n), axis=axis)
    return op(first, second)


def _shift_op(m, L, axis, op):
    n = m.shape[axis]
    a = np.take(m, np.arange(0, n - L), axis=axis)
    b = np.take(m, np.arange(L, n), axis=axis)
    return op(a, b)


def _erode(z, w):
    return _running(_running(z, w, 0, np.minimum), w, 1, np.minimum)


def _dilate(z, w):
    return _running(_running(z, w, 0, np.maximum), w, 1, np.maximum)


# --------------------------------------------------------------------------
# DTM
# --------------------------------------------------------------------------

def _downsample_low(Z, f, q=25.0):
    """Block low percentile (ignoring NaN). The minimum would be the classic ground candidate,
    but photogrammetric noise makes it biased low; a low percentile is robust."""
    H, W = Z.shape
    h, w = -(-H // f), -(-W // f)
    p = np.full((h * f, w * f), np.nan, np.float32)
    p[:H, :W] = Z
    with np.errstate(all="ignore"):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            blocks = p.reshape(h, f, w, f).transpose(0, 2, 1, 3).reshape(h, w, f * f)
            return np.nanpercentile(blocks, q, axis=2).astype(np.float32)


def classify_ground(Z: np.ndarray, cell: float, opt: TerrainOptions) -> np.ndarray:
    """Progressive morphological filter on a (coarse) grid. Returns a bool ground mask."""
    valid = np.isfinite(Z)
    z = np.where(valid, Z, np.nanmax(Z) if valid.any() else 0.0).astype(np.float64)
    # NaN holes: fill with a high value so they never pull the opening down
    ground = valid.copy()
    max_w = max(3, int(math.ceil(opt.max_object_size / cell)) | 1)
    windows, k = [], 1
    while True:
        w = 2 * k + 1
        windows.append(min(w, max_w))
        if w >= max_w:
            break
        k *= 2
    prev_w = 1
    for w in windows:
        opened = _dilate(_erode(z, w), w)
        dh = min(opt.dh_max, opt.dh0 + opt.slope * (w - prev_w) * cell)
        ground &= (z - opened) <= dh
        z = opened
        prev_w = w
    return ground


def dtm_from_dsm(dsm: np.ndarray, gsd: float, opt: TerrainOptions = None):
    """Bare-ground DTM from a DSM grid (NaN = no data).

    Returns (dtm, ground_mask) at the DSM resolution; dtm is NaN where the DSM is."""
    from .mvs import fill_holes
    opt = opt or TerrainOptions()
    valid = np.isfinite(dsm)
    if not valid.any():
        return np.full_like(dsm, np.nan), np.zeros(dsm.shape, bool)
    cell = opt.cell or float(np.clip(opt.max_object_size / 120.0, 0.25, 1.0))
    f = max(1, int(round(cell / gsd)))
    cell = f * gsd
    Zc = _downsample_low(dsm.astype(np.float32), f)
    # low outliers (stereo blunders, pits) would be kept - and spread - by the morphological
    # opening, dragging the terrain down: drop cells far below their neighbourhood first
    from .mvs import _nanmedian_filter
    r = max(2, int(round(opt.outlier_radius / cell)))
    med = _nanmedian_filter(Zc, r)
    Zc = np.where(Zc < med - opt.low_outlier, np.nan, Zc).astype(np.float32)
    gc = classify_ground(Zc, cell, opt)
    # buffer detected objects: stereo "fattening" leaves slightly raised rims around buildings
    # that would otherwise count as ground and pull the terrain up underneath them
    b = int(round(opt.object_buffer / cell))
    if b > 0:
        obj = np.isfinite(Zc) & ~gc
        gc &= ~(_dilate(obj.astype(np.float32), 2 * b + 1) > 0)
    # smooth ground surface on the coarse grid, filled under objects
    ground_c = np.where(gc, Zc, np.nan).astype(np.float32)
    dtm_c = fill_holes(ground_c)
    # back to full resolution
    dtm_up = _resize_bilinear(dtm_c, (dtm_c.shape[0] * f, dtm_c.shape[1] * f))[:dsm.shape[0], :dsm.shape[1]]
    # full-resolution refinement: DSM cells close to the coarse terrain are ground; re-interpolate
    ground = valid & (dsm - dtm_up <= opt.refine_tolerance) & (dsm - dtm_up >= -1.0)
    dtm = fill_holes(np.where(ground, dsm, np.nan).astype(np.float32))
    if opt.smooth > 0:     # terrain is smooth: suppress per-cell matching noise
        from .backend import _box_cumsum
        r = max(1, int(round(opt.smooth / gsd / 2)))
        dtm = _box_cumsum(np, dtm, r) / np.maximum(_box_cumsum(np, np.ones_like(dtm), r), 1e-6)
        dtm = dtm.astype(np.float32)
    # never above the surface -- except at low blunders (the DSM is wrong there, not the terrain)
    surf = np.where(valid & (dsm >= dtm - opt.low_outlier), dsm, np.inf)
    dtm = np.minimum(dtm, surf).astype(np.float32)
    dtm[~valid] = np.nan
    return dtm, ground


def _resize_bilinear(a: np.ndarray, shape) -> np.ndarray:
    """Pixel-centre bilinear resize of a 2D float array to `shape`."""
    H, W = shape
    h, w = a.shape

    def coords(n_out, n_in):
        x = (np.arange(n_out) + 0.5) * (n_in / n_out) - 0.5
        x = np.clip(x, 0, n_in - 1)
        i0 = np.floor(x).astype(np.int64)
        return i0, np.minimum(i0 + 1, n_in - 1), (x - i0).astype(np.float32)

    y0, y1, fy = coords(H, h)
    x0, x1, fx = coords(W, w)
    top = a[y0][:, x0] * (1 - fx) + a[y0][:, x1] * fx
    bot = a[y1][:, x0] * (1 - fx) + a[y1][:, x1] * fx
    return (top * (1 - fy)[:, None] + bot * fy[:, None]).astype(np.float32)


# --------------------------------------------------------------------------
# elevation mapping: ASPRS point classes and contour lines
# --------------------------------------------------------------------------

def classify_surface(dsm: np.ndarray, ground: np.ndarray, ndsm: np.ndarray, gsd: float,
                     building_min: float = 2.5, veg_min: float = 0.4,
                     rough_threshold: float = 0.15) -> np.ndarray:
    #changed here: Pix4D-style point-cloud classification (elevation mapping deliverable).
    """ASPRS-style classes for the dense cloud.

    Ground (2) comes from the morphological DTM filter; above it, flat plateaus are buildings
    (6) and rough surfaces are vegetation (5 high / 3 low).  `rough_threshold` is the local DSM
    standard deviation (m) over ~1 m used to tell a flat roof from a bumpy canopy.
    """
    from .backend import _box_cumsum
    valid = np.isfinite(dsm)
    cls = np.ones(dsm.shape, np.uint8)                          # 1 = unclassified
    hgt = np.where(valid, np.maximum(np.nan_to_num(ndsm), 0.0), 0.0).astype(np.float32)
    r = max(1, int(round(0.5 / max(gsd, 1e-6))))
    cnt = _box_cumsum(np, np.ones_like(hgt), r)
    mean = _box_cumsum(np, hgt, r) / np.maximum(cnt, 1e-6)
    sq = _box_cumsum(np, hgt * hgt, r) / np.maximum(cnt, 1e-6)
    rough = np.sqrt(np.maximum(sq - mean * mean, 0.0))
    low = valid & (hgt > veg_min) & (hgt <= building_min)
    high = valid & (hgt > building_min)
    cls[low] = 3                                                # 3 = low vegetation
    cls[high & (rough > rough_threshold)] = 5                   # 5 = high vegetation
    cls[high & (rough <= rough_threshold)] = 6                  # 6 = building
    if ground is not None:
        cls[ground & valid] = 2                                 # 2 = ground
    return cls


def contour_lines(Z: np.ndarray, gsd: float, minX: float, maxY: float, interval: float,
                  nodata=None):
    #changed here: marching-squares contour lines (Pix4D elevation-mapping deliverable).
    """Contour segments at multiples of `interval` for a height grid.

    Returns a list of ``(level, segs)`` where `segs` is ``(M, 2, 2)`` world-coordinate line
    segments.  `minX`/`maxY` are the world coordinates of the first cell *corner* (the same
    origin the DSM writer uses).  Rows are processed in bands to bound memory.
    """
    z = np.asarray(Z, np.float64)
    if nodata is not None:
        z = np.where(z <= nodata + 1e-6, np.nan, z)
    H, W = z.shape
    if H < 2 or W < 2 or interval <= 0 or not np.isfinite(z).any():
        return []
    Xn = minX + (np.arange(W) + 0.5) * gsd
    Yn = maxY - (np.arange(H) + 0.5) * gsd
    lo, hi = float(np.nanmin(z)), float(np.nanmax(z))
    start = math.floor(lo / interval) * interval
    levels = np.arange(start + interval, hi + 1e-9, interval)
    band = max(64, min(1024, 2_000_000 // max(W, 1)))       # rows per pass (memory bound)
    out = []
    for L in levels:
        seg_list = []
        for r0 in range(0, H - 1, band):
            r1 = min(r0 + band, H - 1)
            a, b = z[r0:r1, :-1], z[r0:r1, 1:]
            c, d = z[r0 + 1:r1 + 1, :-1], z[r0 + 1:r1 + 1, 1:]
            X0g = np.broadcast_to(Xn[:-1], a.shape)
            X1g = np.broadcast_to(Xn[1:], a.shape)
            Y0g = np.broadcast_to(Yn[r0:r1][:, None], a.shape)
            Y1g = np.broadcast_to(Yn[r0 + 1:r1 + 1][:, None], a.shape)
            edges = [(a, b, X0g, Y0g, X1g, Y0g),          # top
                     (b, d, X1g, Y0g, X1g, Y1g),          # right
                     (c, d, X0g, Y1g, X1g, Y1g),          # bottom
                     (a, c, X0g, Y0g, X0g, Y1g)]          # left
            finite = np.isfinite(a) & np.isfinite(b) & np.isfinite(c) & np.isfinite(d)
            cx = np.empty((4,) + a.shape)
            cy = np.empty((4,) + a.shape)
            ck = np.empty((4,) + a.shape, bool)
            for e, (za, zb, xa, ya, xb, yb) in enumerate(edges):
                den = zb - za
                safe = np.where(den == 0, 1.0, den)
                t = np.clip((L - za) / safe, 0.0, 1.0)
                ck[e] = finite & (den != 0) & (((za - L) * (zb - L)) < 0)
                cx[e] = xa + t * (xb - xa)
                cy[e] = ya + t * (yb - ya)
            cnt = ck.sum(0)
            for i in range(4):                            # exactly two crossings -> one segment
                for j in range(i + 1, 4):
                    m = (cnt == 2) & ck[i] & ck[j]
                    if m.any():
                        seg_list.append(np.stack([np.stack([cx[i][m], cy[i][m]], 1),
                                                  np.stack([cx[j][m], cy[j][m]], 1)], 1))
            m4 = cnt == 4                                 # saddle -> two segments
            if m4.any():
                cen = (a + b + c + d) * 0.25
                for m, pairs4 in ((m4 & (cen >= L), ((0, 1), (2, 3))),
                                  (m4 & (cen < L), ((0, 3), (1, 2)))):
                    if not m.any():
                        continue
                    for i, j in pairs4:
                        seg_list.append(np.stack([np.stack([cx[i][m], cy[i][m]], 1),
                                                  np.stack([cx[j][m], cy[j][m]], 1)], 1))
        if seg_list:
            out.append((float(L), np.concatenate(seg_list, 0)))
    return out


# --------------------------------------------------------------------------
# command line: DTM from an existing dsm.tif
# --------------------------------------------------------------------------

def main(argv=None):
    from .geotiff import GeoTIFFWriter, read_geotiff
    ap = argparse.ArgumentParser(prog="python -m orthomosaic.terrain",
                                 description="Bare-ground DTM (and height above ground) from a DSM GeoTIFF")
    ap.add_argument("dsm")
    ap.add_argument("-o", "--output", required=True, help="output dtm.tif")
    ap.add_argument("--max-object", type=float, default=TerrainOptions.max_object_size,
                    help="largest building/tree size to remove, metres (default %(default)s)")
    ap.add_argument("--slope", type=float, default=TerrainOptions.slope)
    ap.add_argument("--ndsm", default=None, help="also write height above ground (DSM - DTM) here")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    dsm, geo = read_geotiff(a.dsm)
    if not geo:
        raise SystemExit("input has no georeferencing (pixel size) -- cannot compute a DTM")
    dtm, ground = dtm_from_dsm(dsm, geo["pixel_size"], TerrainOptions(max_object_size=a.max_object, slope=a.slope))
    for path, arr in [(a.output, dtm)] + ([(a.ndsm, dsm - dtm)] if a.ndsm else []):
        H, W = arr.shape
        w = GeoTIFFWriter(path, W, H, tile=512, kind="float32", nodata=-9999.0, **geo)
        out = np.where(np.isfinite(arr), arr, -9999.0).astype(np.float32)
        for ty in range(0, H, 512):
            for tx in range(0, W, 512):
                blk = out[ty:ty + 512, tx:tx + 512]
                if (blk != -9999.0).any():
                    w.write_tile(tx // 512, ty // 512, w.compress_tile(blk))
        w.close()
        print(path)
    print(f"ground cells: {100 * ground.sum() / max(np.isfinite(dsm).sum(), 1):.1f}%")


if __name__ == "__main__":
    main()
