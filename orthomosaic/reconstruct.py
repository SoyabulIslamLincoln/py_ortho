"""End-to-end 2.5D reconstruction: SfM -> dense DSM -> true orthophoto, point clouds, mesh.

    python -m orthomosaic.reconstruct <image_dir> -o <out_dir>
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import logging
import math
import os
import time
import warnings
import dataclasses
import shutil
from dataclasses import asdict, dataclass, field
from typing import Optional, Sequence, Union

import numpy as np
from PIL import Image

from . import export, mvs, sfm
from .align import solve_gains
from .geotiff import GeoTIFFWriter
from .pipeline import Options, align_images

log = logging.getLogger(__name__)
NODATA = -9999.0


@dataclass
class Options3D(Options):
    feature_max_dim: int = 4096             # detect tie-point features at full resolution (Pix4D "full" keypoint scale):
                                            # half resolution tilted a low-altitude block by ~2 deg in tests
    n_features: int = 0                     # 0 = auto, ~1250 per megapixel (15k on 12 MP; 2D default 5000)
    neighbors: int = 0                      # 0 = auto: images overlapping >= ~50%, 6..12 per image
    alt_sigma: float = 0.5                  # DJI relative-altitude accuracy (m)
    refine_focal: Optional[bool] = None     # None: refine when an altitude reference exists
    dsm_resolution: Optional[float] = None  # metres per DSM cell; default = native GSD  #changed here
    max_views: int = 6
    min_score: float = 0.5
    ncc_window: int = 3
    dsm_smooth: float = 0.45                 # edge-preserving surface smoothing height scale (m); 0 = off
    dsm_smooth_iters: int = 2
    tile: int = 160
    cloud_step: int = 1                     # keep every n-th confident DSM cell in the dense cloud
    mesh_max_vertices: int = 600_000        # Poisson budget (time and RAM grow with it; 8 GB machines)
    texture_max: int = 8192
    formats: tuple = ("ply", "laz", "obj", "glb")   # laz falls back to las without laspy[lazrs]
    # densification (Pix4D stage 2): "depthmap" = per-image depth maps fused into a 3D dense cloud,
    # the DSM is rasterised from that cloud; "sweep" = legacy ground-grid height sweep (2.5D only,
    # used automatically for radiometric thermal, whose low texture suits its SGM regularisation)
    dense_method: str = "depthmap"
    depth_max_image: int = 1600             # matching image long side (~Pix4D "1/2 image scale")
    depth_neighbors: int = 4
    depth_min_views: int = 0                # images that must agree on a dense point; 0 = auto (3, or 2 for low overlap)
    depth_patchmatch: bool = False          # slanted-plane PatchMatch repair of inconsistent pixels (slow on CPU)
    dsm_auto_factor: float = 0.75           # auto DSM cell = this x dense point spacing (when dsm_resolution is None)
    ortho_resolution: Optional[float] = None  # orthophoto cell (m); None = native GSD (Pix4D 1 x GSD)
    ortho_max_cells: float = 0              # orthophoto pixel cap; 0 = auto from the machine's RAM (outputs are
                                            # disk-backed, ~6 bytes/pixel stay in memory)
    dem_gapfill_steps: int = 3              # ODM radius steps: point spacing * sqrt(2)^k, k < steps
    attitude_sigma_deg: float = 2.0         # gimbal pitch/roll prior in the BA (0 = off); fixes block tilt
    ignore_gsd: bool = False                # ODM: never allow a DSM/ortho finer than GSD * (1 - 10%)
    mesh_method: str = "auto"               # "auto": Poisson from the dense cloud (needs open3d), else DSM grid
    # digital terrain model (bare ground)
    dtm: bool = True
    dtm_max_object: float = 60.0            # metres: larger than your largest building's short side
    dtm_slope: float = 0.3                  # terrain slope tolerated by the ground filter
    #changed here: Pix4D-grade true orthophoto + elevation-mapping controls
    true_ortho: bool = True                 # re-render the ortho on the final DSM (occlusion-aware)
    occlusion: bool = True                  # drop views hidden behind buildings/trees
    occlusion_tol: float = 0.20             # metres a ray may pass under the surface before it blocks
    occlusion_steps: int = 96               # max line-of-sight samples per ray (1 DSM cell apart when possible)
    occlusion_stride: int = 2               # test occlusion on every n-th cell, then grow the mask
    ortho_tile: int = 512                   # true-orthophoto tile size (cells)
    view_angle_power: float = 1.5           # nadir preference when blending the true orthophoto
    ortho_views: int = 8                    # orthophoto: max photos per tile, from the global (nadir-first) source map
    fill_hidden: bool = True                # orthophoto: colour cells hidden in all chosen views from the best view
    ortho_blend: str = "seam"               # "seam": one source per cell + seam optimisation; "feather": average
    source_weights: tuple = (1.0, 0.5, 0.7, 0.3)   # angle, resolution, border distance, exposure
    seam_smoothness: float = 0.6            # weight of the seam (colour-disagreement) term
    seam_iters: int = 4
    seam_band: int = 3                      # cells blended on each side of a seam
    dsm_max_fill: float = -1.0              # metres a hole may be interpolated from measured cells;
                                            # -1 = fill every hole photographed by >= 2 cameras (as 0.3.3 / Pix4D);
                                            # filled cells are flagged 2 (interpolated) in dsm_support.tif
    color_balance: bool = True              # per-image gain+offset radiometric correction (Pix4D)
    contour_interval: float = 0.0           # metres between terrain contour lines (0 = off)
    contour_resolution: float = 1.0         # m: contours are traced on the DTM resampled to this (Pix4D: 100 cm)
    classify_cloud: bool = True             # ASPRS ground/building/vegetation classes in the LAS
    # ground control points (survey anchors)
    gcp: Optional[str] = None               # path to a WebODM/Pix4D GCP list file
    gcp_sigma: float = 0.05                 # surveyed GCP accuracy in metres


def _colorize(Z, valid, gsd=1.0):
    """Pix4D-style quick look: green -> yellow -> orange -> brown elevation ramp, multiplied by a
    hillshade (sun from the north-west, 45 deg, slopes in metres) so edges, cars and roof units
    read as relief."""
    z = np.where(valid, Z, np.nan)
    lo, hi = np.nanpercentile(z, [1, 99]) if np.isfinite(z).any() else (0, 1)
    zf = np.nan_to_num(z, nan=lo)
    t = np.clip((zf - lo) / max(hi - lo, 1e-6), 0, 1)
    stops = np.array([[0.00, 0.47, 0.00], [0.31, 0.75, 0.00], [0.90, 0.86, 0.00], [0.86, 0.51, 0.12],
                      [0.67, 0.24, 0.08]])
    pos = t * (len(stops) - 1)
    i = np.clip(pos.astype(int), 0, len(stops) - 2)
    f = (pos - i)[..., None]
    rgb = stops[i] * (1 - f) + stops[i + 1] * f
    gy, gx = np.gradient(zf, gsd)
    slope = np.arctan(np.hypot(gx, gy))
    aspect = np.arctan2(-gx, gy)
    az, alt = np.radians(315.0), np.radians(45.0)
    hs = np.sin(alt) * np.cos(slope) + np.cos(alt) * np.sin(slope) * np.cos(az - aspect)
    shade = 0.35 + 0.75 * np.clip(hs, 0, 1)
    out = np.clip(rgb * shade[..., None] * 255, 0, 255).astype(np.uint8)
    out[~valid] = 255                       # white outside the footprint (even where alpha is ignored)
    alpha = (valid * 255).astype(np.uint8)
    return np.dstack([out, alpha])


def _write_raster(path, arr, kind, epsg, origin_xy, gsd, nodata=None, tile=512):
    H, W = arr.shape[:2]
    geo = dict(epsg=epsg, origin=origin_xy, pixel_size=gsd)
    w = GeoTIFFWriter(path, W, H, tile=tile, kind=kind, nodata=nodata, **geo)

    def job(t):
        tx, ty = t
        blk = arr[ty:ty + tile, tx:tx + tile]
        if kind == "float32":
            blk = blk.astype(np.float32, copy=False)          # int rasters convert per tile, not whole
        if kind == "rgba" and not blk[..., 3].any():
            return None
        if kind == "float32" and nodata is not None and np.all(blk == nodata):
            return None
        return w.compress_tile(blk)                 # zlib releases the GIL: tiles compress in parallel

    tiles = [(tx, ty) for ty in range(0, H, tile) for tx in range(0, W, tile)]
    with ThreadPoolExecutor(os.cpu_count() or 1) as ex:
        for (tx, ty), data in zip(tiles, ex.map(job, tiles)):
            if data is not None:
                w.write_tile(tx // tile, ty // tile, data)
    w.close()


def _block_reduce(a, f, valid):
    H, W = a.shape
    h, w = H // f, W // f
    a = np.where(valid, a, 0.0)[:h * f, :w * f].reshape(h, f, w, f)
    v = valid[:h * f, :w * f].reshape(h, f, w, f)
    cnt = v.sum((1, 3))
    return np.where(cnt > 0, a.sum((1, 3)) / np.maximum(cnt, 1), np.nan), cnt >= (f * f) / 2


def _ortho_cell_budget(fraction: float = 0.15, bytes_per_cell: float = 6.0) -> float:
    """Orthophoto cells that fit in `fraction` of physical RAM (the RGBA/source rasters are disk
    backed; the DSM resample, masks and per-tile buffers stay in memory)."""
    try:
        ram = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except (ValueError, OSError, AttributeError):
        ram = 8e9
    return max(60e6, fraction * ram / bytes_per_cell)


def cap_resolution(requested, gsd: float, ignore_gsd: bool = False, error: float = 0.1) -> float:
    """ODM's gsd.cap_resolution: the output cell may not be finer than GSD * (1 - error)."""
    if requested is None:
        return gsd
    floor = gsd * (1.0 - error)
    if not ignore_gsd and requested < floor:
        log.warning("Requested resolution %.2f cm is finer than the GSD allows; capped to %.2f cm "
                    "(GSD - %d%%). Use ignore_gsd=True to force it.", requested * 100, floor * 100, int(error * 100))
        return floor
    return requested


def las_scale(spacing: Optional[float]) -> float:
    """ODM: LAS coordinate scale = a tenth of the point spacing (rounded to a power of 10), max 1 mm."""
    if not spacing or spacing <= 0:
        return 0.001
    return min(10 ** round(math.log10(spacing)) / 10, 0.001)


def _resample_dsm(dsm, gsd, Wo, Ho, o_gsd, step=0.3, rows=512):
    """DSM -> finer orthophoto grid (same top-left corner). Bilinear on smooth surfaces; at height
    steps (> `step` m within the 2x2 neighbourhood) or next to NoData the nearest cell, so roof
    edges stay sharp instead of becoming ramps."""
    H, W = dsm.shape
    out = np.empty((Ho, Wo), np.float32)
    xs = np.clip((np.arange(Wo) + 0.5) * o_gsd / gsd - 0.5, 0, W - 1)
    x0 = np.minimum(np.floor(xs).astype(np.int64), W - 2) if W > 1 else np.zeros(Wo, np.int64)
    fx = (xs - x0).astype(np.float32)
    xn = np.clip(np.floor((np.arange(Wo) + 0.5) * o_gsd / gsd).astype(np.int64), 0, W - 1)
    for r0 in range(0, Ho, rows):
        r1 = min(Ho, r0 + rows)
        ys = np.clip((np.arange(r0, r1) + 0.5) * o_gsd / gsd - 0.5, 0, H - 1)
        y0 = np.minimum(np.floor(ys).astype(np.int64), H - 2) if H > 1 else np.zeros(r1 - r0, np.int64)
        fy = (ys - y0).astype(np.float32)[:, None]
        yn = np.clip(np.floor((np.arange(r0, r1) + 0.5) * o_gsd / gsd).astype(np.int64), 0, H - 1)
        a = dsm[y0][:, x0]
        b = dsm[y0][:, np.minimum(x0 + 1, W - 1)]
        c = dsm[np.minimum(y0 + 1, H - 1)][:, x0]
        d = dsm[np.minimum(y0 + 1, H - 1)][:, np.minimum(x0 + 1, W - 1)]
        bil = (a * (1 - fx) + b * fx) * (1 - fy) + (c * (1 - fx) + d * fx) * fy
        stack = np.stack([a, b, c, d])
        with np.errstate(invalid="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)          # all-NaN corners at the edge
            rough = ~np.isfinite(bil) | ((np.nanmax(stack, 0) - np.nanmin(stack, 0)) > step)
        near = dsm[yn][:, xn]
        out[r0:r1] = np.where(rough, near, bil)
    return out


def _load_gcps(ar, opt):
    if not opt.gcp:
        return None
    from .gcp import load_gcps, to_local
    if not ar.alignment.georeferenced:
        log.warning("GCPs given but the survey is not georeferenced (no GPS); ignoring them")
        return None
    gset = load_gcps(opt.gcp)
    return to_local(gset, ar.epsg % 100, ar.epsg < 32700, ar.origin)


def sparse_block(images: Union[str, Sequence[str]], opt: Options3D):
    """Aerial triangulation: align + SfM bundle adjustment (with GCPs if given).
    Returns (AlignResult, Reconstruction)."""
    ar = align_images(images, opt)
    gcps = _load_gcps(ar, opt)
    rec = sfm.reconstruct(ar, gps_sigma=opt.gps_sigma, alt_sigma=opt.alt_sigma,
                          rolling_shutter=opt.rolling_shutter, rolling_shutter_readout=opt.rolling_shutter_readout,
                          attitude_sigma_deg=opt.attitude_sigma_deg,
                          refine_focal=opt.refine_focal, gcps=gcps, gcp_sigma=opt.gcp_sigma)
    return ar, rec


def build_3d(images: Union[str, Sequence[str]], out_dir: str, options: Optional[Options3D] = None,
             pre: Optional[tuple] = None) -> dict:
    """Full 2.5D reconstruction. `pre` = (AlignResult, Reconstruction) reuses an existing block
    (e.g. a rig-bound thermal reconstruction) instead of aligning and triangulating again."""
    opt = options or Options3D()
    t0 = time.time()
    os.makedirs(out_dir, exist_ok=True)
    ar, rec = pre if pre is not None else sparse_block(images, opt)
    return _products(ar, rec, out_dir, opt, t0)


def _products(ar, rec, out_dir, opt, t0) -> dict:
    al = ar.alignment
    georef = bool(al.georeferenced)
    epsg = ar.epsg if georef else None
    ox, oy = ar.origin if georef else (0.0, 0.0)
    frames = ar.frames
    offset3 = (ox, oy, 0.0)
    export.write_ply(os.path.join(out_dir, "sparse.ply"), rec.X, rec.color, offset3, "sparse SfM points")
    rel = [frames[i].rel_alt for i in rec.used]
    absa = [frames[i].abs_alt for i in rec.used]
    z_abs_offset = (float(np.median([a - r for a, r in zip(absa, rel)]))
                    if all(v is not None for v in rel + absa) else None)
    cams = {}
    for k, i in enumerate(rec.used):
        it = rec.intr[rec.cam_group[k]]
        cams[frames[i].name] = dict(R=rec.R[k].tolist(), C=(rec.C[k] + np.array(offset3)).tolist(),
                                    f=it.f, k1=it.k1, k2=it.k2, k3=it.k3, cx=it.cx, cy=it.cy,
                                    width=it.width, height=it.height)

    # ---- dense
    thermal = ar.thermal_range is not None          # radiometric thermal: never exposure-compensate
    #changed here: Pix4D radiometric colour balancing -- per-image gain AND offset solved jointly
    #over the matched point colours (falls back to gains-only when disabled).
    if opt.exposure_compensation and not thermal:
        if getattr(opt, "color_balance", True):
            from .color import solve_radiometric
            gains, biases = solve_radiometric(al.used, ar.pairs)
        else:
            gains, biases = solve_gains(al.used, ar.pairs), {}
    else:
        gains, biases = {}, {}
    res = cap_resolution(opt.dsm_resolution, al.gsd, opt.ignore_gsd)
    cloud = None
    if opt.dense_method == "depthmap" and not thermal:
        from .densify import DepthOptions, densify, rasterize
        cloud = densify(ar, rec, gains, biases, out_dir,
                        DepthOptions(max_image_dim=opt.depth_max_image, neighbors=opt.depth_neighbors,
                                     min_views=opt.depth_min_views, patchmatch=opt.depth_patchmatch,
                                     workers=ar.workers))
        from .densify import camera_coverage, radius_steps
        if opt.dsm_resolution is None:
            # one DSM cell per ~dense point: every cell can be measured instead of interpolated
            res = cap_resolution(max(al.gsd, opt.dsm_auto_factor * cloud.spacing), al.gsd, opt.ignore_gsd)
            log.info("DSM resolution (auto): %.3f m from point spacing %.3f m (GSD %.3f m)", res, cloud.spacing, al.gsd)
        minX, maxY, W, H, gsd, _ = mvs.grid_extent(rec, res, al.gsd)
        radii = radius_steps(max(cloud.spacing, gsd), opt.dem_gapfill_steps)
        log.info("Dense cloud -> DSM: %d x %d cells at %.3f m (top-layer median per cell, radius steps %s m)",
                 W, H, gsd, ", ".join(f"{r:.3f}" for r in radii))
        # holes may only be filled where >= 2 calibrated photos actually see the ground
        dense = rasterize(cloud, minX, maxY, W, H, gsd, radii=radii,
                          covered=camera_coverage(rec, minX, maxY, W, H, gsd, min_views=2))
        # products cover the reconstructed area only (ODM crops to the point cloud's hull); the
        # hull has straight edges, so the DSM and ortho borders are clean lines
        from .densify import cloud_footprint
        dense.covered = cloud_footprint(dense.score >= 1, gsd)
        log.info("Reconstructed footprint: %.0f%% of the photographed grid", 100 * dense.covered.mean())
    else:
        dopt = mvs.DenseOptions(gsd=res, max_views=opt.max_views, min_score=opt.min_score,
                                window=opt.ncc_window, tile=opt.tile, cache_mb=opt.cache_mb, workers=ar.workers,
                                view_angle_power=opt.view_angle_power)
        dense = mvs.dense_reconstruct(ar, rec, gains, al.gsd, dopt, biases=biases)
    gsd = dense.gsd
    max_fill = -1 if opt.dsm_max_fill < 0 else int(math.ceil(opt.dsm_max_fill / gsd))
    dsm, conf, support = mvs.postprocess(dense, opt.min_score, smooth_range=opt.dsm_smooth,
                                         smooth_iters=opt.dsm_smooth_iters, max_fill_cells=max_fill)
    if cloud is not None:
        # ODM median-smooths its DEM (radius 4 cells at 5 cm, i.e. ~0.2 m wide): removes the
        # isolated 10-30 cm spikes that otherwise print as dots in the orthophoto, while a median
        # keeps building edges sharp. Radius from the cell size, not a fixed number of cells.
        r = max(1, int(round(0.1 / gsd)))
        valid = np.isfinite(dsm)
        dsm = np.where(valid, mvs._nanmedian_filter(dsm, r), np.nan).astype(np.float32)
        log.info("DSM median smoothing: %d-cell radius (%.2f m)", r, r * gsd)
    covered = np.isfinite(dsm)
    if covered.any():
        # every raster covers the reconstructed footprint's bounding box, not the whole camera
        # extent: smaller files, less memory, and the orthophoto can afford the native GSD
        rows, cols_ = np.flatnonzero(covered.any(1)), np.flatnonzero(covered.any(0))
        r0, r1 = max(0, rows[0] - 2), min(dsm.shape[0], rows[-1] + 3)
        c0, c1 = max(0, cols_[0] - 2), min(dsm.shape[1], cols_[-1] + 3)
        if (r1 - r0) * (c1 - c0) < dsm.size:
            cut = (slice(r0, r1), slice(c0, c1))
            dsm, conf, support, covered = dsm[cut], conf[cut], support[cut], covered[cut]
            dense = dataclasses.replace(
                dense, Z=dense.Z[cut], score=dense.score[cut], rgb=dense.rgb[cut], covered=dense.covered[cut],
                stepped=None if dense.stepped is None else dense.stepped[cut],
                minX=dense.minX + c0 * gsd, maxY=dense.maxY - r0 * gsd)
    origin_xy = (dense.minX + ox, dense.maxY + oy)

    #changed here: Pix4D "Orthomosaic" stage -- re-render the colours on the FINAL DSM with
    #occlusion handling (views hidden behind a building are dropped), view-angle weighting and
    #seam feathering.  This is what turns the dense colours into a true orthophoto.
    colors = dense.rgb
    ortho_frac = None
    src_id = count = rgba_o = ortho_tmp = None
    ortho_valid = covered              # cells of the orthomosaic that carry colour (alpha)
    o_gsd, dsm_o, covered_o = gsd, dsm, covered            # orthophoto grid (same as DSM unless finer)
    if getattr(opt, "true_ortho", True) and not thermal and covered.any():
        from .ortho import true_orthophoto
        t1 = time.time()
        o_gsd = cap_resolution(opt.ortho_resolution or al.gsd, al.gsd, opt.ignore_gsd)
        Wo = int(math.ceil(dsm.shape[1] * gsd / o_gsd))
        Ho = int(math.ceil(dsm.shape[0] * gsd / o_gsd))
        cap = opt.ortho_max_cells or _ortho_cell_budget()
        if Wo * Ho > cap:
            o_gsd = math.sqrt(dsm.shape[0] * dsm.shape[1] * gsd * gsd / cap)
            Wo = int(math.ceil(dsm.shape[1] * gsd / o_gsd))
            Ho = int(math.ceil(dsm.shape[0] * gsd / o_gsd))
        if o_gsd < 0.99 * gsd:
            dsm_o = _resample_dsm(dsm, gsd, Wo, Ho, o_gsd)
            covered_o = np.isfinite(dsm_o)
            log.info("Orthophoto grid %d x %d at %.3f m (DSM %d x %d at %.3f m)", Wo, Ho, o_gsd,
                     dsm.shape[1], dsm.shape[0], gsd)
        else:
            o_gsd = gsd
        rgba_o, src_id, count, ortho_tmp = true_orthophoto(ar, rec, dsm_o, dense.minX, dense.maxY, o_gsd,
                                                           gains, biases, opt)
        # Only cells a calibrated photo actually sees are coloured. Cells hidden in every photo
        # stay transparent (NoData): they cannot be recovered, and filling them with the dense
        # matching colours would paint walls/roofs onto the hidden ground.
        colors = rgba_o[..., :3]
        ortho_valid = rgba_o[..., 3] > 0
        if ortho_valid.any():
            ortho_frac = float(ortho_valid.sum() / max(covered_o.sum(), 1))
            log.info("True orthophoto: %.0f%% of DSM cells coloured only from visible views (%.1fs)",
                     100 * ortho_frac, time.time() - t1)

    # ---- thermal: the dense colours are raw values (gray); keep them and colour with the palette
    if thermal:
        from .thermal import legend, palette_lut
        lo, hi = ar.thermal_range
        raw = np.where(covered, lo + dense.rgb[..., 0].astype(np.float32) * ((hi - lo) / 255.0), np.nan)
        _write_raster(os.path.join(out_dir, "orthophoto_thermal.tif"), raw.astype(np.float32), "float32", epsg,
                      origin_xy, gsd, float("nan"))
        colors = palette_lut(opt.palette)[dense.rgb[..., 0]]
        legend(os.path.join(out_dir, "thermal_legend.png"), opt.palette, ar.thermal_range)

    # ---- rasters
    dsm_out = np.where(covered, dsm, NODATA).astype(np.float32)
    _write_raster(os.path.join(out_dir, "dsm.tif"), dsm_out, "float32", epsg, origin_xy, gsd, NODATA)
    # inspection rasters on the same grid: 1/0 validity, support (1 measured, 2 interpolated),
    # chosen source image (index into report["source_images"]) and number of seeing views
    _write_raster(os.path.join(out_dir, "valid_mask.tif"), covered.astype(np.float32), "float32", epsg,
                  origin_xy, gsd)
    # Pix4D overlap map: number of calibrated images that photograph each DSM point
    from .densify import camera_count
    overlap = camera_count(rec, dense.minX, dense.maxY, dsm.shape[1], dsm.shape[0], gsd,
                           step=max(1, int(round(0.5 / gsd))), Z=dsm)
    _write_raster(os.path.join(out_dir, "overlap.tif"), np.where(covered, overlap, 0).astype(np.float32),
                  "float32", epsg, origin_xy, gsd)
    ov_stats = (dict(median=float(np.median(overlap[covered])), fraction_5plus=float((overlap[covered] >= 5).mean()),
                     fraction_3plus=float((overlap[covered] >= 3).mean())) if covered.any() else None)
    if ov_stats:
        log.info("Overlap: median %.0f images per point, %.0f%% of the area seen by 5+ images",
                 ov_stats["median"], 100 * ov_stats["fraction_5plus"])
    _write_raster(os.path.join(out_dir, "dsm_support.tif"), support.astype(np.float32), "float32", epsg,
                  origin_xy, gsd, 0.0)
    if src_id is not None:
        _write_raster(os.path.join(out_dir, "source_image_id.tif"), src_id, "float32",
                      epsg, origin_xy, o_gsd, -1.0)
        _write_raster(os.path.join(out_dir, "coverage_count.tif"), count, "float32",
                      epsg, origin_xy, o_gsd)
    if rgba_o is not None:
        rgba = rgba_o                                    # already RGBA, alpha = coloured by a photo
    else:
        rgba = np.dstack([colors, (ortho_valid * 255).astype(np.uint8)])
        rgba[~ortho_valid, :3] = 0
    _write_raster(os.path.join(out_dir, "orthophoto.tif"), rgba, "rgba", epsg, origin_xy, o_gsd)
    pf = max(1, math.ceil(max(dsm.shape) / 2048))
    Image.fromarray(_colorize(dsm, covered, gsd)[::pf, ::pf]).save(os.path.join(out_dir, "dsm_preview.png"))
    po = max(1, math.ceil(max(rgba.shape[:2]) / 2048))
    prev = Image.fromarray(rgba[::po, ::po], "RGBA")
    bg = Image.new("RGB", prev.size, (255, 255, 255))
    bg.paste(prev, mask=prev.split()[3])
    bg.save(os.path.join(out_dir, "orthophoto_preview.jpg"), quality=90)

    # ---- DTM: bare ground under buildings / vegetation, and height above ground
    dtm, ground, contours, ndsm_raw = None, None, [], None
    if opt.dtm:
        from .terrain import TerrainOptions, dtm_from_dsm, contour_lines
        t = time.time()
        dtm, ground = dtm_from_dsm(dsm, gsd, TerrainOptions(max_object_size=opt.dtm_max_object,
                                                             slope=opt.dtm_slope))
        _write_raster(os.path.join(out_dir, "dtm.tif"), np.where(covered, dtm, NODATA).astype(np.float32),
                      "float32", epsg, origin_xy, gsd, NODATA)
        ndsm_raw = np.where(covered, np.maximum(dsm - dtm, 0.0), np.nan)
        _write_raster(os.path.join(out_dir, "ndsm.tif"), np.where(covered, ndsm_raw, NODATA).astype(np.float32),
                      "float32", epsg, origin_xy, gsd, NODATA)
        Image.fromarray(_colorize(dtm, covered, gsd)[::pf, ::pf]).save(os.path.join(out_dir, "dtm_preview.png"))
        log.info("DTM: %.0f%% of the surface classified as ground (%.1fs)",
                 100 * ground.sum() / max(covered.sum(), 1), time.time() - t)
        #changed here: Pix4D elevation mapping -- contour lines from the bare-earth DTM
        if getattr(opt, "contour_interval", 0.0) and opt.contour_interval > 0:
            cf = max(1, int(round(opt.contour_resolution / gsd)))
            if cf > 1:
                dtm_c, ok_c = _block_reduce(dtm, cf, covered)
                dtm_c = np.where(ok_c, dtm_c, np.nan)
            else:
                dtm_c = np.where(covered, dtm, np.nan)
            contours = contour_lines(dtm_c, gsd * cf, dense.minX, dense.maxY, opt.contour_interval)
            if contours:
                export.write_geojson_contours(os.path.join(out_dir, "contours.geojson"), contours,
                                              offset=(ox, oy), epsg=epsg)
                log.info("Contours: %d levels at %.2f m", len(contours), opt.contour_interval)

    # ---- dense point cloud: the fused 3D cloud (walls included) or, for the sweep, confident DSM cells
    s = max(1, opt.cloud_step)
    if cloud is not None:
        xyz, col = cloud.xyz[::s], cloud.rgb[::s]
        xs = np.clip(((xyz[:, 0] - dense.minX) / gsd).astype(np.int64), 0, dsm.shape[1] - 1)
        ys = np.clip(((dense.maxY - xyz[:, 1]) / gsd).astype(np.int64), 0, dsm.shape[0] - 1)
    else:
        ys, xs = np.nonzero(conf[::s, ::s])
        ys, xs = ys * s, xs * s
        xyz = np.column_stack([dense.minX + (xs + 0.5) * gsd, dense.maxY - (ys + 0.5) * gsd, dense.Z[ys, xs]])
        col = (colors if colors.shape[:2] == dsm.shape else dense.rgb)[ys, xs]
    cls = None                                       #changed here: ASPRS classes (set with the LAS)
    outputs = {"dsm": "dsm.tif", "orthophoto": "orthophoto.tif", "sparse": "sparse.ply",
               "valid_mask": "valid_mask.tif", "dsm_support": "dsm_support.tif", "overlap": "overlap.tif"}
    if src_id is not None:
        outputs.update(source_image_id="source_image_id.tif", coverage_count="coverage_count.tif")
    if dtm is not None:
        outputs.update(dtm="dtm.tif", ndsm="ndsm.tif")
    if contours:                                     #changed here
        outputs["contours"] = "contours.geojson"
    if "ply" in opt.formats:
        export.write_ply(os.path.join(out_dir, "dense.ply"), xyz, col, offset3, f"EPSG:{epsg}" if epsg else "")
        outputs["dense_ply"] = "dense.ply"
    if "las" in opt.formats or "laz" in opt.formats:
        #changed here: Pix4D-style point-cloud classification for the LAS deliverable
        cls = None
        if ground is not None:
            if getattr(opt, "classify_cloud", True) and ndsm_raw is not None:
                from .terrain import classify_surface
                cls = classify_surface(dsm, ground, ndsm_raw, gsd)[ys, xs]
            else:
                cls = np.where(ground[ys, xs], 2, 1).astype(np.uint8)
            if cloud is not None and cls is not None:
                # a point well below the surface of its cell (wall, under canopy) is not ground
                below = xyz[:, 2] < dsm[ys, xs] - 1.0
                cls = np.where(below & (cls == 2), 1, cls).astype(np.uint8)
        las_path = os.path.join(out_dir, "dense.las")
        spacing = cloud.spacing if cloud is not None else gsd * max(1, opt.cloud_step)
        export.write_las(las_path, xyz + np.array(offset3), col, epsg, scale=las_scale(spacing),
                         classification=cls)
        outputs["dense_las"] = "dense.las"
        if "laz" in opt.formats:
            laz = export.las_to_laz(las_path)
            if laz:
                outputs["dense_laz"] = "dense.laz"
                if "las" not in opt.formats:
                    os.remove(las_path)
                    outputs.pop("dense_las")
            else:
                log.warning("LAZ needs `pip install \"laspy[lazrs]\"`; wrote dense.las instead")

    # ---- mesh: Poisson surface from the dense cloud (Pix4D stage 2), else a 2.5D mesh from the DSM
    mesh_info = None
    want_mesh = any(f in opt.formats for f in ("obj", "glb"))
    if want_mesh and cloud is not None and opt.mesh_method in ("auto", "poisson"):
        t = time.time()
        log.info("Meshing the dense cloud (screened Poisson)")
        vdir = rec.C[cloud.cam] - cloud.xyz
        vdir /= np.linalg.norm(vdir, axis=1, keepdims=True)
        m = export.poisson_mesh(cloud.xyz, cloud.rgb, vdir, cloud.spacing,
                                max_points=2 * opt.mesh_max_vertices, max_vertices=opt.mesh_max_vertices)
        if m is None:
            log.warning("Poisson mesh unavailable (install open3d, see above for errors): mesh built from the DSM")
        else:
            V, F, VC = m
            if "obj" in opt.formats:
                export.write_obj_colored(os.path.join(out_dir, "mesh.obj"), V, F, VC, offset3)
                outputs["mesh_obj"] = "mesh.obj"
            if "glb" in opt.formats:
                export.write_glb(os.path.join(out_dir, "mesh.glb"), V, F, None, None, offset3, vertex_rgb=VC)
                outputs["mesh_glb"] = "mesh.glb"
            mesh_info = dict(method="poisson (dense cloud)", vertices=int(len(V)), triangles=int(len(F)))
            log.info("Mesh: %d vertices, %d triangles from the dense cloud (%.1fs)", len(V), len(F), time.time() - t)
            want_mesh = False
    if want_mesh:
        n_valid = int(covered.sum())
        f = max(1, int(math.ceil(math.sqrt(n_valid / max(opt.mesh_max_vertices, 1)))))
        Zm, vm = _block_reduce(dsm, f, covered)
        V, F, UV = export.grid_mesh(Zm, vm & np.isfinite(Zm), dense.minX, dense.maxY, gsd * f)
        th = max(1, math.ceil(max(rgba.shape[:2]) / opt.texture_max))
        # 3D-model texture only: cells no photo sees take the dense-matching colour instead of black
        tex_rgb = (np.where(ortho_valid[..., None], colors, dense.rgb) if colors.shape[:2] == dsm.shape
                   else colors)        # orthophoto on its own (finer) grid: use it directly as texture
        tex = Image.fromarray(np.ascontiguousarray(tex_rgb[::th, ::th]))
        if "obj" in opt.formats:
            export.write_obj(os.path.join(out_dir, "mesh.obj"), V, F, UV, tex, offset3)
            outputs["mesh_obj"] = "mesh.obj"
        if "glb" in opt.formats:
            export.write_glb(os.path.join(out_dir, "mesh.glb"), V, F, UV, tex, offset3)
            outputs["mesh_glb"] = "mesh.glb"
        mesh_info = dict(method="DSM grid (2.5D)", vertices=int(len(V)), triangles=int(len(F)))
        log.info("Mesh: %d vertices, %d triangles at %.3f m", len(V), len(F), gsd * f)

    report = dict(
        images_total=len(frames), images_used=len(rec.used), images_dropped=ar.dropped,
        backend=ar.backend.name, georeferenced=georef, epsg=epsg, origin=[ox, oy],
        z_datum=rec.stats.get("z_datum"), z_to_absolute_offset=z_abs_offset,
        sfm=rec.stats,
        overlap=ov_stats,
        dense=dict(method="depth maps -> fused 3D cloud -> DSM" if cloud is not None else "ground-grid height sweep",
                   points=int(len(cloud.xyz)) if cloud is not None else None,
                   median_views=int(np.median(cloud.views)) if cloud is not None else None,
                   min_views=opt.depth_min_views if cloud is not None else None,
                   point_spacing_m=float(cloud.spacing) if cloud is not None else None),
        mesh=mesh_info,
        dsm=dict(gsd=gsd, width=int(dsm.shape[1]), height=int(dsm.shape[0]),
                 bounds=[origin_xy[0], origin_xy[1] - dsm.shape[0] * gsd, origin_xy[0] + dsm.shape[1] * gsd,
                         origin_xy[1]],
                 confident_fraction=float(conf.sum() / max(covered.sum(), 1)),
                 measured_cells=int((support == 1).sum()), interpolated_cells=int((support == 2).sum()),
                 max_fill_m=opt.dsm_max_fill, requested_resolution=opt.dsm_resolution,
                 native_gsd=float(al.gsd), gapfill_steps=opt.dem_gapfill_steps,
                 z_range=[float(np.nanpercentile(dsm, 1)), float(np.nanpercentile(dsm, 99))] if covered.any() else None),
        dtm=(dict(ground_fraction=float(ground.sum() / max(covered.sum(), 1)),
                  max_object=opt.dtm_max_object,
                  z_range=[float(np.nanpercentile(dtm, 1)), float(np.nanpercentile(dtm, 99))])
             if dtm is not None and covered.any() else None),
        #changed here: report the Pix4D-style ortho / colour / elevation-mapping settings and results
        ortho=dict(true_ortho=bool(getattr(opt, "true_ortho", True)),
                   occlusion=bool(getattr(opt, "occlusion", True)),
                   view_angle_power=float(getattr(opt, "view_angle_power", 1.5)),
                   visible_fraction=ortho_frac, resolution=float(o_gsd),
                   width=int(colors.shape[1]), height=int(colors.shape[0]), hidden_cells_transparent=True, blend=opt.ortho_blend,
                   source_weights=list(opt.source_weights),
                   coverage_median=float(np.median(count[covered_o])) if count is not None and covered_o.any() else None,
                   single_view_fraction=float((count[covered_o] == 1).mean()) if count is not None and covered_o.any() else None),
        source_images=[frames[i].name for i in rec.used],
        color=dict(balancing=bool(getattr(opt, "color_balance", True)),
                   gain_median=[float(np.median([g[0] for g in gains.values()])) if gains else 1.0,
                                float(np.median([g[1] for g in gains.values()])) if gains else 1.0,
                                float(np.median([g[2] for g in gains.values()])) if gains else 1.0]),
        contours=dict(interval=float(getattr(opt, "contour_interval", 0.0)), levels=len(contours)) if contours else None,
        classification=dict(ground=int((cls == 2).sum()) if cls is not None else 0,
                            building=int((cls == 6).sum()) if cls is not None else 0,
                            vegetation=int(((cls == 3) | (cls == 5)).sum()) if cls is not None else 0,
                            classes="ASPRS 2 ground / 3 low veg / 5 high veg / 6 building")
        if cls is not None else None,
        dense_points=int(len(xyz)), outputs=outputs, seconds=round(time.time() - t0, 1),
        thermal=(dict(raw_range=list(ar.thermal_range), palette=opt.palette, raw_values="orthophoto_thermal.tif",
                      legend="thermal_legend.png") if thermal else None),
        options={k: (list(v) if isinstance(v, tuple) else v) for k, v in asdict(opt).items()},
        cameras=cams,
    )
    with open(os.path.join(out_dir, "report.json"), "w") as fh:
        json.dump(report, fh, indent=1)
    if ortho_tmp:
        shutil.rmtree(ortho_tmp, ignore_errors=True)     # disk-backed orthophoto buffers
    log.info("3D done in %.1fs -> %s", time.time() - t0, out_dir)
    return report


def build_thermal_bound(rgb_images, thermal_images, out_dir, options: Optional[Options3D] = None,
                        thermal_options: Optional[Options3D] = None) -> dict:
    """RGB-driven thermal binding: aerial-triangulate the RGB block, then place the thermal images
    from their RGB twins through a shared rig, so the thermal DSM/orthophoto inherit RGB accuracy
    and are co-registered with the RGB products.

    Writes RGB products to <out_dir>/rgb and bound thermal products to <out_dir>/thermal.
    """
    from .binding import bind_thermal
    opt = options or Options3D()
    topt = thermal_options or Options3D(backend=opt.backend, workers=opt.workers, gps_sigma=opt.gps_sigma,
                                        feature_max_dim=4000, n_features=8000, neighbors=16, min_inliers=15,
                                        exposure_compensation=False, palette=opt.palette, gcp=opt.gcp,
                                        gcp_sigma=opt.gcp_sigma, dsm_resolution=opt.dsm_resolution)
    t0 = time.time()
    os.makedirs(out_dir, exist_ok=True)

    log.info("== RGB block (aerial triangulation) ==")
    ar_rgb, rec_rgb = sparse_block(rgb_images, opt)
    rgb_report = build_3d(None, os.path.join(out_dir, "rgb"), opt, pre=(ar_rgb, rec_rgb))

    log.info("== Thermal block ==")
    ar_th, rec_th = sparse_block(thermal_images, topt)

    log.info("== Binding thermal to RGB ==")
    rec_bound, rig = bind_thermal(rec_rgb, ar_rgb.frames, rec_th, ar_th.frames)
    if rig["translation_scatter_m"] > 0.5 or rig["rotation_scatter_deg"] > 2.0:
        log.warning("Rig is inconsistent (rotation scatter %.2f deg, translation scatter %.2f m): the two "
                    "cameras may not be rigidly mounted, or a pose set is noisy", rig["rotation_scatter_deg"],
                    rig["translation_scatter_m"])
    th_report = build_3d(None, os.path.join(out_dir, "thermal"), topt, pre=(ar_th, rec_bound))
    th_report["rig"] = rig

    report = dict(seconds=round(time.time() - t0, 1), rig=rig,
                  rgb=dict(dir="rgb", **{k: rgb_report[k] for k in ("images_used", "epsg", "dsm")}),
                  thermal=dict(dir="thermal", **{k: th_report[k] for k in ("images_used", "epsg", "dsm")}))
    with open(os.path.join(out_dir, "binding_report.json"), "w") as fh:
        json.dump(report, fh, indent=1)
    log.info("Thermal binding done in %.1fs: rig baseline %.3f m, rotation %.2f deg -> %s",
             time.time() - t0, rig["rig_baseline_m"], max(abs(v) for v in rig["rig_rotation_deg"]), out_dir)
    return report


def main(argv=None):
    d = Options3D()
    ap = argparse.ArgumentParser(prog="orthomosaic-3d",
                                 description="2.5D reconstruction (DSM, true orthophoto, point cloud, mesh) "
                                             "from nadir drone images.")
    ap.add_argument("images", help="folder containing the images")
    ap.add_argument("-o", "--output", default="reconstruction", help="output folder")
    ap.add_argument("--backend", choices=["auto", "cuda", "mps", "cpu"], default=d.backend)
    ap.add_argument("--workers", type=int, default=d.workers)
    ap.add_argument("--dsm-resolution", type=float, default=None,
                    help="DSM cell size in metres (default: native GSD, Pix4D-grade)")   #changed here
    ap.add_argument("--gps-sigma", type=float, default=d.gps_sigma)
    ap.add_argument("--max-views", type=int, default=d.max_views)
    ap.add_argument("--min-score", type=float, default=d.min_score, help="NCC needed for a dense point (0-1)")
    #changed here: Pix4D-grade true-orthophoto / colour / elevation-mapping switches
    ap.add_argument("--no-true-ortho", action="store_true",
                    help="keep the in-sweep colours instead of re-rendering an occlusion-aware true orthophoto")
    ap.add_argument("--no-occlusion", action="store_true",
                    help="disable occlusion handling in the true orthophoto (faster)")
    ap.add_argument("--no-color-balance", action="store_true",
                    help="use gains-only exposure compensation (no per-image offset)")
    ap.add_argument("--view-angle-power", type=float, default=d.view_angle_power,
                    help="nadir preference when blending the true orthophoto (0 = off)")
    ap.add_argument("--ortho-blend", choices=["seam", "feather"], default=d.ortho_blend,
                    help="seam: one source image per cell with optimised seams (no ghosting); "
                         "feather: weighted average of all visible views")
    ap.add_argument("--seam-smoothness", type=float, default=d.seam_smoothness)
    ap.add_argument("--dsm-max-fill", type=float, default=d.dsm_max_fill,
                    help="metres a DSM hole may be interpolated from measurements (-1 = fill all)")
    ap.add_argument("--dem-gapfill-steps", type=int, default=d.dem_gapfill_steps,
                    help="ODM radius steps for filling DSM cells from nearby points")
    ap.add_argument("--ignore-gsd", action="store_true", help="allow a resolution finer than the GSD")
    ap.add_argument("--sky-removal", action="store_true", help="AI sky masks for oblique images (onnxruntime)")
    ap.add_argument("--bg-removal", action="store_true", help="AI background masks (onnxruntime)")
    ap.add_argument("--rolling-shutter", action="store_true", help="correct electronic (rolling) shutter distortion")
    ap.add_argument("--rolling-shutter-readout", type=float, default=0.0, help="sensor readout time in ms (0 = database)")
    ap.add_argument("--contours", type=float, default=d.contour_interval,
                    help="terrain contour interval in metres (0 = off)")
    ap.add_argument("--no-classify", action="store_true", help="skip ASPRS cloud classification")
    ap.add_argument("--cloud-step", type=int, default=d.cloud_step, help="thin the dense cloud (every n-th cell)")
    ap.add_argument("--mesh-max-vertices", type=int, default=d.mesh_max_vertices)
    ap.add_argument("--formats", default=",".join(d.formats), help="comma list of ply,las,laz,obj,glb")
    ap.add_argument("--dense-method", choices=["depthmap", "sweep"], default=d.dense_method,
                    help="depthmap: per-image depth maps fused into a 3D cloud, DSM from the cloud (Pix4D); "
                         "sweep: legacy 2.5D height sweep")
    ap.add_argument("--depth-max-image", type=int, default=d.depth_max_image,
                    help="long side (px) of the images used for depth maps")
    ap.add_argument("--depth-min-views", type=int, default=d.depth_min_views,
                    help="images that must agree on a dense point")
    ap.add_argument("--no-dtm", action="store_true", help="skip the bare-ground DTM")
    ap.add_argument("--dtm-max-object", type=float, default=d.dtm_max_object,
                    help="largest building/tree size (m) the ground filter removes (default %(default)s)")
    ap.add_argument("--cache-mb", type=int, default=d.cache_mb)
    ap.add_argument("--palette", default=d.palette, help="thermal palette (rainbow, iron, white_hot, ...)")
    ap.add_argument("--no-thermal", action="store_true", help="ignore radiometric data; use JPEG colours")
    ap.add_argument("--gcp", default=None, help="GCP list file (WebODM/Pix4D format) to anchor the block")
    ap.add_argument("--gcp-sigma", type=float, default=d.gcp_sigma, help="surveyed GCP accuracy in metres")
    ap.add_argument("--bind-thermal", default=None, metavar="THERMAL_IMAGES",
                    help="RGB-driven thermal binding: `images` is the RGB set, this is the thermal set; "
                         "the thermal block is placed from the RGB poses via a shared rig")
    ap.add_argument("-v", "--verbose", action="store_true")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if a.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    opt = Options3D(backend=a.backend, workers=a.workers, dsm_resolution=a.dsm_resolution, gps_sigma=a.gps_sigma,
                    max_views=a.max_views, min_score=a.min_score, cloud_step=a.cloud_step,
                    mesh_max_vertices=a.mesh_max_vertices, cache_mb=a.cache_mb, palette=a.palette,
                    thermal="off" if a.no_thermal else "auto", dtm=not a.no_dtm, dtm_max_object=a.dtm_max_object,
                    gcp=a.gcp, gcp_sigma=a.gcp_sigma,
                    #changed here: Pix4D-grade ortho / colour / elevation-mapping switches
                    true_ortho=not a.no_true_ortho, occlusion=not a.no_occlusion,
                    color_balance=not a.no_color_balance, view_angle_power=a.view_angle_power,
                    dem_gapfill_steps=a.dem_gapfill_steps, ignore_gsd=a.ignore_gsd,
                    sky_removal=a.sky_removal, bg_removal=a.bg_removal, rolling_shutter=a.rolling_shutter,
                    rolling_shutter_readout=a.rolling_shutter_readout, dense_method=a.dense_method, depth_max_image=a.depth_max_image,
                    depth_min_views=a.depth_min_views, ortho_blend=a.ortho_blend, seam_smoothness=a.seam_smoothness, dsm_max_fill=a.dsm_max_fill,
                    contour_interval=a.contours, classify_cloud=not a.no_classify,
                    formats=tuple(f.strip() for f in a.formats.split(",") if f.strip()))
    if a.bind_thermal:
        build_thermal_bound(a.images, a.bind_thermal, a.output, opt)
    else:
        build_3d(a.images, a.output, opt)


if __name__ == "__main__":
    main()
