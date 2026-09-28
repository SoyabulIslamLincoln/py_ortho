"""End-to-end 2.5D reconstruction: SfM -> dense DSM -> true orthophoto, point clouds, mesh.

    python -m orthomosaic.reconstruct <image_dir> -o <out_dir>
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import os
import time
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
    neighbors: int = 10
    alt_sigma: float = 0.5                  # DJI relative-altitude accuracy (m)
    refine_focal: Optional[bool] = None     # None: refine when an altitude reference exists
    dsm_resolution: Optional[float] = None  # metres per DSM cell; default 2 x native GSD
    max_views: int = 6
    min_score: float = 0.5
    ncc_window: int = 3
    tile: int = 160
    cloud_step: int = 1                     # keep every n-th confident DSM cell in the dense cloud
    mesh_max_vertices: int = 1_500_000
    texture_max: int = 8192
    formats: tuple = ("ply", "las", "obj", "glb")
    # digital terrain model (bare ground)
    dtm: bool = True
    dtm_max_object: float = 60.0            # metres: larger than your largest building's short side
    dtm_slope: float = 0.3                  # terrain slope tolerated by the ground filter
    # ground control points (survey anchors)
    gcp: Optional[str] = None               # path to a WebODM/Pix4D GCP list file
    gcp_sigma: float = 0.05                 # surveyed GCP accuracy in metres


def _colorize(Z, valid):
    """Hillshaded colour relief for a quick-look PNG."""
    z = np.where(valid, Z, np.nan)
    lo, hi = np.nanpercentile(z, [1, 99]) if np.isfinite(z).any() else (0, 1)
    t = np.clip((np.nan_to_num(z, nan=lo) - lo) / max(hi - lo, 1e-6), 0, 1)
    stops = np.array([[0.19, 0.30, 0.58], [0.20, 0.60, 0.55], [0.55, 0.75, 0.30], [0.93, 0.80, 0.35],
                      [0.80, 0.40, 0.25], [0.95, 0.95, 0.95]])
    pos = t * (len(stops) - 1)
    i = np.clip(pos.astype(int), 0, len(stops) - 2)
    f = (pos - i)[..., None]
    rgb = stops[i] * (1 - f) + stops[i + 1] * f
    gy, gx = np.gradient(np.nan_to_num(z, nan=lo))
    shade = np.clip(0.75 + 0.9 * (-gx - gy) / (np.hypot(gx, gy) + 1.0) * 0.5, 0.35, 1.2)
    out = np.clip(rgb * shade[..., None] * 255, 0, 255).astype(np.uint8)
    alpha = (valid * 255).astype(np.uint8)
    return np.dstack([out, alpha])


def _write_raster(path, arr, kind, epsg, origin_xy, gsd, nodata=None, tile=512):
    H, W = arr.shape[:2]
    geo = dict(epsg=epsg, origin=origin_xy, pixel_size=gsd)
    w = GeoTIFFWriter(path, W, H, tile=tile, kind=kind, nodata=nodata, **geo)
    for ty in range(0, H, tile):
        for tx in range(0, W, tile):
            blk = arr[ty:ty + tile, tx:tx + tile]
            if kind == "rgba" and not blk[..., 3].any():
                continue
            if kind == "float32" and nodata is not None and np.all(blk == nodata):
                continue
            w.write_tile(tx // tile, ty // tile, w.compress_tile(blk))
    w.close()


def _block_reduce(a, f, valid):
    H, W = a.shape
    h, w = H // f, W // f
    a = np.where(valid, a, 0.0)[:h * f, :w * f].reshape(h, f, w, f)
    v = valid[:h * f, :w * f].reshape(h, f, w, f)
    cnt = v.sum((1, 3))
    return np.where(cnt > 0, a.sum((1, 3)) / np.maximum(cnt, 1), np.nan), cnt >= (f * f) / 2


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
                                    f=it.f, k1=it.k1, k2=it.k2, cx=it.cx, cy=it.cy,
                                    width=it.width, height=it.height)

    # ---- dense
    thermal = ar.thermal_range is not None          # radiometric thermal: never exposure-compensate
    gains = solve_gains(al.used, ar.pairs) if (opt.exposure_compensation and not thermal) else {}
    dopt = mvs.DenseOptions(gsd=opt.dsm_resolution, max_views=opt.max_views, min_score=opt.min_score,
                            window=opt.ncc_window, tile=opt.tile, cache_mb=opt.cache_mb, workers=ar.workers)
    dense = mvs.dense_reconstruct(ar, rec, gains, al.gsd, dopt)
    dsm, conf = mvs.postprocess(dense, opt.min_score)
    gsd = dense.gsd
    covered = np.isfinite(dsm)
    origin_xy = (dense.minX + ox, dense.maxY + oy)

    # ---- thermal: the dense colours are raw values (gray); keep them and colour with the palette
    colors = dense.rgb
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
    rgba = np.dstack([colors, (covered * 255).astype(np.uint8)])
    rgba[~covered, :3] = 0
    _write_raster(os.path.join(out_dir, "orthophoto.tif"), rgba, "rgba", epsg, origin_xy, gsd)
    pf = max(1, math.ceil(max(dsm.shape) / 2048))
    Image.fromarray(_colorize(dsm, covered)[::pf, ::pf]).save(os.path.join(out_dir, "dsm_preview.png"))
    prev = Image.fromarray(rgba[::pf, ::pf], "RGBA")
    bg = Image.new("RGB", prev.size, (255, 255, 255))
    bg.paste(prev, mask=prev.split()[3])
    bg.save(os.path.join(out_dir, "orthophoto_preview.jpg"), quality=90)

    # ---- DTM: bare ground under buildings / vegetation, and height above ground
    dtm, ground = None, None
    if opt.dtm:
        from .terrain import TerrainOptions, dtm_from_dsm
        t = time.time()
        dtm, ground = dtm_from_dsm(dsm, gsd, TerrainOptions(max_object_size=opt.dtm_max_object,
                                                             slope=opt.dtm_slope))
        _write_raster(os.path.join(out_dir, "dtm.tif"), np.where(covered, dtm, NODATA).astype(np.float32),
                      "float32", epsg, origin_xy, gsd, NODATA)
        ndsm = np.where(covered, np.maximum(dsm - dtm, 0.0), NODATA).astype(np.float32)
        _write_raster(os.path.join(out_dir, "ndsm.tif"), ndsm, "float32", epsg, origin_xy, gsd, NODATA)
        Image.fromarray(_colorize(dtm, covered)[::pf, ::pf]).save(os.path.join(out_dir, "dtm_preview.png"))
        log.info("DTM: %.0f%% of the surface classified as ground (%.1fs)",
                 100 * ground.sum() / max(covered.sum(), 1), time.time() - t)

    # ---- dense point cloud (confident cells only)
    s = max(1, opt.cloud_step)
    ys, xs = np.nonzero(conf[::s, ::s])
    ys, xs = ys * s, xs * s
    xyz = np.column_stack([dense.minX + (xs + 0.5) * gsd, dense.maxY - (ys + 0.5) * gsd, dense.Z[ys, xs]])
    col = colors[ys, xs]
    outputs = {"dsm": "dsm.tif", "orthophoto": "orthophoto.tif", "sparse": "sparse.ply"}
    if dtm is not None:
        outputs.update(dtm="dtm.tif", ndsm="ndsm.tif")
    if "ply" in opt.formats:
        export.write_ply(os.path.join(out_dir, "dense.ply"), xyz, col, offset3, f"EPSG:{epsg}" if epsg else "")
        outputs["dense_ply"] = "dense.ply"
    if "las" in opt.formats:
        cls = (np.where(ground[ys, xs], 2, 1).astype(np.uint8) if ground is not None else None)
        export.write_las(os.path.join(out_dir, "dense.las"), xyz + np.array(offset3), col, epsg, classification=cls)
        outputs["dense_las"] = "dense.las"

    # ---- mesh from the DSM
    if any(f in opt.formats for f in ("obj", "glb")):
        n_valid = int(covered.sum())
        f = max(1, int(math.ceil(math.sqrt(n_valid / max(opt.mesh_max_vertices, 1)))))
        Zm, vm = _block_reduce(dsm, f, covered)
        V, F, UV = export.grid_mesh(Zm, vm & np.isfinite(Zm), dense.minX, dense.maxY, gsd * f)
        th = max(1, math.ceil(max(rgba.shape[:2]) / opt.texture_max))
        tex = Image.fromarray(np.ascontiguousarray(colors[::th, ::th]))
        if "obj" in opt.formats:
            export.write_obj(os.path.join(out_dir, "mesh.obj"), V, F, UV, tex, offset3)
            outputs["mesh_obj"] = "mesh.obj"
        if "glb" in opt.formats:
            export.write_glb(os.path.join(out_dir, "mesh.glb"), V, F, UV, tex, offset3)
            outputs["mesh_glb"] = "mesh.glb"
        log.info("Mesh: %d vertices, %d triangles at %.3f m", len(V), len(F), gsd * f)

    report = dict(
        images_total=len(frames), images_used=len(rec.used), images_dropped=ar.dropped,
        backend=ar.backend.name, georeferenced=georef, epsg=epsg, origin=[ox, oy],
        z_datum=rec.stats.get("z_datum"), z_to_absolute_offset=z_abs_offset,
        sfm=rec.stats,
        dsm=dict(gsd=gsd, width=int(dsm.shape[1]), height=int(dsm.shape[0]),
                 bounds=[origin_xy[0], origin_xy[1] - dsm.shape[0] * gsd, origin_xy[0] + dsm.shape[1] * gsd,
                         origin_xy[1]],
                 confident_fraction=float(conf.sum() / max(covered.sum(), 1)),
                 z_range=[float(np.nanpercentile(dsm, 1)), float(np.nanpercentile(dsm, 99))] if covered.any() else None),
        dtm=(dict(ground_fraction=float(ground.sum() / max(covered.sum(), 1)),
                  max_object=opt.dtm_max_object,
                  z_range=[float(np.nanpercentile(dtm, 1)), float(np.nanpercentile(dtm, 99))])
             if dtm is not None and covered.any() else None),
        dense_points=int(len(xyz)), outputs=outputs, seconds=round(time.time() - t0, 1),
        thermal=(dict(raw_range=list(ar.thermal_range), palette=opt.palette, raw_values="orthophoto_thermal.tif",
                      legend="thermal_legend.png") if thermal else None),
        options={k: (list(v) if isinstance(v, tuple) else v) for k, v in asdict(opt).items()},
        cameras=cams,
    )
    with open(os.path.join(out_dir, "report.json"), "w") as fh:
        json.dump(report, fh, indent=1)
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
    ap.add_argument("--dsm-resolution", type=float, default=None, help="DSM cell size in metres (default 2x GSD)")
    ap.add_argument("--gps-sigma", type=float, default=d.gps_sigma)
    ap.add_argument("--max-views", type=int, default=d.max_views)
    ap.add_argument("--min-score", type=float, default=d.min_score, help="NCC needed for a dense point (0-1)")
    ap.add_argument("--cloud-step", type=int, default=d.cloud_step, help="thin the dense cloud (every n-th cell)")
    ap.add_argument("--mesh-max-vertices", type=int, default=d.mesh_max_vertices)
    ap.add_argument("--formats", default=",".join(d.formats), help="comma list of ply,las,obj,glb")
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
                    formats=tuple(f.strip() for f in a.formats.split(",") if f.strip()))
    if a.bind_thermal:
        build_thermal_bound(a.images, a.bind_thermal, a.output, opt)
    else:
        build_3d(a.images, a.output, opt)


if __name__ == "__main__":
    main()
