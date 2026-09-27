"""End-to-end orthomosaic pipeline."""
from __future__ import annotations

import json
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from typing import Optional, Sequence, Union

import numpy as np

from . import geo
from .align import candidate_pairs, match_pair, solve_alignment, solve_gains
from .backend import MODE_FEATHER, MODE_MAX, select_backend
from .features import extract
from .imageio import list_images, read_frame
from .render import render

log = logging.getLogger(__name__)


@dataclass
class Options:
    backend: str = "auto"              # auto | cuda | cpu
    workers: int = 0                   # 0 -> os.cpu_count()
    # features / matching
    feature_max_dim: int = 2000        # detect features on images downscaled to this
    n_features: int = 5000
    neighbors: int = 8                 # candidate pairs per image (GPS kNN)
    ratio: float = 0.8
    min_inliers: int = 25
    # alignment
    gps_sigma: float = 3.0             # metres
    match_sigma_px: float = 4.0
    # rendering
    resolution: Optional[float] = None  # output metres/pixel; default = native GSD / render_scale
    render_scale: float = 1.0          # decode images at this fraction of full resolution
    blend: str = "feather"             # feather | seam
    feather_power: float = 2.0
    exposure_compensation: bool = True
    block: int = 1024
    cache_mb: int = 1024
    preview: bool = True


def _absolute(A, origin):
    """Affine full-res pixel -> absolute world coordinates (UTM metres when georeferenced)."""
    A = A.copy()
    A[:, 2] += origin
    return A.tolist()


@dataclass
class AlignResult:
    """Everything the 2D alignment produced; reused by the 3D reconstruction."""
    frames: list
    feats: list
    work_scale: float
    pairs: list
    candidates: int
    alignment: object
    positions: Optional[np.ndarray]
    epsg: Optional[int]
    origin: tuple
    backend: object
    workers: int
    dropped: list


def align_images(images: Union[str, Sequence[str]], opt: Options) -> AlignResult:
    """Metadata -> features -> matching -> robust global 2D alignment."""
    workers = opt.workers or os.cpu_count() or 1
    backend = select_backend(opt.backend)

    paths = list_images(images) if isinstance(images, str) else list(images)
    if len(paths) < 2:
        raise ValueError("Need at least two images")
    log.info("Reading metadata of %d images", len(paths))
    with ThreadPoolExecutor(workers) as ex:
        frames = list(ex.map(read_frame, paths))

    # ---- GPS -> UTM (local origin keeps numbers well conditioned)
    positions, epsg, origin = None, None, (0.0, 0.0)
    gps_idx = [k for k, f in enumerate(frames) if f.has_gps]
    if len(gps_idx) >= 2:
        lat = np.array([frames[k].lat for k in gps_idx])
        lon = np.array([frames[k].lon for k in gps_idx])
        zone, north = geo.utm_zone(float(np.median(lat)), float(np.median(lon)))
        E, N = geo.latlon_to_utm(lat, lon, zone, north)
        origin = (float(np.round(E.mean())), float(np.round(N.mean())))
        positions = np.full((len(frames), 2), np.nan)
        positions[gps_idx, 0] = E - origin[0]
        positions[gps_idx, 1] = N - origin[1]
        epsg = geo.utm_epsg(zone, north)
        log.info("GPS found for %d/%d images -> UTM zone %d%s (EPSG:%d)",
                 len(gps_idx), len(frames), zone, "N" if north else "S", epsg)
    else:
        log.warning("No usable GPS: output will not be georeferenced")

    # ---- features
    log.info("Extracting features (%d workers)", workers)
    t = time.time()
    with ThreadPoolExecutor(workers) as ex:
        res = list(ex.map(lambda f: extract(f, opt.feature_max_dim, opt.n_features), frames))
    feats = [r[0] for r in res]
    work_scale = float(np.median([r[1] for r in res]))
    log.info("  %d features/image avg in %.1fs", int(np.mean([len(f) for f in feats])), time.time() - t)

    # ---- matching
    pos_for_pairs = positions if positions is not None and np.isfinite(positions).all() else None
    cand = candidate_pairs(pos_for_pairs, len(frames), opt.neighbors)
    log.info("Matching %d candidate pairs on %s", len(cand), backend.name)
    t = time.time()
    thresh = 3.0 / work_scale

    def _m(ij):
        i, j = ij
        return match_pair(backend, i, j, feats[i], feats[j], thresh, opt.ratio, min_inliers=opt.min_inliers)

    if backend.parallel_blocks:
        with ThreadPoolExecutor(workers) as ex:
            pairs = [p for p in ex.map(_m, cand) if p is not None]
    else:
        pairs = [p for p in map(_m, cand) if p is not None]
    log.info("  %d/%d pairs verified in %.1fs", len(pairs), len(cand), time.time() - t)
    if not pairs:
        raise RuntimeError("No image pairs could be matched; check overlap / image quality")

    # ---- global alignment
    al, pairs = solve_alignment(frames, pairs, positions, opt.gps_sigma, opt.match_sigma_px)
    dropped = [frames[k].name for k in range(len(frames)) if k not in set(al.used)]
    if dropped:
        log.warning("%d image(s) not connected to the main block were skipped: %s",
                    len(dropped), ", ".join(dropped[:10]) + (" ..." if len(dropped) > 10 else ""))
    log.info("Aligned %d images, georeferenced=%s, native GSD=%.4f, match RMS=%.2f px",
             len(al.used), al.georeferenced, al.gsd, al.residual_px)
    return AlignResult(frames, feats, work_scale, pairs, len(cand), al, positions, epsg, origin,
                       backend, workers, dropped)


def build_orthomosaic(images: Union[str, Sequence[str]], output: str,
                      options: Optional[Options] = None) -> dict:
    opt = options or Options()
    t0 = time.time()
    ar = align_images(images, opt)
    frames, pairs, al, backend = ar.frames, ar.pairs, ar.alignment, ar.backend
    epsg, origin, workers, dropped, cand = ar.epsg, ar.origin, ar.workers, ar.dropped, ar.candidates

    gains = solve_gains(al.used, pairs) if opt.exposure_compensation else {}

    # ---- render
    gsd = opt.resolution or al.gsd / opt.render_scale
    preview_path = os.path.splitext(output)[0] + "_preview.jpg" if opt.preview else None
    info = render(frames, al.affines, gains, backend, output, gsd,
                  epsg if al.georeferenced else None, origin if al.georeferenced else (0.0, 0.0),
                  render_scale=opt.render_scale, block=opt.block,
                  mode=MODE_MAX if opt.blend == "seam" else MODE_FEATHER,
                  power=opt.feather_power, workers=workers, cache_mb=opt.cache_mb,
                  preview_path=preview_path)

    report = dict(
        output=output, preview=preview_path, backend=backend.name,
        images_total=len(frames), images_used=len(al.used), images_dropped=dropped,
        pairs_candidate=cand, pairs_verified=len(pairs),
        georeferenced=al.georeferenced, epsg=epsg if al.georeferenced else None,
        gsd=gsd, match_rms_px=al.residual_px, seconds=round(time.time() - t0, 1),
        options=asdict(opt), **info,
        cameras={frames[k].name: dict(affine=_absolute(al.affines[k], origin if al.georeferenced else (0, 0)),
                                      gain=gains[k].tolist() if k in gains else [1, 1, 1])
                 for k in al.used},
    )
    with open(os.path.splitext(output)[0] + "_report.json", "w") as fh:
        json.dump(report, fh, indent=1)
    log.info("Done in %.1fs -> %s", time.time() - t0, output)
    return report
