"""Command line: python -m orthomosaic <image_dir> -o ortho.tif"""
from __future__ import annotations

import argparse
import logging

from .pipeline import Options, build_orthomosaic


def main(argv=None):
    d = Options()
    ap = argparse.ArgumentParser(prog="orthomosaic", description="Build an orthomosaic from nadir drone images.")
    ap.add_argument("images", help="folder containing the images")
    ap.add_argument("-o", "--output", default="orthomosaic.tif")
    ap.add_argument("--backend", choices=["auto", "cuda", "mps", "cpu"], default=d.backend)
    ap.add_argument("--workers", type=int, default=d.workers, help="CPU threads (0 = all cores)")
    ap.add_argument("--resolution", type=float, default=None, help="output GSD in metres/pixel")
    ap.add_argument("--render-scale", type=float, default=d.render_scale,
                    help="decode images at this fraction of full size when rendering (e.g. 0.5 for 4x less RAM)")
    ap.add_argument("--feature-max-dim", type=int, default=d.feature_max_dim)
    ap.add_argument("--features", type=int, default=d.n_features)
    ap.add_argument("--neighbors", type=int, default=d.neighbors)
    ap.add_argument("--gps-sigma", type=float, default=d.gps_sigma, help="GPS accuracy in metres")
    ap.add_argument("--blend", choices=["feather", "seam"], default=d.blend)
    ap.add_argument("--no-exposure", action="store_true", help="disable exposure compensation")
    ap.add_argument("--cache-mb", type=int, default=d.cache_mb, help="decoded-image cache budget")
    ap.add_argument("--no-preview", action="store_true")
    ap.add_argument("--palette", default=d.palette,
                    help="thermal palette: rainbow (default), iron, white_hot, black_hot, arctic, lava, ...")
    ap.add_argument("--no-thermal", action="store_true", help="ignore radiometric data; use JPEG colours")
    ap.add_argument("-v", "--verbose", action="store_true")
    a = ap.parse_args(argv)

    logging.basicConfig(level=logging.DEBUG if a.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    opt = Options(backend=a.backend, workers=a.workers, resolution=a.resolution, render_scale=a.render_scale,
                  feature_max_dim=a.feature_max_dim, n_features=a.features, neighbors=a.neighbors,
                  gps_sigma=a.gps_sigma, blend=a.blend, exposure_compensation=not a.no_exposure,
                  cache_mb=a.cache_mb, preview=not a.no_preview, palette=a.palette,
                  thermal="off" if a.no_thermal else "auto")
    build_orthomosaic(a.images, a.output, opt)


if __name__ == "__main__":
    main()
