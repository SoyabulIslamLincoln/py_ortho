"""Thermal -> RGB co-registration: recover a known affine between two differently-rendered rasters."""
import json
import math
import os
import sys
import tempfile

import numpy as np

try:
    import orthomosaic._core  # noqa: F401
except ImportError:
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from orthomosaic import coregister  # noqa: E402
from orthomosaic.geotiff import GeoTIFFWriter, read_geotiff  # noqa: E402

GSD = 0.04
ORIGIN = (500000.0, 4500000.0)


def _scene(x, y):
    """A synthetic 'roof': rectangular units of different sizes on a plain background."""
    rng = np.random.default_rng(3)
    v = np.zeros(x.shape, np.float32)
    for _ in range(140):
        cx, cy = rng.uniform(2, 58), rng.uniform(-58, -2)
        w, h = rng.uniform(0.6, 3.0, 2)
        v += rng.uniform(0.3, 1.0) * ((np.abs(x - cx) < w / 2) & (np.abs(y - cy) < h / 2))
    return v


def _render(M, n=1500, kind="rgb"):
    """Raster of the scene as seen through `M` (true world -> raster world): RGB-like or thermal-like."""
    r, c = np.mgrid[0:n, 0:n]
    xw, yw = (c + 0.5) * GSD, -(r + 0.5) * GSD                         # raster world (local)
    Mi = np.linalg.inv(M)
    x = Mi[0, 0] * xw + Mi[0, 1] * yw + Mi[0, 2]
    y = Mi[1, 0] * xw + Mi[1, 1] * yw + Mi[1, 2]
    s = _scene(x, y)
    if kind == "rgb":
        g = (60 + 150 * s / s.max()).astype(np.uint8)
        return np.dstack([g, g, g, np.full_like(g, 255)])
    t = 15 + 20 * np.sqrt(s) - 3 * s                                 # different, non-linear contrast
    return (t + np.random.default_rng(1).normal(0, 0.3, t.shape)).astype(np.float32)


def _geo():
    return dict(epsg=32618, origin=ORIGIN, pixel_size=GSD)


def _true_shift():
    a = math.radians(0.8)
    L = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]]) @ np.diag([1.012, 0.988])
    M = np.eye(3)
    M[:2, :2] = L
    M[:2, 2] = [1.4, -0.9]
    # around the raster centre, like a block whose scale / rotation are off
    c = np.array([30.0, -30.0])
    M[:2, 2] += c - L @ c
    return M


def test_recovers_affine():
    M_true = _true_shift()                                             # thermal raster is misplaced by M_true
    rgb = _render(np.eye(3), kind="rgb")
    th = _render(M_true, kind="thermal")
    res = coregister.coregister(rgb, _geo(), th, _geo(), max_shift_m=4.0)
    # the fit maps thermal positions back onto the RGB, i.e. it undoes M_true
    off = np.array([[ORIGIN[0]], [ORIGIN[1]]])
    probe = np.array([[5.0, -5.0], [55.0, -5.0], [5.0, -55.0], [55.0, -55.0], [30.0, -30.0]])
    th_pos = probe @ M_true[:2, :2].T + M_true[:2, 2] + off.T          # where each true feature sits in the thermal
    back = th_pos @ res.matrix[:2, :2].T + res.matrix[:2, 2]
    err = np.linalg.norm(back - (probe + off.T), axis=1)
    assert err.max() < 0.08, err
    assert res.stats["residual_median_m"] < 0.05, res.stats
    assert res.stats["shift_before_median_m"] > 0.8


def test_warp_and_alignment_json():
    M_true = _true_shift()
    with tempfile.TemporaryDirectory() as d:
        for name, arr, kind in (("rgb.tif", _render(np.eye(3), kind="rgb"), "rgba"),
                                ("th.tif", _render(M_true, kind="thermal"), "float32")):
            w = GeoTIFFWriter(os.path.join(d, name), arr.shape[1], arr.shape[0], kind=kind,
                              nodata=float("nan") if kind == "float32" else None, **_geo())
            for ty in range(0, arr.shape[0], 512):
                for tx in range(0, arr.shape[1], 512):
                    w.write_tile(tx // 512, ty // 512, w.compress_tile(arr[ty:ty + 512, tx:tx + 512]))
            w.close()
        out = os.path.join(d, "th_coreg.tif")
        res = coregister.coregister_files(os.path.join(d, "rgb.tif"), os.path.join(d, "th.tif"), out,
                                          max_shift_m=4.0)
        al = json.load(open(os.path.join(d, "th_coreg_alignment.json")))
        assert al["order"] == "fit@manual" and len(al["pairs"]) >= 6
        assert np.allclose(al["matrix"], res.matrix)
        # the corrected raster now lines up with the RGB: re-registering finds ~nothing left to fix
        rgb, g1 = read_geotiff(os.path.join(d, "rgb.tif"))
        fixed, g2 = read_geotiff(out)
        again = coregister.coregister(rgb, g1, fixed, g2, max_shift_m=2.0)
        assert again.stats["shift_before_median_m"] < 0.06, again.stats


if __name__ == "__main__":
    for k, fn in list(globals().items()):
        if k.startswith("test_"):
            fn()
            print("ok", k)
