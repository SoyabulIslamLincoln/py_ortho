"""DTM: running min/max, ground filter on a synthetic surface with buildings, LAS classes."""
import os
import struct
import sys
import tempfile

import numpy as np

try:
    import orthomosaic._core  # noqa: F401
except ImportError:
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from orthomosaic import export, terrain  # noqa: E402


def test_running_min_matches_bruteforce():
    rng = np.random.default_rng(0)
    for n, w in [(50, 1), (50, 3), (50, 7), (33, 15), (20, 31)]:
        a = rng.normal(0, 1, (n, n))
        r = w // 2
        p = np.pad(a, ((r, r), (0, 0)), constant_values=np.inf)
        ref = np.stack([p[i:i + w].min(0) for i in range(n)])
        assert np.allclose(terrain._running(a, w, 0, np.minimum), ref)


def test_dtm_removes_buildings_and_keeps_terrain():
    rng = np.random.default_rng(1)
    H, W, gsd = 600, 800, 0.1
    yy, xx = np.mgrid[0:H, 0:W] * gsd
    ground = 0.05 * xx + 1.0 * np.sin(yy / 15)                       # sloped, rolling terrain
    dsm = ground.copy()
    for x0, y0, w, h, z in [(10, 10, 20, 15, 8), (45, 5, 25, 30, 12), (20, 35, 12, 12, 5), (60, 40, 8, 8, 15)]:
        dsm[(xx >= x0) & (xx < x0 + w) & (yy >= y0) & (yy < y0 + h)] += z
    dsm = (dsm + rng.normal(0, 0.03, dsm.shape)).astype(np.float32)
    dsm[100:103, 300:303] -= 4.0                                      # a low blunder must not dig a pit
    dsm[:20, :] = np.nan                                              # no-data strip stays no-data
    dtm, g = terrain.dtm_from_dsm(dsm, gsd, terrain.TerrainOptions(max_object_size=40))
    ok = np.isfinite(dsm)
    e = np.abs(dtm - ground)[ok]
    assert np.isnan(dtm[:20]).all()
    assert np.median(e) < 0.05 and np.percentile(e, 99) < 0.5, (np.median(e), np.percentile(e, 99))
    bld = ((dsm - ground) > 1)[ok]
    assert (~g[ok][bld]).mean() > 0.99                               # buildings are not ground
    plausible = ok & (dsm >= ground - 1.0)
    assert (dtm <= np.where(plausible, dsm, np.inf) + 1e-6)[plausible].all()   # never above the real surface
    assert abs(dtm[101, 301] - ground[101, 301]) < 0.3


def test_las_ground_class():
    xyz = np.random.default_rng(2).uniform(0, 10, (50, 3)) + 500000
    cls = np.where(np.arange(50) % 2 == 0, 2, 1).astype(np.uint8)
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "c.las")
        export.write_las(p, xyz, np.zeros((50, 3), np.uint8), classification=cls)
        raw = open(p, "rb").read()
        off = struct.unpack("<I", raw[96:100])[0]
        rec = np.frombuffer(raw[off:], dtype=[("X", "<i4"), ("Y", "<i4"), ("Z", "<i4"), ("I", "<u2"), ("ret", "u1"),
                                              ("cls", "u1"), ("ang", "i1"), ("ud", "u1"), ("src", "<u2"),
                                              ("R", "<u2"), ("G", "<u2"), ("B", "<u2")])
        assert (rec["cls"] == cls).all()


if __name__ == "__main__":
    for k, fn in list(globals().items()):
        if k.startswith("test_"):
            fn()
            print("ok", k)
