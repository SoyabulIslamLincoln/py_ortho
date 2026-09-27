"""Unit tests: python -m pytest tests/  (or python tests/test_core.py)"""
import os
import sys
import tempfile

import numpy as np

try:  # prefer the installed package (CI tests the built wheel, not the source tree)
    import orthomosaic._core  # noqa: F401
except ImportError:  # running from a source checkout after `build_ext --inplace`
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from orthomosaic import _core, align, geo  # noqa: E402
from orthomosaic.geotiff import GeoTIFFWriter  # noqa: E402


def test_utm_roundtrip():
    for lat, lon in [(23.8, 90.4), (-33.9, 18.4), (60.1, 5.3), (51.5, -0.12)]:
        z, n = geo.utm_zone(lat, lon)
        e, no = geo.latlon_to_utm(lat, lon, z, n)
        la, lo = geo.utm_to_latlon(e, no, z, n)
        assert abs(la - lat) < 1e-7 and abs(lo - lon) < 1e-7


def test_ransac_recovers_affine():
    rng = np.random.default_rng(0)
    A = np.array([[0.98, -0.17, 40.0], [0.17, 0.98, -12.0]])
    src = rng.uniform(0, 1000, (400, 2))
    dst = src @ A[:, :2].T + A[:, 2] + rng.normal(0, 0.5, (400, 2))
    dst[:150] = rng.uniform(0, 1000, (150, 2))                 # 37% outliers
    est, mask = _core.ransac_affine(np.ascontiguousarray(src), np.ascontiguousarray(dst), 2000, 3.0)
    assert mask[150:].mean() > 0.95 and mask[:150].mean() < 0.05
    assert np.abs(est - A).max() < 1.0


def test_mutual_matcher_equals_two_pass():
    rng = np.random.default_rng(1)
    a = rng.integers(0, 2 ** 63, (500, 4), dtype=np.uint64)
    b = rng.integers(0, 2 ** 63, (400, 4), dtype=np.uint64)
    i12, b12, s12 = _core.match_hamming(a, b)
    i21, _, _ = _core.match_hamming(b, a)
    m = _core.match_hamming_mutual(a, b)
    assert (m[0] == i12).all() and (m[1] == b12).all() and (m[2] == s12).all() and (m[3] == i21).all()


def test_warp_identity_and_finalize():
    rng = np.random.default_rng(2)
    img = rng.integers(0, 256, (64, 80, 3), dtype=np.uint8)
    acc = np.zeros((64, 80, 3), np.float32)
    wsum = np.zeros((64, 80), np.float32)
    M = np.array([1, 0, 0, 0, 1, 0], np.float64)
    _core.warp_accumulate(acc, wsum, img, M, np.ones(3, np.float32), 1.0, 0)
    out = _core.finalize(acc, wsum, 0)
    assert (out[..., 3] == 255).all()
    assert np.abs(out[..., :3].astype(int) - img).max() <= 1


def test_candidate_pairs_knn():
    pos = np.stack(np.meshgrid(np.arange(5.0), np.arange(4.0)), -1).reshape(-1, 2) * 10
    pairs = align.candidate_pairs(pos, len(pos), k=4)
    assert (0, 1) in pairs and (0, 5) in pairs and (0, 19) not in pairs


def test_geotiff_readable():
    from PIL import Image
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "t.tif")
        w = GeoTIFFWriter(p, 700, 600, tile=256, epsg=32646, origin=(1000.0, 2000.0), pixel_size=0.05)
        blk = np.zeros((256, 256, 4), np.uint8)
        blk[..., 0] = 200
        blk[..., 3] = 255
        w.write_tile(1, 1, w.compress_tile(blk))
        w.close()
        im = Image.open(p)
        assert im.size == (700, 600)
        a = np.asarray(im.convert("RGBA"))
        assert a[300, 300, 0] == 200 and a[300, 300, 3] == 255 and a[10, 10, 3] == 0
        assert im.tag_v2[33550][0] == 0.05 and 32646 in im.tag_v2[34735]


if __name__ == "__main__":
    for k, f in list(globals().items()):
        if k.startswith("test_"):
            f()
            print("ok", k)
