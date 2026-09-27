"""Unit tests for the 3D modules (camera model, triangulation, SGM, hole filling, writers)."""
import os
import struct
import sys
import tempfile

import numpy as np

try:
    import orthomosaic._mvs  # noqa: F401
except ImportError:
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from orthomosaic import _mvs, export, sfm  # noqa: E402
from orthomosaic.backend import CPUBackend  # noqa: E402
from orthomosaic.camera import project, rodrigues, undistort_normalized  # noqa: E402
from orthomosaic.mvs import fill_holes  # noqa: E402


def test_undistort_inverts_project():
    rng = np.random.default_rng(0)
    n = 200
    R = np.repeat(rodrigues(np.array([[np.pi, 0, 0]])), n, 0)
    C = np.zeros((n, 3))
    X = np.column_stack([rng.uniform(-20, 20, (n, 2)), np.full(n, -50.0)])
    f, k1, k2 = np.full(n, 1000.0), np.full(n, -0.08), np.full(n, 0.02)
    cx, cy = np.full(n, 600.0), np.full(n, 450.0)
    uv, z = project(R, C, X, f, k1, k2, cx, cy)
    nrm = undistort_normalized(uv, f, k1, k2, cx, cy)
    xc = np.einsum("nij,nj->ni", R, X - C)
    assert np.abs(nrm - xc[:, :2] / xc[:, 2:]).max() < 1e-6


def test_triangulate_exact():
    rng = np.random.default_rng(1)
    N, P = 5, 50
    R = rodrigues(np.column_stack([np.full(N, np.pi), rng.normal(0, 0.03, (N, 2))]))
    C = np.column_stack([rng.uniform(-10, 10, (N, 2)), np.full(N, 60.0)])
    X = np.column_stack([rng.uniform(-15, 15, (P, 2)), rng.uniform(0, 15, P)])
    intr = np.array([[1200.0, -0.05, 0.01]])
    pp = np.array([[600.0, 450.0]])
    cg = np.zeros(N, np.int32)
    oc = np.repeat(np.arange(N, dtype=np.int32)[None], P, 0).ravel()
    op = np.repeat(np.arange(P, dtype=np.int32), N)
    uv, _ = project(R[oc], C[oc], X[op], np.full(len(oc), 1200.0), np.full(len(oc), -0.05),
                    np.full(len(oc), 0.01), np.full(len(oc), 600.0), np.full(len(oc), 450.0))
    Xt, ang = sfm.triangulate(R, C, intr, pp, cg, oc, op, uv, P)
    assert np.abs(Xt - X).max() < 1e-6 and ang.min() > 1.0


def test_sample_view_projects_ground():
    be = CPUBackend()
    img = np.zeros((100, 100, 1), np.float32)
    img[50, 60, 0] = 1.0           # a bright pixel
    R = np.diag([1.0, -1.0, -1.0])  # looking straight down, image x = east, y = south
    C = np.array([0.0, 0.0, 10.0])
    # ground point that projects to (u, v) = (60, 50): x = (60-49.5)/f*10, y = -(50-49.5)/f*10
    f = 100.0
    X, Y = (60 - 49.5) / f * 10, -(50 - 49.5) / f * 10
    Z = np.zeros((1, 1), np.float32)
    out, valid = be.sample_view(img, (R, C, f, 0.0, 0.0, 49.5, 49.5), X - 0.005, Y + 0.005, 0.01, Z)
    assert valid[0, 0] == 1 and abs(out[0, 0, 0] - 1.0) < 1e-4


def test_sgm_prefers_smooth_labels():
    rng = np.random.default_rng(2)
    D, H, W = 20, 30, 40
    cost = (1.0 + rng.uniform(0, 0.01, (D, H, W))).astype(np.float32)   # textureless = flat cost (as in mvs)
    cost[7, :, :W // 4] = 0.0            # strong evidence only in the left quarter
    agg = _mvs.sgm(cost, 0.05, 0.6)
    lab = np.argmin(agg, axis=0)
    assert (np.abs(lab - 7) <= 1).mean() > 0.97   # propagated (within one height step) across the rest


def test_fill_holes_smooth():
    z = np.fromfunction(lambda y, x: 0.01 * x + 0.02 * y, (120, 160)).astype(np.float32)
    t = z.copy()
    z[40:80, 50:110] = np.nan
    f = fill_holes(z)
    assert np.isfinite(f).all() and np.abs(f - t).max() < 0.3


def test_writers_roundtrip():
    rng = np.random.default_rng(3)
    xyz = rng.uniform(0, 10, (100, 3))
    rgb = rng.integers(0, 256, (100, 3)).astype(np.uint8)
    with tempfile.TemporaryDirectory() as d:
        export.write_ply(os.path.join(d, "p.ply"), xyz, rgb)
        raw = open(os.path.join(d, "p.ply"), "rb").read()
        body = raw[raw.index(b"end_header\n") + 11:]
        rec = np.frombuffer(body, dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("r", "u1"), ("g", "u1"), ("b", "u1")])
        assert np.allclose(rec["x"], xyz[:, 0], atol=1e-5) and (rec["r"] == rgb[:, 0]).all()
        export.write_las(os.path.join(d, "p.las"), xyz + 500000, rgb, epsg=32646)
        las = open(os.path.join(d, "p.las"), "rb").read()
        assert las[:4] == b"LASF" and struct.unpack("<I", las[107:111])[0] == 100
        Z = rng.uniform(0, 5, (20, 30)).astype(np.float32)
        V, F, UV = export.grid_mesh(Z, np.ones_like(Z, bool), 0.0, 10.0, 0.5)
        assert len(V) == 600 and len(F) == 2 * 19 * 29 and F.max() < len(V)
        from PIL import Image
        export.write_glb(os.path.join(d, "m.glb"), V, F, UV, Image.new("RGB", (8, 8)))
        g = open(os.path.join(d, "m.glb"), "rb").read()
        assert struct.unpack("<III", g[:12])[2] == len(g)


if __name__ == "__main__":
    for k, fn in list(globals().items()):
        if k.startswith("test_"):
            fn()
            print("ok", k)
