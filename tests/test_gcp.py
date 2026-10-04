"""GCP: parsing, the point-position prior in the BA kernel, and anchoring a block to survey coords."""
import os
import sys
import tempfile

import numpy as np

try:
    import orthomosaic._ba  # noqa: F401
except ImportError:
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from orthomosaic import _ba, gcp, sfm  # noqa: E402
from orthomosaic.camera import Intrinsics, rodrigues  # noqa: E402


def test_parse_projected_and_latlon():
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "gcp.txt")
        open(p, "w").write("EPSG:32646\n"
                           "230012.5 2635008.1 12.3 2456.0 1810.5 DJI_0001_V.JPG GCP1\n"
                           "230012.5 2635008.1 12.3 1200.0 900.0 DJI_0002_V.JPG GCP1\n"
                           "# comment line\n"
                           "230040.0, 2635050.0, 10.0, 800, 640, DJI_0003_V.JPG, GCP2\n")
        g = gcp.load_gcps(p)
        assert g.epsg == 32646 and not g.is_latlon and len(g) == 2
        g1 = [x for x in g.gcps if x.name == "GCP1"][0]
        assert len(g1.marks) == 2 and g1.marks["DJI_0001_V.JPG"] == (2456.0, 1810.5)
        loc = gcp.to_local(g, 46, True, (230000.0, 2635000.0))
        assert np.allclose(loc["GCP1"][0], [12.5, 8.1, 12.3])
        # lat/lon header: geo_x = lon, geo_y = lat (WebODM convention)
        p2 = os.path.join(d, "ll.txt")
        open(p2, "w").write("WGS84\n90.35 23.803 12.0 100 100 DJI_0001_V.JPG A\n")
        g2 = gcp.load_gcps(p2)
        assert g2.is_latlon
        loc2 = gcp.to_local(g2, 46, True, (230000.0, 2635000.0))
        assert abs(loc2["A"][0][2] - 12.0) < 1e-6 and np.isfinite(loc2["A"][0][:2]).all()


def test_point_prior_pulls_point():
    rng = np.random.default_rng(0)
    N, P = 4, 20
    R = rodrigues(rng.normal(0, 0.03, (N, 3)) + [np.pi, 0, 0])
    C = np.c_[rng.uniform(-3, 3, (N, 2)), np.full(N, 40.0)]
    X = np.c_[rng.uniform(-10, 10, (P, 2)), rng.uniform(0, 3, P)]
    intr = np.array([[900.0, 0.0, 0.0, 0.0, 600.0, 450.0]])
    pp = np.array([[600.0, 450.0]])
    cg = np.zeros(N, np.int32)
    oc = np.repeat(np.arange(N), P).astype(np.int32)
    op = np.tile(np.arange(P), N).astype(np.int32)
    order = np.argsort(op)
    oc, op = oc[order], op[order]
    uv = np.array([R[c] @ (X[p] - C[c]) for c, p in zip(oc, op)])
    uv = uv[:, :2] / uv[:, 2:] * 900 + [600, 450]
    ptr = np.searchsorted(op, np.arange(P + 1)).astype(np.int64)
    pw = np.zeros((P, 3))
    pt = np.zeros((P, 3))
    pw[5] = 1e6
    pt[5] = X[5] + [2.0, -1.0, 0.5]
    S, g, Vinv, gp, cost, dU = _ba.reduced_system(R, C, X, intr, pp, cg, oc, uv, ptr, 1e9, 1e-6, pw, pt)
    S = S + np.diag(np.full(S.shape[0], 1e10))   # pin cameras + intrinsics: isolate the point prior
    dc = np.linalg.solve(S + 1e-6 * np.eye(S.shape[0]), -g)
    dp = _ba.back_substitute(R, C, X, intr, pp, cg, oc, uv, ptr, 1e9, Vinv, gp, dc)
    assert np.linalg.norm(dp[5] - [2.0, -1.0, 0.5]) < 0.2                 # point 5 moves to its target
    assert np.median(np.linalg.norm(np.delete(dp, 5, 0), axis=1)) < 0.05  # others barely move


def _synthetic_block(shift):
    """A small block whose cameras/points are offset from truth by `shift`; GCPs know the truth."""
    rng = np.random.default_rng(1)
    N, P = 6, 60
    Rt = rodrigues(np.c_[np.full(N, np.pi), rng.normal(0, 0.02, (N, 2))])
    Ct = np.c_[np.linspace(-15, 15, N), rng.uniform(-2, 2, N), np.full(N, 50.0)]
    Xt = np.c_[rng.uniform(-20, 20, (P, 2)), rng.uniform(0, 8, P)]
    f = 1000.0
    it = Intrinsics(1200, 900, f)
    oc, op, uv = [], [], []
    for c in range(N):
        for p in range(P):
            xc = Rt[c] @ (Xt[p] - Ct[c])
            if xc[2] <= 0:
                continue
            u = xc[:2] / xc[2] * f + [599.5, 449.5]
            if 0 <= u[0] < 1200 and 0 <= u[1] < 900:
                oc.append(c)
                op.append(p)
                uv.append(u)
    oc = np.array(oc, np.int32)
    op = np.array(op, np.int32)
    order = np.argsort(op, kind="stable")
    oc, op, uv = oc[order], op[order], np.array(uv)[order]
    rec = sfm.Reconstruction(list(range(N)), (Rt).copy(), Ct + shift, [it], np.zeros(N, np.int32),
                             Xt + shift, np.zeros((P, 3), np.uint8), oc, op.astype(np.int32), uv)
    return rec, Rt, Ct, Xt


def test_gcp_anchors_block():
    # the reconstruction is offset from truth; 4 GCPs at known truth coords should pull it back
    shift = np.array([5.0, -3.0, 2.0])
    rec, Rt, Ct, Xt = _synthetic_block(shift)
    N = len(rec.R)
    # add 4 GCPs = 4 of the true points, marked in the images that see them
    gpts = [3, 20, 41, 55]
    marks = {}
    for gi in gpts:
        m = rec.obs_pt == gi
        marks[f"G{gi}"] = (Xt[gi], {})     # placeholder; we inject observations directly
    # build gcps dict as sfm._add_gcps expects: name -> (world, {image_basename: uv})
    frames = [type("F", (), {"name": f"IMG_{i}.JPG"})() for i in range(N)]
    gdict = {}
    for gi in gpts:
        obs = np.nonzero(rec.obs_pt == gi)[0]
        gdict[f"G{gi}"] = (Xt[gi], {frames[rec.obs_cam[o]].name: tuple(rec.obs_uv[o]) for o in obs})
    sfm._add_gcps(rec, frames, list(range(N)), gdict, gcp_sigma=0.02)
    # loose camera prior (just gauge stabilisation); the GCPs do the absolute anchoring
    pri = sfm.Priors(rec.C.copy(), np.full((N, 3), 50.0), rec.intr_array(),
                     np.full((1, 6), np.inf))
    for _ in range(4):
        err = sfm.bundle_adjust(rec, pri, huber=1e9)
    rep = sfm.gcp_report(rec, None)
    assert rep["summary"]["rmse_3d_m"] < 0.05, rep["summary"]
    # camera centres should now be near truth (block pulled back from the 5,-3,2 offset)
    cam_err = np.linalg.norm(rec.C - Ct, axis=1)
    assert np.median(cam_err) < 0.3, np.median(cam_err)


if __name__ == "__main__":
    for k, fn in list(globals().items()):
        if k.startswith("test_"):
            fn()
            print("ok", k)
