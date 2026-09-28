"""Bundle-adjustment kernel check: analytic Schur system == dense finite-difference solve."""
import os
import sys

import numpy as np

try:
    import orthomosaic._ba  # noqa: F401
except ImportError:
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from orthomosaic import _ba  # noqa: E402
from orthomosaic.camera import rodrigues  # noqa: E402


def _problem(seed=0):
    rng = np.random.default_rng(seed)
    N, P, G = 4, 30, 2
    R = rodrigues(rng.normal(0, 0.05, (N, 3)) + np.array([np.pi, 0, 0]))   # looking down
    C = np.column_stack([rng.uniform(-5, 5, (N, 2)), np.full(N, 50.0)])
    X = np.column_stack([rng.uniform(-15, 15, (P, 2)), rng.uniform(0, 10, P)])
    intr = np.array([[1000.0, -0.05, 0.01], [900.0, 0.02, 0.0]])
    pp = np.array([[600.0, 450.0], [640.0, 512.0]])
    cam_group = np.array([0, 0, 1, 1], np.int32)
    obs_cam, obs_pt = [], []
    for p in range(P):
        for c in range(N):
            obs_cam.append(c)
            obs_pt.append(p)
    obs_cam = np.array(obs_cam, np.int32)
    obs_pt = np.array(obs_pt, np.int32)
    # observations = projection + noise
    uv = np.zeros((len(obs_cam), 2))
    for q, (c, p) in enumerate(zip(obs_cam, obs_pt)):
        xc = R[c] @ (X[p] - C[c])
        n = xc[:2] / xc[2]
        r2 = n @ n
        g = cam_group[c]
        uv[q] = intr[g, 0] * (1 + intr[g, 1] * r2 + intr[g, 2] * r2 * r2) * n + pp[g]
    uv += rng.normal(0, 2.0, uv.shape)
    pt_ptr = np.searchsorted(obs_pt, np.arange(P + 1)).astype(np.int64)
    return R, C, X, intr, pp, cam_group, obs_cam, obs_pt, uv, pt_ptr


def _residual_vec(R, C, X, intr, pp, cam_group, obs_cam, obs_pt, uv):
    out = []
    for c, p, o in zip(obs_cam, obs_pt, uv):
        xc = R[c] @ (X[p] - C[c])
        n = xc[:2] / xc[2]
        r2 = n @ n
        g = cam_group[c]
        out.append(intr[g, 0] * (1 + intr[g, 1] * r2 + intr[g, 2] * r2 * r2) * n + pp[g] - o)
    return np.concatenate(out)


def test_schur_matches_dense():
    R, C, X, intr, pp, cg, oc, op, uv, ptr = _problem()
    N, P, G = len(R), len(X), len(intr)
    nc = 6 * N + 3 * G
    huber = 1e9  # pure least squares for the comparison
    lam = 0.0
    zpw = np.zeros((P, 3))
    S, g, Vinv, gp, cost, dU = _ba.reduced_system(R, C, X, intr, pp, cg, oc, uv, ptr, huber, lam, zpw, zpw)

    def unpack(x):
        R2 = np.stack([rodrigues(x[6 * i:6 * i + 3]) @ R[i] for i in range(N)])
        C2 = C + x[:6 * N].reshape(N, 6)[:, 3:]
        I2 = intr + x[6 * N:nc].reshape(G, 3)
        X2 = X + x[nc:].reshape(P, 3)
        return R2, C2, X2, I2

    x0 = np.zeros(nc + 3 * P)
    r0 = _residual_vec(R, C, X, intr, pp, cg, oc, op, uv)
    J = np.zeros((len(r0), len(x0)))
    for k in range(len(x0)):
        h = 1e-6 * (1e3 if 6 * N <= k < nc and (k - 6 * N) % 3 == 0 else 1.0)
        xp = x0.copy()
        xp[k] += h
        xm = x0.copy()
        xm[k] -= h
        J[:, k] = (_residual_vec(*unpack(xp)[:2], unpack(xp)[2], unpack(xp)[3], pp, cg, oc, op, uv)
                   - _residual_vec(*unpack(xm)[:2], unpack(xm)[2], unpack(xm)[3], pp, cg, oc, op, uv)) / (2 * h)
    H = J.T @ J
    b = J.T @ r0
    A, B, D = H[:nc, :nc], H[:nc, nc:], H[nc:, nc:]
    Sd = A - B @ np.linalg.solve(D, B.T)
    gd = b[:nc] - B @ np.linalg.solve(D, b[nc:])
    assert np.allclose(g, gd, rtol=1e-4, atol=1e-3 * np.abs(gd).max())
    assert np.allclose(S, Sd, rtol=1e-4, atol=1e-5 * np.abs(Sd).max())
    assert abs(cost - r0 @ r0) < 1e-6 * cost

    # full step: reduced solve + back substitution == dense solve
    reg = 1e-2 * np.mean(np.diag(S)) * np.eye(nc)   # pins the gauge (no GPS priors here)
    dc = np.linalg.solve(S + reg, -g)
    dp = _ba.back_substitute(R, C, X, intr, pp, cg, oc, uv, ptr, huber, Vinv, gp, dc)
    Hr = H.copy()
    Hr[:nc, :nc] += reg
    full = np.linalg.solve(Hr, -b)
    assert np.allclose(dc, full[:nc], rtol=1e-3, atol=1e-6 * np.abs(full).max())
    assert np.allclose(dp.ravel(), full[nc:], rtol=1e-3, atol=1e-6 * np.abs(full).max())


if __name__ == "__main__":
    test_schur_matches_dense()
    print("ok test_schur_matches_dense")
