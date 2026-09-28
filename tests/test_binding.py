"""Thermal binding: pairing by filename, and recovering a known rig from paired poses."""
import math
import os
import sys

import numpy as np

try:
    import orthomosaic._core  # noqa: F401
except ImportError:
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from orthomosaic import binding  # noqa: E402
from orthomosaic.camera import rodrigues  # noqa: E402


def test_pairing():
    rgb = ["DJI_20260915182922_0001_V.JPG", "DJI_20260915182924_0002_V.JPG", "DJI_20260915182926_0003_V.JPG"]
    thr = ["DJI_20260915182922_0001_T.JPG", "DJI_20260915182926_0003_T.JPG", "DJI_20260915182930_0009_T.JPG"]
    pairs = binding.pair_frames(rgb, thr)
    assert pairs == [("DJI_20260915182922_0001_V.JPG", "DJI_20260915182922_0001_T.JPG"),
                     ("DJI_20260915182926_0003_V.JPG", "DJI_20260915182926_0003_T.JPG")]


def test_estimate_rig_recovers_known_offset():
    rng = np.random.default_rng(0)
    n = 40
    R_V = rodrigues(np.c_[np.full(n, np.pi), rng.normal(0, 0.05, (n, 2))])
    C_V = np.c_[np.linspace(-30, 30, n), rng.uniform(-5, 5, n), 50 + rng.normal(0, 0.5, n)]
    # a true rig: thermal rotated ~1.5 deg from RGB, 4 cm baseline
    R_rel_true = rodrigues(np.radians([0.8, -1.2, 0.5]))
    t_rel_true = np.array([0.03, -0.02, 0.01])
    R_T = np.einsum("ij,njk->nik", R_rel_true, R_V)
    C_T = C_V - np.einsum("nji,jk,k->ni", R_V, R_rel_true.T, t_rel_true)
    # add pose noise (thermal SfM is noisier)
    R_T = np.einsum("nij,njk->nik", rodrigues(rng.normal(0, np.radians(0.2), (n, 3))), R_T)
    C_T = C_T + rng.normal(0, 0.03, C_T.shape)
    rig = binding.estimate_rig(R_V, C_V, R_T, C_T)
    assert float(binding._geodesic_deg(R_rel_true[None], rig.R_rel)[0]) < 0.3
    assert np.linalg.norm(rig.t_rel - t_rel_true) < 0.05
    # apply_rig with the recovered rig reproduces the noise-free thermal poses
    R_clean = np.einsum("ij,njk->nik", R_rel_true, R_V)
    C_clean = C_V - np.einsum("nji,jk,k->ni", R_V, R_rel_true.T, t_rel_true)
    R_back, C_back = binding.apply_rig(rig, R_V, C_V)
    assert binding._geodesic_deg(R_back, np.eye(3)).shape == (n,)   # shape sanity
    assert np.abs(C_back - C_clean).max() < 0.05
    assert binding._geodesic_deg(np.einsum("nij,nkj->nik", R_back, R_clean), np.eye(3)).max() < 0.3


def test_rig_outlier_rejection():
    rng = np.random.default_rng(2)
    n = 30
    R_V = rodrigues(np.c_[np.full(n, np.pi), rng.normal(0, 0.05, (n, 2))])
    C_V = np.c_[np.linspace(-20, 20, n), rng.uniform(-3, 3, n), np.full(n, 45.0)]
    R_rel = rodrigues(np.radians([0.5, 0.5, 0.0]))
    t_rel = np.array([0.05, 0.0, 0.0])
    R_T = np.einsum("ij,njk->nik", R_rel, R_V)
    C_T = C_V - np.einsum("nji,jk,k->ni", R_V, R_rel.T, t_rel)
    # 5 gross outliers (bad thermal poses)
    bad = rng.choice(n, 5, replace=False)
    R_T[bad] = rodrigues(rng.normal(0, 1.0, (5, 3))) @ R_T[bad]
    C_T[bad] += rng.normal(0, 5, (5, 3))
    rig = binding.estimate_rig(R_V, C_V, R_T, C_T)
    assert rig.n_inliers >= n - 8
    assert np.linalg.norm(binding._geodesic_deg(R_rel[None], rig.R_rel)) < 0.5


if __name__ == "__main__":
    for k, fn in list(globals().items()):
        if k.startswith("test_"):
            fn()
            print("ok", k)
