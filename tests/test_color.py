"""Radiometric colour balancing: relative corrections that don't collapse the global gauge."""
import os
import sys

import numpy as np

try:
    import orthomosaic._core  # noqa: F401
except ImportError:
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from orthomosaic.align import PairMatch  # noqa: E402
from orthomosaic.color import solve_radiometric  # noqa: E402


def _pairs_with_known_offsets(n=8, seed=0):
    """A chain of overlapping images, each a known gain/bias away from a shared latent colour."""
    rng = np.random.default_rng(seed)
    latent = [rng.uniform(40, 210, (200, 3)) for _ in range(n - 1)]   # shared points on each overlap
    true_gain = rng.uniform(0.8, 1.25, (n, 1))
    true_bias = rng.uniform(-20, 20, (n, 3))
    pairs = []
    for e in range(n - 1):
        i, j = e, e + 1
        L = latent[e]
        ci = np.clip(true_gain[i] * L + true_bias[i], 0, 255)
        cj = np.clip(true_gain[j] * L + true_bias[j], 0, 255)
        pairs.append(PairMatch(i, j, L[:, :2], L[:, :2], ci.astype(np.float32), cj.astype(np.float32), len(L)))
    return list(range(n)), pairs, true_gain, true_bias


def test_gauge_not_collapsed():
    used, pairs, tg, tb = _pairs_with_known_offsets()
    gains, biases = solve_radiometric(used, pairs)
    G = np.array([gains[i] for i in used])
    B = np.array([biases[i] for i in used])
    # gains stay around 1 (no runaway toward the 0.5 clamp), mean ~ 1
    assert 0.9 < G.mean() < 1.1, G.mean()
    assert (G > 0.55).all() and (G < 1.9).all(), (G.min(), G.max())
    # the correction is *relative*: it undoes the per-image differences up to a global gauge.
    # Corrected colours of the two images of each pair must agree.
    for p in pairs:
        corr_i = G[p.i] * p.ci + B[p.i]
        corr_j = G[p.j] * p.cj + B[p.j]
        assert np.abs(corr_i - corr_j).mean() < 3.0, np.abs(corr_i - corr_j).mean()


def test_identity_when_no_differences():
    rng = np.random.default_rng(1)
    n = 5
    pairs = []
    for e in range(n - 1):
        L = rng.uniform(40, 210, (150, 3)).astype(np.float32)
        pairs.append(PairMatch(e, e + 1, L[:, :2], L[:, :2], L.copy(), L.copy(), len(L)))
    gains, biases = solve_radiometric(list(range(n)), pairs)
    G = np.array([gains[i] for i in range(n)])
    B = np.array([biases[i] for i in range(n)])
    assert np.abs(G - 1).max() < 0.05 and np.abs(B).max() < 3.0


if __name__ == "__main__":
    for k, fn in list(globals().items()):
        if k.startswith("test_"):
            fn()
            print("ok", k)
