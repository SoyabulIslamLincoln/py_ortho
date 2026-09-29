#changed here: new module -- Pix4D-style radiometric colour balancing for the 3D mosaic.
"""Radiometric colour balancing across a block (Pix4D "colour balancing").

Pix4D makes every image agree in brightness *and* contrast with its neighbours before the
mosaic is blended, so the seam network has no visible patchwork.  The existing
``align.solve_gains`` only solves a per-image multiplicative gain from the *mean* colour of
the matched feature points; that fixes exposure but not the black level / contrast differences
between cameras and lighting.

Here we generalise it to a full per-channel affine model

    I_i' = a_i * I_i + b_i

solved jointly over every matched point of every verified pair: for two overlapping images the
same physical point must end up with the same corrected colour, i.e. ``a_i*I_i + b_i =
a_j*I_j + b_j``.  A weak prior pulls every image toward the identity (gain 1, offset 0) so the
system stays well posed and the correction never runs away on weakly overlapping blocks.  The
result is applied to each image right before it is blended, exactly like Pix4D's radiometric
correction step.
"""
from __future__ import annotations

import logging

import numpy as np

log = logging.getLogger(__name__)


def solve_radiometric(used, pairs, sigma_n: float = 10.0, gain_prior: float = 0.2,
                      bias_prior: float = 8.0, max_gain: float = 2.0, max_bias: float = 48.0):
    #changed here: whole function -- per-image (gain, bias) solved over matched point colours.
    """Solve a per-image, per-channel affine colour correction.

    `pairs` are `align.PairMatch` objects whose ``ci``/``cj`` hold the (K, 3) RGB colours around
    each matched point.  Returns ``(gains, biases)`` as dicts ``frame_index -> np.ndarray(3,)``
    (float32).  Falls back to the identity correction if the system is degenerate or fails.
    """
    local = {g: l for l, g in enumerate(used)}
    n = len(used)
    if n == 0:
        return {}, {}
    gains = np.ones((n, 3), np.float32)
    biases = np.zeros((n, 3), np.float32)

    # keep only pairs inside the solved block and with usable colour samples
    usable = [p for p in pairs if p.i in local and p.j in local
              and len(p.ci) and len(p.cj) and len(p.ci) == len(p.cj)]
    if not usable:
        return ({g: gains[l] for l, g in enumerate(used)},
                {g: biases[l] for l, g in enumerate(used)})

    for ch in range(3):
        m = 2 * n
        A = np.zeros((m, m), np.float64)
        rhs = np.zeros(m, np.float64)
        # prior: a_i -> 1, b_i -> 0 (keeps the block near the identity, especially on sparse sets)
        for i in range(n):
            A[2 * i, 2 * i] += 1.0 / gain_prior ** 2
            rhs[2 * i] += 1.0 / gain_prior ** 2
            A[2 * i + 1, 2 * i + 1] += 1.0 / bias_prior ** 2
        # one residual per matched point: (a_i*ci + b_i) - (a_j*cj + b_j) = 0
        for p in usable:
            i, j = local[p.i], local[p.j]
            ci = p.ci[:, ch].astype(np.float64)
            cj = p.cj[:, ch].astype(np.float64)
            k = len(ci)
            w = 1.0 / sigma_n ** 2
            idx = (2 * i, 2 * i + 1, 2 * j, 2 * j + 1)
            fac = (ci, np.ones(k), -cj, -np.ones(k))     # factor of each unknown in the residual
            for u in range(4):
                for v in range(4):
                    A[idx[u], idx[v]] += w * float(np.sum(fac[u] * fac[v]))
        A[np.diag_indices(m)] += 1e-6
        try:
            sol = np.linalg.solve(A, rhs)
        except np.linalg.LinAlgError:
            log.warning("Colour balancing is singular (channel %d); leaving images unchanged", ch)
            continue
        if not np.isfinite(sol).all():
            log.warning("Colour balancing produced non-finite values (channel %d); ignoring", ch)
            continue
        a, b = sol[0::2].copy(), sol[1::2].copy()
        # The pairwise constraints a_i*ci+b_i = a_j*cj+b_j are invariant under a global scale and
        # offset of all (a_i, b_i) -- a gauge freedom that lets the whole block drift dark/bright
        # (all gains collapsing toward the clamp). Pin the gauge so the correction is *relative*:
        # mean gain 1 and mean bias 0, which keeps the overall brightness of the mosaic unchanged.
        am = float(np.mean(a))
        if am > 1e-3:
            a = a / am
            b = b / am
        b = b - float(np.mean(b))
        gains[:, ch] = np.clip(a, 1.0 / max_gain, max_gain)
        biases[:, ch] = np.clip(b, -max_bias, max_bias)

    log.info("Radiometric colour balancing: gain median %s, offset median %s",
             np.round(np.median(gains, 0), 3).tolist(), np.round(np.median(biases, 0), 1).tolist())
    return ({g: gains[l] for l, g in enumerate(used)},
            {g: biases[l] for l, g in enumerate(used)})
