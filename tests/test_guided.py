"""Pose-guided matching (align.guided_mutual) and DSM hole filling (mvs.postprocess) regression tests."""
import os
import sys

import numpy as np

try:
    import orthomosaic._core
except ImportError:
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from orthomosaic import _core, align, mvs
from orthomosaic.features import Features

INF = 1 << 30


def _feats(xy, desc):
    return Features(np.asarray(xy, np.float64), np.ascontiguousarray(desc, np.uint8),
                    np.zeros((len(xy), 3), np.float32))


def _reference(fi, fj, H, radius):
    """Brute force: Hamming distances with pairs beyond `radius` of the prediction excluded."""
    n, m = len(fi.xy), len(fj.xy)
    if not n or not m:
        return np.full(n, INF, np.int32), np.full(n, INF, np.int32)
    lut = np.unpackbits(np.arange(256, dtype=np.uint8)[:, None], axis=1).sum(1)
    d = lut[fi.desc[:, None, :] ^ fj.desc[None, :, :]].sum(2).astype(np.int32)
    p = fi.xy @ H[:, :2].T + H[:, 2]
    far = ((p[:, None, :] - fj.xy[None, :, :]) ** 2).sum(2) > radius * radius
    d[far] = INF
    s = np.sort(d, 1)
    return s[:, 0], (s[:, 1] if m > 1 else np.full(n, INF, np.int32))


def _both_paths(fn):
    """Run `fn` with the compiled kernel and with the NumPy fallback."""
    fn()
    kern = getattr(_core, "match_hamming_guided", None)
    if kern is not None:
        del _core.match_hamming_guided
        try:
            fn()
        finally:
            _core.match_hamming_guided = kern


def test_guided_mutual_matches_reference():
    rng = np.random.default_rng(5)
    H = np.array([[0.98, -0.17, 40.0], [0.17, 0.98, -25.0]])

    def check():
        for n, m, radius in [(0, 20, 50.0), (20, 0, 50.0), (1, 1, 1e4), (300, 280, 40.0), (500, 30, 90.0)]:
            fi = _feats(rng.uniform(0, 600, (n, 2)), rng.integers(0, 256, (n, 32)))
            fj = _feats(rng.uniform(0, 600, (m, 2)), rng.integers(0, 256, (m, 32)))
            idx12, b12, s12, idx21 = align.guided_mutual(fi, fj, H, radius)
            rb, rs = _reference(fi, fj, H, radius)
            assert np.array_equal(b12, rb) and np.array_equal(s12, rs)
            ok = idx12 >= 0
            assert np.array_equal(ok, rb < INF)
            assert len(idx21) == m
    _both_paths(check)


def test_guided_matching_survives_a_repetitive_pattern():
    """Corrugated-roof case: the same descriptors repeat every 100 px, so unrestricted matching has
    a best and an equal second best and the ratio test drops everything; within 40 px of where the
    pose puts each feature there is a single candidate."""
    rng = np.random.default_rng(1)
    base = rng.integers(0, 256, (40, 32), dtype=np.uint8)
    xy = np.array([[x + 100 * k, 20 + 12 * (x % 7)] for k in range(8) for x in range(0, 80, 2)], np.float64)
    desc = np.tile(base, (8, 1))
    fi = _feats(xy, desc)
    shift = np.array([37.0, 5.0])
    fj = _feats(xy + shift, desc)
    H = np.array([[1.0, 0.0, shift[0] + 6.0], [0.0, 1.0, shift[1] - 4.0]])      # pose good to ~7 px

    full = _core.match_hamming_mutual(fi.desc.view(np.uint64), fj.desc.view(np.uint64))
    assert not ((full[1] < 0.8 * full[2])).any()               # ratio test: nothing survives

    def check():
        idx12, b12, s12, idx21 = align.guided_mutual(fi, fj, H, 40.0)
        good = (b12 < 0.8 * s12) & (idx12 >= 0)
        assert good.mean() > 0.9
        assert np.allclose(fj.xy[idx12[good]] - fi.xy[good], shift)
    _both_paths(check)


def _dense(Z, score, stepped, gsd=0.05):
    H, W = Z.shape
    return mvs.DenseResult(Z=Z.astype(np.float32), score=score.astype(np.float32),
                           rgb=np.zeros((H, W, 3), np.uint8), covered=np.ones((H, W), bool),
                           minX=0.0, maxY=0.0, gsd=gsd, stepped=stepped)


def test_postprocess_lone_blunder_does_not_seed_a_blob():
    """A single +8 m measurement in a 6 m wide unmeasured patch of flat ground (dark shadow): the
    patch must be filled at ground level, not as a hump around the blunder."""
    Z = np.zeros((240, 240), np.float32)
    score = np.ones_like(Z)
    Z[60:180, 60:180] = np.nan
    score[60:180, 60:180] = 0
    Z[120, 120], score[120, 120] = 8.0, 1.0
    dsm, conf, support = mvs.postprocess(_dense(Z, score, np.zeros(Z.shape, bool)), 0.5, smooth_range=0)
    assert np.nanmax(np.abs(dsm[60:180, 60:180])) < 0.05
    assert not conf[120, 120]


def test_postprocess_wall_gap_leans_to_the_ground():
    """10 m roof beside ground with a 2 m unmeasured strip at the foot of the wall (occluded in most
    photos): the strip is filled mostly at ground level, far below a straight roof-to-ground ramp
    (median 5 m), and the ground half of the strip is at ground height."""
    Z = np.zeros((200, 300), np.float32)
    Z[:, :150] = 10.0                                          # roof on the left
    score = np.ones_like(Z)
    Z[:, 150:190] = np.nan                                     # 2 m gap at 5 cm cells
    score[:, 150:190] = 0
    dsm, _, support = mvs.postprocess(_dense(Z, score, np.zeros(Z.shape, bool)), 0.5, smooth_range=0)
    rows = slice(20, 180)
    assert np.all(support[rows, 150:190] == 2)
    assert np.median(dsm[rows, 150:190]) < 3.0
    assert np.percentile(dsm[rows, 175:190], 90) < 1.5          # ground half of the gap
    assert np.all(dsm[rows, :145] > 9.5)                       # roof untouched
