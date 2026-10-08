"""Exact matcher regression tests, including tile/batch boundaries and equal distances."""
import os
import sys

import numpy as np

try:
    import orthomosaic._core
except ImportError:
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from orthomosaic import _core
from orthomosaic.backend import CPUBackend, cuda_available, mps_available, select_backend

INF = 1 << 30


def reference(a, b):
    n, m = len(a), len(b)
    if not n or not m:
        return (np.full(n, -1, np.int32), np.full(n, INF, np.int32),
                np.full(n, INF, np.int32), np.full(m, -1, np.int32))
    # Independent definition of Hamming distance, not the native implementation.
    lut = np.unpackbits(np.arange(256, dtype=np.uint8)[:, None], axis=1).sum(1)
    d = lut[a[:, None, :] ^ b[None, :, :]].sum(2).astype(np.int32)
    idx, back = d.argmin(1).astype(np.int32), d.argmin(0).astype(np.int32)
    sorted_d = np.sort(d, axis=1)
    second = sorted_d[:, 1] if m > 1 else np.full(n, INF, np.int32)
    return idx, sorted_d[:, 0], second, back


def cases():
    rng = np.random.default_rng(18)
    for n, m in [(0, 0), (0, 33), (33, 0), (1, 1), (1, 35), (35, 1),
                 (31, 33), (32, 32), (63, 65), (64, 64), (129, 127),
                 (2053, 67), (67, 2053), (8201, 1), (1025, 1027)]:
        a = rng.integers(0, 256, (n, 32), dtype=np.uint8)
        b = rng.integers(0, 256, (m, 32), dtype=np.uint8)
        yield a, b
        if n and m:
            a[:] = 0
            b[:] = 255                 # distance 256, every index tied
            yield a.copy(), b.copy()
            if n > 32 and m > 32:
                a[0] = a[-1] = b[32]   # ties across tiles and query batches
                b[0] = b[-1] = 0
                yield a.copy(), b.copy()
    # Noncontiguous user input must be copied by the backend.
    yield rng.integers(0, 256, (80, 64), dtype=np.uint8)[::2, ::2], b[::-1]


def check_backend(be):
    for a, b in cases():
        expected = reference(a, b)
        actual = be.match_mutual(a, b)
        assert len(actual) == 4
        for x, y in zip(expected, actual):
            assert y.dtype == np.int32 and np.array_equal(x, y), (be.name, a.shape, b.shape)
        for x, y in zip(expected[:3], be.match(a, b)):
            assert np.array_equal(x, y), (be.name, 'one-way', a.shape, b.shape)


def test_cpu_matching():
    check_backend(CPUBackend())


def test_gpu_matching():
    for name, available in [('mps', mps_available), ('cuda', cuda_available)]:
        if available():
            check_backend(select_backend(name))
            print('ok exact matching:', name)
        else:
            print('SKIP exact matching:', name, '(unavailable)')


def test_generic_descriptor_width():
    rng = np.random.default_rng(9)
    for width in (1, 2, 8):
        a = rng.integers(0, 256, (25, width * 8), dtype=np.uint8)
        b = rng.integers(0, 256, (35, width * 8), dtype=np.uint8)
        got = _core.match_hamming(a.view(np.uint64), b.view(np.uint64))
        assert all(np.array_equal(x, y) for x, y in zip(got, reference(a, b)[:3]))
    try:
        _core.match_hamming(a.view(np.uint64), b[:, :8].copy().view(np.uint64))
    except ValueError:
        pass
    else:
        raise AssertionError('mismatched descriptor width accepted')


if __name__ == '__main__':
    for name, fn in list(globals().items()):
        if name.startswith('test_'):
            fn()
            print('ok', name)
