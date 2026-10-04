"""GPU backends must agree with the Cython CPU backend (skipped when no GPU is present)."""
import os
import sys

import numpy as np

try:
    import orthomosaic._core  # noqa: F401
except ImportError:
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from orthomosaic.backend import CPUBackend, cuda_available, mps_available, select_backend  # noqa: E402


def _gpu_backends():
    out = []
    if mps_available():
        out.append(select_backend("mps"))
    if cuda_available():
        out.append(select_backend("cuda"))
    return out


def test_gpu_backends_match_cpu():
    cpu = CPUBackend()
    rng = np.random.default_rng(0)
    for gpu in _gpu_backends():
        # matcher
        d1 = rng.integers(0, 256, (700, 32), dtype=np.uint8)
        d2 = rng.integers(0, 256, (650, 32), dtype=np.uint8)
        c, g = cpu.match_mutual(d1, d2), gpu.match_mutual(d1, d2)
        assert all((np.asarray(x) == np.asarray(y)).all() for x, y in zip(c, g)), gpu.name
        # warp + blend (both modes)
        img = rng.integers(0, 256, (120, 160, 3), dtype=np.uint8)
        M = np.array([[0.9, 0.2, 10.3], [-0.15, 0.95, 5.7]])
        for mode in (0, 1):
            bc, bg = cpu.new_block(90, 110), gpu.new_block(90, 110)
            for _ in range(2):
                cpu.warp_accumulate(bc, cpu.upload(img), M, np.array([1.1, 1.0, 0.9]), 2.0, mode)
                gpu.warp_accumulate(bg, gpu.upload(img), M, np.array([1.1, 1.0, 0.9]), 2.0, mode)
            oc, og = cpu.finalize(bc, mode), gpu.finalize(bg, mode)
            assert (oc[..., 3] == og[..., 3]).mean() > 0.999, gpu.name
            both = (oc[..., 3] > 0) & (og[..., 3] > 0)   # footprint-edge pixels may differ (float32 vs 64)
            assert np.abs(oc[both][:, :3].astype(int) - og[both][:, :3]).max() <= 1, gpu.name
        # dense sampler
        im = rng.uniform(0, 255, (200, 240, 3)).astype(np.float32)
        th = 0.05
        R = np.array([[1, 0, 0], [0, np.cos(np.pi + th), -np.sin(np.pi + th)], [0, np.sin(np.pi + th), np.cos(np.pi + th)]])
        cam = (R, np.array([3.0, -2.0, 40.0]), 300.0, -0.05, 0.01, 0.02, 119.5, 99.5)   # f k1 k2 k3 cx cy
        Z = rng.uniform(0, 8, (60, 70)).astype(np.float32)
        oc, vc = cpu.sample_view(im, cam, -10.0, 8.0, 0.25, Z)
        og, vg = gpu.sample_view(gpu.upload(im), cam, -10.0, 8.0, 0.25, gpu.asarray(Z, gpu.xp.float32))
        og, vg = gpu.to_numpy(og), gpu.to_numpy(vg)
        both = (vc == 1) & (vg == 1)
        assert (vc == vg).mean() > 0.999 and np.abs(oc[both] - og[both]).max() < 0.05, gpu.name
        # box filter
        a = rng.uniform(0, 255, (50, 60)).astype(np.float32)
        assert np.abs(cpu.box(a, 3) - gpu.to_numpy(gpu.box(gpu.asarray(a, gpu.xp.float32), 3))).max() < 1e-2
        st = rng.uniform(-1, 1, (6, 4, 30, 40)).astype(np.float32)
        assert np.abs(cpu.topk_mean(st, 3) - gpu.to_numpy(gpu.topk_mean(gpu.asarray(st, gpu.xp.float32), 3))).max() < 1e-5
        b3 = rng.uniform(0, 255, (3, 5, 50, 60)).astype(np.float32)
        assert np.abs(cpu.box(b3, 3) - gpu.to_numpy(gpu.box(gpu.asarray(b3, gpu.xp.float32), 3))).max() < 1e-2
        for fn in ("argmin0", "argmax0"):
            ic, vc_ = getattr(cpu, fn)(st)
            ig, vg_ = getattr(gpu, fn)(gpu.asarray(st, gpu.xp.float32))
            assert (ic == gpu.to_numpy(ig)).all() and np.allclose(vc_, gpu.to_numpy(vg_)), fn
        print("ok", gpu.name, "matches cpu")


if __name__ == "__main__":
    test_gpu_backends_match_cpu()
    print("ok test_gpu_backends_match_cpu")
