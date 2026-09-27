"""Compute backends: Cython (CPU) and CUDA (via CuPy raw kernels).

Both expose the same small interface used by the pipeline:

    match(d1, d2)                         -> (idx, best, second)   numpy int32
    match_mutual(d1, d2)                  -> (idx12, best12, second12, idx21)
    new_block(H, W)                       -> accumulator handle
    upload(img_uint8_rgb)                 -> device image
    warp_accumulate(block, dev_img, M, gain, power, mode)
    finalize(block, mode)                 -> numpy uint8 RGBA
    xp                                    -> array module (numpy / cupy) for dense 3D
    sample_view(img, cam, X0, Y0, gsd, Z) -> (samples (H,W,ch) float32, valid (H,W))
    to_numpy(a)                           -> numpy array
"""
from __future__ import annotations

import logging
import os

import numpy as np

from . import _core, _mvs

log = logging.getLogger(__name__)

MODE_FEATHER = 0
MODE_MAX = 1


class CPUBackend:
    name = "cpu"
    parallel_blocks = True
    xp = np

    def match(self, d1: np.ndarray, d2: np.ndarray):
        return _core.match_hamming(_as_u64(d1), _as_u64(d2))

    def match_mutual(self, d1: np.ndarray, d2: np.ndarray):
        """-> (idx12, best12, second12, idx21)"""
        return _core.match_hamming_mutual(_as_u64(d1), _as_u64(d2))

    def new_block(self, H: int, W: int):
        return np.zeros((H, W, 3), np.float32), np.zeros((H, W), np.float32)

    def upload(self, img: np.ndarray):
        return np.ascontiguousarray(img)

    def image_nbytes(self, dev_img) -> int:
        return dev_img.nbytes

    def warp_accumulate(self, block, dev_img, M, gain, power, mode):
        acc, wsum = block
        _core.warp_accumulate(acc, wsum, dev_img, np.ascontiguousarray(M, np.float64).ravel(),
                              np.ascontiguousarray(gain, np.float32), float(power), int(mode))

    def finalize(self, block, mode):
        acc, wsum = block
        return _core.finalize(acc, wsum, int(mode))

    def sample_view(self, img, cam, X0, Y0, gsd, Z):
        R, C, f, k1, k2, cx, cy = cam
        Z = np.ascontiguousarray(Z, np.float32)
        out = np.zeros(Z.shape + (img.shape[2],), np.float32)
        valid = np.zeros(Z.shape, np.uint8)
        _mvs.sample_view(img, R, C, float(f), float(k1), float(k2), float(cx), float(cy),
                         float(X0), float(Y0), float(gsd), Z, out, valid)
        return out, valid

    def to_numpy(self, a):
        return np.asarray(a)


def _as_u64(d: np.ndarray) -> np.ndarray:
    d = np.ascontiguousarray(d, np.uint8)
    return d.view(np.uint64).reshape(d.shape[0], -1)


_CUDA_HELP = ("Install the CUDA headers CuPy needs to compile kernels: "
              "pip install \"pyOrthomosaic[cuda]\"  (or: pip install \"cupy-cuda12x[ctk]\"), "
              "or install the CUDA Toolkit and set CUDA_PATH to it.")


def _prepare_cuda_env():
    """CuPy compiles kernels at runtime and needs the CUDA headers. If CUDA_PATH is not
    set, point it at an installed toolkit (the pip [ctk] headers are found by CuPy itself)."""
    if os.environ.get("CUDA_PATH") or os.environ.get("CUDA_HOME"):
        return
    import glob
    candidates = ["/usr/local/cuda", "/opt/cuda"]
    candidates += sorted(glob.glob("/usr/local/cuda-*"), reverse=True)
    candidates += sorted(glob.glob(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v*"), reverse=True)
    for c in candidates:
        if os.path.isfile(os.path.join(c, "include", "cuda_runtime.h")):
            os.environ["CUDA_PATH"] = c
            log.debug("CUDA_PATH set to %s", c)
            return


def cuda_available() -> bool:
    if os.environ.get("ORTHO_DISABLE_CUDA") == "1":
        return False
    try:
        _prepare_cuda_env()
        import cupy  # noqa: F401
        return cupy.cuda.runtime.getDeviceCount() > 0
    except Exception:
        return False


def select_backend(prefer: str = "auto"):
    """prefer: 'auto' | 'cuda' | 'cpu'."""
    prefer = prefer.lower()
    if prefer in ("auto", "cuda", "gpu"):
        if cuda_available():
            try:
                from ._cuda import CUDABackend
                be = CUDABackend()
                log.info("Using CUDA backend: %s", be.device_name)
                return be
            except Exception as exc:  # compile/driver problems -> fall back
                msg = str(exc).splitlines()[0] if str(exc) else type(exc).__name__
                if "header" in msg.lower() or "nvrtc" in msg.lower() or "CUDA_PATH" in msg:
                    msg += ". " + _CUDA_HELP
                if prefer != "auto":
                    raise RuntimeError(f"CUDA backend unusable: {msg}") from exc
                log.warning("CUDA present but unusable (%s); falling back to CPU", msg)
        elif prefer != "auto":
            raise RuntimeError("CUDA backend requested but no CUDA device / CuPy found")
    log.info("Using CPU backend (Cython, %d threads)", os.cpu_count() or 1)
    return CPUBackend()
