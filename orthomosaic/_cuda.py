"""CUDA backend. Kernels are compiled at runtime by NVRTC through CuPy,
so no nvcc / toolkit build step is required -- only a driver + cupy wheel."""
from __future__ import annotations

import cupy as cp
import numpy as np

_SRC = r"""
extern "C" __global__
void hamming_match(const unsigned long long* __restrict__ d1,
                   const unsigned long long* __restrict__ d2,
                   const int n1, const int n2,
                   int* __restrict__ out_idx, int* __restrict__ out_best,
                   int* __restrict__ out_second)
{
    // One thread per query; the train set is streamed through shared memory.
    extern __shared__ unsigned long long tile[];
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long q0 = 0, q1 = 0, q2 = 0, q3 = 0;
    if (i < n1) {
        q0 = d1[4 * i]; q1 = d1[4 * i + 1]; q2 = d1[4 * i + 2]; q3 = d1[4 * i + 3];
    }
    int best = 1 << 30, second = 1 << 30, bi = -1;
    for (int base = 0; base < n2; base += blockDim.x) {
        const int j = base + threadIdx.x;
        if (j < n2) {
            tile[4 * threadIdx.x + 0] = d2[4 * j + 0];
            tile[4 * threadIdx.x + 1] = d2[4 * j + 1];
            tile[4 * threadIdx.x + 2] = d2[4 * j + 2];
            tile[4 * threadIdx.x + 3] = d2[4 * j + 3];
        }
        __syncthreads();
        const int lim = min((int)blockDim.x, n2 - base);
        if (i < n1) {
            for (int k = 0; k < lim; ++k) {
                const int d = __popcll(q0 ^ tile[4 * k]) + __popcll(q1 ^ tile[4 * k + 1])
                            + __popcll(q2 ^ tile[4 * k + 2]) + __popcll(q3 ^ tile[4 * k + 3]);
                if (d < best) { second = best; best = d; bi = base + k; }
                else if (d < second) { second = d; }
            }
        }
        __syncthreads();
    }
    if (i < n1) { out_idx[i] = bi; out_best[i] = best; out_second[i] = second; }
}

extern "C" __global__
void warp_accumulate(float* __restrict__ acc, float* __restrict__ wsum, const int H, const int W,
                     const unsigned char* __restrict__ img, const int h, const int w,
                     const double m0, const double m1, const double m2,
                     const double m3, const double m4, const double m5,
                     const float g0, const float g1, const float g2,
                     const float power, const int mode)
{
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    const int r = blockIdx.y * blockDim.y + threadIdx.y;
    if (c >= W || r >= H) return;
    const double u = m0 * c + m1 * r + m2;
    const double v = m3 * c + m4 * r + m5;
    if (u < 0.0 || v < 0.0 || u > w - 1 || v > h - 1) return;
    double d = fmin(fmin(u + 0.5, w - 0.5 - u), fmin(v + 0.5, h - 0.5 - v));
    const double half = 0.5 * (double)min(w, h);
    float wt = (float)(d / half);
    if (power != 1.0f) wt = powf(wt, power);
    if (wt <= 0.0f) return;
    const int idx = r * W + c;
    if (mode == 1 && wt <= wsum[idx]) return;
    const int x0 = (int)u, y0 = (int)v;
    const int x1 = min(x0 + 1, w - 1), y1 = min(y0 + 1, h - 1);
    const float fx = (float)(u - x0), fy = (float)(v - y0);
    const unsigned char* p00 = img + 3 * (y0 * w + x0);
    const unsigned char* p01 = img + 3 * (y0 * w + x1);
    const unsigned char* p10 = img + 3 * (y1 * w + x0);
    const unsigned char* p11 = img + 3 * (y1 * w + x1);
    const float g[3] = {g0, g1, g2};
    for (int ch = 0; ch < 3; ++ch) {
        const float val = (1.f - fy) * ((1.f - fx) * p00[ch] + fx * p01[ch])
                        + fy * ((1.f - fx) * p10[ch] + fx * p11[ch]);
        if (mode == 1) acc[3 * idx + ch] = val * g[ch];
        else           acc[3 * idx + ch] += val * g[ch] * wt;
    }
    if (mode == 1) wsum[idx] = wt;
    else           wsum[idx] += wt;
}
"""

_module = cp.RawModule(code=_SRC, options=("-std=c++11",))
_k_match = _module.get_function("hamming_match")
_k_warp = _module.get_function("warp_accumulate")


class CUDABackend:
    name = "cuda"
    parallel_blocks = False  # the GPU already parallelises inside each block

    def __init__(self):
        dev = cp.cuda.Device()
        props = cp.cuda.runtime.getDeviceProperties(dev.id)
        self.device_name = props["name"].decode() if isinstance(props["name"], bytes) else props["name"]
        free, total = cp.cuda.runtime.memGetInfo()
        self.free_mem = free

    def match(self, d1: np.ndarray, d2: np.ndarray):
        n1, n2 = len(d1), len(d2)
        a = cp.asarray(np.ascontiguousarray(d1, np.uint8).view(np.uint64).reshape(n1, 4))
        b = cp.asarray(np.ascontiguousarray(d2, np.uint8).view(np.uint64).reshape(n2, 4))
        idx = cp.full(n1, -1, cp.int32)
        best = cp.full(n1, 1 << 30, cp.int32)
        second = cp.full(n1, 1 << 30, cp.int32)
        threads = 128
        _k_match(((n1 + threads - 1) // threads,), (threads,),
                 (a, b, np.int32(n1), np.int32(n2), idx, best, second),
                 shared_mem=threads * 4 * 8)
        return cp.asnumpy(idx), cp.asnumpy(best), cp.asnumpy(second)

    def match_mutual(self, d1: np.ndarray, d2: np.ndarray):
        idx12, best12, second12 = self.match(d1, d2)
        idx21, _, _ = self.match(d2, d1)
        return idx12, best12, second12, idx21

    def new_block(self, H: int, W: int):
        return cp.zeros((H, W, 3), cp.float32), cp.zeros((H, W), cp.float32)

    def upload(self, img: np.ndarray):
        return cp.asarray(np.ascontiguousarray(img))

    def image_nbytes(self, dev_img) -> int:
        return int(dev_img.nbytes)

    def warp_accumulate(self, block, dev_img, M, gain, power, mode):
        acc, wsum = block
        H, W = wsum.shape
        h, w = dev_img.shape[:2]
        m = np.asarray(M, np.float64).ravel()
        g = np.asarray(gain, np.float32)
        bs = (32, 8)
        grid = ((W + bs[0] - 1) // bs[0], (H + bs[1] - 1) // bs[1])
        _k_warp(grid, bs, (acc, wsum, np.int32(H), np.int32(W), dev_img, np.int32(h), np.int32(w),
                           *[np.float64(x) for x in m], *[np.float32(x) for x in g],
                           np.float32(power), np.int32(mode)))

    def finalize(self, block, mode):
        acc, wsum = block
        covered = wsum > 0
        if mode == 0:
            rgb = acc / cp.maximum(wsum, 1e-12)[..., None]
        else:
            rgb = acc
        rgb = cp.clip(cp.rint(rgb), 0, 255).astype(cp.uint8)
        out = cp.empty(wsum.shape + (4,), cp.uint8)
        out[..., :3] = rgb * covered[..., None]
        out[..., 3] = covered.astype(cp.uint8) * 255
        return cp.asnumpy(out)
