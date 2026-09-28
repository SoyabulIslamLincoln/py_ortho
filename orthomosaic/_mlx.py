"""Apple-silicon GPU backend (Metal, via MLX custom kernels).

Same interface as the CPU and CUDA backends. Metal has no float64, so kernels use float32
(coordinates are local metres / pixels, well within float32 precision) and the dense
box filter uses shifted sums instead of float64 cumulative sums.
"""
from __future__ import annotations

import mlx.core as mx
import numpy as np

_HAMMING = r"""
    uint i = thread_position_in_grid.x;
    uint n1 = dims[0], n2 = dims[1];
    if (i >= n1) return;
    uint q[8];
    for (int k = 0; k < 8; ++k) q[k] = d1[i * 8 + k];
    int best = 1 << 30, second = 1 << 30, bi = -1;
    for (uint j = 0; j < n2; ++j) {
        int d = 0;
        for (int k = 0; k < 8; ++k) d += popcount(q[k] ^ d2[j * 8 + k]);
        if (d < best) { second = best; best = d; bi = int(j); }
        else if (d < second) { second = d; }
    }
    idx[i] = bi; bst[i] = best; sec[i] = second;
"""

_WARP = r"""
    uint c = thread_position_in_grid.x, r = thread_position_in_grid.y;
    int W = ip[0], H = ip[1], w = ip[2], h = ip[3], mode = ip[4];
    if (c >= uint(W) || r >= uint(H)) return;
    uint idx = r * W + c;
    float a0 = acc[idx * 3], a1 = acc[idx * 3 + 1], a2 = acc[idx * 3 + 2], ws = wsum[idx];
    acc_out[idx * 3] = a0; acc_out[idx * 3 + 1] = a1; acc_out[idx * 3 + 2] = a2; wsum_out[idx] = ws;
    float u = fp[0] * c + fp[1] * r + fp[2];
    float v = fp[3] * c + fp[4] * r + fp[5];
    if (u < 0.0f || v < 0.0f || u > w - 1 || v > h - 1) return;
    float d = min(min(u + 0.5f, w - 0.5f - u), min(v + 0.5f, h - 0.5f - v));
    float wt = d / (0.5f * float(min(w, h)));
    if (fp[9] != 1.0f) wt = pow(wt, fp[9]);
    if (wt <= 0.0f) return;
    if (mode == 1 && wt <= ws) return;
    int x0 = int(u), y0 = int(v);
    int x1 = min(x0 + 1, w - 1), y1 = min(y0 + 1, h - 1);
    float fx = u - x0, fy = v - y0;
    float val[3];
    for (int ch = 0; ch < 3; ++ch) {
        float p00 = img[(y0 * w + x0) * 3 + ch], p01 = img[(y0 * w + x1) * 3 + ch];
        float p10 = img[(y1 * w + x0) * 3 + ch], p11 = img[(y1 * w + x1) * 3 + ch];
        val[ch] = ((1.0f - fy) * ((1.0f - fx) * p00 + fx * p01) + fy * ((1.0f - fx) * p10 + fx * p11)) * fp[6 + ch];
    }
    if (mode == 1) {
        acc_out[idx * 3] = val[0]; acc_out[idx * 3 + 1] = val[1]; acc_out[idx * 3 + 2] = val[2];
        wsum_out[idx] = wt;
    } else {
        acc_out[idx * 3] = a0 + val[0] * wt; acc_out[idx * 3 + 1] = a1 + val[1] * wt;
        acc_out[idx * 3 + 2] = a2 + val[2] * wt; wsum_out[idx] = ws + wt;
    }
"""

_SAMPLE = r"""
    uint c = thread_position_in_grid.x, r = thread_position_in_grid.y;
    int h = ip[0], w = ip[1], nch = ip[2], H = ip[3], W = ip[4], Hs = ip[5];
    if (c >= uint(W) || r >= uint(H)) return;
    uint idx = r * W + c;
    uint rl = r % uint(Hs);            // row within one height slice (Z may be a (D, Hs, W) stack)
    float dx = fp[17] + (float(c) + 0.5f) * fp[19] - fp[9];
    float dy = fp[18] - (float(rl) + 0.5f) * fp[19] - fp[10];
    float dz = Z[idx] - fp[11];
    float zc = fp[6] * dx + fp[7] * dy + fp[8] * dz;
    if (zc <= 1e-6f) return;
    float nx = (fp[0] * dx + fp[1] * dy + fp[2] * dz) / zc;
    float ny = (fp[3] * dx + fp[4] * dy + fp[5] * dz) / zc;
    float r2 = nx * nx + ny * ny;
    float dd = 1.0f + fp[13] * r2 + fp[14] * r2 * r2;
    float u = fp[12] * dd * nx + fp[15], v = fp[12] * dd * ny + fp[16];
    if (u < 0.0f || v < 0.0f || u > w - 1 || v > h - 1) return;
    int x0 = int(u), y0 = int(v);
    int x1 = min(x0 + 1, w - 1), y1 = min(y0 + 1, h - 1);
    float fx = u - x0, fy = v - y0;
    for (int ch = 0; ch < nch; ++ch) {
        float p00 = img[(y0 * w + x0) * nch + ch], p01 = img[(y0 * w + x1) * nch + ch];
        float p10 = img[(y1 * w + x0) * nch + ch], p11 = img[(y1 * w + x1) * nch + ch];
        out[idx * nch + ch] = (1.0f - fy) * ((1.0f - fx) * p00 + fx * p01) + fy * ((1.0f - fx) * p10 + fx * p11);
    }
    valid[idx] = 1;
"""

_k_hamming = mx.fast.metal_kernel(name="om_hamming", input_names=["d1", "d2", "dims"],
                                  output_names=["idx", "bst", "sec"], source=_HAMMING)
_k_warp = mx.fast.metal_kernel(name="om_warp", input_names=["acc", "wsum", "img", "fp", "ip"],
                               output_names=["acc_out", "wsum_out"], source=_WARP)
_k_sample = mx.fast.metal_kernel(name="om_sample", input_names=["img", "Z", "fp", "ip"],
                                 output_names=["out", "valid"], source=_SAMPLE)


def _tg(n, cap=256):
    return max(1, min(cap, int(n)))


class MLXBackend:
    name = "mps"
    parallel_blocks = False  # the GPU parallelises inside each block
    xp = mx

    def __init__(self):
        if not mx.metal.is_available():
            raise RuntimeError("Metal GPU not available")
        mx.set_default_device(mx.gpu)
        # MLX keeps freed GPU buffers for reuse; cap that cache so RAM stays bounded
        # (Apple silicon shares memory between CPU and GPU).
        mx.set_cache_limit(256 * 1024 * 1024)
        self.device_name = "Apple GPU (Metal/MLX)"
        try:
            info = mx.device_info()
            self.device_name = f"{info.get('architecture', 'Apple GPU')} (Metal/MLX)"
        except Exception:
            pass
        self._self_test()

    # -- matching ---------------------------------------------------------------
    def match(self, d1: np.ndarray, d2: np.ndarray):
        n1, n2 = len(d1), len(d2)
        a = mx.array(np.ascontiguousarray(d1, np.uint8).view(np.uint32).reshape(-1))
        b = mx.array(np.ascontiguousarray(d2, np.uint8).view(np.uint32).reshape(-1))
        dims = mx.array(np.array([n1, n2], np.uint32))
        idx, best, sec = _k_hamming(inputs=[a, b, dims], grid=(n1, 1, 1), threadgroup=(_tg(n1), 1, 1),
                                    output_shapes=[(n1,), (n1,), (n1,)],
                                    output_dtypes=[mx.int32, mx.int32, mx.int32])
        return np.array(idx), np.array(best), np.array(sec)

    def match_mutual(self, d1: np.ndarray, d2: np.ndarray):
        idx12, best12, second12 = self.match(d1, d2)
        idx21, _, _ = self.match(d2, d1)
        return idx12, best12, second12, idx21

    # -- 2D rendering -------------------------------------------------------------
    def new_block(self, H: int, W: int):
        return [mx.zeros((H, W, 3), mx.float32), mx.zeros((H, W), mx.float32)]

    def upload(self, img: np.ndarray):
        return mx.array(np.ascontiguousarray(img))

    def image_nbytes(self, dev_img) -> int:
        return int(dev_img.nbytes)

    def warp_accumulate(self, block, dev_img, M, gain, power, mode):
        acc, wsum = block
        H, W = wsum.shape
        h, w = dev_img.shape[:2]
        fp = mx.array(np.concatenate([np.asarray(M, np.float64).ravel(), np.asarray(gain, np.float64).ravel(),
                                      [power]]).astype(np.float32))
        ip = mx.array(np.array([W, H, w, h, mode], np.int32))
        acc2, ws2 = _k_warp(inputs=[acc, wsum, dev_img, fp, ip], grid=(W, H, 1), threadgroup=(32, 8, 1),
                            output_shapes=[acc.shape, wsum.shape], output_dtypes=[mx.float32, mx.float32])
        block[0], block[1] = acc2, ws2

    def finalize(self, block, mode):
        acc, wsum = block
        covered = wsum > 0
        rgb = acc / mx.maximum(wsum, 1e-12)[..., None] if mode == 0 else acc
        rgb = mx.clip(mx.round(rgb), 0, 255).astype(mx.uint8) * covered[..., None].astype(mx.uint8)
        alpha = covered.astype(mx.uint8) * 255
        return np.array(mx.concatenate([rgb, alpha[..., None]], axis=-1))

    # -- dense 3D -----------------------------------------------------------------
    def sample_view(self, img, cam, X0, Y0, gsd, Z):
        R, C, f, k1, k2, cx, cy = cam
        Z = Z.astype(mx.float32) if isinstance(Z, mx.array) else mx.array(np.asarray(Z, np.float32))
        lead = Z.shape[:-2]                        # optional batch of height hypotheses
        Hs, W = Z.shape[-2:]
        H = int(np.prod(lead, dtype=np.int64)) * Hs if lead else Hs
        h, w, nch = img.shape
        fp = mx.array(np.concatenate([np.asarray(R, np.float64).ravel(), np.asarray(C, np.float64).ravel(),
                                      [f, k1, k2, cx, cy, X0, Y0, gsd]]).astype(np.float32))
        ip = mx.array(np.array([h, w, nch, H, W, Hs], np.int32))
        out, valid = _k_sample(inputs=[img, Z, fp, ip], grid=(W, H, 1), threadgroup=(32, 8, 1),
                               output_shapes=[(H, W, nch), (H, W)], output_dtypes=[mx.float32, mx.uint8],
                               init_value=0)
        return out.reshape(*lead, Hs, W, nch), valid.reshape(*lead, Hs, W)

    def box(self, a, r):
        """Mean over a (2r+1)^2 window on the last two axes (zero padded). Row prefix sums
        (small magnitudes, so float32 stays exact enough) + shifted sums down the columns."""
        H, W = a.shape[-2:]
        k = 2 * r + 1
        p = mx.pad(a.astype(mx.float32), [(0, 0)] * (a.ndim - 2) + [(r, r), (r + 1, r)])
        c = mx.cumsum(p, axis=-1)
        s = c[..., :, k:] - c[..., :, :-k]
        q = s[..., 0:H, :]
        for i in range(1, k):
            q = q + s[..., i:i + H, :]
        return q / (k * k)

    def argmin0(self, a):
        """(argmin, min) along axis 0. MLX reductions over a leading axis are slow; for the
        handful of views / height steps here a running elementwise comparison is much faster."""
        best = a[0]
        idx = mx.zeros(best.shape, mx.int32)
        for i in range(1, a.shape[0]):
            m = a[i] < best
            best = mx.where(m, a[i], best)
            idx = mx.where(m, i, idx)
        return idx, best

    def argmax0(self, a):
        idx, best = self.argmin0(-a)
        return idx, -best

    def topk_mean(self, st, k):
        """Mean of the k largest values along axis 0 (odd-even transposition network:
        a few elementwise max/min ops beat a general GPU sort for a handful of views)."""
        v = [st[i] for i in range(st.shape[0])]
        n = len(v)
        for rnd in range(n):
            for i in range(rnd % 2, n - 1, 2):
                hi, lo = mx.maximum(v[i], v[i + 1]), mx.minimum(v[i], v[i + 1])
                v[i], v[i + 1] = hi, lo
        out = v[0]
        for i in range(1, k):
            out = out + v[i]
        return out / k

    _NP = {mx.float32: np.float32, mx.int32: np.int32, mx.uint8: np.uint8, mx.int64: np.int64}

    def asarray(self, a, dtype=None):
        a = np.asarray(a)
        if dtype is not None:
            a = a.astype(self._NP.get(dtype, dtype))
        return mx.array(a)

    def to_numpy(self, a):
        return np.array(a)

    # -- start-up check -------------------------------------------------------------
    def _self_test(self):
        rng = np.random.default_rng(0)
        d = rng.integers(0, 256, (40, 32), dtype=np.uint8)
        idx, best, _ = self.match(d, d)
        if not (idx == np.arange(40)).all() or best.max() != 0:
            raise RuntimeError("Metal self-test failed (matcher)")
        img = self.upload(rng.integers(0, 256, (16, 16, 3), dtype=np.uint8))
        blk = self.new_block(8, 8)
        self.warp_accumulate(blk, img, np.array([[1, 0, 2], [0, 1, 2]], float), np.ones(3), 1.0, 0)
        if self.finalize(blk, 0)[..., 3].max() != 255:
            raise RuntimeError("Metal self-test failed (warp)")
        s, v = self.sample_view(mx.ones((8, 8, 1), mx.float32),
                                (np.diag([1.0, -1.0, -1.0]), np.array([0, 0, 10.0]), 10.0, 0.0, 0.0, 3.5, 3.5),
                                -1.0, 1.0, 0.5, mx.zeros((4, 4), mx.float32))
        if int(np.array(v).sum()) != 16 or abs(float(np.array(s).mean()) - 1.0) > 1e-5:
            raise RuntimeError("Metal self-test failed (sampler)")
