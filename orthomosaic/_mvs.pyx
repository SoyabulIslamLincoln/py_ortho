# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
# cython: initializedcheck=False, nonecheck=False
"""Dense-stereo CPU kernel: project a ground grid at per-cell heights into a view and sample it."""
import numpy as np
cimport numpy as cnp
from libc.stdlib cimport malloc, free
from libc.math cimport floor
from libc.stdint cimport uint8_t

cnp.import_array()


def sample_view(const float[:, :, ::1] img, const double[:, ::1] R, const double[::1] C,
                double f, double k1, double k2, double k3, double cx, double cy,
                double X0, double Y0, double gsd, const float[:, ::1] Z,
                float[:, :, ::1] out, uint8_t[:, ::1] valid):
    """Grid cell (r, c) is the ground point (X0 + (c+.5) gsd, Y0 - (r+.5) gsd, Z[r, c]).
    Writes the bilinear sample of `img` (h, w, ch) to out[r, c] and 1/0 to valid[r, c].
    Camera parameters are in the pixel units of `img` (i.e. already scaled)."""
    cdef Py_ssize_t H = Z.shape[0], W = Z.shape[1], h = img.shape[0], w = img.shape[1]
    cdef Py_ssize_t nch = img.shape[2], r, c, ch, x0, y0, x1, y1
    cdef double X, Y, dx, dy, dz, xc, yc, zc, nx, ny, r2, d, u, v, fx, fy
    with nogil:
        for r in range(H):
            Y = Y0 - (r + 0.5) * gsd
            for c in range(W):
                X = X0 + (c + 0.5) * gsd
                dx = X - C[0]
                dy = Y - C[1]
                dz = Z[r, c] - C[2]
                zc = R[2, 0] * dx + R[2, 1] * dy + R[2, 2] * dz
                valid[r, c] = 0
                if zc <= 1e-6:
                    continue
                xc = R[0, 0] * dx + R[0, 1] * dy + R[0, 2] * dz
                yc = R[1, 0] * dx + R[1, 1] * dy + R[1, 2] * dz
                nx = xc / zc
                ny = yc / zc
                r2 = nx * nx + ny * ny
                d = 1.0 + r2 * (k1 + r2 * (k2 + r2 * k3))
                u = f * d * nx + cx
                v = f * d * ny + cy
                if u < 0 or v < 0 or u > w - 1 or v > h - 1:
                    continue
                x0 = <Py_ssize_t>u
                y0 = <Py_ssize_t>v
                x1 = x0 + 1 if x0 + 1 < w else x0
                y1 = y0 + 1 if y0 + 1 < h else y0
                fx = u - x0
                fy = v - y0
                for ch in range(nch):
                    out[r, c, ch] = <float>((1 - fy) * ((1 - fx) * img[y0, x0, ch] + fx * img[y0, x1, ch])
                                            + fy * ((1 - fx) * img[y1, x0, ch] + fx * img[y1, x1, ch]))
                valid[r, c] = 1


cdef void _sgm_dir(const float* C, float* L, Py_ssize_t D, Py_ssize_t H, Py_ssize_t W, int dy, int dx,
                   float P1, float P2) noexcept nogil:
    """One SGM path direction on (H, W, D)-ordered buffers (the depth values of a cell are
    contiguous); per value the same operations as `sgm` below."""
    cdef Py_ssize_t yy, xx, y, x, py, px, d
    cdef float best_prev, v, a, b
    cdef const float* c
    cdef float* cur
    cdef const float* prv
    for yy in range(H):
        y = yy if dy >= 0 else H - 1 - yy
        for xx in range(W):
            x = xx if dx >= 0 else W - 1 - xx
            py = y - dy
            px = x - dx
            c = C + (y * W + x) * D
            cur = L + (y * W + x) * D
            if py < 0 or py >= H or px < 0 or px >= W:
                for d in range(D):
                    cur[d] = c[d]
            else:
                prv = L + (py * W + px) * D
                best_prev = prv[0]
                for d in range(1, D):
                    if prv[d] < best_prev:
                        best_prev = prv[d]
                for d in range(D):
                    v = prv[d]
                    if d > 0:
                        a = prv[d - 1] + P1
                        if a < v:
                            v = a
                    if d < D - 1:
                        b = prv[d + 1] + P1
                        if b < v:
                            v = b
                    if best_prev + P2 < v:
                        v = best_prev + P2
                    cur[d] = c[d] + v - best_prev


def sgm(const float[:, :, ::1] cost, float P1, float P2, int workers=1):
    """Semi-global matching: aggregate a (D, H, W) cost volume along 8 directions.
    L_r(p, d) = C(p, d) + min(L(p-r, d), L(p-r, d+-1) + P1, min_k L(p-r, k) + P2) - min_k L(p-r, k)
    Returns the summed aggregated volume (D, H, W) float32.

    The paths are computed on a (H, W, D) copy of the volume (contiguous per cell) and, with
    workers > 1, the 8 directions in parallel; each value sees the same operations and the
    directions are summed in the same order, so the result is identical."""
    cdef Py_ssize_t D = cost.shape[0], H = cost.shape[1], W = cost.shape[2]
    if D > 0 and H > 0 and W > 0:
        return _sgm_hwd(cost, P1, P2, max(1, workers))
    agg_np = np.zeros((D, H, W), np.float32)
    L_np = np.zeros((D, H, W), np.float32)
    cdef float[:, :, ::1] agg = agg_np
    cdef float[:, :, ::1] L = L_np
    cdef int dirs[8][2]
    dirs[0][:] = [0, 1]
    dirs[1][:] = [0, -1]
    dirs[2][:] = [1, 0]
    dirs[3][:] = [-1, 0]
    dirs[4][:] = [1, 1]
    dirs[5][:] = [1, -1]
    dirs[6][:] = [-1, 1]
    dirs[7][:] = [-1, -1]
    cdef int k, dy, dx
    cdef Py_ssize_t yy, xx, y, x, py, px, d
    cdef float best_prev, v, a, b
    with nogil:
        for k in range(8):
            dy = dirs[k][0]
            dx = dirs[k][1]
            for yy in range(H):
                y = yy if dy >= 0 else H - 1 - yy
                for xx in range(W):
                    x = xx if dx >= 0 else W - 1 - xx
                    py = y - dy
                    px = x - dx
                    if py < 0 or py >= H or px < 0 or px >= W:
                        for d in range(D):
                            L[d, y, x] = cost[d, y, x]
                    else:
                        best_prev = L[0, py, px]
                        for d in range(1, D):
                            if L[d, py, px] < best_prev:
                                best_prev = L[d, py, px]
                        for d in range(D):
                            v = L[d, py, px]
                            if d > 0:
                                a = L[d - 1, py, px] + P1
                                if a < v:
                                    v = a
                            if d < D - 1:
                                b = L[d + 1, py, px] + P1
                                if b < v:
                                    v = b
                            if best_prev + P2 < v:
                                v = best_prev + P2
                            L[d, y, x] = cost[d, y, x] + v - best_prev
                    for d in range(D):
                        agg[d, y, x] += L[d, y, x]
    return agg_np


cdef void _sgm_to_hwd(const float[:, :, ::1] cost, float* C, Py_ssize_t y0, Py_ssize_t y1) noexcept nogil:
    cdef Py_ssize_t D = cost.shape[0], W = cost.shape[2], y, x, d
    for d in range(D):
        for y in range(y0, y1):
            for x in range(W):
                C[(y * W + x) * D + d] = cost[d, y, x]


cdef void _sgm_from_hwd(const float* A, float[:, :, ::1] agg, Py_ssize_t y0, Py_ssize_t y1) noexcept nogil:
    cdef Py_ssize_t D = agg.shape[0], W = agg.shape[2], y, x, d
    for d in range(D):
        for y in range(y0, y1):
            for x in range(W):
                agg[d, y, x] = A[(y * W + x) * D + d]


def _sgm_hwd(const float[:, :, ::1] cost, float P1, float P2, int workers):
    """sgm on (H, W, D)-ordered buffers. workers > 1: two rounds of four directions computed in
    parallel, each round added to the sum in direction order (6 volumes of memory at most)."""
    cdef Py_ssize_t D = cost.shape[0], H = cost.shape[1], W = cost.shape[2], n = D * H * W, i
    cdef int dirs[8][2]
    dirs[0][:] = [0, 1]
    dirs[1][:] = [0, -1]
    dirs[2][:] = [1, 0]
    dirs[3][:] = [-1, 0]
    dirs[4][:] = [1, 1]
    dirs[5][:] = [1, -1]
    dirs[6][:] = [-1, 1]
    dirs[7][:] = [-1, -1]
    cdef int nbuf = 4 if workers > 1 else 1, k, rnd
    cdef float* C = <float*>malloc(n * sizeof(float))
    cdef float* Ls = <float*>malloc(nbuf * n * sizeof(float))
    cdef float* A = <float*>malloc(n * sizeof(float))
    if C == NULL or Ls == NULL or A == NULL:
        free(C); free(Ls); free(A)
        raise MemoryError()
    agg_np = np.empty((D, H, W), np.float32)
    cdef float[:, :, ::1] agg = agg_np
    try:
        if nbuf == 1:
            with nogil:
                _sgm_to_hwd(cost, C, 0, H)
                for i in range(n):
                    A[i] = 0
                for k in range(8):
                    _sgm_dir(C, Ls, D, H, W, dirs[k][0], dirs[k][1], P1, P2)
                    for i in range(n):
                        A[i] += Ls[i]
                _sgm_from_hwd(A, agg, 0, H)
        else:
            from concurrent.futures import ThreadPoolExecutor
            nt = min(workers, 8)
            rows = [(H * t // nt, H * (t + 1) // nt) for t in range(nt) if H * (t + 1) // nt > H * t // nt]
            parts = [(n * t // nt, n * (t + 1) // nt) for t in range(nt)]

            def to_hwd(r):
                cdef Py_ssize_t a = r[0], b = r[1]
                with nogil:
                    _sgm_to_hwd(cost, C, a, b)

            def from_hwd(r):
                cdef Py_ssize_t a = r[0], b = r[1]
                with nogil:
                    _sgm_from_hwd(A, agg, a, b)

            def one(int kk):
                with nogil:
                    _sgm_dir(C, Ls + (kk % 4) * n, D, H, W, dirs[kk][0], dirs[kk][1], P1, P2)

            def add_round(r):
                cdef Py_ssize_t a = r[0], b = r[1], q
                cdef int first = r[2], kk
                with nogil:
                    for q in range(a, b):
                        if first:
                            A[q] = 0
                        for kk in range(4):              # directions in their original order
                            A[q] += Ls[kk * n + q]

            with ThreadPoolExecutor(nt) as ex:
                list(ex.map(to_hwd, rows))
                for rnd in range(2):
                    list(ex.map(one, range(4 * rnd, 4 * rnd + 4)))
                    list(ex.map(add_round, [(a, b, 1 if rnd == 0 else 0) for a, b in parts]))
                list(ex.map(from_hwd, rows))
    finally:
        free(C)
        free(Ls)
        free(A)
    return agg_np


def bilateral(const float[:, ::1] Z, const unsigned char[:, ::1] valid, int radius,
              float sigma_r, float sigma_s):
    """Edge-preserving surface filter. Each valid cell becomes the range- and distance-weighted
    average of valid neighbours whose height is close to it, so flat surfaces (roofs, ground) are
    smoothed while the tall step at a building edge is preserved (its neighbours across the step are
    down-weighted by the range term). NaN/invalid cells are ignored and stay unchanged."""
    cdef Py_ssize_t H = Z.shape[0], W = Z.shape[1], y, x, yy, xx
    cdef int dy, dx
    out_np = np.array(Z, copy=True)
    cdef float[:, ::1] out = out_np
    cdef float z0, zk, wsum, asum, dr, w
    cdef float inv_r = 1.0 / (2.0 * sigma_r * sigma_r)
    cdef float inv_s = 1.0 / (2.0 * sigma_s * sigma_s)
    cdef float[:, ::1] sw = np.empty((2 * radius + 1, 2 * radius + 1), np.float32)
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            sw[dy + radius, dx + radius] = <float>(2.718281828 ** (-(dy * dy + dx * dx) * inv_s))
    with nogil:
        for y in range(H):
            for x in range(W):
                if valid[y, x] == 0:
                    continue
                z0 = Z[y, x]
                wsum = 0.0
                asum = 0.0
                for dy in range(-radius, radius + 1):
                    yy = y + dy
                    if yy < 0 or yy >= H:
                        continue
                    for dx in range(-radius, radius + 1):
                        xx = x + dx
                        if xx < 0 or xx >= W or valid[yy, xx] == 0:
                            continue
                        zk = Z[yy, xx]
                        dr = zk - z0
                        w = sw[dy + radius, dx + radius] * <float>(2.718281828 ** (-dr * dr * inv_r))
                        wsum += w
                        asum += w * zk
                if wsum > 0:
                    out[y, x] = asum / wsum
    return out_np
