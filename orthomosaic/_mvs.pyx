# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
# cython: initializedcheck=False, nonecheck=False
"""Dense-stereo CPU kernel: project a ground grid at per-cell heights into a view and sample it."""
import numpy as np
cimport numpy as cnp
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


def sgm(const float[:, :, ::1] cost, float P1, float P2):
    """Semi-global matching: aggregate a (D, H, W) cost volume along 8 directions.
    L_r(p, d) = C(p, d) + min(L(p-r, d), L(p-r, d+-1) + P1, min_k L(p-r, k) + P2) - min_k L(p-r, k)
    Returns the summed aggregated volume (D, H, W) float32."""
    cdef Py_ssize_t D = cost.shape[0], H = cost.shape[1], W = cost.shape[2]
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
