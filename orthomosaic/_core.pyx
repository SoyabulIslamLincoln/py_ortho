# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
# cython: initializedcheck=False, nonecheck=False
"""
Low-level CPU kernels for the orthomosaic pipeline.

Every heavy loop runs with the GIL released, so callers can parallelise
across images / pairs / output blocks with plain Python threads.
"""
import numpy as np
cimport numpy as cnp
from libc.math cimport sqrt, exp, floor, ceil, atan2, cos, sin, fabs, pow, log
from libc.stdint cimport uint8_t, uint32_t, uint64_t, int32_t

cnp.import_array()

cdef extern from *:
    """
    #if defined(_MSC_VER)
    #include <intrin.h>
    static inline int om_popcount64(unsigned long long x) { return (int)__popcnt64(x); }
    #else
    static inline int om_popcount64(unsigned long long x) { return __builtin_popcountll(x); }
    #endif
    """
    int om_popcount64(unsigned long long x) nogil


# --------------------------------------------------------------------------
# Image basics
# --------------------------------------------------------------------------

def rgb_to_gray(const uint8_t[:, :, ::1] rgb):
    cdef Py_ssize_t h = rgb.shape[0], w = rgb.shape[1], y, x
    out = np.empty((h, w), np.float32)
    cdef float[:, ::1] o = out
    with nogil:
        for y in range(h):
            for x in range(w):
                o[y, x] = 0.299 * rgb[y, x, 0] + 0.587 * rgb[y, x, 1] + 0.114 * rgb[y, x, 2]
    return out


def gaussian_blur(const float[:, ::1] img, double sigma):
    """Separable Gaussian blur with clamped borders."""
    cdef Py_ssize_t h = img.shape[0], w = img.shape[1], y, x, k, xx, yy
    cdef int r = max(1, <int>(3.0 * sigma + 0.5))
    kern_np = np.exp(-0.5 * (np.arange(-r, r + 1, dtype=np.float64) / sigma) ** 2)
    kern_np = (kern_np / kern_np.sum()).astype(np.float32)
    cdef float[::1] kern = kern_np
    tmp_np = np.empty((h, w), np.float32)
    out_np = np.empty((h, w), np.float32)
    cdef float[:, ::1] tmp = tmp_np
    cdef float[:, ::1] out = out_np
    cdef float s
    with nogil:
        for y in range(h):
            for x in range(w):
                s = 0
                if x >= r and x < w - r:          # interior: no clamping
                    for k in range(-r, r + 1):
                        s = s + kern[k + r] * img[y, x + k]
                else:
                    for k in range(-r, r + 1):
                        xx = x + k
                        if xx < 0:
                            xx = 0
                        elif xx >= w:
                            xx = w - 1
                        s = s + kern[k + r] * img[y, xx]
                tmp[y, x] = s
        for y in range(h):
            if y >= r and y < h - r:              # interior rows: vectorisable over x
                for x in range(w):
                    out[y, x] = 0
                for k in range(-r, r + 1):
                    for x in range(w):
                        out[y, x] = out[y, x] + kern[k + r] * tmp[y + k, x]
            else:
                for x in range(w):
                    s = 0
                    for k in range(-r, r + 1):
                        yy = y + k
                        if yy < 0:
                            yy = 0
                        elif yy >= h:
                            yy = h - 1
                        s = s + kern[k + r] * tmp[yy, x]
                    out[y, x] = s
    return out_np


def resize_bilinear(const float[:, ::1] img, Py_ssize_t nh, Py_ssize_t nw):
    cdef Py_ssize_t h = img.shape[0], w = img.shape[1], y, x, x0, y0, x1, y1
    cdef double sy = <double>h / nh, sx = <double>w / nw, fy, fx, wy, wx
    out_np = np.empty((nh, nw), np.float32)
    cdef float[:, ::1] out = out_np
    with nogil:
        for y in range(nh):
            fy = (y + 0.5) * sy - 0.5
            if fy < 0:
                fy = 0
            if fy > h - 1:
                fy = h - 1
            y0 = <Py_ssize_t>fy
            y1 = y0 + 1 if y0 + 1 < h else y0
            wy = fy - y0
            for x in range(nw):
                fx = (x + 0.5) * sx - 0.5
                if fx < 0:
                    fx = 0
                if fx > w - 1:
                    fx = w - 1
                x0 = <Py_ssize_t>fx
                x1 = x0 + 1 if x0 + 1 < w else x0
                wx = fx - x0
                out[y, x] = <float>((1 - wy) * ((1 - wx) * img[y0, x0] + wx * img[y0, x1])
                                    + wy * ((1 - wx) * img[y1, x0] + wx * img[y1, x1]))
    return out_np


# --------------------------------------------------------------------------
# FAST-9 detector + Harris scoring + non-max suppression
# --------------------------------------------------------------------------

cdef int CIRC_X[16]
cdef int CIRC_Y[16]
CIRC_X[:] = [0, 1, 2, 3, 3, 3, 2, 1, 0, -1, -2, -3, -3, -3, -2, -1]
CIRC_Y[:] = [-3, -3, -2, -1, 0, 1, 2, 3, 3, 3, 2, 1, 0, -1, -2, -3]


cdef inline bint _has_arc9(uint32_t mask) noexcept nogil:
    cdef uint32_t m = mask | (mask << 16)
    cdef uint32_t r = m
    cdef int k
    for k in range(1, 9):
        r &= m >> k
    return r != 0


cdef inline float _harris(const float[:, ::1] img, Py_ssize_t y, Py_ssize_t x) noexcept nogil:
    cdef double sxx = 0, syy = 0, sxy = 0, gx, gy
    cdef Py_ssize_t i, j
    for i in range(-3, 4):
        for j in range(-3, 4):
            gx = img[y + i, x + j + 1] - img[y + i, x + j - 1]
            gy = img[y + i + 1, x + j] - img[y + i - 1, x + j]
            sxx += gx * gx
            syy += gy * gy
            sxy += gx * gy
    return <float>(sxx * syy - sxy * sxy - 0.04 * (sxx + syy) * (sxx + syy))


def detect_corners(const float[:, ::1] img, float threshold, int border):
    """FAST-9 corners scored by Harris response, 3x3 non-max suppressed.

    Returns (xs, ys, scores) as float32 arrays.
    """
    cdef Py_ssize_t h = img.shape[0], w = img.shape[1], y, x, n = 0, i
    if border < 4:
        border = 4
    score_np = np.zeros((h, w), np.float32)
    cdef float[:, ::1] score = score_np
    cdef float p, v, hi, lo, s
    cdef uint32_t bright, dark
    cdef int k, nb, nd
    if h <= 2 * border or w <= 2 * border:
        e = np.empty(0, np.float32)
        return e, e.copy(), e.copy()
    with nogil:
        for y in range(border, h - border):
            for x in range(border, w - border):
                p = img[y, x]
                hi = p + threshold
                lo = p - threshold
                # quick reject on the 4 compass points: any 9-arc contains >= 2 of them
                nb = 0
                nd = 0
                for k in range(0, 16, 4):
                    v = img[y + CIRC_Y[k], x + CIRC_X[k]]
                    if v > hi:
                        nb += 1
                    elif v < lo:
                        nd += 1
                if nb < 2 and nd < 2:
                    continue
                bright = 0
                dark = 0
                for k in range(16):
                    v = img[y + CIRC_Y[k], x + CIRC_X[k]]
                    if v > hi:
                        bright |= (<uint32_t>1 << k)
                    elif v < lo:
                        dark |= (<uint32_t>1 << k)
                if _has_arc9(bright) or _has_arc9(dark):
                    s = _harris(img, y, x)
                    if s > 0:
                        score[y, x] = s
        # non-max suppression (strict on one side to break ties deterministically)
        for y in range(border, h - border):
            for x in range(border, w - border):
                s = score[y, x]
                if s <= 0:
                    continue
                if (s > score[y - 1, x - 1] and s > score[y - 1, x] and s > score[y - 1, x + 1]
                        and s > score[y, x - 1] and s >= score[y, x + 1]
                        and s >= score[y + 1, x - 1] and s >= score[y + 1, x] and s >= score[y + 1, x + 1]):
                    n += 1
    xs_np = np.empty(n, np.float32)
    ys_np = np.empty(n, np.float32)
    sc_np = np.empty(n, np.float32)
    cdef float[::1] xs = xs_np, ys = ys_np, sc = sc_np
    i = 0
    with nogil:
        for y in range(border, h - border):
            for x in range(border, w - border):
                s = score[y, x]
                if s <= 0:
                    continue
                if (s > score[y - 1, x - 1] and s > score[y - 1, x] and s > score[y - 1, x + 1]
                        and s > score[y, x - 1] and s >= score[y, x + 1]
                        and s >= score[y + 1, x - 1] and s >= score[y + 1, x] and s >= score[y + 1, x + 1]):
                    xs[i] = x
                    ys[i] = y
                    sc[i] = s
                    i += 1
    return xs_np, ys_np, sc_np


def harris_subpix(const float[:, ::1] img, const float[::1] xs, const float[::1] ys):
    """Sub-pixel keypoints from the detector response itself (as SIFT does for its DoG peak):
    a parabola through the Harris score at the 3x3 neighbourhood, per axis, shift limited to
    half a pixel. Consistent across views because every view refines the same response peak."""
    cdef Py_ssize_t n = xs.shape[0], k, h = img.shape[0], w = img.shape[1], x, y
    cdef double hl, hc, hr, hu, hd, den, dx, dy
    ox_np = np.empty(n, np.float32)
    oy_np = np.empty(n, np.float32)
    cdef float[::1] ox = ox_np, oy = oy_np
    with nogil:
        for k in range(n):
            x = <Py_ssize_t>(xs[k] + 0.5)
            y = <Py_ssize_t>(ys[k] + 0.5)
            ox[k] = xs[k]
            oy[k] = ys[k]
            if x < 5 or y < 5 or x >= w - 5 or y >= h - 5:
                continue
            hc = _harris(img, y, x)
            hl = _harris(img, y, x - 1)
            hr = _harris(img, y, x + 1)
            hu = _harris(img, y - 1, x)
            hd = _harris(img, y + 1, x)
            den = hl - 2 * hc + hr
            dx = 0.5 * (hl - hr) / den if den < 0 else 0
            den = hu - 2 * hc + hd
            dy = 0.5 * (hu - hd) / den if den < 0 else 0
            if dx > 0.5:
                dx = 0.5
            elif dx < -0.5:
                dx = -0.5
            if dy > 0.5:
                dy = 0.5
            elif dy < -0.5:
                dy = -0.5
            ox[k] = <float>(x + dx)
            oy[k] = <float>(y + dy)
    return ox_np, oy_np


# --------------------------------------------------------------------------
# ORB: orientation by intensity centroid + steered BRIEF
# --------------------------------------------------------------------------

def orientations(const float[:, ::1] img, const float[::1] xs, const float[::1] ys, int radius):
    cdef Py_ssize_t n = xs.shape[0], i
    cdef int dx, dy, x, y, r2 = radius * radius
    cdef double m01, m10, v
    out_np = np.empty(n, np.float32)
    cdef float[::1] out = out_np
    with nogil:
        for i in range(n):
            x = <int>xs[i]
            y = <int>ys[i]
            m01 = 0
            m10 = 0
            for dy in range(-radius, radius + 1):
                for dx in range(-radius, radius + 1):
                    if dx * dx + dy * dy <= r2:
                        v = img[y + dy, x + dx]
                        m10 += dx * v
                        m01 += dy * v
            out[i] = <float>atan2(m01, m10)
    return out_np


def brief_describe(const float[:, ::1] img, const float[::1] xs, const float[::1] ys,
                   const float[::1] angles, const int32_t[:, ::1] pattern):
    """256-bit steered BRIEF. pattern: (256, 4) int32 of (x1, y1, x2, y2)."""
    cdef Py_ssize_t n = xs.shape[0], i, b
    cdef int x, y, ax, ay, bx, by
    cdef double c, s
    out_np = np.zeros((n, 32), np.uint8)
    cdef uint8_t[:, ::1] out = out_np
    with nogil:
        for i in range(n):
            x = <int>xs[i]
            y = <int>ys[i]
            c = cos(angles[i])
            s = sin(angles[i])
            for b in range(256):
                ax = <int>floor(c * pattern[b, 0] - s * pattern[b, 1] + 0.5)
                ay = <int>floor(s * pattern[b, 0] + c * pattern[b, 1] + 0.5)
                bx = <int>floor(c * pattern[b, 2] - s * pattern[b, 3] + 0.5)
                by = <int>floor(s * pattern[b, 2] + c * pattern[b, 3] + 0.5)
                if img[y + ay, x + ax] < img[y + by, x + bx]:
                    out[i, b >> 3] |= <uint8_t>(1 << (b & 7))
    return out_np


# --------------------------------------------------------------------------
# Brute-force Hamming matcher
# --------------------------------------------------------------------------

def match_hamming(const uint64_t[:, ::1] d1, const uint64_t[:, ::1] d2):
    """For each row of d1 return (best index in d2, best distance, 2nd best distance)."""
    cdef Py_ssize_t n1 = d1.shape[0], n2 = d2.shape[0], words = d1.shape[1], i, j, k
    cdef int d, best, second, bi
    idx_np = np.full(n1, -1, np.int32)
    best_np = np.full(n1, 1 << 30, np.int32)
    sec_np = np.full(n1, 1 << 30, np.int32)
    cdef int32_t[::1] idx = idx_np, bst = best_np, sec = sec_np
    if d2.shape[1] != words:
        raise ValueError("descriptor widths must match")
    with nogil:
        for i in range(n1):
            best = 1 << 30
            second = 1 << 30
            bi = -1
            for j in range(n2):
                d = 0
                for k in range(words):
                    d = d + om_popcount64(d1[i, k] ^ d2[j, k])
                if d < best:
                    second = best
                    best = d
                    bi = <int>j
                elif d < second:
                    second = d
            idx[i] = bi
            bst[i] = best
            sec[i] = second
    return idx_np, best_np, sec_np


def match_hamming_mutual(const uint64_t[:, ::1] d1, const uint64_t[:, ::1] d2):
    """One pass over the distance matrix giving, for d1 rows, (best idx, best,
    second best) and, for d2 rows, the best index back into d1 -- everything
    needed for ratio test + mutual check at half the cost of two passes."""
    cdef Py_ssize_t n1 = d1.shape[0], n2 = d2.shape[0], i, j
    cdef int d, best, second, bi
    cdef uint64_t a0, a1, a2, a3
    idx_np = np.full(n1, -1, np.int32)
    best_np = np.full(n1, 1 << 30, np.int32)
    sec_np = np.full(n1, 1 << 30, np.int32)
    back_np = np.full(n2, -1, np.int32)
    colbest_np = np.full(n2, 1 << 30, np.int32)
    cdef int32_t[::1] idx = idx_np, bst = best_np, sec = sec_np, back = back_np, cb = colbest_np
    if d1.shape[1] != 4 or d2.shape[1] != 4:
        raise ValueError("expected 256-bit descriptors")
    with nogil:
        for i in range(n1):
            a0 = d1[i, 0]
            a1 = d1[i, 1]
            a2 = d1[i, 2]
            a3 = d1[i, 3]
            best = 1 << 30
            second = 1 << 30
            bi = -1
            for j in range(n2):
                d = (om_popcount64(a0 ^ d2[j, 0]) + om_popcount64(a1 ^ d2[j, 1])
                     + om_popcount64(a2 ^ d2[j, 2]) + om_popcount64(a3 ^ d2[j, 3]))
                if d < best:
                    second = best
                    best = d
                    bi = <int>j
                elif d < second:
                    second = d
                if d < cb[j]:
                    cb[j] = d
                    back[j] = <int>i
            idx[i] = bi
            bst[i] = best
            sec[i] = second
    return idx_np, best_np, sec_np, back_np


# --------------------------------------------------------------------------
# RANSAC affine (2x3) estimation
# --------------------------------------------------------------------------

cdef inline uint64_t _xorshift(uint64_t* s) noexcept nogil:
    cdef uint64_t x = s[0]
    x ^= x << 13
    x ^= x >> 7
    x ^= x << 17
    s[0] = x
    return x


cdef inline bint _solve_affine3(const double[:, ::1] src, const double[:, ::1] dst,
                                int i0, int i1, int i2, double* A) noexcept nogil:
    """Exact affine through three correspondences (Cramer's rule)."""
    cdef double x0 = src[i0, 0], y0 = src[i0, 1]
    cdef double x1 = src[i1, 0], y1 = src[i1, 1]
    cdef double x2 = src[i2, 0], y2 = src[i2, 1]
    cdef double det = x0 * (y1 - y2) - y0 * (x1 - x2) + (x1 * y2 - x2 * y1)
    cdef double scale = fabs(x1 - x0) + fabs(y1 - y0) + fabs(x2 - x0) + fabs(y2 - y0)
    if fabs(det) < 1e-6 * scale * scale + 1e-12:
        return False
    cdef double inv00 = (y1 - y2) / det, inv01 = (x2 - x1) / det, inv02 = (x1 * y2 - x2 * y1) / det
    cdef double inv10 = (y2 - y0) / det, inv11 = (x0 - x2) / det, inv12 = (x2 * y0 - x0 * y2) / det
    cdef double inv20 = (y0 - y1) / det, inv21 = (x1 - x0) / det, inv22 = (x0 * y1 - x1 * y0) / det
    cdef int r
    cdef double u0, u1, u2
    for r in range(2):
        u0 = dst[i0, r]
        u1 = dst[i1, r]
        u2 = dst[i2, r]
        # [a b c] = [u0 u1 u2] * inv(M) where M rows = (x, y, 1) columns per point
        A[r * 3 + 0] = u0 * inv00 + u1 * inv10 + u2 * inv20
        A[r * 3 + 1] = u0 * inv01 + u1 * inv11 + u2 * inv21
        A[r * 3 + 2] = u0 * inv02 + u1 * inv12 + u2 * inv22
    return True


def ransac_affine(const double[:, ::1] src, const double[:, ::1] dst, int max_iters,
                  double thresh, uint64_t seed=0x9E3779B97F4A7C15, double confidence=0.999):
    """Returns (A (2,3) float64 or None, inlier mask uint8)."""
    cdef Py_ssize_t n = src.shape[0], i
    mask_np = np.zeros(n, np.uint8)
    if n < 3:
        return None, mask_np
    cdef uint8_t[::1] mask = mask_np
    cdef double A[6]
    cdef double best[6]
    cdef int it = 0, i0, i1, i2, cnt, best_cnt = -1, needed = max_iters
    cdef double t2 = thresh * thresh, ex, ey, w, denom
    cdef uint64_t st = seed if seed != 0 else 1
    with nogil:
        while it < needed:
            it += 1
            i0 = <int>(_xorshift(&st) % n)
            i1 = <int>(_xorshift(&st) % n)
            i2 = <int>(_xorshift(&st) % n)
            if i0 == i1 or i1 == i2 or i0 == i2:
                continue
            if not _solve_affine3(src, dst, i0, i1, i2, A):
                continue
            cnt = 0
            for i in range(n):
                ex = A[0] * src[i, 0] + A[1] * src[i, 1] + A[2] - dst[i, 0]
                ey = A[3] * src[i, 0] + A[4] * src[i, 1] + A[5] - dst[i, 1]
                if ex * ex + ey * ey < t2:
                    cnt += 1
            if cnt > best_cnt:
                best_cnt = cnt
                for i in range(6):
                    best[i] = A[i]
                w = <double>cnt / n
                denom = log(1.0 - w * w * w + 1e-12)
                if denom < 0:
                    i0 = <int>(log(1.0 - confidence) / denom) + 1
                    if i0 < needed:
                        needed = i0 if i0 > 50 else 50
        if best_cnt >= 3:
            for i in range(n):
                ex = best[0] * src[i, 0] + best[1] * src[i, 1] + best[2] - dst[i, 0]
                ey = best[3] * src[i, 0] + best[4] * src[i, 1] + best[5] - dst[i, 1]
                mask[i] = 1 if ex * ex + ey * ey < t2 else 0
    if best_cnt < 3:
        return None, mask_np
    A_np = np.array([[best[0], best[1], best[2]], [best[3], best[4], best[5]]])
    # least-squares refit on the consensus set, then re-classify
    src_np = np.asarray(src)
    dst_np = np.asarray(dst)
    X = np.column_stack([src_np, np.ones(n)])
    for _ in range(2):
        m = mask_np.astype(bool)
        if m.sum() < 3:
            break
        A_np = np.linalg.lstsq(X[m], dst_np[m], rcond=None)[0].T
        err = np.sum((X @ A_np.T - dst_np) ** 2, axis=1)
        mask_np = (err < t2).astype(np.uint8)
    return A_np, mask_np


# --------------------------------------------------------------------------
# Rendering: inverse-warp an image into an output block and blend
# --------------------------------------------------------------------------

cdef inline void _clip_range(double a, double b, double lo, double hi,
                             double* cmin, double* cmax) noexcept nogil:
    """Restrict [cmin, cmax] to columns c where lo <= a*c + b <= hi."""
    if fabs(a) < 1e-12:
        if b < lo or b > hi:
            cmin[0] = 1
            cmax[0] = 0
        return
    cdef double c1 = (lo - b) / a, c2 = (hi - b) / a, t
    if c1 > c2:
        t = c1
        c1 = c2
        c2 = t
    if c1 > cmin[0]:
        cmin[0] = c1
    if c2 < cmax[0]:
        cmax[0] = c2


def warp_accumulate(float[:, :, ::1] acc, float[:, ::1] wsum, const uint8_t[:, :, ::1] img,
                    const double[::1] M, const float[::1] gain, double power, int mode):
    """Blend `img` into the block.

    M maps block pixel (col, row) -> source pixel (u, v):
        u = M0*col + M1*row + M2,  v = M3*col + M4*row + M5
    mode 0 = feather (weighted average), mode 1 = max-weight (seamline-like).
    Weight = (distance to nearest image border / half the short side) ** power.
    """
    cdef Py_ssize_t H = acc.shape[0], W = acc.shape[1]
    cdef Py_ssize_t h = img.shape[0], w = img.shape[1]
    cdef Py_ssize_t r, c, c0, c1, x0, y0, x1, y1, ch
    cdef double u, v, fx, fy, d, wt, cmin, cmax, half = 0.5 * (w if w < h else h)
    cdef double val, g
    if h < 2 or w < 2:
        return
    with nogil:
        for r in range(H):
            cmin = 0
            cmax = W - 1
            _clip_range(M[0], M[1] * r + M[2], 0, w - 1, &cmin, &cmax)
            _clip_range(M[3], M[4] * r + M[5], 0, h - 1, &cmin, &cmax)
            if cmin > cmax:
                continue
            c0 = <Py_ssize_t>ceil(cmin)
            c1 = <Py_ssize_t>floor(cmax)
            for c in range(c0, c1 + 1):
                u = M[0] * c + M[1] * r + M[2]
                v = M[3] * c + M[4] * r + M[5]
                if u < 0 or v < 0 or u > w - 1 or v > h - 1:
                    continue
                d = u + 0.5
                if w - 0.5 - u < d:
                    d = w - 0.5 - u
                if v + 0.5 < d:
                    d = v + 0.5
                if h - 0.5 - v < d:
                    d = h - 0.5 - v
                wt = d / half
                if power != 1.0:
                    wt = pow(wt, power)
                if wt <= 0:
                    continue
                x0 = <Py_ssize_t>u
                y0 = <Py_ssize_t>v
                x1 = x0 + 1 if x0 + 1 < w else x0
                y1 = y0 + 1 if y0 + 1 < h else y0
                fx = u - x0
                fy = v - y0
                if mode == 1:
                    if wt <= wsum[r, c]:
                        continue
                    wsum[r, c] = <float>wt
                for ch in range(3):
                    val = ((1 - fy) * ((1 - fx) * img[y0, x0, ch] + fx * img[y0, x1, ch])
                           + fy * ((1 - fx) * img[y1, x0, ch] + fx * img[y1, x1, ch]))
                    g = gain[ch]
                    if mode == 1:
                        acc[r, c, ch] = <float>(val * g)
                    else:
                        acc[r, c, ch] += <float>(val * g * wt)
                if mode == 0:
                    wsum[r, c] += <float>wt


def finalize(const float[:, :, ::1] acc, const float[:, ::1] wsum, int mode):
    """Convert accumulators into an RGBA uint8 block (alpha = coverage)."""
    cdef Py_ssize_t H = acc.shape[0], W = acc.shape[1], r, c, ch
    cdef double s, v
    out_np = np.zeros((H, W, 4), np.uint8)
    cdef uint8_t[:, :, ::1] out = out_np
    with nogil:
        for r in range(H):
            for c in range(W):
                s = wsum[r, c]
                if s <= 0:
                    continue
                for ch in range(3):
                    v = acc[r, c, ch] / s if mode == 0 else acc[r, c, ch]
                    if v < 0:
                        v = 0
                    elif v > 255:
                        v = 255
                    out[r, c, ch] = <uint8_t>(v + 0.5)
                out[r, c, 3] = 255
    return out_np
