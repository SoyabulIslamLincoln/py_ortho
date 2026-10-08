# cython: boundscheck=False, wraparound=False, cdivision=True, initializedcheck=False, nonecheck=False
"""C kernels for dense matching, DSM rasterisation and orthophoto visibility.

All kernels release the GIL, so the Python thread pools run them on every core in parallel.
"""
from libc.math cimport floor, sqrt, isfinite, NAN, INFINITY, hypot, fabs
from libc.stdlib cimport malloc, free
import numpy as np
cimport numpy as cnp


# ---------------------------------------------------------------- depth maps
def ncc_warp(const float[:, ::1] ref, const float[:, ::1] mu_i, const float[:, ::1] sd_i,
             const float[:, ::1] src, const float[:, :, ::1] ray, const double[:, ::1] Rs, const double[::1] b,
             double f, double k1, double k2, double k3, double cx, double cy,
             const float[:, ::1] inv, int r, float[:, ::1] out,
             float[:, ::1] J, float[:, ::1] V, float[:, :, ::1] hs):
    """NCC of the reference against one source at per-pixel inverse depth `inv`.

    ray: (3, H, W) world direction of each reference pixel (camera-axis depth 1), Rs: source
    rotation (world -> camera), b = Rs (C_ref - C_src). For depth d the source camera point is
    b + d * Rs @ ray. NaN source pixels (masks) are invalid. Window sums are zero-padded box sums
    over (2r+1)^2; out = -1 where fewer than 90 % of the window projects validly.
    J, V: (H, W) scratch; hs: (4, H, W) scratch."""
    cdef Py_ssize_t H = ref.shape[0], W = ref.shape[1], h = src.shape[0], w = src.shape[1]
    cdef Py_ssize_t y, x, x0, y0, k = 2 * r + 1
    cdef double d, X, Y, Z, u, v, nx, ny, r2, dd, fx, fy, a00, a01, a10, a11, val
    cdef double r00 = Rs[0, 0], r01 = Rs[0, 1], r02 = Rs[0, 2], r10 = Rs[1, 0], r11 = Rs[1, 1], r12 = Rs[1, 2]
    cdef double r20 = Rs[2, 0], r21 = Rs[2, 1], r22 = Rs[2, 2], b0 = b[0], b1 = b[1], b2 = b[2]
    cdef double rx, ry, rz, sv, sj, sjj, sij, kk = 1.0 / (k * k), cov, mj, m2, mij, sdj, ncc
    cdef double *cs
    with nogil:
        # 1. warp the source into the reference grid
        for y in range(H):
            for x in range(W):
                d = 1.0 / inv[y, x]
                rx = ray[0, y, x]
                ry = ray[1, y, x]
                rz = ray[2, y, x]
                X = b0 + d * (r00 * rx + r01 * ry + r02 * rz)
                Y = b1 + d * (r10 * rx + r11 * ry + r12 * rz)
                Z = b2 + d * (r20 * rx + r21 * ry + r22 * rz)
                J[y, x] = 0
                V[y, x] = 0
                if Z <= 1e-6:
                    continue
                nx = X / Z
                ny = Y / Z
                r2 = nx * nx + ny * ny
                dd = f * (1 + r2 * (k1 + r2 * (k2 + r2 * k3)))
                u = dd * nx + cx
                v = dd * ny + cy
                if u < 0 or v < 0 or u > w - 1.001 or v > h - 1.001:
                    continue
                x0 = <Py_ssize_t>u
                y0 = <Py_ssize_t>v
                fx = u - x0
                fy = v - y0
                a00 = src[y0, x0]
                a01 = src[y0, x0 + 1]
                a10 = src[y0 + 1, x0]
                a11 = src[y0 + 1, x0 + 1]
                val = (a00 * (1 - fx) + a01 * fx) * (1 - fy) + (a10 * (1 - fx) + a11 * fx) * fy
                if isfinite(val):
                    J[y, x] = <float>val
                    V[y, x] = 1
        # 2. horizontal running window sums of V, J, J^2, J*ref
        for y in range(H):
            sv = 0; sj = 0; sjj = 0; sij = 0
            for x in range(min(r, W)):
                sv += V[y, x]; sj += J[y, x]; sjj += J[y, x] * J[y, x]; sij += J[y, x] * ref[y, x]
            for x in range(W):
                if x + r < W:
                    sv += V[y, x + r]; sj += J[y, x + r]
                    sjj += J[y, x + r] * J[y, x + r]; sij += J[y, x + r] * ref[y, x + r]
                if x - r - 1 >= 0:
                    sv -= V[y, x - r - 1]; sj -= J[y, x - r - 1]
                    sjj -= J[y, x - r - 1] * J[y, x - r - 1]; sij -= J[y, x - r - 1] * ref[y, x - r - 1]
                hs[0, y, x] = <float>sv
                hs[1, y, x] = <float>sj
                hs[2, y, x] = <float>sjj
                hs[3, y, x] = <float>sij
        # 3. vertical running sums (column accumulators) -> NCC
        cs = <double *>malloc(4 * W * sizeof(double))
        for x in range(4 * W):
            cs[x] = 0
        for y in range(min(r, H)):
            for x in range(W):
                cs[x] += hs[0, y, x]; cs[W + x] += hs[1, y, x]
                cs[2 * W + x] += hs[2, y, x]; cs[3 * W + x] += hs[3, y, x]
        for y in range(H):
            for x in range(W):
                if y + r < H:
                    cs[x] += hs[0, y + r, x]; cs[W + x] += hs[1, y + r, x]
                    cs[2 * W + x] += hs[2, y + r, x]; cs[3 * W + x] += hs[3, y + r, x]
                if y - r - 1 >= 0:
                    cs[x] -= hs[0, y - r - 1, x]; cs[W + x] -= hs[1, y - r - 1, x]
                    cs[2 * W + x] -= hs[2, y - r - 1, x]; cs[3 * W + x] -= hs[3, y - r - 1, x]
                cov = cs[x] * kk
                if cov <= 0.9:
                    out[y, x] = -1
                    continue
                mj = cs[W + x] * kk / cov
                m2 = cs[2 * W + x] * kk / cov
                mij = cs[3 * W + x] * kk / cov
                sdj = m2 - mj * mj
                sdj = sqrt(sdj) if sdj > 1e-6 else 1e-3
                ncc = (mij - mu_i[y, x] * mj) / (sd_i[y, x] * sdj)
                out[y, x] = <float>(1.0 if ncc > 1 else (-1.0 if ncc < -1 else ncc))
        free(cs)


def combine(const float[:, :, ::1] S, int top_k, int i, float[:, ::1] best, float[:, ::1] prev,
            float[:, ::1] s_prev_best, float[:, ::1] s_next_best, int[:, ::1] idx):
    """Mean of the top_k source scores at hypothesis i, then the streaming winner update
    (best score, its neighbours' scores for the parabola refinement, best index)."""
    cdef Py_ssize_t n = S.shape[0], H = S.shape[1], W = S.shape[2], y, x, a, c
    cdef float buf[16]
    cdef float t, s
    cdef int kk = top_k if top_k < n else n
    with nogil:
        for y in range(H):
            for x in range(W):
                for a in range(n):
                    buf[a] = S[a, y, x]
                for a in range(1, n):                      # insertion sort, descending (n <= 16)
                    t = buf[a]
                    c = a - 1
                    while c >= 0 and buf[c] < t:
                        buf[c + 1] = buf[c]
                        c -= 1
                    buf[c + 1] = t
                s = 0
                for a in range(kk):
                    s += buf[a]
                s /= kk
                if idx[y, x] == i - 1:
                    s_next_best[y, x] = s
                if s > best[y, x]:
                    best[y, x] = s
                    s_prev_best[y, x] = prev[y, x]
                    s_next_best[y, x] = -2
                    idx[y, x] = i
                prev[y, x] = s


# ---------------------------------------------------------------- DSM
def top_layer(const cnp.int64_t[::1] cell, const double[::1] z, const unsigned char[:, ::1] rgb, Py_ssize_t ncell,
              double gap):
    """Per cell: *median* height of the visible top layer (points within `gap` of the cell's
    highest point) and the mean colour of that layer. O(n): points are bucketed per cell with a
    counting sort, then each (small) bucket is sorted. Returns (Z float32, RGB uint8, count int32)."""
    cdef Py_ssize_t n = cell.shape[0], i, c, a, j, lo, m, s0, e0
    cnt_a = np.zeros(ncell, np.int32)
    off_a = np.zeros(ncell + 1, np.int64)
    idx_a = np.empty(n, np.int64)
    Z_a = np.full(ncell, np.nan, np.float32)
    RGB_a = np.zeros((ncell, 3), np.uint8)
    cdef int[::1] cnt = cnt_a
    cdef cnp.int64_t[::1] off = off_a, idx = idx_a
    cdef float[::1] Zo = Z_a
    cdef unsigned char[:, ::1] Co = RGB_a
    cdef double t, zmax, r0, g0, b0
    cdef cnp.int64_t ti
    with nogil:
        for i in range(n):
            cnt[cell[i]] += 1
        for c in range(ncell):
            off[c + 1] = off[c] + cnt[c]
            cnt[c] = 0
        for i in range(n):
            c = cell[i]
            idx[off[c] + cnt[c]] = i
            cnt[c] += 1
        for c in range(ncell):
            s0 = off[c]
            e0 = off[c + 1]
            if e0 == s0:
                continue
            for a in range(s0 + 1, e0):            # insertion sort of the bucket by z (ascending)
                ti = idx[a]
                t = z[ti]
                j = a - 1
                while j >= s0 and z[idx[j]] > t:
                    idx[j + 1] = idx[j]
                    j -= 1
                idx[j + 1] = ti
            zmax = z[idx[e0 - 1]]
            lo = e0 - 1
            while lo > s0 and z[idx[lo - 1]] >= zmax - gap:
                lo -= 1
            m = e0 - lo                            # top-layer size
            if m % 2:
                Zo[c] = <float>z[idx[lo + m // 2]]
            else:
                Zo[c] = <float>(0.5 * (z[idx[lo + m // 2 - 1]] + z[idx[lo + m // 2]]))
            r0 = 0; g0 = 0; b0 = 0
            for a in range(lo, e0):
                r0 += rgb[idx[a], 0]; g0 += rgb[idx[a], 1]; b0 += rgb[idx[a], 2]
            Co[c, 0] = <unsigned char>(r0 / m + 0.5)
            Co[c, 1] = <unsigned char>(g0 / m + 0.5)
            Co[c, 2] = <unsigned char>(b0 / m + 0.5)
    return Z_a, RGB_a, cnt_a


def fill_lower_median(const float[:, ::1] Zsrc, const unsigned char[:, :, ::1] Csrc, float[:, ::1] Z,
                      unsigned char[:, :, ::1] C, float[:, ::1] score, float code, int min_n):
    """One growth step: every empty cell with >= min_n measured 8-neighbours takes the *lower*
    median of their heights (never an average across a height step) and that neighbour's colour.
    Reads Zsrc/Csrc, writes Z/C/score. Returns the number of filled cells."""
    cdef Py_ssize_t H = Zsrc.shape[0], W = Zsrc.shape[1], y, x, dy, dx, a, c, nb
    cdef float vals[8]
    cdef int vy[8]
    cdef int vx[8]
    cdef float t
    cdef int ty, tx, filled = 0
    with nogil:
        for y in range(H):
            for x in range(W):
                if isfinite(Zsrc[y, x]):
                    continue
                nb = 0
                for dy in range(-1, 2):
                    for dx in range(-1, 2):
                        if (dy == 0 and dx == 0) or y + dy < 0 or y + dy >= H or x + dx < 0 or x + dx >= W:
                            continue
                        if isfinite(Zsrc[y + dy, x + dx]):
                            vals[nb] = Zsrc[y + dy, x + dx]
                            vy[nb] = <int>(y + dy)
                            vx[nb] = <int>(x + dx)
                            nb += 1
                if nb < min_n:
                    continue
                for a in range(1, nb):
                    t = vals[a]; ty = vy[a]; tx = vx[a]
                    c = a - 1
                    while c >= 0 and vals[c] > t:
                        vals[c + 1] = vals[c]; vy[c + 1] = vy[c]; vx[c + 1] = vx[c]
                        c -= 1
                    vals[c + 1] = t; vy[c + 1] = ty; vx[c + 1] = tx
                a = (nb - 1) // 2
                Z[y, x] = vals[a]
                C[y, x, 0] = Csrc[vy[a], vx[a], 0]
                C[y, x, 1] = Csrc[vy[a], vx[a], 1]
                C[y, x, 2] = Csrc[vy[a], vx[a], 2]
                score[y, x] = code
                filled += 1
    return filled


def nanmedian_filter(const float[:, ::1] Z, int r, float[:, ::1] out, Py_ssize_t y0, Py_ssize_t y1):
    """NaN-aware median over a (2r+1)^2 window for rows [y0, y1) (numpy nanmedian semantics:
    mean of the two middle values for an even count, NaN if the window is empty)."""
    cdef Py_ssize_t H = Z.shape[0], W = Z.shape[1], y, x, dy, dx, a, c, n
    cdef int m = (2 * r + 1) * (2 * r + 1)
    cdef float *buf = <float *>malloc(m * sizeof(float))
    cdef float t, v
    with nogil:
        for y in range(y0, y1):
            for x in range(W):
                n = 0
                for dy in range(-r, r + 1):
                    if y + dy < 0 or y + dy >= H:
                        continue
                    for dx in range(-r, r + 1):
                        if x + dx < 0 or x + dx >= W:
                            continue
                        v = Z[y + dy, x + dx]
                        if isfinite(v):
                            # insertion into the sorted buffer
                            c = n - 1
                            while c >= 0 and buf[c] > v:
                                buf[c + 1] = buf[c]
                                c -= 1
                            buf[c + 1] = v
                            n += 1
                if n == 0:
                    out[y, x] = NAN
                elif n % 2:
                    out[y, x] = buf[n // 2]
                else:
                    out[y, x] = 0.5 * (buf[n // 2 - 1] + buf[n // 2])
    free(buf)


# ---------------------------------------------------------------- orthophoto visibility
def visibility(const float[:, ::1] dsm, double minX, double maxY, double gsd, double Cx, double Cy, double Cz,
               const double[::1] X, const double[::1] Y, const double[::1] Z, int max_steps, double tol,
               double zmax, unsigned char[::1] vis):
    """Height-field line of sight from surface points to the camera centre (1 = visible): walk
    towards the camera over the stretch where the ray is below `zmax`, ~1 DSM cell per step
    (coarser only beyond `max_steps`), stop at the first blocking cell."""
    cdef Py_ssize_t n = X.shape[0], H = dsm.shape[0], W = dsm.shape[1], i, kstep, nsteps, col, row
    cdef double ux, uy, dh, rise, dmax, step, dist, zz
    with nogil:
        for i in range(n):
            vis[i] = 1
            ux = Cx - X[i]
            uy = Cy - Y[i]
            dh = hypot(ux, uy)
            if dh < 1e-6:
                continue
            rise = (Cz - Z[i]) / dh
            ux /= dh
            uy /= dh
            dmax = (zmax + tol - Z[i]) / (rise if rise > 1e-6 else 1e-6)
            if dmax > dh:
                dmax = dh
            if dmax <= 0:
                continue
            step = dmax / max_steps
            if step < gsd:
                step = gsd
            nsteps = <Py_ssize_t>(dmax / step)
            for kstep in range(1, nsteps + 1):
                dist = kstep * step
                col = <Py_ssize_t>floor((X[i] + ux * dist - minX) / gsd)
                row = <Py_ssize_t>floor((maxY - (Y[i] + uy * dist)) / gsd)
                if col < 0 or row < 0 or col >= W or row >= H:
                    break
                zz = dsm[row, col]
                if isfinite(zz) and zz > Z[i] + rise * dist + tol:
                    vis[i] = 0
                    break


# ---------------------------------------------------------------- orthophoto seams
def seam_icm(const float[:, :, ::1] cost, const float[:, :, :, ::1] cols, int[:, ::1] lab, double smooth, int iters):
    """ICM for the seam MRF (see ortho._seam_labels): synchronous updates, each pixel takes the
    label minimising data cost + smooth * sum over 4-neighbours with another label of
    (0.1 + mean |colour difference| / 64). Returns the number of iterations run."""
    cdef Py_ssize_t L = cost.shape[0], H = cost.shape[1], W = cost.shape[2], y, x, l, q, it, ny, nx
    cdef int dy[4]
    cdef int dx[4]
    dy[0] = -1; dx[0] = 0; dy[1] = 1; dx[1] = 0; dy[2] = 0; dx[2] = -1; dy[3] = 0; dx[3] = 1
    new_a = np.empty((H, W), np.int32)
    cdef int[:, ::1] new = new_a
    cdef double tot, best, diff
    cdef int bl, nl, changed, done = 0
    if L < 2 or smooth <= 0:
        return 0
    with nogil:
        for it in range(iters):
            changed = 0
            for y in range(H):
                for x in range(W):
                    best = 1e30
                    bl = lab[y, x]
                    for l in range(L):
                        tot = cost[l, y, x]
                        for q in range(4):
                            ny = y + dy[q]
                            nx = x + dx[q]
                            if ny < 0 or nx < 0 or ny >= H or nx >= W:
                                continue
                            nl = lab[ny, nx]
                            if nl == l:
                                continue
                            diff = (fabs(cols[l, y, x, 0] - cols[nl, y, x, 0]) + fabs(cols[l, y, x, 1] - cols[nl, y, x, 1])
                                    + fabs(cols[l, y, x, 2] - cols[nl, y, x, 2])) / 3.0
                            tot += smooth * (0.1 + diff / 64.0)
                        if tot < best:
                            best = tot
                            bl = <int>l
                    new[y, x] = bl
                    if bl != lab[y, x]:
                        changed += 1
            for y in range(H):
                for x in range(W):
                    lab[y, x] = new[y, x]
            done += 1
            if changed == 0:
                break
    return done


# ---------------------------------------------------------------- masks
def near(const unsigned char[:, ::1] mask, int k):
    """Cells within city-block distance `k` of a nonzero cell (= k rounds of 4-neighbour dilation),
    via a two-pass distance transform: O(H*W) regardless of k."""
    cdef Py_ssize_t H = mask.shape[0], W = mask.shape[1], y, x
    cdef int big = 1 << 30
    d_a = np.empty((H, W), np.int32)
    cdef int[:, ::1] d = d_a
    with nogil:
        for y in range(H):
            for x in range(W):
                if mask[y, x]:
                    d[y, x] = 0
                else:
                    d[y, x] = big
                    if y > 0 and d[y - 1, x] + 1 < d[y, x]:
                        d[y, x] = d[y - 1, x] + 1
                    if x > 0 and d[y, x - 1] + 1 < d[y, x]:
                        d[y, x] = d[y, x - 1] + 1
        for y in range(H - 1, -1, -1):
            for x in range(W - 1, -1, -1):
                if y < H - 1 and d[y + 1, x] + 1 < d[y, x]:
                    d[y, x] = d[y + 1, x] + 1
                if x < W - 1 and d[y, x + 1] + 1 < d[y, x]:
                    d[y, x] = d[y, x + 1] + 1
    return d_a <= k


# ---------------------------------------------------------------- PatchMatch repair
from libc.stdlib cimport malloc as _malloc, free as _free

cdef struct PMCtx:
    const float* ref
    int H
    int W
    const float* ray
    double cr0
    double cr1
    double cr2
    int S
    float** img
    float** dep
    int* iw
    int* ih
    double* par
    int r
    int step
    int top_k
    double geo_w
    double geo_tol


cdef inline int _proj(const double* p, double X, double Y, double Z, double* u, double* v, double* z) nogil:
    """par layout: R(9) C(3) f k1 k2 k3 cx cy (18 values per camera)."""
    cdef double dx = X - p[9], dy = Y - p[10], dz = Z - p[11]
    cdef double xc = p[0] * dx + p[1] * dy + p[2] * dz
    cdef double yc = p[3] * dx + p[4] * dy + p[5] * dz
    cdef double zc = p[6] * dx + p[7] * dy + p[8] * dz
    cdef double nx, ny, r2, dd
    if zc <= 1e-6:
        return 0
    nx = xc / zc
    ny = yc / zc
    r2 = nx * nx + ny * ny
    dd = p[12] * (1 + r2 * (p[13] + r2 * (p[14] + r2 * p[15])))
    u[0] = dd * nx + p[16]
    v[0] = dd * ny + p[17]
    z[0] = zc
    return 1


cdef inline double _bil(const float* im, int w, int h, double u, double v) nogil:
    cdef int x0, y0
    cdef double fx, fy
    if u < 0 or v < 0 or u > w - 1.001 or v > h - 1.001:
        return NAN
    x0 = <int>u
    y0 = <int>v
    fx = u - x0
    fy = v - y0
    return ((im[y0 * w + x0] * (1 - fx) + im[y0 * w + x0 + 1] * fx) * (1 - fy)
            + (im[(y0 + 1) * w + x0] * (1 - fx) + im[(y0 + 1) * w + x0 + 1] * fx) * fy)


cdef double _pm_cost(PMCtx* c, int y, int x, double d, const double* n, double* ncc_out) nogil:
    """1 - mean(top_k NCC over a slanted window) + geo_w * geometric disagreement."""
    cdef int H = c.H, W = c.W, HW = c.H * c.W, idx = y * c.W + x, s, a, b, q, yy, xx, nsamp = 0, kk
    cdef double rpx = c.ray[idx], rpy = c.ray[HW + idx], rpz = c.ray[2 * HW + idx]
    cdef double ndr = n[0] * rpx + n[1] * rpy + n[2] * rpz, num, den, t, X, Y, Z, u, v, z, I, J
    cdef double sI[16]
    cdef double sJ[16]
    cdef double sII[16]
    cdef double sJJ[16]
    cdef double sIJ[16]
    cdef int cnt[16]
    cdef double ncc[16]
    cdef double tmp, vi, vj, cv, photo, geo, e, dj
    cdef int ui, vv, ng
    if ndr >= -1e-6 or d <= 0:
        return 10.0
    num = d * ndr
    for s in range(c.S):
        sI[s] = 0; sJ[s] = 0; sII[s] = 0; sJJ[s] = 0; sIJ[s] = 0; cnt[s] = 0
    a = -c.r
    while a <= c.r:
        yy = y + a
        a += c.step
        if yy < 0 or yy >= H:
            continue
        b = -c.r
        while b <= c.r:
            xx = x + b
            b += c.step
            if xx < 0 or xx >= W:
                continue
            q = yy * W + xx
            I = c.ref[q]
            if not isfinite(I):
                continue
            den = n[0] * c.ray[q] + n[1] * c.ray[HW + q] + n[2] * c.ray[2 * HW + q]
            if den >= -1e-9:
                continue
            t = num / den
            X = c.cr0 + t * c.ray[q]
            Y = c.cr1 + t * c.ray[HW + q]
            Z = c.cr2 + t * c.ray[2 * HW + q]
            nsamp += 1
            for s in range(c.S):
                if _proj(c.par + 18 * s, X, Y, Z, &u, &v, &z):
                    J = _bil(c.img[s], c.iw[s], c.ih[s], u, v)
                    if isfinite(J):
                        sI[s] += I; sJ[s] += J; sII[s] += I * I; sJJ[s] += J * J; sIJ[s] += I * J; cnt[s] += 1
    for s in range(c.S):
        if nsamp < 4 or cnt[s] < 0.7 * nsamp:
            ncc[s] = -1
            continue
        vi = sII[s] / cnt[s] - (sI[s] / cnt[s]) ** 2
        vj = sJJ[s] / cnt[s] - (sJ[s] / cnt[s]) ** 2
        cv = sIJ[s] / cnt[s] - (sI[s] / cnt[s]) * (sJ[s] / cnt[s])
        if vi < 1e-8 or vj < 1e-8:
            ncc[s] = 0
        else:
            ncc[s] = cv / sqrt(vi * vj)
    for a in range(1, c.S):                       # descending sort (S <= 16)
        tmp = ncc[a]
        b = a - 1
        while b >= 0 and ncc[b] < tmp:
            ncc[b + 1] = ncc[b]
            b -= 1
        ncc[b + 1] = tmp
    kk = c.top_k if c.top_k < c.S else c.S
    photo = 0
    for a in range(kk):
        photo += ncc[a]
    photo /= kk
    ncc_out[0] = photo
    geo = 0
    if c.geo_w > 0:
        X = c.cr0 + d * rpx
        Y = c.cr1 + d * rpy
        Z = c.cr2 + d * rpz
        ng = 0
        for s in range(c.S):
            if c.dep[s] == NULL:
                continue
            ng += 1
            e = 1.0
            if _proj(c.par + 18 * s, X, Y, Z, &u, &v, &z):
                ui = <int>(u + 0.5)
                vv = <int>(v + 0.5)
                if ui >= 0 and vv >= 0 and ui < c.iw[s] and vv < c.ih[s]:
                    dj = c.dep[s][vv * c.iw[s] + ui]
                    if isfinite(dj) and dj > 0:
                        e = fabs(z - dj) / dj / c.geo_tol
                        if e > 1:
                            e = 1
            geo += e
        if ng:
            geo /= ng
    return (1 - photo) + c.geo_w * geo


cdef inline double _rnd(unsigned int* st) nogil:
    st[0] ^= st[0] << 13
    st[0] ^= st[0] >> 17
    st[0] ^= st[0] << 5
    return (st[0] & 0xFFFFFF) / 16777216.0


def pm_refine(const float[:, ::1] ref, const float[:, :, ::1] ray, const double[::1] Cr, list srcs, list src_depths,
              const double[:, ::1] par, float[:, ::1] depth, float[:, :, ::1] normal, float[:, ::1] score,
              const unsigned char[:, ::1] active, int r, int step, int top_k, double geo_w, double geo_tol,
              int iters, double dmin, double dmax, unsigned int seed):
    """PatchMatch repair of a depth map (COLMAP/OpenMVS-style, CPU): only `active` pixels are
    re-estimated; all pixels serve as propagation seeds. depth/normal/score are updated in place
    (normal: world frame, facing the camera; score: photometric top-k NCC of the kept plane).
    Returns the number of pixels whose hypothesis changed."""
    cdef int H = ref.shape[0], W = ref.shape[1], S = len(srcs), s, y, x, it, colour, k, yn, xn, tr
    cdef PMCtx c
    cdef const float[:, ::1] mv
    cdef double n[3]
    cdef double nb[3]
    cdef double best, cst, ncc, dn, dp, ndr, sc, nn
    cdef double dd_scale, nn_scale
    cdef unsigned int st
    cdef int changed = 0, HW = H * W, idx, nidx
    cdef int offy[8]
    cdef int offx[8]
    offy[0] = -1; offx[0] = 0; offy[1] = 1; offx[1] = 0; offy[2] = 0; offx[2] = -1; offy[3] = 0; offx[3] = 1
    offy[4] = -5; offx[4] = 0; offy[5] = 5; offx[5] = 0; offy[6] = 0; offx[6] = -5; offy[7] = 0; offx[7] = 5
    if S == 0 or S > 16:
        return 0
    c.ref = &ref[0, 0]; c.H = H; c.W = W; c.ray = &ray[0, 0, 0]
    c.cr0 = Cr[0]; c.cr1 = Cr[1]; c.cr2 = Cr[2]; c.S = S
    c.img = <float **>_malloc(S * sizeof(float *))
    c.dep = <float **>_malloc(S * sizeof(float *))
    c.iw = <int *>_malloc(S * sizeof(int))
    c.ih = <int *>_malloc(S * sizeof(int))
    for s in range(S):
        mv = srcs[s]
        c.img[s] = <float *>&mv[0, 0]
        c.ih[s] = mv.shape[0]
        c.iw[s] = mv.shape[1]
        if src_depths[s] is None:
            c.dep[s] = NULL
        else:
            mv = src_depths[s]
            c.dep[s] = <float *>&mv[0, 0]
    c.par = &par[0, 0]; c.r = r; c.step = step; c.top_k = top_k; c.geo_w = geo_w; c.geo_tol = geo_tol
    st = seed | 1
    with nogil:
        for it in range(iters):
            dd_scale = 0.05 / (it + 1)
            nn_scale = 0.3 / (it + 1)
            for colour in range(2):
                for y in range(H):
                    for x in range(W):
                        if not active[y, x] or (x + y) % 2 != colour:
                            continue
                        idx = y * W + x
                        n[0] = normal[0, y, x]; n[1] = normal[1, y, x]; n[2] = normal[2, y, x]
                        if isfinite(depth[y, x]):
                            best = _pm_cost(&c, y, x, depth[y, x], n, &ncc)
                            sc = ncc
                        else:
                            best = 1e9
                            sc = -1
                        # propagation: neighbours' planes extended to this pixel
                        for k in range(8):
                            yn = y + offy[k]
                            xn = x + offx[k]
                            if yn < 0 or xn < 0 or yn >= H or xn >= W or not isfinite(depth[yn, xn]):
                                continue
                            nidx = yn * W + xn
                            nb[0] = normal[0, yn, xn]; nb[1] = normal[1, yn, xn]; nb[2] = normal[2, yn, xn]
                            dn = nb[0] * c.ray[nidx] + nb[1] * c.ray[HW + nidx] + nb[2] * c.ray[2 * HW + nidx]
                            ndr = nb[0] * c.ray[idx] + nb[1] * c.ray[HW + idx] + nb[2] * c.ray[2 * HW + idx]
                            if ndr >= -1e-6:
                                continue
                            dp = depth[yn, xn] * dn / ndr
                            if dp < dmin or dp > dmax:
                                continue
                            cst = _pm_cost(&c, y, x, dp, nb, &ncc)
                            if cst < best:
                                best = cst
                                sc = ncc
                                depth[y, x] = <float>dp
                                normal[0, y, x] = <float>nb[0]; normal[1, y, x] = <float>nb[1]; normal[2, y, x] = <float>nb[2]
                                changed += 1
                        # random refinement of depth and tilt
                        if isfinite(depth[y, x]):
                            for tr in range(2):
                                dp = depth[y, x] * (1 + dd_scale * (2 * _rnd(&st) - 1))
                                nb[0] = normal[0, y, x] + nn_scale * (2 * _rnd(&st) - 1)
                                nb[1] = normal[1, y, x] + nn_scale * (2 * _rnd(&st) - 1)
                                nb[2] = normal[2, y, x] + nn_scale * (2 * _rnd(&st) - 1)
                                nn = sqrt(nb[0] * nb[0] + nb[1] * nb[1] + nb[2] * nb[2])
                                if nn < 1e-9 or dp < dmin or dp > dmax:
                                    continue
                                nb[0] /= nn; nb[1] /= nn; nb[2] /= nn
                                cst = _pm_cost(&c, y, x, dp, nb, &ncc)
                                if cst < best:
                                    best = cst
                                    sc = ncc
                                    depth[y, x] = <float>dp
                                    normal[0, y, x] = <float>nb[0]; normal[1, y, x] = <float>nb[1]; normal[2, y, x] = <float>nb[2]
                                    changed += 1
                        score[y, x] = <float>sc
    _free(c.img); _free(c.dep); _free(c.iw); _free(c.ih)
    return changed


# ---------------------------------------------------------------- orthophoto sampling (8-bit images)
def sample_view_u8(const unsigned char[:, :, ::1] img, const double[:, ::1] R, const double[::1] C,
                   double f, double k1, double k2, double k3, double cx, double cy,
                   double X0, double Y0, double gsd, const float[:, ::1] Z,
                   float[:, :, ::1] out, unsigned char[:, ::1] valid):
    """_mvs.sample_view for uint8 images (4x less cache memory than float32): grid cell (r, c) is
    the ground point (X0 + (c+.5) gsd, Y0 - (r+.5) gsd, Z[r, c]); writes the bilinear sample of
    `img` (h, w, ch) to out[r, c] (0-255 floats) and 1/0 to valid[r, c]."""
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


# ---------------------------------------------------------------- fused plane-sweep step
cdef extern from "_sweep.h" nogil:
    ctypedef struct om_sw_src:
        const float* img
        Py_ssize_t h
        Py_ssize_t w
        double r00, r01, r02, r10, r11, r12, r20, r21, r22
        double b0, b1, b2
        double f, k1, k2, k3, cx, cy
        float* Jrow
        float* Vrow
        float* ring
        double* cs
        float* nrow
    void om_sw_warp_row(const float* inv_row, const float* ray0, const float* ray1, const float* ray2,
                        om_sw_src* src, int n, Py_ssize_t W, double* U, double* Vv, double* D,
                        const double* rot, Py_ssize_t HW)
    void om_sw_rotate(const float* ray, const double* geo, Py_ssize_t HW, double* out)
    void om_sw_cs_addsub(double* cs, const float* ha, const float* hs, Py_ssize_t W)
    void om_sw_hsum_row(const float* ref_row, const om_sw_src* p, Py_ssize_t W, int r, float* hs)
    void om_sw_cs_add(double* cs, const float* hs, Py_ssize_t W)
    void om_sw_cs_sub(double* cs, const float* hs, Py_ssize_t W)
    void om_sw_ncc_row(const double* cs, const float* mu_row, const float* sd_row, double kk, Py_ssize_t W,
                       float* out)
    void om_sw_combine_row(const om_sw_src* src, int n, int kk, int i, Py_ssize_t W, float* best, float* prev,
                           float* s_prev_best, float* s_next_best, int* idx)


def sweep_rays(const float[:, :, ::1] ray, const double[:, ::1] geo):
    """Cache invariant source-frame directions, in the warp kernel's exact arithmetic."""
    if ray.shape[0] != 3 or geo.shape[1] != 18:
        raise ValueError("expected ray (3, H, W) and geo (n, 18)")
    cdef Py_ssize_t n = geo.shape[0], H = ray.shape[1], W = ray.shape[2], s
    result = np.empty((n, 3, H, W), np.float64)
    cdef double[:, :, :, ::1] out = result
    if H and W:
        with nogil:
            for s in range(n):
                om_sw_rotate(&ray[0, 0, 0], &geo[s, 0], H * W, &out[s, 0, 0, 0])
    return result


def sweep_hypothesis(const float[:, ::1] ref, const float[:, ::1] mu_i, const float[:, ::1] sd_i,
                     list srcs, const float[:, :, ::1] ray, const double[:, ::1] geo,
                     const float[:, ::1] inv, int r, int top_k, int i, float[:, ::1] best, float[:, ::1] prev,
                     float[:, ::1] s_prev_best, float[:, ::1] s_next_best, int[:, ::1] idx, rotated=None):
    """One hypothesis of the plane sweep for all sources at once: ``ncc_warp`` for every source
    followed by ``combine``, computed row by row (each row's warp, window sums and NCC stay in
    cache instead of five full-image passes per source). Per pixel the arithmetic, its order and
    the float32/float64 rounding points are exactly those of the two kernels (_sweep.h), so the
    result is bit-identical. geo: (n, 18) per source = Rs (9, row-major), b (3), f, k1, k2, k3, cx, cy."""
    cdef Py_ssize_t H = ref.shape[0], W = ref.shape[1], y, x, yy, k = 2 * r + 1, R2 = 2 * r + 2, HW
    cdef int n = len(srcs), s, kk
    cdef double kkd = 1.0 / (k * k)
    cdef const float[:, ::1] mv
    cdef om_sw_src* src
    cdef om_sw_src* p
    cdef float* hs
    cdef float* fblock = NULL
    cdef double* dblock = NULL
    cdef Py_ssize_t per_f
    cdef const double[:, :, :, ::1] rot
    cdef const double* rotp = NULL
    if n < 1 or n > 16:
        raise ValueError("sweep_hypothesis needs 1..16 sources")
    if geo.shape[0] != n or geo.shape[1] != 18:
        raise ValueError("geo must be (n_sources, 18)")
    if (mu_i.shape[0] != H or mu_i.shape[1] != W or sd_i.shape[0] != H or sd_i.shape[1] != W
            or inv.shape[0] != H or inv.shape[1] != W or ray.shape[0] != 3 or ray.shape[1] != H
            or ray.shape[2] != W or best.shape[0] != H or best.shape[1] != W or prev.shape[0] != H
            or prev.shape[1] != W or s_prev_best.shape[0] != H or s_prev_best.shape[1] != W
            or s_next_best.shape[0] != H or s_next_best.shape[1] != W or idx.shape[0] != H
            or idx.shape[1] != W):
        raise ValueError("sweep_hypothesis: array shapes do not match the reference image")
    if r < 0:
        raise ValueError("sweep_hypothesis: negative window radius")
    if H == 0 or W == 0:
        return
    if rotated is not None:
        rot = rotated
        if rot.shape[0] != n or rot.shape[1] != 3 or rot.shape[2] != H or rot.shape[3] != W:
            raise ValueError("rotated rays must be (n_sources, 3, H, W)")
        rotp = &rot[0, 0, 0, 0]
    HW = H * W
    per_f = 3 * W + R2 * 4 * W
    src = <om_sw_src*>malloc(n * sizeof(om_sw_src))
    fblock = <float*>malloc(n * per_f * sizeof(float))
    dblock = <double*>malloc((n * 4 * W + 3 * W) * sizeof(double))
    if src == NULL or fblock == NULL or dblock == NULL:
        free(src); free(fblock); free(dblock)
        raise MemoryError()
    for s in range(n):
        mv = srcs[s]
        p = &src[s]
        p.img = &mv[0, 0]
        p.h = mv.shape[0]
        p.w = mv.shape[1]
        p.r00 = geo[s, 0]; p.r01 = geo[s, 1]; p.r02 = geo[s, 2]
        p.r10 = geo[s, 3]; p.r11 = geo[s, 4]; p.r12 = geo[s, 5]
        p.r20 = geo[s, 6]; p.r21 = geo[s, 7]; p.r22 = geo[s, 8]
        p.b0 = geo[s, 9]; p.b1 = geo[s, 10]; p.b2 = geo[s, 11]
        p.f = geo[s, 12]; p.k1 = geo[s, 13]; p.k2 = geo[s, 14]; p.k3 = geo[s, 15]
        p.cx = geo[s, 16]; p.cy = geo[s, 17]
        p.Jrow = fblock + s * per_f
        p.Vrow = p.Jrow + W
        p.nrow = p.Vrow + W
        p.ring = p.nrow + W
        p.cs = dblock + s * 4 * W
    kk = top_k if top_k < n else n
    cdef const float* rayp = &ray[0, 0, 0]
    cdef double* wU = dblock + n * 4 * W
    cdef double* wV = wU + W
    cdef double* wD = wV + W
    with nogil:
        for s in range(n):
            for x in range(4 * W):
                src[s].cs[x] = 0
        # rows 0 .. r-1 enter the vertical sums first (ncc_warp phase 3 initialisation)
        for yy in range(min(r, H)):
            om_sw_warp_row(&inv[yy, 0], rayp + yy * W, rayp + HW + yy * W, rayp + 2 * HW + yy * W, src, n, W,
                           wU, wV, wD, rotp + yy * W if rotp != NULL else NULL, HW)
            for s in range(n):
                p = &src[s]
                hs = p.ring + (yy % R2) * 4 * W
                om_sw_hsum_row(&ref[yy, 0], p, W, r, hs)
                om_sw_cs_add(p.cs, hs, W)
        for y in range(H):
            if y + r < H:
                yy = y + r
                om_sw_warp_row(&inv[yy, 0], rayp + yy * W, rayp + HW + yy * W, rayp + 2 * HW + yy * W, src, n, W,
                               wU, wV, wD, rotp + yy * W if rotp != NULL else NULL, HW)
            for s in range(n):
                p = &src[s]
                if y + r < H:
                    hs = p.ring + ((y + r) % R2) * 4 * W
                    om_sw_hsum_row(&ref[y + r, 0], p, W, r, hs)
                    if y - r - 1 >= 0:
                        om_sw_cs_addsub(p.cs, hs, p.ring + ((y - r - 1) % R2) * 4 * W, W)
                    else:
                        om_sw_cs_add(p.cs, hs, W)
                elif y - r - 1 >= 0:
                    om_sw_cs_sub(p.cs, p.ring + ((y - r - 1) % R2) * 4 * W, W)
                om_sw_ncc_row(p.cs, &mu_i[y, 0], &sd_i[y, 0], kkd, W, p.nrow)
            om_sw_combine_row(src, n, kk, i, W, &best[y, 0], &prev[y, 0], &s_prev_best[y, 0], &s_next_best[y, 0],
                              &idx[y, 0])
    free(src)
    free(fblock)
    free(dblock)
