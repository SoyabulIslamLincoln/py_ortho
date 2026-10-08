# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
# cython: initializedcheck=False, nonecheck=False
"""Compiled replacements for hot NumPy / pure-Python loops, with bit-identical results.

Each kernel performs the same IEEE operations, in the same order and at the same precision, as
the NumPy expression it replaces (one rounding per NumPy ufunc call). The module is compiled with
floating-point contraction disabled (no fused multiply-add) for that reason; matrix products are
left to NumPy/BLAS by the callers. All loops release the GIL.
"""
import numpy as np
cimport numpy as cnp
from libc.math cimport isfinite, isnan, rint, hypot, fabs, sqrt, floor, NAN, signbit
from libc.stdlib cimport malloc, free
from libc.stdint cimport int32_t, int64_t, uint8_t

cnp.import_array()


# ---------------------------------------------------------------- SfM tracks (union-find)
def union_tracks(const int64_t[::1] na, const int64_t[::1] nb, const int32_t[::1] img_a,
                 const int32_t[::1] img_b, Py_ssize_t n_nodes, Py_ssize_t n_images):
    """sfm._build_tracks' consistency-aware union over the match edges, in edge order: a merge
    that would put two features of one image into one track is refused; the larger image set
    keeps its root (ties: the first node's root). Returns the parent array (int64, length
    n_nodes); roots are where parent[x] == x after `find`."""
    cdef Py_ssize_t E = na.shape[0], k, u, v, ru, rv, x, root, nxt, t, cap
    if nb.shape[0] != E or img_a.shape[0] != E or img_b.shape[0] != E:
        raise ValueError("union_tracks: edge arrays differ in length")
    parent_np = np.arange(n_nodes, dtype=np.int64)
    cdef int64_t[::1] parent = parent_np
    cap = 2 * E + 2
    # per root: linked list of its images in a pool (size 0 = implicit singleton, not in `members`)
    cdef int64_t* head = <int64_t*>malloc(max(n_nodes, 1) * sizeof(int64_t))
    cdef int64_t* tail = <int64_t*>malloc(max(n_nodes, 1) * sizeof(int64_t))
    cdef int64_t* size = <int64_t*>malloc(max(n_nodes, 1) * sizeof(int64_t))
    cdef int32_t* pimg = <int32_t*>malloc(cap * sizeof(int32_t))
    cdef int64_t* pnext = <int64_t*>malloc(cap * sizeof(int64_t))
    cdef int64_t* stamp = <int64_t*>malloc(max(n_images, 1) * sizeof(int64_t))
    cdef int64_t used = 0, gen = 0, su_n, sv_n, it
    cdef int32_t ia, ib
    cdef bint disjoint
    if head == NULL or tail == NULL or size == NULL or pimg == NULL or pnext == NULL or stamp == NULL:
        free(head); free(tail); free(size); free(pimg); free(pnext); free(stamp)
        raise MemoryError()
    with nogil:
        for x in range(n_nodes):
            size[x] = 0
        for x in range(n_images):
            stamp[x] = -1
        for k in range(E):
            u = na[k]
            v = nb[k]
            # find with path compression (roots are unaffected by compression)
            root = u
            while parent[root] != root:
                root = parent[root]
            while parent[u] != root:
                nxt = parent[u]
                parent[u] = root
                u = nxt
            ru = root
            root = v
            while parent[root] != root:
                root = parent[root]
            while parent[v] != root:
                nxt = parent[v]
                parent[v] = root
                v = nxt
            rv = root
            if ru == rv:
                continue
            ia = img_a[k]
            ib = img_b[k]
            # su = members.get(ru) or {ia};  sv = members.get(rv) or {ib}
            su_n = size[ru] if size[ru] > 0 else 1
            sv_n = size[rv] if size[rv] > 0 else 1
            gen += 1
            if size[ru] > 0:
                it = head[ru]
                while it >= 0:
                    stamp[pimg[it]] = gen
                    it = pnext[it]
            else:
                stamp[ia] = gen
            disjoint = True
            if size[rv] > 0:
                it = head[rv]
                while it >= 0:
                    if stamp[pimg[it]] == gen:
                        disjoint = False
                        break
                    it = pnext[it]
            elif stamp[ib] == gen:
                disjoint = False
            if not disjoint:
                continue
            if size[ru] == 0:                      # materialise the implicit singletons
                pimg[used] = ia
                pnext[used] = -1
                head[ru] = used
                tail[ru] = used
                size[ru] = 1
                used += 1
            if size[rv] == 0:
                pimg[used] = ib
                pnext[used] = -1
                head[rv] = used
                tail[rv] = used
                size[rv] = 1
                used += 1
            if su_n < sv_n:                        # the larger set keeps its root
                t = ru
                ru = rv
                rv = t
            parent[rv] = ru
            pnext[tail[ru]] = head[rv]             # su |= sv
            tail[ru] = tail[rv]
            size[ru] += size[rv]
            size[rv] = 0                           # members.pop(rv)
    free(head); free(tail); free(size); free(pimg); free(pnext); free(stamp)
    return parent_np


def find_roots(int64_t[::1] parent, const int64_t[::1] nodes):
    """Root of every node (with the same path compression as sfm._build_tracks' find)."""
    cdef Py_ssize_t n = nodes.shape[0], i
    cdef int64_t u, root, nxt
    out_np = np.empty(n, np.int64)
    cdef int64_t[::1] out = out_np
    with nogil:
        for i in range(n):
            u = nodes[i]
            root = u
            while parent[root] != root:
                root = parent[root]
            while parent[u] != root:
                nxt = parent[u]
                parent[u] = root
                u = nxt
            out[i] = root
    return out_np


# ---------------------------------------------------------------- camera model (densify._Cam)
cdef inline void _undistort(double u, double v, double cx, double cy, double f, double k1, double k2,
                            double k3, double* xo, double* yo) noexcept nogil:
    """_Cam.rays for one pixel: nx = (u - cx) / f, then 8 fixed-point iterations
    x = nx / (1 + r2 (k1 + r2 (k2 + r2 k3))), one rounding per NumPy operation."""
    cdef double nx = (u - cx) / f, ny = (v - cy) / f, x = nx, y = ny, r2, d, t
    cdef int it
    for it in range(8):
        r2 = x * x
        t = y * y
        r2 = r2 + t
        d = r2 * k3
        d = k2 + d
        d = r2 * d
        d = k1 + d
        d = r2 * d
        d = 1.0 + d
        x = nx / d
        y = ny / d
    xo[0] = x
    yo[0] = y


cdef inline void _project(double xc0, double xc1, double xc2, double f, double k1, double k2, double k3,
                          double cx, double cy, double* uo, double* vo) noexcept nogil:
    """_Cam.project_cam for one point (z = np.maximum(xc2, 1e-6), NaN propagating)."""
    cdef double z = xc2 if (xc2 >= 1e-6 or isnan(xc2)) else 1e-6
    cdef double x = xc0 / z, y = xc1 / z, r2, t, d
    r2 = x * x
    t = y * y
    r2 = r2 + t
    d = r2 * k3
    d = k2 + d
    d = r2 * d
    d = k1 + d
    d = r2 * d
    d = 1.0 + d
    d = f * d
    t = d * x
    uo[0] = t + cx
    t = d * y
    vo[0] = t + cy


def rays_grid(Py_ssize_t H, Py_ssize_t W, double cx, double cy, double f, double k1, double k2, double k3):
    """_Cam.rays over the whole pixel grid (u = column, v = row): (x, y) float64 (H, W)."""
    x_np = np.empty((H, W), np.float64)
    y_np = np.empty((H, W), np.float64)
    cdef double[:, ::1] xo = x_np, yo = y_np
    cdef Py_ssize_t r, c
    with nogil:
        for r in range(H):
            for c in range(W):
                _undistort(<double>c, <double>r, cx, cy, f, k1, k2, k3, &xo[r, c], &yo[r, c])
    return x_np, y_np


def ref_points(const float[:, ::1] D, const int64_t[::1] vv, const int64_t[::1] uu, double cx, double cy,
               double f, double k1, double k2, double k3):
    """Fusion / consistency, reference side: d = D[vv, uu] (float64), (x, y) = rays(uu, vv);
    returns the (n, 3) rows [x d, y d, d] (np.stack([x * d, y * d, d], 1))."""
    cdef Py_ssize_t n = vv.shape[0], i
    if uu.shape[0] != n:
        raise ValueError("ref_points: index arrays differ in length")
    M_np = np.empty((n, 3), np.float64)
    cdef double[:, ::1] M = M_np
    cdef double d, x, y
    with nogil:
        for i in range(n):
            d = D[vv[i], uu[i]]
            _undistort(<double>uu[i], <double>vv[i], cx, cy, f, k1, k2, k3, &x, &y)
            M[i, 0] = x * d
            M[i, 1] = y * d
            M[i, 2] = d
    return M_np


def neighbour_sample(const double[:, ::1] xc, const float[:, ::1] Dj, double f, double k1, double k2, double k3,
                     double cx, double cy, Py_ssize_t w, Py_ssize_t h):
    """Fusion / consistency, neighbour j: project xc (camera-j frame), round to the nearest pixel,
    read j's depth there (NaN outside the image or behind the camera) and back-project it.
    Returns ui, vi (int64), dj (float64) and the (n, 3) rows [xj dj, yj dj, dj]."""
    cdef Py_ssize_t n = xc.shape[0], i
    ui_np = np.empty(n, np.int64)
    vi_np = np.empty(n, np.int64)
    dj_np = np.empty(n, np.float64)
    M_np = np.empty((n, 3), np.float64)
    cdef int64_t[::1] ui = ui_np, vi = vi_np
    cdef double[::1] dj = dj_np
    cdef double[:, ::1] M = M_np
    cdef double u, v, x, y, dd
    cdef int64_t a, b
    with nogil:
        for i in range(n):
            _project(xc[i, 0], xc[i, 1], xc[i, 2], f, k1, k2, k3, cx, cy, &u, &v)
            a = <int64_t>rint(u)
            b = <int64_t>rint(v)
            ui[i] = a
            vi[i] = b
            if xc[i, 2] > 0 and a >= 0 and b >= 0 and a < w and b < h:
                dd = Dj[b, a]
            else:
                dd = NAN
            dj[i] = dd
            _undistort(<double>a, <double>b, cx, cy, f, k1, k2, k3, &x, &y)
            M[i, 0] = x * dd
            M[i, 1] = y * dd
            M[i, 2] = dd
    return ui_np, vi_np, dj_np, M_np


def agreement(const double[:, ::1] xk, const double[:, ::1] xc, const double[::1] dj, const int64_t[::1] uu,
              const int64_t[::1] vv, double f, double k1, double k2, double k3, double cx, double cy,
              double px_tol, double rel_tol):
    """Fusion / consistency check: reproject j's point into the reference (xk, reference camera
    frame) and accept when it lands within px_tol pixels of the source pixel and the depths agree:
    isfinite(dj) & (hypot(ub - uu, vb - vv) <= px_tol) & (|dj - xc[:, 2]| <= rel_tol * dj)."""
    cdef Py_ssize_t n = xk.shape[0], i
    ok_np = np.empty(n, np.bool_)
    cdef uint8_t[::1] ok = ok_np.view(np.uint8)
    cdef double ub, vb, t
    with nogil:
        for i in range(n):
            _project(xk[i, 0], xk[i, 1], xk[i, 2], f, k1, k2, k3, cx, cy, &ub, &vb)
            t = rel_tol * dj[i]
            ok[i] = (isfinite(dj[i]) and hypot(ub - <double>uu[i], vb - <double>vv[i]) <= px_tol
                     and fabs(dj[i] - xc[i, 2]) <= t)
    return ok_np


def accumulate(double[:, ::1] acc, int32_t[::1] cnt, const double[:, ::1] Xj, const uint8_t[::1] ok):
    """acc[ok] += Xj[ok]; cnt += ok."""
    cdef Py_ssize_t n = acc.shape[0], i
    with nogil:
        for i in range(n):
            if ok[i]:
                acc[i, 0] = acc[i, 0] + Xj[i, 0]
                acc[i, 1] = acc[i, 1] + Xj[i, 1]
                acc[i, 2] = acc[i, 2] + Xj[i, 2]
                cnt[i] = cnt[i] + 1


# ---------------------------------------------------------------- box filters
def box_edge_mean(const float[:, ::1] a, int r):
    """ortho._box / mvs._box_mean for r > 0: edge-padded (r + 1) float64 summed-area table
    (cumsum down the columns, then along the rows), window sum ((A - B) - C) + D, divided by
    (2r + 1)^2 and rounded to float32. Streams over rows (a ring of 2r + 2 table rows)."""
    cdef Py_ssize_t H = a.shape[0], W = a.shape[1], k = 2 * r + 1, PW, PH, i, j, y, x, R
    if r <= 0:
        raise ValueError("box_edge_mean needs r > 0")
    out_np = np.empty((H, W), np.float32)
    if H == 0 or W == 0:
        return out_np
    cdef float[:, ::1] out = out_np
    PW = W + 2 * (r + 1)
    PH = H + 2 * (r + 1)
    R = k + 1
    cdef double* col = <double*>malloc(PW * sizeof(double))          # running column sums (cumsum axis 0)
    cdef double* ring = <double*>malloc(R * PW * sizeof(double))     # rows of the full table
    cdef double* row
    cdef double* c0
    cdef double* c1
    cdef double s, kk = <double>(k * k)
    cdef Py_ssize_t ay, ax
    if col == NULL or ring == NULL:
        free(col); free(ring)
        raise MemoryError()
    with nogil:
        for i in range(PH):
            ay = i - (r + 1)
            if ay < 0:
                ay = 0
            elif ay >= H:
                ay = H - 1
            row = ring + (i % R) * PW
            for j in range(PW):
                ax = j - (r + 1)
                if ax < 0:
                    ax = 0
                elif ax >= W:
                    ax = W - 1
                if i == 0:
                    col[j] = <double>a[ay, ax]
                else:
                    col[j] = col[j] + <double>a[ay, ax]
            row[0] = col[0]
            for j in range(1, PW):
                row[j] = row[j - 1] + col[j]
            # output row y = i - k uses table rows y and y + k (= i)
            y = i - k
            if 0 <= y < H:
                c0 = ring + (y % R) * PW
                c1 = row
                for x in range(W):
                    s = c1[x + k] - c0[x + k]
                    s = s - c1[x]
                    s = s + c0[x]
                    out[y, x] = <float>(s / kk)
    free(col)
    free(ring)
    return out_np


# ---------------------------------------------------------------- true orthophoto, per view
cdef inline double _clip64(double x, double lo, double hi) noexcept nogil:
    """np.clip for float64 (NaN and -0.0 pass through as NumPy returns them)."""
    if x < lo:
        return lo
    if x > hi:
        return hi
    return x


cdef inline float _clip32(float x, float lo, float hi) noexcept nogil:
    if x < lo:
        return lo
    if x > hi:
        return hi
    return x


cdef inline float _finite0(float v) noexcept nogil:
    """np.nan_to_num(v, nan=0, posinf=0, neginf=0)."""
    return v if isfinite(v) else <float>0.0


def ortho_view_cos(const unsigned char[:, ::1] sub, const float[:, ::1] dsm, Py_ssize_t py0, Py_ssize_t px0,
                   double X0, double Y0, double gsd, const float[:, :, ::1] nrm, double C0, double C1, double C2):
    """true_orthophoto per view, for the covered cells of the tile in row-major order (rr, cc =
    np.nonzero(sub)): wx = X0 + (cc + .5) gsd, wy = Y0 - (rr + .5) gsd, wz = dsm[py0 + rr, px0 + cc];
    ray = C - (wx, wy, wz); cos_n = clip(sum(ray * n) / (|ray| + 1e-9), 0, 1)."""
    cdef Py_ssize_t ph = sub.shape[0], pw = sub.shape[1], r, c, i = 0, n = 0
    with nogil:
        for r in range(ph):
            for c in range(pw):
                if sub[r, c]:
                    n += 1
    out_np = np.empty(n, np.float64)
    cdef double[::1] out = out_np
    cdef double wx, wy, wz, a0, a1, a2, rng, t
    with nogil:
        for r in range(ph):
            for c in range(pw):
                if not sub[r, c]:
                    continue
                t = <double>c + 0.5
                t = t * gsd
                wx = X0 + t
                t = <double>r + 0.5
                t = t * gsd
                wy = Y0 - t
                wz = <double>dsm[py0 + r, px0 + c]
                a0 = C0 - wx
                a1 = C1 - wy
                a2 = C2 - wz
                rng = a0 * a0
                t = a1 * a1
                rng = rng + t
                t = a2 * a2
                rng = rng + t
                rng = sqrt(rng) + 1e-9
                t = a0 * <double>nrm[r, c, 0]
                wx = a1 * <double>nrm[r, c, 1]
                t = t + wx
                wx = a2 * <double>nrm[r, c, 2]
                t = t + wx
                out[i] = _clip64(t / rng, 0.0, 1.0)
                i += 1
    return out_np


def ortho_view_fill(const unsigned char[:, ::1] sub, const float[:, ::1] dsm, Py_ssize_t py0, Py_ssize_t px0,
                    double X0, double Y0, double gsd, const float[:, :, ::1] rgbw, const unsigned char[:, ::1] vv,
                    const unsigned char[:, ::1] vis_map, int stride, bint use_vis, double C0, double C1, double C2,
                    double R20, double R21, double R22, const double[::1] cpow, double w_ang, double w_bord,
                    double w_exp, float[:, :, ::1] cols, float[:, ::1] score, float[:, ::1] score_nv,
                    float[::1] inv_depth):
    """The rest of true_orthophoto's per-view loop body for the covered cells (row-major): sample
    validity, visibility (vis_map at stride), clipped-exposure flag, camera-axis depth, and the
    score w_ang cos^p + w_bord clip(feather) + w_exp unclipped, written to this view's cols,
    score (-1 = unusable), score_nv (ignoring occlusion) and inv_depth. cpow = cos_n ** p."""
    cdef Py_ssize_t ph = sub.shape[0], pw = sub.shape[1], r, c, i = 0
    cdef float p0, p1, p2, a3, lum, f32b = <float>w_bord, t32
    cdef double wx, wy, wz, dep, s, e, t
    cdef bint valid, ok
    with nogil:
        for r in range(ph):
            for c in range(pw):
                if not sub[r, c]:
                    continue
                p0 = _finite0(rgbw[r, c, 0])
                p1 = _finite0(rgbw[r, c, 1])
                p2 = _finite0(rgbw[r, c, 2])
                a3 = _finite0(rgbw[r, c, 3])
                valid = vv[r, c] > 0 and a3 > 0
                ok = valid and (vis_map[r // stride, c // stride] != 0 if use_vis else True)
                lum = p0 + p1
                lum = lum + p2
                lum = lum / <float>3
                e = 0.0 if (lum > <float>250 or lum < <float>5) else 1.0
                t = <double>c + 0.5
                t = t * gsd
                wx = X0 + t
                t = <double>r + 0.5
                t = t * gsd
                wy = Y0 - t
                wz = <double>dsm[py0 + r, px0 + c]
                dep = (wx - C0) * R20
                t = (wy - C1) * R21
                dep = dep + t
                t = (wz - C2) * R22
                dep = dep + t
                if ok:
                    t = dep if (dep >= 1e-3 or isnan(dep)) else 1e-3
                    inv_depth[i] = <float>(1.0 / t)
                else:
                    inv_depth[i] = 0
                s = w_ang * cpow[i]
                t32 = _clip32(a3, <float>0, <float>1)
                t32 = f32b * t32
                s = s + <double>t32
                t = w_exp * e
                s = s + t
                cols[r, c, 0] = p0
                cols[r, c, 1] = p1
                cols[r, c, 2] = p2
                score[r, c] = <float>s if ok else <float>-1.0
                score_nv[r, c] = <float>s if valid else <float>-1.0
                i += 1


def ortho_resolution(const unsigned char[:, ::1] sub, const float[:, ::1] inv_depth, float[:, :, ::1] score,
                     double w_res):
    """Resolution term: best = inv_depth.max(0); res = inv_depth / max(best, 1e-9) where best > 0
    (else 0); usable scores (>= 0) get + w_res * res, the others become -1 (float32 throughout)."""
    cdef Py_ssize_t L = inv_depth.shape[0], n = inv_depth.shape[1], ph = sub.shape[0], pw = sub.shape[1]
    cdef Py_ssize_t r, c, i = 0, j
    cdef float best, v, den, res, fw = <float>w_res, sc
    with nogil:
        for r in range(ph):
            for c in range(pw):
                if not sub[r, c]:
                    continue
                best = inv_depth[0, i]
                for j in range(1, L):
                    v = inv_depth[j, i]
                    if v > best or isnan(v):
                        best = v
                den = best if (best >= <float>1e-9 or isnan(best)) else <float>1e-9
                for j in range(L):
                    res = inv_depth[j, i] / den if best > 0 else <float>0
                    sc = score[j, r, c]
                    if sc >= 0:
                        res = fw * res
                        score[j, r, c] = sc + res
                    else:
                        score[j, r, c] = -1.0
                i += 1


def ortho_rgbw(const unsigned char[:, :, ::1] rgb, gain, bias, mask, bint u8):
    """true_orthophoto's per-image cache entry in one pass: float32 colour (x gain, + bias),
    4th channel = feather weight min(dist to the border) / (0.5 min(h, w)) clipped to [0, 1]
    (0 on masked pixels); returned as float32 (h, w, 4), or, when u8, as
    clip(rgbw * (1, 1, 1, 255) + 0.5, 0, 255) cast to uint8 - the NumPy steps' exact rounding."""
    cdef Py_ssize_t h = rgb.shape[0], w = rgb.shape[1], y, x, c
    if rgb.shape[2] != 3:
        raise ValueError("ortho_rgbw expects (h, w, 3) RGB")
    cdef float g[3]
    cdef float b[3]
    cdef bint has_g = gain is not None, has_b = bias is not None, has_m = mask is not None
    cdef const unsigned char[:, ::1] mv
    if has_g:
        ga = np.asarray(gain, np.float32).reshape(-1)
        for c in range(3):
            g[c] = ga[c]
    if has_b:
        ba = np.asarray(bias, np.float32).reshape(-1)
        for c in range(3):
            b[c] = ba[c]
    if has_m:
        mv = np.ascontiguousarray(mask, np.bool_).view(np.uint8)
        if mv.shape[0] != h or mv.shape[1] != w:
            raise ValueError("mask shape differs from the image")
    fy_np = np.empty(h, np.float64)
    fx_np = np.empty(w, np.float64)
    cdef double[::1] fy = fy_np, fx = fx_np
    cdef double den = 0.5 * <double>(h if h < w else w), a1, a2, mf
    for y in range(h):                          # np.minimum(arange + .5, n - .5 - arange) / den
        a1 = <double>y + 0.5
        a2 = (<double>h - 0.5) - <double>y
        fy[y] = (a1 if (a1 <= a2 or isnan(a1)) else a2) / den
    for x in range(w):
        a1 = <double>x + 0.5
        a2 = (<double>w - 0.5) - <double>x
        fx[x] = (a1 if (a1 <= a2 or isnan(a1)) else a2) / den
    cdef float v, fe
    cdef float[:, :, ::1] of
    cdef unsigned char[:, :, ::1] ou
    if u8:
        out_np = np.empty((h, w, 4), np.uint8)
        ou = out_np
    else:
        out_np = np.empty((h, w, 4), np.float32)
        of = out_np
    with nogil:
        for y in range(h):
            for x in range(w):
                mf = fy[y] if (fy[y] <= fx[x] or isnan(fy[y])) else fx[x]
                fe = <float>_clip64(mf, 0.0, 1.0)
                if has_m and not mv[y, x]:
                    fe = 0.0
                for c in range(3):
                    v = <float>rgb[y, x, c]
                    if has_g:
                        v = v * g[c]
                    if has_b:
                        v = v + b[c]
                    if u8:
                        v = v * <float>1.0
                        v = v + <float>0.5
                        ou[y, x, c] = <unsigned char>_clip32(v, <float>0, <float>255)
                    else:
                        of[y, x, c] = v
                if u8:
                    v = fe * <float>255.0
                    v = v + <float>0.5
                    ou[y, x, 3] = <unsigned char>_clip32(v, <float>0, <float>255)
                else:
                    of[y, x, 3] = fe
    return out_np


def depth_view(const unsigned char[:, :, ::1] rgb, gain, bias, mask):
    """densify's matching image in one pass: rgb as float32 (x gain, + bias); grey =
    ((mean of the channels) - 128) / 64 as float32, NaN where `mask` is False; colour =
    clip(rgb + 0.5, 0, 255) as uint8. Returns (grey (h, w) float32, rgb8 (h, w, 3) uint8)."""
    cdef Py_ssize_t h = rgb.shape[0], w = rgb.shape[1], y, x, c
    if rgb.shape[2] != 3:
        raise ValueError("depth_view expects (h, w, 3) RGB")
    cdef float g[3]
    cdef float b[3]
    cdef float v[3]
    cdef bint has_g = gain is not None, has_b = bias is not None, has_m = mask is not None
    cdef const unsigned char[:, ::1] mv
    if has_g:
        ga = np.asarray(gain, np.float32).reshape(-1)
        for c in range(3):
            g[c] = ga[c]
    if has_b:
        ba = np.asarray(bias, np.float32).reshape(-1)
        for c in range(3):
            b[c] = ba[c]
    if has_m:
        mv = np.ascontiguousarray(mask, np.bool_).view(np.uint8)
        if mv.shape[0] != h or mv.shape[1] != w:
            raise ValueError("mask shape differs from the image")
    grey_np = np.empty((h, w), np.float32)
    rgb8_np = np.empty((h, w, 3), np.uint8)
    cdef float[:, ::1] grey = grey_np
    cdef unsigned char[:, :, ::1] rgb8 = rgb8_np
    cdef float s
    with nogil:
        for y in range(h):
            for x in range(w):
                for c in range(3):
                    v[c] = <float>rgb[y, x, c]
                    if has_g:
                        v[c] = v[c] * g[c]
                    if has_b:
                        v[c] = v[c] + b[c]
                s = v[0] + v[1]
                s = s + v[2]
                s = s / <float>3
                s = s - <float>128.0
                s = s / <float>64.0
                grey[y, x] = NAN if (has_m and not mv[y, x]) else s
                for c in range(3):
                    s = v[c] + <float>0.5
                    rgb8[y, x, c] = <unsigned char>_clip32(s, <float>0, <float>255)
    return grey_np, rgb8_np


# ---------------------------------------------------------------- depth-map helpers
def hyp_linear(const float[:, ::1] base, const float[:, ::1] step, long i):
    """(base + i * step).astype(float32) with NumPy's float32 rounding (i converted to float32)."""
    cdef Py_ssize_t H = base.shape[0], W = base.shape[1], y, x
    if step.shape[0] != H or step.shape[1] != W:
        raise ValueError("hyp_linear: shapes differ")
    out_np = np.empty((H, W), np.float32)
    cdef float[:, ::1] out = out_np
    cdef float fi = <float>i, t
    with nogil:
        for y in range(H):
            for x in range(W):
                t = fi * step[y, x]
                out[y, x] = base[y, x] + t
    return out_np


def hyp_refine(const float[:, ::1] iu, const float[:, ::1] ds, long t):
    """np.maximum(iu + t * ds, 1e-6) in float32 (NaN propagates as in np.maximum)."""
    cdef Py_ssize_t H = iu.shape[0], W = iu.shape[1], y, x
    if ds.shape[0] != H or ds.shape[1] != W:
        raise ValueError("hyp_refine: shapes differ")
    out_np = np.empty((H, W), np.float32)
    cdef float[:, ::1] out = out_np
    cdef float ft = <float>t, v, lo = <float>1e-6
    with nogil:
        for y in range(H):
            for x in range(W):
                v = ft * ds[y, x]
                v = iu[y, x] + v
                out[y, x] = v if (v >= lo or isnan(v)) else lo
    return out_np


def box_zero_mean(const float[:, ::1] a, int r):
    """densify._box: zero-padded window mean via float32 cumulative sums (down the columns,
    then along the rows), times float32(1 / (2r+1)^2)."""
    cdef Py_ssize_t H = a.shape[0], W = a.shape[1], k = 2 * r + 1, y, x, PW
    if r < 0:
        raise ValueError("box_zero_mean needs r >= 0")
    out_np = np.empty((H, W), np.float32)
    if H == 0 or W == 0:
        return out_np
    cdef float[:, ::1] out = out_np
    PW = W + 2 * r + 1
    cdef float* col = <float*>malloc(W * sizeof(float))          # running column sums (padded rows)
    cdef float* ring = <float*>malloc((k + 1) * W * sizeof(float))  # last k + 1 table rows
    cdef float* rowc = <float*>malloc(PW * sizeof(float))
    cdef float scale = <float>(1.0 / (k * k)), t, prevc
    cdef float* cur
    cdef float* old
    cdef Py_ssize_t i, R = k + 1, PH = H + 2 * r + 1, ay
    if col == NULL or ring == NULL or rowc == NULL:
        free(col); free(ring); free(rowc)
        raise MemoryError()
    with nogil:
        for x in range(W):
            col[x] = 0
        for i in range(PH):                     # table row i = cumsum of padded rows 0..i
            ay = i - (r + 1)
            cur = ring + (i % R) * W
            for x in range(W):
                if i == 0:
                    col[x] = <float>0 if (ay < 0 or ay >= H) else a[ay, x]
                else:
                    col[x] = col[x] + (<float>0 if (ay < 0 or ay >= H) else a[ay, x])
                cur[x] = col[x]
            y = i - k                            # output row y: a1 = c[y + k] - c[y]
            if y >= 0:
                old = ring + (y % R) * W
                # second pass along the row: zero-pad (r+1 left, r right), cumsum, window difference
                for x in range(PW):
                    if x < r + 1 or x >= r + 1 + W:
                        t = 0
                    else:
                        t = cur[x - r - 1] - old[x - r - 1]
                    if x == 0:
                        rowc[x] = t
                    else:
                        rowc[x] = rowc[x - 1] + t
                for x in range(W):
                    t = rowc[x + k] - rowc[x]
                    out[y, x] = t * scale
    free(col)
    free(ring)
    free(rowc)
    return out_np


# ---------------------------------------------------------------- nearest keypoints (no SciPy)
def knn_radius(const double[:, ::1] pts, const double[:, ::1] q, int k, double radius):
    """For every query point the k nearest of `pts` closer than `radius` (strictly), nearest
    first (ties: lower index first), in the format of scipy's cKDTree.query(q, k,
    distance_upper_bound=radius): distances (m, k) float64 (inf where missing) and indices
    (m, k) int64 (len(pts) where missing). Grid-bucketed, exact."""
    cdef Py_ssize_t n = pts.shape[0], m = q.shape[0], i, j, a, b, c, gx, gy, cx, cy, s
    d_np = np.full((m, k), np.inf, np.float64)
    i_np = np.full((m, k), n, np.int64)
    if n == 0 or m == 0 or k <= 0 or not (radius > 0):
        return d_np, i_np
    cdef double[:, ::1] dout = d_np
    cdef int64_t[:, ::1] iout = i_np
    cdef double x0 = np.min(np.asarray(pts)[:, 0]), y0 = np.min(np.asarray(pts)[:, 1])
    cdef double x1 = np.max(np.asarray(pts)[:, 0]), y1 = np.max(np.asarray(pts)[:, 1])
    cdef double cell = radius
    gx = <Py_ssize_t>((x1 - x0) / cell) + 1
    gy = <Py_ssize_t>((y1 - y0) / cell) + 1
    if gx * gy > 4 * n + 1024:                 # sparse points: coarser grid, same results
        cell = max(radius, sqrt((x1 - x0) * (y1 - y0) / <double>max(n, 1)))
        gx = <Py_ssize_t>((x1 - x0) / cell) + 1
        gy = <Py_ssize_t>((y1 - y0) / cell) + 1
    start_np = np.zeros(gx * gy + 1, np.int64)
    order_np = np.empty(n, np.int64)
    cdef int64_t[::1] start = start_np, order = order_np
    cdef int64_t* fill = <int64_t*>malloc((gx * gy + 1) * sizeof(int64_t))
    cdef double* bd = <double*>malloc(k * sizeof(double))
    cdef int64_t* bi = <int64_t*>malloc(k * sizeof(int64_t))
    if fill == NULL or bd == NULL or bi == NULL:
        free(fill); free(bd); free(bi)
        raise MemoryError()
    cdef double dx, dy, d2, r2 = radius * radius, qx, qy
    cdef Py_ssize_t nb, a2, reach = <Py_ssize_t>(radius / cell) + 1
    with nogil:
        for i in range(n):                     # counting sort of the points into grid cells
            c = <Py_ssize_t>((pts[i, 1] - y0) / cell) * gx + <Py_ssize_t>((pts[i, 0] - x0) / cell)
            start[c + 1] += 1
        for c in range(gx * gy):
            start[c + 1] += start[c]
            fill[c] = start[c]
        for i in range(n):                     # ascending index inside each cell
            c = <Py_ssize_t>((pts[i, 1] - y0) / cell) * gx + <Py_ssize_t>((pts[i, 0] - x0) / cell)
            order[fill[c]] = i
            fill[c] += 1
        for j in range(m):
            qx = q[j, 0]
            qy = q[j, 1]
            nb = 0
            if not (qx == qx and qy == qy):
                continue
            cx = <Py_ssize_t>floor((qx - x0) / cell)
            cy = <Py_ssize_t>floor((qy - y0) / cell)
            for b in range(cy - reach, cy + reach + 1):
                if b < 0 or b >= gy:
                    continue
                for a in range(cx - reach, cx + reach + 1):
                    if a < 0 or a >= gx:
                        continue
                    c = b * gx + a
                    for s in range(start[c], start[c + 1]):
                        i = order[s]
                        dx = pts[i, 0] - qx
                        dy = pts[i, 1] - qy
                        d2 = dx * dx + dy * dy
                        if not (d2 < r2):
                            continue
                        # insert into the sorted best-k list (distance, then index)
                        if nb == k and (d2 > bd[k - 1] or (d2 == bd[k - 1] and i > bi[k - 1])):
                            continue
                        a2 = nb if nb < k else k - 1
                        while a2 > 0 and (bd[a2 - 1] > d2 or (bd[a2 - 1] == d2 and bi[a2 - 1] > i)):
                            if a2 < k:
                                bd[a2] = bd[a2 - 1]
                                bi[a2] = bi[a2 - 1]
                            a2 -= 1
                        bd[a2] = d2
                        bi[a2] = i
                        if nb < k:
                            nb += 1
            for a in range(nb):
                dout[j, a] = sqrt(bd[a])
                iout[j, a] = bi[a]
    free(fill); free(bd); free(bi)
    return d_np, i_np


# ---------------------------------------------------------------- terrain downsampling
cdef float _select_float(float* a, Py_ssize_t n, Py_ssize_t rank) noexcept nogil:
    """In-place Hoare selection; no floating-point arithmetic on the samples."""
    cdef Py_ssize_t lo = 0, hi = n - 1, i, j
    cdef float pivot, tmp
    while lo < hi:
        pivot = a[lo + (hi - lo) // 2]
        i, j = lo, hi
        while i <= j:
            while a[i] < pivot:
                i += 1
            while a[j] > pivot:
                j -= 1
            if i <= j:
                tmp = a[i]
                a[i] = a[j]
                a[j] = tmp
                i += 1
                j -= 1
        if rank <= j:
            hi = j
        elif rank >= i:
            lo = i
        else:
            break
    return a[rank]


def block_nanpercentile(const float[:, ::1] z, Py_ssize_t factor, double q=25.0):
    """Block nanpercentile (linear), with NumPy's float32 difference / float64 lerp.

    Only one block is buffered. Return None for non-quartile percentiles,
    infinities or signed zero so the caller retains NumPy's version-specific
    scalar promotion, ordering and NaN payload behavior.
    """
    cdef Py_ssize_t H = z.shape[0], W = z.shape[1], h, w, y, x, yy, xx, n, rank
    cdef float v, a, b, diff
    cdef double pos, fraction, value, t, quantile = q / 100.0
    cdef bint unsupported = False
    if factor < 1 or not (0 <= q <= 100):
        raise ValueError("factor must be positive and q must be in [0, 100]")
    if q not in (0.0, 25.0, 50.0, 75.0, 100.0):
        return None
    h = (H + factor - 1) // factor
    w = (W + factor - 1) // factor
    result = np.empty((h, w), np.float32)
    cdef float[:, ::1] out = result
    # Partial edge blocks never need a factor*factor allocation larger than z.
    scratch = np.empty(min(factor, H) * min(factor, W), np.float32)
    cdef float[::1] buf = scratch
    with nogil:
        for y in range(h):
            for x in range(w):
                n = 0
                for yy in range(y * factor, min((y + 1) * factor, H)):
                    for xx in range(x * factor, min((x + 1) * factor, W)):
                        v = z[yy, xx]
                        if isnan(v):
                            continue
                        if not isfinite(v) or (v == 0 and signbit(v)):
                            unsupported = True
                        buf[n] = v
                        n += 1
                if unsupported:
                    break
                if n == 0:
                    out[y, x] = NAN
                    continue
                if n > 16777216:  # NumPy 2 may round the sample count to float32
                    unsupported = True
                    break
                pos = (n - 1) * quantile
                rank = <Py_ssize_t>floor(pos)
                fraction = pos - rank
                a = _select_float(&buf[0], n, rank)
                b = _select_float(&buf[0], n, min(rank + 1, n - 1))
                diff = b - a
                if fraction >= 0.5:
                    t = 1.0 - fraction
                    t = <double>diff * t
                    value = <double>b - t
                else:
                    t = <double>diff * fraction
                    value = <double>a + t
                out[y, x] = <float>value
            if unsupported:
                break
    return None if unsupported else result


ctypedef fused box_scalar:
    float
    double


def box_sat_mean(const box_scalar[:, :, ::1] a, Py_ssize_t r):
    """Float64 summed-area mean, preserving the two cumsums and four-corner order.

    Leading dimensions are flattened by the caller. The integral table is a ring
    of rows: scratch scales with window height, not the entire raster height.
    """
    cdef Py_ssize_t B = a.shape[0], H = a.shape[1], W = a.shape[2], b, i, x, y, ay
    cdef Py_ssize_t k = 2 * r + 1, PW = W + 2 * r + 1, PH = H + 2 * r + 1, R = k + 1
    cdef double val, total, denom = <double>k * <double>k
    cdef double* cur
    cdef double* old
    if r < 0:
        raise ValueError("radius must be nonnegative")
    result = np.empty((B, H, W), np.float32)
    cdef float[:, :, ::1] out = result
    if B == 0 or H == 0 or W == 0:
        return result
    col_np = np.empty(W, np.float64)
    ring_np = np.empty((R, PW), np.float64)
    cdef double[::1] col = col_np
    cdef double[:, ::1] ring = ring_np
    with nogil:
        for b in range(B):
            for x in range(W):
                col[x] = 0.0
            for i in range(PH):
                ay = i - r - 1
                cur = &ring[i % R, 0]
                total = 0.0
                for x in range(r + 1):
                    cur[x] = 0.0
                for x in range(W):
                    val = <double>a[b, ay, x] if 0 <= ay < H else 0.0
                    col[x] = col[x] + val
                    total = total + col[x]
                    cur[x + r + 1] = total
                for x in range(r + 1 + W, PW):
                    cur[x] = total
                y = i - k
                if y >= 0:
                    old = &ring[y % R, 0]
                    for x in range(W):
                        val = cur[x + k] - old[x + k]
                        val = val - cur[x]
                        val = val + old[x]
                        out[b, y, x] = <float>(val / denom)
    return result
