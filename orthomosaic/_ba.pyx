# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
# cython: initializedcheck=False, nonecheck=False
"""
Bundle adjustment kernels (Levenberg-Marquardt with Schur complement).

Parameter layout of the reduced ("camera side") system, size nc = 6*N + 6*G:
    camera i : [w0 w1 w2 | c0 c1 c2]  at 6*i      (left-multiplied rotation update, centre)
    group  g : [f k1 k2 k3 cx cy]      at 6*N + 6*g  (Pix4D/ODM-style self-calibration: focal,
                                                    three radial terms, principal point)
intr is (G, 6) in that order; the `pp` arguments are kept for API compatibility and ignored.
Points (3 each) are eliminated per track. Observations must be sorted by point,
with pt_ptr giving the CSR row pointers.
"""
import numpy as np
cimport numpy as cnp
from libc.math cimport sqrt, fabs
from libc.stdlib cimport malloc, free
from libc.stdint cimport int32_t, int64_t

cnp.import_array()


cdef inline bint _linearize_obs(const double[:, :, ::1] R, const double[:, ::1] C,
                                const double[:, ::1] X, const double[:, ::1] intr,
                                const double[:, ::1] pp, const int32_t[::1] cam_group,
                                int cam, Py_ssize_t pt, double uo, double vo, double huber,
                                double* Jc, double* Jp, double* e, double* cost) noexcept nogil:
    """Jc: 2x12 (w, C, f k1 k2 k3 cx cy), Jp: 2x3, e: 2 residual, pre-scaled by sqrt(Huber weight)."""
    cdef int g = cam_group[cam]
    cdef double f = intr[g, 0], k1 = intr[g, 1], k2 = intr[g, 2], k3 = intr[g, 3]
    cdef double dX0 = X[pt, 0] - C[cam, 0], dX1 = X[pt, 1] - C[cam, 1], dX2 = X[pt, 2] - C[cam, 2]
    cdef double x = R[cam, 0, 0] * dX0 + R[cam, 0, 1] * dX1 + R[cam, 0, 2] * dX2
    cdef double y = R[cam, 1, 0] * dX0 + R[cam, 1, 1] * dX1 + R[cam, 1, 2] * dX2
    cdef double z = R[cam, 2, 0] * dX0 + R[cam, 2, 1] * dX1 + R[cam, 2, 2] * dX2
    if z <= 1e-6:
        return False
    cdef double iz = 1.0 / z, nx = x * iz, ny = y * iz
    cdef double r2 = nx * nx + ny * ny
    cdef double d = 1.0 + r2 * (k1 + r2 * (k2 + r2 * k3))
    cdef double dd = k1 + r2 * (2.0 * k2 + 3.0 * k3 * r2)
    cdef double ex = f * d * nx + intr[g, 4] - uo
    cdef double ey = f * d * ny + intr[g, 5] - vo
    cdef double s = sqrt(ex * ex + ey * ey), w = 1.0
    if s <= huber:
        cost[0] += s * s
    else:
        cost[0] += 2.0 * huber * s - huber * huber
        w = huber / s
    cdef double sw = sqrt(w)
    e[0] = ex * sw
    e[1] = ey * sw
    # d(uv)/d(n)
    cdef double a00 = f * (d + 2.0 * nx * nx * dd), a01 = f * 2.0 * nx * ny * dd
    cdef double a10 = a01, a11 = f * (d + 2.0 * ny * ny * dd)
    # d(uv)/d(xc) = A @ [[iz, 0, -x iz^2], [0, iz, -y iz^2]]
    cdef double J[2][3]
    J[0][0] = a00 * iz
    J[0][1] = a01 * iz
    J[0][2] = -(a00 * nx + a01 * ny) * iz
    J[1][0] = a10 * iz
    J[1][1] = a11 * iz
    J[1][2] = -(a10 * nx + a11 * ny) * iz
    cdef int r, k
    for r in range(2):
        # rotation: J @ (-[xc]_x) with -[xc]_x = [[0, z, -y], [-z, 0, x], [y, -x, 0]]
        Jc[r * 12 + 0] = sw * (-J[r][1] * z + J[r][2] * y)
        Jc[r * 12 + 1] = sw * (J[r][0] * z - J[r][2] * x)
        Jc[r * 12 + 2] = sw * (-J[r][0] * y + J[r][1] * x)
        for k in range(3):
            # centre: J @ (-R) ; point: J @ R
            Jp[r * 3 + k] = sw * (J[r][0] * R[cam, 0, k] + J[r][1] * R[cam, 1, k] + J[r][2] * R[cam, 2, k])
            Jc[r * 12 + 3 + k] = -Jp[r * 3 + k]
    Jc[6] = sw * d * nx
    Jc[18] = sw * d * ny
    Jc[7] = sw * f * r2 * nx
    Jc[19] = sw * f * r2 * ny
    Jc[8] = sw * f * r2 * r2 * nx
    Jc[20] = sw * f * r2 * r2 * ny
    Jc[9] = sw * f * r2 * r2 * r2 * nx
    Jc[21] = sw * f * r2 * r2 * r2 * ny
    Jc[10] = sw
    Jc[22] = 0.0
    Jc[11] = 0.0
    Jc[23] = sw
    return True


cdef inline void _inv3(const double* A, double* B) noexcept nogil:
    cdef double det = (A[0] * (A[4] * A[8] - A[5] * A[7]) - A[1] * (A[3] * A[8] - A[5] * A[6])
                       + A[2] * (A[3] * A[7] - A[4] * A[6]))
    if fabs(det) < 1e-300:
        det = 1e-300
    cdef double inv = 1.0 / det
    B[0] = (A[4] * A[8] - A[5] * A[7]) * inv
    B[1] = (A[2] * A[7] - A[1] * A[8]) * inv
    B[2] = (A[1] * A[5] - A[2] * A[4]) * inv
    B[3] = (A[5] * A[6] - A[3] * A[8]) * inv
    B[4] = (A[0] * A[8] - A[2] * A[6]) * inv
    B[5] = (A[2] * A[3] - A[0] * A[5]) * inv
    B[6] = (A[3] * A[7] - A[4] * A[6]) * inv
    B[7] = (A[1] * A[6] - A[0] * A[7]) * inv
    B[8] = (A[0] * A[4] - A[1] * A[3]) * inv


def reduced_system(const double[:, :, ::1] R, const double[:, ::1] C, const double[:, ::1] X,
                   const double[:, ::1] intr, const double[:, ::1] pp, const int32_t[::1] cam_group,
                   const int32_t[::1] obs_cam, const double[:, ::1] obs_uv, const int64_t[::1] pt_ptr,
                   double huber, double lam,
                   const double[:, ::1] pw, const double[:, ::1] pt_tgt, int workers=1):
    """Build the damped Schur-reduced normal equations.

    pw, pt_tgt: per-point position-prior weight (1/sigma^2) and target (P, 3). Zero weight = no
    prior (ordinary tie point). Ground control points enter this way, as points whose world
    position is pulled toward the surveyed coordinate.

    Returns (S (nc,nc), g (nc), Vinv (P,3,3), gp (P,3), cost, diagU (nc)); S is already
    damped with lam * diag(U).
    Solve S dc = -g, then dp = back_substitute(...).
    workers > 1 spreads the work over threads with bit-identical results (_reduced_system_mt).
    """
    if workers > 1 and obs_cam.shape[0] >= 4096:
        return _reduced_system_mt(R, C, X, intr, pp, cam_group, obs_cam, obs_uv, pt_ptr, huber, lam, pw, pt_tgt,
                                  workers)
    cdef Py_ssize_t N = R.shape[0], G = intr.shape[0], P = X.shape[0]
    cdef Py_ssize_t nc = 6 * N + 6 * G
    S_np = np.zeros((nc, nc), np.float64)
    g_np = np.zeros(nc, np.float64)
    diagU_np = np.zeros(nc, np.float64)
    Vinv_np = np.zeros((P, 3, 3), np.float64)
    gp_np = np.zeros((P, 3), np.float64)
    cdef double[:, ::1] S = S_np
    cdef double[::1] g = g_np, diagU = diagU_np
    cdef double[:, :, ::1] Vinv = Vinv_np
    cdef double[:, ::1] gp = gp_np
    cdef double cost = 0.0
    cdef Py_ssize_t maxlen = 0, p, k, a, b, q, m, ka, kb
    for p in range(P):
        if pt_ptr[p + 1] - pt_ptr[p] > maxlen:
            maxlen = pt_ptr[p + 1] - pt_ptr[p]
    cdef double* Wbuf = <double*>malloc(max(maxlen, 1) * 36 * sizeof(double))   # W_k = Jc^T Jp (12x3)
    cdef int* Ibuf = <int*>malloc(max(maxlen, 1) * 12 * sizeof(int))            # global indices
    cdef char* ok = <char*>malloc(max(maxlen, 1) * sizeof(char))
    cdef double Jc[24]
    cdef double Jp[6]
    cdef double e[2]
    cdef double V[9]
    cdef double Vd[9]
    cdef double Vi[9]
    cdef double gpt[3]
    cdef double WV[36]
    cdef double t
    cdef int cam, grp
    if Wbuf == NULL or Ibuf == NULL or ok == NULL:
        free(Wbuf); free(Ibuf); free(ok)
        raise MemoryError()
    try:
        with nogil:
            for p in range(P):
                for a in range(9):
                    V[a] = 0.0
                gpt[0] = 0.0
                gpt[1] = 0.0
                gpt[2] = 0.0
                m = pt_ptr[p + 1] - pt_ptr[p]
                for k in range(m):
                    q = pt_ptr[p] + k
                    cam = obs_cam[q]
                    grp = cam_group[cam]
                    ok[k] = _linearize_obs(R, C, X, intr, pp, cam_group, cam, p,
                                           obs_uv[q, 0], obs_uv[q, 1], huber, Jc, Jp, e, &cost)
                    if not ok[k]:
                        continue
                    for a in range(6):
                        Ibuf[k * 12 + a] = 6 * cam + a
                    for a in range(6):
                        Ibuf[k * 12 + 6 + a] = 6 * N + 6 * grp + a
                    # U block, gradient
                    for a in range(12):
                        ka = Ibuf[k * 12 + a]
                        g[ka] += Jc[a] * e[0] + Jc[12 + a] * e[1]
                        diagU[ka] += Jc[a] * Jc[a] + Jc[12 + a] * Jc[12 + a]
                        for b in range(12):
                            S[ka, Ibuf[k * 12 + b]] += Jc[a] * Jc[b] + Jc[12 + a] * Jc[12 + b]
                        # W = Jc^T Jp
                        for b in range(3):
                            Wbuf[k * 36 + a * 3 + b] = Jc[a] * Jp[b] + Jc[12 + a] * Jp[3 + b]
                    for a in range(3):
                        gpt[a] += Jp[a] * e[0] + Jp[3 + a] * e[1]
                        for b in range(3):
                            V[a * 3 + b] += Jp[a] * Jp[b] + Jp[3 + a] * Jp[3 + b]
                # position prior (ground control): adds w*(X-target) to the point normal equations
                for a in range(3):
                    if pw[p, a] > 0.0:
                        gpt[a] += pw[p, a] * (X[p, a] - pt_tgt[p, a])
                        V[a * 3 + a] += pw[p, a]
                        cost += pw[p, a] * (X[p, a] - pt_tgt[p, a]) * (X[p, a] - pt_tgt[p, a])
                for a in range(9):
                    Vd[a] = V[a]
                for a in range(3):
                    Vd[a * 4] = V[a * 4] * (1.0 + lam) + 1e-9
                _inv3(Vd, Vi)
                for a in range(9):
                    Vinv[p, a // 3, a % 3] = Vi[a]
                for a in range(3):
                    gp[p, a] = gpt[a]
                # Schur: S -= W_k Vi W_l^T ; g -= W_k Vi gp
                for k in range(m):
                    if not ok[k]:
                        continue
                    for a in range(12):
                        for b in range(3):
                            WV[a * 3 + b] = (Wbuf[k * 36 + a * 3 + 0] * Vi[0 * 3 + b]
                                             + Wbuf[k * 36 + a * 3 + 1] * Vi[1 * 3 + b]
                                             + Wbuf[k * 36 + a * 3 + 2] * Vi[2 * 3 + b])
                    for a in range(12):
                        ka = Ibuf[k * 12 + a]
                        g[ka] -= WV[a * 3 + 0] * gpt[0] + WV[a * 3 + 1] * gpt[1] + WV[a * 3 + 2] * gpt[2]
                    for q in range(m):
                        if not ok[q]:
                            continue
                        for a in range(12):
                            ka = Ibuf[k * 12 + a]
                            for b in range(12):
                                kb = Ibuf[q * 12 + b]
                                t = (WV[a * 3 + 0] * Wbuf[q * 36 + b * 3 + 0]
                                     + WV[a * 3 + 1] * Wbuf[q * 36 + b * 3 + 1]
                                     + WV[a * 3 + 2] * Wbuf[q * 36 + b * 3 + 2])
                                S[ka, kb] -= t
    finally:
        free(Wbuf)
        free(Ibuf)
        free(ok)
    # Levenberg-Marquardt damping on the camera side (diagonal of U only)
    cdef Py_ssize_t i
    for i in range(nc):
        S[i, i] += lam * diagU[i] + 1e-12
    return S_np, g_np, Vinv_np, gp_np, cost, diagU_np


def back_substitute(const double[:, :, ::1] R, const double[:, ::1] C, const double[:, ::1] X,
                    const double[:, ::1] intr, const double[:, ::1] pp, const int32_t[::1] cam_group,
                    const int32_t[::1] obs_cam, const double[:, ::1] obs_uv, const int64_t[::1] pt_ptr,
                    double huber, const double[:, :, ::1] Vinv, const double[:, ::1] gp,
                    const double[::1] dc, int workers=1):
    """dp_p = Vinv_p (-gp_p - sum_k W_k^T dc). workers > 1: same result, points split over threads."""
    if workers > 1 and obs_cam.shape[0] >= 4096:
        return _back_substitute_mt(R, C, X, intr, pp, cam_group, obs_cam, obs_uv, pt_ptr, huber, Vinv, gp, dc,
                                   workers)
    cdef Py_ssize_t N = R.shape[0], P = X.shape[0], p, q
    dp_np = np.zeros((P, 3), np.float64)
    cdef double[:, ::1] dp = dp_np
    cdef double Jc[24]
    cdef double Jp[6]
    cdef double e[2]
    cdef double rhs[3]
    cdef double cost = 0.0, jd0, jd1
    cdef int a, b, cam, grp
    with nogil:
        for p in range(P):
            rhs[0] = -gp[p, 0]
            rhs[1] = -gp[p, 1]
            rhs[2] = -gp[p, 2]
            for q in range(pt_ptr[p], pt_ptr[p + 1]):
                cam = obs_cam[q]
                grp = cam_group[cam]
                if not _linearize_obs(R, C, X, intr, pp, cam_group, cam, p, obs_uv[q, 0], obs_uv[q, 1],
                                      huber, Jc, Jp, e, &cost):
                    continue
                # Jc @ dc (2-vector), then W^T dc = Jp^T (Jc dc)
                jd0 = 0.0
                jd1 = 0.0
                for a in range(6):
                    jd0 += Jc[a] * dc[6 * cam + a]
                    jd1 += Jc[12 + a] * dc[6 * cam + a]
                for a in range(6):
                    jd0 += Jc[6 + a] * dc[6 * N + 6 * grp + a]
                    jd1 += Jc[18 + a] * dc[6 * N + 6 * grp + a]
                for b in range(3):
                    rhs[b] -= Jp[b] * jd0 + Jp[3 + b] * jd1
            for a in range(3):
                dp[p, a] = Vinv[p, a, 0] * rhs[0] + Vinv[p, a, 1] * rhs[1] + Vinv[p, a, 2] * rhs[2]
    return dp_np


def residuals(const double[:, :, ::1] R, const double[:, ::1] C, const double[:, ::1] X,
              const double[:, ::1] intr, const double[:, ::1] pp, const int32_t[::1] cam_group,
              const int32_t[::1] obs_cam, const int32_t[::1] obs_pt, const double[:, ::1] obs_uv,
              double huber, int workers=1):
    """Per-observation reprojection error (px; inf if behind the camera) and total robust cost.
    workers > 1: same result (errors in parallel, the cost summed in observation order)."""
    if workers > 1 and obs_cam.shape[0] >= 4096:
        return _residuals_mt(R, C, X, intr, pp, cam_group, obs_cam, obs_pt, obs_uv, huber, workers)
    cdef Py_ssize_t M = obs_cam.shape[0], q
    out_np = np.empty(M, np.float64)
    cdef double[::1] out = out_np
    cdef double cost = 0.0, x, y, z, nx, ny, r2, dd, ex, ey, s, f, k1, k2, k3, d0, d1, d2
    cdef int cam, g, pt
    with nogil:
        for q in range(M):
            cam = obs_cam[q]
            pt = obs_pt[q]
            g = cam_group[cam]
            f = intr[g, 0]
            k1 = intr[g, 1]
            k2 = intr[g, 2]
            k3 = intr[g, 3]
            d0 = X[pt, 0] - C[cam, 0]
            d1 = X[pt, 1] - C[cam, 1]
            d2 = X[pt, 2] - C[cam, 2]
            x = R[cam, 0, 0] * d0 + R[cam, 0, 1] * d1 + R[cam, 0, 2] * d2
            y = R[cam, 1, 0] * d0 + R[cam, 1, 1] * d1 + R[cam, 1, 2] * d2
            z = R[cam, 2, 0] * d0 + R[cam, 2, 1] * d1 + R[cam, 2, 2] * d2
            if z <= 1e-6:
                out[q] = 1e30
                cost += 2.0 * huber * 1e6
                continue
            nx = x / z
            ny = y / z
            r2 = nx * nx + ny * ny
            dd = 1.0 + r2 * (k1 + r2 * (k2 + r2 * k3))
            ex = f * dd * nx + intr[g, 4] - obs_uv[q, 0]
            ey = f * dd * ny + intr[g, 5] - obs_uv[q, 1]
            s = sqrt(ex * ex + ey * ey)
            out[q] = s
            if s <= huber:
                cost += s * s
            else:
                cost += 2.0 * huber * s - huber * huber
    return out_np, cost


# ------------------------------------------------------------------ multi-threaded kernels
# Same arithmetic as the serial kernels above, distributed over threads so that the results are
# bit-identical for any thread count:
#   * every per-point quantity (Jacobians, V^-1, W, W V^-1) is computed by exactly one thread with
#     the serial code's expressions;
#   * every entry of S, g and diag(U) is owned by exactly one thread, which adds that entry's
#     contributions in the serial order (points ascending, then observations, then the Schur pairs);
#   * the scalar cost is re-accumulated sequentially in observation order.
# Ownership: S[ka, kb] with kb a camera column -> owner of that camera; kb an intrinsics column
# and ka a camera row -> owner of the row's camera; intrinsics x intrinsics entries and the
# intrinsics rows of g / diag(U) are spread over the threads by their index within the 6x6 block.

cdef inline bint _linearize_obs_s(const double[:, :, ::1] R, const double[:, ::1] C,
                                  const double[:, ::1] X, const double[:, ::1] intr,
                                  const double[:, ::1] pp, const int32_t[::1] cam_group,
                                  int cam, Py_ssize_t pt, double uo, double vo, double huber,
                                  double* Jc, double* Jp, double* e, double* s_out) noexcept nogil:
    """_linearize_obs that reports the residual norm instead of adding to the cost."""
    cdef int g = cam_group[cam]
    cdef double f = intr[g, 0], k1 = intr[g, 1], k2 = intr[g, 2], k3 = intr[g, 3]
    cdef double dX0 = X[pt, 0] - C[cam, 0], dX1 = X[pt, 1] - C[cam, 1], dX2 = X[pt, 2] - C[cam, 2]
    cdef double x = R[cam, 0, 0] * dX0 + R[cam, 0, 1] * dX1 + R[cam, 0, 2] * dX2
    cdef double y = R[cam, 1, 0] * dX0 + R[cam, 1, 1] * dX1 + R[cam, 1, 2] * dX2
    cdef double z = R[cam, 2, 0] * dX0 + R[cam, 2, 1] * dX1 + R[cam, 2, 2] * dX2
    if z <= 1e-6:
        return False
    cdef double iz = 1.0 / z, nx = x * iz, ny = y * iz
    cdef double r2 = nx * nx + ny * ny
    cdef double d = 1.0 + r2 * (k1 + r2 * (k2 + r2 * k3))
    cdef double dd = k1 + r2 * (2.0 * k2 + 3.0 * k3 * r2)
    cdef double ex = f * d * nx + intr[g, 4] - uo
    cdef double ey = f * d * ny + intr[g, 5] - vo
    cdef double s = sqrt(ex * ex + ey * ey), w = 1.0
    s_out[0] = s
    if s > huber:
        w = huber / s
    cdef double sw = sqrt(w)
    e[0] = ex * sw
    e[1] = ey * sw
    cdef double a00 = f * (d + 2.0 * nx * nx * dd), a01 = f * 2.0 * nx * ny * dd
    cdef double a10 = a01, a11 = f * (d + 2.0 * ny * ny * dd)
    cdef double J[2][3]
    J[0][0] = a00 * iz
    J[0][1] = a01 * iz
    J[0][2] = -(a00 * nx + a01 * ny) * iz
    J[1][0] = a10 * iz
    J[1][1] = a11 * iz
    J[1][2] = -(a10 * nx + a11 * ny) * iz
    cdef int r, k
    for r in range(2):
        Jc[r * 12 + 0] = sw * (-J[r][1] * z + J[r][2] * y)
        Jc[r * 12 + 1] = sw * (J[r][0] * z - J[r][2] * x)
        Jc[r * 12 + 2] = sw * (-J[r][0] * y + J[r][1] * x)
        for k in range(3):
            Jp[r * 3 + k] = sw * (J[r][0] * R[cam, 0, k] + J[r][1] * R[cam, 1, k] + J[r][2] * R[cam, 2, k])
            Jc[r * 12 + 3 + k] = -Jp[r * 3 + k]
    Jc[6] = sw * d * nx
    Jc[18] = sw * d * ny
    Jc[7] = sw * f * r2 * nx
    Jc[19] = sw * f * r2 * ny
    Jc[8] = sw * f * r2 * r2 * nx
    Jc[20] = sw * f * r2 * r2 * ny
    Jc[9] = sw * f * r2 * r2 * r2 * nx
    Jc[21] = sw * f * r2 * r2 * r2 * ny
    Jc[10] = sw
    Jc[22] = 0.0
    Jc[11] = 0.0
    Jc[23] = sw
    return True


cdef extern from "_ba_kern.h" nogil:
    void om_ba_u_cam(double* S, Py_ssize_t nc, double* g, double* diagU, int ck, Py_ssize_t gk, const double* Jc,
                     const double* e)
    void om_ba_schur_qcols(double* S, Py_ssize_t nc, int ck, Py_ssize_t gk, int cq, const double* WV, const double* Wq)
    void om_ba_schur_kgrp(double* S, Py_ssize_t nc, int ck, Py_ssize_t gq, const double* WV, const double* Wq)


cdef int _rs_owner(const double[:, :, ::1] R, const double[:, ::1] C, const double[:, ::1] X,
                   const double[:, ::1] intr, const double[:, ::1] pp, const int32_t[::1] cam_group,
                   const int32_t[::1] obs_cam, const double[:, ::1] obs_uv, const int64_t[::1] pt_ptr,
                   double huber, double lam, const double[:, ::1] pw, const double[:, ::1] pt_tgt,
                   double[:, :, ::1] Vinv, double[:, ::1] gp, double[:, ::1] S, double[::1] g, double[::1] diagU,
                   const int32_t[::1] cam_owner, Py_ssize_t maxlen, int T, int t, double* cost_out) noexcept nogil:
    """One owner thread of the multi-threaded reduced_system: walks every point in order with the
    serial kernel's arithmetic (so per-point quantities are recomputed, not shared) and adds only
    the S / g / diag(U) entries it owns. Thread 0 also writes Vinv, gp and the cost. The owned
    intrinsics x intrinsics entries and intrinsics rows of g / diag(U) (a few cache lines every
    thread would otherwise write for every observation) are accumulated in a private copy and
    stored back at the end: the same additions in the same order. Returns -1 on allocation
    failure."""
    cdef Py_ssize_t N = R.shape[0], G = intr.shape[0], P = X.shape[0], n6 = 6 * G, base = 6 * N
    cdef Py_ssize_t p, k, q, m, a, b, ka, kb, gk, gq, i
    cdef double* Wbuf = <double*>malloc(max(maxlen, 1) * 36 * sizeof(double))
    cdef double* WVb = <double*>malloc(max(maxlen, 1) * 36 * sizeof(double))
    cdef double* Jcb = <double*>malloc(max(maxlen, 1) * 24 * sizeof(double))
    cdef double* eb = <double*>malloc(max(maxlen, 1) * 2 * sizeof(double))
    cdef int* camb = <int*>malloc(max(maxlen, 1) * sizeof(int))
    cdef char* ok = <char*>malloc(max(maxlen, 1) * sizeof(char))
    cdef char* mine = <char*>malloc(max(maxlen, 1) * sizeof(char))
    cdef double* Sgg = <double*>malloc((n6 * n6 + 2 * n6) * sizeof(double))
    cdef double* gg
    cdef double* dg
    cdef double Jp[6]
    cdef double V[9]
    cdef double Vd[9]
    cdef double Vi[9]
    cdef double gpt[3]
    cdef double* Jc
    cdef double* e
    cdef double* Wk
    cdef double* WV
    cdef double* Wq
    cdef double s, tt, cost = 0.0
    cdef int cam, ck, cq, n_gg = 0, n_gr = 0, any_mine
    cdef int gg_a[36]
    cdef int gg_b[36]
    cdef int gr_a[6]
    cdef double* Sp = &S[0, 0]
    cdef double* gptr = &g[0]
    cdef double* dptr = &diagU[0]
    cdef Py_ssize_t nc = S.shape[1]
    if (Wbuf == NULL or WVb == NULL or Jcb == NULL or eb == NULL or camb == NULL or ok == NULL or mine == NULL
            or Sgg == NULL):
        free(Wbuf); free(WVb); free(Jcb); free(eb); free(camb); free(ok); free(mine); free(Sgg)
        return -1
    gg = Sgg + n6 * n6
    dg = gg + n6
    for a in range(n6):
        gg[a] = 0.0
        dg[a] = 0.0
        for b in range(n6):
            Sgg[a * n6 + b] = 0.0
    for a in range(6, 12):                     # intrinsics entries owned by t
        if (a - 6) % T == t:
            gr_a[n_gr] = <int>a
            n_gr += 1
        for b in range(6, 12):
            if ((a - 6) * 6 + b - 6) % T == t:
                gg_a[n_gg] = <int>a
                gg_b[n_gg] = <int>b
                n_gg += 1
    for p in range(P):
        m = pt_ptr[p + 1] - pt_ptr[p]
        any_mine = t == 0 or n_gr > 0 or n_gg > 0
        for k in range(m):
            mine[k] = cam_owner[obs_cam[pt_ptr[p] + k]] == t
            if mine[k]:
                any_mine = 1
        if not any_mine:
            continue
        for a in range(9):
            V[a] = 0.0
        gpt[0] = 0.0
        gpt[1] = 0.0
        gpt[2] = 0.0
        for k in range(m):
            q = pt_ptr[p] + k
            cam = obs_cam[q]
            camb[k] = cam
            Jc = Jcb + k * 24
            e = eb + k * 2
            ok[k] = _linearize_obs_s(R, C, X, intr, pp, cam_group, cam, p, obs_uv[q, 0], obs_uv[q, 1], huber,
                                     Jc, Jp, e, &s)
            if not ok[k]:
                continue
            if t == 0:
                if s <= huber:
                    cost += s * s
                else:
                    cost += 2.0 * huber * s - huber * huber
            Wk = Wbuf + k * 36
            for a in range(12):
                for b in range(3):
                    Wk[a * 3 + b] = Jc[a] * Jp[b] + Jc[12 + a] * Jp[3 + b]
            for a in range(3):
                gpt[a] += Jp[a] * e[0] + Jp[3 + a] * e[1]
                for b in range(3):
                    V[a * 3 + b] += Jp[a] * Jp[b] + Jp[3 + a] * Jp[3 + b]
        for a in range(3):
            if pw[p, a] > 0.0:
                gpt[a] += pw[p, a] * (X[p, a] - pt_tgt[p, a])
                V[a * 3 + a] += pw[p, a]
                if t == 0:
                    cost += pw[p, a] * (X[p, a] - pt_tgt[p, a]) * (X[p, a] - pt_tgt[p, a])
        for a in range(9):
            Vd[a] = V[a]
        for a in range(3):
            Vd[a * 4] = V[a * 4] * (1.0 + lam) + 1e-9
        _inv3(Vd, Vi)
        if t == 0:
            for a in range(9):
                Vinv[p, a // 3, a % 3] = Vi[a]
            for a in range(3):
                gp[p, a] = gpt[a]
        # U block and gradient, observation by observation
        for k in range(m):
            if not ok[k]:
                continue
            ck = camb[k]
            gk = base + 6 * cam_group[ck]
            Jc = Jcb + k * 24
            e = eb + k * 2
            if mine[k]:
                om_ba_u_cam(Sp, nc, gptr, dptr, ck, gk, Jc, e)
            for i in range(n_gr):
                a = gr_a[i]
                ka = gk + a - 6 - base
                gg[ka] += Jc[a] * e[0] + Jc[12 + a] * e[1]
                dg[ka] += Jc[a] * Jc[a] + Jc[12 + a] * Jc[12 + a]
            for i in range(n_gg):
                a = gg_a[i]
                b = gg_b[i]
                Sgg[(gk + a - 6 - base) * n6 + gk + b - 6 - base] += Jc[a] * Jc[b] + Jc[12 + a] * Jc[12 + b]
        # Schur complement: S -= W_k Vi W_l^T ; g -= W_k Vi gp
        for k in range(m):
            if not ok[k]:
                continue
            Wk = Wbuf + k * 36
            WV = WVb + k * 36
            for a in range(12):
                for b in range(3):
                    WV[a * 3 + b] = (Wk[a * 3 + 0] * Vi[0 * 3 + b]
                                     + Wk[a * 3 + 1] * Vi[1 * 3 + b]
                                     + Wk[a * 3 + 2] * Vi[2 * 3 + b])
        for k in range(m):
            if not ok[k]:
                continue
            ck = camb[k]
            gk = base + 6 * cam_group[ck]
            WV = WVb + k * 36
            if mine[k]:
                for a in range(6):
                    ka = 6 * ck + a
                    g[ka] -= WV[a * 3 + 0] * gpt[0] + WV[a * 3 + 1] * gpt[1] + WV[a * 3 + 2] * gpt[2]
            for i in range(n_gr):
                a = gr_a[i]
                ka = gk + a - 6 - base
                gg[ka] -= WV[a * 3 + 0] * gpt[0] + WV[a * 3 + 1] * gpt[1] + WV[a * 3 + 2] * gpt[2]
            for q in range(m):
                if not ok[q]:
                    continue
                cq = camb[q]
                gq = base + 6 * cam_group[cq]
                Wq = Wbuf + q * 36
                if mine[q]:                    # camera columns of q: all 12 rows
                    om_ba_schur_qcols(Sp, nc, ck, gk, cq, WV, Wq)
                if mine[k]:                    # camera rows of k x intrinsics columns of q
                    om_ba_schur_kgrp(Sp, nc, ck, gq, WV, Wq)
                for i in range(n_gg):          # intrinsics x intrinsics
                    a = gg_a[i]
                    b = gg_b[i]
                    tt = (WV[a * 3 + 0] * Wq[b * 3 + 0]
                          + WV[a * 3 + 1] * Wq[b * 3 + 1]
                          + WV[a * 3 + 2] * Wq[b * 3 + 2])
                    Sgg[(gk + a - 6 - base) * n6 + gq + b - 6 - base] -= tt
    for a in range(n6):                        # store back the owned intrinsics entries
        if (a % 6) % T == t:
            g[base + a] = gg[a]
            diagU[base + a] = dg[a]
        for b in range(n6):
            if ((a % 6) * 6 + b % 6) % T == t:
                S[base + a, base + b] = Sgg[a * n6 + b]
    cost_out[0] = cost
    free(Wbuf); free(WVb); free(Jcb); free(eb); free(camb); free(ok); free(mine); free(Sgg)
    return 0


def _reduced_system_mt(const double[:, :, ::1] R, const double[:, ::1] C, const double[:, ::1] X,
                       const double[:, ::1] intr, const double[:, ::1] pp, const int32_t[::1] cam_group,
                       const int32_t[::1] obs_cam, const double[:, ::1] obs_uv, const int64_t[::1] pt_ptr,
                       double huber, double lam, const double[:, ::1] pw, const double[:, ::1] pt_tgt,
                       int workers):
    """reduced_system on `workers` threads, bit-identical to the serial kernel (see _rs_owner)."""
    from concurrent.futures import ThreadPoolExecutor
    cdef Py_ssize_t N = R.shape[0], G = intr.shape[0], P = X.shape[0], M = obs_cam.shape[0]
    cdef Py_ssize_t nc = 6 * N + 6 * G, p, maxlen = 0, i
    cdef int T = workers
    if (C.shape[0] != N or cam_group.shape[0] != N or pt_ptr.shape[0] != P + 1 or obs_uv.shape[0] != M
            or pw.shape[0] != P or pt_tgt.shape[0] != P or pt_ptr[P] != M or pt_ptr[0] != 0):
        raise ValueError("reduced_system: inconsistent array sizes")
    S_np = np.zeros((nc, nc), np.float64)
    g_np = np.zeros(nc, np.float64)
    diagU_np = np.zeros(nc, np.float64)
    Vinv_np = np.zeros((P, 3, 3), np.float64)
    gp_np = np.zeros((P, 3), np.float64)
    cdef double[:, ::1] S = S_np
    cdef double[::1] g = g_np, diagU = diagU_np
    cdef double[:, :, ::1] Vinv = Vinv_np
    cdef double[:, ::1] gp = gp_np
    for p in range(P):
        if pt_ptr[p + 1] - pt_ptr[p] > maxlen:
            maxlen = pt_ptr[p + 1] - pt_ptr[p]
    # cameras -> owner threads: contiguous ranges with about the same number of observations
    cnt = np.bincount(np.asarray(obs_cam), minlength=N).astype(np.float64)
    cum = np.cumsum(cnt)
    owner_np = np.minimum((T * (cum - cnt / 2) / max(float(cnt.sum()), 1.0)).astype(np.int32), T - 1) \
        if N else np.zeros(0, np.int32)
    cdef int32_t[::1] cam_owner = np.ascontiguousarray(owner_np, np.int32)
    costs_np = np.zeros(T, np.float64)
    cdef double[::1] costs = costs_np

    def owner(int t):
        cdef int rc
        with nogil:
            rc = _rs_owner(R, C, X, intr, pp, cam_group, obs_cam, obs_uv, pt_ptr, huber, lam, pw, pt_tgt, Vinv, gp,
                           S, g, diagU, cam_owner, maxlen, T, t, &costs[t])
        if rc != 0:
            raise MemoryError()

    with ThreadPoolExecutor(T) as ex:
        list(ex.map(owner, range(T)))
    for i in range(nc):
        S[i, i] += lam * diagU[i] + 1e-12
    return S_np, g_np, Vinv_np, gp_np, costs[0], diagU_np


cdef void _bs_range(const double[:, :, ::1] R, const double[:, ::1] C, const double[:, ::1] X,
                    const double[:, ::1] intr, const double[:, ::1] pp, const int32_t[::1] cam_group,
                    const int32_t[::1] obs_cam, const double[:, ::1] obs_uv, const int64_t[::1] pt_ptr,
                    double huber, const double[:, :, ::1] Vinv, const double[:, ::1] gp,
                    const double[::1] dc, double[:, ::1] dp, Py_ssize_t N, Py_ssize_t pa, Py_ssize_t pb) noexcept nogil:
    """back_substitute's loop for points [pa, pb)."""
    cdef Py_ssize_t p, q
    cdef double Jc[24]
    cdef double Jp[6]
    cdef double e[2]
    cdef double rhs[3]
    cdef double cost = 0.0, jd0, jd1
    cdef int a, b, cam, grp
    for p in range(pa, pb):
        rhs[0] = -gp[p, 0]
        rhs[1] = -gp[p, 1]
        rhs[2] = -gp[p, 2]
        for q in range(pt_ptr[p], pt_ptr[p + 1]):
            cam = obs_cam[q]
            grp = cam_group[cam]
            if not _linearize_obs(R, C, X, intr, pp, cam_group, cam, p, obs_uv[q, 0], obs_uv[q, 1],
                                  huber, Jc, Jp, e, &cost):
                continue
            jd0 = 0.0
            jd1 = 0.0
            for a in range(6):
                jd0 += Jc[a] * dc[6 * cam + a]
                jd1 += Jc[12 + a] * dc[6 * cam + a]
            for a in range(6):
                jd0 += Jc[6 + a] * dc[6 * N + 6 * grp + a]
                jd1 += Jc[18 + a] * dc[6 * N + 6 * grp + a]
            for b in range(3):
                rhs[b] -= Jp[b] * jd0 + Jp[3 + b] * jd1
        for a in range(3):
            dp[p, a] = Vinv[p, a, 0] * rhs[0] + Vinv[p, a, 1] * rhs[1] + Vinv[p, a, 2] * rhs[2]


cdef void _res_range(const double[:, :, ::1] R, const double[:, ::1] C, const double[:, ::1] X,
                     const double[:, ::1] intr, const int32_t[::1] cam_group, const int32_t[::1] obs_cam,
                     const int32_t[::1] obs_pt, const double[:, ::1] obs_uv, double[::1] out, char* behind,
                     Py_ssize_t qa, Py_ssize_t qb) noexcept nogil:
    """residuals' per-observation error for observations [qa, qb) (the cost is summed afterwards)."""
    cdef Py_ssize_t q
    cdef double x, y, z, nx, ny, r2, dd, ex, ey, s, f, k1, k2, k3, d0, d1, d2
    cdef int cam, g, pt
    for q in range(qa, qb):
        cam = obs_cam[q]
        pt = obs_pt[q]
        g = cam_group[cam]
        f = intr[g, 0]
        k1 = intr[g, 1]
        k2 = intr[g, 2]
        k3 = intr[g, 3]
        d0 = X[pt, 0] - C[cam, 0]
        d1 = X[pt, 1] - C[cam, 1]
        d2 = X[pt, 2] - C[cam, 2]
        x = R[cam, 0, 0] * d0 + R[cam, 0, 1] * d1 + R[cam, 0, 2] * d2
        y = R[cam, 1, 0] * d0 + R[cam, 1, 1] * d1 + R[cam, 1, 2] * d2
        z = R[cam, 2, 0] * d0 + R[cam, 2, 1] * d1 + R[cam, 2, 2] * d2
        if z <= 1e-6:
            out[q] = 1e30
            behind[q] = 1
            continue
        behind[q] = 0
        nx = x / z
        ny = y / z
        r2 = nx * nx + ny * ny
        dd = 1.0 + r2 * (k1 + r2 * (k2 + r2 * k3))
        ex = f * dd * nx + intr[g, 4] - obs_uv[q, 0]
        ey = f * dd * ny + intr[g, 5] - obs_uv[q, 1]
        s = sqrt(ex * ex + ey * ey)
        out[q] = s


def _split(n, parts):
    return [(n * j // parts, n * (j + 1) // parts) for j in range(parts) if n * (j + 1) // parts > n * j // parts]


def _back_substitute_mt(const double[:, :, ::1] R, const double[:, ::1] C, const double[:, ::1] X,
                        const double[:, ::1] intr, const double[:, ::1] pp, const int32_t[::1] cam_group,
                        const int32_t[::1] obs_cam, const double[:, ::1] obs_uv, const int64_t[::1] pt_ptr,
                        double huber, const double[:, :, ::1] Vinv, const double[:, ::1] gp,
                        const double[::1] dc, int workers):
    from concurrent.futures import ThreadPoolExecutor
    cdef Py_ssize_t N = R.shape[0], P = X.shape[0]
    if pt_ptr.shape[0] != P + 1 or Vinv.shape[0] != P or gp.shape[0] != P or dc.shape[0] < 6 * N:
        raise ValueError("back_substitute: inconsistent array sizes")
    dp_np = np.zeros((P, 3), np.float64)
    cdef double[:, ::1] dp = dp_np

    def job(rng):
        cdef Py_ssize_t pa = rng[0], pb = rng[1]
        with nogil:
            _bs_range(R, C, X, intr, pp, cam_group, obs_cam, obs_uv, pt_ptr, huber, Vinv, gp, dc, dp, N, pa, pb)

    with ThreadPoolExecutor(workers) as ex:
        list(ex.map(job, _split(P, 4 * workers)))
    return dp_np


def _residuals_mt(const double[:, :, ::1] R, const double[:, ::1] C, const double[:, ::1] X,
                  const double[:, ::1] intr, const double[:, ::1] pp, const int32_t[::1] cam_group,
                  const int32_t[::1] obs_cam, const int32_t[::1] obs_pt, const double[:, ::1] obs_uv,
                  double huber, int workers):
    from concurrent.futures import ThreadPoolExecutor
    cdef Py_ssize_t M = obs_cam.shape[0], q
    if obs_pt.shape[0] != M or obs_uv.shape[0] != M:
        raise ValueError("residuals: inconsistent array sizes")
    out_np = np.empty(M, np.float64)
    behind_np = np.empty(M, np.uint8)
    cdef double[::1] out = out_np
    cdef unsigned char[::1] bh = behind_np
    cdef char* behind = <char*>&bh[0] if M else NULL
    cdef double cost = 0.0, s

    def job(rng):
        cdef Py_ssize_t qa = rng[0], qb = rng[1]
        with nogil:
            _res_range(R, C, X, intr, cam_group, obs_cam, obs_pt, obs_uv, out, behind, qa, qb)

    with ThreadPoolExecutor(workers) as ex:
        list(ex.map(job, _split(M, 4 * workers)))
    with nogil:
        for q in range(M):
            if behind[q]:
                cost += 2.0 * huber * 1e6
                continue
            s = out[q]
            if s <= huber:
                cost += s * s
            else:
                cost += 2.0 * huber * s - huber * huber
    return out_np, cost
