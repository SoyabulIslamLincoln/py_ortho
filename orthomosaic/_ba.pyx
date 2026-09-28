# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
# cython: initializedcheck=False, nonecheck=False
"""
Bundle adjustment kernels (Levenberg-Marquardt with Schur complement).

Parameter layout of the reduced ("camera side") system, size nc = 6*N + 3*G:
    camera i : [w0 w1 w2 | c0 c1 c2]  at 6*i      (left-multiplied rotation update, centre)
    group  g : [f k1 k2]               at 6*N + 3*g
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
    """Jc: 2x9 (w, C, f k1 k2), Jp: 2x3, e: 2 residual, all pre-scaled by sqrt(Huber weight)."""
    cdef int g = cam_group[cam]
    cdef double f = intr[g, 0], k1 = intr[g, 1], k2 = intr[g, 2]
    cdef double dX0 = X[pt, 0] - C[cam, 0], dX1 = X[pt, 1] - C[cam, 1], dX2 = X[pt, 2] - C[cam, 2]
    cdef double x = R[cam, 0, 0] * dX0 + R[cam, 0, 1] * dX1 + R[cam, 0, 2] * dX2
    cdef double y = R[cam, 1, 0] * dX0 + R[cam, 1, 1] * dX1 + R[cam, 1, 2] * dX2
    cdef double z = R[cam, 2, 0] * dX0 + R[cam, 2, 1] * dX1 + R[cam, 2, 2] * dX2
    if z <= 1e-6:
        return False
    cdef double iz = 1.0 / z, nx = x * iz, ny = y * iz
    cdef double r2 = nx * nx + ny * ny
    cdef double d = 1.0 + k1 * r2 + k2 * r2 * r2
    cdef double dd = k1 + 2.0 * k2 * r2
    cdef double ex = f * d * nx + pp[g, 0] - uo
    cdef double ey = f * d * ny + pp[g, 1] - vo
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
        Jc[r * 9 + 0] = sw * (-J[r][1] * z + J[r][2] * y)
        Jc[r * 9 + 1] = sw * (J[r][0] * z - J[r][2] * x)
        Jc[r * 9 + 2] = sw * (-J[r][0] * y + J[r][1] * x)
        for k in range(3):
            # centre: J @ (-R) ; point: J @ R
            Jp[r * 3 + k] = sw * (J[r][0] * R[cam, 0, k] + J[r][1] * R[cam, 1, k] + J[r][2] * R[cam, 2, k])
            Jc[r * 9 + 3 + k] = -Jp[r * 3 + k]
    Jc[0 * 9 + 6] = sw * d * nx
    Jc[1 * 9 + 6] = sw * d * ny
    Jc[0 * 9 + 7] = sw * f * r2 * nx
    Jc[1 * 9 + 7] = sw * f * r2 * ny
    Jc[0 * 9 + 8] = sw * f * r2 * r2 * nx
    Jc[1 * 9 + 8] = sw * f * r2 * r2 * ny
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
                   const double[:, ::1] pw, const double[:, ::1] pt_tgt):
    """Build the damped Schur-reduced normal equations.

    pw, pt_tgt: per-point position-prior weight (1/sigma^2) and target (P, 3). Zero weight = no
    prior (ordinary tie point). Ground control points enter this way, as points whose world
    position is pulled toward the surveyed coordinate.

    Returns (S (nc,nc), g (nc), Vinv (P,3,3), gp (P,3), cost, diagU (nc)); S is already
    damped with lam * diag(U).
    Solve S dc = -g, then dp = back_substitute(...).
    """
    cdef Py_ssize_t N = R.shape[0], G = intr.shape[0], P = X.shape[0]
    cdef Py_ssize_t nc = 6 * N + 3 * G
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
    cdef double* Wbuf = <double*>malloc(max(maxlen, 1) * 27 * sizeof(double))   # W_k = Jc^T Jp (9x3)
    cdef int* Ibuf = <int*>malloc(max(maxlen, 1) * 9 * sizeof(int))             # global indices
    cdef char* ok = <char*>malloc(max(maxlen, 1) * sizeof(char))
    cdef double Jc[18]
    cdef double Jp[6]
    cdef double e[2]
    cdef double V[9]
    cdef double Vd[9]
    cdef double Vi[9]
    cdef double gpt[3]
    cdef double WV[27]
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
                        Ibuf[k * 9 + a] = 6 * cam + a
                    for a in range(3):
                        Ibuf[k * 9 + 6 + a] = 6 * N + 3 * grp + a
                    # U block, gradient
                    for a in range(9):
                        ka = Ibuf[k * 9 + a]
                        g[ka] += Jc[a] * e[0] + Jc[9 + a] * e[1]
                        diagU[ka] += Jc[a] * Jc[a] + Jc[9 + a] * Jc[9 + a]
                        for b in range(9):
                            S[ka, Ibuf[k * 9 + b]] += Jc[a] * Jc[b] + Jc[9 + a] * Jc[9 + b]
                        # W = Jc^T Jp
                        for b in range(3):
                            Wbuf[k * 27 + a * 3 + b] = Jc[a] * Jp[b] + Jc[9 + a] * Jp[3 + b]
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
                    for a in range(9):
                        for b in range(3):
                            WV[a * 3 + b] = (Wbuf[k * 27 + a * 3 + 0] * Vi[0 * 3 + b]
                                             + Wbuf[k * 27 + a * 3 + 1] * Vi[1 * 3 + b]
                                             + Wbuf[k * 27 + a * 3 + 2] * Vi[2 * 3 + b])
                    for a in range(9):
                        ka = Ibuf[k * 9 + a]
                        g[ka] -= WV[a * 3 + 0] * gpt[0] + WV[a * 3 + 1] * gpt[1] + WV[a * 3 + 2] * gpt[2]
                    for q in range(m):
                        if not ok[q]:
                            continue
                        for a in range(9):
                            ka = Ibuf[k * 9 + a]
                            for b in range(9):
                                kb = Ibuf[q * 9 + b]
                                t = (WV[a * 3 + 0] * Wbuf[q * 27 + b * 3 + 0]
                                     + WV[a * 3 + 1] * Wbuf[q * 27 + b * 3 + 1]
                                     + WV[a * 3 + 2] * Wbuf[q * 27 + b * 3 + 2])
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
                    const double[::1] dc):
    """dp_p = Vinv_p (-gp_p - sum_k W_k^T dc)."""
    cdef Py_ssize_t N = R.shape[0], P = X.shape[0], p, q
    dp_np = np.zeros((P, 3), np.float64)
    cdef double[:, ::1] dp = dp_np
    cdef double Jc[18]
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
                    jd1 += Jc[9 + a] * dc[6 * cam + a]
                for a in range(3):
                    jd0 += Jc[6 + a] * dc[6 * N + 3 * grp + a]
                    jd1 += Jc[15 + a] * dc[6 * N + 3 * grp + a]
                for b in range(3):
                    rhs[b] -= Jp[b] * jd0 + Jp[3 + b] * jd1
            for a in range(3):
                dp[p, a] = Vinv[p, a, 0] * rhs[0] + Vinv[p, a, 1] * rhs[1] + Vinv[p, a, 2] * rhs[2]
    return dp_np


def residuals(const double[:, :, ::1] R, const double[:, ::1] C, const double[:, ::1] X,
              const double[:, ::1] intr, const double[:, ::1] pp, const int32_t[::1] cam_group,
              const int32_t[::1] obs_cam, const int32_t[::1] obs_pt, const double[:, ::1] obs_uv,
              double huber):
    """Per-observation reprojection error (px; inf if behind the camera) and total robust cost."""
    cdef Py_ssize_t M = obs_cam.shape[0], q
    out_np = np.empty(M, np.float64)
    cdef double[::1] out = out_np
    cdef double cost = 0.0, x, y, z, nx, ny, r2, dd, ex, ey, s, f, k1, k2, d0, d1, d2
    cdef int cam, g, pt
    with nogil:
        for q in range(M):
            cam = obs_cam[q]
            pt = obs_pt[q]
            g = cam_group[cam]
            f = intr[g, 0]
            k1 = intr[g, 1]
            k2 = intr[g, 2]
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
            dd = 1.0 + k1 * r2 + k2 * r2 * r2
            ex = f * dd * nx + pp[g, 0] - obs_uv[q, 0]
            ey = f * dd * ny + pp[g, 1] - obs_uv[q, 1]
            s = sqrt(ex * ex + ey * ey)
            out[q] = s
            if s <= huber:
                cost += s * s
            else:
                cost += 2.0 * huber * s - huber * huber
    return out_np, cost
