/* Accumulation loops of _ba._rs_owner (multi-threaded reduced_system).
 *
 * The expressions are those of _ba.reduced_system (same operands, order and parenthesisation),
 * compiled into the same extension with the same flags, so every S / g / diag(U) entry receives
 * bit-identical contributions; only the loop structure differs (contiguous column blocks through
 * restrict pointers, which the compiler can vectorise).
 *
 * Layout: S is row-major with row stride nc; camera c occupies rows/columns 6c..6c+5, its
 * intrinsics group the 6 entries starting at gk (gq) = 6N + 6 * group.
 */
#ifndef OM_BA_KERN_H
#define OM_BA_KERN_H

#include <stddef.h>

/* U block of one observation, camera rows (all 12 columns) and intrinsics rows x camera columns. */
static inline void om_ba_u_cam(double *restrict S, ptrdiff_t nc, double *restrict g, double *restrict diagU,
                               int ck, ptrdiff_t gk, const double *restrict Jc, const double *restrict e)
{
    int a, b;
    for (a = 0; a < 12; a++) {
        ptrdiff_t ka = a < 6 ? 6 * (ptrdiff_t)ck + a : gk + a - 6;
        double *restrict row = S + ka * nc;
        double ja = Jc[a], jb = Jc[12 + a];
        if (a < 6) {
            g[ka] = (g[ka] + ((ja * e[0]) + (jb * e[1])));
            diagU[ka] = (diagU[ka] + ((ja * ja) + (jb * jb)));
            for (b = 0; b < 6; b++)
                row[6 * (ptrdiff_t)ck + b] = (row[6 * (ptrdiff_t)ck + b] + ((ja * Jc[b]) + (jb * Jc[12 + b])));
            for (b = 6; b < 12; b++)
                row[gk + b - 6] = (row[gk + b - 6] + ((ja * Jc[b]) + (jb * Jc[12 + b])));
        } else {
            for (b = 0; b < 6; b++)
                row[6 * (ptrdiff_t)ck + b] = (row[6 * (ptrdiff_t)ck + b] + ((ja * Jc[b]) + (jb * Jc[12 + b])));
        }
    }
}

/* Schur pair (k, q): columns of camera q for all 12 rows of k (owner of camera q). */
static inline void om_ba_schur_qcols(double *restrict S, ptrdiff_t nc, int ck, ptrdiff_t gk, int cq,
                                     const double *restrict WV, const double *restrict Wq)
{
    int a, b;
    for (a = 0; a < 12; a++) {
        ptrdiff_t ka = a < 6 ? 6 * (ptrdiff_t)ck + a : gk + a - 6;
        double *restrict col = S + ka * nc + 6 * (ptrdiff_t)cq;
        double w0 = WV[a * 3 + 0], w1 = WV[a * 3 + 1], w2 = WV[a * 3 + 2];
        for (b = 0; b < 6; b++) {
            double tt = (((w0 * Wq[b * 3 + 0]) + (w1 * Wq[b * 3 + 1])) + (w2 * Wq[b * 3 + 2]));
            col[b] = (col[b] - tt);
        }
    }
}

/* Schur pair (k, q): camera rows of k x intrinsics columns of q (owner of camera k). */
static inline void om_ba_schur_kgrp(double *restrict S, ptrdiff_t nc, int ck, ptrdiff_t gq,
                                    const double *restrict WV, const double *restrict Wq)
{
    int a, b;
    for (a = 0; a < 6; a++) {
        double *restrict col = S + (6 * (ptrdiff_t)ck + a) * nc + gq;
        double w0 = WV[a * 3 + 0], w1 = WV[a * 3 + 1], w2 = WV[a * 3 + 2];
        for (b = 6; b < 12; b++) {
            double tt = (((w0 * Wq[b * 3 + 0]) + (w1 * Wq[b * 3 + 1])) + (w2 * Wq[b * 3 + 2]));
            col[b - 6] = (col[b - 6] - tt);
        }
    }
}

#endif
