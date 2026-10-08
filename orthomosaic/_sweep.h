/* Inner loops of _dense.sweep_hypothesis (plane-sweep step for all sources).
 *
 * Every arithmetic expression is written exactly as in _dense.ncc_warp / _dense.combine (same
 * operand order, parenthesisation, types and float32/float64 conversions), and this header is
 * compiled into the same extension with the same flags, so the compiler forms the same fused
 * multiply-adds and the results are bit-identical. Only the memory access differs: plain
 * pointers over contiguous rows instead of strided memoryview indexing.
 */
#ifndef OM_SWEEP_H
#define OM_SWEEP_H

#include <math.h>
#include <stddef.h>

typedef struct {
    const float *img;
    ptrdiff_t h, w;
    double r00, r01, r02, r10, r11, r12, r20, r21, r22;
    double b0, b1, b2;
    double f, k1, k2, k3, cx, cy;
    float *Jrow;      /* (W,) warped source row */
    float *Vrow;      /* (W,) 1 = valid sample */
    float *ring;      /* (2r+2, 4, W) horizontal window sums, row y at slot y % (2r+2) */
    double *cs;       /* (4, W) vertical running sums */
    float *nrow;      /* (W,) NCC of the current output row */
} om_sw_src;

/* ncc_warp phase 1 for one row and every source. inv_row: (W,), ray0/1/2: rows of the 3 planes.
 * Two passes per source: the projection of every pixel (no data-dependent memory access, so the
 * compiler vectorises it; pixels behind the camera are flagged with u = -1, which fails the
 * bounds test exactly as the original early exit did), then the bilinear fetch.
 * U, Vv, D: (W,) double scratch. */
static inline void om_sw_warp_row(const float *inv_row, const float *ray0, const float *ray1, const float *ray2,
                                  om_sw_src *src, int n, ptrdiff_t W, double *U, double *Vv, double *D)
{
    ptrdiff_t x, x0, y0;
    int s;
    for (x = 0; x < W; x++)
        D[x] = 1.0 / ((double)inv_row[x]);
    for (s = 0; s < n; s++) {
        om_sw_src *p = &src[s];
        const double r00 = p->r00, r01 = p->r01, r02 = p->r02, r10 = p->r10, r11 = p->r11, r12 = p->r12;
        const double r20 = p->r20, r21 = p->r21, r22 = p->r22, b0 = p->b0, b1 = p->b1, b2 = p->b2;
        const double f = p->f, k1 = p->k1, k2 = p->k2, k3 = p->k3, cx = p->cx, cy = p->cy;
        const float *img = p->img;
        const ptrdiff_t w = p->w;
        const double wl = ((double)p->w - 1.001), hl = ((double)p->h - 1.001);
        float *J = p->Jrow, *V = p->Vrow;
        for (x = 0; x < W; x++) {
            double d = D[x], rx = ray0[x], ry = ray1[x], rz = ray2[x];
            double X = (b0 + (d * (((r00 * rx) + (r01 * ry)) + (r02 * rz))));
            double Y = (b1 + (d * (((r10 * rx) + (r11 * ry)) + (r12 * rz))));
            double Z = (b2 + (d * (((r20 * rx) + (r21 * ry)) + (r22 * rz))));
            double nx = (X / Z), ny = (Y / Z);
            double r2 = ((nx * nx) + (ny * ny));
            double dd = (f * (1.0 + (r2 * (k1 + (r2 * (k2 + (r2 * k3)))))));
            double u = ((dd * nx) + cx), v = ((dd * ny) + cy);
            U[x] = (Z <= 1e-6) ? -1.0 : u;
            Vv[x] = v;
        }
        for (x = 0; x < W; x++) {
            double u = U[x], v = Vv[x], fx, fy, a00, a01, a10, a11, val;
            J[x] = 0.0f;
            V[x] = 0.0f;
            if (u < 0.0 || v < 0.0 || u > wl || v > hl)
                continue;
            x0 = (ptrdiff_t)u;
            y0 = (ptrdiff_t)v;
            fx = (u - (double)x0);
            fy = (v - (double)y0);
            a00 = img[y0 * w + x0];
            a01 = img[y0 * w + x0 + 1];
            a10 = img[(y0 + 1) * w + x0];
            a11 = img[(y0 + 1) * w + x0 + 1];
            val = ((((a00 * (1.0 - fx)) + (a01 * fx)) * (1.0 - fy)) + (((a10 * (1.0 - fx)) + (a11 * fx)) * fy));
            if (isfinite(val)) {
                J[x] = (float)val;
                V[x] = 1.0f;
            }
        }
    }
}

/* ncc_warp phase 2 (horizontal running window sums) of one source row into hs (4, W). */
static inline void om_sw_hsum_row(const float *ref_row, const om_sw_src *p, ptrdiff_t W, int r, float *hs)
{
    ptrdiff_t x, lim = r < W ? r : W;
    double sv = 0, sj = 0, sjj = 0, sij = 0;
    const float *J = p->Jrow, *V = p->Vrow;
    for (x = 0; x < lim; x++) {
        sv = (sv + V[x]);
        sj = (sj + J[x]);
        sjj = (sjj + (J[x] * J[x]));
        sij = (sij + (J[x] * ref_row[x]));
    }
    for (x = 0; x < W; x++) {
        if (x + r < W) {
            sv = (sv + V[x + r]);
            sj = (sj + J[x + r]);
            sjj = (sjj + (J[x + r] * J[x + r]));
            sij = (sij + (J[x + r] * ref_row[x + r]));
        }
        if (x - r - 1 >= 0) {
            sv = (sv - V[x - r - 1]);
            sj = (sj - J[x - r - 1]);
            sjj = (sjj - (J[x - r - 1] * J[x - r - 1]));
            sij = (sij - (J[x - r - 1] * ref_row[x - r - 1]));
        }
        hs[x] = (float)sv;
        hs[W + x] = (float)sj;
        hs[2 * W + x] = (float)sjj;
        hs[3 * W + x] = (float)sij;
    }
}

static inline void om_sw_cs_add(double *cs, const float *hs, ptrdiff_t W)
{
    ptrdiff_t x;
    for (x = 0; x < W; x++) {
        cs[x] = (cs[x] + hs[x]);
        cs[W + x] = (cs[W + x] + hs[W + x]);
        cs[2 * W + x] = (cs[2 * W + x] + hs[2 * W + x]);
        cs[3 * W + x] = (cs[3 * W + x] + hs[3 * W + x]);
    }
}

static inline void om_sw_cs_sub(double *cs, const float *hs, ptrdiff_t W)
{
    ptrdiff_t x;
    for (x = 0; x < W; x++) {
        cs[x] = (cs[x] - hs[x]);
        cs[W + x] = (cs[W + x] - hs[W + x]);
        cs[2 * W + x] = (cs[2 * W + x] - hs[2 * W + x]);
        cs[3 * W + x] = (cs[3 * W + x] - hs[3 * W + x]);
    }
}

/* cs += new row; cs -= old row (the add before the subtract, as ncc_warp does per element) */
static inline void om_sw_cs_addsub(double *cs, const float *ha, const float *hs, ptrdiff_t W)
{
    ptrdiff_t x;
    for (x = 0; x < 4 * W; x++) {
        double c = (cs[x] + ha[x]);
        cs[x] = (c - hs[x]);
    }
}

/* ncc_warp phase 3: NCC of one output row from the vertical sums. */
static inline void om_sw_ncc_row(const double *cs, const float *mu_row, const float *sd_row, double kk,
                                 ptrdiff_t W, float *out)
{
    ptrdiff_t x;
    double cov, mj, m2, mij, sdj, ncc;
    for (x = 0; x < W; x++) {
        cov = (cs[x] * kk);
        if (cov <= 0.9) {
            out[x] = -1.0f;
            continue;
        }
        mj = ((cs[W + x] * kk) / cov);
        m2 = ((cs[2 * W + x] * kk) / cov);
        mij = ((cs[3 * W + x] * kk) / cov);
        sdj = (m2 - (mj * mj));
        if (sdj > 1e-6) {
            sdj = sqrt(sdj);
        } else {
            sdj = 1e-3;
        }
        ncc = ((mij - (mu_row[x] * mj)) / (sd_row[x] * sdj));
        if (ncc > 1.0) {
            out[x] = (float)1.0;
        } else if (ncc < -1.0) {
            out[x] = (float)-1.0;
        } else {
            out[x] = (float)ncc;
        }
    }
}

/* combine for one row: mean of the top_k source scores, then the streaming winner update. */
static inline void om_sw_combine_row(const om_sw_src *src, int n, int kk, int i, ptrdiff_t W, float *best,
                                     float *prev, float *s_prev_best, float *s_next_best, int *idx)
{
    ptrdiff_t x;
    int a, c;
    float buf[16];
    float t, s;
    for (x = 0; x < W; x++) {
        for (a = 0; a < n; a++)
            buf[a] = src[a].nrow[x];
        for (a = 1; a < n; a++) {                 /* insertion sort, descending */
            t = buf[a];
            c = a - 1;
            while (c >= 0 && buf[c] < t) {
                buf[c + 1] = buf[c];
                c -= 1;
            }
            buf[c + 1] = t;
        }
        s = 0;
        for (a = 0; a < kk; a++)
            s = (s + buf[a]);
        s = (s / ((float)kk));
        if (idx[x] == (i - 1))
            s_next_best[x] = s;
        if (s > best[x]) {
            best[x] = s;
            s_prev_best[x] = prev[x];
            s_next_best[x] = -2.0f;
            idx[x] = i;
        }
        prev[x] = s;
    }
}

#endif
