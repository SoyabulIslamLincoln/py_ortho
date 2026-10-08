/* Exact 256-bit descriptor distances. ARM64 NEON is part of the base ISA;
 * other targets retain the portable compiler popcount implementation. */
#ifndef OM_HAMMING_H
#define OM_HAMMING_H
#include <stdint.h>
#include <stddef.h>
#if defined(__aarch64__) || defined(_M_ARM64)
#include <arm_neon.h>
static inline int om_hamming256(const uint64_t *a, const uint64_t *b)
{
    uint8x16_t lo = vcntq_u8(veorq_u8(vld1q_u8((const uint8_t *)a),
                                     vld1q_u8((const uint8_t *)b)));
    uint8x16_t hi = vcntq_u8(veorq_u8(vld1q_u8((const uint8_t *)(a + 2)),
                                     vld1q_u8((const uint8_t *)(b + 2))));
    /* Widen before the horizontal sum: all 256 bits can differ. */
    return (int)vaddlvq_u8(vaddq_u8(lo, hi));
}
#else
static inline int om_hamming256(const uint64_t *a, const uint64_t *b)
{
    return om_popcount64(a[0] ^ b[0]) + om_popcount64(a[1] ^ b[1])
         + om_popcount64(a[2] ^ b[2]) + om_popcount64(a[3] ^ b[3]);
}
#endif

/* Distances and minima are integer-only; visit candidates in their original
 * order so equal distances retain exactly the same first index. */
static void om_match256(const uint64_t *a, const uint64_t *b,
                        ptrdiff_t n1, ptrdiff_t n2, int32_t *idx,
                        int32_t *best, int32_t *second, int32_t *back, int32_t *cb)
{
    ptrdiff_t i, j;
    for (i = 0; i < n1; ++i) {
        int bv = 1 << 30, sv = 1 << 30, bi = -1;
        for (j = 0; j < n2; ++j) {
            int d = om_hamming256(a + 4 * i, b + 4 * j);
            if (d < bv) { sv = bv; bv = d; bi = (int)j; }
            else if (d < sv) sv = d;
            if (back && d < cb[j]) {
                cb[j] = d;
                back[j] = (int)i;
            }
        }
        idx[i] = bi; best[i] = bv; second[i] = sv;
    }
}
#endif
