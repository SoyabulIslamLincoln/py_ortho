"""Shared integer-only tiled matcher for CUDA C++ and Metal.

Each 64x64 tile computes distances once and reduces both directions. Only the two
best distances per row and the best per column leave shared memory. Query batches
bound scratch memory; ties always retain the lowest original descriptor index.
"""
import numpy as np

TILE = 64
INF = 1 << 30
TILED_PAIRS = 1_000_000

# This body is valid in both GPU languages; their small spelling differences are
# supplied by the backend. All threads reach the barrier, including edge tiles.
TILE_BODY = r"""
    SHARED unsigned int qa[512], qb[512];
    SHARED int dist[64 * 65];
    const int tid = LOCAL_ID;
    const int ti = GROUP_X, tj = GROUP_Y;
    const int i0 = ti * 64, j0 = tj * 64;
    for (int t = tid; t < 512; t += 256) {
        const int qi = i0 + t / 8, qj = j0 + t / 8;
        qa[t] = qi < n1 ? d1[qi * 8 + t % 8] : 0;
        qb[t] = qj < n2 ? d2[qj * 8 + t % 8] : 0;
    }
    BARRIER;
    for (int t = tid; t < 4096; t += 256) {
        const int r = t / 64, c = t % 64;
        int d = 1 << 30;
        if (i0 + r < n1 && j0 + c < n2) {
            d = 0;
            for (int k = 0; k < 8; ++k)
                d += POPCOUNT(qa[r * 8 + k] ^ qb[c * 8 + k]);
        }
        dist[r * 65 + c] = d;
    }
    BARRIER;
    if (tid < 64) {
        int best = 1 << 30, second = 1 << 30, bi = -1;
        for (int c = 0; c < 64; ++c) {
            const int d = dist[tid * 65 + c];
            if (d < best) { second = best; best = d; bi = j0 + c; }
            else if (d < second) second = d;
        }
        if (i0 + tid < n1) {
            const int o = tj * n1 + i0 + tid;
            rb[o] = best; rs[o] = second; ri[o] = bi;
        }
        best = 1 << 30; bi = -1;
        for (int r = 0; r < 64; ++r) {
            const int d = dist[r * 65 + tid];
            if (d < best) { best = d; bi = i0 + r; }
        }
        if (j0 + tid < n2) {
            const int o = ti * n2 + j0 + tid;
            cb[o] = best; ci[o] = bi;
        }
    }
"""

REDUCE_BODY = r"""
    const int i = GLOBAL_ID;
    if (i < n1) {
        int best = 1 << 30, second = 1 << 30, bi = -1;
        for (int t = 0; t < (n2 + 63) / 64; ++t) {
            const int o = t * n1 + i, d = rb[o];
            if (d < best) { second = best; best = d; bi = ri[o]; }
            else if (d < second) second = d;
            if (rs[o] < second) second = rs[o];
        }
        idx[i] = bi; bst[i] = best; sec[i] = second;
    }
    if (i < n2) {
        int best = 1 << 30, bi = -1;
        for (int t = 0; t < (n1 + 63) / 64; ++t) {
            const int o = t * n2 + i;
            if (cb[o] < best) { best = cb[o]; bi = ci[o]; }
        }
        back[i] = bi; backbest[i] = best;
    }
"""


def kernel_body(body, metal):
    replacements = ({"SHARED": "threadgroup", "LOCAL_ID": "int(thread_position_in_threadgroup.x)",
                     "GROUP_X": "int(threadgroup_position_in_grid.x)",
                     "GROUP_Y": "int(threadgroup_position_in_grid.y)",
                     "GLOBAL_ID": "int(thread_position_in_grid.x)",
                     "BARRIER": "threadgroup_barrier(mem_flags::mem_threadgroup)", "POPCOUNT": "popcount"}
                    if metal else
                    {"SHARED": "__shared__", "LOCAL_ID": "int(threadIdx.x)",
                     "GROUP_X": "int(blockIdx.x)", "GROUP_Y": "int(blockIdx.y)",
                     "GLOBAL_ID": "int(blockIdx.x * blockDim.x + threadIdx.x)",
                     "BARRIER": "__syncthreads()", "POPCOUNT": "__popc"})
    for key, value in replacements.items():
        body = body.replace(key, value)
    return body


def descriptors(d):
    d = np.ascontiguousarray(d, np.uint8)
    if d.ndim != 2 or d.shape[1] != 32:
        raise ValueError("expected 256-bit descriptors with shape (n, 32)")
    return d


def empty_result(n1, n2):
    return (np.full(n1, -1, np.int32), np.full(n1, INF, np.int32),
            np.full(n1, INF, np.int32), np.full(n2, -1, np.int32))


def match_mutual(d1, d2, upload, run_batch):
    """Backend-independent bounded batching, including empty inputs and exact ties."""
    d1, d2 = descriptors(d1), descriptors(d2)
    n1, n2 = len(d1), len(d2)
    idx, best, second, back = empty_result(n1, n2)
    colbest = np.full(n2, INF, np.int32)
    if not n1 or not n2:
        return idx, best, second, back
    # Five partial int arrays; target <= 32 MiB (plus one tile of rounding).
    rows = max(TILE, min(8192, (32 * 1024 * 1024 // (20 * n2)) * TILE))
    a = upload(d1.view(np.uint32))
    b = upload(d2.view(np.uint32))
    for start in range(0, n1, rows):
        end = min(start + rows, n1)
        ii, bb, ss, jj, dd = run_batch(a[start:end], b)
        idx[start:end], best[start:end], second[start:end] = ii, bb, ss
        better = dd < colbest
        back[better] = jj[better] + start
        colbest[better] = dd[better]
    return idx, best, second, back
