"""Pair selection, robust matching and global (bundle) alignment.

Camera model: nadir images over near-planar ground, so each image i gets an
affine map  world = P_i @ [u, v, 1]  from full-res pixels to ground metres.

Global solve (all linear least squares, X and Y decouple and share one
normal matrix of size 3N x 3N):
  1. relative:  minimise  sum ||P_i p - P_j q||^2 over all inlier matches,
                with one reference image fixed (gauge).
  2. georef:    robust similarity fit of stage-1 image centres -> GPS (UTM).
  3. joint:     re-solve in metres with matches + GPS priors + a weak prior
                on each image's linear part (handles collinear flight lines).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np

from . import _core
from .features import Features

log = logging.getLogger(__name__)


@dataclass
class PairMatch:
    i: int
    j: int
    pi: np.ndarray       # (K, 2) full-res pixels in image i
    pj: np.ndarray       # (K, 2) full-res pixels in image j
    ci: np.ndarray       # (K, 3) colours around the points in image i
    cj: np.ndarray
    n_inliers: int


# --------------------------------------------------------------------------
# Pair candidates
# --------------------------------------------------------------------------

def candidate_pairs(positions: Optional[np.ndarray], n: int, k: int = 8,
                    exhaustive_below: int = 40) -> list[tuple[int, int]]:
    """k nearest neighbours by GPS; otherwise all pairs (small sets) or a
    sliding window over capture order."""
    pairs = set()
    if positions is not None and n > 2:
        k = min(k, n - 1)
        for start in range(0, n, 512):
            blk = positions[start:start + 512]
            d = np.linalg.norm(blk[:, None, :] - positions[None, :, :], axis=2)
            d[np.arange(len(blk)), np.arange(start, start + len(blk))] = np.inf
            nn = np.argpartition(d, k - 1, axis=1)[:, :k]
            for r, row in enumerate(nn):
                i = start + r
                for j in row:
                    pairs.add((min(i, int(j)), max(i, int(j))))
    elif n <= exhaustive_below:
        pairs = {(i, j) for i in range(n) for j in range(i + 1, n)}
    else:
        pairs = {(i, j) for i in range(n) for j in range(i + 1, min(n, i + 1 + k))}
    return sorted(pairs)


# --------------------------------------------------------------------------
# Pairwise matching
# --------------------------------------------------------------------------

def match_pair(backend, i: int, j: int, fi: Features, fj: Features, ransac_thresh: float,
               ratio: float = 0.8, max_hamming: int = 80, min_inliers: int = 25,
               max_points: int = 150, seed: int = 1) -> Optional[PairMatch]:
    if len(fi) < min_inliers or len(fj) < min_inliers:
        return None
    idx12, b12, s12, idx21 = backend.match_mutual(fi.desc, fj.desc)
    q = np.arange(len(idx12))
    ok = (idx12 >= 0) & (b12 <= max_hamming) & (b12 < ratio * s12)
    ok &= idx21[np.clip(idx12, 0, None)] == q            # mutual nearest neighbours
    if ok.sum() < min_inliers:
        return None
    a, b = q[ok], idx12[ok]
    src = np.ascontiguousarray(fi.xy[a])
    dst = np.ascontiguousarray(fj.xy[b])
    A, mask = _core.ransac_affine(src, dst, 4000, ransac_thresh, seed=(i * 7919 + j * 104729 + seed) | 1)
    if A is None:
        return None
    inl = mask.astype(bool)
    n_in = int(inl.sum())
    if n_in < min_inliers:
        return None
    # reject implausible relative geometry (strong shear / scale jump => false match)
    sv = np.linalg.svd(A[:, :2], compute_uv=False)
    if sv[1] < 1e-6 or sv[0] / sv[1] > 1.5 or not (0.33 < np.sqrt(sv[0] * sv[1]) < 3.0):
        return None
    a, b = a[inl], b[inl]
    if len(a) > max_points:
        # keep a spatially spread subset
        rng = np.random.default_rng(i * 1000003 + j)
        sel = rng.choice(len(a), max_points, replace=False)
        a, b = a[sel], b[sel]
    return PairMatch(i, j, fi.xy[a].copy(), fj.xy[b].copy(), fi.color[a].copy(), fj.color[b].copy(), n_in)


# --------------------------------------------------------------------------
# Global alignment
# --------------------------------------------------------------------------

def largest_component(n: int, pairs: list[PairMatch]) -> list[int]:
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for p in pairs:
        ra, rb = find(p.i), find(p.j)
        if ra != rb:
            parent[ra] = rb
    roots = [find(i) for i in range(n)]
    best = max(set(roots), key=roots.count)
    return [i for i in range(n) if roots[i] == best]


class _Normal:
    """Accumulates the shared normal matrix for X and Y parameter rows."""

    def __init__(self, n: int):
        self.N = np.zeros((3 * n, 3 * n))
        self.b = np.zeros((3 * n, 2))

    def add_match(self, li, lj, P, Q, w):
        # residual: P x_i - Q x_j   (P, Q: (K, 3) homogeneous normalised coords)
        si, sj = slice(3 * li, 3 * li + 3), slice(3 * lj, 3 * lj + 3)
        self.N[si, si] += w * P.T @ P
        self.N[sj, sj] += w * Q.T @ Q
        self.N[si, sj] -= w * P.T @ Q
        self.N[sj, si] -= w * Q.T @ P

    def add_prior(self, li, k, target_xy, w):
        """Parameter k (0=a/d, 1=b/e, 2=c/f) of image li ~ target (for X row, Y row)."""
        r = 3 * li + k
        self.N[r, r] += w
        self.b[r] += w * np.asarray(target_xy)

    def solve(self, scale_lock: Optional[np.ndarray] = None):
        """scale_lock: (n, 2, 2) reference linear parts. If given, adds a hard
        constraint per axis that the mean projection of each image's linear part
        onto its reference equals 1 -- this removes the global shrink mode that
        otherwise biases world-space least squares towards smaller scale."""
        n3 = self.N.shape[0]
        N = self.N + 1e-12 * np.eye(n3)
        if scale_lock is None:
            sol = np.linalg.solve(N, self.b)
            return sol[:, 0].reshape(-1, 3), sol[:, 1].reshape(-1, 3)
        out = []
        for ax in range(2):
            ref = scale_lock[:, ax, :]                        # (n, 2): (a0, b0) or (d0, e0)
            v = np.zeros(n3)
            nrm = np.sum(ref ** 2, axis=1)
            v[0::3] = ref[:, 0] / nrm
            v[1::3] = ref[:, 1] / nrm
            # bordered (KKT) system: [N v; v^T 0] [x; lam] = [b; n]
            K = np.zeros((n3 + 1, n3 + 1))
            K[:n3, :n3] = N
            K[:n3, n3] = v
            K[n3, :n3] = v
            rhs = np.append(self.b[:, ax], float(len(ref)))
            out.append(np.linalg.solve(K, rhs)[:n3].reshape(-1, 3))
        return out[0], out[1]


@dataclass
class Alignment:
    used: list[int]                  # frame indices kept (largest connected component)
    affines: dict[int, np.ndarray]   # frame index -> (2, 3): world = P @ [u, v, 1]
    georeferenced: bool
    gsd: float                       # metres (or ref-pixels) per full-res pixel, median
    residual_px: float               # RMS match residual in pixels after the solve


def _norm_coords(pts, c, s):
    q = (pts - c) / s
    return np.column_stack([q, np.ones(len(q))])


def pair_residuals_px(al: "Alignment", pairs: list[PairMatch]) -> np.ndarray:
    """RMS disagreement of each pair after the solve, in pixels of image i
    (NaN for pairs outside the solved block)."""
    out = np.full(len(pairs), np.nan)
    for k, p in enumerate(pairs):
        if p.i in al.affines and p.j in al.affines:
            Ai, Aj = al.affines[p.i], al.affines[p.j]
            a = p.pi @ Ai[:, :2].T + Ai[:, 2]
            b = p.pj @ Aj[:, :2].T + Aj[:, 2]
            scale_i = np.sqrt(abs(np.linalg.det(Ai[:, :2])))
            out[k] = np.sqrt(np.mean(np.sum((a - b) ** 2, axis=1))) / max(scale_i, 1e-12)
    return out


def solve_alignment(frames, pairs: list[PairMatch], positions: Optional[np.ndarray],
                    gps_sigma: float = 3.0, match_sigma_px: float = 4.0,
                    linear_prior_rel: float = 0.1, min_reject_px: float = 10.0,
                    max_rounds: int = 15) -> tuple[Alignment, list[PairMatch]]:
    """Robust global alignment: solve, drop pairs that disagree with the block
    (false matches, e.g. on repetitive roofs), re-solve until stable.
    Returns (alignment, pairs that survived)."""
    active = list(pairs)
    for rnd in range(max_rounds):
        al = _solve_once(frames, active, positions, gps_sigma, match_sigma_px, linear_prior_rel)
        r = pair_residuals_px(al, active)
        ok = np.isfinite(r)
        med = float(np.median(r[ok])) if ok.any() else 0.0
        thr = max(min_reject_px, 3.0 * 1.4826 * med)
        bad = ok & (r > thr)
        if not bad.any():
            break
        # drop the worst offenders first; a single bad pair skews its neighbours too
        worst = np.argsort(-np.where(bad, r, -np.inf))[:max(1, int(bad.sum()) // 2)]
        log.info("  robust solve round %d: median pair error %.1f px, dropping %d pair(s) above %.1f px",
                 rnd + 1, med, len(worst), thr)
        drop = set(worst)
        active = [p for k, p in enumerate(active) if k not in drop]
    return al, active


def _solve_once(frames, pairs: list[PairMatch], positions: Optional[np.ndarray],
                gps_sigma: float, match_sigma_px: float, linear_prior_rel: float) -> Alignment:
    n_all = len(frames)
    used = largest_component(n_all, pairs)
    local = {g: l for l, g in enumerate(used)}
    n = len(used)
    pairs = [p for p in pairs if p.i in local and p.j in local]
    ctr = np.array([[(frames[g].width - 1) / 2.0, (frames[g].height - 1) / 2.0] for g in used])
    scl = np.array([float(max(frames[g].width, frames[g].height)) for g in used])

    pre = []
    deg = np.zeros(n)
    for p in pairs:
        li, lj = local[p.i], local[p.j]
        pre.append((li, lj, _norm_coords(p.pi, ctr[li], scl[li]), _norm_coords(p.pj, ctr[lj], scl[lj])))
        deg[li] += len(p.pi)
        deg[lj] += len(p.pi)
    ref = int(np.argmax(deg)) if n else 0

    # ---- stage 1: relative, gauge = reference image (y flipped -> y-up world)
    ne = _Normal(n)
    for li, lj, P, Q in pre:
        ne.add_match(li, lj, P, Q, 1.0)
    big = 1e6 * max(1.0, float(deg.max(initial=1.0)))
    ne.add_prior(ref, 0, (1.0, 0.0), big)
    ne.add_prior(ref, 1, (0.0, -1.0), big)
    ne.add_prior(ref, 2, (0.0, 0.0), big)
    X, Y = ne.solve()
    L = np.stack([np.stack([X[:, 0], X[:, 1]], 1), np.stack([Y[:, 0], Y[:, 1]], 1)], 1)  # (n,2,2)
    T = np.stack([X[:, 2], Y[:, 2]], 1)                                                 # (n,2)

    georef = False
    if positions is not None:
        has = np.array([np.all(np.isfinite(positions[g])) for g in used])
        gl = np.where(has)[0]
        if len(gl) >= 2:
            z = T[gl, 0] + 1j * T[gl, 1]
            wv = positions[[used[k] for k in gl], 0] + 1j * positions[[used[k] for k in gl], 1]
            keep = np.ones(len(gl), bool)
            alpha = beta = None
            for _ in range(4):
                zm, wm = z[keep].mean(), wv[keep].mean()
                den = np.sum(np.abs(z[keep] - zm) ** 2)
                if den < 1e-12:
                    break
                alpha = np.sum(np.conj(z[keep] - zm) * (wv[keep] - wm)) / den
                beta = wm - alpha * zm
                res = np.abs(alpha * z + beta - wv)
                thr = max(3 * gps_sigma, 3 * 1.4826 * np.median(res[keep]))
                new_keep = res < thr
                if new_keep.sum() < 2 or np.array_equal(new_keep, keep):
                    break
                keep = new_keep
            spread = np.sqrt(np.mean(np.abs(wv - wv.mean()) ** 2))
            if alpha is not None and abs(alpha) > 0 and spread > max(2 * gps_sigma, 1.0):
                georef = True
                S = np.array([[alpha.real, -alpha.imag], [alpha.imag, alpha.real]])
                t0 = np.array([beta.real, beta.imag])
                L = np.einsum("ab,nbc->nac", S, L)
                T = T @ S.T + t0
                if (~keep).any():
                    log.warning("%d image(s) have GPS positions inconsistent with the image matches; "
                                "their GPS was ignored", int((~keep).sum()))
                # ---- stage 3: joint refinement in metres
                scale_i = np.sqrt(np.abs(np.linalg.det(L)))          # metres per normalised unit
                gsd = float(np.median(scale_i / scl))                  # metres per pixel
                sig_m = match_sigma_px * gsd
                ne = _Normal(n)
                for li, lj, P, Q in pre:
                    ne.add_match(li, lj, P, Q, 1.0 / sig_m ** 2)
                for k_, li in enumerate(gl):
                    if keep[k_]:
                        ne.add_prior(li, 2, positions[used[li]], 1.0 / gps_sigma ** 2)
                for li in range(n):
                    wl = 1.0 / (linear_prior_rel * scale_i[li]) ** 2
                    ne.add_prior(li, 0, (L[li, 0, 0], L[li, 1, 0]), wl)
                    ne.add_prior(li, 1, (L[li, 0, 1], L[li, 1, 1]), wl)
                X, Y = ne.solve(scale_lock=L)
                L = np.stack([np.stack([X[:, 0], X[:, 1]], 1), np.stack([Y[:, 0], Y[:, 1]], 1)], 1)
                T = np.stack([X[:, 2], Y[:, 2]], 1)
    if not georef:
        # world units = pixels of the reference image
        L = L * scl[ref]
        T = T * scl[ref]

    affines = {}
    for li, g in enumerate(used):
        Lp = L[li] / scl[li]
        affines[g] = np.column_stack([Lp, T[li] - Lp @ ctr[li]])
    gsd = float(np.median([np.sqrt(abs(np.linalg.det(affines[g][:, :2]))) for g in used]))

    # residuals in output pixels
    sq, cnt = 0.0, 0
    for p in pairs:
        a = p.pi @ affines[p.i][:, :2].T + affines[p.i][:, 2]
        b = p.pj @ affines[p.j][:, :2].T + affines[p.j][:, 2]
        sq += float(np.sum((a - b) ** 2))
        cnt += len(a)
    rms = float(np.sqrt(sq / max(cnt, 1))) / gsd
    return Alignment(used, affines, georef, gsd, rms)


# --------------------------------------------------------------------------
# Exposure compensation (Brown & Lowe style gains, per channel)
# --------------------------------------------------------------------------

def solve_gains(used: list[int], pairs: list[PairMatch], sigma_n: float = 10.0,
                sigma_g: float = 0.1) -> dict[int, np.ndarray]:
    local = {g: l for l, g in enumerate(used)}
    n = len(used)
    gains = np.ones((n, 3))
    for ch in range(3):
        A = np.zeros((n, n))
        b = np.zeros(n)
        for p in pairs:
            if p.i not in local or p.j not in local:
                continue
            i, j = local[p.i], local[p.j]
            k = len(p.ci)
            Ii, Ij = float(p.ci[:, ch].mean()), float(p.cj[:, ch].mean())
            w = k / sigma_n ** 2
            A[i, i] += w * Ii * Ii
            A[j, j] += w * Ij * Ij
            A[i, j] -= w * Ii * Ij
            A[j, i] -= w * Ii * Ij
        A[np.arange(n), np.arange(n)] += 1.0 / sigma_g ** 2
        b[:] = 1.0 / sigma_g ** 2
        gains[:, ch] = np.linalg.solve(A, b)
    gains = np.clip(gains, 0.5, 2.0)
    return {g: gains[l].astype(np.float32) for l, g in enumerate(used)}
