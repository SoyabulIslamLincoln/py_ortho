"""Structure from motion for nadir surveys.

Pipeline (initialised from the robust 2D alignment, so no fragile incremental SfM):
  1. re-match each verified pair and keep matches consistent with *plane + parallax*:
     the 2D alignment predicts where a ground-plane point lands; a point off the plane
     (roof, tree) may deviate only along the epipolar line.  Unlike 8-point fundamental
     matrices this does not degenerate on flat scenes.
  2. link matches into multi-view tracks (union-find)
  3. initial nadir poses from the 2D alignment; triangulate
  4. robust bundle adjustment (Cython, Schur complement, Huber) with GPS / altitude priors,
     self-calibrating focal length and radial distortion; reject outliers; repeat.
"""
from __future__ import annotations

import logging
import math
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from . import _ba
from .camera import (Intrinsics, group_intrinsics, poses_from_alignment, rodrigues,
                     undistort_normalized)

log = logging.getLogger(__name__)


@dataclass
class Reconstruction:
    used: list                      # frame indices, in camera order
    R: np.ndarray                   # (N, 3, 3) world -> camera
    C: np.ndarray                   # (N, 3) camera centres (local ENU)
    intr: list                      # list[Intrinsics]
    cam_group: np.ndarray           # (N,) int32 index into intr
    X: np.ndarray                   # (P, 3) sparse points
    color: np.ndarray               # (P, 3) uint8
    obs_cam: np.ndarray             # (M,) int32  (sorted by point)
    obs_pt: np.ndarray              # (M,) int32
    obs_uv: np.ndarray              # (M, 2) full-res pixels
    rms_px: float = 0.0
    stats: dict = field(default_factory=dict)
    pt_prior_w: Optional[np.ndarray] = None   # (P, 3) point-position prior weight (GCPs > 0)
    pt_prior_t: Optional[np.ndarray] = None   # (P, 3) surveyed target for control points
    gcp_names: Optional[list] = None          # name per GCP point (in point order), else None

    def intr_array(self):
        return np.array([[it.f, it.k1, it.k2] for it in self.intr], np.float64)

    def pp_array(self):
        return np.array([[it.cx, it.cy] for it in self.intr], np.float64)


# --------------------------------------------------------------------------
# 1. plane + parallax matching
# --------------------------------------------------------------------------

def _pair_matches(backend, fi, fj, M_ij, epipole_j, perp_thresh, max_parallax, ratio):
    """Return (idx_i, idx_j) of verified matches between two images."""
    if len(fi) < 8 or len(fj) < 8:
        return None
    i12, b12, s12, i21 = backend.match_mutual(fi.desc, fj.desc)
    q = np.arange(len(i12))
    ok = (i12 >= 0) & (b12 < ratio * s12) & (b12 <= 90)
    ok &= i21[np.clip(i12, 0, None)] == q
    a, b = q[ok], i12[ok]
    if len(a) < 8:
        return None
    pi = fi.xy[a]
    pj = fj.xy[b]
    pred = pi @ M_ij[:, :2].T + M_ij[:, 2]              # where a ground-plane point would land
    r = pj - pred
    # epipolar line in image j through pj and the epipole (homogeneous)
    ph = np.column_stack([pj, np.ones(len(pj))])
    lines = np.cross(np.broadcast_to(epipole_j, ph.shape), ph)
    nrm = np.hypot(lines[:, 0], lines[:, 1])
    predh = np.column_stack([pred, np.ones(len(pred))])
    dist = np.abs(np.sum(lines * predh, axis=1)) / np.maximum(nrm, 1e-12)
    small = np.hypot(r[:, 0], r[:, 1]) < perp_thresh      # on/near the plane: line direction ill-defined
    keep = (small | (dist < perp_thresh)) & (np.hypot(r[:, 0], r[:, 1]) < max_parallax)
    if keep.sum() < 8:
        return None
    return a[keep], b[keep]


def _build_tracks(n_feats, pair_results, min_len=2):
    """Union-find over (image, feature) nodes. Returns list of (img_idx array, feat_idx array)."""
    offs = np.concatenate([[0], np.cumsum(n_feats)])
    parent = np.arange(offs[-1], dtype=np.int64)

    def find(x):
        root = x
        while parent[root] != root:
            root = parent[root]
        while parent[x] != root:
            parent[x], x = root, parent[x]
        return root

    for (i, j), (a, b) in pair_results:
        na = offs[i] + a
        nb = offs[j] + b
        for u, v in zip(na.tolist(), nb.tolist()):
            ru, rv = find(u), find(v)
            if ru != rv:
                parent[ru] = rv
    nodes = np.unique(np.concatenate([np.concatenate([offs[i] + a, offs[j] + b])
                                      for (i, j), (a, b) in pair_results]))
    roots = np.array([find(int(u)) for u in nodes], np.int64)
    order = np.argsort(roots, kind="stable")
    nodes, roots = nodes[order], roots[order]
    img = np.searchsorted(offs, nodes, side="right") - 1
    feat = nodes - offs[img]
    splits = np.flatnonzero(np.diff(roots)) + 1
    tracks = []
    for ti, tf in zip(np.split(img, splits), np.split(feat, splits)):
        if len(ti) < min_len:
            continue
        if len(np.unique(ti)) != len(ti):     # inconsistent: two features of one image
            continue
        tracks.append((ti, tf))
    return tracks


# --------------------------------------------------------------------------
# triangulation
# --------------------------------------------------------------------------

def triangulate(R, C, intr_arr, pp_arr, cam_group, obs_cam, obs_pt, obs_uv, n_points):
    """Least-squares ray intersection (midpoint method), vectorised.
    Returns X (P,3) and the max ray-angle per point (degrees)."""
    g = cam_group[obs_cam]
    n = undistort_normalized(obs_uv, intr_arr[g, 0], intr_arr[g, 1], intr_arr[g, 2], pp_arr[g, 0], pp_arr[g, 1])
    d_cam = np.column_stack([n, np.ones(len(n))])
    d = np.einsum("nji,nj->ni", R[obs_cam], d_cam)       # R^T d
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    I = np.eye(3)
    A_obs = I[None] - d[:, :, None] * d[:, None, :]
    b_obs = np.einsum("nij,nj->ni", A_obs, C[obs_cam])
    A = np.zeros((n_points, 3, 3))
    b = np.zeros((n_points, 3))
    np.add.at(A, obs_pt, A_obs)
    np.add.at(b, obs_pt, b_obs)
    A += 1e-9 * I
    X = np.linalg.solve(A, b[..., None])[..., 0]
    # ray angle: angle between the most divergent ray and the mean ray
    mean_d = np.zeros((n_points, 3))
    np.add.at(mean_d, obs_pt, d)
    mean_d /= np.maximum(np.linalg.norm(mean_d, axis=1, keepdims=True), 1e-12)
    cosang = np.sum(d * mean_d[obs_pt], axis=1)
    min_cos = np.full(n_points, 1.0)
    np.minimum.at(min_cos, obs_pt, cosang)
    ang = np.degrees(2 * np.arccos(np.clip(min_cos, -1, 1)))
    return X, ang


# --------------------------------------------------------------------------
# bundle adjustment
# --------------------------------------------------------------------------

@dataclass
class Priors:
    C_target: np.ndarray            # (N, 3)
    C_sigma: np.ndarray             # (N, 3)  inf = no prior
    intr_target: np.ndarray         # (G, 3)
    intr_sigma: np.ndarray          # (G, 3)
    X_weight: Optional[np.ndarray] = None   # (P, 3) 1/sigma^2 point-position prior (0 = tie point)
    X_target: Optional[np.ndarray] = None   # (P, 3) surveyed coordinate for control points

    def point_arrays(self, P):
        if self.X_weight is None:
            return np.zeros((P, 3)), np.zeros((P, 3))
        return np.ascontiguousarray(self.X_weight), np.ascontiguousarray(self.X_target)


def _apply_priors(S, g, C, intr, pri: Priors, N):
    cost = 0.0
    wC = np.where(np.isfinite(pri.C_sigma), 1.0 / np.square(pri.C_sigma), 0.0)
    rC = C - pri.C_target
    idx = (6 * np.arange(N)[:, None] + 3 + np.arange(3)[None]).ravel()
    g[idx] += (wC * rC).ravel()
    S[idx, idx] += wC.ravel()
    cost += float(np.sum(wC * rC * rC))
    wI = np.where(np.isfinite(pri.intr_sigma), 1.0 / np.square(pri.intr_sigma), 0.0)
    rI = intr - pri.intr_target
    idx = (6 * N + 3 * np.arange(len(intr))[:, None] + np.arange(3)[None]).ravel()
    g[idx] += (wI * rI).ravel()
    S[idx, idx] += wI.ravel()
    cost += float(np.sum(wI * rI * rI))
    return cost


def _prior_cost(C, intr, pri: Priors, X=None):
    wC = np.where(np.isfinite(pri.C_sigma), 1.0 / np.square(pri.C_sigma), 0.0)
    wI = np.where(np.isfinite(pri.intr_sigma), 1.0 / np.square(pri.intr_sigma), 0.0)
    cost = float(np.sum(wC * (C - pri.C_target) ** 2) + np.sum(wI * (intr - pri.intr_target) ** 2))
    if pri.X_weight is not None and X is not None:
        cost += float(np.sum(pri.X_weight * (X - pri.X_target) ** 2))
    return cost


def bundle_adjust(rec: Reconstruction, pri: Priors, huber: float, max_iter: int = 30,
                  tol: float = 1e-6) -> float:
    R = np.ascontiguousarray(rec.R)
    C = np.ascontiguousarray(rec.C)
    X = np.ascontiguousarray(rec.X)
    intr = rec.intr_array()
    pp = rec.pp_array()
    cg = np.ascontiguousarray(rec.cam_group, np.int32)
    oc = np.ascontiguousarray(rec.obs_cam, np.int32)
    op = np.ascontiguousarray(rec.obs_pt, np.int32)
    uv = np.ascontiguousarray(rec.obs_uv, np.float64)
    ptr = np.searchsorted(op, np.arange(len(X) + 1)).astype(np.int64)
    N = len(R)
    nc = 6 * N + 3 * len(intr)
    if rec.pt_prior_w is not None:                     # control-point position priors live on rec
        pri.X_weight, pri.X_target = rec.pt_prior_w, rec.pt_prior_t
    pw, pt_tgt = pri.point_arrays(len(X))
    lam = 1e-3
    _, cost0 = _ba.residuals(R, C, X, intr, pp, cg, oc, op, uv, huber)
    cost = cost0 + _prior_cost(C, intr, pri, X)
    for it in range(max_iter):
        S, g, Vinv, gp, _, _ = _ba.reduced_system(R, C, X, intr, pp, cg, oc, uv, ptr, huber, lam, pw, pt_tgt)
        _apply_priors(S, g, C, intr, pri, N)
        improved = False
        for _ in range(8):
            try:
                dc = np.linalg.solve(S, -g)
            except np.linalg.LinAlgError:
                dc = np.linalg.lstsq(S, -g, rcond=None)[0]
            dp = _ba.back_substitute(R, C, X, intr, pp, cg, oc, uv, ptr, huber, Vinv, gp, dc)
            dcam = dc[:6 * N].reshape(N, 6)
            R2 = np.ascontiguousarray(rodrigues(dcam[:, :3]) @ R)
            C2 = C + dcam[:, 3:]
            I2 = intr + dc[6 * N:].reshape(-1, 3)
            X2 = X + dp
            _, c2 = _ba.residuals(R2, C2, X2, I2, pp, cg, oc, op, uv, huber)
            c2 += _prior_cost(C2, I2, pri, X2)
            if c2 < cost:
                R, C, intr, X = R2, C2, I2, np.ascontiguousarray(X2)
                rel = (cost - c2) / max(cost, 1e-12)
                cost = c2
                lam = max(lam / 3, 1e-9)
                improved = True
                break
            # reject: increase damping and rebuild the damped system
            lam *= 10
            S, g, Vinv, gp, _, _ = _ba.reduced_system(R, C, X, intr, pp, cg, oc, uv, ptr, huber, lam, pw, pt_tgt)
            _apply_priors(S, g, C, intr, pri, N)
        if not improved or rel < tol:
            break
    rec.R, rec.C, rec.X = R, C, X
    for k, it in enumerate(rec.intr):
        it.f, it.k1, it.k2 = (float(v) for v in intr[k])
    err, _ = _ba.residuals(R, C, X, intr, pp, cg, oc, op, uv, huber)
    return err


def _keep_observations(rec: Reconstruction, keep_obs: np.ndarray, min_views: int = 2):
    """Drop observations, then points with too few views; re-index and sort by point.
    Control points (with a position prior) and their marks are always kept."""
    is_gcp = rec.pt_prior_w is not None and np.any(rec.pt_prior_w > 0, axis=1)
    keep_obs = keep_obs.copy()
    if rec.pt_prior_w is not None:
        keep_obs |= is_gcp[rec.obs_pt]                       # never drop a control-point mark
    oc, op, uv = rec.obs_cam[keep_obs], rec.obs_pt[keep_obs], rec.obs_uv[keep_obs]
    counts = np.bincount(op, minlength=len(rec.X))
    good_pt = counts >= min_views
    if rec.pt_prior_w is not None:
        good_pt |= is_gcp                                    # keep control points even with one mark
    m = good_pt[op]
    oc, op, uv = oc[m], op[m], uv[m]
    remap = np.cumsum(good_pt) - 1
    op = remap[op].astype(np.int32)
    order = np.argsort(op, kind="stable")
    rec.obs_cam, rec.obs_pt, rec.obs_uv = oc[order].astype(np.int32), op[order], uv[order]
    rec.X = rec.X[good_pt]
    rec.color = rec.color[good_pt]
    if rec.pt_prior_w is not None:
        rec.pt_prior_w = rec.pt_prior_w[good_pt]
        rec.pt_prior_t = rec.pt_prior_t[good_pt]


# --------------------------------------------------------------------------
# driver
# --------------------------------------------------------------------------

def _add_gcps(rec: Reconstruction, frames, used, gcps: dict, gcp_sigma: float):
    """Append ground control points as prior-constrained 3D points with their image marks."""
    name_to_local = {frames[g].name: k for k, g in enumerate(used)}
    P = len(rec.X)
    new_X, new_col, add_cam, add_pt, add_uv, names = [], [], [], [], [], []
    w = 1.0 / max(gcp_sigma, 1e-6) ** 2
    pw_extra, pt_extra = [], []
    for gname, (world, marks) in gcps.items():
        local_marks = [(name_to_local[img], uv) for img, uv in marks.items() if img in name_to_local]
        if not local_marks:
            log.warning("GCP %s: none of its marked images are in the reconstruction; skipped", gname)
            continue
        pt = P + len(new_X)
        new_X.append(np.asarray(world, np.float64))
        new_col.append([255, 0, 255])
        names.append(gname)
        pw_extra.append([w, w, w])
        pt_extra.append(np.asarray(world, np.float64))
        for cam, (u, v) in local_marks:
            add_cam.append(cam)
            add_pt.append(pt)
            add_uv.append([u, v])
    if not new_X:
        log.warning("No usable GCPs (no marks matched the reconstruction images)")
        return
    rec.X = np.vstack([rec.X, np.array(new_X)])
    rec.color = np.vstack([rec.color, np.array(new_col, np.uint8)])
    rec.obs_cam = np.concatenate([rec.obs_cam, np.array(add_cam, np.int32)])
    rec.obs_pt = np.concatenate([rec.obs_pt, np.array(add_pt, np.int32)])
    rec.obs_uv = np.vstack([rec.obs_uv, np.array(add_uv, np.float64)])
    order = np.argsort(rec.obs_pt, kind="stable")     # keep observations sorted by point
    rec.obs_cam, rec.obs_pt, rec.obs_uv = rec.obs_cam[order], rec.obs_pt[order], rec.obs_uv[order]
    rec.pt_prior_w = np.vstack([np.zeros((P, 3)), np.array(pw_extra)])
    rec.pt_prior_t = np.vstack([np.zeros((P, 3)), np.array(pt_extra)])
    rec.gcp_names = names
    log.info("Added %d ground control point(s) with %d mark(s) (sigma %.3f m)",
             len(new_X), len(add_cam), gcp_sigma)


def gcp_report(rec: Reconstruction, origin3) -> dict:
    """Per-GCP world error (final point vs surveyed) and reprojection RMS of its marks."""
    if rec.pt_prior_w is None or rec.gcp_names is None:
        return {}
    is_gcp = np.any(rec.pt_prior_w > 0, axis=1)
    gcp_pts = np.nonzero(is_gcp)[0]
    err_px, _ = _ba.residuals(np.ascontiguousarray(rec.R), np.ascontiguousarray(rec.C),
                              np.ascontiguousarray(rec.X), rec.intr_array(), rec.pp_array(),
                              np.ascontiguousarray(rec.cam_group, np.int32),
                              np.ascontiguousarray(rec.obs_cam, np.int32),
                              np.ascontiguousarray(rec.obs_pt, np.int32),
                              np.ascontiguousarray(rec.obs_uv), 1e9)
    per = {}
    for local_idx, name in zip(gcp_pts, rec.gcp_names):
        m = rec.obs_pt == local_idx
        dxyz = rec.X[local_idx] - rec.pt_prior_t[local_idx]
        per[name] = dict(world_error_m=[float(v) for v in dxyz],
                         world_error_norm_m=float(np.linalg.norm(dxyz)),
                         n_marks=int(m.sum()),
                         reproj_rms_px=float(np.sqrt(np.mean(err_px[m] ** 2))) if m.any() else None)
    errs = np.array([v["world_error_norm_m"] for v in per.values()])
    horiz = np.array([np.hypot(*v["world_error_m"][:2]) for v in per.values()])
    vert = np.array([abs(v["world_error_m"][2]) for v in per.values()])
    summary = dict(count=len(per), rmse_3d_m=float(np.sqrt(np.mean(errs ** 2))),
                   rmse_horizontal_m=float(np.sqrt(np.mean(horiz ** 2))),
                   rmse_vertical_m=float(np.sqrt(np.mean(vert ** 2))))
    return dict(per_gcp=per, summary=summary)


def reconstruct(ar, gps_sigma: float = 3.0, alt_sigma: float = 0.5, ratio: float = 0.85,
                min_track: int = 2, refine_focal: Optional[bool] = None,
                rolling_shutter: bool = False, rolling_shutter_readout: float = 0.0,
                gcps: Optional[dict] = None, gcp_sigma: float = 0.05) -> Reconstruction:
    """Sparse reconstruction from an AlignResult (see pipeline.align_images).

    gcps: optional {name: (world_local (3,), {image_basename: (u, v)})} control points
    (see gcp.to_local). They anchor the block to the survey coordinate system."""
    t0 = time.time()
    frames, feats, al = ar.frames, ar.feats, ar.alignment
    used = list(al.used)
    local = {g: k for k, g in enumerate(used)}
    intr, cam_group_map = group_intrinsics(frames, used)
    has_rel_alt = all(frames[i].rel_alt is not None for i in used)
    poses = poses_from_alignment(frames, al, intr, cam_group_map, use_rel_alt=has_rel_alt)
    N = len(used)
    R = np.stack([poses[i][0] for i in used])
    C = np.stack([poses[i][1] for i in used])
    cam_group = np.array([cam_group_map[i] for i in used], np.int32)

    # ---- 1. plane + parallax matching on the verified pairs
    pair_list = sorted({(p.i, p.j) for p in ar.pairs if p.i in local and p.j in local})
    work = ar.work_scale
    diag = float(np.median([math.hypot(frames[i].width, frames[i].height) for i in used]))
    perp = max(6.0 / work, 0.004 * diag)
    max_par = 0.35 * diag

    def job(ij):
        i, j = ij
        Pi = np.vstack([al.affines[i], [0, 0, 1]])
        Pj = np.vstack([al.affines[j], [0, 0, 1]])
        M = (np.linalg.inv(Pj) @ Pi)[:2]
        li, lj = local[i], local[j]
        it = intr[cam_group[lj]]
        K = np.array([[it.f, 0, it.cx], [0, it.f, it.cy], [0, 0, 1]])
        e = K @ (R[lj] @ (C[li] - C[lj]))
        res = _pair_matches(ar.backend, feats[i], feats[j], M, e, perp, max_par, ratio)
        return (i, j), res

    t = time.time()
    if ar.backend.parallel_blocks:
        with ThreadPoolExecutor(ar.workers) as ex:
            results = [r for r in ex.map(job, pair_list) if r[1] is not None]
    else:
        results = [r for r in map(job, pair_list) if r[1] is not None]
    n_matches = sum(len(r[1][0]) for r in results)
    log.info("3D matching: %d pairs, %d verified matches in %.1fs", len(results), n_matches, time.time() - t)

    # ---- 2. tracks
    t = time.time()
    tracks = _build_tracks([len(f) for f in feats], results, min_track)
    obs_cam = np.concatenate([np.array([local[i] for i in ti], np.int32) for ti, _ in tracks])
    obs_pt = np.concatenate([np.full(len(ti), k, np.int32) for k, (ti, _) in enumerate(tracks)])
    obs_uv = np.concatenate([np.stack([feats[i].xy[f] for i, f in zip(ti, tf)]) for ti, tf in tracks])
    obs_col = np.concatenate([np.stack([feats[i].color[f] for i, f in zip(ti, tf)]) for ti, tf in tracks])
    P = len(tracks)
    color = np.zeros((P, 3))
    np.add.at(color, obs_pt, obs_col)
    color = np.clip(color / np.bincount(obs_pt, minlength=P)[:, None], 0, 255).astype(np.uint8)
    log.info("  %d tracks (%d observations, mean length %.1f) in %.1fs",
             P, len(obs_pt), len(obs_pt) / max(P, 1), time.time() - t)

    rec = Reconstruction(used, R, C, intr, cam_group, np.zeros((P, 3)), color,
                         obs_cam, obs_pt, obs_uv)

    # ---- 3. triangulate + coarse filtering (initial poses ignore tilt, so be generous)
    X, ang = triangulate(R, C, rec.intr_array(), rec.pp_array(), cam_group, obs_cam, obs_pt, obs_uv, P)
    rec.X = X
    err, _ = _ba.residuals(R, C, X, rec.intr_array(), rec.pp_array(), cam_group, obs_cam, obs_pt,
                           obs_uv, 1e9)
    keep = (err < 0.02 * diag) & (ang[obs_pt] > 1.0)
    _keep_observations(rec, keep)

    # ---- 3b. ground control points (surveyed anchors), if any
    if gcps:
        _add_gcps(rec, frames, used, gcps, gcp_sigma)

    # ---- 4. priors
    pos = ar.positions
    C_t = rec.C.copy()
    C_s = np.full((N, 3), np.inf)
    if pos is not None and al.georeferenced:
        for k, i in enumerate(used):
            if np.all(np.isfinite(pos[i])):
                C_t[k, :2] = pos[i]
                C_s[k, :2] = gps_sigma
    else:  # fix the gauge softly around the 2D solution
        H = float(np.median(rec.C[:, 2]))
        C_s[:, :2] = 0.05 * H
    if has_rel_alt:
        C_s[:, 2] = alt_sigma
    else:
        C_s[:, 2] = 0.2 * float(np.median(np.abs(rec.C[:, 2])))
    I_t = rec.intr_array()
    I_s = np.full_like(I_t, np.inf)
    # Nadir imagery has an exact ambiguity: scaling the focal length and the scene depth below
    # the cameras together leaves every image unchanged (camera heights/altitude priors do not
    # break it; only altitude *variation* between images does, weakly). So by default the focal
    # length is held at the EXIF value; distortion is still self-calibrated.
    if refine_focal is None:
        alt = rec.C[:, 2]
        # GCPs fix absolute scale, so focal length becomes observable and can be self-calibrated
        refine_focal = bool(gcps) or bool(has_rel_alt and np.ptp(alt) > 0.25 * np.median(np.abs(alt)))
    for k, it in enumerate(rec.intr):
        if not it.f_known:
            I_s[k, 0] = 0.2 * it.f
            log.warning("No focal length in EXIF: heights will be scaled by the focal-length error")
        else:
            I_s[k, 0] = 0.02 * it.f if refine_focal else 1e-4 * it.f
        I_s[k, 1] = 0.3
        I_s[k, 2] = 0.3
    pri = Priors(C_t, C_s, I_t, I_s)

    # ---- 5. robust BA rounds with outlier rejection
    px = 1.0 / work                           # one feature-detection pixel in full-res pixels
    for rnd, (huber, thr) in enumerate([(8 * px, 30 * px), (3 * px, 8 * px), (2 * px, 4 * px)]):
        t = time.time()
        err = bundle_adjust(rec, pri, huber)
        keep = err < thr
        log.info("  BA round %d: %d points, %d obs, median err %.2f px, dropping %d obs > %.1f px (%.1fs)",
                 rnd + 1, len(rec.X), len(err), float(np.median(err)), int((~keep).sum()), thr, time.time() - t)
        _keep_observations(rec, keep)
        if rnd == 0 and rolling_shutter:
            # poses and points are now reliable: move observations to the global-shutter
            # position (OpenSfM/ODM rolling-shutter correction), later rounds refine on them
            from .rollingshutter import correct_observations
            rs_stats = correct_observations(rec, ar.frames, rolling_shutter_readout)

    err = bundle_adjust(rec, pri, 2 * px)
    rec.rms_px = float(np.sqrt(np.mean(np.square(err))))
    rec.stats = dict(points=int(len(rec.X)), observations=int(len(err)), pairs=len(results),
                     rms_px=rec.rms_px, focal_px=[it.f for it in rec.intr],
                     k1=[it.k1 for it in rec.intr], k2=[it.k2 for it in rec.intr],
                     z_datum="take-off (DJI relative altitude)" if has_rel_alt else "mean ground",
                     mean_track_length=float(len(rec.obs_pt) / max(len(rec.X), 1)))
    if rolling_shutter:
        rec.stats["rolling_shutter"] = rs_stats
    if gcps:
        gr = gcp_report(rec, None)
        rec.stats["gcp"] = gr
        if gr:
            s = gr["summary"]
            log.info("GCP check: %d control points, 3D RMSE %.3f m (horiz %.3f, vert %.3f)",
                     s["count"], s["rmse_3d_m"], s["rmse_horizontal_m"], s["rmse_vertical_m"])
    log.info("SfM done: %d cameras, %d points, reprojection RMS %.2f px, focal %s in %.1fs",
             N, len(rec.X), rec.rms_px, ", ".join(f"{it.f:.1f}" for it in rec.intr), time.time() - t0)
    return rec
