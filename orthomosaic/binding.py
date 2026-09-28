"""RGB-driven thermal binding.

A DJI dual sensor (M4T, M3T, H20T, ...) carries an RGB and a thermal camera on the same gimbal,
firing together. The RGB block has strong texture and self-calibrates well; thermal is low-texture,
low-resolution and its focal length is weakly observable, so a thermal-only reconstruction drifts.

Binding solves the RGB block by aerial triangulation, then places every thermal image using its
RGB twin's pose plus a single **rig offset** (a fixed rotation + translation from the RGB camera to
the thermal camera, shared by all frames). Thermal geometry then inherits RGB-grade accuracy and,
crucially, the thermal products land in the same coordinates as the RGB ones -- exactly what a
radiometric-overlay workflow needs.

Camera model (as elsewhere):  x_cam = R (X - C).  For a rigid pair,
    x_T = R_rel x_V + t_rel   =>   R_T = R_rel R_V ,   C_T = C_V - R_V^T R_rel^T t_rel .
The rig (R_rel, t_rel) is estimated robustly from the two independently-solved pose sets.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass

import numpy as np

from .camera import orthonormalize

log = logging.getLogger(__name__)

# DJI dual-sensor filename: PREFIX_<seq>_<suffix>.ext, e.g. DJI_20260915182922_0001_V.JPG
_DJI = re.compile(r"^(?P<stem>.+)_(?P<suffix>[A-Za-z]+)\.(?P<ext>[^.]+)$")


def pair_key(basename: str) -> str:
    """Capture key shared by an RGB/thermal pair: the filename without the sensor suffix."""
    m = _DJI.match(basename)
    return m.group("stem") if m else basename


def pair_frames(rgb_names, thermal_names) -> list:
    """Return [(rgb_basename, thermal_basename)] paired by capture (shared filename stem)."""
    rgb = {pair_key(n): n for n in rgb_names}
    pairs = []
    for tn in thermal_names:
        k = pair_key(tn)
        if k in rgb:
            pairs.append((rgb[k], tn))
    return pairs


@dataclass
class Rig:
    R_rel: np.ndarray               # (3, 3) RGB-camera -> thermal-camera rotation
    t_rel: np.ndarray               # (3,) translation in the thermal-camera frame
    rot_scatter_deg: float          # consistency of the rig rotation across frames
    trans_scatter_m: float          # consistency of the rig translation across frames
    n_pairs: int
    n_inliers: int


def _rotation_mean(Rs: np.ndarray) -> np.ndarray:
    """Chordal L2 mean rotation: average the matrices and project back to SO(3)."""
    return orthonormalize(Rs.mean(axis=0))


def _geodesic_deg(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Angle (deg) between rotation matrices A[i] and B."""
    tr = np.trace(np.einsum("nij,kj->nik", A, B), axis1=1, axis2=2)
    return np.degrees(np.arccos(np.clip((tr - 1) / 2, -1, 1)))


def estimate_rig(R_V, C_V, R_T, C_T, rot_tol_deg: float = 2.0, trans_tol_m: float = 1.0) -> Rig:
    """Robustly estimate the fixed RGB->thermal rig from paired poses (same world frame).

    R_V, C_V: RGB poses for the paired frames; R_T, C_T: thermal poses (from thermal-only SfM)."""
    R_rel_i = np.einsum("nij,nkj->nik", R_T, R_V)          # R_T R_V^T per frame
    keep = np.ones(len(R_V), bool)
    R_rel = _rotation_mean(R_rel_i)
    for _ in range(5):
        ang = _geodesic_deg(R_rel_i, R_rel)
        new = ang < max(rot_tol_deg, 3 * np.median(ang[keep]) if keep.any() else rot_tol_deg)
        if new.sum() < 3 or np.array_equal(new, keep):
            keep = new
            break
        keep = new
        R_rel = _rotation_mean(R_rel_i[keep])
    # t_rel = -R_rel R_V (C_T - C_V), robust median over inliers
    t_i = -np.einsum("ij,njk,nk->ni", R_rel, R_V[keep], (C_T - C_V)[keep])
    t_rel = np.median(t_i, axis=0)
    rot_scatter = float(np.sqrt(np.mean(_geodesic_deg(R_rel_i[keep], R_rel) ** 2)))
    trans_scatter = float(np.sqrt(np.mean(np.linalg.norm(t_i - t_rel, axis=1) ** 2)))
    return Rig(R_rel, t_rel, rot_scatter, trans_scatter, len(R_V), int(keep.sum()))


def apply_rig(rig: Rig, R_V: np.ndarray, C_V: np.ndarray):
    """Place thermal cameras from RGB poses via the rig. Returns (R_T, C_T)."""
    R_T = np.einsum("ij,njk->nik", rig.R_rel, R_V)
    C_T = C_V - np.einsum("nji,jk,k->ni", R_V, rig.R_rel.T, rig.t_rel)
    return R_T, C_T


def bind_thermal(rec_rgb, frames_rgb, rec_thermal, frames_thermal, refine: bool = True):
    """Rebind the thermal reconstruction's camera poses to the RGB block through a shared rig.

    Both reconstructions are already in the same UTM frame (shared drone GPS). Returns the thermal
    Reconstruction with poses overwritten by rig-of-RGB, restricted to frames that have an RGB
    twin, plus a dict of rig diagnostics. Thermal intrinsics are kept from the thermal SfM."""
    rgb_pose = {frames_rgb[g].name: (rec_rgb.R[k], rec_rgb.C[k]) for k, g in enumerate(rec_rgb.used)}
    th_local = {frames_thermal[g].name: k for k, g in enumerate(rec_thermal.used)}
    pairs = pair_frames(list(rgb_pose), list(th_local))
    pairs = [(r, t) for r, t in pairs if t in th_local]
    if len(pairs) < 3:
        raise RuntimeError(f"Only {len(pairs)} RGB/thermal pairs found; cannot estimate the rig")

    R_V = np.stack([rgb_pose[r][0] for r, t in pairs])
    C_V = np.stack([rgb_pose[r][1] for r, t in pairs])
    R_T = np.stack([rec_thermal.R[th_local[t]] for r, t in pairs])
    C_T = np.stack([rec_thermal.C[th_local[t]] for r, t in pairs])
    rig = estimate_rig(R_V, C_V, R_T, C_T)
    log.info("Rig: %d/%d consistent pairs, rotation scatter %.3f deg, translation scatter %.3f m, "
             "baseline %.3f m", rig.n_inliers, rig.n_pairs, rig.rot_scatter_deg, rig.trans_scatter_m,
             float(np.linalg.norm(rig.t_rel)))

    # rebuild the thermal reconstruction keeping only twinned frames, poses = rig(RGB)
    keep_local = [th_local[t] for r, t in pairs]
    R_new, C_new = apply_rig(rig, R_V, C_V)
    remap = {old: new for new, old in enumerate(keep_local)}
    rec = rec_thermal
    rec.R = np.ascontiguousarray(R_new)
    rec.C = np.ascontiguousarray(C_new)
    rec.used = [rec_thermal.used[i] for i in keep_local]
    rec.cam_group = np.ascontiguousarray(rec_thermal.cam_group[keep_local], np.int32)
    # drop observations of dropped cameras and re-index cameras
    obs_keep = np.array([c in remap for c in rec_thermal.obs_cam])
    rec.obs_cam = np.array([remap[c] for c in rec_thermal.obs_cam[obs_keep]], np.int32)
    rec.obs_pt = np.ascontiguousarray(rec_thermal.obs_pt[obs_keep], np.int32)
    rec.obs_uv = np.ascontiguousarray(rec_thermal.obs_uv[obs_keep])
    order = np.argsort(rec.obs_pt, kind="stable")
    rec.obs_cam, rec.obs_pt, rec.obs_uv = rec.obs_cam[order], rec.obs_pt[order], rec.obs_uv[order]
    # re-triangulate points with the bound poses so the sparse cloud matches the new geometry
    from .sfm import triangulate
    if len(rec.obs_pt):
        counts = np.bincount(rec.obs_pt, minlength=len(rec.X))
        rec.X, _ = triangulate(rec.R, rec.C, rec.intr_array(), rec.pp_array(), rec.cam_group,
                               rec.obs_cam, rec.obs_pt, rec.obs_uv, len(rec.X))
    diag = dict(rig_rotation_deg=[float(v) for v in _rig_euler(rig.R_rel)],
                rig_translation_m=[float(v) for v in rig.t_rel],
                rig_baseline_m=float(np.linalg.norm(rig.t_rel)),
                rotation_scatter_deg=rig.rot_scatter_deg, translation_scatter_m=rig.trans_scatter_m,
                pairs=rig.n_pairs, inliers=rig.n_inliers)
    return rec, diag


def _rig_euler(R):
    """RGB->thermal rotation as small yaw/pitch/roll degrees, for the report."""
    yaw = np.degrees(np.arctan2(R[1, 0], R[0, 0]))
    pitch = np.degrees(np.arctan2(-R[2, 0], np.hypot(R[2, 1], R[2, 2])))
    roll = np.degrees(np.arctan2(R[2, 1], R[2, 2]))
    return roll, pitch, yaw
