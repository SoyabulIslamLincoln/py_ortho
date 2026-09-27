"""Pinhole camera with radial distortion, plus pose initialisation from the 2D alignment.

Conventions
-----------
World: local East-North-Up metres (UTM minus a rounded origin; Z = height above
take-off when DJI relative altitude is available).
Camera:  x_c = R @ (X - C)   with x right, y down (image axes), z forward (viewing direction).
Pixels:  n = x_c[:2] / x_c[2];  d = 1 + k1 r^2 + k2 r^4;  uv = f * d * n + (cx, cy)
         (cx, cy) is the image centre in full-resolution pixel-centre coordinates.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

SENSOR_DIAG_35MM = 43.2666


@dataclass
class Intrinsics:
    width: int
    height: int
    f: float            # focal length in full-resolution pixels
    k1: float = 0.0
    k2: float = 0.0
    f_known: bool = True
    images: list = field(default_factory=list)

    @property
    def cx(self) -> float:
        return (self.width - 1) / 2.0

    @property
    def cy(self) -> float:
        return (self.height - 1) / 2.0


def focal_from_exif(frame) -> tuple[float, bool]:
    """Focal length in pixels from EXIF (35 mm equivalent). Returns (f, known)."""
    diag = math.hypot(frame.width, frame.height)
    if frame.focal35_mm:
        return frame.focal35_mm * diag / SENSOR_DIAG_35MM, True
    # unknown lens: assume a typical ~73 deg diagonal field of view
    return 0.7 * diag, False


def group_intrinsics(frames, used) -> tuple[list[Intrinsics], dict]:
    """One shared intrinsics block per (size, focal) camera model."""
    groups: dict = {}
    for i in used:
        fr = frames[i]
        f, known = focal_from_exif(fr)
        key = (fr.width, fr.height, round(f, 1))
        if key not in groups:
            groups[key] = Intrinsics(fr.width, fr.height, f, f_known=known)
        groups[key].images.append(i)
    intr = list(groups.values())
    cam_group = {i: g for g, it in enumerate(intr) for i in it.images}
    return intr, cam_group


# --------------------------------------------------------------------------
# rotations
# --------------------------------------------------------------------------

def rodrigues(w: np.ndarray) -> np.ndarray:
    """Axis-angle (..., 3) -> rotation matrices (..., 3, 3)."""
    w = np.asarray(w, np.float64)
    th = np.linalg.norm(w, axis=-1, keepdims=True)
    k = np.where(th > 1e-12, w / np.maximum(th, 1e-300), 0.0)
    K = np.zeros(w.shape[:-1] + (3, 3))
    K[..., 0, 1], K[..., 0, 2] = -k[..., 2], k[..., 1]
    K[..., 1, 0], K[..., 1, 2] = k[..., 2], -k[..., 0]
    K[..., 2, 0], K[..., 2, 1] = -k[..., 1], k[..., 0]
    s = np.sin(th)[..., None]
    c = np.cos(th)[..., None]
    return np.eye(3) + s * K + (1 - c) * (K @ K)


def orthonormalize(R: np.ndarray) -> np.ndarray:
    U, _, Vt = np.linalg.svd(R)
    D = np.ones(R.shape[:-2] + (3,))
    D[..., 2] = np.sign(np.linalg.det(U @ Vt))
    return (U * D[..., None, :]) @ Vt


# --------------------------------------------------------------------------
# projection / back-projection (numpy, vectorised)
# --------------------------------------------------------------------------

def project(R, C, X, f, k1, k2, cx, cy):
    """R (n,3,3), C (n,3), X (n,3), per-row intrinsics (n,) -> uv (n,2), depth (n,)."""
    xc = np.einsum("nij,nj->ni", R, X - C)
    z = xc[:, 2]
    n = xc[:, :2] / z[:, None]
    r2 = np.sum(n * n, axis=1)
    d = 1 + k1 * r2 + k2 * r2 * r2
    uv = (f * d)[:, None] * n + np.stack([cx, cy], 1)
    return uv, z


def undistort_normalized(uv, f, k1, k2, cx, cy, iters: int = 8):
    """Pixels -> undistorted normalised image coordinates (n,2)."""
    nd = (uv - np.stack([cx, cy], 1)) / f[:, None]
    n = nd.copy()
    for _ in range(iters):
        r2 = np.sum(n * n, axis=1)
        n = nd / (1 + k1 * r2 + k2 * r2 * r2)[:, None]
    return n


# --------------------------------------------------------------------------
# initial poses from the 2D alignment
# --------------------------------------------------------------------------

def poses_from_alignment(frames, alignment, intr: list[Intrinsics], cam_group: dict,
                         use_rel_alt: bool):
    """Nadir cameras: yaw and ground position from each image's affine, height from
    DJI relative altitude (if available) or f * GSD. Returns dict i -> (R, C)."""
    poses = {}
    for i in alignment.used:
        A = alignment.affines[i]
        L = A[:, :2]
        it = intr[cam_group[i]]
        centre_px = np.array([it.cx, it.cy])
        xy = L @ centre_px + A[:, 2]
        gsd = math.sqrt(abs(np.linalg.det(L)))
        U, _, Vt = np.linalg.svd(L)
        Q = U @ Vt                                   # pixel axes -> world XY directions
        ex = np.array([Q[0, 0], Q[1, 0], 0.0])
        ey = np.array([Q[0, 1], Q[1, 1], 0.0])
        ez = np.cross(ex, ey)
        if ez[2] > 0:                                # must look down; fix handedness
            ey = -ey
            ez = np.cross(ex, ey)
        R = np.stack([ex, ey, ez])
        fr = frames[i]
        if use_rel_alt and fr.rel_alt is not None:
            z = fr.rel_alt
        else:
            z = it.f * gsd
        poses[i] = (R, np.array([xy[0], xy[1], z]))
    return poses
