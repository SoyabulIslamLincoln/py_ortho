"""Rolling-shutter correction (ODM/OpenSfM approach).

A rolling (electronic) shutter exposes image rows one after another over the *readout time*.
For a moving drone, row y is captured at t(y) = (y / H - 0.5) * readout relative to the image
centre, when the camera centre was at C + v * t(y). A global-shutter model therefore mislocates
features by up to v * readout / 2 on the ground (10 m/s, 30 ms -> 15 cm at the top/bottom rows).

Like OpenSfM (which ODM drives with ``--rolling-shutter``), the correction is applied to the
feature observations: each observation is moved to where a global-shutter camera at the image
centre time would have seen the same 3D point,

    x_gs = x_obs + pi(R (X - C)) - pi(R (X - (C + v t(y))))

and the bundle adjustment continues on the corrected observations. The velocity v comes from the
DJI XMP flight speeds (FlightX/Y/Z = north/east/down), as in ODM; images without speed tags are
not corrected. Dense matching and the orthophoto then use the corrected (global-shutter) poses,
as in ODM.
"""
from __future__ import annotations

import logging

import numpy as np

log = logging.getLogger(__name__)

# ODM's opendm/rollingshutter.py readout database (make model -> ms); callables take the frame
RS_DATABASE = {
    'autel robotics xt701': 25, 'dji phantom vision fc200': 74, 'dji fc300s': 33, 'dji fc300c': 33,
    'dji fc300x': 33, 'dji fc330': 33, 'dji fc6310': 33,
    'dji fc7203': lambda f: 19 if f.width * f.height / 1e6 < 10 else 25,
    'dji fc2103': 32, 'dji fc3170': 27, 'dji fc3411': 32, 'dji fc220': 64,
    'hasselblad l1d-20c': lambda f: 47 if f.width * f.height / 1e6 < 17 else 56,
    'hasselblad l2d-20c': 16.6, 'dji fc3682': 23,
    'dji fc3582': lambda f: 26 if f.width * f.height / 1e6 < 48 else 60,
    'dji fc8482': lambda f: (16 if f.width * f.height / 1e6 < 12 else 21 if f.width * f.height / 1e6 < 20
                             else 43 if f.width * f.height / 1e6 < 45 else 58),
    'dji fc350': 30, 'dji mavic2-enterprise-advanced': 31, 'dji zenmuse z30': 8, 'yuneec e90': 44,
    'gopro hero4 black': 30, 'gopro hero8 black': 17, 'teracube teracube one': 32, 'fujifilm x-a5': 186,
    'fujifilm x-t2': 35, 'autel robotics xl724': 29, 'parrot anafi': 39, 'autel robotics xt705': 30,
}
DEFAULT_READOUT_MS = 30.0            # ODM's default guess
_warned: set = set()


def readout_ms(frame, override: float = 0.0) -> float:
    if override > 0:
        return float(override)
    key = f"{frame.make} {frame.model}".lower().strip()
    v = RS_DATABASE.get(key)
    if v is None:
        if key not in _warned:
            _warned.add(key)
            log.warning("Rolling-shutter readout of '%s' unknown: using ODM's default %.0f ms "
                        "(set rolling_shutter_readout to the measured value)", key, DEFAULT_READOUT_MS)
        return DEFAULT_READOUT_MS
    return float(v(frame) if callable(v) else v)


def velocities(frames, used, C) -> np.ndarray:
    """(N, 3) camera velocity in the local E, N, Up frame (m/s)."""
    V = np.zeros((len(used), 3))
    for k, i in enumerate(used):
        f = frames[i]
        if f.speed_x is not None and f.speed_y is not None:
            V[k] = (f.speed_y, f.speed_x, -(f.speed_z or 0.0))   # DJI N/E/Down -> E/N/Up
    return V


def correct_observations(rec, frames, readout_override: float = 0.0, only_electronic: bool = True) -> dict:
    """Move observations to the global-shutter position (in place). Returns statistics."""
    V = velocities(frames, rec.used, rec.C)
    ro = np.array([readout_ms(frames[i], readout_override) / 1000.0 for i in rec.used])
    if only_electronic:
        mech = np.array([(frames[i].shutter or "").lower() == "mechanical" for i in rec.used])
        ro[mech] = 0.0
    speed = np.linalg.norm(V, axis=1)
    if not np.any((speed > 0.05) & (ro > 0)):
        log.info("Rolling shutter: cameras were (near) stationary or global-shutter; nothing to correct")
        return dict(corrected=0, max_shift_px=0.0, median_speed=float(np.median(speed)))
    cam = rec.obs_cam
    it = [rec.intr[g] for g in rec.cam_group]
    H = np.array([t.height for t in it])[cam]
    f = np.array([t.f for t in it])[cam]
    k1 = np.array([t.k1 for t in it])[cam]
    k2 = np.array([t.k2 for t in it])[cam]
    cx = np.array([t.cx for t in it])[cam]
    cy = np.array([t.cy for t in it])[cam]
    dt = ((rec.obs_uv[:, 1] + 0.5) / H - 0.5) * ro[cam]
    X = rec.X[rec.obs_pt]
    R = rec.R[cam]

    def proj(Cc):
        xc = np.einsum("nij,nj->ni", R, X - Cc)
        n = xc[:, :2] / xc[:, 2:3]
        r2 = np.sum(n * n, 1)
        d = f * (1 + k1 * r2 + k2 * r2 * r2)
        return np.stack([d * n[:, 0] + cx, d * n[:, 1] + cy], 1)

    C = rec.C[cam]
    shift = proj(C) - proj(C + V[cam] * dt[:, None])
    rec.obs_uv = rec.obs_uv + shift
    mag = np.linalg.norm(shift, axis=1)
    stats = dict(corrected=int((mag > 0).sum()), max_shift_px=float(mag.max()),
                 median_shift_px=float(np.median(mag)), median_speed=float(np.median(speed)),
                 readout_ms=float(np.median(ro) * 1000))
    log.info("Rolling shutter: %d observations corrected (median %.2f px, max %.2f px, speed %.1f m/s, readout %.0f ms)",
             stats["corrected"], stats["median_shift_px"], stats["max_shift_px"], stats["median_speed"],
             stats["readout_ms"])
    return stats
