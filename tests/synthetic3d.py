"""Synthetic 3D drone survey with exact ground truth (terrain + buildings), for testing
the 3D reconstruction.

    python tests/synthetic3d.py make <root>          # writes <root>/images, <root>/truth3d.npz
    python tests/synthetic3d.py eval <root> <outdir> # compares a reconstruction with the truth
"""
from __future__ import annotations

import json
import math
import os
import sys

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from orthomosaic import geo  # noqa: E402
from synthetic import make_ground, _dms  # noqa: E402

CELL = 0.04                         # metres per ground cell
ZONE, NORTH = 46, True
E0, N0 = 230000.0, 2635000.0        # UTM of cell (0, 0) top-left corner
IMG_W, IMG_H = 1200, 900
FOCAL = 1150.0                      # px
K1, K2 = -0.04, 0.01
ALT = 60.0                          # metres above take-off (z = 0)


def make_scene(w=4800, h=3600, seed=11):
    rng = np.random.default_rng(seed)
    tex = np.asarray(make_ground(w, h, seed=seed), np.uint8).copy()
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    # gentle terrain: +-1.5 m
    Z = (1.5 * np.sin(xx / w * 2 * np.pi * 0.7 + 0.3) * np.cos(yy / h * 2 * np.pi * 0.5)
         + 0.02 * (xx - w / 2) * CELL * 0.05).astype(np.float32)
    img = Image.fromarray(tex)
    d = ImageDraw.Draw(img)
    buildings = []
    tries = 0
    while len(buildings) < 22 and tries < 500:
        tries += 1
        bw, bh = rng.uniform(200, 550), rng.uniform(200, 450)       # 8-22 m x 8-18 m
        x0, y0 = rng.uniform(150, w - 150 - bw), rng.uniform(150, h - 150 - bh)
        if any(not (x0 > b[0] + b[2] + 60 or x0 + bw + 60 < b[0] or y0 > b[1] + b[3] + 60
                    or y0 + bh + 60 < b[1]) for b in buildings):
            continue
        height = rng.uniform(4, 15)
        gable = rng.random() < 0.4
        buildings.append((x0, y0, bw, bh, height, gable))
        xi0, yi0, xi1, yi1 = int(x0), int(y0), int(x0 + bw), int(y0 + bh)
        base = float(np.min(Z[yi0:yi1, xi0:xi1]))
        if gable:   # ridge along the longer side, 25 degree pitch
            if bw >= bh:
                dist = np.minimum(yy[yi0:yi1, xi0:xi1] - y0, y0 + bh - yy[yi0:yi1, xi0:xi1])
            else:
                dist = np.minimum(xx[yi0:yi1, xi0:xi1] - x0, x0 + bw - xx[yi0:yi1, xi0:xi1])
            roof = base + height + np.tan(np.radians(25)) * dist * CELL
        else:
            roof = np.full((yi1 - yi0, xi1 - xi0), base + height, np.float32)
        Z[yi0:yi1, xi0:xi1] = roof
        col = tuple(int(c) for c in rng.integers(60, 230, 3))
        d.rectangle([xi0, yi0, xi1 - 1, yi1 - 1], fill=col)
        for _ in range(int(bw * bh / 3000)):                          # roof details
            ax, ay = rng.uniform(xi0 + 10, xi1 - 30), rng.uniform(yi0 + 10, yi1 - 30)
            s = rng.uniform(8, 30)
            d.rectangle([ax, ay, ax + s, ay + s * rng.uniform(0.5, 1.5)],
                        fill=tuple(int(c) for c in rng.integers(20, 250, 3)))
    arr = np.asarray(img, np.float32) + rng.normal(0, 7, (h, w, 3))
    tex = np.clip(arr, 0, 255).astype(np.uint8)
    return tex, Z, buildings


def _wall_points(Z, tex):
    """Vertical wall samples where the height jumps, so occlusion is realistic."""
    h, w = Z.shape
    pts = []
    for dy, dx in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
        nb = np.full_like(Z, np.inf)
        ys = slice(max(dy, 0), h + min(dy, 0))
        yd = slice(max(-dy, 0), h + min(-dy, 0))
        xs = slice(max(dx, 0), w + min(dx, 0))
        xd = slice(max(-dx, 0), w + min(-dx, 0))
        nb[yd, xd] = Z[ys, xs]
        drop = Z - nb
        ry, rx = np.nonzero(drop > 0.3)
        for k in np.linspace(0.05, 0.95, 18):
            zz = nb[ry, rx] + k * drop[ry, rx]
            c = (tex[ry, rx].astype(np.float32) * 0.55).astype(np.uint8)
            pts.append((rx + 0.5 + 0.5 * dx, ry + 0.5 + 0.5 * dy, zz, c))
    x = np.concatenate([p[0] for p in pts])
    y = np.concatenate([p[1] for p in pts])
    z = np.concatenate([p[2] for p in pts])
    c = np.concatenate([p[3] for p in pts])
    return x, y, z, c


def render(tex, Z, walls, R, Cw):
    """Z-buffered forward splat of the textured height field into a pinhole camera.
    Cw: camera centre in cell units for x/y and metres for z."""
    h, w = Z.shape
    # footprint window (generous for tall buildings)
    half = 0.5 * math.hypot(IMG_W, IMG_H) * (Cw[2] / FOCAL) / CELL * 1.25
    x0, x1 = int(max(0, Cw[0] - half)), int(min(w, Cw[0] + half))
    y0, y1 = int(max(0, Cw[1] - half)), int(min(h, Cw[1] + half))
    yy, xx = np.mgrid[y0:y1, x0:x1]
    X = np.concatenate([(xx.ravel() + 0.5), walls[0]])
    Y = np.concatenate([(yy.ravel() + 0.5), walls[1]])
    ZZ = np.concatenate([Z[y0:y1, x0:x1].ravel(), walls[2]])
    col = np.concatenate([tex[y0:y1, x0:x1].reshape(-1, 3), walls[3]])
    # world metres: E = x*CELL, N = -y*CELL
    P = np.stack([X * CELL, -Y * CELL, ZZ], 1)
    C = np.array([Cw[0] * CELL, -Cw[1] * CELL, Cw[2]])
    xc = (P - C) @ R.T
    m = xc[:, 2] > 0.1
    xc, col = xc[m], col[m]
    n = xc[:, :2] / xc[:, 2:3]
    r2 = np.sum(n * n, 1)
    uv = FOCAL * (1 + K1 * r2 + K2 * r2 * r2)[:, None] * n + np.array([(IMG_W - 1) / 2, (IMG_H - 1) / 2])
    depth = xc[:, 2]
    out = np.zeros((IMG_H, IMG_W, 3), np.uint8)
    zbuf = np.full((IMG_H, IMG_W), np.inf)
    order = np.argsort(-depth)                       # far first; near overwrites
    uv, depth, col = uv[order], depth[order], col[order]
    for oy in (0, 1):
        for ox in (0, 1):
            u = np.floor(uv[:, 0]).astype(int) + ox
            v = np.floor(uv[:, 1]).astype(int) + oy
            ok = (u >= 0) & (v >= 0) & (u < IMG_W) & (v < IMG_H)
            u, v, dd, cc = u[ok], v[ok], depth[ok], col[ok]
            closer = dd < zbuf[v, u] + 0.05
            out[v[closer], u[closer]] = cc[closer]
            zbuf[v[closer], u[closer]] = np.minimum(zbuf[v[closer], u[closer]], dd[closer])
    return out


def _rot(yaw, pitch, roll):
    """Nadir camera with small tilts; yaw = heading of the image 'up' direction."""
    cz, sz = math.cos(yaw), math.sin(yaw)
    base = np.array([[cz, sz, 0], [sz, -cz, 0], [0, 0, -1]], float)   # x right, y down, z down
    rx = np.array([[1, 0, 0], [0, math.cos(pitch), -math.sin(pitch)], [0, math.sin(pitch), math.cos(pitch)]])
    ry = np.array([[math.cos(roll), 0, math.sin(roll)], [0, 1, 0], [-math.sin(roll), 0, math.cos(roll)]])
    return rx @ ry @ base


def make(root, gps_noise=1.5, seed=5):
    rng = np.random.default_rng(seed)
    out_dir = os.path.join(root, "images")
    os.makedirs(out_dir, exist_ok=True)
    tex, Z, buildings = make_scene()
    walls = _wall_points(Z, tex)
    h, w = Z.shape
    gsd = ALT / FOCAL
    fx, fy = IMG_W * gsd / CELL, IMG_H * gsd / CELL          # footprint in cells
    cams = []
    ys = np.arange(fy * 0.6, h - fy * 0.6, fy * (1 - 0.65))
    k = 0
    for li, cy in enumerate(ys):
        xs = np.arange(fx * 0.6, w - fx * 0.6, fx * (1 - 0.75))
        heading = 0.0 if li % 2 == 0 else math.pi
        if li % 2:
            xs = xs[::-1]
        for cx in xs:
            yaw = heading + math.radians(rng.uniform(-3, 3))
            R = _rot(yaw, math.radians(rng.normal(0, 1.0)), math.radians(rng.normal(0, 1.0)))
            Cw = np.array([cx + rng.normal(0, 10), cy + rng.normal(0, 10), ALT + rng.normal(0, 0.8)])
            img = render(tex, Z, walls, R, Cw)
            E = E0 + Cw[0] * CELL + rng.normal(0, gps_noise)
            N = N0 - Cw[1] * CELL + rng.normal(0, gps_noise)
            lat, lon = geo.utm_to_latlon(E, N, ZONE, NORTH)
            exif = Image.Exif()
            exif[0x8825] = {1: "N", 2: _dms(float(lat)), 3: "E", 4: _dms(float(lon)), 5: b"\x00",
                            6: float(Cw[2] + 20.0)}
            exif_ifd = {0xA405: int(round(FOCAL * 43.2666 / math.hypot(IMG_W, IMG_H)))}
            exif[0x8769] = exif_ifd
            rel = Cw[2] + rng.normal(0, 0.3)
            xmp = (f'<x:xmpmeta xmlns:x="adobe:ns:meta/"><rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">'
                   f'<rdf:Description xmlns:drone-dji="http://www.dji.com/drone-dji/1.0/" '
                   f'drone-dji:RelativeAltitude="+{rel:.3f}" drone-dji:AbsoluteAltitude="+{rel + 20:.3f}" '
                   f'drone-dji:GimbalPitchDegree="-90.0"/></rdf:RDF></x:xmpmeta>').encode()
            name = f"IMG_{k:04d}.JPG"
            Image.fromarray(img).save(os.path.join(out_dir, name), quality=92, exif=exif, xmp=xmp)
            Cm = np.array([E0 + Cw[0] * CELL, N0 - Cw[1] * CELL, Cw[2]])
            cams.append(dict(name=name, R=R.tolist(), C=Cm.tolist()))
            k += 1
    np.savez_compressed(os.path.join(root, "truth3d.npz"), Z=Z, tex=tex)
    with open(os.path.join(root, "truth3d.json"), "w") as fh:
        json.dump(dict(cell=CELL, E0=E0, N0=N0, focal=FOCAL, k1=K1, k2=K2, cams=cams,
                       buildings=[list(map(float, b)) for b in buildings]), fh)
    print(f"wrote {k} images to {out_dir}")


def evaluate(root, outdir):
    truth = json.load(open(os.path.join(root, "truth3d.json")))
    Zt = np.load(os.path.join(root, "truth3d.npz"))["Z"]
    rep = json.load(open(os.path.join(outdir, "report.json")))
    # cameras
    tc = {c["name"]: c for c in truth["cams"]}
    ec = rep["cameras"]
    dC = np.array([np.array(ec[n]["C"]) - np.array(tc[n]["C"]) for n in ec])
    dR = [math.degrees(math.acos(np.clip((np.trace(np.array(ec[n]["R"]) @ np.array(tc[n]["R"]).T) - 1) / 2, -1, 1)))
          for n in ec]
    print(f"cameras {len(ec)}/{len(tc)}: centre error RMS xy {np.sqrt(np.mean(dC[:, :2] ** 2)):.3f} m, "
          f"z {np.sqrt(np.mean(dC[:, 2] ** 2)):.3f} m; rotation error median {np.median(dR):.3f} deg, max {max(dR):.3f}")
    print(f"focal {rep['sfm']['focal_px']} (true {truth['focal']}), k1 {rep['sfm']['k1']} (true {truth['k1']}), "
          f"reprojection RMS {rep['sfm']['rms_px']:.2f} px, points {rep['sfm']['points']}")
    if "dsm" not in rep:
        return
    from PIL import Image as _I
    _I.MAX_IMAGE_PIXELS = None
    dsm_path = os.path.join(outdir, "dsm.tif")
    D = np.asarray(_I.open(dsm_path), np.float32)
    gsd = rep["dsm"]["gsd"]
    minX, maxY = rep["dsm"]["bounds"][0], rep["dsm"]["bounds"][3]
    H, W = D.shape
    rr, cc = np.mgrid[0:H, 0:W]
    X = minX + (cc + 0.5) * gsd
    Y = maxY - (rr + 0.5) * gsd
    gx = ((X - truth["E0"]) / truth["cell"]).astype(int)
    gy = ((truth["N0"] - Y) / truth["cell"]).astype(int)
    ok = (gx >= 0) & (gy >= 0) & (gx < Zt.shape[1]) & (gy < Zt.shape[0]) & (D > -9000)
    zt = Zt[gy[ok], gx[ok]]
    err = D[ok] - zt
    # evaluate away from walls (height discontinuities are ambiguous by +-1 cell)
    gz = np.zeros_like(Zt)
    gz[1:-1, 1:-1] = np.maximum.reduce([np.abs(Zt[1:-1, 1:-1] - Zt[:-2, 1:-1]), np.abs(Zt[1:-1, 1:-1] - Zt[2:, 1:-1]),
                                        np.abs(Zt[1:-1, 1:-1] - Zt[1:-1, :-2]), np.abs(Zt[1:-1, 1:-1] - Zt[1:-1, 2:])])
    from numpy.lib.stride_tricks import sliding_window_view  # noqa: F401
    near_wall = np.zeros_like(Zt, bool)
    edge = gz > 0.3
    k = max(1, int(round(3 * gsd / truth["cell"])))
    for dy in range(-k, k + 1, max(1, k // 2)):
        for dx in range(-k, k + 1, max(1, k // 2)):
            near_wall |= np.roll(np.roll(edge, dy, 0), dx, 1)
    flat = ~near_wall[gy[ok], gx[ok]]
    roof = zt > np.percentile(Zt, 60) + 3
    e = err[flat]
    print(f"DSM {W}x{H} @ {gsd * 100:.1f} cm, coverage {ok.mean() * 100:.0f}%: "
          f"median |err| {np.median(np.abs(e)):.3f} m, RMS {np.sqrt(np.mean(e ** 2)):.3f} m, "
          f"within 0.25 m: {np.mean(np.abs(e) < 0.25) * 100:.1f}%, bias {np.median(e):+.3f} m")
    rf = flat & roof
    if rf.any():
        print(f"   roofs only: median |err| {np.median(np.abs(err[rf])):.3f} m, bias {np.median(err[rf]):+.3f} m "
              f"({rf.sum()} cells)")


if __name__ == "__main__":
    if sys.argv[1] == "make":
        make(sys.argv[2])
    else:
        evaluate(sys.argv[2], sys.argv[3])
