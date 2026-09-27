"""Synthetic drone survey with exact ground truth, for testing the pipeline.

    python tests/synthetic.py make  <out_dir>
    python tests/synthetic.py eval  <out_dir> <ortho.tif>
"""
from __future__ import annotations

import json
import math
import os
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from orthomosaic import geo  # noqa: E402

GROUND_GSD = 0.05                  # metres per ground-texture pixel
ZONE, NORTH = 46, True             # Bangladesh-ish
E0, N0 = 230000.0, 2635000.0       # UTM of ground pixel (0, 0) top-left


def make_ground(w=7000, h=5000, seed=3):
    rng = np.random.default_rng(seed)
    # multi-octave value noise
    base = np.zeros((h, w, 3), np.float32)
    for octave, amp in [(8, 60), (32, 35), (128, 20), (512, 10)]:
        g = rng.normal(0, 1, (h // (w // octave) + 2, octave + 2, 3)).astype(np.float32)
        im = Image.fromarray(((g - g.min()) / (np.ptp(g) + 1e-9) * 255).astype(np.uint8))
        base += (np.asarray(im.resize((w, h), Image.BICUBIC), np.float32) - 128) / 128 * amp
    img = Image.fromarray(np.clip(base + np.array([95, 120, 70]), 0, 255).astype(np.uint8))
    d = ImageDraw.Draw(img)
    for _ in range(120):   # fields
        cx, cy = rng.uniform(0, w), rng.uniform(0, h)
        pts = [(cx + rng.uniform(-400, 400), cy + rng.uniform(-400, 400)) for _ in range(5)]
        col = tuple(int(c) for c in rng.integers([60, 80, 30], [170, 170, 110]))
        d.polygon(pts, fill=col)
    for _ in range(25):    # roads
        x1, y1, x2, y2 = rng.uniform(0, w), rng.uniform(0, h), rng.uniform(0, w), rng.uniform(0, h)
        d.line([(x1, y1), (x2, y2)], fill=(150, 145, 140), width=int(rng.integers(20, 60)))
    for _ in range(900):   # buildings
        x, y = rng.uniform(0, w), rng.uniform(0, h)
        bw, bh = rng.uniform(30, 160), rng.uniform(30, 160)
        col = tuple(int(c) for c in rng.integers(40, 250, 3))
        d.rectangle([x, y, x + bw, y + bh], fill=col, outline=(30, 30, 30), width=3)
    for _ in range(4000):  # trees
        x, y, r = rng.uniform(0, w), rng.uniform(0, h), rng.uniform(6, 25)
        g = int(rng.integers(50, 110))
        d.ellipse([x - r, y - r, x + r, y + r], fill=(20, g, 25))
    img = img.filter(ImageFilter.GaussianBlur(0.8))
    arr = np.asarray(img, np.float32) + rng.normal(0, 6, (h, w, 3))  # fine texture
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def _dms(v):
    v = abs(v)
    d = int(v)
    m = int((v - d) * 60)
    s = (v - d - m / 60) * 3600
    return (float(d), float(m), round(s, 6))


def make(out_dir, img_w=1200, img_h=900, front=0.75, side=0.65, gps_noise=1.5, seed=7):
    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.default_rng(seed)
    ground = make_ground()
    ground.save(os.path.join(out_dir, "..", "ground_truth.png"))
    GW, GH = ground.size
    truth = {}
    margin = 0.5 * math.hypot(img_w, img_h) + 10
    step_along = img_w * (1 - front)
    step_across = img_h * (1 - side)
    ys = np.arange(margin, GH - margin, step_across)
    k = 0
    for li, cy in enumerate(ys):
        xs = np.arange(margin, GW - margin, step_along)
        heading = 0.0 if li % 2 == 0 else 180.0
        if li % 2:
            xs = xs[::-1]
        for cx in xs:
            th = math.radians(heading + rng.uniform(-4, 4))
            scale = rng.uniform(0.97, 1.03)          # altitude variation -> ground px per image px
            jx, jy = rng.normal(0, 8, 2)
            C = np.array([cx + jx, cy + jy])
            R = np.array([[math.cos(th), -math.sin(th)], [math.sin(th), math.cos(th)]]) * scale
            # image pixel (u, v) -> ground pixel: g = C + R @ ((u, v) - c)   [pixel-centre convention]
            c = np.array([(img_w - 1) / 2, (img_h - 1) / 2])
            t = C - R @ c
            # PIL AFFINE uses pixel-corner coords: in = A @ out ; convert centre->corner convention
            a, b, cc = R[0, 0], R[0, 1], t[0] + 0.5 - 0.5 * (R[0, 0] + R[0, 1])
            d_, e, f = R[1, 0], R[1, 1], t[1] + 0.5 - 0.5 * (R[1, 0] + R[1, 1])
            im = ground.transform((img_w, img_h), Image.AFFINE, (a, b, cc, d_, e, f), Image.BILINEAR)
            arr = np.asarray(im, np.float32) * rng.uniform(0.85, 1.15)       # exposure
            yy, xx = np.mgrid[0:img_h, 0:img_w]
            vign = 1 - 0.15 * (((xx - c[0]) / img_w) ** 2 + ((yy - c[1]) / img_h) ** 2) * 2
            arr = arr * vign[..., None] + rng.normal(0, 2, arr.shape)
            im = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
            # GPS of the image centre (+ noise)
            Et = E0 + C[0] * GROUND_GSD
            Nt = N0 - C[1] * GROUND_GSD
            lat, lon = geo.utm_to_latlon(Et + rng.normal(0, gps_noise), Nt + rng.normal(0, gps_noise), ZONE, NORTH)
            exif = Image.Exif()
            exif[0x8825] = {1: "N" if lat >= 0 else "S", 2: _dms(float(lat)),
                            3: "E" if lon >= 0 else "W", 4: _dms(float(lon)),
                            5: b"\x00", 6: 80.0}
            name = f"IMG_{k:04d}.JPG"
            im.save(os.path.join(out_dir, name), quality=90, exif=exif)
            # true full-px -> UTM affine
            G = np.vstack([np.column_stack([R, t]), [0, 0, 1]])
            W2U = np.array([[GROUND_GSD, 0, E0], [0, -GROUND_GSD, N0], [0, 0, 1]])
            truth[name] = (W2U @ G)[:2].tolist()
            k += 1
    with open(os.path.join(out_dir, "..", "truth.json"), "w") as fh:
        json.dump(truth, fh)
    print(f"wrote {k} images to {out_dir}")


def evaluate(out_dir, ortho):
    root = os.path.join(out_dir, "..")
    truth = json.load(open(os.path.join(root, "truth.json")))
    rep = json.load(open(os.path.splitext(ortho)[0] + "_report.json"))
    est, tru = [], []
    for name, cam in rep["cameras"].items():
        A = np.array(cam["affine"])
        T = np.array(truth[name])
        pts = np.array([[0, 0], [1199, 0], [1199, 899], [0, 899], [600, 450]], float)
        est.append(pts @ A[:, :2].T + A[:, 2] + np.array([0, 0]))
        tru.append(pts @ T[:, :2].T + T[:, 2])
    est = np.concatenate(est)
    tru = np.concatenate(tru)
    ab = np.linalg.norm(est - tru, axis=1)
    print(f"images used {rep['images_used']}/{rep['images_total']}, pairs {rep['pairs_verified']}, "
          f"match RMS {rep['match_rms_px']:.2f}px, gsd {rep['gsd']:.4f}, {rep['seconds']}s on {rep['backend']}")
    print(f"absolute geo error: RMS {np.sqrt(np.mean(ab**2)):.3f} m, max {ab.max():.3f} m")
    # relative accuracy: remove best similarity
    z, w = est[:, 0] + 1j * est[:, 1], tru[:, 0] + 1j * tru[:, 1]
    zm, wm = z.mean(), w.mean()
    al = np.sum(np.conj(z - zm) * (w - wm)) / np.sum(np.abs(z - zm) ** 2)
    rel = np.abs(al * (z - zm) + wm - w)
    print(f"relative (shape) error: RMS {np.sqrt(np.mean(rel**2)):.3f} m, max {rel.max():.3f} m "
          f"(= {np.sqrt(np.mean(rel**2))/GROUND_GSD:.2f} ground px)")
    print(f"scale error {abs(abs(al)-1)*100:.3f}%  rotation error {math.degrees(np.angle(al)):.3f} deg")

    # pixel-level check: compare the GeoTIFF with the ground truth texture
    Image.MAX_IMAGE_PIXELS = None
    ort = np.asarray(Image.open(ortho).convert("RGBA"))
    gt = np.asarray(Image.open(os.path.join(root, "ground_truth.png")).convert("RGB"), np.float32)
    gsd = rep["gsd"]
    minX, maxY = rep["bounds"][0], rep["bounds"][3]
    H, W = ort.shape[:2]
    rs = np.random.default_rng(0)
    r, c = rs.integers(0, H, 200000), rs.integers(0, W, 200000)
    m = ort[r, c, 3] > 0
    r, c = r[m], c[m]
    X = minX + (c + 0.5) * gsd
    Y = maxY - (r + 0.5) * gsd
    gx = ((X - E0) / GROUND_GSD).round().astype(int)
    gy = ((N0 - Y) / GROUND_GSD).round().astype(int)
    ok = (gx >= 0) & (gy >= 0) & (gx < gt.shape[1]) & (gy < gt.shape[0])
    diff = np.abs(ort[r[ok], c[ok], :3].astype(np.float32) - gt[gy[ok], gx[ok]])
    print(f"GeoTIFF {W}x{H}; mean abs colour diff vs ground truth at true geo-position: "
          f"{diff.mean():.1f} (0-255)  [coverage {m.mean()*100:.0f}%]")


if __name__ == "__main__":
    if sys.argv[1] == "make":
        make(sys.argv[2])
    else:
        evaluate(sys.argv[2], sys.argv[3])
