"""Branch-coverage differential runs: the same synthetic survey through the baseline and the
candidate build for each option variant, then a bit-for-bit comparison of every product.

    python benchmarks/variants.py --base-python B/bin/python --cand-python C/bin/python \
        --data DIR --work DIR [--only name,name] [--backend cpu]

DIR is a tests/synthetic3d.py survey (created if missing). Derived inputs are made next to it:
<data>_masks (every third image gets a <stem>_mask.png), <data>_thermal (radiometric R-JPEG
copies of the images, raw 16-bit data at half resolution, EXIF kept) and a GCP file built from the
synthetic truth. Results go to <work>/<variant>/{base,cand} and <work>/summary.json.
"""
from __future__ import annotations

import argparse
import io
import json
import os
import struct
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
BENCH = os.path.join(HERE, "bench.py")

VARIANTS = {   # name -> (bench.py arguments, derived dataset)
    "default": ([], None),
    "feather_blend": (["--set", 'ortho_blend="feather"'], None),
    "no_occlusion": (["--set", "occlusion=false"], None),
    "no_fill_hidden": (["--set", "fill_hidden=false"], None),
    "patchmatch": (["--set", "depth_patchmatch=true"], None),
    "sweep": (["--set", 'dense_method="sweep"'], None),
    "rolling_shutter": (["--set", "rolling_shutter=true"], None),
    "no_true_ortho": (["--set", "true_ortho=false", "--set", "color_balance=false"], None),
    "one_worker": (["--workers", "1"], None),
    "masks": ([], "masks"),
    "gcp": ([], "gcp"),
    "thermal": ([], "thermal"),
    "thermal_bound": ([], "bound"),
    "2d_feather": (["--mode", "2d"], "2d"),
    "2d_seam": (["--mode", "2d", "--set", 'blend="seam"'], "2d"),
}


def _masks(data):
    from PIL import Image
    src = os.path.join(data, "images")
    dst = data.rstrip("/") + "_masks"
    os.makedirs(dst, exist_ok=True)
    for k, f in enumerate(sorted(os.listdir(src))):
        if not f.lower().endswith(".jpg"):
            continue
        p = os.path.join(dst, f)
        if not os.path.exists(p):
            os.symlink(os.path.join(src, f), p)
        if k % 3 == 0:
            w, h = Image.open(os.path.join(src, f)).size
            m = np.full((h, w), 255, np.uint8)
            m[: h // 4, : w // 3] = 0                     # a masked corner
            Image.fromarray(m).save(os.path.join(dst, os.path.splitext(f)[0] + "_mask.png"))
    return dst


def _thermal(data):
    """R-JPEG copies: the original JPEG (EXIF kept) with raw uint16 data in APP3 segments."""
    from PIL import Image
    src = os.path.join(data, "images")
    dst = data.rstrip("/") + "_thermal"
    os.makedirs(dst, exist_ok=True)
    for f in sorted(os.listdir(src)):
        if not f.lower().endswith(".jpg") or os.path.exists(os.path.join(dst, f)):
            continue
        jpg = open(os.path.join(src, f), "rb").read()
        im = Image.open(io.BytesIO(jpg)).convert("L")
        w, h = im.size
        g = np.asarray(im.resize((w // 2, h // 2), Image.BILINEAR), np.float32)
        raw = (17000 + 3.5 * g).astype("<u2")
        data_b = raw.tobytes()
        app3 = b"".join(b"\xff\xe3" + struct.pack(">H", len(c) + 2) + c
                        for c in (data_b[i:i + 65000] for i in range(0, len(data_b), 65000)))
        with open(os.path.join(dst, f), "wb") as fh:
            fh.write(jpg[:2] + app3 + jpg[2:])
    return dst


def _gcp(data):
    """Three ground points of the synthetic scene with their marks in the images that see them
    (projected with the generator's camera model: absolute UTM centres, pixel centres at integers)."""
    path = data.rstrip("/") + "_gcp.txt"
    if os.path.exists(path):
        return path
    t = json.load(open(os.path.join(data, "truth3d.json")))
    Zg = np.load(os.path.join(data, "truth3d.npz"))["Z"]
    f, k1, k2, cell, E0, N0 = t["focal"], t["k1"], t["k2"], t["cell"], t["E0"], t["N0"]
    from PIL import Image
    W, H = Image.open(os.path.join(data, "images", t["cams"][0]["name"])).size
    lines = ["EPSG:32646"]
    for gi, (px, py) in enumerate(((1200, 900), (3400, 1500), (2200, 2800))):
        P = np.array([E0 + (px + 0.5) * cell, N0 - (py + 0.5) * cell, float(Zg[py, px])])
        for cam in t["cams"]:
            xc = np.asarray(cam["R"]) @ (P - np.asarray(cam["C"]))
            if xc[2] <= 0.1:
                continue
            n = xc[:2] / xc[2]
            r2 = float(n @ n)
            u, v = f * (1 + k1 * r2 + k2 * r2 * r2) * n + np.array([(W - 1) / 2, (H - 1) / 2])
            if 20 < u < W - 20 and 20 < v < H - 20:
                lines.append(f"{P[0]:.3f} {P[1]:.3f} {P[2]:.3f} {u:.2f} {v:.2f} {cam['name']} GCP{gi + 1}")
    open(path, "w").write("\n".join(lines) + "\n")
    return path


def _2d(data):
    d2 = data.rstrip("/").replace("synth3d", "synth2d")
    return d2


def run_variant(name, args, derived, a):
    out = os.path.join(a.work, name)
    data = a.data
    extra = list(args)
    case = ["--case", "synth3d"]
    if derived == "masks":
        data = _masks(a.data)
        case = ["--case", "real", "--image-dir", data, "--suffix", ""]
    elif derived == "thermal":
        data = _thermal(a.data)
        case = ["--case", "real", "--image-dir", data, "--suffix", ""]
    elif derived == "bound":
        th = _thermal(a.data)
        case = ["--case", "synth3d", "--bind-thermal", th]
    elif derived == "gcp":
        extra += ["--set", json.dumps(_gcp(a.data)).join(["gcp=", ""])]
    elif derived == "2d":
        data = _2d(a.data)
        case = ["--case", "synth2d"]
    res = {}
    for env, py in (("base", a.base_python), ("cand", a.cand_python)):
        o = os.path.join(out, env)
        cmd = [py, BENCH, "run", *case, "--data", data, "--out", o, *extra]
        if a.backend and "--backend" not in extra:
            cmd += ["--backend", a.backend]
        t = time.time()
        p = subprocess.run(cmd, capture_output=True, text=True)
        res[env] = dict(rc=p.returncode, seconds=round(time.time() - t, 1), err=p.stderr[-2000:] if p.returncode else "")
        if a.cool:
            time.sleep(a.cool)
    cmp = subprocess.run([a.cand_python, BENCH, "compare", os.path.join(out, "base"), os.path.join(out, "cand")]
                         + (["--recursive"] if derived == "bound" else []), capture_output=True, text=True)
    res["parity"] = "IDENTICAL" if cmp.returncode == 0 else "DIFFERENT"
    res["compare"] = cmp.stdout[-3000:]
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-python", required=True)
    ap.add_argument("--cand-python", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--work", required=True)
    ap.add_argument("--only", default="")
    ap.add_argument("--backend", default="cpu")
    ap.add_argument("--cool", type=float, default=0.0, help="seconds to idle between runs")
    a = ap.parse_args()
    os.makedirs(a.work, exist_ok=True)
    names = [n for n in VARIANTS if not a.only or n in a.only.split(",")]
    summary_path = os.path.join(a.work, "summary.json")
    summary = json.load(open(summary_path)) if os.path.exists(summary_path) else {}
    for n in names:
        args, derived = VARIANTS[n]
        r = run_variant(n, args, derived, a)
        summary[n] = r
        json.dump(summary, open(summary_path, "w"), indent=1)
        print(f"{n:18s} {r['parity']:10s} base {r['base']['seconds']:7.1f}s rc={r['base']['rc']}  "
              f"cand {r['cand']['seconds']:7.1f}s rc={r['cand']['rc']}", flush=True)
        if r["base"]["rc"] or r["cand"]["rc"]:
            print(r["base"]["err"] or r["cand"]["err"])


if __name__ == "__main__":
    main()
