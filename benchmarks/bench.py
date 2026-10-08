"""Reproducible benchmark + differential-parity harness (not part of the installed package).

    python benchmarks/bench.py run  --case real --image-dir DIR --options-json report.json --out OUT
    python benchmarks/bench.py run  --case synth3d --data DIR --out OUT
    python benchmarks/bench.py run  --case synth2d --data DIR --out OUT
    python benchmarks/bench.py compare BASE_OUT CAND_OUT

`run` processes one case with fixed options and writes OUT/bench.json: end-to-end and per-stage
wall time (the stage boundaries are the library's own log lines, the same ones make_3d.py uses),
inclusive wall time per hot function (summed over threads, with call counts), peak RSS of this
process and of its children (Open3D meshing subprocess), CPU time, the environment (Python,
NumPy, compiler, extension files actually imported and their hashes) and the SHA-256 of every
input image. `compare` decodes every product of two runs and reports bitwise equality (rasters,
masks, point clouds, meshes, JSON reports without the timing fields).

Run the baseline and the candidate from *separate* environments (each importing only its own
compiled extensions); `run` records which files were imported so mixing is detectable.
"""
from __future__ import annotations

import argparse
import functools
import hashlib
import json
import logging
import os
import platform
import re
import resource
import subprocess
import sys
import threading
import time

STAGES = [   # (name, regex of the log line that starts it) - identical to run_ortho/make_3d.py
    ("Reading metadata", r"^Reading metadata"),
    ("Feature extraction", r"^Extracting features"),
    ("Image matching", r"^Matching \d+ candidate"),
    ("Aerial triangulation + bundle adjustment", r"^Aligned \d+ images"),
    ("Dense matching (depth maps)", r"^(Densify:|Dense: \d+ x)"),
    ("Depth-map fusion (dense cloud)", r"^Fusing depth maps"),
    ("DSM from dense cloud + filtering", r"^(Dense cloud -> DSM|Dense sweep done)"),
    ("True orthomosaic", r"^True orthophoto: rendering"),
    ("Rasters + DTM", r"^True orthophoto: \d+%"),
    ("Point cloud export (LAZ/PLY)", r"^(DTM:|Contours:)"),
    ("3D mesh", r"^Meshing the dense cloud"),
]
STAGES_2D = [
    ("Reading metadata", r"^Reading metadata"),
    ("Feature extraction", r"^Extracting features"),
    ("Image matching", r"^Matching \d+ candidate"),
    ("Rendering", r"^Aligned \d+ images"),
]


class StageTimer(logging.Handler):
    def __init__(self, stages):
        super().__init__(logging.DEBUG)
        self.stages, self.cur, self.t0 = stages, -1, time.perf_counter()
        self.t_cur, self.times, self.lines = self.t0, [], []

    def emit(self, record):
        msg = record.getMessage().strip()
        now = time.perf_counter()
        self.lines.append(f"{now - self.t0:9.2f} {record.name} {msg}")
        for i in range(self.cur + 1, len(self.stages)):
            if re.search(self.stages[i][1], msg):
                if self.cur >= 0:
                    self.times.append([self.stages[self.cur][0], round(now - self.t_cur, 3)])
                elif now - self.t0 > 0.05:
                    self.times.append(["Setup", round(now - self.t0, 3)])
                self.cur, self.t_cur = i, now
                return

    def finish(self):
        now = time.perf_counter()
        if self.cur >= 0:
            self.times.append([self.stages[self.cur][0], round(now - self.t_cur, 3)])
            self.cur = len(self.stages)
        return self.times


# ------------------------------------------------------------------ function profile
_prof, _plock = {}, threading.Lock()


def _wrap(owner, name, label):
    fn = getattr(owner, name, None)
    if fn is None or getattr(fn, "_bench_wrapped", False):
        return

    @functools.wraps(fn)
    def w(*a, **k):
        t = time.perf_counter()
        try:
            return fn(*a, **k)
        finally:
            dt = time.perf_counter() - t
            with _plock:
                e = _prof.setdefault(label, [0, 0.0])
                e[0] += 1
                e[1] += dt
    w._bench_wrapped = True
    setattr(owner, name, w)


def install_profile():
    """Inclusive wall time per function (summed over all threads that call it)."""
    import orthomosaic.align as align
    import orthomosaic.densify as densify
    import orthomosaic.export as export
    import orthomosaic.mvs as mvs
    import orthomosaic.ortho as ortho
    import orthomosaic.pipeline as pipeline
    import orthomosaic.reconstruct as reconstruct
    import orthomosaic.sfm as sfm
    import orthomosaic.terrain as terrain
    from orthomosaic import _ba, _dense, _mvs, color, render, features
    try:
        from orthomosaic import _fast
    except ImportError:
        _fast = None
    targets = {
        pipeline: ["extract", "match_pair", "solve_alignment", "candidate_pairs", "render"],
        align: ["solve_gains"], color: ["solve_radiometric"], features: ["load_rgb"],
        sfm: ["reconstruct", "_pair_matches", "_build_tracks", "triangulate", "bundle_adjust",
              "_extend_tracks", "_keep_observations", "_apply_priors", "_prior_cost", "rodrigues"],
        _ba: ["reduced_system", "back_substitute", "residuals", "reduced_system_mt", "back_substitute_mt",
              "residuals_mt"],
        densify: ["densify", "depth_map", "_sweep", "_consistency", "rasterize", "camera_count",
                  "cloud_footprint", "_neighbours", "_depth_range", "_depth_band", "_rays_world", "load_rgb",
                  "_fuse", "_project_neighbour", "_half", "_box", "_num_depths"],
        _dense: ["ncc_warp", "combine", "top_layer", "fill_lower_median", "nanmedian_filter", "visibility",
                 "seam_icm", "sample_view_u8", "pm_refine", "near", "sweep_hypothesis"],
        **({_fast: ["rays_grid", "ref_points", "neighbour_sample", "agreement", "box_edge_mean", "ortho_view_cos",
                    "ortho_view_fill", "ortho_resolution", "ortho_rgbw", "depth_view", "union_tracks"]}
           if _fast is not None else {}),
        _mvs: ["sample_view", "sgm", "bilateral"],
        mvs: ["dense_reconstruct", "postprocess", "_nanmedian_filter", "fill_holes", "_lower_envelope",
              "_box_mean", "grid_extent", "_bilateral_parallel"],
        ortho: ["true_orthophoto", "_global_sources", "_visibility", "_seam_labels", "_box", "load_rgb"],
        terrain: ["dtm_from_dsm", "classify_surface", "contour_lines"],
        reconstruct: ["_write_raster", "_resample_dsm", "_colorize", "_block_reduce", "sparse_block"],
        export: ["write_ply", "write_las", "las_to_laz", "poisson_mesh", "write_obj_colored", "write_glb",
                 "write_geojson_contours"],
        render: ["render"],
    }
    for owner, names in targets.items():
        for n in names:
            _wrap(owner, n, f"{owner.__name__.split('.')[-1]}.{n}")
    import numpy.linalg as la
    _wrap(sfm.np.linalg, "solve", "numpy.linalg.solve")
    del la


# ------------------------------------------------------------------ environment
def _sha(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def environment():
    import numpy as np
    import orthomosaic
    from orthomosaic import _ba, _core, _dense, _mvs
    mods = {m.__name__: dict(file=m.__file__, sha256=_sha(m.__file__)) for m in (_core, _ba, _mvs, _dense)}
    try:
        from orthomosaic import _fast
        mods[_fast.__name__] = dict(file=_fast.__file__, sha256=_sha(_fast.__file__))
    except ImportError:
        pass
    src = os.path.dirname(orthomosaic.__file__)
    try:
        rev = subprocess.run(["git", "-C", src, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
        dirty = subprocess.run(["git", "-C", src, "status", "--porcelain"], capture_output=True, text=True).stdout
    except OSError:
        rev, dirty = "", ""
    env = dict(python=sys.version, executable=sys.executable, platform=platform.platform(),
               machine=platform.machine(), cpu_count=os.cpu_count(), numpy=np.__version__,
               package_dir=src, git_rev=rev, git_dirty=bool(dirty.strip()), extensions=mods,
               env={k: os.environ.get(k) for k in ("ORTHO_DISABLE_CUDA", "ORTHO_DISABLE_MPS", "ORTHO_NATIVE",
                                                     "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                                                     "VECLIB_MAXIMUM_THREADS")})
    for name in ("PIL", "mlx", "open3d", "laspy", "lazrs", "scipy", "cupy"):
        try:
            mod = __import__(name)
            env[name] = getattr(mod, "__version__", "?")
        except Exception:
            env[name] = None
    if platform.system() == "Darwin":
        try:
            env["cpu"] = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True,
                                        text=True).stdout.strip()
            env["ram_bytes"] = int(subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True,
                                                  text=True).stdout)
        except Exception:
            pass
    import sysconfig
    env["cflags"] = sysconfig.get_config_var("CFLAGS")
    env["cc"] = sysconfig.get_config_var("CC")
    return env


# ------------------------------------------------------------------ cases
def _real_images(image_dir, suffix, subset, center=None, names_from=None):
    if names_from:                       # exactly the images a previous run used (report["source_images"])
        with open(names_from) as fh:
            names = sorted(json.load(fh)["source_images"])
    else:
        names = sorted(f for f in os.listdir(image_dir)
                       if f.lower().endswith((".jpg", ".jpeg")) and os.path.splitext(f)[0].upper().endswith(suffix))
    paths = [os.path.join(image_dir, f) for f in names]
    if subset and subset < len(paths):
        # a compact block: the `subset` images nearest the median GPS position (same overlap as the flight)
        import numpy as np
        from orthomosaic.imageio import read_frame
        fr = [read_frame(p) for p in paths]
        xy = np.array([[f.lon or 0.0, (f.lat or 0.0)] for f in fr])
        xy[:, 0] *= np.cos(np.radians(np.median(xy[:, 1])))
        c = np.median(xy, 0) if center is None else np.asarray(center)
        keep = np.sort(np.argsort(np.hypot(*(xy - c).T), kind="stable")[:subset])
        paths = [paths[k] for k in keep]
    return paths


def cmd_run(a):
    if a.threads_env:
        for k in ("OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "OMP_NUM_THREADS"):
            os.environ.setdefault(k, str(a.threads_env))
    import numpy as np
    import orthomosaic
    from orthomosaic import Options, Options3D, build_3d, build_orthomosaic
    os.makedirs(a.out, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S",
                        stream=sys.stdout if a.verbose else open(os.path.join(a.out, "log.txt"), "w"))
    if a.profile:
        install_profile()
    mode = a.mode
    if a.case == "real":
        paths = _real_images(a.image_dir, a.suffix, a.subset, names_from=a.names_from)
    else:
        tests = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tests")
        gen = os.path.join(tests, "synthetic3d.py" if a.case == "synth3d" else "synthetic.py")
        d = os.path.join(a.data, "images") if a.case == "synth3d" else a.data
        if not (os.path.isdir(d) and any(f.lower().endswith(".jpg") for f in os.listdir(d))):
            subprocess.run([sys.executable, gen, "make", a.data], check=True)
        paths = sorted(os.path.join(d, f) for f in os.listdir(d) if f.lower().endswith((".jpg", ".jpeg", ".tif", ".png")))
    opts = {}
    if a.options_json:
        with open(a.options_json) as fh:
            rep = json.load(fh)
        opts = dict(rep.get("options", rep))
    for kv in a.set:
        k, v = kv.split("=", 1)
        opts[k] = json.loads(v)
    if a.backend:
        opts["backend"] = a.backend
    if a.workers is not None:
        opts["workers"] = a.workers
    cls = Options3D if mode == "3d" else Options
    import dataclasses
    fields = {f.name for f in dataclasses.fields(cls)}
    unknown = sorted(set(opts) - fields)
    opts = {k: (tuple(v) if isinstance(v, list) else v) for k, v in opts.items() if k in fields}
    opt = cls(**opts)
    inputs = [dict(name=os.path.basename(p), sha256=_sha(p)) for p in paths]
    timer = StageTimer(STAGES if mode == "3d" else STAGES_2D)
    logging.getLogger().addHandler(timer)
    t0 = time.perf_counter()
    c0 = time.process_time()
    if a.bind_thermal:
        from orthomosaic import build_thermal_bound
        tdir = a.bind_thermal
        tpaths = sorted(os.path.join(tdir, f) for f in os.listdir(tdir) if f.lower().endswith((".jpg", ".jpeg")))
        inputs += [dict(name="thermal/" + os.path.basename(p), sha256=_sha(p)) for p in tpaths]
        rep = build_thermal_bound(paths, tpaths, a.out, opt)
    elif mode == "3d":
        rep = build_3d(paths, a.out, opt)
    else:
        rep = build_orthomosaic(paths, os.path.join(a.out, "orthomosaic.tif"), opt)
    wall = time.perf_counter() - t0
    cpu = time.process_time() - c0
    stages = timer.finish()
    ru_s, ru_c = resource.getrusage(resource.RUSAGE_SELF), resource.getrusage(resource.RUSAGE_CHILDREN)
    scale = 1 if sys.platform == "darwin" else 1024          # ru_maxrss: bytes on macOS, KiB on Linux
    out = dict(case=a.case, mode=mode, wall_seconds=round(wall, 3), cpu_seconds=round(cpu, 3),
               cpu_utilisation=round(cpu / max(wall, 1e-9), 2),
               stages=stages, peak_rss_bytes=ru_s.ru_maxrss * scale,
               peak_rss_children_bytes=ru_c.ru_maxrss * scale,
               major_faults=ru_s.ru_majflt, block_in=ru_s.ru_inblock, block_out=ru_s.ru_oublock,
               profile={k: dict(calls=v[0], seconds=round(v[1], 3)) for k, v in
                        sorted(_prof.items(), key=lambda kv: -kv[1][1])},
               options={k: (list(v) if isinstance(v, tuple) else v) for k, v in dataclasses.asdict(opt).items()},
               ignored_options=unknown, inputs=inputs, n_inputs=len(inputs),
               environment=environment(), library_version=getattr(orthomosaic, "__version__", None),
               report_seconds=rep.get("seconds"))
    with open(os.path.join(a.out, "bench.json"), "w") as fh:
        json.dump(out, fh, indent=1)
    with open(os.path.join(a.out, "stage_log.txt"), "w") as fh:
        fh.write("\n".join(timer.lines))
    print(json.dumps(dict(wall=out["wall_seconds"], stages=stages, peak_rss_gb=round(out["peak_rss_bytes"] / 1e9, 2)),
                     indent=None))


# ------------------------------------------------------------------ comparison
TIME_KEYS = {"seconds", "report_seconds", "timing", "total_seconds", "started", "finished"}


def _strip_times(o):
    if isinstance(o, dict):
        return {k: _strip_times(v) for k, v in o.items() if k not in TIME_KEYS}
    if isinstance(o, list):
        return [_strip_times(v) for v in o]
    return o


def _cmp_json(pa, pb):
    with open(pa) as fa, open(pb) as fb:
        A, B = _strip_times(json.load(fa)), _strip_times(json.load(fb))
    for d in (A, B):                                 # backend auto-selection / run paths are not results
        if isinstance(d, dict):
            d.pop("output", None)
            d.pop("preview", None)
    if A == B:
        return True, ""
    diffs = []

    def walk(x, y, p):
        if len(diffs) > 12:
            return
        if isinstance(x, dict) and isinstance(y, dict):
            for k in sorted(set(x) | set(y)):
                walk(x.get(k), y.get(k), f"{p}.{k}")
        elif isinstance(x, list) and isinstance(y, list) and len(x) == len(y):
            for i, (u, v) in enumerate(zip(x, y)):
                walk(u, v, f"{p}[{i}]")
        elif x != y:
            diffs.append(f"{p}: {str(x)[:80]} != {str(y)[:80]}")
    walk(A, B, "")
    return False, "; ".join(diffs)


def _cmp_array(a, b):
    import numpy as np
    if a.shape != b.shape or a.dtype != b.dtype:
        return False, f"shape/dtype {a.shape} {a.dtype} vs {b.shape} {b.dtype}"
    if a.dtype.kind == "f":
        na, nb = np.isnan(a), np.isnan(b)
        if not (na == nb).all():
            return False, f"NaN positions differ ({int((na != nb).sum())} cells)"
        eq = (a.view(np.uint8 if a.itemsize == 1 else f"u{a.itemsize}") ==
              b.view(np.uint8 if b.itemsize == 1 else f"u{b.itemsize}")) | (na & nb)
    else:
        eq = a == b
    if eq.all():
        return True, ""
    bad = ~eq
    msg = f"{int(bad.sum())} of {bad.size} values differ"
    if a.dtype.kind in "fiu":
        d = np.abs(a.astype(np.float64)[bad] - b.astype(np.float64)[bad])
        d = d[np.isfinite(d)]
        if d.size:
            msg += f", max |diff| {d.max():.3g}, median {np.median(d):.3g}"
    return False, msg


def _cmp_las(pa, pb):
    """Point records + header fields, ignoring the creation date (the writer stamps today's date)."""
    import numpy as np
    try:
        import laspy
    except ImportError:
        same = _sha(pa) == _sha(pb)
        return same, "" if same else "bytes differ (install laspy to compare decoded points)"
    A, B = laspy.read(pa), laspy.read(pb)
    ha, hb = A.header, B.header
    for k in ("point_count", "scales", "offsets", "mins", "maxs", "point_format"):
        va, vb = getattr(ha, k), getattr(hb, k)
        va = getattr(va, "id", va)
        vb = getattr(vb, "id", vb)
        if not np.array_equal(np.asarray(va), np.asarray(vb)):
            return False, f"header {k} differs"
    ok, msg = _cmp_array(np.asarray(A.points.array), np.asarray(B.points.array))
    return ok, msg or f"{ha.point_count} points identical"


def cmd_compare(a):
    if getattr(a, "recursive", False):
        ok = True
        for sub in sorted(set(os.listdir(a.base)) | set(os.listdir(a.cand))):
            pb, pc = os.path.join(a.base, sub), os.path.join(a.cand, sub)
            if os.path.isdir(pb) or os.path.isdir(pc):
                print(f"== {sub}")
                ok &= cmd_compare(argparse.Namespace(base=pb, cand=pc, bytes_images=a.bytes_images)) == 0
        top = cmd_compare(argparse.Namespace(base=a.base, cand=a.cand, bytes_images=a.bytes_images))
        return 0 if ok and top == 0 else 1
    import numpy as np
    from orthomosaic.geotiff import read_geotiff
    base, cand = a.base, a.cand
    files = sorted(set(os.listdir(base)) | set(os.listdir(cand)))
    skip = {"bench.json", "log.txt", "stage_log.txt"}
    ok_all, rows = True, []
    for f in files:
        if f in skip or os.path.isdir(os.path.join(base, f)) or os.path.isdir(os.path.join(cand, f)):
            continue
        pa, pb = os.path.join(base, f), os.path.join(cand, f)
        if not (os.path.exists(pa) and os.path.exists(pb)):
            rows.append((f, False, "missing in " + ("candidate" if os.path.exists(pa) else "baseline")))
            ok_all = False
            continue
        ext = os.path.splitext(f)[1].lower()
        try:
            if ext in (".tif", ".tiff"):
                A, ga = read_geotiff(pa)
                B, gb = read_geotiff(pb)
                ok, msg = _cmp_array(np.asarray(A), np.asarray(B))
                if ok and ga != gb:
                    ok, msg = False, f"georeferencing differs: {ga} vs {gb}"
                if ok:
                    bytes_same = _sha(pa) == _sha(pb)
                    msg = "decoded identical" + ("" if bytes_same else " (file bytes differ)")
            elif ext == ".json":
                ok, msg = _cmp_json(pa, pb)
            elif ext in (".png", ".jpg") and not a.bytes_images:
                from PIL import Image
                ok, msg = _cmp_array(np.asarray(Image.open(pa)), np.asarray(Image.open(pb)))
            elif ext in (".las", ".laz"):
                ok, msg = _cmp_las(pa, pb)
            elif ext in (".html", ".pdf"):
                ok, msg = True, "skipped (contains timings)"
            else:
                ok = _sha(pa) == _sha(pb)
                msg = "" if ok else f"bytes differ ({os.path.getsize(pa)} vs {os.path.getsize(pb)} bytes)"
        except Exception as e:                                  # noqa: BLE001
            ok, msg = False, f"error: {e}"
        rows.append((f, ok, msg))
        ok_all &= ok
    for f, ok, msg in rows:
        print(f"{'OK  ' if ok else 'DIFF'} {f:28s} {msg}")
    print("PARITY:", "IDENTICAL" if ok_all else "DIFFERENT")
    return 0 if ok_all else 1


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--case", choices=["real", "synth3d", "synth2d"], required=True)
    r.add_argument("--mode", choices=["3d", "2d"], default="3d")
    r.add_argument("--image-dir")
    r.add_argument("--suffix", default="_V")
    r.add_argument("--subset", type=int, default=0, help="real: use the N images nearest the median position")
    r.add_argument("--names-from", help="real: take the image names from this report.json's source_images")
    r.add_argument("--data", help="synthetic: dataset folder (created when missing)")
    r.add_argument("--options-json", help="report.json (its 'options') or a plain options dict")
    r.add_argument("--set", action="append", default=[], metavar="KEY=JSON")
    r.add_argument("--backend")
    r.add_argument("--workers", type=int)
    r.add_argument("--threads-env", type=int, default=0, help="also cap BLAS threads (env)")
    r.add_argument("--profile", action="store_true")
    r.add_argument("--bind-thermal", help="RGB-driven thermal binding with this thermal image folder")
    r.add_argument("--out", required=True)
    r.add_argument("-v", "--verbose", action="store_true")
    c = sub.add_parser("compare")
    c.add_argument("base")
    c.add_argument("cand")
    c.add_argument("--bytes-images", action="store_true", help="compare PNG/JPEG previews byte-wise")
    c.add_argument("--recursive", action="store_true", help="also compare sub-folders (thermal binding)")
    a = ap.parse_args(argv)
    if a.cmd == "run":
        cmd_run(a)
        return 0
    return cmd_compare(a)


if __name__ == "__main__":
    sys.exit(main())
