"""Capture the inputs of the expensive stages from a real run and replay a stage alone.

    python benchmarks/capture.py capture DIR -- <bench.py run arguments>
    python benchmarks/capture.py replay DIR {ba,densify,ortho} [--workers N] [--out OUT] [--cprofile]

`capture` runs the pipeline once (via bench.py) and pickles the arguments of
sfm.bundle_adjust (every call), densify.densify and ortho.true_orthophoto. `replay` re-runs one
stage on those inputs, so a kernel change can be timed and checked for bit-identical output in
seconds instead of a full run. The backend object is not pickled (CPU is used on replay; the
MPS backend uses the CPU path for these stages anyway).
"""
from __future__ import annotations

import argparse
import copy
import os
import pickle
import sys
import time

import numpy as np


class _AR:
    """Picklable stand-in for pipeline.AlignResult (what the stages read)."""

    def __init__(self, ar):
        self.frames, self.workers, self.positions = ar.frames, ar.workers, ar.positions
        self.alignment = ar.alignment
        self.thermal_range = getattr(ar, "thermal_range", None)
        self.backend_name = ar.backend.name

    def with_backend(self, name="cpu"):
        from orthomosaic.backend import CPUBackend, select_backend
        self.backend = CPUBackend() if name == "cpu" else select_backend(name)
        return self


def capture(d, argv):
    os.makedirs(d, exist_ok=True)
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import bench
    from orthomosaic import densify, mvs, ortho, sfm
    n_ba = [0]
    ba0, dz0, or0, sw0 = sfm.bundle_adjust, densify.densify, ortho.true_orthophoto, mvs.dense_reconstruct

    def sw(ar, rec, gains, native_gsd, opt, biases=None):
        with open(os.path.join(d, "sweep.pkl"), "wb") as fh:
            pickle.dump((_AR(ar), rec, gains, native_gsd, opt, biases), fh)
        return sw0(ar, rec, gains, native_gsd, opt, biases=biases)

    def ba(rec, pri, huber, *a, **k):
        with open(os.path.join(d, f"ba{n_ba[0]:02d}.pkl"), "wb") as fh:
            pickle.dump((copy.deepcopy(rec), copy.deepcopy(pri), huber, a, {x: y for x, y in k.items()
                                                                           if x != "workers"}), fh)
        n_ba[0] += 1
        return ba0(rec, pri, huber, *a, **k)

    def dz(ar, rec, gains, biases, out_dir, opt):
        with open(os.path.join(d, "densify.pkl"), "wb") as fh:
            pickle.dump((_AR(ar), rec, gains, biases, opt), fh)
        return dz0(ar, rec, gains, biases, out_dir, opt)

    def tor(ar, rec, dsm, minX, maxY, gsd, gains=None, biases=None, opt=None):
        with open(os.path.join(d, "ortho.pkl"), "wb") as fh:
            pickle.dump((_AR(ar), rec, dsm, minX, maxY, gsd, gains, biases, opt), fh)
        return or0(ar, rec, dsm, minX, maxY, gsd, gains, biases, opt)

    sfm.bundle_adjust, densify.densify, ortho.true_orthophoto, mvs.dense_reconstruct = ba, dz, tor, sw
    bench.main(argv)


def replay(d, stage, workers, out, cprof, log=False, fprof=False, backend="cpu"):
    import __main__
    __main__._AR = _AR                        # pickles written when capture.py ran as __main__
    if fprof:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        import bench
        bench.install_profile()
    if log:
        import logging
        logging.basicConfig(level=logging.INFO, format="%(relativeCreated)8.0f %(message)s", stream=sys.stdout)
    from orthomosaic import densify, mvs, ortho, sfm
    prof = None
    if cprof:
        import cProfile
        prof = cProfile.Profile()
    res = {}
    t0 = time.perf_counter()
    if stage == "ba":
        files = sorted(f for f in os.listdir(d) if f.startswith("ba") and f.endswith(".pkl"))
        for f in files:
            with open(os.path.join(d, f), "rb") as fh:
                rec, pri, huber, a, k = pickle.load(fh)
            t = time.perf_counter()
            if prof:
                prof.enable()
            err = sfm.bundle_adjust(rec, pri, huber, *a, workers=workers, **k)
            if prof:
                prof.disable()
            res[f] = (rec.R, rec.C, rec.X, rec.intr_array(), err)
            print(f"  {f}: {len(rec.obs_pt)} obs, {time.perf_counter() - t:.2f}s")
    elif stage == "densify":
        with open(os.path.join(d, "densify.pkl"), "rb") as fh:
            ar, rec, gains, biases, opt = pickle.load(fh)
        ar = ar.with_backend()
        if workers:
            ar.workers = opt.workers = workers
        import tempfile
        with tempfile.TemporaryDirectory(dir=out) as td:
            if prof:
                prof.enable()
            cloud = densify.densify(ar, rec, gains, biases, td, opt)
            if prof:
                prof.disable()
        res["cloud"] = (cloud.xyz, cloud.rgb, cloud.views, cloud.cam, cloud.spacing)
    elif stage == "sweep":
        with open(os.path.join(d, "sweep.pkl"), "rb") as fh:
            ar, rec, gains, native_gsd, opt, biases = pickle.load(fh)
        ar = ar.with_backend(backend)
        if workers:
            ar.workers = opt.workers = workers
        if prof:
            prof.enable()
        dr = mvs.dense_reconstruct(ar, rec, gains, native_gsd, opt, biases=biases)
        if prof:
            prof.disable()
        res["sweep"] = (dr.Z, dr.score, dr.rgb, dr.covered)
    else:
        with open(os.path.join(d, "ortho.pkl"), "rb") as fh:
            ar, rec, dsm, minX, maxY, gsd, gains, biases, opt = pickle.load(fh)
        ar = ar.with_backend()
        if workers:
            ar.workers = workers
            opt.workers = workers
        if prof:
            prof.enable()
        rgba, src, cnt, tmp = ortho.true_orthophoto(ar, rec, dsm, minX, maxY, gsd, gains, biases, opt)
        if prof:
            prof.disable()
        res["ortho"] = (np.array(rgba), np.array(src), np.array(cnt))
        import shutil
        shutil.rmtree(tmp, ignore_errors=True)
    print(f"{stage}: {time.perf_counter() - t0:.2f}s")
    if fprof:
        for k, (n, sec) in sorted(bench._prof.items(), key=lambda kv: -kv[1][1])[:25]:
            print(f"  {k:40s} {n:7d} {sec:9.2f}s")
    if out:
        with open(os.path.join(out, f"{stage}_result.pkl"), "wb") as fh:
            pickle.dump(res, fh)
    if prof:
        import pstats
        pstats.Stats(prof).sort_stats("tottime").print_stats(25)


def same(a, b):
    """Bitwise comparison of two replay results (nested tuples / dicts of arrays)."""
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(same(a[k], b[k]) for k in a)
    if isinstance(a, (tuple, list)):
        return len(a) == len(b) and all(same(x, y) for x, y in zip(a, b))
    a, b = np.asarray(a), np.asarray(b)
    if a.shape != b.shape or a.dtype != b.dtype:
        return False
    if a.dtype.kind == "f":
        return bool((a.view(f"u{a.itemsize}") == b.view(f"u{b.itemsize}")).all())
    return bool((a == b).all())


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "capture":
        i = sys.argv.index("--")
        capture(sys.argv[2], sys.argv[i + 1:])
        return
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("replay")
    r.add_argument("dir")
    r.add_argument("stage", choices=["ba", "densify", "ortho", "sweep"])
    r.add_argument("--backend", default="cpu", help="replay backend (sweep: cpu, mps, cuda)")
    r.add_argument("--workers", type=int, default=0)
    r.add_argument("--out", default="")
    r.add_argument("--cprofile", action="store_true")
    r.add_argument("--log", action="store_true", help="print the library's log lines")
    r.add_argument("--profile", action="store_true", help="per-function wall time (bench.py wrappers)")
    c = sub.add_parser("same")
    c.add_argument("a")
    c.add_argument("b")
    a = ap.parse_args()
    if a.cmd == "replay":
        replay(a.dir, a.stage, a.workers, a.out, a.cprofile, a.log, a.profile, a.backend)
    else:
        with open(a.a, "rb") as fa, open(a.b, "rb") as fb:
            ok = same(pickle.load(fa), pickle.load(fb))
        print("IDENTICAL" if ok else "DIFFERENT")
        sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
