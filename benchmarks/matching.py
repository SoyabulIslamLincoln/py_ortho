"""Fixed-input matching benchmark; run each build in a separate process.

python benchmarks/matching.py --backend mps --sizes 512 4000 12000 --out candidate.json
python benchmarks/matching.py --package-root /path/to/baseline --backend mps --out baseline.json

Times include descriptor upload and result download, with a warmup before repeated
measurements. Output SHA-256 covers all four int32 result arrays, not just distances.
"""
import argparse
import hashlib
import json
import platform
import sys
import time
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--package-root', default=str(Path(__file__).resolve().parents[1]))
    p.add_argument('--backend', choices=['cpu', 'mps', 'cuda'], default='cpu')
    p.add_argument('--sizes', nargs='+', type=int, default=[512, 4000, 12000])
    p.add_argument('--repeat', type=int, default=5)
    p.add_argument('--out')
    args = p.parse_args()
    if args.repeat < 1 or any(n < 1 for n in args.sizes):
        p.error('repeat and sizes must be positive')
    sys.path.insert(0, args.package_root)
    import numpy as np
    import orthomosaic
    from orthomosaic import _core
    from orthomosaic.backend import select_backend
    be = select_backend(args.backend)
    result = dict(backend=be.name, platform=platform.platform(), python=sys.version,
                  numpy=np.__version__, package=orthomosaic.__file__, extension=_core.__file__,
                  extension_sha256=hashlib.sha256(Path(_core.__file__).read_bytes()).hexdigest(), cases=[])
    for n in args.sizes:
        rng = np.random.default_rng(42)
        a = rng.integers(0, 256, (n, 32), dtype=np.uint8)
        b = rng.integers(0, 256, (n + 17, 32), dtype=np.uint8)
        be.match_mutual(a, b)
        times = []
        for _ in range(args.repeat):
            start = time.perf_counter()
            out = be.match_mutual(a, b)
            times.append(time.perf_counter() - start)
        digest = hashlib.sha256(b''.join(x.tobytes() for x in out)).hexdigest()
        row = dict(n1=len(a), n2=len(b), seconds=times, median_seconds=float(np.median(times)), sha256=digest)
        result['cases'].append(row)
        print(json.dumps(row), flush=True)
    if args.out:
        Path(args.out).write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    main()
