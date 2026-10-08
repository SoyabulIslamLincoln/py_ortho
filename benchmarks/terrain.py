"""Same-input CPU terrain-kernel benchmark (also used by CUDA/Metal reconstructions).

python benchmarks/terrain.py --package-root /path/to/baseline --out base.json
python benchmarks/terrain.py --out candidate.json
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
    p.add_argument('--size', type=int, default=2000)
    p.add_argument('--repeat', type=int, default=5)
    p.add_argument('--out')
    args = p.parse_args()
    if args.repeat < 1 or args.size < 1:
        p.error('repeat and size must be positive')
    sys.path.insert(0, args.package_root)
    import numpy as np
    from orthomosaic import terrain, _fast
    from orthomosaic.backend import _box_cumsum
    rng = np.random.default_rng(73)
    z = rng.normal(10, 3, (args.size, args.size)).astype(np.float32)
    result = dict(platform=platform.platform(), numpy=np.__version__, python=sys.version,
                  package=terrain.__file__, extension_sha256=hashlib.sha256(Path(_fast.__file__).read_bytes()).hexdigest(),
                  shape=z.shape, cases=[])
    for name, fn in [('block_percentile_factor10', lambda: terrain._downsample_low(z, 10)),
                     ('box_radius5', lambda: _box_cumsum(np, z, 5)),
                     ('box_radius25', lambda: _box_cumsum(np, z, 25))]:
        fn()
        times = []
        for _ in range(args.repeat):
            start = time.perf_counter()
            out = fn()
            times.append(time.perf_counter() - start)
        row = dict(name=name, seconds=times, median_seconds=float(np.median(times)),
                   sha256=hashlib.sha256(out.tobytes()).hexdigest())
        result['cases'].append(row)
        print(json.dumps(row), flush=True)
    if args.out:
        Path(args.out).write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    main()
