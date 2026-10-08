# Native-code performance work (on top of 0.7.2)

Goal: the same processing results, faster. Every change below reproduces the original arithmetic
(operation order, float32/float64 rounding points, NaN handling, tie-breaking, update order), and
is checked bit for bit against the 0.7.2 code — never with a tolerance.

## What moved to compiled code, and why

Selection was driven by measured stage times (040_v7 report: SfM 28 %, depth maps 25 %, true
orthophoto 18 %, fusion 7 %, matching 6 % of 2752 s) and by profiles of real-data runs.

| Stage | Before (0.7.2) | Now | Why it was slow |
|---|---|---|---|
| Depth maps (`densify._sweep`) | `_dense.ncc_warp` per source + `_dense.combine` per hypothesis; hypothesis maps held in a list | `_dense.sweep_hypothesis` (C loops in `_sweep.h`): all sources + winner update in one row-streaming pass; vectorised projection pass; hypotheses generated one at a time (`_fast.hyp_linear/hyp_refine`) | five full-image memory passes per source and hypothesis; branchy unvectorised projection; up to 192 maps held per worker |
| Depth-map helpers | NumPy | `_fast.rays_grid` (undistortion), `_fast.box_zero_mean`, `_fast.depth_view` (matching image) | 8-step undistortion and image preparation in NumPy temporaries |
| Image cache (densify) | private LRU, duplicate concurrent decodes; fusion reloaded every reference image serially | `render.ImageCache` (coalesced loads); next fusion reference decoded in the background | 141 decodes for 60 images |
| Fusion (`densify._fuse`) | NumPy per neighbour | `_fast.ref_points / neighbour_sample / agreement / accumulate`; neighbours projected in parallel, summed in the original order; 3x3 matrix products stay in NumPy/BLAS | NumPy temporaries, sequential neighbours |
| Bundle adjustment (`_ba.reduced_system`, `back_substitute`, `residuals`) | serial | `workers=` threads: each S/g/diag(U) entry has one owner thread that adds its terms in the serial order (`_ba_kern.h`) | single-threaded Schur assembly, ~10 % of a full run |
| SfM re-matching | `match_mutual` run again for every verified pair | the 2D matcher keeps verified pairs' mutual nearest neighbours (identical inputs/matcher) | the whole 2D matching cost paid twice (34 s of 110 s SfM on 60 images) |
| Tracks | Python union-find + per-observation Python loops | `_fast.union_tracks` + NumPy gathers | per-match Python |
| Track extension | serial over cameras | cameras in threads, results in camera order | |
| 2D matching on GPU backends | GPU match then CPU RANSAC, strictly alternating | GPU matches pair after pair while CPU threads verify | GPU idle during verification |
| True orthophoto | NumPy per view and per tile; ~1 GB of temporaries per image load | `_fast.ortho_view_cos / ortho_view_fill / ortho_resolution / ortho_rgbw`, `_fast.box_edge_mean` | GIL-bound NumPy, image preparation |
| SGM (legacy sweep, thermal) | `[d][y][x]` volume, 8 directions in sequence | cell-contiguous copy; on GPU backends the 8 directions in two parallel rounds summed in order | cache misses (D values H*W apart) |
| Point-cloud writers | full-size record arrays + `tobytes()` copies | block-wise writing, same bytes | GBs of temporaries on an 8 GB machine |

Already native before and unchanged: feature detection/description, Hamming matching, RANSAC, 2D
warp/blend (`_core`), visibility, seam ICM, top-layer DSM, hole filling, median/bilateral
filters, PatchMatch (`_dense`, `_mvs`), the CUDA (CuPy) and Metal (MLX) kernels, JPEG decoding
(Pillow), BLAS/LAPACK, Open3D meshing (subprocess, single-threaded by design).

## Exactness rules used

* Kernels that copy NumPy expressions live in `_fast.pyx`, compiled with `-ffp-contract=off` (one
  rounding per NumPy ufunc). Kernels that copy existing Cython kernels keep the same expression
  trees and the module's original flags, so the compiler forms the same fused multiply-adds.
* Where NumPy's own implementation is build specific (`einsum` uses FMAs in its SIMD loop,
  `power` can use SVML on some x86 builds), the NumPy call is kept.
* Sums whose order matters (BA normal equations, fusion accumulation, SGM directions) are always
  added in the original order; parallelism only splits independent work.
* Bit-exact parity was verified with Apple clang on macOS arm64. Builds with GCC (which contracts
  across statements by default) or MSVC were not verified on real hardware.

## Fixed

* `sfm._extend_tracks` imported SciPy, which is not a dependency: 3D reconstruction failed on a
  minimal install. SciPy's cKDTree is still used when installed (results unchanged); otherwise
  `_fast.knn_radius` gives the same radius-limited 4 nearest keypoints (exact ties between
  identical keypoint distances may be listed in a different order).

## Tests and tools

* `tests/test_native_parity.py` — every new kernel against the code it replaces, bit for bit
  (random data, NaN/inf, behind-camera points, odd sizes, ties, 1..13 threads).
* `benchmarks/bench.py run` — one case with fixed options: stage/function times, peak RSS (self
  and Open3D child), environment, extension files imported, input hashes.
  `bench.py compare A B` — decodes every product of two runs and reports bit-identity.
* `benchmarks/capture.py` — capture a real run's stage inputs once, replay one stage (BA,
  densify, ortho, sweep) for quick timing and parity.
* `benchmarks/variants.py` — option-branch coverage: baseline vs candidate on synthetic data.

Baseline and candidate must run from separate environments; run them from a directory that is
not the repository (or with `python -P`), otherwise `import orthomosaic` picks up the current
folder.

## Results

### 041 flight, RGB, 113 images (`dji_data/DJI_202609151827_041`), 2026-10-08

Exact 041_v7 options (`run_ortho/output/041_v7/rgb/report.json`), backend `auto` (Apple GPU via
MLX), MacBook Air M1 (8 cores, 8 GB, fanless), macOS 27.0.1, Python 3.12.0, NumPy 2.5.3,
Pillow 12.3.0, MLX 0.32.2, SciPy 1.18.1, Open3D 0.20.0, laspy 2.7.0 / lazrs 0.8.2, Apple clang 21,
default build flags (`-O3 -fno-math-errno`). Baseline = 0.7.2 (73cf533) in its own venv; candidate
= this tree. Runs alternated with 2-minute idle gaps; warm disk cache.

| Pair | 0.7.2 (s) | new (s) | time reduction | speedup | products |
|---|---|---|---|---|---|
| 1 | 1574.2 | 860.6 | 45.3 % | 1.83x | identical |
| 2 | 975.4 | 538.5 | 44.8 % | 1.81x | identical |
| 3 | 945.3 | 478.2 | 49.4 % | 1.98x | identical |
| median | 975.4 | 538.5 | **44.8 %** | **1.81x** | |

Pair 1 ran while the machine was hotter/busier (both builds slower); the ratio is stable.
"identical" = every raster decoded bit for bit (DSM, DTM, nDSM, orthophoto, source ids, coverage,
masks), the LAZ points, PLY / OBJ / GLB bytes, contours and report.json except timings.

Median stage times (s):

| Stage | 0.7.2 | new | x |
|---|---|---|---|
| Feature extraction | 37.9 | 37.8 | 1.00 |
| Image matching | 71.3 | 62.9 | 1.13 |
| Aerial triangulation + BA | 269.2 | 127.4 | 2.11 |
| Dense matching (depth maps) | 291.6 | 144.1 | 2.02 |
| Depth-map fusion | 75.9 | 36.3 | 2.09 |
| DSM from cloud + filtering | 15.2 | 19.7 | (noise) |
| True orthomosaic | 165.1 | 53.5 | 3.08 |
| Rasters + DTM | 10.3 | 11.9 | (noise) |
| Point cloud export | 12.0 | 9.2 | 1.30 |
| 3D mesh (Open3D) | 33.7 | 34.0 | 1.00 |

Peak RSS (main process): 0.7.2 2.5-4.4 GB, new 2.9-3.7 GB; Open3D child 1.4-2.7 GB.

The 50 % target is not reached on this flight (44.8 % median). What is left: feature extraction,
GPU matching and Open3D meshing (~135 s, unchanged), the BA Schur assembly (threads limited by
the M1's efficiency cores and per-thread linearisation), and the plane-sweep kernel itself.

Note: `run_ortho/output/041_v7` was produced on 2026-10-04 by code older than 0.7.0 and differs
from 0.7.2 itself (camera poses, grid sizes), so it is not a parity reference.

### Earlier measurements (same machine, noisier: other programs running)

* 040 flight, 60-image block, RGB, MPS: 567 s (0.7.2) vs 413 s (new, one run with outside load).
* Stage replays on that block (bit-identical): orthophoto 152 -> 47 s, fusion 50 -> 14 s.
* 040 thermal, 60 images, 1.9 cm cell: SGM 120 -> 32 s; the MLX height sweep is GPU-bound and
  unchanged (its float32 GPU results would change if the op graph changed).
* Synthetic 3D survey, CPU backend: 95 -> 71 s, identical products.

### Not measured

CUDA (no NVIDIA GPU here), Linux / Windows / Intel-mac builds, the full 300-image 040 flight, the
option-variant matrix (`benchmarks/variants.py`; started, stopped before completion).
