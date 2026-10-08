# pyOrthomosaic

Orthomosaics, DSM/DTM, dense point clouds and 3D meshes from drone photos — natively in Python.
No Docker, no VM, no OpenCV, GDAL or PROJ. The heavy lifting runs in Cython-compiled C that uses
every CPU core; NVIDIA (CUDA) and Apple-silicon (Metal) GPUs are used when available.

Two pipelines:

| | Command | Output |
|---|---|---|
| **2D** | `orthomosaic` | Fast planar orthomosaic (tall objects lean) |
| **2.5D / 3D** | `orthomosaic-3d` | True orthophoto on a DSM, DSM / DTM / nDSM, dense point cloud (LAZ/PLY), 3D mesh (OBJ/GLB), contours, quality report |

Package name `pyOrthomosaic`, import name `orthomosaic`. Python ≥ 3.9.

---

## Install

```bash
pip install pyOrthomosaic                       # CPU (numpy + pillow only)
pip install "pyOrthomosaic[mesh,laz]"           # + 3D mesh (open3d) + LAZ point cloud (laspy/lazrs)
pip install "pyOrthomosaic[mps]"                # + Apple-silicon GPU (MLX)
pip install "pyOrthomosaic[cuda]"               # + NVIDIA GPU (CuPy)
pip install onnxruntime                         # optional: AI sky / background masks
```

macOS: `open3d` also needs `brew install libusb`.

From source (compiles the C kernels `_core`, `_ba`, `_mvs`, `_dense`, `_fast`):

```bash
git clone <repo> && cd ODM
python3 -m venv .venv && source .venv/bin/activate
pip install numpy cython pillow setuptools
pip install -e ".[mesh,laz]"
```

Optional extras only add outputs: without `open3d` the mesh is built from the DSM, without
`laspy[lazrs]` the cloud is written as LAS.

---

## Quick start

### 3D (DSM + true orthophoto + cloud + mesh)

```bash
orthomosaic-3d /path/to/images -o recon/
```

```python
from orthomosaic import build_3d, Options3D

report = build_3d("images/", "recon/", Options3D())
print(report["dsm"]["gsd"], report["outputs"])
```

### 2D (planar orthomosaic)

```bash
orthomosaic /path/to/images -o ortho.tif
```

```python
from orthomosaic import build_orthomosaic, Options

build_orthomosaic("images/", "ortho.tif", Options(render_scale=0.5))
```

---

## Examples

**Higher-resolution DSM, contours, no DTM**

```bash
orthomosaic-3d images/ -o recon/ --dsm-resolution 0.03 --contours 1.0 --no-dtm
```

**RTK drone, ground control points**

```python
from orthomosaic import build_3d, Options3D

opt = Options3D(gps_sigma=0.05,          # RTK-fixed positions (m)
                gcp="gcps.txt",          # WebODM / Pix4D GCP list
                gcp_sigma=0.02)          # surveyed accuracy (m)
build_3d("images/", "recon/", opt)
```

**DJI electronic shutter, AI sky masks for oblique photos**

```bash
orthomosaic-3d images/ -o recon/ --rolling-shutter --rolling-shutter-readout 25 --sky-removal
```

**Fast run on an 8 GB laptop**

```python
opt = Options3D(depth_max_image=1200,     # smaller depth maps
                mesh_method="dsm",        # skip the Poisson mesh
                dtm=False, formats=("laz",),
                cache_mb=512)
build_3d("images/", "recon/", opt)
```

**Pick the photos / thermal binding**

```python
from orthomosaic import build_3d, build_thermal_bound, Options3D
import glob

rgb = sorted(glob.glob("flight/*_V.JPG"))
build_3d(rgb, "recon_rgb/", Options3D())

# thermal placed from the RGB block via a rigid rig -> co-registered thermal DSM/ortho
build_thermal_bound("rgb/", "thermal/", "recon_bound/", Options3D(backend="mps"))
```

**Quality report, DTM from an existing DSM, thermal palette**

```python
from orthomosaic import write_quality_report, dtm_from_dsm, TerrainOptions, recolor

write_quality_report("recon/report.json", "recon/quality_report.html")
dtm, ground = dtm_from_dsm(dsm_array, 0.05, TerrainOptions(max_object_size=80))
recolor("t_thermal.tif", "t_iron.tif", palette="iron")
```

---

## 3D outputs

All rasters share one north-up grid (same origin, cell size and CRS = UTM of the survey).

| File | Content |
|---|---|
| `orthophoto.tif` | True orthophoto, RGBA GeoTIFF (alpha 0 = no data) |
| `dsm.tif` | Digital Surface Model, float32 (nodata −9999) |
| `dtm.tif`, `ndsm.tif` | Bare ground, height above ground |
| `dsm_support.tif` | 1 = measured from the dense cloud, 2 = interpolated, 0 = none |
| `valid_mask.tif` | DSM validity |
| `source_image_id.tif` | Photo used per cell (index into `report["source_images"]`) |
| `coverage_count.tif` | Photos that *see* each cell (0 = hidden in all, coloured by fallback) |
| `overlap.tif` | Pix4D overlap: calibrated photos whose frame contains each DSM point |
| `dense.laz` / `dense.ply` | Dense 3D point cloud (ASPRS classes ground / building / vegetation) |
| `mesh.obj`, `mesh.glb` | Poisson mesh from the dense cloud (vertex colours) |
| `contours.geojson` | Terrain contours (`--contours`) |
| `sparse.ply`, `report.json` | Tie points, cameras, statistics, timing |
| `*_preview.png/jpg` | Quick looks |

Heights are in the take-off datum when DJI relative altitude is present; `report.json` gives the
offset to absolute altitude.

---

## How the 3D pipeline works

1. **Metadata** — EXIF/XMP: GPS, focal length, DJI gimbal/flight speed/shutter type. Optional
   masks: `<image>_mask.png` next to an image, or AI sky/background masks; masked pixels are
   ignored everywhere.
2. **Aerial triangulation** — ORB-style features at full image resolution (automatic count,
   ~1 250 per megapixel), matching against the images that overlap ≥ ~50% (automatic, 6–12 per
   image), plane + parallax verification, multi-view tracks (a match that would put two features
   of one photo in one track is refused instead of discarding the track), guided track extension
   (each tie point is looked up in every other photo that sees it), Levenberg–Marquardt
   bundle adjustment (Schur complement, Huber loss) with GPS / barometric-altitude / GCP priors,
   a **gimbal attitude prior** (DJI pitch/roll, σ 2°) that stops a nadir block from drifting into
   a common tilt, and lens self-calibration (focal, principal point, radial k1-k3, as Pix4D).
   Optional rolling-shutter correction.
3. **Dense matching** — one depth map per photo: coarse-to-fine multi-view plane sweep (NCC,
   ¼ → ½ → full of `depth_max_image`, C kernels), geometric consistency across neighbouring depth
   maps, fusion into a 3D cloud. Settings adapt to each flight instead of being hand-tuned:
   - depth search range from *all* tie points in a photo's view, always down to the lowest ground
     (tall buildings flown low no longer lose the ground);
   - consistency tolerance from the flight's SfM residual and each pair's baseline;
   - neighbours chosen for a ≥ 8° triangulation angle; 3 agreeing photos, or 2 at low overlap;
   - only flat, featureless windows are pre-rejected (low contrast is mostly still correct).
   Optional slanted-plane PatchMatch repair (`depth_patchmatch`, slow on CPU). Depth maps are
   temporary and always deleted, also when a run fails.
4. **DSM** — on its own grid sized from the point spacing (`dsm_auto_factor` × spacing), so cells
   are measured rather than interpolated; per cell the median of the cloud's top layer (roofs never averaged with walls or
   ground below); empty cells filled from nearby points in radius steps (spacing × √2ᵏ); holes
   inside the area seen by ≥ 2 photos filled from the *lower* surrounding surface (no fake
   ramps beside buildings); spike removal and edge-preserving smoothing.
5. **True orthophoto** — on its own, finer grid (native GSD, capped by `ortho_max_cells`), with
   the DSM resampled edge-aware (smooth on flat areas, nearest at height steps). Each cell is projected into photos chosen from different sides of
   the tile, tested for occlusion against the DSM, and scored (viewing angle, resolution, border
   distance, exposure). An MRF picks one photo per cell with seams where photos agree; colours
   blend only near seams. Cells hidden in every photo take the best photo's colour
   (`fill_hidden`).
6. **DTM** — progressive morphological ground filter; nDSM; contours on a 1 m DTM.
7. **Cloud + mesh** — LAZ/PLY (LAS scale from the point spacing), Poisson mesh (open3d).

Resolution is never finer than GSD − 10 % (`ignore_gsd=True` to override).

---

## Main options (`Options3D`)

| Option | CLI | Default | Meaning |
|---|---|---|---|
| `dsm_resolution` | `--dsm-resolution` | auto | DSM cell (m); auto = 0.75 × dense point spacing |
| `gps_sigma` | `--gps-sigma` | 3.0 | GPS accuracy (m); ~0.05 for RTK-fixed |
| `dense_method` | `--dense-method` | `depthmap` | `depthmap` (3D cloud) or `sweep` (legacy 2.5D, used for thermal) |
| `depth_max_image` | `--depth-max-image` | 1600 | Depth-map size (px, long side) |
| `depth_min_views` | `--depth-min-views` | 0 (auto) | Photos that must agree on a point (auto: 3, or 2 at low overlap) |
| `depth_patchmatch` | — | False | Slanted-plane PatchMatch repair (slow on CPU) |
| `attitude_sigma_deg` | — | 2.0 | Gimbal pitch/roll prior in the bundle adjustment (0 = off) |
| `n_features` | `--features` | 0 (auto) | Keypoints per image; auto ≈ 1 250 per megapixel (15 000 on 12 MP) |
| `neighbors` | `--neighbors` | 0 (auto) | Matching partners per image; auto = photos overlapping ≥ ~50%, 6–12 |
| `ortho_resolution` | — | native GSD | Orthophoto cell (m); its grid is separate from the DSM's |
| `ortho_max_cells` | — | 60 M | Orthophoto pixel cap (memory) |
| `dsm_max_fill` | `--dsm-max-fill` | −1 | Hole fill distance (m); −1 = everywhere photographed by ≥ 2 cameras |
| `dem_gapfill_steps` | `--dem-gapfill-steps` | 3 | Radius steps for filling cells from nearby points |
| `ortho_blend` | `--ortho-blend` | `seam` | `seam` (no ghosting) or `feather` |
| `ortho_views` | — | 8 | Max photos per tile, taken from a global nadir-first source map (no tile seams) |
| `fill_hidden` | — | True | Colour fully occluded cells from the best photo |
| `occlusion` | `--no-occlusion` | True | Occlusion test in the orthophoto |
| `color_balance` | `--no-color-balance` | True | Per-image gain + offset |
| `dtm` | `--no-dtm` | True | Bare-ground DTM |
| `contour_interval` | `--contours` | 0 | Contour spacing (m), 0 = off |
| `mesh_method` | — | `auto` | Poisson from the cloud (open3d) or DSM mesh |
| `mesh_max_vertices` | `--mesh-max-vertices` | 600 000 | Mesh budget |
| `formats` | `--formats` | ply,laz,obj,glb | Cloud / mesh files |
| `rolling_shutter` | `--rolling-shutter` | False | Electronic-shutter correction |
| `sky_removal`, `bg_removal` | `--sky-removal`, `--bg-removal` | False | AI masks (onnxruntime) |
| `gcp`, `gcp_sigma` | `--gcp`, `--gcp-sigma` | — , 0.05 | Ground control |
| `backend` | `--backend` | `auto` | `auto`, `cuda`, `mps`, `cpu` |
| `cache_mb` | `--cache-mb` | 1024 | Image cache (main RAM knob) |

`orthomosaic-3d --help` lists every flag.

---

## Ground control points

WebODM / Pix4D format, one row per image mark:

```
EPSG:32646
230012.5 2635008.1 12.3 2456.0 1810.5 DJI_0001_V.JPG GCP1
230012.5 2635008.1 12.3 1203.0  905.0 DJI_0002_V.JPG GCP1
230040.0 2635050.0 10.0  812.0  640.0 DJI_0007_V.JPG GCP2
```

Three or more well-spread GCPs. `report.json` gives per-GCP residuals and RMSE. GCP residuals
are not an independent accuracy check — keep separate checkpoints for that.

---

## Thermal

DJI radiometric R-JPEGs (M30T, M3T, M4T, H20T …) are mosaicked on raw sensor values with one
survey-wide scale, then coloured with a palette (`rainbow`, `iron`, `white_hot`, `black_hot`,
`arctic`, `lava`, `hot_metal`, `medical`, `green_hot`, `rainbow_hc`). Raw values are written to
`*_thermal.tif`; `python -m orthomosaic.thermal recolor` changes the palette afterwards.

---

## Performance

The Python API and CLI remain unchanged. Numerical work runs in compiled C (Cython and
handwritten kernels), CUDA C++ through CuPy, or Metal through MLX. This is not a standalone,
entirely-C application: Python still coordinates the pipeline and its file formats.

The optimizations on top of **0.8.0** preserve the existing defaults, feature counts, depth
hypotheses, resolutions, thresholds and iteration counts:

- **Dense matching:** bounded double-precision caches reuse camera-ray rotations across long
  coarse sweeps (at most 16 MiB per worker), with the original projection arithmetic.
- **Terrain percentiles:** native block selection replaces NumPy's per-cell percentile calls;
  the default lower-quartile interpolation keeps the same floating-point rounding. Unusual
  percentiles, infinities and signed zeros retain the NumPy path.
- **Box filters:** a native streaming summed-area table replaces full-raster float64
  temporaries, preserving cumulative-sum and subtraction order. Terrain generation and cloud
  classification benefit with **all three backends**; CPU dense matching also uses this filter.
- **Matching:** a handwritten C matcher uses ARM64 NEON where available, with a portable
  fallback. CUDA/Metal upload each descriptor set once per mutual match. Small GPU pairs
  reuse distances in shared-memory tiles; large pairs retain the faster streaming scan.
  Matching remains exhaustive, integer-exact, and keeps the same tie-breaking rules.

Local measurements on Apple M1 (8 GB), macOS arm64, Python 3.9.6 / NumPy 2.0.2 (five timed kernel runs after
warmup; identical output hashes):

| Operation | 0.8.0 | Optimized | Speedup |
|---|---:|---:|---:|
| Block percentile, 2000 × 2000 DSM, factor 10 | 1.751 s | 0.094 s | 18.6× |
| Box filter, same grid, radius 5 | 0.077 s | 0.0088 s | 8.7× |
| Box filter, same grid, radius 25 | 0.081 s | 0.0087 s | 9.3× |
| Mutual matching, 12 000 × 12 017 descriptors, CPU | 0.273 s | 0.247 s | 1.1× |
| Mutual matching, 512 × 529 descriptors, Metal | 1.1 ms | 0.6 ms | 1.8× |
| Mutual matching, 12 000 × 12 017 descriptors, Metal | 16.8 ms | 12.4 ms | 1.4× |

A complete 48-image synthetic **CPU** reconstruction with unchanged options produced identical
rasters, LAS/PLY clouds, OBJ/GLB meshes and report values (excluding timing). DTM generation
fell from **3.85 s to 0.72 s**. Total time was **46.3 s versus 45.9 s** in that run: depth matching
and bundle adjustment dominate, and their timing varied. These kernel improvements do **not**
establish a substantial end-to-end speedup on every flight.

CPU and Metal matching have exact regression tests, including duplicates, empty sets, tile
edges and query batches. CUDA runtime/performance and Linux/Windows hardware were not available
for local verification; wheel CI runs the CPU regression tests across its platform matrix.
CPU and GPU floating-point image operations already differed slightly in 0.8.0; exact output
parity is checked against the **same backend**, not promised across different backends.

Reproduce kernel measurements and compare complete runs from isolated baseline/candidate imports:

```bash
python benchmarks/terrain.py --package-root /path/to/baseline --out terrain-base.json
python benchmarks/terrain.py --out terrain-new.json
python benchmarks/matching.py --backend mps --out matching-new.json
python benchmarks/bench.py compare /path/to/baseline-output /path/to/new-output
```

`benchmarks/bench.py run --help` lists fixed-option survey benchmarks. See
[performance notes](benchmarks/PERFORMANCE.md) for the earlier 0.7.2 comparisons; those speedups
are historical and are not additional improvements over 0.8.0.

---

## Limitations

- 2.5D: one height per cell; walls and overhangs are not represented in the DSM/orthophoto.
- Areas hidden in every photo cannot be recovered; with `fill_hidden` they are coloured from the
  best photo (see `coverage_count.tif` = 0).
- Low-texture, shadowed or moving surfaces are interpolated (`dsm_support.tif` = 2).
- Absolute accuracy depends on GNSS/GCPs; low reprojection error is not proof of survey accuracy.

License: MIT.
