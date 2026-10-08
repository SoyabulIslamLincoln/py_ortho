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

Synthetic benchmark, Apple M1 8 GB, 48 photos at 12 MP, 40 M-cell DSM: **≈ 4.5 min** end to end
(depth maps 67 s, DSM 30 s, orthophoto 65 s, DTM 45 s, mesh 17 s). Expect ~7–8 min for 120
photos. On 8 GB machines close other heavy apps — swapping is the main slowdown.

---

## Limitations

- 2.5D: one height per cell; walls and overhangs are not represented in the DSM/orthophoto.
- Areas hidden in every photo cannot be recovered; with `fill_hidden` they are coloured from the
  best photo (see `coverage_count.tif` = 0).
- Low-texture, shadowed or moving surfaces are interpolated (`dsm_support.tif` = 2).
- Absolute accuracy depends on GNSS/GCPs; low reprojection error is not proof of survey accuracy.

License: MIT.
