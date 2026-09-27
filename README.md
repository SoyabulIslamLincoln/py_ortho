# orthomosaic

A native orthomosaic generator for nadir drone imagery. It doesn't use Docker, VMs, OpenCV, GDAL or PROJ.
The heavy lifting is in Cython-compiled C that releases the GIL, with an optional CUDA backend
that is selected automatically when an NVIDIA GPU is present.

Runtime dependencies: `numpy` and `pillow` (plus `cupy` only if you want the GPU path).

## Install

From PyPI (prebuilt wheels for Linux, Windows and macOS 14+):

```bash
pip install pyOrthomosaic            # CPU
pip install "pyOrthomosaic[cuda]"    # + NVIDIA GPU (CUDA, via CuPy)
pip install "pyOrthomosaic[mps]"     # + Apple-silicon GPU (Metal, via MLX)
```

The package is installed as `pyOrthomosaic` but imported as `orthomosaic`.

From source:

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install numpy cython pillow setuptools
python setup.py build_ext --inplace          # compiles orthomosaic/_core.pyx
# optional GPU (Linux/Windows + NVIDIA driver), pick the wheel matching your CUDA major:
pip install "cupy-cuda12x[ctk]"   # [ctk] brings the CUDA headers CuPy compiles kernels with
```

`ORTHO_NATIVE=1 python setup.py build_ext --inplace` adds `-march=native` for a few % extra.

## Use

```bash
python -m orthomosaic /path/to/images -o ortho.tif
python odm.py /path/to/images ortho.tif
```

```python
from orthomosaic import build_orthomosaic, Options
report = build_orthomosaic("images/", "ortho.tif", Options(backend="auto", render_scale=0.5))
```

Outputs:
- `ortho.tif`: a tiled, Deflate-compressed RGBA GeoTIFF in UTM (EPSG:326xx/327xx). It opens in QGIS and GDAL.
- `ortho_preview.jpg`: a preview no larger than 2048 px.
- `ortho_report.json`: per-image affine transforms, exposure gains, statistics and the options used.

Useful flags:

| Flag | Effect |
|---|---|
| `--backend auto\|cuda\|mps\|cpu` | Compute backend. `auto` picks CUDA, then the Apple GPU (`mps`), then the CPU. |
| `--palette rainbow` | Thermal palette (see below). |
| `--render-scale 0.5` | Decode images at half size: about 4x less RAM and faster, at half the output resolution. |
| `--resolution 0.05` | Output GSD in metres per pixel. |
| `--cache-mb 256` | Cap the decoded-image cache used during rendering. This is the main RAM knob. |
| `--blend seam` | Use a crisp "most-nadir pixel wins" blend instead of feathering. Less ghosting over tall objects. |
| `--gps-sigma 3` | Expected GPS error in metres. Use 0.05 for RTK drones. |
| `--workers N` | Number of CPU threads. |

## Pipeline

1. **Metadata:** EXIF GPS is projected to UTM with a built-in transverse Mercator implementation.
2. **Features:** ORB-style features, computed in `_core.pyx`: a 4-level pyramid, FAST-9 corners
   scored by Harris, grid bucketing, intensity-centroid orientation and 256-bit steered BRIEF.
3. **Pairs:** the k nearest GPS neighbours. Without GPS, all pairs for small sets, or a capture-order window.
4. **Matching:** a single-pass Hamming matcher (popcount) with a ratio test and a mutual check,
   followed by RANSAC affine and a least-squares refit. On a GPU, matching uses a shared-memory CUDA kernel.
5. **Global alignment:** a linear least-squares bundle of one affine per image. X and Y decouple,
   so the solve is a small 3N x 3N system:
   - a relative solve,
   - a robust similarity fit to GPS,
   - a joint refinement with GPS priors. This step includes a scale-lock constraint that removes
     the shrinkage bias of world-space least squares.
6. **Exposure:** per-channel gain compensation in the style of Brown & Lowe.
7. **Render:** blocks of 1024 px are rendered in parallel and streamed into the GeoTIFF, with a
   serpentine traversal and an LRU image cache. Memory stays bounded no matter how large the mosaic is.

## Results on the synthetic benchmark

The benchmark in `tests/synthetic.py` simulates:
- 228 images of 1200x900 px,
- a lawnmower flight with heading flips and ±4° yaw jitter,
- ±3% altitude changes,
- exposure changes and vignetting,
- 1.5 m GPS noise.

Results on an 8-core Apple Silicon machine, CPU only:

| Metric | Result |
|---|---|
| Absolute geolocation | 0.14 m RMS. The block averages out the 1.5 m per-image GPS noise. |
| Relative (shape) accuracy | 0.05 m RMS, about 1 ground pixel |
| Run time | 27 s end to end |

```bash
python tests/synthetic.py make /tmp/syn/images
python -m orthomosaic /tmp/syn/images -o /tmp/syn/ortho.tif
python tests/synthetic.py eval /tmp/syn/images /tmp/syn/ortho.tif
python tests/test_core.py
```

## GPU backends

| Backend | Hardware | How |
|---|---|---|
| `cpu` | any | Cython kernels, all cores, no extra dependencies |
| `cuda` | NVIDIA | CuPy raw kernels compiled at runtime (`pip install "pyOrthomosaic[cuda]"`, which includes the CUDA headers) |
| `mps` | Apple M1/M2/M3/M4 | Metal kernels through MLX (`pip install "pyOrthomosaic[mps]"`) |

GPU backends run a self-test at start-up and fall back to the CPU, with a message saying what to
install, if anything fails. The GPU accelerates matching, rendering and the dense 3D sweep. Every
kernel is tested against the CPU implementation (`tests/test_backends.py`).

Timings on an Apple M1 (8 cores) with the same output on both backends:

| Job | CPU | Apple GPU |
|---|---|---|
| 2D, 228 synthetic images | 33 s | 19 s |
| 3D dense step, synthetic survey | 12 s | 7 s |
| 3D dense step, real 157-image RGB site | — | 80 s (a CPU run with the older, pre-batching code took 11 min) |

## Thermal palettes

DJI radiometric thermal images (R-JPEG: M30T, M3T, M4T, H20T, ...) carry the raw 16-bit sensor
image. When every image has it, the pipeline:

1. mosaics the **raw values** on one survey-wide scale (palette colours are never blended, and
   exposure compensation is disabled);
2. applies the palette to the finished product, with **rainbow** as the default;
3. writes `*_thermal.tif`, a float32 raster of raw values, and `*_legend.png`, a colour bar.

Available palettes are `rainbow` (default), `iron`, `white_hot`, `black_hot`, `arctic`, `lava`,
`hot_metal`, `medical`, `green_hot` and `rainbow_hc`. A custom list of RGB stops also works.

```bash
python -m orthomosaic thermal_images/ -o thermal.tif --palette iron
python -m orthomosaic.thermal list
python -m orthomosaic.thermal recolor thermal_thermal.tif -o thermal_arctic.tif --palette arctic
```

The last command switches the palette afterwards without re-processing.

```python
from orthomosaic import Options, build_orthomosaic, recolor
build_orthomosaic("thermal/", "t.tif", Options(palette="white_hot"))
recolor("t_thermal.tif", "t_custom.tif", palette=[(0, 0, 0), (255, 0, 0), (255, 255, 0)])
```

Raw values are the camera's sensor units, which increase monotonically with temperature. Converting
them to degrees needs the camera's radiometric calibration (DJI Thermal SDK), which isn't done here.
The mosaic stores 256 levels over the survey's value range. Images without radiometric data keep
their JPEG colours; `--no-thermal` forces that behaviour.

## 3D reconstruction (DSM, true orthophoto, point cloud, mesh)

The 3D mode is for nadir grid flights. It produces a 2.5D reconstruction: surfaces seen from
above (roofs, ground, trees), but not walls.

```bash
orthomosaic-3d /path/to/images -o recon/          # or: python -m orthomosaic.reconstruct ...
```

```python
from orthomosaic import build_3d, Options3D
report = build_3d("images/", "recon/", Options3D(dsm_resolution=0.05))
```

Outputs, all in the same local height datum (height above take-off when DJI relative altitude is
present; `report.json` gives the offset to absolute altitude):

| File | Content |
|---|---|
| `dsm.tif` | Float32 elevation GeoTIFF (nodata -9999) |
| `orthophoto.tif` | **True** orthophoto rendered on the DSM, so buildings don't lean |
| `dense.las`, `dense.ply` | Coloured dense point cloud (LAS 1.2 with EPSG, and binary PLY) |
| `mesh.glb`, `mesh.obj` | Textured surface mesh (glTF 2.0 and OBJ+MTL+JPEG) |
| `sparse.ply`, `report.json` | SfM points, camera poses and intrinsics, statistics |
| `dsm_preview.png`, `orthophoto_preview.jpg` | Quick looks |

How it works:

1. **Structure from motion.** The 2D alignment gives the initial poses. Matches are verified with a
   plane+parallax test, which unlike the 8-point method doesn't degenerate on flat scenes. Matches are
   linked into multi-view tracks and triangulated. A **Cython bundle adjustment** (Levenberg–Marquardt,
   Schur complement, Huber loss) refines everything using GPS and DJI altitude priors, and
   self-calibrates the radial distortion.
2. **Dense matching.** A coarse-to-fine *height sweep* over a ground grid. Photo-consistency is
   texture-weighted NCC against a per-cell reference view (the most nadir camera), taken over the
   best half of the other views to handle occlusion. The coarse level is regularised by 8-direction
   **semi-global matching**, so weakly textured roofs take their height from their edges. The same
   code runs on numpy (Cython sampler) or CuPy (CUDA kernel).
3. **Products.** Outlier removal and smooth push-pull hole filling, a true orthophoto that blends only
   the views agreeing at each cell's height, point clouds, and a grid mesh that keeps ridges and walls
   sharp.

Focal length: straight-down imagery can't separate focal length from depth. Scaling both together
gives identical images, and relative altitude does not resolve this. The focal length is therefore
held at the EXIF value (35 mm-equivalent), and heights inherit its accuracy (about 1–2%). Pass
`Options3D(refine_focal=True)` only for flights with large altitude changes.

Synthetic check (`tests/synthetic3d.py`: 48 images, 22 buildings 4–15 m tall with walls, lens
distortion, 1° tilts, 1.5 m GPS noise):
- cameras 0.25 m / 0.2° from truth, distortion recovered;
- confidently matched DSM cells 0.12 m median error at a 10.5 cm cell size;
- 45 s on 8 CPU cores.

Flat, untextured areas and building edges are the weak spots: they are interpolated or slightly
"fattened".

## Scope and limitations

- **2D mode** (`orthomosaic`) is a planar mosaic. It is fast, but tall objects lean. For roofs and
  other relief, use the true orthophoto from the 3D mode.
- **3D mode** is 2.5D (one height per ground cell). Walls, overhangs and oblique/orbit flights aren't
  modelled.
- Thermal imagery has little texture, so its DSM is noticeably less reliable than RGB. Prefer the RGB
  DSM for geometry.
- Gimbal pitch isn't assumed to be nadir: tilts are solved in the bundle adjustment. Strongly oblique
  shots should still be excluded.
