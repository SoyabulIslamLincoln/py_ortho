# orthomosaic

A native orthomosaic generator for nadir drone imagery. It doesn't use Docker, VMs, OpenCV, GDAL or PROJ.
The heavy lifting is in Cython-compiled C that releases the GIL, with an optional CUDA backend
that is selected automatically when an NVIDIA GPU is present.

Runtime dependencies: `numpy` and `pillow` (plus `cupy` only if you want the GPU path).

## Install

From PyPI (prebuilt wheels for Linux, Windows and macOS 14+):

```bash
pip install pyOrthomosaic            # CPU
pip install "pyOrthomosaic[cuda]"    # + NVIDIA GPU support via CuPy
```

The package is installed as `pyOrthomosaic` but imported as `orthomosaic`.

From source:

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install numpy cython pillow setuptools
python setup.py build_ext --inplace          # compiles orthomosaic/_core.pyx
# optional GPU (Linux/Windows + NVIDIA driver), pick the wheel matching your CUDA major:
pip install cupy-cuda12x
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
| `--backend cpu\|cuda\|auto` | Choose the compute backend (`auto` uses CUDA if available). |
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

## Scope and limitations

This is a **planar (2D) orthomosaic**: each image is mapped to the ground with an affine transform.
It is accurate for nadir imagery over terrain that is flat or gently rolling relative to the
flight altitude, which covers typical agricultural, thermal and mapping surveys.

It does **not** do:
- full structure-from-motion or DEM-based orthorectification,
- correction for lens distortion,
- correction for gimbal pitch.

Tall buildings and steep terrain will therefore show relief displacement or ghosting. `--blend seam`
reduces the ghosting. The natural next steps are:
- a Brown–Conrady undistortion pass,
- a homography or bundle-adjusted camera model,
- a coarse DEM from triangulated tie points.
