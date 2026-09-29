"""orthomosaic -- native (Cython + optional CUDA) orthomosaic generation, no Docker/VM."""
from .backend import cuda_available, select_backend
#changed here: expose the Pix4D-style ortho / colour / elevation-mapping entry points
from .color import solve_radiometric
from .gcp import load_gcps
from .ortho import true_orthophoto
from .pipeline import Options, build_orthomosaic
from .reconstruct import Options3D, build_3d, build_thermal_bound
from .qcreport import write_quality_report
from .terrain import TerrainOptions, classify_surface, contour_lines, dtm_from_dsm
from .thermal import apply_palette, palette_names, recolor

__all__ = ["Options", "Options3D", "build_orthomosaic", "build_3d", "build_thermal_bound",
           "cuda_available", "select_backend", "palette_names", "apply_palette", "recolor",
           "dtm_from_dsm", "TerrainOptions", "load_gcps", "write_quality_report",
           "true_orthophoto", "solve_radiometric", "classify_surface", "contour_lines"]
try:  # single source of truth: the version in pyproject.toml
    from importlib.metadata import version as _version
    __version__ = _version("pyOrthomosaic")
except Exception:  # running from a source tree that is not installed
    __version__ = "0+unknown"
