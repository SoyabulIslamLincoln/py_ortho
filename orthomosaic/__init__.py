"""orthomosaic -- native (Cython + optional CUDA) orthomosaic generation, no Docker/VM."""
from .backend import cuda_available, select_backend
from .pipeline import Options, build_orthomosaic
from .reconstruct import Options3D, build_3d

__all__ = ["Options", "Options3D", "build_orthomosaic", "build_3d", "cuda_available", "select_backend"]
try:  # single source of truth: the version in pyproject.toml
    from importlib.metadata import version as _version
    __version__ = _version("pyOrthomosaic")
except Exception:  # running from a source tree that is not installed
    __version__ = "0+unknown"
