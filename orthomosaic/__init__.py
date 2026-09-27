"""orthomosaic -- native (Cython + optional CUDA) orthomosaic generation, no Docker/VM."""
from .backend import cuda_available, select_backend
from .pipeline import Options, build_orthomosaic
from .reconstruct import Options3D, build_3d

__all__ = ["Options", "Options3D", "build_orthomosaic", "build_3d", "cuda_available", "select_backend"]
__version__ = "0.1.0"
