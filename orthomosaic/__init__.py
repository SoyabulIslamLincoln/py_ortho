"""orthomosaic -- native (Cython + optional CUDA) orthomosaic generation, no Docker/VM."""
from .backend import cuda_available, select_backend
from .pipeline import Options, build_orthomosaic

__all__ = ["Options", "build_orthomosaic", "cuda_available", "select_backend"]
__version__ = "0.1.0"
