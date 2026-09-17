"""Private CUDA runtime used by :mod:`embodik.gpu.wbc`.

The implementation is shape-specialized internally, but every shape is
derived from the loaded robot contract. Nothing here is a stable public API.
"""

from .multi_pose_solver import (
    COMPACT_PUBLICATION_SCALARS,
    DeviceResidentMultiFramePoseSolver,
    MultiFramePoseBatchResult,
    MultiFramePoseSolveConfig,
)

__all__ = [
    "COMPACT_PUBLICATION_SCALARS",
    "DeviceResidentMultiFramePoseSolver",
    "MultiFramePoseBatchResult",
    "MultiFramePoseSolveConfig",
]
