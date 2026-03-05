"""embodiK: High-performance inverse kinematics with Pinocchio and Viser visualization."""

# Version information (prefer installed package metadata)
try:
    from importlib.metadata import PackageNotFoundError, version

    __version__ = version("embodik")
except Exception:
    __version__ = "0.0.0"

# NOTE: We intentionally do NOT auto-import Pinocchio at import-time.
# In environments where LD_LIBRARY_PATH points to another Pinocchio install (e.g. a local build),
# forcing mixed shared-library stacks into the same process can cause hard crashes (double-free).

# Try to import the C++ extension (only once)
_cpp_extension_available = False
_cpp_extension_error: str | None = None
try:
    import sys

    module_key = f"{__name__}._embodik_impl"

    if module_key not in sys.modules:
        from ._embodik_impl import *

        _cpp_extension_available = True
    else:
        # Module already loaded, just get the symbols
        _embodik_impl = sys.modules[module_key]
        for name in dir(_embodik_impl):
            if not name.startswith("_") and hasattr(_embodik_impl, name):
                globals()[name] = getattr(_embodik_impl, name)
        _cpp_extension_available = True

except ImportError as e:
    import warnings

    _cpp_extension_error = str(e)
    warnings.warn(
        f"C++ extension not available: {e}. Please build and install the package properly.",
        ImportWarning,
    )

# Export utility functions
from .utils import (
    get_pose_error_vector,
    r2q,
    q2r,
    Rt,
)

# Export visualization classes (optional)
# Default to direct Viser visualization (no pip pinocchio dependency)
_visualization_available = False
EmbodikVisualizer = None
InteractiveVisualizer = None

try:
    # Use direct Viser visualization (default - no pip pinocchio needed)
    from .visualization import EmbodikVisualizer, InteractiveVisualizer

    _visualization_available = True
except ImportError:
    # Fall back to Pinocchio-based visualization if direct Viser fails
    try:
        from .visualization_pinocchio import EmbodikVisualizer, InteractiveVisualizer

        _visualization_available = True
    except ImportError:
        _visualization_available = False

# Export robot visualizer (always available if dependencies are installed)
try:
    from .robot_visualizer import RobotVisualizer, create_robot_visualizer

    _robot_visualizer_available = True
except ImportError:
    RobotVisualizer = None
    create_robot_visualizer = None
    _robot_visualizer_available = False

# GPU solver (optional - requires CusADi + CUDA)
_gpu_solver_available = False
solve_velocity_batched = None
solve_velocity_gpu_batched = None
check_gpu_availability = None
try:
    from .gpu_solver import (
        solve_velocity_batched,
        solve_velocity_gpu_batched,
        check_gpu_availability,
        BatchSolveResult,
    )

    _gpu_solver_available = True
except ImportError:
    BatchSolveResult = None

__all__ = [
    # C++ classes (when available)
    "RobotModel",
    "KinematicsSolver",
    "FrameTask",
    "PostureTask",
    "COMTask",
    "JointTask",
    "MultiJointTask",
    "SolverStatus",
    "VelocitySolverConfig",
    "VelocitySolverResult",
    "PositionIKOptions",
    "PositionIKResult",
    # Python utilities
    "get_pose_error_vector",
    "r2q",
    "q2r",
    "Rt",
    # Visualization (optional)
    "EmbodikVisualizer",
    "InteractiveVisualizer",
    # Robot visualizer
    "RobotVisualizer",
    "create_robot_visualizer",
    # GPU solver (optional)
    "solve_velocity_batched",
    "solve_velocity_gpu_batched",
    "check_gpu_availability",
    "BatchSolveResult",
]

# Filter out None values from __all__
__all__ = [name for name in __all__ if globals().get(name) is not None]
