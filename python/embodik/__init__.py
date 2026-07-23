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

# Stall detection & recovery helper
from .stall_handler import StallHandler

# Export transform helpers (Rotation/SO3 - native, no SciPy)
from .transforms import SO3, Rotation

# Export utility functions
from .utils import (
    Rt,
    get_pose_error_vector,
    q2r,
    r2q,
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
        BatchSolveResult,
        check_gpu_availability,
        solve_velocity_batched,
        solve_velocity_gpu_batched,
    )

    _gpu_solver_available = True
except ImportError:
    BatchSolveResult = None


def get_solver_status_hint(status, status_message: str | None = None) -> str:
    """Return an actionable debugging hint for a solver status.

    Parameters
    ----------
    status:
        SolverStatus enum value returned by embodiK solvers.
    status_message:
        Optional backend diagnostic string (e.g. result.status_message).
    """
    status_name = getattr(status, "name", str(status))
    hints = {
        "SUCCESS": "Solve succeeded.",
        "INVALID_INPUT": (
            "Invalid input provided. Verify dimensions and finite numeric values "
            "for goals, Jacobians, constraints, and bounds."
        ),
        "NUMERICAL_ERROR": (
            "Numerical issue during solve. Try reducing task aggressiveness/weights, "
            "increasing damping, or loosening conflicting constraints."
        ),
        "SHAPE_MISMATCH": (
            "Shape mismatch detected. Check that each goal length matches Jacobian rows, "
            "all Jacobians share the same number of columns (nv), and constraints are (m x nv)."
        ),
        "EMPTY_PROBLEM": (
            "Empty problem setup. Provide at least one objective and a non-empty "
            "constraint matrix with matching bounds."
        ),
        "CONSTRAINT_BOUNDS_MISMATCH": (
            "Constraint/bounds mismatch. Ensure lower/upper bounds lengths equal "
            "the number of rows in the constraint matrix."
        ),
        "NON_FINITE_INPUT": (
            "Non-finite values reached the solver. Check for NaN/Inf in task targets, "
            "Jacobians, constraints, and state updates."
        ),
        "INFEASIBLE": (
            "Solver stayed numerically stable, but no feasible solution satisfies "
            "the active goals and constraints together. Relax conflicting constraints, "
            "reduce target magnitudes, or adjust task priorities/weights."
        ),
        "NO_PROGRESS": (
            "Solver stopped because progress stalled near active bounds/constraints. "
            "Try relaxing limits, reducing gains, changing the seed/reference, or "
            "allowing larger per-step motion."
        ),
    }
    base = hints.get(
        status_name,
        "Unknown solver status. Inspect result fields and input dimensions.",
    )
    if status_message:
        return f"{base} Details: {status_message}"
    return base


__all__ = [
    # C++ classes (when available)
    "RobotModel",
    "KinematicsSolver",
    "AccelerationSolver",
    "FrameTask",
    "PostureTask",
    "COMTask",
    "JointTask",
    "MultiJointTask",
    "SolverStatus",
    "VelocitySolverConfig",
    "VelocitySolverResult",
    "AccelerationSolverResult",
    "AccelerationSolverCapabilities",
    "AccelerationSolveOptions",
    "AccelerationTaskReference",
    "AccelerationTaskDiagnostics",
    "AccelerationAllocationDiagnostics",
    "AccelerationAnalyticCollisionPairDiagnostics",
    "AffineAccelerationConstraint",
    "FrozenNextVelocityConstraint",
    "TaskAccelerationBounds",
    "GeneralizedAccelerationAllocation",
    "EffortConstraintOptions",
    "ContactAccelerationConstraint",
    "GeometricConstraintAccelerationPolicy",
    "ComSupportPolygonAccelerationPolicy",
    "ComSupportPolygonAccelerationConstraint",
    "TightPointAccelerationConstraint",
    "TightFramePoseAccelerationConstraint",
    "RelativePoseAccelerationConstraint",
    "TorsoPoseBoundAccelerationConstraint",
    "VelocityCollisionLiftOptions",
    "CollisionConstraintAccelerationPolicy",
    "CollisionGeometryPair",
    "CollisionPairMinimumDistance",
    "CollisionConstraintDefinition",
    "ComSupportPolygonConstraintDefinition",
    "RelativePoseConstraintDefinition",
    "TightPointConstraintDefinition",
    "TightFramePoseConstraintDefinition",
    "TorsoPoseBoundDefinition",
    "ContactType",
    "ComSupportPolygonOutsidePolicy",
    "CollisionConstraintOutsidePolicy",
    "AccelerationCollisionRegime",
    "PositionIKOptions",
    "PositionIKResult",
    "get_solver_status_hint",
    # Python utilities
    "get_pose_error_vector",
    "r2q",
    "q2r",
    "Rt",
    "Rotation",
    "SO3",
    "StallHandler",
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
