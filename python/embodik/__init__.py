"""embodiK: High-performance inverse kinematics with Pinocchio and Viser visualization."""

# Version information
__version__ = "0.1.0"

# Try to import the C++ extension (only once)
_cpp_extension_available = False
try:
    import sys
    module_key = f'{__name__}._embodik_impl'

    if module_key not in sys.modules:
        from ._embodik_impl import *
        _cpp_extension_available = True
    else:
        # Module already loaded, just get the symbols
        _embodik_impl = sys.modules[module_key]
        for name in dir(_embodik_impl):
            if not name.startswith('_') and hasattr(_embodik_impl, name):
                globals()[name] = getattr(_embodik_impl, name)
        _cpp_extension_available = True

except ImportError as e:
    import warnings
    warnings.warn(f"C++ extension not available: {e}. Please build and install the package properly.", ImportWarning)

# Export utility functions
from .utils import (
    get_pose_error_vector,
    r2q,
    q2r,
    Rt,
)

# Export visualization classes (optional)
# Prefer Pinocchio's built-in ViserVisualizer (pin >= 3.8.0) if available
_visualization_available = False
EmbodikVisualizer = None
InteractiveVisualizer = None

try:
    # Try Pinocchio-based visualization first (recommended)
    from .visualization_pinocchio import EmbodikVisualizer, InteractiveVisualizer
    _visualization_available = True
except ImportError:
    # Fall back to custom visualization if Pinocchio's not available
    try:
        from .visualization import EmbodikVisualizer, InteractiveVisualizer
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
]

# Filter out None values from __all__
__all__ = [name for name in __all__ if globals().get(name) is not None]
