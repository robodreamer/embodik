"""Model-derived GPU whole-body inverse kinematics.

This experimental API constructs fixed-shape CUDA kernels from a loaded
:class:`embodik.RobotModel`.  Robot dimensions, joint mappings, task frames,
and collision geometry names are read from the model; robot-family names and
known degree-of-freedom tables are deliberately absent from the contract.

Heavy optional dependencies are imported only when a solver is constructed,
so importing :mod:`embodik` remains CPU-only and lightweight.
"""

from .acceleration import (
    GPU_ACCELERATION_CAPABILITIES,
    GpuAccelerationBatchResult,
    GpuAccelerationCapabilities,
    GpuAccelerationResult,
    GpuAccelerationSolver,
)
from .contracts import ContractViolation, RobotSolveSpec
from .model_spec import (
    PoseModelParameters,
    floating_pose_model_parameters_from_embodik,
    pose_model_parameters_from_embodik,
    robot_solve_spec_from_embodik,
)
from .solver import (
    GPU_WBC_CAPABILITIES,
    GpuWbcCapabilities,
    GpuWbcCollisionDebug,
    GpuWbcFloatingMultiFrameSolver,
    GpuWbcMultiFrameResult,
    GpuWbcMultiFrameSolver,
    GpuWbcResourceReport,
    derive_frame_active_joint_names,
    derive_frames_active_joint_names,
    derive_frames_active_velocity_indices,
    derive_supported_active_joint_names,
    gpu_wbc_collision_control_availability,
)

__all__ = [
    "ContractViolation",
    "GPU_ACCELERATION_CAPABILITIES",
    "GPU_WBC_CAPABILITIES",
    "GpuAccelerationBatchResult",
    "GpuAccelerationCapabilities",
    "GpuAccelerationResult",
    "GpuAccelerationSolver",
    "GpuWbcCollisionDebug",
    "GpuWbcCapabilities",
    "GpuWbcFloatingMultiFrameSolver",
    "GpuWbcMultiFrameResult",
    "GpuWbcMultiFrameSolver",
    "GpuWbcResourceReport",
    "PoseModelParameters",
    "RobotSolveSpec",
    "derive_frame_active_joint_names",
    "derive_frames_active_joint_names",
    "derive_frames_active_velocity_indices",
    "derive_supported_active_joint_names",
    "floating_pose_model_parameters_from_embodik",
    "gpu_wbc_collision_control_availability",
    "pose_model_parameters_from_embodik",
    "robot_solve_spec_from_embodik",
]
