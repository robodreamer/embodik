"""Device-resident model-derived multi-frame pose solver."""

from __future__ import annotations

import math
import os
import sysconfig
import warnings
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from ..contracts import RobotSolveSpec
from ._pose_math import _clamp_norm, _orientation_error
from .cusadi_sm120 import StrictCusadiFunction
from .gpu_constraints import (
    CPU_MARGIN_THRESHOLD,
    CPU_MIN_BOUND_FRACTION,
    CPU_TORSO_BOUND_SLACK_EPS_ROT,
    CPU_TORSO_BOUND_SLACK_EPS_TRANS,
    _capture_point_constraint_rows_trusted,
    _velocity_zmp_constraint_rows_trusted,
    bounded_cyclic_row_projection,
    com_halfspace_velocity_bounds,
    relative_pose_state,
    support_polygon_halfplanes,
    torso_pose_bound_rows,
)
from .gpu_priority import (
    PostureTaskRows,
    _configuration_indices_for_velocity_rows,
    ordered_priority_velocity,
)
from .newton_model import NewtonModelKinematics


def _python_extension_headers_available() -> bool:
    """Return whether first-use Triton helper compilation can include Python.h."""

    search_roots = {
        value
        for value in (
            sysconfig.get_path("include"),
            sysconfig.get_path("platinclude"),
        )
        if value
    }
    for variable in ("C_INCLUDE_PATH", "CPATH"):
        search_roots.update(
            value for value in os.environ.get(variable, "").split(os.pathsep) if value
        )
    return any((Path(root) / "Python.h").is_file() for root in search_roots)


@dataclass(frozen=True)
class MultiFramePoseSolveConfig:
    iterations: int = 2
    dt: float = 0.1
    adaptive_dt: bool = False
    adaptive_dt_max_scale: float = 5.0
    adaptive_dt_reference_distance: float = 0.05
    position_gain: float = 8.0
    orientation_gain: float = 4.0
    max_linear_speed: float = 2.0
    max_angular_speed: float = 2.0
    max_joint_step_rad: float = 0.04
    max_joint_acceleration_rad_s2: float | None = None
    position_tolerance_m: float = 0.008
    orientation_tolerance_rad: float = 0.03
    allow_nonconverged_progress_steps: bool = False
    standalone_cuda_graph_enabled: bool = False
    fi_scalar_type: str = "float64"
    velocity_solver: str = "fi_pesns"
    srinv_tolerance: float = 0.1
    srinv_damping: float = 0.1
    cusolver_specialized_outputs_enabled: bool = False
    cusolver_reuse_locked_primary_inverse_enabled: bool = False
    cusolver_locked_primary_reuse_min_singular_ratio: float = 1.0e-2
    native_compile_enabled: bool = True
    collision_enabled: bool = False
    collision_debug_enabled: bool = False
    collision_min_distance_m: float = 0.05
    collision_query_distance_m: float = 0.10
    collision_tolerance_m: float = 1e-4
    collision_velocity_tolerance_m_s: float = 1e-5
    collision_repulsion_deadband_m: float = 3e-3
    collision_recovery_scale: float = 0.2
    collision_max_separation_speed_nonpenetrating_m_s: float = 0.15
    collision_max_separation_speed_m_s: float = 0.5
    collision_min_recovery_speed_m_s: float = 0.05
    collision_projection_iterations: int = 24
    collision_max_constraints: int = 3
    collision_graph_repair_iterations: int = 2
    collision_contacts_per_world: int = 4096
    collision_triangle_pairs_per_world: int = 1024
    collision_contact_sort_enabled: bool = True
    collision_clear_state_fast_path_enabled: bool = False
    collision_candidate_convex_certificate_enabled: bool = False
    collision_current_convex_certificate_enabled: bool = False
    torso_projection_noop_fast_path_enabled: bool = False
    contact_position_tolerance_m: float = 1e-4
    contact_orientation_tolerance_rad: float = 1e-3

    def __post_init__(self) -> None:
        positive = (
            self.iterations,
            self.dt,
            self.adaptive_dt_max_scale,
            self.adaptive_dt_reference_distance,
            self.position_gain,
            self.orientation_gain,
            self.max_linear_speed,
            self.max_angular_speed,
            self.max_joint_step_rad,
            self.position_tolerance_m,
            self.orientation_tolerance_rad,
            self.srinv_tolerance,
            self.srinv_damping,
            self.collision_min_distance_m,
            self.collision_query_distance_m,
            self.collision_tolerance_m,
            self.collision_velocity_tolerance_m_s,
            self.collision_recovery_scale,
            self.collision_max_separation_speed_nonpenetrating_m_s,
            self.collision_max_separation_speed_m_s,
            self.collision_min_recovery_speed_m_s,
            self.contact_position_tolerance_m,
            self.contact_orientation_tolerance_rad,
        )
        if not all(math.isfinite(value) and value > 0 for value in positive):
            raise ValueError("multi-frame pose budgets and gains must be positive")
        if self.adaptive_dt_max_scale <= 1.0:
            raise ValueError("adaptive_dt_max_scale must exceed 1.0")
        if self.adaptive_dt_reference_distance <= 1e-9:
            raise ValueError("adaptive_dt_reference_distance must exceed 1e-9")
        if self.max_joint_acceleration_rad_s2 is not None and (
            not math.isfinite(self.max_joint_acceleration_rad_s2)
            or self.max_joint_acceleration_rad_s2 <= 0.0
        ):
            raise ValueError("max_joint_acceleration_rad_s2 must be positive")
        if self.fi_scalar_type not in {"float64", "float32"}:
            raise ValueError("fi_scalar_type must be float64 or float32")
        if self.velocity_solver not in {
            "fi_pesns",
            "torch_srinv",
            "warp_srinv",
            "cusolver_srinv",
        }:
            raise ValueError(
                "velocity_solver must be fi_pesns, torch_srinv, warp_srinv, " "or cusolver_srinv"
            )
        if self.velocity_solver == "torch_srinv" and self.standalone_cuda_graph_enabled:
            raise ValueError("torch_srinv does not support standalone CUDA Graph capture")
        if (
            self.cusolver_reuse_locked_primary_inverse_enabled
            and self.velocity_solver != "cusolver_srinv"
        ):
            raise ValueError("locked primary inverse reuse requires the cusolver_srinv backend")
        if (
            not math.isfinite(self.cusolver_locked_primary_reuse_min_singular_ratio)
            or not 0.0 < self.cusolver_locked_primary_reuse_min_singular_ratio <= 1.0
        ):
            raise ValueError("locked primary reuse singular ratio must be in (0, 1]")
        if self.collision_query_distance_m <= self.collision_min_distance_m:
            raise ValueError("collision query distance must exceed minimum distance")
        if self.collision_repulsion_deadband_m < 0.0:
            raise ValueError("collision repulsion deadband must be nonnegative")
        if (
            self.collision_projection_iterations <= 0
            or self.collision_max_constraints <= 0
            or self.collision_contacts_per_world <= 0
            or self.collision_triangle_pairs_per_world <= 0
        ):
            raise ValueError(
                "collision iterations/capacities must be positive and graph repair "
                "iterations nonnegative"
            )
        if (
            type(self.collision_graph_repair_iterations) is not int
            or self.collision_graph_repair_iterations < 0
        ):
            raise ValueError("collision_graph_repair_iterations must be a nonnegative integer")
        if self.collision_clear_state_fast_path_enabled and (
            not self.collision_enabled
            or not self.standalone_cuda_graph_enabled
            or self.velocity_solver not in {"warp_srinv", "cusolver_srinv"}
            or self.collision_graph_repair_iterations != 0
        ):
            raise ValueError(
                "collision clear-state fast path requires graph-captured native "
                "collision with zero repair iterations"
            )
        if self.collision_candidate_convex_certificate_enabled and (
            not self.collision_clear_state_fast_path_enabled
        ):
            raise ValueError(
                "candidate convex certification requires the collision clear-state path"
            )
        if self.collision_current_convex_certificate_enabled and (
            not self.collision_clear_state_fast_path_enabled
        ):
            raise ValueError("current convex certification requires the collision clear-state path")
        if (
            self.torso_projection_noop_fast_path_enabled
            and not self.collision_clear_state_fast_path_enabled
        ):
            raise ValueError("torso no-op fast path requires the certified collision fast path")


@dataclass(frozen=True)
class MultiFramePoseBatchResult:
    status: str
    q_solution: Any
    q_candidate: Any
    position_error_m: Any
    orientation_error_rad: Any
    converged: Any
    fallback_used: bool
    actual_device: str
    kernel_time_ms: float
    minimum_collision_distance_m: Any
    collision_active: Any
    collision_step_accepted: Any
    collision_overflow: Any
    collision_constraint_applied: Any
    closest_collision_point0_world_m: Any
    closest_collision_point1_world_m: Any
    closest_collision_shape_pair: Any
    collision_enabled: bool
    effective_dt: Any = None
    accepted_velocity: Any = None
    com_constraint_enabled: bool = False
    com_constraint_applied: Any = None
    com_constraint_feasible: Any = None
    minimum_com_slack_m: Any = None
    capture_point_constraint_enabled: bool = False
    capture_point_constraint_applied: Any = None
    capture_point_constraint_feasible: Any = None
    minimum_capture_point_slack_m: Any = None
    velocity_zmp_constraint_enabled: bool = False
    velocity_zmp_constraint_applied: Any = None
    velocity_zmp_constraint_feasible: Any = None
    minimum_velocity_zmp_slack_m: Any = None
    minimum_zmp_normal_force_n: Any = None
    centroidal_momentum_task_enabled: bool = False
    centroidal_momentum_task_applied: Any = None
    torso_constraint_enabled: bool = False
    torso_constraint_applied: Any = None
    torso_constraint_feasible: Any = None
    posture_task_enabled: bool = False
    posture_task_applied: Any = None
    posture_primary_residual_increase: Any = None
    posture_secondary_residual_before: Any = None
    posture_secondary_residual_after: Any = None
    secondary_task_enabled: bool = False
    secondary_task_applied: Any = None
    secondary_primary_residual_increase: Any = None
    secondary_residual_before: Any = None
    secondary_residual_after: Any = None
    spectral_solve_ok: Any = None
    compact_publication: Any = None
    collision_clear_state_certified: Any = None
    solved_primary_pose_xyzw: Any = None


COMPACT_PUBLICATION_SCALARS = (
    "converged",
    "minimum_collision_distance_m",
    "collision_active",
    "collision_step_accepted",
    "collision_overflow",
    "collision_constraint_applied",
    "effective_dt",
    "com_constraint_applied",
    "com_constraint_feasible",
    "minimum_com_slack_m",
    "torso_constraint_applied",
    "torso_constraint_feasible",
    "posture_task_applied",
    "posture_primary_residual_increase",
    "posture_secondary_residual_before",
    "posture_secondary_residual_after",
    "secondary_task_applied",
    "secondary_primary_residual_increase",
    "secondary_residual_before",
    "secondary_residual_after",
    "spectral_solve_ok",
    "collision_clear_state_certified",
    "capture_point_constraint_applied",
    "capture_point_constraint_feasible",
    "minimum_capture_point_slack_m",
    "velocity_zmp_constraint_applied",
    "velocity_zmp_constraint_feasible",
    "minimum_velocity_zmp_slack_m",
    "minimum_zmp_normal_force_n",
    "centroidal_momentum_task_applied",
)


def _floating_position_limits_valid(
    q_position: Any,
    lower: Any,
    upper: Any,
    position_velocity_indices: tuple[int, ...] | list[int],
    locked_active_columns: tuple[int, ...],
) -> Any:
    """Check limits only for scalar coordinates the IK is allowed to move.

    Physics-owned locked coordinates can arrive marginally outside the URDF
    limits. Rejecting the whole request would ask the IK to repair coordinates
    whose velocity is constrained to exact zero.
    """

    locked = set(locked_active_columns)
    rows = [
        row
        for row, active_column in enumerate(position_velocity_indices)
        if active_column not in locked
    ]
    if not rows:
        return q_position.new_tensor(True, dtype=q_position.dtype).bool()
    return ((q_position[:, rows] >= lower[rows]) & (q_position[:, rows] <= upper[rows])).all()


def _validate_fi_function(function: Any, velocity_dim: int, frame_count: int) -> None:
    task_rows = 6 * frame_count
    expected = (
        task_rows,
        task_rows * velocity_dim,
        velocity_dim * velocity_dim,
        velocity_dim,
        velocity_dim,
    )
    actual = tuple(function.nnz_in(index) for index in range(function.n_in()))
    if actual != expected or function.n_out() < 1:
        raise ValueError(
            f"multi-frame pose solver requires FI inputs with nnz {expected}; " f"received {actual}"
        )
    if function.nnz_out(0) != velocity_dim:
        raise ValueError(f"multi-frame pose solver requires {velocity_dim} velocity outputs")


def _directional_srinv(torch: Any, matrix: Any, tolerance: float, damping: float):
    """Apply the CPU extended-SRINV spectrum to a batched CUDA matrix."""

    gram = matrix @ matrix.transpose(-2, -1)
    threshold_squared = tolerance * tolerance
    determinant = torch.linalg.det(gram)
    global_regularization = torch.where(
        determinant < threshold_squared,
        (1.0 - (determinant / threshold_squared) ** 2) * threshold_squared,
        torch.zeros_like(determinant),
    )
    left, singular_values, right_transpose = torch.linalg.svd(matrix, full_matrices=False)
    normalized = torch.clamp(singular_values / tolerance, max=1.0)
    per_value_damping = damping * torch.clamp(1.0 - normalized * normalized, min=0.0)
    denominator = (
        singular_values * singular_values + global_regularization[:, None] + per_value_damping
    )
    inverse_spectrum = singular_values / denominator
    return (right_transpose.transpose(-2, -1) * inverse_spectrum[:, None, :]) @ (
        left.transpose(-2, -1)
    )


def _multi_frame_nonworsening(
    torch: Any,
    position: Any,
    initial_position: Any,
    orientation: Any,
    initial_orientation: Any,
    position_tolerance_m: float,
    orientation_tolerance_rad: float,
):
    """Accept improvement without penalizing errors already inside deadbands."""

    position_ok = (position <= initial_position + 1e-6) | (position <= position_tolerance_m)
    orientation_ok = (orientation <= initial_orientation + 1e-6) | (
        orientation <= orientation_tolerance_rad
    )
    return torch.all(position_ok & orientation_ok, dim=-1)


def _adaptive_dt_scale(
    torch: Any,
    max_position_error: Any,
    *,
    enabled: bool,
    max_scale: float,
    reference_distance: float,
    collision_enabled: bool,
    collision_margin: Any | None,
    collision_tolerance_m: float,
):
    """Return the CPU position-step adaptive horizon multiplier per world."""

    scale = torch.ones_like(max_position_error)
    if not enabled:
        return scale
    scale = torch.clamp(
        max_position_error / reference_distance,
        min=1.0,
        max=max_scale,
    )
    if collision_enabled and collision_margin is not None:
        scale = torch.where(
            collision_margin < -collision_tolerance_m,
            torch.ones_like(scale),
            scale,
        )
    return scale


def _torso_candidate_nonworsening(
    torch: Any,
    current_state: Any,
    candidate_state: Any,
    lower_limit: Any,
    upper_limit: Any,
    axis_mask: Any,
    tolerance: Any,
):
    """Apply the CPU-style inside-or-recovering torso candidate policy."""

    current_violation = torch.maximum(
        torch.clamp(lower_limit - current_state, min=0.0),
        torch.clamp(current_state - upper_limit, min=0.0),
    )
    candidate_violation = torch.maximum(
        torch.clamp(lower_limit - candidate_state, min=0.0),
        torch.clamp(candidate_state - upper_limit, min=0.0),
    )
    current_inside = current_violation == 0.0
    valid_axis = torch.where(
        current_inside,
        candidate_violation <= tolerance,
        candidate_violation <= current_violation + tolerance,
    )
    return torch.all(~axis_mask | valid_axis, dim=-1)


def _validate_torso_configuration(
    *,
    velocity_dim: int,
    torso_frame_name: str | None,
    torso_reference_pose_xyzw: tuple[float, ...] | None,
    torso_lower_relative_limits: tuple[float, ...] | None,
    torso_upper_relative_limits: tuple[float, ...] | None,
    torso_axis_mask: tuple[bool, ...] | None,
    torso_velocity_limits: tuple[float, ...] | None,
    torso_acceleration_limits: tuple[float, ...] | None,
    torso_excluded_active_velocity_indices: tuple[int, ...],
    torso_headroom_fraction: float,
    torso_headroom_activation_margin: float,
    torso_projection_iterations: int,
    torso_feasibility_tolerance: float,
) -> bool:
    """Validate fixed-shape torso options before CUDA dependencies are needed."""

    torso_values = (
        torso_reference_pose_xyzw,
        torso_lower_relative_limits,
        torso_upper_relative_limits,
        torso_axis_mask,
        torso_velocity_limits,
        torso_acceleration_limits,
    )
    if torso_frame_name is None:
        if any(value is not None for value in torso_values) or (
            torso_excluded_active_velocity_indices
        ):
            raise ValueError("torso_frame_name is required when torso options are set")
        return False
    if not torso_frame_name:
        raise ValueError("torso_frame_name must not be empty")
    if any(value is None for value in torso_values):
        raise ValueError("enabled torso constraints require all pose and limit options")
    assert torso_reference_pose_xyzw is not None
    assert torso_lower_relative_limits is not None
    assert torso_upper_relative_limits is not None
    assert torso_axis_mask is not None
    assert torso_velocity_limits is not None
    assert torso_acceleration_limits is not None
    vectors = {
        "torso_reference_pose_xyzw": (torso_reference_pose_xyzw, 7),
        "torso_lower_relative_limits": (torso_lower_relative_limits, 6),
        "torso_upper_relative_limits": (torso_upper_relative_limits, 6),
        "torso_axis_mask": (torso_axis_mask, 6),
        "torso_velocity_limits": (torso_velocity_limits, 6),
        "torso_acceleration_limits": (torso_acceleration_limits, 6),
    }
    for name, (values, expected) in vectors.items():
        if len(values) != expected:
            raise ValueError(f"{name} must contain {expected} values")
    numeric_vectors = (
        torso_reference_pose_xyzw,
        torso_lower_relative_limits,
        torso_upper_relative_limits,
        torso_velocity_limits,
        torso_acceleration_limits,
    )
    if not all(math.isfinite(float(value)) for values in numeric_vectors for value in values):
        raise ValueError("torso pose and limits must be finite")
    if math.sqrt(sum(value * value for value in torso_reference_pose_xyzw[3:])) <= 1e-8:
        raise ValueError("torso reference quaternion must be nonzero")
    if any(
        lower >= upper
        for lower, upper in zip(
            torso_lower_relative_limits, torso_upper_relative_limits, strict=True
        )
    ):
        raise ValueError("torso lower limits must be below upper limits")
    if any(value <= 0.0 for value in torso_velocity_limits):
        raise ValueError("torso velocity limits must be positive")
    if any(value <= 0.0 for value in torso_acceleration_limits):
        raise ValueError("torso acceleration limits must be positive")
    excluded = tuple(int(index) for index in torso_excluded_active_velocity_indices)
    if len(set(excluded)) != len(excluded) or any(
        index < 0 or index >= velocity_dim for index in excluded
    ):
        raise ValueError("torso excluded active velocity indices must be unique and valid")
    scalar_values = (
        torso_headroom_fraction,
        torso_headroom_activation_margin,
        torso_feasibility_tolerance,
    )
    if not all(math.isfinite(value) and value >= 0.0 for value in scalar_values):
        raise ValueError("torso headroom and feasibility options must be nonnegative")
    if torso_projection_iterations <= 0:
        raise ValueError("torso_projection_iterations must be positive")
    return True


def _validate_posture_configuration(
    *,
    configuration_dim: int,
    active_velocity_indices: tuple[int, ...],
    posture_target_configuration: tuple[float, ...] | None,
    posture_velocity_indices: tuple[int, ...],
    posture_weights: tuple[float, ...],
    posture_gain: float,
    posture_rank_tolerance: float,
    locked_velocity_indices: tuple[int, ...],
) -> bool:
    """Validate model-source velocity selections before CUDA allocation."""

    active = tuple(int(index) for index in active_velocity_indices)
    selected = tuple(int(index) for index in posture_velocity_indices)
    locked = tuple(int(index) for index in locked_velocity_indices)
    active_set = set(active)
    for name, values in (("posture", selected), ("locked", locked)):
        if len(values) != len(set(values)) or any(value not in active_set for value in values):
            raise ValueError(f"{name} velocity indices must be unique active model indices")
    if posture_target_configuration is None:
        if selected or posture_weights:
            raise ValueError("posture target configuration is required for posture rows")
        return False
    if len(posture_target_configuration) != configuration_dim or not all(
        math.isfinite(float(value)) for value in posture_target_configuration
    ):
        raise ValueError(
            f"posture target configuration must contain {configuration_dim} finite values"
        )
    if not selected or len(posture_weights) != len(selected):
        raise ValueError("posture weights must match a nonempty velocity selection")
    if not all(math.isfinite(float(value)) and float(value) >= 0.0 for value in posture_weights):
        raise ValueError("posture weights must be finite and nonnegative")
    if not math.isfinite(posture_gain) or posture_gain < 0.0:
        raise ValueError("posture gain must be finite and nonnegative")
    if not math.isfinite(posture_rank_tolerance) or posture_rank_tolerance < 0.0:
        raise ValueError("posture rank tolerance must be finite and nonnegative")
    return True


def _validate_secondary_frame_configuration(
    *,
    names: tuple[str, ...],
    dimensions: tuple[int, ...],
    target_poses_wxyz: tuple[tuple[float, ...], ...],
    position_gains: tuple[float, ...],
    orientation_gains: tuple[float, ...],
    weights: tuple[float, ...],
) -> bool:
    """Validate fixed-shape priority-1 frame tasks without model assumptions."""

    lengths = {
        len(names),
        len(dimensions),
        len(target_poses_wxyz),
        len(position_gains),
        len(orientation_gains),
        len(weights),
    }
    if lengths == {0}:
        return False
    if len(lengths) != 1 or not names:
        raise ValueError("secondary frame options must have matching nonzero lengths")
    if len(set(names)) != len(names) or any(not name for name in names):
        raise ValueError("secondary frame names must be unique and nonempty")
    if any(dimension not in (3, 6) for dimension in dimensions):
        raise ValueError("secondary frame dimensions must be 3 or 6")
    if any(len(pose) != 7 for pose in target_poses_wxyz) or not all(
        math.isfinite(float(value)) for pose in target_poses_wxyz for value in pose
    ):
        raise ValueError("secondary target poses must contain seven finite values")
    if any(
        math.sqrt(sum(float(value) ** 2 for value in pose[3:])) <= 1.0e-8
        for pose in target_poses_wxyz
    ):
        raise ValueError("secondary target quaternions must be nonzero")
    if not all(math.isfinite(value) and value > 0.0 for value in position_gains):
        raise ValueError("secondary position gains must be finite and positive")
    if not all(math.isfinite(value) and value >= 0.0 for value in orientation_gains):
        raise ValueError("secondary orientation gains must be finite and nonnegative")
    if not all(math.isfinite(value) and value >= 0.0 for value in weights):
        raise ValueError("secondary frame weights must be finite and nonnegative")
    return True


class DeviceResidentMultiFramePoseSolver:
    """Fixed-shape CUDA pose solver for ordered model-derived frame tasks."""

    def __init__(
        self,
        batch_size: int,
        fi_function: Any | None,
        fi_library: Path | None,
        urdf_path: Path,
        cache_dir: Path,
        *,
        robot_spec: RobotSolveSpec,
        frames: tuple[str, ...],
        frame_task_dimensions: tuple[int, ...] | None = None,
        frame_contact_constraints: tuple[bool, ...] | None = None,
        frame_position_gains: tuple[float, ...] | None = None,
        frame_orientation_gains: tuple[float, ...] | None = None,
        default_configuration: tuple[float, ...],
        joint_lower: tuple[float, ...],
        joint_upper: tuple[float, ...],
        joint_velocity_limits: tuple[float, ...],
        collision_pairs: tuple[tuple[str, str], ...] = (),
        com_support_polygon_xy: Any = None,
        com_robot_spec: RobotSolveSpec | None = None,
        com_default_configuration: tuple[float, ...] | None = None,
        com_max_constraints: int = 8,
        com_margin: float = 0.0,
        com_vel_max: float = 0.4,
        com_acc_max: float = 0.1,
        com_use_acceleration_limits: bool = True,
        com_proximity_fraction: float = 0.2,
        com_excluded_velocity_indices: tuple[int, ...] = (),
        com_projection_iterations: int = 16,
        capture_point_support_polygon_xy: Any = None,
        capture_point_max_constraints: int = 8,
        capture_point_margin: float = 0.0,
        capture_point_omega: float | None = None,
        capture_point_height: float | None = None,
        capture_point_gravity_z: float = -9.81,
        velocity_zmp_support_polygon_xy: Any = None,
        velocity_zmp_max_constraints: int = 8,
        velocity_zmp_margin: float = 0.0,
        velocity_zmp_fz_min: float = 1.0,
        velocity_zmp_gravity_z: float = -9.81,
        centroidal_momentum_target: tuple[float, ...] | None = None,
        centroidal_momentum_axis_mask: tuple[bool, ...] = (),
        centroidal_momentum_weight: float = 1.0,
        centroidal_momentum_priority: int = 1,
        centroidal_momentum_excluded_velocity_indices: tuple[int, ...] = (),
        centroidal_momentum_lower: tuple[float, ...] | None = None,
        centroidal_momentum_upper: tuple[float, ...] | None = None,
        centroidal_projection_iterations: int = 24,
        torso_frame_name: str | None = None,
        torso_reference_pose_xyzw: tuple[float, ...] | None = None,
        torso_lower_relative_limits: tuple[float, ...] | None = None,
        torso_upper_relative_limits: tuple[float, ...] | None = None,
        torso_axis_mask: tuple[bool, ...] | None = None,
        torso_velocity_limits: tuple[float, ...] | None = None,
        torso_acceleration_limits: tuple[float, ...] | None = None,
        torso_excluded_active_velocity_indices: tuple[int, ...] = (),
        torso_acceleration_history_enabled: bool = False,
        torso_headroom_enabled: bool = False,
        torso_headroom_fraction: float = CPU_MIN_BOUND_FRACTION,
        torso_headroom_activation_margin: float = CPU_MARGIN_THRESHOLD,
        torso_projection_iterations: int = 16,
        torso_feasibility_tolerance: float = 1.0e-6,
        posture_target_configuration: tuple[float, ...] | None = None,
        posture_velocity_indices: tuple[int, ...] = (),
        posture_weights: tuple[float, ...] = (),
        posture_gain: float = 1.0,
        posture_rank_tolerance: float = 1.0e-6,
        posture_priority: int = 1,
        secondary_frame_names: tuple[str, ...] = (),
        secondary_frame_task_dimensions: tuple[int, ...] = (),
        secondary_frame_target_poses_wxyz: tuple[tuple[float, ...], ...] = (),
        secondary_frame_position_gains: tuple[float, ...] = (),
        secondary_frame_orientation_gains: tuple[float, ...] = (),
        secondary_frame_weights: tuple[float, ...] = (),
        locked_velocity_indices: tuple[int, ...] = (),
        config: MultiFramePoseSolveConfig | None = None,
    ) -> None:
        import torch

        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        self.config = config or MultiFramePoseSolveConfig()
        self.batch_size = batch_size
        self.frames = tuple(frames)
        self.frame_count = len(self.frames)
        self.frame_task_dimensions = (
            (6,) * self.frame_count
            if frame_task_dimensions is None
            else tuple(int(value) for value in frame_task_dimensions)
        )
        if len(self.frame_task_dimensions) != self.frame_count or any(
            value not in (3, 6) for value in self.frame_task_dimensions
        ):
            raise ValueError("frame_task_dimensions must contain one 3 or 6 per frame")
        self.frame_contact_constraints = (
            (False,) * self.frame_count
            if frame_contact_constraints is None
            else tuple(bool(value) for value in frame_contact_constraints)
        )
        if len(self.frame_contact_constraints) != self.frame_count:
            raise ValueError("frame_contact_constraints must match frames")
        self.task_rows = sum(self.frame_task_dimensions)
        position_gains = (
            (self.config.position_gain,) * self.frame_count
            if frame_position_gains is None
            else tuple(float(value) for value in frame_position_gains)
        )
        orientation_gains = (
            (self.config.orientation_gain,) * self.frame_count
            if frame_orientation_gains is None
            else tuple(float(value) for value in frame_orientation_gains)
        )
        if (
            len(position_gains) != self.frame_count
            or len(orientation_gains) != self.frame_count
            or not all(math.isfinite(value) and value > 0.0 for value in position_gains)
            or not all(math.isfinite(value) and value >= 0.0 for value in orientation_gains)
        ):
            raise ValueError("per-frame gains must match frames and be finite")
        self.configuration_dim = (
            robot_spec.configuration_dim
            if robot_spec.floating_base
            else len(robot_spec.active_velocity_indices)
        )
        self.velocity_dim = len(robot_spec.active_velocity_indices)
        self.robot_spec = robot_spec
        self.target_dim = 7
        if self.frame_count < 1:
            raise ValueError("pose solver requires at least one frame")
        if tuple(robot_spec.task_frames) != self.frames:
            raise ValueError("ordered frames must match RobotSolveSpec.task_frames")
        if self.config.velocity_solver == "fi_pesns":
            if fi_function is None or fi_library is None:
                raise ValueError("fi_pesns requires a generated function and library")
            if self.frame_task_dimensions != (6,) * self.frame_count:
                raise ValueError("fi_pesns currently requires full 6D frame tasks")
            _validate_fi_function(fi_function, self.velocity_dim, self.frame_count)
        values = {
            "joint_lower": joint_lower,
            "joint_upper": joint_upper,
            "joint_velocity_limits": joint_velocity_limits,
        }
        for name, sequence in values.items():
            if len(sequence) != self.velocity_dim or not all(
                math.isfinite(value) for value in sequence
            ):
                raise ValueError(f"{name} must contain {self.velocity_dim} finite values")
        if any(lower >= upper for lower, upper in zip(joint_lower, joint_upper)):
            raise ValueError("joint lower limits must be below upper limits")
        if any(limit <= 0 for limit in joint_velocity_limits):
            raise ValueError("joint velocity limits must be positive")
        self.torso_constraint_enabled = _validate_torso_configuration(
            velocity_dim=self.velocity_dim,
            torso_frame_name=torso_frame_name,
            torso_reference_pose_xyzw=torso_reference_pose_xyzw,
            torso_lower_relative_limits=torso_lower_relative_limits,
            torso_upper_relative_limits=torso_upper_relative_limits,
            torso_axis_mask=torso_axis_mask,
            torso_velocity_limits=torso_velocity_limits,
            torso_acceleration_limits=torso_acceleration_limits,
            torso_excluded_active_velocity_indices=(torso_excluded_active_velocity_indices),
            torso_headroom_fraction=torso_headroom_fraction,
            torso_headroom_activation_margin=torso_headroom_activation_margin,
            torso_projection_iterations=torso_projection_iterations,
            torso_feasibility_tolerance=torso_feasibility_tolerance,
        )
        self.posture_task_enabled = _validate_posture_configuration(
            configuration_dim=self.configuration_dim,
            active_velocity_indices=tuple(robot_spec.active_velocity_indices),
            posture_target_configuration=posture_target_configuration,
            posture_velocity_indices=posture_velocity_indices,
            posture_weights=posture_weights,
            posture_gain=posture_gain,
            posture_rank_tolerance=posture_rank_tolerance,
            locked_velocity_indices=locked_velocity_indices,
        )
        self.secondary_frame_tasks_enabled = _validate_secondary_frame_configuration(
            names=secondary_frame_names,
            dimensions=secondary_frame_task_dimensions,
            target_poses_wxyz=secondary_frame_target_poses_wxyz,
            position_gains=secondary_frame_position_gains,
            orientation_gains=secondary_frame_orientation_gains,
            weights=secondary_frame_weights,
        )
        self.secondary_task_enabled = (
            self.posture_task_enabled or self.secondary_frame_tasks_enabled
        )
        active_lookup = {
            source_index: active_index
            for active_index, source_index in enumerate(robot_spec.active_velocity_indices)
        }
        self._locked_active_columns = tuple(
            active_lookup[int(source_index)] for source_index in locked_velocity_indices
        )
        self._unlocked_active_columns = tuple(
            index
            for index in range(self.velocity_dim)
            if index not in set(self._locked_active_columns)
        )
        if not self._unlocked_active_columns:
            raise ValueError("at least one active velocity must remain unlocked")
        self.torso_frame_name = torso_frame_name
        self._kinematics_frames = self.frames
        if self.torso_constraint_enabled and torso_frame_name not in self.frames:
            self._kinematics_frames += (torso_frame_name,)
        for secondary_frame in secondary_frame_names:
            if secondary_frame not in self._kinematics_frames:
                self._kinematics_frames += (secondary_frame,)
        self._torso_frame_index = (
            self._kinematics_frames.index(torso_frame_name)
            if self.torso_constraint_enabled
            else None
        )
        self._secondary_frame_indices = tuple(
            self._kinematics_frames.index(frame) for frame in secondary_frame_names
        )
        kinematics_spec = replace(robot_spec, task_frames=self._kinematics_frames)

        self.torch = torch
        self.device = torch.device("cuda:0")
        from .warp_publication import WarpCompactPublication

        self._compact_publication_packer = WarpCompactPublication(
            self.batch_size,
            self.configuration_dim,
            self.frame_count,
            device=self.device,
        )
        self._locked_active_columns_tensor = torch.tensor(
            self._locked_active_columns, dtype=torch.long, device=self.device
        )
        self._unlocked_active_columns_tensor = torch.tensor(
            self._unlocked_active_columns, dtype=torch.long, device=self.device
        )
        self._locked_velocity_mask = torch.zeros(
            self.velocity_dim, dtype=torch.bool, device=self.device
        )
        if self._locked_active_columns:
            self._locked_velocity_mask[list(self._locked_active_columns)] = True
        if not torch.cuda.is_available():
            raise RuntimeError("DeviceResidentMultiFramePoseSolver requires CUDA")
        self._fi_dtype = torch.float64 if self.config.fi_scalar_type == "float64" else torch.float32
        self.kinematics = NewtonModelKinematics(
            batch_size,
            urdf_path,
            cache_dir,
            kinematics_spec,
            self._kinematics_frames,
            default_configuration=default_configuration,
            device=str(self.device),
        )
        self.com = None
        self.com_constraint_enabled = com_support_polygon_xy is not None
        self.capture_point_constraint_enabled = capture_point_support_polygon_xy is not None
        self.velocity_zmp_constraint_enabled = velocity_zmp_support_polygon_xy is not None
        momentum_mask = tuple(bool(value) for value in centroidal_momentum_axis_mask)
        if momentum_mask and len(momentum_mask) != 6:
            raise ValueError("centroidal momentum axis mask must contain six values")
        self.centroidal_momentum_task_enabled = centroidal_momentum_target is not None and any(
            momentum_mask
        )
        self.centroidal_momentum_bounds_enabled = (
            centroidal_momentum_lower is not None or centroidal_momentum_upper is not None
        )
        if self.centroidal_momentum_bounds_enabled and (
            centroidal_momentum_lower is None or centroidal_momentum_upper is None
        ):
            raise ValueError("centroidal momentum bounds require lower and upper")
        capacities = {
            "com_max_constraints": com_max_constraints,
            "capture_point_max_constraints": capture_point_max_constraints,
            "velocity_zmp_max_constraints": velocity_zmp_max_constraints,
        }
        for name, capacity in capacities.items():
            if isinstance(capacity, bool) or not isinstance(capacity, int) or capacity < 3:
                raise ValueError(f"{name} must be an integer >= 3")
        if (
            isinstance(com_projection_iterations, bool)
            or not isinstance(com_projection_iterations, int)
            or com_projection_iterations < 1
            or isinstance(centroidal_projection_iterations, bool)
            or not isinstance(centroidal_projection_iterations, int)
            or centroidal_projection_iterations < 1
        ):
            raise ValueError("centroidal projection iterations must be positive integers")
        self._com_projection_iterations = com_projection_iterations
        self._centroidal_projection_iterations = centroidal_projection_iterations
        self.centroidal_enabled = bool(
            self.com_constraint_enabled
            or self.capture_point_constraint_enabled
            or self.velocity_zmp_constraint_enabled
            or self.centroidal_momentum_task_enabled
            or self.centroidal_momentum_bounds_enabled
        )
        if self.centroidal_enabled:
            from .newton_com import NewtonCoMEvaluator

            com_spec = robot_spec if com_robot_spec is None else com_robot_spec
            com_default = (
                default_configuration
                if com_default_configuration is None
                else com_default_configuration
            )
            if (
                com_spec.model_hash != robot_spec.model_hash
                or com_spec.floating_base != robot_spec.floating_base
                or com_spec.active_joint_names != robot_spec.active_joint_names
                or len(com_spec.active_velocity_indices) != self.velocity_dim
            ):
                raise ValueError(
                    "centroidal model must retain the solver's active joint order and root kind"
                )
            if len(com_default) != com_spec.configuration_dim or not all(
                math.isfinite(v) for v in com_default
            ):
                raise ValueError("centroidal default must match its complete model configuration")
            if robot_spec.floating_base and com_spec != robot_spec:
                raise ValueError("floating centroidal model must match the solver model")
            self.com = NewtonCoMEvaluator(
                batch_size,
                urdf_path,
                cache_dir / "com",
                com_spec,
                device=str(self.device),
            )
            self._com_default = (
                torch.tensor(com_default, dtype=torch.float32, device=self.device)
                .expand(batch_size, -1)
                .clone()
            )
            self._com_q_indices = torch.tensor(
                com_spec.active_configuration_indices or com_spec.active_velocity_indices,
                dtype=torch.long,
                device=self.device,
            )
            self._centroidal_active_velocity_indices = torch.tensor(
                com_spec.active_velocity_indices,
                dtype=torch.long,
                device=self.device,
            )
            self._centroidal_full_velocity = torch.zeros(
                (batch_size, com_spec.velocity_dim),
                dtype=torch.float32,
                device=self.device,
            )
            active_indices = tuple(robot_spec.active_velocity_indices)
            self._com_a = torch.zeros(
                (batch_size, com_max_constraints, 2),
                dtype=torch.float64,
                device=self.device,
            )
            self._com_b = torch.zeros(
                (batch_size, com_max_constraints),
                dtype=torch.float64,
                device=self.device,
            )
            self._com_active = torch.zeros_like(self._com_b, dtype=torch.bool)
            self._com_proximity = torch.zeros(
                (batch_size, 1), dtype=torch.float64, device=self.device
            )
            self._com_settings = torch.zeros(3, dtype=torch.float64, device=self.device)
            excluded = tuple(com_excluded_velocity_indices)
            if len(set(excluded)) != len(excluded) or any(
                i not in active_indices for i in excluded
            ):
                raise ValueError("CoM excluded indices must be unique active source velocities")
            self._com_excluded = torch.tensor(
                [i in excluded for i in active_indices],
                dtype=torch.bool,
                device=self.device,
            )
            self._capture_a = torch.zeros(
                (batch_size, capture_point_max_constraints, 2),
                dtype=torch.float64,
                device=self.device,
            )
            self._capture_b = torch.zeros(
                (batch_size, capture_point_max_constraints),
                dtype=torch.float64,
                device=self.device,
            )
            self._capture_active = torch.zeros_like(self._capture_b, dtype=torch.bool)
            self._capture_settings = torch.tensor(
                (
                    -1.0 if capture_point_omega is None else capture_point_omega,
                    -1.0 if capture_point_height is None else capture_point_height,
                    capture_point_gravity_z,
                ),
                dtype=torch.float64,
                device=self.device,
            )
            self._zmp_a = torch.zeros(
                (batch_size, velocity_zmp_max_constraints, 2),
                dtype=torch.float64,
                device=self.device,
            )
            self._zmp_b = torch.zeros(
                (batch_size, velocity_zmp_max_constraints),
                dtype=torch.float64,
                device=self.device,
            )
            self._zmp_active = torch.zeros_like(self._zmp_b, dtype=torch.bool)
            self._zmp_settings = torch.tensor(
                (velocity_zmp_fz_min, velocity_zmp_gravity_z),
                dtype=torch.float64,
                device=self.device,
            )
            self._momentum_target = torch.zeros(
                (batch_size, 6), dtype=torch.float32, device=self.device
            )
            self._momentum_axis_mask = torch.tensor(
                momentum_mask or (False,) * 6, dtype=torch.bool, device=self.device
            )
            self._momentum_weight = torch.tensor(
                centroidal_momentum_weight, dtype=torch.float32, device=self.device
            )
            self._momentum_priority = int(centroidal_momentum_priority)
            if self._momentum_priority not in (1, 2):
                raise ValueError("centroidal momentum priority must be 1 or 2")
            momentum_excluded = tuple(centroidal_momentum_excluded_velocity_indices)
            if len(set(momentum_excluded)) != len(momentum_excluded) or any(
                index not in active_indices for index in momentum_excluded
            ):
                raise ValueError(
                    "centroidal momentum exclusions must be unique active source velocities"
                )
            self._momentum_excluded = torch.tensor(
                [index in momentum_excluded for index in active_indices],
                dtype=torch.bool,
                device=self.device,
            )
            self._momentum_lower = torch.full(
                (batch_size, 6), -1.0e100, dtype=torch.float64, device=self.device
            )
            self._momentum_upper = torch.full_like(self._momentum_lower, 1.0e100)
            self._momentum_bound_active = torch.zeros(
                (batch_size, 6), dtype=torch.bool, device=self.device
            )
            if self.com_constraint_enabled:
                self.configure_com_constraint(
                    support_polygon_xy=com_support_polygon_xy,
                    margin=com_margin,
                    vel_max=com_vel_max,
                    acc_max=com_acc_max,
                    use_acceleration_limits=com_use_acceleration_limits,
                    proximity_fraction=com_proximity_fraction,
                )
            if self.capture_point_constraint_enabled:
                self.configure_capture_point_constraint(
                    support_polygon_xy=capture_point_support_polygon_xy,
                    margin=capture_point_margin,
                    omega=capture_point_omega,
                    height=capture_point_height,
                    gravity_z=capture_point_gravity_z,
                )
            if self.velocity_zmp_constraint_enabled:
                self.configure_velocity_zmp_constraint(
                    support_polygon_xy=velocity_zmp_support_polygon_xy,
                    margin=velocity_zmp_margin,
                    fz_min=velocity_zmp_fz_min,
                    gravity_z=velocity_zmp_gravity_z,
                )
            if self.centroidal_momentum_task_enabled or self.centroidal_momentum_bounds_enabled:
                self.configure_centroidal_momentum(
                    enabled=self.centroidal_momentum_task_enabled,
                    target=centroidal_momentum_target,
                    axis_mask=momentum_mask or (False,) * 6,
                    weight=centroidal_momentum_weight,
                    lower=centroidal_momentum_lower,
                    upper=centroidal_momentum_upper,
                )
        self.secondary_task_enabled = bool(
            self.secondary_task_enabled or self.centroidal_momentum_task_enabled
        )
        self.fi = (
            StrictCusadiFunction(
                fi_function,
                fi_library,
                batch_size,
                device=str(self.device),
                scalar_type=self.config.fi_scalar_type,
            )
            if self.config.velocity_solver == "fi_pesns"
            else None
        )
        self._warp_srinv = None
        self._warp_contact_inverse = None
        self._cusolver_srinv = None
        self._cusolver_contact_inverse = None
        self._cusolver_contact_rhs = None
        self._cusolver_primary_inverse = None
        self._cusolver_primary_status = None
        self._cusolver_locked_primary_reuse_certified = None
        self._warp_com_projection = None
        self._warp_torso_projection = None
        self._warp_priority_solvers = {}
        self._cusolver_priority_solvers = {}
        if self.centroidal_enabled or self.torso_constraint_enabled:
            from .warp_constraints import WarpCyclicRowProjection
        if self.config.velocity_solver == "warp_srinv":
            from .warp_directional_srinv import WarpDirectionalSRINV

            self._warp_srinv = WarpDirectionalSRINV(
                self.task_rows,
                len(self._unlocked_active_columns),
                batch_capacity=batch_size,
                tolerance=self.config.srinv_tolerance,
                damping=self.config.srinv_damping,
                relative_rank_tolerance=posture_rank_tolerance,
                device=str(self.device),
            )
            contact_rows = sum(
                dimension
                for dimension, constrained in zip(
                    self.frame_task_dimensions,
                    self.frame_contact_constraints,
                    strict=True,
                )
                if constrained
            )
            if contact_rows:
                self._warp_contact_inverse = WarpDirectionalSRINV(
                    contact_rows,
                    self.velocity_dim,
                    batch_capacity=batch_size,
                    inverse_mode="undamped_relative",
                    relative_rank_tolerance=(
                        max(contact_rows, self.velocity_dim) * torch.finfo(torch.float64).eps
                    ),
                    device=str(self.device),
                )
        if self.config.velocity_solver == "cusolver_srinv":
            from .cusolver_gesvdj import CuSolverDirectionalSRINV

            self._cusolver_srinv = CuSolverDirectionalSRINV(
                self.task_rows,
                len(self._unlocked_active_columns),
                batch_capacity=batch_size,
                tolerance=self.config.srinv_tolerance,
                damping=self.config.srinv_damping,
                relative_rank_tolerance=posture_rank_tolerance,
                output_mode=(
                    "action_undamped"
                    if self.config.cusolver_specialized_outputs_enabled
                    else "full"
                ),
                fused_status_enabled=bool(
                    self.config.cusolver_specialized_outputs_enabled
                    and self.config.native_compile_enabled
                    and _python_extension_headers_available()
                ),
                profile_label="task_primary",
                device=str(self.device),
            )
            contact_rows = sum(
                dimension
                for dimension, constrained in zip(
                    self.frame_task_dimensions,
                    self.frame_contact_constraints,
                    strict=True,
                )
                if constrained
            )
            if contact_rows:
                self._cusolver_contact_inverse = CuSolverDirectionalSRINV(
                    contact_rows,
                    self.velocity_dim,
                    batch_capacity=batch_size,
                    relative_rank_tolerance=(
                        max(contact_rows, self.velocity_dim) * torch.finfo(torch.float64).eps
                    ),
                    output_mode=(
                        "action_undamped"
                        if self.config.cusolver_specialized_outputs_enabled
                        else "full"
                    ),
                    fused_status_enabled=bool(
                        self.config.cusolver_specialized_outputs_enabled
                        and self.config.native_compile_enabled
                        and _python_extension_headers_available()
                    ),
                    profile_label="contact_inverse",
                    device=str(self.device),
                )
                self._cusolver_contact_rhs = torch.zeros(
                    (batch_size, contact_rows),
                    dtype=torch.float32,
                    device=self.device,
                )
        if self.centroidal_enabled:
            self._centroidal_row_capacity = (
                com_max_constraints
                + capture_point_max_constraints
                + velocity_zmp_max_constraints
                + 1
                + 6
            )
            self._warp_com_projection = WarpCyclicRowProjection(
                self._centroidal_row_capacity,
                self.velocity_dim,
                iterations=self._centroidal_projection_iterations,
                batch_capacity=batch_size,
                feasibility_tolerance=1.0e-6,
                device=str(self.device),
            )
        if self.torso_constraint_enabled:
            self._warp_torso_projection = WarpCyclicRowProjection(
                6,
                self.velocity_dim,
                iterations=torso_projection_iterations,
                batch_capacity=batch_size,
                feasibility_tolerance=torso_feasibility_tolerance,
                device=str(self.device),
            )
        self._constraints = (
            torch.eye(self.velocity_dim, dtype=self._fi_dtype, device=self.device)
            .expand(batch_size, -1, -1)
            .contiguous()
        )
        limits = torch.tensor(
            joint_velocity_limits, dtype=torch.float64, device=self.device
        ).expand(batch_size, -1)
        self._lower_velocity = -limits.contiguous()
        self._upper_velocity = limits.contiguous()
        self._joint_lower = torch.tensor(joint_lower, dtype=torch.float32, device=self.device)
        self._joint_upper = torch.tensor(joint_upper, dtype=torch.float32, device=self.device)
        self._orientation_frame_mask = torch.tensor(
            [dimension == 6 for dimension in self.frame_task_dimensions],
            dtype=torch.bool,
            device=self.device,
        )
        self._frame_position_gains = torch.tensor(
            position_gains, dtype=torch.float32, device=self.device
        )
        self._frame_orientation_gains = torch.tensor(
            orientation_gains, dtype=torch.float32, device=self.device
        )
        if robot_spec.floating_base:
            self._posture_joint_configuration_indices = robot_spec.joint_configuration_indices
            self._posture_joint_configuration_sizes = robot_spec.joint_configuration_sizes
            self._posture_joint_velocity_indices = robot_spec.joint_velocity_indices
            self._posture_joint_velocity_sizes = robot_spec.joint_velocity_sizes
        else:
            # Fixed-base requests carry only active scalar q coordinates. Build
            # equivalent model-source spans whose q indices address that compact
            # input while velocity starts retain their original model indices.
            self._posture_joint_configuration_indices = tuple(range(self.velocity_dim))
            self._posture_joint_configuration_sizes = (1,) * self.velocity_dim
            self._posture_joint_velocity_indices = tuple(robot_spec.active_velocity_indices)
            self._posture_joint_velocity_sizes = (1,) * self.velocity_dim
        self._posture_target = None
        self._posture_velocity_indices = tuple(int(index) for index in posture_velocity_indices)
        self._posture_weights = None
        self._posture_q_indices = None
        self._posture_jacobian = None
        self._posture_gain_device = None
        self._posture_gain = float(posture_gain)
        self._posture_runtime_enabled = False
        self._posture_rank_tolerance = float(posture_rank_tolerance)
        if self.posture_task_enabled:
            assert posture_target_configuration is not None
            self._posture_target = torch.tensor(
                posture_target_configuration, dtype=torch.float32, device=self.device
            )
            self._posture_weights = torch.tensor(
                posture_weights, dtype=torch.float32, device=self.device
            )
            posture_q_indices = _configuration_indices_for_velocity_rows(
                selected_source_velocity_indices=self._posture_velocity_indices,
                floating_base=bool(robot_spec.floating_base),
                joint_configuration_indices=self._posture_joint_configuration_indices,
                joint_configuration_sizes=self._posture_joint_configuration_sizes,
                joint_velocity_indices=self._posture_joint_velocity_indices,
                joint_velocity_sizes=self._posture_joint_velocity_sizes,
                configuration_dim=self.configuration_dim,
            )
            self._posture_q_indices = torch.tensor(
                posture_q_indices, dtype=torch.long, device=self.device
            )
            posture_columns = torch.tensor(
                [active_lookup[index] for index in self._posture_velocity_indices],
                dtype=torch.long,
                device=self.device,
            )
            self._posture_jacobian = torch.zeros(
                (len(posture_q_indices), self.velocity_dim),
                dtype=torch.float32,
                device=self.device,
            )
            self._posture_jacobian[
                torch.arange(len(posture_q_indices), device=self.device),
                posture_columns,
            ] = self._posture_weights
            self._posture_gain_device = torch.tensor(
                self._posture_gain, dtype=torch.float32, device=self.device
            )
            self._posture_runtime_enabled = bool(
                self._posture_gain > 0.0 and any(weight > 0.0 for weight in posture_weights)
            )
        self._secondary_frame_task_dimensions = tuple(
            int(value) for value in secondary_frame_task_dimensions
        )
        self.configure_task_hierarchy(posture_priority=posture_priority)
        self._secondary_frame_targets = None
        self._secondary_frame_position_gains = None
        self._secondary_frame_orientation_gains = None
        self._secondary_frame_weights = None
        if self.secondary_frame_tasks_enabled:
            self._secondary_frame_targets = torch.tensor(
                secondary_frame_target_poses_wxyz,
                dtype=torch.float32,
                device=self.device,
            )
            self._secondary_frame_position_gains = torch.tensor(
                secondary_frame_position_gains,
                dtype=torch.float32,
                device=self.device,
            )
            self._secondary_frame_orientation_gains = torch.tensor(
                secondary_frame_orientation_gains,
                dtype=torch.float32,
                device=self.device,
            )
            self._secondary_frame_weights = torch.tensor(
                secondary_frame_weights,
                dtype=torch.float32,
                device=self.device,
            )
        self._torso_reference_pose = None
        self._torso_lower_relative_limits = None
        self._torso_upper_relative_limits = None
        self._torso_axis_mask = None
        self._torso_velocity_limits = None
        self._torso_acceleration_limits = None
        self._torso_excluded_columns = None
        self._torso_tolerance = None
        self._torso_nominal_dt = None
        self._torso_acceleration_history_enabled_device = None
        self._torso_headroom_enabled_device = None
        self._torso_headroom_fraction_device = None
        self._torso_headroom_activation_margin_device = None
        self._torso_acceleration_history_enabled = torso_acceleration_history_enabled
        self._torso_headroom_enabled = torso_headroom_enabled
        self._torso_headroom_fraction = torso_headroom_fraction
        self._torso_headroom_activation_margin = torso_headroom_activation_margin
        self._torso_projection_iterations = torso_projection_iterations
        self._torso_feasibility_tolerance = torso_feasibility_tolerance
        if self.torso_constraint_enabled:
            assert torso_reference_pose_xyzw is not None
            assert torso_lower_relative_limits is not None
            assert torso_upper_relative_limits is not None
            assert torso_axis_mask is not None
            assert torso_velocity_limits is not None
            assert torso_acceleration_limits is not None
            self._torso_reference_pose = torch.tensor(
                torso_reference_pose_xyzw, dtype=torch.float64, device=self.device
            )
            self._torso_lower_relative_limits = torch.tensor(
                torso_lower_relative_limits, dtype=torch.float64, device=self.device
            )
            self._torso_upper_relative_limits = torch.tensor(
                torso_upper_relative_limits, dtype=torch.float64, device=self.device
            )
            self._torso_axis_mask = torch.tensor(
                torso_axis_mask, dtype=torch.bool, device=self.device
            )
            self._torso_velocity_limits = torch.tensor(
                torso_velocity_limits, dtype=torch.float64, device=self.device
            )
            self._torso_acceleration_limits = torch.tensor(
                torso_acceleration_limits, dtype=torch.float64, device=self.device
            )
            self._torso_excluded_columns = torch.zeros(
                self.velocity_dim, dtype=torch.bool, device=self.device
            )
            self._torso_excluded_columns[list(torso_excluded_active_velocity_indices)] = True
            self._torso_tolerance = torch.tensor(
                (CPU_TORSO_BOUND_SLACK_EPS_TRANS,) * 3 + (CPU_TORSO_BOUND_SLACK_EPS_ROT,) * 3,
                dtype=torch.float64,
                device=self.device,
            )
            # Python scalar conversion can enqueue host-to-device copies during
            # CUDA graph capture. Keep the static policy values resident.
            self._torso_nominal_dt = torch.tensor(
                self.config.dt, dtype=torch.float64, device=self.device
            )
            self._torso_acceleration_history_enabled_device = torch.tensor(
                self._torso_acceleration_history_enabled,
                dtype=torch.bool,
                device=self.device,
            )
            self._torso_headroom_enabled_device = torch.tensor(
                self._torso_headroom_enabled, dtype=torch.bool, device=self.device
            )
            self._torso_headroom_fraction_device = torch.tensor(
                self._torso_headroom_fraction,
                dtype=torch.float64,
                device=self.device,
            )
            self._torso_headroom_activation_margin_device = torch.tensor(
                self._torso_headroom_activation_margin,
                dtype=torch.float64,
                device=self.device,
            )
        self._position_velocity_indices: list[int] = []
        self._position_configuration_indices: list[int] = []
        if robot_spec.floating_base:
            active_lookup = {
                source_index: active_index
                for active_index, source_index in enumerate(robot_spec.active_velocity_indices)
            }
            for q_index, q_size, v_index, v_size in zip(
                robot_spec.joint_configuration_indices,
                robot_spec.joint_configuration_sizes,
                robot_spec.joint_velocity_indices,
                robot_spec.joint_velocity_sizes,
                strict=True,
            ):
                if q_size == 1 and v_size == 1 and v_index in active_lookup:
                    self._position_velocity_indices.append(active_lookup[v_index])
                    self._position_configuration_indices.append(q_index)
        self._position_velocity_indices_tensor = torch.tensor(
            self._position_velocity_indices, dtype=torch.long, device=self.device
        )
        self._position_configuration_indices_tensor = torch.tensor(
            self._position_configuration_indices,
            dtype=torch.long,
            device=self.device,
        )
        self._graph = None
        self._graph_inputs = None
        self._graph_result = None
        self._compiled_task_state = None
        self._compiled_errors = None
        self._compiled_bounds = None
        self._compiled_torso_rows = None
        self._compiled_finalization = None
        self._compiled_native_velocity = None
        self._compiled_secondary_tasks = None
        self.collision = None
        self.collision_convex_envelope = None
        self._compiled_collision_constraint = None
        self._compiled_collision_rows = None
        compile_requested = self.config.collision_enabled or self.config.native_compile_enabled
        compile_available = not compile_requested or _python_extension_headers_available()
        if compile_requested and not compile_available:
            warnings.warn(
                "Python development headers are unavailable; disabling optional "
                "torch.compile kernels and retaining the eager CUDA solver. Install "
                "Python.h or use toolchains/sm120/ensure_python_dev_headers.sh to "
                "restore compiled performance.",
                RuntimeWarning,
                stacklevel=2,
            )
        if self.config.collision_enabled:
            if not collision_pairs:
                raise ValueError("collision_enabled requires collision_pairs")
            from .newton_collision import NewtonModelCollisionQuery

            self.collision = NewtonModelCollisionQuery(
                batch_size,
                urdf_path,
                robot_spec=robot_spec,
                collision_pairs=tuple(collision_pairs),
                default_configuration=tuple(default_configuration),
                cache_dir=cache_dir / "collision-model",
                device=str(self.device),
                query_distance_m=self.config.collision_query_distance_m,
                contacts_per_world=self.config.collision_contacts_per_world,
                triangle_pairs_per_world=(self.config.collision_triangle_pairs_per_world),
                sort_contacts=self.config.collision_contact_sort_enabled,
            )
            if (
                self.config.collision_candidate_convex_certificate_enabled
                or self.config.collision_current_convex_certificate_enabled
            ):
                self.collision_convex_envelope = NewtonModelCollisionQuery(
                    batch_size,
                    urdf_path,
                    robot_spec=robot_spec,
                    collision_pairs=tuple(collision_pairs),
                    default_configuration=tuple(default_configuration),
                    cache_dir=cache_dir / "collision-convex-envelope",
                    device=str(self.device),
                    query_distance_m=self.config.collision_query_distance_m,
                    contacts_per_world=1,
                    triangle_pairs_per_world=1,
                    deterministic=False,
                    sort_contacts=False,
                    _convex_hull_mode=True,
                )
            if compile_available:
                self._compiled_collision_constraint = torch.compile(
                    self._apply_collision_constraint_eager,
                    fullgraph=True,
                    dynamic=False,
                )
                self._compiled_collision_rows = torch.compile(
                    self._apply_collision_rows_eager,
                    fullgraph=True,
                    dynamic=False,
                    options={"triton.cudagraphs": False},
                )
        if self.config.native_compile_enabled and compile_available:
            compile_options = {"triton.cudagraphs": False}
            self._compiled_task_state = torch.compile(
                self._task_state_eager,
                fullgraph=True,
                dynamic=False,
                options=compile_options,
            )
            self._compiled_errors = torch.compile(
                self._errors_eager,
                fullgraph=True,
                dynamic=False,
                options=compile_options,
            )
            self._compiled_bounds = torch.compile(
                self._bounds_eager,
                fullgraph=True,
                dynamic=False,
                options=compile_options,
            )
            if self.torso_constraint_enabled:
                self._compiled_torso_rows = torch.compile(
                    self._torso_rows_eager,
                    fullgraph=True,
                    dynamic=False,
                    options=compile_options,
                )
            # Centroidal final validation evaluates Newton/Warp kinematics.
            # Keep that path eager so the outer CUDA graph records the Warp
            # launches directly instead of asking Dynamo to trace its driver
            # context manager.
            if not self.centroidal_enabled:
                self._compiled_finalization = torch.compile(
                    self._finalize_publication_eager,
                    fullgraph=True,
                    dynamic=False,
                    options=compile_options,
                )
        if (
            self.config.velocity_solver == "torch_srinv"
            and self.config.native_compile_enabled
            and compile_available
        ):
            compile_options = {"triton.cudagraphs": False}
            self._compiled_native_velocity = torch.compile(
                self._native_velocity_eager,
                fullgraph=True,
                dynamic=False,
                options=compile_options,
            )
            self._compiled_secondary_tasks = torch.compile(
                self._apply_secondary_tasks_eager,
                fullgraph=True,
                dynamic=False,
                options=compile_options,
            )

    def _validate(
        self,
        q: Any,
        target: Any,
        previous_velocity: Any | None,
        current_velocity: Any | None,
    ) -> None:
        torch = self.torch
        expected = {
            "q": (q, (self.batch_size, self.configuration_dim)),
            "target": (target, (self.batch_size, self.frame_count, 7)),
        }
        for name, (tensor, shape) in expected.items():
            if not isinstance(tensor, torch.Tensor):
                raise TypeError(f"{name} must be a torch.Tensor")
            if tensor.shape != shape or tensor.device != self.device:
                raise ValueError(f"{name} must have shape {shape} on {self.device}")
            if tensor.dtype is not torch.float32:
                raise ValueError(f"{name} must use float32")
        if previous_velocity is not None and (
            not isinstance(previous_velocity, torch.Tensor)
            or previous_velocity.shape != (self.batch_size, self.velocity_dim)
            or previous_velocity.device != self.device
            or previous_velocity.dtype is not torch.float32
        ):
            raise ValueError("previous_velocity must match the active velocity shape")
        if current_velocity is not None and (
            not isinstance(current_velocity, torch.Tensor)
            or current_velocity.shape != (self.batch_size, self.velocity_dim)
            or current_velocity.device != self.device
            or current_velocity.dtype is not torch.float32
        ):
            raise ValueError("current_velocity must match the active velocity shape")
        if self.velocity_zmp_constraint_enabled and current_velocity is None:
            raise ValueError("velocity ZMP requires explicit current_velocity")
        if self.robot_spec.floating_base:
            q_position = q[:, self._position_configuration_indices]
            lower_position = self._joint_lower[self._position_velocity_indices]
            upper_position = self._joint_upper[self._position_velocity_indices]
            position_limits_valid = _floating_position_limits_valid(
                q_position,
                lower_position,
                upper_position,
                self._position_velocity_indices,
                self._locked_active_columns,
            )
            root = next(
                index
                for index, (q_size, v_size) in enumerate(
                    zip(
                        self.robot_spec.joint_configuration_sizes,
                        self.robot_spec.joint_velocity_sizes,
                        strict=True,
                    )
                )
                if q_size == 7 and v_size == 6
            )
            root_q = self.robot_spec.joint_configuration_indices[root]
            quaternion_valid = (
                torch.linalg.vector_norm(q[:, root_q + 3 : root_q + 7], dim=-1) > 1e-8
            ).all()
        else:
            position_limits_valid = ((q >= self._joint_lower) & (q <= self._joint_upper)).all()
            quaternion_valid = torch.tensor(True, device=self.device)
        checks = torch.stack(
            (
                torch.isfinite(q).all(),
                torch.isfinite(target).all(),
                (
                    torch.tensor(True, device=self.device)
                    if previous_velocity is None
                    else torch.isfinite(previous_velocity).all()
                ),
                (
                    torch.tensor(True, device=self.device)
                    if current_velocity is None
                    else torch.isfinite(current_velocity).all()
                ),
                (torch.linalg.vector_norm(target[:, :, 3:], dim=-1) > 1e-8).all(),
                position_limits_valid,
                quaternion_valid,
            )
        ).tolist()
        if not all(checks):
            raise ValueError("multi-frame request contains invalid values or limits")

    def _bounds(self, q: Any, q_start: Any, previous: Any, effective_dt: Any) -> tuple[Any, Any]:
        if self._compiled_bounds is not None:
            return self._compiled_bounds(q, q_start, previous, effective_dt)
        return self._bounds_eager(q, q_start, previous, effective_dt)

    def _bounds_eager(
        self, q: Any, q_start: Any, previous: Any, effective_dt: Any
    ) -> tuple[Any, Any]:
        torch = self.torch
        horizon = effective_dt[:, None].to(torch.float64)
        if self.robot_spec.floating_base:
            lower = self._lower_velocity.clone()
            upper = self._upper_velocity.clone()
            velocity_indices = self._position_velocity_indices_tensor
            configuration_indices = self._position_configuration_indices_tensor
            scalar_q = q.index_select(1, configuration_indices)
            scalar_lower = self._joint_lower.index_select(0, velocity_indices)
            scalar_upper = self._joint_upper.index_select(0, velocity_indices)
            lower.index_copy_(
                1,
                velocity_indices,
                torch.maximum(
                    lower.index_select(1, velocity_indices),
                    (scalar_lower - scalar_q).to(torch.float64) / horizon,
                ),
            )
            upper.index_copy_(
                1,
                velocity_indices,
                torch.minimum(
                    upper.index_select(1, velocity_indices),
                    (scalar_upper - scalar_q).to(torch.float64) / horizon,
                ),
            )
            trust_velocity = self.config.max_joint_step_rad / horizon
            lower = torch.maximum(lower, -trust_velocity)
            upper = torch.minimum(upper, trust_velocity)
            if self.config.max_joint_acceleration_rad_s2 is not None:
                delta = self.config.max_joint_acceleration_rad_s2 * self.config.dt
                lower = torch.maximum(lower, (previous - delta).to(torch.float64))
                upper = torch.minimum(upper, (previous + delta).to(torch.float64))
            return self._apply_locked_velocity_bounds(lower, upper)
        lower = torch.maximum(
            self._lower_velocity,
            (self._joint_lower - q).to(torch.float64) / horizon,
        )
        upper = torch.minimum(
            self._upper_velocity,
            (self._joint_upper - q).to(torch.float64) / horizon,
        )
        lower = torch.maximum(
            lower,
            (q_start - self.config.max_joint_step_rad - q).to(torch.float64) / horizon,
        )
        upper = torch.minimum(
            upper,
            (q_start + self.config.max_joint_step_rad - q).to(torch.float64) / horizon,
        )
        if self.config.max_joint_acceleration_rad_s2 is not None:
            delta = self.config.max_joint_acceleration_rad_s2 * self.config.dt
            lower = torch.maximum(lower, (previous - delta).to(torch.float64))
            upper = torch.minimum(upper, (previous + delta).to(torch.float64))
        return self._apply_locked_velocity_bounds(lower, upper)

    def _apply_locked_velocity_bounds(self, lower: Any, upper: Any) -> tuple[Any, Any]:
        if self._locked_active_columns:
            lower.index_fill_(1, self._locked_active_columns_tensor, 0.0)
            upper.index_fill_(1, self._locked_active_columns_tensor, 0.0)
        return lower.contiguous(), upper.contiguous()

    def _native_velocity_eager(self, twist: Any, jacobian: Any, lower: Any, upper: Any):
        torch = self.torch
        matrix = jacobian.to(self._fi_dtype)
        if self._locked_active_columns:
            matrix = matrix.index_select(2, self._unlocked_active_columns_tensor)
        inverse = _directional_srinv(
            torch,
            matrix,
            self.config.srinv_tolerance,
            self.config.srinv_damping,
        )
        solved_velocity = (inverse @ twist.to(self._fi_dtype).unsqueeze(-1)).squeeze(-1)
        if self._locked_active_columns:
            velocity = torch.zeros(
                (self.batch_size, self.velocity_dim),
                dtype=self._fi_dtype,
                device=self.device,
            )
            velocity.index_copy_(1, self._unlocked_active_columns_tensor, solved_velocity)
        else:
            velocity = solved_velocity
        lower = lower.to(self._fi_dtype)
        upper = upper.to(self._fi_dtype)
        needs_upper_scale = velocity > upper
        needs_lower_scale = velocity < lower
        scales = torch.ones_like(velocity)
        scales = torch.where(needs_upper_scale, upper / torch.clamp(velocity, min=1e-30), scales)
        scales = torch.where(needs_lower_scale, lower / torch.clamp(velocity, max=-1e-30), scales)
        scale = torch.clamp(torch.amin(scales, dim=-1), min=0.0, max=1.0)
        return (velocity * scale[:, None]).to(torch.float32)

    def _native_velocity(self, twist: Any, jacobian: Any, lower: Any, upper: Any):
        if getattr(self, "_cusolver_srinv", None) is not None:
            torch = self.torch
            matrix = jacobian
            if self._locked_active_columns:
                matrix = matrix.index_select(2, self._unlocked_active_columns_tensor)
            outputs = self._cusolver_srinv.solve(matrix.contiguous(), twist.contiguous())
            solved_velocity = outputs[0]
            self._cusolver_primary_status = outputs[3]
            if self.config.cusolver_reuse_locked_primary_inverse_enabled:
                singular_values = self._cusolver_srinv.factorization.singular_values[
                    : self.batch_size
                ]
                maximum = singular_values.amax(dim=-1)
                minimum = singular_values.amin(dim=-1)
                self._cusolver_locked_primary_reuse_certified = (
                    torch.isfinite(singular_values).all(dim=-1)
                    & (maximum > 0.0)
                    & (
                        minimum
                        > self.config.cusolver_locked_primary_reuse_min_singular_ratio * maximum
                    )
                )
                self._cusolver_primary_status = torch.where(
                    self._cusolver_locked_primary_reuse_certified,
                    self._cusolver_primary_status,
                    torch.ones_like(self._cusolver_primary_status),
                )
            if self._locked_active_columns:
                velocity = torch.zeros(
                    (self.batch_size, self.velocity_dim),
                    dtype=torch.float32,
                    device=self.device,
                )
                velocity.index_copy_(1, self._unlocked_active_columns_tensor, solved_velocity)
                if self.config.cusolver_reuse_locked_primary_inverse_enabled:
                    full_inverse = torch.zeros(
                        (
                            self.batch_size,
                            self.velocity_dim,
                            self.task_rows,
                        ),
                        dtype=torch.float32,
                        device=self.device,
                    )
                    full_inverse.index_copy_(
                        1,
                        self._unlocked_active_columns_tensor,
                        outputs[2],
                    )
                    self._cusolver_primary_inverse = full_inverse
                else:
                    self._cusolver_primary_inverse = outputs[2]
            else:
                velocity = solved_velocity
                self._cusolver_primary_inverse = outputs[2]
            lower32, upper32 = lower.to(torch.float32), upper.to(torch.float32)
            scales = torch.ones_like(velocity)
            scales = torch.where(
                velocity > upper32,
                upper32 / torch.clamp(velocity, min=1e-30),
                scales,
            )
            scales = torch.where(
                velocity < lower32,
                lower32 / torch.clamp(velocity, max=-1e-30),
                scales,
            )
            scale = torch.clamp(torch.amin(scales, dim=-1), min=0.0, max=1.0)
            return velocity * scale[:, None]
        if self._warp_srinv is not None:
            torch = self.torch
            matrix = jacobian
            if self._locked_active_columns:
                matrix = matrix.index_select(2, self._unlocked_active_columns_tensor)
            self._warp_srinv.tolerance = self.config.srinv_tolerance
            self._warp_srinv.damping = self.config.srinv_damping
            solved_velocity = self._warp_srinv.solve(matrix.contiguous(), twist.contiguous())
            if self._locked_active_columns:
                velocity = torch.zeros(
                    (self.batch_size, self.velocity_dim),
                    dtype=torch.float32,
                    device=self.device,
                )
                velocity.index_copy_(1, self._unlocked_active_columns_tensor, solved_velocity)
            else:
                velocity = solved_velocity
            lower32, upper32 = lower.to(torch.float32), upper.to(torch.float32)
            scales = torch.ones_like(velocity)
            scales = torch.where(
                velocity > upper32,
                upper32 / torch.clamp(velocity, min=1e-30),
                scales,
            )
            scales = torch.where(
                velocity < lower32,
                lower32 / torch.clamp(velocity, max=-1e-30),
                scales,
            )
            scale = torch.clamp(torch.amin(scales, dim=-1), min=0.0, max=1.0)
            return velocity * scale[:, None]
        if self._compiled_native_velocity is None:
            return self._native_velocity_eager(twist, jacobian, lower, upper)
        return self._compiled_native_velocity(twist, jacobian, lower, upper)

    def _collision_lower_bound(self, distance: Any) -> Any:
        """Apply EmbodiK's continuous velocity-damper recovery policy."""

        torch = self.torch
        minimum = self.config.collision_min_distance_m
        tolerance = self.config.collision_tolerance_m
        desired = (minimum + tolerance - distance) / self.config.dt
        outside = distance >= minimum + self.config.collision_repulsion_deadband_m
        within_deadband = (distance >= minimum) & ~outside
        nonpenetrating_violation = (distance >= 0.0) & (distance < minimum)
        penetrating = distance < 0.0
        lower = torch.where(outside, desired, torch.zeros_like(distance))
        lower = torch.where(within_deadband, torch.zeros_like(lower), lower)
        recovery = torch.clamp_min(desired * self.config.collision_recovery_scale, 0.0)
        recovery = torch.clamp_max(
            recovery,
            self.config.collision_max_separation_speed_nonpenetrating_m_s,
        )
        lower = torch.where(nonpenetrating_violation, recovery, lower)
        penetrating_recovery = torch.clamp(
            desired,
            min=self.config.collision_min_recovery_speed_m_s,
            max=self.config.collision_max_separation_speed_m_s,
        )
        return torch.where(penetrating, penetrating_recovery, lower)

    def _certified_current_collision(self, q: Any) -> Any:
        """Publish an exact empty current query or a fail-closed rejection."""

        assert self.collision is not None
        assert self.collision_convex_envelope is not None
        self.collision_convex_envelope._query_trusted(
            q,
            compute_gradient=False,
            probe_only=True,
        )
        return self.collision.publish_convex_envelope_certificate(self.collision_convex_envelope)

    def _apply_collision_constraint(
        self,
        velocity: Any,
        gradient: Any,
        distance: Any,
        active: Any,
        overflow: Any,
        lower: Any,
        upper: Any,
    ) -> tuple[Any, Any, Any]:
        if self._compiled_collision_constraint is not None:
            return self._compiled_collision_constraint(
                velocity, gradient, distance, active, overflow, lower, upper
            )
        return self._apply_collision_constraint_eager(
            velocity, gradient, distance, active, overflow, lower, upper
        )

    def _apply_collision_constraint_eager(
        self,
        velocity: Any,
        gradient: Any,
        distance: Any,
        active: Any,
        overflow: Any,
        lower: Any,
        upper: Any,
    ) -> tuple[Any, Any, Any]:
        """Project the primary velocity onto the closest collision half-space."""

        torch = self.torch
        lower32, upper32 = lower.to(torch.float32), upper.to(torch.float32)
        lower_bound = self._collision_lower_bound(distance)
        value = torch.sum(gradient * velocity, dim=-1)
        gradient_norm_squared = torch.sum(gradient * gradient, dim=-1)
        required = (
            active
            & ~overflow
            & (gradient_norm_squared > 1e-14)
            & (value < lower_bound - self.config.collision_tolerance_m)
        )
        maximum_velocity = torch.where(gradient >= 0.0, upper32, lower32)
        maximum_separation = torch.sum(gradient * maximum_velocity, dim=-1)
        target = torch.minimum(lower_bound, maximum_separation)
        positive = gradient > 1e-12
        negative = gradient < -1e-12
        ratios = torch.zeros_like(gradient)
        ratios = torch.where(
            positive,
            (upper32 - velocity) / gradient.clamp_min(1e-12),
            ratios,
        )
        ratios = torch.where(
            negative,
            (lower32 - velocity) / gradient.clamp_max(-1e-12),
            ratios,
        )
        high = torch.amax(torch.clamp_min(ratios, 0.0), dim=-1) + 1e-4
        low = torch.zeros_like(high)
        for _ in range(self.config.collision_projection_iterations):
            middle = 0.5 * (low + high)
            candidate = torch.maximum(
                torch.minimum(velocity + middle[:, None] * gradient, upper32),
                lower32,
            )
            candidate_value = torch.sum(gradient * candidate, dim=-1)
            high = torch.where(candidate_value >= target, middle, high)
            low = torch.where(candidate_value >= target, low, middle)
        projected = torch.maximum(
            torch.minimum(velocity + high[:, None] * gradient, upper32), lower32
        )
        corrected = torch.where(required[:, None], projected, velocity)
        corrected = torch.where(overflow[:, None], torch.zeros_like(corrected), corrected)
        return (
            corrected,
            required,
            torch.linalg.vector_norm(corrected - velocity, dim=-1),
        )

    def solve(
        self,
        q_start: Any,
        target: Any,
        previous_velocity: Any | None = None,
        current_velocity: Any | None = None,
    ) -> MultiFramePoseBatchResult:
        self._validate(q_start, target, previous_velocity, current_velocity)
        if self.config.standalone_cuda_graph_enabled:
            return self._solve_graph(q_start, target, previous_velocity, current_velocity)
        return self._solve_impl(q_start, target, previous_velocity, current_velocity)

    def _solve_graph(
        self,
        q_start: Any,
        target: Any,
        previous_velocity: Any | None,
        current_velocity: Any | None,
    ) -> MultiFramePoseBatchResult:
        torch = self.torch
        if self._graph is None:
            static_q = torch.empty_like(q_start)
            static_target = torch.empty_like(target)
            static_previous = torch.empty(
                (self.batch_size, self.velocity_dim),
                dtype=q_start.dtype,
                device=q_start.device,
            )
            static_current = torch.empty_like(static_previous)
            static_q.copy_(q_start)
            static_target.copy_(target)
            (
                static_previous.zero_()
                if previous_velocity is None
                else static_previous.copy_(previous_velocity)
            )
            (
                static_current.zero_()
                if current_velocity is None
                else static_current.copy_(current_velocity)
            )
            warm_result = self._solve_impl(static_q, static_target, static_previous, static_current)
            self.compact_publication(warm_result)
            torch.cuda.synchronize(self.device)
            import warp as wp

            graph = torch.cuda.CUDAGraph()
            if self.config.velocity_solver in {"warp_srinv", "cusolver_srinv"}:
                # Newton's child graphs and the Warp kernels can be recorded
                # directly by Torch. Warp 1.17 cannot reliably register this
                # mixed Torch/Newton parent capture as an external capture.
                with torch.cuda.graph(graph, capture_error_mode="thread_local"):
                    raw_graph_result = self._solve_impl(
                        static_q, static_target, static_previous, static_current
                    )
                    graph_result = replace(
                        raw_graph_result,
                        compact_publication=self.compact_publication(raw_graph_result),
                    )
            else:
                with torch.cuda.graph(graph, capture_error_mode="thread_local"):
                    torch_stream = torch.cuda.current_stream(self.device)
                    warp_stream = wp.Stream(
                        wp.get_device(str(self.device)),
                        cuda_stream=int(torch_stream.cuda_stream),
                    )
                    with wp.ScopedCapture(
                        device=wp.get_device(str(self.device)),
                        stream=warp_stream,
                        external=True,
                        capture_mode=wp.CaptureMode.THREAD_LOCAL,
                    ):
                        raw_graph_result = self._solve_impl(
                            static_q, static_target, static_previous, static_current
                        )
                        graph_result = replace(
                            raw_graph_result,
                            compact_publication=self.compact_publication(raw_graph_result),
                        )
            self._graph = graph
            self._graph_inputs = (
                static_q,
                static_target,
                static_previous,
                static_current,
            )
            self._graph_result = graph_result
        static_q, static_target, static_previous, static_current = self._graph_inputs
        static_q.copy_(q_start)
        static_target.copy_(target)
        (
            static_previous.zero_()
            if previous_velocity is None
            else static_previous.copy_(previous_velocity)
        )
        (
            static_current.zero_()
            if current_velocity is None
            else static_current.copy_(current_velocity)
        )
        self._graph.replay()
        return self._graph_result

    def compact_publication(self, result: MultiFramePoseBatchResult) -> Any:
        """Pack one solved batch for a single synchronized public readback."""

        return self._compact_publication_packer.pack(
            q=result.q_solution,
            position=result.position_error_m,
            orientation=result.orientation_error_rad,
            converged=result.converged,
            collision_distance=result.minimum_collision_distance_m,
            collision_active=result.collision_active,
            collision_accepted=result.collision_step_accepted,
            collision_overflow=result.collision_overflow,
            collision_applied=result.collision_constraint_applied,
            effective_dt=result.effective_dt,
            com_applied=result.com_constraint_applied,
            com_feasible=result.com_constraint_feasible,
            com_slack=result.minimum_com_slack_m,
            capture_applied=result.capture_point_constraint_applied,
            capture_feasible=result.capture_point_constraint_feasible,
            capture_slack=result.minimum_capture_point_slack_m,
            zmp_applied=result.velocity_zmp_constraint_applied,
            zmp_feasible=result.velocity_zmp_constraint_feasible,
            zmp_slack=result.minimum_velocity_zmp_slack_m,
            zmp_force=result.minimum_zmp_normal_force_n,
            momentum_applied=result.centroidal_momentum_task_applied,
            torso_applied=result.torso_constraint_applied,
            torso_feasible=result.torso_constraint_feasible,
            posture_applied=result.posture_task_applied,
            posture_primary=result.posture_primary_residual_increase,
            posture_before=result.posture_secondary_residual_before,
            posture_after=result.posture_secondary_residual_after,
            secondary_applied=result.secondary_task_applied,
            secondary_primary=result.secondary_primary_residual_increase,
            secondary_before=result.secondary_residual_before,
            secondary_after=result.secondary_residual_after,
            spectral_ok=result.spectral_solve_ok,
            collision_clear_state_certified=(result.collision_clear_state_certified),
        )

    def _errors_eager(self, pose: Any, target: Any) -> tuple[Any, Any, Any, Any]:
        torch = self.torch
        position_vector = target[:, :, :3] - pose[:, :, :3]
        position = torch.linalg.vector_norm(position_vector, dim=-1)
        angular_vector, orientation = _orientation_error(
            target[:, :, 3:].reshape(-1, 4), pose[:, :, 3:].reshape(-1, 4)
        )
        angular_vector = angular_vector.reshape(self.batch_size, self.frame_count, 3)
        orientation = orientation.reshape(self.batch_size, self.frame_count)
        angular_vector = torch.where(
            self._orientation_frame_mask[None, :, None],
            angular_vector,
            torch.zeros_like(angular_vector),
        )
        orientation = torch.where(
            self._orientation_frame_mask[None, :],
            orientation,
            torch.zeros_like(orientation),
        )
        return position_vector, position, angular_vector, orientation

    def _errors(self, pose: Any, target: Any) -> tuple[Any, Any, Any, Any]:
        if self._compiled_errors is None:
            return self._errors_eager(pose, target)
        return self._compiled_errors(pose, target)

    def _task_state_eager(self, pose: Any, target: Any) -> tuple[Any, Any, Any]:
        position_vector, position, angular_vector, orientation = self._errors_eager(pose, target)
        linear = _clamp_norm(
            (position_vector * self._frame_position_gains[None, :, None]).reshape(-1, 3),
            self.config.max_linear_speed,
        ).reshape(self.batch_size, self.frame_count, 3)
        angular = _clamp_norm(
            (angular_vector * self._frame_orientation_gains[None, :, None]).reshape(-1, 3),
            self.config.max_angular_speed,
        ).reshape(self.batch_size, self.frame_count, 3)
        chunks = []
        for frame_index, dimension in enumerate(self.frame_task_dimensions):
            chunks.append(linear[:, frame_index])
            if dimension == 6:
                chunks.append(angular[:, frame_index])
        twist = self.torch.cat(chunks, dim=-1)
        return position, orientation, twist

    def _select_task_jacobian(self, jacobian: Any) -> Any:
        shaped = jacobian.reshape(self.batch_size, self.frame_count, 6, self.velocity_dim)
        chunks = [
            shaped[:, frame_index, :dimension]
            for frame_index, dimension in enumerate(self.frame_task_dimensions)
        ]
        return self.torch.cat(chunks, dim=1)

    def _contact_jacobian(self, jacobian: Any) -> Any | None:
        shaped = jacobian.reshape(self.batch_size, self.frame_count, 6, self.velocity_dim)
        chunks = [
            shaped[:, frame_index, : self.frame_task_dimensions[frame_index]]
            for frame_index, constrained in enumerate(self.frame_contact_constraints)
            if constrained
        ]
        return None if not chunks else self.torch.cat(chunks, dim=1)

    def _contact_projector(self, contact_jacobian: Any | None) -> tuple[Any, Any]:
        torch = self.torch
        spectral_ok = torch.ones(self.batch_size, dtype=torch.bool, device=self.device)
        if contact_jacobian is None:
            return None, spectral_ok
        if self._cusolver_contact_inverse is not None:
            outputs = self._cusolver_contact_inverse.solve(
                contact_jacobian.contiguous(), self._cusolver_contact_rhs
            )
            identity = torch.eye(self.velocity_dim, dtype=torch.float32, device=self.device).expand(
                self.batch_size, -1, -1
            )
            return (
                identity - outputs[2] @ contact_jacobian,
                outputs[3] == 0,
            )
        if self._warp_contact_inverse is not None:
            inverse = self._warp_contact_inverse.solve(contact_jacobian.contiguous())
            identity = torch.eye(self.velocity_dim, dtype=torch.float32, device=self.device).expand(
                self.batch_size, -1, -1
            )
            return (
                identity - inverse @ contact_jacobian,
                self._warp_contact_inverse.status[: self.batch_size] == 0,
            )
        contact = contact_jacobian.to(torch.float64)
        identity = torch.eye(self.velocity_dim, dtype=torch.float64, device=self.device).expand(
            self.batch_size, -1, -1
        )
        return identity - torch.linalg.pinv(contact) @ contact, spectral_ok

    def _split_kinematics(self, pose: Any, jacobian: Any) -> tuple[Any, Any, Any, Any]:
        if pose.ndim == 2:
            pose = pose[:, None, :]
        shaped_jacobian = jacobian.reshape(
            self.batch_size, len(self._kinematics_frames), 6, self.velocity_dim
        )
        primary_pose = pose[:, : self.frame_count]
        primary_jacobian = shaped_jacobian[:, : self.frame_count].reshape(
            self.batch_size, self.frame_count * 6, self.velocity_dim
        )
        if self._torso_frame_index is None:
            return primary_pose, primary_jacobian, None, None
        return (
            primary_pose,
            primary_jacobian,
            pose[:, self._torso_frame_index],
            shaped_jacobian[:, self._torso_frame_index],
        )

    def _primary_pose(self, pose: Any) -> Any:
        if pose.ndim == 2:
            pose = pose[:, None, :]
        return pose[:, : self.frame_count]

    def set_posture_target(self, target_configuration: Any) -> None:
        """Update the device-resident posture bias without changing solver shape."""

        if not self.posture_task_enabled or self._posture_target is None:
            raise RuntimeError("posture task is not configured")
        target = self.torch.as_tensor(
            target_configuration, dtype=self.torch.float32, device=self.device
        )
        if target.shape != (self.configuration_dim,) or not bool(
            self.torch.isfinite(target).all().item()
        ):
            raise ValueError(
                f"posture target must have shape {(self.configuration_dim,)} and be finite"
            )
        self._posture_target.copy_(target)

    def configure_posture(
        self,
        *,
        gain: float | None = None,
        target_configuration: Any | None = None,
        weights: Any | None = None,
    ) -> None:
        """Update shape-stable posture controls used by interactive adapters."""

        if not self.posture_task_enabled:
            raise RuntimeError("posture task is not configured")
        was_enabled = self._posture_runtime_enabled
        if gain is not None:
            if not math.isfinite(gain) or gain < 0.0:
                raise ValueError("posture gain must be finite and nonnegative")
            self._posture_gain = float(gain)
            assert self._posture_gain_device is not None
            self._posture_gain_device.fill_(self._posture_gain)
        if target_configuration is not None:
            self.set_posture_target(target_configuration)
        if weights is not None:
            assert self._posture_weights is not None
            values = self.torch.as_tensor(weights, dtype=self.torch.float32, device=self.device)
            if (
                values.shape != self._posture_weights.shape
                or not bool(self.torch.isfinite(values).all().item())
                or bool(self.torch.any(values < 0.0).item())
            ):
                raise ValueError(
                    f"posture weights must have shape {tuple(self._posture_weights.shape)} "
                    "and be finite and nonnegative"
                )
            self._posture_weights.copy_(values)
            assert self._posture_jacobian is not None
            self._posture_jacobian.zero_()
            posture_columns = torch.tensor(
                [
                    tuple(self.robot_spec.active_velocity_indices).index(index)
                    for index in self._posture_velocity_indices
                ],
                dtype=torch.long,
                device=self.device,
            )
            self._posture_jacobian[
                torch.arange(len(self._posture_velocity_indices), device=self.device),
                posture_columns,
            ] = values
        assert self._posture_weights is not None
        self._posture_runtime_enabled = bool(
            self._posture_gain > 0.0 and self.torch.any(self._posture_weights > 0.0).item()
        )
        if was_enabled != self._posture_runtime_enabled:
            self._graph = None

    def configure_torso_constraint(
        self,
        *,
        reference_pose_xyzw: Any | None = None,
        lower_relative_limits: Any | None = None,
        upper_relative_limits: Any | None = None,
        axis_mask: Any | None = None,
    ) -> None:
        """Update shape-stable torso bounds without rebuilding kinematics."""

        if not self.torso_constraint_enabled:
            raise RuntimeError("torso constraint is not configured")
        updates = (
            (reference_pose_xyzw, self._torso_reference_pose, 7, self.torch.float64),
            (
                lower_relative_limits,
                self._torso_lower_relative_limits,
                6,
                self.torch.float64,
            ),
            (
                upper_relative_limits,
                self._torso_upper_relative_limits,
                6,
                self.torch.float64,
            ),
            (axis_mask, self._torso_axis_mask, 6, self.torch.bool),
        )
        for source, destination, width, dtype in updates:
            if source is None:
                continue
            assert destination is not None
            values = self.torch.as_tensor(source, dtype=dtype, device=self.device)
            if values.shape != (width,):
                raise ValueError(f"torso update must have shape {(width,)}")
            if dtype is not self.torch.bool and not bool(self.torch.isfinite(values).all().item()):
                raise ValueError("torso update must be finite")
            destination.copy_(values)
        assert self._torso_reference_pose is not None
        assert self._torso_lower_relative_limits is not None
        assert self._torso_upper_relative_limits is not None
        if not bool(self.torch.linalg.vector_norm(self._torso_reference_pose[3:]) > 1.0e-8):
            raise ValueError("torso reference quaternion must be nonzero")
        if bool(
            self.torch.any(
                self._torso_lower_relative_limits >= self._torso_upper_relative_limits
            ).item()
        ):
            raise ValueError("torso lower limits must remain below upper limits")

    def configure_secondary_frames(
        self,
        *,
        target_poses_wxyz: Any | None = None,
        position_gains: Any | None = None,
        orientation_gains: Any | None = None,
        weights: Any | None = None,
    ) -> None:
        """Update fixed-shape priority-1 frame targets and gains."""

        if not self.secondary_frame_tasks_enabled:
            raise RuntimeError("secondary frame tasks are not configured")
        destinations = (
            (target_poses_wxyz, self._secondary_frame_targets),
            (position_gains, self._secondary_frame_position_gains),
            (orientation_gains, self._secondary_frame_orientation_gains),
            (weights, self._secondary_frame_weights),
        )
        candidates: list[tuple[Any, Any]] = []
        for source, destination in destinations:
            if source is None:
                continue
            assert destination is not None
            candidate = self.torch.as_tensor(source, dtype=destination.dtype, device=self.device)
            if candidate.shape != destination.shape or not bool(
                self.torch.isfinite(candidate).all().item()
            ):
                raise ValueError(
                    f"secondary frame update must have shape {tuple(destination.shape)} "
                    "and be finite"
                )
            candidates.append((destination, candidate))
        target = next(
            (
                candidate
                for destination, candidate in candidates
                if destination is self._secondary_frame_targets
            ),
            self._secondary_frame_targets,
        )
        if target is not None and bool(
            self.torch.any(self.torch.linalg.vector_norm(target[:, 3:], dim=-1) <= 1.0e-8).item()
        ):
            raise ValueError("secondary target quaternions must be nonzero")
        for destination, candidate in candidates:
            if destination is self._secondary_frame_position_gains and bool(
                self.torch.any(candidate <= 0.0).item()
            ):
                raise ValueError("secondary position gains must remain positive")
            if (
                destination is self._secondary_frame_orientation_gains
                or destination is self._secondary_frame_weights
            ) and bool(self.torch.any(candidate < 0.0).item()):
                raise ValueError("secondary gains and weights must remain nonnegative")
        for destination, candidate in candidates:
            destination.copy_(candidate)

    def configure_task_hierarchy(
        self,
        *,
        posture_priority: int = 1,
        secondary_frame_orientation_only: tuple[bool, ...] = (),
    ) -> None:
        """Select shared secondary (1, default) or tertiary (2) posture.

        Primary pose rows are priority zero. Orientation-only flags remove
        translation rows entirely from selected secondary pose tasks. This is
        model-independent and can be configured before adapter warmup.
        """
        if isinstance(posture_priority, bool) or posture_priority not in (1, 2):
            raise ValueError("posture_priority must be 1 (secondary) or 2 (tertiary)")
        flags = tuple(secondary_frame_orientation_only)
        dimensions = self._secondary_frame_task_dimensions
        if flags and (
            len(flags) != len(dimensions)
            or any(
                not isinstance(flag, bool) or (flag and dim != 6)
                for flag, dim in zip(flags, dimensions)
            )
        ):
            raise ValueError("orientation-only flags must match secondary pose tasks")
        self._posture_priority = int(posture_priority)
        self._secondary_frame_orientation_only = flags or (False,) * len(dimensions)
        if hasattr(self, "_graph"):
            self._graph = None

    def _apply_secondary_tasks_eager(
        self,
        velocity: Any,
        primary_jacobian: Any,
        q: Any,
        full_pose: Any,
        full_jacobian: Any,
        lower: Any,
        upper: Any,
        centroidal_ag: Any = None,
    ) -> tuple[Any, Any, Any, Any, Any, Any]:
        torch = self.torch
        inactive = torch.zeros(self.batch_size, dtype=torch.bool, device=self.device)
        zeros = torch.zeros(self.batch_size, dtype=torch.float32, device=self.device)
        spectral_ok = torch.ones(self.batch_size, dtype=torch.bool, device=self.device)
        if not self.secondary_task_enabled:
            return velocity, inactive, zeros, zeros, zeros, spectral_ok
        goals = []
        jacobians = []
        posture_rows = None
        tertiary_goals = []
        tertiary_jacobians = []
        if self.posture_task_enabled and self._posture_runtime_enabled:
            assert self._posture_target is not None
            assert self._posture_weights is not None
            assert self._posture_q_indices is not None
            assert self._posture_jacobian is not None
            assert self._posture_gain_device is not None
            raw_error = self._posture_target[self._posture_q_indices] - q.index_select(
                -1, self._posture_q_indices
            )
            weighted_error = raw_error * self._posture_weights
            rows = PostureTaskRows(
                error=weighted_error,
                goal=weighted_error * self._posture_gain_device,
                jacobian=self._posture_jacobian.expand(self.batch_size, -1, -1),
            )
            if self._posture_priority == 2:
                tertiary_goals.append(rows.goal)
                tertiary_jacobians.append(rows.jacobian)
            else:
                goals.append(rows.goal)
                jacobians.append(rows.jacobian)
        if self.secondary_frame_tasks_enabled:
            assert self._secondary_frame_targets is not None
            assert self._secondary_frame_position_gains is not None
            assert self._secondary_frame_orientation_gains is not None
            assert self._secondary_frame_weights is not None
            if full_pose.ndim == 2:
                full_pose = full_pose[:, None, :]
            shaped_jacobian = full_jacobian.reshape(
                self.batch_size,
                len(self._kinematics_frames),
                6,
                self.velocity_dim,
            )
            for task_index, frame_index in enumerate(self._secondary_frame_indices):
                pose = full_pose[:, frame_index]
                target = self._secondary_frame_targets[task_index].expand(self.batch_size, -1)
                linear = _clamp_norm(
                    (target[:, :3] - pose[:, :3])
                    * self._secondary_frame_position_gains[task_index],
                    self.config.max_linear_speed,
                )
                chunks = [linear]
                dimension = self._secondary_frame_task_dimensions[task_index]
                if dimension == 6:
                    angular, _ = _orientation_error(target[:, 3:], pose[:, 3:])
                    chunks.append(
                        _clamp_norm(
                            angular * self._secondary_frame_orientation_gains[task_index],
                            self.config.max_angular_speed,
                        )
                    )
                weight = self._secondary_frame_weights[task_index]
                row_start = 0
                if self._secondary_frame_orientation_only[task_index]:
                    chunks = chunks[1:]
                    row_start = 3
                goals.append(torch.cat(chunks, dim=-1) * weight)
                jacobians.append(shaped_jacobian[:, frame_index, row_start:dimension] * weight)
        if getattr(self, "centroidal_momentum_task_enabled", False):
            if centroidal_ag is None:
                raise RuntimeError("centroidal momentum task requires centroidal state")
            row_weight = self._momentum_axis_mask.to(torch.float32) * self._momentum_weight
            momentum_jacobian = centroidal_ag * row_weight[None, :, None]
            momentum_jacobian = momentum_jacobian.masked_fill(
                self._momentum_excluded[None, None, :], 0.0
            )
            momentum_goal = self._momentum_target * row_weight[None, :]
            if self._momentum_priority == 2:
                tertiary_goals.append(momentum_goal)
                tertiary_jacobians.append(momentum_jacobian)
            else:
                goals.append(momentum_goal)
                jacobians.append(momentum_jacobian)
        bands = []
        if jacobians:
            bands.append((torch.cat(jacobians, dim=1), torch.cat(goals, dim=1)))
        if tertiary_jacobians:
            bands.append(
                (
                    torch.cat(tertiary_jacobians, dim=1),
                    torch.cat(tertiary_goals, dim=1),
                )
            )
        if not bands:
            return velocity, inactive, zeros, zeros, zeros, spectral_ok
        if getattr(self, "_cusolver_srinv", None) is not None:
            from .cusolver_priority import CuSolverSecondaryPriority

            protected = primary_jacobian
            priorities = []
            for band_index, (band_jacobian, band_goal) in enumerate(bands):
                key = (protected.shape[-2], band_jacobian.shape[-2])
                solver = self._cusolver_priority_solvers.get(key)
                if solver is None:
                    solver = CuSolverSecondaryPriority(
                        key[0],
                        key[1],
                        self.velocity_dim,
                        srinv_tolerance=self.config.srinv_tolerance,
                        srinv_damping=self.config.srinv_damping,
                        relative_rank_tolerance=self._posture_rank_tolerance,
                        specialized_outputs_enabled=(
                            self.config.cusolver_specialized_outputs_enabled
                        ),
                        fused_status_enabled=bool(
                            self.config.cusolver_specialized_outputs_enabled
                            and self.config.native_compile_enabled
                            and _python_extension_headers_available()
                        ),
                        compiled_postprocess_enabled=bool(
                            self.config.cusolver_specialized_outputs_enabled
                            and self.config.native_compile_enabled
                            and _python_extension_headers_available()
                        ),
                        device=str(self.device),
                    )
                    self._cusolver_priority_solvers[key] = solver
                reuse_primary = band_index == 0 and (
                    not self._locked_active_columns
                    or self.config.cusolver_reuse_locked_primary_inverse_enabled
                )
                priority = solver.solve(
                    velocity,
                    protected.contiguous(),
                    band_jacobian.contiguous(),
                    band_goal.contiguous(),
                    lower.to(torch.float32),
                    upper.to(torch.float32),
                    (self._locked_velocity_mask if self._locked_active_columns else None),
                    protected_inverse=(self._cusolver_primary_inverse if reuse_primary else None),
                    protected_spectral_ok=(
                        self._cusolver_primary_status == 0 if reuse_primary else None
                    ),
                )
                priorities.append(priority)
                velocity = priority.velocity
                spectral_ok &= priority.spectral_ok
                if band_index != len(bands) - 1:
                    protected = torch.cat((protected, band_jacobian), dim=-2)
            priorities = tuple(priorities)
        elif getattr(self, "_warp_srinv", None) is None:
            priorities = ordered_priority_velocity(
                velocity,
                primary_jacobian,
                bands,
                lower.to(torch.float32),
                upper.to(torch.float32),
                srinv_tolerance=self.config.srinv_tolerance,
                srinv_damping=self.config.srinv_damping,
                relative_rank_tolerance=self._posture_rank_tolerance,
                locked_velocity_mask=self._locked_velocity_mask,
                validate_values=False,
            )
        else:
            from .warp_priority import WarpSecondaryPriority

            protected = primary_jacobian
            priorities = []
            for band_index, (band_jacobian, band_goal) in enumerate(bands):
                key = (protected.shape[-2], band_jacobian.shape[-2])
                solver = self._warp_priority_solvers.get(key)
                if solver is None:
                    solver = WarpSecondaryPriority(
                        key[0],
                        key[1],
                        self.velocity_dim,
                        batch_capacity=self.batch_size,
                        srinv_tolerance=self.config.srinv_tolerance,
                        srinv_damping=self.config.srinv_damping,
                        relative_rank_tolerance=self._posture_rank_tolerance,
                        device=str(self.device),
                    )
                    self._warp_priority_solvers[key] = solver
                reuse_primary_factorization = band_index == 0 and not self._locked_active_columns
                priority = solver.solve(
                    velocity,
                    protected.contiguous(),
                    band_jacobian.contiguous(),
                    band_goal.contiguous(),
                    lower.to(torch.float32),
                    upper.to(torch.float32),
                    (None if reuse_primary_factorization else self._locked_velocity_mask),
                    protected_inverse=(
                        self._warp_srinv.undamped_inverse[: self.batch_size]
                        if reuse_primary_factorization
                        else None
                    ),
                    protected_spectral_ok=(
                        self._warp_srinv.status[: self.batch_size] == 0
                        if reuse_primary_factorization
                        else None
                    ),
                )
                priorities.append(priority)
                velocity = priority.velocity
                spectral_ok &= priority.spectral_ok
                if band_index != len(bands) - 1:
                    protected = torch.cat((protected, band_jacobian), dim=-2)
            priorities = tuple(priorities)
        priority = priorities[-1]
        return (
            priority.velocity,
            torch.stack([p.applied for p in priorities]).any(dim=0),
            torch.linalg.vector_norm(
                (primary_jacobian @ (priority.velocity - velocity).unsqueeze(-1)).squeeze(-1),
                dim=-1,
            ),
            priority.secondary_residual_before,
            priority.secondary_residual_after,
            spectral_ok,
        )

    def _apply_secondary_tasks(
        self,
        velocity: Any,
        primary_jacobian: Any,
        q: Any,
        full_pose: Any,
        full_jacobian: Any,
        lower: Any,
        upper: Any,
        centroidal_ag: Any = None,
    ) -> tuple[Any, Any, Any, Any, Any, Any]:
        compiled = getattr(self, "_compiled_secondary_tasks", None)
        if compiled is None:
            return DeviceResidentMultiFramePoseSolver._apply_secondary_tasks_eager(
                self,
                velocity,
                primary_jacobian,
                q,
                full_pose,
                full_jacobian,
                lower,
                upper,
                centroidal_ag,
            )
        return compiled(
            velocity,
            primary_jacobian,
            q,
            full_pose,
            full_jacobian,
            lower,
            upper,
            centroidal_ag,
        )

    def configure_com_constraint(
        self,
        *,
        support_polygon_xy=None,
        enabled=None,
        margin=None,
        vel_max=None,
        acc_max=None,
        use_acceleration_limits=None,
        proximity_fraction=None,
    ):
        """Atomically validate host settings then update fixed-capacity CUDA buffers.

        Polygons may be [vertices,2] shared or [batch,vertices,2]. Updates are
        serialized with solves on the caller's stream; geometry stays world XY.
        """
        import numpy as np

        if self.com is None:
            raise ValueError("CoM capacity must be configured at construction")
        old = getattr(self, "_com_host", {})
        values = dict(old)
        for key, value in dict(
            support_polygon_xy=support_polygon_xy,
            margin=margin,
            vel_max=vel_max,
            acc_max=acc_max,
            use_acceleration_limits=use_acceleration_limits,
            proximity_fraction=proximity_fraction,
        ).items():
            if value is not None:
                values[key] = value
        if any(not math.isfinite(values[k]) or values[k] <= 0 for k in ("vel_max", "acc_max")):
            raise ValueError("CoM velocity/acceleration limits must be finite and positive")
        polygons = np.asarray(values["support_polygon_xy"], dtype=np.float64)
        if polygons.ndim == 2:
            polygons = np.broadcast_to(polygons, (self.batch_size,) + polygons.shape)
        if polygons.ndim != 3 or polygons.shape[0] != self.batch_size:
            raise ValueError("CoM polygon must be [vertices,2] or [batch,vertices,2]")
        a = np.zeros(tuple(self._com_a.shape))
        b = np.zeros(tuple(self._com_b.shape))
        active = np.zeros(b.shape, dtype=bool)
        prox = np.zeros((self.batch_size, 1))
        for world, polygon in enumerate(polygons):
            normals, offsets, proximity = support_polygon_halfplanes(
                polygon,
                margin=values["margin"],
                proximity_fraction=values["proximity_fraction"],
            )
            count = len(offsets)
            if count > a.shape[1]:
                raise ValueError("CoM polygon exceeds fixed constraint capacity")
            a[world, :count], b[world, :count], active[world, :count] = (
                normals,
                offsets,
                True,
            )
            prox[world, 0] = proximity
        for destination, source in (
            (self._com_a, a),
            (self._com_b, b),
            (self._com_active, active),
            (self._com_proximity, prox),
        ):
            destination.copy_(
                self.torch.as_tensor(source, dtype=destination.dtype, device=self.device)
            )
        self._com_settings.copy_(
            self.torch.tensor(
                [
                    values["vel_max"],
                    values["acc_max"],
                    bool(values["use_acceleration_limits"]),
                ],
                dtype=self.torch.float64,
                device=self.device,
            )
        )
        values["support_polygon_xy"] = np.array(values["support_polygon_xy"], copy=True)
        self._com_host = values
        toggle = enabled is not None and bool(enabled) != self.com_constraint_enabled
        if enabled is not None:
            self.com_constraint_enabled = bool(enabled)
        # A captured Python enabled branch must be recaptured on toggles.
        if toggle and hasattr(self, "_graph"):
            self._graph = None

    def _configure_centroidal_polygon(
        self, destination_a, destination_b, destination_active, polygon, margin
    ) -> None:
        import numpy as np

        polygons = np.asarray(polygon, dtype=np.float64)
        if polygons.ndim == 2:
            polygons = np.broadcast_to(polygons, (self.batch_size,) + polygons.shape)
        if polygons.ndim != 3 or polygons.shape[0] != self.batch_size:
            raise ValueError("support polygon must be [vertices,2] or [batch,vertices,2]")
        a = np.zeros(tuple(destination_a.shape))
        b = np.zeros(tuple(destination_b.shape))
        active = np.zeros(b.shape, dtype=bool)
        for world, vertices in enumerate(polygons):
            normals, offsets, _ = support_polygon_halfplanes(
                vertices, margin=float(margin), proximity_fraction=0.0
            )
            count = len(offsets)
            if count > a.shape[1]:
                raise ValueError("support polygon exceeds fixed constraint capacity")
            a[world, :count] = normals
            b[world, :count] = offsets
            active[world, :count] = True
        for destination, source in (
            (destination_a, a),
            (destination_b, b),
            (destination_active, active),
        ):
            destination.copy_(
                self.torch.as_tensor(source, dtype=destination.dtype, device=self.device)
            )

    def configure_capture_point_constraint(
        self,
        *,
        support_polygon_xy=None,
        enabled=None,
        margin=None,
        omega=None,
        height=None,
        gravity_z=None,
    ) -> None:
        """Update the fixed-capacity world-frame capture-point constraint."""

        if self.com is None:
            raise ValueError("centroidal capacity must be configured at construction")
        old = getattr(self, "_capture_host", {})
        values = dict(old)
        for key, value in {
            "support_polygon_xy": support_polygon_xy,
            "margin": margin,
            "omega": omega,
            "height": height,
            "gravity_z": gravity_z,
        }.items():
            if value is not None:
                values[key] = value
        values.setdefault("margin", 0.0)
        values.setdefault("omega", None)
        values.setdefault("height", None)
        values.setdefault("gravity_z", -9.81)
        if "support_polygon_xy" not in values:
            raise ValueError("capture-point support polygon is required")
        if not math.isfinite(float(values["margin"])) or values["margin"] < 0.0:
            raise ValueError("capture-point margin must be finite and nonnegative")
        if values["omega"] is not None and (
            not math.isfinite(float(values["omega"])) or values["omega"] <= 0.0
        ):
            raise ValueError("capture-point omega must be finite and positive")
        if values["height"] is not None and (
            not math.isfinite(float(values["height"])) or values["height"] <= 0.0
        ):
            raise ValueError("capture-point height must be finite and positive")
        if not math.isfinite(float(values["gravity_z"])) or values["gravity_z"] >= 0.0:
            raise ValueError("capture-point gravity_z must be finite and negative")
        self._configure_centroidal_polygon(
            self._capture_a,
            self._capture_b,
            self._capture_active,
            values["support_polygon_xy"],
            values["margin"],
        )
        self._capture_settings.copy_(
            self.torch.tensor(
                (
                    -1.0 if values["omega"] is None else values["omega"],
                    -1.0 if values["height"] is None else values["height"],
                    values["gravity_z"],
                ),
                dtype=self.torch.float64,
                device=self.device,
            )
        )
        self._capture_host = values
        toggle = enabled is not None and bool(enabled) != self.capture_point_constraint_enabled
        if enabled is not None:
            self.capture_point_constraint_enabled = bool(enabled)
        if toggle and hasattr(self, "_graph"):
            self._graph = None

    def configure_velocity_zmp_constraint(
        self,
        *,
        support_polygon_xy=None,
        enabled=None,
        margin=None,
        fz_min=None,
        gravity_z=None,
    ) -> None:
        """Update exact affine velocity-ZMP settings in the world support frame."""

        if self.com is None:
            raise ValueError("centroidal capacity must be configured at construction")
        old = getattr(self, "_zmp_host", {})
        values = dict(old)
        for key, value in {
            "support_polygon_xy": support_polygon_xy,
            "margin": margin,
            "fz_min": fz_min,
            "gravity_z": gravity_z,
        }.items():
            if value is not None:
                values[key] = value
        values.setdefault("margin", 0.0)
        values.setdefault("fz_min", 1.0)
        values.setdefault("gravity_z", -9.81)
        if "support_polygon_xy" not in values:
            raise ValueError("velocity-ZMP support polygon is required")
        if not math.isfinite(float(values["margin"])) or values["margin"] < 0.0:
            raise ValueError("velocity-ZMP margin must be finite and nonnegative")
        if not math.isfinite(float(values["fz_min"])) or values["fz_min"] <= 0.0:
            raise ValueError("velocity-ZMP fz_min must be finite and positive")
        if not math.isfinite(float(values["gravity_z"])) or values["gravity_z"] >= 0.0:
            raise ValueError("velocity-ZMP gravity_z must be finite and negative")
        self._configure_centroidal_polygon(
            self._zmp_a,
            self._zmp_b,
            self._zmp_active,
            values["support_polygon_xy"],
            values["margin"],
        )
        self._zmp_settings.copy_(
            self.torch.tensor(
                (values["fz_min"], values["gravity_z"]),
                dtype=self.torch.float64,
                device=self.device,
            )
        )
        self._zmp_host = values
        toggle = enabled is not None and bool(enabled) != self.velocity_zmp_constraint_enabled
        if enabled is not None:
            self.velocity_zmp_constraint_enabled = bool(enabled)
        if toggle and hasattr(self, "_graph"):
            self._graph = None

    def configure_centroidal_momentum(
        self,
        *,
        enabled=None,
        target=None,
        axis_mask=None,
        weight=None,
        lower=None,
        upper=None,
    ) -> None:
        """Update a shape-stable absolute momentum task and optional hard bounds."""

        import numpy as np

        if self.com is None:
            raise ValueError("centroidal capacity must be configured at construction")
        if target is not None:
            values = np.asarray(target, dtype=np.float32)
            if values.shape != (6,) or not np.isfinite(values).all():
                raise ValueError("centroidal momentum target must be finite with shape (6,)")
            self._momentum_target.copy_(
                self.torch.as_tensor(values, device=self.device).expand(self.batch_size, -1)
            )
        if axis_mask is not None:
            mask = np.asarray(axis_mask, dtype=bool)
            if mask.shape != (6,):
                raise ValueError("centroidal momentum axis mask must have shape (6,)")
            self._momentum_axis_mask.copy_(
                self.torch.as_tensor(mask, dtype=self.torch.bool, device=self.device)
            )
            if self.centroidal_momentum_bounds_enabled:
                self._momentum_bound_active.copy_(
                    self._momentum_axis_mask.expand(self.batch_size, -1)
                )
        if weight is not None:
            if not math.isfinite(float(weight)) or weight < 0.0:
                raise ValueError("centroidal momentum weight must be finite and nonnegative")
            self._momentum_weight.fill_(float(weight))
        if (lower is None) != (upper is None):
            raise ValueError("centroidal momentum lower and upper must be updated together")
        if lower is not None:
            lower_values = np.asarray(lower, dtype=np.float64)
            upper_values = np.asarray(upper, dtype=np.float64)
            if (
                lower_values.shape != (6,)
                or upper_values.shape != (6,)
                or not np.isfinite(lower_values).all()
                or not np.isfinite(upper_values).all()
                or np.any(lower_values > upper_values)
            ):
                raise ValueError(
                    "centroidal momentum bounds must be finite ordered shape-(6,) values"
                )
            self._momentum_lower.copy_(
                self.torch.as_tensor(lower_values, device=self.device).expand(self.batch_size, -1)
            )
            self._momentum_upper.copy_(
                self.torch.as_tensor(upper_values, device=self.device).expand(self.batch_size, -1)
            )
            self._momentum_bound_active.copy_(self._momentum_axis_mask.expand(self.batch_size, -1))
            self.centroidal_momentum_bounds_enabled = True
        toggle = enabled is not None and bool(enabled) != self.centroidal_momentum_task_enabled
        if enabled is not None:
            self.centroidal_momentum_task_enabled = bool(enabled)
        if toggle and hasattr(self, "_graph"):
            self._graph = None

    def _centroidal_configuration(self, q):
        full = q
        if not self.robot_spec.floating_base:
            full = self._com_default.clone()
            full.index_copy_(1, self._com_q_indices, q)
        return full

    def _com_state(self, q):
        position, jacobian = self.com.evaluate(self._centroidal_configuration(q))
        slack = self._com_b - (self._com_a @ position[:, :2, None].to(self.torch.float64)).squeeze(
            -1
        )
        rows = self._com_a @ jacobian[:, :2].to(self.torch.float64)
        return slack, rows.masked_fill(self._com_excluded[None, None, :], 0)

    def _apply_com_constraint(
        self,
        q,
        velocity,
        lower,
        upper,
        contact_projector,
        current_velocity=None,
        centroidal_state=None,
    ):
        torch = self.torch
        if not self.centroidal_enabled:
            inactive = torch.zeros(self.batch_size, dtype=torch.bool, device=self.device)
            return velocity, inactive, ~inactive, None, None, None, None
        full = self._centroidal_configuration(q)
        position, jacobian, ag = (
            self.com.evaluate_centroidal(full) if centroidal_state is None else centroidal_state
        )
        position64 = position.to(torch.float64)
        jacobian64 = jacobian.to(torch.float64)
        ag64 = ag.to(torch.float64)
        row_chunks = []
        lower_chunks = []
        upper_chunks = []
        active_chunks = []
        slack = None
        capture_slack = None
        zmp_slack = None
        zmp_force = None
        if self.com_constraint_enabled:
            slack = self._com_b - (self._com_a @ position64[:, :2, None]).squeeze(-1)
            com_rows = self._com_a @ jacobian64[:, :2]
            com_rows = com_rows.masked_fill(self._com_excluded[None, None, :], 0)
            com_lower, com_upper = com_halfspace_velocity_bounds(
                slack,
                self.config.dt,
                self._com_settings[0],
                self._com_settings[1],
                self._com_settings[2].bool(),
                self._com_proximity,
            )
        else:
            com_rows = torch.zeros(
                (*self._com_a.shape[:2], self.velocity_dim),
                dtype=torch.float64,
                device=self.device,
            )
            com_lower = torch.zeros_like(self._com_b)
            com_upper = torch.zeros_like(self._com_b)
        row_chunks.append(com_rows)
        lower_chunks.append(com_lower)
        upper_chunks.append(com_upper)
        active_chunks.append(
            self._com_active if self.com_constraint_enabled else torch.zeros_like(self._com_active)
        )
        if self.capture_point_constraint_enabled:
            configured_omega = self._capture_settings[0]
            configured_height = self._capture_settings[1]
            height = torch.where(
                configured_height > 0.0,
                configured_height.expand(self.batch_size),
                position64[:, 2],
            )
            derived = torch.sqrt(torch.abs(self._capture_settings[2]) / height.clamp_min(1.0e-30))
            omega = torch.where(
                configured_omega > 0.0,
                configured_omega.expand(self.batch_size),
                derived,
            )
            omega = torch.where(height > 0.0, omega, torch.full_like(omega, float("nan")))
            cp_rows, cp_lower, cp_upper = _capture_point_constraint_rows_trusted(
                self._capture_a,
                self._capture_b,
                position64[:, :2],
                jacobian64[:, :2],
                omega,
            )
            capture_point = (
                position64[:, :2]
                + (jacobian64[:, :2] @ velocity.to(torch.float64).unsqueeze(-1)).squeeze(-1)
                / omega[:, None]
            )
            capture_slack = self._capture_b - (
                self._capture_a @ capture_point.unsqueeze(-1)
            ).squeeze(-1)
        else:
            cp_rows = torch.zeros(
                (*self._capture_a.shape[:2], self.velocity_dim),
                dtype=torch.float64,
                device=self.device,
            )
            cp_lower = torch.zeros_like(self._capture_b)
            cp_upper = torch.zeros_like(self._capture_b)
        row_chunks.append(cp_rows)
        lower_chunks.append(cp_lower)
        upper_chunks.append(cp_upper)
        active_chunks.append(
            self._capture_active
            if self.capture_point_constraint_enabled
            else torch.zeros_like(self._capture_active)
        )
        if self.velocity_zmp_constraint_enabled:
            if current_velocity is None:
                raise ValueError("velocity ZMP requires explicit current_velocity")
            full_current_velocity = self._centroidal_full_velocity.zero_()
            full_current_velocity.index_copy_(
                1,
                self._centroidal_active_velocity_indices,
                current_velocity,
            )
            bias = self.com.evaluate_centroidal_bias(full, full_current_velocity).to(torch.float64)
            weight = torch.zeros((self.batch_size, 3), dtype=torch.float64, device=self.device)
            weight[:, 2] = self.com.total_mass * torch.abs(self._zmp_settings[1])
            dt = torch.full(
                (self.batch_size,), self.config.dt, dtype=torch.float64, device=self.device
            )
            zmp_rows, zmp_lower, zmp_upper = _velocity_zmp_constraint_rows_trusted(
                self._zmp_a,
                self._zmp_b,
                position64,
                ag64,
                bias,
                current_velocity.to(torch.float64),
                weight,
                dt,
                self._zmp_settings[0].expand(self.batch_size),
            )
            zmp_active = torch.cat(
                (
                    self._zmp_active,
                    torch.ones((self.batch_size, 1), dtype=torch.bool, device=self.device),
                ),
                dim=-1,
            )
            zmp_value = (zmp_rows @ velocity.to(torch.float64).unsqueeze(-1)).squeeze(-1)
            zmp_slack = zmp_upper[:, :-1] - zmp_value[:, :-1]
            zmp_force = zmp_value[:, -1] - zmp_lower[:, -1] + self._zmp_settings[0]
        else:
            zmp_rows = torch.zeros(
                (self.batch_size, self._zmp_a.shape[1] + 1, self.velocity_dim),
                dtype=torch.float64,
                device=self.device,
            )
            zmp_lower = torch.zeros(
                (self.batch_size, self._zmp_a.shape[1] + 1),
                dtype=torch.float64,
                device=self.device,
            )
            zmp_upper = torch.zeros_like(zmp_lower)
            zmp_active = torch.zeros_like(zmp_lower, dtype=torch.bool)
        row_chunks.append(zmp_rows)
        lower_chunks.append(zmp_lower)
        upper_chunks.append(zmp_upper)
        active_chunks.append(zmp_active)
        momentum_rows = ag64.masked_fill(self._momentum_excluded[None, None, :], 0)
        row_chunks.append(momentum_rows)
        lower_chunks.append(self._momentum_lower)
        upper_chunks.append(self._momentum_upper)
        active_chunks.append(
            self._momentum_bound_active
            if self.centroidal_momentum_bounds_enabled
            else torch.zeros_like(self._momentum_bound_active)
        )
        rows = torch.cat(row_chunks, dim=1)
        row_lower = torch.cat(lower_chunks, dim=1)
        row_upper = torch.cat(upper_chunks, dim=1)
        active = torch.cat(active_chunks, dim=1)
        if contact_projector is not None:
            # Preserve fixed contact tangent directions in the recovery correction.
            rows = rows @ contact_projector.to(torch.float64)
        self._com_step_rows, self._com_step_lower, self._com_step_upper, self._com_step_active = (
            rows,
            row_lower,
            row_upper,
            active,
        )
        result = (
            self._warp_com_projection.solve(
                velocity,
                rows.contiguous(),
                row_lower.contiguous(),
                row_upper.contiguous(),
                lower.contiguous(),
                upper.contiguous(),
                active.contiguous(),
            )
            if self._warp_com_projection is not None
            else bounded_cyclic_row_projection(
                velocity,
                rows,
                row_lower,
                row_upper,
                lower,
                upper,
                active_mask=active,
                iterations=self._centroidal_projection_iterations,
                feasibility_tolerance=1e-6,
            )
        )
        finite = (
            torch.isfinite(position).all(-1)
            & torch.isfinite(jacobian).all(dim=(-2, -1))
            & torch.isfinite(ag).all(dim=(-2, -1))
            & torch.isfinite(result.velocity).all(-1)
        )
        if slack is not None:
            finite &= torch.isfinite(slack).all(-1)
        if capture_slack is not None:
            finite &= torch.isfinite(capture_slack).all(-1)
        if zmp_slack is not None:
            finite &= torch.isfinite(zmp_slack).all(-1)
            finite &= torch.isfinite(zmp_force)
        applied = (result.velocity - velocity).abs().amax(-1) > 1e-7
        return (
            result.velocity.to(torch.float32),
            applied,
            result.feasible & finite,
            slack,
            capture_slack,
            zmp_slack,
            zmp_force,
        )

    def _com_candidate_valid(self, current_slack, candidate, velocity=None):
        if not self.centroidal_enabled:
            return self.torch.ones(self.batch_size, dtype=self.torch.bool, device=self.device)
        valid = self.torch.ones(self.batch_size, dtype=self.torch.bool, device=self.device)
        if self.com_constraint_enabled:
            slack, _ = self._com_state(candidate)
            floor = self.torch.minimum(
                current_slack, self.torch.zeros_like(current_slack)
            ).clamp_max(-1e-4)
            valid &= (
                self.torch.isfinite(slack) & (~self._com_active | (slack >= floor - 1e-7))
            ).all(-1)
        if velocity is not None:
            value = (self._com_step_rows @ velocity.to(self.torch.float64).unsqueeze(-1)).squeeze(
                -1
            )
            valid &= (
                ~self._com_step_active
                | ((value >= self._com_step_lower - 1e-6) & (value <= self._com_step_upper + 1e-6))
            ).all(-1)
        return valid

    def _apply_torso_constraint(
        self,
        velocity: Any,
        torso_pose: Any,
        torso_jacobian: Any,
        lower: Any,
        upper: Any,
    ) -> tuple[Any, Any, Any, Any]:
        torch = self.torch
        if not self.torso_constraint_enabled:
            inactive = torch.zeros(self.batch_size, dtype=torch.bool, device=self.device)
            return velocity, inactive, ~inactive, None
        rows, row_lower, row_upper, active = self._torso_rows(
            torso_pose,
            torso_jacobian,
        )
        projection = (
            self._warp_torso_projection.solve(
                velocity,
                rows.contiguous(),
                row_lower.contiguous(),
                row_upper.contiguous(),
                lower.contiguous(),
                upper.contiguous(),
                active.contiguous(),
            )
            if self._warp_torso_projection is not None
            else bounded_cyclic_row_projection(
                velocity,
                rows,
                row_lower,
                row_upper,
                lower,
                upper,
                active_mask=active,
                iterations=self._torso_projection_iterations,
                feasibility_tolerance=self._torso_feasibility_tolerance,
            )
        )
        projected = projection.velocity.to(torch.float32)
        applied = torch.any(active, dim=-1) & (
            torch.linalg.vector_norm(projected - velocity, dim=-1)
            > self._torso_feasibility_tolerance
        )
        state = relative_pose_state(torso_pose, self._torso_reference_pose)
        return projected, applied, projection.feasible, state

    def _torso_rows(self, torso_pose: Any, torso_jacobian: Any):
        if self._compiled_torso_rows is not None:
            return self._compiled_torso_rows(torso_pose, torso_jacobian)
        return self._torso_rows_eager(torso_pose, torso_jacobian)

    def _torso_rows_eager(self, torso_pose: Any, torso_jacobian: Any):
        return torso_pose_bound_rows(
            torso_pose,
            self._torso_reference_pose,
            torso_jacobian,
            self._torso_axis_mask,
            self._torso_lower_relative_limits,
            self._torso_upper_relative_limits,
            self._torso_velocity_limits,
            self._torso_acceleration_limits,
            self._torso_nominal_dt,
            self._torso_excluded_columns,
            acceleration_history_enabled=(self._torso_acceleration_history_enabled_device),
            headroom_enabled=self._torso_headroom_enabled_device,
            headroom_fraction=self._torso_headroom_fraction_device,
            headroom_activation_margin=(self._torso_headroom_activation_margin_device),
        )

    def _certify_torso_projection_noop(
        self,
        velocity: Any,
        torso_pose: Any,
        torso_jacobian: Any,
        lower: Any,
        upper: Any,
    ) -> tuple[Any, Any, Any, Any]:
        """Use the projector's own arithmetic and exact fixed-point exit.

        The prior Torch matmul certificate could round differently from the
        Warp kernel's sequential FP64 row accumulation at an interval edge.
        The optimized projector now exits after pass one only when its own
        arithmetic proves all later corrections are zero.
        """

        return self._apply_torso_constraint(
            velocity,
            torso_pose,
            torso_jacobian,
            lower,
            upper,
        )

    def _torso_candidate_valid(self, current_state: Any, candidate: Any) -> Any:
        torch = self.torch
        if not self.torso_constraint_enabled:
            return torch.ones(self.batch_size, dtype=torch.bool, device=self.device)
        candidate_pose = self.kinematics._evaluate_pose_trusted(candidate)
        candidate_state = relative_pose_state(
            candidate_pose[:, self._torso_frame_index],
            self._torso_reference_pose,
        )
        return _torso_candidate_nonworsening(
            torch,
            current_state,
            candidate_state,
            self._torso_lower_relative_limits,
            self._torso_upper_relative_limits,
            self._torso_axis_mask,
            self._torso_tolerance,
        )

    def _apply_collision_rows(
        self,
        velocity: Any,
        pair_gradient: Any,
        pair_distance: Any,
        pair_active: Any,
        overflow: Any,
        lower: Any,
        upper: Any,
        score: Any,
        contact_projector: Any | None,
    ) -> tuple[Any, Any]:
        if self._compiled_collision_rows is not None:
            return self._compiled_collision_rows(
                velocity,
                pair_gradient,
                pair_distance,
                pair_active,
                overflow,
                lower,
                upper,
                score,
                contact_projector,
            )
        return self._apply_collision_rows_eager(
            velocity,
            pair_gradient,
            pair_distance,
            pair_active,
            overflow,
            lower,
            upper,
            score,
            contact_projector,
        )

    def _apply_collision_rows_eager(
        self,
        velocity: Any,
        pair_gradient: Any,
        pair_distance: Any,
        pair_active: Any,
        overflow: Any,
        lower: Any,
        upper: Any,
        score: Any,
        contact_projector: Any | None,
    ) -> tuple[Any, Any]:
        """Project onto a fixed top-K semantic row set at the current state."""

        torch = self.torch
        row_count = min(self.config.collision_max_constraints, pair_distance.shape[1])
        selected = torch.topk(score, row_count, dim=-1, largest=False).indices
        gather_gradient = selected[:, :, None].expand(-1, -1, self.velocity_dim)
        gradient = torch.gather(pair_gradient, 1, gather_gradient)
        distance = torch.gather(pair_distance, 1, selected)
        active = torch.gather(pair_active, 1, selected)
        direction = gradient
        if contact_projector is not None:
            direction = (gradient.to(contact_projector.dtype) @ contact_projector).to(torch.float32)

        lower32, upper32 = lower.to(torch.float32), upper.to(torch.float32)
        lower_bound = self._collision_lower_bound(distance)
        corrected = velocity
        applied = torch.zeros(self.batch_size, dtype=torch.bool, device=self.device)
        for _ in range(self.config.collision_projection_iterations):
            for row in range(row_count):
                row_gradient = gradient[:, row]
                row_direction = direction[:, row]
                directional_derivative = torch.sum(row_gradient * row_direction, dim=-1)
                value = torch.sum(row_gradient * corrected, dim=-1)
                required = (
                    active[:, row]
                    & ~overflow
                    & (directional_derivative > 1e-14)
                    & (value < lower_bound[:, row] - self.config.collision_velocity_tolerance_m_s)
                )
                required_step = torch.clamp_min(
                    (lower_bound[:, row] - value) / directional_derivative.clamp_min(1e-14),
                    0.0,
                )
                positive_limit = torch.where(
                    row_direction > 1e-12,
                    (upper32 - corrected) / row_direction.clamp_min(1e-12),
                    torch.full_like(row_direction, torch.inf),
                )
                negative_limit = torch.where(
                    row_direction < -1e-12,
                    (lower32 - corrected) / row_direction.clamp_max(-1e-12),
                    torch.full_like(row_direction, torch.inf),
                )
                maximum_step = torch.amin(
                    torch.minimum(positive_limit, negative_limit), dim=-1
                ).clamp_min(0.0)
                step = torch.minimum(required_step, maximum_step)
                candidate = corrected + step[:, None] * row_direction
                corrected = torch.where(required[:, None], candidate, corrected)
                applied |= required
        corrected = torch.where(overflow[:, None], torch.zeros_like(corrected), corrected)
        return corrected, applied

    def _project_contact_configuration(
        self,
        q: Any,
        candidate: Any,
        velocity: Any,
        target: Any,
        lower: Any,
        upper: Any,
        effective_dt: Any,
    ) -> tuple[Any, Any, Any, Any]:
        """Reintegrate a bounded velocity correction and validate contact anchors."""

        torch = self.torch
        full_pose, full_jacobian = self.kinematics._evaluate_trusted(candidate)
        pose, full_jacobian, _, _ = self._split_kinematics(full_pose, full_jacobian)
        position_vector, _, angular_vector, _ = self._errors_eager(pose, target)
        shaped = full_jacobian.reshape(self.batch_size, self.frame_count, 6, self.velocity_dim)
        error_chunks = []
        jacobian_chunks = []
        for frame_index, constrained in enumerate(self.frame_contact_constraints):
            if not constrained:
                continue
            error_chunks.append(position_vector[:, frame_index])
            jacobian_chunks.append(shaped[:, frame_index, :3])
            if self.frame_task_dimensions[frame_index] == 6:
                error_chunks.append(angular_vector[:, frame_index])
                jacobian_chunks.append(shaped[:, frame_index, 3:])
        if not error_chunks:
            valid = torch.ones(self.batch_size, dtype=torch.bool, device=self.device)
            return candidate, velocity, valid, valid
        error = torch.cat(error_chunks, dim=-1)
        jacobian = torch.cat(jacobian_chunks, dim=1)
        if self._cusolver_contact_inverse is not None:
            outputs = self._cusolver_contact_inverse.solve(
                jacobian.contiguous(), error.contiguous()
            )
            correction = (outputs[2] @ error.contiguous().unsqueeze(-1)).squeeze(-1)
            spectral_ok = outputs[3] == 0
        elif self._warp_contact_inverse is not None:
            correction = self._warp_contact_inverse.solve(jacobian.contiguous(), error.contiguous())
            spectral_ok = self._warp_contact_inverse.status[: self.batch_size] == 0
        else:
            correction = (
                torch.linalg.pinv(jacobian.to(torch.float64))
                @ error.to(torch.float64).unsqueeze(-1)
            ).squeeze(-1)
            spectral_ok = torch.ones(self.batch_size, dtype=torch.bool, device=self.device)
        correction_norm = torch.linalg.vector_norm(correction, dim=-1, keepdim=True)
        correction *= torch.clamp(0.02 / correction_norm.clamp_min(1e-12), max=1.0)
        corrected_velocity = velocity + correction.to(torch.float32) / effective_dt[:, None]
        corrected_velocity = torch.maximum(
            torch.minimum(corrected_velocity, upper.to(torch.float32)),
            lower.to(torch.float32),
        )
        if self.robot_spec.floating_base:
            corrected_candidate = self.kinematics._integrate_trusted(
                q,
                corrected_velocity * (effective_dt / self.config.dt)[:, None],
                self.config.dt,
            )
        else:
            corrected_candidate = q + effective_dt[:, None] * corrected_velocity
        corrected_pose = self._primary_pose(
            self.kinematics._evaluate_pose_trusted(corrected_candidate)
        )
        _, position, _, orientation = self._errors_eager(corrected_pose, target)
        valid = torch.ones(self.batch_size, dtype=torch.bool, device=self.device)
        for frame_index, constrained in enumerate(self.frame_contact_constraints):
            if constrained:
                valid &= position[:, frame_index] <= self.config.contact_position_tolerance_m
                if self.frame_task_dimensions[frame_index] == 6:
                    valid &= (
                        orientation[:, frame_index] <= self.config.contact_orientation_tolerance_rad
                    )
        return corrected_candidate, corrected_velocity, valid, spectral_ok

    def _task_state(self, pose: Any, target: Any) -> tuple[Any, Any, Any]:
        if self._compiled_task_state is None:
            return self._task_state_eager(pose, target)
        return self._compiled_task_state(pose, target)

    def _finalize_publication_eager(
        self,
        q: Any,
        q_start: Any,
        accepted_velocity: Any,
        position: Any,
        initial_position: Any,
        orientation: Any,
        initial_orientation: Any,
        converged: Any,
        nonworsening: Any,
        moved: Any,
        spectral_solve_ok: Any,
        initial_com_slack: Any,
        final_com_slack: Any,
        initial_collision_distance: Any,
        final_collision_distance: Any,
        initial_collision_active: Any,
        final_collision_active: Any,
        final_collision_overflow: Any,
        collision_overflow_observed: Any,
        collision_clear_state_certified: Any,
        collision_step_accepted: Any,
    ) -> tuple[Any, ...]:
        """Apply the exact ordered acceptance policy and safe-state selection."""

        torch = self.torch
        publish = converged | (self.config.allow_nonconverged_progress_steps & nonworsening & moved)
        publish &= spectral_solve_ok
        collision_diagnostic_publish = publish
        if self.com_constraint_enabled:
            initial_min = initial_com_slack.masked_fill(~self._com_active, float("inf")).amin(-1)
            final_min = final_com_slack.masked_fill(~self._com_active, float("inf")).amin(-1)
            publish |= (initial_min < -1e-4) & (final_min > initial_min + 1e-7)
            converged &= final_min >= -1e-4
        if self.collision is not None and self.config.collision_enabled:
            collision_safe = (
                final_collision_distance
                >= self.config.collision_min_distance_m - self.config.collision_tolerance_m
            ) & ~final_collision_overflow
            collision_recovery = (
                (
                    initial_collision_distance
                    < self.config.collision_min_distance_m - self.config.collision_tolerance_m
                )
                & (
                    final_collision_distance
                    > initial_collision_distance + self.config.collision_tolerance_m
                )
                & ~final_collision_overflow
            )
            publish = (publish & collision_safe) | collision_recovery
            converged &= collision_safe
            collision_diagnostic_publish = publish
            final_collision_distance = torch.where(
                publish, final_collision_distance, initial_collision_distance
            )
            final_collision_active = torch.where(
                publish, final_collision_active, initial_collision_active
            )
        if self.centroidal_enabled:
            publish &= self._com_candidate_valid(initial_com_slack, q, accepted_velocity)
        if self.com_constraint_enabled:
            final_com_slack = torch.where(publish[:, None], final_com_slack, initial_com_slack)
        # Preserve the last spectral veto after all recovery OR-composition.
        publish &= spectral_solve_ok
        safe_q = torch.where(publish[:, None], q, q_start)
        safe_velocity = torch.where(
            publish[:, None], accepted_velocity, torch.zeros_like(accepted_velocity)
        )
        published_position = torch.where(publish[:, None], position, initial_position)
        published_orientation = torch.where(publish[:, None], orientation, initial_orientation)
        clear_state = (
            collision_clear_state_certified & collision_step_accepted & ~collision_overflow_observed
            if self.config.collision_clear_state_fast_path_enabled
            else torch.zeros_like(collision_step_accepted)
        )
        return (
            converged,
            publish,
            collision_diagnostic_publish,
            safe_q,
            safe_velocity,
            published_position,
            published_orientation,
            final_com_slack,
            final_collision_distance,
            final_collision_active,
            clear_state,
        )

    def _finalize_publication(self, *args: Any) -> tuple[Any, ...]:
        if self._compiled_finalization is None:
            return self._finalize_publication_eager(*args)
        return self._compiled_finalization(*args)

    def _solve_impl(
        self,
        q_start: Any,
        target: Any,
        previous_velocity: Any | None,
        current_velocity: Any | None,
    ) -> MultiFramePoseBatchResult:
        torch = self.torch
        collision_active_for_solve = self.collision is not None and self.config.collision_enabled
        graph_safe_native = (
            self.config.standalone_cuda_graph_enabled
            and self.config.velocity_solver in {"warp_srinv", "cusolver_srinv"}
        )
        q = q_start.clone()
        previous = (
            torch.zeros(
                (self.batch_size, self.velocity_dim),
                dtype=torch.float32,
                device=self.device,
            )
            if previous_velocity is None
            else previous_velocity
        )
        measured_velocity = (
            torch.zeros(
                (self.batch_size, self.velocity_dim),
                dtype=torch.float32,
                device=self.device,
            )
            if current_velocity is None
            else current_velocity
        )
        accepted_velocity = torch.zeros(
            (self.batch_size, self.velocity_dim),
            dtype=torch.float32,
            device=self.device,
        )
        initial_position = None
        initial_orientation = None
        kernel_time_ms = 0.0
        initial_collision_distance = None
        initial_collision_active = None
        initial_collision_overflow = None
        initial_collision_point0 = None
        initial_collision_point1 = None
        initial_collision_shape_pair = None
        final_collision_distance = None
        final_collision_active = None
        final_collision_overflow = None
        final_collision_point0 = None
        final_collision_point1 = None
        final_collision_shape_pair = None
        collision_step_accepted = torch.ones(self.batch_size, dtype=torch.bool, device=self.device)
        collision_clear_state_certified = torch.ones_like(collision_step_accepted)
        collision_constraint_applied = torch.zeros_like(collision_step_accepted)
        collision_overflow_observed = torch.zeros_like(collision_step_accepted)
        torso_constraint_applied = torch.zeros_like(collision_step_accepted)
        torso_constraint_feasible = torch.ones_like(collision_step_accepted)
        com_constraint_applied = torch.zeros_like(collision_step_accepted)
        com_constraint_feasible = torch.ones_like(collision_step_accepted)
        initial_com_slack = None
        capture_constraint_applied = torch.zeros_like(collision_step_accepted)
        capture_constraint_feasible = torch.ones_like(collision_step_accepted)
        velocity_zmp_constraint_applied = torch.zeros_like(collision_step_accepted)
        velocity_zmp_constraint_feasible = torch.ones_like(collision_step_accepted)
        current_capture_slack = None
        current_zmp_slack = None
        current_zmp_force = None
        posture_task_applied = torch.zeros_like(collision_step_accepted)
        posture_primary_residual_increase = torch.zeros(
            self.batch_size, dtype=torch.float32, device=self.device
        )
        posture_secondary_residual_before = torch.zeros_like(posture_primary_residual_increase)
        posture_secondary_residual_after = torch.zeros_like(posture_primary_residual_increase)
        spectral_solve_ok = torch.ones_like(collision_step_accepted)
        effective_dt = torch.full(
            (self.batch_size,),
            self.config.dt,
            dtype=torch.float32,
            device=self.device,
        )
        for _ in range(self.config.iterations):
            full_pose, full_jacobian = self.kinematics._evaluate_trusted(q)
            centroidal_state = (
                self.com.evaluate_centroidal(self._centroidal_configuration(q))
                if self.centroidal_enabled
                else None
            )
            pose, jacobian, torso_pose, torso_jacobian = self._split_kinematics(
                full_pose, full_jacobian
            )
            contact_jacobian = self._contact_jacobian(jacobian)
            contact_projector, contact_spectral_ok = self._contact_projector(contact_jacobian)
            spectral_solve_ok &= contact_spectral_ok
            jacobian = self._select_task_jacobian(jacobian)
            position, orientation, twist = self._task_state(pose, target)
            if initial_position is None:
                initial_position = position
                initial_orientation = orientation
            prefetched_collision = None
            collision_margin = None
            clear_state_fast_path = bool(
                graph_safe_native and self.config.collision_clear_state_fast_path_enabled
            )
            if collision_active_for_solve and self.config.adaptive_dt:
                if (
                    clear_state_fast_path
                    and self.config.collision_current_convex_certificate_enabled
                ):
                    assert self.collision_convex_envelope is not None
                    self.collision_convex_envelope._query_trusted(
                        q,
                        compute_gradient=False,
                        probe_only=True,
                    )
                    prefetched_collision = self.collision.publish_convex_envelope_certificate(
                        self.collision_convex_envelope
                    )
                else:
                    prefetched_collision = self.collision._query_trusted(
                        q,
                        compute_gradient=(graph_safe_native and not clear_state_fast_path),
                        probe_only=clear_state_fast_path,
                    )
                collision_margin = (
                    prefetched_collision.distance_m - self.config.collision_min_distance_m
                )
            adaptive_scale = _adaptive_dt_scale(
                torch,
                torch.amax(position, dim=-1),
                enabled=self.config.adaptive_dt,
                max_scale=self.config.adaptive_dt_max_scale,
                reference_distance=self.config.adaptive_dt_reference_distance,
                collision_enabled=collision_active_for_solve,
                collision_margin=collision_margin,
                collision_tolerance_m=self.config.collision_tolerance_m,
            )
            effective_dt = self.config.dt * adaptive_scale
            lower, upper = self._bounds(q, q_start, previous, effective_dt)
            if self.fi is None:
                velocity = self._native_velocity(twist, jacobian, lower, upper)
                if self._warp_srinv is not None:
                    spectral_solve_ok &= self._warp_srinv.status[: self.batch_size] == 0
                if self._cusolver_primary_status is not None:
                    spectral_solve_ok &= self._cusolver_primary_status == 0
            else:
                kernel_time_ms += 1000.0 * self.fi._evaluate_trusted(
                    (
                        twist.to(self._fi_dtype).contiguous(),
                        jacobian.to(self._fi_dtype).reshape(self.batch_size, -1).contiguous(),
                        self._constraints,
                        lower.to(self._fi_dtype).contiguous(),
                        upper.to(self._fi_dtype).contiguous(),
                    )
                )
                velocity = self.fi.dense_output(0).squeeze(-1).to(torch.float32)
            (
                velocity,
                posture_applied,
                posture_primary_change,
                posture_before,
                posture_after,
                priority_spectral_ok,
            ) = self._apply_secondary_tasks(
                velocity,
                jacobian,
                q,
                full_pose,
                full_jacobian,
                lower,
                upper,
                None if centroidal_state is None else centroidal_state[2],
            )
            posture_task_applied |= posture_applied
            spectral_solve_ok &= priority_spectral_ok
            posture_primary_residual_increase = torch.maximum(
                posture_primary_residual_increase, posture_primary_change
            )
            posture_secondary_residual_before = posture_before
            posture_secondary_residual_after = posture_after
            velocity, torso_applied, torso_feasible, current_torso_state = (
                self._certify_torso_projection_noop(
                    velocity,
                    torso_pose,
                    torso_jacobian,
                    lower,
                    upper,
                )
                if self.config.torso_projection_noop_fast_path_enabled
                else self._apply_torso_constraint(
                    velocity,
                    torso_pose,
                    torso_jacobian,
                    lower,
                    upper,
                )
            )
            torso_constraint_applied |= torso_applied
            if collision_active_for_solve:
                primary_velocity = velocity
                current_collision = (
                    prefetched_collision
                    if prefetched_collision is not None
                    else (
                        self._certified_current_collision(q)
                        if (
                            clear_state_fast_path
                            and self.config.collision_current_convex_certificate_enabled
                        )
                        else self.collision._query_trusted(
                            q,
                            compute_gradient=(graph_safe_native and not clear_state_fast_path),
                            probe_only=clear_state_fast_path,
                        )
                    )
                )
                current_gradient_loaded = graph_safe_native and not clear_state_fast_path
                if not graph_safe_native and bool(
                    torch.any(current_collision.active & ~current_collision.overflow).item()
                ):
                    current_collision = self.collision._query_trusted(q)
                    current_gradient_loaded = True
                current_distance = current_collision.distance_m.clone()
                current_pair_distance = current_collision.pair_distance_m.clone()
                current_pair_gradient = current_collision.pair_gradient.clone()
                current_pair_active = current_collision.pair_active.clone()
                current_active = current_collision.active.clone()
                current_overflow = current_collision.overflow.clone()
                collision_clear_state_certified &= ~current_active & ~current_overflow
                collision_overflow_observed |= current_overflow
                if self.config.collision_debug_enabled:
                    current_point0 = current_collision.point0_world_m.clone()
                    current_point1 = current_collision.point1_world_m.clone()
                    current_shape_pair = current_collision.shape_pair.clone()
                if initial_collision_distance is None:
                    initial_collision_distance = current_distance.clone()
                    initial_collision_active = current_active.clone()
                    initial_collision_overflow = current_overflow.clone()
                    if self.config.collision_debug_enabled:
                        initial_collision_point0 = current_collision.point0_world_m.clone()
                        initial_collision_point1 = current_collision.point1_world_m.clone()
                        initial_collision_shape_pair = current_collision.shape_pair.clone()
                if current_gradient_loaded:
                    velocity, constraint_applied = self._apply_collision_rows(
                        primary_velocity,
                        current_pair_gradient,
                        current_pair_distance,
                        current_pair_active,
                        current_overflow,
                        lower,
                        upper,
                        current_pair_distance - self.config.collision_min_distance_m,
                        contact_projector,
                    )
                else:
                    constraint_applied = torch.zeros_like(current_active)
                collision_constraint_applied |= constraint_applied
            (
                velocity,
                com_applied,
                com_feasible,
                current_com_slack,
                current_capture_slack,
                current_zmp_slack,
                current_zmp_force,
            ) = self._apply_com_constraint(
                q,
                velocity,
                lower,
                upper,
                contact_projector,
                measured_velocity,
                centroidal_state,
            )
            com_constraint_applied |= com_applied
            capture_constraint_applied |= com_applied & self.capture_point_constraint_enabled
            velocity_zmp_constraint_applied |= com_applied & self.velocity_zmp_constraint_enabled
            if initial_com_slack is None:
                initial_com_slack = current_com_slack
            if self.robot_spec.floating_base:
                candidate = self.kinematics._integrate_trusted(
                    q,
                    velocity * adaptive_scale[:, None],
                    self.config.dt,
                )
            else:
                candidate = q + effective_dt[:, None] * velocity
                candidate = torch.maximum(
                    torch.minimum(candidate, q_start + self.config.max_joint_step_rad),
                    q_start - self.config.max_joint_step_rad,
                )
                candidate = torch.maximum(
                    torch.minimum(candidate, self._joint_upper), self._joint_lower
                )
            contact_configuration_valid = torch.ones(
                self.batch_size, dtype=torch.bool, device=self.device
            )
            if contact_jacobian is not None:
                (
                    candidate,
                    velocity,
                    contact_configuration_valid,
                    contact_spectral_ok,
                ) = self._project_contact_configuration(
                    q,
                    candidate,
                    velocity,
                    target,
                    lower,
                    upper,
                    effective_dt,
                )
                spectral_solve_ok &= contact_spectral_ok
            torso_candidate_valid = self._torso_candidate_valid(current_torso_state, candidate)
            com_candidate_valid = self._com_candidate_valid(current_com_slack, candidate, velocity)
            com_constraint_feasible &= com_feasible & com_candidate_valid
            if self.capture_point_constraint_enabled:
                capture_constraint_feasible &= com_feasible & com_candidate_valid
            if self.velocity_zmp_constraint_enabled:
                velocity_zmp_constraint_feasible &= com_feasible & com_candidate_valid
            if not collision_active_for_solve:
                accepted = (
                    contact_configuration_valid
                    & torso_feasible
                    & torso_candidate_valid
                    & com_feasible
                    & com_candidate_valid
                )
                q = torch.where(accepted[:, None], candidate, q)
                accepted_velocity = torch.where(accepted[:, None], velocity, accepted_velocity)
                torso_constraint_feasible &= accepted
                continue

            safe_pair_floor = torch.where(
                current_pair_distance
                >= self.config.collision_min_distance_m - self.config.collision_tolerance_m,
                torch.full_like(
                    current_pair_distance,
                    self.config.collision_min_distance_m - self.config.collision_tolerance_m,
                ),
                current_pair_distance - self.config.collision_tolerance_m,
            )
            if clear_state_fast_path and self.config.collision_candidate_convex_certificate_enabled:
                assert self.collision_convex_envelope is not None
                self.collision_convex_envelope._query_trusted(
                    candidate,
                    compute_gradient=False,
                    probe_only=True,
                )
                candidate_collision = self.collision.publish_convex_envelope_certificate(
                    self.collision_convex_envelope
                )
            else:
                candidate_collision = self.collision._query_trusted(
                    candidate,
                    compute_gradient=False,
                    probe_only=clear_state_fast_path,
                )
            repair_iterations = self.config.collision_max_constraints - 1
            if graph_safe_native:
                repair_iterations = min(
                    repair_iterations,
                    self.config.collision_graph_repair_iterations,
                )
            for _repair in range(repair_iterations):
                candidate_distance = candidate_collision.distance_m.clone()
                candidate_pair_distance = candidate_collision.pair_distance_m.clone()
                candidate_overflow = candidate_collision.overflow.clone()
                collision_overflow_observed |= candidate_overflow
                accepted = (
                    ~current_overflow
                    & ~candidate_overflow
                    & contact_configuration_valid
                    & torso_feasible
                    & torso_candidate_valid
                    & com_feasible
                    & com_candidate_valid
                    & torch.all(candidate_pair_distance >= safe_pair_floor, dim=-1)
                )
                repair_needed = (
                    ~accepted & candidate_collision.active & ~candidate_collision.overflow
                )
                if not graph_safe_native and not bool(torch.any(repair_needed).item()):
                    break
                accepted_candidate = candidate
                accepted_velocity_before_repair = velocity
                accepted_contact_valid = contact_configuration_valid
                accepted_torso_valid = torso_candidate_valid
                accepted_com_valid = com_candidate_valid
                if not current_gradient_loaded:
                    refreshed = self.collision._query_trusted(q)
                    current_pair_gradient = refreshed.pair_gradient.clone()
                    current_pair_active = refreshed.pair_active.clone()
                    current_gradient_loaded = True
                repair_score = torch.minimum(
                    current_pair_distance - self.config.collision_min_distance_m,
                    candidate_pair_distance - safe_pair_floor,
                )
                repaired_velocity, repair_applied = self._apply_collision_rows(
                    primary_velocity,
                    current_pair_gradient,
                    current_pair_distance,
                    current_pair_active,
                    current_overflow,
                    lower,
                    upper,
                    repair_score,
                    contact_projector,
                )
                collision_constraint_applied |= repair_applied & ~accepted
                (
                    repaired_velocity,
                    repair_com_applied,
                    repair_com_feasible,
                    _,
                    _,
                    _,
                    _,
                ) = self._apply_com_constraint(
                    q,
                    repaired_velocity,
                    lower,
                    upper,
                    contact_projector,
                    measured_velocity,
                    centroidal_state,
                )
                com_constraint_applied |= repair_com_applied & ~accepted
                com_feasible = torch.where(accepted, com_feasible, repair_com_feasible)
                velocity = torch.where(accepted[:, None], velocity, repaired_velocity)
                if self.robot_spec.floating_base:
                    candidate = self.kinematics._integrate_trusted(
                        q,
                        velocity * adaptive_scale[:, None],
                        self.config.dt,
                    )
                else:
                    candidate = q + effective_dt[:, None] * velocity
                    candidate = torch.maximum(
                        torch.minimum(
                            candidate,
                            q_start + self.config.max_joint_step_rad,
                        ),
                        q_start - self.config.max_joint_step_rad,
                    )
                    candidate = torch.maximum(
                        torch.minimum(candidate, self._joint_upper),
                        self._joint_lower,
                    )
                if current_gradient_loaded:
                    (
                        candidate,
                        velocity,
                        contact_configuration_valid,
                        contact_spectral_ok,
                    ) = self._project_contact_configuration(
                        q,
                        candidate,
                        velocity,
                        target,
                        lower,
                        upper,
                        effective_dt,
                    )
                    contact_spectral_ok = torch.where(
                        accepted,
                        torch.ones_like(contact_spectral_ok),
                        contact_spectral_ok,
                    )
                    spectral_solve_ok &= contact_spectral_ok
                candidate = torch.where(accepted[:, None], accepted_candidate, candidate)
                velocity = torch.where(accepted[:, None], accepted_velocity_before_repair, velocity)
                contact_configuration_valid = torch.where(
                    accepted, accepted_contact_valid, contact_configuration_valid
                )
                torso_candidate_valid = self._torso_candidate_valid(current_torso_state, candidate)
                com_candidate_valid = self._com_candidate_valid(
                    current_com_slack, candidate, velocity
                )
                torso_candidate_valid = torch.where(
                    accepted, accepted_torso_valid, torso_candidate_valid
                )
                com_candidate_valid = torch.where(accepted, accepted_com_valid, com_candidate_valid)
                candidate_collision = self.collision._query_trusted(
                    candidate, compute_gradient=False
                )
            candidate_distance = candidate_collision.distance_m.clone()
            candidate_pair_distance = candidate_collision.pair_distance_m.clone()
            candidate_overflow = candidate_collision.overflow.clone()
            collision_overflow_observed |= candidate_overflow
            accepted = (
                ~current_overflow
                & ~candidate_overflow
                & contact_configuration_valid
                & torso_feasible
                & torso_candidate_valid
                & com_feasible
                & com_candidate_valid
                & (
                    candidate_distance
                    >= self.config.collision_min_distance_m - self.config.collision_tolerance_m
                    if clear_state_fast_path
                    else torch.all(candidate_pair_distance >= safe_pair_floor, dim=-1)
                )
            )
            torso_constraint_feasible &= torso_feasible & torso_candidate_valid
            if self.capture_point_constraint_enabled:
                capture_constraint_feasible &= com_feasible & com_candidate_valid
            if self.velocity_zmp_constraint_enabled:
                velocity_zmp_constraint_feasible &= com_feasible & com_candidate_valid
            q = torch.where(accepted[:, None], candidate, q)
            accepted_velocity = torch.where(accepted[:, None], velocity, accepted_velocity)
            collision_step_accepted &= accepted
            final_collision_distance = torch.where(accepted, candidate_distance, current_distance)
            final_collision_active = torch.where(
                accepted, candidate_collision.active, current_active
            )
            final_collision_overflow = current_overflow | candidate_overflow
            if self.config.collision_debug_enabled:
                final_collision_point0 = torch.where(
                    accepted[:, None],
                    candidate_collision.point0_world_m,
                    current_point0,
                )
                final_collision_point1 = torch.where(
                    accepted[:, None],
                    candidate_collision.point1_world_m,
                    current_point1,
                )
                final_collision_shape_pair = torch.where(
                    accepted[:, None],
                    candidate_collision.shape_pair,
                    current_shape_pair,
                )

        pose = self._primary_pose(self.kinematics._evaluate_pose_trusted(q))
        _, position, _, orientation = self._errors(pose, target)
        converged = torch.all(
            (position <= self.config.position_tolerance_m)
            & (orientation <= self.config.orientation_tolerance_rad),
            dim=-1,
        )
        assert initial_position is not None
        assert initial_orientation is not None
        nonworsening = _multi_frame_nonworsening(
            torch,
            position,
            initial_position,
            orientation,
            initial_orientation,
            self.config.position_tolerance_m,
            self.config.orientation_tolerance_rad,
        )
        moved = torch.linalg.vector_norm(q - q_start, dim=-1) > 1e-7
        final_com_slack = None
        if self.com_constraint_enabled:
            final_com_slack, _ = self._com_state(q)
        if collision_active_for_solve:
            assert initial_collision_distance is not None
            assert initial_collision_overflow is not None
            assert final_collision_distance is not None
            assert final_collision_overflow is not None
            assert initial_collision_active is not None
        (
            converged,
            publish,
            collision_diagnostic_publish,
            safe_q,
            safe_velocity,
            published_position,
            published_orientation,
            final_com_slack,
            final_collision_distance,
            final_collision_active,
            collision_clear_state_certified,
        ) = self._finalize_publication(
            q,
            q_start,
            accepted_velocity,
            position,
            initial_position,
            orientation,
            initial_orientation,
            converged,
            nonworsening,
            moved,
            spectral_solve_ok,
            initial_com_slack,
            final_com_slack,
            initial_collision_distance,
            final_collision_distance,
            initial_collision_active,
            final_collision_active,
            final_collision_overflow,
            collision_overflow_observed,
            collision_clear_state_certified,
            collision_step_accepted,
        )
        if collision_active_for_solve:
            final_collision_overflow = collision_overflow_observed
            if self.config.collision_debug_enabled:
                assert initial_collision_point0 is not None
                assert initial_collision_point1 is not None
                assert initial_collision_shape_pair is not None
                final_collision_point0 = torch.where(
                    collision_diagnostic_publish[:, None],
                    final_collision_point0,
                    initial_collision_point0,
                )
                final_collision_point1 = torch.where(
                    collision_diagnostic_publish[:, None],
                    final_collision_point1,
                    initial_collision_point1,
                )
                final_collision_shape_pair = torch.where(
                    collision_diagnostic_publish[:, None],
                    final_collision_shape_pair,
                    initial_collision_shape_pair,
                )
        minimum_com_slack = (
            final_com_slack.masked_fill(~self._com_active, float("inf")).amin(-1).to(torch.float32)
            if self.com_constraint_enabled
            else None
        )
        minimum_capture_slack = None
        minimum_zmp_slack = None
        minimum_zmp_normal_force = None
        if self.centroidal_enabled:
            row_value = (
                self._com_step_rows @ safe_velocity.to(torch.float64).unsqueeze(-1)
            ).squeeze(-1)
            com_end = self._com_a.shape[1]
            capture_end = com_end + self._capture_a.shape[1]
            zmp_end = capture_end + self._zmp_a.shape[1] + 1
            if self.capture_point_constraint_enabled:
                cp_slack = (
                    self._com_step_upper[:, com_end:capture_end] - row_value[:, com_end:capture_end]
                )
                minimum_capture_slack = (
                    cp_slack.masked_fill(~self._capture_active, float("inf"))
                    .amin(-1)
                    .to(torch.float32)
                )
            if self.velocity_zmp_constraint_enabled:
                zmp_slack = (
                    self._com_step_upper[:, capture_end : zmp_end - 1]
                    - row_value[:, capture_end : zmp_end - 1]
                )
                minimum_zmp_slack = (
                    zmp_slack.masked_fill(~self._zmp_active, float("inf"))
                    .amin(-1)
                    .to(torch.float32)
                )
                minimum_zmp_normal_force = (
                    row_value[:, zmp_end - 1]
                    - self._com_step_lower[:, zmp_end - 1]
                    + self._zmp_settings[0]
                ).to(torch.float32)
        published_collision_distance = (
            final_collision_distance if collision_active_for_solve else None
        )
        published_collision_active = final_collision_active if collision_active_for_solve else None
        published_collision_accepted = (
            collision_step_accepted if collision_active_for_solve else None
        )
        published_collision_overflow = (
            final_collision_overflow if collision_active_for_solve else None
        )
        published_collision_applied = (
            collision_constraint_applied if collision_active_for_solve else None
        )

        return MultiFramePoseBatchResult(
            status="solved_or_held_needs_verification",
            q_solution=safe_q,
            q_candidate=q,
            position_error_m=published_position,
            orientation_error_rad=published_orientation,
            converged=converged,
            fallback_used=False,
            actual_device=str(self.device),
            kernel_time_ms=kernel_time_ms,
            com_constraint_enabled=self.com_constraint_enabled,
            com_constraint_applied=(
                com_constraint_applied if self.com_constraint_enabled else None
            ),
            com_constraint_feasible=(
                com_constraint_feasible if self.com_constraint_enabled else None
            ),
            minimum_com_slack_m=minimum_com_slack,
            capture_point_constraint_enabled=self.capture_point_constraint_enabled,
            capture_point_constraint_applied=(
                capture_constraint_applied if self.capture_point_constraint_enabled else None
            ),
            capture_point_constraint_feasible=(
                capture_constraint_feasible if self.capture_point_constraint_enabled else None
            ),
            minimum_capture_point_slack_m=minimum_capture_slack,
            velocity_zmp_constraint_enabled=self.velocity_zmp_constraint_enabled,
            velocity_zmp_constraint_applied=(
                velocity_zmp_constraint_applied if self.velocity_zmp_constraint_enabled else None
            ),
            velocity_zmp_constraint_feasible=(
                velocity_zmp_constraint_feasible if self.velocity_zmp_constraint_enabled else None
            ),
            minimum_velocity_zmp_slack_m=minimum_zmp_slack,
            minimum_zmp_normal_force_n=minimum_zmp_normal_force,
            centroidal_momentum_task_enabled=self.centroidal_momentum_task_enabled,
            centroidal_momentum_task_applied=(
                posture_task_applied if self.centroidal_momentum_task_enabled else None
            ),
            minimum_collision_distance_m=published_collision_distance,
            collision_active=published_collision_active,
            collision_step_accepted=published_collision_accepted,
            collision_overflow=published_collision_overflow,
            collision_constraint_applied=published_collision_applied,
            closest_collision_point0_world_m=(
                final_collision_point0
                if collision_active_for_solve and self.config.collision_debug_enabled
                else None
            ),
            closest_collision_point1_world_m=(
                final_collision_point1
                if collision_active_for_solve and self.config.collision_debug_enabled
                else None
            ),
            closest_collision_shape_pair=(
                final_collision_shape_pair
                if collision_active_for_solve and self.config.collision_debug_enabled
                else None
            ),
            collision_enabled=collision_active_for_solve,
            effective_dt=effective_dt,
            accepted_velocity=safe_velocity,
            torso_constraint_enabled=self.torso_constraint_enabled,
            torso_constraint_applied=(
                torso_constraint_applied if self.torso_constraint_enabled else None
            ),
            torso_constraint_feasible=(
                torso_constraint_feasible if self.torso_constraint_enabled else None
            ),
            posture_task_enabled=self.posture_task_enabled,
            posture_task_applied=(posture_task_applied if self.posture_task_enabled else None),
            posture_primary_residual_increase=(
                posture_primary_residual_increase if self.posture_task_enabled else None
            ),
            posture_secondary_residual_before=(
                posture_secondary_residual_before if self.posture_task_enabled else None
            ),
            posture_secondary_residual_after=(
                posture_secondary_residual_after if self.posture_task_enabled else None
            ),
            secondary_task_enabled=self.secondary_task_enabled,
            secondary_task_applied=(posture_task_applied if self.secondary_task_enabled else None),
            secondary_primary_residual_increase=(
                posture_primary_residual_increase if self.secondary_task_enabled else None
            ),
            secondary_residual_before=(
                posture_secondary_residual_before if self.secondary_task_enabled else None
            ),
            secondary_residual_after=(
                posture_secondary_residual_after if self.secondary_task_enabled else None
            ),
            spectral_solve_ok=spectral_solve_ok,
            compact_publication=None,
            collision_clear_state_certified=collision_clear_state_certified,
            solved_primary_pose_xyzw=pose,
        )
