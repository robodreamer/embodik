"""Public host and device adapters for model-derived GPU WBC."""

from __future__ import annotations

import hashlib
import json
import time
import warnings
from dataclasses import asdict, dataclass, replace
from pathlib import Path

FI_FUNCTION_NAMES = (
    "fn_fi_pesns_velocity_solve_panda_6",
    "fn_fi_pesns_velocity_solve",
)


@dataclass(frozen=True)
class GpuWbcCapabilities:
    """Feature-discovery contract for the experimental GPU runtime.

    A false field is an explicit capability boundary: callers should disable
    that control instead of assuming CPU semantics or silently falling back.
    """

    primary_pose_tasks: bool = True
    two_level_task_priority: bool = True
    joint_position_velocity_bounds: bool = True
    contact_frame_constraints: bool = True
    collision_constraints: bool = True
    collision_debug: bool = True
    posture_nullspace: bool = True
    torso_bounds: bool = True
    torso_staging: bool = True
    adaptive_dt: bool = True
    acceleration_limits: bool = True
    com_support_polygon: bool = True
    device_resident_batch: bool = True
    capture_point_constraints: bool = True
    velocity_zmp_constraints: bool = True
    centroidal_momentum_tasks: bool = True
    independent_world_reset: bool = True
    per_world_status: bool = True
    acceleration_level_tasks: bool = False
    task_axis_masks: bool = False
    joint_metrics: bool = False
    cpu_collision_policy_parity: bool = False


GPU_WBC_CAPABILITIES = GpuWbcCapabilities()


def gpu_wbc_collision_control_availability(
    *,
    gpu_active: bool,
    cpu_constraint_available: bool,
    cpu_debug_available: bool,
    gpu_constraint_available: bool = False,
    gpu_debug_available: bool = False,
) -> tuple[bool, bool]:
    """Return constraint/debug UI availability for the selected backend.

    Keeping this decision backend-specific prevents a GPU example from
    accidentally invoking CPU collision queries that do not constrain its
    result.
    """

    if gpu_active:
        return gpu_constraint_available, gpu_debug_available
    return cpu_constraint_available, cpu_debug_available


def derive_frame_active_joint_names(robot: object, frame: str) -> tuple[str, ...]:
    """Select scalar joints that can affect a frame, in velocity-index order."""

    import numpy as np

    jacobian = np.asarray(robot.get_frame_jacobian(frame), dtype=float)
    records = []
    for name_value in robot.get_joint_names():
        name = str(name_value)
        if (
            int(robot.get_joint_config_size(name)) != 1
            or int(robot.get_joint_velocity_size(name)) != 1
        ):
            continue
        velocity_index = int(robot.get_joint_velocity_index(name))
        if np.linalg.norm(jacobian[:, velocity_index]) > 1e-10:
            records.append((velocity_index, name))
    records.sort()
    if not records:
        raise ValueError(f"no scalar joints affect frame {frame!r}")
    return tuple(name for _, name in records)


def derive_frames_active_joint_names(robot: object, frames: tuple[str, ...]) -> tuple[str, ...]:
    """Select the ordered scalar-joint union that affects any requested frame."""

    selected = set()
    for frame in frames:
        selected.update(derive_frame_active_joint_names(robot, frame))
    return tuple(sorted(selected, key=lambda name: int(robot.get_joint_velocity_index(name))))


def derive_frames_active_velocity_indices(
    robot: object, frames: tuple[str, ...], *, tolerance: float = 1e-10
) -> tuple[int, ...]:
    """Return all tangent coordinates that affect any requested frame.

    Unlike :func:`derive_frames_active_joint_names`, this includes a floating
    root's six tangent coordinates.  It is therefore the model-neutral selector
    used by the floating-base factory.
    """

    import numpy as np

    if not frames:
        raise ValueError("at least one task frame is required")
    if not np.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("tolerance must be finite and non-negative")
    columns = np.zeros(int(robot.nv), dtype=bool)
    for frame in frames:
        jacobian = np.asarray(robot.get_frame_jacobian(frame), dtype=float)
        if jacobian.ndim != 2 or jacobian.shape[1] != int(robot.nv):
            raise ValueError(f"frame {frame!r} Jacobian must have {int(robot.nv)} columns")
        columns |= np.linalg.norm(jacobian, axis=0) > tolerance
    active = tuple(int(index) for index in np.flatnonzero(columns))
    if not active:
        raise ValueError("no model velocities affect the requested frames")
    return active


def derive_supported_active_joint_names(robot: object) -> tuple[str, ...]:
    """Return every scalar movable joint in source velocity order.

    This is the safe factory default because collision recovery, posture, CoM,
    torso, or secondary tasks may need a joint that has zero influence on the
    primary frame at the current configuration.
    """

    records = []
    for name_value in robot.get_joint_names():
        name = str(name_value)
        q_size = int(robot.get_joint_config_size(name))
        v_size = int(robot.get_joint_velocity_size(name))
        if q_size == 1 and v_size == 1:
            records.append((int(robot.get_joint_velocity_index(name)), name))
        elif v_size != 0:
            raise ValueError(
                f"GPU WBC currently supports scalar joints; {name!r} has "
                f"nq/nv={q_size}/{v_size}"
            )
    records.sort()
    if not records:
        raise ValueError("model has no supported movable scalar joints")
    return tuple(name for _, name in records)


@dataclass(frozen=True)
class GpuWbcCollisionDebug:
    """Closest active GPU collision pair in the shared world frame."""

    object_a: str
    object_b: str
    distance: float
    point_a_world: tuple[float, float, float]
    point_b_world: tuple[float, float, float]


@dataclass(frozen=True)
class GpuWbcInteractiveResult:
    """Compact host result for one interactive GPU-WBC control step."""

    joints: object
    status: str
    position_error: float
    rotation_error: float
    elapsed_ms: float
    kernel_time_ms: float
    minimum_collision_distance_m: float
    collision_active: bool
    collision_step_accepted: bool
    collision_overflow: bool
    collision_debug: GpuWbcCollisionDebug | None


@dataclass(frozen=True)
class GpuWbcMultiFrameResult:
    """Compact result for one model-derived multi-frame GPU step."""

    joints: object
    status: str
    position_errors: tuple[float, ...]
    rotation_errors: tuple[float, ...]
    elapsed_ms: float
    kernel_time_ms: float
    minimum_collision_distance_m: float | None = None
    collision_active: bool = False
    collision_step_accepted: bool = True
    collision_overflow: bool = False
    collision_constraint_applied: bool = False
    collision_clear_state_certified: bool = False
    collision_debug: GpuWbcCollisionDebug | None = None
    effective_dt: float | None = None
    com_constraint_enabled: bool = False
    com_constraint_applied: bool = False
    com_constraint_feasible: bool = True
    minimum_com_slack_m: float | None = None
    capture_point_constraint_enabled: bool = False
    capture_point_constraint_applied: bool = False
    capture_point_constraint_feasible: bool = True
    minimum_capture_point_slack_m: float | None = None
    velocity_zmp_constraint_enabled: bool = False
    velocity_zmp_constraint_applied: bool = False
    velocity_zmp_constraint_feasible: bool = True
    minimum_velocity_zmp_slack_m: float | None = None
    minimum_zmp_normal_force_n: float | None = None
    centroidal_momentum_task_enabled: bool = False
    centroidal_momentum_task_applied: bool = False
    torso_constraint_enabled: bool = False
    torso_constraint_applied: bool = False
    torso_constraint_feasible: bool = True
    posture_task_enabled: bool = False
    posture_task_applied: bool = False
    posture_primary_residual_increase: float | None = None
    posture_secondary_residual_before: float | None = None
    posture_secondary_residual_after: float | None = None
    secondary_task_enabled: bool = False
    secondary_task_applied: bool = False
    secondary_primary_residual_increase: float | None = None
    secondary_residual_before: float | None = None
    secondary_residual_after: float | None = None


@dataclass(frozen=True)
class GpuWbcResourceReport:
    """Solver-path host dispatch, synchronization, and CUDA memory snapshot.

    This is not a full-workload RL profile. Physics, observations, policy
    inference, and rendering remain application-owned measurements.
    """

    result: object
    host_dispatch_ms: float
    synchronization_ms: float
    kernel_time_ms: float
    allocated_bytes: int
    reserved_bytes: int


@dataclass(frozen=True)
class _GpuWbcPendingMultiFrameStep:
    """Device-resident solve awaiting a shared host publication boundary."""

    adapter: object
    q: object
    result: object
    compact: object
    moved: object
    start_event: object
    end_event: object


def _validate_posture_target(target: object, configuration_dim: int) -> None:
    import numpy as np

    if target is not None:
        values = np.asarray(target, dtype=float)
        if values.shape != (configuration_dim,) or not np.isfinite(values).all():
            raise ValueError(
                f"posture target must be finite with public configuration shape {(configuration_dim,)}"
            )


def _result_scalar(result: object, name: str, default=None):
    value = getattr(result, name, None)
    return default if value is None else value[0].item()


def _runtime_option_signature(options: dict) -> tuple:
    """Return a host-only signature for repeated shape-stable UI settings."""

    import numpy as np

    def freeze(value):
        if value is None or isinstance(value, (bool, int, float, str)):
            return value
        array = np.asarray(value)
        return (array.dtype.str, tuple(array.shape), array.tobytes())

    return tuple((name, freeze(value)) for name, value in sorted(options.items()))


def _validate_runtime_tasks(adapter, posture: dict, torso: dict) -> bool:
    """Validate before core setters mutate buffers; detect target changes."""
    import math

    torch = adapter._torch
    core = adapter._solver
    changed = False
    if any(value is not None for value in posture.values()):
        if not core.posture_task_enabled:
            raise RuntimeError("posture task must be configured at construction")
        gain = posture["gain"]
        if gain is not None and (not math.isfinite(gain) or gain < 0):
            raise ValueError("posture gain must be finite and nonnegative")
        target = posture["target_configuration"]
        _validate_posture_target(target, adapter.configuration_dim)
        if target is not None:
            values = torch.as_tensor(target, device=core.device, dtype=core._posture_target.dtype)
            changed |= bool(torch.any(torch.abs(values - core._posture_target) > 1e-7).item())
        weights = posture["weights"]
        if weights is not None:
            values = torch.as_tensor(weights, device=core.device)
            if values.shape != core._posture_weights.shape or not bool(
                (torch.isfinite(values) & (values >= 0)).all().item()
            ):
                raise ValueError("posture weights must retain shape and be finite and nonnegative")
    if any(value is not None for value in torso.values()):
        if not core.torso_constraint_enabled:
            raise RuntimeError("torso constraint must be configured at construction")
        candidates = {}
        for name, current in (
            ("reference_pose_xyzw", core._torso_reference_pose),
            ("lower_relative_limits", core._torso_lower_relative_limits),
            ("upper_relative_limits", core._torso_upper_relative_limits),
            ("axis_mask", core._torso_axis_mask),
        ):
            value = torso[name]
            candidate = (
                current
                if value is None
                else torch.as_tensor(value, dtype=current.dtype, device=core.device)
            )
            if candidate.shape != current.shape or not bool(torch.isfinite(candidate).all().item()):
                raise ValueError("torso updates must retain shape and be finite")
            candidates[name] = candidate
        reference = candidates["reference_pose_xyzw"]
        if not bool(torch.linalg.vector_norm(reference[3:]) > 1e-8):
            raise ValueError("torso reference quaternion must be nonzero")
        if bool(
            torch.any(candidates["lower_relative_limits"] >= candidates["upper_relative_limits"])
        ):
            raise ValueError("torso lower limits must remain below upper limits")
        changed |= _target_geometry_changed(torch, reference, core._torso_reference_pose)
    return changed


def _validate_runtime_secondary(adapter, updates: dict) -> bool:
    """Validate secondary-frame updates before mutating device buffers."""

    if not any(value is not None for value in updates.values()):
        return False
    core = adapter._solver
    if not core.secondary_frame_tasks_enabled:
        raise RuntimeError("secondary frame tasks must be configured at construction")
    torch = adapter._torch
    candidates = {}
    fields = (
        ("target_poses_wxyz", core._secondary_frame_targets),
        ("position_gains", core._secondary_frame_position_gains),
        ("orientation_gains", core._secondary_frame_orientation_gains),
        ("weights", core._secondary_frame_weights),
    )
    for name, current in fields:
        value = updates[name]
        candidate = (
            current
            if value is None
            else torch.as_tensor(value, dtype=current.dtype, device=core.device)
        )
        if candidate.shape != current.shape or not bool(torch.isfinite(candidate).all().item()):
            raise ValueError("secondary frame updates must retain shape and be finite")
        candidates[name] = candidate
    if bool(
        torch.any(
            torch.linalg.vector_norm(candidates["target_poses_wxyz"][:, 3:], dim=-1) <= 1e-8
        ).item()
    ):
        raise ValueError("secondary target quaternions must be nonzero")
    if bool(torch.any(candidates["position_gains"] <= 0.0).item()):
        raise ValueError("secondary position gains must remain positive")
    if bool(
        torch.any(candidates["orientation_gains"] < 0.0).item()
        or torch.any(candidates["weights"] < 0.0).item()
    ):
        raise ValueError("secondary orientation gains and weights must be nonnegative")
    return _target_geometry_changed(
        torch, candidates["target_poses_wxyz"], core._secondary_frame_targets
    )


def _target_geometry_changed(torch, target, previous) -> bool:
    # Quaternions q and -q represent the same orientation.
    translation_changed = torch.any(torch.abs(target[..., :3] - previous[..., :3]) > 1e-7)
    a, b = target[..., 3:], previous[..., 3:]
    rotation_changed = torch.any(
        torch.minimum(torch.amax(torch.abs(a - b), dim=-1), torch.amax(torch.abs(a + b), dim=-1))
        > 1e-7
    )
    return bool((translation_changed | rotation_changed).item())


def _host_target_geometry_changed(target, previous) -> bool:
    """Compare public target rows without a device synchronization."""
    import numpy as np

    if previous is None or target.shape != previous.shape:
        return True
    translation_changed = np.any(np.abs(target[..., :3] - previous[..., :3]) > 1e-7)
    a, b = target[..., 3:], previous[..., 3:]
    rotation_changed = np.any(
        np.minimum(np.max(np.abs(a - b), axis=-1), np.max(np.abs(a + b), axis=-1)) > 1e-7
    )
    return bool(translation_changed or rotation_changed)


class GpuWbcMultiFrameSolver:
    """Fail-closed fixed-base CUDA IK with model-derived constraints.

    Public configurations and posture targets contain only active coordinates,
    in active_joint_names order. Posture and locked indices refer to source
    model velocities; torso excluded indices refer to compact active columns.
    frame_task_dimensions describes each frame (3 or 6 rows); task_dimensions
    retains its legacy meaning as the generated FI artifact's task layout.
    """

    WORLD_STATUS_SUCCESS = 0
    WORLD_STATUS_INVALID_INPUT = 1
    WORLD_STATUS_HELD = 2
    WORLD_STATUS_NUMERICAL_FAILURE = 3
    WORLD_STATUS_INACTIVE = 4

    def __init__(
        self,
        manifest_path: Path | None,
        urdf_path: Path,
        cache_dir: Path,
        *,
        robot: object,
        robot_name: str,
        frames: tuple[str, ...],
        active_joint_names: tuple[str, ...],
        default_configuration: object,
        task_dimensions: tuple[int, ...] | None = None,
        batch_size: int = 1,
        iterations: int = 2,
        dt: float = 0.1,
        position_gain: float = 8.0,
        orientation_gain: float = 4.0,
        max_linear_speed: float = 2.0,
        max_angular_speed: float = 2.0,
        max_joint_acceleration_rad_s2: float | None = None,
        solver_backend: str = "fi_pesns",
        cusolver_specialized_outputs_enabled: bool = True,
        cusolver_reuse_locked_primary_inverse_enabled: bool = False,
        frame_task_dimensions: tuple[int, ...] | None = None,
        frame_contact_constraints: tuple[bool, ...] | None = None,
        frame_position_gains: tuple[float, ...] | None = None,
        frame_orientation_gains: tuple[float, ...] | None = None,
        adaptive_dt: bool = False,
        adaptive_dt_max_scale: float = 5.0,
        adaptive_dt_reference_distance: float = 0.05,
        collision_pairs: tuple[tuple[str, str], ...] = (),
        collision_min_distance_m: float = 0.05,
        collision_query_distance_m: float = 0.10,
        collision_contacts_per_world: int = 4096,
        collision_triangle_pairs_per_world: int = 1024,
        collision_contact_sort_enabled: bool = True,
        collision_max_constraints: int = 3,
        collision_graph_repair_iterations: int = 2,
        com_support_polygon_xy: object = None,
        com_full_robot: object = None,
        com_full_default_configuration: object = None,
        com_max_constraints: int = 8,
        com_excluded_velocity_indices: tuple[int, ...] = (),
        capture_point_support_polygon_xy: object = None,
        capture_point_max_constraints: int = 8,
        capture_point_margin: float = 0.0,
        capture_point_omega: float | None = None,
        capture_point_height: float | None = None,
        capture_point_gravity_z: float = -9.81,
        velocity_zmp_support_polygon_xy: object = None,
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
        posture_target_configuration: tuple[float, ...] | None = None,
        posture_velocity_indices: tuple[int, ...] = (),
        posture_weights: tuple[float, ...] = (),
        posture_gain: float = 1.0,
        posture_rank_tolerance: float = 1e-6,
        secondary_frame_names: tuple[str, ...] = (),
        secondary_frame_task_dimensions: tuple[int, ...] = (),
        secondary_frame_target_poses_wxyz: tuple[tuple[float, ...], ...] = (),
        secondary_frame_position_gains: tuple[float, ...] = (),
        secondary_frame_orientation_gains: tuple[float, ...] = (),
        secondary_frame_weights: tuple[float, ...] = (),
        locked_velocity_indices: tuple[int, ...] = (),
    ) -> None:
        try:
            import torch

            from ._runtime import (
                COMPACT_PUBLICATION_SCALARS,
                DeviceResidentMultiFramePoseSolver,
                MultiFramePoseSolveConfig,
            )
            from .model_spec import pose_model_parameters_from_embodik
        except ImportError as exc:
            raise RuntimeError(
                "multi-frame GPU WBC requires EmbodiK's GPU dependencies: CasADi, "
                "PyTorch CUDA, Newton, and Warp"
            ) from exc
        if not torch.cuda.is_available():
            raise RuntimeError("multi-frame GPU WBC requires CUDA")
        if type(batch_size) is not int or batch_size < 1:
            raise ValueError("batch_size must be a positive integer")
        if batch_size > 1 and solver_backend == "cusolver_srinv":
            raise ValueError(
                "cusolver_srinv currently supports batch_size=1; use warp_srinv "
                "for parallel worlds"
            )
        source = urdf_path.expanduser().resolve()
        params = pose_model_parameters_from_embodik(
            robot,
            name=robot_name,
            model_hash=hashlib.sha256(source.read_bytes()).hexdigest(),
            task_frames=tuple(frames),
            active_joint_names=tuple(active_joint_names),
            default_configuration=default_configuration,
        )
        if solver_backend == "fi_pesns":
            if manifest_path is None:
                raise ValueError("fi_pesns requires a generated manifest")
            function, library_path, artifact = _load_fi_artifact(manifest_path)
        elif solver_backend in {"torch_srinv", "warp_srinv", "cusolver_srinv"}:
            function = None
            library_path = None
            contract = (
                {"robot_spec": asdict(params.robot_spec), "task_layout": {}}
                if manifest_path is None
                else json.loads(manifest_path.expanduser().resolve().read_text())
            )
            layout = contract.get("task_layout", {})
            artifact = {
                "robot_spec": contract.get("robot_spec"),
                "shape": {
                    "velocity_dim": contract.get("robot_spec", {}).get("velocity_dim"),
                    "task_dimensions": layout.get("task_dimensions"),
                    "constraint_rows": layout.get("constraint_rows"),
                },
            }
        else:
            raise ValueError(
                "solver_backend must be fi_pesns, torch_srinv, warp_srinv, " "or cusolver_srinv"
            )
        artifact_spec = artifact.get("robot_spec")
        if artifact_spec is None:
            raise ValueError("multi-frame GPU WBC requires model-derived manifest metadata")
        shape = artifact.get("shape", {})
        com_params = None
        if com_full_robot is not None:
            com_params = pose_model_parameters_from_embodik(
                com_full_robot,
                name=robot_name,
                model_hash=params.robot_spec.model_hash,
                task_frames=tuple(frames),
                active_joint_names=tuple(active_joint_names),
                default_configuration=com_full_default_configuration,
            )
        mismatches = []
        if str(artifact_spec.get("model_hash")) != params.robot_spec.model_hash:
            mismatches.append("URDF content hash")
        if solver_backend == "fi_pesns" and tuple(artifact_spec.get("task_frames", ())) != tuple(
            frames
        ):
            mismatches.append("ordered task frames")
        recorded_active = tuple(artifact_spec.get("active_joint_names", ()))
        if recorded_active != tuple(active_joint_names):
            mismatches.append("active joint order")
        expected_task_dimensions = (
            (6,) * len(frames) if task_dimensions is None else tuple(task_dimensions)
        )
        if solver_backend == "fi_pesns" and sum(expected_task_dimensions) != 6 * len(frames):
            raise ValueError("multi-frame task dimensions must contain six rows per frame")
        if (
            solver_backend == "fi_pesns"
            and tuple(shape.get("task_dimensions", ())) != expected_task_dimensions
        ):
            mismatches.append("multi-frame task layout")
        if solver_backend == "fi_pesns" and int(shape.get("velocity_dim", -1)) != len(
            active_joint_names
        ):
            mismatches.append("velocity dimension")
        if solver_backend == "fi_pesns" and int(shape.get("constraint_rows", -1)) != len(
            active_joint_names
        ):
            mismatches.append("constraint rows")
        for field in ("configuration_dim", "velocity_dim", "active_velocity_indices"):
            recorded = artifact_spec.get(field)
            expected = getattr(params.robot_spec, field)
            if (
                tuple(recorded or ()) if field == "active_velocity_indices" else recorded
            ) != expected:
                mismatches.append(field)
        if mismatches:
            raise ValueError(
                "GPU manifest does not match the loaded multi-frame robot: " + ", ".join(mismatches)
            )
        self._torch = torch
        self.batch_size = batch_size
        self._compact_publication_scalars = COMPACT_PUBLICATION_SCALARS
        self.frames = tuple(frames)
        self.active_joint_names = tuple(active_joint_names)
        self.active_configuration_indices = tuple(
            int(robot.get_joint_config_index(name)) for name in active_joint_names
        )
        self.configuration_dim = len(active_joint_names)
        self.velocity_dim = self.configuration_dim
        self.active_velocity_indices = tuple(params.robot_spec.active_velocity_indices)
        _validate_posture_target(posture_target_configuration, self.configuration_dim)
        self._solver = DeviceResidentMultiFramePoseSolver(
            batch_size,
            function,
            library_path,
            source,
            cache_dir.expanduser().resolve(),
            robot_spec=params.robot_spec,
            frames=self.frames,
            frame_task_dimensions=frame_task_dimensions,
            frame_contact_constraints=frame_contact_constraints,
            frame_position_gains=frame_position_gains,
            frame_orientation_gains=frame_orientation_gains,
            torso_frame_name=torso_frame_name,
            torso_reference_pose_xyzw=torso_reference_pose_xyzw,
            torso_lower_relative_limits=torso_lower_relative_limits,
            torso_upper_relative_limits=torso_upper_relative_limits,
            torso_axis_mask=torso_axis_mask,
            torso_velocity_limits=torso_velocity_limits,
            torso_acceleration_limits=torso_acceleration_limits,
            torso_excluded_active_velocity_indices=torso_excluded_active_velocity_indices,
            torso_acceleration_history_enabled=torso_acceleration_history_enabled,
            torso_headroom_enabled=torso_headroom_enabled,
            posture_target_configuration=posture_target_configuration,
            posture_velocity_indices=posture_velocity_indices,
            posture_weights=posture_weights,
            posture_gain=posture_gain,
            posture_rank_tolerance=posture_rank_tolerance,
            secondary_frame_names=secondary_frame_names,
            secondary_frame_task_dimensions=secondary_frame_task_dimensions,
            secondary_frame_target_poses_wxyz=secondary_frame_target_poses_wxyz,
            secondary_frame_position_gains=secondary_frame_position_gains,
            secondary_frame_orientation_gains=secondary_frame_orientation_gains,
            secondary_frame_weights=secondary_frame_weights,
            locked_velocity_indices=locked_velocity_indices,
            collision_pairs=tuple(collision_pairs),
            default_configuration=params.default_configuration,
            com_support_polygon_xy=com_support_polygon_xy,
            com_robot_spec=None if com_params is None else com_params.robot_spec,
            com_default_configuration=(
                None if com_params is None else com_params.default_configuration
            ),
            com_max_constraints=com_max_constraints,
            com_excluded_velocity_indices=com_excluded_velocity_indices,
            capture_point_support_polygon_xy=capture_point_support_polygon_xy,
            capture_point_max_constraints=capture_point_max_constraints,
            capture_point_margin=capture_point_margin,
            capture_point_omega=capture_point_omega,
            capture_point_height=capture_point_height,
            capture_point_gravity_z=capture_point_gravity_z,
            velocity_zmp_support_polygon_xy=velocity_zmp_support_polygon_xy,
            velocity_zmp_max_constraints=velocity_zmp_max_constraints,
            velocity_zmp_margin=velocity_zmp_margin,
            velocity_zmp_fz_min=velocity_zmp_fz_min,
            velocity_zmp_gravity_z=velocity_zmp_gravity_z,
            centroidal_momentum_target=centroidal_momentum_target,
            centroidal_momentum_axis_mask=centroidal_momentum_axis_mask,
            centroidal_momentum_weight=centroidal_momentum_weight,
            centroidal_momentum_priority=centroidal_momentum_priority,
            centroidal_momentum_excluded_velocity_indices=(
                centroidal_momentum_excluded_velocity_indices
            ),
            centroidal_momentum_lower=centroidal_momentum_lower,
            centroidal_momentum_upper=centroidal_momentum_upper,
            joint_lower=params.joint_lower,
            joint_upper=params.joint_upper,
            joint_velocity_limits=params.joint_velocity_limits,
            config=MultiFramePoseSolveConfig(
                iterations=iterations,
                dt=dt,
                position_gain=position_gain,
                orientation_gain=orientation_gain,
                max_linear_speed=max_linear_speed,
                max_angular_speed=max_angular_speed,
                allow_nonconverged_progress_steps=True,
                standalone_cuda_graph_enabled=(
                    solver_backend in {"fi_pesns", "warp_srinv", "cusolver_srinv"}
                ),
                max_joint_acceleration_rad_s2=max_joint_acceleration_rad_s2,
                velocity_solver=solver_backend,
                cusolver_specialized_outputs_enabled=(
                    solver_backend == "cusolver_srinv" and cusolver_specialized_outputs_enabled
                ),
                cusolver_reuse_locked_primary_inverse_enabled=(
                    cusolver_reuse_locked_primary_inverse_enabled
                ),
                adaptive_dt=adaptive_dt,
                adaptive_dt_max_scale=adaptive_dt_max_scale,
                adaptive_dt_reference_distance=adaptive_dt_reference_distance,
                collision_enabled=bool(collision_pairs),
                collision_debug_enabled=(
                    bool(collision_pairs)
                    and solver_backend in {"fi_pesns", "warp_srinv", "cusolver_srinv"}
                ),
                collision_min_distance_m=collision_min_distance_m,
                collision_query_distance_m=collision_query_distance_m,
                collision_contacts_per_world=collision_contacts_per_world,
                collision_triangle_pairs_per_world=collision_triangle_pairs_per_world,
                collision_contact_sort_enabled=collision_contact_sort_enabled,
                collision_max_constraints=collision_max_constraints,
                collision_graph_repair_iterations=(collision_graph_repair_iterations),
            ),
        )
        self._previous_velocity = torch.zeros(
            (batch_size, self.configuration_dim),
            dtype=torch.float32,
            device=self._solver.device,
        )
        self._compact_publication_enabled = hasattr(self._solver, "compact_publication")

        self._last_target = None
        self._last_target_host = None
        self._last_runtime_option_signature = None

    @classmethod
    def from_robot(cls, urdf_path: Path, cache_dir: Path, **options):
        """Construct a native GPU solver directly from an EmbodiK model.

        ``manifest_path`` is intentionally omitted.  Warp, Torch, and
        cuSOLVER backends derive their specialization contract from ``robot``;
        only the legacy FI-PeSNS backend needs a generated artifact manifest.
        """

        if options.get("solver_backend", "warp_srinv") == "fi_pesns":
            raise ValueError("from_robot does not support the artifact-backed fi_pesns backend")
        if "active_joint_names" not in options:
            options["active_joint_names"] = derive_supported_active_joint_names(options["robot"])
        options.setdefault("solver_backend", "warp_srinv")
        return cls(None, urdf_path, cache_dir, **options)

    @property
    def device_label(self) -> str:
        return str(self._solver.device)

    @property
    def body_names(self) -> tuple[str, ...]:
        """Model body names corresponding to :meth:`evaluate_body_poses_device`."""

        return self._solver.kinematics.body_names

    def evaluate_body_poses_device(self, q):
        """Evaluate all model body poses for an existing CUDA configuration batch.

        The returned tensor has shape ``[batch_size, body_count, 7]`` and stores
        position followed by an XYZW quaternion. This is intended for batched
        visualization and diagnostics; it does not synchronize or copy to host.
        """

        return self._solver.kinematics.evaluate_body_poses(q)

    def reset_state(self, mask=None) -> None:
        GpuWbcFloatingMultiFrameSolver.reset_state(self, mask)
        if mask is None:
            self._last_runtime_option_signature = None

    def extract_active_configuration(self, configuration: object):
        import numpy as np

        return np.asarray(configuration, dtype=float)[
            np.asarray(self.active_configuration_indices, dtype=int)
        ].copy()

    def extract_active_velocity(self, velocity: object):
        """Extract measured tangent state in the GPU solver's active order."""
        import numpy as np

        values = np.asarray(velocity, dtype=float)
        return values[np.asarray(self.active_velocity_indices, dtype=int)].copy()

    def merge_active_configuration(self, configuration: object, active: object):
        import numpy as np

        result = np.asarray(configuration, dtype=float).copy()
        result[np.asarray(self.active_configuration_indices, dtype=int)] = np.asarray(
            active, dtype=float
        )
        return result

    def _target_tensor(self, targets: tuple[object, ...]):
        import numpy as np

        if len(targets) != len(self.frames):
            raise ValueError(f"expected {len(self.frames)} frame targets")
        rows = []
        for target in targets:
            xyzw = np.asarray(
                __import__("embodik").r2q(target.rotation, order="xyzs"),
                dtype=np.float32,
            )
            rows.append(
                np.concatenate(
                    (
                        np.asarray(target.translation, dtype=np.float32),
                        xyzw[[3, 0, 1, 2]],
                    )
                )
            )
        values = np.asarray(rows)
        self._current_target_host = values
        return self._torch.as_tensor(values, device=self._solver.device).reshape(
            1, len(self.frames), 7
        )

    def _q_tensor(self, joints: object):
        import numpy as np

        if self.batch_size != 1:
            raise RuntimeError("solve_step is a batch-1 host API; use solve_device_batch")
        values = np.asarray(joints, dtype=np.float32)
        if values.shape != (self.configuration_dim,):
            raise ValueError(f"expected active configuration {(self.configuration_dim,)}")
        return self._torch.as_tensor(values, device=self._solver.device).reshape(
            1, self.configuration_dim
        )

    def warm_up(self, joints: object, targets: tuple[object, ...]) -> float:
        q = self._q_tensor(joints)
        target = self._target_tensor(targets)
        self._torch.cuda.synchronize(self._solver.device)
        started = time.perf_counter()
        result = self._solver.solve(q, target, self._previous_velocity)
        self._torch.cuda.synchronize(self._solver.device)
        if result.fallback_used or not result.actual_device.startswith("cuda"):
            raise RuntimeError("multi-frame GPU WBC attribution contract failed")
        self._previous_velocity.zero_()
        return (time.perf_counter() - started) * 1e3

    def configure_runtime(self, **options) -> None:
        """Update shape-stable controls; see the floating adapter for options."""
        GpuWbcFloatingMultiFrameSolver.configure_runtime(self, **options)

    @property
    def collision_supported(self) -> bool:
        return self._solver.collision is not None

    def solve_step(
        self,
        joints: object,
        targets: tuple[object, ...],
        *,
        include_collision_debug: bool = False,
        current_velocity: object = None,
    ) -> GpuWbcMultiFrameResult:
        return GpuWbcFloatingMultiFrameSolver.solve_step(
            self,
            joints,
            targets,
            include_collision_debug=include_collision_debug,
            current_velocity=current_velocity,
        )

    def solve_device_batch(
        self,
        q,
        target,
        previous_velocity=None,
        current_velocity=None,
        *,
        reset_mask=None,
        valid_mask=None,
    ):
        """Solve an already device-resident batch without host publication."""

        return GpuWbcFloatingMultiFrameSolver.solve_device_batch(
            self,
            q,
            target,
            previous_velocity,
            current_velocity,
            reset_mask=reset_mask,
            valid_mask=valid_mask,
        )

    def measure_device_batch(
        self,
        q,
        target,
        previous_velocity=None,
        current_velocity=None,
        *,
        reset_mask=None,
        valid_mask=None,
    ) -> GpuWbcResourceReport:
        return GpuWbcFloatingMultiFrameSolver.measure_device_batch(
            self,
            q,
            target,
            previous_velocity,
            current_velocity,
            reset_mask=reset_mask,
            valid_mask=valid_mask,
        )


class GpuWbcFloatingMultiFrameSolver:
    """No-fallback public adapter for free-root mixed-frame CUDA pose IK."""

    WORLD_STATUS_SUCCESS = 0
    WORLD_STATUS_INVALID_INPUT = 1
    WORLD_STATUS_HELD = 2
    WORLD_STATUS_NUMERICAL_FAILURE = 3
    WORLD_STATUS_INACTIVE = 4

    def __init__(
        self,
        manifest_path: Path | None,
        urdf_path: Path,
        cache_dir: Path,
        *,
        robot: object,
        robot_name: str,
        frames: tuple[str, ...],
        frame_task_dimensions: tuple[int, ...],
        frame_contact_constraints: tuple[bool, ...] | None = None,
        frame_position_gains: tuple[float, ...],
        frame_orientation_gains: tuple[float, ...],
        active_velocity_indices: tuple[int, ...],
        default_configuration: object,
        base_velocity_limits: tuple[float, float, float, float, float, float],
        batch_size: int = 1,
        solver_backend: str = "torch_srinv",
        cusolver_specialized_outputs_enabled: bool = True,
        cusolver_reuse_locked_primary_inverse_enabled: bool = False,
        iterations: int = 2,
        dt: float = 0.01,
        position_gain: float = 60.0,
        orientation_gain: float = 60.0,
        max_linear_speed: float = 1.8,
        max_angular_speed: float = 2.5,
        adaptive_dt: bool = False,
        adaptive_dt_max_scale: float = 3.0,
        adaptive_dt_reference_distance: float = 0.04,
        max_joint_acceleration_rad_s2: float | None = None,
        collision_pairs: tuple[tuple[str, str], ...] = (),
        collision_min_distance_m: float = 0.05,
        collision_query_distance_m: float = 0.10,
        collision_contacts_per_world: int = 4096,
        collision_triangle_pairs_per_world: int = 1024,
        collision_contact_sort_enabled: bool = True,
        collision_clear_state_fast_path_enabled: bool = False,
        collision_candidate_convex_certificate_enabled: bool = False,
        collision_current_convex_certificate_enabled: bool = False,
        torso_projection_noop_fast_path_enabled: bool = False,
        collision_debug_enabled: bool = False,
        collision_max_constraints: int = 3,
        collision_graph_repair_iterations: int = 2,
        com_support_polygon_xy: object = None,
        com_max_constraints: int = 8,
        com_excluded_velocity_indices: tuple[int, ...] = (),
        capture_point_support_polygon_xy: object = None,
        capture_point_max_constraints: int = 8,
        capture_point_margin: float = 0.0,
        capture_point_omega: float | None = None,
        capture_point_height: float | None = None,
        capture_point_gravity_z: float = -9.81,
        velocity_zmp_support_polygon_xy: object = None,
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
        posture_target_configuration: tuple[float, ...] | None = None,
        posture_velocity_indices: tuple[int, ...] = (),
        posture_weights: tuple[float, ...] = (),
        posture_gain: float = 1.0,
        posture_rank_tolerance: float = 1e-6,
        secondary_frame_names: tuple[str, ...] = (),
        secondary_frame_task_dimensions: tuple[int, ...] = (),
        secondary_frame_target_poses_wxyz: tuple[tuple[float, ...], ...] = (),
        secondary_frame_position_gains: tuple[float, ...] = (),
        secondary_frame_orientation_gains: tuple[float, ...] = (),
        secondary_frame_weights: tuple[float, ...] = (),
        locked_velocity_indices: tuple[int, ...] = (),
    ) -> None:
        try:
            import torch

            from ._runtime import (
                COMPACT_PUBLICATION_SCALARS,
                DeviceResidentMultiFramePoseSolver,
                MultiFramePoseSolveConfig,
            )
            from .model_spec import floating_pose_model_parameters_from_embodik
        except ImportError as exc:
            raise RuntimeError(
                "floating multi-frame GPU WBC requires EmbodiK's GPU dependencies: "
                "PyTorch CUDA, Newton, and Warp"
            ) from exc
        if not torch.cuda.is_available():
            raise RuntimeError("floating multi-frame GPU WBC requires CUDA")
        if type(batch_size) is not int or batch_size < 1:
            raise ValueError("batch_size must be a positive integer")
        if batch_size > 1 and solver_backend == "cusolver_srinv":
            raise ValueError(
                "cusolver_srinv currently supports batch_size=1; use warp_srinv "
                "for parallel worlds"
            )
        if solver_backend not in {
            "torch_srinv",
            "warp_srinv",
            "cusolver_srinv",
        }:
            raise ValueError("solver_backend must be torch_srinv, warp_srinv, or " "cusolver_srinv")
        source = urdf_path.expanduser().resolve()
        params = floating_pose_model_parameters_from_embodik(
            robot,
            name=robot_name,
            model_hash=hashlib.sha256(source.read_bytes()).hexdigest(),
            task_frames=tuple(frames),
            active_velocity_indices=tuple(active_velocity_indices),
            default_configuration=default_configuration,
            base_velocity_limits=base_velocity_limits,
        )
        contract = (
            {"robot_spec": asdict(params.robot_spec)}
            if manifest_path is None
            else json.loads(manifest_path.expanduser().resolve().read_text())
        )
        artifact_spec = contract.get("robot_spec", {})
        mismatches = []
        if str(artifact_spec.get("model_hash")) != params.robot_spec.model_hash:
            mismatches.append("URDF content hash")
        if tuple(artifact_spec.get("active_velocity_indices", ())) != tuple(
            active_velocity_indices
        ):
            mismatches.append("active tangent velocity order")
        if int(artifact_spec.get("configuration_dim", -1)) != int(robot.nq):
            mismatches.append("configuration dimension")
        if int(artifact_spec.get("velocity_dim", -1)) != int(robot.nv):
            mismatches.append("model velocity dimension")
        if mismatches:
            raise ValueError(
                "GPU manifest does not match the floating-base robot: " + ", ".join(mismatches)
            )

        self._torch = torch
        self.batch_size = batch_size
        self._pending_start_event = torch.cuda.Event(enable_timing=True)
        self._pending_end_event = torch.cuda.Event(enable_timing=True)
        self.collision_clear_state_fast_path_enabled = bool(collision_clear_state_fast_path_enabled)
        self._compact_publication_scalars = COMPACT_PUBLICATION_SCALARS
        self.frames = tuple(frames)
        self.configuration_dim = int(robot.nq)
        self.velocity_dim = len(active_velocity_indices)
        self.active_velocity_indices = tuple(int(v) for v in active_velocity_indices)
        _validate_posture_target(posture_target_configuration, self.configuration_dim)
        self._solver = DeviceResidentMultiFramePoseSolver(
            batch_size,
            None,
            None,
            source,
            cache_dir.expanduser().resolve(),
            robot_spec=params.robot_spec,
            frames=self.frames,
            frame_task_dimensions=tuple(frame_task_dimensions),
            frame_contact_constraints=frame_contact_constraints,
            frame_position_gains=tuple(frame_position_gains),
            frame_orientation_gains=tuple(frame_orientation_gains),
            default_configuration=params.default_configuration,
            joint_lower=params.joint_lower,
            joint_upper=params.joint_upper,
            joint_velocity_limits=params.joint_velocity_limits,
            config=MultiFramePoseSolveConfig(
                iterations=iterations,
                dt=dt,
                adaptive_dt=adaptive_dt,
                adaptive_dt_max_scale=adaptive_dt_max_scale,
                adaptive_dt_reference_distance=adaptive_dt_reference_distance,
                position_gain=position_gain,
                orientation_gain=orientation_gain,
                max_linear_speed=max_linear_speed,
                max_angular_speed=max_angular_speed,
                max_joint_acceleration_rad_s2=max_joint_acceleration_rad_s2,
                allow_nonconverged_progress_steps=True,
                standalone_cuda_graph_enabled=(solver_backend in {"warp_srinv", "cusolver_srinv"}),
                velocity_solver=solver_backend,
                cusolver_specialized_outputs_enabled=(
                    solver_backend == "cusolver_srinv" and cusolver_specialized_outputs_enabled
                ),
                cusolver_reuse_locked_primary_inverse_enabled=(
                    cusolver_reuse_locked_primary_inverse_enabled
                ),
                collision_enabled=bool(collision_pairs),
                collision_debug_enabled=(
                    bool(collision_pairs)
                    and solver_backend in {"warp_srinv", "cusolver_srinv"}
                    and collision_debug_enabled
                ),
                collision_max_constraints=collision_max_constraints,
                collision_min_distance_m=collision_min_distance_m,
                collision_query_distance_m=collision_query_distance_m,
                collision_contacts_per_world=collision_contacts_per_world,
                collision_triangle_pairs_per_world=(collision_triangle_pairs_per_world),
                collision_contact_sort_enabled=collision_contact_sort_enabled,
                collision_clear_state_fast_path_enabled=(collision_clear_state_fast_path_enabled),
                collision_candidate_convex_certificate_enabled=(
                    collision_candidate_convex_certificate_enabled
                ),
                collision_current_convex_certificate_enabled=(
                    collision_current_convex_certificate_enabled
                ),
                torso_projection_noop_fast_path_enabled=(torso_projection_noop_fast_path_enabled),
                collision_graph_repair_iterations=(collision_graph_repair_iterations),
            ),
            collision_pairs=tuple(collision_pairs),
            torso_frame_name=torso_frame_name,
            com_support_polygon_xy=com_support_polygon_xy,
            com_max_constraints=com_max_constraints,
            com_excluded_velocity_indices=com_excluded_velocity_indices,
            capture_point_support_polygon_xy=capture_point_support_polygon_xy,
            capture_point_max_constraints=capture_point_max_constraints,
            capture_point_margin=capture_point_margin,
            capture_point_omega=capture_point_omega,
            capture_point_height=capture_point_height,
            capture_point_gravity_z=capture_point_gravity_z,
            velocity_zmp_support_polygon_xy=velocity_zmp_support_polygon_xy,
            velocity_zmp_max_constraints=velocity_zmp_max_constraints,
            velocity_zmp_margin=velocity_zmp_margin,
            velocity_zmp_fz_min=velocity_zmp_fz_min,
            velocity_zmp_gravity_z=velocity_zmp_gravity_z,
            centroidal_momentum_target=centroidal_momentum_target,
            centroidal_momentum_axis_mask=centroidal_momentum_axis_mask,
            centroidal_momentum_weight=centroidal_momentum_weight,
            centroidal_momentum_priority=centroidal_momentum_priority,
            centroidal_momentum_excluded_velocity_indices=(
                centroidal_momentum_excluded_velocity_indices
            ),
            centroidal_momentum_lower=centroidal_momentum_lower,
            centroidal_momentum_upper=centroidal_momentum_upper,
            torso_reference_pose_xyzw=torso_reference_pose_xyzw,
            torso_lower_relative_limits=torso_lower_relative_limits,
            torso_upper_relative_limits=torso_upper_relative_limits,
            torso_axis_mask=torso_axis_mask,
            torso_velocity_limits=torso_velocity_limits,
            torso_acceleration_limits=torso_acceleration_limits,
            torso_excluded_active_velocity_indices=torso_excluded_active_velocity_indices,
            torso_acceleration_history_enabled=torso_acceleration_history_enabled,
            torso_headroom_enabled=torso_headroom_enabled,
            posture_target_configuration=posture_target_configuration,
            posture_velocity_indices=posture_velocity_indices,
            posture_weights=posture_weights,
            posture_gain=posture_gain,
            posture_rank_tolerance=posture_rank_tolerance,
            secondary_frame_names=secondary_frame_names,
            secondary_frame_task_dimensions=secondary_frame_task_dimensions,
            secondary_frame_target_poses_wxyz=secondary_frame_target_poses_wxyz,
            secondary_frame_position_gains=secondary_frame_position_gains,
            secondary_frame_orientation_gains=secondary_frame_orientation_gains,
            secondary_frame_weights=secondary_frame_weights,
            locked_velocity_indices=locked_velocity_indices,
        )
        self._previous_velocity = torch.zeros(
            (batch_size, self.velocity_dim),
            dtype=torch.float32,
            device=self._solver.device,
        )
        self._compact_publication_enabled = hasattr(self._solver, "compact_publication")
        self._last_target = None
        self._last_target_host = None
        self._last_runtime_option_signature = None

    @classmethod
    def from_robot(cls, urdf_path: Path, cache_dir: Path, **options):
        """Construct a native floating-base GPU solver from model metadata."""

        if "active_velocity_indices" not in options:
            options["active_velocity_indices"] = tuple(range(int(options["robot"].nv)))
        options.setdefault("solver_backend", "warp_srinv")
        return cls(None, urdf_path, cache_dir, **options)

    @property
    def device_label(self) -> str:
        return str(self._solver.device)

    @property
    def body_names(self) -> tuple[str, ...]:
        """Model body names corresponding to :meth:`evaluate_body_poses_device`."""

        return self._solver.kinematics.body_names

    def evaluate_body_poses_device(self, q):
        """Evaluate all model body poses for an existing CUDA configuration batch.

        The returned tensor has shape ``[batch_size, body_count, 7]`` and stores
        position followed by an XYZW quaternion. This is intended for batched
        visualization and diagnostics; it does not synchronize or copy to host.
        """

        return self._solver.kinematics.evaluate_body_poses(q)

    @property
    def collision_supported(self) -> bool:
        return self._solver.collision is not None

    def reset_state(self, mask=None) -> None:
        """Clear accepted-velocity history for the full batch or selected worlds.

        ``mask`` is a boolean CUDA tensor of shape ``[batch_size]``. True worlds
        lose command-acceleration history; False worlds keep their current
        accepted velocity. A full-batch reset also forgets the last target used
        by the interactive host path.
        """

        if mask is None:
            self._previous_velocity.zero_()
            self._last_target = None
            self._last_target_host = None
            return
        selected = self._world_mask(mask, "reset_mask")
        self._previous_velocity.masked_fill_(selected.unsqueeze(-1), 0)

    def _world_mask(self, value, name: str):
        if value is None:
            return None
        torch = self._torch
        if not isinstance(value, torch.Tensor) or value.dtype != torch.bool:
            raise TypeError(f"{name} must be a boolean torch.Tensor")
        device = self._solver.device
        if value.device != device:
            raise ValueError(f"{name} must reside on {device}")
        expected = (int(self.batch_size),)
        if tuple(value.shape) != expected:
            raise ValueError(f"{name} must have shape {expected}")
        return value

    def extract_active_velocity(self, velocity: object):
        """Extract measured tangent state in the GPU solver's active order."""
        import numpy as np

        values = np.asarray(velocity, dtype=float)
        return values[np.asarray(self.active_velocity_indices, dtype=int)].copy()

    def configure_runtime(
        self,
        *,
        dt: float | None = None,
        iterations: int | None = None,
        position_gain: float | None = None,
        orientation_gain: float | None = None,
        frame_position_gains: tuple[float, ...] | None = None,
        frame_orientation_gains: tuple[float, ...] | None = None,
        adaptive_dt: bool | None = None,
        adaptive_dt_max_scale: float | None = None,
        adaptive_dt_reference_distance: float | None = None,
        acceleration_limits_enabled: bool | None = None,
        max_joint_acceleration_rad_s2: float | None = None,
        collision_enabled: bool | None = None,
        collision_min_distance_m: float | None = None,
        collision_query_distance_m: float | None = None,
        collision_tolerance_m: float | None = None,
        com_enabled: bool | None = None,
        com_support_polygon_xy: object = None,
        com_margin: float | None = None,
        com_vel_max: float | None = None,
        com_acc_max: float | None = None,
        com_use_acceleration_limits: bool | None = None,
        com_proximity_fraction: float | None = None,
        capture_point_enabled: bool | None = None,
        capture_point_support_polygon_xy: object = None,
        capture_point_margin: float | None = None,
        capture_point_omega: float | None = None,
        capture_point_height: float | None = None,
        capture_point_gravity_z: float | None = None,
        velocity_zmp_enabled: bool | None = None,
        velocity_zmp_support_polygon_xy: object = None,
        velocity_zmp_margin: float | None = None,
        velocity_zmp_fz_min: float | None = None,
        velocity_zmp_gravity_z: float | None = None,
        centroidal_momentum_enabled: bool | None = None,
        centroidal_momentum_target: tuple[float, ...] | None = None,
        centroidal_momentum_axis_mask: tuple[bool, ...] | None = None,
        centroidal_momentum_weight: float | None = None,
        centroidal_momentum_priority: int | None = None,
        centroidal_momentum_excluded_velocity_indices: tuple[int, ...] | None = None,
        centroidal_momentum_lower: tuple[float, ...] | None = None,
        centroidal_momentum_upper: tuple[float, ...] | None = None,
        posture_target_configuration: tuple[float, ...] | None = None,
        posture_weights: tuple[float, ...] | None = None,
        posture_gain: float | None = None,
        torso_reference_pose_xyzw: tuple[float, ...] | None = None,
        torso_lower_relative_limits: tuple[float, ...] | None = None,
        torso_upper_relative_limits: tuple[float, ...] | None = None,
        torso_axis_mask: tuple[bool, ...] | None = None,
        secondary_frame_target_poses_wxyz: tuple[tuple[float, ...], ...] | None = None,
        secondary_frame_position_gains: tuple[float, ...] | None = None,
        secondary_frame_orientation_gains: tuple[float, ...] | None = None,
        secondary_frame_weights: tuple[float, ...] | None = None,
    ) -> None:
        """Update existing tasks without changing row counts or active order.

        Scalar gains retain the original first-frame/tool semantics. Use the
        per-frame options to update every frame. Fixed-base posture targets use
        compact active coordinates; posture indices always use source velocities.
        Collision query capacity is fixed at construction.
        """
        runtime_signature = _runtime_option_signature(
            {name: value for name, value in locals().items() if name != "self"}
        )
        if runtime_signature == self._last_runtime_option_signature:
            return

        import math

        acceleration_limit = self._solver.config.max_joint_acceleration_rad_s2
        if max_joint_acceleration_rad_s2 is not None:
            if (
                not math.isfinite(max_joint_acceleration_rad_s2)
                or max_joint_acceleration_rad_s2 <= 0.0
            ):
                raise ValueError("maximum joint acceleration must be positive and finite")
            acceleration_limit = float(max_joint_acceleration_rad_s2)
        if acceleration_limits_enabled is False:
            acceleration_limit = None
        elif acceleration_limits_enabled is True and acceleration_limit is None:
            raise ValueError("enabling acceleration limits requires a configured maximum")
        updates = {
            key: value
            for key, value in {
                "dt": dt,
                "iterations": iterations,
                "adaptive_dt": adaptive_dt,
                "adaptive_dt_max_scale": adaptive_dt_max_scale,
                "adaptive_dt_reference_distance": adaptive_dt_reference_distance,
                "collision_enabled": collision_enabled,
                "collision_min_distance_m": collision_min_distance_m,
                "collision_query_distance_m": collision_query_distance_m,
                "collision_tolerance_m": collision_tolerance_m,
            }.items()
            if value is not None
        }
        if acceleration_limits_enabled is not None or max_joint_acceleration_rad_s2 is not None:
            updates["max_joint_acceleration_rad_s2"] = acceleration_limit
        # Newton allocates its query envelope at construction. A replacement
        # config alone would silently leave the actual query radius unchanged.
        if collision_query_distance_m is not None and (
            collision_query_distance_m != self._solver.config.collision_query_distance_m
        ):
            raise ValueError("changing collision query distance requires rebuilding the solver")
        if collision_enabled and not self.collision_supported:
            raise ValueError("collision pairs must be configured at construction")
        if (
            any(
                value is not None
                for value in (
                    collision_min_distance_m,
                    collision_tolerance_m,
                )
            )
            and not self.collision_supported
        ):
            raise ValueError("collision pairs must be configured at construction")
        previous_config = self._solver.config
        config = replace(previous_config, **updates)
        gain_updates = []
        for values, scalar, destination, allow_zero in (
            (
                frame_position_gains,
                position_gain,
                self._solver._frame_position_gains,
                False,
            ),
            (
                frame_orientation_gains,
                orientation_gain,
                self._solver._frame_orientation_gains,
                True,
            ),
        ):
            if values is not None and scalar is not None:
                raise ValueError("specify either scalar or per-frame gains")
            if values is None and scalar is None:
                continue
            selected = tuple(values) if values is not None else (scalar,)
            if values is not None and len(selected) != len(self.frames):
                raise ValueError("per-frame gains must match frames")
            if not all(math.isfinite(v) and (v >= 0 if allow_zero else v > 0) for v in selected):
                raise ValueError("frame gains must be finite and within range")
            gain_updates.append((destination, selected, values is not None))
        posture = {
            "target_configuration": posture_target_configuration,
            "weights": posture_weights,
            "gain": posture_gain,
        }
        torso = {
            "reference_pose_xyzw": torso_reference_pose_xyzw,
            "lower_relative_limits": torso_lower_relative_limits,
            "upper_relative_limits": torso_upper_relative_limits,
            "axis_mask": torso_axis_mask,
        }
        secondary = {
            "target_poses_wxyz": secondary_frame_target_poses_wxyz,
            "position_gains": secondary_frame_position_gains,
            "orientation_gains": secondary_frame_orientation_gains,
            "weights": secondary_frame_weights,
        }
        target_changed = _validate_runtime_tasks(self, posture, torso)
        target_changed |= _validate_runtime_secondary(self, secondary)
        com_options = dict(
            enabled=com_enabled,
            support_polygon_xy=com_support_polygon_xy,
            margin=com_margin,
            vel_max=com_vel_max,
            acc_max=com_acc_max,
            use_acceleration_limits=com_use_acceleration_limits,
            proximity_fraction=com_proximity_fraction,
        )
        if any(v is not None for v in com_options.values()):
            self._solver.configure_com_constraint(**com_options)
        capture_options = dict(
            enabled=capture_point_enabled,
            support_polygon_xy=capture_point_support_polygon_xy,
            margin=capture_point_margin,
            omega=capture_point_omega,
            height=capture_point_height,
            gravity_z=capture_point_gravity_z,
        )
        if any(v is not None for v in capture_options.values()):
            self._solver.configure_capture_point_constraint(**capture_options)
        zmp_options = dict(
            enabled=velocity_zmp_enabled,
            support_polygon_xy=velocity_zmp_support_polygon_xy,
            margin=velocity_zmp_margin,
            fz_min=velocity_zmp_fz_min,
            gravity_z=velocity_zmp_gravity_z,
        )
        if any(v is not None for v in zmp_options.values()):
            self._solver.configure_velocity_zmp_constraint(**zmp_options)
        if (
            centroidal_momentum_priority is not None
            and int(centroidal_momentum_priority) != self._solver._momentum_priority
        ):
            raise ValueError("changing centroidal momentum priority requires rebuilding the solver")
        if centroidal_momentum_excluded_velocity_indices is not None:
            requested = set(int(v) for v in centroidal_momentum_excluded_velocity_indices)
            configured = {
                index
                for index, excluded in zip(
                    self.active_velocity_indices,
                    self._solver._momentum_excluded.detach().cpu().tolist(),
                    strict=True,
                )
                if excluded
            }
            if requested != configured:
                raise ValueError(
                    "changing centroidal momentum excluded velocities requires rebuilding the solver"
                )
        momentum_options = dict(
            enabled=centroidal_momentum_enabled,
            target=centroidal_momentum_target,
            axis_mask=centroidal_momentum_axis_mask,
            weight=centroidal_momentum_weight,
            lower=centroidal_momentum_lower,
            upper=centroidal_momentum_upper,
        )
        if any(v is not None for v in momentum_options.values()):
            self._solver.configure_centroidal_momentum(**momentum_options)
        if any(v is not None for v in posture.values()):
            _validate_posture_target(posture_target_configuration, self.configuration_dim)
            self._solver.configure_posture(**posture)
        if any(v is not None for v in torso.values()):
            self._solver.configure_torso_constraint(**torso)
        if any(v is not None for v in secondary.values()):
            self._solver.configure_secondary_frames(**secondary)
        self._solver.config = config
        if config != previous_config and config.standalone_cuda_graph_enabled:
            # Python config values become constants in captured control flow.
            # Recapture policy changes instead of replaying a stale branch.
            self._solver._graph = None
        if target_changed:
            self.reset_state()
        for destination, selected, all_frames in gain_updates:
            if all_frames:
                destination.copy_(
                    self._torch.as_tensor(
                        selected, dtype=destination.dtype, device=destination.device
                    )
                )
            else:
                destination[0] = selected[0]
        self._last_runtime_option_signature = runtime_signature

    def _q_tensor(self, configuration: object):
        import numpy as np

        if self.batch_size != 1:
            raise RuntimeError("solve_step is a batch-1 host API; use solve_device_batch")
        values = np.asarray(configuration, dtype=np.float32)
        if values.shape != (self.configuration_dim,):
            raise ValueError(f"expected configuration {(self.configuration_dim,)}")
        return self._torch.as_tensor(values, device=self._solver.device).reshape(
            1, self.configuration_dim
        )

    def _target_tensor(self, targets: tuple[object, ...]):
        import numpy as np

        if len(targets) != len(self.frames):
            raise ValueError(f"expected {len(self.frames)} frame targets")
        rows = []
        for target in targets:
            xyzw = np.asarray(
                __import__("embodik").r2q(target.rotation, order="xyzs"),
                dtype=np.float32,
            )
            rows.append(
                np.concatenate(
                    (
                        np.asarray(target.translation, dtype=np.float32),
                        xyzw[[3, 0, 1, 2]],
                    )
                )
            )
        values = np.asarray(rows)
        self._current_target_host = values
        return self._torch.as_tensor(values, device=self._solver.device).reshape(
            1, len(self.frames), 7
        )

    def warm_up(self, configuration: object, targets: tuple[object, ...]) -> float:
        q = self._q_tensor(configuration)
        target = self._target_tensor(targets)
        self._torch.cuda.synchronize(self._solver.device)
        started = time.perf_counter()
        with warnings.catch_warnings():
            # We intentionally retain highest FP32 precision; enabling TF32
            # would weaken the parity contract this warm-up is validating.
            warnings.filterwarnings(
                "ignore",
                message="TensorFloat32 tensor cores for float32 matrix multiplication.*",
            )
            result = self._solver.solve(q, target, self._previous_velocity)
        self._torch.cuda.synchronize(self._solver.device)
        if result.fallback_used or not result.actual_device.startswith("cuda"):
            raise RuntimeError("floating GPU WBC attribution contract failed")
        self._previous_velocity.zero_()
        self._last_target = None
        self._last_target_host = None
        return (time.perf_counter() - started) * 1e3

    def solve_device_batch(
        self,
        q,
        target,
        previous_velocity=None,
        current_velocity=None,
        *,
        reset_mask=None,
        valid_mask=None,
    ):
        """Run all worlds in one CUDA launch path and keep results on device.

        ``q`` must have shape ``[batch_size, configuration_dim]`` and ``target``
        must have shape ``[batch_size, frame_count, 7]`` in position plus WXYZ
        quaternion order. When ``previous_velocity`` is omitted, accepted
        velocity history is retained on device for the next call.

        Provided tensors for a participating world are treated as contemporaneous
        this tick. ``reset_mask`` clears command history for selected worlds.
        ``valid_mask`` excludes worlds from the solve without rebuilding; those
        worlds hold their configuration and keep stored history unless also
        reset. There is no independent per-field stale/fresh schedule: omitted
        history is the retained accepted command, and every supplied field is
        fresh for participating worlds.
        """

        history = self._previous_velocity if previous_velocity is None else previous_velocity
        reset = self._world_mask(reset_mask, "reset_mask")
        valid = self._world_mask(valid_mask, "valid_mask")
        if reset is not None:
            torch = self._torch
            history = torch.where(reset.unsqueeze(-1), torch.zeros_like(history), history)
            if previous_velocity is None:
                self._previous_velocity.masked_fill_(reset.unsqueeze(-1), 0)
        solve_kwargs = {}
        if reset is not None:
            solve_kwargs["reset_mask"] = reset
        if valid is not None:
            solve_kwargs["valid_mask"] = valid
        result = self._solver.solve(q, target, history, current_velocity, **solve_kwargs)
        if previous_velocity is None and result.accepted_velocity is not None:
            accepted = result.accepted_velocity
            if valid is None:
                self._previous_velocity.copy_(accepted)
            else:
                self._previous_velocity.copy_(
                    self._torch.where(valid.unsqueeze(-1), accepted, self._previous_velocity)
                )
        return result

    def measure_device_batch(
        self,
        q,
        target,
        previous_velocity=None,
        current_velocity=None,
        *,
        reset_mask=None,
        valid_mask=None,
    ) -> GpuWbcResourceReport:
        """Time one device batch and snapshot CUDA allocator state.

        The report splits host dispatch from the blocking synchronize used to
        finish the solve. It does not include physics, observations, policy
        inference, or rendering.
        """

        torch = self._torch
        device = self._solver.device
        torch.cuda.synchronize(device)
        started = time.perf_counter()
        result = self.solve_device_batch(
            q,
            target,
            previous_velocity,
            current_velocity,
            reset_mask=reset_mask,
            valid_mask=valid_mask,
        )
        dispatched = time.perf_counter()
        torch.cuda.synchronize(device)
        finished = time.perf_counter()
        return GpuWbcResourceReport(
            result=result,
            host_dispatch_ms=(dispatched - started) * 1e3,
            synchronization_ms=(finished - dispatched) * 1e3,
            kernel_time_ms=float(getattr(result, "kernel_time_ms", 0.0) or 0.0),
            allocated_bytes=int(torch.cuda.memory_allocated(device)),
            reserved_bytes=int(torch.cuda.memory_reserved(device)),
        )

    def solve_device_step(
        self, q, target, *, current_velocity=None
    ) -> _GpuWbcPendingMultiFrameStep:
        """Launch one step without synchronizing or publishing to the host.

        This is the transaction primitive used by composed GPU solvers.  The
        target-change reset remains device-side, including quaternion sign
        equivalence, and persistent velocity/target history advances exactly
        once.  Collision debug intentionally stays on the ordinary synchronous
        path because witness decoding is host-visible and exceptional.
        """

        torch = self._torch
        expected_q = (1, self.configuration_dim)
        expected_target = (1, len(self.frames), 7)
        for name, value, shape in (
            ("q", q, expected_q),
            ("target", target, expected_target),
        ):
            if (
                not isinstance(value, torch.Tensor)
                or tuple(value.shape) != shape
                or value.device != self._solver.device
                or value.dtype is not torch.float32
            ):
                raise ValueError(
                    f"{name} must have shape {shape}, float32, on {self._solver.device}"
                )
        if self._solver.config.collision_debug_enabled:
            self._solver.config = replace(
                self._solver.config,
                collision_debug_enabled=False,
            )
            self._solver._graph = None

        if self._last_target is None:
            previous = torch.zeros_like(self._previous_velocity)
        else:
            translation_changed = torch.any(
                torch.abs(target[..., :3] - self._last_target[..., :3]) > 1e-7
            )
            current_q, previous_q = target[..., 3:], self._last_target[..., 3:]
            rotation_changed = torch.any(
                torch.minimum(
                    torch.amax(torch.abs(current_q - previous_q), dim=-1),
                    torch.amax(torch.abs(current_q + previous_q), dim=-1),
                )
                > 1e-7
            )
            changed = translation_changed | rotation_changed
            previous = torch.where(
                changed,
                torch.zeros_like(self._previous_velocity),
                self._previous_velocity,
            )

        start_event = self._pending_start_event
        end_event = self._pending_end_event
        start_event.record()
        result = self._solver.solve(q, target, previous, current_velocity)
        compact = getattr(result, "compact_publication", None)
        if compact is None:
            compact = self._solver.compact_publication(result)
        moved = torch.any(torch.abs(result.q_solution - q) > 1e-7)
        end_event.record()
        self._previous_velocity.copy_(result.accepted_velocity)
        self._last_target = target.clone()
        self._last_target_host = None
        return _GpuWbcPendingMultiFrameStep(
            adapter=self,
            q=q,
            result=result,
            compact=compact,
            moved=moved,
            start_event=start_event,
            end_event=end_event,
        )

    @staticmethod
    def publish_device_steps(
        pending_steps: tuple[_GpuWbcPendingMultiFrameStep, ...],
    ) -> tuple[GpuWbcMultiFrameResult, ...]:
        """Publish several queued steps through one blocking host transfer."""

        if not pending_steps:
            return ()
        torch = pending_steps[0].adapter._torch
        device = pending_steps[0].adapter._solver.device
        if any(
            step.adapter._torch is not torch or step.adapter._solver.device != device
            for step in pending_steps
        ):
            raise ValueError("pending GPU steps must share one Torch CUDA device")
        pieces = []
        widths = []
        for step in pending_steps:
            packed = step.compact[0]
            pieces.extend((packed, step.moved.to(packed.dtype).reshape(1)))
            widths.append(int(packed.shape[0]))
        host = torch.cat(pieces).detach().cpu().numpy().copy()
        output = []
        offset = 0
        for step, width in zip(pending_steps, widths, strict=True):
            compact_host = host[offset : offset + width]
            moved = bool(host[offset + width])
            offset += width + 1
            elapsed_ms = float(step.start_event.elapsed_time(step.end_event))
            output.append(
                step.adapter._decode_compact_device_step(
                    step.result,
                    compact_host,
                    moved=moved,
                    elapsed_ms=elapsed_ms,
                )
            )
        return tuple(output)

    def _decode_compact_device_step(
        self,
        result,
        host,
        *,
        moved: bool,
        elapsed_ms: float,
    ) -> GpuWbcMultiFrameResult:
        """Decode a compact row after a composed transaction synchronizes."""

        frame_count = len(self.frames)
        scalar_start = self.configuration_dim + 2 * frame_count
        expected = scalar_start + len(self._compact_publication_scalars)
        if host.shape != (expected,):
            raise RuntimeError(
                f"GPU compact publication has shape {host.shape}, expected {(expected,)}"
            )
        solved_host = host[: self.configuration_dim].astype(float, copy=True)
        position = tuple(
            float(value)
            for value in host[self.configuration_dim : self.configuration_dim + frame_count]
        )
        orientation = tuple(
            float(value) for value in host[self.configuration_dim + frame_count : scalar_start]
        )
        compact_values = dict(
            zip(
                self._compact_publication_scalars,
                (float(value) for value in host[scalar_start:]),
                strict=True,
            )
        )

        def optional(name):
            value = compact_values[name]
            return None if value != value else value

        converged = bool(compact_values["converged"])
        collision_distance = optional("minimum_collision_distance_m")
        collision_active = bool(compact_values["collision_active"])
        collision_accepted = bool(compact_values["collision_step_accepted"])
        collision_overflow = bool(compact_values["collision_overflow"])
        collision_constraint_applied = bool(compact_values["collision_constraint_applied"])
        torso_feasible = bool(compact_values["torso_constraint_feasible"])
        com_feasible = bool(compact_values["com_constraint_feasible"])
        capture_feasible = bool(compact_values["capture_point_constraint_feasible"])
        zmp_feasible = bool(compact_values["velocity_zmp_constraint_feasible"])
        if not (com_feasible and capture_feasible and zmp_feasible):
            status = "SAFE_HOLD_COM"
        elif not torso_feasible:
            status = "SAFE_HOLD_TORSO"
        elif collision_overflow:
            status = "SAFE_HOLD_OVERFLOW"
        elif not collision_accepted:
            status = "SAFE_HOLD_COLLISION"
        else:
            status = "SUCCESS" if converged else ("SAFE_STEP" if moved else "SAFE_HOLD")
        return GpuWbcMultiFrameResult(
            joints=solved_host,
            status=status,
            position_errors=position,
            rotation_errors=orientation,
            elapsed_ms=elapsed_ms,
            kernel_time_ms=float(result.kernel_time_ms),
            minimum_collision_distance_m=collision_distance,
            collision_active=collision_active,
            collision_step_accepted=collision_accepted,
            collision_overflow=collision_overflow,
            collision_constraint_applied=collision_constraint_applied,
            collision_clear_state_certified=bool(compact_values["collision_clear_state_certified"]),
            collision_debug=None,
            effective_dt=optional("effective_dt"),
            com_constraint_enabled=bool(getattr(result, "com_constraint_enabled", False)),
            com_constraint_applied=bool(compact_values["com_constraint_applied"]),
            com_constraint_feasible=com_feasible,
            minimum_com_slack_m=optional("minimum_com_slack_m"),
            capture_point_constraint_enabled=bool(
                getattr(result, "capture_point_constraint_enabled", False)
            ),
            capture_point_constraint_applied=bool(
                compact_values["capture_point_constraint_applied"]
            ),
            capture_point_constraint_feasible=capture_feasible,
            minimum_capture_point_slack_m=optional("minimum_capture_point_slack_m"),
            velocity_zmp_constraint_enabled=bool(
                getattr(result, "velocity_zmp_constraint_enabled", False)
            ),
            velocity_zmp_constraint_applied=bool(compact_values["velocity_zmp_constraint_applied"]),
            velocity_zmp_constraint_feasible=zmp_feasible,
            minimum_velocity_zmp_slack_m=optional("minimum_velocity_zmp_slack_m"),
            minimum_zmp_normal_force_n=optional("minimum_zmp_normal_force_n"),
            centroidal_momentum_task_enabled=bool(
                getattr(result, "centroidal_momentum_task_enabled", False)
            ),
            centroidal_momentum_task_applied=bool(
                compact_values["centroidal_momentum_task_applied"]
            ),
            torso_constraint_enabled=bool(getattr(result, "torso_constraint_enabled", False)),
            torso_constraint_applied=bool(compact_values["torso_constraint_applied"]),
            torso_constraint_feasible=torso_feasible,
            posture_task_enabled=bool(getattr(result, "posture_task_enabled", False)),
            posture_task_applied=bool(compact_values["posture_task_applied"]),
            posture_primary_residual_increase=optional("posture_primary_residual_increase"),
            posture_secondary_residual_before=optional("posture_secondary_residual_before"),
            posture_secondary_residual_after=optional("posture_secondary_residual_after"),
            secondary_task_enabled=bool(getattr(result, "secondary_task_enabled", False)),
            secondary_task_applied=bool(compact_values["secondary_task_applied"]),
            secondary_primary_residual_increase=optional("secondary_primary_residual_increase"),
            secondary_residual_before=optional("secondary_residual_before"),
            secondary_residual_after=optional("secondary_residual_after"),
        )

    def solve_step(
        self,
        configuration: object,
        targets: tuple[object, ...],
        *,
        include_collision_debug: bool = False,
        current_velocity: object = None,
    ) -> GpuWbcMultiFrameResult:
        collision_debug_enabled = bool(include_collision_debug and self.collision_supported)
        if collision_debug_enabled != self._solver.config.collision_debug_enabled:
            self._solver.config = replace(
                self._solver.config,
                collision_debug_enabled=collision_debug_enabled,
            )
            self._solver._graph = None
        q = self._q_tensor(configuration)
        target = self._target_tensor(targets)
        measured_velocity = None
        if current_velocity is not None:
            import numpy as np

            measured = np.asarray(current_velocity, dtype=np.float32)
            if measured.shape != (self.velocity_dim,) or not np.isfinite(measured).all():
                raise ValueError(
                    f"current_velocity must be finite with shape {(self.velocity_dim,)}"
                )
            measured_velocity = self._torch.as_tensor(measured, device=self._solver.device).reshape(
                1, self.velocity_dim
            )
        current_target_host = getattr(self, "_current_target_host", None)
        target_changed = self._last_target is None or (
            _target_geometry_changed(self._torch, target, self._last_target)
            if current_target_host is None
            else _host_target_geometry_changed(current_target_host, self._last_target_host)
        )
        if target_changed:
            self._previous_velocity.zero_()
        self._torch.cuda.synchronize(self._solver.device)
        started = time.perf_counter()
        result = self._solver.solve(q, target, self._previous_velocity, measured_velocity)
        compact = getattr(result, "compact_publication", None)
        if compact is None and self._compact_publication_enabled:
            compact = self._solver.compact_publication(result)
        compact_host = None
        if compact is None:
            self._torch.cuda.synchronize(self._solver.device)
        else:
            # The blocking host copy is the sole completion boundary for the
            # solver result. Optional collision witnesses remain lazy.
            compact_host = compact[0].detach().cpu().numpy().copy()
        elapsed_ms = (time.perf_counter() - started) * 1e3
        if result.fallback_used or not result.actual_device.startswith("cuda"):
            raise RuntimeError("floating GPU WBC attribution contract failed")
        solved = result.q_solution[0]
        self._previous_velocity.copy_(result.accepted_velocity)
        self._last_target = target.clone()
        self._last_target_host = None if current_target_host is None else current_target_host.copy()
        compact_values = None
        if compact_host is not None:
            host = compact_host
            frame_count = len(self.frames)
            scalar_start = self.configuration_dim + 2 * frame_count
            expected = scalar_start + len(self._compact_publication_scalars)
            if host.shape != (expected,):
                raise RuntimeError(
                    f"GPU compact publication has shape {host.shape}, expected {(expected,)}"
                )
            solved_host = host[: self.configuration_dim].astype(float, copy=True)
            position = tuple(
                float(value)
                for value in host[self.configuration_dim : self.configuration_dim + frame_count]
            )
            orientation = tuple(
                float(value) for value in host[self.configuration_dim + frame_count : scalar_start]
            )
            compact_values = dict(
                zip(
                    self._compact_publication_scalars,
                    (float(value) for value in host[scalar_start:]),
                    strict=True,
                )
            )

            def optional(name):
                value = compact_values[name]
                return None if value != value else value

            import numpy as np

            moved = bool(
                np.any(
                    np.abs(
                        solved_host
                        - np.asarray(configuration, dtype=float).reshape(self.configuration_dim)
                    )
                    > 1e-7
                )
            )
            converged = bool(compact_values["converged"])
            collision_distance = optional("minimum_collision_distance_m")
            collision_active = bool(compact_values["collision_active"])
            collision_accepted = bool(compact_values["collision_step_accepted"])
            collision_overflow = bool(compact_values["collision_overflow"])
            collision_constraint_applied = bool(compact_values["collision_constraint_applied"])
            collision_clear_state_certified = bool(
                compact_values["collision_clear_state_certified"]
            )
            effective_dt = optional("effective_dt")
        else:
            solved_host = solved.detach().cpu().numpy().astype(float, copy=True)
            position = tuple(float(value) for value in result.position_error_m[0].tolist())
            orientation = tuple(float(value) for value in result.orientation_error_rad[0].tolist())
            moved = bool(self._torch.any(self._torch.abs(solved - q[0]) > 1e-7).item())
            converged = bool(result.converged[0].item())
            collision_distance = (
                None
                if result.minimum_collision_distance_m is None
                else float(result.minimum_collision_distance_m[0].item())
            )
            collision_active = bool(
                result.collision_active is not None and result.collision_active[0].item()
            )
            collision_accepted = bool(
                result.collision_step_accepted is None or result.collision_step_accepted[0].item()
            )
            collision_overflow = bool(
                result.collision_overflow is not None and result.collision_overflow[0].item()
            )
            collision_constraint_applied = bool(
                result.collision_constraint_applied is not None
                and result.collision_constraint_applied[0].item()
            )
            clear_state = getattr(result, "collision_clear_state_certified", None)
            collision_clear_state_certified = bool(
                clear_state is not None and clear_state[0].item()
            )
            effective_dt = (
                None if result.effective_dt is None else float(result.effective_dt[0].item())
            )
        collision_debug = None
        if include_collision_debug and collision_active and not collision_overflow:
            collision_query = self._solver.collision
            if collision_query is None:
                raise RuntimeError("GPU collision diagnostics require collision state")
            shape_pair = result.closest_collision_shape_pair[0].detach().cpu().tolist()
            object_a, object_b = collision_query.decode_shape_pair(
                int(shape_pair[0]), int(shape_pair[1])
            )
            point_a = result.closest_collision_point0_world_m[0].detach().cpu().tolist()
            point_b = result.closest_collision_point1_world_m[0].detach().cpu().tolist()
            collision_debug = GpuWbcCollisionDebug(
                object_a=object_a,
                object_b=object_b,
                distance=float(collision_distance),
                point_a_world=tuple(float(value) for value in point_a),
                point_b_world=tuple(float(value) for value in point_b),
            )
        if compact_values is None:
            torso_feasible = _result_scalar(result, "torso_constraint_feasible", True)
            com_feasible = _result_scalar(result, "com_constraint_feasible", True)
            capture_feasible = _result_scalar(result, "capture_point_constraint_feasible", True)
            zmp_feasible = _result_scalar(result, "velocity_zmp_constraint_feasible", True)
        else:
            torso_feasible = bool(compact_values["torso_constraint_feasible"])
            com_feasible = bool(compact_values["com_constraint_feasible"])
            capture_feasible = bool(compact_values["capture_point_constraint_feasible"])
            zmp_feasible = bool(compact_values["velocity_zmp_constraint_feasible"])
        if not (com_feasible and capture_feasible and zmp_feasible):
            status = "SAFE_HOLD_COM"
        elif not torso_feasible:
            status = "SAFE_HOLD_TORSO"
        elif collision_overflow:
            status = "SAFE_HOLD_OVERFLOW"
        elif not collision_accepted:
            status = "SAFE_HOLD_COLLISION"
        else:
            status = "SUCCESS" if converged else ("SAFE_STEP" if moved else "SAFE_HOLD")
        return GpuWbcMultiFrameResult(
            joints=solved_host,
            status=status,
            position_errors=position,
            rotation_errors=orientation,
            elapsed_ms=elapsed_ms,
            kernel_time_ms=float(result.kernel_time_ms),
            minimum_collision_distance_m=collision_distance,
            collision_active=collision_active,
            collision_step_accepted=collision_accepted,
            collision_overflow=collision_overflow,
            collision_constraint_applied=collision_constraint_applied,
            collision_clear_state_certified=collision_clear_state_certified,
            collision_debug=collision_debug,
            effective_dt=effective_dt,
            com_constraint_enabled=bool(getattr(result, "com_constraint_enabled", False)),
            com_constraint_applied=(
                _result_scalar(result, "com_constraint_applied", False)
                if compact_values is None
                else bool(compact_values["com_constraint_applied"])
            ),
            com_constraint_feasible=com_feasible,
            minimum_com_slack_m=(
                _result_scalar(result, "minimum_com_slack_m")
                if compact_values is None
                else optional("minimum_com_slack_m")
            ),
            capture_point_constraint_enabled=bool(
                getattr(result, "capture_point_constraint_enabled", False)
            ),
            capture_point_constraint_applied=(
                _result_scalar(result, "capture_point_constraint_applied", False)
                if compact_values is None
                else bool(compact_values["capture_point_constraint_applied"])
            ),
            capture_point_constraint_feasible=bool(capture_feasible),
            minimum_capture_point_slack_m=(
                _result_scalar(result, "minimum_capture_point_slack_m")
                if compact_values is None
                else optional("minimum_capture_point_slack_m")
            ),
            velocity_zmp_constraint_enabled=bool(
                getattr(result, "velocity_zmp_constraint_enabled", False)
            ),
            velocity_zmp_constraint_applied=(
                _result_scalar(result, "velocity_zmp_constraint_applied", False)
                if compact_values is None
                else bool(compact_values["velocity_zmp_constraint_applied"])
            ),
            velocity_zmp_constraint_feasible=bool(zmp_feasible),
            minimum_velocity_zmp_slack_m=(
                _result_scalar(result, "minimum_velocity_zmp_slack_m")
                if compact_values is None
                else optional("minimum_velocity_zmp_slack_m")
            ),
            minimum_zmp_normal_force_n=(
                _result_scalar(result, "minimum_zmp_normal_force_n")
                if compact_values is None
                else optional("minimum_zmp_normal_force_n")
            ),
            centroidal_momentum_task_enabled=bool(
                getattr(result, "centroidal_momentum_task_enabled", False)
            ),
            centroidal_momentum_task_applied=(
                _result_scalar(result, "centroidal_momentum_task_applied", False)
                if compact_values is None
                else bool(compact_values["centroidal_momentum_task_applied"])
            ),
            torso_constraint_enabled=bool(getattr(result, "torso_constraint_enabled", False)),
            torso_constraint_applied=(
                _result_scalar(result, "torso_constraint_applied", False)
                if compact_values is None
                else bool(compact_values["torso_constraint_applied"])
            ),
            torso_constraint_feasible=torso_feasible,
            posture_task_enabled=bool(getattr(result, "posture_task_enabled", False)),
            posture_task_applied=(
                _result_scalar(result, "posture_task_applied", False)
                if compact_values is None
                else bool(compact_values["posture_task_applied"])
            ),
            posture_primary_residual_increase=(
                _result_scalar(result, "posture_primary_residual_increase")
                if compact_values is None
                else optional("posture_primary_residual_increase")
            ),
            posture_secondary_residual_before=(
                _result_scalar(result, "posture_secondary_residual_before")
                if compact_values is None
                else optional("posture_secondary_residual_before")
            ),
            posture_secondary_residual_after=(
                _result_scalar(result, "posture_secondary_residual_after")
                if compact_values is None
                else optional("posture_secondary_residual_after")
            ),
            secondary_task_enabled=bool(getattr(result, "secondary_task_enabled", False)),
            secondary_task_applied=(
                _result_scalar(result, "secondary_task_applied", False)
                if compact_values is None
                else bool(compact_values["secondary_task_applied"])
            ),
            secondary_primary_residual_increase=(
                _result_scalar(result, "secondary_primary_residual_increase")
                if compact_values is None
                else optional("secondary_primary_residual_increase")
            ),
            secondary_residual_before=(
                _result_scalar(result, "secondary_residual_before")
                if compact_values is None
                else optional("secondary_residual_before")
            ),
            secondary_residual_after=(
                _result_scalar(result, "secondary_residual_after")
                if compact_values is None
                else optional("secondary_residual_after")
            ),
        )


def _load_fi_artifact(manifest_path: Path):
    import casadi as ca

    resolved = manifest_path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"GPU WBC build manifest not found: {resolved}")
    manifest = json.loads(resolved.read_text(encoding="utf-8"))
    try:
        records = [
            item
            for item in manifest["functions"]
            if item.get("solver") == "fi_pesns" or item.get("name") in FI_FUNCTION_NAMES
        ]
        if len(records) != 1:
            raise StopIteration
        record = records[0]
    except (KeyError, StopIteration) as exc:
        raise ValueError("manifest must contain exactly one FI-PeSNS pose artifact") from exc
    function_path = Path(record["function_path"]).expanduser().resolve()
    library_path = Path(record["library_path"]).expanduser().resolve()
    if not function_path.is_file() or not library_path.is_file():
        raise FileNotFoundError("manifest references a missing CasADi function or CUDA library")
    return ca.Function.load(str(function_path)), library_path, record
