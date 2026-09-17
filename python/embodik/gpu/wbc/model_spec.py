"""Build backend-neutral robot solve metadata from a loaded EmbodiK model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .contracts import ContractViolation, RobotSolveSpec


@dataclass(frozen=True)
class PoseModelParameters:
    """Runtime values needed by one model-derived pose solve."""

    robot_spec: RobotSolveSpec
    default_configuration: tuple[float, ...]
    nominal_configuration: tuple[float, ...]
    joint_lower: tuple[float, ...]
    joint_upper: tuple[float, ...]
    joint_velocity_limits: tuple[float, ...]


def floating_pose_model_parameters_from_embodik(
    robot: Any,
    *,
    name: str,
    model_hash: str,
    task_frames: tuple[str, ...],
    active_velocity_indices: tuple[int, ...],
    default_configuration: Any,
    base_velocity_limits: tuple[float, float, float, float, float, float],
) -> PoseModelParameters:
    """Derive a free-root plus scalar-joint pose contract from a loaded model."""

    import numpy as np

    if not bool(robot.is_floating_base):
        raise ContractViolation(
            "floating pose parameters require a floating-base robot"
        )
    if int(robot.nq) != int(robot.nv) + 1:
        raise ContractViolation("floating pose parameters require nq=nv+1")
    if len(base_velocity_limits) != 6 or not all(
        np.isfinite(value) and value > 0.0 for value in base_velocity_limits
    ):
        raise ContractViolation("base_velocity_limits must contain six positive values")
    active = tuple(int(index) for index in active_velocity_indices)
    if not active or tuple(sorted(set(active))) != active:
        raise ContractViolation("active_velocity_indices must be sorted and unique")
    default = np.asarray(default_configuration, dtype=float)
    if default.shape != (int(robot.nq),) or not np.isfinite(default).all():
        raise ContractViolation(
            f"default configuration must contain {int(robot.nq)} finite values"
        )

    names = tuple(str(value) for value in robot.get_joint_names())
    records = [
        (
            name_value,
            int(robot.get_joint_config_index(name_value)),
            int(robot.get_joint_config_size(name_value)),
            int(robot.get_joint_velocity_index(name_value)),
            int(robot.get_joint_velocity_size(name_value)),
        )
        for name_value in names
    ]
    roots = [record for record in records if record[2:] == (7, 0, 6)]
    if len(roots) != 1:
        roots = [record for record in records if record[2] == 7 and record[4] == 6]
    if len(roots) != 1:
        raise ContractViolation("model must expose exactly one 7/6 free root joint")
    _, _, _, root_velocity_start, _ = roots[0]
    root_velocities = tuple(range(root_velocity_start, root_velocity_start + 6))
    if not set(root_velocities).issubset(active):
        raise ContractViolation("all six free-root velocities must be active")

    lower_all, upper_all = (
        np.asarray(value, dtype=float) for value in robot.get_joint_limits()
    )
    velocity_all = np.asarray(robot.get_velocity_limits(), dtype=float)
    lower: list[float] = []
    upper: list[float] = []
    velocity: list[float] = []
    nominal: list[float] = []
    for velocity_index in active:
        if velocity_index in root_velocities:
            component = velocity_index - root_velocity_start
            lower.append(-1e30)
            upper.append(1e30)
            velocity.append(float(base_velocity_limits[component]))
            nominal.append(0.0)
            continue
        matches = [
            record
            for record in records
            if record[4] == 1 and record[3] == velocity_index and record[2] == 1
        ]
        if len(matches) != 1:
            raise ContractViolation(
                f"active velocity {velocity_index} is not backed by one scalar joint"
            )
        _, q_index, _, _, _ = matches[0]
        values = (lower_all[q_index], upper_all[q_index], velocity_all[velocity_index])
        if not all(np.isfinite(value) for value in values):
            raise ContractViolation(
                f"active scalar velocity {velocity_index} has non-finite limits"
            )
        if values[0] >= values[1] or values[2] <= 0.0:
            raise ContractViolation(
                f"active scalar velocity {velocity_index} has invalid limits"
            )
        lower.append(float(values[0]))
        upper.append(float(values[1]))
        velocity.append(float(values[2]))
        nominal.append(float(default[q_index]))

    spec = robot_solve_spec_from_embodik(
        robot,
        name=name,
        model_hash=model_hash,
        task_frames=task_frames,
        active_velocity_indices=active,
    )
    return PoseModelParameters(
        robot_spec=spec,
        default_configuration=tuple(float(value) for value in default),
        nominal_configuration=tuple(nominal),
        joint_lower=tuple(lower),
        joint_upper=tuple(upper),
        joint_velocity_limits=tuple(velocity),
    )


def robot_solve_spec_from_embodik(
    robot: Any,
    *,
    name: str,
    model_hash: str,
    task_frames: tuple[str, ...] = (),
    active_velocity_indices: tuple[int, ...] | None = None,
    active_joint_names: tuple[str, ...] = (),
    active_configuration_indices: tuple[int, ...] = (),
) -> RobotSolveSpec:
    """Derive dimensions and names without a robot-name or DoF table."""

    available_frames = tuple(str(value) for value in robot.get_frame_names())
    missing_frames = sorted(set(task_frames) - set(available_frames))
    if missing_frames:
        raise ContractViolation(
            f"task frames are absent from loaded model: {', '.join(missing_frames)}"
        )
    nq = int(robot.nq)
    nv = int(robot.nv)
    joint_names = tuple(str(value) for value in robot.get_joint_names())
    has_joint_spans = all(
        hasattr(robot, name)
        for name in (
            "get_joint_config_index",
            "get_joint_config_size",
            "get_joint_velocity_index",
            "get_joint_velocity_size",
        )
    )
    return RobotSolveSpec(
        name=name,
        model_hash=model_hash,
        configuration_dim=nq,
        velocity_dim=nv,
        floating_base=bool(robot.is_floating_base),
        joint_names=joint_names,
        active_velocity_indices=(
            tuple(range(nv))
            if active_velocity_indices is None
            else tuple(active_velocity_indices)
        ),
        task_frames=tuple(task_frames),
        active_joint_names=tuple(active_joint_names),
        active_configuration_indices=tuple(active_configuration_indices),
        joint_configuration_indices=(
            tuple(int(robot.get_joint_config_index(name)) for name in joint_names)
            if has_joint_spans
            else ()
        ),
        joint_configuration_sizes=(
            tuple(int(robot.get_joint_config_size(name)) for name in joint_names)
            if has_joint_spans
            else ()
        ),
        joint_velocity_indices=(
            tuple(int(robot.get_joint_velocity_index(name)) for name in joint_names)
            if has_joint_spans
            else ()
        ),
        joint_velocity_sizes=(
            tuple(int(robot.get_joint_velocity_size(name)) for name in joint_names)
            if has_joint_spans
            else ()
        ),
        collision_geometry_names=tuple(
            str(value) for value in robot.get_collision_geometry_names()
        ),
    )


def pose_model_parameters_from_embodik(
    robot: Any,
    *,
    name: str,
    model_hash: str,
    frame: str | None = None,
    task_frames: tuple[str, ...] = (),
    active_joint_names: tuple[str, ...],
    default_configuration: Any,
) -> PoseModelParameters:
    """Derive limits and scalar-joint mappings without a robot-family table."""

    import numpy as np

    if bool(robot.is_floating_base):
        raise ContractViolation(
            "model-derived pose solver currently requires a fixed-base robot"
        )
    if frame is not None and task_frames:
        raise ContractViolation("provide frame or task_frames, not both")
    frames = (frame,) if frame is not None else tuple(task_frames)
    if not frames:
        raise ContractViolation("at least one task frame is required")
    if not active_joint_names:
        raise ContractViolation("at least one active joint name is required")
    if len(set(active_joint_names)) != len(active_joint_names):
        raise ContractViolation("active joint names must be unique")
    all_joint_names = tuple(str(value) for value in robot.get_joint_names())
    missing = sorted(set(active_joint_names) - set(all_joint_names))
    if missing:
        raise ContractViolation(
            f"active joints are absent from loaded model: {', '.join(missing)}"
        )
    q_indices: list[int] = []
    v_indices: list[int] = []
    for joint_name in active_joint_names:
        q_size = int(robot.get_joint_config_size(joint_name))
        v_size = int(robot.get_joint_velocity_size(joint_name))
        if q_size != 1 or v_size != 1:
            raise ContractViolation(
                f"active joint {joint_name!r} has nq/nv={q_size}/{v_size}; "
                "only scalar joints are currently supported"
            )
        q_indices.append(int(robot.get_joint_config_index(joint_name)))
        v_indices.append(int(robot.get_joint_velocity_index(joint_name)))
    if tuple(sorted(v_indices)) != tuple(v_indices):
        raise ContractViolation(
            "active joints must be ordered by their model velocity indices"
        )

    default = np.asarray(default_configuration, dtype=float)
    if default.shape != (int(robot.nq),) or not np.isfinite(default).all():
        raise ContractViolation(
            f"default configuration must contain {int(robot.nq)} finite values"
        )
    lower, upper = robot.get_joint_limits()
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    velocity = np.asarray(robot.get_velocity_limits(), dtype=float)
    active_lower = lower[q_indices]
    active_upper = upper[q_indices]
    active_velocity = velocity[v_indices]
    if not (
        np.isfinite(active_lower).all()
        and np.isfinite(active_upper).all()
        and np.isfinite(active_velocity).all()
    ):
        raise ContractViolation(
            "active model joints must have finite position and velocity limits"
        )
    if np.any(active_lower >= active_upper) or np.any(active_velocity <= 0.0):
        raise ContractViolation("active model joint limits are invalid")

    spec = robot_solve_spec_from_embodik(
        robot,
        name=name,
        model_hash=model_hash,
        task_frames=frames,
        active_velocity_indices=tuple(v_indices),
        active_joint_names=active_joint_names,
        active_configuration_indices=tuple(q_indices),
    )
    return PoseModelParameters(
        robot_spec=spec,
        default_configuration=tuple(float(value) for value in default),
        nominal_configuration=tuple(float(default[index]) for index in q_indices),
        joint_lower=tuple(float(value) for value in active_lower),
        joint_upper=tuple(float(value) for value in active_upper),
        joint_velocity_limits=tuple(float(value) for value in active_velocity),
    )
