"""Backend-neutral request, result, safety, and attribution contracts.

Runtime tensor payloads are intentionally typed as ``Any``. An adapter may pass
NumPy, Torch, Warp, or another array object without this module importing or
copying it. ``to_dict`` is an artifact/debug boundary and may materialize such
payloads through ``tolist``; it must not be used in the timed solver path.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, fields, is_dataclass
from enum import Enum
from typing import Any


class ContractViolation(ValueError):
    """Raised when a request or result violates the experiment contract."""


class TaskKind(str, Enum):
    FRAME_POSE = "frame_pose"
    FRAME_POSITION = "frame_position"
    FRAME_ORIENTATION = "frame_orientation"
    POSTURE = "posture"
    MANIPULABILITY = "manipulability"


class TargetRepresentation(str, Enum):
    POSITION_QUATERNION_WXYZ = "position_quaternion_wxyz"
    POSITION_XYZ = "position_xyz"
    QUATERNION_WXYZ = "quaternion_wxyz"
    JOINT_POSITION = "joint_position"
    SCALAR = "scalar"


class ConstraintKind(str, Enum):
    JOINT_POSITION = "joint_position"
    JOINT_VELOCITY = "joint_velocity"
    JOINT_ACCELERATION = "joint_acceleration"
    TRUST_REGION = "trust_region"
    SELF_COLLISION = "self_collision"
    WORLD_COLLISION = "world_collision"


class StatusCode(str, Enum):
    SOLVED = "solved"
    INFEASIBLE = "infeasible"
    COLLISION_BLOCKED = "collision_blocked"
    COLLISION_VIOLATION = "collision_violation"
    LIMIT_VIOLATION = "limit_violation"
    CAPACITY_OVERFLOW = "capacity_overflow"
    UNSUPPORTED = "unsupported"
    NONFINITE = "nonfinite"
    VERIFICATION_FAILED = "verification_failed"
    EXECUTION_ERROR = "execution_error"


class Decision(str, Enum):
    APPROVED = "approved"
    NEEDS_VERIFICATION = "needs_verification"
    REJECTED = "rejected"


class ExecutionTarget(str, Enum):
    AUTO = "auto"
    CPU = "cpu"
    GPU = "gpu"


class DeviceKind(str, Enum):
    CPU = "cpu"
    CUDA = "cuda"
    UNKNOWN = "unknown"


class ConfigurationUpdateKind(str, Enum):
    """How a backend maps a tangent-space velocity back to configuration."""

    MODEL_MANIFOLD = "model_manifold"


def _enum(enum_type: type[Enum], value: Any) -> Any:
    return value if isinstance(value, enum_type) else enum_type(value)


def _deep_tuple(value: Any) -> Any:
    if isinstance(value, list):
        return tuple(_deep_tuple(item) for item in value)
    if isinstance(value, dict):
        return {key: _deep_tuple(item) for key, item in value.items()}
    return value


def _jsonable(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, float) and not math.isfinite(value):
        if math.isnan(value):
            label = "nan"
        elif value > 0:
            label = "+inf"
        else:
            label = "-inf"
        return {"__nonfinite_float__": label}
    if is_dataclass(value):
        return {field.name: _jsonable(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "tolist"):
        return _jsonable(value.tolist())
    return value


def _restore_jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        if set(value) == {"__nonfinite_float__"}:
            label = value["__nonfinite_float__"]
            if label == "nan":
                return math.nan
            if label == "+inf":
                return math.inf
            if label == "-inf":
                return -math.inf
            raise ContractViolation(f"unknown non-finite float label: {label}")
        return {key: _restore_jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_restore_jsonable(item) for item in value]
    return value


def _shape(value: Any) -> tuple[int, ...]:
    shape = getattr(value, "shape", None)
    if shape is not None:
        return tuple(int(size) for size in shape)
    if isinstance(value, (tuple, list)):
        if not value:
            return (0,)
        child = _shape(value[0])
        if any(_shape(item) != child for item in value[1:]):
            raise ContractViolation("ragged tensor payload")
        return (len(value), *child)
    return ()


def _pairs(
    value: Mapping[str, str] | Iterable[tuple[str, str]],
) -> tuple[tuple[str, str], ...]:
    items = value.items() if isinstance(value, Mapping) else value
    return tuple(sorted((str(key), str(item)) for key, item in items))


def _metadata_pairs(
    value: Mapping[str, Any] | Iterable[tuple[str, Any]],
) -> tuple[tuple[str, Any], ...]:
    items = value.items() if isinstance(value, Mapping) else value
    return tuple(sorted(((str(key), item) for key, item in items), key=lambda pair: pair[0]))


def _require_nonnegative(name: str, value: float) -> None:
    if value < 0:
        raise ContractViolation(f"{name} must be nonnegative")


@dataclass(frozen=True)
class RobotSolveSpec:
    """Model-derived dimensions and capabilities for solver specialization.

    This deliberately contains no known-robot enum or dimension table. An
    adapter builds the spec from the loaded model, and backends may compile and
    cache a fixed-shape implementation from these values without making that
    shape part of the public API.
    """

    name: str
    model_hash: str
    configuration_dim: int
    velocity_dim: int
    floating_base: bool
    joint_names: tuple[str, ...]
    active_velocity_indices: tuple[int, ...]
    task_frames: tuple[str, ...]
    active_joint_names: tuple[str, ...] = ()
    active_configuration_indices: tuple[int, ...] = ()
    joint_configuration_indices: tuple[int, ...] = ()
    joint_configuration_sizes: tuple[int, ...] = ()
    joint_velocity_indices: tuple[int, ...] = ()
    joint_velocity_sizes: tuple[int, ...] = ()
    collision_geometry_names: tuple[str, ...] = ()
    configuration_update: ConfigurationUpdateKind = ConfigurationUpdateKind.MODEL_MANIFOLD

    def __post_init__(self) -> None:
        object.__setattr__(self, "joint_names", tuple(self.joint_names))
        object.__setattr__(
            self,
            "active_velocity_indices",
            tuple(int(index) for index in self.active_velocity_indices),
        )
        object.__setattr__(self, "task_frames", tuple(self.task_frames))
        object.__setattr__(self, "active_joint_names", tuple(self.active_joint_names))
        object.__setattr__(
            self,
            "active_configuration_indices",
            tuple(int(index) for index in self.active_configuration_indices),
        )
        for field_name in (
            "joint_configuration_indices",
            "joint_configuration_sizes",
            "joint_velocity_indices",
            "joint_velocity_sizes",
        ):
            object.__setattr__(
                self,
                field_name,
                tuple(int(value) for value in getattr(self, field_name)),
            )
        object.__setattr__(
            self,
            "collision_geometry_names",
            tuple(self.collision_geometry_names),
        )
        object.__setattr__(
            self,
            "configuration_update",
            _enum(ConfigurationUpdateKind, self.configuration_update),
        )
        if not self.name:
            raise ContractViolation("robot name must not be empty")
        if not self.model_hash:
            raise ContractViolation("robot model_hash must not be empty")
        if self.configuration_dim <= 0 or self.velocity_dim <= 0:
            raise ContractViolation("robot configuration_dim and velocity_dim must be positive")
        if len(set(self.joint_names)) != len(self.joint_names):
            raise ContractViolation("robot joint_names must be unique")
        if len(set(self.active_joint_names)) != len(self.active_joint_names):
            raise ContractViolation("robot active_joint_names must be unique")
        if len(set(self.task_frames)) != len(self.task_frames):
            raise ContractViolation("robot task_frames must be unique")
        if len(set(self.collision_geometry_names)) != len(self.collision_geometry_names):
            raise ContractViolation("robot collision_geometry_names must be unique")
        if not self.active_velocity_indices:
            raise ContractViolation("robot must expose at least one active velocity")
        if tuple(sorted(self.active_velocity_indices)) != self.active_velocity_indices:
            raise ContractViolation("robot active_velocity_indices must be sorted and unique")
        if (
            self.active_velocity_indices[0] < 0
            or self.active_velocity_indices[-1] >= self.velocity_dim
        ):
            raise ContractViolation("robot active_velocity_indices must lie inside velocity_dim")
        active_count = len(self.active_velocity_indices)
        if self.active_joint_names and len(self.active_joint_names) != active_count:
            raise ContractViolation("robot active_joint_names must match active_velocity_indices")
        if self.active_joint_names and not set(self.active_joint_names).issubset(self.joint_names):
            raise ContractViolation("robot active_joint_names must exist in joint_names")
        if self.active_configuration_indices:
            if len(self.active_configuration_indices) != active_count:
                raise ContractViolation(
                    "robot active_configuration_indices must match active velocities"
                )
            if len(set(self.active_configuration_indices)) != active_count:
                raise ContractViolation("robot active_configuration_indices must be unique")
            if (
                min(self.active_configuration_indices) < 0
                or max(self.active_configuration_indices) >= self.configuration_dim
            ):
                raise ContractViolation(
                    "robot active_configuration_indices must lie inside configuration_dim"
                )
        joint_spans = (
            self.joint_configuration_indices,
            self.joint_configuration_sizes,
            self.joint_velocity_indices,
            self.joint_velocity_sizes,
        )
        if any(joint_spans) and not all(
            len(values) == len(self.joint_names) for values in joint_spans
        ):
            raise ContractViolation("joint coordinate/velocity spans must align with joint_names")
        if all(joint_spans):
            if any(size < 0 for size in self.joint_configuration_sizes) or any(
                size < 0 for size in self.joint_velocity_sizes
            ):
                raise ContractViolation("joint coordinate/velocity sizes must be nonnegative")
            if any(
                start < 0 or start + size > self.configuration_dim
                for start, size in zip(
                    self.joint_configuration_indices,
                    self.joint_configuration_sizes,
                    strict=True,
                )
            ):
                raise ContractViolation("joint configuration spans exceed configuration_dim")
            if any(
                start < 0 or start + size > self.velocity_dim
                for start, size in zip(
                    self.joint_velocity_indices,
                    self.joint_velocity_sizes,
                    strict=True,
                )
            ):
                raise ContractViolation("joint velocity spans exceed velocity_dim")

    def specialization_key(
        self,
        *,
        task_layout_hash: str,
        batch_size: int,
        dtype: str,
        device_arch: str,
    ) -> tuple[str, int, int, int, str, str, str]:
        """Return a content-derived cache key for one compiled graph shape."""

        if not task_layout_hash or not dtype or not device_arch:
            raise ContractViolation("task_layout_hash, dtype, and device_arch must not be empty")
        if batch_size <= 0:
            raise ContractViolation("batch_size must be positive")
        return (
            self.model_hash,
            self.configuration_dim,
            self.velocity_dim,
            batch_size,
            task_layout_hash,
            dtype,
            device_arch,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "model_hash": self.model_hash,
            "configuration_dim": self.configuration_dim,
            "velocity_dim": self.velocity_dim,
            "floating_base": self.floating_base,
            "joint_names": list(self.joint_names),
            "active_velocity_indices": list(self.active_velocity_indices),
            "task_frames": list(self.task_frames),
            "active_joint_names": list(self.active_joint_names),
            "active_configuration_indices": list(self.active_configuration_indices),
            "joint_configuration_indices": list(self.joint_configuration_indices),
            "joint_configuration_sizes": list(self.joint_configuration_sizes),
            "joint_velocity_indices": list(self.joint_velocity_indices),
            "joint_velocity_sizes": list(self.joint_velocity_sizes),
            "collision_geometry_names": list(self.collision_geometry_names),
            "configuration_update": self.configuration_update.value,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> RobotSolveSpec:
        return cls(**value)


@dataclass(frozen=True)
class VelocityTaskLayout:
    """Fixed-shape velocity-task metadata without robot-family assumptions."""

    name: str
    task_dimensions: tuple[int, ...]
    constraint_rows: int
    task_names: tuple[str, ...] = ()
    task_frames: tuple[str | None, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "task_dimensions", tuple(int(value) for value in self.task_dimensions)
        )
        object.__setattr__(self, "task_names", tuple(self.task_names))
        object.__setattr__(self, "task_frames", tuple(self.task_frames))
        if not self.name:
            raise ContractViolation("velocity task layout name must not be empty")
        if not self.task_dimensions or any(dimension <= 0 for dimension in self.task_dimensions):
            raise ContractViolation("velocity task dimensions must be positive")
        if self.constraint_rows <= 0:
            raise ContractViolation("velocity constraint_rows must be positive")
        if self.task_names and len(self.task_names) != len(self.task_dimensions):
            raise ContractViolation("task_names must match task_dimensions")
        if self.task_frames and len(self.task_frames) != len(self.task_dimensions):
            raise ContractViolation("task_frames must match task_dimensions")

    @property
    def layout_hash(self) -> str:
        encoded = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(encoded.encode()).hexdigest()

    def shape_key(self, robot: RobotSolveSpec) -> tuple[int, tuple[int, ...], int]:
        return (
            len(robot.active_velocity_indices),
            self.task_dimensions,
            self.constraint_rows,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "task_dimensions": list(self.task_dimensions),
            "constraint_rows": self.constraint_rows,
            "task_names": list(self.task_names),
            "task_frames": list(self.task_frames),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> VelocityTaskLayout:
        return cls(**value)


@dataclass(frozen=True)
class TaskSpec:
    name: str
    kind: TaskKind
    priority: int
    target: Any
    representation: TargetRepresentation
    weight: float = 1.0
    frame: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "kind", _enum(TaskKind, self.kind))
        object.__setattr__(
            self,
            "representation",
            _enum(TargetRepresentation, self.representation),
        )
        _require_nonnegative("task priority", self.priority)
        if not self.name:
            raise ContractViolation("task name must not be empty")
        if not math.isfinite(self.weight) or self.weight <= 0:
            raise ContractViolation("task weight must be finite and positive")
        expected = {
            TaskKind.FRAME_POSE: TargetRepresentation.POSITION_QUATERNION_WXYZ,
            TaskKind.FRAME_POSITION: TargetRepresentation.POSITION_XYZ,
            TaskKind.FRAME_ORIENTATION: TargetRepresentation.QUATERNION_WXYZ,
            TaskKind.POSTURE: TargetRepresentation.JOINT_POSITION,
            TaskKind.MANIPULABILITY: TargetRepresentation.SCALAR,
        }[self.kind]
        if self.representation is not expected:
            raise ContractViolation(f"{self.kind.value} task requires {expected.value} targets")

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> TaskSpec:
        return cls(
            name=value["name"],
            kind=TaskKind(value["kind"]),
            priority=int(value["priority"]),
            target=_deep_tuple(value["target"]),
            representation=TargetRepresentation(value["representation"]),
            weight=float(value.get("weight", 1.0)),
            frame=value.get("frame"),
        )


@dataclass(frozen=True)
class ConstraintSpec:
    name: str
    kind: ConstraintKind
    safety_critical: bool = True
    capacity: int | None = None
    parameters: tuple[tuple[str, Any], ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "kind", _enum(ConstraintKind, self.kind))
        object.__setattr__(self, "parameters", tuple(self.parameters))
        if not self.name:
            raise ContractViolation("constraint name must not be empty")
        if self.capacity is not None and self.capacity <= 0:
            raise ContractViolation("constraint capacity must be positive")

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> ConstraintSpec:
        return cls(
            name=value["name"],
            kind=ConstraintKind(value["kind"]),
            safety_critical=bool(value.get("safety_critical", True)),
            capacity=value.get("capacity"),
            parameters=tuple(
                (str(key), _deep_tuple(item)) for key, item in value.get("parameters", ())
            ),
        )


@dataclass(frozen=True)
class WbcBatchRequest:
    request_id: str
    batch_size: int
    horizon: int
    q: Any
    tasks: tuple[TaskSpec, ...]
    constraints: tuple[ConstraintSpec, ...]
    execution_target: ExecutionTarget
    require_target: bool
    robot_revision: str
    world_revision: str
    config_hash: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "tasks", tuple(self.tasks))
        object.__setattr__(self, "constraints", tuple(self.constraints))
        object.__setattr__(self, "execution_target", _enum(ExecutionTarget, self.execution_target))
        if not self.request_id:
            raise ContractViolation("request_id must not be empty")
        if self.batch_size <= 0 or self.horizon <= 0:
            raise ContractViolation("batch_size and horizon must be positive")
        q_shape = _shape(self.q)
        if len(q_shape) < 2 or q_shape[0] != self.batch_size:
            raise ContractViolation(
                f"q must have shape [batch, dof...] with batch={self.batch_size}; got {q_shape}"
            )
        if not self.tasks:
            raise ContractViolation("at least one task is required")
        if self.require_target and self.execution_target is ExecutionTarget.AUTO:
            raise ContractViolation("AUTO execution_target cannot be required")
        for name, value in (
            ("robot_revision", self.robot_revision),
            ("world_revision", self.world_revision),
            ("config_hash", self.config_hash),
        ):
            if not value:
                raise ContractViolation(f"{name} must not be empty")

    def to_dict(self) -> dict[str, Any]:
        return _jsonable(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), allow_nan=False, sort_keys=True, separators=(",", ":"))

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> WbcBatchRequest:
        value = _restore_jsonable(value)
        return cls(
            request_id=value["request_id"],
            batch_size=int(value["batch_size"]),
            horizon=int(value["horizon"]),
            q=_deep_tuple(value["q"]),
            tasks=tuple(TaskSpec.from_dict(task) for task in value["tasks"]),
            constraints=tuple(
                ConstraintSpec.from_dict(constraint) for constraint in value["constraints"]
            ),
            execution_target=ExecutionTarget(value["execution_target"]),
            require_target=bool(value["require_target"]),
            robot_revision=value["robot_revision"],
            world_revision=value["world_revision"],
            config_hash=value["config_hash"],
        )


@dataclass(frozen=True)
class ExecutionAttribution:
    requested_target: ExecutionTarget
    target_required: bool
    actual_device: DeviceKind
    backend: str
    device_name: str
    precision: str
    device_execution_proven: bool
    cpu_fallback_used: bool = False
    placeholder_output_used: bool = False
    host_to_device_bytes: int = 0
    device_to_host_bytes: int = 0
    synchronization_count: int = 0
    kernel_launch_count: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "requested_target", _enum(ExecutionTarget, self.requested_target))
        object.__setattr__(self, "actual_device", _enum(DeviceKind, self.actual_device))
        if self.target_required and self.requested_target is ExecutionTarget.AUTO:
            raise ContractViolation("AUTO requested_target cannot be required")
        for name in (
            "host_to_device_bytes",
            "device_to_host_bytes",
            "synchronization_count",
            "kernel_launch_count",
        ):
            _require_nonnegative(name, getattr(self, name))

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> ExecutionAttribution:
        return cls(
            requested_target=ExecutionTarget(value["requested_target"]),
            target_required=bool(value["target_required"]),
            actual_device=DeviceKind(value["actual_device"]),
            backend=value["backend"],
            device_name=value["device_name"],
            precision=value["precision"],
            device_execution_proven=bool(value["device_execution_proven"]),
            cpu_fallback_used=bool(value.get("cpu_fallback_used", False)),
            placeholder_output_used=bool(value.get("placeholder_output_used", False)),
            host_to_device_bytes=int(value.get("host_to_device_bytes", 0)),
            device_to_host_bytes=int(value.get("device_to_host_bytes", 0)),
            synchronization_count=int(value.get("synchronization_count", 0)),
            kernel_launch_count=int(value.get("kernel_launch_count", 0)),
        )


@dataclass(frozen=True)
class TimingBreakdown:
    total_ms: float
    solve_ms: float
    transfer_ms: float = 0.0
    synchronization_ms: float = 0.0
    verification_ms: float = 0.0
    compile_ms: float = 0.0

    def __post_init__(self) -> None:
        for field_name in (
            "total_ms",
            "solve_ms",
            "transfer_ms",
            "synchronization_ms",
            "verification_ms",
            "compile_ms",
        ):
            value = getattr(self, field_name)
            if not math.isfinite(value) or value < 0:
                raise ContractViolation(f"{field_name} must be finite and nonnegative")

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> TimingBreakdown:
        return cls(**{field.name: float(value.get(field.name, 0.0)) for field in fields(cls)})


@dataclass(frozen=True)
class ResultProvenance:
    code_revision: str
    robot_revision: str
    world_revision: str
    config_hash: str
    generated_artifact_hash: str | None = None
    dependencies: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "dependencies", _pairs(self.dependencies))
        for name in (
            "code_revision",
            "robot_revision",
            "world_revision",
            "config_hash",
        ):
            if not getattr(self, name):
                raise ContractViolation(f"{name} must not be empty")

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> ResultProvenance:
        return cls(
            code_revision=value["code_revision"],
            robot_revision=value["robot_revision"],
            world_revision=value["world_revision"],
            config_hash=value["config_hash"],
            generated_artifact_hash=value.get("generated_artifact_hash"),
            dependencies=tuple(tuple(pair) for pair in value.get("dependencies", ())),
        )


@dataclass(frozen=True)
class ItemDiagnostics:
    batch_index: int
    status: StatusCode
    decision: Decision
    position_residual_m: float | None = None
    orientation_residual_rad: float | None = None
    minimum_joint_margin_rad: float | None = None
    minimum_collision_margin_m: float | None = None
    condition_number: float | None = None
    maximum_joint_step_rad: float | None = None
    first_invalid_horizon_sample: int | None = None
    active_pair_count: int = 0
    capacity_overflow: bool = False
    iterations: int = 0
    accepted_step_size: float | None = None
    interventions: tuple[str, ...] = ()
    backend_diagnostics: tuple[tuple[str, Any], ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "status", _enum(StatusCode, self.status))
        object.__setattr__(self, "decision", _enum(Decision, self.decision))
        object.__setattr__(self, "interventions", tuple(self.interventions))
        object.__setattr__(
            self,
            "backend_diagnostics",
            _metadata_pairs(self.backend_diagnostics),
        )
        _require_nonnegative("batch_index", self.batch_index)
        _require_nonnegative("active_pair_count", self.active_pair_count)
        _require_nonnegative("iterations", self.iterations)
        if self.first_invalid_horizon_sample is not None:
            _require_nonnegative("first_invalid_horizon_sample", self.first_invalid_horizon_sample)

    def numerical_values(self) -> tuple[float, ...]:
        names = (
            "position_residual_m",
            "orientation_residual_rad",
            "minimum_joint_margin_rad",
            "minimum_collision_margin_m",
            "condition_number",
            "maximum_joint_step_rad",
            "accepted_step_size",
        )
        return tuple(float(value) for name in names if (value := getattr(self, name)) is not None)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> ItemDiagnostics:
        return cls(
            batch_index=int(value["batch_index"]),
            status=StatusCode(value["status"]),
            decision=Decision(value["decision"]),
            position_residual_m=value.get("position_residual_m"),
            orientation_residual_rad=value.get("orientation_residual_rad"),
            minimum_joint_margin_rad=value.get("minimum_joint_margin_rad"),
            minimum_collision_margin_m=value.get("minimum_collision_margin_m"),
            condition_number=value.get("condition_number"),
            maximum_joint_step_rad=value.get("maximum_joint_step_rad"),
            first_invalid_horizon_sample=value.get("first_invalid_horizon_sample"),
            active_pair_count=int(value.get("active_pair_count", 0)),
            capacity_overflow=bool(value.get("capacity_overflow", False)),
            iterations=int(value.get("iterations", 0)),
            accepted_step_size=value.get("accepted_step_size"),
            interventions=tuple(value.get("interventions", ())),
            backend_diagnostics=tuple(
                (str(key), _deep_tuple(item)) for key, item in value.get("backend_diagnostics", ())
            ),
        )


@dataclass(frozen=True)
class WbcBatchResult:
    request_id: str
    batch_size: int
    q_solution: Any
    items: tuple[ItemDiagnostics, ...]
    attribution: ExecutionAttribution
    timing: TimingBreakdown
    provenance: ResultProvenance

    def __post_init__(self) -> None:
        object.__setattr__(self, "items", tuple(self.items))
        errors = self.validation_errors()
        if errors:
            raise ContractViolation("; ".join(errors))

    def validation_errors(self) -> list[str]:
        errors: list[str] = []
        if not self.request_id:
            errors.append("request_id must not be empty")
        if self.batch_size <= 0:
            errors.append("batch_size must be positive")
        if len(self.items) != self.batch_size:
            errors.append("one ItemDiagnostics entry is required per batch item")
        expected_indices = list(range(self.batch_size))
        if sorted(item.batch_index for item in self.items) != expected_indices:
            errors.append("batch indices must be unique and contiguous")
        solution_shape = _shape(self.q_solution)
        if len(solution_shape) < 2 or solution_shape[0] != self.batch_size:
            errors.append(f"q_solution batch dimension mismatch: {solution_shape}")

        candidates = [
            item
            for item in self.items
            if item.decision in {Decision.APPROVED, Decision.NEEDS_VERIFICATION}
        ]
        if candidates and self.attribution.placeholder_output_used:
            errors.append("placeholder output cannot be accepted as a candidate")
        if candidates and self.attribution.target_required:
            requested = self.attribution.requested_target
            actual = self.attribution.actual_device
            if requested is ExecutionTarget.GPU and actual is not DeviceKind.CUDA:
                errors.append("required GPU execution did not occur")
            if requested is ExecutionTarget.CPU and actual is not DeviceKind.CPU:
                errors.append("required CPU execution did not occur")
            if self.attribution.cpu_fallback_used:
                errors.append("fallback output cannot satisfy a required execution target")
            if not self.attribution.device_execution_proven:
                errors.append("required device execution is not proven")

        for item in self.items:
            if item.decision is Decision.APPROVED and item.status is not StatusCode.SOLVED:
                errors.append(f"item {item.batch_index}: only solved status can be approved")
            if item.decision is Decision.REJECTED and item.status is StatusCode.SOLVED:
                errors.append(f"item {item.batch_index}: solved status cannot be rejected")
            if (
                item.decision is Decision.NEEDS_VERIFICATION
                and item.status is not StatusCode.SOLVED
            ):
                errors.append(f"item {item.batch_index}: only solved status can await verification")
            if (
                item.decision in {Decision.APPROVED, Decision.NEEDS_VERIFICATION}
                and item.capacity_overflow
            ):
                errors.append(f"item {item.batch_index}: capacity overflow cannot be accepted")
            if (
                item.decision in {Decision.APPROVED, Decision.NEEDS_VERIFICATION}
                and item.first_invalid_horizon_sample is not None
            ):
                errors.append(f"item {item.batch_index}: invalid horizon sample cannot be accepted")
            if item.decision in {
                Decision.APPROVED,
                Decision.NEEDS_VERIFICATION,
            } and not all(math.isfinite(value) for value in item.numerical_values()):
                errors.append(f"item {item.batch_index}: non-finite diagnostics cannot be accepted")
            if (
                item.decision in {Decision.APPROVED, Decision.NEEDS_VERIFICATION}
                and item.minimum_joint_margin_rad is not None
                and item.minimum_joint_margin_rad < 0
            ):
                errors.append(f"item {item.batch_index}: negative joint margin cannot be accepted")
            if (
                item.decision in {Decision.APPROVED, Decision.NEEDS_VERIFICATION}
                and item.minimum_collision_margin_m is not None
                and item.minimum_collision_margin_m < 0
            ):
                errors.append(
                    f"item {item.batch_index}: negative collision margin cannot be accepted"
                )
        return errors

    def to_dict(self) -> dict[str, Any]:
        return _jsonable(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), allow_nan=False, sort_keys=True, separators=(",", ":"))

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> WbcBatchResult:
        value = _restore_jsonable(value)
        return cls(
            request_id=value["request_id"],
            batch_size=int(value["batch_size"]),
            q_solution=_deep_tuple(value["q_solution"]),
            items=tuple(ItemDiagnostics.from_dict(item) for item in value["items"]),
            attribution=ExecutionAttribution.from_dict(value["attribution"]),
            timing=TimingBreakdown.from_dict(value["timing"]),
            provenance=ResultProvenance.from_dict(value["provenance"]),
        )
