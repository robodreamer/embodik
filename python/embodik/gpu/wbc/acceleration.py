"""Public, deliberately narrow GPU acceleration-level solver.

This module exposes the first faithful acceleration-level GPU slice: bounded
fixed-base scalar joints, one diagonal ``MIN_ERROR`` generalized-acceleration
objective, and the three native acceleration lock policies. Unsupported
whole-body features are advertised explicitly through
:class:`GpuAccelerationCapabilities`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class GpuAccelerationCapabilities:
    """Feature-discovery contract for the public acceleration GPU slice."""

    supports_fixed_base_scalar_joints: bool = True
    supports_diagonal_generalized_acceleration_min_error: bool = True
    supports_joint_state_box_limits: bool = True
    supports_zero_acceleration_locks: bool = True
    supports_zero_next_velocity_locks: bool = True
    supports_fixed_current_position_locks: bool = True
    supports_device_resident_batch: bool = True
    supports_cuda_graph_capture: bool = True
    supports_frame_tasks: bool = False
    supports_centroidal_tasks_and_constraints: bool = False
    supports_contact_constraints: bool = False
    supports_collision_constraints: bool = False
    supports_effort_constraints: bool = False
    supports_floating_base: bool = False
    supports_multiple_priorities: bool = False
    supports_scale_tasks: bool = False


GPU_ACCELERATION_CAPABILITIES = GpuAccelerationCapabilities()


@dataclass(frozen=True)
class GpuAccelerationBatchResult:
    """Device-resident result with one status and motion tuple per world.

    ``status`` uses zero for success, one for invalid/non-finite input, two for
    an infeasible state box or lock intersection, and three for failed output
    verification.  Motion outputs for every unsuccessful world are NaN and
    must not be executed.
    """

    success: Any
    status: Any
    joint_accelerations: Any
    joint_velocities_next: Any
    q_solution: Any
    acceleration_lower: Any
    acceleration_upper: Any
    state_box_task_fallback_applied: Any
    acceleration_limits_applied: bool
    velocity_limits_applied: bool
    position_limits_applied: bool


@dataclass(frozen=True)
class GpuAccelerationResult:
    """NumPy-owning host result for one explicit acceleration solve."""

    success: bool
    status: str
    joint_accelerations: Any
    joint_velocities_next: Any
    q_solution: Any
    acceleration_lower: Any
    acceleration_upper: Any
    state_box_task_fallback_applied: bool
    acceleration_limits_applied: bool
    velocity_limits_applied: bool
    position_limits_applied: bool


class GpuAccelerationSolver:
    """Solve a diagonal acceleration objective inside the native state box.

    The model must be fixed-base and consist entirely of scalar one-DoF
    joints. ``acceleration_limits`` is mandatory because EmbodiK does not
    infer actuator acceleration authority from URDF velocity metadata.

    ``reference_acceleration`` is the assembled generalized-acceleration
    target. It is equivalent to a CPU posture ``MIN_ERROR`` task whose
    :class:`embodik.AccelerationTaskReference` has proportional and derivative
    gains set to zero. Position-error and velocity-error feedback are not
    inferred by this narrow API.

    The device method is allocation-static under CUDA graph capture for a
    fixed input shape.  It performs no host synchronization and reports
    failures independently for each world in the batch.
    """

    STATUS_SUCCESS = 0
    STATUS_INVALID_INPUT = 1
    STATUS_INFEASIBLE = 2
    STATUS_VERIFICATION_FAILED = 3

    _STATUS_NAMES = (
        "success",
        "invalid_input",
        "infeasible",
        "verification_failed",
    )
    _CONSTRAINT_TOLERANCE = 1.0e-8

    def __init__(
        self,
        robot: object,
        acceleration_limits: object,
        *,
        device: object = "cuda",
        apply_velocity_limits: bool = True,
        apply_position_limits: bool = True,
    ) -> None:
        import numpy as np
        import torch

        if bool(robot.is_floating_base):
            raise NotImplementedError(
                "GpuAccelerationSolver currently supports fixed-base models only"
            )
        self.configuration_dim = int(robot.nq)
        self.velocity_dim = int(robot.nv)
        if self.configuration_dim < 1 or self.configuration_dim != self.velocity_dim:
            raise ValueError(
                "fixed-base scalar-joint models must have equal positive nq and nv"
            )
        self._validate_scalar_joint_topology(robot)

        acceleration = np.asarray(acceleration_limits, dtype=float)
        if acceleration.shape != (self.velocity_dim,):
            raise ValueError(
                f"acceleration_limits must have shape {(self.velocity_dim,)}"
            )
        if not np.isfinite(acceleration).all() or np.any(acceleration <= 0.0):
            raise ValueError("acceleration_limits must be finite and strictly positive")

        expected = (self.velocity_dim,)
        if apply_position_limits:
            lower, upper = (
                np.asarray(value, dtype=float) for value in robot.get_joint_limits()
            )
            if lower.shape != expected or upper.shape != expected:
                raise ValueError(
                    f"joint position limits must both have shape {expected}"
                )
            if not np.isfinite(lower).all() or not np.isfinite(upper).all():
                raise ValueError("joint position limits must be finite")
            if np.any(lower > upper):
                raise ValueError("joint position limits must be ordered")
        else:
            lower = np.zeros(self.velocity_dim, dtype=float)
            upper = np.zeros(self.velocity_dim, dtype=float)
        if apply_velocity_limits:
            velocity = np.asarray(robot.get_velocity_limits(), dtype=float)
            if velocity.shape != expected:
                raise ValueError(f"joint velocity limits must have shape {expected}")
            if not np.isfinite(velocity).all() or np.any(velocity <= 0.0):
                raise ValueError(
                    "joint velocity limits must be finite and strictly positive"
                )
        else:
            velocity = np.zeros(self.velocity_dim, dtype=float)

        self.device = torch.device(device)
        if self.device.type != "cuda":
            raise ValueError("GpuAccelerationSolver requires a CUDA device")
        if self.device.index is None:
            self.device = torch.device("cuda", torch.cuda.current_device())
        self.dtype = torch.float64
        self.apply_velocity_limits = bool(apply_velocity_limits)
        self.apply_position_limits = bool(apply_position_limits)
        self._state_lower = torch.as_tensor(lower, dtype=self.dtype, device=self.device)
        self._state_upper = torch.as_tensor(upper, dtype=self.dtype, device=self.device)
        self._rate_lower = torch.as_tensor(-velocity, dtype=self.dtype, device=self.device)
        self._rate_upper = torch.as_tensor(velocity, dtype=self.dtype, device=self.device)
        self._acceleration_limits = torch.as_tensor(
            acceleration, dtype=self.dtype, device=self.device
        )
        self._zero_reference = torch.zeros(
            self.velocity_dim, dtype=self.dtype, device=self.device
        )
        self._false_mask = torch.zeros(
            self.velocity_dim, dtype=torch.bool, device=self.device
        )

    @staticmethod
    def capabilities() -> GpuAccelerationCapabilities:
        """Return the immutable public feature contract."""

        return GPU_ACCELERATION_CAPABILITIES

    def _validate_scalar_joint_topology(self, robot: object) -> None:
        names = tuple(str(value) for value in robot.get_joint_names())
        span_methods = (
            "get_joint_config_index",
            "get_joint_config_size",
            "get_joint_velocity_index",
            "get_joint_velocity_size",
        )
        if not all(hasattr(robot, method) for method in span_methods):
            if len(names) != self.velocity_dim:
                raise ValueError(
                    "model metadata must identify one scalar joint per velocity"
                )
            return

        configuration_indices = []
        velocity_indices = []
        for name in names:
            q_size = int(robot.get_joint_config_size(name))
            v_size = int(robot.get_joint_velocity_size(name))
            if q_size == 0 and v_size == 0:
                continue
            if q_size != 1 or v_size != 1:
                raise NotImplementedError(
                    f"joint {name!r} has nq/nv={q_size}/{v_size}; only scalar joints are supported"
                )
            configuration_indices.append(int(robot.get_joint_config_index(name)))
            velocity_indices.append(int(robot.get_joint_velocity_index(name)))
        expected = list(range(self.velocity_dim))
        if sorted(configuration_indices) != expected or sorted(velocity_indices) != expected:
            raise ValueError("scalar joint metadata must cover every model coordinate exactly once")
        if configuration_indices != velocity_indices:
            raise NotImplementedError(
                "configuration and velocity coordinates must share scalar-joint ordering"
            )

    def _world_dt(self, dt: object, batch_size: int):
        import torch

        if isinstance(dt, (int, float)) and not isinstance(dt, bool):
            return torch.full((), float(dt), dtype=self.dtype, device=self.device).expand(
                batch_size
            )
        if not isinstance(dt, torch.Tensor):
            raise TypeError("dt must be a real scalar or torch.Tensor")
        if dt.device != self.device:
            raise ValueError(f"dt must reside on {self.device}")
        if not dt.is_floating_point():
            raise TypeError("dt must be floating point")
        value = dt.to(dtype=self.dtype)
        if value.ndim == 0:
            return value.expand(batch_size)
        if value.shape == (batch_size,):
            return value
        if value.shape == (batch_size, 1):
            return value.squeeze(-1)
        raise ValueError(f"dt must be scalar or have shape {(batch_size,)} or {(batch_size, 1)}")

    def _batch_vector(self, value: object, batch_size: int, name: str, default: Any):
        import torch

        if value is None:
            value = default
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor")
        if value.device != self.device:
            raise ValueError(f"{name} must reside on {self.device}")
        if not value.is_floating_point():
            raise TypeError(f"{name} must be floating point")
        value = value.to(dtype=self.dtype)
        if value.shape == (self.velocity_dim,):
            return value.expand(batch_size, -1)
        if value.shape == (batch_size, self.velocity_dim):
            return value
        raise ValueError(
            f"{name} must have shape {(self.velocity_dim,)} or "
            f"{(batch_size, self.velocity_dim)}"
        )

    def _batch_mask(self, value: object, batch_size: int, name: str):
        import torch

        if value is None:
            value = self._false_mask
        if not isinstance(value, torch.Tensor) or value.dtype != torch.bool:
            raise TypeError(f"{name} must be a boolean torch.Tensor")
        if value.device != self.device:
            raise ValueError(f"{name} must reside on {self.device}")
        if value.shape == (self.velocity_dim,):
            return value.expand(batch_size, -1)
        if value.shape == (batch_size, self.velocity_dim):
            return value
        raise ValueError(
            f"{name} must have shape {(self.velocity_dim,)} or "
            f"{(batch_size, self.velocity_dim)}"
        )

    def solve_device_batch(
        self,
        q: object,
        dq: object,
        dt: object,
        reference_acceleration: object | None = None,
        zero_acceleration_mask: object | None = None,
        zero_next_velocity_mask: object | None = None,
        fixed_current_position_mask: object | None = None,
    ) -> GpuAccelerationBatchResult:
        """Solve one fixed-shape device batch without host synchronization.

        Failed worlds retain their status but expose NaN motion outputs, so a
        command cannot be consumed accidentally. All inputs may remain on the
        device across CUDA graph replays.
        """

        import torch

        from ._runtime.gpu_acceleration_constraints import acceleration_state_box_bounds

        if not isinstance(q, torch.Tensor) or not isinstance(dq, torch.Tensor):
            raise TypeError("q and dq must be torch.Tensor instances")
        if q.device != self.device or dq.device != self.device:
            raise ValueError(f"q and dq must reside on {self.device}")
        if not q.is_floating_point() or not dq.is_floating_point():
            raise TypeError("q and dq must be floating point")
        if q.ndim != 2 or q.shape[1] != self.configuration_dim:
            raise ValueError(f"q must have shape [batch, {self.configuration_dim}]")
        if dq.shape != (q.shape[0], self.velocity_dim):
            raise ValueError(f"dq must have shape {(q.shape[0], self.velocity_dim)}")
        batch_size = q.shape[0]
        if batch_size < 1:
            raise ValueError("device batch must contain at least one world")
        q = q.to(dtype=self.dtype)
        dq = dq.to(dtype=self.dtype)
        dt_world = self._world_dt(dt, batch_size)
        dt_column = dt_world.unsqueeze(-1)
        reference = self._batch_vector(
            reference_acceleration,
            batch_size,
            "reference_acceleration",
            self._zero_reference,
        )
        zero_acceleration = self._batch_mask(
            zero_acceleration_mask, batch_size, "zero_acceleration_mask"
        )
        zero_next_velocity = self._batch_mask(
            zero_next_velocity_mask, batch_size, "zero_next_velocity_mask"
        )
        fixed_position = self._batch_mask(
            fixed_current_position_mask,
            batch_size,
            "fixed_current_position_mask",
        )

        state_box = acceleration_state_box_bounds(
            q,
            dq,
            dt_world,
            state_lower=self._state_lower,
            state_upper=self._state_upper,
            rate_lower=self._rate_lower,
            rate_upper=self._rate_upper,
            acceleration_lower=-self._acceleration_limits,
            acceleration_upper=self._acceleration_limits,
            lower_braking_acceleration=self._acceleration_limits,
            upper_braking_acceleration=self._acceleration_limits,
            state_lower_active=self.apply_position_limits,
            state_upper_active=self.apply_position_limits,
            rate_lower_active=self.apply_velocity_limits,
            rate_upper_active=self.apply_velocity_limits,
            acceleration_lower_active=True,
            acceleration_upper_active=True,
        )

        overlap = (
            (zero_acceleration & zero_next_velocity)
            | (zero_acceleration & fixed_position)
            | (zero_next_velocity & fixed_position)
        ).any(dim=-1)
        safe_dt = torch.where(dt_column > 0.0, dt_column, torch.ones_like(dt_column))
        stop_target = -dq / safe_dt
        fixed_target = -2.0 * dq / safe_dt
        locked = zero_acceleration | zero_next_velocity | fixed_position
        lock_target = torch.where(
            zero_acceleration,
            torch.zeros_like(dq),
            torch.where(zero_next_velocity, stop_target, fixed_target),
        )
        tolerance = self._CONSTRAINT_TOLERANCE
        locks_compatible = (
            (~locked)
            | (
                (lock_target >= state_box.lower - tolerance)
                & (lock_target <= state_box.upper + tolerance)
                & torch.isfinite(lock_target)
            )
        ).all(dim=-1)
        lower = torch.where(locked, lock_target, state_box.lower)
        upper = torch.where(locked, lock_target, state_box.upper)

        requested = torch.where(locked, lock_target, reference)
        acceleration = torch.minimum(torch.maximum(requested, lower), upper)
        dq_next = dq + dt_column * acceleration
        q_next = q + dt_column * dq + 0.5 * dt_column * dt_column * acceleration

        candidate_finite = (
            torch.isfinite(reference).all(dim=-1)
            & torch.isfinite(acceleration).all(dim=-1)
            & torch.isfinite(dq_next).all(dim=-1)
            & torch.isfinite(q_next).all(dim=-1)
        )
        acceleration_valid = (
            (acceleration >= -self._acceleration_limits - tolerance)
            & (acceleration <= self._acceleration_limits + tolerance)
            & (acceleration >= lower - tolerance)
            & (acceleration <= upper + tolerance)
        ).all(dim=-1)
        velocity_valid = (
            (
                (dq_next >= self._rate_lower - tolerance)
                & (dq_next <= self._rate_upper + tolerance)
            ).all(dim=-1)
            if self.apply_velocity_limits
            else torch.ones(batch_size, dtype=torch.bool, device=self.device)
        )
        position_valid = (
            (
                (q_next >= self._state_lower - tolerance)
                & (q_next <= self._state_upper + tolerance)
            ).all(dim=-1)
            if self.apply_position_limits
            else torch.ones(batch_size, dtype=torch.bool, device=self.device)
        )
        lock_valid = (
            (~zero_acceleration | (torch.abs(acceleration) <= tolerance))
            & (~zero_next_velocity | (torch.abs(dq_next) <= tolerance))
            & (~fixed_position | (torch.abs(q_next - q) <= tolerance))
        ).all(dim=-1)

        input_valid = (
            torch.isfinite(q).all(dim=-1)
            & torch.isfinite(dq).all(dim=-1)
            & torch.isfinite(dt_world)
            & (dt_world > 0.0)
            & torch.isfinite(reference).all(dim=-1)
            & ~overlap
        )
        feasible = state_box.feasible & locks_compatible
        verified = (
            candidate_finite
            & acceleration_valid
            & velocity_valid
            & position_valid
            & lock_valid
        )
        success = input_valid & feasible & verified
        status = torch.where(
            ~input_valid,
            torch.full_like(input_valid, self.STATUS_INVALID_INPUT, dtype=torch.int8),
            torch.where(
                ~feasible,
                torch.full_like(input_valid, self.STATUS_INFEASIBLE, dtype=torch.int8),
                torch.where(
                    ~verified,
                    torch.full_like(
                        input_valid, self.STATUS_VERIFICATION_FAILED, dtype=torch.int8
                    ),
                    torch.full_like(input_valid, self.STATUS_SUCCESS, dtype=torch.int8),
                ),
            ),
        )
        executable = success.unsqueeze(-1)
        nan = torch.full_like(acceleration, torch.nan)
        acceleration_output = torch.where(executable, acceleration, nan)
        velocity_output = torch.where(executable, dq_next, nan)
        configuration_output = torch.where(executable, q_next, nan)
        fallback = success & (
            (state_box.lower > tolerance).any(dim=-1)
            | (state_box.upper < -tolerance).any(dim=-1)
        )
        return GpuAccelerationBatchResult(
            success=success,
            status=status,
            joint_accelerations=acceleration_output,
            joint_velocities_next=velocity_output,
            q_solution=configuration_output,
            acceleration_lower=lower,
            acceleration_upper=upper,
            state_box_task_fallback_applied=fallback,
            acceleration_limits_applied=True,
            velocity_limits_applied=self.apply_velocity_limits,
            position_limits_applied=self.apply_position_limits,
        )

    def solve(
        self,
        q: object,
        dq: object,
        dt: float,
        reference_acceleration: object | None = None,
        zero_acceleration_mask: object | None = None,
        zero_next_velocity_mask: object | None = None,
        fixed_current_position_mask: object | None = None,
    ) -> GpuAccelerationResult:
        """Run one host-visible solve from NumPy state and return NumPy outputs."""

        import numpy as np
        import torch

        q_array = np.asarray(q, dtype=float)
        dq_array = np.asarray(dq, dtype=float)
        expected = (self.velocity_dim,)
        if q_array.shape != expected or dq_array.shape != expected:
            raise ValueError(f"q and dq must both have shape {expected}")

        def vector(value: object | None, name: str):
            if value is None:
                return None
            array = np.asarray(value, dtype=float)
            if array.shape != expected:
                raise ValueError(f"{name} must have shape {expected}")
            return torch.as_tensor(array, dtype=self.dtype, device=self.device)

        def mask(value: object | None, name: str):
            if value is None:
                return None
            array = np.asarray(value)
            if array.shape != expected or array.dtype != np.bool_:
                raise ValueError(f"{name} must be a boolean NumPy array with shape {expected}")
            return torch.as_tensor(array, dtype=torch.bool, device=self.device)

        batch = self.solve_device_batch(
            torch.as_tensor(q_array, dtype=self.dtype, device=self.device).unsqueeze(0),
            torch.as_tensor(dq_array, dtype=self.dtype, device=self.device).unsqueeze(0),
            float(dt),
            vector(reference_acceleration, "reference_acceleration"),
            mask(zero_acceleration_mask, "zero_acceleration_mask"),
            mask(zero_next_velocity_mask, "zero_next_velocity_mask"),
            mask(fixed_current_position_mask, "fixed_current_position_mask"),
        )
        status_index = int(batch.status[0].cpu().item())
        success = bool(batch.success[0].cpu().item())
        empty = np.empty(0, dtype=float)
        return GpuAccelerationResult(
            success=success,
            status=self._STATUS_NAMES[status_index],
            joint_accelerations=(
                batch.joint_accelerations[0].detach().cpu().numpy().copy()
                if success
                else empty.copy()
            ),
            joint_velocities_next=(
                batch.joint_velocities_next[0].detach().cpu().numpy().copy()
                if success
                else empty.copy()
            ),
            q_solution=(
                batch.q_solution[0].detach().cpu().numpy().copy()
                if success
                else empty.copy()
            ),
            acceleration_lower=batch.acceleration_lower[0].detach().cpu().numpy().copy(),
            acceleration_upper=batch.acceleration_upper[0].detach().cpu().numpy().copy(),
            state_box_task_fallback_applied=bool(
                batch.state_box_task_fallback_applied[0].cpu().item()
            ),
            acceleration_limits_applied=True,
            velocity_limits_applied=self.apply_velocity_limits,
            position_limits_applied=self.apply_position_limits,
        )
