"""Model-derived, device-resident Newton/Warp frame kinematics."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

from ..contracts import RobotSolveSpec


def _unique_suffix_index(labels: list[str], requested: str, *, kind: str) -> int:
    matches = [
        index
        for index, label in enumerate(labels)
        if label == requested or label.endswith(f"/{requested}")
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"Newton model must contain exactly one {kind} matching {requested!r}; "
            f"found {len(matches)}"
        )
    return matches[0]


def _span(starts: list[int], index: int, total: int) -> int:
    start = starts[index]
    following = [value for value in starts[index + 1 :] if value > start]
    return (min(following) if following else total) - start


def _quat_xyzw_to_matrix(torch: Any, quaternion: Any) -> Any:
    quaternion = quaternion / torch.clamp(
        torch.linalg.vector_norm(quaternion, dim=-1, keepdim=True), min=1e-12
    )
    x, y, z, w = quaternion.unbind(dim=-1)
    return torch.stack(
        (
            1.0 - 2.0 * (y * y + z * z),
            2.0 * (x * y - z * w),
            2.0 * (x * z + y * w),
            2.0 * (x * y + z * w),
            1.0 - 2.0 * (x * x + z * z),
            2.0 * (y * z - x * w),
            2.0 * (x * z - y * w),
            2.0 * (y * z + x * w),
            1.0 - 2.0 * (x * x + y * y),
        ),
        dim=-1,
    ).reshape(*quaternion.shape[:-1], 3, 3)


def _skew(torch: Any, vector: Any) -> Any:
    x, y, z = vector.unbind(dim=-1)
    zero = torch.zeros_like(x)
    return torch.stack((zero, -z, y, z, zero, -x, -y, x, zero), dim=-1).reshape(
        *vector.shape[:-1], 3, 3
    )


def _quat_multiply_xyzw(torch: Any, left: Any, right: Any) -> Any:
    lx, ly, lz, lw = left.unbind(dim=-1)
    rx, ry, rz, rw = right.unbind(dim=-1)
    return torch.stack(
        (
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
            lw * rw - lx * rx - ly * ry - lz * rz,
        ),
        dim=-1,
    )


class NewtonModelKinematics:
    """Evaluate model-derived frames and integrate active tangent velocities."""

    def __init__(
        self,
        batch_size: int,
        urdf_path: Path,
        cache_dir: Path,
        robot_spec: RobotSolveSpec,
        frame: str | tuple[str, ...],
        *,
        default_configuration: tuple[float, ...] | None = None,
        device: str = "cuda:0",
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        frames = (frame,) if isinstance(frame, str) else tuple(frame)
        if not frames:
            raise ValueError("at least one frame is required")
        missing_frames = sorted(set(frames) - set(robot_spec.task_frames))
        if missing_frames:
            raise ValueError(
                "frames are absent from RobotSolveSpec.task_frames: "
                + ", ".join(missing_frames)
            )
        if (
            not robot_spec.floating_base
            and robot_spec.configuration_dim != robot_spec.velocity_dim
        ):
            raise RuntimeError(
                "generic Newton kinematics currently requires one coordinate per velocity"
            )
        if robot_spec.floating_base and (
            robot_spec.configuration_dim != robot_spec.velocity_dim + 1
        ):
            raise RuntimeError("floating-base Newton kinematics requires nq=nv+1")
        if robot_spec.floating_base and not all(
            (
                robot_spec.joint_configuration_indices,
                robot_spec.joint_configuration_sizes,
                robot_spec.joint_velocity_indices,
                robot_spec.joint_velocity_sizes,
            )
        ):
            raise RuntimeError(
                "floating-base Newton kinematics requires model-derived joint spans"
            )

        import newton
        import torch
        import warp as wp
        from newton import ik

        if not torch.cuda.is_available() or torch.device(device).type != "cuda":
            raise RuntimeError("NewtonModelKinematics requires CUDA")
        self.batch_size = batch_size
        self.robot_spec = robot_spec
        self.frames = frames
        self.frame = frames[0] if len(frames) == 1 else None
        self.active_dim = len(robot_spec.active_velocity_indices)
        self.input_configuration_dim = (
            robot_spec.configuration_dim
            if robot_spec.floating_base
            else self.active_dim
        )
        self.device = torch.device(device)
        self.torch = torch
        self.wp = wp
        self.newton = newton
        self.ik = ik

        source = urdf_path.expanduser().resolve()
        if not source.is_file():
            raise FileNotFoundError(source)
        cache_dir.mkdir(parents=True, exist_ok=True)
        stripped = cache_dir / f"{robot_spec.model_hash[:16]}-kinematics-only.urdf"
        tree = ET.parse(source)
        root = tree.getroot()
        for link in root.findall("link"):
            for tag in ("visual", "collision", "inertial"):
                for element in list(link.findall(tag)):
                    link.remove(element)
        tree.write(stripped, encoding="unicode")

        builder = newton.ModelBuilder()
        builder.add_urdf(
            str(stripped),
            floating=robot_spec.floating_base,
            hide_visuals=True,
            enable_self_collisions=False,
            collapse_fixed_joints=False,
        )
        self.model = builder.finalize(device=device)
        q_starts = wp.to_torch(self.model.joint_q_start).cpu().tolist()
        qd_starts = wp.to_torch(self.model.joint_qd_start).cpu().tolist()
        if robot_spec.floating_base:
            q_indices, qd_indices, source_q_indices = self._floating_mappings(
                q_starts, qd_starts
            )
        else:
            q_indices, qd_indices, source_q_indices = self._fixed_mappings(
                q_starts, qd_starts
            )
        if (
            len(set(q_indices)) != len(q_indices)
            or len(set(source_q_indices)) != len(source_q_indices)
            or len(set(qd_indices)) != self.active_dim
        ):
            raise RuntimeError("active Newton coordinate mappings must be unique")

        self.ee_indices = tuple(
            _unique_suffix_index(self.model.body_label, value, kind="body frame")
            for value in frames
        )
        self.ee_index = self.ee_indices[0] if len(self.ee_indices) == 1 else None
        self._ee_body_indices = torch.tensor(
            self.ee_indices, dtype=torch.long, device=self.device
        )
        self._jacobian_signs = torch.tensor(
            (-1.0, -1.0, -1.0, 1.0, 1.0, 1.0) * len(self.ee_indices),
            dtype=torch.float32,
            device=self.device,
        )
        active_defaults = (
            (0.0,) * robot_spec.configuration_dim
            if default_configuration is None
            else tuple(default_configuration)
        )
        if len(active_defaults) != robot_spec.configuration_dim:
            raise ValueError(
                "default_configuration must match RobotSolveSpec.configuration_dim"
            )
        defaults = [0.0] * self.model.joint_coord_count
        for newton_index, input_index in zip(
            self._mapped_newton_q_indices, source_q_indices, strict=True
        ):
            defaults[newton_index] = float(active_defaults[input_index])
        self._q_full = (
            torch.tensor(defaults, dtype=torch.float32, device=self.device)
            .expand(batch_size, -1)
            .clone()
        )
        self._active_q_indices = torch.tensor(
            self._mapped_newton_q_indices, dtype=torch.long, device=self.device
        )
        self._source_q_indices = torch.tensor(
            source_q_indices, dtype=torch.long, device=self.device
        )
        self._active_qd_indices = torch.tensor(
            qd_indices, dtype=torch.long, device=self.device
        )
        self._floating_base_active_indices_tensor = (
            torch.tensor(
                self._floating_base_active_indices,
                dtype=torch.long,
                device=self.device,
            )
            if robot_spec.floating_base
            else None
        )
        self._floating_scalar_q_indices_tensor = (
            torch.tensor(
                self._floating_scalar_q_indices,
                dtype=torch.long,
                device=self.device,
            )
            if robot_spec.floating_base
            else None
        )
        self._floating_scalar_active_indices_tensor = (
            torch.tensor(
                self._floating_scalar_active_indices,
                dtype=torch.long,
                device=self.device,
            )
            if robot_spec.floating_base
            else None
        )
        self._q_warp = wp.from_torch(self._q_full)
        objectives = []
        for body_index in self.ee_indices:
            objectives.extend(
                (
                    ik.IKObjectivePosition(
                        body_index,
                        wp.vec3(),
                        wp.zeros(batch_size, dtype=wp.vec3, device=self.model.device),
                    ),
                    ik.IKObjectiveRotation(
                        body_index,
                        wp.quat_identity(),
                        wp.zeros(batch_size, dtype=wp.vec4, device=self.model.device),
                    ),
                )
            )
        solver = ik.IKSolver(
            self.model,
            batch_size,
            objectives,
            jacobian_mode=ik.IKJacobianType.ANALYTIC,
        )
        self._implementation = solver._impl
        for name in ("_ctx_solver", "_residuals_analytic", "_jacobian_at"):
            if not hasattr(self._implementation, name):
                raise RuntimeError(f"Newton kinematics hook missing: {name}")
        self._context = self._implementation._ctx_solver(self._q_warp)
        self._body_torch = wp.to_torch(self._implementation.body_q)
        self._jacobian_torch = wp.to_torch(self._implementation.jacobian)
        self._streams: dict[int, Any] = {}
        self._evaluate_graph = None
        self._evaluate_pose_graph = None
        self._dq_full = torch.zeros(
            (batch_size, self.model.joint_dof_count),
            dtype=torch.float32,
            device=self.device,
        )
        self._q_integrated = torch.empty_like(self._q_full)
        self._qd_scratch = torch.zeros_like(self._dq_full)
        self._dq_warp = wp.from_torch(self._dq_full)
        self._q_integrated_warp = wp.from_torch(self._q_integrated)
        self._qd_scratch_warp = wp.from_torch(self._qd_scratch)

    def _fixed_mappings(
        self, q_starts: list[int], qd_starts: list[int]
    ) -> tuple[list[int], list[int], list[int]]:
        active_joint_names = self.robot_spec.active_joint_names
        if not active_joint_names:
            try:
                active_joint_names = tuple(
                    self.robot_spec.joint_names[index]
                    for index in self.robot_spec.active_velocity_indices
                )
            except IndexError as error:
                raise RuntimeError(
                    "RobotSolveSpec must provide active_joint_names when velocity "
                    "indices do not directly index joint_names"
                ) from error
        q_indices: list[int] = []
        qd_indices: list[int] = []
        for joint_name in active_joint_names:
            joint_index = _unique_suffix_index(
                self.model.joint_label, joint_name, kind="joint"
            )
            if (
                _span(q_starts, joint_index, self.model.joint_coord_count) != 1
                or _span(qd_starts, joint_index, self.model.joint_dof_count) != 1
            ):
                raise RuntimeError(
                    f"joint {joint_name!r} is not a one-coordinate/one-velocity joint"
                )
            q_indices.append(int(q_starts[joint_index]))
            qd_indices.append(int(qd_starts[joint_index]))
        source_q_indices = list(self.robot_spec.active_configuration_indices)
        if not source_q_indices:
            source_q_indices = list(self.robot_spec.active_velocity_indices)
        if len(source_q_indices) != self.active_dim:
            raise RuntimeError(
                "RobotSolveSpec must map every active joint to a configuration index"
            )
        self._mapped_newton_q_indices = q_indices
        return q_indices, qd_indices, source_q_indices

    def _floating_mappings(
        self, q_starts: list[int], qd_starts: list[int]
    ) -> tuple[list[int], list[int], list[int]]:
        spec = self.robot_spec
        span_fields = (
            spec.joint_configuration_indices,
            spec.joint_configuration_sizes,
            spec.joint_velocity_indices,
            spec.joint_velocity_sizes,
        )
        if not all(span_fields):
            raise RuntimeError(
                "floating-base Newton kinematics requires model-derived joint spans"
            )
        free_candidates = [
            index
            for index in range(self.model.joint_count)
            if _span(q_starts, index, self.model.joint_coord_count) == 7
            and _span(qd_starts, index, self.model.joint_dof_count) == 6
        ]
        if len(free_candidates) != 1:
            raise RuntimeError("Newton model must contain exactly one free joint")
        source_free = [
            index
            for index, (q_size, v_size) in enumerate(
                zip(
                    spec.joint_configuration_sizes,
                    spec.joint_velocity_sizes,
                    strict=True,
                )
            )
            if q_size == 7 and v_size == 6
        ]
        if len(source_free) != 1:
            raise RuntimeError("EmbodiK model must contain exactly one free joint")

        free_joint = free_candidates[0]
        source_free_joint = source_free[0]
        source_q_start = spec.joint_configuration_indices[source_free_joint]
        source_v_start = spec.joint_velocity_indices[source_free_joint]
        newton_q_start = int(q_starts[free_joint])
        newton_v_start = int(qd_starts[free_joint])
        mapped_newton_q = [newton_q_start + offset for offset in range(7)]
        mapped_source_q = [source_q_start + offset for offset in range(7)]
        velocity_map = {
            source_v_start + offset: newton_v_start + offset for offset in range(6)
        }
        active_lookup = {
            source_index: active_index
            for active_index, source_index in enumerate(spec.active_velocity_indices)
        }
        base_source_velocities = tuple(range(source_v_start, source_v_start + 6))
        if not all(index in active_lookup for index in base_source_velocities):
            raise RuntimeError(
                "floating-base solve must include all six root tangent velocities"
            )
        self._floating_source_q_start = source_q_start
        self._floating_base_active_indices = tuple(
            active_lookup[index] for index in base_source_velocities
        )
        active_scalar_q_indices: list[int] = []
        active_scalar_velocity_indices: list[int] = []

        for joint_index, joint_name in enumerate(spec.joint_names):
            q_size = spec.joint_configuration_sizes[joint_index]
            v_size = spec.joint_velocity_sizes[joint_index]
            if joint_index == source_free_joint or (q_size == 0 and v_size == 0):
                continue
            if q_size != 1 or v_size != 1:
                raise RuntimeError(
                    f"floating model joint {joint_name!r} has unsupported nq/nv="
                    f"{q_size}/{v_size}; only a free root plus scalar joints are supported"
                )
            newton_joint = _unique_suffix_index(
                self.model.joint_label, joint_name, kind="joint"
            )
            if (
                _span(q_starts, newton_joint, self.model.joint_coord_count) != 1
                or _span(qd_starts, newton_joint, self.model.joint_dof_count) != 1
            ):
                raise RuntimeError(f"Newton joint {joint_name!r} is not scalar")
            mapped_newton_q.append(int(q_starts[newton_joint]))
            mapped_source_q.append(spec.joint_configuration_indices[joint_index])
            velocity_map[spec.joint_velocity_indices[joint_index]] = int(
                qd_starts[newton_joint]
            )
            source_velocity = spec.joint_velocity_indices[joint_index]
            if source_velocity in active_lookup:
                active_scalar_q_indices.append(
                    spec.joint_configuration_indices[joint_index]
                )
                active_scalar_velocity_indices.append(active_lookup[source_velocity])

        missing = [
            index for index in spec.active_velocity_indices if index not in velocity_map
        ]
        if missing:
            raise RuntimeError(f"active velocity mappings are missing: {missing}")
        qd_indices = [velocity_map[index] for index in spec.active_velocity_indices]
        self._floating_scalar_q_indices = tuple(active_scalar_q_indices)
        self._floating_scalar_active_indices = tuple(active_scalar_velocity_indices)
        self._mapped_newton_q_indices = mapped_newton_q
        return mapped_newton_q, qd_indices, mapped_source_q

    def _current_warp_stream(self) -> Any:
        torch_stream = self.torch.cuda.current_stream(self.device)
        pointer = int(torch_stream.cuda_stream)
        stream = self._streams.get(pointer)
        if stream is None:
            stream = self.wp.Stream(self.model.device, cuda_stream=pointer)
            self._streams[pointer] = stream
        return stream

    def _validate(self, q_active: Any) -> None:
        torch = self.torch
        if not isinstance(q_active, torch.Tensor):
            raise TypeError("q_active must be a torch.Tensor")
        if q_active.shape != (self.batch_size, self.input_configuration_dim):
            raise ValueError(
                "q_active must have shape "
                f"{(self.batch_size, self.input_configuration_dim)}, "
                f"got {tuple(q_active.shape)}"
            )
        if q_active.device != self.device or q_active.dtype is not torch.float32:
            raise ValueError("q_active must be CUDA float32 on the configured device")
        if not bool(torch.isfinite(q_active).all().item()):
            raise ValueError("q_active contains non-finite values")

    def evaluate(self, q_active: Any) -> tuple[Any, Any]:
        self._validate(q_active)
        return self._evaluate_trusted(q_active)

    def _evaluate_trusted(self, q_active: Any) -> tuple[Any, Any]:
        torch = self.torch
        mapped_q = q_active.index_select(1, self._source_q_indices)
        self._q_full.index_copy_(1, self._active_q_indices, mapped_q)
        if self._evaluate_graph is None:
            with self.wp.ScopedCapture(device=self.model.device) as capture:
                self._implementation._residuals_analytic(self._context)
                self._implementation._jacobian_at(self._context)
            self._evaluate_graph = capture.graph
        stream = self._current_warp_stream()
        if torch.cuda.is_current_stream_capturing():
            with self.wp.ScopedStream(stream, sync_enter=False):
                self._implementation._residuals_analytic(self._context)
                self._implementation._jacobian_at(self._context)
        else:
            self.wp.capture_launch(self._evaluate_graph, stream=stream)
        pose = self._body_torch.index_select(1, self._ee_body_indices)
        residual_jacobian = self._jacobian_torch.index_select(
            2, self._active_qd_indices
        )
        physical_jacobian = residual_jacobian * self._jacobian_signs[None, :, None]
        if self.robot_spec.floating_base:
            base_start = self._floating_source_q_start
            position = q_active[:, base_start : base_start + 3]
            rotation = _quat_xyzw_to_matrix(
                torch, q_active[:, base_start + 3 : base_start + 7]
            )
            tangent_transform = torch.zeros(
                (self.batch_size, 6, 6),
                dtype=torch.float32,
                device=self.device,
            )
            tangent_transform[:, :3, :3] = rotation
            tangent_transform[:, :3, 3:] = _skew(torch, position) @ rotation
            tangent_transform[:, 3:, 3:] = rotation
            physical_jacobian = physical_jacobian.clone()
            base_jacobian = physical_jacobian.index_select(
                2, self._floating_base_active_indices_tensor
            )
            physical_jacobian.index_copy_(
                2,
                self._floating_base_active_indices_tensor,
                base_jacobian @ tangent_transform,
            )
        if len(self.ee_indices) == 1:
            pose = pose[:, 0]
        return pose, physical_jacobian

    def _evaluate_pose_trusted(self, q_active: Any) -> Any:
        """Evaluate only frame poses for a tensor validated by the caller."""

        torch = self.torch
        mapped_q = q_active.index_select(1, self._source_q_indices)
        self._q_full.index_copy_(1, self._active_q_indices, mapped_q)
        if self._evaluate_pose_graph is None:
            with self.wp.ScopedCapture(device=self.model.device) as capture:
                self._implementation._residuals_analytic(self._context)
            self._evaluate_pose_graph = capture.graph
        stream = self._current_warp_stream()
        if torch.cuda.is_current_stream_capturing():
            with self.wp.ScopedStream(stream, sync_enter=False):
                self._implementation._residuals_analytic(self._context)
        else:
            self.wp.capture_launch(self._evaluate_pose_graph, stream=stream)
        pose = self._body_torch.index_select(1, self._ee_body_indices)
        return pose[:, 0] if len(self.ee_indices) == 1 else pose

    def integrate(self, q: Any, velocity: Any, dt: float) -> Any:
        """Integrate active tangent velocities with Newton's manifold kernel."""

        self._validate(q)
        if (
            not isinstance(velocity, self.torch.Tensor)
            or velocity.shape != (self.batch_size, self.active_dim)
            or velocity.device != self.device
            or velocity.dtype is not self.torch.float32
        ):
            raise ValueError(
                "velocity must be CUDA float32 with the active velocity shape"
            )
        if not isinstance(dt, (float, int)) or not float(dt) > 0.0:
            raise ValueError("dt must be positive")
        return self._integrate_trusted(q, velocity, float(dt))

    def _integrate_trusted(self, q: Any, velocity: Any, dt: float) -> Any:
        """Integrate tensors already validated by an enclosing solver request."""

        if self.robot_spec.floating_base:
            return self._integrate_floating(q, velocity, dt)
        mapped_q = q.index_select(1, self._source_q_indices)
        self._q_full.index_copy_(1, self._active_q_indices, mapped_q)
        self._dq_full.zero_()
        self._dq_full.index_copy_(1, self._active_qd_indices, velocity)
        stream = self._current_warp_stream()
        with self.wp.ScopedStream(stream, sync_enter=False):
            self._implementation._integrate_dq(
                self._q_warp,
                dq_in=self._dq_warp,
                joint_q_out=self._q_integrated_warp,
                joint_qd_out=self._qd_scratch_warp,
                step_size=dt,
            )
        output = q.clone()
        mapped_output = self._q_integrated.index_select(1, self._active_q_indices)
        output.index_copy_(1, self._source_q_indices, mapped_output)
        return output

    def _integrate_floating(self, q: Any, velocity: Any, dt: float) -> Any:
        torch = self.torch
        output = q.clone()
        base_start = self._floating_source_q_start
        base_velocity = velocity.index_select(
            1, self._floating_base_active_indices_tensor
        )
        linear = base_velocity[:, :3]
        angular = base_velocity[:, 3:]
        quaternion = q[:, base_start + 3 : base_start + 7]
        rotation = _quat_xyzw_to_matrix(torch, quaternion)

        phi = angular * dt
        theta = torch.linalg.vector_norm(phi, dim=-1, keepdim=True)
        phi_skew = _skew(torch, phi)
        phi_skew_squared = phi_skew @ phi_skew
        theta_squared = theta * theta
        small = theta < 1e-4
        a = torch.where(
            small,
            0.5 - theta_squared / 24.0,
            (1.0 - torch.cos(theta)) / torch.clamp(theta_squared, min=1e-16),
        )
        b = torch.where(
            small,
            1.0 / 6.0 - theta_squared / 120.0,
            (theta - torch.sin(theta)) / torch.clamp(theta_squared * theta, min=1e-16),
        )
        identity = torch.eye(3, dtype=torch.float32, device=self.device).expand(
            self.batch_size, -1, -1
        )
        translation_jacobian = (
            identity + a[:, None] * phi_skew + b[:, None] * (phi_skew_squared)
        )
        local_translation = (
            translation_jacobian @ (linear * dt).unsqueeze(-1)
        ).squeeze(-1)
        output[:, base_start : base_start + 3] = q[:, base_start : base_start + 3] + (
            rotation @ local_translation.unsqueeze(-1)
        ).squeeze(-1)

        half_theta = 0.5 * theta
        scale = torch.where(
            theta < 1e-8,
            0.5 - theta_squared / 48.0,
            torch.sin(half_theta) / torch.clamp(theta, min=1e-16),
        )
        delta_quaternion = torch.cat((phi * scale, torch.cos(half_theta)), dim=-1)
        integrated_quaternion = _quat_multiply_xyzw(torch, quaternion, delta_quaternion)
        integrated_quaternion = integrated_quaternion / torch.clamp(
            torch.linalg.vector_norm(integrated_quaternion, dim=-1, keepdim=True),
            min=1e-12,
        )
        output[:, base_start + 3 : base_start + 7] = integrated_quaternion
        if self._floating_scalar_q_indices:
            output.index_copy_(
                1,
                self._floating_scalar_q_indices_tensor,
                q.index_select(1, self._floating_scalar_q_indices_tensor)
                + dt
                * velocity.index_select(1, self._floating_scalar_active_indices_tensor),
            )
        return output

    def metadata(self) -> dict[str, Any]:
        return {
            "backend": "newton_warp",
            "newton_version": self.newton.__version__,
            "warp_version": self.wp.__version__,
            "model_hash": self.robot_spec.model_hash,
            "configuration_dim": self.robot_spec.configuration_dim,
            "velocity_dim": self.robot_spec.velocity_dim,
            "newton_configuration_dim": self.model.joint_coord_count,
            "newton_velocity_dim": self.model.joint_dof_count,
            "active_velocity_indices": list(self.robot_spec.active_velocity_indices),
            "floating_base": self.robot_spec.floating_base,
            "frames": list(self.frames),
            "batch_size": self.batch_size,
            "device": str(self.device),
            "precision": "float32",
            "host_roundtrip_in_evaluate": False,
            "private_api_guarded": True,
        }
