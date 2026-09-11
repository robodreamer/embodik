"""Mass-preserving Newton CoM kinematics in EmbodiK active tangent order.

Inputs are full RobotSolveSpec configurations (also for fixed-base models).
Unrepresented URDF scalar joints are locked at their neutral, zero coordinate,
as in EmbodiK's reduced model. Fixed-base universe-attached inertias are excluded
from the reduction, matching Pinocchio/EmbodiK centerOfMass; their URDF inertial
records are nevertheless preserved in the Newton model.
"""

from __future__ import annotations

import math
import tempfile
import warnings
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

from ..contracts import RobotSolveSpec
from .newton_model import _quat_xyzw_to_matrix, _skew, _unique_suffix_index


def _validate_spec(spec: RobotSolveSpec) -> list[tuple[str, int, int, int, int]]:
    fields = (
        spec.joint_configuration_indices,
        spec.joint_configuration_sizes,
        spec.joint_velocity_indices,
        spec.joint_velocity_sizes,
    )
    if any(len(values) != len(spec.joint_names) for values in fields):
        raise ValueError("CoM requires complete model-derived joint q/v spans")
    records = list(zip(spec.joint_names, *fields, strict=True))
    qs, vs, roots = [], [], 0
    for name, qi, nq, vi, nv in records:
        if (nq, nv) not in ((0, 0), (1, 1), (7, 6)):
            raise ValueError(f"unsupported q/v spans for {name}: {nq}/{nv}")
        if nq == 7:
            roots += 1
        qs.extend(range(qi, qi + nq))
        vs.extend(range(vi, vi + nv))
    if roots != int(spec.floating_base):
        raise ValueError("model must have exactly one free root iff floating_base")
    if sorted(qs) != list(range(spec.configuration_dim)) or sorted(vs) != list(
        range(spec.velocity_dim)
    ):
        raise ValueError("joint q/v spans must cover each coordinate exactly once")
    return records


def _inertial_urdf(source: Path) -> tuple[ET.ElementTree, dict[str, float]]:
    tree = ET.parse(source)
    masses = {}
    for link in tree.getroot().findall("link"):
        name = link.attrib["name"]
        for tag in ("visual", "collision"):
            for node in list(link.findall(tag)):
                link.remove(node)
        inertial = link.find("inertial")
        mass = 0.0
        if inertial is not None:
            node = inertial.find("mass")
            if node is None:
                raise ValueError(f"missing inertial mass on {name}")
            mass = float(node.attrib["value"])
            for node in inertial.iter():
                for value in node.attrib.values():
                    if not all(math.isfinite(float(x)) for x in value.split()):
                        raise ValueError(f"nonfinite inertial data on {name}")
        if not math.isfinite(mass) or mass < 0:
            raise ValueError(f"invalid mass on {name}")
        masses[name] = mass
    total = sum(masses.values())
    if not math.isfinite(total) or total <= 0:
        raise ValueError("CoM requires positive finite total mass")
    return tree, masses


class NewtonCoMEvaluator:
    """Replicated CUDA centroidal evaluator with no host reads in evaluation.

    ``evaluate(q)`` returns ``(position[B,3], jacobian[B,3,active_nv])``
    and ``evaluate_centroidal(q)`` additionally returns
    ``Ag[B,6,active_nv]``.  The first three rows of ``Ag`` are linear
    momentum and the last three are angular momentum about the system CoM,
    matching Pinocchio/EmbodiK's force-vector convention.  All outputs are
    float32 tensors on the configured CUDA device.

    Nonfinite configurations or invalid free-root quaternions produce NaNs
    for that world's entire output (fail closed, without a host
    synchronization). Metadata errors raise immediately. Outputs own their
    storage; calls must be serialized across CUDA streams. Use the caller's
    usual stream event dependencies when changing streams.

    ``evaluate_centroidal_bias(q, dq)`` returns the analytic
    ``dAg(q, dq) * dq`` from a stationary, massless outer FREE-joint probe.
    The original inertial tree is nested below that probe. Its root Coriolis
    wrench is therefore the system momentum-rate bias in world coordinates,
    without relying on the convention-corrected wrench of a moving FREE root.
    """

    supports_exact_centroidal_bias = True

    def __init__(
        self,
        batch_size: int,
        urdf_path: Path,
        cache_dir: Path,
        robot_spec: RobotSolveSpec,
        *,
        device: str = "cuda:0",
    ) -> None:
        if (
            isinstance(batch_size, bool)
            or not isinstance(batch_size, int)
            or batch_size <= 0
        ):
            raise ValueError("batch_size must be a positive integer")
        records = _validate_spec(robot_spec)
        tree, masses = _inertial_urdf(Path(urdf_path).expanduser().resolve())
        import newton
        import torch
        import warp as wp
        from newton._src.sim.inverse_dynamics import (
            _InverseDynamicsScratchBuffer,
            _compute_coriolis_force,
        )

        if torch.device(device).type != "cuda" or not torch.cuda.is_available():
            raise ValueError("NewtonCoMEvaluator requires a CUDA device")
        self.device = torch.device(device)
        if self.device.index is None:
            self.device = torch.device("cuda", torch.cuda.current_device())
        self.torch, self.wp, self.newton = torch, wp, newton
        self._compute_coriolis_force = _compute_coriolis_force
        self.batch_size, self.robot_spec = batch_size, robot_spec
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        # Unique path prevents concurrent constructors from overwriting imports.
        with tempfile.TemporaryDirectory(prefix="com-", dir=cache_dir) as directory:
            stripped = Path(directory) / "inertial.urdf"
            tree.write(stripped, encoding="unicode")
            source = newton.ModelBuilder()
            source.add_urdf(
                str(stripped),
                floating=robot_spec.floating_base,
                collapse_fixed_joints=False,
                hide_visuals=True,
                enable_self_collisions=False,
                ignore_inertial_definitions=False,
            )
            # A stationary, massless FREE root exposes the full-system RNEA
            # wrench without crossing the moving root's public COM-velocity
            # convention. The imported robot remains otherwise unchanged.
            bias_source = newton.ModelBuilder()
            probe_body = bias_source.add_link(
                mass=0.0,
                inertia=wp.mat33(0.0),
                com=wp.vec3(0.0),
                label="_embodik_centroidal_probe",
            )
            probe_joint = bias_source.add_joint_free(
                probe_body, parent=-1, label="_embodik_centroidal_probe_joint"
            )
            bias_source.add_articulation(
                [probe_joint], label="_embodik_centroidal_probe_articulation"
            )
            base_joint = None
            floating = False
            if robot_spec.floating_base:
                axis = newton.ModelBuilder.JointDofConfig
                linear_axes = [
                    axis(axis=value)
                    for value in ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
                ]
                angular_axes = [
                    axis(axis=value)
                    for value in ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
                ]
                base_joint = {
                    "joint_type": newton.JointType.FREE,
                    "linear_axes": linear_axes,
                    "angular_axes": angular_axes,
                }
                floating = None
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message="parent_body .* has zero or negative mass",
                    category=UserWarning,
                )
                bias_source.add_urdf(
                    str(stripped),
                    floating=floating,
                    base_joint=base_joint,
                    parent_body=probe_body,
                    collapse_fixed_joints=False,
                    hide_visuals=True,
                    enable_self_collisions=False,
                    ignore_inertial_definitions=False,
                )
        qstarts, vstarts = source.joint_q_start, source.joint_qd_start

        def spans(starts, total):
            return [
                ((starts[i + 1] if i + 1 < len(starts) else total) - starts[i])
                for i in range(len(starts))
            ]

        nq = spans(qstarts, source.joint_coord_count)
        nv = spans(vstarts, source.joint_dof_count)
        if any(pair not in ((0, 0), (1, 1), (7, 6)) for pair in zip(nq, nv)):
            raise ValueError("unsupported Newton q/v spans")
        roots = [i for i, pair in enumerate(zip(nq, nv)) if pair == (7, 6)]
        if len(roots) != int(robot_spec.floating_base):
            raise ValueError("unexpected Newton free-root layout")
        self._root = None
        qmap, qsource, vmap = [], [], {}
        represented_joints = set()
        for name, qi, qsize, vi, vsize in records:
            if not qsize:
                continue
            j = (
                roots[0]
                if qsize == 7
                else _unique_suffix_index(source.joint_label, name, kind="joint")
            )
            if (nq[j], nv[j]) != (qsize, vsize):
                raise ValueError(f"Newton q/v spans disagree for {name}")
            if qsize == 7:
                if source.joint_parent[j] != -1:
                    raise ValueError("free joint must be a root")
                self._root = (qi, vi, vstarts[j], source.joint_child[j])
            represented_joints.add(j)
            qmap.extend(range(qstarts[j], qstarts[j] + qsize))
            qsource.extend(range(qi, qi + qsize))
            vmap.update({vi + k: vstarts[j] + k for k in range(vsize)})
        if len(set(qmap)) != len(qmap):
            raise ValueError("duplicate Newton q mapping")
        # Include all inertias attached to modeled joints, including fixed and
        # neutral-locked descendants. Exclude the fixed universe subtree.
        moving_bodies = set()
        pending = set(range(source.joint_count))
        while pending:
            progressed = False
            for j in list(pending):
                parent = source.joint_parent[j]
                parent_joint = next(
                    (k for k in pending if source.joint_child[k] == parent), None
                )
                if parent_joint is not None:
                    continue
                if parent in moving_bodies or j in represented_joints:
                    moving_bodies.add(source.joint_child[j])
                pending.remove(j)
                progressed = True
            if not progressed:
                raise ValueError("cyclic Newton joint topology")
        weights = []
        for body, label in enumerate(source.body_label):
            matches = [
                mass
                for name, mass in masses.items()
                if label == name or label.endswith("/" + name)
            ]
            if len(matches) != 1 or not math.isclose(
                float(source.body_mass[body]), matches[0], rel_tol=1e-5, abs_tol=1e-8
            ):
                raise ValueError(f"Newton did not preserve URDF mass for {label}")
            weights.append(matches[0] if body in moving_bodies else 0.0)
        self.total_mass = sum(weights)
        if not math.isfinite(self.total_mass) or self.total_mass <= 0:
            raise ValueError("modeled CoM requires positive finite total mass")
        builder = newton.ModelBuilder()
        builder.replicate(source, world_count=batch_size)
        self.model = builder.finalize(device=str(self.device))
        self.state = self.model.state()
        # Construction may be called from a non-default Torch stream; finish
        # Newton's initialization before aliasing its buffers from Torch.
        wp.synchronize_device(self.model.device)
        if (
            self.model.articulation_count != batch_size
            or self.model.joint_count != batch_size * source.joint_count
            or self.model.body_count != batch_size * source.body_count
        ):
            raise ValueError("expected one articulation per replicated world")
        self._source = source

        def tensor(value, dtype=torch.long):
            return torch.tensor(value, dtype=dtype, device=self.device)

        self._qmap, self._qsource = tensor(qmap), tensor(qsource)
        self._vmap = tensor([vmap[i] for i in robot_spec.active_velocity_indices])
        self._weights = tensor(weights, torch.float32) / self.total_mass
        modeled = [body in moving_bodies for body in range(source.body_count)]
        self._modeled_body_mask = tensor(modeled, torch.float32)
        if not bool(torch.isfinite(self._weights).all().item()):
            raise ValueError("mass weights cannot be represented in float32")
        self._q = wp.to_torch(self.model.joint_q).reshape(batch_size, -1).clone()
        self._qd = wp.zeros(
            self.model.joint_dof_count, dtype=float, device=self.model.device
        )
        self._qw = wp.from_torch(self._q.flatten())
        self._J = wp.zeros(
            (
                batch_size,
                self.model.max_joints_per_articulation * 6,
                self.model.max_dofs_per_articulation,
            ),
            dtype=float,
            device=self.model.device,
        )
        self._S = wp.zeros(
            self.model.joint_dof_count,
            dtype=wp.spatial_vector,
            device=self.model.device,
        )
        self._body_q = wp.to_torch(self.state.body_q).reshape(
            batch_size, source.body_count, 7
        )
        self._body_com = wp.to_torch(self.model.body_com).reshape(
            batch_size, source.body_count, 3
        )
        model_mass = wp.to_torch(self.model.body_mass)
        self._body_inertia = wp.to_torch(self.model.body_inertia).reshape(
            batch_size, source.body_count, 3, 3
        )
        if (
            not bool(torch.isfinite(model_mass).all().item())
            or not bool(torch.isfinite(self._body_com).all().item())
            or not bool(torch.isfinite(self._body_inertia).all().item())
        ):
            raise ValueError("inertial mass/offset cannot be represented in float32")
        if source.joint_count != source.body_count or sorted(
            source.joint_child
        ) != list(range(source.body_count)):
            raise ValueError("centroidal reduction requires one Newton joint per body")
        self._jt = wp.to_torch(self._J).reshape(batch_size, source.joint_count, 6, -1)
        self._joint_body_indices = tensor(source.joint_child)
        self._joint_weights = self._weights[self._joint_body_indices]
        self._joint_masses = self._joint_weights * self.total_mass
        self._joint_modeled_mask = self._modeled_body_mask[
            self._joint_body_indices
        ]

        bias_qstarts = bias_source.joint_q_start
        bias_vstarts = bias_source.joint_qd_start
        bias_nq = spans(bias_qstarts, bias_source.joint_coord_count)
        bias_nv = spans(bias_vstarts, bias_source.joint_dof_count)
        if (bias_nq[probe_joint], bias_nv[probe_joint]) != (7, 6):
            raise ValueError("centroidal bias probe must have a 7/6 FREE root")
        inner_roots = [
            index
            for index, pair in enumerate(zip(bias_nq, bias_nv))
            if pair == (7, 6) and index != probe_joint
        ]
        if len(inner_roots) != int(robot_spec.floating_base):
            raise ValueError("centroidal bias probe has an unexpected inner root")
        bias_qmap, bias_qsource, bias_vmap = [], [], {}
        self._bias_inner_root = None
        for name, qi, qsize, vi, vsize in records:
            if not qsize:
                continue
            joint = (
                inner_roots[0]
                if qsize == 7
                else _unique_suffix_index(
                    bias_source.joint_label, name, kind="bias-probe joint"
                )
            )
            if (bias_nq[joint], bias_nv[joint]) != (qsize, vsize):
                raise ValueError(f"bias-probe q/v spans disagree for {name}")
            if qsize == 7:
                self._bias_inner_root = (
                    qi,
                    vi,
                    bias_vstarts[joint],
                    bias_source.joint_child[joint],
                )
            bias_qmap.extend(range(bias_qstarts[joint], bias_qstarts[joint] + qsize))
            bias_qsource.extend(range(qi, qi + qsize))
            bias_vmap.update({vi + k: bias_vstarts[joint] + k for k in range(vsize)})
        if len(bias_vmap) != robot_spec.velocity_dim:
            raise ValueError("bias-probe mapping must cover every model velocity")
        bias_builder = newton.ModelBuilder()
        bias_builder.replicate(bias_source, world_count=batch_size)
        self.bias_model = bias_builder.finalize(device=str(self.device))
        self.bias_state = self.bias_model.state()
        wp.synchronize_device(self.bias_model.device)
        if (
            self.bias_model.articulation_count != batch_size
            or self.bias_model.max_dofs_per_articulation
            != bias_source.joint_dof_count
            or bias_source.body_count != source.body_count + 1
            or bias_source.body_label[1:] != source.body_label
        ):
            raise ValueError("expected one complete bias-probe articulation per world")
        self._bias_qmap = tensor(bias_qmap)
        self._bias_qsource = tensor(bias_qsource)
        self._bias_vmap = tensor(
            [bias_vmap[index] for index in range(robot_spec.velocity_dim)]
        )
        self._bias_q = wp.to_torch(self.bias_state.joint_q).reshape(batch_size, -1)
        self._bias_qd = wp.to_torch(self.bias_state.joint_qd).reshape(batch_size, -1)
        self._bias_coriolis_wp = wp.zeros(
            self.bias_model.joint_dof_count,
            dtype=float,
            device=self.bias_model.device,
        )
        self._bias_coriolis = wp.to_torch(self._bias_coriolis_wp).reshape(
            batch_size, -1
        )
        self._bias_rnea_scratch = _InverseDynamicsScratchBuffer(
            body_count=self.bias_model.body_count,
            articulation_count=self.bias_model.articulation_count,
            joint_dof_count=self.bias_model.joint_dof_count,
            joint_target_q_count=self.bias_model.joint_target_q.shape[0],
            max_dofs_per_articulation=self.bias_model.max_dofs_per_articulation,
            max_joints_per_articulation=self.bias_model.max_joints_per_articulation,
            world_count=self.bias_model.world_count,
            device=self.bias_model.device,
        )
        self._bias_body_q = wp.to_torch(self.bias_state.body_q).reshape(
            batch_size, bias_source.body_count, 7
        )
        self._bias_body_com = wp.to_torch(self.bias_model.body_com).reshape(
            batch_size, bias_source.body_count, 3
        )
        self._bias_weights = tensor((0.0, *weights), torch.float32) / self.total_mass
        if bias_vstarts[probe_joint] != 0:
            raise ValueError("centroidal bias probe root must own the first six DOFs")
        self._streams: dict[int, Any] = {}
        wp.synchronize_device(self.model.device)

    def _validate_q(self, q: Any) -> Any:
        torch = self.torch
        if not isinstance(q, torch.Tensor):
            raise TypeError("q must be a torch.Tensor")
        if q.shape != (self.batch_size, self.robot_spec.configuration_dim):
            raise ValueError("q must have shape (batch_size, configuration_dim)")
        if q.device != self.device or q.dtype != torch.float32:
            raise ValueError("q must be float32 on the configured CUDA device")
        valid = torch.isfinite(q).all(dim=1)
        if self._root is not None:
            qi, _, _, _ = self._root
            norm = torch.linalg.vector_norm(q[:, qi + 3 : qi + 7], dim=-1)
            valid = valid & ((norm - 1.0).abs() < 1e-4)
        return valid

    def _evaluate_kinematics(self, q: Any) -> tuple[Any, Any, Any, Any, Any]:
        torch, wp = self.torch, self.wp
        valid = self._validate_q(q)
        self._q.index_copy_(1, self._qmap, q.index_select(1, self._qsource))
        pointer = int(torch.cuda.current_stream(self.device).cuda_stream)
        if pointer not in self._streams:
            self._streams[pointer] = wp.Stream(self.model.device, cuda_stream=pointer)
        # Both libraries enqueue on the same stream. Warp's default implicit
        # wait on its previous stream is unnecessary and breaks parent capture.
        with wp.ScopedStream(self._streams[pointer], sync_enter=False):
            self.newton.eval_fk(self.model, self._qw, self._qd, self.state)
            self.newton.eval_jacobian(
                self.model, self.state, J=self._J, joint_S_s=self._S
            )
        rotation = _quat_xyzw_to_matrix(torch, self._body_q[..., 3:])
        centers = self._body_q[..., :3] + (
            rotation @ self._body_com.unsqueeze(-1)
        ).squeeze(-1)
        position = (centers * self._weights[None, :, None]).sum(dim=1)
        jacobian = (
            self._jt[:, :, :3, :] * self._joint_weights[None, :, None, None]
        ).sum(dim=1)
        return valid, rotation, centers, position, jacobian

    def _active_tangent(self, value: Any, q: Any) -> Any:
        """Convert Newton columns to EmbodiK local-root active tangent order."""

        torch = self.torch
        if self._root is not None:
            # Newton: parent/world axes, pivot at root body CoM.
            # EmbodiK/Pinocchio: local root-origin linear/angular tangent.
            qi, _, nvi, body = self._root
            R = _quat_xyzw_to_matrix(torch, q[:, qi + 3 : qi + 7])
            offset = (R @ self._body_com[:, body, :, None]).squeeze(-1)
            linear = value[:, :, nvi : nvi + 3].clone()
            angular = value[:, :, nvi + 3 : nvi + 6].clone()
            value[:, :, nvi : nvi + 3] = linear @ R
            value[:, :, nvi + 3 : nvi + 6] = (
                angular - linear @ _skew(torch, offset)
            ) @ R
        return value.index_select(-1, self._vmap)

    def evaluate(self, q: Any) -> tuple[Any, Any]:
        """Return device-resident CoM position and active-tangent Jacobian."""

        torch = self.torch
        valid, _, _, position, jacobian = self._evaluate_kinematics(q)
        jacobian = self._active_tangent(jacobian, q)
        valid = (
            valid
            & torch.isfinite(position).all(dim=1)
            & torch.isfinite(jacobian).all(dim=(1, 2))
        )
        return (
            torch.where(valid[:, None], position, float("nan")),
            torch.where(valid[:, None, None], jacobian, float("nan")),
        )

    def evaluate_centroidal(self, q: Any) -> tuple[Any, Any, Any]:
        """Return CoM position/Jacobian and centroidal momentum matrix ``Ag``.

        The reduction uses each modeled body's Newton world-frame CoM
        Jacobian and body-frame rotational inertia. It performs the parallel
        axis shift about the system CoM directly on the CUDA device, then maps
        columns into EmbodiK's active local-root tangent convention.
        """

        torch = self.torch
        valid, rotation, centers, position, jacobian = self._evaluate_kinematics(q)
        body_rotation = rotation.index_select(1, self._joint_body_indices)
        body_inertia = self._body_inertia.index_select(
            1, self._joint_body_indices
        )
        world_inertia = (
            body_rotation @ body_inertia @ body_rotation.transpose(-1, -2)
        )
        linear_jacobian = self._jt[:, :, :3, :]
        angular_jacobian = self._jt[:, :, 3:, :]
        mass_linear_jacobian = (
            linear_jacobian * self._joint_masses[None, :, None, None]
        )
        linear_momentum = mass_linear_jacobian.sum(dim=1)
        body_centers = centers.index_select(1, self._joint_body_indices)
        offsets = body_centers - position[:, None, :]
        angular_momentum = (
            world_inertia @ angular_jacobian
            + _skew(torch, offsets) @ mass_linear_jacobian
        ) * self._joint_modeled_mask[None, :, None, None]
        angular_momentum = angular_momentum.sum(dim=1)
        ag = torch.cat((linear_momentum, angular_momentum), dim=1)
        jacobian = self._active_tangent(jacobian, q)
        ag = self._active_tangent(ag, q)
        valid = (
            valid
            & torch.isfinite(position).all(dim=1)
            & torch.isfinite(jacobian).all(dim=(1, 2))
            & torch.isfinite(ag).all(dim=(1, 2))
        )
        return (
            torch.where(valid[:, None], position, float("nan")),
            torch.where(valid[:, None, None], jacobian, float("nan")),
            torch.where(valid[:, None, None], ag, float("nan")),
        )

    def evaluate_centroidal_bias(self, q: Any, dq: Any) -> Any:
        """Return analytic device-resident ``dAg(q, dq) * dq`` as ``[B, 6]``.

        ``dq`` is the complete EmbodiK tangent vector, not only its active
        subset. The result follows EmbodiK's linear-then-angular world-frame
        convention and is referenced at the instantaneous system CoM.
        """

        torch, wp = self.torch, self.wp
        if not isinstance(dq, torch.Tensor):
            raise TypeError("dq must be a torch.Tensor")
        if dq.shape != (self.batch_size, self.robot_spec.velocity_dim):
            raise ValueError("dq must have shape (batch_size, velocity_dim)")
        if dq.device != self.device or dq.dtype != torch.float32:
            raise ValueError("dq must be float32 on the configured CUDA device")
        valid = self._validate_q(q) & torch.isfinite(dq).all(dim=1)
        self._bias_q.index_copy_(
            1, self._bias_qmap, q.index_select(1, self._bias_qsource)
        )
        self._bias_qd.zero_()
        self._bias_qd.index_copy_(1, self._bias_vmap, dq)
        if self._bias_inner_root is not None:
            qi, vi, nvi, body = self._bias_inner_root
            root_rotation = _quat_xyzw_to_matrix(torch, q[:, qi + 3 : qi + 7])
            local_linear = dq[:, vi : vi + 3]
            local_angular = dq[:, vi + 3 : vi + 6]
            local_com = self._bias_body_com[:, body]
            com_linear = local_linear + torch.cross(
                local_angular, local_com, dim=-1
            )
            self._bias_qd[:, nvi : nvi + 3] = (
                root_rotation @ com_linear[..., None]
            ).squeeze(-1)
            self._bias_qd[:, nvi + 3 : nvi + 6] = (
                root_rotation @ local_angular[..., None]
            ).squeeze(-1)
        pointer = int(torch.cuda.current_stream(self.device).cuda_stream)
        if pointer not in self._streams:
            self._streams[pointer] = wp.Stream(self.model.device, cuda_stream=pointer)
        with wp.ScopedStream(self._streams[pointer], sync_enter=False):
            self.newton.eval_fk(
                self.bias_model,
                self.bias_state.joint_q,
                self.bias_state.joint_qd,
                self.bias_state,
            )
            self._compute_coriolis_force(
                self.bias_model,
                self.bias_state,
                self._bias_coriolis_wp,
                self._bias_rnea_scratch,
            )
        body_rotation = _quat_xyzw_to_matrix(torch, self._bias_body_q[..., 3:])
        body_centers = self._bias_body_q[..., :3] + (
            body_rotation @ self._bias_body_com.unsqueeze(-1)
        ).squeeze(-1)
        position = (body_centers * self._bias_weights[None, :, None]).sum(dim=1)
        force = self._bias_coriolis[:, :3]
        moment_at_world_origin = self._bias_coriolis[:, 3:6]
        moment_at_com = moment_at_world_origin - torch.cross(
            position, force, dim=-1
        )
        bias = torch.cat((force, moment_at_com), dim=-1)
        valid = valid & torch.isfinite(position).all(dim=1) & torch.isfinite(bias).all(
            dim=1
        )
        return torch.where(valid[:, None], bias, float("nan"))
