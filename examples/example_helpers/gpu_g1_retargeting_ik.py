"""Fail-closed G1 orchestration over the generalized floating GPU WBC solver.

Example 07 uses soft foot pose targets, not fixed contacts. Callers requiring
stationary feet can explicitly supply contact keys; moving such anchors holds.
The GPU hierarchy follows primary pose, torso upright, then posture, preserving
the undamped nullspaces of both preceding bands.
"""

from dataclasses import dataclass

import numpy as np

from .g1_model_utils import (
    g1_collision_pair_preset_options,
    g1_collision_pairs_for_preset,
)
from embodik.gpu.wbc import GpuWbcFloatingMultiFrameSolver


class _G1FloatingSolver(GpuWbcFloatingMultiFrameSolver):
    def __init__(self, *args, **kwargs):
        posture_priority = kwargs.pop("posture_priority", 1)
        super().__init__(*args, **kwargs)
        self._solver.configure_task_hierarchy(
            posture_priority=posture_priority,
            secondary_frame_orientation_only=(True,)
            * len(kwargs.get("secondary_frame_names", ())),
        )


@dataclass
class G1GpuStep:
    joints: np.ndarray
    status: str
    accepted: bool = False
    gpu_result: object = None


class GpuG1RetargetingIK:
    """One live layout, rebuilt on structural changes; no CPU solve fallback."""

    def __init__(self, args, robot, urdf_path, frame_map, q, *, solver_factory=None):
        self.args, self.robot, self.urdf_path = args, robot, urdf_path
        self.factory = solver_factory or _G1FloatingSolver
        self.frames = {
            key: frame_map[key]
            for key in ("right_palm", "left_palm", "right_ankle", "left_ankle")
        }
        self.frames["pelvis"] = (
            "pelvis"
            if "pelvis" in robot.get_frame_names()
            else frame_map["imu_in_torso"]
        )
        self.upright_frame = frame_map["imu_in_torso"]
        self.q0 = np.asarray(q, dtype=float).copy()
        self.posture_target = self.q0.copy()
        self.active = tuple(range(int(robot.nv)))
        # Identify the free joint from its manifold dimensions, not a G1 DOF count.
        roots, posture = [], []
        for name in robot.get_joint_names():
            size = int(robot.get_joint_velocity_size(name))
            start = int(robot.get_joint_velocity_index(name))
            indices = list(range(start, start + size))
            if int(robot.get_joint_config_size(name)) != size and size:
                roots.extend(indices)
            else:
                posture.extend(indices)
        if not roots or sorted(roots + posture) != list(self.active):
            raise ValueError("G1 GPU requires a complete floating joint metadata map")
        self.base_indices, self.posture_indices = tuple(roots), tuple(posture)
        self.pairs = {}
        self.solver = None
        self.key = None
        self.failed_key = None
        self.failure = ""
        self.last_result = None
        self.contact_anchors = {}

    def reset_state(self, q=None):
        if self.solver is not None:
            self.solver.reset_state()
        if q is not None:
            self.posture_target = np.asarray(q, dtype=float).copy()
            self.failed_key = None
        self.contact_anchors.clear()
        self.last_result = None

    def _hold(self, q, reason):
        if self.solver is not None:
            self.solver.reset_state()
        self.last_result = None
        return G1GpuStep(np.asarray(q, dtype=float).copy(), "SAFE_HOLD: " + reason)

    def solve(
        self,
        q,
        targets,
        *,
        gains=None,
        upright_target=None,
        upright_gain=2.0,
        posture_weight=0.0,
        adaptive_dt=True,
        iterations=1,
        dt=0.01,
        collision_enabled=False,
        collision_preset="core",
        collision_min_distance=0.02,
        collision_max_rows=3,
        include_collision_debug=False,
        com_enabled=False,
        com_support_polygon=None,
        com_margin=0.0,
        com_vel_max=0.3,
        com_acc_max=0.3,
        com_use_acceleration_limits=True,
        com_proximity_fraction=0.05,
        solve_policy="GPU_NATIVE",
        quality_recovery=False,
        contact_keys=(),
    ):
        """Return the seed on unsupported semantics, failed allocation or rejected steps."""
        import embodik

        self.last_result = None
        if com_enabled and com_support_polygon is None:
            return self._hold(q, "CoM requires a finite support polygon")
        if solve_policy != "GPU_NATIVE" or quality_recovery:
            return self._hold(
                q, "CPU solve policies / quality recovery are unsupported"
            )
        upright = upright_target is not None and upright_gain > 0
        keys = tuple(key for key in self.frames if key in targets)
        contacts = tuple(contact_keys)
        if not keys or any(key not in keys or "ankle" not in key for key in contacts):
            return self._hold(q, "invalid primary tasks / fixed foot contact keys")
        if collision_preset not in g1_collision_pair_preset_options():
            return self._hold(q, "unknown collision preset")
        if not (0 <= collision_min_distance <= 0.12 and 1 <= collision_max_rows <= 16):
            return self._hold(
                q, "collision distance / row capacity outside allocated range"
            )
        for key in contacts:
            target = np.asarray(targets[key])
            if key in self.contact_anchors and not np.allclose(
                target, self.contact_anchors[key], atol=1e-9, rtol=0
            ):
                return self._hold(
                    q, "fixed foot target moved; reset anchors explicitly"
                )
        gains = gains or {key: (16.0, 8.0) for key in keys}
        rt_targets = tuple(
            embodik.Rt(R=targets[k][:3, :3], t=targets[k][:3, 3]) for k in keys
        )
        layout = (
            keys,
            contacts,
            upright,
            posture_weight > 0,
            collision_preset,
            collision_max_rows,
            com_support_polygon is not None,
        )
        runtime = dict(
            dt=float(dt),
            iterations=int(iterations),
            frame_position_gains=tuple(float(gains[k][0]) for k in keys),
            frame_orientation_gains=tuple(float(gains[k][1]) for k in keys),
            adaptive_dt=bool(adaptive_dt),
            adaptive_dt_max_scale=3.0,
            adaptive_dt_reference_distance=0.04,
        )
        if posture_weight > 0:
            runtime.update(
                posture_target_configuration=tuple(self.posture_target),
                posture_weights=(float(posture_weight),) * len(self.posture_indices),
                posture_gain=1.0,
            )
        if com_support_polygon is not None:
            runtime.update(
                com_enabled=bool(com_enabled),
                com_support_polygon_xy=np.asarray(com_support_polygon, dtype=float),
                com_margin=float(com_margin),
                com_vel_max=float(com_vel_max),
                com_acc_max=float(com_acc_max),
                com_use_acceleration_limits=bool(com_use_acceleration_limits),
                com_proximity_fraction=float(com_proximity_fraction),
            )
        if upright:
            quat = np.asarray(embodik.r2q(upright_target[:3, :3], order="xyzs"))
            runtime.update(
                secondary_frame_target_poses_wxyz=(
                    tuple(np.r_[upright_target[:3, 3], quat[[3, 0, 1, 2]]]),
                ),
                # Shared allocation requires positive gains; the local angular-only
                # assembler never reads this placeholder or forms position rows.
                secondary_frame_position_gains=(1.0,),
                secondary_frame_orientation_gains=(float(upright_gain),),
                secondary_frame_weights=(1.0,),
            )
        try:
            if layout == self.failed_key:
                return self._hold(q, self.failure)
            if collision_preset not in self.pairs:
                self.pairs[collision_preset] = tuple(
                    g1_collision_pairs_for_preset(self.robot, collision_preset)
                )
            pairs = self.pairs[collision_preset]
            if collision_enabled and not pairs:
                return self._hold(q, "selected collision preset has no available pairs")
            if layout != self.key:
                construction_runtime = {
                    name: value
                    for name, value in runtime.items()
                    if not name.startswith("com_")
                }
                self.solver = None
                self.key = None
                self.solver = self.factory(
                    self.args.gpu_wbc_manifest,
                    self.urdf_path,
                    self.args.gpu_wbc_cache_dir,
                    robot=self.robot,
                    robot_name="unitree_g1",
                    frames=tuple(self.frames[k] for k in keys),
                    frame_task_dimensions=(6,) * len(keys),
                    frame_contact_constraints=tuple(k in contacts for k in keys),
                    active_velocity_indices=self.active,
                    default_configuration=self.q0,
                    base_velocity_limits=(1.8,) * len(self.base_indices),
                    max_linear_speed=1.8,
                    max_angular_speed=2.5,
                    collision_pairs=pairs,
                    collision_min_distance_m=float(collision_min_distance),
                    collision_query_distance_m=0.13,
                    collision_max_constraints=int(collision_max_rows),
                    collision_contacts_per_world=4096,
                    collision_triangle_pairs_per_world=1024,
                    com_support_polygon_xy=(
                        np.asarray(com_support_polygon, dtype=float)
                        if com_support_polygon is not None
                        else None
                    ),
                    com_max_constraints=8,
                    posture_velocity_indices=(
                        self.posture_indices if posture_weight > 0 else ()
                    ),
                    posture_priority=2,
                    secondary_frame_names=(self.upright_frame,) if upright else (),
                    # Configure angular-only rows in the generalized hierarchy.
                    secondary_frame_task_dimensions=(6,) if upright else (),
                    **construction_runtime,
                )
                self.solver.configure_runtime(
                    **runtime, collision_enabled=bool(collision_enabled)
                )
                self.solver.warm_up(q, rt_targets)
                self.key = layout
            else:
                self.solver.configure_runtime(
                    **runtime,
                    collision_enabled=bool(collision_enabled),
                    **(
                        {"collision_min_distance_m": float(collision_min_distance)}
                        if pairs
                        else {}
                    ),
                )
            result = self.solver.solve_step(
                q,
                rt_targets,
                include_collision_debug=bool(
                    include_collision_debug and collision_enabled
                ),
            )
            out = np.asarray(result.joints, dtype=float)
            accepted = (
                result.status in ("SUCCESS", "SAFE_STEP")
                and not result.collision_overflow
                and result.collision_step_accepted
                and result.torso_constraint_feasible
                and getattr(result, "com_constraint_feasible", True)
                and out.shape == np.asarray(q).shape
                and np.isfinite(out).all()
            )
            if not accepted:
                step = self._hold(q, result.status)
                self.last_result = result
                step.gpu_result = result
                return step
            self.contact_anchors = {k: np.asarray(targets[k]).copy() for k in contacts}
            self.last_result = result
            return G1GpuStep(out.copy(), result.status, True, result)
        except Exception as exc:
            self.failed_key, self.failure = (
                layout,
                f"GPU unavailable: {type(exc).__name__}: {exc}",
            )
            self.solver = None
            self.key = None
            return self._hold(q, self.failure)


def run_headless(args):
    """GPU-only two-hand displacement smoke; never dispatch a CPU IK harness."""
    from .g1_ik_runtime import _apply_g1_soft_knee_seed
    from .g1_model_utils import (
        create_g1_robot_model,
        resolve_frames_for_g1_base_mode,
        resolve_g1_collision_urdf_path,
    )

    robot = create_g1_robot_model(floating_base=True, reduced_ik=not args.full_ik_model)
    q = _apply_g1_soft_knee_seed(robot, robot.neutral_configuration())
    robot.update_configuration(q)
    frames = resolve_frames_for_g1_base_mode(robot.get_frame_names())
    gpu = GpuG1RetargetingIK(args, robot, resolve_g1_collision_urdf_path(), frames, q)
    targets = {
        key: np.asarray(robot.get_frame_pose(frame).homogeneous()).copy()
        for key, frame in gpu.frames.items()
    }
    for key in ("right_palm", "left_palm"):
        targets[key][0, 3] += 0.02
    seed = q.copy()
    for _ in range(args.headless_gpu_steps):
        step = gpu.solve(
            q,
            targets,
            collision_enabled=args.gpu_wbc_collision,
            collision_preset=args.gpu_wbc_collision_preset,
            quality_recovery=bool(args.quality_mode)
            and not bool(args.performance_mode),
        )
        if not step.accepted:
            raise RuntimeError(step.status)
        q = step.joints
    motion = float(np.linalg.norm(q - seed))
    if motion <= 1e-7:
        raise RuntimeError("GPU headless run did not move")
    print(
        f"G1 GPU: steps={args.headless_gpu_steps}, motion={motion:.6g}, {step.status}"
    )
