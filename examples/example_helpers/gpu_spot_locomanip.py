"""Synchronous GPU arm overlay for the policy-driven Spot example.

The policy owns the legs and base. They are locked in IK: moving them in the
virtual solution would not enforce those contacts in MuJoCo. Walking and contact
transitions use the current measured state, without commanding the base or legs.
Object dynamics, CoM control and asynchronous replay remain unsupported.
"""

from time import perf_counter

import numpy as np

from embodik.gpu.wbc import GpuWbcFloatingMultiFrameSolver
from .spot_whole_body_ik import (
    _LOCO_BASE_ACC_MAX,
    _LOCO_BASE_VEL_MAX,
    _LOCO_NULLSPACE_ARM_GAINS,
    _LOCO_NULLSPACE_GAIN,
    _LOCO_TORSO_POSE_HALF_RANGE,
    FOOT_FRAME_CANDIDATE_SETS,
    MJCF_ARM_JOINT_NAMES,
    OptionalSpotWholeBodyIK,
    SpotIKResult,
    _select_existing_frame,
)

_GPU_MAX_LINEAR_SPEED = 1.8
_GPU_MAX_ANGULAR_SPEED = 2.5


def stationary_contact_state(adapter, data, velocity_command, *, objects_enabled=False):
    """Observe current contacts (including swing feet); missing evidence fails closed.

    The historical function/key names are retained for callers. These observations
    do not assert stationary support or transfer locomotion ownership to IK.
    """
    mj, model = adapter._mujoco, adapter.model
    contacts_valid = all(
        np.isfinite(contact.dist) for contact in data.contact[: data.ncon]
    )
    supported = []
    for name in ("FL", "FR", "HL", "HR"):
        site = mj.mj_name2id(model, mj.mjtObj.mjOBJ_SITE, name)
        if site >= 0:
            body = int(model.site_bodyid[site])
        else:
            # MuJoCo Menagerie's Spot model names the four foot collision
            # geoms FL/FR/HL/HR but does not define same-named sites.
            geom = mj.mj_name2id(model, mj.mjtObj.mjOBJ_GEOM, name)
            body = int(model.geom_bodyid[geom]) if geom >= 0 else -1
        supported.append(
            np.nan
            if body < 0 or not contacts_valid
            else any(
                float(contact.dist) <= 0.001
                and {body, 0}
                == {
                    int(model.geom_bodyid[contact.geom1]),
                    int(model.geom_bodyid[contact.geom2]),
                }
                for contact in data.contact[: data.ncon]
            )
        )
    return {
        "gpu_fixed_contacts": np.asarray(supported, dtype=float),
        "gpu_velocity_command": np.asarray(velocity_command, dtype=float).copy(),
        "gpu_objects_enabled": np.asarray([objects_enabled], dtype=float),
    }


class GpuSpotLocomanipIK(OptionalSpotWholeBodyIK):
    """Reuse Spot model/command mapping, with no CPU IK solve or fallback.

    Torso bounds and a secondary torso reference remain in the GPU layout even
    though the physics-owned base/legs are locked. Posture acts on the arm's
    available nullspace. Collision tuning/capacity are fixed at construction.
    """

    def __init__(self, urdf_path, *, args, dt=0.01, solver_factory=None):
        super().__init__(urdf_path, dt=dt)
        if not self.enabled or self.robot is None:
            raise RuntimeError(f"GPU WBC requires a loaded Spot model: {self.message}")
        if args.gpu_wbc_cache_dir is None:
            raise ValueError("GPU WBC requires --gpu-wbc-cache-dir")
        self._gpu_args, self._gpu_urdf = args, urdf_path
        self._factory = solver_factory or GpuWbcFloatingMultiFrameSolver
        self._gpu = None
        self._gpu_debug = None
        self.include_collision_debug = False
        self._gpu_reference = None
        self._gpu_capacity = None
        self._gpu_tuning = None
        self.last_result = None
        self.foot_frames = tuple(
            _select_existing_frame(self.robot, names)
            for names in FOOT_FRAME_CANDIDATE_SETS
        )
        self.enable_collision = bool(getattr(args, "gpu_wbc_collision", False))
        self.target_solve_mode = "GPU_SRINV"
        self.message = "GPU ready: synchronous arm overlay during policy walking/contact transitions"

    def apply_runtime_options(self):
        # CPU option objects exist for model compatibility only.
        pass

    def reset_reference(self):
        self._gpu_reference = None
        self._gpu_debug = None
        self.last_result = None
        if self._gpu is not None:
            self._gpu.reset_state()

    @property
    def collision_debug_available(self):
        return bool(self.collision_include_pairs)

    def collision_debug_rows(self):
        return (
            [self._gpu_debug]
            if self.enable_collision
            and self.include_collision_debug
            and self._gpu_debug
            else []
        )

    def _hold(self, reason, *, result=None):
        self._gpu_debug = None
        self.last_result = result
        if self._gpu is not None:
            self._gpu.reset_state()
        return SpotIKResult(
            False, {}, status_name="SAFE_HOLD_GPU", message=f"GPU hold: {reason}"
        )

    def _pose_row(self, pose, *, xyzw=False):
        quat = np.asarray(self._eik.r2q(pose.rotation, order="xyzs"))
        return tuple(np.r_[pose.translation, quat if xyzw else quat[[3, 0, 1, 2]]])

    def solve_command(
        self,
        observation,
        arm_command,
        *,
        target_pose,
        body_command=None,
        desired_pose_command=None,
    ):
        started = perf_counter()
        if target_pose is None:
            return self._hold("enable IK and drag the target")
        contacts = np.asarray(observation.get("gpu_fixed_contacts", ()))
        if (
            contacts.shape != (len(self.foot_frames),)
            or not np.isin(contacts, (0, 1)).all()
        ):
            return self._hold(
                "measured contact observations are unavailable or invalid"
            )
        for key in ("gpu_velocity_command",):
            values = np.asarray(observation.get(key, ()), dtype=float)
            if values.shape != (3,) or not np.isfinite(values).all():
                return self._hold("locomotion command is unavailable or invalid")
        for key in ("base_lin_vel", "base_ang_vel"):
            values = np.asarray(observation.get(key, ()), dtype=float)
            if values.shape != (3,) or not np.isfinite(values).all():
                return self._hold("measured base velocity is unavailable")
        objects = np.asarray(observation.get("gpu_objects_enabled", ()))
        if objects.shape != (1,) or not np.all(objects == 0):
            return self._hold("scene object collision/dynamics are unsupported")
        if getattr(self, "enable_com", False) or observation.get(
            "gpu_com_requested", False
        ):
            return self._hold("CoM requests are unsupported")
        if self.target_solve_mode != "GPU_SRINV":
            return self._hold(
                "CPU target solve policy is unsupported; select GPU_SRINV"
            )
        for command in (body_command, desired_pose_command):
            if command is not None and (
                np.asarray(command).shape != (3,) or not np.isfinite(command).all()
            ):
                return self._hold("torso/locomotion command is invalid")
        for key, size in (("base_pose", 7), ("joint_pos", len(self._leg_joint_map))):
            values = np.asarray(observation.get(key, ()), dtype=float)
            if values.shape != (size,) or not np.isfinite(values).all():
                return self._hold(f"measured {key} is unavailable or invalid")
        if not np.isclose(np.linalg.norm(observation["base_pose"][3:]), 1.0, atol=1e-3):
            return self._hold("measured base orientation is invalid")
        arm_count = len(self._arm_joint_map)
        arm_state = np.asarray(observation.get("arm_state", ()), dtype=float)
        if arm_state.shape != (2 * arm_count,) or not np.isfinite(arm_state).all():
            return self._hold("measured arm state is unavailable")
        measured_arm = np.asarray(arm_command, dtype=float).copy()
        if (
            measured_arm.shape != (arm_count + bool(self._gripper_joint_name),)
            or not np.isfinite(measured_arm).all()
        ):
            return self._hold("arm command is invalid")
        measured_arm[:arm_count] = arm_state[:arm_count]
        gripper_state = np.asarray(observation.get("gripper_state", ()), dtype=float)
        if self._gripper_joint_name:
            if gripper_state.shape != (2,) or not np.isfinite(gripper_state).all():
                return self._hold("measured gripper state is unavailable")
            measured_arm[arm_count] = gripper_state[0]
        try:
            q = self._sync_configuration(
                observation, measured_arm, base_pose_override=None
            )
        except Exception as exc:
            return self._hold(f"measured configuration invalid: {exc}")
        measured_q = np.asarray(q).copy()
        if (
            measured_q.shape != (int(self.robot.nq),)
            or not np.isfinite(measured_q).all()
        ):
            return self._hold("measured configuration is invalid")
        seed = measured_q.copy()
        # MuJoCo's actuator can overshoot a URDF arm limit by a small amount.
        # Clamp the commanded/solved arm seed to the model contract; base and
        # leg coordinates remain the exact measured, physics-owned values.
        for name in self._arm_joint_map.values():
            start = int(self.robot.get_joint_config_index(name))
            size = int(self.robot.get_joint_config_size(name))
            stop = start + size
            seed[start:stop] = np.clip(
                seed[start:stop], self._lower[start:stop], self._upper[start:stop]
            )
        self.robot.update_configuration(seed)
        try:
            torso = self.robot.get_frame_pose(self.body_frame)
            feet = tuple(self.robot.get_frame_pose(frame) for frame in self.foot_frames)
            if self._gpu_reference is None:
                self._gpu_reference = seed.copy()
            posture = self._gpu_reference
            torso_ref, torso_target = self._pose_row(torso, xyzw=True), self._pose_row(
                torso
            )
            # All feet, including swing feet, are anchored only for this solve.
            # Exact non-arm locks preserve their measured pose without imposing
            # a stance/contact assumption on the locomotion policy.
            targets = (target_pose, *feet)
            runtime = dict(
                dt=float(self.dt),
                iterations=int(self.max_steps),
                frame_position_gains=(
                    float(self.position_gain),
                    *(float(self.position_gain) for _ in feet),
                ),
                # Three-row foot tasks do not assemble orientation rows, but
                # the shared fixed-layout validator still requires a positive
                # placeholder budget for every frame.
                frame_orientation_gains=(
                    float(self.orientation_gain),
                    *(1.0 for _ in feet),
                ),
                adaptive_dt=bool(self.adaptive_dt),
                adaptive_dt_max_scale=float(self.adaptive_dt_max_scale),
                adaptive_dt_reference_distance=float(
                    self.adaptive_dt_reference_distance
                ),
                collision_enabled=bool(self.enable_collision),
                torso_reference_pose_xyzw=torso_ref,
                posture_target_configuration=tuple(posture),
                secondary_frame_position_gains=(float(self.position_gain),),
                secondary_frame_orientation_gains=(float(self.orientation_gain),),
                secondary_frame_target_poses_wxyz=(torso_target,),
            )
            if self.collision_include_pairs:
                runtime["collision_min_distance_m"] = float(self.collision_min_distance)
            if self._gpu is None:
                gripper = set(self._velocity_indices((self._gripper_joint_name,)))
                active = tuple(i for i in range(int(self.robot.nv)) if i not in gripper)
                locked = tuple(i for i in active if i not in self._arm_velocity_indices)
                # Floating-root size comes from all model velocities minus the
                # mapped articulated joints, never from a robot-specific count.
                articulated = set(
                    self._velocity_indices(
                        tuple(self._arm_joint_map.values())
                        + tuple(self._leg_joint_map.values())
                        + (self._gripper_joint_name,)
                    )
                )
                base = tuple(
                    i for i in range(int(self.robot.nv)) if i not in articulated
                )
                initial_enabled = runtime.pop("collision_enabled")
                self._gpu = self._factory(
                    self._gpu_args.gpu_wbc_manifest,
                    self._gpu_urdf,
                    self._gpu_args.gpu_wbc_cache_dir,
                    robot=self.robot,
                    robot_name="spot_tool_feet",
                    frames=(self.tool_frame, *self.foot_frames),
                    frame_task_dimensions=(6, *(3 for _ in feet)),
                    frame_contact_constraints=(False, *(True for _ in feet)),
                    active_velocity_indices=active,
                    locked_velocity_indices=locked,
                    default_configuration=seed,
                    base_velocity_limits=tuple(_LOCO_BASE_VEL_MAX for _ in base),
                    # CPU zero means "no explicit task-space clamp". The GPU
                    # contract requires finite positive budgets, so use the
                    # same conservative limits as the public Spot full-body path.
                    max_linear_speed=_GPU_MAX_LINEAR_SPEED,
                    max_angular_speed=_GPU_MAX_ANGULAR_SPEED,
                    collision_pairs=tuple(self.collision_include_pairs),
                    collision_query_distance_m=0.11,
                    collision_contacts_per_world=4096,
                    collision_triangle_pairs_per_world=1024,
                    collision_max_constraints=int(self.collision_max_constraints),
                    torso_frame_name=self.body_frame,
                    torso_lower_relative_limits=tuple(-_LOCO_TORSO_POSE_HALF_RANGE),
                    torso_upper_relative_limits=tuple(_LOCO_TORSO_POSE_HALF_RANGE),
                    torso_axis_mask=(True,) * 6,
                    torso_velocity_limits=(_LOCO_BASE_VEL_MAX,) * 6,
                    torso_acceleration_limits=(_LOCO_BASE_ACC_MAX,) * 6,
                    posture_velocity_indices=tuple(self._arm_velocity_indices),
                    posture_weights=tuple(
                        dict(zip(MJCF_ARM_JOINT_NAMES, _LOCO_NULLSPACE_ARM_GAINS))[
                            mj_name
                        ]
                        for mj_name, joint in self._arm_joint_map.items()
                        for _ in range(int(self.robot.get_joint_velocity_size(joint)))
                    ),
                    posture_gain=_LOCO_NULLSPACE_GAIN,
                    secondary_frame_names=(self.body_frame,),
                    secondary_frame_task_dimensions=(6,),
                    secondary_frame_weights=(1.0,),
                    **runtime,
                )
                self._gpu_capacity = int(self.collision_max_constraints)
                self._gpu_tuning = self.collision_tuning_mode
                self._gpu.configure_runtime(collision_enabled=initial_enabled)
            else:
                if (
                    int(self.collision_max_constraints) != self._gpu_capacity
                    or self.collision_tuning_mode != self._gpu_tuning
                ):
                    return self._hold(
                        "collision capacity/tuning changed; restart required"
                    )
                self._gpu.configure_runtime(**runtime)
            result = self._gpu.solve_step(
                seed,
                targets,
                include_collision_debug=bool(self.include_collision_debug),
            )
            self.last_result = result
            candidate = np.asarray(result.joints)
            if (
                result.status not in ("SUCCESS", "SAFE_STEP")
                or candidate.shape != seed.shape
                or not np.isfinite(candidate).all()
                or result.collision_overflow
                or not result.collision_step_accepted
                or not result.torso_constraint_feasible
            ):
                return self._hold(
                    f"constraint rejection ({result.status})", result=result
                )
            # Only the arm is actuated by this overlay. Reject any virtual
            # base/leg/gripper displacement rather than discarding it silently.
            arm_config = {
                i
                for name in self._arm_joint_map.values()
                for i in range(
                    int(self.robot.get_joint_config_index(name)),
                    int(self.robot.get_joint_config_index(name))
                    + int(self.robot.get_joint_config_size(name)),
                )
            }
            fixed = [i for i in range(int(self.robot.nq)) if i not in arm_config]
            # Newton stores configurations in FP32. Accept only an unchanged
            # measured value or its exact FP32 round-trip, never a displacement
            # hidden by a geometric tolerance. Only arm commands are published.
            measured_fixed = measured_q[fixed]
            rounded_fixed = measured_fixed.astype(np.float32).astype(float)
            if not np.all(
                (candidate[fixed] == measured_fixed)
                | (candidate[fixed] == rounded_fixed)
            ):
                return self._hold("GPU moved physics-owned coordinates")
            output = np.asarray(arm_command).copy()
            for i, name in enumerate(self._arm_joint_map.values()):
                output[i] = self._read_joint_position(candidate, name)
            self._gpu_debug = result.collision_debug
            self._q = candidate.copy()
            return SpotIKResult(
                True,
                {},
                arm_command=output,
                status_name=result.status,
                position_error=result.position_errors[0],
                orientation_error=result.rotation_errors[0],
                solve_time_ms=(perf_counter() - started) * 1e3,
                message=f"GPU {result.status}: synchronous arm overlay; base/legs policy-owned",
            )
        except Exception as exc:
            return self._hold(f"{type(exc).__name__}: {exc}")
        finally:
            self.robot.update_configuration(measured_q)
