"""Spot's CPU mode orchestration over the generalized GPU frame solver."""

import numpy as np

from embodik.gpu.wbc import GpuWbcFloatingMultiFrameSolver
from .spot_whole_body_ik import _ARM_TORSO_POSE_AXIS_MASK, SpotFullBodyIKMode


class GpuSpotFullBodyIK:
    """Lazily allocate mode layouts; keep model dimensions and joint maps in metadata.

    Collision allocation, row capacity, and tuning remain fixed at launch.
    CPU SCALE_ELASTIC/MIN_ERROR policy selection is not represented by this adapter.
    """

    def __init__(self, args, backend, urdf_path, *, solver_factory=None):
        self.args, self.backend, self.urdf_path = args, backend, urdf_path
        self.factory = solver_factory or GpuWbcFloatingMultiFrameSolver
        self.cache = {}
        self.last_mode = None
        self.last_recovery_used = False
        self.last_recovery_keys = []
        # Allocate the model-derived pair set once so the public checkbox can
        # toggle constraints without rebuilding Newton or changing row shapes.
        self.pairs = tuple(backend._collision_include_pairs)
        self.collision_supported = bool(self.pairs)
        self.collision_enabled = bool(args.gpu_wbc_collision)
        # Reserve the public slider's full 0--100 mm range at construction.
        self.query_distance = max(
            0.11, float(backend.config.collision_min_distance) + 0.01
        )
        self.max_constraints = int(backend.config.collision_max_constraints)
        self.solver_backend = getattr(
            args, "gpu_wbc_solver_backend", "torch_srinv"
        )
        self.collision_contacts_per_world = int(
            getattr(args, "gpu_wbc_collision_contacts_per_world", 4096)
        )
        self.collision_triangle_pairs_per_world = int(
            getattr(args, "gpu_wbc_collision_triangle_pairs_per_world", 1024)
        )
        self.collision_graph_repair_iterations = int(
            getattr(args, "gpu_wbc_collision_graph_repair_iterations", 2)
        )
        self.certified_nominal_enabled = bool(
            self.solver_backend == "cusolver_srinv"
            and self.collision_graph_repair_iterations > 0
            and getattr(args, "gpu_wbc_certified_nominal_enabled", True)
        )
        self.collision_contact_sort_enabled = bool(
            getattr(args, "gpu_wbc_collision_contact_sort_enabled", True)
        )
        self.candidate_convex_certificate_enabled = bool(
            getattr(args, "gpu_wbc_candidate_convex_certificate_enabled", True)
        )
        self.current_convex_certificate_enabled = bool(
            getattr(args, "gpu_wbc_current_convex_certificate_enabled", True)
        )
        self.specialized_outputs_enabled = bool(
            getattr(args, "gpu_wbc_cusolver_specialized_outputs_enabled", True)
        )
        self.reuse_locked_primary_inverse_enabled = bool(
            getattr(args, "gpu_wbc_reuse_locked_primary_inverse_enabled", True)
        )
        self.device_two_stage_enabled = bool(
            self.certified_nominal_enabled
            and getattr(args, "gpu_wbc_device_two_stage_enabled", False)
        )
        self.parent_two_stage_graph_enabled = bool(
            self.device_two_stage_enabled
            and getattr(args, "gpu_wbc_parent_two_stage_graph_enabled", True)
        )
        self._parent_two_stage_state = None
        self._arm_configuration_indices = tuple(
            index
            for name in backend._arm_joint_map.values()
            for index in range(
                int(backend.robot.get_joint_config_index(name)),
                int(backend.robot.get_joint_config_index(name))
                + int(backend.robot.get_joint_config_size(name)),
            )
        )
        self._arm_configuration_indices_device = None
        self._torso_bias_translation_device = None

    def _rt(self, matrix):
        return self.backend.eik.Rt(R=matrix[:3, :3], t=matrix[:3, 3])

    def _pose_row(self, matrix, *, xyzw=False):
        quat = np.asarray(self.backend.eik.r2q(matrix[:3, :3], order="xyzs"))
        return tuple(np.r_[matrix[:3, 3], quat if xyzw else quat[[3, 0, 1, 2]]])

    def _runtime(self, key):
        b, c = self.backend, self.backend.config
        half = np.asarray(c.torso_pose_half_range)
        arm = key in (SpotFullBodyIKMode.ARM_TORSO, "arm_stage")
        secondary = key != SpotFullBodyIKMode.TORSO_ONLY
        feet = () if arm else b.foot_frames
        options = {
            "dt": float(c.dt),
            "iterations": int(c.max_steps),
            "position_gain": float(c.position_gain),
            "orientation_gain": float(c.orientation_gain),
            "frame_position_gains": (
                float(c.position_gain),
                *(float(c.foot_position_gain) for _ in feet),
            ),
            "frame_orientation_gains": (
                float(c.orientation_gain),
                *(0.0 for _ in feet),
            ),
            "adaptive_dt": bool(c.adaptive_dt),
            "adaptive_dt_max_scale": float(c.adaptive_dt_max_scale),
            "adaptive_dt_reference_distance": float(c.adaptive_dt_reference_distance),
            "collision_enabled": bool(
                self.collision_supported and self.collision_enabled
            ),
            "posture_target_configuration": tuple(b._posture_bias_q),
            "posture_gain": float(c.nullspace_gain),
            "torso_reference_pose_xyzw": self._pose_row(
                b._torso_bounds_reference_pose, xyzw=True
            ),
            "torso_lower_relative_limits": tuple(-half),
            "torso_upper_relative_limits": tuple(half),
            "torso_axis_mask": tuple(
                bool(x) for x in (_ARM_TORSO_POSE_AXIS_MASK if arm else np.ones(6))
            ),
        }
        if self.collision_supported:
            options["collision_min_distance_m"] = float(c.collision_min_distance)
        if secondary:
            options.update(
                secondary_frame_target_poses_wxyz=(self._pose_row(b._torso_bias_pose),),
                secondary_frame_position_gains=(float(c.position_gain),),
                secondary_frame_orientation_gains=(float(c.orientation_gain),),
                secondary_frame_weights=(1.0,),
            )
        return options

    def _get(self, key, q, targets, *, repair_iterations=None):
        b, c = self.backend, self.backend.config
        runtime = self._runtime(key)
        repairs = (
            self.collision_graph_repair_iterations
            if repair_iterations is None
            else int(repair_iterations)
        )
        cache_key = (key, repairs) if self.certified_nominal_enabled else key
        if cache_key not in self.cache:
            initial_collision_enabled = runtime.pop("collision_enabled")
            arm = key in (SpotFullBodyIKMode.ARM_TORSO, "arm_stage")
            torso = key == SpotFullBodyIKMode.TORSO_ONLY
            feet = () if arm else b.foot_frames
            locked = list(b._gripper_velocity_indices)
            if arm:
                locked.extend(b._leg_velocity_indices)
            if key == "arm_stage":
                locked.extend(b._base_velocity_indices)
            if torso:
                locked.extend(b._arm_velocity_indices)
            active = tuple(
                i
                for i in range(int(b.robot.nv))
                if i not in b._gripper_velocity_indices
            )
            # Locks use source velocity indices, not positions in the active vector.
            locked = tuple(sorted(set(locked).intersection(active)))
            solver = self.factory(
                self.args.gpu_wbc_manifest,
                self.urdf_path,
                self.args.gpu_wbc_cache_dir,
                robot=b.robot,
                robot_name="spot_tool_feet",
                solver_backend=self.solver_backend,
                cusolver_specialized_outputs_enabled=bool(
                    self.solver_backend == "cusolver_srinv"
                    and self.specialized_outputs_enabled
                ),
                cusolver_reuse_locked_primary_inverse_enabled=bool(
                    self.certified_nominal_enabled
                    and self.reuse_locked_primary_inverse_enabled
                    and repairs == 0
                ),
                frames=(b.body_frame if torso else b.tool_frame, *feet),
                frame_task_dimensions=(6, *(3 for _ in feet)),
                frame_contact_constraints=(False, *(True for _ in feet)),
                active_velocity_indices=active,
                default_configuration=b.q0,
                base_velocity_limits=tuple(
                    float(c.base_velocity_limit) for _ in b._base_velocity_indices
                ),
                max_linear_speed=float(c.max_linear_speed),
                max_angular_speed=float(c.max_angular_speed),
                collision_pairs=self.pairs,
                collision_query_distance_m=self.query_distance,
                collision_contacts_per_world=self.collision_contacts_per_world,
                collision_triangle_pairs_per_world=(
                    self.collision_triangle_pairs_per_world
                ),
                collision_contact_sort_enabled=self.collision_contact_sort_enabled,
                collision_clear_state_fast_path_enabled=bool(
                    self.certified_nominal_enabled
                    and repairs == 0
                ),
                collision_candidate_convex_certificate_enabled=bool(
                    self.certified_nominal_enabled
                    and self.candidate_convex_certificate_enabled
                    and repairs == 0
                ),
                collision_current_convex_certificate_enabled=bool(
                    self.certified_nominal_enabled
                    and self.current_convex_certificate_enabled
                    and repairs == 0
                ),
                torso_projection_noop_fast_path_enabled=bool(
                    self.certified_nominal_enabled
                    and repairs == 0
                    and key == SpotFullBodyIKMode.ARM_TORSO
                ),
                collision_debug_enabled=False,
                collision_max_constraints=self.max_constraints,
                collision_graph_repair_iterations=(
                    repairs
                ),
                torso_frame_name=b.body_frame,
                torso_velocity_limits=(float(c.base_velocity_limit),) * 6,
                torso_acceleration_limits=(float(c.base_acceleration_limit),) * 6,
                posture_velocity_indices=tuple(
                    b._stage1_posture_velocity_indices
                    if arm
                    else b._posture_velocity_indices
                ),
                posture_weights=tuple(
                    b._stage1_posture_weights if arm else b._full_posture_weights
                ),
                secondary_frame_names=() if torso else (b.body_frame,),
                secondary_frame_task_dimensions=() if torso else (6,),
                locked_velocity_indices=locked,
                **runtime,
            )
            solver.configure_runtime(
                collision_enabled=bool(initial_collision_enabled)
            )
            solver.warm_up(q, targets)
            self.cache[cache_key] = solver
        else:
            # The runtime API accepts either scalar first-frame or per-frame gains.
            runtime.pop("position_gain")
            runtime.pop("orientation_gain")
            self.cache[cache_key].configure_runtime(**runtime)
        return self.cache[cache_key]

    def reset_state(self):
        for solver in self.cache.values():
            solver.reset_state()
        if self._parent_two_stage_state is not None:
            self._parent_two_stage_state["arm_history_valid"].zero_()
            self._parent_two_stage_state["torso_history_valid"].zero_()
        self.last_mode = None

    def _stage(self, key, q, target, debug):
        feet = (
            ()
            if key in (SpotFullBodyIKMode.ARM_TORSO, "arm_stage")
            else self.backend._foot_anchor_poses
        )
        targets = (target, *(self._rt(anchor) for anchor in feet))
        if not self.certified_nominal_enabled:
            return self._get(key, q, targets).solve_step(
                q, targets, include_collision_debug=debug
            )
        nominal = self._get(key, q, targets, repair_iterations=0)
        recovery = self.cache.get((key, self.collision_graph_repair_iterations))
        if recovery is not None:
            self._copy_state(recovery, nominal)
        initial_state = self._snapshot_state(nominal)
        result = nominal.solve_step(q, targets, include_collision_debug=False)
        certified = self._is_certified(result, nominal, debug)
        if certified:
            if recovery is not None:
                self._copy_state(nominal, recovery)
            return result
        self.last_recovery_used = True
        self.last_recovery_keys.append(key)
        if recovery is None:
            recovery = self._get(
                key,
                q,
                targets,
                repair_iterations=self.collision_graph_repair_iterations,
            )
            self._restore_state(initial_state, recovery)
        recovered = recovery.solve_step(q, targets, include_collision_debug=debug)
        self._copy_state(recovery, nominal)
        return recovered

    @staticmethod
    def _snapshot_state(source):
        previous = (
            source._previous_velocity.clone()
            if hasattr(source._previous_velocity, "clone")
            else source._previous_velocity.copy()
        )
        last_target = (
            None
            if source._last_target is None
            else (
                source._last_target.clone()
                if hasattr(source._last_target, "clone")
                else source._last_target.copy()
            )
        )
        last_target_host = (
            None
            if source._last_target_host is None
            else source._last_target_host.copy()
        )
        return previous, last_target, last_target_host

    @staticmethod
    def _restore_state(snapshot, destination):
        previous, last_target, last_target_host = snapshot
        if hasattr(destination._previous_velocity, "copy_"):
            destination._previous_velocity.copy_(previous)
        else:
            np.copyto(destination._previous_velocity, previous)
        destination._last_target = last_target
        destination._last_target_host = last_target_host

    @staticmethod
    def _is_certified(result, nominal, debug):
        clear_state_graph = bool(nominal.collision_clear_state_fast_path_enabled)
        return bool(
            result.status in {"SUCCESS", "SAFE_STEP"}
            and (
                result.collision_clear_state_certified
                if clear_state_graph
                else True
            )
            and result.collision_step_accepted
            and not result.collision_overflow
            and (not debug or not getattr(result, "collision_active", False))
        )

    def _device_two_stage_graph(
        self,
        seed,
        arm_target,
        torso_targets,
        arm_nominal,
        torso_nominal,
    ):
        """Run both warmed nominal stages through one flattened CUDA graph."""

        torch = arm_nominal._torch
        signature = (
            id(arm_nominal._solver),
            id(torso_nominal._solver),
            arm_nominal._solver.config,
            torso_nominal._solver.config,
        )
        state = self._parent_two_stage_state
        if state is None or state["signature"] != signature:
            q = arm_nominal._q_tensor(seed)
            static_q = torch.empty_like(q)
            static_q.copy_(q)
            static_arm_target = torch.empty_like(arm_target)
            static_arm_target.copy_(arm_target)
            static_torso_target = torso_nominal._target_tensor(torso_targets)
            intermediate = torch.empty_like(static_q)
            indices = torch.as_tensor(
                self._arm_configuration_indices,
                dtype=torch.long,
                device=arm_nominal._solver.device,
            )
            torso_bias_translation = torch.as_tensor(
                self.backend._torso_bias_pose[:3, 3],
                dtype=torch.float32,
                device=arm_nominal._solver.device,
            )
            arm_last_target = torch.empty_like(static_arm_target)
            torso_last_target = torch.empty_like(static_torso_target)
            arm_history_valid = torch.zeros((), dtype=torch.bool, device=q.device)
            torso_history_valid = torch.zeros_like(arm_history_valid)
            if arm_nominal._last_target is not None:
                arm_last_target.copy_(arm_nominal._last_target)
                arm_history_valid.fill_(True)
            if torso_nominal._last_target is not None:
                torso_last_target.copy_(torso_nominal._last_target)
                torso_history_valid.fill_(True)
            zero_arm_previous = torch.zeros_like(arm_nominal._previous_velocity)
            zero_torso_previous = torch.zeros_like(torso_nominal._previous_velocity)

            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, capture_error_mode="thread_local"):
                arm_translation_changed = torch.any(
                    torch.abs(static_arm_target[..., :3] - arm_last_target[..., :3])
                    > 1e-7
                )
                arm_current_q = static_arm_target[..., 3:]
                arm_previous_q = arm_last_target[..., 3:]
                arm_rotation_changed = torch.any(
                    torch.minimum(
                        torch.amax(torch.abs(arm_current_q - arm_previous_q), dim=-1),
                        torch.amax(torch.abs(arm_current_q + arm_previous_q), dim=-1),
                    )
                    > 1e-7
                )
                arm_changed = (
                    ~arm_history_valid
                    | arm_translation_changed
                    | arm_rotation_changed
                )
                arm_previous = torch.where(
                    arm_changed,
                    zero_arm_previous,
                    arm_nominal._previous_velocity,
                )
                first_result = arm_nominal._solver._solve_impl(
                    static_q, static_arm_target, arm_previous
                )
                arm_nominal._previous_velocity.copy_(first_result.accepted_velocity)
                arm_last_target.copy_(static_arm_target)
                arm_history_valid.fill_(True)

                intermediate.copy_(static_q)
                intermediate.index_copy_(
                    1,
                    indices,
                    first_result.q_solution.index_select(1, indices),
                )
                static_torso_target[:, 0, :3].copy_(
                    torso_bias_translation
                    + static_arm_target[:, 0, :3]
                    - first_result.solved_primary_pose_xyzw[:, 0, :3]
                )
                torso_translation_changed = torch.any(
                    torch.abs(
                        static_torso_target[..., :3] - torso_last_target[..., :3]
                    )
                    > 1e-7
                )
                torso_current_q = static_torso_target[..., 3:]
                torso_previous_q = torso_last_target[..., 3:]
                torso_rotation_changed = torch.any(
                    torch.minimum(
                        torch.amax(
                            torch.abs(torso_current_q - torso_previous_q), dim=-1
                        ),
                        torch.amax(
                            torch.abs(torso_current_q + torso_previous_q), dim=-1
                        ),
                    )
                    > 1e-7
                )
                torso_changed = (
                    ~torso_history_valid
                    | torso_translation_changed
                    | torso_rotation_changed
                )
                torso_previous = torch.where(
                    torso_changed,
                    zero_torso_previous,
                    torso_nominal._previous_velocity,
                )
                second_result = torso_nominal._solver._solve_impl(
                    intermediate, static_torso_target, torso_previous
                )
                torso_nominal._previous_velocity.copy_(second_result.accepted_velocity)
                torso_last_target.copy_(static_torso_target)
                torso_history_valid.fill_(True)
                first_compact = arm_nominal._solver.compact_publication(first_result)[0]
                second_compact = torso_nominal._solver.compact_publication(
                    second_result
                )[0]
                first_moved = torch.any(
                    torch.abs(first_result.q_solution - static_q) > 1e-7
                ).to(first_compact.dtype)
                second_moved = torch.any(
                    torch.abs(second_result.q_solution - intermediate) > 1e-7
                ).to(second_compact.dtype)
                combined = torch.cat(
                    (
                        first_compact,
                        first_moved.reshape(1),
                        second_compact,
                        second_moved.reshape(1),
                    )
                )
            state = {
                "signature": signature,
                "graph": graph,
                "q": static_q,
                "arm_target": static_arm_target,
                "torso_target": static_torso_target,
                "intermediate": intermediate,
                "arm_indices": indices,
                "torso_bias_translation": torso_bias_translation,
                "zero_arm_previous": zero_arm_previous,
                "zero_torso_previous": zero_torso_previous,
                "arm_last_target": arm_last_target,
                "torso_last_target": torso_last_target,
                "arm_history_valid": arm_history_valid,
                "torso_history_valid": torso_history_valid,
                "first_result": first_result,
                "second_result": second_result,
                "combined": combined,
                "first_width": int(first_compact.shape[0]),
                "second_width": int(second_compact.shape[0]),
                "start_event": start_event,
                "end_event": end_event,
            }
            self._parent_two_stage_state = state
        else:
            for adapter, target_key, valid_key in (
                (arm_nominal, "arm_last_target", "arm_history_valid"),
                (torso_nominal, "torso_last_target", "torso_history_valid"),
            ):
                if adapter._last_target is None:
                    state[valid_key].zero_()
                elif adapter._last_target is not state[target_key]:
                    state[target_key].copy_(adapter._last_target)
                    state[valid_key].fill_(True)
            state["q"].copy_(arm_nominal._q_tensor(seed))
            state["arm_target"].copy_(arm_target)

        state["start_event"].record()
        state["graph"].replay()
        state["end_event"].record()

        host = state["combined"].detach().cpu().numpy().copy()
        first_width = state["first_width"]
        second_width = state["second_width"]
        first_host = host[:first_width]
        first_moved = bool(host[first_width])
        second_start = first_width + 1
        second_host = host[second_start : second_start + second_width]
        second_moved = bool(host[second_start + second_width])
        transaction_elapsed = float(
            state["start_event"].elapsed_time(state["end_event"])
        )
        arm_nominal._last_target = state["arm_last_target"]
        arm_nominal._last_target_host = None
        torso_nominal._last_target = state["torso_last_target"]
        torso_nominal._last_target_host = None
        return (
            arm_nominal._decode_compact_device_step(
                state["first_result"],
                first_host,
                moved=first_moved,
                # The flattened graph exposes one timing boundary. Attribute the
                # transaction total to the final stage so summing stages remains
                # exact without pretending to measure an unavailable split.
                elapsed_ms=0.0,
            ),
            torso_nominal._decode_compact_device_step(
                state["second_result"],
                second_host,
                moved=second_moved,
                elapsed_ms=transaction_elapsed,
            ),
        )

    def _device_two_stage(self, seed, target_pose):
        """Speculate both nominal stages on device and publish them together."""

        b = self.backend
        arm_targets = (target_pose,)
        torso_placeholder = self._rt(b._torso_bias_pose)
        torso_targets = (
            torso_placeholder,
            *(self._rt(anchor) for anchor in b._foot_anchor_poses),
        )
        arm_nominal = self._get("arm_stage", seed, arm_targets, repair_iterations=0)
        torso_nominal = self._get(
            SpotFullBodyIKMode.TORSO_ONLY,
            seed,
            torso_targets,
            repair_iterations=0,
        )
        arm_recovery = self.cache.get(
            ("arm_stage", self.collision_graph_repair_iterations)
        )
        torso_recovery = self.cache.get(
            (SpotFullBodyIKMode.TORSO_ONLY, self.collision_graph_repair_iterations)
        )
        if arm_recovery is not None:
            self._copy_state(arm_recovery, arm_nominal)
        if torso_recovery is not None:
            self._copy_state(torso_recovery, torso_nominal)
        arm_initial_state = self._snapshot_state(arm_nominal)
        torso_initial_state = self._snapshot_state(torso_nominal)

        arm_target = arm_nominal._target_tensor(arm_targets)
        if self.parent_two_stage_graph_enabled:
            first, second = self._device_two_stage_graph(
                seed,
                arm_target,
                torso_targets,
                arm_nominal,
                torso_nominal,
            )
        else:
            q = arm_nominal._q_tensor(seed)
            first_pending = arm_nominal.solve_device_step(q, arm_target)
            solved_tool_pose = first_pending.result.solved_primary_pose_xyzw
            if solved_tool_pose is None:
                solved_tool_pose = (
                    arm_nominal._solver.kinematics._evaluate_pose_trusted(
                        first_pending.result.q_solution
                    )
                )
            if solved_tool_pose.ndim == 3:
                solved_tool_pose = solved_tool_pose[:, 0]

            torch = arm_nominal._torch
            if self._arm_configuration_indices_device is None:
                self._arm_configuration_indices_device = torch.as_tensor(
                    self._arm_configuration_indices,
                    dtype=torch.long,
                    device=arm_nominal._solver.device,
                )
            indices = self._arm_configuration_indices_device
            intermediate = q.clone()
            intermediate.index_copy_(
                1,
                indices,
                first_pending.result.q_solution.index_select(1, indices),
            )
            torso_target = torso_nominal._target_tensor(torso_targets).clone()
            if self._torso_bias_translation_device is None:
                self._torso_bias_translation_device = torch.as_tensor(
                    b._torso_bias_pose[:3, 3],
                    dtype=torch.float32,
                    device=arm_nominal._solver.device,
                )
            torso_bias_translation = self._torso_bias_translation_device
            torso_target[:, 0, :3] = (
                torso_bias_translation
                + arm_target[:, 0, :3]
                - solved_tool_pose[:, :3]
            )
            second_pending = torso_nominal.solve_device_step(intermediate, torso_target)
            first, second = arm_nominal.publish_device_steps(
                (first_pending, second_pending)
            )
        first_ok = self._is_certified(first, arm_nominal, False)
        second_ok = self._is_certified(second, torso_nominal, False)
        if first_ok and second_ok:
            if arm_recovery is not None:
                self._copy_state(arm_nominal, arm_recovery)
            if torso_recovery is not None:
                self._copy_state(torso_nominal, torso_recovery)
            return first, second

        self.last_recovery_used = True
        if first_ok:
            if arm_recovery is not None:
                self._copy_state(arm_nominal, arm_recovery)
            recovered_first = first
        else:
            self.last_recovery_keys.append("arm_stage")
            if arm_recovery is None:
                arm_recovery = self._get(
                    "arm_stage",
                    seed,
                    arm_targets,
                    repair_iterations=self.collision_graph_repair_iterations,
                )
                self._restore_state(arm_initial_state, arm_recovery)
            recovered_first = arm_recovery.solve_step(seed, arm_targets)
            self._copy_state(arm_recovery, arm_nominal)

        b.robot.update_configuration(np.asarray(recovered_first.joints))
        try:
            tool = b._pose_homogeneous(b.tool_frame)
            torso = b._torso_bias_pose.copy()
            torso[:3, 3] += b._as_homogeneous(target_pose)[:3, 3] - tool[:3, 3]
            recovered_intermediate = seed.copy()
            recovered_intermediate[list(self._arm_configuration_indices)] = np.asarray(
                recovered_first.joints
            )[list(self._arm_configuration_indices)]
            self.last_recovery_keys.append(SpotFullBodyIKMode.TORSO_ONLY)
            recovered_targets = (
                self._rt(torso),
                *(self._rt(anchor) for anchor in b._foot_anchor_poses),
            )
            if torso_recovery is None:
                torso_recovery = self._get(
                    SpotFullBodyIKMode.TORSO_ONLY,
                    recovered_intermediate,
                    recovered_targets,
                    repair_iterations=self.collision_graph_repair_iterations,
                )
                self._restore_state(torso_initial_state, torso_recovery)
            recovered_second = torso_recovery.solve_step(
                recovered_intermediate,
                recovered_targets,
            )
            self._copy_state(torso_recovery, torso_nominal)
        finally:
            b.robot.update_configuration(seed)
        return recovered_first, recovered_second

    @staticmethod
    def _copy_state(source, destination):
        """Mirror persistent acceleration and target history without host reads."""

        if hasattr(destination._previous_velocity, "copy_"):
            destination._previous_velocity.copy_(source._previous_velocity)
        else:
            np.copyto(destination._previous_velocity, source._previous_velocity)
        destination._last_target = (
            None
            if source._last_target is None
            else (
                source._last_target.clone()
                if hasattr(source._last_target, "clone")
                else source._last_target.copy()
            )
        )
        destination._last_target_host = (
            None
            if source._last_target_host is None
            else source._last_target_host.copy()
        )

    def solve(
        self,
        mode,
        target_pose,
        torso_target_pose=None,
        *,
        include_collision_debug=False,
    ):
        b = self.backend
        self.last_recovery_used = False
        self.last_recovery_keys = []
        mode = SpotFullBodyIKMode(mode)
        if mode != self.last_mode:
            self.reset_state()
            self.last_mode = mode
        seed = np.asarray(b.q, dtype=float).copy()
        if mode == SpotFullBodyIKMode.TWO_STAGE:
            if (
                self.device_two_stage_enabled
                and not include_collision_debug
                and hasattr(self._get("arm_stage", seed, (target_pose,)), "solve_device_step")
            ):
                result = self._device_two_stage(seed, target_pose)
            else:
                first = self._stage(
                    "arm_stage", seed, target_pose, include_collision_debug
                )
                b.robot.update_configuration(np.asarray(first.joints))
                try:
                    tool = b._pose_homogeneous(b.tool_frame)
                    torso = b._torso_bias_pose.copy()
                    torso[:3, 3] += (
                        b._as_homogeneous(target_pose)[:3, 3] - tool[:3, 3]
                    )
                    intermediate = seed.copy()
                    intermediate[list(self._arm_configuration_indices)] = np.asarray(
                        first.joints
                    )[list(self._arm_configuration_indices)]
                    second = self._stage(
                        SpotFullBodyIKMode.TORSO_ONLY,
                        intermediate,
                        self._rt(torso),
                        include_collision_debug,
                    )
                finally:
                    b.robot.update_configuration(seed)
                result = (first, second)
        else:
            target = (
                torso_target_pose
                if mode == SpotFullBodyIKMode.TORSO_ONLY
                and torso_target_pose is not None
                else target_pose
            )
            result = self._stage(mode, seed, target, include_collision_debug)
        b.q = np.asarray(
            (result[-1] if isinstance(result, tuple) else result).joints, dtype=float
        ).copy()
        b.robot.update_configuration(b.q)
        return result
