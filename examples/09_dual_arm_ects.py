"""ECTS Bimanual Coordination — Dual LBR iiwa Interactive Example

Demonstrates AbsoluteFrameTask and RelativeFrameTask using ECTS
(Extended Cooperative Task Space) for coordinated dual-arm manipulation with
two KUKA LBR iiwa 14 kg arms, matching the MATLAB GUI_example_LBRiiwa_extensions.m
reference implementation.

MATLAB-style layout:
- Left arm at Y = +0.3 m, right arm at Y = -0.3 m (no rotation)
- Default config: [0, 45, 0, -90, 0, 45, 0] deg (both arms identical)
- Gains: Kp = 10, Ko = 10 (matching MATLAB SNS-IK reference)

Interactive Viser controls:
* **Object marker** (blue)  — drives AbsoluteFrameTask (midpoint object pose)
* **Grasp marker** (green)  — drives RelativeFrameTask (grasp configuration)
* **Coordination Mode**     — 6 modes matching MATLAB dropdown
* **Absolute/Relative Tool Axes** — per-axis enable/disable (MATLAB S_cart1/S_cart2)
* **Constraint**            — Disable / Relative Bounds
* **IK Settings**           — dt, gains, step limits, damping, tolerance
* **Nullspace**             — gain (log scale) + secondary objective dropdown
* **Joint sliders**         — 7 left + 7 right arm joints
* **Snap / Reset** buttons  — via on_click callbacks (no polling)

Usage
-----
    cd examples/
    python 09_dual_arm_ects.py
    # Or: pixi run python3 examples/09_dual_arm_ects.py

Requires: robot_descriptions, viser, yourdfpy
"""

import tempfile
import time
import os

import numpy as np

try:
    import viser
except ImportError:
    print("This example requires viser. Install with: pip install viser")
    raise SystemExit(1)

import embodik
from embodik import Rt

from utils.dual_iiwa_urdf import (
    build_dual_iiwa_urdf,
    get_dual_iiwa_frame_names,
    get_dual_iiwa_default_configuration,
    get_dual_iiwa_joint_names,
)

LEFT_FRAME, RIGHT_FRAME = get_dual_iiwa_frame_names()

# Minimum clearance (m) enforced by the collision avoidance constraint
COLLISION_MIN_DISTANCE = 0.05

# 6 coordination modes matching MATLAB GUI dropdown
COORDINATION_MODES = [
    "Orthogonal",
    "Serial (L)",
    "Blended (0.75)",
    "Parallel",
    "Blended (0.25)",
    "Serial (R)",
]

CONSTRAINT_OPTIONS = ["disable", "relative_bounds"]

NULLSPACE_OBJECTIVES = [
    "Bias Pose",
    "Metric Optimization",
    "None",
]

# IK defaults matching MATLAB: Kp=10, Ko=10, ts=0.01
DEFAULT_SOLVER_DT = 0.01
DEFAULT_POS_GAIN = 10.0
DEFAULT_ROT_GAIN = 10.0
DEFAULT_NULLSPACE_GAIN_EXP = -2.0   # 10^-2 = 0.01
DEFAULT_DAMPING = 0.1
DEFAULT_TOLERANCE = 0.1
COLLISION_TUNING_OPTIONS = ("speed", "balanced", "precise")

ARM_JOINT_LABELS = [f"L{j}" for j in range(1, 8)] + [f"R{j}" for j in range(1, 8)]


def _apply_collision_tuning_mode(
    solver: "embodik.KinematicsSolver",
    mode_label: str,
) -> None:
    label = mode_label.lower()
    if hasattr(solver, "set_collision_tuning_mode") and hasattr(embodik, "CollisionTuningMode"):
        enum_map = {
            "precise": embodik.CollisionTuningMode.PRECISE,
            "balanced": embodik.CollisionTuningMode.BALANCED,
            "speed": embodik.CollisionTuningMode.SPEED,
        }
        solver.set_collision_tuning_mode(enum_map.get(label, embodik.CollisionTuningMode.SPEED))
        return

    # Backward-compatible fallback for older bindings.
    if label == "precise":
        if hasattr(solver, "enable_collision_pair_cache"):
            solver.enable_collision_pair_cache(False, 1, 0.0, 128)
        if hasattr(solver, "set_collision_refinement_time_budget_us"):
            solver.set_collision_refinement_time_budget_us(0)
    elif label == "balanced":
        if hasattr(solver, "enable_collision_pair_cache"):
            solver.enable_collision_pair_cache(True, 20, 0.05, 256)
        if hasattr(solver, "set_collision_refinement_time_budget_us"):
            solver.set_collision_refinement_time_budget_us(0)
    else:
        if hasattr(solver, "enable_collision_pair_cache"):
            solver.enable_collision_pair_cache(True, 100, 0.03, 128)
        if hasattr(solver, "set_collision_refinement_time_budget_us"):
            solver.set_collision_refinement_time_budget_us(300)


def _dual_iiwa_collision_exclusions(robot: "embodik.RobotModel") -> list:
    """Return collision pair exclusions for the dual LBR iiwa 14.

    Rules:
    - Same-arm pairs whose link-index gap is ≤ 3 are excluded (links that are
      always in contact / permanently adjacent), matching the iiwa rule from
      example 02.
    - Cross-arm pairs where both links have index 0 or 1 are excluded because
      those base links are rigidly attached to the shared base and can never
      separate.  All other cross-arm pairs are kept active so the solver will
      push the arms apart if they come within COLLISION_MIN_DISTANCE.

    Geometry names have the form ``{prefix}iiwa_link_{n}_{geom_idx}``.
    The link index n is the second-to-last numeric token.
    """
    import re as _re
    # Matches the link number in e.g. "iiwa_left_iiwa_link_3_0"
    _link_idx_pat = _re.compile(r"iiwa_link_(\d+)_\d+$")

    def _link_idx(name: str):
        m = _link_idx_pat.search(name)
        return int(m.group(1)) if m else None

    exclusions = []
    for a, b in robot.get_collision_pair_names():
        left_a = a.startswith("iiwa_left_")
        left_b = b.startswith("iiwa_left_")
        same_arm = left_a == left_b
        ia, ib = _link_idx(a), _link_idx(b)
        if same_arm:
            if ia is not None and ib is not None and abs(ia - ib) <= 3:
                exclusions.append((a, b))
        else:
            # Cross-arm: exclude only the rigidly mounted base links (0 & 1)
            if ia is not None and ib is not None and ia <= 1 and ib <= 1:
                exclusions.append((a, b))
    return exclusions


def _mat_from_wxyz(wxyz: np.ndarray) -> np.ndarray:
    """Convert wxyz quaternion to 3x3 rotation matrix."""
    from scipy.spatial.transform import Rotation as Rscipy
    xyzw = np.array([wxyz[1], wxyz[2], wxyz[3], wxyz[0]])
    return Rscipy.from_quat(xyzw).as_matrix()


def _wxyz_from_mat(R: np.ndarray) -> np.ndarray:
    """Convert 3x3 rotation matrix to wxyz quaternion."""
    from scipy.spatial.transform import Rotation as Rscipy
    xyzw = Rscipy.from_matrix(R).as_quat()
    return np.array([xyzw[3], xyzw[0], xyzw[1], xyzw[2]])


def _ects_config(mode_label: str) -> embodik.ECTSConfig:
    """Map MATLAB-style coordination mode label to ECTSConfig."""
    mapping = {
        "Orthogonal":      embodik.map_ects_mode("orthogonal"),
        "Serial (L)":      embodik.map_ects_mode("serial_left"),
        "Blended (0.75)":  embodik.map_ects_mode_blended(0.75),
        "Parallel":        embodik.map_ects_mode("parallel"),
        "Blended (0.25)":  embodik.map_ects_mode_blended(0.25),
        "Serial (R)":      embodik.map_ects_mode("serial_right"),
    }
    return mapping[mode_label]


def _effective_absolute_pose(robot, left_frame: str, right_frame: str, alpha: float):
    """Compute ECTS absolute frame pose from left/right EE poses and alpha.

    With set_object_center_frame, abs_task.current_position is the object center
    and does not change with alpha. For mode-change snap we need the effective
    frame: position = alpha*left + (1-alpha)*right, orientation = slerp.
    Returns (position, orientation) as numpy arrays.
    """
    from scipy.spatial.transform import Rotation as Rscipy

    left_se3 = robot.get_frame_pose(left_frame)
    right_se3 = robot.get_frame_pose(right_frame)
    left_pos = np.array(left_se3.translation)
    right_pos = np.array(right_se3.translation)
    left_R = np.array(left_se3.rotation)
    right_R = np.array(right_se3.rotation)

    pos = alpha * left_pos + (1.0 - alpha) * right_pos
    R_left = Rscipy.from_matrix(left_R)
    R_right = Rscipy.from_matrix(right_R)
    # Slerp: R_result = R_left * (R_left^{-1} * R_right)^{1-alpha} -> alpha=1 gives left, alpha=0 gives right
    r_rel = R_left.inv() * R_right
    r_part = Rscipy.from_rotvec((1.0 - alpha) * r_rel.as_rotvec())
    R_abs = R_left * r_part
    ori = R_abs.as_matrix()
    return pos, ori


def main():
    from scipy.spatial.transform import Rotation as Rscipy

    # Build and load dual iiwa robot
    dual_iiwa_urdf = build_dual_iiwa_urdf()
    fd, urdf_path = tempfile.mkstemp(suffix=".urdf")
    with os.fdopen(fd, "w") as f:
        f.write(dual_iiwa_urdf)

    try:
        robot = embodik.RobotModel(urdf_path)
    finally:
        os.unlink(urdf_path)

    q_init = np.array(get_dual_iiwa_default_configuration(), dtype=np.float64)
    q = q_init.copy()
    robot.update_configuration(q)

    solver = embodik.KinematicsSolver(robot)
    solver.dt = DEFAULT_SOLVER_DT
    solver.set_damping(DEFAULT_DAMPING)
    solver.set_tolerance(DEFAULT_TOLERANCE)
    _collision_tuning_mode = "speed"
    _apply_collision_tuning_mode(solver, _collision_tuning_mode)
    solver.enable_velocity_limits(True)
    solver.enable_position_limits(True)

    # Compute and apply collision exclusions (adjacent same-arm links + rigid bases)
    _collision_exclusions = _dual_iiwa_collision_exclusions(robot)
    if _collision_exclusions:
        robot.apply_collision_exclusions(_collision_exclusions)
    _collision_enabled = False  # toggled by GUI checkbox

    # ECTS tasks: absolute (midpoint) + relative (grasp configuration)
    abs_task = solver.add_absolute_frame_task("absolute", LEFT_FRAME, RIGHT_FRAME, 0.5)
    abs_task.weight = 1.0
    abs_task.priority = 0
    abs_task.solve_mode = embodik.TaskSolveMode.SCALE
    abs_task.allow_min_error_fallback = False

    rel_task = solver.add_relative_frame_task("relative", LEFT_FRAME, RIGHT_FRAME)
    rel_task.weight = 1.0
    rel_task.priority = 0
    rel_task.solve_mode = embodik.TaskSolveMode.SCALE
    rel_task.allow_min_error_fallback = False

    # Single-arm frame tasks for Orthogonal mode (two independent EE pose controls)
    left_ee_task = solver.add_frame_task("left_ee", LEFT_FRAME)
    left_ee_task.weight = 1.0
    left_ee_task.priority = 0
    left_ee_task.solve_mode = embodik.TaskSolveMode.SCALE
    left_ee_task.allow_min_error_fallback = False
    left_ee_task.active = False

    right_ee_task = solver.add_frame_task("right_ee", RIGHT_FRAME)
    right_ee_task.weight = 1.0
    right_ee_task.priority = 0
    right_ee_task.solve_mode = embodik.TaskSolveMode.SCALE
    right_ee_task.allow_min_error_fallback = False
    right_ee_task.active = False

    posture = solver.add_posture_task("posture")
    posture.set_target_configuration(q_init.copy())
    posture.weight = 10.0 ** DEFAULT_NULLSPACE_GAIN_EXP
    posture.priority = 1

    # Initialize tasks: set object center frame and capture relative target
    abs_task.update(robot)
    Tobj = np.eye(4)
    Tobj[:3, :3] = abs_task.current_orientation
    Tobj[:3, 3] = abs_task.current_position
    abs_task.set_object_center_frame(Tobj)

    abs_task.update(robot)
    rel_task.update(robot)
    rel_task.capture_current_as_target()

    init_abs_pos = abs_task.current_position.copy()
    init_abs_ori = abs_task.current_orientation.copy()

    # Initial relative target (relative frame: right EE relative to left EE in left frame)
    left_se3_init = robot.get_frame_pose(LEFT_FRAME)
    right_se3_init = robot.get_frame_pose(RIGHT_FRAME)
    init_rel_pos = rel_task.current_position.copy()
    init_rel_ori = rel_task.current_orientation.copy()

    # Viser server
    server = viser.ViserServer(port=8080, label="ECTS Dual-Arm IK")

    # URDF visualization
    fd2, urdf_path2 = tempfile.mkstemp(suffix=".urdf")
    with os.fdopen(fd2, "w") as f:
        f.write(dual_iiwa_urdf)

    try:
        import yourdfpy
        from viser.extras import ViserUrdf
        urdf_yourdfpy = yourdfpy.URDF.load(urdf_path2)
        urdf_vis = ViserUrdf(server, urdf_yourdfpy, root_node_name="/robot")
    finally:
        os.unlink(urdf_path2)

    # Apply initial configuration
    joint_names = robot.get_joint_names()
    cfg_dict = {name: float(q[i]) for i, name in enumerate(joint_names)}
    urdf_vis.update_cfg(cfg_dict)

    # --- Interactive markers ---
    # Object marker (absolute frame). Differentiate with a blue indicator sphere.
    quat_abs_init = _wxyz_from_mat(init_abs_ori)
    object_handle = server.scene.add_transform_controls(
        "/object_marker",
        scale=0.15,
        position=tuple(init_abs_pos),
        wxyz=tuple(quat_abs_init),
    )
    # Blue sphere = absolute (object) frame
    object_sphere = server.scene.add_icosphere(
        "/object_marker_sphere",
        radius=0.025,
        color=(0.2, 0.4, 1.0),  # blue
        position=tuple(init_abs_pos),
        wxyz=tuple(quat_abs_init),
    )

    # Grasp marker (relative frame shown in world at right arm EE). Green indicator.
    right_init_pos = right_se3_init.translation
    right_init_ori = np.array(right_se3_init.rotation)
    quat_grasp_init = _wxyz_from_mat(right_init_ori)
    grasp_handle = server.scene.add_transform_controls(
        "/grasp_marker",
        scale=0.12,
        position=tuple(right_init_pos),
        wxyz=tuple(quat_grasp_init),
    )
    # Green sphere = relative (grasp) frame
    grasp_sphere = server.scene.add_icosphere(
        "/grasp_marker_sphere",
        radius=0.02,
        color=(0.2, 0.85, 0.3),  # green
        position=tuple(right_init_pos),
        wxyz=tuple(quat_grasp_init),
    )

    # Joint limits (14 arm joints, no grippers)
    q_lower, q_upper = robot.get_joint_limits()
    left_joint_names, right_joint_names = get_dual_iiwa_joint_names()
    all_joint_names = robot.get_joint_names()
    arm_indices = [all_joint_names.index(n) for n in left_joint_names + right_joint_names]

    # --- GUI controls ---
    with server.gui.add_folder("ECTS Controls"):
        coord_mode = server.gui.add_dropdown(
            "Coordination Mode",
            options=COORDINATION_MODES,
            initial_value="Parallel",
        )

    with server.gui.add_folder("Absolute Tool Axes"):
        abs_pos_x = server.gui.add_checkbox("Abs Pos X", initial_value=True)
        abs_pos_y = server.gui.add_checkbox("Abs Pos Y", initial_value=True)
        abs_pos_z = server.gui.add_checkbox("Abs Pos Z", initial_value=True)
        abs_ori_x = server.gui.add_checkbox("Abs Ori X", initial_value=True)
        abs_ori_y = server.gui.add_checkbox("Abs Ori Y", initial_value=True)
        abs_ori_z = server.gui.add_checkbox("Abs Ori Z", initial_value=True)

    with server.gui.add_folder("Relative Tool Axes"):
        rel_pos_x = server.gui.add_checkbox("Rel Pos X", initial_value=True)
        rel_pos_y = server.gui.add_checkbox("Rel Pos Y", initial_value=True)
        rel_pos_z = server.gui.add_checkbox("Rel Pos Z", initial_value=True)
        rel_ori_x = server.gui.add_checkbox("Rel Ori X", initial_value=True)
        rel_ori_y = server.gui.add_checkbox("Rel Ori Y", initial_value=True)
        rel_ori_z = server.gui.add_checkbox("Rel Ori Z", initial_value=True)

    with server.gui.add_folder("Constraint"):
        constraint_mode = server.gui.add_dropdown(
            "Constraint",
            options=CONSTRAINT_OPTIONS,
            initial_value="disable",
        )
        constraint_bound = server.gui.add_slider(
            "Constraint Bound (mm)", min=1, max=50, step=1, initial_value=10
        )

    with server.gui.add_folder("IK Settings"):
        solver_dt_slider = server.gui.add_slider(
            "Solver dt (s)", min=0.001, max=0.05, step=0.001, initial_value=DEFAULT_SOLVER_DT
        )
        pos_gain_slider = server.gui.add_slider(
            "Position Gain (Kp)", min=0.5, max=50.0, step=0.5, initial_value=DEFAULT_POS_GAIN
        )
        rot_gain_slider = server.gui.add_slider(
            "Rotation Gain (Ko)", min=0.5, max=50.0, step=0.5, initial_value=DEFAULT_ROT_GAIN
        )
        iterations_slider = server.gui.add_slider(
            "IK Iterations", min=1, max=50, step=1, initial_value=1
        )
        ee_mode_dropdown = server.gui.add_dropdown(
            "EE Solve Mode",
            options=("SCALE", "SCALE_ELASTIC", "MIN_ERROR"),
            initial_value="SCALE",
        )
        ee_fallback_checkbox = server.gui.add_checkbox(
            "Allow SCALE fallback to MIN_ERROR",
            initial_value=False,
        )
        damping_slider = server.gui.add_slider(
            "Damping", min=0.01, max=1.0, step=0.01, initial_value=DEFAULT_DAMPING
        )
        tolerance_slider = server.gui.add_slider(
            "Tolerance", min=0.01, max=1.0, step=0.01, initial_value=DEFAULT_TOLERANCE
        )
        manual_control = server.gui.add_checkbox("Manual Joint Control", initial_value=False)

    with server.gui.add_folder("Nullspace"):
        nullspace_exp_slider = server.gui.add_slider(
            "Gain (10^n)", min=-4.0, max=2.0, step=0.1, initial_value=DEFAULT_NULLSPACE_GAIN_EXP
        )
        nullspace_objective = server.gui.add_dropdown(
            "Secondary Objective",
            options=NULLSPACE_OBJECTIVES,
            initial_value="Bias Pose",
        )

    joint_sliders: list = []
    with server.gui.add_folder("Joint Configuration (Arm Only)", expand_by_default=False):
        for i, idx in enumerate(arm_indices):
            lo, hi = float(q_lower[idx]), float(q_upper[idx])
            joint_sliders.append(
                server.gui.add_slider(
                    ARM_JOINT_LABELS[i],
                    min=lo,
                    max=hi,
                    step=0.01,
                    initial_value=float(q[idx]),
                )
            )

    with server.gui.add_folder("Collision Avoidance"):
        self_collision_checkbox = server.gui.add_checkbox(
            "Enable Self-Collision Avoidance", initial_value=False
        )
        collision_tuning_dropdown = server.gui.add_dropdown(
            "Tuning Mode",
            options=COLLISION_TUNING_OPTIONS,
            initial_value=_collision_tuning_mode,
        )
        collision_min_dist_slider = server.gui.add_slider(
            "Min Distance (mm)", min=1, max=100, step=1,
            initial_value=int(COLLISION_MIN_DISTANCE * 1000),
        )
        collision_debug_checkbox = server.gui.add_checkbox(
            "Show Collision Debug", initial_value=False
        )
        collision_debug_text = server.gui.add_text(
            "Closest Pair", initial_value="--"
        )

    with server.gui.add_folder("Actions"):
        snap_button = server.gui.add_button("Snap Object Marker to Current")
        reset_button = server.gui.add_button("Reset Robot & Target")

    with server.gui.add_folder("Status"):
        elapsed_text = server.gui.add_text("Elapsed (ms)", initial_value="0.00")
        status_text = server.gui.add_text("Solve Status", initial_value="--")
        abs_error_text = server.gui.add_text("Abs Error (m)", initial_value="0.000")
        rel_error_text = server.gui.add_text("Rel Error (m)", initial_value="0.000")

    # --- Collision debug scene objects ---
    _collision_root = "/collision_debug"
    _col_sphere_a = server.scene.add_icosphere(
        f"{_collision_root}/point_a", radius=0.015,
        color=(1.0, 0.2, 0.2), visible=False,
    )
    _col_sphere_b = server.scene.add_icosphere(
        f"{_collision_root}/point_b", radius=0.015,
        color=(0.2, 0.8, 0.2), visible=False,
    )
    _col_line_handle = None
    _last_collision_debug = None

    def _update_collision_visuals() -> None:
        nonlocal _col_line_handle, _last_collision_debug
        show = (
            collision_debug_checkbox.value
            and self_collision_checkbox.value
            and hasattr(solver, "get_last_collision_debug")
        )
        if not show:
            _col_sphere_a.visible = False
            _col_sphere_b.visible = False
            if _col_line_handle is not None:
                _col_line_handle.visible = False
            if not collision_debug_checkbox.value or not self_collision_checkbox.value:
                collision_debug_text.value = "--"
            return

        debug = solver.get_last_collision_debug()
        if debug is None:
            _col_sphere_a.visible = False
            _col_sphere_b.visible = False
            if _col_line_handle is not None:
                _col_line_handle.visible = False
            collision_debug_text.value = "No active pair"
            _last_collision_debug = None
            return

        pa = np.array(debug.point_a_world, dtype=float)
        pb = np.array(debug.point_b_world, dtype=float)
        _col_sphere_a.position = tuple(pa)
        _col_sphere_b.position = tuple(pb)
        _col_sphere_a.visible = True
        _col_sphere_b.visible = True

        if _col_line_handle is not None:
            _col_line_handle.remove()
        seg = np.zeros((1, 2, 3), dtype=float)
        seg[0, 0] = pa
        seg[0, 1] = pb
        colors = np.array([[[1.0, 0.2, 0.2], [0.2, 0.8, 0.2]]], dtype=float)
        _col_line_handle = server.scene.add_line_segments(
            f"{_collision_root}/segment", points=seg, colors=colors,
            line_width=3.0, visible=True,
        )
        collision_debug_text.value = (
            f"{debug.object_a} ↔ {debug.object_b} | d={debug.distance:.3f} m"
        )
        _last_collision_debug = debug

    def _apply_collision_toggle() -> None:
        nonlocal _collision_enabled
        _apply_collision_tuning_mode(solver, collision_tuning_dropdown.value)
        want = self_collision_checkbox.value
        if want and not _collision_enabled:
            try:
                solver.configure_collision_constraint(
                    min_distance=collision_min_dist_slider.value * 0.001,
                    include_pairs=[],
                    exclude_pairs=list(_collision_exclusions),
                )
                _collision_enabled = True
            except RuntimeError as exc:
                print(f"[collision] configure failed: {exc}")
                _collision_enabled = False
        elif not want and _collision_enabled:
            solver.clear_collision_constraint()
            _collision_enabled = False

    @self_collision_checkbox.on_update
    def _(_evt) -> None:
        _apply_collision_toggle()
        _update_collision_visuals()

    @collision_debug_checkbox.on_update
    def _(_evt) -> None:
        _update_collision_visuals()

    @collision_min_dist_slider.on_update
    def _(_evt) -> None:
        # Re-apply constraint with updated distance if currently enabled
        if _collision_enabled:
            try:
                _apply_collision_tuning_mode(solver, collision_tuning_dropdown.value)
                solver.configure_collision_constraint(
                    min_distance=collision_min_dist_slider.value * 0.001,
                    include_pairs=[],
                    exclude_pairs=list(_collision_exclusions),
                )
            except RuntimeError as exc:
                print(f"[collision] reconfigure failed: {exc}")

    @collision_tuning_dropdown.on_update
    def _(_evt) -> None:
        _apply_collision_tuning_mode(solver, collision_tuning_dropdown.value)
        if _collision_enabled:
            _apply_collision_toggle()

    # --- Button callbacks (fired once per click, no polling) ---
    _snap_requested = False
    _reset_requested = False

    @snap_button.on_click
    def _(_evt) -> None:
        nonlocal _snap_requested
        _snap_requested = True

    @reset_button.on_click
    def _(_evt) -> None:
        nonlocal _reset_requested
        _reset_requested = True

    prev_manual = False
    rel_error_vec = np.zeros(6)

    # Relative target stored in relative-frame coordinates (right-EE relative to left-EE
    # expressed in left-EE frame). This avoids recomputing from world space every loop,
    # which would create phantom errors as the left arm moves.
    _rel_target_pos = init_rel_pos.copy()
    _rel_target_ori = init_rel_ori.copy()

    # Track previous grasp handle pose to detect user drag
    _prev_grasp_pos = np.array(grasp_handle.position)
    _prev_grasp_wxyz = np.array(grasp_handle.wxyz)

    # Track previous mode to detect changes and snap markers
    _prev_mode = coord_mode.value
    _mode_change_requested = False
    # Skip one IK step on mode change so arms stay at current config (markers snap only)
    _skip_ik_this_frame = False

    @coord_mode.on_update
    def _on_mode_change(_) -> None:
        nonlocal _mode_change_requested
        _mode_change_requested = True

    # Print collision geometry info at startup
    try:
        n_obj = len(robot.get_collision_geometry_names())
        n_pairs = len(robot.get_collision_pair_names())
        n_excl = len(_collision_exclusions)
        print(f"[collision] {n_obj} geometries, {n_pairs} pairs "
              f"({n_excl} excluded, {n_pairs - n_excl} active)")
    except Exception:
        pass

    print("ECTS Dual-Arm IK (dual iiwa) running on http://localhost:8080")
    print("  Blue marker  = absolute frame (object midpoint); in Orthogonal = left EE")
    print("  Green marker = relative frame (right arm grasp); in Orthogonal = right EE")

    step_opts = embodik.PositionStepOptions()

    while True:
        try:
            # --- Update tasks with current robot state first ---
            abs_task.update(robot)
            rel_task.update(robot)
            left_ee_task.update(robot)
            right_ee_task.update(robot)

            # --- Snap: move markers to current robot state ---
            if _snap_requested:
                _snap_requested = False
                if coord_mode.value == "Orthogonal":
                    left_se3 = robot.get_frame_pose(LEFT_FRAME)
                    snap_abs_pos = np.array(left_se3.translation)
                    snap_abs_ori = np.array(left_se3.rotation)
                else:
                    snap_abs_pos = abs_task.current_position.copy()
                    snap_abs_ori = abs_task.current_orientation.copy()
                object_handle.position = tuple(snap_abs_pos)
                object_handle.wxyz = tuple(_wxyz_from_mat(snap_abs_ori))
                object_sphere.position = tuple(snap_abs_pos)
                object_sphere.wxyz = tuple(_wxyz_from_mat(snap_abs_ori))
                right_se3 = robot.get_frame_pose(RIGHT_FRAME)
                new_g_pos = np.array(right_se3.translation)
                new_g_wxyz = _wxyz_from_mat(np.array(right_se3.rotation))
                grasp_handle.position = tuple(new_g_pos)
                grasp_handle.wxyz = tuple(new_g_wxyz)
                grasp_sphere.position = tuple(new_g_pos)
                grasp_sphere.wxyz = tuple(new_g_wxyz)
                # Capture current relative pose as the new target (for non-Orthogonal)
                _rel_target_pos = rel_task.current_position.copy()
                _rel_target_ori = rel_task.current_orientation.copy()
                _prev_grasp_pos = new_g_pos.copy()
                _prev_grasp_wxyz = new_g_wxyz.copy()

            # --- Reset: restore to initial config ---
            if _reset_requested:
                _reset_requested = False
                q = q_init.copy()
                robot.update_configuration(q)
                abs_task.update(robot)
                rel_task.update(robot)
                left_ee_task.update(robot)
                right_ee_task.update(robot)
                rel_task.capture_current_as_target()
                if coord_mode.value == "Orthogonal":
                    left_se3 = robot.get_frame_pose(LEFT_FRAME)
                    reset_obj_pos = np.array(left_se3.translation)
                    reset_obj_ori = np.array(left_se3.rotation)
                    object_handle.position = tuple(reset_obj_pos)
                    object_handle.wxyz = tuple(_wxyz_from_mat(reset_obj_ori))
                    object_sphere.position = tuple(reset_obj_pos)
                    object_sphere.wxyz = tuple(_wxyz_from_mat(reset_obj_ori))
                else:
                    object_handle.position = tuple(init_abs_pos)
                    object_handle.wxyz = tuple(quat_abs_init)
                    object_sphere.position = tuple(init_abs_pos)
                    object_sphere.wxyz = tuple(quat_abs_init)
                grasp_handle.position = tuple(right_init_pos)
                grasp_handle.wxyz = tuple(quat_grasp_init)
                grasp_sphere.position = tuple(right_init_pos)
                grasp_sphere.wxyz = tuple(quat_grasp_init)
                _rel_target_pos = rel_task.current_position.copy()
                _rel_target_ori = rel_task.current_orientation.copy()
                _prev_grasp_pos = np.array(right_init_pos)
                _prev_grasp_wxyz = np.array(quat_grasp_init)
                for i, idx in enumerate(arm_indices):
                    joint_sliders[i].value = float(q[idx])

            # --- Update IK settings ---
            solver.dt = solver_dt_slider.value
            solver.set_damping(damping_slider.value)
            solver.set_tolerance(tolerance_slider.value)

            # --- Coordination mode ---
            mode_changed = _mode_change_requested or (coord_mode.value != _prev_mode)
            is_orthogonal = coord_mode.value == "Orthogonal"
            ects_cfg = _ects_config(coord_mode.value)
            abs_task.set_alpha(ects_cfg.alpha)
            if is_orthogonal:
                abs_task.active = False
                rel_task.active = False
                left_ee_task.active = True
                right_ee_task.active = True
            else:
                left_ee_task.active = False
                right_ee_task.active = False
                abs_task.active = True
                rel_task.active = ects_cfg.coordinated

            ee_mode = getattr(
                embodik.TaskSolveMode, ee_mode_dropdown.value, embodik.TaskSolveMode.SCALE
            )
            ee_fallback = bool(ee_fallback_checkbox.value)
            abs_task.solve_mode = ee_mode
            rel_task.solve_mode = ee_mode
            left_ee_task.solve_mode = ee_mode
            right_ee_task.solve_mode = ee_mode
            abs_task.allow_min_error_fallback = ee_fallback
            rel_task.allow_min_error_fallback = ee_fallback
            left_ee_task.allow_min_error_fallback = ee_fallback
            right_ee_task.allow_min_error_fallback = ee_fallback

            if mode_changed:
                _mode_change_requested = False
                _prev_mode = coord_mode.value
                _skip_ik_this_frame = True  # keep arms at current config this frame
                if is_orthogonal:
                    # Orthogonal: markers = left EE and right EE
                    left_se3 = robot.get_frame_pose(LEFT_FRAME)
                    eff_pos = np.array(left_se3.translation)
                    eff_ori = np.array(left_se3.rotation)
                else:
                    # ECTS: object marker = effective absolute frame (alpha blend)
                    eff_pos, eff_ori = _effective_absolute_pose(
                        robot, LEFT_FRAME, RIGHT_FRAME, ects_cfg.alpha
                    )
                eff_wxyz = _wxyz_from_mat(eff_ori)
                object_handle.position = tuple(eff_pos)
                object_handle.wxyz = tuple(eff_wxyz)
                object_sphere.position = tuple(eff_pos)
                object_sphere.wxyz = tuple(eff_wxyz)
                # Snap grasp_handle to current right arm EE world position
                right_se3 = robot.get_frame_pose(RIGHT_FRAME)
                new_g_pos = np.array(right_se3.translation)
                new_g_wxyz = _wxyz_from_mat(np.array(right_se3.rotation))
                grasp_handle.position = tuple(new_g_pos)
                grasp_handle.wxyz = tuple(new_g_wxyz)
                grasp_sphere.position = tuple(new_g_pos)
                grasp_sphere.wxyz = tuple(new_g_wxyz)
                # Reset relative target to current relative pose (for ECTS)
                rel_task.update(robot)
                _rel_target_pos = rel_task.current_position.copy()
                _rel_target_ori = rel_task.current_orientation.copy()
                _prev_grasp_pos = new_g_pos.copy()
                _prev_grasp_wxyz = new_g_wxyz.copy()

            # --- Axis masks (MATLAB S_cart1 / S_cart2) ---
            abs_task.set_position_mask(np.array([
                1.0 if abs_pos_x.value else 0.0,
                1.0 if abs_pos_y.value else 0.0,
                1.0 if abs_pos_z.value else 0.0,
            ]))
            abs_task.set_orientation_mask(np.array([
                1.0 if abs_ori_x.value else 0.0,
                1.0 if abs_ori_y.value else 0.0,
                1.0 if abs_ori_z.value else 0.0,
            ]))
            rel_task.set_position_mask(np.array([
                1.0 if rel_pos_x.value else 0.0,
                1.0 if rel_pos_y.value else 0.0,
                1.0 if rel_pos_z.value else 0.0,
            ]))
            rel_task.set_orientation_mask(np.array([
                1.0 if rel_ori_x.value else 0.0,
                1.0 if rel_ori_y.value else 0.0,
                1.0 if rel_ori_z.value else 0.0,
            ]))
            if is_orthogonal:
                left_ee_task.set_position_mask(np.array([
                    1.0 if abs_pos_x.value else 0.0,
                    1.0 if abs_pos_y.value else 0.0,
                    1.0 if abs_pos_z.value else 0.0,
                ]))
                left_ee_task.set_orientation_mask(np.array([
                    1.0 if abs_ori_x.value else 0.0,
                    1.0 if abs_ori_y.value else 0.0,
                    1.0 if abs_ori_z.value else 0.0,
                ]))
                right_ee_task.set_position_mask(np.array([
                    1.0 if rel_pos_x.value else 0.0,
                    1.0 if rel_pos_y.value else 0.0,
                    1.0 if rel_pos_z.value else 0.0,
                ]))
                right_ee_task.set_orientation_mask(np.array([
                    1.0 if rel_ori_x.value else 0.0,
                    1.0 if rel_ori_y.value else 0.0,
                    1.0 if rel_ori_z.value else 0.0,
                ]))

            # --- Nullspace secondary objective ---
            ns_gain = 10.0 ** nullspace_exp_slider.value
            ns_obj = nullspace_objective.value
            if ns_obj == "None":
                posture.weight = 0.0
            elif ns_obj == "Bias Pose":
                posture.weight = ns_gain
                posture.set_target_configuration(q_init.copy())
            elif ns_obj == "Metric Optimization":
                posture.weight = ns_gain
                eps = 1e-4
                J0 = robot.get_frame_jacobian(LEFT_FRAME)
                m0 = embodik.velocity_manipulability(J0)
                grad = np.zeros(len(q))
                for k in range(len(q)):
                    q_plus = q.copy()
                    q_plus[k] += eps
                    robot.update_configuration(q_plus)
                    Jk = robot.get_frame_jacobian(LEFT_FRAME)
                    grad[k] = (embodik.velocity_manipulability(Jk) - m0) / eps
                robot.update_configuration(q)
                q_bias = q + grad * 0.01
                posture.set_target_configuration(q_bias)

            # --- Relative constraint (bounds on stored relative target) ---
            if constraint_mode.value == "relative_bounds":
                bound_m = constraint_bound.value * 0.001
                lower = np.array([
                    _rel_target_pos[0] - bound_m,
                    _rel_target_pos[1] - bound_m,
                    _rel_target_pos[2] - bound_m,
                    -10.0, -10.0, -10.0,
                ])
                upper = np.array([
                    _rel_target_pos[0] + bound_m,
                    _rel_target_pos[1] + bound_m,
                    _rel_target_pos[2] + bound_m,
                    10.0, 10.0, 10.0,
                ])
                mask = np.array([1, 1, 1, 0, 0, 0], dtype=np.float64)
                solver.configure_relative_pose_constraint(
                    LEFT_FRAME, RIGHT_FRAME, lower, upper, mask
                )
            else:
                solver.clear_relative_pose_constraint()

            solver_elapsed_ms = 0.0

            if manual_control.value:
                if not prev_manual:
                    for i, idx in enumerate(arm_indices):
                        joint_sliders[i].value = float(q[idx])
                for i, idx in enumerate(arm_indices):
                    q[idx] = joint_sliders[i].value
                robot.update_configuration(q)
                status_text.value = "Manual"

            else:
                if _skip_ik_this_frame:
                    _skip_ik_this_frame = False
                    status_text.value = "Mode changed (arms held)"
                else:
                    Kp = pos_gain_slider.value
                    Ko = rot_gain_slider.value
                    step_opts.max_steps = int(iterations_slider.value)

                    if is_orthogonal:
                        # Orthogonal: two independent EE pose targets.
                        target_left_pose = Rt(
                            R=_mat_from_wxyz(np.array(object_handle.wxyz)),
                            t=np.array(object_handle.position),
                        )
                        target_right_pose = Rt(
                            R=_mat_from_wxyz(np.array(grasp_handle.wxyz)),
                            t=np.array(grasp_handle.position),
                        )

                        targets = [
                            embodik.TaskTarget.from_se3("left_ee", target_left_pose, Kp, Ko),
                            embodik.TaskTarget.from_se3("right_ee", target_right_pose, Kp, Ko),
                        ]
                        rel_error_vec = np.zeros(6)
                    else:
                        # ECTS: absolute task from object marker, relative task from grasp target.
                        marker_pos = np.array(object_handle.position)
                        marker_R = _mat_from_wxyz(np.array(object_handle.wxyz))
                        target_abs_pose = Rt(R=marker_R, t=marker_pos)

                        # Relative task: detect grasp handle drag, maintain stored target
                        curr_grasp_pos = np.array(grasp_handle.position)
                        curr_grasp_wxyz = np.array(grasp_handle.wxyz)
                        if not np.allclose(curr_grasp_pos, _prev_grasp_pos, atol=1e-4) or \
                                not np.allclose(curr_grasp_wxyz, _prev_grasp_wxyz, atol=1e-4):
                            left_se3_cur = robot.get_frame_pose(LEFT_FRAME)
                            R_left_cur = np.array(left_se3_cur.rotation)
                            t_left_cur = np.array(left_se3_cur.translation)
                            _rel_target_pos = R_left_cur.T @ (curr_grasp_pos - t_left_cur)
                            _rel_target_ori = R_left_cur.T @ _mat_from_wxyz(curr_grasp_wxyz)
                            _prev_grasp_pos = curr_grasp_pos.copy()
                            _prev_grasp_wxyz = curr_grasp_wxyz.copy()

                        target_rel_pose = Rt(R=_rel_target_ori, t=_rel_target_pos)

                        targets = [
                            embodik.TaskTarget.from_se3("absolute", target_abs_pose, Kp, Ko),
                            embodik.TaskTarget.from_se3("relative", target_rel_pose, Kp, Ko),
                        ]

                    # --- Solve ---
                    t0 = time.perf_counter()
                    result = solver.solve_position_step(q, targets, step_opts)
                    solver_elapsed_ms = (time.perf_counter() - t0) * 1000.0

                    q = np.array(result.q_solution)
                    robot.update_configuration(q)
                    rel_task.update(robot)
                    rel_error_vec[:3] = rel_task.current_position - _rel_target_pos
                    rel_error_vec[3:] = 0.0

                    status_text.value = str(result.status)
                    for i, idx in enumerate(arm_indices):
                        joint_sliders[i].value = float(q[idx])

            prev_manual = manual_control.value

            # --- Batch visualization update ---
            cfg_dict = {name: float(q[i]) for i, name in enumerate(joint_names)}
            urdf_vis.update_cfg(cfg_dict)

            # Keep colored spheres in sync with marker poses (so they move when user drags)
            object_sphere.position = tuple(object_handle.position)
            object_sphere.wxyz = tuple(object_handle.wxyz)
            grasp_sphere.position = tuple(grasp_handle.position)
            grasp_sphere.wxyz = tuple(grasp_handle.wxyz)

            # --- Collision debug visualization ---
            _update_collision_visuals()

            # --- Status display ---
            abs_task.update(robot)
            abs_err_m = np.linalg.norm(abs_task.current_position - np.array(object_handle.position))
            rel_err_m = float(np.linalg.norm(rel_error_vec[:3])) if not manual_control.value else 0.0
            elapsed_text.value = f"{solver_elapsed_ms:.2f}"
            abs_error_text.value = f"{abs_err_m:.4f}"
            rel_error_text.value = f"{rel_err_m:.4f}" if not manual_control.value else "--"

            time.sleep(solver.dt)

        except KeyboardInterrupt:
            print("\nShutting down...")
            break


if __name__ == "__main__":
    main()
