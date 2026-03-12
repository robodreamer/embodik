"""CoM Support-Polygon Constraint Example

Demonstrates the ``configure_com_constraint`` API.  Visualization style:

* **Outer polygon** (blue, thick lines) – the full support region.
* **Inner / active polygon** (green, medium lines) – after inward margin is applied;
  this is the boundary the solver actually enforces.
* **CoM 3D sphere** – colour-coded: green = inside, orange = near boundary,
  red = outside.
* **Floor disk** – CoM projected to z=0.
* **Drop line** – vertical line connecting the 3D CoM to its floor projection.

Usage
-----
    cd examples/
    python 08_com_constraint_example.py [--robot panda] [--visualizer pinocchio]

Interact
--------
* Drag the green transform control to move the EE target.
* Watch the CoM sphere change colour as it approaches / crosses the polygon edge.
* Adjust "Safety margin %" to shrink the active boundary and tighten the constraint.
* Toggle "Enable CoM Constraint" to compare free vs. constrained motion.
"""

import argparse
import logging
import time
import warnings

import numpy as np

import embodik
from embodik import r2q, q2r, Rt
from embodik import create_robot_visualizer
from embodik.utils import compute_pose_error, limit_task_velocity

from utils.robot_models import load_robot_presets, resolve_robot_configuration

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Colours (RGB 0-255)
# ---------------------------------------------------------------------------
COLOR_OUTER_POLY = (30, 120, 220)     # blue – outer boundary
COLOR_INNER_POLY = (50, 200, 80)      # green – active (margined) boundary
COLOR_COM_INSIDE = (50, 200, 80)      # green – CoM inside
COLOR_COM_NEAR   = (255, 180, 0)      # orange – CoM near boundary
COLOR_COM_OUTSIDE = (220, 50, 30)     # red – CoM outside
COLOR_DROP_LINE  = (180, 180, 180)    # light grey
COLOR_FLOOR_DISK = (180, 180, 180)    # light grey

# ---------------------------------------------------------------------------
# Default support polygon (world XY).
# Panda default CoM is roughly at (0.025, 0.006) so this polygon is
# deliberately snug: moving the EE forward ~15 cm will push the CoM to the
# right boundary and the constraint visibly saturates.
# ---------------------------------------------------------------------------
DEFAULT_POLYGON = np.array([
    [-0.08, -0.10],
    [ 0.18, -0.10],
    [ 0.18,  0.10],
    [-0.08,  0.10],
], dtype=float)

# How close to the boundary before the CoM sphere turns orange (metres).
NEAR_BOUNDARY_THRESHOLD = 0.02


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def _polygon_edge_pts(poly_xy: np.ndarray, z: float = 0.002) -> np.ndarray:
    """Return (N, 2, 3) array of line segments for the polygon boundary."""
    n = len(poly_xy)
    pts3 = np.column_stack([poly_xy, np.full(n, z)])
    return np.array([[pts3[i], pts3[(i + 1) % n]] for i in range(n)])


def _shrink_polygon_2d(poly: np.ndarray, margin_frac: float) -> np.ndarray:
    """Shrink polygon vertices toward centroid by ``margin_frac`` × char_size.

    Uses mean distance from centroid to vertices (char_size) to match the
    C++ shrink_polygon behavior.
    """
    if margin_frac <= 0.0:
        return poly.copy()
    centroid = poly.mean(axis=0)
    char_size = np.mean(np.linalg.norm(poly - centroid, axis=1))
    shrink = np.clip(margin_frac, 0.0, 1.0) * max(0.0, char_size - 1e-9)
    result = []
    for v in poly:
        d = centroid - v
        dn = np.linalg.norm(d)
        result.append(v if dn < 1e-12 else v + (shrink / dn) * d)
    return np.array(result)


def _polygon_slack(poly_xy: np.ndarray, com_xy: np.ndarray) -> np.ndarray:
    """Compute per-edge slack (positive = inside) using outward-normal half-planes.

    Convention matches the C++ ``polygon_to_halfplanes``:
    outward normal = (edge.y, -edge.x) normalised; slack = n·(v0 - com).
    """
    n = len(poly_xy)
    slacks = np.empty(n)
    for i in range(n):
        v0 = poly_xy[i]
        v1 = poly_xy[(i + 1) % n]
        edge = v1 - v0
        normal = np.array([edge[1], -edge[0]])
        nn = np.linalg.norm(normal)
        if nn > 1e-12:
            normal /= nn
        slacks[i] = np.dot(normal, v0 - com_xy)
    return slacks


def _com_color(min_slack: float) -> tuple:
    if min_slack < 0:
        return COLOR_COM_OUTSIDE
    if min_slack < NEAR_BOUNDARY_THRESHOLD:
        return COLOR_COM_NEAR
    return COLOR_COM_INSIDE


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    presets = load_robot_presets()
    parser = argparse.ArgumentParser(
        description="embodiK CoM support-polygon constraint demo."
    )
    parser.add_argument(
        "--robot",
        choices=sorted(presets.keys()),
        default="panda" if "panda" in presets else sorted(presets.keys())[0],
    )
    parser.add_argument(
        "--visualizer",
        choices=["pinocchio", "viserurdf"],
        default="pinocchio",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args: argparse.Namespace) -> None:
    # ------------------------------------------------------------------
    # Robot
    # ------------------------------------------------------------------
    config = resolve_robot_configuration(args.robot)
    robot: embodik.RobotModel = config["robot"]
    target_link: str = config["target_link"]
    q_default: np.ndarray = config["default_configuration"]

    logger.info(f"Loaded {config['display_name']} – nq={robot.nq}, nv={robot.nv}")

    q_current = q_default.copy()
    robot.update_configuration(q_current)

    q_lower, q_upper = robot.get_joint_limits()
    initial_ee_pose = robot.get_frame_pose(target_link)

    # ------------------------------------------------------------------
    # Solver
    # ------------------------------------------------------------------
    solver = embodik.KinematicsSolver(robot)
    solver.dt = 0.01
    solver.set_damping(0.1)
    solver.set_tolerance(0.1)

    frame_task = solver.add_frame_task("ee_task", target_link)
    frame_task.priority = 0
    frame_task.weight = 1.0
    frame_task.solve_mode = embodik.TaskSolveMode.SCALE
    frame_task.allow_min_error_fallback = True
    frame_task.set_target_velocity(np.zeros(6))

    posture_task = solver.add_posture_task("posture")
    posture_task.priority = 1
    posture_task.weight = 0.01
    posture_task.solve_mode = embodik.TaskSolveMode.MIN_ERROR
    posture_task.allow_min_error_fallback = True
    posture_task.set_target_configuration(q_default)

    # ------------------------------------------------------------------
    # Visualizer
    # ------------------------------------------------------------------
    presets = load_robot_presets()
    description_name = presets[args.robot].get("description_name", "")
    viz = create_robot_visualizer(
        robot_model=robot,
        backend=args.visualizer,
        description_name=description_name,
        port=8080,
        open_browser=True,
    )
    viz.add_grid("/ground", width=2, height=2)
    viz.display(q_current)
    server = viz.server

    # ------------------------------------------------------------------
    # GUI
    # ------------------------------------------------------------------
    with server.gui.add_folder("CoM Constraint"):
        enable_com = server.gui.add_checkbox("Enable CoM Constraint", initial_value=True)
        margin_pct_slider = server.gui.add_slider(
            "Safety margin (%)", min=0.0, max=40.0, initial_value=5.0, step=1.0
        )
        proximity_checkbox = server.gui.add_checkbox(
            "Use proximity activation", initial_value=True
        )
        # Read-only display: shows the auto-computed threshold (5 % of inradius).
        prox_display = server.gui.add_number(
            "Proximity threshold (m)", initial_value=0.0, disabled=True
        )
        com_vel_max_slider = server.gui.add_slider(
            "com_vel_max (m/s)", min=0.05, max=1.0, initial_value=0.4, step=0.05
        )
        com_acc_max_slider = server.gui.add_slider(
            "com_acc_max (m/s²)", min=0.01, max=0.5, initial_value=0.1, step=0.01
        )
        use_acc_limits = server.gui.add_checkbox("Acceleration limits", initial_value=True)

    with server.gui.add_folder("Visualization"):
        show_polygon = server.gui.add_checkbox("Show polygon", initial_value=True)
        show_com_viz = server.gui.add_checkbox("Show CoM", initial_value=True)
        show_drop_line = server.gui.add_checkbox("Show drop line", initial_value=True)

    with server.gui.add_folder("IK Controls"):
        timing_handle = server.gui.add_number("Elapsed (ms)", 0.001, disabled=True)
        pos_gain = server.gui.add_slider("Position Gain", min=10, max=100, initial_value=60, step=5)
        rot_gain = server.gui.add_slider("Rotation Gain", min=10, max=100, initial_value=60, step=5)
        max_lin_step = server.gui.add_slider("Max Linear Step (m/s)", min=0.1, max=1.0, initial_value=0.5, step=0.05)
        max_ang_step = server.gui.add_slider("Max Angular Step (rad/s)", min=0.1, max=1.0, initial_value=0.5, step=0.05)
        ee_mode_dropdown = server.gui.add_dropdown(
            "EE Solve Mode",
            options=("SCALE", "MIN_ERROR"),
            initial_value="SCALE",
        )
        ee_fallback_checkbox = server.gui.add_checkbox(
            "Allow SCALE fallback to MIN_ERROR",
            initial_value=False,
        )
        solve_diag = server.gui.add_text("Solve Diagnostic", initial_value="mode=SCALE, fb=False, scale=1.000")
        reset_btn = server.gui.add_button("Reset Arm")

    # ------------------------------------------------------------------
    # EE transform control
    # ------------------------------------------------------------------
    ik_target = server.scene.add_transform_controls(
        "/ik_target",
        scale=0.2,
        position=tuple(initial_ee_pose.translation),
        wxyz=tuple(r2q(initial_ee_pose.rotation)),
    )

    @reset_btn.on_click
    def _(_):
        nonlocal q_current
        q_current = q_default.copy()
        robot.update_configuration(q_current)
        viz.display(q_current)
        ee = robot.get_frame_pose(target_link)
        ik_target.position = tuple(ee.translation)
        ik_target.wxyz = tuple(r2q(ee.rotation))
        logger.info("Arm reset to default configuration.")

    # ------------------------------------------------------------------
    # Constraint configuration (called whenever a GUI value changes)
    # ------------------------------------------------------------------
    support_polygon = DEFAULT_POLYGON.copy()

    def _margin_frac() -> float:
        return margin_pct_slider.value / 100.0

    _PROXIMITY_FRACTION = 0.05  # 5 % of polygon inradius → thin band near each edge

    def reconfigure_com() -> None:
        if enable_com.value:
            solver.configure_com_constraint(
                support_polygon=support_polygon,
                margin=_margin_frac(),
                frame_name="world",
                com_vel_max=com_vel_max_slider.value,
                com_acc_max=com_acc_max_slider.value,
                use_acceleration_limits=use_acc_limits.value,
                proximity_fraction=_PROXIMITY_FRACTION if proximity_checkbox.value else 0.0,
            )
            prox_display.value = round(solver.get_com_proximity_threshold(), 4)
        else:
            solver.clear_com_constraint()
            prox_display.value = 0.0

    for handle in (enable_com, margin_pct_slider,
                   proximity_checkbox, com_vel_max_slider,
                   com_acc_max_slider, use_acc_limits):
        @handle.on_update
        def _(_):
            reconfigure_com()

    reconfigure_com()

    # ------------------------------------------------------------------
    # Scene objects – created once, updated in place every frame
    # ------------------------------------------------------------------

    # Polygon boundaries
    _outer_poly_h = server.scene.add_line_segments(
        "/com_viz/outer_polygon",
        points=_polygon_edge_pts(support_polygon, z=0.002),
        colors=COLOR_OUTER_POLY,
        line_width=5.0,
    )
    _inner_poly_h = server.scene.add_line_segments(
        "/com_viz/inner_polygon",
        points=_polygon_edge_pts(support_polygon, z=0.003),
        colors=COLOR_INNER_POLY,
        line_width=3.0,
    )

    # CoM 3D sphere
    _com_sphere_h = server.scene.add_icosphere(
        "/com_viz/sphere",
        radius=0.03,
        color=COLOR_COM_INSIDE,
        position=(0.0, 0.0, 0.5),
    )

    # Floor disk (tiny sphere at z≈0)
    _floor_disk_h = server.scene.add_icosphere(
        "/com_viz/floor_disk",
        radius=0.02,
        color=COLOR_FLOOR_DISK,
        position=(0.0, 0.0, 0.002),
    )

    # Vertical drop line: 1 segment (CoM → floor projection), shape (1, 2, 3)
    _drop_line_h = server.scene.add_line_segments(
        "/com_viz/drop_line",
        points=np.zeros((1, 2, 3), dtype=float),
        colors=COLOR_DROP_LINE,
        line_width=2.0,
    )

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info(f"embodiK CoM Constraint Demo – {config['display_name']}")
    logger.info(f"EE target: {target_link}")
    logger.info("Drag the green handle to move the end-effector.")
    logger.info(f"Initial CoM XY: {robot.get_com_position()[:2].round(4)}")
    logger.info("=" * 60)

    while True:
        start_t = time.time()

        robot.update_configuration(q_current)
        com_pos = robot.get_com_position()            # (3,)
        com_xy = com_pos[:2]

        # ---- Compute active (margined) polygon ----
        inner_poly = _shrink_polygon_2d(support_polygon, _margin_frac())

        # ---- CoM status ----
        slacks_outer = _polygon_slack(support_polygon, com_xy)
        slacks_inner = _polygon_slack(inner_poly, com_xy)
        min_slack = float(slacks_inner.min())
        sphere_color = _com_color(min_slack)

        # ---- Update polygon lines ----
        vis_poly = show_polygon.value
        _outer_poly_h.points = _polygon_edge_pts(support_polygon, z=0.002)
        _outer_poly_h.visible = vis_poly
        _inner_poly_h.points = _polygon_edge_pts(inner_poly, z=0.003)
        _inner_poly_h.visible = vis_poly and (_margin_frac() > 0.0)

        # ---- Update CoM sphere ----
        _com_sphere_h.position = tuple(float(v) for v in com_pos)
        _com_sphere_h.color = sphere_color
        _com_sphere_h.visible = show_com_viz.value

        # ---- Update floor disk ----
        _floor_disk_h.position = (float(com_xy[0]), float(com_xy[1]), 0.001)
        _floor_disk_h.color = sphere_color
        _floor_disk_h.visible = show_com_viz.value

        # ---- Update drop line ---- shape (1, 2, 3)
        _drop_line_h.points = np.array([[
            [float(com_xy[0]), float(com_xy[1]), float(com_pos[2])],
            [float(com_xy[0]), float(com_xy[1]), 0.001],
        ]])
        _drop_line_h.visible = show_com_viz.value and show_drop_line.value

        # ---- IK ----
        target_position = np.array(ik_target.position)
        target_rotation = q2r(np.array(ik_target.wxyz))
        target_pose = Rt(R=target_rotation, t=target_position)

        current_ee_pose = robot.get_frame_pose(target_link)
        pose_error = compute_pose_error(current_ee_pose, target_pose)

        if np.linalg.norm(pose_error) < 5e-4:
            time.sleep(1e-3)
            continue

        target_velocity = np.concatenate([
            pos_gain.value * pose_error[:3],
            rot_gain.value * pose_error[3:],
        ])
        target_velocity = limit_task_velocity(
            target_velocity,
            max_linear_step=max_lin_step.value,
            max_angular_step=max_ang_step.value,
        )

        frame_task.weight = 1.0
        frame_task.solve_mode = (
            embodik.TaskSolveMode.MIN_ERROR
            if ee_mode_dropdown.value == "MIN_ERROR"
            else embodik.TaskSolveMode.SCALE
        )
        frame_task.allow_min_error_fallback = bool(ee_fallback_checkbox.value)
        frame_task.set_target_velocity(target_velocity)

        result = solver.solve_velocity(q_current, apply_limits=True)
        effective_mode = (
            result.task_modes_effective[0].name
            if len(result.task_modes_effective) > 0
            else "SCALE"
        )
        used_fallback = (
            bool(result.task_used_fallback[0])
            if len(result.task_used_fallback) > 0
            else False
        )
        scale_value = (
            float(result.task_scales[0])
            if len(result.task_scales) > 0
            else 1.0
        )
        solve_diag.value = f"mode={effective_mode}, fb={used_fallback}, scale={scale_value:.3f}"

        if result.status == embodik.SolverStatus.SUCCESS:
            dq = result.joint_velocities * solver.dt
            q_current = np.clip(q_current + dq, q_lower, q_upper)
            robot.update_configuration(q_current)
            viz.display(q_current)
        else:
            logger.debug(f"Solver: {result.status}")

        elapsed_ms = (time.time() - start_t) * 1000.0
        timing_handle.value = 0.9 * timing_handle.value + 0.1 * elapsed_ms

        time.sleep(max(0.0, solver.dt - (time.time() - start_t)))


if __name__ == "__main__":
    main(parse_args())
