"""CoM Support-Polygon Constraint Example

Demonstrates the ``configure_com_constraint`` API.  Visualization style:

* **Outer polygon** (blue, thick lines) – the full support region.
* **Inner / active polygon** (green, medium lines) – after inward margin is applied;
  this is the boundary the solver actually enforces.
* **CoM 3D sphere** – colour-coded: green = inside, orange = near boundary,
  red = outside.
* **Capture point disk** (cyan) – commanded-velocity capture point.
* **ZMP disk** (magenta) – enforced explicit-state finite-difference ZMP.
* **Floor disk** – CoM projected to z=0.
* **Drop line** – vertical line connecting the 3D CoM to its floor projection.

Usage
-----
    cd examples/
    python 04_com_constraint_example.py [--robot KEY] [--visualizer pinocchio]

Interact
--------
* Drag the green transform control to move the EE target.
* Watch the CoM sphere change colour as it approaches / crosses the polygon edge.
* Adjust "Safety margin %" to shrink the active boundary and tighten the constraint.
* Toggle CoM and capture-point constraints independently.
* Enable horizontal momentum damping to regularize dynamic CoM motion.
* With ``--gpu-wbc --gpu-wbc-manifest PATH``, select GPU to exercise bounded
  world-XY CoM constraints, posture priority, adaptive dt, and self-collision.
  The manifest must match the loaded URDF and active joint order. Polygon
  updates retain a fixed row capacity; infeasible steps hold without fallback.
"""

import argparse
import logging
import time
import warnings
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from example_helpers.centroidal_support import (
    configure_capture_point_constraint,
    configure_centroidal_diagnostic_solver,
    configure_horizontal_momentum_damping,
    configure_velocity_zmp_constraint,
    evaluate_centroidal_diagnostics,
)
from example_helpers.ik_common import DEFAULT_VISER_PORT, configure_solver_runtime_policy
from embodik.gpu.wbc import (
    GPU_WBC_CAPABILITIES,
    GpuWbcMultiFrameSolver,
    derive_frame_active_joint_names,
)
from utils.robot_models import load_robot_presets, resolve_robot_configuration

import embodik
from embodik import Rt, create_robot_visualizer, q2r, r2q

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Colours (RGB 0-255)
# ---------------------------------------------------------------------------
COLOR_OUTER_POLY = (30, 120, 220)  # blue – outer boundary
COLOR_INNER_POLY = (50, 200, 80)  # green – active (margined) boundary
COLOR_COM_INSIDE = (50, 200, 80)  # green – CoM inside
COLOR_COM_NEAR = (255, 180, 0)  # orange – CoM near boundary
COLOR_COM_OUTSIDE = (220, 50, 30)  # red – CoM outside
COLOR_DROP_LINE = (180, 180, 180)  # light grey
COLOR_FLOOR_DISK = (180, 180, 180)  # light grey
COLOR_CAPTURE_POINT = (35, 175, 255)  # cyan
COLOR_ZMP = (235, 65, 180)  # magenta

# ---------------------------------------------------------------------------
# Default support polygon (world XY).
# Panda default CoM is roughly at (0.025, 0.006) so this polygon is
# deliberately snug: moving the EE forward ~15 cm will push the CoM to the
# right boundary and the constraint visibly saturates.
# ---------------------------------------------------------------------------
DEFAULT_POLYGON = np.array(
    [
        [-0.08, -0.10],
        [0.18, -0.10],
        [0.18, 0.10],
        [-0.08, 0.10],
    ],
    dtype=float,
)

# How close to the boundary before the CoM sphere turns orange (metres).
NEAR_BOUNDARY_THRESHOLD = 0.02


def gpu_collision_pairs(robot, urdf_path: Path, exclusions=(), *, auto=False) -> tuple:
    """Filter structural neighbors and explicit preset exclusions.

    Auto presets exclude links within two movable joints, including near-neighbor
    housings whose designed clearance can be smaller than the collision floor.
    This is a topology policy, independent of joint names and robot dimensions.
    """
    root = ET.parse(urdf_path).getroot()
    graph = {}
    for joint in root.findall("joint"):
        a = joint.find("parent").get("link")
        b = joint.find("child").get("link")
        cost = 0 if joint.get("type") == "fixed" else 1
        graph.setdefault(a, []).append((b, cost))
        graph.setdefault(b, []).append((a, cost))
    adjacent = set()
    for start in graph:
        pending = [(start, 0)]
        distances = {start: 0}
        while pending:
            link, distance = pending.pop()
            adjacent.add(frozenset((start, link)))
            for neighbor, cost in graph[link]:
                candidate = distance + cost
                if candidate <= (2 if auto else 1) and candidate < distances.get(
                    neighbor, float("inf")
                ):
                    distances[neighbor] = candidate
                    pending.append((neighbor, candidate))
    parents = {
        str(geom["name"]): str(geom["parent_frame"])
        for geom in robot.get_collision_geometries()
    }
    excluded = {frozenset(pair) for pair in exclusions}
    return tuple(
        (a, b)
        for a, b in robot.get_collision_pair_names()
        if frozenset((a, b)) not in excluded
        and not (parents.get(a) and parents.get(a) == parents.get(b))
        and frozenset((parents.get(a), parents.get(b))) not in adjacent
    )


def create_gpu_solver(args, config):
    """Build the optional fixed-base solver without changing the CPU model."""
    robot = config["robot"]
    names = derive_frame_active_joint_names(robot, config["target_link"])
    indices = tuple(int(robot.get_joint_config_index(name)) for name in names)
    gpu_robot = embodik.RobotModel(
        str(config["urdf_path"]), actuated_joint_names=names, floating_base=False
    )
    velocity_indices = tuple(
        int(gpu_robot.get_joint_velocity_index(name)) for name in names
    )
    active_default = np.asarray(config["default_configuration"])[list(indices)]
    preset = load_robot_presets()[config["key"]]
    exclusions = preset.get("collision_exclusions", ())
    auto = exclusions == "auto"
    if auto:
        exclusions = preset.get("collision_exclusion_overrides", ())
    gpu = GpuWbcMultiFrameSolver(
        args.gpu_wbc_manifest,
        Path(config["urdf_path"]),
        args.gpu_wbc_cache_dir,
        robot=gpu_robot,
        robot_name=config["key"],
        frames=(config["target_link"],),
        active_joint_names=names,
        default_configuration=active_default,
        solver_backend="torch_srinv",
        com_support_polygon_xy=DEFAULT_POLYGON,
        com_full_robot=robot,
        com_full_default_configuration=config["default_configuration"],
        iterations=1,
        dt=0.01,
        position_gain=10.0,
        orientation_gain=10.0,
        collision_pairs=gpu_collision_pairs(
            gpu_robot, Path(config["urdf_path"]), exclusions, auto=auto
        ),
        collision_min_distance_m=0.03,
        collision_query_distance_m=0.15,
        posture_target_configuration=tuple(active_default),
        posture_velocity_indices=velocity_indices,
        posture_weights=tuple(1.0 for _ in names),
        posture_gain=1.0,
    )
    gpu.configure_runtime(collision_enabled=False)
    return gpu, indices


def gpu_step(
    gpu,
    q,
    indices,
    target,
    *,
    com_enabled,
    collision_enabled,
    collision_min_distance,
    show_debug,
    posture_gain,
    adaptive_dt,
    position_gain,
    orientation_gain,
    iterations,
    dt,
    support_polygon=None,
    com_margin=0.0,
    com_vel_max=0.4,
    com_acc_max=0.1,
    com_use_acceleration_limits=True,
    com_proximity_fraction=0.05,
):
    """Run bounded CoM and optional constraints on the GPU backend."""
    if collision_enabled and not gpu.collision_supported:
        raise RuntimeError("self-collision constraints unavailable for this model")
    gpu.configure_runtime(
        dt=float(dt),
        iterations=int(iterations),
        frame_position_gains=(float(position_gain),),
        frame_orientation_gains=(float(orientation_gain),),
        collision_enabled=bool(collision_enabled),
        collision_min_distance_m=(
            float(collision_min_distance) if gpu.collision_supported else None
        ),
        posture_gain=float(posture_gain),
        adaptive_dt=bool(adaptive_dt),
        com_enabled=bool(com_enabled),
        com_support_polygon_xy=(
            DEFAULT_POLYGON if support_polygon is None else support_polygon
        ),
        com_margin=float(com_margin),
        com_vel_max=float(com_vel_max),
        com_acc_max=float(com_acc_max),
        com_use_acceleration_limits=bool(com_use_acceleration_limits),
        com_proximity_fraction=float(com_proximity_fraction),
    )
    result = gpu.solve_step(
        np.asarray(q)[list(indices)],
        (target,),
        include_collision_debug=bool(show_debug and collision_enabled),
    )
    updated = np.asarray(q).copy()
    updated[list(indices)] = result.joints
    return updated, result


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
    parser.add_argument(
        "--port", type=int, default=DEFAULT_VISER_PORT, help="Viser server port."
    )
    parser.add_argument("--gpu-wbc", action="store_true")
    parser.add_argument("--gpu-wbc-manifest", type=Path)
    parser.add_argument(
        "--gpu-wbc-cache-dir",
        type=Path,
        default=Path("build/gpu-wbc-newton-cache"),
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
    dq_current = np.zeros(robot.nv, dtype=float)
    robot.update_configuration(q_current)

    initial_ee_pose = robot.get_frame_pose(target_link)

    gpu_solver: GpuWbcMultiFrameSolver | None = None
    gpu_indices = ()
    gpu_fault: str | None = None
    if args.gpu_wbc:
        gpu_solver, gpu_indices = create_gpu_solver(args, config)
        gpu_solver.warm_up(
            q_current[list(gpu_indices)],
            (initial_ee_pose,),
        )

    # ------------------------------------------------------------------
    # Solver
    # ------------------------------------------------------------------
    solver = embodik.KinematicsSolver(robot)
    solver.dt = 0.01
    configure_solver_runtime_policy(solver)

    frame_task = solver.add_frame_task("ee_task", target_link)
    frame_task.priority = 0
    frame_task.weight = 1.0
    frame_task.solve_mode = embodik.TaskSolveMode.SCALE
    frame_task.allow_min_error_fallback = False

    posture_task = solver.add_posture_task("posture")
    posture_task.priority = 1
    posture_task.weight = 0.01
    posture_task.solve_mode = embodik.TaskSolveMode.MIN_ERROR
    posture_task.allow_min_error_fallback = False
    posture_task.set_target_configuration(q_default)

    momentum_task = solver.add_centroidal_momentum_task("horizontal_momentum_damping")
    momentum_task.solve_mode = embodik.TaskSolveMode.MIN_ERROR
    momentum_task.allow_min_error_fallback = False
    configure_horizontal_momentum_damping(
        momentum_task,
        enabled=False,
        weight=0.01,
        priority=1,
    )

    centroidal_observer = embodik.KinematicsSolver(robot)
    centroidal_observer.dt = solver.dt

    # ------------------------------------------------------------------
    # Visualizer
    # ------------------------------------------------------------------
    presets = load_robot_presets()
    description_name = presets[args.robot].get("description_name", "")
    viz = create_robot_visualizer(
        robot_model=robot,
        backend=args.visualizer,
        description_name=description_name,
        port=args.port,
        open_browser=True,
    )
    viz.add_grid("/ground", width=2, height=2)
    viz.display(q_current)
    server = viz.server

    # ------------------------------------------------------------------
    # GUI
    # ------------------------------------------------------------------
    with server.gui.add_folder("Centroidal Support"):
        enable_com = server.gui.add_checkbox("Enable CoM Constraint", initial_value=True)
        enable_capture_point = server.gui.add_checkbox(
            "Enable Capture Point Constraint", initial_value=True
        )
        enable_zmp = server.gui.add_checkbox("Enable Velocity ZMP Constraint", initial_value=True)
        enable_momentum_damping = server.gui.add_checkbox(
            "Damp Horizontal Momentum", initial_value=False
        )
        momentum_damping_weight = server.gui.add_slider(
            "Momentum Damping Weight", min=0.0, max=0.2, initial_value=0.01, step=0.005
        )
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
        use_acc_limits = server.gui.add_checkbox(
            "Acceleration limits", initial_value=True
        )

    with server.gui.add_folder("Visualization"):
        show_polygon = server.gui.add_checkbox("Show polygon", initial_value=True)
        show_com_viz = server.gui.add_checkbox("Show CoM", initial_value=True)
        show_drop_line = server.gui.add_checkbox("Show drop line", initial_value=True)
        show_centroidal_points = server.gui.add_checkbox(
            "Show Capture Point and ZMP", initial_value=True
        )

    with server.gui.add_folder("IK Controls"):
        backend_select = server.gui.add_dropdown(
            "Solver Backend",
            options=("CPU EmbodiK",)
            + (("GPU Newton/Warp",) if gpu_solver is not None else ()),
            initial_value="CPU EmbodiK",
        )
        timing_handle = server.gui.add_number("Elapsed (ms)", 0.001, disabled=True)
        pos_gain = server.gui.add_slider(
            "Position Gain", min=0.1, max=200, initial_value=10.0, step=1.0
        )
        rot_gain = server.gui.add_slider(
            "Orientation Gain", min=0.1, max=200, initial_value=10.0, step=1.0
        )
        iterations_slider = server.gui.add_slider(
            "IK Iterations", min=1, max=20, initial_value=1, step=1
        )
        ee_mode_dropdown = server.gui.add_dropdown(
            "EE Solve Mode",
            options=("SCALE", "SCALE_ELASTIC", "MIN_ERROR"),
            initial_value="SCALE_ELASTIC",
        )
        ee_fallback_checkbox = server.gui.add_checkbox(
            "Allow SCALE fallback to MIN_ERROR",
            initial_value=False,
        )
        solve_diag = server.gui.add_text(
            "Solve Diagnostic", initial_value="mode=SCALE, fb=False, scale=1.000"
        )
        centroidal_diag = server.gui.add_text(
            "Centroidal Diagnostic", initial_value="CP=-- | ZMP=--"
        )
        reset_btn = server.gui.add_button("Reset Arm")

    with server.gui.add_folder("GPU Controls (GPU backend only)"):
        server.gui.add_markdown(
            "CoM controls apply to both backends (world XY polygon). GPU uses "
            "bounded recovery and holds infeasible steps. EE solve-mode/fallback controls "
            "apply to CPU only; GPU uses directional extended SRINV. Capture point, "
            "velocity ZMP, and momentum damping currently apply to CPU only."
        )
        gpu_collision = server.gui.add_checkbox(
            "GPU Self Collision", initial_value=False
        )
        gpu_collision_distance = server.gui.add_slider(
            "GPU Minimum Distance (m)",
            min=0.001,
            max=0.10,
            initial_value=0.03,
            step=0.001,
        )
        gpu_debug = server.gui.add_checkbox(
            "Show GPU Collision Debug", initial_value=False
        )
        gpu_posture = server.gui.add_slider(
            "GPU Posture Gain", min=0.0, max=10.0, initial_value=1.0, step=0.1
        )
        gpu_adaptive = server.gui.add_checkbox("GPU Adaptive dt", initial_value=False)
    collision_a = server.scene.add_icosphere(
        "/gpu_collision/a", radius=0.01, color=(255, 80, 80), visible=False
    )
    collision_b = server.scene.add_icosphere(
        "/gpu_collision/b", radius=0.01, color=(80, 80, 255), visible=False
    )
    collision_line = server.scene.add_line_segments(
        "/gpu_collision/line",
        points=np.zeros((1, 2, 3)),
        colors=(255, 180, 0),
        line_width=3.0,
        visible=False,
    )

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
        nonlocal q_current, dq_current, gpu_fault
        gpu_fault = None
        if gpu_solver is not None:
            gpu_solver.reset_state()
        q_current = q_default.copy()
        dq_current = np.zeros(robot.nv, dtype=float)
        robot.update_configuration(q_current)
        viz.display(q_current)
        ee = robot.get_frame_pose(target_link)
        ik_target.position = tuple(ee.translation)
        ik_target.wxyz = tuple(r2q(ee.rotation))
        logger.info("Arm reset to default configuration.")

    @backend_select.on_update
    def _(_) -> None:
        nonlocal gpu_fault
        gpu_fault = None
        gpu_selected = backend_select.value == "GPU Newton/Warp"
        ee_mode_dropdown.disabled = gpu_selected
        ee_fallback_checkbox.disabled = gpu_selected
        enable_capture_point.disabled = gpu_selected and not (
            GPU_WBC_CAPABILITIES.capture_point_constraints
        )
        enable_zmp.disabled = gpu_selected and not (
            GPU_WBC_CAPABILITIES.velocity_zmp_constraints
        )
        enable_momentum_damping.disabled = gpu_selected and not (
            GPU_WBC_CAPABILITIES.centroidal_momentum_tasks
        )
        momentum_damping_weight.disabled = enable_momentum_damping.disabled
        if enable_capture_point.disabled:
            enable_capture_point.value = False
        if enable_zmp.disabled:
            enable_zmp.value = False
        if enable_momentum_damping.disabled:
            enable_momentum_damping.value = False
        if gpu_solver is not None:
            gpu_solver.reset_state()

    # ------------------------------------------------------------------
    # Constraint configuration (called whenever a GUI value changes)
    # ------------------------------------------------------------------
    support_polygon = DEFAULT_POLYGON.copy()

    def _margin_frac() -> float:
        return margin_pct_slider.value / 100.0

    _PROXIMITY_FRACTION = 0.05  # 5 % of polygon inradius → thin band near each edge

    def reconfigure_centroidal_support() -> None:
        if enable_com.value:
            solver.configure_com_constraint(
                support_polygon=support_polygon,
                margin=_margin_frac(),
                frame_name="world",
                com_vel_max=com_vel_max_slider.value,
                com_acc_max=com_acc_max_slider.value,
                use_acceleration_limits=use_acc_limits.value,
                proximity_fraction=(
                    _PROXIMITY_FRACTION if proximity_checkbox.value else 0.0
                ),
            )
            prox_display.value = round(solver.get_com_proximity_threshold(), 4)
        else:
            solver.clear_com_constraint()
            prox_display.value = 0.0

        configure_capture_point_constraint(
            solver,
            enabled=bool(enable_capture_point.value),
            support_polygon=support_polygon,
            margin=_margin_frac(),
            frame_name="world",
        )
        configure_velocity_zmp_constraint(
            solver,
            enabled=bool(enable_zmp.value),
            support_polygon=support_polygon,
            margin=_margin_frac(),
            frame_name="world",
        )
        configure_horizontal_momentum_damping(
            momentum_task,
            enabled=bool(enable_momentum_damping.value),
            weight=float(momentum_damping_weight.value),
            priority=1,
        )
        configure_centroidal_diagnostic_solver(
            centroidal_observer,
            support_polygon=support_polygon,
            margin=_margin_frac(),
            frame_name="world",
            dt=solver.dt,
        )

    for handle in (
        enable_com,
        enable_capture_point,
        enable_zmp,
        enable_momentum_damping,
        momentum_damping_weight,
        margin_pct_slider,
        proximity_checkbox,
        com_vel_max_slider,
        com_acc_max_slider,
        use_acc_limits,
    ):

        @handle.on_update
        def _(_):
            reconfigure_centroidal_support()

    reconfigure_centroidal_support()

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

    _capture_point_h = server.scene.add_icosphere(
        "/com_viz/capture_point",
        radius=0.018,
        color=COLOR_CAPTURE_POINT,
        position=(0.0, 0.0, 0.006),
    )
    _zmp_h = server.scene.add_icosphere(
        "/com_viz/zmp",
        radius=0.014,
        color=COLOR_ZMP,
        position=(0.0, 0.0, 0.009),
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

    step_opts = embodik.PositionStepOptions()

    while True:
        start_t = time.time()

        robot.update_configuration(q_current)
        com_pos = robot.get_com_position()  # (3,)
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
        _drop_line_h.points = np.array(
            [
                [
                    [float(com_xy[0]), float(com_xy[1]), float(com_pos[2])],
                    [float(com_xy[0]), float(com_xy[1]), 0.001],
                ]
            ]
        )
        _drop_line_h.visible = show_com_viz.value and show_drop_line.value

        # ---- IK ----
        target_position = np.array(ik_target.position)
        target_rotation = q2r(np.array(ik_target.wxyz))
        target_pose = Rt(R=target_rotation, t=target_position)

        collision_a.visible = collision_b.visible = collision_line.visible = False

        if backend_select.value == "GPU Newton/Warp":
            assert gpu_solver is not None
            if gpu_fault is None:
                try:
                    q_before = q_current.copy()
                    q_next, gpu_result = gpu_step(
                        gpu_solver,
                        q_current,
                        gpu_indices,
                        target_pose,
                        com_enabled=enable_com.value,
                        support_polygon=support_polygon,
                        com_margin=_margin_frac(),
                        com_vel_max=com_vel_max_slider.value,
                        com_acc_max=com_acc_max_slider.value,
                        com_use_acceleration_limits=use_acc_limits.value,
                        com_proximity_fraction=(
                            _PROXIMITY_FRACTION if proximity_checkbox.value else 0.0
                        ),
                        collision_enabled=gpu_collision.value,
                        collision_min_distance=gpu_collision_distance.value,
                        show_debug=gpu_debug.value,
                        posture_gain=gpu_posture.value,
                        adaptive_dt=gpu_adaptive.value,
                        position_gain=pos_gain.value,
                        orientation_gain=rot_gain.value,
                        iterations=iterations_slider.value,
                        dt=solver.dt,
                    )
                except Exception as exc:
                    gpu_fault = f"{type(exc).__name__}: {exc}"
                    solve_diag.value = f"GPU FAULT — SAFE HOLD: {gpu_fault}"
                else:
                    q_current = q_next
                    if gpu_result is None:
                        continue
                    dq_current = (q_current - q_before) / solver.dt
                    robot.update_configuration(q_current)
                    viz.display(q_current)
                    solve_diag.value = (
                        f"GPU {gpu_result.status}, "
                        f"pos={max(gpu_result.position_errors) * 1e3:.1f} mm, "
                        f"wall={gpu_result.elapsed_ms:.2f} ms"
                        + (
                            f", CoM slack={gpu_result.minimum_com_slack_m * 1e3:.1f} mm"
                            if gpu_result.minimum_com_slack_m is not None
                            else ""
                        )
                    )
                    debug = gpu_result.collision_debug
                    if gpu_debug.value and gpu_collision.value and debug is not None:
                        collision_a.position = tuple(debug.point_a_world)
                        collision_b.position = tuple(debug.point_b_world)
                        collision_line.points = np.array(
                            [[debug.point_a_world, debug.point_b_world]]
                        )
                        collision_a.visible = collision_b.visible = (
                            collision_line.visible
                        ) = True
        else:
            frame_task.solve_mode = getattr(
                embodik.TaskSolveMode,
                ee_mode_dropdown.value,
                embodik.TaskSolveMode.SCALE,
            )
            frame_task.allow_min_error_fallback = bool(ee_fallback_checkbox.value)
            step_opts.position_gain = pos_gain.value
            step_opts.orientation_gain = rot_gain.value
            step_opts.max_steps = int(iterations_slider.value)
            step_opts.current_joint_velocity = dq_current
            result = solver.solve_position_step(
                q_current, target_pose, "ee_task", step_opts
            )
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
                float(result.task_scales[0]) if len(result.task_scales) > 0 else 1.0
            )
            solve_diag.value = f"CPU mode={effective_mode}, fb={used_fallback}, scale={scale_value:.3f}"
            dq_command = np.asarray(result.joint_velocities, dtype=float)
            if dq_command.shape != (robot.nv,) or not np.all(np.isfinite(dq_command)):
                dq_command = np.zeros(robot.nv, dtype=float)
            if bool(show_centroidal_points.value):
                diagnostics = evaluate_centroidal_diagnostics(
                    centroidal_observer,
                    q_current,
                    dq_current,
                    dq_command,
                )
                _capture_point_h.visible = diagnostics.capture_point is not None
                _zmp_h.visible = diagnostics.zmp is not None
                if diagnostics.capture_point is not None:
                    _capture_point_h.position = (
                        float(diagnostics.capture_point[0]),
                        float(diagnostics.capture_point[1]),
                        0.006,
                    )
                if diagnostics.zmp is not None:
                    _zmp_h.position = (
                        float(diagnostics.zmp[0]),
                        float(diagnostics.zmp[1]),
                        0.009,
                    )
                cp_text = (
                    f"{diagnostics.capture_point_min_slack:.4f} m"
                    if diagnostics.capture_point_min_slack is not None
                    else "invalid"
                )
                zmp_text = (
                    f"{diagnostics.zmp_min_slack:.4f} m"
                    if diagnostics.zmp_min_slack is not None
                    else "invalid"
                )
                centroidal_diag.value = f"CP slack={cp_text} | ZMP slack={zmp_text}"
            else:
                _capture_point_h.visible = False
                _zmp_h.visible = False
                centroidal_diag.value = "CP=hidden | ZMP=hidden"
            if result.status in (
                embodik.SolverStatus.SUCCESS,
                embodik.SolverStatus.INFEASIBLE,
                embodik.SolverStatus.NUMERICAL_ERROR,
            ):
                q_current = np.asarray(result.q_solution, dtype=float)
                dq_current = dq_command
                robot.update_configuration(q_current)
                viz.display(q_current)
            else:
                dq_current = np.zeros(robot.nv, dtype=float)
                logger.debug(f"Solver: {result.status}")

        elapsed_ms = (time.time() - start_t) * 1000.0
        timing_handle.value = 0.9 * timing_handle.value + 0.1 * elapsed_ms

        time.sleep(max(0.0, solver.dt - (time.time() - start_t)))


if __name__ == "__main__":
    main(parse_args())
