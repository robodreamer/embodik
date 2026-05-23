#!/usr/bin/env python3
"""Demo: collision hardening guarantees and per-pair distance overrides (Franka Panda).

Shows:
  - SolverStatus.COLLISION_VIOLATED when no safe step exists
  - Per-link-pair min_distance override via GUI
  - Live nearest-pair distance display and sphere markers
"""

from __future__ import annotations

import argparse
import os
import re
import time
from pathlib import Path
from typing import List, Optional, Tuple

import embodik
import numpy as np
import viser
from embodik import Rt, q2r, r2q
from example_helpers.ik_common import DEFAULT_VISER_PORT, configure_solver_runtime_policy
from viser.extras import ViserUrdf

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_Q = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785])
TARGET_LINK = "panda_hand"
DEFAULT_MIN_DIST = 0.05
DEFAULT_POS_GAIN = 10.0

_LINK_INDEX_PATTERN = re.compile(r"link_?([0-9]+)")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def ensure_ros_package_path(urdf_path: Path) -> None:
    """Ensure ROS_PACKAGE_PATH includes ancestors that contain meshes."""
    resolved = urdf_path.resolve()
    candidate_roots: List[Path] = []
    for depth in range(1, 5):
        if len(resolved.parents) > depth:
            candidate_roots.append(resolved.parents[depth])

    current = os.environ.get("ROS_PACKAGE_PATH", "")
    paths = [Path(p) for p in current.split(":") if p]
    updated = False
    for root in candidate_roots:
        if root.is_dir() and root not in paths:
            paths.insert(0, root)
            updated = True

    if updated:
        os.environ["ROS_PACKAGE_PATH"] = ":".join(str(p) for p in paths)


def _extract_link_index(name: str) -> Optional[int]:
    match = _LINK_INDEX_PATTERN.search(name)
    if match:
        try:
            return int(match.group(1))
        except ValueError:
            return None
    return None


def _should_auto_exclude_pair(name_a: str, name_b: str) -> bool:
    a_lower = name_a.lower()
    b_lower = name_b.lower()

    end_effector_tokens = ("finger", "hand")
    a_is_ee = any(token in a_lower for token in end_effector_tokens)
    b_is_ee = any(token in b_lower for token in end_effector_tokens)

    if a_is_ee and b_is_ee:
        return True

    idx_a = _extract_link_index(a_lower)
    idx_b = _extract_link_index(b_lower)

    if a_is_ee != b_is_ee:
        other_idx = idx_b if a_is_ee else idx_a
        if other_idx is not None and other_idx >= 5:
            return True
        return False

    if idx_a is None or idx_b is None:
        return False

    return abs(idx_a - idx_b) <= 2


def generate_auto_collision_exclusions(robot: embodik.RobotModel) -> List[Tuple[str, str]]:
    exclusions: List[Tuple[str, str]] = []
    for name_a, name_b in robot.get_collision_pair_names():
        if _should_auto_exclude_pair(name_a, name_b):
            exclusions.append((name_a, name_b))
    return exclusions


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=DEFAULT_VISER_PORT, help="Viser server port.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    # Load Panda URDF
    try:
        from robot_descriptions.panda_description import URDF_PATH as _URDF_PATH

        urdf_path = Path(_URDF_PATH)
    except ImportError as exc:
        raise RuntimeError(
            "robot_descriptions package is required. Install with:\n"
            "  pip install robot_descriptions"
        ) from exc

    ensure_ros_package_path(urdf_path)

    print(f"[demo] Loading Panda from: {urdf_path}")
    robot = embodik.RobotModel(str(urdf_path), floating_base=False)

    # Build and apply auto collision exclusions
    exclusions = generate_auto_collision_exclusions(robot)
    if exclusions:
        robot.apply_collision_exclusions(exclusions)
        print(f"[demo] Applied {len(exclusions)} auto collision exclusions")

    # Build solver
    solver = embodik.KinematicsSolver(robot)
    solver.dt = 0.01
    configure_solver_runtime_policy(solver)

    # Configure collision constraint
    solver.configure_collision_constraint(
        min_distance=DEFAULT_MIN_DIST,
        include_pairs=[],
        exclude_pairs=list(exclusions),
    )
    if hasattr(embodik, "CollisionTuningMode") and hasattr(solver, "set_collision_tuning_mode"):
        solver.set_collision_tuning_mode(embodik.CollisionTuningMode.BALANCED)

    if hasattr(solver, "enable_sphere_broadphase"):
        solver.enable_sphere_broadphase(True)
        print("[demo] Sphere broadphase enabled")

    # Add EE frame task
    frame_task = solver.add_frame_task("ee_task", TARGET_LINK)
    frame_task.priority = 0
    frame_task.weight = 1.0
    frame_task.solve_mode = embodik.TaskSolveMode.SCALE_ELASTIC
    frame_task.allow_min_error_fallback = False

    # Initial configuration
    nq = robot.nq
    q = np.zeros(nq, dtype=float)
    q[: len(DEFAULT_Q)] = DEFAULT_Q
    robot.update_configuration(q)

    # -----------------------------------------------------------------------
    # Viser setup
    # -----------------------------------------------------------------------
    server = viser.ViserServer(port=args.port)
    server.scene.add_grid("/ground", width=2, height=2)

    # ViserUrdf for robot visualization
    try:
        from robot_descriptions.loaders.yourdfpy import load_robot_description

        urdf_obj = load_robot_description("panda_description")
        urdf_vis = ViserUrdf(server, urdf_obj, root_node_name="/robot")
        _actuated_names = list(getattr(urdf_vis._urdf, "actuated_joint_names", []))
        _joint_names = robot.get_joint_names()
        _name_to_idx = {name: idx for idx, name in enumerate(_joint_names)}

        def make_visual_config(q_arm: np.ndarray) -> np.ndarray:
            if not _actuated_names:
                return q_arm
            cfg_vec = np.zeros(len(_actuated_names), dtype=float)
            for i, jname in enumerate(_actuated_names):
                idx = _name_to_idx.get(jname)
                if idx is not None and idx < q_arm.size:
                    cfg_vec[i] = q_arm[idx]
            return cfg_vec

        def update_vis(q_full: np.ndarray) -> None:
            urdf_vis.update_cfg(make_visual_config(q_full))

    except Exception as exc:
        print(f"[demo] ViserUrdf unavailable ({exc}); visualization will be skipped")
        urdf_vis = None

        def update_vis(_q: np.ndarray) -> None:  # type: ignore[misc]
            pass

    # Initial EE pose for transform controls
    initial_pose = robot.get_frame_pose(TARGET_LINK)
    init_quat_xyzw = r2q(initial_pose.rotation, order="xyzs")
    init_wxyz = (init_quat_xyzw[3], init_quat_xyzw[0], init_quat_xyzw[1], init_quat_xyzw[2])

    ik_target = server.scene.add_transform_controls(
        "/ik_target",
        scale=0.2,
        position=tuple(initial_pose.translation),
        wxyz=init_wxyz,
    )

    # World frame
    server.scene.add_frame("/world", show_axes=True, axes_length=0.15)

    update_vis(q)

    # -----------------------------------------------------------------------
    # GUI
    # -----------------------------------------------------------------------
    with server.gui.add_folder("Solver Params"):
        position_gain_slider = server.gui.add_slider(
            "Position Gain", min=0.1, max=50.0, step=0.1, initial_value=DEFAULT_POS_GAIN
        )
        global_min_dist_slider = server.gui.add_slider(
            "Global Min Distance (m)",
            min=0.01,
            max=0.20,
            step=0.005,
            initial_value=DEFAULT_MIN_DIST,
        )
        adaptive_dt_checkbox = server.gui.add_checkbox("Adaptive dt", initial_value=False)
        adaptive_dt_scale_slider = server.gui.add_slider(
            "Adaptive dt Max Scale", min=1.0, max=10.0, step=0.5, initial_value=5.0
        )

    LINK_A_OPTIONS = ["panda_link3", "panda_link4", "panda_link5", "panda_link6"]
    LINK_B_OPTIONS = ["panda_link4", "panda_link5", "panda_link6", "panda_hand"]

    with server.gui.add_folder("Per-Pair Override"):
        link_a_dropdown = server.gui.add_dropdown(
            "Link A", options=LINK_A_OPTIONS, initial_value=LINK_A_OPTIONS[0]
        )
        link_b_dropdown = server.gui.add_dropdown(
            "Link B", options=LINK_B_OPTIONS, initial_value=LINK_B_OPTIONS[-1]
        )
        pair_dist_slider = server.gui.add_slider(
            "Pair Min Distance (m)", min=0.01, max=0.30, step=0.005, initial_value=0.10
        )
        apply_btn = server.gui.add_button("Apply Override")
        clear_btn = server.gui.add_button("Clear Override")

    status_md = server.gui.add_markdown(
        "**Status:** --  \n**Min collision dist:** --  \n**Rejections:** --  \n**Per-pair overrides:** 0"
    )

    # -----------------------------------------------------------------------
    # Per-pair override callbacks
    # -----------------------------------------------------------------------

    @apply_btn.on_click
    def _(_evt) -> None:
        la = link_a_dropdown.value
        lb = link_b_dropdown.value
        dist = float(pair_dist_slider.value)
        if hasattr(solver, "set_collision_pair_min_distance"):
            solver.set_collision_pair_min_distance(la, lb, dist)
            print(f"[demo] Per-pair override set: {la} <-> {lb} @ {dist:.3f} m")
        else:
            print("[demo] set_collision_pair_min_distance not available in this build")

    @clear_btn.on_click
    def _(_evt) -> None:
        la = link_a_dropdown.value
        lb = link_b_dropdown.value
        if hasattr(solver, "clear_collision_pair_min_distance"):
            solver.clear_collision_pair_min_distance(la, lb)
            print(f"[demo] Per-pair override cleared: {la} <-> {lb}")
        else:
            print("[demo] clear_collision_pair_min_distance not available in this build")

    # -----------------------------------------------------------------------
    # Collision sphere markers
    # -----------------------------------------------------------------------
    collision_root = "/collision"
    col_sphere_a = server.scene.add_icosphere(
        f"{collision_root}/point_a",
        radius=0.012,
        color=(1.0, 0.2, 0.2),
        visible=False,
    )
    col_sphere_b = server.scene.add_icosphere(
        f"{collision_root}/point_b",
        radius=0.012,
        color=(0.2, 0.8, 1.0),
        visible=False,
    )
    col_line_handle = None

    # -----------------------------------------------------------------------
    # Step options
    # -----------------------------------------------------------------------
    step_opts = embodik.PositionStepOptions()

    print(f"[demo] Viser server started at http://localhost:{args.port}")
    print("[demo] Drag the transform controls to move the IK target.")

    # -----------------------------------------------------------------------
    # Main loop
    # -----------------------------------------------------------------------
    while True:
        # Build target pose from transform controls
        target_position = np.array(ik_target.position, dtype=float)
        target_wxyz = np.array(ik_target.wxyz, dtype=float)
        # viser wxyz -> xyzw for q2r
        target_xyzw = np.array(
            [target_wxyz[1], target_wxyz[2], target_wxyz[3], target_wxyz[0]], dtype=float
        )
        target_rotation = q2r(target_xyzw, order="xyzs")
        target_pose = Rt(R=target_rotation, t=target_position)

        # Update solver options from GUI
        step_opts.position_gain = float(position_gain_slider.value)
        step_opts.max_linear_speed = 2.0
        step_opts.adaptive_dt = bool(adaptive_dt_checkbox.value)
        if hasattr(step_opts, "adaptive_dt_max_scale"):
            step_opts.adaptive_dt_max_scale = float(adaptive_dt_scale_slider.value)

        # Update global min_distance
        if hasattr(solver, "set_collision_min_distance"):
            solver.set_collision_min_distance(float(global_min_dist_slider.value))

        # Solve
        result = solver.solve_position_step(q, target_pose, "ee_task", step_opts)

        if result.status == embodik.SolverStatus.SUCCESS:
            q = np.asarray(result.q_solution, dtype=float)
            robot.update_configuration(q)
        elif result.status == embodik.SolverStatus.COLLISION_VIOLATED:
            pass  # hold q — no safe step exists
        else:
            # INFEASIBLE, NUMERICAL_ERROR, etc. — hold q
            pass

        update_vis(q)

        # ---- Collision debug visuals ----------------------------------------
        collision_dist = float("nan")
        col_line_handle_local = col_line_handle  # capture for closure

        debug_info = None
        if hasattr(solver, "get_last_collision_debug"):
            debug_info = solver.get_last_collision_debug()
        elif hasattr(solver, "evaluate_collision_debug"):
            try:
                debug_info = solver.evaluate_collision_debug(q)
            except Exception:
                debug_info = None

        if debug_info is not None:
            point_a = np.array(debug_info.point_a_world, dtype=float)
            point_b = np.array(debug_info.point_b_world, dtype=float)
            collision_dist = float(debug_info.distance)

            col_sphere_a.position = tuple(point_a)
            col_sphere_b.position = tuple(point_b)
            col_sphere_a.visible = True
            col_sphere_b.visible = True

            # Redraw connecting line
            if col_line_handle is not None:
                try:
                    col_line_handle.remove()
                except Exception:
                    pass
            seg_points = np.array([[point_a, point_b]], dtype=float)
            col_line_handle = server.scene.add_line_segments(
                f"{collision_root}/link",
                points=seg_points,
                colors=np.array([[255, 100, 0]], dtype=np.uint8),
                line_width=3.0,
                visible=True,
            )
        else:
            col_sphere_a.visible = False
            col_sphere_b.visible = False
            if col_line_handle is not None:
                try:
                    col_line_handle.remove()
                except Exception:
                    pass
                col_line_handle = None

        # ---- Status markdown -----------------------------------------------
        rejection_count = int(getattr(result, "collision_rejection_count", 0))
        override_count = 0
        if hasattr(solver, "get_collision_pair_min_distance_overrides"):
            try:
                override_count = len(solver.get_collision_pair_min_distance_overrides())
            except Exception:
                override_count = 0

        dist_str = f"{collision_dist:.4f} m" if not np.isnan(collision_dist) else "N/A"
        status_md.content = (
            f"**Status:** {result.status.name}  \n"
            f"**Min collision dist:** {dist_str}  \n"
            f"**Rejections:** {rejection_count}  \n"
            f"**Per-pair overrides:** {override_count}"
        )

        time.sleep(0.01)


if __name__ == "__main__":
    main()
