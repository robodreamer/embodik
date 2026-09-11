#!/usr/bin/env python3
"""Model-derived GPU WBC across many independently targeted robot worlds.

This showcase uses EmbodiK's Newton kinematics and Warp SRINV backend for the
actual solve. It visualizes only a small sample of the solved worlds so browser
rendering does not hide solver throughput.

Examples:
    python examples/parallel_trajectory_tracking.py --robot panda --worlds 1024
    python examples/parallel_trajectory_tracking.py --robot ai-worker --worlds 1024
    python examples/parallel_trajectory_tracking.py --robot g1 --worlds 1024
    python examples/parallel_trajectory_tracking.py --robot panda --headless --steps 100
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import subprocess
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np

import embodik
from embodik.gpu.wbc import GpuWbcMultiFrameSolver, derive_frames_active_joint_names

try:
    from example_helpers.common_bimanual_model_utils import (
        default_common_bimanual_ik_joint_names,
        resolve_common_bimanual_frames,
    )
    from example_helpers.common_bimanual_teleop_app import (
        _apply_named_joint_seed,
        _apply_soft_lift_margin,
        _default_seed_for_joint_names,
        _prepare_viewer_urdf_path,
    )
    from example_helpers.g1_ik_runtime import _apply_g1_soft_knee_seed
    from example_helpers.g1_model_utils import (
        create_g1_robot_model,
        prepare_g1_viewer_urdf_path,
        resolve_frames_for_g1_base_mode,
        resolve_g1_collision_urdf_path,
        resolve_g1_urdf_path,
    )
    from example_helpers.public_ai_worker_paths import resolve_public_ai_worker_urdf_paths
    from example_helpers.visualization_helpers import make_visual_config_mapper
except ModuleNotFoundError as exc:
    if exc.name != "example_helpers" and not str(exc.name).startswith("example_helpers."):
        raise
    from examples.example_helpers.common_bimanual_model_utils import (
        default_common_bimanual_ik_joint_names,
        resolve_common_bimanual_frames,
    )
    from examples.example_helpers.common_bimanual_teleop_app import (
        _apply_named_joint_seed,
        _apply_soft_lift_margin,
        _default_seed_for_joint_names,
        _prepare_viewer_urdf_path,
    )
    from examples.example_helpers.g1_ik_runtime import _apply_g1_soft_knee_seed
    from examples.example_helpers.g1_model_utils import (
        create_g1_robot_model,
        prepare_g1_viewer_urdf_path,
        resolve_frames_for_g1_base_mode,
        resolve_g1_collision_urdf_path,
        resolve_g1_urdf_path,
    )
    from examples.example_helpers.public_ai_worker_paths import (
        resolve_public_ai_worker_urdf_paths,
    )
    from examples.example_helpers.visualization_helpers import make_visual_config_mapper


@dataclass(frozen=True)
class RobotProfile:
    """All model-specific data needed by the generic batch loop."""

    key: str
    label: str
    model_urdf: Path
    visual_urdf: Path
    visual_mesh_dir: Path
    robot: object
    default_configuration: np.ndarray
    frames: tuple[str, ...]
    moving_frames: tuple[bool, ...]
    motion_scale_m: float
    root_height_m: float


def _joint_index_map(robot: object) -> dict[str, int]:
    result: dict[str, int] = {}
    for name in robot.get_joint_names():
        try:
            result[str(name)] = int(robot.get_joint_config_index(name))
        except Exception:
            continue
    return result


def _panda_profile() -> RobotProfile:
    from robot_descriptions.panda_description import URDF_PATH

    urdf = Path(URDF_PATH).expanduser().resolve()
    full_robot = embodik.RobotModel(str(urdf), floating_base=False)
    active_names = derive_frames_active_joint_names(full_robot, ("panda_hand",))
    robot = embodik.RobotModel(
        str(urdf), actuated_joint_names=list(active_names), floating_base=False
    )
    q = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785], dtype=float)
    return RobotProfile(
        key="panda",
        label="Franka Panda",
        model_urdf=urdf,
        visual_urdf=urdf,
        visual_mesh_dir=urdf.parent,
        robot=robot,
        default_configuration=q,
        frames=("panda_hand",),
        moving_frames=(True,),
        motion_scale_m=0.045,
        root_height_m=0.0,
    )


def _ai_worker_profile(args: argparse.Namespace) -> RobotProfile:
    visual_urdf, collision_urdf = resolve_public_ai_worker_urdf_paths(
        variant=args.ai_worker_variant,
        ai_worker_root=args.ai_worker_root,
        urdf=args.urdf,
        collision_urdf=args.collision_urdf,
        # Headless runs only need the bundled model. The viewer fetches the
        # public ROBOTIS visual meshes when no explicit source is provided.
        allow_bundled_base_fallback=args.headless,
    )
    model_urdf = collision_urdf or visual_urdf
    full_robot = embodik.RobotModel(str(model_urdf), floating_base=False)
    reduced_names = default_common_bimanual_ik_joint_names(full_robot.get_joint_names())
    robot = embodik.RobotModel(
        str(model_urdf), actuated_joint_names=reduced_names, floating_base=False
    )
    frame_map = resolve_common_bimanual_frames(robot.get_frame_names())
    frames = (frame_map["right_tool"], frame_map["left_tool"])
    active_names = derive_frames_active_joint_names(robot, frames)
    if tuple(active_names) != tuple(reduced_names):
        robot = embodik.RobotModel(
            str(model_urdf), actuated_joint_names=list(active_names), floating_base=False
        )
    q = np.asarray(robot.neutral_configuration(), dtype=float)
    q_lo, q_hi = robot.get_joint_limits()
    joint_indices = _joint_index_map(robot)
    q = _apply_named_joint_seed(
        q,
        joint_indices,
        np.asarray(q_lo, dtype=float),
        np.asarray(q_hi, dtype=float),
        _default_seed_for_joint_names(list(robot.get_joint_names())),
    )
    q = _apply_soft_lift_margin(
        q,
        joint_name_to_cfg=joint_indices,
        q_lo=np.asarray(q_lo, dtype=float),
        q_hi=np.asarray(q_hi, dtype=float),
    )
    viewer_urdf = _prepare_viewer_urdf_path(
        visual_urdf,
        allow_recursive_mesh_fallback=visual_urdf.resolve() != model_urdf.resolve(),
    )
    return RobotProfile(
        key="ai-worker",
        label=f"ROBOTIS AI Worker {args.ai_worker_variant.upper()}",
        model_urdf=model_urdf,
        visual_urdf=viewer_urdf,
        visual_mesh_dir=visual_urdf.parent,
        robot=robot,
        default_configuration=q,
        frames=frames,
        moving_frames=(True, True),
        motion_scale_m=0.035,
        root_height_m=0.0,
    )


def _g1_profile() -> RobotProfile:
    model_urdf = resolve_g1_collision_urdf_path()
    visual_urdf = resolve_g1_urdf_path()
    robot = create_g1_robot_model(floating_base=False, reduced_ik=True)
    frame_map = resolve_frames_for_g1_base_mode(robot.get_frame_names())
    frames = (
        frame_map["right_palm"],
        frame_map["left_palm"],
        frame_map["right_ankle"],
        frame_map["left_ankle"],
    )
    q = _apply_g1_soft_knee_seed(robot, robot.neutral_configuration())
    return RobotProfile(
        key="g1",
        label="Unitree G1",
        model_urdf=model_urdf,
        visual_urdf=prepare_g1_viewer_urdf_path(visual_urdf),
        visual_mesh_dir=visual_urdf.parent,
        robot=robot,
        default_configuration=np.asarray(q, dtype=float),
        frames=frames,
        moving_frames=(True, True, False, False),
        motion_scale_m=0.025,
        root_height_m=0.82,
    )


def load_robot_profile(args: argparse.Namespace) -> RobotProfile:
    if args.robot == "panda":
        return _panda_profile()
    if args.robot == "ai-worker":
        return _ai_worker_profile(args)
    if args.robot == "g1":
        return _g1_profile()
    raise ValueError(f"unsupported robot profile: {args.robot}")


def _initial_targets(profile: RobotProfile) -> np.ndarray:
    profile.robot.update_configuration(profile.default_configuration)
    rows: list[np.ndarray] = []
    for frame in profile.frames:
        pose = profile.robot.get_frame_pose(frame)
        xyzw = np.asarray(embodik.r2q(pose.rotation, order="xyzs"), dtype=np.float32)
        rows.append(
            np.concatenate(
                (np.asarray(pose.translation, dtype=np.float32), xyzw[[3, 0, 1, 2]])
            )
        )
    return np.asarray(rows, dtype=np.float32)


def _make_targets(torch, base_targets, phase, profile: RobotProfile, elapsed_s: float):
    targets = base_targets.unsqueeze(0).expand(phase.shape[0], -1, -1).clone()
    for frame_index, moving in enumerate(profile.moving_frames):
        if not moving:
            continue
        angle = phase + float(elapsed_s) * (0.65 + 0.11 * frame_index)
        side = -1.0 if frame_index % 2 else 1.0
        targets[:, frame_index, 0] += profile.motion_scale_m * torch.sin(angle)
        targets[:, frame_index, 1] += (
            side * profile.motion_scale_m * 0.65 * torch.cos(angle * 1.3)
        )
        targets[:, frame_index, 2] += profile.motion_scale_m * 0.45 * torch.sin(
            angle * 0.7 + frame_index
        )
    return targets


def _active_configuration(profile: RobotProfile, active_names: tuple[str, ...]) -> np.ndarray:
    indices = [int(profile.robot.get_joint_config_index(name)) for name in active_names]
    return profile.default_configuration[np.asarray(indices, dtype=int)].astype(np.float32)


def build_solver(profile: RobotProfile, args: argparse.Namespace):
    active_names = derive_frames_active_joint_names(profile.robot, profile.frames)
    q0 = _active_configuration(profile, active_names)
    solver = GpuWbcMultiFrameSolver.from_robot(
        profile.model_urdf,
        args.cache_dir,
        robot=profile.robot,
        robot_name=profile.key,
        frames=profile.frames,
        active_joint_names=active_names,
        default_configuration=profile.default_configuration,
        frame_task_dimensions=(6,) * len(profile.frames),
        frame_position_gains=(8.0,) * len(profile.frames),
        frame_orientation_gains=(4.0,) * len(profile.frames),
        solver_backend="warp_srinv",
        batch_size=args.worlds,
        iterations=args.iterations,
        dt=args.dt,
        max_linear_speed=1.0,
        max_angular_speed=1.5,
    )
    return solver, active_names, q0


def _source_revision() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=Path(__file__).resolve().parent,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except Exception:
        return "unknown"


def _summary(
    *, profile: RobotProfile, args: argparse.Namespace, solver, timings_ms: list[float]
) -> dict[str, object]:
    ordered = sorted(timings_ms)
    p95_index = max(0, math.ceil(0.95 * len(ordered)) - 1)
    mean_ms = statistics.fmean(ordered)
    return {
        "schema": "embodik.gpu-wbc.parallel-showcase.v1",
        "revision": _source_revision(),
        "robot": profile.key,
        "robot_label": profile.label,
        "backend": "newton+warp_srinv",
        "actual_device": solver.device_label,
        "worlds": args.worlds,
        "visible_worlds": 0 if args.headless else min(args.show, args.worlds),
        "tasks_per_world": len(profile.frames),
        "task_rows_per_world": 6 * len(profile.frames),
        "active_configuration_dim": solver.configuration_dim,
        "iterations_per_step": args.iterations,
        "warmup_steps": args.warmup_steps,
        "measured_steps": len(ordered),
        "solve_only_mean_ms": mean_ms,
        "solve_only_p50_ms": statistics.median(ordered),
        "solve_only_p95_ms": ordered[p95_index],
        "worlds_per_second_at_mean": args.worlds / (mean_ms * 1e-3),
        "measurement_scope": "CUDA solve only; target generation and visualization excluded",
        "features": [
            "model-derived dimensions",
            "device-resident batch state",
            "independent target phase per world",
            "joint position and velocity bounds",
        ],
    }


def run_headless(profile: RobotProfile, args: argparse.Namespace) -> dict[str, object]:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("this showcase requires CUDA; there is no CPU fallback")
    solver, _active_names, q0 = build_solver(profile, args)
    device = torch.device(solver.device_label)
    q = torch.as_tensor(q0, dtype=torch.float32, device=device).repeat(args.worlds, 1)
    base_targets = torch.as_tensor(_initial_targets(profile), device=device)
    phase = torch.linspace(0.0, 2.0 * math.pi, args.worlds + 1, device=device)[:-1]
    timings: list[float] = []
    total_steps = args.warmup_steps + args.steps
    for step in range(total_steps):
        targets = _make_targets(torch, base_targets, phase, profile, step * args.dt)
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        result = solver.solve_device_batch(q, targets)
        end.record()
        end.synchronize()
        if result.fallback_used or not result.actual_device.startswith("cuda"):
            raise RuntimeError("GPU attribution contract failed")
        if not bool(torch.isfinite(result.q_solution).all().item()):
            raise RuntimeError("non-finite GPU solution")
        q = result.q_solution
        if step >= args.warmup_steps:
            timings.append(float(start.elapsed_time(end)))
    summary = _summary(profile=profile, args=args, solver=solver, timings_ms=timings)
    print(json.dumps(summary, indent=2, sort_keys=True))
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


def _grid_positions(count: int, spacing: float, root_height: float) -> list[np.ndarray]:
    columns = max(1, math.ceil(math.sqrt(count)))
    rows = math.ceil(count / columns)
    return [
        np.array(
            [
                (index % columns - (columns - 1) / 2.0) * spacing,
                (index // columns - (rows - 1) / 2.0) * spacing,
                root_height,
            ],
            dtype=float,
        )
        for index in range(count)
    ]


def run_visualization(profile: RobotProfile, args: argparse.Namespace) -> None:
    import torch
    import viser
    import yourdfpy
    from viser.extras import ViserUrdf

    if not torch.cuda.is_available():
        raise RuntimeError("this showcase requires CUDA; there is no CPU fallback")
    solver, _active_names, q0 = build_solver(profile, args)
    device = torch.device(solver.device_label)
    q = torch.as_tensor(q0, dtype=torch.float32, device=device).repeat(args.worlds, 1)
    base_targets = torch.as_tensor(_initial_targets(profile), device=device)
    phase = torch.linspace(0.0, 2.0 * math.pi, args.worlds + 1, device=device)[:-1]

    visible_worlds = min(args.show, args.worlds)
    spacing = args.spacing or {"panda": 1.25, "ai-worker": 2.2, "g1": 1.35}[profile.key]
    positions = _grid_positions(visible_worlds, spacing, profile.root_height_m)
    server = viser.ViserServer(port=args.port)
    width = max(4.0, spacing * math.ceil(math.sqrt(visible_worlds)) + 1.5)
    server.scene.add_grid("/ground", width=width, height=width)

    urdf = yourdfpy.URDF.load(str(profile.visual_urdf), mesh_dir=profile.visual_mesh_dir)
    visuals: list[tuple[ViserUrdf, object]] = []
    target_handles: list[list[object]] = []
    for index, position in enumerate(positions):
        root = f"/worlds/{index:03d}"
        server.scene.add_frame(root, position=tuple(position), show_axes=False)
        visual = ViserUrdf(server, urdf, root_node_name=f"{root}/robot")
        mapper = make_visual_config_mapper(profile.robot, visual)
        visual.update_cfg(mapper(profile.default_configuration))
        visuals.append((visual, mapper))
        handles = []
        for frame_index, _frame in enumerate(profile.frames):
            if profile.moving_frames[frame_index]:
                handles.append(
                    server.scene.add_icosphere(
                        f"{root}/targets/{frame_index}",
                        radius=0.028 if profile.key == "panda" else 0.04,
                        color=(35, 168, 255) if frame_index % 2 == 0 else (255, 116, 74),
                    )
                )
            else:
                handles.append(None)
        target_handles.append(handles)

    server.gui.add_markdown(
        "## GPU WBC parallel worlds\n"
        "Newton kinematics + Warp directional SRINV. The browser renders only "
        "a sample; every listed world is solved on CUDA."
    )
    server.gui.add_text("Robot", initial_value=profile.label, disabled=True)
    server.gui.add_text("CUDA worlds", initial_value=f"{args.worlds:,}", disabled=True)
    server.gui.add_text(
        "Pose tasks / world", initial_value=str(len(profile.frames)), disabled=True
    )
    p50_text = server.gui.add_text("Solve p50", initial_value="warming up", disabled=True)
    p95_text = server.gui.add_text("Solve p95", initial_value="warming up", disabled=True)
    throughput_text = server.gui.add_text(
        "Throughput", initial_value="warming up", disabled=True
    )
    server.gui.add_text("Timing scope", initial_value="CUDA solve only", disabled=True)

    timings: deque[float] = deque(maxlen=max(30, args.stats_window))
    step = 0
    started = time.perf_counter()
    frame_period = 1.0 / max(args.fps, 1.0)
    print(
        f"Serving {profile.label}: {args.worlds} CUDA worlds, "
        f"{visible_worlds} visible at http://localhost:{args.port}"
    )
    try:
        while args.steps <= 0 or step < args.steps:
            frame_started = time.perf_counter()
            elapsed = time.perf_counter() - started
            targets = _make_targets(torch, base_targets, phase, profile, elapsed)
            begin = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            begin.record()
            result = solver.solve_device_batch(q, targets)
            end.record()
            end.synchronize()
            if result.fallback_used or not result.actual_device.startswith("cuda"):
                raise RuntimeError("GPU attribution contract failed")
            q = result.q_solution
            timings.append(float(begin.elapsed_time(end)))

            q_host = q[:visible_worlds].detach().cpu().numpy()
            target_host = targets[:visible_worlds, :, :3].detach().cpu().numpy()
            for index, (visual, mapper) in enumerate(visuals):
                merged = solver.merge_active_configuration(
                    profile.default_configuration, q_host[index]
                )
                visual.update_cfg(mapper(merged))
                for frame_index, handle in enumerate(target_handles[index]):
                    if handle is not None:
                        handle.position = tuple(target_host[index, frame_index])

            ordered = sorted(timings)
            p50 = statistics.median(ordered)
            p95 = ordered[max(0, math.ceil(0.95 * len(ordered)) - 1)]
            p50_text.value = f"{p50:.3f} ms"
            p95_text.value = f"{p95:.3f} ms"
            throughput_text.value = f"{args.worlds / (p50 * 1e-3):,.0f} worlds/s"
            step += 1
            remaining = frame_period - (time.perf_counter() - frame_started)
            if remaining > 0:
                time.sleep(remaining)
    except KeyboardInterrupt:
        pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robot", choices=("panda", "ai-worker", "g1"), default="panda")
    parser.add_argument("--worlds", type=int, default=1024)
    parser.add_argument("--show", type=int, default=9, help="Worlds rendered in Viser")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--steps", type=int, default=0, help="0 runs the viewer until Ctrl+C")
    parser.add_argument("--warmup-steps", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=2)
    parser.add_argument("--dt", type=float, default=0.02)
    parser.add_argument("--fps", type=float, default=20.0)
    parser.add_argument("--stats-window", type=int, default=60)
    parser.add_argument("--spacing", type=float)
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--cache-dir", type=Path, default=Path("build/gpu-wbc-cache"))
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--ai-worker-variant", choices=("sg2", "bg2"), default="sg2")
    parser.add_argument("--ai-worker-root", type=Path)
    parser.add_argument("--urdf", type=Path)
    parser.add_argument("--collision-urdf", type=Path)
    args = parser.parse_args()
    if args.worlds < 1 or args.show < 1:
        parser.error("--worlds and --show must be positive")
    if args.steps < 0 or args.warmup_steps < 0 or args.iterations < 1:
        parser.error("steps/warmup must be nonnegative and iterations must be positive")
    if args.headless and args.steps < 1:
        parser.error("--headless requires --steps greater than zero")
    return args


def main() -> None:
    args = parse_args()
    profile = load_robot_profile(args)
    if args.headless:
        run_headless(profile, args)
    else:
        run_visualization(profile, args)


if __name__ == "__main__":
    main()
