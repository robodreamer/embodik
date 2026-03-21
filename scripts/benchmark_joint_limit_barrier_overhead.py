#!/usr/bin/env python3
"""Rough A/B timing for joint-limit barrier under ``solve_position_step`` (Panda).

Usage (from repo root, with pixi):
  pixi run python scripts/benchmark_joint_limit_barrier_overhead.py

Prints median wall time and summed ``solver_computation_time_ms`` over repeated
steps. Variance is normal; compare across git revisions for before/after C++
changes (cache + single-pass barrier injection).

Requires: robot_descriptions (same as unit tests).
"""

from __future__ import annotations

import statistics
import time

import numpy as np

import embodik as eik

try:
    from robot_descriptions.panda_description import URDF_PATH
except ImportError as exc:
    raise SystemExit("Install robot_descriptions (pixi default env includes it).") from exc

_PANDA_DEFAULT_Q = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785])
_GRIP = np.array([0.05, 0.05])
_EE_FRAME = "panda_hand"

_SOLVER_DT = 0.01
_SOLVER_DAMPING = 0.1
_BARRIER_MARGIN = 0.3
_BARRIER_GAIN = 1.0
_EE_TASK_PRIORITY = 0
_EE_TASK_WEIGHT = 10.0
_EE_OFFSET_PROBE = np.array([0.05, 0.0, 0.0], dtype=float)
_TASK_POSITION_GAIN = 25.0
_TASK_ORIENTATION_GAIN = 25.0
_POSITION_GAIN = 20.0
_ORIENTATION_GAIN = 20.0
_MAX_LINEAR_SPEED = 2.0
_MAX_ANGULAR_SPEED = 2.0

_DEFAULT_N_STEPS = 800
_DEFAULT_N_TRIALS = 7
_RATIO_PRINT_MIN_MS = 0.01


def _run(*, use_barrier: bool, n_steps: int, n_trials: int) -> tuple[float, float]:
    med_wall: list[float] = []
    med_solver: list[float] = []
    for _ in range(n_trials):
        robot = eik.RobotModel(URDF_PATH, floating_base=False)
        q = np.concatenate([_PANDA_DEFAULT_Q, _GRIP])
        robot.update_configuration(q)
        solver = eik.KinematicsSolver(robot)
        solver.dt = _SOLVER_DT
        solver.enable_position_limits(True)
        solver.set_damping(_SOLVER_DAMPING)
        if use_barrier:
            solver.set_joint_limit_barrier_task(_BARRIER_MARGIN, _BARRIER_GAIN)
        else:
            solver.clear_joint_limit_barrier_task()
        solver.clear_tasks()
        solver.add_frame_task("ee", _EE_FRAME)
        solver.get_task("ee").priority = _EE_TASK_PRIORITY
        solver.get_task("ee").weight = _EE_TASK_WEIGHT
        robot.update_configuration(q)
        pose0 = robot.get_frame_pose(_EE_FRAME)
        R = np.asarray(pose0.rotation, float)
        p0 = np.asarray(pose0.translation, float)
        pf = p0 + _EE_OFFSET_PROBE
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = pf
        tt = eik.TaskTarget("ee", T, _TASK_POSITION_GAIN, _TASK_ORIENTATION_GAIN)
        opts = eik.PositionStepOptions()
        opts.dt = _SOLVER_DT
        opts.max_steps = 1
        opts.position_gain = _POSITION_GAIN
        opts.orientation_gain = _ORIENTATION_GAIN
        opts.max_linear_speed = _MAX_LINEAR_SPEED
        opts.max_angular_speed = _MAX_ANGULAR_SPEED
        scm = 0.0
        t0 = time.perf_counter()
        for _ in range(n_steps):
            r = solver.solve_position_step(q, [tt], opts)
            s = getattr(r, "solver_computation_time_ms", None)
            if s is not None:
                scm += float(s)
            q = np.asarray(r.q_solution, float)
            robot.update_configuration(q)
        wall = (time.perf_counter() - t0) * 1000
        med_wall.append(wall)
        med_solver.append(scm)
    return statistics.median(med_solver), statistics.median(med_wall)


def main() -> None:
    off_s, off_w = _run(
        use_barrier=False, n_steps=_DEFAULT_N_STEPS, n_trials=_DEFAULT_N_TRIALS
    )
    on_s, on_w = _run(
        use_barrier=True, n_steps=_DEFAULT_N_STEPS, n_trials=_DEFAULT_N_TRIALS
    )
    print(
        f"Panda position_step x{_DEFAULT_N_STEPS}, "
        f"median of {_DEFAULT_N_TRIALS} trials"
    )
    print(f"  barrier OFF: solver_ms_sum={off_s:.2f}  wall_ms={off_w:.2f}")
    print(f"  barrier ON:  solver_ms_sum={on_s:.2f}  wall_ms={on_w:.2f}")
    if off_s > _RATIO_PRINT_MIN_MS:
        print(f"  ON/OFF ratio (solver_ms): {on_s/off_s:.3f}x")
    if off_w > _RATIO_PRINT_MIN_MS:
        print(f"  ON/OFF ratio (wall_ms):   {on_w/off_w:.3f}x")


if __name__ == "__main__":
    main()
