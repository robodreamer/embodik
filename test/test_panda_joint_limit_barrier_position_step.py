"""Panda: joint-limit barrier vs. ``clear_joint_limit_barrier_task`` under ``solve_position_step``.

``solve_position_step`` wraps ``solve_velocity``; the C++ barrier injects extra
priority-1 objectives when joints lie outside the barrier deadband (see
barrier injection inside ``solve_velocity``).

``test_panda_narrowed_limits.py`` already shows the barrier helps **velocity**
tracking when limits are artificially tight. Here we add:

- **Teleop-like incremental targets** (small per-step EE moves toward a far goal):
  with narrowed limits, final return error is similar with or without the
  barrier, so the extra cost may be hard to justify for this path.
- **Solver-time regression**: with the barrier enabled, summed
  ``solver_computation_time_ms`` over many identical steps is measurably higher,
  matching the observation that the barrier adds non-trivial work whenever
  joints enter the outer portion of their ranges (deadband uses ``barrier_margin``).
"""

from __future__ import annotations

import numpy as np
import pytest

import embodik as eik

pytest.importorskip("robot_descriptions")

_PANDA_DEFAULT_Q = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785])
_PANDA_GRIPPER_EXTRA = np.array([0.05, 0.05])
_PANDA_EE_FRAME = "panda_hand"
_PANDA_ARM_DOFS = 7

# --- Solver / barrier (match alpha wheelbase teleop defaults where applicable) ---
_SOLVER_DT = 0.01
_SOLVER_DAMPING = 0.1
_BARRIER_MARGIN = 0.3
_BARRIER_GAIN = 1.0
_EE_TASK_PRIORITY = 0
_EE_TASK_WEIGHT = 10.0

# --- Incremental EE chase (teleop-like) ---
_CHASE_POSITION_GAIN = 15.0
_CHASE_ORIENTATION_GAIN = 15.0
_CHASE_MAX_LINEAR_SPEED = 0.25
_CHASE_MAX_ANGULAR_SPEED = 0.5
_CHASE_GOAL_CONVERGENCE_M = 0.004
_INCREMENTAL_STEP_CLIP_M = 0.012
_INCREMENTAL_TASK_POSITION_GAIN = 30.0
_INCREMENTAL_TASK_ORIENTATION_GAIN = 30.0
_DEFAULT_MAX_STEPS_PER_LEG = 500

# --- Fixed-target loop (solver-time probe) ---
_SUM_PROBE_EE_OFFSET = np.array([0.05, 0.0, 0.0], dtype=float)
_SUM_PROBE_POSITION_GAIN = 20.0
_SUM_PROBE_ORIENTATION_GAIN = 20.0
_SUM_PROBE_MAX_LINEAR_SPEED = 2.0
_SUM_PROBE_MAX_ANGULAR_SPEED = 2.0
_SUM_PROBE_TASK_POSITION_GAIN = 25.0
_SUM_PROBE_TASK_ORIENTATION_GAIN = 25.0

# --- Assertions ---
_MAX_RETURN_POSITION_ERROR_M = 0.02
_MAX_RETURN_ERROR_DELTA_M = 0.01
_MAX_EE_POSITION_DIFF_M = 0.02
_TIMING_MIN_TOTAL_MS = 0.05
_BARRIER_SOLVER_TIME_RATIO_MIN = 1.15
_N_STEPS_TIMING_COMPARE = 350
_FULL_LIMITS_CHASE_MAX_STEPS = 350
_NARROW_LIMIT_MARGIN_RAD = 0.25
_GOAL_OFFSET_NARROW_X_M = 0.08
_GOAL_OFFSET_FULL_LIMITS_M = np.array([0.12, 0.03, -0.02], dtype=float)


def _load_panda() -> tuple[eik.RobotModel, eik.KinematicsSolver]:
    from robot_descriptions.panda_description import URDF_PATH

    robot = eik.RobotModel(URDF_PATH, floating_base=False)
    q_init = np.concatenate([_PANDA_DEFAULT_Q, _PANDA_GRIPPER_EXTRA])
    robot.update_configuration(q_init)
    solver = eik.KinematicsSolver(robot)
    solver.dt = _SOLVER_DT
    return robot, solver


def _narrow_arm_limits(robot: eik.RobotModel, margin: float) -> None:
    q_lower_orig, q_upper_orig = robot.get_joint_limits()
    q_lower = q_lower_orig.copy()
    q_upper = q_upper_orig.copy()
    q_center = np.concatenate([_PANDA_DEFAULT_Q, _PANDA_GRIPPER_EXTRA])
    for i in range(_PANDA_ARM_DOFS):
        q_lower[i] = max(q_lower_orig[i], q_center[i] - margin)
        q_upper[i] = min(q_upper_orig[i], q_center[i] + margin)
    robot.set_joint_limits(q_lower, q_upper)


def _incremental_chase_return_error(
    *,
    use_barrier: bool,
    narrow_margin: float | None,
    goal_offset: np.ndarray,
    max_steps_per_leg: int = _DEFAULT_MAX_STEPS_PER_LEG,
    step_clip_m: float = _INCREMENTAL_STEP_CLIP_M,
) -> tuple[float, int, float, np.ndarray]:
    """Move EE toward ``p0+goal_offset`` in small steps, then back to ``p0``.

    Returns ``(return_position_error_m, infeasible_steps, summed_solver_time_ms, q_final)``.
    """
    robot, solver = _load_panda()
    q = np.concatenate([_PANDA_DEFAULT_Q, _PANDA_GRIPPER_EXTRA])
    robot.update_configuration(q)
    if narrow_margin is not None:
        _narrow_arm_limits(robot, narrow_margin)

    solver.enable_position_limits(True)
    solver.set_damping(_SOLVER_DAMPING)
    if use_barrier:
        solver.set_joint_limit_barrier_task(_BARRIER_MARGIN, _BARRIER_GAIN)
    else:
        solver.clear_joint_limit_barrier_task()

    solver.clear_tasks()
    solver.add_frame_task("ee", _PANDA_EE_FRAME)
    ee_task = solver.get_task("ee")
    ee_task.priority = _EE_TASK_PRIORITY
    ee_task.weight = _EE_TASK_WEIGHT

    robot.update_configuration(q)
    pose0 = robot.get_frame_pose(_PANDA_EE_FRAME)
    p0 = np.asarray(pose0.translation, dtype=float)
    R = np.asarray(pose0.rotation, dtype=float)
    goal_forward = p0 + goal_offset

    opts = eik.PositionStepOptions()
    opts.dt = _SOLVER_DT
    opts.max_steps = 1
    opts.position_gain = _CHASE_POSITION_GAIN
    opts.orientation_gain = _CHASE_ORIENTATION_GAIN
    opts.max_linear_speed = _CHASE_MAX_LINEAR_SPEED
    opts.max_angular_speed = _CHASE_MAX_ANGULAR_SPEED

    solver_ms = 0.0
    infeasible = 0

    def leg(goal: np.ndarray) -> None:
        nonlocal q, solver_ms, infeasible
        for _ in range(max_steps_per_leg):
            robot.update_configuration(q)
            pc = np.asarray(robot.get_frame_pose(_PANDA_EE_FRAME).translation, dtype=float)
            if np.linalg.norm(pc - goal) < _CHASE_GOAL_CONVERGENCE_M:
                break
            step = goal - pc
            nrm = np.linalg.norm(step)
            if nrm > step_clip_m:
                step = step * (step_clip_m / nrm)
            target_p = pc + step
            T = np.eye(4, dtype=float)
            T[:3, :3] = R
            T[:3, 3] = target_p
            tt = eik.TaskTarget(
                "ee", T, _INCREMENTAL_TASK_POSITION_GAIN, _INCREMENTAL_TASK_ORIENTATION_GAIN
            )
            res = solver.solve_position_step(q, [tt], opts)
            scm = getattr(res, "solver_computation_time_ms", None)
            if scm is not None:
                solver_ms += float(scm)
            if res.status != eik.SolverStatus.SUCCESS:
                infeasible += 1
            q = np.asarray(res.q_solution, dtype=float)

    leg(goal_forward)
    leg(p0)

    robot.update_configuration(q)
    p_end = np.asarray(robot.get_frame_pose(_PANDA_EE_FRAME).translation, dtype=float)
    return float(np.linalg.norm(p_end - p0)), infeasible, solver_ms, q.copy()


def _sum_solver_time_identical_steps(
    *,
    use_barrier: bool,
    n_steps: int,
) -> float:
    """Fixed far target (infeasible in one step) but identical each call — isolates per-step cost."""
    robot, solver = _load_panda()
    q = np.concatenate([_PANDA_DEFAULT_Q, _PANDA_GRIPPER_EXTRA])
    robot.update_configuration(q)
    solver.enable_position_limits(True)
    solver.set_damping(_SOLVER_DAMPING)
    if use_barrier:
        solver.set_joint_limit_barrier_task(_BARRIER_MARGIN, _BARRIER_GAIN)
    else:
        solver.clear_joint_limit_barrier_task()
    solver.clear_tasks()
    solver.add_frame_task("ee", _PANDA_EE_FRAME)
    solver.get_task("ee").priority = _EE_TASK_PRIORITY
    solver.get_task("ee").weight = _EE_TASK_WEIGHT

    robot.update_configuration(q)
    pose0 = robot.get_frame_pose(_PANDA_EE_FRAME)
    R = np.asarray(pose0.rotation, dtype=float)
    p0 = np.asarray(pose0.translation, dtype=float)
    pf = p0 + _SUM_PROBE_EE_OFFSET
    T = np.eye(4, dtype=float)
    T[:3, :3] = R
    T[:3, 3] = pf
    tt = eik.TaskTarget(
        "ee", T, _SUM_PROBE_TASK_POSITION_GAIN, _SUM_PROBE_TASK_ORIENTATION_GAIN
    )
    opts = eik.PositionStepOptions()
    opts.dt = _SOLVER_DT
    opts.max_steps = 1
    opts.position_gain = _SUM_PROBE_POSITION_GAIN
    opts.orientation_gain = _SUM_PROBE_ORIENTATION_GAIN
    opts.max_linear_speed = _SUM_PROBE_MAX_LINEAR_SPEED
    opts.max_angular_speed = _SUM_PROBE_MAX_ANGULAR_SPEED

    total = 0.0
    for _ in range(n_steps):
        res = solver.solve_position_step(q, [tt], opts)
        scm = getattr(res, "solver_computation_time_ms", None)
        if scm is not None:
            total += float(scm)
        q = np.asarray(res.q_solution, dtype=float)
        robot.update_configuration(q)
    return total


@pytest.mark.parametrize("narrow_m", [_NARROW_LIMIT_MARGIN_RAD])
def test_incremental_position_step_barrier_does_not_improve_return_error_panda(
    narrow_m: float,
):
    """Under incremental position stepping, barrier vs. off yields similar home error.

    This does **not** contradict ``test_panda_narrowed_limits`` (velocity mode);
    position stepping already integrates small feasible twists each cycle.
    """
    off_e, off_inf, _, _ = _incremental_chase_return_error(
        use_barrier=False,
        narrow_margin=narrow_m,
        goal_offset=np.array([_GOAL_OFFSET_NARROW_X_M, 0.0, 0.0], dtype=float),
    )
    on_e, on_inf, _, _ = _incremental_chase_return_error(
        use_barrier=True,
        narrow_margin=narrow_m,
        goal_offset=np.array([_GOAL_OFFSET_NARROW_X_M, 0.0, 0.0], dtype=float),
    )
    assert off_inf == 0 and on_inf == 0
    assert off_e < _MAX_RETURN_POSITION_ERROR_M and on_e < _MAX_RETURN_POSITION_ERROR_M
    assert abs(off_e - on_e) < _MAX_RETURN_ERROR_DELTA_M


def test_barrier_raises_summed_solver_time_ms_on_repeated_position_steps():
    """Barrier path should report higher core solver time than cleared barrier."""
    t_off = _sum_solver_time_identical_steps(
        use_barrier=False, n_steps=_N_STEPS_TIMING_COMPARE
    )
    t_on = _sum_solver_time_identical_steps(
        use_barrier=True, n_steps=_N_STEPS_TIMING_COMPARE
    )
    assert t_off > _TIMING_MIN_TOTAL_MS and t_on > _TIMING_MIN_TOTAL_MS, (
        "timing fields too small for assertion"
    )
    assert t_on >= _BARRIER_SOLVER_TIME_RATIO_MIN * t_off, (
        f"expected barrier-on solver time higher: off={t_off:.3f} ms on={t_on:.3f} ms"
    )


def test_full_limits_incremental_chase_ee_matches_despite_nullspace_drift():
    """Same EE return quality; joint-space paths can differ (nullspace / barrier)."""
    err_off, inf_off, _, q_off = _incremental_chase_return_error(
        use_barrier=False,
        narrow_margin=None,
        goal_offset=_GOAL_OFFSET_FULL_LIMITS_M,
        max_steps_per_leg=_FULL_LIMITS_CHASE_MAX_STEPS,
    )
    err_on, inf_on, _, q_on = _incremental_chase_return_error(
        use_barrier=True,
        narrow_margin=None,
        goal_offset=_GOAL_OFFSET_FULL_LIMITS_M,
        max_steps_per_leg=_FULL_LIMITS_CHASE_MAX_STEPS,
    )
    assert inf_off == 0 and inf_on == 0
    assert err_off < _MAX_RETURN_POSITION_ERROR_M and err_on < _MAX_RETURN_POSITION_ERROR_M
    assert abs(err_off - err_on) < _MAX_RETURN_ERROR_DELTA_M

    probe, _ = _load_panda()
    probe.update_configuration(q_off)
    p_off = np.asarray(probe.get_frame_pose(_PANDA_EE_FRAME).translation, dtype=float)
    probe.update_configuration(q_on)
    p_on = np.asarray(probe.get_frame_pose(_PANDA_EE_FRAME).translation, dtype=float)
    assert np.linalg.norm(p_off - p_on) < _MAX_EE_POSITION_DIFF_M
