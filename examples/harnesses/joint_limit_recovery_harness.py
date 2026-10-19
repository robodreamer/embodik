"""Reproduce joint-limit redistribution and recovery using only the CPU solver.

Run with ``pixi run python examples/harnesses/joint_limit_recovery_harness.py``.
The generated two-slider robot requires no downloaded model or GPU dependencies.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from scipy.optimize import linprog, lsq_linear

import embodik as eik

DT = 0.01
TARGET_SPEED = 0.1
BOUND_TOLERANCE = 1e-8
ROBOT_URDF = """<robot name="joint_limit_recovery">
  <link name="base"/><link name="middle"/><link name="tool"/>
  <joint name="free_slider" type="prismatic">
    <parent link="base"/><child link="middle"/><axis xyz="1 0 0"/>
    <limit lower="-2" upper="2" effort="100" velocity="2"/>
  </joint>
  <joint name="blocked_slider" type="prismatic">
    <parent link="middle"/><child link="tool"/><axis xyz="0 1 0"/>
    <limit lower="-2" upper="0" effort="100" velocity="2"/>
  </joint>
</robot>
"""


def redundant_task_case() -> dict[str, object]:
    """A saturated coordinate can be replaced by another redundant coordinate."""
    jacobian = np.ones((1, 2))
    target = np.ones(1)
    lower = np.full(2, -2.0)
    upper = np.array([2.0, 0.0])
    result = eik.computeMultiObjectiveVelocitySolutionEigen(
        [target], [jacobian], np.eye(2), lower, upper
    )
    velocity = np.asarray(result.solution)
    unconstrained = np.linalg.pinv(jacobian) @ target
    # The second component's outward headroom is exactly zero. Uniform scaling
    # therefore holds both components; this is an algebraic reference, not a
    # GPU backend invocation or a benchmark.
    uniformly_scaled = np.zeros_like(unconstrained)
    return {
        "task": "v_free + v_blocked = 1, v_blocked <= 0",
        "unconstrained_velocity": unconstrained.tolist(),
        "uniform_scaling_reference_velocity": uniformly_scaled.tolist(),
        "cpu_velocity": velocity.tolist(),
        "cpu_status": result.status.name,
        "cpu_task_scales": list(result.task_scales),
        "achieved_task": (jacobian @ velocity).tolist(),
        "bounds_satisfied": bool(
            np.all(velocity >= lower - BOUND_TOLERANCE)
            and np.all(velocity <= upper + BOUND_TOLERANCE)
        ),
    }


def conflicting_task_case(
    urdf: Path,
    mode: eik.TaskSolveMode,
    *,
    task_fallback: bool,
    weighted_fallback: bool,
) -> dict[str, object]:
    """A Cartesian request contains both a feasible and an impossible axis."""
    robot = eik.RobotModel(str(urdf), floating_base=False)
    solver = eik.KinematicsSolver(robot)
    solver.dt = DT
    solver.enable_position_limits(True)
    solver.enable_velocity_limits(True)
    runtime = eik.SolverRuntimeConfig()
    runtime.weighted_fallback_enabled = weighted_fallback
    solver.configure_runtime(runtime)
    task = solver.add_frame_task("tool_motion", "tool", eik.TaskType.FRAME_POSITION)
    task.solve_mode = mode
    task.allow_min_error_fallback = task_fallback
    desired_twist = np.array([TARGET_SPEED, TARGET_SPEED, 0.0])
    task.set_target_velocity(desired_twist)
    configuration = np.zeros(robot.nq)
    robot.update_configuration(configuration)
    result = solver.solve_velocity(configuration, apply_limits=True)
    velocity = np.asarray(result.joint_velocities)
    achieved_twist = robot.get_frame_jacobian("tool")[:3] @ velocity
    candidate = configuration + DT * velocity
    lower, upper = robot.get_joint_limits()
    return {
        "requested_mode": mode.name,
        "task_fallback_enabled": task_fallback,
        "weighted_fallback_enabled": weighted_fallback,
        "status": result.status.name,
        "status_message": result.status_message,
        "velocity": velocity.tolist(),
        "task_scales": list(result.task_scales),
        "effective_modes": [value.name for value in result.task_modes_effective],
        "task_used_fallback": list(result.task_used_fallback),
        "weighted_fallback_used": bool(result.weighted_fallback_used),
        "desired_twist": desired_twist.tolist(),
        "achieved_twist": achieved_twist.tolist(),
        "twist_residual_norm": float(np.linalg.norm(desired_twist - achieved_twist)),
        "position_bounds_satisfied": bool(
            np.all(candidate >= np.asarray(lower) - BOUND_TOLERANCE)
            and np.all(candidate <= np.asarray(upper) + BOUND_TOLERANCE)
        ),
    }


def active_constraint_release_case() -> dict[str, object]:
    """Check that recovery can release a constraint activated too early."""
    jacobian = np.array([[1.0, -9.0], [0.0, np.sqrt(19.0)]])
    target = jacobian @ np.array([3.0, 2.0])
    lower = np.full(2, -10.0)
    upper = np.ones(2)
    result = eik.computeConstrainedWeightedVelocitySolutionEigen(
        [target], [jacobian], np.eye(2), lower, upper
    )
    velocity = np.asarray(result.solution)
    # Pinning the most violated first coordinate and then the second yields
    # [1, 1]. The optimum needs the first constraint released. This comparator
    # illustrates add-only saturation; it is not an invocation of task MIN_ERROR.
    pinned_reference = np.ones(2)
    oracle = lsq_linear(jacobian, target, bounds=(lower, upper), tol=1e-12)
    if not oracle.success:
        raise RuntimeError(f"Independent bounded least-squares oracle failed: {oracle.message}")
    return {
        "cpu_status": result.status.name,
        "cpu_weighted_velocity": velocity.tolist(),
        "unregularized_optimum": [-6.0, 1.0],
        "add_only_saturation_reference": pinned_reference.tolist(),
        "cpu_residual_norm": float(np.linalg.norm(jacobian @ velocity - target)),
        "independent_lsq_velocity": oracle.x.tolist(),
        "independent_lsq_residual_norm": float(np.linalg.norm(jacobian @ oracle.x - target)),
        "add_only_residual_norm": float(np.linalg.norm(jacobian @ pinned_reference - target)),
        "bounds_satisfied": bool(
            np.all(velocity >= lower - BOUND_TOLERANCE)
            and np.all(velocity <= upper + BOUND_TOLERANCE)
        ),
    }


def equivalent_task_row_cases() -> list[dict[str, object]]:
    """Compare equivalent task representations against an independent LP."""
    problems = (
        ("one_row", np.array([[1.0, 1.0]]), np.array([1.0])),
        ("zero_row", np.array([[1.0, 1.0], [0.0, 0.0]]), np.array([1.0, 0.0])),
        ("dependent_row", np.array([[1.0, 1.0], [2.0, 2.0]]), np.array([1.0, 2.0])),
    )
    lower = np.full(2, -2.0)
    upper = np.array([2.0, 0.0])
    rows = []
    for name, jacobian, target in problems:
        objective = np.array([0.0, 0.0, -1.0])
        oracle = linprog(
            objective,
            A_eq=np.column_stack((jacobian, -target)),
            b_eq=np.zeros(target.size),
            bounds=[*zip(lower, upper), (0.0, 1.0)],
            method="highs",
        )
        if not oracle.success:
            raise RuntimeError(f"Independent maximum-scale oracle failed: {oracle.message}")
        result = eik.computeMultiObjectiveVelocitySolutionEigen(
            [target], [jacobian], np.eye(2), lower, upper
        )
        velocity = np.asarray(result.solution)
        rows.append(
            {
                "representation": name,
                "cpu_status": result.status.name,
                "cpu_velocity": velocity.tolist(),
                "cpu_task_scales": list(result.task_scales),
                "cpu_residual_norm": float(np.linalg.norm(jacobian @ velocity - target)),
                "independent_maximum_scale": float(oracle.x[-1]),
                "independent_velocity": oracle.x[:-1].tolist(),
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="Also save the diagnostic JSON")
    args = parser.parse_args()
    configurations = (
        (eik.TaskSolveMode.SCALE, False, False),
        (eik.TaskSolveMode.SCALE, True, False),
        (eik.TaskSolveMode.MIN_ERROR, False, False),
        (eik.TaskSolveMode.SCALE, False, True),
    )
    with TemporaryDirectory(prefix="embodik-limit-recovery-") as directory:
        urdf = Path(directory) / "two_sliders.urdf"
        urdf.write_text(ROBOT_URDF)
        cases = [
            conflicting_task_case(urdf, mode, task_fallback=task, weighted_fallback=weighted)
            for mode, task, weighted in configurations
        ]
    report = {
        "embodik_version": eik.__version__,
        "scope": "CPU velocity solve; no nonlinear rollout, GPU execution or policy evaluation",
        "redundant_task": redundant_task_case(),
        "conflicting_cartesian_task": cases,
        "active_constraint_release": active_constraint_release_case(),
        "equivalent_task_rows": equivalent_task_row_cases(),
    }
    document = json.dumps(report, indent=2) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(document)
    print(document, end="")


if __name__ == "__main__":
    main()
