"""Minimum-error position steps solve against their actual integration box."""

from pathlib import Path

import numpy as np
import pytest

import embodik as eik


@pytest.mark.parametrize("direction", [-1.0, 1.0])
@pytest.mark.parametrize("adaptive", [False, True])
@pytest.mark.parametrize("multi", [False, True])
def test_min_error_step_preserves_native_motion_at_boundary(
    tmp_path: Path, direction: float, adaptive: bool, multi: bool
) -> None:
    lower, upper = (-2.0, 0.0) if direction > 0 else (0.0, 2.0)
    urdf = tmp_path / "slides.urdf"
    urdf.write_text(f"""<robot name="slides">
<link name="base"/><link name="middle"/><link name="tool"/>
<joint name="first" type="prismatic"><parent link="base"/><child link="middle"/>
<axis xyz="1 0 0"/><limit lower="{lower}" upper="{upper}" velocity="1" effort="1"/></joint>
<joint name="second" type="prismatic"><parent link="middle"/><child link="tool"/>
<axis xyz="1 0 0"/><limit lower="-2" upper="2" velocity="1" effort="1"/></joint>
</robot>""")
    robot = eik.RobotModel(str(urdf), floating_base=False)
    solver = eik.KinematicsSolver(robot)
    solver.dt = 0.01 if adaptive else 0.05
    solver.add_frame_task("move", "tool", eik.TaskType.FRAME_POSITION)
    posture = solver.add_posture_task("bias")
    posture.priority = 2
    posture.set_target_configuration(direction * np.array([0.5, -0.5]))
    options = eik.PositionStepOptions()
    options.max_steps = 1
    options.position_gain = options.orientation_gain = 10.0
    options.adaptive_dt = adaptive
    options.adaptive_dt_max_scale = 5.0
    options.adaptive_dt_reference_distance = 0.002
    options.primary_solve_mode = eik.TaskSolveMode.MIN_ERROR
    options.primary_allow_min_error_fallback = False
    options.record_velocity_candidates = True
    target = np.eye(4)
    target[0, 3] = direction * 0.01
    q = np.zeros(2)
    if multi:
        result = solver.solve_position_step(
            q, [eik.TaskTarget("move", target, 10.0, 10.0)], options
        )
    else:
        result = solver.solve_position_step(q, target, "move", options)
    assert result.status == eik.SolverStatus.SUCCESS
    assert len(result.velocity_candidates) == 1
    candidate = result.velocity_candidates[0]
    raw_speed = float(np.sum(candidate.solver_velocity))
    integrated_speed = float(np.sum(candidate.integrated_velocity))
    assert direction * raw_speed > 0.09
    assert raw_speed == pytest.approx(integrated_speed, abs=1e-8)
    assert -direction * np.asarray(candidate.candidate_q)[0] >= 1e-4 - 1e-12
    assert direction * np.sum(result.q_solution) > 0.002
    assert abs(target[0, 3] - np.sum(result.q_solution)) < abs(target[0, 3])


@pytest.mark.parametrize("direction", [-1.0, 1.0])
@pytest.mark.parametrize("side", [-1.0, 1.0])
def test_pose_min_error_retains_legal_inward_motion(
    tmp_path: Path, direction: float, side: float
) -> None:
    lower, upper = (-2.0, 0.0) if side > 0 else (0.0, 2.0)
    axis = -side * direction
    urdf = tmp_path / "pose_slides.urdf"
    urdf.write_text(f"""<robot name="pose_slides">
<link name="base"/><link name="middle"/><link name="tool"/>
<joint name="first" type="prismatic"><parent link="base"/><child link="middle"/>
<axis xyz="{axis} 0 0"/><limit lower="{lower}" upper="{upper}" velocity="1" effort="1"/></joint>
<joint name="second" type="prismatic"><parent link="middle"/><child link="tool"/>
<axis xyz="{axis} 0 0"/><limit lower="-2" upper="2" velocity="1" effort="1"/></joint>
</robot>""")
    robot = eik.RobotModel(str(urdf), floating_base=False)
    solver = eik.KinematicsSolver(robot)
    solver.dt = 0.05
    solver.add_frame_task("move", "tool", eik.TaskType.FRAME_POSE)
    q = np.array([-side * 1e-4, 0.0])
    target = np.eye(4)
    target[0, 3] = axis * q[0] + direction * 0.01
    options = eik.PositionStepOptions()
    options.max_steps = 1
    options.position_gain = options.orientation_gain = 10.0
    options.primary_solve_mode = eik.TaskSolveMode.MIN_ERROR
    options.locked_joint_indices = [1]
    options.record_velocity_candidates = True
    result = solver.solve_position_step(q, target, "move", options)
    assert result.status == eik.SolverStatus.SUCCESS
    assert len(result.velocity_candidates) == 1
    candidate = result.velocity_candidates[0]
    raw_speed = axis * candidate.solver_velocity[0]
    assert direction * raw_speed > 0.05
    assert raw_speed == pytest.approx(axis * candidate.integrated_velocity[0], abs=1e-8)
    assert np.asarray(result.q_solution)[1] == 0.0
    progress = axis * (np.asarray(result.q_solution)[0] - q[0])
    assert direction * progress > 0.002
