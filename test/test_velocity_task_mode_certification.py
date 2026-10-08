"""The public CPU setter must forward certification to the native solver."""

from __future__ import annotations

import numpy as np
import pytest
from test_velocity_sparse_collision_clamp import (
    COLLISION_NORMALS,
    JOINT_EFFORT,
    NUMERIC_TOLERANCE,
    RECOVERY_CAP,
    RECOVERY_SCALE,
    SECONDARY_VELOCITY,
    STEP_DT,
    _solve,
)

import embodik as eik


@pytest.mark.parametrize(
    "mode", [eik.TaskSolveMode.MIN_ERROR, eik.TaskSolveMode.SCALE, eik.TaskSolveMode.SCALE_ELASTIC]
)
def test_certified_runtime_returns_original_physical_collision_rows(tmp_path, mode):
    result = _solve(tmp_path, position_limits=True, certification=True, mode=mode)
    assert result.status == eik.SolverStatus.SUCCESS
    rates = COLLISION_NORMALS @ np.asarray(result.joint_velocities)
    assert np.all(rates >= RECOVERY_CAP - NUMERIC_TOLERANCE)


FRAME_PRIMARY_AXIS = np.array([-0.4, -0.7])
FRAME_AXES = np.vstack((FRAME_PRIMARY_AXIS, np.sqrt(1.0 - FRAME_PRIMARY_AXIS**2)))
FRAME_RECOVERY_ROWS = np.array([[-0.4, -1.0], [-0.6, 0.4]])
FRAME_NORMALS_UNSCALED = np.linalg.solve(FRAME_AXES.T, FRAME_RECOVERY_ROWS.T).T
FRAME_NORMALS = FRAME_NORMALS_UNSCALED / np.linalg.norm(FRAME_NORMALS_UNSCALED, axis=1)[:, None]
FRAME_COLLISION_JACOBIAN = FRAME_NORMALS @ FRAME_AXES
FRAME_CENTER_DISTANCE = 0.05
FRAME_SPHERE_RADIUS = 0.01
FRAME_MINIMUM_DISTANCE = 0.05
FRAME_JOINT_LIMIT = 1.0
FRAME_JOINT_SPEED = 1.0
FRAME_GOAL = -1.0
FRAME_OBSTACLES = -FRAME_CENTER_DISTANCE * FRAME_NORMALS


def test_runtime_setter_obtains_analytic_constrained_frame_min_error(tmp_path):
    obstacle_links = "".join(
        f'<link name="obstacle_{index}"><collision><geometry><sphere radius="{FRAME_SPHERE_RADIUS}"/>'
        f'</geometry></collision></link><joint name="fixed_{index}" type="fixed">'
        f'<parent link="world"/><child link="obstacle_{index}"/>'
        f'<origin xyz="{center[0]} {center[1]} 0"/></joint>'
        for index, center in enumerate(FRAME_OBSTACLES)
    )
    slider_links = "".join(
        f'<link name="{child}">{collision}</link><joint name="slide_{index}" type="prismatic">'
        f'<parent link="{parent}"/><child link="{child}"/>'
        f'<axis xyz="{axis[0]} {axis[1]} 0"/>'
        f'<limit lower="{-FRAME_JOINT_LIMIT}" upper="{FRAME_JOINT_LIMIT}" effort="{JOINT_EFFORT}" '
        f'velocity="{FRAME_JOINT_SPEED}"/></joint>'
        for index, (parent, child, collision), axis in zip(
            range(FRAME_AXES.shape[1]),
            (
                ("world", "slider", ""),
                (
                    "slider",
                    "moving",
                    f"<collision><geometry>"
                    f'<sphere radius="{FRAME_SPHERE_RADIUS}"/></geometry></collision>',
                ),
            ),
            FRAME_AXES.T,
        )
    )
    path = tmp_path / "certified_frame.urdf"
    path.write_text(
        f'<robot name="certified_frame"><link name="world"/>{obstacle_links}{slider_links}</robot>'
    )
    robot = eik.RobotModel(str(path), floating_base=False)
    solver = eik.KinematicsSolver(robot)
    solver.dt = STEP_DT
    solver.enable_velocity_limits(True)
    solver.enable_position_limits(False)
    solver.enable_velocity_task_mode_certification(True)
    primary = solver.add_frame_task("primary", "moving", eik.TaskType.FRAME_POSITION)
    primary.solve_mode = eik.TaskSolveMode.MIN_ERROR
    primary.set_position_mask(np.array([True, False, False]))
    primary.set_target_position_velocity(np.array([FRAME_GOAL, 0.0, 0.0]))
    secondary = solver.add_posture_task("secondary")
    secondary.priority = 1
    secondary.solve_mode = eik.TaskSolveMode.MIN_ERROR
    secondary.weight = 1.0
    secondary.set_target_velocity(SECONDARY_VELOCITY)
    pairs = [
        pair for pair in robot.get_collision_pair_names() if any("moving" in name for name in pair)
    ]
    assert len(pairs) == len(FRAME_NORMALS)
    solver.configure_collision_constraint(
        FRAME_MINIMUM_DISTANCE, include_pairs=pairs, max_constraints=len(pairs)
    )
    solver.set_collision_recovery_scale(RECOVERY_SCALE)
    solver.set_collision_max_separation_speed_nonpenetrating(RECOVERY_CAP)
    result = solver.solve_velocity(np.zeros(robot.nv), apply_limits=True)
    assert result.status == eik.SolverStatus.SUCCESS
    velocity = np.asarray(result.joint_velocities)
    expected = np.linalg.solve(FRAME_COLLISION_JACOBIAN, np.full(len(FRAME_NORMALS), RECOVERY_CAP))
    np.testing.assert_allclose(velocity, expected, atol=NUMERIC_TOLERANCE, rtol=0.0)
    assert np.all(FRAME_COLLISION_JACOBIAN @ velocity >= RECOVERY_CAP - NUMERIC_TOLERANCE)
