"""Sparse collision rows must never be interpreted as joint position boxes."""

from __future__ import annotations

import numpy as np
import pytest

import embodik as eik

SPHERE_RADIUS = 0.01
CENTER_DISTANCE = 0.05
MINIMUM_DISTANCE = 0.05
JOINT_LIMIT = 1.0
JOINT_SPEED = 1.0
JOINT_EFFORT = 100.0
STEP_DT = 0.02
NUMERIC_TOLERANCE = 1e-6
RECOVERY_DIRECTION = -1.0
RECOVERY_CAP = 0.15
RECOVERY_SCALE = 1.0
SECONDARY_VELOCITY = np.array([0.6, 0.2])
COLLISION_NORMALS = np.array([[-0.6, -0.8], [-0.8, 0.6]])
OBSTACLE_CENTERS = -CENTER_DISTANCE * COLLISION_NORMALS


def _urdf():
    obstacles = "".join(
        f'<link name="obstacle_{index}"><collision><geometry>'
        f'<sphere radius="{SPHERE_RADIUS}"/></geometry></collision></link>'
        f'<joint name="obstacle_fixed_{index}" type="fixed">'
        f'<parent link="world"/><child link="obstacle_{index}"/>'
        f'<origin xyz="{center[0]} {center[1]} 0"/></joint>'
        for index, center in enumerate(OBSTACLE_CENTERS)
    )
    joints = "".join(
        f'<link name="{child}">{collision}</link>'
        f'<joint name="{joint}" type="prismatic"><parent link="{parent}"/>'
        f'<child link="{child}"/><axis xyz="{axis}"/>'
        f'<limit lower="{-JOINT_LIMIT}" upper="{JOINT_LIMIT}" '
        f'effort="{JOINT_EFFORT}" velocity="{JOINT_SPEED}"/></joint>'
        for parent, child, joint, axis, collision in (
            ("world", "slider_x", "slide_x", "1 0 0", ""),
            (
                "slider_x",
                "moving",
                "slide_y",
                "0 1 0",
                f'<collision><geometry><sphere radius="{SPHERE_RADIUS}"/></geometry></collision>',
            ),
        )
    )
    return f'<robot name="sparse_collision_clamp"><link name="world"/>{obstacles}{joints}</robot>'


def _solve(
    tmp_path,
    position_limits,
    certification=False,
    mode=eik.TaskSolveMode.MIN_ERROR,
    primary_velocity=RECOVERY_DIRECTION,
):
    path = tmp_path / "sparse_collision_clamp.urdf"
    path.write_text(_urdf())
    robot = eik.RobotModel(str(path), floating_base=False)
    solver = eik.KinematicsSolver(robot)
    solver.dt = STEP_DT
    solver.enable_velocity_limits(True)
    solver.enable_position_limits(position_limits)
    if certification:
        solver.enable_velocity_task_mode_certification(True)
    primary = solver.add_joint_task("primary", "slide_x")
    primary.priority = 0
    primary.solve_mode = mode
    primary.allow_min_error_fallback = False
    primary.set_target_velocity(np.array([primary_velocity]))
    secondary = solver.add_posture_task("secondary")
    secondary.priority = 1
    secondary.solve_mode = eik.TaskSolveMode.MIN_ERROR
    secondary.weight = 1.0
    secondary.set_target_velocity(SECONDARY_VELOCITY)
    pairs = [
        pair for pair in robot.get_collision_pair_names() if any("moving" in name for name in pair)
    ]
    assert len(pairs) == len(COLLISION_NORMALS)
    solver.set_collision_recovery_scale(RECOVERY_SCALE)
    solver.set_collision_max_separation_speed_nonpenetrating(RECOVERY_CAP)
    solver.configure_collision_constraint(
        min_distance=MINIMUM_DISTANCE, include_pairs=pairs, max_constraints=len(pairs)
    )
    q = np.zeros(robot.nv)
    result = solver.solve_velocity(q, apply_limits=True)
    return result


@pytest.mark.parametrize(
    "mode", [eik.TaskSolveMode.MIN_ERROR, eik.TaskSolveMode.SCALE, eik.TaskSolveMode.SCALE_ELASTIC]
)
def test_sparse_collision_rows_do_not_change_velocity_when_position_limits_are_inactive(
    tmp_path, mode
):
    reference = _solve(tmp_path, position_limits=False, mode=mode)
    actual = _solve(tmp_path, position_limits=True, mode=mode)
    assert reference.status == eik.SolverStatus.SUCCESS
    assert actual.status == eik.SolverStatus.SUCCESS
    reference_velocity = np.asarray(reference.joint_velocities)
    actual_velocity = np.asarray(actual.joint_velocities)
    assert np.all(COLLISION_NORMALS @ reference_velocity > NUMERIC_TOLERANCE)
    np.testing.assert_allclose(
        actual_velocity, reference_velocity, atol=NUMERIC_TOLERANCE, rtol=0.0
    )
    assert np.all(COLLISION_NORMALS @ actual_velocity > NUMERIC_TOLERANCE)
