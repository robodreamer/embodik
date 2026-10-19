"""Secondary bound recovery must preserve achieved primary Cartesian motion."""

import numpy as np
import pytest

import embodik as eik


@pytest.mark.parametrize("direction", [-1.0, 1.0])
def test_lower_min_error_bounds_preserve_primary_motion(tmp_path, direction: float) -> None:
    rng = np.random.default_rng(5)
    axes = rng.normal(size=(7, 3))
    axes /= np.linalg.norm(axes, axis=1)[:, None]
    parts = ['<robot name="redundant_stage"><link name="base"/>']
    parent = "base"
    for index, axis in enumerate(axes):
        child = f"stage_{index}"
        parts.append(
            f'<link name="{child}"/><joint name="joint_{index}" type="prismatic">'
            f'<parent link="{parent}"/><child link="{child}"/>'
            f'<axis xyz="{axis[0]} {axis[1]} {axis[2]}"/>'
            '<limit lower="-10" upper="10" velocity="1" effort="1"/></joint>'
        )
        parent = child
    for index in range(2):
        parts.append(
            f'<link name="aux_{index}"/><joint name="aux_{index}" type="prismatic">'
            f'<parent link="{parent}"/><child link="aux_{index}"/>'
            '<axis xyz="1 0 0"/>'
            '<limit lower="-10" upper="10" velocity="1" effort="1"/></joint>'
        )
    parts.append("</robot>")
    path = tmp_path / "redundant_stage.urdf"
    path.write_text("".join(parts))
    robot = eik.RobotModel(str(path), floating_base=False)
    solver = eik.KinematicsSolver(robot)
    solver.enable_position_limits(True)
    solver.enable_velocity_limits(True)
    runtime = solver.runtime_config()
    runtime.enable_auto_task_layout = False
    runtime.weighted_fallback_enabled = False
    solver.configure_runtime(runtime)
    primary = solver.add_frame_task("primary", parent, eik.TaskType.FRAME_POSITION)
    primary.priority = 0
    primary.solve_mode = eik.TaskSolveMode.SCALE
    primary.set_target_pose(direction * np.array([0.1, 0.02, 0.03]), np.eye(3))
    q = np.zeros(robot.nq)
    reference = solver.solve_velocity(q)
    assert reference.status == eik.SolverStatus.SUCCESS
    jacobian = np.asarray(robot.get_frame_jacobian(parent))[:3].copy()
    indices = [int(robot.get_joint_velocity_index(f"joint_{index}")) for index in range(7)]
    secondary = solver.add_posture_task("secondary", indices)
    secondary.priority = 1
    secondary.solve_mode = eik.TaskSolveMode.MIN_ERROR
    secondary.set_controlled_joint_targets(direction * rng.normal(size=7) * 50.0)
    result = solver.solve_velocity(q)
    assert result.status == eik.SolverStatus.SUCCESS
    assert result.weighted_fallback_used is False
    velocity = np.asarray(result.solution)
    assert np.all(np.abs(velocity) <= 1.0 + 1e-8)
    expected_motion = jacobian @ np.asarray(reference.solution)
    assert np.linalg.norm(expected_motion) > 0.01
    np.testing.assert_allclose(jacobian @ velocity, expected_motion, atol=1e-8)
