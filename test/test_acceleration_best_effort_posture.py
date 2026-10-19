"""Explicit secondary preferences retain achieved primary acceleration."""

from pathlib import Path

import numpy as np
import pytest

import embodik as eik

_URDF = """<robot name="independent_acceleration_sliders">
<link name="base"/><link name="first"/><link name="second"/>
<joint name="first_slide" type="prismatic">
  <parent link="base"/><child link="first"/><axis xyz="1 0 0"/>
  <limit lower="-2" upper="2" velocity="10" effort="100"/>
</joint>
<joint name="second_slide" type="prismatic">
  <parent link="base"/><child link="second"/><axis xyz="0 1 0"/>
  <limit lower="-2" upper="2" velocity="10" effort="100"/>
</joint>
</robot>"""


@pytest.mark.parametrize("best_effort_posture", [False, True])
@pytest.mark.parametrize("allow_best_effort_lower_tasks", [False, True])
def test_secondary_posture_mode_controls_strict_hierarchy_feasibility(
    tmp_path: Path, best_effort_posture: bool, allow_best_effort_lower_tasks: bool
) -> None:
    path = tmp_path / "independent_acceleration_sliders.urdf"
    path.write_text(_URDF)
    robot = eik.RobotModel(str(path), floating_base=False)
    solver = eik.AccelerationSolver(robot)
    primary = solver.add_joint_task("primary", "first_slide", 0.0)
    primary.priority = 0
    primary.solve_mode = eik.TaskSolveMode.SCALE
    primary.allow_min_error_fallback = False
    primary_reference = eik.AccelerationTaskReference()
    primary_reference.proportional_gain = 0.0
    primary_reference.derivative_gain = 0.0
    primary_reference.desired_acceleration = np.array([2.0])
    solver.set_task_reference("primary", primary_reference)

    secondary = solver.add_posture_task("posture")
    secondary.priority = 1
    secondary.solve_mode = (
        eik.TaskSolveMode.MIN_ERROR if best_effort_posture else eik.TaskSolveMode.SCALE
    )
    secondary.allow_min_error_fallback = False
    secondary.set_target_configuration(np.zeros(2))
    secondary_reference = eik.AccelerationTaskReference()
    secondary_reference.proportional_gain = 0.0
    secondary_reference.derivative_gain = 0.0
    secondary_reference.desired_acceleration = np.array([0.0, 1.0])
    solver.set_task_reference("posture", secondary_reference)

    options = eik.AccelerationSolveOptions()
    assert options.allow_best_effort_lower_tasks is False
    options.allow_best_effort_lower_tasks = allow_best_effort_lower_tasks
    assert options.allow_best_effort_lower_tasks is allow_best_effort_lower_tasks
    options.acceleration_limits_override = np.array([1.0, 2.0])
    options.allow_state_box_task_fallback = False
    result = solver.solve(np.zeros(2), np.zeros(2), 0.01, options)

    # The primary fixes ddq0=1 at scale .5. No scale of the secondary's
    # [0,1] ray can preserve that image, including secondary scale zero.
    assert result.task_scales[0] == pytest.approx(0.5, abs=1e-9)
    if not best_effort_posture:
        assert result.status == eik.SolverStatus.NO_PROGRESS
        assert "task equality" in result.status_message
        assert np.asarray(result.joint_accelerations).size == 0
        assert np.asarray(result.q_solution).size == 0
        return

    assert result.status == eik.SolverStatus.SUCCESS, result.status_message
    np.testing.assert_allclose(result.joint_accelerations, [1.0, 1.0], atol=1e-9)
    np.testing.assert_allclose(result.joint_velocities_next, [0.01, 0.01], atol=1e-11)
    np.testing.assert_allclose(result.q_solution, [0.00005, 0.00005], atol=1e-12)
    assert list(result.task_modes_effective) == [
        eik.TaskSolveMode.SCALE,
        eik.TaskSolveMode.MIN_ERROR,
    ]
    assert list(result.task_used_fallback) == [False, False]
    assert result.task_scales[1] == -1.0
