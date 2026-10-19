"""Advanced physical momentum-rate objectives share bounded elasticity."""

import numpy as np
import pytest

import embodik

DT = 0.05
NOMINAL_LIMIT = 0.5
URDF = """<robot name="elastic_centroidal"><link name="base"/>
<link name="tip"><inertial><origin xyz="0 0 0"/><mass value="1"/>
<inertia ixx="1" iyy="1" izz="1" ixy="0" ixz="0" iyz="0"/>
</inertial></link><joint name="joint" type="prismatic">
<parent link="base"/><child link="tip"/><axis xyz="1 0 0"/>
<limit lower="-1" upper="1" velocity="2" effort="10"/></joint></robot>"""


def test_centroidal_elastic_objective_activates_band_without_registered_tasks(tmp_path):
    path = tmp_path / "centroidal.urdf"
    path.write_text(URDF)
    robot = embodik.RobotModel(str(path), floating_base=False)
    solver = embodik.AccelerationSolver(robot)
    objective = embodik.CentroidalMomentumRateObjective()
    objective.source_id = "momentum"
    objective.h_target = np.zeros(6)
    objective.hdot_feedforward = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    objective.axis_mask = [True, False, False, False, False, False]
    objective.allow_min_error_fallback = False
    options = embodik.AccelerationSolveOptions()
    options.acceleration_limits_override = np.array([10.0])
    options.allow_state_box_task_fallback = False
    options.nominal_position_lower_bounds = np.array([-NOMINAL_LIMIT])
    options.nominal_position_upper_bounds = np.array([NOMINAL_LIMIT])
    options.centroidal_momentum_rate_objectives = [objective]
    seed, velocity = np.array([NOMINAL_LIMIT]), np.zeros(1)
    strict = solver.solve(seed, velocity, DT, options)
    assert strict.status == embodik.SolverStatus.SUCCESS, strict.status_message
    np.testing.assert_allclose(strict.q_solution, seed)
    objective.solve_mode = embodik.TaskSolveMode.SCALE_ELASTIC
    options.centroidal_momentum_rate_objectives = [objective]
    elastic = solver.solve(seed, velocity, DT, options)
    assert elastic.status == embodik.SolverStatus.SUCCESS, elastic.status_message
    assert elastic.elastic_band_enabled
    assert elastic.q_solution[0] > strict.q_solution[0]
    assert elastic.q_solution[0] <= NOMINAL_LIMIT + options.elastic_delta_max
    report = elastic.centroidal_momentum_rate_diagnostics[0]
    assert report.effective_mode == embodik.TaskSolveMode.SCALE
    assert not report.used_min_error_fallback


@pytest.mark.parametrize("elastic_centroidal", [False, True])
def test_mixed_centroidal_min_error_sentinel_does_not_expand_band(tmp_path, elastic_centroidal):
    path = tmp_path / "mixed_centroidal.urdf"
    path.write_text(URDF)
    robot = embodik.RobotModel(str(path), floating_base=False)
    solver = embodik.AccelerationSolver(robot)
    task = solver.add_joint_task("joint", "joint", 0.0)
    task.priority = 1 if elastic_centroidal else 0
    task.solve_mode = (
        embodik.TaskSolveMode.MIN_ERROR
        if elastic_centroidal
        else embodik.TaskSolveMode.SCALE_ELASTIC
    )
    task.allow_min_error_fallback = False
    reference = embodik.AccelerationTaskReference()
    reference.proportional_gain = reference.derivative_gain = 0.0
    reference.desired_acceleration = np.array([0.2])
    solver.set_task_reference("joint", reference)
    objective = embodik.CentroidalMomentumRateObjective()
    objective.source_id = "momentum"
    objective.priority = 0 if elastic_centroidal else 1
    objective.solve_mode = (
        embodik.TaskSolveMode.SCALE_ELASTIC
        if elastic_centroidal
        else embodik.TaskSolveMode.MIN_ERROR
    )
    objective.h_target = np.zeros(6)
    objective.hdot_feedforward = np.array([0.2, 0.0, 0.0, 0.0, 0.0, 0.0])
    objective.axis_mask = [True, False, False, False, False, False]
    objective.allow_min_error_fallback = False
    options = embodik.AccelerationSolveOptions()
    options.acceleration_limits_override = np.array([10.0])
    options.nominal_position_lower_bounds = np.array([-NOMINAL_LIMIT])
    options.nominal_position_upper_bounds = np.array([NOMINAL_LIMIT])
    options.allow_state_box_task_fallback = False
    options.centroidal_momentum_rate_objectives = [objective]

    for _ in range(3):
        result = solver.solve(np.zeros(1), np.zeros(1), DT, options)
        assert result.status == embodik.SolverStatus.SUCCESS, result.status_message
        assert result.elastic_band_enabled
        report = result.centroidal_momentum_rate_diagnostics[0]
        assert result.task_scales == [-1.0 if elastic_centroidal else 1.0]
        assert report.scale == (1.0 if elastic_centroidal else -1.0)
        np.testing.assert_allclose(result.task_errors, [0.0], atol=1e-9)
        np.testing.assert_allclose(report.residual, np.zeros(6), atol=1e-9)
        assert not result.saturated_position_indices
        np.testing.assert_array_equal(result.elastic_band_delta, np.zeros(1))
        np.testing.assert_array_equal(solver.get_elastic_band_deltas(), np.zeros(1))
