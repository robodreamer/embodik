"""Advanced physical momentum-rate objectives share bounded elasticity."""

import numpy as np

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
