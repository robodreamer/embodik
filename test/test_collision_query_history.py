"""Fresh pair distances cannot inherit an unrelated GJK warm-start history."""

import numpy as np

import embodik as eik


def make_cylinder_solver(tmp_path):
    path = tmp_path / "rotated_cylinders.urdf"
    path.write_text(
        '<robot name="query_history"><link name="world"/><link name="fixed">'
        '<collision><origin rpy="0.2 0.3 0.4"/><geometry>'
        '<cylinder radius="0.1" length="0.4"/></geometry></collision></link>'
        '<joint name="fixed" type="fixed"><parent link="world"/><child link="fixed"/></joint>'
        '<link name="moving"><collision><origin rpy="0.7 0.2 0.8"/><geometry>'
        '<cylinder radius="0.12" length="0.35"/></geometry></collision></link>'
        '<joint name="slide" type="prismatic"><origin xyz="0.4 0.15 0.1"/>'
        '<parent link="world"/><child link="moving"/><axis xyz="1 0 0"/>'
        '<limit lower="-0.2" upper="0.5" velocity="1" effort="100"/></joint></robot>'
    )
    robot = eik.RobotModel(str(path), floating_base=False)
    solver = eik.KinematicsSolver(robot)
    solver.configure_collision_constraint(min_distance=0.035)
    return robot, solver


def test_pair_query_is_repeatable_after_other_configuration_queries(tmp_path):
    robot, solver = make_cylinder_solver(tmp_path)
    q = np.zeros(1)
    robot.update_configuration(q)
    initial = solver.evaluate_collision_pair_distances(q)[0].distance
    for other in np.linspace(-0.1, 0.4, 20):
        solver.evaluate_min_collision_distance(np.array([other]))
        records = solver.evaluate_collision_pair_distances(q)
        assert len(records) == 1
        assert abs(records[0].distance - initial) <= 1e-12
        np.testing.assert_array_equal(robot.get_current_configuration(), q)
    # A floor at the measured distance must not be broken by query history.
    solver.set_collision_pair_min_distance(
        "fixed", "moving", initial + 0.01, activate_when_clear=False
    )
    solver.set_collision_trajectory_floor_anchor(q, 1e-9)
    task = solver.add_frame_task("hold", "world", eik.TaskType.FRAME_POSITION)
    target = eik.TaskTarget.from_se3("hold", robot.get_frame_pose("world"), 1.0, 0.0)
    for other in (-0.1, 0.4, 0.1):
        solver.evaluate_min_collision_distance(np.array([other]))
        result = solver.solve_position_step(q, [target], eik.PositionStepOptions())
        assert result.status != eik.SolverStatus.COLLISION_VIOLATED
        np.testing.assert_array_equal(result.q_solution, q)


def test_strict_policy_distance_calculations_match_fresh_final_query(tmp_path):
    robot, solver = make_cylinder_solver(tmp_path)
    q = np.zeros(1)
    solver.set_collision_trajectory_floor_anchor(q, 1e-9)
    for other in np.linspace(-0.1, 0.4, 20):
        solver.evaluate_min_collision_distance(np.array([other]))
        internal = solver.evaluate_min_collision_distance(q)
        fresh = solver.evaluate_collision_pair_distances(q)[0].distance
        assert abs(internal - fresh) <= 1e-12
