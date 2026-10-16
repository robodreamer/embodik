"""Elastic joint realizations retain achieved Cartesian motion during integration."""

import numpy as np
import pytest

import embodik as eik


@pytest.mark.parametrize("direction", [-1.0, 1.0])
@pytest.mark.parametrize("adaptive", [False, True])
@pytest.mark.parametrize("multiple_targets", [False, True])
def test_elastic_boundary_realization_preserves_primary_motion(
    tmp_path, direction, adaptive, multiple_targets
):
    lower, upper = (-2, 0) if direction > 0 else (0, 2)
    urdf = tmp_path / "coupled_slides.urdf"
    urdf.write_text(f"""<robot name="coupled_slides"><link name="base"/>
<link name="middle"/><link name="tool"/>
<joint name="first" type="prismatic"><parent link="base"/><child link="middle"/>
<axis xyz="1 0 0"/><limit lower="{lower}" upper="{upper}" velocity="1" effort="1"/></joint>
<joint name="second" type="prismatic"><parent link="middle"/><child link="tool"/>
<axis xyz="1 0 0"/><limit lower="-2" upper="2" velocity="1" effort="1"/></joint>
</robot>""")
    robot = eik.RobotModel(str(urdf), floating_base=False)
    solver = eik.KinematicsSolver(robot)
    solver.dt = 0.01 if adaptive else 0.05
    solver.enable_position_limits(True)
    solver.enable_velocity_limits(True)
    solver.add_frame_task("move", "tool", eik.TaskType.FRAME_POSITION)
    posture = solver.add_posture_task("bias")
    posture.priority = 2
    posture.set_target_configuration(direction * np.array([0.5, -0.5]))
    options = eik.PositionStepOptions()
    options.max_steps = 1
    options.position_gain = options.orientation_gain = 10.0
    options.primary_solve_mode = eik.TaskSolveMode.SCALE_ELASTIC
    options.primary_allow_min_error_fallback = False
    options.adaptive_dt = adaptive
    options.adaptive_dt_max_scale = 5.0
    options.adaptive_dt_reference_distance = 0.002
    options.record_velocity_candidates = True
    target = np.eye(4)
    target[0, 3] = direction * 0.01
    q = np.zeros(2)
    if multiple_targets:
        result = solver.solve_position_step(
            q, [eik.TaskTarget("move", target, 10.0, 10.0)], options
        )
    else:
        result = solver.solve_position_step(q, target, "move", options)
    assert len(result.velocity_candidates) == 1
    candidate = result.velocity_candidates[0]
    np.testing.assert_allclose(
        np.sum(candidate.integrated_velocity), np.sum(candidate.solver_velocity), atol=1e-10
    )
    actual = np.asarray(result.q_solution)
    assert direction * actual[0] <= 1e-9
    assert direction * actual.sum() > 0.002
    assert abs(target[0, 3] - actual.sum()) < abs(target[0, 3])
