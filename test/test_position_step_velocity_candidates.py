"""Opt-in diagnostics preserve solve behavior and use manifold velocities."""

import numpy as np
import pytest

import embodik as eik


@pytest.mark.parametrize("multiple_targets", [False, True])
@pytest.mark.parametrize("adaptive", [False, True])
@pytest.mark.parametrize("masked", [False, True])
def test_velocity_candidates_are_observational(tmp_path, multiple_targets, adaptive, masked):
    urdf = tmp_path / "slide.urdf"
    urdf.write_text("""<robot name="slide"><link name="base"/><link name="tool"/>
<joint name="slide" type="prismatic"><parent link="base"/><child link="tool"/>
<axis xyz="1 0 0"/><limit lower="-2" upper="2" velocity="2" effort="1"/></joint>
</robot>""")
    outputs = []
    for recording in (False, True):
        robot = eik.RobotModel(str(urdf), floating_base=False)
        solver = eik.KinematicsSolver(robot)
        solver.dt = 0.01
        solver.add_frame_task("move", "tool", eik.TaskType.FRAME_POSITION)
        options = eik.PositionStepOptions()
        assert not options.record_velocity_candidates
        options.record_velocity_candidates = recording
        options.max_steps = 2
        options.position_gain = 5.0
        options.adaptive_dt = adaptive
        options.adaptive_dt_reference_distance = 0.05
        if masked:
            options.integration_zero_velocity_indices = [0]
        q = np.array([0.5])
        target = np.eye(4)
        target[0, 3] = 0.8
        history = []
        for _ in range(3):
            if multiple_targets:
                result = solver.solve_position_step(
                    q, [eik.TaskTarget("move", target, 5.0, 1.0)], options
                )
            else:
                result = solver.solve_position_step(q, target, "move", options)
            if recording:
                assert len(result.velocity_candidates) == result.iterations_used == 2
                reference = q
                for step, candidate in enumerate(result.velocity_candidates):
                    assert candidate.step_index == step
                    assert candidate.integration_dt >= solver.dt
                    if adaptive:
                        assert candidate.integration_dt > solver.dt
                    np.testing.assert_array_equal(candidate.reference_q, reference)
                    np.testing.assert_allclose(
                        candidate.integrated_velocity,
                        (candidate.candidate_q - candidate.reference_q) / candidate.integration_dt,
                        atol=1e-12,
                    )
                    assert candidate.solver_velocity.shape == (robot.nv,)
                    assert np.isfinite(candidate.solver_velocity).all()
                    if masked:
                        assert candidate.solver_velocity[0] > 0.01
                        np.testing.assert_array_equal(candidate.integrated_velocity, [0.0])
                    assert candidate.task_scales
                    reference = candidate.candidate_q
            else:
                assert not result.velocity_candidates
            q = np.asarray(result.q_solution).copy()
            history.append((q.copy(), result.status, result.status_message))
        outputs.append(history)
    for plain, traced in zip(*outputs):
        np.testing.assert_array_equal(plain[0], traced[0])
        assert plain[1:] == traced[1:]


def test_velocity_candidate_uses_tangent_dimension_for_continuous_joint(tmp_path):
    urdf = tmp_path / "continuous.urdf"
    urdf.write_text("""<robot name="continuous"><link name="base"/><link name="pivot"/>
<link name="tool"/><joint name="turn" type="continuous"><parent link="base"/>
<child link="pivot"/><axis xyz="0 0 1"/><limit velocity="2" effort="1"/></joint>
<joint name="offset" type="fixed"><parent link="pivot"/><child link="tool"/>
<origin xyz="1 0 0"/></joint></robot>""")
    robot = eik.RobotModel(str(urdf), floating_base=False)
    solver = eik.KinematicsSolver(robot)
    solver.dt = 0.01
    solver.add_frame_task("move", "tool", eik.TaskType.FRAME_POSITION)
    options = eik.PositionStepOptions()
    options.record_velocity_candidates = True
    target = np.eye(4)
    target[:3, 3] = [np.cos(0.1), np.sin(0.1), 0.0]
    result = solver.solve_position_step(np.array([1.0, 0.0]), target, "move", options)
    candidate = result.velocity_candidates[0]
    assert candidate.reference_q.shape == candidate.candidate_q.shape == (2,)
    assert candidate.solver_velocity.shape == candidate.integrated_velocity.shape == (1,)
    theta = np.arctan2(candidate.candidate_q[1], candidate.candidate_q[0])
    np.testing.assert_allclose(
        candidate.integrated_velocity, [theta / candidate.integration_dt], atol=1e-12
    )
