"""A coarse accounting regression, not a latency benchmark."""

from time import perf_counter

import numpy as np
import pytest

import embodik as eik


@pytest.mark.parametrize("single", [False, True])
def test_total_native_time_includes_all_inner_steps(tmp_path, single):
    path = tmp_path / "timing.urdf"
    path.write_text("""<robot name="timing"><link name="world"/><link name="tool"/>
<joint name="slide" type="prismatic"><parent link="world"/><child link="tool"/>
<axis xyz="1 0 0"/><limit lower="-2" upper="2" effort="100" velocity="1"/></joint>
</robot>""")
    robot = eik.RobotModel(str(path), floating_base=False)
    fractions = []
    # Repeats and a loose ratio tolerate binding overhead and scheduling. The
    # old last-inner-solve metric accounts for under 1% on this 256-step case.
    for _ in range(3):
        solver = eik.KinematicsSolver(robot)
        solver.dt = 0.01
        solver.add_frame_task("reach", "tool", eik.TaskType.FRAME_POSITION)
        options = eik.PositionStepOptions()
        options.max_steps = 256
        options.limit_change_from_seed = False
        options.position_gain, options.orientation_gain = 0.5, 0.0
        options.stall_recovery = False
        options.record_velocity_candidates = True
        pose = np.eye(4)
        pose[0, 3] = 1.0
        started = perf_counter()
        result = (
            solver.solve_position_step(np.zeros(1), pose, "reach", options)
            if single
            else solver.solve_position_step(
                np.zeros(1), [eik.TaskTarget("reach", pose, 0.5, 0.0)], options
            )
        )
        wall_ms = 1000 * (perf_counter() - started)
        assert result.status == eik.SolverStatus.SUCCESS
        assert len(result.velocity_candidates) == 256
        assert 0 <= result.computation_time_ms <= 1.1 * wall_ms + 0.1
        fractions.append(result.computation_time_ms / max(wall_ms, 1e-12))
    assert max(fractions) > 0.1, fractions
