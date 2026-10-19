"""Direct limit desaturation must preserve explicit position-step locks."""

from pathlib import Path

import numpy as np
import pytest

import embodik as eik


@pytest.mark.parametrize("side", [-1.0, 1.0])
@pytest.mark.parametrize("multi", [False, True])
@pytest.mark.parametrize("lock_kind", ["explicit", "excluded"])
@pytest.mark.parametrize(
    "mode", [eik.TaskSolveMode.SCALE, eik.TaskSolveMode.SCALE_ELASTIC, eik.TaskSolveMode.MIN_ERROR]
)
def test_desaturation_keeps_valid_locked_coordinates(
    tmp_path: Path, side: float, multi: bool, lock_kind: str, mode: eik.TaskSolveMode
) -> None:
    lower, upper = (0.0, 2.0) if side < 0 else (-2.0, 0.0)
    links = [f'<link name="link{i}"/>' for i in range(5)]
    joints = [f"""<joint name="slide{i}" type="prismatic">
<parent link="link{i}"/><child link="link{i+1}"/><axis xyz="1 0 0"/>
<limit lower="{lower}" upper="{upper}" velocity="1" effort="1"/></joint>""" for i in range(4)]
    urdf = tmp_path / "slides.urdf"
    urdf.write_text('<robot name="locked">' + "".join(links + joints) + "</robot>")
    robot = eik.RobotModel(str(urdf), floating_base=False)
    solver = eik.KinematicsSolver(robot)
    solver.dt = 0.01
    solver.add_frame_task("move", "link4", eik.TaskType.FRAME_POSE)
    seed = np.full(4, -side * 1e-4)
    target = np.eye(4)
    target[0, 3] = float(seed.sum() + side * 0.04)
    options = eik.PositionStepOptions()
    options.max_steps = 1
    options.primary_solve_mode = mode
    options.primary_allow_min_error_fallback = False
    if lock_kind == "explicit":
        options.locked_joint_indices = list(range(4))
    else:
        options.excluded_joint_indices = list(range(4))
    if multi:
        result = solver.solve_position_step(
            seed, [eik.TaskTarget("move", target, 10.0, 10.0)], options
        )
    else:
        result = solver.solve_position_step(seed, target, "move", options)
    np.testing.assert_allclose(result.q_solution, seed, atol=1e-12, rtol=0)
