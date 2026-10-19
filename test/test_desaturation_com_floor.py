"""Configuration-space desaturation must not bypass an active CoM floor."""

from pathlib import Path

import numpy as np
import pytest

import embodik as eik


@pytest.mark.parametrize("side", [-1.0, 1.0])
@pytest.mark.parametrize("multi", [False, True])
@pytest.mark.parametrize("initial_slack", [-1e-5, 1e-5])
@pytest.mark.parametrize("frame", ["world", "support"])
@pytest.mark.parametrize(
    "mode", [eik.TaskSolveMode.SCALE, eik.TaskSolveMode.SCALE_ELASTIC, eik.TaskSolveMode.MIN_ERROR]
)
def test_desaturation_respects_com_boundary(
    tmp_path: Path,
    side: float,
    multi: bool,
    initial_slack: float,
    frame: str,
    mode: eik.TaskSolveMode,
) -> None:
    lower, upper = (0.0, 2.0) if side < 0 else (-2.0, 0.0)
    links = [
        f'<link name="link{i}"><inertial><mass value="1"/><inertia ixx="0.01" iyy="0.01" izz="0.01" ixy="0" ixz="0" iyz="0"/></inertial></link>'
        for i in range(5)
    ]
    joints = [f"""<joint name="slide{i}" type="prismatic">
<parent link="link{i}"/><child link="link{i+1}"/><axis xyz="1 0 0"/>
<limit lower="{lower}" upper="{upper}" velocity="1" effort="1"/></joint>""" for i in range(4)]
    urdf = tmp_path / "slides.urdf"
    support = '<link name="support"/><joint name="support_fixed" type="fixed"><parent link="link0"/><child link="support"/><origin xyz="0.3 0.4 0" rpy="0 0 0.4"/></joint>'
    urdf.write_text('<robot name="balance">' + "".join(links + joints) + support + "</robot>")
    robot = eik.RobotModel(str(urdf), floating_base=False)
    solver = eik.KinematicsSolver(robot)
    solver.dt = 0.01
    solver.add_frame_task("move", "link4", eik.TaskType.FRAME_POSE)
    seed = np.full(4, -side * 1e-4)
    robot.update_configuration(seed)

    def com_x() -> float:
        com = np.asarray(robot.get_com_position())
        if frame != "world":
            pose = robot.get_frame_pose(frame)
            com = np.asarray(pose.rotation).T @ (com - np.asarray(pose.translation))
        return float(com[0])

    initial_com = com_x()
    edge = initial_com - side * initial_slack
    xmin, xmax = (edge, 1.0) if side > 0 else (-1.0, edge)
    solver.configure_com_constraint(
        np.array([[xmin, -1.0], [xmax, -1.0], [xmax, 1.0], [xmin, 1.0]]),
        margin=0.0,
        frame_name=frame,
        com_vel_max=0.4,
        com_acc_max=0.1,
    )
    target = np.eye(4)
    target[0, 3] = float(seed.sum() + side * 0.04)
    options = eik.PositionStepOptions()
    options.max_steps = 1
    options.primary_solve_mode = mode
    options.primary_allow_min_error_fallback = False
    if multi:
        result = solver.solve_position_step(
            seed, [eik.TaskTarget("move", target, 10.0, 10.0)], options
        )
    else:
        result = solver.solve_position_step(seed, target, "move", options)
    robot.update_configuration(np.asarray(result.q_solution))
    final_com = com_x()
    assert min(xmin, initial_com) - 1e-9 <= final_com <= max(xmax, initial_com) + 1e-9
