"""Layout diagnostics describe active tasks, including with auto policy enabled."""

import numpy as np
import pytest

import embodik as eik


@pytest.fixture
def solver(tmp_path):
    path = tmp_path / "slider.urdf"
    path.write_text(
        '<robot name="slider"><link name="base"/><link name="tool"/>'
        '<joint name="slide" type="prismatic"><parent link="base"/>'
        '<child link="tool"/><axis xyz="1 0 0"/>'
        '<limit lower="-1" upper="1" effort="1" velocity="1"/></joint></robot>'
    )
    robot = eik.RobotModel(str(path), floating_base=False)
    return eik.KinematicsSolver(robot)


@pytest.mark.parametrize("automatic", [False, True])
@pytest.mark.parametrize("kind", ["raw_pose", "fixed_pose", "fixed_split", "auto_group"])
def test_layout_reports_actual_active_frame_types(solver, automatic, kind):
    cfg = solver.runtime_config()
    cfg.enable_auto_task_layout = automatic
    cfg.weighted_fallback_enabled = False
    solver.configure_runtime(cfg)
    target = np.eye(4)
    target[1, 3] = 1.0  # Binding demand changes auto state but cannot change raw tasks.
    if kind == "raw_pose":
        task = solver.add_frame_task("tool", "tool", eik.TaskType.FRAME_POSE)
        task.set_target_pose(target[:3, 3], target[:3, :3])
    else:
        group = solver.add_pose_task_group(
            "tool", "tool", merged_pose=kind == "fixed_pose", auto_switch=kind == "auto_group"
        )
        group.set_target(target, position_gain=1.0, rotation_gain=1.0)
    expected = eik.TaskLayout.SPLIT if kind == "fixed_split" else eik.TaskLayout.MERGED
    for _ in range(4):
        result = solver.solve_velocity(np.zeros(1))
        actual_expected = group.current_layout if kind == "auto_group" else expected
        assert result.active_task_layout == actual_expected


def test_raw_pose_step_does_not_claim_split_retry(solver):
    cfg = solver.runtime_config()
    cfg.enable_auto_task_layout = True
    cfg.weighted_fallback_enabled = False
    solver.configure_runtime(cfg)
    solver.add_frame_task("tool", "tool", eik.TaskType.FRAME_POSE)
    target = np.eye(4)
    target[1, 3] = 1
    options = eik.PositionStepOptions()
    options.max_steps = 1
    result = solver.solve_position_step(np.zeros(1), target, "tool", options)
    assert "retried split" not in result.status_message
    assert result.active_task_layout == eik.TaskLayout.MERGED


def test_inactive_component_does_not_change_pose_layout(solver):
    solver.add_frame_task("inactive", "tool", eik.TaskType.FRAME_POSITION).active = False
    task = solver.add_frame_task("pose", "tool", eik.TaskType.FRAME_POSE)
    task.set_target_pose(np.array([0.0, 1.0, 0.0]), np.eye(3))
    assert solver.solve_velocity(np.zeros(1)).active_task_layout == eik.TaskLayout.MERGED
