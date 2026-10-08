"""FrameTask cache coherence must not depend on error/Jacobian read order."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import embodik as eik
from embodik import _embodik_impl as native

JOINT_POSITION_LIMIT = 1.0
JOINT_VELOCITY_LIMIT = 1.0
JOINT_EFFORT_LIMIT = 100.0
LEVER_ARM = 1.0
URDF = f"""<robot name="frame_cache_order">
<link name="world"/><link name="slider"/>
<joint name="slide" type="prismatic">
  <parent link="world"/><child link="slider"/><axis xyz="0 1 0"/>
  <limit lower="{-JOINT_POSITION_LIMIT}" upper="{JOINT_POSITION_LIMIT}"
    effort="{JOINT_EFFORT_LIMIT}" velocity="{JOINT_VELOCITY_LIMIT}"/>
</joint>
<link name="wrist"/>
<joint name="yaw" type="revolute">
  <parent link="slider"/><child link="wrist"/><axis xyz="0 0 1"/>
  <limit lower="{-JOINT_POSITION_LIMIT}" upper="{JOINT_POSITION_LIMIT}"
    effort="{JOINT_EFFORT_LIMIT}" velocity="{JOINT_VELOCITY_LIMIT}"/>
</joint>
<link name="ee"/>
<joint name="ee_fixed" type="fixed">
  <parent link="wrist"/><child link="ee"/><origin xyz="{LEVER_ARM} 0 0"/>
</joint>
</robot>"""

FRAME_NAME = "ee"
INITIAL_CONFIGURATION = np.array([0.03, 0.1])
CHANGED_CONFIGURATION = np.array([0.06, -0.15])
TARGET_TRANSLATION_CHANGE = np.array([0.0, 0.02, 0.0])
TARGET_ROTATION_CHANGE = 0.02
POSITION_MASK = np.array([0.0, 0.5, 1.0])
ORIENTATION_MASK = np.array([1.0, 0.5, 0.0])
NUMERIC_TOLERANCE = 1e-12
EXCLUDED_JOINT = 1


@pytest.fixture
def frame_case(tmp_path: Path):
    urdf_path = tmp_path / "cache_order.urdf"
    urdf_path.write_text(URDF)
    robot = eik.RobotModel(str(urdf_path), floating_base=False)
    robot.update_configuration(np.zeros(robot.nv))
    target = robot.get_frame_pose(FRAME_NAME)
    return robot, np.asarray(target.translation).copy(), np.asarray(target.rotation).copy()


def _expected_error_and_jacobian(robot, kind, target_position, target_rotation, masks, exclusions):
    pose = robot.get_frame_pose(FRAME_NAME)
    position_error = (target_position - np.asarray(pose.translation)) * masks[0]
    orientation_error = (
        np.asarray(native.log3(target_rotation @ np.asarray(pose.rotation).T)) * masks[1]
    )
    physical_jacobian = np.asarray(robot.get_frame_jacobian(FRAME_NAME)).copy()
    # Physical Jacobian masks select active axes; their nonzero magnitude is
    # applied to commanded errors, not physical row units.
    physical_jacobian[:3] *= (masks[0] != 0.0)[:, None]
    physical_jacobian[3:] *= (masks[1] != 0.0)[:, None]
    for index in exclusions:
        physical_jacobian[:, index] = 0.0
    match kind:
        case eik.TaskType.FRAME_POSITION:
            return position_error, physical_jacobian[:3]
        case eik.TaskType.FRAME_ORIENTATION:
            return orientation_error, physical_jacobian[3:]
        case _:
            return np.r_[position_error, orientation_error], physical_jacobian


@pytest.mark.parametrize(
    "kind", [eik.TaskType.FRAME_POSE, eik.TaskType.FRAME_POSITION, eik.TaskType.FRAME_ORIENTATION]
)
@pytest.mark.parametrize(
    "mutation", ["configuration", "target", "mask", "exclusions", "configuration_exclusions"]
)
@pytest.mark.parametrize("jacobian_first", [True, False])
def test_frame_cache_refreshes_both_quantities_after_invalidation(
    frame_case, kind, mutation, jacobian_first
):
    robot, target_position, target_rotation = frame_case
    solver = eik.KinematicsSolver(robot)
    task = solver.add_frame_task("cache_task", FRAME_NAME, kind)
    robot.update_configuration(INITIAL_CONFIGURATION)
    task.set_target_pose(target_position, target_rotation)
    task.update(robot)
    task.get_error()
    task.get_jacobian()
    masks = [np.ones(3), np.ones(3)]
    exclusions = []
    match mutation:
        case "configuration" | "configuration_exclusions":
            robot.update_configuration(CHANGED_CONFIGURATION)
            task.update(robot)
        case "target":
            target_position += TARGET_TRANSLATION_CHANGE
            cosine, sine = np.cos(TARGET_ROTATION_CHANGE), np.sin(TARGET_ROTATION_CHANGE)
            target_rotation = np.array([[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]])
            task.set_target_pose(target_position, target_rotation)
        case "mask":
            masks = [POSITION_MASK, ORIENTATION_MASK]
            task.set_position_mask(masks[0])
            task.set_orientation_mask(masks[1])
    if mutation in ("exclusions", "configuration_exclusions"):
        exclusions = [EXCLUDED_JOINT]
        task.set_excluded_joint_indices(exclusions)
    expected_error, expected_jacobian = _expected_error_and_jacobian(
        robot,
        kind,
        target_position,
        target_rotation,
        masks,
        exclusions,
    )
    if jacobian_first:
        jacobian = np.asarray(task.get_jacobian()).copy()
        error = np.asarray(task.get_error()).copy()
    else:
        error = np.asarray(task.get_error()).copy()
        jacobian = np.asarray(task.get_jacobian()).copy()
    np.testing.assert_allclose(error, expected_error, atol=NUMERIC_TOLERANCE, rtol=0.0)
    np.testing.assert_allclose(jacobian, expected_jacobian, atol=NUMERIC_TOLERANCE, rtol=0.0)
    # Repeated access must preserve both results too.
    np.testing.assert_allclose(task.get_error(), expected_error, atol=NUMERIC_TOLERANCE, rtol=0.0)
    np.testing.assert_allclose(
        task.get_jacobian(), expected_jacobian, atol=NUMERIC_TOLERANCE, rtol=0.0
    )


@pytest.mark.parametrize(
    "kind", [eik.TaskType.FRAME_POSE, eik.TaskType.FRAME_POSITION, eik.TaskType.FRAME_ORIENTATION]
)
def test_direct_velocity_solve_refreshes_error_despite_bypassing_feedback(frame_case, kind):
    robot, target_position, target_rotation = frame_case
    solver = eik.KinematicsSolver(robot)
    task = solver.add_frame_task("direct_task", FRAME_NAME, kind)
    robot.update_configuration(INITIAL_CONFIGURATION)
    task.set_target_pose(target_position, target_rotation)
    task.update(robot)
    task.get_error()
    task.get_jacobian()
    dimension = 6 if kind == eik.TaskType.FRAME_POSE else 3
    task.set_target_velocity(np.zeros(dimension))
    result = solver.solve_velocity(CHANGED_CONFIGURATION, True)
    assert result.status == eik.SolverStatus.SUCCESS
    expected_error, expected_jacobian = _expected_error_and_jacobian(
        robot, kind, target_position, target_rotation, [np.ones(3), np.ones(3)], []
    )
    np.testing.assert_allclose(task.get_error(), expected_error, atol=NUMERIC_TOLERANCE, rtol=0.0)
    np.testing.assert_allclose(
        task.get_jacobian(), expected_jacobian, atol=NUMERIC_TOLERANCE, rtol=0.0
    )
