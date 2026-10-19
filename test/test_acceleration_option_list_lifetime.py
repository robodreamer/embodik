"""Acceleration option records must survive replacement of their source list."""

import subprocess
import sys

import pytest

OPTION_RECORDS = [
    ("affine_constraints", "AffineAccelerationConstraint"),
    ("frozen_next_velocity_constraints", "FrozenNextVelocityConstraint"),
    ("task_acceleration_bounds", "TaskAccelerationBounds"),
    ("centroidal_momentum_rate_objectives", "CentroidalMomentumRateObjective"),
    ("centroidal_momentum_rate_bounds", "CentroidalMomentumRateBounds"),
    ("contact_acceleration_constraints", "ContactAccelerationConstraint"),
    ("tight_point_constraints", "TightPointAccelerationConstraint"),
    ("tight_frame_pose_constraints", "TightFramePoseAccelerationConstraint"),
    ("relative_pose_constraints", "RelativePoseAccelerationConstraint"),
    ("torso_pose_bound_constraints", "TorsoPoseBoundAccelerationConstraint"),
    ("com_support_polygon_constraints", "ComSupportPolygonAccelerationConstraint"),
    ("capture_point_constraints", "CapturePointAccelerationConstraint"),
    ("zmp_constraints", "ZmpAccelerationConstraint"),
]


@pytest.mark.parametrize(("field", "record_type"), OPTION_RECORDS)
def test_record_snapshot_survives_list_replacement(field, record_type):
    # A dangling native record can abort the interpreter; isolate each case so
    # the regression is reported as a test failure rather than killing pytest.
    code = f"""
import gc
import numpy as np
import embodik as eik
options = eik.AccelerationSolveOptions()
record = eik.{record_type}()
record.source_id = 'retained-record-' + 'x' * 100
if {field!r} == 'affine_constraints':
    record.coefficient_matrix = np.eye(2)
    record.affine_bias = np.zeros(2)
    record.lower_bounds = -np.ones(2)
    record.upper_bounds = np.ones(2)
    record.lower_bound_active = [False, True]
if {field!r} == 'tight_frame_pose_constraints':
    definition = eik.TightFramePoseConstraintDefinition()
    definition.frame_name = 'tip'
    definition.target_pose = np.diag([1., 1., 1., 1.])
    definition.position_epsilon = .0123
    policy = eik.GeometricConstraintAccelerationPolicy()
    policy.rate_limits = np.arange(1., 7.)
    record.definition, record.policy = definition, policy
if {field!r} == 'centroidal_momentum_rate_objectives':
    record.h_target = np.arange(6.)
    record.axis_mask = [True, False, True, False, True, False]
options.{field} = [record]
snapshot = list(options.{field})
options.{field} = []
gc.collect()
options.{field} = snapshot
assert options.{field}[0].source_id == record.source_id
snapshot[0].source_id = 'edited snapshot'
assert options.{field}[0].source_id == record.source_id
options.{field} = snapshot
assert options.{field}[0].source_id == 'edited snapshot'
if {field!r} == 'affine_constraints':
    np.testing.assert_array_equal(options.{field}[0].coefficient_matrix, np.eye(2))
    assert options.{field}[0].lower_bound_active == [False, True]
if {field!r} == 'tight_frame_pose_constraints':
    assert options.{field}[0].definition.frame_name == 'tip'
    assert options.{field}[0].definition.position_epsilon == .0123
    np.testing.assert_array_equal(options.{field}[0].policy.rate_limits, np.arange(1., 7.))
    definition = snapshot[0].definition
    definition.frame_name = 'edited nested snapshot'
    snapshot[0].definition = definition
    assert options.{field}[0].definition.frame_name == 'tip'
    options.{field} = snapshot
    assert options.{field}[0].definition.frame_name == 'edited nested snapshot'
if {field!r} == 'centroidal_momentum_rate_objectives':
    np.testing.assert_array_equal(options.{field}[0].h_target, np.arange(6.))
    assert options.{field}[0].axis_mask == [True, False, True, False, True, False]
del options
gc.collect()
assert snapshot[0].source_id == 'edited snapshot'
"""
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=30
    )
    assert result.returncode == 0, result.stderr[-4000:]
