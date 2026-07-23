import importlib

import numpy as np

import embodik as eik

ACCELERATION_EXPORTS = [
    "AccelerationSolver",
    "AccelerationSolverResult",
    "AccelerationSolverCapabilities",
    "AccelerationSolveOptions",
    "AccelerationTaskReference",
    "AccelerationTaskDiagnostics",
    "AccelerationAllocationDiagnostics",
    "AccelerationAnalyticCollisionPairDiagnostics",
    "AffineAccelerationConstraint",
    "FrozenNextVelocityConstraint",
    "TaskAccelerationBounds",
    "GeneralizedAccelerationAllocation",
    "EffortConstraintOptions",
    "ContactAccelerationConstraint",
    "GeometricConstraintAccelerationPolicy",
    "ComSupportPolygonAccelerationPolicy",
    "ComSupportPolygonAccelerationConstraint",
    "TightPointAccelerationConstraint",
    "TightFramePoseAccelerationConstraint",
    "RelativePoseAccelerationConstraint",
    "TorsoPoseBoundAccelerationConstraint",
    "VelocityCollisionLiftOptions",
    "CollisionConstraintAccelerationPolicy",
    "ComSupportPolygonOutsidePolicy",
    "CollisionConstraintOutsidePolicy",
    "AccelerationCollisionRegime",
    "CollisionGeometryPair",
    "CollisionPairMinimumDistance",
    "CollisionConstraintDefinition",
    "ComSupportPolygonConstraintDefinition",
    "RelativePoseConstraintDefinition",
    "TightPointConstraintDefinition",
    "TightFramePoseConstraintDefinition",
    "TorsoPoseBoundDefinition",
]


def test_acceleration_symbols_exported_once_from_package_and_extension():
    extension = importlib.import_module("embodik._embodik_impl")

    for name in ACCELERATION_EXPORTS:
        assert getattr(eik, name) is getattr(extension, name)
        assert name in eik.__all__

    assert eik.ContactType is extension.ContactType
    assert "ContactType" in eik.__all__
    assert eik.ContactType.POINT_CONTACT.name == "POINT_CONTACT"


def test_extension_version_matches_package_metadata():
    extension = importlib.import_module("embodik._embodik_impl")

    assert extension.__version__ == eik.__version__


def test_shared_constraint_definitions_round_trip_by_value():
    pair = eik.CollisionGeometryPair()
    pair.geometry_a = "base_sphere"
    pair.geometry_b = "moving_sphere"

    override = eik.CollisionPairMinimumDistance()
    override.pair = pair
    override.min_distance = 0.04

    definition = eik.CollisionConstraintDefinition()
    definition.min_distance = 0.02
    definition.include_pairs = [pair]
    definition.exclude_pairs = [pair]
    definition.pair_minimum_distances = [override]

    pair.geometry_a = "mutated_after_assignment"
    assert definition.include_pairs[0].geometry_a == "base_sphere"
    assert definition.pair_minimum_distances[0].pair.geometry_b == "moving_sphere"

    support = eik.ComSupportPolygonConstraintDefinition()
    support.support_polygon = np.array([[0.0, 0.0], [0.2, 0.0], [0.0, 0.2]], dtype=float)
    support.margin = 0.01
    support.frame_name = "world"
    support.proximity_fraction = 0.25
    support_copy = support.support_polygon
    support_copy[0, 0] = 9.0
    assert support.support_polygon[0, 0] == 0.0


def test_acceleration_option_records_round_trip_optionals_and_nested_values():
    allocation = eik.GeneralizedAccelerationAllocation()
    allocation.metric_diagonal = np.array([2.0, 3.0])
    allocation.reference_acceleration = np.array([0.1, -0.2])

    effort = eik.EffortConstraintOptions()
    assert effort.limits_override is None
    effort.limits_override = np.array([10.0, 11.0])
    effort.margin_fraction = 0.2

    options = eik.AccelerationSolveOptions()
    assert options.acceleration_limits_override is None
    options.acceleration_limits_override = np.array([5.0, 6.0])
    options.generalized_acceleration_allocation = allocation
    options.effort_constraints = effort
    options.zero_acceleration_joint_indices = [1]
    options.zero_next_velocity_joint_indices = [0]
    options.fixed_current_position_joint_indices = [0, 1]

    allocation.metric_diagonal = np.array([99.0, 99.0])
    assert np.allclose(options.generalized_acceleration_allocation.metric_diagonal, [2.0, 3.0])
    assert np.allclose(options.effort_constraints.limits_override, [10.0, 11.0])
    assert np.allclose(options.acceleration_limits_override, [5.0, 6.0])

    options.acceleration_limits_override = None
    options.generalized_acceleration_allocation = None
    options.effort_constraints = None
    assert options.acceleration_limits_override is None
    assert options.generalized_acceleration_allocation is None
    assert options.effort_constraints is None


def test_geometric_constraint_records_round_trip_by_value():
    policy = eik.GeometricConstraintAccelerationPolicy()
    policy.rate_limits = np.array([0.1, 0.2, 0.3])
    policy.acceleration_limits = np.array([1.0, 2.0, 3.0])
    policy.lower_braking_accelerations = np.array([0.4, 0.5, 0.6])
    policy.upper_braking_accelerations = np.array([0.7, 0.8, 0.9])

    tight_point_def = eik.TightPointConstraintDefinition()
    tight_point_def.frame_name = "tip"
    tight_point_def.target_point = np.array([1.0, 2.0, 3.0])
    tight_point = eik.TightPointAccelerationConstraint()
    tight_point.source_id = "tight_point"
    tight_point.definition = tight_point_def
    tight_point.policy = policy

    tight_point_def.target_point = np.zeros(3)
    policy.rate_limits = np.ones(3)

    assert np.allclose(tight_point.definition.target_point, [1.0, 2.0, 3.0])
    assert np.allclose(tight_point.policy.rate_limits, [0.1, 0.2, 0.3])

    torso_def = eik.TorsoPoseBoundDefinition()
    assert torso_def.reference_pose is None
    reference_pose = np.eye(4)
    reference_pose[0, 3] = 0.2
    torso_def.reference_pose = reference_pose
    assert torso_def.reference_pose[0, 3] == 0.2
    torso_def.reference_pose = None
    assert torso_def.reference_pose is None
