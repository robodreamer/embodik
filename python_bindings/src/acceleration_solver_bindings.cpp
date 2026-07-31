/**
 * @file acceleration_solver_bindings.cpp
 * @brief Python bindings for AccelerationSolver.
 */

#include <nanobind/eigen/dense.h>
#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <embodik/acceleration_solver.hpp>
#include <embodik/kinematics_solver.hpp>
#include <embodik/robot_model.hpp>
#include <embodik/tasks.hpp>

#include <utility>

namespace nb = nanobind;
using namespace embodik;

namespace {

#define DEF_VALUE_PROP(cls, name)                                              \
  .def_prop_rw(                                                               \
      #name, [](const cls &self) { return self.name; },                       \
      [](cls &self, decltype(cls::name) value) { self.name = std::move(value); })

void bind_shared_acceleration_definitions(nb::module_ &m) {
  nb::class_<CollisionGeometryPair>(m, "CollisionGeometryPair")
      .def(nb::init<>())
      .def_rw("geometry_a", &CollisionGeometryPair::geometry_a)
      .def_rw("geometry_b", &CollisionGeometryPair::geometry_b);

  nb::class_<CollisionPairMinimumDistance>(
      m, "CollisionPairMinimumDistance")
      .def(nb::init<>())
      DEF_VALUE_PROP(CollisionPairMinimumDistance, pair)
      .def_rw("min_distance", &CollisionPairMinimumDistance::min_distance);

  nb::class_<CollisionConstraintDefinition>(
      m, "CollisionConstraintDefinition")
      .def(nb::init<>())
      .def_rw("min_distance", &CollisionConstraintDefinition::min_distance)
      .def_rw("include_pairs", &CollisionConstraintDefinition::include_pairs)
      .def_rw("exclude_pairs", &CollisionConstraintDefinition::exclude_pairs)
      .def_rw("pair_minimum_distances",
              &CollisionConstraintDefinition::pair_minimum_distances);

  nb::class_<ComSupportPolygonConstraintDefinition>(
      m, "ComSupportPolygonConstraintDefinition")
      .def(nb::init<>())
      DEF_VALUE_PROP(ComSupportPolygonConstraintDefinition, support_polygon)
      .def_rw("margin", &ComSupportPolygonConstraintDefinition::margin)
      .def_rw("frame_name", &ComSupportPolygonConstraintDefinition::frame_name)
      .def_rw("proximity_fraction",
              &ComSupportPolygonConstraintDefinition::proximity_fraction);

  nb::class_<RelativePoseConstraintDefinition>(
      m, "RelativePoseConstraintDefinition")
      .def(nb::init<>())
      .def_rw("frame_a", &RelativePoseConstraintDefinition::frame_a)
      .def_rw("frame_b", &RelativePoseConstraintDefinition::frame_b)
      DEF_VALUE_PROP(RelativePoseConstraintDefinition, lower_bounds)
      DEF_VALUE_PROP(RelativePoseConstraintDefinition, upper_bounds)
      DEF_VALUE_PROP(RelativePoseConstraintDefinition, axis_mask);

  nb::class_<TightPointConstraintDefinition>(
      m, "TightPointConstraintDefinition")
      .def(nb::init<>())
      .def_rw("frame_name", &TightPointConstraintDefinition::frame_name)
      DEF_VALUE_PROP(TightPointConstraintDefinition, target_point)
      .def_rw("position_epsilon",
              &TightPointConstraintDefinition::position_epsilon)
      DEF_VALUE_PROP(TightPointConstraintDefinition, axis_mask);

  nb::class_<TightFramePoseConstraintDefinition>(
      m, "TightFramePoseConstraintDefinition")
      .def(nb::init<>())
      .def_rw("frame_name", &TightFramePoseConstraintDefinition::frame_name)
      DEF_VALUE_PROP(TightFramePoseConstraintDefinition, target_pose)
      .def_rw("position_epsilon",
              &TightFramePoseConstraintDefinition::position_epsilon)
      .def_rw("orientation_epsilon",
              &TightFramePoseConstraintDefinition::orientation_epsilon)
      DEF_VALUE_PROP(TightFramePoseConstraintDefinition, axis_mask);

  nb::class_<TorsoPoseBoundDefinition>(m, "TorsoPoseBoundDefinition")
      .def(nb::init<>())
      .def_rw("frame_name", &TorsoPoseBoundDefinition::frame_name)
      .def_prop_rw(
          "reference_pose",
          [](const TorsoPoseBoundDefinition &self) {
            return self.reference_pose;
          },
          [](TorsoPoseBoundDefinition &self,
             std::optional<Eigen::Matrix4d> value) {
            self.reference_pose = std::move(value);
          })
      DEF_VALUE_PROP(TorsoPoseBoundDefinition, lower_bounds)
      DEF_VALUE_PROP(TorsoPoseBoundDefinition, upper_bounds)
      DEF_VALUE_PROP(TorsoPoseBoundDefinition, axis_mask);
}

} // namespace

void bind_acceleration_solver(nb::module_ &m) {
  bind_shared_acceleration_definitions(m);

  nb::enum_<ComSupportPolygonOutsidePolicy>(
      m, "ComSupportPolygonOutsidePolicy")
      .value("REJECT", ComSupportPolygonOutsidePolicy::kReject)
      .value("RECOVER_NON_WORSENING",
             ComSupportPolygonOutsidePolicy::kRecoverNonWorsening);

  nb::enum_<CollisionConstraintOutsidePolicy>(
      m, "CollisionConstraintOutsidePolicy")
      .value("REJECT", CollisionConstraintOutsidePolicy::kReject)
      .value("RECOVER_NON_WORSENING",
             CollisionConstraintOutsidePolicy::kRecoverNonWorsening);

  nb::enum_<AccelerationCollisionRegime>(m, "AccelerationCollisionRegime")
      .value("UNKNOWN", AccelerationCollisionRegime::kUnknown)
      .value("STRICT_INTERIOR",
             AccelerationCollisionRegime::kStrictInterior)
      .value("EXACT_FLOOR", AccelerationCollisionRegime::kExactFloor)
      .value("BELOW_LIMIT_RECOVERY",
             AccelerationCollisionRegime::kBelowLimitRecovery);

  nb::class_<AffineAccelerationConstraint>(
      m, "AffineAccelerationConstraint")
      .def(nb::init<>())
      .def_rw("source_id", &AffineAccelerationConstraint::source_id)
      DEF_VALUE_PROP(AffineAccelerationConstraint, coefficient_matrix)
      DEF_VALUE_PROP(AffineAccelerationConstraint, affine_bias)
      DEF_VALUE_PROP(AffineAccelerationConstraint, lower_bounds)
      DEF_VALUE_PROP(AffineAccelerationConstraint, upper_bounds)
      .def_rw("lower_bound_active",
              &AffineAccelerationConstraint::lower_bound_active)
      .def_rw("upper_bound_active",
              &AffineAccelerationConstraint::upper_bound_active);

  nb::class_<FrozenNextVelocityConstraint>(
      m, "FrozenNextVelocityConstraint")
      .def(nb::init<>())
      .def_rw("source_id", &FrozenNextVelocityConstraint::source_id)
      DEF_VALUE_PROP(FrozenNextVelocityConstraint, coefficient_matrix)
      DEF_VALUE_PROP(FrozenNextVelocityConstraint, affine_bias)
      DEF_VALUE_PROP(FrozenNextVelocityConstraint, lower_bounds)
      DEF_VALUE_PROP(FrozenNextVelocityConstraint, upper_bounds)
      .def_rw("lower_bound_active",
              &FrozenNextVelocityConstraint::lower_bound_active)
      .def_rw("upper_bound_active",
              &FrozenNextVelocityConstraint::upper_bound_active);

  nb::class_<TaskAccelerationBounds>(m, "TaskAccelerationBounds")
      .def(nb::init<>())
      .def_rw("source_id", &TaskAccelerationBounds::source_id)
      .def_rw("task_name", &TaskAccelerationBounds::task_name)
      DEF_VALUE_PROP(TaskAccelerationBounds, lower_bounds)
      DEF_VALUE_PROP(TaskAccelerationBounds, upper_bounds)
      .def_rw("lower_bound_active",
              &TaskAccelerationBounds::lower_bound_active)
      .def_rw("upper_bound_active",
              &TaskAccelerationBounds::upper_bound_active);

  nb::class_<CentroidalMomentumRateObjective>(
      m, "CentroidalMomentumRateObjective",
      "Fixed-base physical hdot objective: Ag*ddq + dAg*dq.")
      .def(nb::init<>())
      .def_rw("source_id", &CentroidalMomentumRateObjective::source_id)
      DEF_VALUE_PROP(CentroidalMomentumRateObjective, h_target)
      DEF_VALUE_PROP(CentroidalMomentumRateObjective, hdot_feedforward)
      .def_rw("proportional_gain",
              &CentroidalMomentumRateObjective::proportional_gain)
      .def_rw("axis_mask", &CentroidalMomentumRateObjective::axis_mask)
      .def_rw("priority", &CentroidalMomentumRateObjective::priority)
      .def_rw("solve_mode", &CentroidalMomentumRateObjective::solve_mode)
      .def_rw("allow_min_error_fallback",
              &CentroidalMomentumRateObjective::allow_min_error_fallback);

  nb::class_<CentroidalMomentumRateBounds>(
      m, "CentroidalMomentumRateBounds",
      "Selected-axis hard bounds on physical centroidal momentum rate.")
      .def(nb::init<>())
      .def_rw("source_id", &CentroidalMomentumRateBounds::source_id)
      DEF_VALUE_PROP(CentroidalMomentumRateBounds, lower_bounds)
      DEF_VALUE_PROP(CentroidalMomentumRateBounds, upper_bounds)
      .def_rw("axis_mask", &CentroidalMomentumRateBounds::axis_mask)
      .def_rw("lower_bound_active",
              &CentroidalMomentumRateBounds::lower_bound_active)
      .def_rw("upper_bound_active",
              &CentroidalMomentumRateBounds::upper_bound_active);

  nb::class_<GeneralizedAccelerationAllocation>(
      m, "GeneralizedAccelerationAllocation")
      .def(nb::init<>())
      DEF_VALUE_PROP(GeneralizedAccelerationAllocation, metric_diagonal)
      DEF_VALUE_PROP(GeneralizedAccelerationAllocation, reference_acceleration);

  nb::class_<EffortConstraintOptions>(m, "EffortConstraintOptions")
      .def(nb::init<>())
      .def_prop_rw(
          "limits_override",
          [](const EffortConstraintOptions &self) {
            return self.limits_override;
          },
          [](EffortConstraintOptions &self,
             std::optional<Eigen::VectorXd> value) {
            self.limits_override = std::move(value);
          })
      .def_rw("margin_fraction", &EffortConstraintOptions::margin_fraction);

  nb::class_<ContactAccelerationConstraint>(
      m, "ContactAccelerationConstraint")
      .def(nb::init<>())
      .def_rw("source_id", &ContactAccelerationConstraint::source_id)
      .def_rw("frame_name", &ContactAccelerationConstraint::frame_name)
      .def_rw("type", &ContactAccelerationConstraint::type);

  nb::class_<GeometricConstraintAccelerationPolicy>(
      m, "GeometricConstraintAccelerationPolicy")
      .def(nb::init<>())
      DEF_VALUE_PROP(GeometricConstraintAccelerationPolicy, rate_limits)
      DEF_VALUE_PROP(GeometricConstraintAccelerationPolicy, acceleration_limits)
      DEF_VALUE_PROP(GeometricConstraintAccelerationPolicy,
                     lower_braking_accelerations)
      DEF_VALUE_PROP(GeometricConstraintAccelerationPolicy,
                     upper_braking_accelerations);

  nb::class_<ComSupportPolygonAccelerationPolicy>(
      m, "ComSupportPolygonAccelerationPolicy")
      .def(nb::init<>())
      .def_rw("rate_limit", &ComSupportPolygonAccelerationPolicy::rate_limit)
      .def_rw("acceleration_limit",
              &ComSupportPolygonAccelerationPolicy::acceleration_limit)
      .def_rw("braking_acceleration",
              &ComSupportPolygonAccelerationPolicy::braking_acceleration)
      .def_rw("boundary_epsilon",
              &ComSupportPolygonAccelerationPolicy::boundary_epsilon)
      .def_rw("outside_recovery_scale",
              &ComSupportPolygonAccelerationPolicy::outside_recovery_scale)
      .def_rw("outside_min_recovery_speed",
              &ComSupportPolygonAccelerationPolicy::outside_min_recovery_speed)
      .def_rw("outside_policy",
              &ComSupportPolygonAccelerationPolicy::outside_policy);

  nb::class_<ComSupportPolygonAccelerationConstraint>(
      m, "ComSupportPolygonAccelerationConstraint")
      .def(nb::init<>())
      .def_rw("source_id", &ComSupportPolygonAccelerationConstraint::source_id)
      DEF_VALUE_PROP(ComSupportPolygonAccelerationConstraint, definition)
      DEF_VALUE_PROP(ComSupportPolygonAccelerationConstraint, policy);

  nb::class_<CapturePointAccelerationConstraint>(
      m, "CapturePointAccelerationConstraint",
      "Predicted capture-point polygon constraint with per-solve frozen omega.")
      .def(nb::init<>())
      .def_rw("source_id", &CapturePointAccelerationConstraint::source_id)
      DEF_VALUE_PROP(CapturePointAccelerationConstraint, definition)
      .def_prop_rw(
          "omega",
          [](const CapturePointAccelerationConstraint &self) {
            return self.omega;
          },
          [](CapturePointAccelerationConstraint &self,
             std::optional<double> value) { self.omega = value; });

  nb::class_<ZmpAccelerationConstraint>(
      m, "ZmpAccelerationConstraint",
      "Physical centroidal-rate ZMP polygon rows with a positive Fz gate.")
      .def(nb::init<>())
      .def_rw("source_id", &ZmpAccelerationConstraint::source_id)
      DEF_VALUE_PROP(ZmpAccelerationConstraint, definition)
      .def_rw("fz_min", &ZmpAccelerationConstraint::fz_min);

  nb::class_<TightPointAccelerationConstraint>(
      m, "TightPointAccelerationConstraint")
      .def(nb::init<>())
      .def_rw("source_id", &TightPointAccelerationConstraint::source_id)
      DEF_VALUE_PROP(TightPointAccelerationConstraint, definition)
      DEF_VALUE_PROP(TightPointAccelerationConstraint, policy);

  nb::class_<TightFramePoseAccelerationConstraint>(
      m, "TightFramePoseAccelerationConstraint")
      .def(nb::init<>())
      .def_rw("source_id",
              &TightFramePoseAccelerationConstraint::source_id)
      DEF_VALUE_PROP(TightFramePoseAccelerationConstraint, definition)
      DEF_VALUE_PROP(TightFramePoseAccelerationConstraint, policy);

  nb::class_<RelativePoseAccelerationConstraint>(
      m, "RelativePoseAccelerationConstraint")
      .def(nb::init<>())
      .def_rw("source_id", &RelativePoseAccelerationConstraint::source_id)
      DEF_VALUE_PROP(RelativePoseAccelerationConstraint, definition)
      DEF_VALUE_PROP(RelativePoseAccelerationConstraint, policy);

  nb::class_<TorsoPoseBoundAccelerationConstraint>(
      m, "TorsoPoseBoundAccelerationConstraint")
      .def(nb::init<>())
      .def_rw("source_id",
              &TorsoPoseBoundAccelerationConstraint::source_id)
      DEF_VALUE_PROP(TorsoPoseBoundAccelerationConstraint, definition)
      DEF_VALUE_PROP(TorsoPoseBoundAccelerationConstraint, policy);

  nb::class_<AccelerationSolveOptions>(m, "AccelerationSolveOptions")
      .def(nb::init<>())
      .def_prop_rw(
          "acceleration_limits_override",
          [](const AccelerationSolveOptions &self) {
            return self.acceleration_limits_override;
          },
          [](AccelerationSolveOptions &self,
             std::optional<Eigen::VectorXd> value) {
            self.acceleration_limits_override = std::move(value);
          })
      .def_rw("apply_velocity_limits",
              &AccelerationSolveOptions::apply_velocity_limits)
      .def_rw("apply_position_limits",
              &AccelerationSolveOptions::apply_position_limits)
      .def_rw("allow_state_box_task_fallback",
              &AccelerationSolveOptions::allow_state_box_task_fallback,
              "Allow an explicit MIN_ERROR fallback when the compatible "
              "state box requires nonzero braking acceleration.")
      .def_rw("collect_task_diagnostics",
              &AccelerationSolveOptions::collect_task_diagnostics,
              "Retain rich per-task acceleration diagnostics. Disable in "
              "latency-sensitive loops that only require scales and errors.")
      .def_prop_rw(
          "generalized_acceleration_allocation",
          [](const AccelerationSolveOptions &self) {
            return self.generalized_acceleration_allocation;
          },
          [](AccelerationSolveOptions &self,
             std::optional<GeneralizedAccelerationAllocation> value) {
            self.generalized_acceleration_allocation = std::move(value);
          })
      .def_prop_rw(
          "effort_constraints",
          [](const AccelerationSolveOptions &self) {
            return self.effort_constraints;
          },
          [](AccelerationSolveOptions &self,
             std::optional<EffortConstraintOptions> value) {
            self.effort_constraints = std::move(value);
          })
      .def_rw("affine_constraints",
              &AccelerationSolveOptions::affine_constraints)
      .def_rw("frozen_next_velocity_constraints",
              &AccelerationSolveOptions::frozen_next_velocity_constraints)
      .def_rw("task_acceleration_bounds",
              &AccelerationSolveOptions::task_acceleration_bounds)
      .def_rw("centroidal_momentum_rate_objectives",
              &AccelerationSolveOptions::
                  centroidal_momentum_rate_objectives)
      .def_rw("centroidal_momentum_rate_bounds",
              &AccelerationSolveOptions::centroidal_momentum_rate_bounds)
      .def_rw("contact_acceleration_constraints",
              &AccelerationSolveOptions::contact_acceleration_constraints)
      .def_rw("tight_point_constraints",
              &AccelerationSolveOptions::tight_point_constraints)
      .def_rw("tight_frame_pose_constraints",
              &AccelerationSolveOptions::tight_frame_pose_constraints)
      .def_rw("relative_pose_constraints",
              &AccelerationSolveOptions::relative_pose_constraints)
      .def_rw("torso_pose_bound_constraints",
              &AccelerationSolveOptions::torso_pose_bound_constraints)
      .def_rw("com_support_polygon_constraints",
              &AccelerationSolveOptions::com_support_polygon_constraints)
      .def_rw("capture_point_constraints",
              &AccelerationSolveOptions::capture_point_constraints)
      .def_rw("zmp_constraints",
              &AccelerationSolveOptions::zmp_constraints)
      .def_rw("zero_acceleration_joint_indices",
              &AccelerationSolveOptions::zero_acceleration_joint_indices)
      .def_rw("zero_next_velocity_joint_indices",
              &AccelerationSolveOptions::zero_next_velocity_joint_indices)
      .def_rw("fixed_current_position_joint_indices",
              &AccelerationSolveOptions::fixed_current_position_joint_indices);

  nb::class_<VelocityCollisionLiftOptions>(
      m, "VelocityCollisionLiftOptions")
      .def(nb::init<>())
      .def_rw("validation_substeps",
              &VelocityCollisionLiftOptions::validation_substeps);

  nb::class_<CollisionConstraintAccelerationPolicy>(
      m, "CollisionConstraintAccelerationPolicy")
      .def(nb::init<>())
      .def_rw("proximity_activation_enabled",
              &CollisionConstraintAccelerationPolicy::
                  proximity_activation_enabled)
      .def_rw("activation_margin",
              &CollisionConstraintAccelerationPolicy::activation_margin)
      .def_rw("maximum_approach_rate",
              &CollisionConstraintAccelerationPolicy::maximum_approach_rate)
      .def_rw("maximum_inward_acceleration",
              &CollisionConstraintAccelerationPolicy::
                  maximum_inward_acceleration)
      .def_rw("minimum_braking_acceleration",
              &CollisionConstraintAccelerationPolicy::
                  minimum_braking_acceleration)
      .def_rw("outside_policy",
              &CollisionConstraintAccelerationPolicy::outside_policy)
      .def_rw("recovery_scale",
              &CollisionConstraintAccelerationPolicy::recovery_scale)
      .def_rw("minimum_recovery_rate",
              &CollisionConstraintAccelerationPolicy::minimum_recovery_rate)
      .def_rw("maximum_recovery_rate",
              &CollisionConstraintAccelerationPolicy::maximum_recovery_rate);

  nb::class_<AccelerationTaskReference>(m, "AccelerationTaskReference")
      .def(nb::init<>())
      DEF_VALUE_PROP(AccelerationTaskReference, desired_velocity)
      DEF_VALUE_PROP(AccelerationTaskReference, desired_acceleration)
      .def_rw("proportional_gain",
              &AccelerationTaskReference::proportional_gain)
      .def_rw("derivative_gain", &AccelerationTaskReference::derivative_gain);

  nb::class_<AccelerationTaskDiagnostics>(
      m, "AccelerationTaskDiagnostics")
      .def_ro("task_name", &AccelerationTaskDiagnostics::task_name)
      .def_prop_ro("reference_acceleration",
                   [](const AccelerationTaskDiagnostics &self) {
                     return self.reference_acceleration;
                   })
      .def_prop_ro("jacobian_bias",
                   [](const AccelerationTaskDiagnostics &self) {
                     return self.jacobian_bias;
                   })
      .def_prop_ro("achieved_acceleration",
                   [](const AccelerationTaskDiagnostics &self) {
                     return self.achieved_acceleration;
                   })
      .def_prop_ro("residual", [](const AccelerationTaskDiagnostics &self) {
        return self.residual;
      })
      .def_ro("scale", &AccelerationTaskDiagnostics::scale)
      .def_ro("effective_mode", &AccelerationTaskDiagnostics::effective_mode)
      .def_ro("used_min_error_fallback",
              &AccelerationTaskDiagnostics::used_min_error_fallback);

  nb::class_<AccelerationAllocationDiagnostics>(
      m, "AccelerationAllocationDiagnostics")
      .def_ro("applied", &AccelerationAllocationDiagnostics::applied)
      .def_prop_ro("physical_metric_diagonal",
                   [](const AccelerationAllocationDiagnostics &self) {
                     return self.physical_metric_diagonal;
                   })
      .def_prop_ro("reference_acceleration",
                   [](const AccelerationAllocationDiagnostics &self) {
                     return self.reference_acceleration;
                   })
      .def_prop_ro("weighted_physical_residual",
                   [](const AccelerationAllocationDiagnostics &self) {
                     return self.weighted_physical_residual;
                   })
      .def_ro("objective_value",
              &AccelerationAllocationDiagnostics::objective_value);

  nb::class_<CentroidalMomentumRateDiagnostics>(
      m, "CentroidalMomentumRateDiagnostics")
      .def_ro("source_id", &CentroidalMomentumRateDiagnostics::source_id)
      .def_prop_ro("target_momentum",
                   [](const CentroidalMomentumRateDiagnostics &self) {
                     return self.target_momentum;
                   })
      .def_prop_ro("reference_momentum_rate",
                   [](const CentroidalMomentumRateDiagnostics &self) {
                     return self.reference_momentum_rate;
                   })
      .def_prop_ro("current_momentum",
                   [](const CentroidalMomentumRateDiagnostics &self) {
                     return self.current_momentum;
                   })
      .def_prop_ro("bias_momentum_rate",
                   [](const CentroidalMomentumRateDiagnostics &self) {
                     return self.bias_momentum_rate;
                   })
      .def_prop_ro("achieved_momentum_rate",
                   [](const CentroidalMomentumRateDiagnostics &self) {
                     return self.achieved_momentum_rate;
                   })
      .def_prop_ro("residual",
                   [](const CentroidalMomentumRateDiagnostics &self) {
                     return self.residual;
                   })
      .def_ro("selected_axes",
              &CentroidalMomentumRateDiagnostics::selected_axes)
      .def_ro("scale", &CentroidalMomentumRateDiagnostics::scale)
      .def_ro("effective_mode",
              &CentroidalMomentumRateDiagnostics::effective_mode)
      .def_ro("used_min_error_fallback",
              &CentroidalMomentumRateDiagnostics::used_min_error_fallback);

  nb::class_<CapturePointAccelerationDiagnostics>(
      m, "CapturePointAccelerationDiagnostics")
      .def_ro("source_id", &CapturePointAccelerationDiagnostics::source_id)
      .def_prop_ro("predicted_point_xy",
                   [](const CapturePointAccelerationDiagnostics &self) {
                     return self.predicted_point_xy;
                   })
      .def_prop_ro("half_plane_slacks",
                   [](const CapturePointAccelerationDiagnostics &self) {
                     return self.half_plane_slacks;
                   })
      .def_ro("min_slack", &CapturePointAccelerationDiagnostics::min_slack)
      .def_ro("frozen_omega",
              &CapturePointAccelerationDiagnostics::frozen_omega)
      .def_ro("postvalidated",
              &CapturePointAccelerationDiagnostics::postvalidated);

  nb::class_<ZmpAccelerationDiagnostics>(m, "ZmpAccelerationDiagnostics")
      .def_ro("source_id", &ZmpAccelerationDiagnostics::source_id)
      .def_prop_ro("predicted_point_xy",
                   [](const ZmpAccelerationDiagnostics &self) {
                     return self.predicted_point_xy;
                   })
      .def_prop_ro("half_plane_slacks",
                   [](const ZmpAccelerationDiagnostics &self) {
                     return self.half_plane_slacks;
                   })
      .def_ro("min_slack", &ZmpAccelerationDiagnostics::min_slack)
      .def_ro("force_z", &ZmpAccelerationDiagnostics::force_z)
      .def_ro("postvalidated", &ZmpAccelerationDiagnostics::postvalidated);

  nb::class_<AccelerationAnalyticCollisionPairDiagnostics>(
      m, "AccelerationAnalyticCollisionPairDiagnostics")
      .def_ro("pair_index",
              &AccelerationAnalyticCollisionPairDiagnostics::pair_index)
      .def_ro("pair_key",
              &AccelerationAnalyticCollisionPairDiagnostics::pair_key)
      .def_ro("object_a",
              &AccelerationAnalyticCollisionPairDiagnostics::object_a)
      .def_ro("object_b",
              &AccelerationAnalyticCollisionPairDiagnostics::object_b)
      .def_ro("regime", &AccelerationAnalyticCollisionPairDiagnostics::regime)
      .def_ro("minimum_distance",
              &AccelerationAnalyticCollisionPairDiagnostics::minimum_distance)
      .def_ro("current_signed_distance",
              &AccelerationAnalyticCollisionPairDiagnostics::
                  current_signed_distance)
      .def_ro("current_signed_distance_lower_bound",
              &AccelerationAnalyticCollisionPairDiagnostics::
                  current_signed_distance_lower_bound)
      .def_ro("current_signed_distance_upper_bound",
              &AccelerationAnalyticCollisionPairDiagnostics::
                  current_signed_distance_upper_bound)
      .def_ro("current_rate_lower_bound",
              &AccelerationAnalyticCollisionPairDiagnostics::
                  current_rate_lower_bound)
      .def_ro("current_rate_upper_bound",
              &AccelerationAnalyticCollisionPairDiagnostics::
                  current_rate_upper_bound)
      .def_ro("endpoint_signed_distance_lower_bound",
              &AccelerationAnalyticCollisionPairDiagnostics::
                  endpoint_signed_distance_lower_bound)
      .def_ro("lowest_path_signed_distance_lower_bound",
              &AccelerationAnalyticCollisionPairDiagnostics::
                  lowest_path_signed_distance_lower_bound)
      .def_ro("state_rate_shaping_active",
              &AccelerationAnalyticCollisionPairDiagnostics::
                  state_rate_shaping_active)
      .def_ro("braking_witness_required",
              &AccelerationAnalyticCollisionPairDiagnostics::
                  braking_witness_required)
      .def_ro("step_certified",
              &AccelerationAnalyticCollisionPairDiagnostics::step_certified);

  nb::class_<AccelerationSolverResult, SolverResult>(
      m, "AccelerationSolverResult")
      .def_ro("preprocessing_time_ms",
              &AccelerationSolverResult::preprocessing_time_ms)
      .def_ro("backend_computation_time_ms",
              &AccelerationSolverResult::backend_computation_time_ms)
      .def_ro("postprocessing_time_ms",
              &AccelerationSolverResult::postprocessing_time_ms)
      .def_prop_ro("joint_accelerations",
                   [](const AccelerationSolverResult &self) {
                     return self.joint_accelerations;
                   })
      .def_prop_ro("joint_velocities_next",
                   [](const AccelerationSolverResult &self) {
                     return self.joint_velocities_next;
                   })
      .def_prop_ro("q_solution",
                   [](const AccelerationSolverResult &self) {
                     return self.q_solution;
                   })
      .def_ro("acceleration_limits_applied",
              &AccelerationSolverResult::acceleration_limits_applied)
      .def_ro("state_box_task_fallback_applied",
              &AccelerationSolverResult::state_box_task_fallback_applied)
      .def_ro("saturated_acceleration_indices",
              &AccelerationSolverResult::saturated_acceleration_indices)
      .def_ro("saturated_velocity_indices",
              &AccelerationSolverResult::saturated_velocity_indices)
      .def_ro("saturated_position_indices",
              &AccelerationSolverResult::saturated_position_indices)
      .def_prop_ro("predicted_torques",
                   [](const AccelerationSolverResult &self) {
                     return self.predicted_torques;
                   })
      .def_ro("effort_limits_applied",
              &AccelerationSolverResult::effort_limits_applied)
      .def_ro("saturated_effort_indices",
              &AccelerationSolverResult::saturated_effort_indices)
      .def_ro("task_diagnostics",
              &AccelerationSolverResult::task_diagnostics)
      .def_prop_ro("allocation_diagnostics",
                   [](const AccelerationSolverResult &self) {
                     return self.allocation_diagnostics;
                   })
      .def_ro("centroidal_momentum_rate_diagnostics",
              &AccelerationSolverResult::
                  centroidal_momentum_rate_diagnostics)
      .def_ro("capture_point_diagnostics",
              &AccelerationSolverResult::capture_point_diagnostics)
      .def_ro("zmp_diagnostics",
              &AccelerationSolverResult::zmp_diagnostics)
      .def_ro("velocity_collision_lift_applied",
              &AccelerationSolverResult::velocity_collision_lift_applied)
      .def_ro("collision_endpoint_validated",
              &AccelerationSolverResult::collision_endpoint_validated)
      .def_ro("collision_step_certified",
              &AccelerationSolverResult::collision_step_certified)
      .def_ro("collision_validation_samples",
              &AccelerationSolverResult::collision_validation_samples)
      .def_ro("collision_validation_allowed_pairs",
              &AccelerationSolverResult::collision_validation_allowed_pairs)
      .def_ro("collision_validation_pairs_checked",
              &AccelerationSolverResult::collision_validation_pairs_checked)
      .def_ro("collision_validation_exact_queries",
              &AccelerationSolverResult::collision_validation_exact_queries)
      .def_ro("collision_validation_initial_exact_queries",
              &AccelerationSolverResult::
                  collision_validation_initial_exact_queries)
      .def_ro("collision_validation_sample_exact_queries",
              &AccelerationSolverResult::
                  collision_validation_sample_exact_queries)
      .def_ro("collision_validation_conservative_checks",
              &AccelerationSolverResult::
                  collision_validation_conservative_checks)
      .def_ro("collision_validation_conservative_certified_pairs",
              &AccelerationSolverResult::
                  collision_validation_conservative_certified_pairs)
      .def_ro("collision_validation_kinematics_updates",
              &AccelerationSolverResult::
                  collision_validation_kinematics_updates)
      .def_ro("collision_validation_geometry_updates",
              &AccelerationSolverResult::
                  collision_validation_geometry_updates)
      .def_ro("collision_validation_initial_certificate_reused",
              &AccelerationSolverResult::
                  collision_validation_initial_certificate_reused)
      .def_ro("collision_lift_pairs_considered",
              &AccelerationSolverResult::collision_lift_pairs_considered)
      .def_ro("collision_lift_row_pairs",
              &AccelerationSolverResult::collision_lift_row_pairs)
      .def_ro("collision_lift_row_exact_queries",
              &AccelerationSolverResult::collision_lift_row_exact_queries)
      .def_ro("native_collision_constraint_applied",
              &AccelerationSolverResult::native_collision_constraint_applied)
      .def_ro("native_collision_diagnostics",
              &AccelerationSolverResult::native_collision_diagnostics)
      .def_ro("native_collision_pair_evaluations",
              &AccelerationSolverResult::native_collision_pair_evaluations)
      .def_ro("native_collision_path_visited_nodes",
              &AccelerationSolverResult::native_collision_path_visited_nodes)
      .def_ro("native_collision_certified_intervals",
              &AccelerationSolverResult::native_collision_certified_intervals);

  nb::class_<AccelerationSolverCapabilities>(
      m, "AccelerationSolverCapabilities")
      .def_ro("supports_fixed_base_scalar_joints",
              &AccelerationSolverCapabilities::
                  supports_fixed_base_scalar_joints)
      .def_ro("supports_floating_base",
              &AccelerationSolverCapabilities::supports_floating_base)
      .def_ro("supports_scale_elastic",
              &AccelerationSolverCapabilities::supports_scale_elastic)
      .def_ro("supports_collision_constraints",
              &AccelerationSolverCapabilities::supports_collision_constraints)
      .def_ro("supports_analytic_sphere_collision_constraints",
              &AccelerationSolverCapabilities::
                  supports_analytic_sphere_collision_constraints)
      .def_ro("supports_effort_constraints",
              &AccelerationSolverCapabilities::supports_effort_constraints)
      .def_ro("supports_fixed_base_contact_kinematics",
              &AccelerationSolverCapabilities::
                  supports_fixed_base_contact_kinematics)
      .def_ro("supports_tight_point_constraints",
              &AccelerationSolverCapabilities::supports_tight_point_constraints)
      .def_ro("supports_tight_frame_pose_constraints",
              &AccelerationSolverCapabilities::
                  supports_tight_frame_pose_constraints)
      .def_ro("supports_relative_pose_constraints",
              &AccelerationSolverCapabilities::
                  supports_relative_pose_constraints)
      .def_ro("supports_torso_pose_bound_constraints",
              &AccelerationSolverCapabilities::
                  supports_torso_pose_bound_constraints)
      .def_ro("supports_com_support_polygon_constraints",
              &AccelerationSolverCapabilities::
                  supports_com_support_polygon_constraints)
      .def_ro("supports_fixed_base_capture_point_constraints",
              &AccelerationSolverCapabilities::
                  supports_fixed_base_capture_point_constraints)
      .def_ro("supports_fixed_base_zmp_constraints",
              &AccelerationSolverCapabilities::
                  supports_fixed_base_zmp_constraints)
      .def_ro("supports_fixed_base_centroidal_momentum_rate_objective",
              &AccelerationSolverCapabilities::
                  supports_fixed_base_centroidal_momentum_rate_objective)
      .def_ro("supports_fixed_base_centroidal_momentum_rate_bounds",
              &AccelerationSolverCapabilities::
                  supports_fixed_base_centroidal_momentum_rate_bounds)
      .def_ro("supports_floating_base_centroidal_momentum_rate",
              &AccelerationSolverCapabilities::
                  supports_floating_base_centroidal_momentum_rate)
      .def_ro("supports_velocity_collision_lift",
              &AccelerationSolverCapabilities::
                  supports_velocity_collision_lift)
      .def_ro("supports_dynamic_contact",
              &AccelerationSolverCapabilities::supports_dynamic_contact)
      .def_ro("supports_dynamic_balance",
              &AccelerationSolverCapabilities::supports_dynamic_balance);

  nb::class_<AccelerationSolver>(m, "AccelerationSolver")
      .def(nb::init<std::shared_ptr<RobotModel>>(), nb::arg("robot"))
      .def_static("capabilities", &AccelerationSolver::capabilities)
      .def("add_frame_task", &AccelerationSolver::add_frame_task,
           nb::arg("name"), nb::arg("frame_name"),
           nb::arg("task_type") = TaskType::FRAME_POSE)
      .def("add_com_task", &AccelerationSolver::add_com_task, nb::arg("name"))
      .def("add_posture_task", &AccelerationSolver::add_posture_task,
           nb::arg("name"),
           nb::arg("controlled_joints") = std::vector<int>{})
      .def("add_joint_task", &AccelerationSolver::add_joint_task,
           nb::arg("name"), nb::arg("joint_name"),
           nb::arg("target_value") = 0.0)
      .def("set_task_reference", &AccelerationSolver::set_task_reference,
           nb::arg("name"), nb::arg("reference"))
      .def("get_task", &AccelerationSolver::get_task, nb::arg("name"))
      .def("remove_task", &AccelerationSolver::remove_task, nb::arg("name"))
      .def("clear_tasks", &AccelerationSolver::clear_tasks)
      .def("configure_collision_constraint",
           &AccelerationSolver::configure_collision_constraint,
           nb::arg("definition"), nb::arg("policy"))
      .def("clear_collision_constraint",
           &AccelerationSolver::clear_collision_constraint)
      .def("has_collision_constraint",
           &AccelerationSolver::has_collision_constraint)
      .def("get_collision_min_distance",
           &AccelerationSolver::get_collision_min_distance)
      .def("get_collision_constraint_definition",
           &AccelerationSolver::get_collision_constraint_definition)
      .def("get_collision_constraint_policy",
           &AccelerationSolver::get_collision_constraint_policy)
      .def("get_active_collision_pairs",
           &AccelerationSolver::get_active_collision_pairs)
      .def("solve", &AccelerationSolver::solve, nb::arg("q"), nb::arg("dq"),
           nb::arg("dt"),
           nb::arg("options") = AccelerationSolveOptions{})
      .def("solve_with_velocity_collision",
           &AccelerationSolver::solve_with_velocity_collision,
           nb::arg("collision_solver"), nb::arg("q"), nb::arg("dq"),
           nb::arg("dt"),
           nb::arg("options") = AccelerationSolveOptions{},
           nb::arg("lift_options") = VelocityCollisionLiftOptions{});
}

#undef DEF_VALUE_PROP
