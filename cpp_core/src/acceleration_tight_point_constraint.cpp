#include "acceleration_tight_point_constraint.hpp"

#include "acceleration_geometric_constraint_policy.hpp"
#include "frame_kinematic_differential.hpp"

#include <cmath>
#include <exception>
#include <pinocchio/multibody/data.hpp>
#include <string>
#include <utility>

namespace embodik::detail {
namespace {

TightPointConstraintResult failure(SolverStatus status, std::string message) {
  TightPointConstraintResult result;
  result.status = status;
  result.message = std::move(message);
  return result;
}

TightPointConstraintResult as_tight_point_result(
    const ScalarGeometricConstraintResult &result) {
  return failure(result.status, result.message);
}

ScalarGeometricConstraintSpecification make_scalar_specification(
    const TightPointAccelerationConstraint &constraint,
    std::vector<int> active_axes) {
  ScalarGeometricConstraintSpecification specification;
  specification.family_label = "tight point constraint";
  specification.source_id = constraint.source_id;
  specification.dimension = 3;
  specification.active_axes = std::move(active_axes);
  specification.state_lower_bounds =
      Eigen::Vector3d::Constant(-constraint.definition.position_epsilon);
  specification.state_upper_bounds =
      Eigen::Vector3d::Constant(constraint.definition.position_epsilon);
  specification.policy = constraint.policy;
  return specification;
}

Eigen::Vector3d point_coordinate_value(
    const FrameKinematicDifferential &differential,
    const TightPointConstraintDefinition &definition) {
  return differential.pose.translation() - definition.target_point;
}

Eigen::Vector3d point_coordinate_rate(
    const FrameKinematicDifferential &differential,
    const Eigen::VectorXd &dq) {
  return differential.jacobian.topRows<3>() * dq;
}

ScalarGeometricCoordinateSample point_coordinate_sample(
    const TightPointAccelerationConstraint &constraint,
    const FrameKinematicDifferential &differential,
    const Eigen::VectorXd &dq, const Eigen::VectorXd &ddq) {
  ScalarGeometricCoordinateSample sample;
  sample.value = point_coordinate_value(differential, constraint.definition);
  sample.rate = point_coordinate_rate(differential, dq);
  sample.acceleration =
      differential.jacobian.topRows<3>() * ddq +
      differential.affine_bias.head<3>();
  return sample;
}

ScalarGeometricDifferential make_point_scalar_differential(
    const TightPointAccelerationConstraint &constraint,
    const FrameKinematicDifferential &differential,
    const Eigen::VectorXd &dq, const Eigen::VectorXd &ddq) {
  ScalarGeometricDifferential scalar;
  scalar.coefficient_matrix = differential.jacobian.topRows<3>();
  scalar.affine_bias = differential.affine_bias.head<3>();
  scalar.coordinates =
      point_coordinate_sample(constraint, differential, dq, ddq);
  return scalar;
}

TightPointConstraintResult validate_nonlinear_path(
    const PreparedTightPointConstraint &prepared, const RobotModel &robot,
    const Eigen::VectorXd &q, const Eigen::VectorXd &dq,
    const Eigen::VectorXd &ddq, double dt) {
  pinocchio::Data scratch(robot.model());
  const ScalarGeometricPathValidationOptions options;
  const auto path = validate_scalar_geometric_path(
      prepared.scalar.specification, dq, ddq, dt, options,
      [&](const ScalarGeometricPathPoint &path_point,
          ScalarGeometricCoordinateSample *coordinates) {
        Eigen::VectorXd sample_q;
        try {
          sample_q = robot.integrate(q, path_point.tangent);
        } catch (const std::exception &error) {
          return ScalarGeometricConstraintResult{
              SolverStatus::kNumericalError,
              "accepted tight point constraint '" +
                  prepared.specification.source_id +
                  "' nonlinear path integration failed: " + error.what()};
        }

        FrameKinematicDifferential differential;
        try {
          differential = evaluate_frame_kinematic_differential_at_state(
              robot, scratch, prepared.specification.definition.frame_name,
              sample_q, path_point.rate);
        } catch (const std::exception &error) {
          return ScalarGeometricConstraintResult{
              SolverStatus::kNumericalError,
              "accepted tight point constraint '" +
                  prepared.specification.source_id +
                  "' nonlinear path evaluation failed: " + error.what()};
        }
        *coordinates = point_coordinate_sample(prepared.specification,
                                               differential, path_point.rate,
                                               ddq);
        return ScalarGeometricConstraintResult{};
      });
  return as_tight_point_result(path);
}

} // namespace

TightPointConstraintResult validate_tight_point_constraint(
    const TightPointAccelerationConstraint &constraint,
    const RobotModel &robot) {
  const auto &definition = constraint.definition;
  const auto &policy = constraint.policy;
  if (constraint.source_id.empty()) {
    return failure(SolverStatus::kInvalidInput,
                   "tight point constraint source_id must not be empty");
  }
  if (definition.frame_name.empty() || !robot.has_frame(definition.frame_name)) {
    return failure(SolverStatus::kInvalidInput,
                   "tight point constraint '" + constraint.source_id +
                       "' frame_name must identify a robot frame");
  }
  if (!definition.target_point.allFinite() ||
      !std::isfinite(definition.position_epsilon)) {
    return failure(SolverStatus::kNonFiniteInput,
                   "tight point constraint '" + constraint.source_id +
                       "' definition must contain only finite values");
  }
  if (definition.position_epsilon <= 0.0) {
    return failure(SolverStatus::kInvalidInput,
                   "tight point constraint '" + constraint.source_id +
                       "' position_epsilon must be greater than zero");
  }
  const Eigen::VectorXd axis_mask = definition.axis_mask;
  const auto axis_validation = validate_geometric_constraint_axis_mask(
      "tight point constraint", constraint.source_id, axis_mask, 3);
  if (!axis_validation.satisfied()) {
    return failure(axis_validation.status, axis_validation.message);
  }
  const auto policy_validation = validate_geometric_constraint_policy(
      "tight point constraint", constraint.source_id, policy, 3);
  if (!policy_validation.satisfied()) {
    return failure(policy_validation.status, policy_validation.message);
  }
  return {};
}

TightPointPreparationResult prepare_tight_point_constraint(
    const TightPointAccelerationConstraint &constraint, const RobotModel &robot,
    double dt, const Eigen::VectorXd &joint_acceleration_lower,
    const Eigen::VectorXd &joint_acceleration_upper) {
  TightPointPreparationResult result;
  const auto validation = validate_tight_point_constraint(constraint, robot);
  if (!validation.satisfied()) {
    result.status = validation.status;
    result.message = validation.message;
    return result;
  }

  const Eigen::VectorXd axis_mask = constraint.definition.axis_mask;
  auto axis_validation = validate_geometric_constraint_axis_mask(
      "tight point constraint", constraint.source_id, axis_mask, 3);
  if (!axis_validation.satisfied()) {
    result.status = axis_validation.status;
    result.message = axis_validation.message;
    return result;
  }
  std::vector<int> active_axes = std::move(axis_validation.active_axes);

  FrameKinematicDifferential differential;
  try {
    differential =
        evaluate_frame_kinematic_differential(robot,
                                              constraint.definition.frame_name);
  } catch (const std::exception &error) {
    result.status = SolverStatus::kNumericalError;
    result.message = "tight point constraint '" + constraint.source_id +
                     "' evaluation failed: " + error.what();
    return result;
  }
  if (differential.jacobian.rows() != 6 ||
      differential.jacobian.cols() != robot.nv() ||
      differential.affine_bias.size() != 6 ||
      !differential.pose.translation().allFinite() ||
      !differential.jacobian.allFinite() ||
      !differential.affine_bias.allFinite()) {
    result.status = SolverStatus::kNumericalError;
    result.message = "tight point constraint '" + constraint.source_id +
                     "' differential was non-finite or inconsistent";
    return result;
  }

  const Eigen::VectorXd zero_acceleration =
      Eigen::VectorXd::Zero(robot.nv());
  auto scalar_differential = make_point_scalar_differential(
      constraint, differential, robot.get_current_velocity(),
      zero_acceleration);
  if (!scalar_differential.coordinates.value.allFinite() ||
      !scalar_differential.coordinates.rate.allFinite()) {
    result.status = SolverStatus::kNumericalError;
    result.message = "tight point constraint '" + constraint.source_id +
                     "' state or rate was non-finite";
    return result;
  }

  const auto scalar_specification =
      make_scalar_specification(constraint, active_axes);
  auto scalar_preparation = prepare_scalar_geometric_constraint(
      scalar_specification,
      scalar_differential, dt, robot.nv(), joint_acceleration_lower,
      joint_acceleration_upper);
  if (!scalar_preparation.satisfied()) {
    result.status = scalar_preparation.status;
    result.message = scalar_preparation.message;
    return result;
  }

  PreparedTightPointConstraint prepared;
  prepared.specification = constraint;
  prepared.scalar = std::move(scalar_preparation.prepared.value());
  prepared.active_axes = std::move(active_axes);
  result.prepared = std::move(prepared);
  return result;
}

TightPointConstraintResult validate_tight_point_constraint_acceptance(
    const PreparedTightPointConstraint &prepared, const RobotModel &robot,
    const Eigen::VectorXd &q, const Eigen::VectorXd &dq,
    const Eigen::VectorXd &accepted_acceleration, double dt,
    const Eigen::VectorXd &predicted_joint_acceleration_lower,
    const Eigen::VectorXd &predicted_joint_acceleration_upper) {
  const auto linearized = validate_scalar_geometric_linearized_acceptance(
      prepared.scalar, accepted_acceleration, dt);
  if (!linearized.satisfied()) {
    return as_tight_point_result(linearized);
  }

  const auto nonlinear =
      validate_nonlinear_path(prepared, robot, q, dq, accepted_acceleration, dt);
  if (!nonlinear.satisfied()) {
    return nonlinear;
  }

  Eigen::VectorXd q_next;
  try {
    q_next =
        robot.integrate(q, dt * dq + 0.5 * dt * dt * accepted_acceleration);
  } catch (const std::exception &error) {
    return failure(SolverStatus::kNumericalError,
                   "accepted tight point constraint '" +
                       prepared.specification.source_id +
                       "' predicted support integration failed: " +
                       error.what());
  }
  const Eigen::VectorXd dq_next = dq + dt * accepted_acceleration;
  FrameKinematicDifferential predicted_differential;
  try {
    predicted_differential = evaluate_frame_kinematic_differential_at_state(
        robot, prepared.specification.definition.frame_name, q_next, dq_next);
  } catch (const std::exception &error) {
    return failure(SolverStatus::kNumericalError,
                   "accepted tight point constraint '" +
                       prepared.specification.source_id +
                       "' predicted support evaluation failed: " +
                       error.what());
  }
  const auto predicted_scalar_differential = make_point_scalar_differential(
      prepared.specification, predicted_differential, dq_next,
      accepted_acceleration);
  const auto support = validate_scalar_geometric_joint_box_support(
      prepared.scalar.specification, predicted_scalar_differential,
      predicted_joint_acceleration_lower, predicted_joint_acceleration_upper,
      "predicted next-state");
  if (!support.satisfied()) {
    return failure(SolverStatus::kNumericalError, support.message);
  }
  return {};
}

} // namespace embodik::detail
