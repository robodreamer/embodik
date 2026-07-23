#include "acceleration_tight_point_constraint.hpp"

#include "acceleration_geometric_constraint_policy.hpp"
#include "frame_kinematic_differential.hpp"

#include <algorithm>
#include <cmath>
#include <exception>
#include <limits>
#include <pinocchio/multibody/data.hpp>
#include <string>
#include <utility>

namespace embodik::detail {
namespace {

constexpr double kConstraintTolerance = 1e-8;

double comparison_tolerance(double lhs, double rhs) {
  return std::max(kConstraintTolerance,
                  1e-12 * (1.0 + std::max(std::abs(lhs), std::abs(rhs))));
}

TightPointConstraintResult failure(SolverStatus status, std::string message) {
  TightPointConstraintResult result;
  result.status = status;
  result.message = std::move(message);
  return result;
}

Eigen::Vector3d coordinate_value(
    const FrameKinematicDifferential &differential,
    const TightPointConstraintDefinition &definition) {
  return differential.pose.translation() - definition.target_point;
}

Eigen::Vector3d coordinate_rate(
    const FrameKinematicDifferential &differential,
    const Eigen::VectorXd &dq) {
  return differential.jacobian.topRows<3>() * dq;
}

TightPointConstraintResult validate_joint_box_support(
    const TightPointAccelerationConstraint &constraint,
    const FrameKinematicDifferential &differential,
    const std::vector<int> &active_axes,
    const Eigen::VectorXd &joint_acceleration_lower,
    const Eigen::VectorXd &joint_acceleration_upper,
    const std::string &state_label) {
  if (differential.jacobian.cols() != joint_acceleration_lower.size() ||
      joint_acceleration_lower.size() != joint_acceleration_upper.size()) {
    return failure(SolverStatus::kShapeMismatch,
                   "tight point constraint '" + constraint.source_id +
                       "' support-check dimensions are inconsistent");
  }
  if (!joint_acceleration_lower.allFinite() ||
      !joint_acceleration_upper.allFinite() ||
      (joint_acceleration_lower.array() > joint_acceleration_upper.array())
          .any()) {
    return failure(SolverStatus::kInvalidInput,
                   "tight point constraint '" + constraint.source_id +
                       "' " + state_label +
                       " joint acceleration support box is invalid");
  }

  for (int axis : active_axes) {
    double minimum = differential.affine_bias(axis);
    double maximum = differential.affine_bias(axis);
    const Eigen::VectorXd row = differential.jacobian.row(axis);
    for (Eigen::Index index = 0; index < row.size(); ++index) {
      const double coefficient = row(index);
      if (coefficient >= 0.0) {
        minimum += coefficient * joint_acceleration_lower(index);
        maximum += coefficient * joint_acceleration_upper(index);
      } else {
        minimum += coefficient * joint_acceleration_upper(index);
        maximum += coefficient * joint_acceleration_lower(index);
      }
    }
    if (!std::isfinite(minimum) || !std::isfinite(maximum)) {
      return failure(SolverStatus::kNumericalError,
                     "tight point constraint '" + constraint.source_id +
                         "' " + state_label +
                         " joint-box support was non-finite");
    }

    const double lower_braking =
        constraint.policy.lower_braking_accelerations(axis);
    const double upper_braking =
        constraint.policy.upper_braking_accelerations(axis);
    if (maximum < lower_braking -
                      comparison_tolerance(maximum, lower_braking)) {
      return failure(SolverStatus::kInfeasible,
                     "tight point constraint '" + constraint.source_id +
                         "' " + state_label + " axis " +
                         std::to_string(axis) +
                         " lacks lower-side braking support");
    }
    if (minimum > -upper_braking +
                      comparison_tolerance(minimum, -upper_braking)) {
      return failure(SolverStatus::kInfeasible,
                     "tight point constraint '" + constraint.source_id +
                         "' " + state_label + " axis " +
                         std::to_string(axis) +
                         " lacks upper-side braking support");
    }
  }

  return {};
}

TightPointConstraintResult validate_linearized_acceptance(
    const PreparedTightPointConstraint &prepared,
    const Eigen::VectorXd &accepted_acceleration, double dt) {
  const auto &constraint = prepared.specification;
  const auto &physical_constraint = prepared.state_box.physical_constraint;
  const Eigen::VectorXd physical =
      physical_constraint.coefficient_matrix * accepted_acceleration +
      physical_constraint.affine_bias;
  if (!physical.allFinite()) {
    return failure(SolverStatus::kNumericalError,
                   "accepted tight point constraint '" +
                       constraint.source_id +
                       "' produced non-finite physical acceleration");
  }
  if (physical.size() !=
      static_cast<Eigen::Index>(prepared.state_box.state_rows.size())) {
    return failure(SolverStatus::kShapeMismatch,
                   "accepted tight point constraint '" +
                       constraint.source_id +
                       "' validation dimensions are inconsistent");
  }

  for (Eigen::Index row_index = 0; row_index < physical.size(); ++row_index) {
    const auto &row =
        prepared.state_box.state_rows[static_cast<std::size_t>(row_index)];
    const double acceleration = physical(row_index);
    if (acceleration <
        row.acceleration_bounds.lower -
            comparison_tolerance(acceleration, row.acceleration_bounds.lower)) {
      return failure(SolverStatus::kNumericalError,
                     "accepted tight point constraint '" +
                         constraint.source_id +
                         "' violated linearized acceleration lower bound");
    }
    if (acceleration >
        row.acceleration_bounds.upper +
            comparison_tolerance(acceleration, row.acceleration_bounds.upper)) {
      return failure(SolverStatus::kNumericalError,
                     "accepted tight point constraint '" +
                         constraint.source_id +
                         "' violated linearized acceleration upper bound");
    }

    const double next_rate = row.rate + dt * acceleration;
    if (next_rate <
        row.rate_bounds.lower -
            comparison_tolerance(next_rate, row.rate_bounds.lower)) {
      return failure(SolverStatus::kNumericalError,
                     "accepted tight point constraint '" +
                         constraint.source_id +
                         "' violated linearized rate lower bound");
    }
    if (next_rate >
        row.rate_bounds.upper +
            comparison_tolerance(next_rate, row.rate_bounds.upper)) {
      return failure(SolverStatus::kNumericalError,
                     "accepted tight point constraint '" +
                         constraint.source_id +
                         "' violated linearized rate upper bound");
    }

    const double endpoint =
        row.value + dt * row.rate + 0.5 * dt * dt * acceleration;
    if (endpoint <
        row.state_bounds.lower -
            comparison_tolerance(endpoint, row.state_bounds.lower)) {
      return failure(SolverStatus::kNumericalError,
                     "accepted tight point constraint '" +
                         constraint.source_id +
                         "' violated linearized state lower bound");
    }
    if (endpoint >
        row.state_bounds.upper +
            comparison_tolerance(endpoint, row.state_bounds.upper)) {
      return failure(SolverStatus::kNumericalError,
                     "accepted tight point constraint '" +
                         constraint.source_id +
                         "' violated linearized state upper bound");
    }
  }
  return {};
}

TightPointConstraintResult validate_sample(
    const TightPointAccelerationConstraint &constraint,
    const std::vector<int> &active_axes,
    const FrameKinematicDifferential &differential,
    const Eigen::VectorXd &dq, const Eigen::VectorXd &ddq,
    bool endpoint) {
  const Eigen::Vector3d state = coordinate_value(differential, constraint.definition);
  const Eigen::Vector3d rate = coordinate_rate(differential, dq);
  const Eigen::Vector3d acceleration =
      differential.jacobian.topRows<3>() * ddq +
      differential.affine_bias.head<3>();
  if (!state.allFinite() || !rate.allFinite() || !acceleration.allFinite()) {
    return failure(SolverStatus::kNumericalError,
                   "accepted tight point constraint '" +
                       constraint.source_id +
                       "' nonlinear sample was non-finite");
  }

  for (int axis : active_axes) {
    const double epsilon = constraint.definition.position_epsilon;
    if (state(axis) < -epsilon - comparison_tolerance(state(axis), epsilon) ||
        state(axis) > epsilon + comparison_tolerance(state(axis), epsilon)) {
      return failure(SolverStatus::kNumericalError,
                     "accepted tight point constraint '" +
                         constraint.source_id +
                         "' nonlinear path violated axis " +
                         std::to_string(axis) + " state bound");
    }
    const double rate_limit = constraint.policy.rate_limits(axis);
    if (std::abs(rate(axis)) >
        rate_limit + comparison_tolerance(rate(axis), rate_limit)) {
      return failure(SolverStatus::kNumericalError,
                     "accepted tight point constraint '" +
                         constraint.source_id +
                         "' nonlinear path violated axis " +
                         std::to_string(axis) + " rate bound");
    }
    const double acceleration_limit =
        constraint.policy.acceleration_limits(axis);
    if (std::abs(acceleration(axis)) >
        acceleration_limit +
            comparison_tolerance(acceleration(axis), acceleration_limit)) {
      return failure(SolverStatus::kNumericalError,
                     "accepted tight point constraint '" +
                         constraint.source_id +
                         "' nonlinear path violated axis " +
                         std::to_string(axis) + " acceleration bound");
    }
    if (!endpoint) {
      continue;
    }
    if (rate(axis) < 0.0) {
      const double margin = state(axis) + epsilon;
      const double capacity =
          2.0 * constraint.policy.lower_braking_accelerations(axis) * margin;
      if (margin < -comparison_tolerance(margin, 0.0) ||
          rate(axis) * rate(axis) >
              capacity + comparison_tolerance(rate(axis) * rate(axis),
                                              capacity)) {
        return failure(SolverStatus::kNumericalError,
                       "accepted tight point constraint '" +
                           constraint.source_id + "' endpoint failed axis " +
                           std::to_string(axis) +
                           " lower braking viability");
      }
    }
    if (rate(axis) > 0.0) {
      const double margin = epsilon - state(axis);
      const double capacity =
          2.0 * constraint.policy.upper_braking_accelerations(axis) * margin;
      if (margin < -comparison_tolerance(margin, 0.0) ||
          rate(axis) * rate(axis) >
              capacity + comparison_tolerance(rate(axis) * rate(axis),
                                              capacity)) {
        return failure(SolverStatus::kNumericalError,
                       "accepted tight point constraint '" +
                           constraint.source_id + "' endpoint failed axis " +
                           std::to_string(axis) +
                           " upper braking viability");
      }
    }
  }
  return {};
}

TightPointConstraintResult validate_nonlinear_path(
    const PreparedTightPointConstraint &prepared, const RobotModel &robot,
    const Eigen::VectorXd &q, const Eigen::VectorXd &dq,
    const Eigen::VectorXd &ddq, double dt) {
  constexpr double kMaximumJointTangentStep = 2e-3;
  constexpr int kMinimumSegments = 16;
  constexpr int kMaximumSegments = 512;

  const Eigen::VectorXd next_velocity = dq + dt * ddq;
  if (!next_velocity.allFinite()) {
    return failure(SolverStatus::kNumericalError,
                   "accepted tight point constraint '" +
                       prepared.specification.source_id +
                       "' nonlinear path velocity was non-finite");
  }
  const double maximum_speed =
      dq.cwiseAbs().cwiseMax(next_velocity.cwiseAbs()).maxCoeff();
  const double required_segments_value =
      std::ceil(dt * maximum_speed / kMaximumJointTangentStep);
  if (!std::isfinite(required_segments_value) ||
      required_segments_value > kMaximumSegments) {
    return failure(SolverStatus::kNumericalError,
                   "accepted tight point constraint '" +
                       prepared.specification.source_id +
                       "' nonlinear path exceeded validation trust step");
  }
  const int segment_count =
      std::max(kMinimumSegments, static_cast<int>(required_segments_value));

  pinocchio::Data scratch(robot.model());
  for (int segment = 0; segment <= segment_count; ++segment) {
    const double time = dt * static_cast<double>(segment) /
                        static_cast<double>(segment_count);
    const Eigen::VectorXd tangent = time * dq + 0.5 * time * time * ddq;
    Eigen::VectorXd sample_q;
    try {
      sample_q = robot.integrate(q, tangent);
    } catch (const std::exception &error) {
      return failure(SolverStatus::kNumericalError,
                     "accepted tight point constraint '" +
                         prepared.specification.source_id +
                         "' nonlinear path integration failed: " +
                         error.what());
    }
    const Eigen::VectorXd sample_dq = dq + time * ddq;

    FrameKinematicDifferential differential;
    try {
      differential = evaluate_frame_kinematic_differential_at_state(
          robot, scratch, prepared.specification.definition.frame_name,
          sample_q, sample_dq);
    } catch (const std::exception &error) {
      return failure(SolverStatus::kNumericalError,
                     "accepted tight point constraint '" +
                         prepared.specification.source_id +
                         "' nonlinear path evaluation failed: " +
                         error.what());
    }
    const auto sample = validate_sample(
        prepared.specification, prepared.active_axes, differential, sample_dq,
        ddq, segment == segment_count);
    if (!sample.satisfied()) {
      return sample;
    }
  }
  return {};
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

  const Eigen::Vector3d value =
      coordinate_value(differential, constraint.definition);
  const Eigen::Vector3d rate =
      coordinate_rate(differential, robot.get_current_velocity());
  if (!value.allFinite() || !rate.allFinite()) {
    result.status = SolverStatus::kNumericalError;
    result.message = "tight point constraint '" + constraint.source_id +
                     "' state or rate was non-finite";
    return result;
  }

  LinearizedStateBoxInput input;
  input.source_id = constraint.source_id;
  input.coefficient_matrix.resize(
      static_cast<Eigen::Index>(active_axes.size()), robot.nv());
  input.affine_bias.resize(static_cast<Eigen::Index>(active_axes.size()));
  input.rows.reserve(active_axes.size());
  for (std::size_t row_index = 0; row_index < active_axes.size();
       ++row_index) {
    const int axis = active_axes[row_index];
    input.coefficient_matrix.row(static_cast<Eigen::Index>(row_index)) =
        differential.jacobian.row(axis);
    input.affine_bias(static_cast<Eigen::Index>(row_index)) =
        differential.affine_bias(axis);

    StateBoxRowInput row;
    row.value = value(axis);
    row.rate = rate(axis);
    row.state_bounds = {-constraint.definition.position_epsilon,
                        constraint.definition.position_epsilon, true, true};
    row.rate_bounds = {-constraint.policy.rate_limits(axis),
                       constraint.policy.rate_limits(axis), true, true};
    row.acceleration_bounds = {-constraint.policy.acceleration_limits(axis),
                               constraint.policy.acceleration_limits(axis),
                               true, true};
    row.lower_braking_acceleration =
        constraint.policy.lower_braking_accelerations(axis);
    row.upper_braking_acceleration =
        constraint.policy.upper_braking_accelerations(axis);
    input.rows.push_back(row);
  }

  auto state_box = prepare_linearized_state_box(input, dt, robot.nv());
  if (state_box.status != SolverStatus::kSuccess) {
    result.status = state_box.status;
    result.message = "tight point constraint '" + constraint.source_id +
                     "': " + state_box.message;
    return result;
  }

  const auto support = validate_joint_box_support(
      constraint, differential, active_axes, joint_acceleration_lower,
      joint_acceleration_upper, "current-state");
  if (!support.satisfied()) {
    result.status = support.status;
    result.message = support.message;
    return result;
  }

  PreparedTightPointConstraint prepared;
  prepared.specification = constraint;
  prepared.active_axes = std::move(active_axes);
  prepared.state_box = std::move(state_box);
  result.prepared = std::move(prepared);
  return result;
}

TightPointConstraintResult validate_tight_point_constraint_acceptance(
    const PreparedTightPointConstraint &prepared, const RobotModel &robot,
    const Eigen::VectorXd &q, const Eigen::VectorXd &dq,
    const Eigen::VectorXd &accepted_acceleration, double dt,
    const Eigen::VectorXd &predicted_joint_acceleration_lower,
    const Eigen::VectorXd &predicted_joint_acceleration_upper) {
  const auto linearized =
      validate_linearized_acceptance(prepared, accepted_acceleration, dt);
  if (!linearized.satisfied()) {
    return linearized;
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
  const auto support = validate_joint_box_support(
      prepared.specification, predicted_differential, prepared.active_axes,
      predicted_joint_acceleration_lower, predicted_joint_acceleration_upper,
      "predicted next-state");
  if (!support.satisfied()) {
    return failure(SolverStatus::kNumericalError, support.message);
  }
  return {};
}

} // namespace embodik::detail
