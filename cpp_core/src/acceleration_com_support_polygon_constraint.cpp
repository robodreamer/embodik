#include "acceleration_com_support_polygon_constraint.hpp"

#include "acceleration_braking_witness.hpp"
#include "acceleration_scalar_geometric_constraint.hpp"
#include "geometric_constraint_differential.hpp"

#include <algorithm>
#include <cmath>
#include <exception>
#include <limits>
#include <stdexcept>
#include <utility>

namespace embodik::detail {
namespace {

constexpr char kFamilyLabel[] = "CoM support-polygon constraint";
constexpr double kConstraintTolerance = 1e-8;

struct ProjectedComDifferential {
  Eigen::VectorXd values;
  Eigen::VectorXd rates;
  Eigen::MatrixXd acceleration_matrix;
  Eigen::VectorXd affine_bias;
};

ComSupportPolygonConstraintResult failure(SolverStatus status,
                                          std::string message) {
  return {status, std::move(message)};
}

std::string identity(const ComSupportPolygonAccelerationConstraint &constraint) {
  return std::string(kFamilyLabel) + " '" + constraint.source_id + "'";
}

bool outside_policy_is_valid(ComSupportPolygonOutsidePolicy policy) {
  switch (policy) {
  case ComSupportPolygonOutsidePolicy::kReject:
  case ComSupportPolygonOutsidePolicy::kRecoverNonWorsening:
    return true;
  }
  return false;
}

double comparison_tolerance(double lhs, double rhs) {
  return std::max(kConstraintTolerance,
                  1e-12 * (1.0 + std::max(std::abs(lhs), std::abs(rhs))));
}

ComSupportPolygonConstraintResult validate_com_differential(
    const GeometricCoordinateDifferential &differential,
    const Eigen::VectorXd &velocity, Eigen::Index variable_count,
    const std::string &context) {
  if (differential.value.size() != 3 || differential.rate.size() != 3 ||
      differential.jacobian.rows() != 3 ||
      differential.jacobian.cols() != variable_count ||
      differential.affine_bias.size() != 3 ||
      velocity.size() != variable_count) {
    return failure(SolverStatus::kShapeMismatch,
                   context + " CoM differential dimensions are inconsistent");
  }
  if (!differential.value.allFinite() || !differential.rate.allFinite() ||
      !differential.jacobian.allFinite() ||
      !differential.affine_bias.allFinite() || !velocity.allFinite()) {
    return failure(SolverStatus::kNumericalError,
                   context + " CoM differential was non-finite");
  }
  const Eigen::VectorXd expected_rate = differential.jacobian * velocity;
  if (!expected_rate.allFinite() ||
      !differential.rate.isApprox(expected_rate, 1e-10)) {
    return failure(SolverStatus::kNumericalError,
                   context + " CoM rate is inconsistent with J*dq");
  }
  return {};
}

ProjectedComDifferential project_com_differential(
    const SupportPolygonGeometry &geometry,
    const GeometricCoordinateDifferential &differential) {
  ProjectedComDifferential result;
  result.values =
      geometry.halfspace_normals * differential.value.head<2>() -
      geometry.halfspace_offsets;
  result.rates = geometry.halfspace_normals * differential.rate.head<2>();
  result.acceleration_matrix =
      geometry.halfspace_normals * differential.jacobian.topRows<2>();
  result.affine_bias =
      geometry.halfspace_normals * differential.affine_bias.head<2>();
  return result;
}

bool policy_is_valid(const ComSupportPolygonAccelerationPolicy &policy) {
  return std::isfinite(policy.rate_limit) &&
         std::isfinite(policy.acceleration_limit) &&
         std::isfinite(policy.braking_acceleration) &&
         std::isfinite(policy.boundary_epsilon) &&
         std::isfinite(policy.outside_recovery_scale) &&
         std::isfinite(policy.outside_min_recovery_speed) &&
         policy.rate_limit > 0.0 && policy.acceleration_limit > 0.0 &&
         policy.braking_acceleration > 0.0 &&
         policy.braking_acceleration <= policy.acceleration_limit &&
         policy.boundary_epsilon >= 0.0 &&
         policy.outside_recovery_scale > 0.0 &&
         policy.outside_min_recovery_speed > 0.0 &&
         policy.outside_min_recovery_speed <= policy.rate_limit &&
         outside_policy_is_valid(policy.outside_policy);
}

bool row_can_reach_proximity(double slack, double rate,
                             const ComSupportPolygonAccelerationPolicy &policy,
                             double proximity_threshold, double dt) {
  const double maximum_outward_displacement =
      std::max(0.0, rate * dt + 0.5 * policy.acceleration_limit * dt * dt);
  return slack <= proximity_threshold + maximum_outward_displacement +
                      policy.boundary_epsilon;
}

bool row_requires_braking_witness(
    double slack, double rate,
    const ComSupportPolygonAccelerationPolicy &policy,
    double proximity_threshold, double dt) {
  if (rate <= 0.0) {
    return false;
  }
  const double stopping_distance =
      rate * rate / (2.0 * policy.braking_acceleration);
  const double next_tick_reach =
      std::max(0.0, rate * dt + 0.5 * policy.acceleration_limit * dt * dt);
  const bool within_explicit_proximity =
      std::isfinite(proximity_threshold) &&
      slack <= proximity_threshold + policy.boundary_epsilon;
  return within_explicit_proximity ||
         slack <= stopping_distance + next_tick_reach +
                      policy.boundary_epsilon;
}

ComSupportPolygonConstraintResult validate_common_braking_witness(
    const PreparedComSupportPolygonConstraint &prepared,
    const Eigen::VectorXd &joint_acceleration_lower,
    const Eigen::VectorXd &joint_acceleration_upper,
    const std::string &state_label,
    const Eigen::VectorXd *candidate = nullptr) {
  std::vector<AccelerationBrakingWitnessRow> rows;
  rows.reserve(prepared.outward_rows.size());
  for (int row : prepared.outward_rows) {
    rows.push_back({row, prepared.specification.policy.braking_acceleration,
                    AccelerationBrakingDirection::kDecreaseCoordinate});
  }
  const auto witness = validate_common_acceleration_braking_witness(
      prepared.state_box.physical_constraint, joint_acceleration_lower,
      joint_acceleration_upper, rows, state_label, candidate);
  if (!witness.satisfied()) {
    return failure(witness.status,
                   identity(prepared.specification) + " " + witness.message);
  }
  return {};
}

ComSupportPolygonPreparationResult prepare_from_differential(
    const ComSupportPolygonAccelerationConstraint &constraint,
    const SupportPolygonGeometry &geometry,
    const GeometricCoordinateDifferential &differential,
    const Eigen::VectorXd &velocity, double dt,
    const Eigen::VectorXd &joint_acceleration_lower,
    const Eigen::VectorXd &joint_acceleration_upper,
    const std::string &state_label,
    const Eigen::VectorXd *witness_candidate = nullptr) {
  ComSupportPolygonPreparationResult result;
  const std::string context = identity(constraint) + " " + state_label;
  const auto differential_validation = validate_com_differential(
      differential, velocity, joint_acceleration_lower.size(), context);
  if (!differential_validation.satisfied()) {
    result.status = differential_validation.status;
    result.message = differential_validation.message;
    return result;
  }
  if (joint_acceleration_lower.size() != differential.jacobian.cols() ||
      joint_acceleration_upper.size() != differential.jacobian.cols() ||
      !joint_acceleration_lower.allFinite() ||
      !joint_acceleration_upper.allFinite() ||
      (joint_acceleration_lower.array() > joint_acceleration_upper.array())
          .any()) {
    result.status = SolverStatus::kInvalidInput;
    result.message = context + " joint acceleration box is invalid";
    return result;
  }

  const auto projected = project_com_differential(geometry, differential);
  const Eigen::Index row_count = projected.values.size();
  LinearizedStateBoxInput input;
  input.source_id = constraint.source_id;
  input.coefficient_matrix = projected.acceleration_matrix;
  input.affine_bias = projected.affine_bias;
  input.rows.reserve(static_cast<std::size_t>(row_count));
  Eigen::VectorXd recovery_targets = Eigen::VectorXd::Zero(row_count);
  std::vector<int> outward_rows;
  std::vector<int> recovery_rows;

  for (Eigen::Index row_index = 0; row_index < row_count; ++row_index) {
    const double value = projected.values(row_index);
    const double slack = -value;
    const double rate = projected.rates(row_index);
    if (std::abs(rate) >
        constraint.policy.rate_limit +
            comparison_tolerance(rate, constraint.policy.rate_limit)) {
      result.status = SolverStatus::kInfeasible;
      result.message = context + " edge " + std::to_string(row_index) +
                       " current rate exceeds its hard limit";
      return result;
    }

    StateBoxRowInput row;
    row.value = std::min(value, 0.0);
    row.rate = rate;
    row.state_bounds.upper = 0.0;
    row.state_bounds.upper_active = true;
    row.rate_bounds = {-constraint.policy.rate_limit,
                       constraint.policy.rate_limit, true, true};
    row.acceleration_bounds = {-constraint.policy.acceleration_limit,
                               constraint.policy.acceleration_limit, true,
                               true};
    row.lower_braking_acceleration =
        constraint.policy.braking_acceleration;
    row.upper_braking_acceleration =
        constraint.policy.braking_acceleration;

    const bool outside = value > constraint.policy.boundary_epsilon;
    if (outside) {
      if (constraint.policy.outside_policy ==
          ComSupportPolygonOutsidePolicy::kReject) {
        result.status = SolverStatus::kInfeasible;
        result.message = context + " edge " + std::to_string(row_index) +
                         " is outside the support polygon";
        return result;
      }
      if (rate > comparison_tolerance(rate, 0.0)) {
        result.status = SolverStatus::kInfeasible;
        result.message = context + " edge " + std::to_string(row_index) +
                         " is outside with an outward rate";
        return result;
      }
      const auto recovery = compute_reachable_recovery_rate(
          value, -rate, dt, constraint.policy.outside_recovery_scale,
          constraint.policy.outside_min_recovery_speed,
          constraint.policy.rate_limit,
          constraint.policy.acceleration_limit,
          constraint.policy.boundary_epsilon);
      if (!recovery.satisfied()) {
        result.status = recovery.status;
        result.message = context + " edge " + std::to_string(row_index) +
                         ": " + recovery.message;
        return result;
      }
      recovery_targets(row_index) = recovery.rate;
      recovery_rows.push_back(static_cast<int>(row_index));
      row.value = value;
      row.state_bounds.upper = value;
      row.rate_bounds.upper = -recovery.rate;
    } else if (row_requires_braking_witness(
                   std::max(0.0, slack), rate, constraint.policy,
                   geometry.proximity_threshold, dt)) {
      outward_rows.push_back(static_cast<int>(row_index));
    }

    if (!outside && !row_can_reach_proximity(std::max(0.0, slack), rate,
                                             constraint.policy,
                                             geometry.proximity_threshold,
                                             dt)) {
      row.state_bounds.upper_active = false;
    }
    input.rows.push_back(row);
  }

  auto state_box =
      prepare_linearized_state_box(input, dt, differential.jacobian.cols());
  if (state_box.status != SolverStatus::kSuccess) {
    result.status = state_box.status;
    result.message = context + ": " + state_box.message;
    return result;
  }

  PreparedComSupportPolygonConstraint prepared;
  prepared.specification = constraint;
  prepared.geometry = geometry;
  prepared.state_box = std::move(state_box);
  prepared.current_values = projected.values;
  prepared.current_slacks = -projected.values;
  prepared.recovery_rate_targets = std::move(recovery_targets);
  prepared.outward_rows = std::move(outward_rows);
  prepared.recovery_rows = std::move(recovery_rows);
  const auto witness =
      validate_common_braking_witness(prepared, joint_acceleration_lower,
                                      joint_acceleration_upper, state_label,
                                      witness_candidate);
  if (!witness.satisfied()) {
    result.status = witness.status;
    result.message = witness.message;
    return result;
  }
  result.prepared = std::move(prepared);
  return result;
}

ComSupportPolygonConstraintResult validate_linearized_acceptance(
    const PreparedComSupportPolygonConstraint &prepared,
    const Eigen::VectorXd &accepted_acceleration, double dt) {
  const auto &constraint = prepared.state_box.physical_constraint;
  if (constraint.coefficient_matrix.cols() != accepted_acceleration.size() ||
      !accepted_acceleration.allFinite()) {
    return failure(SolverStatus::kShapeMismatch,
                   "accepted CoM support-polygon validation dimensions are "
                   "inconsistent");
  }
  const Eigen::VectorXd acceleration =
      constraint.coefficient_matrix * accepted_acceleration +
      constraint.affine_bias;
  if (!acceleration.allFinite()) {
    return failure(SolverStatus::kNumericalError,
                   "accepted CoM support-polygon acceleration is non-finite");
  }
  for (Eigen::Index row_index = 0; row_index < acceleration.size();
       ++row_index) {
    const auto &row =
        prepared.state_box.state_rows[static_cast<std::size_t>(row_index)];
    const double a = acceleration(row_index);
    if (a < row.acceleration_bounds.lower -
                comparison_tolerance(a, row.acceleration_bounds.lower) ||
        a > row.acceleration_bounds.upper +
                comparison_tolerance(a, row.acceleration_bounds.upper)) {
      return failure(SolverStatus::kNumericalError,
                     "accepted CoM support-polygon violated acceleration "
                     "bounds");
    }
    const double next_rate = row.rate + dt * a;
    if (next_rate < row.rate_bounds.lower -
                        comparison_tolerance(next_rate,
                                             row.rate_bounds.lower) ||
        next_rate > row.rate_bounds.upper +
                        comparison_tolerance(next_rate,
                                             row.rate_bounds.upper)) {
      return failure(SolverStatus::kNumericalError,
                     "accepted CoM support-polygon violated rate bounds");
    }
    if (row.state_bounds.upper_active) {
      const double endpoint = row.value + dt * row.rate + 0.5 * dt * dt * a;
      if (endpoint > row.state_bounds.upper +
                         comparison_tolerance(endpoint,
                                              row.state_bounds.upper)) {
        return failure(SolverStatus::kNumericalError,
                       "accepted CoM support-polygon violated endpoint bounds");
      }
    }
  }
  return {};
}

ComSupportPolygonConstraintResult validate_sample(
    const PreparedComSupportPolygonConstraint &prepared,
    const GeometricCoordinateDifferential &differential,
    const Eigen::VectorXd &ddq, double dt, bool endpoint) {
  const auto projected = project_com_differential(prepared.geometry,
                                                  differential);
  const Eigen::VectorXd acceleration =
      projected.acceleration_matrix * ddq + projected.affine_bias;
  for (Eigen::Index row = 0; row < projected.values.size(); ++row) {
    const bool recovery =
        prepared.current_slacks(row) < -prepared.specification.policy
                                           .boundary_epsilon;
    const double upper =
        recovery ? prepared.current_values(row)
                 : prepared.specification.policy.boundary_epsilon;
    if (projected.values(row) >
        upper + comparison_tolerance(projected.values(row), upper)) {
      return failure(SolverStatus::kNumericalError,
                     "nonlinear CoM support-polygon path violated edge " +
                         std::to_string(row) + " state bound");
    }
    if (std::abs(projected.rates(row)) >
        prepared.specification.policy.rate_limit +
            comparison_tolerance(projected.rates(row),
                                 prepared.specification.policy.rate_limit)) {
      return failure(SolverStatus::kNumericalError,
                     "nonlinear CoM support-polygon path violated edge " +
                         std::to_string(row) + " rate bound");
    }
    if (std::abs(acceleration(row)) >
        prepared.specification.policy.acceleration_limit +
            comparison_tolerance(
                acceleration(row),
                prepared.specification.policy.acceleration_limit)) {
      auto result = failure(
          SolverStatus::kNumericalError,
          "nonlinear CoM support-polygon path violated edge " +
              std::to_string(row) + " acceleration bound");
      result.path_acceleration_violation = true;
      return result;
    }
    if (endpoint && recovery) {
      const double target = prepared.recovery_rate_targets(row);
      if (projected.rates(row) >
          -target + comparison_tolerance(projected.rates(row), -target)) {
        return failure(SolverStatus::kNumericalError,
                       "nonlinear CoM support-polygon endpoint failed edge " +
                           std::to_string(row) + " recovery-rate target");
      }
      const double required_improvement = 0.25 * dt * target;
      if (projected.values(row) >
          prepared.current_values(row) - required_improvement +
              comparison_tolerance(projected.values(row),
                                   prepared.current_values(row))) {
        return failure(
            SolverStatus::kNumericalError,
            "nonlinear CoM support-polygon endpoint failed edge " +
                std::to_string(row) + " deterministic recovery improvement");
      }
    }
    if (endpoint && !recovery && projected.rates(row) > 0.0) {
      const double slack = -projected.values(row);
      const double capacity =
          2.0 * prepared.specification.policy.braking_acceleration *
          std::max(0.0, slack);
      const double requirement = projected.rates(row) * projected.rates(row);
      if (requirement > capacity + comparison_tolerance(requirement,
                                                        capacity)) {
        return failure(SolverStatus::kNumericalError,
                       "nonlinear CoM support-polygon endpoint failed edge " +
                           std::to_string(row) + " braking viability");
      }
    }
  }
  return {};
}

ScalarGeometricConstraintSpecification com_path_specification(
    const PreparedComSupportPolygonConstraint &prepared) {
  ScalarGeometricConstraintSpecification specification;
  specification.family_label = kFamilyLabel;
  specification.source_id = prepared.specification.source_id;
  specification.dimension =
      prepared.state_box.physical_constraint.coefficient_matrix.rows();
  specification.state_lower_bounds =
      Eigen::VectorXd::Constant(specification.dimension, -1e9);
  specification.state_upper_bounds =
      Eigen::VectorXd::Constant(specification.dimension, 1e9);
  specification.active_axes.reserve(
      static_cast<std::size_t>(specification.dimension));
  for (Eigen::Index row = 0; row < specification.dimension; ++row) {
    specification.active_axes.push_back(static_cast<int>(row));
  }
  specification.policy.rate_limits = Eigen::VectorXd::Constant(
      specification.dimension, prepared.specification.policy.rate_limit);
  specification.policy.acceleration_limits = Eigen::VectorXd::Constant(
      specification.dimension, prepared.specification.policy.acceleration_limit);
  specification.policy.lower_braking_accelerations = Eigen::VectorXd::Constant(
      specification.dimension, prepared.specification.policy.braking_acceleration);
  specification.policy.upper_braking_accelerations =
      specification.policy.lower_braking_accelerations;
  return specification;
}

} // namespace

bool is_structurally_fixed_support_frame(const RobotModel &robot,
                                         const std::string &frame_name) {
  if (frame_name == "world") {
    return true;
  }
  if (frame_name.empty() || !robot.has_frame(frame_name)) {
    return false;
  }
  const auto &model = robot.model();
  const pinocchio::FrameIndex frame_id = model.getFrameId(frame_name);
  if (frame_id >= model.frames.size()) {
    return false;
  }
  pinocchio::JointIndex joint_id = model.frames[frame_id].parentJoint;
  while (joint_id != 0) {
    if (joint_id >= model.joints.size() || joint_id >= model.parents.size() ||
        model.joints[joint_id].nv() != 0) {
      return false;
    }
    joint_id = model.parents[joint_id];
  }
  return true;
}

ComSupportPolygonConstraintResult validate_com_support_polygon_constraint(
    const ComSupportPolygonAccelerationConstraint &constraint,
    const RobotModel &robot) {
  if (constraint.source_id.empty()) {
    return failure(SolverStatus::kInvalidInput,
                   std::string(kFamilyLabel) +
                       " source_id must not be empty");
  }
  if (constraint.definition.frame_name.empty() ||
      (constraint.definition.frame_name != "world" &&
       !robot.has_frame(constraint.definition.frame_name))) {
    return failure(SolverStatus::kInvalidInput,
                   identity(constraint) +
                       " frame_name must identify world or a robot frame");
  }
  if (!is_structurally_fixed_support_frame(
          robot, constraint.definition.frame_name)) {
    return failure(SolverStatus::kInvalidInput,
                   identity(constraint) +
                       " support frame must be world or structurally "
                       "root-fixed; moving frames are unsupported");
  }
  const auto geometry = prepare_support_polygon_geometry(
      constraint.definition, kFamilyLabel, constraint.source_id);
  if (!geometry.satisfied()) {
    return failure(geometry.status, geometry.message);
  }
  const auto &policy = constraint.policy;
  if (!std::isfinite(policy.rate_limit) ||
      !std::isfinite(policy.acceleration_limit) ||
      !std::isfinite(policy.braking_acceleration) ||
      !std::isfinite(policy.boundary_epsilon) ||
      !std::isfinite(policy.outside_recovery_scale) ||
      !std::isfinite(policy.outside_min_recovery_speed)) {
    return failure(SolverStatus::kNonFiniteInput,
                   identity(constraint) +
                       " policy must contain only finite values");
  }
  if (!policy_is_valid(constraint.policy)) {
    return failure(SolverStatus::kInvalidInput,
                   identity(constraint) +
                       " policy magnitudes or outside policy are invalid");
  }
  return {};
}

ComSupportPolygonPreparationResult prepare_com_support_polygon_constraint(
    const ComSupportPolygonAccelerationConstraint &constraint,
    const RobotModel &robot, double dt,
    const Eigen::VectorXd &joint_acceleration_lower,
    const Eigen::VectorXd &joint_acceleration_upper) {
  ComSupportPolygonPreparationResult result;
  const auto validation =
      validate_com_support_polygon_constraint(constraint, robot);
  if (!validation.satisfied()) {
    result.status = validation.status;
    result.message = validation.message;
    return result;
  }
  const auto geometry = prepare_support_polygon_geometry(
      constraint.definition, kFamilyLabel, constraint.source_id);
  if (!geometry.satisfied()) {
    result.status = geometry.status;
    result.message = geometry.message;
    return result;
  }
  try {
    return prepare_from_differential(
        constraint, geometry.geometry,
        evaluate_com_in_frame_differential(robot,
                                           constraint.definition.frame_name),
        robot.get_current_velocity(), dt, joint_acceleration_lower,
        joint_acceleration_upper, "current-state");
  } catch (const std::exception &error) {
    result.status = SolverStatus::kNumericalError;
    result.message = identity(constraint) +
                     " current evaluation failed: " + error.what();
    return result;
  }
}

ComSupportPolygonPreparationResult
prepare_com_support_polygon_constraint_at_state(
    const ComSupportPolygonAccelerationConstraint &constraint,
    const SupportPolygonGeometry &geometry, const RobotModel &robot,
    pinocchio::Data &scratch, const Eigen::VectorXd &q,
    const Eigen::VectorXd &dq, double dt,
    const Eigen::VectorXd &joint_acceleration_lower,
    const Eigen::VectorXd &joint_acceleration_upper,
    const Eigen::VectorXd *witness_candidate) {
  try {
    return prepare_from_differential(
        constraint, geometry,
        evaluate_com_in_frame_differential_at_state(
            robot, scratch, constraint.definition.frame_name, q, dq),
        dq, dt, joint_acceleration_lower, joint_acceleration_upper,
        "predicted next-state", witness_candidate);
  } catch (const std::exception &error) {
    ComSupportPolygonPreparationResult result;
    result.status = SolverStatus::kNumericalError;
    result.message = identity(constraint) +
                     " predicted evaluation failed: " + error.what();
    return result;
  }
}

ComSupportPolygonConstraintResult
linearize_com_support_polygon_path_acceleration(
    const PreparedComSupportPolygonConstraint &prepared,
    const RobotModel &robot, const Eigen::VectorXd &q,
    const Eigen::VectorXd &dq, const Eigen::VectorXd &candidate_acceleration,
    double dt, AffineAccelerationConstraint *refinement) {
  if (refinement == nullptr) {
    return failure(SolverStatus::kInvalidInput,
                   "CoM path refinement output is missing");
  }
  const auto specification = com_path_specification(prepared);
  ScalarGeometricConstraintResult segment_failure;
  const int segments = scalar_geometric_path_segment_count(
      dq, candidate_acceleration, dt, specification,
      ScalarGeometricPathValidationOptions{}, &segment_failure);
  if (!segment_failure.satisfied()) {
    return failure(segment_failure.status, segment_failure.message);
  }
  const auto times = scalar_geometric_path_sample_times(segments, dt);
  const int sample_segments = static_cast<int>(times.size()) - 1;
  if (sample_segments > ScalarGeometricPathValidationOptions{}.maximum_segments) {
    return failure(SolverStatus::kNumericalError,
                   "CoM path refinement exceeded validation sample budget");
  }
  const Eigen::Index edges = specification.dimension;
  Eigen::VectorXd worst = Eigen::VectorXd::Zero(edges);
  Eigen::MatrixXd matrices = Eigen::MatrixXd::Zero(edges, robot.nv());
  Eigen::VectorXd biases = Eigen::VectorXd::Zero(edges);
  Eigen::VectorXi worst_segment = Eigen::VectorXi::Constant(edges, -1);
  Eigen::VectorXd worst_values = Eigen::VectorXd::Zero(edges);
  pinocchio::Data scratch(robot.model());
  try {
    for (int segment = 0; segment <= sample_segments; ++segment) {
      const double time = times[segment];
      const Eigen::VectorXd sample_q = robot.integrate(
          q, time * dq + 0.5 * time * time * candidate_acceleration);
      const Eigen::VectorXd sample_dq = dq + time * candidate_acceleration;
      const auto differential = evaluate_com_in_frame_differential_at_state(
          robot, scratch, prepared.specification.definition.frame_name,
          sample_q, sample_dq);
      const auto valid = validate_com_differential(
          differential, sample_dq, robot.nv(), "CoM path refinement");
      if (!valid.satisfied()) {
        return valid;
      }
      const auto projected = project_com_differential(prepared.geometry,
                                                      differential);
      const Eigen::VectorXd acceleration =
          projected.acceleration_matrix * candidate_acceleration +
          projected.affine_bias;
      if (!acceleration.allFinite()) {
        return failure(SolverStatus::kNumericalError,
                       "CoM path refinement acceleration was non-finite");
      }
      for (Eigen::Index edge = 0; edge < edges; ++edge) {
        const double excess = std::abs(acceleration(edge)) -
                              prepared.specification.policy.acceleration_limit;
        if (excess > comparison_tolerance(
                         acceleration(edge),
                         prepared.specification.policy.acceleration_limit) &&
            excess > worst(edge)) {
          worst(edge) = excess;
          worst_segment(edge) = segment;
          worst_values(edge) = acceleration(edge);
        }
      }
    }
    // Differentiate the complete constant-acceleration path map at each
    // selected time. Opposite/duplicate polygon edges share the same probes.
    for (int segment = 0; segment <= sample_segments; ++segment) {
      if (!(worst_segment.array() == segment).any()) {
        continue;
      }
      const double time = times[segment];
      const auto evaluate_acceleration = [&](const Eigen::VectorXd &acceleration) {
        const Eigen::VectorXd sample_q = robot.integrate(
            q, time * dq + 0.5 * time * time * acceleration);
        const Eigen::VectorXd sample_dq = dq + time * acceleration;
        const auto differential = evaluate_com_in_frame_differential_at_state(
            robot, scratch, prepared.specification.definition.frame_name,
            sample_q, sample_dq);
        const auto valid = validate_com_differential(
            differential, sample_dq, robot.nv(), "CoM path refinement derivative");
        if (!valid.satisfied()) {
          throw std::runtime_error(valid.message);
        }
        const auto projected = project_com_differential(prepared.geometry,
                                                        differential);
        const Eigen::VectorXd value =
            projected.acceleration_matrix * acceleration + projected.affine_bias;
        if (!value.allFinite()) {
          throw std::runtime_error("CoM path derivative sample was non-finite");
        }
        return value;
      };
      Eigen::MatrixXd gradient(edges, robot.nv());
      for (Eigen::Index joint = 0; joint < robot.nv(); ++joint) {
        const double step = std::cbrt(std::numeric_limits<double>::epsilon()) *
                            std::max(1.0, std::abs(candidate_acceleration(joint)));
        Eigen::VectorXd plus = candidate_acceleration;
        Eigen::VectorXd minus = candidate_acceleration;
        plus(joint) += step;
        minus(joint) -= step;
        const double separation = plus(joint) - minus(joint);
        if (!(separation > 0.0) || !std::isfinite(separation)) {
          throw std::runtime_error("CoM path derivative perturbation is invalid");
        }
        // Perturbed states are differentiation probes, never commands. Both
        // q(t) and dq(t), including the affine acceleration bias, are refreshed.
        const Eigen::VectorXd plus_value = evaluate_acceleration(plus);
        const Eigen::VectorXd minus_value = evaluate_acceleration(minus);
        gradient.col(joint) = (plus_value - minus_value) / separation;
      }
      if (!gradient.allFinite()) {
        throw std::runtime_error("CoM path derivative was non-finite");
      }
      for (Eigen::Index edge = 0; edge < edges; ++edge) {
        if (worst_segment(edge) != segment) {
          continue;
        }
        matrices.row(edge) = gradient.row(edge);
        // Anchor the local affine row at the exact evaluated candidate value.
        biases(edge) =
            worst_values(edge) - gradient.row(edge).dot(candidate_acceleration);
        if (!std::isfinite(biases(edge))) {
          throw std::runtime_error("CoM path derivative bias was non-finite");
        }
      }
    }
  } catch (const std::exception &error) {
    return failure(SolverStatus::kNumericalError,
                   std::string("CoM path refinement failed: ") + error.what());
  }
  const Eigen::Index count = (worst.array() > 0.0).count();
  if (count == 0) {
    return failure(SolverStatus::kNumericalError,
                   "CoM path refinement found no violated acceleration row");
  }
  // Sampled path linearizations are local models, not feasibility certificates.
  // Every corrected solution must pass the unchanged full nonlinear validator.
  refinement->coefficient_matrix.resize(count, robot.nv());
  refinement->affine_bias.resize(count);
  refinement->lower_bounds = Eigen::VectorXd::Constant(
      count, -prepared.specification.policy.acceleration_limit);
  refinement->upper_bounds = -refinement->lower_bounds;
  Eigen::Index row = 0;
  for (Eigen::Index edge = 0; edge < edges; ++edge) {
    if (worst(edge) > 0.0) {
      refinement->coefficient_matrix.row(row) = matrices.row(edge);
      refinement->affine_bias(row++) = biases(edge);
    }
  }
  return {};
}

ComSupportPolygonConstraintResult
validate_com_support_polygon_constraint_acceptance(
    const PreparedComSupportPolygonConstraint &prepared,
    const RobotModel &robot, const Eigen::VectorXd &q,
    const Eigen::VectorXd &dq, const Eigen::VectorXd &accepted_acceleration,
    double dt, const Eigen::VectorXd &next_joint_acceleration_lower,
    const Eigen::VectorXd &next_joint_acceleration_upper) {
  const auto linearized =
      validate_linearized_acceptance(prepared, accepted_acceleration, dt);
  if (!linearized.satisfied()) {
    return linearized;
  }

  pinocchio::Data scratch(robot.model());
  const auto path_spec = com_path_specification(prepared);
  bool path_acceleration_violation = false;
  const ScalarGeometricPathValidationOptions path_options;
  const auto path = validate_scalar_geometric_path(
      path_spec, dq, accepted_acceleration, dt, path_options,
      [&](const ScalarGeometricPathPoint &path_point,
          ScalarGeometricCoordinateSample *coordinates) {
        Eigen::VectorXd sample_q;
        try {
          sample_q = robot.integrate(q, path_point.tangent);
          const auto differential = evaluate_com_in_frame_differential_at_state(
              robot, scratch, prepared.specification.definition.frame_name,
              sample_q, path_point.rate);
          const auto projected = project_com_differential(prepared.geometry,
                                                          differential);
          coordinates->value = projected.values;
          coordinates->rate = projected.rates;
          coordinates->acceleration =
              projected.acceleration_matrix * accepted_acceleration +
              projected.affine_bias;
          const auto sample = validate_sample(
              prepared, differential, accepted_acceleration,
              dt,
              path_point.segment_index == path_point.segment_count);
          path_acceleration_violation = sample.path_acceleration_violation;
          return ScalarGeometricConstraintResult{sample.status,
                                                 sample.message};
        } catch (const std::exception &error) {
          return ScalarGeometricConstraintResult{
              SolverStatus::kNumericalError,
              std::string("accepted CoM support-polygon path failed: ") +
                  error.what()};
        }
      });
  if (!path.satisfied()) {
    auto result = failure(path.status, path.message);
    result.path_acceleration_violation = path_acceleration_violation;
    return result;
  }

  Eigen::VectorXd q_next;
  try {
    q_next =
        robot.integrate(q, dt * dq + 0.5 * dt * dt * accepted_acceleration);
  } catch (const std::exception &error) {
    return failure(SolverStatus::kNumericalError,
                   "accepted CoM support-polygon predicted integration "
                   "failed: " +
                       std::string(error.what()));
  }
  const Eigen::VectorXd dq_next = dq + dt * accepted_acceleration;
  const auto predicted = prepare_com_support_polygon_constraint_at_state(
      prepared.specification, prepared.geometry, robot, scratch, q_next,
      dq_next, dt, next_joint_acceleration_lower,
      next_joint_acceleration_upper, &accepted_acceleration);
  if (!predicted.satisfied()) {
    return failure(SolverStatus::kNumericalError,
                   "accepted CoM support-polygon predicted support failed: " +
                       predicted.message);
  }
  return {};
}

} // namespace embodik::detail
