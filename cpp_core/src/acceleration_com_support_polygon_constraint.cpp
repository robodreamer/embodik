#include "acceleration_com_support_polygon_constraint.hpp"

#include "acceleration_braking_witness.hpp"
#include "acceleration_scalar_geometric_constraint.hpp"
#include "geometric_constraint_differential.hpp"

#include <algorithm>
#include <cmath>
#include <exception>
#include <limits>
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
      return failure(SolverStatus::kNumericalError,
                     "nonlinear CoM support-polygon path violated edge " +
                         std::to_string(row) + " acceleration bound");
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
  ScalarGeometricConstraintSpecification path_spec;
  path_spec.family_label = kFamilyLabel;
  path_spec.source_id = prepared.specification.source_id;
  path_spec.dimension = prepared.state_box.physical_constraint
                            .coefficient_matrix.rows();
  path_spec.state_lower_bounds =
      Eigen::VectorXd::Constant(path_spec.dimension, -1e9);
  path_spec.state_upper_bounds =
      Eigen::VectorXd::Constant(path_spec.dimension, 1e9);
  path_spec.active_axes.reserve(static_cast<std::size_t>(path_spec.dimension));
  for (Eigen::Index row = 0; row < path_spec.dimension; ++row) {
    path_spec.active_axes.push_back(static_cast<int>(row));
  }
  path_spec.policy.rate_limits =
      Eigen::VectorXd::Constant(path_spec.dimension,
                                prepared.specification.policy.rate_limit);
  path_spec.policy.acceleration_limits =
      Eigen::VectorXd::Constant(
          path_spec.dimension,
          prepared.specification.policy.acceleration_limit);
  path_spec.policy.lower_braking_accelerations =
      Eigen::VectorXd::Constant(
          path_spec.dimension,
          prepared.specification.policy.braking_acceleration);
  path_spec.policy.upper_braking_accelerations =
      Eigen::VectorXd::Constant(
          path_spec.dimension,
          prepared.specification.policy.braking_acceleration);

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
    return failure(path.status, path.message);
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
