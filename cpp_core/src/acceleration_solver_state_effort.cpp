#include "acceleration_solver_internal.hpp"

namespace embodik {
namespace acceleration_solver_internal {

AccelerationSolverResult failure(SolverStatus status, std::string message) {
  AccelerationSolverResult result;
  result.status = status;
  result.status_message = std::move(message);
  return result;
}

bool finite_positive_vector(const Eigen::VectorXd &values) {
  return values.size() > 0 && values.allFinite() &&
         (values.array() > 0.0).all();
}

bool fixed_base_scalar_joints_only(const RobotModel &robot) {
  if (robot.is_floating_base() || robot.nq() != robot.nv()) {
    return false;
  }
  for (const auto &joint_name : robot.get_joint_names()) {
    if (robot.get_joint_config_size(joint_name) != 1 ||
        robot.get_joint_velocity_size(joint_name) != 1) {
      return false;
    }
  }
  return true;
}

bool native_collision_options_are_supported(
    const AccelerationSolveOptions &options, std::string *message) {
  const bool unsupported =
      options.effort_constraints.has_value() ||
      !options.affine_constraints.empty() ||
      !options.frozen_next_velocity_constraints.empty() ||
      !options.task_acceleration_bounds.empty() ||
      !options.centroidal_momentum_rate_bounds.empty() ||
      !options.contact_acceleration_constraints.empty() ||
      !options.tight_point_constraints.empty() ||
      !options.tight_frame_pose_constraints.empty() ||
      !options.relative_pose_constraints.empty() ||
      !options.torso_pose_bound_constraints.empty() ||
      !options.com_support_polygon_constraints.empty() ||
      !options.capture_point_constraints.empty() ||
      !options.zmp_constraints.empty() ||
      !options.zero_acceleration_joint_indices.empty() ||
      !options.zero_next_velocity_joint_indices.empty() ||
      !options.fixed_current_position_joint_indices.empty();
  if (unsupported && message != nullptr) {
    *message =
        "native collision certification currently permits joint state boxes, "
        "generalized allocation, and soft tasks only; another hard family "
        "lacks a predicted-state certificate provider";
  }
  return !unsupported;
}

detail::CompiledAnalyticCollisionConstraint make_stored_collision_compilation(
    const std::vector<std::size_t> &pair_indices,
    const std::vector<double> &minimum_distances,
    const std::vector<CollisionGeometryPair> &active_pairs) {
  detail::CompiledAnalyticCollisionConstraint compiled;
  compiled.pair_indices = pair_indices;
  compiled.minimum_distances = minimum_distances;
  compiled.active_pairs = active_pairs;
  return compiled;
}

bool inactive_bound_magnitude(const Eigen::Ref<const Eigen::RowVectorXd> &row,
                              double *magnitude) {
  const double row_norm = row.stableNorm();
  if (!std::isfinite(row_norm) ||
      (row_norm > 0.0 &&
       row_norm > std::numeric_limits<double>::max() /
                      kInactiveAffineBackendBound)) {
    return false;
  }
  *magnitude =
      row_norm == 0.0 ? kInactiveAffineBackendBound
                      : kInactiveAffineBackendBound * row_norm;
  return std::isfinite(*magnitude);
}

bool checked_product(double lhs, double rhs, double *out) {
  const double product = lhs * rhs;
  if (!std::isfinite(product)) {
    return false;
  }
  if (lhs != 0.0 && rhs != 0.0 && product == 0.0) {
    return false;
  }
  *out = product;
  return true;
}

bool checked_quotient(double numerator, double denominator, double *out) {
  if (numerator == 0.0) {
    *out = 0.0;
    return true;
  }
  const double quotient = numerator / denominator;
  if (!std::isfinite(quotient) || quotient == 0.0) {
    return false;
  }
  *out = quotient;
  return true;
}

bool checked_add(double lhs, double rhs, double *out) {
  const double sum = lhs + rhs;
  if (!std::isfinite(sum)) {
    return false;
  }
  *out = sum;
  return true;
}

bool checked_scaled_matrix(const Eigen::MatrixXd &matrix, double scale,
                           Eigen::MatrixXd *scaled) {
  scaled->resize(matrix.rows(), matrix.cols());
  for (Eigen::Index row = 0; row < matrix.rows(); ++row) {
    for (Eigen::Index col = 0; col < matrix.cols(); ++col) {
      if (!checked_product(scale, matrix(row, col), &(*scaled)(row, col))) {
        return false;
      }
    }
  }
  return true;
}

bool checked_dot_row(const Eigen::Ref<const Eigen::RowVectorXd> &row,
                     const Eigen::VectorXd &vector, double *out) {
  double sum = 0.0;
  for (Eigen::Index col = 0; col < row.cols(); ++col) {
    double product = 0.0;
    if (!checked_product(row(col), vector(col), &product) ||
        !checked_add(sum, product, &sum)) {
      return false;
    }
  }
  *out = sum;
  return true;
}

bool checked_shifted_bound(double bound, double bias) {
  double shifted = 0.0;
  return checked_add(bound, -bias, &shifted);
}

VelocitySolverConfig acceleration_backend_config() {
  VelocitySolverConfig config;
  config.epsilon = 1e-10;
  config.precision_threshold = 1e-12;
  config.iteration_limit = 50;
  config.magnitude_limit = 1e10;
  config.stall_detection_count = 2;
  config.regularization_config.epsilon = 1e-10;
  config.regularization_config.regularization_factor = 0.0;
  return config;
}

Eigen::VectorXd resolve_acceleration_limits(
    const RobotModel &robot, const AccelerationSolveOptions &options,
    SolverStatus *status, std::string *message) {
  if (options.acceleration_limits_override.has_value()) {
    return options.acceleration_limits_override.value();
  }
  if (!robot.has_custom_acceleration_limits()) {
    *status = SolverStatus::kInvalidInput;
    *message =
        "AccelerationSolver requires explicit acceleration limits; pass an "
        "override or call RobotModel::set_acceleration_limits()";
    return {};
  }
  return robot.get_acceleration_limits();
}

StateBoxAssembly build_joint_state_box(const RobotModel &robot,
                                       const Eigen::VectorXd &q,
                                       const Eigen::VectorXd &dq, double dt,
                                       const Eigen::VectorXd &accel_limits,
                                       const AccelerationSolveOptions &options) {
  StateBoxAssembly assembly;
  const Eigen::Index nv = robot.nv();
  assembly.lower.resize(nv);
  assembly.upper.resize(nv);
  assembly.lower_causes.resize(static_cast<std::size_t>(nv), 0U);
  assembly.upper_causes.resize(static_cast<std::size_t>(nv), 0U);

  Eigen::VectorXd velocity_limits;
  if (options.apply_velocity_limits) {
    velocity_limits = robot.get_velocity_limits();
    if (velocity_limits.size() != nv ||
        !finite_positive_vector(velocity_limits)) {
      assembly.status = SolverStatus::kInvalidInput;
      assembly.message =
          "velocity limits must have size nv and be finite and positive";
      return assembly;
    }
  }

  Eigen::VectorXd lower_positions;
  Eigen::VectorXd upper_positions;
  if (options.apply_position_limits) {
    const auto joint_limits = robot.get_joint_limits();
    lower_positions = joint_limits.first;
    upper_positions = joint_limits.second;
    if (lower_positions.size() != q.size() || upper_positions.size() != q.size() ||
        !lower_positions.allFinite() || !upper_positions.allFinite() ||
        (lower_positions.array() > upper_positions.array()).any()) {
      assembly.status = SolverStatus::kInvalidInput;
      assembly.message =
          "position limits must be finite, ordered, and have size nq";
      return assembly;
    }
  }

  for (Eigen::Index index = 0; index < nv; ++index) {
    detail::StateBoxRowInput row;
    row.value = q(index);
    row.rate = dq(index);
    row.acceleration_bounds = {-accel_limits(index), accel_limits(index), true,
                               true};
    if (options.apply_velocity_limits) {
      row.rate_bounds = {-velocity_limits(index), velocity_limits(index), true,
                         true};
    }
    if (options.apply_position_limits) {
      row.state_bounds = {lower_positions(index), upper_positions(index), true,
                          true};
      row.lower_braking_acceleration = accel_limits(index);
      row.upper_braking_acceleration = accel_limits(index);
    }

    const auto shaped = detail::shape_state_box_row(row, dt);
    if (shaped.status != SolverStatus::kSuccess) {
      assembly.status = shaped.status;
      assembly.message = shaped.message;
      return assembly;
    }
    assembly.lower(index) = shaped.lower_acceleration;
    assembly.upper(index) = shaped.upper_acceleration;
    assembly.lower_causes[static_cast<std::size_t>(index)] =
        shaped.lower_causes;
    assembly.upper_causes[static_cast<std::size_t>(index)] =
        shaped.upper_causes;
  }
  return assembly;
}

Eigen::VectorXd vector_from_solution(const SolverResult &result) {
  if (result.solution.empty()) {
    return {};
  }
  return Eigen::Map<const Eigen::VectorXd>(result.solution.data(),
                                           result.solution.size());
}

void assign_solution_from_vector(SolverResult *result,
                                 const Eigen::VectorXd &solution) {
  result->solution.assign(solution.data(), solution.data() + solution.size());
}

bool zero_excluded_by_bounds(const Eigen::VectorXd &lower,
                             const Eigen::VectorXd &upper) {
  return (lower.array() > kConstraintTolerance).any() ||
         (upper.array() < -kConstraintTolerance).any();
}

bool populate_allocation_diagnostics(
    AccelerationSolverResult *result,
    const detail::PreparedAccelerationAllocationTransform &allocation,
    const Eigen::VectorXd &physical_acceleration) {
  AccelerationAllocationDiagnostics diagnostics;
  diagnostics.applied = true;
  diagnostics.physical_metric_diagonal =
      allocation.requested_metric_diagonal;
  diagnostics.reference_acceleration =
      allocation.reference_acceleration;
  const Eigen::VectorXd residual =
      physical_acceleration - allocation.reference_acceleration;
  diagnostics.weighted_physical_residual =
      allocation.requested_metric_diagonal.array().sqrt().matrix()
          .cwiseProduct(residual);
  diagnostics.objective_value =
      0.5 * diagnostics.weighted_physical_residual.squaredNorm();
  if (!diagnostics.weighted_physical_residual.allFinite() ||
      !std::isfinite(diagnostics.objective_value)) {
    return false;
  }
  result->allocation_diagnostics = std::move(diagnostics);
  return true;
}

PreparedEffortConstraint prepare_effort_constraint(
    const RobotModel &robot, const Eigen::VectorXd &q,
    const Eigen::VectorXd &dq, const EffortConstraintOptions &options) {
  PreparedEffortConstraint prepared;
  const auto fail = [&](SolverStatus status, std::string message) {
    prepared = {};
    prepared.status = status;
    prepared.message = std::move(message);
    return prepared;
  };

  if (!std::isfinite(options.margin_fraction)) {
    return fail(SolverStatus::kNonFiniteInput,
                "effort margin fraction must be finite");
  }
  if (options.margin_fraction < 0.0 || options.margin_fraction >= 1.0) {
    return fail(SolverStatus::kInvalidInput,
                "effort margin fraction must be in [0, 1)");
  }

  const Eigen::Index nv = robot.nv();
  Eigen::VectorXd raw_limits;
  if (options.limits_override.has_value()) {
    raw_limits = options.limits_override.value();
    if (raw_limits.size() != nv) {
      return fail(SolverStatus::kShapeMismatch,
                  "effort limits override must have size nv");
    }
    if (!raw_limits.allFinite()) {
      return fail(SolverStatus::kNonFiniteInput,
                  "effort limits override must be finite");
    }
  } else {
    raw_limits = robot.get_effort_limits();
    if (raw_limits.size() != nv) {
      return fail(SolverStatus::kShapeMismatch,
                  "model effort limits must have size nv");
    }
    if (!raw_limits.allFinite()) {
      return fail(SolverStatus::kInvalidInput,
                  "model effort limits must be finite and positive");
    }
  }
  if ((raw_limits.array() <= 0.0).any()) {
    return fail(SolverStatus::kInvalidInput,
                "effort limits must be strictly positive");
  }

  prepared.limits = (1.0 - options.margin_fraction) * raw_limits;
  if (!finite_positive_vector(prepared.limits)) {
    return fail(SolverStatus::kInvalidInput,
                "effort margin must leave finite positive limits");
  }

  try {
    prepared.mass_matrix = robot.compute_mass_matrix(q);
    prepared.mass_matrix =
        0.5 * (prepared.mass_matrix + prepared.mass_matrix.transpose());
    prepared.bias = robot.rnea(q, dq, Eigen::VectorXd::Zero(nv));
  } catch (const std::exception &error) {
    return fail(SolverStatus::kNumericalError,
                std::string("effort dynamics evaluation failed: ") +
                    error.what());
  }

  if (prepared.mass_matrix.rows() != nv || prepared.mass_matrix.cols() != nv ||
      prepared.bias.size() != nv) {
    return fail(SolverStatus::kShapeMismatch,
                "effort dynamics returned inconsistent dimensions");
  }
  if (!prepared.mass_matrix.allFinite() || !prepared.bias.allFinite()) {
    return fail(SolverStatus::kNumericalError,
                "effort dynamics returned non-finite values");
  }
  if (!(-prepared.limits - prepared.bias).allFinite() ||
      !(prepared.limits - prepared.bias).allFinite()) {
    return fail(SolverStatus::kNumericalError,
                "shifted effort bounds are not finite");
  }
  return prepared;
}

InverseDynamicsEvaluation evaluate_inverse_dynamics(
    const RobotModel &robot, const Eigen::VectorXd &q,
    const Eigen::VectorXd &dq, const Eigen::VectorXd &ddq) {
  InverseDynamicsEvaluation evaluation;
  if (ddq.size() != robot.nv()) {
    evaluation.status = SolverStatus::kShapeMismatch;
    evaluation.message = "accepted acceleration must have size nv";
    return evaluation;
  }
  if (!ddq.allFinite()) {
    evaluation.status = SolverStatus::kNonFiniteInput;
    evaluation.message = "accepted acceleration must be finite";
    return evaluation;
  }
  try {
    evaluation.torques = robot.rnea(q, dq, ddq);
  } catch (const std::exception &error) {
    evaluation.status = SolverStatus::kNumericalError;
    evaluation.message =
        std::string("inverse dynamics evaluation failed: ") + error.what();
    return evaluation;
  }
  if (evaluation.torques.size() != robot.nv()) {
    evaluation.status = SolverStatus::kShapeMismatch;
    evaluation.message = "inverse dynamics returned inconsistent dimensions";
    evaluation.torques.resize(0);
  } else if (!evaluation.torques.allFinite()) {
    evaluation.status = SolverStatus::kNumericalError;
    evaluation.message = "inverse dynamics returned non-finite torques";
    evaluation.torques.resize(0);
  }
  return evaluation;
}

AccelerationSolverResult clear_outputs(AccelerationSolverResult result) {
  result.solution.clear();
  result.joint_accelerations.resize(0);
  result.joint_velocities_next.resize(0);
  result.q_solution.resize(0);
  result.predicted_torques.resize(0);
  result.saturated_effort_indices.clear();
  result.allocation_diagnostics = {};
  return result;
}

void append_unique(std::vector<int> *indices, int index) {
  if (std::find(indices->begin(), indices->end(), index) == indices->end()) {
    indices->push_back(index);
  }
}

void attribute_saturation(AccelerationSolverResult *result,
                          const StateBoxAssembly &state_box,
                          const Eigen::VectorXd &ddq) {
  const auto position_causes =
      detail::state_box_cause(detail::StateBoxBoundCause::kStateRate) |
      detail::state_box_cause(detail::StateBoxBoundCause::kStateEndpoint) |
      detail::state_box_cause(detail::StateBoxBoundCause::kContinuousPath) |
      detail::state_box_cause(detail::StateBoxBoundCause::kNextStateViability);
  for (Eigen::Index index = 0; index < ddq.size(); ++index) {
    detail::StateBoxCauseSet causes = 0U;
    if (std::abs(ddq(index) - state_box.lower(index)) <=
        kConstraintTolerance) {
      causes |= state_box.lower_causes[static_cast<std::size_t>(index)];
    }
    if (std::abs(ddq(index) - state_box.upper(index)) <=
        kConstraintTolerance) {
      causes |= state_box.upper_causes[static_cast<std::size_t>(index)];
    }
    if (causes == 0U) {
      continue;
    }
    if ((causes & detail::state_box_cause(
                      detail::StateBoxBoundCause::kAcceleration)) != 0U) {
      append_unique(&result->saturated_acceleration_indices,
                    static_cast<int>(index));
    }
    if ((causes & detail::state_box_cause(detail::StateBoxBoundCause::kRate)) !=
        0U) {
      append_unique(&result->saturated_velocity_indices,
                    static_cast<int>(index));
    }
    if ((causes & position_causes) != 0U) {
      append_unique(&result->saturated_position_indices,
                    static_cast<int>(index));
    }
  }
}


} // namespace acceleration_solver_internal
} // namespace embodik
