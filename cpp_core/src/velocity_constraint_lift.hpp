#pragma once

#include <Eigen/Dense>

#include <cmath>
#include <stdexcept>

namespace embodik::detail {

struct LiftedVelocityConstraint {
  Eigen::MatrixXd coefficient_matrix;
  Eigen::VectorXd lower_bounds;
  Eigen::VectorXd upper_bounds;
};

// Algebraic sampled-next-velocity lift:
//   lower <= A * (dq + dt * ddq) <= upper
// becomes:
//   (lower - A*dq) / dt <= A * ddq <= (upper - A*dq) / dt.
// This preserves the frozen row exactly at the sampled next velocity. It is not
// a continuous collision certificate between samples.
inline LiftedVelocityConstraint lift_velocity_constraint_to_acceleration(
    const Eigen::MatrixXd &coefficient_matrix,
    const Eigen::VectorXd &velocity_lower_bounds,
    const Eigen::VectorXd &velocity_upper_bounds,
    const Eigen::VectorXd &current_velocity, double dt) {
  if (dt <= 0.0 || !std::isfinite(dt)) {
    throw std::invalid_argument("dt must be finite and positive");
  }
  if (coefficient_matrix.rows() == 0 || coefficient_matrix.cols() == 0) {
    throw std::invalid_argument(
        "velocity constraint lift requires non-empty rows");
  }
  if (coefficient_matrix.rows() != velocity_lower_bounds.size() ||
      coefficient_matrix.rows() != velocity_upper_bounds.size() ||
      coefficient_matrix.cols() != current_velocity.size()) {
    throw std::invalid_argument(
        "velocity constraint lift dimension mismatch");
  }
  if (!coefficient_matrix.allFinite() ||
      !velocity_lower_bounds.allFinite() ||
      !velocity_upper_bounds.allFinite() ||
      !current_velocity.allFinite()) {
    throw std::invalid_argument(
        "velocity constraint lift inputs must be finite");
  }
  if ((velocity_lower_bounds.array() > velocity_upper_bounds.array()).any()) {
    throw std::invalid_argument(
        "velocity constraint lift lower bounds exceed upper bounds");
  }

  LiftedVelocityConstraint result;
  result.coefficient_matrix = coefficient_matrix;
  const Eigen::VectorXd row_velocity =
      coefficient_matrix * current_velocity;
  result.lower_bounds = (velocity_lower_bounds - row_velocity) / dt;
  result.upper_bounds = (velocity_upper_bounds - row_velocity) / dt;
  return result;
}

} // namespace embodik::detail
