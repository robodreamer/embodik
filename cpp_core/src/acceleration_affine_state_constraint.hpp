#pragma once

#include "acceleration_scalar_geometric_constraint.hpp"

namespace embodik::detail {

struct PreparedAffineStateConstraint {
  AffineStateConstraint definition;
  ScalarGeometricDifferential current;
  PreparedScalarGeometricConstraint scalar;
};

struct AffineStatePreparationResult : ScalarGeometricConstraintResult {
  std::optional<PreparedAffineStateConstraint> prepared;
};

AffineStatePreparationResult prepare_affine_state_constraint(
    const AffineStateConstraint &constraint, const RobotModel &robot,
    const Eigen::VectorXd &q, const Eigen::VectorXd &dq, double dt,
    const Eigen::VectorXd &acceleration_lower,
    const Eigen::VectorXd &acceleration_upper);

ScalarGeometricConstraintResult validate_affine_state_constraint_acceptance(
    const PreparedAffineStateConstraint &prepared,
    const Eigen::VectorXd &ddq, double dt,
    const Eigen::VectorXd &q_next, const Eigen::VectorXd &dq_next,
    const Eigen::VectorXd &predicted_acceleration_lower,
    const Eigen::VectorXd &predicted_acceleration_upper);

} // namespace embodik::detail
