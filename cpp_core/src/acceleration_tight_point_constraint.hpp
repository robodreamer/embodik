#pragma once

#include <embodik/acceleration_solver.hpp>

#include "acceleration_scalar_geometric_constraint.hpp"

#include <Eigen/Core>
#include <optional>
#include <string>
#include <vector>

namespace embodik::detail {

struct TightPointConstraintResult {
  SolverStatus status = SolverStatus::kSuccess;
  std::string message;

  bool satisfied() const { return status == SolverStatus::kSuccess; }
};

struct PreparedTightPointConstraint {
  TightPointAccelerationConstraint specification;
  std::vector<int> active_axes;
  PreparedScalarGeometricConstraint scalar;
};

struct TightPointPreparationResult : TightPointConstraintResult {
  std::optional<PreparedTightPointConstraint> prepared;
};

TightPointConstraintResult validate_tight_point_constraint(
    const TightPointAccelerationConstraint &constraint,
    const RobotModel &robot);

TightPointPreparationResult prepare_tight_point_constraint(
    const TightPointAccelerationConstraint &constraint, const RobotModel &robot,
    double dt, const Eigen::VectorXd &joint_acceleration_lower,
    const Eigen::VectorXd &joint_acceleration_upper);

TightPointConstraintResult validate_tight_point_constraint_acceptance(
    const PreparedTightPointConstraint &prepared, const RobotModel &robot,
    const Eigen::VectorXd &q, const Eigen::VectorXd &dq,
    const Eigen::VectorXd &accepted_acceleration, double dt,
    const Eigen::VectorXd &predicted_joint_acceleration_lower,
    const Eigen::VectorXd &predicted_joint_acceleration_upper);

} // namespace embodik::detail
