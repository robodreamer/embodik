#pragma once

#include <embodik/acceleration_solver.hpp>

#include "acceleration_scalar_geometric_constraint.hpp"

#include <Eigen/Core>
#include <optional>
#include <string>
#include <vector>

namespace embodik::detail {

struct RelativePoseConstraintResult {
  SolverStatus status = SolverStatus::kSuccess;
  std::string message;

  bool satisfied() const { return status == SolverStatus::kSuccess; }
};

struct PreparedRelativePoseConstraint {
  RelativePoseAccelerationConstraint specification;
  std::vector<int> active_axes;
  bool rotation_active = false;
  Eigen::VectorXd angular_path_support_mask;
  PreparedScalarGeometricConstraint scalar;
};

struct RelativePosePreparationResult : RelativePoseConstraintResult {
  std::optional<PreparedRelativePoseConstraint> prepared;
};

Eigen::VectorXd relative_pose_angular_path_support_mask(
    const RobotModel &robot, const std::string &frame_a,
    const std::string &frame_b);

RelativePosePreparationResult prepare_relative_pose_constraint(
    const RelativePoseAccelerationConstraint &constraint,
    const RobotModel &robot, double dt,
    const Eigen::VectorXd &joint_acceleration_lower,
    const Eigen::VectorXd &joint_acceleration_upper);

RelativePoseConstraintResult validate_relative_pose_constraint_acceptance(
    const PreparedRelativePoseConstraint &prepared, const RobotModel &robot,
    const Eigen::VectorXd &q, const Eigen::VectorXd &dq,
    const Eigen::VectorXd &accepted_acceleration, double dt,
    const Eigen::VectorXd &predicted_joint_acceleration_lower,
    const Eigen::VectorXd &predicted_joint_acceleration_upper);

} // namespace embodik::detail
