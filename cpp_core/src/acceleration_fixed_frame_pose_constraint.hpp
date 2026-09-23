#pragma once

#include <embodik/acceleration_solver.hpp>

#include "acceleration_scalar_geometric_constraint.hpp"

#include <Eigen/Core>
#include <optional>
#include <string>
#include <vector>

namespace embodik::detail {

struct FixedFramePoseConstraintRecord {
  std::string family_label;
  std::string source_id;
  std::string frame_name;
  Eigen::Matrix4d reference_pose = Eigen::Matrix4d::Identity();
  Eigen::Matrix<double, 6, 1> lower_bounds =
      Eigen::Matrix<double, 6, 1>::Zero();
  Eigen::Matrix<double, 6, 1> upper_bounds =
      Eigen::Matrix<double, 6, 1>::Zero();
  Eigen::Matrix<double, 6, 1> axis_mask =
      Eigen::Matrix<double, 6, 1>::Ones();
  GeometricConstraintAccelerationPolicy policy;
};

struct FixedFramePoseConstraintResult {
  SolverStatus status = SolverStatus::kSuccess;
  std::string message;

  bool satisfied() const { return status == SolverStatus::kSuccess; }
};

struct PreparedFixedFramePoseConstraint {
  FixedFramePoseConstraintRecord specification;
  std::vector<int> active_axes;
  bool rotation_active = false;
  PreparedScalarGeometricConstraint scalar;
};

struct FixedFramePosePreparationResult : FixedFramePoseConstraintResult {
  std::optional<PreparedFixedFramePoseConstraint> prepared;
};

FixedFramePoseConstraintRecord make_tight_frame_pose_record(
    const TightFramePoseAccelerationConstraint &constraint);

FixedFramePoseConstraintResult make_torso_pose_bound_record(
    const TorsoPoseBoundAccelerationConstraint &constraint,
    FixedFramePoseConstraintRecord *record);

FixedFramePosePreparationResult prepare_fixed_frame_pose_constraint(
    const FixedFramePoseConstraintRecord &constraint, const RobotModel &robot,
    double dt, const Eigen::VectorXd &joint_acceleration_lower,
    const Eigen::VectorXd &joint_acceleration_upper);

FixedFramePoseConstraintResult validate_fixed_frame_pose_constraint_acceptance(
    const PreparedFixedFramePoseConstraint &prepared, const RobotModel &robot,
    const Eigen::VectorXd &q, const Eigen::VectorXd &dq,
    const Eigen::VectorXd &accepted_acceleration, double dt,
    const Eigen::VectorXd &predicted_joint_acceleration_lower,
    const Eigen::VectorXd &predicted_joint_acceleration_upper);

} // namespace embodik::detail
