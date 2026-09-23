#pragma once

#include <embodik/acceleration_solver.hpp>

#include "acceleration_state_box.hpp"
#include "support_polygon_geometry.hpp"

#include <Eigen/Core>
#include <pinocchio/multibody/data.hpp>

#include <optional>
#include <string>
#include <vector>

namespace embodik::detail {

struct ComSupportPolygonConstraintResult {
  SolverStatus status = SolverStatus::kSuccess;
  std::string message;

  bool satisfied() const { return status == SolverStatus::kSuccess; }
};

struct PreparedComSupportPolygonConstraint {
  ComSupportPolygonAccelerationConstraint specification;
  SupportPolygonGeometry geometry;
  LinearizedStateBoxResult state_box;
  Eigen::VectorXd current_values;
  Eigen::VectorXd current_slacks;
  Eigen::VectorXd recovery_rate_targets;
  std::vector<int> outward_rows;
  std::vector<int> recovery_rows;
};

struct ComSupportPolygonPreparationResult
    : ComSupportPolygonConstraintResult {
  std::optional<PreparedComSupportPolygonConstraint> prepared;
};

bool is_structurally_fixed_support_frame(const RobotModel &robot,
                                         const std::string &frame_name);

ComSupportPolygonConstraintResult validate_com_support_polygon_constraint(
    const ComSupportPolygonAccelerationConstraint &constraint,
    const RobotModel &robot);

ComSupportPolygonPreparationResult prepare_com_support_polygon_constraint(
    const ComSupportPolygonAccelerationConstraint &constraint,
    const RobotModel &robot, double dt,
    const Eigen::VectorXd &joint_acceleration_lower,
    const Eigen::VectorXd &joint_acceleration_upper);

ComSupportPolygonPreparationResult
prepare_com_support_polygon_constraint_at_state(
    const ComSupportPolygonAccelerationConstraint &constraint,
    const SupportPolygonGeometry &geometry, const RobotModel &robot,
    pinocchio::Data &scratch, const Eigen::VectorXd &q,
    const Eigen::VectorXd &dq, double dt,
    const Eigen::VectorXd &joint_acceleration_lower,
    const Eigen::VectorXd &joint_acceleration_upper,
    const Eigen::VectorXd *witness_candidate = nullptr);

ComSupportPolygonConstraintResult
validate_com_support_polygon_constraint_acceptance(
    const PreparedComSupportPolygonConstraint &prepared,
    const RobotModel &robot, const Eigen::VectorXd &q,
    const Eigen::VectorXd &dq, const Eigen::VectorXd &accepted_acceleration,
    double dt, const Eigen::VectorXd &next_joint_acceleration_lower,
    const Eigen::VectorXd &next_joint_acceleration_upper);

} // namespace embodik::detail
