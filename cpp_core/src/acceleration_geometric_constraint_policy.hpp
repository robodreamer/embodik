#pragma once

#include <embodik/acceleration_solver.hpp>

#include <Eigen/Core>

#include <string>
#include <vector>

namespace embodik::detail {

struct GeometricConstraintPolicyValidation {
  SolverStatus status = SolverStatus::kSuccess;
  std::string message;

  bool satisfied() const { return status == SolverStatus::kSuccess; }
};

struct GeometricConstraintAxisValidation
    : GeometricConstraintPolicyValidation {
  std::vector<int> active_axes;
};

GeometricConstraintPolicyValidation validate_geometric_constraint_policy(
    const std::string &family_label, const std::string &source_id,
    const GeometricConstraintAccelerationPolicy &policy,
    Eigen::Index expected_dimension);

GeometricConstraintAxisValidation validate_geometric_constraint_axis_mask(
    const std::string &family_label, const std::string &source_id,
    const Eigen::Ref<const Eigen::VectorXd> &axis_mask,
    Eigen::Index expected_dimension);

} // namespace embodik::detail
