#pragma once

#include <embodik/types.hpp>

#include <Eigen/Core>

#include <string>

namespace embodik::detail {

struct AccelerationAllocationMatrixResult {
  SolverStatus status = SolverStatus::kSuccess;
  std::string message;
  Eigen::MatrixXd matrix;

  bool satisfied() const { return status == SolverStatus::kSuccess; }
};

struct AccelerationAllocationVectorResult {
  SolverStatus status = SolverStatus::kSuccess;
  std::string message;
  Eigen::VectorXd vector;

  bool satisfied() const { return status == SolverStatus::kSuccess; }
};

struct AccelerationAllocationHardRowsResult {
  SolverStatus status = SolverStatus::kSuccess;
  std::string message;
  Eigen::MatrixXd coefficient_matrix;
  Eigen::VectorXd lower_bounds;
  Eigen::VectorXd upper_bounds;

  bool satisfied() const { return status == SolverStatus::kSuccess; }
};

struct PreparedAccelerationAllocationTransform {
  SolverStatus status = SolverStatus::kSuccess;
  std::string message;
  Eigen::VectorXd requested_metric_diagonal;
  Eigen::VectorXd reference_acceleration;
  Eigen::VectorXd variable_scale;
  Eigen::VectorXd inverse_variable_scale;

  bool satisfied() const { return status == SolverStatus::kSuccess; }

  AccelerationAllocationMatrixResult transform_objective_matrix(
      const Eigen::MatrixXd &physical_matrix) const;
  AccelerationAllocationVectorResult transform_objective_bias(
      const Eigen::MatrixXd &physical_matrix,
      const Eigen::VectorXd &physical_bias) const;
  AccelerationAllocationHardRowsResult transform_backend_hard_rows(
      const Eigen::MatrixXd &coefficient_matrix,
      const Eigen::VectorXd &lower_shifted,
      const Eigen::VectorXd &upper_shifted) const;
  AccelerationAllocationVectorResult reconstruct_physical_acceleration(
      const Eigen::VectorXd &solver_variable) const;
  AccelerationAllocationVectorResult inverse_scale_physical_residual(
      const Eigen::VectorXd &physical_residual) const;
};

PreparedAccelerationAllocationTransform prepare_acceleration_allocation_transform(
    const Eigen::VectorXd &metric_diagonal,
    const Eigen::VectorXd &reference_acceleration,
    Eigen::Index variable_count);

} // namespace embodik::detail
