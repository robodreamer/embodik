#pragma once

#include <embodik/tasks.hpp>

#include <Eigen/Core>

#include <string>

namespace embodik::detail {

enum class AccelerationTaskDifferentialStatus {
  kSuccess,
  kUnsupportedTask,
  kInvalidInput,
};

struct AccelerationTaskDifferential {
  AccelerationTaskDifferentialStatus status =
      AccelerationTaskDifferentialStatus::kInvalidInput;
  std::string message;
  Eigen::MatrixXd physical_jacobian;
  Eigen::MatrixXd control_jacobian;
  Eigen::VectorXd position_error;
  Eigen::VectorXd jacobian_bias;
  Eigen::VectorXd reference_row_scale;
};

AccelerationTaskDifferential evaluate_acceleration_task_differential(
    const Task &task, const RobotModel &model);

} // namespace embodik::detail
