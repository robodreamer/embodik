#pragma once

#include <embodik/acceleration_solver.hpp>

#include <Eigen/Core>

#include <string>
#include <vector>

namespace embodik::detail {

enum class AccelerationBrakingDirection {
  kIncreaseCoordinate,
  kDecreaseCoordinate,
};

struct AccelerationBrakingWitnessRow {
  int constraint_row = -1;
  double minimum_magnitude = 0.0;
  AccelerationBrakingDirection direction =
      AccelerationBrakingDirection::kIncreaseCoordinate;
};

struct AccelerationBrakingWitnessResult {
  SolverStatus status = SolverStatus::kSuccess;
  std::string message;

  bool satisfied() const { return status == SolverStatus::kSuccess; }
};

AccelerationBrakingWitnessResult validate_common_acceleration_braking_witness(
    const AffineAccelerationConstraint &physical_constraint,
    const Eigen::VectorXd &joint_acceleration_lower,
    const Eigen::VectorXd &joint_acceleration_upper,
    const std::vector<AccelerationBrakingWitnessRow> &braking_rows,
    const std::string &state_label, const Eigen::VectorXd *candidate = nullptr);

} // namespace embodik::detail
