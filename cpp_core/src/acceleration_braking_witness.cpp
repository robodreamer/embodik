#include "acceleration_braking_witness.hpp"

#include "generalized_constraint_set.hpp"

#include <embodik/ik_baseline.hpp>

#include <cmath>
#include <limits>
#include <utility>

namespace embodik::detail {
namespace {

constexpr double kConstraintTolerance = 1e-8;
constexpr double kInactiveAffineBackendBound = 1e10;

AccelerationBrakingWitnessResult failure(SolverStatus status,
                                         std::string message) {
  return {status, std::move(message)};
}

bool direction_is_valid(AccelerationBrakingDirection direction) {
  switch (direction) {
  case AccelerationBrakingDirection::kIncreaseCoordinate:
  case AccelerationBrakingDirection::kDecreaseCoordinate:
    return true;
  }
  return false;
}

bool append_activity_aware_affine_block(
    GeneralizedConstraintSet *set,
    const AffineAccelerationConstraint &constraint) {
  if (set == nullptr) {
    return false;
  }
  const Eigen::Index rows = constraint.coefficient_matrix.rows();
  GeneralizedConstraintBlock block;
  block.coefficient_matrix = constraint.coefficient_matrix;
  block.affine_bias = constraint.affine_bias;
  block.physical_lower_bounds = constraint.lower_bounds;
  block.physical_upper_bounds = constraint.upper_bounds;
  for (Eigen::Index row = 0; row < rows; ++row) {
    const bool lower_active =
        constraint.lower_bound_active.empty() ||
        constraint.lower_bound_active[static_cast<std::size_t>(row)];
    const bool upper_active =
        constraint.upper_bound_active.empty() ||
        constraint.upper_bound_active[static_cast<std::size_t>(row)];
    const double inactive =
        kInactiveAffineBackendBound *
        std::max(1.0, constraint.coefficient_matrix.row(row).stableNorm());
    if (!lower_active) {
      block.physical_lower_bounds(row) = constraint.affine_bias(row) - inactive;
    }
    if (!upper_active) {
      block.physical_upper_bounds(row) = constraint.affine_bias(row) + inactive;
    }
  }
  return set->append_block(std::move(block));
}

} // namespace

AccelerationBrakingWitnessResult validate_common_acceleration_braking_witness(
    const AffineAccelerationConstraint &physical_constraint,
    const Eigen::VectorXd &joint_acceleration_lower,
    const Eigen::VectorXd &joint_acceleration_upper,
    const std::vector<AccelerationBrakingWitnessRow> &braking_rows,
    const std::string &state_label, const Eigen::VectorXd *candidate) {
  if (braking_rows.empty()) {
    return {};
  }
  const Eigen::Index variable_count =
      physical_constraint.coefficient_matrix.cols();
  const Eigen::Index row_count =
      physical_constraint.coefficient_matrix.rows();
  if (variable_count <= 0 || row_count <= 0 ||
      physical_constraint.affine_bias.size() != row_count ||
      physical_constraint.lower_bounds.size() != row_count ||
      physical_constraint.upper_bounds.size() != row_count ||
      !physical_constraint.coefficient_matrix.allFinite() ||
      !physical_constraint.affine_bias.allFinite() ||
      !physical_constraint.lower_bounds.allFinite() ||
      !physical_constraint.upper_bounds.allFinite() ||
      joint_acceleration_lower.size() != variable_count ||
      joint_acceleration_upper.size() != variable_count ||
      !joint_acceleration_lower.allFinite() ||
      !joint_acceleration_upper.allFinite() ||
      (joint_acceleration_lower.array() > joint_acceleration_upper.array())
          .any() ||
      state_label.empty()) {
    return failure(SolverStatus::kInvalidInput,
                   state_label + " common braking witness inputs are invalid");
  }

  std::vector<bool> seen(static_cast<std::size_t>(row_count), false);
  for (const auto &braking : braking_rows) {
    if (braking.constraint_row < 0 || braking.constraint_row >= row_count ||
        seen[static_cast<std::size_t>(braking.constraint_row)] ||
        !std::isfinite(braking.minimum_magnitude) ||
        braking.minimum_magnitude <= 0.0 ||
        !direction_is_valid(braking.direction)) {
      return failure(SolverStatus::kInvalidInput,
                     state_label +
                         " common braking witness rows are invalid");
    }
    seen[static_cast<std::size_t>(braking.constraint_row)] = true;
  }

  const Eigen::Index proof_rows =
      static_cast<Eigen::Index>(braking_rows.size());
  GeneralizedConstraintSet phase_one(variable_count,
                                     variable_count + row_count + proof_rows,
                                     false);
  GeneralizedConstraintBlock joint_box;
  joint_box.coefficient_matrix =
      Eigen::MatrixXd::Identity(variable_count, variable_count);
  joint_box.affine_bias = Eigen::VectorXd::Zero(variable_count);
  joint_box.physical_lower_bounds = joint_acceleration_lower;
  joint_box.physical_upper_bounds = joint_acceleration_upper;
  if (!phase_one.append_block(std::move(joint_box)) ||
      !append_activity_aware_affine_block(&phase_one, physical_constraint)) {
    return failure(SolverStatus::kShapeMismatch,
                   state_label + " common braking witness assembly failed");
  }

  AffineAccelerationConstraint braking_constraint;
  braking_constraint.source_id = state_label + ":braking_targets";
  braking_constraint.coefficient_matrix.resize(proof_rows, variable_count);
  braking_constraint.affine_bias.resize(proof_rows);
  braking_constraint.lower_bounds = Eigen::VectorXd::Zero(proof_rows);
  braking_constraint.upper_bounds = Eigen::VectorXd::Zero(proof_rows);
  braking_constraint.lower_bound_active.assign(
      static_cast<std::size_t>(proof_rows), false);
  braking_constraint.upper_bound_active.assign(
      static_cast<std::size_t>(proof_rows), false);
  for (Eigen::Index row = 0; row < proof_rows; ++row) {
    const auto &target = braking_rows[static_cast<std::size_t>(row)];
    braking_constraint.coefficient_matrix.row(row) =
        physical_constraint.coefficient_matrix.row(target.constraint_row);
    braking_constraint.affine_bias(row) =
        physical_constraint.affine_bias(target.constraint_row);
    if (target.direction == AccelerationBrakingDirection::kIncreaseCoordinate) {
      braking_constraint.lower_bounds(row) = target.minimum_magnitude;
      braking_constraint.lower_bound_active[static_cast<std::size_t>(row)] =
          true;
    } else {
      braking_constraint.upper_bounds(row) = -target.minimum_magnitude;
      braking_constraint.upper_bound_active[static_cast<std::size_t>(row)] =
          true;
    }
  }
  if (!append_activity_aware_affine_block(&phase_one, braking_constraint) ||
      !phase_one.finalize()) {
    return failure(SolverStatus::kShapeMismatch,
                   state_label + " common braking witness assembly failed");
  }

  if (candidate != nullptr && candidate->size() == variable_count &&
      candidate->allFinite() &&
      ComputeMaxLinearConstraintViolation(
          phase_one.coefficient_matrix(), phase_one.lower_bounds(),
          phase_one.upper_bounds(), *candidate) <= kConstraintTolerance) {
    return {};
  }

  Eigen::VectorXd witness;
  VelocitySolverConfig config;
  config.epsilon = 1e-12;
  config.precision_threshold = 1e-12;
  config.iteration_limit = 50;
  config.magnitude_limit = 1e10;
  const auto status = FindHardConstraintFeasibleWitness(
      phase_one.coefficient_matrix(), phase_one.lower_bounds(),
      phase_one.upper_bounds(), config, &witness);
  if (status == HardConstraintFeasibilityStatus::kProvenInfeasible) {
    return failure(SolverStatus::kInfeasible,
                   state_label + " common braking witness is infeasible");
  }
  if (status == HardConstraintFeasibilityStatus::kUnknown ||
      witness.size() != variable_count || !witness.allFinite() ||
      ComputeMaxLinearConstraintViolation(
          phase_one.coefficient_matrix(), phase_one.lower_bounds(),
          phase_one.upper_bounds(), witness) > kConstraintTolerance) {
    return failure(
        SolverStatus::kNoProgress,
        state_label + " common braking witness could not be certified");
  }
  return {};
}

} // namespace embodik::detail
