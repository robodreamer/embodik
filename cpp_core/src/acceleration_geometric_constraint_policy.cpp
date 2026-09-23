#include "acceleration_geometric_constraint_policy.hpp"

#include <cmath>
#include <utility>

namespace embodik::detail {
namespace {

constexpr double kBinaryMaskTolerance = 1e-8;

GeometricConstraintPolicyValidation failure(SolverStatus status,
                                            std::string message) {
  GeometricConstraintPolicyValidation result;
  result.status = status;
  result.message = std::move(message);
  return result;
}

bool all_policy_vectors_have_dimension(
    const GeometricConstraintAccelerationPolicy &policy,
    Eigen::Index expected_dimension) {
  return policy.rate_limits.size() == expected_dimension &&
         policy.acceleration_limits.size() == expected_dimension &&
         policy.lower_braking_accelerations.size() == expected_dimension &&
         policy.upper_braking_accelerations.size() == expected_dimension;
}

} // namespace

GeometricConstraintPolicyValidation validate_geometric_constraint_policy(
    const std::string &family_label, const std::string &source_id,
    const GeometricConstraintAccelerationPolicy &policy,
    Eigen::Index expected_dimension) {
  const std::string identity =
      family_label + " '" + source_id + "' geometric policy";
  if (expected_dimension <= 0 ||
      !all_policy_vectors_have_dimension(policy, expected_dimension)) {
    return failure(SolverStatus::kShapeMismatch,
                   identity + " vectors must all have size " +
                       std::to_string(expected_dimension));
  }
  if (!policy.rate_limits.allFinite() ||
      !policy.acceleration_limits.allFinite() ||
      !policy.lower_braking_accelerations.allFinite() ||
      !policy.upper_braking_accelerations.allFinite()) {
    return failure(SolverStatus::kNonFiniteInput,
                   identity + " must contain only finite values");
  }
  if ((policy.rate_limits.array() <= 0.0).any() ||
      (policy.acceleration_limits.array() <= 0.0).any() ||
      (policy.lower_braking_accelerations.array() <= 0.0).any() ||
      (policy.upper_braking_accelerations.array() <= 0.0).any()) {
    return failure(SolverStatus::kInvalidInput,
                   identity + " magnitudes must be positive");
  }
  if ((policy.lower_braking_accelerations.array() >
       policy.acceleration_limits.array())
          .any() ||
      (policy.upper_braking_accelerations.array() >
       policy.acceleration_limits.array())
          .any()) {
    return failure(
        SolverStatus::kInvalidInput,
        identity + " braking magnitudes cannot exceed acceleration limits");
  }
  return {};
}

GeometricConstraintAxisValidation validate_geometric_constraint_axis_mask(
    const std::string &family_label, const std::string &source_id,
    const Eigen::Ref<const Eigen::VectorXd> &axis_mask,
    Eigen::Index expected_dimension) {
  GeometricConstraintAxisValidation result;
  const std::string identity =
      family_label + " '" + source_id + "' axis_mask";
  if (expected_dimension <= 0 || axis_mask.size() != expected_dimension) {
    result.status = SolverStatus::kShapeMismatch;
    result.message =
        identity + " must have size " + std::to_string(expected_dimension);
    return result;
  }
  if (!axis_mask.allFinite()) {
    result.status = SolverStatus::kNonFiniteInput;
    result.message = identity + " must contain only finite values";
    return result;
  }
  for (Eigen::Index axis = 0; axis < axis_mask.size(); ++axis) {
    const double value = axis_mask(axis);
    if (std::abs(value) <= kBinaryMaskTolerance) {
      continue;
    }
    if (std::abs(value - 1.0) <= kBinaryMaskTolerance) {
      result.active_axes.push_back(static_cast<int>(axis));
      continue;
    }
    result.status = SolverStatus::kInvalidInput;
    result.message = identity + " must be binary";
    result.active_axes.clear();
    return result;
  }
  if (result.active_axes.empty()) {
    result.status = SolverStatus::kInvalidInput;
    result.message = identity + " must activate at least one axis";
  }
  return result;
}

} // namespace embodik::detail
