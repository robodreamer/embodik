/*
 * Copyright 2025-2026 Andy Park <andypark.purdue@gmail.com>
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     https://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include "types.hpp"

#include <chrono>
#include <cmath>
#include <limits>
#include <optional>
#include <stdexcept>
#include <vector>

#include <Eigen/Core>
#include <Eigen/LU>
#include <Eigen/SVD>
#include <algorithm>
#include <embodik/constraints.hpp>

namespace embodik {

namespace detail {

/// @brief Minimum acceptable magnitude for scaling coefficients
constexpr double kMinScalingMagnitude = 1e-10;

/// @brief Maximum acceptable magnitude for scaling coefficients
constexpr double kMaxScalingMagnitude = 1e+10;

/**
 * @brief Calculates the generalized inverse using orthogonal decomposition.
 */
template <typename Derived>
inline void ComputeGeneralizedInverse(
    const Eigen::MatrixBase<Derived> &matrix_input, double relative_tolerance,
    Eigen::Matrix<typename Derived::Scalar,
                  Derived::PlainMatrix::ColsAtCompileTime,
                  Derived::PlainMatrix::RowsAtCompileTime> *result_matrix) {
  using MatrixType = typename Derived::PlainMatrix;
  Eigen::CompleteOrthogonalDecomposition<MatrixType> orthogonal_decomp;
  orthogonal_decomp.setThreshold(relative_tolerance);
  orthogonal_decomp.compute(matrix_input);
  result_matrix->noalias() = orthogonal_decomp.pseudoInverse();
}

} // namespace detail

namespace linalg {

inline Eigen::MatrixXd ComputeRegularizedInverse(
    const embodik::RegularizedInverseConfig &regularization_config,
    const Eigen::MatrixXd &input_matrix,
    double *condition_number_out = nullptr) {
  using MatrixDouble = Eigen::MatrixXd;
  using SquareMatrix = Eigen::MatrixXd; // square matrix for gram computation

  SquareMatrix gram_matrix = input_matrix * input_matrix.transpose();

  // Singular value decomposition is already needed for the SRINV damping law;
  // reuse it to report condition-number diagnostics to callers.
  const Eigen::BDCSVD<MatrixDouble> svd_decomp(input_matrix,
                                               Eigen::ComputeFullU);
  const auto &sigma_values = svd_decomp.singularValues();
  const auto &left_singular_vectors = svd_decomp.matrixU();

  // Compute condition number and report to caller
  double condition_number = 1.0;
  if (sigma_values.size() > 0) {
    const double sigma_max = sigma_values(0);
    const double sigma_min = sigma_values(sigma_values.size() - 1);
    condition_number = (sigma_min > regularization_config.epsilon * 1e-4)
                           ? sigma_max / sigma_min
                           : std::numeric_limits<double>::infinity();
  }
  if (condition_number_out) {
    *condition_number_out = condition_number;
  }

  // Two-layer singularity-robust damping:
  //
  // Layer 1: Global regularization based on gram matrix condition.
  // Provides a safety floor when the matrix is poorly conditioned even
  // if no individual singular value is below epsilon (e.g., many moderate
  // SVs whose product is tiny). Uses the same determinant-based formula
  // as the original but provides numerical stability.
  const double threshold_squared =
      std::pow(regularization_config.epsilon, 2.0);
  const double det_value = gram_matrix.determinant();
  const double global_regularization =
      (det_value < threshold_squared)
          ? (1.0 - std::pow(det_value / threshold_squared, 2.0)) *
                threshold_squared
          : 0.0;

  SquareMatrix regularized_gram = gram_matrix;
  regularized_gram.diagonal().array() += global_regularization;

  // Layer 2: Per-singular-value damping with smooth quadratic ramp from full
  // damping (at sigma=0) to zero (at sigma>=epsilon).
  if (sigma_values.size() > 0) {
    const double eps = regularization_config.epsilon;
    const double lambda_max = regularization_config.regularization_factor;

    // Continuous quadratic ramp: lambda(sigma) = lambda_max * (1 - (sigma/eps)^2)
    // for sigma < eps, 0 otherwise. This is C1-continuous at sigma=eps.
    Eigen::ArrayXd per_sv_damping =
        lambda_max *
        (1.0 - (sigma_values.array() / eps).min(1.0).square());
    per_sv_damping = per_sv_damping.max(0.0);

    if ((per_sv_damping > 0.0).any()) {
      regularized_gram.noalias() +=
          left_singular_vectors *
          per_sv_damping.matrix().asDiagonal() *
          left_singular_vectors.transpose();
    }
  }

  return input_matrix.transpose() * regularized_gram.inverse();
}

inline constexpr std::pair<double, double>
ComputeFeasibleScalingRange(double bound_lower, double bound_upper,
                            double coefficient) {
  // Computes valid scaling range to satisfy bounds
  double max_allowed_scale = 1.0;
  double min_allowed_scale = 0.0;
  if (detail::kMinScalingMagnitude < std::fabs(coefficient) &&
      std::fabs(coefficient) < detail::kMaxScalingMagnitude) {
    if (coefficient < 0.0 && bound_lower < 0.0 && coefficient <= bound_upper) {
      max_allowed_scale = bound_lower / coefficient;
      min_allowed_scale = bound_upper / coefficient;
    } else if (coefficient > 0.0 && bound_upper > 0.0 &&
               coefficient >= bound_lower) {
      max_allowed_scale = bound_upper / coefficient;
      min_allowed_scale = bound_lower / coefficient;
    }
  }
  return {std::min(1.0, max_allowed_scale), std::max(0.0, min_allowed_scale)};
}

} // namespace linalg

// Forward declarations for overloaded solver implementations
inline SolverResult computeMultiObjectiveVelocitySolution(
    const std::vector<std::vector<double>> &target_velocities,
    const std::vector<std::vector<std::vector<double>>> &task_jacobians,
    const std::vector<std::vector<double>> &constraint_matrix,
    const std::vector<double> &min_bounds,
    const std::vector<double> &max_bounds,
    const VelocitySolverConfig &solver_config);

inline SolverResult computeMultiObjectiveVelocitySolutionEigen(
    const std::vector<Eigen::VectorXd> &objective_targets,
    const std::vector<Eigen::MatrixXd> &objective_jacobians,
    const Eigen::MatrixXd &constraint_coefficients,
    const Eigen::VectorXd &min_bounds, const Eigen::VectorXd &max_bounds,
    const VelocitySolverConfig &solver_config,
    const std::vector<ObjectiveSolveConfig> &objective_configs = {},
    const Eigen::VectorXd *max_constraint_softening_factors = nullptr);

inline double
calculateConfigurationDistance(const std::vector<double> &position_a,
                               const std::vector<double> &position_b) {
  if (position_a.size() != position_b.size() || position_a.empty()) {
    return std::numeric_limits<double>::infinity();
  }
  double squared_distance = 0.0;
  for (size_t idx = 0; idx < position_a.size(); ++idx) {
    const double difference = position_a[idx] - position_b[idx];
    squared_distance += difference * difference;
  }
  return std::sqrt(squared_distance);
}

// Multi-objective velocity solver with hierarchical constraint enforcement
inline SolverResult computeMultiObjectiveVelocitySolution(
    const std::vector<std::vector<double>>
        &target_velocities, // collection of m_i dimensional targets
    const std::vector<std::vector<std::vector<double>>>
        &task_jacobians, // collection of m_i x n matrices
    const std::vector<std::vector<double>>
        &constraint_matrix,                // (n+k) x n system constraints
    const std::vector<double> &min_bounds, // (n+k) minimum limits
    const std::vector<double> &max_bounds, // (n+k) maximum limits
    const VelocitySolverConfig &solver_config = VelocitySolverConfig{}) {
  auto t0 = std::chrono::high_resolution_clock::now();
  auto get_elapsed_ms = [&t0]() {
    return std::chrono::duration_cast<std::chrono::microseconds>(
               std::chrono::high_resolution_clock::now() - t0)
               .count() /
           1000.0;
  };

  auto fail = [&](SolverStatus status, const char *msg) {
    SolverResult out{};
    out.status = status;
    out.computation_time_ms = get_elapsed_ms();
    out.status_message = msg;
    return out;
  };

  // Validate input dimensions
  if (target_velocities.size() != task_jacobians.size())
    return fail(SolverStatus::kShapeMismatch,
                "goals/jacobians size mismatch in multi-objective input");
  if (constraint_matrix.empty())
    return fail(SolverStatus::kEmptyProblem, "constraint matrix is empty");
  if (target_velocities.empty())
    return fail(SolverStatus::kEmptyProblem, "no objectives provided");

  const size_t num_dof = constraint_matrix[0].size();
  const size_t objective_count = target_velocities.size();

  // Verify matrix dimensions consistency
  for (size_t obj_idx = 0; obj_idx < objective_count; ++obj_idx) {
    if (task_jacobians[obj_idx].empty() ||
        task_jacobians[obj_idx][0].size() != num_dof) {
      return fail(SolverStatus::kShapeMismatch,
                  "jacobian has invalid or mismatched width");
    }
    if (target_velocities[obj_idx].size() != task_jacobians[obj_idx].size()) {
      return fail(SolverStatus::kShapeMismatch,
                  "goal size does not match jacobian rows");
    }
  }

  if (constraint_matrix.size() != min_bounds.size() ||
      constraint_matrix.size() != max_bounds.size()) {
    return fail(SolverStatus::kConstraintBoundsMismatch,
                "constraint row count must match lower/upper bounds sizes");
  }

  // Transform input data to Eigen format for numerical computation
  std::vector<Eigen::VectorXd> eigen_objectives;
  std::vector<Eigen::MatrixXd> eigen_jacobians;
  eigen_objectives.reserve(objective_count);
  eigen_jacobians.reserve(objective_count);

  for (size_t obj_idx = 0; obj_idx < objective_count; ++obj_idx) {
    const auto &target_vel = target_velocities[obj_idx];
    const auto &jacobian = task_jacobians[obj_idx];

    Eigen::VectorXd objective_vector(target_vel.size());
    for (size_t elem = 0; elem < target_vel.size(); ++elem) {
      objective_vector(static_cast<Eigen::Index>(elem)) = target_vel[elem];
    }

    Eigen::MatrixXd jacobian_matrix(jacobian.size(), num_dof);
    for (size_t row = 0; row < jacobian.size(); ++row) {
      if (jacobian[row].size() != num_dof)
        return fail(SolverStatus::kShapeMismatch,
                    "inconsistent jacobian row width");
      for (size_t col = 0; col < num_dof; ++col) {
        jacobian_matrix(static_cast<Eigen::Index>(row),
                        static_cast<Eigen::Index>(col)) = jacobian[row][col];
      }
    }

    eigen_objectives.push_back(std::move(objective_vector));
    eigen_jacobians.push_back(std::move(jacobian_matrix));
  }

  // Transform constraint data to matrix form
  Eigen::MatrixXd eigen_constraints(constraint_matrix.size(), num_dof);
  for (size_t row = 0; row < constraint_matrix.size(); ++row) {
    if (constraint_matrix[row].size() != num_dof)
      return fail(SolverStatus::kShapeMismatch,
                  "inconsistent constraint row width");
    for (size_t col = 0; col < num_dof; ++col) {
      eigen_constraints(static_cast<Eigen::Index>(row),
                        static_cast<Eigen::Index>(col)) =
          constraint_matrix[row][col];
    }
  }

  Eigen::VectorXd eigen_min_bounds(min_bounds.size());
  Eigen::VectorXd eigen_max_bounds(max_bounds.size());
  for (size_t idx = 0; idx < min_bounds.size(); ++idx) {
    eigen_min_bounds(static_cast<Eigen::Index>(idx)) = min_bounds[idx];
    eigen_max_bounds(static_cast<Eigen::Index>(idx)) = max_bounds[idx];
  }

  // Delegate to Eigen-based implementation
  return computeMultiObjectiveVelocitySolutionEigen(
      eigen_objectives, eigen_jacobians, eigen_constraints, eigen_min_bounds,
      eigen_max_bounds, solver_config, {}, nullptr);
}

// Eigen-based hierarchical velocity solver implementation
inline SolverResult computeMultiObjectiveVelocitySolutionEigen(
    const std::vector<Eigen::VectorXd>
        &objective_targets, // m_i dimensional targets
    const std::vector<Eigen::MatrixXd>
        &objective_jacobians, // m_i x n transformation matrices
    const Eigen::MatrixXd
        &constraint_coefficients,      // (n+k) x n constraint system
    const Eigen::VectorXd &min_bounds, // (n+k) lower bounds
    const Eigen::VectorXd &max_bounds, // (n+k) upper bounds
    const VelocitySolverConfig &solver_config,
    const std::vector<ObjectiveSolveConfig> &objective_configs,
    const Eigen::VectorXd *max_constraint_softening_factors) {
  auto t0 = std::chrono::high_resolution_clock::now();
  auto get_elapsed_ms = [&t0]() {
    return std::chrono::duration_cast<std::chrono::microseconds>(
               std::chrono::high_resolution_clock::now() - t0)
               .count() /
           1000.0;
  };

  auto fail = [&](SolverStatus status, const char *msg) {
    SolverResult out{};
    out.status = status;
    out.computation_time_ms = get_elapsed_ms();
    out.status_message = msg;
    return out;
  };

  // Validate input consistency
  if (objective_targets.size() != objective_jacobians.size())
    return fail(SolverStatus::kShapeMismatch,
                "goals/jacobians size mismatch in Eigen input");
  if (constraint_coefficients.cols() == 0 ||
      constraint_coefficients.rows() == 0)
    return fail(SolverStatus::kEmptyProblem, "empty constraint matrix");
  if (min_bounds.size() != constraint_coefficients.rows() ||
      max_bounds.size() != constraint_coefficients.rows())
    return fail(SolverStatus::kConstraintBoundsMismatch,
                "constraint rows must match lower/upper bounds");
  if (objective_targets.empty())
    return fail(SolverStatus::kEmptyProblem, "no objectives provided");

  // Extract system dimensions
  const auto degrees_of_freedom = constraint_coefficients.cols();
  const auto additional_constraints =
      constraint_coefficients.rows() - degrees_of_freedom;
  const auto total_constraints = constraint_coefficients.rows();
  const bool enhanced_mode = max_constraint_softening_factors != nullptr;
  const auto num_objectives = objective_jacobians.size();
  auto fail_non_finite = [&](const char *msg) {
    SolverResult out = fail(SolverStatus::kNonFiniteInput, msg);
    out.solution.assign(static_cast<std::size_t>(degrees_of_freedom), 0.0);
    return out;
  };
  Eigen::VectorXd softening_factors = Eigen::VectorXd::Ones(total_constraints);
  if (enhanced_mode && max_constraint_softening_factors != nullptr) {
    if (max_constraint_softening_factors->size() != total_constraints) {
      return fail(SolverStatus::kShapeMismatch,
                  "softening factors size must match constraint rows");
    }
    softening_factors = *max_constraint_softening_factors;
    for (Eigen::Index i = 0; i < softening_factors.size(); ++i) {
      if (!std::isfinite(softening_factors(i)) || softening_factors(i) < 1.0) {
        return fail(SolverStatus::kInvalidInput,
                    "softening factors must be finite and >= 1");
      }
    }
  }
  Eigen::VectorXd objective_scaling_factors =
      Eigen::VectorXd::Zero(num_objectives);
  std::vector<TaskSolveMode> objective_effective_modes(
      num_objectives, TaskSolveMode::kScale);
  std::vector<bool> objective_used_fallback(num_objectives, false);

  // Verify objective dimension compatibility
  for (size_t obj_idx = 0; obj_idx < num_objectives; ++obj_idx) {
    if (objective_jacobians[obj_idx].cols() != degrees_of_freedom ||
        objective_jacobians[obj_idx].rows() !=
            objective_targets[obj_idx].rows()) {
      return fail(SolverStatus::kShapeMismatch,
                  "objective Jacobian dimension mismatch");
    }
    if (!objective_jacobians[obj_idx].allFinite() ||
        !objective_targets[obj_idx].allFinite()) {
      return fail_non_finite(
          "objective inputs must contain only finite values");
    }
  }
  if (!constraint_coefficients.allFinite() || !min_bounds.allFinite() ||
      !max_bounds.allFinite()) {
    return fail_non_finite(
        "constraint inputs must contain only finite values");
  }

  // Track worst-case condition number across all tasks.
  double worst_condition_number = 1.0;

  // Fast path for the common interactive case: one combined objective whose
  // unconstrained damped least-squares velocity already satisfies all rows.
  // This avoids the active-set SNS machinery without changing constrained
  // behavior; any violation falls through to the existing solver below.
  if (!enhanced_mode && num_objectives == 1U) {
    const ObjectiveSolveConfig objective_config =
        objective_configs.empty() ? ObjectiveSolveConfig{} : objective_configs[0];
    const auto &current_jacobian = objective_jacobians[0];
    const auto &current_target = objective_targets[0];
    double direct_condition_number = 1.0;
    Eigen::VectorXd direct_velocity =
        linalg::ComputeRegularizedInverse(solver_config.regularization_config,
                                          current_jacobian,
                                          &direct_condition_number) *
        current_target;
    worst_condition_number =
        std::max(worst_condition_number, direct_condition_number);
    if (direct_velocity.size() == degrees_of_freedom &&
        direct_velocity.allFinite()) {
      const Eigen::VectorXd direct_constraint_eval =
          constraint_coefficients * direct_velocity;
      const bool feasible =
          !((direct_constraint_eval.array() <
             (min_bounds.array() - solver_config.epsilon))
                .any() ||
            (direct_constraint_eval.array() >
             (max_bounds.array() + solver_config.epsilon))
                .any());
      if (feasible) {
        std::vector<double> final_velocities(
            static_cast<size_t>(degrees_of_freedom));
        for (Eigen::Index dof_idx = 0; dof_idx < degrees_of_freedom;
             ++dof_idx) {
          final_velocities[static_cast<size_t>(dof_idx)] =
              direct_velocity(dof_idx);
        }
        std::vector<double> applied_scales{
            objective_config.solve_mode == TaskSolveMode::kMinError ? -1.0
                                                                    : 1.0};
        std::vector<double> objective_errors{
            (current_jacobian * direct_velocity - current_target).norm()};
        std::vector<TaskSolveMode> objective_effective_modes{
            objective_config.solve_mode == TaskSolveMode::kScaleElastic
                ? TaskSolveMode::kScale
                : objective_config.solve_mode};
        std::vector<bool> objective_used_fallback{false};
        return SolverResult{
            std::move(final_velocities),
            SolverStatus::kSuccess,
            get_elapsed_ms(),
            1U,
            0.0,
            std::move(applied_scales),
            std::move(objective_errors),
            std::move(objective_effective_modes),
            std::move(objective_used_fallback),
            "",
            worst_condition_number};
      }
    }
  }

  // Initialize workspace matrices for null-space computation
  Eigen::MatrixXd null_space_projector =
      Eigen::MatrixXd::Identity(degrees_of_freedom, degrees_of_freedom);
  Eigen::MatrixXd saturated_constraint_selector =
      Eigen::MatrixXd::Zero(total_constraints, total_constraints);
  Eigen::VectorXd velocity_solution = Eigen::VectorXd::Zero(degrees_of_freedom);
  Eigen::VectorXd saturated_values =
      Eigen::VectorXd::Zero(total_constraints);
  Eigen::MatrixXd saturated_constraint_matrix = Eigen::MatrixXd::Zero(
      total_constraints, degrees_of_freedom);

  // Pre-allocate working memory for optimization loop
  Eigen::VectorXd previous_velocity(degrees_of_freedom);
  Eigen::VectorXd best_saturated_values(saturated_values.rows());
  Eigen::VectorXd constraint_evaluation(saturated_values.rows());
  Eigen::VectorXd scaled_velocity_contribution(saturated_values.rows());
  Eigen::VectorXd unscaled_contribution(saturated_values.rows());
  Eigen::VectorXd min_margin(saturated_values.rows());
  Eigen::VectorXd max_margin(saturated_values.rows());
  Eigen::VectorXd feasible_scales(saturated_values.rows());
  Eigen::MatrixXd augmented_projector(
      degrees_of_freedom, degrees_of_freedom + additional_constraints);
  Eigen::MatrixXd best_augmented_projector = Eigen::MatrixXd::Zero(
      degrees_of_freedom, degrees_of_freedom + additional_constraints);
  Eigen::MatrixXd inverse_saturated_constraints_projected;
  Eigen::MatrixXd inverse_objective_jacobian_projected;

  auto solver_status = SolverStatus::kSuccess;
  Eigen::VectorXd final_check_min_bounds = min_bounds;
  Eigen::VectorXd final_check_max_bounds = max_bounds;

  // Process objectives hierarchically with constraint enforcement
  for (auto objective_index = 0U; objective_index < num_objectives;
       ++objective_index) {
    const ObjectiveSolveConfig objective_config =
        (objective_index < objective_configs.size())
            ? objective_configs[objective_index]
            : ObjectiveSolveConfig{};
    bool objective_is_min_error =
        (objective_config.solve_mode == TaskSolveMode::kMinError);
    objective_effective_modes[objective_index] = objective_config.solve_mode;
    const auto &current_jacobian = objective_jacobians[objective_index];
    const auto &current_target = objective_targets[objective_index];
    const auto target_dimension = current_jacobian.rows();
    Eigen::VectorXd objective_min_bounds_storage = min_bounds;
    Eigen::VectorXd objective_max_bounds_storage = max_bounds;
    Eigen::VectorXd objective_min_bounds_original = min_bounds;
    Eigen::VectorXd objective_max_bounds_original = max_bounds;
    const Eigen::VectorXd *active_min_bounds = &min_bounds;
    const Eigen::VectorXd *active_max_bounds = &max_bounds;
    if (enhanced_mode) {
      active_min_bounds = &objective_min_bounds_storage;
      active_max_bounds = &objective_max_bounds_storage;
    }

    // Store current state for iterative refinement
    auto previous_null_space = null_space_projector;
    previous_velocity = velocity_solution;

    // Initialize iteration state for current objective
    auto constrained_projector = previous_null_space;
    auto velocity_scale = 1.0;
    auto optimal_scale = 0.0;
    auto consecutive_zero_scales = 0U;
    auto consecutive_min_error_no_progress = 0U;
    auto constraints_violated = true;
    auto iteration_counter = 0U;
    auto best_constraint_selection = saturated_constraint_selector;
    best_saturated_values = saturated_values;
    Eigen::MatrixXd optimal_constrained_projector = Eigen::MatrixXd::Zero(
        constrained_projector.rows(), constrained_projector.cols());
    best_augmented_projector.setZero();
    if (current_target.template lpNorm<1>() <
        solver_config.precision_threshold) {
      velocity_solution = previous_velocity;
      constraints_violated = false;
    }

    // Pre-compute transformation matrices for efficiency
    double task_condition_number = 1.0;
    Eigen::MatrixXd damped_inverse_projected_jacobian =
        linalg::ComputeRegularizedInverse(solver_config.regularization_config,
                                          current_jacobian *
                                              constrained_projector,
                                          &task_condition_number);
    worst_condition_number =
        std::max(worst_condition_number, task_condition_number);
    Eigen::MatrixXd jacobian_velocity_product =
        current_jacobian * previous_velocity;
    Eigen::MatrixXd saturated_constraints_on_previous_space(
        degrees_of_freedom + additional_constraints, degrees_of_freedom);
    Eigen::MatrixXd saturated_constraints_velocity(
        degrees_of_freedom + additional_constraints, 1);
    if (saturated_constraint_selector.isZero()) {
      saturated_constraints_on_previous_space = Eigen::MatrixXd::Zero(
          degrees_of_freedom + additional_constraints, degrees_of_freedom);
      saturated_constraints_velocity =
          Eigen::MatrixXd::Zero(degrees_of_freedom + additional_constraints, 1);
    } else {
      saturated_constraints_on_previous_space =
          saturated_constraint_matrix * previous_null_space;
      saturated_constraints_velocity =
          saturated_constraint_matrix * previous_velocity;
    }

    // Iterative constraint satisfaction with hierarchical optimization
    while (constraints_violated) {
      constraints_violated = false;

      // Compute augmented projection operator
      detail::ComputeGeneralizedInverse(
          saturated_constraints_on_previous_space, solver_config.epsilon,
          &inverse_saturated_constraints_projected);
      augmented_projector.noalias() =
          (Eigen::MatrixXd::Identity(damped_inverse_projected_jacobian.rows(),
                                     current_jacobian.cols()) -
           damped_inverse_projected_jacobian * current_jacobian) *
          inverse_saturated_constraints_projected;

      // Calculate unscaled velocity solution
      velocity_solution.noalias() =
          previous_velocity +
          damped_inverse_projected_jacobian *
              (current_target - jacobian_velocity_product) +
          augmented_projector *
              (saturated_values - saturated_constraints_velocity);

      // Evaluate constraint satisfaction
      constraint_evaluation.noalias() =
          constraint_coefficients * velocity_solution;
      constraints_violated = (constraint_evaluation.array() <
                              (active_min_bounds->array() -
                               solver_config.epsilon))
                                 .any() ||
                             (constraint_evaluation.array() >
                              (active_max_bounds->array() +
                               solver_config.epsilon))
                                 .any();

      Eigen::Index critical_constraint_index = 0;
      if (objective_is_min_error) {
        velocity_scale = 1.0;
        if (constraints_violated) {
          Eigen::VectorXd lower_violation =
              (active_min_bounds->array() - constraint_evaluation.array())
                  .max(0.0);
          Eigen::VectorXd upper_violation =
              (constraint_evaluation.array() - active_max_bounds->array())
                  .max(0.0);
          Eigen::VectorXd total_violation = lower_violation + upper_violation;
          total_violation.maxCoeff(&critical_constraint_index);
        }
      } else {
        // Decompose constraint space velocity contributions
        scaled_velocity_contribution.noalias() =
            constraint_coefficients * damped_inverse_projected_jacobian *
            current_target;
        unscaled_contribution.noalias() =
            constraint_evaluation - scaled_velocity_contribution;

        const auto contribution_magnitude = scaled_velocity_contribution.norm();
        if (contribution_magnitude < solver_config.epsilon) {
          velocity_scale = 1.0;
        } else if (contribution_magnitude > solver_config.magnitude_limit) {
          velocity_scale = 0.0;
          solver_status = SolverStatus::kNumericalError;
        } else {
          min_margin = *active_min_bounds - unscaled_contribution;
          max_margin = *active_max_bounds - unscaled_contribution;

          for (auto constraint_idx = 0U;
               constraint_idx < degrees_of_freedom + additional_constraints;
               ++constraint_idx) {
            if (saturated_constraint_selector(constraint_idx, constraint_idx) ==
                1) {
              feasible_scales[constraint_idx] =
                  std::numeric_limits<double>::infinity();
            } else {
              feasible_scales[constraint_idx] =
                  linalg::ComputeFeasibleScalingRange(
                      min_margin[constraint_idx], max_margin[constraint_idx],
                      scaled_velocity_contribution[constraint_idx])
                      .first;
            }
          }

          velocity_scale = feasible_scales.minCoeff(&critical_constraint_index);
        }
      }

      if (velocity_scale == std::numeric_limits<double>::infinity() ||
          velocity_scale == -std::numeric_limits<double>::infinity()) {
        velocity_scale = 0.0;
      }

      if (!objective_is_min_error && velocity_scale <= solver_config.epsilon &&
          objective_config.allow_min_error_fallback) {
        objective_is_min_error = true;
        objective_effective_modes[objective_index] = TaskSolveMode::kMinError;
        objective_used_fallback[objective_index] = true;
        velocity_scale = 1.0;
      }

      if (enhanced_mode && !objective_is_min_error && velocity_scale > 0.0 &&
          velocity_scale < solver_config.epsilon) {
        velocity_scale = solver_config.epsilon;
      }
      if (velocity_scale == 0) {
        consecutive_zero_scales++;
      } else {
        consecutive_zero_scales = 0; // Reset counter for positive scales
      }

      // Process feasible solutions within iteration limits
      if ((objective_index == 0 || velocity_scale > 0 || objective_is_min_error) &&
          iteration_counter < solver_config.iteration_limit) {
        if (enhanced_mode && !objective_is_min_error) {
          const unsigned int soft_trigger = std::max(
              1U, solver_config.stall_detection_count / 2U);
          if (consecutive_zero_scales >= soft_trigger) {
            const double relax_step =
                1.0 + 0.5 * static_cast<double>(consecutive_zero_scales - soft_trigger + 1U);
            for (Eigen::Index i = 0; i < objective_min_bounds_storage.size();
                 ++i) {
              const double max_relax = softening_factors(i);
              if (max_relax <= 1.0) {
                continue;
              }
              const double half_span = 0.5 *
                                       (objective_max_bounds_original(i) -
                                        objective_min_bounds_original(i));
              if (!std::isfinite(half_span) || half_span <= 0.0) {
                continue;
              }
              const double center = 0.5 *
                                    (objective_max_bounds_original(i) +
                                     objective_min_bounds_original(i));
              const double relax = std::min(max_relax, relax_step);
              objective_min_bounds_storage(i) = center - relax * half_span;
              objective_max_bounds_storage(i) = center + relax * half_span;
            }
          }
        }
        const Eigen::VectorXd target_term =
            objective_is_min_error ? current_target
                                   : (velocity_scale * current_target);
        constraint_evaluation.noalias() =
            constraint_coefficients *
            (previous_velocity +
             damped_inverse_projected_jacobian *
                 (target_term - jacobian_velocity_product) +
             augmented_projector *
                 (saturated_values - saturated_constraints_velocity));

        // Identify constraint violations using boolean mask
        using ConstraintViolationMask = Eigen::Array<bool, Eigen::Dynamic, 1>;
        ConstraintViolationMask constraint_violations =
            (constraint_evaluation.array() <
             (active_min_bounds->array() - solver_config.epsilon)) ||
            (constraint_evaluation.array() >
             (active_max_bounds->array() + solver_config.epsilon));

        // Fast path: if all constraints are satisfied, accept this objective
        // update directly and stop saturating rows for this objective.
        if (!constraint_violations.any()) {
          velocity_solution.noalias() =
              previous_velocity +
              damped_inverse_projected_jacobian *
                  (target_term - jacobian_velocity_product) +
              augmented_projector *
                  (saturated_values - saturated_constraints_velocity);
          optimal_scale = velocity_scale;
          best_constraint_selection = saturated_constraint_selector;
          best_saturated_values = saturated_values;
          optimal_constrained_projector = constrained_projector;
          best_augmented_projector = augmented_projector;
          constraints_violated = false;
        } else {
          bool made_saturation_progress = false;
          // Update saturated constraints based on violation patterns
          if (!objective_is_min_error && velocity_scale == 1.0 &&
              constraint_violations.any() &&
              !constraint_violations(critical_constraint_index)) {
            // Full scale with violations excluding critical constraint

            // Saturate all violating constraints
            for (int constraint_idx = 0;
                 constraint_idx < constraint_violations.size();
                 ++constraint_idx) {
              if (constraint_violations(constraint_idx)) {
                // Mark constraint as saturated
                if (saturated_constraint_selector(constraint_idx,
                                                  constraint_idx) == 0) {
                  made_saturation_progress = true;
                  saturated_constraint_selector(constraint_idx, constraint_idx) =
                      1;
                }

                // Clamp constraint value to feasible bounds
                saturated_values(constraint_idx, 0) =
                    std::min(std::max((*active_min_bounds)(constraint_idx),
                                      constraint_evaluation(constraint_idx)),
                             (*active_max_bounds)(constraint_idx));
              }
            }
          } else {
            // Saturate only the critical constraint

            // Mark critical constraint as saturated
            if (saturated_constraint_selector(critical_constraint_index,
                                              critical_constraint_index) == 0) {
              made_saturation_progress = true;
            }
            saturated_constraint_selector(critical_constraint_index,
                                          critical_constraint_index) = 1;

            // Set saturated value for critical constraint
            saturated_values(critical_constraint_index, 0) = std::min(
                std::max((*active_min_bounds)(critical_constraint_index),
                         constraint_evaluation(critical_constraint_index)),
                (*active_max_bounds)(critical_constraint_index));
          }

          // Update constraint system with newly saturated constraints
          saturated_constraint_matrix.noalias() =
              saturated_constraint_selector * constraint_coefficients;
          saturated_constraints_on_previous_space.noalias() =
              saturated_constraint_matrix * previous_null_space;

          detail::ComputeGeneralizedInverse(
              saturated_constraints_on_previous_space, solver_config.epsilon,
              &inverse_saturated_constraints_projected);
          constrained_projector.noalias() =
              previous_null_space - inverse_saturated_constraints_projected *
                                        saturated_constraints_on_previous_space;
          auto effective_rank = 0L;
          if (constrained_projector.colwise().template lpNorm<1>().maxCoeff() >=
              solver_config.precision_threshold) {
            Eigen::ColPivHouseholderQR<Eigen::MatrixXd> rank_analysis(
                current_jacobian * constrained_projector);
            rank_analysis.setThreshold(solver_config.epsilon);
            effective_rank = rank_analysis.rank();
          }

          // Termination criteria: objective redundancy exhausted or no progress
          // detected The algorithm saturates constraints until the effective
          // rank drops below target dimension, indicating all available degrees
          // of freedom have been utilized.
          bool should_terminate = false;
          if (objective_is_min_error) {
            // MIN_ERROR: keep refining active-set until constraints are
            // satisfied or no new active constraints can be added.
            if (made_saturation_progress) {
              consecutive_min_error_no_progress = 0;
            } else {
              consecutive_min_error_no_progress++;
            }
            should_terminate = (consecutive_min_error_no_progress >
                                solver_config.stall_detection_count);
          } else {
            should_terminate =
                effective_rank < target_dimension; // redundancy exhausted
            should_terminate =
                should_terminate ||
                (consecutive_zero_scales >
                 solver_config.stall_detection_count); // stalled progress
          }
          if (should_terminate) {
            bool use_direct_constrained_solve = true;
            if (objective_is_min_error) {
              if (optimal_scale > solver_config.epsilon) {
                velocity_scale = optimal_scale;
                saturated_constraint_selector = best_constraint_selection;
                saturated_constraint_matrix.noalias() =
                    saturated_constraint_selector * constraint_coefficients;
                saturated_values = best_saturated_values;
                constrained_projector = optimal_constrained_projector;
                augmented_projector = best_augmented_projector;
              } else if (constraint_violations.any() &&
                         !made_saturation_progress) {
                // MIN_ERROR active-set can stall if no new rows can be
                // saturated. Prefer a direct constrained least-squares step
                // with the current saturated set to preserve progress.
                velocity_scale = 1.0;
              } else {
                // No feasible snapshot yet, but active-set still evolving:
                // solve constrained least-squares with current saturated set.
                velocity_scale = 1.0;
              }
            } else {
              velocity_scale = optimal_scale;
              saturated_constraint_selector = best_constraint_selection;
              saturated_constraint_matrix.noalias() =
                  saturated_constraint_selector * constraint_coefficients;
              saturated_values = best_saturated_values;
              constrained_projector = optimal_constrained_projector;
              augmented_projector = best_augmented_projector;
            }

            if (use_direct_constrained_solve) {
              const Eigen::VectorXd target_term_final =
                  objective_is_min_error ? current_target
                                         : (velocity_scale * current_target);
              double cond_tmp = 1.0;
              auto adaptive_regularization = solver_config.regularization_config;
              if (!objective_is_min_error && target_dimension > 0 &&
                  effective_rank > 0 && effective_rank < target_dimension) {
                const double rank_ratio =
                    static_cast<double>(target_dimension) /
                    static_cast<double>(effective_rank);
                adaptive_regularization.regularization_factor *= rank_ratio;
              }
              velocity_solution.noalias() =
                  previous_velocity +
                  linalg::ComputeRegularizedInverse(
                      adaptive_regularization,
                      current_jacobian * constrained_projector,
                      &cond_tmp) *
                      (target_term_final - jacobian_velocity_product) +
                  augmented_projector *
                      (saturated_values -
                       saturated_constraint_matrix * previous_velocity);
              worst_condition_number =
                  std::max(worst_condition_number, cond_tmp);
            }
            constraints_violated = false;
          }
        }

        // Maximum iteration safeguard - typically indicates numerical issues
        if (iteration_counter == solver_config.iteration_limit) {
          solver_status = SolverStatus::kNumericalError;
          std::vector<double> zero_solution(
              static_cast<size_t>(degrees_of_freedom), 0.0);
          std::vector<double> zero_scales(static_cast<size_t>(num_objectives),
                                          0.0);
          return SolverResult{
              std::move(zero_solution),
              solver_status,
              std::chrono::duration_cast<std::chrono::microseconds>(
                  std::chrono::high_resolution_clock::now() - t0)
                      .count() /
                  1000.0,
              iteration_counter,
              0.0,
              std::move(zero_scales),
              {},
              {},
              {},
              "iteration limit reached"};
        }
      } else {
        velocity_scale = 0.0;
        velocity_solution = previous_velocity;
        constraints_violated = false;
        if (iteration_counter == solver_config.iteration_limit) {
          solver_status = SolverStatus::kNumericalError;
          std::vector<double> zero_solution(
              static_cast<size_t>(degrees_of_freedom), 0.0);
          std::vector<double> zero_scales(static_cast<size_t>(num_objectives),
                                          0.0);
          return SolverResult{
              std::move(zero_solution),
              solver_status,
              std::chrono::duration_cast<std::chrono::microseconds>(
                  std::chrono::high_resolution_clock::now() - t0)
                      .count() /
                  1000.0,
              iteration_counter,
              0.0,
              std::move(zero_scales),
              {},
              {},
              {},
              "iteration limit reached"};
        }
      }

      ++iteration_counter;
      if (velocity_scale > 0.0 || objective_is_min_error) {
        detail::ComputeGeneralizedInverse(
            current_jacobian * previous_null_space, solver_config.epsilon,
            &inverse_objective_jacobian_projected);
        null_space_projector.noalias() =
            previous_null_space - inverse_objective_jacobian_projected *
                                      current_jacobian * previous_null_space;

        // Threshold small values to zero for numerical stability
        null_space_projector = (null_space_projector.array().abs() <
                                solver_config.precision_threshold)
                                   .select(0.0, null_space_projector);
      }

      // Refresh cached computations for next iteration
      {
        double cond_tmp = 1.0;
        damped_inverse_projected_jacobian = linalg::ComputeRegularizedInverse(
            solver_config.regularization_config,
            current_jacobian * constrained_projector, &cond_tmp);
        worst_condition_number =
            std::max(worst_condition_number, cond_tmp);
      }
      saturated_constraints_on_previous_space.noalias() =
          saturated_constraint_matrix * previous_null_space;
      saturated_constraints_velocity.noalias() =
          saturated_constraint_matrix * previous_velocity;
    }

    objective_scaling_factors[objective_index] =
        objective_is_min_error ? -1.0 : velocity_scale;
    objective_effective_modes[objective_index] =
        objective_is_min_error ? TaskSolveMode::kMinError
                               : TaskSolveMode::kScale;
    if (enhanced_mode) {
      final_check_min_bounds = objective_min_bounds_storage;
      final_check_max_bounds = objective_max_bounds_storage;
    } else {
      final_check_min_bounds = min_bounds;
      final_check_max_bounds = max_bounds;
    }
  }

  // Verify final solution satisfies all constraints within tolerance
  if (objective_scaling_factors.sum() > solver_config.epsilon &&
      (((constraint_coefficients * velocity_solution).array() <
        (final_check_min_bounds.array() - solver_config.epsilon))
           .any() ||
       ((constraint_coefficients * velocity_solution).array() >
        (final_check_max_bounds.array() + solver_config.epsilon))
           .any())) {
    solver_status = SolverStatus::kNumericalError;
  }

  const bool generated_non_finite =
      !objective_scaling_factors.allFinite() || !velocity_solution.allFinite();
  if (generated_non_finite) {
    solver_status = SolverStatus::kNumericalError;
    objective_scaling_factors.setZero();
    velocity_solution.setZero();
  }

  // Package results for output
  std::vector<double> final_velocities(static_cast<size_t>(degrees_of_freedom));
  std::vector<double> applied_scales(static_cast<size_t>(num_objectives));
  std::vector<double> objective_errors(static_cast<size_t>(num_objectives), 0.0);
  for (Eigen::Index dof_idx = 0; dof_idx < degrees_of_freedom; ++dof_idx) {
    final_velocities[static_cast<size_t>(dof_idx)] = velocity_solution(dof_idx);
  }
  for (Eigen::Index obj_idx = 0;
       obj_idx < static_cast<Eigen::Index>(num_objectives); ++obj_idx) {
    applied_scales[static_cast<size_t>(obj_idx)] =
        objective_scaling_factors(obj_idx);
    objective_errors[static_cast<size_t>(obj_idx)] =
        (objective_jacobians[static_cast<size_t>(obj_idx)] * velocity_solution -
         objective_targets[static_cast<size_t>(obj_idx)])
            .norm();
  }

  auto end_time = std::chrono::high_resolution_clock::now();
  const double elapsed_milliseconds =
      std::chrono::duration_cast<std::chrono::microseconds>(end_time - t0)
          .count() /
      1000.0;

  return SolverResult{
      std::move(final_velocities),
      solver_status,
      elapsed_milliseconds,
      static_cast<unsigned int>(
          num_objectives), // number of objectives processed
      0.0,                 // error metric computed externally if required
      std::move(applied_scales),
      std::move(objective_errors),
      std::move(objective_effective_modes),
      std::move(objective_used_fallback),
      (solver_status == SolverStatus::kSuccess)
          ? ""
          : (solver_status == SolverStatus::kNumericalError
                 ? (generated_non_finite
                        ? "non-finite values generated in solver state"
                        : "numerical constraint violation beyond epsilon after solve")
                 : (solver_status == SolverStatus::kNonFiniteInput
                        ? "non-finite values detected in solver state"
                        : "solver failed with input/status error")),
      worst_condition_number};
}

// No additional aliases defined

} // namespace embodik
