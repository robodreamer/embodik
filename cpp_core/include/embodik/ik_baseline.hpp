/*
 * MIT License
 *
 * Copyright (c) 2025 Andy Park <andypark.purdue@gmail.com>
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

#pragma once

#include "types.hpp"

#include <chrono>
#include <cmath>
#include <limits>
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
    const Eigen::MatrixXd &input_matrix) {
  using MatrixDouble = Eigen::MatrixXd;
  using SquareMatrix = Eigen::MatrixXd; // square matrix for gram computation

  SquareMatrix gram_matrix = input_matrix * input_matrix.transpose();

  const double threshold_squared = std::pow(regularization_config.epsilon, 2.0);
  const double det_value = gram_matrix.determinant();

  const double regularization =
      (det_value < threshold_squared)
          ? (1.0 - std::pow(det_value / threshold_squared, 2.0)) *
                threshold_squared
          : 0.0;

  SquareMatrix regularized_gram = gram_matrix;
  regularized_gram.diagonal().array() += regularization;

  // Singular value decomposition for damping computation
  const Eigen::BDCSVD<MatrixDouble> svd_decomp(input_matrix,
                                               Eigen::ComputeFullU);
  const auto &sigma_values = svd_decomp.singularValues();
  const auto &left_singular_vectors = svd_decomp.matrixU();

  if (sigma_values.size() > 0) {
    Eigen::Array<bool, Eigen::Dynamic, 1> small_values =
        sigma_values.array() < regularization_config.epsilon;
    if (small_values.any()) {
      regularized_gram.noalias() +=
          left_singular_vectors *
          (((regularization_config.regularization_factor *
             (1.0 - (sigma_values.array() / regularization_config.epsilon)
                        .square())) *
            small_values.cast<double>())
               .matrix()
               .asDiagonal()) *
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
    const VelocitySolverConfig &solver_config);

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

  // Validate input dimensions
  if (target_velocities.size() != task_jacobians.size())
    return SolverResult{
        {}, SolverStatus::kInvalidInput, get_elapsed_ms(), 0, 0.0, {}, {}};
  if (constraint_matrix.empty())
    return SolverResult{
        {}, SolverStatus::kInvalidInput, get_elapsed_ms(), 0, 0.0, {}, {}};
  if (target_velocities.empty())
    return SolverResult{
        {}, SolverStatus::kInvalidInput, get_elapsed_ms(), 0, 0.0, {}, {}};

  const size_t num_dof = constraint_matrix[0].size();
  const size_t objective_count = target_velocities.size();

  // Verify matrix dimensions consistency
  for (size_t obj_idx = 0; obj_idx < objective_count; ++obj_idx) {
    if (task_jacobians[obj_idx].empty() ||
        task_jacobians[obj_idx][0].size() != num_dof) {
      return SolverResult{
          {}, SolverStatus::kInvalidInput, get_elapsed_ms(), 0, 0.0, {}, {}};
    }
    if (target_velocities[obj_idx].size() != task_jacobians[obj_idx].size()) {
      return SolverResult{
          {}, SolverStatus::kInvalidInput, get_elapsed_ms(), 0, 0.0, {}, {}};
    }
  }

  if (constraint_matrix.size() != min_bounds.size() ||
      constraint_matrix.size() != max_bounds.size()) {
    return SolverResult{
        {}, SolverStatus::kInvalidInput, get_elapsed_ms(), 0, 0.0, {}, {}};
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
        return SolverResult{
            {}, SolverStatus::kInvalidInput, get_elapsed_ms(), 0, 0.0, {}, {}};
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
      return SolverResult{
          {}, SolverStatus::kInvalidInput, get_elapsed_ms(), 0, 0.0, {}, {}};
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
      eigen_max_bounds, solver_config);
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
    const VelocitySolverConfig &solver_config = VelocitySolverConfig{}) {
  auto t0 = std::chrono::high_resolution_clock::now();
  auto get_elapsed_ms = [&t0]() {
    return std::chrono::duration_cast<std::chrono::microseconds>(
               std::chrono::high_resolution_clock::now() - t0)
               .count() /
           1000.0;
  };

  // Validate input consistency
  if (objective_targets.size() != objective_jacobians.size())
    return SolverResult{
        {}, SolverStatus::kInvalidInput, get_elapsed_ms(), 0, 0.0, {}, {}};
  if (constraint_coefficients.cols() == 0 ||
      constraint_coefficients.rows() == 0)
    return SolverResult{
        {}, SolverStatus::kInvalidInput, get_elapsed_ms(), 0, 0.0, {}, {}};
  if (min_bounds.size() != constraint_coefficients.rows() ||
      max_bounds.size() != constraint_coefficients.rows())
    return SolverResult{
        {}, SolverStatus::kInvalidInput, get_elapsed_ms(), 0, 0.0, {}, {}};
  if (objective_targets.empty())
    return SolverResult{
        {}, SolverStatus::kInvalidInput, get_elapsed_ms(), 0, 0.0, {}, {}};

  // Extract system dimensions
  const auto degrees_of_freedom = constraint_coefficients.cols();
  const auto additional_constraints =
      constraint_coefficients.rows() - degrees_of_freedom;
  const auto num_objectives = objective_jacobians.size();
  Eigen::VectorXd objective_scaling_factors =
      Eigen::VectorXd::Zero(num_objectives);

  // Verify objective dimension compatibility
  for (size_t obj_idx = 0; obj_idx < num_objectives; ++obj_idx) {
    if (objective_jacobians[obj_idx].cols() != degrees_of_freedom ||
        objective_jacobians[obj_idx].rows() !=
            objective_targets[obj_idx].rows()) {
      return SolverResult{
          {}, SolverStatus::kInvalidInput, get_elapsed_ms(), 0, 0.0, {}, {}};
    }
  }

  // Initialize workspace matrices for null-space computation
  Eigen::MatrixXd null_space_projector =
      Eigen::MatrixXd::Identity(degrees_of_freedom, degrees_of_freedom);
  Eigen::MatrixXd saturated_constraint_selector =
      Eigen::MatrixXd::Zero(degrees_of_freedom + additional_constraints,
                            degrees_of_freedom + additional_constraints);
  Eigen::VectorXd velocity_solution = Eigen::VectorXd::Zero(degrees_of_freedom);
  Eigen::VectorXd saturated_values =
      Eigen::VectorXd::Zero(degrees_of_freedom + additional_constraints);
  Eigen::MatrixXd saturated_constraint_matrix = Eigen::MatrixXd::Zero(
      degrees_of_freedom + additional_constraints, degrees_of_freedom);

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

  // Process objectives hierarchically with constraint enforcement
  for (auto objective_index = 0U; objective_index < num_objectives;
       ++objective_index) {
    const auto &current_jacobian = objective_jacobians[objective_index];
    const auto &current_target = objective_targets[objective_index];
    const auto target_dimension = current_jacobian.rows();

    // Store current state for iterative refinement
    auto previous_null_space = null_space_projector;
    previous_velocity = velocity_solution;

    // Initialize iteration state for current objective
    auto constrained_projector = previous_null_space;
    auto velocity_scale = 1.0;
    auto optimal_scale = 0.0;
    auto consecutive_zero_scales = 0U;
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
    Eigen::MatrixXd damped_inverse_projected_jacobian =
        linalg::ComputeRegularizedInverse(solver_config.regularization_config,
                                          current_jacobian *
                                              constrained_projector);
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
                              (min_bounds.array() - solver_config.epsilon))
                                 .any() ||
                             (constraint_evaluation.array() >
                              (max_bounds.array() + solver_config.epsilon))
                                 .any();

      // Decompose constraint space velocity contributions
      scaled_velocity_contribution.noalias() =
          constraint_coefficients * damped_inverse_projected_jacobian *
          current_target;
      unscaled_contribution.noalias() =
          constraint_evaluation - scaled_velocity_contribution;

      const auto contribution_magnitude = scaled_velocity_contribution.norm();
      Eigen::Index critical_constraint_index = 0;
      if (contribution_magnitude < solver_config.epsilon) {
        velocity_scale = 1.0;
      } else if (contribution_magnitude > solver_config.magnitude_limit) {
        velocity_scale = 0.0;
        solver_status = SolverStatus::kNumericalError;
      } else {
        min_margin = min_bounds - unscaled_contribution;
        max_margin = max_bounds - unscaled_contribution;

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

      if (velocity_scale == std::numeric_limits<double>::infinity() ||
          velocity_scale == -std::numeric_limits<double>::infinity()) {
        velocity_scale = 0.0;
      }

      if (velocity_scale == 0) {
        consecutive_zero_scales++;
      } else {
        consecutive_zero_scales = 0; // Reset counter for positive scales
      }

      // Process feasible solutions within iteration limits
      if ((objective_index == 0 || velocity_scale > 0) &&
          iteration_counter < solver_config.iteration_limit) {
        constraint_evaluation.noalias() =
            constraint_coefficients *
            (previous_velocity +
             damped_inverse_projected_jacobian *
                 (velocity_scale * current_target - jacobian_velocity_product) +
             augmented_projector *
                 (saturated_values - saturated_constraints_velocity));

        // Identify constraint violations using boolean mask
        using ConstraintViolationMask = Eigen::Array<bool, Eigen::Dynamic, 1>;
        ConstraintViolationMask constraint_violations =
            (constraint_evaluation.array() <
             (min_bounds.array() - solver_config.epsilon)) ||
            (constraint_evaluation.array() >
             (max_bounds.array() + solver_config.epsilon));

        // Track optimal solution configuration
        if (velocity_scale > optimal_scale && !constraint_violations.any()) {
          optimal_scale = velocity_scale;
          best_constraint_selection = saturated_constraint_selector;
          best_saturated_values = saturated_values;
          optimal_constrained_projector = constrained_projector;
          best_augmented_projector = augmented_projector;
        }

        // Update saturated constraints based on violation patterns
        if (velocity_scale == 1.0 && constraint_violations.any() &&
            !constraint_violations(critical_constraint_index)) {
          // Full scale with violations excluding critical constraint

          // Saturate all violating constraints
          for (int constraint_idx = 0;
               constraint_idx < constraint_violations.size();
               ++constraint_idx) {
            if (constraint_violations(constraint_idx)) {
              // Mark constraint as saturated
              saturated_constraint_selector(constraint_idx, constraint_idx) = 1;

              // Clamp constraint value to feasible bounds
              saturated_values(constraint_idx, 0) =
                  std::min(std::max(min_bounds(constraint_idx),
                                    constraint_evaluation(constraint_idx)),
                           max_bounds(constraint_idx));
            }
          }
        } else {
          // Saturate only the critical constraint

          // Mark critical constraint as saturated
          saturated_constraint_selector(critical_constraint_index,
                                        critical_constraint_index) = 1;

          // Set saturated value for critical constraint
          saturated_values(critical_constraint_index, 0) = std::min(
              std::max(min_bounds(critical_constraint_index),
                       constraint_evaluation(critical_constraint_index)),
              max_bounds(critical_constraint_index));
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
        // detected The algorithm saturates constraints until the effective rank
        // drops below target dimension, indicating all available degrees of
        // freedom have been utilized.
        bool should_terminate =
            effective_rank < target_dimension; // redundancy exhausted
        should_terminate =
            should_terminate ||
            (consecutive_zero_scales >
             solver_config.stall_detection_count); // stalled progress
        if (should_terminate) {
          velocity_scale = optimal_scale;
          saturated_constraint_selector = best_constraint_selection;
          saturated_constraint_matrix.noalias() =
              saturated_constraint_selector * constraint_coefficients;
          saturated_values = best_saturated_values;
          constrained_projector = optimal_constrained_projector;
          augmented_projector = best_augmented_projector;

          velocity_solution.noalias() =
              previous_velocity +
              linalg::ComputeRegularizedInverse(
                  solver_config.regularization_config,
                  current_jacobian * constrained_projector) *
                  (velocity_scale * current_target -
                   jacobian_velocity_product) +
              augmented_projector *
                  (saturated_values -
                   saturated_constraint_matrix * previous_velocity);
          constraints_violated = false;
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
              {}};
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
              {}};
        }
      }

      ++iteration_counter;
      if (velocity_scale > 0.0) {
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
      damped_inverse_projected_jacobian = linalg::ComputeRegularizedInverse(
          solver_config.regularization_config,
          current_jacobian * constrained_projector);
      saturated_constraints_on_previous_space.noalias() =
          saturated_constraint_matrix * previous_null_space;
      saturated_constraints_velocity.noalias() =
          saturated_constraint_matrix * previous_velocity;
    }

    objective_scaling_factors[objective_index] = velocity_scale;
  }

  // Verify final solution satisfies all constraints within tolerance
  if (objective_scaling_factors.sum() > solver_config.epsilon &&
      (((constraint_coefficients * velocity_solution).array() <
        (min_bounds.array() - solver_config.epsilon))
           .any() ||
       ((constraint_coefficients * velocity_solution).array() >
        (max_bounds.array() + solver_config.epsilon))
           .any())) {
    solver_status = SolverStatus::kNumericalError;
  }

  if (objective_scaling_factors.hasNaN() || velocity_solution.hasNaN()) {
    solver_status = SolverStatus::kInvalidInput;
    objective_scaling_factors.setZero();
    velocity_solution.setZero();
  }

  // Package results for output
  std::vector<double> final_velocities(static_cast<size_t>(degrees_of_freedom));
  std::vector<double> applied_scales(static_cast<size_t>(num_objectives));
  for (Eigen::Index dof_idx = 0; dof_idx < degrees_of_freedom; ++dof_idx) {
    final_velocities[static_cast<size_t>(dof_idx)] = velocity_solution(dof_idx);
  }
  for (Eigen::Index obj_idx = 0;
       obj_idx < static_cast<Eigen::Index>(num_objectives); ++obj_idx) {
    applied_scales[static_cast<size_t>(obj_idx)] =
        objective_scaling_factors(obj_idx);
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
      {} // task_errors
  };
}

// No additional aliases defined

} // namespace embodik
