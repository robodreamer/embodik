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

/// @brief Legacy velocity backend coefficient guard.
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

inline double ComputeMaxLinearConstraintViolation(
    const Eigen::MatrixXd &constraint_coefficients,
    const Eigen::VectorXd &min_bounds, const Eigen::VectorXd &max_bounds,
    const Eigen::VectorXd &candidate) {
  double max_violation = 0.0;
  for (Eigen::Index row = 0; row < constraint_coefficients.rows(); ++row) {
    const double value = constraint_coefficients.row(row).dot(candidate);
    max_violation =
        std::max(max_violation, min_bounds(row) - value);
    max_violation =
        std::max(max_violation, value - max_bounds(row));
  }
  return std::max(0.0, max_violation);
}

inline void NormalizeLinearConstraintRows(
    const Eigen::MatrixXd &constraint_coefficients,
    const Eigen::VectorXd &min_bounds, const Eigen::VectorXd &max_bounds,
    Eigen::MatrixXd *normalized_coefficients,
    Eigen::VectorXd *normalized_min_bounds,
    Eigen::VectorXd *normalized_max_bounds) {
  *normalized_coefficients = constraint_coefficients;
  *normalized_min_bounds = min_bounds;
  *normalized_max_bounds = max_bounds;
  if (min_bounds.size() != constraint_coefficients.rows() ||
      max_bounds.size() != constraint_coefficients.rows()) {
    return;
  }
  for (Eigen::Index row = 0; row < constraint_coefficients.rows(); ++row) {
    const double row_norm = constraint_coefficients.row(row).stableNorm();
    if (row_norm == 0.0 || !std::isfinite(row_norm)) {
      continue;
    }
    normalized_coefficients->row(row) /= row_norm;
    (*normalized_min_bounds)(row) /= row_norm;
    (*normalized_max_bounds)(row) /= row_norm;
  }
}

enum class HardConstraintFeasibilityStatus {
  kFeasible,
  kProvenInfeasible,
  kUnknown,
};

enum class HierarchicalLinearSolverPolicy {
  kGeneralizedEsns,
  kLegacyVelocity,
};

struct HierarchicalLinearSolverExecutionOptions {
  HierarchicalLinearSolverPolicy policy =
      HierarchicalLinearSolverPolicy::kGeneralizedEsns;
  double objective_equality_tolerance = 0.0;
};

/**
 * @brief Search for a feasible witness for linear interval constraints.
 *
 * The witness is used only to distinguish a feasible hard-constraint set from
 * simple, provable infeasibility. It must not seed the hierarchical solution:
 * the minimum-norm point of the hard set is not generally the minimum-norm
 * point after task equalities are imposed.
 */
inline HardConstraintFeasibilityStatus FindHardConstraintFeasibleWitness(
    const Eigen::MatrixXd &constraint_coefficients,
    const Eigen::VectorXd &min_bounds, const Eigen::VectorXd &max_bounds,
    const VelocitySolverConfig &solver_config, Eigen::VectorXd *result) {
  const Eigen::Index num_dof = constraint_coefficients.cols();
  const Eigen::Index num_constraints = constraint_coefficients.rows();
  const double tolerance = std::max(1e-12, solver_config.epsilon);

  Eigen::VectorXd candidate = Eigen::VectorXd::Zero(num_dof);
  if (ComputeMaxLinearConstraintViolation(constraint_coefficients, min_bounds,
                                          max_bounds, candidate) <= tolerance) {
    *result = std::move(candidate);
    return HardConstraintFeasibilityStatus::kFeasible;
  }

  // Phase I minimizes the nonnegative maximum normalized violation t:
  //   min 0.5 t^2
  //   s.t. lower_i <= C_i u + t, C_i u - t <= upper_i, t >= 0.
  // The feasible witness is detection-only and is never used to seed eSNS.
  std::vector<Eigen::RowVectorXd> inequality_rows;
  std::vector<double> inequality_bounds;
  inequality_rows.reserve(static_cast<size_t>(2 * num_constraints + 1));
  inequality_bounds.reserve(static_cast<size_t>(2 * num_constraints + 1));

  for (Eigen::Index row = 0; row < num_constraints; ++row) {
    const double row_norm = constraint_coefficients.row(row).norm();
    if (row_norm == 0.0) {
      if (0.0 < min_bounds(row) - tolerance ||
          0.0 > max_bounds(row) + tolerance) {
        return HardConstraintFeasibilityStatus::kProvenInfeasible;
      }
      continue;
    }

    const Eigen::RowVectorXd normalized_row =
        constraint_coefficients.row(row) / row_norm;
    const double normalized_lower = min_bounds(row) / row_norm;
    const double normalized_upper = max_bounds(row) / row_norm;
    if (!normalized_row.allFinite() || !std::isfinite(normalized_lower) ||
        !std::isfinite(normalized_upper)) {
      return HardConstraintFeasibilityStatus::kUnknown;
    }

    Eigen::RowVectorXd upper_row(num_dof + 1);
    upper_row.head(num_dof) = normalized_row;
    upper_row(num_dof) = -1.0;
    inequality_rows.push_back(std::move(upper_row));
    inequality_bounds.push_back(normalized_upper);

    Eigen::RowVectorXd lower_row(num_dof + 1);
    lower_row.head(num_dof) = -normalized_row;
    lower_row(num_dof) = -1.0;
    inequality_rows.push_back(std::move(lower_row));
    inequality_bounds.push_back(-normalized_lower);
  }

  Eigen::RowVectorXd nonnegative_slack_row =
      Eigen::RowVectorXd::Zero(num_dof + 1);
  nonnegative_slack_row(num_dof) = -1.0;
  inequality_rows.push_back(std::move(nonnegative_slack_row));
  inequality_bounds.push_back(0.0);

  const Eigen::Index num_inequalities =
      static_cast<Eigen::Index>(inequality_rows.size());
  Eigen::MatrixXd inequalities(num_inequalities, num_dof + 1);
  Eigen::VectorXd bounds(num_inequalities);
  for (Eigen::Index row = 0; row < num_inequalities; ++row) {
    inequalities.row(row) = inequality_rows[static_cast<size_t>(row)];
    bounds(row) = inequality_bounds[static_cast<size_t>(row)];
  }

  Eigen::VectorXd phase_point = Eigen::VectorXd::Zero(num_dof + 1);
  double initial_slack = 0.0;
  for (Eigen::Index row = 0; row < num_inequalities - 1; ++row) {
    initial_slack = std::max(initial_slack, -bounds(row));
  }
  if (!std::isfinite(initial_slack)) {
    return HardConstraintFeasibilityStatus::kUnknown;
  }
  const double strict_feasibility_margin = 10.0 * tolerance;
  if (initial_slack >
      std::numeric_limits<double>::max() - strict_feasibility_margin) {
    return HardConstraintFeasibilityStatus::kUnknown;
  }
  phase_point(num_dof) = initial_slack + strict_feasibility_margin;

  std::vector<Eigen::Index> active;
  active.reserve(static_cast<size_t>(num_dof + 1));
  auto add_independent_active_row = [&](Eigen::Index row) {
    if (std::find(active.begin(), active.end(), row) != active.end()) {
      return false;
    }
    Eigen::MatrixXd candidate_active(active.size() + 1, num_dof + 1);
    for (size_t index = 0; index < active.size(); ++index) {
      candidate_active.row(static_cast<Eigen::Index>(index)) =
          inequalities.row(active[index]);
    }
    candidate_active.row(static_cast<Eigen::Index>(active.size())) =
        inequalities.row(row);
    Eigen::ColPivHouseholderQR<Eigen::MatrixXd> rank_analysis(
        candidate_active);
    rank_analysis.setThreshold(tolerance);
    if (rank_analysis.rank() <= static_cast<Eigen::Index>(active.size())) {
      return false;
    }
    active.push_back(row);
    return true;
  };
  const Eigen::VectorXd initial_margins =
      bounds - inequalities * phase_point;
  for (Eigen::Index row = 0; row < num_inequalities; ++row) {
    if (initial_margins(row) <= 10.0 * tolerance) {
      (void)add_independent_active_row(row);
    }
  }

  Eigen::MatrixXd hessian = Eigen::MatrixXd::Zero(num_dof + 1, num_dof + 1);
  hessian(num_dof, num_dof) = 1.0;
  const unsigned int configured_iterations =
      std::max<unsigned int>(
          64U, static_cast<unsigned int>(
                   4 * (num_inequalities + num_dof + 1)));
  const unsigned int max_iterations =
      std::min<unsigned int>(2048U, configured_iterations);

  for (unsigned int iteration = 0; iteration < max_iterations; ++iteration) {
    Eigen::MatrixXd active_matrix(active.size(), num_dof + 1);
    for (size_t index = 0; index < active.size(); ++index) {
      active_matrix.row(static_cast<Eigen::Index>(index)) =
          inequalities.row(active[index]);
    }

    const Eigen::VectorXd gradient = hessian * phase_point;
    Eigen::MatrixXd kkt(num_dof + 1 + active_matrix.rows(),
                        num_dof + 1 + active_matrix.rows());
    kkt.setZero();
    kkt.topLeftCorner(num_dof + 1, num_dof + 1) = hessian;
    if (active_matrix.rows() > 0) {
      kkt.topRightCorner(num_dof + 1, active_matrix.rows()) =
          active_matrix.transpose();
      kkt.bottomLeftCorner(active_matrix.rows(), num_dof + 1) =
          active_matrix;
    }

    Eigen::VectorXd rhs = Eigen::VectorXd::Zero(kkt.rows());
    rhs.head(num_dof + 1) = -gradient;
    const Eigen::VectorXd kkt_solution =
        kkt.completeOrthogonalDecomposition().solve(rhs);
    if (!kkt_solution.allFinite()) {
      return HardConstraintFeasibilityStatus::kUnknown;
    }
    const double kkt_tolerance =
        100.0 * tolerance * std::max(1.0, rhs.norm());
    if ((kkt * kkt_solution - rhs).norm() > kkt_tolerance) {
      return HardConstraintFeasibilityStatus::kUnknown;
    }

    const Eigen::VectorXd direction = kkt_solution.head(num_dof + 1);
    const Eigen::VectorXd multipliers =
        kkt_solution.tail(active_matrix.rows());
    const double direction_tolerance =
        tolerance * std::max(1.0, phase_point.norm());
    if (direction.norm() <= direction_tolerance) {
      Eigen::Index remove_position = -1;
      double minimum_multiplier = 0.0;
      if (multipliers.size() > 0) {
        minimum_multiplier = multipliers.minCoeff(&remove_position);
      }
      if (remove_position >= 0 && minimum_multiplier < -10.0 * tolerance) {
        active.erase(active.begin() + remove_position);
        continue;
      }

      const double max_phase_violation =
          (inequalities * phase_point - bounds).maxCoeff();
      if (max_phase_violation > 10.0 * tolerance) {
        return HardConstraintFeasibilityStatus::kUnknown;
      }
      const double optimal_slack = std::max(0.0, phase_point(num_dof));
      if (optimal_slack <= 10.0 * tolerance) {
        const Eigen::VectorXd original_inequality_values =
            inequalities.topRows(num_inequalities - 1).leftCols(num_dof) *
            phase_point.head(num_dof);
        const double original_normalized_violation =
            (original_inequality_values -
             bounds.head(num_inequalities - 1))
                .maxCoeff();
        if (original_normalized_violation <= 100.0 * tolerance) {
          *result = phase_point.head(num_dof);
          return HardConstraintFeasibilityStatus::kFeasible;
        }
        return HardConstraintFeasibilityStatus::kUnknown;
      }

      Eigen::VectorXd dual = Eigen::VectorXd::Zero(num_inequalities);
      for (size_t index = 0; index < active.size(); ++index) {
        dual(active[index]) = multipliers(static_cast<Eigen::Index>(index));
      }
      if (dual.minCoeff() < -10.0 * tolerance) {
        return HardConstraintFeasibilityStatus::kUnknown;
      }
      const Eigen::MatrixXd original_inequalities =
          inequalities.topRows(num_inequalities - 1).leftCols(num_dof);
      const Eigen::VectorXd original_bounds =
          bounds.head(num_inequalities - 1);
      std::vector<Eigen::Index> certificate_rows;
      certificate_rows.reserve(active.size());
      for (Eigen::Index row : active) {
        if (row < num_inequalities - 1) {
          certificate_rows.push_back(row);
        }
      }
      const Eigen::Index certificate_size =
          static_cast<Eigen::Index>(certificate_rows.size());
      if (certificate_size != num_dof + 1) {
        return HardConstraintFeasibilityStatus::kUnknown;
      }

      Eigen::MatrixXd certificate_system(num_dof + 1, certificate_size);
      for (Eigen::Index column = 0; column < certificate_size; ++column) {
        certificate_system.col(column).head(num_dof) =
            original_inequalities.row(
                certificate_rows[static_cast<size_t>(column)])
                .transpose();
        certificate_system(num_dof, column) = 1.0;
      }
      Eigen::JacobiSVD<Eigen::MatrixXd> certificate_svd(certificate_system);
      const Eigen::VectorXd singular_values = certificate_svd.singularValues();
      if (singular_values.size() == 0 ||
          singular_values(singular_values.size() - 1) <= 0.0) {
        return HardConstraintFeasibilityStatus::kUnknown;
      }
      const double certificate_condition =
          singular_values(0) /
          singular_values(singular_values.size() - 1);
      const double max_certificate_condition =
          1.0 / std::sqrt(std::numeric_limits<double>::epsilon());
      if (!std::isfinite(certificate_condition) ||
          certificate_condition > max_certificate_condition) {
        return HardConstraintFeasibilityStatus::kUnknown;
      }

      const size_t system_size = static_cast<size_t>(num_dof + 1);
      std::vector<std::vector<long double>> augmented_system(
          system_size, std::vector<long double>(system_size + 1, 0.0L));
      for (size_t row = 0; row < system_size; ++row) {
        for (size_t column = 0; column < system_size; ++column) {
          augmented_system[row][column] = static_cast<long double>(
              certificate_system(static_cast<Eigen::Index>(row),
                                 static_cast<Eigen::Index>(column)));
        }
      }
      augmented_system[system_size - 1][system_size] = 1.0L;

      for (size_t pivot_column = 0; pivot_column < system_size;
           ++pivot_column) {
        size_t pivot_row = pivot_column;
        long double pivot_magnitude =
            std::abs(augmented_system[pivot_row][pivot_column]);
        for (size_t row = pivot_column + 1; row < system_size; ++row) {
          const long double candidate_magnitude =
              std::abs(augmented_system[row][pivot_column]);
          if (candidate_magnitude > pivot_magnitude) {
            pivot_magnitude = candidate_magnitude;
            pivot_row = row;
          }
        }
        if (pivot_magnitude == 0.0L) {
          return HardConstraintFeasibilityStatus::kUnknown;
        }
        if (pivot_row != pivot_column) {
          std::swap(augmented_system[pivot_row],
                    augmented_system[pivot_column]);
        }
        for (size_t row = pivot_column + 1; row < system_size; ++row) {
          const long double factor =
              augmented_system[row][pivot_column] /
              augmented_system[pivot_column][pivot_column];
          for (size_t column = pivot_column; column <= system_size; ++column) {
            augmented_system[row][column] -=
                factor * augmented_system[pivot_column][column];
          }
        }
      }

      std::vector<long double> refined_dual(system_size, 0.0L);
      for (size_t reverse_index = system_size; reverse_index-- > 0;) {
        long double rhs_value = augmented_system[reverse_index][system_size];
        for (size_t column = reverse_index + 1; column < system_size;
             ++column) {
          rhs_value -= augmented_system[reverse_index][column] *
                       refined_dual[column];
        }
        refined_dual[reverse_index] =
            rhs_value /
            augmented_system[reverse_index][reverse_index];
      }

      const long double refinement_tolerance =
          64.0L * std::numeric_limits<long double>::epsilon() *
          static_cast<long double>(certificate_condition);
      for (long double &value : refined_dual) {
        if (value < -refinement_tolerance) {
          return HardConstraintFeasibilityStatus::kUnknown;
        }
        if (value < 0.0L) {
          value = 0.0L;
        }
      }

      bool stationarity_is_resolved = true;
      for (Eigen::Index column = 0; column < num_dof; ++column) {
        long double stationarity = 0.0L;
        long double absolute_sum = 0.0L;
        for (size_t index = 0; index < system_size; ++index) {
          const long double term = static_cast<long double>(
                                       original_inequalities(
                                           certificate_rows[index], column)) *
                                   refined_dual[index];
          stationarity += term;
          absolute_sum += std::abs(term);
        }
        const long double stationarity_tolerance =
            64.0L * std::numeric_limits<long double>::epsilon() *
            static_cast<long double>(certificate_condition) *
            std::max(1.0L, absolute_sum);
        if (std::abs(stationarity) > stationarity_tolerance) {
          stationarity_is_resolved = false;
          break;
        }
      }

      long double certificate_value = 0.0L;
      long double certificate_absolute_sum = 0.0L;
      for (size_t index = 0; index < system_size; ++index) {
        const long double term =
            static_cast<long double>(
                original_bounds(certificate_rows[index])) *
            refined_dual[index];
        certificate_value += term;
        certificate_absolute_sum += std::abs(term);
      }
      const long double negativity_margin =
          100.0L * static_cast<long double>(tolerance) *
          std::max(1.0L, certificate_absolute_sum);
      const bool certificate_valid =
          stationarity_is_resolved &&
          certificate_value < -negativity_margin;
      return certificate_valid
                 ? HardConstraintFeasibilityStatus::kProvenInfeasible
                 : HardConstraintFeasibilityStatus::kUnknown;
    }

    double step = 1.0;
    Eigen::Index blocking_row = -1;
    for (Eigen::Index row = 0; row < num_inequalities; ++row) {
      if (std::find(active.begin(), active.end(), row) != active.end()) {
        continue;
      }
      const double directional_change = inequalities.row(row).dot(direction);
      if (directional_change <= tolerance) {
        continue;
      }
      const double margin =
          bounds(row) - inequalities.row(row).dot(phase_point);
      const double candidate_step =
          std::clamp(margin / directional_change, 0.0, 1.0);
      if (candidate_step < step) {
        step = candidate_step;
        blocking_row = row;
      }
    }

    phase_point.noalias() += step * direction;
    if (phase_point(num_dof) < 0.0 &&
        phase_point(num_dof) >= -10.0 * tolerance) {
      phase_point(num_dof) = 0.0;
    }
    if (!phase_point.allFinite()) {
      return HardConstraintFeasibilityStatus::kUnknown;
    }
    if (blocking_row >= 0 && step < 1.0 - tolerance) {
      if (!add_independent_active_row(blocking_row)) {
        return HardConstraintFeasibilityStatus::kUnknown;
      }
    }
  }

  return HardConstraintFeasibilityStatus::kUnknown;
}

} // namespace detail

namespace linalg {

inline Eigen::MatrixXd ComputeRegularizedInverse(
    const embodik::RegularizedInverseConfig &regularization_config,
    const Eigen::MatrixXd &input_matrix,
    double *condition_number_out = nullptr,
    bool enable_condition_aware_svd_fallback = false) {
  using MatrixDouble = Eigen::MatrixXd;
  using SquareMatrix = Eigen::MatrixXd; // square matrix for gram computation

  SquareMatrix gram_matrix = input_matrix * input_matrix.transpose();

  // Singular value decomposition is already needed for the SRINV damping law;
  // reuse it to report condition-number diagnostics to callers.
  const bool is_tall = input_matrix.rows() > input_matrix.cols();
  const bool exact_mode =
      regularization_config.regularization_factor == 0.0;
  const unsigned int svd_options =
      enable_condition_aware_svd_fallback && (is_tall || exact_mode)
          ? (Eigen::ComputeThinU | Eigen::ComputeThinV)
          : Eigen::ComputeFullU;
  const Eigen::BDCSVD<MatrixDouble> svd_decomp(input_matrix, svd_options);
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
  const double bounded_det_ratio =
      std::clamp(det_value / threshold_squared, 0.0, 1.0);
  const double global_regularization =
      (det_value < threshold_squared)
          ? (1.0 - std::pow(bounded_det_ratio, 2.0)) *
                threshold_squared
          : 0.0;

  SquareMatrix regularized_gram = gram_matrix;
  regularized_gram.diagonal().array() += global_regularization;
  bool gram_condition_reliable = true;

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

    if (enable_condition_aware_svd_fallback && exact_mode) {
      Eigen::VectorXd filtered_inverse =
          Eigen::VectorXd::Zero(sigma_values.size());
      for (Eigen::Index index = 0; index < sigma_values.size(); ++index) {
        if (sigma_values(index) > eps) {
          filtered_inverse(index) = 1.0 / sigma_values(index);
        }
      }
      return svd_decomp.matrixV() * filtered_inverse.asDiagonal() *
             left_singular_vectors.transpose();
    }

    if (enable_condition_aware_svd_fallback && is_tall) {
      const Eigen::ArrayXd filtered_inverse =
          sigma_values.array() /
          (sigma_values.array().square() + global_regularization +
           per_sv_damping);
      return svd_decomp.matrixV() * filtered_inverse.matrix().asDiagonal() *
             left_singular_vectors.transpose();
    }

    if (enable_condition_aware_svd_fallback) {
      const Eigen::ArrayXd regularized_gram_eigenvalues =
          sigma_values.array().square() + global_regularization +
          per_sv_damping;
      const double smallest_gram_eigenvalue =
          regularized_gram_eigenvalues.minCoeff();
      const double largest_gram_eigenvalue =
          regularized_gram_eigenvalues.maxCoeff();
      gram_condition_reliable =
          smallest_gram_eigenvalue > 0.0 &&
          largest_gram_eigenvalue / smallest_gram_eigenvalue <=
              1.0 / std::numeric_limits<double>::epsilon();
    }

    if ((per_sv_damping > 0.0).any()) {
      regularized_gram.noalias() +=
          left_singular_vectors *
          per_sv_damping.matrix().asDiagonal() *
          left_singular_vectors.transpose();
    }
  }

  if (!enable_condition_aware_svd_fallback) {
    return input_matrix.transpose() * regularized_gram.inverse();
  }

  const Eigen::MatrixXd gram_inverse = regularized_gram.inverse();
  Eigen::MatrixXd regularized_inverse = input_matrix.transpose() * gram_inverse;
  const Eigen::MatrixXd gram_identity =
      regularized_gram * gram_inverse;
  const double inverse_residual =
      (gram_identity -
       Eigen::MatrixXd::Identity(gram_identity.rows(), gram_identity.cols()))
          .norm();
  const double inverse_residual_tolerance =
      std::sqrt(std::numeric_limits<double>::epsilon()) *
      std::max<Eigen::Index>(1, regularized_gram.rows());
  if (gram_condition_reliable && regularized_inverse.allFinite() &&
      gram_identity.allFinite() &&
      inverse_residual <= inverse_residual_tolerance) {
    return regularized_inverse;
  }

  // Explicit Gram inversion can overflow or lose all meaningful precision for
  // square or wide matrices whose smallest singular value is just above the
  // damping threshold. Preserve the established result whenever its condition
  // and inverse residual are sound, and use the equivalent filtered SVD
  // formulation only for this numerical failure case.
  const Eigen::BDCSVD<MatrixDouble> fallback_svd(
      input_matrix, Eigen::ComputeThinU | Eigen::ComputeThinV);
  const Eigen::VectorXd fallback_sigma = fallback_svd.singularValues();
  Eigen::VectorXd filtered_inverse =
      Eigen::VectorXd::Zero(fallback_sigma.size());
  for (Eigen::Index index = 0; index < fallback_sigma.size(); ++index) {
    const double sigma = fallback_sigma(index);
    const double normalized_sigma =
        std::min(1.0, sigma / regularization_config.epsilon);
    const double damping =
        regularization_config.regularization_factor *
        std::max(0.0, 1.0 - normalized_sigma * normalized_sigma);
    const double denominator =
        sigma * sigma + global_regularization + damping;
    if (denominator > 0.0) {
      filtered_inverse(index) = sigma / denominator;
    }
  }
  return fallback_svd.matrixV() * filtered_inverse.asDiagonal() *
         fallback_svd.matrixU().transpose();
}

inline std::pair<double, double>
ComputeGeneralizedFeasibleScalingRange(double bound_lower, double bound_upper,
                                       double coefficient) {
  // Return {maximum, minimum}; minimum > maximum denotes an empty interval.
  if (!std::isfinite(bound_lower) || !std::isfinite(bound_upper) ||
      !std::isfinite(coefficient) || bound_lower > bound_upper) {
    return {0.0, 1.0};
  }
  if (std::fabs(coefficient) <= detail::kMinScalingMagnitude) {
    return (bound_lower <= 0.0 && 0.0 <= bound_upper)
               ? std::pair<double, double>{1.0, 0.0}
               : std::pair<double, double>{0.0, 1.0};
  }

  const double endpoint_a = bound_lower / coefficient;
  const double endpoint_b = bound_upper / coefficient;
  const double min_allowed_scale =
      std::max(0.0, std::min(endpoint_a, endpoint_b));
  const double max_allowed_scale =
      std::min(1.0, std::max(endpoint_a, endpoint_b));
  return {max_allowed_scale, min_allowed_scale};
}

inline constexpr std::pair<double, double>
ComputeFeasibleScalingRange(double bound_lower, double bound_upper,
                            double coefficient) {
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
  return {std::min(1.0, max_allowed_scale),
          std::max(0.0, min_allowed_scale)};
}

} // namespace linalg

// Formulation-neutral hierarchical linear solver. The caller chooses the
// physical decision variable u. Each objective is represented as
// A_k u = s_k b'_k - b''_k, where b'_k is scalable and b''_k is not.
inline SolverResult solveHierarchicalLinearSystemEigen(
    const std::vector<Eigen::VectorXd> &scalable_objective_targets,
    const std::vector<Eigen::VectorXd> &affine_objective_biases,
    const std::vector<Eigen::MatrixXd> &objective_matrices,
    const Eigen::MatrixXd &constraint_coefficients,
    const Eigen::VectorXd &min_bounds, const Eigen::VectorXd &max_bounds,
    const VelocitySolverConfig &solver_config,
    const std::vector<ObjectiveSolveConfig> &objective_configs = {},
    const Eigen::VectorXd *max_constraint_softening_factors = nullptr);

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
inline SolverResult solveHierarchicalLinearSystemEigen(
    const std::vector<Eigen::VectorXd>
        &scalable_objective_targets, // b'_k in A_k u = s_k b'_k - b''_k
    const std::vector<Eigen::VectorXd>
        &affine_objective_biases, // positive, unscaled b''_k term
    const std::vector<Eigen::MatrixXd>
        &objective_matrices, // A_k in A_k u = s_k b'_k - b''_k
    const Eigen::MatrixXd
        &constraint_coefficients,      // (n+k) x n constraint system
    const Eigen::VectorXd &min_bounds, // (n+k) lower bounds
    const Eigen::VectorXd &max_bounds, // (n+k) upper bounds
    const VelocitySolverConfig &solver_config,
    const std::vector<ObjectiveSolveConfig> &objective_configs,
    const Eigen::VectorXd *max_constraint_softening_factors,
    const detail::HierarchicalLinearSolverExecutionOptions &execution_options) {
  const bool preserve_legacy_velocity_behavior =
      execution_options.policy ==
      detail::HierarchicalLinearSolverPolicy::kLegacyVelocity;
  const double objective_equality_tolerance =
      std::max(std::max(solver_config.epsilon,
                        solver_config.precision_threshold),
               execution_options.objective_equality_tolerance);
  const auto compute_projected_objective_inverse =
      [&](const RegularizedInverseConfig &regularization_config,
          const Eigen::MatrixXd &objective_matrix,
          const Eigen::MatrixXd &projector, double *condition_number) {
        Eigen::MatrixXd inverse = linalg::ComputeRegularizedInverse(
            regularization_config, objective_matrix * projector,
            condition_number, !preserve_legacy_velocity_behavior);
        if (!preserve_legacy_velocity_behavior) {
          // Regularized inversion can leak numerically outside the strict
          // higher-priority space even though (A P)^+ should lie in range(P).
          inverse = projector * inverse;
        }
        return inverse;
      };
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
  if (scalable_objective_targets.size() != objective_matrices.size())
    return fail(SolverStatus::kShapeMismatch,
                "goals/jacobians size mismatch in Eigen input");
  if ((!preserve_legacy_velocity_behavior ||
       !affine_objective_biases.empty()) &&
      affine_objective_biases.size() != scalable_objective_targets.size())
    return fail(SolverStatus::kShapeMismatch,
                "affine biases/objectives size mismatch in Eigen input");
  if (constraint_coefficients.cols() == 0 ||
      constraint_coefficients.rows() == 0)
    return fail(SolverStatus::kEmptyProblem, "empty constraint matrix");
  if (min_bounds.size() != constraint_coefficients.rows() ||
      max_bounds.size() != constraint_coefficients.rows())
    return fail(SolverStatus::kConstraintBoundsMismatch,
                "constraint rows must match lower/upper bounds");
  if (scalable_objective_targets.empty())
    return fail(SolverStatus::kEmptyProblem, "no objectives provided");
  if (!constraint_coefficients.allFinite() || !min_bounds.allFinite() ||
      !max_bounds.allFinite())
    return fail(SolverStatus::kNonFiniteInput,
                "constraints and bounds must contain only finite values");
  if ((min_bounds.array() > max_bounds.array()).any())
    return fail(SolverStatus::kInvalidInput,
                "constraint lower bounds must not exceed upper bounds");

  // Extract system dimensions
  const auto degrees_of_freedom = constraint_coefficients.cols();
  const auto additional_constraints =
      constraint_coefficients.rows() - degrees_of_freedom;
  const auto total_constraints = constraint_coefficients.rows();
  const bool enhanced_mode = max_constraint_softening_factors != nullptr;
  const auto num_objectives = objective_matrices.size();
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
    if (objective_matrices[obj_idx].cols() != degrees_of_freedom ||
        objective_matrices[obj_idx].rows() !=
            scalable_objective_targets[obj_idx].rows()) {
      return fail(SolverStatus::kShapeMismatch,
                  "objective Jacobian dimension mismatch");
    }
    if (!preserve_legacy_velocity_behavior &&
        affine_objective_biases[obj_idx].rows() !=
            scalable_objective_targets[obj_idx].rows()) {
      return fail(SolverStatus::kShapeMismatch,
                  "affine bias dimension mismatch");
    }
    if (!objective_matrices[obj_idx].allFinite() ||
        !scalable_objective_targets[obj_idx].allFinite() ||
        (!preserve_legacy_velocity_behavior &&
         !affine_objective_biases[obj_idx].allFinite())) {
      return fail(SolverStatus::kNonFiniteInput,
                  "objective matrices and vectors must be finite");
    }
  }

  auto phase_one_failure = [&](SolverStatus status, const char *message) {
    SolverResult out{};
    out.solution.assign(static_cast<size_t>(degrees_of_freedom), 0.0);
    out.status = status;
    out.computation_time_ms = get_elapsed_ms();
    out.task_scales.reserve(num_objectives);
    out.task_errors.reserve(num_objectives);
    out.task_modes_effective.reserve(num_objectives);
    out.task_used_fallback.assign(num_objectives, false);
    for (size_t objective_index = 0; objective_index < num_objectives;
         ++objective_index) {
      out.task_errors.push_back(
          preserve_legacy_velocity_behavior
              ? scalable_objective_targets[objective_index].norm()
              : (scalable_objective_targets[objective_index] -
                 affine_objective_biases[objective_index])
                    .norm());
      const TaskSolveMode requested_mode =
          objective_index < objective_configs.size()
              ? objective_configs[objective_index].solve_mode
              : TaskSolveMode::kScale;
      out.task_scales.push_back(requested_mode == TaskSolveMode::kMinError
                                    ? -1.0
                                    : 0.0);
      out.task_modes_effective.push_back(
          requested_mode == TaskSolveMode::kScaleElastic
              ? TaskSolveMode::kScale
              : requested_mode);
    }
    out.status_message = message;
    return out;
  };

  std::vector<Eigen::Index> immutable_constraint_rows;
  immutable_constraint_rows.reserve(static_cast<size_t>(total_constraints));
  if (!preserve_legacy_velocity_behavior) {
    for (Eigen::Index row = 0; row < total_constraints; ++row) {
      if (!enhanced_mode || softening_factors(row) <= 1.0) {
        immutable_constraint_rows.push_back(row);
      }
    }
  }
  const bool run_hard_constraint_phase_one =
      !immutable_constraint_rows.empty();
  Eigen::MatrixXd phase_one_coefficients(immutable_constraint_rows.size(),
                                         degrees_of_freedom);
  Eigen::VectorXd phase_one_min_bounds(immutable_constraint_rows.size());
  Eigen::VectorXd phase_one_max_bounds(immutable_constraint_rows.size());
  for (size_t index = 0; index < immutable_constraint_rows.size(); ++index) {
    const Eigen::Index row = immutable_constraint_rows[index];
    phase_one_coefficients.row(static_cast<Eigen::Index>(index)) =
        constraint_coefficients.row(row);
    phase_one_min_bounds(static_cast<Eigen::Index>(index)) = min_bounds(row);
    phase_one_max_bounds(static_cast<Eigen::Index>(index)) = max_bounds(row);
  }
  Eigen::VectorXd hard_constraint_witness;
  const auto hard_constraint_feasibility =
      run_hard_constraint_phase_one
          ? detail::FindHardConstraintFeasibleWitness(
                phase_one_coefficients, phase_one_min_bounds,
                phase_one_max_bounds, solver_config, &hard_constraint_witness)
          : detail::HardConstraintFeasibilityStatus::kUnknown;
  if (run_hard_constraint_phase_one &&
      hard_constraint_feasibility ==
          detail::HardConstraintFeasibilityStatus::kProvenInfeasible) {
    return phase_one_failure(
        SolverStatus::kInfeasible,
        "hard constraint Phase I certified the system infeasible");
  }
  if (run_hard_constraint_phase_one &&
      hard_constraint_feasibility ==
          detail::HardConstraintFeasibilityStatus::kUnknown) {
    return phase_one_failure(
        SolverStatus::kNoProgress,
        "hard constraint Phase I did not reach a certified result");
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
    const auto &current_jacobian = objective_matrices[0];
    const auto &current_target = scalable_objective_targets[0];
    Eigen::VectorXd affine_target_storage;
    const Eigen::VectorXd *current_full_target = &current_target;
    if (!preserve_legacy_velocity_behavior) {
      affine_target_storage = current_target - affine_objective_biases[0];
      current_full_target = &affine_target_storage;
    }
    double direct_condition_number = 1.0;
    Eigen::VectorXd direct_velocity =
        linalg::ComputeRegularizedInverse(solver_config.regularization_config,
                                          current_jacobian,
                                          &direct_condition_number) *
        *current_full_target;
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
      const double direct_objective_error =
          (current_jacobian * direct_velocity - *current_full_target).norm();
      const double objective_tolerance =
          solver_config.precision_threshold *
          (1.0 + current_full_target->norm());
      const bool objective_satisfied =
          preserve_legacy_velocity_behavior ||
          objective_config.solve_mode == TaskSolveMode::kMinError ||
          direct_objective_error <= objective_tolerance;
      if (feasible && objective_satisfied) {
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
        std::vector<double> objective_errors{direct_objective_error};
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
    const bool allow_min_error_fallback =
        (preserve_legacy_velocity_behavior ||
         objective_index < objective_configs.size()) &&
        objective_config.allow_min_error_fallback;
    bool objective_is_min_error =
        (objective_config.solve_mode == TaskSolveMode::kMinError);
    objective_effective_modes[objective_index] = objective_config.solve_mode;
    const auto &current_jacobian = objective_matrices[objective_index];
    const auto &current_target = scalable_objective_targets[objective_index];
    const Eigen::VectorXd current_full_target =
        preserve_legacy_velocity_behavior
            ? current_target
            : current_target - affine_objective_biases[objective_index];
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
    bool has_feasible_scale_snapshot = false;
    auto best_constraint_selection = saturated_constraint_selector;
    best_saturated_values = saturated_values;
    Eigen::MatrixXd optimal_constrained_projector = Eigen::MatrixXd::Zero(
        constrained_projector.rows(), constrained_projector.cols());
    best_augmented_projector.setZero();
    const Eigen::VectorXd objective_residual_at_previous =
        current_full_target - current_jacobian * previous_velocity;
    const bool previous_satisfies_hard_constraints =
        detail::ComputeMaxLinearConstraintViolation(
            constraint_coefficients, *active_min_bounds, *active_max_bounds,
            previous_velocity) <= solver_config.epsilon;
    const bool skip_objective = preserve_legacy_velocity_behavior
                                    ? current_target.template lpNorm<1>() <
                                          solver_config.precision_threshold
                                    : previous_satisfies_hard_constraints &&
                                          objective_residual_at_previous
                                                  .template lpNorm<1>() <
                                              solver_config.precision_threshold;
    if (skip_objective) {
      velocity_solution = previous_velocity;
      constraints_violated = false;
    }

    // Pre-compute transformation matrices for efficiency
    double task_condition_number = 1.0;
    Eigen::MatrixXd damped_inverse_projected_jacobian =
        compute_projected_objective_inverse(
            solver_config.regularization_config, current_jacobian,
            constrained_projector, &task_condition_number);
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
              (current_full_target - jacobian_velocity_product) +
          augmented_projector *
              (saturated_values - saturated_constraints_velocity);

      // Evaluate constraint satisfaction
      constraint_evaluation.noalias() =
          constraint_coefficients * velocity_solution;
      const bool full_scale_constraints_violated =
          (constraint_evaluation.array() <
           (active_min_bounds->array() - solver_config.epsilon))
              .any() ||
          (constraint_evaluation.array() >
           (active_max_bounds->array() + solver_config.epsilon))
              .any();
      constraints_violated = full_scale_constraints_violated;

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
        if (preserve_legacy_velocity_behavior &&
            contribution_magnitude < solver_config.epsilon) {
          velocity_scale = 1.0;
        } else if (contribution_magnitude > solver_config.magnitude_limit) {
          velocity_scale = 0.0;
          solver_status = SolverStatus::kNumericalError;
        } else if (preserve_legacy_velocity_behavior) {
          min_margin = *active_min_bounds - unscaled_contribution;
          max_margin = *active_max_bounds - unscaled_contribution;
          Eigen::VectorXd feasible_scales =
              Eigen::VectorXd::Zero(total_constraints);
          for (Eigen::Index constraint_idx = 0;
               constraint_idx < total_constraints; ++constraint_idx) {
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
          velocity_scale =
              feasible_scales.minCoeff(&critical_constraint_index);
        } else {
          min_margin = *active_min_bounds - unscaled_contribution;
          max_margin = *active_max_bounds - unscaled_contribution;

          double global_min_scale = 0.0;
          double global_max_scale = 1.0;
          bool scale_interval_feasible = true;
          for (Eigen::Index constraint_idx = 0;
               constraint_idx < total_constraints; ++constraint_idx) {
            if (saturated_constraint_selector(constraint_idx, constraint_idx) ==
                1) {
              continue;
            }
            const auto scale_range =
                linalg::ComputeGeneralizedFeasibleScalingRange(
                min_margin[constraint_idx], max_margin[constraint_idx],
                scaled_velocity_contribution[constraint_idx]);
            if (scale_range.first < global_max_scale) {
              global_max_scale = scale_range.first;
              critical_constraint_index = constraint_idx;
            }
            if (scale_range.second > global_min_scale) {
              global_min_scale = scale_range.second;
              if (global_min_scale > global_max_scale +
                                         solver_config.epsilon) {
                critical_constraint_index = constraint_idx;
              }
            }
            if (global_min_scale > global_max_scale +
                                       solver_config.epsilon) {
              scale_interval_feasible = false;
            }
          }

          velocity_scale = scale_interval_feasible
                               ? std::clamp(global_max_scale, 0.0, 1.0)
                               : 0.0;
        }
      }

      if (velocity_scale == std::numeric_limits<double>::infinity() ||
          velocity_scale == -std::numeric_limits<double>::infinity()) {
        velocity_scale = 0.0;
      }

      bool saturation_preserves_task_rank = false;
      if (!preserve_legacy_velocity_behavior && !objective_is_min_error &&
          full_scale_constraints_violated &&
          velocity_scale <= solver_config.epsilon) {
        const Eigen::MatrixXd critical_row_on_space =
            constraint_coefficients.row(critical_constraint_index) *
            constrained_projector;
        Eigen::MatrixXd critical_row_inverse;
        detail::ComputeGeneralizedInverse(critical_row_on_space,
                                          solver_config.epsilon,
                                          &critical_row_inverse);
        const Eigen::MatrixXd projector_after_saturation =
            constrained_projector -
            critical_row_inverse * critical_row_on_space;
        Eigen::ColPivHouseholderQR<Eigen::MatrixXd> rank_analysis(
            current_jacobian * projector_after_saturation);
        rank_analysis.setThreshold(solver_config.epsilon);
        saturation_preserves_task_rank =
            rank_analysis.rank() >= target_dimension;
      }

      const bool should_use_min_error_fallback =
          preserve_legacy_velocity_behavior || !saturation_preserves_task_rank;
      if (!objective_is_min_error && velocity_scale <= solver_config.epsilon &&
          allow_min_error_fallback && should_use_min_error_fallback) {
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
        Eigen::VectorXd target_term =
            objective_is_min_error ? current_target
                                   : (velocity_scale * current_target);
        if (!preserve_legacy_velocity_behavior) {
          target_term -= affine_objective_biases[objective_index];
        }
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

        const bool activate_critical_constraint =
            !preserve_legacy_velocity_behavior && !objective_is_min_error &&
            full_scale_constraints_violated &&
            velocity_scale < 1.0 - solver_config.epsilon;

        // Fast path: if all constraints are satisfied, accept this objective
        // update directly and stop saturating rows for this objective.
        if (!constraint_violations.any() && !activate_critical_constraint) {
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
          has_feasible_scale_snapshot = true;
          constraints_violated = false;
        } else {
          if (!constraint_violations.any() &&
              velocity_scale >= optimal_scale) {
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
            has_feasible_scale_snapshot = true;
          }
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
            bool rebuild_current_augmented_projector = false;
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
                rebuild_current_augmented_projector = true;
              } else {
                // No feasible snapshot yet, but active-set still evolving:
                // solve constrained least-squares with current saturated set.
                velocity_scale = 1.0;
                rebuild_current_augmented_projector = true;
              }
            } else if (preserve_legacy_velocity_behavior) {
              velocity_scale = optimal_scale;
              saturated_constraint_selector = best_constraint_selection;
              saturated_constraint_matrix.noalias() =
                  saturated_constraint_selector * constraint_coefficients;
              saturated_values = best_saturated_values;
              constrained_projector = optimal_constrained_projector;
              augmented_projector = best_augmented_projector;
            } else {
              if (has_feasible_scale_snapshot) {
                velocity_scale = optimal_scale;
                saturated_constraint_selector = best_constraint_selection;
                saturated_constraint_matrix.noalias() =
                    saturated_constraint_selector * constraint_coefficients;
                saturated_values = best_saturated_values;
                constrained_projector = optimal_constrained_projector;
                augmented_projector = best_augmented_projector;
              } else if (previous_satisfies_hard_constraints) {
                // A lower-priority task with no feasible scale must not move an
                // already feasible higher-priority solution. Restoring the
                // objective-entry state also avoids retaining trial saturation
                // rows from the failed task.
                velocity_scale = 0.0;
                velocity_solution = previous_velocity;
                saturated_constraint_selector = best_constraint_selection;
                saturated_constraint_matrix.noalias() =
                    saturated_constraint_selector * constraint_coefficients;
                saturated_values = best_saturated_values;
                constrained_projector = previous_null_space;
                augmented_projector.setZero();
                use_direct_constrained_solve = false;
              } else {
                // The hard set can exclude u=0 even for a zero task command.
                // In that case there is no earlier feasible scaled snapshot to
                // restore; retain the current saturation set and let hard rows
                // take precedence over the exhausted task.
                velocity_scale = 0.0;
                rebuild_current_augmented_projector = true;
              }
            }

            if (rebuild_current_augmented_projector) {
              saturated_constraints_on_previous_space.noalias() =
                  saturated_constraint_matrix * previous_null_space;
              detail::ComputeGeneralizedInverse(
                  saturated_constraints_on_previous_space,
                  solver_config.epsilon,
                  &inverse_saturated_constraints_projected);
              double augmented_cond = 1.0;
              const Eigen::MatrixXd final_task_inverse =
                  compute_projected_objective_inverse(
                      solver_config.regularization_config, current_jacobian,
                      constrained_projector, &augmented_cond);
              augmented_projector.noalias() =
                  (Eigen::MatrixXd::Identity(final_task_inverse.rows(),
                                             current_jacobian.cols()) -
                   final_task_inverse * current_jacobian) *
                  inverse_saturated_constraints_projected;
              worst_condition_number =
                  std::max(worst_condition_number, augmented_cond);
            }

            if (use_direct_constrained_solve) {
              Eigen::VectorXd target_term_final =
                  objective_is_min_error ? current_target
                                         : (velocity_scale * current_target);
              if (!preserve_legacy_velocity_behavior) {
                target_term_final -= affine_objective_biases[objective_index];
              }
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
                  compute_projected_objective_inverse(
                      adaptive_regularization, current_jacobian,
                      constrained_projector, &cond_tmp) *
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
      bool should_project_objective =
          velocity_scale > 0.0 || objective_is_min_error;
      if (!preserve_legacy_velocity_behavior && !objective_is_min_error) {
        const Eigen::VectorXd scaled_rhs =
            velocity_scale * current_target -
            affine_objective_biases[objective_index];
        const double equality_tolerance =
            objective_equality_tolerance *
            std::max(1.0, scaled_rhs.norm());
        should_project_objective =
            (current_jacobian * velocity_solution - scaled_rhs).norm() <=
            equality_tolerance;
      }
      if (should_project_objective) {
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
        damped_inverse_projected_jacobian =
            compute_projected_objective_inverse(
                solver_config.regularization_config, current_jacobian,
                constrained_projector, &cond_tmp);
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
  const bool validate_final_hard_constraints =
      !preserve_legacy_velocity_behavior ||
      objective_scaling_factors.sum() > solver_config.epsilon;
  if (validate_final_hard_constraints &&
      (((constraint_coefficients * velocity_solution).array() <
        (final_check_min_bounds.array() - solver_config.epsilon))
           .any() ||
       ((constraint_coefficients * velocity_solution).array() >
        (final_check_max_bounds.array() + solver_config.epsilon))
           .any())) {
    solver_status = SolverStatus::kNumericalError;
  }

  if (!preserve_legacy_velocity_behavior &&
      solver_status == SolverStatus::kSuccess) {
    for (size_t objective_index = 0; objective_index < num_objectives;
         ++objective_index) {
      if (objective_effective_modes[objective_index] ==
          TaskSolveMode::kMinError) {
        continue;
      }
      const Eigen::VectorXd scaled_rhs =
          objective_scaling_factors(static_cast<Eigen::Index>(objective_index)) *
              scalable_objective_targets[objective_index] -
          affine_objective_biases[objective_index];
      const double equality_tolerance =
          objective_equality_tolerance *
          std::max(1.0, scaled_rhs.norm());
      if ((objective_matrices[objective_index] * velocity_solution - scaled_rhs)
              .norm() > equality_tolerance) {
        solver_status = SolverStatus::kNoProgress;
        break;
      }
    }
  }

  bool generated_nonfinite_solver_state = false;
  if (!objective_scaling_factors.allFinite() || !velocity_solution.allFinite()) {
    generated_nonfinite_solver_state = true;
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
    const auto &scalable_target =
        scalable_objective_targets[static_cast<size_t>(obj_idx)];
    Eigen::VectorXd affine_target_storage;
    const Eigen::VectorXd *full_target = &scalable_target;
    if (!preserve_legacy_velocity_behavior) {
      affine_target_storage =
          scalable_target -
          affine_objective_biases[static_cast<size_t>(obj_idx)];
      full_target = &affine_target_storage;
    }
    objective_errors[static_cast<size_t>(obj_idx)] =
        (objective_matrices[static_cast<size_t>(obj_idx)] * velocity_solution -
         *full_target)
            .norm();
  }

  auto end_time = std::chrono::high_resolution_clock::now();
  const double elapsed_milliseconds =
      std::chrono::duration_cast<std::chrono::microseconds>(end_time - t0)
          .count() /
      1000.0;

  std::string status_message;
  if (solver_status == SolverStatus::kNumericalError) {
    status_message = generated_nonfinite_solver_state
                         ? "non-finite values generated in solver state"
                         : "numerical constraint violation beyond epsilon after solve";
  } else if (solver_status == SolverStatus::kNoProgress) {
    status_message = "no feasible scale satisfies the task equality";
  } else if (solver_status == SolverStatus::kNonFiniteInput) {
    status_message = "non-finite values detected in solver state";
  } else if (solver_status != SolverStatus::kSuccess) {
    status_message = "solver failed with input/status error";
  }

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
      std::move(status_message),
      worst_condition_number};
}

namespace detail {

inline SolverResult SolveGeneralizedHierarchicalLinearSystemEigen(
    const std::vector<Eigen::VectorXd> &scalable_objective_targets,
    const std::vector<Eigen::VectorXd> &affine_objective_biases,
    const std::vector<Eigen::MatrixXd> &objective_matrices,
    const Eigen::MatrixXd &constraint_coefficients,
    const Eigen::VectorXd &min_bounds, const Eigen::VectorXd &max_bounds,
    const VelocitySolverConfig &solver_config,
    const std::vector<ObjectiveSolveConfig> &objective_configs,
    const Eigen::VectorXd *max_constraint_softening_factors,
    double objective_equality_tolerance) {
  Eigen::MatrixXd normalized_coefficients;
  Eigen::VectorXd normalized_min_bounds;
  Eigen::VectorXd normalized_max_bounds;
  NormalizeLinearConstraintRows(
      constraint_coefficients, min_bounds, max_bounds,
      &normalized_coefficients, &normalized_min_bounds,
      &normalized_max_bounds);
  return solveHierarchicalLinearSystemEigen(
      scalable_objective_targets, affine_objective_biases, objective_matrices,
      normalized_coefficients, normalized_min_bounds, normalized_max_bounds,
      solver_config, objective_configs, max_constraint_softening_factors,
      HierarchicalLinearSolverExecutionOptions{
          HierarchicalLinearSolverPolicy::kGeneralizedEsns,
          objective_equality_tolerance});
}

} // namespace detail

inline SolverResult solveHierarchicalLinearSystemEigen(
    const std::vector<Eigen::VectorXd> &scalable_objective_targets,
    const std::vector<Eigen::VectorXd> &affine_objective_biases,
    const std::vector<Eigen::MatrixXd> &objective_matrices,
    const Eigen::MatrixXd &constraint_coefficients,
    const Eigen::VectorXd &min_bounds, const Eigen::VectorXd &max_bounds,
    const VelocitySolverConfig &solver_config,
    const std::vector<ObjectiveSolveConfig> &objective_configs,
    const Eigen::VectorXd *max_constraint_softening_factors) {
  return detail::SolveGeneralizedHierarchicalLinearSystemEigen(
      scalable_objective_targets, affine_objective_biases, objective_matrices,
      constraint_coefficients, min_bounds, max_bounds, solver_config,
      objective_configs, max_constraint_softening_factors, 0.0);
}

inline SolverResult computeMultiObjectiveVelocitySolutionEigen(
    const std::vector<Eigen::VectorXd> &objective_targets,
    const std::vector<Eigen::MatrixXd> &objective_jacobians,
    const Eigen::MatrixXd &constraint_coefficients,
    const Eigen::VectorXd &min_bounds, const Eigen::VectorXd &max_bounds,
    const VelocitySolverConfig &solver_config,
    const std::vector<ObjectiveSolveConfig> &objective_configs,
    const Eigen::VectorXd *max_constraint_softening_factors) {
  return solveHierarchicalLinearSystemEigen(
      objective_targets, {}, objective_jacobians,
      constraint_coefficients, min_bounds, max_bounds, solver_config,
      objective_configs, max_constraint_softening_factors,
      detail::HierarchicalLinearSolverExecutionOptions{
          detail::HierarchicalLinearSolverPolicy::kLegacyVelocity, 0.0});
}

// No additional aliases defined

} // namespace embodik
