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
#include <cstring>
#include <limits>
#include <optional>
#include <stdexcept>
#include <vector>

#include <Eigen/Cholesky>
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

// Internal terminal-retry state; the caller retains status/finite checks.
struct ObjectiveMinErrorRetryState {
  TaskSolveMode effective_mode = TaskSolveMode::kScale;
  bool used_fallback = false;
  bool final_bound_failure = false;
  double scale = 1.0;
  double error = 0.0;
  bool failed_requested_equality = false;
};

inline bool ShouldRetryObjectiveAsMinError(
    const ObjectiveSolveConfig &objective,
    const VelocitySolverConfig &solver,
    const ObjectiveMinErrorRetryState &state) {
  const bool failed_authorized_min_error_fallback =
      state.final_bound_failure && state.used_fallback &&
      state.effective_mode == TaskSolveMode::kMinError;
  return objective.allow_min_error_fallback &&
      objective.solve_mode != TaskSolveMode::kMinError &&
      (state.effective_mode != TaskSolveMode::kMinError ||
       failed_authorized_min_error_fallback) &&
      (state.scale <= solver.epsilon || state.failed_requested_equality) &&
      (state.error > solver.precision_threshold ||
       failed_authorized_min_error_fallback);
}

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

// The owning solver validates shapes and finite original inputs first. This
// gate validates a completed numerical checkpoint; it does not certify that
// an unoptimized lower task is optimal or that nonlinear constraints hold.
inline bool CanRetainBestEffortLowerTaskCheckpoint(
    bool enabled, const Eigen::VectorXd &checkpoint,
    const std::vector<Eigen::VectorXd> &completed_images,
    size_t first_unoptimized,
    const std::vector<Eigen::MatrixXd> &objective_matrices,
    const std::vector<Eigen::VectorXd> &scalable_objective_targets,
    const std::vector<Eigen::VectorXd> &affine_objective_biases,
    const std::vector<ObjectiveSolveConfig> &objective_configs,
    const Eigen::MatrixXd &constraint_coefficients,
    const Eigen::VectorXd &min_bounds, const Eigen::VectorXd &max_bounds,
    double constraint_tolerance, double objective_equality_tolerance) {
  const size_t num_objectives = objective_matrices.size();
  if (!enabled ||
      first_unoptimized == 0 ||
      first_unoptimized >= num_objectives ||
      completed_images.size() != first_unoptimized ||
      objective_configs.size() != num_objectives ||
      checkpoint.size() != constraint_coefficients.cols() || !checkpoint.allFinite() ||
      !(constraint_coefficients * checkpoint).allFinite() ||
      detail::ComputeMaxLinearConstraintViolation(
          constraint_coefficients, min_bounds, max_bounds, checkpoint) >
          constraint_tolerance) return false;
  for (size_t index = first_unoptimized; index < num_objectives; ++index) {
    if (objective_configs[index].solve_mode != TaskSolveMode::kMinError)
      return false;
  }
  for (size_t index = 0; index < first_unoptimized; ++index) {
    const auto &image = completed_images[index];
    if (!image.allFinite() ||
        image.size() != objective_matrices[index].rows()) return false;
    const Eigen::VectorXd difference =
        objective_matrices[index] * checkpoint - image;
    const double image_norm = image.norm();
    const double residual_norm = difference.norm();
    if (!difference.allFinite() || !std::isfinite(image_norm) ||
        !std::isfinite(residual_norm) ||
        residual_norm > objective_equality_tolerance * std::max(1.0, image_norm))
      return false;
  }
  for (size_t index = 0; index < num_objectives; ++index) {
    const Eigen::VectorXd target =
        scalable_objective_targets[index] - affine_objective_biases[index];
    const Eigen::VectorXd residual =
        objective_matrices[index] * checkpoint - target;
    if (!target.allFinite() || !residual.allFinite() ||
        !std::isfinite(residual.norm())) return false;
  }
  return true;
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

/**
 * Recover a SCALE task from a feasible witness at seed_scale. Unlike
 * saturation-only SNS, this primal active-set search can release a bound whose
 * multiplier has the wrong sign. All moves retain the task ray, prior task
 * values and the original interval constraints. Failure leaves the caller's
 * solution unchanged; this is not a minimum-error fallback.
 */
inline bool RecoverZeroScaleTask(
    const Eigen::MatrixXd &jacobian, const Eigen::VectorXd &target,
    const Eigen::VectorXd &bias, const Eigen::MatrixXd &protected_tasks,
    const Eigen::MatrixXd &coefficients, const Eigen::VectorXd &lower,
    const Eigen::VectorXd &upper, const Eigen::VectorXd &seed,
    Eigen::VectorXd *velocity, double *scale, double seed_scale = 0.0) {
  constexpr double tolerance = 1e-10;
  const Eigen::Index n = seed.size();
  const Eigen::Index equality_rows = jacobian.rows() + protected_tasks.rows();
  if (!std::isfinite(seed_scale) || seed_scale < 0.0 || seed_scale > 1.0 ||
      !seed.allFinite() ||
      ComputeMaxLinearConstraintViolation(coefficients, lower, upper, seed) >
          tolerance ||
      (jacobian * seed + bias - seed_scale * target).norm() >
          tolerance * std::max({1.0, bias.norm(), seed_scale * target.norm()})) {
    return false;
  }
  Eigen::MatrixXd equalities = Eigen::MatrixXd::Zero(equality_rows, n + 1);
  equalities.topLeftCorner(jacobian.rows(), n) = jacobian;
  equalities.col(n).head(jacobian.rows()) = -target;
  equalities.bottomLeftCorner(protected_tasks.rows(), n) = protected_tasks;
  for (Eigen::Index row = 0; row < equality_rows; ++row) {
    const double norm = equalities.row(row).stableNorm();
    if (norm > 0.0) equalities.row(row) /= norm;
  }
  const Eigen::Index m = coefficients.rows();
  Eigen::MatrixXd inequalities = Eigen::MatrixXd::Zero(2 * m + 2, n + 1);
  Eigen::VectorXd bounds(2 * m + 2);
  inequalities.topLeftCorner(m, n) = coefficients;
  inequalities.block(m, 0, m, n) = -coefficients;
  bounds.head(m) = upper;
  bounds.segment(m, m) = -lower;
  inequalities(2 * m, n) = 1.0;
  inequalities(2 * m + 1, n) = -1.0;
  bounds(2 * m) = 1.0;
  bounds(2 * m + 1) = 0.0;
  for (Eigen::Index row = 0; row < 2 * m; ++row) {
    const double norm = inequalities.row(row).stableNorm();
    if (norm > 0.0) {
      inequalities.row(row) /= norm;
      bounds(row) /= norm;
    }
  }
  Eigen::VectorXd point = Eigen::VectorXd::Zero(n + 1);
  point.head(n) = seed;
  point(n) = seed_scale;
  const Eigen::VectorXd equality_values = equalities * point;
  Eigen::VectorXd objective = Eigen::VectorXd::Zero(n + 1);
  objective(n) = 1.0;
  std::vector<Eigen::Index> active;
  const Eigen::Index iteration_limit = std::min<Eigen::Index>(
      512, std::max<Eigen::Index>(64, 8 * (n + m + 1)));
  for (Eigen::Index iteration = 0; iteration < iteration_limit; ++iteration) {
    Eigen::MatrixXd rows(equality_rows + active.size(), n + 1);
    rows.topRows(equality_rows) = equalities;
    for (size_t index = 0; index < active.size(); ++index) {
      rows.row(equality_rows + index) = inequalities.row(active[index]);
    }
    // Form the null space directly. Subtracting pseudoinverse projectors can
    // lose orthogonality near dependent task/bound rows and step outside bounds.
    Eigen::JacobiSVD<Eigen::MatrixXd> decomposition(rows, Eigen::ComputeFullV);
    decomposition.setThreshold(tolerance);
    const Eigen::MatrixXd free_space =
        decomposition.matrixV().rightCols(n + 1 - decomposition.rank());
    const Eigen::VectorXd direction = free_space * (free_space.transpose() * objective);
    if (!direction.allFinite()) return false;
    if (direction.norm() <= tolerance) {
      auto dual_solver = rows.transpose().completeOrthogonalDecomposition();
      dual_solver.setThreshold(tolerance);
      const Eigen::VectorXd multipliers = dual_solver.solve(objective);
      if ((rows.transpose() * multipliers - objective).norm() > 1e-8)
        return false;
      Eigen::Index remove = -1;
      double most_negative = -tolerance;
      for (size_t index = 0; index < active.size(); ++index) {
        if (multipliers(equality_rows + index) < most_negative) {
          most_negative = multipliers(equality_rows + index);
          remove = static_cast<Eigen::Index>(index);
        }
      }
      if (remove >= 0) {
        active.erase(active.begin() + remove);
        continue;
      }
      break;
    }
    const Eigen::VectorXd slopes = inequalities * direction;
    const Eigen::VectorXd slack = bounds - inequalities * point;
    double distance = std::numeric_limits<double>::infinity();
    Eigen::Index blocker = -1;
    for (Eigen::Index row = 0; row < inequalities.rows(); ++row) {
      if (slopes(row) <= tolerance ||
          std::find(active.begin(), active.end(), row) != active.end()) continue;
      const double ratio = std::max(0.0, slack(row)) / slopes(row);
      if (ratio < distance) {
        distance = ratio;
        blocker = row;
      }
    }
    if (blocker < 0 || !std::isfinite(distance)) return false;
    point += distance * direction;
    if (!point.allFinite() ||
        (inequalities * point - bounds).maxCoeff() > 1e-8 ||
        (equalities * point - equality_values).norm() > 1e-8) return false;
    active.push_back(blocker);
    if (point(n) >= 1.0 - tolerance) break;
  }
  const double recovered_scale = std::clamp(point(n), 0.0, 1.0);
  const Eigen::VectorXd candidate = point.head(n);
  if (recovered_scale <= seed_scale + tolerance ||
      ComputeMaxLinearConstraintViolation(coefficients, lower, upper, candidate) > 1e-8 ||
      (jacobian * candidate + bias - recovered_scale * target).norm() >
          1e-8 * std::max(1.0, target.norm()) ||
      (protected_tasks * (candidate - seed)).norm() > 1e-8) return false;
  *velocity = candidate;
  *scale = recovered_scale;
  return true;
}

enum class HierarchicalLinearSolverPolicy {
  kGeneralizedEsns,
  kLegacyVelocity,
};

struct HierarchicalLinearSolverExecutionOptions {
  HierarchicalLinearSolverPolicy policy =
      HierarchicalLinearSolverPolicy::kGeneralizedEsns;
  double objective_equality_tolerance = 0.0;
  // Internal fast-path contract: the caller has already proven that the
  // immutable hard-row set has a feasible witness.
  bool hard_constraints_prevalidated_feasible = false;
  // Optional physical primary rows, before adapter task filtering. Only used
  // to protect stationary final-bound repairs; never changes the task objective.
  const Eigen::MatrixXd *stationary_recovery_protected_tasks = nullptr;
  // Direct velocity requests validate their original primary task range.
  // Position-step adapters retain their legacy projected pose-error policy.
  bool validate_legacy_primary_range = true;
  // Explicitly permit retaining completed task images when a lower
  // MIN_ERROR numerical search fails; never certifies its optimum.
  bool allow_best_effort_lower_tasks = false;
  // Use the feasible-primal quadratic route only after a complete hard-set
  // witness is available and every effective objective requests MIN_ERROR.
  bool prefer_direct_min_error_qp = false;
  // Synchronous recursive retries may reuse the original Phase-I witness.
  // Always revalidate against the current immutable rows before accepting it.
  const Eigen::VectorXd *initial_hard_constraint_witness = nullptr;
};

// Select a wrong active pin only when the computed motion is numerical drift
// in the Phase-I objective nullspace. This changes the search, never acceptance.
inline Eigen::Index FindObjectiveNullPhasePinToRelease(
    const Eigen::MatrixXd &hessian, const Eigen::VectorXd &gradient,
    const Eigen::MatrixXd &active_matrix, const Eigen::VectorXd &direction,
    const Eigen::VectorXd &multipliers, double tolerance,
    double direction_tolerance, double kkt_tolerance) {
  if (multipliers.size() == 0 || !direction.allFinite() ||
      !multipliers.allFinite() ||
      !(direction.norm() > direction_tolerance)) return -1;
  const Eigen::VectorXd objective_direction = hessian * direction;
  const Eigen::VectorXd stationarity =
      gradient + active_matrix.transpose() * multipliers;
  if (!objective_direction.allFinite() || !stationarity.allFinite() ||
      !(objective_direction.norm() <= direction_tolerance) ||
      !(stationarity.norm() <= kkt_tolerance)) return -1;
  Eigen::Index remove_position = -1;
  if (multipliers.minCoeff(&remove_position) < -10.0 * tolerance)
    return remove_position;
  return -1;
}

/**
 * @brief Search for a feasible witness for linear interval constraints.
 *
 * The witness distinguishes a feasible hard set from provable infeasibility;
 * it does not optimize the hierarchy. A caller may seed a constrained search
 * only after including and validating all required task equalities and prefix
 * images. A hard-set witness alone cannot replace that task solution.
 */
inline HardConstraintFeasibilityStatus FindHardConstraintFeasibleWitnessImpl(
    const Eigen::MatrixXd &constraint_coefficients,
    const Eigen::VectorXd &min_bounds, const Eigen::VectorXd &max_bounds,
    const VelocitySolverConfig &solver_config, Eigen::VectorXd *result,
    bool allow_objective_null_pin_release) {
  const Eigen::Index num_dof = constraint_coefficients.cols();
  const Eigen::Index num_constraints = constraint_coefficients.rows();
  const double tolerance = std::max(1e-12, solver_config.epsilon);

  Eigen::VectorXd candidate = Eigen::VectorXd::Zero(num_dof);
  if (ComputeMaxLinearConstraintViolation(constraint_coefficients, min_bounds,
                                          max_bounds, candidate) <= tolerance) {
    *result = std::move(candidate);
    return HardConstraintFeasibilityStatus::kFeasible;
  }

  // Scalar rows give an inexpensive infeasibility certificate even when
  // dependent active rows prevent Phase I from producing a dual witness.
  Eigen::VectorXd scalar_lower = Eigen::VectorXd::Constant(
      num_dof, -std::numeric_limits<double>::infinity());
  Eigen::VectorXd scalar_upper = Eigen::VectorXd::Constant(
      num_dof, std::numeric_limits<double>::infinity());
  for (Eigen::Index row = 0; row < num_constraints; ++row) {
    Eigen::Index coordinate = -1;
    bool scalar_row = true;
    for (Eigen::Index col = 0; col < num_dof; ++col) {
      if (constraint_coefficients(row, col) == 0.0) continue;
      if (coordinate >= 0) {
        scalar_row = false;
        break;
      }
      coordinate = col;
    }
    if (!scalar_row || coordinate < 0) continue;
    const double coefficient = constraint_coefficients(row, coordinate);
    if (!std::isfinite(coefficient) || !std::isfinite(min_bounds(row)) ||
        !std::isfinite(max_bounds(row))) continue;
    const double first = (min_bounds(row) - tolerance) / coefficient;
    const double second = (max_bounds(row) + tolerance) / coefficient;
    if (!std::isfinite(first) || !std::isfinite(second)) continue;
    scalar_lower(coordinate) =
        std::max(scalar_lower(coordinate), std::min(first, second));
    scalar_upper(coordinate) =
        std::min(scalar_upper(coordinate), std::max(first, second));
    if (scalar_lower(coordinate) > scalar_upper(coordinate)) {
      return HardConstraintFeasibilityStatus::kProvenInfeasible;
    }
  }

  // Phase I minimizes the nonnegative maximum normalized violation t:
  //   min 0.5 t^2
  //   s.t. lower_i <= C_i u + t, C_i u - t <= upper_i, t >= 0.
  // A witness alone does not establish any task objective. Callers that use it
  // as an optimization seed must separately audit their original hard rows,
  // affine task equations, and protected task images.
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
    if (!normalized_row.allFinite() || std::isnan(normalized_lower) ||
        std::isnan(normalized_upper) ||
        normalized_lower == std::numeric_limits<double>::infinity() ||
        normalized_upper == -std::numeric_limits<double>::infinity()) {
      return HardConstraintFeasibilityStatus::kUnknown;
    }

    Eigen::RowVectorXd upper_row(num_dof + 1);
    upper_row.head(num_dof) = normalized_row;
    upper_row(num_dof) = -1.0;
    if (std::isfinite(normalized_upper)) {
      inequality_rows.push_back(std::move(upper_row));
      inequality_bounds.push_back(normalized_upper);
    }

    Eigen::RowVectorXd lower_row(num_dof + 1);
    lower_row.head(num_dof) = -normalized_row;
    lower_row(num_dof) = -1.0;
    if (std::isfinite(normalized_lower)) {
      inequality_rows.push_back(std::move(lower_row));
      inequality_bounds.push_back(-normalized_lower);
    }
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
    // Active directions preserve the current row value. A strictly interior
    // row must not become an equality with its initial slack still attached.
    const double touching_tolerance =
        64.0 * std::numeric_limits<double>::epsilon() *
        std::max({1.0, std::abs(bounds(row)),
                  std::abs(inequalities.row(row).dot(phase_point))});
    if (initial_margins(row) <= touching_tolerance) {
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
    Eigen::VectorXd kkt_solution;
    if (active_matrix.rows() == num_dof + 1) {
      // Active rows are independent by construction. A square active matrix
      // therefore pins every phase variable: its only tangent direction is
      // zero. Solving the indefinite KKT system can introduce a tiny spurious
      // direction here and repeatedly miss the stationary-point branch.
      kkt_solution = Eigen::VectorXd::Zero(kkt.rows());
      kkt_solution.tail(active_matrix.rows()) =
          active_matrix.transpose().colPivHouseholderQr().solve(-gradient);
    } else {
      kkt_solution = kkt.completeOrthogonalDecomposition().solve(rhs);
    }
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
    if (allow_objective_null_pin_release) {
      const Eigen::Index remove_position = FindObjectiveNullPhasePinToRelease(
          hessian, gradient, active_matrix, direction, multipliers,
          tolerance, direction_tolerance, kkt_tolerance);
      if (remove_position >= 0) {
        active.erase(active.begin() + remove_position);
        continue;
      }
    }
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

// Preserve every original feasible witness and infeasibility result. Only an
// unresolved search may retry with objective-null pin release, and only an
// independently audited original-row witness may replace UNKNOWN.
inline HardConstraintFeasibilityStatus FindHardConstraintFeasibleWitness(
    const Eigen::MatrixXd &constraint_coefficients,
    const Eigen::VectorXd &min_bounds, const Eigen::VectorXd &max_bounds,
    const VelocitySolverConfig &solver_config, Eigen::VectorXd *result) {
  const auto original_status = FindHardConstraintFeasibleWitnessImpl(
      constraint_coefficients, min_bounds, max_bounds, solver_config, result,
      false);
  if (original_status != HardConstraintFeasibilityStatus::kUnknown)
    return original_status;
  // An unresolved caller without output storage must not enter promotion.
  if (result == nullptr) return original_status;

  Eigen::VectorXd retry_witness;
  const auto retry_status = FindHardConstraintFeasibleWitnessImpl(
      constraint_coefficients, min_bounds, max_bounds, solver_config,
      &retry_witness, true);
  const double tolerance = std::max(1e-12, solver_config.epsilon);
  if (retry_status != HardConstraintFeasibilityStatus::kFeasible ||
      retry_witness.size() != constraint_coefficients.cols() ||
      !retry_witness.allFinite() ||
      !(constraint_coefficients * retry_witness).allFinite() ||
      ComputeMaxLinearConstraintViolation(constraint_coefficients, min_bounds,
                                          max_bounds, retry_witness) > tolerance)
    return original_status;
  *result = std::move(retry_witness);
  return HardConstraintFeasibilityStatus::kFeasible;
}

// Obtain an affine-ray seed only when the original seed does not already meet
// the recovery contract. Prefix values belong to the trusted objective-entry
// command, not to a failed saturation trial. All successful outputs remain on
// that same ray and inside the original hard set.
template <class Validator>
inline bool AcceptAffineRayPhaseSeed(
    const Eigen::VectorXd &point, Eigen::Index degrees_of_freedom,
    bool project_scale, const Validator &validate,
    Eigen::VectorXd *candidate, double *candidate_scale) {
  if (degrees_of_freedom <= 0 || point.size() != degrees_of_freedom + 1 ||
      !point.allFinite()) return false;
  // Project only the auxiliary scale, never the physical command. The caller's
  // original hard/ray/prefix audit must accept the resulting candidate.
  double alpha = project_scale ? std::clamp(point(degrees_of_freedom), 0.0, 1.0)
                               : point(degrees_of_freedom);
  const double scale_roundoff =
      64.0 * std::numeric_limits<double>::epsilon() *
      std::max(1.0, std::abs(alpha));
  if (alpha < 0.0 && -alpha <= scale_roundoff)
    alpha = 0.0;
  else if (alpha > 1.0 && alpha - 1.0 <= scale_roundoff)
    alpha = 1.0;
  const Eigen::VectorXd value = point.head(degrees_of_freedom);
  if (!validate(value, alpha)) return false;
  *candidate = value;
  *candidate_scale = alpha;
  return true;
}

inline bool RecoverScaleTaskWithFeasibleRaySeed(
    const Eigen::MatrixXd &jacobian, const Eigen::VectorXd &target,
    const Eigen::VectorXd &bias, const Eigen::MatrixXd &protected_tasks,
    const Eigen::VectorXd &protected_values,
    const Eigen::MatrixXd &coefficients, const Eigen::VectorXd &lower,
    const Eigen::VectorXd &upper, const Eigen::VectorXd &seed,
    const VelocitySolverConfig &config, Eigen::VectorXd *velocity,
    double *scale, double seed_scale = 0.0) {
  constexpr double tolerance = 1e-10;
  const Eigen::Index n = seed.size();
  if (n == 0 || !seed.allFinite() || !jacobian.allFinite() ||
      !target.allFinite() || !bias.allFinite() ||
      !protected_tasks.allFinite() || !coefficients.allFinite() ||
      !lower.allFinite() || !upper.allFinite() ||
      jacobian.cols() != n ||
      protected_tasks.cols() != n || coefficients.cols() != n ||
      jacobian.rows() != target.size() || bias.size() != target.size() ||
      protected_tasks.rows() != protected_values.size() ||
      coefficients.rows() != lower.size() || lower.size() != upper.size() ||
      !protected_values.allFinite()) return false;
  if ((lower.array() > upper.array()).any()) return false;
  auto valid_point = [&](const Eigen::VectorXd &point, double alpha) {
    return point.allFinite() && std::isfinite(alpha) && alpha >= 0.0 &&
           alpha <= 1.0 &&
           ComputeMaxLinearConstraintViolation(coefficients, lower, upper,
                                                point) <= tolerance &&
           (jacobian * point + bias - alpha * target).norm() <=
               tolerance * std::max({1.0, bias.norm(), alpha * target.norm()}) &&
           (protected_tasks * point - protected_values).norm() <=
               tolerance * std::max(1.0, protected_values.norm());
  };
  if (valid_point(seed, seed_scale)) {
    // Keep the established feasible-seed search unchanged.
    return RecoverZeroScaleTask(jacobian, target, bias, protected_tasks,
                                coefficients, lower, upper, seed, velocity,
                                scale, seed_scale);
  }

  const Eigen::Index m = coefficients.rows(), r = jacobian.rows();
  const Eigen::Index p = protected_tasks.rows();
  Eigen::MatrixXd augmented = Eigen::MatrixXd::Zero(m + r + p + 1, n + 1);
  augmented.topLeftCorner(m, n) = coefficients;
  augmented.block(m, 0, r, n) = jacobian;
  augmented.block(m, n, r, 1) = -target;
  augmented.block(m + r, 0, p, n) = protected_tasks;
  augmented(m + r + p, n) = 1.0;
  Eigen::VectorXd augmented_lower(m + r + p + 1);
  Eigen::VectorXd augmented_upper(m + r + p + 1);
  augmented_lower << lower, -bias, protected_values, 0.0;
  augmented_upper << upper, -bias, protected_values, 1.0;
  Eigen::VectorXd phase_point;
  double candidate_scale = 0.0;
  Eigen::VectorXd candidate;
  auto accept_phase_seed = [&](const Eigen::VectorXd &point,
                               bool project_scale = false) {
    return AcceptAffineRayPhaseSeed(point, n, project_scale, valid_point,
                                   &candidate, &candidate_scale);
  };
  const auto phase_status = FindHardConstraintFeasibleWitness(
      augmented, augmented_lower, augmented_upper, config, &phase_point);
  if (phase_status == HardConstraintFeasibilityStatus::kProvenInfeasible)
    return false;
  if (phase_status != HardConstraintFeasibilityStatus::kFeasible ||
      !(accept_phase_seed(phase_point) || accept_phase_seed(phase_point, true))) {
    // A caller-tolerance witness may fail the stricter ray/scale audit. Retry
    // construction once; an already valid seed retains its original search.
    auto phase_config = config;
    phase_config.epsilon = std::min(config.epsilon, tolerance * 0.01);
    if (phase_config.epsilon >= config.epsilon) return false;
    Eigen::VectorXd refined_phase_point;
    if (FindHardConstraintFeasibleWitness(
            augmented, augmented_lower, augmented_upper, phase_config,
            &refined_phase_point) != HardConstraintFeasibilityStatus::kFeasible ||
        !accept_phase_seed(refined_phase_point)) return false;
  }
  if (candidate_scale < 1.0) {
    const Eigen::VectorXd feasible_seed = candidate;
    (void)RecoverZeroScaleTask(
        jacobian, target, bias, protected_tasks, coefficients, lower, upper,
        feasible_seed, &candidate, &candidate_scale, candidate_scale);
  }
  // An invalid original trial scale is not a certified lower bound. Any
  // positive feasible ray is useful; alpha=1 already reaches the scale cap.
  if (candidate_scale <= tolerance || !valid_point(candidate, candidate_scale))
    return false;
  *velocity = std::move(candidate);
  *scale = candidate_scale;
  return true;
}

// Recover a failed equality-set search from a feasible primal point.
// Successful legacy results do not enter this path.
inline bool RecoverFeasibleWeightedQuadratic(
    const Eigen::MatrixXd &H, const Eigen::VectorXd &g,
    const Eigen::MatrixXd &C, const Eigen::VectorXd &lower,
    const Eigen::VectorXd &upper, const VelocitySolverConfig &config,
    Eigen::VectorXd *solution, unsigned int *iterations_out,
    const Eigen::VectorXd *initial_solution = nullptr,
    bool allow_certified_checkpoint = false) {
  *iterations_out = 0;
  if (!H.allFinite() || !g.allFinite() || !C.allFinite()) return false;
  Eigen::VectorXd v;
  if (initial_solution != nullptr) {
    if (initial_solution->size() != H.rows() ||
        !initial_solution->allFinite() ||
        ComputeMaxLinearConstraintViolation(C, lower, upper, *initial_solution) >
            config.epsilon) return false;
    v = *initial_solution;
  } else if (FindHardConstraintFeasibleWitness(C, lower, upper, config, &v) !=
             HardConstraintFeasibilityStatus::kFeasible) {
    return false;
  }
  const Eigen::Index nv = H.rows();
  std::vector<Eigen::RowVectorXd> inequality_rows, equality_rows;
  std::vector<double> bounds;
  auto append_inequality = [&](const Eigen::RowVectorXd &row, double bound) {
    const double norm = row.norm();
    if (norm == 0.0) return bound >= -config.epsilon;
    inequality_rows.push_back(row / norm);
    bounds.push_back(bound / norm);
    return true;
  };
  for (Eigen::Index row = 0; row < C.rows(); ++row) {
    if (std::isfinite(lower[row]) && std::isfinite(upper[row]) &&
        lower[row] == upper[row]) {
      if (C.row(row).norm() > 0.0)
        equality_rows.push_back(C.row(row) / C.row(row).norm());
      continue;
    }
    if (std::isfinite(upper[row]) &&
        !append_inequality(C.row(row), upper[row])) return false;
    if (std::isfinite(lower[row]) &&
        !append_inequality(-C.row(row), -lower[row])) return false;
  }
  Eigen::MatrixXd E(equality_rows.size(), nv);
  for (size_t row = 0; row < equality_rows.size(); ++row) E.row(row) = equality_rows[row];
  if (E.rows() > 0) {
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(E, Eigen::ComputeThinU);
    svd.setThreshold(1e-12);
    E = (svd.matrixU().leftCols(svd.rank()).transpose() * E).eval();
    for (Eigen::Index row = 0; row < E.rows(); ++row) E.row(row).normalize();
  }
  std::vector<size_t> active;
  std::optional<Eigen::VectorXd> certified_checkpoint;
  unsigned int attempted_iterations = 0;
  auto return_validated_checkpoint = [&]() {
    if (!certified_checkpoint ||
        !certified_checkpoint->allFinite() ||
        !(C * *certified_checkpoint).allFinite() ||
        ComputeMaxLinearConstraintViolation(C, lower, upper,
                                            *certified_checkpoint) > config.epsilon)
      return false;
    *solution = *certified_checkpoint;
    *iterations_out = attempted_iterations;
    return true;
  };
  const size_t limit = std::max<size_t>(config.iteration_limit,
                                      4 * inequality_rows.size() + 40);
  for (size_t iteration = 0; iteration < limit; ++iteration) {
    attempted_iterations = static_cast<unsigned int>(iteration + 1);
    *iterations_out = attempted_iterations;
    if (ComputeMaxLinearConstraintViolation(C, lower, upper, v) > config.epsilon)
      return return_validated_checkpoint();
    Eigen::MatrixXd basis(E.rows() + active.size(), nv);
    basis.topRows(E.rows()) = E;
    for (size_t row = 0; row < active.size(); ++row)
      basis.row(E.rows() + row) = inequality_rows[active[row]];
    Eigen::MatrixXd free = Eigen::MatrixXd::Identity(nv, nv);
    if (basis.rows() > 0) {
      Eigen::JacobiSVD<Eigen::MatrixXd> svd(basis, Eigen::ComputeFullV);
      svd.setThreshold(1e-12);
      free = svd.matrixV().rightCols(nv - svd.rank());
    }
    const Eigen::VectorXd gradient = H * v - g;
    Eigen::VectorXd direction = Eigen::VectorXd::Zero(nv);
    if (free.cols() > 0) {
      const Eigen::MatrixXd reduced_H = free.transpose() * H * free;
      direction = -free * reduced_H.ldlt().solve(free.transpose() * gradient);
    }
    if (!direction.allFinite()) return return_validated_checkpoint();
    if (allow_certified_checkpoint) {
      // The caller explicitly permits a roundoff certificate in its already
      // failed numerical recovery. Other recovery paths retain
      // their original search and task-null tie semantics.
      const Eigen::VectorXd reduced_gradient = free.transpose() * gradient;
      const Eigen::VectorXd reduced_direction = free.transpose() * direction;
      const double predicted_decrease =
          -0.5 * reduced_gradient.dot(reduced_direction);
      const double state_scale = std::max(1.0, v.norm());
      const double model_scale = std::max(
          H.norm() * state_scale * state_scale, g.norm() * state_scale);
      const double decrease_tolerance =
          64.0 * std::numeric_limits<double>::epsilon() * model_scale;
      if (std::isfinite(predicted_decrease) && predicted_decrease >= 0.0 &&
          std::isfinite(decrease_tolerance) &&
          predicted_decrease <= decrease_tolerance) {
        Eigen::VectorXd checkpoint_lambda = Eigen::VectorXd::Zero(basis.rows());
        if (basis.rows() > 0)
          checkpoint_lambda = basis.transpose().completeOrthogonalDecomposition().solve(-gradient);
        const double checkpoint_minimum = active.empty() ? 0.0
            : checkpoint_lambda.tail(active.size()).minCoeff();
        const double checkpoint_stationarity =
            (gradient + basis.transpose() * checkpoint_lambda).norm();
        bool checkpoint_active_tight = true;
        for (const size_t row : active) {
          if (std::abs(inequality_rows[row].dot(v) - bounds[row]) > config.epsilon) {
            checkpoint_active_tight = false;
            break;
          }
        }
        if (checkpoint_active_tight && checkpoint_lambda.allFinite() &&
            gradient.allFinite() && std::isfinite(gradient.norm()) &&
            std::isfinite(checkpoint_stationarity) &&
            checkpoint_minimum >= -std::max(1e-10, config.epsilon) &&
            checkpoint_stationarity <= std::max(1e-8, 10.0 * config.epsilon) *
                                           std::max(1.0, gradient.norm())) {
          certified_checkpoint = v;
          return return_validated_checkpoint();
        }
      }
    }
    if (direction.norm() <= 1e-9) {
      Eigen::VectorXd lambda = Eigen::VectorXd::Zero(basis.rows());
      if (basis.rows() > 0)
        lambda = basis.transpose().completeOrthogonalDecomposition().solve(-gradient);
      if (!lambda.allFinite()) return return_validated_checkpoint();
      Eigen::Index remove = -1;
      const double minimum = active.empty() ? 0.0 : lambda.tail(active.size()).minCoeff(&remove);
      // A negative inequality multiplier can release a trial pin even when an
      // ill-conditioned working set prevents a stationarity certificate.
      // This continues the bounded feasible search; it does not certify success.
      if (minimum < -std::max(1e-10, config.epsilon)) {
        active.erase(active.begin() + remove);
        continue;
      }
      if ((gradient + basis.transpose() * lambda).norm() >
          std::max(1e-8, 10.0 * config.epsilon) * std::max(1.0, gradient.norm()))
        return return_validated_checkpoint();
      *solution = std::move(v);
      *iterations_out = static_cast<unsigned int>(iteration + 1);
      return true;
    }
    double step = 1.0;
    size_t blocker = inequality_rows.size();
    for (size_t row = 0; row < inequality_rows.size(); ++row) {
      if (std::find(active.begin(), active.end(), row) != active.end()) continue;
      const double rate = inequality_rows[row].dot(direction);
      if (rate <= 1e-12) continue;
      const double candidate = std::max(0.0, (bounds[row] - inequality_rows[row].dot(v)) / rate);
      if (candidate < step) { step = candidate; blocker = row; }
    }
    v.noalias() += step * direction;
    if (blocker < inequality_rows.size()) active.push_back(blocker);
  }
  return return_validated_checkpoint();
}

template <typename Decomposition>
inline bool HasUsableSvdFactors(const Decomposition &decomposition) {
  // Eigen 3.3, used by the manylinux wheel image, has no SVD::info().
#if EIGEN_WORLD_VERSION > 3 || \
    (EIGEN_WORLD_VERSION == 3 && EIGEN_MAJOR_VERSION >= 4)
  if (decomposition.info() != Eigen::Success) return false;
#endif
  return decomposition.singularValues().allFinite() &&
         (!decomposition.computeU() || decomposition.matrixU().allFinite()) &&
         (!decomposition.computeV() || decomposition.matrixV().allFinite());
}

// Owned by one hierarchy invocation: no robot-state invalidation or shared
// mutable state. Reuse requires identical bits, including signed zero.
class ProjectedInverseCache {
 public:
  const Eigen::MatrixXd *Find(const RegularizedInverseConfig &config,
                             const Eigen::MatrixXd &input,
                             bool condition_aware, double *condition) const {
    for (const auto &entry : entries_) {
      if (entry.condition_aware == condition_aware &&
          SameBits(entry.config.epsilon, config.epsilon) &&
          SameBits(entry.config.regularization_factor,
                   config.regularization_factor) &&
          entry.input.rows() == input.rows() &&
          entry.input.cols() == input.cols() &&
          (input.size() == 0 ||
           std::memcmp(entry.input.data(), input.data(),
                       sizeof(double) * input.size()) == 0)) {
        if (condition) *condition = entry.condition;
        return &entry.inverse;
      }
    }
    return nullptr;
  }

  void Insert(const RegularizedInverseConfig &config,
              const Eigen::MatrixXd &input, bool condition_aware,
              const Eigen::MatrixXd &inverse, double condition) {
    // Bound storage even for large dense systems; infinite rank diagnostics
    // are legitimate, whereas nonfinite coefficients must never be reused.
    if (input.size() > kMaxMatrixCoefficients ||
        inverse.size() > kMaxMatrixCoefficients || !input.allFinite() ||
        !inverse.allFinite() || !std::isfinite(config.epsilon) ||
        !std::isfinite(config.regularization_factor) || std::isnan(condition))
      return;
    if (entries_.size() == kMaxEntries) entries_.erase(entries_.begin());
    entries_.push_back({config, input, inverse, condition_aware, condition});
  }

 private:
  static bool SameBits(double left, double right) {
    return std::memcmp(&left, &right, sizeof(double)) == 0;
  }
  static constexpr size_t kMaxEntries = 8;
  static constexpr Eigen::Index kMaxMatrixCoefficients = 4096;
  struct Entry {
    RegularizedInverseConfig config;
    Eigen::MatrixXd input, inverse;
    bool condition_aware;
    double condition;
  };
  std::vector<Entry> entries_;
};

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
  // Some finite rank-deficient inputs produce nonfinite BDC singular vectors
  // even when Eigen reports Success. Retry the factorization, preserving the
  // same cutoff and damping law below, before using any of those factors.
  std::optional<Eigen::JacobiSVD<MatrixDouble>> robust_svd;
  if (!detail::HasUsableSvdFactors(svd_decomp)) {
    robust_svd.emplace(input_matrix, svd_options);
    if (!detail::HasUsableSvdFactors(*robust_svd)) {
      if (condition_number_out) {
        *condition_number_out = std::numeric_limits<double>::infinity();
      }
      return MatrixDouble::Constant(
          input_matrix.cols(), input_matrix.rows(),
          std::numeric_limits<double>::quiet_NaN());
    }
  }
  const auto &sigma_values = robust_svd ? robust_svd->singularValues()
                                        : svd_decomp.singularValues();
  const auto &left_singular_vectors = robust_svd ? robust_svd->matrixU()
                                               : svd_decomp.matrixU();

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
      const auto &right_singular_vectors = robust_svd ? robust_svd->matrixV()
                                                     : svd_decomp.matrixV();
      return right_singular_vectors * filtered_inverse.asDiagonal() *
             left_singular_vectors.transpose();
    }

    if (enable_condition_aware_svd_fallback && is_tall) {
      const Eigen::ArrayXd filtered_inverse =
          sigma_values.array() /
          (sigma_values.array().square() + global_regularization +
           per_sv_damping);
      const auto &right_singular_vectors = robust_svd ? robust_svd->matrixV()
                                                     : svd_decomp.matrixV();
      return right_singular_vectors * filtered_inverse.matrix().asDiagonal() *
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
      // A tall matrix's full U has more columns than singular values.
      // Only the corresponding singular directions have damping entries.
      const auto damped_left_singular_vectors =
          left_singular_vectors.leftCols(sigma_values.size());
      regularized_gram.noalias() +=
          damped_left_singular_vectors *
          per_sv_damping.matrix().asDiagonal() *
          damped_left_singular_vectors.transpose();
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
  const Eigen::JacobiSVD<MatrixDouble> fallback_svd(
      input_matrix, Eigen::ComputeThinU | Eigen::ComputeThinV);
  if (!detail::HasUsableSvdFactors(fallback_svd)) {
    return MatrixDouble::Constant(
        input_matrix.cols(), input_matrix.rows(),
        std::numeric_limits<double>::quiet_NaN());
  }
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
  const double active_constraint_rank_tolerance = solver_config.epsilon;
  const double objective_equality_tolerance =
      std::max(std::max(solver_config.epsilon,
                        solver_config.precision_threshold),
               execution_options.objective_equality_tolerance);
  detail::ProjectedInverseCache projected_inverse_cache;
  const auto compute_projected_objective_inverse =
      [&](const RegularizedInverseConfig &regularization_config,
          const Eigen::MatrixXd &objective_matrix,
          const Eigen::MatrixXd &projector, double *condition_number) {
        const Eigen::MatrixXd input = objective_matrix * projector;
        Eigen::MatrixXd inverse;
        const auto *cached = preserve_legacy_velocity_behavior ? nullptr
            : projected_inverse_cache.Find(regularization_config, input, true,
                                           condition_number);
        if (cached) {
          inverse = *cached;
        } else {
          double condition = 1.0;
          inverse = linalg::ComputeRegularizedInverse(
              regularization_config, input, &condition,
              !preserve_legacy_velocity_behavior);
          if (condition_number) *condition_number = condition;
          if (!preserve_legacy_velocity_behavior)
            projected_inverse_cache.Insert(regularization_config, input, true,
                                           inverse, condition);
        }
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
  const auto degrees_of_freedom = constraint_coefficients.cols();
  auto fail_non_finite = [&](const char *msg) {
    SolverResult out = fail(SolverStatus::kNonFiniteInput, msg);
    out.solution.assign(static_cast<size_t>(degrees_of_freedom), 0.0);
    return out;
  };
  if (!constraint_coefficients.allFinite() || !min_bounds.allFinite() ||
      !max_bounds.allFinite())
    return fail_non_finite(
        "constraints and bounds must contain only finite values");
  if ((min_bounds.array() > max_bounds.array()).any())
    return fail(SolverStatus::kInvalidInput,
                "constraint lower bounds must not exceed upper bounds");

  // Extract system dimensions
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
      return fail_non_finite(
          "objective matrices and vectors must be finite");
    }
  }

  struct MinErrorSuffixEntry {
    size_t first_objective;
    Eigen::VectorXd velocity;
    std::vector<Eigen::VectorXd> prefix_targets;
    std::vector<double> prefix_scales;
    std::vector<TaskSolveMode> prefix_modes;
    std::vector<bool> prefix_fallbacks;
  };
  std::optional<MinErrorSuffixEntry> min_error_suffix_entry;
  std::vector<Eigen::VectorXd> completed_objective_images;
  if (!preserve_legacy_velocity_behavior)
    completed_objective_images.resize(num_objectives);

  auto can_retain_lower_task_checkpoint = [&](
      const Eigen::VectorXd &checkpoint,
      const std::vector<Eigen::VectorXd> &completed_images,
      size_t first_unoptimized) {
    return !preserve_legacy_velocity_behavior &&
        detail::CanRetainBestEffortLowerTaskCheckpoint(
            execution_options.allow_best_effort_lower_tasks, checkpoint,
            completed_images, first_unoptimized, objective_matrices,
            scalable_objective_targets, affine_objective_biases,
            objective_configs, constraint_coefficients, min_bounds, max_bounds,
            solver_config.epsilon, objective_equality_tolerance);
  };

  // Recover only an explicitly minimum-error suffix. Its completed prefix has
  // already produced a feasible command and is protected as exact task images;
  // SCALE objectives retain their affine rays and achieved scales.
  auto retry_invalid_min_error_suffix = [&]() -> std::optional<SolverResult> {
    if (!min_error_suffix_entry) return std::nullopt;
    const auto &entry = *min_error_suffix_entry;
    Eigen::Index prefix_rows = 0;
    for (size_t index = 0; index < entry.first_objective; ++index)
      prefix_rows += objective_matrices[index].rows();
    Eigen::MatrixXd recovery_C(constraint_coefficients.rows() + prefix_rows,
                               degrees_of_freedom);
    recovery_C.topRows(constraint_coefficients.rows()) = constraint_coefficients;
    Eigen::VectorXd recovery_lower(min_bounds.size() + prefix_rows);
    Eigen::VectorXd recovery_upper(max_bounds.size() + prefix_rows);
    recovery_lower.head(min_bounds.size()) = min_bounds;
    recovery_upper.head(max_bounds.size()) = max_bounds;
    Eigen::Index offset = constraint_coefficients.rows();
    for (size_t index = 0; index < entry.first_objective; ++index) {
      const auto rows = objective_matrices[index].rows();
      recovery_C.middleRows(offset, rows) = objective_matrices[index];
      recovery_lower.segment(offset, rows) = entry.prefix_targets[index];
      recovery_upper.segment(offset, rows) = entry.prefix_targets[index];
      offset += rows;
    }
    Eigen::VectorXd recovered = entry.velocity;
    if (detail::ComputeMaxLinearConstraintViolation(
            recovery_C, recovery_lower, recovery_upper, recovered) >
        solver_config.epsilon) return std::nullopt;
    unsigned int iterations = 0;
    std::vector<Eigen::VectorXd> achieved_suffix;
    std::optional<size_t> first_unoptimized;
    for (size_t index = entry.first_objective; index < num_objectives; ++index) {
      const auto &matrix = objective_matrices[index];
      const Eigen::VectorXd target =
          scalable_objective_targets[index] - affine_objective_biases[index];
      Eigen::MatrixXd hessian = matrix.transpose() * matrix;
      const double diagonal_scale = hessian.diagonal().cwiseAbs().maxCoeff();
      // Match the existing numerical MIN_ERROR recovery's null-space tie-break.
      hessian.diagonal().array() +=
          1e-12 * (diagonal_scale > 0.0 ? diagonal_scale : 1.0);
      Eigen::VectorXd candidate;
      unsigned int task_iterations = 0;
      if (!detail::RecoverFeasibleWeightedQuadratic(
              hessian, matrix.transpose() * target, recovery_C, recovery_lower,
              recovery_upper, solver_config, &candidate, &task_iterations,
              &recovered, true)) {
        std::vector<Eigen::VectorXd> completed_images = entry.prefix_targets;
        completed_images.insert(completed_images.end(), achieved_suffix.begin(),
                                achieved_suffix.end());
        if (!can_retain_lower_task_checkpoint(recovered, completed_images, index))
          return std::nullopt;
        iterations += task_iterations;
        first_unoptimized = index;
        break;
      }
      recovered = std::move(candidate);
      iterations += task_iterations;
      achieved_suffix.push_back(matrix * recovered);
      if (index + 1 < num_objectives) {
        const auto old_rows = recovery_C.rows();
        recovery_C.conservativeResize(old_rows + matrix.rows(), degrees_of_freedom);
        recovery_C.bottomRows(matrix.rows()) = matrix;
        recovery_lower.conservativeResize(old_rows + matrix.rows());
        recovery_upper.conservativeResize(old_rows + matrix.rows());
        recovery_lower.tail(matrix.rows()) = achieved_suffix.back();
        recovery_upper.tail(matrix.rows()) = achieved_suffix.back();
      }
    }
    if (!recovered.allFinite() || detail::ComputeMaxLinearConstraintViolation(
            recovery_C, recovery_lower, recovery_upper, recovered) >
        solver_config.epsilon) return std::nullopt;
    const size_t protected_count = first_unoptimized.value_or(num_objectives - 1);
    for (size_t index = 0; index < protected_count; ++index) {
      const Eigen::VectorXd &image = index < entry.first_objective
          ? entry.prefix_targets[index]
          : achieved_suffix[index - entry.first_objective];
      if ((objective_matrices[index] * recovered - image).norm() >
          objective_equality_tolerance * std::max(1.0, image.norm()))
        return std::nullopt;
    }
    SolverResult out{};
    out.solution.assign(recovered.data(), recovered.data() + recovered.size());
    out.status = SolverStatus::kSuccess;
    out.computation_time_ms = get_elapsed_ms();
    out.iterations = iterations;
    out.task_scales = entry.prefix_scales;
    out.task_modes_effective = entry.prefix_modes;
    out.task_used_fallback = entry.prefix_fallbacks;
    out.task_scales.resize(num_objectives, -1.0);
    out.task_modes_effective.resize(num_objectives, TaskSolveMode::kMinError);
    out.task_used_fallback.resize(num_objectives, false);
    if (first_unoptimized) {
      for (size_t index = *first_unoptimized; index < num_objectives; ++index)
        out.task_used_fallback[index] = true;
    }
    double condition = 1.0;
    for (size_t index = 0; index < num_objectives; ++index) {
      const Eigen::VectorXd target =
          scalable_objective_targets[index] - affine_objective_biases[index];
      out.task_errors.push_back((objective_matrices[index] * recovered - target).norm());
      double task_condition = 1.0;
      (void)linalg::ComputeRegularizedInverse(solver_config.regularization_config,
          objective_matrices[index], &task_condition, true);
      condition = std::max(condition, task_condition);
    }
    out.final_error = out.task_errors.front();
    out.condition_number = condition;
    // Rank-deficient tasks can legitimately report infinite conditioning.
    // The retained finite command and every protected image/hard row were
    // validated above; conditioning alone does not invalidate that witness.
    out.status_message = first_unoptimized
        ? "best-effort lower-task numerical recovery; trailing objectives not optimized"
        : "hard-constrained trailing minimum-error numerical recovery";
    return out;
  };

  // Recover failed explicit MIN_ERROR objectives under the same hard intervals.
  // A trial saturation set can be inconsistent even when the inequalities are
  // feasible. The feasible-primal search can release those trial pins, while
  // exact achieved-image rows retain each earlier objective's hierarchy.
  Eigen::VectorXd hard_constraint_witness;
  bool full_hard_constraint_witness_available = false;
  auto retry_invalid_min_error = [&](const Eigen::VectorXd &failed_velocity,
                                     bool direct_primal = false)
      -> std::optional<SolverResult> {
    const bool all_explicit_min_error =
        objective_configs.size() == num_objectives &&
        std::all_of(objective_configs.begin(), objective_configs.end(),
                    [](const ObjectiveSolveConfig &objective) {
                      return objective.solve_mode == TaskSolveMode::kMinError;
                    });
    const bool has_relaxable_rows = enhanced_mode &&
        (softening_factors.array() > 1.0).any();
    // A satisfied primary hold must not suppress recovery for a moving lower
    // objective. Its achieved image is protected by the same hierarchy rows.
    bool has_nonzero_target = false;
    for (size_t index = 0; index < num_objectives; ++index) {
      const Eigen::VectorXd target = preserve_legacy_velocity_behavior
          ? scalable_objective_targets[index]
          : scalable_objective_targets[index] - affine_objective_biases[index];
      has_nonzero_target = has_nonzero_target ||
          target.norm() > solver_config.precision_threshold;
    }
    if (has_relaxable_rows ||
        !all_explicit_min_error || !failed_velocity.allFinite() ||
        !has_nonzero_target ||
        (!direct_primal && detail::ComputeMaxLinearConstraintViolation(
            constraint_coefficients, min_bounds, max_bounds, failed_velocity) <=
            solver_config.epsilon)) {
      return std::nullopt;
    }
    if (direct_primal &&
        (preserve_legacy_velocity_behavior ||
         !execution_options.prefer_direct_min_error_qp ||
         !full_hard_constraint_witness_available ||
         hard_constraint_witness.size() != degrees_of_freedom ||
         !hard_constraint_witness.allFinite() ||
         !(constraint_coefficients * hard_constraint_witness).allFinite() ||
         detail::ComputeMaxLinearConstraintViolation(
             constraint_coefficients, min_bounds, max_bounds,
             hard_constraint_witness) > solver_config.epsilon)) {
      return std::nullopt;
    }
    if (preserve_legacy_velocity_behavior) {
      std::vector<Eigen::VectorXd> zero_biases;
      zero_biases.reserve(num_objectives);
      for (const auto &target : scalable_objective_targets) {
        zero_biases.push_back(Eigen::VectorXd::Zero(target.size()));
      }
      SolverResult constrained = solveHierarchicalLinearSystemEigen(
          scalable_objective_targets, zero_biases, objective_matrices,
          constraint_coefficients, min_bounds, max_bounds, solver_config,
          objective_configs, nullptr);
      if (constrained.status == SolverStatus::kSuccess) {
        constrained.computation_time_ms = get_elapsed_ms();
        return constrained;
      }
    }

    Eigen::MatrixXd recovery_C = constraint_coefficients;
    Eigen::VectorXd recovery_lower = min_bounds;
    Eigen::VectorXd recovery_upper = max_bounds;
    Eigen::VectorXd recovered;
    std::vector<Eigen::VectorXd> achieved_images;
    std::optional<size_t> first_unoptimized;
    unsigned int recovery_iterations = 0;
    double recovery_condition = 1.0;
    for (size_t index = 0; index < num_objectives; ++index) {
      const auto &matrix = objective_matrices[index];
      const Eigen::VectorXd target = preserve_legacy_velocity_behavior
          ? scalable_objective_targets[index]
          : scalable_objective_targets[index] - affine_objective_biases[index];
      Eigen::MatrixXd hessian = matrix.transpose() * matrix;
      const Eigen::VectorXd gradient_target = matrix.transpose() * target;
      // A tiny scale-relative ridge resolves task-null ties without applying
      // the advisor's larger epsilon-squared preference to task directions.
      const double diagonal_scale = hessian.diagonal().cwiseAbs().maxCoeff();
      hessian.diagonal().array() +=
          1e-12 * (diagonal_scale > 0.0 ? diagonal_scale : 1.0);
      unsigned int iterations = 0;
      Eigen::VectorXd candidate;
      if (!detail::RecoverFeasibleWeightedQuadratic(
              hessian, gradient_target, recovery_C, recovery_lower,
              recovery_upper, solver_config, &candidate, &iterations,
              index == 0 ? (direct_primal ? &hard_constraint_witness : nullptr)
                         : &recovered,
              !direct_primal)) {
        if (direct_primal ||
            !can_retain_lower_task_checkpoint(recovered, achieved_images, index))
          return std::nullopt;
        recovery_iterations += iterations;
        first_unoptimized = index;
        break;
      }
      recovered = std::move(candidate);
      recovery_iterations += iterations;
      double condition = 1.0;
      (void)linalg::ComputeRegularizedInverse(
          solver_config.regularization_config, matrix, &condition,
          !preserve_legacy_velocity_behavior);
      recovery_condition = std::max(recovery_condition, condition);
      achieved_images.push_back(matrix * recovered);
      if (index + 1 < num_objectives) {
        const Eigen::Index old_rows = recovery_C.rows();
        recovery_C.conservativeResize(old_rows + matrix.rows(),
                                      degrees_of_freedom);
        recovery_C.bottomRows(matrix.rows()) = matrix;
        recovery_lower.conservativeResize(old_rows + matrix.rows());
        recovery_upper.conservativeResize(old_rows + matrix.rows());
        recovery_lower.tail(matrix.rows()) = achieved_images.back();
        recovery_upper.tail(matrix.rows()) = achieved_images.back();
      }
    }
    if (direct_primal &&
        (!recovered.allFinite() || !(recovery_C * recovered).allFinite()))
      return std::nullopt;
    if (detail::ComputeMaxLinearConstraintViolation(
            recovery_C, recovery_lower, recovery_upper, recovered) >
        solver_config.epsilon) return std::nullopt;
    const size_t protected_count = first_unoptimized.value_or(num_objectives - 1);
    for (size_t index = 0; index < protected_count; ++index) {
      if ((objective_matrices[index] * recovered - achieved_images[index]).norm() >
          objective_equality_tolerance *
              std::max(1.0, achieved_images[index].norm())) return std::nullopt;
    }
    SolverResult out{};
    out.solution.assign(recovered.data(), recovered.data() + recovered.size());
    out.status = SolverStatus::kSuccess;
    out.computation_time_ms = get_elapsed_ms();
    out.iterations = recovery_iterations;
    out.task_scales.assign(num_objectives, -1.0);
    out.task_modes_effective.assign(num_objectives, TaskSolveMode::kMinError);
    out.task_used_fallback.assign(num_objectives, false);
    if (first_unoptimized) {
      for (size_t index = *first_unoptimized; index < num_objectives; ++index)
        out.task_used_fallback[index] = true;
    }
    for (size_t index = 0; index < num_objectives; ++index) {
      const Eigen::VectorXd target = preserve_legacy_velocity_behavior
          ? scalable_objective_targets[index]
          : scalable_objective_targets[index] - affine_objective_biases[index];
      out.task_errors.push_back((objective_matrices[index] * recovered - target).norm());
    }
    out.final_error = out.task_errors.front();
    if (direct_primal &&
        !std::all_of(out.task_errors.begin(), out.task_errors.end(),
                     [](double value) { return std::isfinite(value); }))
      return std::nullopt;
    out.status_message = direct_primal
        ? "opt-in direct hard-constrained minimum-error QP"
        : (first_unoptimized
            ? "best-effort lower-task numerical recovery; trailing objectives not optimized"
            : "hard-constrained minimum-error numerical recovery");
    out.condition_number = recovery_condition;
    // Conditioning is diagnostic; rank deficiency does not invalidate the
    // independently checked best-effort command or its protected images.
    return out;
  };

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

  // Legacy iterations cannot repair an exactly zero immutable row. Certify
  // that contradiction before the iteration-limit exit bypasses the final
  // feasibility audit. Compare coefficients directly: a norm can underflow
  // for a nonzero row, whose original numerical path must remain unchanged.
  if (preserve_legacy_velocity_behavior) {
    for (Eigen::Index row = 0; row < total_constraints; ++row) {
      if ((!enhanced_mode || softening_factors(row) <= 1.0) &&
          (constraint_coefficients.row(row).array() == 0.0).all() &&
          (min_bounds(row) > solver_config.epsilon ||
           max_bounds(row) < -solver_config.epsilon)) {
        return phase_one_failure(
            SolverStatus::kInfeasible,
            "immutable zero constraint row excludes zero beyond tolerance");
      }
    }
  }

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
      !execution_options.hard_constraints_prevalidated_feasible &&
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
  const Eigen::VectorXd *initial_witness =
      execution_options.initial_hard_constraint_witness;
  const bool reuse_hard_constraint_witness =
      run_hard_constraint_phase_one && initial_witness != nullptr &&
      initial_witness->size() == degrees_of_freedom &&
      initial_witness->allFinite() &&
      (phase_one_coefficients * *initial_witness).allFinite() &&
      detail::ComputeMaxLinearConstraintViolation(
          phase_one_coefficients, phase_one_min_bounds, phase_one_max_bounds,
          *initial_witness) <= solver_config.epsilon;
  if (reuse_hard_constraint_witness)
    hard_constraint_witness = *initial_witness;
  const auto hard_constraint_feasibility =
      reuse_hard_constraint_witness
          ? detail::HardConstraintFeasibilityStatus::kFeasible
          : run_hard_constraint_phase_one
          ? detail::FindHardConstraintFeasibleWitness(
                phase_one_coefficients, phase_one_min_bounds,
                phase_one_max_bounds, solver_config, &hard_constraint_witness)
          : detail::HardConstraintFeasibilityStatus::kUnknown;
  full_hard_constraint_witness_available =
      !preserve_legacy_velocity_behavior && run_hard_constraint_phase_one &&
      immutable_constraint_rows.size() == static_cast<size_t>(total_constraints) &&
      hard_constraint_feasibility ==
          detail::HardConstraintFeasibilityStatus::kFeasible;
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

  if (execution_options.prefer_direct_min_error_qp &&
      full_hard_constraint_witness_available) {
    if (auto direct = retry_invalid_min_error(hard_constraint_witness, true))
      return *direct;
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
      // A feasible direct step is scale 1 only when it meets the task. An
      // all-zero Jacobian lies inside every finite box, so the legacy bypass
      // would otherwise report scale 1 for a target it cannot produce.
      const bool objective_satisfied =
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
  // Zero need not satisfy a joint box (for example, a joint must move inward
  // from an integration margin). Start from a feasible task-null velocity when
  // the required box motion does not affect the primary objective. Later
  // objectives use their residual at this seed during normal optimization.
  // Otherwise retain the existing active-set recovery; never assume the box proves
  // feasibility of the remaining coupled constraints.
  if (total_constraints >= degrees_of_freedom &&
      ((min_bounds.head(degrees_of_freedom).array() > 0.0).any() ||
       (max_bounds.head(degrees_of_freedom).array() < 0.0).any()) &&
      constraint_coefficients.topRows(degrees_of_freedom)
          .isApprox(Eigen::MatrixXd::Identity(degrees_of_freedom,
                                            degrees_of_freedom),
                    solver_config.precision_threshold)) {
    Eigen::VectorXd feasible_seed = Eigen::VectorXd::Zero(degrees_of_freedom);
    for (Eigen::Index joint = 0; joint < degrees_of_freedom; ++joint) {
      feasible_seed(joint) =
          std::clamp(0.0, min_bounds(joint), max_bounds(joint));
    }
    if (feasible_seed.squaredNorm() > 0.0) {
      const bool preserves_primary =
          (objective_matrices.front() * feasible_seed).norm() <=
          solver_config.precision_threshold;
      if (preserves_primary &&
          detail::ComputeMaxLinearConstraintViolation(
              constraint_coefficients, min_bounds, max_bounds, feasible_seed) <=
              solver_config.epsilon) {
        velocity_solution = feasible_seed;
      }
    }
  }
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
    // Row count is not the ESNS rank budget. A zero row, a repeated row, or a
    // linear combination of earlier rows that agrees with the target is the
    // same task; counting it as lost rank aborts a solvable objective.
    Eigen::ColPivHouseholderQR<Eigen::MatrixXd> task_rank_qr(current_jacobian);
    task_rank_qr.setThreshold(solver_config.epsilon);
    const Eigen::Index independent_task_rank = task_rank_qr.rank();
    Eigen::MatrixXd jacobian_and_target(current_jacobian.rows(),
                                        current_jacobian.cols() + 1);
    jacobian_and_target.leftCols(current_jacobian.cols()) = current_jacobian;
    jacobian_and_target.col(current_jacobian.cols()) = current_full_target;
    Eigen::ColPivHouseholderQR<Eigen::MatrixXd> target_rank_qr(
        jacobian_and_target);
    target_rank_qr.setThreshold(solver_config.epsilon);
    const bool redundant_rows_consistent =
        target_rank_qr.rank() == independent_task_rank;
    // Only an explicit zero, repeated, or linearly dependent row changes the
    // ESNS rank budget. A column singularity, whose rows are not combinations
    // of one another, keeps the original row-count check.
    const double jacobian_scale = std::max(1.0, current_jacobian.norm());
    const double row_tolerance = solver_config.epsilon * jacobian_scale;
    auto row_explicitly_redundant = [&](Eigen::Index row) -> bool {
      const Eigen::VectorXd row_vector = current_jacobian.row(row).transpose();
      if (row_vector.norm() <= row_tolerance) {
        return true;
      }
      if (row == 0) {
        return false;
      }
      const Eigen::MatrixXd earlier_rows =
          current_jacobian.topRows(row).transpose();
      Eigen::ColPivHouseholderQR<Eigen::MatrixXd> earlier_qr(earlier_rows);
      earlier_qr.setThreshold(solver_config.epsilon);
      const Eigen::VectorXd coefficients = earlier_qr.solve(row_vector);
      return (earlier_rows * coefficients - row_vector).norm() <= row_tolerance;
    };
    Eigen::Index explicit_redundant_row_count = 0;
    for (Eigen::Index row = 0; row < target_dimension; ++row) {
      if (row_explicitly_redundant(row)) {
        ++explicit_redundant_row_count;
      }
    }
    const bool explicit_row_redundancy =
        explicit_redundant_row_count > 0 &&
        independent_task_rank + explicit_redundant_row_count ==
            target_dimension;
    // An inconsistent extra row still uses the original row-count budget.
    // Only a consistent dependent row is the same solvable task.
    const bool explicit_consistent_redundancy =
        explicit_row_redundancy && redundant_rows_consistent;
    const Eigen::Index task_rank_budget = explicit_consistent_redundancy
                                              ? independent_task_rank
                                              : target_dimension;
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
    if (!preserve_legacy_velocity_behavior && !min_error_suffix_entry &&
        solver_status == SolverStatus::kSuccess && objective_index > 0 &&
        objective_configs.size() == num_objectives &&
        objective_is_min_error &&
        !(enhanced_mode && (softening_factors.array() > 1.0).any()) &&
        std::all_of(objective_configs.begin() + objective_index,
                    objective_configs.end(), [](const ObjectiveSolveConfig &c) {
                      return c.solve_mode == TaskSolveMode::kMinError;
                    }) &&
        previous_velocity.allFinite() &&
        (constraint_coefficients * previous_velocity).allFinite() &&
        detail::ComputeMaxLinearConstraintViolation(
            constraint_coefficients, min_bounds, max_bounds, previous_velocity) <=
            solver_config.epsilon) {
      MinErrorSuffixEntry entry{objective_index, previous_velocity, {}, {}, {}, {}};
      bool prefix_valid = true;
      for (size_t index = 0; index < objective_index; ++index) {
        const double scale = objective_scaling_factors(index);
        Eigen::VectorXd image;
        if (objective_effective_modes[index] == TaskSolveMode::kScale &&
            std::isfinite(scale) && scale >= 0.0 && scale <= 1.0) {
          image = scale * scalable_objective_targets[index] -
                  affine_objective_biases[index];
        } else if (objective_effective_modes[index] == TaskSolveMode::kMinError &&
                   std::isfinite(scale) && scale == -1.0 &&
                   completed_objective_images[index].size() ==
                       objective_matrices[index].rows()) {
          image = completed_objective_images[index];
        } else {
          prefix_valid = false;
          break;
        }
        if (!image.allFinite()) {
          prefix_valid = false;
          break;
        }
        const Eigen::VectorXd residual =
            objective_matrices[index] * previous_velocity - image;
        if (!residual.allFinite() || !std::isfinite(residual.norm()) ||
            !std::isfinite(image.norm()) ||
            residual.norm() > objective_equality_tolerance *
                                  std::max(1.0, image.norm()) ||
            (residual.size() > 0 &&
             residual.cwiseAbs().maxCoeff() > solver_config.epsilon)) {
          prefix_valid = false;
          break;
        }
        entry.prefix_targets.push_back(std::move(image));
        entry.prefix_scales.push_back(scale);
        entry.prefix_modes.push_back(objective_effective_modes[index]);
        entry.prefix_fallbacks.push_back(objective_used_fallback[index]);
      }
      if (prefix_valid) min_error_suffix_entry = std::move(entry);
    }

    // Initialize iteration state for current objective
    auto constrained_projector = previous_null_space;
    auto velocity_scale = 1.0;
    auto optimal_scale = 0.0;
    auto consecutive_zero_scales = 0U;
    auto consecutive_min_error_no_progress = 0U;
    auto constraints_violated = true;
    auto iteration_counter = 0U;
    bool has_feasible_scale_snapshot = false;
    bool attempted_bound_release = false;
    bool attempted_partial_scale_refinement = false;
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
    // A zero request still requires solving when hard rows mandate motion or
    // earlier objectives leave a nonzero residual. A shortcut must preserve
    // this objective's achieved value against lower-priority motion too.
    const bool skip_objective = preserve_legacy_velocity_behavior
                                    ? current_target.template lpNorm<1>() <
                                          solver_config.precision_threshold &&
                                          previous_satisfies_hard_constraints &&
                                          objective_residual_at_previous
                                                  .template lpNorm<1>() <
                                              solver_config.precision_threshold
                                    : previous_satisfies_hard_constraints &&
                                          objective_residual_at_previous
                                                  .template lpNorm<1>() <
                                              solver_config.precision_threshold;
    auto protect_achieved_objective = [&]() {
      detail::ComputeGeneralizedInverse(
          current_jacobian * previous_null_space, solver_config.epsilon,
          &inverse_objective_jacobian_projected);
      null_space_projector.noalias() =
          previous_null_space - inverse_objective_jacobian_projected *
                                    current_jacobian * previous_null_space;
      null_space_projector = (null_space_projector.array().abs() <
                             solver_config.precision_threshold)
                                .select(0.0, null_space_projector);
    };
    if (skip_objective) {
      velocity_solution = previous_velocity;
      constraints_violated = false;
      if (objective_index + 1 < num_objectives) {
        protect_achieved_objective();
      }
    }

    const auto compute_current_objective_inverse =
        [&](const RegularizedInverseConfig &regularization_config,
            const Eigen::MatrixXd &projector, double *condition_number) {
          Eigen::MatrixXd inverse = compute_projected_objective_inverse(
              regularization_config, current_jacobian, projector,
              condition_number);
          if (preserve_legacy_velocity_behavior && !objective_is_min_error &&
              current_target.template lpNorm<1>() <
                  solver_config.precision_threshold &&
              objective_residual_at_previous.template lpNorm<1>() >=
                  solver_config.precision_threshold) {
            // A zero SCALE target has no adjustable ray magnitude. Damping
            // its cancellation of an earlier task's residual can falsely
            // declare it unreachable. Rank-aware equality correction stays
            // in the higher-priority space and still faces all hard bounds.
            detail::ComputeGeneralizedInverse(
                current_jacobian * projector, solver_config.epsilon, &inverse);
            inverse = projector * inverse;
          }
          return inverse;
        };

    // Pre-compute transformation matrices for efficiency
    double task_condition_number = 1.0;
    Eigen::MatrixXd damped_inverse_projected_jacobian =
        compute_current_objective_inverse(
            solver_config.regularization_config, constrained_projector,
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

    auto try_bound_release = [&](bool refine_partial = false) {
      bool &attempted = refine_partial ? attempted_partial_scale_refinement
                                       : attempted_bound_release;
      if (attempted) return false;
      attempted = true;
      Eigen::Index protected_rows = 0;
      for (size_t prior = 0; prior < objective_index; ++prior)
        protected_rows += objective_matrices[prior].rows();
      Eigen::MatrixXd protected_tasks(protected_rows, degrees_of_freedom);
      Eigen::Index offset = 0;
      for (size_t prior = 0; prior < objective_index; ++prior) {
        protected_tasks.middleRows(offset, objective_matrices[prior].rows()) =
            objective_matrices[prior];
        offset += objective_matrices[prior].rows();
      }
      const Eigen::VectorXd bias = preserve_legacy_velocity_behavior
                                      ? Eigen::VectorXd::Zero(target_dimension)
                                      : affine_objective_biases[objective_index];
      const Eigen::VectorXd seed = refine_partial ? velocity_solution : previous_velocity;
      const double seed_scale = refine_partial ? velocity_scale : 0.0;
      bool recovered = false;
      if (preserve_legacy_velocity_behavior || refine_partial) {
        // A certified positive-scale refinement keeps its original search;
        // a new Phase-I seed must not replace that known scale snapshot.
        recovered = detail::RecoverZeroScaleTask(
            current_jacobian, current_target, bias, protected_tasks,
            constraint_coefficients, min_bounds, max_bounds,
            seed, &velocity_solution, &velocity_scale, seed_scale);
      } else {
        if (objective_index > 0) {
          // Only the objective-entry command can define protected images.
          // Failed trial vectors and invalid prefix scales are not witnesses.
          if (!previous_velocity.allFinite() ||
              detail::ComputeMaxLinearConstraintViolation(
                  constraint_coefficients, min_bounds, max_bounds,
                  previous_velocity) > 1e-10) return false;
          for (size_t prior = 0; prior < objective_index; ++prior) {
            if (objective_effective_modes[prior] == TaskSolveMode::kMinError)
              continue;
            const double alpha = objective_scaling_factors(prior);
            const auto &prior_bias = affine_objective_biases[prior];
            const auto &prior_target = scalable_objective_targets[prior];
            if (!std::isfinite(alpha) || alpha < 0.0 || alpha > 1.0 ||
                (objective_matrices[prior] * previous_velocity + prior_bias -
                 alpha * prior_target).norm() >
                    1e-10 * std::max({1.0, prior_bias.norm(),
                                     alpha * prior_target.norm()})) return false;
          }
        }
        const Eigen::VectorXd protected_values =
            protected_tasks * previous_velocity;
        recovered = detail::RecoverScaleTaskWithFeasibleRaySeed(
            current_jacobian, current_target, bias, protected_tasks,
            protected_values, constraint_coefficients, min_bounds, max_bounds,
            seed, solver_config, &velocity_solution, &velocity_scale, seed_scale);
      }
      if (recovered) {
        // Trial pins are not task equalities. Keep the achieved task values
        // protected while allowing later objectives to release bound faces.
        saturated_constraint_selector.setZero();
        saturated_constraint_matrix.setZero();
        saturated_values.setZero();
        Eigen::MatrixXd achieved_tasks(protected_rows + target_dimension,
                                       degrees_of_freedom);
        achieved_tasks.topRows(protected_rows) = protected_tasks;
        achieved_tasks.bottomRows(target_dimension) = current_jacobian;
        detail::ComputeGeneralizedInverse(
            achieved_tasks, solver_config.epsilon,
            &inverse_objective_jacobian_projected);
        null_space_projector =
            Eigen::MatrixXd::Identity(degrees_of_freedom, degrees_of_freedom) -
            inverse_objective_jacobian_projected * achieved_tasks;
        return true;
      }
      return false;
    };

    // Iterative constraint satisfaction with hierarchical optimization
    while (constraints_violated) {
      constraints_violated = false;

      // Compute augmented projection operator
      detail::ComputeGeneralizedInverse(
          saturated_constraints_on_previous_space, active_constraint_rank_tolerance,
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

      // A zero constraint step must not fall through to min-error. That path
      // reports scale -1 for a Jacobian that cannot produce the target.
      bool inert_task_unreachable = false;

      // Evaluate constraint satisfaction
      constraint_evaluation.noalias() =
          constraint_coefficients * velocity_solution;
      // Retained for the rank-termination pass below. The scaled-target
      // evaluation overwrites constraint_evaluation after this point.
      const Eigen::VectorXd full_scale_constraint_evaluation =
          constraint_evaluation;
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
          // A step that does not move the constraints is full scale only when
          // it also meets the task. A zero Jacobian otherwise reports scale 0
          // for a target it cannot produce.
          const double task_residual =
              (current_jacobian * velocity_solution - current_full_target)
                  .norm();
          velocity_scale =
              task_residual <= solver_config.precision_threshold *
                                   std::max(1.0, current_full_target.norm())
                  ? 1.0
                  : 0.0;
          inert_task_unreachable = velocity_scale == 0.0;
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
            rank_analysis.rank() >= task_rank_budget;
      }

      const bool should_use_min_error_fallback =
          preserve_legacy_velocity_behavior || !saturation_preserves_task_rank;
      if (!objective_is_min_error && velocity_scale <= solver_config.epsilon &&
          try_bound_release()) {
        constraints_violated = false;
        ++iteration_counter;
        break;
      }
      if (!objective_is_min_error && velocity_scale <= solver_config.epsilon &&
          allow_min_error_fallback && should_use_min_error_fallback &&
          !inert_task_unreachable) {
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
              saturated_constraints_on_previous_space, active_constraint_rank_tolerance,
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

          // Terminate when active bounds make the requested target unreachable,
          // or when saturation makes no progress. Dependent task rows alone do
          // not establish that the requested target is unreachable.
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
            // Rank still ends the loop, including the legacy ESNS entry point.
            // The budget is the independent task rank, so a consistent
            // duplicate row is not a rank failure.
            auto residual_reachable = [&]() -> bool {
              const Eigen::MatrixXd projected_task =
                  current_jacobian * constrained_projector;
              Eigen::MatrixXd task_inverse;
              detail::ComputeGeneralizedInverse(
                  projected_task, solver_config.epsilon, &task_inverse);
              const Eigen::VectorXd constraint_offset =
                  previous_velocity + inverse_saturated_constraints_projected *
                      (saturated_values -
                       saturated_constraint_matrix * previous_velocity);
              const Eigen::VectorXd task_rhs =
                  current_full_target - current_jacobian * constraint_offset;
              const Eigen::VectorXd residual =
                  task_rhs - projected_task * (task_inverse * task_rhs);
              const Eigen::VectorXd saturation_residual =
                  saturated_constraint_matrix * constraint_offset -
                  saturated_values;
              return residual.allFinite() && saturation_residual.allFinite() &&
                     residual.norm() <= solver_config.precision_threshold *
                                            std::max(1.0, task_rhs.norm()) &&
                     saturation_residual.norm() <=
                         solver_config.precision_threshold *
                             std::max(1.0, saturated_values.norm());
            };
            const bool independent_rank_lost =
                effective_rank < task_rank_budget;
            // A consistent full-rank row hits the same iteration cap as a
            // repeated row: one blocked joint per pass cannot free the last
            // joint of a 21-DoF task. Saturate every full-scale violation
            // together when that set still reaches the target.
            const bool consistent_reachable_completion =
                redundant_rows_consistent && !independent_rank_lost;
            if (consistent_reachable_completion && residual_reachable()) {
              const Eigen::MatrixXd saved_selector =
                  saturated_constraint_selector;
              const Eigen::VectorXd saved_values = saturated_values;
              bool pinned_redundant_bounds = false;
              for (Eigen::Index constraint_idx = 0;
                   constraint_idx < full_scale_constraint_evaluation.size();
                   ++constraint_idx) {
                if (saturated_constraint_selector(constraint_idx,
                                                  constraint_idx) != 0) {
                  continue;
                }
                const double value =
                    full_scale_constraint_evaluation(constraint_idx);
                if (value >= (*active_min_bounds)(constraint_idx) -
                                 solver_config.epsilon &&
                    value <= (*active_max_bounds)(constraint_idx) +
                                 solver_config.epsilon) {
                  continue;
                }
                saturated_constraint_selector(constraint_idx,
                                              constraint_idx) = 1;
                saturated_values(constraint_idx, 0) = std::min(
                    std::max((*active_min_bounds)(constraint_idx), value),
                    (*active_max_bounds)(constraint_idx));
                pinned_redundant_bounds = true;
              }
              auto rebuild_saturated_projector = [&]() {
                saturated_constraint_matrix.noalias() =
                    saturated_constraint_selector * constraint_coefficients;
                saturated_constraints_on_previous_space.noalias() =
                    saturated_constraint_matrix * previous_null_space;
                detail::ComputeGeneralizedInverse(
                    saturated_constraints_on_previous_space,
                    active_constraint_rank_tolerance,
                    &inverse_saturated_constraints_projected);
                constrained_projector.noalias() =
                    previous_null_space -
                    inverse_saturated_constraints_projected *
                        saturated_constraints_on_previous_space;
              };
              auto restore_saved_saturation = [&]() {
                saturated_constraint_selector = saved_selector;
                saturated_values = saved_values;
                rebuild_saturated_projector();
              };
              // A joint forces scale 0 when the full step has no room in the
              // direction it moves. Joints that still allow a positive scale
              // stay free: pinning them too makes a reachable target look
              // unreachable and the one-joint loop then stops early.
              auto pin_scale_zero_blockers = [&]() -> bool {
                bool pinned_blockers = false;
                for (Eigen::Index constraint_idx = 0;
                     constraint_idx < full_scale_constraint_evaluation.size();
                     ++constraint_idx) {
                  if (saturated_constraint_selector(constraint_idx,
                                                    constraint_idx) != 0) {
                    continue;
                  }
                  const double origin = unscaled_contribution(constraint_idx);
                  const double step =
                      scaled_velocity_contribution(constraint_idx);
                  double scale_limit = 1.0;
                  if (step > solver_config.epsilon) {
                    scale_limit =
                        ((*active_max_bounds)(constraint_idx) - origin) / step;
                  } else if (step < -solver_config.epsilon) {
                    scale_limit =
                        ((*active_min_bounds)(constraint_idx) - origin) / step;
                  }
                  if (scale_limit > solver_config.epsilon) {
                    continue;
                  }
                  const double value =
                      full_scale_constraint_evaluation(constraint_idx);
                  saturated_constraint_selector(constraint_idx,
                                                constraint_idx) = 1;
                  saturated_values(constraint_idx, 0) = std::min(
                      std::max((*active_min_bounds)(constraint_idx), value),
                      (*active_max_bounds)(constraint_idx));
                  pinned_blockers = true;
                }
                return pinned_blockers;
              };
              if (pinned_redundant_bounds) {
                rebuild_saturated_projector();
                if (!residual_reachable()) {
                  restore_saved_saturation();
                  if (pin_scale_zero_blockers()) {
                    rebuild_saturated_projector();
                    if (!residual_reachable()) {
                      restore_saved_saturation();
                      // Full scale does not fit these bounds. Stop with the
                      // best feasible scale instead of walking every joint
                      // into the iteration-limit failure.
                      if (has_feasible_scale_snapshot) {
                        should_terminate = true;
                      }
                    }
                  } else if (has_feasible_scale_snapshot) {
                    should_terminate = true;
                  }
                }
              } else if (!made_saturation_progress) {
                should_terminate = true;
              }
            } else {
              should_terminate = independent_rank_lost;
            }
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
                  active_constraint_rank_tolerance,
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
              if (!objective_is_min_error && task_rank_budget > 0 &&
                  effective_rank > 0 && effective_rank < task_rank_budget) {
                const double rank_ratio =
                    static_cast<double>(task_rank_budget) /
                    static_cast<double>(effective_rank);
                adaptive_regularization.regularization_factor *= rank_ratio;
              }
              velocity_solution.noalias() =
                  previous_velocity +
                  compute_current_objective_inverse(
                      adaptive_regularization, constrained_projector,
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
          if (auto recovered = retry_invalid_min_error_suffix()) return *recovered;
          if (velocity_solution.allFinite()) {
            if (auto recovered = retry_invalid_min_error(
                    Eigen::VectorXd::Zero(degrees_of_freedom))) {
              return *recovered;
            }
          }
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
          if (auto recovered = retry_invalid_min_error_suffix()) return *recovered;
          if (velocity_solution.allFinite()) {
            if (auto recovered = retry_invalid_min_error(
                    Eigen::VectorXd::Zero(degrees_of_freedom))) {
              return *recovered;
            }
          }
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
        protect_achieved_objective();
      }

      // Refresh cached computations for next iteration
      {
        double cond_tmp = 1.0;
        damped_inverse_projected_jacobian =
            compute_current_objective_inverse(
                solver_config.regularization_config, constrained_projector,
                &cond_tmp);
        worst_condition_number =
            std::max(worst_condition_number, cond_tmp);
      }
      saturated_constraints_on_previous_space.noalias() =
          saturated_constraint_matrix * previous_null_space;
      saturated_constraints_velocity.noalias() =
          saturated_constraint_matrix * previous_velocity;
    }

    // Rank termination can restore a zero snapshot without visiting the
    // in-loop fallback branch. Give that terminal stall the same recovery.
    if (!objective_is_min_error && velocity_scale <= solver_config.epsilon)
      try_bound_release();
    else if (!objective_is_min_error && solver_config.refine_partial_scale &&
             velocity_scale < 1.0 - solver_config.epsilon)
      try_bound_release(true);

    if (!preserve_legacy_velocity_behavior && objective_is_min_error) {
      completed_objective_images[objective_index] =
          objective_matrices[objective_index] * velocity_solution;
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
  bool final_bound_failure =
      (((constraint_coefficients * velocity_solution).array() <
        (final_check_min_bounds.array() - solver_config.epsilon))
           .any() ||
       ((constraint_coefficients * velocity_solution).array() >
        (final_check_max_bounds.array() + solver_config.epsilon))
           .any());
  if (final_bound_failure && preserve_legacy_velocity_behavior &&
      !enhanced_mode && velocity_solution.allFinite()) {
    Eigen::VectorXd feasible_witness;
    const auto feasibility = detail::FindHardConstraintFeasibleWitness(
        constraint_coefficients, final_check_min_bounds, final_check_max_bounds,
        solver_config, &feasible_witness);
    if (feasibility == detail::HardConstraintFeasibilityStatus::kProvenInfeasible) {
      return phase_one_failure(
          SolverStatus::kInfeasible,
          "final hard constraint check certified the system infeasible");
    }
    // A collapsed stationary command can require compensating motion. Preserve
    // every achieved task value while realizing the same command within bounds.
    if (feasibility == detail::HardConstraintFeasibilityStatus::kFeasible &&
        velocity_solution.norm() <= solver_config.epsilon) {
      Eigen::Index rows = 0;
      for (const auto &task : objective_matrices) rows += task.rows();
      const auto *physical = execution_options.stationary_recovery_protected_tasks;
      if (physical) rows += physical->rows();
      Eigen::MatrixXd protected_tasks(rows, degrees_of_freedom);
      Eigen::Index offset = 0;
      for (const auto &task : objective_matrices) {
        protected_tasks.middleRows(offset, task.rows()) = task;
        offset += task.rows();
      }
      if (physical) protected_tasks.bottomRows(physical->rows()) = *physical;
      Eigen::JacobiSVD<Eigen::MatrixXd> decomposition(protected_tasks,
                                                     Eigen::ComputeFullV);
      decomposition.setThreshold(1e-10);
      const Eigen::MatrixXd free = decomposition.matrixV().rightCols(
          degrees_of_freedom - decomposition.rank());
      if (free.cols() > 0) {
        const Eigen::VectorXd values = constraint_coefficients * velocity_solution;
        Eigen::VectorXd correction;
        const auto null_feasibility = detail::FindHardConstraintFeasibleWitness(
            constraint_coefficients * free, final_check_min_bounds - values,
            final_check_max_bounds - values, solver_config, &correction);
        if (null_feasibility == detail::HardConstraintFeasibilityStatus::kFeasible) {
          const Eigen::VectorXd delta = free * correction;
          const Eigen::VectorXd candidate = velocity_solution + delta;
          if ((protected_tasks * delta).norm() <= 1e-8 &&
              detail::ComputeMaxLinearConstraintViolation(
                  constraint_coefficients, final_check_min_bounds,
                  final_check_max_bounds, candidate) <= solver_config.epsilon) {
            velocity_solution = candidate;
            final_bound_failure = false;
          }
        }
      }
    }
  }
  if (final_bound_failure) {
    solver_status = SolverStatus::kNumericalError;
  }

  if (final_bound_failure) {
    if (auto recovered = retry_invalid_min_error(velocity_solution)) {
      return *recovered;
    }
  }
  if (final_bound_failure || !velocity_solution.allFinite()) {
    if (auto recovered = retry_invalid_min_error_suffix()) return *recovered;
  }

  bool unsupported_legacy_primary = false;
  if (preserve_legacy_velocity_behavior &&
      solver_status == SolverStatus::kSuccess &&
      objective_effective_modes.front() != TaskSolveMode::kMinError &&
      objective_scaling_factors(0) > solver_config.epsilon &&
      (affine_objective_biases.empty() || affine_objective_biases.front().isZero(0.0))) {
    const auto &matrix = objective_matrices.front();
    const Eigen::VectorXd rhs = objective_scaling_factors(0) *
        scalable_objective_targets.front();
    const double tolerance = objective_equality_tolerance *
        std::max(1.0, rhs.norm());
    // Preserve the legacy absolute numerical-zero row policy independently
    // of the relative structural rank test below.
    for (Eigen::Index row = 0; row < matrix.rows(); ++row) {
      if (matrix.row(row).squaredNorm() <= 1e-20 &&
          std::abs(rhs(row)) > tolerance) {
        unsupported_legacy_primary = true;
        break;
      }
    }
    // Test the original primary range independently of its regularized inverse.
    // Damping a supported direction remains the legacy policy; an inconsistent
    // zero, repeated or dependent row cannot represent a positive primary ray.
    // Lower legacy objectives retain their projected preference semantics.
    // A structural threshold retains supported near-singular directions that
    // the regularizer intentionally attenuates.
    if (execution_options.validate_legacy_primary_range) {
      Eigen::ColPivHouseholderQR<Eigen::MatrixXd> range_qr(matrix);
      range_qr.setThreshold(std::numeric_limits<double>::epsilon() *
          std::max<Eigen::Index>(1, std::max(matrix.rows(), matrix.cols())));
      unsupported_legacy_primary = unsupported_legacy_primary ||
          (range_qr.rank() < matrix.rows() &&
           (matrix * range_qr.solve(rhs) - rhs).norm() > tolerance);
    }
    if (unsupported_legacy_primary) {
      solver_status = SolverStatus::kNoProgress;
    }
  }

  if (!preserve_legacy_velocity_behavior &&
      solver_status == SolverStatus::kSuccess) {
    for (size_t objective_index = 0; objective_index < num_objectives;
         ++objective_index) {
      if (objective_effective_modes[objective_index] ==
          TaskSolveMode::kMinError) {
        continue;
      }
      Eigen::VectorXd scaled_rhs =
          objective_scaling_factors(static_cast<Eigen::Index>(objective_index)) *
              scalable_objective_targets[objective_index] -
          affine_objective_biases[objective_index];
      double equality_tolerance =
          objective_equality_tolerance *
          std::max(1.0, scaled_rhs.norm());
      const Eigen::VectorXd achieved =
          objective_matrices[objective_index] * velocity_solution;
      if ((achieved - scaled_rhs).norm() > equality_tolerance &&
          !affine_objective_biases[objective_index].isZero(0.0)) {
        // Active hard rows can fix an affine task at an intermediate scale
        // after its projected Jacobian loses rank. Recover the scale only
        // when the final realization lies on the original affine task ray.
        // The velocity and every prior achieved task value remain unchanged.
        const auto &target = scalable_objective_targets[objective_index];
        const double target_squared_norm = target.squaredNorm();
        if (target_squared_norm > solver_config.precision_threshold *
                                      solver_config.precision_threshold) {
          const Eigen::VectorXd realized_target =
              achieved + affine_objective_biases[objective_index];
          const double realized_scale =
              target.dot(realized_target) / target_squared_norm;
          if (std::isfinite(realized_scale) &&
              realized_scale >= -objective_equality_tolerance &&
              realized_scale <= 1.0 + objective_equality_tolerance) {
            const double bounded_scale = std::clamp(realized_scale, 0.0, 1.0);
            const Eigen::VectorXd corrected_rhs =
                bounded_scale * target - affine_objective_biases[objective_index];
            const double corrected_tolerance = objective_equality_tolerance *
                std::max(1.0, corrected_rhs.norm());
            if ((achieved - corrected_rhs).norm() <= corrected_tolerance) {
              objective_scaling_factors(
                  static_cast<Eigen::Index>(objective_index)) = bounded_scale;
              scaled_rhs = corrected_rhs;
              equality_tolerance = corrected_tolerance;
            }
          }
        }
      }
      if ((objective_matrices[objective_index] * velocity_solution - scaled_rhs)
              .norm() > equality_tolerance) {
        solver_status = SolverStatus::kNoProgress;
        break;
      }
    }
  }
  if (solver_status == SolverStatus::kNoProgress) {
    if (auto recovered = retry_invalid_min_error_suffix()) return *recovered;
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

  // Saturation can terminate with a zero best scale after the in-loop fallback
  // check (notably at exact zero headroom). Retry explicitly configured tasks
  // from the original problem so failed trial active rows do not carry over.
  // Each retry converts one SCALE objective to MIN_ERROR, bounding recursion by
  // the number of objectives. The implicit legacy low-level policy is unchanged.
  if (solver_status == SolverStatus::kSuccess ||
      solver_status == SolverStatus::kNoProgress ||
      (final_bound_failure && !generated_nonfinite_solver_state &&
       solver_status == SolverStatus::kNumericalError)) {
    for (size_t index = 0; index < objective_configs.size() &&
                           index < applied_scales.size(); ++index) {
      const auto &objective_config = objective_configs[index];
      // A rank-deficient objective can retain a positive trial scale while its
      // exact task ray is unrealizable. Retry only that failed equality when
      // fallback is explicitly authorized; a valid higher objective must not
      // fall back merely because a lower equality failed.
      bool failed_requested_equality = false;
      if ((!preserve_legacy_velocity_behavior || (unsupported_legacy_primary && index == 0)) &&
          solver_status == SolverStatus::kNoProgress &&
          objective_effective_modes[index] != TaskSolveMode::kMinError) {
        const Eigen::VectorXd bias = affine_objective_biases.empty()
            ? Eigen::VectorXd::Zero(scalable_objective_targets[index].size())
            : affine_objective_biases[index];
        const Eigen::VectorXd scaled_rhs = applied_scales[index] *
            scalable_objective_targets[index] - bias;
        failed_requested_equality =
            (objective_matrices[index] * velocity_solution - scaled_rhs).norm() >
            objective_equality_tolerance * std::max(1.0, scaled_rhs.norm());
      }
      if (detail::ShouldRetryObjectiveAsMinError(
              objective_config, solver_config,
              detail::ObjectiveMinErrorRetryState{
                  objective_effective_modes[index], objective_used_fallback[index],
                  final_bound_failure, applied_scales[index],
                  objective_errors[index], failed_requested_equality})) {
        auto retry_configs = objective_configs;
        retry_configs[index].solve_mode = TaskSolveMode::kMinError;
        retry_configs[index].allow_min_error_fallback = false;
        auto retry_execution_options = execution_options;
        if (full_hard_constraint_witness_available) {
          retry_execution_options.initial_hard_constraint_witness =
              &hard_constraint_witness;
        }
        SolverResult retry = solveHierarchicalLinearSystemEigen(
            scalable_objective_targets, affine_objective_biases,
            objective_matrices, constraint_coefficients, min_bounds, max_bounds,
            solver_config, retry_configs, max_constraint_softening_factors,
            retry_execution_options);
        if (retry.task_used_fallback.size() > index) {
          retry.task_used_fallback[index] = true;
          for (size_t prior = 0; prior < objective_used_fallback.size() &&
                                 prior < retry.task_used_fallback.size(); ++prior) {
            retry.task_used_fallback[prior] =
                retry.task_used_fallback[prior] || objective_used_fallback[prior];
          }
        }
        retry.computation_time_ms += elapsed_milliseconds;
        return retry;
      }
    }
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
    double objective_equality_tolerance,
    bool constraint_rows_pre_normalized = false,
    bool hard_constraints_prevalidated_feasible = false,
    bool allow_best_effort_lower_tasks = false,
    bool prefer_direct_min_error_qp = false) {
  Eigen::MatrixXd normalized_coefficients_storage;
  Eigen::VectorXd normalized_min_bounds_storage;
  Eigen::VectorXd normalized_max_bounds_storage;
  const Eigen::MatrixXd *normalized_coefficients = &constraint_coefficients;
  const Eigen::VectorXd *normalized_min_bounds = &min_bounds;
  const Eigen::VectorXd *normalized_max_bounds = &max_bounds;
  if (!constraint_rows_pre_normalized) {
    NormalizeLinearConstraintRows(
        constraint_coefficients, min_bounds, max_bounds,
        &normalized_coefficients_storage, &normalized_min_bounds_storage,
        &normalized_max_bounds_storage);
    normalized_coefficients = &normalized_coefficients_storage;
    normalized_min_bounds = &normalized_min_bounds_storage;
    normalized_max_bounds = &normalized_max_bounds_storage;
  }
  return solveHierarchicalLinearSystemEigen(
      scalable_objective_targets, affine_objective_biases, objective_matrices,
      *normalized_coefficients, *normalized_min_bounds, *normalized_max_bounds,
      solver_config, objective_configs, max_constraint_softening_factors,
      HierarchicalLinearSolverExecutionOptions{
          HierarchicalLinearSolverPolicy::kGeneralizedEsns,
          objective_equality_tolerance,
          hard_constraints_prevalidated_feasible,
          nullptr,
          true,
          allow_best_effort_lower_tasks,
          prefer_direct_min_error_qp});
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
