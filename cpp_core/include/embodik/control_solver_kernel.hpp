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

#include <Eigen/Cholesky>
#include <Eigen/Core>
#include <Eigen/LU>
#include <Eigen/QR>
#include <Eigen/SVD>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <optional>
#include <string>
#include <utility>
#include <vector>

// Shared numerical implementation for standalone optimization and hierarchical
// control solvers. It operates on matrices and vectors only; the existing
// VelocitySolverConfig remains the numerical configuration for compatibility.
// Caller validation, objective policy and physical admission stay outside it.
namespace embodik {
namespace detail {

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

// Borrowed for one synchronous numerical solve. Preserve residual factors when
// available: forming normal equations squares their condition number and loses
// accuracy precisely in redundant task directions. No factors escape the call.
struct LeastSquaresQuadratic {
  const Eigen::MatrixXd &matrix;
  const Eigen::VectorXd &target;
  double ridge;
};

// Recover a failed equality-set search from a feasible primal point.
// Successful legacy results do not enter this path.
inline bool RecoverFeasibleWeightedQuadratic(
    const Eigen::MatrixXd &H, const Eigen::VectorXd &g,
    const Eigen::MatrixXd &C, const Eigen::VectorXd &lower,
    const Eigen::VectorXd &upper, const VelocitySolverConfig &config,
    Eigen::VectorXd *solution, unsigned int *iterations_out,
    const Eigen::VectorXd *initial_solution = nullptr,
    bool allow_certified_checkpoint = false,
    std::string *failure_reason = nullptr,
    const LeastSquaresQuadratic *least_squares = nullptr,
    unsigned int iteration_budget = 0) {
  if (failure_reason) failure_reason->clear();
  const auto note_failure = [&](const char *reason) {
    if (failure_reason) *failure_reason = reason;
  };
  *iterations_out = 0;
  if (!H.allFinite() || !g.allFinite() || !C.allFinite()) return false;
  if (least_squares &&
      (least_squares->matrix.cols() != H.rows() ||
       least_squares->matrix.rows() != least_squares->target.size() ||
       !least_squares->matrix.allFinite() || !least_squares->target.allFinite() ||
       !std::isfinite(least_squares->ridge) || least_squares->ridge < 0.0)) {
    note_failure("invalid least-squares objective factors");
    return false;
  }
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
  const size_t limit = iteration_budget > 0 ? iteration_budget :
      std::max<size_t>(config.iteration_limit, 4 * inequality_rows.size() + 40);
  for (size_t iteration = 0; iteration < limit; ++iteration) {
    attempted_iterations = static_cast<unsigned int>(iteration + 1);
    *iterations_out = attempted_iterations;
    if (ComputeMaxLinearConstraintViolation(C, lower, upper, v) > config.epsilon) {
      note_failure("working candidate left original hard bounds");
      return return_validated_checkpoint();
    }
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
    Eigen::VectorXd gradient;
    if (least_squares) {
      gradient = least_squares->matrix.transpose() *
                     (least_squares->matrix * v - least_squares->target) +
                 least_squares->ridge * v;
    } else {
      gradient = H * v - g;
    }
    Eigen::VectorXd direction = Eigen::VectorXd::Zero(nv);
    if (free.cols() > 0) {
      if (least_squares) {
        const auto &matrix = least_squares->matrix;
        const double root_ridge = std::sqrt(least_squares->ridge);
        Eigen::MatrixXd residual(matrix.rows() + nv, free.cols());
        residual.topRows(matrix.rows()).noalias() = matrix * free;
        residual.bottomRows(nv) = root_ridge * free;
        Eigen::VectorXd target(matrix.rows() + nv);
        target.head(matrix.rows()) = least_squares->target - matrix * v;
        target.tail(nv) = -root_ridge * v;
        if (least_squares->ridge == 0.0)
          direction = free * residual.completeOrthogonalDecomposition().solve(target);
        else
          direction = free * residual.colPivHouseholderQr().solve(target);
      } else {
        const Eigen::MatrixXd reduced_H = free.transpose() * H * free;
        direction = -free * reduced_H.ldlt().solve(free.transpose() * gradient);
      }
    }
    if (!direction.allFinite()) {
      note_failure("nonfinite reduced quadratic direction");
      return return_validated_checkpoint();
    }
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
      if (!lambda.allFinite()) {
        note_failure("nonfinite active-set multipliers");
        return return_validated_checkpoint();
      }
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
          std::max(1e-8, 10.0 * config.epsilon) * std::max(1.0, gradient.norm())) {
        note_failure("stationarity certificate not satisfied");
        return return_validated_checkpoint();
      }
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
  note_failure("quadratic active-set iteration budget exhausted");
  return return_validated_checkpoint();
}

} // namespace detail
} // namespace embodik
