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

#include <embodik/ik_baseline.hpp>
#include <embodik/types.hpp>

#include <Eigen/Core>
#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

namespace embodik {

/// Result of one constrained weighted-advisor solve.
///
/// The advisor solves a weighted stacked problem on the same objective vectors
/// that the prioritized solver receives:
///
///   v* = argmin sum_k w_k ||J_k v - b_k||^2
///        subject to l <= C v <= u
///
/// Runtime diagnostics compute this candidate without changing the prioritized
/// output. The optional weighted fallback can accept it only when the candidate
/// satisfies the hard constraints.
struct WeightedAdvisoryResult {
  bool available = false;
  Eigen::VectorXd v;
  double v_norm = std::numeric_limits<double>::quiet_NaN();
  std::vector<double> per_objective_error;
  double condition_number = std::numeric_limits<double>::quiet_NaN();
  SolverStatus status = SolverStatus::kInvalidInput;
  std::string status_message;
};

inline bool build_weighted_stacked_objective(
    const std::vector<Eigen::VectorXd> &objective_targets,
    const std::vector<Eigen::MatrixXd> &objective_jacobians,
    const std::vector<double> &objective_weights, Eigen::MatrixXd *J_stack,
    Eigen::VectorXd *b_stack,
    const std::vector<Eigen::VectorXd> &objective_row_weights =
        std::vector<Eigen::VectorXd>{}) {
  const size_t n_obj = objective_targets.size();
  if (n_obj == 0 || objective_jacobians.size() != n_obj || J_stack == nullptr ||
      b_stack == nullptr) {
    return false;
  }

  Eigen::Index nv = -1;
  Eigen::Index total_rows = 0;
  for (size_t k = 0; k < n_obj; ++k) {
    const auto &J = objective_jacobians[k];
    const auto &b = objective_targets[k];
    if (J.rows() != b.rows()) {
      return false;
    }
    if (J.rows() == 0) {
      continue;
    }
    if (nv < 0) {
      nv = J.cols();
    } else if (J.cols() != nv) {
      return false;
    }
    total_rows += J.rows();
  }
  if (nv <= 0 || total_rows == 0) {
    return false;
  }

  std::vector<double> weights(n_obj, 1.0);
  for (size_t k = 0; k < n_obj && k < objective_weights.size(); ++k) {
    const double wk = objective_weights[k];
    weights[k] = (std::isfinite(wk) && wk >= 0.0) ? wk : 0.0;
  }

  J_stack->resize(total_rows, nv);
  b_stack->resize(total_rows);
  J_stack->setZero();
  b_stack->setZero();

  Eigen::Index offset = 0;
  for (size_t k = 0; k < n_obj; ++k) {
    const auto &J = objective_jacobians[k];
    const auto &b = objective_targets[k];
    if (J.rows() == 0) {
      continue;
    }
    const bool use_row_weights = k < objective_row_weights.size() &&
                                 objective_row_weights[k].size() == J.rows();
    if (use_row_weights) {
      for (Eigen::Index r = 0; r < J.rows(); ++r) {
        const double row_weight = objective_row_weights[k](r);
        const double sqrt_w =
            (std::isfinite(row_weight) && row_weight >= 0.0)
                ? std::sqrt(row_weight)
                : 0.0;
        J_stack->row(offset + r) = sqrt_w * J.row(r);
        (*b_stack)(offset + r) = sqrt_w * b(r);
      }
    } else {
      const double sqrt_w = std::sqrt(weights[k]);
      J_stack->block(offset, 0, J.rows(), nv) = sqrt_w * J;
      b_stack->segment(offset, b.rows()) = sqrt_w * b;
    }
    offset += J.rows();
  }
  return true;
}

inline void populate_weighted_advisory_residuals(
    WeightedAdvisoryResult *out,
    const std::vector<Eigen::VectorXd> &objective_targets,
    const std::vector<Eigen::MatrixXd> &objective_jacobians) {
  if (out == nullptr) {
    return;
  }
  const size_t n_obj = objective_targets.size();
  out->per_objective_error.assign(n_obj,
                                  std::numeric_limits<double>::quiet_NaN());
  for (size_t k = 0; k < n_obj; ++k) {
    const auto &J = objective_jacobians[k];
    const auto &b = objective_targets[k];
    if (J.rows() == 0) {
      out->per_objective_error[k] = 0.0;
      continue;
    }
    out->per_objective_error[k] = (J * out->v - b).norm();
  }
}

inline SolverResult computeConstrainedWeightedVelocitySolutionEigen(
    const std::vector<Eigen::VectorXd> &objective_targets,
    const std::vector<Eigen::MatrixXd> &objective_jacobians,
    const Eigen::MatrixXd &constraint_coefficients,
    const Eigen::VectorXd &min_bounds, const Eigen::VectorXd &max_bounds,
    const std::vector<double> &objective_weights,
    const VelocitySolverConfig &solver_config,
    const std::vector<Eigen::VectorXd> &objective_row_weights =
        std::vector<Eigen::VectorXd>{}) {
  auto fail = [](SolverStatus status, const char *msg) {
    SolverResult out{};
    out.status = status;
    out.status_message = msg;
    return out;
  };

  Eigen::MatrixXd J_stack;
  Eigen::VectorXd b_stack;
  if (!build_weighted_stacked_objective(objective_targets, objective_jacobians,
                                        objective_weights, &J_stack,
                                        &b_stack, objective_row_weights)) {
    return fail(SolverStatus::kInvalidInput,
                "could not build weighted stacked objective");
  }

  if (constraint_coefficients.cols() != J_stack.cols() ||
      min_bounds.size() != constraint_coefficients.rows() ||
      max_bounds.size() != constraint_coefficients.rows()) {
    return fail(SolverStatus::kConstraintBoundsMismatch,
                "constraint dimensions do not match weighted objective");
  }

  struct InequalityRow {
    Eigen::RowVectorXd coeff;
    double bound = 0.0;
  };
  std::vector<InequalityRow> inequalities;
  inequalities.reserve(static_cast<size_t>(2 * constraint_coefficients.rows()));
  for (Eigen::Index r = 0; r < constraint_coefficients.rows(); ++r) {
    if (std::isfinite(max_bounds(r))) {
      inequalities.push_back(
          InequalityRow{constraint_coefficients.row(r), max_bounds(r)});
    }
    if (std::isfinite(min_bounds(r))) {
      inequalities.push_back(
          InequalityRow{-constraint_coefficients.row(r), -min_bounds(r)});
    }
  }

  const Eigen::Index nv = J_stack.cols();
  const double ridge = std::max(
      solver_config.regularization_config.epsilon *
          solver_config.regularization_config.epsilon,
      1e-12);
  Eigen::MatrixXd H = J_stack.transpose() * J_stack;
  H.diagonal().array() += ridge;
  const Eigen::VectorXd g = J_stack.transpose() * b_stack;

  auto solve_with_active_set = [&](const std::vector<size_t> &active,
                                   Eigen::VectorXd *lambda_out) -> Eigen::VectorXd {
    if (active.empty()) {
      if (lambda_out != nullptr) {
        lambda_out->resize(0);
      }
      return H.ldlt().solve(g).eval();
    }

    Eigen::MatrixXd A(active.size(), static_cast<size_t>(nv));
    Eigen::VectorXd d(active.size());
    for (size_t i = 0; i < active.size(); ++i) {
      A.row(static_cast<Eigen::Index>(i)) = inequalities[active[i]].coeff;
      d(static_cast<Eigen::Index>(i)) = inequalities[active[i]].bound;
    }

    Eigen::MatrixXd kkt(nv + static_cast<Eigen::Index>(active.size()),
                        nv + static_cast<Eigen::Index>(active.size()));
    kkt.setZero();
    kkt.topLeftCorner(nv, nv) = H;
    kkt.topRightCorner(nv, static_cast<Eigen::Index>(active.size())) =
        A.transpose();
    kkt.bottomLeftCorner(static_cast<Eigen::Index>(active.size()), nv) = A;

    Eigen::VectorXd rhs(nv + static_cast<Eigen::Index>(active.size()));
    rhs.head(nv) = g;
    rhs.tail(static_cast<Eigen::Index>(active.size())) = d;

    Eigen::VectorXd z = kkt.completeOrthogonalDecomposition().solve(rhs);
    if (lambda_out != nullptr) {
      *lambda_out = z.tail(static_cast<Eigen::Index>(active.size()));
    }
    return z.head(nv).eval();
  };

  std::vector<size_t> active;
  Eigen::VectorXd v = Eigen::VectorXd::Zero(nv);
  constexpr size_t kExtraIterations = 20;
  const size_t iteration_limit =
      std::max<size_t>(solver_config.iteration_limit,
                       inequalities.size() + kExtraIterations);
  unsigned int iterations = 0;
  for (; iterations < iteration_limit; ++iterations) {
    Eigen::VectorXd lambda;
    v = solve_with_active_set(active, &lambda);
    if (!v.allFinite()) {
      return fail(SolverStatus::kNumericalError,
                  "constrained weighted solve produced non-finite entries");
    }

    double max_violation = solver_config.epsilon;
    size_t most_violated = inequalities.size();
    for (size_t i = 0; i < inequalities.size(); ++i) {
      const double violation = inequalities[i].coeff.dot(v) - inequalities[i].bound;
      if (violation > max_violation) {
        max_violation = violation;
        most_violated = i;
      }
    }
    if (most_violated != inequalities.size()) {
      bool already_active = false;
      for (size_t idx : active) {
        already_active = already_active || idx == most_violated;
      }
      if (already_active) {
        return fail(SolverStatus::kNumericalError,
                    "active constrained weighted row remained violated");
      }
      active.push_back(most_violated);
      continue;
    }

    if (!active.empty() && lambda.size() == static_cast<Eigen::Index>(active.size())) {
      Eigen::Index remove_index = -1;
      const double min_lambda = lambda.minCoeff(&remove_index);
      if (min_lambda < -solver_config.epsilon && remove_index >= 0) {
        active.erase(active.begin() + remove_index);
        continue;
      }
    }
    break;
  }

  if (iterations >= iteration_limit) {
    return fail(SolverStatus::kNoProgress,
                "constrained weighted active set did not converge");
  }

  for (const auto &ineq : inequalities) {
    if (ineq.coeff.dot(v) > ineq.bound + solver_config.epsilon) {
      return fail(SolverStatus::kNumericalError,
                  "constrained weighted result violates hard constraints");
    }
  }

  std::vector<double> final_velocities(static_cast<size_t>(nv));
  for (Eigen::Index i = 0; i < nv; ++i) {
    final_velocities[static_cast<size_t>(i)] = v(i);
  }
  const double final_error = (J_stack * v - b_stack).norm();
  double condition = 1.0;
  (void)linalg::ComputeRegularizedInverse(solver_config.regularization_config,
                                          J_stack, &condition);

  SolverResult out{};
  out.solution = std::move(final_velocities);
  out.status = SolverStatus::kSuccess;
  out.iterations = iterations + 1U;
  out.final_error = final_error;
  out.task_scales = {-1.0};
  out.task_errors = {final_error};
  out.task_modes_effective = {TaskSolveMode::kMinError};
  out.task_used_fallback = {false};
  out.condition_number = condition;
  return out;
}

inline WeightedAdvisoryResult compute_constrained_weighted_advisory(
    const std::vector<Eigen::VectorXd> &objective_targets,
    const std::vector<Eigen::MatrixXd> &objective_jacobians,
    const Eigen::MatrixXd &constraint_coefficients,
    const Eigen::VectorXd &min_bounds, const Eigen::VectorXd &max_bounds,
    const std::vector<double> &objective_weights,
    const VelocitySolverConfig &solver_config,
    const std::vector<Eigen::VectorXd> &objective_row_weights =
        std::vector<Eigen::VectorXd>{}) {
  WeightedAdvisoryResult out;
  SolverResult constrained_result = computeConstrainedWeightedVelocitySolutionEigen(
      objective_targets, objective_jacobians, constraint_coefficients, min_bounds,
      max_bounds, objective_weights, solver_config, objective_row_weights);
  out.status = constrained_result.status;
  out.status_message = constrained_result.status_message;
  out.condition_number = constrained_result.condition_number;
  if (constrained_result.status != SolverStatus::kSuccess ||
      constrained_result.solution.empty()) {
    return out;
  }
  out.v = Eigen::Map<const Eigen::VectorXd>(
      constrained_result.solution.data(),
      static_cast<Eigen::Index>(constrained_result.solution.size()));
  if (!out.v.allFinite()) {
    out.status = SolverStatus::kNonFiniteInput;
    out.status_message = "constrained weighted solution contains non-finite entries";
    return out;
  }
  out.available = true;
  out.v_norm = out.v.norm();
  populate_weighted_advisory_residuals(&out, objective_targets,
                                       objective_jacobians);
  return out;
}

} // namespace embodik
