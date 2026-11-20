#include <embodik/optimization.hpp>
#include <embodik/control_solver_kernel.hpp>

#include <Eigen/Eigenvalues>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <vector>

namespace embodik {
namespace {
using Clock = std::chrono::steady_clock;

SolveResult failure(SolveStatus status, std::string message) {
  SolveResult result;
  result.status = status;
  result.message = std::move(message);
  return result;
}

template <typename Before, typename After>
bool rescaling_lost_coefficients(const Eigen::MatrixBase<Before> &before,
                                const Eigen::MatrixBase<After> &after) {
  return !after.allFinite() ||
      ((before.array() != 0.0) && (after.array() == 0.0)).any();
}

bool validate(Eigen::Index n, const Eigen::MatrixXd &A,
              const Eigen::VectorXd &l, const Eigen::VectorXd &u,
              const SolveOptions &options, SolveResult *result) {
  if (n <= 0 || A.cols() != n || A.rows() != l.size() || l.size() != u.size()) {
    *result = failure(SolveStatus::kInvalidInput, "inconsistent variable/constraint dimensions");
    return false;
  }
  if (!A.allFinite() || l.array().isNaN().any() || u.array().isNaN().any() ||
      !std::isfinite(options.tolerance) || options.tolerance <= 0.0 ||
      options.max_iterations == 0) {
    *result = failure(SolveStatus::kInvalidInput, "finite coefficients, non-NaN bounds and positive solver options required");
    return false;
  }
  if ((l.array() > u.array()).any() ||
      (l.array() == std::numeric_limits<double>::infinity()).any() ||
      (u.array() == -std::numeric_limits<double>::infinity()).any()) {
    *result = failure(SolveStatus::kInfeasible, "constraint interval contains no finite value");
    return false;
  }
  if (options.x0 && (options.x0->size() != n || !options.x0->allFinite())) {
    *result = failure(SolveStatus::kInvalidInput, "x0 must be finite with one value per variable");
    return false;
  }
  return true;
}

bool validate_residuals(const Eigen::MatrixXd &J, const Eigen::VectorXd &target,
                        double ridge, SolveResult *result) {
  if (J.rows() != target.size() || !J.allFinite() || !target.allFinite() ||
      !std::isfinite(ridge) || ridge < 0.0) {
    *result = failure(SolveStatus::kInvalidInput,
        "matching finite residual factors and nonnegative ridge required");
    return false;
  }
  return true;
}

SolveResult optimize(const Eigen::MatrixXd &H, const Eigen::VectorXd &g,
                     const detail::LeastSquaresQuadratic &factors,
                     const Eigen::MatrixXd &A, const Eigen::VectorXd &l,
                     const Eigen::VectorXd &u, const SolveOptions &options,
                     const detail::LeastSquaresQuadratic *rank_factors = nullptr) {
  SolveResult result;
  if (!validate(H.rows(), A, l, u, options, &result)) return result;
  if (!H.allFinite() || !g.allFinite())
    return failure(SolveStatus::kInvalidInput, "quadratic arithmetic overflowed");
  Eigen::MatrixXd C;
  Eigen::VectorXd lower, upper;
  detail::NormalizeLinearConstraintRows(A, l, u, &C, &lower, &upper);
  if (!C.allFinite() || lower.array().isNaN().any() || upper.array().isNaN().any())
    return failure(SolveStatus::kInvalidInput, "constraint normalization overflowed");
  for (Eigen::Index row = 0; row < A.rows(); ++row) {
    if ((std::isfinite(l(row)) && !std::isfinite(lower(row))) ||
        (std::isfinite(u(row)) && !std::isfinite(upper(row))))
      return failure(SolveStatus::kInvalidInput, "finite bound overflowed during normalization");
  }
  if (options.x0 && !(C * *options.x0).allFinite())
    return failure(SolveStatus::kInvalidInput, "x0 constraint arithmetic overflowed");
  VelocitySolverConfig config;
  config.epsilon = options.tolerance;
  config.precision_threshold = std::min(1e-12, options.tolerance);
  config.iteration_limit = options.max_iterations;
  config.regularization_config.regularization_factor = 0.0;
  Eigen::VectorXd seed;
  if (options.x0 && detail::ComputeMaxLinearConstraintViolation(
          C, lower, upper, *options.x0) <= options.tolerance) {
    seed = *options.x0;
  } else if (C.rows() == 0) {
    seed = Eigen::VectorXd::Zero(H.rows());
  } else {
    const auto feasible = detail::FindHardConstraintFeasibleWitness(C, lower, upper, config, &seed);
    if (feasible == detail::HardConstraintFeasibilityStatus::kProvenInfeasible)
      return failure(SolveStatus::kInfeasible, "Phase I certified infeasible hard constraints");
    if (feasible != detail::HardConstraintFeasibilityStatus::kFeasible)
      return failure(SolveStatus::kNoProgress, "Phase I did not certify a feasible point");
  }
  Eigen::VectorXd x;
  std::string reason;
  if (!detail::RecoverFeasibleWeightedQuadratic(
          H, g, C, lower, upper, config, &x, &result.iterations, &seed,
          false, &reason, &factors, options.max_iterations)) {
    result.status = SolveStatus::kNoProgress;
    result.message = reason.empty() ? "optimization did not certify a solution" : reason;
    return result;
  }
  if (!x.allFinite() || !(C*x).allFinite() || !(A*x).allFinite())
    return failure(SolveStatus::kNoProgress, "nonfinite candidate arithmetic");
  // A rank cutoff is not evidence that the discarded direction is exactly
  // zero. For ambiguous rank with nonzero residual, do not claim the minimum
  // of the original objective. Zero residual attains the global LS lower bound.
  // LS callers retain the original factors for this conservative acceptance
  // check: rounding in objective rescaling cannot manufacture zero residual.
  const auto &rank_objective = rank_factors ? *rank_factors : factors;
  if (rank_objective.ridge == 0.0 && rank_objective.matrix.size() > 0) {
    Eigen::CompleteOrthogonalDecomposition<Eigen::MatrixXd> numerical(rank_objective.matrix);
    if (numerical.rank() < std::min(rank_objective.matrix.rows(), rank_objective.matrix.cols()) &&
        !(rank_objective.matrix*x-rank_objective.target).isZero(0.0))
      return failure(SolveStatus::kNoProgress, "nonzero residual with unresolved objective rank");
  }
  result.normalized_violation = detail::ComputeMaxLinearConstraintViolation(C, lower, upper, x);
  result.max_violation = detail::ComputeMaxLinearConstraintViolation(A, l, u, x);
  if (result.normalized_violation > options.tolerance)
    return failure(SolveStatus::kNoProgress, "candidate failed original normalized hard rows");
  result.status = SolveStatus::kSolved;
  result.x = std::move(x);
  return result;
}
} // namespace

SolveResult solve_ls(const Eigen::MatrixXd &J, const Eigen::VectorXd &target,
                     const Eigen::MatrixXd &A, const Eigen::VectorXd &l,
                     const Eigen::VectorXd &u, double ridge,
                     const SolveOptions &options) {
  const auto start = Clock::now();
  const auto finish = [&](SolveResult result) {
    result.elapsed_ms = std::chrono::duration<double, std::milli>(Clock::now()-start).count();
    return result;
  };
  SolveResult invalid;
  if (!validate(J.cols(), A, l, u, options, &invalid)) return finish(std::move(invalid));
  if (!validate_residuals(J, target, ridge, &invalid)) return finish(std::move(invalid));
  // Working-set multiplier tolerances have objective units. Normalize the
  // complete objective by one positive constant so a change of residual units
  // cannot turn a negative multiplier into an apparently optimal active face.
  // Keep residual factors (and explicit ridge) together; never scale rows
  // independently or change their relative weights.
  const double root_ridge = std::sqrt(ridge);
  const double objective_scale = std::max(
      J.size() ? J.cwiseAbs().maxCoeff() : 0.0, root_ridge);
  const Eigen::MatrixXd normalized_J = objective_scale > 0.0 ?
      (J / objective_scale).eval() : J;
  const Eigen::VectorXd normalized_target = objective_scale > 0.0 ?
      (target / objective_scale).eval() : target;
  const double normalized_root_ridge = objective_scale > 0.0 ?
      root_ridge / objective_scale : 0.0;
  const double normalized_ridge = normalized_root_ridge * normalized_root_ridge;
  if (rescaling_lost_coefficients(J, normalized_J) ||
      rescaling_lost_coefficients(target, normalized_target) ||
      (ridge > 0.0 && normalized_ridge == 0.0))
    return finish(failure(SolveStatus::kNoProgress,
        "objective normalization could not preserve finite nonzero coefficients"));
  Eigen::MatrixXd H = normalized_J.transpose()*normalized_J;
  H.diagonal().array() += normalized_ridge;
  const Eigen::VectorXd g = normalized_J.transpose()*normalized_target;
  const detail::LeastSquaresQuadratic factors{
      normalized_J, normalized_target, normalized_ridge};
  const detail::LeastSquaresQuadratic original_factors{J, target, ridge};
  auto result = optimize(H, g, factors, A, l, u, options, &original_factors);
  if (result.success()) {
    const double residual_norm = (J*result.x-target).stableNorm();
    result.objective = (0.5*residual_norm)*residual_norm;
    if (ridge > 0.0) {
      const double regularized_norm = (std::sqrt(ridge)*result.x).stableNorm();
      result.objective += (0.5*regularized_norm)*regularized_norm;
    }
    if (!std::isfinite(result.objective)) result = failure(SolveStatus::kNoProgress, "objective overflowed");
  }
  return finish(std::move(result));
}

SolveResult solve_ls(const Eigen::MatrixXd &J, const Eigen::VectorXd &target,
                     const Eigen::MatrixXd &A, const Eigen::VectorXd &l,
                     const Eigen::VectorXd &u, const SoftBounds &soft,
                     double ridge, const SolveOptions &options) {
  const auto start = Clock::now();
  const auto finish = [&](SolveResult result) {
    result.elapsed_ms = std::chrono::duration<double, std::milli>(Clock::now()-start).count();
    return result;
  };
  SolveResult invalid;
  const Eigen::Index n = J.cols(), m = soft.matrix.rows();
  if (!validate(n, A, l, u, options, &invalid) ||
      !validate_residuals(J, target, ridge, &invalid)) return finish(std::move(invalid));
  if (soft.matrix.cols() != n || soft.lower.size() != m || soft.upper.size() != m ||
      (soft.weights.size() != 0 && soft.weights.size() != m) ||
      !soft.matrix.allFinite() || soft.lower.array().isNaN().any() ||
      soft.upper.array().isNaN().any() || !soft.weights.allFinite() ||
      (soft.weights.array() < 0.0).any())
    return finish(failure(SolveStatus::kInvalidInput,
        "matching finite soft matrix, non-NaN bounds and nonnegative finite weights required"));
  const double infinity = std::numeric_limits<double>::infinity();
  if ((soft.lower.array() > soft.upper.array()).any() ||
      (soft.lower.array() == infinity).any() ||
      (soft.upper.array() == -infinity).any())
    return finish(failure(SolveStatus::kInvalidInput,
        "each soft interval must contain a finite value"));
  std::vector<Eigen::Index> effective;
  for (Eigen::Index i = 0; i < m; ++i) {
    if ((soft.weights.size() == 0 || soft.weights(i) > 0.0) &&
        (std::isfinite(soft.lower(i)) || std::isfinite(soft.upper(i))))
      effective.push_back(i);
  }
  // Preserve the original objective, rank handling and working set when all
  // supplied penalties are disabled. Do not introduce irrelevant variables.
  if (effective.empty()) return solve_ls(J, target, A, l, u, ridge, options);
  const Eigen::Index k = static_cast<Eigen::Index>(effective.size());
  const Eigen::Index ridge_rows = ridge > 0.0 ? n : 0;
  Eigen::MatrixXd factors = Eigen::MatrixXd::Zero(J.rows()+k+ridge_rows, n+k);
  Eigen::VectorXd goals = Eigen::VectorXd::Zero(factors.rows());
  factors.topLeftCorner(J.rows(), n) = J;
  goals.head(target.size()) = target;
  Eigen::MatrixXd constraints = Eigen::MatrixXd::Zero(A.rows()+k, n+k);
  constraints.topLeftCorner(A.rows(), n) = A;
  Eigen::VectorXd lower(A.rows()+k), upper(A.rows()+k);
  lower.head(l.size()) = l;
  upper.head(u.size()) = u;
  SolveOptions augmented_options = options;
  Eigen::VectorXd seed(n+k);
  seed.head(n) = options.x0.value_or(Eigen::VectorXd::Zero(n));
  for (Eigen::Index j = 0; j < k; ++j) {
    const Eigen::Index i = effective[j];
    const double root_weight = std::sqrt(soft.weights.size() == 0 ? 1.0 : soft.weights(i));
    factors.block(J.rows()+j, 0, 1, n) = root_weight*soft.matrix.row(i);
    if (rescaling_lost_coefficients(soft.matrix.row(i), factors.block(J.rows()+j, 0, 1, n)))
      return finish(failure(SolveStatus::kNoProgress,
          "soft weighting could not preserve finite nonzero coefficients"));
    factors(J.rows()+j, n+j) = -root_weight;
    constraints(A.rows()+j, n+j) = 1.0;
    lower(A.rows()+j) = soft.lower(i);
    upper(A.rows()+j) = soft.upper(i);
    const double value = soft.matrix.row(i).dot(seed.head(n));
    if (!std::isfinite(value))
      return finish(failure(SolveStatus::kNoProgress, "soft seed arithmetic overflowed"));
    seed(n+j) = std::clamp(value, soft.lower(i), soft.upper(i));
  }
  if (ridge_rows) factors.bottomLeftCorner(n, n).diagonal().setConstant(std::sqrt(ridge));
  augmented_options.x0 = std::move(seed);
  // For fixed x the minimizing auxiliary is projection of S*x onto its
  // interval. This exact lifted LS problem needs one bounded kernel solve;
  // ridge is explicit on the original columns, never on the auxiliaries.
  auto result = solve_ls(factors, goals, constraints, lower, upper, 0.0, augmented_options);
  if (result.success()) {
    result.x = result.x.head(n).eval();
    const double residual_norm = (J*result.x-target).stableNorm();
    result.objective = (0.5*residual_norm)*residual_norm;
    Eigen::VectorXd penalties(k);
    for (Eigen::Index j = 0; j < k; ++j) {
      const Eigen::Index i = effective[j];
      const double value = soft.matrix.row(i).dot(result.x);
      if (!std::isfinite(value))
        return finish(failure(SolveStatus::kNoProgress, "soft objective arithmetic overflowed"));
      penalties(j) = std::sqrt(soft.weights.size() == 0 ? 1.0 : soft.weights(i))*
          (value-std::clamp(value, soft.lower(i), soft.upper(i)));
    }
    const double penalty_norm = penalties.stableNorm();
    result.objective += (0.5*penalty_norm)*penalty_norm;
    if (ridge > 0.0) {
      const double norm = (std::sqrt(ridge)*result.x).stableNorm();
      result.objective += (0.5*norm)*norm;
    }
    if (!penalties.allFinite() || !std::isfinite(result.objective))
      return finish(failure(SolveStatus::kNoProgress, "soft objective overflowed"));
    result.max_violation = detail::ComputeMaxLinearConstraintViolation(A, l, u, result.x);
    Eigen::MatrixXd normalized_A;
    Eigen::VectorXd normalized_l, normalized_u;
    detail::NormalizeLinearConstraintRows(A, l, u, &normalized_A, &normalized_l, &normalized_u);
    result.normalized_violation = detail::ComputeMaxLinearConstraintViolation(
        normalized_A, normalized_l, normalized_u, result.x);
    if (!std::isfinite(result.max_violation) || !std::isfinite(result.normalized_violation) ||
        result.normalized_violation > options.tolerance)
      return finish(failure(SolveStatus::kNoProgress,
          "candidate failed original normalized hard rows"));
  }
  return finish(std::move(result));
}

SolveResult solve_qp(const Eigen::MatrixXd &P, const Eigen::VectorXd &q,
                     const Eigen::MatrixXd &A, const Eigen::VectorXd &l,
                     const Eigen::VectorXd &u, const SolveOptions &options) {
  const auto start = Clock::now();
  const auto finish = [&](SolveResult result) {
    result.elapsed_ms = std::chrono::duration<double, std::milli>(Clock::now()-start).count();
    return result;
  };
  SolveResult invalid;
  if (!validate(q.size(), A, l, u, options, &invalid)) return finish(std::move(invalid));
  if (P.rows() != q.size() || P.cols() != q.size() || !P.allFinite() || !q.allFinite())
    return finish(failure(SolveStatus::kInvalidInput, "matching finite P and q required"));
  const double matrix_scale = P.cwiseAbs().maxCoeff();
  const Eigen::MatrixXd scaled_P = matrix_scale > 0.0 ? (P/matrix_scale).eval() : P;
  if (!scaled_P.isApprox(scaled_P.transpose(), 1e-12))
    return finish(failure(SolveStatus::kInvalidInput, "symmetric P required"));
  if (rescaling_lost_coefficients(P, scaled_P))
    return finish(failure(SolveStatus::kNoProgress,
        "objective normalization could not preserve finite nonzero coefficients"));
  // Use one objective scale, preserving the arbitrary linear term. The
  // symmetric part defines the same quadratic form; its scale affects neither
  // minimizers nor the hard rows. Report the original objective below.
  const Eigen::MatrixXd H = (0.5*(scaled_P+scaled_P.transpose())).eval();
  Eigen::LLT<Eigen::MatrixXd> decomposition(H);
  if (decomposition.info() != Eigen::Success)
    return finish(failure(SolveStatus::kUnsupported, "CPU solve_qp currently requires positive-definite P; use solve_ls for singular residual objectives"));
  const Eigen::VectorXd normalized_q = q / matrix_scale;
  if (rescaling_lost_coefficients(q, normalized_q))
    return finish(failure(SolveStatus::kNoProgress,
        "objective normalization could not preserve finite nonzero coefficients"));
  const Eigen::MatrixXd R = decomposition.matrixU();
  const Eigen::VectorXd target = R.transpose().triangularView<Eigen::Lower>().solve(-normalized_q);
  const Eigen::VectorXd g = -normalized_q;
  const detail::LeastSquaresQuadratic factors{R, target, 0.0};
  auto result = optimize(H, g, factors, A, l, u, options);
  if (result.success()) {
    const Eigen::MatrixXd original_H = (0.5*P+0.5*P.transpose()).eval();
    result.objective = 0.5*result.x.dot(original_H*result.x)+q.dot(result.x);
    if (!std::isfinite(result.objective)) result = failure(SolveStatus::kNoProgress, "objective overflowed");
  }
  return finish(std::move(result));
}
} // namespace embodik
