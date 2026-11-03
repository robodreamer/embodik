#include <embodik/optimization.hpp>
#include <embodik/control_solver_kernel.hpp>

#include <Eigen/Eigenvalues>
#include <chrono>
#include <cmath>

namespace embodik {
namespace {
using Clock = std::chrono::steady_clock;

SolveResult failure(SolveStatus status, std::string message) {
  SolveResult result;
  result.status = status;
  result.message = std::move(message);
  return result;
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

SolveResult optimize(const Eigen::MatrixXd &H, const Eigen::VectorXd &g,
                     const detail::LeastSquaresQuadratic &factors,
                     const Eigen::MatrixXd &A, const Eigen::VectorXd &l,
                     const Eigen::VectorXd &u, const SolveOptions &options) {
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
  if (factors.ridge == 0.0 && factors.matrix.size() > 0) {
    Eigen::CompleteOrthogonalDecomposition<Eigen::MatrixXd> numerical(factors.matrix);
    if (numerical.rank() < std::min(factors.matrix.rows(), factors.matrix.cols()) &&
        !(factors.matrix*x-factors.target).isZero(0.0))
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
  if (J.rows() != target.size() || !J.allFinite() || !target.allFinite() ||
      !std::isfinite(ridge) || ridge < 0.0)
    return finish(failure(SolveStatus::kInvalidInput, "matching finite residual factors and nonnegative ridge required"));
  Eigen::MatrixXd H = J.transpose()*J;
  H.diagonal().array() += ridge;
  const Eigen::VectorXd g = J.transpose()*target;
  const detail::LeastSquaresQuadratic factors{J, target, ridge};
  auto result = optimize(H, g, factors, A, l, u, options);
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
  // The symmetric part defines precisely the same quadratic form.
  const Eigen::MatrixXd H = (0.5*P+0.5*P.transpose()).eval();
  Eigen::LLT<Eigen::MatrixXd> decomposition(H);
  if (decomposition.info() != Eigen::Success)
    return finish(failure(SolveStatus::kUnsupported, "CPU solve_qp currently requires positive-definite P; use solve_ls for singular residual objectives"));
  const Eigen::MatrixXd R = decomposition.matrixU();
  const Eigen::VectorXd target = R.transpose().triangularView<Eigen::Lower>().solve(-q);
  const Eigen::VectorXd g = -q;
  const detail::LeastSquaresQuadratic factors{R, target, 0.0};
  auto result = optimize(H, g, factors, A, l, u, options);
  if (result.success()) {
    result.objective = 0.5*result.x.dot(H*result.x)+q.dot(result.x);
    if (!std::isfinite(result.objective)) result = failure(SolveStatus::kNoProgress, "objective overflowed");
  }
  return finish(std::move(result));
}
} // namespace embodik
