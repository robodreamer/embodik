#pragma once

#include <Eigen/Core>
#include <limits>
#include <optional>
#include <string>

namespace embodik {

enum class SolveStatus { kSolved, kInfeasible, kNoProgress, kInvalidInput, kUnsupported };

/// Numerical optimization result, independent of robot state or integration.
/// Only kSolved provides x. It is not a robot execution certificate.
struct SolveResult {
  SolveStatus status = SolveStatus::kInvalidInput;
  Eigen::VectorXd x;
  std::string message;
  std::string backend = "cpu";
  double objective = std::numeric_limits<double>::quiet_NaN();
  double max_violation = std::numeric_limits<double>::quiet_NaN();
  double normalized_violation = std::numeric_limits<double>::quiet_NaN();
  double elapsed_ms = 0.0;
  unsigned int iterations = 0;
  bool success() const { return status == SolveStatus::kSolved; }
};

struct SolveOptions {
  double tolerance = 1e-8;
  /// Optimization working-set budget; feasibility has its own bounded search.
  unsigned int max_iterations = 200;
  std::optional<Eigen::VectorXd> x0;
};

/// Weighted distances from affine values to closed scalar intervals.
/// Empty weights mean unit weights. Zero weights and [-inf,+inf] rows have
/// no effect. Bounds describe penalties, never additional hard constraints.
struct SoftBounds {
  Eigen::MatrixXd matrix;
  Eigen::VectorXd lower, upper;
  Eigen::VectorXd weights;
};

/// min .5 ||J x - target||^2 + .5 ridge ||x||^2, l <= A x <= u.
/// Zero ridge preserves the exact least-squares objective, including rank loss.
SolveResult solve_ls(const Eigen::MatrixXd &J, const Eigen::VectorXd &target,
                     const Eigen::MatrixXd &A, const Eigen::VectorXd &l,
                     const Eigen::VectorXd &u, double ridge = 0.0,
                     const SolveOptions &options = {});

/// Add .5 sum_i weights_i dist((soft.matrix*x)_i,
///                             [soft.lower_i,soft.upper_i])^2 to solve_ls.
/// Ridge applies only to x. Initial CPU implementation uses one auxiliary
/// projection variable per effective soft row and the existing LS kernel.
/// Ridge and options are explicit to keep existing brace arguments unambiguous.
SolveResult solve_ls(const Eigen::MatrixXd &J, const Eigen::VectorXd &target,
                     const Eigen::MatrixXd &A, const Eigen::VectorXd &l,
                     const Eigen::VectorXd &u, const SoftBounds &soft,
                     double ridge, const SolveOptions &options);

/// min .5 x' P x + q' x, l <= A x <= u.
/// Initial CPU implementation requires positive-definite P. Other Hessian
/// families report UNSUPPORTED rather than adding an implicit regularizer.
SolveResult solve_qp(const Eigen::MatrixXd &P, const Eigen::VectorXd &q,
                     const Eigen::MatrixXd &A, const Eigen::VectorXd &l,
                     const Eigen::VectorXd &u,
                     const SolveOptions &options = {});

} // namespace embodik
