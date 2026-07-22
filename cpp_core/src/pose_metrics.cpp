#include <embodik/pose_metrics.hpp>

#include <algorithm>
#include <cmath>

namespace embodik {

namespace {
constexpr double kRangeEpsilon = 1e-6;
constexpr double kAbEpsilon = 1e-12;
} // namespace

std::pair<Eigen::VectorXd, double> joint_limit_distance(
    const Eigen::VectorXd &q,
    const Eigen::VectorXd &q_lower,
    const Eigen::VectorXd &q_upper,
    double epsilon) {
  const Eigen::Index n = q.size();
  Eigen::VectorXd per_joint(n);
  double aggregate = 0.0;

  const Eigen::VectorXd ranges = q_upper - q_lower;

  for (Eigen::Index i = 0; i < n; ++i) {
    if (ranges(i) < kRangeEpsilon || !std::isfinite(q_lower(i)) ||
        !std::isfinite(q_upper(i))) {
      per_joint(i) = 0.0;
      continue;
    }
    const double p = 2.0 * (q(i) - q_lower(i)) / ranges(i) - 1.0;
    const double a = 1.0 + epsilon - p;
    const double b = p + 1.0 + epsilon;
    const double ab = a * b;
    if (ab < kAbEpsilon) {
      per_joint(i) = 0.0;
    } else {
      per_joint(i) = p * p / ab;
    }
    aggregate += per_joint(i);
  }

  return {per_joint, aggregate};
}

Eigen::VectorXd joint_limit_distance_gradient(
    const Eigen::VectorXd &q,
    const Eigen::VectorXd &q_lower,
    const Eigen::VectorXd &q_upper,
    double epsilon) {
  const Eigen::Index n = q.size();
  Eigen::VectorXd grad(n);

  const Eigen::VectorXd ranges = q_upper - q_lower;

  for (Eigen::Index i = 0; i < n; ++i) {
    if (ranges(i) < kRangeEpsilon || !std::isfinite(q_lower(i)) ||
        !std::isfinite(q_upper(i))) {
      grad(i) = 0.0;
      continue;
    }
    const double p = 2.0 * (q(i) - q_lower(i)) / ranges(i) - 1.0;
    const double a = 1.0 + epsilon - p;
    const double b = p + 1.0 + epsilon;
    const double ab = a * b;
    double dhdp;
    if (ab < kAbEpsilon) {
      dhdp = 0.0;
    } else {
      dhdp = (2.0 * p * ab - p * p * (a - b)) / (ab * ab);
    }
    const double dpdq = 2.0 / ranges(i);
    grad(i) = -dhdp * dpdq;
  }

  return grad;
}

double velocity_manipulability(const Eigen::MatrixXd &jacobian) {
  const Eigen::MatrixXd jjt = jacobian * jacobian.transpose();
  const double det_val = jjt.determinant();
  return std::sqrt(std::max(0.0, det_val));
}

double singularity_joint_limit_metric(
    const Eigen::VectorXd &q,
    const Eigen::MatrixXd &jacobian,
    const Eigen::VectorXd &q_lower,
    const Eigen::VectorXd &q_upper,
    double epsilon) {
  const double manip = velocity_manipulability(jacobian);
  const auto [per_joint, limit_dist] =
      joint_limit_distance(q, q_lower, q_upper, epsilon);
  (void)per_joint;
  return manip / (limit_dist * limit_dist + 1.0);
}

} // namespace embodik
