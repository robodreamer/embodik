#pragma once

#include "interval_scalar_model.hpp"

#include <optional>

namespace embodik::detail {

struct IntervalScalarPositionBounds {
  Eigen::VectorXd lower, upper;
  double comparison_tolerance = 0.0;
};

struct IntervalScalarMagnitudeBounds {
  Eigen::VectorXd maximum;
  double comparison_tolerance = 0.0;
};

struct IntervalAffinePositionBounds {
  Eigen::MatrixXd matrix;
  Eigen::VectorXd bias, lower, upper;
  double comparison_tolerance = 0.0;
};

// All ceilings and comparisons are explicit caller policy. In particular,
// affine POSITION rows imply no rate, acceleration, braking or recovery rule.
struct IntervalScalarStatePolicy {
  std::vector<std::string> coordinate_names;
  std::optional<IntervalScalarPositionBounds> position;
  std::optional<IntervalScalarMagnitudeBounds> velocity;
  std::optional<IntervalScalarMagnitudeBounds> acceleration;
  std::optional<IntervalAffinePositionBounds> affine_position;
};

// Owned policy on the literal exact-real quadratic represented by the motion.
// This is not a rounded native endpoint/publication or connection certificate.
// Adding these residuals to the common walker does not change that boundary:
// q_next/dq_next copies and any rounding connection require separate admission.
class IntervalScalarStateContainment {
public:
  IntervalScalarStateContainment(const pinocchio::Model &model,
                                 IntervalScalarStatePolicy policy)
      : policy_(std::move(policy)) {
    const auto catalog = make_interval_scalar_model_catalog(model);
    size_ = static_cast<Eigen::Index>(catalog.joint_names.size());
    if (policy_.coordinate_names != catalog.joint_names)
      throw std::domain_error("scalar policy coordinate names must match owned native ordering");
    if (policy_.position) {
      const auto &p = *policy_.position;
      validate_bounds(p.lower, p.upper, size_, p.comparison_tolerance);
      count_ += 2 * static_cast<std::size_t>(size_);
    }
    const auto magnitude = [&](const std::optional<IntervalScalarMagnitudeBounds> &p) {
      if (!p) return;
      validate_tolerance(p->comparison_tolerance);
      if (p->maximum.size() != size_ || !p->maximum.allFinite() ||
          (p->maximum.array() < 0.0).any())
        throw std::domain_error("matching finite nonnegative scalar ceilings required");
      count_ += static_cast<std::size_t>(size_);
    };
    magnitude(policy_.velocity);
    magnitude(policy_.acceleration);
    if (policy_.affine_position) {
      const auto &p = *policy_.affine_position;
      if (p.matrix.rows() <= 0 || p.matrix.cols() != size_ || !p.matrix.allFinite() ||
          p.bias.size() != p.matrix.rows() || !p.bias.allFinite())
        throw std::domain_error("nonempty matching finite affine position map required");
      validate_bounds(p.lower, p.upper, p.matrix.rows(), p.comparison_tolerance);
      count_ += 2 * static_cast<std::size_t>(p.matrix.rows());
    }
    if (!count_) throw std::domain_error("at least one declared scalar or affine position family required");
  }

  std::size_t inequality_count() const { return count_; }

  std::vector<double> lower_bounds(const QuadraticIntervalMotion &motion,
                                  double lo, double hi) const {
    const ScopedIntervalEnvironment environment;
    motion.validate_times(lo, hi);
    if (motion.size() != size_)
      throw std::domain_error("scalar policy and quadratic motion dimensions differ");
    std::vector<double> result;
    result.reserve(count_);
    if (policy_.position) {
      const auto &p = *policy_.position;
      for (Eigen::Index i = 0; i < size_; ++i)
        append_bounds(result, motion.position_range(i, lo, hi), p.lower(i), p.upper(i),
                      p.comparison_tolerance);
    }
    const auto magnitude = [&](const std::optional<IntervalScalarMagnitudeBounds> &p,
                               bool acceleration) {
      if (!p) return;
      for (Eigen::Index i = 0; i < size_; ++i) {
        const Interval value = acceleration ? motion.acceleration_range(i)
                                            : motion.velocity_range(i, lo, hi);
        const double absolute_upper = std::max(std::abs(value.lower()), std::abs(value.upper()));
        result.push_back((Interval(p->maximum(i)) + Interval(p->comparison_tolerance) -
                          Interval(absolute_upper)).lower());
      }
    };
    magnitude(policy_.velocity, false);
    magnitude(policy_.acceleration, true);
    if (policy_.affine_position) {
      const auto &p = *policy_.affine_position;
      for (Eigen::Index i = 0; i < p.matrix.rows(); ++i)
        append_bounds(result, motion.affine_position_range(p.matrix.row(i).transpose(), p.bias(i), lo, hi),
                      p.lower(i), p.upper(i), p.comparison_tolerance);
    }
    return result;
  }

private:
  static void validate_tolerance(double tolerance) {
    if (!std::isfinite(tolerance) || tolerance < 0.0)
      throw std::domain_error("finite nonnegative explicit comparison tolerance required");
  }
  static void validate_bounds(const Eigen::VectorXd &lower, const Eigen::VectorXd &upper,
                              Eigen::Index size, double tolerance) {
    validate_tolerance(tolerance);
    if (lower.size() != size || upper.size() != size || !lower.allFinite() || !upper.allFinite() ||
        (lower.array() > upper.array()).any())
      throw std::domain_error("matching finite ordered explicit position bounds required");
  }
  static void append_bounds(std::vector<double> &result, const Interval &value,
                            double lower, double upper, double tolerance) {
    result.push_back((Interval(value.lower()) - Interval(lower) + Interval(tolerance)).lower());
    result.push_back((Interval(upper) + Interval(tolerance) - Interval(value.upper())).lower());
  }
  IntervalScalarStatePolicy policy_;
  Eigen::Index size_ = 0;
  std::size_t count_ = 0;
};

} // namespace embodik::detail
