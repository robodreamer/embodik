#pragma once

#include "bounded_interval.hpp"

#include <Eigen/Core>
#include <array>
#include <utility>
#include <vector>

namespace embodik::detail {

struct IntervalTransform {
  std::array<Interval, 9> rotation;
  std::array<Interval, 3> translation;

  static IntervalTransform identity() {
    IntervalTransform result;
    for (int i = 0; i < 3; ++i) result.rotation[3 * i + i] = Interval(1.0);
    return result;
  }
  static IntervalTransform fixed(const Eigen::Matrix3d &rotation,
                                 const Eigen::Vector3d &translation) {
    IntervalTransform result;
    for (int i = 0; i < 3; ++i) {
      result.translation[i] = Interval(translation(i));
      for (int j = 0; j < 3; ++j) result.rotation[3 * i + j] = Interval(rotation(i, j));
    }
    return result;
  }
};

inline IntervalTransform compose(const IntervalTransform &a,
                                 const IntervalTransform &b) {
  IntervalTransform result;
  for (int i = 0; i < 3; ++i) {
    result.translation[i] = a.translation[i];
    for (int k = 0; k < 3; ++k)
      result.translation[i] = result.translation[i] + a.rotation[3 * i + k] * b.translation[k];
    for (int j = 0; j < 3; ++j)
      for (int k = 0; k < 3; ++k)
        result.rotation[3 * i + j] = result.rotation[3 * i + j] +
            a.rotation[3 * i + k] * b.rotation[3 * k + j];
  }
  return result;
}

class QuadraticIntervalMotion {
public:
  QuadraticIntervalMotion(Eigen::VectorXd q, Eigen::VectorXd dq,
                          Eigen::VectorXd ddq, double dt)
      : q_(std::move(q)), dq_(std::move(dq)), ddq_(std::move(ddq)), dt_(dt) {
    if (q_.size() <= 0 || dq_.size() != q_.size() || ddq_.size() != q_.size() ||
        !q_.allFinite() || !dq_.allFinite() || !ddq_.allFinite() ||
        !std::isfinite(dt_) || dt_ <= 0.0)
      throw std::domain_error("finite matching quadratic motion required");
  }
  Eigen::Index size() const { return q_.size(); }
  double dt() const { return dt_; }
  void validate_times(double lo, double hi) const {
    if (!std::isfinite(lo) || !std::isfinite(hi) || lo < 0.0 || lo > hi || hi > dt_)
      throw std::domain_error("motion subinterval outside declared horizon");
  }
  Interval position_range(Eigen::Index index, double lo, double hi) const {
    const ScopedIntervalEnvironment environment;
    return position_range_impl(index, lo, hi);
  }
  Interval velocity_range(Eigen::Index index, double lo, double hi) const {
    const ScopedIntervalEnvironment environment;
    validate_times(lo, hi);
    if (index < 0 || index >= size()) throw std::domain_error("invalid coordinate index");
    return Interval(dq_(index)) + Interval(lo, hi) * Interval(ddq_(index));
  }
  Interval acceleration_range(Eigen::Index index) const {
    if (index < 0 || index >= size()) throw std::domain_error("invalid coordinate index");
    return Interval(ddq_(index));
  }
  // Accumulate literal affine coefficients outward, before ranging in time.
  // A rounded Eigen dot product would define a different polynomial.
  Interval affine_position_range(const Eigen::VectorXd &coefficients, double bias,
                                 double lo, double hi) const {
    const ScopedIntervalEnvironment environment;
    validate_times(lo, hi);
    if (coefficients.size() != size() || !coefficients.allFinite() || !std::isfinite(bias))
      throw std::domain_error("matching finite affine position row required");
    Interval q(bias), v, a;
    for (Eigen::Index i = 0; i < size(); ++i) {
      const Interval coefficient(coefficients(i));
      q = q + coefficient * Interval(q_(i));
      v = v + coefficient * Interval(dq_(i));
      a = a + coefficient * Interval(ddq_(i));
    }
    const auto at = [&](const Interval &t) { return q + t * (v + Interval(0.5) * a * t); };
    // Coefficient enclosures can straddle zero even when the exact coefficient
    // does not. Horner over the whole time interval remains sound in that case.
    if (a.lower() <= 0.0 && a.upper() >= 0.0 && !a.point(0.0))
      return at(Interval(lo, hi));
    const auto left = at(Interval(lo)), right = at(Interval(hi));
    double lower = std::min(left.lower(), right.lower());
    double upper = std::max(left.upper(), right.upper());
    if (!a.point(0.0)) {
      const Interval vertex = -v / a;
      const double start = std::max(lo, vertex.lower()), end = std::min(hi, vertex.upper());
      if (start <= end) {
        const auto value = at(Interval(start, end));
        lower = std::min(lower, value.lower());
        upper = std::max(upper, value.upper());
      }
    }
    return Interval(lower, upper);
  }

private:
  friend class IntervalScalarKinematics;
  Interval centered_position_range(Eigen::Index index, double lo, double hi) const {
    validate_times(lo, hi);
    if (index < 0 || index >= size()) throw std::domain_error("invalid coordinate index");
    const double midpoint = std::clamp(lo + 0.5 * (hi - lo), lo, hi);
    const Interval q(q_(index)), v(dq_(index)), a(ddq_(index)), center(midpoint);
    // The center is a representable TIME, not a rounded replacement state.
    const Interval qc = q + center * (v + Interval(0.5) * a * center);
    const Interval vc = v + a * center;
    const auto at = [&](const Interval &time) {
      const Interval delta = time - center;
      return qc + delta * (vc + Interval(0.5) * a * delta);
    };
    const auto left = at(Interval(lo)), right = at(Interval(hi));
    double lower = std::min(left.lower(), right.lower());
    double upper = std::max(left.upper(), right.upper());
    if (!a.point(0.0)) {
      const Interval vertex = -v / a;
      const double start = std::max(lo, vertex.lower()), end = std::min(hi, vertex.upper());
      if (start <= end) {
        const auto value = at(Interval(start, end));
        lower = std::min(lower, value.lower());
        upper = std::max(upper, value.upper());
      }
    }
    return Interval(lower, upper);
  }
  Interval position_range_impl(Eigen::Index index, double lo, double hi) const {
    validate_times(lo, hi);
    if (index < 0 || index >= size()) throw std::domain_error("invalid coordinate index");
    const Interval q(q_(index)), v(dq_(index)), a(ddq_(index));
    const auto at = [&](const Interval &t) { return q + t * (v + Interval(0.5) * a * t); };
    const auto left = at(Interval(lo)), right = at(Interval(hi));
    double lower = std::min(left.lower(), right.lower());
    double upper = std::max(left.upper(), right.upper());
    if (!a.point(0.0)) {
      const Interval vertex = -v / a;
      const double start = std::max(lo, vertex.lower()), end = std::min(hi, vertex.upper());
      if (start <= end) {
        const auto value = at(Interval(start, end));
        lower = std::min(lower, value.lower());
        upper = std::max(upper, value.upper());
      }
    }
    return Interval(lower, upper);
  }

  Eigen::VectorXd q_, dq_, ddq_;
  double dt_;
};

enum class IntervalJointKind { kRevolute, kPrismatic };

struct IntervalScalarJoint {
  int parent = 0;
  IntervalJointKind kind = IntervalJointKind::kRevolute;
  Eigen::Vector3d axis = Eigen::Vector3d::UnitZ();
  IntervalTransform placement = IntervalTransform::identity();
};

// Value and TIME derivatives, not factorial-scaled Taylor coefficients.
using IntervalTransformMotion = std::array<IntervalTransform, 5>;

inline IntervalTransformMotion fixed_motion(const IntervalTransform &pose) {
  IntervalTransformMotion result;
  result[0] = pose;
  return result;
}

inline IntervalTransformMotion compose_motion(const IntervalTransformMotion &a,
                                             const IntervalTransformMotion &b) {
  // Small exact integer coefficients; no factorial arithmetic or dynamic AD.
  constexpr int binomial[5][5] = {{1}, {1,1}, {1,2,1}, {1,3,3,1}, {1,4,6,4,1}};
  IntervalTransformMotion result;
  for (int order = 0; order <= 4; ++order) {
    for (int row = 0; row < 3; ++row) {
      result[order].translation[row] = a[order].translation[row];
      for (int part = 0; part <= order; ++part) {
        const Interval coefficient(binomial[order][part]);
        for (int k = 0; k < 3; ++k) {
          result[order].translation[row] = result[order].translation[row] +
              coefficient * a[part].rotation[3 * row + k] * b[order-part].translation[k];
          for (int col = 0; col < 3; ++col)
            result[order].rotation[3 * row + col] = result[order].rotation[3 * row + col] +
                coefficient * a[part].rotation[3 * row + k] * b[order-part].rotation[3 * k + col];
        }
      }
    }
  }
  return result;
}

class IntervalKinematicMotionSnapshot {
public:
  const IntervalTransformMotion &at(std::size_t index) const {
    if (index >= jets_.size() || !computed_[index])
      throw std::domain_error("unrequested interval joint motion");
    return jets_[index];
  }
private:
  friend class IntervalScalarKinematics;
  explicit IntervalKinematicMotionSnapshot(std::size_t size) : jets_(size), computed_(size, false) {}
  std::vector<IntervalTransformMotion> jets_;
  std::vector<bool> computed_;
};

class IntervalKinematicSnapshot {
public:
  const IntervalTransform &at(std::size_t index) const {
    if (index >= poses_.size() || !computed_[index])
      throw std::domain_error("unrequested interval joint pose");
    return poses_[index];
  }
private:
  friend class IntervalScalarKinematics;
  explicit IntervalKinematicSnapshot(std::size_t size) : poses_(size), computed_(size, false) {}
  std::vector<IntervalTransform> poses_;
  std::vector<bool> computed_;
};

class IntervalScalarKinematics {
public:
  explicit IntervalScalarKinematics(std::vector<IntervalScalarJoint> joints)
      : joints_(std::move(joints)) {
    const ScopedIntervalEnvironment environment;
    if (joints_.empty()) throw std::domain_error("nonempty scalar tree required");
    for (std::size_t i = 0; i < joints_.size(); ++i) {
      const auto &joint = joints_[i];
      if (joint.parent < 0 || static_cast<std::size_t>(joint.parent) > i || !joint.axis.allFinite())
        throw std::domain_error("ordered scalar tree with finite axes required");
      if (joint.kind == IntervalJointKind::kRevolute) {
        int nonzero = 0;
        for (double x : joint.axis) {
          if (x != 0.0 && std::abs(x) != 1.0)
            throw std::domain_error("exact signed coordinate rotation axis required");
          nonzero += x != 0.0;
        }
        if (nonzero != 1) throw std::domain_error("one nonzero rotation axis required");
      } else if (joint.kind != IntervalJointKind::kPrismatic || (joint.axis.array() == 0.0).all()) {
        throw std::domain_error("nonzero prismatic axis required");
      }
    }
  }

  IntervalKinematicSnapshot poses(const QuadraticIntervalMotion &motion,
                                  double lo, double hi,
                                  const std::vector<int> &requested) const {
    const ScopedIntervalEnvironment environment;
    motion.validate_times(lo, hi);
    if (motion.size() != static_cast<Eigen::Index>(joints_.size()))
      throw std::domain_error("motion and scalar tree dimensions differ");
    std::vector<bool> needed(joints_.size() + 1, false);
    needed[0] = true;
    for (int index : requested) {
      if (index < 0 || static_cast<std::size_t>(index) > joints_.size())
        throw std::domain_error("requested joint outside scalar tree");
      while (!needed[index]) {
        needed[index] = true;
        index = joints_[index - 1].parent;
      }
    }
    IntervalKinematicSnapshot result(joints_.size() + 1);
    result.poses_[0] = IntervalTransform::identity();
    result.computed_[0] = true;
    for (std::size_t i = 0; i < joints_.size(); ++i) {
      if (!needed[i + 1]) continue;
      const auto &joint = joints_[i];
      const Interval coordinate = motion.position_range_impl(static_cast<Eigen::Index>(i), lo, hi);
      auto local = IntervalTransform::identity();
      if (joint.kind == IntervalJointKind::kPrismatic) {
        for (int axis = 0; axis < 3; ++axis)
          local.translation[axis] = coordinate * Interval(joint.axis(axis));
      } else {
        const auto trig = sin_cos(coordinate);
        Eigen::Matrix3d skew;
        const auto &a = joint.axis;
        skew << 0.0, -a.z(), a.y(), a.z(), 0.0, -a.x(), -a.y(), a.x(), 0.0;
        const Eigen::Matrix3d squared = skew * skew;
        for (int row = 0; row < 3; ++row)
          for (int col = 0; col < 3; ++col)
            local.rotation[3 * row + col] = local.rotation[3 * row + col] +
                trig.first * Interval(skew(row, col)) +
                (Interval(1.0) - trig.second) * Interval(squared(row, col));
      }
      result.poses_[i + 1] = compose(compose(result.poses_[joint.parent], joint.placement), local);
      result.computed_[i + 1] = true;
    }
    return result;
  }

  // Additive scalar quadratic trajectories only. Existing poses() and its
  // numerical path remain unchanged when derivatives are not requested.
  IntervalKinematicMotionSnapshot motion_jets(const QuadraticIntervalMotion &motion,
                                             double lo, double hi,
                                             const std::vector<int> &requested) const {
    const ScopedIntervalEnvironment environment;
    motion.validate_times(lo, hi);
    if (motion.size() != static_cast<Eigen::Index>(joints_.size()))
      throw std::domain_error("motion and scalar tree dimensions differ");
    std::vector<bool> needed(joints_.size() + 1, false);
    needed[0] = true;
    for (int index : requested) {
      if (index < 0 || static_cast<std::size_t>(index) > joints_.size())
        throw std::domain_error("requested joint outside scalar tree");
      while (!needed[index]) {
        needed[index] = true;
        index = joints_[index - 1].parent;
      }
    }
    IntervalKinematicMotionSnapshot result(joints_.size() + 1);
    result.jets_[0] = fixed_motion(IntervalTransform::identity());
    result.computed_[0] = true;
    for (std::size_t i = 0; i < joints_.size(); ++i) {
      if (!needed[i + 1]) continue;
      const auto &joint = joints_[i];
      const auto coordinate = motion.centered_position_range(static_cast<Eigen::Index>(i), lo, hi);
      const Interval rate = Interval(motion.dq_(i)) + Interval(motion.ddq_(i)) * Interval(lo, hi);
      const Interval acceleration(motion.ddq_(i));
      auto local = fixed_motion(IntervalTransform::identity());
      if (joint.kind == IntervalJointKind::kPrismatic) {
        for (int axis = 0; axis < 3; ++axis) {
          local[0].translation[axis] = coordinate * Interval(joint.axis(axis));
          local[1].translation[axis] = rate * Interval(joint.axis(axis));
          local[2].translation[axis] = acceleration * Interval(joint.axis(axis));
        }
      } else {
        const auto trig = sin_cos(coordinate);
        const auto &axis = joint.axis;
        Eigen::Matrix3d skew;
        skew << 0, -axis.z(), axis.y(), axis.z(), 0, -axis.x(), -axis.y(), axis.x(), 0;
        const Eigen::Matrix3d squared = skew * skew;
        const Interval v2 = square(rate), v3 = v2 * rate, v4 = square(v2);
        for (int row = 0; row < 3; ++row) {
          for (int col = 0; col < 3; ++col) {
            const auto index = 3 * row + col;
            const Interval k(skew(row,col)), k2(squared(row,col));
            const Interval first = trig.second * k + trig.first * k2;
            const Interval second = -trig.first * k + trig.second * k2;
            local[0].rotation[index] = local[0].rotation[index] +
                trig.first * k + (Interval(1) - trig.second) * k2;
            local[1].rotation[index] = rate * first;
            local[2].rotation[index] = acceleration * first + v2 * second;
            local[3].rotation[index] = Interval(3) * rate * acceleration * second - v3 * first;
            local[4].rotation[index] = Interval(3) * square(acceleration) * second -
                Interval(6) * v2 * acceleration * first - v4 * second;
          }
        }
      }
      result.jets_[i+1] = compose_motion(
          compose_motion(result.jets_[joint.parent], fixed_motion(joint.placement)), local);
      result.computed_[i+1] = true;
    }
    return result;
  }

private:
  std::vector<IntervalScalarJoint> joints_;
};

} // namespace embodik::detail
