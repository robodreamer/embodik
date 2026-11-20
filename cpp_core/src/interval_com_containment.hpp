#pragma once

#include "interval_scalar_model.hpp"
#include <optional>

namespace embodik::detail {

// Physical symmetric ceilings in the literal supplied half-plane coordinates.
// No magnitudes are inferred from containment or from another solver's policy.
struct IntervalComDerivativePolicy {
  Eigen::VectorXd rate_limits;
  Eigen::VectorXd acceleration_limits;
  double comparison_tolerance = 0.0;
};

// Literal half-planes A * com_in_frame.xy <= b. Obtain canonical polygon
// margins from prepare_support_polygon_geometry; this policy never reshrinks or
// normalizes them. Tolerance is in the supplied half-plane residual's units.
struct IntervalComContainmentPolicy {
  Eigen::MatrixXd halfspace_normals;
  Eigen::VectorXd halfspace_offsets;
  std::string frame_name = "world";
  double tolerance = 1e-9;
  std::optional<IntervalComDerivativePolicy> derivatives;
};

namespace interval_com_internal {
// Taylor upper envelope on nonnegative time. Coefficient upper endpoints give
// a scalar quadratic majorant; include its interior maximum, with outward
// vertex/time arithmetic. Inputs are derivatives, not Taylor coefficients.
inline double taylor_quadratic_upper(const Interval &value, const Interval &slope,
                                     const Interval &curvature, const Interval &width) {
  if (width.lower() < 0.0) throw std::domain_error("nonnegative Taylor width required");
  const Interval q(value.upper()), v(slope.upper()), a(curvature.upper());
  const auto at = [&](const Interval &time) {
    return q + v * time + Interval(0.5) * a * square(time);
  };
  double upper = std::max(q.upper(), at(width).upper());
  if (a.upper() < 0.0) {
    const auto vertex = -v / a;
    const double lo = std::max(0.0, vertex.lower()), hi = std::min(width.upper(), vertex.upper());
    if (lo <= hi) upper = std::max(upper, at(Interval(lo, hi)).upper());
  }
  return upper;
}

inline Interval intersect(const Interval &direct, const Interval &tightened) {
  const double lo = std::max(direct.lower(), tightened.lower());
  const double hi = std::min(direct.upper(), tightened.upper());
  if (lo > hi) throw std::domain_error("inconsistent direct and Taylor enclosures");
  return Interval(lo, hi);
}
} // namespace interval_com_internal

// Owned per-invocation snapshot. Optional derivatives support world/root-fixed
// frames only. Braking/recovery/support witnesses and complete stopping remain
// outside this component's contract. Stored rotations are literal linear maps.
class IntervalComContainment {
public:
  IntervalComContainment(const pinocchio::Model &model,
                         const IntervalComContainmentPolicy &policy)
      : tree_(make_interval_scalar_model_catalog(model).joints), policy_(policy) {
    const ScopedIntervalEnvironment environment;
    if (model.inertias.size() != static_cast<std::size_t>(model.njoints))
      throw std::domain_error("native inertia catalog must match scalar joints");
    if (policy_.halfspace_normals.rows() == 0 || policy_.halfspace_normals.cols() != 2 ||
        policy_.halfspace_offsets.size() != policy_.halfspace_normals.rows() ||
        !policy_.halfspace_normals.allFinite() || !policy_.halfspace_offsets.allFinite() ||
        !std::isfinite(policy_.tolerance) || policy_.tolerance < 0.0)
      throw std::domain_error("finite matching CoM half-planes and nonnegative tolerance required");
    for (Eigen::Index i = 0; i < policy_.halfspace_normals.rows(); ++i)
      if ((policy_.halfspace_normals.row(i).array() == 0.0).all())
        throw std::domain_error("nonzero CoM half-plane normal required");
    // Pinocchio centerOfMass excludes the universe inertia at index zero.
    for (int i = 1; i < model.njoints; ++i) {
      const auto &inertia = model.inertias[i];
      if (!std::isfinite(inertia.mass()) || inertia.mass() < 0.0 || !inertia.lever().allFinite())
        throw std::domain_error("finite nonnegative body masses and finite local CoM points required");
      if (inertia.mass() > 0.0) {
        Body body;
        body.parent = i;
        body.mass = Interval(inertia.mass());
        for (int axis = 0; axis < 3; ++axis) body.local[axis] = Interval(inertia.lever()(axis));
        bodies_.push_back(std::move(body));
        requested_.push_back(i);
      }
    }
    if (bodies_.empty()) throw std::domain_error("positive non-universe CoM mass required");
    if (policy_.frame_name != "world") {
      if (policy_.frame_name.empty()) throw std::domain_error("CoM support frame required");
      const auto id = model.getFrameId(policy_.frame_name);
      if (id >= model.frames.size()) throw std::domain_error("CoM support frame not found");
      const auto &frame = model.frames[id];
      if (frame.parentJoint >= static_cast<pinocchio::JointIndex>(model.njoints))
        throw std::domain_error("CoM support frame outside scalar tree");
      frame_parent_ = static_cast<int>(frame.parentJoint);
      frame_placement_ = IntervalTransform::fixed(frame.placement.rotation(), frame.placement.translation());
    }
    requested_.push_back(frame_parent_);
    if (policy_.derivatives) {
      const auto &limits = *policy_.derivatives;
      if (limits.rate_limits.size() != policy_.halfspace_normals.rows() ||
          limits.acceleration_limits.size() != policy_.halfspace_normals.rows() ||
          !limits.rate_limits.allFinite() || !limits.acceleration_limits.allFinite() ||
          (limits.rate_limits.array() <= 0.0).any() ||
          (limits.acceleration_limits.array() <= 0.0).any() ||
          !std::isfinite(limits.comparison_tolerance) || limits.comparison_tolerance < 0.0)
        throw std::domain_error("matching positive finite CoM derivative ceilings and nonnegative tolerance required");
      if (frame_parent_ != 0)
        throw std::domain_error("CoM derivatives require a world or structurally root-fixed frame");
      if (static_cast<std::size_t>(policy_.halfspace_normals.rows()) >
          std::numeric_limits<std::size_t>::max() / 5)
        throw std::domain_error("CoM derivative inequality count overflowed");
    }
  }

  std::size_t inequality_count() const {
    return static_cast<std::size_t>(policy_.halfspace_normals.rows()) * (policy_.derivatives ? 5 : 1);
  }

  // Position and time derivatives through order four in the fixed support
  // frame. Never apply a fixed-frame derivative formula to a moving frame.
  std::array<std::array<Interval, 3>, 5> motion_bounds(
      const QuadraticIntervalMotion &motion, double lo, double hi) const {
    const ScopedIntervalEnvironment environment;
    if (frame_parent_ != 0)
      throw std::domain_error("CoM derivatives require a world or structurally root-fixed frame");
    const auto snapshot = tree_.motion_jets(motion, lo, hi, requested_);
    std::array<std::array<Interval, 3>, 5> weighted, relative;
    Interval mass;
    for (const auto &body : bodies_) {
      mass = mass + body.mass;
      const auto &jet = snapshot.at(body.parent);
      for (int order = 0; order <= 4; ++order) {
        for (int axis = 0; axis < 3; ++axis) {
          Interval point = jet[order].translation[axis];
          for (int col = 0; col < 3; ++col)
            point = point + jet[order].rotation[3 * axis + col] * body.local[col];
          weighted[order][axis] = weighted[order][axis] + body.mass * point;
        }
      }
    }
    if (mass.lower() <= 0.0) throw std::domain_error("CoM total mass is not provably positive");
    for (int order = 0; order <= 4; ++order) {
      for (int row = 0; row < 3; ++row) {
        Interval point = weighted[order][row] / mass;
        if (order == 0) point = point - frame_placement_.translation[row];
        for (int axis = 0; axis < 3; ++axis)
          relative[order][axis] = relative[order][axis] + frame_placement_.rotation[3 * row + axis] * point;
      }
    }
    return relative;
  }

  std::array<Interval, 3> position_bounds(const QuadraticIntervalMotion &motion,
                                        double lo, double hi) const {
    const ScopedIntervalEnvironment environment;
    const auto snapshot = tree_.poses(motion, lo, hi, requested_);
    std::array<Interval, 3> weighted;
    Interval mass;
    for (const auto &body : bodies_) {
      const auto &pose = snapshot.at(body.parent);
      mass = mass + body.mass;
      for (int axis = 0; axis < 3; ++axis) {
        Interval point = pose.translation[axis];
        for (int col = 0; col < 3; ++col)
          point = point + pose.rotation[3 * axis + col] * body.local[col];
        weighted[axis] = weighted[axis] + body.mass * point;
      }
    }
    if (mass.lower() <= 0.0) throw std::domain_error("CoM total mass is not provably positive");
    const auto support = compose(snapshot.at(frame_parent_), frame_placement_);
    std::array<Interval, 3> displacement, relative;
    for (int axis = 0; axis < 3; ++axis)
      displacement[axis] = weighted[axis] / mass - support.translation[axis];
    for (int axis = 0; axis < 3; ++axis)
      for (int row = 0; row < 3; ++row)
        relative[axis] = relative[axis] + support.rotation[3 * row + axis] * displacement[row];
    return relative;
  }

  std::vector<double> lower_bounds(const QuadraticIntervalMotion &motion,
                                   double lo, double hi) const {
    const ScopedIntervalEnvironment environment;
    const auto position = position_bounds(motion, lo, hi);
    std::vector<double> result;
    result.reserve(inequality_count());
    for (Eigen::Index row = 0; row < policy_.halfspace_normals.rows(); ++row) {
      Interval residual = Interval(policy_.halfspace_offsets(row)) + Interval(policy_.tolerance);
      for (int axis = 0; axis < 2; ++axis)
        residual = residual - Interval(policy_.halfspace_normals(row, axis)) * position[axis];
      result.push_back(residual.lower());
    }
    if (policy_.derivatives) {
      const auto whole = motion_bounds(motion, lo, hi);
      const auto left = motion_bounds(motion, lo, lo);
      const Interval width = Interval(hi) - Interval(lo);
      const auto &limits = *policy_.derivatives;
      for (Eigen::Index row = 0; row < policy_.halfspace_normals.rows(); ++row) {
        std::array<Interval, 5> direct, point;
        for (int order = 1; order <= 4; ++order) {
          for (int axis = 0; axis < 2; ++axis) {
            const Interval normal(policy_.halfspace_normals(row, axis));
            direct[order] = direct[order] + normal * whole[order][axis];
            point[order] = point[order] + normal * left[order][axis];
          }
        }
        for (int order = 1; order <= 2; ++order) {
          const double upper = interval_com_internal::taylor_quadratic_upper(
              point[order], point[order+1], direct[order+2], width);
          const double negative_upper = interval_com_internal::taylor_quadratic_upper(
              -point[order], -point[order+1], -direct[order+2], width);
          const auto bounded = interval_com_internal::intersect(direct[order], Interval(-negative_upper, upper));
          const double ceiling = order == 1 ? limits.rate_limits(row) : limits.acceleration_limits(row);
          const Interval allowed = Interval(ceiling) + Interval(limits.comparison_tolerance);
          result.push_back((allowed - bounded).lower());
          result.push_back((allowed + bounded).lower());
        }
      }
    }
    return result;
  }

private:
  struct Body {
    int parent = 0;
    Interval mass;
    std::array<Interval, 3> local;
  };
  IntervalScalarKinematics tree_;
  IntervalComContainmentPolicy policy_;
  std::vector<Body> bodies_;
  std::vector<int> requested_;
  int frame_parent_ = 0;
  IntervalTransform frame_placement_ = IntervalTransform::identity();
};

} // namespace embodik::detail
