#pragma once

#include "interval_scalar_kinematics.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <tuple>
#include <utility>
#include <vector>

namespace embodik::detail {

enum class IntervalPrimitiveKind { kBox, kSphere, kCylinder };

// Dimensions: box half-extents; sphere radius; cylinder radius/half-length.
// The swept radius belongs to the primitive's local coordinates.
struct IntervalPrimitive {
  IntervalPrimitiveKind kind = IntervalPrimitiveKind::kSphere;
  std::vector<double> dimensions;
  double swept_radius = 0.0;

  void validate() const {
    const ScopedIntervalEnvironment environment;
    validate_impl();
  }

  // The returned value bounds support over an interval of local directions.
  // No orthogonality assumption is needed for a stored attachment matrix.
  double support_upper(const std::array<Interval, 3> &direction) const {
    const ScopedIntervalEnvironment environment;
    return support_upper_impl(direction);
  }

private:
  friend class IntervalPrimitiveProjection;
  void validate_impl() const {
    std::size_t expected;
    switch (kind) {
    case IntervalPrimitiveKind::kBox: expected = 3; break;
    case IntervalPrimitiveKind::kSphere: expected = 1; break;
    case IntervalPrimitiveKind::kCylinder: expected = 2; break;
    default: throw std::domain_error("unsupported interval primitive");
    }
    if (dimensions.size() != expected || !std::isfinite(swept_radius) || swept_radius < 0.0)
      throw std::domain_error("finite nonnegative matching primitive dimensions required");
    for (double size : dimensions)
      if (!std::isfinite(size) || size < 0.0)
        throw std::domain_error("finite nonnegative primitive dimensions required");
  }

  double support_upper_impl(const std::array<Interval, 3> &direction) const {
    validate_impl();
    std::array<Interval, 3> sizes;
    for (int i = 0; i < 3; ++i)
      sizes[i] = Interval(std::max(std::abs(direction[i].lower()),
                                   std::abs(direction[i].upper())));
    const auto norm = sqrt(square(sizes[0]) + square(sizes[1]) + square(sizes[2]));
    Interval value;
    switch (kind) {
    case IntervalPrimitiveKind::kBox:
      for (int i = 0; i < 3; ++i) value = value + sizes[i] * Interval(dimensions[i]);
      break;
    case IntervalPrimitiveKind::kSphere:
      value = Interval(dimensions[0]) * norm;
      break;
    case IntervalPrimitiveKind::kCylinder:
      value = Interval(dimensions[0]) * sqrt(square(sizes[0]) + square(sizes[1])) +
              Interval(dimensions[1]) * sizes[2];
      break;
    default: throw std::domain_error("unsupported interval primitive");
    }
    return (value + Interval(swept_radius) * norm).upper();
  }
};

struct IntervalAttachedPrimitive {
  int parent = 0;
  IntervalPrimitive shape;
  IntervalTransform placement = IntervalTransform::identity();
};

namespace interval_projection_internal {
// Literal value, first and second TIME derivatives (not Taylor coefficients).
using ScalarMotion = std::array<Interval, 3>;
using DirectionMotion = std::array<ScalarMotion, 3>;

inline ScalarMotion scaled(const ScalarMotion &value, double scale) {
  ScalarMotion result;
  for (int order = 0; order < 3; ++order) result[order] = value[order] * Interval(scale);
  return result;
}

inline void accumulate(ScalarMotion &sum, const ScalarMotion &value) {
  for (int order = 0; order < 3; ++order) sum[order] = sum[order] + value[order];
}

inline ScalarMotion norm_motion(const DirectionMotion &direction, int dimensions) {
  if (dimensions < 1 || dimensions > 3)
    throw std::domain_error("one to three support norm components required");
  ScalarMotion squared;
  for (int i = 0; i < dimensions; ++i) {
    const auto &d = direction[i];
    squared[0] = squared[0] + square(d[0]);
    squared[1] = squared[1] + Interval(2.0) * d[0] * d[1];
    squared[2] = squared[2] + Interval(2.0) * (square(d[1]) + d[0] * d[2]);
  }
  if (squared[0].lower() <= 0.0)
    throw std::domain_error("support norm is not provably positive");
  ScalarMotion result;
  result[0] = sqrt(squared[0]);
  const Interval divisor = Interval(2.0) * result[0];
  result[1] = squared[1] / divisor;
  result[2] = (squared[2] - Interval(2.0) * square(result[1])) / divisor;
  return result;
}

inline ScalarMotion signed_absolute(const ScalarMotion &value,
                                    const ScalarMotion &whole) {
  // The sign is proved over the WHOLE interval, including its boundary.
  // This also handles an identically zero component without a smoothness guess.
  if (whole[0].lower() >= 0.0) return value;
  if (whole[0].upper() <= 0.0) return scaled(value, -1.0);
  throw std::domain_error("support absolute value crosses zero");
}

inline ScalarMotion support_motion(const IntervalPrimitive &shape,
                                   const DirectionMotion &direction,
                                   const DirectionMotion &whole) {
  shape.validate();
  ScalarMotion result;
  const auto add_norm = [&](double scale, int dimensions) {
    if (scale == 0.0) return;
    // Validate the complete domain before evaluating an anchor or subrange.
    (void)norm_motion(whole, dimensions);
    accumulate(result, scaled(norm_motion(direction, dimensions), scale));
  };
  switch (shape.kind) {
  case IntervalPrimitiveKind::kBox:
    for (int axis = 0; axis < 3; ++axis)
      if (shape.dimensions[axis] != 0.0)
        accumulate(result, scaled(signed_absolute(direction[axis], whole[axis]), shape.dimensions[axis]));
    break;
  case IntervalPrimitiveKind::kSphere:
    add_norm(shape.dimensions[0], 3);
    break;
  case IntervalPrimitiveKind::kCylinder:
    add_norm(shape.dimensions[0], 2);
    if (shape.dimensions[1] != 0.0)
      accumulate(result, scaled(signed_absolute(direction[2], whole[2]), shape.dimensions[1]));
    break;
  default: throw std::domain_error("unsupported support motion primitive");
  }
  // Literal attachment matrices need not be orthogonal. Sphere and swept
  // support therefore retain ||R^T n|| rather than assuming a unit local axis.
  add_norm(shape.swept_radius, 3);
  return result;
}

inline double quadratic_lower(const Interval &value, const Interval &slope,
                               const Interval &curvature, const Interval &width) {
  if (width.lower() < 0.0) throw std::domain_error("nonnegative Taylor width required");
  const Interval q(value.lower()), v(slope.lower()), a(curvature.lower());
  const auto at = [&](const Interval &time) {
    return q + v * time + Interval(0.5) * a * square(time);
  };
  double lower = std::min(q.lower(), at(width).lower());
  if (a.lower() > 0.0) {
    const Interval vertex = -v / a;
    const double lo = std::max(0.0, vertex.lower()), hi = std::min(width.upper(), vertex.upper());
    if (lo <= hi) lower = std::min(lower, at(Interval(lo, hi)).lower());
  }
  return lower;
}
} // namespace interval_projection_internal

// Owned geometry and scalar tree, independent of a robot/query cache. Six fixed
// unit world axes give conservative separation bounds throughout each interval.
// A negative result is not a penetration-depth estimate or an infeasibility
// certificate. This component neither admits a command nor certifies stopping.
class IntervalPrimitiveProjection {
public:
  IntervalPrimitiveProjection(std::vector<IntervalScalarJoint> joints,
                              std::vector<IntervalAttachedPrimitive> geometries)
      : joint_count_(joints.size()), tree_(std::move(joints)),
        geometries_(std::move(geometries)) {
    const ScopedIntervalEnvironment environment;
    if (geometries_.empty()) throw std::domain_error("nonempty primitive catalog required");
    for (const auto &geometry : geometries_) {
      if (geometry.parent < 0 || static_cast<std::size_t>(geometry.parent) > joint_count_)
        throw std::domain_error("geometry parent outside scalar tree");
      geometry.shape.validate_impl();
    }
  }

  std::vector<double> lower_bounds(const QuadraticIntervalMotion &motion,
                                   double lo, double hi,
                                   const std::vector<std::pair<int, int>> &pairs,
                                   const std::vector<Eigen::Vector3d> &directions = {},
                                   bool support_gap_taylor = false) const {
    const ScopedIntervalEnvironment environment;
    if (pairs.empty()) throw std::domain_error("nonempty requested pair catalog required");
    if (!directions.empty() && directions.size() != pairs.size())
      throw std::domain_error("one optional separating direction per pair required");
    for (const auto &direction : directions)
      if (!direction.allFinite() || (direction.array() == 0.0).all())
        throw std::domain_error("finite nonzero separating direction required");
    std::vector<bool> needed(geometries_.size(), false);
    std::vector<int> parents;
    for (const auto &pair : pairs) {
      if (pair.first == pair.second) throw std::domain_error("distinct geometry indices required");
      for (int index : {pair.first, pair.second}) {
        if (index < 0 || static_cast<std::size_t>(index) >= geometries_.size())
          throw std::domain_error("geometry index outside owned catalog");
        if (!needed[index]) {
          needed[index] = true;
          parents.push_back(geometries_[index].parent);
        }
      }
    }
    const auto snapshot = tree_.poses(motion, lo, hi, parents);
    std::vector<IntervalTransform> poses(geometries_.size());
    for (std::size_t i = 0; i < geometries_.size(); ++i)
      if (needed[i]) poses[i] = compose(snapshot.at(geometries_[i].parent), geometries_[i].placement);
    std::vector<IntervalTransformMotion> whole_jets, anchor_jets;
    bool jets_available = false;
    if (support_gap_taylor) {
      try {
        const auto whole = tree_.motion_jets(motion, lo, hi, parents);
        const auto anchor = tree_.motion_jets(motion, lo, lo, parents);
        whole_jets.resize(geometries_.size()); anchor_jets.resize(geometries_.size());
        for (std::size_t i = 0; i < geometries_.size(); ++i) if (needed[i]) {
          const auto fixed = fixed_motion(geometries_[i].placement);
          whole_jets[i] = compose_motion(whole.at(geometries_[i].parent), fixed);
          anchor_jets[i] = compose_motion(anchor.at(geometries_[i].parent), fixed);
        }
        jets_available = true;
      } catch (const std::domain_error &) {
        // Optional derivative arithmetic may fail while finite poses remain
        // usable (for example huge rates on a very short horizon).
      } catch (const std::overflow_error &) {
      }
    }
    std::vector<double> result;
    result.reserve(pairs.size());
    for (std::size_t index = 0; index < pairs.size(); ++index) {
      const auto &pair = pairs[index];
      const auto &a = poses[pair.first], &b = poses[pair.second];
      const auto along = [&](const Eigen::Vector3d &direction, bool known_unit) {
        std::array<Interval, 3> da, db;
        Interval center, norm_squared;
        for (int row = 0; row < 3; ++row) {
          const Interval n(direction(row));
          center = center + n * (a.translation[row] - b.translation[row]);
          norm_squared = norm_squared + square(n);
          for (int col = 0; col < 3; ++col) {
            da[col] = da[col] - n * a.rotation[3 * row + col];
            db[col] = db[col] + n * b.rotation[3 * row + col];
          }
        }
        const auto gap = center - Interval(geometries_[pair.first].shape.support_upper_impl(da)) -
                         Interval(geometries_[pair.second].shape.support_upper_impl(db));
        if (known_unit) {
          double lower = gap.lower();
          if (jets_available) {
            try {
              lower = std::max(lower, support_gap_lower(geometries_[pair.first].shape,
                  geometries_[pair.second].shape, whole_jets[pair.first], whole_jets[pair.second],
                  anchor_jets[pair.first], anchor_jets[pair.second], direction, lo, hi));
            } catch (const std::domain_error &) {
              // Nonsmooth/degenerate support domains retain the original bound.
            } catch (const std::overflow_error &) {
            }
          }
          return lower;
        }
        const Interval norm = sqrt(norm_squared);
        if (norm.lower() <= 0.0)
          throw std::domain_error("separating direction norm is not provably positive");
        // Division encloses both signs of gap; a negative bound must use the
        // lower norm endpoint, whereas a positive bound uses the upper one.
        return (gap / norm).lower();
      };
      double best = -std::numeric_limits<double>::infinity();
      for (int axis = 0; axis < 3; ++axis)
        for (double sign : {-1.0, 1.0})
          best = std::max(best, along(sign * Eigen::Vector3d::Unit(axis), true));
      if (!directions.empty()) best = std::max(best, along(directions[index], false));
      if (!std::isfinite(best)) throw std::domain_error("nonfinite separation enclosure");
      result.push_back(best);
    }
    return result;
  }

private:
  static double support_gap_lower(const IntervalPrimitive &shape_a,
                                  const IntervalPrimitive &shape_b,
                                  const IntervalTransformMotion &whole_a,
                                  const IntervalTransformMotion &whole_b,
                                  const IntervalTransformMotion &anchor_a,
                                  const IntervalTransformMotion &anchor_b,
                                  const Eigen::Vector3d &axis, double lo, double hi) {
    using namespace interval_projection_internal;
    const auto project = [&](const IntervalTransformMotion &a, const IntervalTransformMotion &b) {
      ScalarMotion center;
      DirectionMotion da, db;
      for (int order = 0; order < 3; ++order)
        for (int row = 0; row < 3; ++row) {
          const Interval n(axis(row));
          center[order] = center[order] + n * (a[order].translation[row] - b[order].translation[row]);
          for (int col = 0; col < 3; ++col) {
            da[col][order] = da[col][order] - n * a[order].rotation[3 * row + col];
            db[col][order] = db[col][order] + n * b[order].rotation[3 * row + col];
          }
        }
      return std::make_tuple(center, da, db);
    };
    const auto [whole_center, whole_da, whole_db] = project(whole_a, whole_b);
    const auto [anchor_center, anchor_da, anchor_db] = project(anchor_a, anchor_b);
    const auto whole_sa = support_motion(shape_a, whole_da, whole_da);
    const auto whole_sb = support_motion(shape_b, whole_db, whole_db);
    const auto anchor_sa = support_motion(shape_a, anchor_da, whole_da);
    const auto anchor_sb = support_motion(shape_b, anchor_db, whole_db);
    ScalarMotion anchor_gap, whole_gap;
    for (int order = 0; order < 3; ++order) {
      anchor_gap[order] = anchor_center[order] - anchor_sa[order] - anchor_sb[order];
      whole_gap[order] = whole_center[order] - whole_sa[order] - whole_sb[order];
    }
    const double lower = quadratic_lower(anchor_gap[0], anchor_gap[1], whole_gap[2], Interval(hi)-Interval(lo));
    if (!std::isfinite(lower)) throw std::domain_error("nonfinite support gap Taylor bound");
    return lower;
  }
  std::size_t joint_count_;
  IntervalScalarKinematics tree_;
  std::vector<IntervalAttachedPrimitive> geometries_;
};

} // namespace embodik::detail
