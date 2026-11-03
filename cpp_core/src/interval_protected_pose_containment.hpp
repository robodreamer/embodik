#pragma once

#include "interval_scalar_model.hpp"

#include <optional>

namespace embodik::detail {

// Fixed world targets and literal physical radii. Orientation selects one of
// the named mathematical metrics below, never both. Neither metric certifies
// external matrix conversion, binary FK or floating library execution.
struct IntervalProtectedFramePosePolicy {
  std::string frame_name;
  Eigen::Vector3d target_position = Eigen::Vector3d::Zero();
  Eigen::Matrix3d target_rotation = Eigen::Matrix3d::Identity();
  std::optional<double> position_radius;
  std::optional<double> markley_trace_orientation_radius;
  double position_comparison_tolerance = 0.0;
  double orientation_comparison_tolerance = 0.0;
  std::optional<double> pin_log_orientation_radius;
};

struct IntervalProtectedPosePolicy {
  std::vector<IntervalProtectedFramePosePolicy> frames;
};

namespace interval_protected_pose_internal {

inline Interval frobenius_norm(const std::array<Interval, 9> &matrix) {
  Interval squared;
  for (const auto &element : matrix) squared = squared + square(element);
  return sqrt(squared);
}

inline Interval determinant(const std::array<Interval, 9> &a) {
  return a[0] * (a[4] * a[8] - a[5] * a[7]) -
         a[1] * (a[3] * a[8] - a[5] * a[6]) +
         a[2] * (a[3] * a[7] - a[4] * a[6]);
}

// ||M-polar(M)||F <= epsilon/(1+sqrt(1-epsilon)), where
// epsilon >= ||M^T M-I||F < 1 and det(M)>0. No SVD or projection is performed.
inline Interval rigidity_error(const std::array<Interval, 9> &matrix) {
  if (determinant(matrix).lower() <= 0.0)
    throw std::domain_error("orientation requires a provably positive determinant");
  std::array<Interval, 9> gram;
  for (int row = 0; row < 3; ++row) {
    for (int col = 0; col < 3; ++col) {
      for (int k = 0; k < 3; ++k)
        gram[3 * row + col] = gram[3 * row + col] +
            matrix[3 * k + row] * matrix[3 * k + col];
      if (row == col) gram[3 * row + col] = gram[3 * row + col] - Interval(1);
    }
  }
  const Interval epsilon(frobenius_norm(gram).upper());
  if (epsilon.upper() >= 1.0)
    throw std::domain_error("orientation requires a certified nearby proper rotation");
  return Interval((epsilon / (Interval(1) + sqrt(Interval(1) - epsilon))).upper());
}

inline Interval composed_rigidity_error(const Interval &a, const Interval &b) {
  return Interval((a + b + a * b).upper());
}

// For E, the unnormalized TRACE quaternion is (s,w), with
// s=(E21-E12,E02-E20,E10-E01), w=1+trace(E). If w>0 its mathematical
// angle is 2 atan(||s||/w), even for nonorthogonal E. The upper bound below
// uses atan(z)<=z. Normalization cancels only in this mathematical contract.
inline double markley_trace_angle_upper(const std::array<Interval, 9> &relative,
                                       const Interval &rigidity_defect) {
  // Structural rigidity avoids dependency inflation from interval E E^T.
  // The threshold is a mathematical near-rigidity capability condition,
  // not assurance of a rounded external allclose/SVD decision.
  if ((rigidity_defect * (Interval(2) + rigidity_defect)).upper() > 1e-12)
    throw std::domain_error("Markley TRACE near-rigidity capability not established");
  auto displacement = relative;
  for (int axis = 0; axis < 3; ++axis)
    displacement[4 * axis] = displacement[4 * axis] - Interval(1);
  if (frobenius_norm(displacement).upper() >= 0.1)
    throw std::domain_error("Markley TRACE near-identity capability not established");
  const Interval trace = relative[0] + relative[4] + relative[8];
  for (int axis = 0; axis < 3; ++axis)
    if (trace.lower() <= relative[4 * axis].upper())
      throw std::domain_error("Markley TRACE branch not established");
  const Interval w = Interval(1) + trace;
  if (w.lower() <= 0.0)
    throw std::domain_error("positive Markley TRACE scalar not established");
  const Interval squared = square(relative[7] - relative[5]) +
                           square(relative[2] - relative[6]) +
                           square(relative[3] - relative[1]);
  return (Interval(2) * sqrt(squared) / w).upper();
}

using Vector3 = std::array<Interval, 3>;

inline Vector3 cross(const Vector3 &a, const Vector3 &b) {
  return {a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]};
}

inline Vector3 normalized(const Vector3 &a) {
  Interval squared;
  for (const auto &x : a) squared = squared + square(x);
  const auto norm = sqrt(squared);
  if (norm.lower() <= 0.0)
    throw std::domain_error("Pin log input column norm not provably positive");
  return {a[0]/norm, a[1]/norm, a[2]/norm};
}

// Mathematical counterpart of Pinocchio log3's nominal non-pi path. Use the
// literal relative matrix supplied by native fixed-frame policy: target^T R.
// Enclose its column normalization and cross products in their original order.
// The exact renormalized column norms are <=1, hence trace<=3 structurally.
// This bounds selected formulas, not eager unused branches or binary libm.
inline double pin_log_norm_upper(const std::array<Interval, 9> &relative) {
  const Vector3 u = normalized({relative[0], relative[3], relative[6]});
  const Vector3 v = normalized({relative[1], relative[4], relative[7]});
  const Vector3 c2 = cross(u, v), c0 = cross(v, c2);
  const Interval cosine = Interval(.5) * (c0[0] + v[1] + c2[2] - Interval(1));
  if (cosine.lower() < 0.0 || cosine.lower() > 1.0)
    throw std::domain_error("Pin log nonnegative cosine domain not established");
  const Vector3 antisymmetric{
      Interval(.5)*(v[2]-c2[1]), Interval(.5)*(c2[0]-c0[2]),
      Interval(.5)*(c0[1]-v[0])};
  Interval squared;
  for (const auto &x : antisymmetric) squared = squared + square(x);
  const Interval magnitude(sqrt(squared).upper());
  const Interval epsilon(std::numeric_limits<double>::epsilon());
  const Interval large = Interval(2) /
      (Interval(1) + Interval(cosine.lower()) - Interval(.5)*square(epsilon));
  const Interval n2 = square(magnitude);
  const Interval small = Interval(1) + n2/Interval(6) +
      Interval(3)*square(n2)/Interval(40);
  return (magnitude * Interval(std::max(large.upper(), small.upper()))).upper();
}

} // namespace interval_protected_pose_internal

// Owned mathematical continuous-pose provider for additive scalar motion.
// Position is the literal world Euclidean norm. Orientation is the explicitly
// selected mathematical Markley TRACE angle or normalized Pin log3 norm. It
// does not certify external SciPy/native FK execution or complete stopping.
class IntervalProtectedPoseContainment {
public:
  IntervalProtectedPoseContainment(const pinocchio::Model &model,
                                   const IntervalProtectedPosePolicy &policy)
      : tree_(make_interval_scalar_model_catalog(model).joints) {
    const ScopedIntervalEnvironment environment;
    if (policy.frames.empty()) throw std::domain_error("nonempty protected pose policy required");
    for (const auto &source : policy.frames) {
      if (source.frame_name.empty() || !source.target_position.allFinite() ||
          !source.target_rotation.allFinite() ||
          (!source.position_radius && !source.markley_trace_orientation_radius &&
           !source.pin_log_orientation_radius))
        throw std::domain_error("finite fixed targets and an enabled protected pose block required");
      if (source.markley_trace_orientation_radius && source.pin_log_orientation_radius)
        throw std::domain_error("protected orientation metrics are mutually exclusive");
      Entry entry;
      const auto frame_id = model.getFrameId(source.frame_name);
      if (frame_id >= model.frames.size()) throw std::domain_error("protected TCP frame not found");
      const auto &frame = model.frames[frame_id];
      if (frame.parentJoint >= static_cast<pinocchio::JointIndex>(model.njoints))
        throw std::domain_error("protected TCP frame outside scalar tree");
      entry.parent = static_cast<int>(frame.parentJoint);
      entry.placement = IntervalTransform::fixed(frame.placement.rotation(), frame.placement.translation());
      entry.position = source.position_radius.has_value();
      entry.pin_log = source.pin_log_orientation_radius.has_value();
      entry.orientation = entry.pin_log || source.markley_trace_orientation_radius.has_value();
      for (int row = 0; row < 3; ++row) {
        entry.target_position[row] = Interval(source.target_position(row));
        for (int col = 0; col < 3; ++col)
          entry.target_rotation[3 * row + col] = Interval(source.target_rotation(row, col));
      }
      if (entry.position)
        add_radius(*source.position_radius, source.position_comparison_tolerance);
      else validate_tolerance(source.position_comparison_tolerance);
      if (entry.orientation) {
        add_radius(entry.pin_log ? *source.pin_log_orientation_radius :
                   *source.markley_trace_orientation_radius, source.orientation_comparison_tolerance);
        if (!entry.pin_log) {
          using namespace interval_protected_pose_internal;
          entry.rigidity_defect = composed_rigidity_error(
              rigidity_error(entry.target_rotation), rigidity_error(entry.placement.rotation));
          for (int parent = entry.parent; parent != 0; parent = static_cast<int>(model.parents[parent])) {
            const auto fixed = IntervalTransform::fixed(model.jointPlacements[parent].rotation(),
                                                         model.jointPlacements[parent].translation());
            entry.rigidity_defect = composed_rigidity_error(entry.rigidity_defect,
                                                           rigidity_error(fixed.rotation));
          }
        }
      } else validate_tolerance(source.orientation_comparison_tolerance);
      requested_.push_back(entry.parent);
      entries_.push_back(std::move(entry));
    }
  }

  std::size_t inequality_count() const { return allowed_.size(); }

  // Stable policy order, position before orientation within each entry.
  std::vector<double> upper_bounds(const QuadraticIntervalMotion &motion,
                                    double lo, double hi) const {
    const ScopedIntervalEnvironment environment;
    const auto snapshot = tree_.poses(motion, lo, hi, requested_);
    std::vector<double> result;
    result.reserve(inequality_count());
    for (const auto &entry : entries_) {
      const auto pose = compose(snapshot.at(entry.parent), entry.placement);
      if (entry.position) {
        Interval squared;
        for (int axis = 0; axis < 3; ++axis)
          squared = squared + square(pose.translation[axis] - entry.target_position[axis]);
        result.push_back(sqrt(squared).upper());
      }
      if (entry.orientation) {
        std::array<Interval, 9> relative;
        for (int row = 0; row < 3; ++row)
          for (int col = 0; col < 3; ++col)
            for (int k = 0; k < 3; ++k)
              relative[3 * row + col] = relative[3 * row + col] +
                  (entry.pin_log ? entry.target_rotation[3 * k + row] * pose.rotation[3 * k + col] :
                   entry.target_rotation[3 * row + k] * pose.rotation[3 * col + k]);
        result.push_back(entry.pin_log ? interval_protected_pose_internal::pin_log_norm_upper(relative) :
            interval_protected_pose_internal::markley_trace_angle_upper(relative, entry.rigidity_defect));
      }
    }
    return result;
  }

  std::vector<double> lower_bounds(const QuadraticIntervalMotion &motion,
                                   double lo, double hi) const {
    const ScopedIntervalEnvironment environment;
    auto result = upper_bounds(motion, lo, hi);
    for (std::size_t i = 0; i < result.size(); ++i)
      result[i] = (allowed_[i] - Interval(result[i])).lower();
    return result;
  }

private:
  static void validate_tolerance(double tolerance) {
    if (!std::isfinite(tolerance) || tolerance < 0.0)
      throw std::domain_error("finite nonnegative protected pose comparison tolerance required");
  }
  void add_radius(double radius, double tolerance) {
    validate_tolerance(tolerance);
    if (!std::isfinite(radius) || radius <= 0.0)
      throw std::domain_error("positive finite protected pose radius required");
    allowed_.push_back(Interval(radius) + Interval(tolerance));
  }
  struct Entry {
    int parent = 0;
    bool position = false, orientation = false;
    bool pin_log = false;
    IntervalTransform placement;
    std::array<Interval, 3> target_position;
    std::array<Interval, 9> target_rotation;
    Interval rigidity_defect;
  };
  IntervalScalarKinematics tree_;
  std::vector<Entry> entries_;
  std::vector<int> requested_;
  std::vector<Interval> allowed_;
};

} // namespace embodik::detail
