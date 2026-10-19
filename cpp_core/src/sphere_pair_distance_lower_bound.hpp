#pragma once

#include <embodik/sphere_broadphase.hpp>

namespace embodik::detail {

// The enclosing spheres give a conservative signed separation, including
// overlap. Keep the arithmetic shared with the full box/SAT bound.
inline double sphere_pair_distance_lower_bound(
    const BoundingSphere &a, const BoundingSphere &b,
    const Eigen::Vector3d &translation_a, const Eigen::Matrix3d &rotation_a,
    const Eigen::Vector3d &translation_b, const Eigen::Matrix3d &rotation_b) {
  const Eigen::Vector3d center_a = translation_a + rotation_a * a.center;
  const Eigen::Vector3d center_b = translation_b + rotation_b * b.center;
  return (center_a - center_b).norm() - a.radius - b.radius;
}

} // namespace embodik::detail
