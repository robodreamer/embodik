#include "embodik/sphere_broadphase.hpp"
#include <cmath>

namespace embodik {

#ifdef PINOCCHIO_WITH_HPP_FCL

void SphereBroadphase::build(const pinocchio::GeometryModel &geom_model) {
  spheres_.clear();
  spheres_.reserve(geom_model.geometryObjects.size());

  for (const auto &geom_obj : geom_model.geometryObjects) {
    BoundingSphere bs;
    const auto &shape = geom_obj.geometry;
    if (shape) {
      shape->computeLocalAABB();
      const auto &aabb = shape->aabb_local;
      Eigen::Vector3d aabb_center(aabb.center()[0], aabb.center()[1], aabb.center()[2]);
      bs.center = geom_obj.placement.act(aabb_center);
      Eigen::Vector3d half_extents(aabb.width() / 2.0, aabb.height() / 2.0, aabb.depth() / 2.0);
      bs.radius = half_extents.norm();
    } else {
      bs.center = geom_obj.placement.translation();
      bs.radius = 0.0;
    }
    spheres_.push_back(bs);
  }
}

#endif

double SphereBroadphase::compute_pair_lower_bound(
    std::size_t geom_idx_a, std::size_t geom_idx_b,
    const Eigen::Vector3d &translation_a, const Eigen::Matrix3d &rotation_a,
    const Eigen::Vector3d &translation_b, const Eigen::Matrix3d &rotation_b) const {
  const auto &sa = spheres_[geom_idx_a];
  const auto &sb = spheres_[geom_idx_b];
  const Eigen::Vector3d center_a = translation_a + rotation_a * sa.center;
  const Eigen::Vector3d center_b = translation_b + rotation_b * sb.center;
  return (center_a - center_b).norm() - sa.radius - sb.radius;
}

}  // namespace embodik
