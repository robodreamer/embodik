#include "embodik/sphere_broadphase.hpp"
#include <coal/BVH/BVH_model.h>
#include <coal/shape/geometric_shapes.h>
#include <algorithm>
#include <cmath>
#include <limits>

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
      Eigen::Vector3d sphere_center(aabb.center()[0], aabb.center()[1],
                                    aabb.center()[2]);
      Eigen::Vector3d half_extents(aabb.width() / 2.0, aabb.height() / 2.0,
                                   aabb.depth() / 2.0);
      bs.box_center = geom_obj.placement.act(sphere_center);
      bs.box_rotation = geom_obj.placement.rotation();
      const auto *shape_base = dynamic_cast<const coal::ShapeBase *>(shape.get());
      const double swept_radius =
          shape_base != nullptr ? shape_base->getSweptSphereRadius() : 0.0;
      double sphere_radius = half_extents.norm() + swept_radius;
      bs.box_half_extents =
          half_extents + Eigen::Vector3d::Constant(swept_radius);

      if (const auto *sphere = dynamic_cast<const coal::Sphere *>(shape.get())) {
        sphere_center.setZero();
        sphere_radius = sphere->radius + swept_radius;
      } else if (const auto *capsule =
                     dynamic_cast<const coal::Capsule *>(shape.get())) {
        sphere_center.setZero();
        sphere_radius = capsule->halfLength + capsule->radius + swept_radius;
      } else if (const auto *ellipsoid =
                     dynamic_cast<const coal::Ellipsoid *>(shape.get())) {
        sphere_center.setZero();
        sphere_radius = ellipsoid->radii.maxCoeff() + swept_radius;
      } else if (const auto *box =
                     dynamic_cast<const coal::Box *>(shape.get())) {
        sphere_center.setZero();
        sphere_radius = box->halfSide.norm() + swept_radius;
      } else if (const auto *cylinder =
                     dynamic_cast<const coal::Cylinder *>(shape.get())) {
        sphere_center.setZero();
        sphere_radius =
            std::hypot(cylinder->radius, cylinder->halfLength) + swept_radius;
      } else if (const auto *cone =
                     dynamic_cast<const coal::Cone *>(shape.get())) {
        sphere_center.setZero();
        sphere_radius =
            std::hypot(cone->radius, cone->halfLength) + swept_radius;
      } else if (const auto *mesh =
                     dynamic_cast<const coal::BVHModelBase *>(shape.get());
                 mesh != nullptr && mesh->vertices != nullptr &&
          !mesh->vertices->empty()) {
        Eigen::Vector3d vertex_centroid = Eigen::Vector3d::Zero();
        for (const auto &vertex : *mesh->vertices) {
          vertex_centroid +=
              Eigen::Vector3d(vertex[0], vertex[1], vertex[2]);
        }
        vertex_centroid /= static_cast<double>(mesh->vertices->size());

        const auto enclosing_radius = [&](const Eigen::Vector3d &center) {
          double radius = 0.0;
          for (const auto &vertex : *mesh->vertices) {
            const Eigen::Vector3d point(vertex[0], vertex[1], vertex[2]);
            radius = std::max(radius, (point - center).norm());
          }
          return std::nextafter(radius,
                                std::numeric_limits<double>::infinity());
        };
        const double aabb_center_radius = enclosing_radius(sphere_center);
        const double centroid_radius = enclosing_radius(vertex_centroid);
        if (centroid_radius < aabb_center_radius) {
          sphere_center = vertex_centroid;
          sphere_radius = centroid_radius + swept_radius;
        } else {
          sphere_radius = aabb_center_radius + swept_radius;
        }
      }

      bs.center = geom_obj.placement.act(sphere_center);
      bs.radius = sphere_radius;
      bs.motion_radius = std::nextafter(
          bs.center.norm() + bs.radius,
          std::numeric_limits<double>::infinity());
    } else {
      bs.center = geom_obj.placement.translation();
      bs.radius = 0.0;
      bs.box_center = geom_obj.placement.translation();
      bs.box_rotation = geom_obj.placement.rotation();
      bs.box_half_extents.setZero();
      bs.motion_radius = std::nextafter(
          bs.center.norm(), std::numeric_limits<double>::infinity());
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
  const double sphere_lower_bound =
      (center_a - center_b).norm() - sa.radius - sb.radius;

  const Eigen::Vector3d box_center_a =
      translation_a + rotation_a * sa.box_center;
  const Eigen::Vector3d box_center_b =
      translation_b + rotation_b * sb.box_center;
  const Eigen::Matrix3d box_rotation_a = rotation_a * sa.box_rotation;
  const Eigen::Matrix3d box_rotation_b = rotation_b * sb.box_rotation;
  const Eigen::Vector3d world_half_extents_a =
      box_rotation_a.cwiseAbs() * sa.box_half_extents;
  const Eigen::Vector3d world_half_extents_b =
      box_rotation_b.cwiseAbs() * sb.box_half_extents;
  const Eigen::Vector3d axis_separations =
      (box_center_a - box_center_b).cwiseAbs() - world_half_extents_a -
      world_half_extents_b;
  const Eigen::Vector3d positive_separations =
      axis_separations.cwiseMax(Eigen::Vector3d::Zero());
  double box_lower_bound = positive_separations.norm();

  const Eigen::Vector3d center_delta = box_center_a - box_center_b;
  const auto projected_radius = [](const Eigen::Matrix3d &rotation,
                                   const Eigen::Vector3d &half_extents,
                                   const Eigen::Vector3d &axis) {
    double radius = 0.0;
    for (int column = 0; column < 3; ++column) {
      radius += half_extents[column] *
                std::abs(rotation.col(column).dot(axis));
    }
    return radius;
  };
  const auto accumulate_separating_axis =
      [&](const Eigen::Vector3d &candidate_axis) {
        const double squared_norm = candidate_axis.squaredNorm();
        if (squared_norm <= std::numeric_limits<double>::epsilon()) {
          return;
        }
        const Eigen::Vector3d axis = candidate_axis / std::sqrt(squared_norm);
        const double projected_gap =
            std::abs(center_delta.dot(axis)) -
            projected_radius(box_rotation_a, sa.box_half_extents, axis) -
            projected_radius(box_rotation_b, sb.box_half_extents, axis);
        box_lower_bound = std::max(box_lower_bound, projected_gap);
      };

  for (int axis = 0; axis < 3; ++axis) {
    accumulate_separating_axis(box_rotation_a.col(axis));
    accumulate_separating_axis(box_rotation_b.col(axis));
  }
  for (int axis_a = 0; axis_a < 3; ++axis_a) {
    for (int axis_b = 0; axis_b < 3; ++axis_b) {
      accumulate_separating_axis(
          box_rotation_a.col(axis_a).cross(box_rotation_b.col(axis_b)));
    }
  }

  if (box_lower_bound <= 0.0) {
    return sphere_lower_bound;
  }
  return std::max(sphere_lower_bound, box_lower_bound);
}

}  // namespace embodik
