#pragma once

#include "interval_primitive_projection.hpp"
#include "interval_scalar_model.hpp"

#include <pinocchio/multibody/geometry.hpp>
#include <pinocchio/multibody/model.hpp>
#ifdef PINOCCHIO_WITH_HPP_FCL
#include <coal/shape/geometric_shapes.h>
#endif
#include <map>
#include <set>
#include <string>

namespace embodik::detail {

// Owned constants only. No borrowed model/cache addresses, floor ownership or
// safety-certificate reuse is implied by this snapshot.
struct IntervalCollisionCatalog {
  std::vector<IntervalScalarJoint> joints;
  std::vector<IntervalAttachedPrimitive> geometries;
  std::vector<std::string> joint_names;
  std::vector<std::string> geometry_names;
  std::vector<std::pair<int, int>> pairs;
};

inline IntervalCollisionCatalog make_interval_collision_catalog(
    const pinocchio::Model &model, const pinocchio::GeometryModel &geometry,
    const std::vector<std::pair<std::string, std::string>> &selected_pairs) {
  const ScopedIntervalEnvironment environment;
#ifndef PINOCCHIO_WITH_HPP_FCL
  (void)model;
  (void)geometry;
  (void)selected_pairs;
  throw std::domain_error("native interval collision catalog requires collision support");
#else
  if (model.nq <= 0 || model.nq != model.nv || static_cast<long long>(model.njoints) != static_cast<long long>(model.nq) + 1 ||
      selected_pairs.empty() || model.joints.size() != static_cast<std::size_t>(model.njoints) ||
      model.parents.size() != static_cast<std::size_t>(model.njoints) ||
      model.names.size() != static_cast<std::size_t>(model.njoints) ||
      model.jointPlacements.size() != static_cast<std::size_t>(model.njoints) ||
      geometry.geometryObjects.size() > static_cast<std::size_t>(std::numeric_limits<int>::max()))
    throw std::domain_error("additive ordered scalar model and explicit pair catalog required");
  IntervalCollisionCatalog result;
  auto scalar = make_interval_scalar_model_catalog(model);
  result.joints = std::move(scalar.joints);
  result.joint_names = std::move(scalar.joint_names);
  std::map<std::string, int> indices;
  for (const auto &source : geometry.geometryObjects) {
    if (source.name.empty() || !indices.emplace(source.name, static_cast<int>(result.geometries.size())).second)
      throw std::domain_error("native collision geometry names must be unique and nonempty");
    if (source.parentJoint >= static_cast<pinocchio::JointIndex>(model.njoints))
      throw std::domain_error("native collision attachment outside scalar tree");
    IntervalAttachedPrimitive attached;
    attached.parent = static_cast<int>(source.parentJoint);
    attached.placement = IntervalTransform::fixed(source.placement.rotation(), source.placement.translation());
    if (const auto *box = dynamic_cast<const coal::Box *>(source.geometry.get())) {
      attached.shape = {IntervalPrimitiveKind::kBox, {box->halfSide.x(), box->halfSide.y(), box->halfSide.z()}, box->getSweptSphereRadius()};
    } else if (const auto *sphere = dynamic_cast<const coal::Sphere *>(source.geometry.get())) {
      attached.shape = {IntervalPrimitiveKind::kSphere, {sphere->radius}, sphere->getSweptSphereRadius()};
    } else if (const auto *cylinder = dynamic_cast<const coal::Cylinder *>(source.geometry.get())) {
      attached.shape = {IntervalPrimitiveKind::kCylinder, {cylinder->radius, cylinder->halfLength}, cylinder->getSweptSphereRadius()};
    } else throw std::domain_error("native interval collision catalog supports boxes, spheres and cylinders only");
    attached.shape.validate();
    result.geometries.push_back(attached);
    result.geometry_names.push_back(source.name);
  }
  std::set<std::pair<int, int>> available;
  for (const auto &pair : geometry.collisionPairs) {
    if (pair.first >= result.geometries.size() || pair.second >= result.geometries.size() || pair.first == pair.second)
      throw std::domain_error("invalid native collision pair indices");
    if (!available.emplace(std::min(pair.first, pair.second), std::max(pair.first, pair.second)).second)
      throw std::domain_error("native collision catalog contains duplicate unordered pairs");
  }
  std::set<std::pair<int, int>> selected;
  for (const auto &pair : selected_pairs) {
    const auto a = indices.find(pair.first), b = indices.find(pair.second);
    if (a == indices.end() || b == indices.end() || a == b)
      throw std::domain_error("requested pair must name distinct native geometries");
    const auto key = std::make_pair(std::min(a->second, b->second), std::max(a->second, b->second));
    if (!available.count(key) || !selected.insert(key).second)
      throw std::domain_error("requested collision pair absent or duplicated");
    result.pairs.emplace_back(a->second, b->second);
  }
  return result;
#endif
}

} // namespace embodik::detail
