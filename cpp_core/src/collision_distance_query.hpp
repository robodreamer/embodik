#pragma once

#include "collision_distance_evidence.hpp"

#ifdef PINOCCHIO_WITH_HPP_FCL
#include <pinocchio/collision/distance.hpp>
#include <coal/shape/geometric_shapes.h>
#include <algorithm>
#include <optional>
#include <stdexcept>

namespace embodik::detail {

// Use initialized local boxes without mutating shared geometry. Infinite or
// unavailable bounds cannot provide this auxiliary evidence check. Swept
// shape radii enlarge the box, matching the effective queried geometry.
inline std::optional<double> collision_aabb_separation_lower_bound(
    const pinocchio::GeometryObject &object_a, const pinocchio::SE3 &pose_a,
    const pinocchio::GeometryObject &object_b, const pinocchio::SE3 &pose_b) {
  const auto world_box = [](const pinocchio::GeometryObject &object,
                            const pinocchio::SE3 &pose)
      -> std::optional<std::pair<Eigen::Vector3d, Eigen::Vector3d>> {
    const auto &shape = object.geometry;
    if (!shape || !std::isfinite(shape->aabb_radius) || shape->aabb_radius < 0.0 ||
        shape->getNodeType() == coal::GEOM_PLANE ||
        shape->getNodeType() == coal::GEOM_HALFSPACE ||
        !shape->aabb_local.min_.allFinite() ||
        !shape->aabb_local.max_.allFinite()) return std::nullopt;
    const auto *primitive = dynamic_cast<const coal::ShapeBase *>(shape.get());
    const double swept_radius = primitive ? primitive->getSweptSphereRadius() : 0.0;
    if (!std::isfinite(swept_radius) || swept_radius < 0.0) return std::nullopt;
    const Eigen::Vector3d local_half =
        0.5 * (shape->aabb_local.max_ - shape->aabb_local.min_);
    const Eigen::Vector3d local_center =
        0.5 * shape->aabb_local.min_ + 0.5 * shape->aabb_local.max_;
    if (!local_half.allFinite() || (local_half.array() < 0.0).any())
      return std::nullopt;
    const Eigen::Vector3d center = pose.act(local_center);
    const Eigen::Vector3d half = pose.rotation().cwiseAbs() *
        (local_half + Eigen::Vector3d::Constant(swept_radius));
    if (!center.allFinite() || !half.allFinite()) return std::nullopt;
    return std::make_pair(center, half);
  };
  const auto a = world_box(object_a, pose_a), b = world_box(object_b, pose_b);
  if (!a || !b) return std::nullopt;
  const Eigen::Vector3d gap =
      ((a->first - b->first).cwiseAbs() - a->second - b->second).cwiseMax(0.0);
  const double scale = std::max({1.0, a->first.cwiseAbs().maxCoeff(),
      b->first.cwiseAbs().maxCoeff(), a->second.maxCoeff(), b->second.maxCoeff(),
      pose_a.translation().cwiseAbs().maxCoeff(),
      pose_b.translation().cwiseAbs().maxCoeff()});
  const double lower = gap.norm() -
      128.0 * std::numeric_limits<double>::epsilon() * scale;
  if (!std::isfinite(lower)) return std::nullopt;
  return std::max(0.0, lower);
}

// Strict original floors need the same numerical query in row construction,
// refinement cuts, candidate checks and final publication. A coarse row/cut
// distance can otherwise admit a step that every precise final query rejects.
inline void compute_policy_collision_distance(
    const pinocchio::GeometryModel &model, pinocchio::GeometryData &data,
    std::size_t index, bool strict) {
  const auto &pair = model.collisionPairs[index];
  auto request = data.distanceRequests[index];
  auto &result = data.distanceResults[index];
  const auto lower_bound = collision_aabb_separation_lower_bound(
      model.geometryObjects[pair.first], data.oMg[pair.first],
      model.geometryObjects[pair.second], data.oMg[pair.second]);
  const auto valid_evidence = [&]() {
    return lower_bound
        ? detail::collision_distance_evidence_is_valid(result.min_distance, *lower_bound)
        : detail::collision_distance_evidence_is_valid(result.min_distance);
  };
  const auto cold_query = [&](const hpp::fcl::DistanceRequest &cold_request) {
    pinocchio::GeometryData::ComputeDistance cold_distance(
        model.geometryObjects[pair.first], model.geometryObjects[pair.second]);
    result.clear();
    cold_distance(pinocchio::toFclTransform3f(data.oMg[pair.first]),
                  pinocchio::toFclTransform3f(data.oMg[pair.second]),
                  cold_request, result);
  };
  if (strict) {
    request.enable_cached_gjk_guess = false;
    request.gjk_initial_guess = hpp::fcl::GJKInitialGuess::DefaultGuess;
    request.gjk_tolerance = std::min(request.gjk_tolerance, 1e-12);
    request.gjk_max_iterations = std::max<std::size_t>(request.gjk_max_iterations, 512);
    cold_query(request);
  } else {
    pinocchio::computeDistance(model, data, index);
  }
  if (valid_evidence()) return;

  // A degenerate initial simplex can fail or return an impossible finite
  // distance for separated shapes. Retry exactly
  // once from their local AABB centers, preserving all precision settings and
  // avoiding history from previous configurations. This remains a full signed
  // distance query, not a bounding-volume distance substituted for its result.
  request.enable_cached_gjk_guess = false;
  request.gjk_initial_guess = hpp::fcl::GJKInitialGuess::BoundingVolumeGuess;
  cold_query(request);
  if (valid_evidence()) return;
  result.clear();
  throw std::runtime_error(
      "Collision distance query returned invalid evidence for " +
      model.geometryObjects[pair.first].name + " / " +
      model.geometryObjects[pair.second].name);
}

} // namespace embodik::detail
#endif
