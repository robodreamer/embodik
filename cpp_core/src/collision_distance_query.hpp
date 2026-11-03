#pragma once

#include "collision_distance_evidence.hpp"

#ifdef PINOCCHIO_WITH_HPP_FCL
#include <pinocchio/collision/distance.hpp>
#include <algorithm>
#include <stdexcept>

namespace embodik::detail {

// Strict original floors need the same numerical query in row construction,
// refinement cuts, candidate checks and final publication. A coarse row/cut
// distance can otherwise admit a step that every precise final query rejects.
inline void compute_policy_collision_distance(
    const pinocchio::GeometryModel &model, pinocchio::GeometryData &data,
    std::size_t index, bool strict) {
  const auto &pair = model.collisionPairs[index];
  auto request = data.distanceRequests[index];
  auto &result = data.distanceResults[index];
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
  if (detail::collision_distance_evidence_is_valid(result.min_distance)) return;

  // A degenerate initial simplex can fail for separated shapes. Retry exactly
  // once from their local AABB centers, preserving all precision settings and
  // avoiding history from previous configurations. This remains a full signed
  // distance query, not a bounding-volume distance substituted for its result.
  request.enable_cached_gjk_guess = false;
  request.gjk_initial_guess = hpp::fcl::GJKInitialGuess::BoundingVolumeGuess;
  cold_query(request);
  if (detail::collision_distance_evidence_is_valid(result.min_distance)) return;
  result.clear();
  throw std::runtime_error(
      "Collision distance query returned invalid evidence for " +
      model.geometryObjects[pair.first].name + " / " +
      model.geometryObjects[pair.second].name);
}

} // namespace embodik::detail
#endif
