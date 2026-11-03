#pragma once

#include "interval_collision_catalog.hpp"
#include "interval_com_containment.hpp"
#include "interval_protected_pose_containment.hpp"
#include "interval_scalar_state_containment.hpp"
#include "interval_proof.hpp"

#include <pinocchio/algorithm/geometry.hpp>
#include <pinocchio/collision/distance.hpp>
#include <pinocchio/algorithm/kinematics.hpp>
#include <optional>

namespace embodik::detail {

// Source-private bounded path proof. Query workspaces are isolated, and query
// distances only propose directions. Original positive-clearance floors are
// fixed for the invocation. Optional CoM containment uses the same interval
// traversal and a fresh snapshot of this model. Optional declared CoM derivative
// ceilings require a world/root-fixed frame; they do not certify braking.
// This does not certify an executable stopping chain,
// native model provenance across calls, external pose-metric floating execution,
// CoM braking/recovery or predicted support, undeclared physical rows or a robot
// command. Optional protected poses certify their declared mathematical
// selected mathematical orientation metric only (Markley TRACE by default,
// or the explicitly requested normalized Pin log3 norm).
// Optional scalar/affine-position policy adds only its explicit bounds on this
// exact quadratic. Captured rounded q_next/dq_next and their connections are
// outside this result, even when every declared family is proved.
inline IntervalProofResult prove_collision_motion(
    const pinocchio::Model &model, const pinocchio::GeometryModel &geometry,
    const std::vector<std::pair<std::string, std::string>> &selected_pairs,
    Eigen::VectorXd q, Eigen::VectorXd dq, Eigen::VectorXd ddq, double dt,
    const std::vector<double> &original_floors, double tolerance = 1e-9,
    IntervalProofBudget budget = {},
    const IntervalComContainmentPolicy *com_containment = nullptr,
    const IntervalProtectedPosePolicy *protected_poses = nullptr,
    const IntervalScalarStatePolicy *scalar_state = nullptr,
    bool support_gap_taylor = false) {
  const ScopedIntervalEnvironment environment;
  IntervalProofResult invalid;
  invalid.status = IntervalProofStatus::kInvalidInput;
  if (original_floors.size() != selected_pairs.size() || selected_pairs.empty() ||
      !std::isfinite(tolerance) || tolerance < 0.0 || q.size() != model.nq) {
    invalid.message = "matching explicit collision floors/state and finite nonnegative tolerance required";
    return invalid;
  }
#ifndef PINOCCHIO_WITH_HPP_FCL
  (void)model; (void)geometry; (void)q; (void)dq; (void)ddq; (void)dt; (void)budget; (void)com_containment; (void)protected_poses; (void)scalar_state; (void)support_gap_taylor;
  invalid.message = "native collision motion proof requires collision support";
  return invalid;
#else
  try {
    const QuadraticIntervalMotion motion(q, dq, ddq, dt);
    auto catalog = make_interval_collision_catalog(model, geometry, selected_pairs);
    std::optional<IntervalComContainment> com;
    std::optional<IntervalProtectedPoseContainment> poses;
    std::optional<IntervalScalarStateContainment> scalars;
    std::size_t inequality_count = selected_pairs.size();
    if (com_containment) {
      com.emplace(model, *com_containment);
      if (com->inequality_count() > std::numeric_limits<std::size_t>::max() - inequality_count) {
        invalid.message = "combined collision and CoM inequality count overflowed";
        return invalid;
      }
      inequality_count += com->inequality_count();
    }
    if (protected_poses) {
      poses.emplace(model, *protected_poses);
      if (poses->inequality_count() > std::numeric_limits<std::size_t>::max() - inequality_count) {
        invalid.message = "combined protected pose inequality count overflowed";
        return invalid;
      }
      inequality_count += poses->inequality_count();
    }
    if (scalar_state) {
      scalars.emplace(model, *scalar_state);
      if (scalars->inequality_count() > std::numeric_limits<std::size_t>::max() - inequality_count) {
        invalid.message = "combined scalar state inequality count overflowed";
        return invalid;
      }
      inequality_count += scalars->inequality_count();
    }
    std::vector<Interval> required;
    required.reserve(original_floors.size());
    for (double floor : original_floors) {
      if (!std::isfinite(floor) || floor <= tolerance) {
        invalid.message = "positive-clearance original floors must exceed the comparison tolerance";
        return invalid;
      }
      // One outward threshold, fixed throughout every subdivision.
      required.emplace_back((Interval(floor) - Interval(tolerance)).upper());
    }
    const IntervalPrimitiveProjection projection(std::move(catalog.joints), std::move(catalog.geometries));
    std::map<std::pair<int, int>, std::size_t> pair_indices;
    for (std::size_t i = 0; i < geometry.collisionPairs.size(); ++i) {
      const auto &pair = geometry.collisionPairs[i];
      pair_indices.emplace(std::make_pair(std::min(pair.first, pair.second), std::max(pair.first, pair.second)), i);
    }
    std::vector<std::size_t> selected_indices;
    for (const auto &pair : catalog.pairs)
      selected_indices.push_back(pair_indices.at(std::make_pair(std::min(pair.first, pair.second), std::max(pair.first, pair.second))));
    pinocchio::Data data(model);
    pinocchio::GeometryData geometry_data(geometry);
    for (auto index : selected_indices) {
      geometry_data.distanceRequests[index].enable_nearest_points = true;
      geometry_data.distanceRequests[index].enable_signed_distance = true;
    }
    auto result = prove_interval_inequalities(dt, inequality_count, [&](double lo, double hi) {
      const double midpoint = lo + 0.5 * (hi - lo);
      const Eigen::VectorXd query_q = q + midpoint * dq + (0.5 * midpoint * midpoint) * ddq;
      if (!query_q.allFinite()) throw std::domain_error("hint configuration arithmetic overflowed");
      pinocchio::forwardKinematics(model, data, query_q);
      pinocchio::updateGeometryPlacements(model, data, geometry, geometry_data);
      std::vector<Eigen::Vector3d> directions;
      directions.reserve(selected_indices.size());
      for (std::size_t i = 0; i < selected_indices.size(); ++i) {
        const auto index = selected_indices[i];
        pinocchio::computeDistance(geometry, geometry_data, index);
        const auto &distance = geometry_data.distanceResults[index];
        Eigen::Vector3d direction = (distance.nearest_points[0] - distance.nearest_points[1]).cast<double>();
        // Native pair orientation may differ from the requested catalog order.
        if (static_cast<int>(geometry.collisionPairs[index].first) != catalog.pairs[i].first)
          direction = -direction;
        const double scale = direction.cwiseAbs().maxCoeff();
        if (!direction.allFinite() || scale == 0.0) direction = Eigen::Vector3d::UnitX();
        else direction /= scale;
        directions.push_back(direction);
      }
      auto residuals = projection.lower_bounds(motion, lo, hi, catalog.pairs, directions, support_gap_taylor);
      for (std::size_t i = 0; i < residuals.size(); ++i)
        residuals[i] = (Interval(residuals[i]) - required[i]).lower();
      if (com) {
        const auto com_residuals = com->lower_bounds(motion, lo, hi);
        residuals.insert(residuals.end(), com_residuals.begin(), com_residuals.end());
      }
      if (poses) {
        const auto pose_residuals = poses->lower_bounds(motion, lo, hi);
        residuals.insert(residuals.end(), pose_residuals.begin(), pose_residuals.end());
      }
      if (scalars) {
        const auto scalar_residuals = scalars->lower_bounds(motion, lo, hi);
        residuals.insert(residuals.end(), scalar_residuals.begin(), scalar_residuals.end());
      }
      return residuals;
    }, budget);
    if (scalar_state && result.proved())
      result.message = "all declared exact-quadratic inequalities proved; rounded native endpoints/connections not certified";
    return result;
  } catch (const std::domain_error &error) {
    invalid.message = std::string("native collision path preparation failed: ") + error.what();
    return invalid;
  } catch (const std::exception &error) {
    invalid.status = IntervalProofStatus::kUnproved;
    invalid.message = std::string("native collision path evaluation unproved: ") + error.what();
    return invalid;
  }
#endif
}

} // namespace embodik::detail
