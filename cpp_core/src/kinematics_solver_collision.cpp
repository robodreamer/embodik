/**
 * @file kinematics_solver_collision.cpp
 * @brief Collision constraint ownership for KinematicsSolver
 */

#include "kinematics_solver_internal.hpp"

namespace embodik {

void KinematicsSolver::configure_collision_constraint(
    double min_distance,
    const std::vector<std::pair<std::string, std::string>> &include_pairs,
    const std::vector<std::pair<std::string, std::string>> &exclude_pairs,
    bool nearest_points_all_pairs,
    int max_constraints) {

#ifdef PINOCCHIO_WITH_HPP_FCL
  if (!robot_->has_collision_geometry()) {
    throw std::runtime_error(
        "Collision geometry is not available for this robot model.");
  }

  const auto *collision_model_ptr =
      static_cast<const RobotModel &>(*robot_).collision_model();
  auto *collision_data = robot_->collision_data();

  if (!collision_constraint_.has_value()) {
    collision_constraint_.emplace();
  }

  auto &config = *collision_constraint_;
  config.enabled = true;
  config.min_distance = std::max(0.0, min_distance);
  if (config.constraint_activation_multiplier > 0.0) {
    config.constraint_activation_margin =
        config.constraint_activation_multiplier * config.min_distance;
  } else {
    config.constraint_activation_margin = 0.0;
  }
  config.upper_distance = kCollisionUpperDistance;
  config.tolerance = kCollisionTolerance;
  config.nearest_points_all_pairs = nearest_points_all_pairs;
  config.max_constraints = std::max(1, max_constraints);
  config.include_pairs.clear();
  config.exclude_pairs.clear();

  for (const auto &pair : include_pairs) {
    config.include_pairs.insert(canonical_pair_key(pair.first, pair.second));
  }

  for (const auto &pair : exclude_pairs) {
    config.exclude_pairs.insert(canonical_pair_key(pair.first, pair.second));
  }

  if (collision_data != nullptr && collision_model_ptr != nullptr) {
    const std::size_t num_pairs = collision_model_ptr->collisionPairs.size();
    if (collision_data->distanceRequests.size() != num_pairs) {
      collision_data->distanceRequests.resize(num_pairs);
      collision_data->distanceResults.resize(num_pairs);
    }
    for (auto &request : collision_data->distanceRequests) {
      request.enable_nearest_points = nearest_points_all_pairs;
      request.enable_signed_distance = true;
    }
    collision_data->activateAllCollisionPairs();

    collision_allowed_pair_mask_.assign(num_pairs, 0);
    for (std::size_t idx = 0; idx < num_pairs; ++idx) {
      const auto &pair = collision_model_ptr->collisionPairs[idx];
      const auto &name_a =
          collision_model_ptr->geometryObjects[pair.first].name;
      const auto &name_b =
          collision_model_ptr->geometryObjects[pair.second].name;
      const auto key = canonical_pair_key(name_a, name_b);
      bool allowed = true;
      if (!config.include_pairs.empty() &&
          config.include_pairs.find(key) == config.include_pairs.end()) {
        allowed = false;
      }
      if (allowed &&
          config.exclude_pairs.find(key) != config.exclude_pairs.end()) {
        allowed = false;
      }
      collision_allowed_pair_mask_[idx] = allowed ? 1 : 0;
      if (!allowed) {
        collision_data->activeCollisionPairs[idx] = false;
      }
    }
    last_collision_constraint_pair_indices_.clear();
    collision_cached_candidate_pair_indices_.clear();
    collision_pair_bound_valid_.assign(num_pairs, 0);
    collision_pair_last_signed_distance_.assign(
        num_pairs, std::numeric_limits<double>::infinity());
    collision_pair_last_rel_translation_norm_.assign(num_pairs, 0.0);
    collision_pair_last_rel_rotation_.assign(
        num_pairs, pack_rotation_matrix(Eigen::Matrix3d::Identity()));
    collision_pair_cache_has_full_scan_ = false;
    collision_pair_cache_steps_since_refresh_ = 0;
    last_collision_pairs_considered_ = 0;
    last_collision_exact_distance_queries_ = 0;
    last_collision_bound_culled_pairs_ = 0;
    last_collision_sphere_culled_pairs_ = 0;
    last_collision_budget_exhausted_ = false;
    last_constraint_min_distance_ = std::numeric_limits<double>::infinity();
    last_constraint_min_recovery_margin_ =
        std::numeric_limits<double>::infinity();
    last_constraint_was_full_scan_ = false;
    last_collision_constraint_result_.reset();
    last_collision_constraint_row_dt_ =
        std::numeric_limits<double>::quiet_NaN();
    last_collision_constraint_q_ = Eigen::VectorXd();
    collision_stuck_counters_.clear();
    collision_stuck_last_distances_.clear();
    collision_pair_distance_floor_.clear();
    post_step_collision_distance_cache_.clear();
    collision_cache_frozen_indices_.clear();
    ++collision_validation_policy_revision_;
    invalidate_collision_validation_cache();
  }
  if (sphere_broadphase_enabled_ && !sphere_broadphase_.is_built()) {
    sphere_broadphase_.build(*collision_model_ptr);
  }
  if (!stall_user_configured_) {
    stall_config_.stall_threshold = 3;
    stall_config_.restore_rate = 0.2;
    stall_config_.floor_fraction = 0.0;
  }
#else
  (void)min_distance;
  (void)include_pairs;
  (void)exclude_pairs;
  (void)nearest_points_all_pairs;
  (void)max_constraints;
  throw std::runtime_error("Collision avoidance requires Pinocchio to be built "
                           "with hpp-fcl support.");
#endif
}

std::optional<KinematicsSolver::CollisionDebugInfo>
KinematicsSolver::evaluate_collision_debug(const Eigen::VectorXd &current_q) {
#ifdef PINOCCHIO_WITH_HPP_FCL
  if (!robot_->has_collision_geometry()) {
    return std::nullopt;
  }

  const ScopedRobotKinematicsRestore restore_robot_state(*robot_);

  if (current_q.size() > 0) {
    if (current_q.size() != robot_->nq()) {
      throw std::runtime_error(
          "Invalid configuration size for collision evaluation.");
    }
    if (!current_q.allFinite()) {
      throw std::runtime_error(
          "Collision evaluation configuration must be finite.");
    }
    robot_->update_kinematics(current_q);
  } else {
    robot_->update_kinematics(robot_->get_current_configuration());
  }

  const auto *collision_model =
      static_cast<const RobotModel &>(*robot_).collision_model();
  auto *collision_data = robot_->collision_data();
  if (collision_model == nullptr || collision_data == nullptr) {
    return std::nullopt;
  }

  // Fast debug evaluation path: compute the globally closest active pair
  // directly without running full constraint assembly/caching/hysteresis logic.
  pinocchio::updateGeometryPlacements(robot_->model(), robot_->data(),
                                      *collision_model, *collision_data);
  const auto &pairs = collision_model->collisionPairs;
  double best_distance = std::numeric_limits<double>::infinity();
  std::optional<std::size_t> best_index;

  for (std::size_t idx = 0; idx < pairs.size(); ++idx) {
    if (!collision_data->activeCollisionPairs.empty() &&
        !collision_data->activeCollisionPairs[idx]) {
      continue;
    }
    pinocchio::computeDistance(*collision_model, *collision_data, idx);
    const double distance = collision_data->distanceResults[idx].min_distance;
    if (!std::isfinite(distance)) {
      continue;
    }
    if (distance < best_distance) {
      best_distance = distance;
      best_index = idx;
    }
  }

  if (!best_index.has_value()) {
    return std::nullopt;
  }

  const std::size_t idx = *best_index;
  bool restore_nearest_points_flag = false;
  bool previous_nearest_points_flag = false;
  if (idx < collision_data->distanceRequests.size()) {
    auto &req = collision_data->distanceRequests[idx];
    previous_nearest_points_flag = req.enable_nearest_points;
    if (!req.enable_nearest_points) {
      req.enable_nearest_points = true;
      pinocchio::computeDistance(*collision_model, *collision_data, idx);
      restore_nearest_points_flag = true;
    }
  }

  const auto &pair = pairs[idx];
  const auto &obj_a = collision_model->geometryObjects[pair.first];
  const auto &obj_b = collision_model->geometryObjects[pair.second];
  const auto &res = collision_data->distanceResults[idx];
  CollisionDebugInfo debug;
  debug.object_a = obj_a.name;
  debug.object_b = obj_b.name;
  debug.distance = res.min_distance;
  debug.point_a_world = res.nearest_points[0].cast<double>();
  debug.point_b_world = res.nearest_points[1].cast<double>();

  if (restore_nearest_points_flag && idx < collision_data->distanceRequests.size()) {
    collision_data->distanceRequests[idx].enable_nearest_points =
        previous_nearest_points_flag;
  }

  return debug;
#else
  (void)current_q;
  return std::nullopt;
#endif
}

std::optional<double>
KinematicsSolver::evaluate_min_collision_distance(const Eigen::VectorXd &current_q) {
#ifdef PINOCCHIO_WITH_HPP_FCL
  if (!robot_->has_collision_geometry()) {
    return std::nullopt;
  }

  const ScopedRobotKinematicsRestore restore_robot_state(*robot_);

  if (current_q.size() > 0) {
    if (current_q.size() != robot_->nq()) {
      throw std::runtime_error(
          "Invalid configuration size for collision evaluation.");
    }
    if (!current_q.allFinite()) {
      throw std::runtime_error(
          "Collision evaluation configuration must be finite.");
    }
    robot_->update_kinematics(current_q);
  } else {
    robot_->update_kinematics(robot_->get_current_configuration());
  }

  const auto *collision_model =
      static_cast<const RobotModel &>(*robot_).collision_model();
  auto *collision_data = robot_->collision_data();
  if (collision_model == nullptr || collision_data == nullptr) {
    return std::nullopt;
  }

  pinocchio::updateGeometryPlacements(robot_->model(), robot_->data(),
                                      *collision_model, *collision_data);
  const auto &pairs = collision_model->collisionPairs;
  double best_distance = std::numeric_limits<double>::infinity();
  bool found = false;

  for (std::size_t idx = 0; idx < pairs.size(); ++idx) {
    if (!collision_data->activeCollisionPairs.empty() &&
        !collision_data->activeCollisionPairs[idx]) {
      continue;
    }
    pinocchio::computeDistance(*collision_model, *collision_data, idx);
    const double distance = collision_data->distanceResults[idx].min_distance;
    if (!std::isfinite(distance)) {
      continue;
    }
    best_distance = std::min(best_distance, distance);
    found = true;
  }

  if (!found) {
    return std::nullopt;
  }
  return best_distance;
#else
  (void)current_q;
  return std::nullopt;
#endif
}

std::optional<double>
KinematicsSolver::evaluate_min_collision_distance_targeted(
    const Eigen::VectorXd &current_q,
    const std::vector<std::size_t> &pair_indices) {
#ifdef PINOCCHIO_WITH_HPP_FCL
  if (!robot_->has_collision_geometry() || pair_indices.empty()) {
    return std::nullopt;
  }

  if (current_q.size() > 0) {
    if (current_q.size() != robot_->nq()) {
      throw std::runtime_error(
          "Invalid configuration size for collision evaluation.");
    }
    robot_->update_kinematics(current_q);
  } else {
    robot_->update_kinematics(robot_->get_current_configuration());
  }

  const auto *collision_model =
      static_cast<const RobotModel &>(*robot_).collision_model();
  auto *collision_data = robot_->collision_data();
  if (collision_model == nullptr || collision_data == nullptr) {
    return std::nullopt;
  }

  pinocchio::updateGeometryPlacements(robot_->model(), robot_->data(),
                                      *collision_model, *collision_data);
  double best_distance = std::numeric_limits<double>::infinity();
  bool found = false;

  for (std::size_t idx : pair_indices) {
    if (idx >= collision_model->collisionPairs.size()) continue;
    if (!collision_data->activeCollisionPairs.empty() &&
        !collision_data->activeCollisionPairs[idx]) {
      continue;
    }
    pinocchio::computeDistance(*collision_model, *collision_data, idx);
    const double distance = collision_data->distanceResults[idx].min_distance;
    if (!std::isfinite(distance)) continue;
    best_distance = std::min(best_distance, distance);
    found = true;
  }

  if (!found) return std::nullopt;
  return best_distance;
#else
  (void)current_q;
  (void)pair_indices;
  return std::nullopt;
#endif
}

std::vector<std::size_t>
KinematicsSolver::get_post_step_rejection_pair_indices() const {
  std::unordered_set<std::size_t> unique;
  for (std::size_t idx : last_collision_constraint_pair_indices_) {
    unique.insert(idx);
  }
  for (std::size_t idx : collision_cached_candidate_pair_indices_) {
    unique.insert(idx);
  }
  return std::vector<std::size_t>(unique.begin(), unique.end());
}

// Post-step collision rejection safety margin.  The early-exit gate skips
std::optional<double>
KinematicsSolver::evaluate_post_step_collision_distance(
    const Eigen::VectorXd &q) {
  const PositionStepMutableStateSnapshot snapshot =
      capture_position_step_mutable_state();
  try {
    const auto distance =
        evaluate_post_step_collision_distance_mutating(q);
    restore_position_step_mutable_state(snapshot);
    return distance;
  } catch (...) {
    restore_position_step_mutable_state(snapshot);
    throw;
  }
}

std::optional<double>
KinematicsSolver::evaluate_post_step_collision_distance_mutating(
    const Eigen::VectorXd &q) {
  // Tier 1: Early-exit gate.
  // If the constraint computation already found all pairs well above the
  // penetration threshold, a single bounded integration step cannot create
  // penetration.  Skip the scan entirely.
  // Disabled when budget was exhausted (some pairs may not have been checked).
  if (std::isfinite(last_constraint_min_distance_) &&
      !last_collision_budget_exhausted_ &&
      last_constraint_min_distance_ >=
          kCollisionPenetrationDistanceThreshold + kPostStepSafeMargin) {
    return last_constraint_min_distance_;
  }

  // Tier 2: evaluate every active pair's effective recovery margin once, then
  // derive the scalar minimum from those same distance results. This avoids a
  // targeted exact scan followed immediately by a duplicate recovery scan.
  const auto recovery_margins =
      evaluate_post_step_collision_recovery_margins(q);
  if (!recovery_margins.empty()) {
    const auto current_distance =
        evaluate_post_step_collision_distance_from_current_results();
    if (current_distance.has_value()) {
      return current_distance;
    }
  }

  // Tier 3: Targeted fallback when recovery margins are unavailable.
  auto targeted_pairs = get_post_step_rejection_pair_indices();
  if (!targeted_pairs.empty()) {
    return evaluate_min_collision_distance_targeted(q, targeted_pairs);
  }

  // Tier 4: Full scan fallback (no constraint data available).
  return evaluate_min_collision_distance(q);
}

std::vector<double>
KinematicsSolver::evaluate_post_step_collision_recovery_margins(
    const Eigen::VectorXd &current_q) {
#ifdef PINOCCHIO_WITH_HPP_FCL
  if (!collision_constraint_.has_value() ||
      !collision_constraint_->enabled) {
    return {};
  }
  if (current_q.size() != robot_->nq()) {
    throw std::runtime_error(
        "Invalid configuration size for collision recovery evaluation.");
  }

  const auto *collision_model =
      static_cast<const RobotModel &>(*robot_).collision_model();
  auto *collision_data = robot_->collision_data();
  if (collision_model == nullptr || collision_data == nullptr) {
    return {};
  }

  robot_->update_kinematics(current_q);
  pinocchio::updateGeometryPlacements(robot_->model(), robot_->data(),
                                      *collision_model, *collision_data);

  const std::size_t pair_count = collision_model->collisionPairs.size();
  const auto &pairs = collision_model->collisionPairs;
  const auto &oMi = robot_->data().oMi;
  std::vector<std::uint8_t> enabled_pair_mask(pair_count, 0);
  for (std::size_t pair_index = 0; pair_index < pair_count; ++pair_index) {
    const bool allowed =
        collision_allowed_pair_mask_.empty() ||
        pair_index >= collision_allowed_pair_mask_.size() ||
        collision_allowed_pair_mask_[pair_index];
    const bool active = collision_data->activeCollisionPairs.empty() ||
                        collision_data->activeCollisionPairs[pair_index];
    enabled_pair_mask[pair_index] = allowed && active ? 1 : 0;
  }

  std::vector<double> signed_distance_lower_bounds;
  for (const auto &entry_ptr : post_step_collision_distance_cache_) {
    const auto &entry = *entry_ptr;
    if (entry.q.size() == current_q.size() &&
        (entry.q.array() == current_q.array()).all() &&
        entry.enabled_pair_mask == enabled_pair_mask &&
        entry.certified_signed_distance_lower_bounds.size() == pair_count) {
      signed_distance_lower_bounds =
          entry.certified_signed_distance_lower_bounds;
      break;
    }
  }

  if (signed_distance_lower_bounds.empty()) {
    signed_distance_lower_bounds.assign(
        pair_count, std::numeric_limits<double>::quiet_NaN());
    const PostStepCollisionDistanceCertificate *nearest_certificate = nullptr;
    double nearest_certificate_distance =
        std::numeric_limits<double>::infinity();
    for (const auto &entry_ptr : post_step_collision_distance_cache_) {
      const auto &entry = *entry_ptr;
      if (entry.enabled_pair_mask == enabled_pair_mask &&
          entry.certified_signed_distance_lower_bounds.size() == pair_count &&
          entry.joint_translations.size() == oMi.size() &&
          entry.joint_rotations.size() == oMi.size()) {
        const Eigen::VectorXd delta =
            pinocchio::difference(robot_->model(), entry.q, current_q);
        const double distance = delta.allFinite()
                                    ? delta.squaredNorm()
                                    : std::numeric_limits<double>::infinity();
        if (distance < nearest_certificate_distance) {
          nearest_certificate = &entry;
          nearest_certificate_distance = distance;
        }
      }
    }

    for (std::size_t pair_index = 0; pair_index < pair_count; ++pair_index) {
      if (!enabled_pair_mask[pair_index]) {
        continue;
      }

      const auto &pair = pairs[pair_index];
      const auto &name_a =
          collision_model->geometryObjects[pair.first].name;
      const auto &name_b =
          collision_model->geometryObjects[pair.second].name;
      const std::string pair_key = canonical_pair_key(name_a, name_b);
      double effective_min_distance = collision_constraint_->min_distance;
      const auto override_iter =
          per_pair_min_distance_overrides_.find(pair_key);
      if (override_iter != per_pair_min_distance_overrides_.end()) {
        effective_min_distance = override_iter->second;
      }

      double recovery_target = effective_min_distance;
      if (non_worsening_collision_floor_enabled_) {
        const auto floor_iter = collision_pair_distance_floor_.find(pair_key);
        if (floor_iter != collision_pair_distance_floor_.end()) {
          recovery_target =
              std::min(floor_iter->second, effective_min_distance);
        }
      }
      if (!acceleration_limits_enabled_ && position_step_call_depth_ > 0 &&
          pair_index < position_step_collision_command_floor_distances_.size() &&
          std::isfinite(
              position_step_collision_command_floor_distances_[pair_index])) {
        recovery_target = std::max(
            recovery_target,
            position_step_collision_command_floor_distances_[pair_index] -
                kCollisionTolerance);
      }

      const double required_safe_distance =
          std::max(
              {collision_constraint_->min_distance,
               effective_min_distance + collision_repulsion_deadband_,
               recovery_target + kCollisionTolerance}) +
          sphere_broadphase_.safety_margin;

      double broadphase_lower_bound =
          -std::numeric_limits<double>::infinity();
      if (sphere_broadphase_enabled_ && sphere_broadphase_.is_built()) {
        const auto joint_a =
            collision_model->geometryObjects[pair.first].parentJoint;
        const auto joint_b =
            collision_model->geometryObjects[pair.second].parentJoint;
        if (joint_a < static_cast<pinocchio::JointIndex>(oMi.size()) &&
            joint_b < static_cast<pinocchio::JointIndex>(oMi.size())) {
          broadphase_lower_bound = sphere_broadphase_.compute_pair_lower_bound(
              pair.first, pair.second, oMi[joint_a].translation(),
              oMi[joint_a].rotation(), oMi[joint_b].translation(),
              oMi[joint_b].rotation());
        }
      }

      double motion_lower_bound =
          -std::numeric_limits<double>::infinity();
      if (sphere_broadphase_enabled_ && sphere_broadphase_.is_built() &&
          pair.first < sphere_broadphase_.size() &&
          pair.second < sphere_broadphase_.size()) {
        const auto joint_a =
            collision_model->geometryObjects[pair.first].parentJoint;
        const auto joint_b =
            collision_model->geometryObjects[pair.second].parentJoint;
        if (joint_a < static_cast<pinocchio::JointIndex>(oMi.size()) &&
            joint_b < static_cast<pinocchio::JointIndex>(oMi.size())) {
          const auto geometry_motion_bound =
              [&](const PostStepCollisionDistanceCertificate &entry,
                  std::size_t geometry_index,
                  pinocchio::JointIndex joint_index) {
                const Eigen::Vector3d translation_delta =
                    oMi[joint_index].translation() -
                    entry.joint_translations[joint_index];
                const Eigen::Matrix3d rotation_delta =
                    entry.joint_rotations[joint_index].transpose() *
                    oMi[joint_index].rotation();
                const double cosine = std::clamp(
                    (rotation_delta.trace() - 1.0) * 0.5, -1.0, 1.0);
                const double rotation_chord =
                    std::sqrt(2.0 * std::max(0.0, 1.0 - cosine));
                return translation_delta.norm() +
                       sphere_broadphase_.sphere(geometry_index).motion_radius *
                           rotation_chord;
              };

          if (nearest_certificate != nullptr) {
            const double cached_lower_bound =
                nearest_certificate
                    ->certified_signed_distance_lower_bounds[pair_index];
            if (std::isfinite(cached_lower_bound)) {
              motion_lower_bound =
                  cached_lower_bound -
                  geometry_motion_bound(*nearest_certificate, pair.first,
                                        joint_a) -
                  geometry_motion_bound(*nearest_certificate, pair.second,
                                        joint_b);
            }
          }
        }
      }

      const bool broadphase_certified =
          std::isfinite(broadphase_lower_bound) &&
          broadphase_lower_bound > required_safe_distance;
      const bool motion_bound_certified =
          std::isfinite(motion_lower_bound) &&
          motion_lower_bound > required_safe_distance;
      if (broadphase_certified || motion_bound_certified) {
        signed_distance_lower_bounds[pair_index] =
            std::max(broadphase_lower_bound, motion_lower_bound);
        if (motion_bound_certified && !broadphase_certified) {
          last_post_step_collision_motion_bound_culled_pairs_++;
        }
        if (non_worsening_collision_floor_enabled_ &&
            collision_pair_distance_floor_.find(pair_key) ==
                collision_pair_distance_floor_.end()) {
          collision_pair_distance_floor_.emplace(pair_key,
                                                 effective_min_distance);
        }
        continue;
      }

      last_post_step_collision_exact_distance_queries_++;
      pinocchio::computeDistance(*collision_model, *collision_data, pair_index);
      signed_distance_lower_bounds[pair_index] =
          collision_data->distanceResults[pair_index].min_distance;
    }

    PostStepCollisionDistanceCertificate certificate;
    certificate.q = current_q;
    certificate.enabled_pair_mask = enabled_pair_mask;
    certificate.certified_signed_distance_lower_bounds =
        signed_distance_lower_bounds;
    certificate.joint_translations.reserve(oMi.size());
    certificate.joint_rotations.reserve(oMi.size());
    for (const auto &joint_placement : oMi) {
      certificate.joint_translations.push_back(joint_placement.translation());
      certificate.joint_rotations.push_back(joint_placement.rotation());
    }
    if (post_step_collision_distance_cache_.size() >=
        kPostStepCollisionCacheCapacity) {
      post_step_collision_distance_cache_.erase(
          post_step_collision_distance_cache_.begin());
    }
    post_step_collision_distance_cache_.push_back(
        std::make_shared<const PostStepCollisionDistanceCertificate>(
            std::move(certificate)));
  }
  for (std::size_t pair_index = 0; pair_index < pair_count; ++pair_index) {
    if (enabled_pair_mask[pair_index] &&
        std::isfinite(signed_distance_lower_bounds[pair_index])) {
      collision_data->distanceResults[pair_index].min_distance =
          signed_distance_lower_bounds[pair_index];
    }
  }

  std::vector<double> margins(
      pair_count, std::numeric_limits<double>::quiet_NaN());
  for (std::size_t pair_index = 0; pair_index < pair_count; ++pair_index) {
    if (!enabled_pair_mask[pair_index]) {
      continue;
    }

    const double signed_distance = signed_distance_lower_bounds[pair_index];
    if (!std::isfinite(signed_distance)) {
      continue;
    }

    const auto &pair = collision_model->collisionPairs[pair_index];
    const auto &name_a =
        collision_model->geometryObjects[pair.first].name;
    const auto &name_b =
        collision_model->geometryObjects[pair.second].name;
    const std::string pair_key = canonical_pair_key(name_a, name_b);

    double effective_min_distance = collision_constraint_->min_distance;
    const auto override_iter =
        per_pair_min_distance_overrides_.find(pair_key);
    if (override_iter != per_pair_min_distance_overrides_.end()) {
      effective_min_distance = override_iter->second;
    }

    double recovery_target = effective_min_distance;
    if (non_worsening_collision_floor_enabled_) {
      auto floor_iter = collision_pair_distance_floor_.find(pair_key);
      if (floor_iter == collision_pair_distance_floor_.end()) {
        const double seeded =
            signed_distance < effective_min_distance
                ? std::min(effective_min_distance,
                           collision_structural_floor_)
                : effective_min_distance;
        floor_iter =
            collision_pair_distance_floor_.emplace(pair_key, seeded).first;
      }
      recovery_target =
          std::min(floor_iter->second, effective_min_distance);
    }
    if (!acceleration_limits_enabled_ && position_step_call_depth_ > 0 &&
        pair_index < position_step_collision_command_floor_distances_.size() &&
        std::isfinite(
            position_step_collision_command_floor_distances_[pair_index])) {
      recovery_target = std::max(
          recovery_target,
          position_step_collision_command_floor_distances_[pair_index] -
              kCollisionTolerance);
    }

    margins[pair_index] = signed_distance - recovery_target;
  }

  return margins;
#else
  (void)current_q;
  return {};
#endif
}

std::optional<double>
KinematicsSolver::evaluate_post_step_collision_distance_from_current_results() {
#ifdef PINOCCHIO_WITH_HPP_FCL
  const auto *collision_model =
      static_cast<const RobotModel &>(*robot_).collision_model();
  auto *collision_data = robot_->collision_data();
  if (collision_model == nullptr || collision_data == nullptr) {
    return std::nullopt;
  }

  const auto minimum_for_pairs =
      [&](const std::vector<std::size_t> &pair_indices)
      -> std::optional<double> {
    double minimum_distance = std::numeric_limits<double>::infinity();
    bool found = false;
    for (const std::size_t pair_index : pair_indices) {
      if (pair_index >= collision_model->collisionPairs.size()) {
        continue;
      }
      if (!collision_data->activeCollisionPairs.empty() &&
          !collision_data->activeCollisionPairs[pair_index]) {
        continue;
      }
      const double distance =
          collision_data->distanceResults[pair_index].min_distance;
      if (!std::isfinite(distance)) {
        continue;
      }
      minimum_distance = std::min(minimum_distance, distance);
      found = true;
    }
    return found ? std::optional<double>(minimum_distance) : std::nullopt;
  };

  const std::vector<std::size_t> targeted_pairs =
      get_post_step_rejection_pair_indices();
  if (!targeted_pairs.empty()) {
    return minimum_for_pairs(targeted_pairs);
  }

  std::vector<std::size_t> all_pairs(collision_model->collisionPairs.size());
  std::iota(all_pairs.begin(), all_pairs.end(), std::size_t{0});
  return minimum_for_pairs(all_pairs);
#else
  return std::nullopt;
#endif
}

void KinematicsSolver::capture_position_step_collision_command_floor(
    const Eigen::VectorXd &current_q) {
  position_step_collision_command_floor_distances_.clear();
#ifdef PINOCCHIO_WITH_HPP_FCL
  // A command-local floor can jump when a moving target starts a new command,
  // invalidating the previous tick's acceleration-feasible stopping proof.
  // Acceleration-limited steps use the persistent recovery floor and braking
  // certificate instead.
  if (acceleration_limits_enabled_) {
    return;
  }
  if (!collision_constraint_.has_value() ||
      !collision_constraint_->enabled) {
    return;
  }
  const std::vector<double> current_margins =
      evaluate_post_step_collision_recovery_margins(current_q);
  const auto *collision_model =
      static_cast<const RobotModel &>(*robot_).collision_model();
  if (collision_model == nullptr || current_margins.empty()) {
    return;
  }

  position_step_collision_command_floor_distances_.assign(
      current_margins.size(), std::numeric_limits<double>::quiet_NaN());
  for (std::size_t pair_index = 0; pair_index < current_margins.size();
       ++pair_index) {
    const double margin = current_margins[pair_index];
    if (!std::isfinite(margin) ||
        pair_index >= collision_model->collisionPairs.size()) {
      continue;
    }

    const auto &pair = collision_model->collisionPairs[pair_index];
    const auto &name_a = collision_model->geometryObjects[pair.first].name;
    const auto &name_b = collision_model->geometryObjects[pair.second].name;
    const std::string pair_key = canonical_pair_key(name_a, name_b);
    double effective_min_distance = collision_constraint_->min_distance;
    const auto override_iter =
        per_pair_min_distance_overrides_.find(pair_key);
    if (override_iter != per_pair_min_distance_overrides_.end()) {
      effective_min_distance = override_iter->second;
    }
    double recovery_target = effective_min_distance;
    if (non_worsening_collision_floor_enabled_) {
      const auto floor_iter = collision_pair_distance_floor_.find(pair_key);
      if (floor_iter != collision_pair_distance_floor_.end()) {
        recovery_target =
            std::min(floor_iter->second, effective_min_distance);
      }
    }
    const double nominal_margin =
        margin + recovery_target - effective_min_distance;
    if (nominal_margin >= -kCollisionTolerance &&
        nominal_margin <= kCollisionRepulsionDeadband) {
      position_step_collision_command_floor_distances_[pair_index] =
          margin + recovery_target;
    }
  }
#else
  (void)current_q;
#endif
}

std::optional<double>
KinematicsSolver::evaluate_per_pair_override_violations(const Eigen::VectorXd &q) {
#ifdef PINOCCHIO_WITH_HPP_FCL
  if (per_pair_min_distance_overrides_.empty()) return std::nullopt;
  const auto *geom_model =
      static_cast<const RobotModel &>(*robot_).collision_model();
  if (!geom_model) return std::nullopt;

  double worst_margin = std::numeric_limits<double>::infinity();
  bool any_checked = false;
  const auto &pairs = geom_model->collisionPairs;

  for (std::size_t pi = 0; pi < pairs.size(); ++pi) {
    // Skip excluded pairs.
    if (!collision_allowed_pair_mask_.empty() &&
        pi < collision_allowed_pair_mask_.size() &&
        !collision_allowed_pair_mask_[pi]) continue;

    const auto &cp = pairs[pi];
    const auto &name_a = geom_model->geometryObjects[cp.first].name;
    const auto &name_b = geom_model->geometryObjects[cp.second].name;
    const auto oit =
        per_pair_min_distance_overrides_.find(canonical_pair_key(name_a, name_b));
    if (oit == per_pair_min_distance_overrides_.end()) continue;

    // This pair has an active override — compute its exact distance.
    auto dist_opt = evaluate_min_collision_distance_targeted(q, {pi});
    if (!dist_opt.has_value() || !std::isfinite(*dist_opt)) continue;

    const double margin = *dist_opt - oit->second;  // negative → violated
    worst_margin = std::min(worst_margin, margin);
    any_checked = true;
  }

  return any_checked ? std::make_optional(worst_margin) : std::nullopt;
#else
  (void)q;
  return std::nullopt;
#endif
}

void KinematicsSolver::add_collision_constraint(
    const std::vector<std::pair<std::string, std::string>> &link_pairs,
    double min_distance) {
#ifdef PINOCCHIO_WITH_HPP_FCL
  configure_collision_constraint(min_distance, link_pairs, {}, true);
#else
  (void)link_pairs;
  (void)min_distance;
  throw std::runtime_error("Collision avoidance requires Pinocchio to be built "
                           "with hpp-fcl support.");
#endif
}

bool KinematicsSolver::set_collision_min_distance(double min_distance) {
  if (!collision_constraint_.has_value() || !collision_constraint_->enabled) {
    return false;
  }
  collision_constraint_->min_distance = std::max(0.0, min_distance);
  if (collision_constraint_->constraint_activation_multiplier > 0.0) {
    const double activation_distance =
        stall_config_.enabled
            ? std::max(collision_constraint_->min_distance,
                       stall_state_.nominal_min_distance)
            : collision_constraint_->min_distance;
    collision_constraint_->constraint_activation_margin =
        collision_constraint_->constraint_activation_multiplier *
        activation_distance;
  } else {
    collision_constraint_->constraint_activation_margin = 0.0;
  }
  post_step_collision_distance_cache_.clear();
  ++collision_validation_policy_revision_;
  invalidate_collision_validation_state_certificates();
  return true;
}

void KinematicsSolver::set_proximity_gated_collision_activation_enabled(
    bool enabled) {
  if (!collision_constraint_.has_value()) {
    collision_constraint_.emplace();
  }
  collision_constraint_->constraint_activation_enabled = enabled;
}

bool KinematicsSolver::get_proximity_gated_collision_activation_enabled() const {
  if (!collision_constraint_.has_value()) {
    return false;
  }
  return collision_constraint_->constraint_activation_enabled;
}

void KinematicsSolver::set_collision_constraint_activation_multiplier(
    double multiplier) {
  if (!collision_constraint_.has_value()) {
    collision_constraint_.emplace();
  }
  auto &config = *collision_constraint_;
  config.constraint_activation_multiplier = std::max(0.0, multiplier);
  if (config.constraint_activation_multiplier > 0.0) {
    const double activation_distance =
        stall_config_.enabled
            ? std::max(config.min_distance, stall_state_.nominal_min_distance)
            : std::max(0.0, config.min_distance);
    config.constraint_activation_margin =
        config.constraint_activation_multiplier * activation_distance;
  } else {
    config.constraint_activation_margin = 0.0;
  }
}

double KinematicsSolver::get_collision_constraint_activation_multiplier() const {
  if (!collision_constraint_.has_value()) {
    return 0.0;
  }
  return collision_constraint_->constraint_activation_multiplier;
}

double KinematicsSolver::get_collision_constraint_activation_margin() const {
  if (!collision_constraint_.has_value()) {
    return 0.0;
  }
  return collision_constraint_->constraint_activation_margin;
}

void KinematicsSolver::enable_collision_pair_cache(
    bool enable, int full_refresh_interval, double candidate_distance_margin,
    int max_cached_candidates) {
  collision_pair_cache_enabled_ = enable;
  collision_pair_cache_refresh_interval_ = std::max(1, full_refresh_interval);
  collision_pair_cache_distance_margin_ = std::max(0.0, candidate_distance_margin);
  collision_pair_cache_max_candidates_ = std::max(1, max_cached_candidates);
  collision_pair_cache_has_full_scan_ = false;
  collision_pair_cache_steps_since_refresh_ = 0;
  collision_cached_candidate_pair_indices_.clear();
  std::fill(collision_pair_bound_valid_.begin(), collision_pair_bound_valid_.end(),
            static_cast<std::uint8_t>(0));
  std::fill(collision_pair_last_signed_distance_.begin(),
            collision_pair_last_signed_distance_.end(),
            std::numeric_limits<double>::infinity());
  std::fill(collision_pair_last_rel_translation_norm_.begin(),
            collision_pair_last_rel_translation_norm_.end(), 0.0);
  std::fill(collision_pair_last_rel_rotation_.begin(),
            collision_pair_last_rel_rotation_.end(),
            pack_rotation_matrix(Eigen::Matrix3d::Identity()));
  last_collision_pairs_considered_ = 0;
  last_collision_exact_distance_queries_ = 0;
  last_collision_bound_culled_pairs_ = 0;
  last_collision_sphere_culled_pairs_ = 0;
  last_collision_budget_exhausted_ = false;
}

void KinematicsSolver::set_collision_refinement_time_budget_us(int budget_us) {
  collision_refinement_time_budget_us_ = std::max(0, budget_us);
}

int KinematicsSolver::get_collision_refinement_time_budget_us() const {
  return collision_refinement_time_budget_us_;
}

void KinematicsSolver::set_collision_tuning_mode(CollisionTuningMode mode) {
  collision_tuning_mode_ = mode;
  switch (mode) {
  case CollisionTuningMode::kPrecise:
    // Exact full-scan behavior: safest/most accurate, highest compute cost.
    //
    // Budget is intentionally disabled (0) so exact distance refinement does not
    // stop early under time pressure.
    // Use legacy-equivalent disabled-cache defaults.
    enable_collision_pair_cache(false, 20, 0.03, 128);
    set_collision_refinement_time_budget_us(0);
    set_proximity_gated_collision_activation_enabled(false);
    set_collision_constraint_activation_multiplier(0.0);
    enable_sphere_broadphase(false);
    break;
  case CollisionTuningMode::kBalanced:
    // Conservative compromise:
    // - keep cache enabled to skip obviously far pairs in clear space
    // - refresh frequently enough to catch newly critical pairs in dense
    //   self-collision models before they can drift through the clearance shell
    // - but keep budget disabled (0), so once candidate pairs are selected we
    //   avoid early termination and preserve higher collision-distance fidelity
    //   than the speed preset.
    enable_collision_pair_cache(true, 5, 0.05, 256);
    set_collision_refinement_time_budget_us(0);
    set_proximity_gated_collision_activation_enabled(true);
    // Optional proximity-gated activation: rows are emitted only near
    // min_distance, with a conservative activation band.
    set_collision_constraint_activation_multiplier(5.0);
    enable_sphere_broadphase(true);
    break;
  case CollisionTuningMode::kSpeed:
  default:
    // Teleop-optimized path.
    // A positive budget intentionally bounds worst-case per-step collision
    // refinement cost, trading some edge-case precision for predictable latency.
    enable_collision_pair_cache(true, 100, 0.03, 128);
    set_collision_refinement_time_budget_us(300);
    set_proximity_gated_collision_activation_enabled(true);
    // Tighter activation band than BALANCED for lower steady-state overhead.
    set_collision_constraint_activation_multiplier(3.0);
    enable_sphere_broadphase(true);
    break;
  }
}

CollisionTuningMode KinematicsSolver::get_collision_tuning_mode() const {
  return collision_tuning_mode_;
}

void KinematicsSolver::enable_sphere_broadphase(bool enable) {
#ifdef PINOCCHIO_WITH_HPP_FCL
  if (sphere_broadphase_enabled_ != enable) {
    post_step_collision_distance_cache_.clear();
  }
  sphere_broadphase_enabled_ = enable;
  if (enable && !sphere_broadphase_.is_built() && robot_->has_collision_geometry()) {
    sphere_broadphase_.build(
        *static_cast<const RobotModel &>(*robot_).collision_model());
  }
#else
  (void)enable;
#endif
}

double KinematicsSolver::get_collision_min_distance() const {
  if (!collision_constraint_.has_value() || !collision_constraint_->enabled) {
    return -1.0;
  }
  return collision_constraint_->min_distance;
}

void KinematicsSolver::clear_collision_constraint() {
  if (collision_constraint_.has_value()) {
    collision_constraint_->enabled = false;
  }
  collision_allowed_pair_mask_.clear();
  last_collision_constraint_pair_indices_.clear();
  collision_cached_candidate_pair_indices_.clear();
  collision_pair_bound_valid_.clear();
  collision_pair_last_signed_distance_.clear();
  collision_pair_last_rel_translation_norm_.clear();
  collision_pair_last_rel_rotation_.clear();
  collision_pair_distance_floor_.clear();
  post_step_collision_distance_cache_.clear();
  collision_cache_frozen_indices_.clear();
  collision_pair_cache_has_full_scan_ = false;
  collision_pair_cache_steps_since_refresh_ = 0;
  last_collision_pairs_considered_ = 0;
  last_collision_exact_distance_queries_ = 0;
  last_collision_bound_culled_pairs_ = 0;
  last_collision_sphere_culled_pairs_ = 0;
  last_collision_budget_exhausted_ = false;
  last_constraint_min_distance_ = std::numeric_limits<double>::infinity();
  last_constraint_min_recovery_margin_ =
      std::numeric_limits<double>::infinity();
  last_constraint_was_full_scan_ = false;
  last_collision_constraint_result_.reset();
  last_collision_constraint_row_dt_ =
      std::numeric_limits<double>::quiet_NaN();
  last_collision_constraint_q_ = Eigen::VectorXd();
  collision_stuck_counters_.clear();
  collision_stuck_last_distances_.clear();
  ++collision_validation_policy_revision_;
  invalidate_collision_validation_cache();
}

void KinematicsSolver::set_collision_pair_min_distance(const std::string &link_a,
                                                        const std::string &link_b,
                                                        double min_distance,
                                                        bool activate_when_clear) {
  const auto *geom_model =
      static_cast<const RobotModel &>(*robot_).collision_model();
  if (!geom_model) return;

  // Collect geometry indices whose parent frame name contains link_a or link_b.
  std::vector<std::size_t> geoms_a, geoms_b;
  for (std::size_t gi = 0; gi < geom_model->ngeoms; ++gi) {
    const auto &go = geom_model->geometryObjects[gi];
    const std::string &frame_name =
        robot_->model().frames[go.parentFrame].name;
    if (frame_name.find(link_a) != std::string::npos) geoms_a.push_back(gi);
    if (frame_name.find(link_b) != std::string::npos) geoms_b.push_back(gi);
  }

  for (std::size_t pi = 0; pi < geom_model->collisionPairs.size(); ++pi) {
    const auto &cp = geom_model->collisionPairs[pi];
    const bool fwd =
        std::find(geoms_a.begin(), geoms_a.end(), cp.first) != geoms_a.end() &&
        std::find(geoms_b.begin(), geoms_b.end(), cp.second) != geoms_b.end();
    const bool rev =
        std::find(geoms_b.begin(), geoms_b.end(), cp.first) != geoms_b.end() &&
        std::find(geoms_a.begin(), geoms_a.end(), cp.second) != geoms_a.end();
    if (fwd || rev) {
      const auto &name_a = geom_model->geometryObjects[cp.first].name;
      const auto &name_b = geom_model->geometryObjects[cp.second].name;
      const std::string key = canonical_pair_key(name_a, name_b);
      if (activate_when_clear) {
        // Deferred: store as pending; compute_bounds_for_pair promotes to active
        // the first time signed_distance >= min_distance (latch-on semantics).
        // Prevents immediate stall when called from inside the threshold.
        per_pair_deferred_overrides_[key] = min_distance;
        per_pair_min_distance_overrides_.erase(key);  // not active yet
      } else {
        // Immediate: override takes effect on the next solve tick.
        per_pair_min_distance_overrides_[key] = min_distance;
        per_pair_deferred_overrides_.erase(key);
      }
    }
  }
  post_step_collision_distance_cache_.clear();
  ++collision_validation_policy_revision_;
  invalidate_collision_validation_state_certificates();
}

void KinematicsSolver::clear_collision_pair_min_distance(const std::string &link_a,
                                                          const std::string &link_b) {
  const auto *geom_model =
      static_cast<const RobotModel &>(*robot_).collision_model();
  if (!geom_model) return;

  std::vector<std::size_t> geoms_a, geoms_b;
  for (std::size_t gi = 0; gi < geom_model->ngeoms; ++gi) {
    const auto &go = geom_model->geometryObjects[gi];
    const std::string &frame_name =
        robot_->model().frames[go.parentFrame].name;
    if (frame_name.find(link_a) != std::string::npos) geoms_a.push_back(gi);
    if (frame_name.find(link_b) != std::string::npos) geoms_b.push_back(gi);
  }

  for (std::size_t pi = 0; pi < geom_model->collisionPairs.size(); ++pi) {
    const auto &cp = geom_model->collisionPairs[pi];
    const bool fwd =
        std::find(geoms_a.begin(), geoms_a.end(), cp.first) != geoms_a.end() &&
        std::find(geoms_b.begin(), geoms_b.end(), cp.second) != geoms_b.end();
    const bool rev =
        std::find(geoms_b.begin(), geoms_b.end(), cp.first) != geoms_b.end() &&
        std::find(geoms_a.begin(), geoms_a.end(), cp.second) != geoms_a.end();
    if (fwd || rev) {
      const auto &name_a = geom_model->geometryObjects[cp.first].name;
      const auto &name_b = geom_model->geometryObjects[cp.second].name;
      const std::string key = canonical_pair_key(name_a, name_b);
      per_pair_min_distance_overrides_.erase(key);
      per_pair_deferred_overrides_.erase(key);
    }
  }
  post_step_collision_distance_cache_.clear();
  ++collision_validation_policy_revision_;
  invalidate_collision_validation_state_certificates();
}

std::vector<std::pair<std::string, double>>
KinematicsSolver::get_collision_pair_min_distance_overrides() const {
  std::vector<std::pair<std::string, double>> result;
  result.reserve(per_pair_min_distance_overrides_.size() +
                 per_pair_deferred_overrides_.size());
  for (const auto &kv : per_pair_min_distance_overrides_) {
    result.emplace_back(kv.first, kv.second);
  }
  for (const auto &kv : per_pair_deferred_overrides_) {
    result.emplace_back(kv.first, kv.second);  // pending (not yet active)
  }
  return result;
}

std::string KinematicsSolver::canonical_pair_key(const std::string &a,
                                                 const std::string &b) const {
  if (a <= b) {
    return a + "|" + b;
  }
  return b + "|" + a;
}

bool KinematicsSolver::collision_pair_allowed(const std::string &a,
                                              const std::string &b) const {
  if (!collision_constraint_.has_value() || !collision_constraint_->enabled) {
    return false;
  }

  const auto key = canonical_pair_key(a, b);
  const auto &config = *collision_constraint_;

  if (!config.include_pairs.empty() &&
      config.include_pairs.find(key) == config.include_pairs.end()) {
    return false;
  }

  if (config.exclude_pairs.find(key) != config.exclude_pairs.end()) {
    return false;
  }

  return true;
}

std::vector<std::pair<std::string, std::string>>
KinematicsSolver::get_active_collision_pairs() const {
  std::vector<std::pair<std::string, std::string>> result;
#ifdef PINOCCHIO_WITH_HPP_FCL
  if (!robot_->has_collision_geometry()) {
    return result;
  }

  const auto *collision_model_ptr =
      static_cast<const RobotModel &>(*robot_).collision_model();
  if (collision_model_ptr == nullptr) {
    return result;
  }

  if (!collision_constraint_.has_value() || !collision_constraint_->enabled) {
    return result;
  }

  const auto &pairs = collision_model_ptr->collisionPairs;
  if (collision_allowed_pair_mask_.size() == pairs.size()) {
    for (std::size_t idx = 0; idx < pairs.size(); ++idx) {
      if (!collision_allowed_pair_mask_[idx]) {
        continue;
      }
      const auto &pair = pairs[idx];
      const auto &name_a =
          collision_model_ptr->geometryObjects[pair.first].name;
      const auto &name_b =
          collision_model_ptr->geometryObjects[pair.second].name;
      result.emplace_back(name_a, name_b);
    }
  } else {
    for (const auto &pair : pairs) {
      const auto &name_a =
          collision_model_ptr->geometryObjects[pair.first].name;
      const auto &name_b =
          collision_model_ptr->geometryObjects[pair.second].name;
      if (collision_pair_allowed(name_a, name_b)) {
        result.emplace_back(name_a, name_b);
      }
    }
  }
#endif
  return result;
}

KinematicsSolver::CollisionRecoveryDistances
KinematicsSolver::resolve_collision_recovery_distances(
    std::size_t pair_index, double signed_distance) {
#ifdef PINOCCHIO_WITH_HPP_FCL
  if (!collision_constraint_.has_value() ||
      !collision_constraint_->enabled) {
    throw std::logic_error(
        "collision recovery distances require a configured constraint");
  }
  const auto *collision_model =
      static_cast<const RobotModel &>(*robot_).collision_model();
  if (collision_model == nullptr ||
      pair_index >= collision_model->collisionPairs.size()) {
    throw std::out_of_range("collision recovery pair index is invalid");
  }

  const auto &pair = collision_model->collisionPairs[pair_index];
  const auto &name_a = collision_model->geometryObjects[pair.first].name;
  const auto &name_b = collision_model->geometryObjects[pair.second].name;
  const std::string pair_key = canonical_pair_key(name_a, name_b);

  CollisionRecoveryDistances distances;
  distances.effective_min_distance = collision_constraint_->min_distance;
  const auto deferred = per_pair_deferred_overrides_.find(pair_key);
  if (deferred != per_pair_deferred_overrides_.end() &&
      signed_distance >= deferred->second) {
    per_pair_min_distance_overrides_[pair_key] = deferred->second;
    per_pair_deferred_overrides_.erase(deferred);
    post_step_collision_distance_cache_.clear();
    ++collision_validation_policy_revision_;
    invalidate_collision_validation_state_certificates();
  }
  const auto override = per_pair_min_distance_overrides_.find(pair_key);
  if (override != per_pair_min_distance_overrides_.end()) {
    distances.effective_min_distance = override->second;
  }

  distances.recovery_target = distances.effective_min_distance;
  if (non_worsening_collision_floor_enabled_) {
    auto floor = collision_pair_distance_floor_.find(pair_key);
    if (floor == collision_pair_distance_floor_.end()) {
      const double seeded =
          signed_distance < distances.effective_min_distance
              ? std::min(distances.effective_min_distance,
                         collision_structural_floor_)
              : distances.effective_min_distance;
      floor = collision_pair_distance_floor_.emplace(pair_key, seeded).first;
    }
    distances.recovery_target =
        std::min(floor->second, distances.effective_min_distance);
  }
  if (!acceleration_limits_enabled_ && position_step_call_depth_ > 0 &&
      pair_index < position_step_collision_command_floor_distances_.size() &&
      std::isfinite(
          position_step_collision_command_floor_distances_[pair_index])) {
    distances.recovery_target =
        std::max(distances.recovery_target,
                 position_step_collision_command_floor_distances_[pair_index] -
                     kCollisionTolerance);
  }
  return distances;
#else
  (void)pair_index;
  (void)signed_distance;
  throw std::logic_error(
      "collision recovery distances require Pinocchio collision support");
#endif
}

std::optional<KinematicsSolver::CollisionConstraintResult>
KinematicsSolver::compute_collision_constraint() {
  return compute_collision_constraint(std::max(dt_, 1e-6));
}

std::optional<KinematicsSolver::CollisionConstraintResult>
KinematicsSolver::compute_collision_constraint(double row_dt) {
  if (row_dt <= 0.0 || !std::isfinite(row_dt)) {
    throw std::invalid_argument(
        "collision constraint row_dt must be finite and positive");
  }
#ifdef PINOCCHIO_WITH_HPP_FCL
  const double constraint_dt = row_dt;
  const bool acceleration_braking_lookahead_active =
      acceleration_limits_enabled_ && position_step_call_depth_ > 0 &&
      previous_dq_.size() == robot_->nv() && previous_dq_.allFinite() &&
      acceleration_limits_.size() == robot_->nv() &&
      acceleration_limits_.allFinite() &&
      previous_dq_.cwiseAbs().maxCoeff() > constraint_tolerance_;
  // Lazy reuse: when configuration change is small and we have a safe margin,
  // reuse the previous constraint result.  The constraint Jacobian and bounds
  // remain approximately valid for small dq, and the safety margin absorbs
  // the approximation error.
  // Checked BEFORE resetting debug state so that last_collision_debug_ and
  // last_collision_debug_list_ remain valid for callers (avoids flickering
  // in visualization).
  {
    const bool constraint_active_for_reuse =
        collision_constraint_.has_value() && collision_constraint_->enabled;
    if (constraint_active_for_reuse && collision_pair_cache_enabled_ &&
        !acceleration_braking_lookahead_active &&
        last_collision_constraint_result_.has_value() &&
        std::isfinite(last_collision_constraint_row_dt_) &&
        last_collision_constraint_row_dt_ == constraint_dt &&
        last_collision_constraint_q_.size() == robot_->nq() &&
        std::isfinite(last_constraint_min_distance_) &&
        !last_collision_budget_exhausted_ &&
        // Invalidate when the frozen (locked/excluded) DOF set changed: it alters
        // which pairs are controllable / selected, so the cached result is stale.
        collision_cache_frozen_indices_ == active_collision_lock_indices() &&
        // Respect the cache refresh interval: periodic full recomputation
        // prevents the cached result from going permanently stale.
        collision_pair_cache_steps_since_refresh_ <
            collision_pair_cache_refresh_interval_) {
      const Eigen::VectorXd &q_current = robot_->get_current_configuration();
      const double dq_norm =
          (q_current - last_collision_constraint_q_).squaredNorm();
      if (dq_norm < kLazyReuseMaxDqSqNorm &&
          last_constraint_min_distance_ >
              kCollisionPenetrationDistanceThreshold +
                  kLazyReuseMinDistMargin) {
        // Reuse previous result — configuration barely changed.
        // Preserve last_collision_debug_ and instrumentation counters.
        collision_pair_cache_steps_since_refresh_++;
        last_collision_pairs_considered_ = 0;
        last_collision_exact_distance_queries_ = 0;
        last_collision_bound_culled_pairs_ = 0;
        return last_collision_constraint_result_;
      }
    }
  }

  last_collision_pairs_considered_ = 0;
  last_collision_exact_distance_queries_ = 0;
  last_collision_bound_culled_pairs_ = 0;
  last_collision_sphere_culled_pairs_ = 0;
  last_collision_budget_exhausted_ = false;
  last_constraint_min_recovery_margin_ =
      std::numeric_limits<double>::infinity();
  last_collision_debug_.reset();
  last_collision_debug_list_.clear();
  if (!robot_->has_collision_geometry()) {
    return std::nullopt;
  }

  const auto *collision_model =
      static_cast<const RobotModel &>(*robot_).collision_model();
  auto *collision_data = robot_->collision_data();
  if (collision_model == nullptr || collision_data == nullptr) {
    return std::nullopt;
  }

  bool constraint_active =
      collision_constraint_.has_value() && collision_constraint_->enabled;
  const bool nearest_points_all_pairs =
      collision_constraint_.has_value() &&
      collision_constraint_->nearest_points_all_pairs;

  // Fast path: when the cache is warm and previous step confirmed all pairs
  // are well clear of the activation threshold, skip the expensive geometry
  // update entirely and return an empty constraint (no active rows).
  //
  // Two guards prevent stale data from permanently silencing the constraint:
  //   1. Refresh interval — forces a full scan every N steps (catches gradual
  //      approach that was missed while all pairs appeared far away).
  //   2. Delta-q threshold — if the robot moved significantly since the last
  //      full scan, last_constraint_min_distance_ may no longer reflect reality
  //      (e.g., arm folded from extended clear-space back toward the torso).
  //      In that case skip the fast path and recompute.
  // Guard 2 for fast-path: robot must not have moved substantially since the
  // last full scan (last_constraint_min_distance_ might be stale otherwise).
  // kFastPathMaxDqSqNorm ≈ 0.1 rad total joint change.
  constexpr double kFastPathMaxDqSqNorm = 0.01;
  const bool robot_q_stable =
      last_collision_constraint_q_.size() == robot_->nq() &&
      (robot_->get_current_configuration() - last_collision_constraint_q_)
              .squaredNorm() <= kFastPathMaxDqSqNorm;

  if (constraint_active && collision_pair_cache_enabled_ &&
      !acceleration_braking_lookahead_active &&
      collision_pair_cache_has_full_scan_ && robot_q_stable &&
      !last_collision_budget_exhausted_ &&
      std::isfinite(last_constraint_min_distance_) &&
      collision_constraint_->constraint_activation_enabled &&
      collision_constraint_->constraint_activation_margin > 0.0 &&
      collision_pair_cache_steps_since_refresh_ < collision_pair_cache_refresh_interval_) {
    const double activation_threshold =
        collision_constraint_->min_distance +
        collision_constraint_->constraint_activation_margin;
    if (last_constraint_min_distance_ > activation_threshold) {
      // All pairs are beyond the activation margin — no constraint rows
      // would be emitted.  Increment the cache step counter and return
      // an empty result, preserving the cached candidate set.
      collision_pair_cache_steps_since_refresh_++;
      last_collision_pairs_considered_ = 0;
      last_collision_exact_distance_queries_ = 0;
      return std::nullopt;
    }
  }

  // Update collision placements and compute distances for all pairs
  pinocchio::updateGeometryPlacements(robot_->model(), robot_->data(),
                                      *collision_model, *collision_data);
  const auto &pairs = collision_model->collisionPairs;

  struct BrakingPlacementSample {
    std::vector<Eigen::Vector3d> joint_translations;
    std::vector<Eigen::Matrix3d> joint_rotations;
  };
  std::vector<BrakingPlacementSample> braking_placement_samples;
  bool braking_lookahead_certified = !acceleration_braking_lookahead_active;
  if (acceleration_braking_lookahead_active &&
      sphere_broadphase_enabled_ && sphere_broadphase_.is_built()) {
    int braking_steps = 0;
    braking_lookahead_certified = true;
    for (int index = 0; index < robot_->nv(); ++index) {
      const double acceleration_step =
          acceleration_limits_[index] * constraint_dt;
      const double deadband = acceleration_step * 0.01;
      const double step = acceleration_step + deadband;
      if (step <= constraint_tolerance_) {
        if (std::abs(previous_dq_[index]) > constraint_tolerance_) {
          braking_lookahead_certified = false;
          break;
        }
        continue;
      }
      braking_steps = std::max(
          braking_steps,
          static_cast<int>(
              std::ceil(std::abs(previous_dq_[index]) / step - 1e-12)));
    }

    if (braking_lookahead_certified && braking_steps > 0) {
      Eigen::VectorXd rollout_q = robot_->get_current_configuration();
      Eigen::VectorXd rollout_velocity = previous_dq_;
      pinocchio::Data rollout_data(robot_->model());
      braking_placement_samples.reserve(braking_steps);
      for (int step_index = 0; step_index < braking_steps; ++step_index) {
        for (int index = 0; index < robot_->nv(); ++index) {
          const double acceleration_step =
              acceleration_limits_[index] * constraint_dt;
          const double deadband = acceleration_step * 0.01;
          const double step = acceleration_step + deadband;
          if (rollout_velocity[index] > step) {
            rollout_velocity[index] -= step;
          } else if (rollout_velocity[index] < -step) {
            rollout_velocity[index] += step;
          } else {
            rollout_velocity[index] = 0.0;
          }
        }
        rollout_q = pinocchio::integrate(
            robot_->model(), rollout_q, constraint_dt * rollout_velocity);
        pinocchio::forwardKinematics(robot_->model(), rollout_data, rollout_q);

        BrakingPlacementSample sample;
        sample.joint_translations.reserve(rollout_data.oMi.size());
        sample.joint_rotations.reserve(rollout_data.oMi.size());
        for (const auto &joint_placement : rollout_data.oMi) {
          sample.joint_translations.push_back(joint_placement.translation());
          sample.joint_rotations.push_back(joint_placement.rotation());
        }
        braking_placement_samples.push_back(std::move(sample));
      }
    }
  }

  double best_distance_debug = std::numeric_limits<double>::infinity();
  std::optional<std::size_t> best_index_debug;

  // All allowed pairs with finite distances: (distance, pair_index)
  std::vector<std::pair<double, std::size_t>> allowed_candidates;
  bool use_cached_candidate_subset = false;
  // Keep cache mode active even when the candidate set is currently empty:
  // in clear-space regimes this avoids an expensive full scan on every tick.
  // A full scan is still forced periodically by refresh_interval.
  if (collision_pair_cache_enabled_ && constraint_active &&
      collision_pair_cache_has_full_scan_ &&
      collision_pair_cache_steps_since_refresh_ <
          collision_pair_cache_refresh_interval_) {
    use_cached_candidate_subset = true;
  }

  auto ensure_nearest_points_for_pair = [&](std::size_t idx) {
    if (nearest_points_all_pairs) {
      return;
    }
    if (idx >= collision_data->distanceRequests.size()) {
      return;
    }
    auto &req = collision_data->distanceRequests[idx];
    if (req.enable_nearest_points) {
      return;
    }
    req.enable_nearest_points = true;
    last_collision_exact_distance_queries_++;
    pinocchio::computeDistance(*collision_model, *collision_data, idx);
    req.enable_nearest_points = false;
  };

  std::vector<std::size_t> eval_indices;
  if (use_cached_candidate_subset) {
    std::unordered_set<std::size_t> unique_candidates;
    for (std::size_t idx : collision_cached_candidate_pair_indices_) {
      if (idx < pairs.size()) {
        unique_candidates.insert(idx);
      }
    }
    for (std::size_t idx : last_collision_constraint_pair_indices_) {
      if (idx < pairs.size()) {
        unique_candidates.insert(idx);
      }
    }
    eval_indices.reserve(unique_candidates.size());
    for (std::size_t idx : unique_candidates) {
      eval_indices.push_back(idx);
    }
  } else {
    eval_indices.reserve(pairs.size());
    for (std::size_t idx = 0; idx < pairs.size(); ++idx) {
      eval_indices.push_back(idx);
    }
  }
  if (use_cached_candidate_subset && constraint_active &&
      collision_constraint_.has_value() &&
      collision_constraint_->max_constraints > 1 &&
      !last_collision_constraint_pair_indices_.empty() &&
      eval_indices.size() <
          static_cast<std::size_t>(collision_constraint_->max_constraints)) {
    // Safety fallback: near-contact cached subsets can become underfilled
    // (e.g. only one historical pair left while max_constraints>1), which can
    // miss newly critical pairs for a whole tick.  Only force a full scan
    // when near penetration; in clear space the underfilled cache is harmless
    // and a full scan wastes the tuning mode's latency budget.
    const bool underfill_near_penetration =
        !std::isfinite(last_constraint_min_distance_) ||
        last_collision_budget_exhausted_ ||
        last_constraint_min_distance_ <
            kCollisionPenetrationDistanceThreshold +
                kPostStepSafeMargin;
    if (underfill_near_penetration) {
      use_cached_candidate_subset = false;
      eval_indices.clear();
      eval_indices.reserve(pairs.size());
      for (std::size_t idx = 0; idx < pairs.size(); ++idx) {
        eval_indices.push_back(idx);
      }
    }
  }
  if (use_cached_candidate_subset && eval_indices.empty()) {
    // Safety fallback: an empty cache creates a collision-blind step where no
    // distance queries run. Fall back to a full scan immediately.
    use_cached_candidate_subset = false;
    eval_indices.reserve(pairs.size());
    for (std::size_t idx = 0; idx < pairs.size(); ++idx) {
      eval_indices.push_back(idx);
    }
  }

  const bool budget_enabled = constraint_active &&
                              (collision_refinement_time_budget_us_ > 0) &&
                              use_cached_candidate_subset;
  if (budget_enabled) {
    std::sort(eval_indices.begin(), eval_indices.end(),
              [&](std::size_t a, std::size_t b) {
                const bool a_valid =
                    (a < collision_pair_bound_valid_.size()) &&
                    collision_pair_bound_valid_[a];
                const bool b_valid =
                    (b < collision_pair_bound_valid_.size()) &&
                    collision_pair_bound_valid_[b];
                if (a_valid != b_valid) {
                  return a_valid > b_valid;
                }
                const double da =
                    (a < collision_pair_last_signed_distance_.size())
                        ? collision_pair_last_signed_distance_[a]
                        : std::numeric_limits<double>::infinity();
                const double db =
                    (b < collision_pair_last_signed_distance_.size())
                        ? collision_pair_last_signed_distance_[b]
                        : std::numeric_limits<double>::infinity();
                if (da != db) {
                  return da < db;
                }
                return a < b;
              });
  }
  const auto budget_start = std::chrono::high_resolution_clock::now();

  for (std::size_t idx : eval_indices) {
    if (budget_enabled) {
      const auto elapsed_us = std::chrono::duration_cast<std::chrono::microseconds>(
                                  std::chrono::high_resolution_clock::now() -
                                  budget_start)
                                  .count();
      if (elapsed_us >= collision_refinement_time_budget_us_) {
        last_collision_budget_exhausted_ = true;
        break;
      }
    }

    // Honor Pinocchio's active-pair mask *before* distance computation.
    if (!collision_data->activeCollisionPairs.empty() &&
        !collision_data->activeCollisionPairs[idx]) {
      continue;
    }
    last_collision_pairs_considered_++;

    const auto &pair = pairs[idx];
    const auto &object_a = collision_model->geometryObjects[pair.first];
    const auto &object_b = collision_model->geometryObjects[pair.second];
    const auto frame_a_id = object_a.parentFrame;
    const auto frame_b_id = object_b.parentFrame;
    const auto &transform_a = robot_->data().oMf[frame_a_id];
    const auto &transform_b = robot_->data().oMf[frame_b_id];
    const double rel_translation_norm =
        (transform_a.translation() - transform_b.translation()).norm();
    const Eigen::Matrix3d rel_rotation =
        transform_a.rotation().transpose() * transform_b.rotation();

    const auto braking_path_clears_cutoff = [&](double cutoff) {
      if (!acceleration_braking_lookahead_active) {
        return true;
      }
      if (!braking_lookahead_certified ||
          !sphere_broadphase_enabled_ || !sphere_broadphase_.is_built()) {
        return false;
      }
      const auto joint_a_id = object_a.parentJoint;
      const auto joint_b_id = object_b.parentJoint;
      for (const auto &sample : braking_placement_samples) {
        if (joint_a_id >= static_cast<pinocchio::JointIndex>(
                              sample.joint_translations.size()) ||
            joint_b_id >= static_cast<pinocchio::JointIndex>(
                              sample.joint_translations.size())) {
          return false;
        }
        const double braking_lower_bound =
            sphere_broadphase_.compute_pair_lower_bound(
                pair.first, pair.second,
                sample.joint_translations[joint_a_id],
                sample.joint_rotations[joint_a_id],
                sample.joint_translations[joint_b_id],
                sample.joint_rotations[joint_b_id]);
        if (!std::isfinite(braking_lower_bound) ||
            braking_lower_bound <= cutoff) {
          return false;
        }
      }
      return true;
    };

    bool bound_culled = false;
    if (constraint_active && use_cached_candidate_subset &&
        idx < collision_pair_bound_valid_.size() &&
        idx < collision_pair_last_signed_distance_.size() &&
        idx < collision_pair_last_rel_translation_norm_.size() &&
        idx < collision_pair_last_rel_rotation_.size() &&
        collision_pair_bound_valid_[idx]) {
      const double prev_distance = collision_pair_last_signed_distance_[idx];
      const double prev_rel_translation_norm =
          collision_pair_last_rel_translation_norm_[idx];
      const Eigen::Matrix3d prev_rel_rotation =
          unpack_rotation_matrix(collision_pair_last_rel_rotation_[idx]);
      const double delta_m =
          std::abs(rel_translation_norm - prev_rel_translation_norm);
      const Eigen::Matrix3d delta_rel_rotation =
          prev_rel_rotation.transpose() * rel_rotation;
      const double cos_theta = std::clamp(
          (delta_rel_rotation.trace() - 1.0) * 0.5, -1.0, 1.0);
      const double rotation_penalty = std::sqrt(
          2.0 * kCollisionBoundRotationRadius * kCollisionBoundRotationRadius *
          std::max(0.0, 1.0 - cos_theta));
      const double lower_bound = prev_distance - delta_m - rotation_penalty -
                                 kCollisionBoundSafetyMargin;
      const double candidate_cutoff =
          collision_constraint_.has_value()
              ? collision_constraint_->min_distance +
                    collision_pair_cache_distance_margin_
              : std::numeric_limits<double>::infinity();
      if (std::isfinite(lower_bound) && lower_bound > candidate_cutoff &&
          braking_path_clears_cutoff(
              candidate_cutoff + sphere_broadphase_.safety_margin)) {
        bound_culled = true;
      }
    }
    // Sphere broadphase culling: skip computeDistance() if sphere-sphere
    // lower bound exceeds the activation threshold + safety margin.
    // Safety: AABB sphere strictly contains mesh, so sphere_dist <= true_dist.
    if (!bound_culled && sphere_broadphase_enabled_ &&
        sphere_broadphase_.is_built() && constraint_active) {
      const double candidate_cutoff =
          collision_constraint_.has_value()
              ? collision_constraint_->min_distance +
                    collision_pair_cache_distance_margin_ +
                    sphere_broadphase_.safety_margin
              : std::numeric_limits<double>::infinity();

      const auto joint_a_id = object_a.parentJoint;
      const auto joint_b_id = object_b.parentJoint;
      const auto &oMi = robot_->data().oMi;
      if (joint_a_id < static_cast<pinocchio::JointIndex>(oMi.size()) &&
          joint_b_id < static_cast<pinocchio::JointIndex>(oMi.size())) {
        const double sphere_lb = sphere_broadphase_.compute_pair_lower_bound(
            pair.first, pair.second, oMi[joint_a_id].translation(),
            oMi[joint_a_id].rotation(), oMi[joint_b_id].translation(),
            oMi[joint_b_id].rotation());
        if (std::isfinite(sphere_lb) && sphere_lb > candidate_cutoff &&
            braking_path_clears_cutoff(candidate_cutoff)) {
          last_collision_sphere_culled_pairs_++;
          bound_culled = true;
        }
      }
    }
    if (bound_culled) {
      last_collision_bound_culled_pairs_++;
      continue;
    }

    // Rank candidates with distance-only queries. Nearest points are refined
    // below only for the globally closest debug pair and selected QP rows.
    // Cached subsets can contain dozens of candidates on dense robot models.
    last_collision_exact_distance_queries_++;
    pinocchio::computeDistance(*collision_model, *collision_data, idx);

    const auto &distance_result = collision_data->distanceResults[idx];
    double distance = distance_result.min_distance;
    if (!std::isfinite(distance)) {
      continue;
    }

    if (idx < collision_pair_bound_valid_.size() &&
        idx < collision_pair_last_signed_distance_.size() &&
        idx < collision_pair_last_rel_translation_norm_.size() &&
        idx < collision_pair_last_rel_rotation_.size()) {
      collision_pair_bound_valid_[idx] = 1;
      collision_pair_last_signed_distance_[idx] = distance;
      collision_pair_last_rel_translation_norm_[idx] = rel_translation_norm;
      collision_pair_last_rel_rotation_[idx] = pack_rotation_matrix(rel_rotation);
    }

    if (distance < best_distance_debug) {
      best_distance_debug = distance;
      best_index_debug = idx;
    }

    if (!constraint_active) {
      continue;
    }

    // Only consider include/exclude-allowed pairs for the constraint.
    if (!collision_allowed_pair_mask_.empty() &&
        idx < collision_allowed_pair_mask_.size() &&
        !collision_allowed_pair_mask_[idx]) {
      continue;
    }

    allowed_candidates.emplace_back(distance, idx);
  }
  if (use_cached_candidate_subset) {
    collision_pair_cache_steps_since_refresh_++;
  } else {
    collision_pair_cache_has_full_scan_ = true;
    collision_pair_cache_steps_since_refresh_ = 0;
  }

  // Expose global minimum distance for the post-step rejection fast path.
  last_constraint_min_distance_ = best_distance_debug;
  last_constraint_was_full_scan_ = !use_cached_candidate_subset;

  // Skip collision pairs that no free DOF can move: both links are governed only
  // by locked/excluded joints (zero relative Jacobian). Constraining such a pair
  // is meaningless, can be infeasible if it is penetrating, and otherwise wastes
  // the limited constraint slots that should go to controllable pairs.
  const std::vector<int> &active_locks = active_collision_lock_indices();
  const std::unordered_set<int> frozen_vdofs(active_locks.begin(),
                                             active_locks.end());
  const pinocchio::Model &pin_model = robot_->model();
  auto link_has_free_dof = [&](std::size_t gidx) -> bool {
    const auto parent_joint = collision_model->geometryObjects[gidx].parentJoint;
    for (const auto jid : pin_model.supports[parent_joint]) {
      if (jid == 0) {  // universe / fixed base contributes no DOF
        continue;
      }
      const int vi = pin_model.idx_vs[jid];
      for (int k = 0; k < pin_model.nvs[jid]; ++k) {
        if (frozen_vdofs.find(vi + k) == frozen_vdofs.end()) {
          return true;
        }
      }
    }
    return false;
  };
  auto pair_uncontrollable = [&](std::size_t pidx) -> bool {
    if (frozen_vdofs.empty()) {
      return false;
    }
    return !link_has_free_dof(pairs[pidx].first) && !link_has_free_dof(pairs[pidx].second);
  };

  // ---- Pair selection: top-K with hysteresis ----
  // Sort all allowed candidates by distance (ascending).
  std::sort(allowed_candidates.begin(), allowed_candidates.end());

  // Uncontrollable pairs (both links governed only by frozen DOFs) keep their
  // place in allowed_candidates so they still contribute to the global minimum
  // distance signal (last_constraint_min_distance_, computed above) and to the
  // next-step candidate cache (built below) -- those drive collision caution
  // (adaptive step size, post-step rejection) and must reflect the true
  // geometry. They are excluded only from the constraint *slots* (a meaningless
  // zero-Jacobian row that would starve controllable pairs), via this
  // selectable view.
  std::vector<std::pair<double, std::size_t>> selectable_candidates;
  selectable_candidates.reserve(allowed_candidates.size());
  for (const auto &entry : allowed_candidates) {
    if (!pair_uncontrollable(entry.second)) {
      selectable_candidates.push_back(entry);
    }
  }

  const auto &config = *collision_constraint_;

  const int max_k = config.max_constraints;

  // The nominal row budget is a performance control, not a safety limit.
  // Every controllable penetrating pair and every pair close enough to cross
  // its recovery floor during a row switch must remain in the QP. Reuse the
  // pair-switch hysteresis as a selection-only guard band; this does not alter
  // the continuous velocity-damper bounds.
  const double recovery_selection_margin = kCollisionPairSwitchHysteresis;
  const bool proximity_gating_active =
      config.constraint_activation_enabled &&
      config.constraint_activation_margin > 0.0;
  std::unordered_set<std::size_t> mandatory_pair_indices;
  for (const auto &[distance, pair_idx] : selectable_candidates) {
    const auto recovery =
        resolve_collision_recovery_distances(pair_idx, distance);
    const double recovery_target = recovery.recovery_target;
    const bool command_floor_active =
        !acceleration_limits_enabled_ && position_step_call_depth_ > 0 &&
        pair_idx < position_step_collision_command_floor_distances_.size() &&
        std::isfinite(
            position_step_collision_command_floor_distances_[pair_idx]);
    if ((!proximity_gating_active && distance < recovery_target) ||
        ((non_worsening_collision_floor_enabled_ || command_floor_active) &&
         distance <= recovery_target + recovery_selection_margin)) {
      mandatory_pair_indices.insert(pair_idx);
    }
  }

  // Determine the distance threshold below which a previous pair "sticks".
  // A previous pair is kept if its distance is within hysteresis of the
  // current K-th best candidate.
  double hysteresis_cutoff = std::numeric_limits<double>::infinity();
  if (!selectable_candidates.empty()) {
    const std::size_t kth = static_cast<std::size_t>(
        std::min(max_k, static_cast<int>(selectable_candidates.size())) - 1);
    hysteresis_cutoff = selectable_candidates[kth].first +
                        kCollisionPairSwitchHysteresis;
  }

  // Build the selected set: start with top-K candidates, then admit any
  // previous pairs that fall within hysteresis_cutoff.
  std::unordered_set<std::size_t> selected_set;
  for (int i = 0;
       i < max_k && i < static_cast<int>(selectable_candidates.size()); ++i) {
    selected_set.insert(selectable_candidates[i].second);
  }
  for (std::size_t prev_idx : last_collision_constraint_pair_indices_) {
    if (static_cast<int>(selected_set.size()) >= max_k) {
      break;
    }
    if (prev_idx >= pairs.size()) {
      continue;
    }
    if (pair_uncontrollable(prev_idx)) {
      continue;
    }
    if (!collision_data->activeCollisionPairs.empty() &&
        !collision_data->activeCollisionPairs[prev_idx]) {
      continue;
    }
    if (!collision_allowed_pair_mask_.empty() &&
        prev_idx < collision_allowed_pair_mask_.size() &&
        !collision_allowed_pair_mask_[prev_idx]) {
      continue;
    }
    const double prev_dist =
        collision_data->distanceResults[prev_idx].min_distance;
    if (std::isfinite(prev_dist) && prev_dist <= hysteresis_cutoff) {
      selected_set.insert(prev_idx);
    }
  }
  selected_set.insert(mandatory_pair_indices.begin(),
                      mandatory_pair_indices.end());

  // Sort the selected set by distance to produce a deterministic order and
  // retain all mandatory rows even when they exceed the nominal budget.
  std::vector<std::pair<double, std::size_t>> selected_sorted;
  selected_sorted.reserve(selected_set.size());
  for (std::size_t idx : selected_set) {
    selected_sorted.emplace_back(
        collision_data->distanceResults[idx].min_distance, idx);
  }
  std::sort(selected_sorted.begin(), selected_sorted.end());

  const std::size_t selected_row_budget = std::max(
      static_cast<std::size_t>(max_k), mandatory_pair_indices.size());
  if (selected_sorted.size() > selected_row_budget) {
    std::vector<std::pair<double, std::size_t>> retained;
    retained.reserve(selected_row_budget);
    for (const auto &entry : selected_sorted) {
      if (mandatory_pair_indices.count(entry.second) != 0) {
        retained.push_back(entry);
      }
    }
    for (const auto &entry : selected_sorted) {
      if (retained.size() >= selected_row_budget) {
        break;
      }
      if (mandatory_pair_indices.count(entry.second) == 0) {
        retained.push_back(entry);
      }
    }
    std::sort(retained.begin(), retained.end());
    selected_sorted = std::move(retained);
  }

  // Optional proximity-gated row activation.
  // When disabled (margin <= 0), behavior is unchanged.
  if (config.constraint_activation_enabled &&
      config.constraint_activation_margin > 0.0) {
    const double activation_threshold =
        config.min_distance + config.constraint_activation_margin;
    selected_sorted.erase(
        std::remove_if(
            selected_sorted.begin(), selected_sorted.end(),
            [activation_threshold, &mandatory_pair_indices](
                const std::pair<double, std::size_t> &entry) {
              if (mandatory_pair_indices.count(entry.second) != 0) {
                return false;
              }
              return entry.first > activation_threshold;
            }),
        selected_sorted.end());
  }

  // Update the per-step active indices for next step's hysteresis.
  last_collision_constraint_pair_indices_.clear();
  for (const auto &[dist, idx] : selected_sorted) {
    last_collision_constraint_pair_indices_.push_back(idx);
  }

  // Update conservative candidate cache for next step.
  collision_cached_candidate_pair_indices_.clear();
  if (constraint_active) {
    const auto &config = *collision_constraint_;
    const double candidate_cutoff =
        config.min_distance + collision_pair_cache_distance_margin_;
    for (const auto &[dist, idx] : allowed_candidates) {
      if (dist <= candidate_cutoff) {
        collision_cached_candidate_pair_indices_.push_back(idx);
        if (static_cast<int>(collision_cached_candidate_pair_indices_.size()) >=
            collision_pair_cache_max_candidates_) {
          break;
        }
      }
    }
    std::unordered_set<std::size_t> candidate_set(
        collision_cached_candidate_pair_indices_.begin(),
        collision_cached_candidate_pair_indices_.end());
    for (std::size_t idx : last_collision_constraint_pair_indices_) {
      if (static_cast<int>(collision_cached_candidate_pair_indices_.size()) >=
          collision_pair_cache_max_candidates_) {
        break;
      }
      if (candidate_set.insert(idx).second) {
        collision_cached_candidate_pair_indices_.push_back(idx);
      }
    }
  }

  // Clean up stuck-detection state for pairs no longer active.
  {
    std::unordered_set<std::size_t> active_set(
        last_collision_constraint_pair_indices_.begin(),
        last_collision_constraint_pair_indices_.end());
    for (auto it = collision_stuck_counters_.begin();
         it != collision_stuck_counters_.end();) {
      if (active_set.count(it->first) == 0) {
        collision_stuck_last_distances_.erase(it->first);
        it = collision_stuck_counters_.erase(it);
      } else {
        ++it;
      }
    }
  }

  // ---- Debug info ----
  // Always report the globally closest pair so that evaluate_collision_debug()
  // returns the true minimum distance even when the pair is not in the active
  // constraint set (top-K).
  std::optional<std::size_t> debug_index_to_use = best_index_debug;

  if (debug_index_to_use.has_value()) {
    ensure_nearest_points_for_pair(*debug_index_to_use);
    const auto &pair_debug = pairs[*debug_index_to_use];
    const auto &obj_da = collision_model->geometryObjects[pair_debug.first];
    const auto &obj_db = collision_model->geometryObjects[pair_debug.second];
    const auto &res_debug =
        collision_data->distanceResults[*debug_index_to_use];
    CollisionDebugInfo debug_info;
    debug_info.object_a = obj_da.name;
    debug_info.object_b = obj_db.name;
    debug_info.distance = res_debug.min_distance;
    debug_info.point_a_world = res_debug.nearest_points[0].cast<double>();
    debug_info.point_b_world = res_debug.nearest_points[1].cast<double>();
    last_collision_debug_ = debug_info;
  } else {
    last_collision_debug_.reset();
  }

  if (!constraint_active || selected_sorted.empty()) {
    return std::nullopt;
  }

  const int nv = robot_->nv();
  const int num_selected = static_cast<int>(selected_sorted.size());

  CollisionConstraintResult result;
  result.jacobian.resize(num_selected, nv);
  result.lower_bounds.resize(num_selected);
  result.upper_bounds.resize(num_selected);

  // Per-pair velocity-damper bound computation (with continuous recovery ramp).
  auto compute_bounds_for_pair = [&](std::size_t pair_idx,
                                     double signed_distance,
                                     bool *stuck_out,
                                     double *recovery_margin_out)
      -> std::pair<double, double> {
    const auto recovery =
        resolve_collision_recovery_distances(pair_idx, signed_distance);
    const double effective_min_distance = recovery.effective_min_distance;
    const double recovery_target = recovery.recovery_target;
    if (recovery_margin_out != nullptr) {
      *recovery_margin_out = signed_distance - recovery_target;
    }

    // Detect stuck: non-penetrating but significantly inside min_distance for
    // multiple consecutive cycles AND the previous dq was near-zero.
    bool stuck_active = false;
    const bool deep_non_penetration =
        (signed_distance >= 0.0) &&
        (signed_distance < (recovery_target - kCollisionStuckBand));
    const bool dq_small = (last_solution_dq_norm_ < kCollisionStuckDqNormEps);
    if (deep_non_penetration && dq_small) {
      int &counter = collision_stuck_counters_[pair_idx];
      double &last_dist = collision_stuck_last_distances_[pair_idx];
      const bool not_improving =
          std::isfinite(last_dist) && (signed_distance <= last_dist + 1e-6);
      counter = not_improving ? counter + 1 : 1;
      last_dist = signed_distance;
      stuck_active = (counter >= kCollisionStuckCountThreshold);
    } else {
      collision_stuck_counters_.erase(pair_idx);
      collision_stuck_last_distances_.erase(pair_idx);
    }
    if (stuck_out != nullptr) {
      *stuck_out = stuck_active;
    }

    // Velocity-damper lower bound:
    // - outside deadband: allow approach up to the damper limit (negative lb)
    // - inside deadband (min <= d < min+deadband): no-approach (lb=0)
    // - violated (d < min): continuous recovery ramp without discrete tiers
    // Note: collision_repulsion_deadband_ defaults to kCollisionRepulsionDeadband
    // (3mm) but is tunable at runtime via set_collision_repulsion_deadband().
    // Setting it to 0 removes the discontinuity that causes boundary oscillation.
    const double repulsion_deadband  = collision_repulsion_deadband_;
    const double recovery_scale      = collision_recovery_scale_;
    const double max_sep_speed_nonpen = collision_max_sep_speed_nonpen_;
    double lower_bound = 0.0;
    if (signed_distance >= (recovery_target + repulsion_deadband)) {
      lower_bound =
          (recovery_target + config.tolerance - signed_distance) /
          constraint_dt;
    } else if (signed_distance >= recovery_target) {
      lower_bound = 0.0;
    } else {
      // Violated region: uniform continuous recovery ramp from the moment
      // signed_distance drops below the (non-worsening) recovery target. No
      // dead-zone — recovery force is active at all violation depths.
      const double desired =
          (recovery_target + config.tolerance - signed_distance) /
          constraint_dt;
      if (signed_distance >= 0.0) {
        // Non-penetrating: proportional recovery, capped.
        lower_bound = std::min(max_sep_speed_nonpen,
                               std::max(0.0, desired * recovery_scale));
      } else {
        // Penetrating: enforce a minimum recovery speed, still capped.
        lower_bound = std::min(kCollisionMaxSeparationSpeed, desired);
        lower_bound = std::max(lower_bound, kCollisionMinRecoverySpeed);
      }
    }

    // Stuck override: ensure a floor that can actually produce motion.
    if (stuck_active && signed_distance >= 0.0) {
      const double desired =
          (effective_min_distance + config.tolerance - signed_distance) /
          constraint_dt;
      lower_bound = std::max(
          lower_bound,
          std::min(max_sep_speed_nonpen,
                   std::max(kCollisionStuckRecoverySpeed,
                            desired * recovery_scale)));
    }

    const double upper_bound =
        (config.upper_distance - config.tolerance + signed_distance) /
        constraint_dt;
    return {lower_bound, upper_bound};
  };

  // Build one constraint row per selected pair.
  for (int row = 0; row < num_selected; ++row) {
    const std::size_t pair_idx = selected_sorted[row].second;
    ensure_nearest_points_for_pair(pair_idx);

    const auto &pair = pairs[pair_idx];
    const auto &object_a = collision_model->geometryObjects[pair.first];
    const auto &object_b = collision_model->geometryObjects[pair.second];
    const auto &distance_result = collision_data->distanceResults[pair_idx];
    const double signed_distance = distance_result.min_distance;

    const Eigen::Vector3d p1_world =
        distance_result.nearest_points[0].cast<double>();
    const Eigen::Vector3d p2_world =
        distance_result.nearest_points[1].cast<double>();

    const Eigen::Vector3d distance_vector = p1_world - p2_world;
    const double distance_norm = distance_vector.norm();
    Eigen::Vector3d normal = Eigen::Vector3d::UnitX();
    if (distance_norm > kDistanceEpsilon) {
      normal = distance_vector / distance_norm;
    }

    const auto frame_a_id = object_a.parentFrame;
    const auto frame_b_id = object_b.parentFrame;
    const auto &frame_a = robot_->model().frames[frame_a_id];
    const auto &frame_b = robot_->model().frames[frame_b_id];
    const auto &transform_a = robot_->data().oMf[frame_a_id];
    const auto &transform_b = robot_->data().oMf[frame_b_id];

    const Eigen::Vector3d p1_local =
        transform_a.rotation().transpose() *
        (p1_world - transform_a.translation());
    const Eigen::Vector3d p2_local =
        transform_b.rotation().transpose() *
        (p2_world - transform_b.translation());

    const Eigen::Matrix<double, 3, Eigen::Dynamic> jacobian_a =
        robot_->get_point_jacobian(frame_a.name, p1_local);
    const Eigen::Matrix<double, 3, Eigen::Dynamic> jacobian_b =
        robot_->get_point_jacobian(frame_b.name, p2_local);

    Eigen::Matrix3d rotation_to_x = Eigen::Matrix3d::Identity();
    Eigen::Matrix3d rotation_from_negative = Eigen::Matrix3d::Identity();
    if (distance_norm > kDistanceEpsilon) {
      rotation_to_x =
          Eigen::Quaterniond::FromTwoVectors(normal, Eigen::Vector3d::UnitX())
              .toRotationMatrix();
      rotation_from_negative =
          Eigen::Quaterniond::FromTwoVectors(-normal, Eigen::Vector3d::UnitX())
              .toRotationMatrix();
    }

    // Relative separating velocity constraint: normalᵀ (v_a - v_b) >= lb
    result.jacobian.row(row) =
        (rotation_to_x * jacobian_a).row(0) +
        (rotation_from_negative * jacobian_b).row(0);

    double recovery_margin = std::numeric_limits<double>::infinity();
    auto [lb, ub] = compute_bounds_for_pair(
        pair_idx, signed_distance, nullptr, &recovery_margin);
    if (acceleration_limits_enabled_ &&
        acceleration_limits_.size() == nv &&
        std::isfinite(recovery_margin) && recovery_margin > 0.0) {
      double separating_acceleration_limit = 0.0;
      const std::vector<int> locked_indices =
          active_collision_lock_indices();
      for (int index = 0; index < nv; ++index) {
        if (std::find(locked_indices.begin(), locked_indices.end(), index) !=
            locked_indices.end()) {
          continue;
        }
        separating_acceleration_limit = std::max(
            separating_acceleration_limit,
            std::abs(result.jacobian(row, index)) *
                acceleration_limits_[index]);
      }
      const double braking_slack =
          std::max(0.0, recovery_margin - config.tolerance);
      const double braking_speed = std::max(
          0.0,
          std::sqrt(2.0 * separating_acceleration_limit * braking_slack) -
              separating_acceleration_limit * constraint_dt);
      const double braking_lower_bound = -braking_speed;
      lb = std::max(lb, braking_lower_bound);
    }
    if (std::isfinite(recovery_margin)) {
      last_constraint_min_recovery_margin_ =
          std::min(last_constraint_min_recovery_margin_, recovery_margin);
    }
    result.lower_bounds(row) = lb;
    result.upper_bounds(row) = ub;

    // Populate per-pair debug info.
    CollisionDebugInfo pair_debug;
    pair_debug.object_a = object_a.name;
    pair_debug.object_b = object_b.name;
    pair_debug.distance = signed_distance;
    pair_debug.point_a_world = p1_world;
    pair_debug.point_b_world = p2_world;
    last_collision_debug_list_.push_back(pair_debug);

    // Keep the result fields for the closest pair (backward compat).
    if (row == 0) {
      result.distance = signed_distance;
      result.object_a = object_a.name;
      result.object_b = object_b.name;
      result.point_a_world = p1_world;
      result.point_b_world = p2_world;
    }
  }

  // Cache the result and configuration for lazy reuse.
  last_collision_constraint_result_ = result;
  last_collision_constraint_row_dt_ = constraint_dt;
  last_collision_constraint_q_ = robot_->get_current_configuration();
  collision_cache_frozen_indices_ = active_collision_lock_indices();

  return result;
#else
  last_collision_debug_.reset();
  last_collision_debug_list_.clear();
  return std::nullopt;
#endif
}

std::optional<KinematicsSolver::CollisionVelocityConstraintLinearization>
KinematicsSolver::linearize_collision_velocity_constraint(double row_dt) {
  auto rows = compute_collision_constraint(row_dt);
  if (!rows.has_value()) {
    return std::nullopt;
  }

  CollisionVelocityConstraintLinearization linearization;
  linearization.coefficient_matrix = rows->jacobian;
  linearization.lower_bounds = rows->lower_bounds;
  linearization.upper_bounds = rows->upper_bounds;
  linearization.dt = row_dt;
  linearization.distance = rows->distance;
  linearization.object_a = rows->object_a;
  linearization.object_b = rows->object_b;
  linearization.point_a_world = rows->point_a_world;
  linearization.point_b_world = rows->point_b_world;
  return linearization;
}

void KinematicsSolver::invalidate_collision_validation_state_certificates() {
  collision_validation_seed_certificate_.reset();
}

void KinematicsSolver::invalidate_collision_validation_cache() {
  collision_validation_catalog_.clear();
  collision_validation_effective_pair_mask_.clear();
  collision_validation_geometry_model_ = nullptr;
  collision_validation_geometry_object_count_ = 0;
  collision_validation_pair_count_ = 0;
  collision_validation_data_.reset();
  collision_validation_geometry_data_.reset();
  ++collision_validation_catalog_revision_;
  invalidate_collision_validation_state_certificates();
}

KinematicsSolver::CollisionSampleValidationResult
KinematicsSolver::validate_collision_samples(
    const Eigen::VectorXd &q_from,
    const std::vector<Eigen::VectorXd> &q_samples) {
  CollisionSampleValidationResult result;
  const auto fail = [&](SolverStatus status, std::string message) {
    result.status = status;
    result.message = std::move(message);
    result.acceptable = false;
    return result;
  };

  if (q_samples.empty()) {
    return fail(SolverStatus::kInvalidInput,
                "collision sample sequence must not be empty");
  }
  if (q_from.size() != robot_->nq()) {
    return fail(SolverStatus::kShapeMismatch,
                "collision sample seed must have size nq");
  }
  if (!q_from.allFinite()) {
    return fail(SolverStatus::kNonFiniteInput,
                "collision sample seed must be finite");
  }

  if (!collision_constraint_.has_value() ||
      !collision_constraint_->enabled) {
    return fail(SolverStatus::kInvalidInput,
                "collision sample validation requires configured collision "
                "pairs");
  }

#ifdef PINOCCHIO_WITH_HPP_FCL
  const auto *collision_model =
      static_cast<const RobotModel &>(*robot_).collision_model();
  const auto *shared_collision_data =
      static_cast<const RobotModel &>(*robot_).collision_data();
  if (collision_model == nullptr || shared_collision_data == nullptr ||
      !robot_->has_collision_geometry()) {
    return fail(SolverStatus::kInvalidInput,
                "collision sample validation requires collision geometry");
  }
  const bool geometry_trackable =
      static_cast<const RobotModel &>(*robot_)
          .collision_geometry_provenance_trackable();
  const std::uint64_t geometry_revision =
      static_cast<const RobotModel &>(*robot_).collision_geometry_revision();

  const auto &pairs = collision_model->collisionPairs;
  if (pairs.empty() ||
      collision_allowed_pair_mask_.size() != pairs.size() ||
      shared_collision_data->activeCollisionPairs.size() != pairs.size()) {
    return fail(SolverStatus::kInvalidInput,
                "collision sample validation topology or masks are "
                "inconsistent");
  }

  std::vector<std::uint8_t> effective_pair_mask(pairs.size(), 0U);
  for (std::size_t pair_index = 0; pair_index < pairs.size(); ++pair_index) {
    effective_pair_mask[pair_index] =
        collision_allowed_pair_mask_[pair_index] != 0U &&
                shared_collision_data->activeCollisionPairs[pair_index]
            ? 1U
            : 0U;
  }
  const bool catalog_stale =
      collision_validation_geometry_model_ != collision_model ||
      collision_validation_geometry_object_count_ !=
          collision_model->geometryObjects.size() ||
      collision_validation_pair_count_ != pairs.size() ||
      collision_validation_effective_pair_mask_ != effective_pair_mask;
  if (catalog_stale) {
    invalidate_collision_validation_cache();
  }
  if (collision_validation_catalog_.empty()) {
    collision_validation_geometry_model_ = collision_model;
    collision_validation_geometry_object_count_ =
        collision_model->geometryObjects.size();
    collision_validation_pair_count_ = pairs.size();
    collision_validation_effective_pair_mask_ = effective_pair_mask;
    collision_validation_catalog_.reserve(pairs.size());
    for (std::size_t pair_index = 0; pair_index < pairs.size();
         ++pair_index) {
      if (effective_pair_mask[pair_index] == 0U) {
        continue;
      }
      const auto &pair = pairs[pair_index];
      const auto &object_a = collision_model->geometryObjects[pair.first];
      const auto &object_b = collision_model->geometryObjects[pair.second];
      CollisionValidationPairMetadata metadata;
      metadata.pair_index = pair_index;
      metadata.object_a = pair.first;
      metadata.object_b = pair.second;
      metadata.key = canonical_pair_key(object_a.name, object_b.name);
      collision_validation_catalog_.push_back(std::move(metadata));
    }
  }
  if (collision_validation_catalog_.empty()) {
    return fail(SolverStatus::kInvalidInput,
                "collision sample validation found no allowed active pairs");
  }
  result.allowed_pair_count = collision_validation_catalog_.size();

  if (!collision_validation_data_) {
    collision_validation_data_ =
        std::make_unique<pinocchio::Data>(robot_->model());
  }
  if (!collision_validation_geometry_data_) {
    collision_validation_geometry_data_ =
        std::make_unique<pinocchio::GeometryData>(*collision_model);
  }
  if (collision_validation_geometry_data_->distanceRequests.size() !=
          pairs.size() ||
      collision_validation_geometry_data_->distanceResults.size() !=
          pairs.size()) {
    return fail(SolverStatus::kInvalidInput,
                "collision sample validation workspace topology is "
                "inconsistent");
  }
  for (auto &request : collision_validation_geometry_data_->distanceRequests) {
    request.enable_signed_distance = true;
    request.enable_nearest_points = false;
  }

  const auto pair_sphere_lower_bound =
      [&](const CollisionValidationPairMetadata &metadata)
      -> std::optional<double> {
    if (!sphere_broadphase_enabled_ || !sphere_broadphase_.is_built()) {
      return std::nullopt;
    }
    const auto &object_a = collision_model->geometryObjects[metadata.object_a];
    const auto &object_b = collision_model->geometryObjects[metadata.object_b];
    const auto joint_a_id = object_a.parentJoint;
    const auto joint_b_id = object_b.parentJoint;
    if (joint_a_id >=
            static_cast<pinocchio::JointIndex>(
                collision_validation_data_->oMi.size()) ||
        joint_b_id >=
            static_cast<pinocchio::JointIndex>(
                collision_validation_data_->oMi.size())) {
      return std::nullopt;
    }
    const double lower_bound = sphere_broadphase_.compute_pair_lower_bound(
        metadata.object_a, metadata.object_b,
        collision_validation_data_->oMi[joint_a_id].translation(),
        collision_validation_data_->oMi[joint_a_id].rotation(),
        collision_validation_data_->oMi[joint_b_id].translation(),
        collision_validation_data_->oMi[joint_b_id].rotation());
    if (!std::isfinite(lower_bound)) {
      return std::nullopt;
    }
    return lower_bound;
  };

  const auto update_validation_geometry =
      [&](const Eigen::VectorXd &sample_q) -> std::optional<std::string> {
    try {
      pinocchio::forwardKinematics(robot_->model(),
                                   *collision_validation_data_, sample_q);
      ++result.kinematics_updates;
      pinocchio::updateGeometryPlacements(
          robot_->model(), *collision_validation_data_, *collision_model,
          *collision_validation_geometry_data_);
      ++result.geometry_updates;
    } catch (const std::exception &error) {
      return std::string("collision sample geometry update failed: ") +
             error.what();
    }
    return std::nullopt;
  };

  const auto query_exact_distance =
      [&](const CollisionValidationPairMetadata &metadata,
          bool initial_query, double *distance_out)
      -> std::optional<std::string> {
    if (metadata.pair_index >= pairs.size() ||
        collision_allowed_pair_mask_[metadata.pair_index] == 0U ||
        !shared_collision_data->activeCollisionPairs[metadata.pair_index]) {
      return "collision pair mask changed during sample validation";
    }
    try {
      pinocchio::computeDistance(*collision_model,
                                 *collision_validation_geometry_data_,
                                 metadata.pair_index);
    } catch (const std::exception &error) {
      return std::string("collision distance query failed: ") + error.what();
    }
    ++result.exact_distance_queries;
    if (initial_query) {
      ++result.initial_exact_distance_queries;
    } else {
      ++result.sample_exact_distance_queries;
    }
    const double distance =
        collision_validation_geometry_data_->distanceResults
            [metadata.pair_index]
                .min_distance;
    if (!std::isfinite(distance)) {
      return "collision distance evidence was non-finite";
    }
    *distance_out = distance;
    return std::nullopt;
  };

  std::vector<double> seed_distance_lower_bounds(
      collision_validation_catalog_.size(),
      std::numeric_limits<double>::quiet_NaN());
  const bool certificate_matches =
      geometry_trackable && collision_validation_seed_certificate_.has_value() &&
      collision_validation_seed_certificate_->geometry_model ==
          collision_model &&
      collision_validation_seed_certificate_->geometry_revision ==
          geometry_revision &&
      collision_validation_seed_certificate_->catalog_revision ==
          collision_validation_catalog_revision_ &&
      collision_validation_seed_certificate_->policy_revision ==
          collision_validation_policy_revision_ &&
      collision_validation_seed_certificate_->q.size() == q_from.size() &&
      collision_validation_seed_certificate_->q.isApprox(q_from, 0.0) &&
      collision_validation_seed_certificate_->distance_lower_bounds.size() ==
          collision_validation_catalog_.size();
  if (certificate_matches) {
    seed_distance_lower_bounds =
        collision_validation_seed_certificate_->distance_lower_bounds;
    result.initial_state_certificate_reused = true;
  } else if (const auto error = update_validation_geometry(q_from);
             error.has_value()) {
    return fail(SolverStatus::kNumericalError, *error);
  }

  std::vector<double> effective_floors(collision_validation_catalog_.size());
  for (std::size_t cursor = 0; cursor < collision_validation_catalog_.size();
       ++cursor) {
    if (!result.initial_state_certificate_reused) {
      double seed_distance = 0.0;
      if (const auto error = query_exact_distance(
              collision_validation_catalog_[cursor], true, &seed_distance);
          error.has_value()) {
        return fail(SolverStatus::kNumericalError, *error);
      }
      seed_distance_lower_bounds[cursor] = seed_distance;
    }
    const auto policy_revision_before = collision_validation_policy_revision_;
    const auto recovery = resolve_collision_recovery_distances(
        collision_validation_catalog_[cursor].pair_index,
        seed_distance_lower_bounds[cursor]);
    if (collision_validation_policy_revision_ != policy_revision_before) {
      result.initial_state_certificate_reused = false;
    }
    if (!std::isfinite(recovery.recovery_target)) {
      return fail(SolverStatus::kNumericalError,
                  "collision effective floor was non-finite");
    }
    effective_floors[cursor] = recovery.recovery_target;
  }

  constexpr double kSampleNonWorseningTolerance = 1e-8;
  if (geometry_trackable && !result.initial_state_certificate_reused) {
    CollisionStateSafetyCertificate certificate;
    certificate.q = q_from;
    certificate.distance_lower_bounds = seed_distance_lower_bounds;
    certificate.geometry_model = collision_model;
    certificate.geometry_revision = geometry_revision;
    certificate.catalog_revision = collision_validation_catalog_revision_;
    certificate.policy_revision = collision_validation_policy_revision_;
    collision_validation_seed_certificate_ = std::move(certificate);
  } else if (!geometry_trackable) {
    invalidate_collision_validation_state_certificates();
  }

  result.samples_checked = 0;
  for (std::size_t sample_index = 0; sample_index < q_samples.size();
       ++sample_index) {
    const auto &sample_q = q_samples[sample_index];
    if (sample_q.size() != robot_->nq()) {
      result.failed_sample_index = sample_index + 1;
      return fail(SolverStatus::kShapeMismatch,
                  "collision samples must have size nq");
    }
    if (!sample_q.allFinite()) {
      result.failed_sample_index = sample_index + 1;
      return fail(SolverStatus::kNonFiniteInput,
                  "collision samples must be finite");
    }
    if (const auto error = update_validation_geometry(sample_q);
        error.has_value()) {
      return fail(SolverStatus::kNumericalError, *error);
    }
    ++result.samples_checked;
    for (std::size_t cursor = 0; cursor < collision_validation_catalog_.size();
         ++cursor) {
      ++result.pairs_checked;
      const auto &metadata = collision_validation_catalog_[cursor];
      const double seed_distance = seed_distance_lower_bounds[cursor];
      const double floor = effective_floors[cursor];
      double sample_distance = std::numeric_limits<double>::quiet_NaN();
      bool sample_certified = false;
      ++result.conservative_bound_checks;
      if (const auto lower_bound = pair_sphere_lower_bound(metadata);
          lower_bound.has_value()) {
        sample_distance = *lower_bound;
        if (seed_distance >= floor - kSampleNonWorseningTolerance &&
            sample_distance >= floor - kSampleNonWorseningTolerance) {
          sample_certified = true;
        } else if (
            seed_distance < floor - kSampleNonWorseningTolerance &&
            sample_distance >=
                seed_distance - kSampleNonWorseningTolerance) {
          sample_certified = true;
        }
      }
      if (sample_certified) {
        ++result.conservative_bound_certified_pairs;
      } else if (const auto error =
                     query_exact_distance(metadata, false, &sample_distance);
                 error.has_value()) {
        return fail(SolverStatus::kNumericalError, *error);
      }
      if (seed_distance >= floor - kSampleNonWorseningTolerance) {
        if (sample_distance < floor - kSampleNonWorseningTolerance) {
          result.failed_sample_index = sample_index + 1;
          result.failed_pair_catalog_index = cursor;
          result.failed_pair_model_index = metadata.pair_index;
          result.failed_pair_key = metadata.key;
          return fail(
              SolverStatus::kCollisionViolated,
              "collision sample violated an effective collision floor");
        }
      } else if (sample_distance <
                 seed_distance - kSampleNonWorseningTolerance) {
        result.failed_sample_index = sample_index + 1;
        result.failed_pair_catalog_index = cursor;
        result.failed_pair_model_index = metadata.pair_index;
        result.failed_pair_key = metadata.key;
        return fail(
            SolverStatus::kCollisionViolated,
            "collision sample worsened an initially violated pair");
      }
    }
  }

  result.status = SolverStatus::kSuccess;
  result.acceptable = true;
  return result;
#else
  return fail(SolverStatus::kInvalidInput,
              "collision sample validation requires Pinocchio collision "
              "support");
#endif
}


} // namespace embodik
