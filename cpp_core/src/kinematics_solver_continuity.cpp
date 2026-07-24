/**
 * @file kinematics_solver_continuity.cpp
 * @brief Continuity, stall, and elastic-band state for KinematicsSolver
 */

#include "kinematics_solver_internal.hpp"

namespace embodik {

bool KinematicsSolver::position_step_target_geometry_changed(
    const PositionStepTargetSignature &signature) const {
  if (!last_position_step_target_signature_.has_value()) {
    return true;
  }
  const auto &previous = *last_position_step_target_signature_;
  if (previous.task_names != signature.task_names) {
    return true;
  }

  const bool use_reference_geometry =
      !previous.reference_target_poses.empty() &&
      !signature.reference_target_poses.empty();
  const auto &before = use_reference_geometry
                           ? previous.reference_target_poses
                           : previous.target_poses;
  const auto &after = use_reference_geometry
                          ? signature.reference_target_poses
                          : signature.target_poses;
  if (before.size() != after.size()) {
    return true;
  }
  for (std::size_t index = 0; index < after.size(); ++index) {
    const double translation_delta =
        (before[index].block<3, 1>(0, 3) -
         after[index].block<3, 1>(0, 3))
            .norm();
    const Eigen::Matrix3d rotation_delta =
        before[index].block<3, 3>(0, 0).transpose() *
        after[index].block<3, 3>(0, 0);
    const double rotation_angle = pinocchio::log3(rotation_delta).norm();
    if (!std::isfinite(translation_delta) ||
        !std::isfinite(rotation_angle) ||
        translation_delta > kStationaryTargetTranslationTolerance ||
        rotation_angle > kStationaryTargetRotationTolerance) {
      return true;
    }
  }
  return false;
}

void KinematicsSolver::update_position_step_target_motion_blocks(
    const PositionStepTargetSignature &signature) {
  position_step_target_motion_blocks_.clear();
  if (!last_position_step_target_signature_.has_value()) {
    return;
  }

  const auto &previous = *last_position_step_target_signature_;
  if (previous.task_names != signature.task_names) {
    return;
  }
  const bool use_reference_geometry =
      !previous.reference_target_poses.empty() &&
      previous.reference_target_poses.size() ==
          signature.reference_target_poses.size();
  const auto &before = use_reference_geometry
                           ? previous.reference_target_poses
                           : previous.target_poses;
  const auto &after = use_reference_geometry
                          ? signature.reference_target_poses
                          : signature.target_poses;
  if (before.size() != after.size() ||
      after.size() > signature.task_names.size()) {
    return;
  }

  constexpr char kSecondarySuffix[] = "#secondary";
  constexpr std::size_t kSecondarySuffixLength = sizeof(kSecondarySuffix) - 1;
  for (std::size_t index = 0; index < after.size(); ++index) {
    std::string task_name = signature.task_names[index];
    if (task_name.size() >= kSecondarySuffixLength &&
        task_name.compare(task_name.size() - kSecondarySuffixLength,
                          kSecondarySuffixLength, kSecondarySuffix) == 0) {
      task_name.resize(task_name.size() - kSecondarySuffixLength);
    }

    const double translation_delta =
        (before[index].block<3, 1>(0, 3) -
         after[index].block<3, 1>(0, 3))
            .norm();
    const Eigen::Matrix3d rotation_delta =
        before[index].block<3, 3>(0, 0).transpose() *
        after[index].block<3, 3>(0, 0);
    const double rotation_angle = pinocchio::log3(rotation_delta).norm();
    auto [motion_iter, inserted] = position_step_target_motion_blocks_.try_emplace(
        task_name, std::array<double, 2>{0.0, 0.0});
    (void)inserted;
    auto &motion_blocks = motion_iter->second;
    motion_blocks[0] =
        !std::isfinite(translation_delta)
            ? std::numeric_limits<double>::infinity()
            : std::max(motion_blocks[0], translation_delta);
    motion_blocks[1] =
        !std::isfinite(rotation_angle)
            ? std::numeric_limits<double>::infinity()
            : std::max(motion_blocks[1], rotation_angle);
  }
}

double KinematicsSolver::position_step_task_terminal_prediction_weight(
    const Task &task, const Eigen::VectorXd &current_error) const {
  if (task.getType() == TaskType::FRAME_ORIENTATION) {
    return 0.0;
  }
  const auto motion_iter =
      position_step_target_motion_blocks_.find(task.getName());
  if (motion_iter != position_step_target_motion_blocks_.end()) {
    const double target_translation_step = motion_iter->second[0];
    if (!std::isfinite(target_translation_step)) {
      return 1.0;
    }
    if (target_translation_step <= 0.0) {
      return 0.0;
    }
    if (target_translation_step <= kStationaryTargetTranslationTolerance) {
      return 0.0;
    }
    if (target_translation_step >=
        kTargetTranslationPredictionTransition) {
      return 1.0;
    }
    const double normalized =
        (target_translation_step - kStationaryTargetTranslationTolerance) /
        (kTargetTranslationPredictionTransition -
         kStationaryTargetTranslationTolerance);
    return normalized * normalized * (3.0 - 2.0 * normalized);
  }
  if (task.getType() == TaskType::FRAME_POSE && current_error.size() >= 6) {
    return current_error.head<3>().squaredNorm() > kCollisionEscapeNormEps
               ? 1.0
               : 0.0;
  }
  return 1.0;
}

bool KinematicsSolver::update_position_step_target_signature(
    PositionStepTargetSignature signature) {
  const bool signature_is_finite =
      std::all_of(signature.target_poses.begin(), signature.target_poses.end(),
                  [](const Eigen::Matrix4d &pose) { return pose.allFinite(); }) &&
      std::all_of(signature.reference_target_poses.begin(),
                  signature.reference_target_poses.end(),
                  [](const Eigen::Matrix4d &pose) { return pose.allFinite(); }) &&
      std::all_of(signature.gains.begin(), signature.gains.end(),
                  [](double value) { return std::isfinite(value); });
  if (!signature_is_finite) {
    reset_position_step_continuity_state();
    return false;
  }
  bool existing_target_geometry_changed = false;
  if (last_position_step_target_signature_.has_value()) {
    const auto &previous = *last_position_step_target_signature_;
    const bool target_topology_matches =
        previous.task_names == signature.task_names &&
        previous.target_poses.size() == signature.target_poses.size() &&
        previous.reference_target_poses.size() ==
            signature.reference_target_poses.size();
    const auto poses_match = [](const std::vector<Eigen::Matrix4d> &before,
                                const std::vector<Eigen::Matrix4d> &after) {
      for (std::size_t index = 0; index < after.size(); ++index) {
        const double translation_delta =
            (before[index].block<3, 1>(0, 3) -
             after[index].block<3, 1>(0, 3))
                .norm();
        const Eigen::Matrix3d rotation_delta =
            before[index].block<3, 3>(0, 0).transpose() *
            after[index].block<3, 3>(0, 0);
        const double rotation_angle = pinocchio::log3(rotation_delta).norm();
        if (!std::isfinite(translation_delta) ||
            !std::isfinite(rotation_angle) ||
            translation_delta > kStationaryTargetTranslationTolerance ||
            rotation_angle > kStationaryTargetRotationTolerance) {
          return false;
        }
      }
      return true;
    };
    const bool world_target_geometry_matches =
        target_topology_matches &&
        poses_match(previous.target_poses, signature.target_poses);
    const bool reference_target_geometry_matches =
        target_topology_matches && !signature.reference_target_poses.empty() &&
        poses_match(previous.reference_target_poses,
                    signature.reference_target_poses);
    const bool explicit_command_revision_enabled =
        previous.command_revision >= 0 || signature.command_revision >= 0;
    const bool command_revision_matches =
        previous.command_revision >= 0 && signature.command_revision >= 0 &&
        previous.command_revision == signature.command_revision;
    const bool target_geometry_matches =
        target_topology_matches &&
        (explicit_command_revision_enabled
             ? command_revision_matches
             : (world_target_geometry_matches ||
                reference_target_geometry_matches));
    bool matches = target_geometry_matches &&
                   previous.gains.size() == signature.gains.size();
    if (matches) {
      for (std::size_t index = 0; index < signature.gains.size(); ++index) {
        if (std::abs(previous.gains[index] - signature.gains[index]) >
            kStationaryTargetGainTolerance) {
          matches = false;
          break;
        }
      }
    }
    if (!matches) {
      reset_position_step_merit_window();
      position_step_stationary_anchor_blocks_.reset();
      existing_target_geometry_changed = !target_geometry_matches;
    }
    if (!explicit_command_revision_enabled && world_target_geometry_matches &&
        !reference_target_geometry_matches) {
      signature.reference_target_poses = previous.reference_target_poses;
    }
  } else {
    reset_position_step_merit_window();
    position_step_stationary_anchor_blocks_.reset();
    position_step_collision_command_floor_distances_.clear();
  }
  last_position_step_target_signature_ = std::move(signature);
  return existing_target_geometry_changed;
}

Eigen::Matrix4d KinematicsSolver::canonicalize_position_step_signature_pose(
    const std::string &task_name, const Eigen::Matrix4d &target_pose,
    const std::optional<pinocchio::SE3> &reference_pose) const {
  if (!reference_pose.has_value()) {
    return target_pose;
  }
  const auto task_iter = task_map_.find(task_name);
  if (task_iter != task_map_.end() &&
      std::dynamic_pointer_cast<RelativeFrameTask>(task_iter->second)) {
    return target_pose;
  }

  const pinocchio::SE3 world_target(target_pose.block<3, 3>(0, 0),
                                    target_pose.block<3, 1>(0, 3));
  const pinocchio::SE3 local_target = reference_pose->inverse() * world_target;
  Eigen::Matrix4d canonical_pose = Eigen::Matrix4d::Identity();
  canonical_pose.block<3, 3>(0, 0) = local_target.rotation();
  canonical_pose.block<3, 1>(0, 3) = local_target.translation();
  return canonical_pose;
}

void KinematicsSolver::reset_position_step_continuity_state() {
  last_position_step_target_signature_.reset();
  position_step_target_motion_blocks_.clear();
  position_step_target_motion_observed_ = false;
  position_step_collision_command_floor_distances_.clear();
  position_step_merit_priority_.reset();
  position_step_stationary_anchor_blocks_.reset();
  reset_position_step_merit_window();
}

void KinematicsSolver::reset_position_step_merit_window() {
  position_step_merit_window_anchor_.reset();
  position_step_merit_window_motion_ = 0.0;
  position_step_merit_window_samples_ = 0;
  position_step_merit_window_last_delta_.reset();
  position_step_merit_window_direction_reversals_ = 0;
  position_step_merit_window_last_merits_.reset();
  position_step_merit_window_error_increases_ = 0;
  position_step_stationary_guard_active_ = false;
  position_step_stationary_guard_can_reopen_ = true;
}

bool KinematicsSolver::should_hold_position_step_for_continuity(
    const PositionIKResult &result, const Eigen::VectorXd &current_q,
    double initial_merit, double final_merit,
    const std::vector<double> &initial_target_merits,
    const std::vector<double> &final_target_merits,
    const std::vector<double> &initial_target_block_merits,
    const std::vector<double> &final_target_block_merits,
    int merit_priority, double configuration_step_norm,
    bool owns_position_step_continuity,
    bool collision_violated, bool step_constraint_tradeoff_active) {
  if (!owns_position_step_continuity) {
    return false;
  }

  if (!position_step_merit_priority_.has_value() ||
      *position_step_merit_priority_ != merit_priority) {
    reset_position_step_merit_window();
    position_step_stationary_anchor_blocks_.reset();
    position_step_merit_priority_ = merit_priority;
  }

  const bool candidate_status = result.status == SolverStatus::kSuccess ||
                                result.status == SolverStatus::kNoProgress;
  const bool nominal_candidate =
      candidate_status && !collision_violated &&
      result.collision_rejection_count == 0 && result.stall_escape_count == 0 &&
      result.q_solution.size() == current_q.size() &&
      std::isfinite(initial_merit) && std::isfinite(final_merit) &&
      !initial_target_merits.empty() &&
      initial_target_merits.size() == final_target_merits.size() &&
      !initial_target_block_merits.empty() &&
      initial_target_block_merits.size() ==
          final_target_block_merits.size() &&
      std::all_of(initial_target_merits.begin(), initial_target_merits.end(),
                  [](double merit) { return std::isfinite(merit); }) &&
      std::all_of(final_target_merits.begin(), final_target_merits.end(),
                  [](double merit) { return std::isfinite(merit); }) &&
      std::all_of(initial_target_block_merits.begin(),
                  initial_target_block_merits.end(),
                  [](double merit) { return std::isfinite(merit); }) &&
      std::all_of(final_target_block_merits.begin(),
                  final_target_block_merits.end(),
                  [](double merit) { return std::isfinite(merit); }) &&
      std::isfinite(configuration_step_norm);
  if (!nominal_candidate) {
    const bool zero_motion_collision_rejection =
        collision_violated && result.collision_rejection_count > 0 &&
        result.stall_escape_count == 0 &&
        std::isfinite(configuration_step_norm) &&
        configuration_step_norm <= kPositionStepMeritMotionThreshold;
    if (zero_motion_collision_rejection) {
      return false;
    }
    reset_position_step_merit_window();
    return false;
  }

  const bool collision_constraint_active =
      collision_constraint_.has_value() && collision_constraint_->enabled;
  const bool com_constraint_active =
      com_constraint_.has_value() && com_constraint_->enabled;
  const bool relative_pose_constraint_active =
      relative_pose_constraint_.has_value() &&
      relative_pose_constraint_->enabled;
  const bool other_constraint_tradeoff_active =
      step_constraint_tradeoff_active ||
      get_linear_velocity_constraint_rows() > 0 ||
      !tight_frame_pose_constraints_.empty() ||
      !tight_point_constraints_.empty() || has_contact_frames();
  const bool stationary_unconstrained_velocity_step =
      !acceleration_limits_enabled_ && position_step_target_motion_observed_ &&
      !position_step_target_geometry_moved_ && !collision_constraint_active &&
      !com_constraint_active && !relative_pose_constraint_active &&
      !other_constraint_tradeoff_active;
  if (stationary_unconstrained_velocity_step &&
      !position_step_stationary_anchor_blocks_.has_value()) {
    position_step_stationary_anchor_blocks_ = initial_target_block_merits;
  }
  const std::vector<double> &stationary_anchor_blocks =
      position_step_stationary_anchor_blocks_.has_value()
          ? *position_step_stationary_anchor_blocks_
          : initial_target_block_merits;
  const auto dominant_anchor_block = std::max_element(
      stationary_anchor_blocks.begin(), stationary_anchor_blocks.end());
  const std::size_t dominant_anchor_block_index =
      static_cast<std::size_t>(std::distance(
          stationary_anchor_blocks.begin(), dominant_anchor_block));
  const bool stationary_target_regressed =
      stationary_unconstrained_velocity_step &&
      final_target_block_merits[dominant_anchor_block_index] >
          *dominant_anchor_block +
              kStationaryTargetMeritRegressionTolerance;
  if (stationary_target_regressed) {
    position_step_stationary_guard_active_ = true;
    position_step_stationary_guard_can_reopen_ = false;
    position_step_merit_window_anchor_ = initial_merit;
    position_step_merit_window_motion_ = 0.0;
    position_step_merit_window_samples_ = 0;
    position_step_merit_window_last_delta_.reset();
    position_step_merit_window_direction_reversals_ = 0;
    position_step_merit_window_last_merits_ = initial_target_merits;
    position_step_merit_window_error_increases_ = 0;
    return true;
  }

  if (!position_step_merit_window_anchor_.has_value()) {
    position_step_merit_window_anchor_ = initial_merit;
  }
  const Eigen::VectorXd candidate_delta =
      pinocchio::difference(robot_->model(), current_q, result.q_solution);
  if (position_step_merit_window_last_delta_.has_value() &&
      position_step_merit_window_last_delta_->size() ==
          candidate_delta.size() &&
      position_step_merit_window_last_delta_->norm() >
          kStationaryDirectionChangeMotionThreshold &&
      candidate_delta.norm() > kStationaryDirectionChangeMotionThreshold &&
      position_step_merit_window_last_delta_->dot(candidate_delta) < 0.0) {
    ++position_step_merit_window_direction_reversals_;
  }
  position_step_merit_window_last_delta_ = candidate_delta;
  if (position_step_merit_window_last_merits_.has_value() &&
      position_step_merit_window_last_merits_->size() ==
          final_target_merits.size()) {
    const auto dominant_iter = std::max_element(
        position_step_merit_window_last_merits_->begin(),
        position_step_merit_window_last_merits_->end());
    if (dominant_iter != position_step_merit_window_last_merits_->end()) {
      const std::size_t dominant_index = static_cast<std::size_t>(
          std::distance(position_step_merit_window_last_merits_->begin(),
                        dominant_iter));
      if (final_target_merits[dominant_index] > *dominant_iter + 1e-4) {
        ++position_step_merit_window_error_increases_;
      }
    }
  }
  position_step_merit_window_last_merits_ = final_target_merits;
  position_step_merit_window_motion_ +=
      std::max(0.0, configuration_step_norm);
  ++position_step_merit_window_samples_;

  if (position_step_stationary_guard_active_) {
    if (!position_step_stationary_guard_can_reopen_) {
      return true;
    }
    const bool hold_candidate = should_hold_non_improving_position_step(
        result, current_q, initial_merit, final_merit,
        configuration_step_norm, collision_violated);
    const bool made_sufficient_progress =
        !hold_candidate && has_sufficient_position_step_merit_reduction(
                               initial_merit, final_merit,
                               configuration_step_norm);
    if (made_sufficient_progress) {
      reset_position_step_merit_window();
    }
    return !made_sufficient_progress;
  }

  const int required_window_samples =
      runtime_config_.enable_auto_task_layout
          ? std::max(kStationaryTargetDwellCalls,
                     runtime_config_.auto_layout_cooldown_ticks + 1)
          : kStationaryTargetDwellCalls;
  if (position_step_merit_window_samples_ < required_window_samples) {
    return false;
  }

  const bool target_is_satisfied =
      final_merit <= kPositionStepSatisfiedMeritTolerance;
  const bool window_has_repeated_direction_reversals =
      position_step_merit_window_direction_reversals_ >
      kStationaryMaxDirectionReversals;
  const bool window_has_repeated_dominant_target_increases =
      position_step_merit_window_error_increases_ >
      kStationaryMaxDirectionReversals;
  const bool window_has_strong_net_progress =
      has_sufficient_position_step_merit_reduction(
          *position_step_merit_window_anchor_, final_merit,
          position_step_merit_window_motion_,
          kStationaryOscillationMinErrorReductionPerConfiguration);
  // Stateful layouts and redundant robots can reverse individual joint steps
  // while still making decisive nonlinear task-space progress. That exception
  // does not apply when the currently worst target repeatedly gets worse: an
  // aggregate improvement from sacrificing one commanded target is not useful
  // stationary-target progress.
  const bool window_was_oscillatory =
      window_has_repeated_dominant_target_increases ||
      (window_has_repeated_direction_reversals &&
       !window_has_strong_net_progress);
  const bool caller_owns_command_identity =
      last_position_step_target_signature_.has_value() &&
      last_position_step_target_signature_->command_revision >= 0;
  const bool window_has_required_progress =
      caller_owns_command_identity
          ? window_has_strong_net_progress
          : has_sufficient_position_step_merit_reduction(
                *position_step_merit_window_anchor_, final_merit,
                position_step_merit_window_motion_,
                kPositionStepMinErrorReductionPerConfiguration,
                kStationaryMinErrorReductionPerCall *
                    static_cast<double>(position_step_merit_window_samples_));
  const bool window_was_productive =
      !target_is_satisfied && !window_was_oscillatory &&
      window_has_required_progress;
  if (window_was_productive) {
    position_step_merit_window_anchor_ = final_merit;
    position_step_merit_window_motion_ = 0.0;
    position_step_merit_window_samples_ = 0;
    position_step_merit_window_last_delta_.reset();
    position_step_merit_window_direction_reversals_ = 0;
    position_step_merit_window_last_merits_.reset();
    position_step_merit_window_error_increases_ = 0;
    return false;
  }

  position_step_stationary_guard_active_ = true;
  position_step_stationary_guard_can_reopen_ =
      !caller_owns_command_identity && !target_is_satisfied &&
      !window_was_oscillatory;
  position_step_merit_window_anchor_ = initial_merit;
  position_step_merit_window_motion_ = 0.0;
  position_step_merit_window_samples_ = 0;
  return true;
}

// ============================================================
// Stall handler
// ============================================================

// Tolerance for floating-point distance comparisons in stall handler logic.
constexpr double kStallDistanceTolerance = 1e-8;

void KinematicsSolver::enable_stall_handler(double nominal_min_distance) {
  const double nom = std::max(0.0, nominal_min_distance);
  if (stall_config_.enabled) {
    if (std::abs(nom - stall_state_.nominal_min_distance) >
        kStallDistanceTolerance) {
      stall_state_.nominal_min_distance = nom;
    }
    if (collision_constraint_.has_value() &&
        collision_constraint_->constraint_activation_multiplier > 0.0) {
      collision_constraint_->constraint_activation_margin =
          collision_constraint_->constraint_activation_multiplier *
          std::max(collision_constraint_->min_distance,
                   stall_state_.nominal_min_distance);
    }
    return;
  }
  stall_config_.enabled = true;
  stall_state_.nominal_min_distance = nom;
  stall_state_.current_min_distance = nom;
  stall_state_.consecutive_stall_steps = 0;
  if (collision_constraint_.has_value() &&
      collision_constraint_->constraint_activation_multiplier > 0.0) {
    collision_constraint_->constraint_activation_margin =
        collision_constraint_->constraint_activation_multiplier *
        std::max(collision_constraint_->min_distance,
                 stall_state_.nominal_min_distance);
  }
}

void KinematicsSolver::disable_stall_handler() {
  if (stall_config_.enabled) {
    set_collision_min_distance(stall_state_.nominal_min_distance);
    stall_state_.current_min_distance = stall_state_.nominal_min_distance;
    stall_state_.consecutive_stall_steps = 0;
  }
  stall_config_.enabled = false;
}

bool KinematicsSolver::stall_handler_enabled() const {
  return stall_config_.enabled;
}

void KinematicsSolver::configure_stall_handler(int stall_threshold,
                                                double restore_rate,
                                                double floor_fraction) {
  stall_config_.stall_threshold = std::max(1, stall_threshold);
  stall_config_.restore_rate = std::max(0.0, restore_rate);
  stall_config_.floor_fraction = std::clamp(floor_fraction, 0.0, 1.0);
  stall_user_configured_ = true;
}

bool KinematicsSolver::stall_handler_is_relaxed() const {
  return stall_config_.enabled &&
         stall_state_.current_min_distance <
             stall_state_.nominal_min_distance - kStallDistanceTolerance;
}

double KinematicsSolver::stall_handler_current_min_distance() const {
  return stall_state_.current_min_distance;
}

int KinematicsSolver::stall_handler_consecutive_stall_steps() const {
  return stall_state_.consecutive_stall_steps;
}

// ============================================================
// Elastic band joint limit expansion
// ============================================================

void KinematicsSolver::enable_elastic_band(double delta_max) {
  if (elastic_band_config_.enabled) {
    elastic_band_config_.delta_max = std::max(0.0, delta_max);
    return;
  }
  elastic_band_config_.enabled = true;
  elastic_band_config_.delta_max = std::max(0.0, delta_max);
  const int nv = robot_->nv();
  elastic_band_state_.delta = Eigen::VectorXd::Zero(nv);
  elastic_band_state_.consecutive_stall_steps = 0;
  elastic_band_state_.total_expansion_steps = 0;

  // Pre-seed expansion for joints that are at or very near their limits.
  // This avoids the initial stall when the seed configuration has joints
  // sitting exactly on a limit boundary (common with zero-config seeds).
  const auto &v2c = velocity_to_config_index_cache();
  auto [q_min, q_max] = robot_->get_joint_limits();
  const Eigen::VectorXd q_cur = robot_->get_current_configuration();
  const double seed_delta = std::min(elastic_band_config_.expand_rate,
                                     elastic_band_config_.delta_max);
  for (int i = 0; i < nv; ++i) {
    const int q_idx =
        (i < static_cast<int>(v2c.size())) ? v2c[i] : kVelocityToConfigUnmapped;
    if (q_idx == kVelocityToConfigUnmapped || q_idx >= q_cur.size()) {
      continue;
    }
    if (!std::isfinite(q_min[q_idx]) || !std::isfinite(q_max[q_idx])) {
      continue;
    }
    const double margin_lo = q_cur[q_idx] - q_min[q_idx];
    const double margin_hi = q_max[q_idx] - q_cur[q_idx];
    if (margin_lo < kElasticAtLimitMargin || margin_hi < kElasticAtLimitMargin) {
      elastic_band_state_.delta[i] = seed_delta;
    }
  }

  // Warm-start collision margin: if the initial config already violates
  // the collision min_distance (but is not penetrating), temporarily relax
}

void KinematicsSolver::disable_elastic_band() {
  elastic_band_config_.enabled = false;
  elastic_band_state_.delta.setZero();
  elastic_band_state_.consecutive_stall_steps = 0;
}

bool KinematicsSolver::elastic_band_enabled() const {
  return elastic_band_config_.enabled;
}

void KinematicsSolver::configure_elastic_band(double delta_max,
                                               double expand_rate,
                                               double decay_rate,
                                               int stall_threshold,
                                               bool expand_only_saturated) {
  elastic_band_config_.delta_max = std::max(0.0, delta_max);
  elastic_band_config_.expand_rate = std::max(0.0, expand_rate);
  elastic_band_config_.decay_rate = std::clamp(decay_rate, 0.0, 1.0);
  elastic_band_config_.stall_threshold = std::max(1, stall_threshold);
  elastic_band_config_.expand_only_saturated = expand_only_saturated;
}

double KinematicsSolver::elastic_band_max_delta() const {
  if (!elastic_band_config_.enabled ||
      elastic_band_state_.delta.size() == 0) {
    return 0.0;
  }
  return elastic_band_state_.delta.maxCoeff();
}

Eigen::VectorXd KinematicsSolver::elastic_band_deltas() const {
  if (!elastic_band_config_.enabled ||
      elastic_band_state_.delta.size() == 0) {
    return Eigen::VectorXd::Zero(robot_->nv());
  }
  return elastic_band_state_.delta;
}

bool KinematicsSolver::elastic_band_is_expanded() const {
  return elastic_band_config_.enabled &&
         elastic_band_state_.delta.size() > 0 &&
         elastic_band_state_.delta.maxCoeff() > 1e-10;
}

void KinematicsSolver::elastic_band_update(
    const VelocitySolverResult &result) {
  if (!elastic_band_config_.enabled) {
    return;
  }

  const auto &cfg = elastic_band_config_;
  auto &st = elastic_band_state_;
  const int nv = robot_->nv();

  // Ensure delta vector is sized correctly.
  if (st.delta.size() != nv) {
    st.delta = Eigen::VectorXd::Zero(nv);
  }

  // --- Proactive proximity-based expansion ---
  // Instead of waiting for stalls, expand margins proactively when:
  //   1. A joint is saturated (at its velocity limit), AND
  //   2. The task scale is low (solver is struggling)
  // This prevents the stop-start oscillation of reactive expansion.

  const double primary_scale =
      result.task_scales.empty() ? 1.0 : result.task_scales[0];
  const bool explicit_task_error =
      !result.task_errors.empty() && result.task_errors[0] > 1e-4;
  const bool has_task_error =
      explicit_task_error || primary_scale < 1e-6;
  const bool scale_is_low = primary_scale < 0.5 && has_task_error;

  // Build saturated joint set.
  std::vector<bool> is_saturated(nv, false);
  for (int idx : result.saturated_joints) {
    if (idx >= 0 && idx < nv) {
      is_saturated[idx] = true;
    }
  }


  // Expansion: proportional to how constrained we are.
  // - Scale = 0 (infeasible) → boost to delta_max/2 immediately
  // - Scale near 0 → expand at full rate
  // - Scale near 0.5 → expand at half rate
  // - No saturated joints or scale >= 0.5 → no expansion
  if (scale_is_low &&
      (!result.saturated_joints.empty() || primary_scale < 1e-6)) {
    const double expansion_factor =
        std::max(0.0, 1.0 - 2.0 * primary_scale);
    const double step_expand = cfg.expand_rate * expansion_factor;
    // When completely infeasible, boost immediately to reduce the number
    // of stalled steps needed to reach useful expansion.
    const double boost_floor =
        (primary_scale < 1e-6) ? cfg.delta_max * 0.5 : 0.0;

    for (int i = 0; i < nv; ++i) {
      const bool eligible =
          !cfg.expand_only_saturated || is_saturated[i] ||
          (primary_scale < 1e-6 && result.saturated_joints.empty());
      if (eligible) {
        const double new_delta = std::max(st.delta[i] + step_expand, boost_floor);
        st.delta[i] = std::min(new_delta, cfg.delta_max);
      }
    }
    st.total_expansion_steps++;
  }

  // --- Decay: shrink deltas for joints that are NOT saturated ---
  // Only decay joints that have room — saturated joints keep their expansion
  // to avoid the oscillation between expand/decay at the limit boundary.
  for (int i = 0; i < nv; ++i) {
    if (st.delta[i] > 0.0 && !is_saturated[i]) {
      st.delta[i] *= (1.0 - cfg.decay_rate);
      if (st.delta[i] < 1e-8) {
        st.delta[i] = 0.0;
      }
    }
  }

  // Global decay when task is satisfied (no error) — all joints.
  if (!has_task_error) {
    for (int i = 0; i < nv; ++i) {
      st.delta[i] *= (1.0 - cfg.decay_rate);
      if (st.delta[i] < 1e-8) {
        st.delta[i] = 0.0;
      }
    }
  }
}

void KinematicsSolver::stall_handler_update(VelocitySolverResult &result) {
  if (!stall_config_.enabled) {
    return;
  }

  const auto &cfg = stall_config_;
  auto &st = stall_state_;

  const double dq_norm = result.joint_velocities.norm();

  // --- Read active collision-row distances ---
  // Avoid per-step temporary allocations by computing these aggregates directly.
  double collision_dist = std::numeric_limits<double>::infinity();
  bool any_collision_binding = false;
  bool any_penetration = false;
  bool have_collision_distance = false;
  for (const auto &row : last_collision_debug_list_) {
    if (!std::isfinite(row.distance)) {
      continue;
    }
    have_collision_distance = true;
    collision_dist = std::min(collision_dist, row.distance);
    any_collision_binding =
        any_collision_binding || (row.distance <= st.current_min_distance);
    any_penetration = any_penetration || (row.distance < 0.0);
  }
  if (!have_collision_distance && last_collision_debug_.has_value() &&
      std::isfinite(last_collision_debug_->distance)) {
    collision_dist = last_collision_debug_->distance;
    any_collision_binding = (collision_dist <= st.current_min_distance);
    any_penetration = (collision_dist < 0.0);
  }
  // If any joint velocity is saturated at a combined velocity/position bound,
  // treat this as limit-dominated and avoid collision-margin relaxation.
  const bool limit_dominated_stall = !result.saturated_joints.empty();
  const bool near_zero_motion = dq_norm < cfg.dq_stall_eps;
  const bool success_but_blocked =
      (result.status == SolverStatus::kSuccess) && near_zero_motion;
  const bool is_stall =
      ((result.status != SolverStatus::kSuccess) && near_zero_motion) ||
      success_but_blocked;
  const bool solver_healthy =
      (result.status == SolverStatus::kSuccess) && !success_but_blocked;

  // --- Update stall counter ---
  // Keep the counter sticky while collision rows are binding, even if the
  // solver occasionally reports Success, so max_steps=1 teleop loops can
  // accumulate stall evidence across ticks.
  const bool sticky_success_blocked =
      (result.status == SolverStatus::kSuccess) && any_collision_binding;
  if (is_stall) {
    st.consecutive_stall_steps++;
    st.total_stall_steps++;
  } else if (!sticky_success_blocked) {
    st.consecutive_stall_steps = 0;
  }

  const double floor_min = st.nominal_min_distance * cfg.floor_fraction;

  // --- Phase 1: Stall threshold reached → relax collision margin ---
  if (st.consecutive_stall_steps >= cfg.stall_threshold) {
    // Collision is only the bottleneck when the QP row is actually
    // binding: distance at or below the active min_distance.  The
    // proximity band is intentionally NOT used here — it would cause
    // false relaxation when joint-limit clamping (not collision) is the
    // true cause of infeasibility.
    const bool collision_is_bottleneck =
        any_collision_binding && !limit_dominated_stall;

    if (any_penetration && !limit_dominated_stall) {
      // Penetration escape: set margin just below actual penetration depth
      // so the collision QP row has slack and motion can resume.
      const double escape_margin =
          collision_dist - cfg.collision_proximity_band;
      st.current_min_distance = escape_margin;
      set_collision_min_distance(escape_margin);
      st.total_relaxation_steps++;
    } else if (collision_is_bottleneck &&
               st.current_min_distance > floor_min) {
      // Positive-distance stall: ratchet margin down by one drop step.
      const double drop = cfg.relax_drop_fraction * st.nominal_min_distance;
      const double new_min =
          std::max(floor_min, st.current_min_distance - drop);
      st.current_min_distance = new_min;
      set_collision_min_distance(new_min);
      st.total_relaxation_steps++;
    }
    st.consecutive_stall_steps = 0;
    return;
  }

  // --- Phase 2: Solver healthy → restore margin toward nominal ---
  if (solver_healthy &&
      st.current_min_distance <
          st.nominal_min_distance - kStallDistanceTolerance) {
    const double gap = st.nominal_min_distance - st.current_min_distance;
    const double step = std::max(cfg.restore_rate * st.nominal_min_distance,
                                 cfg.relax_drop_fraction * gap);

    // Ceiling: never push margin above actual clearance minus proximity
    // band — that would re-create the infeasible QP.
    double ceiling = st.nominal_min_distance;
    if (std::isfinite(collision_dist)) {
      ceiling = std::min(ceiling,
                         collision_dist - cfg.collision_proximity_band);
    }
    const double new_min =
        std::min(ceiling, st.current_min_distance + step);

    if (new_min > st.current_min_distance) {
      st.current_min_distance = new_min;
      set_collision_min_distance(new_min);
    }
  }
}

} // namespace embodik
