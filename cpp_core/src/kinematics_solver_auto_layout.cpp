/**
 * @file kinematics_solver_auto_layout.cpp
 * @brief Automatic pose-task layout policy for KinematicsSolver
 */

#include <algorithm>
#include <cmath>
#include <string>

#include <embodik/kinematics_solver.hpp>

namespace embodik {

void KinematicsSolver::reset_auto_task_layout_state() {
  current_auto_task_layout_ = TaskLayout::kMerged;
  auto_layout_below_low_count_ = 0;
  auto_layout_has_feedback_ = false;
  auto_layout_binding_score_ = 0.0;
}

void KinematicsSolver::select_auto_task_layout() {
  if (!runtime_config_.enable_auto_task_layout) {
    return;
  }

  if (auto_layout_has_feedback_) {
    const double high = runtime_config_.auto_layout_binding_threshold_high;
    const double low = runtime_config_.auto_layout_binding_threshold_low;
    const int cooldown =
        std::max(1, runtime_config_.auto_layout_cooldown_ticks);
    if (current_auto_task_layout_ == TaskLayout::kMerged) {
      if (auto_layout_binding_score_ > high) {
        current_auto_task_layout_ = TaskLayout::kSplit;
        auto_layout_below_low_count_ = 0;
      }
    } else {
      if (auto_layout_binding_score_ < low) {
        ++auto_layout_below_low_count_;
        if (auto_layout_below_low_count_ >= cooldown) {
          current_auto_task_layout_ = TaskLayout::kMerged;
          auto_layout_below_low_count_ = 0;
        }
      } else {
        auto_layout_below_low_count_ = 0;
      }
    }
  }

  for (auto &kv : pose_task_groups_) {
    if (kv.second && kv.second->auto_switch()) {
      kv.second->set_layout(current_auto_task_layout_);
    }
  }
}

VelocitySolverResult KinematicsSolver::retry_auto_task_layout_as_split_if_needed(
    const Eigen::VectorXd &q, VelocitySolverResult result,
    const std::vector<int> &velocity_lock_indices,
    const std::optional<TorsoPoseConstraintOptions> &torso_constraint,
    const std::optional<double> &step_validation_dt,
    const std::vector<PositionStepPriorityConstraintSpec>
        &priority_constraints,
    bool apply_position_step_acceleration_limits) {
  const bool merged_succeeded = result.status == SolverStatus::kSuccess;
  const bool merged_binds =
      auto_layout_binding_score_ >
      runtime_config_.auto_layout_binding_threshold_high;
  if (!runtime_config_.enable_auto_task_layout ||
      result.active_task_layout != TaskLayout::kMerged ||
      (merged_succeeded && !merged_binds)) {
    return result;
  }

  current_auto_task_layout_ = TaskLayout::kSplit;
  auto_layout_below_low_count_ = 0;
  for (auto &kv : pose_task_groups_) {
    if (kv.second && kv.second->auto_switch()) {
      kv.second->set_layout(TaskLayout::kSplit);
    }
  }

  pending_velocity_lock_indices_ = velocity_lock_indices;
  pending_step_torso_constraint_ = torso_constraint;
  pending_step_validation_dt_ = step_validation_dt;
  pending_position_step_priority_constraints_ = priority_constraints;
  pending_reuse_current_kinematics_ = true;
  pending_position_step_acceleration_limits_ =
      apply_position_step_acceleration_limits;
  VelocitySolverResult retry = solve_velocity(q, true);
  const char *reason =
      merged_succeeded ? "binding merged solve" : "non-success merged solve";
  if (retry.status_message.empty()) {
    retry.status_message =
        std::string("auto task layout retried split after ") + reason;
  } else {
    retry.status_message = std::string("auto task layout retried split after ") +
                           reason + ": " + retry.status_message;
  }
  if (merged_succeeded && retry.status != SolverStatus::kSuccess) {
    if (result.status_message.empty()) {
      result.status_message =
          std::string("auto task layout kept successful merged solve after "
                      "failed split retry: ") +
          retry.status_message;
    }
    return result;
  }
  return retry;
}

void KinematicsSolver::update_auto_task_layout_feedback(
    const VelocitySolverResult &result) {
  if (!runtime_config_.enable_auto_task_layout) {
    return;
  }
  auto clip01 = [](double x) {
    if (!std::isfinite(x)) {
      return 0.0;
    }
    return std::clamp(x, 0.0, 1.0);
  };

  double scale_binding = 0.0;
  if (!result.task_scales.empty()) {
    double min_scale = 1.0;
    for (double s : result.task_scales) {
      if (std::isfinite(s)) {
        min_scale = std::min(min_scale, s);
      }
    }
    scale_binding = clip01(1.0 - min_scale);
  }

  const double joint_clip_fraction =
      robot_->nv() > 0 ? static_cast<double>(result.saturated_joints.size()) /
                             static_cast<double>(robot_->nv())
                       : 0.0;

  double collision_binding = 0.0;
  if (collision_constraint_.has_value() && collision_constraint_->enabled &&
      std::isfinite(last_constraint_min_recovery_margin_)) {
    const double margin =
        std::max(collision_constraint_->constraint_activation_margin, 1e-9);
    const double clearance = last_constraint_min_recovery_margin_;
    collision_binding = clip01(1.0 - clearance / margin);
  }

  auto_layout_binding_score_ =
      std::max(scale_binding, std::max(joint_clip_fraction, collision_binding));
  auto_layout_has_feedback_ = true;
}

} // namespace embodik
