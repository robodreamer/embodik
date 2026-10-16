/**
 * @file kinematics_solver_velocity.cpp
 * @brief Velocity-level solve path for KinematicsSolver
 */

#include "kinematics_solver_internal.hpp"
#include <cstdlib>

namespace embodik {

VelocitySolverResult
KinematicsSolver::solve_velocity_with_state(const Eigen::VectorXd &current_q,
                                            const Eigen::VectorXd &current_dq,
                                            bool apply_limits,
                                            bool stall_recovery) {
  if (current_dq.size() != robot_->nv()) {
    VelocitySolverResult result;
    result.status = SolverStatus::kInvalidInput;
    result.status_message =
        "current_dq size does not match robot nv in solve_velocity_with_state";
    result.joint_velocities = Eigen::VectorXd::Zero(robot_->nv());
    result.solution.assign(static_cast<std::size_t>(robot_->nv()), 0.0);
    return result;
  }
  if (!current_dq.allFinite()) {
    VelocitySolverResult result;
    result.status = SolverStatus::kNonFiniteInput;
    result.status_message =
        "current_dq contains non-finite values in solve_velocity_with_state";
    result.joint_velocities = Eigen::VectorXd::Zero(robot_->nv());
    result.solution.assign(static_cast<std::size_t>(robot_->nv()), 0.0);
    return result;
  }
  pending_explicit_current_dq_ = current_dq;
  return solve_velocity(current_q, apply_limits, stall_recovery);
}

std::pair<double, double> KinematicsSolver::calculate_velocity_box_constraint(
    double position_margin_lower, double position_margin_upper,
    double velocity_limit, double acceleration_limit, double dt,
    double min_velocity_headroom, double headroom_activation_margin) const {
  const double raw_margin_lower = position_margin_lower;
  const double raw_margin_upper = position_margin_upper;
  const bool outside_lower = raw_margin_lower < -limit_recovery_enter_epsilon_;
  const bool outside_upper = raw_margin_upper < -limit_recovery_enter_epsilon_;
  const double effective_release_margin = std::max(
      limit_exit_release_margin_,
      limit_recovery_exit_epsilon_ - limit_recovery_enter_epsilon_);

  if (outside_lower && outside_upper) {
    return std::make_pair(-velocity_limit, velocity_limit);
  }

  // Clamp margins to be non-negative for nominal bounds.
  position_margin_lower = std::max(0.0, position_margin_lower);
  position_margin_upper = std::max(0.0, position_margin_upper);

  // Preserve the legacy continuous stopping bound when acceleration history is
  // not enforced.  With acceleration limits enabled, use the exact sampled-data
  // stopping distance so the outer-tick command remains recursively feasible.
  double vel_from_pos_lower = -position_margin_lower / dt;
  double vel_from_pos_upper = position_margin_upper / dt;

  const double vel_from_accel_lower =
      acceleration_limits_enabled_
          ? -discrete_stopping_velocity_limit(position_margin_lower,
                                               acceleration_limit, dt)
          : -std::sqrt(2 * acceleration_limit * position_margin_lower);
  const double vel_from_accel_upper =
      acceleration_limits_enabled_
          ? discrete_stopping_velocity_limit(position_margin_upper,
                                              acceleration_limit, dt)
          : std::sqrt(2 * acceleration_limit * position_margin_upper);

  // Take most restrictive limits
  double lower_limit =
      std::max({vel_from_pos_lower, -velocity_limit, vel_from_accel_lower});
  double upper_limit =
      std::min({vel_from_pos_upper, velocity_limit, vel_from_accel_upper});

  // Shared velocity-box headroom policy:
  // - explicit per-call headroom (min_velocity_headroom >= 0), or
  // - legacy solver-level saturation-exit behavior.
  if (!outside_lower && !outside_upper) {
    bool apply_headroom = false;
    double min_vel = 0.0;
    double activation_margin = std::max(0.0, headroom_activation_margin);
    if (min_velocity_headroom >= 0.0) {
      apply_headroom = true;
      min_vel = std::max(0.0, min_velocity_headroom);
    } else if (enable_saturation_exit_behavior_) {
      apply_headroom = true;
      min_vel = kMinBoundFraction * velocity_limit;
      activation_margin = kMarginThreshold;
    }
    if (apply_headroom && min_vel > 0.0) {
      if (lower_limit > -min_vel && raw_margin_lower > activation_margin) {
        lower_limit = -min_vel;
      }
      if (upper_limit < min_vel && raw_margin_upper > activation_margin) {
        upper_limit = min_vel;
      }
    }
  }

  if (outside_lower) {
    const double violation =
        std::max(0.0, -raw_margin_lower - effective_release_margin);
    const double recovery_min = (limit_recovery_gain_ * violation) / dt;
    const double recovery_lower = std::min(recovery_min, velocity_limit);
    if (recovery_lower > lower_limit) {
      lower_limit = recovery_lower;
    }
  } else if (outside_upper) {
    const double violation =
        std::max(0.0, -raw_margin_upper - effective_release_margin);
    const double recovery_max = -(limit_recovery_gain_ * violation) / dt;
    const double recovery_upper = std::max(recovery_max, -velocity_limit);
    if (recovery_upper < upper_limit) {
      upper_limit = recovery_upper;
    }
  }

  if (lower_limit > upper_limit) {
    const double midpoint = 0.5 * (lower_limit + upper_limit);
    lower_limit = midpoint;
    upper_limit = midpoint;
  }

  return std::make_pair(lower_limit, upper_limit);
}

const std::vector<int> &KinematicsSolver::velocity_to_config_index_cache() {
  const int nv = robot_->nv();
  const RobotModel *model = robot_.get();
  if (velocity_to_config_cache_robot_ == model &&
      velocity_to_config_cache_nv_ == nv &&
      static_cast<int>(velocity_to_config_index_cache_.size()) == nv) {
    return velocity_to_config_index_cache_;
  }
  velocity_to_config_index_cache_.assign(nv, kVelocityToConfigUnmapped);
  const auto joint_names = robot_->get_joint_names();
  for (const auto &joint_name : joint_names) {
    const int v_idx = robot_->get_joint_velocity_index(joint_name);
    const int v_size = robot_->get_joint_velocity_size(joint_name);
    const int q_idx = robot_->get_joint_config_index(joint_name);
    const int q_size = robot_->get_joint_config_size(joint_name);
    if (v_size <= 0 || v_idx < 0 || q_idx < 0) {
      continue;
    }
    if (v_size == 1 && q_size != 1) {
      continue;
    }
    const int dims = std::min(v_size, q_size);
    for (int k = 0; k < dims; ++k) {
      const int vk = v_idx + k;
      const int qk = q_idx + k;
      if (vk >= 0 && vk < nv && qk >= 0 && qk < robot_->nq()) {
        velocity_to_config_index_cache_[vk] = qk;
      }
    }
  }
  velocity_to_config_cache_robot_ = model;
  velocity_to_config_cache_nv_ = nv;
  return velocity_to_config_index_cache_;
}

void KinematicsSolver::clamp_jacobians_near_joint_limits(
    std::vector<Eigen::MatrixXd> &jacobians,
    const std::vector<Eigen::VectorXd> &goals,
    const std::vector<ObjectiveSolveConfig> &objective_configs,
    const std::vector<bool> &preserve_physical_jacobians,
    const std::vector<int> &velocity_to_config_index) const {
  constexpr double kJointLimitClampMargin = 5e-4;
  const int nv_clamp = robot_->nv();
  auto [q_min_clamp, q_max_clamp] = robot_->get_joint_limits();
  const Eigen::VectorXd q_cur_clamp = robot_->get_current_configuration();
  const int fb_offset = robot_->is_floating_base() ? 6 : 0;

  for (int i = fb_offset; i < nv_clamp; ++i) {
    const int q_idx = (i < static_cast<int>(velocity_to_config_index.size()))
                          ? velocity_to_config_index[i]
                          : kVelocityToConfigUnmapped;
    if (q_idx == kVelocityToConfigUnmapped || q_idx >= q_cur_clamp.size() ||
        q_idx >= q_min_clamp.size() || q_idx >= q_max_clamp.size()) {
      continue;
    }
    if (!std::isfinite(q_min_clamp[q_idx]) ||
        !std::isfinite(q_max_clamp[q_idx])) {
      continue;
    }

    // Skip Jacobian clamping for joints with active elastic band expansion.
    // The elastic zone explicitly allows motion near/beyond nominal limits.
    if (elastic_band_config_.enabled &&
        i < elastic_band_state_.delta.size() &&
        elastic_band_state_.delta[i] > kJointLimitClampMargin) {
      continue;
    }

    const double margin_lower = q_cur_clamp[q_idx] - q_min_clamp[q_idx];
    const double margin_upper = q_max_clamp[q_idx] - q_cur_clamp[q_idx];
    const bool clamp_lower =
        margin_lower > 0.0 && margin_lower < kJointLimitClampMargin;
    const bool clamp_upper =
        margin_upper > 0.0 && margin_upper < kJointLimitClampMargin;
    if (!clamp_lower && !clamp_upper) {
      continue;
    }

    for (size_t objective_index = 0; objective_index < jacobians.size();
         ++objective_index) {
      auto &J = jacobians[objective_index];
      // Preserved frame objectives retain the physical motion model. Removing
      // a column based on its gradient can also remove an inward nullspace
      // direction needed to leave a singular boundary configuration. The
      // position-derived velocity bounds enforce the allowed joint motion.
      if (objective_index < preserve_physical_jacobians.size() &&
          preserve_physical_jacobians[objective_index]) {
        continue;
      }
      if (J.cols() != nv_clamp || J.rows() <= 0) {
        continue;
      }
      const bool goal_directed_clamp =
          objective_index < objective_configs.size() &&
          objective_configs[objective_index].use_goal_directed_limit_clamp;
      if (goal_directed_clamp) {
        if (objective_index >= goals.size() ||
            goals[objective_index].size() != J.rows()) {
          continue;
        }
        const double requested_direction =
            J.col(i).dot(goals[objective_index]);
        if ((clamp_lower && requested_direction < 0.0) ||
            (clamp_upper && requested_direction > 0.0)) {
          J.col(i).setZero();
        }
      } else {
        for (int row = 0; row < static_cast<int>(J.rows()); ++row) {
          double &value = J(row, i);
          if (clamp_lower && value < 0.0) {
            value = 0.0;
          }
          if (clamp_upper && value > 0.0) {
            value = 0.0;
          }
        }
      }
    }
  }
}


VelocitySolverResult
KinematicsSolver::solve_velocity(const Eigen::VectorXd &current_q,
                                 bool apply_limits,
                                 bool stall_recovery) {
  if (stall_recovery && !stall_config_.enabled) {
    const double nominal =
        get_collision_min_distance() > 0.0 ? get_collision_min_distance() : 0.0;
    enable_stall_handler(nominal);
  }
  const bool timing = timing_breakdown_enabled_;
  auto get_elapsed_ms = [](auto start) {
    return std::chrono::duration_cast<std::chrono::microseconds>(
               std::chrono::high_resolution_clock::now() - start)
               .count() /
           1000.0;
  };

  VelocitySolverResult result;
  if (position_step_hard_constraints_ != nullptr) {
    position_step_hard_constraints_->valid = false;
  }
  select_auto_task_layout();
  struct ClearPendingVelocityLocks {
    KinematicsSolver *solver;
    ~ClearPendingVelocityLocks() {
      if (solver != nullptr) {
        solver->pending_velocity_lock_indices_.clear();
        solver->pending_step_torso_constraint_.reset();
        solver->pending_reuse_current_kinematics_ = false;
        solver->pending_step_validation_dt_.reset();
        solver->pending_position_step_priority_constraints_.clear();
        solver->pending_position_step_acceleration_limits_ = false;
        solver->pending_explicit_current_dq_.reset();
      }
    }
  } clear_pending_locks{this};
  (void)clear_pending_locks;
  // Timing fields default to 0.0; only populate when timing is enabled.
  const Eigen::VectorXd q_eval =
      (current_q.size() > 0) ? current_q : robot_->get_current_configuration();

  if (q_eval.size() != robot_->nq()) {
    result.status = SolverStatus::kInvalidInput;
    result.status_message =
        "current configuration size does not match robot nq in solve_velocity";
    return result;
  }
  if (!q_eval.allFinite()) {
    result.status = SolverStatus::kNonFiniteInput;
    result.status_message =
        "current configuration contains non-finite values in solve_velocity";
    result.solution.assign(static_cast<std::size_t>(robot_->nv()), 0.0);
    result.joint_velocities = Eigen::VectorXd::Zero(robot_->nv());
    result.limits_applied = apply_limits;
    last_solution_dq_norm_ = 0.0;
    return result;
  }
  const Eigen::VectorXd *explicit_current_dq = active_explicit_current_dq();
  if (velocity_zmp_constraint_.has_value() &&
      velocity_zmp_constraint_->enabled && explicit_current_dq == nullptr) {
    result.status = SolverStatus::kInvalidInput;
    result.status_message =
        "velocity-ZMP constraint requires explicit current_dq; use "
        "solve_velocity_with_state or PositionStepOptions.current_joint_velocity";
    result.solution.assign(static_cast<std::size_t>(robot_->nv()), 0.0);
    result.joint_velocities = Eigen::VectorXd::Zero(robot_->nv());
    result.limits_applied = apply_limits;
    last_solution_dq_norm_ = 0.0;
    return result;
  }

  // Use provided configuration or robot's current
  if (current_q.size() > 0) {
    if (current_q.size() != robot_->nq()) {
      result.status = SolverStatus::kInvalidInput;
      result.status_message =
          "current_q size does not match robot nq in solve_velocity";
      return result;
    }
    const Eigen::VectorXd &robot_q = robot_->get_current_configuration();
    const bool can_reuse_current_kinematics =
        pending_reuse_current_kinematics_ && robot_q.size() == current_q.size() &&
        (robot_q - current_q).squaredNorm() <= 1e-24;
    if (can_reuse_current_kinematics) {
      result.pinocchio_kinematics_time_ms = 0.0;
    } else if (timing) {
      auto t_kin_start = std::chrono::high_resolution_clock::now();
      robot_->update_kinematics(current_q);
      result.pinocchio_kinematics_time_ms = get_elapsed_ms(t_kin_start);
    } else {
      robot_->update_kinematics(current_q);
    }
  } else {
    if (timing) {
      auto t_kin_start = std::chrono::high_resolution_clock::now();
      robot_->update_kinematics(robot_->get_current_configuration());
      result.pinocchio_kinematics_time_ms = get_elapsed_ms(t_kin_start);
    } else {
      robot_->update_kinematics(robot_->get_current_configuration());
    }
  }

  // Auto-enable elastic band when any task uses SCALE_ELASTIC mode.
  {
    bool any_scale_elastic = false;
    for (const auto &task : tasks_) {
      if (task && task->isActive() &&
          task->getSolveMode() == TaskSolveMode::kScaleElastic) {
        any_scale_elastic = true;
        break;
      }
    }
    if (any_scale_elastic && !elastic_band_config_.enabled) {
      enable_elastic_band(0.05);
      configure_elastic_band(/*delta_max=*/0.05, /*expand_rate=*/0.01,
                             /*decay_rate=*/0.2, /*stall_threshold=*/3,
                             /*expand_only_saturated=*/true);
    }
  }

  // Sort tasks by priority
  sort_tasks_by_priority();

  // Update all tasks with current robot state
  if (timing) {
    auto t_task_start = std::chrono::high_resolution_clock::now();
    for (auto &task : tasks_) {
      if (task->isActive()) {
        task->update(*robot_);
      }
    }
    result.task_update_time_ms = get_elapsed_ms(t_task_start);
  } else {
    for (auto &task : tasks_) {
      if (task->isActive()) {
        task->update(*robot_);
      }
    }
  }

  // Collect active tasks: group by priority for order-invariant behavior.
  auto &goals = scratch_goals_;
  auto &jacobians = scratch_jacobians_;
  auto &objective_configs = scratch_objective_configs_;
  auto &objective_tasks = scratch_objective_tasks_;
  auto &excluded_union = scratch_excluded_union_;
  goals.clear();
  jacobians.clear();
  objective_configs.clear();
  objective_tasks.clear();
  excluded_union.clear();
  goals.reserve(tasks_.size());
  jacobians.reserve(tasks_.size());
  objective_configs.reserve(tasks_.size());
  objective_tasks.reserve(tasks_.size());
  excluded_union.reserve(tasks_.size());
  std::vector<int> task_exclusion_counts(
      static_cast<std::size_t>(robot_->nv()), 0);
  std::vector<bool> project_collision_tangent_objective;
  project_collision_tangent_objective.reserve(tasks_.size());
  std::vector<bool> project_elastic_collision_recovery_objective;
  project_elastic_collision_recovery_objective.reserve(tasks_.size());
  int active_task_count = 0;
  std::vector<bool> preserve_physical_jacobians;
  preserve_physical_jacobians.reserve(tasks_.size());

  int current_priority = std::numeric_limits<int>::min();
  auto &group_tasks = scratch_group_tasks_;
  group_tasks.clear();
  group_tasks.reserve(tasks_.size());

  auto velocity_task_jacobian = [&](const std::shared_ptr<Task> &task,
                                    const Eigen::VectorXd &goal) {
    Eigen::MatrixXd jacobian = task->getJacobian();
    const auto *frame = dynamic_cast<const FrameTask *>(task.get());
    const bool unconfigured_frame = frame && !frame->hasConfiguredReference();
    const bool inactive_conditioning =
        dynamic_cast<const ManipulabilityTask *>(task.get()) != nullptr &&
        !task->target_velocity_.has_value() &&
        goal.template lpNorm<1>() < tight_tolerance_;
    if (unconfigured_frame || inactive_conditioning) {
      // An unused frame or a stationary conditioning gradient is not a
      // commanded hold. Retain the row layout without consuming nullspace.
      jacobian.setZero();
    }
    return jacobian;
  };

  auto flush_group = [&]() {
    if (group_tasks.empty()) {
      return;
    }

    bool all_scale_family = true;
    bool all_strict_scale = true;
    bool all_min_error = true;
    bool all_frame_components = true;
    const bool direct_velocity_solve = !pending_step_validation_dt_.has_value();
    bool fallback_value = group_tasks.front()->getAllowMinErrorFallback();
    bool fallback_consistent = true;
    bool use_goal_directed_limit_clamp = false;
    for (const auto &task : group_tasks) {
      const auto mode = task->getSolveMode();
      all_scale_family =
          all_scale_family &&
          (mode == TaskSolveMode::kScale || mode == TaskSolveMode::kScaleElastic);
      all_strict_scale = all_strict_scale && mode == TaskSolveMode::kScale;
      all_min_error = all_min_error && (mode == TaskSolveMode::kMinError);
      all_frame_components = all_frame_components &&
          (task->getType() == TaskType::FRAME_POSITION ||
           task->getType() == TaskType::FRAME_ORIENTATION ||
           (task->getType() == TaskType::FRAME_POSE &&
            (direct_velocity_solve || acceleration_limits_enabled_)) ||
           (direct_velocity_solve &&
            dynamic_cast<const PostureTask *>(task.get()) != nullptr));
      fallback_consistent =
          fallback_consistent &&
          (task->getAllowMinErrorFallback() == fallback_value);
      use_goal_directed_limit_clamp =
          use_goal_directed_limit_clamp || task->usesGoalDirectedLimitClamp();
    }

    const bool combine_group =
        fallback_consistent && (all_scale_family || all_min_error);
    if (combine_group) {
      int total_rows = 0;
      for (const auto &task : group_tasks) {
        total_rows += task->getDimension();
      }
      Eigen::VectorXd combined_goal(total_rows);
      Eigen::MatrixXd combined_jac(total_rows, robot_->nv());
      combined_goal.setZero();
      combined_jac.setZero();
      int offset = 0;
      for (const auto &task : group_tasks) {
        Eigen::VectorXd g = task->getVelocity();
        Eigen::MatrixXd J = velocity_task_jacobian(task, g);
        if (g.rows() > 0) {
          combined_goal.segment(offset, g.rows()) = g;
          combined_jac.block(offset, 0, J.rows(), robot_->nv()) = J;
          offset += static_cast<int>(g.rows());
        }
      }
      goals.push_back(std::move(combined_goal));
      preserve_physical_jacobians.push_back(all_frame_components);
      jacobians.push_back(std::move(combined_jac));
      objective_configs.push_back(ObjectiveSolveConfig{
          current_priority,
          all_min_error ? TaskSolveMode::kMinError : TaskSolveMode::kScale,
          all_min_error ? false : fallback_value,
          use_goal_directed_limit_clamp,
      });
      objective_tasks.push_back(nullptr);
      project_collision_tangent_objective.push_back(all_strict_scale);
      project_elastic_collision_recovery_objective.push_back(
          all_scale_family && !all_strict_scale);
    } else {
      for (const auto &task : group_tasks) {
        goals.push_back(task->getVelocity());
        preserve_physical_jacobians.push_back(
            task->getType() == TaskType::FRAME_POSITION ||
            task->getType() == TaskType::FRAME_ORIENTATION ||
            (task->getType() == TaskType::FRAME_POSE &&
             (direct_velocity_solve || acceleration_limits_enabled_)) ||
            (direct_velocity_solve &&
             dynamic_cast<const PostureTask *>(task.get()) != nullptr));
        jacobians.push_back(velocity_task_jacobian(task, goals.back()));
        // SCALE_ELASTIC is treated as SCALE in the SNS solver;
        // the elastic band mechanism handles the limit expansion.
        const auto effective_mode =
            (task->getSolveMode() == TaskSolveMode::kScaleElastic)
                ? TaskSolveMode::kScale
                : task->getSolveMode();
        objective_configs.push_back(ObjectiveSolveConfig{
            task->getPriority(),
            effective_mode,
            task->getAllowMinErrorFallback(),
            task->usesGoalDirectedLimitClamp(),
        });
        objective_tasks.push_back(task);
        project_collision_tangent_objective.push_back(
            task->getSolveMode() == TaskSolveMode::kScale);
        project_elastic_collision_recovery_objective.push_back(
            task->getSolveMode() == TaskSolveMode::kScaleElastic);
      }
    }

    group_tasks.clear();
  };

  for (const auto &task : tasks_) {
    if (!task->isActive()) {
      continue;
    }
    active_task_count++;
    std::unordered_set<int> task_exclusions;
    for (int idx : task->get_excluded_joint_indices()) {
      if (idx >= 0 && idx < robot_->nv() &&
          task_exclusions.insert(idx).second) {
        task_exclusion_counts[static_cast<std::size_t>(idx)]++;
      }
    }

    const int prio = task->getPriority();
    if (group_tasks.empty()) {
      current_priority = prio;
    } else if (prio != current_priority) {
      flush_group();
      current_priority = prio;
    }

    group_tasks.push_back(task);
  }
  flush_group();

  // Task-local exclusions define ownership of each objective, not global DoF
  // availability. A hard constraint may use a joint whenever at least one
  // active task owns it; only joints excluded by every active task are removed
  // from global constraint rows.
  if (active_task_count > 0) {
    for (int idx = 0; idx < robot_->nv(); ++idx) {
      if (task_exclusion_counts[static_cast<std::size_t>(idx)] ==
          active_task_count) {
        excluded_union.insert(idx);
      }
    }
  }

  // Velocity-to-configuration index mapping (cached; used by
  // position-based velocity constraints).
  const std::vector<int> &velocity_to_config_index =
      velocity_to_config_index_cache();

  if (!pending_velocity_lock_indices_.empty()) {
    const int nv_lock = robot_->nv();
    for (int idx : pending_velocity_lock_indices_) {
      if (idx < 0 || idx >= nv_lock) {
        continue;
      }
      for (auto &J : jacobians) {
        if (J.cols() == nv_lock && J.rows() > 0) {
          J.col(idx).setZero();
        }
      }
    }
  }

  Eigen::MatrixXd contact_P_c;
  const bool use_contact_projection = !contact_frames_.empty();
  if (use_contact_projection) {
    contact_P_c = compute_contact_projector();
    for (auto &J : jacobians) {
      J = J * contact_P_c;
    }
  }

  const Eigen::MatrixXd physical_primary_jacobian =
      jacobians.empty() ? Eigen::MatrixXd{} : jacobians.front();
  // A held position command may need a different, model-consistent local
  // realization. Capture its physical task before proximity/tangent edits.
  Eigen::MatrixXd recovery_jacobian;
  Eigen::VectorXd recovery_goal;
  std::vector<std::shared_ptr<Task>> recovery_tasks;
  bool recovery_has_full_pose = false;
  if (pending_step_validation_dt_.has_value() && !goals.empty() &&
      !acceleration_limits_enabled_ && !use_contact_projection &&
      robot_->nq() == robot_->nv() &&
      !has_active_velocity_centroidal_hard_constraints() &&
      objective_configs.front().solve_mode == TaskSolveMode::kScale) {
    const int priority = objective_configs.front().priority;
    bool pose_only = true;
    for (const auto &task : tasks_) {
      if (!task->isActive() || task->getPriority() != priority) {
        continue;
      }
      const auto type = task->getType();
      recovery_has_full_pose = recovery_has_full_pose || type == TaskType::FRAME_POSE;
      pose_only = pose_only && (type == TaskType::FRAME_POSE ||
                                type == TaskType::FRAME_POSITION ||
                                type == TaskType::FRAME_ORIENTATION);
      recovery_tasks.push_back(task);
    }
    if (pose_only && !recovery_tasks.empty()) {
      recovery_jacobian = jacobians.front();
      recovery_goal = goals.front();
      recovery_tasks.clear();
      for (const auto &task : tasks_) {
        const auto type = task->getType();
        if (task->isActive() &&
            (type == TaskType::FRAME_POSE || type == TaskType::FRAME_POSITION ||
             type == TaskType::FRAME_ORIENTATION)) {
          recovery_tasks.push_back(task);
        }
      }
    }
  }

  const bool min_error_integration_box =
      apply_limits && use_position_limits_ && !objective_configs.empty() &&
      objective_configs.front().solve_mode == TaskSolveMode::kMinError &&
      pending_step_validation_dt_.has_value() &&
      std::isfinite(*pending_step_validation_dt_) &&
      *pending_step_validation_dt_ > 0.0;
  const bool acceleration_position_integration_box =
      apply_limits && use_position_limits_ && acceleration_limits_enabled_ &&
      pending_step_validation_dt_.has_value() &&
      std::isfinite(*pending_step_validation_dt_) &&
      *pending_step_validation_dt_ > 0.0;
  if (min_error_integration_box && !preserve_physical_jacobians.empty()) {
    // The primary residual must use its physical motion model when the actual
    // integration box enforces the permitted joint directions.
    preserve_physical_jacobians.front() = true;
  }
  if (apply_limits && use_position_limits_) {
    clamp_jacobians_near_joint_limits(jacobians, goals, objective_configs,
                                    preserve_physical_jacobians,
                                    velocity_to_config_index);
  }

  // If no active tasks, return early
  if (goals.empty()) {
    result.status = SolverStatus::kSuccess;
    result.solution.resize(robot_->nv(), 0.0);
    result.joint_velocities = Eigen::VectorXd::Zero(robot_->nv());
    result.limits_applied = false;
    last_solution_dq_norm_ = 0.0;
    return result;
  }

  std::optional<ConstraintBlock> joint_limit_non_worsening_result =
      std::nullopt;
  if (apply_limits && use_position_limits_ &&
      runtime_config_.joint_limit_non_worsening_enabled) {
    joint_limit_non_worsening_result = build_joint_limit_non_worsening_rows(
        *robot_, velocity_to_config_index, excluded_union,
        runtime_config_.joint_limit_non_worsening_margin,
        acceleration_limits_enabled_, acceleration_limits_, dt_,
        joint_limit_non_worsening_lower_modes_,
        joint_limit_non_worsening_upper_modes_);
    if (use_contact_projection &&
        joint_limit_non_worsening_result.has_value()) {
      joint_limit_non_worsening_result->jacobian =
          joint_limit_non_worsening_result->jacobian * contact_P_c;
    }
  }

  // Collision constraints are expensive when collision geometry exists.
  std::optional<CollisionConstraintResult> collision_constraint_result =
      std::nullopt;
  if (collision_constraint_.has_value() && collision_constraint_->enabled) {
    if (timing) {
      auto t_collision_start = std::chrono::high_resolution_clock::now();
      collision_constraint_result = compute_collision_constraint();
      result.collision_constraint_time_ms = get_elapsed_ms(t_collision_start);
    } else {
      collision_constraint_result = compute_collision_constraint();
    }
    result.collision_pairs_considered = last_collision_pairs_considered_;
    result.collision_exact_distance_queries =
        last_collision_exact_distance_queries_;
    result.collision_bound_culled_pairs = last_collision_bound_culled_pairs_;
    result.collision_sphere_culled_pairs = last_collision_sphere_culled_pairs_;
    result.collision_budget_exhausted = last_collision_budget_exhausted_;
  }
  if (collision_constraint_result.has_value() && !excluded_union.empty()) {
    zero_excluded_columns(collision_constraint_result->jacobian, excluded_union);
  }
  if (use_contact_projection && collision_constraint_result.has_value()) {
    collision_constraint_result->jacobian =
        collision_constraint_result->jacobian * contact_P_c;
  }

  // Preserve original soft residuals for weighted diagnostics/fallback while
  // composing hard-constraint tangent projections on their Jacobians.
  std::vector<Eigen::VectorXd> constrained_weighted_goals;
  std::vector<Eigen::MatrixXd> constrained_weighted_jacobians;
  bool constraint_projection_applied = false;
  const bool preserve_for_weighted =
      runtime_config_.weighted_advisor_enabled ||
      runtime_config_.weighted_fallback_enabled;

  if (joint_limit_non_worsening_result.has_value()) {
    std::vector<bool> project_limit_tangent_objective;
    project_limit_tangent_objective.reserve(objective_configs.size());
    for (const auto &objective : objective_configs) {
      project_limit_tangent_objective.push_back(
          objective.solve_mode == TaskSolveMode::kScale);
    }
    const auto &limit_rows = *joint_limit_non_worsening_result;
    constraint_projection_applied =
        project_objectives_into_constraint_tangent_space(
            goals, jacobians, project_limit_tangent_objective,
            limit_rows.jacobian, limit_rows.lower_bounds,
            constraint_tolerance_, limit_rows.prefer_exact_feasibility_rows,
            solver_tolerance_, true,
            preserve_for_weighted ? &constrained_weighted_goals : nullptr,
            preserve_for_weighted ? &constrained_weighted_jacobians : nullptr);
  }

  // Project task objectives into the collision tangent space whenever their
  // unconstrained velocity would violate an active collision row. The projected
  // goal is recomputed from the achievable tangent velocity, so SCALE does not
  // collapse on an objective component that was intentionally removed. A task
  // that already satisfies every separating lower bound is left unchanged.
  bool collision_projection_applied = false;
  if (apply_limits && collision_constraint_result.has_value()) {
    const auto &coll = collision_constraint_result.value();
    // Strict SCALE always projects at an active boundary. SCALE_ELASTIC keeps
    // its normal elastic tradeoff unless contact recovery requires a positive
    // separating velocity; projecting it earlier stalls useful tangent motion.
    const std::size_t collision_row_count = std::min(
        last_collision_debug_list_.size(),
        static_cast<std::size_t>(coll.lower_bounds.size()));
    bool contact_recovery_required = false;
    for (std::size_t row = 0; row < collision_row_count; ++row) {
      if (last_collision_debug_list_[row].distance <= kCollisionTolerance &&
          coll.lower_bounds(static_cast<Eigen::Index>(row)) >
              constraint_tolerance_) {
        contact_recovery_required = true;
        break;
      }
    }
    if (contact_recovery_required) {
      for (std::size_t index = 0;
           index < project_collision_tangent_objective.size(); ++index) {
        project_collision_tangent_objective[index] =
            project_collision_tangent_objective[index] ||
            project_elastic_collision_recovery_objective[index];
      }
    }
    collision_projection_applied =
        project_objectives_into_constraint_tangent_space(
            goals, jacobians, project_collision_tangent_objective,
            coll.jacobian, coll.lower_bounds, constraint_tolerance_,
            {}, 0.0, false,
            preserve_for_weighted ? &constrained_weighted_goals : nullptr,
            preserve_for_weighted ? &constrained_weighted_jacobians : nullptr);
    constraint_projection_applied =
        constraint_projection_applied || collision_projection_applied;
  }
  bool use_constrained_weighted_objectives =
      constraint_projection_applied && !constrained_weighted_goals.empty();

  // CoM support-polygon constraint
  std::optional<ComConstraintResult> com_constraint_result = std::nullopt;
  if (com_constraint_.has_value() && com_constraint_->enabled) {
    com_constraint_result = compute_com_constraint();
  }
  if (com_constraint_result.has_value() && !excluded_union.empty()) {
    zero_excluded_columns(com_constraint_result->jacobian, excluded_union);
  }
  if (use_contact_projection && com_constraint_result.has_value()) {
    com_constraint_result->jacobian =
        com_constraint_result->jacobian * contact_P_c;
  }

  std::optional<ComConstraintResult> centroidal_momentum_bounds_result =
      compute_centroidal_momentum_bounds_constraint();
  if (use_contact_projection && centroidal_momentum_bounds_result.has_value()) {
    centroidal_momentum_bounds_result->jacobian =
        centroidal_momentum_bounds_result->jacobian * contact_P_c;
  }
  std::optional<ComConstraintResult> capture_point_constraint_result =
      compute_capture_point_constraint();
  if (use_contact_projection && capture_point_constraint_result.has_value()) {
    capture_point_constraint_result->jacobian =
        capture_point_constraint_result->jacobian * contact_P_c;
  }
  std::optional<ComConstraintResult> velocity_zmp_constraint_result =
      std::nullopt;
  if (explicit_current_dq != nullptr) {
    velocity_zmp_constraint_result =
        compute_velocity_zmp_constraint(*explicit_current_dq);
    if (use_contact_projection && velocity_zmp_constraint_result.has_value()) {
      velocity_zmp_constraint_result->jacobian =
          velocity_zmp_constraint_result->jacobian * contact_P_c;
    }
  }

  // Task normal projection for violated CoM rows:
  // when outside a half-plane, remove task components that increase outward
  // velocity along that half-plane normal. This mirrors collision behavior and
  // keeps tangential/inward motion available instead of globally scaling tasks.
  int com_violated_rows = 0;
  if (apply_limits && com_constraint_result.has_value()) {
    const auto &com = com_constraint_result.value();
    for (int i = 0; i < com.violated_rows.size(); ++i) {
      if (com.violated_rows(i) != 0) {
        com_violated_rows++;
      }
    }
    project_task_jacobians_away_from_violated_rows(
        jacobians, com.jacobian, com.violated_rows,
        /*outward_is_positive_projection=*/true);
    if (use_constrained_weighted_objectives) {
      project_task_jacobians_away_from_violated_rows(
          constrained_weighted_jacobians, com.jacobian, com.violated_rows,
          /*outward_is_positive_projection=*/true);
    }
  }

  // Relative pose constraint
  std::optional<RelativePoseConstraintResult> rel_pose_constraint_result =
      std::nullopt;
  if (relative_pose_constraint_.has_value() &&
      relative_pose_constraint_->enabled) {
    rel_pose_constraint_result = compute_relative_pose_constraint();
  }
  if (rel_pose_constraint_result.has_value() && !excluded_union.empty()) {
    zero_excluded_columns(rel_pose_constraint_result->jacobian, excluded_union);
  }
  int rel_pose_violated_rows = 0;
  if (use_contact_projection && rel_pose_constraint_result.has_value()) {
    rel_pose_constraint_result->jacobian =
        rel_pose_constraint_result->jacobian * contact_P_c;
  }
  if (apply_limits && rel_pose_constraint_result.has_value()) {
    const auto &rpc = rel_pose_constraint_result.value();
    for (int i = 0; i < rpc.violated_rows.size(); ++i) {
      if (rpc.violated_rows(i) != 0) {
        rel_pose_violated_rows++;
      }
    }
    project_task_jacobians_away_from_violated_rows(
        jacobians, rpc.jacobian, rpc.violated_rows,
        /*outward_is_positive_projection=*/true);
    if (use_constrained_weighted_objectives) {
      project_task_jacobians_away_from_violated_rows(
          constrained_weighted_jacobians, rpc.jacobian, rpc.violated_rows,
          /*outward_is_positive_projection=*/true);
    }
  }

  std::optional<LinearVelocityConstraintResult> linear_constraint_result =
      compute_linear_velocity_constraints();
  if (linear_constraint_result.has_value() && !excluded_union.empty()) {
    for (int idx : excluded_union) {
      if (idx >= 0 && idx < linear_constraint_result->jacobian.cols()) {
        linear_constraint_result->jacobian.col(idx).setZero();
      }
    }
  }
  if (use_contact_projection && linear_constraint_result.has_value()) {
    linear_constraint_result->jacobian =
        linear_constraint_result->jacobian * contact_P_c;
  }

  std::optional<LinearVelocityConstraintResult> tight_pose_constraint_result =
      compute_tight_frame_pose_constraints();
  if (tight_pose_constraint_result.has_value() && !excluded_union.empty()) {
    for (int idx : excluded_union) {
      if (idx >= 0 && idx < tight_pose_constraint_result->jacobian.cols()) {
        tight_pose_constraint_result->jacobian.col(idx).setZero();
      }
    }
  }
  if (use_contact_projection && tight_pose_constraint_result.has_value()) {
    tight_pose_constraint_result->jacobian =
        tight_pose_constraint_result->jacobian * contact_P_c;
  }
  if (apply_limits && tight_pose_constraint_result.has_value()) {
    const auto &tpc = tight_pose_constraint_result.value();
    project_task_jacobians_away_from_violated_rows(
        jacobians, tpc.jacobian, tpc.violated_rows,
        /*outward_is_positive_projection=*/true);
    if (use_constrained_weighted_objectives) {
      project_task_jacobians_away_from_violated_rows(
          constrained_weighted_jacobians, tpc.jacobian, tpc.violated_rows,
          /*outward_is_positive_projection=*/true);
    }
  }

  std::optional<LinearVelocityConstraintResult> tight_point_constraint_result =
      compute_tight_point_constraints();
  if (tight_point_constraint_result.has_value() && !excluded_union.empty()) {
    for (int idx : excluded_union) {
      if (idx >= 0 && idx < tight_point_constraint_result->jacobian.cols()) {
        tight_point_constraint_result->jacobian.col(idx).setZero();
      }
    }
  }
  if (use_contact_projection && tight_point_constraint_result.has_value()) {
    tight_point_constraint_result->jacobian =
        tight_point_constraint_result->jacobian * contact_P_c;
  }
  if (apply_limits && tight_point_constraint_result.has_value()) {
    const auto &tpc = tight_point_constraint_result.value();
    project_task_jacobians_away_from_violated_rows(
        jacobians, tpc.jacobian, tpc.violated_rows,
        /*outward_is_positive_projection=*/true);
    if (use_constrained_weighted_objectives) {
      project_task_jacobians_away_from_violated_rows(
          constrained_weighted_jacobians, tpc.jacobian, tpc.violated_rows,
          /*outward_is_positive_projection=*/true);
    }
  }
  std::optional<ConstraintBlock> step_torso_constraint_result = std::nullopt;
  if (pending_step_torso_constraint_.has_value()) {
    const auto &torso_opts = *pending_step_torso_constraint_;
    if (torso_opts.enabled && robot_->has_frame(torso_opts.frame_name)) {
      pinocchio::SE3 torso_pose_bounds_reference = pinocchio::SE3::Identity();
      if (torso_opts.pose_bounds_reference_pose.has_value()) {
        const Eigen::Matrix4d &M = torso_opts.pose_bounds_reference_pose.value();
        torso_pose_bounds_reference =
            pinocchio::SE3(M.block<3, 3>(0, 0), M.block<3, 1>(0, 3));
      } else {
        torso_pose_bounds_reference = robot_->get_frame_pose(torso_opts.frame_name);
      }
      step_torso_constraint_result = build_torso_pose_bound_rows(
          *this, *robot_, torso_opts, torso_pose_bounds_reference, dt_,
          excluded_union);
      if (use_contact_projection && step_torso_constraint_result.has_value()) {
        step_torso_constraint_result->jacobian =
            step_torso_constraint_result->jacobian * contact_P_c;
      }
    }
  }

  std::optional<ConstraintBlock> position_step_priority_constraint_result =
      std::nullopt;
  if (apply_limits && acceleration_limits_enabled_ &&
      acceleration_limits_.size() == robot_->nv() &&
      !pending_position_step_priority_constraints_.empty()) {
    struct PriorityViabilityRow {
      Eigen::RowVectorXd jacobian;
      double lower_bound;
    };
    std::vector<PriorityViabilityRow> rows;
    rows.reserve(2 * pending_position_step_priority_constraints_.size());

    const auto append_priority_block =
        [&](const Eigen::VectorXd &error, const Eigen::MatrixXd &jacobian,
            int offset, double tolerance, double step_dt) {
          if (!std::isfinite(tolerance) || tolerance <= 0.0 ||
              !std::isfinite(step_dt) || step_dt <= 0.0 ||
              offset < 0 || offset + 3 > error.size() ||
              offset + 3 > jacobian.rows()) {
            return;
          }
          const Eigen::Vector3d block_error = error.segment<3>(offset);
          const double error_norm = block_error.norm();
          if (!std::isfinite(error_norm) || error_norm <= 1e-12) {
            return;
          }

          Eigen::RowVectorXd row =
              (block_error / error_norm).transpose() *
              jacobian.middleRows(offset, 3);
          for (int velocity_index : pending_velocity_lock_indices_) {
            if (velocity_index >= 0 && velocity_index < row.size()) {
              row[velocity_index] = 0.0;
            }
          }
          if (use_contact_projection) {
            row *= contact_P_c;
          }
          if (!row.allFinite() || row.squaredNorm() <= 1e-24) {
            return;
          }

          double acceleration_support = 0.0;
          for (int velocity_index = 0; velocity_index < row.size();
               ++velocity_index) {
            const double acceleration_limit =
                acceleration_limits_[velocity_index];
            if (!std::isfinite(acceleration_limit) ||
                acceleration_limit <= 0.0) {
              continue;
            }
            acceleration_support +=
                std::abs(row[velocity_index]) * acceleration_limit;
          }
          if (!std::isfinite(acceleration_support) ||
              acceleration_support <= 0.0) {
            return;
          }

          const double viable_tolerance =
              std::max(0.0, tolerance - constraint_tolerance_);
          const double slack = viable_tolerance - error_norm;
          double lower_bound = 0.0;
          if (slack > 0.0) {
            lower_bound = -discrete_stopping_velocity_limit(
                slack, acceleration_support, step_dt);
          } else {
            lower_bound = (-slack) / step_dt;
          }
          rows.push_back({std::move(row), lower_bound});
        };

    for (const auto &spec : pending_position_step_priority_constraints_) {
      if (!spec.task || !spec.task->isActive()) {
        continue;
      }
      const Eigen::VectorXd error = spec.task->getError();
      const Eigen::MatrixXd jacobian = spec.task->getJacobian();
      if (jacobian.cols() != robot_->nv() || error.size() != jacobian.rows()) {
        continue;
      }
      const TaskType task_type = spec.task->getType();
      if (error.size() >= 6) {
        append_priority_block(error, jacobian, 0, spec.position_tolerance,
                              spec.step_dt);
        append_priority_block(error, jacobian, 3, spec.orientation_tolerance,
                              spec.step_dt);
      } else if (error.size() >= 3) {
        const bool orientation_only =
            task_type == TaskType::FRAME_ORIENTATION;
        append_priority_block(
            error, jacobian, 0,
            orientation_only ? spec.orientation_tolerance
                             : spec.position_tolerance,
            spec.step_dt);
      }
    }

    if (!rows.empty()) {
      ConstraintBlock block;
      block.jacobian =
          Eigen::MatrixXd::Zero(static_cast<int>(rows.size()), robot_->nv());
      block.lower_bounds = Eigen::VectorXd::Zero(static_cast<int>(rows.size()));
      block.upper_bounds = Eigen::VectorXd::Constant(
          static_cast<int>(rows.size()), kUnboundedConstraintLimit);
      for (int row_index = 0; row_index < static_cast<int>(rows.size());
           ++row_index) {
        block.jacobian.row(row_index) =
            rows[static_cast<std::size_t>(row_index)].jacobian;
        block.lower_bounds[row_index] =
            rows[static_cast<std::size_t>(row_index)].lower_bound;
      }
      position_step_priority_constraint_result = std::move(block);
    }
  }

  if (position_step_priority_constraint_result.has_value()) {
    int protected_priority = std::numeric_limits<int>::max();
    for (const auto &spec : pending_position_step_priority_constraints_) {
      if (spec.task && spec.task->isActive()) {
        protected_priority =
            std::min(protected_priority, spec.task->getPriority());
      }
    }
    std::vector<bool> project_priority_tangent_objective;
    project_priority_tangent_objective.reserve(objective_configs.size());
    for (const auto &objective : objective_configs) {
      project_priority_tangent_objective.push_back(
          objective.priority > protected_priority &&
          objective.solve_mode == TaskSolveMode::kScale);
    }
    const auto &priority_rows = *position_step_priority_constraint_result;
    const bool priority_projection_applied =
        project_objectives_into_constraint_tangent_space(
            goals, jacobians, project_priority_tangent_objective,
            priority_rows.jacobian, priority_rows.lower_bounds,
            constraint_tolerance_, {}, 0.0, true,
            preserve_for_weighted ? &constrained_weighted_goals : nullptr,
            preserve_for_weighted ? &constrained_weighted_jacobians : nullptr);
    constraint_projection_applied =
        constraint_projection_applied || priority_projection_applied;
    use_constrained_weighted_objectives =
        constraint_projection_applied && !constrained_weighted_goals.empty();
  }

  // Build constraint matrix
  std::optional<std::chrono::high_resolution_clock::time_point>
      t_constraint_start;
  if (timing) {
    t_constraint_start = std::chrono::high_resolution_clock::now();
  }
  Eigen::MatrixXd C;
  Eigen::VectorXd c_lower, c_upper;

  // For floating-base robots, we need to handle base and joint constraints
  // separately
  int num_constraints = robot_->nv(); // Velocity constraints

  // Sparse position-limit pre-pass: compute per-joint bounds and keep only
  // rows where position limits actually tighten beyond the velocity limit.
  // For joints deep in their range the bound equals [-vel, +vel] and the row
  // is redundant. Omitting it reduces the QP matrix significantly.
  //
  // Performance design: use dense arrays and cheap early-exit to avoid
  // calculate_velocity_box_constraint calls for far-from-limit joints.
  struct SparsePosLimitEntry { int nv_idx; double lower; double upper; };
  std::vector<SparsePosLimitEntry> sparse_pos_limits;
  // Dense per-joint bounds for post-QP clamping and saturation detection.
  // Initialized to "not active" sentinel; filled for active joints only.
  const int nv_sp = robot_->nv();
  constexpr double kNoPosBound = std::numeric_limits<double>::infinity();
  std::vector<double> dense_pos_lower_sp(nv_sp,  kNoPosBound);
  std::vector<double> dense_pos_upper_sp(nv_sp, -kNoPosBound);
  std::vector<double> merged_active_lower_sp(nv_sp, kNoPosBound);
  std::vector<double> merged_active_upper_sp(nv_sp, -kNoPosBound);
  std::vector<double> non_worsening_lower_sp(
      nv_sp, -kUnboundedConstraintLimit);
  std::vector<double> non_worsening_upper_sp(
      nv_sp, kUnboundedConstraintLimit);
  if (!use_contact_projection &&
      joint_limit_non_worsening_result.has_value()) {
    const Eigen::MatrixXd &rows =
        joint_limit_non_worsening_result->jacobian;
    for (int row = 0; row < rows.rows(); ++row) {
      for (int column = 0; column < rows.cols(); ++column) {
        const double coefficient = rows(row, column);
        if (coefficient > 0.5) {
          non_worsening_lower_sp[column] = std::max(
              non_worsening_lower_sp[column],
              joint_limit_non_worsening_result->lower_bounds[row] /
                  coefficient);
          break;
        }
        if (coefficient < -0.5) {
          non_worsening_upper_sp[column] = std::min(
              non_worsening_upper_sp[column],
              joint_limit_non_worsening_result->lower_bounds[row] /
                  coefficient);
          break;
        }
      }
    }
  }

  if (apply_limits && use_position_limits_) {
    // 0.1% of vel_limit — preserves borderline near-limit joints.
    constexpr double kPosBoundActiveFraction = 1e-3;
    constexpr double margin_limit_sp = kPositionLimitMarginEpsilon;
    auto [q_min_sp, q_max_sp] = robot_->get_joint_limits();
    Eigen::VectorXd q_cur_sp = robot_->get_current_configuration();
    const auto &vel_limits_sp = robot_->get_velocity_limits();
    const auto &accel_limits_sp = robot_->get_acceleration_limits();
    // Elastic band effective limits.
    Eigen::VectorXd q_min_eff_sp = q_min_sp;
    Eigen::VectorXd q_max_eff_sp = q_max_sp;
    const bool elastic_sp = elastic_band_config_.enabled &&
                            elastic_band_state_.delta.size() == robot_->nv();
    if (elastic_sp) {
      for (int i = 0; i < nv_sp; ++i) {
        const int qi = (i < static_cast<int>(velocity_to_config_index.size()))
                       ? velocity_to_config_index[i] : kVelocityToConfigUnmapped;
        if (qi == kVelocityToConfigUnmapped || qi >= q_min_sp.size()) continue;
        if (!std::isfinite(q_min_sp[qi]) || !std::isfinite(q_max_sp[qi])) continue;
        q_min_eff_sp[qi] -= elastic_band_state_.delta[i];
        q_max_eff_sp[qi] += elastic_band_state_.delta[i];
      }
    }
    // Dense locked-joint flag (avoid hash lookup per iteration).
    std::vector<bool> is_locked_sp(nv_sp, false);
    for (int idx : pending_velocity_lock_indices_) {
      if (idx >= 0 && idx < nv_sp) is_locked_sp[idx] = true;
    }
    const int fb_dof = robot_->is_floating_base() ? 6 : 0;
    const double dt_safe = std::max(dt_, 1e-6);

    for (int i = 0; i < nv_sp; ++i) {
      double vl = (i < static_cast<int>(vel_limits_sp.size()))
                  ? vel_limits_sp[i] : kUnboundedConstraintLimit;
      double al = (i < static_cast<int>(accel_limits_sp.size()))
                  ? accel_limits_sp[i] : kUnboundedConstraintLimit;
      bool is_locked = is_locked_sp[i];

      if (i < fb_dof) {
        // Floating base: only add when explicit base bounds are set.
        bool has_bound = false;
        double lm = 0.0, um = 0.0;
        if (i < 3 && base_position_lower_.has_value() && base_position_upper_.has_value()) {
          lm = q_cur_sp[i] - base_position_lower_.value()[i] - margin_limit_sp;
          um = base_position_upper_.value()[i] - q_cur_sp[i] - margin_limit_sp;
          has_bound = true;
        } else if (i >= 3 && i < 6 &&
                   base_orientation_lower_.has_value() && base_orientation_upper_.has_value()) {
          lm = q_cur_sp[i] - base_orientation_lower_.value()[i-3] - margin_limit_sp;
          um = base_orientation_upper_.value()[i-3] - q_cur_sp[i] - margin_limit_sp;
          has_bound = true;
        }
        if (!has_bound && !is_locked) continue;
        double lo = -vl, hi = vl;
        if (has_bound) {
          auto [ll, ul] = calculate_velocity_box_constraint(lm, um, vl, al, dt_);
          lo = ll; hi = ul;
        }
        if (is_locked) { lo = 0.0; hi = 0.0; }
        const double tol = kPosBoundActiveFraction * vl;
        if ((lo > -vl + tol) || (hi < vl - tol) || is_locked) {
          sparse_pos_limits.push_back({i, lo, hi});
          dense_pos_lower_sp[i] = lo; dense_pos_upper_sp[i] = hi;
        }
      } else {
        // Regular joint: cheap early-exit before calling calculate_velocity_box_constraint.
        const int qi = (i < static_cast<int>(velocity_to_config_index.size()))
                       ? velocity_to_config_index[i] : kVelocityToConfigUnmapped;
        if (qi == kVelocityToConfigUnmapped || qi >= q_cur_sp.size() ||
            qi >= q_min_eff_sp.size() || !std::isfinite(q_min_eff_sp[qi]) ||
            !std::isfinite(q_max_eff_sp[qi])) {
          if (is_locked) {
            sparse_pos_limits.push_back({i, 0.0, 0.0});
            dense_pos_lower_sp[i] = 0.0; dense_pos_upper_sp[i] = 0.0;
          }
          continue;
        }
        const double lm = q_cur_sp[qi] - q_min_eff_sp[qi] - margin_limit_sp;
        const double um = q_max_eff_sp[qi] - q_cur_sp[qi] - margin_limit_sp;
        // Cheap check: if both margins exceed max possible displacement in one step,
        // bounds = [-vel_limit, vel_limit] → skip (no tightening needed).
        const bool enforce_non_worsening =
            non_worsening_lower_sp[i] > -kUnboundedConstraintLimit ||
            non_worsening_upper_sp[i] < kUnboundedConstraintLimit;
        if (!is_locked && !enforce_non_worsening) {
          const double pos_room_l = lm / dt_safe;
          const double pos_room_u = um / dt_safe;
          const double accel_room_l = (al > 0.0 && std::isfinite(al))
              ? std::sqrt(2.0 * al * std::max(0.0, lm)) : vl;
          const double accel_room_u = (al > 0.0 && std::isfinite(al))
              ? std::sqrt(2.0 * al * std::max(0.0, um)) : vl;
          if (pos_room_l >= vl && accel_room_l >= vl &&
              pos_room_u >= vl && accel_room_u >= vl) {
            continue;  // definitely [-vel_limit, vel_limit] — skip row
          }
        }
        auto [lo, hi] = calculate_velocity_box_constraint(lm, um, vl, al, dt_);
        lo = std::max(lo, non_worsening_lower_sp[i]);
        hi = std::min(hi, non_worsening_upper_sp[i]);
        if (is_locked) { lo = 0.0; hi = 0.0; }
        if (enforce_non_worsening) {
          sparse_pos_limits.push_back({i, lo, hi});
          merged_active_lower_sp[i] = lo;
          merged_active_upper_sp[i] = hi;
          dense_pos_lower_sp[i] = lo;
          dense_pos_upper_sp[i] = hi;
          continue;
        }
        const double tol = kPosBoundActiveFraction * vl;
        if ((lo > -vl + tol) || (hi < vl - tol) || is_locked) {
          sparse_pos_limits.push_back({i, lo, hi});
          dense_pos_lower_sp[i] = lo; dense_pos_upper_sp[i] = hi;
        }
      }
    }
    num_constraints += static_cast<int>(sparse_pos_limits.size());
  }

  Eigen::VectorXd true_integration_lower;
  Eigen::VectorXd true_integration_upper;
  std::vector<int> true_integration_indices;
  if (acceleration_position_integration_box) {
    if (acceleration_limits_.size() != robot_->nv()) {
      result.status = SolverStatus::kInvalidInput;
      result.status_message =
          "position integration requires configured acceleration limits";
      return result;
    }
    const auto [q_min, q_max] = robot_->get_joint_limits();
    const Eigen::VectorXd velocity_limits = robot_->get_velocity_limits();
    const auto &model = robot_->model();
    const double horizon = *pending_step_validation_dt_;
    true_integration_lower = Eigen::VectorXd::Constant(
        robot_->nv(), -kUnboundedConstraintLimit);
    true_integration_upper = Eigen::VectorXd::Constant(
        robot_->nv(), kUnboundedConstraintLimit);
    // Only scalar joints have additive q += dt*v integration. Quaternion and
    // floating-base coordinates must not be treated as scalar joint bounds.
    for (pinocchio::JointIndex joint = 1; joint < model.joints.size(); ++joint) {
      if (model.nqs[joint] != 1 || model.nvs[joint] != 1) {
        continue;
      }
      const int qi = model.idx_qs[joint];
      const int vi = model.idx_vs[joint];
      if (!std::isfinite(q_min[qi]) || !std::isfinite(q_max[qi])) {
        continue;
      }
      true_integration_lower[vi] = std::max(
          -kUnboundedConstraintLimit, (q_min[qi] - q_eval[qi]) / horizon);
      true_integration_upper[vi] = std::min(
          kUnboundedConstraintLimit, (q_max[qi] - q_eval[qi]) / horizon);
      // Match the existing outer realization's inset/stopping policy before
      // solving, so its later scalar limiter cannot change the task image.
      constexpr double kOuterPositionMargin = 1e-4;
      const double velocity_limit =
          vi < velocity_limits.size() && std::isfinite(velocity_limits[vi]) &&
                  velocity_limits[vi] > 0.0
              ? velocity_limits[vi] : kUnboundedConstraintLimit;
      const auto [viability_lower, viability_upper] =
          calculate_velocity_box_constraint(
              q_eval[qi] - q_min[qi] - kOuterPositionMargin,
              q_max[qi] - q_eval[qi] - kOuterPositionMargin, velocity_limit,
              acceleration_limits_[vi], horizon, 0.0, 0.0);
      true_integration_lower[vi] =
          std::max(true_integration_lower[vi], viability_lower);
      true_integration_upper[vi] =
          std::min(true_integration_upper[vi], viability_upper);
      true_integration_indices.push_back(vi);
    }
    if (use_contact_projection) {
      num_constraints += static_cast<int>(true_integration_indices.size());
    }
  }

  if (use_contact_projection &&
      joint_limit_non_worsening_result.has_value()) {
    num_constraints += static_cast<int>(
        joint_limit_non_worsening_result->jacobian.rows());
  }

  if (collision_constraint_result.has_value()) {
    num_constraints +=
        static_cast<int>(collision_constraint_result->jacobian.rows());
  }

  if (com_constraint_result.has_value()) {
    num_constraints +=
        static_cast<int>(com_constraint_result->jacobian.rows());
  }
  if (centroidal_momentum_bounds_result.has_value()) {
    num_constraints += static_cast<int>(
        centroidal_momentum_bounds_result->jacobian.rows());
  }
  if (capture_point_constraint_result.has_value()) {
    num_constraints +=
        static_cast<int>(capture_point_constraint_result->jacobian.rows());
  }
  if (velocity_zmp_constraint_result.has_value()) {
    num_constraints +=
        static_cast<int>(velocity_zmp_constraint_result->jacobian.rows());
  }

  if (rel_pose_constraint_result.has_value()) {
    num_constraints +=
        static_cast<int>(rel_pose_constraint_result->jacobian.rows());
  }
  if (step_torso_constraint_result.has_value()) {
    num_constraints +=
        static_cast<int>(step_torso_constraint_result->jacobian.rows());
  }
  if (position_step_priority_constraint_result.has_value()) {
    num_constraints += static_cast<int>(
        position_step_priority_constraint_result->jacobian.rows());
  }
  if (linear_constraint_result.has_value()) {
    num_constraints +=
        static_cast<int>(linear_constraint_result->jacobian.rows());
  }
  if (tight_pose_constraint_result.has_value()) {
    num_constraints +=
        static_cast<int>(tight_pose_constraint_result->jacobian.rows());
  }
  if (tight_point_constraint_result.has_value()) {
    num_constraints +=
        static_cast<int>(tight_point_constraint_result->jacobian.rows());
  }

  C = Eigen::MatrixXd::Zero(num_constraints, robot_->nv());
  c_lower = Eigen::VectorXd::Zero(num_constraints);
  c_upper = Eigen::VectorXd::Zero(num_constraints);
  Eigen::VectorXd max_softening_factors =
      Eigen::VectorXd::Ones(num_constraints);

  int constraint_idx = 0;

  // Joint velocity constraints
  C.block(constraint_idx, 0, robot_->nv(), robot_->nv()) =
      Eigen::MatrixXd::Identity(robot_->nv(), robot_->nv());

  if (apply_limits && use_velocity_limits_) {
    auto vel_limits = robot_->get_velocity_limits();
    c_lower.segment(constraint_idx, robot_->nv()) = -vel_limits;
    c_upper.segment(constraint_idx, robot_->nv()) = vel_limits;
  } else {
    c_lower.segment(constraint_idx, robot_->nv())
        .setConstant(-kUnboundedConstraintLimit);
    c_upper.segment(constraint_idx, robot_->nv())
        .setConstant(kUnboundedConstraintLimit);
  }
  // Per-joint velocity-limit overrides (contribution knob): shrink the box so the
  // solver recruits other DOFs when a group is throttled. Applied in both
  // branches; hard velocity locks below still take priority.
  for (const auto &kv : joint_velocity_limit_overrides_) {
    const int idx = kv.first;
    if (idx >= 0 && idx < robot_->nv()) {
      c_lower(constraint_idx + idx) = -kv.second;
      c_upper(constraint_idx + idx) = kv.second;
    }
  }
  for (int idx = 0; idx < robot_->nv(); ++idx) {
    if (std::isfinite(merged_active_lower_sp[idx])) {
      c_lower(constraint_idx + idx) =
          std::max(c_lower(constraint_idx + idx),
                   merged_active_lower_sp[idx]);
    }
    if (std::isfinite(merged_active_upper_sp[idx])) {
      c_upper(constraint_idx + idx) =
          std::min(c_upper(constraint_idx + idx),
                   merged_active_upper_sp[idx]);
    }
  }
  for (int idx : pending_velocity_lock_indices_) {
    if (idx >= 0 && idx < robot_->nv()) {
      c_lower(constraint_idx + idx) = 0.0;
      c_upper(constraint_idx + idx) = 0.0;
    }
  }
  if (position_step_seed_corridor_ != nullptr) {
    const auto &corridor = *position_step_seed_corridor_;
    if (!pending_step_validation_dt_.has_value() ||
        !std::isfinite(*pending_step_validation_dt_) ||
        *pending_step_validation_dt_ <= 0.0) {
      result.status = SolverStatus::kInvalidInput;
      result.status_message = "seed corridor requires an actual integration interval";
      result.joint_velocities = Eigen::VectorXd::Zero(robot_->nv());
      result.solution.assign(static_cast<std::size_t>(robot_->nv()), 0.0);
      return result;
    }
    tighten_bounds_with_reference_corridor(
        c_lower, c_upper, corridor.reference_q, q_eval,
        corridor.velocity_limits, corridor.outer_dt,
        *pending_step_validation_dt_, corridor.scalar_velocity_to_config,
        robot_->nv());
    // Do not let legacy input sanitization midpoint-collapse a contradictory
    // intersection outside the original physical velocity box.
    if ((c_lower.head(robot_->nv()).array() >
         c_upper.head(robot_->nv()).array()).any()) {
      result.status = SolverStatus::kInfeasible;
      result.status_message = "seed corridor conflicts with the physical velocity box";
      result.joint_velocities = Eigen::VectorXd::Zero(robot_->nv());
      result.solution.assign(static_cast<std::size_t>(robot_->nv()), 0.0);
      return result;
    }
  }
  // Solve against the actual scalar integration box rather than changing the
  // achieved Cartesian velocity through component clipping afterwards.
  if (min_error_integration_box) {
    const auto [q_min, q_max] = robot_->get_joint_limits();
    const double horizon = *pending_step_validation_dt_;
    for (int vi = 0; vi < robot_->nv(); ++vi) {
      const int qi = velocity_to_config_index[vi];
      if (qi < 0 || qi >= q_eval.size() ||
          !std::isfinite(q_min[qi]) || !std::isfinite(q_max[qi]) ||
          q_eval[qi] < q_min[qi] || q_eval[qi] > q_max[qi] ||
          excluded_union.count(vi) > 0 ||
          std::find(pending_velocity_lock_indices_.begin(),
                    pending_velocity_lock_indices_.end(), vi) !=
              pending_velocity_lock_indices_.end()) {
        continue;
      }
      const double lower = (q_min[qi] + 1e-4 - q_eval[qi]) / horizon;
      const double upper = (q_max[qi] - 1e-4 - q_eval[qi]) / horizon;
      if (lower <= upper) {
        c_lower(constraint_idx + vi) = std::max(c_lower(constraint_idx + vi), lower);
        c_upper(constraint_idx + vi) = std::min(c_upper(constraint_idx + vi), upper);
      }
    }
  }
  if (acceleration_position_integration_box && !use_contact_projection) {
    // Elastic limits may shape task selection, but the accepted position
    // command is integrated against the unchanged true model limits.
    c_lower.head(robot_->nv()) =
        c_lower.head(robot_->nv()).cwiseMax(true_integration_lower);
    c_upper.head(robot_->nv()) =
        c_upper.head(robot_->nv()).cwiseMin(true_integration_upper);
  }
  constraint_idx += robot_->nv();

  // Position-based velocity constraints (sparse: only joints near their limits)
  if (apply_limits && use_position_limits_) {
    // sparse_pos_limits was pre-computed above; fill only those rows.
    for (int k = 0; k < static_cast<int>(sparse_pos_limits.size()); ++k) {
      const auto &e = sparse_pos_limits[k];
      C(constraint_idx + k, e.nv_idx) = 1.0;
      c_lower(constraint_idx + k) = e.lower;
      c_upper(constraint_idx + k) = e.upper;
    }
    constraint_idx += static_cast<int>(sparse_pos_limits.size());
    // Skip the old dense fill block below (it is now a no-op guarded by false).
    if (false) {
    auto [q_min, q_max] = robot_->get_joint_limits();
    Eigen::VectorXd q_current = robot_->get_current_configuration();
    auto vel_limits = robot_->get_velocity_limits();

    // Get acceleration limits from robot model
    Eigen::VectorXd accel_limits = robot_->get_acceleration_limits();

    // velocity_to_config_index was built earlier.

    // For each joint, compute maximum velocity to stay within position limits
    C.block(constraint_idx, 0, robot_->nv(), robot_->nv()) =
        Eigen::MatrixXd::Identity(robot_->nv(), robot_->nv());

    // Safety margin for position-based velocity bounds.  The original
    // 1 mrad value causes the velocity bound to reach exactly zero when
    // the joint is within 1 mrad of its limit, which makes the
    // hierarchical (SNS) solver scale the entire task to zero — even
    // when only one DOF is blocked.  A smaller margin keeps a residual
    // velocity that lets the solver find partial solutions while the
    // post-solve clamp (below) still prevents actual limit violations.
    constexpr double margin_limit = 1e-4;

    // Elastic band: compute effective limits with expansion.
    Eigen::VectorXd q_min_eff = q_min;
    Eigen::VectorXd q_max_eff = q_max;
    const bool elastic_active = elastic_band_config_.enabled &&
                                elastic_band_state_.delta.size() == robot_->nv();
    if (elastic_active) {
      for (int i = 0; i < robot_->nv(); ++i) {
        const int q_idx =
            (i < static_cast<int>(velocity_to_config_index.size()))
                ? velocity_to_config_index[i]
                : kVelocityToConfigUnmapped;
        if (q_idx == kVelocityToConfigUnmapped || q_idx >= q_min.size()) {
          continue;
        }
        if (!std::isfinite(q_min[q_idx]) || !std::isfinite(q_max[q_idx])) {
          continue;
        }
        q_min_eff[q_idx] -= elastic_band_state_.delta[i];
        q_max_eff[q_idx] += elastic_band_state_.delta[i];
      }
    }

    if (robot_->is_floating_base()) {
      // Handle floating-base constraints (first 6 DoFs)
      // Get current base pose error if bounds are set
      if (base_position_lower_.has_value() ||
          base_position_upper_.has_value() ||
          base_orientation_lower_.has_value() ||
          base_orientation_upper_.has_value()) {

        // Get current base position
        Eigen::Vector3d base_pos = q_current.head<3>();

        // Position constraints (first 3 DoFs)
        for (int i = 0; i < 3; ++i) {
          if (base_position_lower_.has_value() &&
              base_position_upper_.has_value()) {
            double lower_margin =
                base_pos[i] - base_position_lower_.value()[i] - margin_limit;
            double upper_margin =
                base_position_upper_.value()[i] - base_pos[i] - margin_limit;

            auto [lower_limit, upper_limit] = calculate_velocity_box_constraint(
                lower_margin, upper_margin, vel_limits[i], accel_limits[i],
                dt_);

            c_lower(constraint_idx + i) = lower_limit;
            c_upper(constraint_idx + i) = upper_limit;
          } else {
            c_lower(constraint_idx + i) = -kUnboundedConstraintLimit;
            c_upper(constraint_idx + i) = kUnboundedConstraintLimit;
          }
        }

        // Orientation constraints (next 3 DoFs)
        for (int i = 3; i < 6; ++i) {
          if (base_orientation_lower_.has_value() &&
              base_orientation_upper_.has_value()) {
            // For orientation, we work directly in velocity space
            c_lower(constraint_idx + i) = std::max(
                base_orientation_lower_.value()[i - 3], -vel_limits[i]);
            c_upper(constraint_idx + i) =
                std::min(base_orientation_upper_.value()[i - 3], vel_limits[i]);
          } else {
            c_lower(constraint_idx + i) = -kUnboundedConstraintLimit;
            c_upper(constraint_idx + i) = kUnboundedConstraintLimit;
          }
        }
      } else {
        // No bounds set, use unlimited
        for (int i = 0; i < 6; ++i) {
          c_lower(constraint_idx + i) = -kUnboundedConstraintLimit;
          c_upper(constraint_idx + i) = kUnboundedConstraintLimit;
        }
      }

      // Handle joint constraints (remaining DoFs)
      for (int i = 6; i < robot_->nv(); ++i) {
        const int q_idx = velocity_to_config_index[i];
        if (q_idx == kVelocityToConfigUnmapped ||
            q_idx >= q_current.size() || q_idx >= q_min.size() ||
            q_idx >= q_max.size()) {
          c_lower(constraint_idx + i) = -kUnboundedConstraintLimit;
          c_upper(constraint_idx + i) = kUnboundedConstraintLimit;
          continue;
        }

        if (!std::isfinite(q_min[q_idx]) || !std::isfinite(q_max[q_idx])) {
          c_lower(constraint_idx + i) = -kUnboundedConstraintLimit;
          c_upper(constraint_idx + i) = kUnboundedConstraintLimit;
          continue;
        }

        // Calculate margins to limits (elastic band expands effective limits)
        double lower_margin = q_current[q_idx] - q_min_eff[q_idx] - margin_limit;
        double upper_margin = q_max_eff[q_idx] - q_current[q_idx] - margin_limit;

        auto [lower_limit, upper_limit] = calculate_velocity_box_constraint(
            lower_margin, upper_margin, vel_limits[i], accel_limits[i], dt_);

        c_lower(constraint_idx + i) = lower_limit;
        c_upper(constraint_idx + i) = upper_limit;
      }
    } else {
      // Fixed-base robot: q-v mapping is only direct for joints with nqs == nvs.
      // Continuous joints (nqs=2, nvs=1) require explicit index mapping.
      for (int i = 0; i < robot_->nv(); ++i) {
        const int q_idx = velocity_to_config_index[i];
        if (q_idx == kVelocityToConfigUnmapped ||
            q_idx >= q_current.size() || q_idx >= q_min.size() ||
            q_idx >= q_max.size()) {
          c_lower(constraint_idx + i) = -kUnboundedConstraintLimit;
          c_upper(constraint_idx + i) = kUnboundedConstraintLimit;
          continue;
        }

        if (!std::isfinite(q_min[q_idx]) || !std::isfinite(q_max[q_idx])) {
          c_lower(constraint_idx + i) = -kUnboundedConstraintLimit;
          c_upper(constraint_idx + i) = kUnboundedConstraintLimit;
          continue;
        }

        // Calculate margins to limits (elastic band expands effective limits)
        double lower_margin = q_current[q_idx] - q_min_eff[q_idx] - margin_limit;
        double upper_margin = q_max_eff[q_idx] - q_current[q_idx] - margin_limit;

        auto [lower_limit, upper_limit] = calculate_velocity_box_constraint(
            lower_margin, upper_margin, vel_limits[i], accel_limits[i], dt_);

        c_lower(constraint_idx + i) = lower_limit;
        c_upper(constraint_idx + i) = upper_limit;
      }
    }
    for (int idx : pending_velocity_lock_indices_) {
      if (idx >= 0 && idx < robot_->nv()) {
        c_lower(constraint_idx + idx) = 0.0;
        c_upper(constraint_idx + idx) = 0.0;
      }
    }
    constraint_idx += robot_->nv();
    }  // end if (false) — old dense position-limit block (disabled)
  }

  if (use_contact_projection &&
      joint_limit_non_worsening_result.has_value()) {
    constraint_idx = append_constraint_block(
        C, c_lower, c_upper, constraint_idx, robot_->nv(),
        joint_limit_non_worsening_result->jacobian,
        joint_limit_non_worsening_result->lower_bounds,
        joint_limit_non_worsening_result->upper_bounds);
  }

  if (acceleration_position_integration_box && use_contact_projection) {
    // The backend variable is projected before publication. Bound P_c*v,
    // rather than assuming a box on the unprojected variable bounds its image.
    for (int vi : true_integration_indices) {
      C.row(constraint_idx) = contact_P_c.row(vi);
      c_lower[constraint_idx] = true_integration_lower[vi];
      c_upper[constraint_idx] = true_integration_upper[vi];
      ++constraint_idx;
    }
  }

  if (collision_constraint_result.has_value()) {
    constraint_idx = append_constraint_block(
        C, c_lower, c_upper, constraint_idx, robot_->nv(),
        collision_constraint_result->jacobian,
        collision_constraint_result->lower_bounds,
        collision_constraint_result->upper_bounds);
  }

  if (com_constraint_result.has_value()) {
    constraint_idx = append_constraint_block(
        C, c_lower, c_upper, constraint_idx, robot_->nv(),
        com_constraint_result->jacobian, com_constraint_result->lower_bounds,
        com_constraint_result->upper_bounds);
  }
  if (centroidal_momentum_bounds_result.has_value()) {
    constraint_idx = append_constraint_block(
        C, c_lower, c_upper, constraint_idx, robot_->nv(),
        centroidal_momentum_bounds_result->jacobian,
        centroidal_momentum_bounds_result->lower_bounds,
        centroidal_momentum_bounds_result->upper_bounds);
  }
  if (capture_point_constraint_result.has_value()) {
    constraint_idx = append_constraint_block(
        C, c_lower, c_upper, constraint_idx, robot_->nv(),
        capture_point_constraint_result->jacobian,
        capture_point_constraint_result->lower_bounds,
        capture_point_constraint_result->upper_bounds);
  }
  if (velocity_zmp_constraint_result.has_value()) {
    constraint_idx = append_constraint_block(
        C, c_lower, c_upper, constraint_idx, robot_->nv(),
        velocity_zmp_constraint_result->jacobian,
        velocity_zmp_constraint_result->lower_bounds,
        velocity_zmp_constraint_result->upper_bounds);
  }

  if (rel_pose_constraint_result.has_value()) {
    constraint_idx = append_constraint_block(
        C, c_lower, c_upper, constraint_idx, robot_->nv(),
        rel_pose_constraint_result->jacobian,
        rel_pose_constraint_result->lower_bounds,
        rel_pose_constraint_result->upper_bounds);
  }

  if (step_torso_constraint_result.has_value()) {
    constraint_idx = append_constraint_block(
        C, c_lower, c_upper, constraint_idx, robot_->nv(),
        step_torso_constraint_result->jacobian,
        step_torso_constraint_result->lower_bounds,
        step_torso_constraint_result->upper_bounds);
  }
  if (position_step_priority_constraint_result.has_value()) {
    constraint_idx = append_constraint_block(
        C, c_lower, c_upper, constraint_idx, robot_->nv(),
        position_step_priority_constraint_result->jacobian,
        position_step_priority_constraint_result->lower_bounds,
        position_step_priority_constraint_result->upper_bounds);
  }
  if (linear_constraint_result.has_value()) {
    const auto &lc = linear_constraint_result.value();
    int rows = static_cast<int>(lc.jacobian.rows());
    C.block(constraint_idx, 0, rows, robot_->nv()) = lc.jacobian;
    c_lower.segment(constraint_idx, rows) = lc.lower_bounds;
    c_upper.segment(constraint_idx, rows) = lc.upper_bounds;
    constraint_idx += rows;
  }
  if (tight_pose_constraint_result.has_value()) {
    const auto &tpc = tight_pose_constraint_result.value();
    int rows = static_cast<int>(tpc.jacobian.rows());
    C.block(constraint_idx, 0, rows, robot_->nv()) = tpc.jacobian;
    c_lower.segment(constraint_idx, rows) = tpc.lower_bounds;
    c_upper.segment(constraint_idx, rows) = tpc.upper_bounds;
    max_softening_factors.segment(constraint_idx, rows).setConstant(4.0);
    constraint_idx += rows;
  }
  if (tight_point_constraint_result.has_value()) {
    const auto &tpc = tight_point_constraint_result.value();
    int rows = static_cast<int>(tpc.jacobian.rows());
    C.block(constraint_idx, 0, rows, robot_->nv()) = tpc.jacobian;
    c_lower.segment(constraint_idx, rows) = tpc.lower_bounds;
    c_upper.segment(constraint_idx, rows) = tpc.upper_bounds;
    max_softening_factors.segment(constraint_idx, rows).setConstant(4.0);
    constraint_idx += rows;
  }

  if (timing && t_constraint_start.has_value()) {
    result.constraint_setup_time_ms = get_elapsed_ms(*t_constraint_start);
  }

  // Acceleration-level constraint cascading: tighten the first nv rows
  // of c_lower/c_upper (the velocity limit block) based on previous tick's
  // velocity and the configured acceleration limits.
  const bool acceleration_corridor_active =
      acceleration_limits_enabled_ &&
      (position_step_call_depth_ == 0 ||
       pending_position_step_acceleration_limits_) &&
      previous_dq_.size() == robot_->nv() &&
      acceleration_limits_.size() == robot_->nv();
  const Eigen::VectorXd velocity_lower_before_acceleration =
      c_lower.head(robot_->nv());
  const Eigen::VectorXd velocity_upper_before_acceleration =
      c_upper.head(robot_->nv());
  if (acceleration_corridor_active) {
    const int nv = robot_->nv();
    for (int i = 0; i < nv; ++i) {
      const double accel_step =
          acceleration_velocity_step(acceleration_limits_[i], dt_);
      double a_lb = previous_dq_[i] - accel_step;
      double a_ub = previous_dq_[i] + accel_step;
      c_lower[i] = std::max(c_lower[i], a_lb);
      c_upper[i] = std::min(c_upper[i], a_ub);
    }
  }
  if (acceleration_position_integration_box && acceleration_corridor_active &&
      !use_contact_projection) {
    for (int vi : true_integration_indices) {
      const double lower = velocity_lower_before_acceleration[vi];
      const double upper = velocity_upper_before_acceleration[vi];
      if (c_lower[vi] <= c_upper[vi] || !std::isfinite(lower) ||
          !std::isfinite(upper) || lower > upper) {
        continue;
      }
      const double face = std::clamp(previous_dq_[vi], lower, upper);
      const double step =
          acceleration_velocity_step(acceleration_limits_[vi], dt_);
      const double change = std::abs(face - previous_dq_[vi]);
      const double tolerance = acceleration_velocity_comparison_tolerance(
          previous_dq_[vi], face, step);
      if (std::isfinite(face) && std::isfinite(step) && std::isfinite(change) &&
          change <= step + tolerance) {
        // An applied integration roundtrip can place the previous velocity an
        // ulp beyond a braking face. Keep the physical face, using the same
        // dimensioned comparison as outer realization; do not midpoint a gap.
        c_lower[vi] = face;
        c_upper[vi] = face;
      }
    }
  }
  const Eigen::VectorXd position_acceleration_lower =
      acceleration_position_integration_box
          ? Eigen::VectorXd(c_lower.head(robot_->nv())) : Eigen::VectorXd();
  const Eigen::VectorXd position_acceleration_upper =
      acceleration_position_integration_box
          ? Eigen::VectorXd(c_upper.head(robot_->nv())) : Eigen::VectorXd();

  // The active-limit viability interval is a hard safety invariant. If an
  // externally synchronized state leaves stale acceleration history outside
  // that interval, select the nearest safe boundary instead of letting the
  // generic interval sanitizer move the command back toward the limit.
  if (acceleration_corridor_active && !use_contact_projection &&
      joint_limit_non_worsening_result.has_value() &&
      previous_dq_.size() == robot_->nv()) {
    for (int i = 0; i < robot_->nv(); ++i) {
      if (!std::isfinite(merged_active_lower_sp[i]) ||
          !std::isfinite(merged_active_upper_sp[i]) ||
          c_lower[i] <= c_upper[i]) {
        continue;
      }
      const double safe_velocity = std::clamp(
          previous_dq_[i], merged_active_lower_sp[i],
          merged_active_upper_sp[i]);
      c_lower[i] = safe_velocity;
      c_upper[i] = safe_velocity;
    }
  }

  // Contact projection turns a scalar joint-limit row into a coupled row. If
  // stale acceleration history makes that row unreachable, restore the
  // pre-acceleration velocity interval only for its participating joints. The
  // projected contact row and active limit remain hard in the backend solve.
  if (acceleration_corridor_active && use_contact_projection &&
      joint_limit_non_worsening_result.has_value()) {
    const auto &active_limits = *joint_limit_non_worsening_result;
    for (int row = 0; row < active_limits.jacobian.rows(); ++row) {
      double maximum_value = 0.0;
      for (int column = 0; column < active_limits.jacobian.cols(); ++column) {
        const double coefficient = active_limits.jacobian(row, column);
        maximum_value += coefficient >= 0.0
                             ? coefficient * c_upper[column]
                             : coefficient * c_lower[column];
      }
      if (maximum_value >=
          active_limits.lower_bounds[row] - constraint_tolerance_) {
        continue;
      }
      for (int column = 0; column < active_limits.jacobian.cols(); ++column) {
        if (std::abs(active_limits.jacobian(row, column)) <= 1e-12) {
          continue;
        }
        c_lower[column] = velocity_lower_before_acceleration[column];
        c_upper[column] = velocity_upper_before_acceleration[column];
      }
    }
  }

  if (position_step_seed_corridor_ != nullptr) {
    // Stale acceleration-history repair must not replace the immutable seed,
    // velocity-override or lock intersection with an out-of-corridor midpoint.
    c_lower.head(robot_->nv()) = c_lower.head(robot_->nv()).cwiseMax(
        velocity_lower_before_acceleration);
    c_upper.head(robot_->nv()) = c_upper.head(robot_->nv()).cwiseMin(
        velocity_upper_before_acceleration);
    if ((c_lower.head(robot_->nv()).array() >
         c_upper.head(robot_->nv()).array()).any()) {
      result.status = SolverStatus::kInfeasible;
      result.status_message = "seed corridor conflicts with the physical velocity interval";
      result.joint_velocities = Eigen::VectorXd::Zero(robot_->nv());
      result.solution.assign(static_cast<std::size_t>(robot_->nv()), 0.0);
      return result;
    }
  }
  if (acceleration_position_integration_box) {
    // A stale-history repair must not erase the original acceleration/true
    // integration intersection, and sanitization must not midpoint-collapse it.
    c_lower.head(robot_->nv()) = c_lower.head(robot_->nv()).cwiseMax(
        position_acceleration_lower);
    c_upper.head(robot_->nv()) = c_upper.head(robot_->nv()).cwiseMin(
        position_acceleration_upper);
    if ((c_lower.head(robot_->nv()).array() >
         c_upper.head(robot_->nv()).array()).any() ||
        (true_integration_lower.array() >
         true_integration_upper.array()).any()) {
      result.status = SolverStatus::kInfeasible;
      result.status_message =
          "true integration box conflicts with the physical velocity interval";
      result.joint_velocities = Eigen::VectorXd::Zero(robot_->nv());
      result.solution.assign(static_cast<std::size_t>(robot_->nv()), 0.0);
      return result;
    }
  }
  sanitize_solver_inputs(goals, jacobians, C, c_lower, c_upper);
  const bool has_soft_rows = max_softening_factors.maxCoeff() > 1.0;
  if (has_soft_rows) {
    const int nv = robot_->nv();
    Eigen::VectorXd dq_lower_seed = Eigen::VectorXd::Constant(nv, -1e10);
    Eigen::VectorXd dq_upper_seed = Eigen::VectorXd::Constant(nv, 1e10);
    for (int i = 0; i < nv && i < c_lower.size() && i < c_upper.size(); ++i) {
      dq_lower_seed(i) = c_lower(i);
      dq_upper_seed(i) = c_upper(i);
    }
    if (!has_rowwise_interval_feasibility(C, c_lower, c_upper, dq_lower_seed,
                                          dq_upper_seed,
                                          constraint_tolerance_)) {
      const double pre_relax_scale =
          std::max(1.0, max_softening_factors.maxCoeff());
      apply_constraint_softening_in_place(c_lower, c_upper,
                                          max_softening_factors,
                                          pre_relax_scale);
    }
  }

  // Configure solver
  VelocitySolverConfig config;
  config.epsilon = constraint_tolerance_;
  config.precision_threshold = tight_tolerance_;
  config.iteration_limit = max_iterations_;
  config.magnitude_limit = norm_threshold_;
  config.stall_detection_count = max_zero_scale_iterations_;
  if (has_soft_rows) {
    config.stall_detection_count = std::max(5, max_zero_scale_iterations_);
  }
  config.regularization_config.epsilon = solver_tolerance_;
  config.regularization_config.regularization_factor = damping_;
  config.refine_partial_scale = runtime_config_.refine_partial_scale;

  WeightedAdvisoryResult advisory;
  const auto &advisor_goals = use_constrained_weighted_objectives
                                  ? constrained_weighted_goals
                                  : goals;
  const auto &advisor_jacobians = use_constrained_weighted_objectives
                                      ? constrained_weighted_jacobians
                                      : jacobians;
  auto compute_weighted_advisory = [&]() -> WeightedAdvisoryResult {
    auto weight_at_priority = [&](int priority) -> double {
      double sum = 0.0;
      int count = 0;
      for (const auto &task : tasks_) {
        if (!task || !task->isActive() || task->getPriority() != priority) {
          continue;
        }
        if (std::isfinite(task->getWeight()) && task->getWeight() >= 0.0) {
          sum += task->getWeight();
          ++count;
        }
      }
      return count > 0 ? sum / static_cast<double>(count) : 1.0;
    };
    auto type_at_priority = [&](int priority) -> TaskType {
      bool saw_position = false;
      bool saw_orientation = false;
      bool saw_pose = false;
      for (const auto &task : tasks_) {
        if (!task || !task->isActive() || task->getPriority() != priority) {
          continue;
        }
        const TaskType type = task->getType();
        saw_position = saw_position || type == TaskType::FRAME_POSITION;
        saw_orientation = saw_orientation || type == TaskType::FRAME_ORIENTATION;
        saw_pose = saw_pose || type == TaskType::FRAME_POSE;
      }
      if (saw_pose) {
        return TaskType::FRAME_POSE;
      }
      if (saw_position && !saw_orientation) {
        return TaskType::FRAME_POSITION;
      }
      if (saw_orientation && !saw_position) {
        return TaskType::FRAME_ORIENTATION;
      }
      return TaskType::POSTURE;
    };

    if (!runtime_config_.enable_advisor_scale_adapt) {
      advisor_scale_current_ =
          sanitize_advisor_scale(runtime_config_.advisor_position_weight_scale);
    }
    double position_weight_scale =
        runtime_config_.enable_advisor_scale_adapt
            ? advisor_scale_current_
            : sanitize_advisor_scale(
                  runtime_config_.advisor_position_weight_scale);
    const double advisor_scale_min =
        std::isfinite(runtime_config_.advisor_scale_min)
            ? runtime_config_.advisor_scale_min
            : 0.25;
    const double advisor_scale_max =
        std::isfinite(runtime_config_.advisor_scale_max)
            ? runtime_config_.advisor_scale_max
            : 8.0;
    if (runtime_config_.enable_advisor_scale_adapt) {
      const double lo =
          std::max(0.0, std::min(advisor_scale_min, advisor_scale_max));
      const double hi =
          std::max(lo, std::max(advisor_scale_min, advisor_scale_max));
      position_weight_scale = std::clamp(position_weight_scale, lo, hi);
      advisor_scale_current_ = position_weight_scale;
    }
    double orientation_weight_scale =
        runtime_config_.advisor_orientation_weight_scale;
    if (!std::isfinite(orientation_weight_scale) ||
        orientation_weight_scale < 0.0) {
      orientation_weight_scale = 1.0;
    }

    auto sanitized_task_weight = [](const std::shared_ptr<Task> &task) {
      if (!task || !std::isfinite(task->getWeight()) ||
          task->getWeight() < 0.0) {
        return 1.0;
      }
      return task->getWeight();
    };
    auto make_task_row_weights = [&](const std::shared_ptr<Task> &task) {
      Eigen::VectorXd row_weights;
      if (!task) {
        return row_weights;
      }
      const int dimension = task->getDimension();
      if (dimension <= 0) {
        return row_weights;
      }
      row_weights =
          Eigen::VectorXd::Constant(dimension, sanitized_task_weight(task));
      const TaskType type = task->getType();
      if (type == TaskType::FRAME_POSITION) {
        row_weights.array() *= position_weight_scale;
      } else if (type == TaskType::FRAME_ORIENTATION) {
        row_weights.array() *= orientation_weight_scale;
      } else if (type == TaskType::FRAME_POSE && dimension == 6) {
        row_weights.head<3>().array() *= position_weight_scale;
        row_weights.tail<3>().array() *= orientation_weight_scale;
      }
      return row_weights;
    };
    auto row_weights_at_priority = [&](int priority) {
      int total_rows = 0;
      for (const auto &task : tasks_) {
        if (task && task->isActive() && task->getPriority() == priority) {
          total_rows += task->getDimension();
        }
      }
      Eigen::VectorXd row_weights;
      if (total_rows <= 0) {
        return row_weights;
      }
      row_weights.resize(total_rows);
      int offset = 0;
      for (const auto &task : tasks_) {
        if (!task || !task->isActive() || task->getPriority() != priority) {
          continue;
        }
        const Eigen::VectorXd task_weights = make_task_row_weights(task);
        if (task_weights.size() == 0) {
          continue;
        }
        row_weights.segment(offset, task_weights.size()) = task_weights;
        offset += static_cast<int>(task_weights.size());
      }
      if (offset != total_rows) {
        row_weights.conservativeResize(offset);
      }
      return row_weights;
    };

    std::vector<double> advisor_weights;
    std::vector<Eigen::VectorXd> advisor_row_weights;
    advisor_weights.reserve(advisor_goals.size());
    advisor_row_weights.resize(advisor_goals.size());
    for (size_t i = 0; i < objective_configs.size(); ++i) {
      double w = 1.0;
      TaskType type = TaskType::POSTURE;
      if (i < objective_tasks.size() && objective_tasks[i]) {
        w = objective_tasks[i]->getWeight();
        type = objective_tasks[i]->getType();
        advisor_row_weights[i] = make_task_row_weights(objective_tasks[i]);
      } else {
        w = weight_at_priority(objective_configs[i].priority);
        type = type_at_priority(objective_configs[i].priority);
        advisor_row_weights[i] =
            row_weights_at_priority(objective_configs[i].priority);
      }
      if (!std::isfinite(w) || w < 0.0) {
        w = 1.0;
      }
      if (type == TaskType::FRAME_POSITION) {
        w *= position_weight_scale;
      } else if (type == TaskType::FRAME_ORIENTATION) {
        w *= orientation_weight_scale;
      }
      advisor_weights.push_back(w);
    }

    WeightedAdvisoryResult out = compute_constrained_weighted_advisory(
        advisor_goals, advisor_jacobians, C, c_lower, c_upper, advisor_weights,
        config, advisor_row_weights);
    if (!out.available) {
      return out;
    }

    result.weighted_advisory_available = true;
    result.weighted_advisory_v_norm = out.v_norm;
    result.weighted_advisory_condition_number = out.condition_number;

    double position_error = std::numeric_limits<double>::quiet_NaN();
    double orientation_error = std::numeric_limits<double>::quiet_NaN();
    for (size_t i = 0; i < objective_configs.size() &&
                       i < out.per_objective_error.size();
         ++i) {
      TaskType type = TaskType::POSTURE;
      if (i < objective_tasks.size() && objective_tasks[i]) {
        type = objective_tasks[i]->getType();
      } else {
        type = type_at_priority(objective_configs[i].priority);
      }
      if (type == TaskType::FRAME_POSITION &&
          !std::isfinite(position_error)) {
        position_error = out.per_objective_error[i];
      } else if (type == TaskType::FRAME_ORIENTATION &&
                 !std::isfinite(orientation_error)) {
        orientation_error = out.per_objective_error[i];
      } else if (type == TaskType::FRAME_POSE) {
        const auto &J = advisor_jacobians[i];
        const auto &b = advisor_goals[i];
        if (J.rows() >= 6 && b.rows() >= 6 && out.v.size() == J.cols()) {
          const Eigen::VectorXd residual = J * out.v - b;
          if (i < objective_tasks.size() && objective_tasks[i]) {
            if (!std::isfinite(position_error)) {
              position_error = residual.head(3).norm();
            }
            if (!std::isfinite(orientation_error)) {
              orientation_error = residual.tail(3).norm();
            }
          } else {
            Eigen::Index offset = 0;
            for (const auto &task : tasks_) {
              if (!task || !task->isActive() ||
                  task->getPriority() != objective_configs[i].priority) {
                continue;
              }
              const Eigen::Index dim = task->getDimension();
              if (offset + dim > residual.size()) {
                break;
              }
              if (task->getType() == TaskType::FRAME_POSITION &&
                  !std::isfinite(position_error)) {
                position_error = residual.segment(offset, dim).norm();
              } else if (task->getType() == TaskType::FRAME_ORIENTATION &&
                         !std::isfinite(orientation_error)) {
                orientation_error = residual.segment(offset, dim).norm();
              } else if (task->getType() == TaskType::FRAME_POSE && dim >= 6) {
                if (!std::isfinite(position_error)) {
                  position_error = residual.segment(offset, 3).norm();
                }
                if (!std::isfinite(orientation_error)) {
                  orientation_error = residual.segment(offset + dim - 3, 3).norm();
                }
              }
              offset += dim;
            }
          }
        }
      }
    }
    result.weighted_advisory_pos_task_error_norm = position_error;
    result.weighted_advisory_ori_task_error_norm = orientation_error;
    result.advisor_position_weight_scale_current = position_weight_scale;
    result.advisor_scale_adapt_active = false;
    if (runtime_config_.enable_advisor_scale_adapt &&
        std::isfinite(position_error) && std::isfinite(orientation_error)) {
      const double denom = position_error + orientation_error;
      if (denom > 1e-12) {
        advisor_scale_ratio_sum_ += position_error / denom;
        advisor_scale_epoch_time_s_ += std::max(dt_, 1e-9);
        ++advisor_scale_sample_count_;
      }
      const double epoch_s =
          (std::isfinite(runtime_config_.advisor_scale_epoch_s) &&
           runtime_config_.advisor_scale_epoch_s > 0.0)
              ? runtime_config_.advisor_scale_epoch_s
              : 0.1;
      if (advisor_scale_epoch_time_s_ >= epoch_s &&
          advisor_scale_sample_count_ > 0) {
        const double avg_ratio =
            advisor_scale_ratio_sum_ /
            static_cast<double>(advisor_scale_sample_count_);
        const double target_ratio = std::clamp(
            runtime_config_.advisor_scale_adapt_target_ratio, 0.0, 1.0);
        const double ki =
            std::isfinite(runtime_config_.advisor_scale_adapt_ki)
                ? runtime_config_.advisor_scale_adapt_ki
                : 0.0;
        const double lo =
            std::max(0.0, std::min(advisor_scale_min, advisor_scale_max));
        const double hi =
            std::max(lo, std::max(advisor_scale_min, advisor_scale_max));
        const double exponent =
            std::clamp(ki * (avg_ratio - target_ratio), -1.0, 1.0);
        advisor_scale_current_ =
            std::clamp(advisor_scale_current_ * std::exp(exponent), lo, hi);
        advisor_scale_ratio_sum_ = 0.0;
        advisor_scale_epoch_time_s_ = 0.0;
        advisor_scale_sample_count_ = 0;
        result.advisor_position_weight_scale_current = advisor_scale_current_;
        result.advisor_scale_adapt_active = true;
      }
    }
    return out;
  };

  if (runtime_config_.weighted_advisor_enabled && !goals.empty()) {
    advisory = compute_weighted_advisory();
  }

  warm_start_selector_cache_.reset();
  warm_start_constraint_rows_ = -1;
  // Soft joint-space metric (contribution knob): change of variables dq = D u with
  // D = diag(1/sqrt(w)). Solve the existing prioritized/SNS problem in u-space on
  // scaled copies (J D, C D), then un-scale the solution dq = D u. This yields the
  // weighted least-norm solution while keeping the EE task and the true-space
  // velocity/position limits intact (bounds are unchanged; columns carry D).
  const bool metric_active =
      joint_metric_col_scale_.size() == robot_->nv();
  std::vector<Eigen::MatrixXd> metric_jacobians;
  Eigen::MatrixXd metric_C;
  const std::vector<Eigen::MatrixXd> *solve_jacobians = &jacobians;
  const Eigen::MatrixXd *solve_C = &C;
  if (metric_active) {
    const auto D = joint_metric_col_scale_.asDiagonal();
    metric_jacobians.reserve(jacobians.size());
    for (const auto &J : jacobians) metric_jacobians.push_back(J * D);
    metric_C = C * D;
    solve_jacobians = &metric_jacobians;
    solve_C = &metric_C;
  }
  Eigen::MatrixXd protected_stationary_primary = physical_primary_jacobian;
  if (metric_active)
    protected_stationary_primary =
        physical_primary_jacobian * joint_metric_col_scale_.asDiagonal();
  auto backend_result = solveHierarchicalLinearSystemEigen(
      goals, {}, *solve_jacobians, *solve_C, c_lower, c_upper, config,
      objective_configs, has_soft_rows ? &max_softening_factors : nullptr,
      detail::HierarchicalLinearSolverExecutionOptions{
          detail::HierarchicalLinearSolverPolicy::kLegacyVelocity, 0.0, false,
          &protected_stationary_primary, position_step_call_depth_ == 0});
  if (metric_active &&
      static_cast<int>(backend_result.solution.size()) == robot_->nv()) {
    for (int j = 0; j < robot_->nv(); ++j)
      backend_result.solution[j] *= joint_metric_col_scale_(j);
  }
  bool backend_solution_non_finite = false;
  for (double v : backend_result.solution) {
    if (!std::isfinite(v)) {
      backend_solution_non_finite = true;
      break;
    }
  }
  const bool backend_non_finite_input =
      backend_result.status == SolverStatus::kNonFiniteInput ||
      backend_solution_non_finite;
  const bool backend_generated_non_finite =
      backend_result.status == SolverStatus::kNumericalError &&
      backend_result.status_message ==
          "non-finite values generated in solver state";
  if (backend_non_finite_input) {
    backend_result.status = SolverStatus::kNonFiniteInput;
    backend_result.status_message =
        backend_solution_non_finite
            ? "backend returned non-finite velocity entries; using zero velocity step"
            : "backend produced non-finite internal state; using zero velocity step";
    backend_result.solution.assign(static_cast<size_t>(robot_->nv()), 0.0);
    backend_result.final_error = 0.0;
  } else if (backend_generated_non_finite) {
    backend_result.status = SolverStatus::kNoProgress;
    backend_result.status_message =
        "backend generated non-finite internal state; using zero velocity step";
    backend_result.solution.assign(static_cast<size_t>(robot_->nv()), 0.0);
    backend_result.final_error = 0.0;
  }
  if (use_contact_projection &&
      static_cast<int>(backend_result.solution.size()) == robot_->nv()) {
    Eigen::Map<Eigen::VectorXd> dq_projected(backend_result.solution.data(),
                                             backend_result.solution.size());
    dq_projected = contact_P_c * dq_projected;
  }
  const double primary_goal_norm = goals.empty() ? 0.0 : goals[0].norm();

  auto classified_velocity = classify_velocity_outcome(
      backend_result.status, backend_result.status_message,
      backend_result.task_scales, primary_goal_norm,
      backend_result.task_modes_effective, backend_result.task_used_fallback);
  auto clamp_final_velocity_candidate = [&](Eigen::VectorXd &candidate) {
    if (apply_limits && c_lower.size() >= robot_->nv() &&
        !use_contact_projection) {
      clamp_joint_velocity_solution_in_place(
          candidate, c_lower, c_upper, false, robot_->nv());
      if (use_position_limits_) {
        const int n_dq = static_cast<int>(candidate.size());
        const int n_dens = static_cast<int>(dense_pos_lower_sp.size());
        const int n_clamp = std::min(n_dq, n_dens);
        for (int k = 0; k < n_clamp; ++k) {
          if (dense_pos_lower_sp[k] < kNoPosBound)
            candidate[k] = std::max(candidate[k], dense_pos_lower_sp[k]);
          if (dense_pos_upper_sp[k] > -kNoPosBound)
            candidate[k] = std::min(candidate[k], dense_pos_upper_sp[k]);
        }
      }
    }
  };
  auto contact_velocity_norm = [&](const Eigen::VectorXd &candidate) {
    double max_norm = 0.0;
    for (const auto &cfg : contact_frames_) {
      const Matrix6Xd J_full = robot_->get_frame_jacobian(cfg.frame_name);
      Eigen::VectorXd v;
      if (cfg.type == ContactType::kPointContact) {
        v = J_full.topRows(3) * candidate;
      } else {
        v = J_full * candidate;
      }
      max_norm = std::max(max_norm, v.norm());
    }
    return max_norm;
  };
  auto com_max_violation = [&]() {
    if (!com_constraint_.has_value() || !com_constraint_->enabled) {
      return 0.0;
    }
    const auto &cfg = *com_constraint_;
    const Eigen::Vector3d com_world = robot_->get_com_position();
    Eigen::MatrixXd A_world = cfg.A;
    Eigen::VectorXd b_world = cfg.b;
    if (cfg.frame_name != "world") {
      const auto frame_pose = robot_->get_frame_pose(cfg.frame_name);
      const Eigen::Matrix3d R = frame_pose.rotation();
      const Eigen::Vector3d t = frame_pose.translation();
      const Eigen::Matrix2d R_xy = R.topLeftCorner<2, 2>();
      A_world = cfg.A * R_xy.transpose();
      b_world = cfg.b;
      for (int i = 0; i < static_cast<int>(b_world.size()); ++i) {
        b_world(i) += A_world.row(i).dot(t.head<2>());
      }
    }
    const Eigen::VectorXd slack = b_world - A_world * com_world.head<2>();
    double violation = 0.0;
    for (int i = 0; i < slack.size(); ++i) {
      violation = std::max(violation, -slack(i));
    }
    return violation;
  };
  auto relative_pose_max_violation = [&]() {
    if (!relative_pose_constraint_.has_value() ||
        !relative_pose_constraint_->enabled) {
      return 0.0;
    }
    const auto &cfg = *relative_pose_constraint_;
    const pinocchio::SE3 T_a = robot_->get_frame_pose(cfg.frame_a);
    const pinocchio::SE3 T_b = robot_->get_frame_pose(cfg.frame_b);
    const pinocchio::SE3 T_rel = compute_relative_frame(T_a, T_b);
    Eigen::VectorXd rel_state = Eigen::VectorXd::Zero(6);
    rel_state.head<3>() = T_rel.translation();
    rel_state.tail<3>() = pinocchio::log3(T_rel.rotation());
    double violation = 0.0;
    for (int i = 0; i < 6; ++i) {
      if (cfg.axis_mask(i) <= 0.5) {
        continue;
      }
      violation = std::max(violation, cfg.lower_bounds(i) - rel_state(i));
      violation = std::max(violation, rel_state(i) - cfg.upper_bounds(i));
    }
    return violation;
  };
  const double fallback_validation_dt =
      (pending_step_validation_dt_.has_value() &&
       std::isfinite(*pending_step_validation_dt_) &&
       *pending_step_validation_dt_ > 0.0)
          ? *pending_step_validation_dt_
          : dt_;
  auto weighted_fallback_candidate_acceptable =
      [&](const Eigen::VectorXd &candidate) {
        if (!candidate.allFinite() || candidate.size() != robot_->nv()) {
          return false;
        }
        if (C.rows() > 0) {
          const Eigen::VectorXd values = C * candidate;
          const double feasibility_tol =
              std::max(1e-8, 10.0 * constraint_tolerance_);
          for (int i = 0; i < values.size(); ++i) {
            if (values(i) < c_lower(i) - feasibility_tol ||
                values(i) > c_upper(i) + feasibility_tol) {
              return false;
            }
          }
        }
        if (use_contact_projection &&
            contact_velocity_norm(candidate) >
                std::max(1e-8, 10.0 * constraint_tolerance_)) {
          return false;
        }

        const bool need_post_step_validation =
            (collision_constraint_.has_value() &&
             collision_constraint_->enabled) ||
            (com_constraint_.has_value() && com_constraint_->enabled) ||
            (relative_pose_constraint_.has_value() &&
             relative_pose_constraint_->enabled);
        if (!need_post_step_validation) {
          return true;
        }

        const double current_com_violation = com_max_violation();
        const double current_rel_violation = relative_pose_max_violation();
        const std::vector<double> current_collision_margins =
            (collision_constraint_.has_value() &&
             collision_constraint_->enabled)
                ? evaluate_post_step_collision_recovery_margins(q_eval)
                : std::vector<double>{};

        const Eigen::VectorXd q_candidate = pinocchio::integrate(
            robot_->model(), q_eval, fallback_validation_dt * candidate);
        robot_->update_kinematics(q_candidate);
        const double candidate_com_violation = com_max_violation();
        const double candidate_rel_violation = relative_pose_max_violation();
        const std::vector<double> candidate_collision_margins =
            (collision_constraint_.has_value() &&
             collision_constraint_->enabled)
                ? evaluate_post_step_collision_recovery_margins(q_candidate)
                : std::vector<double>{};
        robot_->update_kinematics(q_eval);

        constexpr double kViolationImproveTolerance = 1e-9;
        if (candidate_com_violation >
            current_com_violation + kViolationImproveTolerance) {
          return false;
        }
        if (candidate_rel_violation >
            current_rel_violation + kViolationImproveTolerance) {
          return false;
        }
        if (collision_constraint_.has_value() &&
            collision_constraint_->enabled) {
          if (!collision_recovery_margins_acceptable(
                  current_collision_margins, candidate_collision_margins,
                  0.0)) {
            return false;
          }
        }
        return true;
      };
  bool normalized_recovery_accepted = false;
  bool bounded_preference_accepted = false;
  bool normalized_scope = runtime_config_.normalized_pose_recovery_enabled &&
      recovery_has_full_pose && recovery_jacobian.size() > 0 &&
      objective_configs.size() == 2 && objective_configs.front().allow_min_error_fallback;
  bool saw_posture = false;
  for (const auto &task : tasks_) {
    if (!task->isActive() || !normalized_scope) continue;
    if (task->getPriority() == objective_configs[1].priority) {
      saw_posture = true;
      normalized_scope = task->getType() == TaskType::POSTURE;
    } else if (task->getPriority() == objective_configs.front().priority) {
      normalized_scope = task->getType() == TaskType::FRAME_POSE;
    }
  }
  normalized_scope = normalized_scope && saw_posture;
  if (normalized_scope) {
    const bool trace = std::getenv("EMBODIK_NORMALIZED_RECOVERY_TRACE") != nullptr;
    const int nv = robot_->nv();
    const double horizon = fallback_validation_dt;
    const auto [q_min, q_max] = robot_->get_joint_limits();
    const auto limits = robot_->get_velocity_limits();
    Eigen::MatrixXd execution_C(C.rows() + nv, nv);
    execution_C.topRows(C.rows()) = C;
    execution_C.bottomRows(nv).setIdentity();
    Eigen::VectorXd lower(C.rows() + nv), upper(C.rows() + nv);
    lower.head(C.rows()) = c_lower; upper.head(C.rows()) = c_upper;
    lower.tail(nv) = -limits; upper.tail(nv) = limits;
    for (int j = 0; j < nv; ++j) {
      const int qi = velocity_to_config_index[j];
      if (qi < 0) continue;
      lower(C.rows() + j) = std::max(lower(C.rows() + j), (q_min[qi] - q_eval[qi]) / horizon);
      upper(C.rows() + j) = std::min(upper(C.rows() + j), (q_max[qi] - q_eval[qi]) / horizon);
    }
    auto blocks_at = [&](const Eigen::VectorXd &configuration) {
      robot_->update_kinematics(configuration);
      std::vector<double> blocks;
      for (const auto &task : recovery_tasks) {
        task->update(*robot_);
        const auto &error = task->getError();
        if (error.size() != 6) return std::vector<double>{};
        blocks.push_back(error.head<3>().norm());
        blocks.push_back(error.tail<3>().norm());
      }
      return blocks;
    };
    const auto initial_blocks = blocks_at(q_eval);
    const auto &reference = runtime_config_.normalized_pose_recovery_reference;
    bool guide_compatible = reference.size() == nv && reference.allFinite() &&
        (reference.array() >= q_min.array()).all() &&
        (reference.array() <= q_max.array()).all();
    Eigen::VectorXd guide_velocity;
    if (guide_compatible) {
      const auto reference_blocks = blocks_at(reference);
      guide_compatible = reference_blocks.size() == initial_blocks.size();
      for (size_t i = 0; guide_compatible && i < reference_blocks.size(); i += 2)
        guide_compatible = reference_blocks[i] <= runtime_config_.normalized_pose_position_tolerance &&
                           reference_blocks[i + 1] <= runtime_config_.normalized_pose_orientation_tolerance;
      blocks_at(q_eval);
      bool all_targets_satisfied = true;
      for (size_t i = 0; i < initial_blocks.size(); i += 2)
        all_targets_satisfied = all_targets_satisfied &&
            initial_blocks[i] <= runtime_config_.normalized_pose_position_tolerance &&
            initial_blocks[i + 1] <= runtime_config_.normalized_pose_orientation_tolerance;
      guide_compatible = guide_compatible && !all_targets_satisfied;
      if (guide_compatible) {
        guide_velocity = pinocchio::difference(robot_->model(), q_eval, reference);
        double fraction = 1.0;
        for (int j = 0; j < nv; ++j)
          if (std::abs(guide_velocity[j]) > limits[j] && limits[j] > 0.0)
            fraction = std::min(fraction, limits[j] / std::abs(guide_velocity[j]));
        guide_velocity *= fraction;
        // A pose-valid endpoint is not an executable local route. Only let
        // guide-distance consistency veto Cartesian progress when at least
        // one bounded guide step meets the same nonlinear admission contract.
        // Otherwise keep the physical SCALE/recovery paths available.
        bool usable_step = false;
        for (int backoff = 0; backoff < 8 && !usable_step; ++backoff) {
          const Eigen::VectorXd candidate = std::ldexp(1.0, -backoff) * guide_velocity;
          if (detail::ComputeMaxLinearConstraintViolation(
                  execution_C, lower, upper, candidate) > 1e-8) continue;
          const Eigen::VectorXd endpoint = pinocchio::integrate(
              robot_->model(), q_eval, horizon * candidate);
          const auto after = blocks_at(endpoint);
          const bool progress = normalized_pose_progress(initial_blocks, after, runtime_config_);
          blocks_at(q_eval);
          usable_step = progress && weighted_fallback_candidate_acceptable(candidate);
        }
        guide_compatible = usable_step;
        if (trace) std::cerr << "normalized_guide_usable=" << usable_step << "\n";
      }
    }
    Eigen::MatrixXd normalized_J = recovery_jacobian;
    Eigen::VectorXd normalized_goal = recovery_goal;
    for (int row = 0; row < normalized_J.rows(); ++row) {
      const double scale = 1.0 / (row % 6 < 3
          ? runtime_config_.normalized_pose_position_tolerance
          : runtime_config_.normalized_pose_orientation_tolerance);
      normalized_J.row(row) *= scale; normalized_goal[row] *= scale;
    }
    Eigen::VectorXd proposed;
    if (backend_result.solution.size() == static_cast<size_t>(nv)) {
      proposed = Eigen::Map<const Eigen::VectorXd>(backend_result.solution.data(), nv);
      auto expanded_min = q_min, expanded_max = q_max;
      if (elastic_band_config_.enabled && elastic_band_state_.delta.size() == nv)
        expand_scalar_joint_limits_for_velocity_delta(expanded_min, expanded_max,
            elastic_band_state_.delta, velocity_to_config_index, nv);
      clamp_joint_velocity_to_position_limits_for_integration(proposed, q_eval,
          expanded_min, expanded_max, velocity_to_config_index, horizon, nv,
          pending_velocity_lock_indices_, tasks_);
      Eigen::VectorXd endpoint = pinocchio::integrate(robot_->model(), q_eval, horizon * proposed);
      project_scalar_configuration_to_true_joint_limits(endpoint, q_min, q_max,
          velocity_to_config_index, nv);
      proposed = pinocchio::difference(robot_->model(), q_eval, endpoint) / horizon;
    }
    for (int proposal = 0; proposal < 7 && !normalized_recovery_accepted; ++proposal) {
      // A compatible guide selects the joint realization of the maximum
      // physical SCALE ray below; it does not replace the primary task.
      if (proposal == 0 && guide_compatible) continue;
      if (proposal == 1) {
        Eigen::MatrixXd ray_C = Eigen::MatrixXd::Zero(
            execution_C.rows() + recovery_jacobian.rows() + 1, nv + 1);
        ray_C.topLeftCorner(execution_C.rows(), nv) = execution_C;
        ray_C.block(execution_C.rows(), 0, recovery_jacobian.rows(), nv) = recovery_jacobian;
        ray_C.block(execution_C.rows(), nv, recovery_goal.size(), 1) = -recovery_goal;
        ray_C(ray_C.rows() - 1, nv) = 1.0;
        Eigen::VectorXd ray_lower = Eigen::VectorXd::Zero(ray_C.rows());
        Eigen::VectorXd ray_upper = Eigen::VectorXd::Zero(ray_C.rows());
        ray_lower.head(execution_C.rows()) = lower;
        ray_upper.head(execution_C.rows()) = upper;
        ray_upper[ray_C.rows() - 1] = 1.0;
        Eigen::MatrixXd objective = Eigen::MatrixXd::Zero(1, nv + 1);
        objective(0, nv) = 1.0;
        auto fit_config = config; fit_config.epsilon = 1e-8; fit_config.iteration_limit = 128;
        const auto ray = computeConstrainedWeightedVelocitySolutionEigen(
            {Eigen::VectorXd::Ones(1)}, {objective}, ray_C, ray_lower, ray_upper, {}, fit_config);
        if (trace) std::cerr << "normalized_exact_ray status=" << static_cast<int>(ray.status)
            << " scale=" << (ray.solution.size() == static_cast<size_t>(nv + 1) ? ray.solution[nv] : -1.0) << "\n";
        if (ray.status != SolverStatus::kSuccess || ray.solution.size() != static_cast<size_t>(nv + 1) || ray.solution[nv] <= kSoftInfeasibleScaleHoldThreshold) continue;
        proposed = Eigen::Map<const Eigen::VectorXd>(ray.solution.data(), nv);
        if (guide_compatible) {
          Eigen::VectorXd seed = Eigen::Map<const Eigen::VectorXd>(ray.solution.data(), nv + 1);
          ray_lower[ray_C.rows() - 1] = seed[nv];
          ray_upper[ray_C.rows() - 1] = seed[nv];
          Eigen::MatrixXd guide_objective = Eigen::MatrixXd::Zero(nv, nv + 1);
          guide_objective.leftCols(nv).setIdentity();
          const auto guided_ray = computeConstrainedWeightedVelocitySolutionEigen(
              {guide_velocity}, {guide_objective}, ray_C, ray_lower,
              ray_upper, {}, fit_config, {}, &seed);
          if (trace) std::cerr << "normalized_guided_ray status=" << static_cast<int>(guided_ray.status) << "\n";
          if (guided_ray.status == SolverStatus::kSuccess && guided_ray.solution.size() == static_cast<size_t>(nv + 1))
            proposed = Eigen::Map<const Eigen::VectorXd>(guided_ray.solution.data(), nv);
        }
      }
      if (proposal == 2) {
        if (!guide_compatible) continue;
        // One-second feedback time constant; execution is still checked below.
        proposed = guide_velocity;
      }
      if (proposal == 3) {
        // Component bounds imply a Euclidean translation/orientation deviation
        // of at most 5 mm / 10 mrad over this integration horizon.
        Eigen::MatrixXd tube_C = Eigen::MatrixXd::Zero(
            execution_C.rows() + recovery_jacobian.rows() + 1, nv + 1);
        tube_C.topLeftCorner(execution_C.rows(), nv) = execution_C;
        tube_C.block(execution_C.rows(), 0, recovery_jacobian.rows(), nv) = recovery_jacobian;
        tube_C.block(execution_C.rows(), nv, recovery_goal.size(), 1) = -recovery_goal;
        tube_C(tube_C.rows() - 1, nv) = 1.0;
        Eigen::VectorXd tube_lower = Eigen::VectorXd::Zero(tube_C.rows());
        Eigen::VectorXd tube_upper = Eigen::VectorXd::Zero(tube_C.rows());
        tube_lower.head(execution_C.rows()) = lower;
        tube_upper.head(execution_C.rows()) = upper;
        for (int row = 0; row < recovery_jacobian.rows(); ++row) {
          const double budget = (row % 6 < 3 ? 0.005 : 0.01) / (std::sqrt(3.0) * horizon);
          tube_lower[execution_C.rows() + row] = -budget;
          tube_upper[execution_C.rows() + row] = budget;
        }
        tube_upper[tube_C.rows() - 1] = 1.0;
        Eigen::MatrixXd rate_objective = Eigen::MatrixXd::Zero(1, nv + 1);
        rate_objective(0, nv) = 1.0;
        auto fit_config = config; fit_config.epsilon = 1e-8; fit_config.iteration_limit = 128;
        const auto rate = computeConstrainedWeightedVelocitySolutionEigen(
            {Eigen::VectorXd::Ones(1)}, {rate_objective}, tube_C, tube_lower, tube_upper, {}, fit_config);
        if (rate.status != SolverStatus::kSuccess || rate.solution.size() != static_cast<size_t>(nv + 1) ||
            rate.solution[nv] <= kSoftInfeasibleScaleHoldThreshold) continue;
        const double alpha = std::min(1.0, rate.solution[nv]);
        tube_lower[tube_C.rows() - 1] = alpha;
        tube_upper[tube_C.rows() - 1] = alpha;
        Eigen::MatrixXd direction_objective = Eigen::MatrixXd::Zero(normalized_J.rows(), nv + 1);
        direction_objective.leftCols(nv) = normalized_J;
        const Eigen::VectorXd seed = Eigen::Map<const Eigen::VectorXd>(rate.solution.data(), nv + 1);
        const auto direction = computeConstrainedWeightedVelocitySolutionEigen(
            {alpha * normalized_goal}, {direction_objective}, tube_C,
            tube_lower, tube_upper, {}, fit_config, {}, &seed);
        if (trace) std::cerr << "normalized_direction_tube status=" << static_cast<int>(direction.status)
            << " scale=" << alpha << "\n";
        if (direction.status != SolverStatus::kSuccess || direction.solution.size() != static_cast<size_t>(nv + 1)) continue;
        proposed = Eigen::Map<const Eigen::VectorXd>(direction.solution.data(), nv);
      }
      if (proposal == 4) {
        auto fit_config = config;
        fit_config.epsilon = 1e-8;
        fit_config.iteration_limit = 128;
        Eigen::VectorXd seed = proposed;
        const bool feasible_seed = seed.size() == nv && seed.allFinite() &&
            detail::ComputeMaxLinearConstraintViolation(execution_C, lower, upper, seed) <= fit_config.epsilon;
        const auto fit = computeConstrainedWeightedVelocitySolutionEigen(
            {normalized_goal}, {normalized_J}, execution_C, lower, upper, {},
            fit_config, {}, feasible_seed ? &seed : nullptr);
        if (trace) std::cerr << "normalized_fit status=" << static_cast<int>(fit.status) << " message=" << fit.status_message << "\n";
        if (fit.status != SolverStatus::kSuccess || fit.solution.size() != static_cast<size_t>(nv)) continue;
        proposed = Eigen::Map<const Eigen::VectorXd>(fit.solution.data(), nv);
      }
      if (proposal == 5) {
        std::vector<Eigen::RowVectorXd> gradients;
        blocks_at(q_eval);
        for (const auto &task : recovery_tasks) {
          const Eigen::VectorXd error = task->getError();
          if (error.head<3>().norm() <= runtime_config_.normalized_pose_position_tolerance &&
              error.tail<3>().norm() <= runtime_config_.normalized_pose_orientation_tolerance) continue;
          Eigen::VectorXd weighted_error = error;
          weighted_error.head<3>() /= runtime_config_.normalized_pose_position_tolerance;
          weighted_error.tail<3>() /= runtime_config_.normalized_pose_orientation_tolerance;
          Eigen::MatrixXd J = task->getJacobian();
          J.topRows(3) /= runtime_config_.normalized_pose_position_tolerance;
          J.bottomRows(3) /= runtime_config_.normalized_pose_orientation_tolerance;
          gradients.push_back(-2.0 * weighted_error.transpose() * J /
              std::max(1.0, weighted_error.squaredNorm()));
        }
        if (gradients.empty()) continue;
        Eigen::MatrixXd common_C = Eigen::MatrixXd::Zero(execution_C.rows() + gradients.size() + 1, nv + 1);
        common_C.topLeftCorner(execution_C.rows(), nv) = execution_C;
        Eigen::VectorXd common_lower(common_C.rows()), common_upper(common_C.rows());
        common_lower.head(execution_C.rows()) = lower; common_upper.head(execution_C.rows()) = upper;
        for (size_t i = 0; i < gradients.size(); ++i) {
          common_C.row(execution_C.rows() + i).head(nv) = gradients[i];
          common_C(execution_C.rows() + i, nv) = 1.0;
          common_lower[execution_C.rows() + i] = -std::numeric_limits<double>::infinity();
          common_upper[execution_C.rows() + i] = 0.0;
        }
        common_C(common_C.rows() - 1, nv) = 1.0;
        common_lower[common_C.rows() - 1] = 0.0; common_upper[common_C.rows() - 1] = 1.0;
        Eigen::MatrixXd objective = Eigen::MatrixXd::Zero(1, nv + 1); objective(0, nv) = 1.0;
        Eigen::VectorXd goal = Eigen::VectorXd::Ones(1);
        auto fit_config = config; fit_config.epsilon = 1e-8; fit_config.iteration_limit = 128;
        const auto common = computeConstrainedWeightedVelocitySolutionEigen(
            {goal}, {objective}, common_C, common_lower, common_upper, {}, fit_config);
        if (trace) std::cerr << "normalized_common status=" << static_cast<int>(common.status) << " message=" << common.status_message << "\n";
        if (common.status != SolverStatus::kSuccess || common.solution.size() != static_cast<size_t>(nv + 1) || common.solution[nv] <= 1e-8) continue;
        if (trace) std::cerr << "normalized_common rate=" << common.solution[nv] << "\n";
        proposed = Eigen::Map<const Eigen::VectorXd>(common.solution.data(), nv);
      }
      if (proposal == 6) {
        Eigen::JacobiSVD<Eigen::MatrixXd> decomposition(recovery_jacobian, Eigen::ComputeFullV);
        decomposition.setThreshold(1e-10);
        const Eigen::MatrixXd free = decomposition.matrixV().rightCols(nv - decomposition.rank());
        if (free.cols() == 0) continue;
        Eigen::VectorXd seed = Eigen::VectorXd::Zero(free.cols());
        auto fit_config = config; fit_config.epsilon = 1e-8; fit_config.iteration_limit = 128;
        const bool zero_feasible = detail::ComputeMaxLinearConstraintViolation(
            execution_C * free, lower, upper, seed) <= fit_config.epsilon;
        const auto posture = computeConstrainedWeightedVelocitySolutionEigen(
            {goals[1]}, {jacobians[1] * free}, execution_C * free, lower, upper,
            {}, fit_config, {}, zero_feasible ? &seed : nullptr);
        if (trace) std::cerr << "normalized_posture status=" << static_cast<int>(posture.status) << " message=" << posture.status_message << "\n";
        if (posture.status != SolverStatus::kSuccess || posture.solution.size() != static_cast<size_t>(free.cols())) continue;
        proposed = free * Eigen::Map<const Eigen::VectorXd>(posture.solution.data(), free.cols());
        if ((recovery_jacobian * proposed).norm() > 1e-8) continue;
      }
      if (proposed.size() != nv || !proposed.allFinite()) continue;
      const int max_backoffs = proposal == 5 ? 20 : 8;
      for (int backoff = 0; backoff < max_backoffs; ++backoff) {
        const Eigen::VectorXd candidate = std::ldexp(1.0, -backoff) * proposed;
        const double residual = (normalized_J * candidate - normalized_goal).norm();
        const double violation = detail::ComputeMaxLinearConstraintViolation(execution_C, lower, upper, candidate);
        // Common descent optimizes physical pose energy, not the separately
        // capped velocity reference. Admission still requires actual nonlinear
        // descent for each hand and the original safety constraints below.
        const bool linear = proposal == 2 || proposal == 5 || (proposal == 6
            ? residual <= normalized_goal.norm() + 1e-8
            : residual < normalized_goal.norm() - 1e-8);
        if (!linear || violation > 1e-8) {
          if (trace) std::cerr << "normalized_gate proposal=" << proposal << " backoff=" << backoff << " linear=" << linear << " violation=" << violation << " residual=" << residual << " baseline=" << normalized_goal.norm() << "\n";
          continue;
        }
        const Eigen::VectorXd endpoint = pinocchio::integrate(robot_->model(), q_eval, horizon * candidate);
        if (guide_compatible &&
            (reference - endpoint).squaredNorm() >= (reference - q_eval).squaredNorm() - 1e-10) {
          if (trace) std::cerr << "normalized_guide_progress_rejected proposal=" << proposal
              << " backoff=" << backoff << "\n";
          continue;
        }
        const auto after = blocks_at(endpoint);
        const bool progress = normalized_pose_progress(initial_blocks, after, runtime_config_);
        blocks_at(q_eval);
        const bool safety = progress && weighted_fallback_candidate_acceptable(candidate);
        if (trace) {
          std::cerr << "normalized_gate proposal=" << proposal << " backoff=" << backoff << " pose=" << progress << " safety=" << safety << " before=";
          for (double value : initial_blocks) std::cerr << value << ',';
          std::cerr << " after=";
          for (double value : after) std::cerr << value << ',';
          std::cerr << "\n";
        }
        if (!progress || !safety) continue;
        backend_result.solution.assign(candidate.data(), candidate.data() + nv);
        backend_result.status = SolverStatus::kSuccess;
        backend_result.status_message = proposal == 2
            ? "bounded pose-solution guide accepted"
            : proposal == 6
            ? "normalized nullspace posture recovery accepted"
            : "execution-validated normalized pose recovery accepted";
        backend_result.task_scales.assign(goals.size(), -1.0);
        backend_result.task_modes_effective.assign(goals.size(), TaskSolveMode::kMinError);
        backend_result.task_used_fallback.assign(goals.size(), true);
        const Eigen::VectorXd image = recovery_jacobian * candidate;
        const double alpha = recovery_goal.dot(image) / std::max(1e-30, recovery_goal.squaredNorm());
        if (alpha > kSoftInfeasibleScaleHoldThreshold && alpha <= 1.0 + 1e-8 &&
            (image - alpha * recovery_goal).norm() <= 1e-8) {
          backend_result.task_scales.front() = std::min(1.0, alpha);
          backend_result.task_modes_effective.front() = TaskSolveMode::kScale;
          backend_result.task_used_fallback.front() = false;
          backend_result.status_message = "execution-validated physical SCALE ray accepted";
        }
        backend_result.task_errors.resize(goals.size());
        for (size_t i = 0; i < goals.size(); ++i)
          backend_result.task_errors[i] = ((i == 0 ? recovery_jacobian : jacobians[i]) * candidate -
              (i == 0 ? recovery_goal : goals[i])).norm();
        classified_velocity = classify_velocity_outcome(backend_result.status,
            backend_result.status_message, backend_result.task_scales, recovery_goal.norm(),
            backend_result.task_modes_effective, backend_result.task_used_fallback);
        normalized_recovery_accepted = true;
        break;
      }
    }
    blocks_at(q_eval);
  }
  // An expanded position band can select a ray whose joint realization is
  // clipped at the true limits during integration. Prefer an executable SCALE
  // ray before accepting that change in Cartesian direction.
  bool elastic_execution_changes_primary = false;
  if (backend_result.status == SolverStatus::kSuccess &&
      apply_limits && use_position_limits_ &&
      pending_step_validation_dt_.has_value() &&
      !project_elastic_collision_recovery_objective.empty() &&
      project_elastic_collision_recovery_objective.front() &&
      recovery_has_full_pose && recovery_jacobian.size() > 0 &&
      backend_result.solution.size() == static_cast<std::size_t>(robot_->nv())) {
    const auto [q_min, q_max] = robot_->get_joint_limits();
    bool valid_seed = true;
    for (int vi = 0; vi < robot_->nv(); ++vi) {
      const int qi = velocity_to_config_index[vi];
      if (qi >= 0 && qi < q_eval.size() &&
          (q_eval[qi] < q_min[qi] || q_eval[qi] > q_max[qi])) {
        valid_seed = false;
        break;
      }
    }
    const Eigen::VectorXd nominal = Eigen::Map<const Eigen::VectorXd>(
        backend_result.solution.data(), robot_->nv());
    if (valid_seed && nominal.allFinite()) {
      auto expanded_min = q_min, expanded_max = q_max;
      if (elastic_band_config_.enabled)
        expand_scalar_joint_limits_for_velocity_delta(
            expanded_min, expanded_max, elastic_band_state_.delta,
            velocity_to_config_index, robot_->nv());
      Eigen::VectorXd executable = nominal;
      clamp_joint_velocity_to_position_limits_for_integration(
          executable, q_eval, expanded_min, expanded_max,
          velocity_to_config_index, fallback_validation_dt, robot_->nv(),
          pending_velocity_lock_indices_, tasks_);
      Eigen::VectorXd endpoint = pinocchio::integrate(
          robot_->model(), q_eval, fallback_validation_dt * executable);
      project_scalar_configuration_to_true_joint_limits(
          endpoint, q_min, q_max, velocity_to_config_index, robot_->nv());
      executable = pinocchio::difference(robot_->model(), q_eval, endpoint) /
                   fallback_validation_dt;
      elastic_execution_changes_primary =
          (recovery_jacobian * (executable - nominal)).norm() > 1e-8;
    }
  }
  const bool recovery_model_changed =
      recovery_has_full_pose && recovery_jacobian.size() > 0 && !jacobians.empty() &&
      (recovery_jacobian.rows() != jacobians.front().rows() ||
       (recovery_jacobian - jacobians.front()).norm() > 1e-10);
  const bool legacy_recovery_requested =
      position_step_physical_recovery_active_ ||
      position_step_continuity_state_ != PositionStepContinuityState::kMonitoring;
  const bool recovery_requested = legacy_recovery_requested || recovery_model_changed ||
                                  elastic_execution_changes_primary;
  if (!normalized_recovery_accepted && recovery_requested && recovery_goal.size() > 0 && recovery_goal.norm() > 1e-8 &&
      recovery_jacobian.cols() == robot_->nv()) {
    const int nv = robot_->nv();
    const double horizon = fallback_validation_dt;
    const Eigen::VectorXd limits = robot_->get_velocity_limits();
    const auto [q_min, q_max] = robot_->get_joint_limits();
    Eigen::MatrixXd recovery_C(C.rows() + nv, nv);
    recovery_C.topRows(C.rows()) = C;
    recovery_C.bottomRows(nv).setIdentity();
    Eigen::VectorXd lower(C.rows() + nv), upper(C.rows() + nv);
    lower.head(C.rows()) = c_lower;
    upper.head(C.rows()) = c_upper;
    Eigen::JacobiSVD<Eigen::MatrixXd> decomposition(recovery_jacobian,
                                                  Eigen::ComputeFullV);
    decomposition.setThreshold(1e-10);
    const Eigen::MatrixXd free =
        decomposition.matrixV().rightCols(nv - decomposition.rank());
    const Eigen::VectorXd metric = limits.cwiseMax(1e-6).cwiseInverse();
    auto pose_blocks = [&]() {
      std::vector<double> blocks;
      for (const auto &task : recovery_tasks) {
        task->update(*robot_);
        const auto &error = task->getError();
        if (error.size() >= 6) {
          blocks.push_back(error.head<3>().norm());
          blocks.push_back(error.segment<3>(3).norm());
        } else {
          blocks.push_back(error.norm());
        }
      }
      return blocks;
    };
    const auto initial_blocks = pose_blocks();
    if (legacy_recovery_requested && position_step_physical_recovery_anchor_blocks_.size() !=
        initial_blocks.size()) {
      position_step_physical_recovery_anchor_blocks_ = initial_blocks;
    }
    const auto recovery_anchor_blocks = legacy_recovery_requested
        ? position_step_physical_recovery_anchor_blocks_ : initial_blocks;
    auto task_merits = [&]() {
      std::vector<double> merits;
      for (const auto &task : recovery_tasks) {
        merits.push_back(task->getError().norm());
      }
      return merits;
    };
    const auto initial_merits = task_merits();
    double incumbent_reduction = 0.0;
    if (backend_result.solution.size() == static_cast<std::size_t>(nv)) {
      Eigen::VectorXd incumbent = Eigen::Map<const Eigen::VectorXd>(
          backend_result.solution.data(), nv);
      if (use_position_limits_) {
        clamp_joint_velocity_to_position_limits_for_integration(
            incumbent, q_eval, q_min, q_max, velocity_to_config_index, horizon,
            nv, pending_velocity_lock_indices_, tasks_);
      }
      const Eigen::VectorXd endpoint = pinocchio::integrate(
          robot_->model(), q_eval, horizon * incumbent);
      bool inside_limits = endpoint.allFinite();
      for (int qi = 0; inside_limits && qi < endpoint.size(); ++qi) {
        inside_limits = endpoint[qi] >= q_min[qi] && endpoint[qi] <= q_max[qi];
      }
      if (inside_limits && weighted_fallback_candidate_acceptable(incumbent)) {
        robot_->update_kinematics(endpoint);
        const auto incumbent_blocks = pose_blocks();
        for (std::size_t index = 0; index < initial_blocks.size(); ++index) {
          incumbent_reduction += initial_blocks[index] - incumbent_blocks[index];
        }
        robot_->update_kinematics(q_eval);
        pose_blocks();
      }
    }
    double failed_fraction = 1.0;
    double accepted_fraction = 0.0;
    double full_preference_scale = 0.0;
    bool preference_backoff_enabled = false;
    int refinements = 0;
    // Once a safe radius is found, refine its bracket without accepting an
    // unverified endpoint. Limit the extra solves to three.
    for (int trial = elastic_execution_changes_primary ? -4 : 0; trial < 15; ++trial) {
      const bool bounded_ray_preference = trial < 0;
      const int preference_retry = bounded_ray_preference ? trial + 4 : 0;
      if (bounded_ray_preference && preference_retry > 0 &&
          !preference_backoff_enabled) continue;
      if (!legacy_recovery_requested && trial > 0) break;
      if (accepted_fraction <= 0.0 && trial >= 12) {
        break;
      }
      if (accepted_fraction > 0.0 && refinements++ >= 3) {
        break;
      }
      const double fraction =
          bounded_ray_preference ? 1.0 : accepted_fraction > 0.0
              ? 0.5 * (accepted_fraction + failed_fraction)
              : std::ldexp(1.0, -trial);
      const double previous_failed_fraction = failed_fraction;
      failed_fraction = fraction;
      lower.tail(nv) = -fraction * limits;
      upper.tail(nv) = fraction * limits;
      if (use_position_limits_) {
        for (int index = 0; index < nv; ++index) {
          const int qi = velocity_to_config_index[index];
          if (qi < 0 || qi >= q_eval.size()) {
            continue;
          }
          if (excluded_union.count(index) > 0 ||
              std::find(pending_velocity_lock_indices_.begin(),
                        pending_velocity_lock_indices_.end(), index) !=
                  pending_velocity_lock_indices_.end()) {
            continue;
          }
          const double execution_lower =
              (q_min(qi) + 1e-4 - q_eval(qi)) / horizon;
          const double execution_upper =
              (q_max(qi) - 1e-4 - q_eval(qi)) / horizon;
          lower(C.rows() + index) =
              std::max(lower(C.rows() + index),
                       bounded_ray_preference ? std::min(0.0, execution_lower)
                                              : execution_lower);
          upper(C.rows() + index) =
              std::min(upper(C.rows() + index),
                       bounded_ray_preference ? std::max(0.0, execution_upper)
                                              : execution_upper);
        }
      }
      Eigen::VectorXd candidate;
      double scale = 0.0;
      SolverResult generalized_result;
      bool generalized_candidate = false;
      bool generalized_min_error_candidate = false;
      if (bounded_ray_preference || (trial == 0 && recovery_model_changed)) {
        auto alternative_jacobians = jacobians;
        auto alternative_goals = goals;
        alternative_jacobians.front() = recovery_jacobian;
        alternative_goals.front() = recovery_goal;
        // Re-solve the hierarchy with a smaller primary request after a valid
        // full ray fails its nonlinear endpoint checks. Keep all hard rows and
        // lower objectives intact; scaling a velocity would change affine
        // task equations and need not preserve an origin-excluding hard row.
        const double primary_request_scale =
            bounded_ray_preference && preference_retry > 0
                ? std::ldexp(full_preference_scale, -preference_retry) : 1.0;
        alternative_goals.front() *= primary_request_scale;
        std::vector<Eigen::VectorXd> biases;
        for (const auto &goal : alternative_goals) {
          biases.push_back(Eigen::VectorXd::Zero(goal.size()));
        }
        Eigen::MatrixXd alternative_C = recovery_C;
        if (metric_active) {
          const auto D = joint_metric_col_scale_.asDiagonal();
          for (auto &J : alternative_jacobians) J = J * D;
          alternative_C = alternative_C * D;
        }
        auto alternative_config = config;
        // This candidate must realize the original task ray. Damping the
        // supported directions of a redundant row set can make a feasible
        // equality fail the recovery check. Bounds and endpoint checks provide
        // the trust region for this exact-inverse trial.
        alternative_config.epsilon = 1e-8;
        alternative_config.regularization_config.epsilon = 1e-8;
        alternative_config.regularization_config.regularization_factor = 0.0;
        auto alternative_objectives = objective_configs;
        if (bounded_ray_preference) {
          alternative_objectives.front().solve_mode = TaskSolveMode::kScale;
          alternative_objectives.front().allow_min_error_fallback = false;
        }
        generalized_result = solveHierarchicalLinearSystemEigen(
            alternative_goals, biases, alternative_jacobians, alternative_C,
            lower, upper, alternative_config, alternative_objectives, nullptr,
            detail::HierarchicalLinearSolverExecutionOptions{});
        const bool authorized_min_error =
            alternative_objectives.front().allow_min_error_fallback &&
            !generalized_result.task_scales.empty() &&
            generalized_result.task_scales.front() == -1.0 &&
            !generalized_result.task_modes_effective.empty() &&
            generalized_result.task_modes_effective.front() == TaskSolveMode::kMinError &&
            !generalized_result.task_used_fallback.empty() &&
            generalized_result.task_used_fallback.front();
        // A lower SCALE task can have no feasible equality while the primary
        // ray remains valid. The preference candidate still has to pass the
        // physical equality and every executable hard row below.
        const bool usable_status =
            generalized_result.status == SolverStatus::kSuccess ||
            (bounded_ray_preference &&
             generalized_result.status == SolverStatus::kNoProgress);
        if (usable_status &&
            generalized_result.solution.size() == static_cast<std::size_t>(nv) &&
            !generalized_result.task_scales.empty() &&
            ((generalized_result.task_scales.front() > 0.0 &&
              generalized_result.task_scales.front() <= 1.0) || authorized_min_error)) {
          candidate = Eigen::Map<const Eigen::VectorXd>(
              generalized_result.solution.data(), nv);
          if (metric_active) candidate.array() *= joint_metric_col_scale_.array();
          scale = generalized_result.task_scales.front();
          if (bounded_ray_preference) {
            scale *= primary_request_scale;
            generalized_result.task_scales.front() = scale;
          }
          generalized_candidate = true;
          generalized_min_error_candidate = authorized_min_error;
        }
      }

      if (!generalized_candidate &&
          (bounded_ray_preference || !legacy_recovery_requested)) continue;
      if (!generalized_candidate && !detail::RecoverZeroScaleTask(
              recovery_jacobian, recovery_goal,
              Eigen::VectorXd::Zero(recovery_goal.size()), Eigen::MatrixXd(0, nv),
              recovery_C, lower, upper, Eigen::VectorXd::Zero(nv), &candidate,
              &scale)) {
        continue;
      }

      if (!generalized_candidate && free.cols() > 0) {
        VelocitySolverConfig polish_config = config;
        polish_config.epsilon = 1e-8;
        polish_config.iteration_limit = 128;
        const Eigen::VectorXd value = recovery_C * candidate;
        const auto polished = computeConstrainedWeightedVelocitySolutionEigen(
            {metric.asDiagonal() * (-candidate)}, {metric.asDiagonal() * free},
            recovery_C * free, lower - value, upper - value, {}, polish_config);
        if (polished.status != SolverStatus::kSuccess) {
          continue;
        }
        candidate += free * Eigen::Map<const Eigen::VectorXd>(
            polished.solution.data(), polished.solution.size());
      }
      const double physical_residual =
          (recovery_jacobian * candidate - recovery_goal).norm();
      if (!candidate.allFinite() ||
          (bounded_ray_preference &&
           detail::ComputeMaxLinearConstraintViolation(
               recovery_C, lower, upper, candidate) > 1e-8) ||
          (generalized_min_error_candidate
               ? physical_residual >= recovery_goal.norm() - 1e-8
               : (recovery_jacobian * candidate - scale * recovery_goal).norm() > 1e-8)) {
        continue;
      }
      robot_->update_kinematics(pinocchio::integrate(robot_->model(), q_eval,
                                                  horizon * candidate));
      const auto proposed_blocks = pose_blocks();
      const auto proposed_merits = task_merits();
      robot_->update_kinematics(q_eval);
      pose_blocks();
      bool improving = proposed_blocks.size() == initial_blocks.size();
      double reduction = 0.0;
      for (std::size_t index = 0; index < initial_blocks.size() && improving;
           ++index) {
        const double tolerance = 64.0 * std::numeric_limits<double>::epsilon() *
            std::max({1.0, initial_blocks[index], proposed_blocks[index]});
        improving = proposed_blocks[index] <=
                    recovery_anchor_blocks[index] +
                        tolerance;
        reduction += initial_blocks[index] - proposed_blocks[index];
      }
      for (std::size_t index = 0; index < initial_merits.size() && improving;
           ++index) {
        const double tolerance = 64.0 * std::numeric_limits<double>::epsilon() *
            std::max({1.0, initial_merits[index], proposed_merits[index]});
        improving = proposed_merits[index] <= initial_merits[index] + tolerance;
      }
      const double required_reduction = generalized_candidate && !bounded_ray_preference
          ? std::max(1e-8, incumbent_reduction + 1e-8) : 1e-8;
      if (!improving || reduction <= required_reduction) {
        if (bounded_ray_preference && preference_retry == 0) {
          full_preference_scale = scale;
          preference_backoff_enabled = true;
        }
        continue;
      }
      if (!weighted_fallback_candidate_acceptable(candidate)) {
        if (bounded_ray_preference && preference_retry == 0) {
          full_preference_scale = scale;
          preference_backoff_enabled = true;
        }
        continue;
      }

      failed_fraction = previous_failed_fraction;
      accepted_fraction = fraction;
      if (!bounded_ray_preference) {
        position_step_physical_recovery_active_ = true;
        position_step_physical_recovery_anchor_blocks_ = recovery_anchor_blocks;
      }
      if (generalized_candidate) {
        backend_result = std::move(generalized_result);
      }
      bounded_preference_accepted = bounded_ray_preference;
      backend_result.solution.assign(candidate.data(), candidate.data() + nv);
      backend_result.status = SolverStatus::kSuccess;
      backend_result.status_message =
          bounded_ray_preference
              ? (preference_retry > 0
                     ? "bounded physical SCALE ray accepted after primary goal backoff"
                     : "bounded physical SCALE ray preferred to elastic clipping")
              : generalized_candidate
              ? "guarded generalized physical-task position recovery accepted"
              : "bounded physical-task position recovery accepted";
      if (!generalized_candidate) {
        backend_result.task_scales.assign(goals.size(), 0.0);
        backend_result.task_scales.front() = scale;
        backend_result.task_modes_effective.clear();
        for (const auto &objective : objective_configs) {
          backend_result.task_modes_effective.push_back(objective.solve_mode);
        }
        backend_result.task_used_fallback.assign(goals.size(), false);
      }
      backend_result.task_errors.resize(goals.size());
      for (std::size_t index = 0; index < goals.size(); ++index) {
        backend_result.task_errors[index] =
            ((generalized_candidate && index == 0 ? recovery_jacobian : jacobians[index]) *
                 candidate - (generalized_candidate && index == 0 ? recovery_goal : goals[index])).norm();
      }
      classified_velocity = classify_velocity_outcome(
          backend_result.status, backend_result.status_message,
          backend_result.task_scales, recovery_goal.norm(),
          backend_result.task_modes_effective, backend_result.task_used_fallback);
      if (fraction == 1.0) {
        break;
      }
    }
  }
  // An invalid realization need not discard its achieved primary motion.
  // Repair native bounds, or elastic position integration, without changing
  // the hierarchy's initial problem or its already achieved primary image.
  const bool elastic_position_realization = !bounded_preference_accepted &&
      backend_result.status == SolverStatus::kSuccess && apply_limits &&
      use_position_limits_ && pending_step_validation_dt_.has_value() &&
      !use_contact_projection && !project_elastic_collision_recovery_objective.empty() &&
      project_elastic_collision_recovery_objective.front();
  // Normalized recovery already validates the original hard rows and true
  // executable bounds. A later nullspace repair can invalidate its nonlinear
  // pose descent even when it preserves the linear primary image.
  if (!normalized_recovery_accepted &&
      (backend_result.status == SolverStatus::kNumericalError ||
       elastic_position_realization) &&
      !goals.empty() && physical_primary_jacobian.rows() > 0 &&
      static_cast<int>(backend_result.solution.size()) == robot_->nv()) {
    const Eigen::VectorXd nominal = Eigen::Map<const Eigen::VectorXd>(
        backend_result.solution.data(), backend_result.solution.size());
    Eigen::MatrixXd repair_C = C;
    Eigen::VectorXd repair_lower = c_lower;
    Eigen::VectorXd repair_upper = c_upper;
    if (elastic_position_realization && nominal.allFinite()) {
      const int nv = robot_->nv();
      const auto [q_min, q_max] = robot_->get_joint_limits();
      Eigen::VectorXd lower = Eigen::VectorXd::Constant(nv, -kUnboundedConstraintLimit);
      Eigen::VectorXd upper = Eigen::VectorXd::Constant(nv, kUnboundedConstraintLimit);
      bool valid_seed = true;
      for (int vi = 0; vi < nv; ++vi) {
        const int qi = velocity_to_config_index[vi];
        if (qi < 0 || qi >= q_eval.size() || !std::isfinite(q_min[qi]) ||
            !std::isfinite(q_max[qi])) continue;
        if (q_eval[qi] < q_min[qi] || q_eval[qi] > q_max[qi]) {
          valid_seed = false;
          break;
        }
        if (excluded_union.count(vi) > 0 ||
            std::find(pending_velocity_lock_indices_.begin(),
                      pending_velocity_lock_indices_.end(), vi) !=
                pending_velocity_lock_indices_.end()) {
          lower[vi] = upper[vi] = 0.0;
        } else {
          lower[vi] = (q_min[qi] + 1e-4 - q_eval[qi]) / fallback_validation_dt;
          upper[vi] = (q_max[qi] - 1e-4 - q_eval[qi]) / fallback_validation_dt;
        }
      }
      Eigen::VectorXd clipped = nominal;
      clamp_joint_velocity_solution_in_place(clipped, lower, upper, false, nv);
      if (valid_seed && (lower.array() <= upper.array()).all() &&
          (physical_primary_jacobian * nominal).norm() > 1e-8 &&
          (physical_primary_jacobian * (clipped - nominal)).norm() > 1e-8) {
        repair_C.conservativeResize(C.rows() + nv, nv);
        repair_C.bottomRows(nv).setIdentity();
        repair_lower.conservativeResize(C.rows() + nv);
        repair_upper.conservativeResize(C.rows() + nv);
        repair_lower.tail(nv) = lower;
        repair_upper.tail(nv) = upper;
      }
    }
    if (nominal.allFinite() &&
        detail::ComputeMaxLinearConstraintViolation(repair_C, repair_lower, repair_upper,
                                                    nominal) > config.epsilon) {
      Eigen::MatrixXd protected_primary(
          physical_primary_jacobian.rows() + jacobians.front().rows(),
          robot_->nv());
      protected_primary.topRows(physical_primary_jacobian.rows()) =
          physical_primary_jacobian;
      protected_primary.bottomRows(jacobians.front().rows()) = jacobians.front();
      Eigen::JacobiSVD<Eigen::MatrixXd> decomposition(protected_primary,
                                                     Eigen::ComputeFullV);
      decomposition.setThreshold(1e-10);
      const Eigen::MatrixXd free = decomposition.matrixV().rightCols(
          robot_->nv() - decomposition.rank());
      if (free.cols() > 0) {
        Eigen::VectorXd clipped = nominal;
        clamp_final_velocity_candidate(clipped);
        if (repair_C.rows() > C.rows()) {
          clamp_joint_velocity_solution_in_place(clipped, repair_lower.tail(robot_->nv()),
                                                repair_upper.tail(robot_->nv()), false, robot_->nv());
        }
        const Eigen::VectorXd values = repair_C * nominal;
        VelocitySolverConfig repair_config = config;
        repair_config.epsilon = 1e-10;
        repair_config.iteration_limit = 128;
        repair_config.regularization_config.regularization_factor = 0.0;
        const auto repaired = computeConstrainedWeightedVelocitySolutionEigen(
            {clipped - nominal}, {free}, repair_C * free, repair_lower - values,
            repair_upper - values, {}, repair_config);
        if (repaired.status == SolverStatus::kSuccess &&
            static_cast<int>(repaired.solution.size()) == free.cols()) {
          const Eigen::VectorXd candidate = nominal + free *
              Eigen::Map<const Eigen::VectorXd>(repaired.solution.data(),
                                               repaired.solution.size());
          if ((protected_primary * (candidate - nominal)).norm() <= 1e-8 &&
              detail::ComputeMaxLinearConstraintViolation(
                  repair_C, repair_lower, repair_upper, candidate) <= 1e-8 &&
              weighted_fallback_candidate_acceptable(candidate)) {
            backend_result.solution.assign(candidate.data(),
                                           candidate.data() + candidate.size());
            backend_result.status = SolverStatus::kSuccess;
            backend_result.status_message =
                repair_C.rows() > C.rows()
                    ? "elastic integration realization repaired with primary motion preserved"
                    : "native hard bounds repaired with primary motion preserved";
            for (std::size_t index = 0; index < goals.size(); ++index) {
              if (index < backend_result.task_errors.size())
                backend_result.task_errors[index] =
                    (jacobians[index] * candidate - goals[index]).norm();
            }
            classified_velocity = classify_velocity_outcome(
                backend_result.status, backend_result.status_message,
                backend_result.task_scales, primary_goal_norm,
                backend_result.task_modes_effective,
                backend_result.task_used_fallback);
          }
        }
      }
    }
  }
  const bool can_try_weighted_fallback =
      runtime_config_.weighted_fallback_enabled &&
      classified_velocity.status != SolverStatus::kSuccess &&
      classified_velocity.status != SolverStatus::kNonFiniteInput &&
      !goals.empty();
  if (can_try_weighted_fallback && !advisory.available) {
    advisory = compute_weighted_advisory();
  }
  if (can_try_weighted_fallback && advisory.available) {
    Eigen::VectorXd accepted_candidate = advisory.v;
    if (use_contact_projection) {
      accepted_candidate = contact_P_c * accepted_candidate;
    }
    clamp_final_velocity_candidate(accepted_candidate);
    const bool accept_weighted_fallback =
        weighted_fallback_candidate_acceptable(accepted_candidate);
    if (accept_weighted_fallback) {
      backend_result.solution.assign(
          accepted_candidate.data(),
          accepted_candidate.data() + accepted_candidate.size());
      backend_result.status = SolverStatus::kSuccess;
      backend_result.status_message =
          "constrained weighted fallback accepted after prioritized solver: " +
          classified_velocity.status_message;
      backend_result.final_error = advisory.per_objective_error.empty()
                                       ? 0.0
                                       : advisory.per_objective_error.front();
      backend_result.task_scales = {-1.0};
      backend_result.task_errors = advisory.per_objective_error;
      backend_result.task_modes_effective = {TaskSolveMode::kMinError};
      backend_result.task_used_fallback = {false};
      backend_result.condition_number = advisory.condition_number;
      result.weighted_fallback_used = true;
      result.recovery_stage = SolverRecoveryStage::kWeightedFallback;
      classified_velocity = classify_velocity_outcome(
          backend_result.status, backend_result.status_message,
          backend_result.task_scales, primary_goal_norm,
          backend_result.task_modes_effective,
          backend_result.task_used_fallback);
    }
  }

  auto infer_active_task_layout = [&]() {
    bool saw_pose = false;
    bool saw_split = false;
    // Grouped objectives have no single task pointer. Inspect the active
    // source tasks rather than the policy state, which cannot split raw tasks.
    for (const auto &task : tasks_) {
      if (!task || !task->isActive()) {
        continue;
      }
      const TaskType type = task->getType();
      saw_pose = saw_pose || type == TaskType::FRAME_POSE;
      saw_split = saw_split || type == TaskType::FRAME_POSITION ||
                              type == TaskType::FRAME_ORIENTATION;
    }
    if (saw_pose && !saw_split) {
      return TaskLayout::kMerged;
    }
    return TaskLayout::kSplit;
  };
  result.active_task_layout = infer_active_task_layout();
  result.binding_score = auto_layout_binding_score_;

  // Create velocity-specific result
  result.status = classified_velocity.status;
  result.solution = backend_result.solution;
  result.computation_time_ms = backend_result.computation_time_ms;
  result.solver_computation_time_ms = backend_result.computation_time_ms;
  result.iterations = backend_result.iterations;
  result.final_error = backend_result.final_error;
  result.task_scales = backend_result.task_scales;
  result.task_errors = backend_result.task_errors;
  result.task_modes_effective = backend_result.task_modes_effective;
  result.task_used_fallback = backend_result.task_used_fallback;
  result.status_message = classified_velocity.status_message;
  result.limits_applied = apply_limits;
  result.condition_number = backend_result.condition_number;

  for (size_t i = 0; i < objective_tasks.size() &&
                     i < result.task_modes_effective.size() &&
                     i < result.task_used_fallback.size();
       ++i) {
    if (objective_tasks[i]) {
      objective_tasks[i]->setLastEffectiveMode(result.task_modes_effective[i]);
      objective_tasks[i]->setUsedMinErrorFallback(result.task_used_fallback[i]);
    }
  }

  // Convert solution to Eigen vector
  if (!result.solution.empty()) {
    // Clamp velocity solution to constraint bounds to avoid post-integration
    // limit violations from QP tolerance.
    if (apply_limits && c_lower.size() >= robot_->nv() && !use_contact_projection) {
      Eigen::Map<Eigen::VectorXd> dq(result.solution.data(),
                                     result.solution.size());
      // Velocity-box clamping (always applies to first nv rows).
      clamp_joint_velocity_solution_in_place(
          dq, c_lower, c_upper, false, robot_->nv());
      // Sparse position-limit clamping using dense per-joint bounds array.
      if (use_position_limits_) {
        const int n_dq = static_cast<int>(dq.size());
        const int n_dens = static_cast<int>(dense_pos_lower_sp.size());
        const int n_clamp = std::min(n_dq, n_dens);
        for (int k = 0; k < n_clamp; ++k) {
          if (dense_pos_lower_sp[k] < kNoPosBound)
            dq[k] = std::max(dq[k], dense_pos_lower_sp[k]);
          if (dense_pos_upper_sp[k] > -kNoPosBound)
            dq[k] = std::min(dq[k], dense_pos_upper_sp[k]);
        }
      }
    }

    Eigen::Map<Eigen::VectorXd> dq_final(result.solution.data(),
                                         result.solution.size());
    if (has_active_velocity_centroidal_hard_constraints()) {
      std::string validation_message;
      if (!validate_centroidal_velocity_candidate(
              q_eval, explicit_current_dq, dq_final, &validation_message)) {
        result.status = SolverStatus::kInfeasible;
        result.status_message = validation_message;
        dq_final.setZero();
        std::fill(result.solution.begin(), result.solution.end(), 0.0);
      }
    }

    result.joint_velocities = Eigen::Map<const Eigen::VectorXd>(
        result.solution.data(), result.solution.size());
    last_solution_dq_norm_ = result.joint_velocities.norm();

    // A position step may invoke several speculative velocity solves before it
    // accepts one outer-tick command.  Only the accepted position result may
    // advance acceleration history; direct velocity solves still commit here.
    if (acceleration_limits_enabled_ && position_step_call_depth_ == 0) {
      previous_dq_ = result.joint_velocities;
    }

    // Identify saturated joints based on constraint bounds
    if (apply_limits && c_lower.size() >= robot_->nv()) {
      double tolerance = 0.01; // 1% tolerance

      // Use the same combined bounds as post-solve clamping so saturation
      // reporting reflects either velocity-row or position-row activation.
      for (int i = 0; i < robot_->nv(); ++i) {
        double joint_vel = result.joint_velocities[i];
        double lower = c_lower[i];
        double upper = c_upper[i];
        if (use_position_limits_ && i < static_cast<int>(dense_pos_lower_sp.size())) {
          // Use dense per-joint position bounds (O(1) lookup, no hash overhead).
          if (dense_pos_lower_sp[i] < kNoPosBound)
            lower = std::max(lower, dense_pos_lower_sp[i]);
          if (dense_pos_upper_sp[i] > -kNoPosBound)
            upper = std::min(upper, dense_pos_upper_sp[i]);
        }

        // Check if joint velocity is near its constraint bounds
        if (joint_vel <= lower + tolerance || joint_vel >= upper - tolerance) {
          result.saturated_joints.push_back(i);
        }
      }
    }
  } else {
    last_solution_dq_norm_ = 0.0;
  }

  // C is still physical here: the metric solve uses a separate C*D copy.
  // Capture only for a stalled/failing position step. Direct escape does not
  // certify softened rows, contact projection, or dynamic braking/history.
  if (position_step_hard_constraints_ != nullptr && !has_soft_rows &&
      contact_frames_.empty() && !acceleration_limits_enabled_ &&
      !has_active_velocity_centroidal_hard_constraints() &&
      pending_step_validation_dt_.has_value() &&
      std::isfinite(*pending_step_validation_dt_) &&
      *pending_step_validation_dt_ > 0.0 &&
      (result.status != SolverStatus::kSuccess ||
       result.joint_velocities.norm() < stall_config_.dq_stall_eps)) {
    auto &hard = *position_step_hard_constraints_;
    hard.q = current_q;
    hard.dt = *pending_step_validation_dt_;
    hard.matrix = C;
    hard.lower = c_lower;
    hard.upper = c_upper;
    hard.valid = C.allFinite() && c_lower.allFinite() && c_upper.allFinite();
  }
  stall_handler_update(result);
  elastic_band_update(result);
  update_auto_task_layout_feedback(result);

  return result;
}


} // namespace embodik
