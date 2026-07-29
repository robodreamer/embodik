/**
 * @file kinematics_solver_velocity.cpp
 * @brief Velocity-level solve path for KinematicsSolver
 */

#include "kinematics_solver_internal.hpp"

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
  if (velocity_zmp_constraint_.has_value() &&
      velocity_zmp_constraint_->enabled &&
      !pending_explicit_current_dq_.has_value()) {
    result.status = SolverStatus::kInvalidInput;
    result.status_message =
        "velocity-ZMP constraint requires explicit current_dq; use "
        "solve_velocity_with_state";
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

  int current_priority = std::numeric_limits<int>::min();
  auto &group_tasks = scratch_group_tasks_;
  group_tasks.clear();
  group_tasks.reserve(tasks_.size());

  auto flush_group = [&]() {
    if (group_tasks.empty()) {
      return;
    }

    bool all_scale_family = true;
    bool all_strict_scale = true;
    bool all_min_error = true;
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
        Eigen::MatrixXd J = task->getJacobian();
        if (g.rows() > 0) {
          combined_goal.segment(offset, g.rows()) = g;
          combined_jac.block(offset, 0, J.rows(), robot_->nv()) = J;
          offset += static_cast<int>(g.rows());
        }
      }
      goals.push_back(std::move(combined_goal));
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
        jacobians.push_back(task->getJacobian());
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

  // Near a joint position limit, zero Jacobian entries that would command
  // motion further into that limit. This avoids whole-task SNS scale collapse
  // and preserves partial solutions through remaining DOFs.
  if (apply_limits && use_position_limits_) {
    clamp_jacobians_near_joint_limits(jacobians, goals, objective_configs,
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
  // that already satisfies the separating lower bound is left unchanged.
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
  if (pending_explicit_current_dq_.has_value()) {
    velocity_zmp_constraint_result =
        compute_velocity_zmp_constraint(*pending_explicit_current_dq_);
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
  // A deadband of 1% of the acceleration step prevents oscillation when
  // the velocity is small and the acceleration limit is tight.
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
      const double accel_step = acceleration_limits_[i] * dt_;
      const double deadband = accel_step * 0.01;
      double a_lb = previous_dq_[i] - accel_step - deadband;
      double a_ub = previous_dq_[i] + accel_step + deadband;
      c_lower[i] = std::max(c_lower[i], a_lb);
      c_upper[i] = std::min(c_upper[i], a_ub);
    }
  }

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
  auto backend_result = computeMultiObjectiveVelocitySolutionEigen(
      goals, *solve_jacobians, *solve_C, c_lower, c_upper, config,
      objective_configs, has_soft_rows ? &max_softening_factors : nullptr);
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
          candidate, c_lower, c_upper, use_position_limits_, robot_->nv());
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
    for (const auto &task : objective_tasks) {
      if (!task) {
        continue;
      }
      const TaskType type = task->getType();
      saw_pose = saw_pose || type == TaskType::FRAME_POSE;
      saw_split = saw_split || type == TaskType::FRAME_POSITION ||
                              type == TaskType::FRAME_ORIENTATION;
    }
    if (runtime_config_.enable_auto_task_layout) {
      return current_auto_task_layout_;
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
          dq, c_lower, c_upper, use_position_limits_, robot_->nv());
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
    auto final_centroidal_constraints_acceptable =
        [&](const Eigen::VectorXd &candidate, std::string *message) {
          if (candidate.size() != robot_->nv() || !candidate.allFinite()) {
            if (message != nullptr) {
              *message = "final centroidal validation found non-finite velocity";
            }
            return false;
          }
          const double tol = std::max(1e-8, 10.0 * constraint_tolerance_);
          if (centroidal_momentum_bounds_.has_value() &&
              centroidal_momentum_bounds_->enabled) {
            const auto &cfg = *centroidal_momentum_bounds_;
            const Eigen::VectorXd h =
                robot_->get_centroidal_momentum_matrix() * candidate;
            int selected = 0;
            for (int row = 0; row < 6; ++row) {
              if (cfg.axis_mask(row) == 0.0) {
                continue;
              }
              if (h(row) < cfg.lower_h(selected) - tol ||
                  h(row) > cfg.upper_h(selected) + tol) {
                if (message != nullptr) {
                  *message =
                      "final centroidal momentum bound validation failed";
                }
                return false;
              }
              ++selected;
            }
          }
          if (capture_point_constraint_.has_value() &&
              capture_point_constraint_->enabled) {
            const auto debug =
                evaluate_capture_point_constraint(q_eval, candidate);
            if (debug.status != SolverStatus::kSuccess ||
                debug.slacks.size() == 0 || !debug.slacks.allFinite() ||
                debug.slacks.minCoeff() < -tol) {
              if (message != nullptr) {
                *message = "final capture-point validation failed";
              }
              return false;
            }
          }
          if (velocity_zmp_constraint_.has_value() &&
              velocity_zmp_constraint_->enabled) {
            if (!pending_explicit_current_dq_.has_value()) {
              if (message != nullptr) {
                *message = "final velocity-ZMP validation lacks current_dq";
              }
              return false;
            }
            const auto debug = evaluate_velocity_zmp_constraint(
                q_eval, *pending_explicit_current_dq_, candidate);
            if (debug.status != SolverStatus::kSuccess ||
                debug.slacks.size() == 0 || !debug.slacks.allFinite() ||
                debug.slacks.minCoeff() < -tol ||
                debug.force_z <
                    velocity_zmp_constraint_->fz_min - tol) {
              if (message != nullptr) {
                *message = "final velocity-ZMP validation failed";
              }
              return false;
            }
          }
          return true;
        };
    if ((centroidal_momentum_bounds_.has_value() ||
         capture_point_constraint_.has_value() ||
         velocity_zmp_constraint_.has_value())) {
      std::string validation_message;
      if (!final_centroidal_constraints_acceptable(dq_final,
                                                   &validation_message)) {
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

  stall_handler_update(result);
  elastic_band_update(result);
  update_auto_task_layout_feedback(result);

  return result;
}


} // namespace embodik
