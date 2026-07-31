/**
 * @file kinematics_solver_position_step.cpp
 * @brief Position and position-step solve paths for KinematicsSolver
 */

#include "kinematics_solver_internal.hpp"

namespace embodik {

void KinematicsSolver::apply_position_step_primary_task_options(
    const PositionStepOptions &options, Task *task) {
  if (task == nullptr) {
    return;
  }
  if (options.primary_solve_mode != TaskSolveMode::kScale &&
      task->getSolveMode() != TaskSolveMode::kMinError) {
    task->setSolveMode(options.primary_solve_mode);
  }
  if (options.primary_allow_min_error_fallback) {
    task->setAllowMinErrorFallback(true);
  }
}

std::optional<VelocitySolverResult>
KinematicsSolver::apply_position_step_task_metric_projection(
    const Eigen::VectorXd &current_q, double outer_dt,
    const Eigen::VectorXd &first_tick_velocity,
    const std::vector<int> &velocity_lock_indices,
    const std::optional<TorsoPoseConstraintOptions> &torso_constraint,
    const std::vector<PositionStepPriorityConstraintSpec>
        &priority_constraints,
    const Eigen::VectorXd &current_dq,
    Eigen::VectorXd &q_candidate) {
  if (!acceleration_limits_enabled_ || current_q.size() != robot_->nq() ||
      q_candidate.size() != robot_->nq()) {
    return std::nullopt;
  }

  const Eigen::VectorXd predictive_delta =
      pinocchio::difference(robot_->model(), current_q, q_candidate);
  const bool have_terminal_prediction =
      predictive_delta.allFinite() &&
      predictive_delta.squaredNorm() > kCollisionEscapeNormEps;
  const bool have_first_tick_prediction =
      first_tick_velocity.size() == robot_->nv() &&
      first_tick_velocity.allFinite() &&
      first_tick_velocity.squaredNorm() > kCollisionEscapeNormEps;
  if (!have_terminal_prediction && !have_first_tick_prediction) {
    return std::nullopt;
  }

  const double dt_safe = std::max(outer_dt, 1e-9);
  std::optional<Eigen::VectorXd> previous_position_step_current_dq =
      position_step_explicit_current_dq_;
  position_step_explicit_current_dq_ =
      current_dq.size() == robot_->nv()
          ? std::optional<Eigen::VectorXd>(current_dq)
          : std::nullopt;
  struct RestoreProjectionCurrentVelocity {
    std::optional<Eigen::VectorXd> *slot;
    std::optional<Eigen::VectorXd> previous;
    ~RestoreProjectionCurrentVelocity() {
      if (slot != nullptr) {
        *slot = std::move(previous);
      }
    }
  } restore_projection_current_velocity{
      &position_step_explicit_current_dq_,
      std::move(previous_position_step_current_dq)};
  (void)restore_projection_current_velocity;
  const Eigen::VectorXd terminal_predictive_velocity =
      have_terminal_prediction ? predictive_delta / dt_safe
                               : first_tick_velocity;
  const Eigen::VectorXd first_tick_predictive_velocity =
      have_first_tick_prediction ? first_tick_velocity
                                 : terminal_predictive_velocity;
  robot_->update_configuration(current_q);

  bool have_task_objective = false;
  for (auto &task : tasks_) {
    if (!task || !task->isActive()) {
      continue;
    }

    task->update(*robot_);
    const Eigen::MatrixXd jacobian = task->getJacobian();
    const Eigen::VectorXd commanded_velocity = task->getVelocity();
    if (jacobian.cols() != robot_->nv() || jacobian.rows() <= 0 ||
        commanded_velocity.size() != jacobian.rows()) {
      continue;
    }

    const Eigen::VectorXd terminal_task_velocity =
        jacobian * terminal_predictive_velocity;
    const Eigen::VectorXd first_tick_task_velocity =
        jacobian * first_tick_predictive_velocity;
    if (!terminal_task_velocity.allFinite() ||
        !first_tick_task_velocity.allFinite()) {
      continue;
    }
    Eigen::VectorXd projected_velocity =
        Eigen::VectorXd::Zero(commanded_velocity.size());
    const double terminal_prediction_weight =
        position_step_task_terminal_prediction_weight(*task, task->getError());
    const Eigen::VectorXd selected_task_velocity =
        first_tick_task_velocity +
        terminal_prediction_weight *
            (terminal_task_velocity - first_tick_task_velocity);
    const auto protected_spec = std::find_if(
        priority_constraints.begin(), priority_constraints.end(),
        [&](const PositionStepPriorityConstraintSpec &spec) {
          return spec.task.get() == task.get();
        });

    if (task->getType() == TaskType::FRAME_POSE &&
        commanded_velocity.size() == 6) {
      const bool has_linear_command =
          commanded_velocity.head<3>().squaredNorm() >
          kCollisionEscapeNormEps;
      const bool has_angular_command =
          commanded_velocity.tail<3>().squaredNorm() >
          kCollisionEscapeNormEps;
      if (has_linear_command) {
        projected_velocity.head<3>() = selected_task_velocity.head<3>();
      }
      if (has_linear_command || has_angular_command) {
        projected_velocity.tail<3>() = selected_task_velocity.tail<3>();
      }
    } else {
      if (commanded_velocity.squaredNorm() > kCollisionEscapeNormEps) {
        projected_velocity = selected_task_velocity;
      }
    }
    if (protected_spec != priority_constraints.end()) {
      if (commanded_velocity.size() >= 6) {
        if (protected_spec->position_tolerance > 0.0) {
          projected_velocity.head<3>() = commanded_velocity.head<3>();
        }
        if (protected_spec->orientation_tolerance > 0.0) {
          projected_velocity.segment<3>(3) =
              commanded_velocity.segment<3>(3);
        }
      } else if (commanded_velocity.size() >= 3) {
        const bool orientation_only =
            task->getType() == TaskType::FRAME_ORIENTATION;
        const double protected_tolerance =
            orientation_only ? protected_spec->orientation_tolerance
                             : protected_spec->position_tolerance;
        if (protected_tolerance > 0.0) {
          projected_velocity.head<3>() = commanded_velocity.head<3>();
        }
      }
    }
    task->setPositionStepTargetVelocity(projected_velocity);
    have_task_objective = true;
  }
  if (!have_task_objective) {
    return std::nullopt;
  }

  pending_velocity_lock_indices_ = velocity_lock_indices;
  pending_step_torso_constraint_ = torso_constraint;
  pending_step_validation_dt_ = dt_safe;
  pending_position_step_priority_constraints_ = priority_constraints;
  pending_reuse_current_kinematics_ = true;
  pending_position_step_acceleration_limits_ = true;
  VelocitySolverResult projected = solve_velocity(current_q, true);
  projected = retry_auto_task_layout_as_split_if_needed(
      current_q, std::move(projected), velocity_lock_indices,
      torso_constraint, dt_safe, priority_constraints, true);
  if (projected.joint_velocities.size() != robot_->nv() ||
      !projected.joint_velocities.allFinite()) {
    robot_->update_configuration(q_candidate);
    return projected;
  }

  q_candidate = pinocchio::integrate(
      robot_->model(), current_q, dt_safe * projected.joint_velocities);
  if (use_position_limits_) {
    auto [q_min, q_max] = robot_->get_joint_limits();
    project_scalar_configuration_to_true_joint_limits(
        q_candidate, q_min, q_max, velocity_to_config_index_cache(),
        robot_->nv());
  }
  robot_->update_configuration(q_candidate);
  return projected;
}

std::optional<Eigen::VectorXd>
KinematicsSolver::estimate_position_step_componentwise_outer_candidate(
    const Eigen::VectorXd &current_q,
    const Eigen::VectorXd &previous_applied_velocity,
    const PositionStepOptions &options, double outer_dt,
    const Eigen::VectorXd &terminal_q) {
  if (!acceleration_limits_enabled_ || terminal_q.size() != robot_->nq()) {
    return std::nullopt;
  }

  const bool collision_was_enabled =
      collision_constraint_.has_value() && collision_constraint_->enabled;
  if (collision_was_enabled) {
    collision_constraint_->enabled = false;
  }
  Eigen::VectorXd candidate = terminal_q;
  const bool applied = apply_position_step_outer_acceleration_limit(
      current_q, previous_applied_velocity, options, outer_dt, candidate);
  if (collision_was_enabled) {
    collision_constraint_->enabled = true;
  }
  robot_->update_configuration(terminal_q);
  if (!applied || candidate.size() != robot_->nq() ||
      !candidate.allFinite()) {
    return std::nullopt;
  }
  return candidate;
}

bool KinematicsSolver::apply_position_step_outer_acceleration_limit(
    const Eigen::VectorXd &current_q,
    const Eigen::VectorXd &previous_applied_velocity,
    const PositionStepOptions &options, double outer_dt,
    Eigen::VectorXd &q_candidate) {
  if (!acceleration_limits_enabled_ || q_candidate.size() != robot_->nq() ||
      previous_applied_velocity.size() != robot_->nv() ||
      acceleration_limits_.size() != robot_->nv()) {
    return false;
  }

  const double dt_safe = std::max(outer_dt, 1e-9);
  const int nv = robot_->nv();
  Eigen::VectorXd desired_velocity =
      pinocchio::difference(robot_->model(), current_q, q_candidate) / dt_safe;
  Eigen::VectorXd velocity_lower = Eigen::VectorXd::Constant(
      nv, -kUnboundedConstraintLimit);
  Eigen::VectorXd velocity_upper = Eigen::VectorXd::Constant(
      nv, kUnboundedConstraintLimit);
  const Eigen::VectorXd velocity_limits = robot_->get_velocity_limits();
  Eigen::VectorXd position_lower;
  Eigen::VectorXd position_upper;
  std::vector<int> velocity_to_config_index;
  if (use_position_limits_) {
    auto limits = robot_->get_joint_limits();
    position_lower = std::move(limits.first);
    position_upper = std::move(limits.second);
    velocity_to_config_index = velocity_to_config_index_cache();
  }
  for (int i = 0; i < nv; ++i) {
    const double acceleration_step = acceleration_limits_[i] * dt_safe;
    const double deadband = acceleration_step * 0.01;
    velocity_lower[i] =
        previous_applied_velocity[i] - acceleration_step - deadband;
    velocity_upper[i] =
        previous_applied_velocity[i] + acceleration_step + deadband;
    if (i < velocity_limits.size() && std::isfinite(velocity_limits[i]) &&
        velocity_limits[i] > 0.0) {
      velocity_lower[i] =
          std::max(velocity_lower[i], -velocity_limits[i]);
      velocity_upper[i] =
          std::min(velocity_upper[i], velocity_limits[i]);
    }
    if (use_position_limits_ &&
        i < static_cast<int>(velocity_to_config_index.size())) {
      const int q_index = velocity_to_config_index[i];
      if (q_index >= 0 && q_index < current_q.size() &&
          q_index < position_lower.size() && q_index < position_upper.size() &&
          std::isfinite(position_lower[q_index]) &&
          std::isfinite(position_upper[q_index])) {
        constexpr double kOuterPositionMargin = 1e-4;
        const double lower_margin =
            current_q[q_index] - position_lower[q_index] - kOuterPositionMargin;
        const double upper_margin =
            position_upper[q_index] - current_q[q_index] - kOuterPositionMargin;
        const double velocity_limit =
            i < velocity_limits.size() && std::isfinite(velocity_limits[i]) &&
                    velocity_limits[i] > 0.0
                ? velocity_limits[i]
                : kUnboundedConstraintLimit;
        const auto [position_velocity_lower, position_velocity_upper] =
            calculate_velocity_box_constraint(
                lower_margin, upper_margin, velocity_limit,
                acceleration_limits_[i], dt_safe, 0.0, 0.0);
        const double combined_lower =
            std::max(velocity_lower[i], position_velocity_lower);
        const double combined_upper =
            std::min(velocity_upper[i], position_velocity_upper);
        if (combined_lower <= combined_upper + constraint_tolerance_) {
          velocity_lower[i] = combined_lower;
          velocity_upper[i] = combined_upper;
        } else {
          // The caller can enter outside the controlled invariant set. In that
          // case the hard position limit takes precedence over continuity.
          velocity_lower[i] = position_velocity_lower;
          velocity_upper[i] = position_velocity_upper;
        }
      }
    }
  }

  const auto lock_velocity = [&](const std::vector<int> &indices) {
    for (const int index : indices) {
      if (index >= 0 && index < nv) {
        velocity_lower[index] = 0.0;
        velocity_upper[index] = 0.0;
      }
    }
  };
  lock_velocity(options.excluded_joint_indices);
  lock_velocity(options.locked_joint_indices);
  lock_velocity(options.integration_zero_velocity_indices);

  Eigen::VectorXd limited_velocity = desired_velocity;
  Eigen::VectorXd minimum_norm_feasible_velocity(nv);
  for (int i = 0; i < nv; ++i) {
    limited_velocity[i] = std::clamp(
        limited_velocity[i], velocity_lower[i], velocity_upper[i]);
    minimum_norm_feasible_velocity[i] =
        std::clamp(0.0, velocity_lower[i], velocity_upper[i]);
  }
  bool minimum_norm_velocity_satisfies_collision_rows = true;

  if (collision_constraint_.has_value() && collision_constraint_->enabled) {
    robot_->update_configuration(current_q);
    const auto collision_rows = compute_collision_constraint();
    if (collision_rows.has_value() && collision_rows->jacobian.rows() > 0) {
      const int collision_row_count =
          static_cast<int>(collision_rows->jacobian.rows());
      Eigen::MatrixXd constraints = Eigen::MatrixXd::Zero(
          nv + collision_row_count, nv);
      constraints.topRows(nv).setIdentity();
      constraints.bottomRows(collision_row_count) =
          collision_rows->jacobian;
      Eigen::VectorXd lower(nv + collision_row_count);
      Eigen::VectorXd upper(nv + collision_row_count);
      lower.head(nv) = velocity_lower;
      upper.head(nv) = velocity_upper;
      lower.tail(collision_row_count) = collision_rows->lower_bounds;
      upper.tail(collision_row_count) = collision_rows->upper_bounds;

      VelocitySolverConfig projection_config;
      projection_config.epsilon = constraint_tolerance_;
      projection_config.precision_threshold = tight_tolerance_;
      projection_config.iteration_limit = max_iterations_;
      projection_config.magnitude_limit = norm_threshold_;
      projection_config.stall_detection_count = max_zero_scale_iterations_;
      projection_config.regularization_config.epsilon = solver_tolerance_;
      projection_config.regularization_config.regularization_factor = damping_;
      ObjectiveSolveConfig objective_config;
      objective_config.solve_mode = TaskSolveMode::kMinError;
      objective_config.allow_min_error_fallback = false;
      const auto projection = computeMultiObjectiveVelocitySolutionEigen(
          {desired_velocity}, {Eigen::MatrixXd::Identity(nv, nv)},
          constraints, lower, upper, projection_config, {objective_config});
      if (static_cast<int>(projection.solution.size()) == nv) {
        const Eigen::Map<const Eigen::VectorXd> projected_velocity(
            projection.solution.data(), nv);
        const Eigen::VectorXd constraint_values =
            constraints * projected_velocity;
        const bool projection_feasible = projected_velocity.allFinite() &&
            (constraint_values.array() >=
             (lower.array() - 10.0 * constraint_tolerance_)).all() &&
            (constraint_values.array() <=
             (upper.array() + 10.0 * constraint_tolerance_)).all();
        if (projection_feasible) {
          limited_velocity = projected_velocity;
        }
      }
      const double max_outer_velocity_norm =
          options.max_configuration_step_norm > 0.0
              ? options.max_configuration_step_norm / dt_safe
              : std::numeric_limits<double>::infinity();
      if (limited_velocity.norm() >
          max_outer_velocity_norm + 10.0 * constraint_tolerance_) {
        const auto minimum_norm_projection =
            computeMultiObjectiveVelocitySolutionEigen(
                {Eigen::VectorXd::Zero(nv)},
                {Eigen::MatrixXd::Identity(nv, nv)}, constraints, lower,
                upper, projection_config, {objective_config});
        minimum_norm_velocity_satisfies_collision_rows = false;
        if (static_cast<int>(minimum_norm_projection.solution.size()) == nv) {
          const Eigen::Map<const Eigen::VectorXd> minimum_norm_velocity(
              minimum_norm_projection.solution.data(), nv);
          const Eigen::VectorXd minimum_norm_constraint_values =
              constraints * minimum_norm_velocity;
          const bool minimum_norm_projection_feasible =
              minimum_norm_velocity.allFinite() &&
              (minimum_norm_constraint_values.array() >=
               (lower.array() - 10.0 * constraint_tolerance_))
                  .all() &&
              (minimum_norm_constraint_values.array() <=
               (upper.array() + 10.0 * constraint_tolerance_))
                  .all();
          if (minimum_norm_projection_feasible) {
            minimum_norm_feasible_velocity = minimum_norm_velocity;
            minimum_norm_velocity_satisfies_collision_rows = true;
          }
        }
      }
    }
  }

  const double max_outer_velocity_norm =
      options.max_configuration_step_norm > 0.0
          ? options.max_configuration_step_norm / dt_safe
          : std::numeric_limits<double>::infinity();
  if (limited_velocity.norm() >
      max_outer_velocity_norm + 10.0 * constraint_tolerance_) {
    if (minimum_norm_velocity_satisfies_collision_rows &&
        minimum_norm_feasible_velocity.norm() <=
            max_outer_velocity_norm + 10.0 * constraint_tolerance_) {
      double feasible_fraction = 0.0;
      double infeasible_fraction = 1.0;
      for (int iteration = 0; iteration < 40; ++iteration) {
        const double fraction =
            0.5 * (feasible_fraction + infeasible_fraction);
        const Eigen::VectorXd trial_velocity =
            minimum_norm_feasible_velocity +
            fraction *
                (limited_velocity - minimum_norm_feasible_velocity);
        if (trial_velocity.norm() <= max_outer_velocity_norm) {
          feasible_fraction = fraction;
        } else {
          infeasible_fraction = fraction;
        }
      }
      limited_velocity =
          minimum_norm_feasible_velocity +
          feasible_fraction *
              (limited_velocity - minimum_norm_feasible_velocity);
    } else if (minimum_norm_velocity_satisfies_collision_rows) {
      // The norm cap is infeasible with the acceleration/collision rows. Keep
      // the minimum-norm hard-constraint solution instead of radially scaling
      // outside the acceleration box.
      limited_velocity = minimum_norm_feasible_velocity;
    }
  }

  const auto configuration_for_velocity =
      [&](const Eigen::VectorXd &reference_q,
          const Eigen::VectorXd &velocity) {
        Eigen::VectorXd candidate_q = pinocchio::integrate(
            robot_->model(), reference_q, dt_safe * velocity);
        if (use_position_limits_) {
          auto [q_min, q_max] = robot_->get_joint_limits();
          project_scalar_configuration_to_true_joint_limits(
              candidate_q, q_min, q_max, velocity_to_config_index_cache(),
              robot_->nv());
        }
        return candidate_q;
      };

  Eigen::VectorXd limited_q =
      configuration_for_velocity(current_q, limited_velocity);

  if (collision_constraint_.has_value() && collision_constraint_->enabled) {
    robot_->update_configuration(current_q);
    const auto current_recovery_margins =
        evaluate_post_step_collision_recovery_margins(current_q);
    const auto current_distance =
        evaluate_post_step_collision_distance_from_current_results();
    const double nominal_floor = collision_constraint_->min_distance;

    const auto collision_acceptable =
        [&](const Eigen::VectorXd &candidate,
            double recovery_margin_threshold) {
      robot_->update_configuration(candidate);
      const auto candidate_recovery_margins =
          evaluate_post_step_collision_recovery_margins(candidate);
      const auto candidate_distance =
          evaluate_post_step_collision_distance_from_current_results();
      if (!collision_recovery_margins_acceptable(
              current_recovery_margins, candidate_recovery_margins,
              recovery_margin_threshold)) {
        return false;
      }
      if (!current_distance.has_value() ||
          !std::isfinite(*current_distance) ||
          !candidate_distance.has_value() ||
          !std::isfinite(*candidate_distance)) {
        return true;
      }
      if (*current_distance >= nominal_floor - kCollisionTolerance) {
        return *candidate_distance >= nominal_floor - kCollisionTolerance;
      }
      return *candidate_distance >=
             *current_distance - kCollisionPenetrationWorsenTolerance;
    };

    const auto braking_velocity_from = [&](const Eigen::VectorXd &velocity) {
      Eigen::VectorXd braking_velocity = velocity;
      for (int i = 0; i < nv; ++i) {
        const double acceleration_step = acceleration_limits_[i] * dt_safe;
        const double deadband = acceleration_step * 0.01;
        const double step = acceleration_step + deadband;
        if (braking_velocity[i] > step) {
          braking_velocity[i] -= step;
        } else if (braking_velocity[i] < -step) {
          braking_velocity[i] += step;
        } else {
          braking_velocity[i] = 0.0;
        }
      }
      return braking_velocity;
    };

    const auto componentwise_braking_rollout_acceptable =
        [&](const Eigen::VectorXd &candidate_q,
            const Eigen::VectorXd &candidate_velocity,
            double minimum_recovery_margin,
            Eigen::VectorXd *first_q_out,
            Eigen::VectorXd *first_velocity_out) {
          if (!acceleration_limits_enabled_) {
            return true;
          }

          int braking_steps = 0;
          for (int i = 0; i < nv; ++i) {
            const double acceleration_step = acceleration_limits_[i] * dt_safe;
            const double deadband = acceleration_step * 0.01;
            const double step = acceleration_step + deadband;
            if (step <= constraint_tolerance_) {
              if (std::abs(candidate_velocity[i]) > constraint_tolerance_) {
                return false;
              }
              continue;
            }
            braking_steps = std::max(
                braking_steps,
                static_cast<int>(std::ceil(
                    std::abs(candidate_velocity[i]) / step - 1e-12)));
          }

          Eigen::VectorXd rollout_q = candidate_q;
          Eigen::VectorXd rollout_velocity = candidate_velocity;
          Eigen::VectorXd first_q;
          Eigen::VectorXd first_velocity;
          for (int step_index = 0; step_index < braking_steps; ++step_index) {
            rollout_velocity = braking_velocity_from(rollout_velocity);
            rollout_q =
                configuration_for_velocity(rollout_q, rollout_velocity);
            if (step_index == 0) {
              first_q = rollout_q;
              first_velocity = rollout_velocity;
            }
            if (!collision_acceptable(rollout_q, minimum_recovery_margin)) {
              return false;
            }
          }
          if (braking_steps == 0) {
            first_q = candidate_q;
            first_velocity = candidate_velocity;
          }
          if (first_q_out != nullptr) {
            *first_q_out = std::move(first_q);
          }
          if (first_velocity_out != nullptr) {
            *first_velocity_out = std::move(first_velocity);
          }
          return true;
        };

    const auto collision_aware_braking_rollout_acceptable =
        [&](const Eigen::VectorXd &candidate_q,
            const Eigen::VectorXd &candidate_velocity,
            double minimum_recovery_margin,
            Eigen::VectorXd *first_q_out,
            Eigen::VectorXd *first_velocity_out) {
          int braking_step_budget = 0;
          for (int i = 0; i < nv; ++i) {
            const double acceleration_step = acceleration_limits_[i] * dt_safe;
            const double deadband = acceleration_step * 0.01;
            const double step = acceleration_step + deadband;
            if (step <= constraint_tolerance_) {
              if (std::abs(candidate_velocity[i]) > constraint_tolerance_) {
                return false;
              }
              continue;
            }
            braking_step_budget += static_cast<int>(std::ceil(
                std::abs(candidate_velocity[i]) / step - 1e-12));
          }
          if (braking_step_budget == 0) {
            if (first_q_out != nullptr) {
              *first_q_out = candidate_q;
            }
            if (first_velocity_out != nullptr) {
              *first_velocity_out = candidate_velocity;
            }
            return true;
          }

          VelocitySolverConfig rollout_solver_config;
          rollout_solver_config.epsilon = constraint_tolerance_;
          rollout_solver_config.precision_threshold = tight_tolerance_;
          rollout_solver_config.iteration_limit = max_iterations_;
          rollout_solver_config.magnitude_limit = norm_threshold_;
          rollout_solver_config.stall_detection_count =
              max_zero_scale_iterations_;
          rollout_solver_config.regularization_config.epsilon =
              solver_tolerance_;
          rollout_solver_config.regularization_config.regularization_factor =
              damping_;
          ObjectiveSolveConfig rollout_objective_config;
          rollout_objective_config.solve_mode = TaskSolveMode::kMinError;
          rollout_objective_config.allow_min_error_fallback = false;

          Eigen::VectorXd rollout_q = candidate_q;
          Eigen::VectorXd rollout_velocity = candidate_velocity;
          Eigen::VectorXd first_q;
          Eigen::VectorXd first_velocity;
          bool have_first_step = false;
          const auto publish_first_step = [&]() {
            if (!have_first_step) {
              return;
            }
            if (first_q_out != nullptr) {
              *first_q_out = first_q;
            }
            if (first_velocity_out != nullptr) {
              *first_velocity_out = first_velocity;
            }
          };
          const auto fail_with_first_step = [&]() {
            publish_first_step();
            return false;
          };
          for (int step_index = 0; step_index < braking_step_budget;
               ++step_index) {
            Eigen::VectorXd successor_lower(nv);
            Eigen::VectorXd successor_upper(nv);
            for (int i = 0; i < nv; ++i) {
              const double acceleration_step =
                  acceleration_limits_[i] * dt_safe;
              const double deadband = acceleration_step * 0.01;
              successor_lower[i] =
                  rollout_velocity[i] - acceleration_step - deadband;
              successor_upper[i] =
                  rollout_velocity[i] + acceleration_step + deadband;
              if (i < velocity_limits.size() &&
                  std::isfinite(velocity_limits[i]) &&
                  velocity_limits[i] > 0.0) {
                successor_lower[i] =
                    std::max(successor_lower[i], -velocity_limits[i]);
                successor_upper[i] =
                    std::min(successor_upper[i], velocity_limits[i]);
              }
              if (use_position_limits_ &&
                  i < static_cast<int>(velocity_to_config_index.size())) {
                const int q_index = velocity_to_config_index[i];
                if (q_index >= 0 && q_index < rollout_q.size() &&
                    q_index < position_lower.size() &&
                    q_index < position_upper.size() &&
                    std::isfinite(position_lower[q_index]) &&
                    std::isfinite(position_upper[q_index])) {
                  constexpr double kOuterPositionMargin = 1e-4;
                  const double lower_margin =
                      rollout_q[q_index] - position_lower[q_index] -
                      kOuterPositionMargin;
                  const double upper_margin =
                      position_upper[q_index] - rollout_q[q_index] -
                      kOuterPositionMargin;
                  const double velocity_limit =
                      i < velocity_limits.size() &&
                              std::isfinite(velocity_limits[i]) &&
                              velocity_limits[i] > 0.0
                          ? velocity_limits[i]
                          : kUnboundedConstraintLimit;
                  const auto [position_velocity_lower,
                              position_velocity_upper] =
                      calculate_velocity_box_constraint(
                          lower_margin, upper_margin, velocity_limit,
                          acceleration_limits_[i], dt_safe, 0.0, 0.0);
                  const double combined_lower =
                      std::max(successor_lower[i], position_velocity_lower);
                  const double combined_upper =
                      std::min(successor_upper[i], position_velocity_upper);
                  if (combined_lower <=
                      combined_upper + constraint_tolerance_) {
                    successor_lower[i] = combined_lower;
                    successor_upper[i] = combined_upper;
                  } else {
                    successor_lower[i] = position_velocity_lower;
                    successor_upper[i] = position_velocity_upper;
                  }
                }
              }
            }
            const auto lock_successor_velocity =
                [&](const std::vector<int> &indices) {
                  for (const int index : indices) {
                    if (index >= 0 && index < nv) {
                      successor_lower[index] = 0.0;
                      successor_upper[index] = 0.0;
                    }
                  }
                };
            lock_successor_velocity(options.excluded_joint_indices);
            lock_successor_velocity(options.locked_joint_indices);
            lock_successor_velocity(
                options.integration_zero_velocity_indices);

            robot_->update_configuration(rollout_q);
            const auto rollout_collision_rows =
                compute_collision_constraint();
            if (!rollout_collision_rows.has_value() ||
                rollout_collision_rows->jacobian.rows() == 0) {
              return fail_with_first_step();
            }

            const int collision_row_count = static_cast<int>(
                rollout_collision_rows->jacobian.rows());
            Eigen::MatrixXd constraints = Eigen::MatrixXd::Zero(
                nv + collision_row_count, nv);
            constraints.topRows(nv).setIdentity();
            constraints.bottomRows(collision_row_count) =
                rollout_collision_rows->jacobian;
            Eigen::VectorXd lower(nv + collision_row_count);
            Eigen::VectorXd upper(nv + collision_row_count);
            lower.head(nv) = successor_lower;
            upper.head(nv) = successor_upper;
            lower.tail(collision_row_count) =
                rollout_collision_rows->lower_bounds;
            upper.tail(collision_row_count) =
                rollout_collision_rows->upper_bounds;

            Eigen::VectorXd terminal_lower = lower;
            terminal_lower.tail(collision_row_count) =
                terminal_lower.tail(collision_row_count)
                    .cwiseMax(Eigen::VectorXd::Zero(collision_row_count));
            const auto terminal_solution =
                computeMultiObjectiveVelocitySolutionEigen(
                    {braking_velocity_from(rollout_velocity)},
                    {Eigen::MatrixXd::Identity(nv, nv)}, constraints,
                    terminal_lower, upper, rollout_solver_config,
                    {rollout_objective_config});
            if (static_cast<int>(terminal_solution.solution.size()) == nv) {
              const Eigen::Map<const Eigen::VectorXd> terminal_velocity_map(
                  terminal_solution.solution.data(), nv);
              const Eigen::VectorXd terminal_velocity = terminal_velocity_map;
              const Eigen::VectorXd terminal_constraint_values =
                  constraints * terminal_velocity;
              const bool terminal_feasible = terminal_velocity.allFinite() &&
                  (terminal_constraint_values.array() >=
                   (terminal_lower.array() -
                    10.0 * constraint_tolerance_))
                      .all() &&
                  (terminal_constraint_values.array() <=
                   (upper.array() + 10.0 * constraint_tolerance_))
                      .all();
              if (terminal_feasible) {
                Eigen::VectorXd terminal_q = configuration_for_velocity(
                    rollout_q, terminal_velocity);
                const Eigen::VectorXd applied_terminal_velocity =
                    pinocchio::difference(robot_->model(), rollout_q,
                                          terminal_q) /
                    dt_safe;
                bool terminal_acceleration_feasible = true;
                for (int i = 0; i < nv; ++i) {
                  if (applied_terminal_velocity[i] <
                          successor_lower[i] -
                              10.0 * constraint_tolerance_ ||
                      applied_terminal_velocity[i] >
                          successor_upper[i] +
                              10.0 * constraint_tolerance_) {
                    terminal_acceleration_feasible = false;
                    break;
                  }
                }
                if (terminal_acceleration_feasible &&
                    collision_acceptable(terminal_q,
                                         minimum_recovery_margin)) {
                  robot_->update_configuration(terminal_q);
                  const auto terminal_collision_rows =
                      compute_collision_constraint();
                  if (!terminal_collision_rows.has_value() ||
                      terminal_collision_rows->jacobian.rows() == 0 ||
                      ((terminal_collision_rows->jacobian *
                        applied_terminal_velocity)
                               .array() >=
                           -10.0 * constraint_tolerance_)
                          .all()) {
                    rollout_q = std::move(terminal_q);
                    rollout_velocity = applied_terminal_velocity;
                    if (!have_first_step) {
                      first_q = rollout_q;
                      first_velocity = rollout_velocity;
                      have_first_step = true;
                    }
                    if (rollout_velocity.norm() <=
                        10.0 * constraint_tolerance_) {
                      if (first_q_out != nullptr) {
                        *first_q_out = std::move(first_q);
                      }
                      if (first_velocity_out != nullptr) {
                        *first_velocity_out = std::move(first_velocity);
                      }
                      return true;
                    }
                    continue;
                  }
                }
              }
            }

            const auto rollout_solution =
                computeMultiObjectiveVelocitySolutionEigen(
                    {Eigen::VectorXd::Zero(nv)},
                    {Eigen::MatrixXd::Identity(nv, nv)}, constraints, lower,
                    upper, rollout_solver_config,
                    {rollout_objective_config});
            if (static_cast<int>(rollout_solution.solution.size()) != nv) {
              return fail_with_first_step();
            }
            const Eigen::Map<const Eigen::VectorXd> next_velocity_map(
                rollout_solution.solution.data(), nv);
            const Eigen::VectorXd next_velocity = next_velocity_map;
            const Eigen::VectorXd constraint_values =
                constraints * next_velocity;
            const bool feasible = next_velocity.allFinite() &&
                (constraint_values.array() >=
                 (lower.array() - 10.0 * constraint_tolerance_))
                    .all() &&
                (constraint_values.array() <=
                 (upper.array() + 10.0 * constraint_tolerance_))
                    .all();
            if (!feasible) {
              return fail_with_first_step();
            }

            Eigen::VectorXd next_q =
                configuration_for_velocity(rollout_q, next_velocity);
            const Eigen::VectorXd applied_next_velocity =
                pinocchio::difference(robot_->model(), rollout_q, next_q) /
                dt_safe;
            for (int i = 0; i < nv; ++i) {
              if (applied_next_velocity[i] <
                      successor_lower[i] - 10.0 * constraint_tolerance_ ||
                  applied_next_velocity[i] >
                      successor_upper[i] + 10.0 * constraint_tolerance_) {
                return fail_with_first_step();
              }
            }
            if (!collision_acceptable(next_q, minimum_recovery_margin)) {
              return fail_with_first_step();
            }
            rollout_q = std::move(next_q);
            rollout_velocity = applied_next_velocity;
            if (!have_first_step) {
              first_q = rollout_q;
              first_velocity = rollout_velocity;
              have_first_step = true;
            }
          }
          publish_first_step();
          return false;
        };

    const auto braking_rollout_acceptable =
        [&](const Eigen::VectorXd &candidate_q,
            const Eigen::VectorXd &candidate_velocity,
            double minimum_recovery_margin,
            Eigen::VectorXd *first_q_out,
            Eigen::VectorXd *first_velocity_out) {
          return componentwise_braking_rollout_acceptable(
                     candidate_q, candidate_velocity, minimum_recovery_margin,
                     first_q_out, first_velocity_out) ||
                 collision_aware_braking_rollout_acceptable(
                     candidate_q, candidate_velocity, minimum_recovery_margin,
                     first_q_out, first_velocity_out);
        };

    const auto candidate_and_braking_rollout_acceptable =
        [&](const Eigen::VectorXd &candidate_q,
            const Eigen::VectorXd &candidate_velocity) {
          return collision_acceptable(candidate_q, 0.0) &&
                 braking_rollout_acceptable(candidate_q, candidate_velocity,
                                             kCollisionTolerance, nullptr,
                                             nullptr);
        };

    if (!candidate_and_braking_rollout_acceptable(limited_q,
                                                   limited_velocity)) {
      bool accepted_backoff = false;
      if (acceleration_limits_enabled_) {
        Eigen::VectorXd braking_q;
        Eigen::VectorXd braking_velocity;
        const bool full_braking_rollout_acceptable =
            braking_rollout_acceptable(
                current_q, previous_applied_velocity, 0.0, &braking_q,
                &braking_velocity);
        const bool have_safe_braking_step =
            braking_q.size() == robot_->nq() &&
            braking_velocity.size() == robot_->nv() &&
            braking_q.allFinite() && braking_velocity.allFinite();
        if ((full_braking_rollout_acceptable || have_safe_braking_step) &&
            collision_acceptable(braking_q, 0.0)) {
          Eigen::VectorXd best_velocity = braking_velocity;
          Eigen::VectorXd best_q = std::move(braking_q);
          double safe_fraction = 0.0;
          double unsafe_fraction = 1.0;
          // Three probes retain useful task motion to 12.5% resolution while
          // bounding the repeated braking-rollout work inside a WBC tick.
          for (int iteration = 0; iteration < 3; ++iteration) {
            const double fraction =
                0.5 * (safe_fraction + unsafe_fraction);
            const Eigen::VectorXd trial_velocity =
                braking_velocity +
                fraction * (limited_velocity - braking_velocity);
            Eigen::VectorXd trial_q =
                configuration_for_velocity(current_q, trial_velocity);
            if (candidate_and_braking_rollout_acceptable(trial_q,
                                                          trial_velocity)) {
              safe_fraction = fraction;
              best_velocity = trial_velocity;
              best_q = std::move(trial_q);
            } else {
              unsafe_fraction = fraction;
            }
          }
          limited_velocity = std::move(best_velocity);
          limited_q = std::move(best_q);
          accepted_backoff = true;
        }
      }
      if (!accepted_backoff && !acceleration_limits_enabled_) {
        for (const double fraction : kCollisionRejectionBackoffFractions) {
          const Eigen::VectorXd backoff_velocity =
              fraction * limited_velocity;
          Eigen::VectorXd backoff_q =
              configuration_for_velocity(current_q, backoff_velocity);
          if (collision_acceptable(backoff_q, 0.0)) {
            limited_velocity = backoff_velocity;
            limited_q = std::move(backoff_q);
            accepted_backoff = true;
            break;
          }
        }
      }
      if (!accepted_backoff) {
        limited_velocity.setZero();
        limited_q = current_q;
      }
    }
  }

  q_candidate = std::move(limited_q);
  robot_->update_configuration(q_candidate);
  return true;
}

bool KinematicsSolver::compute_position_step_continuity_brake(
    const Eigen::VectorXd &current_q,
    const Eigen::VectorXd &previous_applied_velocity,
    const PositionStepOptions &options, double outer_dt,
    Eigen::VectorXd &q_candidate) {
  if (!acceleration_limits_enabled_ ||
      previous_applied_velocity.size() != robot_->nv() ||
      acceleration_limits_.size() != robot_->nv()) {
    return false;
  }

  const double dt_safe = std::max(outer_dt, 1e-9);
  Eigen::VectorXd braking_velocity = previous_applied_velocity;
  for (int i = 0; i < robot_->nv(); ++i) {
    const double acceleration_step = acceleration_limits_[i] * dt_safe;
    const double deadband = acceleration_step * 0.01;
    const double step = acceleration_step + deadband;
    if (braking_velocity[i] > step) {
      braking_velocity[i] -= step;
    } else if (braking_velocity[i] < -step) {
      braking_velocity[i] += step;
    } else {
      braking_velocity[i] = 0.0;
    }
  }

  q_candidate = pinocchio::integrate(robot_->model(), current_q,
                                     dt_safe * braking_velocity);
  apply_position_step_outer_acceleration_limit(
      current_q, previous_applied_velocity, options, dt_safe, q_candidate);

  const Eigen::VectorXd applied_velocity =
      pinocchio::difference(robot_->model(), current_q, q_candidate) / dt_safe;
  if (applied_velocity.size() != robot_->nv() || !applied_velocity.allFinite()) {
    q_candidate = current_q;
    robot_->update_configuration(q_candidate);
    return false;
  }
  for (int i = 0; i < robot_->nv(); ++i) {
    const double allowed_change =
        acceleration_limits_[i] * dt_safe * 1.01 + 10.0 * constraint_tolerance_;
    if (std::abs(applied_velocity[i] - previous_applied_velocity[i]) >
        allowed_change) {
      q_candidate = current_q;
      robot_->update_configuration(q_candidate);
      return false;
    }
  }
  return applied_velocity.norm() > 10.0 * constraint_tolerance_;
}

std::optional<PositionIKResult>
KinematicsSolver::attempt_min_error_position_step_retry(
    const Eigen::VectorXd &entry_q, const PositionStepOptions &options,
    double step_dt, bool have_vel_result,
    const VelocitySolverResult &last_vel_result,
    const Eigen::VectorXd &q_after_primary,
    const std::function<bool(std::vector<std::pair<Task *, TaskSolveMode>> *)>
        &flip_primary_tasks,
    const std::function<PositionIKResult()> &rerun_step) {
  if (!options.primary_allow_min_error_fallback || suppress_min_error_step_retry_ ||
      !have_vel_result) {
    return std::nullopt;
  }

  const double applied_step_eps =
      stall_config_.dq_stall_eps * std::max(step_dt, 1e-9);
  const double applied_step_norm = (q_after_primary - entry_q).norm();
  const bool constraints_active =
      (collision_constraint_.has_value() && collision_constraint_->enabled) ||
      (com_constraint_.has_value() && com_constraint_->enabled);
  const bool scale_collapsed =
      last_vel_result.status == SolverStatus::kInfeasible &&
      is_primary_scale_collapsed_message(last_vel_result.status_message);
  const bool numerical_stall =
      last_vel_result.status == SolverStatus::kNumericalError && constraints_active;
  constexpr double kPrimaryScaleEps = 1e-4;
  const bool primary_scale_saturated =
      constraints_active && !last_vel_result.task_scales.empty() &&
      std::abs(last_vel_result.task_scales.front()) <= kPrimaryScaleEps;
  const bool soft_saturated =
      primary_scale_saturated &&
      (last_vel_result.status == SolverStatus::kSuccess ||
       last_vel_result.status == SolverStatus::kNoProgress ||
       last_vel_result.status == SolverStatus::kInfeasible);
  const bool primary_saturation_retry = scale_collapsed || soft_saturated;
  const bool zero_motion_numerical_retry =
      numerical_stall && applied_step_norm <= applied_step_eps;
  if (!primary_saturation_retry && !zero_motion_numerical_retry) {
    return std::nullopt;
  }

  const PositionStepMutableStateSnapshot retry_snapshot =
      capture_position_step_mutable_state();
  std::vector<std::pair<Task *, TaskSolveMode>> saved_modes;
  if (!flip_primary_tasks(&saved_modes)) {
    return std::nullopt;
  }

  const TaskLayout saved_layout = current_auto_task_layout_;
  if (runtime_config_.enable_auto_task_layout) {
    current_auto_task_layout_ = TaskLayout::kSplit;
    auto_layout_below_low_count_ = 0;
    for (auto &kv : pose_task_groups_) {
      if (kv.second && kv.second->auto_switch()) {
        kv.second->set_layout(TaskLayout::kSplit);
      }
    }
  }

  suppress_min_error_step_retry_ = true;
  PositionIKResult retry_result = rerun_step();
  suppress_min_error_step_retry_ = false;

  restore_task_solve_modes(saved_modes);
  if (runtime_config_.enable_auto_task_layout) {
    current_auto_task_layout_ = saved_layout;
    for (auto &kv : pose_task_groups_) {
      if (kv.second && kv.second->auto_switch()) {
        kv.second->set_layout(saved_layout);
      }
    }
  }

  const double retry_norm = (retry_result.q_solution - entry_q).norm();
  const bool accept_retry =
      retry_norm > applied_step_eps ||
      retry_result.status == SolverStatus::kSuccess;
  if (!accept_retry) {
    restore_position_step_mutable_state(retry_snapshot);
    return std::nullopt;
  }
  return retry_result;
}

KinematicsSolver::PositionStepMutableStateSnapshot
KinematicsSolver::capture_position_step_mutable_state() const {
  PositionStepMutableStateSnapshot snapshot;
  snapshot.robot_q = robot_->get_current_configuration();
  snapshot.task_states.reserve(tasks_.size());
  for (const auto &task : tasks_) {
    if (!task) {
      continue;
    }
    snapshot.task_states.push_back(
        {task.get(), task->getSolveMode(), task->getAllowMinErrorFallback(),
         task->getLastEffectiveMode(), task->getUsedMinErrorFallback()});
  }
  snapshot.current_auto_task_layout = current_auto_task_layout_;
  snapshot.auto_layout_below_low_count = auto_layout_below_low_count_;
  snapshot.auto_layout_has_feedback = auto_layout_has_feedback_;
  snapshot.auto_layout_binding_score = auto_layout_binding_score_;
  snapshot.advisor_scale_current = advisor_scale_current_;
  snapshot.advisor_scale_ratio_sum = advisor_scale_ratio_sum_;
  snapshot.advisor_scale_epoch_time_s = advisor_scale_epoch_time_s_;
  snapshot.advisor_scale_sample_count = advisor_scale_sample_count_;
  snapshot.previous_dq = previous_dq_;
  snapshot.position_step_merit_window_anchor =
      position_step_merit_window_anchor_;
  snapshot.position_step_merit_priority = position_step_merit_priority_;
  snapshot.position_step_stationary_anchor_blocks =
      position_step_stationary_anchor_blocks_;
  snapshot.position_step_merit_window_motion =
      position_step_merit_window_motion_;
  snapshot.position_step_merit_window_samples =
      position_step_merit_window_samples_;
  snapshot.position_step_merit_window_last_delta =
      position_step_merit_window_last_delta_;
  snapshot.position_step_merit_window_direction_reversals =
      position_step_merit_window_direction_reversals_;
  snapshot.position_step_merit_window_last_merits =
      position_step_merit_window_last_merits_;
  snapshot.position_step_merit_window_error_increases =
      position_step_merit_window_error_increases_;
  snapshot.position_step_stationary_guard_active =
      position_step_stationary_guard_active_;
  snapshot.position_step_stationary_guard_can_reopen =
      position_step_stationary_guard_can_reopen_;
  snapshot.stall_config = stall_config_;
  snapshot.stall_state = stall_state_;
  snapshot.stall_user_configured = stall_user_configured_;
  snapshot.elastic_band_config = elastic_band_config_;
  snapshot.elastic_band_state = elastic_band_state_;
  snapshot.collision_constraint = collision_constraint_;
  snapshot.per_pair_min_distance_overrides = per_pair_min_distance_overrides_;
  snapshot.per_pair_deferred_overrides = per_pair_deferred_overrides_;
  snapshot.collision_pair_distance_floor = collision_pair_distance_floor_;
  snapshot.position_step_collision_command_floor_distances =
      position_step_collision_command_floor_distances_;
  snapshot.collision_cache_frozen_indices = collision_cache_frozen_indices_;
  snapshot.last_collision_debug = last_collision_debug_;
  snapshot.last_collision_debug_list = last_collision_debug_list_;
  snapshot.last_collision_constraint_pair_indices =
      last_collision_constraint_pair_indices_;
  snapshot.collision_cached_candidate_pair_indices =
      collision_cached_candidate_pair_indices_;
  snapshot.collision_pair_bound_valid = collision_pair_bound_valid_;
  snapshot.collision_pair_last_signed_distance =
      collision_pair_last_signed_distance_;
  snapshot.collision_pair_last_rel_translation_norm =
      collision_pair_last_rel_translation_norm_;
  snapshot.collision_pair_last_rel_rotation =
      collision_pair_last_rel_rotation_;
  snapshot.collision_pair_cache_has_full_scan =
      collision_pair_cache_has_full_scan_;
  snapshot.collision_pair_cache_steps_since_refresh =
      collision_pair_cache_steps_since_refresh_;
  snapshot.last_collision_pairs_considered = last_collision_pairs_considered_;
  snapshot.last_collision_exact_distance_queries =
      last_collision_exact_distance_queries_;
  snapshot.last_collision_bound_culled_pairs =
      last_collision_bound_culled_pairs_;
  snapshot.last_collision_budget_exhausted = last_collision_budget_exhausted_;
  snapshot.last_constraint_min_distance = last_constraint_min_distance_;
  snapshot.last_constraint_min_recovery_margin =
      last_constraint_min_recovery_margin_;
  snapshot.last_constraint_was_full_scan = last_constraint_was_full_scan_;
  snapshot.last_collision_constraint_result =
      last_collision_constraint_result_;
  snapshot.last_collision_constraint_row_dt =
      last_collision_constraint_row_dt_;
  snapshot.last_collision_constraint_q = last_collision_constraint_q_;
  snapshot.post_step_collision_distance_cache =
      post_step_collision_distance_cache_;
  snapshot.last_post_step_collision_exact_distance_queries =
      last_post_step_collision_exact_distance_queries_;
  snapshot.last_post_step_collision_motion_bound_culled_pairs =
      last_post_step_collision_motion_bound_culled_pairs_;
  snapshot.last_collision_sphere_culled_pairs =
      last_collision_sphere_culled_pairs_;
  snapshot.last_solution_dq_norm = last_solution_dq_norm_;
  snapshot.collision_stuck_counters = collision_stuck_counters_;
  snapshot.collision_stuck_last_distances = collision_stuck_last_distances_;
  return snapshot;
}

void KinematicsSolver::restore_position_step_mutable_state(
    const PositionStepMutableStateSnapshot &snapshot) {
  if (snapshot.robot_q.size() == robot_->nq()) {
    robot_->update_configuration(snapshot.robot_q);
  }
  for (const auto &state : snapshot.task_states) {
    if (state.task == nullptr) {
      continue;
    }
    state.task->setSolveMode(state.solve_mode);
    state.task->setAllowMinErrorFallback(state.allow_min_error_fallback);
    state.task->setLastEffectiveMode(state.last_effective_mode);
    state.task->setUsedMinErrorFallback(state.used_min_error_fallback);
  }
  current_auto_task_layout_ = snapshot.current_auto_task_layout;
  auto_layout_below_low_count_ = snapshot.auto_layout_below_low_count;
  auto_layout_has_feedback_ = snapshot.auto_layout_has_feedback;
  auto_layout_binding_score_ = snapshot.auto_layout_binding_score;
  advisor_scale_current_ = snapshot.advisor_scale_current;
  advisor_scale_ratio_sum_ = snapshot.advisor_scale_ratio_sum;
  advisor_scale_epoch_time_s_ = snapshot.advisor_scale_epoch_time_s;
  advisor_scale_sample_count_ = snapshot.advisor_scale_sample_count;
  previous_dq_ = snapshot.previous_dq;
  position_step_merit_window_anchor_ =
      snapshot.position_step_merit_window_anchor;
  position_step_merit_priority_ = snapshot.position_step_merit_priority;
  position_step_stationary_anchor_blocks_ =
      snapshot.position_step_stationary_anchor_blocks;
  position_step_merit_window_motion_ =
      snapshot.position_step_merit_window_motion;
  position_step_merit_window_samples_ =
      snapshot.position_step_merit_window_samples;
  position_step_merit_window_last_delta_ =
      snapshot.position_step_merit_window_last_delta;
  position_step_merit_window_direction_reversals_ =
      snapshot.position_step_merit_window_direction_reversals;
  position_step_merit_window_last_merits_ =
      snapshot.position_step_merit_window_last_merits;
  position_step_merit_window_error_increases_ =
      snapshot.position_step_merit_window_error_increases;
  position_step_stationary_guard_active_ =
      snapshot.position_step_stationary_guard_active;
  position_step_stationary_guard_can_reopen_ =
      snapshot.position_step_stationary_guard_can_reopen;
  for (auto &kv : pose_task_groups_) {
    if (kv.second && kv.second->auto_switch()) {
      kv.second->set_layout(current_auto_task_layout_);
    }
  }
  stall_config_ = snapshot.stall_config;
  stall_state_ = snapshot.stall_state;
  stall_user_configured_ = snapshot.stall_user_configured;
  elastic_band_config_ = snapshot.elastic_band_config;
  elastic_band_state_ = snapshot.elastic_band_state;
  collision_constraint_ = snapshot.collision_constraint;
  per_pair_min_distance_overrides_ =
      snapshot.per_pair_min_distance_overrides;
  per_pair_deferred_overrides_ = snapshot.per_pair_deferred_overrides;
  collision_pair_distance_floor_ = snapshot.collision_pair_distance_floor;
  position_step_collision_command_floor_distances_ =
      snapshot.position_step_collision_command_floor_distances;
  collision_cache_frozen_indices_ = snapshot.collision_cache_frozen_indices;
  last_collision_debug_ = snapshot.last_collision_debug;
  last_collision_debug_list_ = snapshot.last_collision_debug_list;
  last_collision_constraint_pair_indices_ =
      snapshot.last_collision_constraint_pair_indices;
  collision_cached_candidate_pair_indices_ =
      snapshot.collision_cached_candidate_pair_indices;
  collision_pair_bound_valid_ = snapshot.collision_pair_bound_valid;
  collision_pair_last_signed_distance_ =
      snapshot.collision_pair_last_signed_distance;
  collision_pair_last_rel_translation_norm_ =
      snapshot.collision_pair_last_rel_translation_norm;
  collision_pair_last_rel_rotation_ =
      snapshot.collision_pair_last_rel_rotation;
  collision_pair_cache_has_full_scan_ =
      snapshot.collision_pair_cache_has_full_scan;
  collision_pair_cache_steps_since_refresh_ =
      snapshot.collision_pair_cache_steps_since_refresh;
  last_collision_pairs_considered_ = snapshot.last_collision_pairs_considered;
  last_collision_exact_distance_queries_ =
      snapshot.last_collision_exact_distance_queries;
  last_collision_bound_culled_pairs_ =
      snapshot.last_collision_bound_culled_pairs;
  last_collision_budget_exhausted_ = snapshot.last_collision_budget_exhausted;
  last_constraint_min_distance_ = snapshot.last_constraint_min_distance;
  last_constraint_min_recovery_margin_ =
      snapshot.last_constraint_min_recovery_margin;
  last_constraint_was_full_scan_ = snapshot.last_constraint_was_full_scan;
  last_collision_constraint_result_ =
      snapshot.last_collision_constraint_result;
  last_collision_constraint_row_dt_ =
      snapshot.last_collision_constraint_row_dt;
  last_collision_constraint_q_ = snapshot.last_collision_constraint_q;
  post_step_collision_distance_cache_ =
      snapshot.post_step_collision_distance_cache;
  last_post_step_collision_exact_distance_queries_ =
      snapshot.last_post_step_collision_exact_distance_queries;
  last_post_step_collision_motion_bound_culled_pairs_ =
      snapshot.last_post_step_collision_motion_bound_culled_pairs;
  last_collision_sphere_culled_pairs_ =
      snapshot.last_collision_sphere_culled_pairs;
  last_solution_dq_norm_ = snapshot.last_solution_dq_norm;
  collision_stuck_counters_ = snapshot.collision_stuck_counters;
  collision_stuck_last_distances_ = snapshot.collision_stuck_last_distances;
  pending_velocity_lock_indices_.clear();
  position_step_locked_indices_.clear();
  pending_step_torso_constraint_.reset();
  pending_reuse_current_kinematics_ = false;
  pending_step_validation_dt_.reset();
  pending_position_step_priority_constraints_.clear();
}

PositionIKResult KinematicsSolver::solve_position_step_with_preferred_lock(
    const Eigen::VectorXd &current_q, const Eigen::Matrix4d &target_pose,
    const std::string &frame_task_name, const PositionStepOptions &options) {
  PositionStepOptions continuity_options =
      make_preferred_lock_continuity_options(options);
  PositionStepOptions locked_options =
      make_preferred_lock_candidate_options(continuity_options);
  const PositionStepMutableStateSnapshot snapshot =
      capture_position_step_mutable_state();

  struct ScopedPreferredLockSuppression {
    KinematicsSolver *solver;
    bool saved = false;
    explicit ScopedPreferredLockSuppression(KinematicsSolver *solver_in)
        : solver(solver_in),
          saved(solver_in != nullptr
                    ? solver_in->suppress_preferred_lock_step_retry_
                    : false) {
      if (solver != nullptr) {
        solver->suppress_preferred_lock_step_retry_ = true;
      }
    }
    ~ScopedPreferredLockSuppression() {
      if (solver != nullptr) {
        solver->suppress_preferred_lock_step_retry_ = saved;
      }
    }
  } suppress(this);
  (void)suppress;

  auto now = []() { return std::chrono::high_resolution_clock::now(); };
  auto elapsed_ms = [](auto start) {
    return std::chrono::duration_cast<std::chrono::microseconds>(
               std::chrono::high_resolution_clock::now() - start)
               .count() /
           1000.0;
  };

  const auto candidate_start = now();
  PositionIKResult candidate = solve_position_step(
      current_q, target_pose, frame_task_name, locked_options);
  const double candidate_time_ms = elapsed_ms(candidate_start);

  auto compute_errors_at = [&](const Eigen::VectorXd &q) {
    PositionStepTargetErrorSummary out;
    if (q.size() != robot_->nq() || !q.allFinite()) {
      return out;
    }
    const Eigen::VectorXd saved_q = robot_->get_current_configuration();
    robot_->update_configuration(q);
    auto it = task_map_.find(frame_task_name);
    auto frame_task =
        (it != task_map_.end()) ? std::dynamic_pointer_cast<FrameTask>(it->second)
                                : nullptr;
    if (frame_task) {
      frame_task->setTargetPose(target_pose.block<3, 1>(0, 3),
                                target_pose.block<3, 3>(0, 0));
      frame_task->update(*robot_);
      const Eigen::VectorXd &error = frame_task->getError();
      if (error.size() == 3) {
        if (frame_task->getType() == TaskType::FRAME_ORIENTATION) {
          out.max_position_error = 0.0;
          out.max_orientation_error = error.head<3>().norm();
        } else {
          out.max_position_error = error.head<3>().norm();
          out.max_orientation_error = 0.0;
        }
      } else if (error.size() >= 6) {
        out.max_position_error = error.head<3>().norm();
        out.max_orientation_error = error.tail<3>().norm();
      }
    }
    robot_->update_configuration(saved_q);
    return out;
  };

  const PositionStepTargetErrorSummary entry_errors =
      compute_errors_at(current_q);
  const PositionStepTargetErrorSummary errors =
      compute_errors_at(candidate.q_solution);

  const double step_norm =
      (candidate.q_solution.size() == current_q.size())
          ? (candidate.q_solution - current_q).norm()
          : std::numeric_limits<double>::quiet_NaN();
  const bool accept = preferred_lock_candidate_acceptable(
      candidate, entry_errors, errors, step_norm, options);
  if (accept) {
    annotate_preferred_lock_result(candidate, candidate, errors, step_norm,
                                   candidate_time_ms, true);
    return candidate;
  }

  restore_position_step_mutable_state(snapshot);
  PositionIKResult fallback =
      solve_position_step(current_q, target_pose, frame_task_name,
                          continuity_options);
  annotate_preferred_lock_result(fallback, candidate, errors, step_norm,
                                 candidate_time_ms, false);
  return fallback;
}

PositionIKResult KinematicsSolver::solve_position_step_with_preferred_lock(
    const Eigen::VectorXd &current_q, const std::vector<TaskTarget> &targets,
    const PositionStepOptions &options) {
  PositionStepOptions continuity_options =
      make_preferred_lock_continuity_options(options);
  PositionStepOptions locked_options =
      make_preferred_lock_candidate_options(continuity_options);
  const PositionStepMutableStateSnapshot snapshot =
      capture_position_step_mutable_state();

  struct ScopedPreferredLockSuppression {
    KinematicsSolver *solver;
    bool saved = false;
    explicit ScopedPreferredLockSuppression(KinematicsSolver *solver_in)
        : solver(solver_in),
          saved(solver_in != nullptr
                    ? solver_in->suppress_preferred_lock_step_retry_
                    : false) {
      if (solver != nullptr) {
        solver->suppress_preferred_lock_step_retry_ = true;
      }
    }
    ~ScopedPreferredLockSuppression() {
      if (solver != nullptr) {
        solver->suppress_preferred_lock_step_retry_ = saved;
      }
    }
  } suppress(this);
  (void)suppress;

  auto now = []() { return std::chrono::high_resolution_clock::now(); };
  auto elapsed_ms = [](auto start) {
    return std::chrono::duration_cast<std::chrono::microseconds>(
               std::chrono::high_resolution_clock::now() - start)
               .count() /
           1000.0;
  };

  const auto candidate_start = now();
  PositionIKResult candidate =
      solve_position_step(current_q, targets, locked_options);
  const double candidate_time_ms = elapsed_ms(candidate_start);

  auto compute_errors_at = [&](const Eigen::VectorXd &q) {
    PositionStepTargetErrorSummary out;
    if (q.size() != robot_->nq() || !q.allFinite()) {
      return out;
    }
    const Eigen::VectorXd saved_q = robot_->get_current_configuration();
    robot_->update_configuration(q);
    double max_position = 0.0;
    double max_orientation = 0.0;
    bool saw_target = false;
    for (const auto &target : targets) {
      auto it = task_map_.find(target.task_name);
      if (it == task_map_.end() || !it->second) {
        continue;
      }
      const auto &task = it->second;
      if (auto frame_task = std::dynamic_pointer_cast<FrameTask>(task)) {
        frame_task->setTargetPose(target.target_pose.block<3, 1>(0, 3),
                                  target.target_pose.block<3, 3>(0, 0));
      } else if (auto absolute_task =
                     std::dynamic_pointer_cast<AbsoluteFrameTask>(task)) {
        if (target.has_secondary_target_pose) {
          absolute_task->set_target_from_arm_targets(
              target.target_pose, target.secondary_target_pose);
        } else {
          absolute_task->setTargetPose(target.target_pose.block<3, 1>(0, 3),
                                       target.target_pose.block<3, 3>(0, 0));
        }
      } else if (auto relative_task =
                     std::dynamic_pointer_cast<RelativeFrameTask>(task)) {
        relative_task->setTargetPose(target.target_pose.block<3, 1>(0, 3),
                                     target.target_pose.block<3, 3>(0, 0));
      } else {
        continue;
      }

      task->update(*robot_);
      const Eigen::VectorXd &error = task->getError();
      if (error.size() == 3) {
        bool is_orientation_only = false;
        if (auto frame_task = std::dynamic_pointer_cast<FrameTask>(task)) {
          is_orientation_only =
              frame_task->getType() == TaskType::FRAME_ORIENTATION;
        }
        if (is_orientation_only) {
          max_orientation = std::max(max_orientation, error.head<3>().norm());
        } else {
          max_position = std::max(max_position, error.head<3>().norm());
        }
        saw_target = true;
      } else if (error.size() >= 6) {
        max_position = std::max(max_position, error.head<3>().norm());
        max_orientation = std::max(max_orientation, error.tail<3>().norm());
        saw_target = true;
      }
    }
    if (saw_target) {
      out.max_position_error = max_position;
      out.max_orientation_error = max_orientation;
    }
    robot_->update_configuration(saved_q);
    return out;
  };

  const PositionStepTargetErrorSummary entry_errors =
      compute_errors_at(current_q);
  const PositionStepTargetErrorSummary errors =
      compute_errors_at(candidate.q_solution);

  const double step_norm =
      (candidate.q_solution.size() == current_q.size())
          ? (candidate.q_solution - current_q).norm()
          : std::numeric_limits<double>::quiet_NaN();
  const bool accept = preferred_lock_candidate_acceptable(
      candidate, entry_errors, errors, step_norm, options);
  if (accept) {
    annotate_preferred_lock_result(candidate, candidate, errors, step_norm,
                                   candidate_time_ms, true);
    return candidate;
  }

  restore_position_step_mutable_state(snapshot);
  PositionIKResult fallback =
      solve_position_step(current_q, targets, continuity_options);
  annotate_preferred_lock_result(fallback, candidate, errors, step_norm,
                                 candidate_time_ms, false);
  return fallback;
}



PositionIKResult KinematicsSolver::solve_position(
    const Eigen::VectorXd &seed_q, const Eigen::Matrix4d &target_pose,
    const std::string &frame_name, const PositionIKOptions &options) {

  PositionIKResult result;
  result.status = SolverStatus::kSuccess;
  std::vector<double> position_trace;
  std::vector<double> orientation_trace;
  position_trace.reserve(options.max_iterations);
  orientation_trace.reserve(options.max_iterations);
  double prev_combined_error = std::numeric_limits<double>::infinity();
  int stagnation_iters = 0;

  if (seed_q.size() != robot_->nq()) {
    result.status = SolverStatus::kInvalidInput;
    result.status_message =
        "seed_q size does not match robot nq in solve_position";
    return result;
  }
  if (options.nullspace_bias.has_value() &&
      options.nullspace_bias->size() != robot_->nq()) {
    result.status = SolverStatus::kInvalidInput;
    result.status_message =
        "nullspace_bias size does not match robot nq in solve_position";
    return result;
  }
  {
    std::unordered_set<int> seen;
    for (int idx : options.nullspace_active_joints) {
      if (idx < 0 || idx >= robot_->nv()) {
        result.status = SolverStatus::kInvalidInput;
        result.status_message =
            "nullspace_active_joints contains out-of-range index";
        return result;
      }
      if (!seen.insert(idx).second) {
        result.status = SolverStatus::kInvalidInput;
        result.status_message =
            "nullspace_active_joints contains duplicate index";
        return result;
      }
    }
  }
  if (options.nullspace_joint_weights.has_value()) {
    const int expected_size =
        options.nullspace_active_joints.empty()
            ? static_cast<int>(robot_->nv())
            : static_cast<int>(options.nullspace_active_joints.size());
    if (options.nullspace_joint_weights->size() != expected_size) {
      result.status = SolverStatus::kInvalidInput;
      result.status_message =
          "nullspace_joint_weights size mismatch for selected nullspace joints";
      return result;
    }
  }

  const auto &torso_opts = options.torso_constraint;
  const bool torso_enabled = torso_opts.enabled;
  const bool torso_has_pose_bounds = torso_opts.pose_lower_bounds.has_value() ||
                                     torso_opts.pose_upper_bounds.has_value();
  if (torso_enabled) {
    if (torso_opts.frame_name.empty()) {
      result.status = SolverStatus::kInvalidInput;
      result.status_message =
          "torso_constraint.frame_name must be set when torso constraint is enabled";
      return result;
    }
    if (!robot_->has_frame(torso_opts.frame_name)) {
      result.status = SolverStatus::kInvalidInput;
      result.status_message =
          "torso_constraint.frame_name not found in robot model";
      return result;
    }
    if (torso_has_pose_bounds !=
        (torso_opts.pose_lower_bounds.has_value() &&
         torso_opts.pose_upper_bounds.has_value())) {
      result.status = SolverStatus::kInvalidInput;
      result.status_message =
          "torso_constraint pose bounds require both lower and upper vectors";
      return result;
    }
    if (torso_has_pose_bounds) {
      if (torso_opts.pose_bounds_reference_pose.has_value()) {
        const Eigen::Matrix4d &M = torso_opts.pose_bounds_reference_pose.value();
        if (M.rows() != 4 || M.cols() != 4) {
          result.status = SolverStatus::kInvalidInput;
          result.status_message =
              "torso_constraint.pose_bounds_reference_pose must be 4x4";
          return result;
        }
        for (int r = 0; r < 4; ++r) {
          for (int c = 0; c < 4; ++c) {
            if (!std::isfinite(M(r, c))) {
              result.status = SolverStatus::kInvalidInput;
              result.status_message =
                  "torso_constraint.pose_bounds_reference_pose contains non-finite values";
              return result;
            }
          }
        }
      }
      if (torso_opts.pose_lower_bounds->size() != 6 ||
          torso_opts.pose_upper_bounds->size() != 6 ||
          torso_opts.pose_axis_mask.size() != 6 ||
          torso_opts.velocity_limits.size() != 6 ||
          torso_opts.acceleration_limits.size() != 6) {
        result.status = SolverStatus::kInvalidInput;
        result.status_message =
            "torso_constraint bounds, mask, velocity_limits, and acceleration_limits must all be size 6";
        return result;
      }
      for (int i = 0; i < 6; ++i) {
        if (torso_opts.pose_lower_bounds->coeff(i) >
            torso_opts.pose_upper_bounds->coeff(i)) {
          result.status = SolverStatus::kInvalidInput;
          result.status_message =
              "torso_constraint lower bound exceeds upper bound";
          return result;
        }
        if (torso_opts.pose_axis_mask.coeff(i) > 0.5 &&
            (torso_opts.velocity_limits.coeff(i) <= 0.0 ||
             torso_opts.acceleration_limits.coeff(i) <= 0.0)) {
          result.status = SolverStatus::kInvalidInput;
          result.status_message =
              "torso_constraint velocity/acceleration limits must be positive on constrained axes";
          return result;
        }
      }
      if (!std::isfinite(torso_opts.pose_bound_softening_fraction) ||
          torso_opts.pose_bound_softening_fraction < 0.0 ||
          torso_opts.pose_bound_softening_fraction > 1.0) {
        result.status = SolverStatus::kInvalidInput;
        result.status_message =
            "torso_constraint.pose_bound_softening_fraction must be in [0, 1]";
        return result;
      }
      if (!std::isfinite(torso_opts.velocity_box_headroom.fraction) ||
          torso_opts.velocity_box_headroom.fraction < 0.0 ||
          torso_opts.velocity_box_headroom.fraction > 1.0) {
        result.status = SolverStatus::kInvalidInput;
        result.status_message =
            "torso_constraint.velocity_box_headroom.fraction must be in [0, 1]";
        return result;
      }
      if (!std::isfinite(torso_opts.velocity_box_headroom.activation_margin) ||
          torso_opts.velocity_box_headroom.activation_margin < 0.0) {
        result.status = SolverStatus::kInvalidInput;
        result.status_message = "torso_constraint.velocity_box_headroom.activation_margin must be >= 0";
        return result;
      }
    }
  }

  auto frame_task = std::make_shared<FrameTask>(
      "position_ik_task", robot_, frame_name, TaskType::FRAME_POSE);
  if (!options.excluded_joint_indices.empty()) {
    frame_task->set_excluded_joint_indices(options.excluded_joint_indices);
  }

  std::shared_ptr<FrameTask> torso_task = nullptr;
  if (torso_enabled) {
    torso_task = std::make_shared<FrameTask>(
        "torso_orientation_task", robot_, torso_opts.frame_name,
        TaskType::FRAME_ORIENTATION);
    torso_task->setOrientationMask(torso_opts.orientation_mask);
    torso_task->setWeight(torso_opts.orientation_gain);
    torso_task->setPriority(1);
    torso_task->setSolveMode(TaskSolveMode::kMinError);
    torso_task->setAllowMinErrorFallback(false);
    if (!options.excluded_joint_indices.empty()) {
      torso_task->set_excluded_joint_indices(options.excluded_joint_indices);
    }
  }

  std::shared_ptr<PostureTask> posture_task = nullptr;
  if (options.nullspace_bias.has_value()) {
    if (!options.nullspace_active_joints.empty()) {
      posture_task = std::make_shared<PostureTask>(
          "nullspace_task", robot_, options.nullspace_active_joints);
    } else {
      posture_task = std::make_shared<PostureTask>("nullspace_task", robot_);
    }
    posture_task->setTargetConfiguration(options.nullspace_bias.value());
    posture_task->setWeight(options.nullspace_gain);
    if (options.nullspace_joint_weights.has_value()) {
      if (options.nullspace_active_joints.empty()) {
        posture_task->setJointWeights(options.nullspace_joint_weights.value());
      } else {
        posture_task->setControlledJointWeights(
            options.nullspace_joint_weights.value());
      }
    }
    posture_task->setPriority(torso_task ? 2 : 1);
    posture_task->setSolveMode(TaskSolveMode::kMinError);
    posture_task->setAllowMinErrorFallback(false);
    if (!options.excluded_joint_indices.empty()) {
      posture_task->set_excluded_joint_indices(options.excluded_joint_indices);
    }
  }

  const bool auto_stall = options.stall_recovery && !stall_config_.enabled;
  if (auto_stall) {
    const double nominal =
        get_collision_min_distance() > 0.0 ? get_collision_min_distance() : 0.0;
    enable_stall_handler(nominal);
  }

  if (options.elastic_band && !elastic_band_config_.enabled) {
    robot_->update_configuration(seed_q);
    enable_elastic_band(0.05);
    configure_elastic_band(/*delta_max=*/0.05, /*expand_rate=*/0.01,
                           /*decay_rate=*/0.2, /*stall_threshold=*/3,
                           /*expand_only_saturated=*/true);
  }

  Eigen::VectorXd q_current = seed_q;
  const Eigen::VectorXd q_reference = seed_q;
  const auto &velocity_to_config_index = velocity_to_config_index_cache();
  robot_->update_configuration(q_current);

  pinocchio::SE3 torso_reference_pose = pinocchio::SE3::Identity();
  if (torso_task) {
    torso_reference_pose = robot_->get_frame_pose(torso_opts.frame_name);
    const Eigen::Matrix3d torso_target_orientation =
        torso_opts.target_orientation.has_value()
            ? torso_opts.target_orientation.value()
            : torso_reference_pose.rotation();
    torso_task->setTargetOrientation(torso_target_orientation);
  }

  pinocchio::SE3 torso_pose_bounds_reference = pinocchio::SE3::Identity();
  if (torso_has_pose_bounds) {
    if (torso_opts.pose_bounds_reference_pose.has_value()) {
      const Eigen::Matrix4d &M = torso_opts.pose_bounds_reference_pose.value();
      torso_pose_bounds_reference =
          pinocchio::SE3(M.block<3, 3>(0, 0), M.block<3, 1>(0, 3));
    } else {
      torso_pose_bounds_reference =
          robot_->get_frame_pose(torso_opts.frame_name);
    }
  }

  Eigen::Vector3d target_position = target_pose.block<3, 1>(0, 3);
  Eigen::Matrix3d target_rotation = target_pose.block<3, 3>(0, 0);

  frame_task->setTargetPosition(target_position);
  frame_task->setTargetOrientation(target_rotation);
  frame_task->setWeight(10.0);
  frame_task->setPriority(0);
  frame_task->setSolveMode(options.primary_solve_mode);
  frame_task->setAllowMinErrorFallback(
      options.primary_allow_min_error_fallback);

  int iter = 0;
  bool converged = false;
  bool stagnation_abort = false;
  bool collision_violated_flag_mt = false;
  std::vector<Eigen::VectorXd> goals;
  std::vector<Eigen::MatrixXd> jacobians;
  std::vector<ObjectiveSolveConfig> objective_configs;
  goals.reserve(3);
  jacobians.reserve(3);
  objective_configs.reserve(3);
  Eigen::MatrixXd C;
  Eigen::VectorXd c_lower;
  Eigen::VectorXd c_upper;

  while (iter < options.max_iterations && !converged) {
    frame_task->update(*robot_);

    Eigen::VectorXd error = frame_task->getError();
    double pos_error = error.head(3).norm();
    double ori_error = error.tail(3).norm();
    position_trace.push_back(pos_error);
    orientation_trace.push_back(ori_error);

    if (position_ik_debug_) {
      std::cout << "[embodiK][IKDebug] iter " << iter
                << " pos_err=" << pos_error << " ori_err=" << ori_error
                << std::endl;
    }

    if (pos_error < options.position_tolerance &&
        ori_error < options.orientation_tolerance) {
      converged = true;
      break;
    }

    double combined_error = pos_error + ori_error;
    if (combined_error < prev_combined_error - options.stagnation_tolerance) {
      prev_combined_error = combined_error;
      stagnation_iters = 0;
    } else {
      stagnation_iters++;
      if (stagnation_iters >= options.stagnation_iterations) {
        if (position_ik_debug_) {
          std::cout << "[embodiK][IKDebug] Stagnation detected after "
                    << stagnation_iters << " iterations; aborting."
                    << std::endl;
        }
        stagnation_abort = true;
        break;
      }
    }

    goals.clear();
    jacobians.clear();
    objective_configs.clear();

    Eigen::VectorXd v_desired = frame_task->getVelocity();
    if (v_desired.size() >= 6) {
      v_desired.head(3) *= options.position_gain;
      v_desired.tail(3) *= options.orientation_gain;
    }

    if (options.max_linear_step > 0 || options.max_angular_step > 0) {
      double linear_vel = v_desired.head(3).norm();
      double angular_vel = v_desired.tail(3).norm();

      double scale = 1.0;
      if (linear_vel > options.max_linear_step / options.dt) {
        scale = std::min(scale,
                         (options.max_linear_step / options.dt) / linear_vel);
      }
      if (angular_vel > options.max_angular_step / options.dt) {
        scale = std::min(scale,
                         (options.max_angular_step / options.dt) / angular_vel);
      }
      v_desired *= scale;
    }

    goals.push_back(v_desired);
    jacobians.push_back(frame_task->getJacobian());
    // SCALE_ELASTIC is treated as SCALE in the SNS solver.
    const auto effective_primary_mode =
        (options.primary_solve_mode == TaskSolveMode::kScaleElastic)
            ? TaskSolveMode::kScale
            : options.primary_solve_mode;
    objective_configs.push_back(
        ObjectiveSolveConfig{0, effective_primary_mode,
                             options.primary_allow_min_error_fallback});

    if (torso_task) {
      torso_task->update(*robot_);
      goals.push_back(torso_task->getVelocity());
      jacobians.push_back(torso_task->getJacobian());
      objective_configs.push_back(
          ObjectiveSolveConfig{1, TaskSolveMode::kMinError, false});
    }

    if (posture_task) {
      posture_task->update(*robot_);
      goals.push_back(posture_task->getVelocity());
      jacobians.push_back(posture_task->getJacobian());
      objective_configs.push_back(
          ObjectiveSolveConfig{torso_task ? 2 : 1, TaskSolveMode::kMinError,
                               false});
    }

    Eigen::MatrixXd contact_P_c;
    const bool use_contact_projection = !contact_frames_.empty();
    if (use_contact_projection) {
      contact_P_c = compute_contact_projector();
      for (auto &J : jacobians) {
        J = J * contact_P_c;
      }
    }

    std::optional<CollisionConstraintResult> collision_constraint_result =
        std::nullopt;
    if (collision_constraint_.has_value() && collision_constraint_->enabled) {
      collision_constraint_result = compute_collision_constraint();
    }
    if (collision_constraint_result.has_value() &&
        !options.excluded_joint_indices.empty()) {
      zero_excluded_columns(collision_constraint_result->jacobian,
                            options.excluded_joint_indices);
    }
    std::optional<ComConstraintResult> com_constraint_result = std::nullopt;
    if (com_constraint_.has_value() && com_constraint_->enabled) {
      com_constraint_result = compute_com_constraint();
    }
    if (com_constraint_result.has_value() &&
        !options.excluded_joint_indices.empty()) {
      zero_excluded_columns(com_constraint_result->jacobian,
                            options.excluded_joint_indices);
    }
    if (use_contact_projection && collision_constraint_result.has_value()) {
      collision_constraint_result->jacobian =
          collision_constraint_result->jacobian * contact_P_c;
    }

    std::optional<LinearVelocityConstraintResult> linear_constraint_result =
        compute_linear_velocity_constraints();
    if (linear_constraint_result.has_value() &&
        !options.excluded_joint_indices.empty()) {
      for (int idx : options.excluded_joint_indices) {
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
    if (tight_pose_constraint_result.has_value() &&
        !options.excluded_joint_indices.empty()) {
      for (int idx : options.excluded_joint_indices) {
        if (idx >= 0 && idx < tight_pose_constraint_result->jacobian.cols()) {
          tight_pose_constraint_result->jacobian.col(idx).setZero();
        }
      }
    }
    if (use_contact_projection && tight_pose_constraint_result.has_value()) {
      tight_pose_constraint_result->jacobian =
          tight_pose_constraint_result->jacobian * contact_P_c;
    }

    std::optional<LinearVelocityConstraintResult> tight_point_constraint_result =
        compute_tight_point_constraints();
    if (tight_point_constraint_result.has_value() &&
        !options.excluded_joint_indices.empty()) {
      for (int idx : options.excluded_joint_indices) {
        if (idx >= 0 && idx < tight_point_constraint_result->jacobian.cols()) {
          tight_point_constraint_result->jacobian.col(idx).setZero();
        }
      }
    }
    if (use_contact_projection && tight_point_constraint_result.has_value()) {
      tight_point_constraint_result->jacobian =
          tight_point_constraint_result->jacobian * contact_P_c;
    }

    std::optional<ConstraintBlock> torso_constraint_result = std::nullopt;
    if (torso_task && torso_has_pose_bounds) {
      torso_constraint_result = build_torso_pose_bound_rows(
          *this, *robot_, torso_opts, torso_pose_bounds_reference, options.dt,
          options.excluded_joint_indices);
      if (use_contact_projection && torso_constraint_result.has_value()) {
        torso_constraint_result->jacobian =
            torso_constraint_result->jacobian * contact_P_c;
      }
    }

    int num_constraints = robot_->nv();
    if (collision_constraint_result.has_value()) {
      num_constraints +=
          static_cast<int>(collision_constraint_result->jacobian.rows());
    }
    if (com_constraint_result.has_value()) {
      num_constraints +=
          static_cast<int>(com_constraint_result->jacobian.rows());
    }
    if (torso_constraint_result.has_value()) {
      num_constraints +=
          static_cast<int>(torso_constraint_result->jacobian.rows());
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

    Eigen::MatrixXd C = Eigen::MatrixXd::Zero(num_constraints, robot_->nv());
    Eigen::VectorXd c_lower =
        Eigen::VectorXd::Constant(num_constraints, -1e10);
    Eigen::VectorXd c_upper =
        Eigen::VectorXd::Constant(num_constraints, 1e10);
    Eigen::VectorXd max_softening_factors =
        Eigen::VectorXd::Ones(num_constraints);

    C.block(0, 0, robot_->nv(), robot_->nv()) =
        Eigen::MatrixXd::Identity(robot_->nv(), robot_->nv());
    int constraint_idx = robot_->nv();

    if (use_position_limits_ || use_velocity_limits_) {
      auto vel_limits = robot_->get_velocity_limits();
      auto accel_limits = robot_->get_acceleration_limits();
      auto [q_min, q_max] = robot_->get_joint_limits();

      for (int i = 0; i < robot_->nv(); ++i) {
        double lower_margin = q_current[i] - q_min[i] - 0.02;
        double upper_margin = q_max[i] - q_current[i] - 0.02;

        auto [lower_limit, upper_limit] = calculate_velocity_box_constraint(
            lower_margin, upper_margin, vel_limits[i], accel_limits[i],
            options.dt);

        c_lower[i] = lower_limit;
        c_upper[i] = upper_limit;
      }
      if (options.limit_change_from_seed) {
        tighten_bounds_with_reference_corridor(
            c_lower, c_upper, q_reference, q_current, vel_limits, options.dt,
            velocity_to_config_index, robot_->nv());
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
    if (torso_constraint_result.has_value()) {
      constraint_idx = append_constraint_block(
          C, c_lower, c_upper, constraint_idx, robot_->nv(),
          torso_constraint_result->jacobian,
          torso_constraint_result->lower_bounds,
          torso_constraint_result->upper_bounds);
    }
    if (linear_constraint_result.has_value()) {
      const auto &lc = linear_constraint_result.value();
      const int rows = static_cast<int>(lc.jacobian.rows());
      C.block(constraint_idx, 0, rows, robot_->nv()) = lc.jacobian;
      c_lower.segment(constraint_idx, rows) = lc.lower_bounds;
      c_upper.segment(constraint_idx, rows) = lc.upper_bounds;
      constraint_idx += rows;
    }
    if (tight_pose_constraint_result.has_value()) {
      const auto &tpc = tight_pose_constraint_result.value();
      const int rows = static_cast<int>(tpc.jacobian.rows());
      C.block(constraint_idx, 0, rows, robot_->nv()) = tpc.jacobian;
      c_lower.segment(constraint_idx, rows) = tpc.lower_bounds;
      c_upper.segment(constraint_idx, rows) = tpc.upper_bounds;
      max_softening_factors.segment(constraint_idx, rows).setConstant(4.0);
      constraint_idx += rows;
    }
    if (tight_point_constraint_result.has_value()) {
      const auto &tpc = tight_point_constraint_result.value();
      const int rows = static_cast<int>(tpc.jacobian.rows());
      C.block(constraint_idx, 0, rows, robot_->nv()) = tpc.jacobian;
      c_lower.segment(constraint_idx, rows) = tpc.lower_bounds;
      c_upper.segment(constraint_idx, rows) = tpc.upper_bounds;
      max_softening_factors.segment(constraint_idx, rows).setConstant(4.0);
      constraint_idx += rows;
    }

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

    sanitize_solver_inputs(goals, jacobians, C, c_lower, c_upper);

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

    // Soft joint-space metric (contribution knob), same change of variables as
    // solve_velocity: solve in u-space on scaled copies (J D, C D), un-scale dq.
    const bool metric_active_ps =
        joint_metric_col_scale_.size() == robot_->nv();
    std::vector<Eigen::MatrixXd> metric_jacobians_ps;
    Eigen::MatrixXd metric_C_ps;
    const std::vector<Eigen::MatrixXd> *solve_jacobians_ps = &jacobians;
    const Eigen::MatrixXd *solve_C_ps = &C;
    if (metric_active_ps) {
      const auto D = joint_metric_col_scale_.asDiagonal();
      metric_jacobians_ps.reserve(jacobians.size());
      for (const auto &J : jacobians) metric_jacobians_ps.push_back(J * D);
      metric_C_ps = C * D;
      solve_jacobians_ps = &metric_jacobians_ps;
      solve_C_ps = &metric_C_ps;
    }
    auto vel_result = computeMultiObjectiveVelocitySolutionEigen(
        goals, *solve_jacobians_ps, *solve_C_ps, c_lower, c_upper, config,
        objective_configs);
    if (metric_active_ps &&
        static_cast<int>(vel_result.solution.size()) == robot_->nv()) {
      for (int j = 0; j < robot_->nv(); ++j)
        vel_result.solution[j] *= joint_metric_col_scale_(j);
    }

    if (stall_config_.enabled) {
      double primary_goal_norm =
          goals.empty() ? 0.0 : goals[0].norm();
      auto classified = classify_velocity_outcome(
          vel_result.status, vel_result.status_message,
          vel_result.task_scales, primary_goal_norm,
          vel_result.task_modes_effective,
          vel_result.task_used_fallback);

      VelocitySolverResult vsr;
      vsr.status = classified.status;
      vsr.status_message = classified.status_message;
      vsr.joint_velocities = Eigen::Map<const Eigen::VectorXd>(
          vel_result.solution.data(),
          static_cast<Eigen::Index>(vel_result.solution.size()));
      if (c_lower.size() >= robot_->nv()) {
        constexpr double kSaturationTol = 0.01;
        for (int i = 0; i < robot_->nv(); ++i) {
          double lower = c_lower[i];
          double upper = c_upper[i];
          if (static_cast<int>(c_lower.size()) >= 2 * robot_->nv()) {
            lower = std::max(lower, c_lower[robot_->nv() + i]);
            upper = std::min(upper, c_upper[robot_->nv() + i]);
          }
          const double joint_vel = vsr.joint_velocities[i];
          if (joint_vel <= lower + kSaturationTol ||
              joint_vel >= upper - kSaturationTol) {
            vsr.saturated_joints.push_back(i);
          }
        }
      }
      stall_handler_update(vsr);
    }

    if (vel_result.status != SolverStatus::kSuccess &&
        !(stall_config_.enabled &&
          vel_result.status == SolverStatus::kInfeasible)) {
      result.status = vel_result.status;
      result.status_message = vel_result.status_message;
      if (position_ik_debug_) {
        std::cout << "[embodiK][IKDebug] velocity solver failure at iter "
                  << iter << " status=" << static_cast<int>(vel_result.status)
                  << std::endl;
      }
      break;
    }

    Eigen::VectorXd dq = Eigen::Map<const Eigen::VectorXd>(
        vel_result.solution.data(), vel_result.solution.size());
    if (use_contact_projection) {
      dq = contact_P_c * dq;
    }
    if (!use_contact_projection) {
      clamp_joint_velocity_solution_in_place(dq, c_lower, c_upper,
                                             /*include_position_rows=*/false,
                                             robot_->nv());
    }
    if (use_position_limits_ && dq.size() == robot_->nv()) {
      auto [q_min, q_max] = robot_->get_joint_limits();
      if (elastic_band_config_.enabled &&
          elastic_band_state_.delta.size() == robot_->nv()) {
        expand_scalar_joint_limits_for_velocity_delta(
            q_min, q_max, elastic_band_state_.delta, velocity_to_config_index,
            robot_->nv());
      }
      clamp_joint_velocity_to_position_limits_for_integration(
          dq, q_current, q_min, q_max, velocity_to_config_index, options.dt,
          robot_->nv());
    }
    Eigen::VectorXd q_pre_step = q_current;
    q_current =
        pinocchio::integrate(robot_->model(), q_current, options.dt * dq);
    if (use_position_limits_) {
      auto [q_min_true, q_max_true] = robot_->get_joint_limits();
      project_scalar_configuration_to_true_joint_limits(
          q_current, q_min_true, q_max_true, velocity_to_config_index,
          robot_->nv());
    }
    robot_->update_configuration(q_current);

    // Skip expensive post-step checks when integration produced no motion.
    const bool step_moved =
        (q_current - q_pre_step).squaredNorm() > kCollisionEscapeNormEps;
    // Post-solve collision rejection (same logic as solve_position_step).
    if (step_moved && collision_constraint_.has_value() &&
        collision_constraint_->enabled) {
      // Keep the legacy global guard for penetration diagnostics while the
      // recovery margin below enforces every pair's effective floor.
      const double violation_threshold = collision_constraint_->min_distance;
      const auto pre_dist_debug =
          evaluate_post_step_collision_distance_mutating(q_pre_step);
      const auto pre_recovery_margins =
          evaluate_post_step_collision_recovery_margins(q_pre_step);
      const auto post_dist_debug =
          evaluate_post_step_collision_distance_mutating(q_current);
      const auto post_recovery_margins =
          evaluate_post_step_collision_recovery_margins(q_current);
      const double pre_dist =
          (pre_dist_debug.has_value() && std::isfinite(*pre_dist_debug))
              ? *pre_dist_debug
              : std::numeric_limits<double>::infinity();
      const double post_dist =
          (post_dist_debug.has_value() && std::isfinite(*post_dist_debug))
              ? *post_dist_debug
              : std::numeric_limits<double>::infinity();
      const bool seed_was_safe = pre_dist >= violation_threshold;
      const bool deepened =
          pre_dist < violation_threshold &&
          post_dist < pre_dist - kCollisionPenetrationWorsenTolerance;
      const bool hard_jump =
          std::isfinite(pre_dist) &&
          post_dist < kCollisionHardPenetrationRejectDistance &&
          post_dist < pre_dist - kCollisionHardWorsenTolerance;
      const bool recovery_floor_unacceptable =
          !collision_recovery_margins_acceptable(
              pre_recovery_margins, post_recovery_margins, 0.0);
      const bool recovery_floor_seed_was_safe =
          collision_recovery_margins_are_safe(pre_recovery_margins, 0.0);
      if (seed_was_safe || deepened || hard_jump ||
          recovery_floor_unacceptable) {
        q_current = q_pre_step;
        robot_->update_configuration(q_current);
        result.collision_rejection_count++;
        if (seed_was_safe || recovery_floor_seed_was_safe) {
          collision_violated_flag_mt = true;
          break;
        }
      }
    }

    iter++;
  }

  result.q_solution = q_current;
  result.achieved_pose = robot_->get_frame_pose(frame_name);

  frame_task->update(*robot_);
  Eigen::VectorXd final_error = frame_task->getError();
  result.position_error = final_error.head(3).norm();
  result.orientation_error = final_error.tail(3).norm();
  result.iterations_used = iter;
  result.position_error_trace = std::move(position_trace);
  result.orientation_error_trace = std::move(orientation_trace);

  const bool within_tolerance =
      (result.position_error <= options.position_tolerance) &&
      (result.orientation_error <= options.orientation_tolerance);
  const bool max_iterations_reached =
      (!converged) && (iter >= options.max_iterations);
  const auto classified_position = classify_position_outcome(
      result.status, result.status_message, converged || within_tolerance,
      stagnation_abort, options.classify_stagnation_as_no_progress,
      max_iterations_reached, result.position_error,
      result.orientation_error);
  result.status = classified_position.status;
  result.status_message = classified_position.status_message;
  if (collision_violated_flag_mt) {
    result.status = SolverStatus::kCollisionViolated;
    result.status_message =
        "solve_position: no step could maintain collision min_distance; "
        "q_solution is the last safe configuration";
  }

  if (position_ik_debug_) {
    std::cout << "[embodiK][IKDebug] solve_position finished with status="
              << static_cast<int>(result.status) << " iterations=" << iter
              << " final_pos_err=" << result.position_error
              << " final_ori_err=" << result.orientation_error << std::endl;
  }

  if (auto_stall) {
    disable_stall_handler();
  }

  return result;
}

PositionIKResult KinematicsSolver::solve_position_step(
    const Eigen::VectorXd &current_q, const Eigen::Matrix4d &target_pose,
    const std::string &frame_task_name,
    const PositionStepOptions &options) {

  PositionIKResult result;
  const double step_dt = (options.dt > 0.0) ? options.dt : dt_;
  const Eigen::VectorXd previous_applied_velocity = previous_dq_;

  if (current_q.size() != robot_->nq()) {
    result.status = SolverStatus::kInvalidInput;
    result.status_message =
        "current_q size does not match robot nq in solve_position_step";
    return result;
  }

  if (options.current_joint_velocity.size() != 0 &&
      options.current_joint_velocity.size() != robot_->nv()) {
    result.status = SolverStatus::kInvalidInput;
    result.status_message =
        "PositionStepOptions.current_joint_velocity size does not match robot nv";
    result.q_solution = current_q;
    result.joint_velocities = Eigen::VectorXd::Zero(robot_->nv());
    result.solution.assign(static_cast<std::size_t>(robot_->nv()), 0.0);
    return result;
  }
  if (options.current_joint_velocity.size() != 0 &&
      !options.current_joint_velocity.allFinite()) {
    result.status = SolverStatus::kNonFiniteInput;
    result.status_message =
        "PositionStepOptions.current_joint_velocity contains non-finite values";
    result.q_solution = current_q;
    result.joint_velocities = Eigen::VectorXd::Zero(robot_->nv());
    result.solution.assign(static_cast<std::size_t>(robot_->nv()), 0.0);
    return result;
  }
  if (velocity_zmp_constraint_.has_value() &&
      velocity_zmp_constraint_->enabled &&
      options.current_joint_velocity.size() == 0) {
    result.status = SolverStatus::kInvalidInput;
    result.status_message =
        "velocity-ZMP constraint requires "
        "PositionStepOptions.current_joint_velocity";
    result.q_solution = current_q;
    result.joint_velocities = Eigen::VectorXd::Zero(robot_->nv());
    result.solution.assign(static_cast<std::size_t>(robot_->nv()), 0.0);
    return result;
  }
  std::optional<Eigen::VectorXd> previous_position_step_current_dq =
      position_step_explicit_current_dq_;
  position_step_explicit_current_dq_ =
      options.current_joint_velocity.size() == robot_->nv()
          ? std::optional<Eigen::VectorXd>(options.current_joint_velocity)
          : std::nullopt;
  struct RestorePositionStepCurrentVelocity {
    std::optional<Eigen::VectorXd> *slot;
    std::optional<Eigen::VectorXd> previous;
    ~RestorePositionStepCurrentVelocity() {
      if (slot != nullptr) {
        *slot = std::move(previous);
      }
    }
  } restore_position_step_current_velocity{
      &position_step_explicit_current_dq_,
      std::move(previous_position_step_current_dq)};
  (void)restore_position_step_current_velocity;

  {
    std::string opt_err;
    if (!validate_position_step_joint_index_options(
            options, static_cast<int>(robot_->nv()), &opt_err)) {
      result.status = SolverStatus::kInvalidInput;
      result.status_message = std::move(opt_err);
      return result;
    }
  }

  const bool outermost_position_step_call = position_step_call_depth_ == 0;
  if (outermost_position_step_call) {
    last_post_step_collision_exact_distance_queries_ = 0;
    last_post_step_collision_motion_bound_culled_pairs_ = 0;
  }
  const bool owns_position_step_continuity =
      outermost_position_step_call ||
      (suppress_preferred_lock_step_retry_ &&
       position_step_call_depth_ == 1) ||
      (suppress_min_error_step_retry_ && position_step_call_depth_ <= 2);
  ScopedPositionStepCallDepth position_step_depth(position_step_call_depth_);
  std::optional<pinocchio::SE3> continuity_reference_pose;
  if (!options.continuity_reference_frame.empty()) {
    if (!robot_->has_frame(options.continuity_reference_frame)) {
      result.status = SolverStatus::kInvalidInput;
      result.status_message =
          "continuity_reference_frame not found in robot model";
      return result;
    }
    robot_->update_configuration(current_q);
    continuity_reference_pose =
        robot_->get_frame_pose(options.continuity_reference_frame);
  }
  if (outermost_position_step_call) {
    const bool had_previous_target_signature =
        last_position_step_target_signature_.has_value();
    PositionStepTargetSignature signature;
    signature.command_revision = options.continuity_command_revision;
    const std::unordered_set<std::string> explicit_target_task_names = {
        frame_task_name};
    signature.task_names.push_back(frame_task_name);
    signature.target_poses.push_back(target_pose);
    if (continuity_reference_pose.has_value()) {
      signature.reference_target_poses.push_back(
          canonicalize_position_step_signature_pose(
              frame_task_name, target_pose, continuity_reference_pose));
    }
    signature.gains = {options.position_gain, options.orientation_gain};
    signature.task_names.push_back("#continuity:" +
                                   options.continuity_reference_frame);
    signature.task_names.push_back("#torso:" +
                                   options.torso_constraint.frame_name);
    for (const auto &task : tasks_) {
      if (!task) {
        continue;
      }
      signature.task_names.push_back("#policy:" + task->getName());
      append_task_policy_signature(
          signature.gains, *task,
          explicit_target_task_names.find(task->getName()) ==
              explicit_target_task_names.end());
    }
    append_position_step_option_signature(signature.gains, options);
    update_position_step_target_motion_blocks(signature);
    position_step_target_geometry_moved_ =
        position_step_target_geometry_changed(signature);
    position_step_target_motion_observed_ =
        position_step_target_motion_observed_ ||
        (had_previous_target_signature &&
         position_step_target_geometry_moved_);
    if (update_position_step_target_signature(std::move(signature))) {
      capture_position_step_collision_command_floor(current_q);
    }
  }

  if (!options.preferred_locked_joint_indices.empty() &&
      !suppress_preferred_lock_step_retry_) {
    return solve_position_step_with_preferred_lock(
        current_q, target_pose, frame_task_name, options);
  }

  std::optional<TorsoPoseConstraintOptions> step_torso_constraint = std::nullopt;
  const auto &torso_opts = options.torso_constraint;
  const bool torso_enabled = torso_opts.enabled;
  const bool torso_has_pose_bounds = torso_opts.pose_lower_bounds.has_value() ||
                                     torso_opts.pose_upper_bounds.has_value();
  if (torso_enabled) {
    if (torso_opts.frame_name.empty()) {
      result.status = SolverStatus::kInvalidInput;
      result.status_message =
          "torso_constraint.frame_name must be set when torso constraint is enabled";
      return result;
    }
    if (!robot_->has_frame(torso_opts.frame_name)) {
      result.status = SolverStatus::kInvalidInput;
      result.status_message =
          "torso_constraint.frame_name not found in robot model";
      return result;
    }
    if (torso_has_pose_bounds !=
        (torso_opts.pose_lower_bounds.has_value() &&
         torso_opts.pose_upper_bounds.has_value())) {
      result.status = SolverStatus::kInvalidInput;
      result.status_message =
          "torso_constraint pose bounds require both lower and upper vectors";
      return result;
    }
    if (torso_has_pose_bounds) {
      if (torso_opts.pose_bounds_reference_pose.has_value()) {
        const Eigen::Matrix4d &M = torso_opts.pose_bounds_reference_pose.value();
        if (M.rows() != 4 || M.cols() != 4) {
          result.status = SolverStatus::kInvalidInput;
          result.status_message =
              "torso_constraint.pose_bounds_reference_pose must be 4x4";
          return result;
        }
        for (int r = 0; r < 4; ++r) {
          for (int c = 0; c < 4; ++c) {
            if (!std::isfinite(M(r, c))) {
              result.status = SolverStatus::kInvalidInput;
              result.status_message =
                  "torso_constraint.pose_bounds_reference_pose contains non-finite values";
              return result;
            }
          }
        }
      }
      if (torso_opts.pose_lower_bounds->size() != 6 ||
          torso_opts.pose_upper_bounds->size() != 6 ||
          torso_opts.pose_axis_mask.size() != 6 ||
          torso_opts.velocity_limits.size() != 6 ||
          torso_opts.acceleration_limits.size() != 6) {
        result.status = SolverStatus::kInvalidInput;
        result.status_message =
            "torso_constraint bounds, mask, velocity_limits, and acceleration_limits must all be size 6";
        return result;
      }
      for (int i = 0; i < 6; ++i) {
        if (torso_opts.pose_lower_bounds->coeff(i) >
            torso_opts.pose_upper_bounds->coeff(i)) {
          result.status = SolverStatus::kInvalidInput;
          result.status_message =
              "torso_constraint lower bound exceeds upper bound";
          return result;
        }
        if (torso_opts.pose_axis_mask.coeff(i) > 0.5 &&
            (torso_opts.velocity_limits.coeff(i) <= 0.0 ||
             torso_opts.acceleration_limits.coeff(i) <= 0.0)) {
          result.status = SolverStatus::kInvalidInput;
          result.status_message =
              "torso_constraint velocity/acceleration limits must be positive on constrained axes";
          return result;
        }
      }
      if (!std::isfinite(torso_opts.pose_bound_softening_fraction) ||
          torso_opts.pose_bound_softening_fraction < 0.0 ||
          torso_opts.pose_bound_softening_fraction > 1.0) {
        result.status = SolverStatus::kInvalidInput;
        result.status_message =
            "torso_constraint.pose_bound_softening_fraction must be in [0, 1]";
        return result;
      }
      if (!std::isfinite(torso_opts.velocity_box_headroom.fraction) ||
          torso_opts.velocity_box_headroom.fraction < 0.0 ||
          torso_opts.velocity_box_headroom.fraction > 1.0) {
        result.status = SolverStatus::kInvalidInput;
        result.status_message =
            "torso_constraint.velocity_box_headroom.fraction must be in [0, 1]";
        return result;
      }
      if (!std::isfinite(torso_opts.velocity_box_headroom.activation_margin) ||
          torso_opts.velocity_box_headroom.activation_margin < 0.0) {
        result.status = SolverStatus::kInvalidInput;
        result.status_message =
            "torso_constraint.velocity_box_headroom.activation_margin must be >= 0";
        return result;
      }

      TorsoPoseConstraintOptions normalized = torso_opts;
      if (!normalized.pose_bounds_reference_pose.has_value()) {
        const pinocchio::SE3 ref_pose = robot_->get_frame_pose(normalized.frame_name);
        Eigen::Matrix4d ref = Eigen::Matrix4d::Identity();
        ref.block<3, 3>(0, 0) = ref_pose.rotation();
        ref.block<3, 1>(0, 3) = ref_pose.translation();
        normalized.pose_bounds_reference_pose = ref;
      }
      step_torso_constraint = std::move(normalized);
    }
  }

  auto it = task_map_.find(frame_task_name);
  if (it == task_map_.end() || !it->second) {
    result.status = SolverStatus::kInvalidInput;
    result.status_message = "no task named '" + frame_task_name +
                            "' registered on this solver";
    return result;
  }
  auto frame_task = std::dynamic_pointer_cast<FrameTask>(it->second);
  if (!frame_task) {
    result.status = SolverStatus::kInvalidInput;
    result.status_message = "task '" + frame_task_name +
                            "' is not a FrameTask";
    return result;
  }

  frame_task->setTargetPose(target_pose.block<3, 1>(0, 3),
                            target_pose.block<3, 3>(0, 0));
  if (!suppress_min_error_step_retry_) {
    apply_position_step_primary_task_options(options, frame_task.get());
  }

  // Enable stall handler if requested. enable_stall_handler is idempotent:
  // repeated calls preserve accumulated stall counters so detection works
  // across successive single-step solve_position_step calls.
  if (options.stall_recovery) {
    const double nominal =
        get_collision_min_distance() > 0.0 ? get_collision_min_distance() : 0.0;
    enable_stall_handler(nominal);
  }

  if (options.elastic_band && !elastic_band_config_.enabled) {
    robot_->update_configuration(current_q);
    enable_elastic_band(0.05);
    configure_elastic_band(/*delta_max=*/0.05, /*expand_rate=*/0.01,
                           /*decay_rate=*/0.2, /*stall_threshold=*/3,
                           /*expand_only_saturated=*/true);
  }

  Eigen::VectorXd q = current_q;
  const Eigen::VectorXd q_reference = current_q;
  robot_->update_configuration(q);
  const auto &velocity_to_config_index = velocity_to_config_index_cache();
  const Eigen::VectorXd vel_limits = robot_->get_velocity_limits();

  const std::vector<int> step_locked_indices =
      build_step_locked_indices(options);
  // Keep the frozen set visible to collision computations that run after the
  // inner solve_velocity() clears pending_velocity_lock_indices_ (stall escape,
  // post-step validation), so the collision debug list and QP slots agree on
  // which pairs are controllable. Cleared on every exit path.
  position_step_locked_indices_ = step_locked_indices;
  struct ClearPositionStepLocks {
    KinematicsSolver *solver;
    ~ClearPositionStepLocks() {
      if (solver != nullptr) {
        solver->position_step_locked_indices_.clear();
      }
    }
  } clear_position_step_locks{this};
  (void)clear_position_step_locks;

  VelocitySolverResult last_vel_result;
  bool have_vel_result = false;
  Eigen::VectorXd first_tick_velocity;
  const int steps = std::max(1, options.max_steps);
  int steps_used = 0;
  Eigen::VectorXd vel(6);
  double prev_combined_error = std::numeric_limits<double>::infinity();
  double initial_combined_error = std::numeric_limits<double>::quiet_NaN();
  double initial_commanded_error = std::numeric_limits<double>::quiet_NaN();
  std::vector<double> initial_target_block_merits;
  int no_progress_count = 0;
  bool no_progress_exit = false;
  bool collision_violated_flag = false;
  bool recovery_inside_collision_margin = false;
  bool configuration_step_limited = false;
  bool outer_acceleration_limit_applied = false;

  for (int step = 0; step < steps; ++step) {
    frame_task->update(*robot_);
    const Eigen::VectorXd &error = frame_task->getError();
    double combined_error = 0.0;
    if (error.size() == 3) {
      combined_error = error.head<3>().norm();
      const bool is_orientation_only =
          frame_task->getType() == TaskType::FRAME_ORIENTATION;
      Eigen::VectorXd v3 = (is_orientation_only ? options.orientation_gain
                                                : options.position_gain) *
                           error.head<3>();
      if (is_orientation_only) {
        if (options.max_angular_speed > 0.0) {
          const double n = v3.norm();
          if (n > options.max_angular_speed) {
            v3 *= options.max_angular_speed / n;
          }
        }
      } else {
        if (options.max_linear_speed > 0.0) {
          const double n = v3.norm();
          if (n > options.max_linear_speed) {
            v3 *= options.max_linear_speed / n;
          }
        }
      }
      frame_task->setPositionStepTargetVelocity(v3);
    } else if (error.size() >= 6) {
      combined_error = error.head<3>().norm() + error.tail<3>().norm();
      vel.head<3>() = options.position_gain * error.head<3>();
      vel.tail<3>() = options.orientation_gain * error.tail<3>();
      clamp_spatial_velocity_components(vel, options.max_linear_speed,
                                        options.max_angular_speed);
      frame_task->setPositionStepTargetVelocity(vel);
    } else {
      result.status = SolverStatus::kInvalidInput;
      result.status_message =
          "frame task has invalid pose error dimension for solve_position_step";
      break;
    }
    if (step == 0) {
      initial_combined_error = combined_error;
      initial_commanded_error = commanded_frame_merit(
          error, frame_task->getType(), options.position_gain,
          options.orientation_gain);
      const auto block_merits = commanded_frame_block_merits(
          error, frame_task->getType(), options.position_gain,
          options.orientation_gain);
      initial_target_block_merits.assign(block_merits.begin(),
                                         block_merits.end());
    }
    if (step > 0 && options.no_progress_max_steps > 0 && have_vel_result) {
      const bool low_error_change =
          std::abs(prev_combined_error - combined_error) <=
          options.no_progress_error_tolerance;
      const bool low_velocity =
          last_vel_result.joint_velocities.size() == robot_->nv() &&
          last_vel_result.joint_velocities.norm() <=
              options.no_progress_dq_norm_tolerance;
      if (low_error_change && low_velocity) {
        ++no_progress_count;
        if (no_progress_count >= options.no_progress_max_steps) {
          no_progress_exit = true;
          break;
        }
      } else {
        no_progress_count = 0;
      }
    }
    prev_combined_error = combined_error;

    const double step_dt_eff = compute_adaptive_position_step_dt(
        options, step_dt, error.head<3>().norm(),
        collision_constraint_.has_value() && collision_constraint_->enabled,
        last_constraint_min_recovery_margin_);
    pending_velocity_lock_indices_ = step_locked_indices;
    pending_step_torso_constraint_ = step_torso_constraint;
    pending_step_validation_dt_ = step_dt_eff;
    pending_reuse_current_kinematics_ = true;
    const bool apply_setpoint_acceleration_limits =
        acceleration_limits_enabled_ && !position_step_target_geometry_moved_ &&
        step == 0;
    pending_position_step_acceleration_limits_ =
        apply_setpoint_acceleration_limits;
    VelocitySolverResult vel_out = solve_velocity(q, true);
    vel_out = retry_auto_task_layout_as_split_if_needed(
        q, std::move(vel_out), step_locked_indices, step_torso_constraint,
        step_dt_eff, {}, apply_setpoint_acceleration_limits);
    have_vel_result = true;
    last_vel_result = std::move(vel_out);
    ++steps_used;

    if (last_vel_result.status != SolverStatus::kSuccess &&
        last_vel_result.status != SolverStatus::kInfeasible &&
        last_vel_result.status != SolverStatus::kNumericalError) {
      break;
    }

    if (!options.integration_zero_velocity_indices.empty() &&
        last_vel_result.joint_velocities.size() == robot_->nv()) {
      apply_integration_velocity_mask(
          last_vel_result.joint_velocities,
          options.integration_zero_velocity_indices);
    }
    if (options.limit_change_from_seed &&
        last_vel_result.joint_velocities.size() == robot_->nv()) {
      Eigen::VectorXd corridor_lower = Eigen::VectorXd::Constant(
          robot_->nv(), -kUnboundedConstraintLimit);
      Eigen::VectorXd corridor_upper = Eigen::VectorXd::Constant(
          robot_->nv(), kUnboundedConstraintLimit);
      tighten_bounds_with_reference_corridor(
          corridor_lower, corridor_upper, q_reference, q, vel_limits, step_dt,
          velocity_to_config_index, robot_->nv());
      clamp_joint_velocity_solution_in_place(last_vel_result.joint_velocities,
                                             corridor_lower, corridor_upper,
                                             false, robot_->nv());
    }

    Eigen::VectorXd q_pre_step = q;
    bool step_collision_rejected = false;
    // Adaptive dt: scale integration step proportional to current position
    // error so large-jump approach is faster; reverts to base dt near target.
    // Proximity-aware cap: limit scale so the integration step cannot overshoot
    // past min_distance in one tick (scale * step_dt * max_ee_speed <= clearance).
    const bool adaptive_step_large = (step_dt_eff > step_dt * 1.01);
    if (use_position_limits_ &&
        last_vel_result.joint_velocities.size() == robot_->nv()) {
      auto [q_min, q_max] = robot_->get_joint_limits();
      if (elastic_band_config_.enabled &&
          elastic_band_state_.delta.size() == robot_->nv()) {
        expand_scalar_joint_limits_for_velocity_delta(
            q_min, q_max, elastic_band_state_.delta, velocity_to_config_index,
            robot_->nv());
      }
      clamp_joint_velocity_to_position_limits_for_integration(
          last_vel_result.joint_velocities, q, q_min, q_max,
          velocity_to_config_index, step_dt_eff, robot_->nv());
    }
    if (step == 0 &&
        last_vel_result.joint_velocities.size() == robot_->nv() &&
        last_vel_result.joint_velocities.allFinite()) {
      first_tick_velocity = last_vel_result.joint_velocities;
    }
    q = pinocchio::integrate(robot_->model(), q,
                             step_dt_eff * last_vel_result.joint_velocities);
    if (use_position_limits_) {
      auto [q_min_true, q_max_true] = robot_->get_joint_limits();
      project_scalar_configuration_to_true_joint_limits(
          q, q_min_true, q_max_true, velocity_to_config_index, robot_->nv());
    }
    const bool step_configuration_limited =
        clamp_configuration_delta_from_reference(
            robot_->model(), q_reference, q,
            options.max_configuration_step_norm);
    configuration_step_limited =
        step_configuration_limited || configuration_step_limited;
    robot_->update_configuration(q);

    // Skip expensive post-step checks when integration produced no motion.
    const bool step_moved =
        (q - q_pre_step).squaredNorm() > kCollisionEscapeNormEps;

    // Adaptive steps need all active pairs reconsidered at q_eval. The recovery
    // evaluator already performs that broadphase expansion and leaves a scalar
    // distance result behind, so reuse it instead of issuing a second scan.
    auto eval_broadphase_expanded = [&](const Eigen::VectorXd &q_eval)
        -> std::optional<double> {
      evaluate_post_step_collision_recovery_margins(q_eval);
      return evaluate_post_step_collision_distance_from_current_results();
    };

    // Post-solve collision rejection (same logic as multi-target overload).
    if (step_moved && collision_constraint_.has_value() &&
        collision_constraint_->enabled) {
      // Keep the legacy global guard for penetration diagnostics while the
      // recovery margin below enforces every pair's effective floor.
      const double violation_threshold = collision_constraint_->min_distance;
      const auto pre_dist_debug =
          evaluate_post_step_collision_distance_mutating(q_pre_step);
      const auto pre_recovery_margins =
          evaluate_post_step_collision_recovery_margins(q_pre_step);
      std::optional<double> post_dist_debug =
          adaptive_step_large
          ? eval_broadphase_expanded(q)
          : evaluate_post_step_collision_distance_mutating(q);
      const auto post_recovery_margins =
          evaluate_post_step_collision_recovery_margins(q);
      const double pre_dist =
          (pre_dist_debug.has_value() && std::isfinite(*pre_dist_debug))
              ? *pre_dist_debug
              : std::numeric_limits<double>::infinity();
      const double post_dist =
          (post_dist_debug.has_value() && std::isfinite(*post_dist_debug))
              ? *post_dist_debug
              : std::numeric_limits<double>::infinity();
      const bool recovery_floor_unacceptable =
          !collision_recovery_margins_acceptable(
              pre_recovery_margins, post_recovery_margins, 0.0);
      const bool recovery_floor_seed_was_safe =
          collision_recovery_margins_are_safe(pre_recovery_margins, 0.0);
      if (post_dist < violation_threshold ||
          recovery_floor_unacceptable) {
        bool seed_was_safe = (pre_dist >= violation_threshold);
        bool deepened =
            (pre_dist < violation_threshold &&
             post_dist <
                 pre_dist - kCollisionPenetrationWorsenTolerance);
        const bool hard_jump =
            std::isfinite(pre_dist) &&
            post_dist < kCollisionHardPenetrationRejectDistance &&
            post_dist < pre_dist - kCollisionHardWorsenTolerance;
        const bool pure_primary_min_error =
            !last_vel_result.task_modes_effective.empty() &&
            last_vel_result.task_modes_effective[0] == TaskSolveMode::kMinError &&
            (last_vel_result.task_used_fallback.empty() ||
             !last_vel_result.task_used_fallback[0]);
        const bool step_has_velocity_locks =
            !options.excluded_joint_indices.empty() ||
            !options.integration_zero_velocity_indices.empty();
        const bool enforce_material_penetration_guard =
            !pure_primary_min_error || step_has_velocity_locks;
        const bool post_penetrating =
            post_dist < kCollisionPenetrationDistanceThreshold;
        const bool crossed_into_penetration =
            enforce_material_penetration_guard && post_penetrating &&
            (!std::isfinite(pre_dist) ||
             pre_dist >= kCollisionPenetrationDistanceThreshold);
        const bool material_penetration_worsened =
            enforce_material_penetration_guard &&
            post_penetrating && std::isfinite(pre_dist) &&
            pre_dist < kCollisionPenetrationDistanceThreshold &&
            post_dist < pre_dist - kCollisionTolerance;
        if (seed_was_safe || deepened || hard_jump ||
            crossed_into_penetration || material_penetration_worsened ||
            recovery_floor_unacceptable) {
          // Backoff threshold depends on seed state:
          //   safe seed   → must restore full safety (>= min_distance)
          //   violated seed + hard geometry jump → stop geometry penetration
          //   violated seed + deepened → must not worsen beyond tolerance
          double backoff_threshold;
          if (seed_was_safe) {
            backoff_threshold = violation_threshold;
          } else if (crossed_into_penetration) {
            backoff_threshold = kCollisionPenetrationDistanceThreshold;
          } else if (material_penetration_worsened) {
            backoff_threshold = pre_dist - kCollisionTolerance;
          } else if (hard_jump) {
            backoff_threshold = kCollisionPenetrationDistanceThreshold;
          } else {
            backoff_threshold = pre_dist - kCollisionPenetrationWorsenTolerance;
          }
          bool accepted_backoff = false;
          const Eigen::VectorXd dq_nominal =
              step_dt_eff * last_vel_result.joint_velocities;
          for (double frac : kCollisionRejectionBackoffFractions) {
            Eigen::VectorXd q_backoff = pinocchio::integrate(
                robot_->model(), q_pre_step, frac * dq_nominal);
            robot_->update_configuration(q_backoff);
            // Use same evaluation level as initial check: broadphase-expanded
            // when adaptive_step_large so newly-entered pairs are checked at
            // each backoff position without a full GJK scan.
            auto backoff_dist_debug =
                adaptive_step_large
                ? eval_broadphase_expanded(q_backoff)
                : evaluate_post_step_collision_distance_mutating(q_backoff);
            const auto backoff_recovery_margins =
                evaluate_post_step_collision_recovery_margins(q_backoff);
            const bool backoff_recovery_acceptable =
                collision_recovery_margins_acceptable(
                    pre_recovery_margins, backoff_recovery_margins, 0.0);
            if (backoff_dist_debug.has_value() &&
                std::isfinite(*backoff_dist_debug) &&
                *backoff_dist_debug >= backoff_threshold &&
                backoff_recovery_acceptable) {
              q = q_backoff;
              accepted_backoff = true;
              break;
            }
          }
          if (!accepted_backoff) {
            q = q_pre_step;
            robot_->update_configuration(q);
            step_collision_rejected = true;
            result.collision_rejection_count++;
            if (seed_was_safe || recovery_floor_seed_was_safe) {
              // Safe seed: no step could maintain min_distance → violation.
              // Signal after loop; break so we don't attempt further steps.
              collision_violated_flag = true;
              break;
            }
            // Violated seed: hold at seed, let stall handler relax min_distance.
          }
        }
      }
    }

    // Cache invalidation on stall (same as multi-target overload).
    // Only invalidate when actually near collision — in clear space the cache
    // remains valid and avoids an expensive full scan on the next step.
    const double step_dq_norm = last_vel_result.joint_velocities.norm();
    const double effective_step_dq_norm =
        step_collision_rejected ? 0.0 : step_dq_norm;
    if (effective_step_dq_norm < stall_config_.dq_stall_eps) {
      const bool near_penetration =
          !std::isfinite(last_constraint_min_distance_) ||
          last_collision_budget_exhausted_ ||
          last_constraint_min_distance_ <
              kCollisionPenetrationDistanceThreshold +
                  kPostStepSafeMargin;
      if (near_penetration) {
        collision_pair_cache_has_full_scan_ = false;
      }
    }

    // Stall escape (same logic as multi-target overload).
    // Skip the expensive escape logic entirely when we know from the
    // constraint computation that all pairs are well clear.
    const bool stall_escape_eligible =
        collision_constraint_.has_value() && collision_constraint_->enabled &&
        (effective_step_dq_norm < stall_config_.dq_stall_eps ||
         step_collision_rejected);
    const bool stall_escape_near_collision =
        stall_escape_eligible &&
        (!std::isfinite(last_constraint_min_distance_) ||
         last_collision_budget_exhausted_ ||
         last_constraint_min_distance_ <
             collision_constraint_->min_distance +
                 kPostStepSafeMargin);
    if (stall_escape_near_collision) {
      double min_dist = std::numeric_limits<double>::infinity();
      int worst_row = -1;
      for (int i = 0; i < static_cast<int>(last_collision_debug_list_.size()); ++i) {
        if (last_collision_debug_list_[i].distance < min_dist) {
          min_dist = last_collision_debug_list_[i].distance;
          worst_row = i;
        }
      }

      // For stalled steps, also allow escape when we violate configured
      // clearance (distance < min_distance) even if still non-penetrating.
      const double escape_activation_threshold =
          (step_collision_rejected
               ? kCollisionEscapeActivationDistance
               : std::max(kCollisionEscapeActivationDistance,
                          collision_constraint_->min_distance));
      bool use_jacobian_escape =
          (min_dist < escape_activation_threshold && worst_row >= 0);
      bool use_normal_escape = false;
      Eigen::Vector3d escape_normal = Eigen::Vector3d::Zero();
      std::string escape_frame_a, escape_frame_b;
      double global_min_dist = min_dist;

      if (!use_jacobian_escape) {
        // Fast path: when the constraint computation already has a reliable
        // global minimum distance, use it directly instead of the expensive
        // full-scan evaluate_collision_debug().  Only fall back to the full
        // scan when no constraint data is available or budget was exhausted.
        const bool have_reliable_constraint_dist =
            std::isfinite(last_constraint_min_distance_) &&
            !last_collision_budget_exhausted_;
        if (have_reliable_constraint_dist) {
          if (last_constraint_min_distance_ < escape_activation_threshold &&
              last_collision_debug_.has_value() &&
              std::isfinite(last_collision_debug_->distance)) {
            global_min_dist = last_collision_debug_->distance;
            Eigen::Vector3d diff = last_collision_debug_->point_b_world -
                                   last_collision_debug_->point_a_world;
            double diff_norm = diff.norm();
            if (diff_norm > kCollisionEscapeNormEps) {
              escape_normal = diff / diff_norm;
            } else {
              escape_normal = Eigen::Vector3d(0, 0, 1);
            }
            escape_frame_a = last_collision_debug_->object_a;
            escape_frame_b = last_collision_debug_->object_b;
            use_normal_escape = true;
          }
        } else {
          auto global_debug = evaluate_collision_debug(q);
          if (global_debug.has_value() && std::isfinite(global_debug->distance) &&
              global_debug->distance < escape_activation_threshold) {
            global_min_dist = global_debug->distance;
            Eigen::Vector3d diff = global_debug->point_b_world - global_debug->point_a_world;
            double diff_norm = diff.norm();
            if (diff_norm > kCollisionEscapeNormEps) {
              escape_normal = diff / diff_norm;
            } else {
              escape_normal = Eigen::Vector3d(0, 0, 1);
            }
            escape_frame_a = global_debug->object_a;
            escape_frame_b = global_debug->object_b;
            use_normal_escape = true;
          }
        }
      }

      if (use_jacobian_escape) {
        auto coll_result = compute_collision_constraint();
        if (coll_result.has_value() &&
            worst_row < coll_result->jacobian.rows()) {
          Eigen::RowVectorXd n_row = coll_result->jacobian.row(worst_row);
          double nn = n_row.squaredNorm();
          if (nn > kCollisionEscapeNormEps) {
            double escape_mag =
                std::min(kCollisionEscapeStepMax, std::abs(min_dist));
            if (step_collision_rejected) {
              escape_mag = std::max(escape_mag, kCollisionEscapeStepMinOnRejection);
            }
            const Eigen::VectorXd dq_unit =
                (1.0 / std::sqrt(nn)) * n_row.transpose();
            auto try_signed_escape = [&](double sign) -> bool {
              Eigen::VectorXd q_candidate = pinocchio::integrate(
                  robot_->model(), q, sign * escape_mag * dq_unit);
              auto [q_min, q_max] = robot_->get_joint_limits();
              for (int j = 0; j < static_cast<int>(q_candidate.size()); ++j) {
                if (j < q_min.size()) {
                  q_candidate[j] = std::max(q_candidate[j], q_min[j]);
                  q_candidate[j] = std::min(q_candidate[j], q_max[j]);
                }
              }
              robot_->update_configuration(q_candidate);
              auto esc_dist =
                  evaluate_post_step_collision_distance_mutating(q_candidate);
              if (esc_dist.has_value() && std::isfinite(*esc_dist) &&
                  *esc_dist > min_dist) {
                q = q_candidate;
                result.stall_escape_count++;
                return true;
              }
              robot_->update_configuration(q);
              return false;
            };
            if (!try_signed_escape(+1.0)) {
              try_signed_escape(-1.0);
            }
          }
        }
      } else if (use_normal_escape) {
        const auto *collision_model =
            static_cast<const RobotModel &>(*robot_).collision_model();
        auto object_to_frame_name = [&](const std::string &obj_name) -> std::string {
          if (!collision_model) return std::string();
          for (std::size_t gi = 0; gi < collision_model->ngeoms; ++gi) {
            const auto &go = collision_model->geometryObjects[gi];
            if (go.name == obj_name) {
              return robot_->model().frames[go.parentFrame].name;
            }
          }
          return std::string();
        };
        const std::string escape_frame_a_name = object_to_frame_name(escape_frame_a);
        const std::string escape_frame_b_name = object_to_frame_name(escape_frame_b);

        auto try_frame_jacobian_escape = [&](const std::string &frame_name) -> bool {
          if (frame_name.empty() || !robot_->has_frame(frame_name)) return false;

          pinocchio::Data &data = robot_->data();
          pinocchio::computeJointJacobians(robot_->model(), data, q);
          auto frame_id = robot_->model().getFrameId(frame_name);
          Eigen::MatrixXd J(6, robot_->nv());
          J.setZero();
          pinocchio::getFrameJacobian(robot_->model(), data, frame_id,
                                      pinocchio::LOCAL_WORLD_ALIGNED, J);
          Eigen::MatrixXd J_lin = J.topRows<3>();
          Eigen::VectorXd dq_escape = J_lin.transpose() * escape_normal;
          double dq_n = dq_escape.norm();
          if (dq_n < kCollisionEscapeNormEps) return false;
          double escape_mag =
              std::min(kCollisionEscapeStepMax, std::abs(global_min_dist));
          if (step_collision_rejected) {
            escape_mag = std::max(escape_mag, kCollisionEscapeStepMinOnRejection);
          }
          dq_escape = (escape_mag / dq_n) * dq_escape;

          Eigen::VectorXd q_candidate = pinocchio::integrate(robot_->model(), q, dq_escape);
          auto [q_min, q_max] = robot_->get_joint_limits();
          for (int j = 0; j < static_cast<int>(q_candidate.size()); ++j) {
            if (j < q_min.size()) {
              q_candidate[j] = std::max(q_candidate[j], q_min[j]);
              q_candidate[j] = std::min(q_candidate[j], q_max[j]);
            }
          }
          robot_->update_configuration(q_candidate);
          auto esc_dist =
              evaluate_post_step_collision_distance_mutating(q_candidate);
          if (esc_dist.has_value() && std::isfinite(*esc_dist) &&
              *esc_dist > global_min_dist) {
            q = q_candidate;
            result.stall_escape_count++;
            return true;
          }
          robot_->update_configuration(q);
          return false;
        };

        if (!try_frame_jacobian_escape(escape_frame_b_name)) {
          try_frame_jacobian_escape(escape_frame_a_name);
        }
      }
    }

    // Limit-dominated lock breaker: if the QP reports near-zero joint motion
    // while several joints are saturated, nudge those joints slightly away
    // from hard limits to recover feasible directions.
    const bool collapsed_primary_scale =
        !last_vel_result.task_scales.empty() &&
        std::abs(last_vel_result.task_scales[0]) <= 1e-6;
    const bool plateau_status =
        last_vel_result.status == SolverStatus::kNumericalError ||
        last_vel_result.status == SolverStatus::kInfeasible ||
        collapsed_primary_scale;
    const int plateau_stall_steps =
        stall_handler_enabled() ? stall_state_.consecutive_stall_steps : 0;
    std::vector<int> desaturation_candidates = last_vel_result.saturated_joints;
    const bool saturated_zero_motion_plateau =
        combined_error > 1e-4 &&
        effective_step_dq_norm < stall_config_.dq_stall_eps &&
        !desaturation_candidates.empty() &&
        plateau_stall_steps >= kJointLimitDesaturationPlateauThreshold;
    const bool plateau_escape =
        (plateau_status || saturated_zero_motion_plateau) &&
        plateau_stall_steps >= kJointLimitDesaturationPlateauThreshold;
    auto [q_min, q_max] = robot_->get_joint_limits();
    if (plateau_escape) {
      for (int vi = 0;
           vi < static_cast<int>(velocity_to_config_index.size()); ++vi) {
        if (std::find(desaturation_candidates.begin(),
                      desaturation_candidates.end(),
                      vi) != desaturation_candidates.end()) {
          continue;
        }
        const int qi = velocity_to_config_index[vi];
        if (qi < 0 || qi >= static_cast<int>(q.size()) ||
            qi >= static_cast<int>(q_min.size()) ||
            qi >= static_cast<int>(q_max.size())) {
          continue;
        }
        const double margin_low = q[qi] - q_min[qi];
        const double margin_up = q_max[qi] - q[qi];
        if ((margin_low >= 0.0 &&
             margin_low < kJointLimitDesaturationExpandedMargin) ||
            (margin_up >= 0.0 &&
             margin_up < kJointLimitDesaturationExpandedMargin)) {
          desaturation_candidates.push_back(vi);
        }
      }
    }
    if (effective_step_dq_norm < stall_config_.dq_stall_eps &&
        (desaturation_candidates.size() >= 4 ||
         (plateau_escape && !desaturation_candidates.empty()))) {
      Eigen::VectorXd q_candidate = q;
      bool changed = false;
      const double desaturation_margin =
          plateau_escape ? kJointLimitDesaturationExpandedMargin
                         : kJointLimitDesaturationMargin;
      const double desaturation_step =
          plateau_escape ? kJointLimitDesaturationBoostStep
                         : kJointLimitDesaturationStep;
      for (int vi : desaturation_candidates) {
        if (vi < 0 || vi >= static_cast<int>(velocity_to_config_index.size())) {
          continue;
        }
        const int qi = velocity_to_config_index[vi];
        if (qi < 0 || qi >= static_cast<int>(q_candidate.size()) ||
            qi >= static_cast<int>(q_min.size()) ||
            qi >= static_cast<int>(q_max.size())) {
          continue;
        }
        const double margin_low = q_candidate[qi] - q_min[qi];
        const double margin_up = q_max[qi] - q_candidate[qi];
        if (margin_low >= 0.0 && margin_low < desaturation_margin) {
          q_candidate[qi] += desaturation_step;
          changed = true;
        } else if (margin_up >= 0.0 && margin_up < desaturation_margin) {
          q_candidate[qi] -= desaturation_step;
          changed = true;
        }
      }
      if (changed) {
        for (int j = 0; j < static_cast<int>(q_candidate.size()); ++j) {
          if (j < q_min.size()) {
            q_candidate[j] = std::max(q_candidate[j], q_min[j]);
            q_candidate[j] = std::min(q_candidate[j], q_max[j]);
          }
        }
        auto curr_dist = evaluate_post_step_collision_distance_mutating(q);
        robot_->update_configuration(q_candidate);
        auto cand_dist =
            evaluate_post_step_collision_distance_mutating(q_candidate);
        const double desat_safe_threshold =
            (collision_constraint_.has_value() && collision_constraint_->enabled)
            ? collision_constraint_->min_distance
            : kCollisionPenetrationDistanceThreshold;
        if (collision_recovery_candidate_acceptable(
                curr_dist, cand_dist, desat_safe_threshold)) {
          q = q_candidate;
          result.stall_escape_count++;
          recovery_inside_collision_margin =
              recovery_inside_collision_margin ||
              (curr_dist.has_value() && std::isfinite(*curr_dist) &&
               *curr_dist < desat_safe_threshold &&
               cand_dist.has_value() && std::isfinite(*cand_dist) &&
               *cand_dist < desat_safe_threshold);
        } else {
          robot_->update_configuration(q);
        }
      }
    }
    if (has_active_velocity_centroidal_hard_constraints()) {
      const Eigen::VectorXd accepted_velocity =
          pinocchio::difference(robot_->model(), q_pre_step, q) /
          std::max(step_dt_eff, 1e-9);
      std::string validation_message;
      if (!validate_centroidal_velocity_candidate(
              q_pre_step, active_explicit_current_dq(), accepted_velocity,
              &validation_message)) {
        q = q_pre_step;
        robot_->update_configuration(q);
        last_vel_result.status = SolverStatus::kInfeasible;
        last_vel_result.status_message = validation_message;
        last_vel_result.joint_velocities =
            Eigen::VectorXd::Zero(robot_->nv());
        last_vel_result.solution.assign(
            static_cast<std::size_t>(robot_->nv()), 0.0);
        break;
      }
      last_vel_result.joint_velocities = accepted_velocity;
      last_vel_result.solution.assign(
          accepted_velocity.data(),
          accepted_velocity.data() + accepted_velocity.size());
      if (position_step_explicit_current_dq_.has_value()) {
        position_step_explicit_current_dq_ = accepted_velocity;
      }
    }
  }

  if (auto retry_result = attempt_min_error_position_step_retry(
          current_q, options, step_dt, have_vel_result, last_vel_result, q,
          [&](std::vector<std::pair<Task *, TaskSolveMode>> *saved_modes) {
            return flip_scale_family_task(frame_task.get(), saved_modes);
          },
          [&]() {
            return solve_position_step(current_q, target_pose, frame_task_name,
                                       options);
          });
      retry_result.has_value()) {
    return retry_result.value();
  }

  if (position_step_target_geometry_moved_) {
    const Eigen::VectorXd terminal_q = q;
    robot_->update_configuration(current_q);
    frame_task->update(*robot_);
    const Eigen::VectorXd &current_error = frame_task->getError();
    const bool all_task_blocks_commanded = all_frame_task_blocks_commanded(
        current_error, frame_task->getType(), options.position_gain,
        options.orientation_gain);
    if (current_error.size() == 3) {
      const bool is_orientation_only =
          frame_task->getType() == TaskType::FRAME_ORIENTATION;
      Eigen::VectorXd current_velocity =
          (is_orientation_only ? options.orientation_gain
                               : options.position_gain) *
          current_error.head<3>();
      const double speed_limit = is_orientation_only
                                     ? options.max_angular_speed
                                     : options.max_linear_speed;
      if (speed_limit > 0.0 && current_velocity.norm() > speed_limit) {
        current_velocity *= speed_limit / current_velocity.norm();
      }
      frame_task->setPositionStepTargetVelocity(current_velocity);
    } else if (current_error.size() >= 6) {
      vel.head<3>() = options.position_gain * current_error.head<3>();
      vel.tail<3>() = options.orientation_gain * current_error.tail<3>();
      clamp_spatial_velocity_components(vel, options.max_linear_speed,
                                        options.max_angular_speed);
      frame_task->setPositionStepTargetVelocity(vel);
    }
    const double terminal_prediction_weight =
        position_step_task_terminal_prediction_weight(*frame_task,
                                                      current_error);
    const auto componentwise_candidate =
        !has_active_velocity_centroidal_hard_constraints() &&
                all_task_blocks_commanded && terminal_prediction_weight > 0.0
            ? estimate_position_step_componentwise_outer_candidate(
                  current_q, previous_applied_velocity, options, step_dt,
                  terminal_q)
            : std::nullopt;
    if (auto projected_result = apply_position_step_task_metric_projection(
            current_q, step_dt, first_tick_velocity, step_locked_indices,
            step_torso_constraint, {}, options.current_joint_velocity, q);
        projected_result.has_value()) {
      last_vel_result = std::move(*projected_result);
      have_vel_result = true;
      if (componentwise_candidate.has_value() && all_task_blocks_commanded &&
          terminal_prediction_weight > 0.0) {
        const Eigen::VectorXd projected_q = q;
        const auto block_merits_at = [&](const Eigen::VectorXd &candidate_q) {
          robot_->update_configuration(candidate_q);
          frame_task->update(*robot_);
          return commanded_frame_block_merits(
              frame_task->getError(), frame_task->getType(),
              options.position_gain, options.orientation_gain);
        };
        const auto projected_merits = block_merits_at(q);
        const auto candidate_is_preferred =
            [&](const Eigen::VectorXd &candidate_q) {
          const auto candidate_merits = block_merits_at(candidate_q);
          const double projected_total =
              projected_merits[0] + projected_merits[1];
          const double candidate_total =
              candidate_merits[0] + candidate_merits[1];
          return std::isfinite(projected_merits[0]) &&
                 std::isfinite(projected_merits[1]) &&
                 std::isfinite(candidate_merits[0]) &&
                 std::isfinite(candidate_merits[1]) &&
                 candidate_merits[0] <= projected_merits[0] + 1e-9 &&
                 candidate_merits[1] <= projected_merits[1] + 1e-9 &&
                 candidate_total + 1e-9 < projected_total;
        };
        Eigen::VectorXd safe_componentwise_candidate =
            *componentwise_candidate;
        if (terminal_prediction_weight < 1.0) {
          const Eigen::VectorXd projected_delta = pinocchio::difference(
              robot_->model(), current_q, projected_q);
          const Eigen::VectorXd componentwise_delta = pinocchio::difference(
              robot_->model(), current_q, safe_componentwise_candidate);
          safe_componentwise_candidate = pinocchio::integrate(
              robot_->model(), current_q,
              projected_delta + terminal_prediction_weight *
                                    (componentwise_delta - projected_delta));
        }
        if (candidate_is_preferred(safe_componentwise_candidate)) {
          apply_position_step_outer_acceleration_limit(
              current_q, previous_applied_velocity, options, step_dt,
              safe_componentwise_candidate);
          if (candidate_is_preferred(safe_componentwise_candidate)) {
            q = std::move(safe_componentwise_candidate);
            outer_acceleration_limit_applied = true;
          }
        }
        if (outer_acceleration_limit_applied) {
          last_vel_result.joint_velocities =
              pinocchio::difference(robot_->model(), current_q, q) / step_dt;
        }
        robot_->update_configuration(q);
      }
    }
  } else if (acceleration_limits_enabled_ &&
             first_tick_velocity.size() == robot_->nv() &&
             pinocchio::difference(robot_->model(), current_q, q)
                     .squaredNorm() > kCollisionEscapeNormEps) {
    q = pinocchio::integrate(robot_->model(), current_q,
                             step_dt * first_tick_velocity);
    if (use_position_limits_) {
      auto [q_min, q_max] = robot_->get_joint_limits();
      project_scalar_configuration_to_true_joint_limits(
          q, q_min, q_max, velocity_to_config_index, robot_->nv());
    }
    robot_->update_configuration(q);
  }

  for (auto &task : tasks_) {
    if (task) {
      task->clearPositionStepTargetVelocity();
    }
  }

  const bool final_configuration_limited =
      clamp_configuration_delta_from_reference(
          robot_->model(), q_reference, q,
          options.max_configuration_step_norm);
  configuration_step_limited =
      final_configuration_limited || configuration_step_limited;
  if (final_configuration_limited) {
    robot_->update_configuration(q);
  }
  if (!outer_acceleration_limit_applied || final_configuration_limited) {
    apply_position_step_outer_acceleration_limit(
        current_q, previous_applied_velocity, options, step_dt, q);
  }
  if (has_active_velocity_centroidal_hard_constraints()) {
    const Eigen::VectorXd final_velocity =
        pinocchio::difference(robot_->model(), current_q, q) /
        std::max(step_dt, 1e-9);
    const Eigen::VectorXd *final_current_dq =
        options.current_joint_velocity.size() == robot_->nv()
            ? &options.current_joint_velocity
            : nullptr;
    std::string validation_message;
    if (!validate_centroidal_velocity_candidate(
            current_q, final_current_dq, final_velocity, &validation_message)) {
      q = current_q;
      last_vel_result.status = SolverStatus::kInfeasible;
      last_vel_result.status_message = validation_message;
      last_vel_result.joint_velocities = Eigen::VectorXd::Zero(robot_->nv());
      last_vel_result.solution.assign(
          static_cast<std::size_t>(robot_->nv()), 0.0);
    } else {
      last_vel_result.joint_velocities = final_velocity;
      last_vel_result.solution.assign(
          final_velocity.data(), final_velocity.data() + final_velocity.size());
    }
    have_vel_result = true;
  }
  robot_->update_configuration(q);

  result.q_solution = q;
  result.achieved_pose = robot_->get_frame_pose(frame_task->getFrameName());
  result.iterations_used = steps_used;

  frame_task->update(*robot_);
  const Eigen::VectorXd &final_error = frame_task->getError();
  if (final_error.size() == 3) {
    result.position_error = final_error.head<3>().norm();
    result.orientation_error = 0.0;
  } else if (final_error.size() >= 6) {
    result.position_error = final_error.head<3>().norm();
    result.orientation_error = final_error.tail<3>().norm();
  }

  if (have_vel_result) {
    if (last_vel_result.joint_velocities.size() == robot_->nv()) {
      last_vel_result.solution.assign(last_vel_result.joint_velocities.data(),
                                      last_vel_result.joint_velocities.data() +
                                          last_vel_result.joint_velocities.size());
    }
    static_cast<VelocitySolverResult &>(result) = std::move(last_vel_result);
    if (!runtime_config_.enable_auto_task_layout) {
      result.active_task_layout =
          frame_task->getType() == TaskType::FRAME_POSE ? TaskLayout::kMerged
                                                        : TaskLayout::kSplit;
    }
  }
  if (no_progress_exit && result.status != SolverStatus::kInvalidInput) {
    result.status = SolverStatus::kNoProgress;
    result.status_message =
        "solve_position_step exited due to no progress near active bounds/"
        "constraints";
  }
  if (collision_violated_flag) {
    result.status = SolverStatus::kCollisionViolated;
    result.status_message =
        "solve_position_step: no step could maintain collision min_distance; "
        "q_solution is the last safe configuration";
  }
  if (collision_constraint_.has_value() && collision_constraint_->enabled &&
      (result.stall_escape_count > 0 ||
       result.collision_rejection_count > 0 ||
       recovery_inside_collision_margin)) {
    auto final_dist = evaluate_min_collision_distance(q);
    if (final_dist.has_value() && std::isfinite(*final_dist) &&
        *final_dist < collision_constraint_->min_distance - kCollisionTolerance) {
      recovery_inside_collision_margin = true;
    }
  }
  if (recovery_inside_collision_margin &&
      result.status == SolverStatus::kSuccess) {
    result.status = SolverStatus::kNoProgress;
    result.status_message =
        "solve_position_step applied recovery motion inside collision margin";
  }
  if (!recovery_inside_collision_margin &&
      should_report_position_recovery_success(
          result, current_q,
          stall_config_.dq_stall_eps * std::max(step_dt, 1e-9))) {
    result.status = SolverStatus::kSuccess;
    result.status_message =
        "solve_position_step applied constraint recovery motion";
  }
  const double final_commanded_error = commanded_frame_merit(
      final_error, frame_task->getType(), options.position_gain,
      options.orientation_gain);
  const auto final_block_merits = commanded_frame_block_merits(
      final_error, frame_task->getType(), options.position_gain,
      options.orientation_gain);
  const std::vector<double> final_target_block_merits(
      final_block_merits.begin(), final_block_merits.end());
  const double candidate_step_norm =
      result.q_solution.size() == current_q.size()
          ? pinocchio::difference(robot_->model(), current_q,
                                  result.q_solution)
                .norm()
          : std::numeric_limits<double>::quiet_NaN();
  const bool held_non_improving_step =
      should_hold_position_step_for_continuity(
          result, current_q, initial_commanded_error, final_commanded_error,
          {initial_commanded_error}, {final_commanded_error},
          initial_target_block_merits, final_target_block_merits,
          frame_task->getPriority(), candidate_step_norm,
          owns_position_step_continuity,
          collision_violated_flag, step_torso_constraint.has_value());
  if (held_non_improving_step) {
    const bool target_satisfied =
        initial_commanded_error <= kPositionStepSatisfiedMeritTolerance;
    Eigen::VectorXd braking_q = current_q;
    const bool braking_to_hold = compute_position_step_continuity_brake(
        current_q, previous_applied_velocity, options, step_dt, braking_q);
    result.position_step_hold_active = !braking_to_hold;
    q = braking_to_hold ? std::move(braking_q) : current_q;
    robot_->update_configuration(q);
    result.q_solution = q;
    result.achieved_pose = robot_->get_frame_pose(frame_task->getFrameName());
    frame_task->update(*robot_);
    const Eigen::VectorXd &held_error = frame_task->getError();
    if (held_error.size() == 3) {
      result.position_error = held_error.head<3>().norm();
      result.orientation_error = 0.0;
    } else if (held_error.size() >= 6) {
      result.position_error = held_error.head<3>().norm();
      result.orientation_error = held_error.tail<3>().norm();
    }
    result.status = braking_to_hold
                        ? SolverStatus::kSuccess
                        : (target_satisfied ? SolverStatus::kSuccess
                                            : SolverStatus::kNoProgress);
    result.status_message =
        braking_to_hold
            ? "solve_position_step decelerating before continuity hold"
            : (target_satisfied
                   ? "solve_position_step held satisfied stationary target at "
                     "the current configuration"
                   : "solve_position_step held current configuration because "
                     "the nominal step did not reduce commanded task error");
    sync_position_result_applied_velocity(result, robot_->model(), current_q,
                                          step_dt);
    last_solution_dq_norm_ = result.joint_velocities.norm();
    if (!braking_to_hold && previous_dq_.size() == robot_->nv()) {
      previous_dq_.setZero();
    }
  }
  if (result.stall_escape_count > 0 || result.collision_rejection_count > 0 ||
      collision_violated_flag || configuration_step_limited) {
    sync_position_result_applied_velocity(result, robot_->model(), current_q,
                                          step_dt);
  }
  if (!held_non_improving_step &&
      should_hold_soft_infeasible_position_step(
          result, current_q, initial_combined_error,
          result.position_error + result.orientation_error,
          result.position_error, result.orientation_error,
          collision_violated_flag,
          recovery_inside_collision_margin)) {
    Eigen::VectorXd braking_q = current_q;
    const bool braking_to_hold = compute_position_step_continuity_brake(
        current_q, previous_applied_velocity, options, step_dt, braking_q);
    result.position_step_hold_active = !braking_to_hold;
    q = braking_to_hold ? std::move(braking_q) : current_q;
    robot_->update_configuration(q);
    result.q_solution = q;
    result.achieved_pose = robot_->get_frame_pose(frame_task->getFrameName());
    frame_task->update(*robot_);
    const Eigen::VectorXd &held_error = frame_task->getError();
    if (held_error.size() == 3) {
      result.position_error = held_error.head<3>().norm();
      result.orientation_error = 0.0;
    } else if (held_error.size() >= 6) {
      result.position_error = held_error.head<3>().norm();
      result.orientation_error = held_error.tail<3>().norm();
    }
    result.status = braking_to_hold ? SolverStatus::kSuccess
                                    : SolverStatus::kNoProgress;
    result.status_message = braking_to_hold
                                ? "solve_position_step decelerating before "
                                  "soft-infeasible continuity hold"
                                : "solve_position_step held current configuration "
                                  "because the primary SCALE task was "
                                  "soft-infeasible";
    sync_position_result_applied_velocity(result, robot_->model(), current_q,
                                          step_dt);
    last_solution_dq_norm_ = result.joint_velocities.norm();
    if (!braking_to_hold && previous_dq_.size() == robot_->nv()) {
      previous_dq_.setZero();
    }
  }

  sync_position_result_applied_velocity(result, robot_->model(), current_q,
                                        step_dt);
  if (!enforce_final_position_step_centroidal_candidate(
          current_q, options.current_joint_velocity, step_dt, result)) {
    q = current_q;
    result.achieved_pose = robot_->get_frame_pose(frame_task->getFrameName());
    frame_task->update(*robot_);
    const Eigen::VectorXd &held_error = frame_task->getError();
    if (held_error.size() == 3) {
      result.position_error = held_error.head<3>().norm();
      result.orientation_error = 0.0;
    } else if (held_error.size() >= 6) {
      result.position_error = held_error.head<3>().norm();
      result.orientation_error = held_error.tail<3>().norm();
    }
    last_solution_dq_norm_ = 0.0;
  }
  if (acceleration_limits_enabled_ &&
      result.joint_velocities.size() == robot_->nv()) {
    previous_dq_ = result.joint_velocities;
  }

  // In teleop-style loops (max_steps=1), stall handling must accumulate across
  // successive solve_position_step calls based on *applied* motion, not only
  // raw QP status.  This post-step update prevents success/no-progress
  // oscillations from resetting the consecutive stall counter.
  if (stall_handler_enabled()) {
    if (held_non_improving_step) {
      stall_state_.consecutive_stall_steps = 0;
    } else {
      const double applied_step_norm = (result.q_solution - current_q).norm();
      const double applied_step_eps =
          stall_config_.dq_stall_eps * std::max(step_dt, 1e-9);
      if (applied_step_norm < applied_step_eps) {
        stall_state_.consecutive_stall_steps++;
        stall_state_.total_stall_steps++;
      } else {
        stall_state_.consecutive_stall_steps = 0;
      }
    }
  }

  return result;
}

PositionIKResult KinematicsSolver::solve_position_step(
    const Eigen::VectorXd &current_q, const std::vector<TaskTarget> &targets,
    const PositionStepOptions &options) {

  PositionIKResult result;
  const double step_dt = (options.dt > 0.0) ? options.dt : dt_;
  const Eigen::VectorXd previous_applied_velocity = previous_dq_;

  if (current_q.size() != robot_->nq()) {
    result.status = SolverStatus::kInvalidInput;
    result.status_message =
        "current_q size does not match robot nq in solve_position_step";
    return result;
  }
  if (options.current_joint_velocity.size() != 0 &&
      options.current_joint_velocity.size() != robot_->nv()) {
    result.status = SolverStatus::kInvalidInput;
    result.status_message =
        "PositionStepOptions.current_joint_velocity size does not match robot nv";
    result.q_solution = current_q;
    result.joint_velocities = Eigen::VectorXd::Zero(robot_->nv());
    result.solution.assign(static_cast<std::size_t>(robot_->nv()), 0.0);
    return result;
  }
  if (options.current_joint_velocity.size() != 0 &&
      !options.current_joint_velocity.allFinite()) {
    result.status = SolverStatus::kNonFiniteInput;
    result.status_message =
        "PositionStepOptions.current_joint_velocity contains non-finite values";
    result.q_solution = current_q;
    result.joint_velocities = Eigen::VectorXd::Zero(robot_->nv());
    result.solution.assign(static_cast<std::size_t>(robot_->nv()), 0.0);
    return result;
  }
  if (velocity_zmp_constraint_.has_value() &&
      velocity_zmp_constraint_->enabled &&
      options.current_joint_velocity.size() == 0) {
    result.status = SolverStatus::kInvalidInput;
    result.status_message =
        "velocity-ZMP constraint requires "
        "PositionStepOptions.current_joint_velocity";
    result.q_solution = current_q;
    result.joint_velocities = Eigen::VectorXd::Zero(robot_->nv());
    result.solution.assign(static_cast<std::size_t>(robot_->nv()), 0.0);
    return result;
  }
  std::optional<Eigen::VectorXd> previous_position_step_current_dq =
      position_step_explicit_current_dq_;
  position_step_explicit_current_dq_ =
      options.current_joint_velocity.size() == robot_->nv()
          ? std::optional<Eigen::VectorXd>(options.current_joint_velocity)
          : std::nullopt;
  struct RestorePositionStepCurrentVelocity {
    std::optional<Eigen::VectorXd> *slot;
    std::optional<Eigen::VectorXd> previous;
    ~RestorePositionStepCurrentVelocity() {
      if (slot != nullptr) {
        *slot = std::move(previous);
      }
    }
  } restore_position_step_current_velocity{
      &position_step_explicit_current_dq_,
      std::move(previous_position_step_current_dq)};
  (void)restore_position_step_current_velocity;
  if (targets.empty()) {
    result.status = SolverStatus::kInvalidInput;
    result.status_message = "targets must be non-empty in solve_position_step";
    return result;
  }

  {
    std::string opt_err;
    if (!validate_position_step_joint_index_options(
            options, static_cast<int>(robot_->nv()), &opt_err)) {
      result.status = SolverStatus::kInvalidInput;
      result.status_message = std::move(opt_err);
      return result;
    }
  }

  const bool outermost_position_step_call = position_step_call_depth_ == 0;
  if (outermost_position_step_call) {
    last_post_step_collision_exact_distance_queries_ = 0;
    last_post_step_collision_motion_bound_culled_pairs_ = 0;
  }
  const bool owns_position_step_continuity =
      outermost_position_step_call ||
      (suppress_preferred_lock_step_retry_ &&
       position_step_call_depth_ == 1) ||
      (suppress_min_error_step_retry_ && position_step_call_depth_ <= 2);
  ScopedPositionStepCallDepth position_step_depth(position_step_call_depth_);
  std::optional<pinocchio::SE3> continuity_reference_pose;
  if (!options.continuity_reference_frame.empty()) {
    if (!robot_->has_frame(options.continuity_reference_frame)) {
      result.status = SolverStatus::kInvalidInput;
      result.status_message =
          "continuity_reference_frame not found in robot model";
      return result;
    }
    robot_->update_configuration(current_q);
    continuity_reference_pose =
        robot_->get_frame_pose(options.continuity_reference_frame);
  }
  if (outermost_position_step_call) {
    const bool had_previous_target_signature =
        last_position_step_target_signature_.has_value();
    PositionStepTargetSignature signature;
    signature.command_revision = options.continuity_command_revision;
    std::unordered_set<std::string> explicit_target_task_names;
    for (const auto &target : targets) {
      explicit_target_task_names.insert(target.task_name);
      signature.task_names.push_back(target.task_name);
      signature.target_poses.push_back(target.target_pose);
      if (continuity_reference_pose.has_value()) {
        signature.reference_target_poses.push_back(
            canonicalize_position_step_signature_pose(
                target.task_name, target.target_pose,
                continuity_reference_pose));
      }
      signature.gains.push_back(target.position_gain);
      signature.gains.push_back(target.orientation_gain);
      signature.gains.push_back(target.priority_position_tolerance);
      signature.gains.push_back(target.priority_orientation_tolerance);
      if (target.has_secondary_target_pose) {
        signature.task_names.push_back(target.task_name + "#secondary");
        signature.target_poses.push_back(target.secondary_target_pose);
        if (continuity_reference_pose.has_value()) {
          signature.reference_target_poses.push_back(
              canonicalize_position_step_signature_pose(
                  target.task_name, target.secondary_target_pose,
                  continuity_reference_pose));
        }
      }
    }
    signature.task_names.push_back("#continuity:" +
                                   options.continuity_reference_frame);
    signature.task_names.push_back("#torso:" +
                                   options.torso_constraint.frame_name);
    for (const auto &task : tasks_) {
      if (!task) {
        continue;
      }
      signature.task_names.push_back("#policy:" + task->getName());
      append_task_policy_signature(
          signature.gains, *task,
          explicit_target_task_names.find(task->getName()) ==
              explicit_target_task_names.end());
    }
    append_position_step_option_signature(signature.gains, options);
    update_position_step_target_motion_blocks(signature);
    position_step_target_geometry_moved_ =
        position_step_target_geometry_changed(signature);
    position_step_target_motion_observed_ =
        position_step_target_motion_observed_ ||
        (had_previous_target_signature &&
         position_step_target_geometry_moved_);
    if (update_position_step_target_signature(std::move(signature))) {
      capture_position_step_collision_command_floor(current_q);
    }
  }

  if (!options.preferred_locked_joint_indices.empty() &&
      !suppress_preferred_lock_step_retry_) {
    return solve_position_step_with_preferred_lock(current_q, targets, options);
  }

  std::optional<TorsoPoseConstraintOptions> step_torso_constraint = std::nullopt;
  const auto &torso_opts = options.torso_constraint;
  const bool torso_enabled = torso_opts.enabled;
  const bool torso_has_pose_bounds = torso_opts.pose_lower_bounds.has_value() ||
                                     torso_opts.pose_upper_bounds.has_value();
  if (torso_enabled) {
    if (torso_opts.frame_name.empty()) {
      result.status = SolverStatus::kInvalidInput;
      result.status_message =
          "torso_constraint.frame_name must be set when torso constraint is enabled";
      return result;
    }
    if (!robot_->has_frame(torso_opts.frame_name)) {
      result.status = SolverStatus::kInvalidInput;
      result.status_message =
          "torso_constraint.frame_name not found in robot model";
      return result;
    }
    if (torso_has_pose_bounds !=
        (torso_opts.pose_lower_bounds.has_value() &&
         torso_opts.pose_upper_bounds.has_value())) {
      result.status = SolverStatus::kInvalidInput;
      result.status_message =
          "torso_constraint pose bounds require both lower and upper vectors";
      return result;
    }
    if (torso_has_pose_bounds) {
      if (torso_opts.pose_bounds_reference_pose.has_value()) {
        const Eigen::Matrix4d &M = torso_opts.pose_bounds_reference_pose.value();
        if (M.rows() != 4 || M.cols() != 4) {
          result.status = SolverStatus::kInvalidInput;
          result.status_message =
              "torso_constraint.pose_bounds_reference_pose must be 4x4";
          return result;
        }
        for (int r = 0; r < 4; ++r) {
          for (int c = 0; c < 4; ++c) {
            if (!std::isfinite(M(r, c))) {
              result.status = SolverStatus::kInvalidInput;
              result.status_message =
                  "torso_constraint.pose_bounds_reference_pose contains non-finite values";
              return result;
            }
          }
        }
      }
      if (torso_opts.pose_lower_bounds->size() != 6 ||
          torso_opts.pose_upper_bounds->size() != 6 ||
          torso_opts.pose_axis_mask.size() != 6 ||
          torso_opts.velocity_limits.size() != 6 ||
          torso_opts.acceleration_limits.size() != 6) {
        result.status = SolverStatus::kInvalidInput;
        result.status_message =
            "torso_constraint bounds, mask, velocity_limits, and acceleration_limits must all be size 6";
        return result;
      }
      for (int i = 0; i < 6; ++i) {
        if (torso_opts.pose_lower_bounds->coeff(i) >
            torso_opts.pose_upper_bounds->coeff(i)) {
          result.status = SolverStatus::kInvalidInput;
          result.status_message =
              "torso_constraint lower bound exceeds upper bound";
          return result;
        }
        if (torso_opts.pose_axis_mask.coeff(i) > 0.5 &&
            (torso_opts.velocity_limits.coeff(i) <= 0.0 ||
             torso_opts.acceleration_limits.coeff(i) <= 0.0)) {
          result.status = SolverStatus::kInvalidInput;
          result.status_message =
              "torso_constraint velocity/acceleration limits must be positive on constrained axes";
          return result;
        }
      }
      if (!std::isfinite(torso_opts.pose_bound_softening_fraction) ||
          torso_opts.pose_bound_softening_fraction < 0.0 ||
          torso_opts.pose_bound_softening_fraction > 1.0) {
        result.status = SolverStatus::kInvalidInput;
        result.status_message =
            "torso_constraint.pose_bound_softening_fraction must be in [0, 1]";
        return result;
      }
      if (!std::isfinite(torso_opts.velocity_box_headroom.fraction) ||
          torso_opts.velocity_box_headroom.fraction < 0.0 ||
          torso_opts.velocity_box_headroom.fraction > 1.0) {
        result.status = SolverStatus::kInvalidInput;
        result.status_message =
            "torso_constraint.velocity_box_headroom.fraction must be in [0, 1]";
        return result;
      }
      if (!std::isfinite(torso_opts.velocity_box_headroom.activation_margin) ||
          torso_opts.velocity_box_headroom.activation_margin < 0.0) {
        result.status = SolverStatus::kInvalidInput;
        result.status_message =
            "torso_constraint.velocity_box_headroom.activation_margin must be >= 0";
        return result;
      }

      TorsoPoseConstraintOptions normalized = torso_opts;
      if (!normalized.pose_bounds_reference_pose.has_value()) {
        const pinocchio::SE3 ref_pose = robot_->get_frame_pose(normalized.frame_name);
        Eigen::Matrix4d ref = Eigen::Matrix4d::Identity();
        ref.block<3, 3>(0, 0) = ref_pose.rotation();
        ref.block<3, 1>(0, 3) = ref_pose.translation();
        normalized.pose_bounds_reference_pose = ref;
      }
      step_torso_constraint = std::move(normalized);
    }
  }

  enum class PoseTaskKind { kFrame, kAbsolute, kRelative };

  struct ResolvedTask {
    std::shared_ptr<Task> task;
    PoseTaskKind kind;
  };

  const size_t n_targets = targets.size();
  std::vector<ResolvedTask> resolved;
  resolved.reserve(n_targets);
  for (const auto &target : targets) {
    auto it = task_map_.find(target.task_name);
    if (it == task_map_.end() || !it->second) {
      result.status = SolverStatus::kInvalidInput;
      result.status_message = "no task named '" + target.task_name +
                              "' registered on this solver";
      return result;
    }
    const auto &task = it->second;
    if (std::dynamic_pointer_cast<FrameTask>(task)) {
      resolved.push_back({task, PoseTaskKind::kFrame});
    } else if (std::dynamic_pointer_cast<AbsoluteFrameTask>(task)) {
      resolved.push_back({task, PoseTaskKind::kAbsolute});
    } else if (std::dynamic_pointer_cast<RelativeFrameTask>(task)) {
      resolved.push_back({task, PoseTaskKind::kRelative});
    } else {
      result.status = SolverStatus::kInvalidInput;
      result.status_message = "task '" + target.task_name +
                              "' is not a supported pose task type";
      return result;
    }
  }

  for (const auto &rt : resolved) {
    if (!suppress_min_error_step_retry_) {
      apply_position_step_primary_task_options(options, rt.task.get());
    }
  }

  int highest_active_target_priority = std::numeric_limits<int>::max();
  for (const auto &rt : resolved) {
    if (rt.task->isActive()) {
      highest_active_target_priority =
          std::min(highest_active_target_priority, rt.task->getPriority());
    }
  }

  // Enable stall handler if requested. enable_stall_handler is idempotent:
  // repeated calls preserve accumulated stall counters so detection works
  // across successive single-step solve_position_step calls.
  if (options.stall_recovery) {
    const double nominal =
        get_collision_min_distance() > 0.0 ? get_collision_min_distance() : 0.0;
    enable_stall_handler(nominal);
  }

  if (options.elastic_band && !elastic_band_config_.enabled) {
    enable_elastic_band(0.05);
    configure_elastic_band(/*delta_max=*/0.05, /*expand_rate=*/0.01,
                           /*decay_rate=*/0.2, /*stall_threshold=*/3,
                           /*expand_only_saturated=*/true);
  }

  Eigen::VectorXd q = current_q;
  const Eigen::VectorXd q_reference = current_q;
  robot_->update_configuration(q);
  const auto &velocity_to_config_index = velocity_to_config_index_cache();
  const Eigen::VectorXd vel_limits = robot_->get_velocity_limits();

  std::vector<std::shared_ptr<Task>> pose_only;
  pose_only.reserve(resolved.size());
  for (const auto &rt : resolved) {
    pose_only.push_back(rt.task);
  }
  const std::vector<int> step_locked_indices =
      build_step_locked_indices(options);
  // Keep the frozen set visible to collision computations that run after the
  // inner solve_velocity() clears pending_velocity_lock_indices_ (stall escape,
  // post-step validation), so the collision debug list and QP slots agree on
  // which pairs are controllable. Cleared on every exit path.
  position_step_locked_indices_ = step_locked_indices;
  struct ClearPositionStepLocks {
    KinematicsSolver *solver;
    ~ClearPositionStepLocks() {
      if (solver != nullptr) {
        solver->position_step_locked_indices_.clear();
      }
    }
  } clear_position_step_locks{this};
  (void)clear_position_step_locks;

  VelocitySolverResult last_vel_result;
  bool have_vel_result = false;
  Eigen::VectorXd first_tick_velocity;
  const int steps = std::max(1, options.max_steps);
  int steps_used = 0;
  Eigen::Matrix<double, 6, 1> vel;
  double prev_combined_error = std::numeric_limits<double>::infinity();
  double initial_primary_combined_error =
      std::numeric_limits<double>::quiet_NaN();
  std::vector<double> initial_target_merits;
  std::vector<double> initial_target_block_merits;
  int no_progress_count = 0;
  bool no_progress_exit = false;
  bool collision_violated_flag_mts = false;  // multi-target solve_position_step
  bool recovery_inside_collision_margin = false;
  bool configuration_step_limited = false;
  bool outer_acceleration_limit_applied = false;
  bool priority_candidate_rejected = false;

  int max_active_target_priority = std::numeric_limits<int>::min();
  for (const auto &rt : resolved) {
    if (rt.task->isActive()) {
      max_active_target_priority =
          std::max(max_active_target_priority, rt.task->getPriority());
    }
  }
  std::vector<int> protected_target_priorities;
  for (const auto &rt : resolved) {
    if (rt.task->isActive() &&
        rt.task->getPriority() < max_active_target_priority) {
      protected_target_priorities.push_back(rt.task->getPriority());
    }
  }
  std::sort(protected_target_priorities.begin(),
            protected_target_priorities.end());
  protected_target_priorities.erase(
      std::unique(protected_target_priorities.begin(),
                  protected_target_priorities.end()),
      protected_target_priorities.end());

  const auto build_priority_constraint_specs = [&](double step_dt_eff) {
    std::vector<PositionStepPriorityConstraintSpec> specs;
    if (!acceleration_limits_enabled_ ||
        acceleration_limits_.size() != robot_->nv()) {
      return specs;
    }
    specs.reserve(resolved.size());
    for (std::size_t index = 0; index < resolved.size(); ++index) {
      const auto &resolved_target = resolved[index];
      if (!resolved_target.task->isActive() ||
          !std::binary_search(protected_target_priorities.begin(),
                              protected_target_priorities.end(),
                              resolved_target.task->getPriority())) {
        continue;
      }
      const auto &target = targets[index];
      const bool protect_position =
          std::isfinite(target.priority_position_tolerance) &&
          target.priority_position_tolerance > 0.0;
      const bool protect_orientation =
          std::isfinite(target.priority_orientation_tolerance) &&
          target.priority_orientation_tolerance > 0.0;
      if (!protect_position && !protect_orientation) {
        continue;
      }
      specs.push_back({resolved_target.task,
                       protect_position ? target.priority_position_tolerance
                                        : -1.0,
                       protect_orientation
                           ? target.priority_orientation_tolerance
                           : -1.0,
                       step_dt_eff});
    }
    return specs;
  };

  const auto target_block_merits_at = [&](const Eigen::VectorXd &q_eval) {
    robot_->update_configuration(q_eval);
    std::vector<std::array<double, 2>> merits;
    merits.reserve(resolved.size());
    for (std::size_t index = 0; index < resolved.size(); ++index) {
      const auto &resolved_target = resolved[index];
      resolved_target.task->update(*robot_);
      TaskType merit_task_type = TaskType::FRAME_POSE;
      if (resolved_target.kind == PoseTaskKind::kFrame) {
        merit_task_type =
            static_cast<const FrameTask *>(resolved_target.task.get())
                ->getType();
      }
      merits.push_back(commanded_frame_block_merits(
          resolved_target.task->getError(), merit_task_type,
          targets[index].position_gain, targets[index].orientation_gain));
    }
    return merits;
  };

  const auto priority_candidate_acceptable =
      [&](const std::vector<std::array<double, 2>> &baseline_merits,
          const std::vector<std::array<double, 2>> &candidate_merits) {
        if (baseline_merits.size() != resolved.size() ||
            candidate_merits.size() != resolved.size()) {
          return false;
        }
        for (int priority : protected_target_priorities) {
          for (int block = 0; block < 2; ++block) {
            double acceptable_total = 0.0;
            double candidate_total = 0.0;
            bool saw_task = false;
            for (std::size_t index = 0; index < resolved.size(); ++index) {
              if (!resolved[index].task->isActive() ||
                  resolved[index].task->getPriority() != priority) {
                continue;
              }
              saw_task = true;
              const double baseline = baseline_merits[index][block];
              const double candidate = candidate_merits[index][block];
              const double configured_tolerance =
                  block == 0 ? targets[index].priority_position_tolerance
                             : targets[index].priority_orientation_tolerance;
              const bool has_configured_tolerance =
                  std::isfinite(configured_tolerance) &&
                  configured_tolerance > 0.0;
              const double protected_tolerance =
                  has_configured_tolerance
                      ? configured_tolerance
                      : kPositionStepSatisfiedMeritTolerance;
              if (!std::isfinite(baseline) || !std::isfinite(candidate)) {
                return false;
              }
              acceptable_total += std::max(baseline, protected_tolerance);
              candidate_total += candidate;
            }
            if (saw_task && candidate_total > acceptable_total + 1e-9) {
              return false;
            }
          }
        }
        return true;
      };

  const auto position_step_priority_scale =
      [&](const Eigen::VectorXd &q_before,
          const Eigen::VectorXd &q_after) {
        if (protected_target_priorities.empty()) {
          return 1.0;
        }
        const Eigen::VectorXd delta =
            pinocchio::difference(robot_->model(), q_before, q_after);
        if (!delta.allFinite() ||
            delta.squaredNorm() <= kCollisionEscapeNormEps) {
          robot_->update_configuration(q_after);
          return 1.0;
        }
        const auto baseline_merits = target_block_merits_at(q_before);
        const auto candidate_merits = target_block_merits_at(q_after);
        if (priority_candidate_acceptable(baseline_merits,
                                          candidate_merits)) {
          robot_->update_configuration(q_after);
          return 1.0;
        }

        double accepted_fraction = 0.0;
        double rejected_fraction = 1.0;
        for (int iteration = 0;
             iteration < kPositionStepPriorityBacktrackIterations;
             ++iteration) {
          const double fraction =
              0.5 * (accepted_fraction + rejected_fraction);
          Eigen::VectorXd candidate_q = pinocchio::integrate(
              robot_->model(), q_before, fraction * delta);
          const auto backoff_merits = target_block_merits_at(candidate_q);
          if (priority_candidate_acceptable(baseline_merits,
                                            backoff_merits)) {
            accepted_fraction = fraction;
          } else {
            rejected_fraction = fraction;
          }
        }
        robot_->update_configuration(q_after);
        return accepted_fraction;
      };

  const bool direct_priority_backtrack_is_safe =
      !acceleration_limits_enabled_ && !step_torso_constraint.has_value() &&
      !(collision_constraint_.has_value() &&
        collision_constraint_->enabled) &&
      !(com_constraint_.has_value() && com_constraint_->enabled) &&
      !has_active_velocity_centroidal_hard_constraints() &&
      !(relative_pose_constraint_.has_value() &&
        relative_pose_constraint_->enabled) &&
      get_linear_velocity_constraint_rows() == 0 &&
      tight_frame_pose_constraints_.empty() &&
      tight_point_constraints_.empty() && !has_contact_frames();

  for (int step = 0; step < steps; ++step) {
    double combined_error = 0.0;
    double commanded_error = 0.0;
    std::vector<double> current_target_merits;
    current_target_merits.reserve(n_targets);
    std::vector<double> current_target_block_merits;
    current_target_block_merits.reserve(2 * n_targets);
    double max_position_error = 0.0;
    double primary_combined_error = std::numeric_limits<double>::quiet_NaN();
    std::vector<Eigen::VectorXd> target_velocities(n_targets);
    bool task_apply_failed = false;
    std::string task_apply_error;
    for (size_t i = 0; i < n_targets; ++i) {
      const auto &target = targets[i];
      const auto &rt = resolved[i];

      switch (rt.kind) {
      case PoseTaskKind::kFrame:
        static_cast<FrameTask *>(rt.task.get())
            ->setTargetPose(target.target_pose.block<3, 1>(0, 3),
                            target.target_pose.block<3, 3>(0, 0));
        break;
      case PoseTaskKind::kAbsolute:
        if (target.has_secondary_target_pose) {
          static_cast<AbsoluteFrameTask *>(rt.task.get())
              ->set_target_from_arm_targets(target.target_pose,
                                            target.secondary_target_pose);
        } else {
          static_cast<AbsoluteFrameTask *>(rt.task.get())
              ->setTargetPose(target.target_pose.block<3, 1>(0, 3),
                              target.target_pose.block<3, 3>(0, 0));
        }
        break;
      case PoseTaskKind::kRelative:
        static_cast<RelativeFrameTask *>(rt.task.get())
            ->setTargetPose(target.target_pose.block<3, 1>(0, 3),
                            target.target_pose.block<3, 3>(0, 0));
        break;
      }

      rt.task->update(*robot_);
      const Eigen::VectorXd &error = rt.task->getError();
      TaskType merit_task_type = TaskType::FRAME_POSE;
      if (rt.kind == PoseTaskKind::kFrame) {
        merit_task_type =
            static_cast<const FrameTask *>(rt.task.get())->getType();
      }
      const auto block_merits = commanded_frame_block_merits(
          error, merit_task_type, target.position_gain,
          target.orientation_gain);
      const double target_merit = block_merits[0] + block_merits[1];
      commanded_error += target_merit;
      current_target_merits.push_back(target_merit);
      current_target_block_merits.push_back(block_merits[0]);
      current_target_block_merits.push_back(block_merits[1]);
      if (error.size() == 3) {
        bool is_orientation_only = false;
        if (rt.kind == PoseTaskKind::kFrame) {
          const auto *ft = static_cast<const FrameTask *>(rt.task.get());
          is_orientation_only = ft->getType() == TaskType::FRAME_ORIENTATION;
        }
        Eigen::VectorXd v3 = (is_orientation_only ? target.orientation_gain
                                                  : target.position_gain) *
                             error.head<3>();
        if (is_orientation_only) {
          if (options.max_angular_speed > 0.0) {
            const double n = v3.norm();
            if (n > options.max_angular_speed) {
              v3 *= options.max_angular_speed / n;
            }
          }
        } else {
          if (options.max_linear_speed > 0.0) {
            const double n = v3.norm();
            if (n > options.max_linear_speed) {
              v3 *= options.max_linear_speed / n;
            }
          }
        }
        const double task_position_error = error.head<3>().norm();
        if (rt.task->isActive() &&
            rt.task->getPriority() == highest_active_target_priority) {
          primary_combined_error =
              std::isfinite(primary_combined_error)
                  ? std::max(primary_combined_error, task_position_error)
                  : task_position_error;
        }
        combined_error += task_position_error;
        if (!is_orientation_only) {
          max_position_error =
              std::max(max_position_error, task_position_error);
        }
        target_velocities[i] = v3;
      } else if (error.size() >= 6) {
        const double task_position_error = error.head<3>().norm();
        const double task_combined_error =
            task_position_error + error.tail<3>().norm();
        if (rt.task->isActive() &&
            rt.task->getPriority() == highest_active_target_priority) {
          primary_combined_error =
              std::isfinite(primary_combined_error)
                  ? std::max(primary_combined_error, task_combined_error)
                  : task_combined_error;
        }
        combined_error += task_combined_error;
        max_position_error = std::max(max_position_error, task_position_error);
        vel.head<3>() = target.position_gain * error.head<3>();
        vel.tail<3>() = target.orientation_gain * error.tail<3>();
        clamp_spatial_velocity_components(vel, options.max_linear_speed,
                                          options.max_angular_speed);
        target_velocities[i] = vel;
      } else {
        task_apply_failed = true;
        task_apply_error =
            "task '" + target.task_name + "' has invalid pose error dimension";
        break;
      }
    }

    if (task_apply_failed) {
      result.status = SolverStatus::kInvalidInput;
      result.status_message = task_apply_error;
      break;
    }
    if (step == 0) {
      initial_primary_combined_error = primary_combined_error;
      initial_target_merits = current_target_merits;
      initial_target_block_merits = current_target_block_merits;
    }
    if (step > 0 && options.no_progress_max_steps > 0 && have_vel_result) {
      const bool low_error_change =
          std::abs(prev_combined_error - combined_error) <=
          options.no_progress_error_tolerance;
      const bool low_velocity =
          last_vel_result.joint_velocities.size() == robot_->nv() &&
          last_vel_result.joint_velocities.norm() <=
              options.no_progress_dq_norm_tolerance;
      if (low_error_change && low_velocity) {
        ++no_progress_count;
        if (no_progress_count >= options.no_progress_max_steps) {
          no_progress_exit = true;
          break;
        }
      } else {
        no_progress_count = 0;
      }
    }
    prev_combined_error = combined_error;

    for (size_t i = 0; i < n_targets; ++i) {
      resolved[i].task->setPositionStepTargetVelocity(target_velocities[i]);
    }

    const double step_dt_eff = compute_adaptive_position_step_dt(
        options, step_dt, max_position_error,
        collision_constraint_.has_value() && collision_constraint_->enabled,
        last_constraint_min_recovery_margin_);
    const auto priority_constraint_specs =
        build_priority_constraint_specs(step_dt_eff);
    pending_velocity_lock_indices_ = step_locked_indices;
    pending_step_torso_constraint_ = step_torso_constraint;
    pending_step_validation_dt_ = step_dt_eff;
    pending_position_step_priority_constraints_ = priority_constraint_specs;
    pending_reuse_current_kinematics_ = true;
    const bool apply_setpoint_acceleration_limits =
        acceleration_limits_enabled_ && !position_step_target_geometry_moved_ &&
        step == 0;
    pending_position_step_acceleration_limits_ =
        apply_setpoint_acceleration_limits;
    VelocitySolverResult vel_out = solve_velocity(q, true);
    vel_out = retry_auto_task_layout_as_split_if_needed(
        q, std::move(vel_out), step_locked_indices, step_torso_constraint,
        step_dt_eff, priority_constraint_specs,
        apply_setpoint_acceleration_limits);
    const bool allow_backtrack =
        options.stall_recovery && vel_out.status != SolverStatus::kSuccess &&
        (vel_out.status == SolverStatus::kInfeasible ||
         vel_out.status == SolverStatus::kNumericalError ||
         vel_out.status == SolverStatus::kNoProgress) &&
        vel_out.joint_velocities.size() == robot_->nv() &&
        (!vel_out.joint_velocities.allFinite() ||
         vel_out.joint_velocities.norm() <= stall_config_.dq_stall_eps);
    if (allow_backtrack) {
      for (size_t i = 0; i < n_targets; ++i) {
        resolved[i].task->setPositionStepTargetVelocity(
            kPositionStepBacktrackGainScale * target_velocities[i]);
      }
      pending_velocity_lock_indices_ = step_locked_indices;
      pending_step_torso_constraint_ = step_torso_constraint;
      pending_step_validation_dt_ = step_dt_eff;
      pending_position_step_priority_constraints_ = priority_constraint_specs;
      pending_reuse_current_kinematics_ = true;
      pending_position_step_acceleration_limits_ =
          apply_setpoint_acceleration_limits;
      VelocitySolverResult retry = solve_velocity(q, true);
      if (retry.status == SolverStatus::kSuccess) {
        vel_out = std::move(retry);
      }
    }
    have_vel_result = true;
    last_vel_result = std::move(vel_out);
    ++steps_used;

    if (last_vel_result.status != SolverStatus::kSuccess &&
        last_vel_result.status != SolverStatus::kInfeasible &&
        last_vel_result.status != SolverStatus::kNumericalError) {
      break;
    }

    Eigen::VectorXd q_pre_step = q;
    bool step_collision_rejected = false;
    const bool adaptive_step_large = (step_dt_eff > step_dt * 1.01);
    const auto integrate_velocity_candidate =
        [&](VelocitySolverResult &velocity_result, double integration_dt) {
          if (!options.integration_zero_velocity_indices.empty() &&
              velocity_result.joint_velocities.size() == robot_->nv()) {
            apply_integration_velocity_mask(
                velocity_result.joint_velocities,
                options.integration_zero_velocity_indices);
          }
          if (options.limit_change_from_seed &&
              velocity_result.joint_velocities.size() == robot_->nv()) {
            Eigen::VectorXd corridor_lower = Eigen::VectorXd::Constant(
                robot_->nv(), -kUnboundedConstraintLimit);
            Eigen::VectorXd corridor_upper = Eigen::VectorXd::Constant(
                robot_->nv(), kUnboundedConstraintLimit);
            tighten_bounds_with_reference_corridor(
                corridor_lower, corridor_upper, q_reference, q_pre_step,
                vel_limits, step_dt, velocity_to_config_index, robot_->nv());
            clamp_joint_velocity_solution_in_place(
                velocity_result.joint_velocities, corridor_lower,
                corridor_upper, false, robot_->nv());
          }
          if (use_position_limits_ &&
              velocity_result.joint_velocities.size() == robot_->nv()) {
            auto [q_min, q_max] = robot_->get_joint_limits();
            if (elastic_band_config_.enabled &&
                elastic_band_state_.delta.size() == robot_->nv()) {
              expand_scalar_joint_limits_for_velocity_delta(
                  q_min, q_max, elastic_band_state_.delta,
                  velocity_to_config_index, robot_->nv());
            }
            clamp_joint_velocity_to_position_limits_for_integration(
                velocity_result.joint_velocities, q_pre_step, q_min, q_max,
                velocity_to_config_index, integration_dt, robot_->nv());
          }
          Eigen::VectorXd candidate = pinocchio::integrate(
              robot_->model(), q_pre_step,
              integration_dt * velocity_result.joint_velocities);
          if (use_position_limits_) {
            auto [q_min_true, q_max_true] = robot_->get_joint_limits();
            project_scalar_configuration_to_true_joint_limits(
                candidate, q_min_true, q_max_true, velocity_to_config_index,
                robot_->nv());
          }
          const bool candidate_configuration_limited =
              clamp_configuration_delta_from_reference(
                  robot_->model(), q_reference, candidate,
                  options.max_configuration_step_norm);
          configuration_step_limited =
              candidate_configuration_limited || configuration_step_limited;
          robot_->update_configuration(candidate);
          return candidate;
        };

    q = integrate_velocity_candidate(last_vel_result, step_dt_eff);
    double priority_scale = position_step_priority_scale(q_pre_step, q);
    if (priority_scale < 1.0) {
      bool accepted_priority_retry = false;
      if (direct_priority_backtrack_is_safe && priority_scale > 0.0) {
        const Eigen::VectorXd nominal_delta =
            pinocchio::difference(robot_->model(), q_pre_step, q);
        Eigen::VectorXd backtracked_q = pinocchio::integrate(
            robot_->model(), q_pre_step, priority_scale * nominal_delta);
        if (position_step_priority_scale(q_pre_step, backtracked_q) >= 1.0) {
          q = std::move(backtracked_q);
          last_vel_result.joint_velocities =
              pinocchio::difference(robot_->model(), q_pre_step, q) /
              std::max(step_dt_eff, 1e-9);
          last_vel_result.solution.assign(
              last_vel_result.joint_velocities.data(),
              last_vel_result.joint_velocities.data() +
                  last_vel_result.joint_velocities.size());
          accepted_priority_retry = true;
        }
      }
      double retry_dt =
          step_dt_eff * (priority_scale > 0.0 ? priority_scale : 0.5);
      for (int retry_index = 0;
           !direct_priority_backtrack_is_safe &&
           !accepted_priority_retry &&
           retry_index < kPositionStepPriorityRetryIterations;
           ++retry_index) {
        for (std::size_t index = 0; index < resolved.size(); ++index) {
          resolved[index].task->setPositionStepTargetVelocity(
              target_velocities[index]);
        }
        robot_->update_configuration(q_pre_step);
        const auto retry_priority_constraint_specs =
            build_priority_constraint_specs(retry_dt);
        pending_velocity_lock_indices_ = step_locked_indices;
        pending_step_torso_constraint_ = step_torso_constraint;
        pending_step_validation_dt_ = retry_dt;
        pending_position_step_priority_constraints_ =
            retry_priority_constraint_specs;
        pending_reuse_current_kinematics_ = true;
        pending_position_step_acceleration_limits_ =
            apply_setpoint_acceleration_limits;
        VelocitySolverResult retry = solve_velocity(q_pre_step, true);
        retry = retry_auto_task_layout_as_split_if_needed(
            q_pre_step, std::move(retry), step_locked_indices,
            step_torso_constraint, retry_dt, retry_priority_constraint_specs,
            apply_setpoint_acceleration_limits);
        const bool retry_has_candidate =
            (retry.status == SolverStatus::kSuccess ||
             retry.status == SolverStatus::kInfeasible ||
             retry.status == SolverStatus::kNumericalError) &&
            retry.joint_velocities.size() == robot_->nv() &&
            retry.joint_velocities.allFinite();
        if (!retry_has_candidate) {
          retry_dt *= 0.5;
          continue;
        }
        Eigen::VectorXd retry_q =
            integrate_velocity_candidate(retry, retry_dt);
        const double retry_scale =
            position_step_priority_scale(q_pre_step, retry_q);
        if (retry_scale >= 1.0) {
          last_vel_result = std::move(retry);
          q = std::move(retry_q);
          accepted_priority_retry = true;
          break;
        }
        retry_dt *= retry_scale > 0.0 ? retry_scale : 0.5;
      }
      if (!accepted_priority_retry) {
        q = q_pre_step;
        robot_->update_configuration(q);
        priority_candidate_rejected = true;
        if (last_vel_result.joint_velocities.size() == robot_->nv()) {
          last_vel_result.joint_velocities.setZero();
        }
        break;
      }
    }
    if (step == 0 &&
        last_vel_result.joint_velocities.size() == robot_->nv() &&
        last_vel_result.joint_velocities.allFinite()) {
      first_tick_velocity = last_vel_result.joint_velocities;
    }

    // Skip expensive post-step checks when integration produced no motion.
    const bool step_moved =
        (q - q_pre_step).squaredNorm() > kCollisionEscapeNormEps;

    auto eval_broadphase_expanded = [&](const Eigen::VectorXd &q_eval)
        -> std::optional<double> {
      evaluate_post_step_collision_recovery_margins(q_eval);
      return evaluate_post_step_collision_distance_from_current_results();
    };

    // Post-solve collision rejection: use min_distance as the violation
    // threshold (mirrors single-target overload fix).
    if (step_moved && collision_constraint_.has_value() &&
        collision_constraint_->enabled) {
      // Keep the legacy global guard for penetration diagnostics while the
      // recovery margin below enforces every pair's effective floor.
      const double violation_threshold = collision_constraint_->min_distance;
      const auto pre_dist_debug =
          evaluate_post_step_collision_distance_mutating(q_pre_step);
      const auto pre_recovery_margins =
          evaluate_post_step_collision_recovery_margins(q_pre_step);
      auto post_dist_debug =
          adaptive_step_large ? eval_broadphase_expanded(q)
                              : evaluate_post_step_collision_distance_mutating(q);
      const auto post_recovery_margins =
          evaluate_post_step_collision_recovery_margins(q);
      const double pre_dist =
          (pre_dist_debug.has_value() && std::isfinite(*pre_dist_debug))
              ? *pre_dist_debug
              : std::numeric_limits<double>::infinity();
      const double post_dist =
          (post_dist_debug.has_value() && std::isfinite(*post_dist_debug))
              ? *post_dist_debug
              : std::numeric_limits<double>::infinity();
      const bool recovery_floor_unacceptable =
          !collision_recovery_margins_acceptable(
              pre_recovery_margins, post_recovery_margins, 0.0);
      const bool recovery_floor_seed_was_safe =
          collision_recovery_margins_are_safe(pre_recovery_margins, 0.0);
      if (post_dist < violation_threshold ||
          recovery_floor_unacceptable) {
        bool seed_was_safe = (pre_dist >= violation_threshold);
        bool deepened =
            (pre_dist < violation_threshold &&
             post_dist <
                 pre_dist - kCollisionPenetrationWorsenTolerance);
        const bool hard_jump =
            std::isfinite(pre_dist) &&
            post_dist < kCollisionHardPenetrationRejectDistance &&
            post_dist < pre_dist - kCollisionHardWorsenTolerance;
        const bool pure_primary_min_error =
            !last_vel_result.task_modes_effective.empty() &&
            last_vel_result.task_modes_effective[0] == TaskSolveMode::kMinError &&
            (last_vel_result.task_used_fallback.empty() ||
             !last_vel_result.task_used_fallback[0]);
        const bool step_has_velocity_locks =
            !options.excluded_joint_indices.empty() ||
            !options.integration_zero_velocity_indices.empty();
        const bool enforce_material_penetration_guard =
            !pure_primary_min_error || step_has_velocity_locks;
        const bool post_penetrating =
            post_dist < kCollisionPenetrationDistanceThreshold;
        const bool crossed_into_penetration =
            enforce_material_penetration_guard && post_penetrating &&
            (!std::isfinite(pre_dist) ||
             pre_dist >= kCollisionPenetrationDistanceThreshold);
        const bool material_penetration_worsened =
            enforce_material_penetration_guard &&
            post_penetrating && std::isfinite(pre_dist) &&
            pre_dist < kCollisionPenetrationDistanceThreshold &&
            post_dist < pre_dist - kCollisionTolerance;
        if (seed_was_safe || deepened || hard_jump ||
            crossed_into_penetration || material_penetration_worsened ||
            recovery_floor_unacceptable) {
          double backoff_threshold;
          if (seed_was_safe) {
            backoff_threshold = violation_threshold;
          } else if (crossed_into_penetration) {
            backoff_threshold = kCollisionPenetrationDistanceThreshold;
          } else if (material_penetration_worsened) {
            backoff_threshold = pre_dist - kCollisionTolerance;
          } else if (hard_jump) {
            backoff_threshold = kCollisionPenetrationDistanceThreshold;
          } else {
            backoff_threshold = pre_dist - kCollisionPenetrationWorsenTolerance;
          }
          bool accepted_backoff = false;
          const Eigen::VectorXd dq_nominal =
              pinocchio::difference(robot_->model(), q_pre_step, q);
          for (double frac : kCollisionRejectionBackoffFractions) {
            Eigen::VectorXd q_backoff = pinocchio::integrate(
                robot_->model(), q_pre_step, frac * dq_nominal);
            robot_->update_configuration(q_backoff);
            auto backoff_dist_debug =
                adaptive_step_large
                    ? eval_broadphase_expanded(q_backoff)
                    : evaluate_post_step_collision_distance_mutating(q_backoff);
            const auto backoff_recovery_margins =
                evaluate_post_step_collision_recovery_margins(q_backoff);
            const bool backoff_recovery_acceptable =
                collision_recovery_margins_acceptable(
                    pre_recovery_margins, backoff_recovery_margins, 0.0);
            if (backoff_dist_debug.has_value() &&
                std::isfinite(*backoff_dist_debug) &&
                *backoff_dist_debug >= backoff_threshold &&
                backoff_recovery_acceptable &&
                position_step_priority_scale(q_pre_step, q_backoff) >=
                    1.0) {
              q = q_backoff;
              accepted_backoff = true;
              break;
            }
          }
          if (!accepted_backoff) {
            q = q_pre_step;
            robot_->update_configuration(q);
            step_collision_rejected = true;
            result.collision_rejection_count++;
            if (seed_was_safe || recovery_floor_seed_was_safe) {
              collision_violated_flag_mts = true;
              break;
            }
          }
        }
      }
    }

    // When stalled, invalidate the collision pair cache so the next solve
    // tick performs a full scan and picks up any penetrating pairs that the
    // cached candidate set may have missed.
    // Only invalidate when actually near collision — in clear space the cache
    // remains valid and avoids an expensive full scan on the next step.
    const double step_dq_norm = last_vel_result.joint_velocities.norm();
    const double effective_step_dq_norm =
        step_collision_rejected ? 0.0 : step_dq_norm;
    if (effective_step_dq_norm < stall_config_.dq_stall_eps) {
      const bool near_penetration =
          !std::isfinite(last_constraint_min_distance_) ||
          last_collision_budget_exhausted_ ||
          last_constraint_min_distance_ <
              kCollisionPenetrationDistanceThreshold +
                  kPostStepSafeMargin;
      if (near_penetration) {
        collision_pair_cache_has_full_scan_ = false;
      }
    }

    // Stall escape: when stalled in penetration, nudge the config along the
    // collision normal to escape.  First check the QP's active constraint
    // list; if that doesn't show penetration, fall back to the full-scan
    // evaluate_collision_debug() which finds pairs the cache may have missed.
    // Skip the expensive escape logic entirely when we know from the
    // constraint computation that all pairs are well clear.
    const bool mt_stall_escape_eligible =
        collision_constraint_.has_value() && collision_constraint_->enabled &&
        (effective_step_dq_norm < stall_config_.dq_stall_eps ||
         step_collision_rejected);
    const bool mt_stall_escape_near_collision =
        mt_stall_escape_eligible &&
        (!std::isfinite(last_constraint_min_distance_) ||
         last_collision_budget_exhausted_ ||
         last_constraint_min_distance_ <
             collision_constraint_->min_distance +
                 kPostStepSafeMargin);
    if (mt_stall_escape_near_collision) {
      double min_dist = std::numeric_limits<double>::infinity();
      int worst_row = -1;
      for (int i = 0; i < static_cast<int>(last_collision_debug_list_.size()); ++i) {
        if (last_collision_debug_list_[i].distance < min_dist) {
          min_dist = last_collision_debug_list_[i].distance;
          worst_row = i;
        }
      }

      // For stalled steps, also allow escape when we violate configured
      // clearance (distance < min_distance) even if still non-penetrating.
      const double escape_activation_threshold =
          (step_collision_rejected
               ? kCollisionEscapeActivationDistance
               : std::max(kCollisionEscapeActivationDistance,
                          collision_constraint_->min_distance));
      bool use_jacobian_escape =
          (min_dist < escape_activation_threshold && worst_row >= 0);
      bool use_normal_escape = false;
      Eigen::Vector3d escape_normal = Eigen::Vector3d::Zero();
      std::string escape_frame_a, escape_frame_b;
      double global_min_dist = min_dist;

      if (!use_jacobian_escape) {
        // Fast path: when the constraint computation already has a reliable
        // global minimum distance, use it directly instead of the expensive
        // full-scan evaluate_collision_debug().  Only fall back to the full
        // scan when no constraint data is available or budget was exhausted.
        const bool have_reliable_constraint_dist =
            std::isfinite(last_constraint_min_distance_) &&
            !last_collision_budget_exhausted_;
        if (have_reliable_constraint_dist) {
          if (last_constraint_min_distance_ < escape_activation_threshold &&
              last_collision_debug_.has_value() &&
              std::isfinite(last_collision_debug_->distance)) {
            global_min_dist = last_collision_debug_->distance;
            Eigen::Vector3d diff = last_collision_debug_->point_b_world -
                                   last_collision_debug_->point_a_world;
            double diff_norm = diff.norm();
            if (diff_norm > kCollisionEscapeNormEps) {
              escape_normal = diff / diff_norm;
            } else {
              escape_normal = Eigen::Vector3d(0, 0, 1);
            }
            escape_frame_a = last_collision_debug_->object_a;
            escape_frame_b = last_collision_debug_->object_b;
            use_normal_escape = true;
          }
        } else {
          auto global_debug = evaluate_collision_debug(q);
          if (global_debug.has_value() && std::isfinite(global_debug->distance) &&
              global_debug->distance < escape_activation_threshold) {
            global_min_dist = global_debug->distance;
            Eigen::Vector3d diff = global_debug->point_b_world - global_debug->point_a_world;
            double diff_norm = diff.norm();
            if (diff_norm > kCollisionEscapeNormEps) {
              escape_normal = diff / diff_norm;
            } else {
              escape_normal = Eigen::Vector3d(0, 0, 1);
            }
            escape_frame_a = global_debug->object_a;
            escape_frame_b = global_debug->object_b;
            use_normal_escape = true;
          }
        }
      }

      if (use_jacobian_escape) {
        auto coll_result = compute_collision_constraint();
        if (coll_result.has_value() &&
            worst_row < coll_result->jacobian.rows()) {
          Eigen::RowVectorXd n_row = coll_result->jacobian.row(worst_row);
          double nn = n_row.squaredNorm();
          if (nn > kCollisionEscapeNormEps) {
            double escape_mag =
                std::min(kCollisionEscapeStepMax, std::abs(min_dist));
            if (step_collision_rejected) {
              escape_mag = std::max(escape_mag, kCollisionEscapeStepMinOnRejection);
            }
            const Eigen::VectorXd dq_unit =
                (1.0 / std::sqrt(nn)) * n_row.transpose();
            auto try_signed_escape = [&](double sign) -> bool {
              Eigen::VectorXd q_candidate = pinocchio::integrate(
                  robot_->model(), q, sign * escape_mag * dq_unit);
              auto [q_min, q_max] = robot_->get_joint_limits();
              for (int j = 0; j < static_cast<int>(q_candidate.size()); ++j) {
                if (j < q_min.size()) {
                  q_candidate[j] = std::max(q_candidate[j], q_min[j]);
                  q_candidate[j] = std::min(q_candidate[j], q_max[j]);
                }
              }
              robot_->update_configuration(q_candidate);
              auto esc_dist =
                  evaluate_post_step_collision_distance_mutating(q_candidate);
              if (esc_dist.has_value() && std::isfinite(*esc_dist) &&
                  *esc_dist > min_dist) {
                q = q_candidate;
                result.stall_escape_count++;
                return true;
              }
              robot_->update_configuration(q);
              return false;
            };
            if (!try_signed_escape(+1.0)) {
              try_signed_escape(-1.0);
            }
          }
        }
      } else if (use_normal_escape) {
        // Use contact normal from the globally-closest penetrating pair.
        // Build a body-level Jacobian for the link owning object_a, project
        // the Cartesian escape direction into joint space.
        const auto *collision_model =
            static_cast<const RobotModel &>(*robot_).collision_model();
        auto object_to_frame_name = [&](const std::string &obj_name) -> std::string {
          if (!collision_model) return std::string();
          for (std::size_t gi = 0; gi < collision_model->ngeoms; ++gi) {
            const auto &go = collision_model->geometryObjects[gi];
            if (go.name == obj_name) {
              return robot_->model().frames[go.parentFrame].name;
            }
          }
          return std::string();
        };
        const std::string escape_frame_a_name = object_to_frame_name(escape_frame_a);
        const std::string escape_frame_b_name = object_to_frame_name(escape_frame_b);

        auto try_frame_jacobian_escape = [&](const std::string &frame_name) -> bool {
          if (frame_name.empty() || !robot_->has_frame(frame_name)) return false;

          pinocchio::Data &data = robot_->data();
          pinocchio::computeJointJacobians(robot_->model(), data, q);
          auto frame_id = robot_->model().getFrameId(frame_name);
          Eigen::MatrixXd J(6, robot_->nv());
          J.setZero();
          pinocchio::getFrameJacobian(robot_->model(), data, frame_id,
                                      pinocchio::LOCAL_WORLD_ALIGNED, J);
          Eigen::MatrixXd J_lin = J.topRows<3>();
          Eigen::VectorXd dq_escape = J_lin.transpose() * escape_normal;
          double dq_n = dq_escape.norm();
          if (dq_n < kCollisionEscapeNormEps) return false;
          double escape_mag =
              std::min(kCollisionEscapeStepMax, std::abs(global_min_dist));
          if (step_collision_rejected) {
            escape_mag = std::max(escape_mag, kCollisionEscapeStepMinOnRejection);
          }
          dq_escape = (escape_mag / dq_n) * dq_escape;

          Eigen::VectorXd q_candidate = pinocchio::integrate(robot_->model(), q, dq_escape);
          auto [q_min, q_max] = robot_->get_joint_limits();
          for (int j = 0; j < static_cast<int>(q_candidate.size()); ++j) {
            if (j < q_min.size()) {
              q_candidate[j] = std::max(q_candidate[j], q_min[j]);
              q_candidate[j] = std::min(q_candidate[j], q_max[j]);
            }
          }
          robot_->update_configuration(q_candidate);
          auto esc_dist =
              evaluate_post_step_collision_distance_mutating(q_candidate);
          if (esc_dist.has_value() && std::isfinite(*esc_dist) &&
              *esc_dist > global_min_dist) {
            q = q_candidate;
            result.stall_escape_count++;
            return true;
          }
          robot_->update_configuration(q);
          return false;
        };

        if (!try_frame_jacobian_escape(escape_frame_b_name)) {
          try_frame_jacobian_escape(escape_frame_a_name);
        }
      }
    }

    // Limit-dominated lock breaker: if the QP reports near-zero joint motion
    // while several joints are saturated, nudge those joints slightly away
    // from hard limits to recover feasible directions.
    const bool collapsed_primary_scale =
        !last_vel_result.task_scales.empty() &&
        std::abs(last_vel_result.task_scales[0]) <= 1e-6;
    const bool plateau_status =
        last_vel_result.status == SolverStatus::kNumericalError ||
        last_vel_result.status == SolverStatus::kInfeasible ||
        collapsed_primary_scale;
    const int plateau_stall_steps =
        stall_handler_enabled() ? stall_state_.consecutive_stall_steps : 0;
    std::vector<int> desaturation_candidates = last_vel_result.saturated_joints;
    const bool saturated_zero_motion_plateau =
        combined_error > 1e-4 &&
        effective_step_dq_norm < stall_config_.dq_stall_eps &&
        !desaturation_candidates.empty() &&
        plateau_stall_steps >= kJointLimitDesaturationPlateauThreshold;
    const bool plateau_escape =
        (plateau_status || saturated_zero_motion_plateau) &&
        plateau_stall_steps >= kJointLimitDesaturationPlateauThreshold;
    auto [q_min, q_max] = robot_->get_joint_limits();
    if (plateau_escape) {
      for (int vi = 0;
           vi < static_cast<int>(velocity_to_config_index.size()); ++vi) {
        if (std::find(desaturation_candidates.begin(),
                      desaturation_candidates.end(),
                      vi) != desaturation_candidates.end()) {
          continue;
        }
        const int qi = velocity_to_config_index[vi];
        if (qi < 0 || qi >= static_cast<int>(q.size()) ||
            qi >= static_cast<int>(q_min.size()) ||
            qi >= static_cast<int>(q_max.size())) {
          continue;
        }
        const double margin_low = q[qi] - q_min[qi];
        const double margin_up = q_max[qi] - q[qi];
        if ((margin_low >= 0.0 &&
             margin_low < kJointLimitDesaturationExpandedMargin) ||
            (margin_up >= 0.0 &&
             margin_up < kJointLimitDesaturationExpandedMargin)) {
          desaturation_candidates.push_back(vi);
        }
      }
    }
    if (effective_step_dq_norm < stall_config_.dq_stall_eps &&
        (desaturation_candidates.size() >= 4 ||
         (plateau_escape && !desaturation_candidates.empty()))) {
      Eigen::VectorXd q_candidate = q;
      bool changed = false;
      const double desaturation_margin =
          plateau_escape ? kJointLimitDesaturationExpandedMargin
                         : kJointLimitDesaturationMargin;
      const double desaturation_step =
          plateau_escape ? kJointLimitDesaturationBoostStep
                         : kJointLimitDesaturationStep;
      for (int vi : desaturation_candidates) {
        if (vi < 0 || vi >= static_cast<int>(velocity_to_config_index.size())) {
          continue;
        }
        const int qi = velocity_to_config_index[vi];
        if (qi < 0 || qi >= static_cast<int>(q_candidate.size()) ||
            qi >= static_cast<int>(q_min.size()) ||
            qi >= static_cast<int>(q_max.size())) {
          continue;
        }
        const double margin_low = q_candidate[qi] - q_min[qi];
        const double margin_up = q_max[qi] - q_candidate[qi];
        if (margin_low >= 0.0 && margin_low < desaturation_margin) {
          q_candidate[qi] += desaturation_step;
          changed = true;
        } else if (margin_up >= 0.0 && margin_up < desaturation_margin) {
          q_candidate[qi] -= desaturation_step;
          changed = true;
        }
      }
      if (changed) {
        for (int j = 0; j < static_cast<int>(q_candidate.size()); ++j) {
          if (j < q_min.size()) {
            q_candidate[j] = std::max(q_candidate[j], q_min[j]);
            q_candidate[j] = std::min(q_candidate[j], q_max[j]);
          }
        }
        auto curr_dist = evaluate_post_step_collision_distance_mutating(q);
        robot_->update_configuration(q_candidate);
        auto cand_dist =
            evaluate_post_step_collision_distance_mutating(q_candidate);
        const double desat_safe_threshold_mt =
            (collision_constraint_.has_value() && collision_constraint_->enabled)
            ? collision_constraint_->min_distance
            : kCollisionPenetrationDistanceThreshold;
        if (collision_recovery_candidate_acceptable(
                curr_dist, cand_dist, desat_safe_threshold_mt)) {
          q = q_candidate;
          result.stall_escape_count++;
          recovery_inside_collision_margin =
              recovery_inside_collision_margin ||
              (curr_dist.has_value() && std::isfinite(*curr_dist) &&
               *curr_dist < desat_safe_threshold_mt &&
               cand_dist.has_value() && std::isfinite(*cand_dist) &&
               *cand_dist < desat_safe_threshold_mt);
        } else {
          robot_->update_configuration(q);
        }
      }
    }
    if (has_active_velocity_centroidal_hard_constraints()) {
      const Eigen::VectorXd accepted_velocity =
          pinocchio::difference(robot_->model(), q_pre_step, q) /
          std::max(step_dt_eff, 1e-9);
      std::string validation_message;
      if (!validate_centroidal_velocity_candidate(
              q_pre_step, active_explicit_current_dq(), accepted_velocity,
              &validation_message)) {
        q = q_pre_step;
        robot_->update_configuration(q);
        last_vel_result.status = SolverStatus::kInfeasible;
        last_vel_result.status_message = validation_message;
        last_vel_result.joint_velocities =
            Eigen::VectorXd::Zero(robot_->nv());
        last_vel_result.solution.assign(
            static_cast<std::size_t>(robot_->nv()), 0.0);
        break;
      }
      last_vel_result.joint_velocities = accepted_velocity;
      last_vel_result.solution.assign(
          accepted_velocity.data(),
          accepted_velocity.data() + accepted_velocity.size());
      if (position_step_explicit_current_dq_.has_value()) {
        position_step_explicit_current_dq_ = accepted_velocity;
      }
    }
  }

  if (auto retry_result = attempt_min_error_position_step_retry(
          current_q, options, step_dt, have_vel_result, last_vel_result, q,
          [&resolved](std::vector<std::pair<Task *, TaskSolveMode>> *saved_modes) {
            return flip_primary_scale_tasks_to_min_error(resolved, saved_modes);
          },
          [&]() { return solve_position_step(current_q, targets, options); });
      retry_result.has_value()) {
    return retry_result.value();
  }

  if (position_step_target_geometry_moved_) {
    const Eigen::VectorXd terminal_q = q;
    robot_->update_configuration(current_q);
    bool all_target_blocks_commanded = true;
    double componentwise_prediction_weight = 1.0;
    for (size_t i = 0; i < n_targets; ++i) {
      const auto &target = targets[i];
      const auto &rt = resolved[i];
      rt.task->update(*robot_);
      const Eigen::VectorXd &current_error = rt.task->getError();
      TaskType command_task_type = TaskType::FRAME_POSE;
      if (rt.kind == PoseTaskKind::kFrame) {
        command_task_type =
            static_cast<const FrameTask *>(rt.task.get())->getType();
      }
      all_target_blocks_commanded =
          all_target_blocks_commanded &&
          all_frame_task_blocks_commanded(
              current_error, command_task_type, target.position_gain,
              target.orientation_gain);
      if (current_error.size() == 3) {
        bool is_orientation_only = false;
        if (rt.kind == PoseTaskKind::kFrame) {
          const auto *ft = static_cast<const FrameTask *>(rt.task.get());
          is_orientation_only =
              ft->getType() == TaskType::FRAME_ORIENTATION;
        }
        Eigen::VectorXd current_velocity =
            (is_orientation_only ? target.orientation_gain
                                 : target.position_gain) *
            current_error.head<3>();
        const double speed_limit = is_orientation_only
                                       ? options.max_angular_speed
                                       : options.max_linear_speed;
        if (speed_limit > 0.0 && current_velocity.norm() > speed_limit) {
          current_velocity *= speed_limit / current_velocity.norm();
        }
        rt.task->setPositionStepTargetVelocity(current_velocity);
      } else if (current_error.size() >= 6) {
        vel.head<3>() = target.position_gain * current_error.head<3>();
        vel.tail<3>() = target.orientation_gain * current_error.tail<3>();
        clamp_spatial_velocity_components(vel, options.max_linear_speed,
                                          options.max_angular_speed);
        rt.task->setPositionStepTargetVelocity(vel);
      }
      componentwise_prediction_weight =
          std::min(componentwise_prediction_weight,
                   position_step_task_terminal_prediction_weight(
                       *rt.task, current_error));
    }
    const auto componentwise_candidate =
        !has_active_velocity_centroidal_hard_constraints() &&
                all_target_blocks_commanded &&
                componentwise_prediction_weight > 0.0
            ? estimate_position_step_componentwise_outer_candidate(
                  current_q, previous_applied_velocity, options, step_dt,
                  terminal_q)
            : std::nullopt;
    const auto outer_priority_constraint_specs =
        build_priority_constraint_specs(step_dt);
    if (auto projected_result = apply_position_step_task_metric_projection(
            current_q, step_dt, first_tick_velocity, step_locked_indices,
            step_torso_constraint, outer_priority_constraint_specs,
            options.current_joint_velocity, q);
        projected_result.has_value()) {
      last_vel_result = std::move(*projected_result);
      have_vel_result = true;
      if (outer_priority_constraint_specs.empty()) {
        outer_acceleration_limit_applied = true;
        if (componentwise_candidate.has_value() &&
            all_target_blocks_commanded &&
            componentwise_prediction_weight > 0.0) {
          const Eigen::VectorXd projected_q = q;
          const auto legacy_target_block_merits_at =
              [&](const Eigen::VectorXd &candidate_q) {
                robot_->update_configuration(candidate_q);
                std::vector<std::array<double, 2>> merits;
                merits.reserve(resolved.size());
                for (std::size_t index = 0; index < resolved.size(); ++index) {
                  const auto &resolved_target = resolved[index];
                  resolved_target.task->update(*robot_);
                  TaskType merit_task_type = TaskType::FRAME_POSE;
                  if (resolved_target.kind == PoseTaskKind::kFrame) {
                    merit_task_type =
                        static_cast<const FrameTask *>(resolved_target.task.get())
                            ->getType();
                  }
                  merits.push_back(commanded_frame_block_merits(
                      resolved_target.task->getError(), merit_task_type,
                      targets[index].position_gain,
                      targets[index].orientation_gain));
                }
                return merits;
              };
          const auto projected_merits = legacy_target_block_merits_at(q);
          const auto candidate_is_preferred =
              [&](const Eigen::VectorXd &candidate_q) {
                const auto candidate_merits =
                    legacy_target_block_merits_at(candidate_q);
                double projected_total = 0.0;
                double candidate_total = 0.0;
                bool all_targets_non_worsening =
                    projected_merits.size() == candidate_merits.size();
                for (std::size_t index = 0;
                     all_targets_non_worsening &&
                     index < projected_merits.size();
                     ++index) {
                  for (int block = 0; block < 2; ++block) {
                    all_targets_non_worsening =
                        all_targets_non_worsening &&
                        std::isfinite(projected_merits[index][block]) &&
                        std::isfinite(candidate_merits[index][block]) &&
                        candidate_merits[index][block] <=
                            projected_merits[index][block] + 1e-9;
                    projected_total += projected_merits[index][block];
                    candidate_total += candidate_merits[index][block];
                  }
                }
                return all_targets_non_worsening &&
                       std::isfinite(projected_total) &&
                       std::isfinite(candidate_total) &&
                       candidate_total + 1e-9 < projected_total;
              };
          Eigen::VectorXd safe_componentwise_candidate =
              *componentwise_candidate;
          if (componentwise_prediction_weight < 1.0) {
            const Eigen::VectorXd projected_delta = pinocchio::difference(
                robot_->model(), current_q, projected_q);
            const Eigen::VectorXd componentwise_delta = pinocchio::difference(
                robot_->model(), current_q, safe_componentwise_candidate);
            safe_componentwise_candidate = pinocchio::integrate(
                robot_->model(), current_q,
                projected_delta + componentwise_prediction_weight *
                                      (componentwise_delta - projected_delta));
          }
          if (candidate_is_preferred(safe_componentwise_candidate)) {
            apply_position_step_outer_acceleration_limit(
                current_q, previous_applied_velocity, options, step_dt,
                safe_componentwise_candidate);
            if (candidate_is_preferred(safe_componentwise_candidate)) {
              q = std::move(safe_componentwise_candidate);
              outer_acceleration_limit_applied = true;
            }
          }
          if (outer_acceleration_limit_applied) {
            last_vel_result.joint_velocities =
                pinocchio::difference(robot_->model(), current_q, q) / step_dt;
          }
          robot_->update_configuration(q);
        }
      } else {
        const Eigen::VectorXd projected_q = q;
        Eigen::VectorXd outer_validated_projected_q = projected_q;
        const bool projection_was_validated =
            apply_position_step_outer_acceleration_limit(
                current_q, previous_applied_velocity, options, step_dt,
                outer_validated_projected_q);
        const Eigen::VectorXd projection_adjustment = pinocchio::difference(
            robot_->model(), projected_q, outer_validated_projected_q);
        outer_acceleration_limit_applied =
            projection_was_validated && projection_adjustment.allFinite() &&
            projection_adjustment.norm() <=
                10.0 * constraint_tolerance_ * std::max(step_dt, 1e-9);
        robot_->update_configuration(projected_q);
        if (componentwise_candidate.has_value() &&
            all_target_blocks_commanded &&
            componentwise_prediction_weight > 0.0) {
          const auto current_merits = target_block_merits_at(current_q);
          const auto projected_merits = target_block_merits_at(q);
          const auto candidate_is_preferred =
              [&](const Eigen::VectorXd &candidate_q) {
                const auto candidate_merits = target_block_merits_at(candidate_q);
                if (!priority_candidate_acceptable(current_merits,
                                                    candidate_merits)) {
                  return false;
                }
                double projected_total = 0.0;
                double candidate_total = 0.0;
                bool lower_priority_merits_valid =
                    projected_merits.size() == candidate_merits.size();
                for (std::size_t index = 0;
                     lower_priority_merits_valid &&
                     index < projected_merits.size();
                     ++index) {
                  if (!resolved[index].task->isActive() ||
                      std::binary_search(
                          protected_target_priorities.begin(),
                          protected_target_priorities.end(),
                          resolved[index].task->getPriority())) {
                    continue;
                  }
                  for (int block = 0; block < 2; ++block) {
                    lower_priority_merits_valid =
                        lower_priority_merits_valid &&
                        std::isfinite(projected_merits[index][block]) &&
                        std::isfinite(candidate_merits[index][block]);
                    projected_total += projected_merits[index][block];
                    candidate_total += candidate_merits[index][block];
                  }
                }
                return lower_priority_merits_valid &&
                       std::isfinite(projected_total) &&
                       std::isfinite(candidate_total) &&
                       candidate_total + 1e-9 < projected_total;
              };
          Eigen::VectorXd safe_componentwise_candidate =
              *componentwise_candidate;
          if (componentwise_prediction_weight < 1.0) {
            const Eigen::VectorXd projected_delta = pinocchio::difference(
                robot_->model(), current_q, projected_q);
            const Eigen::VectorXd componentwise_delta = pinocchio::difference(
                robot_->model(), current_q, safe_componentwise_candidate);
            safe_componentwise_candidate = pinocchio::integrate(
                robot_->model(), current_q,
                projected_delta + componentwise_prediction_weight *
                                      (componentwise_delta - projected_delta));
          }
          if (candidate_is_preferred(safe_componentwise_candidate)) {
            apply_position_step_outer_acceleration_limit(
                current_q, previous_applied_velocity, options, step_dt,
                safe_componentwise_candidate);
            if (candidate_is_preferred(safe_componentwise_candidate)) {
              const Eigen::VectorXd componentwise_velocity =
                  pinocchio::difference(robot_->model(), current_q,
                                        safe_componentwise_candidate) /
                  std::max(step_dt, 1e-9);
              auto componentwise_projection =
                  apply_position_step_task_metric_projection(
                      current_q, step_dt, componentwise_velocity,
                      step_locked_indices, step_torso_constraint,
                      outer_priority_constraint_specs,
                      options.current_joint_velocity,
                      safe_componentwise_candidate);
              if (componentwise_projection.has_value() &&
                  componentwise_projection->joint_velocities.size() ==
                      robot_->nv() &&
                  componentwise_projection->joint_velocities.allFinite() &&
                  candidate_is_preferred(safe_componentwise_candidate)) {
                q = std::move(safe_componentwise_candidate);
                last_vel_result = std::move(*componentwise_projection);
                outer_acceleration_limit_applied = true;
              }
            }
          }
          if (outer_acceleration_limit_applied) {
            last_vel_result.joint_velocities =
                pinocchio::difference(robot_->model(), current_q, q) / step_dt;
          }
          robot_->update_configuration(q);
        }
      }
    }
  } else if (acceleration_limits_enabled_ &&
             first_tick_velocity.size() == robot_->nv() &&
             pinocchio::difference(robot_->model(), current_q, q)
                     .squaredNorm() > kCollisionEscapeNormEps) {
    q = pinocchio::integrate(robot_->model(), current_q,
                             step_dt * first_tick_velocity);
    if (use_position_limits_) {
      auto [q_min, q_max] = robot_->get_joint_limits();
      project_scalar_configuration_to_true_joint_limits(
          q, q_min, q_max, velocity_to_config_index, robot_->nv());
    }
    robot_->update_configuration(q);
  }

  const bool final_configuration_limited =
      clamp_configuration_delta_from_reference(
          robot_->model(), q_reference, q,
          options.max_configuration_step_norm);
  configuration_step_limited =
      final_configuration_limited || configuration_step_limited;
  if (final_configuration_limited) {
    robot_->update_configuration(q);
  }
  const bool final_outer_acceleration_limit_applied =
      apply_position_step_outer_acceleration_limit(
          current_q, previous_applied_velocity, options, step_dt, q);
  outer_acceleration_limit_applied =
      outer_acceleration_limit_applied ||
      final_outer_acceleration_limit_applied;
  double final_priority_scale = position_step_priority_scale(current_q, q);
  if (final_priority_scale < 1.0) {
    const Eigen::VectorXd terminal_delta =
        pinocchio::difference(robot_->model(), current_q, q);
    bool accepted_final_priority_retry = false;
    if (direct_priority_backtrack_is_safe && final_priority_scale > 0.0) {
      Eigen::VectorXd backtracked_q = pinocchio::integrate(
          robot_->model(), current_q,
          final_priority_scale * terminal_delta);
      if (position_step_priority_scale(current_q, backtracked_q) >= 1.0) {
        q = std::move(backtracked_q);
        last_vel_result.joint_velocities =
            pinocchio::difference(robot_->model(), current_q, q) /
            std::max(step_dt, 1e-9);
        last_vel_result.solution.assign(
            last_vel_result.joint_velocities.data(),
            last_vel_result.joint_velocities.data() +
                last_vel_result.joint_velocities.size());
        have_vel_result = true;
        accepted_final_priority_retry = true;
      }
    }
    double retry_fraction =
        final_priority_scale > 0.0 ? final_priority_scale : 0.5;
    for (int retry_index = 0;
         !direct_priority_backtrack_is_safe &&
         !accepted_final_priority_retry &&
         retry_index < kPositionStepPriorityRetryIterations;
         ++retry_index) {
      Eigen::VectorXd retry_q = pinocchio::integrate(
          robot_->model(), current_q, retry_fraction * terminal_delta);
      auto projected_result = apply_position_step_task_metric_projection(
          current_q, step_dt, Eigen::VectorXd(), step_locked_indices,
          step_torso_constraint, build_priority_constraint_specs(step_dt),
          options.current_joint_velocity, retry_q);
      const bool retry_has_candidate =
          projected_result.has_value() &&
          projected_result->joint_velocities.size() == robot_->nv() &&
          projected_result->joint_velocities.allFinite();
      if (!retry_has_candidate) {
        retry_fraction *= 0.5;
        continue;
      }
      if (acceleration_limits_enabled_ &&
          !apply_position_step_outer_acceleration_limit(
              current_q, previous_applied_velocity, options, step_dt,
              retry_q)) {
        retry_fraction *= 0.5;
        continue;
      }

      bool collision_acceptable = true;
      if (collision_constraint_.has_value() &&
          collision_constraint_->enabled) {
        const auto current_dist =
            evaluate_post_step_collision_distance_mutating(current_q);
        const auto current_margins =
            evaluate_post_step_collision_recovery_margins(current_q);
        const auto retry_dist =
            evaluate_post_step_collision_distance_mutating(retry_q);
        const auto retry_margins =
            evaluate_post_step_collision_recovery_margins(retry_q);
        collision_acceptable =
            current_dist.has_value() && retry_dist.has_value() &&
            std::isfinite(*current_dist) && std::isfinite(*retry_dist) &&
            collision_recovery_margins_acceptable(
                current_margins, retry_margins, 0.0) &&
            ((*current_dist >= collision_constraint_->min_distance)
                 ? (*retry_dist >= collision_constraint_->min_distance -
                                      kCollisionTolerance)
                 : (*retry_dist >=
                    *current_dist - kCollisionPenetrationWorsenTolerance));
      }
      const double retry_priority_scale =
          position_step_priority_scale(current_q, retry_q);
      if (collision_acceptable && retry_priority_scale >= 1.0) {
        q = std::move(retry_q);
        projected_result->joint_velocities =
            pinocchio::difference(robot_->model(), current_q, q) /
            std::max(step_dt, 1e-9);
        last_vel_result = std::move(*projected_result);
        have_vel_result = true;
        outer_acceleration_limit_applied = true;
        accepted_final_priority_retry = true;
        break;
      }
      retry_fraction *=
          retry_priority_scale > 0.0 ? retry_priority_scale : 0.5;
    }
    if (!accepted_final_priority_retry) {
      q = current_q;
      priority_candidate_rejected = true;
    }
    robot_->update_configuration(q);
  }

  std::unordered_set<Task *> resolved_tasks;
  resolved_tasks.reserve(resolved.size());
  for (const auto &rt : resolved) {
    resolved_tasks.insert(rt.task.get());
    rt.task->clearPositionStepTargetVelocity();
  }
  for (auto &task : tasks_) {
    if (task && resolved_tasks.find(task.get()) == resolved_tasks.end()) {
      task->clearPositionStepTargetVelocity();
    }
  }

  if (has_active_velocity_centroidal_hard_constraints()) {
    const Eigen::VectorXd final_velocity =
        pinocchio::difference(robot_->model(), current_q, q) /
        std::max(step_dt, 1e-9);
    const Eigen::VectorXd *final_current_dq =
        options.current_joint_velocity.size() == robot_->nv()
            ? &options.current_joint_velocity
            : nullptr;
    std::string validation_message;
    if (!validate_centroidal_velocity_candidate(
            current_q, final_current_dq, final_velocity, &validation_message)) {
      q = current_q;
      last_vel_result.status = SolverStatus::kInfeasible;
      last_vel_result.status_message = validation_message;
      last_vel_result.joint_velocities = Eigen::VectorXd::Zero(robot_->nv());
      last_vel_result.solution.assign(
          static_cast<std::size_t>(robot_->nv()), 0.0);
    } else {
      last_vel_result.joint_velocities = final_velocity;
      last_vel_result.solution.assign(
          final_velocity.data(), final_velocity.data() + final_velocity.size());
    }
    have_vel_result = true;
  }
  robot_->update_configuration(q);

  result.q_solution = q;
  result.iterations_used = steps_used;

  const auto &primary = resolved.front();
  primary.task->update(*robot_);
  const Eigen::VectorXd &final_error = primary.task->getError();
  if (final_error.size() == 3) {
    result.position_error = final_error.head<3>().norm();
    result.orientation_error = 0.0;
  } else if (final_error.size() >= 6) {
    result.position_error = final_error.head<3>().norm();
    result.orientation_error = final_error.tail<3>().norm();
  }

  if (primary.kind == PoseTaskKind::kFrame) {
    result.achieved_pose = robot_->get_frame_pose(
        static_cast<FrameTask *>(primary.task.get())->getFrameName());
  } else {
    result.achieved_pose = targets.front().target_pose;
  }

  if (have_vel_result) {
    if (last_vel_result.joint_velocities.size() == robot_->nv()) {
      last_vel_result.solution.assign(last_vel_result.joint_velocities.data(),
                                      last_vel_result.joint_velocities.data() +
                                          last_vel_result.joint_velocities.size());
    }
    auto saved_status = result.status;
    auto saved_msg = std::move(result.status_message);
    static_cast<VelocitySolverResult &>(result) = std::move(last_vel_result);
    if (saved_status == SolverStatus::kInvalidInput && !saved_msg.empty()) {
      result.status = saved_status;
      result.status_message = std::move(saved_msg);
    }
    if (!runtime_config_.enable_auto_task_layout) {
      bool saw_pose = false;
      bool saw_split = false;
      for (const auto &rt : resolved) {
        if (rt.kind != PoseTaskKind::kFrame) {
          continue;
        }
        const auto *frame_task = static_cast<const FrameTask *>(rt.task.get());
        const TaskType type = frame_task->getType();
        saw_pose = saw_pose || type == TaskType::FRAME_POSE;
        saw_split = saw_split || type == TaskType::FRAME_POSITION ||
                                type == TaskType::FRAME_ORIENTATION;
      }
      result.active_task_layout =
          (saw_pose && !saw_split) ? TaskLayout::kMerged : TaskLayout::kSplit;
    }
  } else if (result.status_message.empty()) {
    result.status = SolverStatus::kInvalidInput;
    result.status_message = "no solve step executed in solve_position_step";
  }
  if (no_progress_exit && result.status != SolverStatus::kInvalidInput) {
    result.status = SolverStatus::kNoProgress;
    result.status_message =
        "solve_position_step exited due to no progress near active bounds/"
        "constraints";
  }
  if (collision_violated_flag_mts) {
    result.status = SolverStatus::kCollisionViolated;
    result.status_message =
        "solve_position_step (multi-target): no step could maintain collision "
        "min_distance; q_solution is the last safe configuration";
  }
  if (collision_constraint_.has_value() && collision_constraint_->enabled &&
      (result.stall_escape_count > 0 ||
       result.collision_rejection_count > 0 ||
       recovery_inside_collision_margin)) {
    auto final_dist = evaluate_min_collision_distance(q);
    if (final_dist.has_value() && std::isfinite(*final_dist) &&
        *final_dist < collision_constraint_->min_distance - kCollisionTolerance) {
      recovery_inside_collision_margin = true;
    }
  }
  if (recovery_inside_collision_margin &&
      result.status == SolverStatus::kSuccess) {
    result.status = SolverStatus::kNoProgress;
    result.status_message =
        "solve_position_step applied recovery motion inside collision margin";
  }
  if (!recovery_inside_collision_margin &&
      should_report_position_recovery_success(
          result, current_q,
          stall_config_.dq_stall_eps * std::max(step_dt, 1e-9))) {
    result.status = SolverStatus::kSuccess;
    result.status_message =
        "solve_position_step applied constraint recovery motion";
  }
  if (priority_candidate_rejected &&
      result.status != SolverStatus::kInvalidInput &&
      result.status != SolverStatus::kCollisionViolated) {
    const bool priority_hold =
        pinocchio::difference(robot_->model(), current_q, q).squaredNorm() <=
        kPositionStepMeritMotionThreshold;
    result.position_step_hold_active = priority_hold;
    result.status = priority_hold ? SolverStatus::kNoProgress
                                  : SolverStatus::kSuccess;
    result.status_message =
        priority_hold
            ? "solve_position_step held current configuration because no "
              "priority-safe candidate passed final constraint validation"
            : "solve_position_step decelerating before priority-safe hold";
  }
  double final_commanded_error = 0.0;
  double final_primary_combined_error =
      std::numeric_limits<double>::quiet_NaN();
  double final_primary_position_error = 0.0;
  double final_primary_orientation_error = 0.0;
  std::vector<double> final_target_merits;
  final_target_merits.reserve(resolved.size());
  std::vector<double> final_target_block_merits;
  final_target_block_merits.reserve(2 * resolved.size());
  for (std::size_t index = 0; index < resolved.size(); ++index) {
    const auto &resolved_target = resolved[index];
    resolved_target.task->update(*robot_);
    TaskType merit_task_type = TaskType::FRAME_POSE;
    if (resolved_target.kind == PoseTaskKind::kFrame) {
      merit_task_type = static_cast<const FrameTask *>(
                            resolved_target.task.get())
                            ->getType();
    }
    const auto block_merits = commanded_frame_block_merits(
        resolved_target.task->getError(), merit_task_type,
        targets[index].position_gain, targets[index].orientation_gain);
    const double target_merit = block_merits[0] + block_merits[1];
    final_commanded_error += target_merit;
    final_target_merits.push_back(target_merit);
    final_target_block_merits.push_back(block_merits[0]);
    final_target_block_merits.push_back(block_merits[1]);
    if (resolved_target.task->isActive() &&
        resolved_target.task->getPriority() ==
            highest_active_target_priority) {
      final_primary_combined_error =
          std::isfinite(final_primary_combined_error)
              ? std::max(final_primary_combined_error, target_merit)
              : target_merit;
      final_primary_position_error =
          std::max(final_primary_position_error, block_merits[0]);
      final_primary_orientation_error =
          std::max(final_primary_orientation_error, block_merits[1]);
    }
  }
  const double candidate_step_norm =
      result.q_solution.size() == current_q.size()
          ? pinocchio::difference(robot_->model(), current_q,
                                  result.q_solution)
                .norm()
          : std::numeric_limits<double>::quiet_NaN();
  int continuity_priority = highest_active_target_priority;
  std::vector<int> active_target_priorities;
  active_target_priorities.reserve(resolved.size());
  for (const auto &resolved_target : resolved) {
    if (resolved_target.task->isActive()) {
      active_target_priorities.push_back(resolved_target.task->getPriority());
    }
  }
  std::sort(active_target_priorities.begin(), active_target_priorities.end());
  active_target_priorities.erase(
      std::unique(active_target_priorities.begin(),
                  active_target_priorities.end()),
      active_target_priorities.end());
  for (int priority : active_target_priorities) {
    bool priority_unsatisfied = false;
    for (std::size_t index = 0; index < resolved.size(); ++index) {
      if (!resolved[index].task->isActive() ||
          resolved[index].task->getPriority() != priority) {
        continue;
      }
      priority_unsatisfied =
          priority_unsatisfied ||
          std::max(initial_target_merits[index], final_target_merits[index]) >
              kPositionStepSatisfiedMeritTolerance;
    }
    if (priority_unsatisfied) {
      continuity_priority = priority;
      break;
    }
  }
  double continuity_initial_merit = 0.0;
  double continuity_final_merit = 0.0;
  std::vector<double> continuity_initial_target_merits;
  std::vector<double> continuity_final_target_merits;
  std::vector<double> continuity_initial_block_merits;
  std::vector<double> continuity_final_block_merits;
  continuity_initial_block_merits.reserve(2 * resolved.size());
  continuity_final_block_merits.reserve(2 * resolved.size());
  for (std::size_t index = 0; index < resolved.size(); ++index) {
    if (!resolved[index].task->isActive() ||
        resolved[index].task->getPriority() != continuity_priority) {
      continue;
    }
    continuity_initial_merit += initial_target_merits[index];
    continuity_final_merit += final_target_merits[index];
    continuity_initial_target_merits.push_back(initial_target_merits[index]);
    continuity_final_target_merits.push_back(final_target_merits[index]);
    continuity_initial_block_merits.push_back(
        initial_target_block_merits[2 * index]);
    continuity_initial_block_merits.push_back(
        initial_target_block_merits[2 * index + 1]);
    continuity_final_block_merits.push_back(
        final_target_block_merits[2 * index]);
    continuity_final_block_merits.push_back(
        final_target_block_merits[2 * index + 1]);
  }
  const bool held_non_improving_step =
      !priority_candidate_rejected &&
      should_hold_position_step_for_continuity(
          result, current_q, continuity_initial_merit,
          continuity_final_merit, continuity_initial_target_merits,
          continuity_final_target_merits, continuity_initial_block_merits,
          continuity_final_block_merits, continuity_priority,
          candidate_step_norm,
          owns_position_step_continuity, collision_violated_flag_mts,
          step_torso_constraint.has_value() ||
              !build_priority_constraint_specs(step_dt).empty());
  if (held_non_improving_step) {
    const bool target_satisfied =
        continuity_initial_merit <= kPositionStepSatisfiedMeritTolerance;
    Eigen::VectorXd braking_q = current_q;
    const bool braking_to_hold = compute_position_step_continuity_brake(
        current_q, previous_applied_velocity, options, step_dt, braking_q);
    result.position_step_hold_active = !braking_to_hold;
    q = braking_to_hold ? std::move(braking_q) : current_q;
    robot_->update_configuration(q);
    result.q_solution = q;
    primary.task->update(*robot_);
    const Eigen::VectorXd &held_error = primary.task->getError();
    if (held_error.size() == 3) {
      result.position_error = held_error.head<3>().norm();
      result.orientation_error = 0.0;
    } else if (held_error.size() >= 6) {
      result.position_error = held_error.head<3>().norm();
      result.orientation_error = held_error.tail<3>().norm();
    }
    if (primary.kind == PoseTaskKind::kFrame) {
      result.achieved_pose = robot_->get_frame_pose(
          static_cast<FrameTask *>(primary.task.get())->getFrameName());
    } else {
      result.achieved_pose = targets.front().target_pose;
    }
    result.status = braking_to_hold
                        ? SolverStatus::kSuccess
                        : (target_satisfied ? SolverStatus::kSuccess
                                            : SolverStatus::kNoProgress);
    result.status_message =
        braking_to_hold
            ? "solve_position_step decelerating before continuity hold"
            : (target_satisfied
                   ? "solve_position_step held satisfied stationary target at "
                     "the current configuration"
                   : "solve_position_step held current configuration because "
                     "the nominal step did not reduce commanded task error");
    sync_position_result_applied_velocity(result, robot_->model(), current_q,
                                          step_dt);
    last_solution_dq_norm_ = result.joint_velocities.norm();
    if (!braking_to_hold && previous_dq_.size() == robot_->nv()) {
      previous_dq_.setZero();
    }
  }
  if (result.stall_escape_count > 0 || result.collision_rejection_count > 0 ||
      collision_violated_flag_mts || configuration_step_limited) {
    sync_position_result_applied_velocity(result, robot_->model(), current_q,
                                          step_dt);
  }
  if (!priority_candidate_rejected && !held_non_improving_step &&
      should_hold_soft_infeasible_position_step(
          result, current_q, initial_primary_combined_error,
          final_primary_combined_error, final_primary_position_error,
          final_primary_orientation_error,
          collision_violated_flag_mts, recovery_inside_collision_margin)) {
    Eigen::VectorXd braking_q = current_q;
    const bool braking_to_hold = compute_position_step_continuity_brake(
        current_q, previous_applied_velocity, options, step_dt, braking_q);
    result.position_step_hold_active = !braking_to_hold;
    q = braking_to_hold ? std::move(braking_q) : current_q;
    robot_->update_configuration(q);
    result.q_solution = q;
    primary.task->update(*robot_);
    const Eigen::VectorXd &held_error = primary.task->getError();
    if (held_error.size() == 3) {
      result.position_error = held_error.head<3>().norm();
      result.orientation_error = 0.0;
    } else if (held_error.size() >= 6) {
      result.position_error = held_error.head<3>().norm();
      result.orientation_error = held_error.tail<3>().norm();
    }
    if (primary.kind == PoseTaskKind::kFrame) {
      result.achieved_pose = robot_->get_frame_pose(
          static_cast<FrameTask *>(primary.task.get())->getFrameName());
    } else {
      result.achieved_pose = targets.front().target_pose;
    }
    result.status = braking_to_hold ? SolverStatus::kSuccess
                                    : SolverStatus::kNoProgress;
    result.status_message = braking_to_hold
                                ? "solve_position_step decelerating before "
                                  "soft-infeasible continuity hold"
                                : "solve_position_step held current configuration "
                                  "because the primary SCALE task was "
                                  "soft-infeasible";
    sync_position_result_applied_velocity(result, robot_->model(), current_q,
                                          step_dt);
    last_solution_dq_norm_ = result.joint_velocities.norm();
    if (!braking_to_hold && previous_dq_.size() == robot_->nv()) {
      previous_dq_.setZero();
    }
  }

  sync_position_result_applied_velocity(result, robot_->model(), current_q,
                                        step_dt);
  if (!enforce_final_position_step_centroidal_candidate(
          current_q, options.current_joint_velocity, step_dt, result)) {
    q = current_q;
    primary.task->update(*robot_);
    const Eigen::VectorXd &held_error = primary.task->getError();
    if (held_error.size() == 3) {
      result.position_error = held_error.head<3>().norm();
      result.orientation_error = 0.0;
    } else if (held_error.size() >= 6) {
      result.position_error = held_error.head<3>().norm();
      result.orientation_error = held_error.tail<3>().norm();
    }
    if (primary.kind == PoseTaskKind::kFrame) {
      result.achieved_pose = robot_->get_frame_pose(
          static_cast<FrameTask *>(primary.task.get())->getFrameName());
    } else {
      result.achieved_pose = targets.front().target_pose;
    }
    last_solution_dq_norm_ = 0.0;
  }
  if (acceleration_limits_enabled_ &&
      result.joint_velocities.size() == robot_->nv()) {
    previous_dq_ = result.joint_velocities;
  }

  // In teleop-style loops (max_steps=1), stall handling must accumulate across
  // successive solve_position_step calls based on *applied* motion, not only
  // raw QP status.  This post-step update prevents success/no-progress
  // oscillations from resetting the consecutive stall counter.
  if (stall_handler_enabled()) {
    if (held_non_improving_step) {
      stall_state_.consecutive_stall_steps = 0;
    } else {
      const double applied_step_norm = (result.q_solution - current_q).norm();
      const double applied_step_eps =
          stall_config_.dq_stall_eps * std::max(step_dt, 1e-9);
      if (applied_step_norm < applied_step_eps) {
        stall_state_.consecutive_stall_steps++;
        stall_state_.total_stall_steps++;
      } else {
        stall_state_.consecutive_stall_steps = 0;
      }
    }
  }

  return result;
}

PositionIKResult KinematicsSolver::solve_position_in_tcp(
    const Eigen::VectorXd &seed_q, const Eigen::Matrix4d &relative_target,
    const std::string &frame_name, const PositionIKOptions &options) {

  // Set robot to seed configuration
  robot_->update_configuration(seed_q);

  // Get current TCP pose
  Eigen::Matrix4d current_tcp_pose = robot_->get_frame_pose(frame_name);

  // Calculate target in base frame
  Eigen::Matrix4d target_pose = current_tcp_pose * relative_target;

  // Use regular position IK solver
  return solve_position(seed_q, target_pose, frame_name, options);
}


} // namespace embodik
