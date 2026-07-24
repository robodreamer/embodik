#include "acceleration_solver_internal.hpp"

namespace embodik {
using namespace acceleration_solver_internal;

AccelerationSolverResult
AccelerationSolver::solve(const Eigen::VectorXd &q, const Eigen::VectorXd &dq,
                          double dt,
                          const AccelerationSolveOptions &options) {
  const auto start = std::chrono::steady_clock::now();
  auto backend_start = start;
  auto backend_end = start;
  bool backend_completed = false;
  const auto finish = [&start, &backend_start, &backend_end,
                       &backend_completed](AccelerationSolverResult result) {
    const auto end = std::chrono::steady_clock::now();
    result.computation_time_ms =
        std::chrono::duration<double, std::milli>(end - start).count();
    if (backend_completed) {
      result.preprocessing_time_ms =
          std::chrono::duration<double, std::milli>(backend_start - start)
              .count();
      result.postprocessing_time_ms =
          std::chrono::duration<double, std::milli>(end - backend_end).count();
    }
    return result;
  };

  if (q.size() != robot_->nq() || dq.size() != robot_->nv()) {
    return finish(failure(SolverStatus::kShapeMismatch,
                          "q and dq must match robot nq/nv"));
  }
  if (!q.allFinite() || !dq.allFinite() || !std::isfinite(dt)) {
    return finish(failure(SolverStatus::kNonFiniteInput,
                          "q, dq, and dt must be finite"));
  }
  if (dt <= 0.0) {
    return finish(failure(SolverStatus::kInvalidInput, "dt must be positive"));
  }

  SolverStatus limit_status = SolverStatus::kSuccess;
  std::string limit_message;
  Eigen::VectorXd accel_limits =
      resolve_acceleration_limits(*robot_, options, &limit_status,
                                  &limit_message);
  if (limit_status != SolverStatus::kSuccess) {
    return finish(failure(limit_status, limit_message));
  }
  if (accel_limits.size() != robot_->nv() ||
      !finite_positive_vector(accel_limits)) {
    return finish(failure(
        SolverStatus::kInvalidInput,
        "acceleration limits must have size nv and be finite and positive"));
  }

  try {
    robot_->update_kinematics(q, dq);
  } catch (const std::exception &error) {
    return finish(failure(
        SolverStatus::kNumericalError,
        std::string("acceleration kinematics update failed: ") + error.what()));
  }

  const auto state_box =
      build_joint_state_box(*robot_, q, dq, dt, accel_limits, options);
  if (state_box.status != SolverStatus::kSuccess) {
    return finish(failure(state_box.status, state_box.message));
  }
  std::optional<detail::CompiledAnalyticCollisionConstraint>
      native_collision_compiled;
  std::unique_ptr<detail::CollisionDifferentialScratch>
      native_collision_scratch;
  detail::PreparedAnalyticCollisionConstraint native_collision_prepared;
  if (has_collision_constraint()) {
    std::string unsupported_message;
    if (!native_collision_options_are_supported(options,
                                                &unsupported_message)) {
      auto failed =
          failure(SolverStatus::kInvalidInput, unsupported_message);
      failed.native_collision_constraint_applied = true;
      return finish(clear_outputs(std::move(failed)));
    }
    native_collision_compiled = make_stored_collision_compilation(
        native_collision_pair_indices_, native_collision_minimum_distances_,
        native_collision_active_pairs_);
    if (!native_collision_compiled->satisfied()) {
      auto failed = failure(
          SolverStatus::kNumericalError,
          "stored native collision configuration is inconsistent");
      failed.native_collision_constraint_applied = true;
      return finish(clear_outputs(std::move(failed)));
    }
    native_collision_scratch =
        std::make_unique<detail::CollisionDifferentialScratch>(*robot_);
    native_collision_prepared =
        detail::prepare_analytic_collision_constraint(
            *robot_, *native_collision_scratch, *native_collision_compiled,
            *native_collision_policy_, q, dq, dt, state_box.lower,
            state_box.upper, "current-state");
    if (!native_collision_prepared.satisfied()) {
      auto failed = failure(native_collision_prepared.status,
                            native_collision_prepared.message);
      failed.native_collision_constraint_applied = true;
      failed.native_collision_diagnostics =
          native_collision_prepared.diagnostics;
      failed.native_collision_pair_evaluations =
          native_collision_prepared.pair_evaluations;
      return finish(clear_outputs(std::move(failed)));
    }
  }
  std::optional<detail::PreparedAccelerationAllocationTransform> allocation;
  if (options.generalized_acceleration_allocation.has_value()) {
    const auto &requested_allocation =
        options.generalized_acceleration_allocation.value();
    auto prepared = detail::prepare_acceleration_allocation_transform(
        requested_allocation.metric_diagonal,
        requested_allocation.reference_acceleration, robot_->nv());
    if (!prepared.satisfied()) {
      return finish(failure(prepared.status, prepared.message));
    }
    allocation = std::move(prepared);
  }
  std::optional<PreparedEffortConstraint> effort_constraint;
  if (options.effort_constraints.has_value()) {
    auto prepared_effort = prepare_effort_constraint(
        *robot_, q, dq, options.effort_constraints.value());
    if (prepared_effort.status != SolverStatus::kSuccess) {
      return finish(
          failure(prepared_effort.status, prepared_effort.message));
    }
    effort_constraint = std::move(prepared_effort);
  }

  std::unordered_set<std::string> constraint_source_ids;
  auto constraint_validation = validate_acceleration_constraints(
      options.affine_constraints, options.frozen_next_velocity_constraints,
      robot_->nv(), &constraint_source_ids);
  if (constraint_validation.status != SolverStatus::kSuccess) {
    return finish(
        failure(constraint_validation.status, constraint_validation.message));
  }
  const auto frozen_transform = transform_frozen_constraints(
      options.frozen_next_velocity_constraints, dq, dt);
  if (frozen_transform.status != SolverStatus::kSuccess) {
    return finish(
        failure(frozen_transform.status, frozen_transform.message));
  }
  const auto lock_rows = build_lock_rows(options, dq, dt);
  if (lock_rows.status != SolverStatus::kSuccess) {
    return finish(failure(lock_rows.status, lock_rows.message));
  }

  const bool task_exclusions_require_physical_differentials =
      std::any_of(tasks_.begin(), tasks_.end(), [](const auto &task) {
        return task && task->isActive() &&
               !task->get_excluded_joint_indices().empty();
      });
  const bool retain_task_differentials =
      options.collect_task_diagnostics ||
      !options.task_acceleration_bounds.empty() ||
      task_exclusions_require_physical_differentials;
  auto &objectives = workspace_->objectives;
  try {
    assemble_objectives(&objectives, tasks_, ordered_task_scratch_,
                        task_references_, *robot_, dq,
                        retain_task_differentials);
  } catch (const std::exception &error) {
    return finish(failure(
        SolverStatus::kNumericalError,
        std::string("acceleration task assembly failed: ") + error.what()));
  }
  if (objectives.status != SolverStatus::kSuccess) {
    return finish(failure(objectives.status, objectives.message));
  }

  const auto task_bound_validation = validate_task_acceleration_bounds(
      options.task_acceleration_bounds, objectives, &constraint_source_ids);
  if (task_bound_validation.status != SolverStatus::kSuccess) {
    return finish(failure(task_bound_validation.status,
                          task_bound_validation.message));
  }

  const auto task_bound_constraints = make_task_bound_affine_constraints(
      options.task_acceleration_bounds, objectives);
  const auto contact_constraints = make_contact_acceleration_constraints(
      *robot_, options.contact_acceleration_constraints, &constraint_source_ids);
  if (contact_constraints.status != SolverStatus::kSuccess) {
    return finish(
        failure(contact_constraints.status, contact_constraints.message));
  }
  const auto tight_point_constraints = make_tight_point_constraints(
      *robot_, options.tight_point_constraints, dt, state_box.lower,
      state_box.upper, &constraint_source_ids);
  if (tight_point_constraints.status != SolverStatus::kSuccess) {
    return finish(failure(tight_point_constraints.status,
                          tight_point_constraints.message));
  }
  const auto tight_frame_pose_constraints = make_fixed_frame_pose_constraints(
      *robot_, options.tight_frame_pose_constraints, dt, state_box.lower,
      state_box.upper, &constraint_source_ids,
      [](const TightFramePoseAccelerationConstraint &constraint,
         detail::FixedFramePoseConstraintRecord *record) {
        *record = detail::make_tight_frame_pose_record(constraint);
        return detail::FixedFramePoseConstraintResult{};
      });
  if (tight_frame_pose_constraints.status != SolverStatus::kSuccess) {
    return finish(failure(tight_frame_pose_constraints.status,
                          tight_frame_pose_constraints.message));
  }
  const auto torso_pose_bound_constraints = make_fixed_frame_pose_constraints(
      *robot_, options.torso_pose_bound_constraints, dt, state_box.lower,
      state_box.upper, &constraint_source_ids,
      [](const TorsoPoseBoundAccelerationConstraint &constraint,
         detail::FixedFramePoseConstraintRecord *record) {
        return detail::make_torso_pose_bound_record(constraint, record);
      });
  if (torso_pose_bound_constraints.status != SolverStatus::kSuccess) {
    return finish(failure(torso_pose_bound_constraints.status,
                          torso_pose_bound_constraints.message));
  }
  const auto relative_pose_constraints = make_relative_pose_constraints(
      *robot_, options.relative_pose_constraints, dt, state_box.lower,
      state_box.upper, &constraint_source_ids);
  if (relative_pose_constraints.status != SolverStatus::kSuccess) {
    return finish(failure(relative_pose_constraints.status,
                          relative_pose_constraints.message));
  }
  const auto com_support_polygon_constraints =
      make_com_support_polygon_constraints(
          *robot_, options.com_support_polygon_constraints, dt,
          state_box.lower, state_box.upper, &constraint_source_ids);
  if (com_support_polygon_constraints.status != SolverStatus::kSuccess) {
    return finish(failure(com_support_polygon_constraints.status,
                          com_support_polygon_constraints.message));
  }
  const bool state_box_task_fallback_applied =
      options.allow_state_box_task_fallback &&
      zero_excluded_by_bounds(state_box.lower, state_box.upper);
  if (state_box_task_fallback_applied) {
    for (auto &config : objectives.configs) {
      config.allow_min_error_fallback = true;
    }
  }

  const Eigen::Index extra_hard_rows =
      constraint_validation.row_count + task_bound_validation.row_count +
      contact_constraints.row_count + tight_point_constraints.row_count +
      tight_frame_pose_constraints.row_count +
      torso_pose_bound_constraints.row_count +
      relative_pose_constraints.row_count +
      com_support_polygon_constraints.row_count +
      lock_rows.coefficients.rows() +
      native_collision_prepared.physical_constraint.coefficient_matrix.rows() +
      (effort_constraint.has_value() ? robot_->nv() : 0);
  const bool state_box_only = extra_hard_rows == 0;
  auto &state_box_matrix = workspace_->state_box_identity;
  std::optional<detail::GeneralizedConstraintSet> constraints;
  if (state_box_only) {
    if (state_box_matrix.rows() != robot_->nv() ||
        state_box_matrix.cols() != robot_->nv()) {
      state_box_matrix.setIdentity(robot_->nv(), robot_->nv());
    }
  } else {
    constraints.emplace(robot_->nv(), robot_->nv() + extra_hard_rows, false);
    detail::GeneralizedConstraintBlock joint_box;
    joint_box.coefficient_matrix =
        Eigen::MatrixXd::Identity(robot_->nv(), robot_->nv());
    joint_box.affine_bias = Eigen::VectorXd::Zero(robot_->nv());
    joint_box.physical_lower_bounds = state_box.lower;
    joint_box.physical_upper_bounds = state_box.upper;
    if (!constraints->append_block(std::move(joint_box))) {
      return finish(failure(SolverStatus::kNumericalError,
                            "failed to assemble acceleration constraints"));
    }
  }
  if (native_collision_prepared.has_rows() &&
      !constraints->append_block(make_affine_constraint_block(
          {native_collision_prepared.physical_constraint}, robot_->nv(),
          native_collision_prepared.physical_constraint.coefficient_matrix
              .rows()))) {
    return finish(failure(
        SolverStatus::kNumericalError,
        "failed to assemble native collision acceleration constraints"));
  }
  if (effort_constraint.has_value() &&
      !constraints->append_block(
          make_effort_constraint_block(*effort_constraint, robot_->nv()))) {
    return finish(failure(SolverStatus::kNumericalError,
                          "failed to assemble effort constraints"));
  }
  if (!options.affine_constraints.empty() &&
      !constraints->append_block(make_affine_constraint_block(
          options.affine_constraints, robot_->nv(),
          count_constraint_rows(options.affine_constraints)))) {
    return finish(failure(
        SolverStatus::kNumericalError,
        "failed to assemble affine acceleration constraints"));
  }
  if (frozen_transform.row_count > 0 &&
      !constraints->append_block(make_affine_constraint_block(
          frozen_transform.transformed_constraints, robot_->nv(),
          frozen_transform.row_count))) {
    return finish(failure(
        SolverStatus::kNumericalError,
        "failed to assemble frozen next-velocity constraints"));
  }
  if (lock_rows.bounds.size() > 0) {
    if (!constraints->append_block(
            make_lock_constraint_block(lock_rows, robot_->nv()))) {
      return finish(failure(SolverStatus::kNumericalError,
                            "failed to assemble acceleration lock "
                            "constraints"));
    }
  }
  if (task_bound_validation.row_count > 0 &&
      !constraints->append_block(make_affine_constraint_block(
          task_bound_constraints, robot_->nv(),
          task_bound_validation.row_count))) {
    return finish(failure(
        SolverStatus::kNumericalError,
        "failed to assemble task acceleration bounds"));
  }
  if (contact_constraints.row_count > 0 &&
      !constraints->append_block(make_affine_constraint_block(
          contact_constraints.constraints, robot_->nv(),
          contact_constraints.row_count))) {
    return finish(failure(
        SolverStatus::kNumericalError,
        "failed to assemble contact acceleration constraints"));
  }
  if (tight_point_constraints.row_count > 0 &&
      !constraints->append_block(make_affine_constraint_block(
          tight_point_constraints.constraints, robot_->nv(),
          tight_point_constraints.row_count))) {
    return finish(failure(
        SolverStatus::kNumericalError,
        "failed to assemble tight point acceleration constraints"));
  }
  if (tight_frame_pose_constraints.row_count > 0 &&
      !constraints->append_block(make_affine_constraint_block(
          tight_frame_pose_constraints.constraints, robot_->nv(),
          tight_frame_pose_constraints.row_count))) {
    return finish(failure(
        SolverStatus::kNumericalError,
        "failed to assemble tight frame pose acceleration constraints"));
  }
  if (torso_pose_bound_constraints.row_count > 0 &&
      !constraints->append_block(make_affine_constraint_block(
          torso_pose_bound_constraints.constraints, robot_->nv(),
          torso_pose_bound_constraints.row_count))) {
    return finish(failure(
        SolverStatus::kNumericalError,
        "failed to assemble torso pose bound acceleration constraints"));
  }
  if (relative_pose_constraints.row_count > 0 &&
      !constraints->append_block(make_affine_constraint_block(
          relative_pose_constraints.constraints, robot_->nv(),
          relative_pose_constraints.row_count))) {
    return finish(failure(
        SolverStatus::kNumericalError,
        "failed to assemble relative pose acceleration constraints"));
  }
  if (com_support_polygon_constraints.row_count > 0 &&
      !constraints->append_block(make_affine_constraint_block(
          com_support_polygon_constraints.constraints, robot_->nv(),
          com_support_polygon_constraints.row_count))) {
    return finish(failure(
        SolverStatus::kNumericalError,
        "failed to assemble CoM support-polygon acceleration constraints"));
  }
  if (!state_box_only && !constraints->finalize()) {
    return finish(failure(SolverStatus::kNumericalError,
                          "failed to assemble acceleration constraints"));
  }

  const std::vector<Eigen::VectorXd> *backend_targets = &objectives.targets;
  const std::vector<Eigen::VectorXd> *backend_biases = &objectives.biases;
  const std::vector<Eigen::MatrixXd> *backend_matrices = &objectives.matrices;
  std::vector<Eigen::VectorXd> transformed_targets;
  std::vector<Eigen::VectorXd> transformed_biases;
  std::vector<Eigen::MatrixXd> transformed_matrices;
  const Eigen::MatrixXd *backend_constraint_matrix =
      state_box_only ? &state_box_matrix : &constraints->coefficient_matrix();
  const Eigen::VectorXd *backend_lower_bounds =
      state_box_only ? &state_box.lower : &constraints->lower_bounds();
  const Eigen::VectorXd *backend_upper_bounds =
      state_box_only ? &state_box.upper : &constraints->upper_bounds();
  Eigen::MatrixXd transformed_constraint_matrix;
  Eigen::VectorXd transformed_lower_bounds;
  Eigen::VectorXd transformed_upper_bounds;
  if (allocation.has_value()) {
    transformed_targets = objectives.targets;
    transformed_biases = objectives.biases;
    transformed_matrices = objectives.matrices;
    for (std::size_t objective_index = 0;
         objective_index < transformed_matrices.size(); ++objective_index) {
      if (objectives.synthetic_hold_objective && objective_index == 0U) {
        transformed_matrices[objective_index] =
            Eigen::MatrixXd::Identity(robot_->nv(), robot_->nv());
        transformed_biases[objective_index] =
            Eigen::VectorXd::Zero(robot_->nv());
        transformed_targets[objective_index] =
            Eigen::VectorXd::Zero(robot_->nv());
        continue;
      }
      const auto transformed_matrix =
          allocation->transform_objective_matrix(
              objectives.matrices[objective_index]);
      if (!transformed_matrix.satisfied()) {
        return finish(
            failure(transformed_matrix.status, transformed_matrix.message));
      }
      const auto transformed_bias =
          allocation->transform_objective_bias(
              objectives.matrices[objective_index],
              objectives.biases[objective_index]);
      if (!transformed_bias.satisfied()) {
        return finish(
            failure(transformed_bias.status, transformed_bias.message));
      }
      transformed_matrices[objective_index] =
          std::move(transformed_matrix.matrix);
      transformed_biases[objective_index] = std::move(transformed_bias.vector);
    }
    backend_targets = &transformed_targets;
    backend_biases = &transformed_biases;
    backend_matrices = &transformed_matrices;
    const auto transformed_hard = allocation->transform_backend_hard_rows(
        *backend_constraint_matrix, *backend_lower_bounds,
        *backend_upper_bounds);
    if (!transformed_hard.satisfied()) {
      return finish(failure(transformed_hard.status, transformed_hard.message));
    }
    transformed_constraint_matrix =
        std::move(transformed_hard.coefficient_matrix);
    transformed_lower_bounds = std::move(transformed_hard.lower_bounds);
    transformed_upper_bounds = std::move(transformed_hard.upper_bounds);
    backend_constraint_matrix = &transformed_constraint_matrix;
    backend_lower_bounds = &transformed_lower_bounds;
    backend_upper_bounds = &transformed_upper_bounds;
  }

  // The compatible state box has already proven one finite interval per
  // scalar joint. When it is the only hard set, its identity rows are already
  // normalized and Phase I would duplicate that feasibility proof.
  const bool prevalidated_state_box_only =
      !allocation.has_value() && state_box_only;
  backend_start = std::chrono::steady_clock::now();
  auto backend = detail::SolveGeneralizedHierarchicalLinearSystemEigen(
      *backend_targets, *backend_biases, *backend_matrices,
      *backend_constraint_matrix, *backend_lower_bounds, *backend_upper_bounds,
      acceleration_backend_config(), objectives.configs, nullptr,
      kConstraintTolerance, prevalidated_state_box_only,
      prevalidated_state_box_only);
  backend_end = std::chrono::steady_clock::now();
  backend_completed = true;

  AccelerationSolverResult result;
  result.backend_computation_time_ms = backend.computation_time_ms;
  static_cast<SolverResult &>(result) = std::move(backend);
  result.acceleration_limits_applied = true;
  result.state_box_task_fallback_applied =
      state_box_task_fallback_applied;
  result.effort_limits_applied = effort_constraint.has_value();
  result.native_collision_constraint_applied =
      native_collision_compiled.has_value();
  if (native_collision_compiled.has_value()) {
    result.native_collision_diagnostics =
        native_collision_prepared.diagnostics;
    result.native_collision_pair_evaluations =
        native_collision_prepared.pair_evaluations;
  }
  if (result.status != SolverStatus::kSuccess) {
    return finish(clear_outputs(std::move(result)));
  }
  const auto fail_after_backend =
      [&result, &native_collision_compiled](SolverStatus status,
                                            std::string message) {
        if (native_collision_compiled.has_value()) {
          result.status = status;
          result.status_message = std::move(message);
          return clear_outputs(std::move(result));
        }
        return clear_outputs(failure(status, std::move(message)));
      };

  Eigen::VectorXd backend_solution = vector_from_solution(result);
  if (allocation.has_value()) {
    const auto physical =
        allocation->reconstruct_physical_acceleration(backend_solution);
    if (!physical.satisfied()) {
      return finish(fail_after_backend(physical.status, physical.message));
    }
    result.joint_accelerations = physical.vector;
    assign_solution_from_vector(&result, result.joint_accelerations);
    if (!populate_allocation_diagnostics(&result, *allocation,
                                         result.joint_accelerations)) {
      return finish(fail_after_backend(
          SolverStatus::kNumericalError,
          "acceleration allocation diagnostics are not finite"));
    }
  } else {
    result.joint_accelerations = std::move(backend_solution);
  }
  if (result.joint_accelerations.size() != robot_->nv() ||
      !result.joint_accelerations.allFinite()) {
    return finish(fail_after_backend(
        SolverStatus::kNumericalError,
        "backend returned an invalid acceleration vector"));
  }
  if ((result.joint_accelerations.array() <
       state_box.lower.array() - kConstraintTolerance)
          .any() ||
      (result.joint_accelerations.array() >
       state_box.upper.array() + kConstraintTolerance)
          .any()) {
    return finish(fail_after_backend(
        SolverStatus::kNumericalError,
        "accepted acceleration violates the compatible state box"));
  }
  if (native_collision_prepared.has_rows() &&
      !accepted_affine_constraints_are_satisfied(
          {native_collision_prepared.physical_constraint},
          result.joint_accelerations)) {
    return finish(fail_after_backend(
        SolverStatus::kNumericalError,
        "accepted acceleration violates native collision constraints"));
  }
  if (!accepted_affine_constraints_are_satisfied(options.affine_constraints,
                                                 result.joint_accelerations)) {
    return finish(clear_outputs(failure(
        SolverStatus::kNumericalError,
        "accepted acceleration violates an affine acceleration constraint")));
  }
  if (!accepted_frozen_constraints_are_satisfied(
          options.frozen_next_velocity_constraints, dq, dt,
          result.joint_accelerations)) {
    return finish(clear_outputs(failure(
        SolverStatus::kNumericalError,
        "accepted acceleration violates a frozen next-velocity constraint")));
  }
  if (!accepted_lock_constraints_are_satisfied(options, dq, dt,
                                               result.joint_accelerations)) {
    return finish(clear_outputs(failure(
        SolverStatus::kNumericalError,
        "accepted acceleration violates an acceleration lock constraint")));
  }
  if (!accepted_affine_constraints_are_satisfied(task_bound_constraints,
                                                 result.joint_accelerations)) {
    return finish(clear_outputs(failure(
        SolverStatus::kNumericalError,
        "accepted acceleration violates task acceleration bounds")));
  }
  if (!accepted_affine_constraints_are_satisfied(
          contact_constraints.constraints, result.joint_accelerations)) {
    return finish(clear_outputs(failure(
        SolverStatus::kNumericalError,
        "accepted acceleration violates contact acceleration constraints")));
  }
  if (!accepted_affine_constraints_are_satisfied(
          tight_point_constraints.constraints, result.joint_accelerations)) {
    return finish(clear_outputs(failure(
        SolverStatus::kNumericalError,
        "accepted acceleration violates tight point acceleration constraints")));
  }
  if (!accepted_affine_constraints_are_satisfied(
          tight_frame_pose_constraints.constraints,
          result.joint_accelerations)) {
    return finish(clear_outputs(failure(
        SolverStatus::kNumericalError,
        "accepted acceleration violates tight frame pose acceleration "
        "constraints")));
  }
  if (!accepted_affine_constraints_are_satisfied(
          torso_pose_bound_constraints.constraints,
          result.joint_accelerations)) {
    return finish(clear_outputs(failure(
        SolverStatus::kNumericalError,
        "accepted acceleration violates torso pose bound acceleration "
        "constraints")));
  }
  if (!accepted_affine_constraints_are_satisfied(
          relative_pose_constraints.constraints, result.joint_accelerations)) {
    return finish(clear_outputs(failure(
        SolverStatus::kNumericalError,
        "accepted acceleration violates relative pose acceleration "
        "constraints")));
  }
  if (!accepted_affine_constraints_are_satisfied(
          com_support_polygon_constraints.constraints,
          result.joint_accelerations)) {
    return finish(clear_outputs(failure(
        SolverStatus::kNumericalError,
        "accepted acceleration violates CoM support-polygon acceleration "
        "constraints")));
  }
  if (effort_constraint.has_value()) {
    auto evaluated =
        evaluate_inverse_dynamics(*robot_, q, dq, result.joint_accelerations);
    if (evaluated.status != SolverStatus::kSuccess) {
      auto failed = failure(
          evaluated.status,
          "accepted effort evaluation failed: " + evaluated.message);
      failed.effort_limits_applied = true;
      return finish(clear_outputs(std::move(failed)));
    }
    result.predicted_torques = std::move(evaluated.torques);
    for (Eigen::Index index = 0; index < robot_->nv(); ++index) {
      const double tolerance =
          std::max(kConstraintTolerance,
                   1e-12 * std::max(
                                effort_constraint->mass_matrix.row(index)
                                    .stableNorm() *
                                    result.joint_accelerations.stableNorm(),
                                effort_constraint->limits(index)));
      if (std::abs(result.predicted_torques(index)) >
          effort_constraint->limits(index) + tolerance) {
        auto failed = failure(
            SolverStatus::kNumericalError,
            "accepted acceleration violates effort constraints");
        failed.effort_limits_applied = true;
        return finish(clear_outputs(std::move(failed)));
      }
      if (std::abs(std::abs(result.predicted_torques(index)) -
                   effort_constraint->limits(index)) <= tolerance) {
        append_unique(&result.saturated_effort_indices,
                      static_cast<int>(index));
      }
    }
  }
  result.joint_velocities_next = dq + dt * result.joint_accelerations;
  try {
    result.q_solution = robot_->integrate(
        q, dt * dq + 0.5 * dt * dt * result.joint_accelerations);
  } catch (const std::exception &error) {
    return finish(fail_after_backend(
        SolverStatus::kNumericalError,
        std::string("accepted acceleration integration failed: ") +
            error.what()));
  }
  if (!result.joint_velocities_next.allFinite() ||
      !result.q_solution.allFinite()) {
    return finish(fail_after_backend(
        SolverStatus::kNumericalError,
        "accepted acceleration produced non-finite next state"));
  }
  const bool has_predicted_geometric_acceptance =
      !tight_point_constraints.prepared_constraints.empty() ||
      !tight_frame_pose_constraints.prepared_constraints.empty() ||
      !torso_pose_bound_constraints.prepared_constraints.empty() ||
      !relative_pose_constraints.prepared_constraints.empty() ||
      !com_support_polygon_constraints.prepared_constraints.empty();
  std::optional<StateBoxAssembly> predicted_state_box;
  if (has_predicted_geometric_acceptance ||
      native_collision_compiled.has_value()) {
    predicted_state_box = build_joint_state_box(
        *robot_, result.q_solution, result.joint_velocities_next, dt,
        accel_limits, options);
    if (predicted_state_box->status != SolverStatus::kSuccess) {
      return finish(fail_after_backend(
          predicted_state_box->status,
          "accepted geometric constraints could not construct the predicted "
          "joint acceleration support box: " +
              predicted_state_box->message));
    }
  }
  if (has_predicted_geometric_acceptance) {
    for (const auto &prepared :
         tight_point_constraints.prepared_constraints) {
      const auto acceptance =
          detail::validate_tight_point_constraint_acceptance(
              prepared, *robot_, q, dq, result.joint_accelerations, dt,
              predicted_state_box->lower, predicted_state_box->upper);
      if (!acceptance.satisfied()) {
        return finish(clear_outputs(
            failure(acceptance.status, acceptance.message)));
      }
    }
    for (const auto &prepared :
         tight_frame_pose_constraints.prepared_constraints) {
      const auto acceptance =
          detail::validate_fixed_frame_pose_constraint_acceptance(
              prepared, *robot_, q, dq, result.joint_accelerations, dt,
              predicted_state_box->lower, predicted_state_box->upper);
      if (!acceptance.satisfied()) {
        return finish(clear_outputs(
            failure(acceptance.status, acceptance.message)));
      }
    }
    for (const auto &prepared :
         torso_pose_bound_constraints.prepared_constraints) {
      const auto acceptance =
          detail::validate_fixed_frame_pose_constraint_acceptance(
              prepared, *robot_, q, dq, result.joint_accelerations, dt,
              predicted_state_box->lower, predicted_state_box->upper);
      if (!acceptance.satisfied()) {
        return finish(clear_outputs(
            failure(acceptance.status, acceptance.message)));
      }
    }
    for (const auto &prepared :
         relative_pose_constraints.prepared_constraints) {
      const auto acceptance =
          detail::validate_relative_pose_constraint_acceptance(
              prepared, *robot_, q, dq, result.joint_accelerations, dt,
              predicted_state_box->lower, predicted_state_box->upper);
      if (!acceptance.satisfied()) {
        return finish(clear_outputs(
            failure(acceptance.status, acceptance.message)));
      }
    }
    for (const auto &prepared :
         com_support_polygon_constraints.prepared_constraints) {
      const auto acceptance =
          detail::validate_com_support_polygon_constraint_acceptance(
              prepared, *robot_, q, dq, result.joint_accelerations, dt,
              predicted_state_box->lower, predicted_state_box->upper);
      if (!acceptance.satisfied()) {
        return finish(clear_outputs(
            failure(acceptance.status, acceptance.message)));
      }
    }
  }
  if (native_collision_compiled.has_value()) {
    const auto certificate = detail::certify_analytic_collision_step(
        *robot_, *native_collision_scratch, *native_collision_compiled,
        *native_collision_policy_, native_collision_prepared, q, dq,
        result.joint_accelerations, dt, result.q_solution,
        result.joint_velocities_next, predicted_state_box->lower,
        predicted_state_box->upper);
    result.native_collision_pair_evaluations +=
        certificate.pair_evaluations;
    result.native_collision_path_visited_nodes =
        certificate.path_visited_nodes;
    result.native_collision_certified_intervals =
        certificate.certified_intervals;
    result.collision_endpoint_validated =
        certificate.endpoint_validated;
    result.collision_step_certified = certificate.step_certified;
    if (!certificate.diagnostics.empty()) {
      result.native_collision_diagnostics = certificate.diagnostics;
    }
    if (!certificate.satisfied()) {
      result.status = certificate.status;
      result.status_message = certificate.message;
      return finish(clear_outputs(std::move(result)));
    }
  }
  if (options.apply_velocity_limits) {
    const Eigen::VectorXd velocity_limits = robot_->get_velocity_limits();
    if ((result.joint_velocities_next.array() <
         -velocity_limits.array() - kConstraintTolerance)
            .any() ||
        (result.joint_velocities_next.array() >
         velocity_limits.array() + kConstraintTolerance)
            .any()) {
      return finish(fail_after_backend(
          SolverStatus::kNumericalError,
          "accepted acceleration violates next velocity limits"));
    }
  }
  if (options.apply_position_limits) {
    const auto position_limits = robot_->get_joint_limits();
    if ((result.q_solution.array() <
         position_limits.first.array() - kConstraintTolerance)
            .any() ||
        (result.q_solution.array() >
         position_limits.second.array() + kConstraintTolerance)
            .any()) {
      return finish(fail_after_backend(
          SolverStatus::kNumericalError,
          "accepted acceleration violates next position limits"));
    }
  }
  attribute_saturation(&result, state_box, result.joint_accelerations);

  const auto objective_scales = result.task_scales;
  const auto objective_modes = result.task_modes_effective;
  const auto objective_fallbacks = result.task_used_fallback;
  result.task_scales.clear();
  result.task_errors.clear();
  result.task_modes_effective.clear();
  result.task_used_fallback.clear();
  result.task_scales.reserve(tasks_.size());
  result.task_errors.reserve(tasks_.size());
  result.task_modes_effective.reserve(tasks_.size());
  result.task_used_fallback.reserve(tasks_.size());
  if (options.collect_task_diagnostics) {
    result.task_diagnostics.reserve(tasks_.size());
  }
  double squared_error = 0.0;
  for (std::size_t group_index = 0; group_index < objectives.groups.size();
       ++group_index) {
    Eigen::Index task_row_cursor = 0;
    for (std::size_t task_index = 0;
         task_index < objectives.groups[group_index].size(); ++task_index) {
      const auto &task = objectives.groups[group_index][task_index];
      const Eigen::Index task_dimension = task->getDimension();
      const double task_scale = group_index < objective_scales.size()
                                    ? objective_scales[group_index]
                                    : 1.0;
      const TaskSolveMode task_mode = group_index < objective_modes.size()
                                          ? objective_modes[group_index]
                                          : task->getSolveMode();
      const bool task_fallback = group_index < objective_fallbacks.size()
                                     ? objective_fallbacks[group_index]
                                     : false;
      result.task_scales.push_back(task_scale);
      result.task_modes_effective.push_back(task_mode);
      result.task_used_fallback.push_back(task_fallback);

      Eigen::VectorXd residual;
      std::optional<AccelerationTaskDiagnostics> diagnostic;
      if (options.collect_task_diagnostics) {
        diagnostic.emplace();
        diagnostic->task_name = task->getName();
        diagnostic->scale = task_scale;
        diagnostic->effective_mode = task_mode;
        diagnostic->used_min_error_fallback = task_fallback;
        diagnostic->reference_acceleration =
            objectives.references[group_index][task_index];
        diagnostic->jacobian_bias =
            objectives.differentials[group_index][task_index].jacobian_bias;
        diagnostic->achieved_acceleration =
            objectives.differentials[group_index][task_index]
                .physical_jacobian *
            result.joint_accelerations;
        diagnostic->residual =
            diagnostic->achieved_acceleration + diagnostic->jacobian_bias -
            diagnostic->reference_acceleration;
        residual = diagnostic->residual;
      } else if (retain_task_differentials) {
        residual =
            objectives.differentials[group_index][task_index]
                    .physical_jacobian *
                result.joint_accelerations +
            objectives.differentials[group_index][task_index].jacobian_bias -
            objectives.references[group_index][task_index];
      } else {
        residual =
            objectives.matrices[group_index]
                    .middleRows(task_row_cursor, task_dimension) *
                result.joint_accelerations +
            objectives.biases[group_index].segment(task_row_cursor,
                                                   task_dimension) -
            objectives.targets[group_index].segment(task_row_cursor,
                                                    task_dimension);
      }
      const double task_error = residual.norm();
      result.task_errors.push_back(task_error);
      squared_error += residual.squaredNorm();
      task->setLastEffectiveMode(task_mode);
      task->setUsedMinErrorFallback(task_fallback);
      if (diagnostic.has_value()) {
        result.task_diagnostics.push_back(std::move(*diagnostic));
      }
      task_row_cursor += task_dimension;
    }
  }
  result.final_error = std::sqrt(squared_error);
  return finish(std::move(result));
}

} // namespace embodik
