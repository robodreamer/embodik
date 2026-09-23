#include "acceleration_solver_internal.hpp"

namespace embodik {
namespace acceleration_solver_internal {

void apply_velocity_collision_lift_diagnostics(
    AccelerationSolverResult *result,
    const VelocityCollisionLiftDiagnostics &diagnostics) {
  result->velocity_collision_lift_applied = diagnostics.applied;
  result->collision_endpoint_validated = diagnostics.endpoint_validated;
  result->collision_step_certified = false;
  result->collision_validation_samples = diagnostics.validation_samples;
  result->collision_validation_allowed_pairs =
      diagnostics.validation_allowed_pairs;
  result->collision_validation_pairs_checked =
      diagnostics.validation_pairs_checked;
  result->collision_validation_exact_queries =
      diagnostics.validation_exact_queries;
  result->collision_validation_initial_exact_queries =
      diagnostics.validation_initial_exact_queries;
  result->collision_validation_sample_exact_queries =
      diagnostics.validation_sample_exact_queries;
  result->collision_validation_conservative_checks =
      diagnostics.validation_conservative_checks;
  result->collision_validation_conservative_certified_pairs =
      diagnostics.validation_conservative_certified_pairs;
  result->collision_validation_kinematics_updates =
      diagnostics.validation_kinematics_updates;
  result->collision_validation_geometry_updates =
      diagnostics.validation_geometry_updates;
  result->collision_validation_initial_certificate_reused =
      diagnostics.validation_initial_certificate_reused;
  result->collision_lift_pairs_considered = diagnostics.pairs_considered;
  result->collision_lift_row_pairs = diagnostics.row_pairs;
  result->collision_lift_row_exact_queries = diagnostics.row_exact_queries;
}

AccelerationSolverResult velocity_collision_lift_failure(
    SolverStatus status, const std::string &message,
    const VelocityCollisionLiftDiagnostics &diagnostics) {
  auto result = clear_outputs(failure(status, message));
  apply_velocity_collision_lift_diagnostics(&result, diagnostics);
  return result;
}

} // namespace acceleration_solver_internal

using namespace acceleration_solver_internal;

AccelerationSolverResult AccelerationSolver::solve_with_velocity_collision(
    KinematicsSolver &collision_solver, const Eigen::VectorXd &q,
    const Eigen::VectorXd &dq, double dt,
    const AccelerationSolveOptions &options,
    const VelocityCollisionLiftOptions &lift_options) {
  const auto start = std::chrono::steady_clock::now();
  const auto finish = [&start](AccelerationSolverResult result) {
    const auto end = std::chrono::steady_clock::now();
    result.computation_time_ms =
        std::chrono::duration<double, std::milli>(end - start).count();
    return result;
  };

  VelocityCollisionLiftDiagnostics diagnostics;
  if (lift_options.validation_substeps < 1 ||
      lift_options.validation_substeps > 1024) {
    return finish(velocity_collision_lift_failure(
        SolverStatus::kInvalidInput,
        "velocity collision lift validation_substeps must be in [1, 1024]",
        diagnostics));
  }
  if (collision_solver.robot().get() != robot_.get()) {
    return finish(velocity_collision_lift_failure(
        SolverStatus::kInvalidInput,
        "velocity collision lift requires a KinematicsSolver sharing the "
        "same RobotModel",
        diagnostics));
  }
  if (q.size() != robot_->nq() || dq.size() != robot_->nv()) {
    return finish(velocity_collision_lift_failure(
        SolverStatus::kShapeMismatch, "q and dq must match robot nq/nv",
        diagnostics));
  }
  if (!q.allFinite() || !dq.allFinite() || !std::isfinite(dt)) {
    return finish(velocity_collision_lift_failure(
        SolverStatus::kNonFiniteInput, "q, dq, and dt must be finite",
        diagnostics));
  }
  if (dt <= 0.0) {
    return finish(velocity_collision_lift_failure(
        SolverStatus::kInvalidInput, "dt must be positive", diagnostics));
  }
  detail::VelocityCollisionConstraintProvider collision_provider(
      collision_solver);
  if (!collision_provider.has_enabled_collision_constraint()) {
    return finish(velocity_collision_lift_failure(
        SolverStatus::kInvalidInput,
        "velocity collision lift requires a configured collision constraint",
        diagnostics));
  }
  diagnostics.applied = true;

  try {
    robot_->update_kinematics(q, dq);
  } catch (const std::exception &error) {
    return finish(velocity_collision_lift_failure(
        SolverStatus::kNumericalError,
        std::string("velocity collision lift kinematics update failed: ") +
            error.what(),
        diagnostics));
  }

  std::optional<detail::VelocityCollisionConstraintLinearization> linearization;
  try {
    linearization = collision_provider.linearize(dt);
  } catch (const std::invalid_argument &error) {
    return finish(velocity_collision_lift_failure(
        SolverStatus::kInvalidInput,
        std::string("velocity collision lift row construction failed: ") +
            error.what(),
        diagnostics));
  } catch (const std::exception &error) {
    return finish(velocity_collision_lift_failure(
        SolverStatus::kNumericalError,
        std::string("velocity collision lift row construction failed: ") +
            error.what(),
        diagnostics));
  }
  const auto row_accounting = collision_provider.accounting_snapshot();
  diagnostics.pairs_considered = row_accounting.pairs_considered;
  diagnostics.row_exact_queries = row_accounting.exact_distance_queries;
  AccelerationSolveOptions lifted_options = options;
  if (linearization.has_value()) {
    if (linearization->coefficient_matrix.rows() == 0 ||
        linearization->coefficient_matrix.cols() != robot_->nv() ||
        linearization->lower_bounds.size() !=
            linearization->coefficient_matrix.rows() ||
        linearization->upper_bounds.size() !=
            linearization->coefficient_matrix.rows() ||
        !linearization->coefficient_matrix.allFinite() ||
        !linearization->lower_bounds.allFinite() ||
        !linearization->upper_bounds.allFinite()) {
      return finish(velocity_collision_lift_failure(
          SolverStatus::kNumericalError,
          "velocity collision lift row construction returned invalid rows",
          diagnostics));
    }
    FrozenNextVelocityConstraint frozen;
    frozen.source_id = "velocity_collision_lift";
    frozen.coefficient_matrix = linearization->coefficient_matrix;
    frozen.affine_bias =
        Eigen::VectorXd::Zero(linearization->coefficient_matrix.rows());
    frozen.lower_bounds = linearization->lower_bounds;
    frozen.upper_bounds = linearization->upper_bounds;
    diagnostics.row_pairs = static_cast<std::uint64_t>(
        linearization->coefficient_matrix.rows());
    lifted_options.frozen_next_velocity_constraints.push_back(
        std::move(frozen));
  }

  auto result = solve(q, dq, dt, lifted_options);
  apply_velocity_collision_lift_diagnostics(&result, diagnostics);
  if (result.status != SolverStatus::kSuccess) {
    return finish(std::move(result));
  }

  std::vector<Eigen::VectorXd> validation_samples;
  validation_samples.reserve(
      static_cast<std::size_t>(lift_options.validation_substeps));
  for (int step = 1; step <= lift_options.validation_substeps; ++step) {
    const double sample_time =
        dt * static_cast<double>(step) /
        static_cast<double>(lift_options.validation_substeps);
    try {
      auto sample_q = robot_->integrate(
          q, sample_time * dq +
                 0.5 * sample_time * sample_time *
                     result.joint_accelerations);
      if (!sample_q.allFinite()) {
        diagnostics.validation_samples =
            static_cast<std::uint64_t>(step);
        return finish(velocity_collision_lift_failure(
            SolverStatus::kNumericalError,
            "velocity collision lift generated a non-finite sample",
            diagnostics));
      }
      validation_samples.push_back(std::move(sample_q));
    } catch (const std::exception &error) {
      diagnostics.validation_samples =
          static_cast<std::uint64_t>(step);
      return finish(velocity_collision_lift_failure(
          SolverStatus::kNumericalError,
          std::string("velocity collision lift sample integration failed: ") +
              error.what(),
          diagnostics));
    }
  }

  const auto validation =
      collision_provider.validate_samples(q, validation_samples);
  diagnostics.validation_samples = validation.samples_checked;
  diagnostics.validation_allowed_pairs = validation.allowed_pair_count;
  diagnostics.validation_pairs_checked = validation.pairs_checked;
  diagnostics.validation_exact_queries =
      validation.exact_distance_queries;
  diagnostics.validation_initial_exact_queries =
      validation.initial_exact_distance_queries;
  diagnostics.validation_sample_exact_queries =
      validation.sample_exact_distance_queries;
  diagnostics.validation_conservative_checks =
      validation.conservative_bound_checks;
  diagnostics.validation_conservative_certified_pairs =
      validation.conservative_bound_certified_pairs;
  diagnostics.validation_kinematics_updates = validation.kinematics_updates;
  diagnostics.validation_geometry_updates = validation.geometry_updates;
  diagnostics.validation_initial_certificate_reused =
      validation.initial_state_certificate_reused;
  if (validation.status != SolverStatus::kSuccess ||
      !validation.acceptable) {
    auto failed =
        clear_outputs(failure(validation.status, validation.message));
    apply_velocity_collision_lift_diagnostics(&failed, diagnostics);
    return finish(std::move(failed));
  }
  diagnostics.endpoint_validated = true;
  apply_velocity_collision_lift_diagnostics(&result, diagnostics);
  return finish(std::move(result));
}

} // namespace embodik
