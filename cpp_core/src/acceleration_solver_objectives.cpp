#include "acceleration_solver_internal.hpp"

namespace embodik {
namespace acceleration_solver_internal {

void assemble_objectives(
    ObjectiveAssembly *assembly,
    const std::vector<std::shared_ptr<Task>> &tasks,
    std::vector<std::shared_ptr<Task>> &ordered_tasks,
    const std::unordered_map<std::string, AccelerationTaskReference> &references,
    const RobotModel &robot, const Eigen::VectorXd &dq,
    bool retain_task_differentials) {
  assembly->status = SolverStatus::kSuccess;
  assembly->message.clear();
  assembly->synthetic_hold_objective = false;
  ordered_tasks.clear();
  ordered_tasks.reserve(tasks.size());
  for (const auto &task : tasks) {
    if (task && task->isActive()) {
      if (task->getSolveMode() == TaskSolveMode::kScaleElastic) {
        assembly->status = SolverStatus::kInvalidInput;
        assembly->message =
            "AccelerationSolver does not support SCALE_ELASTIC tasks";
        return;
      }
      ordered_tasks.push_back(task);
    }
  }
  if (ordered_tasks.empty()) {
    assembly->matrices.resize(1);
    assembly->targets.resize(1);
    assembly->biases.resize(1);
    assembly->configs.resize(1);
    assembly->groups.resize(1);
    assembly->differentials.resize(1);
    assembly->references.resize(1);
    assembly->matrices[0].setIdentity(robot.nv(), robot.nv());
    assembly->targets[0].setZero(robot.nv());
    assembly->biases[0].setZero(robot.nv());
    ObjectiveSolveConfig config;
    config.priority = 0;
    config.solve_mode = TaskSolveMode::kMinError;
    config.allow_min_error_fallback = true;
    assembly->configs[0] = config;
    assembly->groups[0].clear();
    assembly->differentials[0].clear();
    assembly->references[0].clear();
    assembly->synthetic_hold_objective = true;
    return;
  }

  const auto priority_less = [](const std::shared_ptr<Task> &lhs,
                                const std::shared_ptr<Task> &rhs) {
    if (!lhs) {
      return static_cast<bool>(rhs);
    }
    if (!rhs) {
      return false;
    }
    return lhs->getPriority() < rhs->getPriority();
  };
  if (!std::is_sorted(ordered_tasks.begin(), ordered_tasks.end(),
                      priority_less)) {
    std::stable_sort(ordered_tasks.begin(), ordered_tasks.end(), priority_less);
  }
  assembly->matrices.reserve(ordered_tasks.size());
  assembly->targets.reserve(ordered_tasks.size());
  assembly->biases.reserve(ordered_tasks.size());
  assembly->configs.reserve(ordered_tasks.size());
  assembly->groups.reserve(ordered_tasks.size());
  assembly->differentials.reserve(ordered_tasks.size());
  assembly->references.reserve(ordered_tasks.size());
  const bool control_only_zero_velocity =
      !retain_task_differentials && dq.isZero(0.0);
  std::size_t group_index = 0;
  for (std::size_t group_begin = 0; group_begin < ordered_tasks.size();) {
    const int priority = ordered_tasks[group_begin]->getPriority();
    std::size_t group_end = group_begin;
    while (group_end < ordered_tasks.size()) {
      if (ordered_tasks[group_end]->getPriority() != priority) {
        break;
      }
      ++group_end;
    }
    if (assembly->groups.size() <= group_index) {
      assembly->groups.emplace_back();
      assembly->matrices.emplace_back();
      assembly->targets.emplace_back();
      assembly->biases.emplace_back();
      assembly->configs.emplace_back();
      assembly->differentials.emplace_back();
      assembly->references.emplace_back();
    }
    auto &group = assembly->groups[group_index];
    group.assign(ordered_tasks.begin() + group_begin,
                 ordered_tasks.begin() + group_end);
    Eigen::Index rows = 0;
    for (const auto &task : group) {
      rows += task->getDimension();
    }
    auto &matrix = assembly->matrices[group_index];
    auto &target = assembly->targets[group_index];
    auto &bias = assembly->biases[group_index];
    matrix.resize(rows, robot.nv());
    target.resize(rows);
    bias.resize(rows);
    auto &group_differentials = assembly->differentials[group_index];
    auto &group_references = assembly->references[group_index];
    group_differentials.clear();
    group_references.clear();
    group_differentials.reserve(group.size());
    group_references.reserve(group.size());

    const TaskSolveMode mode = group.front()->getSolveMode();
    const bool allow_fallback = group.front()->getAllowMinErrorFallback();
    Eigen::Index cursor = 0;
    for (const auto &task : group) {
      if (task->getSolveMode() != mode ||
          task->getAllowMinErrorFallback() != allow_fallback) {
        assembly->status = SolverStatus::kInvalidInput;
        assembly->message =
            "same-priority acceleration tasks must share solve mode and "
            "fallback settings";
        return;
      }
      task->update(robot);
      auto differential =
          detail::evaluate_acceleration_task_differential(
              *task, robot, control_only_zero_velocity);
      if (differential.status !=
          detail::AccelerationTaskDifferentialStatus::kSuccess) {
        assembly->status =
            differential.status ==
                    detail::AccelerationTaskDifferentialStatus::kInvalidInput
                ? SolverStatus::kInvalidInput
                : SolverStatus::kInvalidInput;
        assembly->message = differential.message;
        return;
      }
      const Eigen::Index dimension = differential.control_jacobian.rows();
      const auto found = references.find(task->getName());
      static const AccelerationTaskReference kDefaultReference;
      const auto &reference =
          found == references.end() ? kDefaultReference : found->second;
      if ((reference.desired_velocity.size() != 0 &&
           reference.desired_velocity.size() != dimension) ||
          (reference.desired_acceleration.size() != 0 &&
           reference.desired_acceleration.size() != dimension) ||
          (reference.desired_velocity.size() != 0 &&
           !reference.desired_velocity.allFinite()) ||
          (reference.desired_acceleration.size() != 0 &&
           !reference.desired_acceleration.allFinite())) {
        assembly->status = SolverStatus::kInvalidInput;
        assembly->message =
            task->getName() + ": acceleration reference is inconsistent";
        return;
      }
      if (!std::isfinite(reference.proportional_gain) ||
          !std::isfinite(reference.derivative_gain) ||
          reference.proportional_gain < 0.0 ||
          reference.derivative_gain < 0.0) {
        assembly->status = SolverStatus::kInvalidInput;
        assembly->message = task->getName() + ": acceleration gains invalid";
        return;
      }

      Eigen::VectorXd scalable =
          reference.proportional_gain * task->getWeight() *
          differential.position_error;
      if (control_only_zero_velocity) {
        // No measured task-rate term is needed.
      } else {
        scalable.noalias() -=
            reference.derivative_gain *
            (differential.physical_jacobian * dq);
      }
      if (reference.desired_velocity.size() != 0) {
        scalable.array() +=
            reference.derivative_gain *
            differential.reference_row_scale.array() *
            reference.desired_velocity.array();
      }
      if (reference.desired_acceleration.size() != 0) {
        scalable.array() += differential.reference_row_scale.array() *
                            reference.desired_acceleration.array();
      }
      matrix.middleRows(cursor, dimension) = differential.control_jacobian;
      target.segment(cursor, dimension) = scalable;
      bias.segment(cursor, dimension) = differential.jacobian_bias;
      if (retain_task_differentials) {
        group_differentials.push_back(std::move(differential));
        group_references.push_back(scalable);
      }
      cursor += dimension;
    }

    ObjectiveSolveConfig config;
    config.priority = priority;
    config.solve_mode = mode;
    config.allow_min_error_fallback = allow_fallback;
    assembly->configs[group_index] = config;
    ++group_index;
    group_begin = group_end;
  }
  assembly->matrices.resize(group_index);
  assembly->targets.resize(group_index);
  assembly->biases.resize(group_index);
  assembly->configs.resize(group_index);
  assembly->groups.resize(group_index);
  assembly->differentials.resize(group_index);
  assembly->references.resize(group_index);
}

const detail::AccelerationTaskDifferential *
find_active_task_differential(const ObjectiveAssembly &objectives,
                              const std::string &task_name) {
  for (std::size_t group_index = 0; group_index < objectives.groups.size();
       ++group_index) {
    for (std::size_t task_index = 0;
         task_index < objectives.groups[group_index].size(); ++task_index) {
      if (objectives.groups[group_index][task_index]->getName() ==
          task_name) {
        return &objectives.differentials[group_index][task_index];
      }
    }
  }
  return nullptr;
}

ConstraintValidation validate_task_acceleration_bounds(
    const std::vector<TaskAccelerationBounds> &bounds,
    const ObjectiveAssembly &objectives,
    std::unordered_set<std::string> *source_ids) {
  ConstraintValidation validation;
  std::unordered_set<std::string> task_names;
  for (const auto &constraint : bounds) {
    const std::string family = constraint_family_name(constraint);
    if (constraint.source_id.empty()) {
      validation.status = SolverStatus::kInvalidInput;
      validation.message = family + " source_id must not be empty";
      return validation;
    }
    if (!source_ids->insert(constraint.source_id).second) {
      validation.status = SolverStatus::kInvalidInput;
      validation.message = "duplicate acceleration constraint source_id '" +
                           constraint.source_id + "'";
      return validation;
    }
    if (constraint.task_name.empty()) {
      validation.status = SolverStatus::kInvalidInput;
      validation.message = family + " '" + constraint.source_id +
                           "' task_name must not be empty";
      return validation;
    }
    if (!task_names.insert(constraint.task_name).second) {
      validation.status = SolverStatus::kInvalidInput;
      validation.message = "duplicate task acceleration bounds for task '" +
                           constraint.task_name + "'";
      return validation;
    }
    const auto *differential =
        find_active_task_differential(objectives, constraint.task_name);
    if (differential == nullptr) {
      validation.status = SolverStatus::kInvalidInput;
      validation.message = family + " '" + constraint.source_id +
                           "' references an unknown or inactive task '" +
                           constraint.task_name + "'";
      return validation;
    }
    const Eigen::Index rows = differential->physical_jacobian.rows();
    if (rows == 0) {
      validation.status = SolverStatus::kInvalidInput;
      validation.message = family + " '" + constraint.source_id +
                           "' task has no physical rows";
      return validation;
    }
    if (constraint.lower_bounds.size() != rows ||
        constraint.upper_bounds.size() != rows) {
      validation.status = SolverStatus::kConstraintBoundsMismatch;
      validation.message = family + " '" + constraint.source_id +
                           "' bounds must match physical task rows";
      return validation;
    }
    if ((!constraint.lower_bound_active.empty() &&
         constraint.lower_bound_active.size() !=
             static_cast<std::size_t>(rows)) ||
        (!constraint.upper_bound_active.empty() &&
         constraint.upper_bound_active.size() !=
             static_cast<std::size_t>(rows))) {
      validation.status = SolverStatus::kShapeMismatch;
      validation.message = family + " '" + constraint.source_id +
                           "' active-side flags must match physical task rows";
      return validation;
    }
    if (!constraint.lower_bounds.allFinite() ||
        !constraint.upper_bounds.allFinite()) {
      validation.status = SolverStatus::kNonFiniteInput;
      validation.message = family + " '" + constraint.source_id +
                           "' must contain only finite bounds";
      return validation;
    }
    for (Eigen::Index row = 0; row < rows; ++row) {
      const bool lower_active = lower_side_active(constraint, row);
      const bool upper_active = upper_side_active(constraint, row);
      if (!lower_active && !upper_active) {
        validation.status = SolverStatus::kInvalidInput;
        validation.message = family + " '" + constraint.source_id + "' row " +
                             std::to_string(row) +
                             " has no active bound side";
        return validation;
      }
      if (lower_active && upper_active &&
          constraint.lower_bounds(row) > constraint.upper_bounds(row)) {
        validation.status = SolverStatus::kInvalidInput;
        validation.message = family + " '" + constraint.source_id +
                             "' lower bound exceeds upper bound";
        return validation;
      }
      if ((lower_active &&
           !checked_shifted_bound(constraint.lower_bounds(row),
                                  differential->jacobian_bias(row))) ||
          (upper_active &&
           !checked_shifted_bound(constraint.upper_bounds(row),
                                  differential->jacobian_bias(row)))) {
        validation.status = SolverStatus::kInvalidInput;
        validation.message = family + " '" + constraint.source_id +
                             "' shifted bound is not finite";
        return validation;
      }
      double inactive_magnitude = 0.0;
      double inactive_side_bound = 0.0;
      if (!inactive_bound_magnitude(differential->physical_jacobian.row(row),
                                    &inactive_magnitude) ||
          (!lower_active &&
           !checked_add(differential->jacobian_bias(row), -inactive_magnitude,
                        &inactive_side_bound)) ||
          (!upper_active &&
           !checked_add(differential->jacobian_bias(row), inactive_magnitude,
                        &inactive_side_bound))) {
        validation.status = SolverStatus::kInvalidInput;
        validation.message = family + " '" + constraint.source_id +
                             "' inactive side cannot be represented in the "
                             "finite backend range";
        return validation;
      }
    }
    validation.row_count += rows;
  }
  return validation;
}

std::vector<AffineAccelerationConstraint> make_task_bound_affine_constraints(
    const std::vector<TaskAccelerationBounds> &bounds,
    const ObjectiveAssembly &objectives) {
  std::vector<AffineAccelerationConstraint> constraints;
  constraints.reserve(bounds.size());
  for (const auto &bound : bounds) {
    const auto *differential =
        find_active_task_differential(objectives, bound.task_name);
    AffineAccelerationConstraint constraint;
    constraint.source_id = bound.source_id;
    constraint.coefficient_matrix = differential->physical_jacobian;
    constraint.affine_bias = differential->jacobian_bias;
    constraint.lower_bounds = bound.lower_bounds;
    constraint.upper_bounds = bound.upper_bounds;
    constraint.lower_bound_active = bound.lower_bound_active;
    constraint.upper_bound_active = bound.upper_bound_active;
    constraints.push_back(std::move(constraint));
  }
  return constraints;
}


} // namespace acceleration_solver_internal
} // namespace embodik
