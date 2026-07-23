#include <embodik/acceleration_solver.hpp>

#include "acceleration_allocation_transform.hpp"
#include "acceleration_state_box.hpp"
#include "acceleration_task_differential.hpp"
#include "acceleration_tight_point_constraint.hpp"
#include "frame_kinematic_differential.hpp"
#include "generalized_constraint_set.hpp"

#include <embodik/ik_baseline.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <map>
#include <stdexcept>
#include <unordered_set>
#include <utility>

namespace embodik {
namespace {

constexpr double kConstraintTolerance = 1e-8;
constexpr double kInactiveAffineBackendBound = 1e10;

AccelerationSolverResult failure(SolverStatus status, std::string message) {
  AccelerationSolverResult result;
  result.status = status;
  result.status_message = std::move(message);
  return result;
}

bool finite_positive_vector(const Eigen::VectorXd &values) {
  return values.size() > 0 && values.allFinite() &&
         (values.array() > 0.0).all();
}

bool fixed_base_scalar_joints_only(const RobotModel &robot) {
  if (robot.is_floating_base() || robot.nq() != robot.nv()) {
    return false;
  }
  for (const auto &joint_name : robot.get_joint_names()) {
    if (robot.get_joint_config_size(joint_name) != 1 ||
        robot.get_joint_velocity_size(joint_name) != 1) {
      return false;
    }
  }
  return true;
}

template <typename Constraint>
const char *constraint_family_name(const Constraint &) {
  return "constraint";
}

template <>
const char *
constraint_family_name<AffineAccelerationConstraint>(
    const AffineAccelerationConstraint &) {
  return "affine acceleration constraint";
}

template <>
const char *
constraint_family_name<FrozenNextVelocityConstraint>(
    const FrozenNextVelocityConstraint &) {
  return "frozen next-velocity constraint";
}

const char *constraint_family_name(const TaskAccelerationBounds &) {
  return "task acceleration bounds";
}

const char *constraint_family_name(const ContactAccelerationConstraint &) {
  return "contact acceleration constraint";
}

const char *constraint_family_name(const TightPointAccelerationConstraint &) {
  return "tight point constraint";
}

template <typename Constraint>
bool lower_side_active(const Constraint &constraint, Eigen::Index row) {
  return constraint.lower_bound_active.empty() ||
         constraint.lower_bound_active[static_cast<std::size_t>(row)];
}

template <typename Constraint>
bool upper_side_active(const Constraint &constraint, Eigen::Index row) {
  return constraint.upper_bound_active.empty() ||
         constraint.upper_bound_active[static_cast<std::size_t>(row)];
}

template <typename Constraint>
Eigen::Index count_constraint_rows(const std::vector<Constraint> &constraints) {
  Eigen::Index rows = 0;
  for (const auto &constraint : constraints) {
    rows += constraint.coefficient_matrix.rows();
  }
  return rows;
}

bool inactive_bound_magnitude(const Eigen::Ref<const Eigen::RowVectorXd> &row,
                              double *magnitude) {
  const double row_norm = row.stableNorm();
  if (!std::isfinite(row_norm) ||
      (row_norm > 0.0 &&
       row_norm > std::numeric_limits<double>::max() /
                      kInactiveAffineBackendBound)) {
    return false;
  }
  *magnitude =
      row_norm == 0.0 ? kInactiveAffineBackendBound
                      : kInactiveAffineBackendBound * row_norm;
  return std::isfinite(*magnitude);
}

template <typename Constraint>
bool inactive_affine_bound_magnitude(const Constraint &constraint,
                                     Eigen::Index row, double *magnitude) {
  return inactive_bound_magnitude(constraint.coefficient_matrix.row(row),
                                  magnitude);
}

bool checked_product(double lhs, double rhs, double *out) {
  const double product = lhs * rhs;
  if (!std::isfinite(product)) {
    return false;
  }
  if (lhs != 0.0 && rhs != 0.0 && product == 0.0) {
    return false;
  }
  *out = product;
  return true;
}

bool checked_quotient(double numerator, double denominator, double *out) {
  if (numerator == 0.0) {
    *out = 0.0;
    return true;
  }
  const double quotient = numerator / denominator;
  if (!std::isfinite(quotient) || quotient == 0.0) {
    return false;
  }
  *out = quotient;
  return true;
}

bool checked_add(double lhs, double rhs, double *out) {
  const double sum = lhs + rhs;
  if (!std::isfinite(sum)) {
    return false;
  }
  *out = sum;
  return true;
}

bool checked_scaled_matrix(const Eigen::MatrixXd &matrix, double scale,
                           Eigen::MatrixXd *scaled) {
  scaled->resize(matrix.rows(), matrix.cols());
  for (Eigen::Index row = 0; row < matrix.rows(); ++row) {
    for (Eigen::Index col = 0; col < matrix.cols(); ++col) {
      if (!checked_product(scale, matrix(row, col), &(*scaled)(row, col))) {
        return false;
      }
    }
  }
  return true;
}

bool checked_dot_row(const Eigen::Ref<const Eigen::RowVectorXd> &row,
                     const Eigen::VectorXd &vector, double *out) {
  double sum = 0.0;
  for (Eigen::Index col = 0; col < row.cols(); ++col) {
    double product = 0.0;
    if (!checked_product(row(col), vector(col), &product) ||
        !checked_add(sum, product, &sum)) {
      return false;
    }
  }
  *out = sum;
  return true;
}

bool checked_shifted_bound(double bound, double bias) {
  double shifted = 0.0;
  return checked_add(bound, -bias, &shifted);
}

struct ConstraintValidation {
  SolverStatus status = SolverStatus::kSuccess;
  std::string message;
  Eigen::Index row_count = 0;
};

struct PreparedEffortConstraint {
  SolverStatus status = SolverStatus::kSuccess;
  std::string message;
  Eigen::MatrixXd mass_matrix;
  Eigen::VectorXd bias;
  Eigen::VectorXd limits;
};

struct InverseDynamicsEvaluation {
  SolverStatus status = SolverStatus::kSuccess;
  std::string message;
  Eigen::VectorXd torques;
};

struct ContactConstraintAssembly {
  SolverStatus status = SolverStatus::kSuccess;
  std::string message;
  std::vector<AffineAccelerationConstraint> constraints;
  Eigen::Index row_count = 0;
};

struct TightPointConstraintAssembly {
  SolverStatus status = SolverStatus::kSuccess;
  std::string message;
  std::vector<AffineAccelerationConstraint> constraints;
  std::vector<detail::PreparedTightPointConstraint> prepared_constraints;
  Eigen::Index row_count = 0;
};

template <typename Constraint>
ConstraintValidation validate_constraint_family(
    const std::vector<Constraint> &constraints, Eigen::Index variable_count,
    std::unordered_set<std::string> *source_ids,
    bool validate_affine_backend_range) {
  ConstraintValidation validation;
  for (const auto &constraint : constraints) {
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
    if (constraint.coefficient_matrix.rows() == 0) {
      validation.status = SolverStatus::kInvalidInput;
      validation.message = family + " '" + constraint.source_id +
                           "' must contain at least one row";
      return validation;
    }
    if (constraint.coefficient_matrix.cols() != variable_count) {
      validation.status = SolverStatus::kShapeMismatch;
      validation.message = family + " '" + constraint.source_id +
                           "' matrix must have nv columns";
      return validation;
    }
    const Eigen::Index rows = constraint.coefficient_matrix.rows();
    if (constraint.affine_bias.size() != rows) {
      validation.status = SolverStatus::kShapeMismatch;
      validation.message = family + " '" + constraint.source_id +
                           "' bias must match matrix rows";
      return validation;
    }
    if (constraint.lower_bounds.size() != rows ||
        constraint.upper_bounds.size() != rows) {
      validation.status = SolverStatus::kConstraintBoundsMismatch;
      validation.message = family + " '" + constraint.source_id +
                           "' bounds must match matrix rows";
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
                           "' active-side flags must match matrix rows";
      return validation;
    }
    if (!constraint.coefficient_matrix.allFinite() ||
        !constraint.affine_bias.allFinite() ||
        !constraint.lower_bounds.allFinite() ||
        !constraint.upper_bounds.allFinite()) {
      validation.status = SolverStatus::kNonFiniteInput;
      validation.message = family + " '" + constraint.source_id +
                           "' must contain only finite values";
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
      if (validate_affine_backend_range) {
        if ((lower_active &&
             !checked_shifted_bound(constraint.lower_bounds(row),
                                    constraint.affine_bias(row))) ||
            (upper_active &&
             !checked_shifted_bound(constraint.upper_bounds(row),
                                    constraint.affine_bias(row)))) {
          validation.status = SolverStatus::kInvalidInput;
          validation.message = family + " '" + constraint.source_id +
                               "' shifted bound is not finite";
          return validation;
        }
        double inactive_magnitude = 0.0;
        double inactive_side_bound = 0.0;
        if (!inactive_affine_bound_magnitude(constraint, row,
                                             &inactive_magnitude) ||
            (!lower_active &&
             !checked_add(constraint.affine_bias(row), -inactive_magnitude,
                          &inactive_side_bound)) ||
            (!upper_active &&
             !checked_add(constraint.affine_bias(row), inactive_magnitude,
                          &inactive_side_bound))) {
          validation.status = SolverStatus::kInvalidInput;
          validation.message = family + " '" + constraint.source_id +
                               "' inactive side cannot be represented in the "
                               "finite backend range";
          return validation;
        }
      }
    }
    validation.row_count += rows;
  }
  return validation;
}

ConstraintValidation validate_acceleration_constraints(
    const std::vector<AffineAccelerationConstraint> &affine_constraints,
    const std::vector<FrozenNextVelocityConstraint> &frozen_constraints,
    Eigen::Index variable_count,
    std::unordered_set<std::string> *source_ids) {
  ConstraintValidation validation;
  validation = validate_constraint_family(affine_constraints, variable_count,
                                          source_ids, true);
  if (validation.status != SolverStatus::kSuccess) {
    return validation;
  }
  const Eigen::Index affine_rows = validation.row_count;
  validation = validate_constraint_family(frozen_constraints, variable_count,
                                          source_ids, false);
  validation.row_count += affine_rows;
  return validation;
}

VelocitySolverConfig acceleration_backend_config() {
  VelocitySolverConfig config;
  config.epsilon = 1e-10;
  config.precision_threshold = 1e-12;
  config.iteration_limit = 50;
  config.magnitude_limit = 1e10;
  config.stall_detection_count = 2;
  config.regularization_config.epsilon = 1e-10;
  config.regularization_config.regularization_factor = 0.0;
  return config;
}

Eigen::VectorXd resolve_acceleration_limits(
    const RobotModel &robot, const AccelerationSolveOptions &options,
    SolverStatus *status, std::string *message) {
  if (options.acceleration_limits_override.has_value()) {
    return options.acceleration_limits_override.value();
  }
  if (!robot.has_custom_acceleration_limits()) {
    *status = SolverStatus::kInvalidInput;
    *message =
        "AccelerationSolver requires explicit acceleration limits; pass an "
        "override or call RobotModel::set_acceleration_limits()";
    return {};
  }
  return robot.get_acceleration_limits();
}

struct StateBoxAssembly {
  SolverStatus status = SolverStatus::kSuccess;
  std::string message;
  Eigen::VectorXd lower;
  Eigen::VectorXd upper;
  std::vector<detail::StateBoxCauseSet> lower_causes;
  std::vector<detail::StateBoxCauseSet> upper_causes;
};

StateBoxAssembly build_joint_state_box(const RobotModel &robot,
                                       const Eigen::VectorXd &q,
                                       const Eigen::VectorXd &dq, double dt,
                                       const Eigen::VectorXd &accel_limits,
                                       const AccelerationSolveOptions &options) {
  StateBoxAssembly assembly;
  const Eigen::Index nv = robot.nv();
  assembly.lower.resize(nv);
  assembly.upper.resize(nv);
  assembly.lower_causes.resize(static_cast<std::size_t>(nv), 0U);
  assembly.upper_causes.resize(static_cast<std::size_t>(nv), 0U);

  Eigen::VectorXd velocity_limits;
  if (options.apply_velocity_limits) {
    velocity_limits = robot.get_velocity_limits();
    if (velocity_limits.size() != nv ||
        !finite_positive_vector(velocity_limits)) {
      assembly.status = SolverStatus::kInvalidInput;
      assembly.message =
          "velocity limits must have size nv and be finite and positive";
      return assembly;
    }
  }

  Eigen::VectorXd lower_positions;
  Eigen::VectorXd upper_positions;
  if (options.apply_position_limits) {
    const auto joint_limits = robot.get_joint_limits();
    lower_positions = joint_limits.first;
    upper_positions = joint_limits.second;
    if (lower_positions.size() != q.size() || upper_positions.size() != q.size() ||
        !lower_positions.allFinite() || !upper_positions.allFinite() ||
        (lower_positions.array() > upper_positions.array()).any()) {
      assembly.status = SolverStatus::kInvalidInput;
      assembly.message =
          "position limits must be finite, ordered, and have size nq";
      return assembly;
    }
  }

  for (Eigen::Index index = 0; index < nv; ++index) {
    detail::StateBoxRowInput row;
    row.value = q(index);
    row.rate = dq(index);
    row.acceleration_bounds = {-accel_limits(index), accel_limits(index), true,
                               true};
    if (options.apply_velocity_limits) {
      row.rate_bounds = {-velocity_limits(index), velocity_limits(index), true,
                         true};
    }
    if (options.apply_position_limits) {
      row.state_bounds = {lower_positions(index), upper_positions(index), true,
                          true};
      row.lower_braking_acceleration = accel_limits(index);
      row.upper_braking_acceleration = accel_limits(index);
    }

    const auto shaped = detail::shape_state_box_row(row, dt);
    if (shaped.status != SolverStatus::kSuccess) {
      assembly.status = shaped.status;
      assembly.message = shaped.message;
      return assembly;
    }
    assembly.lower(index) = shaped.lower_acceleration;
    assembly.upper(index) = shaped.upper_acceleration;
    assembly.lower_causes[static_cast<std::size_t>(index)] =
        shaped.lower_causes;
    assembly.upper_causes[static_cast<std::size_t>(index)] =
        shaped.upper_causes;
  }
  return assembly;
}

Eigen::VectorXd reference_or_zero(const Eigen::VectorXd &value,
                                  Eigen::Index dimension,
                                  const std::string &field_name) {
  if (value.size() == 0) {
    return Eigen::VectorXd::Zero(dimension);
  }
  if (value.size() != dimension) {
    throw std::invalid_argument(field_name + " dimension mismatch");
  }
  if (!value.allFinite()) {
    throw std::invalid_argument(field_name + " must be finite");
  }
  return value;
}

struct ObjectiveAssembly {
  SolverStatus status = SolverStatus::kSuccess;
  std::string message;
  std::vector<Eigen::MatrixXd> matrices;
  std::vector<Eigen::VectorXd> targets;
  std::vector<Eigen::VectorXd> biases;
  std::vector<ObjectiveSolveConfig> configs;
  std::vector<std::vector<std::shared_ptr<Task>>> groups;
  std::vector<std::vector<detail::AccelerationTaskDifferential>> differentials;
  std::vector<std::vector<Eigen::VectorXd>> references;
  bool synthetic_hold_objective = false;
};

ObjectiveAssembly assemble_objectives(
    const std::vector<std::shared_ptr<Task>> &tasks,
    const std::unordered_map<std::string, AccelerationTaskReference> &references,
    const RobotModel &robot, const Eigen::VectorXd &dq) {
  ObjectiveAssembly assembly;
  std::map<int, std::vector<std::shared_ptr<Task>>> grouped_tasks;
  for (const auto &task : tasks) {
    if (task && task->isActive()) {
      if (task->getSolveMode() == TaskSolveMode::kScaleElastic) {
        assembly.status = SolverStatus::kInvalidInput;
        assembly.message =
            "AccelerationSolver does not support SCALE_ELASTIC tasks";
        return assembly;
      }
      grouped_tasks[task->getPriority()].push_back(task);
    }
  }
  if (grouped_tasks.empty()) {
    assembly.matrices.push_back(
        Eigen::MatrixXd::Identity(robot.nv(), robot.nv()));
    assembly.targets.push_back(Eigen::VectorXd::Zero(robot.nv()));
    assembly.biases.push_back(Eigen::VectorXd::Zero(robot.nv()));
    ObjectiveSolveConfig config;
    config.priority = 0;
    config.solve_mode = TaskSolveMode::kMinError;
    config.allow_min_error_fallback = true;
    assembly.configs.push_back(config);
    assembly.groups.push_back({});
    assembly.differentials.push_back({});
    assembly.references.push_back({});
    assembly.synthetic_hold_objective = true;
    return assembly;
  }

  for (const auto &[priority, group] : grouped_tasks) {
    Eigen::Index rows = 0;
    for (const auto &task : group) {
      rows += task->getDimension();
    }
    Eigen::MatrixXd matrix(rows, robot.nv());
    Eigen::VectorXd target(rows);
    Eigen::VectorXd bias(rows);
    std::vector<detail::AccelerationTaskDifferential> group_differentials;
    std::vector<Eigen::VectorXd> group_references;

    const TaskSolveMode mode = group.front()->getSolveMode();
    const bool allow_fallback = group.front()->getAllowMinErrorFallback();
    Eigen::Index cursor = 0;
    for (const auto &task : group) {
      if (task->getSolveMode() != mode ||
          task->getAllowMinErrorFallback() != allow_fallback) {
        assembly.status = SolverStatus::kInvalidInput;
        assembly.message =
            "same-priority acceleration tasks must share solve mode and "
            "fallback settings";
        return assembly;
      }
      task->update(robot);
      const auto differential =
          detail::evaluate_acceleration_task_differential(*task, robot);
      if (differential.status !=
          detail::AccelerationTaskDifferentialStatus::kSuccess) {
        assembly.status =
            differential.status ==
                    detail::AccelerationTaskDifferentialStatus::kInvalidInput
                ? SolverStatus::kInvalidInput
                : SolverStatus::kInvalidInput;
        assembly.message = differential.message;
        return assembly;
      }
      const Eigen::Index dimension = differential.control_jacobian.rows();
      const auto found = references.find(task->getName());
      const AccelerationTaskReference reference =
          found == references.end() ? AccelerationTaskReference{}
                                    : found->second;
      Eigen::VectorXd desired_velocity;
      Eigen::VectorXd desired_acceleration;
      try {
        desired_velocity =
            reference_or_zero(reference.desired_velocity, dimension,
                              "desired velocity");
        desired_acceleration =
            reference_or_zero(reference.desired_acceleration, dimension,
                              "desired acceleration");
      } catch (const std::invalid_argument &error) {
        assembly.status = SolverStatus::kInvalidInput;
        assembly.message = task->getName() + ": " + error.what();
        return assembly;
      }
      if (!std::isfinite(reference.proportional_gain) ||
          !std::isfinite(reference.derivative_gain) ||
          reference.proportional_gain < 0.0 ||
          reference.derivative_gain < 0.0) {
        assembly.status = SolverStatus::kInvalidInput;
        assembly.message = task->getName() + ": acceleration gains invalid";
        return assembly;
      }

      const Eigen::VectorXd task_velocity =
          differential.physical_jacobian * dq;
      const Eigen::VectorXd desired_velocity_scaled =
          differential.reference_row_scale.cwiseProduct(desired_velocity);
      const Eigen::VectorXd desired_acceleration_scaled =
          differential.reference_row_scale.cwiseProduct(desired_acceleration);
      Eigen::VectorXd scalable =
          desired_acceleration_scaled +
          reference.derivative_gain *
              (desired_velocity_scaled - task_velocity) +
          reference.proportional_gain * task->getWeight() *
              differential.position_error;
      matrix.middleRows(cursor, dimension) = differential.control_jacobian;
      target.segment(cursor, dimension) = scalable;
      bias.segment(cursor, dimension) = differential.jacobian_bias;
      group_differentials.push_back(differential);
      group_references.push_back(scalable);
      cursor += dimension;
    }

    assembly.matrices.push_back(std::move(matrix));
    assembly.targets.push_back(std::move(target));
    assembly.biases.push_back(std::move(bias));
    ObjectiveSolveConfig config;
    config.priority = priority;
    config.solve_mode = mode;
    config.allow_min_error_fallback = allow_fallback;
    assembly.configs.push_back(config);
    assembly.groups.push_back(group);
    assembly.differentials.push_back(std::move(group_differentials));
    assembly.references.push_back(std::move(group_references));
  }
  return assembly;
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

ContactConstraintAssembly make_contact_acceleration_constraints(
    const RobotModel &robot,
    const std::vector<ContactAccelerationConstraint> &contacts,
    std::unordered_set<std::string> *source_ids) {
  ContactConstraintAssembly assembly;
  assembly.constraints.reserve(contacts.size());

  for (const auto &contact : contacts) {
    const std::string family = constraint_family_name(contact);
    if (contact.source_id.empty()) {
      assembly.status = SolverStatus::kInvalidInput;
      assembly.message = family + " source_id must not be empty";
      return assembly;
    }
    if (!source_ids->insert(contact.source_id).second) {
      assembly.status = SolverStatus::kInvalidInput;
      assembly.message = "duplicate acceleration constraint source_id '" +
                         contact.source_id + "'";
      return assembly;
    }
    if (contact.frame_name.empty()) {
      assembly.status = SolverStatus::kInvalidInput;
      assembly.message = family + " '" + contact.source_id +
                         "' frame_name must not be empty";
      return assembly;
    }
    if (contact.type != ContactType::kPointContact &&
        contact.type != ContactType::kRigidContact) {
      assembly.status = SolverStatus::kInvalidInput;
      assembly.message = family + " '" + contact.source_id +
                         "' has an unsupported contact type";
      return assembly;
    }
    if (!robot.has_frame(contact.frame_name)) {
      assembly.status = SolverStatus::kInvalidInput;
      assembly.message = family + " '" + contact.source_id +
                         "' references an unknown frame '" +
                         contact.frame_name + "'";
      return assembly;
    }

    detail::FrameKinematicDifferential differential;
    try {
      differential =
          detail::evaluate_frame_kinematic_differential(robot,
                                                        contact.frame_name);
    } catch (const std::exception &error) {
      assembly.status = SolverStatus::kNumericalError;
      assembly.message = family + " '" + contact.source_id +
                         "' kinematic differential failed: " + error.what();
      return assembly;
    }
    if (differential.jacobian.cols() != robot.nv() ||
        differential.jacobian.rows() != 6 ||
        differential.affine_bias.size() != 6 ||
        !differential.jacobian.allFinite() ||
        !differential.affine_bias.allFinite()) {
      assembly.status = SolverStatus::kNumericalError;
      assembly.message = family + " '" + contact.source_id +
                         "' produced a non-finite kinematic row";
      return assembly;
    }

    const Eigen::Index rows =
        contact.type == ContactType::kPointContact ? 3 : 6;
    AffineAccelerationConstraint constraint;
    constraint.source_id = contact.source_id;
    constraint.coefficient_matrix =
        differential.jacobian.topRows(rows);
    constraint.affine_bias = differential.affine_bias.head(rows);
    constraint.lower_bounds = Eigen::VectorXd::Zero(rows);
    constraint.upper_bounds = Eigen::VectorXd::Zero(rows);
    assembly.row_count += rows;
    assembly.constraints.push_back(std::move(constraint));
  }

  return assembly;
}

TightPointConstraintAssembly make_tight_point_constraints(
    const RobotModel &robot,
    const std::vector<TightPointAccelerationConstraint> &tight_points,
    double dt, const Eigen::VectorXd &joint_acceleration_lower,
    const Eigen::VectorXd &joint_acceleration_upper,
    std::unordered_set<std::string> *source_ids) {
  TightPointConstraintAssembly assembly;
  assembly.constraints.reserve(tight_points.size());
  assembly.prepared_constraints.reserve(tight_points.size());

  for (const auto &tight_point : tight_points) {
    const std::string family = constraint_family_name(tight_point);
    if (tight_point.source_id.empty()) {
      assembly.status = SolverStatus::kInvalidInput;
      assembly.message = family + " source_id must not be empty";
      return assembly;
    }
    if (!source_ids->insert(tight_point.source_id).second) {
      assembly.status = SolverStatus::kInvalidInput;
      assembly.message = "duplicate acceleration constraint source_id '" +
                         tight_point.source_id + "'";
      return assembly;
    }

    const auto prepared = detail::prepare_tight_point_constraint(
        tight_point, robot, dt, joint_acceleration_lower,
        joint_acceleration_upper);
    if (!prepared.satisfied()) {
      assembly.status = prepared.status;
      assembly.message = prepared.message;
      return assembly;
    }
    assembly.row_count +=
        prepared.prepared->scalar.state_box.physical_constraint
            .coefficient_matrix
            .rows();
    assembly.constraints.push_back(
        prepared.prepared->scalar.state_box.physical_constraint);
    assembly.prepared_constraints.push_back(std::move(*prepared.prepared));
  }

  return assembly;
}

Eigen::VectorXd vector_from_solution(const SolverResult &result) {
  if (result.solution.empty()) {
    return {};
  }
  return Eigen::Map<const Eigen::VectorXd>(result.solution.data(),
                                           result.solution.size());
}

void assign_solution_from_vector(SolverResult *result,
                                 const Eigen::VectorXd &solution) {
  result->solution.assign(solution.data(), solution.data() + solution.size());
}

bool zero_excluded_by_bounds(const Eigen::VectorXd &lower,
                             const Eigen::VectorXd &upper) {
  return (lower.array() > kConstraintTolerance).any() ||
         (upper.array() < -kConstraintTolerance).any();
}

bool populate_allocation_diagnostics(
    AccelerationSolverResult *result,
    const detail::PreparedAccelerationAllocationTransform &allocation,
    const Eigen::VectorXd &physical_acceleration) {
  AccelerationAllocationDiagnostics diagnostics;
  diagnostics.applied = true;
  diagnostics.physical_metric_diagonal =
      allocation.requested_metric_diagonal;
  diagnostics.reference_acceleration =
      allocation.reference_acceleration;
  const Eigen::VectorXd residual =
      physical_acceleration - allocation.reference_acceleration;
  diagnostics.weighted_physical_residual =
      allocation.requested_metric_diagonal.array().sqrt().matrix()
          .cwiseProduct(residual);
  diagnostics.objective_value =
      0.5 * diagnostics.weighted_physical_residual.squaredNorm();
  if (!diagnostics.weighted_physical_residual.allFinite() ||
      !std::isfinite(diagnostics.objective_value)) {
    return false;
  }
  result->allocation_diagnostics = std::move(diagnostics);
  return true;
}

PreparedEffortConstraint prepare_effort_constraint(
    const RobotModel &robot, const Eigen::VectorXd &q,
    const Eigen::VectorXd &dq, const EffortConstraintOptions &options) {
  PreparedEffortConstraint prepared;
  const auto fail = [&](SolverStatus status, std::string message) {
    prepared = {};
    prepared.status = status;
    prepared.message = std::move(message);
    return prepared;
  };

  if (!std::isfinite(options.margin_fraction)) {
    return fail(SolverStatus::kNonFiniteInput,
                "effort margin fraction must be finite");
  }
  if (options.margin_fraction < 0.0 || options.margin_fraction >= 1.0) {
    return fail(SolverStatus::kInvalidInput,
                "effort margin fraction must be in [0, 1)");
  }

  const Eigen::Index nv = robot.nv();
  Eigen::VectorXd raw_limits;
  if (options.limits_override.has_value()) {
    raw_limits = options.limits_override.value();
    if (raw_limits.size() != nv) {
      return fail(SolverStatus::kShapeMismatch,
                  "effort limits override must have size nv");
    }
    if (!raw_limits.allFinite()) {
      return fail(SolverStatus::kNonFiniteInput,
                  "effort limits override must be finite");
    }
  } else {
    raw_limits = robot.get_effort_limits();
    if (raw_limits.size() != nv) {
      return fail(SolverStatus::kShapeMismatch,
                  "model effort limits must have size nv");
    }
    if (!raw_limits.allFinite()) {
      return fail(SolverStatus::kInvalidInput,
                  "model effort limits must be finite and positive");
    }
  }
  if ((raw_limits.array() <= 0.0).any()) {
    return fail(SolverStatus::kInvalidInput,
                "effort limits must be strictly positive");
  }

  prepared.limits = (1.0 - options.margin_fraction) * raw_limits;
  if (!finite_positive_vector(prepared.limits)) {
    return fail(SolverStatus::kInvalidInput,
                "effort margin must leave finite positive limits");
  }

  try {
    prepared.mass_matrix = robot.compute_mass_matrix(q);
    prepared.mass_matrix =
        0.5 * (prepared.mass_matrix + prepared.mass_matrix.transpose());
    prepared.bias = robot.rnea(q, dq, Eigen::VectorXd::Zero(nv));
  } catch (const std::exception &error) {
    return fail(SolverStatus::kNumericalError,
                std::string("effort dynamics evaluation failed: ") +
                    error.what());
  }

  if (prepared.mass_matrix.rows() != nv || prepared.mass_matrix.cols() != nv ||
      prepared.bias.size() != nv) {
    return fail(SolverStatus::kShapeMismatch,
                "effort dynamics returned inconsistent dimensions");
  }
  if (!prepared.mass_matrix.allFinite() || !prepared.bias.allFinite()) {
    return fail(SolverStatus::kNumericalError,
                "effort dynamics returned non-finite values");
  }
  if (!(-prepared.limits - prepared.bias).allFinite() ||
      !(prepared.limits - prepared.bias).allFinite()) {
    return fail(SolverStatus::kNumericalError,
                "shifted effort bounds are not finite");
  }
  return prepared;
}

InverseDynamicsEvaluation evaluate_inverse_dynamics(
    const RobotModel &robot, const Eigen::VectorXd &q,
    const Eigen::VectorXd &dq, const Eigen::VectorXd &ddq) {
  InverseDynamicsEvaluation evaluation;
  if (ddq.size() != robot.nv()) {
    evaluation.status = SolverStatus::kShapeMismatch;
    evaluation.message = "accepted acceleration must have size nv";
    return evaluation;
  }
  if (!ddq.allFinite()) {
    evaluation.status = SolverStatus::kNonFiniteInput;
    evaluation.message = "accepted acceleration must be finite";
    return evaluation;
  }
  try {
    evaluation.torques = robot.rnea(q, dq, ddq);
  } catch (const std::exception &error) {
    evaluation.status = SolverStatus::kNumericalError;
    evaluation.message =
        std::string("inverse dynamics evaluation failed: ") + error.what();
    return evaluation;
  }
  if (evaluation.torques.size() != robot.nv()) {
    evaluation.status = SolverStatus::kShapeMismatch;
    evaluation.message = "inverse dynamics returned inconsistent dimensions";
    evaluation.torques.resize(0);
  } else if (!evaluation.torques.allFinite()) {
    evaluation.status = SolverStatus::kNumericalError;
    evaluation.message = "inverse dynamics returned non-finite torques";
    evaluation.torques.resize(0);
  }
  return evaluation;
}

AccelerationSolverResult clear_outputs(AccelerationSolverResult result) {
  result.solution.clear();
  result.joint_accelerations.resize(0);
  result.joint_velocities_next.resize(0);
  result.q_solution.resize(0);
  result.predicted_torques.resize(0);
  result.saturated_effort_indices.clear();
  result.allocation_diagnostics = {};
  return result;
}

void append_unique(std::vector<int> *indices, int index) {
  if (std::find(indices->begin(), indices->end(), index) == indices->end()) {
    indices->push_back(index);
  }
}

void attribute_saturation(AccelerationSolverResult *result,
                          const StateBoxAssembly &state_box,
                          const Eigen::VectorXd &ddq) {
  const auto position_causes =
      detail::state_box_cause(detail::StateBoxBoundCause::kStateRate) |
      detail::state_box_cause(detail::StateBoxBoundCause::kStateEndpoint) |
      detail::state_box_cause(detail::StateBoxBoundCause::kContinuousPath) |
      detail::state_box_cause(detail::StateBoxBoundCause::kNextStateViability);
  for (Eigen::Index index = 0; index < ddq.size(); ++index) {
    detail::StateBoxCauseSet causes = 0U;
    if (std::abs(ddq(index) - state_box.lower(index)) <=
        kConstraintTolerance) {
      causes |= state_box.lower_causes[static_cast<std::size_t>(index)];
    }
    if (std::abs(ddq(index) - state_box.upper(index)) <=
        kConstraintTolerance) {
      causes |= state_box.upper_causes[static_cast<std::size_t>(index)];
    }
    if (causes == 0U) {
      continue;
    }
    if ((causes & detail::state_box_cause(
                      detail::StateBoxBoundCause::kAcceleration)) != 0U) {
      append_unique(&result->saturated_acceleration_indices,
                    static_cast<int>(index));
    }
    if ((causes & detail::state_box_cause(detail::StateBoxBoundCause::kRate)) !=
        0U) {
      append_unique(&result->saturated_velocity_indices,
                    static_cast<int>(index));
    }
    if ((causes & position_causes) != 0U) {
      append_unique(&result->saturated_position_indices,
                    static_cast<int>(index));
    }
  }
}

struct LockRows {
  SolverStatus status = SolverStatus::kSuccess;
  std::string message;
  Eigen::MatrixXd coefficients;
  Eigen::VectorXd bounds;
};

bool insert_lock_indices(const std::vector<int> &indices,
                         const std::string &field_name, Eigen::Index nv,
                         std::unordered_set<int> *seen,
                         SolverStatus *status, std::string *message) {
  std::unordered_set<int> local_seen;
  for (int index : indices) {
    if (index < 0 || index >= nv) {
      *status = SolverStatus::kInvalidInput;
      *message = field_name + " contains out-of-range joint index";
      return false;
    }
    if (!local_seen.insert(index).second) {
      *status = SolverStatus::kInvalidInput;
      *message = field_name + " contains duplicate joint index";
      return false;
    }
    if (!seen->insert(index).second) {
      *status = SolverStatus::kInvalidInput;
      *message = "acceleration lock joint indices overlap across policies";
      return false;
    }
  }
  return true;
}

LockRows build_lock_rows(const AccelerationSolveOptions &options,
                         const Eigen::VectorXd &dq, double dt) {
  LockRows rows;
  const Eigen::Index nv = dq.size();
  std::unordered_set<int> seen;
  if (!insert_lock_indices(options.zero_acceleration_joint_indices,
                           "zero_acceleration_joint_indices", nv, &seen,
                           &rows.status, &rows.message) ||
      !insert_lock_indices(options.zero_next_velocity_joint_indices,
                           "zero_next_velocity_joint_indices", nv, &seen,
                           &rows.status, &rows.message) ||
      !insert_lock_indices(options.fixed_current_position_joint_indices,
                           "fixed_current_position_joint_indices", nv, &seen,
                           &rows.status, &rows.message)) {
    return rows;
  }
  if (seen.empty()) {
    rows.coefficients.resize(0, nv);
    rows.bounds.resize(0);
    return rows;
  }

  const Eigen::Index row_count = static_cast<Eigen::Index>(seen.size());
  rows.coefficients = Eigen::MatrixXd::Zero(row_count, nv);
  rows.bounds.resize(row_count);
  Eigen::Index cursor = 0;
  const auto append = [&](std::vector<int> indices, double numerator_factor,
                          bool divide_by_dt) {
    std::sort(indices.begin(), indices.end());
    for (int index : indices) {
      double target = 0.0;
      double numerator = 0.0;
      if (!checked_product(numerator_factor, dq(index), &numerator) ||
          (divide_by_dt && !checked_quotient(numerator, dt, &target))) {
        rows.status = SolverStatus::kNumericalError;
        rows.message =
            "acceleration lock target cannot be represented finitely";
        return false;
      }
      if (!divide_by_dt) {
        target = numerator;
      }
      rows.coefficients(cursor, index) = 1.0;
      rows.bounds(cursor) = target;
      ++cursor;
    }
    return true;
  };
  if (!append(options.zero_acceleration_joint_indices, 0.0, false) ||
      !append(options.zero_next_velocity_joint_indices, -1.0, true) ||
      !append(options.fixed_current_position_joint_indices, -2.0, true)) {
    return rows;
  }
  return rows;
}

detail::GeneralizedConstraintBlock make_affine_constraint_block(
    const std::vector<AffineAccelerationConstraint> &constraints,
    Eigen::Index variable_count, Eigen::Index row_count) {
  detail::GeneralizedConstraintBlock block;
  block.coefficient_matrix.resize(row_count, variable_count);
  block.affine_bias.resize(row_count);
  block.physical_lower_bounds.resize(row_count);
  block.physical_upper_bounds.resize(row_count);

  Eigen::Index cursor = 0;
  for (const auto &constraint : constraints) {
    const Eigen::Index rows = constraint.coefficient_matrix.rows();
    block.coefficient_matrix.middleRows(cursor, rows) =
        constraint.coefficient_matrix;
    block.affine_bias.segment(cursor, rows) = constraint.affine_bias;
    for (Eigen::Index row = 0; row < rows; ++row) {
      const Eigen::Index output_row = cursor + row;
      double inactive_magnitude = 0.0;
      inactive_affine_bound_magnitude(constraint, row, &inactive_magnitude);
      block.physical_lower_bounds(output_row) =
          lower_side_active(constraint, row)
              ? constraint.lower_bounds(row)
              : constraint.affine_bias(row) - inactive_magnitude;
      block.physical_upper_bounds(output_row) =
          upper_side_active(constraint, row)
              ? constraint.upper_bounds(row)
              : constraint.affine_bias(row) + inactive_magnitude;
    }
    cursor += rows;
  }
  return block;
}

detail::GeneralizedConstraintBlock make_lock_constraint_block(
    const LockRows &locks, Eigen::Index variable_count) {
  detail::GeneralizedConstraintBlock block;
  block.coefficient_matrix = locks.coefficients;
  block.affine_bias = Eigen::VectorXd::Zero(locks.bounds.size());
  block.physical_lower_bounds = locks.bounds;
  block.physical_upper_bounds = locks.bounds;
  if (locks.coefficients.cols() != variable_count) {
    block.coefficient_matrix.resize(0, variable_count);
    block.affine_bias.resize(0);
    block.physical_lower_bounds.resize(0);
    block.physical_upper_bounds.resize(0);
  }
  return block;
}

detail::GeneralizedConstraintBlock make_effort_constraint_block(
    const PreparedEffortConstraint &effort, Eigen::Index variable_count) {
  detail::GeneralizedConstraintBlock block;
  block.coefficient_matrix = effort.mass_matrix;
  block.affine_bias = effort.bias;
  block.physical_lower_bounds = -effort.limits;
  block.physical_upper_bounds = effort.limits;
  if (block.coefficient_matrix.cols() != variable_count) {
    block.coefficient_matrix.resize(0, variable_count);
    block.affine_bias.resize(0);
    block.physical_lower_bounds.resize(0);
    block.physical_upper_bounds.resize(0);
  }
  return block;
}

struct FrozenTransform {
  SolverStatus status = SolverStatus::kSuccess;
  std::string message;
  std::vector<AffineAccelerationConstraint> transformed_constraints;
  Eigen::Index row_count = 0;
};

FrozenTransform transform_frozen_constraints(
    const std::vector<FrozenNextVelocityConstraint> &constraints,
    const Eigen::VectorXd &dq, double dt) {
  FrozenTransform transform;
  transform.transformed_constraints.reserve(constraints.size());
  for (const auto &constraint : constraints) {
    AffineAccelerationConstraint transformed;
    transformed.source_id = constraint.source_id;
    transformed.lower_bounds = constraint.lower_bounds;
    transformed.upper_bounds = constraint.upper_bounds;
    transformed.lower_bound_active = constraint.lower_bound_active;
    transformed.upper_bound_active = constraint.upper_bound_active;
    if (!checked_scaled_matrix(constraint.coefficient_matrix, dt,
                               &transformed.coefficient_matrix)) {
      transform.status = SolverStatus::kNumericalError;
      transform.message = "frozen next-velocity constraint '" +
                          constraint.source_id +
                          "' transformed coefficient is not reliable";
      return transform;
    }
    transformed.affine_bias.resize(constraint.affine_bias.size());
    for (Eigen::Index row = 0; row < constraint.coefficient_matrix.rows();
         ++row) {
      double cdq = 0.0;
      double dt_bias = 0.0;
      if (!checked_dot_row(constraint.coefficient_matrix.row(row), dq, &cdq) ||
          !checked_product(dt, constraint.affine_bias(row), &dt_bias) ||
          !checked_add(cdq, dt_bias, &transformed.affine_bias(row))) {
        transform.status = SolverStatus::kNumericalError;
        transform.message = "frozen next-velocity constraint '" +
                            constraint.source_id +
                            "' transformed bias is not reliable";
        return transform;
      }
      if ((lower_side_active(transformed, row) &&
           !checked_shifted_bound(transformed.lower_bounds(row),
                                  transformed.affine_bias(row))) ||
          (upper_side_active(transformed, row) &&
           !checked_shifted_bound(transformed.upper_bounds(row),
                                  transformed.affine_bias(row)))) {
        transform.status = SolverStatus::kNumericalError;
        transform.message = "frozen next-velocity constraint '" +
                            constraint.source_id +
                            "' transformed bound is not reliable";
        return transform;
      }
      double inactive_magnitude = 0.0;
      double unused = 0.0;
      if (!inactive_affine_bound_magnitude(transformed, row,
                                           &inactive_magnitude) ||
          (!lower_side_active(transformed, row) &&
           !checked_add(transformed.affine_bias(row), -inactive_magnitude,
                        &unused)) ||
          (!upper_side_active(transformed, row) &&
           !checked_add(transformed.affine_bias(row), inactive_magnitude,
                        &unused))) {
        transform.status = SolverStatus::kNumericalError;
        transform.message = "frozen next-velocity constraint '" +
                            constraint.source_id +
                            "' inactive side cannot be represented in the "
                            "finite backend range";
        return transform;
      }
    }
    transform.row_count += transformed.coefficient_matrix.rows();
    transform.transformed_constraints.push_back(std::move(transformed));
  }
  return transform;
}

bool accepted_affine_constraints_are_satisfied(
    const std::vector<AffineAccelerationConstraint> &constraints,
    const Eigen::VectorXd &ddq) {
  for (const auto &constraint : constraints) {
    const Eigen::VectorXd physical =
        constraint.coefficient_matrix * ddq + constraint.affine_bias;
    if (!physical.allFinite()) {
      return false;
    }
    for (Eigen::Index row = 0; row < physical.size(); ++row) {
      const double lower_tolerance =
          std::max(kConstraintTolerance,
                   1e-12 * std::max(std::abs(physical(row)),
                                    std::abs(constraint.lower_bounds(row))));
      const double upper_tolerance =
          std::max(kConstraintTolerance,
                   1e-12 * std::max(std::abs(physical(row)),
                                    std::abs(constraint.upper_bounds(row))));
      if (lower_side_active(constraint, row) &&
          physical(row) < constraint.lower_bounds(row) - lower_tolerance) {
        return false;
      }
      if (upper_side_active(constraint, row) &&
          physical(row) > constraint.upper_bounds(row) + upper_tolerance) {
        return false;
      }
    }
  }
  return true;
}

bool accepted_frozen_constraints_are_satisfied(
    const std::vector<FrozenNextVelocityConstraint> &constraints,
    const Eigen::VectorXd &dq, double dt, const Eigen::VectorXd &ddq) {
  for (const auto &constraint : constraints) {
    for (Eigen::Index row = 0; row < constraint.coefficient_matrix.rows();
         ++row) {
      double cdq = 0.0;
      double cddq = 0.0;
      double row_velocity_delta = 0.0;
      double physical = 0.0;
      if (!checked_dot_row(constraint.coefficient_matrix.row(row), dq, &cdq) ||
          !checked_dot_row(constraint.coefficient_matrix.row(row), ddq,
                           &cddq) ||
          !checked_add(cddq, constraint.affine_bias(row),
                       &row_velocity_delta) ||
          !checked_product(dt, row_velocity_delta, &row_velocity_delta) ||
          !checked_add(cdq, row_velocity_delta, &physical)) {
        return false;
      }
      const double lower_tolerance =
          std::max(kConstraintTolerance,
                   1e-12 * std::max(std::abs(physical),
                                    std::abs(constraint.lower_bounds(row))));
      const double upper_tolerance =
          std::max(kConstraintTolerance,
                   1e-12 * std::max(std::abs(physical),
                                    std::abs(constraint.upper_bounds(row))));
      if (lower_side_active(constraint, row) &&
          physical < constraint.lower_bounds(row) - lower_tolerance) {
        return false;
      }
      if (upper_side_active(constraint, row) &&
          physical > constraint.upper_bounds(row) + upper_tolerance) {
        return false;
      }
    }
  }
  return true;
}

bool accepted_lock_constraints_are_satisfied(
    const AccelerationSolveOptions &options, const Eigen::VectorXd &dq,
    double dt, const Eigen::VectorXd &ddq) {
  for (int index : options.zero_acceleration_joint_indices) {
    if (std::abs(ddq(index)) > kConstraintTolerance) {
      return false;
    }
  }
  for (int index : options.zero_next_velocity_joint_indices) {
    double velocity_delta = 0.0;
    double next_velocity = 0.0;
    if (!checked_product(dt, ddq(index), &velocity_delta) ||
        !checked_add(dq(index), velocity_delta, &next_velocity) ||
        std::abs(next_velocity) > kConstraintTolerance) {
      return false;
    }
  }
  for (int index : options.fixed_current_position_joint_indices) {
    double half_dt = 0.0;
    double half_acceleration_rate = 0.0;
    double average_velocity = 0.0;
    double displacement = 0.0;
    if (!checked_product(0.5, dt, &half_dt) ||
        !checked_product(half_dt, ddq(index), &half_acceleration_rate) ||
        !checked_add(dq(index), half_acceleration_rate, &average_velocity) ||
        !checked_product(dt, average_velocity, &displacement) ||
        std::abs(displacement) > kConstraintTolerance) {
      return false;
    }
  }
  return true;
}

} // namespace

AccelerationSolver::AccelerationSolver(std::shared_ptr<RobotModel> robot)
    : robot_(std::move(robot)) {
  if (!robot_) {
    throw std::invalid_argument("AccelerationSolver requires a RobotModel");
  }
  if (robot_->is_floating_base()) {
    throw std::invalid_argument(
        "AccelerationSolver v1 does not support floating-base models");
  }
  if (!fixed_base_scalar_joints_only(*robot_)) {
    throw std::invalid_argument(
        "AccelerationSolver v1 requires fixed-base joints with nq == nv == 1");
  }
}

AccelerationSolverCapabilities AccelerationSolver::capabilities() {
  return {};
}

std::shared_ptr<FrameTask>
AccelerationSolver::add_frame_task(const std::string &name,
                                   const std::string &frame_name,
                                   TaskType task_type) {
  ensure_unique_task_name(name);
  auto task = std::make_shared<FrameTask>(name, robot_, frame_name, task_type);
  tasks_.push_back(task);
  task_map_[name] = task;
  return task;
}

std::shared_ptr<COMTask>
AccelerationSolver::add_com_task(const std::string &name) {
  ensure_unique_task_name(name);
  auto task = std::make_shared<COMTask>(name, robot_);
  tasks_.push_back(task);
  task_map_[name] = task;
  return task;
}

std::shared_ptr<PostureTask>
AccelerationSolver::add_posture_task(const std::string &name,
                                      const std::vector<int> &controlled_joints) {
  ensure_unique_task_name(name);
  auto task = controlled_joints.empty()
                  ? std::make_shared<PostureTask>(name, robot_)
                  : std::make_shared<PostureTask>(name, robot_,
                                                  controlled_joints);
  tasks_.push_back(task);
  task_map_[name] = task;
  return task;
}

std::shared_ptr<JointTask>
AccelerationSolver::add_joint_task(const std::string &name,
                                   const std::string &joint_name,
                                   double target_value) {
  ensure_unique_task_name(name);
  auto task =
      std::make_shared<JointTask>(name, robot_, joint_name, target_value);
  tasks_.push_back(task);
  task_map_[name] = task;
  return task;
}

void AccelerationSolver::set_task_reference(
    const std::string &name, const AccelerationTaskReference &reference) {
  const auto found = task_map_.find(name);
  if (found == task_map_.end()) {
    throw std::invalid_argument("Unknown acceleration task: " + name);
  }
  const Eigen::Index dimension = found->second->getDimension();
  if ((reference.desired_velocity.size() != 0 &&
       reference.desired_velocity.size() != dimension) ||
      (reference.desired_acceleration.size() != 0 &&
       reference.desired_acceleration.size() != dimension)) {
    throw std::invalid_argument(
        "AccelerationTaskReference dimension must match task dimension");
  }
  if ((reference.desired_velocity.size() != 0 &&
       !reference.desired_velocity.allFinite()) ||
      (reference.desired_acceleration.size() != 0 &&
       !reference.desired_acceleration.allFinite()) ||
      !std::isfinite(reference.proportional_gain) ||
      !std::isfinite(reference.derivative_gain) ||
      reference.proportional_gain < 0.0 || reference.derivative_gain < 0.0) {
    throw std::invalid_argument(
        "AccelerationTaskReference values and gains must be finite");
  }
  task_references_[name] = reference;
}

std::shared_ptr<Task>
AccelerationSolver::get_task(const std::string &name) const {
  const auto found = task_map_.find(name);
  return found == task_map_.end() ? nullptr : found->second;
}

void AccelerationSolver::remove_task(const std::string &name) {
  const auto found = task_map_.find(name);
  if (found == task_map_.end()) {
    return;
  }
  tasks_.erase(std::remove(tasks_.begin(), tasks_.end(), found->second),
               tasks_.end());
  task_map_.erase(found);
  task_references_.erase(name);
}

void AccelerationSolver::clear_tasks() {
  tasks_.clear();
  task_map_.clear();
  task_references_.clear();
}

AccelerationSolverResult
AccelerationSolver::solve(const Eigen::VectorXd &q, const Eigen::VectorXd &dq,
                          double dt,
                          const AccelerationSolveOptions &options) {
  const auto start = std::chrono::steady_clock::now();
  const auto finish = [&start](AccelerationSolverResult result) {
    const auto end = std::chrono::steady_clock::now();
    result.computation_time_ms =
        std::chrono::duration<double, std::milli>(end - start).count();
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

  ObjectiveAssembly objectives;
  try {
    objectives =
        assemble_objectives(tasks_, task_references_, *robot_, dq);
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
  if (zero_excluded_by_bounds(state_box.lower, state_box.upper)) {
    for (auto &config : objectives.configs) {
      config.allow_min_error_fallback = true;
    }
  }

  detail::GeneralizedConstraintSet constraints(
      robot_->nv(),
      robot_->nv() + constraint_validation.row_count +
          task_bound_validation.row_count + contact_constraints.row_count +
          tight_point_constraints.row_count + lock_rows.coefficients.rows() +
          (effort_constraint.has_value() ? robot_->nv() : 0),
      false);
  detail::GeneralizedConstraintBlock joint_box;
  joint_box.coefficient_matrix =
      Eigen::MatrixXd::Identity(robot_->nv(), robot_->nv());
  joint_box.affine_bias = Eigen::VectorXd::Zero(robot_->nv());
  joint_box.physical_lower_bounds = state_box.lower;
  joint_box.physical_upper_bounds = state_box.upper;
  if (!constraints.append_block(std::move(joint_box))) {
    return finish(failure(SolverStatus::kNumericalError,
                          "failed to assemble acceleration constraints"));
  }
  if (effort_constraint.has_value() &&
      !constraints.append_block(
          make_effort_constraint_block(*effort_constraint, robot_->nv()))) {
    return finish(failure(SolverStatus::kNumericalError,
                          "failed to assemble effort constraints"));
  }
  if (!options.affine_constraints.empty() &&
      !constraints.append_block(make_affine_constraint_block(
          options.affine_constraints, robot_->nv(),
          count_constraint_rows(options.affine_constraints)))) {
    return finish(failure(
        SolverStatus::kNumericalError,
        "failed to assemble affine acceleration constraints"));
  }
  if (frozen_transform.row_count > 0 &&
      !constraints.append_block(make_affine_constraint_block(
          frozen_transform.transformed_constraints, robot_->nv(),
          frozen_transform.row_count))) {
    return finish(failure(
        SolverStatus::kNumericalError,
        "failed to assemble frozen next-velocity constraints"));
  }
  if (lock_rows.bounds.size() > 0) {
    if (!constraints.append_block(
            make_lock_constraint_block(lock_rows, robot_->nv()))) {
      return finish(failure(SolverStatus::kNumericalError,
                            "failed to assemble acceleration lock "
                            "constraints"));
    }
  }
  if (task_bound_validation.row_count > 0 &&
      !constraints.append_block(make_affine_constraint_block(
          task_bound_constraints, robot_->nv(),
          task_bound_validation.row_count))) {
    return finish(failure(
        SolverStatus::kNumericalError,
        "failed to assemble task acceleration bounds"));
  }
  if (contact_constraints.row_count > 0 &&
      !constraints.append_block(make_affine_constraint_block(
          contact_constraints.constraints, robot_->nv(),
          contact_constraints.row_count))) {
    return finish(failure(
        SolverStatus::kNumericalError,
        "failed to assemble contact acceleration constraints"));
  }
  if (tight_point_constraints.row_count > 0 &&
      !constraints.append_block(make_affine_constraint_block(
          tight_point_constraints.constraints, robot_->nv(),
          tight_point_constraints.row_count))) {
    return finish(failure(
        SolverStatus::kNumericalError,
        "failed to assemble tight point acceleration constraints"));
  }
  if (!constraints.finalize()) {
    return finish(failure(SolverStatus::kNumericalError,
                          "failed to assemble acceleration constraints"));
  }

  auto backend_targets = objectives.targets;
  auto backend_biases = objectives.biases;
  auto backend_matrices = objectives.matrices;
  Eigen::MatrixXd backend_constraint_matrix = constraints.coefficient_matrix();
  Eigen::VectorXd backend_lower_bounds = constraints.lower_bounds();
  Eigen::VectorXd backend_upper_bounds = constraints.upper_bounds();
  if (allocation.has_value()) {
    for (std::size_t objective_index = 0;
         objective_index < backend_matrices.size(); ++objective_index) {
      if (objectives.synthetic_hold_objective && objective_index == 0U) {
        backend_matrices[objective_index] =
            Eigen::MatrixXd::Identity(robot_->nv(), robot_->nv());
        backend_biases[objective_index] = Eigen::VectorXd::Zero(robot_->nv());
        backend_targets[objective_index] = Eigen::VectorXd::Zero(robot_->nv());
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
      backend_matrices[objective_index] = std::move(transformed_matrix.matrix);
      backend_biases[objective_index] = std::move(transformed_bias.vector);
    }
    const auto transformed_hard = allocation->transform_backend_hard_rows(
        constraints.coefficient_matrix(), constraints.lower_bounds(),
        constraints.upper_bounds());
    if (!transformed_hard.satisfied()) {
      return finish(failure(transformed_hard.status, transformed_hard.message));
    }
    backend_constraint_matrix = std::move(transformed_hard.coefficient_matrix);
    backend_lower_bounds = std::move(transformed_hard.lower_bounds);
    backend_upper_bounds = std::move(transformed_hard.upper_bounds);
  }

  auto backend = detail::SolveGeneralizedHierarchicalLinearSystemEigen(
      backend_targets, backend_biases, backend_matrices,
      backend_constraint_matrix, backend_lower_bounds, backend_upper_bounds,
      acceleration_backend_config(), objectives.configs, nullptr,
      kConstraintTolerance);

  AccelerationSolverResult result;
  static_cast<SolverResult &>(result) = std::move(backend);
  result.acceleration_limits_applied = true;
  result.effort_limits_applied = effort_constraint.has_value();
  if (result.status != SolverStatus::kSuccess) {
    return finish(clear_outputs(std::move(result)));
  }

  Eigen::VectorXd backend_solution = vector_from_solution(result);
  if (allocation.has_value()) {
    const auto physical =
        allocation->reconstruct_physical_acceleration(backend_solution);
    if (!physical.satisfied()) {
      return finish(failure(physical.status, physical.message));
    }
    result.joint_accelerations = physical.vector;
    assign_solution_from_vector(&result, result.joint_accelerations);
    if (!populate_allocation_diagnostics(&result, *allocation,
                                         result.joint_accelerations)) {
      return finish(clear_outputs(failure(
          SolverStatus::kNumericalError,
          "acceleration allocation diagnostics are not finite")));
    }
  } else {
    result.joint_accelerations = std::move(backend_solution);
  }
  if (result.joint_accelerations.size() != robot_->nv() ||
      !result.joint_accelerations.allFinite()) {
    return finish(failure(SolverStatus::kNumericalError,
                          "backend returned an invalid acceleration vector"));
  }
  if ((result.joint_accelerations.array() <
       state_box.lower.array() - kConstraintTolerance)
          .any() ||
      (result.joint_accelerations.array() >
       state_box.upper.array() + kConstraintTolerance)
          .any()) {
    return finish(clear_outputs(failure(
        SolverStatus::kNumericalError,
        "accepted acceleration violates the compatible state box")));
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
    return finish(clear_outputs(failure(
        SolverStatus::kNumericalError,
        std::string("accepted acceleration integration failed: ") +
            error.what())));
  }
  if (!result.joint_velocities_next.allFinite() ||
      !result.q_solution.allFinite()) {
    return finish(clear_outputs(failure(
        SolverStatus::kNumericalError,
        "accepted acceleration produced non-finite next state")));
  }
  if (!tight_point_constraints.prepared_constraints.empty()) {
    const auto predicted_state_box = build_joint_state_box(
        *robot_, result.q_solution, result.joint_velocities_next, dt,
        accel_limits, options);
    if (predicted_state_box.status != SolverStatus::kSuccess) {
      return finish(clear_outputs(failure(
          predicted_state_box.status,
          "accepted tight point constraints could not construct the predicted "
          "joint acceleration support box: " +
              predicted_state_box.message)));
    }
    for (const auto &prepared :
         tight_point_constraints.prepared_constraints) {
      const auto acceptance =
          detail::validate_tight_point_constraint_acceptance(
              prepared, *robot_, q, dq, result.joint_accelerations, dt,
              predicted_state_box.lower, predicted_state_box.upper);
      if (!acceptance.satisfied()) {
        return finish(clear_outputs(
            failure(acceptance.status, acceptance.message)));
      }
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
      return finish(clear_outputs(failure(
          SolverStatus::kNumericalError,
          "accepted acceleration violates next velocity limits")));
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
      return finish(clear_outputs(failure(
          SolverStatus::kNumericalError,
          "accepted acceleration violates next position limits")));
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
  double squared_error = 0.0;
  for (std::size_t group_index = 0; group_index < objectives.groups.size();
       ++group_index) {
    for (std::size_t task_index = 0;
         task_index < objectives.groups[group_index].size(); ++task_index) {
      const auto &task = objectives.groups[group_index][task_index];
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

      AccelerationTaskDiagnostics diagnostic;
      diagnostic.task_name = task->getName();
      diagnostic.scale = task_scale;
      diagnostic.effective_mode = task_mode;
      diagnostic.used_min_error_fallback = task_fallback;
      diagnostic.reference_acceleration =
          objectives.references[group_index][task_index];
      diagnostic.jacobian_bias =
          objectives.differentials[group_index][task_index].jacobian_bias;
      diagnostic.achieved_acceleration =
          objectives.differentials[group_index][task_index].physical_jacobian *
          result.joint_accelerations;
      diagnostic.residual =
          diagnostic.achieved_acceleration + diagnostic.jacobian_bias -
          diagnostic.reference_acceleration;
      const double task_error = diagnostic.residual.norm();
      result.task_errors.push_back(task_error);
      squared_error += diagnostic.residual.squaredNorm();
      task->setLastEffectiveMode(task_mode);
      task->setUsedMinErrorFallback(task_fallback);
      result.task_diagnostics.push_back(std::move(diagnostic));
    }
  }
  result.final_error = std::sqrt(squared_error);
  return finish(std::move(result));
}

void AccelerationSolver::ensure_unique_task_name(
    const std::string &name) const {
  if (name.empty()) {
    throw std::invalid_argument("Task name must not be empty");
  }
  if (task_map_.count(name)) {
    throw std::invalid_argument("Duplicate task name: " + name);
  }
}

} // namespace embodik
