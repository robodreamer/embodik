#include <embodik/acceleration_solver.hpp>

#include "acceleration_analytic_collision.hpp"
#include "acceleration_allocation_transform.hpp"
#include "acceleration_com_support_polygon_constraint.hpp"
#include "acceleration_fixed_frame_pose_constraint.hpp"
#include "acceleration_relative_pose_constraint.hpp"
#include "acceleration_state_box.hpp"
#include "acceleration_task_differential.hpp"
#include "acceleration_tight_point_constraint.hpp"
#include "frame_kinematic_differential.hpp"
#include "generalized_constraint_set.hpp"

#include <embodik/ik_baseline.hpp>
#include <embodik/kinematics_solver.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
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

bool native_collision_options_are_supported(
    const AccelerationSolveOptions &options, std::string *message) {
  const bool unsupported =
      options.effort_constraints.has_value() ||
      !options.affine_constraints.empty() ||
      !options.frozen_next_velocity_constraints.empty() ||
      !options.task_acceleration_bounds.empty() ||
      !options.contact_acceleration_constraints.empty() ||
      !options.tight_point_constraints.empty() ||
      !options.tight_frame_pose_constraints.empty() ||
      !options.relative_pose_constraints.empty() ||
      !options.torso_pose_bound_constraints.empty() ||
      !options.com_support_polygon_constraints.empty() ||
      !options.zero_acceleration_joint_indices.empty() ||
      !options.zero_next_velocity_joint_indices.empty() ||
      !options.fixed_current_position_joint_indices.empty();
  if (unsupported && message != nullptr) {
    *message =
        "native collision certification currently permits joint state boxes, "
        "generalized allocation, and soft tasks only; another hard family "
        "lacks a predicted-state certificate provider";
  }
  return !unsupported;
}

detail::CompiledAnalyticCollisionConstraint make_stored_collision_compilation(
    const std::vector<std::size_t> &pair_indices,
    const std::vector<double> &minimum_distances,
    const std::vector<CollisionGeometryPair> &active_pairs) {
  detail::CompiledAnalyticCollisionConstraint compiled;
  compiled.pair_indices = pair_indices;
  compiled.minimum_distances = minimum_distances;
  compiled.active_pairs = active_pairs;
  return compiled;
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

const char *
constraint_family_name(const TightFramePoseAccelerationConstraint &) {
  return "tight frame pose constraint";
}

const char *constraint_family_name(const RelativePoseAccelerationConstraint &) {
  return "relative pose constraint";
}

const char *
constraint_family_name(const TorsoPoseBoundAccelerationConstraint &) {
  return "torso pose bound constraint";
}

const char *
constraint_family_name(const ComSupportPolygonAccelerationConstraint &) {
  return "CoM support-polygon constraint";
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

struct FixedFramePoseConstraintAssembly {
  SolverStatus status = SolverStatus::kSuccess;
  std::string message;
  std::vector<AffineAccelerationConstraint> constraints;
  std::vector<detail::PreparedFixedFramePoseConstraint> prepared_constraints;
  Eigen::Index row_count = 0;
};

struct RelativePoseConstraintAssembly {
  SolverStatus status = SolverStatus::kSuccess;
  std::string message;
  std::vector<AffineAccelerationConstraint> constraints;
  std::vector<detail::PreparedRelativePoseConstraint> prepared_constraints;
  Eigen::Index row_count = 0;
};

struct ComSupportPolygonConstraintAssembly {
  SolverStatus status = SolverStatus::kSuccess;
  std::string message;
  std::vector<AffineAccelerationConstraint> constraints;
  std::vector<detail::PreparedComSupportPolygonConstraint>
      prepared_constraints;
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

template <typename Constraint, typename RecordMaker>
FixedFramePoseConstraintAssembly make_fixed_frame_pose_constraints(
    const RobotModel &robot, const std::vector<Constraint> &pose_constraints,
    double dt, const Eigen::VectorXd &joint_acceleration_lower,
    const Eigen::VectorXd &joint_acceleration_upper,
    std::unordered_set<std::string> *source_ids, RecordMaker make_record) {
  FixedFramePoseConstraintAssembly assembly;
  assembly.constraints.reserve(pose_constraints.size());
  assembly.prepared_constraints.reserve(pose_constraints.size());

  for (const auto &pose_constraint : pose_constraints) {
    const std::string family = constraint_family_name(pose_constraint);
    if (pose_constraint.source_id.empty()) {
      assembly.status = SolverStatus::kInvalidInput;
      assembly.message = family + " source_id must not be empty";
      return assembly;
    }
    if (!source_ids->insert(pose_constraint.source_id).second) {
      assembly.status = SolverStatus::kInvalidInput;
      assembly.message = "duplicate acceleration constraint source_id '" +
                         pose_constraint.source_id + "'";
      return assembly;
    }

    detail::FixedFramePoseConstraintRecord record;
    const auto record_result = make_record(pose_constraint, &record);
    if (!record_result.satisfied()) {
      assembly.status = record_result.status;
      assembly.message = record_result.message;
      return assembly;
    }
    const auto prepared = detail::prepare_fixed_frame_pose_constraint(
        record, robot, dt, joint_acceleration_lower,
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

RelativePoseConstraintAssembly make_relative_pose_constraints(
    const RobotModel &robot,
    const std::vector<RelativePoseAccelerationConstraint> &relative_poses,
    double dt, const Eigen::VectorXd &joint_acceleration_lower,
    const Eigen::VectorXd &joint_acceleration_upper,
    std::unordered_set<std::string> *source_ids) {
  RelativePoseConstraintAssembly assembly;
  assembly.constraints.reserve(relative_poses.size());
  assembly.prepared_constraints.reserve(relative_poses.size());

  for (const auto &relative_pose : relative_poses) {
    const std::string family = constraint_family_name(relative_pose);
    if (relative_pose.source_id.empty()) {
      assembly.status = SolverStatus::kInvalidInput;
      assembly.message = family + " source_id must not be empty";
      return assembly;
    }
    if (!source_ids->insert(relative_pose.source_id).second) {
      assembly.status = SolverStatus::kInvalidInput;
      assembly.message = "duplicate acceleration constraint source_id '" +
                         relative_pose.source_id + "'";
      return assembly;
    }

    const auto prepared = detail::prepare_relative_pose_constraint(
        relative_pose, robot, dt, joint_acceleration_lower,
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

ComSupportPolygonConstraintAssembly make_com_support_polygon_constraints(
    const RobotModel &robot,
    const std::vector<ComSupportPolygonAccelerationConstraint> &polygons,
    double dt, const Eigen::VectorXd &joint_acceleration_lower,
    const Eigen::VectorXd &joint_acceleration_upper,
    std::unordered_set<std::string> *source_ids) {
  ComSupportPolygonConstraintAssembly assembly;
  assembly.constraints.reserve(polygons.size());
  assembly.prepared_constraints.reserve(polygons.size());

  for (const auto &polygon : polygons) {
    const std::string family = constraint_family_name(polygon);
    if (polygon.source_id.empty()) {
      assembly.status = SolverStatus::kInvalidInput;
      assembly.message = family + " source_id must not be empty";
      return assembly;
    }
    if (!source_ids->insert(polygon.source_id).second) {
      assembly.status = SolverStatus::kInvalidInput;
      assembly.message = "duplicate acceleration constraint source_id '" +
                         polygon.source_id + "'";
      return assembly;
    }

    const auto prepared = detail::prepare_com_support_polygon_constraint(
        polygon, robot, dt, joint_acceleration_lower,
        joint_acceleration_upper);
    if (!prepared.satisfied() || !prepared.prepared.has_value()) {
      assembly.status = prepared.status;
      assembly.message = prepared.message;
      return assembly;
    }
    assembly.row_count +=
        prepared.prepared->state_box.physical_constraint.coefficient_matrix
            .rows();
    assembly.constraints.push_back(
        prepared.prepared->state_box.physical_constraint);
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

struct VelocityCollisionLiftDiagnostics {
  bool applied = false;
  bool endpoint_validated = false;
  std::uint64_t validation_samples = 0;
  std::uint64_t validation_allowed_pairs = 0;
  std::uint64_t validation_pairs_checked = 0;
  std::uint64_t validation_exact_queries = 0;
  std::uint64_t validation_initial_exact_queries = 0;
  std::uint64_t validation_sample_exact_queries = 0;
  std::uint64_t validation_conservative_checks = 0;
  std::uint64_t validation_conservative_certified_pairs = 0;
  std::uint64_t validation_kinematics_updates = 0;
  std::uint64_t validation_geometry_updates = 0;
  bool validation_initial_certificate_reused = false;
  std::uint64_t pairs_considered = 0;
  std::uint64_t row_pairs = 0;
  std::uint64_t row_exact_queries = 0;
};

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

} // namespace

namespace detail {

struct AccelerationSolverWorkspace {
  ObjectiveAssembly objectives;
  Eigen::MatrixXd state_box_identity;
};

} // namespace detail

AccelerationSolver::AccelerationSolver(std::shared_ptr<RobotModel> robot)
    : robot_(std::move(robot)),
      workspace_(std::make_unique<detail::AccelerationSolverWorkspace>()) {
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

AccelerationSolver::~AccelerationSolver() = default;
AccelerationSolver::AccelerationSolver(AccelerationSolver &&) noexcept =
    default;
AccelerationSolver &
AccelerationSolver::operator=(AccelerationSolver &&) noexcept = default;

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
  ordered_task_scratch_.clear();
  task_map_[name] = task;
  return task;
}

std::shared_ptr<COMTask>
AccelerationSolver::add_com_task(const std::string &name) {
  ensure_unique_task_name(name);
  auto task = std::make_shared<COMTask>(name, robot_);
  tasks_.push_back(task);
  ordered_task_scratch_.clear();
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
  ordered_task_scratch_.clear();
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
  ordered_task_scratch_.clear();
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
  ordered_task_scratch_.clear();
  task_map_.erase(found);
  task_references_.erase(name);
}

void AccelerationSolver::clear_tasks() {
  tasks_.clear();
  ordered_task_scratch_.clear();
  task_map_.clear();
  task_references_.clear();
}

void AccelerationSolver::configure_collision_constraint(
    const CollisionConstraintDefinition &definition,
    const CollisionConstraintAccelerationPolicy &policy) {
  const auto compiled = detail::compile_analytic_collision_constraint(
      *robot_, definition, policy);
  if (!compiled.satisfied()) {
    throw std::invalid_argument(compiled.message);
  }

  native_collision_definition_ = definition;
  native_collision_policy_ = policy;
  native_collision_active_pairs_ = compiled.active_pairs;
  native_collision_pair_indices_ = compiled.pair_indices;
  native_collision_minimum_distances_ = compiled.minimum_distances;
}

void AccelerationSolver::clear_collision_constraint() {
  native_collision_definition_.reset();
  native_collision_policy_.reset();
  native_collision_active_pairs_.clear();
  native_collision_pair_indices_.clear();
  native_collision_minimum_distances_.clear();
}

bool AccelerationSolver::has_collision_constraint() const {
  return native_collision_definition_.has_value() &&
         native_collision_policy_.has_value() &&
         !native_collision_active_pairs_.empty();
}

double AccelerationSolver::get_collision_min_distance() const {
  return native_collision_definition_.has_value()
             ? native_collision_definition_->min_distance
             : -1.0;
}

std::optional<CollisionConstraintDefinition>
AccelerationSolver::get_collision_constraint_definition() const {
  return native_collision_definition_;
}

std::optional<CollisionConstraintAccelerationPolicy>
AccelerationSolver::get_collision_constraint_policy() const {
  return native_collision_policy_;
}

std::vector<CollisionGeometryPair>
AccelerationSolver::get_active_collision_pairs() const {
  return native_collision_active_pairs_;
}

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
  if (collision_solver.robot_.get() != robot_.get()) {
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
  if (!collision_solver.collision_constraint_.has_value() ||
      !collision_solver.collision_constraint_->enabled) {
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

  std::optional<KinematicsSolver::CollisionVelocityConstraintLinearization>
      linearization;
  try {
    linearization =
        collision_solver.linearize_collision_velocity_constraint(dt);
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
  diagnostics.pairs_considered =
      collision_solver.last_collision_pairs_considered_;
  diagnostics.row_exact_queries =
      collision_solver.last_collision_exact_distance_queries_;
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
      collision_solver.validate_collision_samples(q, validation_samples);
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
