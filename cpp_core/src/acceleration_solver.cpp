#include <embodik/acceleration_solver.hpp>

#include "acceleration_state_box.hpp"
#include "acceleration_task_differential.hpp"
#include "generalized_constraint_set.hpp"

#include <embodik/ik_baseline.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <map>
#include <stdexcept>
#include <utility>

namespace embodik {
namespace {

constexpr double kConstraintTolerance = 1e-8;

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

Eigen::VectorXd vector_from_solution(const SolverResult &result) {
  if (result.solution.empty()) {
    return {};
  }
  return Eigen::Map<const Eigen::VectorXd>(result.solution.data(),
                                           result.solution.size());
}

bool zero_excluded_by_bounds(const Eigen::VectorXd &lower,
                             const Eigen::VectorXd &upper) {
  return (lower.array() > kConstraintTolerance).any() ||
         (upper.array() < -kConstraintTolerance).any();
}

AccelerationSolverResult clear_outputs(AccelerationSolverResult result) {
  result.solution.clear();
  result.joint_accelerations.resize(0);
  result.joint_velocities_next.resize(0);
  result.q_solution.resize(0);
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
  if (zero_excluded_by_bounds(state_box.lower, state_box.upper)) {
    for (auto &config : objectives.configs) {
      config.allow_min_error_fallback = true;
    }
  }

  detail::GeneralizedConstraintSet constraints(
      robot_->nv(), robot_->nv(), false);
  detail::GeneralizedConstraintBlock joint_box;
  joint_box.coefficient_matrix =
      Eigen::MatrixXd::Identity(robot_->nv(), robot_->nv());
  joint_box.affine_bias = Eigen::VectorXd::Zero(robot_->nv());
  joint_box.physical_lower_bounds = state_box.lower;
  joint_box.physical_upper_bounds = state_box.upper;
  if (!constraints.append_block(std::move(joint_box)) ||
      !constraints.finalize()) {
    return finish(failure(SolverStatus::kNumericalError,
                          "failed to assemble acceleration constraints"));
  }

  auto backend = detail::SolveGeneralizedHierarchicalLinearSystemEigen(
      objectives.targets, objectives.biases, objectives.matrices,
      constraints.coefficient_matrix(), constraints.lower_bounds(),
      constraints.upper_bounds(), acceleration_backend_config(),
      objectives.configs, nullptr, kConstraintTolerance);

  AccelerationSolverResult result;
  static_cast<SolverResult &>(result) = std::move(backend);
  result.acceleration_limits_applied = true;
  if (result.status != SolverStatus::kSuccess) {
    return finish(clear_outputs(std::move(result)));
  }

  result.joint_accelerations = vector_from_solution(result);
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
