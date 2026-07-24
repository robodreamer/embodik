#include "acceleration_solver_internal.hpp"

namespace embodik {
using namespace acceleration_solver_internal;

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
