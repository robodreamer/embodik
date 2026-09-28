/**
 * @file kinematics_solver_tasks.cpp
 * @brief Task registry and ordering for KinematicsSolver
 */

#include <algorithm>
#include <stdexcept>
#include <unordered_set>

#include <embodik/kinematics_solver.hpp>
#include <embodik/tasks.hpp>

namespace embodik {

std::shared_ptr<FrameTask>
KinematicsSolver::add_frame_task(const std::string &name,
                                 const std::string &frame_name,
                                 TaskType task_type) {
  if (task_map_.find(name) != task_map_.end()) {
    throw std::runtime_error("Task with name '" + name + "' already exists");
  }

  auto task = std::make_shared<FrameTask>(name, robot_, frame_name, task_type);
  tasks_.push_back(task);
  task_map_[name] = task;

  return task;
}

std::shared_ptr<COMTask>
KinematicsSolver::add_com_task(const std::string &name) {
  if (task_map_.find(name) != task_map_.end()) {
    throw std::runtime_error("Task with name '" + name + "' already exists");
  }

  auto task = std::make_shared<COMTask>(name, robot_);
  tasks_.push_back(task);
  task_map_[name] = task;

  return task;
}

std::shared_ptr<CentroidalMomentumTask>
KinematicsSolver::add_centroidal_momentum_task(const std::string &name) {
  if (task_map_.find(name) != task_map_.end()) {
    throw std::runtime_error("Task with name '" + name + "' already exists");
  }

  auto task = std::make_shared<CentroidalMomentumTask>(name, robot_);
  tasks_.push_back(task);
  task_map_[name] = task;

  return task;
}

std::shared_ptr<PostureTask>
KinematicsSolver::add_posture_task(const std::string &name,
                                   const std::vector<int> &controlled_joints) {
  if (task_map_.find(name) != task_map_.end()) {
    throw std::runtime_error("Task with name '" + name + "' already exists");
  }

  std::shared_ptr<PostureTask> task;
  if (controlled_joints.empty()) {
    task = std::make_shared<PostureTask>(name, robot_);
  } else {
    task = std::make_shared<PostureTask>(name, robot_, controlled_joints);
  }

  tasks_.push_back(task);
  task_map_[name] = task;

  return task;
}

std::shared_ptr<ManipulabilityTask>
KinematicsSolver::add_manipulability_task(const std::string &name,
                                          const std::string &frame_name,
                                          TaskType frame_task_type) {
  if (task_map_.find(name) != task_map_.end()) {
    throw std::runtime_error("Task with name '" + name + "' already exists");
  }

  auto task = std::make_shared<ManipulabilityTask>(
      name, robot_, frame_name, frame_task_type);
  tasks_.push_back(task);
  task_map_[name] = task;
  return task;
}

std::shared_ptr<JointLimitAvoidanceTask>
KinematicsSolver::add_joint_limit_avoidance_task(
    const std::string &name,
    const std::vector<int> &controlled_joint_indices) {
  if (task_map_.find(name) != task_map_.end()) {
    throw std::runtime_error("Task with name '" + name + "' already exists");
  }

  auto task = std::make_shared<JointLimitAvoidanceTask>(
      name, robot_, controlled_joint_indices);
  tasks_.push_back(task);
  task_map_[name] = task;
  return task;
}

std::shared_ptr<JointTask>
KinematicsSolver::add_joint_task(const std::string &name,
                                 const std::string &joint_name,
                                 double target_value) {
  if (task_map_.find(name) != task_map_.end()) {
    throw std::runtime_error("Task with name '" + name + "' already exists");
  }

  auto task =
      std::make_shared<JointTask>(name, robot_, joint_name, target_value);
  tasks_.push_back(task);
  task_map_[name] = task;

  return task;
}

std::shared_ptr<RelativeFrameTask>
KinematicsSolver::add_relative_frame_task(const std::string &name,
                                          const std::string &frame_a,
                                          const std::string &frame_b) {
  if (task_map_.find(name) != task_map_.end()) {
    throw std::runtime_error("Task with name '" + name + "' already exists");
  }

  auto task =
      std::make_shared<RelativeFrameTask>(name, robot_, frame_a, frame_b);
  tasks_.push_back(task);
  task_map_[name] = task;
  return task;
}

std::shared_ptr<AbsoluteFrameTask>
KinematicsSolver::add_absolute_frame_task(const std::string &name,
                                          const std::string &frame_a,
                                          const std::string &frame_b,
                                          double alpha) {
  if (task_map_.find(name) != task_map_.end()) {
    throw std::runtime_error("Task with name '" + name + "' already exists");
  }

  auto task =
      std::make_shared<AbsoluteFrameTask>(name, robot_, frame_a, frame_b, alpha);
  tasks_.push_back(task);
  task_map_[name] = task;
  return task;
}

std::shared_ptr<PoseTaskGroup>
KinematicsSolver::add_pose_task_group(const std::string &name,
                                      const std::string &tcp_frame,
                                      int base_priority,
                                      int rotation_priority_offset,
                                      bool merged_pose, bool auto_switch) {
  if (pose_task_groups_.find(name) != pose_task_groups_.end()) {
    throw std::runtime_error("Pose task group with name '" + name +
                             "' already exists");
  }
  if (merged_pose && auto_switch) {
    throw std::runtime_error("Pose task group '" + name +
                             "' cannot use merged_pose and auto_switch "
                             "together");
  }

  const auto position_name = name + "__position";
  const auto orientation_name = name + "__orientation";
  const auto task_name_conflicts = [&](const std::string &task_name) {
    return task_map_.find(task_name) != task_map_.end();
  };
  if (task_name_conflicts(name) ||
      (!merged_pose && task_name_conflicts(position_name)) ||
      (!merged_pose && task_name_conflicts(orientation_name))) {
    throw std::runtime_error("Pose task group '" + name +
                             "' conflicts with an existing task name");
  }

  std::shared_ptr<PoseTaskGroup> group;
  if (auto_switch) {
    auto merged_task = std::make_shared<FrameTask>(name, robot_, tcp_frame,
                                                   TaskType::FRAME_POSE);
    auto position_task = std::make_shared<FrameTask>(
        position_name, robot_, tcp_frame, TaskType::FRAME_POSITION);
    auto orientation_task = std::make_shared<FrameTask>(
        orientation_name, robot_, tcp_frame, TaskType::FRAME_ORIENTATION);
    merged_task->setPriority(base_priority);
    position_task->setPriority(base_priority);
    orientation_task->setPriority(base_priority + rotation_priority_offset);
    tasks_.push_back(merged_task);
    task_map_[name] = merged_task;
    tasks_.push_back(position_task);
    task_map_[position_name] = position_task;
    tasks_.push_back(orientation_task);
    task_map_[orientation_name] = orientation_task;
    group = std::make_shared<PoseTaskGroup>(
        name, tcp_frame, base_priority, rotation_priority_offset, position_task,
        orientation_task, merged_task);
  } else if (merged_pose) {
    auto pose_task = std::make_shared<FrameTask>(name, robot_, tcp_frame,
                                                 TaskType::FRAME_POSE);
    pose_task->setPriority(base_priority);
    tasks_.push_back(pose_task);
    task_map_[name] = pose_task;
    group = std::make_shared<PoseTaskGroup>(name, tcp_frame, base_priority,
                                            pose_task);
  } else {
    auto position_task = std::make_shared<FrameTask>(
        position_name, robot_, tcp_frame, TaskType::FRAME_POSITION);
    auto orientation_task = std::make_shared<FrameTask>(
        orientation_name, robot_, tcp_frame, TaskType::FRAME_ORIENTATION);
    position_task->setPriority(base_priority);
    orientation_task->setPriority(base_priority + rotation_priority_offset);
    tasks_.push_back(position_task);
    task_map_[position_name] = position_task;
    tasks_.push_back(orientation_task);
    task_map_[orientation_name] = orientation_task;
    group = std::make_shared<PoseTaskGroup>(
        name, tcp_frame, base_priority, rotation_priority_offset, position_task,
        orientation_task);
  }

  pose_task_groups_[name] = group;
  sort_tasks_by_priority();
  return group;
}

std::shared_ptr<PoseTaskGroup>
KinematicsSolver::pose_task_group(const std::string &name) const {
  auto it = pose_task_groups_.find(name);
  return (it != pose_task_groups_.end()) ? it->second : nullptr;
}

std::optional<std::string> KinematicsSolver::task_stack_validation_error(
    const TaskStackConfig &config) const {
  if (config.levels.empty()) {
    return "explicit task stack must contain at least one level";
  }
  switch (config.backend) {
  case TaskStackBackend::kSns:
  case TaskStackBackend::kLexicographicLeastSquares:
    break;
  default:
    return "explicit task stack has an unsupported backend";
  }

  std::unordered_set<std::string> level_names;
  std::unordered_set<std::string> task_names;
  for (std::size_t level_index = 0; level_index < config.levels.size();
       ++level_index) {
    const auto &level = config.levels[level_index];
    if (level.name.empty()) {
      return "explicit task-stack level " + std::to_string(level_index) +
             " has an empty name";
    }
    if (!level_names.insert(level.name).second) {
      return "duplicate explicit task-stack level name '" + level.name + "'";
    }
    if (level.task_names.empty()) {
      return "explicit task-stack level '" + level.name +
             "' must contain at least one task";
    }
    switch (level.solve_mode) {
    case TaskSolveMode::kScale:
    case TaskSolveMode::kScaleElastic:
      break;
    case TaskSolveMode::kMinError:
      if (level.allow_min_error_fallback) {
        return "explicit task-stack level '" + level.name +
               "' cannot enable MIN_ERROR fallback when its solve mode is "
               "already MIN_ERROR";
      }
      break;
    default:
      return "explicit task-stack level '" + level.name +
             "' has an unsupported solve mode";
    }
    if (config.backend == TaskStackBackend::kLexicographicLeastSquares) {
      if (level.solve_mode == TaskSolveMode::kScaleElastic) {
        return "explicit lexicographic task-stack level '" + level.name +
               "' cannot use SCALE_ELASTIC";
      }
      if (level.allow_min_error_fallback) {
        return "explicit lexicographic task-stack level '" + level.name +
               "' cannot enable an SNS MIN_ERROR fallback policy";
      }
    }

    for (const auto &task_name : level.task_names) {
      if (task_name.empty()) {
        return "explicit task-stack level '" + level.name +
               "' contains an empty task name";
      }
      if (!task_names.insert(task_name).second) {
        return "task '" + task_name +
               "' appears more than once in the explicit task stack";
      }
      const auto task_it = task_map_.find(task_name);
      if (task_it == task_map_.end() || !task_it->second) {
        return "explicit task-stack task '" + task_name +
               "' is not registered";
      }
      if (!task_it->second->isActive()) {
        return "explicit task-stack task '" + task_name + "' is inactive";
      }
    }
  }
  return std::nullopt;
}

void KinematicsSolver::configure_task_stack(const TaskStackConfig &config) {
  if (const auto error = task_stack_validation_error(config);
      error.has_value()) {
    throw std::invalid_argument(*error);
  }

  TaskStackConfig normalized = config;
  for (auto &level : normalized.levels) {
    std::sort(level.task_names.begin(), level.task_names.end());
  }
  task_stack_config_ = std::move(normalized);
  reset_position_step_continuity_state();
}

void KinematicsSolver::clear_task_stack() {
  if (task_stack_config_.has_value()) {
    task_stack_config_.reset();
    reset_position_step_continuity_state();
  }
}

void KinematicsSolver::remove_task(const std::string &name) {
  auto group_it = pose_task_groups_.find(name);
  if (group_it != pose_task_groups_.end()) {
    auto group = group_it->second;
    pose_task_groups_.erase(group_it);
    if (group && group->position_task()) {
      remove_task(group->position_task()->getName());
    }
    if (group && group->orientation_task()) {
      remove_task(group->orientation_task()->getName());
    }
    if (group && group->merged_task()) {
      remove_task(group->merged_task()->getName());
    }
    return;
  }

  auto it = task_map_.find(name);
  if (it != task_map_.end()) {
    reset_position_step_continuity_state();
    auto task = it->second;
    task_map_.erase(it);

    tasks_.erase(std::remove(tasks_.begin(), tasks_.end(), task), tasks_.end());
    for (auto group_it = pose_task_groups_.begin();
         group_it != pose_task_groups_.end();) {
      const auto &group = group_it->second;
      const bool references_removed_task =
          group &&
          ((group->position_task() &&
            group->position_task()->getName() == name) ||
           (group->orientation_task() &&
            group->orientation_task()->getName() == name) ||
           (group->merged_task() && group->merged_task()->getName() == name));
      if (references_removed_task) {
        group_it = pose_task_groups_.erase(group_it);
      } else {
        ++group_it;
      }
    }
  }
}

void KinematicsSolver::clear_tasks() {
  tasks_.clear();
  task_map_.clear();
  pose_task_groups_.clear();
  reset_position_step_continuity_state();
}

void KinematicsSolver::clear_all_target_velocities() {
  for (auto &task : tasks_) {
    if (task) {
      task->clearTargetVelocity();
    }
  }
}

std::shared_ptr<Task> KinematicsSolver::get_task(const std::string &name) {
  auto it = task_map_.find(name);
  return (it != task_map_.end()) ? it->second : nullptr;
}

void KinematicsSolver::sort_tasks_by_priority() {
  if (std::is_sorted(
          tasks_.begin(), tasks_.end(),
          [](const std::shared_ptr<Task> &a, const std::shared_ptr<Task> &b) {
            return a->getPriority() <= b->getPriority();
          })) {
    return;
  }
  std::stable_sort(
      tasks_.begin(), tasks_.end(),
      [](const std::shared_ptr<Task> &a, const std::shared_ptr<Task> &b) {
        return a->getPriority() < b->getPriority();
      });
}

} // namespace embodik
