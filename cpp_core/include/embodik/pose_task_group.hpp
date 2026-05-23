/**
 * @file pose_task_group.hpp
 * @brief Small adapter for coordinated pose task target bookkeeping.
 */

#pragma once

#include <Eigen/Dense>
#include <embodik/tasks.hpp>
#include <embodik/types.hpp>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace embodik {

/**
 * @brief Owns either a split position/orientation FrameTask pair or one merged
 *        pose FrameTask behind a single target API.
 *
 * The adapter deliberately does not change solver math. It only creates the
 * same FrameTask objects a caller could already create manually, then emits the
 * corresponding TaskTarget list for solve_position_step().
 */
class PoseTaskGroup {
public:
  PoseTaskGroup(std::string name, std::string tcp_frame, int base_priority,
                int rotation_priority_offset,
                std::shared_ptr<FrameTask> position_task,
                std::shared_ptr<FrameTask> orientation_task)
      : name_(std::move(name)), tcp_frame_(std::move(tcp_frame)),
        base_priority_(base_priority),
        rotation_priority_offset_(rotation_priority_offset),
        merged_pose_(false), position_task_(std::move(position_task)),
        orientation_task_(std::move(orientation_task)) {}

  PoseTaskGroup(std::string name, std::string tcp_frame, int base_priority,
                std::shared_ptr<FrameTask> pose_task)
      : name_(std::move(name)), tcp_frame_(std::move(tcp_frame)),
        base_priority_(base_priority), rotation_priority_offset_(0),
        merged_pose_(true), position_task_(std::move(pose_task)),
        orientation_task_(nullptr) {}

  PoseTaskGroup(std::string name, std::string tcp_frame, int base_priority,
                int rotation_priority_offset,
                std::shared_ptr<FrameTask> position_task,
                std::shared_ptr<FrameTask> orientation_task,
                std::shared_ptr<FrameTask> merged_task)
      : name_(std::move(name)), tcp_frame_(std::move(tcp_frame)),
        base_priority_(base_priority),
        rotation_priority_offset_(rotation_priority_offset),
        merged_pose_(false), auto_switch_(true),
        current_layout_(TaskLayout::kMerged),
        position_task_(std::move(position_task)),
        orientation_task_(std::move(orientation_task)),
        merged_task_(std::move(merged_task)) {
    apply_layout(current_layout_);
  }

  const std::string &name() const { return name_; }
  const std::string &tcp_frame() const { return tcp_frame_; }
  int base_priority() const { return base_priority_; }
  int rotation_priority_offset() const { return rotation_priority_offset_; }
  bool merged_pose() const { return merged_pose_; }
  bool auto_switch() const { return auto_switch_; }
  TaskLayout current_layout() const {
    return auto_switch_ ? current_layout_
                        : (merged_pose_ ? TaskLayout::kMerged
                                        : TaskLayout::kSplit);
  }
  std::shared_ptr<FrameTask> position_task() const { return position_task_; }
  std::shared_ptr<FrameTask> orientation_task() const {
    return orientation_task_;
  }
  std::shared_ptr<FrameTask> merged_task() const { return merged_task_; }

  void set_layout(TaskLayout layout) {
    if (!auto_switch_ ||
        (layout != TaskLayout::kMerged && layout != TaskLayout::kSplit) ||
        layout == current_layout_) {
      return;
    }
    apply_layout(layout);
    current_layout_ = layout;
  }

  void set_active(bool active) {
    if (auto_switch_) {
      if (active) {
        apply_layout(current_layout_);
      } else {
        if (position_task_) {
          position_task_->setActive(false);
        }
        if (orientation_task_) {
          orientation_task_->setActive(false);
        }
        if (merged_task_) {
          merged_task_->setActive(false);
        }
      }
      return;
    }
    if (position_task_) {
      position_task_->setActive(active);
    }
    if (orientation_task_) {
      orientation_task_->setActive(active);
    }
  }

  void set_solve_mode(TaskSolveMode mode) {
    if (position_task_) {
      position_task_->setSolveMode(mode);
    }
    if (orientation_task_) {
      orientation_task_->setSolveMode(mode);
    }
    if (merged_task_) {
      merged_task_->setSolveMode(mode);
    }
  }

  void set_allow_min_error_fallback(bool allow) {
    if (position_task_) {
      position_task_->setAllowMinErrorFallback(allow);
    }
    if (orientation_task_) {
      orientation_task_->setAllowMinErrorFallback(allow);
    }
    if (merged_task_) {
      merged_task_->setAllowMinErrorFallback(allow);
    }
  }

  void set_target(const Eigen::Matrix4d &target_pose, double position_gain,
                  double rotation_gain) {
    last_target_ = target_pose;
    last_position_gain_ = position_gain;
    last_rotation_gain_ = rotation_gain;
    target_set_ = true;
  }

  std::vector<TaskTarget> task_targets() const {
    if (!target_set_) {
      throw std::runtime_error("PoseTaskGroup '" + name_ +
                               "': set_target() must be called before "
                               "task_targets()");
    }

    if (auto_switch_) {
      return {TaskTarget{merged_task_->getName(), last_target_,
                         last_position_gain_, last_rotation_gain_},
              TaskTarget{position_task_->getName(), last_target_,
                         last_position_gain_, 0.0},
              TaskTarget{orientation_task_->getName(), last_target_, 0.0,
                         last_rotation_gain_}};
    }

    if (merged_pose_) {
      return {TaskTarget{position_task_->getName(), last_target_,
                         last_position_gain_, last_rotation_gain_}};
    }

    return {TaskTarget{position_task_->getName(), last_target_,
                       last_position_gain_, 0.0},
            TaskTarget{orientation_task_->getName(), last_target_, 0.0,
                       last_rotation_gain_}};
  }

  const Eigen::Matrix4d &last_target() const { return last_target_; }
  double last_position_gain() const { return last_position_gain_; }
  double last_rotation_gain() const { return last_rotation_gain_; }
  bool target_set() const { return target_set_; }

private:
  void apply_layout(TaskLayout layout) {
    if (!auto_switch_) {
      return;
    }
    const bool use_merged = layout == TaskLayout::kMerged;
    if (merged_task_) {
      merged_task_->setActive(use_merged);
    }
    if (position_task_) {
      position_task_->setActive(!use_merged);
    }
    if (orientation_task_) {
      orientation_task_->setActive(!use_merged);
    }
  }

  std::string name_;
  std::string tcp_frame_;
  int base_priority_;
  int rotation_priority_offset_;
  bool merged_pose_;
  bool auto_switch_ = false;
  TaskLayout current_layout_ = TaskLayout::kMerged;
  std::shared_ptr<FrameTask> position_task_;
  std::shared_ptr<FrameTask> orientation_task_;
  std::shared_ptr<FrameTask> merged_task_;
  Eigen::Matrix4d last_target_ = Eigen::Matrix4d::Identity();
  double last_position_gain_ = 1.0;
  double last_rotation_gain_ = 1.0;
  bool target_set_ = false;
};

} // namespace embodik
