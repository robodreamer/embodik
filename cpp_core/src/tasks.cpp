/**
 * @file tasks.cpp
 * @brief Implementation of task classes for EmbodiK solver
 */

#include <algorithm>
#include <embodik/robot_model.hpp>
#include <embodik/tasks.hpp>
#include <iostream>
#include <pinocchio/algorithm/kinematics.hpp>

namespace embodik {

// Helper function to compute orientation error using log map
static Eigen::Vector3d logMap(const Eigen::Matrix3d &R) {
  // Clamp the trace to avoid numerical issues with acos
  double trace = R.trace();
  double cos_theta = (trace - 1.0) / 2.0;
  cos_theta = std::max(-1.0, std::min(1.0, cos_theta));

  double theta = std::acos(cos_theta);

  if (std::abs(theta) < 1e-6) {
    // Small angle approximation
    return 0.5 * Eigen::Vector3d(R(2, 1) - R(1, 2), R(0, 2) - R(2, 0),
                                 R(1, 0) - R(0, 1));
  }

  Eigen::Vector3d w =
      (1.0 / (2.0 * std::sin(theta))) *
      Eigen::Vector3d(R(2, 1) - R(1, 2), R(0, 2) - R(2, 0), R(1, 0) - R(0, 1));
  return theta * w;
}

//=============================================================================
// FrameTask Implementation
//=============================================================================

FrameTask::FrameTask(const std::string &name, std::shared_ptr<RobotModel> model,
                     const std::string &frame_name, TaskType task_type,
                     int priority, double weight)
    : Task(name, priority, weight), model_(model), frame_name_(frame_name),
      task_type_(task_type) {

  // Validate task type
  if (task_type != TaskType::FRAME_POSITION &&
      task_type != TaskType::FRAME_ORIENTATION &&
      task_type != TaskType::FRAME_POSE) {
    throw std::invalid_argument("Invalid task type for FrameTask");
  }

  // Check if frame exists
  if (!model_->has_frame(frame_name)) {
    throw std::invalid_argument("Frame '" + frame_name +
                                "' not found in robot model");
  }
}

void FrameTask::setTargetPosition(const Eigen::Vector3d &position) {
  target_position_ = position;
  invalidateCache();
}

void FrameTask::setTargetOrientation(const Eigen::Matrix3d &rotation) {
  target_orientation_ = rotation;
  invalidateCache();
}

void FrameTask::setTargetPose(const Eigen::Vector3d &position,
                              const Eigen::Matrix3d &rotation) {
  target_position_ = position;
  target_orientation_ = rotation;
  invalidateCache();
}

void FrameTask::setTargetPositionVelocity(const Eigen::Vector3d &velocity) {
  if (task_type_ == TaskType::FRAME_ORIENTATION) {
    throw std::runtime_error(
        "Cannot set position velocity for orientation-only task");
  }

  if (task_type_ == TaskType::FRAME_POSITION) {
    target_velocity_ = velocity;
  } else { // FRAME_POSE
    Eigen::VectorXd full_velocity(6);
    full_velocity.head(3) = velocity;
    full_velocity.tail(3) = Eigen::Vector3d::Zero();
    target_velocity_ = full_velocity;
  }
}

void FrameTask::setTargetAngularVelocity(const Eigen::Vector3d &omega) {
  if (task_type_ == TaskType::FRAME_POSITION) {
    throw std::runtime_error(
        "Cannot set angular velocity for position-only task");
  }

  if (task_type_ == TaskType::FRAME_ORIENTATION) {
    target_velocity_ = omega;
  } else { // FRAME_POSE
    Eigen::VectorXd full_velocity(6);
    full_velocity.head(3) = Eigen::Vector3d::Zero();
    full_velocity.tail(3) = omega;
    target_velocity_ = full_velocity;
  }
}

void FrameTask::setTargetVelocity(const Eigen::VectorXd &velocity) {
  // Validate velocity dimension based on task type
  if (task_type_ == TaskType::FRAME_POSITION && velocity.size() != 3) {
    throw std::invalid_argument("Position task requires 3D velocity");
  } else if (task_type_ == TaskType::FRAME_ORIENTATION &&
             velocity.size() != 3) {
    throw std::invalid_argument(
        "Orientation task requires 3D angular velocity");
  } else if (task_type_ == TaskType::FRAME_POSE && velocity.size() != 6) {
    throw std::invalid_argument(
        "Pose task requires 6D velocity (linear + angular)");
  }

  target_velocity_ = velocity;
}

void FrameTask::update(const RobotModel &model) {
  // Get current frame pose
  pinocchio::SE3 frame_pose = model.get_frame_pose(frame_name_);
  current_position_ = frame_pose.translation();
  current_orientation_ = frame_pose.rotation();

  // Update Jacobians based on task type
  // Optimize: For FRAME_POSE, compute Jacobian once and extract both parts
  if (task_type_ == TaskType::FRAME_POSE) {
    // Get 6D Jacobian once and extract both position and orientation parts
    Matrix6Xd J = model.get_frame_jacobian(frame_name_);
    position_jacobian_ = J.topRows<3>();
    orientation_jacobian_ = J.bottomRows<3>();
  } else {
    // For single-type tasks, only compute what's needed
    if (task_type_ == TaskType::FRAME_POSITION) {
      Matrix6Xd J = model.get_frame_jacobian(frame_name_);
      position_jacobian_ = J.topRows<3>();
    }

    if (task_type_ == TaskType::FRAME_ORIENTATION) {
      Matrix6Xd J = model.get_frame_jacobian(frame_name_);
      orientation_jacobian_ = J.bottomRows<3>();
    }
  }

  invalidateCache();
}

Eigen::VectorXd FrameTask::getError() const {
  if (!cache_valid_) {
    switch (task_type_) {
    case TaskType::FRAME_POSITION: {
      if (!target_position_) {
        error_cache_ = Eigen::Vector3d::Zero();
      } else {
        Eigen::Vector3d error = *target_position_ - current_position_;
        // Apply mask
        error = error.cwiseProduct(position_mask_);
        error_cache_ = error;
      }
      break;
    }

    case TaskType::FRAME_ORIENTATION: {
      if (!target_orientation_) {
        error_cache_ = Eigen::Vector3d::Zero();
      } else {
        Eigen::Vector3d error =
            computeOrientationError(current_orientation_, *target_orientation_);
        // Apply mask
        error = error.cwiseProduct(orientation_mask_);
        error_cache_ = error;
      }
      break;
    }

    case TaskType::FRAME_POSE: {
      error_cache_.resize(6);

      // Position error
      if (!target_position_) {
        error_cache_.head<3>() = Eigen::Vector3d::Zero();
      } else {
        Eigen::Vector3d pos_error = *target_position_ - current_position_;
        pos_error = pos_error.cwiseProduct(position_mask_);
        error_cache_.head<3>() = pos_error;
      }

      // Orientation error
      if (!target_orientation_) {
        error_cache_.tail<3>() = Eigen::Vector3d::Zero();
      } else {
        Eigen::Vector3d ori_error =
            computeOrientationError(current_orientation_, *target_orientation_);
        ori_error = ori_error.cwiseProduct(orientation_mask_);
        error_cache_.tail<3>() = ori_error;
      }
      break;
    }

    default:
      error_cache_ = Eigen::VectorXd::Zero(0);
    }
  }

  return error_cache_;
}

Eigen::MatrixXd FrameTask::getJacobian() const {
  if (!cache_valid_) {
    switch (task_type_) {
    case TaskType::FRAME_POSITION:
      jacobian_cache_ = position_jacobian_;
      // Apply mask by zeroing out rows
      for (int i = 0; i < 3; ++i) {
        if (position_mask_(i) == 0) {
          jacobian_cache_.row(i).setZero();
        }
      }
      break;

    case TaskType::FRAME_ORIENTATION:
      jacobian_cache_ = orientation_jacobian_;
      // Apply mask by zeroing out rows
      for (int i = 0; i < 3; ++i) {
        if (orientation_mask_(i) == 0) {
          jacobian_cache_.row(i).setZero();
        }
      }
      break;

    case TaskType::FRAME_POSE:
      jacobian_cache_.resize(6, position_jacobian_.cols());
      jacobian_cache_.topRows<3>() = position_jacobian_;
      jacobian_cache_.bottomRows<3>() = orientation_jacobian_;

      // Apply masks
      for (int i = 0; i < 3; ++i) {
        if (position_mask_(i) == 0) {
          jacobian_cache_.row(i).setZero();
        }
        if (orientation_mask_(i) == 0) {
          jacobian_cache_.row(i + 3).setZero();
        }
      }
      break;

    default:
      jacobian_cache_ = Eigen::MatrixXd::Zero(0, model_->nv());
    }

    // Apply joint exclusion: zero out excluded joint columns
    // Excluded joints have zero columns in the Jacobian
    for (int excluded_idx : excluded_joint_indices_) {
      if (excluded_idx >= 0 && excluded_idx < jacobian_cache_.cols()) {
        jacobian_cache_.col(excluded_idx).setZero();
      }
    }

    cache_valid_ = true;
  }

  return jacobian_cache_;
}

int FrameTask::getDimension() const {
  switch (task_type_) {
  case TaskType::FRAME_POSITION:
    return 3;
  case TaskType::FRAME_ORIENTATION:
    return 3;
  case TaskType::FRAME_POSE:
    return 6;
  default:
    return 0;
  }
}

Eigen::Vector3d
FrameTask::computeOrientationError(const Eigen::Matrix3d &R_current,
                                   const Eigen::Matrix3d &R_desired) const {
  // Compute rotation error: R_error = R_desired * R_current^T
  Eigen::Matrix3d R_error = R_desired * R_current.transpose();

  // Convert to axis-angle representation using log map
  return logMap(R_error);
}

//=============================================================================
// COMTask Implementation
//=============================================================================

COMTask::COMTask(const std::string &name, std::shared_ptr<RobotModel> model,
                 int priority, double weight)
    : Task(name, priority, weight), model_(model) {}

void COMTask::setTargetPosition(const Eigen::Vector3d &position) {
  target_position_ = position;
}

void COMTask::update(const RobotModel &model) {
  // Get current COM position
  current_position_ = model.get_com_position();

  // Get COM Jacobian
  com_jacobian_ = model.get_com_jacobian();
}

Eigen::VectorXd COMTask::getError() const {
  Eigen::Vector3d error = target_position_ - current_position_;
  // Apply mask
  return error.cwiseProduct(position_mask_);
}

Eigen::MatrixXd COMTask::getJacobian() const {
  Eigen::MatrixXd J = com_jacobian_;

  // Apply mask by zeroing out rows
  for (int i = 0; i < 3; ++i) {
    if (position_mask_(i) == 0) {
      J.row(i).setZero();
    }
  }

  // Apply joint exclusion: zero out excluded joint columns
  for (int excluded_idx : excluded_joint_indices_) {
    if (excluded_idx >= 0 && excluded_idx < J.cols()) {
      J.col(excluded_idx).setZero();
    }
  }

  return J;
}

int COMTask::getDimension() const { return 3; }

//=============================================================================
// PostureTask Implementation
//=============================================================================

PostureTask::PostureTask(const std::string &name,
                         std::shared_ptr<RobotModel> model, int priority,
                         double weight)
    : Task(name, priority, weight), model_(model) {

  // Initialize with current configuration
  q_target_ = model->get_current_configuration();
  q_current_ = q_target_;

  // Initialize masks and weights
  joint_mask_ = Eigen::VectorXd::Ones(model->nq());
  joint_weights_ = Eigen::VectorXd::Ones(model->nq());

  // Initialize projection matrix and Jacobian
  updateProjectionMatrix();
}

PostureTask::PostureTask(const std::string &name,
                         std::shared_ptr<RobotModel> model,
                         const std::vector<int> &controlled_joint_indices,
                         int priority, double weight)
    : Task(name, priority, weight), model_(model),
      controlled_joint_indices_(controlled_joint_indices) {

  // Initialize with current configuration
  q_target_ = model->get_current_configuration();
  q_current_ = q_target_;

  // Initialize masks and weights based on controlled joints
  joint_mask_ = Eigen::VectorXd::Zero(model->nq());
  joint_weights_ = Eigen::VectorXd::Ones(model->nq());

  // Set mask for controlled joints
  for (int idx : controlled_joint_indices_) {
    if (idx >= 0 && idx < model->nv()) {
      // For velocity space indexing
      if (model->is_floating_base() && idx >= 6) {
        // Map from velocity index to configuration index
        int q_idx = idx + 1; // +1 for quaternion vs angular velocity
        if (q_idx < model->nq()) {
          joint_mask_(q_idx) = 1.0;
        }
      } else if (!model->is_floating_base() && idx < model->nq()) {
        joint_mask_(idx) = 1.0;
      }
    }
  }

  // Initialize projection matrix and Jacobian
  updateProjectionMatrix();
}

void PostureTask::setTargetConfiguration(const Eigen::VectorXd &q_target) {
  if (q_target.size() != model_->nq()) {
    throw std::invalid_argument("Target configuration size mismatch");
  }
  q_target_ = q_target;
}

void PostureTask::setControlledJointTargets(
    const Eigen::VectorXd &target_values) {
  if (target_values.size() !=
      static_cast<int>(controlled_joint_indices_.size())) {
    throw std::invalid_argument(
        "Target values size must match number of controlled joints");
  }

  // Set target values for controlled joints only
  for (size_t i = 0; i < controlled_joint_indices_.size(); ++i) {
    int v_idx = controlled_joint_indices_[i];
    if (v_idx >= 0 && v_idx < model_->nv()) {
      if (model_->is_floating_base() && v_idx >= 6) {
        // Map from velocity index to configuration index
        int q_idx = v_idx + 1;
        if (q_idx < model_->nq()) {
          q_target_(q_idx) = target_values(i);
        }
      } else if (!model_->is_floating_base() && v_idx < model_->nq()) {
        q_target_(v_idx) = target_values(i);
      }
    }
  }
}

void PostureTask::setControlledJointIndices(const std::vector<int> &indices) {
  controlled_joint_indices_ = indices;

  // Update mask based on new indices
  joint_mask_.setZero();
  for (int idx : controlled_joint_indices_) {
    if (idx >= 0 && idx < model_->nv()) {
      if (model_->is_floating_base() && idx >= 6) {
        int q_idx = idx + 1;
        if (q_idx < model_->nq()) {
          joint_mask_(q_idx) = 1.0;
        }
      } else if (!model_->is_floating_base() && idx < model_->nq()) {
        joint_mask_(idx) = 1.0;
      }
    }
  }

  updateProjectionMatrix();
}

void PostureTask::setControlledJointWeights(const Eigen::VectorXd &weights) {
  if (weights.size() != static_cast<int>(controlled_joint_indices_.size())) {
    throw std::invalid_argument(
        "Weights size must match number of controlled joints");
  }

  // Set weights for controlled joints only
  for (size_t i = 0; i < controlled_joint_indices_.size(); ++i) {
    int v_idx = controlled_joint_indices_[i];
    if (v_idx >= 0 && v_idx < model_->nv()) {
      if (model_->is_floating_base() && v_idx >= 6) {
        int q_idx = v_idx + 1;
        if (q_idx < model_->nq()) {
          joint_weights_(q_idx) = weights(i);
        }
      } else if (!model_->is_floating_base() && v_idx < model_->nq()) {
        joint_weights_(v_idx) = weights(i);
      }
    }
  }
}

void PostureTask::update(const RobotModel &model) {
  q_current_ = model.get_current_configuration();
}

void PostureTask::updateProjectionMatrix() {
  projection_matrix_ = Eigen::MatrixXd::Zero(model_->nv(), model_->nv());

  if (controlled_joint_indices_.empty()) {
    // Control all joints
    projection_matrix_.setIdentity();
  } else {
    // Control only specified joints
    for (int idx : controlled_joint_indices_) {
      if (idx >= 0 && idx < model_->nv()) {
        projection_matrix_(idx, idx) = 1.0;
      }
    }
  }

  // Update Jacobian
  jacobian_ = projection_matrix_;
}

Eigen::VectorXd PostureTask::getError() const {
  // Configuration space error
  Eigen::VectorXd q_error = q_target_ - q_current_;

  // If controlling specific joints, return reduced error
  if (!controlled_joint_indices_.empty()) {
    Eigen::VectorXd reduced_error(controlled_joint_indices_.size());

    for (size_t i = 0; i < controlled_joint_indices_.size(); ++i) {
      int v_idx = controlled_joint_indices_[i];

      if (model_->is_floating_base()) {
        if (v_idx < 6) {
          // Floating base velocities
          reduced_error(i) = q_error(v_idx) * joint_weights_(v_idx);
        } else {
          // Joint velocities (account for quaternion)
          int q_idx = v_idx + 1;
          if (q_idx < q_error.size()) {
            reduced_error(i) = q_error(q_idx) * joint_weights_(q_idx);
          } else {
            reduced_error(i) = 0.0;
          }
        }
      } else {
        // Fixed base
        if (v_idx < q_error.size()) {
          reduced_error(i) = q_error(v_idx) * joint_weights_(v_idx);
        } else {
          reduced_error(i) = 0.0;
        }
      }
    }

    return reduced_error;
  }

  // For all joints, use standard approach
  if (model_->is_floating_base()) {
    Eigen::VectorXd v_error(model_->nv());
    v_error.setZero();

    // Handle floating base part (TODO: proper SE3 error)
    // For now, just use position error for translation
    v_error.head<3>() = q_error.head<3>();

    // Copy joint errors (skip quaternion)
    if (model_->nq() > 7) {
      v_error.tail(model_->nv() - 6) = q_error.tail(model_->nq() - 7);
    }

    // Apply weights
    return v_error.cwiseProduct(joint_weights_.head(model_->nv()));
  } else {
    // For fixed base, nq == nv
    return q_error.cwiseProduct(joint_weights_.head(model_->nv()));
  }
}

Eigen::MatrixXd PostureTask::getJacobian() const {
  Eigen::MatrixXd result_jacobian;

  // If controlling specific joints, return reduced Jacobian
  if (!controlled_joint_indices_.empty()) {
    result_jacobian.resize(controlled_joint_indices_.size(), model_->nv());
    result_jacobian.setZero();

    for (size_t i = 0; i < controlled_joint_indices_.size(); ++i) {
      int v_idx = controlled_joint_indices_[i];
      result_jacobian(i, v_idx) = joint_weights_(v_idx);
    }
  } else {
    // For all joints, return weighted identity
    result_jacobian = jacobian_;
    for (int i = 0; i < model_->nv(); ++i) {
      result_jacobian.row(i) *= joint_weights_(i);
    }
  }

  // Apply joint exclusion: zero out excluded joint columns
  // Note: For posture tasks, exclusion means those joints won't be regularized
  for (int excluded_idx : excluded_joint_indices_) {
    if (excluded_idx >= 0 && excluded_idx < result_jacobian.cols()) {
      result_jacobian.col(excluded_idx).setZero();
    }
  }

  return result_jacobian;
}

int PostureTask::getDimension() const {
  if (controlled_joint_indices_.empty()) {
    return model_->nv();
  } else {
    return static_cast<int>(controlled_joint_indices_.size());
  }
}

//=============================================================================
// JointTask Implementation
//=============================================================================

JointTask::JointTask(const std::string &name, std::shared_ptr<RobotModel> model,
                     const std::string &joint_name, double target_value,
                     int priority, double weight)
    : Task(name, priority, weight), model_(model), target_value_(target_value) {

  // Find joint index from name
  const auto &joint_names = model->model().names;
  auto it = std::find(joint_names.begin(), joint_names.end(), joint_name);

  if (it == joint_names.end()) {
    throw std::invalid_argument("Joint '" + joint_name +
                                "' not found in robot model");
  }

  joint_index_ = std::distance(joint_names.begin(), it) -
                 1; // -1 because names[0] is "universe"

  // Initialize Jacobian
  jacobian_ = Eigen::MatrixXd::Zero(1, model->nv());
}

JointTask::JointTask(const std::string &name, std::shared_ptr<RobotModel> model,
                     int joint_index, double target_value, int priority,
                     double weight)
    : Task(name, priority, weight), model_(model), joint_index_(joint_index),
      target_value_(target_value) {

  if (joint_index < 0 || joint_index >= model->nv()) {
    throw std::invalid_argument("Joint index out of bounds");
  }

  // Initialize Jacobian
  jacobian_ = Eigen::MatrixXd::Zero(1, model->nv());
}

void JointTask::update(const RobotModel &model) {
  // Get current joint value
  // For floating base, we need to handle the offset
  if (model.is_floating_base() && joint_index_ >= 6) {
    // Floating base: first 7 elements of q are position + quaternion
    current_value_ = model.get_current_configuration()(joint_index_ + 1);
  } else if (!model.is_floating_base()) {
    // Fixed base: direct mapping
    current_value_ = model.get_current_configuration()(joint_index_);
  } else {
    // Floating base DOF - not supported for single joint task
    current_value_ = 0.0;
  }

  // Update Jacobian (single DOF)
  jacobian_.setZero();
  jacobian_(0, joint_index_) = 1.0;
}

Eigen::VectorXd JointTask::getError() const {
  Eigen::VectorXd error(1);
  error(0) = target_value_ - current_value_;
  return error;
}

Eigen::MatrixXd JointTask::getJacobian() const { return jacobian_; }

//=============================================================================
// MultiJointTask Implementation
//=============================================================================

MultiJointTask::MultiJointTask(const std::string &name,
                               std::shared_ptr<RobotModel> model,
                               const std::vector<int> &joint_indices,
                               const Eigen::VectorXd &target_values,
                               int priority, double weight)
    : Task(name, priority, weight), model_(model),
      joint_indices_(joint_indices) {

  // Validate joint indices
  for (int idx : joint_indices) {
    if (idx < 0 || idx >= model->nv()) {
      throw std::invalid_argument("Joint index " + std::to_string(idx) +
                                  " out of bounds");
    }
  }

  // Initialize target values
  if (target_values.size() == 0) {
    target_values_ = Eigen::VectorXd::Zero(joint_indices_.size());
    // Initialize with current values
    const auto &q = model->get_current_configuration();
    for (size_t i = 0; i < joint_indices_.size(); ++i) {
      int v_idx = joint_indices_[i];
      if (model->is_floating_base() && v_idx >= 6) {
        // Map from velocity index to configuration index
        int q_idx = v_idx + 1;
        if (q_idx < q.size()) {
          target_values_(i) = q(q_idx);
        }
      } else if (!model->is_floating_base() && v_idx < q.size()) {
        target_values_(i) = q(v_idx);
      }
    }
  } else if (target_values.size() != static_cast<int>(joint_indices_.size())) {
    throw std::invalid_argument(
        "Target values size must match number of joint indices");
  } else {
    target_values_ = target_values;
  }

  // Initialize current values and weights
  current_values_ = Eigen::VectorXd::Zero(joint_indices_.size());
  joint_weights_ = Eigen::VectorXd::Ones(joint_indices_.size());

  // Initialize Jacobian (sparse matrix with 1s at controlled joint columns)
  jacobian_ = Eigen::MatrixXd::Zero(joint_indices_.size(), model->nv());
  for (size_t i = 0; i < joint_indices_.size(); ++i) {
    jacobian_(i, joint_indices_[i]) = 1.0;
  }
}

MultiJointTask::MultiJointTask(const std::string &name,
                               std::shared_ptr<RobotModel> model,
                               const std::vector<std::string> &joint_names,
                               const Eigen::VectorXd &target_values,
                               int priority, double weight)
    : Task(name, priority, weight), model_(model) {

  // Convert joint names to indices
  const auto &model_joint_names = model->model().names;
  for (const auto &joint_name : joint_names) {
    auto it = std::find(model_joint_names.begin(), model_joint_names.end(),
                        joint_name);
    if (it == model_joint_names.end()) {
      throw std::invalid_argument("Joint '" + joint_name +
                                  "' not found in robot model");
    }
    int idx =
        std::distance(model_joint_names.begin(), it) - 1; // -1 for "universe"

    // For floating base robots, the first 6 DOFs are the base
    if (model->is_floating_base()) {
      // Joint indices in velocity space
      if (idx >= 0) {
        joint_indices_.push_back(idx + 6); // Add 6 for floating base DOFs
      }
    } else {
      joint_indices_.push_back(idx);
    }
  }

  // Initialize target values
  if (target_values.size() == 0) {
    target_values_ = Eigen::VectorXd::Zero(joint_indices_.size());
    // Initialize with current values
    const auto &q = model->get_current_configuration();
    for (size_t i = 0; i < joint_indices_.size(); ++i) {
      int v_idx = joint_indices_[i];
      if (model->is_floating_base() && v_idx >= 6) {
        int q_idx = v_idx + 1;
        if (q_idx < q.size()) {
          target_values_(i) = q(q_idx);
        }
      } else if (!model->is_floating_base() && v_idx < q.size()) {
        target_values_(i) = q(v_idx);
      }
    }
  } else if (target_values.size() != static_cast<int>(joint_indices_.size())) {
    throw std::invalid_argument(
        "Target values size must match number of joint names");
  } else {
    target_values_ = target_values;
  }

  // Initialize current values and weights
  current_values_ = Eigen::VectorXd::Zero(joint_indices_.size());
  joint_weights_ = Eigen::VectorXd::Ones(joint_indices_.size());

  // Initialize Jacobian
  jacobian_ = Eigen::MatrixXd::Zero(joint_indices_.size(), model->nv());
  for (size_t i = 0; i < joint_indices_.size(); ++i) {
    jacobian_(i, joint_indices_[i]) = 1.0;
  }
}

void MultiJointTask::setTargetValues(const Eigen::VectorXd &values) {
  if (values.size() != static_cast<int>(joint_indices_.size())) {
    throw std::invalid_argument(
        "Target values size must match number of controlled joints");
  }
  target_values_ = values;
}

void MultiJointTask::setTargetValue(int idx, double value) {
  if (idx < 0 || idx >= static_cast<int>(joint_indices_.size())) {
    throw std::invalid_argument("Index out of bounds for controlled joints");
  }
  target_values_(idx) = value;
}

void MultiJointTask::setJointWeights(const Eigen::VectorXd &weights) {
  if (weights.size() != static_cast<int>(joint_indices_.size())) {
    throw std::invalid_argument(
        "Weights size must match number of controlled joints");
  }
  joint_weights_ = weights;
}

void MultiJointTask::update(const RobotModel &model) {
  const auto &q = model.get_current_configuration();

  // Update current values
  for (size_t i = 0; i < joint_indices_.size(); ++i) {
    int v_idx = joint_indices_[i];
    if (model.is_floating_base() && v_idx >= 6) {
      // Map from velocity index to configuration index
      int q_idx = v_idx + 1;
      if (q_idx < q.size()) {
        current_values_(i) = q(q_idx);
      }
    } else if (!model.is_floating_base() && v_idx < q.size()) {
      current_values_(i) = q(v_idx);
    }
  }
}

Eigen::VectorXd MultiJointTask::getError() const {
  Eigen::VectorXd error = target_values_ - current_values_;
  return error.cwiseProduct(joint_weights_);
}

Eigen::MatrixXd MultiJointTask::getJacobian() const {
  // Apply weights to Jacobian rows
  Eigen::MatrixXd J = jacobian_;
  for (int i = 0; i < static_cast<int>(joint_indices_.size()); ++i) {
    J.row(i) *= joint_weights_(i);
  }
  return J;
}

} // namespace embodik
