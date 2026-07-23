#include "acceleration_task_differential.hpp"

#include "dual_arm_ects_differential.hpp"

#include <embodik/robot_model.hpp>

#include <stdexcept>

namespace embodik::detail {

struct TaskAccelerationDifferentialAdapter {
  static AccelerationTaskDifferential evaluate(const Task &task,
                                               const RobotModel &model,
                                               bool control_only_zero_velocity);

private:
  static void finish_success(const Task &task,
                             AccelerationTaskDifferential &result);
  static AccelerationTaskDifferential evaluate_frame_task(
      const FrameTask &task, const RobotModel &model);
  static AccelerationTaskDifferential evaluate_com_task(const COMTask &task,
                                                       const RobotModel &model);
  static AccelerationTaskDifferential evaluate_posture_task(
      const PostureTask &task);
  static AccelerationTaskDifferential evaluate_joint_task(
      const JointTask &task);
  static AccelerationTaskDifferential evaluate_multi_joint_task(
      const MultiJointTask &task);
  static AccelerationTaskDifferential evaluate_relative_frame_task(
      const RelativeFrameTask &task, const RobotModel &model);
  static AccelerationTaskDifferential evaluate_absolute_frame_task(
      const AbsoluteFrameTask &task, const RobotModel &model);
  static AccelerationTaskDifferential
  evaluate_control_only_zero_velocity(const Task &task);
};

namespace {

AccelerationTaskDifferential unsupported_result(const Task &task,
                                                const std::string &reason) {
  AccelerationTaskDifferential result;
  result.status = AccelerationTaskDifferentialStatus::kUnsupportedTask;
  result.message = "task '" + task.getName() +
                   "' does not support acceleration differential: " + reason;
  return result;
}

AccelerationTaskDifferential invalid_result(const Task &task,
                                            const std::string &reason) {
  AccelerationTaskDifferential result;
  result.status = AccelerationTaskDifferentialStatus::kInvalidInput;
  result.message = "task '" + task.getName() +
                   "' acceleration differential failed: " + reason;
  return result;
}

void scale_bias_rows(AccelerationTaskDifferential &result) {
  for (Eigen::Index row = 0; row < result.reference_row_scale.size(); ++row) {
    result.jacobian_bias(row) *= result.reference_row_scale(row);
  }
}

} // namespace

void TaskAccelerationDifferentialAdapter::finish_success(
    const Task &task, AccelerationTaskDifferential &result) {
  result.control_jacobian =
      task.apply_excluded_joint_columns(result.physical_jacobian);
  const Eigen::Index row_count = result.physical_jacobian.rows();
  if (result.control_jacobian.rows() != row_count ||
      result.control_jacobian.cols() != result.physical_jacobian.cols() ||
      result.position_error.size() != row_count ||
      result.jacobian_bias.size() != row_count ||
      result.reference_row_scale.size() != row_count) {
    throw std::invalid_argument(
        "acceleration task differential has inconsistent dimensions");
  }
  if (!result.physical_jacobian.allFinite() ||
      !result.control_jacobian.allFinite() ||
      !result.position_error.allFinite() ||
      !result.jacobian_bias.allFinite() ||
      !result.reference_row_scale.allFinite()) {
    throw std::invalid_argument(
        "acceleration task differential contains non-finite values");
  }
  result.status = AccelerationTaskDifferentialStatus::kSuccess;
  result.message.clear();
}

AccelerationTaskDifferential
TaskAccelerationDifferentialAdapter::evaluate_frame_task(
    const FrameTask &task, const RobotModel &model) {
  AccelerationTaskDifferential result;
  const Eigen::Matrix<double, 6, 1> full_bias =
      model.get_frame_jacobian_bias(task.frame_name_);
  result.physical_jacobian = task.buildPhysicalJacobian();
  result.reference_row_scale = task.referenceRowScale();

  switch (task.task_type_) {
  case TaskType::FRAME_POSITION:
    result.jacobian_bias = full_bias.head<3>();
    break;
  case TaskType::FRAME_ORIENTATION:
    result.jacobian_bias = full_bias.tail<3>();
    break;
  case TaskType::FRAME_POSE:
    result.jacobian_bias = full_bias;
    break;
  default:
    return unsupported_result(task, "invalid frame task type");
  }

  scale_bias_rows(result);
  result.position_error = task.getError();
  finish_success(task, result);
  return result;
}

AccelerationTaskDifferential TaskAccelerationDifferentialAdapter::evaluate_com_task(
    const COMTask &task, const RobotModel &model) {
  AccelerationTaskDifferential result;
  result.physical_jacobian = task.buildPhysicalJacobian();
  result.position_error = task.getError();
  result.jacobian_bias = model.get_com_jacobian_bias();
  result.reference_row_scale = task.referenceRowScale();
  scale_bias_rows(result);
  finish_success(task, result);
  return result;
}

AccelerationTaskDifferential
TaskAccelerationDifferentialAdapter::evaluate_posture_task(
    const PostureTask &task) {
  AccelerationTaskDifferential result;
  result.position_error = task.getError();
  result.physical_jacobian = task.buildPhysicalJacobian();
  result.reference_row_scale = task.referenceRowScale();
  result.jacobian_bias =
      Eigen::VectorXd::Zero(result.physical_jacobian.rows());
  finish_success(task, result);
  return result;
}

AccelerationTaskDifferential TaskAccelerationDifferentialAdapter::evaluate_joint_task(
    const JointTask &task) {
  AccelerationTaskDifferential result;
  result.physical_jacobian = task.buildPhysicalJacobian();
  result.position_error = task.getError();
  result.jacobian_bias = Eigen::VectorXd::Zero(1);
  result.reference_row_scale = Eigen::VectorXd::Ones(1);
  finish_success(task, result);
  return result;
}

AccelerationTaskDifferential
TaskAccelerationDifferentialAdapter::evaluate_multi_joint_task(
    const MultiJointTask &task) {
  AccelerationTaskDifferential result;
  result.physical_jacobian = task.buildPhysicalJacobian();
  result.position_error = task.getError();
  result.jacobian_bias =
      Eigen::VectorXd::Zero(result.physical_jacobian.rows());
  result.reference_row_scale = task.referenceRowScale();
  finish_success(task, result);
  return result;
}

AccelerationTaskDifferential
TaskAccelerationDifferentialAdapter::evaluate_relative_frame_task(
    const RelativeFrameTask &task, const RobotModel &model) {
  const auto differential = evaluate_relative_ects_task_differential(
      model, task.frame_a_, task.frame_b_);
  AccelerationTaskDifferential result;
  result.physical_jacobian = differential.spatial_jacobian;
  result.jacobian_bias = differential.spatial_bias;
  result.reference_row_scale = task.referenceRowScale();
  for (Eigen::Index row = 0; row < result.reference_row_scale.size(); ++row) {
    result.physical_jacobian.row(row) *= result.reference_row_scale(row);
    result.jacobian_bias(row) *= result.reference_row_scale(row);
  }
  result.position_error = task.getError();
  finish_success(task, result);
  return result;
}

AccelerationTaskDifferential
TaskAccelerationDifferentialAdapter::evaluate_absolute_frame_task(
    const AbsoluteFrameTask &task, const RobotModel &model) {
  AccelerationTaskDifferential result;
  result.physical_jacobian = task.buildPhysicalJacobian();
  result.jacobian_bias =
      task.alpha_ * model.get_frame_jacobian_bias(task.frame_a_) +
      (1.0 - task.alpha_) * model.get_frame_jacobian_bias(task.frame_b_);
  result.reference_row_scale = task.referenceRowScale();
  scale_bias_rows(result);
  result.position_error = task.getError();
  finish_success(task, result);
  return result;
}

AccelerationTaskDifferential
TaskAccelerationDifferentialAdapter::evaluate_control_only_zero_velocity(
    const Task &task) {
  AccelerationTaskDifferential result;
  result.position_error = task.getError();
  result.control_jacobian = task.getJacobian();
  result.jacobian_bias = Eigen::VectorXd::Zero(result.control_jacobian.rows());

  if (const auto *frame = dynamic_cast<const FrameTask *>(&task)) {
    result.reference_row_scale = frame->referenceRowScale();
  } else if (const auto *com = dynamic_cast<const COMTask *>(&task)) {
    result.reference_row_scale = com->referenceRowScale();
  } else if (const auto *posture = dynamic_cast<const PostureTask *>(&task)) {
    result.reference_row_scale = posture->referenceRowScale();
  } else if (dynamic_cast<const JointTask *>(&task)) {
    result.reference_row_scale =
        Eigen::VectorXd::Ones(result.control_jacobian.rows());
  } else if (const auto *multi_joint =
                 dynamic_cast<const MultiJointTask *>(&task)) {
    result.reference_row_scale = multi_joint->referenceRowScale();
  } else if (const auto *relative =
                 dynamic_cast<const RelativeFrameTask *>(&task)) {
    result.reference_row_scale = relative->referenceRowScale();
  } else if (const auto *absolute =
                 dynamic_cast<const AbsoluteFrameTask *>(&task)) {
    result.reference_row_scale = absolute->referenceRowScale();
  } else if (dynamic_cast<const ManipulabilityTask *>(&task)) {
    return unsupported_result(task,
                              "manipulability acceleration rows are not "
                              "implemented");
  } else if (dynamic_cast<const JointLimitAvoidanceTask *>(&task)) {
    return unsupported_result(task,
                              "joint-limit avoidance acceleration rows are "
                              "not implemented");
  } else {
    return unsupported_result(task, "custom task type");
  }

  const Eigen::Index row_count = result.control_jacobian.rows();
  if (result.position_error.size() != row_count ||
      result.jacobian_bias.size() != row_count ||
      result.reference_row_scale.size() != row_count ||
      !result.control_jacobian.allFinite() ||
      !result.position_error.allFinite() || !result.jacobian_bias.allFinite() ||
      !result.reference_row_scale.allFinite()) {
    throw std::invalid_argument(
        "control-only acceleration task differential is inconsistent");
  }
  result.status = AccelerationTaskDifferentialStatus::kSuccess;
  result.message.clear();
  return result;
}

AccelerationTaskDifferential TaskAccelerationDifferentialAdapter::evaluate(
    const Task &task, const RobotModel &model,
    bool control_only_zero_velocity) {
  try {
    if (control_only_zero_velocity) {
      return evaluate_control_only_zero_velocity(task);
    }
    if (const auto *frame = dynamic_cast<const FrameTask *>(&task)) {
      return evaluate_frame_task(*frame, model);
    }
    if (const auto *com = dynamic_cast<const COMTask *>(&task)) {
      return evaluate_com_task(*com, model);
    }
    if (const auto *posture = dynamic_cast<const PostureTask *>(&task)) {
      return evaluate_posture_task(*posture);
    }
    if (const auto *joint = dynamic_cast<const JointTask *>(&task)) {
      return evaluate_joint_task(*joint);
    }
    if (const auto *multi_joint = dynamic_cast<const MultiJointTask *>(&task)) {
      return evaluate_multi_joint_task(*multi_joint);
    }
    if (const auto *relative =
            dynamic_cast<const RelativeFrameTask *>(&task)) {
      return evaluate_relative_frame_task(*relative, model);
    }
    if (const auto *absolute =
            dynamic_cast<const AbsoluteFrameTask *>(&task)) {
      return evaluate_absolute_frame_task(*absolute, model);
    }
    if (dynamic_cast<const ManipulabilityTask *>(&task)) {
      return unsupported_result(task,
                                "manipulability acceleration rows are not "
                                "implemented");
    }
    if (dynamic_cast<const JointLimitAvoidanceTask *>(&task)) {
      return unsupported_result(task,
                                "joint-limit avoidance acceleration rows are "
                                "not implemented");
    }
    return unsupported_result(task, "custom task type");
  } catch (const std::invalid_argument &e) {
    return invalid_result(task, e.what());
  } catch (const std::domain_error &e) {
    return invalid_result(task, e.what());
  } catch (const std::runtime_error &e) {
    return invalid_result(task, e.what());
  }
}

AccelerationTaskDifferential evaluate_acceleration_task_differential(
    const Task &task, const RobotModel &model,
    bool control_only_zero_velocity) {
  return TaskAccelerationDifferentialAdapter::evaluate(
      task, model, control_only_zero_velocity);
}

} // namespace embodik::detail
