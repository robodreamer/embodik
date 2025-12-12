/**
 * @file tasks_bindings.cpp
 * @brief Python bindings for embodiK tasks
 */

#include <nanobind/eigen/dense.h>
#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <embodik/robot_model.hpp>
#include <embodik/tasks.hpp>

namespace nb = nanobind;
using namespace embodik;

void bind_tasks(nb::module_ &m) {
  // TaskType enum
  nb::enum_<TaskType>(m, "TaskType")
      .value("FRAME_POSITION", TaskType::FRAME_POSITION)
      .value("FRAME_ORIENTATION", TaskType::FRAME_ORIENTATION)
      .value("FRAME_POSE", TaskType::FRAME_POSE)
      .value("COM", TaskType::COM)
      .value("POSTURE", TaskType::POSTURE)
      .value("JOINT", TaskType::JOINT);

  // Base Task class (abstract, so we don't expose constructor)
  nb::class_<Task>(m, "Task")
      .def("update", &Task::update, nb::arg("model"),
           "Update task computations based on current robot state")
      .def("get_error", &Task::getError,
           "Get task error vector (desired - current)")
      .def("get_jacobian", &Task::getJacobian, "Get task Jacobian matrix")
      .def("get_velocity", &Task::getVelocity,
           "Get task velocity (typically -gain * error)")
      .def("set_target_velocity", &Task::setTargetVelocity, nb::arg("velocity"),
           "Set target velocity directly")
      .def("clear_target_velocity", &Task::clearTargetVelocity,
           "Clear target velocity (revert to error-based velocity)")
      .def("get_dimension", &Task::getDimension, "Get task dimension")
      .def("get_type", &Task::getType, "Get task type")
      .def_prop_ro("name", &Task::getName, "Task name")
      .def_prop_rw("priority", &Task::getPriority, &Task::setPriority,
                   "Task priority (0 = highest)")
      .def_prop_rw("weight", &Task::getWeight, &Task::setWeight,
                   "Task weight/gain")
      .def_prop_rw("active", &Task::isActive, &Task::setActive,
                   "Whether task is active")
      .def("set_excluded_joint_indices", &Task::set_excluded_joint_indices,
           nb::arg("excluded_indices"),
           "Set excluded joint indices (velocity space indices to exclude from "
           "Jacobian)")
      .def("clear_excluded_joint_indices", &Task::clear_excluded_joint_indices,
           "Clear excluded joint indices")
      .def("get_excluded_joint_indices", &Task::get_excluded_joint_indices,
           "Get excluded joint indices");

  // FrameTask
  nb::class_<FrameTask, Task>(m, "FrameTask")
      .def(nb::init<const std::string &, std::shared_ptr<RobotModel>,
                    const std::string &, TaskType, int, double>(),
           nb::arg("name"), nb::arg("model"), nb::arg("frame_name"),
           nb::arg("task_type") = TaskType::FRAME_POSE, nb::arg("priority") = 0,
           nb::arg("weight") = 1.0, "Create a frame tracking task")
      .def("set_target_position", &FrameTask::setTargetPosition,
           nb::arg("position"), "Set desired position")
      .def("set_target_orientation", &FrameTask::setTargetOrientation,
           nb::arg("rotation"), "Set desired orientation as rotation matrix")
      .def("set_target_pose", &FrameTask::setTargetPose, nb::arg("position"),
           nb::arg("rotation"), "Set desired pose (position + orientation)")
      .def("set_target_position_velocity",
           &FrameTask::setTargetPositionVelocity, nb::arg("velocity"),
           "Set target linear velocity (3D)")
      .def("set_target_angular_velocity", &FrameTask::setTargetAngularVelocity,
           nb::arg("omega"), "Set target angular velocity (3D)")
      .def("set_target_velocity", &FrameTask::setTargetVelocity,
           nb::arg("velocity"), "Set full spatial velocity (linear + angular)")
      .def("set_position_mask", &FrameTask::setPositionMask, nb::arg("mask"),
           "Set position mask (which axes to control)")
      .def("set_orientation_mask", &FrameTask::setOrientationMask,
           nb::arg("mask"), "Set orientation mask (which axes to control)")
      .def_prop_ro("current_position", &FrameTask::getCurrentPosition,
                   "Current frame position")
      .def_prop_ro("current_orientation", &FrameTask::getCurrentOrientation,
                   "Current frame orientation");

  // COMTask
  nb::class_<COMTask, Task>(m, "COMTask")
      .def(nb::init<const std::string &, std::shared_ptr<RobotModel>, int,
                    double>(),
           nb::arg("name"), nb::arg("model"), nb::arg("priority") = 0,
           nb::arg("weight") = 1.0, "Create a center of mass tracking task")
      .def("set_target_position", &COMTask::setTargetPosition,
           nb::arg("position"), "Set desired COM position")
      .def("set_position_mask", &COMTask::setPositionMask, nb::arg("mask"),
           "Set position mask (which axes to control)")
      .def_prop_ro("current_position", &COMTask::getCurrentPosition,
                   "Current COM position");

  // PostureTask
  nb::class_<PostureTask, Task>(m, "PostureTask")
      .def(nb::init<const std::string &, std::shared_ptr<RobotModel>, int,
                    double>(),
           nb::arg("name"), nb::arg("model"), nb::arg("priority") = 10,
           nb::arg("weight") = 0.1,
           "Create a posture regularization task for all joints")
      .def(nb::init<const std::string &, std::shared_ptr<RobotModel>,
                    const std::vector<int> &, int, double>(),
           nb::arg("name"), nb::arg("model"),
           nb::arg("controlled_joint_indices"), nb::arg("priority") = 10,
           nb::arg("weight") = 0.1,
           "Create a posture regularization task for specific joints")
      .def("set_target_configuration", &PostureTask::setTargetConfiguration,
           nb::arg("q_target"), "Set target joint configuration")
      .def("set_controlled_joint_targets",
           &PostureTask::setControlledJointTargets, nb::arg("target_values"),
           "Set target values for controlled joints only")
      .def("set_joint_mask", &PostureTask::setJointMask, nb::arg("mask"),
           "Set joint mask (which joints to control)")
      .def("set_controlled_joint_indices",
           &PostureTask::setControlledJointIndices, nb::arg("indices"),
           "Set controlled joint indices")
      .def("set_joint_weights", &PostureTask::setJointWeights,
           nb::arg("weights"), "Set per-joint weights")
      .def("set_controlled_joint_weights",
           &PostureTask::setControlledJointWeights, nb::arg("weights"),
           "Set weights for controlled joints only")
      .def_prop_ro("controlled_joint_indices",
                   &PostureTask::getControlledJointIndices,
                   "Get controlled joint indices");

  // JointTask
  nb::class_<JointTask, Task>(m, "JointTask")
      .def(nb::init<const std::string &, std::shared_ptr<RobotModel>,
                    const std::string &, double, int, double>(),
           nb::arg("name"), nb::arg("model"), nb::arg("joint_name"),
           nb::arg("target_value") = 0.0, nb::arg("priority") = 0,
           nb::arg("weight") = 1.0,
           "Create a single joint tracking task by name")
      .def(nb::init<const std::string &, std::shared_ptr<RobotModel>, int,
                    double, int, double>(),
           nb::arg("name"), nb::arg("model"), nb::arg("joint_index"),
           nb::arg("target_value") = 0.0, nb::arg("priority") = 0,
           nb::arg("weight") = 1.0,
           "Create a single joint tracking task by index")
      .def("set_target_value", &JointTask::setTargetValue, nb::arg("value"),
           "Set target joint value");

  // MultiJointTask
  nb::class_<MultiJointTask, Task>(m, "MultiJointTask")
      .def(nb::init<const std::string &, std::shared_ptr<RobotModel>,
                    const std::vector<int> &, const Eigen::VectorXd &, int,
                    double>(),
           nb::arg("name"), nb::arg("model"), nb::arg("joint_indices"),
           nb::arg("target_values") = Eigen::VectorXd(),
           nb::arg("priority") = 0, nb::arg("weight") = 1.0,
           "Create a multi-joint tracking task by indices")
      .def(nb::init<const std::string &, std::shared_ptr<RobotModel>,
                    const std::vector<std::string> &, const Eigen::VectorXd &,
                    int, double>(),
           nb::arg("name"), nb::arg("model"), nb::arg("joint_names"),
           nb::arg("target_values") = Eigen::VectorXd(),
           nb::arg("priority") = 0, nb::arg("weight") = 1.0,
           "Create a multi-joint tracking task by names")
      .def("set_target_values", &MultiJointTask::setTargetValues,
           nb::arg("values"), "Set target values for all controlled joints")
      .def("set_target_value", &MultiJointTask::setTargetValue, nb::arg("idx"),
           nb::arg("value"), "Set target value for a specific controlled joint")
      .def("set_joint_weights", &MultiJointTask::setJointWeights,
           nb::arg("weights"), "Set per-joint weights")
      .def_prop_ro("joint_indices", &MultiJointTask::getJointIndices,
                   "Get controlled joint indices")
      .def_prop_ro("target_values", &MultiJointTask::getTargetValues,
                   "Get target values");

  // Helper function to create rotation matrix from roll-pitch-yaw
  m.def(
      "rotation_from_rpy",
      [](double roll, double pitch, double yaw) {
        return Eigen::AngleAxisd(yaw, Eigen::Vector3d::UnitZ()) *
               Eigen::AngleAxisd(pitch, Eigen::Vector3d::UnitY()) *
               Eigen::AngleAxisd(roll, Eigen::Vector3d::UnitX())
                   .toRotationMatrix();
      },
      nb::arg("roll"), nb::arg("pitch"), nb::arg("yaw"),
      "Create rotation matrix from roll-pitch-yaw angles");
}
