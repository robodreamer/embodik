/*
 * MIT License
 *
 * Copyright (c) 2025 Andy Park <andypark.purdue@gmail.com>
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

#include <nanobind/eigen/dense.h>
#include <nanobind/nanobind.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/unordered_map.h>
#include <nanobind/stl/vector.h>

#include <embodik/robot_model.hpp>

namespace nb = nanobind;
using namespace embodik;

void bind_robot_model(nb::module_ &m) {
  // Bind ReferenceFrame enum from Pinocchio
  nb::enum_<pinocchio::ReferenceFrame>(m, "ReferenceFrame")
      .value("WORLD", pinocchio::WORLD)
      .value("LOCAL", pinocchio::LOCAL)
      .value("LOCAL_WORLD_ALIGNED", pinocchio::LOCAL_WORLD_ALIGNED);

  // Bind SE3 transformation
  nb::class_<pinocchio::SE3>(m, "SE3")
      .def(nb::init<>())
      .def_prop_rw(
          "rotation",
          [](const pinocchio::SE3 &self) { return self.rotation(); },
          [](pinocchio::SE3 &self, const Eigen::Matrix3d &R) {
            self.rotation() = R;
          })
      .def_prop_rw(
          "translation",
          [](const pinocchio::SE3 &self) { return self.translation(); },
          [](pinocchio::SE3 &self, const Eigen::Vector3d &t) {
            self.translation() = t;
          })
      .def(
          "homogeneous",
          [](const pinocchio::SE3 &self) { return self.toHomogeneousMatrix(); },
          "Get 4x4 homogeneous transformation matrix")
      .def("inverse", &pinocchio::SE3::inverse)
      .def("__repr__", [](const pinocchio::SE3 &self) {
        std::stringstream ss;
        ss << "SE3(translation=" << self.translation().transpose()
           << ", rotation=<3x3 matrix>)";
        return ss.str();
      });

  // Bind RobotModel class
  nb::class_<RobotModel>(m, "RobotModel")
      .def(nb::init<const std::string &, bool>(), nb::arg("urdf_path"),
           nb::arg("floating_base") = false, "Load robot model from URDF file")

      .def_static("from_xacro", &RobotModel::from_xacro, nb::arg("xacro_path"),
                  nb::arg("floating_base") = false,
                  "Create robot model from XACRO file")

      // Configuration methods
      .def("update_configuration", &RobotModel::update_configuration,
           nb::arg("q"),
           "Update robot configuration and compute forward kinematics")

      .def(
          "update_kinematics",
          [](RobotModel &self, const Eigen::VectorXd &q,
             const Eigen::VectorXd &v) { self.update_kinematics(q, v); },
          nb::arg("q"), nb::arg("v") = Eigen::VectorXd(),
          "Update robot configuration and optionally velocity")

      // Frame operations
      .def("get_frame_pose", &RobotModel::get_frame_pose, nb::arg("frame_name"),
           "Get pose of specified frame as SE3 transformation")

      .def("get_frame_jacobian", &RobotModel::get_frame_jacobian,
           nb::arg("frame_name"),
           nb::arg("reference_frame") = pinocchio::LOCAL_WORLD_ALIGNED,
           "Get 6xN Jacobian of specified frame")

      // Center of mass
      .def("get_com_position", &RobotModel::get_com_position,
           "Get center of mass position")

      .def("get_com_velocity", &RobotModel::get_com_velocity,
           "Get center of mass velocity")

      .def("get_com_jacobian", &RobotModel::get_com_jacobian,
           "Get 3xN center of mass Jacobian")

      // Information queries
      .def("get_frame_names", &RobotModel::get_frame_names,
           "Get list of all frame names")

      .def("get_joint_names", &RobotModel::get_joint_names,
           "Get list of all joint names")

      .def("get_joint_limits", &RobotModel::get_joint_limits,
           "Get joint position limits as (lower, upper) pair")

      .def("set_joint_limits", &RobotModel::set_joint_limits, nb::arg("lower"),
           nb::arg("upper"), "Set joint position limits")

      .def("get_velocity_limits", &RobotModel::get_velocity_limits,
           "Get joint velocity limits")

      .def("get_acceleration_limits", &RobotModel::get_acceleration_limits,
           "Get joint acceleration limits (returns defaults if not specified "
           "in URDF)")

      .def("set_acceleration_limits", &RobotModel::set_acceleration_limits,
           nb::arg("accel_limits"), "Set custom joint acceleration limits")

      .def("get_effort_limits", &RobotModel::get_effort_limits,
           "Get joint effort/torque limits")

      .def("has_frame", &RobotModel::has_frame, nb::arg("frame_name"),
           "Check if frame exists")

      // State accessors
      .def("get_current_configuration", &RobotModel::get_current_configuration,
           nb::rv_policy::reference_internal, "Get current joint configuration")

      .def("get_current_velocity", &RobotModel::get_current_velocity,
           nb::rv_policy::reference_internal, "Get current joint velocities")

      // Properties
      .def_prop_ro("nq", &RobotModel::nq, "Number of configuration variables")
      .def_prop_ro("nv", &RobotModel::nv, "Number of velocity variables")
      .def_prop_ro("is_floating_base", &RobotModel::is_floating_base,
                   "Whether robot has floating base")

      // Expose Pinocchio model and data for visualization
      .def_prop_ro(
          "_pinocchio_model",
          [](const RobotModel &self) -> const pinocchio::Model & {
            return self.model();
          },
          nb::rv_policy::reference_internal,
          "Internal Pinocchio model (for visualization)")
      .def_prop_ro(
          "_pinocchio_data",
          [](const RobotModel &self) -> const pinocchio::Data & {
            return self.data();
          },
          nb::rv_policy::reference_internal,
          "Internal Pinocchio data (for visualization)")

      // Expose visual and collision models (may be nullptr)
      .def_prop_ro(
          "visual_model",
          [](const RobotModel &self) { return self.visual_model(); },
          nb::rv_policy::reference_internal,
          "Visual geometry model (may be None)")
      .def_prop_ro(
          "collision_model",
          [](const RobotModel &self) { return self.collision_model(); },
          nb::rv_policy::reference_internal,
          "Collision geometry model (may be None)")
      .def_prop_ro(
          "collision_data",
          [](const RobotModel &self) { return self.collision_data(); },
          nb::rv_policy::reference_internal,
          "Collision geometry data (may be None)")
      .def_prop_ro(
          "visual_data",
          [](const RobotModel &self) { return self.visual_data(); },
          nb::rv_policy::reference_internal,
          "Visual geometry data (may be None)")

      .def("get_collision_geometry_names",
           &RobotModel::get_collision_geometry_names,
           "Return the list of collision geometry object names")

      .def("get_collision_pair_names", &RobotModel::get_collision_pair_names,
           "Return the list of collision pairs as name tuples")

      .def("apply_collision_exclusions",
           &RobotModel::apply_collision_exclusions, nb::arg("collision_pairs"),
           "Disable the provided collision pairs using an SRDF-style "
           "specification")

      // Expose URDF path for visualization
      .def_prop_ro("urdf_path", &RobotModel::urdf_path, "Path to URDF file")

      // Expose controlled joints info (use lambdas for public member access)
      .def_prop_rw(
          "controlled_joint_names",
          [](const RobotModel &self) -> const std::vector<std::string> & {
            return self.controlled_joint_names;
          },
          [](RobotModel &self, const std::vector<std::string> &names) {
            self.controlled_joint_names = names;
          },
          "List of controlled joint names")
      .def_prop_rw(
          "controlled_joint_indices",
          [](const RobotModel &self)
              -> const std::unordered_map<std::string, int> & {
            return self.controlled_joint_indices;
          },
          [](RobotModel &self,
             const std::unordered_map<std::string, int> &indices) {
            self.controlled_joint_indices = indices;
          },
          "Dictionary mapping joint names to indices")

      .def("__repr__", [](const RobotModel &self) {
        std::stringstream ss;
        ss << "RobotModel(nq=" << self.nq() << ", nv=" << self.nv()
           << ", floating_base=" << (self.is_floating_base() ? "True" : "False")
           << ", frames=" << self.get_frame_names().size() << ")";
        return ss.str();
      });
}
