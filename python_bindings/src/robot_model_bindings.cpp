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
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/unordered_map.h>
#include <nanobind/stl/vector.h>

#include <embodik/robot_model.hpp>

// Pinocchio exponential map functions
#include <pinocchio/spatial/explog.hpp>

// Coal geometry types for collision info extraction
#include <coal/shape/geometric_shapes.h>

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
      .def(nb::init<const Eigen::Matrix3d &, const Eigen::Vector3d &>(),
           nb::arg("rotation"), nb::arg("translation"),
           "Create SE3 from rotation matrix and translation vector")
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

  // ============================================================================
  // Rotation utilities (log3, exp3) - replaces Python pinocchio imports
  // ============================================================================

  m.def(
      "log3", [](const Eigen::Matrix3d &R) { return pinocchio::log3(R); },
      nb::arg("rotation"),
      "Compute axis-angle vector from 3x3 rotation matrix.\n\n"
      "Args:\n"
      "    rotation: 3x3 rotation matrix\n\n"
      "Returns:\n"
      "    3D axis-angle vector (rotation axis * angle in radians)");

  m.def(
      "exp3",
      [](const Eigen::Vector3d &omega) { return pinocchio::exp3(omega); },
      nb::arg("omega"),
      "Compute 3x3 rotation matrix from axis-angle vector.\n\n"
      "Args:\n"
      "    omega: 3D axis-angle vector (rotation axis * angle in radians)\n\n"
      "Returns:\n"
      "    3x3 rotation matrix");

  // ============================================================================
  // Quaternion utilities - replaces Python pinocchio.Quaternion
  // ============================================================================

  m.def(
      "matrix_to_quaternion_wxyz",
      [](const Eigen::Matrix3d &R) {
        Eigen::Quaterniond q(R);
        q.normalize(); // Ensure unit quaternion
        return std::make_tuple(q.w(), q.x(), q.y(), q.z());
      },
      nb::arg("rotation"),
      "Convert 3x3 rotation matrix to quaternion (w, x, y, z) format.\n\n"
      "Args:\n"
      "    rotation: 3x3 rotation matrix\n\n"
      "Returns:\n"
      "    Tuple of (w, x, y, z) quaternion components");

  m.def(
      "quaternion_wxyz_to_matrix",
      [](double w, double x, double y, double z) {
        Eigen::Quaterniond q(w, x, y, z);
        q.normalize(); // Ensure unit quaternion
        return q.toRotationMatrix();
      },
      nb::arg("w"), nb::arg("x"), nb::arg("y"), nb::arg("z"),
      "Convert quaternion (w, x, y, z) to 3x3 rotation matrix.\n\n"
      "Args:\n"
      "    w: Quaternion scalar component\n"
      "    x: Quaternion x component\n"
      "    y: Quaternion y component\n"
      "    z: Quaternion z component\n\n"
      "Returns:\n"
      "    3x3 rotation matrix");

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

      // Configuration-space operations (Lie-group aware)
      .def("integrate", &RobotModel::integrate, nb::arg("q"), nb::arg("v"),
           nb::arg("dt") = 1.0,
           "Integrate velocity into configuration using Lie group "
           "operations.\n\n"
           "For standard revolute/prismatic joints this is equivalent to q + "
           "v*dt,\n"
           "but for floating-base (SE3), spherical (quaternion), and other\n"
           "non-Euclidean joint types it performs the correct manifold\n"
           "integration (e.g. quaternion exponential map).\n\n"
           "Args:\n"
           "    q: Current configuration vector (size nq)\n"
           "    v: Velocity / tangent vector (size nv)\n"
           "    dt: Time step (default 1.0, i.e. v is already scaled)\n\n"
           "Returns:\n"
           "    Integrated configuration vector (size nq)")

      .def("difference", &RobotModel::difference, nb::arg("q0"), nb::arg("q1"),
           "Compute the tangent-vector difference between two "
           "configurations.\n\n"
           "Returns the velocity v such that q1 = integrate(q0, v).\n"
           "For Euclidean joints this is simply q1 - q0, but for quaternion /\n"
           "floating-base joints the result lives in the tangent space (size "
           "nv).\n\n"
           "Args:\n"
           "    q0: Start configuration (size nq)\n"
           "    q1: End configuration (size nq)\n\n"
           "Returns:\n"
           "    Tangent vector v (size nv)")

      .def("neutral_configuration", &RobotModel::neutral_configuration,
           "Return the neutral (zero / home) configuration.\n\n"
           "For floating-base robots this includes a valid unit quaternion\n"
           "for the base orientation rather than all-zeros.\n\n"
           "Returns:\n"
           "    Neutral configuration vector (size nq)")

      .def("random_configuration", &RobotModel::random_configuration,
           "Generate a random valid configuration within joint limits.\n\n"
           "Uses Pinocchio's randomConfiguration which respects the joint\n"
           "topology (e.g. generates valid quaternions for floating-base).\n\n"
           "Returns:\n"
           "    Random configuration vector (size nq)")

      .def("normalize", &RobotModel::normalize, nb::arg("q"),
           "Normalize a configuration vector.\n\n"
           "For joints on a manifold (quaternion components of floating-base\n"
           "or spherical joints) this re-normalizes the quaternion part.\n"
           "For standard revolute/prismatic joints this is a no-op.\n\n"
           "Args:\n"
           "    q: Configuration vector (size nq)\n\n"
           "Returns:\n"
           "    Normalized configuration vector (size nq)")

      // ================================================================
      // Inverse dynamics / gravity
      // ================================================================

      .def("compute_generalized_gravity",
           &RobotModel::compute_generalized_gravity, nb::arg("q"),
           "Compute the generalized gravity torque vector g(q).\n\n"
           "Returns the joint torques required to compensate gravity at the\n"
           "given configuration (equivalent to RNEA with zero velocity and\n"
           "acceleration).\n\n"
           "Args:\n"
           "    q: Joint configuration vector (size nq)\n\n"
           "Returns:\n"
           "    Gravity torque vector (size nv)")

      .def("rnea", &RobotModel::rnea, nb::arg("q"), nb::arg("v"),
           nb::arg("a"),
           "Compute inverse dynamics using the Recursive Newton-Euler "
           "Algorithm.\n\n"
           "Returns tau = M(q)*a + C(q,v)*v + g(q).\n\n"
           "Common usage patterns:\n"
           "  - Gravity only:   rnea(q, zeros, zeros)\n"
           "  - Coriolis+grav:  rnea(q, v, zeros)\n"
           "  - Full dynamics:  rnea(q, v, a)\n\n"
           "Args:\n"
           "    q: Joint configuration vector (size nq)\n"
           "    v: Joint velocity vector (size nv)\n"
           "    a: Joint acceleration vector (size nv)\n\n"
           "Returns:\n"
           "    Joint torque vector tau (size nv)")

      .def("compute_mass_matrix", &RobotModel::compute_mass_matrix,
           nb::arg("q"),
           "Compute the joint-space mass/inertia matrix M(q).\n\n"
           "Uses the Composite Rigid Body Algorithm (CRBA). The returned\n"
           "matrix is symmetric positive-definite.\n\n"
           "Args:\n"
           "    q: Joint configuration vector (size nq)\n\n"
           "Returns:\n"
           "    Mass matrix M(q) (size nv x nv)")

      .def("compute_coriolis", &RobotModel::compute_coriolis, nb::arg("q"),
           nb::arg("v"),
           "Compute the Coriolis + centrifugal torque vector C(q,v)*v.\n\n"
           "Args:\n"
           "    q: Joint configuration vector (size nq)\n"
           "    v: Joint velocity vector (size nv)\n\n"
           "Returns:\n"
           "    Coriolis torque vector (size nv)")

      .def("set_gravity", &RobotModel::set_gravity, nb::arg("gravity"),
           "Set the gravity vector for the model.\n\n"
           "Default is [0, 0, -9.81]. Affects gravity torque computations.\n\n"
           "Args:\n"
           "    gravity: 3D gravity vector (e.g. [0, 0, -9.81])")

      .def("get_gravity", &RobotModel::get_gravity,
           "Get the current gravity vector.\n\n"
           "Returns:\n"
           "    3D gravity vector")

      // ================================================================
      // Boolean collision checking
      // ================================================================

      .def(
          "check_collision",
          [](const RobotModel &self, double min_distance) {
            return self.check_collision(min_distance);
          },
          nb::arg("min_distance") = 0.0,
          "Check whether any collision pair has distance below a threshold.\n\n"
          "Returns True if any pair has distance <= min_distance.\n"
          "With default min_distance=0.0, checks for actual contact/overlap.\n\n"
          "Args:\n"
          "    min_distance: Distance threshold (default 0.0)\n\n"
          "Returns:\n"
          "    True if collision detected, False otherwise")

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

      .def("has_collision_geometry", &RobotModel::has_collision_geometry,
           "Check if collision geometry is available")

      // Get collision geometry info as Python-friendly dicts
      .def(
          "get_collision_geometries",
          [](const RobotModel &self) {
            std::vector<nb::dict> geometries;
            if (!self.collision_model()) {
              return geometries;
            }
            const auto &geom_model = *self.collision_model();
            for (size_t i = 0; i < geom_model.geometryObjects.size(); ++i) {
              const auto &geom = geom_model.geometryObjects[i];
              nb::dict d;
              d["name"] = geom.name;
              d["parent_frame"] = self.model().frames[geom.parentFrame].name;
              d["parent_joint"] = geom.parentJoint;
              // Placement as translation + rotation
              d["placement_translation"] =
                  Eigen::Vector3d(geom.placement.translation());
              d["placement_rotation"] =
                  Eigen::Matrix3d(geom.placement.rotation());
              // Geometry type and parameters
              std::string geom_type = "unknown";
              nb::dict params;
              if (geom.geometry) {
                auto type = geom.geometry->getNodeType();
                switch (type) {
                case coal::GEOM_BOX: {
                  geom_type = "box";
                  auto box =
                      std::dynamic_pointer_cast<coal::Box>(geom.geometry);
                  if (box) {
                    params["half_extents"] = Eigen::Vector3d(
                        box->halfSide[0], box->halfSide[1], box->halfSide[2]);
                  }
                  break;
                }
                case coal::GEOM_SPHERE: {
                  geom_type = "sphere";
                  auto sphere =
                      std::dynamic_pointer_cast<coal::Sphere>(geom.geometry);
                  if (sphere) {
                    params["radius"] = sphere->radius;
                  }
                  break;
                }
                case coal::GEOM_CYLINDER: {
                  geom_type = "cylinder";
                  auto cyl =
                      std::dynamic_pointer_cast<coal::Cylinder>(geom.geometry);
                  if (cyl) {
                    params["radius"] = cyl->radius;
                    params["half_length"] = cyl->halfLength;
                  }
                  break;
                }
                case coal::GEOM_CAPSULE: {
                  geom_type = "capsule";
                  auto cap =
                      std::dynamic_pointer_cast<coal::Capsule>(geom.geometry);
                  if (cap) {
                    params["radius"] = cap->radius;
                    params["half_length"] = cap->halfLength;
                  }
                  break;
                }
                case coal::GEOM_CONVEX:
                case coal::BV_AABB:
                case coal::BV_OBB:
                case coal::BV_RSS:
                case coal::BV_kIOS:
                case coal::BV_OBBRSS:
                case coal::BV_KDOP16:
                case coal::BV_KDOP18:
                case coal::BV_KDOP24:
                  geom_type = "convex";
                  break;
                default:
                  geom_type = "mesh";
                  break;
                }
              }
              d["geometry_type"] = geom_type;
              d["params"] = params;
              geometries.push_back(d);
            }
            return geometries;
          },
          "Get collision geometries as list of dicts with name, parent_frame, "
          "placement, geometry_type, and params")

      .def("apply_collision_exclusions",
           &RobotModel::apply_collision_exclusions, nb::arg("collision_pairs"),
           "Disable the provided collision pairs using an SRDF-style "
           "specification")

      .def("compute_min_collision_distance",
           &RobotModel::compute_min_collision_distance,
           "Compute minimum collision distance at current configuration.\n\n"
           "Returns:\n"
           "    Minimum distance across all collision pairs, or inf if no "
           "collision geometry")

      .def("compute_collision_distances",
           &RobotModel::compute_collision_distances,
           "Compute collision distances for all pairs at current "
           "configuration.\n\n"
           "Returns:\n"
           "    List of distances for each collision pair")

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
