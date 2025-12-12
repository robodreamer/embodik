/**
 * @file kinematics_solver_bindings.cpp
 * @brief Python bindings for KinematicsSolver
 */

#include <nanobind/eigen/dense.h>
#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <embodik/kinematics_solver.hpp>
#include <embodik/robot_model.hpp>
#include <embodik/tasks.hpp>

namespace nb = nanobind;
using namespace embodik;

void bind_kinematics_solver(nb::module_ &m) {
  nb::class_<KinematicsSolver::CollisionDebugInfo>(m, "CollisionDebugInfo")
      .def_prop_ro("object_a",
                   [](const KinematicsSolver::CollisionDebugInfo &self) {
                     return self.object_a;
                   })
      .def_prop_ro("object_b",
                   [](const KinematicsSolver::CollisionDebugInfo &self) {
                     return self.object_b;
                   })
      .def_prop_ro("point_a_world",
                   [](const KinematicsSolver::CollisionDebugInfo &self) {
                     return self.point_a_world;
                   })
      .def_prop_ro("point_b_world",
                   [](const KinematicsSolver::CollisionDebugInfo &self) {
                     return self.point_b_world;
                   })
      .def_prop_ro("distance",
                   [](const KinematicsSolver::CollisionDebugInfo &self) {
                     return self.distance;
                   });

  nb::class_<KinematicsSolver>(
      m, "KinematicsSolver",
      "High-level kinematics solver providing simple API for IK problems")
      .def(nb::init<std::shared_ptr<RobotModel>>(), nb::arg("robot"),
           "Create a kinematics solver for the given robot model")

      // Task management
      .def("add_frame_task", &KinematicsSolver::add_frame_task, nb::arg("name"),
           nb::arg("frame_name"), nb::arg("task_type") = TaskType::FRAME_POSE,
           "Add a frame tracking task")

      .def("add_com_task", &KinematicsSolver::add_com_task, nb::arg("name"),
           "Add a center of mass tracking task")

      .def(
          "add_posture_task",
          [](KinematicsSolver &self, const std::string &name) {
            return self.add_posture_task(name);
          },
          nb::arg("name"), "Add a posture regularization task for all joints")

      .def(
          "add_posture_task",
          [](KinematicsSolver &self, const std::string &name,
             const std::vector<int> &controlled_joints) {
            return self.add_posture_task(name, controlled_joints);
          },
          nb::arg("name"), nb::arg("controlled_joints"),
          "Add a posture regularization task for specific joints")

      .def("add_joint_task", &KinematicsSolver::add_joint_task, nb::arg("name"),
           nb::arg("joint_name"), nb::arg("target_value") = 0.0,
           "Add a single joint tracking task")

      .def("remove_task", &KinematicsSolver::remove_task, nb::arg("name"),
           "Remove a task by name")

      .def("clear_tasks", &KinematicsSolver::clear_tasks, "Remove all tasks")

      .def("get_task", &KinematicsSolver::get_task, nb::arg("name"),
           "Get a task by name")

      // Solving
      .def("solve_velocity", &KinematicsSolver::solve_velocity,
           nb::arg("current_q") = Eigen::VectorXd(),
           nb::arg("apply_limits") = true,
           "Solve for joint velocities without integration. Returns velocities "
           "and identifies saturated joints.")

      // Configuration
      .def("enable_velocity_limits", &KinematicsSolver::enable_velocity_limits,
           nb::arg("enable"), "Enable or disable velocity limit constraints")

      .def("enable_position_limits", &KinematicsSolver::enable_position_limits,
           nb::arg("enable"), "Enable or disable position limit constraints")

      .def("set_base_position_bounds",
           &KinematicsSolver::set_base_position_bounds, nb::arg("lower"),
           nb::arg("upper"), "Set floating-base position bounds (3D)")

      .def("set_base_orientation_bounds",
           &KinematicsSolver::set_base_orientation_bounds, nb::arg("lower"),
           nb::arg("upper"),
           "Set floating-base orientation bounds (3D, in velocity space)")

      .def("clear_base_bounds", &KinematicsSolver::clear_base_bounds,
           "Clear floating-base bounds (use unlimited bounds)")

      // Position IK methods
      .def("solve_position", &KinematicsSolver::solve_position,
           nb::arg("seed_q"), nb::arg("target_pose"), nb::arg("frame_name"),
           nb::arg("options") = PositionIKOptions(),
           "Solve position-level IK to reach target pose")

      .def("solve_position_in_tcp", &KinematicsSolver::solve_position_in_tcp,
           nb::arg("seed_q"), nb::arg("relative_target"), nb::arg("frame_name"),
           nb::arg("options") = PositionIKOptions(),
           "Solve position-level IK with target relative to TCP frame")

      .def_prop_rw("dt", &KinematicsSolver::get_dt, &KinematicsSolver::set_dt,
                   "Time step for velocity integration")

      .def("set_tolerance", &KinematicsSolver::set_tolerance,
           nb::arg("tolerance"), "Set solver convergence tolerance")

      .def("set_max_iterations", &KinematicsSolver::set_max_iterations,
           nb::arg("max_iter"), "Set maximum solver iterations")

      .def("set_damping", &KinematicsSolver::set_damping, nb::arg("damping"),
           "Set singularity robust damping factor")
      .def("enable_position_ik_debug",
           &KinematicsSolver::enable_position_ik_debug, nb::arg("enable"),
           "Enable verbose logging for position IK iterations")

      .def(
          "configure_collision_constraint",
          [](KinematicsSolver &self, double min_distance,
             const std::vector<std::pair<std::string, std::string>>
                 &include_pairs,
             const std::vector<std::pair<std::string, std::string>>
                 &exclude_pairs) {
            self.configure_collision_constraint(min_distance, include_pairs,
                                                exclude_pairs);
          },
          nb::arg("min_distance"),
          nb::arg("include_pairs") =
              std::vector<std::pair<std::string, std::string>>{},
          nb::arg("exclude_pairs") =
              std::vector<std::pair<std::string, std::string>>{},
          "Enable collision avoidance with optional include/exclude geometry "
          "pair filters.")

      .def(
          "add_collision_constraint",
          [](KinematicsSolver &self,
             const std::vector<std::pair<std::string, std::string>> &link_pairs,
             double min_distance) {
            self.add_collision_constraint(link_pairs, min_distance);
          },
          nb::arg("link_pairs"), nb::arg("min_distance") = 0.05,
          "Convenience helper to enable collision avoidance using a specific "
          "set of link pairs.")

      .def("clear_collision_constraint",
           &KinematicsSolver::clear_collision_constraint,
           "Disable collision avoidance constraint.")

      .def("get_last_collision_debug",
           &KinematicsSolver::get_last_collision_debug,
           "Retrieve debug information for the last evaluated collision pair, "
           "if available.")

      .def("get_active_collision_pairs",
           &KinematicsSolver::get_active_collision_pairs,
           "Return the list of collision pairs currently considered by the "
           "solver.")

      // Properties
      .def_prop_ro("robot", &KinematicsSolver::robot, "Get the robot model")

      .def_prop_ro("tasks", &KinematicsSolver::tasks, "Get all tasks")

      .def("__repr__", [](const KinematicsSolver &self) {
        return "KinematicsSolver(tasks=" + std::to_string(self.tasks().size()) +
               ")";
      });
}
