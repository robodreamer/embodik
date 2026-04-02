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
  const auto solve_position_step_single =
      static_cast<PositionIKResult (KinematicsSolver::*)(
          const Eigen::VectorXd &, const Eigen::Matrix4d &,
          const std::string &, const PositionStepOptions &)>(
          &KinematicsSolver::solve_position_step);
  const auto solve_position_step_multi =
      static_cast<PositionIKResult (KinematicsSolver::*)(
          const Eigen::VectorXd &, const std::vector<TaskTarget> &,
          const PositionStepOptions &)>(&KinematicsSolver::solve_position_step);

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

      .def("add_relative_frame_task",
           &KinematicsSolver::add_relative_frame_task, nb::arg("name"),
           nb::arg("frame_a"), nb::arg("frame_b"),
           "Add a relative frame task (tracks T_a^{-1} * T_b)")

      .def("add_absolute_frame_task",
           &KinematicsSolver::add_absolute_frame_task, nb::arg("name"),
           nb::arg("frame_a"), nb::arg("frame_b"),
           nb::arg("alpha") = 0.5,
           "Add an absolute frame task (weighted average of two frames)")

      .def(
          "configure_relative_pose_constraint",
          [](KinematicsSolver &self, const std::string &frame_a,
             const std::string &frame_b, const Eigen::VectorXd &lower_bounds,
             const Eigen::VectorXd &upper_bounds,
             const Eigen::VectorXd &axis_mask) {
            self.configure_relative_pose_constraint(frame_a, frame_b,
                                                    lower_bounds, upper_bounds,
                                                    axis_mask);
          },
          nb::arg("frame_a"), nb::arg("frame_b"), nb::arg("lower_bounds"),
          nb::arg("upper_bounds"),
          nb::arg("axis_mask") = Eigen::VectorXd(),
          "Configure a relative pose inequality constraint.\n\n"
          "Constrains each masked axis of T_a^{-1} * T_b to stay within\n"
          "the given bounds. Uses relative Jacobian as QP inequality rows.\n\n"
          "Args:\n"
          "  frame_a: Reference frame name\n"
          "  frame_b: Target frame name\n"
          "  lower_bounds: 6D lower bounds (pos xyz + ori xyz)\n"
          "  upper_bounds: 6D upper bounds (pos xyz + ori xyz)\n"
          "  axis_mask: 6D mask (1=constrained, 0=free). Empty = all.")

      .def("clear_relative_pose_constraint",
           &KinematicsSolver::clear_relative_pose_constraint,
           "Disable relative pose constraint")

      .def("remove_task", &KinematicsSolver::remove_task, nb::arg("name"),
           "Remove a task by name")

      .def("clear_tasks", &KinematicsSolver::clear_tasks, "Remove all tasks")
      .def("clear_all_target_velocities",
           &KinematicsSolver::clear_all_target_velocities,
           "Clear direct target velocities on all registered tasks.")

      .def("get_task", &KinematicsSolver::get_task, nb::arg("name"),
           "Get a task by name")

      // Solving
      .def("solve_velocity", &KinematicsSolver::solve_velocity,
           nb::arg("current_q") = Eigen::VectorXd(),
           nb::arg("apply_limits") = true,
           nb::arg("stall_recovery") = false,
           "Solve for joint velocities without integration. Returns velocities "
           "and identifies saturated joints. When stall_recovery=True, "
           "enables automatic stall detection and collision-margin relaxation "
           "/ restoration. The handler stays active across calls so stall "
           "counts accumulate correctly in user loops.")
      .def(
          "solve_velocity_dq",
          [](KinematicsSolver &self, const Eigen::VectorXd &current_q,
             bool apply_limits, bool stall_recovery) {
            auto r = self.solve_velocity(current_q, apply_limits,
                                         stall_recovery);
            return r.joint_velocities;
          },
          nb::arg("current_q") = Eigen::VectorXd(),
          nb::arg("apply_limits") = true,
          nb::arg("stall_recovery") = false,
          "Solve for joint velocities and return only dq as a NumPy array. "
          "When stall_recovery=True, enables automatic stall handler.")

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
           "Solve position-level IK to reach target pose. With "
           "options.classify_stagnation_as_no_progress=True, stagnation exits "
           "return SolverStatus.NO_PROGRESS.")

      .def("solve_position_step", solve_position_step_single,
           nb::arg("current_q"), nb::arg("target_pose"),
           nb::arg("frame_task_name"),
           nb::arg("options") = PositionStepOptions(),
           "Stepping position IK using the solver's registered tasks. "
           "Sets the target pose on the named FrameTask, computes pose "
           "error internally, scales it by position_gain / "
           "orientation_gain, calls solve_velocity() up to max_steps "
           "times, integrating after each step. The task weight is left "
           "untouched. Unlike solve_position(), this does not create "
           "temporary tasks. The recovery state machine (stuck detection, "
           "collision homotopy, etc.) is automatically exercised. Optional "
           "no-progress detection can return SolverStatus.NO_PROGRESS.")
      .def(
          "solve_position_step",
          [](KinematicsSolver &self, const Eigen::VectorXd &current_q,
             const pinocchio::SE3 &target_pose,
             const std::string &frame_task_name,
             const PositionStepOptions &options) {
            return self.solve_position_step(current_q,
                                            target_pose.toHomogeneousMatrix(),
                                            frame_task_name, options);
          },
          nb::arg("current_q"), nb::arg("target_pose"),
          nb::arg("frame_task_name"),
          nb::arg("options") = PositionStepOptions(),
          "Stepping position IK overload that accepts SE3/Rt directly. "
          "Equivalent to passing target_pose.homogeneous().")
      .def("solve_position_step", solve_position_step_multi,
           nb::arg("current_q"), nb::arg("targets"),
           nb::arg("options") = PositionStepOptions(),
           "Stepping position IK for multiple registered pose tasks. "
           "Each TaskTarget carries task_name, target_pose, and per-task "
           "position/orientation gains. For each step, all task target "
           "velocities are computed from pose errors, then solve_velocity() "
           "is called once to preserve coordinated multi-task behavior.")

      .def("solve_position_in_tcp", &KinematicsSolver::solve_position_in_tcp,
           nb::arg("seed_q"), nb::arg("relative_target"), nb::arg("frame_name"),
           nb::arg("options") = PositionIKOptions(),
           "Solve position-level IK with target relative to TCP frame")

      .def_prop_rw("dt", &KinematicsSolver::get_dt, &KinematicsSolver::set_dt,
                   "Time step for velocity integration")

      .def("set_tolerance", &KinematicsSolver::set_tolerance,
           nb::arg("tolerance"),
           "Set singular-value damping threshold for the regularized "
           "pseudoinverse (default 1e-6).")
      .def("set_regularization_epsilon",
           &KinematicsSolver::set_regularization_epsilon, nb::arg("epsilon"),
           "Alias of set_tolerance(): set singular-value damping threshold "
           "for the regularized pseudoinverse.")
      .def("set_constraint_tolerance",
           &KinematicsSolver::set_constraint_tolerance, nb::arg("epsilon"),
           "Set constraint violation deadband and COD pseudoinverse "
           "relative threshold (VelocitySolverConfig.epsilon).")

      .def("enable_timing_breakdown",
           &KinematicsSolver::enable_timing_breakdown, nb::arg("enable"),
           "Enable/disable detailed timing breakdown fields in "
           "VelocitySolverResult")

      .def("set_max_iterations", &KinematicsSolver::set_max_iterations,
           nb::arg("max_iter"), "Set maximum solver iterations")

      .def("set_damping", &KinematicsSolver::set_damping, nb::arg("damping"),
           "Set singularity robust damping factor")
      .def("set_limit_recovery_gain",
           &KinematicsSolver::set_limit_recovery_gain, nb::arg("gain"),
           "Set joint limit recovery gain in [0, 1]")
      .def("set_limit_recovery_hysteresis",
           &KinematicsSolver::set_limit_recovery_hysteresis,
           nb::arg("enter_epsilon"), nb::arg("exit_epsilon"),
           "Set enter/exit hysteresis epsilons for joint-limit recovery.")
      .def("set_limit_exit_release_margin",
           &KinematicsSolver::set_limit_exit_release_margin, nb::arg("margin"),
           "Set release margin that relaxes tiny post-limit recovery forcing.")
      .def("enable_saturation_exit_behavior",
           &KinematicsSolver::enable_saturation_exit_behavior, nb::arg("enable"),
           "Enable velocity-box softening near limits (disabled by default).")
      .def("saturation_exit_behavior_enabled",
           &KinematicsSolver::saturation_exit_behavior_enabled,
           "Return whether saturation-exit softening is enabled.")
      .def("set_solver_recovery_enabled",
           &KinematicsSolver::set_solver_recovery_enabled, nb::arg("enable"),
           "Deprecated no-op. Recovery state machine has been removed.")
      .def("solver_recovery_enabled",
           &KinematicsSolver::solver_recovery_enabled,
           "Deprecated no-op. Always returns False.")
      .def("enable_position_ik_debug",
           &KinematicsSolver::enable_position_ik_debug, nb::arg("enable"),
           "Enable verbose logging for position IK iterations")

      .def(
          "configure_collision_constraint",
          [](KinematicsSolver &self, double min_distance,
             const std::vector<std::pair<std::string, std::string>>
                 &include_pairs,
             const std::vector<std::pair<std::string, std::string>>
                 &exclude_pairs,
             bool nearest_points_all_pairs,
             int max_constraints) {
            self.configure_collision_constraint(min_distance, include_pairs,
                                                exclude_pairs,
                                                nearest_points_all_pairs,
                                                max_constraints);
          },
          nb::arg("min_distance"),
          nb::arg("include_pairs") =
              std::vector<std::pair<std::string, std::string>>{},
          nb::arg("exclude_pairs") =
              std::vector<std::pair<std::string, std::string>>{},
          nb::arg("nearest_points_all_pairs") = true,
          nb::arg("max_constraints") = 1,
          "Enable collision avoidance with optional include/exclude geometry "
          "pair filters.\n\n"
          "Args:\n"
          "  min_distance: Minimum separation distance to enforce (metres).\n"
          "  include_pairs: List of (geom_a, geom_b) tuples to consider "
          "(empty = all).\n"
          "  exclude_pairs: List of (geom_a, geom_b) tuples to ignore.\n"
          "  nearest_points_all_pairs: If False, compute nearest points only "
          "for the selected pair.\n"
          "  max_constraints: Number of simultaneous QP constraint rows. Each "
          "row protects one of the closest pairs independently. Defaults to 1 "
          "(original behaviour). Values of 3-5 are recommended for complex "
          "robots with multiple tight-clearance regions (e.g. base/leg and "
          "arm/torso simultaneously).")

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

      .def("set_collision_min_distance",
           &KinematicsSolver::set_collision_min_distance,
           nb::arg("min_distance"),
           "Update only the min_distance of an already-configured collision "
           "constraint without rebuilding pair masks. Returns True if "
           "updated, False if no constraint exists.")
      .def("get_collision_min_distance",
           &KinematicsSolver::get_collision_min_distance,
           "Read the current collision min_distance. Returns -1 if no "
           "collision constraint is active.")
      .def("set_proximity_gated_collision_activation_enabled",
           &KinematicsSolver::set_proximity_gated_collision_activation_enabled,
           nb::arg("enabled"),
           "Enable/disable proximity-gated collision-row activation without "
           "changing multiplier or margin values.")
      .def("get_proximity_gated_collision_activation_enabled",
           &KinematicsSolver::get_proximity_gated_collision_activation_enabled,
           "Get whether proximity-gated collision-row activation is enabled.")
      .def("set_collision_constraint_activation_multiplier",
           &KinematicsSolver::set_collision_constraint_activation_multiplier,
           nb::arg("multiplier"),
           "Set proximity-gated collision-row activation multiplier.\n\n"
           "Effective activation margin is multiplier * min_distance.\n"
           "Values <= 0 disable gating and preserve legacy row-emission "
           "behavior.")
      .def("get_collision_constraint_activation_multiplier",
           &KinematicsSolver::get_collision_constraint_activation_multiplier,
           "Get collision-row activation multiplier.")
      .def("get_collision_constraint_activation_margin",
           &KinematicsSolver::get_collision_constraint_activation_margin,
           "Get effective collision-row activation margin in meters.")
      .def("clear_collision_constraint",
           &KinematicsSolver::clear_collision_constraint,
           "Disable collision avoidance constraint.")
      .def("enable_collision_pair_cache",
           &KinematicsSolver::enable_collision_pair_cache,
           nb::arg("enable"),
           nb::arg("full_refresh_interval") = 20,
           nb::arg("candidate_distance_margin") = 0.03,
           nb::arg("max_cached_candidates") = 128,
           "Enable conservative collision pair candidate caching.\n\n"
           "When enabled, collision distance queries are evaluated on cached\n"
           "active/near-active candidate pairs between periodic full scans.\n"
           "This is intended for teleop loops where active pairs evolve\n"
           "smoothly over time.\n\n"
           "Args:\n"
           "  enable: Enable/disable candidate caching.\n"
           "  full_refresh_interval: Steps between mandatory full pair scans.\n"
           "  candidate_distance_margin: Extra margin (m) above min_distance\n"
           "    for retaining near-active pairs in the candidate cache.\n"
           "  max_cached_candidates: Cap on cached pair indices.")
      .def("set_collision_refinement_time_budget_us",
           &KinematicsSolver::set_collision_refinement_time_budget_us,
           nb::arg("budget_us"),
           "Set optional exact collision refinement budget per solve in "
           "microseconds. Values <= 0 disable budgeting.")
      .def("get_collision_refinement_time_budget_us",
           &KinematicsSolver::get_collision_refinement_time_budget_us,
           "Get exact collision refinement budget per solve (microseconds).")
      .def("set_collision_tuning_mode",
           &KinematicsSolver::set_collision_tuning_mode,
           nb::arg("mode"),
           "Apply high-level collision tuning preset.\n\n"
           "Modes:\n"
           "  PRECISE  - full exact checks (highest accuracy, highest cost)\n"
           "  BALANCED - conservative cache cadence without time budget\n"
           "  SPEED    - fastest teleop-oriented path")
      .def("get_collision_tuning_mode",
           &KinematicsSolver::get_collision_tuning_mode,
           "Get the active high-level collision tuning preset.")

      // Stall handler
      .def("enable_stall_handler",
           &KinematicsSolver::enable_stall_handler,
           nb::arg("nominal_min_distance"),
           "Enable the automatic stall handler. Detects consecutive stalled "
           "velocity solves (non-success with near-zero ||dq||) and applies "
           "collision margin relaxation/restoration to break out of stalls.")
      .def("disable_stall_handler",
           &KinematicsSolver::disable_stall_handler,
           "Disable the stall handler and restore nominal parameters.")
      .def("stall_handler_enabled",
           &KinematicsSolver::stall_handler_enabled,
           "Return True if the stall handler is enabled.")
      .def("configure_stall_handler",
           &KinematicsSolver::configure_stall_handler,
           nb::arg("stall_threshold") = 5,
           nb::arg("restore_rate") = 0.005,
           nb::arg("floor_fraction") = 0.3,
           "Configure stall handler tuning parameters.")
      .def("stall_handler_is_relaxed",
           &KinematicsSolver::stall_handler_is_relaxed,
           "Return True if collision margin is currently relaxed.")
      .def("stall_handler_current_min_distance",
           &KinematicsSolver::stall_handler_current_min_distance,
           "Return the current effective collision min_distance.")
      .def("stall_handler_consecutive_stall_steps",
           &KinematicsSolver::stall_handler_consecutive_stall_steps,
           "Return the number of consecutive stall steps.")

      // Elastic band joint limit expansion
      .def("enable_elastic_band",
           &KinematicsSolver::enable_elastic_band,
           nb::arg("delta_max") = 0.05,
           "Enable elastic band joint limit expansion for limit-dominated "
           "stalls. Temporarily expands joint limit margins to keep more DOFs "
           "active in the SNS solver.")
      .def("disable_elastic_band",
           &KinematicsSolver::disable_elastic_band,
           "Disable elastic band and reset expansion state.")
      .def("elastic_band_enabled",
           &KinematicsSolver::elastic_band_enabled,
           "Return True if elastic band is enabled.")
      .def("configure_elastic_band",
           &KinematicsSolver::configure_elastic_band,
           nb::arg("delta_max") = 0.05,
           nb::arg("expand_rate") = 0.01,
           nb::arg("decay_rate") = 0.2,
           nb::arg("stall_threshold") = 3,
           nb::arg("expand_only_saturated") = true,
           "Configure elastic band tuning parameters.")
      .def("elastic_band_max_delta",
           &KinematicsSolver::elastic_band_max_delta,
           "Return the maximum delta currently active across all joints.")
      .def("elastic_band_deltas",
           &KinematicsSolver::elastic_band_deltas,
           "Return per-joint delta vector (size == nv).")
      .def("elastic_band_is_expanded",
           &KinematicsSolver::elastic_band_is_expanded,
           "Return True if any joint has nonzero expansion.")

      // CoM support-polygon constraint
      .def(
          "configure_com_constraint",
          [](KinematicsSolver &self, const Eigen::MatrixXd &vertices_xy,
             double margin, const std::string &frame_name, double com_vel_max,
             double com_acc_max, bool use_acceleration_limits,
             double proximity_fraction) {
            self.configure_com_constraint(vertices_xy, margin, frame_name,
                                          com_vel_max, com_acc_max,
                                          use_acceleration_limits,
                                          proximity_fraction);
          },
          nb::arg("support_polygon"), nb::arg("margin") = 0.0,
          nb::arg("frame_name") = "world", nb::arg("com_vel_max") = 0.4,
          nb::arg("com_acc_max") = 0.1,
          nb::arg("use_acceleration_limits") = true,
          nb::arg("proximity_fraction") = 0.0,
          "Configure a CoM support-polygon inequality constraint.\n\n"
          "Keeps the 2D projection of the center of mass inside the given\n"
          "convex polygon. Velocity and acceleration limits are applied to\n"
          "smoothly saturate CoM velocity near the polygon boundary,\n"
          "bounding tipping energy.\n\n"
          "Args:\n"
          "  support_polygon: Nx2 or Nx3 array of polygon vertices in the\n"
          "    XY plane of frame_name (Z column is ignored if Nx3).\n"
          "  margin: Fractional inward shrink in [0, 1]. Applied as margin *\n"
          "    char_size (mean distance centroid→vertices). Matches feasibility check.\n"
          "  frame_name: Frame in which vertices are expressed.\n"
          "  com_vel_max: Maximum CoM velocity (m/s).\n"
          "  com_acc_max: Maximum CoM acceleration (m/s²).\n"
          "  use_acceleration_limits: If True, clamp approach velocity by\n"
          "    sqrt(2 * com_acc_max * margin) near boundary.\n"
          "  proximity_fraction: Fraction of the polygon inradius used as\n"
          "    the per-row activation distance.  A half-plane row is only\n"
          "    added to the QP when the CoM slack for that row is less than\n"
          "    proximity_fraction * inradius.  The inradius (minimum\n"
          "    perpendicular distance from centroid to any edge) is computed\n"
          "    automatically from the vertices.  Set to 0 (default) to\n"
          "    disable proximity filtering and always include every row.\n"
          "    Use get_com_proximity_threshold() to read back the computed\n"
          "    threshold in metres.")
      .def("get_com_proximity_threshold",
           &KinematicsSolver::get_com_proximity_threshold,
           "Return the proximity threshold (m) computed by the last call to\n"
           "configure_com_constraint().  Equals proximity_fraction * inradius\n"
           "where the inradius is the minimum perpendicular distance from the\n"
           "polygon centroid to any edge.  Returns 0 if no constraint is set.")

      .def("clear_com_constraint", &KinematicsSolver::clear_com_constraint,
           "Disable CoM support-polygon constraint.")

      .def("get_last_collision_debug",
           &KinematicsSolver::get_last_collision_debug,
           "Retrieve debug information for the closest active collision pair "
           "after the last solve, if available. When max_constraints > 1, "
           "use get_last_collision_debug_list() for all active pairs.")

      .def("get_last_collision_debug_list",
           &KinematicsSolver::get_last_collision_debug_list,
           "Retrieve debug information for all active collision constraint "
           "pairs after the last solve (one entry per constraint row, up to "
           "max_constraints). Returns an empty list when no collision "
           "constraint is configured or no solve has been performed.")

      .def("evaluate_collision_debug",
           &KinematicsSolver::evaluate_collision_debug,
           nb::arg("current_q") = Eigen::VectorXd(),
           "Evaluate collisions at the provided configuration and return debug "
           "info (side-effect free).")

      .def("get_active_collision_pairs",
           &KinematicsSolver::get_active_collision_pairs,
           "Return the list of collision pairs currently considered by the "
           "solver.")

      .def(
          "calculate_velocity_box_constraint",
          &KinematicsSolver::calculate_velocity_box_constraint,
          nb::arg("position_margin_lower"), nb::arg("position_margin_upper"),
          nb::arg("velocity_limit"), nb::arg("acceleration_limit"),
          nb::arg("dt"), nb::arg("min_velocity_headroom") = -1.0,
          nb::arg("headroom_activation_margin") = 0.01,
          "Compute velocity bounds from position/velocity/acceleration limits.")

      // Properties
      .def_prop_ro("robot", &KinematicsSolver::robot, "Get the robot model")

      .def_prop_ro("tasks", &KinematicsSolver::tasks, "Get all tasks")

      .def("__repr__", [](const KinematicsSolver &self) {
        return "KinematicsSolver(tasks=" + std::to_string(self.tasks().size()) +
               ")";
      });
}
