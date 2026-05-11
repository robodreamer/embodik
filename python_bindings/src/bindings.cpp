/*
 * Copyright 2025-2026 Andy Park <andypark.purdue@gmail.com>
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     https://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <nanobind/eigen/dense.h>
#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <embodik/ik_baseline.hpp>
#include <embodik/types.hpp>
#include <pinocchio/spatial/se3.hpp>

namespace nb = nanobind;
namespace eik = embodik;

// Forward declarations for sub-module bindings
void bind_robot_model(nb::module_ &m);
void bind_tasks(nb::module_ &m);
void bind_kinematics_solver(nb::module_ &m);
void bind_pose_metrics(nb::module_ &m);

NB_MODULE(_embodik_impl, m) {
  m.doc() = "embodiK: High-performance inverse kinematics with Pinocchio";
  using nb::literals::operator""_a;

  // Enums
  nb::enum_<eik::SolverStatus>(m, "SolverStatus",
                               "Status codes for embodiK baseline")
      .value("SUCCESS", eik::SolverStatus::kSuccess)
      .value("INVALID_INPUT", eik::SolverStatus::kInvalidInput)
      .value("NUMERICAL_ERROR", eik::SolverStatus::kNumericalError)
      .value("SHAPE_MISMATCH", eik::SolverStatus::kShapeMismatch)
      .value("EMPTY_PROBLEM", eik::SolverStatus::kEmptyProblem)
      .value("CONSTRAINT_BOUNDS_MISMATCH",
             eik::SolverStatus::kConstraintBoundsMismatch)
      .value("NON_FINITE_INPUT", eik::SolverStatus::kNonFiniteInput)
      .value("INFEASIBLE", eik::SolverStatus::kInfeasible)
      .value("NO_PROGRESS", eik::SolverStatus::kNoProgress)
      .value("COLLISION_VIOLATED", eik::SolverStatus::kCollisionViolated);

  nb::enum_<eik::CollisionTuningMode>(
      m, "CollisionTuningMode",
      "High-level collision tuning presets for speed/accuracy tradeoffs")
      .value("PRECISE", eik::CollisionTuningMode::kPrecise)
      .value("BALANCED", eik::CollisionTuningMode::kBalanced)
      .value("SPEED", eik::CollisionTuningMode::kSpeed);

  // Types
  nb::class_<eik::BasicSolverConfig>(m, "BasicSolverConfig",
                                     "Basic configuration for solver")
      .def(nb::init<>())
      .def_rw("epsilon", &eik::BasicSolverConfig::epsilon)
      .def_rw("iteration_limit", &eik::BasicSolverConfig::iteration_limit)
      .def_rw("regularization", &eik::BasicSolverConfig::regularization);

  // (Intentionally do not bind VelocitySolverConfig/RegularizedInverseConfig
  // yet to avoid import-time issues.)

  nb::class_<eik::JointConfiguration>(m, "JointConfiguration",
                                      "Joint positions container")
      .def(nb::init<>())
      .def_rw("positions", &eik::JointConfiguration::positions);

  nb::class_<eik::SolverResult>(m, "SolverResult", "Result from IK solve step")
      .def_ro("solution", &eik::SolverResult::solution)
      .def_ro("status", &eik::SolverResult::status)
      .def_ro("computation_time_ms", &eik::SolverResult::computation_time_ms)
      .def_ro("iterations", &eik::SolverResult::iterations)
      .def_ro("final_error", &eik::SolverResult::final_error)
      .def_ro("task_scales", &eik::SolverResult::task_scales)
      .def_ro("task_errors", &eik::SolverResult::task_errors)
      .def_ro("task_modes_effective", &eik::SolverResult::task_modes_effective)
      .def_ro("task_used_fallback", &eik::SolverResult::task_used_fallback)
      .def_ro("status_message", &eik::SolverResult::status_message)
      .def_ro("condition_number", &eik::SolverResult::condition_number);

  nb::class_<eik::VelocitySolverResult, eik::SolverResult>(
      m, "VelocitySolverResult", "Extended result for velocity-level solving")
      .def_ro("saturated_joints", &eik::VelocitySolverResult::saturated_joints)
      .def_ro("limits_applied", &eik::VelocitySolverResult::limits_applied)
      .def_prop_ro(
          "joint_velocities",
          [](const eik::VelocitySolverResult &r) { return r.joint_velocities; })
      .def_ro("pinocchio_kinematics_time_ms",
              &eik::VelocitySolverResult::pinocchio_kinematics_time_ms,
              "Time spent in Pinocchio forward kinematics (ms)")
      .def_ro("collision_constraint_time_ms",
              &eik::VelocitySolverResult::collision_constraint_time_ms,
              "Time spent evaluating collision distances/constraints (ms)")
      .def_ro("task_update_time_ms",
              &eik::VelocitySolverResult::task_update_time_ms,
              "Time spent updating tasks (Jacobian computation) (ms)")
      .def_ro("solver_computation_time_ms",
              &eik::VelocitySolverResult::solver_computation_time_ms,
              "Time spent in core solver computation (ms)")
      .def_ro("constraint_setup_time_ms",
              &eik::VelocitySolverResult::constraint_setup_time_ms,
              "Time spent setting up constraint matrices (ms)")
      .def_ro("collision_pairs_considered",
              &eik::VelocitySolverResult::collision_pairs_considered,
              "Number of collision pairs considered this solve")
      .def_ro("collision_exact_distance_queries",
              &eik::VelocitySolverResult::collision_exact_distance_queries,
              "Number of exact collision distance queries this solve")
      .def_ro("collision_bound_culled_pairs",
              &eik::VelocitySolverResult::collision_bound_culled_pairs,
              "Number of pairs culled by conservative bounds this solve")
      .def_ro("collision_budget_exhausted",
              &eik::VelocitySolverResult::collision_budget_exhausted,
              "Whether the collision refinement budget was exhausted")
      .def_ro("collision_sphere_culled_pairs",
              &eik::VelocitySolverResult::collision_sphere_culled_pairs,
              "Number of collision pairs culled by sphere broadphase.");

  nb::class_<eik::VelocityBoxHeadroomPolicy>(
      m, "VelocityBoxHeadroomPolicy",
      "Shared velocity-box minimum-headroom policy")
      .def(nb::init<>())
      .def_rw("enabled", &eik::VelocityBoxHeadroomPolicy::enabled,
              "Enable minimum velocity headroom away from nearby limits")
      .def_rw("fraction", &eik::VelocityBoxHeadroomPolicy::fraction,
              "Fraction in [0, 1] of velocity limit used as minimum headroom")
      .def_rw("activation_margin",
              &eik::VelocityBoxHeadroomPolicy::activation_margin,
              "Position-margin threshold above which headroom can be injected");

  nb::class_<eik::TorsoPoseConstraintOptions>(
      m, "TorsoPoseConstraintOptions",
      "Optional torso orientation tracking and 6D torso pose bounds for "
      "position IK")
      .def(nb::init<>())
      .def_rw("enabled", &eik::TorsoPoseConstraintOptions::enabled,
              "Enable torso secondary objective/constraints")
      .def_rw("frame_name", &eik::TorsoPoseConstraintOptions::frame_name,
              "Torso frame name")
      .def_prop_rw(
          "target_orientation",
          [](const eik::TorsoPoseConstraintOptions &opt) -> nb::object {
            if (opt.target_orientation.has_value()) {
              return nb::cast(opt.target_orientation.value());
            }
            return nb::none();
          },
          [](eik::TorsoPoseConstraintOptions &opt, nb::object obj) {
            if (!obj.is_none()) {
              opt.target_orientation = nb::cast<Eigen::Matrix3d>(obj);
            } else {
              opt.target_orientation.reset();
            }
          },
          "Optional torso target orientation (3x3 rotation matrix). "
          "None => seed torso orientation")
      .def_rw("orientation_mask",
              &eik::TorsoPoseConstraintOptions::orientation_mask,
              "Orientation mask [roll, pitch, yaw] in radians "
              "(1=enabled, 0=disabled)")
      .def_rw("orientation_gain",
              &eik::TorsoPoseConstraintOptions::orientation_gain,
              "Secondary torso orientation objective gain")
      .def_prop_rw(
          "pose_bounds_reference_pose",
          [](const eik::TorsoPoseConstraintOptions &opt) -> nb::object {
            if (opt.pose_bounds_reference_pose.has_value()) {
              return nb::cast(opt.pose_bounds_reference_pose.value());
            }
            return nb::none();
          },
          [](eik::TorsoPoseConstraintOptions &opt, nb::object obj) {
            if (!obj.is_none()) {
              opt.pose_bounds_reference_pose =
                  nb::cast<Eigen::Matrix4d>(obj);
            } else {
              opt.pose_bounds_reference_pose.reset();
            }
          },
          "Optional 4x4 homogeneous torso reference for pose bounds only. "
          "None => use torso frame at seed_q. Fixed for all inner iterations; "
          "use in teleop to anchor bounds to session start (not each tick's seed).")
      .def_prop_rw(
          "pose_lower_bounds",
          [](const eik::TorsoPoseConstraintOptions &opt) -> nb::object {
            if (opt.pose_lower_bounds.has_value()) {
              return nb::cast(opt.pose_lower_bounds.value());
            }
            return nb::none();
          },
          [](eik::TorsoPoseConstraintOptions &opt, nb::object obj) {
            if (!obj.is_none()) {
              opt.pose_lower_bounds = nb::cast<Eigen::VectorXd>(obj);
            } else {
              opt.pose_lower_bounds.reset();
            }
          },
          "Optional 6D lower torso pose bounds vs pose reference (see "
          "pose_bounds_reference_pose). Order/units: "
          "[x,y,z,rx,ry,rz] = [m,m,m,rad,rad,rad]")
      .def_prop_rw(
          "pose_upper_bounds",
          [](const eik::TorsoPoseConstraintOptions &opt) -> nb::object {
            if (opt.pose_upper_bounds.has_value()) {
              return nb::cast(opt.pose_upper_bounds.value());
            }
            return nb::none();
          },
          [](eik::TorsoPoseConstraintOptions &opt, nb::object obj) {
            if (!obj.is_none()) {
              opt.pose_upper_bounds = nb::cast<Eigen::VectorXd>(obj);
            } else {
              opt.pose_upper_bounds.reset();
            }
          },
          "Optional 6D upper torso pose bounds vs pose reference (see "
          "pose_bounds_reference_pose). Order/units: "
          "[x,y,z,rx,ry,rz] = [m,m,m,rad,rad,rad]")
      .def_rw("pose_axis_mask", &eik::TorsoPoseConstraintOptions::pose_axis_mask,
              "6D axis mask for torso pose bounds [x,y,z,rx,ry,rz]")
      .def_rw("velocity_limits",
              &eik::TorsoPoseConstraintOptions::velocity_limits,
              "Per-axis torso pose bound velocity limits (6D). "
              "Units: [m/s,m/s,m/s,rad/s,rad/s,rad/s]")
      .def_rw("acceleration_limits",
              &eik::TorsoPoseConstraintOptions::acceleration_limits,
              "Per-axis torso pose bound acceleration limits (6D). "
              "Units: [m/s^2,m/s^2,m/s^2,rad/s^2,rad/s^2,rad/s^2]")
      .def_rw("velocity_box_headroom",
              &eik::TorsoPoseConstraintOptions::velocity_box_headroom,
              "Shared velocity-box minimum-headroom policy for torso pose "
              "bound rows")
      .def_rw("pose_bound_softening_enabled",
              &eik::TorsoPoseConstraintOptions::pose_bound_softening_enabled,
              "Deprecated alias for velocity_box_headroom.enabled")
      .def_rw("pose_bound_softening_fraction",
              &eik::TorsoPoseConstraintOptions::pose_bound_softening_fraction,
              "Deprecated alias for velocity_box_headroom.fraction");

  nb::class_<eik::PositionIKOptions>(m, "PositionIKOptions",
                                     "Options for position-level IK solving")
      .def(nb::init<>())
      .def_rw("position_tolerance", &eik::PositionIKOptions::position_tolerance)
      .def_rw("orientation_tolerance",
              &eik::PositionIKOptions::orientation_tolerance)
      .def_rw("max_iterations", &eik::PositionIKOptions::max_iterations)
      .def_rw("dt", &eik::PositionIKOptions::dt)
      .def_rw("stagnation_tolerance",
              &eik::PositionIKOptions::stagnation_tolerance,
              "Minimum improvement required per iteration to avoid early "
              "termination")
      .def_rw("stagnation_iterations",
              &eik::PositionIKOptions::stagnation_iterations,
              "Number of consecutive stagnant iterations before aborting")
      .def_rw("classify_stagnation_as_no_progress",
              &eik::PositionIKOptions::classify_stagnation_as_no_progress,
              "Default True. When enabled, stagnation-triggered exits return "
              "NO_PROGRESS instead of INFEASIBLE")
      .def_rw("limit_change_from_seed",
              &eik::PositionIKOptions::limit_change_from_seed,
              "When True, tighten each solve_position iteration so q stays "
              "within seed_q ± velocity_limit*dt")
      .def_prop_rw(
          "nullspace_bias",
          [](const eik::PositionIKOptions &opt) -> nb::object {
            if (opt.nullspace_bias.has_value()) {
              return nb::cast(opt.nullspace_bias.value());
            }
            return nb::none();
          },
          [](eik::PositionIKOptions &opt, nb::object obj) {
            if (!obj.is_none()) {
              opt.nullspace_bias = nb::cast<Eigen::VectorXd>(obj);
            } else {
              opt.nullspace_bias.reset();
            }
          },
          "Target configuration for nullspace control (optional)")
      .def_rw("nullspace_gain", &eik::PositionIKOptions::nullspace_gain,
              "Nullspace task gain/weight")
      .def_rw(
          "nullspace_active_joints",
          &eik::PositionIKOptions::nullspace_active_joints,
          "List of joint indices for nullspace control (empty = all joints)")
      .def_prop_rw(
          "nullspace_joint_weights",
          [](const eik::PositionIKOptions &opt) -> nb::object {
            if (opt.nullspace_joint_weights.has_value()) {
              return nb::cast(opt.nullspace_joint_weights.value());
            }
            return nb::none();
          },
          [](eik::PositionIKOptions &opt, nb::object obj) {
            if (!obj.is_none()) {
              opt.nullspace_joint_weights = nb::cast<Eigen::VectorXd>(obj);
            } else {
              opt.nullspace_joint_weights.reset();
            }
          },
          "Optional per-joint nullspace weights. Size must match nv (all joints) "
          "or len(nullspace_active_joints)")
      .def_rw("torso_constraint", &eik::PositionIKOptions::torso_constraint,
              "Optional torso orientation task and torso pose bounds for "
              "secondary-objective enforcement")
      .def_rw("excluded_joint_indices",
              &eik::PositionIKOptions::excluded_joint_indices,
              "Nv-indices excluded from the temporary frame and nullspace "
              "posture tasks in solve_position (same convention as "
              "Task.set_excluded_joint_indices and "
              "PositionStepOptions.excluded_joint_indices)")
      .def_rw("max_linear_step", &eik::PositionIKOptions::max_linear_step,
              "Maximum linear step per iteration (meters)")
      .def_rw("max_angular_step", &eik::PositionIKOptions::max_angular_step,
              "Maximum angular step per iteration (radians)")
      .def_rw("position_gain", &eik::PositionIKOptions::position_gain,
              "Linear error gain used by position IK")
      .def_rw("orientation_gain", &eik::PositionIKOptions::orientation_gain,
              "Angular error gain used by position IK")
      .def_rw("primary_solve_mode", &eik::PositionIKOptions::primary_solve_mode,
              "Primary end-effector solve mode used by position IK")
      .def_rw("primary_allow_min_error_fallback",
              &eik::PositionIKOptions::primary_allow_min_error_fallback,
              "Allow SCALE primary task to fall back to MIN_ERROR in position IK")
      .def_rw("stall_recovery",
              &eik::PositionIKOptions::stall_recovery,
              "When True, enable automatic stall recovery. Detects consecutive "
              "stalled velocity solves (non-success with near-zero ||dq||) "
              "and temporarily relaxes/restores the effective collision "
              "min_distance. Default False.");

  nb::class_<eik::PositionStepOptions>(
      m, "PositionStepOptions",
      "Options for solve_position_step(): gains, timestep, and optional joint "
      "controls. Field names match PositionIKOptions where applicable "
      "(excluded_joint_indices uses the same nv convention as "
      "PositionIKOptions and Task.set_excluded_joint_indices).")
      .def(nb::init<>())
      .def_rw("position_gain", &eik::PositionStepOptions::position_gain,
              "Multiplier on the linear pose error (default 1.0)")
      .def_rw("orientation_gain", &eik::PositionStepOptions::orientation_gain,
              "Multiplier on the angular pose error (default 1.0)")
      .def_rw("max_steps", &eik::PositionStepOptions::max_steps,
              "Number of velocity-IK iterations per call (default 1)")
      .def_rw("dt", &eik::PositionStepOptions::dt,
              "Integration timestep per step; <=0 uses solver.dt (default -1)")
      .def_rw("max_linear_speed", &eik::PositionStepOptions::max_linear_speed,
              "Maximum linear speed magnitude in solve_position_step (m/s); <=0 means unlimited")
      .def_rw("max_angular_speed", &eik::PositionStepOptions::max_angular_speed,
              "Maximum angular speed magnitude in solve_position_step (rad/s); <=0 means unlimited")
      .def_rw("torso_constraint", &eik::PositionStepOptions::torso_constraint,
              "Optional torso orientation task and torso pose bounds for step IK")
      .def_rw("excluded_joint_indices",
              &eik::PositionStepOptions::excluded_joint_indices,
              "Same role as PositionIKOptions.excluded_joint_indices: merged "
              "into exclusions on the driven pose task(s) and all PostureTasks "
              "for each inner solve_velocity (restored after the step).")
      .def_rw("locked_joint_indices",
              &eik::PositionStepOptions::locked_joint_indices,
              "Nv-indices forced to v=0 inside the velocity QP (bounds + "
              "zeroed Jacobian columns). Preferred for consistent diagnostics. "
              "Out-of-range → InvalidInput.")
      .def_rw("integration_zero_velocity_indices",
              &eik::PositionStepOptions::integration_zero_velocity_indices,
              "Nv-indices zeroed on joint_velocities after each inner "
              "solve_velocity and before integrate (legacy post-QP mask). "
              "Out-of-range → InvalidInput.")
      .def_rw("stall_recovery",
              &eik::PositionStepOptions::stall_recovery,
              "When True, enable automatic stall recovery. Detects consecutive "
              "stalled velocity solves (non-success with near-zero ||dq||) "
              "and temporarily relaxes/restores the effective collision "
              "min_distance. Default False.")
      .def_rw("elastic_band",
              &eik::PositionStepOptions::elastic_band,
              "When True, enable elastic band joint limit expansion. "
              "Temporarily expands joint position limit margins when the "
              "solver is overconstrained by joint limits, keeping more DOFs "
              "active. Uses proven defaults (delta_max=0.05). Default False.")
      .def_rw("limit_change_from_seed",
              &eik::PositionStepOptions::limit_change_from_seed,
              "When True, tighten each inner step so q stays within "
              "current_q(at entry) ± velocity_limit*dt")
      .def_rw("no_progress_max_steps",
              &eik::PositionStepOptions::no_progress_max_steps,
              "Consecutive low-progress inner steps before returning "
              "NO_PROGRESS (<=0 disables)")
      .def_rw("no_progress_error_tolerance",
              &eik::PositionStepOptions::no_progress_error_tolerance,
              "Threshold on change in combined pose error used by "
              "no-progress detection")
      .def_rw("no_progress_dq_norm_tolerance",
              &eik::PositionStepOptions::no_progress_dq_norm_tolerance,
              "Threshold on ||dq|| used by no-progress detection")
      .def_rw("adaptive_dt", &eik::PositionStepOptions::adaptive_dt,
              "Enable adaptive integration timestep. Scales dt by "
              "(position_error / reference_distance), clamped to [1x, max_scale]. "
              "Improves large-jump convergence without requiring gain tuning.")
      .def_rw("adaptive_dt_max_scale",
              &eik::PositionStepOptions::adaptive_dt_max_scale,
              "Maximum dt multiplier when adaptive_dt=True (default 5.0)")
      .def_rw("adaptive_dt_reference_distance",
              &eik::PositionStepOptions::adaptive_dt_reference_distance,
              "Position error (m) at which dt scale = 1.0 (default 0.05)");

  nb::class_<eik::TaskTarget>(
      m, "TaskTarget",
      "Per-task target and gains for multi-task solve_position_step()")
      .def(nb::init<>())
      .def(nb::init<const std::string &, const Eigen::Matrix4d &, double, double>(),
           nb::arg("task_name"), nb::arg("target_pose"),
           nb::arg("position_gain") = 1.0, nb::arg("orientation_gain") = 1.0)
      .def_static(
          "from_se3",
          [](const std::string &task_name, const pinocchio::SE3 &target_pose,
             double position_gain, double orientation_gain) {
            eik::TaskTarget t;
            t.task_name = task_name;
            t.target_pose = target_pose.toHomogeneousMatrix();
            t.position_gain = position_gain;
            t.orientation_gain = orientation_gain;
            return t;
          },
          nb::arg("task_name"), nb::arg("target_pose"),
          nb::arg("position_gain") = 1.0, nb::arg("orientation_gain") = 1.0)
      .def_rw("task_name", &eik::TaskTarget::task_name)
      .def_rw("target_pose", &eik::TaskTarget::target_pose)
      .def_rw("position_gain", &eik::TaskTarget::position_gain)
      .def_rw("orientation_gain", &eik::TaskTarget::orientation_gain);

  nb::class_<eik::PositionIKResult, eik::VelocitySolverResult>(
      m, "PositionIKResult", "Result from position-level IK solving")
      .def_ro("q_solution", &eik::PositionIKResult::q_solution)
      .def_prop_ro(
          "achieved_pose",
          [](const eik::PositionIKResult &r) { return r.achieved_pose; })
      .def_ro("position_error", &eik::PositionIKResult::position_error)
      .def_ro("orientation_error", &eik::PositionIKResult::orientation_error)
      .def_ro("iterations_used", &eik::PositionIKResult::iterations_used)
      .def_ro("position_error_trace",
              &eik::PositionIKResult::position_error_trace)
      .def_ro("orientation_error_trace",
              &eik::PositionIKResult::orientation_error_trace)
      .def_ro("collision_rejection_count",
              &eik::PositionIKResult::collision_rejection_count)
      .def_ro("stall_escape_count",
              &eik::PositionIKResult::stall_escape_count);

  m.def("pose_error_norm", &eik::calculateConfigurationDistance, "current"_a,
        "target"_a,
        R"pbdoc(
          Compute L2 norm between two pose vectors.
          Returns inf for invalid inputs.
          )pbdoc");

  // No additional velocity IK bindings defined

  // Eigen-first overloads (preferred)
  m.def(
      "computeMultiObjectiveVelocitySolutionEigen",
      [](const std::vector<Eigen::VectorXd> &goals,
         const std::vector<Eigen::MatrixXd> &jacobians,
         const Eigen::MatrixXd &C, const Eigen::VectorXd &lower_limits,
         const Eigen::VectorXd &upper_limits, double solver_tolerance,
         double solver_tight_tolerance, unsigned int max_iters,
         double norm_threshold, unsigned int max_zero_scale_iters,
         double sr_tolerance, double sr_damping) {
        eik::VelocitySolverConfig p;
        p.epsilon = solver_tolerance;
        p.precision_threshold = solver_tight_tolerance;
        p.iteration_limit = max_iters;
        p.magnitude_limit = norm_threshold;
        p.stall_detection_count = max_zero_scale_iters;
        p.regularization_config.epsilon = sr_tolerance;
        p.regularization_config.regularization_factor = sr_damping;
        return eik::computeMultiObjectiveVelocitySolutionEigen(
            goals, jacobians, C, lower_limits, upper_limits, p);
      },
      "goals"_a, "jacobians"_a, "C"_a, "lower_limits"_a, "upper_limits"_a,
      "solver_tolerance"_a = 1e-6, "solver_tight_tolerance"_a = 1e-10,
      "max_iters"_a = 20, "norm_threshold"_a = 1e10,
      "max_zero_scale_iters"_a = 2, "sr_tolerance"_a = 1e-6,
      "sr_damping"_a = 1e-1,
      R"pbdoc(
          Full multi-task velocity IK using Eigen types. Preferred API.
          )pbdoc");

  // Module metadata
  m.attr("__version__") = "0.2.0";
  m.attr("DEFAULT_REGULARIZATION") = eik::BasicSolverConfig{}.regularization;

  // Bind robot model with Pinocchio integration
  bind_robot_model(m);

  // Bind task framework
  bind_tasks(m);

  // Bind high-level kinematics solver
  bind_kinematics_solver(m);

  // Bind pose metrics free functions
  bind_pose_metrics(m);
}
