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
#include <nanobind/stl/vector.h>

#include <embodik/ik_baseline.hpp>
#include <embodik/types.hpp>

namespace nb = nanobind;
namespace eik = embodik;

// Forward declarations for sub-module bindings
void bind_robot_model(nb::module_ &m);
void bind_tasks(nb::module_ &m);
void bind_kinematics_solver(nb::module_ &m);

NB_MODULE(_embodik_impl, m) {
  m.doc() = "embodiK: High-performance inverse kinematics with Pinocchio";
  using nb::literals::operator""_a;

  // Enums
  nb::enum_<eik::SolverStatus>(m, "SolverStatus",
                               "Status codes for embodiK baseline")
      .value("SUCCESS", eik::SolverStatus::kSuccess)
      .value("INVALID_INPUT", eik::SolverStatus::kInvalidInput)
      .value("NUMERICAL_ERROR", eik::SolverStatus::kNumericalError);

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
      .def_ro("task_errors", &eik::SolverResult::task_errors);

  nb::class_<eik::VelocitySolverResult, eik::SolverResult>(
      m, "VelocitySolverResult", "Extended result for velocity-level solving")
      .def_ro("saturated_joints", &eik::VelocitySolverResult::saturated_joints)
      .def_ro("limits_applied", &eik::VelocitySolverResult::limits_applied)
      .def_prop_ro("joint_velocities", [](const eik::VelocitySolverResult &r) {
        return r.joint_velocities;
      });

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
      .def_rw("max_linear_step", &eik::PositionIKOptions::max_linear_step,
              "Maximum linear step per iteration (meters)")
      .def_rw("max_angular_step", &eik::PositionIKOptions::max_angular_step,
              "Maximum angular step per iteration (radians)");

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
              &eik::PositionIKResult::orientation_error_trace);

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
  m.attr("__version__") = "0.1.0";
  m.attr("DEFAULT_REGULARIZATION") = eik::BasicSolverConfig{}.regularization;

  // Bind robot model with Pinocchio integration
  bind_robot_model(m);

  // Bind task framework
  bind_tasks(m);

  // Bind high-level kinematics solver
  bind_kinematics_solver(m);
}
