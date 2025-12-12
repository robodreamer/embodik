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

#pragma once

#include <Eigen/Core>
#include <optional>
#include <vector>

namespace embodik {

enum class SolverStatus {
  kSuccess = 0,
  kInvalidInput = 1,
  kNumericalError = 2
};

struct BasicSolverConfig {
  double epsilon = 1e-6;
  unsigned int iteration_limit = 20; // Reserved for future multi-step solvers
  double regularization = 1e-1;      // Tikhonov regularization parameter
};

struct JointConfiguration {
  std::vector<double> positions;
};

struct SolverResult {
  std::vector<double> solution; // dq or q depending on solver
  SolverStatus status = SolverStatus::kInvalidInput;
  double computation_time_ms = 0.0;
  unsigned int iterations = 0;
  double final_error = 0.0;        // ||J dq - v|| for velocity IK
  std::vector<double> task_scales; // for full velocity IK
  std::vector<double> task_errors; // Individual task errors
};

// Extended result for velocity-level solving
struct VelocitySolverResult : public SolverResult {
  std::vector<int> saturated_joints; // Indices of joints at velocity limits
  bool limits_applied = false;       // Whether limits were enforced
  Eigen::VectorXd
      joint_velocities; // Convenience access to solution as VectorXd
};

// Configuration for regularized matrix inversion
struct RegularizedInverseConfig {
  double epsilon = 1e-6; // Numerical tolerance threshold
  double regularization_factor =
      1e-1; // Regularization coefficient for stability
};

struct VelocitySolverConfig {
  double epsilon = 1e-6;
  double precision_threshold = 1e-10;
  unsigned int iteration_limit = 20;
  double magnitude_limit = 1e10;
  unsigned int stall_detection_count = 2;
  RegularizedInverseConfig regularization_config{};
};

// Position IK options
struct PositionIKOptions {
  double position_tolerance = 1e-3;    // Position error tolerance (meters)
  double orientation_tolerance = 1e-3; // Orientation error tolerance (radians)
  int max_iterations = 100;            // Maximum iterations
  double dt = 0.01;                    // Integration timestep
  double stagnation_tolerance =
      1e-6;                      // Minimum improvement required per iteration
  int stagnation_iterations = 5; // Max stagnant iterations before abort

  // Nullspace control
  std::optional<Eigen::VectorXd>
      nullspace_bias;          // Target configuration for nullspace
  double nullspace_gain = 0.1; // Nullspace task gain/weight
  std::vector<int> nullspace_active_joints; // Empty = all joints active

  // Step size limits (for stability)
  double max_linear_step = 0.3;  // Max meters per iteration
  double max_angular_step = 0.3; // Max radians per iteration
};

// Position IK result
struct PositionIKResult : public VelocitySolverResult {
  Eigen::VectorXd q_solution;               // Final joint configuration
  Eigen::Matrix4d achieved_pose;            // Final achieved pose
  double position_error = 0.0;              // Final position error
  double orientation_error = 0.0;           // Final orientation error
  int iterations_used = 0;                  // Number of iterations used
  std::vector<double> position_error_trace; // Per-iteration position error
  std::vector<double>
      orientation_error_trace; // Per-iteration orientation error
};

} // namespace embodik
