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

#pragma once

#include <cstdint>
#include <Eigen/Core>
#include <optional>
#include <string>
#include <vector>

namespace embodik {

enum class TaskSolveMode {
  kScale = 0,
  kMinError = 1,
  /// SCALE with automatic elastic band joint limit expansion.
  /// Behaves identically to kScale in the SNS solver, but automatically
  /// enables the elastic band mechanism that temporarily expands joint
  /// limit margins when the solver is overconstrained by joint limits.
  /// This keeps more DOFs active, preventing premature task scale collapse.
  /// Uses proven defaults: delta_max=0.05, expand_rate=0.01, decay_rate=0.2.
  kScaleElastic = 2,
};

enum class CollisionTuningMode {
  kPrecise = 0,
  kBalanced = 1,
  kSpeed = 2,
};

enum class TaskLayout {
  kSplit = 0,
  kMerged = 1,
};

enum class SolverRecoveryStage {
  kPrioritized = 0,
  kWeightedFallback = 1,
};

enum class SolverStatus {
  kSuccess = 0,
  kInvalidInput = 1,
  kNumericalError = 2,
  kShapeMismatch = 3,
  kEmptyProblem = 4,
  kConstraintBoundsMismatch = 5,
  kNonFiniteInput = 6,
  kInfeasible = 7,
  kNoProgress = 8,
  // Returned by solve_position_step when the input q was collision-safe but no
  // integration step (including backoffs) could maintain min_distance.
  // q_solution is set to the input q (safe position held).
  kCollisionViolated = 9
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
  std::vector<TaskSolveMode>
      task_modes_effective; // Effective mode used per task
  std::vector<bool> task_used_fallback; // True when SCALE fell back to MIN_ERROR
  std::string status_message;      // Human-readable diagnostic for failures
  double condition_number = 1.0;   // Worst-case Jacobian condition number
};

// Extended result for velocity-level solving
struct VelocitySolverResult : public SolverResult {
  std::vector<int> saturated_joints; // Indices of joints at velocity limits
  bool limits_applied = false;       // Whether limits were enforced
  Eigen::VectorXd
      joint_velocities; // Convenience access to solution as VectorXd

  // Performance breakdown (for debugging)
  double pinocchio_kinematics_time_ms = 0.0; // Forward kinematics time
  double collision_constraint_time_ms =
      0.0;                                 // Collision distance/constraint time
  double task_update_time_ms = 0.0;        // Task update (Jacobian) time
  double solver_computation_time_ms = 0.0; // Actual solver time (from backend)
  double constraint_setup_time_ms = 0.0;   // Constraint matrix setup time

  // Collision query instrumentation counters
  std::uint64_t collision_pairs_considered = 0;
  std::uint64_t collision_exact_distance_queries = 0;
  std::uint64_t collision_bound_culled_pairs = 0;
  bool collision_budget_exhausted = false;
  std::uint64_t collision_sphere_culled_pairs = 0;

  /// Read-only weighted-advisor diagnostics. These fields are populated only
  /// when SolverRuntimeConfig::weighted_advisor_enabled is true. The advisor
  /// never overwrites the prioritized solver output unless the separate
  /// weighted_fallback_enabled runtime flag is enabled.
  bool weighted_advisory_available = false;
  bool weighted_fallback_used = false;
  double weighted_advisory_v_norm = std::numeric_limits<double>::quiet_NaN();
  double weighted_advisory_pos_task_error_norm =
      std::numeric_limits<double>::quiet_NaN();
  double weighted_advisory_ori_task_error_norm =
      std::numeric_limits<double>::quiet_NaN();
  double weighted_advisory_condition_number =
      std::numeric_limits<double>::quiet_NaN();
  /// Advisor position-weight scale used for this solve. When PI adaptation is
  /// disabled, this mirrors SolverRuntimeConfig::advisor_position_weight_scale.
  double advisor_position_weight_scale_current = 1.0;
  /// True when the advisor-scale PI controller updated its internal state on
  /// this solve.
  bool advisor_scale_adapt_active = false;

  /// Active pose-task layout used by the solve.
  TaskLayout active_task_layout = TaskLayout::kSplit;
  /// Recovery stage that supplied the accepted velocity. This is separate from
  /// task layout: weighted fallback solves the active layout's constrained
  /// stacked objective; it is not a pose-task layout.
  SolverRecoveryStage recovery_stage = SolverRecoveryStage::kPrioritized;
  /// Bounded binding score from the previous constrained velocity solve used by
  /// the optional stateful layout switcher. Zero when the switcher is disabled
  /// or has not yet seen a prior solve.
  double binding_score = 0.0;
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

struct ObjectiveSolveConfig {
  int priority = 0;
  TaskSolveMode solve_mode = TaskSolveMode::kScale;
  bool allow_min_error_fallback = true;
};

// Optional torso tracking/constraint configuration for position IK.
struct VelocityBoxHeadroomPolicy {
  // Enable minimum velocity headroom away from nearby limits.
  bool enabled = false;
  // Fraction in [0, 1] of velocity_limit used as minimum headroom.
  double fraction = 0.10;
  // Margin threshold (same units as position margins) above which headroom can
  // be injected. Prevents pushing toward an already-active limit.
  double activation_margin = 0.01;
};

struct TorsoPoseConstraintOptions {
  bool enabled = false;
  std::string frame_name;
  // If unset, target orientation is captured from the seed configuration.
  std::optional<Eigen::Matrix3d> target_orientation;
  // Default "upright" behavior: constrain roll/pitch, leave yaw free.
  // Rotation components are in radians.
  Eigen::Vector3d orientation_mask = Eigen::Vector3d(1.0, 1.0, 0.0);
  double orientation_gain = 1.0;

  // Optional 6D torso pose bounds relative to a fixed torso reference pose:
  // [x, y, z, rx, ry, rz] where translation is meters and rotation is radians.
  // Set both lower and upper to enable.
  //
  // Reference pose for these bounds (world frame, same convention as
  // RobotModel::get_frame_pose):
  // - If pose_bounds_reference_pose is set (4x4 homogeneous), bounds are
  //   measured vs that transform (held fixed for the whole solve_position
  //   call; does not move with internal IK iterations).
  // - If unset, the reference is the torso frame pose at the seed
  //   configuration passed into solve_position (also fixed for all inner
  //   iterations, but recenters each outer call if seed_q changes).
  std::optional<Eigen::Matrix4d> pose_bounds_reference_pose;
  std::optional<Eigen::VectorXd> pose_lower_bounds;
  std::optional<Eigen::VectorXd> pose_upper_bounds;
  // 6D mask for bounds rows (1 = constrained, 0 = unconstrained).
  Eigen::VectorXd pose_axis_mask = Eigen::VectorXd::Ones(6);
  // Per-axis velocity/acceleration limits used for box constraints.
  // velocity_limits units: [m/s, m/s, m/s, rad/s, rad/s, rad/s]
  // acceleration_limits units: [m/s^2, m/s^2, m/s^2, rad/s^2, rad/s^2, rad/s^2]
  Eigen::VectorXd velocity_limits = Eigen::VectorXd::Constant(6, 0.5);
  Eigen::VectorXd acceleration_limits = Eigen::VectorXd::Constant(6, 1.0);
  // Shared velocity-box headroom policy for torso pose-bound rows.
  VelocityBoxHeadroomPolicy velocity_box_headroom;

  // Deprecated alias fields retained for backward compatibility.
  // These map to velocity_box_headroom.enabled/fraction in solve_position().
  // New code should use velocity_box_headroom.
  bool pose_bound_softening_enabled = false;
  double pose_bound_softening_fraction = 0.10;
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
  // Stagnation classification:
  // by default, stagnation exits are reported as kNoProgress instead of kInfeasible.
  // Disable only for backward-compatible status semantics.
  bool classify_stagnation_as_no_progress = true;
  // Optional seed-referenced bound tightening:
  // each iteration constrains dq to remain within seed_q ± v_limit*dt.
  bool limit_change_from_seed = false;

  // Nullspace control
  std::optional<Eigen::VectorXd>
      nullspace_bias;          // Target configuration for nullspace
  double nullspace_gain = 0.1; // Nullspace task gain/weight
  std::vector<int> nullspace_active_joints; // Empty = all joints active
  // Optional per-joint weights for nullspace bias. Size must match nv when
  // nullspace_active_joints is empty, else size must match active-joint count.
  std::optional<Eigen::VectorXd> nullspace_joint_weights;
  // Secondary torso objective and optional torso pose bounds.
  TorsoPoseConstraintOptions torso_constraint;

  // Step size limits (for stability)
  double max_linear_step = 0.3;  // Max meters per iteration
  double max_angular_step = 0.3; // Max radians per iteration
  double position_gain = 1.0;
  double orientation_gain = 1.0;

  // Primary EE objective solve behavior
  TaskSolveMode primary_solve_mode = TaskSolveMode::kScale;
  bool primary_allow_min_error_fallback = false;

  // Optional joint exclusions (e.g., lock torso joints in position IK).
  // Same nv-index convention as PositionStepOptions::excluded_joint_indices.
  std::vector<int> excluded_joint_indices;

  /// When true, automatically enable the solver's stall handler for this
  /// call. The handler counts consecutive stalled velocity solves (non-success
  /// with near-zero ||dq||) and temporarily relaxes/restores the effective
  /// collision min_distance. The nominal min_distance is read from the
  /// current collision constraint config. Default false (opt-in).
  bool stall_recovery = false;

  /// When true, enable elastic band joint limit expansion.
  /// Default false (opt-in).
  bool elastic_band = false;
};

// Options for solve_position_step() — lightweight struct for interactive loops.
// Field names mirror PositionIKOptions where applicable (same nv index
// convention as Task::set_excluded_joint_indices).
struct PositionStepOptions {
  double position_gain = 1.0;    // Multiplier on the linear error → velocity
  double orientation_gain = 1.0; // Multiplier on the angular error → velocity
  int max_steps = 1;             // Number of velocity-IK iterations
  double dt = -1.0;              // Integration timestep per step (≤0 → solver.dt)
  // Optional task-space speed caps (0 or negative = unlimited).
  double max_linear_speed = 0.0;  // m/s cap on ||v_linear||
  double max_angular_speed = 0.0; // rad/s cap on ||v_angular||
  // Optional torso orientation task and torso pose bounds. Bounds are enforced
  // in solve_position_step via additional inequality rows, consistent with
  // solve_position semantics.
  TorsoPoseConstraintOptions torso_constraint;
  /// Same intent as PositionIKOptions::excluded_joint_indices: treat these
  /// nv-indices as inactive in solve_position_step without mutating registered
  /// task exclusion lists. Internally this maps to zero-velocity locks during
  /// each inner solve_velocity() call. Empty = no effect.
  std::vector<int> excluded_joint_indices;
  /// Enforce v_i = 0 in the velocity QP for these nv-indices (tight bounds +
  /// zero Jacobian columns on all objectives and inequality Jacobians). Empty =
  /// no effect. **Warning:** this alters the stacked QP (not equivalent to
  /// post-QP velocity masking) and can reroute motion onto other joints; for
  /// interactive ``solve_position_step`` loops, prefer
  /// ``integration_zero_velocity_indices`` unless you need reported
  /// ``joint_velocities`` to match the integrated step. Any index outside [0,
  /// nv) yields kInvalidInput at solve_position_step entry.
  std::vector<int> locked_joint_indices;
  /// After each inner solve_velocity(), set joint_velocities[i]=0 for these
  /// nv-indices before pinocchio::integrate (legacy “zero dq then integrate”).
  /// The QP may still assign non-zero velocity there. If an index is also in
  /// locked_joint_indices, the QP already yields zero; this pass is redundant.
  /// **Typical choice** for teleop / marker IK: use this field alone (leave
  /// ``locked_joint_indices`` empty) to match legacy mask-then-integrate
  /// behavior. Any index outside [0, nv) yields kInvalidInput at entry.
  std::vector<int> integration_zero_velocity_indices;
  /// When true, enable the solver's stall handler. The handler counts
  /// consecutive stalled velocity solves (non-success with near-zero ||dq||)
  /// and temporarily relaxes/restores the effective collision min_distance.
  /// The handler is enabled once and persists across successive
  /// solve_position_step calls so that stall counts accumulate correctly in
  /// outer loops (e.g. teleop ticks with max_steps=1). Call
  /// disable_stall_handler() explicitly to tear it down.
  /// Default false (opt-in).
  bool stall_recovery = false;
  /// When true, enable the elastic band joint limit expansion mechanism.
  /// The handler temporarily expands joint position limit margins when the
  /// solver is overconstrained by joint limits, keeping more DOFs active.
  /// The handler is enabled once and persists across successive
  /// solve_position_step calls. Uses proven defaults (delta_max=0.05).
  /// Default false (opt-in).
  bool elastic_band = false;
  // Optional seed-referenced bound tightening:
  // each inner step constrains dq so integrated q stays within
  // initial_current_q ± v_limit*dt.
  bool limit_change_from_seed = false;
  // Optional early exit when progress is below threshold for consecutive steps.
  // Set <=0 to disable (default).
  int no_progress_max_steps = 0;
  double no_progress_error_tolerance = 1e-8;
  double no_progress_dq_norm_tolerance = 1e-8;
  // Adaptive dt: scale the integration timestep proportional to current
  // position error so large target jumps converge faster without tuning gains.
  //   effective_dt = clamp(dt * (pos_error / reference_distance), dt, dt * max_scale)
  // Keeps snappy far-from-target response while reverting to base dt near target.
  // Only active when adaptive_dt=true and adaptive_dt_reference_distance > 0.
  bool adaptive_dt = false;
  double adaptive_dt_max_scale = 5.0;            // Max multiplier on base dt
  double adaptive_dt_reference_distance = 0.05;  // Distance (m) where scale = 1.0
};

// Per-task target for multi-task solve_position_step().
struct TaskTarget {
  std::string task_name;
  Eigen::Matrix4d target_pose = Eigen::Matrix4d::Identity();
  double position_gain = 1.0;
  double orientation_gain = 1.0;
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

  /// Number of integration steps rejected because they would have created
  /// or deepened collision penetration.
  int collision_rejection_count = 0;
  /// Number of steps where a Jacobian-based escape nudge was applied to
  /// move the configuration out of collision penetration during a stall.
  int stall_escape_count = 0;
};

/// Lightweight, derived bundle of diagnostics from a PositionIKResult.
///
/// Exposed in Python as ``PositionIKResult.diagnostics`` so wrappers can read
/// one object instead of scattered top-level fields. The struct holds copies,
/// not references, and does not add solver state.
struct SolveDiagnostics {
  /// Mirrors PositionIKResult::collision_rejection_count.
  int collision_rejection_count = 0;
  /// Mirrors PositionIKResult::stall_escape_count.
  int stall_escape_count = 0;
  /// Mirrors SolverResult::condition_number.
  double condition_number = 1.0;
  /// Mirrors SolverResult::task_scales.
  std::vector<double> task_scales;
  /// Mirrors SolverResult::task_used_fallback.
  std::vector<bool> task_used_fallback;
  /// Mirrors SolverResult::task_modes_effective.
  std::vector<TaskSolveMode> task_modes_effective;
  /// True when an existing intervention counter indicates a non-nominal solve.
  bool any_intervention = false;
  /// Mirrors VelocitySolverResult::weighted_advisory_available.
  bool weighted_advisory_available = false;
  /// Mirrors VelocitySolverResult::weighted_fallback_used.
  bool weighted_fallback_used = false;
  /// Mirrors VelocitySolverResult::weighted_advisory_v_norm.
  double weighted_advisory_v_norm = std::numeric_limits<double>::quiet_NaN();
  /// Mirrors VelocitySolverResult::weighted_advisory_pos_task_error_norm.
  double weighted_advisory_pos_task_error_norm =
      std::numeric_limits<double>::quiet_NaN();
  /// Mirrors VelocitySolverResult::weighted_advisory_ori_task_error_norm.
  double weighted_advisory_ori_task_error_norm =
      std::numeric_limits<double>::quiet_NaN();
  /// Mirrors VelocitySolverResult::weighted_advisory_condition_number.
  double weighted_advisory_condition_number =
      std::numeric_limits<double>::quiet_NaN();
  /// Mirrors VelocitySolverResult::advisor_position_weight_scale_current.
  double advisor_position_weight_scale_current = 1.0;
  /// Mirrors VelocitySolverResult::advisor_scale_adapt_active.
  bool advisor_scale_adapt_active = false;
  /// Mirrors VelocitySolverResult::active_task_layout.
  TaskLayout active_task_layout = TaskLayout::kSplit;
  /// Mirrors VelocitySolverResult::recovery_stage.
  SolverRecoveryStage recovery_stage = SolverRecoveryStage::kPrioritized;
  /// Mirrors VelocitySolverResult::binding_score.
  double binding_score = 0.0;
};

/// Bundled runtime defaults for interactive solve loops.
///
/// Purely additive: existing setters and per-call PositionStepOptions remain
/// authoritative. KinematicsSolver::configure_runtime() stamps ``damping`` onto
/// the solver and stores the position-step fields so callers can materialize
/// fresh options through make_position_step_options().
struct SolverRuntimeConfig {
  /// Singular-value damping for the regularized pseudoinverse.
  double damping = 0.1;
  /// Default PositionStepOptions::max_steps value.
  int position_step_max_steps = 1;
  /// Default PositionStepOptions::adaptive_dt value.
  bool adaptive_dt = false;
  /// Default PositionStepOptions::adaptive_dt_max_scale value.
  double adaptive_dt_max_scale = 5.0;
  /// Default PositionStepOptions::adaptive_dt_reference_distance value.
  double adaptive_dt_reference_distance = 0.05;
  /// Enable read-only constrained weighted-advisor diagnostics. When true, each
  /// velocity solve also computes a weighted stacked MIN_ERROR candidate under
  /// the same hard constraints. The prioritized SNS output remains
  /// authoritative unless weighted_fallback_enabled also accepts the candidate.
  bool weighted_advisor_enabled = false;
  /// Allow the constrained weighted candidate to replace a non-success
  /// prioritized solve when the candidate satisfies the hard constraints.
  /// Default true; the candidate is computed lazily only after a non-success
  /// prioritized solve unless weighted_advisor_enabled is also true.
  bool weighted_fallback_enabled = true;
  /// Multiplicative weight scale for FRAME_POSITION objectives in the
  /// diagnostic advisor only. Non-finite or negative values are treated as 1.0.
  double advisor_position_weight_scale = 1.0;
  /// Multiplicative weight scale for FRAME_ORIENTATION objectives in the
  /// diagnostic advisor only. Non-finite or negative values are treated as 1.0.
  double advisor_orientation_weight_scale = 1.0;
  /// Enable default-off advisor-scale PI adaptation for experiments. The
  /// adapted value affects only constrained weighted-advisor/fallback row
  /// weights; it never changes prioritized task targets or hard constraints.
  bool enable_advisor_scale_adapt = false;
  /// Integral gain for advisor-scale adaptation. Positive error means the
  /// weighted advisor's position residual ratio is above target, so the
  /// position weight is increased.
  double advisor_scale_adapt_ki = 0.25;
  /// Target residual ratio: pos_error / (pos_error + ori_error).
  double advisor_scale_adapt_target_ratio = 0.5;
  /// Lower bound for the adapted advisor position-weight scale.
  double advisor_scale_min = 0.25;
  /// Upper bound for the adapted advisor position-weight scale.
  double advisor_scale_max = 8.0;
  /// Minimum accumulated solver time between advisor-scale PI updates.
  double advisor_scale_epoch_s = 0.1;
  /// Enable opt-in stateful PoseTaskGroup layout switching. Auto-switch groups
  /// own both a merged FRAME_POSE task and split FRAME_POSITION/FRAME_ORIENTATION
  /// tasks; the solver toggles active tasks only at solve boundaries based on
  /// the previous constrained solve's binding score.
  bool enable_auto_task_layout = false;
  /// Switch from merged to split when binding_score exceeds this value.
  double auto_layout_binding_threshold_high = 0.30;
  /// Switch back from split to merged after cooldown_ticks consecutive scores
  /// below this value.
  double auto_layout_binding_threshold_low = 0.15;
  /// Hysteresis count for returning from split to merged.
  int auto_layout_cooldown_ticks = 20;
};

} // namespace embodik
