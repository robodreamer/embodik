/**
 * @file kinematics_solver.hpp
 * @brief High-level kinematics solver for EmbodiK
 *
 * Provides a simple, high-level API for solving IK problems.
 * Handles velocity integration, limits, and solver details internally.
 */

#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <functional>
#include <embodik/dual_arm_ects.hpp>
#include <embodik/pose_task_group.hpp>
#include <embodik/robot_model.hpp>
#include <embodik/sphere_broadphase.hpp>
#include <embodik/tasks.hpp>
#include <embodik/types.hpp>
#include <limits>
#include <memory>
#include <optional>
#include <pinocchio/spatial/se3.hpp>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace embodik {

enum class ContactType {
  kPointContact, // 3 rows (linear velocity only)
  kRigidContact, // 6 rows (full spatial velocity)
};

/**
 * @brief High-level kinematics solver
 *
 * Example usage:
 * ```cpp
 * KinematicsSolver solver(robot);
 * solver.set_dt(0.01);
 *
 * auto task = solver.add_frame_task("ee_task", "end_effector");
 * task->setTargetPose(target_pos, target_rot);
 * task->setPriority(0);
 * task->setWeight(10.0);
 *
 * solver.solve();  // Updates robot configuration
 * ```
 */
class KinematicsSolver {
public:
  /**
   * @brief Constructor
   * @param robot Robot model to control
   */
  explicit KinematicsSolver(std::shared_ptr<RobotModel> robot);

  /**
   * @brief Add a frame tracking task
   * @param name Unique task name
   * @param frame_name Frame to track
   * @param task_type Type of tracking (position/orientation/pose)
   * @return Shared pointer to the created task
   */
  std::shared_ptr<FrameTask>
  add_frame_task(const std::string &name, const std::string &frame_name,
                 TaskType task_type = TaskType::FRAME_POSE);

  /**
   * @brief Add a COM tracking task
   * @param name Unique task name
   * @return Shared pointer to the created task
   */
  std::shared_ptr<COMTask> add_com_task(const std::string &name);

  /**
   * @brief Add a posture regularization task
   * @param name Unique task name
   * @param controlled_joints Optional list of joint indices to control
   * @return Shared pointer to the created task
   */
  std::shared_ptr<PostureTask>
  add_posture_task(const std::string &name,
                   const std::vector<int> &controlled_joints = {});

  /**
   * @brief Add a lower-priority frame manipulability gradient task.
   * @param name Unique task name
   * @param frame_name Frame whose Jacobian conditioning should improve
   * @param frame_task_type Jacobian block (position/orientation/pose)
   * @return Shared pointer to the created task
   */
  std::shared_ptr<ManipulabilityTask>
  add_manipulability_task(
      const std::string &name, const std::string &frame_name,
      TaskType frame_task_type = TaskType::FRAME_POSITION);

  /**
   * @brief Add a smooth joint-limit avoidance objective.
   * @param name Unique task name
   * @param controlled_joint_indices Velocity-space indices; empty controls all
   * scalar joints
   * @return Shared pointer to the created task
   */
  std::shared_ptr<JointLimitAvoidanceTask> add_joint_limit_avoidance_task(
      const std::string &name,
      const std::vector<int> &controlled_joint_indices = {});

  /**
   * @brief Add a joint task
   * @param name Unique task name
   * @param joint_name Joint to control
   * @param target_value Target joint value
   * @return Shared pointer to the created task
   */
  std::shared_ptr<JointTask> add_joint_task(const std::string &name,
                                            const std::string &joint_name,
                                            double target_value = 0.0);

  /**
   * @brief Add a relative frame task (tracks T_a^{-1} * T_b)
   * @param name Unique task name
   * @param frame_a Reference frame name
   * @param frame_b Target frame name
   * @return Shared pointer to the created task
   */
  std::shared_ptr<RelativeFrameTask>
  add_relative_frame_task(const std::string &name,
                          const std::string &frame_a,
                          const std::string &frame_b);

  /**
   * @brief Add an absolute frame task (tracks weighted average of two frames)
   * @param name Unique task name
   * @param frame_a First frame name
   * @param frame_b Second frame name
   * @param alpha Coordination ratio [0,1] (0.5 = midpoint)
   * @return Shared pointer to the created task
   */
  std::shared_ptr<AbsoluteFrameTask>
  add_absolute_frame_task(const std::string &name,
                          const std::string &frame_a,
                          const std::string &frame_b,
                          double alpha = 0.5);

  /**
   * @brief Add a pose task group adapter.
   *
   * Split mode creates two regular FrameTasks:
   *   name + "__position" at base_priority, FRAME_POSITION
   *   name + "__orientation" at base_priority + rotation_priority_offset,
   *   FRAME_ORIENTATION
   *
   * Merged mode creates one regular FRAME_POSE task named name. The adapter
   * only owns task bookkeeping; solve_position_step() consumes the emitted
   * TaskTarget list through the existing multi-target path.
   */
  std::shared_ptr<PoseTaskGroup>
  add_pose_task_group(const std::string &name, const std::string &tcp_frame,
                      int base_priority = 0,
                      int rotation_priority_offset = 1,
                      bool merged_pose = false, bool auto_switch = false);

  /**
   * @brief Get a pose task group by name.
   */
  std::shared_ptr<PoseTaskGroup>
  pose_task_group(const std::string &name) const;

  /**
   * @brief Configure a relative pose inequality constraint between two frames
   *
   * Constrains each masked axis of the relative pose (T_a^{-1} * T_b) to stay
   * within the given bounds. Uses the relative Jacobian to formulate velocity
   * inequality rows in the QP, following the same pattern as the CoM constraint.
   *
   * @param frame_a Reference frame
   * @param frame_b Target frame
   * @param lower_bounds 6D lower bounds (pos xyz + ori xyz)
   * @param upper_bounds 6D upper bounds (pos xyz + ori xyz)
   * @param axis_mask 6D mask (1=constrained, 0=free). Empty = all constrained.
   */
  void configure_relative_pose_constraint(
      const std::string &frame_a,
      const std::string &frame_b,
      const Eigen::VectorXd &lower_bounds,
      const Eigen::VectorXd &upper_bounds,
      const Eigen::VectorXd &axis_mask = Eigen::VectorXd());

  /**
   * @brief Disable relative pose constraint
   */
  void clear_relative_pose_constraint();

  /**
   * @brief Remove a task by name
   * @param name Task name to remove
   */
  void remove_task(const std::string &name);

  /**
   * @brief Clear all tasks
   */
  void clear_tasks();

  /**
   * @brief Clear direct target velocities on all registered tasks.
   */
  void clear_all_target_velocities();

  /**
   * @brief Get task by name
   * @param name Task name
   * @return Task pointer or nullptr if not found
   */
  std::shared_ptr<Task> get_task(const std::string &name);

  /**
   * @brief Solve for joint velocities without integration
   * @param current_q Current joint configuration (optional, uses robot's
   * current if empty)
   * @param apply_limits If true, apply velocity and position-based velocity
   * limits
   * @param stall_recovery When true, enable automatic stall recovery for
   *        this call. See PositionStepOptions::stall_recovery for details.
   *        No-op if the stall handler is already enabled.
   * @return Velocity solver result with joint velocities and saturation info
   */
  VelocitySolverResult
  solve_velocity(const Eigen::VectorXd &current_q = Eigen::VectorXd(),
                 bool apply_limits = true,
                 bool stall_recovery = false);

  /**
   * @brief Enable/disable detailed timing breakdown fields in
   * VelocitySolverResult.
   *
   * When disabled (default), timing fields remain zero and we avoid extra
   * high_resolution_clock calls in the hot path.
   */
  void enable_timing_breakdown(bool enable) {
    timing_breakdown_enabled_ = enable;
  }
  bool timing_breakdown_enabled() const { return timing_breakdown_enabled_; }

  /**
   * @brief Enable/disable joint velocity limits
   * @param enable True to enable velocity limit constraints
   */
  void enable_velocity_limits(bool enable) { use_velocity_limits_ = enable; }

  /**
   * @brief Override the velocity limit for a single joint (nv index).
   *
   * Shrinks (or relaxes) that joint's velocity-box bound in the QP. Used to bias
   * how much a joint group contributes to a task: throttling arm joints makes the
   * solver recruit other DOFs (e.g. torso) to keep tracking the EE. Negative
   * values clamp to 0. Applied whether or not global velocity limits are enabled.
   */
  void set_joint_velocity_limit(int nv_idx, double limit) {
    joint_velocity_limit_overrides_[nv_idx] = limit < 0.0 ? 0.0 : limit;
  }

  /** @brief Clear all per-joint velocity-limit overrides (restore defaults). */
  void clear_joint_velocity_limit_overrides() {
    joint_velocity_limit_overrides_.clear();
  }

  /**
   * @brief Set a soft per-joint joint-space metric for the velocity solve.
   *
   * @param weights Per-joint cost weights (size nv). A higher weight makes that
   *   joint "more expensive", so it contributes less to the achieved task motion
   *   (weighted least-norm: min dq^T W dq). The EE task is still tracked and no
   *   joint is hard-excluded, so the feasibility override is preserved. All-ones
   *   (or empty) is a no-op. This is the torso-vs-arm contribution knob.
   */
  void set_joint_metric_weights(const Eigen::VectorXd &weights) {
    if (weights.size() == 0) {
      joint_metric_col_scale_.resize(0);
      return;
    }
    bool all_one = true;
    double log_sum = 0.0;
    Eigen::VectorXd w(weights.size());
    for (Eigen::Index i = 0; i < weights.size(); ++i) {
      w(i) = weights(i) > 1e-12 ? weights(i) : 1e-12;
      log_sum += std::log(w(i));
      if (std::abs(weights(i) - 1.0) > 1e-12) all_one = false;
    }
    if (all_one) {  // exact no-op vs. today's solve
      joint_metric_col_scale_.resize(0);
      return;
    }
    // Normalize to geometric mean 1 so only the RATIOS bias distribution: this
    // keeps the task-space gram J W^-1 J^T at a stable magnitude, so the solver's
    // damping does not eat task achievement under suppression.
    const double gm = std::exp(log_sum / static_cast<double>(w.size()));
    Eigen::VectorXd s(w.size());
    for (Eigen::Index i = 0; i < w.size(); ++i)
      s(i) = 1.0 / std::sqrt(w(i) / gm);
    joint_metric_col_scale_ = s;
  }

  /** @brief Clear the joint-space metric (restore unweighted solve). */
  void clear_joint_metric_weights() { joint_metric_col_scale_.resize(0); }

  /**
   * @brief Enable/disable joint position limits
   * @param enable True to enable position limit constraints
   */
  void enable_position_limits(bool enable) { use_position_limits_ = enable; }

  /**
   * @brief Enable/disable inter-tick acceleration limit constraints.
   *
   * When enabled, tightens velocity bounds based on the previous tick's
   * velocity and the configured acceleration limits, producing smoother
   * velocity profiles (lower jerk). Requires set_acceleration_limits().
   * @param enable True to enable acceleration constraints
   */
  void enable_acceleration_limits(bool enable) {
    acceleration_limits_enabled_ = enable;
    if (enable) {
      previous_dq_ = Eigen::VectorXd::Zero(robot_->nv());
    } else {
      previous_dq_ = Eigen::VectorXd();
    }
  }

  /**
   * @brief Set per-joint acceleration limits (rad/s^2).
   * @param limits Vector of size nv with max acceleration per joint
   */
  void set_acceleration_limits(const Eigen::VectorXd &limits) {
    if (limits.size() != robot_->nv()) {
      throw std::invalid_argument("acceleration limits must have size nv");
    }
    for (Eigen::Index i = 0; i < limits.size(); ++i) {
      if (!std::isfinite(limits[i]) || limits[i] < 0.0) {
        throw std::invalid_argument(
            "acceleration limits must be finite and non-negative");
      }
    }
    acceleration_limits_ = limits;
  }

  /**
   * @brief Set floating-base position bounds (for floating-base robots)
   * @param lower Lower bounds for base position (3D)
   * @param upper Upper bounds for base position (3D)
   */
  void set_base_position_bounds(const Eigen::Vector3d &lower,
                                const Eigen::Vector3d &upper);

  /**
   * @brief Set floating-base orientation bounds (for floating-base robots)
   * @param lower Lower bounds for base orientation (3D, in velocity space)
   * @param upper Upper bounds for base orientation (3D, in velocity space)
   */
  void set_base_orientation_bounds(const Eigen::Vector3d &lower,
                                   const Eigen::Vector3d &upper);

  /**
   * @brief Clear floating-base bounds (use unlimited bounds)
   */
  void clear_base_bounds();

  /**
   * @brief Set time step for integration
   * @param dt Time step in seconds
   */
  void set_dt(double dt) { dt_ = dt; }

  /**
   * @brief Get current time step
   * @return Time step in seconds
   */
  double get_dt() const { return dt_; }

  /**
   * @brief Set singular-value damping threshold for regularized pseudoinverse.
   *
   * This preserves historical behavior: set_tolerance() maps to
   * VelocitySolverConfig::regularization_config.epsilon.
   *
   * @param tolerance Regularization epsilon (default 0.1).
   */
  void set_tolerance(double tolerance) { solver_tolerance_ = tolerance; }

  /**
   * @brief Override the singular-value damping threshold for the regularized
   *        pseudoinverse, decoupling it from the solver tolerance.
   *
   * By default (when this has never been called), the regularization epsilon
   * mirrors @c solver_tolerance_ for backward compatibility.  Call this to
   * pin it to an independent value.
   *
   * Any Jacobian singular value below @p epsilon is treated as near-singular
   * and receives additional damping proportional to
   * @c damping * (1 - (sigma/epsilon)^2).  Lower values, e.g. 1e-6, preserve
   * sharper singular-value sensitivity for solver math tests and specialized
   * experiments.
   *
   * Call this when you need to decouple the regularization threshold from
   * the solver's default tolerance.
   *
   * @param epsilon Singular-value damping threshold.
   */
  void set_regularization_epsilon(double epsilon) { solver_tolerance_ = epsilon; }

  /**
   * @brief Set constraint violation deadband for bound checks.
   *
   * Maps to VelocitySolverConfig::epsilon and is independent from
   * regularization epsilon.
   */
  void set_constraint_tolerance(double epsilon) {
    constraint_tolerance_ = epsilon;
  }

  /**
   * @brief Set maximum iterations
   * @param max_iter Maximum solver iterations
   */
  void set_max_iterations(int max_iter) { max_iterations_ = max_iter; }

  /**
   * @brief Set singularity robust damping
   * @param damping Damping factor for pseudo-inverse
   */
  void set_damping(double damping) {
    damping_ = damping;
    runtime_config_.damping = damping;
  }

  /**
   * @brief Apply a bundled runtime configuration to this solver.
   *
   * Stamps cfg.damping onto the solver and stores position-step defaults for
   * make_position_step_options(). Existing setters and per-call options remain
   * supported and authoritative.
   */
  void configure_runtime(const SolverRuntimeConfig &cfg) {
    runtime_config_ = cfg;
    damping_ = cfg.damping;
    reset_adaptive_state();
    reset_auto_task_layout_state();
    reset_position_step_continuity_state();
  }

  /**
   * @brief Reset default-off stateful runtime adapters without changing config.
   */
  void reset_adaptive_state() {
    advisor_scale_current_ =
        sanitize_advisor_scale(runtime_config_.advisor_position_weight_scale);
    advisor_scale_ratio_sum_ = 0.0;
    advisor_scale_epoch_time_s_ = 0.0;
    advisor_scale_sample_count_ = 0;
  }

  /**
   * @brief Return the last-applied runtime configuration.
   */
  const SolverRuntimeConfig &runtime_config() const { return runtime_config_; }

  /**
   * @brief Build fresh PositionStepOptions from stored runtime defaults.
   */
  PositionStepOptions make_position_step_options() const {
    PositionStepOptions opts;
    opts.max_steps = runtime_config_.position_step_max_steps;
    opts.adaptive_dt = runtime_config_.adaptive_dt;
    opts.adaptive_dt_max_scale = runtime_config_.adaptive_dt_max_scale;
    opts.adaptive_dt_reference_distance =
        runtime_config_.adaptive_dt_reference_distance;
    return opts;
  }

  /**
   * @brief Set recovery gain for joint limit violations.
   *
   * Values are clamped to [0, 1]. Higher values recover faster.
   */
  void set_limit_recovery_gain(double gain) {
    limit_recovery_gain_ = std::clamp(gain, 0.0, 1.0);
  }
  /**
   * @brief Configure joint-limit recovery hysteresis thresholds (in radians/meters).
   *
   * A joint is treated as outside if margin < -enter_epsilon.  exit_epsilon is
   * retained for API symmetry and future stateful recovery policies.
   */
  void set_limit_recovery_hysteresis(double enter_epsilon,
                                     double exit_epsilon) {
    limit_recovery_enter_epsilon_ = std::max(0.0, enter_epsilon);
    limit_recovery_exit_epsilon_ =
        std::max(limit_recovery_enter_epsilon_, exit_epsilon);
  }

  /**
   * @brief Set release margin used to relax recovery forcing near boundaries.
   *
   * Small violations smaller than this value are treated as already recovered
   * for the recovery-forcing term, reducing chattering near limits.
   */
  void set_limit_exit_release_margin(double margin) {
    limit_exit_release_margin_ = std::max(0.0, margin);
  }

  /**
   * @brief Enable or disable saturation-exit velocity-box softening.
   *
   * When enabled, joints inside both limits get a minimum velocity headroom
   * (kMinBoundFraction) so the SNS solver can find partial solutions near
   * saturation. Disabled by default; requires more testing.
   */
  void enable_saturation_exit_behavior(bool enable) {
    enable_saturation_exit_behavior_ = enable;
  }
  bool saturation_exit_behavior_enabled() const {
    return enable_saturation_exit_behavior_;
  }

  /// @deprecated Recovery state machine has been removed. These methods are
  /// retained as no-ops for API backward compatibility.
  void set_solver_recovery_enabled(bool /*enable*/) {}
  bool solver_recovery_enabled() const { return false; }

  /**
   * @brief Enable verbose debugging for position IK iterations.
   * @param enable True to print/log per-iteration errors and store traces.
   */
  void enable_position_ik_debug(bool enable) { position_ik_debug_ = enable; }
  bool position_ik_debug() const { return position_ik_debug_; }

  /**
   * @brief Get robot model
   * @return Shared pointer to robot model
   */
  std::shared_ptr<RobotModel> robot() { return robot_; }

  /**
   * @brief Get all tasks
   * @return Vector of all tasks
   */
  const std::vector<std::shared_ptr<Task>> &tasks() const { return tasks_; }

  /**
   * @brief Configure collision avoidance constraint.
   * @param min_distance Minimum separation distance to enforce between
   * collision objects.
   * @param include_pairs Optional list of geometry/frame name pairs to consider
   * (empty = all pairs).
   * @param exclude_pairs Optional list of geometry/frame name pairs to ignore.
   * @param nearest_points_all_pairs If false, nearest points will be computed
   * only for the selected closest pair (constraint/debug) instead of for every
   *        evaluated pair.
   * @param max_constraints Nominal collision-row budget. The closest pairs get
   * their own Jacobian rows and velocity-damper bounds. Controllable pairs that
   * are penetrating or at their non-worsening recovery floor remain active even
   * when this exceeds the nominal budget. Defaults to 1. Values of 3-5 are
   * recommended for complex robots with multiple tight-clearance regions.
   */
  void configure_collision_constraint(
      double min_distance,
      const std::vector<std::pair<std::string, std::string>> &include_pairs =
          {},
      const std::vector<std::pair<std::string, std::string>> &exclude_pairs =
          {},
      bool nearest_points_all_pairs = true,
      int max_constraints = 1);

  /**
   * @brief Convenience helper for specifying a list of collision pairs to
   * monitor.
   * @param link_pairs Pairs of geometry/frame names to include.
   * @param min_distance Minimum separation distance.
   */
  void add_collision_constraint(
      const std::vector<std::pair<std::string, std::string>> &link_pairs,
      double min_distance = 0.05);

  /**
   * @brief Update only the min_distance of an already-configured collision
   *        constraint without rebuilding pair masks.
   *
   * Much cheaper than calling configure_collision_constraint() each tick.
   * No-op if no collision constraint has been configured yet.
   *
   * @param min_distance  New minimum separation distance (clamped to >= 0).
   * @return true if the distance was updated, false if no constraint exists.
   */
  bool set_collision_min_distance(double min_distance);

  /**
   * @brief Enable/disable proximity-gated collision-row activation.
   *
   * This toggle does not modify the configured activation multiplier/margin.
   * Use this when comparing gated and non-gated behavior under the same
   * threshold parameters.
   */
  void set_proximity_gated_collision_activation_enabled(bool enabled);

  /**
   * @brief Read whether proximity-gated collision-row activation is enabled.
   */
  bool get_proximity_gated_collision_activation_enabled() const;

  /**
   * @brief Set collision-row activation multiplier relative to min_distance.
   *
   * Effective activation margin is computed as:
   *   constraint_activation_margin = multiplier * min_distance
   *
   * A multiplier <= 0 disables proximity-gated row activation, preserving
   * legacy behavior (rows emitted for selected pairs regardless of distance).
   */
  void set_collision_constraint_activation_multiplier(double multiplier);

  /**
   * @brief Read the activation multiplier used for proximity-gated row emission.
   */
  double get_collision_constraint_activation_multiplier() const;

  /**
   * @brief Read current effective activation margin (meters).
   */
  double get_collision_constraint_activation_margin() const;

  /**
   * @brief Read the current collision min_distance.
   * @return Current min_distance, or -1 if no collision constraint is active.
   */
  double get_collision_min_distance() const;

  /**
   * @brief Disable collision avoidance constraints.
   *
   * Clears internal collision state as well (active pair indices, stuck
   * counters, per-pair last distances, and recovery homotopy margins) so a
   * later re-enable does not inherit stale solver-side data.
   */
  void clear_collision_constraint();

  /**
   * @brief Set a custom minimum distance for all collision pairs involving
   * geometries parented to link_a and link_b (matched by frame name substring).
   * Overrides the global min_distance for those pairs only.
   *
   * @param activate_when_clear  If true (default), the override is stored as
   *   "pending" and activates the first time the pair achieves the desired
   *   clearance during a solve — preventing immediate stall when called from
   *   a configuration already inside the threshold (latch-on semantics).
   *   If false, the override takes effect immediately (legacy behaviour).
   */
  void set_collision_pair_min_distance(const std::string &link_a,
                                       const std::string &link_b,
                                       double min_distance,
                                       bool activate_when_clear = true);

  /**
   * @brief Remove per-pair min_distance overrides for geometries involving
   * link_a and link_b. Affected pairs revert to the global min_distance.
   */
  void clear_collision_pair_min_distance(const std::string &link_a,
                                         const std::string &link_b);

  /**
   * @brief Return all active per-pair min_distance overrides as a list of
   * (canonical_pair_key, min_distance) pairs.
   */
  std::vector<std::pair<std::string, double>>
  get_collision_pair_min_distance_overrides() const;

  // ---- Tunable collision boundary behaviour --------------------------------
  /** Width (metres) of the no-braking zone above min_distance.  Default 3 mm.
   *  Set to 0 to eliminate the discontinuity that causes boundary oscillation. */
  void   set_collision_repulsion_deadband(double metres) { collision_repulsion_deadband_ = std::max(0.0, metres); }
  double get_collision_repulsion_deadband() const        { return collision_repulsion_deadband_; }

  /** Fraction of desired recovery velocity applied when inside min_distance.
   *  Default 0.2.  Lower = gentler push-back; higher = faster escape. */
  void   set_collision_recovery_scale(double scale) { collision_recovery_scale_ = std::clamp(scale, 0.01, 2.0); }
  double get_collision_recovery_scale() const       { return collision_recovery_scale_; }

  /** Non-worsening recovery floor. Default OFF (opt-in).
   *  When enabled, a pair first seen closer than min_distance uses the structural
   *  floor as its recovery and retention target. Pairs below the floor recover to
   *  it; pairs above the floor may move without being pinned to their initial
   *  clearance as long as they stay above it. This avoids demanding the full
   *  global clearance while preserving safe tangential freedom. Use a per-pair
   *  min-distance override for geometry that cannot reach the floor. Off by
   *  default because it changes recovery semantics for violated seeds. */
  void   set_non_worsening_collision_floor_enabled(bool enable) { non_worsening_collision_floor_enabled_ = enable; }
  bool   get_non_worsening_collision_floor_enabled() const      { return non_worsening_collision_floor_enabled_; }

  /** Minimum penetration-prevention clearance (metres) for pairs first observed
   *  below min_distance. Default 5 mm. */
  void   set_collision_structural_floor(double metres) { collision_structural_floor_ = std::max(0.0, metres); }
  double get_collision_structural_floor() const        { return collision_structural_floor_; }

  /** Maximum separation speed (m/s) for non-penetrating recovery.
   *  Default 0.15 m/s. */
  void   set_collision_max_separation_speed_nonpenetrating(double mps) { collision_max_sep_speed_nonpen_ = std::max(0.0, mps); }
  double get_collision_max_separation_speed_nonpenetrating() const     { return collision_max_sep_speed_nonpen_; }

  /**
   * @brief Enable/disable cached collision pair candidate evaluation.
   *
   * When enabled, collision distance queries are evaluated on a conservative
   * candidate subset between periodic full refreshes. This can reduce
   * computeDistance calls significantly in teleop loops where active collision
   * pairs change slowly.
   *
   * Safety behavior:
   * - Full refresh runs every @p full_refresh_interval steps.
   * - Candidate set is seeded from previous active/near-active pairs.
   * - Distances are never treated as configuration-invariant truths.
   *
   * @param enable Whether to enable candidate caching.
   * @param full_refresh_interval Number of solve steps between mandatory
   * full scans (>=1).
   * @param candidate_distance_margin Extra margin (m) above min_distance for
   * retaining near-active pairs as next-step candidates.
   * @param max_cached_candidates Cap on cached candidate pair indices.
   */
  void enable_collision_pair_cache(
      bool enable,
      int full_refresh_interval = 20,
      double candidate_distance_margin = 0.03,
      int max_cached_candidates = 128);

  /**
   * @brief Set optional per-step time budget (microseconds) for exact
   * collision distance refinement in cached-subset mode.
   *
   * When > 0, exact distance checks are prioritized by risk and stop once
   * the budget is exhausted; conservative behavior is preserved for
   * unprocessed pairs. A value <= 0 disables the budget.
   */
  void set_collision_refinement_time_budget_us(int budget_us);
  int get_collision_refinement_time_budget_us() const;

  /**
   * @brief Apply a high-level collision tuning preset.
   *
   * Presets map to conservative low-level cache/refinement parameters:
   * - kPrecise:
   *   Full exact checks (cache off, budget disabled).
   *   This minimizes approximation/culling effects and is intended for
   *   users who prioritize collision-distance fidelity over cycle time.
   * - kBalanced:
   *   Conservative cache cadence with budget disabled.
   *   Cache still reduces unnecessary pair checks in clear space, but without
   *   a time budget the solver does not early-stop exact refinement, reducing
   *   risk of missing borderline cases compared to speed mode.
   * - kSpeed:
   *   Aggressive cache cadence with bounded exact-refinement budget.
   *   This is optimized for high-rate teleop loops where bounded compute
   *   latency is the primary objective.
   */
  void set_collision_tuning_mode(CollisionTuningMode mode);
  CollisionTuningMode get_collision_tuning_mode() const;

  void enable_sphere_broadphase(bool enable);
  bool sphere_broadphase_enabled() const { return sphere_broadphase_enabled_; }

  // ========== Stall Handler ==========

  /**
   * @brief Enable the automatic stall handler.
   *
   * When enabled, after each velocity solve the handler updates stall state:
   * a **stall** is a non-success status with joint velocity norm below
   * `dq_stall_eps` (see `configure_stall_handler` defaults).
   *
   * After enough consecutive stalled steps, recovery **only** adjusts the
   * effective collision `min_distance`: ratchet down (or set an escape margin
   * under penetration) when self-collision actually binds the QP (collision
   * distance <= current min_distance), then gradually restore toward
   * `nominal_min_distance` when solves succeed again.
   *
   * Stalls caused by joint-limit clamping (where the collision QP row has
   * slack) do **not** trigger margin relaxation — this prevents the handler
   * from inadvertently opening a collision gap that lets the task drive the
   * arm into the body.
   *
   * Primary-task `MIN_ERROR` fallback is **not** toggled by the stall handler;
   * use `Task::setAllowMinErrorFallback` / position IK options separately if
   * desired.
   *
   * @param nominal_min_distance  The original user-intended collision
   *        min_distance. The handler relaxes below this during stalls
   *        and restores back to it afterward.
   */
  void enable_stall_handler(double nominal_min_distance);

  /// Disable the stall handler and restore nominal parameters.
  void disable_stall_handler();

  /// @return true if the stall handler is enabled.
  bool stall_handler_enabled() const;

  /**
   * @brief Configure stall handler tuning parameters.
   *
   * Only call after enable_stall_handler(). All parameters have sensible
   * defaults for interactive loops at 50–200 Hz.
   *
   * @param stall_threshold  Consecutive stalled steps to trigger (default 5)
   * @param restore_rate     Per-step restoration of min_distance as fraction of nominal (default 0.005)
   * @param floor_fraction   Minimum min_distance as fraction of nominal (default 0.3)
   */
  void configure_stall_handler(int stall_threshold = 5,
                                double restore_rate = 0.005,
                                double floor_fraction = 0.3);

  /// @return true if the stall handler has currently relaxed collision margin.
  bool stall_handler_is_relaxed() const;

  /// @return current effective collision min_distance (may be < nominal if relaxed).
  double stall_handler_current_min_distance() const;

  /// @return number of consecutive stall steps in the current streak.
  int stall_handler_consecutive_stall_steps() const;

  /// @return current stall threshold in consecutive near-zero failed solves.
  int stall_handler_threshold() const { return stall_config_.stall_threshold; }

  /// @return current per-step nominal-margin restoration rate.
  double stall_handler_restore_rate() const {
    return stall_config_.restore_rate;
  }

  /// @return current minimum relaxed margin as a fraction of nominal.
  double stall_handler_floor_fraction() const {
    return stall_config_.floor_fraction;
  }

  // ========== Elastic Band Joint Limit Expansion ==========

  /**
   * @brief Enable elastic band joint limit expansion.
   *
   * When enabled, the solver temporarily expands joint position limit margins
   * for joints that are saturated during limit-dominated stalls. This keeps
   * more DOFs active in the SNS solver, allowing task progress even when
   * multiple joints are near their limits.
   *
   * The expansion is elastic: it grows when the solver is stuck at joint
   * limits and decays exponentially when the solver is healthy.
   *
   * @param delta_max  Maximum expansion per joint in radians (default 0.05).
   */
  void enable_elastic_band(double delta_max = 0.05);

  /// Disable elastic band and reset expansion state.
  void disable_elastic_band();

  /// @return true if elastic band is enabled.
  bool elastic_band_enabled() const;

  /**
   * @brief Configure elastic band tuning parameters.
   *
   * Only call after enable_elastic_band().
   *
   * @param delta_max  Maximum expansion per joint (radians, default 0.05)
   * @param expand_rate  Expansion per stall trigger (radians/step, default 0.01)
   * @param decay_rate  Exponential decay per healthy step (0-1, default 0.2)
   * @param stall_threshold  Consecutive stalls before expansion (default 3)
   * @param expand_only_saturated  Only expand saturated joints (default true)
   */
  void configure_elastic_band(double delta_max = 0.05,
                               double expand_rate = 0.01,
                               double decay_rate = 0.2,
                               int stall_threshold = 3,
                               bool expand_only_saturated = true);

  /// @return maximum delta currently active across all joints.
  double elastic_band_max_delta() const;

  /// @return per-joint delta vector (size == nv).
  Eigen::VectorXd elastic_band_deltas() const;

  /// @return true if any joint has nonzero expansion.
  bool elastic_band_is_expanded() const;

  /**
   * @brief Configure a CoM support-polygon constraint (inequality).
   *
   * Keeps the 2D projection of the center of mass inside the given convex
   * polygon expressed in @p frame_name.  The constraint is enforced as a set
   * of half-plane velocity inequalities: A * J_com_xy * dq <= upper_bound.
   *
   * Velocity and acceleration limits are applied following the Spot Flex IK
   * pattern:
   *   upper = min(margin / dt, com_vel_max)
   *   if use_acceleration_limits:
   *     upper = min(upper, sqrt(2 * com_acc_max * margin))  // smooth saturation
   *
   * The sqrt term ensures that near the polygon boundary (small margin) the
   * allowed approach velocity is reduced so that the CoM can stop at the
   * boundary under max deceleration, preventing tipping overshoot.
   *
   * @param vertices_xy  Nx2 matrix of polygon vertices in the XY plane of
   *                     frame_name (Z column is ignored if Nx3 is passed).
   * @param margin       Fractional inward shrink in [0, 1]. Applied by moving
   *                     each vertex toward the centroid by margin * char_size,
   *                     where char_size is the mean distance from centroid to
   *                     vertices for consistent shrink behavior across polygon shapes.
   * @param frame_name   Frame in which @p vertices_xy are expressed.
   * @param com_vel_max  Maximum CoM velocity (m/s, default 0.4).
   * @param com_acc_max  Maximum CoM acceleration (m/s², default 0.1).
   * @param use_acceleration_limits  If true, also clamp by sqrt(2*a*margin).
   * @param proximity_fraction  Fraction of the polygon inradius used as the
   *   per-row activation distance.  A half-plane row is only added to the QP
   *   when the CoM slack for that row (b[i] - A[i]·com_xy) is less than
   *   ``proximity_fraction * inradius``.  The inradius is the minimum
   *   perpendicular distance from the polygon centroid to any edge and is
   *   computed automatically from the vertices — no external geometry needed.
   *   Set to 0 (default) to disable proximity filtering and always include
   *   every row (backward-compatible).  Mirrors the Spot Flex IK
   *   ``check_proximity_to_com_constraints`` pattern at per-row granularity.
   */
  void configure_com_constraint(
      const Eigen::MatrixXd &vertices_xy,
      double margin = 0.0,
      const std::string &frame_name = "world",
      double com_vel_max = 0.4,
      double com_acc_max = 0.1,
      bool use_acceleration_limits = true,
      double proximity_fraction = 0.0);

  /**
   * @brief Return the proximity threshold (metres) computed by the last call to
   *   configure_com_constraint(), or 0 if no constraint is configured.
   *
   * Equals ``proximity_fraction * inradius`` where the inradius is the minimum
   * perpendicular distance from the polygon centroid to any edge.
   */
  double get_com_proximity_threshold() const;

  /**
   * @brief Disable CoM support-polygon constraint.
   */
  void clear_com_constraint();

  /**
   * @brief Replace user-defined linear velocity constraints.
   *
   * Constraints are enforced as:
   *   lower_bounds <= C * dq <= upper_bounds
   *
   * @param C Constraint matrix (m x nv)
   * @param lower_bounds Lower bounds (m)
   * @param upper_bounds Upper bounds (m)
   */
  void set_linear_velocity_constraints(const Eigen::MatrixXd &C,
                                       const Eigen::VectorXd &lower_bounds,
                                       const Eigen::VectorXd &upper_bounds);

  /**
   * @brief Append user-defined linear velocity constraints.
   *
   * @param C Constraint matrix (m x nv)
   * @param lower_bounds Lower bounds (m)
   * @param upper_bounds Upper bounds (m)
   */
  void append_linear_velocity_constraints(const Eigen::MatrixXd &C,
                                          const Eigen::VectorXd &lower_bounds,
                                          const Eigen::VectorXd &upper_bounds);

  /**
   * @brief Clear all user-defined linear velocity constraints.
   */
  void clear_linear_velocity_constraints();

  /**
   * @brief Number of active user-defined linear velocity constraints.
   */
  int get_linear_velocity_constraint_rows() const;

  /**
   * @brief Add a contact frame for contact-root Jacobian projection.
   *
   * Added contact rows are stacked into J_c and used to compute
   * P_c = I - J_c^+ J_c, then all task Jacobians are projected by P_c before
   * solving.
   *
   * @param frame_name Frame to treat as active contact.
   * @param type Contact constraint type: point (3D) or rigid (6D).
   */
  void add_contact_frame(const std::string &frame_name,
                         ContactType type = ContactType::kRigidContact);

  /**
   * @brief Convenience helper: clear and set multiple contact frames.
   */
  void configure_contact_frames(
      const std::vector<std::string> &frame_names,
      ContactType type = ContactType::kRigidContact);

  /**
   * @brief Clear all contact-root projection frames.
   */
  void clear_contact_frames();

  /**
   * @brief Returns true when contact-root projection is active.
   */
  bool has_contact_frames() const { return !contact_frames_.empty(); }

  /**
   * @brief Add a tight 6D frame pose constraint around a target pose.
   *
   * Enforces epsilon-box bounds in task-space around @p target_pose:
   *   -position_epsilon <= [dx,dy,dz] <= position_epsilon
   *   -orientation_epsilon <= [rx,ry,rz] <= orientation_epsilon
   *
   * @param frame_name Frame to constrain.
   * @param target_pose Target 4x4 homogeneous pose in world frame.
   * @param position_epsilon Translation epsilon (meters).
   * @param orientation_epsilon Orientation epsilon (radians).
   * @param axis_mask Optional 6D axis mask (empty = all ones).
   */
  void add_tight_frame_pose_constraint(
      const std::string &frame_name, const Eigen::Matrix4d &target_pose,
      double position_epsilon = 1e-5, double orientation_epsilon = 1e-4,
      const Eigen::VectorXd &axis_mask = Eigen::VectorXd());

  /**
   * @brief Clear all tight 6D frame pose constraints.
   */
  void clear_tight_frame_pose_constraints();

  /**
   * @brief Add a tight 3D point constraint on a frame translation.
   *
   * Enforces:
   *   -position_epsilon <= [dx,dy,dz] <= position_epsilon
   * for masked translational axes.
   *
   * @param frame_name Frame to constrain.
   * @param target_point Target 3D point in world frame.
   * @param position_epsilon Translation epsilon (meters).
   * @param axis_mask Optional 3D axis mask (empty = all ones).
   */
  void add_tight_point_constraint(const std::string &frame_name,
                                  const Eigen::Vector3d &target_point,
                                  double position_epsilon = 1e-5,
                                  const Eigen::Vector3d &axis_mask =
                                      Eigen::Vector3d::Ones());

  /**
   * @brief Clear all tight point constraints.
   */
  void clear_tight_point_constraints();

  struct CollisionDebugInfo {
    std::string object_a;
    std::string object_b;
    Eigen::Vector3d point_a_world = Eigen::Vector3d::Zero();
    Eigen::Vector3d point_b_world = Eigen::Vector3d::Zero();
    double distance = std::numeric_limits<double>::infinity();
  };

  /**
   * @brief Retrieve debug information about the last evaluated collision pair.
   * When max_constraints > 1, this returns info for the closest active pair.
   */
  std::optional<CollisionDebugInfo> get_last_collision_debug() const {
    return last_collision_debug_;
  }

  /**
   * @brief Retrieve debug information for all active collision constraint pairs.
   * Returns one entry per active constraint row. Safety-critical rows can exceed
   * the nominal max_constraints budget. Empty when no collision constraint is
   * configured or no solve has been performed.
   */
  std::vector<CollisionDebugInfo> get_last_collision_debug_list() const {
    return last_collision_debug_list_;
  }

  /**
   * @brief Evaluate collisions at the provided configuration and return debug
   * info.
   *
   * This runs collision distance computation (respecting active pair masks /
   * include-exclude filtering) and returns the closest-pair debug info.
   * Intended for validating final solutions (e.g., reachability sweeps) where
   * "SUCCESS" from IK should still be rejected if it ends inside the collision
   * threshold.
   *
   * @param current_q Optional configuration to evaluate (empty => current robot
   * configuration).
   */
  std::optional<CollisionDebugInfo> evaluate_collision_debug(
      const Eigen::VectorXd &current_q = Eigen::VectorXd());

  /**
   * @brief Evaluate the scalar collision distance used by post-step safety
   * checks.
   *
   * Prefers cached / targeted collision data from the most recent constraint
   * solve before falling back to a global distance scan.
   */
  std::optional<double> evaluate_post_step_collision_distance(
      const Eigen::VectorXd &q);

  /**
   * @brief Check per-pair override violations at q.
   *
   * For each pair that has an active per-pair min_distance override, computes
   * the exact signed distance and subtracts the override threshold.  Returns
   * the worst (most negative) margin across all such pairs, or nullopt if there
   * are no active overrides.  A negative return means at least one custom pair
   * is inside its override threshold even if the global min_distance is not
   * violated — this is the "gap" the global post-step check misses.
   */
  std::optional<double> evaluate_per_pair_override_violations(
      const Eigen::VectorXd &q);

  /**
   * @brief Evaluate minimum collision distance at the given configuration.
   * @param current_q Configuration to evaluate (empty = use current).
   * @return Minimum distance, or nullopt if no collision geometry.
   */
  std::optional<double> evaluate_min_collision_distance(
      const Eigen::VectorXd &current_q = Eigen::VectorXd());

  /**
   * @brief Retrieve the list of currently active collision pairs (after
   * include/exclude filtering).
   */
  std::vector<std::pair<std::string, std::string>>
  get_active_collision_pairs() const;

  /**
   * @brief Calculate velocity box constraints based on position, velocity, and
   * acceleration limits.
   *
   * Velocity limits are computed as:
   * min(position_margin/dt, velocity_limit, sqrt(2*accel*margin))
   *
   * @param position_margin_lower Distance from current position to lower limit
   * @param position_margin_upper Distance from current position to upper limit
   * @param velocity_limit Maximum allowed velocity
   * @param acceleration_limit Maximum allowed acceleration
   * @param dt Time step
   * @param min_velocity_headroom Optional minimum |velocity| headroom to
   * guarantee away from nearby limits. Pass negative to disable and use solver
   * default behavior.
   * @param headroom_activation_margin Margin threshold above which headroom can
   * be injected (prevents pushing toward an already-active limit).
   * @return Pair of (lower_velocity_limit, upper_velocity_limit)
   */
  std::pair<double, double> calculate_velocity_box_constraint(
      double position_margin_lower, double position_margin_upper,
      double velocity_limit, double acceleration_limit, double dt,
      double min_velocity_headroom = -1.0,
      double headroom_activation_margin = 0.01) const;

private:
  struct ContactFrameConfig {
    std::string frame_name;
    ContactType type = ContactType::kRigidContact;
  };

  std::shared_ptr<RobotModel> robot_;
  std::vector<std::shared_ptr<Task>> tasks_;
  std::unordered_map<std::string, std::shared_ptr<Task>> task_map_;
  std::unordered_map<std::string, std::shared_ptr<PoseTaskGroup>>
      pose_task_groups_;
  std::vector<ContactFrameConfig> contact_frames_;

  // Solver parameters
  double dt_ = 0.01;
  // Singular-value damping threshold for regularized pseudoinverse.
  double solver_tolerance_ = 0.1;
  // Constraint violation deadband + COD pseudoinverse relative threshold.
  double constraint_tolerance_ = 1e-6;
  double tight_tolerance_ = 1e-10;
  int max_iterations_ = 20;
  double damping_ = 0.1;
  SolverRuntimeConfig runtime_config_{};
  static double sanitize_advisor_scale(double scale) {
    return (std::isfinite(scale) && scale >= 0.0) ? scale : 1.0;
  }
  void reset_auto_task_layout_state();
  void select_auto_task_layout();
  VelocitySolverResult retry_auto_task_layout_as_split_if_needed(
      const Eigen::VectorXd &q, VelocitySolverResult result,
      const std::vector<int> &velocity_lock_indices,
      const std::optional<TorsoPoseConstraintOptions> &torso_constraint,
      const std::optional<double> &step_validation_dt);
  void update_auto_task_layout_feedback(const VelocitySolverResult &result);
  TaskLayout current_auto_task_layout_ = TaskLayout::kMerged;
  int auto_layout_below_low_count_ = 0;
  bool auto_layout_has_feedback_ = false;
  double auto_layout_binding_score_ = 0.0;
  double advisor_scale_current_ = 1.0;
  double advisor_scale_ratio_sum_ = 0.0;
  double advisor_scale_epoch_time_s_ = 0.0;
  int advisor_scale_sample_count_ = 0;
  double norm_threshold_ = 1e10;
  int max_zero_scale_iterations_ = 2;
  bool position_ik_debug_ = false;
  double limit_recovery_gain_ = 0.5;
  double limit_recovery_enter_epsilon_ = 1e-4;
  double limit_recovery_exit_epsilon_ = 1e-4;
  double limit_exit_release_margin_ = 0.0;
  /// When false (default), skip velocity-box softening near limits (saturation
  /// exit behavior). Enable for testing; behavior may change in future.
  bool enable_saturation_exit_behavior_ = false;

  // Constraint options
  bool use_velocity_limits_ = true;
  bool use_position_limits_ = true;
  // Per-joint velocity-limit overrides (nv index -> max |v|); override lever.
  std::unordered_map<int, double> joint_velocity_limit_overrides_;
  // Soft joint-space metric as a column scale (1/sqrt(w), size nv); empty = off.
  Eigen::VectorXd joint_metric_col_scale_;

  /// Sentinel in ``velocity_to_config_index_cache_`` for unmapped velocity indices.
  static constexpr int kVelocityToConfigUnmapped = -1;

  // Debug/perf instrumentation (off by default)
  bool timing_breakdown_enabled_ = false;

  // Inter-tick acceleration constraint cascading
  bool acceleration_limits_enabled_ = false;
  Eigen::VectorXd acceleration_limits_;
  Eigen::VectorXd previous_dq_;

  // Floating-base bounds (optional)
  std::optional<Eigen::Vector3d> base_position_lower_;
  std::optional<Eigen::Vector3d> base_position_upper_;
  std::optional<Eigen::Vector3d> base_orientation_lower_;
  std::optional<Eigen::Vector3d> base_orientation_upper_;

  /// Set by solve_position_step before each inner solve_velocity(); cleared at
  /// end of solve_velocity(). Enforces v_i = 0 in the QP for listed nv-indices.
  std::vector<int> pending_velocity_lock_indices_;
  /// Step-scoped frozen (locked/excluded) nv-indices, set for the whole duration
  /// of solve_position_step (not just the inner solve_velocity()). Lets the
  /// post-solve collision computations (stall escape, post-step validation) see
  /// the same frozen set the velocity solve used, so the collision debug list
  /// and the QP constraint slots agree on which pairs are controllable.
  std::vector<int> position_step_locked_indices_;
  /// Frozen nv-index set the collision code should honor: the inner velocity
  /// solve's pending set when active, otherwise the step-scoped set.
  const std::vector<int> &active_collision_lock_indices() const {
    return pending_velocity_lock_indices_.empty() ? position_step_locked_indices_
                                                   : pending_velocity_lock_indices_;
  }
  /// Optional torso constraint rows injected by solve_position_step into the
  /// next solve_velocity() call; cleared at end of solve_velocity().
  std::optional<TorsoPoseConstraintOptions> pending_step_torso_constraint_;
  /// One-shot hint set by solve_position_step when RobotModel already holds
  /// the exact q passed into the next solve_velocity() call.
  bool pending_reuse_current_kinematics_ = false;
  /// One-shot integration dt used by solve_position_step so solve_velocity()
  /// can validate fallback candidates against the actual accepted step length.
  std::optional<double> pending_step_validation_dt_;
  /// Guards against recursive MIN_ERROR step retry in solve_position_step.
  bool suppress_min_error_step_retry_ = false;
  /// Guards against recursive preferred-lock candidate retry in solve_position_step.
  bool suppress_preferred_lock_step_retry_ = false;
  struct PositionStepTargetSignature {
    std::vector<std::string> task_names;
    std::vector<Eigen::Matrix4d> target_poses;
    std::vector<Eigen::Matrix4d> reference_target_poses;
    std::vector<double> gains;
    std::int64_t command_revision = -1;
  };
  std::optional<PositionStepTargetSignature> last_position_step_target_signature_;
  std::vector<double> position_step_collision_command_floor_distances_;
  int position_step_call_depth_ = 0;
  std::optional<double> position_step_merit_window_anchor_;
  double position_step_merit_window_motion_ = 0.0;
  int position_step_merit_window_samples_ = 0;
  std::optional<Eigen::VectorXd> position_step_merit_window_last_delta_;
  int position_step_merit_window_direction_reversals_ = 0;
  std::optional<std::vector<double>> position_step_merit_window_last_merits_;
  int position_step_merit_window_error_increases_ = 0;
  bool position_step_stationary_guard_active_ = false;
  bool position_step_stationary_guard_can_reopen_ = true;
  bool update_position_step_target_signature(
      PositionStepTargetSignature signature);
  void capture_position_step_collision_command_floor(
      const Eigen::VectorXd &current_q);
  Eigen::Matrix4d canonicalize_position_step_signature_pose(
      const std::string &task_name, const Eigen::Matrix4d &target_pose,
      const std::optional<pinocchio::SE3> &reference_pose) const;
  void reset_position_step_continuity_state();
  void reset_position_step_merit_window();
  bool should_hold_position_step_for_continuity(
      const PositionIKResult &result, const Eigen::VectorXd &current_q,
      double initial_merit, double final_merit,
      const std::vector<double> &initial_target_merits,
      const std::vector<double> &final_target_merits,
      double configuration_step_norm, bool owns_position_step_continuity,
      bool collision_violated);
  std::optional<Eigen::MatrixXd> warm_start_selector_cache_;
  int warm_start_constraint_rows_ = -1;

  // Reused solve_velocity scratch containers. Matrix/vector entries are still
  // resized per solve, but preserving container capacity avoids repeated heap
  // churn in high-rate interactive IK loops.
  std::vector<Eigen::VectorXd> scratch_goals_;
  std::vector<Eigen::MatrixXd> scratch_jacobians_;
  std::vector<ObjectiveSolveConfig> scratch_objective_configs_;
  std::vector<std::shared_ptr<Task>> scratch_objective_tasks_;
  std::vector<std::shared_ptr<Task>> scratch_group_tasks_;
  std::unordered_set<int> scratch_excluded_union_;

  /// Cached velocity-index → configuration-index map for the current robot
  /// (rebuilt when the model pointer or ``nv`` changes).
  std::vector<int> velocity_to_config_index_cache_;
  const RobotModel *velocity_to_config_cache_robot_ = nullptr;
  int velocity_to_config_cache_nv_ = -1;

  void apply_position_step_primary_task_options(const PositionStepOptions &options,
                                                Task *task);

  std::optional<PositionIKResult> attempt_min_error_position_step_retry(
      const Eigen::VectorXd &entry_q, const PositionStepOptions &options,
      double step_dt, bool have_vel_result,
      const VelocitySolverResult &last_vel_result,
      const Eigen::VectorXd &q_after_primary,
      const std::function<bool(std::vector<std::pair<Task *, TaskSolveMode>> *)>
          &flip_primary_tasks,
      const std::function<PositionIKResult()> &rerun_step);

  PositionIKResult solve_position_step_with_preferred_lock(
      const Eigen::VectorXd &current_q, const Eigen::Matrix4d &target_pose,
      const std::string &frame_task_name, const PositionStepOptions &options);

  PositionIKResult solve_position_step_with_preferred_lock(
      const Eigen::VectorXd &current_q, const std::vector<TaskTarget> &targets,
      const PositionStepOptions &options);

  // Sort tasks by priority
  void sort_tasks_by_priority();

  /// Rebuild ``velocity_to_config_index_cache_`` if needed; return reference.
  const std::vector<int> &velocity_to_config_index_cache();

  /// Zero Jacobian entries that command motion into nearby joint limits.
  void clamp_jacobians_near_joint_limits(
      std::vector<Eigen::MatrixXd> &jacobians,
      const std::vector<Eigen::VectorXd> &goals,
      const std::vector<ObjectiveSolveConfig> &objective_configs,
      const std::vector<int> &velocity_to_config_index) const;

  struct CollisionConstraintConfig {
    bool enabled = false;
    double min_distance = 0.05;
    double upper_distance = 10.0;
    double tolerance = 1e-4;
    // Defaults mirror the BALANCED tuning preset (see set_collision_tuning_mode):
    // proximity-gated activation on with a conservative 5x activation band. This
    // only takes effect once configure_collision_constraint() enables a
    // constraint; an unconfigured solver still performs no collision checks.
    bool constraint_activation_enabled = true;
    // Optional proximity gate for emitting collision QP rows.
    // <= 0 disables gating and preserves legacy behavior.
    double constraint_activation_margin = 0.0;
    // Auto-tuning factor applied to min_distance:
    //   margin = multiplier * min_distance
    // <= 0 keeps gating disabled.
    double constraint_activation_multiplier = 5.0;
    bool nearest_points_all_pairs = true;
    std::unordered_set<std::string> include_pairs;
    std::unordered_set<std::string> exclude_pairs;
    // Maximum number of simultaneous QP constraint rows (one per pair).
    int max_constraints = 1;
  };

  struct CollisionConstraintResult {
    Eigen::MatrixXd jacobian;
    Eigen::VectorXd lower_bounds;
    Eigen::VectorXd upper_bounds;
    double distance = std::numeric_limits<double>::infinity();
    std::string object_a;
    std::string object_b;
    Eigen::Vector3d point_a_world = Eigen::Vector3d::Zero();
    Eigen::Vector3d point_b_world = Eigen::Vector3d::Zero();
  };

  // ---- Stall handler ----
  struct StallHandlerConfig {
    bool enabled = false;
    /// Consecutive stall steps before triggering margin relaxation.
    int stall_threshold = 5;
    /// Joint-velocity norm below which a step counts as "no motion".
    double dq_stall_eps = 1e-5;
    /// Fraction of nominal margin to restore per healthy step.
    double restore_rate = 0.005;
    /// Minimum collision margin as a fraction of nominal (hard floor).
    double floor_fraction = 0.3;
    /// Distance band kept between collision margin and actual collision
    /// distance.  Used both for bottleneck detection and as headroom
    /// during margin restoration to avoid re-creating infeasibility.
    double collision_proximity_band = 0.02;
    /// Fraction of nominal margin dropped each time the stall threshold fires.
    double relax_drop_fraction = 0.10;
  };

  struct StallHandlerState {
    double nominal_min_distance = 0.0;
    double current_min_distance = 0.0;
    int consecutive_stall_steps = 0;
    // Cumulative stats
    int total_stall_steps = 0;
    int total_relaxation_steps = 0;
  };

  StallHandlerConfig stall_config_;
  StallHandlerState stall_state_;
  bool stall_user_configured_ = false;

  void stall_handler_update(VelocitySolverResult &result);

  // ---- Elastic band joint limit expansion ----
  struct ElasticBandConfig {
    bool enabled = false;
    /// Maximum expansion per joint (radians).
    double delta_max = 0.05;
    /// Expansion rate per stall trigger (radians/step).
    double expand_rate = 0.01;
    /// Exponential decay rate per healthy step (dimensionless, 0-1).
    double decay_rate = 0.2;
    /// Consecutive stall steps before expansion starts.
    int stall_threshold = 3;
    /// Joint-velocity norm below which a step counts as "no motion".
    double dq_stall_eps = 1e-5;
    /// Whether to expand only saturated joints or all joints.
    bool expand_only_saturated = true;
  };

  struct ElasticBandState {
    /// Per-joint expansion amounts (size == nv, initialized to 0).
    Eigen::VectorXd delta;
    int consecutive_stall_steps = 0;
    int total_expansion_steps = 0;
  };

  ElasticBandConfig elastic_band_config_;
  ElasticBandState elastic_band_state_;

  void elastic_band_update(const VelocitySolverResult &result);

  // ---- CoM support-polygon constraint ----
  struct ComConstraintConfig {
    bool enabled = false;
    Eigen::MatrixXd vertices_xy;        // Nx2 in frame_name coordinates
    double margin = 0.0;
    std::string frame_name = "world";
    double com_vel_max = 0.4;
    double com_acc_max = 0.1;
    bool use_acceleration_limits = true;
    // Per-row activation distance: only include half-plane row i in the QP
    // when its slack (b[i] - A[i]*com_xy) < proximity_threshold.
    // +inf → always include all rows (proximity filtering disabled).
    // Auto-computed as proximity_fraction * inradius inside configure_com_constraint().
    double proximity_threshold = std::numeric_limits<double>::infinity();
    // Precomputed half-plane representation: A * x <= b (2D, frame_name)
    Eigen::MatrixXd A;
    Eigen::VectorXd b;
  };

  struct ComConstraintResult {
    Eigen::MatrixXd jacobian;      // (#half-planes x nv)
    Eigen::VectorXd lower_bounds;
    Eigen::VectorXd upper_bounds;
    Eigen::ArrayXi violated_rows;  // 1 where row slack is outside (< -eps)
  };

  std::optional<ComConstraintConfig> com_constraint_;
  std::optional<ComConstraintResult> compute_com_constraint();

  // ---- Relative pose constraint ----
  struct RelativePoseConstraintConfig {
    bool enabled = false;
    std::string frame_a;
    std::string frame_b;
    Eigen::VectorXd lower_bounds;  // 6D
    Eigen::VectorXd upper_bounds;  // 6D
    Eigen::VectorXd axis_mask;     // 6D: 1=constrained, 0=free
  };

  struct RelativePoseConstraintResult {
    Eigen::MatrixXd jacobian;      // (num_active_axes x nv)
    Eigen::VectorXd lower_bounds;
    Eigen::VectorXd upper_bounds;
    Eigen::ArrayXi violated_rows;  // 1 where row is outside lower/upper bounds
  };

  std::optional<RelativePoseConstraintConfig> relative_pose_constraint_;
  std::optional<RelativePoseConstraintResult> compute_relative_pose_constraint();

  struct LinearVelocityConstraintConfig {
    bool enabled = false;
    Eigen::MatrixXd C;
    Eigen::VectorXd lower_bounds;
    Eigen::VectorXd upper_bounds;
  };

  struct LinearVelocityConstraintResult {
    Eigen::MatrixXd jacobian;
    Eigen::VectorXd lower_bounds;
    Eigen::VectorXd upper_bounds;
    Eigen::ArrayXi violated_rows;
  };

  struct TightFramePoseConstraintConfig {
    std::string frame_name;
    pinocchio::SE3 target_pose = pinocchio::SE3::Identity();
    double position_epsilon = 1e-5;
    double orientation_epsilon = 1e-4;
    Eigen::VectorXd axis_mask = Eigen::VectorXd::Ones(6);
  };

  struct TightPointConstraintConfig {
    std::string frame_name;
    Eigen::Vector3d target_point = Eigen::Vector3d::Zero();
    double position_epsilon = 1e-5;
    Eigen::Vector3d axis_mask = Eigen::Vector3d::Ones();
  };

  std::optional<LinearVelocityConstraintConfig> linear_velocity_constraints_;
  std::vector<TightFramePoseConstraintConfig> tight_frame_pose_constraints_;
  std::vector<TightPointConstraintConfig> tight_point_constraints_;
  std::optional<LinearVelocityConstraintResult>
  compute_linear_velocity_constraints();
  std::optional<LinearVelocityConstraintResult>
  compute_tight_frame_pose_constraints();
  std::optional<LinearVelocityConstraintResult>
  compute_tight_point_constraints();
  Eigen::MatrixXd compute_contact_projector() const;

  std::optional<CollisionConstraintConfig> collision_constraint_;
  // Per-geometry-pair min_distance overrides. Key is canonical_pair_key(geom_a, geom_b).
  // Set via set_collision_pair_min_distance(link_a, link_b, distance) which resolves
  // link names to geometry names at call time. Survives configure_collision_constraint().
  // Active per-pair overrides: applied immediately every solve tick.
  std::unordered_map<std::string, double> per_pair_min_distance_overrides_;
  // Tunable collision boundary behaviour (default values mirror the file-scope
  // constexpr constants; exposed via Python for runtime sweep / autoresearch).
  // Default changed from 3e-3 to 0: the 3mm deadband created a 0.145 m/s
  // step-jump in lb at dist=min_dist+3mm, causing boundary-bounce oscillation.
  // With 0, lb is continuous everywhere → smooth deceleration at boundary.
  double collision_repulsion_deadband_           = 0.0;   // m above min_dist: lb=0 zone
  double collision_recovery_scale_               = 0.2;   // fraction of desired recovery vel
  double collision_max_sep_speed_nonpen_         = 0.15;  // m/s cap for non-penetrating recovery
  // Pending (deferred) overrides: promoted to active the first time the pair
  // achieves the desired clearance (latch-on). Set via activate_when_clear=true.
  std::unordered_map<std::string, double> per_pair_deferred_overrides_;
  // Non-worsening recovery floor: per-pair ratcheting clearance (keyed by
  // canonical pair key) and its enable flag. Cleared on (re)configure / clear.
  bool non_worsening_collision_floor_enabled_ = false;  // opt-in (changes recovery semantics)
  double collision_structural_floor_ = 0.005;  // 5 mm penetration-prevention floor
  std::unordered_map<std::string, double> collision_pair_distance_floor_;
  // Frozen (locked/excluded) velocity DOFs used when the lazy-reuse cache was
  // built; lazy reuse is invalidated when the current frozen set differs, since
  // it changes which collision pairs are controllable / selectable.
  std::vector<int> collision_cache_frozen_indices_;
  std::optional<CollisionDebugInfo> last_collision_debug_;
  // All active constraint pairs, including safety-critical rows beyond the
  // nominal max_constraints budget, populated after each solve.
  std::vector<CollisionDebugInfo> last_collision_debug_list_;
  // Cached allow-mask aligned with Pinocchio's collisionPairs indices.
  std::vector<std::uint8_t> collision_allowed_pair_mask_;
  // Active constraint pair indices from the previous solve step (for hysteresis).
  std::vector<std::size_t> last_collision_constraint_pair_indices_;
  // Cached candidate indices used for conservative subset evaluation.
  std::vector<std::size_t> collision_cached_candidate_pair_indices_;
  // Per-pair cached state for conservative lower-bound gating.
  std::vector<std::uint8_t> collision_pair_bound_valid_;
  std::vector<double> collision_pair_last_signed_distance_;
  std::vector<double> collision_pair_last_rel_translation_norm_;
  std::vector<std::array<double, 9>> collision_pair_last_rel_rotation_;
  // Collision-tuning member defaults mirror the BALANCED preset
  // (see set_collision_tuning_mode); keep them in sync if BALANCED changes.
  bool collision_pair_cache_enabled_ = true;
  int collision_pair_cache_refresh_interval_ = 5;
  double collision_pair_cache_distance_margin_ = 0.05;
  int collision_pair_cache_max_candidates_ = 256;
  int collision_refinement_time_budget_us_ = 0;
  bool collision_pair_cache_has_full_scan_ = false;
  int collision_pair_cache_steps_since_refresh_ = 0;
  // Instrumentation from the latest compute_collision_constraint() call.
  std::uint64_t last_collision_pairs_considered_ = 0;
  std::uint64_t last_collision_exact_distance_queries_ = 0;
  std::uint64_t last_collision_bound_culled_pairs_ = 0;
  bool last_collision_budget_exhausted_ = false;
  // Post-step rejection fast path: minimum signed distance from the most
  // recent compute_collision_constraint() call and whether it was a full scan.
  double last_constraint_min_distance_ = std::numeric_limits<double>::infinity();
  // Minimum signed clearance relative to the effective per-pair recovery
  // target represented by the active collision rows. Unlike the raw distance,
  // this honors structural/non-worsening floors and per-pair overrides.
  double last_constraint_min_recovery_margin_ =
      std::numeric_limits<double>::infinity();
  bool last_constraint_was_full_scan_ = false;
  // Cached constraint result for lazy reuse when configuration change is small.
  std::optional<CollisionConstraintResult> last_collision_constraint_result_;
  Eigen::VectorXd last_collision_constraint_q_;
  CollisionTuningMode collision_tuning_mode_ = CollisionTuningMode::kBalanced;
  SphereBroadphase sphere_broadphase_;
  bool sphere_broadphase_enabled_ = true;
  std::uint64_t last_collision_sphere_culled_pairs_ = 0;
  // Track solver stagnation near collision boundary for stronger recovery (per-pair).
  double last_solution_dq_norm_ = 0.0;
  std::unordered_map<std::size_t, int> collision_stuck_counters_;
  std::unordered_map<std::size_t, double> collision_stuck_last_distances_;
  std::string canonical_pair_key(const std::string &a,
                                 const std::string &b) const;
  bool collision_pair_allowed(const std::string &a, const std::string &b) const;
  std::optional<double> evaluate_min_collision_distance_targeted(
      const Eigen::VectorXd &current_q,
      const std::vector<std::size_t> &pair_indices);
  std::vector<std::size_t> get_post_step_rejection_pair_indices() const;
  std::vector<double> evaluate_post_step_collision_recovery_margins(
      const Eigen::VectorXd &current_q);
  std::optional<CollisionConstraintResult> compute_collision_constraint();

  struct PositionStepMutableStateSnapshot {
    struct TaskState {
      Task *task = nullptr;
      TaskSolveMode solve_mode = TaskSolveMode::kScale;
      bool allow_min_error_fallback = false;
      TaskSolveMode last_effective_mode = TaskSolveMode::kScale;
      bool used_min_error_fallback = false;
    };
    Eigen::VectorXd robot_q;
    std::vector<TaskState> task_states;
    TaskLayout current_auto_task_layout = TaskLayout::kMerged;
    int auto_layout_below_low_count = 0;
    bool auto_layout_has_feedback = false;
    double auto_layout_binding_score = 0.0;
    double advisor_scale_current = 1.0;
    double advisor_scale_ratio_sum = 0.0;
    double advisor_scale_epoch_time_s = 0.0;
    int advisor_scale_sample_count = 0;
    Eigen::VectorXd previous_dq;
    std::optional<double> position_step_merit_window_anchor;
    double position_step_merit_window_motion = 0.0;
    int position_step_merit_window_samples = 0;
    std::optional<Eigen::VectorXd> position_step_merit_window_last_delta;
    int position_step_merit_window_direction_reversals = 0;
    std::optional<std::vector<double>> position_step_merit_window_last_merits;
    int position_step_merit_window_error_increases = 0;
    bool position_step_stationary_guard_active = false;
    bool position_step_stationary_guard_can_reopen = true;
    StallHandlerConfig stall_config;
    StallHandlerState stall_state;
    bool stall_user_configured = false;
    ElasticBandConfig elastic_band_config;
    ElasticBandState elastic_band_state;
    std::optional<CollisionConstraintConfig> collision_constraint;
    std::unordered_map<std::string, double> per_pair_min_distance_overrides;
    std::unordered_map<std::string, double> per_pair_deferred_overrides;
    std::unordered_map<std::string, double> collision_pair_distance_floor;
    std::vector<double> position_step_collision_command_floor_distances;
    std::vector<int> collision_cache_frozen_indices;
    std::optional<CollisionDebugInfo> last_collision_debug;
    std::vector<CollisionDebugInfo> last_collision_debug_list;
    std::vector<std::size_t> last_collision_constraint_pair_indices;
    std::vector<std::size_t> collision_cached_candidate_pair_indices;
    std::vector<std::uint8_t> collision_pair_bound_valid;
    std::vector<double> collision_pair_last_signed_distance;
    std::vector<double> collision_pair_last_rel_translation_norm;
    std::vector<std::array<double, 9>> collision_pair_last_rel_rotation;
    bool collision_pair_cache_has_full_scan = false;
    int collision_pair_cache_steps_since_refresh = 0;
    std::uint64_t last_collision_pairs_considered = 0;
    std::uint64_t last_collision_exact_distance_queries = 0;
    std::uint64_t last_collision_bound_culled_pairs = 0;
    bool last_collision_budget_exhausted = false;
    double last_constraint_min_distance =
        std::numeric_limits<double>::infinity();
    double last_constraint_min_recovery_margin =
        std::numeric_limits<double>::infinity();
    bool last_constraint_was_full_scan = false;
    std::optional<CollisionConstraintResult> last_collision_constraint_result;
    Eigen::VectorXd last_collision_constraint_q;
    std::uint64_t last_collision_sphere_culled_pairs = 0;
    double last_solution_dq_norm = 0.0;
    std::unordered_map<std::size_t, int> collision_stuck_counters;
    std::unordered_map<std::size_t, double> collision_stuck_last_distances;
  };

  PositionStepMutableStateSnapshot capture_position_step_mutable_state() const;
  void restore_position_step_mutable_state(
      const PositionStepMutableStateSnapshot &snapshot);

public:
  // ========== Position IK Methods ==========

  /**
   * @brief Basic position IK solver
   * @param seed_q Initial joint configuration
   * @param target_pose Target SE3 pose in base frame
   * @param frame_name Name of the frame to control
   * @param options Position IK options (tolerances, max iterations, etc.)
   * @return Position IK result with solution and error metrics
   */
  PositionIKResult
  solve_position(const Eigen::VectorXd &seed_q,
                 const Eigen::Matrix4d &target_pose,
                 const std::string &frame_name,
                 const PositionIKOptions &options = PositionIKOptions());

  /**
   * @brief Stepping position IK using the solver's registered tasks.
   *
   * Sets the target pose on the named FrameTask, computes pose error
   * internally, scales the error by position_gain / orientation_gain to
   * produce the desired velocity, calls solve_velocity() up to max_steps
   * times, integrates after each step, and returns the result.  Because it
   * routes through solve_velocity(), the recovery state machine (stuck
   * detection, collision homotopy, etc.) is automatically exercised.
   *
   * Unlike solve_position(), this method does NOT create temporary tasks or
   * swap the solver's task list — it uses whatever tasks the caller has
   * already added via add_frame_task() / add_posture_task().
   *
   * Typical usage in an interactive loop (single step per frame):
   * @code
   *   auto task = solver.add_frame_task("ee", "end_effector");
   *   PositionStepOptions opts;
   *   opts.position_gain = 60.0;
   *   opts.orientation_gain = 60.0;
   *   while (running) {
   *     auto result = solver.solve_position_step(q, target_pose, "ee", opts);
   *     q = result.q_solution;
   *   }
   * @endcode
   *
   * @param current_q      Current joint configuration
   * @param target_pose    Target SE3 pose (4×4 homogeneous matrix)
   * @param frame_task_name Name of the registered FrameTask to drive
   * @param options        Gains, step count, timestep, speed caps, and optional
   *                       excluded_joint_indices / locked_joint_indices /
   *                       integration_zero_velocity_indices (see
   *                       PositionStepOptions)
   * @return PositionIKResult with q_solution and velocity-level diagnostics
   */
  PositionIKResult
  solve_position_step(const Eigen::VectorXd &current_q,
                      const Eigen::Matrix4d &target_pose,
                      const std::string &frame_task_name,
                      const PositionStepOptions &options = PositionStepOptions());

  /**
   * @brief Stepping position IK for multiple registered pose tasks.
   *
   * Each TaskTarget provides a task name, target pose, and per-task gains.
   * For each step, target velocities are computed from each task's pose error,
   * solve_velocity() is called once to preserve coordinated multi-task
   * behavior, then q is integrated.
   *
   * Supported task types are FrameTask, AbsoluteFrameTask, and
   * RelativeFrameTask.
   *
   * @param current_q Current joint configuration
   * @param targets List of task target descriptors
   * @param options Step count/timestep, speed caps, joint-index options;
   *                per-target gains are taken from each TaskTarget
   * @return PositionIKResult with q_solution and velocity-level diagnostics
   */
  PositionIKResult
  solve_position_step(const Eigen::VectorXd &current_q,
                      const std::vector<TaskTarget> &targets,
                      const PositionStepOptions &options = PositionStepOptions());

  /**
   * @brief TCP-relative position IK solver
   * @param seed_q Initial joint configuration
   * @param relative_target Target pose relative to current TCP frame
   * @param frame_name Name of the frame to control
   * @param options Position IK options
   * @return Position IK result with solution and error metrics
   */
  PositionIKResult
  solve_position_in_tcp(const Eigen::VectorXd &seed_q,
                        const Eigen::Matrix4d &relative_target,
                        const std::string &frame_name,
                        const PositionIKOptions &options = PositionIKOptions());
};

} // namespace embodik
