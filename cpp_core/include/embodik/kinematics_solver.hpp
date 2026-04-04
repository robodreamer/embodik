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
#include <cstdint>
#include <embodik/dual_arm_ects.hpp>
#include <embodik/robot_model.hpp>
#include <embodik/sphere_broadphase.hpp>
#include <embodik/tasks.hpp>
#include <embodik/types.hpp>
#include <limits>
#include <memory>
#include <optional>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace embodik {

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
   * @brief Enable/disable joint position limits
   * @param enable True to enable position limit constraints
   */
  void enable_position_limits(bool enable) { use_position_limits_ = enable; }

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
   * @param tolerance Regularization epsilon (default 1e-6).
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
   * @c damping * (1 - (sigma/epsilon)^2).  Setting this too large (e.g. 0.1)
   * will over-regularize the pseudoinverse and suppress joint velocities for
   * high-DOF robots where many singular values are naturally small but nonzero.
   *
   * For most robots, 1e-6 is the right value.  Call this when you need to set
   * @c set_tolerance() to a larger value for constraint-violation leniency
   * without inflating the regularization threshold.
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
  void set_damping(double damping) { damping_ = damping; }

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
   * @param max_constraints Maximum number of simultaneous collision constraint
   * rows to emit into the QP. The @p max_constraints closest pairs (each
   * within @p upper_distance of the corresponding min_distance) each get their
   * own Jacobian row and velocity-damper bounds, so the QP protects multiple
   * pairs at once. Defaults to 1 (original behaviour). Values of 3-5 are
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
   * Returns one entry per active constraint row (up to max_constraints). Empty
   * when no collision constraint is configured or no solve has been performed.
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
  std::shared_ptr<RobotModel> robot_;
  std::vector<std::shared_ptr<Task>> tasks_;
  std::unordered_map<std::string, std::shared_ptr<Task>> task_map_;

  // Solver parameters
  double dt_ = 0.01;
  // Singular-value damping threshold for regularized pseudoinverse.
  double solver_tolerance_ = 1e-6;
  // Constraint violation deadband + COD pseudoinverse relative threshold.
  double constraint_tolerance_ = 1e-6;
  double tight_tolerance_ = 1e-10;
  int max_iterations_ = 20;
  double damping_ = 1e-3;
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

  /// Sentinel in ``velocity_to_config_index_cache_`` for unmapped velocity indices.
  static constexpr int kVelocityToConfigUnmapped = -1;

  // Debug/perf instrumentation (off by default)
  bool timing_breakdown_enabled_ = false;

  // Floating-base bounds (optional)
  std::optional<Eigen::Vector3d> base_position_lower_;
  std::optional<Eigen::Vector3d> base_position_upper_;
  std::optional<Eigen::Vector3d> base_orientation_lower_;
  std::optional<Eigen::Vector3d> base_orientation_upper_;

  /// Set by solve_position_step before each inner solve_velocity(); cleared at
  /// end of solve_velocity(). Enforces v_i = 0 in the QP for listed nv-indices.
  std::vector<int> pending_velocity_lock_indices_;
  /// Optional torso constraint rows injected by solve_position_step into the
  /// next solve_velocity() call; cleared at end of solve_velocity().
  std::optional<TorsoPoseConstraintOptions> pending_step_torso_constraint_;

  /// Cached velocity-index → configuration-index map for the current robot
  /// (rebuilt when the model pointer or ``nv`` changes).
  std::vector<int> velocity_to_config_index_cache_;
  const RobotModel *velocity_to_config_cache_robot_ = nullptr;
  int velocity_to_config_cache_nv_ = -1;

  // Sort tasks by priority
  void sort_tasks_by_priority();

  /// Rebuild ``velocity_to_config_index_cache_`` if needed; return reference.
  const std::vector<int> &velocity_to_config_index_cache();

  /// Zero Jacobian entries that command motion into nearby joint limits.
  void clamp_jacobians_near_joint_limits(
      std::vector<Eigen::MatrixXd> &jacobians,
      const std::vector<int> &velocity_to_config_index) const;

  struct CollisionConstraintConfig {
    bool enabled = false;
    double min_distance = 0.05;
    double upper_distance = 10.0;
    double tolerance = 1e-4;
    bool constraint_activation_enabled = false;
    // Optional proximity gate for emitting collision QP rows.
    // <= 0 disables gating and preserves legacy behavior.
    double constraint_activation_margin = 0.0;
    // Auto-tuning factor applied to min_distance:
    //   margin = multiplier * min_distance
    // <= 0 keeps gating disabled.
    double constraint_activation_multiplier = 0.0;
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

  void stall_handler_update(VelocitySolverResult &result);

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

  std::optional<CollisionConstraintConfig> collision_constraint_;
  std::optional<CollisionDebugInfo> last_collision_debug_;
  // All active constraint pairs (up to max_constraints), populated after each solve.
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
  bool collision_pair_cache_enabled_ = true;
  int collision_pair_cache_refresh_interval_ = 100;
  double collision_pair_cache_distance_margin_ = 0.03;
  int collision_pair_cache_max_candidates_ = 128;
  int collision_refinement_time_budget_us_ = 300;
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
  bool last_constraint_was_full_scan_ = false;
  // Cached constraint result for lazy reuse when configuration change is small.
  std::optional<CollisionConstraintResult> last_collision_constraint_result_;
  Eigen::VectorXd last_collision_constraint_q_;
  CollisionTuningMode collision_tuning_mode_ = CollisionTuningMode::kSpeed;
  SphereBroadphase sphere_broadphase_;
  bool sphere_broadphase_enabled_ = false;
  std::uint64_t last_collision_sphere_culled_pairs_ = 0;
  // Track solver stagnation near collision boundary for stronger recovery (per-pair).
  double last_solution_dq_norm_ = 0.0;
  std::unordered_map<std::size_t, int> collision_stuck_counters_;
  std::unordered_map<std::size_t, double> collision_stuck_last_distances_;
  std::string canonical_pair_key(const std::string &a,
                                 const std::string &b) const;
  bool collision_pair_allowed(const std::string &a, const std::string &b) const;
  std::optional<double> evaluate_min_collision_distance(
      const Eigen::VectorXd &current_q = Eigen::VectorXd());
  std::optional<double> evaluate_min_collision_distance_targeted(
      const Eigen::VectorXd &current_q,
      const std::vector<std::size_t> &pair_indices);
  std::optional<double> evaluate_post_step_collision_distance(
      const Eigen::VectorXd &q);
  std::vector<std::size_t> get_post_step_rejection_pair_indices() const;
  std::optional<CollisionConstraintResult> compute_collision_constraint();

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
