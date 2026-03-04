/**
 * @file kinematics_solver.hpp
 * @brief High-level kinematics solver for EmbodiK
 *
 * Provides a simple, high-level API for solving IK problems.
 * Handles velocity integration, limits, and solver details internally.
 */

#pragma once

#include <algorithm>
#include <cstdint>
#include <embodik/dual_arm_ects.hpp>
#include <embodik/robot_model.hpp>
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
   * @return Velocity solver result with joint velocities and saturation info
   */
  VelocitySolverResult
  solve_velocity(const Eigen::VectorXd &current_q = Eigen::VectorXd(),
                 bool apply_limits = true);

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
   * @brief Set solver tolerance
   * @param tolerance Convergence tolerance
   */
  void set_tolerance(double tolerance) { solver_tolerance_ = tolerance; }

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
   * @brief Enable a joint-limit barrier gradient task (priority 1, nullspace).
   *
   * When enabled, the solver automatically computes an analytical gradient of
   * the joint-limit-distance metric and injects it as a priority-1 velocity
   * target.  The gradient uses a barrier shape: near-zero in a configurable
   * deadband and growing as ~1/dist^2 near each limit.
   *
   * If other priority-1 tasks already exist (e.g. a posture bias), the barrier
   * velocity is appended to the same priority group so the SNS finds a single
   * least-squares solution instead of competing scales.
   *
   * The task is skipped (zero cost) when all joints are inside the deadband.
   *
   * @param barrier_margin Fraction of joint range from each limit where the
   *   barrier activates (e.g. 0.3 = outer 30% on each side).
   * @param gain Peak velocity magnitude at the limit boundary (rad/s).
   */
  void set_joint_limit_barrier_task(double barrier_margin, double gain);

  /**
   * @brief Disable the joint-limit barrier gradient task.
   */
  void clear_joint_limit_barrier_task();

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
   */
  void configure_collision_constraint(
      double min_distance,
      const std::vector<std::pair<std::string, std::string>> &include_pairs =
          {},
      const std::vector<std::pair<std::string, std::string>> &exclude_pairs =
          {},
      bool nearest_points_all_pairs = true);

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
   * @brief Disable collision avoidance constraints.
   */
  void clear_collision_constraint();

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
   */
  std::optional<CollisionDebugInfo> get_last_collision_debug() const {
    return last_collision_debug_;
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
   * @return Pair of (lower_velocity_limit, upper_velocity_limit)
   */
  std::pair<double, double> calculate_velocity_box_constraint(
      double position_margin_lower, double position_margin_upper,
      double velocity_limit, double acceleration_limit, double dt) const;

private:
  std::shared_ptr<RobotModel> robot_;
  std::vector<std::shared_ptr<Task>> tasks_;
  std::unordered_map<std::string, std::shared_ptr<Task>> task_map_;

  // Solver parameters
  double dt_ = 0.01;
  double solver_tolerance_ = 1e-6;
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

  // Constraint options
  bool use_velocity_limits_ = true;
  bool use_position_limits_ = true;

  // Joint-limit barrier task
  bool barrier_task_enabled_ = false;
  double barrier_margin_ = 0.3;
  double barrier_gain_ = 1.0;
  static constexpr double barrier_epsilon_ = 0.04;

  // Debug/perf instrumentation (off by default)
  bool timing_breakdown_enabled_ = false;

  // Floating-base bounds (optional)
  std::optional<Eigen::Vector3d> base_position_lower_;
  std::optional<Eigen::Vector3d> base_position_upper_;
  std::optional<Eigen::Vector3d> base_orientation_lower_;
  std::optional<Eigen::Vector3d> base_orientation_upper_;

  // Sort tasks by priority
  void sort_tasks_by_priority();

  // Compute barrier gradient in velocity space (nv).
  Eigen::VectorXd compute_joint_limit_barrier_gradient(
      const Eigen::VectorXd &q_current,
      const Eigen::VectorXd &q_min,
      const Eigen::VectorXd &q_max,
      const std::vector<int> &velocity_to_config_index) const;

  struct CollisionConstraintConfig {
    bool enabled = false;
    double min_distance = 0.05;
    double upper_distance = 10.0;
    double tolerance = 1e-4;
    bool nearest_points_all_pairs = true;
    std::unordered_set<std::string> include_pairs;
    std::unordered_set<std::string> exclude_pairs;
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
  };

  std::optional<RelativePoseConstraintConfig> relative_pose_constraint_;
  std::optional<RelativePoseConstraintResult> compute_relative_pose_constraint();

  std::optional<CollisionConstraintConfig> collision_constraint_;
  std::optional<CollisionDebugInfo> last_collision_debug_;
  // Cached allow-mask aligned with Pinocchio's collisionPairs indices.
  std::vector<std::uint8_t> collision_allowed_pair_mask_;
  std::optional<std::size_t> last_collision_constraint_pair_index_;
  // Track solver stagnation near collision boundary for stronger recovery.
  double last_solution_dq_norm_ = 0.0;
  int collision_stuck_counter_ = 0;
  std::optional<std::size_t> collision_stuck_pair_index_;
  double collision_stuck_last_distance_ =
      std::numeric_limits<double>::infinity();

  std::string canonical_pair_key(const std::string &a,
                                 const std::string &b) const;
  bool collision_pair_allowed(const std::string &a, const std::string &b) const;
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
