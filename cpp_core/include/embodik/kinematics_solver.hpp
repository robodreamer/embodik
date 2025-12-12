/**
 * @file kinematics_solver.hpp
 * @brief High-level kinematics solver for EmbodiK
 *
 * Provides a simple, high-level API for solving IK problems.
 * Handles velocity integration, limits, and solver details internally.
 */

#pragma once

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
   */
  void configure_collision_constraint(
      double min_distance,
      const std::vector<std::pair<std::string, std::string>> &include_pairs =
          {},
      const std::vector<std::pair<std::string, std::string>> &exclude_pairs =
          {});

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
   * @brief Retrieve the list of currently active collision pairs (after
   * include/exclude filtering).
   */
  std::vector<std::pair<std::string, std::string>>
  get_active_collision_pairs() const;

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

  // Constraint options
  bool use_velocity_limits_ = true;
  bool use_position_limits_ = true;

  // Floating-base bounds (optional)
  std::optional<Eigen::Vector3d> base_position_lower_;
  std::optional<Eigen::Vector3d> base_position_upper_;
  std::optional<Eigen::Vector3d> base_orientation_lower_;
  std::optional<Eigen::Vector3d> base_orientation_upper_;

  // Sort tasks by priority
  void sort_tasks_by_priority();

  struct CollisionConstraintConfig {
    bool enabled = false;
    double min_distance = 0.05;
    double upper_distance = 10.0;
    double tolerance = 1e-4;
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

  std::optional<CollisionConstraintConfig> collision_constraint_;
  std::optional<CollisionDebugInfo> last_collision_debug_;

  std::string canonical_pair_key(const std::string &a,
                                 const std::string &b) const;
  bool collision_pair_allowed(const std::string &a, const std::string &b) const;
  std::optional<CollisionConstraintResult> compute_collision_constraint();

  /**
   * @brief Calculate velocity box constraints based on position, velocity, and
   * acceleration limits
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
