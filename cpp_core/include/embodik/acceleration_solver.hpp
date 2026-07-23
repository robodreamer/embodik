/**
 * @file acceleration_solver.hpp
 * @brief Fixed-base acceleration-level solver API.
 */

#pragma once

#include <embodik/robot_model.hpp>
#include <embodik/tasks.hpp>
#include <embodik/types.hpp>

#include <Eigen/Core>

#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace embodik {

/**
 * @brief Named caller-owned physical affine acceleration constraint.
 *
 * Each row enforces:
 *   lower_bounds <= coefficient_matrix * ddq + affine_bias <= upper_bounds
 *
 * Empty active-side vectors mean every side is active. Otherwise each vector
 * must have one flag per row, and every row must keep at least one active side.
 * The record is per-call only; the solver never stores it across solve() calls.
 */
struct AffineAccelerationConstraint {
  std::string source_id;
  Eigen::MatrixXd coefficient_matrix;
  Eigen::VectorXd affine_bias;
  Eigen::VectorXd lower_bounds;
  Eigen::VectorXd upper_bounds;
  std::vector<bool> lower_bound_active;
  std::vector<bool> upper_bound_active;
};

/**
 * @brief Named caller-owned frozen next-velocity affine constraint.
 *
 * Each row enforces:
 *   lower_bounds <= C * dq + dt * (C * ddq + affine_bias) <= upper_bounds
 *
 * The coefficient matrix is frozen for this single solve call. The solver does
 * not infer or apply any derivative of C.
 */
struct FrozenNextVelocityConstraint {
  std::string source_id;
  Eigen::MatrixXd coefficient_matrix;
  Eigen::VectorXd affine_bias;
  Eigen::VectorXd lower_bounds;
  Eigen::VectorXd upper_bounds;
  std::vector<bool> lower_bound_active;
  std::vector<bool> upper_bound_active;
};

/**
 * @brief Named caller-owned physical task acceleration bounds.
 *
 * The record names one active registered task and enforces the physical rows:
 *   lower_bounds <= J_physical * ddq + Jdot_dq <= upper_bounds
 *
 * Empty active-side vectors mean every side is active. Otherwise each vector
 * must have one flag per task row, and every row must keep at least one active
 * side. Task-local excluded joint columns are ignored for these hard physical
 * rows; exclusions still apply to the soft task objective.
 */
struct TaskAccelerationBounds {
  std::string source_id;
  std::string task_name;
  Eigen::VectorXd lower_bounds;
  Eigen::VectorXd upper_bounds;
  std::vector<bool> lower_bound_active;
  std::vector<bool> upper_bound_active;
};

struct AccelerationSolveOptions {
  std::optional<Eigen::VectorXd> acceleration_limits_override;
  bool apply_velocity_limits = true;
  bool apply_position_limits = true;
  std::vector<AffineAccelerationConstraint> affine_constraints;
  std::vector<FrozenNextVelocityConstraint> frozen_next_velocity_constraints;
  std::vector<TaskAccelerationBounds> task_acceleration_bounds;
  std::vector<int> zero_acceleration_joint_indices;
  std::vector<int> zero_next_velocity_joint_indices;
  std::vector<int> fixed_current_position_joint_indices;
};

struct AccelerationTaskReference {
  Eigen::VectorXd desired_velocity;
  Eigen::VectorXd desired_acceleration;
  double proportional_gain = 1.0;
  double derivative_gain = 1.0;
};

struct AccelerationTaskDiagnostics {
  std::string task_name;
  Eigen::VectorXd reference_acceleration;
  Eigen::VectorXd jacobian_bias;
  Eigen::VectorXd achieved_acceleration;
  Eigen::VectorXd residual;
  double scale = 1.0;
  TaskSolveMode effective_mode = TaskSolveMode::kScale;
  bool used_min_error_fallback = false;
};

struct AccelerationSolverResult : public SolverResult {
  Eigen::VectorXd joint_accelerations;
  Eigen::VectorXd joint_velocities_next;
  Eigen::VectorXd q_solution;

  /// True once the compatible acceleration box participated in the solve,
  /// including backend attempts that return no executable motion.
  bool acceleration_limits_applied = false;
  std::vector<int> saturated_acceleration_indices;
  std::vector<int> saturated_velocity_indices;
  std::vector<int> saturated_position_indices;
  std::vector<AccelerationTaskDiagnostics> task_diagnostics;
};

struct AccelerationSolverCapabilities {
  bool supports_fixed_base_scalar_joints = true;
  bool supports_floating_base = false;
  bool supports_scale_elastic = false;
  bool supports_collision_constraints = false;
  bool supports_effort_constraints = false;
};

/**
 * @brief Native acceleration-level eSNS solver for fixed-base scalar joints.
 *
 * This initial surface consumes explicit q, dq, and dt on every solve. It does
 * not store or infer previous command state. Unsupported model/task families
 * fail closed instead of falling back to velocity-level behavior.
 */
class AccelerationSolver {
public:
  explicit AccelerationSolver(std::shared_ptr<RobotModel> robot);

  AccelerationSolver(const AccelerationSolver &) = delete;
  AccelerationSolver &operator=(const AccelerationSolver &) = delete;
  AccelerationSolver(AccelerationSolver &&) noexcept = default;
  AccelerationSolver &operator=(AccelerationSolver &&) noexcept = default;

  static AccelerationSolverCapabilities capabilities();

  std::shared_ptr<FrameTask>
  add_frame_task(const std::string &name, const std::string &frame_name,
                 TaskType task_type = TaskType::FRAME_POSE);
  std::shared_ptr<COMTask> add_com_task(const std::string &name);
  std::shared_ptr<PostureTask>
  add_posture_task(const std::string &name,
                   const std::vector<int> &controlled_joints = {});
  std::shared_ptr<JointTask>
  add_joint_task(const std::string &name, const std::string &joint_name,
                 double target_value = 0.0);

  void set_task_reference(const std::string &name,
                          const AccelerationTaskReference &reference);

  std::shared_ptr<Task> get_task(const std::string &name) const;
  void remove_task(const std::string &name);
  void clear_tasks();

  AccelerationSolverResult
  solve(const Eigen::VectorXd &q, const Eigen::VectorXd &dq, double dt,
        const AccelerationSolveOptions &options = AccelerationSolveOptions{});

private:
  void ensure_unique_task_name(const std::string &name) const;

  std::shared_ptr<RobotModel> robot_;
  std::vector<std::shared_ptr<Task>> tasks_;
  std::unordered_map<std::string, std::shared_ptr<Task>> task_map_;
  std::unordered_map<std::string, AccelerationTaskReference> task_references_;
};

} // namespace embodik
