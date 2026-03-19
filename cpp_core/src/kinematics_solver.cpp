/**
 * @file kinematics_solver.cpp
 * @brief Implementation of high-level kinematics solver
 */

#include <Eigen/Geometry>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <limits>
#include <string>
#include <pinocchio/algorithm/geometry.hpp>
#include <stdexcept>
#include <unordered_set>
#include <utility>
#ifdef PINOCCHIO_WITH_HPP_FCL
#include <pinocchio/collision/distance.hpp>
#endif
#include <pinocchio/algorithm/joint-configuration.hpp>

#include <embodik/ik_baseline.hpp>
#include <embodik/kinematics_solver.hpp>
#include <embodik/tasks.hpp>

namespace embodik {

namespace {
constexpr double kCollisionTolerance = 1e-4;
constexpr double kCollisionUpperDistance = 1e1;
constexpr double kDistanceEpsilon = 1e-9;
constexpr double kCollisionMaxSeparationSpeed = 0.5;
constexpr double kCollisionMaxSeparationSpeedNonPenetration = 0.15;
constexpr double kCollisionPairSwitchHysteresis = 2e-3;
constexpr double kCollisionRepulsionDeadband = 3e-3;
constexpr double kCollisionViolationDeadband = 1e-3;
constexpr double kCollisionMinRecoverySpeed = 0.05;
constexpr double kCollisionRecoveryScale = 0.2;
constexpr double kCollisionStuckBand = 3e-3;
constexpr double kCollisionStuckDqNormEps = 1e-6;
constexpr int kCollisionStuckCountThreshold = 10;
// Minimum recovery speed when the stuck condition is active (non-penetrating).
constexpr double kCollisionStuckRecoverySpeed = 0.10;
constexpr int kRecoveryErrorTriggerTicks = 15;
constexpr int kRecoveryNearZeroTriggerTicks = 12;
constexpr double kRecoveryNearZeroDqEps = 5e-6;
constexpr double kRecoveryJointLimitMarginTrigger = 5e-5;
constexpr double kRecoveryJointLimitMarginHealthy = 3e-4;
constexpr int kRecoveryHealthyExitTicks = 10;
constexpr std::size_t kRecoveryHistorySize = 40;
constexpr double kRecoveryPrimaryTaskScale = 0.25;
constexpr double kRecoveryRollbackPostureGain = 0.15;
constexpr double kRecoveryCollisionBuffer = 5e-4;
constexpr double kRecoveryHomotopyRampStep = 5e-4;

// Velocity box constraint: minimum fraction of vel_limit when inside limits
constexpr double kMinBoundFraction = 0.10;
// Margin (rad) below which we do NOT inject headroom toward a limit
constexpr double kMarginThreshold = 0.01;

struct HalfspaceBoundResult {
  Eigen::VectorXd lower;
  Eigen::VectorXd upper;
  Eigen::ArrayXi violated_rows;
};

struct ClassifiedOutcome {
  SolverStatus status;
  std::string status_message;
};

static HalfspaceBoundResult compute_halfspace_velocity_bounds(
    const Eigen::VectorXd &slack, double dt, double vel_max, double acc_max,
    bool use_acceleration_limits, double proximity_threshold, double slack_eps,
    double recovery_scale, double min_recovery_speed) {
  const int n = static_cast<int>(slack.size());
  HalfspaceBoundResult out;
  out.lower.resize(n);
  out.upper.resize(n);
  out.violated_rows = Eigen::ArrayXi::Zero(n);

  const double dt_safe = std::max(dt, 1e-6);
  for (int i = 0; i < n; ++i) {
    const double m = slack(i);
    const double m_clamped = std::max(0.0, m);

    double ub = vel_max;
    if (use_acceleration_limits) {
      ub = std::min(ub, std::sqrt(2.0 * acc_max * m_clamped));
    }
    if (m < proximity_threshold) {
      ub = std::min(ub, m_clamped / dt_safe);
    }

    if (m < -slack_eps) {
      out.violated_rows(i) = 1;
      const double violation = -m - slack_eps;
      const double desired = violation / dt_safe;
      const double recovery_speed = std::min(
          vel_max, std::max(min_recovery_speed, desired * recovery_scale));
      ub = -recovery_speed;
    }

    out.lower(i) = -vel_max;
    out.upper(i) = ub;
  }
  return out;
}

static void project_task_jacobians_away_from_violated_rows(
    std::vector<Eigen::MatrixXd> &task_jacobians,
    const Eigen::MatrixXd &constraint_jacobian,
    const Eigen::ArrayXi &violated_rows, bool outward_is_positive_projection) {
  for (int k = 0; k < static_cast<int>(constraint_jacobian.rows()); ++k) {
    if (k >= violated_rows.size() || violated_rows(k) == 0) {
      continue;
    }
    Eigen::RowVectorXd n_row = constraint_jacobian.row(k);
    const double nn = n_row.squaredNorm();
    if (nn <= 1e-12) {
      continue;
    }
    n_row /= std::sqrt(nn);
    for (auto &jac : task_jacobians) {
      for (int r = 0; r < static_cast<int>(jac.rows()); ++r) {
        const double proj = jac.row(r).dot(n_row);
        const bool is_outward =
            outward_is_positive_projection ? (proj > 0.0) : (proj < 0.0);
        if (is_outward) {
          jac.row(r) -= proj * n_row;
        }
      }
    }
  }
}

static void sanitize_solver_inputs(std::vector<Eigen::VectorXd> &goals,
                                   std::vector<Eigen::MatrixXd> &jacobians,
                                   Eigen::MatrixXd &C, Eigen::VectorXd &c_lower,
                                   Eigen::VectorXd &c_upper) {
  for (auto &g : goals) {
    for (int i = 0; i < static_cast<int>(g.size()); ++i) {
      if (!std::isfinite(g(i))) {
        g(i) = 0.0;
      }
    }
  }
  for (auto &J : jacobians) {
    for (int r = 0; r < static_cast<int>(J.rows()); ++r) {
      for (int c = 0; c < static_cast<int>(J.cols()); ++c) {
        if (!std::isfinite(J(r, c))) {
          J(r, c) = 0.0;
        }
      }
    }
  }
  for (int r = 0; r < static_cast<int>(C.rows()); ++r) {
    for (int c = 0; c < static_cast<int>(C.cols()); ++c) {
      if (!std::isfinite(C(r, c))) {
        C(r, c) = 0.0;
      }
    }
    if (!std::isfinite(c_lower(r))) {
      c_lower(r) = -1e10;
    }
    if (!std::isfinite(c_upper(r))) {
      c_upper(r) = 1e10;
    }
    if (c_lower(r) > c_upper(r)) {
      const double mid = 0.5 * (c_lower(r) + c_upper(r));
      c_lower(r) = mid;
      c_upper(r) = mid;
    }
  }
}

static double compute_min_joint_limit_margin(const RobotModel &robot,
                                             const Eigen::VectorXd &q_current) {
  auto [q_min, q_max] = robot.get_joint_limits();
  const int n = std::min({static_cast<int>(q_current.size()),
                          static_cast<int>(q_min.size()),
                          static_cast<int>(q_max.size())});
  double min_margin = std::numeric_limits<double>::infinity();
  for (int i = 0; i < n; ++i) {
    if (!std::isfinite(q_min[i]) || !std::isfinite(q_max[i])) {
      continue;
    }
    min_margin = std::min(min_margin, q_current[i] - q_min[i]);
    min_margin = std::min(min_margin, q_max[i] - q_current[i]);
  }
  return min_margin;
}

static ClassifiedOutcome classify_velocity_outcome(
    SolverStatus backend_status, const std::string &backend_status_message,
    const std::vector<double> &task_scales, double primary_goal_norm,
    const std::vector<TaskSolveMode> &task_modes_effective,
    const std::vector<bool> &task_used_fallback) {
  if (backend_status != SolverStatus::kSuccess) {
    return {backend_status, backend_status_message};
  }

  constexpr double kGoalNormEps = 1e-9;
  constexpr double kScaleEps = 1e-9;
  const bool primary_is_min_error =
      !task_modes_effective.empty() &&
      task_modes_effective[0] == TaskSolveMode::kMinError;
  const bool primary_used_fallback =
      !task_used_fallback.empty() && task_used_fallback[0];
  if (primary_goal_norm > kGoalNormEps && !task_scales.empty() &&
      std::abs(task_scales[0]) <= kScaleEps && !primary_is_min_error &&
      !primary_used_fallback) {
    return {SolverStatus::kInfeasible,
            "primary task scale collapsed to zero under active constraints"};
  }

  return {SolverStatus::kSuccess, backend_status_message};
}

template <typename Derived>
static void clamp_spatial_velocity_components(Eigen::MatrixBase<Derived> &vel,
                                              double max_linear_speed,
                                              double max_angular_speed) {
  if (vel.size() < 6) {
    return;
  }
  if (max_linear_speed > 0.0) {
    const double linear_norm = vel.head(3).norm();
    if (linear_norm > max_linear_speed && linear_norm > 1e-12) {
      vel.head(3) *= (max_linear_speed / linear_norm);
    }
  }
  if (max_angular_speed > 0.0) {
    const double angular_norm = vel.tail(3).norm();
    if (angular_norm > max_angular_speed && angular_norm > 1e-12) {
      vel.tail(3) *= (max_angular_speed / angular_norm);
    }
  }
}

static bool validate_nv_index_list(const std::vector<int> &indices, int nv,
                                   const std::string &field_name,
                                   std::string *out_message) {
  for (int idx : indices) {
    if (idx < 0 || idx >= nv) {
      if (out_message != nullptr) {
        *out_message =
            field_name + " must list valid nv indices in [0, " +
            std::to_string(std::max(0, nv - 1)) + "]; got " + std::to_string(idx);
      }
      return false;
    }
  }
  return true;
}

static bool validate_position_step_joint_index_options(
    const PositionStepOptions &options, int nv, std::string *err) {
  if (!validate_nv_index_list(options.excluded_joint_indices, nv,
                              "excluded_joint_indices", err)) {
    return false;
  }
  if (!validate_nv_index_list(options.locked_joint_indices, nv,
                              "locked_joint_indices", err)) {
    return false;
  }
  if (!validate_nv_index_list(options.integration_zero_velocity_indices, nv,
                              "integration_zero_velocity_indices", err)) {
    return false;
  }
  return true;
}

static void apply_integration_velocity_mask(Eigen::VectorXd &joint_velocities,
                                            const std::vector<int> &indices) {
  for (int idx : indices) {
    joint_velocities[idx] = 0.0;
  }
}

/// Merges extra exclusions into tasks for the duration of the scope, then
/// restores previous exclusions (solve_position_step parity with solve_position).
class ScopedMergedTaskExclusions {
  std::vector<std::pair<std::shared_ptr<Task>, std::vector<int>>> saved_;

public:
  ScopedMergedTaskExclusions(
      const std::vector<int> &extra,
      const std::vector<std::shared_ptr<Task>> &tasks_to_patch) {
    if (extra.empty()) {
      return;
    }
    std::unordered_set<int> extra_set(extra.begin(), extra.end());
    saved_.reserve(tasks_to_patch.size());
    for (const auto &t : tasks_to_patch) {
      if (!t) {
        continue;
      }
      saved_.emplace_back(t, t->get_excluded_joint_indices());
      const auto &prev = saved_.back().second;
      std::unordered_set<int> merged(prev.begin(), prev.end());
      merged.insert(extra_set.begin(), extra_set.end());
      std::vector<int> merged_vec(merged.begin(), merged.end());
      std::sort(merged_vec.begin(), merged_vec.end());
      t->set_excluded_joint_indices(merged_vec);
    }
  }

  ~ScopedMergedTaskExclusions() {
    for (auto &p : saved_) {
      p.first->set_excluded_joint_indices(std::move(p.second));
    }
  }

  ScopedMergedTaskExclusions(const ScopedMergedTaskExclusions &) = delete;
  ScopedMergedTaskExclusions &
  operator=(const ScopedMergedTaskExclusions &) = delete;
};

static std::vector<std::shared_ptr<Task>>
collect_frame_and_posture_tasks_for_step_options(
    const std::shared_ptr<FrameTask> &frame_task,
    const std::vector<std::shared_ptr<Task>> &all_tasks) {
  std::vector<std::shared_ptr<Task>> out;
  std::unordered_set<Task *> seen;
  auto add = [&](const std::shared_ptr<Task> &t) {
    if (!t || seen.count(t.get()) != 0u) {
      return;
    }
    seen.insert(t.get());
    out.push_back(t);
  };
  add(std::static_pointer_cast<Task>(frame_task));
  for (const auto &t : all_tasks) {
    if (t && dynamic_cast<PostureTask *>(t.get()) != nullptr) {
      add(t);
    }
  }
  return out;
}

static std::vector<std::shared_ptr<Task>>
collect_multi_pose_and_posture_tasks_for_step_options(
    const std::vector<std::shared_ptr<Task>> &pose_tasks,
    const std::vector<std::shared_ptr<Task>> &all_tasks) {
  std::vector<std::shared_ptr<Task>> out;
  std::unordered_set<Task *> seen;
  auto add = [&](const std::shared_ptr<Task> &t) {
    if (!t || seen.count(t.get()) != 0u) {
      return;
    }
    seen.insert(t.get());
    out.push_back(t);
  };
  for (const auto &t : pose_tasks) {
    add(t);
  }
  for (const auto &t : all_tasks) {
    if (t && dynamic_cast<PostureTask *>(t.get()) != nullptr) {
      add(t);
    }
  }
  return out;
}

static ClassifiedOutcome classify_position_outcome(
    SolverStatus current_status, const std::string &current_status_message,
    bool converged_or_within_tolerance, bool stagnation_abort,
    bool max_iterations_reached, double position_error,
    double orientation_error) {
  if (current_status != SolverStatus::kSuccess) {
    return {current_status, current_status_message};
  }

  if (converged_or_within_tolerance) {
    return {SolverStatus::kSuccess, current_status_message};
  }

  if (stagnation_abort || max_iterations_reached) {
    return {SolverStatus::kInfeasible,
            "position IK did not reach tolerance (likely infeasible under "
            "active constraints): final_position_error=" +
                std::to_string(position_error) +
                ", final_orientation_error=" +
                std::to_string(orientation_error)};
  }

  return {SolverStatus::kNumericalError,
          current_status_message.empty()
              ? "position IK terminated without convergence due to numerical "
                "instability"
              : current_status_message};
}
} // namespace

KinematicsSolver::KinematicsSolver(std::shared_ptr<RobotModel> robot)
    : robot_(robot) {
  if (!robot_) {
    throw std::invalid_argument("Robot model cannot be null");
  }
}

std::shared_ptr<FrameTask>
KinematicsSolver::add_frame_task(const std::string &name,
                                 const std::string &frame_name,
                                 TaskType task_type) {

  // Check if task already exists
  if (task_map_.find(name) != task_map_.end()) {
    throw std::runtime_error("Task with name '" + name + "' already exists");
  }

  auto task = std::make_shared<FrameTask>(name, robot_, frame_name, task_type);
  tasks_.push_back(task);
  task_map_[name] = task;

  return task;
}

std::shared_ptr<COMTask>
KinematicsSolver::add_com_task(const std::string &name) {
  // Check if task already exists
  if (task_map_.find(name) != task_map_.end()) {
    throw std::runtime_error("Task with name '" + name + "' already exists");
  }

  auto task = std::make_shared<COMTask>(name, robot_);
  tasks_.push_back(task);
  task_map_[name] = task;

  return task;
}

std::shared_ptr<PostureTask>
KinematicsSolver::add_posture_task(const std::string &name,
                                   const std::vector<int> &controlled_joints) {

  // Check if task already exists
  if (task_map_.find(name) != task_map_.end()) {
    throw std::runtime_error("Task with name '" + name + "' already exists");
  }

  std::shared_ptr<PostureTask> task;
  if (controlled_joints.empty()) {
    task = std::make_shared<PostureTask>(name, robot_);
  } else {
    task = std::make_shared<PostureTask>(name, robot_, controlled_joints);
  }

  tasks_.push_back(task);
  task_map_[name] = task;

  return task;
}

std::shared_ptr<JointTask>
KinematicsSolver::add_joint_task(const std::string &name,
                                 const std::string &joint_name,
                                 double target_value) {

  // Check if task already exists
  if (task_map_.find(name) != task_map_.end()) {
    throw std::runtime_error("Task with name '" + name + "' already exists");
  }

  auto task =
      std::make_shared<JointTask>(name, robot_, joint_name, target_value);
  tasks_.push_back(task);
  task_map_[name] = task;

  return task;
}

std::shared_ptr<RelativeFrameTask>
KinematicsSolver::add_relative_frame_task(const std::string &name,
                                          const std::string &frame_a,
                                          const std::string &frame_b) {
  if (task_map_.find(name) != task_map_.end())
    throw std::runtime_error("Task with name '" + name + "' already exists");

  auto task =
      std::make_shared<RelativeFrameTask>(name, robot_, frame_a, frame_b);
  tasks_.push_back(task);
  task_map_[name] = task;
  return task;
}

std::shared_ptr<AbsoluteFrameTask>
KinematicsSolver::add_absolute_frame_task(const std::string &name,
                                          const std::string &frame_a,
                                          const std::string &frame_b,
                                          double alpha) {
  if (task_map_.find(name) != task_map_.end())
    throw std::runtime_error("Task with name '" + name + "' already exists");

  auto task =
      std::make_shared<AbsoluteFrameTask>(name, robot_, frame_a, frame_b, alpha);
  tasks_.push_back(task);
  task_map_[name] = task;
  return task;
}

void KinematicsSolver::configure_relative_pose_constraint(
    const std::string &frame_a, const std::string &frame_b,
    const Eigen::VectorXd &lower_bounds, const Eigen::VectorXd &upper_bounds,
    const Eigen::VectorXd &axis_mask) {
  if (lower_bounds.size() != 6 || upper_bounds.size() != 6)
    throw std::invalid_argument(
        "lower_bounds and upper_bounds must be 6D (pos xyz + ori xyz)");

  RelativePoseConstraintConfig cfg;
  cfg.enabled = true;
  cfg.frame_a = frame_a;
  cfg.frame_b = frame_b;
  cfg.lower_bounds = lower_bounds;
  cfg.upper_bounds = upper_bounds;

  if (axis_mask.size() == 0) {
    cfg.axis_mask = Eigen::VectorXd::Ones(6);
  } else if (axis_mask.size() == 6) {
    cfg.axis_mask = axis_mask;
  } else {
    throw std::invalid_argument("axis_mask must be empty or 6D");
  }

  relative_pose_constraint_ = std::move(cfg);
}

void KinematicsSolver::clear_relative_pose_constraint() {
  relative_pose_constraint_.reset();
}

std::optional<KinematicsSolver::RelativePoseConstraintResult>
KinematicsSolver::compute_relative_pose_constraint() {
  if (!relative_pose_constraint_.has_value() ||
      !relative_pose_constraint_->enabled)
    return std::nullopt;

  const auto &cfg = *relative_pose_constraint_;

  pinocchio::SE3 T_a = robot_->get_frame_pose(cfg.frame_a);
  pinocchio::SE3 T_b = robot_->get_frame_pose(cfg.frame_b);

  pinocchio::SE3 T_rel = compute_relative_frame(T_a, T_b);

  Matrix6Xd J_a = robot_->get_frame_jacobian(cfg.frame_a);
  Matrix6Xd J_b = robot_->get_frame_jacobian(cfg.frame_b);

  Eigen::MatrixXd J_rel_full =
      compute_relative_jacobian(J_a, J_b, T_a.rotation(), T_rel.translation());

  // Current relative pose as 6D vector: [pos_x, pos_y, pos_z, ori_x, ori_y, ori_z]
  Eigen::Vector3d rel_pos = T_rel.translation();
  Eigen::Vector3d rel_ori_log = Eigen::Vector3d::Zero();
  {
    Eigen::Matrix3d R = T_rel.rotation();
    double trace = R.trace();
    double cos_theta = std::clamp((trace - 1.0) / 2.0, -1.0, 1.0);
    double theta = std::acos(cos_theta);
    if (std::abs(theta) > 1e-6) {
      rel_ori_log = (theta / (2.0 * std::sin(theta))) *
                    Eigen::Vector3d(R(2, 1) - R(1, 2), R(0, 2) - R(2, 0),
                                    R(1, 0) - R(0, 1));
    } else {
      rel_ori_log = 0.5 * Eigen::Vector3d(R(2, 1) - R(1, 2), R(0, 2) - R(2, 0),
                                           R(1, 0) - R(0, 1));
    }
  }

  Eigen::VectorXd rel_state(6);
  rel_state.head<3>() = rel_pos;
  rel_state.tail<3>() = rel_ori_log;

  // Count active axes
  int num_active = 0;
  for (int i = 0; i < 6; ++i)
    if (cfg.axis_mask(i) > 0.5)
      num_active++;

  if (num_active == 0)
    return std::nullopt;

  RelativePoseConstraintResult result;
  result.jacobian.resize(num_active, robot_->nv());
  result.lower_bounds.resize(num_active);
  result.upper_bounds.resize(num_active);
  result.violated_rows = Eigen::ArrayXi::Zero(num_active);

  int row = 0;
  for (int i = 0; i < 6; ++i) {
    if (cfg.axis_mask(i) > 0.5) {
      result.jacobian.row(row) = J_rel_full.row(i);
      // Velocity bounds to keep within positional bounds
      double slack_lower = rel_state(i) - cfg.lower_bounds(i);
      double slack_upper = cfg.upper_bounds(i) - rel_state(i);
      const double dt_safe = std::max(dt_, 1e-6);
      const double lower_clamped = std::max(0.0, slack_lower);
      const double upper_clamped = std::max(0.0, slack_upper);
      result.lower_bounds(row) = -lower_clamped / dt_safe;
      result.upper_bounds(row) = upper_clamped / dt_safe;
      if (slack_lower < -1e-4 || slack_upper < -1e-4) {
        result.violated_rows(row) = 1;
      }
      row++;
    }
  }

  return result;
}

void KinematicsSolver::remove_task(const std::string &name) {
  auto it = task_map_.find(name);
  if (it != task_map_.end()) {
    auto task = it->second;
    task_map_.erase(it);

    // Remove from tasks vector
    tasks_.erase(std::remove(tasks_.begin(), tasks_.end(), task), tasks_.end());
  }
}

void KinematicsSolver::clear_tasks() {
  tasks_.clear();
  task_map_.clear();
}

void KinematicsSolver::clear_all_target_velocities() {
  for (auto &task : tasks_) {
    if (task) {
      task->clearTargetVelocity();
    }
  }
}

std::shared_ptr<Task> KinematicsSolver::get_task(const std::string &name) {
  auto it = task_map_.find(name);
  return (it != task_map_.end()) ? it->second : nullptr;
}

void KinematicsSolver::set_base_position_bounds(const Eigen::Vector3d &lower,
                                                const Eigen::Vector3d &upper) {
  base_position_lower_ = lower;
  base_position_upper_ = upper;
}

void KinematicsSolver::set_base_orientation_bounds(
    const Eigen::Vector3d &lower, const Eigen::Vector3d &upper) {
  base_orientation_lower_ = lower;
  base_orientation_upper_ = upper;
}

void KinematicsSolver::clear_base_bounds() {
  base_position_lower_.reset();
  base_position_upper_.reset();
  base_orientation_lower_.reset();
  base_orientation_upper_.reset();
}

void KinematicsSolver::configure_collision_constraint(
    double min_distance,
    const std::vector<std::pair<std::string, std::string>> &include_pairs,
    const std::vector<std::pair<std::string, std::string>> &exclude_pairs,
    bool nearest_points_all_pairs,
    int max_constraints) {

#ifdef PINOCCHIO_WITH_HPP_FCL
  if (!robot_->has_collision_geometry()) {
    throw std::runtime_error(
        "Collision geometry is not available for this robot model.");
  }

  auto *collision_model_ptr = robot_->collision_model();
  auto *collision_data = robot_->collision_data();

  if (!collision_constraint_.has_value()) {
    collision_constraint_.emplace();
  }

  auto &config = *collision_constraint_;
  config.enabled = true;
  config.min_distance = std::max(0.0, min_distance);
  config.upper_distance = kCollisionUpperDistance;
  config.tolerance = kCollisionTolerance;
  config.nearest_points_all_pairs = nearest_points_all_pairs;
  config.max_constraints = std::max(1, max_constraints);
  config.include_pairs.clear();
  config.exclude_pairs.clear();

  for (const auto &pair : include_pairs) {
    config.include_pairs.insert(canonical_pair_key(pair.first, pair.second));
  }

  for (const auto &pair : exclude_pairs) {
    config.exclude_pairs.insert(canonical_pair_key(pair.first, pair.second));
  }

  if (collision_data != nullptr && collision_model_ptr != nullptr) {
    const std::size_t num_pairs = collision_model_ptr->collisionPairs.size();
    if (collision_data->distanceRequests.size() != num_pairs) {
      collision_data->distanceRequests.resize(num_pairs);
      collision_data->distanceResults.resize(num_pairs);
    }
    for (auto &request : collision_data->distanceRequests) {
      request.enable_nearest_points = nearest_points_all_pairs;
      request.enable_signed_distance = true;
    }
    collision_data->activateAllCollisionPairs();

    collision_allowed_pair_mask_.assign(num_pairs, 0);
    for (std::size_t idx = 0; idx < num_pairs; ++idx) {
      const auto &pair = collision_model_ptr->collisionPairs[idx];
      const auto &name_a =
          collision_model_ptr->geometryObjects[pair.first].name;
      const auto &name_b =
          collision_model_ptr->geometryObjects[pair.second].name;
      const auto key = canonical_pair_key(name_a, name_b);
      bool allowed = true;
      if (!config.include_pairs.empty() &&
          config.include_pairs.find(key) == config.include_pairs.end()) {
        allowed = false;
      }
      if (allowed &&
          config.exclude_pairs.find(key) != config.exclude_pairs.end()) {
        allowed = false;
      }
      collision_allowed_pair_mask_[idx] = allowed ? 1 : 0;
      if (!allowed) {
        collision_data->activeCollisionPairs[idx] = false;
      }
    }
    last_collision_constraint_pair_indices_.clear();
    collision_stuck_counters_.clear();
    collision_stuck_last_distances_.clear();
  }
#else
  (void)min_distance;
  (void)include_pairs;
  (void)exclude_pairs;
  (void)nearest_points_all_pairs;
  (void)max_constraints;
  throw std::runtime_error("Collision avoidance requires Pinocchio to be built "
                           "with hpp-fcl support.");
#endif
}

std::optional<KinematicsSolver::CollisionDebugInfo>
KinematicsSolver::evaluate_collision_debug(const Eigen::VectorXd &current_q) {
#ifdef PINOCCHIO_WITH_HPP_FCL
  if (!robot_->has_collision_geometry()) {
    return std::nullopt;
  }

  if (current_q.size() > 0) {
    if (current_q.size() != robot_->nq()) {
      throw std::runtime_error(
          "Invalid configuration size for collision evaluation.");
    }
    robot_->update_kinematics(current_q);
  } else {
    robot_->update_kinematics(robot_->get_current_configuration());
  }

  // Save and restore all collision state so this call is side-effect-free.
  const auto prev_last_collision_debug = last_collision_debug_;
  const auto prev_last_collision_debug_list = last_collision_debug_list_;
  const auto prev_last_pair_indices = last_collision_constraint_pair_indices_;
  const auto prev_stuck_counters = collision_stuck_counters_;
  const auto prev_stuck_last_distances = collision_stuck_last_distances_;

  (void)compute_collision_constraint();
  const auto debug = last_collision_debug_;

  last_collision_debug_ = prev_last_collision_debug;
  last_collision_debug_list_ = prev_last_collision_debug_list;
  last_collision_constraint_pair_indices_ = prev_last_pair_indices;
  collision_stuck_counters_ = prev_stuck_counters;
  collision_stuck_last_distances_ = prev_stuck_last_distances;

  return debug;
#else
  (void)current_q;
  return std::nullopt;
#endif
}

void KinematicsSolver::add_collision_constraint(
    const std::vector<std::pair<std::string, std::string>> &link_pairs,
    double min_distance) {
#ifdef PINOCCHIO_WITH_HPP_FCL
  configure_collision_constraint(min_distance, link_pairs, {}, true);
#else
  (void)link_pairs;
  (void)min_distance;
  throw std::runtime_error("Collision avoidance requires Pinocchio to be built "
                           "with hpp-fcl support.");
#endif
}

void KinematicsSolver::clear_collision_constraint() {
  if (collision_constraint_.has_value()) {
    collision_constraint_->enabled = false;
  }
  collision_allowed_pair_mask_.clear();
}

// ============================================================
// CoM support-polygon constraint helpers
// ============================================================
namespace {

// 2D cross product of vectors OA and OB.
static double cross2d(const Eigen::Vector2d &O, const Eigen::Vector2d &A,
                      const Eigen::Vector2d &B) {
  return (A.x() - O.x()) * (B.y() - O.y()) -
         (A.y() - O.y()) * (B.x() - O.x());
}

// Graham scan: returns convex hull vertices in CCW order.
static std::vector<Eigen::Vector2d>
convex_hull_2d(std::vector<Eigen::Vector2d> pts) {
  const int n = static_cast<int>(pts.size());
  if (n < 3)
    return pts;

  std::sort(pts.begin(), pts.end(),
            [](const Eigen::Vector2d &a, const Eigen::Vector2d &b) {
              return a.x() < b.x() || (a.x() == b.x() && a.y() < b.y());
            });
  pts.erase(std::unique(pts.begin(), pts.end()), pts.end());

  if (static_cast<int>(pts.size()) < 3)
    return pts;

  std::vector<Eigen::Vector2d> hull;
  hull.reserve(2 * pts.size());

  // Lower hull
  for (const auto &p : pts) {
    while (hull.size() >= 2 &&
           cross2d(hull[hull.size() - 2], hull[hull.size() - 1], p) <= 0.0)
      hull.pop_back();
    hull.push_back(p);
  }

  // Upper hull
  const int lower_size = static_cast<int>(hull.size()) + 1;
  for (int i = static_cast<int>(pts.size()) - 2; i >= 0; --i) {
    while (static_cast<int>(hull.size()) >= lower_size &&
           cross2d(hull[hull.size() - 2], hull[hull.size() - 1], pts[i]) <=
               0.0)
      hull.pop_back();
    hull.push_back(pts[i]);
  }
  hull.pop_back();
  return hull;
}

// Build half-plane representation A * x <= b from a CCW convex polygon.
// For each edge (v_i -> v_{i+1}), the outward normal points to the right.
static void polygon_to_halfplanes(const std::vector<Eigen::Vector2d> &hull,
                                  Eigen::MatrixXd &A, Eigen::VectorXd &b) {
  const int n = static_cast<int>(hull.size());
  A.resize(n, 2);
  b.resize(n);
  Eigen::Vector2d centroid = Eigen::Vector2d::Zero();
  for (const auto &v : hull)
    centroid += v;
  centroid /= static_cast<double>(n);

  for (int i = 0; i < n; ++i) {
    const Eigen::Vector2d &v0 = hull[i];
    const Eigen::Vector2d &v1 = hull[(i + 1) % n];
    Eigen::Vector2d edge = v1 - v0;
    // Candidate outward normal (CCW hull: right side is outside).
    Eigen::Vector2d normal(edge.y(), -edge.x());
    const double len = normal.norm();
    if (len < 1e-12)
      normal = Eigen::Vector2d(1.0, 0.0);
    else
      normal /= len;

    double bi = normal.dot(v0);
    // Robust orientation guard: enforce that polygon centroid lies in the
    // feasible half-space A*x <= b (inside polygon), regardless of winding.
    if (normal.dot(centroid) > bi) {
      normal = -normal;
      bi = -bi;
    }
    A.row(i) = normal.transpose();
    b(i) = bi;
  }
}

// Shrink polygon vertices toward centroid by fractional margin in [0, 1].
// Uses mean distance from centroid to vertices (char_size) to match the
// Uses char_size (mean centroid→vertex distance) for consistent shrink behavior.
static std::vector<Eigen::Vector2d>
shrink_polygon(const std::vector<Eigen::Vector2d> &hull, double margin) {
  if (margin <= 0.0 || hull.empty())
    return hull;

  Eigen::Vector2d centroid = Eigen::Vector2d::Zero();
  for (const auto &v : hull)
    centroid += v;
  centroid /= static_cast<double>(hull.size());

  double sum_dist = 0.0;
  for (const auto &v : hull)
    sum_dist += (v - centroid).norm();
  const double char_size =
      (hull.size() > 0) ? (sum_dist / static_cast<double>(hull.size())) : 0.0;

  const double shrink_dist =
      std::clamp(margin, 0.0, 1.0) * std::max(0.0, char_size - 1e-9);

  std::vector<Eigen::Vector2d> shrunk;
  shrunk.reserve(hull.size());
  for (const auto &v : hull) {
    Eigen::Vector2d dir = centroid - v;
    const double d = dir.norm();
    if (d < 1e-12)
      shrunk.push_back(v);
    else
      shrunk.push_back(v + (shrink_dist / d) * dir);
  }
  return shrunk;
}

// Minimum perpendicular distance from the polygon centroid to any edge.
// This is the radius of the largest inscribed circle (inradius) and serves
// as the natural distance scale for the polygon.
static double polygon_inradius_2d(const std::vector<Eigen::Vector2d> &hull) {
  if (hull.size() < 3)
    return 0.0;

  Eigen::Vector2d centroid = Eigen::Vector2d::Zero();
  for (const auto &v : hull)
    centroid += v;
  centroid /= static_cast<double>(hull.size());

  double min_dist = std::numeric_limits<double>::infinity();
  const int n = static_cast<int>(hull.size());
  for (int i = 0; i < n; ++i) {
    const Eigen::Vector2d &a = hull[i];
    const Eigen::Vector2d &b = hull[(i + 1) % n];
    const Eigen::Vector2d edge = b - a;
    const double edge_len = edge.norm();
    if (edge_len < 1e-12)
      continue;
    // Outward unit normal (same convention as polygon_to_halfplanes: CCW hull)
    const Eigen::Vector2d normal(edge.y(), -edge.x());
    const double dist = std::fabs((centroid - a).dot(normal) / edge_len);
    min_dist = std::min(min_dist, dist);
  }
  return std::isfinite(min_dist) ? min_dist : 0.0;
}

} // namespace

void KinematicsSolver::configure_com_constraint(
    const Eigen::MatrixXd &vertices_xy, double margin,
    const std::string &frame_name, double com_vel_max, double com_acc_max,
    bool use_acceleration_limits, double proximity_fraction) {
  if (vertices_xy.rows() < 3) {
    throw std::invalid_argument(
        "configure_com_constraint: support polygon must have at least 3 "
        "vertices.");
  }
  if (vertices_xy.cols() < 2) {
    throw std::invalid_argument(
        "configure_com_constraint: vertices must have at least 2 columns (xy).");
  }
  if (frame_name != "world" && !robot_->has_frame(frame_name)) {
    throw std::runtime_error("configure_com_constraint: frame '" + frame_name +
                             "' not found in robot model.");
  }

  // Collect 2D points
  std::vector<Eigen::Vector2d> pts;
  pts.reserve(vertices_xy.rows());
  for (int i = 0; i < vertices_xy.rows(); ++i)
    pts.emplace_back(vertices_xy(i, 0), vertices_xy(i, 1));

  // Compute convex hull (CCW)
  auto hull = convex_hull_2d(pts);
  if (hull.size() < 3) {
    throw std::runtime_error(
        "configure_com_constraint: convex hull has fewer than 3 vertices "
        "(points may be collinear).");
  }

  // Apply inward margin
  if (margin > 0.0)
    hull = shrink_polygon(hull, margin);

  // Build half-plane representation in frame_name
  ComConstraintConfig cfg;
  cfg.enabled = true;
  cfg.vertices_xy = vertices_xy;
  cfg.margin = margin;
  cfg.frame_name = frame_name;
  cfg.com_vel_max = com_vel_max;
  cfg.com_acc_max = com_acc_max;
  cfg.use_acceleration_limits = use_acceleration_limits;
  // Auto-compute proximity threshold from the convex hull inradius so the
  // caller never needs to reason about polygon geometry themselves.
  if (proximity_fraction > 0.0)
    cfg.proximity_threshold = proximity_fraction * polygon_inradius_2d(hull);
  else
    cfg.proximity_threshold = std::numeric_limits<double>::infinity();
  polygon_to_halfplanes(hull, cfg.A, cfg.b);
  com_constraint_ = std::move(cfg);
}

void KinematicsSolver::clear_com_constraint() { com_constraint_.reset(); }

double KinematicsSolver::get_com_proximity_threshold() const {
  if (!com_constraint_.has_value())
    return 0.0;
  return com_constraint_->proximity_threshold;
}

std::optional<KinematicsSolver::ComConstraintResult>
KinematicsSolver::compute_com_constraint() {
  if (!com_constraint_.has_value() || !com_constraint_->enabled)
    return std::nullopt;

  const auto &cfg = *com_constraint_;

  // CoM position (world) and Jacobian (3 x nv)
  const Eigen::Vector3d com_world = robot_->get_com_position();
  const Eigen::MatrixXd J_com = robot_->get_com_jacobian(); // 3 x nv

  // Half-planes in frame_name; transform to world if necessary
  Eigen::MatrixXd A_world = cfg.A; // #hp x 2
  Eigen::VectorXd b_world = cfg.b; // #hp

  if (cfg.frame_name != "world") {
    const auto frame_pose = robot_->get_frame_pose(cfg.frame_name);
    const Eigen::Matrix3d R = frame_pose.rotation();
    const Eigen::Vector3d t = frame_pose.translation();
    const Eigen::Matrix2d R_xy = R.topLeftCorner<2, 2>();

    // x_F = R_xy^T * (x_world_xy - t_xy)
    // A_F * x_F <= b_F
    // => A_F * R_xy^T * x_world_xy <= b_F + A_F * R_xy^T * t_xy
    A_world = cfg.A * R_xy.transpose(); // #hp x 2
    b_world = cfg.b;
    for (int i = 0; i < static_cast<int>(b_world.size()); ++i)
      b_world(i) += A_world.row(i).dot(t.head<2>());
  }

  // Slack per half-plane: positive when CoM is inside polygon.
  const Eigen::VectorXd slack = b_world - A_world * com_world.head<2>();

  // Constraint Jacobian: all half-planes (#hp x nv)
  const Eigen::MatrixXd J_all = A_world * J_com.topRows(2);

  const int n_hp = static_cast<int>(slack.size());
  const double vel_max = cfg.com_vel_max;
  const double acc_max = cfg.com_acc_max;
  const double prox = cfg.proximity_threshold;

  // Anti-chattering: small epsilon dead-zone at the boundary.  When the
  // slack is within [-eps, 0] the CoM is treated as "at the boundary"
  // rather than outside, preventing sign-flip oscillations between
  // frames.  Matches the kMarginEpsilon pattern in
  // calculate_velocity_box_constraint().
  constexpr double kSlackEps = 1e-4; // 0.1 mm
  constexpr double kComRecoveryScale = 0.2;
  constexpr double kComMinRecoverySpeed = 0.01;

  // All half-plane rows always participate so that vel_max and acceleration
  // limits are enforced everywhere (matching the Spot Flex IK pattern).
  //
  // Three bound layers, from coarsest to tightest:
  //   1. vel_max              — always active, caps speed in every direction.
  //   2. sqrt(2*acc*slack_c)  — always active when use_acceleration_limits is
  //      set; starts tapering velocity well before the boundary, ensuring
  //      the CoM can decelerate smoothly (bounded tipping energy).
  //   3. slack_c/dt           — only active when slack < proximity_threshold;
  //      the hard position-based limit that prevents overshooting the
  //      boundary in a single time step.
  //
  // slack_c = max(0, slack): clamped to non-negative so that position and
  // acceleration terms never flip sign (same as the joint-limit pattern in
  // calculate_velocity_box_constraint).  When the CoM is slightly outside
  // (slack < 0 but > -eps), slack_c = 0 produces upper = 0: the solver
  // stops outward motion without commanding a recovery kick that would
  // cause chattering.
  const auto com_bounds =
      compute_halfspace_velocity_bounds(slack, dt_, vel_max, acc_max,
                                        cfg.use_acceleration_limits, prox,
                                        kSlackEps, kComRecoveryScale,
                                        kComMinRecoverySpeed);

  ComConstraintResult result;
  result.jacobian = J_all;
  result.lower_bounds = com_bounds.lower;
  result.upper_bounds = com_bounds.upper;
  result.violated_rows = com_bounds.violated_rows;
  return result;
}

std::string KinematicsSolver::canonical_pair_key(const std::string &a,
                                                 const std::string &b) const {
  if (a <= b) {
    return a + "|" + b;
  }
  return b + "|" + a;
}

bool KinematicsSolver::collision_pair_allowed(const std::string &a,
                                              const std::string &b) const {
  if (!collision_constraint_.has_value() || !collision_constraint_->enabled) {
    return false;
  }

  const auto key = canonical_pair_key(a, b);
  const auto &config = *collision_constraint_;

  if (!config.include_pairs.empty() &&
      config.include_pairs.find(key) == config.include_pairs.end()) {
    return false;
  }

  if (config.exclude_pairs.find(key) != config.exclude_pairs.end()) {
    return false;
  }

  return true;
}

std::vector<std::pair<std::string, std::string>>
KinematicsSolver::get_active_collision_pairs() const {
  std::vector<std::pair<std::string, std::string>> result;
#ifdef PINOCCHIO_WITH_HPP_FCL
  if (!robot_->has_collision_geometry()) {
    return result;
  }

  const auto *collision_model_ptr = robot_->collision_model();
  if (collision_model_ptr == nullptr) {
    return result;
  }

  if (!collision_constraint_.has_value() || !collision_constraint_->enabled) {
    return result;
  }

  const auto &pairs = collision_model_ptr->collisionPairs;
  if (collision_allowed_pair_mask_.size() == pairs.size()) {
    for (std::size_t idx = 0; idx < pairs.size(); ++idx) {
      if (!collision_allowed_pair_mask_[idx]) {
        continue;
      }
      const auto &pair = pairs[idx];
      const auto &name_a =
          collision_model_ptr->geometryObjects[pair.first].name;
      const auto &name_b =
          collision_model_ptr->geometryObjects[pair.second].name;
      result.emplace_back(name_a, name_b);
    }
  } else {
    for (const auto &pair : pairs) {
      const auto &name_a =
          collision_model_ptr->geometryObjects[pair.first].name;
      const auto &name_b =
          collision_model_ptr->geometryObjects[pair.second].name;
      if (collision_pair_allowed(name_a, name_b)) {
        result.emplace_back(name_a, name_b);
      }
    }
  }
#endif
  return result;
}

std::optional<KinematicsSolver::CollisionConstraintResult>
KinematicsSolver::compute_collision_constraint() {
#ifdef PINOCCHIO_WITH_HPP_FCL
  last_collision_debug_.reset();
  last_collision_debug_list_.clear();
  if (!robot_->has_collision_geometry()) {
    return std::nullopt;
  }

  auto *collision_model = robot_->collision_model();
  auto *collision_data = robot_->collision_data();
  if (collision_model == nullptr || collision_data == nullptr) {
    return std::nullopt;
  }

  bool constraint_active =
      collision_constraint_.has_value() && collision_constraint_->enabled;
  const bool nearest_points_all_pairs =
      collision_constraint_.has_value() &&
      collision_constraint_->nearest_points_all_pairs;

  // Update collision placements and compute distances for all pairs
  pinocchio::updateGeometryPlacements(robot_->model(), robot_->data(),
                                      *collision_model, *collision_data);
  const auto &pairs = collision_model->collisionPairs;

  double best_distance_debug = std::numeric_limits<double>::infinity();
  std::optional<std::size_t> best_index_debug;

  // All allowed pairs with finite distances: (distance, pair_index)
  std::vector<std::pair<double, std::size_t>> allowed_candidates;

  auto ensure_nearest_points_for_pair = [&](std::size_t idx) {
    if (nearest_points_all_pairs) {
      return;
    }
    if (idx >= collision_data->distanceRequests.size()) {
      return;
    }
    auto &req = collision_data->distanceRequests[idx];
    if (req.enable_nearest_points) {
      return;
    }
    req.enable_nearest_points = true;
    pinocchio::computeDistance(*collision_model, *collision_data, idx);
    req.enable_nearest_points = false;
  };

  for (std::size_t idx = 0; idx < pairs.size(); ++idx) {
    // Honor Pinocchio's active-pair mask *before* distance computation.
    if (!collision_data->activeCollisionPairs.empty() &&
        !collision_data->activeCollisionPairs[idx]) {
      continue;
    }
    pinocchio::computeDistance(*collision_model, *collision_data, idx);

    const auto &distance_result = collision_data->distanceResults[idx];
    double distance = distance_result.min_distance;
    if (!std::isfinite(distance)) {
      continue;
    }

    if (distance < best_distance_debug) {
      best_distance_debug = distance;
      best_index_debug = idx;
    }

    if (!constraint_active) {
      continue;
    }

    // Only consider include/exclude-allowed pairs for the constraint.
    if (!collision_allowed_pair_mask_.empty() &&
        idx < collision_allowed_pair_mask_.size() &&
        !collision_allowed_pair_mask_[idx]) {
      continue;
    }

    allowed_candidates.emplace_back(distance, idx);
  }

  // ---- Pair selection: top-K with hysteresis ----
  // Sort all allowed candidates by distance (ascending).
  std::sort(allowed_candidates.begin(), allowed_candidates.end());

  const int max_k = collision_constraint_.has_value()
                        ? collision_constraint_->max_constraints
                        : 1;

  // Determine the distance threshold below which a previous pair "sticks".
  // A previous pair is kept if its distance is within hysteresis of the
  // current K-th best candidate.
  double hysteresis_cutoff = std::numeric_limits<double>::infinity();
  if (!allowed_candidates.empty()) {
    const std::size_t kth = static_cast<std::size_t>(
        std::min(max_k, static_cast<int>(allowed_candidates.size())) - 1);
    hysteresis_cutoff = allowed_candidates[kth].first +
                        kCollisionPairSwitchHysteresis;
  }

  // Build the selected set: start with top-K candidates, then admit any
  // previous pairs that fall within hysteresis_cutoff.
  std::unordered_set<std::size_t> selected_set;
  for (int i = 0;
       i < max_k && i < static_cast<int>(allowed_candidates.size()); ++i) {
    selected_set.insert(allowed_candidates[i].second);
  }
  for (std::size_t prev_idx : last_collision_constraint_pair_indices_) {
    if (static_cast<int>(selected_set.size()) >= max_k) {
      break;
    }
    if (prev_idx >= pairs.size()) {
      continue;
    }
    if (!collision_data->activeCollisionPairs.empty() &&
        !collision_data->activeCollisionPairs[prev_idx]) {
      continue;
    }
    if (!collision_allowed_pair_mask_.empty() &&
        prev_idx < collision_allowed_pair_mask_.size() &&
        !collision_allowed_pair_mask_[prev_idx]) {
      continue;
    }
    const double prev_dist =
        collision_data->distanceResults[prev_idx].min_distance;
    if (std::isfinite(prev_dist) && prev_dist <= hysteresis_cutoff) {
      selected_set.insert(prev_idx);
    }
  }

  // Sort the selected set by distance to produce a deterministic order and
  // trim to max_k if hysteresis temporarily pushed us over.
  std::vector<std::pair<double, std::size_t>> selected_sorted;
  selected_sorted.reserve(selected_set.size());
  for (std::size_t idx : selected_set) {
    selected_sorted.emplace_back(
        collision_data->distanceResults[idx].min_distance, idx);
  }
  std::sort(selected_sorted.begin(), selected_sorted.end());
  if (static_cast<int>(selected_sorted.size()) > max_k) {
    selected_sorted.resize(static_cast<std::size_t>(max_k));
  }

  // Update the per-step active indices for next step's hysteresis.
  last_collision_constraint_pair_indices_.clear();
  for (const auto &[dist, idx] : selected_sorted) {
    last_collision_constraint_pair_indices_.push_back(idx);
  }

  // Clean up stuck-detection state for pairs no longer active.
  {
    std::unordered_set<std::size_t> active_set(
        last_collision_constraint_pair_indices_.begin(),
        last_collision_constraint_pair_indices_.end());
    for (auto it = collision_stuck_counters_.begin();
         it != collision_stuck_counters_.end();) {
      if (active_set.count(it->first) == 0) {
        collision_stuck_last_distances_.erase(it->first);
        it = collision_stuck_counters_.erase(it);
      } else {
        ++it;
      }
    }
  }

  // ---- Debug info ----
  // For the single-pair API (backward compat), use the closest active pair
  // or, if no constraint is active, the globally closest.
  std::optional<std::size_t> debug_index_to_use = best_index_debug;
  if (constraint_active && !selected_sorted.empty()) {
    debug_index_to_use = selected_sorted.front().second;
  }

  if (debug_index_to_use.has_value()) {
    ensure_nearest_points_for_pair(*debug_index_to_use);
    const auto &pair_debug = pairs[*debug_index_to_use];
    const auto &obj_da = collision_model->geometryObjects[pair_debug.first];
    const auto &obj_db = collision_model->geometryObjects[pair_debug.second];
    const auto &res_debug =
        collision_data->distanceResults[*debug_index_to_use];
    CollisionDebugInfo debug_info;
    debug_info.object_a = obj_da.name;
    debug_info.object_b = obj_db.name;
    debug_info.distance = res_debug.min_distance;
    debug_info.point_a_world = res_debug.nearest_points[0].cast<double>();
    debug_info.point_b_world = res_debug.nearest_points[1].cast<double>();
    last_collision_debug_ = debug_info;
  } else {
    last_collision_debug_.reset();
  }

  if (!constraint_active || selected_sorted.empty()) {
    return std::nullopt;
  }

  const auto &config = *collision_constraint_;
  const double dt = std::max(dt_, 1e-6);
  const int nv = robot_->nv();
  const int num_selected = static_cast<int>(selected_sorted.size());

  CollisionConstraintResult result;
  result.jacobian.resize(num_selected, nv);
  result.lower_bounds.resize(num_selected);
  result.upper_bounds.resize(num_selected);

  // Per-pair velocity-damper bound computation (with continuous recovery ramp).
  auto compute_bounds_for_pair = [&](std::size_t pair_idx,
                                     double signed_distance,
                                     bool *stuck_out) -> std::pair<double, double> {
    const double target_min_distance = config.min_distance;
    double effective_min_distance = target_min_distance;
    if (recovery_state_.active) {
      auto it = collision_effective_min_distance_.find(pair_idx);
      if (it == collision_effective_min_distance_.end()) {
        const double seeded = signed_distance + kRecoveryCollisionBuffer;
        it = collision_effective_min_distance_.emplace(pair_idx, seeded).first;
      }
      it->second = std::min(target_min_distance,
                            std::max(it->second, signed_distance));
      effective_min_distance = it->second;
    } else {
      collision_effective_min_distance_.erase(pair_idx);
    }

    // Detect stuck: non-penetrating but significantly inside min_distance for
    // multiple consecutive cycles AND the previous dq was near-zero.
    bool stuck_active = false;
    const bool deep_non_penetration =
        (signed_distance >= 0.0) &&
        (signed_distance < (effective_min_distance - kCollisionStuckBand));
    const bool dq_small = (last_solution_dq_norm_ < kCollisionStuckDqNormEps);
    if (deep_non_penetration && dq_small) {
      int &counter = collision_stuck_counters_[pair_idx];
      double &last_dist = collision_stuck_last_distances_[pair_idx];
      const bool not_improving =
          std::isfinite(last_dist) && (signed_distance <= last_dist + 1e-6);
      counter = not_improving ? counter + 1 : 1;
      last_dist = signed_distance;
      stuck_active = (counter >= kCollisionStuckCountThreshold);
    } else {
      collision_stuck_counters_.erase(pair_idx);
      collision_stuck_last_distances_.erase(pair_idx);
    }
    if (stuck_out != nullptr) {
      *stuck_out = stuck_active;
    }

    // Velocity-damper lower bound:
    // - outside deadband: allow approach up to the damper limit (negative lb)
    // - inside deadband (min <= d < min+deadband): no-approach (lb=0)
    // - violated (d < min): continuous recovery ramp without discrete tiers
    double lower_bound = 0.0;
    if (signed_distance >= (effective_min_distance + kCollisionRepulsionDeadband)) {
      lower_bound =
          (effective_min_distance + config.tolerance - signed_distance) / dt;
    } else if (signed_distance >= effective_min_distance) {
      lower_bound = 0.0;
    } else {
      if (signed_distance >=
          (effective_min_distance - kCollisionViolationDeadband)) {
        // Small violation dead-zone to reduce chatter at the active boundary.
        lower_bound = 0.0;
        const double upper_bound =
            (config.upper_distance - config.tolerance + signed_distance) / dt;
        return {lower_bound, upper_bound};
      }
      // Violated region: uniform continuous recovery ramp for both
      // slightly-inside and deeper violations. The old "gentle_scale = 0.01"
      // for the slightly-inside case produced ~0.005 m/s which was too weak
      // to overcome typical EE task pulls. Using kCollisionRecoveryScale
      // uniformly provides a meaningful push at all violation depths while
      // remaining capped for stability.
      const double desired =
          (effective_min_distance + config.tolerance - signed_distance) / dt;
      if (signed_distance >= 0.0) {
        // Non-penetrating: proportional recovery, capped.
        lower_bound = std::min(kCollisionMaxSeparationSpeedNonPenetration,
                               std::max(0.0, desired * kCollisionRecoveryScale));
      } else {
        // Penetrating: enforce a minimum recovery speed, still capped.
        lower_bound = std::min(kCollisionMaxSeparationSpeed, desired);
        lower_bound = std::max(lower_bound, kCollisionMinRecoverySpeed);
      }
    }

    // Stuck override: ensure a floor that can actually produce motion.
    if (stuck_active && signed_distance >= 0.0) {
      const double desired =
          (effective_min_distance + config.tolerance - signed_distance) / dt;
      lower_bound = std::max(
          lower_bound,
          std::min(kCollisionMaxSeparationSpeedNonPenetration,
                   std::max(kCollisionStuckRecoverySpeed,
                            desired * kCollisionRecoveryScale)));
    }

    if (recovery_state_.active) {
      auto it = collision_effective_min_distance_.find(pair_idx);
      if (it != collision_effective_min_distance_.end()) {
        it->second = std::min(target_min_distance,
                              std::max(it->second, signed_distance) +
                                  kRecoveryHomotopyRampStep);
      }
    }

    const double upper_bound =
        (config.upper_distance - config.tolerance + signed_distance) / dt;
    return {lower_bound, upper_bound};
  };

  // Build one constraint row per selected pair.
  for (int row = 0; row < num_selected; ++row) {
    const std::size_t pair_idx = selected_sorted[row].second;
    ensure_nearest_points_for_pair(pair_idx);

    const auto &pair = pairs[pair_idx];
    const auto &object_a = collision_model->geometryObjects[pair.first];
    const auto &object_b = collision_model->geometryObjects[pair.second];
    const auto &distance_result = collision_data->distanceResults[pair_idx];
    const double signed_distance = distance_result.min_distance;

    const Eigen::Vector3d p1_world =
        distance_result.nearest_points[0].cast<double>();
    const Eigen::Vector3d p2_world =
        distance_result.nearest_points[1].cast<double>();

    const Eigen::Vector3d distance_vector = p1_world - p2_world;
    const double distance_norm = distance_vector.norm();
    Eigen::Vector3d normal = Eigen::Vector3d::UnitX();
    if (distance_norm > kDistanceEpsilon) {
      normal = distance_vector / distance_norm;
    }

    const auto frame_a_id = object_a.parentFrame;
    const auto frame_b_id = object_b.parentFrame;
    const auto &frame_a = robot_->model().frames[frame_a_id];
    const auto &frame_b = robot_->model().frames[frame_b_id];
    const auto &transform_a = robot_->data().oMf[frame_a_id];
    const auto &transform_b = robot_->data().oMf[frame_b_id];

    const Eigen::Vector3d p1_local =
        transform_a.rotation().transpose() *
        (p1_world - transform_a.translation());
    const Eigen::Vector3d p2_local =
        transform_b.rotation().transpose() *
        (p2_world - transform_b.translation());

    const Eigen::Matrix<double, 3, Eigen::Dynamic> jacobian_a =
        robot_->get_point_jacobian(frame_a.name, p1_local);
    const Eigen::Matrix<double, 3, Eigen::Dynamic> jacobian_b =
        robot_->get_point_jacobian(frame_b.name, p2_local);

    Eigen::Matrix3d rotation_to_x = Eigen::Matrix3d::Identity();
    Eigen::Matrix3d rotation_from_negative = Eigen::Matrix3d::Identity();
    if (distance_norm > kDistanceEpsilon) {
      rotation_to_x =
          Eigen::Quaterniond::FromTwoVectors(normal, Eigen::Vector3d::UnitX())
              .toRotationMatrix();
      rotation_from_negative =
          Eigen::Quaterniond::FromTwoVectors(-normal, Eigen::Vector3d::UnitX())
              .toRotationMatrix();
    }

    // Relative separating velocity constraint: normalᵀ (v_a - v_b) >= lb
    result.jacobian.row(row) =
        (rotation_to_x * jacobian_a).row(0) +
        (rotation_from_negative * jacobian_b).row(0);

    const auto [lb, ub] =
        compute_bounds_for_pair(pair_idx, signed_distance, nullptr);
    result.lower_bounds(row) = lb;
    result.upper_bounds(row) = ub;

    // Populate per-pair debug info.
    CollisionDebugInfo pair_debug;
    pair_debug.object_a = object_a.name;
    pair_debug.object_b = object_b.name;
    pair_debug.distance = signed_distance;
    pair_debug.point_a_world = p1_world;
    pair_debug.point_b_world = p2_world;
    last_collision_debug_list_.push_back(pair_debug);

    // Keep the result fields for the closest pair (backward compat).
    if (row == 0) {
      result.distance = signed_distance;
      result.object_a = object_a.name;
      result.object_b = object_b.name;
      result.point_a_world = p1_world;
      result.point_b_world = p2_world;
    }
  }

  return result;
#else
  last_collision_debug_.reset();
  last_collision_debug_list_.clear();
  return std::nullopt;
#endif
}

std::pair<double, double> KinematicsSolver::calculate_velocity_box_constraint(
    double position_margin_lower, double position_margin_upper,
    double velocity_limit, double acceleration_limit, double dt) const {
  const double raw_margin_lower = position_margin_lower;
  const double raw_margin_upper = position_margin_upper;
  const bool outside_lower = raw_margin_lower < -limit_recovery_enter_epsilon_;
  const bool outside_upper = raw_margin_upper < -limit_recovery_enter_epsilon_;
  const double effective_release_margin = std::max(
      limit_exit_release_margin_,
      limit_recovery_exit_epsilon_ - limit_recovery_enter_epsilon_);

  if (outside_lower && outside_upper) {
    return std::make_pair(-velocity_limit, velocity_limit);
  }

  // Clamp margins to be non-negative for nominal bounds.
  position_margin_lower = std::max(0.0, position_margin_lower);
  position_margin_upper = std::max(0.0, position_margin_upper);

  // Calculate velocity limits based on position margins
  // Computed as: min of position/dt, vel_max, and sqrt(2*accel*margin)
  double vel_from_pos_lower = -position_margin_lower / dt;
  double vel_from_pos_upper = position_margin_upper / dt;

  double vel_from_accel_lower =
      -std::sqrt(2 * acceleration_limit * position_margin_lower);
  double vel_from_accel_upper =
      std::sqrt(2 * acceleration_limit * position_margin_upper);

  // Take most restrictive limits
  double lower_limit =
      std::max({vel_from_pos_lower, -velocity_limit, vel_from_accel_lower});
  double upper_limit =
      std::min({vel_from_pos_upper, velocity_limit, vel_from_accel_upper});

  // When a joint is inside both limits, guarantee a minimum velocity
  // allowance so the hierarchical (SNS) solver can find partial solutions
  // instead of collapsing the task scale to zero.  The post-solve
  // position clamp prevents actual limit violations, so this "softening"
  // only affects the velocity-level QP feasibility.
  //
  // Only guarantee headroom in the direction AWAY from a nearby limit.
  // kMarginThreshold prevents softening from injecting velocity toward a
  // limit the joint is already at.
  //
  // Disabled by default (enable_saturation_exit_behavior_); requires more
  // testing before enabling.
  if (enable_saturation_exit_behavior_ && !outside_lower && !outside_upper) {
    const double min_vel = kMinBoundFraction * velocity_limit;
    if (lower_limit > -min_vel && raw_margin_lower > kMarginThreshold)
      lower_limit = -min_vel;
    if (upper_limit < min_vel && raw_margin_upper > kMarginThreshold)
      upper_limit = min_vel;
  }

  if (outside_lower) {
    const double violation =
        std::max(0.0, -raw_margin_lower - effective_release_margin);
    const double recovery_min = (limit_recovery_gain_ * violation) / dt;
    const double recovery_lower = std::min(recovery_min, velocity_limit);
    if (recovery_lower > lower_limit) {
      lower_limit = recovery_lower;
    }
  } else if (outside_upper) {
    const double violation =
        std::max(0.0, -raw_margin_upper - effective_release_margin);
    const double recovery_max = -(limit_recovery_gain_ * violation) / dt;
    const double recovery_upper = std::max(recovery_max, -velocity_limit);
    if (recovery_upper < upper_limit) {
      upper_limit = recovery_upper;
    }
  }

  if (lower_limit > upper_limit) {
    const double midpoint = 0.5 * (lower_limit + upper_limit);
    lower_limit = midpoint;
    upper_limit = midpoint;
  }

  return std::make_pair(lower_limit, upper_limit);
}

void KinematicsSolver::set_joint_limit_barrier_task(double barrier_margin,
                                                    double gain) {
  barrier_margin_ = std::clamp(barrier_margin, 0.01, 0.5);
  barrier_gain_ = std::max(0.0, gain);
  barrier_task_enabled_ = true;
}

void KinematicsSolver::clear_joint_limit_barrier_task() {
  barrier_task_enabled_ = false;
}

Eigen::VectorXd KinematicsSolver::compute_joint_limit_barrier_gradient(
    const Eigen::VectorXd &q_current,
    const Eigen::VectorXd &q_min,
    const Eigen::VectorXd &q_max,
    const std::vector<int> &velocity_to_config_index) const {
  const int nv = robot_->nv();
  Eigen::VectorXd grad = Eigen::VectorXd::Zero(nv);

  for (int i = 0; i < nv; ++i) {
    const int q_idx = velocity_to_config_index[i];
    if (q_idx < 0 || q_idx >= q_min.size() || q_idx >= q_max.size()) {
      continue;
    }
    const double range = q_max[q_idx] - q_min[q_idx];
    if (range < 1e-6 || !std::isfinite(q_min[q_idx]) ||
        !std::isfinite(q_max[q_idx])) {
      continue;
    }
    // Normalized position: 0 at center, +/-1 at limits
    const double p = 2.0 * (q_current[q_idx] - q_min[q_idx]) / range - 1.0;
    // Deadband: barrier is zero when |p| < (1 - 2*barrier_margin_)
    const double deadband = 1.0 - 2.0 * barrier_margin_;
    if (std::abs(p) < deadband) {
      continue;
    }
    const double e = barrier_epsilon_;
    const double a = 1.0 + e - p;
    const double b = p + 1.0 + e;
    const double ab = a * b;
    if (ab < 1e-12) {
      continue;
    }
    // d/dp [p^2 / (a*b)] = (2*p*a*b + p^2*(a - b)) / (a*b)^2
    const double dhdp = (2.0 * p * ab + p * p * (a - b)) / (ab * ab);
    // Chain rule: dh/dq = dh/dp * dp/dq = dh/dp * (2/range)
    // Negative sign: gradient descent (push away from limits)
    grad[i] = -dhdp * (2.0 / range);
  }
  return grad;
}

void KinematicsSolver::sort_tasks_by_priority() {
  std::stable_sort(
      tasks_.begin(), tasks_.end(),
      [](const std::shared_ptr<Task> &a, const std::shared_ptr<Task> &b) {
        return a->getPriority() < b->getPriority();
      });
}

VelocitySolverResult
KinematicsSolver::solve_velocity(const Eigen::VectorXd &current_q,
                                 bool apply_limits) {
  const bool timing = timing_breakdown_enabled_;
  auto get_elapsed_ms = [](auto start) {
    return std::chrono::duration_cast<std::chrono::microseconds>(
               std::chrono::high_resolution_clock::now() - start)
               .count() /
           1000.0;
  };

  VelocitySolverResult result;
  struct ClearPendingVelocityLocks {
    KinematicsSolver *solver;
    ~ClearPendingVelocityLocks() {
      if (solver != nullptr) {
        solver->pending_velocity_lock_indices_.clear();
      }
    }
  } clear_pending_locks{this};
  (void)clear_pending_locks;
  // Timing fields default to 0.0; only populate when timing is enabled.
  const Eigen::VectorXd q_eval =
      (current_q.size() > 0) ? current_q : robot_->get_current_configuration();

  // Use provided configuration or robot's current
  if (current_q.size() > 0) {
    if (current_q.size() != robot_->nq()) {
      result.status = SolverStatus::kInvalidInput;
      result.status_message =
          "current_q size does not match robot nq in solve_velocity";
      return result;
    }
    if (timing) {
      auto t_kin_start = std::chrono::high_resolution_clock::now();
      robot_->update_kinematics(current_q);
      result.pinocchio_kinematics_time_ms = get_elapsed_ms(t_kin_start);
    } else {
      robot_->update_kinematics(current_q);
    }
  } else {
    if (timing) {
      auto t_kin_start = std::chrono::high_resolution_clock::now();
      robot_->update_kinematics(robot_->get_current_configuration());
      result.pinocchio_kinematics_time_ms = get_elapsed_ms(t_kin_start);
    } else {
      robot_->update_kinematics(robot_->get_current_configuration());
    }
  }

  // Sort tasks by priority
  sort_tasks_by_priority();

  // Update all tasks with current robot state
  if (timing) {
    auto t_task_start = std::chrono::high_resolution_clock::now();
    for (auto &task : tasks_) {
      if (task->isActive()) {
        task->update(*robot_);
      }
    }
    result.task_update_time_ms = get_elapsed_ms(t_task_start);
  } else {
    for (auto &task : tasks_) {
      if (task->isActive()) {
        task->update(*robot_);
      }
    }
  }

  // Collect active tasks: group by priority for order-invariant behavior.
  std::vector<Eigen::VectorXd> goals;
  std::vector<Eigen::MatrixXd> jacobians;
  std::vector<ObjectiveSolveConfig> objective_configs;
  std::vector<std::shared_ptr<Task>> objective_tasks;
  std::unordered_set<int> excluded_union;

  int current_priority = std::numeric_limits<int>::min();
  std::vector<std::shared_ptr<Task>> group_tasks;
  std::vector<Eigen::VectorXd> group_goals;
  std::vector<Eigen::MatrixXd> group_jacobians;
  group_tasks.reserve(tasks_.size());
  group_goals.reserve(tasks_.size());
  group_jacobians.reserve(tasks_.size());

  auto flush_group = [&]() {
    if (group_tasks.empty()) {
      return;
    }

    bool all_scale_no_fallback = true;
    for (const auto &task : group_tasks) {
      if (task->getSolveMode() != TaskSolveMode::kScale ||
          task->getAllowMinErrorFallback()) {
        all_scale_no_fallback = false;
        break;
      }
    }

    if (all_scale_no_fallback) {
      int total_rows = 0;
      for (const auto &g : group_goals) {
        total_rows += static_cast<int>(g.rows());
      }
      Eigen::VectorXd combined_goal(total_rows);
      Eigen::MatrixXd combined_jac(total_rows, robot_->nv());
      combined_goal.setZero();
      combined_jac.setZero();
      int offset = 0;
      for (size_t i = 0; i < group_goals.size(); ++i) {
        const auto &g = group_goals[i];
        const auto &J = group_jacobians[i];
        if (g.rows() > 0) {
          combined_goal.segment(offset, g.rows()) = g;
          combined_jac.block(offset, 0, J.rows(), robot_->nv()) = J;
          offset += static_cast<int>(g.rows());
        }
      }
      goals.push_back(std::move(combined_goal));
      jacobians.push_back(std::move(combined_jac));
      objective_configs.push_back(
          ObjectiveSolveConfig{current_priority, TaskSolveMode::kScale, false});
      objective_tasks.push_back(nullptr);
    } else {
      for (size_t i = 0; i < group_tasks.size(); ++i) {
        const auto &task = group_tasks[i];
        goals.push_back(group_goals[i]);
        jacobians.push_back(group_jacobians[i]);
        objective_configs.push_back(ObjectiveSolveConfig{
            task->getPriority(),
            task->getSolveMode(),
            task->getAllowMinErrorFallback(),
        });
        objective_tasks.push_back(task);
      }
    }

    group_tasks.clear();
    group_goals.clear();
    group_jacobians.clear();
  };

  for (const auto &task : tasks_) {
    if (!task->isActive()) {
      continue;
    }
    for (int idx : task->get_excluded_joint_indices()) {
      if (idx >= 0 && idx < robot_->nv()) {
        excluded_union.insert(idx);
      }
    }

    const int prio = task->getPriority();
    if (group_tasks.empty()) {
      current_priority = prio;
    } else if (prio != current_priority) {
      flush_group();
      current_priority = prio;
    }

    group_tasks.push_back(task);
    group_goals.push_back(task->getVelocity());
    group_jacobians.push_back(task->getJacobian());
  }
  flush_group();

  for (int idx : pending_velocity_lock_indices_) {
    if (idx >= 0 && idx < robot_->nv()) {
      excluded_union.insert(idx);
    }
  }

  if (recovery_state_.active && !goals.empty()) {
    // During recovery, reduce primary EE pull and allow min-error adaptation.
    goals[0] *= kRecoveryPrimaryTaskScale;
    if (!objective_configs.empty()) {
      objective_configs[0].solve_mode = TaskSolveMode::kMinError;
      objective_configs[0].allow_min_error_fallback = true;
    }
    // Use the best buffered feasible state as a posture bias to walk out of
    // deadlock without requiring an abrupt external state reset.
    if (recovery_state_.has_rollback_target &&
        recovery_state_.rollback_target_q.size() == robot_->nq()) {
      const int nv = robot_->nv();
      Eigen::VectorXd posture_goal =
          Eigen::VectorXd::Zero(nv);
      const int n = std::min(nv, static_cast<int>(q_eval.size()));
      for (int i = 0; i < n; ++i) {
        posture_goal(i) =
            (recovery_state_.rollback_target_q(i) - q_eval(i)) /
            std::max(dt_, 1e-6);
      }
      posture_goal *= kRecoveryRollbackPostureGain;
      goals.push_back(posture_goal);
      jacobians.push_back(Eigen::MatrixXd::Identity(nv, nv));
      objective_configs.push_back(
          ObjectiveSolveConfig{1, TaskSolveMode::kMinError, false});
      objective_tasks.push_back(nullptr);
    } else {
      auto [q_min, q_max] = robot_->get_joint_limits();
      const int nv = robot_->nv();
      Eigen::VectorXd posture_goal = Eigen::VectorXd::Zero(nv);
      const int n = std::min({nv, static_cast<int>(q_eval.size()),
                              static_cast<int>(q_min.size()),
                              static_cast<int>(q_max.size())});
      for (int i = 0; i < n; ++i) {
        if (!std::isfinite(q_min[i]) || !std::isfinite(q_max[i])) {
          continue;
        }
        const double mid = 0.5 * (q_min[i] + q_max[i]);
        posture_goal(i) = (mid - q_eval(i)) / std::max(dt_, 1e-6);
      }
      posture_goal *= kRecoveryRollbackPostureGain;
      goals.push_back(posture_goal);
      jacobians.push_back(Eigen::MatrixXd::Identity(nv, nv));
      objective_configs.push_back(
          ObjectiveSolveConfig{1, TaskSolveMode::kMinError, false});
      objective_tasks.push_back(nullptr);
    }
  }

  // Build velocity-to-configuration index mapping (needed by barrier task
  // and position-based velocity constraints).
  std::vector<int> velocity_to_config_index(robot_->nv(), -1);
  {
    const auto joint_names = robot_->get_joint_names();
    for (const auto &joint_name : joint_names) {
      const int v_idx = robot_->get_joint_velocity_index(joint_name);
      const int v_size = robot_->get_joint_velocity_size(joint_name);
      const int q_idx = robot_->get_joint_config_index(joint_name);
      const int q_size = robot_->get_joint_config_size(joint_name);
      if (v_size <= 0 || v_idx < 0 || q_idx < 0) {
        continue;
      }
      if (v_size == 1 && q_size != 1) {
        continue;
      }
      const int dims = std::min(v_size, q_size);
      for (int k = 0; k < dims; ++k) {
        const int vk = v_idx + k;
        const int qk = q_idx + k;
        if (vk >= 0 && vk < robot_->nv() && qk >= 0 &&
            qk < robot_->nq()) {
          velocity_to_config_index[vk] = qk;
        }
      }
    }
  }

  // Inject joint-limit barrier gradient as a priority-1 nullspace task.
  // Only add rows for joints with non-zero gradient (near limits) to avoid
  // expensive nv×nv identity matrix when most joints are in the deadband.
  if (barrier_task_enabled_ && apply_limits) {
    const int nv = robot_->nv();
    auto [q_min, q_max] = robot_->get_joint_limits();
    Eigen::VectorXd q_current = robot_->get_current_configuration();
    Eigen::VectorXd barrier_vel =
        barrier_gain_ *
        compute_joint_limit_barrier_gradient(q_current, q_min, q_max,
                                             velocity_to_config_index);

    if (barrier_vel.squaredNorm() >= 1e-12) {
      // Collect indices for near-limit joints (non-zero gradient)
      std::vector<int> active_indices;
      active_indices.reserve(nv);
      for (int i = 0; i < nv; ++i) {
        if (std::abs(barrier_vel[i]) >= 1e-12) {
          active_indices.push_back(i);
        }
      }
      const int k = static_cast<int>(active_indices.size());
      if (k > 0) {
        Eigen::VectorXd barrier_goal(k);
        Eigen::MatrixXd barrier_jac(k, nv);
        barrier_jac.setZero();
        for (int j = 0; j < k; ++j) {
          const int idx = active_indices[j];
          barrier_goal[j] = barrier_vel[idx];
          barrier_jac(j, idx) = 1.0;
        }
        goals.push_back(std::move(barrier_goal));
        jacobians.push_back(std::move(barrier_jac));
        objective_configs.push_back(ObjectiveSolveConfig{
            1,
            TaskSolveMode::kMinError,
            false,
        });
        objective_tasks.push_back(nullptr);
      }
    }
  }

  if (!pending_velocity_lock_indices_.empty()) {
    const int nv_lock = robot_->nv();
    for (int idx : pending_velocity_lock_indices_) {
      if (idx < 0 || idx >= nv_lock) {
        continue;
      }
      for (auto &J : jacobians) {
        if (J.cols() == nv_lock && J.rows() > 0) {
          J.col(idx).setZero();
        }
      }
    }
  }

  // If no active tasks, return early
  if (goals.empty()) {
    result.status = SolverStatus::kSuccess;
    result.solution.resize(robot_->nv(), 0.0);
    result.joint_velocities = Eigen::VectorXd::Zero(robot_->nv());
    result.limits_applied = false;
    last_solution_dq_norm_ = 0.0;
    return result;
  }

  // Collision constraints are expensive when collision geometry exists.
  std::optional<CollisionConstraintResult> collision_constraint_result =
      std::nullopt;
  if (collision_constraint_.has_value() && collision_constraint_->enabled) {
    if (timing) {
      auto t_collision_start = std::chrono::high_resolution_clock::now();
      collision_constraint_result = compute_collision_constraint();
      result.collision_constraint_time_ms = get_elapsed_ms(t_collision_start);
    } else {
      collision_constraint_result = compute_collision_constraint();
    }
  }
  if (collision_constraint_result.has_value() && !excluded_union.empty()) {
    for (int idx : excluded_union) {
      collision_constraint_result->jacobian.col(idx).setZero();
    }
  }

  // Task normal projection: when a collision constraint is violated
  // (lower_bound > 0), project out the collision normal from each task
  // Jacobian so the task cannot command approach velocity toward the violated
  // boundary. This allows the EE task to drive tangential and away-from-
  // collision motion freely while the constraint handles recovery, preventing
  // the "frozen in all directions" symptom caused by the SNS solver scaling
  // the entire task down to satisfy the inequality.
  if (apply_limits && collision_constraint_result.has_value()) {
    const auto &coll = collision_constraint_result.value();
    Eigen::ArrayXi violated = Eigen::ArrayXi::Zero(coll.jacobian.rows());
    for (int i = 0; i < static_cast<int>(coll.jacobian.rows()); ++i) {
      if (coll.lower_bounds(i) > 0.0) {
        violated(i) = 1;
      }
    }
    project_task_jacobians_away_from_violated_rows(
        jacobians, coll.jacobian, violated,
        /*outward_is_positive_projection=*/false);
  }

  // CoM support-polygon constraint
  std::optional<ComConstraintResult> com_constraint_result = std::nullopt;
  if (com_constraint_.has_value() && com_constraint_->enabled) {
    com_constraint_result = compute_com_constraint();
  }
  if (com_constraint_result.has_value() && !excluded_union.empty()) {
    for (int idx : excluded_union)
      com_constraint_result->jacobian.col(idx).setZero();
  }

  // Task normal projection for violated CoM rows:
  // when outside a half-plane, remove task components that increase outward
  // velocity along that half-plane normal. This mirrors collision behavior and
  // keeps tangential/inward motion available instead of globally scaling tasks.
  if (apply_limits && com_constraint_result.has_value()) {
    const auto &com = com_constraint_result.value();
    project_task_jacobians_away_from_violated_rows(
        jacobians, com.jacobian, com.violated_rows,
        /*outward_is_positive_projection=*/true);
  }

  // Relative pose constraint
  std::optional<RelativePoseConstraintResult> rel_pose_constraint_result =
      std::nullopt;
  if (relative_pose_constraint_.has_value() &&
      relative_pose_constraint_->enabled) {
    rel_pose_constraint_result = compute_relative_pose_constraint();
  }
  if (rel_pose_constraint_result.has_value() && !excluded_union.empty()) {
    for (int idx : excluded_union)
      rel_pose_constraint_result->jacobian.col(idx).setZero();
  }
  if (apply_limits && rel_pose_constraint_result.has_value()) {
    const auto &rpc = rel_pose_constraint_result.value();
    project_task_jacobians_away_from_violated_rows(
        jacobians, rpc.jacobian, rpc.violated_rows,
        /*outward_is_positive_projection=*/true);
  }

  // Build constraint matrix
  std::optional<std::chrono::high_resolution_clock::time_point>
      t_constraint_start;
  if (timing) {
    t_constraint_start = std::chrono::high_resolution_clock::now();
  }
  Eigen::MatrixXd C;
  Eigen::VectorXd c_lower, c_upper;

  // For floating-base robots, we need to handle base and joint constraints
  // separately
  int num_constraints = robot_->nv(); // Velocity constraints

  // Add position-based velocity constraints if enabled
  if (apply_limits && use_position_limits_) {
    num_constraints += robot_->nv(); // Position constraints
  }

  if (collision_constraint_result.has_value()) {
    num_constraints +=
        static_cast<int>(collision_constraint_result->jacobian.rows());
  }

  if (com_constraint_result.has_value()) {
    num_constraints +=
        static_cast<int>(com_constraint_result->jacobian.rows());
  }

  if (rel_pose_constraint_result.has_value()) {
    num_constraints +=
        static_cast<int>(rel_pose_constraint_result->jacobian.rows());
  }

  C = Eigen::MatrixXd::Zero(num_constraints, robot_->nv());
  c_lower = Eigen::VectorXd::Zero(num_constraints);
  c_upper = Eigen::VectorXd::Zero(num_constraints);

  int constraint_idx = 0;

  // Joint velocity constraints
  C.block(constraint_idx, 0, robot_->nv(), robot_->nv()) =
      Eigen::MatrixXd::Identity(robot_->nv(), robot_->nv());

  if (apply_limits && use_velocity_limits_) {
    auto vel_limits = robot_->get_velocity_limits();
    c_lower.segment(constraint_idx, robot_->nv()) = -vel_limits;
    c_upper.segment(constraint_idx, robot_->nv()) = vel_limits;
  } else {
    c_lower.segment(constraint_idx, robot_->nv()).setConstant(-1e10);
    c_upper.segment(constraint_idx, robot_->nv()).setConstant(1e10);
  }
  for (int idx : pending_velocity_lock_indices_) {
    if (idx >= 0 && idx < robot_->nv()) {
      c_lower(constraint_idx + idx) = 0.0;
      c_upper(constraint_idx + idx) = 0.0;
    }
  }
  constraint_idx += robot_->nv();

  // Position-based velocity constraints
  if (apply_limits && use_position_limits_) {
    auto [q_min, q_max] = robot_->get_joint_limits();
    Eigen::VectorXd q_current = robot_->get_current_configuration();
    auto vel_limits = robot_->get_velocity_limits();

    // Get acceleration limits from robot model
    Eigen::VectorXd accel_limits = robot_->get_acceleration_limits();

    // velocity_to_config_index was built earlier (before barrier injection).

    // For each joint, compute maximum velocity to stay within position limits
    C.block(constraint_idx, 0, robot_->nv(), robot_->nv()) =
        Eigen::MatrixXd::Identity(robot_->nv(), robot_->nv());

    // Safety margin for position-based velocity bounds.  The original
    // 1 mrad value causes the velocity bound to reach exactly zero when
    // the joint is within 1 mrad of its limit, which makes the
    // hierarchical (SNS) solver scale the entire task to zero — even
    // when only one DOF is blocked.  A smaller margin keeps a residual
    // velocity that lets the solver find partial solutions while the
    // post-solve clamp (below) still prevents actual limit violations.
    constexpr double margin_limit = 1e-4;

    if (robot_->is_floating_base()) {
      // Handle floating-base constraints (first 6 DoFs)
      // Get current base pose error if bounds are set
      if (base_position_lower_.has_value() ||
          base_position_upper_.has_value() ||
          base_orientation_lower_.has_value() ||
          base_orientation_upper_.has_value()) {

        // Get current base position
        Eigen::Vector3d base_pos = q_current.head<3>();

        // Position constraints (first 3 DoFs)
        for (int i = 0; i < 3; ++i) {
          if (base_position_lower_.has_value() &&
              base_position_upper_.has_value()) {
            double lower_margin =
                base_pos[i] - base_position_lower_.value()[i] - margin_limit;
            double upper_margin =
                base_position_upper_.value()[i] - base_pos[i] - margin_limit;

            auto [lower_limit, upper_limit] = calculate_velocity_box_constraint(
                lower_margin, upper_margin, vel_limits[i], accel_limits[i],
                dt_);

            c_lower(constraint_idx + i) = lower_limit;
            c_upper(constraint_idx + i) = upper_limit;
          } else {
            c_lower(constraint_idx + i) = -1e10;
            c_upper(constraint_idx + i) = 1e10;
          }
        }

        // Orientation constraints (next 3 DoFs)
        for (int i = 3; i < 6; ++i) {
          if (base_orientation_lower_.has_value() &&
              base_orientation_upper_.has_value()) {
            // For orientation, we work directly in velocity space
            c_lower(constraint_idx + i) = std::max(
                base_orientation_lower_.value()[i - 3], -vel_limits[i]);
            c_upper(constraint_idx + i) =
                std::min(base_orientation_upper_.value()[i - 3], vel_limits[i]);
          } else {
            c_lower(constraint_idx + i) = -1e10;
            c_upper(constraint_idx + i) = 1e10;
          }
        }
      } else {
        // No bounds set, use unlimited
        for (int i = 0; i < 6; ++i) {
          c_lower(constraint_idx + i) = -1e10;
          c_upper(constraint_idx + i) = 1e10;
        }
      }

      // Handle joint constraints (remaining DoFs)
      for (int i = 6; i < robot_->nv(); ++i) {
        const int q_idx = velocity_to_config_index[i];
        if (q_idx < 0 || q_idx >= q_current.size() || q_idx >= q_min.size() ||
            q_idx >= q_max.size()) {
          c_lower(constraint_idx + i) = -1e10;
          c_upper(constraint_idx + i) = 1e10;
          continue;
        }

        if (!std::isfinite(q_min[q_idx]) || !std::isfinite(q_max[q_idx])) {
          c_lower(constraint_idx + i) = -1e10;
          c_upper(constraint_idx + i) = 1e10;
          continue;
        }

        // Calculate margins to limits
        double lower_margin = q_current[q_idx] - q_min[q_idx] - margin_limit;
        double upper_margin = q_max[q_idx] - q_current[q_idx] - margin_limit;

        auto [lower_limit, upper_limit] = calculate_velocity_box_constraint(
            lower_margin, upper_margin, vel_limits[i], accel_limits[i], dt_);

        c_lower(constraint_idx + i) = lower_limit;
        c_upper(constraint_idx + i) = upper_limit;
      }
    } else {
      // Fixed-base robot: q-v mapping is only direct for joints with nqs == nvs.
      // Continuous joints (nqs=2, nvs=1) require explicit index mapping.
      for (int i = 0; i < robot_->nv(); ++i) {
        const int q_idx = velocity_to_config_index[i];
        if (q_idx < 0 || q_idx >= q_current.size() || q_idx >= q_min.size() ||
            q_idx >= q_max.size()) {
          c_lower(constraint_idx + i) = -1e10;
          c_upper(constraint_idx + i) = 1e10;
          continue;
        }

        if (!std::isfinite(q_min[q_idx]) || !std::isfinite(q_max[q_idx])) {
          c_lower(constraint_idx + i) = -1e10;
          c_upper(constraint_idx + i) = 1e10;
          continue;
        }

        // Calculate margins to limits
        double lower_margin = q_current[q_idx] - q_min[q_idx] - margin_limit;
        double upper_margin = q_max[q_idx] - q_current[q_idx] - margin_limit;

        auto [lower_limit, upper_limit] = calculate_velocity_box_constraint(
            lower_margin, upper_margin, vel_limits[i], accel_limits[i], dt_);

        c_lower(constraint_idx + i) = lower_limit;
        c_upper(constraint_idx + i) = upper_limit;
      }
    }
    for (int idx : pending_velocity_lock_indices_) {
      if (idx >= 0 && idx < robot_->nv()) {
        c_lower(constraint_idx + idx) = 0.0;
        c_upper(constraint_idx + idx) = 0.0;
      }
    }
    constraint_idx += robot_->nv();
  }

  if (collision_constraint_result.has_value()) {
    const auto &collision = collision_constraint_result.value();
    int rows = static_cast<int>(collision.jacobian.rows());
    C.block(constraint_idx, 0, rows, robot_->nv()) = collision.jacobian;
    c_lower.segment(constraint_idx, rows) = collision.lower_bounds;
    c_upper.segment(constraint_idx, rows) = collision.upper_bounds;
    constraint_idx += rows;
  }

  if (com_constraint_result.has_value()) {
    const auto &com = com_constraint_result.value();
    int rows = static_cast<int>(com.jacobian.rows());
    C.block(constraint_idx, 0, rows, robot_->nv()) = com.jacobian;
    c_lower.segment(constraint_idx, rows) = com.lower_bounds;
    c_upper.segment(constraint_idx, rows) = com.upper_bounds;
    constraint_idx += rows;
  }

  if (rel_pose_constraint_result.has_value()) {
    const auto &rpc = rel_pose_constraint_result.value();
    int rows = static_cast<int>(rpc.jacobian.rows());
    C.block(constraint_idx, 0, rows, robot_->nv()) = rpc.jacobian;
    c_lower.segment(constraint_idx, rows) = rpc.lower_bounds;
    c_upper.segment(constraint_idx, rows) = rpc.upper_bounds;
    constraint_idx += rows;
  }

  if (timing && t_constraint_start.has_value()) {
    result.constraint_setup_time_ms = get_elapsed_ms(*t_constraint_start);
  }

  sanitize_solver_inputs(goals, jacobians, C, c_lower, c_upper);

  // Configure solver
  VelocitySolverConfig config;
  config.epsilon = constraint_tolerance_;
  config.precision_threshold = tight_tolerance_;
  config.iteration_limit = max_iterations_;
  config.magnitude_limit = norm_threshold_;
  config.stall_detection_count = max_zero_scale_iterations_;
  config.regularization_config.epsilon = solver_tolerance_;
  config.regularization_config.regularization_factor = damping_;

  // Call the backend solver
  auto backend_result = computeMultiObjectiveVelocitySolutionEigen(
      goals, jacobians, C, c_lower, c_upper, config, objective_configs);
  const double primary_goal_norm = goals.empty() ? 0.0 : goals[0].norm();

  // Create velocity-specific result
  const auto classified_velocity = classify_velocity_outcome(
      backend_result.status, backend_result.status_message,
      backend_result.task_scales, primary_goal_norm,
      backend_result.task_modes_effective, backend_result.task_used_fallback);
  result.status = classified_velocity.status;
  result.solution = backend_result.solution;
  result.computation_time_ms = backend_result.computation_time_ms;
  result.solver_computation_time_ms = backend_result.computation_time_ms;
  result.iterations = backend_result.iterations;
  result.final_error = backend_result.final_error;
  result.task_scales = backend_result.task_scales;
  result.task_errors = backend_result.task_errors;
  result.task_modes_effective = backend_result.task_modes_effective;
  result.task_used_fallback = backend_result.task_used_fallback;
  result.status_message = classified_velocity.status_message;
  result.limits_applied = apply_limits;

  for (size_t i = 0; i < objective_tasks.size() &&
                     i < result.task_modes_effective.size() &&
                     i < result.task_used_fallback.size();
       ++i) {
    if (objective_tasks[i]) {
      objective_tasks[i]->setLastEffectiveMode(result.task_modes_effective[i]);
      objective_tasks[i]->setUsedMinErrorFallback(result.task_used_fallback[i]);
    }
  }

  // Convert solution to Eigen vector
  if (!result.solution.empty()) {
    // Clamp velocity solution to constraint bounds to avoid post-integration
    // limit violations from QP tolerance.
    if (apply_limits && c_lower.size() >= robot_->nv()) {
      Eigen::Map<Eigen::VectorXd> dq(result.solution.data(),
                                     result.solution.size());
      for (int i = 0; i < robot_->nv(); ++i) {
        double lower = c_lower[i];
        double upper = c_upper[i];
        if (use_position_limits_ &&
            static_cast<int>(c_lower.size()) >= 2 * robot_->nv()) {
          lower = std::max(lower, c_lower[robot_->nv() + i]);
          upper = std::min(upper, c_upper[robot_->nv() + i]);
        }
        dq[i] = std::clamp(dq[i], lower, upper);
      }
    }

    result.joint_velocities = Eigen::Map<const Eigen::VectorXd>(
        result.solution.data(), result.solution.size());
    last_solution_dq_norm_ = result.joint_velocities.norm();

    // Identify saturated joints based on constraint bounds
    if (apply_limits && c_lower.size() >= robot_->nv()) {
      double tolerance = 0.01; // 1% tolerance

      // Use the same combined bounds as post-solve clamping so saturation
      // reporting reflects either velocity-row or position-row activation.
      for (int i = 0; i < robot_->nv(); ++i) {
        double joint_vel = result.joint_velocities[i];
        double lower = c_lower[i];
        double upper = c_upper[i];
        if (use_position_limits_ &&
            static_cast<int>(c_lower.size()) >= 2 * robot_->nv()) {
          lower = std::max(lower, c_lower[robot_->nv() + i]);
          upper = std::min(upper, c_upper[robot_->nv() + i]);
        }

        // Check if joint velocity is near its constraint bounds
        if (joint_vel <= lower + tolerance || joint_vel >= upper - tolerance) {
          result.saturated_joints.push_back(i);
        }
      }
    }
  } else {
    last_solution_dq_norm_ = 0.0;
  }

  // Recovery-state update logic.
  const double dq_norm = last_solution_dq_norm_;
  const bool has_error_status =
      (result.status == SolverStatus::kNumericalError ||
       result.status == SolverStatus::kInfeasible);
  const bool near_zero_dq = dq_norm < kRecoveryNearZeroDqEps;
  const double min_joint_limit_margin = compute_min_joint_limit_margin(*robot_, q_eval);
  const double collision_distance =
      collision_constraint_result.has_value()
          ? collision_constraint_result->distance
          : std::numeric_limits<double>::infinity();
  const bool collision_unhealthy =
      collision_constraint_.has_value() && collision_constraint_->enabled &&
      std::isfinite(collision_distance) &&
      collision_distance < collision_constraint_->min_distance;
  const bool joint_unhealthy =
      std::isfinite(min_joint_limit_margin) &&
      min_joint_limit_margin < kRecoveryJointLimitMarginTrigger;
  const bool unhealthy = collision_unhealthy || joint_unhealthy;

  recovery_state_.error_streak = has_error_status ? (recovery_state_.error_streak + 1) : 0;
  recovery_state_.near_zero_dq_streak =
      near_zero_dq ? (recovery_state_.near_zero_dq_streak + 1) : 0;

  const bool trigger_recovery =
      !recovery_state_.active &&
      recovery_state_.error_streak >= kRecoveryErrorTriggerTicks &&
      recovery_state_.near_zero_dq_streak >= kRecoveryNearZeroTriggerTicks;
  if (trigger_recovery) {
    recovery_state_.active = true;
    recovery_state_.healthy_streak = 0;
    if (!recovery_state_.history.empty()) {
      // Prefer the most recent healthiest sample.
      const auto best_it = std::max_element(
          recovery_state_.history.begin(), recovery_state_.history.end(),
          [](const RecoveryHistoryEntry &a, const RecoveryHistoryEntry &b) {
            const double sa =
                (std::isfinite(a.collision_distance) ? a.collision_distance : -1e9) +
                (std::isfinite(a.joint_limit_margin) ? a.joint_limit_margin : -1e9);
            const double sb =
                (std::isfinite(b.collision_distance) ? b.collision_distance : -1e9) +
                (std::isfinite(b.joint_limit_margin) ? b.joint_limit_margin : -1e9);
            return sa < sb;
          });
      if (best_it != recovery_state_.history.end()) {
        recovery_state_.rollback_target_q = best_it->q;
        recovery_state_.has_rollback_target = true;
      }
    }
    result.status_message +=
        " | recovery mode activated after repeated stalled errors";
  }

  const bool healthy_collision =
      !collision_constraint_.has_value() || !collision_constraint_->enabled ||
      !std::isfinite(collision_distance) ||
      collision_distance >= collision_constraint_->min_distance;
  const bool healthy_limits =
      !std::isfinite(min_joint_limit_margin) ||
      min_joint_limit_margin >= kRecoveryJointLimitMarginHealthy;
  if (recovery_state_.active) {
    if (!has_error_status && !near_zero_dq && healthy_collision && healthy_limits) {
      recovery_state_.healthy_streak += 1;
    } else {
      recovery_state_.healthy_streak = 0;
    }
    if (recovery_state_.healthy_streak >= kRecoveryHealthyExitTicks) {
      recovery_state_.active = false;
      recovery_state_.healthy_streak = 0;
      collision_effective_min_distance_.clear();
      result.status_message += " | recovery mode exited";
    }
  } else {
    collision_effective_min_distance_.clear();
  }

  if (result.status == SolverStatus::kSuccess) {
    RecoveryHistoryEntry entry;
    entry.q = q_eval;
    entry.dq = result.joint_velocities;
    entry.collision_distance = collision_distance;
    entry.joint_limit_margin = min_joint_limit_margin;
    entry.status = result.status;
    recovery_state_.history.push_back(std::move(entry));
    while (recovery_state_.history.size() > kRecoveryHistorySize) {
      recovery_state_.history.pop_front();
    }
  }

  return result;
}

PositionIKResult KinematicsSolver::solve_position(
    const Eigen::VectorXd &seed_q, const Eigen::Matrix4d &target_pose,
    const std::string &frame_name, const PositionIKOptions &options) {

  PositionIKResult result;
  result.status = SolverStatus::kSuccess;
  std::vector<double> position_trace;
  std::vector<double> orientation_trace;
  position_trace.reserve(options.max_iterations);
  orientation_trace.reserve(options.max_iterations);
  double prev_combined_error = std::numeric_limits<double>::infinity();
  int stagnation_iters = 0;

  if (seed_q.size() != robot_->nq()) {
    result.status = SolverStatus::kInvalidInput;
    result.status_message =
        "seed_q size does not match robot nq in solve_position";
    return result;
  }

  auto frame_task = std::make_shared<FrameTask>(
      "position_ik_task", robot_, frame_name, TaskType::FRAME_POSE);
  if (!options.excluded_joint_indices.empty()) {
    frame_task->set_excluded_joint_indices(options.excluded_joint_indices);
  }

  std::shared_ptr<PostureTask> posture_task = nullptr;
  if (options.nullspace_bias.has_value()) {
    if (!options.nullspace_active_joints.empty()) {
      posture_task = std::make_shared<PostureTask>(
          "nullspace_task", robot_, options.nullspace_active_joints);
    } else {
      posture_task = std::make_shared<PostureTask>("nullspace_task", robot_);
    }
    posture_task->setTargetConfiguration(options.nullspace_bias.value());
    posture_task->setWeight(options.nullspace_gain);
    posture_task->setPriority(1);
    posture_task->setSolveMode(TaskSolveMode::kMinError);
    posture_task->setAllowMinErrorFallback(false);
    if (!options.excluded_joint_indices.empty()) {
      posture_task->set_excluded_joint_indices(options.excluded_joint_indices);
    }
  }

  Eigen::VectorXd q_current = seed_q;
  robot_->update_configuration(q_current);

  Eigen::Vector3d target_position = target_pose.block<3, 1>(0, 3);
  Eigen::Matrix3d target_rotation = target_pose.block<3, 3>(0, 0);

  frame_task->setTargetPosition(target_position);
  frame_task->setTargetOrientation(target_rotation);
  frame_task->setWeight(10.0);
  frame_task->setPriority(0);
  frame_task->setSolveMode(options.primary_solve_mode);
  frame_task->setAllowMinErrorFallback(
      options.primary_allow_min_error_fallback);

  int iter = 0;
  bool converged = false;
  bool stagnation_abort = false;

  while (iter < options.max_iterations && !converged) {
    frame_task->update(*robot_);

    Eigen::VectorXd error = frame_task->getError();
    double pos_error = error.head(3).norm();
    double ori_error = error.tail(3).norm();
    position_trace.push_back(pos_error);
    orientation_trace.push_back(ori_error);

    if (position_ik_debug_) {
      std::cout << "[embodiK][IKDebug] iter " << iter
                << " pos_err=" << pos_error << " ori_err=" << ori_error
                << std::endl;
    }

    if (pos_error < options.position_tolerance &&
        ori_error < options.orientation_tolerance) {
      converged = true;
      break;
    }

    double combined_error = pos_error + ori_error;
    if (combined_error < prev_combined_error - options.stagnation_tolerance) {
      prev_combined_error = combined_error;
      stagnation_iters = 0;
    } else {
      stagnation_iters++;
      if (stagnation_iters >= options.stagnation_iterations) {
        if (position_ik_debug_) {
          std::cout << "[embodiK][IKDebug] Stagnation detected after "
                    << stagnation_iters << " iterations; aborting."
                    << std::endl;
        }
        stagnation_abort = true;
        break;
      }
    }

    std::vector<Eigen::VectorXd> goals;
    std::vector<Eigen::MatrixXd> jacobians;
    std::vector<ObjectiveSolveConfig> objective_configs;

    Eigen::VectorXd v_desired = frame_task->getVelocity();
    if (v_desired.size() >= 6) {
      v_desired.head(3) *= options.position_gain;
      v_desired.tail(3) *= options.orientation_gain;
    }

    if (options.max_linear_step > 0 || options.max_angular_step > 0) {
      double linear_vel = v_desired.head(3).norm();
      double angular_vel = v_desired.tail(3).norm();

      double scale = 1.0;
      if (linear_vel > options.max_linear_step / options.dt) {
        scale = std::min(scale,
                         (options.max_linear_step / options.dt) / linear_vel);
      }
      if (angular_vel > options.max_angular_step / options.dt) {
        scale = std::min(scale,
                         (options.max_angular_step / options.dt) / angular_vel);
      }
      v_desired *= scale;
    }

    goals.push_back(v_desired);
    jacobians.push_back(frame_task->getJacobian());
    objective_configs.push_back(
        ObjectiveSolveConfig{0, options.primary_solve_mode,
                             options.primary_allow_min_error_fallback});

    if (posture_task) {
      posture_task->update(*robot_);
      goals.push_back(posture_task->getVelocity());
      jacobians.push_back(posture_task->getJacobian());
      objective_configs.push_back(
          ObjectiveSolveConfig{1, TaskSolveMode::kMinError, false});
    }

    std::optional<CollisionConstraintResult> collision_constraint_result =
        std::nullopt;
    if (collision_constraint_.has_value() && collision_constraint_->enabled) {
      collision_constraint_result = compute_collision_constraint();
    }
    if (collision_constraint_result.has_value() &&
        !options.excluded_joint_indices.empty()) {
      for (int idx : options.excluded_joint_indices) {
        if (idx >= 0 && idx < collision_constraint_result->jacobian.cols()) {
          collision_constraint_result->jacobian.col(idx).setZero();
        }
      }
    }

    int num_constraints = robot_->nv();
    if (collision_constraint_result.has_value()) {
      num_constraints +=
          static_cast<int>(collision_constraint_result->jacobian.rows());
    }

    Eigen::MatrixXd C = Eigen::MatrixXd::Zero(num_constraints, robot_->nv());
    Eigen::VectorXd c_lower =
        Eigen::VectorXd::Constant(num_constraints, -1e10);
    Eigen::VectorXd c_upper =
        Eigen::VectorXd::Constant(num_constraints, 1e10);

    C.block(0, 0, robot_->nv(), robot_->nv()) =
        Eigen::MatrixXd::Identity(robot_->nv(), robot_->nv());

    if (use_position_limits_ || use_velocity_limits_) {
      auto vel_limits = robot_->get_velocity_limits();
      auto accel_limits = robot_->get_acceleration_limits();
      auto [q_min, q_max] = robot_->get_joint_limits();

      for (int i = 0; i < robot_->nv(); ++i) {
        double lower_margin = q_current[i] - q_min[i] - 0.02;
        double upper_margin = q_max[i] - q_current[i] - 0.02;

        auto [lower_limit, upper_limit] = calculate_velocity_box_constraint(
            lower_margin, upper_margin, vel_limits[i], accel_limits[i],
            options.dt);

        c_lower[i] = lower_limit;
        c_upper[i] = upper_limit;
      }
    }

    if (collision_constraint_result.has_value()) {
      int collision_rows =
          static_cast<int>(collision_constraint_result->jacobian.rows());
      int constraint_idx = robot_->nv();
      C.block(constraint_idx, 0, collision_rows, robot_->nv()) =
          collision_constraint_result->jacobian;
      c_lower.segment(constraint_idx, collision_rows) =
          collision_constraint_result->lower_bounds;
      c_upper.segment(constraint_idx, collision_rows) =
          collision_constraint_result->upper_bounds;
    }

    VelocitySolverConfig config;
    config.epsilon = constraint_tolerance_;
    config.precision_threshold = tight_tolerance_;
    config.iteration_limit = max_iterations_;
    config.magnitude_limit = norm_threshold_;
    config.stall_detection_count = max_zero_scale_iterations_;
    config.regularization_config.epsilon = solver_tolerance_;
    config.regularization_config.regularization_factor = damping_;

    auto vel_result = computeMultiObjectiveVelocitySolutionEigen(
        goals, jacobians, C, c_lower, c_upper, config, objective_configs);

    if (vel_result.status != SolverStatus::kSuccess) {
      result.status = vel_result.status;
      result.status_message = vel_result.status_message;
      if (position_ik_debug_) {
        std::cout << "[embodiK][IKDebug] velocity solver failure at iter "
                  << iter << " status=" << static_cast<int>(vel_result.status)
                  << std::endl;
      }
      break;
    }

    Eigen::VectorXd dq = Eigen::Map<const Eigen::VectorXd>(
        vel_result.solution.data(), vel_result.solution.size());
    q_current =
        pinocchio::integrate(robot_->model(), q_current, options.dt * dq);
    robot_->update_configuration(q_current);

    iter++;
  }

  result.q_solution = q_current;
  result.achieved_pose = robot_->get_frame_pose(frame_name);

  frame_task->update(*robot_);
  Eigen::VectorXd final_error = frame_task->getError();
  result.position_error = final_error.head(3).norm();
  result.orientation_error = final_error.tail(3).norm();
  result.iterations_used = iter;
  result.position_error_trace = std::move(position_trace);
  result.orientation_error_trace = std::move(orientation_trace);

  const bool within_tolerance =
      (result.position_error <= options.position_tolerance) &&
      (result.orientation_error <= options.orientation_tolerance);
  const bool max_iterations_reached =
      (!converged) && (iter >= options.max_iterations);
  const auto classified_position = classify_position_outcome(
      result.status, result.status_message, converged || within_tolerance,
      stagnation_abort, max_iterations_reached, result.position_error,
      result.orientation_error);
  result.status = classified_position.status;
  result.status_message = classified_position.status_message;

  if (position_ik_debug_) {
    std::cout << "[embodiK][IKDebug] solve_position finished with status="
              << static_cast<int>(result.status) << " iterations=" << iter
              << " final_pos_err=" << result.position_error
              << " final_ori_err=" << result.orientation_error << std::endl;
  }

  return result;
}

PositionIKResult KinematicsSolver::solve_position_step(
    const Eigen::VectorXd &current_q, const Eigen::Matrix4d &target_pose,
    const std::string &frame_task_name,
    const PositionStepOptions &options) {

  PositionIKResult result;
  const double step_dt = (options.dt > 0.0) ? options.dt : dt_;

  if (current_q.size() != robot_->nq()) {
    result.status = SolverStatus::kInvalidInput;
    result.status_message =
        "current_q size does not match robot nq in solve_position_step";
    return result;
  }

  {
    std::string opt_err;
    if (!validate_position_step_joint_index_options(
            options, static_cast<int>(robot_->nv()), &opt_err)) {
      result.status = SolverStatus::kInvalidInput;
      result.status_message = std::move(opt_err);
      return result;
    }
  }

  auto it = task_map_.find(frame_task_name);
  if (it == task_map_.end() || !it->second) {
    result.status = SolverStatus::kInvalidInput;
    result.status_message = "no task named '" + frame_task_name +
                            "' registered on this solver";
    return result;
  }
  auto frame_task = std::dynamic_pointer_cast<FrameTask>(it->second);
  if (!frame_task) {
    result.status = SolverStatus::kInvalidInput;
    result.status_message = "task '" + frame_task_name +
                            "' is not a FrameTask";
    return result;
  }

  frame_task->setTargetPose(target_pose.block<3, 1>(0, 3),
                            target_pose.block<3, 3>(0, 0));

  Eigen::VectorXd q = current_q;
  robot_->update_configuration(q);

  std::vector<std::shared_ptr<Task>> step_exclusion_targets;
  if (!options.excluded_joint_indices.empty()) {
    step_exclusion_targets = collect_frame_and_posture_tasks_for_step_options(
        frame_task, tasks_);
  }

  VelocitySolverResult last_vel_result;
  bool have_vel_result = false;
  const int steps = std::max(1, options.max_steps);
  Eigen::VectorXd vel(6);

  for (int step = 0; step < steps; ++step) {
    frame_task->update(*robot_);
    const Eigen::VectorXd &error = frame_task->getError();
    vel.head<3>() = options.position_gain * error.head<3>();
    vel.tail<3>() = options.orientation_gain * error.tail<3>();
    clamp_spatial_velocity_components(vel, options.max_linear_speed,
                                      options.max_angular_speed);
    frame_task->setTargetVelocity(vel);

    pending_velocity_lock_indices_ = options.locked_joint_indices;
    VelocitySolverResult vel_out;
    if (!step_exclusion_targets.empty()) {
      ScopedMergedTaskExclusions merge_guard(options.excluded_joint_indices,
                                             step_exclusion_targets);
      vel_out = solve_velocity(q, true);
    } else {
      vel_out = solve_velocity(q, true);
    }
    have_vel_result = true;
    last_vel_result = std::move(vel_out);

    if (last_vel_result.status != SolverStatus::kSuccess &&
        last_vel_result.status != SolverStatus::kInfeasible &&
        last_vel_result.status != SolverStatus::kNumericalError) {
      break;
    }

    if (!options.integration_zero_velocity_indices.empty() &&
        last_vel_result.joint_velocities.size() == robot_->nv()) {
      apply_integration_velocity_mask(
          last_vel_result.joint_velocities,
          options.integration_zero_velocity_indices);
    }

    q = pinocchio::integrate(robot_->model(), q,
                             step_dt * last_vel_result.joint_velocities);
    robot_->update_configuration(q);
  }

  clear_all_target_velocities();

  result.q_solution = q;
  result.achieved_pose = robot_->get_frame_pose(frame_task->getFrameName());
  result.iterations_used = steps;

  frame_task->update(*robot_);
  const Eigen::VectorXd &final_error = frame_task->getError();
  result.position_error = final_error.head<3>().norm();
  result.orientation_error = final_error.tail<3>().norm();

  if (have_vel_result) {
    static_cast<VelocitySolverResult &>(result) = std::move(last_vel_result);
  }

  return result;
}

PositionIKResult KinematicsSolver::solve_position_step(
    const Eigen::VectorXd &current_q, const std::vector<TaskTarget> &targets,
    const PositionStepOptions &options) {

  PositionIKResult result;
  const double step_dt = (options.dt > 0.0) ? options.dt : dt_;

  if (current_q.size() != robot_->nq()) {
    result.status = SolverStatus::kInvalidInput;
    result.status_message =
        "current_q size does not match robot nq in solve_position_step";
    return result;
  }
  if (targets.empty()) {
    result.status = SolverStatus::kInvalidInput;
    result.status_message = "targets must be non-empty in solve_position_step";
    return result;
  }

  {
    std::string opt_err;
    if (!validate_position_step_joint_index_options(
            options, static_cast<int>(robot_->nv()), &opt_err)) {
      result.status = SolverStatus::kInvalidInput;
      result.status_message = std::move(opt_err);
      return result;
    }
  }

  enum class PoseTaskKind { kFrame, kAbsolute, kRelative };

  struct ResolvedTask {
    std::shared_ptr<Task> task;
    PoseTaskKind kind;
  };

  const size_t n_targets = targets.size();
  std::vector<ResolvedTask> resolved;
  resolved.reserve(n_targets);
  for (const auto &target : targets) {
    auto it = task_map_.find(target.task_name);
    if (it == task_map_.end() || !it->second) {
      result.status = SolverStatus::kInvalidInput;
      result.status_message = "no task named '" + target.task_name +
                              "' registered on this solver";
      return result;
    }
    const auto &task = it->second;
    if (std::dynamic_pointer_cast<FrameTask>(task)) {
      resolved.push_back({task, PoseTaskKind::kFrame});
    } else if (std::dynamic_pointer_cast<AbsoluteFrameTask>(task)) {
      resolved.push_back({task, PoseTaskKind::kAbsolute});
    } else if (std::dynamic_pointer_cast<RelativeFrameTask>(task)) {
      resolved.push_back({task, PoseTaskKind::kRelative});
    } else {
      result.status = SolverStatus::kInvalidInput;
      result.status_message = "task '" + target.task_name +
                              "' is not a supported pose task type";
      return result;
    }
  }

  Eigen::VectorXd q = current_q;
  robot_->update_configuration(q);

  std::vector<std::shared_ptr<Task>> pose_only;
  pose_only.reserve(resolved.size());
  for (const auto &rt : resolved) {
    pose_only.push_back(rt.task);
  }
  std::vector<std::shared_ptr<Task>> step_exclusion_targets;
  if (!options.excluded_joint_indices.empty()) {
    step_exclusion_targets =
        collect_multi_pose_and_posture_tasks_for_step_options(pose_only,
                                                                tasks_);
  }

  VelocitySolverResult last_vel_result;
  bool have_vel_result = false;
  const int steps = std::max(1, options.max_steps);
  int steps_used = 0;
  Eigen::Matrix<double, 6, 1> vel;

  for (int step = 0; step < steps; ++step) {
    bool task_apply_failed = false;
    std::string task_apply_error;
    for (size_t i = 0; i < n_targets; ++i) {
      const auto &target = targets[i];
      const auto &rt = resolved[i];

      switch (rt.kind) {
      case PoseTaskKind::kFrame:
        static_cast<FrameTask *>(rt.task.get())
            ->setTargetPose(target.target_pose.block<3, 1>(0, 3),
                            target.target_pose.block<3, 3>(0, 0));
        break;
      case PoseTaskKind::kAbsolute:
        static_cast<AbsoluteFrameTask *>(rt.task.get())
            ->setTargetPose(target.target_pose.block<3, 1>(0, 3),
                            target.target_pose.block<3, 3>(0, 0));
        break;
      case PoseTaskKind::kRelative:
        static_cast<RelativeFrameTask *>(rt.task.get())
            ->setTargetPose(target.target_pose.block<3, 1>(0, 3),
                            target.target_pose.block<3, 3>(0, 0));
        break;
      }

      rt.task->update(*robot_);
      const Eigen::VectorXd &error = rt.task->getError();
      if (error.size() < 6) {
        task_apply_failed = true;
        task_apply_error =
            "task '" + target.task_name + "' has invalid pose error dimension";
        break;
      }
      vel.head<3>() = target.position_gain * error.head<3>();
      vel.tail<3>() = target.orientation_gain * error.tail<3>();
      clamp_spatial_velocity_components(vel, options.max_linear_speed,
                                        options.max_angular_speed);
      rt.task->setTargetVelocity(vel);
    }

    if (task_apply_failed) {
      result.status = SolverStatus::kInvalidInput;
      result.status_message = task_apply_error;
      break;
    }

    pending_velocity_lock_indices_ = options.locked_joint_indices;
    VelocitySolverResult vel_out;
    if (!step_exclusion_targets.empty()) {
      ScopedMergedTaskExclusions merge_guard(options.excluded_joint_indices,
                                             step_exclusion_targets);
      vel_out = solve_velocity(q, true);
    } else {
      vel_out = solve_velocity(q, true);
    }
    have_vel_result = true;
    last_vel_result = std::move(vel_out);
    ++steps_used;

    if (last_vel_result.status != SolverStatus::kSuccess &&
        last_vel_result.status != SolverStatus::kInfeasible &&
        last_vel_result.status != SolverStatus::kNumericalError) {
      break;
    }

    if (!options.integration_zero_velocity_indices.empty() &&
        last_vel_result.joint_velocities.size() == robot_->nv()) {
      apply_integration_velocity_mask(
          last_vel_result.joint_velocities,
          options.integration_zero_velocity_indices);
    }

    q = pinocchio::integrate(robot_->model(), q,
                             step_dt * last_vel_result.joint_velocities);
    robot_->update_configuration(q);
  }

  std::unordered_set<Task *> resolved_tasks;
  resolved_tasks.reserve(resolved.size());
  for (const auto &rt : resolved) {
    resolved_tasks.insert(rt.task.get());
    rt.task->clearTargetVelocity();
  }
  for (auto &task : tasks_) {
    if (task && resolved_tasks.find(task.get()) == resolved_tasks.end()) {
      task->clearTargetVelocity();
    }
  }

  result.q_solution = q;
  result.iterations_used = steps_used;

  const auto &primary = resolved.front();
  primary.task->update(*robot_);
  const Eigen::VectorXd &final_error = primary.task->getError();
  if (final_error.size() >= 6) {
    result.position_error = final_error.head<3>().norm();
    result.orientation_error = final_error.tail<3>().norm();
  }

  if (primary.kind == PoseTaskKind::kFrame) {
    result.achieved_pose = robot_->get_frame_pose(
        static_cast<FrameTask *>(primary.task.get())->getFrameName());
  } else {
    result.achieved_pose = targets.front().target_pose;
  }

  if (have_vel_result) {
    auto saved_status = result.status;
    auto saved_msg = std::move(result.status_message);
    static_cast<VelocitySolverResult &>(result) = std::move(last_vel_result);
    if (saved_status == SolverStatus::kInvalidInput && !saved_msg.empty()) {
      result.status = saved_status;
      result.status_message = std::move(saved_msg);
    }
  } else if (result.status_message.empty()) {
    result.status = SolverStatus::kInvalidInput;
    result.status_message = "no solve step executed in solve_position_step";
  }

  return result;
}

PositionIKResult KinematicsSolver::solve_position_in_tcp(
    const Eigen::VectorXd &seed_q, const Eigen::Matrix4d &relative_target,
    const std::string &frame_name, const PositionIKOptions &options) {

  // Set robot to seed configuration
  robot_->update_configuration(seed_q);

  // Get current TCP pose
  Eigen::Matrix4d current_tcp_pose = robot_->get_frame_pose(frame_name);

  // Calculate target in base frame
  Eigen::Matrix4d target_pose = current_tcp_pose * relative_target;

  // Use regular position IK solver
  return solve_position(seed_q, target_pose, frame_name, options);
}

} // namespace embodik
