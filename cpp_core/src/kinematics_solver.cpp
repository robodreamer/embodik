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
#include <sstream>
#include <string>
#include <pinocchio/algorithm/geometry.hpp>
#include <stdexcept>
#include <unordered_set>
#include <utility>
#ifdef PINOCCHIO_WITH_HPP_FCL
#include <pinocchio/collision/distance.hpp>
#endif
#include <pinocchio/algorithm/joint-configuration.hpp>
#include <pinocchio/spatial/explog.hpp>

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
// Penetration handling and stall-escape thresholds used by position-step paths.
// Treat any measurable negative signed distance as penetration for rejection.
constexpr double kCollisionPenetrationDistanceThreshold = -1e-5;
constexpr double kCollisionPenetrationWorsenTolerance = 5e-4;
constexpr double kCollisionHardPenetrationRejectDistance = -2e-2;
constexpr double kCollisionHardWorsenTolerance = 2e-2;
constexpr double kCollisionEscapeStepMax = 1.5e-2;
constexpr double kCollisionEscapeNormEps = 1e-12;
constexpr double kCollisionEscapeActivationDistance = 0.0;
constexpr double kCollisionEscapeStepMinOnRejection = 1e-4;
constexpr std::array<double, 4> kCollisionRejectionBackoffFractions = {
    0.5, 0.25, 0.1, 0.05};
constexpr double kJointLimitDesaturationMargin = 5e-4;
constexpr double kJointLimitDesaturationStep = 2e-4;
constexpr double kJointLimitDesaturationExpandedMargin = 2e-3;
constexpr double kJointLimitDesaturationBoostStep = 1e-3;
constexpr int kJointLimitDesaturationPlateauThreshold = 3;
// Conservative bound gate constants (Proxima-inspired).
constexpr double kCollisionBoundRotationRadius = 1.5; // meters
constexpr double kCollisionBoundSafetyMargin = 5e-3; // meters
// Post-step rejection: safe margin above penetration threshold for early-exit.
constexpr double kPostStepSafeMargin = 0.01; // 1cm
// Lazy constraint reuse: skip recomputation when dq is tiny and distance is safe.
constexpr double kLazyReuseMaxDqSqNorm = 1e-6;  // ~0.001 rad change
constexpr double kLazyReuseMinDistMargin = 0.005; // 5mm safety margin

// Velocity box constraint: minimum fraction of vel_limit when inside limits
constexpr double kMinBoundFraction = 0.10;
// Margin (rad) below which we do NOT inject headroom toward a limit
constexpr double kMarginThreshold = 0.01;
// Small torso pose-box dead-zones to avoid boundary chatter. When the torso is
// only slightly outside due to numerical noise, treat it as exactly on the
// boundary instead of flipping recovery direction frame-to-frame.
constexpr double kTorsoBoundSlackEpsTrans = 1e-4; // 0.1 mm
constexpr double kTorsoBoundSlackEpsRot = 1e-3;   // ~0.057 deg
// Large finite bound used where a constraint side is intentionally inactive.
constexpr double kUnboundedConstraintLimit = 1e10;
// Elastic band: margin (rad) within which a joint is considered "at limit".
constexpr double kElasticAtLimitMargin = 1e-3; // 1 mrad
// Elastic band: collision warm-start slack below actual clearance (meters).
constexpr double kElasticWarmStartSlack = 0.002; // 2mm

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

static Eigen::Matrix3d unpack_rotation_matrix(
    const std::array<double, 9> &packed) {
  Eigen::Matrix3d R;
  R << packed[0], packed[1], packed[2], packed[3], packed[4], packed[5],
      packed[6], packed[7], packed[8];
  return R;
}

static std::array<double, 9> pack_rotation_matrix(const Eigen::Matrix3d &R) {
  return std::array<double, 9>{R(0, 0), R(0, 1), R(0, 2), R(1, 0), R(1, 1),
                               R(1, 2), R(2, 0), R(2, 1), R(2, 2)};
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

struct ConstraintBlock {
  Eigen::MatrixXd jacobian;
  Eigen::VectorXd lower_bounds;
  Eigen::VectorXd upper_bounds;
};

template <typename IntContainer>
static void zero_excluded_columns(Eigen::MatrixXd &jacobian,
                                  const IntContainer &excluded_indices) {
  for (int idx : excluded_indices) {
    if (idx >= 0 && idx < jacobian.cols()) {
      jacobian.col(idx).setZero();
    }
  }
}

static int append_constraint_block(Eigen::MatrixXd &C, Eigen::VectorXd &c_lower,
                                   Eigen::VectorXd &c_upper, int constraint_idx,
                                   int nv, const Eigen::MatrixXd &jacobian,
                                   const Eigen::VectorXd &lower_bounds,
                                   const Eigen::VectorXd &upper_bounds) {
  const int rows = static_cast<int>(jacobian.rows());
  C.block(constraint_idx, 0, rows, nv) = jacobian;
  c_lower.segment(constraint_idx, rows) = lower_bounds;
  c_upper.segment(constraint_idx, rows) = upper_bounds;
  return constraint_idx + rows;
}

template <typename IntContainer>
static std::optional<ConstraintBlock> build_torso_pose_bound_rows(
    const KinematicsSolver &solver, const RobotModel &robot,
    const TorsoPoseConstraintOptions &torso_opts,
    const pinocchio::SE3 &torso_pose_bounds_reference, double dt,
    const IntContainer &excluded_indices) {
  if (!torso_opts.enabled || !torso_opts.pose_lower_bounds.has_value() ||
      !torso_opts.pose_upper_bounds.has_value() ||
      torso_opts.pose_axis_mask.size() != 6 ||
      torso_opts.velocity_limits.size() != 6 ||
      torso_opts.acceleration_limits.size() != 6 ||
      !robot.has_frame(torso_opts.frame_name)) {
    return std::nullopt;
  }

  const pinocchio::SE3 torso_pose = robot.get_frame_pose(torso_opts.frame_name);
  const Matrix6Xd torso_jacobian_full =
      robot.get_frame_jacobian(torso_opts.frame_name);
  Eigen::VectorXd torso_rel_state = Eigen::VectorXd::Zero(6);
  torso_rel_state.head<3>() =
      torso_pose.translation() - torso_pose_bounds_reference.translation();
  torso_rel_state.tail<3>() = pinocchio::log3(
      torso_pose_bounds_reference.rotation().transpose() * torso_pose.rotation());

  int torso_constraint_rows = 0;
  for (int i = 0; i < 6; ++i) {
    if (torso_opts.pose_axis_mask.coeff(i) > 0.5) {
      ++torso_constraint_rows;
    }
  }
  if (torso_constraint_rows <= 0) {
    return std::nullopt;
  }

  ConstraintBlock result;
  result.jacobian = Eigen::MatrixXd::Zero(torso_constraint_rows, robot.nv());
  result.lower_bounds = Eigen::VectorXd::Constant(
      torso_constraint_rows, -kUnboundedConstraintLimit);
  result.upper_bounds = Eigen::VectorXd::Constant(
      torso_constraint_rows, kUnboundedConstraintLimit);

  int row = 0;
  for (int i = 0; i < 6; ++i) {
    if (torso_opts.pose_axis_mask.coeff(i) <= 0.5) {
      continue;
    }
    double slack_lower =
        torso_rel_state(i) - torso_opts.pose_lower_bounds->coeff(i);
    double slack_upper =
        torso_opts.pose_upper_bounds->coeff(i) - torso_rel_state(i);
    const double torso_slack_eps =
        (i < 3) ? kTorsoBoundSlackEpsTrans : kTorsoBoundSlackEpsRot;
    if (slack_lower < 0.0 && slack_lower >= -torso_slack_eps) {
      slack_lower = 0.0;
    }
    if (slack_upper < 0.0 && slack_upper >= -torso_slack_eps) {
      slack_upper = 0.0;
    }
    const bool torso_headroom_enabled =
        torso_opts.velocity_box_headroom.enabled ||
        torso_opts.pose_bound_softening_enabled;
    const double torso_headroom_fraction =
        torso_opts.velocity_box_headroom.enabled
            ? torso_opts.velocity_box_headroom.fraction
            : torso_opts.pose_bound_softening_fraction;
    const double torso_headroom_activation_margin =
        torso_opts.velocity_box_headroom.enabled
            ? torso_opts.velocity_box_headroom.activation_margin
            : kMarginThreshold;
    auto [lower_limit, upper_limit] = solver.calculate_velocity_box_constraint(
        slack_lower, slack_upper, torso_opts.velocity_limits.coeff(i),
        torso_opts.acceleration_limits.coeff(i), dt,
        torso_headroom_enabled
            ? torso_headroom_fraction * torso_opts.velocity_limits.coeff(i)
            : -1.0,
        torso_headroom_activation_margin);
    result.jacobian.row(row) = torso_jacobian_full.row(i);
    result.lower_bounds(row) = lower_limit;
    result.upper_bounds(row) = upper_limit;
    ++row;
  }

  if (!excluded_indices.empty()) {
    zero_excluded_columns(result.jacobian, excluded_indices);
  }
  return result;
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
      c_lower(r) = -kUnboundedConstraintLimit;
    }
    if (!std::isfinite(c_upper(r))) {
      c_upper(r) = kUnboundedConstraintLimit;
    }
    if (c_lower(r) > c_upper(r)) {
      const double mid = 0.5 * (c_lower(r) + c_upper(r));
      c_lower(r) = mid;
      c_upper(r) = mid;
    }
  }
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

static std::vector<int>
build_step_locked_indices(const PositionStepOptions &options) {
  std::vector<int> merged = options.locked_joint_indices;
  merged.insert(merged.end(), options.excluded_joint_indices.begin(),
                options.excluded_joint_indices.end());
  std::sort(merged.begin(), merged.end());
  merged.erase(std::unique(merged.begin(), merged.end()), merged.end());
  return merged;
}

static void clamp_joint_velocity_solution_in_place(
    Eigen::Ref<Eigen::VectorXd> dq, const Eigen::VectorXd &c_lower,
    const Eigen::VectorXd &c_upper, bool include_position_rows, int nv) {
  if (dq.size() < nv || c_lower.size() < nv || c_upper.size() < nv) {
    return;
  }
  for (int i = 0; i < nv; ++i) {
    double lower = c_lower[i];
    double upper = c_upper[i];
    if (include_position_rows && c_lower.size() >= 2 * nv &&
        c_upper.size() >= 2 * nv) {
      lower = std::max(lower, c_lower[nv + i]);
      upper = std::min(upper, c_upper[nv + i]);
    }
    dq[i] = std::clamp(dq[i], lower, upper);
  }
}

static void tighten_bounds_with_reference_corridor(
    Eigen::VectorXd &c_lower, Eigen::VectorXd &c_upper,
    const Eigen::VectorXd &q_reference, const Eigen::VectorXd &q_current,
    const Eigen::VectorXd &vel_limits, double dt,
    const std::vector<int> &velocity_to_config_index, int nv) {
  if (dt <= 0.0 || c_lower.size() < nv || c_upper.size() < nv ||
      q_reference.size() == 0 || q_current.size() == 0 ||
      vel_limits.size() < nv) {
    return;
  }
  for (int i = 0; i < nv; ++i) {
    int q_idx = i;
    if (!velocity_to_config_index.empty()) {
      if (i >= static_cast<int>(velocity_to_config_index.size())) {
        continue;
      }
      q_idx = velocity_to_config_index[i];
      if (q_idx < 0) {
        continue;
      }
    }
    if (q_idx >= q_reference.size() || q_idx >= q_current.size()) {
      continue;
    }
    const double max_delta = std::max(0.0, vel_limits[i] * dt);
    const double corridor_lower_q = q_reference[q_idx] - max_delta;
    const double corridor_upper_q = q_reference[q_idx] + max_delta;
    const double corridor_lower_v = (corridor_lower_q - q_current[q_idx]) / dt;
    const double corridor_upper_v = (corridor_upper_q - q_current[q_idx]) / dt;
    c_lower[i] = std::max(c_lower[i], corridor_lower_v);
    c_upper[i] = std::min(c_upper[i], corridor_upper_v);
    if (c_lower[i] > c_upper[i]) {
      const double mid = 0.5 * (c_lower[i] + c_upper[i]);
      c_lower[i] = mid;
      c_upper[i] = mid;
    }
  }
}

static ClassifiedOutcome classify_position_outcome(
    SolverStatus current_status, const std::string &current_status_message,
    bool converged_or_within_tolerance, bool stagnation_abort,
    bool classify_stagnation_as_no_progress,
    bool max_iterations_reached, double position_error,
    double orientation_error) {
  if (current_status != SolverStatus::kSuccess) {
    return {current_status, current_status_message};
  }

  if (converged_or_within_tolerance) {
    return {SolverStatus::kSuccess, current_status_message};
  }

  if (stagnation_abort || max_iterations_reached) {
    if (stagnation_abort && classify_stagnation_as_no_progress) {
      return {SolverStatus::kNoProgress,
              "position IK exited due to no progress near active bounds/"
              "constraints: final_position_error=" +
                  std::to_string(position_error) +
                  ", final_orientation_error=" +
                  std::to_string(orientation_error)};
    }
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
  if (config.constraint_activation_multiplier > 0.0) {
    config.constraint_activation_margin =
        config.constraint_activation_multiplier * config.min_distance;
  } else {
    config.constraint_activation_margin = 0.0;
  }
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
    collision_cached_candidate_pair_indices_.clear();
    collision_pair_bound_valid_.assign(num_pairs, 0);
    collision_pair_last_signed_distance_.assign(
        num_pairs, std::numeric_limits<double>::infinity());
    collision_pair_last_rel_translation_norm_.assign(num_pairs, 0.0);
    collision_pair_last_rel_rotation_.assign(
        num_pairs, pack_rotation_matrix(Eigen::Matrix3d::Identity()));
    collision_pair_cache_has_full_scan_ = false;
    collision_pair_cache_steps_since_refresh_ = 0;
    last_collision_pairs_considered_ = 0;
    last_collision_exact_distance_queries_ = 0;
    last_collision_bound_culled_pairs_ = 0;
    last_collision_budget_exhausted_ = false;
    last_constraint_min_distance_ = std::numeric_limits<double>::infinity();
    last_constraint_was_full_scan_ = false;
    last_collision_constraint_result_.reset();
    last_collision_constraint_q_ = Eigen::VectorXd();
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

  auto *collision_model = robot_->collision_model();
  auto *collision_data = robot_->collision_data();
  if (collision_model == nullptr || collision_data == nullptr) {
    return std::nullopt;
  }

  // Fast debug evaluation path: compute the globally closest active pair
  // directly without running full constraint assembly/caching/hysteresis logic.
  pinocchio::updateGeometryPlacements(robot_->model(), robot_->data(),
                                      *collision_model, *collision_data);
  const auto &pairs = collision_model->collisionPairs;
  double best_distance = std::numeric_limits<double>::infinity();
  std::optional<std::size_t> best_index;

  for (std::size_t idx = 0; idx < pairs.size(); ++idx) {
    if (!collision_data->activeCollisionPairs.empty() &&
        !collision_data->activeCollisionPairs[idx]) {
      continue;
    }
    pinocchio::computeDistance(*collision_model, *collision_data, idx);
    const double distance = collision_data->distanceResults[idx].min_distance;
    if (!std::isfinite(distance)) {
      continue;
    }
    if (distance < best_distance) {
      best_distance = distance;
      best_index = idx;
    }
  }

  if (!best_index.has_value()) {
    return std::nullopt;
  }

  const std::size_t idx = *best_index;
  bool restore_nearest_points_flag = false;
  bool previous_nearest_points_flag = false;
  if (idx < collision_data->distanceRequests.size()) {
    auto &req = collision_data->distanceRequests[idx];
    previous_nearest_points_flag = req.enable_nearest_points;
    if (!req.enable_nearest_points) {
      req.enable_nearest_points = true;
      pinocchio::computeDistance(*collision_model, *collision_data, idx);
      restore_nearest_points_flag = true;
    }
  }

  const auto &pair = pairs[idx];
  const auto &obj_a = collision_model->geometryObjects[pair.first];
  const auto &obj_b = collision_model->geometryObjects[pair.second];
  const auto &res = collision_data->distanceResults[idx];
  CollisionDebugInfo debug;
  debug.object_a = obj_a.name;
  debug.object_b = obj_b.name;
  debug.distance = res.min_distance;
  debug.point_a_world = res.nearest_points[0].cast<double>();
  debug.point_b_world = res.nearest_points[1].cast<double>();

  if (restore_nearest_points_flag && idx < collision_data->distanceRequests.size()) {
    collision_data->distanceRequests[idx].enable_nearest_points =
        previous_nearest_points_flag;
  }

  return debug;
#else
  (void)current_q;
  return std::nullopt;
#endif
}

std::optional<double>
KinematicsSolver::evaluate_min_collision_distance(const Eigen::VectorXd &current_q) {
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

  auto *collision_model = robot_->collision_model();
  auto *collision_data = robot_->collision_data();
  if (collision_model == nullptr || collision_data == nullptr) {
    return std::nullopt;
  }

  pinocchio::updateGeometryPlacements(robot_->model(), robot_->data(),
                                      *collision_model, *collision_data);
  const auto &pairs = collision_model->collisionPairs;
  double best_distance = std::numeric_limits<double>::infinity();
  bool found = false;

  for (std::size_t idx = 0; idx < pairs.size(); ++idx) {
    if (!collision_data->activeCollisionPairs.empty() &&
        !collision_data->activeCollisionPairs[idx]) {
      continue;
    }
    pinocchio::computeDistance(*collision_model, *collision_data, idx);
    const double distance = collision_data->distanceResults[idx].min_distance;
    if (!std::isfinite(distance)) {
      continue;
    }
    best_distance = std::min(best_distance, distance);
    found = true;
  }

  if (!found) {
    return std::nullopt;
  }
  return best_distance;
#else
  (void)current_q;
  return std::nullopt;
#endif
}

std::optional<double>
KinematicsSolver::evaluate_min_collision_distance_targeted(
    const Eigen::VectorXd &current_q,
    const std::vector<std::size_t> &pair_indices) {
#ifdef PINOCCHIO_WITH_HPP_FCL
  if (!robot_->has_collision_geometry() || pair_indices.empty()) {
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

  auto *collision_model = robot_->collision_model();
  auto *collision_data = robot_->collision_data();
  if (collision_model == nullptr || collision_data == nullptr) {
    return std::nullopt;
  }

  pinocchio::updateGeometryPlacements(robot_->model(), robot_->data(),
                                      *collision_model, *collision_data);
  double best_distance = std::numeric_limits<double>::infinity();
  bool found = false;

  for (std::size_t idx : pair_indices) {
    if (idx >= collision_model->collisionPairs.size()) continue;
    if (!collision_data->activeCollisionPairs.empty() &&
        !collision_data->activeCollisionPairs[idx]) {
      continue;
    }
    pinocchio::computeDistance(*collision_model, *collision_data, idx);
    const double distance = collision_data->distanceResults[idx].min_distance;
    if (!std::isfinite(distance)) continue;
    best_distance = std::min(best_distance, distance);
    found = true;
  }

  if (!found) return std::nullopt;
  return best_distance;
#else
  (void)current_q;
  (void)pair_indices;
  return std::nullopt;
#endif
}

std::vector<std::size_t>
KinematicsSolver::get_post_step_rejection_pair_indices() const {
  std::unordered_set<std::size_t> unique;
  for (std::size_t idx : last_collision_constraint_pair_indices_) {
    unique.insert(idx);
  }
  for (std::size_t idx : collision_cached_candidate_pair_indices_) {
    unique.insert(idx);
  }
  return std::vector<std::size_t>(unique.begin(), unique.end());
}

// Post-step collision rejection safety margin.  The early-exit gate skips
std::optional<double>
KinematicsSolver::evaluate_post_step_collision_distance(
    const Eigen::VectorXd &q) {
  // Tier 1: Early-exit gate.
  // If the constraint computation already found all pairs well above the
  // penetration threshold, a single bounded integration step cannot create
  // penetration.  Skip the scan entirely.
  // Disabled when budget was exhausted (some pairs may not have been checked).
  if (std::isfinite(last_constraint_min_distance_) &&
      !last_collision_budget_exhausted_ &&
      last_constraint_min_distance_ >=
          kCollisionPenetrationDistanceThreshold + kPostStepSafeMargin) {
    return last_constraint_min_distance_;
  }

  // Tier 2: Targeted scan of known-close pairs.
  // Only check pairs the constraint computation already identified as close
  // (active constraints + cached candidates).
  auto targeted_pairs = get_post_step_rejection_pair_indices();
  if (!targeted_pairs.empty()) {
    return evaluate_min_collision_distance_targeted(q, targeted_pairs);
  }

  // Tier 3: Full scan fallback (no constraint data available).
  return evaluate_min_collision_distance(q);
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

bool KinematicsSolver::set_collision_min_distance(double min_distance) {
  if (!collision_constraint_.has_value() || !collision_constraint_->enabled) {
    return false;
  }
  collision_constraint_->min_distance = std::max(0.0, min_distance);
  if (collision_constraint_->constraint_activation_multiplier > 0.0) {
    collision_constraint_->constraint_activation_margin =
        collision_constraint_->constraint_activation_multiplier *
        collision_constraint_->min_distance;
  } else {
    collision_constraint_->constraint_activation_margin = 0.0;
  }
  return true;
}

void KinematicsSolver::set_proximity_gated_collision_activation_enabled(
    bool enabled) {
  if (!collision_constraint_.has_value()) {
    collision_constraint_.emplace();
  }
  collision_constraint_->constraint_activation_enabled = enabled;
}

bool KinematicsSolver::get_proximity_gated_collision_activation_enabled() const {
  if (!collision_constraint_.has_value()) {
    return false;
  }
  return collision_constraint_->constraint_activation_enabled;
}

void KinematicsSolver::set_collision_constraint_activation_multiplier(
    double multiplier) {
  if (!collision_constraint_.has_value()) {
    collision_constraint_.emplace();
  }
  auto &config = *collision_constraint_;
  config.constraint_activation_multiplier = std::max(0.0, multiplier);
  if (config.constraint_activation_multiplier > 0.0) {
    config.constraint_activation_margin =
        config.constraint_activation_multiplier * std::max(0.0, config.min_distance);
  } else {
    config.constraint_activation_margin = 0.0;
  }
}

double KinematicsSolver::get_collision_constraint_activation_multiplier() const {
  if (!collision_constraint_.has_value()) {
    return 0.0;
  }
  return collision_constraint_->constraint_activation_multiplier;
}

double KinematicsSolver::get_collision_constraint_activation_margin() const {
  if (!collision_constraint_.has_value()) {
    return 0.0;
  }
  return collision_constraint_->constraint_activation_margin;
}

void KinematicsSolver::enable_collision_pair_cache(
    bool enable, int full_refresh_interval, double candidate_distance_margin,
    int max_cached_candidates) {
  collision_pair_cache_enabled_ = enable;
  collision_pair_cache_refresh_interval_ = std::max(1, full_refresh_interval);
  collision_pair_cache_distance_margin_ = std::max(0.0, candidate_distance_margin);
  collision_pair_cache_max_candidates_ = std::max(1, max_cached_candidates);
  collision_pair_cache_has_full_scan_ = false;
  collision_pair_cache_steps_since_refresh_ = 0;
  collision_cached_candidate_pair_indices_.clear();
  std::fill(collision_pair_bound_valid_.begin(), collision_pair_bound_valid_.end(),
            static_cast<std::uint8_t>(0));
  std::fill(collision_pair_last_signed_distance_.begin(),
            collision_pair_last_signed_distance_.end(),
            std::numeric_limits<double>::infinity());
  std::fill(collision_pair_last_rel_translation_norm_.begin(),
            collision_pair_last_rel_translation_norm_.end(), 0.0);
  std::fill(collision_pair_last_rel_rotation_.begin(),
            collision_pair_last_rel_rotation_.end(),
            pack_rotation_matrix(Eigen::Matrix3d::Identity()));
  last_collision_pairs_considered_ = 0;
  last_collision_exact_distance_queries_ = 0;
  last_collision_bound_culled_pairs_ = 0;
  last_collision_budget_exhausted_ = false;
}

void KinematicsSolver::set_collision_refinement_time_budget_us(int budget_us) {
  collision_refinement_time_budget_us_ = std::max(0, budget_us);
}

int KinematicsSolver::get_collision_refinement_time_budget_us() const {
  return collision_refinement_time_budget_us_;
}

void KinematicsSolver::set_collision_tuning_mode(CollisionTuningMode mode) {
  collision_tuning_mode_ = mode;
  switch (mode) {
  case CollisionTuningMode::kPrecise:
    // Exact full-scan behavior: safest/most accurate, highest compute cost.
    //
    // Budget is intentionally disabled (0) so exact distance refinement does not
    // stop early under time pressure.
    // Use legacy-equivalent disabled-cache defaults.
    enable_collision_pair_cache(false, 20, 0.03, 128);
    set_collision_refinement_time_budget_us(0);
    set_proximity_gated_collision_activation_enabled(false);
    set_collision_constraint_activation_multiplier(0.0);
    break;
  case CollisionTuningMode::kBalanced:
    // Conservative compromise:
    // - keep cache enabled to skip obviously far pairs in clear space
    // - but keep budget disabled (0), so once candidate pairs are selected we
    //   avoid early termination and preserve higher collision-distance fidelity
    //   than the speed preset.
    enable_collision_pair_cache(true, 20, 0.05, 256);
    set_collision_refinement_time_budget_us(0);
    set_proximity_gated_collision_activation_enabled(true);
    // Optional proximity-gated activation: rows are emitted only near
    // min_distance, with a conservative activation band.
    set_collision_constraint_activation_multiplier(5.0);
    break;
  case CollisionTuningMode::kSpeed:
  default:
    // Teleop-optimized path.
    // A positive budget intentionally bounds worst-case per-step collision
    // refinement cost, trading some edge-case precision for predictable latency.
    enable_collision_pair_cache(true, 100, 0.03, 128);
    set_collision_refinement_time_budget_us(300);
    set_proximity_gated_collision_activation_enabled(true);
    // Tighter activation band than BALANCED for lower steady-state overhead.
    set_collision_constraint_activation_multiplier(3.0);
    break;
  }
}

CollisionTuningMode KinematicsSolver::get_collision_tuning_mode() const {
  return collision_tuning_mode_;
}

double KinematicsSolver::get_collision_min_distance() const {
  if (!collision_constraint_.has_value() || !collision_constraint_->enabled) {
    return -1.0;
  }
  return collision_constraint_->min_distance;
}

void KinematicsSolver::clear_collision_constraint() {
  if (collision_constraint_.has_value()) {
    collision_constraint_->enabled = false;
  }
  collision_allowed_pair_mask_.clear();
  last_collision_constraint_pair_indices_.clear();
  collision_cached_candidate_pair_indices_.clear();
  collision_pair_bound_valid_.clear();
  collision_pair_last_signed_distance_.clear();
  collision_pair_last_rel_translation_norm_.clear();
  collision_pair_last_rel_rotation_.clear();
  collision_pair_cache_has_full_scan_ = false;
  collision_pair_cache_steps_since_refresh_ = 0;
  last_collision_pairs_considered_ = 0;
  last_collision_exact_distance_queries_ = 0;
  last_collision_bound_culled_pairs_ = 0;
  last_collision_budget_exhausted_ = false;
  collision_stuck_counters_.clear();
  collision_stuck_last_distances_.clear();
}

// ============================================================
// Stall handler
// ============================================================

// Tolerance for floating-point distance comparisons in stall handler logic.
constexpr double kStallDistanceTolerance = 1e-8;

void KinematicsSolver::enable_stall_handler(double nominal_min_distance) {
  const double nom = std::max(0.0, nominal_min_distance);
  if (stall_config_.enabled) {
    if (std::abs(nom - stall_state_.nominal_min_distance) >
        kStallDistanceTolerance) {
      stall_state_.nominal_min_distance = nom;
    }
    return;
  }
  stall_config_.enabled = true;
  stall_state_.nominal_min_distance = nom;
  stall_state_.current_min_distance = nom;
  stall_state_.consecutive_stall_steps = 0;
}

void KinematicsSolver::disable_stall_handler() {
  if (stall_config_.enabled) {
    set_collision_min_distance(stall_state_.nominal_min_distance);
    stall_state_.current_min_distance = stall_state_.nominal_min_distance;
    stall_state_.consecutive_stall_steps = 0;
  }
  stall_config_.enabled = false;
}

bool KinematicsSolver::stall_handler_enabled() const {
  return stall_config_.enabled;
}

void KinematicsSolver::configure_stall_handler(int stall_threshold,
                                                double restore_rate,
                                                double floor_fraction) {
  stall_config_.stall_threshold = std::max(1, stall_threshold);
  stall_config_.restore_rate = std::max(0.0, restore_rate);
  stall_config_.floor_fraction = std::clamp(floor_fraction, 0.0, 1.0);
}

bool KinematicsSolver::stall_handler_is_relaxed() const {
  return stall_config_.enabled &&
         stall_state_.current_min_distance <
             stall_state_.nominal_min_distance - kStallDistanceTolerance;
}

double KinematicsSolver::stall_handler_current_min_distance() const {
  return stall_state_.current_min_distance;
}

int KinematicsSolver::stall_handler_consecutive_stall_steps() const {
  return stall_state_.consecutive_stall_steps;
}

// ============================================================
// Elastic band joint limit expansion
// ============================================================

void KinematicsSolver::enable_elastic_band(double delta_max) {
  if (elastic_band_config_.enabled) {
    elastic_band_config_.delta_max = std::max(0.0, delta_max);
    return;
  }
  elastic_band_config_.enabled = true;
  elastic_band_config_.delta_max = std::max(0.0, delta_max);
  const int nv = robot_->nv();
  elastic_band_state_.delta = Eigen::VectorXd::Zero(nv);
  elastic_band_state_.consecutive_stall_steps = 0;
  elastic_band_state_.total_expansion_steps = 0;

  // Pre-seed expansion for joints that are at or very near their limits.
  // This avoids the initial stall when the seed configuration has joints
  // sitting exactly on a limit boundary (common with zero-config seeds).
  const auto &v2c = velocity_to_config_index_cache();
  auto [q_min, q_max] = robot_->get_joint_limits();
  const Eigen::VectorXd q_cur = robot_->get_current_configuration();
  const double seed_delta = std::min(elastic_band_config_.expand_rate,
                                     elastic_band_config_.delta_max);
  for (int i = 0; i < nv; ++i) {
    const int q_idx =
        (i < static_cast<int>(v2c.size())) ? v2c[i] : kVelocityToConfigUnmapped;
    if (q_idx == kVelocityToConfigUnmapped || q_idx >= q_cur.size()) {
      continue;
    }
    if (!std::isfinite(q_min[q_idx]) || !std::isfinite(q_max[q_idx])) {
      continue;
    }
    const double margin_lo = q_cur[q_idx] - q_min[q_idx];
    const double margin_hi = q_max[q_idx] - q_cur[q_idx];
    if (margin_lo < kElasticAtLimitMargin || margin_hi < kElasticAtLimitMargin) {
      elastic_band_state_.delta[i] = seed_delta;
    }
  }

  // Warm-start collision margin: if the initial config already violates
  // the collision min_distance (but is not penetrating), temporarily relax
  // the collision margin to the actual clearance so the solver isn't stuck
  // from step 0.  The stall handler's restore logic will gradually bring
  // it back to nominal as the robot gains clearance.
  //
  // Only warm-start when the violation is significant enough to actually
  // block the solver (actual clearance < 50% of nominal).  Small violations
  // are handled by the normal stall handler without needing a warm-start.
  if (elastic_band_config_.warm_start_collision_margin &&
      collision_constraint_.has_value() && collision_constraint_->enabled) {
    const double nominal_min = collision_constraint_->min_distance;
    auto actual_min = evaluate_min_collision_distance();
    if (actual_min.has_value() && *actual_min >= 0.0 &&
        *actual_min < nominal_min * 0.5) {
      const double warm_start_min =
          std::max(0.0, *actual_min - kElasticWarmStartSlack);
      set_collision_min_distance(warm_start_min);

      if (!stall_config_.enabled) {
        enable_stall_handler(nominal_min);
      }
      stall_state_.current_min_distance = warm_start_min;
    }
  }
}

void KinematicsSolver::disable_elastic_band() {
  elastic_band_config_.enabled = false;
  elastic_band_state_.delta.setZero();
  elastic_band_state_.consecutive_stall_steps = 0;
}

bool KinematicsSolver::elastic_band_enabled() const {
  return elastic_band_config_.enabled;
}

void KinematicsSolver::configure_elastic_band(double delta_max,
                                               double expand_rate,
                                               double decay_rate,
                                               int stall_threshold,
                                               bool expand_only_saturated) {
  elastic_band_config_.delta_max = std::max(0.0, delta_max);
  elastic_band_config_.expand_rate = std::max(0.0, expand_rate);
  elastic_band_config_.decay_rate = std::clamp(decay_rate, 0.0, 1.0);
  elastic_band_config_.stall_threshold = std::max(1, stall_threshold);
  elastic_band_config_.expand_only_saturated = expand_only_saturated;
}

double KinematicsSolver::elastic_band_max_delta() const {
  if (!elastic_band_config_.enabled ||
      elastic_band_state_.delta.size() == 0) {
    return 0.0;
  }
  return elastic_band_state_.delta.maxCoeff();
}

Eigen::VectorXd KinematicsSolver::elastic_band_deltas() const {
  if (!elastic_band_config_.enabled ||
      elastic_band_state_.delta.size() == 0) {
    return Eigen::VectorXd::Zero(robot_->nv());
  }
  return elastic_band_state_.delta;
}

bool KinematicsSolver::elastic_band_is_expanded() const {
  return elastic_band_config_.enabled &&
         elastic_band_state_.delta.size() > 0 &&
         elastic_band_state_.delta.maxCoeff() > 1e-10;
}

void KinematicsSolver::elastic_band_update(
    const VelocitySolverResult &result) {
  if (!elastic_band_config_.enabled) {
    return;
  }

  const auto &cfg = elastic_band_config_;
  auto &st = elastic_band_state_;
  const int nv = robot_->nv();

  // Ensure delta vector is sized correctly.
  if (st.delta.size() != nv) {
    st.delta = Eigen::VectorXd::Zero(nv);
  }

  // --- Proactive proximity-based expansion ---
  // Instead of waiting for stalls, expand margins proactively when:
  //   1. A joint is saturated (at its velocity limit), AND
  //   2. The task scale is low (solver is struggling)
  // This prevents the stop-start oscillation of reactive expansion.

  const double primary_scale =
      result.task_scales.empty() ? 1.0 : result.task_scales[0];
  const bool has_task_error =
      !result.task_errors.empty() && result.task_errors[0] > 1e-4;
  const bool scale_is_low = primary_scale < 0.5 && has_task_error;

  // Build saturated joint set.
  std::vector<bool> is_saturated(nv, false);
  for (int idx : result.saturated_joints) {
    if (idx >= 0 && idx < nv) {
      is_saturated[idx] = true;
    }
  }

  // Expansion: proportional to how constrained we are.
  // - Scale = 0 (infeasible) → boost to delta_max/2 immediately
  // - Scale near 0 → expand at full rate
  // - Scale near 0.5 → expand at half rate
  // - No saturated joints or scale >= 0.5 → no expansion
  if (scale_is_low && !result.saturated_joints.empty()) {
    const double expansion_factor =
        std::max(0.0, 1.0 - 2.0 * primary_scale);
    const double step_expand = cfg.expand_rate * expansion_factor;
    // When completely infeasible, boost immediately to reduce the number
    // of stalled steps needed to reach useful expansion.
    const double boost_floor =
        (primary_scale < 1e-6) ? cfg.delta_max * 0.5 : 0.0;

    for (int i = 0; i < nv; ++i) {
      if (!cfg.expand_only_saturated || is_saturated[i]) {
        const double new_delta = std::max(st.delta[i] + step_expand, boost_floor);
        st.delta[i] = std::min(new_delta, cfg.delta_max);
      }
    }
    st.total_expansion_steps++;
  }

  // --- Decay: shrink deltas for joints that are NOT saturated ---
  // Only decay joints that have room — saturated joints keep their expansion
  // to avoid the oscillation between expand/decay at the limit boundary.
  for (int i = 0; i < nv; ++i) {
    if (st.delta[i] > 0.0 && !is_saturated[i]) {
      st.delta[i] *= (1.0 - cfg.decay_rate);
      if (st.delta[i] < 1e-8) {
        st.delta[i] = 0.0;
      }
    }
  }

  // Global decay when task is satisfied (no error) — all joints.
  if (!has_task_error) {
    for (int i = 0; i < nv; ++i) {
      st.delta[i] *= (1.0 - cfg.decay_rate);
      if (st.delta[i] < 1e-8) {
        st.delta[i] = 0.0;
      }
    }
  }
}

void KinematicsSolver::stall_handler_update(VelocitySolverResult &result) {
  if (!stall_config_.enabled) {
    return;
  }

  const auto &cfg = stall_config_;
  auto &st = stall_state_;

  const double dq_norm = result.joint_velocities.norm();

  // --- Read active collision-row distances ---
  // Avoid per-step temporary allocations by computing these aggregates directly.
  double collision_dist = std::numeric_limits<double>::infinity();
  bool any_collision_binding = false;
  bool any_penetration = false;
  bool have_collision_distance = false;
  for (const auto &row : last_collision_debug_list_) {
    if (!std::isfinite(row.distance)) {
      continue;
    }
    have_collision_distance = true;
    collision_dist = std::min(collision_dist, row.distance);
    any_collision_binding =
        any_collision_binding || (row.distance <= st.current_min_distance);
    any_penetration = any_penetration || (row.distance < 0.0);
  }
  if (!have_collision_distance && last_collision_debug_.has_value() &&
      std::isfinite(last_collision_debug_->distance)) {
    collision_dist = last_collision_debug_->distance;
    any_collision_binding = (collision_dist <= st.current_min_distance);
    any_penetration = (collision_dist < 0.0);
  }
  // If any joint velocity is saturated at a combined velocity/position bound,
  // treat this as limit-dominated and avoid collision-margin relaxation.
  const bool limit_dominated_stall = !result.saturated_joints.empty();
  const bool near_zero_motion = dq_norm < cfg.dq_stall_eps;
  const bool success_but_blocked =
      (result.status == SolverStatus::kSuccess) && near_zero_motion;
  const bool is_stall =
      ((result.status != SolverStatus::kSuccess) && near_zero_motion) ||
      success_but_blocked;
  const bool solver_healthy =
      (result.status == SolverStatus::kSuccess) && !success_but_blocked;

  // --- Update stall counter ---
  // Keep the counter sticky while collision rows are binding, even if the
  // solver occasionally reports Success, so max_steps=1 teleop loops can
  // accumulate stall evidence across ticks.
  const bool sticky_success_blocked =
      (result.status == SolverStatus::kSuccess) && any_collision_binding;
  if (is_stall) {
    st.consecutive_stall_steps++;
    st.total_stall_steps++;
  } else if (!sticky_success_blocked) {
    st.consecutive_stall_steps = 0;
  }

  const double floor_min = st.nominal_min_distance * cfg.floor_fraction;

  // --- Phase 1: Stall threshold reached → relax collision margin ---
  if (st.consecutive_stall_steps >= cfg.stall_threshold) {
    // Collision is only the bottleneck when the QP row is actually
    // binding: distance at or below the active min_distance.  The
    // proximity band is intentionally NOT used here — it would cause
    // false relaxation when joint-limit clamping (not collision) is the
    // true cause of infeasibility.
    const bool collision_is_bottleneck =
        any_collision_binding && !limit_dominated_stall;

    if (any_penetration && !limit_dominated_stall) {
      // Penetration escape: set margin just below actual penetration depth
      // so the collision QP row has slack and motion can resume.
      const double escape_margin =
          collision_dist - cfg.collision_proximity_band;
      st.current_min_distance = escape_margin;
      set_collision_min_distance(escape_margin);
      st.total_relaxation_steps++;
    } else if (collision_is_bottleneck &&
               st.current_min_distance > floor_min) {
      // Positive-distance stall: ratchet margin down by one drop step.
      const double drop = cfg.relax_drop_fraction * st.nominal_min_distance;
      const double new_min =
          std::max(floor_min, st.current_min_distance - drop);
      st.current_min_distance = new_min;
      set_collision_min_distance(new_min);
      st.total_relaxation_steps++;
    }
    st.consecutive_stall_steps = 0;
    return;
  }

  // --- Phase 2: Solver healthy → restore margin toward nominal ---
  if (solver_healthy &&
      st.current_min_distance <
          st.nominal_min_distance - kStallDistanceTolerance) {
    const double gap = st.nominal_min_distance - st.current_min_distance;
    const double step = std::max(cfg.restore_rate * st.nominal_min_distance,
                                 cfg.relax_drop_fraction * gap);

    // Ceiling: never push margin above actual clearance minus proximity
    // band — that would re-create the infeasible QP.
    double ceiling = st.nominal_min_distance;
    if (std::isfinite(collision_dist)) {
      ceiling = std::min(ceiling,
                         collision_dist - cfg.collision_proximity_band);
    }
    const double new_min =
        std::min(ceiling, st.current_min_distance + step);

    if (new_min > st.current_min_distance) {
      st.current_min_distance = new_min;
      set_collision_min_distance(new_min);
    }
  }
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
  // Lazy reuse: when configuration change is small and we have a safe margin,
  // reuse the previous constraint result.  The constraint Jacobian and bounds
  // remain approximately valid for small dq, and the safety margin absorbs
  // the approximation error.
  // Checked BEFORE resetting debug state so that last_collision_debug_ and
  // last_collision_debug_list_ remain valid for callers (avoids flickering
  // in visualization).
  {
    const bool constraint_active_for_reuse =
        collision_constraint_.has_value() && collision_constraint_->enabled;
    if (constraint_active_for_reuse && collision_pair_cache_enabled_ &&
        last_collision_constraint_result_.has_value() &&
        last_collision_constraint_q_.size() == robot_->nq() &&
        std::isfinite(last_constraint_min_distance_) &&
        !last_collision_budget_exhausted_ &&
        // Respect the cache refresh interval: periodic full recomputation
        // prevents the cached result from going permanently stale.
        collision_pair_cache_steps_since_refresh_ <
            collision_pair_cache_refresh_interval_) {
      const Eigen::VectorXd &q_current = robot_->get_current_configuration();
      const double dq_norm =
          (q_current - last_collision_constraint_q_).squaredNorm();
      if (dq_norm < kLazyReuseMaxDqSqNorm &&
          last_constraint_min_distance_ >
              kCollisionPenetrationDistanceThreshold +
                  kLazyReuseMinDistMargin) {
        // Reuse previous result — configuration barely changed.
        // Preserve last_collision_debug_ and instrumentation counters.
        collision_pair_cache_steps_since_refresh_++;
        last_collision_pairs_considered_ = 0;
        last_collision_exact_distance_queries_ = 0;
        last_collision_bound_culled_pairs_ = 0;
        return last_collision_constraint_result_;
      }
    }
  }

  last_collision_pairs_considered_ = 0;
  last_collision_exact_distance_queries_ = 0;
  last_collision_bound_culled_pairs_ = 0;
  last_collision_budget_exhausted_ = false;
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

  // Fast path: when the cache is warm and previous step confirmed all pairs
  // are well clear of the activation threshold, skip the expensive geometry
  // update entirely and return an empty constraint (no active rows).
  if (constraint_active && collision_pair_cache_enabled_ &&
      collision_pair_cache_has_full_scan_ &&
      !last_collision_budget_exhausted_ &&
      std::isfinite(last_constraint_min_distance_) &&
      collision_constraint_->constraint_activation_enabled &&
      collision_constraint_->constraint_activation_margin > 0.0) {
    const double activation_threshold =
        collision_constraint_->min_distance +
        collision_constraint_->constraint_activation_margin;
    if (last_constraint_min_distance_ > activation_threshold) {
      // All pairs are beyond the activation margin — no constraint rows
      // would be emitted.  Increment the cache step counter and return
      // an empty result, preserving the cached candidate set.
      collision_pair_cache_steps_since_refresh_++;
      last_collision_pairs_considered_ = 0;
      last_collision_exact_distance_queries_ = 0;
      return std::nullopt;
    }
  }

  // Update collision placements and compute distances for all pairs
  pinocchio::updateGeometryPlacements(robot_->model(), robot_->data(),
                                      *collision_model, *collision_data);
  const auto &pairs = collision_model->collisionPairs;

  double best_distance_debug = std::numeric_limits<double>::infinity();
  std::optional<std::size_t> best_index_debug;

  // All allowed pairs with finite distances: (distance, pair_index)
  std::vector<std::pair<double, std::size_t>> allowed_candidates;
  bool use_cached_candidate_subset = false;
  // Keep cache mode active even when the candidate set is currently empty:
  // in clear-space regimes this avoids an expensive full scan on every tick.
  // A full scan is still forced periodically by refresh_interval.
  if (collision_pair_cache_enabled_ && constraint_active &&
      collision_pair_cache_has_full_scan_ &&
      collision_pair_cache_steps_since_refresh_ <
          collision_pair_cache_refresh_interval_) {
    use_cached_candidate_subset = true;
  }

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
    last_collision_exact_distance_queries_++;
    pinocchio::computeDistance(*collision_model, *collision_data, idx);
    req.enable_nearest_points = false;
  };

  std::vector<std::size_t> eval_indices;
  if (use_cached_candidate_subset) {
    std::unordered_set<std::size_t> unique_candidates;
    for (std::size_t idx : collision_cached_candidate_pair_indices_) {
      if (idx < pairs.size()) {
        unique_candidates.insert(idx);
      }
    }
    for (std::size_t idx : last_collision_constraint_pair_indices_) {
      if (idx < pairs.size()) {
        unique_candidates.insert(idx);
      }
    }
    eval_indices.reserve(unique_candidates.size());
    for (std::size_t idx : unique_candidates) {
      eval_indices.push_back(idx);
    }
  } else {
    eval_indices.reserve(pairs.size());
    for (std::size_t idx = 0; idx < pairs.size(); ++idx) {
      eval_indices.push_back(idx);
    }
  }
  if (use_cached_candidate_subset && constraint_active &&
      collision_constraint_.has_value() &&
      collision_constraint_->max_constraints > 1 &&
      !last_collision_constraint_pair_indices_.empty() &&
      eval_indices.size() <
          static_cast<std::size_t>(collision_constraint_->max_constraints)) {
    // Safety fallback: near-contact cached subsets can become underfilled
    // (e.g. only one historical pair left while max_constraints>1), which can
    // miss newly critical pairs for a whole tick.  Only force a full scan
    // when near penetration; in clear space the underfilled cache is harmless
    // and a full scan wastes the tuning mode's latency budget.
    const bool underfill_near_penetration =
        !std::isfinite(last_constraint_min_distance_) ||
        last_collision_budget_exhausted_ ||
        last_constraint_min_distance_ <
            kCollisionPenetrationDistanceThreshold +
                kPostStepSafeMargin;
    if (underfill_near_penetration) {
      use_cached_candidate_subset = false;
      eval_indices.clear();
      eval_indices.reserve(pairs.size());
      for (std::size_t idx = 0; idx < pairs.size(); ++idx) {
        eval_indices.push_back(idx);
      }
    }
  }
  if (use_cached_candidate_subset && eval_indices.empty()) {
    // Safety fallback: an empty cache creates a collision-blind step where no
    // distance queries run. Fall back to a full scan immediately.
    use_cached_candidate_subset = false;
    eval_indices.reserve(pairs.size());
    for (std::size_t idx = 0; idx < pairs.size(); ++idx) {
      eval_indices.push_back(idx);
    }
  }

  const bool budget_enabled = constraint_active &&
                              (collision_refinement_time_budget_us_ > 0) &&
                              use_cached_candidate_subset;
  if (budget_enabled) {
    std::sort(eval_indices.begin(), eval_indices.end(),
              [&](std::size_t a, std::size_t b) {
                const bool a_valid =
                    (a < collision_pair_bound_valid_.size()) &&
                    collision_pair_bound_valid_[a];
                const bool b_valid =
                    (b < collision_pair_bound_valid_.size()) &&
                    collision_pair_bound_valid_[b];
                if (a_valid != b_valid) {
                  return a_valid > b_valid;
                }
                const double da =
                    (a < collision_pair_last_signed_distance_.size())
                        ? collision_pair_last_signed_distance_[a]
                        : std::numeric_limits<double>::infinity();
                const double db =
                    (b < collision_pair_last_signed_distance_.size())
                        ? collision_pair_last_signed_distance_[b]
                        : std::numeric_limits<double>::infinity();
                if (da != db) {
                  return da < db;
                }
                return a < b;
              });
  }
  const auto budget_start = std::chrono::high_resolution_clock::now();

  for (std::size_t idx : eval_indices) {
    if (budget_enabled) {
      const auto elapsed_us = std::chrono::duration_cast<std::chrono::microseconds>(
                                  std::chrono::high_resolution_clock::now() -
                                  budget_start)
                                  .count();
      if (elapsed_us >= collision_refinement_time_budget_us_) {
        last_collision_budget_exhausted_ = true;
        break;
      }
    }

    // Honor Pinocchio's active-pair mask *before* distance computation.
    if (!collision_data->activeCollisionPairs.empty() &&
        !collision_data->activeCollisionPairs[idx]) {
      continue;
    }
    last_collision_pairs_considered_++;

    const auto &pair = pairs[idx];
    const auto &object_a = collision_model->geometryObjects[pair.first];
    const auto &object_b = collision_model->geometryObjects[pair.second];
    const auto frame_a_id = object_a.parentFrame;
    const auto frame_b_id = object_b.parentFrame;
    const auto &transform_a = robot_->data().oMf[frame_a_id];
    const auto &transform_b = robot_->data().oMf[frame_b_id];
    const double rel_translation_norm =
        (transform_a.translation() - transform_b.translation()).norm();
    const Eigen::Matrix3d rel_rotation =
        transform_a.rotation().transpose() * transform_b.rotation();

    bool bound_culled = false;
    if (constraint_active && use_cached_candidate_subset &&
        idx < collision_pair_bound_valid_.size() &&
        idx < collision_pair_last_signed_distance_.size() &&
        idx < collision_pair_last_rel_translation_norm_.size() &&
        idx < collision_pair_last_rel_rotation_.size() &&
        collision_pair_bound_valid_[idx]) {
      const double prev_distance = collision_pair_last_signed_distance_[idx];
      const double prev_rel_translation_norm =
          collision_pair_last_rel_translation_norm_[idx];
      const Eigen::Matrix3d prev_rel_rotation =
          unpack_rotation_matrix(collision_pair_last_rel_rotation_[idx]);
      const double delta_m =
          std::abs(rel_translation_norm - prev_rel_translation_norm);
      const Eigen::Matrix3d delta_rel_rotation =
          prev_rel_rotation.transpose() * rel_rotation;
      const double cos_theta = std::clamp(
          (delta_rel_rotation.trace() - 1.0) * 0.5, -1.0, 1.0);
      const double rotation_penalty = std::sqrt(
          2.0 * kCollisionBoundRotationRadius * kCollisionBoundRotationRadius *
          std::max(0.0, 1.0 - cos_theta));
      const double lower_bound = prev_distance - delta_m - rotation_penalty -
                                 kCollisionBoundSafetyMargin;
      const double candidate_cutoff =
          collision_constraint_.has_value()
              ? collision_constraint_->min_distance +
                    collision_pair_cache_distance_margin_
              : std::numeric_limits<double>::infinity();
      if (std::isfinite(lower_bound) && lower_bound > candidate_cutoff) {
        bound_culled = true;
      }
    }
    if (bound_culled) {
      last_collision_bound_culled_pairs_++;
      continue;
    }

    // When evaluating a small cached subset WITHOUT a time budget, enable
    // nearest points on the initial query to avoid a costly re-query later
    // for constraint pairs.  Skip this when budget is active — the extra
    // cost would exhaust the budget and trigger safety fallbacks.
    if (use_cached_candidate_subset && !budget_enabled &&
        !nearest_points_all_pairs &&
        idx < collision_data->distanceRequests.size() &&
        !collision_data->distanceRequests[idx].enable_nearest_points) {
      collision_data->distanceRequests[idx].enable_nearest_points = true;
    }
    last_collision_exact_distance_queries_++;
    pinocchio::computeDistance(*collision_model, *collision_data, idx);

    const auto &distance_result = collision_data->distanceResults[idx];
    double distance = distance_result.min_distance;
    if (!std::isfinite(distance)) {
      continue;
    }

    if (idx < collision_pair_bound_valid_.size() &&
        idx < collision_pair_last_signed_distance_.size() &&
        idx < collision_pair_last_rel_translation_norm_.size() &&
        idx < collision_pair_last_rel_rotation_.size()) {
      collision_pair_bound_valid_[idx] = 1;
      collision_pair_last_signed_distance_[idx] = distance;
      collision_pair_last_rel_translation_norm_[idx] = rel_translation_norm;
      collision_pair_last_rel_rotation_[idx] = pack_rotation_matrix(rel_rotation);
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
  if (use_cached_candidate_subset) {
    collision_pair_cache_steps_since_refresh_++;
  } else {
    collision_pair_cache_has_full_scan_ = true;
    collision_pair_cache_steps_since_refresh_ = 0;
  }

  // Expose global minimum distance for the post-step rejection fast path.
  last_constraint_min_distance_ = best_distance_debug;
  last_constraint_was_full_scan_ = !use_cached_candidate_subset;

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

  // Safety: any penetrating pair (distance < 0) MUST be in the active set,
  // even if max_constraints would normally exclude it.  Without this, the
  // QP may miss a pair that transitions from out-of-cache to penetrating.
  if (best_index_debug.has_value() && best_distance_debug < 0.0) {
    bool found = false;
    for (const auto &entry : selected_sorted) {
      if (entry.second == *best_index_debug) { found = true; break; }
    }
    if (!found) {
      selected_sorted.emplace_back(best_distance_debug, *best_index_debug);
      std::sort(selected_sorted.begin(), selected_sorted.end());
    }
  }

  if (static_cast<int>(selected_sorted.size()) > max_k) {
    selected_sorted.resize(static_cast<std::size_t>(max_k));
  }

  // Optional proximity-gated row activation.
  // When disabled (margin <= 0), behavior is unchanged.
  const auto &config = *collision_constraint_;
  if (config.constraint_activation_enabled &&
      config.constraint_activation_margin > 0.0) {
    const double activation_threshold =
        config.min_distance + config.constraint_activation_margin;
    selected_sorted.erase(
        std::remove_if(
            selected_sorted.begin(), selected_sorted.end(),
            [activation_threshold](const std::pair<double, std::size_t> &entry) {
              return entry.first > activation_threshold;
            }),
        selected_sorted.end());
  }

  // Update the per-step active indices for next step's hysteresis.
  last_collision_constraint_pair_indices_.clear();
  for (const auto &[dist, idx] : selected_sorted) {
    last_collision_constraint_pair_indices_.push_back(idx);
  }

  // Update conservative candidate cache for next step.
  collision_cached_candidate_pair_indices_.clear();
  if (constraint_active) {
    const auto &config = *collision_constraint_;
    const double candidate_cutoff =
        config.min_distance + collision_pair_cache_distance_margin_;
    for (const auto &[dist, idx] : allowed_candidates) {
      if (dist <= candidate_cutoff) {
        collision_cached_candidate_pair_indices_.push_back(idx);
        if (static_cast<int>(collision_cached_candidate_pair_indices_.size()) >=
            collision_pair_cache_max_candidates_) {
          break;
        }
      }
    }
    std::unordered_set<std::size_t> candidate_set(
        collision_cached_candidate_pair_indices_.begin(),
        collision_cached_candidate_pair_indices_.end());
    for (std::size_t idx : last_collision_constraint_pair_indices_) {
      if (static_cast<int>(collision_cached_candidate_pair_indices_.size()) >=
          collision_pair_cache_max_candidates_) {
        break;
      }
      if (candidate_set.insert(idx).second) {
        collision_cached_candidate_pair_indices_.push_back(idx);
      }
    }
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
  // Always report the globally closest pair so that evaluate_collision_debug()
  // returns the true minimum distance even when the pair is not in the active
  // constraint set (top-K).
  std::optional<std::size_t> debug_index_to_use = best_index_debug;

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
    const double effective_min_distance = target_min_distance;

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

  // Cache the result and configuration for lazy reuse.
  last_collision_constraint_result_ = result;
  last_collision_constraint_q_ = robot_->get_current_configuration();

  return result;
#else
  last_collision_debug_.reset();
  last_collision_debug_list_.clear();
  return std::nullopt;
#endif
}

std::pair<double, double> KinematicsSolver::calculate_velocity_box_constraint(
    double position_margin_lower, double position_margin_upper,
    double velocity_limit, double acceleration_limit, double dt,
    double min_velocity_headroom, double headroom_activation_margin) const {
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

  // Shared velocity-box headroom policy:
  // - explicit per-call headroom (min_velocity_headroom >= 0), or
  // - legacy solver-level saturation-exit behavior.
  if (!outside_lower && !outside_upper) {
    bool apply_headroom = false;
    double min_vel = 0.0;
    double activation_margin = std::max(0.0, headroom_activation_margin);
    if (min_velocity_headroom >= 0.0) {
      apply_headroom = true;
      min_vel = std::max(0.0, min_velocity_headroom);
    } else if (enable_saturation_exit_behavior_) {
      apply_headroom = true;
      min_vel = kMinBoundFraction * velocity_limit;
      activation_margin = kMarginThreshold;
    }
    if (apply_headroom && min_vel > 0.0) {
      if (lower_limit > -min_vel && raw_margin_lower > activation_margin) {
        lower_limit = -min_vel;
      }
      if (upper_limit < min_vel && raw_margin_upper > activation_margin) {
        upper_limit = min_vel;
      }
    }
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

const std::vector<int> &KinematicsSolver::velocity_to_config_index_cache() {
  const int nv = robot_->nv();
  const RobotModel *model = robot_.get();
  if (velocity_to_config_cache_robot_ == model &&
      velocity_to_config_cache_nv_ == nv &&
      static_cast<int>(velocity_to_config_index_cache_.size()) == nv) {
    return velocity_to_config_index_cache_;
  }
  velocity_to_config_index_cache_.assign(nv, kVelocityToConfigUnmapped);
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
      if (vk >= 0 && vk < nv && qk >= 0 && qk < robot_->nq()) {
        velocity_to_config_index_cache_[vk] = qk;
      }
    }
  }
  velocity_to_config_cache_robot_ = model;
  velocity_to_config_cache_nv_ = nv;
  return velocity_to_config_index_cache_;
}

void KinematicsSolver::clamp_jacobians_near_joint_limits(
    std::vector<Eigen::MatrixXd> &jacobians,
    const std::vector<int> &velocity_to_config_index) const {
  constexpr double kJointLimitClampMargin = 5e-4;
  const int nv_clamp = robot_->nv();
  auto [q_min_clamp, q_max_clamp] = robot_->get_joint_limits();
  const Eigen::VectorXd q_cur_clamp = robot_->get_current_configuration();
  const int fb_offset = robot_->is_floating_base() ? 6 : 0;

  for (int i = fb_offset; i < nv_clamp; ++i) {
    const int q_idx = (i < static_cast<int>(velocity_to_config_index.size()))
                          ? velocity_to_config_index[i]
                          : kVelocityToConfigUnmapped;
    if (q_idx == kVelocityToConfigUnmapped || q_idx >= q_cur_clamp.size() ||
        q_idx >= q_min_clamp.size() || q_idx >= q_max_clamp.size()) {
      continue;
    }
    if (!std::isfinite(q_min_clamp[q_idx]) ||
        !std::isfinite(q_max_clamp[q_idx])) {
      continue;
    }

    // Skip Jacobian clamping for joints with active elastic band expansion.
    // The elastic zone explicitly allows motion near/beyond nominal limits.
    if (elastic_band_config_.enabled &&
        i < elastic_band_state_.delta.size() &&
        elastic_band_state_.delta[i] > kJointLimitClampMargin) {
      continue;
    }

    const double margin_lower = q_cur_clamp[q_idx] - q_min_clamp[q_idx];
    const double margin_upper = q_max_clamp[q_idx] - q_cur_clamp[q_idx];
    const bool clamp_lower =
        margin_lower > 0.0 && margin_lower < kJointLimitClampMargin;
    const bool clamp_upper =
        margin_upper > 0.0 && margin_upper < kJointLimitClampMargin;
    if (!clamp_lower && !clamp_upper) {
      continue;
    }

    for (auto &J : jacobians) {
      if (J.cols() != nv_clamp || J.rows() <= 0) {
        continue;
      }
      for (int r = 0; r < static_cast<int>(J.rows()); ++r) {
        double &val = J(r, i);
        if (clamp_lower && val < 0.0) {
          val = 0.0;
        }
        if (clamp_upper && val > 0.0) {
          val = 0.0;
        }
      }
    }
  }
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
                                 bool apply_limits,
                                 bool stall_recovery) {
  if (stall_recovery && !stall_config_.enabled) {
    const double nominal =
        get_collision_min_distance() > 0.0 ? get_collision_min_distance() : 0.0;
    enable_stall_handler(nominal);
  }
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
        solver->pending_step_torso_constraint_.reset();
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

  // Auto-enable elastic band when any task uses SCALE_ELASTIC mode.
  {
    bool any_scale_elastic = false;
    for (const auto &task : tasks_) {
      if (task && task->isActive() &&
          task->getSolveMode() == TaskSolveMode::kScaleElastic) {
        any_scale_elastic = true;
        break;
      }
    }
    if (any_scale_elastic && !elastic_band_config_.enabled) {
      enable_elastic_band(0.05);
      configure_elastic_band(/*delta_max=*/0.05, /*expand_rate=*/0.01,
                             /*decay_rate=*/0.2, /*stall_threshold=*/3,
                             /*expand_only_saturated=*/true);
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
  goals.reserve(tasks_.size());
  jacobians.reserve(tasks_.size());
  objective_configs.reserve(tasks_.size());
  objective_tasks.reserve(tasks_.size());
  excluded_union.reserve(tasks_.size());

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
      const auto mode = task->getSolveMode();
      if ((mode != TaskSolveMode::kScale &&
           mode != TaskSolveMode::kScaleElastic) ||
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
        // SCALE_ELASTIC is treated as SCALE in the SNS solver;
        // the elastic band mechanism handles the limit expansion.
        const auto effective_mode =
            (task->getSolveMode() == TaskSolveMode::kScaleElastic)
                ? TaskSolveMode::kScale
                : task->getSolveMode();
        objective_configs.push_back(ObjectiveSolveConfig{
            task->getPriority(),
            effective_mode,
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


  // Velocity-to-configuration index mapping (cached; used by
  // position-based velocity constraints).
  const std::vector<int> &velocity_to_config_index =
      velocity_to_config_index_cache();

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

  // Near a joint position limit, zero Jacobian entries that would command
  // motion further into that limit. This avoids whole-task SNS scale collapse
  // and preserves partial solutions through remaining DOFs.
  if (apply_limits && use_position_limits_) {
    clamp_jacobians_near_joint_limits(jacobians, velocity_to_config_index);
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
    result.collision_pairs_considered = last_collision_pairs_considered_;
    result.collision_exact_distance_queries =
        last_collision_exact_distance_queries_;
    result.collision_bound_culled_pairs = last_collision_bound_culled_pairs_;
    result.collision_budget_exhausted = last_collision_budget_exhausted_;
  }
  if (collision_constraint_result.has_value() && !excluded_union.empty()) {
    zero_excluded_columns(collision_constraint_result->jacobian, excluded_union);
  }

  // Task normal projection: when a collision constraint is violated
  // (lower_bound > 0), project out the collision normal from each task
  // Jacobian so the task cannot command approach velocity toward the violated
  // boundary. This allows the EE task to drive tangential and away-from-
  // collision motion freely while the constraint handles recovery, preventing
  // the "frozen in all directions" symptom caused by the SNS solver scaling
  // the entire task down to satisfy the inequality.
  int collision_violated_rows = 0;
  if (apply_limits && collision_constraint_result.has_value()) {
    const auto &coll = collision_constraint_result.value();
    Eigen::ArrayXi violated = Eigen::ArrayXi::Zero(coll.jacobian.rows());
    for (int i = 0; i < static_cast<int>(coll.jacobian.rows()); ++i) {
      if (coll.lower_bounds(i) > 0.0) {
        violated(i) = 1;
        collision_violated_rows++;
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
    zero_excluded_columns(com_constraint_result->jacobian, excluded_union);
  }

  // Task normal projection for violated CoM rows:
  // when outside a half-plane, remove task components that increase outward
  // velocity along that half-plane normal. This mirrors collision behavior and
  // keeps tangential/inward motion available instead of globally scaling tasks.
  int com_violated_rows = 0;
  if (apply_limits && com_constraint_result.has_value()) {
    const auto &com = com_constraint_result.value();
    for (int i = 0; i < com.violated_rows.size(); ++i) {
      if (com.violated_rows(i) != 0) {
        com_violated_rows++;
      }
    }
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
    zero_excluded_columns(rel_pose_constraint_result->jacobian, excluded_union);
  }
  int rel_pose_violated_rows = 0;
  if (apply_limits && rel_pose_constraint_result.has_value()) {
    const auto &rpc = rel_pose_constraint_result.value();
    for (int i = 0; i < rpc.violated_rows.size(); ++i) {
      if (rpc.violated_rows(i) != 0) {
        rel_pose_violated_rows++;
      }
    }
    project_task_jacobians_away_from_violated_rows(
        jacobians, rpc.jacobian, rpc.violated_rows,
        /*outward_is_positive_projection=*/true);
  }

  std::optional<ConstraintBlock> step_torso_constraint_result = std::nullopt;
  if (pending_step_torso_constraint_.has_value()) {
    const auto &torso_opts = *pending_step_torso_constraint_;
    if (torso_opts.enabled && robot_->has_frame(torso_opts.frame_name)) {
      pinocchio::SE3 torso_pose_bounds_reference = pinocchio::SE3::Identity();
      if (torso_opts.pose_bounds_reference_pose.has_value()) {
        const Eigen::Matrix4d &M = torso_opts.pose_bounds_reference_pose.value();
        torso_pose_bounds_reference =
            pinocchio::SE3(M.block<3, 3>(0, 0), M.block<3, 1>(0, 3));
      } else {
        torso_pose_bounds_reference = robot_->get_frame_pose(torso_opts.frame_name);
      }
      step_torso_constraint_result = build_torso_pose_bound_rows(
          *this, *robot_, torso_opts, torso_pose_bounds_reference, dt_,
          excluded_union);
    }
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
  if (step_torso_constraint_result.has_value()) {
    num_constraints +=
        static_cast<int>(step_torso_constraint_result->jacobian.rows());
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
    c_lower.segment(constraint_idx, robot_->nv())
        .setConstant(-kUnboundedConstraintLimit);
    c_upper.segment(constraint_idx, robot_->nv())
        .setConstant(kUnboundedConstraintLimit);
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

    // velocity_to_config_index was built earlier.

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

    // Elastic band: compute effective limits with expansion.
    Eigen::VectorXd q_min_eff = q_min;
    Eigen::VectorXd q_max_eff = q_max;
    const bool elastic_active = elastic_band_config_.enabled &&
                                elastic_band_state_.delta.size() == robot_->nv();
    if (elastic_active) {
      for (int i = 0; i < robot_->nv(); ++i) {
        const int q_idx =
            (i < static_cast<int>(velocity_to_config_index.size()))
                ? velocity_to_config_index[i]
                : kVelocityToConfigUnmapped;
        if (q_idx == kVelocityToConfigUnmapped || q_idx >= q_min.size()) {
          continue;
        }
        if (!std::isfinite(q_min[q_idx]) || !std::isfinite(q_max[q_idx])) {
          continue;
        }
        q_min_eff[q_idx] -= elastic_band_state_.delta[i];
        q_max_eff[q_idx] += elastic_band_state_.delta[i];
      }
    }

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
            c_lower(constraint_idx + i) = -kUnboundedConstraintLimit;
            c_upper(constraint_idx + i) = kUnboundedConstraintLimit;
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
            c_lower(constraint_idx + i) = -kUnboundedConstraintLimit;
            c_upper(constraint_idx + i) = kUnboundedConstraintLimit;
          }
        }
      } else {
        // No bounds set, use unlimited
        for (int i = 0; i < 6; ++i) {
          c_lower(constraint_idx + i) = -kUnboundedConstraintLimit;
          c_upper(constraint_idx + i) = kUnboundedConstraintLimit;
        }
      }

      // Handle joint constraints (remaining DoFs)
      for (int i = 6; i < robot_->nv(); ++i) {
        const int q_idx = velocity_to_config_index[i];
        if (q_idx == kVelocityToConfigUnmapped ||
            q_idx >= q_current.size() || q_idx >= q_min.size() ||
            q_idx >= q_max.size()) {
          c_lower(constraint_idx + i) = -kUnboundedConstraintLimit;
          c_upper(constraint_idx + i) = kUnboundedConstraintLimit;
          continue;
        }

        if (!std::isfinite(q_min[q_idx]) || !std::isfinite(q_max[q_idx])) {
          c_lower(constraint_idx + i) = -kUnboundedConstraintLimit;
          c_upper(constraint_idx + i) = kUnboundedConstraintLimit;
          continue;
        }

        // Calculate margins to limits (elastic band expands effective limits)
        double lower_margin = q_current[q_idx] - q_min_eff[q_idx] - margin_limit;
        double upper_margin = q_max_eff[q_idx] - q_current[q_idx] - margin_limit;

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
        if (q_idx == kVelocityToConfigUnmapped ||
            q_idx >= q_current.size() || q_idx >= q_min.size() ||
            q_idx >= q_max.size()) {
          c_lower(constraint_idx + i) = -kUnboundedConstraintLimit;
          c_upper(constraint_idx + i) = kUnboundedConstraintLimit;
          continue;
        }

        if (!std::isfinite(q_min[q_idx]) || !std::isfinite(q_max[q_idx])) {
          c_lower(constraint_idx + i) = -kUnboundedConstraintLimit;
          c_upper(constraint_idx + i) = kUnboundedConstraintLimit;
          continue;
        }

        // Calculate margins to limits (elastic band expands effective limits)
        double lower_margin = q_current[q_idx] - q_min_eff[q_idx] - margin_limit;
        double upper_margin = q_max_eff[q_idx] - q_current[q_idx] - margin_limit;

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
    constraint_idx = append_constraint_block(
        C, c_lower, c_upper, constraint_idx, robot_->nv(),
        collision_constraint_result->jacobian,
        collision_constraint_result->lower_bounds,
        collision_constraint_result->upper_bounds);
  }

  if (com_constraint_result.has_value()) {
    constraint_idx = append_constraint_block(
        C, c_lower, c_upper, constraint_idx, robot_->nv(),
        com_constraint_result->jacobian, com_constraint_result->lower_bounds,
        com_constraint_result->upper_bounds);
  }

  if (rel_pose_constraint_result.has_value()) {
    constraint_idx = append_constraint_block(
        C, c_lower, c_upper, constraint_idx, robot_->nv(),
        rel_pose_constraint_result->jacobian,
        rel_pose_constraint_result->lower_bounds,
        rel_pose_constraint_result->upper_bounds);
  }

  if (step_torso_constraint_result.has_value()) {
    constraint_idx = append_constraint_block(
        C, c_lower, c_upper, constraint_idx, robot_->nv(),
        step_torso_constraint_result->jacobian,
        step_torso_constraint_result->lower_bounds,
        step_torso_constraint_result->upper_bounds);
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
      clamp_joint_velocity_solution_in_place(
          dq, c_lower, c_upper, use_position_limits_, robot_->nv());
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

  stall_handler_update(result);
  elastic_band_update(result);

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
  if (options.nullspace_bias.has_value() &&
      options.nullspace_bias->size() != robot_->nq()) {
    result.status = SolverStatus::kInvalidInput;
    result.status_message =
        "nullspace_bias size does not match robot nq in solve_position";
    return result;
  }
  {
    std::unordered_set<int> seen;
    for (int idx : options.nullspace_active_joints) {
      if (idx < 0 || idx >= robot_->nv()) {
        result.status = SolverStatus::kInvalidInput;
        result.status_message =
            "nullspace_active_joints contains out-of-range index";
        return result;
      }
      if (!seen.insert(idx).second) {
        result.status = SolverStatus::kInvalidInput;
        result.status_message =
            "nullspace_active_joints contains duplicate index";
        return result;
      }
    }
  }
  if (options.nullspace_joint_weights.has_value()) {
    const int expected_size =
        options.nullspace_active_joints.empty()
            ? static_cast<int>(robot_->nv())
            : static_cast<int>(options.nullspace_active_joints.size());
    if (options.nullspace_joint_weights->size() != expected_size) {
      result.status = SolverStatus::kInvalidInput;
      result.status_message =
          "nullspace_joint_weights size mismatch for selected nullspace joints";
      return result;
    }
  }

  const auto &torso_opts = options.torso_constraint;
  const bool torso_enabled = torso_opts.enabled;
  const bool torso_has_pose_bounds = torso_opts.pose_lower_bounds.has_value() ||
                                     torso_opts.pose_upper_bounds.has_value();
  if (torso_enabled) {
    if (torso_opts.frame_name.empty()) {
      result.status = SolverStatus::kInvalidInput;
      result.status_message =
          "torso_constraint.frame_name must be set when torso constraint is enabled";
      return result;
    }
    if (!robot_->has_frame(torso_opts.frame_name)) {
      result.status = SolverStatus::kInvalidInput;
      result.status_message =
          "torso_constraint.frame_name not found in robot model";
      return result;
    }
    if (torso_has_pose_bounds !=
        (torso_opts.pose_lower_bounds.has_value() &&
         torso_opts.pose_upper_bounds.has_value())) {
      result.status = SolverStatus::kInvalidInput;
      result.status_message =
          "torso_constraint pose bounds require both lower and upper vectors";
      return result;
    }
    if (torso_has_pose_bounds) {
      if (torso_opts.pose_bounds_reference_pose.has_value()) {
        const Eigen::Matrix4d &M = torso_opts.pose_bounds_reference_pose.value();
        if (M.rows() != 4 || M.cols() != 4) {
          result.status = SolverStatus::kInvalidInput;
          result.status_message =
              "torso_constraint.pose_bounds_reference_pose must be 4x4";
          return result;
        }
        for (int r = 0; r < 4; ++r) {
          for (int c = 0; c < 4; ++c) {
            if (!std::isfinite(M(r, c))) {
              result.status = SolverStatus::kInvalidInput;
              result.status_message =
                  "torso_constraint.pose_bounds_reference_pose contains non-finite values";
              return result;
            }
          }
        }
      }
      if (torso_opts.pose_lower_bounds->size() != 6 ||
          torso_opts.pose_upper_bounds->size() != 6 ||
          torso_opts.pose_axis_mask.size() != 6 ||
          torso_opts.velocity_limits.size() != 6 ||
          torso_opts.acceleration_limits.size() != 6) {
        result.status = SolverStatus::kInvalidInput;
        result.status_message =
            "torso_constraint bounds, mask, velocity_limits, and acceleration_limits must all be size 6";
        return result;
      }
      for (int i = 0; i < 6; ++i) {
        if (torso_opts.pose_lower_bounds->coeff(i) >
            torso_opts.pose_upper_bounds->coeff(i)) {
          result.status = SolverStatus::kInvalidInput;
          result.status_message =
              "torso_constraint lower bound exceeds upper bound";
          return result;
        }
        if (torso_opts.pose_axis_mask.coeff(i) > 0.5 &&
            (torso_opts.velocity_limits.coeff(i) <= 0.0 ||
             torso_opts.acceleration_limits.coeff(i) <= 0.0)) {
          result.status = SolverStatus::kInvalidInput;
          result.status_message =
              "torso_constraint velocity/acceleration limits must be positive on constrained axes";
          return result;
        }
      }
      if (!std::isfinite(torso_opts.pose_bound_softening_fraction) ||
          torso_opts.pose_bound_softening_fraction < 0.0 ||
          torso_opts.pose_bound_softening_fraction > 1.0) {
        result.status = SolverStatus::kInvalidInput;
        result.status_message =
            "torso_constraint.pose_bound_softening_fraction must be in [0, 1]";
        return result;
      }
      if (!std::isfinite(torso_opts.velocity_box_headroom.fraction) ||
          torso_opts.velocity_box_headroom.fraction < 0.0 ||
          torso_opts.velocity_box_headroom.fraction > 1.0) {
        result.status = SolverStatus::kInvalidInput;
        result.status_message =
            "torso_constraint.velocity_box_headroom.fraction must be in [0, 1]";
        return result;
      }
      if (!std::isfinite(torso_opts.velocity_box_headroom.activation_margin) ||
          torso_opts.velocity_box_headroom.activation_margin < 0.0) {
        result.status = SolverStatus::kInvalidInput;
        result.status_message = "torso_constraint.velocity_box_headroom.activation_margin must be >= 0";
        return result;
      }
    }
  }

  auto frame_task = std::make_shared<FrameTask>(
      "position_ik_task", robot_, frame_name, TaskType::FRAME_POSE);
  if (!options.excluded_joint_indices.empty()) {
    frame_task->set_excluded_joint_indices(options.excluded_joint_indices);
  }

  std::shared_ptr<FrameTask> torso_task = nullptr;
  if (torso_enabled) {
    torso_task = std::make_shared<FrameTask>(
        "torso_orientation_task", robot_, torso_opts.frame_name,
        TaskType::FRAME_ORIENTATION);
    torso_task->setOrientationMask(torso_opts.orientation_mask);
    torso_task->setWeight(torso_opts.orientation_gain);
    torso_task->setPriority(1);
    torso_task->setSolveMode(TaskSolveMode::kMinError);
    torso_task->setAllowMinErrorFallback(false);
    if (!options.excluded_joint_indices.empty()) {
      torso_task->set_excluded_joint_indices(options.excluded_joint_indices);
    }
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
    if (options.nullspace_joint_weights.has_value()) {
      if (options.nullspace_active_joints.empty()) {
        posture_task->setJointWeights(options.nullspace_joint_weights.value());
      } else {
        posture_task->setControlledJointWeights(
            options.nullspace_joint_weights.value());
      }
    }
    posture_task->setPriority(torso_task ? 2 : 1);
    posture_task->setSolveMode(TaskSolveMode::kMinError);
    posture_task->setAllowMinErrorFallback(false);
    if (!options.excluded_joint_indices.empty()) {
      posture_task->set_excluded_joint_indices(options.excluded_joint_indices);
    }
  }

  const bool auto_stall = options.stall_recovery && !stall_config_.enabled;
  if (auto_stall) {
    const double nominal =
        get_collision_min_distance() > 0.0 ? get_collision_min_distance() : 0.0;
    enable_stall_handler(nominal);
  }

  if (options.elastic_band && !elastic_band_config_.enabled) {
    enable_elastic_band(0.05);
    configure_elastic_band(/*delta_max=*/0.05, /*expand_rate=*/0.01,
                           /*decay_rate=*/0.2, /*stall_threshold=*/3,
                           /*expand_only_saturated=*/true);
  }

  Eigen::VectorXd q_current = seed_q;
  const Eigen::VectorXd q_reference = seed_q;
  const auto &velocity_to_config_index = velocity_to_config_index_cache();
  robot_->update_configuration(q_current);

  pinocchio::SE3 torso_reference_pose = pinocchio::SE3::Identity();
  if (torso_task) {
    torso_reference_pose = robot_->get_frame_pose(torso_opts.frame_name);
    const Eigen::Matrix3d torso_target_orientation =
        torso_opts.target_orientation.has_value()
            ? torso_opts.target_orientation.value()
            : torso_reference_pose.rotation();
    torso_task->setTargetOrientation(torso_target_orientation);
  }

  pinocchio::SE3 torso_pose_bounds_reference = pinocchio::SE3::Identity();
  if (torso_has_pose_bounds) {
    if (torso_opts.pose_bounds_reference_pose.has_value()) {
      const Eigen::Matrix4d &M = torso_opts.pose_bounds_reference_pose.value();
      torso_pose_bounds_reference =
          pinocchio::SE3(M.block<3, 3>(0, 0), M.block<3, 1>(0, 3));
    } else {
      torso_pose_bounds_reference =
          robot_->get_frame_pose(torso_opts.frame_name);
    }
  }

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
  std::vector<Eigen::VectorXd> goals;
  std::vector<Eigen::MatrixXd> jacobians;
  std::vector<ObjectiveSolveConfig> objective_configs;
  goals.reserve(3);
  jacobians.reserve(3);
  objective_configs.reserve(3);
  Eigen::MatrixXd C;
  Eigen::VectorXd c_lower;
  Eigen::VectorXd c_upper;

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

    goals.clear();
    jacobians.clear();
    objective_configs.clear();

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
    // SCALE_ELASTIC is treated as SCALE in the SNS solver.
    const auto effective_primary_mode =
        (options.primary_solve_mode == TaskSolveMode::kScaleElastic)
            ? TaskSolveMode::kScale
            : options.primary_solve_mode;
    objective_configs.push_back(
        ObjectiveSolveConfig{0, effective_primary_mode,
                             options.primary_allow_min_error_fallback});

    if (torso_task) {
      torso_task->update(*robot_);
      goals.push_back(torso_task->getVelocity());
      jacobians.push_back(torso_task->getJacobian());
      objective_configs.push_back(
          ObjectiveSolveConfig{1, TaskSolveMode::kMinError, false});
    }

    if (posture_task) {
      posture_task->update(*robot_);
      goals.push_back(posture_task->getVelocity());
      jacobians.push_back(posture_task->getJacobian());
      objective_configs.push_back(
          ObjectiveSolveConfig{torso_task ? 2 : 1, TaskSolveMode::kMinError,
                               false});
    }

    std::optional<CollisionConstraintResult> collision_constraint_result =
        std::nullopt;
    if (collision_constraint_.has_value() && collision_constraint_->enabled) {
      collision_constraint_result = compute_collision_constraint();
    }
    if (collision_constraint_result.has_value() &&
        !options.excluded_joint_indices.empty()) {
      zero_excluded_columns(collision_constraint_result->jacobian,
                            options.excluded_joint_indices);
    }
    std::optional<ComConstraintResult> com_constraint_result = std::nullopt;
    if (com_constraint_.has_value() && com_constraint_->enabled) {
      com_constraint_result = compute_com_constraint();
    }
    if (com_constraint_result.has_value() &&
        !options.excluded_joint_indices.empty()) {
      zero_excluded_columns(com_constraint_result->jacobian,
                            options.excluded_joint_indices);
    }

    std::optional<ConstraintBlock> torso_constraint_result = std::nullopt;
    if (torso_task && torso_has_pose_bounds) {
      torso_constraint_result = build_torso_pose_bound_rows(
          *this, *robot_, torso_opts, torso_pose_bounds_reference, options.dt,
          options.excluded_joint_indices);
    }

    int num_constraints = robot_->nv();
    if (collision_constraint_result.has_value()) {
      num_constraints +=
          static_cast<int>(collision_constraint_result->jacobian.rows());
    }
    if (com_constraint_result.has_value()) {
      num_constraints +=
          static_cast<int>(com_constraint_result->jacobian.rows());
    }
    if (torso_constraint_result.has_value()) {
      num_constraints +=
          static_cast<int>(torso_constraint_result->jacobian.rows());
    }

    C.setZero(num_constraints, robot_->nv());
    c_lower.setConstant(num_constraints, -kUnboundedConstraintLimit);
    c_upper.setConstant(num_constraints, kUnboundedConstraintLimit);

    C.block(0, 0, robot_->nv(), robot_->nv()) =
        Eigen::MatrixXd::Identity(robot_->nv(), robot_->nv());
    int constraint_idx = robot_->nv();

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
      if (options.limit_change_from_seed) {
        tighten_bounds_with_reference_corridor(
            c_lower, c_upper, q_reference, q_current, vel_limits, options.dt,
            velocity_to_config_index, robot_->nv());
      }
    }

    if (collision_constraint_result.has_value()) {
      constraint_idx = append_constraint_block(
          C, c_lower, c_upper, constraint_idx, robot_->nv(),
          collision_constraint_result->jacobian,
          collision_constraint_result->lower_bounds,
          collision_constraint_result->upper_bounds);
    }
    if (com_constraint_result.has_value()) {
      constraint_idx = append_constraint_block(
          C, c_lower, c_upper, constraint_idx, robot_->nv(),
          com_constraint_result->jacobian, com_constraint_result->lower_bounds,
          com_constraint_result->upper_bounds);
    }
    if (torso_constraint_result.has_value()) {
      constraint_idx = append_constraint_block(
          C, c_lower, c_upper, constraint_idx, robot_->nv(),
          torso_constraint_result->jacobian,
          torso_constraint_result->lower_bounds,
          torso_constraint_result->upper_bounds);
    }

    sanitize_solver_inputs(goals, jacobians, C, c_lower, c_upper);

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

    if (stall_config_.enabled) {
      double primary_goal_norm =
          goals.empty() ? 0.0 : goals[0].norm();
      auto classified = classify_velocity_outcome(
          vel_result.status, vel_result.status_message,
          vel_result.task_scales, primary_goal_norm,
          vel_result.task_modes_effective,
          vel_result.task_used_fallback);

      VelocitySolverResult vsr;
      vsr.status = classified.status;
      vsr.status_message = classified.status_message;
      vsr.joint_velocities = Eigen::Map<const Eigen::VectorXd>(
          vel_result.solution.data(),
          static_cast<Eigen::Index>(vel_result.solution.size()));
      if (c_lower.size() >= robot_->nv()) {
        constexpr double kSaturationTol = 0.01;
        for (int i = 0; i < robot_->nv(); ++i) {
          double lower = c_lower[i];
          double upper = c_upper[i];
          if (static_cast<int>(c_lower.size()) >= 2 * robot_->nv()) {
            lower = std::max(lower, c_lower[robot_->nv() + i]);
            upper = std::min(upper, c_upper[robot_->nv() + i]);
          }
          const double joint_vel = vsr.joint_velocities[i];
          if (joint_vel <= lower + kSaturationTol ||
              joint_vel >= upper - kSaturationTol) {
            vsr.saturated_joints.push_back(i);
          }
        }
      }
      stall_handler_update(vsr);
    }

    if (vel_result.status != SolverStatus::kSuccess &&
        !(stall_config_.enabled &&
          vel_result.status == SolverStatus::kInfeasible)) {
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
    clamp_joint_velocity_solution_in_place(dq, c_lower, c_upper,
                                           /*include_position_rows=*/false,
                                           robot_->nv());
    Eigen::VectorXd q_pre_step = q_current;
    q_current =
        pinocchio::integrate(robot_->model(), q_current, options.dt * dq);
    robot_->update_configuration(q_current);

    // Skip expensive post-step checks when integration produced no motion.
    const bool step_moved =
        (q_current - q_pre_step).squaredNorm() > kCollisionEscapeNormEps;
    // Post-solve collision rejection (same logic as solve_position_step).
    if (step_moved && collision_constraint_.has_value() &&
        collision_constraint_->enabled) {
      auto post_dist_debug = evaluate_post_step_collision_distance(q_current);
      if (post_dist_debug.has_value() &&
          std::isfinite(*post_dist_debug) &&
          *post_dist_debug < kCollisionPenetrationDistanceThreshold) {
        auto pre_dist_debug = evaluate_post_step_collision_distance(q_pre_step);
        double pre_dist = (pre_dist_debug.has_value() && std::isfinite(*pre_dist_debug))
                              ? *pre_dist_debug
                              : std::numeric_limits<double>::infinity();
        bool seed_was_safe = (pre_dist >= kCollisionPenetrationDistanceThreshold);
        bool deepened =
            (pre_dist < kCollisionPenetrationDistanceThreshold &&
             *post_dist_debug <
                 pre_dist - kCollisionPenetrationWorsenTolerance);
        const bool hard_jump =
            std::isfinite(pre_dist) &&
            *post_dist_debug < kCollisionHardPenetrationRejectDistance &&
            *post_dist_debug < pre_dist - kCollisionHardWorsenTolerance;
        if (seed_was_safe || deepened || hard_jump) {
          q_current = q_pre_step;
          robot_->update_configuration(q_current);
          result.collision_rejection_count++;
        }
      }
    }

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
      stagnation_abort, options.classify_stagnation_as_no_progress,
      max_iterations_reached, result.position_error,
      result.orientation_error);
  result.status = classified_position.status;
  result.status_message = classified_position.status_message;

  if (position_ik_debug_) {
    std::cout << "[embodiK][IKDebug] solve_position finished with status="
              << static_cast<int>(result.status) << " iterations=" << iter
              << " final_pos_err=" << result.position_error
              << " final_ori_err=" << result.orientation_error << std::endl;
  }

  if (auto_stall) {
    disable_stall_handler();
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

  std::optional<TorsoPoseConstraintOptions> step_torso_constraint = std::nullopt;
  const auto &torso_opts = options.torso_constraint;
  const bool torso_enabled = torso_opts.enabled;
  const bool torso_has_pose_bounds = torso_opts.pose_lower_bounds.has_value() ||
                                     torso_opts.pose_upper_bounds.has_value();
  if (torso_enabled) {
    if (torso_opts.frame_name.empty()) {
      result.status = SolverStatus::kInvalidInput;
      result.status_message =
          "torso_constraint.frame_name must be set when torso constraint is enabled";
      return result;
    }
    if (!robot_->has_frame(torso_opts.frame_name)) {
      result.status = SolverStatus::kInvalidInput;
      result.status_message =
          "torso_constraint.frame_name not found in robot model";
      return result;
    }
    if (torso_has_pose_bounds !=
        (torso_opts.pose_lower_bounds.has_value() &&
         torso_opts.pose_upper_bounds.has_value())) {
      result.status = SolverStatus::kInvalidInput;
      result.status_message =
          "torso_constraint pose bounds require both lower and upper vectors";
      return result;
    }
    if (torso_has_pose_bounds) {
      if (torso_opts.pose_bounds_reference_pose.has_value()) {
        const Eigen::Matrix4d &M = torso_opts.pose_bounds_reference_pose.value();
        if (M.rows() != 4 || M.cols() != 4) {
          result.status = SolverStatus::kInvalidInput;
          result.status_message =
              "torso_constraint.pose_bounds_reference_pose must be 4x4";
          return result;
        }
        for (int r = 0; r < 4; ++r) {
          for (int c = 0; c < 4; ++c) {
            if (!std::isfinite(M(r, c))) {
              result.status = SolverStatus::kInvalidInput;
              result.status_message =
                  "torso_constraint.pose_bounds_reference_pose contains non-finite values";
              return result;
            }
          }
        }
      }
      if (torso_opts.pose_lower_bounds->size() != 6 ||
          torso_opts.pose_upper_bounds->size() != 6 ||
          torso_opts.pose_axis_mask.size() != 6 ||
          torso_opts.velocity_limits.size() != 6 ||
          torso_opts.acceleration_limits.size() != 6) {
        result.status = SolverStatus::kInvalidInput;
        result.status_message =
            "torso_constraint bounds, mask, velocity_limits, and acceleration_limits must all be size 6";
        return result;
      }
      for (int i = 0; i < 6; ++i) {
        if (torso_opts.pose_lower_bounds->coeff(i) >
            torso_opts.pose_upper_bounds->coeff(i)) {
          result.status = SolverStatus::kInvalidInput;
          result.status_message =
              "torso_constraint lower bound exceeds upper bound";
          return result;
        }
        if (torso_opts.pose_axis_mask.coeff(i) > 0.5 &&
            (torso_opts.velocity_limits.coeff(i) <= 0.0 ||
             torso_opts.acceleration_limits.coeff(i) <= 0.0)) {
          result.status = SolverStatus::kInvalidInput;
          result.status_message =
              "torso_constraint velocity/acceleration limits must be positive on constrained axes";
          return result;
        }
      }
      if (!std::isfinite(torso_opts.pose_bound_softening_fraction) ||
          torso_opts.pose_bound_softening_fraction < 0.0 ||
          torso_opts.pose_bound_softening_fraction > 1.0) {
        result.status = SolverStatus::kInvalidInput;
        result.status_message =
            "torso_constraint.pose_bound_softening_fraction must be in [0, 1]";
        return result;
      }
      if (!std::isfinite(torso_opts.velocity_box_headroom.fraction) ||
          torso_opts.velocity_box_headroom.fraction < 0.0 ||
          torso_opts.velocity_box_headroom.fraction > 1.0) {
        result.status = SolverStatus::kInvalidInput;
        result.status_message =
            "torso_constraint.velocity_box_headroom.fraction must be in [0, 1]";
        return result;
      }
      if (!std::isfinite(torso_opts.velocity_box_headroom.activation_margin) ||
          torso_opts.velocity_box_headroom.activation_margin < 0.0) {
        result.status = SolverStatus::kInvalidInput;
        result.status_message =
            "torso_constraint.velocity_box_headroom.activation_margin must be >= 0";
        return result;
      }

      TorsoPoseConstraintOptions normalized = torso_opts;
      if (!normalized.pose_bounds_reference_pose.has_value()) {
        const pinocchio::SE3 ref_pose = robot_->get_frame_pose(normalized.frame_name);
        Eigen::Matrix4d ref = Eigen::Matrix4d::Identity();
        ref.block<3, 3>(0, 0) = ref_pose.rotation();
        ref.block<3, 1>(0, 3) = ref_pose.translation();
        normalized.pose_bounds_reference_pose = ref;
      }
      step_torso_constraint = std::move(normalized);
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

  // Enable stall handler if requested. enable_stall_handler is idempotent:
  // repeated calls preserve accumulated stall counters so detection works
  // across successive single-step solve_position_step calls.
  if (options.stall_recovery) {
    const double nominal =
        get_collision_min_distance() > 0.0 ? get_collision_min_distance() : 0.0;
    enable_stall_handler(nominal);
  }

  if (options.elastic_band && !elastic_band_config_.enabled) {
    enable_elastic_band(0.05);
    configure_elastic_band(/*delta_max=*/0.05, /*expand_rate=*/0.01,
                           /*decay_rate=*/0.2, /*stall_threshold=*/3,
                           /*expand_only_saturated=*/true);
  }

  Eigen::VectorXd q = current_q;
  const Eigen::VectorXd q_reference = current_q;
  robot_->update_configuration(q);
  const auto &velocity_to_config_index = velocity_to_config_index_cache();
  const Eigen::VectorXd vel_limits = robot_->get_velocity_limits();

  const std::vector<int> step_locked_indices =
      build_step_locked_indices(options);

  VelocitySolverResult last_vel_result;
  bool have_vel_result = false;
  const int steps = std::max(1, options.max_steps);
  int steps_used = 0;
  Eigen::VectorXd vel(6);
  double prev_combined_error = std::numeric_limits<double>::infinity();
  int no_progress_count = 0;
  bool no_progress_exit = false;

  for (int step = 0; step < steps; ++step) {
    frame_task->update(*robot_);
    const Eigen::VectorXd &error = frame_task->getError();
    const double combined_error = error.head<3>().norm() + error.tail<3>().norm();
    if (step > 0 && options.no_progress_max_steps > 0 && have_vel_result) {
      const bool low_error_change =
          std::abs(prev_combined_error - combined_error) <=
          options.no_progress_error_tolerance;
      const bool low_velocity =
          last_vel_result.joint_velocities.size() == robot_->nv() &&
          last_vel_result.joint_velocities.norm() <=
              options.no_progress_dq_norm_tolerance;
      if (low_error_change && low_velocity) {
        ++no_progress_count;
        if (no_progress_count >= options.no_progress_max_steps) {
          no_progress_exit = true;
          break;
        }
      } else {
        no_progress_count = 0;
      }
    }
    prev_combined_error = combined_error;
    vel.head<3>() = options.position_gain * error.head<3>();
    vel.tail<3>() = options.orientation_gain * error.tail<3>();
    clamp_spatial_velocity_components(vel, options.max_linear_speed,
                                      options.max_angular_speed);
    frame_task->setTargetVelocity(vel);

    pending_velocity_lock_indices_ = step_locked_indices;
    pending_step_torso_constraint_ = step_torso_constraint;
    VelocitySolverResult vel_out = solve_velocity(q, true);
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
    if (options.limit_change_from_seed &&
        last_vel_result.joint_velocities.size() == robot_->nv()) {
      Eigen::VectorXd corridor_lower = Eigen::VectorXd::Constant(
          robot_->nv(), -kUnboundedConstraintLimit);
      Eigen::VectorXd corridor_upper = Eigen::VectorXd::Constant(
          robot_->nv(), kUnboundedConstraintLimit);
      tighten_bounds_with_reference_corridor(
          corridor_lower, corridor_upper, q_reference, q, vel_limits, step_dt,
          velocity_to_config_index, robot_->nv());
      clamp_joint_velocity_solution_in_place(last_vel_result.joint_velocities,
                                             corridor_lower, corridor_upper,
                                             false, robot_->nv());
    }

    Eigen::VectorXd q_pre_step = q;
    bool step_collision_rejected = false;
    q = pinocchio::integrate(robot_->model(), q,
                             step_dt * last_vel_result.joint_velocities);
    robot_->update_configuration(q);

    // Skip expensive post-step checks when integration produced no motion.
    const bool step_moved =
        (q - q_pre_step).squaredNorm() > kCollisionEscapeNormEps;
    // Post-solve collision rejection (same logic as multi-target overload).
    if (step_moved && collision_constraint_.has_value() &&
        collision_constraint_->enabled) {
      auto post_dist_debug = evaluate_post_step_collision_distance(q);
      if (post_dist_debug.has_value() &&
          std::isfinite(*post_dist_debug) &&
          *post_dist_debug < kCollisionPenetrationDistanceThreshold) {
        auto pre_dist_debug = evaluate_post_step_collision_distance(q_pre_step);
        double pre_dist = (pre_dist_debug.has_value() && std::isfinite(*pre_dist_debug))
                              ? *pre_dist_debug
                              : std::numeric_limits<double>::infinity();
        bool seed_was_safe = (pre_dist >= kCollisionPenetrationDistanceThreshold);
        bool deepened =
            (pre_dist < kCollisionPenetrationDistanceThreshold &&
             *post_dist_debug <
                 pre_dist - kCollisionPenetrationWorsenTolerance);
        const bool hard_jump =
            std::isfinite(pre_dist) &&
            *post_dist_debug < kCollisionHardPenetrationRejectDistance &&
            *post_dist_debug < pre_dist - kCollisionHardWorsenTolerance;
        if (seed_was_safe || deepened || hard_jump) {
          bool accepted_backoff = false;
          const Eigen::VectorXd dq_nominal =
              step_dt * last_vel_result.joint_velocities;
          for (double frac : kCollisionRejectionBackoffFractions) {
            Eigen::VectorXd q_backoff = pinocchio::integrate(
                robot_->model(), q_pre_step, frac * dq_nominal);
            robot_->update_configuration(q_backoff);
            auto backoff_dist_debug = evaluate_post_step_collision_distance(q_backoff);
            if (backoff_dist_debug.has_value() &&
                std::isfinite(*backoff_dist_debug) &&
                *backoff_dist_debug >=
                    kCollisionPenetrationDistanceThreshold) {
              q = q_backoff;
              accepted_backoff = true;
              break;
            }
          }
          if (!accepted_backoff) {
            q = q_pre_step;
            robot_->update_configuration(q);
            step_collision_rejected = true;
            result.collision_rejection_count++;
          }
        }
      }
    }

    // Cache invalidation on stall (same as multi-target overload).
    // Only invalidate when actually near collision — in clear space the cache
    // remains valid and avoids an expensive full scan on the next step.
    const double step_dq_norm = last_vel_result.joint_velocities.norm();
    const double effective_step_dq_norm =
        step_collision_rejected ? 0.0 : step_dq_norm;
    if (effective_step_dq_norm < stall_config_.dq_stall_eps) {
      const bool near_penetration =
          !std::isfinite(last_constraint_min_distance_) ||
          last_collision_budget_exhausted_ ||
          last_constraint_min_distance_ <
              kCollisionPenetrationDistanceThreshold +
                  kPostStepSafeMargin;
      if (near_penetration) {
        collision_pair_cache_has_full_scan_ = false;
      }
    }

    // Stall escape (same logic as multi-target overload).
    // Skip the expensive escape logic entirely when we know from the
    // constraint computation that all pairs are well clear.
    const bool stall_escape_eligible =
        collision_constraint_.has_value() && collision_constraint_->enabled &&
        (effective_step_dq_norm < stall_config_.dq_stall_eps ||
         step_collision_rejected);
    const bool stall_escape_near_collision =
        stall_escape_eligible &&
        (!std::isfinite(last_constraint_min_distance_) ||
         last_collision_budget_exhausted_ ||
         last_constraint_min_distance_ <
             collision_constraint_->min_distance +
                 kPostStepSafeMargin);
    if (stall_escape_near_collision) {
      double min_dist = std::numeric_limits<double>::infinity();
      int worst_row = -1;
      for (int i = 0; i < static_cast<int>(last_collision_debug_list_.size()); ++i) {
        if (last_collision_debug_list_[i].distance < min_dist) {
          min_dist = last_collision_debug_list_[i].distance;
          worst_row = i;
        }
      }

      // For stalled steps, also allow escape when we violate configured
      // clearance (distance < min_distance) even if still non-penetrating.
      const double escape_activation_threshold =
          (step_collision_rejected
               ? kCollisionEscapeActivationDistance
               : std::max(kCollisionEscapeActivationDistance,
                          collision_constraint_->min_distance));
      bool use_jacobian_escape =
          (min_dist < escape_activation_threshold && worst_row >= 0);
      bool use_normal_escape = false;
      Eigen::Vector3d escape_normal = Eigen::Vector3d::Zero();
      std::string escape_frame_a, escape_frame_b;
      double global_min_dist = min_dist;

      if (!use_jacobian_escape) {
        // Fast path: when the constraint computation already has a reliable
        // global minimum distance, use it directly instead of the expensive
        // full-scan evaluate_collision_debug().  Only fall back to the full
        // scan when no constraint data is available or budget was exhausted.
        const bool have_reliable_constraint_dist =
            std::isfinite(last_constraint_min_distance_) &&
            !last_collision_budget_exhausted_;
        if (have_reliable_constraint_dist) {
          if (last_constraint_min_distance_ < escape_activation_threshold &&
              last_collision_debug_.has_value() &&
              std::isfinite(last_collision_debug_->distance)) {
            global_min_dist = last_collision_debug_->distance;
            Eigen::Vector3d diff = last_collision_debug_->point_b_world -
                                   last_collision_debug_->point_a_world;
            double diff_norm = diff.norm();
            if (diff_norm > kCollisionEscapeNormEps) {
              escape_normal = diff / diff_norm;
            } else {
              escape_normal = Eigen::Vector3d(0, 0, 1);
            }
            escape_frame_a = last_collision_debug_->object_a;
            escape_frame_b = last_collision_debug_->object_b;
            use_normal_escape = true;
          }
        } else {
          auto global_debug = evaluate_collision_debug(q);
          if (global_debug.has_value() && std::isfinite(global_debug->distance) &&
              global_debug->distance < escape_activation_threshold) {
            global_min_dist = global_debug->distance;
            Eigen::Vector3d diff = global_debug->point_b_world - global_debug->point_a_world;
            double diff_norm = diff.norm();
            if (diff_norm > kCollisionEscapeNormEps) {
              escape_normal = diff / diff_norm;
            } else {
              escape_normal = Eigen::Vector3d(0, 0, 1);
            }
            escape_frame_a = global_debug->object_a;
            escape_frame_b = global_debug->object_b;
            use_normal_escape = true;
          }
        }
      }

      if (use_jacobian_escape) {
        auto coll_result = compute_collision_constraint();
        if (coll_result.has_value() &&
            worst_row < coll_result->jacobian.rows()) {
          Eigen::RowVectorXd n_row = coll_result->jacobian.row(worst_row);
          double nn = n_row.squaredNorm();
          if (nn > kCollisionEscapeNormEps) {
            double escape_mag =
                std::min(kCollisionEscapeStepMax, std::abs(min_dist));
            if (step_collision_rejected) {
              escape_mag = std::max(escape_mag, kCollisionEscapeStepMinOnRejection);
            }
            const Eigen::VectorXd dq_unit =
                (1.0 / std::sqrt(nn)) * n_row.transpose();
            auto try_signed_escape = [&](double sign) -> bool {
              Eigen::VectorXd q_candidate = pinocchio::integrate(
                  robot_->model(), q, sign * escape_mag * dq_unit);
              auto [q_min, q_max] = robot_->get_joint_limits();
              for (int j = 0; j < static_cast<int>(q_candidate.size()); ++j) {
                if (j < q_min.size()) {
                  q_candidate[j] = std::max(q_candidate[j], q_min[j]);
                  q_candidate[j] = std::min(q_candidate[j], q_max[j]);
                }
              }
              robot_->update_configuration(q_candidate);
              auto esc_dist = evaluate_post_step_collision_distance(q_candidate);
              if (esc_dist.has_value() && std::isfinite(*esc_dist) &&
                  *esc_dist > min_dist) {
                q = q_candidate;
                result.stall_escape_count++;
                return true;
              }
              robot_->update_configuration(q);
              return false;
            };
            if (!try_signed_escape(+1.0)) {
              try_signed_escape(-1.0);
            }
          }
        }
      } else if (use_normal_escape) {
        const auto *collision_model = robot_->collision_model();
        auto object_to_frame_name = [&](const std::string &obj_name) -> std::string {
          if (!collision_model) return std::string();
          for (std::size_t gi = 0; gi < collision_model->ngeoms; ++gi) {
            const auto &go = collision_model->geometryObjects[gi];
            if (go.name == obj_name) {
              return robot_->model().frames[go.parentFrame].name;
            }
          }
          return std::string();
        };
        const std::string escape_frame_a_name = object_to_frame_name(escape_frame_a);
        const std::string escape_frame_b_name = object_to_frame_name(escape_frame_b);

        auto try_frame_jacobian_escape = [&](const std::string &frame_name) -> bool {
          if (frame_name.empty() || !robot_->has_frame(frame_name)) return false;

          pinocchio::Data &data = robot_->data();
          pinocchio::computeJointJacobians(robot_->model(), data, q);
          auto frame_id = robot_->model().getFrameId(frame_name);
          Eigen::MatrixXd J(6, robot_->nv());
          J.setZero();
          pinocchio::getFrameJacobian(robot_->model(), data, frame_id,
                                      pinocchio::LOCAL_WORLD_ALIGNED, J);
          Eigen::MatrixXd J_lin = J.topRows<3>();
          Eigen::VectorXd dq_escape = J_lin.transpose() * escape_normal;
          double dq_n = dq_escape.norm();
          if (dq_n < kCollisionEscapeNormEps) return false;
          double escape_mag =
              std::min(kCollisionEscapeStepMax, std::abs(global_min_dist));
          if (step_collision_rejected) {
            escape_mag = std::max(escape_mag, kCollisionEscapeStepMinOnRejection);
          }
          dq_escape = (escape_mag / dq_n) * dq_escape;

          Eigen::VectorXd q_candidate = pinocchio::integrate(robot_->model(), q, dq_escape);
          auto [q_min, q_max] = robot_->get_joint_limits();
          for (int j = 0; j < static_cast<int>(q_candidate.size()); ++j) {
            if (j < q_min.size()) {
              q_candidate[j] = std::max(q_candidate[j], q_min[j]);
              q_candidate[j] = std::min(q_candidate[j], q_max[j]);
            }
          }
          robot_->update_configuration(q_candidate);
          auto esc_dist = evaluate_post_step_collision_distance(q_candidate);
          if (esc_dist.has_value() && std::isfinite(*esc_dist) &&
              *esc_dist > global_min_dist) {
            q = q_candidate;
            result.stall_escape_count++;
            return true;
          }
          robot_->update_configuration(q);
          return false;
        };

        if (!try_frame_jacobian_escape(escape_frame_b_name)) {
          try_frame_jacobian_escape(escape_frame_a_name);
        }
      }
    }

    // Limit-dominated lock breaker: if the QP reports near-zero joint motion
    // while several joints are saturated, nudge those joints slightly away
    // from hard limits to recover feasible directions.
    const bool collapsed_primary_scale =
        !last_vel_result.task_scales.empty() &&
        std::abs(last_vel_result.task_scales[0]) <= 1e-6;
    const bool plateau_status =
        last_vel_result.status == SolverStatus::kNumericalError ||
        last_vel_result.status == SolverStatus::kInfeasible ||
        collapsed_primary_scale;
    const int plateau_stall_steps =
        stall_handler_enabled() ? stall_state_.consecutive_stall_steps : 0;
    const bool plateau_escape =
        plateau_status &&
        plateau_stall_steps >= kJointLimitDesaturationPlateauThreshold;
    std::vector<int> desaturation_candidates = last_vel_result.saturated_joints;
    auto [q_min, q_max] = robot_->get_joint_limits();
    if (plateau_escape) {
      for (int vi = 0;
           vi < static_cast<int>(velocity_to_config_index.size()); ++vi) {
        if (std::find(desaturation_candidates.begin(),
                      desaturation_candidates.end(),
                      vi) != desaturation_candidates.end()) {
          continue;
        }
        const int qi = velocity_to_config_index[vi];
        if (qi < 0 || qi >= static_cast<int>(q.size()) ||
            qi >= static_cast<int>(q_min.size()) ||
            qi >= static_cast<int>(q_max.size())) {
          continue;
        }
        const double margin_low = q[qi] - q_min[qi];
        const double margin_up = q_max[qi] - q[qi];
        if ((margin_low >= 0.0 &&
             margin_low < kJointLimitDesaturationExpandedMargin) ||
            (margin_up >= 0.0 &&
             margin_up < kJointLimitDesaturationExpandedMargin)) {
          desaturation_candidates.push_back(vi);
        }
      }
    }
    if (effective_step_dq_norm < stall_config_.dq_stall_eps &&
        (desaturation_candidates.size() >= 4 ||
         (plateau_escape && !desaturation_candidates.empty()))) {
      Eigen::VectorXd q_candidate = q;
      bool changed = false;
      const double desaturation_margin =
          plateau_escape ? kJointLimitDesaturationExpandedMargin
                         : kJointLimitDesaturationMargin;
      const double desaturation_step =
          plateau_escape ? kJointLimitDesaturationBoostStep
                         : kJointLimitDesaturationStep;
      for (int vi : desaturation_candidates) {
        if (vi < 0 || vi >= static_cast<int>(velocity_to_config_index.size())) {
          continue;
        }
        const int qi = velocity_to_config_index[vi];
        if (qi < 0 || qi >= static_cast<int>(q_candidate.size()) ||
            qi >= static_cast<int>(q_min.size()) ||
            qi >= static_cast<int>(q_max.size())) {
          continue;
        }
        const double margin_low = q_candidate[qi] - q_min[qi];
        const double margin_up = q_max[qi] - q_candidate[qi];
        if (margin_low >= 0.0 && margin_low < desaturation_margin) {
          q_candidate[qi] += desaturation_step;
          changed = true;
        } else if (margin_up >= 0.0 && margin_up < desaturation_margin) {
          q_candidate[qi] -= desaturation_step;
          changed = true;
        }
      }
      if (changed) {
        for (int j = 0; j < static_cast<int>(q_candidate.size()); ++j) {
          if (j < q_min.size()) {
            q_candidate[j] = std::max(q_candidate[j], q_min[j]);
            q_candidate[j] = std::min(q_candidate[j], q_max[j]);
          }
        }
        auto curr_dist = evaluate_post_step_collision_distance(q);
        robot_->update_configuration(q_candidate);
        auto cand_dist = evaluate_post_step_collision_distance(q_candidate);
        const bool safe_candidate =
            (!cand_dist.has_value() || !std::isfinite(*cand_dist) ||
             *cand_dist >= kCollisionPenetrationDistanceThreshold);
        const bool not_worse = (!curr_dist.has_value() || !cand_dist.has_value() ||
                                !std::isfinite(*curr_dist) ||
                                !std::isfinite(*cand_dist) ||
                                *cand_dist >=
                                    *curr_dist - kCollisionPenetrationWorsenTolerance);
        if (safe_candidate && not_worse) {
          q = q_candidate;
          result.stall_escape_count++;
        } else {
          robot_->update_configuration(q);
        }
      }
    }
  }

  clear_all_target_velocities();

  result.q_solution = q;
  result.achieved_pose = robot_->get_frame_pose(frame_task->getFrameName());
  result.iterations_used = steps_used;

  frame_task->update(*robot_);
  const Eigen::VectorXd &final_error = frame_task->getError();
  result.position_error = final_error.head<3>().norm();
  result.orientation_error = final_error.tail<3>().norm();

  if (have_vel_result) {
    if (last_vel_result.joint_velocities.size() == robot_->nv()) {
      last_vel_result.solution.assign(last_vel_result.joint_velocities.data(),
                                      last_vel_result.joint_velocities.data() +
                                          last_vel_result.joint_velocities.size());
    }
    static_cast<VelocitySolverResult &>(result) = std::move(last_vel_result);
  }
  if (no_progress_exit && result.status != SolverStatus::kInvalidInput) {
    result.status = SolverStatus::kNoProgress;
    result.status_message =
        "solve_position_step exited due to no progress near active bounds/"
        "constraints";
  }

  // In teleop-style loops (max_steps=1), stall handling must accumulate across
  // successive solve_position_step calls based on *applied* motion, not only
  // raw QP status.  This post-step update prevents success/no-progress
  // oscillations from resetting the consecutive stall counter.
  if (stall_handler_enabled()) {
    const double applied_step_norm = (result.q_solution - current_q).norm();
    if (applied_step_norm < stall_config_.dq_stall_eps) {
      stall_state_.consecutive_stall_steps++;
      stall_state_.total_stall_steps++;
    } else {
      stall_state_.consecutive_stall_steps = 0;
    }
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

  std::optional<TorsoPoseConstraintOptions> step_torso_constraint = std::nullopt;
  const auto &torso_opts = options.torso_constraint;
  const bool torso_enabled = torso_opts.enabled;
  const bool torso_has_pose_bounds = torso_opts.pose_lower_bounds.has_value() ||
                                     torso_opts.pose_upper_bounds.has_value();
  if (torso_enabled) {
    if (torso_opts.frame_name.empty()) {
      result.status = SolverStatus::kInvalidInput;
      result.status_message =
          "torso_constraint.frame_name must be set when torso constraint is enabled";
      return result;
    }
    if (!robot_->has_frame(torso_opts.frame_name)) {
      result.status = SolverStatus::kInvalidInput;
      result.status_message =
          "torso_constraint.frame_name not found in robot model";
      return result;
    }
    if (torso_has_pose_bounds !=
        (torso_opts.pose_lower_bounds.has_value() &&
         torso_opts.pose_upper_bounds.has_value())) {
      result.status = SolverStatus::kInvalidInput;
      result.status_message =
          "torso_constraint pose bounds require both lower and upper vectors";
      return result;
    }
    if (torso_has_pose_bounds) {
      if (torso_opts.pose_bounds_reference_pose.has_value()) {
        const Eigen::Matrix4d &M = torso_opts.pose_bounds_reference_pose.value();
        if (M.rows() != 4 || M.cols() != 4) {
          result.status = SolverStatus::kInvalidInput;
          result.status_message =
              "torso_constraint.pose_bounds_reference_pose must be 4x4";
          return result;
        }
        for (int r = 0; r < 4; ++r) {
          for (int c = 0; c < 4; ++c) {
            if (!std::isfinite(M(r, c))) {
              result.status = SolverStatus::kInvalidInput;
              result.status_message =
                  "torso_constraint.pose_bounds_reference_pose contains non-finite values";
              return result;
            }
          }
        }
      }
      if (torso_opts.pose_lower_bounds->size() != 6 ||
          torso_opts.pose_upper_bounds->size() != 6 ||
          torso_opts.pose_axis_mask.size() != 6 ||
          torso_opts.velocity_limits.size() != 6 ||
          torso_opts.acceleration_limits.size() != 6) {
        result.status = SolverStatus::kInvalidInput;
        result.status_message =
            "torso_constraint bounds, mask, velocity_limits, and acceleration_limits must all be size 6";
        return result;
      }
      for (int i = 0; i < 6; ++i) {
        if (torso_opts.pose_lower_bounds->coeff(i) >
            torso_opts.pose_upper_bounds->coeff(i)) {
          result.status = SolverStatus::kInvalidInput;
          result.status_message =
              "torso_constraint lower bound exceeds upper bound";
          return result;
        }
        if (torso_opts.pose_axis_mask.coeff(i) > 0.5 &&
            (torso_opts.velocity_limits.coeff(i) <= 0.0 ||
             torso_opts.acceleration_limits.coeff(i) <= 0.0)) {
          result.status = SolverStatus::kInvalidInput;
          result.status_message =
              "torso_constraint velocity/acceleration limits must be positive on constrained axes";
          return result;
        }
      }
      if (!std::isfinite(torso_opts.pose_bound_softening_fraction) ||
          torso_opts.pose_bound_softening_fraction < 0.0 ||
          torso_opts.pose_bound_softening_fraction > 1.0) {
        result.status = SolverStatus::kInvalidInput;
        result.status_message =
            "torso_constraint.pose_bound_softening_fraction must be in [0, 1]";
        return result;
      }
      if (!std::isfinite(torso_opts.velocity_box_headroom.fraction) ||
          torso_opts.velocity_box_headroom.fraction < 0.0 ||
          torso_opts.velocity_box_headroom.fraction > 1.0) {
        result.status = SolverStatus::kInvalidInput;
        result.status_message =
            "torso_constraint.velocity_box_headroom.fraction must be in [0, 1]";
        return result;
      }
      if (!std::isfinite(torso_opts.velocity_box_headroom.activation_margin) ||
          torso_opts.velocity_box_headroom.activation_margin < 0.0) {
        result.status = SolverStatus::kInvalidInput;
        result.status_message =
            "torso_constraint.velocity_box_headroom.activation_margin must be >= 0";
        return result;
      }

      TorsoPoseConstraintOptions normalized = torso_opts;
      if (!normalized.pose_bounds_reference_pose.has_value()) {
        const pinocchio::SE3 ref_pose = robot_->get_frame_pose(normalized.frame_name);
        Eigen::Matrix4d ref = Eigen::Matrix4d::Identity();
        ref.block<3, 3>(0, 0) = ref_pose.rotation();
        ref.block<3, 1>(0, 3) = ref_pose.translation();
        normalized.pose_bounds_reference_pose = ref;
      }
      step_torso_constraint = std::move(normalized);
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

  // Enable stall handler if requested. enable_stall_handler is idempotent:
  // repeated calls preserve accumulated stall counters so detection works
  // across successive single-step solve_position_step calls.
  if (options.stall_recovery) {
    const double nominal =
        get_collision_min_distance() > 0.0 ? get_collision_min_distance() : 0.0;
    enable_stall_handler(nominal);
  }

  if (options.elastic_band && !elastic_band_config_.enabled) {
    enable_elastic_band(0.05);
    configure_elastic_band(/*delta_max=*/0.05, /*expand_rate=*/0.01,
                           /*decay_rate=*/0.2, /*stall_threshold=*/3,
                           /*expand_only_saturated=*/true);
  }

  Eigen::VectorXd q = current_q;
  const Eigen::VectorXd q_reference = current_q;
  robot_->update_configuration(q);
  const auto &velocity_to_config_index = velocity_to_config_index_cache();
  const Eigen::VectorXd vel_limits = robot_->get_velocity_limits();

  std::vector<std::shared_ptr<Task>> pose_only;
  pose_only.reserve(resolved.size());
  for (const auto &rt : resolved) {
    pose_only.push_back(rt.task);
  }
  const std::vector<int> step_locked_indices =
      build_step_locked_indices(options);

  VelocitySolverResult last_vel_result;
  bool have_vel_result = false;
  const int steps = std::max(1, options.max_steps);
  int steps_used = 0;
  Eigen::Matrix<double, 6, 1> vel;
  double prev_combined_error = std::numeric_limits<double>::infinity();
  int no_progress_count = 0;
  bool no_progress_exit = false;

  for (int step = 0; step < steps; ++step) {
    double combined_error = 0.0;
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
      combined_error += error.head<3>().norm() + error.tail<3>().norm();
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
    if (step > 0 && options.no_progress_max_steps > 0 && have_vel_result) {
      const bool low_error_change =
          std::abs(prev_combined_error - combined_error) <=
          options.no_progress_error_tolerance;
      const bool low_velocity =
          last_vel_result.joint_velocities.size() == robot_->nv() &&
          last_vel_result.joint_velocities.norm() <=
              options.no_progress_dq_norm_tolerance;
      if (low_error_change && low_velocity) {
        ++no_progress_count;
        if (no_progress_count >= options.no_progress_max_steps) {
          no_progress_exit = true;
          break;
        }
      } else {
        no_progress_count = 0;
      }
    }
    prev_combined_error = combined_error;

    pending_velocity_lock_indices_ = step_locked_indices;
    pending_step_torso_constraint_ = step_torso_constraint;
    VelocitySolverResult vel_out = solve_velocity(q, true);
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
    if (options.limit_change_from_seed &&
        last_vel_result.joint_velocities.size() == robot_->nv()) {
      Eigen::VectorXd corridor_lower = Eigen::VectorXd::Constant(
          robot_->nv(), -kUnboundedConstraintLimit);
      Eigen::VectorXd corridor_upper = Eigen::VectorXd::Constant(
          robot_->nv(), kUnboundedConstraintLimit);
      tighten_bounds_with_reference_corridor(
          corridor_lower, corridor_upper, q_reference, q, vel_limits, step_dt,
          velocity_to_config_index, robot_->nv());
      clamp_joint_velocity_solution_in_place(last_vel_result.joint_velocities,
                                             corridor_lower, corridor_upper,
                                             false, robot_->nv());
    }

    Eigen::VectorXd q_pre_step = q;
    bool step_collision_rejected = false;
    q = pinocchio::integrate(robot_->model(), q,
                             step_dt * last_vel_result.joint_velocities);
    robot_->update_configuration(q);

    // Skip expensive post-step checks when integration produced no motion.
    const bool step_moved =
        (q - q_pre_step).squaredNorm() > kCollisionEscapeNormEps;
    // Post-solve collision rejection: if collision is configured and the
    // integration step created new penetration or deepened existing
    // penetration past a safety threshold, revert to pre-step config.
    if (step_moved && collision_constraint_.has_value() &&
        collision_constraint_->enabled) {
      auto post_dist_debug = evaluate_post_step_collision_distance(q);
      if (post_dist_debug.has_value() &&
          std::isfinite(*post_dist_debug) &&
          *post_dist_debug < kCollisionPenetrationDistanceThreshold) {
        // Check pre-step distance to decide if this step caused the problem.
        auto pre_dist_debug = evaluate_post_step_collision_distance(q_pre_step);
        double pre_dist = (pre_dist_debug.has_value() && std::isfinite(*pre_dist_debug))
                              ? *pre_dist_debug
                              : std::numeric_limits<double>::infinity();
        bool seed_was_safe = (pre_dist >= kCollisionPenetrationDistanceThreshold);
        bool deepened =
            (pre_dist < kCollisionPenetrationDistanceThreshold &&
             *post_dist_debug <
                 pre_dist - kCollisionPenetrationWorsenTolerance);
        if (seed_was_safe || deepened) {
          bool accepted_backoff = false;
          const Eigen::VectorXd dq_nominal =
              step_dt * last_vel_result.joint_velocities;
          for (double frac : kCollisionRejectionBackoffFractions) {
            Eigen::VectorXd q_backoff = pinocchio::integrate(
                robot_->model(), q_pre_step, frac * dq_nominal);
            robot_->update_configuration(q_backoff);
            auto backoff_dist_debug = evaluate_post_step_collision_distance(q_backoff);
            if (backoff_dist_debug.has_value() &&
                std::isfinite(*backoff_dist_debug) &&
                *backoff_dist_debug >=
                    kCollisionPenetrationDistanceThreshold) {
              q = q_backoff;
              accepted_backoff = true;
              break;
            }
          }
          if (!accepted_backoff) {
            q = q_pre_step;
            robot_->update_configuration(q);
            step_collision_rejected = true;
            result.collision_rejection_count++;
          }
        }
      }
    }

    // When stalled, invalidate the collision pair cache so the next solve
    // tick performs a full scan and picks up any penetrating pairs that the
    // cached candidate set may have missed.
    // Only invalidate when actually near collision — in clear space the cache
    // remains valid and avoids an expensive full scan on the next step.
    const double step_dq_norm = last_vel_result.joint_velocities.norm();
    const double effective_step_dq_norm =
        step_collision_rejected ? 0.0 : step_dq_norm;
    if (effective_step_dq_norm < stall_config_.dq_stall_eps) {
      const bool near_penetration =
          !std::isfinite(last_constraint_min_distance_) ||
          last_collision_budget_exhausted_ ||
          last_constraint_min_distance_ <
              kCollisionPenetrationDistanceThreshold +
                  kPostStepSafeMargin;
      if (near_penetration) {
        collision_pair_cache_has_full_scan_ = false;
      }
    }

    // Stall escape: when stalled in penetration, nudge the config along the
    // collision normal to escape.  First check the QP's active constraint
    // list; if that doesn't show penetration, fall back to the full-scan
    // evaluate_collision_debug() which finds pairs the cache may have missed.
    // Skip the expensive escape logic entirely when we know from the
    // constraint computation that all pairs are well clear.
    const bool mt_stall_escape_eligible =
        collision_constraint_.has_value() && collision_constraint_->enabled &&
        (effective_step_dq_norm < stall_config_.dq_stall_eps ||
         step_collision_rejected);
    const bool mt_stall_escape_near_collision =
        mt_stall_escape_eligible &&
        (!std::isfinite(last_constraint_min_distance_) ||
         last_collision_budget_exhausted_ ||
         last_constraint_min_distance_ <
             collision_constraint_->min_distance +
                 kPostStepSafeMargin);
    if (mt_stall_escape_near_collision) {
      double min_dist = std::numeric_limits<double>::infinity();
      int worst_row = -1;
      for (int i = 0; i < static_cast<int>(last_collision_debug_list_.size()); ++i) {
        if (last_collision_debug_list_[i].distance < min_dist) {
          min_dist = last_collision_debug_list_[i].distance;
          worst_row = i;
        }
      }

      // For stalled steps, also allow escape when we violate configured
      // clearance (distance < min_distance) even if still non-penetrating.
      const double escape_activation_threshold =
          (step_collision_rejected
               ? kCollisionEscapeActivationDistance
               : std::max(kCollisionEscapeActivationDistance,
                          collision_constraint_->min_distance));
      bool use_jacobian_escape =
          (min_dist < escape_activation_threshold && worst_row >= 0);
      bool use_normal_escape = false;
      Eigen::Vector3d escape_normal = Eigen::Vector3d::Zero();
      std::string escape_frame_a, escape_frame_b;
      double global_min_dist = min_dist;

      if (!use_jacobian_escape) {
        // Fast path: when the constraint computation already has a reliable
        // global minimum distance, use it directly instead of the expensive
        // full-scan evaluate_collision_debug().  Only fall back to the full
        // scan when no constraint data is available or budget was exhausted.
        const bool have_reliable_constraint_dist =
            std::isfinite(last_constraint_min_distance_) &&
            !last_collision_budget_exhausted_;
        if (have_reliable_constraint_dist) {
          if (last_constraint_min_distance_ < escape_activation_threshold &&
              last_collision_debug_.has_value() &&
              std::isfinite(last_collision_debug_->distance)) {
            global_min_dist = last_collision_debug_->distance;
            Eigen::Vector3d diff = last_collision_debug_->point_b_world -
                                   last_collision_debug_->point_a_world;
            double diff_norm = diff.norm();
            if (diff_norm > kCollisionEscapeNormEps) {
              escape_normal = diff / diff_norm;
            } else {
              escape_normal = Eigen::Vector3d(0, 0, 1);
            }
            escape_frame_a = last_collision_debug_->object_a;
            escape_frame_b = last_collision_debug_->object_b;
            use_normal_escape = true;
          }
        } else {
          auto global_debug = evaluate_collision_debug(q);
          if (global_debug.has_value() && std::isfinite(global_debug->distance) &&
              global_debug->distance < escape_activation_threshold) {
            global_min_dist = global_debug->distance;
            Eigen::Vector3d diff = global_debug->point_b_world - global_debug->point_a_world;
            double diff_norm = diff.norm();
            if (diff_norm > kCollisionEscapeNormEps) {
              escape_normal = diff / diff_norm;
            } else {
              escape_normal = Eigen::Vector3d(0, 0, 1);
            }
            escape_frame_a = global_debug->object_a;
            escape_frame_b = global_debug->object_b;
            use_normal_escape = true;
          }
        }
      }

      if (use_jacobian_escape) {
        auto coll_result = compute_collision_constraint();
        if (coll_result.has_value() &&
            worst_row < coll_result->jacobian.rows()) {
          Eigen::RowVectorXd n_row = coll_result->jacobian.row(worst_row);
          double nn = n_row.squaredNorm();
          if (nn > kCollisionEscapeNormEps) {
            double escape_mag =
                std::min(kCollisionEscapeStepMax, std::abs(min_dist));
            if (step_collision_rejected) {
              escape_mag = std::max(escape_mag, kCollisionEscapeStepMinOnRejection);
            }
            const Eigen::VectorXd dq_unit =
                (1.0 / std::sqrt(nn)) * n_row.transpose();
            auto try_signed_escape = [&](double sign) -> bool {
              Eigen::VectorXd q_candidate = pinocchio::integrate(
                  robot_->model(), q, sign * escape_mag * dq_unit);
              auto [q_min, q_max] = robot_->get_joint_limits();
              for (int j = 0; j < static_cast<int>(q_candidate.size()); ++j) {
                if (j < q_min.size()) {
                  q_candidate[j] = std::max(q_candidate[j], q_min[j]);
                  q_candidate[j] = std::min(q_candidate[j], q_max[j]);
                }
              }
              robot_->update_configuration(q_candidate);
              auto esc_dist = evaluate_post_step_collision_distance(q_candidate);
              if (esc_dist.has_value() && std::isfinite(*esc_dist) &&
                  *esc_dist > min_dist) {
                q = q_candidate;
                result.stall_escape_count++;
                return true;
              }
              robot_->update_configuration(q);
              return false;
            };
            if (!try_signed_escape(+1.0)) {
              try_signed_escape(-1.0);
            }
          }
        }
      } else if (use_normal_escape) {
        // Use contact normal from the globally-closest penetrating pair.
        // Build a body-level Jacobian for the link owning object_a, project
        // the Cartesian escape direction into joint space.
        const auto *collision_model = robot_->collision_model();
        auto object_to_frame_name = [&](const std::string &obj_name) -> std::string {
          if (!collision_model) return std::string();
          for (std::size_t gi = 0; gi < collision_model->ngeoms; ++gi) {
            const auto &go = collision_model->geometryObjects[gi];
            if (go.name == obj_name) {
              return robot_->model().frames[go.parentFrame].name;
            }
          }
          return std::string();
        };
        const std::string escape_frame_a_name = object_to_frame_name(escape_frame_a);
        const std::string escape_frame_b_name = object_to_frame_name(escape_frame_b);

        auto try_frame_jacobian_escape = [&](const std::string &frame_name) -> bool {
          if (frame_name.empty() || !robot_->has_frame(frame_name)) return false;

          pinocchio::Data &data = robot_->data();
          pinocchio::computeJointJacobians(robot_->model(), data, q);
          auto frame_id = robot_->model().getFrameId(frame_name);
          Eigen::MatrixXd J(6, robot_->nv());
          J.setZero();
          pinocchio::getFrameJacobian(robot_->model(), data, frame_id,
                                      pinocchio::LOCAL_WORLD_ALIGNED, J);
          Eigen::MatrixXd J_lin = J.topRows<3>();
          Eigen::VectorXd dq_escape = J_lin.transpose() * escape_normal;
          double dq_n = dq_escape.norm();
          if (dq_n < kCollisionEscapeNormEps) return false;
          double escape_mag =
              std::min(kCollisionEscapeStepMax, std::abs(global_min_dist));
          if (step_collision_rejected) {
            escape_mag = std::max(escape_mag, kCollisionEscapeStepMinOnRejection);
          }
          dq_escape = (escape_mag / dq_n) * dq_escape;

          Eigen::VectorXd q_candidate = pinocchio::integrate(robot_->model(), q, dq_escape);
          auto [q_min, q_max] = robot_->get_joint_limits();
          for (int j = 0; j < static_cast<int>(q_candidate.size()); ++j) {
            if (j < q_min.size()) {
              q_candidate[j] = std::max(q_candidate[j], q_min[j]);
              q_candidate[j] = std::min(q_candidate[j], q_max[j]);
            }
          }
          robot_->update_configuration(q_candidate);
          auto esc_dist = evaluate_post_step_collision_distance(q_candidate);
          if (esc_dist.has_value() && std::isfinite(*esc_dist) &&
              *esc_dist > global_min_dist) {
            q = q_candidate;
            result.stall_escape_count++;
            return true;
          }
          robot_->update_configuration(q);
          return false;
        };

        if (!try_frame_jacobian_escape(escape_frame_b_name)) {
          try_frame_jacobian_escape(escape_frame_a_name);
        }
      }
    }

    // Limit-dominated lock breaker: if the QP reports near-zero joint motion
    // while several joints are saturated, nudge those joints slightly away
    // from hard limits to recover feasible directions.
    const bool collapsed_primary_scale =
        !last_vel_result.task_scales.empty() &&
        std::abs(last_vel_result.task_scales[0]) <= 1e-6;
    const bool plateau_status =
        last_vel_result.status == SolverStatus::kNumericalError ||
        last_vel_result.status == SolverStatus::kInfeasible ||
        collapsed_primary_scale;
    const int plateau_stall_steps =
        stall_handler_enabled() ? stall_state_.consecutive_stall_steps : 0;
    const bool plateau_escape =
        plateau_status &&
        plateau_stall_steps >= kJointLimitDesaturationPlateauThreshold;
    std::vector<int> desaturation_candidates = last_vel_result.saturated_joints;
    auto [q_min, q_max] = robot_->get_joint_limits();
    if (plateau_escape) {
      for (int vi = 0;
           vi < static_cast<int>(velocity_to_config_index.size()); ++vi) {
        if (std::find(desaturation_candidates.begin(),
                      desaturation_candidates.end(),
                      vi) != desaturation_candidates.end()) {
          continue;
        }
        const int qi = velocity_to_config_index[vi];
        if (qi < 0 || qi >= static_cast<int>(q.size()) ||
            qi >= static_cast<int>(q_min.size()) ||
            qi >= static_cast<int>(q_max.size())) {
          continue;
        }
        const double margin_low = q[qi] - q_min[qi];
        const double margin_up = q_max[qi] - q[qi];
        if ((margin_low >= 0.0 &&
             margin_low < kJointLimitDesaturationExpandedMargin) ||
            (margin_up >= 0.0 &&
             margin_up < kJointLimitDesaturationExpandedMargin)) {
          desaturation_candidates.push_back(vi);
        }
      }
    }
    if (effective_step_dq_norm < stall_config_.dq_stall_eps &&
        (desaturation_candidates.size() >= 4 ||
         (plateau_escape && !desaturation_candidates.empty()))) {
      Eigen::VectorXd q_candidate = q;
      bool changed = false;
      const double desaturation_margin =
          plateau_escape ? kJointLimitDesaturationExpandedMargin
                         : kJointLimitDesaturationMargin;
      const double desaturation_step =
          plateau_escape ? kJointLimitDesaturationBoostStep
                         : kJointLimitDesaturationStep;
      for (int vi : desaturation_candidates) {
        if (vi < 0 || vi >= static_cast<int>(velocity_to_config_index.size())) {
          continue;
        }
        const int qi = velocity_to_config_index[vi];
        if (qi < 0 || qi >= static_cast<int>(q_candidate.size()) ||
            qi >= static_cast<int>(q_min.size()) ||
            qi >= static_cast<int>(q_max.size())) {
          continue;
        }
        const double margin_low = q_candidate[qi] - q_min[qi];
        const double margin_up = q_max[qi] - q_candidate[qi];
        if (margin_low >= 0.0 && margin_low < desaturation_margin) {
          q_candidate[qi] += desaturation_step;
          changed = true;
        } else if (margin_up >= 0.0 && margin_up < desaturation_margin) {
          q_candidate[qi] -= desaturation_step;
          changed = true;
        }
      }
      if (changed) {
        for (int j = 0; j < static_cast<int>(q_candidate.size()); ++j) {
          if (j < q_min.size()) {
            q_candidate[j] = std::max(q_candidate[j], q_min[j]);
            q_candidate[j] = std::min(q_candidate[j], q_max[j]);
          }
        }
        auto curr_dist = evaluate_post_step_collision_distance(q);
        robot_->update_configuration(q_candidate);
        auto cand_dist = evaluate_post_step_collision_distance(q_candidate);
        const bool safe_candidate =
            (!cand_dist.has_value() || !std::isfinite(*cand_dist) ||
             *cand_dist >= kCollisionPenetrationDistanceThreshold);
        const bool not_worse = (!curr_dist.has_value() || !cand_dist.has_value() ||
                                !std::isfinite(*curr_dist) ||
                                !std::isfinite(*cand_dist) ||
                                *cand_dist >=
                                    *curr_dist - kCollisionPenetrationWorsenTolerance);
        if (safe_candidate && not_worse) {
          q = q_candidate;
          result.stall_escape_count++;
        } else {
          robot_->update_configuration(q);
        }
      }
    }
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
    if (last_vel_result.joint_velocities.size() == robot_->nv()) {
      last_vel_result.solution.assign(last_vel_result.joint_velocities.data(),
                                      last_vel_result.joint_velocities.data() +
                                          last_vel_result.joint_velocities.size());
    }
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
  if (no_progress_exit && result.status != SolverStatus::kInvalidInput) {
    result.status = SolverStatus::kNoProgress;
    result.status_message =
        "solve_position_step exited due to no progress near active bounds/"
        "constraints";
  }

  // In teleop-style loops (max_steps=1), stall handling must accumulate across
  // successive solve_position_step calls based on *applied* motion, not only
  // raw QP status.  This post-step update prevents success/no-progress
  // oscillations from resetting the consecutive stall counter.
  if (stall_handler_enabled()) {
    const double applied_step_norm = (result.q_solution - current_q).norm();
    if (applied_step_norm < stall_config_.dq_stall_eps) {
      stall_state_.consecutive_stall_steps++;
      stall_state_.total_stall_steps++;
    } else {
      stall_state_.consecutive_stall_steps = 0;
    }
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
