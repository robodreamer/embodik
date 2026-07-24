/**
 * @file kinematics_solver_internal.hpp
 * @brief Source-private helpers shared by KinematicsSolver implementation units
 */

#pragma once

#include <Eigen/Geometry>
#include <Eigen/SVD>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <exception>
#include <iostream>
#include <limits>
#include <numeric>
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
#include <embodik/weighted_advisor.hpp>

#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"
#endif

namespace embodik {

namespace {
constexpr double kCollisionTolerance = 1e-4;
constexpr double kCollisionUpperDistance = 1e1;
constexpr double kDistanceEpsilon = 1e-9;
constexpr double kCollisionMaxSeparationSpeed = 0.5;
constexpr double kCollisionMaxSeparationSpeedNonPenetration = 0.15;
constexpr double kCollisionPairSwitchHysteresis = 2e-3;
constexpr double kCollisionRepulsionDeadband = 3e-3;
// kCollisionViolationDeadband removed: recovery ramp now activates at the
// exact min_distance boundary so violations never get zero recovery force.
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
constexpr double kJointLimitDesaturationBoostStep = 5e-3;
constexpr int kJointLimitDesaturationPlateauThreshold = 3;
// Conservative bound gate constants (Proxima-inspired).
constexpr double kCollisionBoundRotationRadius = 1.5; // meters
constexpr double kCollisionBoundSafetyMargin = 5e-3; // meters
// Post-step rejection: safe margin above penetration threshold for early-exit.
constexpr double kPostStepSafeMargin = 0.01; // 1cm
constexpr double kPositionStepBacktrackGainScale = 0.1;
// Lazy constraint reuse: skip recomputation when dq is tiny and distance is safe.
constexpr double kLazyReuseMaxDqSqNorm = 1e-6;  // ~0.001 rad change
constexpr double kLazyReuseMinDistMargin = 0.005; // 5mm safety margin
constexpr double kSoftInfeasibleScaleHoldThreshold = 1e-4;
constexpr double kSoftInfeasiblePositionErrorThreshold = 5e-2;
constexpr double kSoftInfeasibleOrientationErrorThreshold = 2.5e-1;
constexpr double kSoftInfeasibleMotionThreshold = 1e-2;
constexpr double kSoftInfeasibleMinImprovement = 2e-3;
constexpr double kSoftInfeasibleRelativeImprovement = 1e-2;
constexpr double kPositionStepSatisfiedMeritTolerance = 2e-3;
constexpr double kPositionStepMeritMotionThreshold = 1e-12;
constexpr int kPositionStepPriorityBacktrackIterations = 12;
constexpr int kPositionStepPriorityRetryIterations = 8;
constexpr double kPositionStepMinAbsoluteErrorReduction = 1e-5;
constexpr double kPositionStepMinErrorReductionPerConfiguration = 2e-3;
constexpr double kStationaryMinErrorReductionPerCall = 1e-6;
constexpr double kStationaryOscillationMinErrorReductionPerConfiguration =
    5e-2;
constexpr double kStationaryDirectionChangeMotionThreshold = 1e-4;
// A single reversal or dominant-target regression is enough to latch an
// exhausted stationary target unless the same window made strong net progress.
constexpr int kStationaryMaxDirectionReversals = 0;
constexpr double kStationaryTargetTranslationTolerance = 1e-9;
constexpr double kStationaryTargetRotationTolerance = 1e-9;
// Target deltas below one micrometer are not a distinct physical command for
// this velocity-level solver. Transition predictor ownership smoothly across
// that numerical-resolution band instead of turning target noise into a mode
// switch, while preserving the established moving-target path above it.
constexpr double kTargetTranslationPredictionTransition = 1e-6;
constexpr double kStationaryTargetGainTolerance = 1e-12;
constexpr int kStationaryTargetDwellCalls = 20;
constexpr double kStationaryTargetMeritRegressionTolerance = 1e-4;

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
constexpr double kPositionLimitMarginEpsilon = 1e-4;

struct HalfspaceBoundResult {
  Eigen::VectorXd lower;
  Eigen::VectorXd upper;
  Eigen::ArrayXi violated_rows;
};

struct ClassifiedOutcome {
  SolverStatus status;
  std::string status_message;
};

struct PositionStepTargetErrorSummary {
  double max_position_error = std::numeric_limits<double>::infinity();
  double max_orientation_error = 0.0;
};

double discrete_stopping_velocity_limit(double margin, double acceleration,
                                        double dt) {
  const double margin_safe = std::max(0.0, margin);
  const double dt_safe = std::max(dt, 1e-9);
  if (margin_safe <= 0.0) {
    return 0.0;
  }
  if (!std::isfinite(acceleration) || acceleration <= 0.0) {
    return margin_safe / dt_safe;
  }

  // For n sampled braking intervals and x=v/(a*dt), the exact stopping
  // distance is a*dt^2*(n*x - n*(n-1)/2), where n=ceil(x).
  const double normalized_margin =
      margin_safe / (acceleration * dt_safe * dt_safe);
  const double root =
      0.5 * (std::sqrt(1.0 + 8.0 * normalized_margin) - 1.0);
  const double interval_count =
      std::max(1.0, std::ceil(root - 1e-12));
  const double normalized_velocity =
      (normalized_margin +
       0.5 * interval_count * (interval_count - 1.0)) /
      interval_count;
  return std::min(margin_safe / dt_safe,
                  acceleration * dt_safe * normalized_velocity);
}

struct ScopedPositionStepCallDepth {
  explicit ScopedPositionStepCallDepth(int &depth_in) : depth(depth_in) {
    ++depth;
  }
  ~ScopedPositionStepCallDepth() { --depth; }

  int &depth;
};

class ScopedRobotKinematicsRestore {
 public:
  explicit ScopedRobotKinematicsRestore(RobotModel &robot)
      : robot_(robot), q_(robot.get_current_configuration()),
        v_(robot.get_current_velocity()) {}

  ScopedRobotKinematicsRestore(const ScopedRobotKinematicsRestore &) = delete;
  ScopedRobotKinematicsRestore &
  operator=(const ScopedRobotKinematicsRestore &) = delete;

  ~ScopedRobotKinematicsRestore() noexcept {
    try {
      robot_.update_kinematics(q_, v_);
    } catch (...) {
      std::terminate();
    }
  }

 private:
  RobotModel &robot_;
  Eigen::VectorXd q_;
  Eigen::VectorXd v_;
};

static void sync_position_result_applied_velocity(
    PositionIKResult &result, const pinocchio::Model &model,
    const Eigen::VectorXd &current_q, double outer_dt) {
  if (result.q_solution.size() != current_q.size()) {
    return;
  }
  const double dt_safe = std::max(outer_dt, 1e-9);
  result.joint_velocities =
      pinocchio::difference(model, current_q, result.q_solution) / dt_safe;
  result.solution.assign(result.joint_velocities.data(),
                         result.joint_velocities.data() +
                             result.joint_velocities.size());
}

static bool clamp_configuration_delta_from_reference(
    const pinocchio::Model &model, const Eigen::VectorXd &q_reference,
    Eigen::VectorXd &q, double max_step_norm) {
  if (max_step_norm <= 0.0 || q_reference.size() != model.nq ||
      q.size() != model.nq) {
    return false;
  }
  Eigen::VectorXd dq = pinocchio::difference(model, q_reference, q);
  const double norm = dq.norm();
  if (!std::isfinite(norm) || norm <= max_step_norm || norm <= 1e-12) {
    return false;
  }
  dq *= max_step_norm / norm;
  q = pinocchio::integrate(model, q_reference, dq);
  return true;
}

static bool should_report_position_recovery_success(
    const PositionIKResult &result, const Eigen::VectorXd &current_q,
    double motion_eps) {
  if (result.stall_escape_count <= 0 ||
      result.q_solution.size() != current_q.size()) {
    return false;
  }
  if (result.status == SolverStatus::kSuccess ||
      result.status == SolverStatus::kInvalidInput ||
      result.status == SolverStatus::kCollisionViolated) {
    return false;
  }
  return (result.q_solution - current_q).norm() > motion_eps;
}

static bool is_scale_family_mode(TaskSolveMode mode) {
  return mode == TaskSolveMode::kScale ||
         mode == TaskSolveMode::kScaleElastic;
}

static std::array<double, 2> commanded_frame_block_merits(
    const Eigen::VectorXd &error, TaskType task_type, double position_gain,
    double orientation_gain) {
  if (error.size() == 3) {
    const bool orientation_only = task_type == TaskType::FRAME_ORIENTATION;
    return orientation_only
               ? std::array<double, 2>{
                     0.0, orientation_gain > 0.0 ? error.head<3>().norm() : 0.0}
               : std::array<double, 2>{
                     position_gain > 0.0 ? error.head<3>().norm() : 0.0, 0.0};
  }
  if (error.size() < 6) {
    const double nan = std::numeric_limits<double>::quiet_NaN();
    return {nan, nan};
  }
  return {position_gain > 0.0 ? error.head<3>().norm() : 0.0,
          orientation_gain > 0.0 ? error.tail<3>().norm() : 0.0};
}

static bool all_frame_task_blocks_commanded(
    const Eigen::VectorXd &error, TaskType task_type, double position_gain,
    double orientation_gain) {
  if (error.size() == 3) {
    const bool orientation_only = task_type == TaskType::FRAME_ORIENTATION;
    const double gain = orientation_only ? orientation_gain : position_gain;
    return gain > 0.0 &&
           error.head<3>().squaredNorm() > kCollisionEscapeNormEps;
  }
  if (error.size() < 6) {
    return false;
  }
  return position_gain > 0.0 && orientation_gain > 0.0 &&
         error.head<3>().squaredNorm() > kCollisionEscapeNormEps &&
         error.tail<3>().squaredNorm() > kCollisionEscapeNormEps;
}

static double commanded_frame_merit(
    const Eigen::VectorXd &error, TaskType task_type, double position_gain,
    double orientation_gain) {
  const auto block_merits = commanded_frame_block_merits(
      error, task_type, position_gain, orientation_gain);
  return block_merits[0] + block_merits[1];
}

template <typename Derived>
static void append_eigen_signature(std::vector<double> &signature,
                                   const Eigen::MatrixBase<Derived> &values) {
  signature.push_back(static_cast<double>(values.rows()));
  signature.push_back(static_cast<double>(values.cols()));
  signature.insert(signature.end(), values.derived().data(),
                   values.derived().data() + values.size());
}

template <typename MatrixType>
static void append_optional_eigen_signature(
    std::vector<double> &signature,
    const std::optional<MatrixType> &optional_values) {
  signature.push_back(optional_values.has_value() ? 1.0 : 0.0);
  if (optional_values.has_value()) {
    append_eigen_signature(signature, *optional_values);
  }
}

static void append_index_signature(std::vector<double> &signature,
                                   const std::vector<int> &indices) {
  signature.push_back(static_cast<double>(indices.size()));
  for (int index : indices) {
    signature.push_back(static_cast<double>(index));
  }
}

static void append_position_step_option_signature(
    std::vector<double> &signature, const PositionStepOptions &options) {
  signature.insert(
      signature.end(),
      {options.position_gain,
       options.orientation_gain,
       static_cast<double>(options.max_steps),
       options.dt,
       options.max_linear_speed,
       options.max_angular_speed,
       options.max_configuration_step_norm,
       options.adaptive_dt ? 1.0 : 0.0,
       options.adaptive_dt_max_scale,
       options.adaptive_dt_reference_distance,
       static_cast<double>(options.primary_solve_mode),
       options.primary_allow_min_error_fallback ? 1.0 : 0.0,
       options.stall_recovery ? 1.0 : 0.0,
       options.elastic_band ? 1.0 : 0.0,
       options.limit_change_from_seed ? 1.0 : 0.0,
       static_cast<double>(options.no_progress_max_steps),
       options.no_progress_error_tolerance,
       options.no_progress_dq_norm_tolerance,
       static_cast<double>(options.preferred_lock_solve_mode),
       options.preferred_lock_tracking_tolerance,
       options.preferred_lock_orientation_tolerance,
       options.preferred_lock_max_step_norm,
       options.preferred_lock_min_error_reduction_ratio,
       options.torso_constraint.enabled ? 1.0 : 0.0,
       options.torso_constraint.orientation_gain,
       options.torso_constraint.pose_bound_softening_enabled ? 1.0 : 0.0,
       options.torso_constraint.pose_bound_softening_fraction,
       options.torso_constraint.velocity_box_headroom.enabled ? 1.0 : 0.0,
       options.torso_constraint.velocity_box_headroom.fraction,
       options.torso_constraint.velocity_box_headroom.activation_margin});
  append_optional_eigen_signature(
      signature, options.torso_constraint.target_orientation);
  append_eigen_signature(signature,
                         options.torso_constraint.orientation_mask);
  append_optional_eigen_signature(
      signature, options.torso_constraint.pose_bounds_reference_pose);
  append_optional_eigen_signature(
      signature, options.torso_constraint.pose_lower_bounds);
  append_optional_eigen_signature(
      signature, options.torso_constraint.pose_upper_bounds);
  append_eigen_signature(signature, options.torso_constraint.pose_axis_mask);
  append_eigen_signature(signature, options.torso_constraint.velocity_limits);
  append_eigen_signature(signature,
                         options.torso_constraint.acceleration_limits);
  append_index_signature(signature, options.excluded_joint_indices);
  append_index_signature(signature, options.locked_joint_indices);
  append_index_signature(signature,
                         options.integration_zero_velocity_indices);
  append_index_signature(signature, options.preferred_locked_joint_indices);
}

static void append_task_policy_signature(std::vector<double> &signature,
                                         const Task &task,
                                         bool include_target_revision) {
  signature.insert(signature.end(),
                   {static_cast<double>(task.getType()),
                    static_cast<double>(task.getPriority()), task.getWeight(),
                    task.isActive() ? 1.0 : 0.0,
                    static_cast<double>(task.getSolveMode()),
                    task.getAllowMinErrorFallback() ? 1.0 : 0.0,
                    static_cast<double>(task.getContinuityRevision()),
                    include_target_revision
                        ? static_cast<double>(task.getContinuityTargetRevision())
                        : 0.0});
  append_index_signature(signature, task.get_excluded_joint_indices());
}

static bool has_sufficient_position_step_merit_reduction(
    double initial_merit, double final_merit, double configuration_step_norm,
    double min_reduction_per_configuration =
        kPositionStepMinErrorReductionPerConfiguration,
    double min_absolute_reduction =
        kPositionStepMinAbsoluteErrorReduction) {
  if (!std::isfinite(initial_merit) || !std::isfinite(final_merit) ||
      !std::isfinite(configuration_step_norm) ||
      configuration_step_norm <= kPositionStepMeritMotionThreshold) {
    return false;
  }
  const double numerical_tolerance =
      64.0 * std::numeric_limits<double>::epsilon() *
      std::max({1.0, initial_merit, final_merit});
  const double required_reduction = std::max(
      {numerical_tolerance, min_absolute_reduction,
       min_reduction_per_configuration * configuration_step_norm});
  return initial_merit - final_merit >= required_reduction;
}

static bool should_hold_non_improving_position_step(
    const PositionIKResult &result, const Eigen::VectorXd &current_q,
    double initial_commanded_error, double final_commanded_error,
    double configuration_step_norm, bool collision_violated) {
  const bool candidate_status = result.status == SolverStatus::kSuccess ||
                                result.status == SolverStatus::kNoProgress;
  if (!candidate_status || collision_violated ||
      result.collision_rejection_count > 0 ||
      result.stall_escape_count > 0 ||
      result.q_solution.size() != current_q.size()) {
    return false;
  }
  if (!std::isfinite(initial_commanded_error) ||
      !std::isfinite(final_commanded_error)) {
    return false;
  }
  if (!std::isfinite(configuration_step_norm) ||
      configuration_step_norm <= kPositionStepMeritMotionThreshold) {
    return false;
  }
  if (initial_commanded_error <= kPositionStepSatisfiedMeritTolerance) {
    return true;
  }

  return !has_sufficient_position_step_merit_reduction(
      initial_commanded_error, final_commanded_error,
      configuration_step_norm);
}

static bool should_hold_soft_infeasible_position_step(
    const PositionIKResult &result, const Eigen::VectorXd &current_q,
    double initial_combined_error, double final_combined_error,
    double final_position_error, double final_orientation_error,
    bool collision_violated,
    bool recovery_inside_collision_margin) {
  const bool primary_scale_collapsed_infeasible =
      result.status == SolverStatus::kInfeasible &&
      result.status_message.find("primary task scale collapsed") !=
          std::string::npos;
  const bool primary_scale_collapsed_numerical =
      result.status == SolverStatus::kNumericalError &&
      !result.task_scales.empty() &&
      std::abs(result.task_scales[0]) <= kSoftInfeasibleScaleHoldThreshold;
  if ((result.status != SolverStatus::kSuccess &&
       !primary_scale_collapsed_infeasible &&
       !primary_scale_collapsed_numerical) ||
      collision_violated ||
      recovery_inside_collision_margin || result.stall_escape_count > 0 ||
      result.collision_rejection_count > 0 ||
      result.q_solution.size() != current_q.size()) {
    return false;
  }
  if (result.task_scales.empty() || result.task_modes_effective.empty()) {
    return false;
  }
  if (!is_scale_family_mode(result.task_modes_effective[0])) {
    return false;
  }
  if (!result.task_used_fallback.empty() && result.task_used_fallback[0]) {
    return false;
  }
  if (std::abs(result.task_scales[0]) >
      kSoftInfeasibleScaleHoldThreshold) {
    return false;
  }

  const bool large_residual =
      final_position_error > kSoftInfeasiblePositionErrorThreshold ||
      final_orientation_error > kSoftInfeasibleOrientationErrorThreshold;
  if (!large_residual) {
    return false;
  }

  if ((result.q_solution - current_q).norm() <=
      kSoftInfeasibleMotionThreshold) {
    return false;
  }

  if (!std::isfinite(initial_combined_error) ||
      !std::isfinite(final_combined_error) ||
      initial_combined_error <= 0.0) {
    return false;
  }
  const double improvement = initial_combined_error - final_combined_error;
  const double required_improvement =
      std::max(kSoftInfeasibleMinImprovement,
               kSoftInfeasibleRelativeImprovement * initial_combined_error);
  return improvement < required_improvement;
}

static bool collision_recovery_candidate_acceptable(
    const std::optional<double> &current_dist,
    const std::optional<double> &candidate_dist, double safe_threshold) {
  if (!candidate_dist.has_value() || !std::isfinite(*candidate_dist)) {
    return true;
  }
  if (!current_dist.has_value() || !std::isfinite(*current_dist)) {
    return *candidate_dist >= safe_threshold;
  }
  if (*current_dist >= safe_threshold) {
    return *candidate_dist >= safe_threshold;
  }
  return *candidate_dist >=
         *current_dist - kCollisionPenetrationWorsenTolerance;
}

static bool collision_recovery_margins_acceptable(
    const std::vector<double> &current_margins,
    const std::vector<double> &candidate_margins, double safe_threshold) {
  const std::size_t pair_count =
      std::max(current_margins.size(), candidate_margins.size());
  for (std::size_t pair_index = 0; pair_index < pair_count; ++pair_index) {
    const double current_margin =
        pair_index < current_margins.size()
            ? current_margins[pair_index]
            : std::numeric_limits<double>::quiet_NaN();
    const double candidate_margin =
        pair_index < candidate_margins.size()
            ? candidate_margins[pair_index]
            : std::numeric_limits<double>::quiet_NaN();
    if (!std::isfinite(candidate_margin)) {
      if (std::isfinite(current_margin)) {
        return false;
      }
      continue;
    }
    if (!std::isfinite(current_margin)) {
      if (candidate_margin < safe_threshold) {
        return false;
      }
      continue;
    }
    if (current_margin >= safe_threshold) {
      if (candidate_margin < safe_threshold) {
        return false;
      }
      continue;
    }
    if (candidate_margin <
        current_margin - kCollisionPenetrationWorsenTolerance) {
      return false;
    }
  }
  return true;
}

static bool collision_recovery_margins_are_safe(
    const std::vector<double> &margins, double safe_threshold) {
  bool found = false;
  for (double margin : margins) {
    if (!std::isfinite(margin)) {
      continue;
    }
    found = true;
    if (margin < safe_threshold) {
      return false;
    }
  }
  return found;
}

static double compute_adaptive_position_step_dt(
    const PositionStepOptions &options, double step_dt, double position_error,
    bool collision_constraint_enabled,
    double last_constraint_min_recovery_margin) {
  if (!options.adaptive_dt ||
      options.adaptive_dt_reference_distance <= 1e-9 ||
      options.adaptive_dt_max_scale <= 1.0 ||
      !std::isfinite(position_error) || position_error <= 0.0) {
    return step_dt;
  }

  double scale = position_error / options.adaptive_dt_reference_distance;
  scale = std::min(scale, options.adaptive_dt_max_scale);
  scale = std::max(scale, 1.0);

  if (collision_constraint_enabled &&
      std::isfinite(last_constraint_min_recovery_margin) &&
      last_constraint_min_recovery_margin < -kCollisionTolerance) {
    // During a true recovery-floor violation, keep the nominal integration
    // horizon. At and above the floor, directional QP rows and exact post-step
    // validation constrain normal approach without throttling tangential motion.
    scale = 1.0;
  }

  return step_dt * scale;
}

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

static bool project_objectives_into_constraint_tangent_space(
    std::vector<Eigen::VectorXd> &task_goals,
    std::vector<Eigen::MatrixXd> &task_jacobians,
    const std::vector<bool> &project_objective,
    const Eigen::MatrixXd &constraint_jacobian,
    const Eigen::VectorXd &constraint_lower_bounds, double tolerance,
    const std::vector<bool> &prefer_exact_feasibility_rows,
    double exact_feasibility_singularity_threshold,
    bool use_closest_achievable_goal,
    std::vector<Eigen::VectorXd> *weighted_goals,
    std::vector<Eigen::MatrixXd> *weighted_jacobians) {
  if (constraint_jacobian.rows() != constraint_lower_bounds.size()) {
    return false;
  }

  const double feasibility_tolerance = std::max(1e-10, tolerance);
  bool projection_applied = false;
  for (std::size_t task_idx = 0;
       task_idx < task_jacobians.size() && task_idx < task_goals.size();
       ++task_idx) {
    if (task_idx >= project_objective.size() ||
        !project_objective[task_idx]) {
      continue;
    }
    const Eigen::MatrixXd original_jacobian = task_jacobians[task_idx];
    const Eigen::VectorXd original_goal = task_goals[task_idx];
    if (original_jacobian.rows() != original_goal.size() ||
        original_jacobian.cols() != constraint_jacobian.cols() ||
        original_jacobian.rows() == 0) {
      continue;
    }

    Eigen::MatrixXd objective_inverse;
    detail::ComputeGeneralizedInverse(original_jacobian,
                                      feasibility_tolerance,
                                      &objective_inverse);
    const Eigen::VectorXd desired_velocity = objective_inverse * original_goal;
    if (!desired_velocity.allFinite()) {
      continue;
    }

    bool has_violated_row = false;
    bool prefer_exact_feasibility =
        prefer_exact_feasibility_rows.size() ==
        static_cast<std::size_t>(constraint_jacobian.rows());
    for (int row = 0; row < constraint_jacobian.rows(); ++row) {
      if (constraint_jacobian.row(row).dot(desired_velocity) <
          constraint_lower_bounds(row) - feasibility_tolerance) {
        has_violated_row = true;
        prefer_exact_feasibility =
            prefer_exact_feasibility &&
            prefer_exact_feasibility_rows[static_cast<std::size_t>(row)];
      }
    }
    if (!has_violated_row) {
      continue;
    }
    if (std::isfinite(exact_feasibility_singularity_threshold) &&
        exact_feasibility_singularity_threshold > 0.0) {
      const Eigen::VectorXd singular_values =
          Eigen::JacobiSVD<Eigen::MatrixXd>(original_jacobian)
              .singularValues();
      if (singular_values.size() > 0) {
        const double normalized_minimum =
            singular_values(singular_values.size() - 1) /
            std::max(singular_values(0), 1e-12);
        prefer_exact_feasibility =
            prefer_exact_feasibility ||
            normalized_minimum <= exact_feasibility_singularity_threshold;
      }
    }

    // At the sampled-data boundary, a minimum-norm task velocity can point out
    // of a hard half-space even when an equivalent Cartesian velocity exists
    // in the task nullspace. Prefer that exact inward/tangent realization
    // before removing any task component. Interior activation rows instead
    // preserve their current margin through the tangent objective. The hard
    // rows still constrain all lower-priority objectives.
    if (prefer_exact_feasibility) {
      Eigen::VectorXd feasible_desired_velocity = desired_velocity;
      const Eigen::MatrixXd task_nullspace =
          Eigen::MatrixXd::Identity(original_jacobian.cols(),
                                    original_jacobian.cols()) -
          objective_inverse * original_jacobian;
      const int correction_iterations =
          std::max(1, 4 * static_cast<int>(constraint_jacobian.rows()));
      for (int iteration = 0; iteration < correction_iterations; ++iteration) {
        int most_violated_row = -1;
        double largest_deficit = feasibility_tolerance;
        for (int row = 0; row < constraint_jacobian.rows(); ++row) {
          const double deficit = constraint_lower_bounds(row) -
                                 constraint_jacobian.row(row).dot(
                                     feasible_desired_velocity);
          if (deficit > largest_deficit) {
            most_violated_row = row;
            largest_deficit = deficit;
          }
        }
        if (most_violated_row < 0) {
          break;
        }
        const Eigen::RowVectorXd correction_direction =
            constraint_jacobian.row(most_violated_row) * task_nullspace;
        const double correction_norm_squared =
            correction_direction.squaredNorm();
        if (correction_norm_squared <= 1e-12) {
          break;
        }
        feasible_desired_velocity.noalias() +=
            correction_direction.transpose() *
            (largest_deficit / correction_norm_squared);
      }
      bool exact_goal_is_feasible = true;
      for (int row = 0; row < constraint_jacobian.rows(); ++row) {
        if (constraint_jacobian.row(row).dot(feasible_desired_velocity) <
            constraint_lower_bounds(row) - feasibility_tolerance) {
          exact_goal_is_feasible = false;
          break;
        }
      }
      if (exact_goal_is_feasible) {
        continue;
      }
    }

    Eigen::MatrixXd tangent_projector = Eigen::MatrixXd::Identity(
        original_jacobian.cols(), original_jacobian.cols());
    bool projection_required = false;
    for (int row = 0; row < constraint_jacobian.rows(); ++row) {
      const Eigen::RowVectorXd constraint_row = constraint_jacobian.row(row);
      if (constraint_row.squaredNorm() <= 1e-12) {
        continue;
      }
      const double desired_separation = constraint_row.dot(desired_velocity);
      if (desired_separation >=
          constraint_lower_bounds(row) - feasibility_tolerance) {
        continue;
      }

      const Eigen::RowVectorXd remaining_normal =
          constraint_row * tangent_projector;
      const double remaining_norm_squared = remaining_normal.squaredNorm();
      if (remaining_norm_squared <= 1e-12) {
        continue;
      }
      tangent_projector.noalias() -=
          remaining_normal.transpose() * remaining_normal /
          remaining_norm_squared;
      projection_required = true;
    }

    if (!projection_required) {
      continue;
    }

    // The hard row owns affine recovery. Strict SCALE sees only a homogeneous
    // tangent objective, and its goal is generated through the projected
    // Jacobian so no removed task row can force the scale to collapse.
    const Eigen::MatrixXd projected_jacobian =
        original_jacobian * tangent_projector;
    Eigen::VectorXd projected_goal;
    if (use_closest_achievable_goal) {
      Eigen::MatrixXd projected_inverse;
      detail::ComputeGeneralizedInverse(projected_jacobian,
                                        feasibility_tolerance,
                                        &projected_inverse);
      projected_goal =
          projected_jacobian * projected_inverse * original_goal;
    } else {
      projected_goal = projected_jacobian * desired_velocity;
    }

    // SNS compares the rank remaining after constraint saturation with the
    // objective row count. Remove dependent projected rows so that count is the
    // true tangent-task rank. U_r is orthonormal, so this preserves the task's
    // Euclidean residual norm on its achievable row space.
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(projected_jacobian,
                                          Eigen::ComputeThinU);
    const Eigen::VectorXd singular_values = svd.singularValues();
    const double largest_singular_value =
        singular_values.size() > 0 ? singular_values(0) : 0.0;
    const double rank_threshold =
        feasibility_tolerance *
        static_cast<double>(
            std::max(projected_jacobian.rows(), projected_jacobian.cols())) *
        std::max(1.0, largest_singular_value);
    int rank = 0;
    while (rank < singular_values.size() &&
           singular_values(rank) > rank_threshold) {
      ++rank;
    }
    if (weighted_goals != nullptr && weighted_jacobians != nullptr &&
        weighted_jacobians->empty()) {
      *weighted_goals = task_goals;
      *weighted_jacobians = task_jacobians;
    }
    if (weighted_jacobians != nullptr && !weighted_jacobians->empty()) {
      // Weighted MIN_ERROR keeps the original residual but cannot use the
      // infeasible constraint-normal direction. It does not need strict row-rank
      // compression because dependent rows remain valid soft residuals.
      (*weighted_jacobians)[task_idx] = projected_jacobian;
    }
    projection_applied = true;
    if (rank == 0) {
      continue;
    }
    const Eigen::MatrixXd row_basis = svd.matrixU().leftCols(rank).transpose();
    task_jacobians[task_idx] = row_basis * projected_jacobian;
    task_goals[task_idx] = row_basis * projected_goal;
  }
  return projection_applied;
}

struct ConstraintBlock {
  Eigen::MatrixXd jacobian;
  Eigen::VectorXd lower_bounds;
  Eigen::VectorXd upper_bounds;
  std::vector<bool> prefer_exact_feasibility_rows;
};

static std::optional<ConstraintBlock>
build_joint_limit_non_worsening_rows(
    const RobotModel &robot,
    const std::vector<int> &velocity_to_config_index,
    const std::unordered_set<int> &excluded_indices,
    double activation_margin, bool acceleration_limits_enabled,
    const Eigen::VectorXd &acceleration_limits, double dt,
    std::vector<int> &lower_modes,
    std::vector<int> &upper_modes) {
  if (!std::isfinite(activation_margin) || activation_margin <= 0.0) {
    return std::nullopt;
  }

  const auto [q_lower, q_upper] = robot.get_joint_limits();
  const Eigen::VectorXd q = robot.get_current_configuration();
  const Eigen::VectorXd velocity_limits = robot.get_velocity_limits();
  const int nv = robot.nv();
  if (lower_modes.size() != static_cast<std::size_t>(nv)) {
    lower_modes.assign(static_cast<std::size_t>(nv), 0);
  }
  if (upper_modes.size() != static_cast<std::size_t>(nv)) {
    upper_modes.assign(static_cast<std::size_t>(nv), 0);
  }
  const int floating_base_offset = robot.is_floating_base() ? 6 : 0;
  struct ActiveLimitRow {
    int velocity_index;
    double sign;
    bool at_sampled_data_boundary;
    double outward_speed_limit;
  };
  std::vector<ActiveLimitRow> active_rows;
  active_rows.reserve(static_cast<std::size_t>(nv));

  for (int velocity_index = floating_base_offset; velocity_index < nv;
       ++velocity_index) {
    if (excluded_indices.find(velocity_index) != excluded_indices.end()) {
      lower_modes[static_cast<std::size_t>(velocity_index)] = 0;
      upper_modes[static_cast<std::size_t>(velocity_index)] = 0;
      continue;
    }
    const int config_index =
        velocity_index < static_cast<int>(velocity_to_config_index.size())
            ? velocity_to_config_index[velocity_index]
            : -1;
    if (config_index < 0 || config_index >= q.size() ||
        config_index >= q_lower.size() || config_index >= q_upper.size() ||
        !std::isfinite(q_lower[config_index]) ||
        !std::isfinite(q_upper[config_index])) {
      lower_modes[static_cast<std::size_t>(velocity_index)] = 0;
      upper_modes[static_cast<std::size_t>(velocity_index)] = 0;
      continue;
    }

    const double lower_slack = q[config_index] - q_lower[config_index];
    const double upper_slack = q_upper[config_index] - q[config_index];
    const auto outward_speed_limit = [&](double slack) -> std::optional<double> {
      if (slack <= activation_margin) {
        return 0.0;
      }
      if (!acceleration_limits_enabled ||
          velocity_index >= acceleration_limits.size() ||
          velocity_index >= velocity_limits.size() ||
          !std::isfinite(acceleration_limits[velocity_index]) ||
          acceleration_limits[velocity_index] <= 0.0 ||
          !std::isfinite(velocity_limits[velocity_index]) ||
          velocity_limits[velocity_index] <= 0.0) {
        return std::nullopt;
      }
      const double stopping_limit = discrete_stopping_velocity_limit(
          slack - activation_margin, acceleration_limits[velocity_index], dt);
      if (stopping_limit >= velocity_limits[velocity_index] - 1e-12) {
        return std::nullopt;
      }
      return stopping_limit;
    };

    const std::optional<double> lower_outward_limit =
        outward_speed_limit(lower_slack);
    if (lower_outward_limit.has_value()) {
      int &mode = lower_modes[static_cast<std::size_t>(velocity_index)];
      if (lower_slack > activation_margin) {
        mode = 0;
      } else if (mode == 0 ||
                 lower_slack <= kPositionLimitMarginEpsilon) {
        mode = lower_slack <= kPositionLimitMarginEpsilon ? 2 : 1;
      }
      active_rows.push_back({velocity_index, 1.0,
                             mode == 2, *lower_outward_limit});
    } else {
      lower_modes[static_cast<std::size_t>(velocity_index)] = 0;
    }
    const std::optional<double> upper_outward_limit =
        outward_speed_limit(upper_slack);
    if (upper_outward_limit.has_value()) {
      int &mode = upper_modes[static_cast<std::size_t>(velocity_index)];
      if (upper_slack > activation_margin) {
        mode = 0;
      } else if (mode == 0 ||
                 upper_slack <= kPositionLimitMarginEpsilon) {
        mode = upper_slack <= kPositionLimitMarginEpsilon ? 2 : 1;
      }
      active_rows.push_back({velocity_index, -1.0,
                             mode == 2, *upper_outward_limit});
    } else {
      upper_modes[static_cast<std::size_t>(velocity_index)] = 0;
    }
  }

  if (active_rows.empty()) {
    return std::nullopt;
  }

  ConstraintBlock result;
  result.jacobian = Eigen::MatrixXd::Zero(
      static_cast<int>(active_rows.size()), nv);
  result.lower_bounds =
      Eigen::VectorXd::Zero(static_cast<int>(active_rows.size()));
  result.upper_bounds = Eigen::VectorXd::Constant(
      static_cast<int>(active_rows.size()), kUnboundedConstraintLimit);
  result.prefer_exact_feasibility_rows.reserve(active_rows.size());
  for (int row = 0; row < static_cast<int>(active_rows.size()); ++row) {
    const auto &[velocity_index, sign, at_sampled_data_boundary,
                 outward_speed_limit] =
        active_rows[static_cast<std::size_t>(row)];
    result.jacobian(row, velocity_index) = sign;
    result.lower_bounds(row) = -outward_speed_limit;
    result.prefer_exact_feasibility_rows.push_back(
        at_sampled_data_boundary);
  }
  return result;
}

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

static void apply_constraint_softening_in_place(
    Eigen::VectorXd &lower, Eigen::VectorXd &upper,
    const Eigen::VectorXd &max_softening_factors, double softening_scale = 1.0) {
  if (lower.size() != upper.size() ||
      lower.size() != max_softening_factors.size()) {
    return;
  }
  for (Eigen::Index i = 0; i < lower.size(); ++i) {
    const double max_factor = max_softening_factors(i);
    if (!std::isfinite(max_factor) || max_factor <= 1.0) {
      continue;
    }
    const double base_half_span = 0.5 * (upper(i) - lower(i));
    if (!std::isfinite(base_half_span) || base_half_span <= 0.0) {
      continue;
    }
    const double center = 0.5 * (upper(i) + lower(i));
    const double factor = std::clamp(softening_scale, 1.0, max_factor);
    lower(i) = center - factor * base_half_span;
    upper(i) = center + factor * base_half_span;
  }
}

static bool has_rowwise_interval_feasibility(
    const Eigen::MatrixXd &C, const Eigen::VectorXd &c_lower,
    const Eigen::VectorXd &c_upper, const Eigen::VectorXd &dq_lower_seed,
    const Eigen::VectorXd &dq_upper_seed, double tol) {
  if (C.rows() != c_lower.size() || C.rows() != c_upper.size()) {
    return false;
  }
  if (C.cols() != dq_lower_seed.size() || C.cols() != dq_upper_seed.size()) {
    return false;
  }
  Eigen::VectorXd dq_lower = dq_lower_seed;
  Eigen::VectorXd dq_upper = dq_upper_seed;
  for (int j = 0; j < C.cols(); ++j) {
    for (int r = 0; r < C.rows(); ++r) {
      const double a_j = C(r, j);
      if (std::abs(a_j) <= 1e-12) {
        continue;
      }
      double others_min = 0.0;
      double others_max = 0.0;
      for (int k = 0; k < C.cols(); ++k) {
        if (k == j) {
          continue;
        }
        const double a_k = C(r, k);
        if (std::abs(a_k) <= 1e-12) {
          continue;
        }
        const double lk = dq_lower(k);
        const double uk = dq_upper(k);
        if (!std::isfinite(lk) || !std::isfinite(uk)) {
          continue;
        }
        if (a_k >= 0.0) {
          others_min += a_k * lk;
          others_max += a_k * uk;
        } else {
          others_min += a_k * uk;
          others_max += a_k * lk;
        }
      }

      double local_lower = (c_lower(r) - others_max) / a_j;
      double local_upper = (c_upper(r) - others_min) / a_j;
      if (a_j < 0.0) {
        std::swap(local_lower, local_upper);
      }
      dq_lower(j) = std::max(dq_lower(j), local_lower);
      dq_upper(j) = std::min(dq_upper(j), local_upper);
      if (dq_lower(j) > dq_upper(j) + tol) {
        return false;
      }
    }
  }
  return true;
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

static bool is_primary_scale_collapsed_message(const std::string &message) {
  return message.find("primary task scale collapsed") != std::string::npos;
}

template <typename ResolvedTaskRange>
static bool flip_primary_scale_tasks_to_min_error(
    const ResolvedTaskRange &resolved,
    std::vector<std::pair<Task *, TaskSolveMode>> *saved_modes) {
  saved_modes->clear();
  int min_priority = std::numeric_limits<int>::max();
  for (const auto &rt : resolved) {
    if (rt.task && rt.task->isActive()) {
      min_priority = std::min(min_priority, rt.task->getPriority());
    }
  }
  if (min_priority == std::numeric_limits<int>::max()) {
    return false;
  }
  for (const auto &rt : resolved) {
    Task *task = rt.task.get();
    if (task == nullptr || !task->isActive() ||
        task->getPriority() != min_priority) {
      continue;
    }
    const TaskSolveMode mode = task->getSolveMode();
    if (mode != TaskSolveMode::kScale && mode != TaskSolveMode::kScaleElastic) {
      continue;
    }
    saved_modes->emplace_back(task, mode);
    task->setSolveMode(TaskSolveMode::kMinError);
  }
  return !saved_modes->empty();
}

static void restore_task_solve_modes(
    const std::vector<std::pair<Task *, TaskSolveMode>> &saved_modes) {
  for (const auto &[task, mode] : saved_modes) {
    if (task != nullptr) {
      task->setSolveMode(mode);
    }
  }
}

static bool flip_scale_family_task(
    Task *task, std::vector<std::pair<Task *, TaskSolveMode>> *saved_modes) {
  if (task == nullptr || !task->isActive()) {
    return false;
  }
  const TaskSolveMode mode = task->getSolveMode();
  if (mode != TaskSolveMode::kScale && mode != TaskSolveMode::kScaleElastic) {
    return false;
  }
  saved_modes->emplace_back(task, mode);
  task->setSolveMode(TaskSolveMode::kMinError);
  return true;
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
  if (!validate_nv_index_list(options.preferred_locked_joint_indices, nv,
                              "preferred_locked_joint_indices", err)) {
    return false;
  }
  if (!std::isfinite(options.max_configuration_step_norm) ||
      options.max_configuration_step_norm < 0.0) {
    if (err != nullptr) {
      *err = "max_configuration_step_norm must be finite and >= 0";
    }
    return false;
  }
  if (!std::isfinite(options.preferred_lock_tracking_tolerance) ||
      options.preferred_lock_tracking_tolerance < 0.0) {
    if (err != nullptr) {
      *err = "preferred_lock_tracking_tolerance must be finite and >= 0";
    }
    return false;
  }
  if (!std::isfinite(options.preferred_lock_orientation_tolerance)) {
    if (err != nullptr) {
      *err = "preferred_lock_orientation_tolerance must be finite";
    }
    return false;
  }
  if (!std::isfinite(options.preferred_lock_max_step_norm)) {
    if (err != nullptr) {
      *err = "preferred_lock_max_step_norm must be finite";
    }
    return false;
  }
  if (!std::isfinite(options.preferred_lock_min_error_reduction_ratio) ||
      options.preferred_lock_min_error_reduction_ratio < 0.0) {
    if (err != nullptr) {
      *err =
          "preferred_lock_min_error_reduction_ratio must be finite and >= 0";
    }
    return false;
  }
  return true;
}

static bool validate_linear_constraint_triplet(const Eigen::MatrixXd &C,
                                               const Eigen::VectorXd &lower,
                                               const Eigen::VectorXd &upper,
                                               int nv,
                                               std::string *out_message) {
  if (C.cols() != nv) {
    if (out_message != nullptr) {
      *out_message = "linear constraint matrix C must have " +
                     std::to_string(nv) + " columns (nv); got " +
                     std::to_string(C.cols());
    }
    return false;
  }
  if (C.rows() != lower.size() || C.rows() != upper.size()) {
    if (out_message != nullptr) {
      *out_message =
          "linear constraint bounds must match C rows; got C.rows=" +
          std::to_string(C.rows()) + ", lower.size=" +
          std::to_string(lower.size()) + ", upper.size=" +
          std::to_string(upper.size());
    }
    return false;
  }
  if (!C.allFinite() || !lower.allFinite() || !upper.allFinite()) {
    if (out_message != nullptr) {
      *out_message =
          "linear constraints require finite C/lower_bounds/upper_bounds";
    }
    return false;
  }
  for (int i = 0; i < lower.size(); ++i) {
    if (lower(i) > upper(i)) {
      if (out_message != nullptr) {
        *out_message = "linear constraint lower_bounds[" + std::to_string(i) +
                       "] > upper_bounds[" + std::to_string(i) + "]";
      }
      return false;
    }
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

static void append_unique_indices(std::vector<int> &dst,
                                  const std::vector<int> &src) {
  dst.insert(dst.end(), src.begin(), src.end());
  std::sort(dst.begin(), dst.end());
  dst.erase(std::unique(dst.begin(), dst.end()), dst.end());
}

static PositionStepOptions
make_preferred_lock_candidate_options(const PositionStepOptions &options) {
  PositionStepOptions candidate = options;
  append_unique_indices(candidate.locked_joint_indices,
                        options.preferred_locked_joint_indices);
  append_unique_indices(candidate.integration_zero_velocity_indices,
                        options.preferred_locked_joint_indices);
  candidate.preferred_locked_joint_indices.clear();
  candidate.primary_solve_mode = options.preferred_lock_solve_mode;
  candidate.primary_allow_min_error_fallback = false;
  return candidate;
}

static PositionStepOptions
make_preferred_lock_continuity_options(const PositionStepOptions &options) {
  PositionStepOptions bounded = options;
  bounded.max_steps = 1;
  return bounded;
}

static bool preferred_lock_status_acceptable(SolverStatus status) {
  return status == SolverStatus::kSuccess || status == SolverStatus::kNoProgress;
}

static bool preferred_lock_error_component_acceptable(
    double entry_error, double candidate_error, double tolerance,
    double min_reduction_ratio) {
  if (std::isfinite(candidate_error) && candidate_error <= tolerance) {
    return true;
  }
  if (!std::isfinite(entry_error) || !std::isfinite(candidate_error) ||
      entry_error <= tolerance || candidate_error >= entry_error) {
    return false;
  }
  constexpr double kMinAbsoluteReduction = 1e-4;
  const double required_reduction =
      std::max(kMinAbsoluteReduction, entry_error * min_reduction_ratio);
  return (entry_error - candidate_error) >= required_reduction;
}

static bool preferred_lock_candidate_acceptable(
    const PositionIKResult &candidate,
    const PositionStepTargetErrorSummary &entry_errors,
    const PositionStepTargetErrorSummary &candidate_errors, double step_norm,
    const PositionStepOptions &options) {
  if (!preferred_lock_status_acceptable(candidate.status) ||
      candidate.q_solution.size() == 0 || !candidate.q_solution.allFinite()) {
    return false;
  }
  if (!preferred_lock_error_component_acceptable(
          entry_errors.max_position_error, candidate_errors.max_position_error,
          options.preferred_lock_tracking_tolerance,
          options.preferred_lock_min_error_reduction_ratio)) {
    return false;
  }
  if (options.preferred_lock_orientation_tolerance > 0.0 &&
      !preferred_lock_error_component_acceptable(
          entry_errors.max_orientation_error,
          candidate_errors.max_orientation_error,
          options.preferred_lock_orientation_tolerance,
          options.preferred_lock_min_error_reduction_ratio)) {
    return false;
  }
  if (options.preferred_lock_max_step_norm > 0.0 &&
      (!std::isfinite(step_norm) ||
       step_norm > options.preferred_lock_max_step_norm)) {
    return false;
  }
  return true;
}

static void annotate_preferred_lock_result(
    PositionIKResult &result, const PositionIKResult &candidate,
    const PositionStepTargetErrorSummary &errors, double step_norm,
    double candidate_time_ms, bool used) {
  result.preferred_lock_attempted = true;
  result.preferred_lock_used = used;
  result.preferred_lock_fallback_used = !used;
  result.preferred_lock_candidate_status = candidate.status;
  result.preferred_lock_candidate_position_error = errors.max_position_error;
  result.preferred_lock_candidate_orientation_error =
      errors.max_orientation_error;
  result.preferred_lock_candidate_step_norm = step_norm;
  result.preferred_lock_candidate_time_ms = candidate_time_ms;
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

static void clamp_joint_velocity_to_position_limits_for_integration(
    Eigen::Ref<Eigen::VectorXd> dq, const Eigen::VectorXd &q_current,
    const Eigen::VectorXd &q_min, const Eigen::VectorXd &q_max,
    const std::vector<int> &velocity_to_config_index, double integration_dt,
    int nv) {
  if (integration_dt <= 0.0 || dq.size() < nv || q_current.size() == 0) {
    return;
  }

  constexpr double margin_limit = 1e-4;
  for (int vi = 0; vi < nv; ++vi) {
    if (vi >= static_cast<int>(velocity_to_config_index.size())) {
      continue;
    }
    const int qi = velocity_to_config_index[vi];
    if (qi < 0 || qi >= q_current.size() ||
        qi >= q_min.size() || qi >= q_max.size()) {
      continue;
    }
    if (!std::isfinite(q_current[qi]) || !std::isfinite(q_min[qi]) ||
        !std::isfinite(q_max[qi])) {
      continue;
    }

    const double lower_velocity =
        (q_min[qi] + margin_limit - q_current[qi]) / integration_dt;
    const double upper_velocity =
        (q_max[qi] - margin_limit - q_current[qi]) / integration_dt;
    if (lower_velocity <= upper_velocity) {
      dq[vi] = std::clamp(dq[vi], lower_velocity, upper_velocity);
    } else {
      dq[vi] = 0.5 * (lower_velocity + upper_velocity);
    }
  }
}

static void expand_scalar_joint_limits_for_velocity_delta(
    Eigen::VectorXd &q_min, Eigen::VectorXd &q_max,
    const Eigen::VectorXd &delta,
    const std::vector<int> &velocity_to_config_index, int nv) {
  if (delta.size() != nv) {
    return;
  }
  for (int vi = 0; vi < nv; ++vi) {
    if (vi >= static_cast<int>(velocity_to_config_index.size())) {
      continue;
    }
    const int qi = velocity_to_config_index[vi];
    if (qi < 0 || qi >= q_min.size() || qi >= q_max.size() ||
        !std::isfinite(q_min[qi]) || !std::isfinite(q_max[qi])) {
      continue;
    }
    q_min[qi] -= delta[vi];
    q_max[qi] += delta[vi];
  }
}

static void project_scalar_configuration_to_true_joint_limits(
    Eigen::VectorXd &q, const Eigen::VectorXd &q_min,
    const Eigen::VectorXd &q_max,
    const std::vector<int> &velocity_to_config_index, int nv) {
  if (q.size() == 0 || q_min.size() == 0 || q_max.size() == 0) {
    return;
  }

  for (int vi = 0; vi < nv; ++vi) {
    if (vi >= static_cast<int>(velocity_to_config_index.size())) {
      continue;
    }
    const int qi = velocity_to_config_index[vi];
    if (qi < 0 || qi >= q.size() || qi >= q_min.size() || qi >= q_max.size()) {
      continue;
    }
    if (!std::isfinite(q[qi]) || !std::isfinite(q_min[qi]) ||
        !std::isfinite(q_max[qi])) {
      continue;
    }
    q[qi] = std::clamp(q[qi], q_min[qi], q_max[qi]);
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


} // namespace embodik

#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic pop
#endif
