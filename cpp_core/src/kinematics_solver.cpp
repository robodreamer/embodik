/**
 * @file kinematics_solver.cpp
 * @brief Implementation of high-level kinematics solver
 */

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

bool KinematicsSolver::position_step_target_geometry_changed(
    const PositionStepTargetSignature &signature) const {
  if (!last_position_step_target_signature_.has_value()) {
    return true;
  }
  const auto &previous = *last_position_step_target_signature_;
  if (previous.task_names != signature.task_names) {
    return true;
  }

  const bool use_reference_geometry =
      !previous.reference_target_poses.empty() &&
      !signature.reference_target_poses.empty();
  const auto &before = use_reference_geometry
                           ? previous.reference_target_poses
                           : previous.target_poses;
  const auto &after = use_reference_geometry
                          ? signature.reference_target_poses
                          : signature.target_poses;
  if (before.size() != after.size()) {
    return true;
  }
  for (std::size_t index = 0; index < after.size(); ++index) {
    const double translation_delta =
        (before[index].block<3, 1>(0, 3) -
         after[index].block<3, 1>(0, 3))
            .norm();
    const Eigen::Matrix3d rotation_delta =
        before[index].block<3, 3>(0, 0).transpose() *
        after[index].block<3, 3>(0, 0);
    const double rotation_angle = pinocchio::log3(rotation_delta).norm();
    if (!std::isfinite(translation_delta) ||
        !std::isfinite(rotation_angle) ||
        translation_delta > kStationaryTargetTranslationTolerance ||
        rotation_angle > kStationaryTargetRotationTolerance) {
      return true;
    }
  }
  return false;
}

void KinematicsSolver::update_position_step_target_motion_blocks(
    const PositionStepTargetSignature &signature) {
  position_step_target_motion_blocks_.clear();
  if (!last_position_step_target_signature_.has_value()) {
    return;
  }

  const auto &previous = *last_position_step_target_signature_;
  if (previous.task_names != signature.task_names) {
    return;
  }
  const bool use_reference_geometry =
      !previous.reference_target_poses.empty() &&
      previous.reference_target_poses.size() ==
          signature.reference_target_poses.size();
  const auto &before = use_reference_geometry
                           ? previous.reference_target_poses
                           : previous.target_poses;
  const auto &after = use_reference_geometry
                          ? signature.reference_target_poses
                          : signature.target_poses;
  if (before.size() != after.size() ||
      after.size() > signature.task_names.size()) {
    return;
  }

  constexpr char kSecondarySuffix[] = "#secondary";
  constexpr std::size_t kSecondarySuffixLength = sizeof(kSecondarySuffix) - 1;
  for (std::size_t index = 0; index < after.size(); ++index) {
    std::string task_name = signature.task_names[index];
    if (task_name.size() >= kSecondarySuffixLength &&
        task_name.compare(task_name.size() - kSecondarySuffixLength,
                          kSecondarySuffixLength, kSecondarySuffix) == 0) {
      task_name.resize(task_name.size() - kSecondarySuffixLength);
    }

    const double translation_delta =
        (before[index].block<3, 1>(0, 3) -
         after[index].block<3, 1>(0, 3))
            .norm();
    const Eigen::Matrix3d rotation_delta =
        before[index].block<3, 3>(0, 0).transpose() *
        after[index].block<3, 3>(0, 0);
    const double rotation_angle = pinocchio::log3(rotation_delta).norm();
    auto [motion_iter, inserted] = position_step_target_motion_blocks_.try_emplace(
        task_name, std::array<double, 2>{0.0, 0.0});
    (void)inserted;
    auto &motion_blocks = motion_iter->second;
    motion_blocks[0] =
        !std::isfinite(translation_delta)
            ? std::numeric_limits<double>::infinity()
            : std::max(motion_blocks[0], translation_delta);
    motion_blocks[1] =
        !std::isfinite(rotation_angle)
            ? std::numeric_limits<double>::infinity()
            : std::max(motion_blocks[1], rotation_angle);
  }
}

double KinematicsSolver::position_step_task_terminal_prediction_weight(
    const Task &task, const Eigen::VectorXd &current_error) const {
  if (task.getType() == TaskType::FRAME_ORIENTATION) {
    return 0.0;
  }
  const auto motion_iter =
      position_step_target_motion_blocks_.find(task.getName());
  if (motion_iter != position_step_target_motion_blocks_.end()) {
    const double target_translation_step = motion_iter->second[0];
    if (!std::isfinite(target_translation_step)) {
      return 1.0;
    }
    if (target_translation_step <= 0.0) {
      return 0.0;
    }
    if (target_translation_step <= kStationaryTargetTranslationTolerance) {
      return 0.0;
    }
    if (target_translation_step >=
        kTargetTranslationPredictionTransition) {
      return 1.0;
    }
    const double normalized =
        (target_translation_step - kStationaryTargetTranslationTolerance) /
        (kTargetTranslationPredictionTransition -
         kStationaryTargetTranslationTolerance);
    return normalized * normalized * (3.0 - 2.0 * normalized);
  }
  if (task.getType() == TaskType::FRAME_POSE && current_error.size() >= 6) {
    return current_error.head<3>().squaredNorm() > kCollisionEscapeNormEps
               ? 1.0
               : 0.0;
  }
  return 1.0;
}

bool KinematicsSolver::update_position_step_target_signature(
    PositionStepTargetSignature signature) {
  const bool signature_is_finite =
      std::all_of(signature.target_poses.begin(), signature.target_poses.end(),
                  [](const Eigen::Matrix4d &pose) { return pose.allFinite(); }) &&
      std::all_of(signature.reference_target_poses.begin(),
                  signature.reference_target_poses.end(),
                  [](const Eigen::Matrix4d &pose) { return pose.allFinite(); }) &&
      std::all_of(signature.gains.begin(), signature.gains.end(),
                  [](double value) { return std::isfinite(value); });
  if (!signature_is_finite) {
    reset_position_step_continuity_state();
    return false;
  }
  bool existing_target_geometry_changed = false;
  if (last_position_step_target_signature_.has_value()) {
    const auto &previous = *last_position_step_target_signature_;
    const bool target_topology_matches =
        previous.task_names == signature.task_names &&
        previous.target_poses.size() == signature.target_poses.size() &&
        previous.reference_target_poses.size() ==
            signature.reference_target_poses.size();
    const auto poses_match = [](const std::vector<Eigen::Matrix4d> &before,
                                const std::vector<Eigen::Matrix4d> &after) {
      for (std::size_t index = 0; index < after.size(); ++index) {
        const double translation_delta =
            (before[index].block<3, 1>(0, 3) -
             after[index].block<3, 1>(0, 3))
                .norm();
        const Eigen::Matrix3d rotation_delta =
            before[index].block<3, 3>(0, 0).transpose() *
            after[index].block<3, 3>(0, 0);
        const double rotation_angle = pinocchio::log3(rotation_delta).norm();
        if (!std::isfinite(translation_delta) ||
            !std::isfinite(rotation_angle) ||
            translation_delta > kStationaryTargetTranslationTolerance ||
            rotation_angle > kStationaryTargetRotationTolerance) {
          return false;
        }
      }
      return true;
    };
    const bool world_target_geometry_matches =
        target_topology_matches &&
        poses_match(previous.target_poses, signature.target_poses);
    const bool reference_target_geometry_matches =
        target_topology_matches && !signature.reference_target_poses.empty() &&
        poses_match(previous.reference_target_poses,
                    signature.reference_target_poses);
    const bool explicit_command_revision_enabled =
        previous.command_revision >= 0 || signature.command_revision >= 0;
    const bool command_revision_matches =
        previous.command_revision >= 0 && signature.command_revision >= 0 &&
        previous.command_revision == signature.command_revision;
    const bool target_geometry_matches =
        target_topology_matches &&
        (explicit_command_revision_enabled
             ? command_revision_matches
             : (world_target_geometry_matches ||
                reference_target_geometry_matches));
    bool matches = target_geometry_matches &&
                   previous.gains.size() == signature.gains.size();
    if (matches) {
      for (std::size_t index = 0; index < signature.gains.size(); ++index) {
        if (std::abs(previous.gains[index] - signature.gains[index]) >
            kStationaryTargetGainTolerance) {
          matches = false;
          break;
        }
      }
    }
    if (!matches) {
      reset_position_step_merit_window();
      position_step_stationary_anchor_blocks_.reset();
      existing_target_geometry_changed = !target_geometry_matches;
    }
    if (!explicit_command_revision_enabled && world_target_geometry_matches &&
        !reference_target_geometry_matches) {
      signature.reference_target_poses = previous.reference_target_poses;
    }
  } else {
    reset_position_step_merit_window();
    position_step_stationary_anchor_blocks_.reset();
    position_step_collision_command_floor_distances_.clear();
  }
  last_position_step_target_signature_ = std::move(signature);
  return existing_target_geometry_changed;
}

Eigen::Matrix4d KinematicsSolver::canonicalize_position_step_signature_pose(
    const std::string &task_name, const Eigen::Matrix4d &target_pose,
    const std::optional<pinocchio::SE3> &reference_pose) const {
  if (!reference_pose.has_value()) {
    return target_pose;
  }
  const auto task_iter = task_map_.find(task_name);
  if (task_iter != task_map_.end() &&
      std::dynamic_pointer_cast<RelativeFrameTask>(task_iter->second)) {
    return target_pose;
  }

  const pinocchio::SE3 world_target(target_pose.block<3, 3>(0, 0),
                                    target_pose.block<3, 1>(0, 3));
  const pinocchio::SE3 local_target = reference_pose->inverse() * world_target;
  Eigen::Matrix4d canonical_pose = Eigen::Matrix4d::Identity();
  canonical_pose.block<3, 3>(0, 0) = local_target.rotation();
  canonical_pose.block<3, 1>(0, 3) = local_target.translation();
  return canonical_pose;
}

void KinematicsSolver::reset_position_step_continuity_state() {
  last_position_step_target_signature_.reset();
  position_step_target_motion_blocks_.clear();
  position_step_target_motion_observed_ = false;
  position_step_collision_command_floor_distances_.clear();
  position_step_merit_priority_.reset();
  position_step_stationary_anchor_blocks_.reset();
  reset_position_step_merit_window();
}

void KinematicsSolver::reset_position_step_merit_window() {
  position_step_merit_window_anchor_.reset();
  position_step_merit_window_motion_ = 0.0;
  position_step_merit_window_samples_ = 0;
  position_step_merit_window_last_delta_.reset();
  position_step_merit_window_direction_reversals_ = 0;
  position_step_merit_window_last_merits_.reset();
  position_step_merit_window_error_increases_ = 0;
  position_step_stationary_guard_active_ = false;
  position_step_stationary_guard_can_reopen_ = true;
}

bool KinematicsSolver::should_hold_position_step_for_continuity(
    const PositionIKResult &result, const Eigen::VectorXd &current_q,
    double initial_merit, double final_merit,
    const std::vector<double> &initial_target_merits,
    const std::vector<double> &final_target_merits,
    const std::vector<double> &initial_target_block_merits,
    const std::vector<double> &final_target_block_merits,
    int merit_priority, double configuration_step_norm,
    bool owns_position_step_continuity,
    bool collision_violated, bool step_constraint_tradeoff_active) {
  if (!owns_position_step_continuity) {
    return false;
  }

  if (!position_step_merit_priority_.has_value() ||
      *position_step_merit_priority_ != merit_priority) {
    reset_position_step_merit_window();
    position_step_stationary_anchor_blocks_.reset();
    position_step_merit_priority_ = merit_priority;
  }

  const bool candidate_status = result.status == SolverStatus::kSuccess ||
                                result.status == SolverStatus::kNoProgress;
  const bool nominal_candidate =
      candidate_status && !collision_violated &&
      result.collision_rejection_count == 0 && result.stall_escape_count == 0 &&
      result.q_solution.size() == current_q.size() &&
      std::isfinite(initial_merit) && std::isfinite(final_merit) &&
      !initial_target_merits.empty() &&
      initial_target_merits.size() == final_target_merits.size() &&
      !initial_target_block_merits.empty() &&
      initial_target_block_merits.size() ==
          final_target_block_merits.size() &&
      std::all_of(initial_target_merits.begin(), initial_target_merits.end(),
                  [](double merit) { return std::isfinite(merit); }) &&
      std::all_of(final_target_merits.begin(), final_target_merits.end(),
                  [](double merit) { return std::isfinite(merit); }) &&
      std::all_of(initial_target_block_merits.begin(),
                  initial_target_block_merits.end(),
                  [](double merit) { return std::isfinite(merit); }) &&
      std::all_of(final_target_block_merits.begin(),
                  final_target_block_merits.end(),
                  [](double merit) { return std::isfinite(merit); }) &&
      std::isfinite(configuration_step_norm);
  if (!nominal_candidate) {
    const bool zero_motion_collision_rejection =
        collision_violated && result.collision_rejection_count > 0 &&
        result.stall_escape_count == 0 &&
        std::isfinite(configuration_step_norm) &&
        configuration_step_norm <= kPositionStepMeritMotionThreshold;
    if (zero_motion_collision_rejection) {
      return false;
    }
    reset_position_step_merit_window();
    return false;
  }

  const bool collision_constraint_active =
      collision_constraint_.has_value() && collision_constraint_->enabled;
  const bool com_constraint_active =
      com_constraint_.has_value() && com_constraint_->enabled;
  const bool relative_pose_constraint_active =
      relative_pose_constraint_.has_value() &&
      relative_pose_constraint_->enabled;
  const bool other_constraint_tradeoff_active =
      step_constraint_tradeoff_active ||
      get_linear_velocity_constraint_rows() > 0 ||
      !tight_frame_pose_constraints_.empty() ||
      !tight_point_constraints_.empty() || has_contact_frames();
  const bool stationary_unconstrained_velocity_step =
      !acceleration_limits_enabled_ && position_step_target_motion_observed_ &&
      !position_step_target_geometry_moved_ && !collision_constraint_active &&
      !com_constraint_active && !relative_pose_constraint_active &&
      !other_constraint_tradeoff_active;
  if (stationary_unconstrained_velocity_step &&
      !position_step_stationary_anchor_blocks_.has_value()) {
    position_step_stationary_anchor_blocks_ = initial_target_block_merits;
  }
  const std::vector<double> &stationary_anchor_blocks =
      position_step_stationary_anchor_blocks_.has_value()
          ? *position_step_stationary_anchor_blocks_
          : initial_target_block_merits;
  const auto dominant_anchor_block = std::max_element(
      stationary_anchor_blocks.begin(), stationary_anchor_blocks.end());
  const std::size_t dominant_anchor_block_index =
      static_cast<std::size_t>(std::distance(
          stationary_anchor_blocks.begin(), dominant_anchor_block));
  const bool stationary_target_regressed =
      stationary_unconstrained_velocity_step &&
      final_target_block_merits[dominant_anchor_block_index] >
          *dominant_anchor_block +
              kStationaryTargetMeritRegressionTolerance;
  if (stationary_target_regressed) {
    position_step_stationary_guard_active_ = true;
    position_step_stationary_guard_can_reopen_ = false;
    position_step_merit_window_anchor_ = initial_merit;
    position_step_merit_window_motion_ = 0.0;
    position_step_merit_window_samples_ = 0;
    position_step_merit_window_last_delta_.reset();
    position_step_merit_window_direction_reversals_ = 0;
    position_step_merit_window_last_merits_ = initial_target_merits;
    position_step_merit_window_error_increases_ = 0;
    return true;
  }

  if (!position_step_merit_window_anchor_.has_value()) {
    position_step_merit_window_anchor_ = initial_merit;
  }
  const Eigen::VectorXd candidate_delta =
      pinocchio::difference(robot_->model(), current_q, result.q_solution);
  if (position_step_merit_window_last_delta_.has_value() &&
      position_step_merit_window_last_delta_->size() ==
          candidate_delta.size() &&
      position_step_merit_window_last_delta_->norm() >
          kStationaryDirectionChangeMotionThreshold &&
      candidate_delta.norm() > kStationaryDirectionChangeMotionThreshold &&
      position_step_merit_window_last_delta_->dot(candidate_delta) < 0.0) {
    ++position_step_merit_window_direction_reversals_;
  }
  position_step_merit_window_last_delta_ = candidate_delta;
  if (position_step_merit_window_last_merits_.has_value() &&
      position_step_merit_window_last_merits_->size() ==
          final_target_merits.size()) {
    const auto dominant_iter = std::max_element(
        position_step_merit_window_last_merits_->begin(),
        position_step_merit_window_last_merits_->end());
    if (dominant_iter != position_step_merit_window_last_merits_->end()) {
      const std::size_t dominant_index = static_cast<std::size_t>(
          std::distance(position_step_merit_window_last_merits_->begin(),
                        dominant_iter));
      if (final_target_merits[dominant_index] > *dominant_iter + 1e-4) {
        ++position_step_merit_window_error_increases_;
      }
    }
  }
  position_step_merit_window_last_merits_ = final_target_merits;
  position_step_merit_window_motion_ +=
      std::max(0.0, configuration_step_norm);
  ++position_step_merit_window_samples_;

  if (position_step_stationary_guard_active_) {
    if (!position_step_stationary_guard_can_reopen_) {
      return true;
    }
    const bool hold_candidate = should_hold_non_improving_position_step(
        result, current_q, initial_merit, final_merit,
        configuration_step_norm, collision_violated);
    const bool made_sufficient_progress =
        !hold_candidate && has_sufficient_position_step_merit_reduction(
                               initial_merit, final_merit,
                               configuration_step_norm);
    if (made_sufficient_progress) {
      reset_position_step_merit_window();
    }
    return !made_sufficient_progress;
  }

  const int required_window_samples =
      runtime_config_.enable_auto_task_layout
          ? std::max(kStationaryTargetDwellCalls,
                     runtime_config_.auto_layout_cooldown_ticks + 1)
          : kStationaryTargetDwellCalls;
  if (position_step_merit_window_samples_ < required_window_samples) {
    return false;
  }

  const bool target_is_satisfied =
      final_merit <= kPositionStepSatisfiedMeritTolerance;
  const bool window_has_repeated_direction_reversals =
      position_step_merit_window_direction_reversals_ >
      kStationaryMaxDirectionReversals;
  const bool window_has_repeated_dominant_target_increases =
      position_step_merit_window_error_increases_ >
      kStationaryMaxDirectionReversals;
  const bool window_has_strong_net_progress =
      has_sufficient_position_step_merit_reduction(
          *position_step_merit_window_anchor_, final_merit,
          position_step_merit_window_motion_,
          kStationaryOscillationMinErrorReductionPerConfiguration);
  // Stateful layouts and redundant robots can reverse individual joint steps
  // while still making decisive nonlinear task-space progress. That exception
  // does not apply when the currently worst target repeatedly gets worse: an
  // aggregate improvement from sacrificing one commanded target is not useful
  // stationary-target progress.
  const bool window_was_oscillatory =
      window_has_repeated_dominant_target_increases ||
      (window_has_repeated_direction_reversals &&
       !window_has_strong_net_progress);
  const bool caller_owns_command_identity =
      last_position_step_target_signature_.has_value() &&
      last_position_step_target_signature_->command_revision >= 0;
  const bool window_has_required_progress =
      caller_owns_command_identity
          ? window_has_strong_net_progress
          : has_sufficient_position_step_merit_reduction(
                *position_step_merit_window_anchor_, final_merit,
                position_step_merit_window_motion_,
                kPositionStepMinErrorReductionPerConfiguration,
                kStationaryMinErrorReductionPerCall *
                    static_cast<double>(position_step_merit_window_samples_));
  const bool window_was_productive =
      !target_is_satisfied && !window_was_oscillatory &&
      window_has_required_progress;
  if (window_was_productive) {
    position_step_merit_window_anchor_ = final_merit;
    position_step_merit_window_motion_ = 0.0;
    position_step_merit_window_samples_ = 0;
    position_step_merit_window_last_delta_.reset();
    position_step_merit_window_direction_reversals_ = 0;
    position_step_merit_window_last_merits_.reset();
    position_step_merit_window_error_increases_ = 0;
    return false;
  }

  position_step_stationary_guard_active_ = true;
  position_step_stationary_guard_can_reopen_ =
      !caller_owns_command_identity && !target_is_satisfied &&
      !window_was_oscillatory;
  position_step_merit_window_anchor_ = initial_merit;
  position_step_merit_window_motion_ = 0.0;
  position_step_merit_window_samples_ = 0;
  return true;
}

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

std::shared_ptr<ManipulabilityTask>
KinematicsSolver::add_manipulability_task(const std::string &name,
                                          const std::string &frame_name,
                                          TaskType frame_task_type) {
  if (task_map_.find(name) != task_map_.end()) {
    throw std::runtime_error("Task with name '" + name + "' already exists");
  }

  auto task = std::make_shared<ManipulabilityTask>(
      name, robot_, frame_name, frame_task_type);
  tasks_.push_back(task);
  task_map_[name] = task;
  return task;
}

std::shared_ptr<JointLimitAvoidanceTask>
KinematicsSolver::add_joint_limit_avoidance_task(
    const std::string &name,
    const std::vector<int> &controlled_joint_indices) {
  if (task_map_.find(name) != task_map_.end()) {
    throw std::runtime_error("Task with name '" + name + "' already exists");
  }

  auto task = std::make_shared<JointLimitAvoidanceTask>(
      name, robot_, controlled_joint_indices);
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

std::shared_ptr<PoseTaskGroup>
KinematicsSolver::add_pose_task_group(const std::string &name,
                                      const std::string &tcp_frame,
                                      int base_priority,
                                      int rotation_priority_offset,
                                      bool merged_pose, bool auto_switch) {
  if (pose_task_groups_.find(name) != pose_task_groups_.end()) {
    throw std::runtime_error("Pose task group with name '" + name +
                             "' already exists");
  }
  if (merged_pose && auto_switch) {
    throw std::runtime_error("Pose task group '" + name +
                             "' cannot use merged_pose and auto_switch "
                             "together");
  }

  const auto position_name = name + "__position";
  const auto orientation_name = name + "__orientation";
  const auto task_name_conflicts = [&](const std::string &task_name) {
    return task_map_.find(task_name) != task_map_.end();
  };
  if (task_name_conflicts(name) ||
      (!merged_pose && task_name_conflicts(position_name)) ||
      (!merged_pose && task_name_conflicts(orientation_name))) {
    throw std::runtime_error("Pose task group '" + name +
                             "' conflicts with an existing task name");
  }

  std::shared_ptr<PoseTaskGroup> group;
  if (auto_switch) {
    auto merged_task = std::make_shared<FrameTask>(name, robot_, tcp_frame,
                                                   TaskType::FRAME_POSE);
    auto position_task = std::make_shared<FrameTask>(
        position_name, robot_, tcp_frame, TaskType::FRAME_POSITION);
    auto orientation_task = std::make_shared<FrameTask>(
        orientation_name, robot_, tcp_frame, TaskType::FRAME_ORIENTATION);
    merged_task->setPriority(base_priority);
    position_task->setPriority(base_priority);
    orientation_task->setPriority(base_priority + rotation_priority_offset);
    tasks_.push_back(merged_task);
    task_map_[name] = merged_task;
    tasks_.push_back(position_task);
    task_map_[position_name] = position_task;
    tasks_.push_back(orientation_task);
    task_map_[orientation_name] = orientation_task;
    group = std::make_shared<PoseTaskGroup>(
        name, tcp_frame, base_priority, rotation_priority_offset, position_task,
        orientation_task, merged_task);
  } else if (merged_pose) {
    auto pose_task = std::make_shared<FrameTask>(name, robot_, tcp_frame,
                                                 TaskType::FRAME_POSE);
    pose_task->setPriority(base_priority);
    tasks_.push_back(pose_task);
    task_map_[name] = pose_task;
    group = std::make_shared<PoseTaskGroup>(name, tcp_frame, base_priority,
                                            pose_task);
  } else {
    auto position_task = std::make_shared<FrameTask>(
        position_name, robot_, tcp_frame, TaskType::FRAME_POSITION);
    auto orientation_task = std::make_shared<FrameTask>(
        orientation_name, robot_, tcp_frame, TaskType::FRAME_ORIENTATION);
    position_task->setPriority(base_priority);
    orientation_task->setPriority(base_priority + rotation_priority_offset);
    tasks_.push_back(position_task);
    task_map_[position_name] = position_task;
    tasks_.push_back(orientation_task);
    task_map_[orientation_name] = orientation_task;
    group = std::make_shared<PoseTaskGroup>(
        name, tcp_frame, base_priority, rotation_priority_offset, position_task,
        orientation_task);
  }

  pose_task_groups_[name] = group;
  sort_tasks_by_priority();
  return group;
}

std::shared_ptr<PoseTaskGroup>
KinematicsSolver::pose_task_group(const std::string &name) const {
  auto it = pose_task_groups_.find(name);
  return (it != pose_task_groups_.end()) ? it->second : nullptr;
}

void KinematicsSolver::reset_auto_task_layout_state() {
  current_auto_task_layout_ = TaskLayout::kMerged;
  auto_layout_below_low_count_ = 0;
  auto_layout_has_feedback_ = false;
  auto_layout_binding_score_ = 0.0;
}

void KinematicsSolver::select_auto_task_layout() {
  if (!runtime_config_.enable_auto_task_layout) {
    return;
  }

  if (auto_layout_has_feedback_) {
    const double high = runtime_config_.auto_layout_binding_threshold_high;
    const double low = runtime_config_.auto_layout_binding_threshold_low;
    const int cooldown =
        std::max(1, runtime_config_.auto_layout_cooldown_ticks);
    if (current_auto_task_layout_ == TaskLayout::kMerged) {
      if (auto_layout_binding_score_ > high) {
        current_auto_task_layout_ = TaskLayout::kSplit;
        auto_layout_below_low_count_ = 0;
      }
    } else {
      if (auto_layout_binding_score_ < low) {
        ++auto_layout_below_low_count_;
        if (auto_layout_below_low_count_ >= cooldown) {
          current_auto_task_layout_ = TaskLayout::kMerged;
          auto_layout_below_low_count_ = 0;
        }
      } else {
        auto_layout_below_low_count_ = 0;
      }
    }
  }

  for (auto &kv : pose_task_groups_) {
    if (kv.second && kv.second->auto_switch()) {
      kv.second->set_layout(current_auto_task_layout_);
    }
  }
}

VelocitySolverResult KinematicsSolver::retry_auto_task_layout_as_split_if_needed(
    const Eigen::VectorXd &q, VelocitySolverResult result,
    const std::vector<int> &velocity_lock_indices,
    const std::optional<TorsoPoseConstraintOptions> &torso_constraint,
    const std::optional<double> &step_validation_dt,
    const std::vector<PositionStepPriorityConstraintSpec>
        &priority_constraints,
    bool apply_position_step_acceleration_limits) {
  const bool merged_succeeded = result.status == SolverStatus::kSuccess;
  const bool merged_binds =
      auto_layout_binding_score_ >
      runtime_config_.auto_layout_binding_threshold_high;
  if (!runtime_config_.enable_auto_task_layout ||
      result.active_task_layout != TaskLayout::kMerged ||
      (merged_succeeded && !merged_binds)) {
    return result;
  }

  current_auto_task_layout_ = TaskLayout::kSplit;
  auto_layout_below_low_count_ = 0;
  for (auto &kv : pose_task_groups_) {
    if (kv.second && kv.second->auto_switch()) {
      kv.second->set_layout(TaskLayout::kSplit);
    }
  }

  pending_velocity_lock_indices_ = velocity_lock_indices;
  pending_step_torso_constraint_ = torso_constraint;
  pending_step_validation_dt_ = step_validation_dt;
  pending_position_step_priority_constraints_ = priority_constraints;
  pending_reuse_current_kinematics_ = true;
  pending_position_step_acceleration_limits_ =
      apply_position_step_acceleration_limits;
  VelocitySolverResult retry = solve_velocity(q, true);
  const char *reason =
      merged_succeeded ? "binding merged solve" : "non-success merged solve";
  if (retry.status_message.empty()) {
    retry.status_message =
        std::string("auto task layout retried split after ") + reason;
  } else {
    retry.status_message = std::string("auto task layout retried split after ") +
                           reason + ": " + retry.status_message;
  }
  if (merged_succeeded && retry.status != SolverStatus::kSuccess) {
    if (result.status_message.empty()) {
      result.status_message =
          std::string("auto task layout kept successful merged solve after "
                      "failed split retry: ") +
          retry.status_message;
    }
    return result;
  }
  return retry;
}

void KinematicsSolver::update_auto_task_layout_feedback(
    const VelocitySolverResult &result) {
  if (!runtime_config_.enable_auto_task_layout) {
    return;
  }
  auto clip01 = [](double x) {
    if (!std::isfinite(x)) {
      return 0.0;
    }
    return std::clamp(x, 0.0, 1.0);
  };

  double scale_binding = 0.0;
  if (!result.task_scales.empty()) {
    double min_scale = 1.0;
    for (double s : result.task_scales) {
      if (std::isfinite(s)) {
        min_scale = std::min(min_scale, s);
      }
    }
    scale_binding = clip01(1.0 - min_scale);
  }

  const double joint_clip_fraction =
      robot_->nv() > 0 ? static_cast<double>(result.saturated_joints.size()) /
                             static_cast<double>(robot_->nv())
                       : 0.0;

  double collision_binding = 0.0;
  if (collision_constraint_.has_value() && collision_constraint_->enabled &&
      std::isfinite(last_constraint_min_recovery_margin_)) {
    const double margin =
        std::max(collision_constraint_->constraint_activation_margin, 1e-9);
    const double clearance = last_constraint_min_recovery_margin_;
    collision_binding = clip01(1.0 - clearance / margin);
  }

  auto_layout_binding_score_ =
      std::max(scale_binding, std::max(joint_clip_fraction, collision_binding));
  auto_layout_has_feedback_ = true;
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

void KinematicsSolver::add_contact_frame(const std::string &frame_name,
                                         ContactType type) {
  if (frame_name.empty()) {
    throw std::invalid_argument("contact frame name must not be empty");
  }
  if (!robot_->has_frame(frame_name)) {
    throw std::invalid_argument("unknown frame for contact projection: " +
                                frame_name);
  }
  auto it = std::find_if(
      contact_frames_.begin(), contact_frames_.end(),
      [&](const ContactFrameConfig &cfg) { return cfg.frame_name == frame_name; });
  if (it != contact_frames_.end()) {
    it->type = type;
    return;
  }
  contact_frames_.push_back(ContactFrameConfig{frame_name, type});
}

void KinematicsSolver::configure_contact_frames(
    const std::vector<std::string> &frame_names, ContactType type) {
  clear_contact_frames();
  for (const auto &frame_name : frame_names) {
    add_contact_frame(frame_name, type);
  }
}

void KinematicsSolver::clear_contact_frames() { contact_frames_.clear(); }

void KinematicsSolver::set_linear_velocity_constraints(
    const Eigen::MatrixXd &C, const Eigen::VectorXd &lower_bounds,
    const Eigen::VectorXd &upper_bounds) {
  std::string err;
  if (!validate_linear_constraint_triplet(C, lower_bounds, upper_bounds,
                                          robot_->nv(), &err)) {
    throw std::invalid_argument(err);
  }
  LinearVelocityConstraintConfig cfg;
  cfg.enabled = true;
  cfg.C = C;
  cfg.lower_bounds = lower_bounds;
  cfg.upper_bounds = upper_bounds;
  linear_velocity_constraints_ = std::move(cfg);
}

void KinematicsSolver::append_linear_velocity_constraints(
    const Eigen::MatrixXd &C, const Eigen::VectorXd &lower_bounds,
    const Eigen::VectorXd &upper_bounds) {
  std::string err;
  if (!validate_linear_constraint_triplet(C, lower_bounds, upper_bounds,
                                          robot_->nv(), &err)) {
    throw std::invalid_argument(err);
  }
  if (!linear_velocity_constraints_.has_value() ||
      !linear_velocity_constraints_->enabled ||
      linear_velocity_constraints_->C.rows() == 0) {
    set_linear_velocity_constraints(C, lower_bounds, upper_bounds);
    return;
  }

  auto &cfg = *linear_velocity_constraints_;
  const int old_rows = static_cast<int>(cfg.C.rows());
  const int add_rows = static_cast<int>(C.rows());
  Eigen::MatrixXd C_all(old_rows + add_rows, robot_->nv());
  Eigen::VectorXd lower_all(old_rows + add_rows);
  Eigen::VectorXd upper_all(old_rows + add_rows);
  C_all.topRows(old_rows) = cfg.C;
  C_all.bottomRows(add_rows) = C;
  lower_all.head(old_rows) = cfg.lower_bounds;
  lower_all.tail(add_rows) = lower_bounds;
  upper_all.head(old_rows) = cfg.upper_bounds;
  upper_all.tail(add_rows) = upper_bounds;
  cfg.C = std::move(C_all);
  cfg.lower_bounds = std::move(lower_all);
  cfg.upper_bounds = std::move(upper_all);
}

void KinematicsSolver::clear_linear_velocity_constraints() {
  linear_velocity_constraints_.reset();
}

int KinematicsSolver::get_linear_velocity_constraint_rows() const {
  if (!linear_velocity_constraints_.has_value() ||
      !linear_velocity_constraints_->enabled) {
    return 0;
  }
  return static_cast<int>(linear_velocity_constraints_->C.rows());
}

void KinematicsSolver::add_tight_frame_pose_constraint(
    const std::string &frame_name, const Eigen::Matrix4d &target_pose,
    double position_epsilon, double orientation_epsilon,
    const Eigen::VectorXd &axis_mask) {
  if (!robot_->has_frame(frame_name)) {
    throw std::invalid_argument("unknown frame for tight pose constraint: " +
                                frame_name);
  }
  if (!std::isfinite(position_epsilon) || position_epsilon <= 0.0 ||
      !std::isfinite(orientation_epsilon) || orientation_epsilon <= 0.0) {
    throw std::invalid_argument(
        "tight pose constraint epsilons must be finite and > 0");
  }
  TightFramePoseConstraintConfig cfg;
  cfg.frame_name = frame_name;
  cfg.target_pose = pinocchio::SE3(target_pose.block<3, 3>(0, 0),
                                   target_pose.block<3, 1>(0, 3));
  cfg.position_epsilon = position_epsilon;
  cfg.orientation_epsilon = orientation_epsilon;
  if (axis_mask.size() == 0) {
    cfg.axis_mask = Eigen::VectorXd::Ones(6);
  } else if (axis_mask.size() == 6 && axis_mask.allFinite()) {
    cfg.axis_mask = axis_mask;
  } else {
    throw std::invalid_argument(
        "tight pose constraint axis_mask must be empty or finite 6D");
  }
  tight_frame_pose_constraints_.push_back(std::move(cfg));
}

void KinematicsSolver::clear_tight_frame_pose_constraints() {
  tight_frame_pose_constraints_.clear();
}

void KinematicsSolver::add_tight_point_constraint(
    const std::string &frame_name, const Eigen::Vector3d &target_point,
    double position_epsilon, const Eigen::Vector3d &axis_mask) {
  if (!robot_->has_frame(frame_name)) {
    throw std::invalid_argument("unknown frame for tight point constraint: " +
                                frame_name);
  }
  if (!target_point.allFinite()) {
    throw std::invalid_argument("tight point constraint target must be finite");
  }
  if (!std::isfinite(position_epsilon) || position_epsilon <= 0.0) {
    throw std::invalid_argument(
        "tight point constraint position_epsilon must be finite and > 0");
  }
  if (!axis_mask.allFinite()) {
    throw std::invalid_argument("tight point constraint axis_mask must be finite");
  }
  TightPointConstraintConfig cfg;
  cfg.frame_name = frame_name;
  cfg.target_point = target_point;
  cfg.position_epsilon = position_epsilon;
  cfg.axis_mask = axis_mask;
  tight_point_constraints_.push_back(std::move(cfg));
}

void KinematicsSolver::clear_tight_point_constraints() {
  tight_point_constraints_.clear();
}

Eigen::MatrixXd KinematicsSolver::compute_contact_projector() const {
  const int nv = robot_->nv();
  if (contact_frames_.empty()) {
    return Eigen::MatrixXd::Identity(nv, nv);
  }

  int total_rows = 0;
  for (const auto &cfg : contact_frames_) {
    total_rows += (cfg.type == ContactType::kPointContact) ? 3 : 6;
  }
  if (total_rows <= 0) {
    return Eigen::MatrixXd::Identity(nv, nv);
  }

  Eigen::MatrixXd J_c(total_rows, nv);
  J_c.setZero();
  int row = 0;
  for (const auto &cfg : contact_frames_) {
    const Matrix6Xd J_full = robot_->get_frame_jacobian(cfg.frame_name);
    if (cfg.type == ContactType::kPointContact) {
      J_c.block(row, 0, 3, nv) = J_full.topRows(3);
      row += 3;
    } else {
      J_c.block(row, 0, 6, nv) = J_full;
      row += 6;
    }
  }

  Eigen::MatrixXd J_c_pinv;
  detail::ComputeGeneralizedInverse(J_c, constraint_tolerance_, &J_c_pinv);
  return Eigen::MatrixXd::Identity(nv, nv) - J_c_pinv * J_c;
}

std::optional<KinematicsSolver::LinearVelocityConstraintResult>
KinematicsSolver::compute_linear_velocity_constraints() {
  if (!linear_velocity_constraints_.has_value() ||
      !linear_velocity_constraints_->enabled ||
      linear_velocity_constraints_->C.rows() == 0) {
    return std::nullopt;
  }
  LinearVelocityConstraintResult out;
  out.jacobian = linear_velocity_constraints_->C;
  out.lower_bounds = linear_velocity_constraints_->lower_bounds;
  out.upper_bounds = linear_velocity_constraints_->upper_bounds;
  out.violated_rows = Eigen::ArrayXi::Zero(out.jacobian.rows());
  return out;
}

std::optional<KinematicsSolver::LinearVelocityConstraintResult>
KinematicsSolver::compute_tight_frame_pose_constraints() {
  if (tight_frame_pose_constraints_.empty()) {
    return std::nullopt;
  }

  int total_rows = 0;
  for (const auto &cfg : tight_frame_pose_constraints_) {
    for (int i = 0; i < 6; ++i) {
      if (cfg.axis_mask(i) > 0.5) {
        ++total_rows;
      }
    }
  }
  if (total_rows <= 0) {
    return std::nullopt;
  }

  LinearVelocityConstraintResult out;
  out.jacobian = Eigen::MatrixXd::Zero(total_rows, robot_->nv());
  out.lower_bounds = Eigen::VectorXd::Constant(total_rows, -1e10);
  out.upper_bounds = Eigen::VectorXd::Constant(total_rows, 1e10);
  out.violated_rows = Eigen::ArrayXi::Zero(total_rows);

  int row = 0;
  for (const auto &cfg : tight_frame_pose_constraints_) {
    const pinocchio::SE3 pose = robot_->get_frame_pose(cfg.frame_name);
    const Matrix6Xd J = robot_->get_frame_jacobian(cfg.frame_name);

    Eigen::VectorXd err = Eigen::VectorXd::Zero(6);
    err.head<3>() = pose.translation() - cfg.target_pose.translation();
    err.tail<3>() =
        pinocchio::log3(cfg.target_pose.rotation().transpose() * pose.rotation());

    for (int i = 0; i < 6; ++i) {
      if (cfg.axis_mask(i) <= 0.5) {
        continue;
      }
      const double eps = (i < 3) ? cfg.position_epsilon : cfg.orientation_epsilon;
      const double slack_lower = err(i) + eps;
      const double slack_upper = eps - err(i);
      const double vel_limit = (i < 3) ? 0.5 : 1.0;
      const double acc_limit = (i < 3) ? 1.0 : 2.0;
      const double min_headroom = 0.05 * vel_limit;
      const double activation_margin = 0.5 * eps;
      auto [lower, upper] = calculate_velocity_box_constraint(
          slack_lower, slack_upper, vel_limit, acc_limit, dt_, min_headroom,
          activation_margin);
      out.jacobian.row(row) = J.row(i);
      out.lower_bounds(row) = lower;
      out.upper_bounds(row) = upper;
      if (slack_lower < -constraint_tolerance_ ||
          slack_upper < -constraint_tolerance_) {
        out.violated_rows(row) = 1;
      }
      ++row;
    }
  }

  return out;
}

std::optional<KinematicsSolver::LinearVelocityConstraintResult>
KinematicsSolver::compute_tight_point_constraints() {
  if (tight_point_constraints_.empty()) {
    return std::nullopt;
  }

  int total_rows = 0;
  for (const auto &cfg : tight_point_constraints_) {
    for (int i = 0; i < 3; ++i) {
      if (cfg.axis_mask(i) > 0.5) {
        ++total_rows;
      }
    }
  }
  if (total_rows <= 0) {
    return std::nullopt;
  }

  LinearVelocityConstraintResult out;
  out.jacobian = Eigen::MatrixXd::Zero(total_rows, robot_->nv());
  out.lower_bounds = Eigen::VectorXd::Constant(total_rows, -1e10);
  out.upper_bounds = Eigen::VectorXd::Constant(total_rows, 1e10);
  out.violated_rows = Eigen::ArrayXi::Zero(total_rows);

  int row = 0;
  for (const auto &cfg : tight_point_constraints_) {
    const pinocchio::SE3 pose = robot_->get_frame_pose(cfg.frame_name);
    const Matrix6Xd J = robot_->get_frame_jacobian(cfg.frame_name);
    const Eigen::Vector3d err = pose.translation() - cfg.target_point;
    for (int i = 0; i < 3; ++i) {
      if (cfg.axis_mask(i) <= 0.5) {
        continue;
      }
      const double eps = cfg.position_epsilon;
      const double slack_lower = err(i) + eps;
      const double slack_upper = eps - err(i);
      constexpr double kTightPointVelLimit = 0.5;
      constexpr double kTightPointAccLimit = 1.0;
      const double min_headroom = 0.05 * kTightPointVelLimit;
      const double activation_margin = 0.5 * eps;
      auto [lower, upper] = calculate_velocity_box_constraint(
          slack_lower, slack_upper, kTightPointVelLimit, kTightPointAccLimit,
          dt_, min_headroom, activation_margin);
      out.jacobian.row(row) = J.row(i);
      out.lower_bounds(row) = lower;
      out.upper_bounds(row) = upper;
      if (slack_lower < -constraint_tolerance_ ||
          slack_upper < -constraint_tolerance_) {
        out.violated_rows(row) = 1;
      }
      ++row;
    }
  }

  return out;
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
  auto group_it = pose_task_groups_.find(name);
  if (group_it != pose_task_groups_.end()) {
    auto group = group_it->second;
    pose_task_groups_.erase(group_it);
    if (group && group->position_task()) {
      remove_task(group->position_task()->getName());
    }
    if (group && group->orientation_task()) {
      remove_task(group->orientation_task()->getName());
    }
    if (group && group->merged_task()) {
      remove_task(group->merged_task()->getName());
    }
    return;
  }

  auto it = task_map_.find(name);
  if (it != task_map_.end()) {
    reset_position_step_continuity_state();
    auto task = it->second;
    task_map_.erase(it);

    // Remove from tasks vector
    tasks_.erase(std::remove(tasks_.begin(), tasks_.end(), task), tasks_.end());
    for (auto group_it = pose_task_groups_.begin();
         group_it != pose_task_groups_.end();) {
      const auto &group = group_it->second;
      const bool references_removed_task =
          group &&
          ((group->position_task() &&
            group->position_task()->getName() == name) ||
           (group->orientation_task() &&
            group->orientation_task()->getName() == name) ||
           (group->merged_task() && group->merged_task()->getName() == name));
      if (references_removed_task) {
        group_it = pose_task_groups_.erase(group_it);
      } else {
        ++group_it;
      }
    }
  }
}

void KinematicsSolver::clear_tasks() {
  tasks_.clear();
  task_map_.clear();
  pose_task_groups_.clear();
  reset_position_step_continuity_state();
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
    last_collision_sphere_culled_pairs_ = 0;
    last_collision_budget_exhausted_ = false;
    last_constraint_min_distance_ = std::numeric_limits<double>::infinity();
    last_constraint_min_recovery_margin_ =
        std::numeric_limits<double>::infinity();
    last_constraint_was_full_scan_ = false;
    last_collision_constraint_result_.reset();
    last_collision_constraint_row_dt_ =
        std::numeric_limits<double>::quiet_NaN();
    last_collision_constraint_q_ = Eigen::VectorXd();
    collision_stuck_counters_.clear();
    collision_stuck_last_distances_.clear();
    collision_pair_distance_floor_.clear();
    post_step_collision_distance_cache_.clear();
    collision_cache_frozen_indices_.clear();
  }
  if (sphere_broadphase_enabled_ && !sphere_broadphase_.is_built()) {
    sphere_broadphase_.build(*collision_model_ptr);
  }
  if (!stall_user_configured_) {
    stall_config_.stall_threshold = 3;
    stall_config_.restore_rate = 0.2;
    stall_config_.floor_fraction = 0.0;
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

  const ScopedRobotKinematicsRestore restore_robot_state(*robot_);

  if (current_q.size() > 0) {
    if (current_q.size() != robot_->nq()) {
      throw std::runtime_error(
          "Invalid configuration size for collision evaluation.");
    }
    if (!current_q.allFinite()) {
      throw std::runtime_error(
          "Collision evaluation configuration must be finite.");
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

  const ScopedRobotKinematicsRestore restore_robot_state(*robot_);

  if (current_q.size() > 0) {
    if (current_q.size() != robot_->nq()) {
      throw std::runtime_error(
          "Invalid configuration size for collision evaluation.");
    }
    if (!current_q.allFinite()) {
      throw std::runtime_error(
          "Collision evaluation configuration must be finite.");
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
  const PositionStepMutableStateSnapshot snapshot =
      capture_position_step_mutable_state();
  try {
    const auto distance =
        evaluate_post_step_collision_distance_mutating(q);
    restore_position_step_mutable_state(snapshot);
    return distance;
  } catch (...) {
    restore_position_step_mutable_state(snapshot);
    throw;
  }
}

std::optional<double>
KinematicsSolver::evaluate_post_step_collision_distance_mutating(
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

  // Tier 2: evaluate every active pair's effective recovery margin once, then
  // derive the scalar minimum from those same distance results. This avoids a
  // targeted exact scan followed immediately by a duplicate recovery scan.
  const auto recovery_margins =
      evaluate_post_step_collision_recovery_margins(q);
  if (!recovery_margins.empty()) {
    const auto current_distance =
        evaluate_post_step_collision_distance_from_current_results();
    if (current_distance.has_value()) {
      return current_distance;
    }
  }

  // Tier 3: Targeted fallback when recovery margins are unavailable.
  auto targeted_pairs = get_post_step_rejection_pair_indices();
  if (!targeted_pairs.empty()) {
    return evaluate_min_collision_distance_targeted(q, targeted_pairs);
  }

  // Tier 4: Full scan fallback (no constraint data available).
  return evaluate_min_collision_distance(q);
}

std::vector<double>
KinematicsSolver::evaluate_post_step_collision_recovery_margins(
    const Eigen::VectorXd &current_q) {
#ifdef PINOCCHIO_WITH_HPP_FCL
  if (!collision_constraint_.has_value() ||
      !collision_constraint_->enabled) {
    return {};
  }
  if (current_q.size() != robot_->nq()) {
    throw std::runtime_error(
        "Invalid configuration size for collision recovery evaluation.");
  }

  auto *collision_model = robot_->collision_model();
  auto *collision_data = robot_->collision_data();
  if (collision_model == nullptr || collision_data == nullptr) {
    return {};
  }

  robot_->update_kinematics(current_q);
  pinocchio::updateGeometryPlacements(robot_->model(), robot_->data(),
                                      *collision_model, *collision_data);

  const std::size_t pair_count = collision_model->collisionPairs.size();
  const auto &pairs = collision_model->collisionPairs;
  const auto &oMi = robot_->data().oMi;
  std::vector<std::uint8_t> enabled_pair_mask(pair_count, 0);
  for (std::size_t pair_index = 0; pair_index < pair_count; ++pair_index) {
    const bool allowed =
        collision_allowed_pair_mask_.empty() ||
        pair_index >= collision_allowed_pair_mask_.size() ||
        collision_allowed_pair_mask_[pair_index];
    const bool active = collision_data->activeCollisionPairs.empty() ||
                        collision_data->activeCollisionPairs[pair_index];
    enabled_pair_mask[pair_index] = allowed && active ? 1 : 0;
  }

  std::vector<double> signed_distance_lower_bounds;
  for (const auto &entry_ptr : post_step_collision_distance_cache_) {
    const auto &entry = *entry_ptr;
    if (entry.q.size() == current_q.size() &&
        (entry.q.array() == current_q.array()).all() &&
        entry.enabled_pair_mask == enabled_pair_mask &&
        entry.certified_signed_distance_lower_bounds.size() == pair_count) {
      signed_distance_lower_bounds =
          entry.certified_signed_distance_lower_bounds;
      break;
    }
  }

  if (signed_distance_lower_bounds.empty()) {
    signed_distance_lower_bounds.assign(
        pair_count, std::numeric_limits<double>::quiet_NaN());
    const PostStepCollisionDistanceCertificate *nearest_certificate = nullptr;
    double nearest_certificate_distance =
        std::numeric_limits<double>::infinity();
    for (const auto &entry_ptr : post_step_collision_distance_cache_) {
      const auto &entry = *entry_ptr;
      if (entry.enabled_pair_mask == enabled_pair_mask &&
          entry.certified_signed_distance_lower_bounds.size() == pair_count &&
          entry.joint_translations.size() == oMi.size() &&
          entry.joint_rotations.size() == oMi.size()) {
        const Eigen::VectorXd delta =
            pinocchio::difference(robot_->model(), entry.q, current_q);
        const double distance = delta.allFinite()
                                    ? delta.squaredNorm()
                                    : std::numeric_limits<double>::infinity();
        if (distance < nearest_certificate_distance) {
          nearest_certificate = &entry;
          nearest_certificate_distance = distance;
        }
      }
    }

    for (std::size_t pair_index = 0; pair_index < pair_count; ++pair_index) {
      if (!enabled_pair_mask[pair_index]) {
        continue;
      }

      const auto &pair = pairs[pair_index];
      const auto &name_a =
          collision_model->geometryObjects[pair.first].name;
      const auto &name_b =
          collision_model->geometryObjects[pair.second].name;
      const std::string pair_key = canonical_pair_key(name_a, name_b);
      double effective_min_distance = collision_constraint_->min_distance;
      const auto override_iter =
          per_pair_min_distance_overrides_.find(pair_key);
      if (override_iter != per_pair_min_distance_overrides_.end()) {
        effective_min_distance = override_iter->second;
      }

      double recovery_target = effective_min_distance;
      if (non_worsening_collision_floor_enabled_) {
        const auto floor_iter = collision_pair_distance_floor_.find(pair_key);
        if (floor_iter != collision_pair_distance_floor_.end()) {
          recovery_target =
              std::min(floor_iter->second, effective_min_distance);
        }
      }
      if (!acceleration_limits_enabled_ && position_step_call_depth_ > 0 &&
          pair_index < position_step_collision_command_floor_distances_.size() &&
          std::isfinite(
              position_step_collision_command_floor_distances_[pair_index])) {
        recovery_target = std::max(
            recovery_target,
            position_step_collision_command_floor_distances_[pair_index] -
                kCollisionTolerance);
      }

      const double required_safe_distance =
          std::max(
              {collision_constraint_->min_distance,
               effective_min_distance + collision_repulsion_deadband_,
               recovery_target + kCollisionTolerance}) +
          sphere_broadphase_.safety_margin;

      double broadphase_lower_bound =
          -std::numeric_limits<double>::infinity();
      if (sphere_broadphase_enabled_ && sphere_broadphase_.is_built()) {
        const auto joint_a =
            collision_model->geometryObjects[pair.first].parentJoint;
        const auto joint_b =
            collision_model->geometryObjects[pair.second].parentJoint;
        if (joint_a < static_cast<pinocchio::JointIndex>(oMi.size()) &&
            joint_b < static_cast<pinocchio::JointIndex>(oMi.size())) {
          broadphase_lower_bound = sphere_broadphase_.compute_pair_lower_bound(
              pair.first, pair.second, oMi[joint_a].translation(),
              oMi[joint_a].rotation(), oMi[joint_b].translation(),
              oMi[joint_b].rotation());
        }
      }

      double motion_lower_bound =
          -std::numeric_limits<double>::infinity();
      if (sphere_broadphase_enabled_ && sphere_broadphase_.is_built() &&
          pair.first < sphere_broadphase_.size() &&
          pair.second < sphere_broadphase_.size()) {
        const auto joint_a =
            collision_model->geometryObjects[pair.first].parentJoint;
        const auto joint_b =
            collision_model->geometryObjects[pair.second].parentJoint;
        if (joint_a < static_cast<pinocchio::JointIndex>(oMi.size()) &&
            joint_b < static_cast<pinocchio::JointIndex>(oMi.size())) {
          const auto geometry_motion_bound =
              [&](const PostStepCollisionDistanceCertificate &entry,
                  std::size_t geometry_index,
                  pinocchio::JointIndex joint_index) {
                const Eigen::Vector3d translation_delta =
                    oMi[joint_index].translation() -
                    entry.joint_translations[joint_index];
                const Eigen::Matrix3d rotation_delta =
                    entry.joint_rotations[joint_index].transpose() *
                    oMi[joint_index].rotation();
                const double cosine = std::clamp(
                    (rotation_delta.trace() - 1.0) * 0.5, -1.0, 1.0);
                const double rotation_chord =
                    std::sqrt(2.0 * std::max(0.0, 1.0 - cosine));
                return translation_delta.norm() +
                       sphere_broadphase_.sphere(geometry_index).motion_radius *
                           rotation_chord;
              };

          if (nearest_certificate != nullptr) {
            const double cached_lower_bound =
                nearest_certificate
                    ->certified_signed_distance_lower_bounds[pair_index];
            if (std::isfinite(cached_lower_bound)) {
              motion_lower_bound =
                  cached_lower_bound -
                  geometry_motion_bound(*nearest_certificate, pair.first,
                                        joint_a) -
                  geometry_motion_bound(*nearest_certificate, pair.second,
                                        joint_b);
            }
          }
        }
      }

      const bool broadphase_certified =
          std::isfinite(broadphase_lower_bound) &&
          broadphase_lower_bound > required_safe_distance;
      const bool motion_bound_certified =
          std::isfinite(motion_lower_bound) &&
          motion_lower_bound > required_safe_distance;
      if (broadphase_certified || motion_bound_certified) {
        signed_distance_lower_bounds[pair_index] =
            std::max(broadphase_lower_bound, motion_lower_bound);
        if (motion_bound_certified && !broadphase_certified) {
          last_post_step_collision_motion_bound_culled_pairs_++;
        }
        if (non_worsening_collision_floor_enabled_ &&
            collision_pair_distance_floor_.find(pair_key) ==
                collision_pair_distance_floor_.end()) {
          collision_pair_distance_floor_.emplace(pair_key,
                                                 effective_min_distance);
        }
        continue;
      }

      last_post_step_collision_exact_distance_queries_++;
      pinocchio::computeDistance(*collision_model, *collision_data, pair_index);
      signed_distance_lower_bounds[pair_index] =
          collision_data->distanceResults[pair_index].min_distance;
    }

    PostStepCollisionDistanceCertificate certificate;
    certificate.q = current_q;
    certificate.enabled_pair_mask = enabled_pair_mask;
    certificate.certified_signed_distance_lower_bounds =
        signed_distance_lower_bounds;
    certificate.joint_translations.reserve(oMi.size());
    certificate.joint_rotations.reserve(oMi.size());
    for (const auto &joint_placement : oMi) {
      certificate.joint_translations.push_back(joint_placement.translation());
      certificate.joint_rotations.push_back(joint_placement.rotation());
    }
    if (post_step_collision_distance_cache_.size() >=
        kPostStepCollisionCacheCapacity) {
      post_step_collision_distance_cache_.erase(
          post_step_collision_distance_cache_.begin());
    }
    post_step_collision_distance_cache_.push_back(
        std::make_shared<const PostStepCollisionDistanceCertificate>(
            std::move(certificate)));
  }
  for (std::size_t pair_index = 0; pair_index < pair_count; ++pair_index) {
    if (enabled_pair_mask[pair_index] &&
        std::isfinite(signed_distance_lower_bounds[pair_index])) {
      collision_data->distanceResults[pair_index].min_distance =
          signed_distance_lower_bounds[pair_index];
    }
  }

  std::vector<double> margins(
      pair_count, std::numeric_limits<double>::quiet_NaN());
  for (std::size_t pair_index = 0; pair_index < pair_count; ++pair_index) {
    if (!enabled_pair_mask[pair_index]) {
      continue;
    }

    const double signed_distance = signed_distance_lower_bounds[pair_index];
    if (!std::isfinite(signed_distance)) {
      continue;
    }

    const auto &pair = collision_model->collisionPairs[pair_index];
    const auto &name_a =
        collision_model->geometryObjects[pair.first].name;
    const auto &name_b =
        collision_model->geometryObjects[pair.second].name;
    const std::string pair_key = canonical_pair_key(name_a, name_b);

    double effective_min_distance = collision_constraint_->min_distance;
    const auto override_iter =
        per_pair_min_distance_overrides_.find(pair_key);
    if (override_iter != per_pair_min_distance_overrides_.end()) {
      effective_min_distance = override_iter->second;
    }

    double recovery_target = effective_min_distance;
    if (non_worsening_collision_floor_enabled_) {
      auto floor_iter = collision_pair_distance_floor_.find(pair_key);
      if (floor_iter == collision_pair_distance_floor_.end()) {
        const double seeded =
            signed_distance < effective_min_distance
                ? std::min(effective_min_distance,
                           collision_structural_floor_)
                : effective_min_distance;
        floor_iter =
            collision_pair_distance_floor_.emplace(pair_key, seeded).first;
      }
      recovery_target =
          std::min(floor_iter->second, effective_min_distance);
    }
    if (!acceleration_limits_enabled_ && position_step_call_depth_ > 0 &&
        pair_index < position_step_collision_command_floor_distances_.size() &&
        std::isfinite(
            position_step_collision_command_floor_distances_[pair_index])) {
      recovery_target = std::max(
          recovery_target,
          position_step_collision_command_floor_distances_[pair_index] -
              kCollisionTolerance);
    }

    margins[pair_index] = signed_distance - recovery_target;
  }

  return margins;
#else
  (void)current_q;
  return {};
#endif
}

std::optional<double>
KinematicsSolver::evaluate_post_step_collision_distance_from_current_results() {
#ifdef PINOCCHIO_WITH_HPP_FCL
  auto *collision_model = robot_->collision_model();
  auto *collision_data = robot_->collision_data();
  if (collision_model == nullptr || collision_data == nullptr) {
    return std::nullopt;
  }

  const auto minimum_for_pairs =
      [&](const std::vector<std::size_t> &pair_indices)
      -> std::optional<double> {
    double minimum_distance = std::numeric_limits<double>::infinity();
    bool found = false;
    for (const std::size_t pair_index : pair_indices) {
      if (pair_index >= collision_model->collisionPairs.size()) {
        continue;
      }
      if (!collision_data->activeCollisionPairs.empty() &&
          !collision_data->activeCollisionPairs[pair_index]) {
        continue;
      }
      const double distance =
          collision_data->distanceResults[pair_index].min_distance;
      if (!std::isfinite(distance)) {
        continue;
      }
      minimum_distance = std::min(minimum_distance, distance);
      found = true;
    }
    return found ? std::optional<double>(minimum_distance) : std::nullopt;
  };

  const std::vector<std::size_t> targeted_pairs =
      get_post_step_rejection_pair_indices();
  if (!targeted_pairs.empty()) {
    return minimum_for_pairs(targeted_pairs);
  }

  std::vector<std::size_t> all_pairs(collision_model->collisionPairs.size());
  std::iota(all_pairs.begin(), all_pairs.end(), std::size_t{0});
  return minimum_for_pairs(all_pairs);
#else
  return std::nullopt;
#endif
}

void KinematicsSolver::capture_position_step_collision_command_floor(
    const Eigen::VectorXd &current_q) {
  position_step_collision_command_floor_distances_.clear();
#ifdef PINOCCHIO_WITH_HPP_FCL
  // A command-local floor can jump when a moving target starts a new command,
  // invalidating the previous tick's acceleration-feasible stopping proof.
  // Acceleration-limited steps use the persistent recovery floor and braking
  // certificate instead.
  if (acceleration_limits_enabled_) {
    return;
  }
  if (!collision_constraint_.has_value() ||
      !collision_constraint_->enabled) {
    return;
  }
  const std::vector<double> current_margins =
      evaluate_post_step_collision_recovery_margins(current_q);
  const auto *collision_model = robot_->collision_model();
  if (collision_model == nullptr || current_margins.empty()) {
    return;
  }

  position_step_collision_command_floor_distances_.assign(
      current_margins.size(), std::numeric_limits<double>::quiet_NaN());
  for (std::size_t pair_index = 0; pair_index < current_margins.size();
       ++pair_index) {
    const double margin = current_margins[pair_index];
    if (!std::isfinite(margin) ||
        pair_index >= collision_model->collisionPairs.size()) {
      continue;
    }

    const auto &pair = collision_model->collisionPairs[pair_index];
    const auto &name_a = collision_model->geometryObjects[pair.first].name;
    const auto &name_b = collision_model->geometryObjects[pair.second].name;
    const std::string pair_key = canonical_pair_key(name_a, name_b);
    double effective_min_distance = collision_constraint_->min_distance;
    const auto override_iter =
        per_pair_min_distance_overrides_.find(pair_key);
    if (override_iter != per_pair_min_distance_overrides_.end()) {
      effective_min_distance = override_iter->second;
    }
    double recovery_target = effective_min_distance;
    if (non_worsening_collision_floor_enabled_) {
      const auto floor_iter = collision_pair_distance_floor_.find(pair_key);
      if (floor_iter != collision_pair_distance_floor_.end()) {
        recovery_target =
            std::min(floor_iter->second, effective_min_distance);
      }
    }
    const double nominal_margin =
        margin + recovery_target - effective_min_distance;
    if (nominal_margin >= -kCollisionTolerance &&
        nominal_margin <= kCollisionRepulsionDeadband) {
      position_step_collision_command_floor_distances_[pair_index] =
          margin + recovery_target;
    }
  }
#else
  (void)current_q;
#endif
}

std::optional<double>
KinematicsSolver::evaluate_per_pair_override_violations(const Eigen::VectorXd &q) {
#ifdef PINOCCHIO_WITH_HPP_FCL
  if (per_pair_min_distance_overrides_.empty()) return std::nullopt;
  const auto *geom_model = robot_->collision_model();
  if (!geom_model) return std::nullopt;

  double worst_margin = std::numeric_limits<double>::infinity();
  bool any_checked = false;
  const auto &pairs = geom_model->collisionPairs;

  for (std::size_t pi = 0; pi < pairs.size(); ++pi) {
    // Skip excluded pairs.
    if (!collision_allowed_pair_mask_.empty() &&
        pi < collision_allowed_pair_mask_.size() &&
        !collision_allowed_pair_mask_[pi]) continue;

    const auto &cp = pairs[pi];
    const auto &name_a = geom_model->geometryObjects[cp.first].name;
    const auto &name_b = geom_model->geometryObjects[cp.second].name;
    const auto oit =
        per_pair_min_distance_overrides_.find(canonical_pair_key(name_a, name_b));
    if (oit == per_pair_min_distance_overrides_.end()) continue;

    // This pair has an active override — compute its exact distance.
    auto dist_opt = evaluate_min_collision_distance_targeted(q, {pi});
    if (!dist_opt.has_value() || !std::isfinite(*dist_opt)) continue;

    const double margin = *dist_opt - oit->second;  // negative → violated
    worst_margin = std::min(worst_margin, margin);
    any_checked = true;
  }

  return any_checked ? std::make_optional(worst_margin) : std::nullopt;
#else
  (void)q;
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

bool KinematicsSolver::set_collision_min_distance(double min_distance) {
  if (!collision_constraint_.has_value() || !collision_constraint_->enabled) {
    return false;
  }
  collision_constraint_->min_distance = std::max(0.0, min_distance);
  if (collision_constraint_->constraint_activation_multiplier > 0.0) {
    const double activation_distance =
        stall_config_.enabled
            ? std::max(collision_constraint_->min_distance,
                       stall_state_.nominal_min_distance)
            : collision_constraint_->min_distance;
    collision_constraint_->constraint_activation_margin =
        collision_constraint_->constraint_activation_multiplier *
        activation_distance;
  } else {
    collision_constraint_->constraint_activation_margin = 0.0;
  }
  post_step_collision_distance_cache_.clear();
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
    const double activation_distance =
        stall_config_.enabled
            ? std::max(config.min_distance, stall_state_.nominal_min_distance)
            : std::max(0.0, config.min_distance);
    config.constraint_activation_margin =
        config.constraint_activation_multiplier * activation_distance;
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
  last_collision_sphere_culled_pairs_ = 0;
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
    enable_sphere_broadphase(false);
    break;
  case CollisionTuningMode::kBalanced:
    // Conservative compromise:
    // - keep cache enabled to skip obviously far pairs in clear space
    // - refresh frequently enough to catch newly critical pairs in dense
    //   self-collision models before they can drift through the clearance shell
    // - but keep budget disabled (0), so once candidate pairs are selected we
    //   avoid early termination and preserve higher collision-distance fidelity
    //   than the speed preset.
    enable_collision_pair_cache(true, 5, 0.05, 256);
    set_collision_refinement_time_budget_us(0);
    set_proximity_gated_collision_activation_enabled(true);
    // Optional proximity-gated activation: rows are emitted only near
    // min_distance, with a conservative activation band.
    set_collision_constraint_activation_multiplier(5.0);
    enable_sphere_broadphase(true);
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
    enable_sphere_broadphase(true);
    break;
  }
}

CollisionTuningMode KinematicsSolver::get_collision_tuning_mode() const {
  return collision_tuning_mode_;
}

void KinematicsSolver::enable_sphere_broadphase(bool enable) {
#ifdef PINOCCHIO_WITH_HPP_FCL
  if (sphere_broadphase_enabled_ != enable) {
    post_step_collision_distance_cache_.clear();
  }
  sphere_broadphase_enabled_ = enable;
  if (enable && !sphere_broadphase_.is_built() && robot_->has_collision_geometry()) {
    sphere_broadphase_.build(*robot_->collision_model());
  }
#else
  (void)enable;
#endif
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
  collision_pair_distance_floor_.clear();
  post_step_collision_distance_cache_.clear();
  collision_cache_frozen_indices_.clear();
  collision_pair_cache_has_full_scan_ = false;
  collision_pair_cache_steps_since_refresh_ = 0;
  last_collision_pairs_considered_ = 0;
  last_collision_exact_distance_queries_ = 0;
  last_collision_bound_culled_pairs_ = 0;
  last_collision_sphere_culled_pairs_ = 0;
  last_collision_budget_exhausted_ = false;
  last_constraint_min_distance_ = std::numeric_limits<double>::infinity();
  last_constraint_min_recovery_margin_ =
      std::numeric_limits<double>::infinity();
  last_constraint_was_full_scan_ = false;
  last_collision_constraint_result_.reset();
  last_collision_constraint_row_dt_ =
      std::numeric_limits<double>::quiet_NaN();
  last_collision_constraint_q_ = Eigen::VectorXd();
  collision_stuck_counters_.clear();
  collision_stuck_last_distances_.clear();
}

void KinematicsSolver::set_collision_pair_min_distance(const std::string &link_a,
                                                        const std::string &link_b,
                                                        double min_distance,
                                                        bool activate_when_clear) {
  const auto *geom_model = robot_->collision_model();
  if (!geom_model) return;

  // Collect geometry indices whose parent frame name contains link_a or link_b.
  std::vector<std::size_t> geoms_a, geoms_b;
  for (std::size_t gi = 0; gi < geom_model->ngeoms; ++gi) {
    const auto &go = geom_model->geometryObjects[gi];
    const std::string &frame_name =
        robot_->model().frames[go.parentFrame].name;
    if (frame_name.find(link_a) != std::string::npos) geoms_a.push_back(gi);
    if (frame_name.find(link_b) != std::string::npos) geoms_b.push_back(gi);
  }

  for (std::size_t pi = 0; pi < geom_model->collisionPairs.size(); ++pi) {
    const auto &cp = geom_model->collisionPairs[pi];
    const bool fwd =
        std::find(geoms_a.begin(), geoms_a.end(), cp.first) != geoms_a.end() &&
        std::find(geoms_b.begin(), geoms_b.end(), cp.second) != geoms_b.end();
    const bool rev =
        std::find(geoms_b.begin(), geoms_b.end(), cp.first) != geoms_b.end() &&
        std::find(geoms_a.begin(), geoms_a.end(), cp.second) != geoms_a.end();
    if (fwd || rev) {
      const auto &name_a = geom_model->geometryObjects[cp.first].name;
      const auto &name_b = geom_model->geometryObjects[cp.second].name;
      const std::string key = canonical_pair_key(name_a, name_b);
      if (activate_when_clear) {
        // Deferred: store as pending; compute_bounds_for_pair promotes to active
        // the first time signed_distance >= min_distance (latch-on semantics).
        // Prevents immediate stall when called from inside the threshold.
        per_pair_deferred_overrides_[key] = min_distance;
        per_pair_min_distance_overrides_.erase(key);  // not active yet
      } else {
        // Immediate: override takes effect on the next solve tick.
        per_pair_min_distance_overrides_[key] = min_distance;
        per_pair_deferred_overrides_.erase(key);
      }
    }
  }
  post_step_collision_distance_cache_.clear();
}

void KinematicsSolver::clear_collision_pair_min_distance(const std::string &link_a,
                                                          const std::string &link_b) {
  const auto *geom_model = robot_->collision_model();
  if (!geom_model) return;

  std::vector<std::size_t> geoms_a, geoms_b;
  for (std::size_t gi = 0; gi < geom_model->ngeoms; ++gi) {
    const auto &go = geom_model->geometryObjects[gi];
    const std::string &frame_name =
        robot_->model().frames[go.parentFrame].name;
    if (frame_name.find(link_a) != std::string::npos) geoms_a.push_back(gi);
    if (frame_name.find(link_b) != std::string::npos) geoms_b.push_back(gi);
  }

  for (std::size_t pi = 0; pi < geom_model->collisionPairs.size(); ++pi) {
    const auto &cp = geom_model->collisionPairs[pi];
    const bool fwd =
        std::find(geoms_a.begin(), geoms_a.end(), cp.first) != geoms_a.end() &&
        std::find(geoms_b.begin(), geoms_b.end(), cp.second) != geoms_b.end();
    const bool rev =
        std::find(geoms_b.begin(), geoms_b.end(), cp.first) != geoms_b.end() &&
        std::find(geoms_a.begin(), geoms_a.end(), cp.second) != geoms_a.end();
    if (fwd || rev) {
      const auto &name_a = geom_model->geometryObjects[cp.first].name;
      const auto &name_b = geom_model->geometryObjects[cp.second].name;
      const std::string key = canonical_pair_key(name_a, name_b);
      per_pair_min_distance_overrides_.erase(key);
      per_pair_deferred_overrides_.erase(key);
    }
  }
  post_step_collision_distance_cache_.clear();
}

std::vector<std::pair<std::string, double>>
KinematicsSolver::get_collision_pair_min_distance_overrides() const {
  std::vector<std::pair<std::string, double>> result;
  result.reserve(per_pair_min_distance_overrides_.size() +
                 per_pair_deferred_overrides_.size());
  for (const auto &kv : per_pair_min_distance_overrides_) {
    result.emplace_back(kv.first, kv.second);
  }
  for (const auto &kv : per_pair_deferred_overrides_) {
    result.emplace_back(kv.first, kv.second);  // pending (not yet active)
  }
  return result;
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
    if (collision_constraint_.has_value() &&
        collision_constraint_->constraint_activation_multiplier > 0.0) {
      collision_constraint_->constraint_activation_margin =
          collision_constraint_->constraint_activation_multiplier *
          std::max(collision_constraint_->min_distance,
                   stall_state_.nominal_min_distance);
    }
    return;
  }
  stall_config_.enabled = true;
  stall_state_.nominal_min_distance = nom;
  stall_state_.current_min_distance = nom;
  stall_state_.consecutive_stall_steps = 0;
  if (collision_constraint_.has_value() &&
      collision_constraint_->constraint_activation_multiplier > 0.0) {
    collision_constraint_->constraint_activation_margin =
        collision_constraint_->constraint_activation_multiplier *
        std::max(collision_constraint_->min_distance,
                 stall_state_.nominal_min_distance);
  }
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
  stall_user_configured_ = true;
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
  const bool explicit_task_error =
      !result.task_errors.empty() && result.task_errors[0] > 1e-4;
  const bool has_task_error =
      explicit_task_error || primary_scale < 1e-6;
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
  if (scale_is_low &&
      (!result.saturated_joints.empty() || primary_scale < 1e-6)) {
    const double expansion_factor =
        std::max(0.0, 1.0 - 2.0 * primary_scale);
    const double step_expand = cfg.expand_rate * expansion_factor;
    // When completely infeasible, boost immediately to reduce the number
    // of stalled steps needed to reach useful expansion.
    const double boost_floor =
        (primary_scale < 1e-6) ? cfg.delta_max * 0.5 : 0.0;

    for (int i = 0; i < nv; ++i) {
      const bool eligible =
          !cfg.expand_only_saturated || is_saturated[i] ||
          (primary_scale < 1e-6 && result.saturated_joints.empty());
      if (eligible) {
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
  return compute_collision_constraint(std::max(dt_, 1e-6));
}

std::optional<KinematicsSolver::CollisionConstraintResult>
KinematicsSolver::compute_collision_constraint(double row_dt) {
  if (row_dt <= 0.0 || !std::isfinite(row_dt)) {
    throw std::invalid_argument(
        "collision constraint row_dt must be finite and positive");
  }
#ifdef PINOCCHIO_WITH_HPP_FCL
  const double constraint_dt = row_dt;
  const bool acceleration_braking_lookahead_active =
      acceleration_limits_enabled_ && position_step_call_depth_ > 0 &&
      previous_dq_.size() == robot_->nv() && previous_dq_.allFinite() &&
      acceleration_limits_.size() == robot_->nv() &&
      acceleration_limits_.allFinite() &&
      previous_dq_.cwiseAbs().maxCoeff() > constraint_tolerance_;
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
        !acceleration_braking_lookahead_active &&
        last_collision_constraint_result_.has_value() &&
        std::isfinite(last_collision_constraint_row_dt_) &&
        last_collision_constraint_row_dt_ == constraint_dt &&
        last_collision_constraint_q_.size() == robot_->nq() &&
        std::isfinite(last_constraint_min_distance_) &&
        !last_collision_budget_exhausted_ &&
        // Invalidate when the frozen (locked/excluded) DOF set changed: it alters
        // which pairs are controllable / selected, so the cached result is stale.
        collision_cache_frozen_indices_ == active_collision_lock_indices() &&
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
  last_collision_sphere_culled_pairs_ = 0;
  last_collision_budget_exhausted_ = false;
  last_constraint_min_recovery_margin_ =
      std::numeric_limits<double>::infinity();
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
  //
  // Two guards prevent stale data from permanently silencing the constraint:
  //   1. Refresh interval — forces a full scan every N steps (catches gradual
  //      approach that was missed while all pairs appeared far away).
  //   2. Delta-q threshold — if the robot moved significantly since the last
  //      full scan, last_constraint_min_distance_ may no longer reflect reality
  //      (e.g., arm folded from extended clear-space back toward the torso).
  //      In that case skip the fast path and recompute.
  // Guard 2 for fast-path: robot must not have moved substantially since the
  // last full scan (last_constraint_min_distance_ might be stale otherwise).
  // kFastPathMaxDqSqNorm ≈ 0.1 rad total joint change.
  constexpr double kFastPathMaxDqSqNorm = 0.01;
  const bool robot_q_stable =
      last_collision_constraint_q_.size() == robot_->nq() &&
      (robot_->get_current_configuration() - last_collision_constraint_q_)
              .squaredNorm() <= kFastPathMaxDqSqNorm;

  if (constraint_active && collision_pair_cache_enabled_ &&
      !acceleration_braking_lookahead_active &&
      collision_pair_cache_has_full_scan_ && robot_q_stable &&
      !last_collision_budget_exhausted_ &&
      std::isfinite(last_constraint_min_distance_) &&
      collision_constraint_->constraint_activation_enabled &&
      collision_constraint_->constraint_activation_margin > 0.0 &&
      collision_pair_cache_steps_since_refresh_ < collision_pair_cache_refresh_interval_) {
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

  struct BrakingPlacementSample {
    std::vector<Eigen::Vector3d> joint_translations;
    std::vector<Eigen::Matrix3d> joint_rotations;
  };
  std::vector<BrakingPlacementSample> braking_placement_samples;
  bool braking_lookahead_certified = !acceleration_braking_lookahead_active;
  if (acceleration_braking_lookahead_active &&
      sphere_broadphase_enabled_ && sphere_broadphase_.is_built()) {
    int braking_steps = 0;
    braking_lookahead_certified = true;
    for (int index = 0; index < robot_->nv(); ++index) {
      const double acceleration_step =
          acceleration_limits_[index] * constraint_dt;
      const double deadband = acceleration_step * 0.01;
      const double step = acceleration_step + deadband;
      if (step <= constraint_tolerance_) {
        if (std::abs(previous_dq_[index]) > constraint_tolerance_) {
          braking_lookahead_certified = false;
          break;
        }
        continue;
      }
      braking_steps = std::max(
          braking_steps,
          static_cast<int>(
              std::ceil(std::abs(previous_dq_[index]) / step - 1e-12)));
    }

    if (braking_lookahead_certified && braking_steps > 0) {
      Eigen::VectorXd rollout_q = robot_->get_current_configuration();
      Eigen::VectorXd rollout_velocity = previous_dq_;
      pinocchio::Data rollout_data(robot_->model());
      braking_placement_samples.reserve(braking_steps);
      for (int step_index = 0; step_index < braking_steps; ++step_index) {
        for (int index = 0; index < robot_->nv(); ++index) {
          const double acceleration_step =
              acceleration_limits_[index] * constraint_dt;
          const double deadband = acceleration_step * 0.01;
          const double step = acceleration_step + deadband;
          if (rollout_velocity[index] > step) {
            rollout_velocity[index] -= step;
          } else if (rollout_velocity[index] < -step) {
            rollout_velocity[index] += step;
          } else {
            rollout_velocity[index] = 0.0;
          }
        }
        rollout_q = pinocchio::integrate(
            robot_->model(), rollout_q, constraint_dt * rollout_velocity);
        pinocchio::forwardKinematics(robot_->model(), rollout_data, rollout_q);

        BrakingPlacementSample sample;
        sample.joint_translations.reserve(rollout_data.oMi.size());
        sample.joint_rotations.reserve(rollout_data.oMi.size());
        for (const auto &joint_placement : rollout_data.oMi) {
          sample.joint_translations.push_back(joint_placement.translation());
          sample.joint_rotations.push_back(joint_placement.rotation());
        }
        braking_placement_samples.push_back(std::move(sample));
      }
    }
  }

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

    const auto braking_path_clears_cutoff = [&](double cutoff) {
      if (!acceleration_braking_lookahead_active) {
        return true;
      }
      if (!braking_lookahead_certified ||
          !sphere_broadphase_enabled_ || !sphere_broadphase_.is_built()) {
        return false;
      }
      const auto joint_a_id = object_a.parentJoint;
      const auto joint_b_id = object_b.parentJoint;
      for (const auto &sample : braking_placement_samples) {
        if (joint_a_id >= static_cast<pinocchio::JointIndex>(
                              sample.joint_translations.size()) ||
            joint_b_id >= static_cast<pinocchio::JointIndex>(
                              sample.joint_translations.size())) {
          return false;
        }
        const double braking_lower_bound =
            sphere_broadphase_.compute_pair_lower_bound(
                pair.first, pair.second,
                sample.joint_translations[joint_a_id],
                sample.joint_rotations[joint_a_id],
                sample.joint_translations[joint_b_id],
                sample.joint_rotations[joint_b_id]);
        if (!std::isfinite(braking_lower_bound) ||
            braking_lower_bound <= cutoff) {
          return false;
        }
      }
      return true;
    };

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
      if (std::isfinite(lower_bound) && lower_bound > candidate_cutoff &&
          braking_path_clears_cutoff(
              candidate_cutoff + sphere_broadphase_.safety_margin)) {
        bound_culled = true;
      }
    }
    // Sphere broadphase culling: skip computeDistance() if sphere-sphere
    // lower bound exceeds the activation threshold + safety margin.
    // Safety: AABB sphere strictly contains mesh, so sphere_dist <= true_dist.
    if (!bound_culled && sphere_broadphase_enabled_ &&
        sphere_broadphase_.is_built() && constraint_active) {
      const double candidate_cutoff =
          collision_constraint_.has_value()
              ? collision_constraint_->min_distance +
                    collision_pair_cache_distance_margin_ +
                    sphere_broadphase_.safety_margin
              : std::numeric_limits<double>::infinity();

      const auto joint_a_id = object_a.parentJoint;
      const auto joint_b_id = object_b.parentJoint;
      const auto &oMi = robot_->data().oMi;
      if (joint_a_id < static_cast<pinocchio::JointIndex>(oMi.size()) &&
          joint_b_id < static_cast<pinocchio::JointIndex>(oMi.size())) {
        const double sphere_lb = sphere_broadphase_.compute_pair_lower_bound(
            pair.first, pair.second, oMi[joint_a_id].translation(),
            oMi[joint_a_id].rotation(), oMi[joint_b_id].translation(),
            oMi[joint_b_id].rotation());
        if (std::isfinite(sphere_lb) && sphere_lb > candidate_cutoff &&
            braking_path_clears_cutoff(candidate_cutoff)) {
          last_collision_sphere_culled_pairs_++;
          bound_culled = true;
        }
      }
    }
    if (bound_culled) {
      last_collision_bound_culled_pairs_++;
      continue;
    }

    // Rank candidates with distance-only queries. Nearest points are refined
    // below only for the globally closest debug pair and selected QP rows.
    // Cached subsets can contain dozens of candidates on dense robot models.
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

  // Skip collision pairs that no free DOF can move: both links are governed only
  // by locked/excluded joints (zero relative Jacobian). Constraining such a pair
  // is meaningless, can be infeasible if it is penetrating, and otherwise wastes
  // the limited constraint slots that should go to controllable pairs.
  const std::vector<int> &active_locks = active_collision_lock_indices();
  const std::unordered_set<int> frozen_vdofs(active_locks.begin(),
                                             active_locks.end());
  const pinocchio::Model &pin_model = robot_->model();
  auto link_has_free_dof = [&](std::size_t gidx) -> bool {
    const auto parent_joint = collision_model->geometryObjects[gidx].parentJoint;
    for (const auto jid : pin_model.supports[parent_joint]) {
      if (jid == 0) {  // universe / fixed base contributes no DOF
        continue;
      }
      const int vi = pin_model.idx_vs[jid];
      for (int k = 0; k < pin_model.nvs[jid]; ++k) {
        if (frozen_vdofs.find(vi + k) == frozen_vdofs.end()) {
          return true;
        }
      }
    }
    return false;
  };
  auto pair_uncontrollable = [&](std::size_t pidx) -> bool {
    if (frozen_vdofs.empty()) {
      return false;
    }
    return !link_has_free_dof(pairs[pidx].first) && !link_has_free_dof(pairs[pidx].second);
  };

  // ---- Pair selection: top-K with hysteresis ----
  // Sort all allowed candidates by distance (ascending).
  std::sort(allowed_candidates.begin(), allowed_candidates.end());

  // Uncontrollable pairs (both links governed only by frozen DOFs) keep their
  // place in allowed_candidates so they still contribute to the global minimum
  // distance signal (last_constraint_min_distance_, computed above) and to the
  // next-step candidate cache (built below) -- those drive collision caution
  // (adaptive step size, post-step rejection) and must reflect the true
  // geometry. They are excluded only from the constraint *slots* (a meaningless
  // zero-Jacobian row that would starve controllable pairs), via this
  // selectable view.
  std::vector<std::pair<double, std::size_t>> selectable_candidates;
  selectable_candidates.reserve(allowed_candidates.size());
  for (const auto &entry : allowed_candidates) {
    if (!pair_uncontrollable(entry.second)) {
      selectable_candidates.push_back(entry);
    }
  }

  const auto &config = *collision_constraint_;
  auto collision_recovery_distances_for_pair =
      [&](std::size_t pair_idx, double signed_distance,
          double *effective_min_distance_out,
          double *recovery_target_out) {
        double effective_min_distance = config.min_distance;
        const auto &pair = pairs[pair_idx];
        const auto &name_a =
            collision_model->geometryObjects[pair.first].name;
        const auto &name_b =
            collision_model->geometryObjects[pair.second].name;
        const std::string pair_key = canonical_pair_key(name_a, name_b);

        if (!per_pair_deferred_overrides_.empty()) {
          const auto deferred = per_pair_deferred_overrides_.find(pair_key);
          if (deferred != per_pair_deferred_overrides_.end() &&
              signed_distance >= deferred->second) {
            per_pair_min_distance_overrides_[pair_key] = deferred->second;
            per_pair_deferred_overrides_.erase(deferred);
            post_step_collision_distance_cache_.clear();
          }
        }
        if (!per_pair_min_distance_overrides_.empty()) {
          const auto override =
              per_pair_min_distance_overrides_.find(pair_key);
          if (override != per_pair_min_distance_overrides_.end()) {
            effective_min_distance = override->second;
          }
        }

        double recovery_target = effective_min_distance;
        if (non_worsening_collision_floor_enabled_ && !pair_key.empty()) {
          auto floor = collision_pair_distance_floor_.find(pair_key);
          if (floor == collision_pair_distance_floor_.end()) {
            const double seeded =
                (signed_distance < effective_min_distance)
                    ? std::min(effective_min_distance,
                               collision_structural_floor_)
                    : effective_min_distance;
            floor = collision_pair_distance_floor_.emplace(pair_key, seeded)
                        .first;
          }
          recovery_target =
              std::min(floor->second, effective_min_distance);
        }
        if (!acceleration_limits_enabled_ && position_step_call_depth_ > 0 &&
            pair_idx <
                position_step_collision_command_floor_distances_.size() &&
            std::isfinite(
                position_step_collision_command_floor_distances_[pair_idx])) {
          recovery_target = std::max(
              recovery_target,
              position_step_collision_command_floor_distances_[pair_idx] -
                  kCollisionTolerance);
        }
        if (effective_min_distance_out != nullptr) {
          *effective_min_distance_out = effective_min_distance;
        }
        *recovery_target_out = recovery_target;
      };

  const int max_k = config.max_constraints;

  // The nominal row budget is a performance control, not a safety limit.
  // Every controllable penetrating pair and every pair close enough to cross
  // its recovery floor during a row switch must remain in the QP. Reuse the
  // pair-switch hysteresis as a selection-only guard band; this does not alter
  // the continuous velocity-damper bounds.
  const double recovery_selection_margin = kCollisionPairSwitchHysteresis;
  const bool proximity_gating_active =
      config.constraint_activation_enabled &&
      config.constraint_activation_margin > 0.0;
  std::unordered_set<std::size_t> mandatory_pair_indices;
  for (const auto &[distance, pair_idx] : selectable_candidates) {
    double recovery_target = config.min_distance;
    collision_recovery_distances_for_pair(
        pair_idx, distance, nullptr, &recovery_target);
    const bool command_floor_active =
        !acceleration_limits_enabled_ && position_step_call_depth_ > 0 &&
        pair_idx < position_step_collision_command_floor_distances_.size() &&
        std::isfinite(
            position_step_collision_command_floor_distances_[pair_idx]);
    if ((!proximity_gating_active && distance < recovery_target) ||
        ((non_worsening_collision_floor_enabled_ || command_floor_active) &&
         distance <= recovery_target + recovery_selection_margin)) {
      mandatory_pair_indices.insert(pair_idx);
    }
  }

  // Determine the distance threshold below which a previous pair "sticks".
  // A previous pair is kept if its distance is within hysteresis of the
  // current K-th best candidate.
  double hysteresis_cutoff = std::numeric_limits<double>::infinity();
  if (!selectable_candidates.empty()) {
    const std::size_t kth = static_cast<std::size_t>(
        std::min(max_k, static_cast<int>(selectable_candidates.size())) - 1);
    hysteresis_cutoff = selectable_candidates[kth].first +
                        kCollisionPairSwitchHysteresis;
  }

  // Build the selected set: start with top-K candidates, then admit any
  // previous pairs that fall within hysteresis_cutoff.
  std::unordered_set<std::size_t> selected_set;
  for (int i = 0;
       i < max_k && i < static_cast<int>(selectable_candidates.size()); ++i) {
    selected_set.insert(selectable_candidates[i].second);
  }
  for (std::size_t prev_idx : last_collision_constraint_pair_indices_) {
    if (static_cast<int>(selected_set.size()) >= max_k) {
      break;
    }
    if (prev_idx >= pairs.size()) {
      continue;
    }
    if (pair_uncontrollable(prev_idx)) {
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
  selected_set.insert(mandatory_pair_indices.begin(),
                      mandatory_pair_indices.end());

  // Sort the selected set by distance to produce a deterministic order and
  // retain all mandatory rows even when they exceed the nominal budget.
  std::vector<std::pair<double, std::size_t>> selected_sorted;
  selected_sorted.reserve(selected_set.size());
  for (std::size_t idx : selected_set) {
    selected_sorted.emplace_back(
        collision_data->distanceResults[idx].min_distance, idx);
  }
  std::sort(selected_sorted.begin(), selected_sorted.end());

  const std::size_t selected_row_budget = std::max(
      static_cast<std::size_t>(max_k), mandatory_pair_indices.size());
  if (selected_sorted.size() > selected_row_budget) {
    std::vector<std::pair<double, std::size_t>> retained;
    retained.reserve(selected_row_budget);
    for (const auto &entry : selected_sorted) {
      if (mandatory_pair_indices.count(entry.second) != 0) {
        retained.push_back(entry);
      }
    }
    for (const auto &entry : selected_sorted) {
      if (retained.size() >= selected_row_budget) {
        break;
      }
      if (mandatory_pair_indices.count(entry.second) == 0) {
        retained.push_back(entry);
      }
    }
    std::sort(retained.begin(), retained.end());
    selected_sorted = std::move(retained);
  }

  // Optional proximity-gated row activation.
  // When disabled (margin <= 0), behavior is unchanged.
  if (config.constraint_activation_enabled &&
      config.constraint_activation_margin > 0.0) {
    const double activation_threshold =
        config.min_distance + config.constraint_activation_margin;
    selected_sorted.erase(
        std::remove_if(
            selected_sorted.begin(), selected_sorted.end(),
            [activation_threshold, &mandatory_pair_indices](
                const std::pair<double, std::size_t> &entry) {
              if (mandatory_pair_indices.count(entry.second) != 0) {
                return false;
              }
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

  const int nv = robot_->nv();
  const int num_selected = static_cast<int>(selected_sorted.size());

  CollisionConstraintResult result;
  result.jacobian.resize(num_selected, nv);
  result.lower_bounds.resize(num_selected);
  result.upper_bounds.resize(num_selected);

  // Per-pair velocity-damper bound computation (with continuous recovery ramp).
  auto compute_bounds_for_pair = [&](std::size_t pair_idx,
                                     double signed_distance,
                                     bool *stuck_out,
                                     double *recovery_margin_out)
      -> std::pair<double, double> {
    double effective_min_distance = config.min_distance;
    double recovery_target = config.min_distance;
    collision_recovery_distances_for_pair(
        pair_idx, signed_distance, &effective_min_distance, &recovery_target);
    if (recovery_margin_out != nullptr) {
      *recovery_margin_out = signed_distance - recovery_target;
    }

    // Detect stuck: non-penetrating but significantly inside min_distance for
    // multiple consecutive cycles AND the previous dq was near-zero.
    bool stuck_active = false;
    const bool deep_non_penetration =
        (signed_distance >= 0.0) &&
        (signed_distance < (recovery_target - kCollisionStuckBand));
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
    // Note: collision_repulsion_deadband_ defaults to kCollisionRepulsionDeadband
    // (3mm) but is tunable at runtime via set_collision_repulsion_deadband().
    // Setting it to 0 removes the discontinuity that causes boundary oscillation.
    const double repulsion_deadband  = collision_repulsion_deadband_;
    const double recovery_scale      = collision_recovery_scale_;
    const double max_sep_speed_nonpen = collision_max_sep_speed_nonpen_;
    double lower_bound = 0.0;
    if (signed_distance >= (recovery_target + repulsion_deadband)) {
      lower_bound =
          (recovery_target + config.tolerance - signed_distance) /
          constraint_dt;
    } else if (signed_distance >= recovery_target) {
      lower_bound = 0.0;
    } else {
      // Violated region: uniform continuous recovery ramp from the moment
      // signed_distance drops below the (non-worsening) recovery target. No
      // dead-zone — recovery force is active at all violation depths.
      const double desired =
          (recovery_target + config.tolerance - signed_distance) /
          constraint_dt;
      if (signed_distance >= 0.0) {
        // Non-penetrating: proportional recovery, capped.
        lower_bound = std::min(max_sep_speed_nonpen,
                               std::max(0.0, desired * recovery_scale));
      } else {
        // Penetrating: enforce a minimum recovery speed, still capped.
        lower_bound = std::min(kCollisionMaxSeparationSpeed, desired);
        lower_bound = std::max(lower_bound, kCollisionMinRecoverySpeed);
      }
    }

    // Stuck override: ensure a floor that can actually produce motion.
    if (stuck_active && signed_distance >= 0.0) {
      const double desired =
          (effective_min_distance + config.tolerance - signed_distance) /
          constraint_dt;
      lower_bound = std::max(
          lower_bound,
          std::min(max_sep_speed_nonpen,
                   std::max(kCollisionStuckRecoverySpeed,
                            desired * recovery_scale)));
    }

    const double upper_bound =
        (config.upper_distance - config.tolerance + signed_distance) /
        constraint_dt;
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

    double recovery_margin = std::numeric_limits<double>::infinity();
    auto [lb, ub] = compute_bounds_for_pair(
        pair_idx, signed_distance, nullptr, &recovery_margin);
    if (acceleration_limits_enabled_ &&
        acceleration_limits_.size() == nv &&
        std::isfinite(recovery_margin) && recovery_margin > 0.0) {
      double separating_acceleration_limit = 0.0;
      const std::vector<int> locked_indices =
          active_collision_lock_indices();
      for (int index = 0; index < nv; ++index) {
        if (std::find(locked_indices.begin(), locked_indices.end(), index) !=
            locked_indices.end()) {
          continue;
        }
        separating_acceleration_limit = std::max(
            separating_acceleration_limit,
            std::abs(result.jacobian(row, index)) *
                acceleration_limits_[index]);
      }
      const double braking_slack =
          std::max(0.0, recovery_margin - config.tolerance);
      const double braking_speed = std::max(
          0.0,
          std::sqrt(2.0 * separating_acceleration_limit * braking_slack) -
              separating_acceleration_limit * constraint_dt);
      const double braking_lower_bound = -braking_speed;
      lb = std::max(lb, braking_lower_bound);
    }
    if (std::isfinite(recovery_margin)) {
      last_constraint_min_recovery_margin_ =
          std::min(last_constraint_min_recovery_margin_, recovery_margin);
    }
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
  last_collision_constraint_row_dt_ = constraint_dt;
  last_collision_constraint_q_ = robot_->get_current_configuration();
  collision_cache_frozen_indices_ = active_collision_lock_indices();

  return result;
#else
  last_collision_debug_.reset();
  last_collision_debug_list_.clear();
  return std::nullopt;
#endif
}

std::optional<KinematicsSolver::CollisionVelocityConstraintLinearization>
KinematicsSolver::linearize_collision_velocity_constraint(double row_dt) {
  auto rows = compute_collision_constraint(row_dt);
  if (!rows.has_value()) {
    return std::nullopt;
  }

  CollisionVelocityConstraintLinearization linearization;
  linearization.coefficient_matrix = rows->jacobian;
  linearization.lower_bounds = rows->lower_bounds;
  linearization.upper_bounds = rows->upper_bounds;
  linearization.dt = row_dt;
  linearization.distance = rows->distance;
  linearization.object_a = rows->object_a;
  linearization.object_b = rows->object_b;
  linearization.point_a_world = rows->point_a_world;
  linearization.point_b_world = rows->point_b_world;
  return linearization;
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

  // Preserve the legacy continuous stopping bound when acceleration history is
  // not enforced.  With acceleration limits enabled, use the exact sampled-data
  // stopping distance so the outer-tick command remains recursively feasible.
  double vel_from_pos_lower = -position_margin_lower / dt;
  double vel_from_pos_upper = position_margin_upper / dt;

  const double vel_from_accel_lower =
      acceleration_limits_enabled_
          ? -discrete_stopping_velocity_limit(position_margin_lower,
                                               acceleration_limit, dt)
          : -std::sqrt(2 * acceleration_limit * position_margin_lower);
  const double vel_from_accel_upper =
      acceleration_limits_enabled_
          ? discrete_stopping_velocity_limit(position_margin_upper,
                                              acceleration_limit, dt)
          : std::sqrt(2 * acceleration_limit * position_margin_upper);

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
    const std::vector<Eigen::VectorXd> &goals,
    const std::vector<ObjectiveSolveConfig> &objective_configs,
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

    for (size_t objective_index = 0; objective_index < jacobians.size();
         ++objective_index) {
      auto &J = jacobians[objective_index];
      if (J.cols() != nv_clamp || J.rows() <= 0) {
        continue;
      }
      const bool goal_directed_clamp =
          objective_index < objective_configs.size() &&
          objective_configs[objective_index].use_goal_directed_limit_clamp;
      if (goal_directed_clamp) {
        if (objective_index >= goals.size() ||
            goals[objective_index].size() != J.rows()) {
          continue;
        }
        const double requested_direction =
            J.col(i).dot(goals[objective_index]);
        if ((clamp_lower && requested_direction < 0.0) ||
            (clamp_upper && requested_direction > 0.0)) {
          J.col(i).setZero();
        }
      } else {
        for (int row = 0; row < static_cast<int>(J.rows()); ++row) {
          double &value = J(row, i);
          if (clamp_lower && value < 0.0) {
            value = 0.0;
          }
          if (clamp_upper && value > 0.0) {
            value = 0.0;
          }
        }
      }
    }
  }
}

void KinematicsSolver::apply_position_step_primary_task_options(
    const PositionStepOptions &options, Task *task) {
  if (task == nullptr) {
    return;
  }
  if (options.primary_solve_mode != TaskSolveMode::kScale &&
      task->getSolveMode() != TaskSolveMode::kMinError) {
    task->setSolveMode(options.primary_solve_mode);
  }
  if (options.primary_allow_min_error_fallback) {
    task->setAllowMinErrorFallback(true);
  }
}

std::optional<VelocitySolverResult>
KinematicsSolver::apply_position_step_task_metric_projection(
    const Eigen::VectorXd &current_q, double outer_dt,
    const Eigen::VectorXd &first_tick_velocity,
    const std::vector<int> &velocity_lock_indices,
    const std::optional<TorsoPoseConstraintOptions> &torso_constraint,
    const std::vector<PositionStepPriorityConstraintSpec>
        &priority_constraints,
    Eigen::VectorXd &q_candidate) {
  if (!acceleration_limits_enabled_ || current_q.size() != robot_->nq() ||
      q_candidate.size() != robot_->nq()) {
    return std::nullopt;
  }

  const Eigen::VectorXd predictive_delta =
      pinocchio::difference(robot_->model(), current_q, q_candidate);
  const bool have_terminal_prediction =
      predictive_delta.allFinite() &&
      predictive_delta.squaredNorm() > kCollisionEscapeNormEps;
  const bool have_first_tick_prediction =
      first_tick_velocity.size() == robot_->nv() &&
      first_tick_velocity.allFinite() &&
      first_tick_velocity.squaredNorm() > kCollisionEscapeNormEps;
  if (!have_terminal_prediction && !have_first_tick_prediction) {
    return std::nullopt;
  }

  const double dt_safe = std::max(outer_dt, 1e-9);
  const Eigen::VectorXd terminal_predictive_velocity =
      have_terminal_prediction ? predictive_delta / dt_safe
                               : first_tick_velocity;
  const Eigen::VectorXd first_tick_predictive_velocity =
      have_first_tick_prediction ? first_tick_velocity
                                 : terminal_predictive_velocity;
  robot_->update_configuration(current_q);

  bool have_task_objective = false;
  for (auto &task : tasks_) {
    if (!task || !task->isActive()) {
      continue;
    }

    task->update(*robot_);
    const Eigen::MatrixXd jacobian = task->getJacobian();
    const Eigen::VectorXd commanded_velocity = task->getVelocity();
    if (jacobian.cols() != robot_->nv() || jacobian.rows() <= 0 ||
        commanded_velocity.size() != jacobian.rows()) {
      continue;
    }

    const Eigen::VectorXd terminal_task_velocity =
        jacobian * terminal_predictive_velocity;
    const Eigen::VectorXd first_tick_task_velocity =
        jacobian * first_tick_predictive_velocity;
    if (!terminal_task_velocity.allFinite() ||
        !first_tick_task_velocity.allFinite()) {
      continue;
    }
    Eigen::VectorXd projected_velocity =
        Eigen::VectorXd::Zero(commanded_velocity.size());
    const double terminal_prediction_weight =
        position_step_task_terminal_prediction_weight(*task, task->getError());
    const Eigen::VectorXd selected_task_velocity =
        first_tick_task_velocity +
        terminal_prediction_weight *
            (terminal_task_velocity - first_tick_task_velocity);
    const auto protected_spec = std::find_if(
        priority_constraints.begin(), priority_constraints.end(),
        [&](const PositionStepPriorityConstraintSpec &spec) {
          return spec.task.get() == task.get();
        });

    if (task->getType() == TaskType::FRAME_POSE &&
        commanded_velocity.size() == 6) {
      const bool has_linear_command =
          commanded_velocity.head<3>().squaredNorm() >
          kCollisionEscapeNormEps;
      const bool has_angular_command =
          commanded_velocity.tail<3>().squaredNorm() >
          kCollisionEscapeNormEps;
      if (has_linear_command) {
        projected_velocity.head<3>() = selected_task_velocity.head<3>();
      }
      if (has_linear_command || has_angular_command) {
        projected_velocity.tail<3>() = selected_task_velocity.tail<3>();
      }
    } else {
      if (commanded_velocity.squaredNorm() > kCollisionEscapeNormEps) {
        projected_velocity = selected_task_velocity;
      }
    }
    if (protected_spec != priority_constraints.end()) {
      if (commanded_velocity.size() >= 6) {
        if (protected_spec->position_tolerance > 0.0) {
          projected_velocity.head<3>() = commanded_velocity.head<3>();
        }
        if (protected_spec->orientation_tolerance > 0.0) {
          projected_velocity.segment<3>(3) =
              commanded_velocity.segment<3>(3);
        }
      } else if (commanded_velocity.size() >= 3) {
        const bool orientation_only =
            task->getType() == TaskType::FRAME_ORIENTATION;
        const double protected_tolerance =
            orientation_only ? protected_spec->orientation_tolerance
                             : protected_spec->position_tolerance;
        if (protected_tolerance > 0.0) {
          projected_velocity.head<3>() = commanded_velocity.head<3>();
        }
      }
    }
    task->setPositionStepTargetVelocity(projected_velocity);
    have_task_objective = true;
  }
  if (!have_task_objective) {
    return std::nullopt;
  }

  pending_velocity_lock_indices_ = velocity_lock_indices;
  pending_step_torso_constraint_ = torso_constraint;
  pending_step_validation_dt_ = dt_safe;
  pending_position_step_priority_constraints_ = priority_constraints;
  pending_reuse_current_kinematics_ = true;
  pending_position_step_acceleration_limits_ = true;
  VelocitySolverResult projected = solve_velocity(current_q, true);
  projected = retry_auto_task_layout_as_split_if_needed(
      current_q, std::move(projected), velocity_lock_indices,
      torso_constraint, dt_safe, priority_constraints, true);
  if (projected.joint_velocities.size() != robot_->nv() ||
      !projected.joint_velocities.allFinite()) {
    robot_->update_configuration(q_candidate);
    return projected;
  }

  q_candidate = pinocchio::integrate(
      robot_->model(), current_q, dt_safe * projected.joint_velocities);
  if (use_position_limits_) {
    auto [q_min, q_max] = robot_->get_joint_limits();
    project_scalar_configuration_to_true_joint_limits(
        q_candidate, q_min, q_max, velocity_to_config_index_cache(),
        robot_->nv());
  }
  robot_->update_configuration(q_candidate);
  return projected;
}

std::optional<Eigen::VectorXd>
KinematicsSolver::estimate_position_step_componentwise_outer_candidate(
    const Eigen::VectorXd &current_q,
    const Eigen::VectorXd &previous_applied_velocity,
    const PositionStepOptions &options, double outer_dt,
    const Eigen::VectorXd &terminal_q) {
  if (!acceleration_limits_enabled_ || terminal_q.size() != robot_->nq()) {
    return std::nullopt;
  }

  const bool collision_was_enabled =
      collision_constraint_.has_value() && collision_constraint_->enabled;
  if (collision_was_enabled) {
    collision_constraint_->enabled = false;
  }
  Eigen::VectorXd candidate = terminal_q;
  const bool applied = apply_position_step_outer_acceleration_limit(
      current_q, previous_applied_velocity, options, outer_dt, candidate);
  if (collision_was_enabled) {
    collision_constraint_->enabled = true;
  }
  robot_->update_configuration(terminal_q);
  if (!applied || candidate.size() != robot_->nq() ||
      !candidate.allFinite()) {
    return std::nullopt;
  }
  return candidate;
}

bool KinematicsSolver::apply_position_step_outer_acceleration_limit(
    const Eigen::VectorXd &current_q,
    const Eigen::VectorXd &previous_applied_velocity,
    const PositionStepOptions &options, double outer_dt,
    Eigen::VectorXd &q_candidate) {
  if (!acceleration_limits_enabled_ || q_candidate.size() != robot_->nq() ||
      previous_applied_velocity.size() != robot_->nv() ||
      acceleration_limits_.size() != robot_->nv()) {
    return false;
  }

  const double dt_safe = std::max(outer_dt, 1e-9);
  const int nv = robot_->nv();
  Eigen::VectorXd desired_velocity =
      pinocchio::difference(robot_->model(), current_q, q_candidate) / dt_safe;
  Eigen::VectorXd velocity_lower = Eigen::VectorXd::Constant(
      nv, -kUnboundedConstraintLimit);
  Eigen::VectorXd velocity_upper = Eigen::VectorXd::Constant(
      nv, kUnboundedConstraintLimit);
  const Eigen::VectorXd velocity_limits = robot_->get_velocity_limits();
  Eigen::VectorXd position_lower;
  Eigen::VectorXd position_upper;
  std::vector<int> velocity_to_config_index;
  if (use_position_limits_) {
    auto limits = robot_->get_joint_limits();
    position_lower = std::move(limits.first);
    position_upper = std::move(limits.second);
    velocity_to_config_index = velocity_to_config_index_cache();
  }
  for (int i = 0; i < nv; ++i) {
    const double acceleration_step = acceleration_limits_[i] * dt_safe;
    const double deadband = acceleration_step * 0.01;
    velocity_lower[i] =
        previous_applied_velocity[i] - acceleration_step - deadband;
    velocity_upper[i] =
        previous_applied_velocity[i] + acceleration_step + deadband;
    if (i < velocity_limits.size() && std::isfinite(velocity_limits[i]) &&
        velocity_limits[i] > 0.0) {
      velocity_lower[i] =
          std::max(velocity_lower[i], -velocity_limits[i]);
      velocity_upper[i] =
          std::min(velocity_upper[i], velocity_limits[i]);
    }
    if (use_position_limits_ &&
        i < static_cast<int>(velocity_to_config_index.size())) {
      const int q_index = velocity_to_config_index[i];
      if (q_index >= 0 && q_index < current_q.size() &&
          q_index < position_lower.size() && q_index < position_upper.size() &&
          std::isfinite(position_lower[q_index]) &&
          std::isfinite(position_upper[q_index])) {
        constexpr double kOuterPositionMargin = 1e-4;
        const double lower_margin =
            current_q[q_index] - position_lower[q_index] - kOuterPositionMargin;
        const double upper_margin =
            position_upper[q_index] - current_q[q_index] - kOuterPositionMargin;
        const double velocity_limit =
            i < velocity_limits.size() && std::isfinite(velocity_limits[i]) &&
                    velocity_limits[i] > 0.0
                ? velocity_limits[i]
                : kUnboundedConstraintLimit;
        const auto [position_velocity_lower, position_velocity_upper] =
            calculate_velocity_box_constraint(
                lower_margin, upper_margin, velocity_limit,
                acceleration_limits_[i], dt_safe, 0.0, 0.0);
        const double combined_lower =
            std::max(velocity_lower[i], position_velocity_lower);
        const double combined_upper =
            std::min(velocity_upper[i], position_velocity_upper);
        if (combined_lower <= combined_upper + constraint_tolerance_) {
          velocity_lower[i] = combined_lower;
          velocity_upper[i] = combined_upper;
        } else {
          // The caller can enter outside the controlled invariant set. In that
          // case the hard position limit takes precedence over continuity.
          velocity_lower[i] = position_velocity_lower;
          velocity_upper[i] = position_velocity_upper;
        }
      }
    }
  }

  const auto lock_velocity = [&](const std::vector<int> &indices) {
    for (const int index : indices) {
      if (index >= 0 && index < nv) {
        velocity_lower[index] = 0.0;
        velocity_upper[index] = 0.0;
      }
    }
  };
  lock_velocity(options.excluded_joint_indices);
  lock_velocity(options.locked_joint_indices);
  lock_velocity(options.integration_zero_velocity_indices);

  Eigen::VectorXd limited_velocity = desired_velocity;
  Eigen::VectorXd minimum_norm_feasible_velocity(nv);
  for (int i = 0; i < nv; ++i) {
    limited_velocity[i] = std::clamp(
        limited_velocity[i], velocity_lower[i], velocity_upper[i]);
    minimum_norm_feasible_velocity[i] =
        std::clamp(0.0, velocity_lower[i], velocity_upper[i]);
  }
  bool minimum_norm_velocity_satisfies_collision_rows = true;

  if (collision_constraint_.has_value() && collision_constraint_->enabled) {
    robot_->update_configuration(current_q);
    const auto collision_rows = compute_collision_constraint();
    if (collision_rows.has_value() && collision_rows->jacobian.rows() > 0) {
      const int collision_row_count =
          static_cast<int>(collision_rows->jacobian.rows());
      Eigen::MatrixXd constraints = Eigen::MatrixXd::Zero(
          nv + collision_row_count, nv);
      constraints.topRows(nv).setIdentity();
      constraints.bottomRows(collision_row_count) =
          collision_rows->jacobian;
      Eigen::VectorXd lower(nv + collision_row_count);
      Eigen::VectorXd upper(nv + collision_row_count);
      lower.head(nv) = velocity_lower;
      upper.head(nv) = velocity_upper;
      lower.tail(collision_row_count) = collision_rows->lower_bounds;
      upper.tail(collision_row_count) = collision_rows->upper_bounds;

      VelocitySolverConfig projection_config;
      projection_config.epsilon = constraint_tolerance_;
      projection_config.precision_threshold = tight_tolerance_;
      projection_config.iteration_limit = max_iterations_;
      projection_config.magnitude_limit = norm_threshold_;
      projection_config.stall_detection_count = max_zero_scale_iterations_;
      projection_config.regularization_config.epsilon = solver_tolerance_;
      projection_config.regularization_config.regularization_factor = damping_;
      ObjectiveSolveConfig objective_config;
      objective_config.solve_mode = TaskSolveMode::kMinError;
      objective_config.allow_min_error_fallback = false;
      const auto projection = computeMultiObjectiveVelocitySolutionEigen(
          {desired_velocity}, {Eigen::MatrixXd::Identity(nv, nv)},
          constraints, lower, upper, projection_config, {objective_config});
      if (static_cast<int>(projection.solution.size()) == nv) {
        const Eigen::Map<const Eigen::VectorXd> projected_velocity(
            projection.solution.data(), nv);
        const Eigen::VectorXd constraint_values =
            constraints * projected_velocity;
        const bool projection_feasible = projected_velocity.allFinite() &&
            (constraint_values.array() >=
             (lower.array() - 10.0 * constraint_tolerance_)).all() &&
            (constraint_values.array() <=
             (upper.array() + 10.0 * constraint_tolerance_)).all();
        if (projection_feasible) {
          limited_velocity = projected_velocity;
        }
      }
      const double max_outer_velocity_norm =
          options.max_configuration_step_norm > 0.0
              ? options.max_configuration_step_norm / dt_safe
              : std::numeric_limits<double>::infinity();
      if (limited_velocity.norm() >
          max_outer_velocity_norm + 10.0 * constraint_tolerance_) {
        const auto minimum_norm_projection =
            computeMultiObjectiveVelocitySolutionEigen(
                {Eigen::VectorXd::Zero(nv)},
                {Eigen::MatrixXd::Identity(nv, nv)}, constraints, lower,
                upper, projection_config, {objective_config});
        minimum_norm_velocity_satisfies_collision_rows = false;
        if (static_cast<int>(minimum_norm_projection.solution.size()) == nv) {
          const Eigen::Map<const Eigen::VectorXd> minimum_norm_velocity(
              minimum_norm_projection.solution.data(), nv);
          const Eigen::VectorXd minimum_norm_constraint_values =
              constraints * minimum_norm_velocity;
          const bool minimum_norm_projection_feasible =
              minimum_norm_velocity.allFinite() &&
              (minimum_norm_constraint_values.array() >=
               (lower.array() - 10.0 * constraint_tolerance_))
                  .all() &&
              (minimum_norm_constraint_values.array() <=
               (upper.array() + 10.0 * constraint_tolerance_))
                  .all();
          if (minimum_norm_projection_feasible) {
            minimum_norm_feasible_velocity = minimum_norm_velocity;
            minimum_norm_velocity_satisfies_collision_rows = true;
          }
        }
      }
    }
  }

  const double max_outer_velocity_norm =
      options.max_configuration_step_norm > 0.0
          ? options.max_configuration_step_norm / dt_safe
          : std::numeric_limits<double>::infinity();
  if (limited_velocity.norm() >
      max_outer_velocity_norm + 10.0 * constraint_tolerance_) {
    if (minimum_norm_velocity_satisfies_collision_rows &&
        minimum_norm_feasible_velocity.norm() <=
            max_outer_velocity_norm + 10.0 * constraint_tolerance_) {
      double feasible_fraction = 0.0;
      double infeasible_fraction = 1.0;
      for (int iteration = 0; iteration < 40; ++iteration) {
        const double fraction =
            0.5 * (feasible_fraction + infeasible_fraction);
        const Eigen::VectorXd trial_velocity =
            minimum_norm_feasible_velocity +
            fraction *
                (limited_velocity - minimum_norm_feasible_velocity);
        if (trial_velocity.norm() <= max_outer_velocity_norm) {
          feasible_fraction = fraction;
        } else {
          infeasible_fraction = fraction;
        }
      }
      limited_velocity =
          minimum_norm_feasible_velocity +
          feasible_fraction *
              (limited_velocity - minimum_norm_feasible_velocity);
    } else if (minimum_norm_velocity_satisfies_collision_rows) {
      // The norm cap is infeasible with the acceleration/collision rows. Keep
      // the minimum-norm hard-constraint solution instead of radially scaling
      // outside the acceleration box.
      limited_velocity = minimum_norm_feasible_velocity;
    }
  }

  const auto configuration_for_velocity =
      [&](const Eigen::VectorXd &reference_q,
          const Eigen::VectorXd &velocity) {
        Eigen::VectorXd candidate_q = pinocchio::integrate(
            robot_->model(), reference_q, dt_safe * velocity);
        if (use_position_limits_) {
          auto [q_min, q_max] = robot_->get_joint_limits();
          project_scalar_configuration_to_true_joint_limits(
              candidate_q, q_min, q_max, velocity_to_config_index_cache(),
              robot_->nv());
        }
        return candidate_q;
      };

  Eigen::VectorXd limited_q =
      configuration_for_velocity(current_q, limited_velocity);

  if (collision_constraint_.has_value() && collision_constraint_->enabled) {
    robot_->update_configuration(current_q);
    const auto current_recovery_margins =
        evaluate_post_step_collision_recovery_margins(current_q);
    const auto current_distance =
        evaluate_post_step_collision_distance_from_current_results();
    const double nominal_floor = collision_constraint_->min_distance;

    const auto collision_acceptable =
        [&](const Eigen::VectorXd &candidate,
            double recovery_margin_threshold) {
      robot_->update_configuration(candidate);
      const auto candidate_recovery_margins =
          evaluate_post_step_collision_recovery_margins(candidate);
      const auto candidate_distance =
          evaluate_post_step_collision_distance_from_current_results();
      if (!collision_recovery_margins_acceptable(
              current_recovery_margins, candidate_recovery_margins,
              recovery_margin_threshold)) {
        return false;
      }
      if (!current_distance.has_value() ||
          !std::isfinite(*current_distance) ||
          !candidate_distance.has_value() ||
          !std::isfinite(*candidate_distance)) {
        return true;
      }
      if (*current_distance >= nominal_floor - kCollisionTolerance) {
        return *candidate_distance >= nominal_floor - kCollisionTolerance;
      }
      return *candidate_distance >=
             *current_distance - kCollisionPenetrationWorsenTolerance;
    };

    const auto braking_velocity_from = [&](const Eigen::VectorXd &velocity) {
      Eigen::VectorXd braking_velocity = velocity;
      for (int i = 0; i < nv; ++i) {
        const double acceleration_step = acceleration_limits_[i] * dt_safe;
        const double deadband = acceleration_step * 0.01;
        const double step = acceleration_step + deadband;
        if (braking_velocity[i] > step) {
          braking_velocity[i] -= step;
        } else if (braking_velocity[i] < -step) {
          braking_velocity[i] += step;
        } else {
          braking_velocity[i] = 0.0;
        }
      }
      return braking_velocity;
    };

    const auto componentwise_braking_rollout_acceptable =
        [&](const Eigen::VectorXd &candidate_q,
            const Eigen::VectorXd &candidate_velocity,
            double minimum_recovery_margin,
            Eigen::VectorXd *first_q_out,
            Eigen::VectorXd *first_velocity_out) {
          if (!acceleration_limits_enabled_) {
            return true;
          }

          int braking_steps = 0;
          for (int i = 0; i < nv; ++i) {
            const double acceleration_step = acceleration_limits_[i] * dt_safe;
            const double deadband = acceleration_step * 0.01;
            const double step = acceleration_step + deadband;
            if (step <= constraint_tolerance_) {
              if (std::abs(candidate_velocity[i]) > constraint_tolerance_) {
                return false;
              }
              continue;
            }
            braking_steps = std::max(
                braking_steps,
                static_cast<int>(std::ceil(
                    std::abs(candidate_velocity[i]) / step - 1e-12)));
          }

          Eigen::VectorXd rollout_q = candidate_q;
          Eigen::VectorXd rollout_velocity = candidate_velocity;
          Eigen::VectorXd first_q;
          Eigen::VectorXd first_velocity;
          for (int step_index = 0; step_index < braking_steps; ++step_index) {
            rollout_velocity = braking_velocity_from(rollout_velocity);
            rollout_q =
                configuration_for_velocity(rollout_q, rollout_velocity);
            if (step_index == 0) {
              first_q = rollout_q;
              first_velocity = rollout_velocity;
            }
            if (!collision_acceptable(rollout_q, minimum_recovery_margin)) {
              return false;
            }
          }
          if (braking_steps == 0) {
            first_q = candidate_q;
            first_velocity = candidate_velocity;
          }
          if (first_q_out != nullptr) {
            *first_q_out = std::move(first_q);
          }
          if (first_velocity_out != nullptr) {
            *first_velocity_out = std::move(first_velocity);
          }
          return true;
        };

    const auto collision_aware_braking_rollout_acceptable =
        [&](const Eigen::VectorXd &candidate_q,
            const Eigen::VectorXd &candidate_velocity,
            double minimum_recovery_margin,
            Eigen::VectorXd *first_q_out,
            Eigen::VectorXd *first_velocity_out) {
          int braking_step_budget = 0;
          for (int i = 0; i < nv; ++i) {
            const double acceleration_step = acceleration_limits_[i] * dt_safe;
            const double deadband = acceleration_step * 0.01;
            const double step = acceleration_step + deadband;
            if (step <= constraint_tolerance_) {
              if (std::abs(candidate_velocity[i]) > constraint_tolerance_) {
                return false;
              }
              continue;
            }
            braking_step_budget += static_cast<int>(std::ceil(
                std::abs(candidate_velocity[i]) / step - 1e-12));
          }
          if (braking_step_budget == 0) {
            if (first_q_out != nullptr) {
              *first_q_out = candidate_q;
            }
            if (first_velocity_out != nullptr) {
              *first_velocity_out = candidate_velocity;
            }
            return true;
          }

          VelocitySolverConfig rollout_solver_config;
          rollout_solver_config.epsilon = constraint_tolerance_;
          rollout_solver_config.precision_threshold = tight_tolerance_;
          rollout_solver_config.iteration_limit = max_iterations_;
          rollout_solver_config.magnitude_limit = norm_threshold_;
          rollout_solver_config.stall_detection_count =
              max_zero_scale_iterations_;
          rollout_solver_config.regularization_config.epsilon =
              solver_tolerance_;
          rollout_solver_config.regularization_config.regularization_factor =
              damping_;
          ObjectiveSolveConfig rollout_objective_config;
          rollout_objective_config.solve_mode = TaskSolveMode::kMinError;
          rollout_objective_config.allow_min_error_fallback = false;

          Eigen::VectorXd rollout_q = candidate_q;
          Eigen::VectorXd rollout_velocity = candidate_velocity;
          Eigen::VectorXd first_q;
          Eigen::VectorXd first_velocity;
          bool have_first_step = false;
          const auto publish_first_step = [&]() {
            if (!have_first_step) {
              return;
            }
            if (first_q_out != nullptr) {
              *first_q_out = first_q;
            }
            if (first_velocity_out != nullptr) {
              *first_velocity_out = first_velocity;
            }
          };
          const auto fail_with_first_step = [&]() {
            publish_first_step();
            return false;
          };
          for (int step_index = 0; step_index < braking_step_budget;
               ++step_index) {
            Eigen::VectorXd successor_lower(nv);
            Eigen::VectorXd successor_upper(nv);
            for (int i = 0; i < nv; ++i) {
              const double acceleration_step =
                  acceleration_limits_[i] * dt_safe;
              const double deadband = acceleration_step * 0.01;
              successor_lower[i] =
                  rollout_velocity[i] - acceleration_step - deadband;
              successor_upper[i] =
                  rollout_velocity[i] + acceleration_step + deadband;
              if (i < velocity_limits.size() &&
                  std::isfinite(velocity_limits[i]) &&
                  velocity_limits[i] > 0.0) {
                successor_lower[i] =
                    std::max(successor_lower[i], -velocity_limits[i]);
                successor_upper[i] =
                    std::min(successor_upper[i], velocity_limits[i]);
              }
              if (use_position_limits_ &&
                  i < static_cast<int>(velocity_to_config_index.size())) {
                const int q_index = velocity_to_config_index[i];
                if (q_index >= 0 && q_index < rollout_q.size() &&
                    q_index < position_lower.size() &&
                    q_index < position_upper.size() &&
                    std::isfinite(position_lower[q_index]) &&
                    std::isfinite(position_upper[q_index])) {
                  constexpr double kOuterPositionMargin = 1e-4;
                  const double lower_margin =
                      rollout_q[q_index] - position_lower[q_index] -
                      kOuterPositionMargin;
                  const double upper_margin =
                      position_upper[q_index] - rollout_q[q_index] -
                      kOuterPositionMargin;
                  const double velocity_limit =
                      i < velocity_limits.size() &&
                              std::isfinite(velocity_limits[i]) &&
                              velocity_limits[i] > 0.0
                          ? velocity_limits[i]
                          : kUnboundedConstraintLimit;
                  const auto [position_velocity_lower,
                              position_velocity_upper] =
                      calculate_velocity_box_constraint(
                          lower_margin, upper_margin, velocity_limit,
                          acceleration_limits_[i], dt_safe, 0.0, 0.0);
                  const double combined_lower =
                      std::max(successor_lower[i], position_velocity_lower);
                  const double combined_upper =
                      std::min(successor_upper[i], position_velocity_upper);
                  if (combined_lower <=
                      combined_upper + constraint_tolerance_) {
                    successor_lower[i] = combined_lower;
                    successor_upper[i] = combined_upper;
                  } else {
                    successor_lower[i] = position_velocity_lower;
                    successor_upper[i] = position_velocity_upper;
                  }
                }
              }
            }
            const auto lock_successor_velocity =
                [&](const std::vector<int> &indices) {
                  for (const int index : indices) {
                    if (index >= 0 && index < nv) {
                      successor_lower[index] = 0.0;
                      successor_upper[index] = 0.0;
                    }
                  }
                };
            lock_successor_velocity(options.excluded_joint_indices);
            lock_successor_velocity(options.locked_joint_indices);
            lock_successor_velocity(
                options.integration_zero_velocity_indices);

            robot_->update_configuration(rollout_q);
            const auto rollout_collision_rows =
                compute_collision_constraint();
            if (!rollout_collision_rows.has_value() ||
                rollout_collision_rows->jacobian.rows() == 0) {
              return fail_with_first_step();
            }

            const int collision_row_count = static_cast<int>(
                rollout_collision_rows->jacobian.rows());
            Eigen::MatrixXd constraints = Eigen::MatrixXd::Zero(
                nv + collision_row_count, nv);
            constraints.topRows(nv).setIdentity();
            constraints.bottomRows(collision_row_count) =
                rollout_collision_rows->jacobian;
            Eigen::VectorXd lower(nv + collision_row_count);
            Eigen::VectorXd upper(nv + collision_row_count);
            lower.head(nv) = successor_lower;
            upper.head(nv) = successor_upper;
            lower.tail(collision_row_count) =
                rollout_collision_rows->lower_bounds;
            upper.tail(collision_row_count) =
                rollout_collision_rows->upper_bounds;

            Eigen::VectorXd terminal_lower = lower;
            terminal_lower.tail(collision_row_count) =
                terminal_lower.tail(collision_row_count)
                    .cwiseMax(Eigen::VectorXd::Zero(collision_row_count));
            const auto terminal_solution =
                computeMultiObjectiveVelocitySolutionEigen(
                    {braking_velocity_from(rollout_velocity)},
                    {Eigen::MatrixXd::Identity(nv, nv)}, constraints,
                    terminal_lower, upper, rollout_solver_config,
                    {rollout_objective_config});
            if (static_cast<int>(terminal_solution.solution.size()) == nv) {
              const Eigen::Map<const Eigen::VectorXd> terminal_velocity_map(
                  terminal_solution.solution.data(), nv);
              const Eigen::VectorXd terminal_velocity = terminal_velocity_map;
              const Eigen::VectorXd terminal_constraint_values =
                  constraints * terminal_velocity;
              const bool terminal_feasible = terminal_velocity.allFinite() &&
                  (terminal_constraint_values.array() >=
                   (terminal_lower.array() -
                    10.0 * constraint_tolerance_))
                      .all() &&
                  (terminal_constraint_values.array() <=
                   (upper.array() + 10.0 * constraint_tolerance_))
                      .all();
              if (terminal_feasible) {
                Eigen::VectorXd terminal_q = configuration_for_velocity(
                    rollout_q, terminal_velocity);
                const Eigen::VectorXd applied_terminal_velocity =
                    pinocchio::difference(robot_->model(), rollout_q,
                                          terminal_q) /
                    dt_safe;
                bool terminal_acceleration_feasible = true;
                for (int i = 0; i < nv; ++i) {
                  if (applied_terminal_velocity[i] <
                          successor_lower[i] -
                              10.0 * constraint_tolerance_ ||
                      applied_terminal_velocity[i] >
                          successor_upper[i] +
                              10.0 * constraint_tolerance_) {
                    terminal_acceleration_feasible = false;
                    break;
                  }
                }
                if (terminal_acceleration_feasible &&
                    collision_acceptable(terminal_q,
                                         minimum_recovery_margin)) {
                  robot_->update_configuration(terminal_q);
                  const auto terminal_collision_rows =
                      compute_collision_constraint();
                  if (!terminal_collision_rows.has_value() ||
                      terminal_collision_rows->jacobian.rows() == 0 ||
                      ((terminal_collision_rows->jacobian *
                        applied_terminal_velocity)
                               .array() >=
                           -10.0 * constraint_tolerance_)
                          .all()) {
                    rollout_q = std::move(terminal_q);
                    rollout_velocity = applied_terminal_velocity;
                    if (!have_first_step) {
                      first_q = rollout_q;
                      first_velocity = rollout_velocity;
                      have_first_step = true;
                    }
                    if (rollout_velocity.norm() <=
                        10.0 * constraint_tolerance_) {
                      if (first_q_out != nullptr) {
                        *first_q_out = std::move(first_q);
                      }
                      if (first_velocity_out != nullptr) {
                        *first_velocity_out = std::move(first_velocity);
                      }
                      return true;
                    }
                    continue;
                  }
                }
              }
            }

            const auto rollout_solution =
                computeMultiObjectiveVelocitySolutionEigen(
                    {Eigen::VectorXd::Zero(nv)},
                    {Eigen::MatrixXd::Identity(nv, nv)}, constraints, lower,
                    upper, rollout_solver_config,
                    {rollout_objective_config});
            if (static_cast<int>(rollout_solution.solution.size()) != nv) {
              return fail_with_first_step();
            }
            const Eigen::Map<const Eigen::VectorXd> next_velocity_map(
                rollout_solution.solution.data(), nv);
            const Eigen::VectorXd next_velocity = next_velocity_map;
            const Eigen::VectorXd constraint_values =
                constraints * next_velocity;
            const bool feasible = next_velocity.allFinite() &&
                (constraint_values.array() >=
                 (lower.array() - 10.0 * constraint_tolerance_))
                    .all() &&
                (constraint_values.array() <=
                 (upper.array() + 10.0 * constraint_tolerance_))
                    .all();
            if (!feasible) {
              return fail_with_first_step();
            }

            Eigen::VectorXd next_q =
                configuration_for_velocity(rollout_q, next_velocity);
            const Eigen::VectorXd applied_next_velocity =
                pinocchio::difference(robot_->model(), rollout_q, next_q) /
                dt_safe;
            for (int i = 0; i < nv; ++i) {
              if (applied_next_velocity[i] <
                      successor_lower[i] - 10.0 * constraint_tolerance_ ||
                  applied_next_velocity[i] >
                      successor_upper[i] + 10.0 * constraint_tolerance_) {
                return fail_with_first_step();
              }
            }
            if (!collision_acceptable(next_q, minimum_recovery_margin)) {
              return fail_with_first_step();
            }
            rollout_q = std::move(next_q);
            rollout_velocity = applied_next_velocity;
            if (!have_first_step) {
              first_q = rollout_q;
              first_velocity = rollout_velocity;
              have_first_step = true;
            }
          }
          publish_first_step();
          return false;
        };

    const auto braking_rollout_acceptable =
        [&](const Eigen::VectorXd &candidate_q,
            const Eigen::VectorXd &candidate_velocity,
            double minimum_recovery_margin,
            Eigen::VectorXd *first_q_out,
            Eigen::VectorXd *first_velocity_out) {
          return componentwise_braking_rollout_acceptable(
                     candidate_q, candidate_velocity, minimum_recovery_margin,
                     first_q_out, first_velocity_out) ||
                 collision_aware_braking_rollout_acceptable(
                     candidate_q, candidate_velocity, minimum_recovery_margin,
                     first_q_out, first_velocity_out);
        };

    const auto candidate_and_braking_rollout_acceptable =
        [&](const Eigen::VectorXd &candidate_q,
            const Eigen::VectorXd &candidate_velocity) {
          return collision_acceptable(candidate_q, 0.0) &&
                 braking_rollout_acceptable(candidate_q, candidate_velocity,
                                             kCollisionTolerance, nullptr,
                                             nullptr);
        };

    if (!candidate_and_braking_rollout_acceptable(limited_q,
                                                   limited_velocity)) {
      bool accepted_backoff = false;
      if (acceleration_limits_enabled_) {
        Eigen::VectorXd braking_q;
        Eigen::VectorXd braking_velocity;
        const bool full_braking_rollout_acceptable =
            braking_rollout_acceptable(
                current_q, previous_applied_velocity, 0.0, &braking_q,
                &braking_velocity);
        const bool have_safe_braking_step =
            braking_q.size() == robot_->nq() &&
            braking_velocity.size() == robot_->nv() &&
            braking_q.allFinite() && braking_velocity.allFinite();
        if ((full_braking_rollout_acceptable || have_safe_braking_step) &&
            collision_acceptable(braking_q, 0.0)) {
          Eigen::VectorXd best_velocity = braking_velocity;
          Eigen::VectorXd best_q = std::move(braking_q);
          double safe_fraction = 0.0;
          double unsafe_fraction = 1.0;
          // Three probes retain useful task motion to 12.5% resolution while
          // bounding the repeated braking-rollout work inside a WBC tick.
          for (int iteration = 0; iteration < 3; ++iteration) {
            const double fraction =
                0.5 * (safe_fraction + unsafe_fraction);
            const Eigen::VectorXd trial_velocity =
                braking_velocity +
                fraction * (limited_velocity - braking_velocity);
            Eigen::VectorXd trial_q =
                configuration_for_velocity(current_q, trial_velocity);
            if (candidate_and_braking_rollout_acceptable(trial_q,
                                                          trial_velocity)) {
              safe_fraction = fraction;
              best_velocity = trial_velocity;
              best_q = std::move(trial_q);
            } else {
              unsafe_fraction = fraction;
            }
          }
          limited_velocity = std::move(best_velocity);
          limited_q = std::move(best_q);
          accepted_backoff = true;
        }
      }
      if (!accepted_backoff && !acceleration_limits_enabled_) {
        for (const double fraction : kCollisionRejectionBackoffFractions) {
          const Eigen::VectorXd backoff_velocity =
              fraction * limited_velocity;
          Eigen::VectorXd backoff_q =
              configuration_for_velocity(current_q, backoff_velocity);
          if (collision_acceptable(backoff_q, 0.0)) {
            limited_velocity = backoff_velocity;
            limited_q = std::move(backoff_q);
            accepted_backoff = true;
            break;
          }
        }
      }
      if (!accepted_backoff) {
        limited_velocity.setZero();
        limited_q = current_q;
      }
    }
  }

  q_candidate = std::move(limited_q);
  robot_->update_configuration(q_candidate);
  return true;
}

bool KinematicsSolver::compute_position_step_continuity_brake(
    const Eigen::VectorXd &current_q,
    const Eigen::VectorXd &previous_applied_velocity,
    const PositionStepOptions &options, double outer_dt,
    Eigen::VectorXd &q_candidate) {
  if (!acceleration_limits_enabled_ ||
      previous_applied_velocity.size() != robot_->nv() ||
      acceleration_limits_.size() != robot_->nv()) {
    return false;
  }

  const double dt_safe = std::max(outer_dt, 1e-9);
  Eigen::VectorXd braking_velocity = previous_applied_velocity;
  for (int i = 0; i < robot_->nv(); ++i) {
    const double acceleration_step = acceleration_limits_[i] * dt_safe;
    const double deadband = acceleration_step * 0.01;
    const double step = acceleration_step + deadband;
    if (braking_velocity[i] > step) {
      braking_velocity[i] -= step;
    } else if (braking_velocity[i] < -step) {
      braking_velocity[i] += step;
    } else {
      braking_velocity[i] = 0.0;
    }
  }

  q_candidate = pinocchio::integrate(robot_->model(), current_q,
                                     dt_safe * braking_velocity);
  apply_position_step_outer_acceleration_limit(
      current_q, previous_applied_velocity, options, dt_safe, q_candidate);

  const Eigen::VectorXd applied_velocity =
      pinocchio::difference(robot_->model(), current_q, q_candidate) / dt_safe;
  if (applied_velocity.size() != robot_->nv() || !applied_velocity.allFinite()) {
    q_candidate = current_q;
    robot_->update_configuration(q_candidate);
    return false;
  }
  for (int i = 0; i < robot_->nv(); ++i) {
    const double allowed_change =
        acceleration_limits_[i] * dt_safe * 1.01 + 10.0 * constraint_tolerance_;
    if (std::abs(applied_velocity[i] - previous_applied_velocity[i]) >
        allowed_change) {
      q_candidate = current_q;
      robot_->update_configuration(q_candidate);
      return false;
    }
  }
  return applied_velocity.norm() > 10.0 * constraint_tolerance_;
}

std::optional<PositionIKResult>
KinematicsSolver::attempt_min_error_position_step_retry(
    const Eigen::VectorXd &entry_q, const PositionStepOptions &options,
    double step_dt, bool have_vel_result,
    const VelocitySolverResult &last_vel_result,
    const Eigen::VectorXd &q_after_primary,
    const std::function<bool(std::vector<std::pair<Task *, TaskSolveMode>> *)>
        &flip_primary_tasks,
    const std::function<PositionIKResult()> &rerun_step) {
  if (!options.primary_allow_min_error_fallback || suppress_min_error_step_retry_ ||
      !have_vel_result) {
    return std::nullopt;
  }

  const double applied_step_eps =
      stall_config_.dq_stall_eps * std::max(step_dt, 1e-9);
  const double applied_step_norm = (q_after_primary - entry_q).norm();
  const bool constraints_active =
      (collision_constraint_.has_value() && collision_constraint_->enabled) ||
      (com_constraint_.has_value() && com_constraint_->enabled);
  const bool scale_collapsed =
      last_vel_result.status == SolverStatus::kInfeasible &&
      is_primary_scale_collapsed_message(last_vel_result.status_message);
  const bool numerical_stall =
      last_vel_result.status == SolverStatus::kNumericalError && constraints_active;
  constexpr double kPrimaryScaleEps = 1e-4;
  const bool primary_scale_saturated =
      constraints_active && !last_vel_result.task_scales.empty() &&
      std::abs(last_vel_result.task_scales.front()) <= kPrimaryScaleEps;
  const bool soft_saturated =
      primary_scale_saturated &&
      (last_vel_result.status == SolverStatus::kSuccess ||
       last_vel_result.status == SolverStatus::kNoProgress ||
       last_vel_result.status == SolverStatus::kInfeasible);
  const bool primary_saturation_retry = scale_collapsed || soft_saturated;
  const bool zero_motion_numerical_retry =
      numerical_stall && applied_step_norm <= applied_step_eps;
  if (!primary_saturation_retry && !zero_motion_numerical_retry) {
    return std::nullopt;
  }

  const PositionStepMutableStateSnapshot retry_snapshot =
      capture_position_step_mutable_state();
  std::vector<std::pair<Task *, TaskSolveMode>> saved_modes;
  if (!flip_primary_tasks(&saved_modes)) {
    return std::nullopt;
  }

  const TaskLayout saved_layout = current_auto_task_layout_;
  if (runtime_config_.enable_auto_task_layout) {
    current_auto_task_layout_ = TaskLayout::kSplit;
    auto_layout_below_low_count_ = 0;
    for (auto &kv : pose_task_groups_) {
      if (kv.second && kv.second->auto_switch()) {
        kv.second->set_layout(TaskLayout::kSplit);
      }
    }
  }

  suppress_min_error_step_retry_ = true;
  PositionIKResult retry_result = rerun_step();
  suppress_min_error_step_retry_ = false;

  restore_task_solve_modes(saved_modes);
  if (runtime_config_.enable_auto_task_layout) {
    current_auto_task_layout_ = saved_layout;
    for (auto &kv : pose_task_groups_) {
      if (kv.second && kv.second->auto_switch()) {
        kv.second->set_layout(saved_layout);
      }
    }
  }

  const double retry_norm = (retry_result.q_solution - entry_q).norm();
  const bool accept_retry =
      retry_norm > applied_step_eps ||
      retry_result.status == SolverStatus::kSuccess;
  if (!accept_retry) {
    restore_position_step_mutable_state(retry_snapshot);
    return std::nullopt;
  }
  return retry_result;
}

KinematicsSolver::PositionStepMutableStateSnapshot
KinematicsSolver::capture_position_step_mutable_state() const {
  PositionStepMutableStateSnapshot snapshot;
  snapshot.robot_q = robot_->get_current_configuration();
  snapshot.task_states.reserve(tasks_.size());
  for (const auto &task : tasks_) {
    if (!task) {
      continue;
    }
    snapshot.task_states.push_back(
        {task.get(), task->getSolveMode(), task->getAllowMinErrorFallback(),
         task->getLastEffectiveMode(), task->getUsedMinErrorFallback()});
  }
  snapshot.current_auto_task_layout = current_auto_task_layout_;
  snapshot.auto_layout_below_low_count = auto_layout_below_low_count_;
  snapshot.auto_layout_has_feedback = auto_layout_has_feedback_;
  snapshot.auto_layout_binding_score = auto_layout_binding_score_;
  snapshot.advisor_scale_current = advisor_scale_current_;
  snapshot.advisor_scale_ratio_sum = advisor_scale_ratio_sum_;
  snapshot.advisor_scale_epoch_time_s = advisor_scale_epoch_time_s_;
  snapshot.advisor_scale_sample_count = advisor_scale_sample_count_;
  snapshot.previous_dq = previous_dq_;
  snapshot.position_step_merit_window_anchor =
      position_step_merit_window_anchor_;
  snapshot.position_step_merit_priority = position_step_merit_priority_;
  snapshot.position_step_stationary_anchor_blocks =
      position_step_stationary_anchor_blocks_;
  snapshot.position_step_merit_window_motion =
      position_step_merit_window_motion_;
  snapshot.position_step_merit_window_samples =
      position_step_merit_window_samples_;
  snapshot.position_step_merit_window_last_delta =
      position_step_merit_window_last_delta_;
  snapshot.position_step_merit_window_direction_reversals =
      position_step_merit_window_direction_reversals_;
  snapshot.position_step_merit_window_last_merits =
      position_step_merit_window_last_merits_;
  snapshot.position_step_merit_window_error_increases =
      position_step_merit_window_error_increases_;
  snapshot.position_step_stationary_guard_active =
      position_step_stationary_guard_active_;
  snapshot.position_step_stationary_guard_can_reopen =
      position_step_stationary_guard_can_reopen_;
  snapshot.stall_config = stall_config_;
  snapshot.stall_state = stall_state_;
  snapshot.stall_user_configured = stall_user_configured_;
  snapshot.elastic_band_config = elastic_band_config_;
  snapshot.elastic_band_state = elastic_band_state_;
  snapshot.collision_constraint = collision_constraint_;
  snapshot.per_pair_min_distance_overrides = per_pair_min_distance_overrides_;
  snapshot.per_pair_deferred_overrides = per_pair_deferred_overrides_;
  snapshot.collision_pair_distance_floor = collision_pair_distance_floor_;
  snapshot.position_step_collision_command_floor_distances =
      position_step_collision_command_floor_distances_;
  snapshot.collision_cache_frozen_indices = collision_cache_frozen_indices_;
  snapshot.last_collision_debug = last_collision_debug_;
  snapshot.last_collision_debug_list = last_collision_debug_list_;
  snapshot.last_collision_constraint_pair_indices =
      last_collision_constraint_pair_indices_;
  snapshot.collision_cached_candidate_pair_indices =
      collision_cached_candidate_pair_indices_;
  snapshot.collision_pair_bound_valid = collision_pair_bound_valid_;
  snapshot.collision_pair_last_signed_distance =
      collision_pair_last_signed_distance_;
  snapshot.collision_pair_last_rel_translation_norm =
      collision_pair_last_rel_translation_norm_;
  snapshot.collision_pair_last_rel_rotation =
      collision_pair_last_rel_rotation_;
  snapshot.collision_pair_cache_has_full_scan =
      collision_pair_cache_has_full_scan_;
  snapshot.collision_pair_cache_steps_since_refresh =
      collision_pair_cache_steps_since_refresh_;
  snapshot.last_collision_pairs_considered = last_collision_pairs_considered_;
  snapshot.last_collision_exact_distance_queries =
      last_collision_exact_distance_queries_;
  snapshot.last_collision_bound_culled_pairs =
      last_collision_bound_culled_pairs_;
  snapshot.last_collision_budget_exhausted = last_collision_budget_exhausted_;
  snapshot.last_constraint_min_distance = last_constraint_min_distance_;
  snapshot.last_constraint_min_recovery_margin =
      last_constraint_min_recovery_margin_;
  snapshot.last_constraint_was_full_scan = last_constraint_was_full_scan_;
  snapshot.last_collision_constraint_result =
      last_collision_constraint_result_;
  snapshot.last_collision_constraint_row_dt =
      last_collision_constraint_row_dt_;
  snapshot.last_collision_constraint_q = last_collision_constraint_q_;
  snapshot.post_step_collision_distance_cache =
      post_step_collision_distance_cache_;
  snapshot.last_post_step_collision_exact_distance_queries =
      last_post_step_collision_exact_distance_queries_;
  snapshot.last_post_step_collision_motion_bound_culled_pairs =
      last_post_step_collision_motion_bound_culled_pairs_;
  snapshot.last_collision_sphere_culled_pairs =
      last_collision_sphere_culled_pairs_;
  snapshot.last_solution_dq_norm = last_solution_dq_norm_;
  snapshot.collision_stuck_counters = collision_stuck_counters_;
  snapshot.collision_stuck_last_distances = collision_stuck_last_distances_;
  return snapshot;
}

void KinematicsSolver::restore_position_step_mutable_state(
    const PositionStepMutableStateSnapshot &snapshot) {
  if (snapshot.robot_q.size() == robot_->nq()) {
    robot_->update_configuration(snapshot.robot_q);
  }
  for (const auto &state : snapshot.task_states) {
    if (state.task == nullptr) {
      continue;
    }
    state.task->setSolveMode(state.solve_mode);
    state.task->setAllowMinErrorFallback(state.allow_min_error_fallback);
    state.task->setLastEffectiveMode(state.last_effective_mode);
    state.task->setUsedMinErrorFallback(state.used_min_error_fallback);
  }
  current_auto_task_layout_ = snapshot.current_auto_task_layout;
  auto_layout_below_low_count_ = snapshot.auto_layout_below_low_count;
  auto_layout_has_feedback_ = snapshot.auto_layout_has_feedback;
  auto_layout_binding_score_ = snapshot.auto_layout_binding_score;
  advisor_scale_current_ = snapshot.advisor_scale_current;
  advisor_scale_ratio_sum_ = snapshot.advisor_scale_ratio_sum;
  advisor_scale_epoch_time_s_ = snapshot.advisor_scale_epoch_time_s;
  advisor_scale_sample_count_ = snapshot.advisor_scale_sample_count;
  previous_dq_ = snapshot.previous_dq;
  position_step_merit_window_anchor_ =
      snapshot.position_step_merit_window_anchor;
  position_step_merit_priority_ = snapshot.position_step_merit_priority;
  position_step_stationary_anchor_blocks_ =
      snapshot.position_step_stationary_anchor_blocks;
  position_step_merit_window_motion_ =
      snapshot.position_step_merit_window_motion;
  position_step_merit_window_samples_ =
      snapshot.position_step_merit_window_samples;
  position_step_merit_window_last_delta_ =
      snapshot.position_step_merit_window_last_delta;
  position_step_merit_window_direction_reversals_ =
      snapshot.position_step_merit_window_direction_reversals;
  position_step_merit_window_last_merits_ =
      snapshot.position_step_merit_window_last_merits;
  position_step_merit_window_error_increases_ =
      snapshot.position_step_merit_window_error_increases;
  position_step_stationary_guard_active_ =
      snapshot.position_step_stationary_guard_active;
  position_step_stationary_guard_can_reopen_ =
      snapshot.position_step_stationary_guard_can_reopen;
  for (auto &kv : pose_task_groups_) {
    if (kv.second && kv.second->auto_switch()) {
      kv.second->set_layout(current_auto_task_layout_);
    }
  }
  stall_config_ = snapshot.stall_config;
  stall_state_ = snapshot.stall_state;
  stall_user_configured_ = snapshot.stall_user_configured;
  elastic_band_config_ = snapshot.elastic_band_config;
  elastic_band_state_ = snapshot.elastic_band_state;
  collision_constraint_ = snapshot.collision_constraint;
  per_pair_min_distance_overrides_ =
      snapshot.per_pair_min_distance_overrides;
  per_pair_deferred_overrides_ = snapshot.per_pair_deferred_overrides;
  collision_pair_distance_floor_ = snapshot.collision_pair_distance_floor;
  position_step_collision_command_floor_distances_ =
      snapshot.position_step_collision_command_floor_distances;
  collision_cache_frozen_indices_ = snapshot.collision_cache_frozen_indices;
  last_collision_debug_ = snapshot.last_collision_debug;
  last_collision_debug_list_ = snapshot.last_collision_debug_list;
  last_collision_constraint_pair_indices_ =
      snapshot.last_collision_constraint_pair_indices;
  collision_cached_candidate_pair_indices_ =
      snapshot.collision_cached_candidate_pair_indices;
  collision_pair_bound_valid_ = snapshot.collision_pair_bound_valid;
  collision_pair_last_signed_distance_ =
      snapshot.collision_pair_last_signed_distance;
  collision_pair_last_rel_translation_norm_ =
      snapshot.collision_pair_last_rel_translation_norm;
  collision_pair_last_rel_rotation_ =
      snapshot.collision_pair_last_rel_rotation;
  collision_pair_cache_has_full_scan_ =
      snapshot.collision_pair_cache_has_full_scan;
  collision_pair_cache_steps_since_refresh_ =
      snapshot.collision_pair_cache_steps_since_refresh;
  last_collision_pairs_considered_ = snapshot.last_collision_pairs_considered;
  last_collision_exact_distance_queries_ =
      snapshot.last_collision_exact_distance_queries;
  last_collision_bound_culled_pairs_ =
      snapshot.last_collision_bound_culled_pairs;
  last_collision_budget_exhausted_ = snapshot.last_collision_budget_exhausted;
  last_constraint_min_distance_ = snapshot.last_constraint_min_distance;
  last_constraint_min_recovery_margin_ =
      snapshot.last_constraint_min_recovery_margin;
  last_constraint_was_full_scan_ = snapshot.last_constraint_was_full_scan;
  last_collision_constraint_result_ =
      snapshot.last_collision_constraint_result;
  last_collision_constraint_row_dt_ =
      snapshot.last_collision_constraint_row_dt;
  last_collision_constraint_q_ = snapshot.last_collision_constraint_q;
  post_step_collision_distance_cache_ =
      snapshot.post_step_collision_distance_cache;
  last_post_step_collision_exact_distance_queries_ =
      snapshot.last_post_step_collision_exact_distance_queries;
  last_post_step_collision_motion_bound_culled_pairs_ =
      snapshot.last_post_step_collision_motion_bound_culled_pairs;
  last_collision_sphere_culled_pairs_ =
      snapshot.last_collision_sphere_culled_pairs;
  last_solution_dq_norm_ = snapshot.last_solution_dq_norm;
  collision_stuck_counters_ = snapshot.collision_stuck_counters;
  collision_stuck_last_distances_ = snapshot.collision_stuck_last_distances;
  pending_velocity_lock_indices_.clear();
  position_step_locked_indices_.clear();
  pending_step_torso_constraint_.reset();
  pending_reuse_current_kinematics_ = false;
  pending_step_validation_dt_.reset();
  pending_position_step_priority_constraints_.clear();
}

PositionIKResult KinematicsSolver::solve_position_step_with_preferred_lock(
    const Eigen::VectorXd &current_q, const Eigen::Matrix4d &target_pose,
    const std::string &frame_task_name, const PositionStepOptions &options) {
  PositionStepOptions continuity_options =
      make_preferred_lock_continuity_options(options);
  PositionStepOptions locked_options =
      make_preferred_lock_candidate_options(continuity_options);
  const PositionStepMutableStateSnapshot snapshot =
      capture_position_step_mutable_state();

  struct ScopedPreferredLockSuppression {
    KinematicsSolver *solver;
    bool saved = false;
    explicit ScopedPreferredLockSuppression(KinematicsSolver *solver_in)
        : solver(solver_in),
          saved(solver_in != nullptr
                    ? solver_in->suppress_preferred_lock_step_retry_
                    : false) {
      if (solver != nullptr) {
        solver->suppress_preferred_lock_step_retry_ = true;
      }
    }
    ~ScopedPreferredLockSuppression() {
      if (solver != nullptr) {
        solver->suppress_preferred_lock_step_retry_ = saved;
      }
    }
  } suppress(this);
  (void)suppress;

  auto now = []() { return std::chrono::high_resolution_clock::now(); };
  auto elapsed_ms = [](auto start) {
    return std::chrono::duration_cast<std::chrono::microseconds>(
               std::chrono::high_resolution_clock::now() - start)
               .count() /
           1000.0;
  };

  const auto candidate_start = now();
  PositionIKResult candidate = solve_position_step(
      current_q, target_pose, frame_task_name, locked_options);
  const double candidate_time_ms = elapsed_ms(candidate_start);

  auto compute_errors_at = [&](const Eigen::VectorXd &q) {
    PositionStepTargetErrorSummary out;
    if (q.size() != robot_->nq() || !q.allFinite()) {
      return out;
    }
    const Eigen::VectorXd saved_q = robot_->get_current_configuration();
    robot_->update_configuration(q);
    auto it = task_map_.find(frame_task_name);
    auto frame_task =
        (it != task_map_.end()) ? std::dynamic_pointer_cast<FrameTask>(it->second)
                                : nullptr;
    if (frame_task) {
      frame_task->setTargetPose(target_pose.block<3, 1>(0, 3),
                                target_pose.block<3, 3>(0, 0));
      frame_task->update(*robot_);
      const Eigen::VectorXd &error = frame_task->getError();
      if (error.size() == 3) {
        if (frame_task->getType() == TaskType::FRAME_ORIENTATION) {
          out.max_position_error = 0.0;
          out.max_orientation_error = error.head<3>().norm();
        } else {
          out.max_position_error = error.head<3>().norm();
          out.max_orientation_error = 0.0;
        }
      } else if (error.size() >= 6) {
        out.max_position_error = error.head<3>().norm();
        out.max_orientation_error = error.tail<3>().norm();
      }
    }
    robot_->update_configuration(saved_q);
    return out;
  };

  const PositionStepTargetErrorSummary entry_errors =
      compute_errors_at(current_q);
  const PositionStepTargetErrorSummary errors =
      compute_errors_at(candidate.q_solution);

  const double step_norm =
      (candidate.q_solution.size() == current_q.size())
          ? (candidate.q_solution - current_q).norm()
          : std::numeric_limits<double>::quiet_NaN();
  const bool accept = preferred_lock_candidate_acceptable(
      candidate, entry_errors, errors, step_norm, options);
  if (accept) {
    annotate_preferred_lock_result(candidate, candidate, errors, step_norm,
                                   candidate_time_ms, true);
    return candidate;
  }

  restore_position_step_mutable_state(snapshot);
  PositionIKResult fallback =
      solve_position_step(current_q, target_pose, frame_task_name,
                          continuity_options);
  annotate_preferred_lock_result(fallback, candidate, errors, step_norm,
                                 candidate_time_ms, false);
  return fallback;
}

PositionIKResult KinematicsSolver::solve_position_step_with_preferred_lock(
    const Eigen::VectorXd &current_q, const std::vector<TaskTarget> &targets,
    const PositionStepOptions &options) {
  PositionStepOptions continuity_options =
      make_preferred_lock_continuity_options(options);
  PositionStepOptions locked_options =
      make_preferred_lock_candidate_options(continuity_options);
  const PositionStepMutableStateSnapshot snapshot =
      capture_position_step_mutable_state();

  struct ScopedPreferredLockSuppression {
    KinematicsSolver *solver;
    bool saved = false;
    explicit ScopedPreferredLockSuppression(KinematicsSolver *solver_in)
        : solver(solver_in),
          saved(solver_in != nullptr
                    ? solver_in->suppress_preferred_lock_step_retry_
                    : false) {
      if (solver != nullptr) {
        solver->suppress_preferred_lock_step_retry_ = true;
      }
    }
    ~ScopedPreferredLockSuppression() {
      if (solver != nullptr) {
        solver->suppress_preferred_lock_step_retry_ = saved;
      }
    }
  } suppress(this);
  (void)suppress;

  auto now = []() { return std::chrono::high_resolution_clock::now(); };
  auto elapsed_ms = [](auto start) {
    return std::chrono::duration_cast<std::chrono::microseconds>(
               std::chrono::high_resolution_clock::now() - start)
               .count() /
           1000.0;
  };

  const auto candidate_start = now();
  PositionIKResult candidate =
      solve_position_step(current_q, targets, locked_options);
  const double candidate_time_ms = elapsed_ms(candidate_start);

  auto compute_errors_at = [&](const Eigen::VectorXd &q) {
    PositionStepTargetErrorSummary out;
    if (q.size() != robot_->nq() || !q.allFinite()) {
      return out;
    }
    const Eigen::VectorXd saved_q = robot_->get_current_configuration();
    robot_->update_configuration(q);
    double max_position = 0.0;
    double max_orientation = 0.0;
    bool saw_target = false;
    for (const auto &target : targets) {
      auto it = task_map_.find(target.task_name);
      if (it == task_map_.end() || !it->second) {
        continue;
      }
      const auto &task = it->second;
      if (auto frame_task = std::dynamic_pointer_cast<FrameTask>(task)) {
        frame_task->setTargetPose(target.target_pose.block<3, 1>(0, 3),
                                  target.target_pose.block<3, 3>(0, 0));
      } else if (auto absolute_task =
                     std::dynamic_pointer_cast<AbsoluteFrameTask>(task)) {
        if (target.has_secondary_target_pose) {
          absolute_task->set_target_from_arm_targets(
              target.target_pose, target.secondary_target_pose);
        } else {
          absolute_task->setTargetPose(target.target_pose.block<3, 1>(0, 3),
                                       target.target_pose.block<3, 3>(0, 0));
        }
      } else if (auto relative_task =
                     std::dynamic_pointer_cast<RelativeFrameTask>(task)) {
        relative_task->setTargetPose(target.target_pose.block<3, 1>(0, 3),
                                     target.target_pose.block<3, 3>(0, 0));
      } else {
        continue;
      }

      task->update(*robot_);
      const Eigen::VectorXd &error = task->getError();
      if (error.size() == 3) {
        bool is_orientation_only = false;
        if (auto frame_task = std::dynamic_pointer_cast<FrameTask>(task)) {
          is_orientation_only =
              frame_task->getType() == TaskType::FRAME_ORIENTATION;
        }
        if (is_orientation_only) {
          max_orientation = std::max(max_orientation, error.head<3>().norm());
        } else {
          max_position = std::max(max_position, error.head<3>().norm());
        }
        saw_target = true;
      } else if (error.size() >= 6) {
        max_position = std::max(max_position, error.head<3>().norm());
        max_orientation = std::max(max_orientation, error.tail<3>().norm());
        saw_target = true;
      }
    }
    if (saw_target) {
      out.max_position_error = max_position;
      out.max_orientation_error = max_orientation;
    }
    robot_->update_configuration(saved_q);
    return out;
  };

  const PositionStepTargetErrorSummary entry_errors =
      compute_errors_at(current_q);
  const PositionStepTargetErrorSummary errors =
      compute_errors_at(candidate.q_solution);

  const double step_norm =
      (candidate.q_solution.size() == current_q.size())
          ? (candidate.q_solution - current_q).norm()
          : std::numeric_limits<double>::quiet_NaN();
  const bool accept = preferred_lock_candidate_acceptable(
      candidate, entry_errors, errors, step_norm, options);
  if (accept) {
    annotate_preferred_lock_result(candidate, candidate, errors, step_norm,
                                   candidate_time_ms, true);
    return candidate;
  }

  restore_position_step_mutable_state(snapshot);
  PositionIKResult fallback =
      solve_position_step(current_q, targets, continuity_options);
  annotate_preferred_lock_result(fallback, candidate, errors, step_norm,
                                 candidate_time_ms, false);
  return fallback;
}

void KinematicsSolver::sort_tasks_by_priority() {
  if (std::is_sorted(
          tasks_.begin(), tasks_.end(),
          [](const std::shared_ptr<Task> &a, const std::shared_ptr<Task> &b) {
            return a->getPriority() <= b->getPriority();
          })) {
    return;
  }
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
  select_auto_task_layout();
  struct ClearPendingVelocityLocks {
    KinematicsSolver *solver;
    ~ClearPendingVelocityLocks() {
      if (solver != nullptr) {
        solver->pending_velocity_lock_indices_.clear();
        solver->pending_step_torso_constraint_.reset();
        solver->pending_reuse_current_kinematics_ = false;
        solver->pending_step_validation_dt_.reset();
        solver->pending_position_step_priority_constraints_.clear();
        solver->pending_position_step_acceleration_limits_ = false;
      }
    }
  } clear_pending_locks{this};
  (void)clear_pending_locks;
  // Timing fields default to 0.0; only populate when timing is enabled.
  const Eigen::VectorXd q_eval =
      (current_q.size() > 0) ? current_q : robot_->get_current_configuration();

  if (q_eval.size() != robot_->nq()) {
    result.status = SolverStatus::kInvalidInput;
    result.status_message =
        "current configuration size does not match robot nq in solve_velocity";
    return result;
  }
  if (!q_eval.allFinite()) {
    result.status = SolverStatus::kNonFiniteInput;
    result.status_message =
        "current configuration contains non-finite values in solve_velocity";
    result.solution.assign(static_cast<std::size_t>(robot_->nv()), 0.0);
    result.joint_velocities = Eigen::VectorXd::Zero(robot_->nv());
    result.limits_applied = apply_limits;
    last_solution_dq_norm_ = 0.0;
    return result;
  }

  // Use provided configuration or robot's current
  if (current_q.size() > 0) {
    if (current_q.size() != robot_->nq()) {
      result.status = SolverStatus::kInvalidInput;
      result.status_message =
          "current_q size does not match robot nq in solve_velocity";
      return result;
    }
    const Eigen::VectorXd &robot_q = robot_->get_current_configuration();
    const bool can_reuse_current_kinematics =
        pending_reuse_current_kinematics_ && robot_q.size() == current_q.size() &&
        (robot_q - current_q).squaredNorm() <= 1e-24;
    if (can_reuse_current_kinematics) {
      result.pinocchio_kinematics_time_ms = 0.0;
    } else if (timing) {
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
  auto &goals = scratch_goals_;
  auto &jacobians = scratch_jacobians_;
  auto &objective_configs = scratch_objective_configs_;
  auto &objective_tasks = scratch_objective_tasks_;
  auto &excluded_union = scratch_excluded_union_;
  goals.clear();
  jacobians.clear();
  objective_configs.clear();
  objective_tasks.clear();
  excluded_union.clear();
  goals.reserve(tasks_.size());
  jacobians.reserve(tasks_.size());
  objective_configs.reserve(tasks_.size());
  objective_tasks.reserve(tasks_.size());
  excluded_union.reserve(tasks_.size());
  std::vector<int> task_exclusion_counts(
      static_cast<std::size_t>(robot_->nv()), 0);
  std::vector<bool> project_collision_tangent_objective;
  project_collision_tangent_objective.reserve(tasks_.size());
  std::vector<bool> project_elastic_collision_recovery_objective;
  project_elastic_collision_recovery_objective.reserve(tasks_.size());
  int active_task_count = 0;

  int current_priority = std::numeric_limits<int>::min();
  auto &group_tasks = scratch_group_tasks_;
  group_tasks.clear();
  group_tasks.reserve(tasks_.size());

  auto flush_group = [&]() {
    if (group_tasks.empty()) {
      return;
    }

    bool all_scale_family = true;
    bool all_strict_scale = true;
    bool all_min_error = true;
    bool fallback_value = group_tasks.front()->getAllowMinErrorFallback();
    bool fallback_consistent = true;
    bool use_goal_directed_limit_clamp = false;
    for (const auto &task : group_tasks) {
      const auto mode = task->getSolveMode();
      all_scale_family =
          all_scale_family &&
          (mode == TaskSolveMode::kScale || mode == TaskSolveMode::kScaleElastic);
      all_strict_scale = all_strict_scale && mode == TaskSolveMode::kScale;
      all_min_error = all_min_error && (mode == TaskSolveMode::kMinError);
      fallback_consistent =
          fallback_consistent &&
          (task->getAllowMinErrorFallback() == fallback_value);
      use_goal_directed_limit_clamp =
          use_goal_directed_limit_clamp || task->usesGoalDirectedLimitClamp();
    }

    const bool combine_group =
        fallback_consistent && (all_scale_family || all_min_error);
    if (combine_group) {
      int total_rows = 0;
      for (const auto &task : group_tasks) {
        total_rows += task->getDimension();
      }
      Eigen::VectorXd combined_goal(total_rows);
      Eigen::MatrixXd combined_jac(total_rows, robot_->nv());
      combined_goal.setZero();
      combined_jac.setZero();
      int offset = 0;
      for (const auto &task : group_tasks) {
        Eigen::VectorXd g = task->getVelocity();
        Eigen::MatrixXd J = task->getJacobian();
        if (g.rows() > 0) {
          combined_goal.segment(offset, g.rows()) = g;
          combined_jac.block(offset, 0, J.rows(), robot_->nv()) = J;
          offset += static_cast<int>(g.rows());
        }
      }
      goals.push_back(std::move(combined_goal));
      jacobians.push_back(std::move(combined_jac));
      objective_configs.push_back(ObjectiveSolveConfig{
          current_priority,
          all_min_error ? TaskSolveMode::kMinError : TaskSolveMode::kScale,
          all_min_error ? false : fallback_value,
          use_goal_directed_limit_clamp,
      });
      objective_tasks.push_back(nullptr);
      project_collision_tangent_objective.push_back(all_strict_scale);
      project_elastic_collision_recovery_objective.push_back(
          all_scale_family && !all_strict_scale);
    } else {
      for (const auto &task : group_tasks) {
        goals.push_back(task->getVelocity());
        jacobians.push_back(task->getJacobian());
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
            task->usesGoalDirectedLimitClamp(),
        });
        objective_tasks.push_back(task);
        project_collision_tangent_objective.push_back(
            task->getSolveMode() == TaskSolveMode::kScale);
        project_elastic_collision_recovery_objective.push_back(
            task->getSolveMode() == TaskSolveMode::kScaleElastic);
      }
    }

    group_tasks.clear();
  };

  for (const auto &task : tasks_) {
    if (!task->isActive()) {
      continue;
    }
    active_task_count++;
    std::unordered_set<int> task_exclusions;
    for (int idx : task->get_excluded_joint_indices()) {
      if (idx >= 0 && idx < robot_->nv() &&
          task_exclusions.insert(idx).second) {
        task_exclusion_counts[static_cast<std::size_t>(idx)]++;
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
  }
  flush_group();

  // Task-local exclusions define ownership of each objective, not global DoF
  // availability. A hard constraint may use a joint whenever at least one
  // active task owns it; only joints excluded by every active task are removed
  // from global constraint rows.
  if (active_task_count > 0) {
    for (int idx = 0; idx < robot_->nv(); ++idx) {
      if (task_exclusion_counts[static_cast<std::size_t>(idx)] ==
          active_task_count) {
        excluded_union.insert(idx);
      }
    }
  }

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

  Eigen::MatrixXd contact_P_c;
  const bool use_contact_projection = !contact_frames_.empty();
  if (use_contact_projection) {
    contact_P_c = compute_contact_projector();
    for (auto &J : jacobians) {
      J = J * contact_P_c;
    }
  }

  // Near a joint position limit, zero Jacobian entries that would command
  // motion further into that limit. This avoids whole-task SNS scale collapse
  // and preserves partial solutions through remaining DOFs.
  if (apply_limits && use_position_limits_) {
    clamp_jacobians_near_joint_limits(jacobians, goals, objective_configs,
                                      velocity_to_config_index);
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

  std::optional<ConstraintBlock> joint_limit_non_worsening_result =
      std::nullopt;
  if (apply_limits && use_position_limits_ &&
      runtime_config_.joint_limit_non_worsening_enabled) {
    joint_limit_non_worsening_result = build_joint_limit_non_worsening_rows(
        *robot_, velocity_to_config_index, excluded_union,
        runtime_config_.joint_limit_non_worsening_margin,
        acceleration_limits_enabled_, acceleration_limits_, dt_,
        joint_limit_non_worsening_lower_modes_,
        joint_limit_non_worsening_upper_modes_);
    if (use_contact_projection &&
        joint_limit_non_worsening_result.has_value()) {
      joint_limit_non_worsening_result->jacobian =
          joint_limit_non_worsening_result->jacobian * contact_P_c;
    }
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
    result.collision_sphere_culled_pairs = last_collision_sphere_culled_pairs_;
    result.collision_budget_exhausted = last_collision_budget_exhausted_;
  }
  if (collision_constraint_result.has_value() && !excluded_union.empty()) {
    zero_excluded_columns(collision_constraint_result->jacobian, excluded_union);
  }
  if (use_contact_projection && collision_constraint_result.has_value()) {
    collision_constraint_result->jacobian =
        collision_constraint_result->jacobian * contact_P_c;
  }

  // Preserve original soft residuals for weighted diagnostics/fallback while
  // composing hard-constraint tangent projections on their Jacobians.
  std::vector<Eigen::VectorXd> constrained_weighted_goals;
  std::vector<Eigen::MatrixXd> constrained_weighted_jacobians;
  bool constraint_projection_applied = false;
  const bool preserve_for_weighted =
      runtime_config_.weighted_advisor_enabled ||
      runtime_config_.weighted_fallback_enabled;

  if (joint_limit_non_worsening_result.has_value()) {
    std::vector<bool> project_limit_tangent_objective;
    project_limit_tangent_objective.reserve(objective_configs.size());
    for (const auto &objective : objective_configs) {
      project_limit_tangent_objective.push_back(
          objective.solve_mode == TaskSolveMode::kScale);
    }
    const auto &limit_rows = *joint_limit_non_worsening_result;
    constraint_projection_applied =
        project_objectives_into_constraint_tangent_space(
            goals, jacobians, project_limit_tangent_objective,
            limit_rows.jacobian, limit_rows.lower_bounds,
            constraint_tolerance_, limit_rows.prefer_exact_feasibility_rows,
            solver_tolerance_, true,
            preserve_for_weighted ? &constrained_weighted_goals : nullptr,
            preserve_for_weighted ? &constrained_weighted_jacobians : nullptr);
  }

  // Project task objectives into the collision tangent space whenever their
  // unconstrained velocity would violate an active collision row. The projected
  // goal is recomputed from the achievable tangent velocity, so SCALE does not
  // collapse on an objective component that was intentionally removed. A task
  // that already satisfies the separating lower bound is left unchanged.
  bool collision_projection_applied = false;
  if (apply_limits && collision_constraint_result.has_value()) {
    const auto &coll = collision_constraint_result.value();
    // Strict SCALE always projects at an active boundary. SCALE_ELASTIC keeps
    // its normal elastic tradeoff unless contact recovery requires a positive
    // separating velocity; projecting it earlier stalls useful tangent motion.
    const std::size_t collision_row_count = std::min(
        last_collision_debug_list_.size(),
        static_cast<std::size_t>(coll.lower_bounds.size()));
    bool contact_recovery_required = false;
    for (std::size_t row = 0; row < collision_row_count; ++row) {
      if (last_collision_debug_list_[row].distance <= kCollisionTolerance &&
          coll.lower_bounds(static_cast<Eigen::Index>(row)) >
              constraint_tolerance_) {
        contact_recovery_required = true;
        break;
      }
    }
    if (contact_recovery_required) {
      for (std::size_t index = 0;
           index < project_collision_tangent_objective.size(); ++index) {
        project_collision_tangent_objective[index] =
            project_collision_tangent_objective[index] ||
            project_elastic_collision_recovery_objective[index];
      }
    }
    collision_projection_applied =
        project_objectives_into_constraint_tangent_space(
            goals, jacobians, project_collision_tangent_objective,
            coll.jacobian, coll.lower_bounds, constraint_tolerance_,
            {}, 0.0, false,
            preserve_for_weighted ? &constrained_weighted_goals : nullptr,
            preserve_for_weighted ? &constrained_weighted_jacobians : nullptr);
    constraint_projection_applied =
        constraint_projection_applied || collision_projection_applied;
  }
  bool use_constrained_weighted_objectives =
      constraint_projection_applied && !constrained_weighted_goals.empty();

  // CoM support-polygon constraint
  std::optional<ComConstraintResult> com_constraint_result = std::nullopt;
  if (com_constraint_.has_value() && com_constraint_->enabled) {
    com_constraint_result = compute_com_constraint();
  }
  if (com_constraint_result.has_value() && !excluded_union.empty()) {
    zero_excluded_columns(com_constraint_result->jacobian, excluded_union);
  }
  if (use_contact_projection && com_constraint_result.has_value()) {
    com_constraint_result->jacobian =
        com_constraint_result->jacobian * contact_P_c;
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
    if (use_constrained_weighted_objectives) {
      project_task_jacobians_away_from_violated_rows(
          constrained_weighted_jacobians, com.jacobian, com.violated_rows,
          /*outward_is_positive_projection=*/true);
    }
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
  if (use_contact_projection && rel_pose_constraint_result.has_value()) {
    rel_pose_constraint_result->jacobian =
        rel_pose_constraint_result->jacobian * contact_P_c;
  }
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
    if (use_constrained_weighted_objectives) {
      project_task_jacobians_away_from_violated_rows(
          constrained_weighted_jacobians, rpc.jacobian, rpc.violated_rows,
          /*outward_is_positive_projection=*/true);
    }
  }

  std::optional<LinearVelocityConstraintResult> linear_constraint_result =
      compute_linear_velocity_constraints();
  if (linear_constraint_result.has_value() && !excluded_union.empty()) {
    for (int idx : excluded_union) {
      if (idx >= 0 && idx < linear_constraint_result->jacobian.cols()) {
        linear_constraint_result->jacobian.col(idx).setZero();
      }
    }
  }
  if (use_contact_projection && linear_constraint_result.has_value()) {
    linear_constraint_result->jacobian =
        linear_constraint_result->jacobian * contact_P_c;
  }

  std::optional<LinearVelocityConstraintResult> tight_pose_constraint_result =
      compute_tight_frame_pose_constraints();
  if (tight_pose_constraint_result.has_value() && !excluded_union.empty()) {
    for (int idx : excluded_union) {
      if (idx >= 0 && idx < tight_pose_constraint_result->jacobian.cols()) {
        tight_pose_constraint_result->jacobian.col(idx).setZero();
      }
    }
  }
  if (use_contact_projection && tight_pose_constraint_result.has_value()) {
    tight_pose_constraint_result->jacobian =
        tight_pose_constraint_result->jacobian * contact_P_c;
  }
  if (apply_limits && tight_pose_constraint_result.has_value()) {
    const auto &tpc = tight_pose_constraint_result.value();
    project_task_jacobians_away_from_violated_rows(
        jacobians, tpc.jacobian, tpc.violated_rows,
        /*outward_is_positive_projection=*/true);
    if (use_constrained_weighted_objectives) {
      project_task_jacobians_away_from_violated_rows(
          constrained_weighted_jacobians, tpc.jacobian, tpc.violated_rows,
          /*outward_is_positive_projection=*/true);
    }
  }

  std::optional<LinearVelocityConstraintResult> tight_point_constraint_result =
      compute_tight_point_constraints();
  if (tight_point_constraint_result.has_value() && !excluded_union.empty()) {
    for (int idx : excluded_union) {
      if (idx >= 0 && idx < tight_point_constraint_result->jacobian.cols()) {
        tight_point_constraint_result->jacobian.col(idx).setZero();
      }
    }
  }
  if (use_contact_projection && tight_point_constraint_result.has_value()) {
    tight_point_constraint_result->jacobian =
        tight_point_constraint_result->jacobian * contact_P_c;
  }
  if (apply_limits && tight_point_constraint_result.has_value()) {
    const auto &tpc = tight_point_constraint_result.value();
    project_task_jacobians_away_from_violated_rows(
        jacobians, tpc.jacobian, tpc.violated_rows,
        /*outward_is_positive_projection=*/true);
    if (use_constrained_weighted_objectives) {
      project_task_jacobians_away_from_violated_rows(
          constrained_weighted_jacobians, tpc.jacobian, tpc.violated_rows,
          /*outward_is_positive_projection=*/true);
    }
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
      if (use_contact_projection && step_torso_constraint_result.has_value()) {
        step_torso_constraint_result->jacobian =
            step_torso_constraint_result->jacobian * contact_P_c;
      }
    }
  }

  std::optional<ConstraintBlock> position_step_priority_constraint_result =
      std::nullopt;
  if (apply_limits && acceleration_limits_enabled_ &&
      acceleration_limits_.size() == robot_->nv() &&
      !pending_position_step_priority_constraints_.empty()) {
    struct PriorityViabilityRow {
      Eigen::RowVectorXd jacobian;
      double lower_bound;
    };
    std::vector<PriorityViabilityRow> rows;
    rows.reserve(2 * pending_position_step_priority_constraints_.size());

    const auto append_priority_block =
        [&](const Eigen::VectorXd &error, const Eigen::MatrixXd &jacobian,
            int offset, double tolerance, double step_dt) {
          if (!std::isfinite(tolerance) || tolerance <= 0.0 ||
              !std::isfinite(step_dt) || step_dt <= 0.0 ||
              offset < 0 || offset + 3 > error.size() ||
              offset + 3 > jacobian.rows()) {
            return;
          }
          const Eigen::Vector3d block_error = error.segment<3>(offset);
          const double error_norm = block_error.norm();
          if (!std::isfinite(error_norm) || error_norm <= 1e-12) {
            return;
          }

          Eigen::RowVectorXd row =
              (block_error / error_norm).transpose() *
              jacobian.middleRows(offset, 3);
          for (int velocity_index : pending_velocity_lock_indices_) {
            if (velocity_index >= 0 && velocity_index < row.size()) {
              row[velocity_index] = 0.0;
            }
          }
          if (use_contact_projection) {
            row *= contact_P_c;
          }
          if (!row.allFinite() || row.squaredNorm() <= 1e-24) {
            return;
          }

          double acceleration_support = 0.0;
          for (int velocity_index = 0; velocity_index < row.size();
               ++velocity_index) {
            const double acceleration_limit =
                acceleration_limits_[velocity_index];
            if (!std::isfinite(acceleration_limit) ||
                acceleration_limit <= 0.0) {
              continue;
            }
            acceleration_support +=
                std::abs(row[velocity_index]) * acceleration_limit;
          }
          if (!std::isfinite(acceleration_support) ||
              acceleration_support <= 0.0) {
            return;
          }

          const double viable_tolerance =
              std::max(0.0, tolerance - constraint_tolerance_);
          const double slack = viable_tolerance - error_norm;
          double lower_bound = 0.0;
          if (slack > 0.0) {
            lower_bound = -discrete_stopping_velocity_limit(
                slack, acceleration_support, step_dt);
          } else {
            lower_bound = (-slack) / step_dt;
          }
          rows.push_back({std::move(row), lower_bound});
        };

    for (const auto &spec : pending_position_step_priority_constraints_) {
      if (!spec.task || !spec.task->isActive()) {
        continue;
      }
      const Eigen::VectorXd error = spec.task->getError();
      const Eigen::MatrixXd jacobian = spec.task->getJacobian();
      if (jacobian.cols() != robot_->nv() || error.size() != jacobian.rows()) {
        continue;
      }
      const TaskType task_type = spec.task->getType();
      if (error.size() >= 6) {
        append_priority_block(error, jacobian, 0, spec.position_tolerance,
                              spec.step_dt);
        append_priority_block(error, jacobian, 3, spec.orientation_tolerance,
                              spec.step_dt);
      } else if (error.size() >= 3) {
        const bool orientation_only =
            task_type == TaskType::FRAME_ORIENTATION;
        append_priority_block(
            error, jacobian, 0,
            orientation_only ? spec.orientation_tolerance
                             : spec.position_tolerance,
            spec.step_dt);
      }
    }

    if (!rows.empty()) {
      ConstraintBlock block;
      block.jacobian =
          Eigen::MatrixXd::Zero(static_cast<int>(rows.size()), robot_->nv());
      block.lower_bounds = Eigen::VectorXd::Zero(static_cast<int>(rows.size()));
      block.upper_bounds = Eigen::VectorXd::Constant(
          static_cast<int>(rows.size()), kUnboundedConstraintLimit);
      for (int row_index = 0; row_index < static_cast<int>(rows.size());
           ++row_index) {
        block.jacobian.row(row_index) =
            rows[static_cast<std::size_t>(row_index)].jacobian;
        block.lower_bounds[row_index] =
            rows[static_cast<std::size_t>(row_index)].lower_bound;
      }
      position_step_priority_constraint_result = std::move(block);
    }
  }

  if (position_step_priority_constraint_result.has_value()) {
    int protected_priority = std::numeric_limits<int>::max();
    for (const auto &spec : pending_position_step_priority_constraints_) {
      if (spec.task && spec.task->isActive()) {
        protected_priority =
            std::min(protected_priority, spec.task->getPriority());
      }
    }
    std::vector<bool> project_priority_tangent_objective;
    project_priority_tangent_objective.reserve(objective_configs.size());
    for (const auto &objective : objective_configs) {
      project_priority_tangent_objective.push_back(
          objective.priority > protected_priority &&
          objective.solve_mode == TaskSolveMode::kScale);
    }
    const auto &priority_rows = *position_step_priority_constraint_result;
    const bool priority_projection_applied =
        project_objectives_into_constraint_tangent_space(
            goals, jacobians, project_priority_tangent_objective,
            priority_rows.jacobian, priority_rows.lower_bounds,
            constraint_tolerance_, {}, 0.0, true,
            preserve_for_weighted ? &constrained_weighted_goals : nullptr,
            preserve_for_weighted ? &constrained_weighted_jacobians : nullptr);
    constraint_projection_applied =
        constraint_projection_applied || priority_projection_applied;
    use_constrained_weighted_objectives =
        constraint_projection_applied && !constrained_weighted_goals.empty();
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

  // Sparse position-limit pre-pass: compute per-joint bounds and keep only
  // rows where position limits actually tighten beyond the velocity limit.
  // For joints deep in their range the bound equals [-vel, +vel] and the row
  // is redundant. Omitting it reduces the QP matrix significantly.
  //
  // Performance design: use dense arrays and cheap early-exit to avoid
  // calculate_velocity_box_constraint calls for far-from-limit joints.
  struct SparsePosLimitEntry { int nv_idx; double lower; double upper; };
  std::vector<SparsePosLimitEntry> sparse_pos_limits;
  // Dense per-joint bounds for post-QP clamping and saturation detection.
  // Initialized to "not active" sentinel; filled for active joints only.
  const int nv_sp = robot_->nv();
  constexpr double kNoPosBound = std::numeric_limits<double>::infinity();
  std::vector<double> dense_pos_lower_sp(nv_sp,  kNoPosBound);
  std::vector<double> dense_pos_upper_sp(nv_sp, -kNoPosBound);
  std::vector<double> merged_active_lower_sp(nv_sp, kNoPosBound);
  std::vector<double> merged_active_upper_sp(nv_sp, -kNoPosBound);
  std::vector<double> non_worsening_lower_sp(
      nv_sp, -kUnboundedConstraintLimit);
  std::vector<double> non_worsening_upper_sp(
      nv_sp, kUnboundedConstraintLimit);
  if (!use_contact_projection &&
      joint_limit_non_worsening_result.has_value()) {
    const Eigen::MatrixXd &rows =
        joint_limit_non_worsening_result->jacobian;
    for (int row = 0; row < rows.rows(); ++row) {
      for (int column = 0; column < rows.cols(); ++column) {
        const double coefficient = rows(row, column);
        if (coefficient > 0.5) {
          non_worsening_lower_sp[column] = std::max(
              non_worsening_lower_sp[column],
              joint_limit_non_worsening_result->lower_bounds[row] /
                  coefficient);
          break;
        }
        if (coefficient < -0.5) {
          non_worsening_upper_sp[column] = std::min(
              non_worsening_upper_sp[column],
              joint_limit_non_worsening_result->lower_bounds[row] /
                  coefficient);
          break;
        }
      }
    }
  }

  if (apply_limits && use_position_limits_) {
    // 0.1% of vel_limit — preserves borderline near-limit joints.
    constexpr double kPosBoundActiveFraction = 1e-3;
    constexpr double margin_limit_sp = kPositionLimitMarginEpsilon;
    auto [q_min_sp, q_max_sp] = robot_->get_joint_limits();
    Eigen::VectorXd q_cur_sp = robot_->get_current_configuration();
    const auto &vel_limits_sp = robot_->get_velocity_limits();
    const auto &accel_limits_sp = robot_->get_acceleration_limits();
    // Elastic band effective limits.
    Eigen::VectorXd q_min_eff_sp = q_min_sp;
    Eigen::VectorXd q_max_eff_sp = q_max_sp;
    const bool elastic_sp = elastic_band_config_.enabled &&
                            elastic_band_state_.delta.size() == robot_->nv();
    if (elastic_sp) {
      for (int i = 0; i < nv_sp; ++i) {
        const int qi = (i < static_cast<int>(velocity_to_config_index.size()))
                       ? velocity_to_config_index[i] : kVelocityToConfigUnmapped;
        if (qi == kVelocityToConfigUnmapped || qi >= q_min_sp.size()) continue;
        if (!std::isfinite(q_min_sp[qi]) || !std::isfinite(q_max_sp[qi])) continue;
        q_min_eff_sp[qi] -= elastic_band_state_.delta[i];
        q_max_eff_sp[qi] += elastic_band_state_.delta[i];
      }
    }
    // Dense locked-joint flag (avoid hash lookup per iteration).
    std::vector<bool> is_locked_sp(nv_sp, false);
    for (int idx : pending_velocity_lock_indices_) {
      if (idx >= 0 && idx < nv_sp) is_locked_sp[idx] = true;
    }
    const int fb_dof = robot_->is_floating_base() ? 6 : 0;
    const double dt_safe = std::max(dt_, 1e-6);

    for (int i = 0; i < nv_sp; ++i) {
      double vl = (i < static_cast<int>(vel_limits_sp.size()))
                  ? vel_limits_sp[i] : kUnboundedConstraintLimit;
      double al = (i < static_cast<int>(accel_limits_sp.size()))
                  ? accel_limits_sp[i] : kUnboundedConstraintLimit;
      bool is_locked = is_locked_sp[i];

      if (i < fb_dof) {
        // Floating base: only add when explicit base bounds are set.
        bool has_bound = false;
        double lm = 0.0, um = 0.0;
        if (i < 3 && base_position_lower_.has_value() && base_position_upper_.has_value()) {
          lm = q_cur_sp[i] - base_position_lower_.value()[i] - margin_limit_sp;
          um = base_position_upper_.value()[i] - q_cur_sp[i] - margin_limit_sp;
          has_bound = true;
        } else if (i >= 3 && i < 6 &&
                   base_orientation_lower_.has_value() && base_orientation_upper_.has_value()) {
          lm = q_cur_sp[i] - base_orientation_lower_.value()[i-3] - margin_limit_sp;
          um = base_orientation_upper_.value()[i-3] - q_cur_sp[i] - margin_limit_sp;
          has_bound = true;
        }
        if (!has_bound && !is_locked) continue;
        double lo = -vl, hi = vl;
        if (has_bound) {
          auto [ll, ul] = calculate_velocity_box_constraint(lm, um, vl, al, dt_);
          lo = ll; hi = ul;
        }
        if (is_locked) { lo = 0.0; hi = 0.0; }
        const double tol = kPosBoundActiveFraction * vl;
        if ((lo > -vl + tol) || (hi < vl - tol) || is_locked) {
          sparse_pos_limits.push_back({i, lo, hi});
          dense_pos_lower_sp[i] = lo; dense_pos_upper_sp[i] = hi;
        }
      } else {
        // Regular joint: cheap early-exit before calling calculate_velocity_box_constraint.
        const int qi = (i < static_cast<int>(velocity_to_config_index.size()))
                       ? velocity_to_config_index[i] : kVelocityToConfigUnmapped;
        if (qi == kVelocityToConfigUnmapped || qi >= q_cur_sp.size() ||
            qi >= q_min_eff_sp.size() || !std::isfinite(q_min_eff_sp[qi]) ||
            !std::isfinite(q_max_eff_sp[qi])) {
          if (is_locked) {
            sparse_pos_limits.push_back({i, 0.0, 0.0});
            dense_pos_lower_sp[i] = 0.0; dense_pos_upper_sp[i] = 0.0;
          }
          continue;
        }
        const double lm = q_cur_sp[qi] - q_min_eff_sp[qi] - margin_limit_sp;
        const double um = q_max_eff_sp[qi] - q_cur_sp[qi] - margin_limit_sp;
        // Cheap check: if both margins exceed max possible displacement in one step,
        // bounds = [-vel_limit, vel_limit] → skip (no tightening needed).
        const bool enforce_non_worsening =
            non_worsening_lower_sp[i] > -kUnboundedConstraintLimit ||
            non_worsening_upper_sp[i] < kUnboundedConstraintLimit;
        if (!is_locked && !enforce_non_worsening) {
          const double pos_room_l = lm / dt_safe;
          const double pos_room_u = um / dt_safe;
          const double accel_room_l = (al > 0.0 && std::isfinite(al))
              ? std::sqrt(2.0 * al * std::max(0.0, lm)) : vl;
          const double accel_room_u = (al > 0.0 && std::isfinite(al))
              ? std::sqrt(2.0 * al * std::max(0.0, um)) : vl;
          if (pos_room_l >= vl && accel_room_l >= vl &&
              pos_room_u >= vl && accel_room_u >= vl) {
            continue;  // definitely [-vel_limit, vel_limit] — skip row
          }
        }
        auto [lo, hi] = calculate_velocity_box_constraint(lm, um, vl, al, dt_);
        lo = std::max(lo, non_worsening_lower_sp[i]);
        hi = std::min(hi, non_worsening_upper_sp[i]);
        if (is_locked) { lo = 0.0; hi = 0.0; }
        if (enforce_non_worsening) {
          sparse_pos_limits.push_back({i, lo, hi});
          merged_active_lower_sp[i] = lo;
          merged_active_upper_sp[i] = hi;
          dense_pos_lower_sp[i] = lo;
          dense_pos_upper_sp[i] = hi;
          continue;
        }
        const double tol = kPosBoundActiveFraction * vl;
        if ((lo > -vl + tol) || (hi < vl - tol) || is_locked) {
          sparse_pos_limits.push_back({i, lo, hi});
          dense_pos_lower_sp[i] = lo; dense_pos_upper_sp[i] = hi;
        }
      }
    }
    num_constraints += static_cast<int>(sparse_pos_limits.size());
  }

  if (use_contact_projection &&
      joint_limit_non_worsening_result.has_value()) {
    num_constraints += static_cast<int>(
        joint_limit_non_worsening_result->jacobian.rows());
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
  if (position_step_priority_constraint_result.has_value()) {
    num_constraints += static_cast<int>(
        position_step_priority_constraint_result->jacobian.rows());
  }
  if (linear_constraint_result.has_value()) {
    num_constraints +=
        static_cast<int>(linear_constraint_result->jacobian.rows());
  }
  if (tight_pose_constraint_result.has_value()) {
    num_constraints +=
        static_cast<int>(tight_pose_constraint_result->jacobian.rows());
  }
  if (tight_point_constraint_result.has_value()) {
    num_constraints +=
        static_cast<int>(tight_point_constraint_result->jacobian.rows());
  }

  C = Eigen::MatrixXd::Zero(num_constraints, robot_->nv());
  c_lower = Eigen::VectorXd::Zero(num_constraints);
  c_upper = Eigen::VectorXd::Zero(num_constraints);
  Eigen::VectorXd max_softening_factors =
      Eigen::VectorXd::Ones(num_constraints);

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
  // Per-joint velocity-limit overrides (contribution knob): shrink the box so the
  // solver recruits other DOFs when a group is throttled. Applied in both
  // branches; hard velocity locks below still take priority.
  for (const auto &kv : joint_velocity_limit_overrides_) {
    const int idx = kv.first;
    if (idx >= 0 && idx < robot_->nv()) {
      c_lower(constraint_idx + idx) = -kv.second;
      c_upper(constraint_idx + idx) = kv.second;
    }
  }
  for (int idx = 0; idx < robot_->nv(); ++idx) {
    if (std::isfinite(merged_active_lower_sp[idx])) {
      c_lower(constraint_idx + idx) =
          std::max(c_lower(constraint_idx + idx),
                   merged_active_lower_sp[idx]);
    }
    if (std::isfinite(merged_active_upper_sp[idx])) {
      c_upper(constraint_idx + idx) =
          std::min(c_upper(constraint_idx + idx),
                   merged_active_upper_sp[idx]);
    }
  }
  for (int idx : pending_velocity_lock_indices_) {
    if (idx >= 0 && idx < robot_->nv()) {
      c_lower(constraint_idx + idx) = 0.0;
      c_upper(constraint_idx + idx) = 0.0;
    }
  }
  constraint_idx += robot_->nv();

  // Position-based velocity constraints (sparse: only joints near their limits)
  if (apply_limits && use_position_limits_) {
    // sparse_pos_limits was pre-computed above; fill only those rows.
    for (int k = 0; k < static_cast<int>(sparse_pos_limits.size()); ++k) {
      const auto &e = sparse_pos_limits[k];
      C(constraint_idx + k, e.nv_idx) = 1.0;
      c_lower(constraint_idx + k) = e.lower;
      c_upper(constraint_idx + k) = e.upper;
    }
    constraint_idx += static_cast<int>(sparse_pos_limits.size());
    // Skip the old dense fill block below (it is now a no-op guarded by false).
    if (false) {
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
    }  // end if (false) — old dense position-limit block (disabled)
  }

  if (use_contact_projection &&
      joint_limit_non_worsening_result.has_value()) {
    constraint_idx = append_constraint_block(
        C, c_lower, c_upper, constraint_idx, robot_->nv(),
        joint_limit_non_worsening_result->jacobian,
        joint_limit_non_worsening_result->lower_bounds,
        joint_limit_non_worsening_result->upper_bounds);
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
  if (position_step_priority_constraint_result.has_value()) {
    constraint_idx = append_constraint_block(
        C, c_lower, c_upper, constraint_idx, robot_->nv(),
        position_step_priority_constraint_result->jacobian,
        position_step_priority_constraint_result->lower_bounds,
        position_step_priority_constraint_result->upper_bounds);
  }
  if (linear_constraint_result.has_value()) {
    const auto &lc = linear_constraint_result.value();
    int rows = static_cast<int>(lc.jacobian.rows());
    C.block(constraint_idx, 0, rows, robot_->nv()) = lc.jacobian;
    c_lower.segment(constraint_idx, rows) = lc.lower_bounds;
    c_upper.segment(constraint_idx, rows) = lc.upper_bounds;
    constraint_idx += rows;
  }
  if (tight_pose_constraint_result.has_value()) {
    const auto &tpc = tight_pose_constraint_result.value();
    int rows = static_cast<int>(tpc.jacobian.rows());
    C.block(constraint_idx, 0, rows, robot_->nv()) = tpc.jacobian;
    c_lower.segment(constraint_idx, rows) = tpc.lower_bounds;
    c_upper.segment(constraint_idx, rows) = tpc.upper_bounds;
    max_softening_factors.segment(constraint_idx, rows).setConstant(4.0);
    constraint_idx += rows;
  }
  if (tight_point_constraint_result.has_value()) {
    const auto &tpc = tight_point_constraint_result.value();
    int rows = static_cast<int>(tpc.jacobian.rows());
    C.block(constraint_idx, 0, rows, robot_->nv()) = tpc.jacobian;
    c_lower.segment(constraint_idx, rows) = tpc.lower_bounds;
    c_upper.segment(constraint_idx, rows) = tpc.upper_bounds;
    max_softening_factors.segment(constraint_idx, rows).setConstant(4.0);
    constraint_idx += rows;
  }

  if (timing && t_constraint_start.has_value()) {
    result.constraint_setup_time_ms = get_elapsed_ms(*t_constraint_start);
  }

  // Acceleration-level constraint cascading: tighten the first nv rows
  // of c_lower/c_upper (the velocity limit block) based on previous tick's
  // velocity and the configured acceleration limits.
  // A deadband of 1% of the acceleration step prevents oscillation when
  // the velocity is small and the acceleration limit is tight.
  const bool acceleration_corridor_active =
      acceleration_limits_enabled_ &&
      (position_step_call_depth_ == 0 ||
       pending_position_step_acceleration_limits_) &&
      previous_dq_.size() == robot_->nv() &&
      acceleration_limits_.size() == robot_->nv();
  const Eigen::VectorXd velocity_lower_before_acceleration =
      c_lower.head(robot_->nv());
  const Eigen::VectorXd velocity_upper_before_acceleration =
      c_upper.head(robot_->nv());
  if (acceleration_corridor_active) {
    const int nv = robot_->nv();
    for (int i = 0; i < nv; ++i) {
      const double accel_step = acceleration_limits_[i] * dt_;
      const double deadband = accel_step * 0.01;
      double a_lb = previous_dq_[i] - accel_step - deadband;
      double a_ub = previous_dq_[i] + accel_step + deadband;
      c_lower[i] = std::max(c_lower[i], a_lb);
      c_upper[i] = std::min(c_upper[i], a_ub);
    }
  }

  // The active-limit viability interval is a hard safety invariant. If an
  // externally synchronized state leaves stale acceleration history outside
  // that interval, select the nearest safe boundary instead of letting the
  // generic interval sanitizer move the command back toward the limit.
  if (acceleration_corridor_active && !use_contact_projection &&
      joint_limit_non_worsening_result.has_value() &&
      previous_dq_.size() == robot_->nv()) {
    for (int i = 0; i < robot_->nv(); ++i) {
      if (!std::isfinite(merged_active_lower_sp[i]) ||
          !std::isfinite(merged_active_upper_sp[i]) ||
          c_lower[i] <= c_upper[i]) {
        continue;
      }
      const double safe_velocity = std::clamp(
          previous_dq_[i], merged_active_lower_sp[i],
          merged_active_upper_sp[i]);
      c_lower[i] = safe_velocity;
      c_upper[i] = safe_velocity;
    }
  }

  // Contact projection turns a scalar joint-limit row into a coupled row. If
  // stale acceleration history makes that row unreachable, restore the
  // pre-acceleration velocity interval only for its participating joints. The
  // projected contact row and active limit remain hard in the backend solve.
  if (acceleration_corridor_active && use_contact_projection &&
      joint_limit_non_worsening_result.has_value()) {
    const auto &active_limits = *joint_limit_non_worsening_result;
    for (int row = 0; row < active_limits.jacobian.rows(); ++row) {
      double maximum_value = 0.0;
      for (int column = 0; column < active_limits.jacobian.cols(); ++column) {
        const double coefficient = active_limits.jacobian(row, column);
        maximum_value += coefficient >= 0.0
                             ? coefficient * c_upper[column]
                             : coefficient * c_lower[column];
      }
      if (maximum_value >=
          active_limits.lower_bounds[row] - constraint_tolerance_) {
        continue;
      }
      for (int column = 0; column < active_limits.jacobian.cols(); ++column) {
        if (std::abs(active_limits.jacobian(row, column)) <= 1e-12) {
          continue;
        }
        c_lower[column] = velocity_lower_before_acceleration[column];
        c_upper[column] = velocity_upper_before_acceleration[column];
      }
    }
  }

  sanitize_solver_inputs(goals, jacobians, C, c_lower, c_upper);
  const bool has_soft_rows = max_softening_factors.maxCoeff() > 1.0;
  if (has_soft_rows) {
    const int nv = robot_->nv();
    Eigen::VectorXd dq_lower_seed = Eigen::VectorXd::Constant(nv, -1e10);
    Eigen::VectorXd dq_upper_seed = Eigen::VectorXd::Constant(nv, 1e10);
    for (int i = 0; i < nv && i < c_lower.size() && i < c_upper.size(); ++i) {
      dq_lower_seed(i) = c_lower(i);
      dq_upper_seed(i) = c_upper(i);
    }
    if (!has_rowwise_interval_feasibility(C, c_lower, c_upper, dq_lower_seed,
                                          dq_upper_seed,
                                          constraint_tolerance_)) {
      const double pre_relax_scale =
          std::max(1.0, max_softening_factors.maxCoeff());
      apply_constraint_softening_in_place(c_lower, c_upper,
                                          max_softening_factors,
                                          pre_relax_scale);
    }
  }

  // Configure solver
  VelocitySolverConfig config;
  config.epsilon = constraint_tolerance_;
  config.precision_threshold = tight_tolerance_;
  config.iteration_limit = max_iterations_;
  config.magnitude_limit = norm_threshold_;
  config.stall_detection_count = max_zero_scale_iterations_;
  if (has_soft_rows) {
    config.stall_detection_count = std::max(5, max_zero_scale_iterations_);
  }
  config.regularization_config.epsilon = solver_tolerance_;
  config.regularization_config.regularization_factor = damping_;

  WeightedAdvisoryResult advisory;
  const auto &advisor_goals = use_constrained_weighted_objectives
                                  ? constrained_weighted_goals
                                  : goals;
  const auto &advisor_jacobians = use_constrained_weighted_objectives
                                      ? constrained_weighted_jacobians
                                      : jacobians;
  auto compute_weighted_advisory = [&]() -> WeightedAdvisoryResult {
    auto weight_at_priority = [&](int priority) -> double {
      double sum = 0.0;
      int count = 0;
      for (const auto &task : tasks_) {
        if (!task || !task->isActive() || task->getPriority() != priority) {
          continue;
        }
        if (std::isfinite(task->getWeight()) && task->getWeight() >= 0.0) {
          sum += task->getWeight();
          ++count;
        }
      }
      return count > 0 ? sum / static_cast<double>(count) : 1.0;
    };
    auto type_at_priority = [&](int priority) -> TaskType {
      bool saw_position = false;
      bool saw_orientation = false;
      bool saw_pose = false;
      for (const auto &task : tasks_) {
        if (!task || !task->isActive() || task->getPriority() != priority) {
          continue;
        }
        const TaskType type = task->getType();
        saw_position = saw_position || type == TaskType::FRAME_POSITION;
        saw_orientation = saw_orientation || type == TaskType::FRAME_ORIENTATION;
        saw_pose = saw_pose || type == TaskType::FRAME_POSE;
      }
      if (saw_pose) {
        return TaskType::FRAME_POSE;
      }
      if (saw_position && !saw_orientation) {
        return TaskType::FRAME_POSITION;
      }
      if (saw_orientation && !saw_position) {
        return TaskType::FRAME_ORIENTATION;
      }
      return TaskType::POSTURE;
    };

    if (!runtime_config_.enable_advisor_scale_adapt) {
      advisor_scale_current_ =
          sanitize_advisor_scale(runtime_config_.advisor_position_weight_scale);
    }
    double position_weight_scale =
        runtime_config_.enable_advisor_scale_adapt
            ? advisor_scale_current_
            : sanitize_advisor_scale(
                  runtime_config_.advisor_position_weight_scale);
    const double advisor_scale_min =
        std::isfinite(runtime_config_.advisor_scale_min)
            ? runtime_config_.advisor_scale_min
            : 0.25;
    const double advisor_scale_max =
        std::isfinite(runtime_config_.advisor_scale_max)
            ? runtime_config_.advisor_scale_max
            : 8.0;
    if (runtime_config_.enable_advisor_scale_adapt) {
      const double lo =
          std::max(0.0, std::min(advisor_scale_min, advisor_scale_max));
      const double hi =
          std::max(lo, std::max(advisor_scale_min, advisor_scale_max));
      position_weight_scale = std::clamp(position_weight_scale, lo, hi);
      advisor_scale_current_ = position_weight_scale;
    }
    double orientation_weight_scale =
        runtime_config_.advisor_orientation_weight_scale;
    if (!std::isfinite(orientation_weight_scale) ||
        orientation_weight_scale < 0.0) {
      orientation_weight_scale = 1.0;
    }

    auto sanitized_task_weight = [](const std::shared_ptr<Task> &task) {
      if (!task || !std::isfinite(task->getWeight()) ||
          task->getWeight() < 0.0) {
        return 1.0;
      }
      return task->getWeight();
    };
    auto make_task_row_weights = [&](const std::shared_ptr<Task> &task) {
      Eigen::VectorXd row_weights;
      if (!task) {
        return row_weights;
      }
      const int dimension = task->getDimension();
      if (dimension <= 0) {
        return row_weights;
      }
      row_weights =
          Eigen::VectorXd::Constant(dimension, sanitized_task_weight(task));
      const TaskType type = task->getType();
      if (type == TaskType::FRAME_POSITION) {
        row_weights.array() *= position_weight_scale;
      } else if (type == TaskType::FRAME_ORIENTATION) {
        row_weights.array() *= orientation_weight_scale;
      } else if (type == TaskType::FRAME_POSE && dimension == 6) {
        row_weights.head<3>().array() *= position_weight_scale;
        row_weights.tail<3>().array() *= orientation_weight_scale;
      }
      return row_weights;
    };
    auto row_weights_at_priority = [&](int priority) {
      int total_rows = 0;
      for (const auto &task : tasks_) {
        if (task && task->isActive() && task->getPriority() == priority) {
          total_rows += task->getDimension();
        }
      }
      Eigen::VectorXd row_weights;
      if (total_rows <= 0) {
        return row_weights;
      }
      row_weights.resize(total_rows);
      int offset = 0;
      for (const auto &task : tasks_) {
        if (!task || !task->isActive() || task->getPriority() != priority) {
          continue;
        }
        const Eigen::VectorXd task_weights = make_task_row_weights(task);
        if (task_weights.size() == 0) {
          continue;
        }
        row_weights.segment(offset, task_weights.size()) = task_weights;
        offset += static_cast<int>(task_weights.size());
      }
      if (offset != total_rows) {
        row_weights.conservativeResize(offset);
      }
      return row_weights;
    };

    std::vector<double> advisor_weights;
    std::vector<Eigen::VectorXd> advisor_row_weights;
    advisor_weights.reserve(advisor_goals.size());
    advisor_row_weights.resize(advisor_goals.size());
    for (size_t i = 0; i < objective_configs.size(); ++i) {
      double w = 1.0;
      TaskType type = TaskType::POSTURE;
      if (i < objective_tasks.size() && objective_tasks[i]) {
        w = objective_tasks[i]->getWeight();
        type = objective_tasks[i]->getType();
        advisor_row_weights[i] = make_task_row_weights(objective_tasks[i]);
      } else {
        w = weight_at_priority(objective_configs[i].priority);
        type = type_at_priority(objective_configs[i].priority);
        advisor_row_weights[i] =
            row_weights_at_priority(objective_configs[i].priority);
      }
      if (!std::isfinite(w) || w < 0.0) {
        w = 1.0;
      }
      if (type == TaskType::FRAME_POSITION) {
        w *= position_weight_scale;
      } else if (type == TaskType::FRAME_ORIENTATION) {
        w *= orientation_weight_scale;
      }
      advisor_weights.push_back(w);
    }

    WeightedAdvisoryResult out = compute_constrained_weighted_advisory(
        advisor_goals, advisor_jacobians, C, c_lower, c_upper, advisor_weights,
        config, advisor_row_weights);
    if (!out.available) {
      return out;
    }

    result.weighted_advisory_available = true;
    result.weighted_advisory_v_norm = out.v_norm;
    result.weighted_advisory_condition_number = out.condition_number;

    double position_error = std::numeric_limits<double>::quiet_NaN();
    double orientation_error = std::numeric_limits<double>::quiet_NaN();
    for (size_t i = 0; i < objective_configs.size() &&
                       i < out.per_objective_error.size();
         ++i) {
      TaskType type = TaskType::POSTURE;
      if (i < objective_tasks.size() && objective_tasks[i]) {
        type = objective_tasks[i]->getType();
      } else {
        type = type_at_priority(objective_configs[i].priority);
      }
      if (type == TaskType::FRAME_POSITION &&
          !std::isfinite(position_error)) {
        position_error = out.per_objective_error[i];
      } else if (type == TaskType::FRAME_ORIENTATION &&
                 !std::isfinite(orientation_error)) {
        orientation_error = out.per_objective_error[i];
      } else if (type == TaskType::FRAME_POSE) {
        const auto &J = advisor_jacobians[i];
        const auto &b = advisor_goals[i];
        if (J.rows() >= 6 && b.rows() >= 6 && out.v.size() == J.cols()) {
          const Eigen::VectorXd residual = J * out.v - b;
          if (i < objective_tasks.size() && objective_tasks[i]) {
            if (!std::isfinite(position_error)) {
              position_error = residual.head(3).norm();
            }
            if (!std::isfinite(orientation_error)) {
              orientation_error = residual.tail(3).norm();
            }
          } else {
            Eigen::Index offset = 0;
            for (const auto &task : tasks_) {
              if (!task || !task->isActive() ||
                  task->getPriority() != objective_configs[i].priority) {
                continue;
              }
              const Eigen::Index dim = task->getDimension();
              if (offset + dim > residual.size()) {
                break;
              }
              if (task->getType() == TaskType::FRAME_POSITION &&
                  !std::isfinite(position_error)) {
                position_error = residual.segment(offset, dim).norm();
              } else if (task->getType() == TaskType::FRAME_ORIENTATION &&
                         !std::isfinite(orientation_error)) {
                orientation_error = residual.segment(offset, dim).norm();
              } else if (task->getType() == TaskType::FRAME_POSE && dim >= 6) {
                if (!std::isfinite(position_error)) {
                  position_error = residual.segment(offset, 3).norm();
                }
                if (!std::isfinite(orientation_error)) {
                  orientation_error = residual.segment(offset + dim - 3, 3).norm();
                }
              }
              offset += dim;
            }
          }
        }
      }
    }
    result.weighted_advisory_pos_task_error_norm = position_error;
    result.weighted_advisory_ori_task_error_norm = orientation_error;
    result.advisor_position_weight_scale_current = position_weight_scale;
    result.advisor_scale_adapt_active = false;
    if (runtime_config_.enable_advisor_scale_adapt &&
        std::isfinite(position_error) && std::isfinite(orientation_error)) {
      const double denom = position_error + orientation_error;
      if (denom > 1e-12) {
        advisor_scale_ratio_sum_ += position_error / denom;
        advisor_scale_epoch_time_s_ += std::max(dt_, 1e-9);
        ++advisor_scale_sample_count_;
      }
      const double epoch_s =
          (std::isfinite(runtime_config_.advisor_scale_epoch_s) &&
           runtime_config_.advisor_scale_epoch_s > 0.0)
              ? runtime_config_.advisor_scale_epoch_s
              : 0.1;
      if (advisor_scale_epoch_time_s_ >= epoch_s &&
          advisor_scale_sample_count_ > 0) {
        const double avg_ratio =
            advisor_scale_ratio_sum_ /
            static_cast<double>(advisor_scale_sample_count_);
        const double target_ratio = std::clamp(
            runtime_config_.advisor_scale_adapt_target_ratio, 0.0, 1.0);
        const double ki =
            std::isfinite(runtime_config_.advisor_scale_adapt_ki)
                ? runtime_config_.advisor_scale_adapt_ki
                : 0.0;
        const double lo =
            std::max(0.0, std::min(advisor_scale_min, advisor_scale_max));
        const double hi =
            std::max(lo, std::max(advisor_scale_min, advisor_scale_max));
        const double exponent =
            std::clamp(ki * (avg_ratio - target_ratio), -1.0, 1.0);
        advisor_scale_current_ =
            std::clamp(advisor_scale_current_ * std::exp(exponent), lo, hi);
        advisor_scale_ratio_sum_ = 0.0;
        advisor_scale_epoch_time_s_ = 0.0;
        advisor_scale_sample_count_ = 0;
        result.advisor_position_weight_scale_current = advisor_scale_current_;
        result.advisor_scale_adapt_active = true;
      }
    }
    return out;
  };

  if (runtime_config_.weighted_advisor_enabled && !goals.empty()) {
    advisory = compute_weighted_advisory();
  }

  warm_start_selector_cache_.reset();
  warm_start_constraint_rows_ = -1;
  // Soft joint-space metric (contribution knob): change of variables dq = D u with
  // D = diag(1/sqrt(w)). Solve the existing prioritized/SNS problem in u-space on
  // scaled copies (J D, C D), then un-scale the solution dq = D u. This yields the
  // weighted least-norm solution while keeping the EE task and the true-space
  // velocity/position limits intact (bounds are unchanged; columns carry D).
  const bool metric_active =
      joint_metric_col_scale_.size() == robot_->nv();
  std::vector<Eigen::MatrixXd> metric_jacobians;
  Eigen::MatrixXd metric_C;
  const std::vector<Eigen::MatrixXd> *solve_jacobians = &jacobians;
  const Eigen::MatrixXd *solve_C = &C;
  if (metric_active) {
    const auto D = joint_metric_col_scale_.asDiagonal();
    metric_jacobians.reserve(jacobians.size());
    for (const auto &J : jacobians) metric_jacobians.push_back(J * D);
    metric_C = C * D;
    solve_jacobians = &metric_jacobians;
    solve_C = &metric_C;
  }
  auto backend_result = computeMultiObjectiveVelocitySolutionEigen(
      goals, *solve_jacobians, *solve_C, c_lower, c_upper, config,
      objective_configs, has_soft_rows ? &max_softening_factors : nullptr);
  if (metric_active &&
      static_cast<int>(backend_result.solution.size()) == robot_->nv()) {
    for (int j = 0; j < robot_->nv(); ++j)
      backend_result.solution[j] *= joint_metric_col_scale_(j);
  }
  bool backend_solution_non_finite = false;
  for (double v : backend_result.solution) {
    if (!std::isfinite(v)) {
      backend_solution_non_finite = true;
      break;
    }
  }
  const bool backend_non_finite_input =
      backend_result.status == SolverStatus::kNonFiniteInput ||
      backend_solution_non_finite;
  const bool backend_generated_non_finite =
      backend_result.status == SolverStatus::kNumericalError &&
      backend_result.status_message ==
          "non-finite values generated in solver state";
  if (backend_non_finite_input) {
    backend_result.status = SolverStatus::kNonFiniteInput;
    backend_result.status_message =
        backend_solution_non_finite
            ? "backend returned non-finite velocity entries; using zero velocity step"
            : "backend produced non-finite internal state; using zero velocity step";
    backend_result.solution.assign(static_cast<size_t>(robot_->nv()), 0.0);
    backend_result.final_error = 0.0;
  } else if (backend_generated_non_finite) {
    backend_result.status = SolverStatus::kNoProgress;
    backend_result.status_message =
        "backend generated non-finite internal state; using zero velocity step";
    backend_result.solution.assign(static_cast<size_t>(robot_->nv()), 0.0);
    backend_result.final_error = 0.0;
  }
  if (use_contact_projection &&
      static_cast<int>(backend_result.solution.size()) == robot_->nv()) {
    Eigen::Map<Eigen::VectorXd> dq_projected(backend_result.solution.data(),
                                             backend_result.solution.size());
    dq_projected = contact_P_c * dq_projected;
  }
  const double primary_goal_norm = goals.empty() ? 0.0 : goals[0].norm();

  auto classified_velocity = classify_velocity_outcome(
      backend_result.status, backend_result.status_message,
      backend_result.task_scales, primary_goal_norm,
      backend_result.task_modes_effective, backend_result.task_used_fallback);
  auto clamp_final_velocity_candidate = [&](Eigen::VectorXd &candidate) {
    if (apply_limits && c_lower.size() >= robot_->nv() &&
        !use_contact_projection) {
      clamp_joint_velocity_solution_in_place(
          candidate, c_lower, c_upper, use_position_limits_, robot_->nv());
      if (use_position_limits_) {
        const int n_dq = static_cast<int>(candidate.size());
        const int n_dens = static_cast<int>(dense_pos_lower_sp.size());
        const int n_clamp = std::min(n_dq, n_dens);
        for (int k = 0; k < n_clamp; ++k) {
          if (dense_pos_lower_sp[k] < kNoPosBound)
            candidate[k] = std::max(candidate[k], dense_pos_lower_sp[k]);
          if (dense_pos_upper_sp[k] > -kNoPosBound)
            candidate[k] = std::min(candidate[k], dense_pos_upper_sp[k]);
        }
      }
    }
  };
  auto contact_velocity_norm = [&](const Eigen::VectorXd &candidate) {
    double max_norm = 0.0;
    for (const auto &cfg : contact_frames_) {
      const Matrix6Xd J_full = robot_->get_frame_jacobian(cfg.frame_name);
      Eigen::VectorXd v;
      if (cfg.type == ContactType::kPointContact) {
        v = J_full.topRows(3) * candidate;
      } else {
        v = J_full * candidate;
      }
      max_norm = std::max(max_norm, v.norm());
    }
    return max_norm;
  };
  auto com_max_violation = [&]() {
    if (!com_constraint_.has_value() || !com_constraint_->enabled) {
      return 0.0;
    }
    const auto &cfg = *com_constraint_;
    const Eigen::Vector3d com_world = robot_->get_com_position();
    Eigen::MatrixXd A_world = cfg.A;
    Eigen::VectorXd b_world = cfg.b;
    if (cfg.frame_name != "world") {
      const auto frame_pose = robot_->get_frame_pose(cfg.frame_name);
      const Eigen::Matrix3d R = frame_pose.rotation();
      const Eigen::Vector3d t = frame_pose.translation();
      const Eigen::Matrix2d R_xy = R.topLeftCorner<2, 2>();
      A_world = cfg.A * R_xy.transpose();
      b_world = cfg.b;
      for (int i = 0; i < static_cast<int>(b_world.size()); ++i) {
        b_world(i) += A_world.row(i).dot(t.head<2>());
      }
    }
    const Eigen::VectorXd slack = b_world - A_world * com_world.head<2>();
    double violation = 0.0;
    for (int i = 0; i < slack.size(); ++i) {
      violation = std::max(violation, -slack(i));
    }
    return violation;
  };
  auto relative_pose_max_violation = [&]() {
    if (!relative_pose_constraint_.has_value() ||
        !relative_pose_constraint_->enabled) {
      return 0.0;
    }
    const auto &cfg = *relative_pose_constraint_;
    const pinocchio::SE3 T_a = robot_->get_frame_pose(cfg.frame_a);
    const pinocchio::SE3 T_b = robot_->get_frame_pose(cfg.frame_b);
    const pinocchio::SE3 T_rel = compute_relative_frame(T_a, T_b);
    Eigen::VectorXd rel_state = Eigen::VectorXd::Zero(6);
    rel_state.head<3>() = T_rel.translation();
    rel_state.tail<3>() = pinocchio::log3(T_rel.rotation());
    double violation = 0.0;
    for (int i = 0; i < 6; ++i) {
      if (cfg.axis_mask(i) <= 0.5) {
        continue;
      }
      violation = std::max(violation, cfg.lower_bounds(i) - rel_state(i));
      violation = std::max(violation, rel_state(i) - cfg.upper_bounds(i));
    }
    return violation;
  };
  const double fallback_validation_dt =
      (pending_step_validation_dt_.has_value() &&
       std::isfinite(*pending_step_validation_dt_) &&
       *pending_step_validation_dt_ > 0.0)
          ? *pending_step_validation_dt_
          : dt_;
  auto weighted_fallback_candidate_acceptable =
      [&](const Eigen::VectorXd &candidate) {
        if (!candidate.allFinite() || candidate.size() != robot_->nv()) {
          return false;
        }
        if (C.rows() > 0) {
          const Eigen::VectorXd values = C * candidate;
          const double feasibility_tol =
              std::max(1e-8, 10.0 * constraint_tolerance_);
          for (int i = 0; i < values.size(); ++i) {
            if (values(i) < c_lower(i) - feasibility_tol ||
                values(i) > c_upper(i) + feasibility_tol) {
              return false;
            }
          }
        }
        if (use_contact_projection &&
            contact_velocity_norm(candidate) >
                std::max(1e-8, 10.0 * constraint_tolerance_)) {
          return false;
        }

        const bool need_post_step_validation =
            (collision_constraint_.has_value() &&
             collision_constraint_->enabled) ||
            (com_constraint_.has_value() && com_constraint_->enabled) ||
            (relative_pose_constraint_.has_value() &&
             relative_pose_constraint_->enabled);
        if (!need_post_step_validation) {
          return true;
        }

        const double current_com_violation = com_max_violation();
        const double current_rel_violation = relative_pose_max_violation();
        const std::vector<double> current_collision_margins =
            (collision_constraint_.has_value() &&
             collision_constraint_->enabled)
                ? evaluate_post_step_collision_recovery_margins(q_eval)
                : std::vector<double>{};

        const Eigen::VectorXd q_candidate = pinocchio::integrate(
            robot_->model(), q_eval, fallback_validation_dt * candidate);
        robot_->update_kinematics(q_candidate);
        const double candidate_com_violation = com_max_violation();
        const double candidate_rel_violation = relative_pose_max_violation();
        const std::vector<double> candidate_collision_margins =
            (collision_constraint_.has_value() &&
             collision_constraint_->enabled)
                ? evaluate_post_step_collision_recovery_margins(q_candidate)
                : std::vector<double>{};
        robot_->update_kinematics(q_eval);

        constexpr double kViolationImproveTolerance = 1e-9;
        if (candidate_com_violation >
            current_com_violation + kViolationImproveTolerance) {
          return false;
        }
        if (candidate_rel_violation >
            current_rel_violation + kViolationImproveTolerance) {
          return false;
        }
        if (collision_constraint_.has_value() &&
            collision_constraint_->enabled) {
          if (!collision_recovery_margins_acceptable(
                  current_collision_margins, candidate_collision_margins,
                  0.0)) {
            return false;
          }
        }
        return true;
      };
  const bool can_try_weighted_fallback =
      runtime_config_.weighted_fallback_enabled &&
      classified_velocity.status != SolverStatus::kSuccess &&
      classified_velocity.status != SolverStatus::kNonFiniteInput &&
      !goals.empty();
  if (can_try_weighted_fallback && !advisory.available) {
    advisory = compute_weighted_advisory();
  }
  if (can_try_weighted_fallback && advisory.available) {
    Eigen::VectorXd accepted_candidate = advisory.v;
    if (use_contact_projection) {
      accepted_candidate = contact_P_c * accepted_candidate;
    }
    clamp_final_velocity_candidate(accepted_candidate);
    const bool accept_weighted_fallback =
        weighted_fallback_candidate_acceptable(accepted_candidate);
    if (accept_weighted_fallback) {
      backend_result.solution.assign(
          accepted_candidate.data(),
          accepted_candidate.data() + accepted_candidate.size());
      backend_result.status = SolverStatus::kSuccess;
      backend_result.status_message =
          "constrained weighted fallback accepted after prioritized solver: " +
          classified_velocity.status_message;
      backend_result.final_error = advisory.per_objective_error.empty()
                                       ? 0.0
                                       : advisory.per_objective_error.front();
      backend_result.task_scales = {-1.0};
      backend_result.task_errors = advisory.per_objective_error;
      backend_result.task_modes_effective = {TaskSolveMode::kMinError};
      backend_result.task_used_fallback = {false};
      backend_result.condition_number = advisory.condition_number;
      result.weighted_fallback_used = true;
      result.recovery_stage = SolverRecoveryStage::kWeightedFallback;
      classified_velocity = classify_velocity_outcome(
          backend_result.status, backend_result.status_message,
          backend_result.task_scales, primary_goal_norm,
          backend_result.task_modes_effective,
          backend_result.task_used_fallback);
    }
  }

  auto infer_active_task_layout = [&]() {
    bool saw_pose = false;
    bool saw_split = false;
    for (const auto &task : objective_tasks) {
      if (!task) {
        continue;
      }
      const TaskType type = task->getType();
      saw_pose = saw_pose || type == TaskType::FRAME_POSE;
      saw_split = saw_split || type == TaskType::FRAME_POSITION ||
                              type == TaskType::FRAME_ORIENTATION;
    }
    if (runtime_config_.enable_auto_task_layout) {
      return current_auto_task_layout_;
    }
    if (saw_pose && !saw_split) {
      return TaskLayout::kMerged;
    }
    return TaskLayout::kSplit;
  };
  result.active_task_layout = infer_active_task_layout();
  result.binding_score = auto_layout_binding_score_;

  // Create velocity-specific result
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
  result.condition_number = backend_result.condition_number;

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
    if (apply_limits && c_lower.size() >= robot_->nv() && !use_contact_projection) {
      Eigen::Map<Eigen::VectorXd> dq(result.solution.data(),
                                     result.solution.size());
      // Velocity-box clamping (always applies to first nv rows).
      clamp_joint_velocity_solution_in_place(
          dq, c_lower, c_upper, use_position_limits_, robot_->nv());
      // Sparse position-limit clamping using dense per-joint bounds array.
      if (use_position_limits_) {
        const int n_dq = static_cast<int>(dq.size());
        const int n_dens = static_cast<int>(dense_pos_lower_sp.size());
        const int n_clamp = std::min(n_dq, n_dens);
        for (int k = 0; k < n_clamp; ++k) {
          if (dense_pos_lower_sp[k] < kNoPosBound)
            dq[k] = std::max(dq[k], dense_pos_lower_sp[k]);
          if (dense_pos_upper_sp[k] > -kNoPosBound)
            dq[k] = std::min(dq[k], dense_pos_upper_sp[k]);
        }
      }
    }

    result.joint_velocities = Eigen::Map<const Eigen::VectorXd>(
        result.solution.data(), result.solution.size());
    last_solution_dq_norm_ = result.joint_velocities.norm();

    // A position step may invoke several speculative velocity solves before it
    // accepts one outer-tick command.  Only the accepted position result may
    // advance acceleration history; direct velocity solves still commit here.
    if (acceleration_limits_enabled_ && position_step_call_depth_ == 0) {
      previous_dq_ = result.joint_velocities;
    }

    // Identify saturated joints based on constraint bounds
    if (apply_limits && c_lower.size() >= robot_->nv()) {
      double tolerance = 0.01; // 1% tolerance

      // Use the same combined bounds as post-solve clamping so saturation
      // reporting reflects either velocity-row or position-row activation.
      for (int i = 0; i < robot_->nv(); ++i) {
        double joint_vel = result.joint_velocities[i];
        double lower = c_lower[i];
        double upper = c_upper[i];
        if (use_position_limits_ && i < static_cast<int>(dense_pos_lower_sp.size())) {
          // Use dense per-joint position bounds (O(1) lookup, no hash overhead).
          if (dense_pos_lower_sp[i] < kNoPosBound)
            lower = std::max(lower, dense_pos_lower_sp[i]);
          if (dense_pos_upper_sp[i] > -kNoPosBound)
            upper = std::min(upper, dense_pos_upper_sp[i]);
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
  update_auto_task_layout_feedback(result);

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
    robot_->update_configuration(seed_q);
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
  bool collision_violated_flag_mt = false;
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

    Eigen::MatrixXd contact_P_c;
    const bool use_contact_projection = !contact_frames_.empty();
    if (use_contact_projection) {
      contact_P_c = compute_contact_projector();
      for (auto &J : jacobians) {
        J = J * contact_P_c;
      }
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
    if (use_contact_projection && collision_constraint_result.has_value()) {
      collision_constraint_result->jacobian =
          collision_constraint_result->jacobian * contact_P_c;
    }

    std::optional<LinearVelocityConstraintResult> linear_constraint_result =
        compute_linear_velocity_constraints();
    if (linear_constraint_result.has_value() &&
        !options.excluded_joint_indices.empty()) {
      for (int idx : options.excluded_joint_indices) {
        if (idx >= 0 && idx < linear_constraint_result->jacobian.cols()) {
          linear_constraint_result->jacobian.col(idx).setZero();
        }
      }
    }
    if (use_contact_projection && linear_constraint_result.has_value()) {
      linear_constraint_result->jacobian =
          linear_constraint_result->jacobian * contact_P_c;
    }

    std::optional<LinearVelocityConstraintResult> tight_pose_constraint_result =
        compute_tight_frame_pose_constraints();
    if (tight_pose_constraint_result.has_value() &&
        !options.excluded_joint_indices.empty()) {
      for (int idx : options.excluded_joint_indices) {
        if (idx >= 0 && idx < tight_pose_constraint_result->jacobian.cols()) {
          tight_pose_constraint_result->jacobian.col(idx).setZero();
        }
      }
    }
    if (use_contact_projection && tight_pose_constraint_result.has_value()) {
      tight_pose_constraint_result->jacobian =
          tight_pose_constraint_result->jacobian * contact_P_c;
    }

    std::optional<LinearVelocityConstraintResult> tight_point_constraint_result =
        compute_tight_point_constraints();
    if (tight_point_constraint_result.has_value() &&
        !options.excluded_joint_indices.empty()) {
      for (int idx : options.excluded_joint_indices) {
        if (idx >= 0 && idx < tight_point_constraint_result->jacobian.cols()) {
          tight_point_constraint_result->jacobian.col(idx).setZero();
        }
      }
    }
    if (use_contact_projection && tight_point_constraint_result.has_value()) {
      tight_point_constraint_result->jacobian =
          tight_point_constraint_result->jacobian * contact_P_c;
    }

    std::optional<ConstraintBlock> torso_constraint_result = std::nullopt;
    if (torso_task && torso_has_pose_bounds) {
      torso_constraint_result = build_torso_pose_bound_rows(
          *this, *robot_, torso_opts, torso_pose_bounds_reference, options.dt,
          options.excluded_joint_indices);
      if (use_contact_projection && torso_constraint_result.has_value()) {
        torso_constraint_result->jacobian =
            torso_constraint_result->jacobian * contact_P_c;
      }
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
    if (linear_constraint_result.has_value()) {
      num_constraints +=
          static_cast<int>(linear_constraint_result->jacobian.rows());
    }
    if (tight_pose_constraint_result.has_value()) {
      num_constraints +=
          static_cast<int>(tight_pose_constraint_result->jacobian.rows());
    }
    if (tight_point_constraint_result.has_value()) {
      num_constraints +=
          static_cast<int>(tight_point_constraint_result->jacobian.rows());
    }

    Eigen::MatrixXd C = Eigen::MatrixXd::Zero(num_constraints, robot_->nv());
    Eigen::VectorXd c_lower =
        Eigen::VectorXd::Constant(num_constraints, -1e10);
    Eigen::VectorXd c_upper =
        Eigen::VectorXd::Constant(num_constraints, 1e10);
    Eigen::VectorXd max_softening_factors =
        Eigen::VectorXd::Ones(num_constraints);

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
    if (linear_constraint_result.has_value()) {
      const auto &lc = linear_constraint_result.value();
      const int rows = static_cast<int>(lc.jacobian.rows());
      C.block(constraint_idx, 0, rows, robot_->nv()) = lc.jacobian;
      c_lower.segment(constraint_idx, rows) = lc.lower_bounds;
      c_upper.segment(constraint_idx, rows) = lc.upper_bounds;
      constraint_idx += rows;
    }
    if (tight_pose_constraint_result.has_value()) {
      const auto &tpc = tight_pose_constraint_result.value();
      const int rows = static_cast<int>(tpc.jacobian.rows());
      C.block(constraint_idx, 0, rows, robot_->nv()) = tpc.jacobian;
      c_lower.segment(constraint_idx, rows) = tpc.lower_bounds;
      c_upper.segment(constraint_idx, rows) = tpc.upper_bounds;
      max_softening_factors.segment(constraint_idx, rows).setConstant(4.0);
      constraint_idx += rows;
    }
    if (tight_point_constraint_result.has_value()) {
      const auto &tpc = tight_point_constraint_result.value();
      const int rows = static_cast<int>(tpc.jacobian.rows());
      C.block(constraint_idx, 0, rows, robot_->nv()) = tpc.jacobian;
      c_lower.segment(constraint_idx, rows) = tpc.lower_bounds;
      c_upper.segment(constraint_idx, rows) = tpc.upper_bounds;
      max_softening_factors.segment(constraint_idx, rows).setConstant(4.0);
      constraint_idx += rows;
    }

    const bool has_soft_rows = max_softening_factors.maxCoeff() > 1.0;
    if (has_soft_rows) {
      const int nv = robot_->nv();
      Eigen::VectorXd dq_lower_seed = Eigen::VectorXd::Constant(nv, -1e10);
      Eigen::VectorXd dq_upper_seed = Eigen::VectorXd::Constant(nv, 1e10);
      for (int i = 0; i < nv && i < c_lower.size() && i < c_upper.size(); ++i) {
        dq_lower_seed(i) = c_lower(i);
        dq_upper_seed(i) = c_upper(i);
      }
      if (!has_rowwise_interval_feasibility(C, c_lower, c_upper, dq_lower_seed,
                                            dq_upper_seed,
                                            constraint_tolerance_)) {
        const double pre_relax_scale =
            std::max(1.0, max_softening_factors.maxCoeff());
        apply_constraint_softening_in_place(c_lower, c_upper,
                                            max_softening_factors,
                                            pre_relax_scale);
      }
    }

    sanitize_solver_inputs(goals, jacobians, C, c_lower, c_upper);

    VelocitySolverConfig config;
    config.epsilon = constraint_tolerance_;
    config.precision_threshold = tight_tolerance_;
    config.iteration_limit = max_iterations_;
    config.magnitude_limit = norm_threshold_;
    config.stall_detection_count = max_zero_scale_iterations_;
    if (has_soft_rows) {
      config.stall_detection_count = std::max(5, max_zero_scale_iterations_);
    }
    config.regularization_config.epsilon = solver_tolerance_;
    config.regularization_config.regularization_factor = damping_;

    // Soft joint-space metric (contribution knob), same change of variables as
    // solve_velocity: solve in u-space on scaled copies (J D, C D), un-scale dq.
    const bool metric_active_ps =
        joint_metric_col_scale_.size() == robot_->nv();
    std::vector<Eigen::MatrixXd> metric_jacobians_ps;
    Eigen::MatrixXd metric_C_ps;
    const std::vector<Eigen::MatrixXd> *solve_jacobians_ps = &jacobians;
    const Eigen::MatrixXd *solve_C_ps = &C;
    if (metric_active_ps) {
      const auto D = joint_metric_col_scale_.asDiagonal();
      metric_jacobians_ps.reserve(jacobians.size());
      for (const auto &J : jacobians) metric_jacobians_ps.push_back(J * D);
      metric_C_ps = C * D;
      solve_jacobians_ps = &metric_jacobians_ps;
      solve_C_ps = &metric_C_ps;
    }
    auto vel_result = computeMultiObjectiveVelocitySolutionEigen(
        goals, *solve_jacobians_ps, *solve_C_ps, c_lower, c_upper, config,
        objective_configs);
    if (metric_active_ps &&
        static_cast<int>(vel_result.solution.size()) == robot_->nv()) {
      for (int j = 0; j < robot_->nv(); ++j)
        vel_result.solution[j] *= joint_metric_col_scale_(j);
    }

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
    if (use_contact_projection) {
      dq = contact_P_c * dq;
    }
    if (!use_contact_projection) {
      clamp_joint_velocity_solution_in_place(dq, c_lower, c_upper,
                                             /*include_position_rows=*/false,
                                             robot_->nv());
    }
    if (use_position_limits_ && dq.size() == robot_->nv()) {
      auto [q_min, q_max] = robot_->get_joint_limits();
      if (elastic_band_config_.enabled &&
          elastic_band_state_.delta.size() == robot_->nv()) {
        expand_scalar_joint_limits_for_velocity_delta(
            q_min, q_max, elastic_band_state_.delta, velocity_to_config_index,
            robot_->nv());
      }
      clamp_joint_velocity_to_position_limits_for_integration(
          dq, q_current, q_min, q_max, velocity_to_config_index, options.dt,
          robot_->nv());
    }
    Eigen::VectorXd q_pre_step = q_current;
    q_current =
        pinocchio::integrate(robot_->model(), q_current, options.dt * dq);
    if (use_position_limits_) {
      auto [q_min_true, q_max_true] = robot_->get_joint_limits();
      project_scalar_configuration_to_true_joint_limits(
          q_current, q_min_true, q_max_true, velocity_to_config_index,
          robot_->nv());
    }
    robot_->update_configuration(q_current);

    // Skip expensive post-step checks when integration produced no motion.
    const bool step_moved =
        (q_current - q_pre_step).squaredNorm() > kCollisionEscapeNormEps;
    // Post-solve collision rejection (same logic as solve_position_step).
    if (step_moved && collision_constraint_.has_value() &&
        collision_constraint_->enabled) {
      // Keep the legacy global guard for penetration diagnostics while the
      // recovery margin below enforces every pair's effective floor.
      const double violation_threshold = collision_constraint_->min_distance;
      const auto pre_dist_debug =
          evaluate_post_step_collision_distance_mutating(q_pre_step);
      const auto pre_recovery_margins =
          evaluate_post_step_collision_recovery_margins(q_pre_step);
      const auto post_dist_debug =
          evaluate_post_step_collision_distance_mutating(q_current);
      const auto post_recovery_margins =
          evaluate_post_step_collision_recovery_margins(q_current);
      const double pre_dist =
          (pre_dist_debug.has_value() && std::isfinite(*pre_dist_debug))
              ? *pre_dist_debug
              : std::numeric_limits<double>::infinity();
      const double post_dist =
          (post_dist_debug.has_value() && std::isfinite(*post_dist_debug))
              ? *post_dist_debug
              : std::numeric_limits<double>::infinity();
      const bool seed_was_safe = pre_dist >= violation_threshold;
      const bool deepened =
          pre_dist < violation_threshold &&
          post_dist < pre_dist - kCollisionPenetrationWorsenTolerance;
      const bool hard_jump =
          std::isfinite(pre_dist) &&
          post_dist < kCollisionHardPenetrationRejectDistance &&
          post_dist < pre_dist - kCollisionHardWorsenTolerance;
      const bool recovery_floor_unacceptable =
          !collision_recovery_margins_acceptable(
              pre_recovery_margins, post_recovery_margins, 0.0);
      const bool recovery_floor_seed_was_safe =
          collision_recovery_margins_are_safe(pre_recovery_margins, 0.0);
      if (seed_was_safe || deepened || hard_jump ||
          recovery_floor_unacceptable) {
        q_current = q_pre_step;
        robot_->update_configuration(q_current);
        result.collision_rejection_count++;
        if (seed_was_safe || recovery_floor_seed_was_safe) {
          collision_violated_flag_mt = true;
          break;
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
  if (collision_violated_flag_mt) {
    result.status = SolverStatus::kCollisionViolated;
    result.status_message =
        "solve_position: no step could maintain collision min_distance; "
        "q_solution is the last safe configuration";
  }

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
  const Eigen::VectorXd previous_applied_velocity = previous_dq_;

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

  const bool outermost_position_step_call = position_step_call_depth_ == 0;
  if (outermost_position_step_call) {
    last_post_step_collision_exact_distance_queries_ = 0;
    last_post_step_collision_motion_bound_culled_pairs_ = 0;
  }
  const bool owns_position_step_continuity =
      outermost_position_step_call ||
      (suppress_preferred_lock_step_retry_ &&
       position_step_call_depth_ == 1) ||
      (suppress_min_error_step_retry_ && position_step_call_depth_ <= 2);
  ScopedPositionStepCallDepth position_step_depth(position_step_call_depth_);
  std::optional<pinocchio::SE3> continuity_reference_pose;
  if (!options.continuity_reference_frame.empty()) {
    if (!robot_->has_frame(options.continuity_reference_frame)) {
      result.status = SolverStatus::kInvalidInput;
      result.status_message =
          "continuity_reference_frame not found in robot model";
      return result;
    }
    robot_->update_configuration(current_q);
    continuity_reference_pose =
        robot_->get_frame_pose(options.continuity_reference_frame);
  }
  if (outermost_position_step_call) {
    const bool had_previous_target_signature =
        last_position_step_target_signature_.has_value();
    PositionStepTargetSignature signature;
    signature.command_revision = options.continuity_command_revision;
    const std::unordered_set<std::string> explicit_target_task_names = {
        frame_task_name};
    signature.task_names.push_back(frame_task_name);
    signature.target_poses.push_back(target_pose);
    if (continuity_reference_pose.has_value()) {
      signature.reference_target_poses.push_back(
          canonicalize_position_step_signature_pose(
              frame_task_name, target_pose, continuity_reference_pose));
    }
    signature.gains = {options.position_gain, options.orientation_gain};
    signature.task_names.push_back("#continuity:" +
                                   options.continuity_reference_frame);
    signature.task_names.push_back("#torso:" +
                                   options.torso_constraint.frame_name);
    for (const auto &task : tasks_) {
      if (!task) {
        continue;
      }
      signature.task_names.push_back("#policy:" + task->getName());
      append_task_policy_signature(
          signature.gains, *task,
          explicit_target_task_names.find(task->getName()) ==
              explicit_target_task_names.end());
    }
    append_position_step_option_signature(signature.gains, options);
    update_position_step_target_motion_blocks(signature);
    position_step_target_geometry_moved_ =
        position_step_target_geometry_changed(signature);
    position_step_target_motion_observed_ =
        position_step_target_motion_observed_ ||
        (had_previous_target_signature &&
         position_step_target_geometry_moved_);
    if (update_position_step_target_signature(std::move(signature))) {
      capture_position_step_collision_command_floor(current_q);
    }
  }

  if (!options.preferred_locked_joint_indices.empty() &&
      !suppress_preferred_lock_step_retry_) {
    return solve_position_step_with_preferred_lock(
        current_q, target_pose, frame_task_name, options);
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
  if (!suppress_min_error_step_retry_) {
    apply_position_step_primary_task_options(options, frame_task.get());
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
    robot_->update_configuration(current_q);
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
  // Keep the frozen set visible to collision computations that run after the
  // inner solve_velocity() clears pending_velocity_lock_indices_ (stall escape,
  // post-step validation), so the collision debug list and QP slots agree on
  // which pairs are controllable. Cleared on every exit path.
  position_step_locked_indices_ = step_locked_indices;
  struct ClearPositionStepLocks {
    KinematicsSolver *solver;
    ~ClearPositionStepLocks() {
      if (solver != nullptr) {
        solver->position_step_locked_indices_.clear();
      }
    }
  } clear_position_step_locks{this};
  (void)clear_position_step_locks;

  VelocitySolverResult last_vel_result;
  bool have_vel_result = false;
  Eigen::VectorXd first_tick_velocity;
  const int steps = std::max(1, options.max_steps);
  int steps_used = 0;
  Eigen::VectorXd vel(6);
  double prev_combined_error = std::numeric_limits<double>::infinity();
  double initial_combined_error = std::numeric_limits<double>::quiet_NaN();
  double initial_commanded_error = std::numeric_limits<double>::quiet_NaN();
  std::vector<double> initial_target_block_merits;
  int no_progress_count = 0;
  bool no_progress_exit = false;
  bool collision_violated_flag = false;
  bool recovery_inside_collision_margin = false;
  bool configuration_step_limited = false;
  bool outer_acceleration_limit_applied = false;

  for (int step = 0; step < steps; ++step) {
    frame_task->update(*robot_);
    const Eigen::VectorXd &error = frame_task->getError();
    double combined_error = 0.0;
    if (error.size() == 3) {
      combined_error = error.head<3>().norm();
      const bool is_orientation_only =
          frame_task->getType() == TaskType::FRAME_ORIENTATION;
      Eigen::VectorXd v3 = (is_orientation_only ? options.orientation_gain
                                                : options.position_gain) *
                           error.head<3>();
      if (is_orientation_only) {
        if (options.max_angular_speed > 0.0) {
          const double n = v3.norm();
          if (n > options.max_angular_speed) {
            v3 *= options.max_angular_speed / n;
          }
        }
      } else {
        if (options.max_linear_speed > 0.0) {
          const double n = v3.norm();
          if (n > options.max_linear_speed) {
            v3 *= options.max_linear_speed / n;
          }
        }
      }
      frame_task->setPositionStepTargetVelocity(v3);
    } else if (error.size() >= 6) {
      combined_error = error.head<3>().norm() + error.tail<3>().norm();
      vel.head<3>() = options.position_gain * error.head<3>();
      vel.tail<3>() = options.orientation_gain * error.tail<3>();
      clamp_spatial_velocity_components(vel, options.max_linear_speed,
                                        options.max_angular_speed);
      frame_task->setPositionStepTargetVelocity(vel);
    } else {
      result.status = SolverStatus::kInvalidInput;
      result.status_message =
          "frame task has invalid pose error dimension for solve_position_step";
      break;
    }
    if (step == 0) {
      initial_combined_error = combined_error;
      initial_commanded_error = commanded_frame_merit(
          error, frame_task->getType(), options.position_gain,
          options.orientation_gain);
      const auto block_merits = commanded_frame_block_merits(
          error, frame_task->getType(), options.position_gain,
          options.orientation_gain);
      initial_target_block_merits.assign(block_merits.begin(),
                                         block_merits.end());
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

    const double step_dt_eff = compute_adaptive_position_step_dt(
        options, step_dt, error.head<3>().norm(),
        collision_constraint_.has_value() && collision_constraint_->enabled,
        last_constraint_min_recovery_margin_);
    pending_velocity_lock_indices_ = step_locked_indices;
    pending_step_torso_constraint_ = step_torso_constraint;
    pending_step_validation_dt_ = step_dt_eff;
    pending_reuse_current_kinematics_ = true;
    const bool apply_setpoint_acceleration_limits =
        acceleration_limits_enabled_ && !position_step_target_geometry_moved_ &&
        step == 0;
    pending_position_step_acceleration_limits_ =
        apply_setpoint_acceleration_limits;
    VelocitySolverResult vel_out = solve_velocity(q, true);
    vel_out = retry_auto_task_layout_as_split_if_needed(
        q, std::move(vel_out), step_locked_indices, step_torso_constraint,
        step_dt_eff, {}, apply_setpoint_acceleration_limits);
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
    // Adaptive dt: scale integration step proportional to current position
    // error so large-jump approach is faster; reverts to base dt near target.
    // Proximity-aware cap: limit scale so the integration step cannot overshoot
    // past min_distance in one tick (scale * step_dt * max_ee_speed <= clearance).
    const bool adaptive_step_large = (step_dt_eff > step_dt * 1.01);
    if (use_position_limits_ &&
        last_vel_result.joint_velocities.size() == robot_->nv()) {
      auto [q_min, q_max] = robot_->get_joint_limits();
      if (elastic_band_config_.enabled &&
          elastic_band_state_.delta.size() == robot_->nv()) {
        expand_scalar_joint_limits_for_velocity_delta(
            q_min, q_max, elastic_band_state_.delta, velocity_to_config_index,
            robot_->nv());
      }
      clamp_joint_velocity_to_position_limits_for_integration(
          last_vel_result.joint_velocities, q, q_min, q_max,
          velocity_to_config_index, step_dt_eff, robot_->nv());
    }
    if (step == 0 &&
        last_vel_result.joint_velocities.size() == robot_->nv() &&
        last_vel_result.joint_velocities.allFinite()) {
      first_tick_velocity = last_vel_result.joint_velocities;
    }
    q = pinocchio::integrate(robot_->model(), q,
                             step_dt_eff * last_vel_result.joint_velocities);
    if (use_position_limits_) {
      auto [q_min_true, q_max_true] = robot_->get_joint_limits();
      project_scalar_configuration_to_true_joint_limits(
          q, q_min_true, q_max_true, velocity_to_config_index, robot_->nv());
    }
    const bool step_configuration_limited =
        clamp_configuration_delta_from_reference(
            robot_->model(), q_reference, q,
            options.max_configuration_step_norm);
    configuration_step_limited =
        step_configuration_limited || configuration_step_limited;
    robot_->update_configuration(q);

    // Skip expensive post-step checks when integration produced no motion.
    const bool step_moved =
        (q - q_pre_step).squaredNorm() > kCollisionEscapeNormEps;

    // Adaptive steps need all active pairs reconsidered at q_eval. The recovery
    // evaluator already performs that broadphase expansion and leaves a scalar
    // distance result behind, so reuse it instead of issuing a second scan.
    auto eval_broadphase_expanded = [&](const Eigen::VectorXd &q_eval)
        -> std::optional<double> {
      evaluate_post_step_collision_recovery_margins(q_eval);
      return evaluate_post_step_collision_distance_from_current_results();
    };

    // Post-solve collision rejection (same logic as multi-target overload).
    if (step_moved && collision_constraint_.has_value() &&
        collision_constraint_->enabled) {
      // Keep the legacy global guard for penetration diagnostics while the
      // recovery margin below enforces every pair's effective floor.
      const double violation_threshold = collision_constraint_->min_distance;
      const auto pre_dist_debug =
          evaluate_post_step_collision_distance_mutating(q_pre_step);
      const auto pre_recovery_margins =
          evaluate_post_step_collision_recovery_margins(q_pre_step);
      std::optional<double> post_dist_debug =
          adaptive_step_large
          ? eval_broadphase_expanded(q)
          : evaluate_post_step_collision_distance_mutating(q);
      const auto post_recovery_margins =
          evaluate_post_step_collision_recovery_margins(q);
      const double pre_dist =
          (pre_dist_debug.has_value() && std::isfinite(*pre_dist_debug))
              ? *pre_dist_debug
              : std::numeric_limits<double>::infinity();
      const double post_dist =
          (post_dist_debug.has_value() && std::isfinite(*post_dist_debug))
              ? *post_dist_debug
              : std::numeric_limits<double>::infinity();
      const bool recovery_floor_unacceptable =
          !collision_recovery_margins_acceptable(
              pre_recovery_margins, post_recovery_margins, 0.0);
      const bool recovery_floor_seed_was_safe =
          collision_recovery_margins_are_safe(pre_recovery_margins, 0.0);
      if (post_dist < violation_threshold ||
          recovery_floor_unacceptable) {
        bool seed_was_safe = (pre_dist >= violation_threshold);
        bool deepened =
            (pre_dist < violation_threshold &&
             post_dist <
                 pre_dist - kCollisionPenetrationWorsenTolerance);
        const bool hard_jump =
            std::isfinite(pre_dist) &&
            post_dist < kCollisionHardPenetrationRejectDistance &&
            post_dist < pre_dist - kCollisionHardWorsenTolerance;
        const bool pure_primary_min_error =
            !last_vel_result.task_modes_effective.empty() &&
            last_vel_result.task_modes_effective[0] == TaskSolveMode::kMinError &&
            (last_vel_result.task_used_fallback.empty() ||
             !last_vel_result.task_used_fallback[0]);
        const bool step_has_velocity_locks =
            !options.excluded_joint_indices.empty() ||
            !options.integration_zero_velocity_indices.empty();
        const bool enforce_material_penetration_guard =
            !pure_primary_min_error || step_has_velocity_locks;
        const bool post_penetrating =
            post_dist < kCollisionPenetrationDistanceThreshold;
        const bool crossed_into_penetration =
            enforce_material_penetration_guard && post_penetrating &&
            (!std::isfinite(pre_dist) ||
             pre_dist >= kCollisionPenetrationDistanceThreshold);
        const bool material_penetration_worsened =
            enforce_material_penetration_guard &&
            post_penetrating && std::isfinite(pre_dist) &&
            pre_dist < kCollisionPenetrationDistanceThreshold &&
            post_dist < pre_dist - kCollisionTolerance;
        if (seed_was_safe || deepened || hard_jump ||
            crossed_into_penetration || material_penetration_worsened ||
            recovery_floor_unacceptable) {
          // Backoff threshold depends on seed state:
          //   safe seed   → must restore full safety (>= min_distance)
          //   violated seed + hard geometry jump → stop geometry penetration
          //   violated seed + deepened → must not worsen beyond tolerance
          double backoff_threshold;
          if (seed_was_safe) {
            backoff_threshold = violation_threshold;
          } else if (crossed_into_penetration) {
            backoff_threshold = kCollisionPenetrationDistanceThreshold;
          } else if (material_penetration_worsened) {
            backoff_threshold = pre_dist - kCollisionTolerance;
          } else if (hard_jump) {
            backoff_threshold = kCollisionPenetrationDistanceThreshold;
          } else {
            backoff_threshold = pre_dist - kCollisionPenetrationWorsenTolerance;
          }
          bool accepted_backoff = false;
          const Eigen::VectorXd dq_nominal =
              step_dt_eff * last_vel_result.joint_velocities;
          for (double frac : kCollisionRejectionBackoffFractions) {
            Eigen::VectorXd q_backoff = pinocchio::integrate(
                robot_->model(), q_pre_step, frac * dq_nominal);
            robot_->update_configuration(q_backoff);
            // Use same evaluation level as initial check: broadphase-expanded
            // when adaptive_step_large so newly-entered pairs are checked at
            // each backoff position without a full GJK scan.
            auto backoff_dist_debug =
                adaptive_step_large
                ? eval_broadphase_expanded(q_backoff)
                : evaluate_post_step_collision_distance_mutating(q_backoff);
            const auto backoff_recovery_margins =
                evaluate_post_step_collision_recovery_margins(q_backoff);
            const bool backoff_recovery_acceptable =
                collision_recovery_margins_acceptable(
                    pre_recovery_margins, backoff_recovery_margins, 0.0);
            if (backoff_dist_debug.has_value() &&
                std::isfinite(*backoff_dist_debug) &&
                *backoff_dist_debug >= backoff_threshold &&
                backoff_recovery_acceptable) {
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
            if (seed_was_safe || recovery_floor_seed_was_safe) {
              // Safe seed: no step could maintain min_distance → violation.
              // Signal after loop; break so we don't attempt further steps.
              collision_violated_flag = true;
              break;
            }
            // Violated seed: hold at seed, let stall handler relax min_distance.
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
              auto esc_dist =
                  evaluate_post_step_collision_distance_mutating(q_candidate);
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
          auto esc_dist =
              evaluate_post_step_collision_distance_mutating(q_candidate);
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
    std::vector<int> desaturation_candidates = last_vel_result.saturated_joints;
    const bool saturated_zero_motion_plateau =
        combined_error > 1e-4 &&
        effective_step_dq_norm < stall_config_.dq_stall_eps &&
        !desaturation_candidates.empty() &&
        plateau_stall_steps >= kJointLimitDesaturationPlateauThreshold;
    const bool plateau_escape =
        (plateau_status || saturated_zero_motion_plateau) &&
        plateau_stall_steps >= kJointLimitDesaturationPlateauThreshold;
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
        auto curr_dist = evaluate_post_step_collision_distance_mutating(q);
        robot_->update_configuration(q_candidate);
        auto cand_dist =
            evaluate_post_step_collision_distance_mutating(q_candidate);
        const double desat_safe_threshold =
            (collision_constraint_.has_value() && collision_constraint_->enabled)
            ? collision_constraint_->min_distance
            : kCollisionPenetrationDistanceThreshold;
        if (collision_recovery_candidate_acceptable(
                curr_dist, cand_dist, desat_safe_threshold)) {
          q = q_candidate;
          result.stall_escape_count++;
          recovery_inside_collision_margin =
              recovery_inside_collision_margin ||
              (curr_dist.has_value() && std::isfinite(*curr_dist) &&
               *curr_dist < desat_safe_threshold &&
               cand_dist.has_value() && std::isfinite(*cand_dist) &&
               *cand_dist < desat_safe_threshold);
        } else {
          robot_->update_configuration(q);
        }
      }
    }
  }

  if (auto retry_result = attempt_min_error_position_step_retry(
          current_q, options, step_dt, have_vel_result, last_vel_result, q,
          [&](std::vector<std::pair<Task *, TaskSolveMode>> *saved_modes) {
            return flip_scale_family_task(frame_task.get(), saved_modes);
          },
          [&]() {
            return solve_position_step(current_q, target_pose, frame_task_name,
                                       options);
          });
      retry_result.has_value()) {
    return retry_result.value();
  }

  if (position_step_target_geometry_moved_) {
    const Eigen::VectorXd terminal_q = q;
    robot_->update_configuration(current_q);
    frame_task->update(*robot_);
    const Eigen::VectorXd &current_error = frame_task->getError();
    const bool all_task_blocks_commanded = all_frame_task_blocks_commanded(
        current_error, frame_task->getType(), options.position_gain,
        options.orientation_gain);
    if (current_error.size() == 3) {
      const bool is_orientation_only =
          frame_task->getType() == TaskType::FRAME_ORIENTATION;
      Eigen::VectorXd current_velocity =
          (is_orientation_only ? options.orientation_gain
                               : options.position_gain) *
          current_error.head<3>();
      const double speed_limit = is_orientation_only
                                     ? options.max_angular_speed
                                     : options.max_linear_speed;
      if (speed_limit > 0.0 && current_velocity.norm() > speed_limit) {
        current_velocity *= speed_limit / current_velocity.norm();
      }
      frame_task->setPositionStepTargetVelocity(current_velocity);
    } else if (current_error.size() >= 6) {
      vel.head<3>() = options.position_gain * current_error.head<3>();
      vel.tail<3>() = options.orientation_gain * current_error.tail<3>();
      clamp_spatial_velocity_components(vel, options.max_linear_speed,
                                        options.max_angular_speed);
      frame_task->setPositionStepTargetVelocity(vel);
    }
    const double terminal_prediction_weight =
        position_step_task_terminal_prediction_weight(*frame_task,
                                                      current_error);
    const auto componentwise_candidate =
        all_task_blocks_commanded && terminal_prediction_weight > 0.0
            ? estimate_position_step_componentwise_outer_candidate(
                  current_q, previous_applied_velocity, options, step_dt,
                  terminal_q)
            : std::nullopt;
    if (auto projected_result = apply_position_step_task_metric_projection(
            current_q, step_dt, first_tick_velocity, step_locked_indices,
            step_torso_constraint, {}, q);
        projected_result.has_value()) {
      last_vel_result = std::move(*projected_result);
      have_vel_result = true;
      if (componentwise_candidate.has_value() && all_task_blocks_commanded &&
          terminal_prediction_weight > 0.0) {
        const Eigen::VectorXd projected_q = q;
        const auto block_merits_at = [&](const Eigen::VectorXd &candidate_q) {
          robot_->update_configuration(candidate_q);
          frame_task->update(*robot_);
          return commanded_frame_block_merits(
              frame_task->getError(), frame_task->getType(),
              options.position_gain, options.orientation_gain);
        };
        const auto projected_merits = block_merits_at(q);
        const auto candidate_is_preferred =
            [&](const Eigen::VectorXd &candidate_q) {
          const auto candidate_merits = block_merits_at(candidate_q);
          const double projected_total =
              projected_merits[0] + projected_merits[1];
          const double candidate_total =
              candidate_merits[0] + candidate_merits[1];
          return std::isfinite(projected_merits[0]) &&
                 std::isfinite(projected_merits[1]) &&
                 std::isfinite(candidate_merits[0]) &&
                 std::isfinite(candidate_merits[1]) &&
                 candidate_merits[0] <= projected_merits[0] + 1e-9 &&
                 candidate_merits[1] <= projected_merits[1] + 1e-9 &&
                 candidate_total + 1e-9 < projected_total;
        };
        Eigen::VectorXd safe_componentwise_candidate =
            *componentwise_candidate;
        if (terminal_prediction_weight < 1.0) {
          const Eigen::VectorXd projected_delta = pinocchio::difference(
              robot_->model(), current_q, projected_q);
          const Eigen::VectorXd componentwise_delta = pinocchio::difference(
              robot_->model(), current_q, safe_componentwise_candidate);
          safe_componentwise_candidate = pinocchio::integrate(
              robot_->model(), current_q,
              projected_delta + terminal_prediction_weight *
                                    (componentwise_delta - projected_delta));
        }
        if (candidate_is_preferred(safe_componentwise_candidate)) {
          apply_position_step_outer_acceleration_limit(
              current_q, previous_applied_velocity, options, step_dt,
              safe_componentwise_candidate);
          if (candidate_is_preferred(safe_componentwise_candidate)) {
            q = std::move(safe_componentwise_candidate);
            outer_acceleration_limit_applied = true;
          }
        }
        if (outer_acceleration_limit_applied) {
          last_vel_result.joint_velocities =
              pinocchio::difference(robot_->model(), current_q, q) / step_dt;
        }
        robot_->update_configuration(q);
      }
    }
  } else if (acceleration_limits_enabled_ &&
             first_tick_velocity.size() == robot_->nv() &&
             pinocchio::difference(robot_->model(), current_q, q)
                     .squaredNorm() > kCollisionEscapeNormEps) {
    q = pinocchio::integrate(robot_->model(), current_q,
                             step_dt * first_tick_velocity);
    if (use_position_limits_) {
      auto [q_min, q_max] = robot_->get_joint_limits();
      project_scalar_configuration_to_true_joint_limits(
          q, q_min, q_max, velocity_to_config_index, robot_->nv());
    }
    robot_->update_configuration(q);
  }

  for (auto &task : tasks_) {
    if (task) {
      task->clearPositionStepTargetVelocity();
    }
  }

  const bool final_configuration_limited =
      clamp_configuration_delta_from_reference(
          robot_->model(), q_reference, q,
          options.max_configuration_step_norm);
  configuration_step_limited =
      final_configuration_limited || configuration_step_limited;
  if (final_configuration_limited) {
    robot_->update_configuration(q);
  }
  if (!outer_acceleration_limit_applied || final_configuration_limited) {
    apply_position_step_outer_acceleration_limit(
        current_q, previous_applied_velocity, options, step_dt, q);
  }

  result.q_solution = q;
  result.achieved_pose = robot_->get_frame_pose(frame_task->getFrameName());
  result.iterations_used = steps_used;

  frame_task->update(*robot_);
  const Eigen::VectorXd &final_error = frame_task->getError();
  if (final_error.size() == 3) {
    result.position_error = final_error.head<3>().norm();
    result.orientation_error = 0.0;
  } else if (final_error.size() >= 6) {
    result.position_error = final_error.head<3>().norm();
    result.orientation_error = final_error.tail<3>().norm();
  }

  if (have_vel_result) {
    if (last_vel_result.joint_velocities.size() == robot_->nv()) {
      last_vel_result.solution.assign(last_vel_result.joint_velocities.data(),
                                      last_vel_result.joint_velocities.data() +
                                          last_vel_result.joint_velocities.size());
    }
    static_cast<VelocitySolverResult &>(result) = std::move(last_vel_result);
    if (!runtime_config_.enable_auto_task_layout) {
      result.active_task_layout =
          frame_task->getType() == TaskType::FRAME_POSE ? TaskLayout::kMerged
                                                        : TaskLayout::kSplit;
    }
  }
  if (no_progress_exit && result.status != SolverStatus::kInvalidInput) {
    result.status = SolverStatus::kNoProgress;
    result.status_message =
        "solve_position_step exited due to no progress near active bounds/"
        "constraints";
  }
  if (collision_violated_flag) {
    result.status = SolverStatus::kCollisionViolated;
    result.status_message =
        "solve_position_step: no step could maintain collision min_distance; "
        "q_solution is the last safe configuration";
  }
  if (collision_constraint_.has_value() && collision_constraint_->enabled &&
      (result.stall_escape_count > 0 ||
       result.collision_rejection_count > 0 ||
       recovery_inside_collision_margin)) {
    auto final_dist = evaluate_min_collision_distance(q);
    if (final_dist.has_value() && std::isfinite(*final_dist) &&
        *final_dist < collision_constraint_->min_distance - kCollisionTolerance) {
      recovery_inside_collision_margin = true;
    }
  }
  if (recovery_inside_collision_margin &&
      result.status == SolverStatus::kSuccess) {
    result.status = SolverStatus::kNoProgress;
    result.status_message =
        "solve_position_step applied recovery motion inside collision margin";
  }
  if (!recovery_inside_collision_margin &&
      should_report_position_recovery_success(
          result, current_q,
          stall_config_.dq_stall_eps * std::max(step_dt, 1e-9))) {
    result.status = SolverStatus::kSuccess;
    result.status_message =
        "solve_position_step applied constraint recovery motion";
  }
  const double final_commanded_error = commanded_frame_merit(
      final_error, frame_task->getType(), options.position_gain,
      options.orientation_gain);
  const auto final_block_merits = commanded_frame_block_merits(
      final_error, frame_task->getType(), options.position_gain,
      options.orientation_gain);
  const std::vector<double> final_target_block_merits(
      final_block_merits.begin(), final_block_merits.end());
  const double candidate_step_norm =
      result.q_solution.size() == current_q.size()
          ? pinocchio::difference(robot_->model(), current_q,
                                  result.q_solution)
                .norm()
          : std::numeric_limits<double>::quiet_NaN();
  const bool held_non_improving_step =
      should_hold_position_step_for_continuity(
          result, current_q, initial_commanded_error, final_commanded_error,
          {initial_commanded_error}, {final_commanded_error},
          initial_target_block_merits, final_target_block_merits,
          frame_task->getPriority(), candidate_step_norm,
          owns_position_step_continuity,
          collision_violated_flag, step_torso_constraint.has_value());
  if (held_non_improving_step) {
    const bool target_satisfied =
        initial_commanded_error <= kPositionStepSatisfiedMeritTolerance;
    Eigen::VectorXd braking_q = current_q;
    const bool braking_to_hold = compute_position_step_continuity_brake(
        current_q, previous_applied_velocity, options, step_dt, braking_q);
    result.position_step_hold_active = !braking_to_hold;
    q = braking_to_hold ? std::move(braking_q) : current_q;
    robot_->update_configuration(q);
    result.q_solution = q;
    result.achieved_pose = robot_->get_frame_pose(frame_task->getFrameName());
    frame_task->update(*robot_);
    const Eigen::VectorXd &held_error = frame_task->getError();
    if (held_error.size() == 3) {
      result.position_error = held_error.head<3>().norm();
      result.orientation_error = 0.0;
    } else if (held_error.size() >= 6) {
      result.position_error = held_error.head<3>().norm();
      result.orientation_error = held_error.tail<3>().norm();
    }
    result.status = braking_to_hold
                        ? SolverStatus::kSuccess
                        : (target_satisfied ? SolverStatus::kSuccess
                                            : SolverStatus::kNoProgress);
    result.status_message =
        braking_to_hold
            ? "solve_position_step decelerating before continuity hold"
            : (target_satisfied
                   ? "solve_position_step held satisfied stationary target at "
                     "the current configuration"
                   : "solve_position_step held current configuration because "
                     "the nominal step did not reduce commanded task error");
    sync_position_result_applied_velocity(result, robot_->model(), current_q,
                                          step_dt);
    last_solution_dq_norm_ = result.joint_velocities.norm();
    if (!braking_to_hold && previous_dq_.size() == robot_->nv()) {
      previous_dq_.setZero();
    }
  }
  if (result.stall_escape_count > 0 || result.collision_rejection_count > 0 ||
      collision_violated_flag || configuration_step_limited) {
    sync_position_result_applied_velocity(result, robot_->model(), current_q,
                                          step_dt);
  }
  if (!held_non_improving_step &&
      should_hold_soft_infeasible_position_step(
          result, current_q, initial_combined_error,
          result.position_error + result.orientation_error,
          result.position_error, result.orientation_error,
          collision_violated_flag,
          recovery_inside_collision_margin)) {
    Eigen::VectorXd braking_q = current_q;
    const bool braking_to_hold = compute_position_step_continuity_brake(
        current_q, previous_applied_velocity, options, step_dt, braking_q);
    result.position_step_hold_active = !braking_to_hold;
    q = braking_to_hold ? std::move(braking_q) : current_q;
    robot_->update_configuration(q);
    result.q_solution = q;
    result.achieved_pose = robot_->get_frame_pose(frame_task->getFrameName());
    frame_task->update(*robot_);
    const Eigen::VectorXd &held_error = frame_task->getError();
    if (held_error.size() == 3) {
      result.position_error = held_error.head<3>().norm();
      result.orientation_error = 0.0;
    } else if (held_error.size() >= 6) {
      result.position_error = held_error.head<3>().norm();
      result.orientation_error = held_error.tail<3>().norm();
    }
    result.status = braking_to_hold ? SolverStatus::kSuccess
                                    : SolverStatus::kNoProgress;
    result.status_message = braking_to_hold
                                ? "solve_position_step decelerating before "
                                  "soft-infeasible continuity hold"
                                : "solve_position_step held current configuration "
                                  "because the primary SCALE task was "
                                  "soft-infeasible";
    sync_position_result_applied_velocity(result, robot_->model(), current_q,
                                          step_dt);
    last_solution_dq_norm_ = result.joint_velocities.norm();
    if (!braking_to_hold && previous_dq_.size() == robot_->nv()) {
      previous_dq_.setZero();
    }
  }

  sync_position_result_applied_velocity(result, robot_->model(), current_q,
                                        step_dt);
  if (acceleration_limits_enabled_ &&
      result.joint_velocities.size() == robot_->nv()) {
    previous_dq_ = result.joint_velocities;
  }

  // In teleop-style loops (max_steps=1), stall handling must accumulate across
  // successive solve_position_step calls based on *applied* motion, not only
  // raw QP status.  This post-step update prevents success/no-progress
  // oscillations from resetting the consecutive stall counter.
  if (stall_handler_enabled()) {
    if (held_non_improving_step) {
      stall_state_.consecutive_stall_steps = 0;
    } else {
      const double applied_step_norm = (result.q_solution - current_q).norm();
      const double applied_step_eps =
          stall_config_.dq_stall_eps * std::max(step_dt, 1e-9);
      if (applied_step_norm < applied_step_eps) {
        stall_state_.consecutive_stall_steps++;
        stall_state_.total_stall_steps++;
      } else {
        stall_state_.consecutive_stall_steps = 0;
      }
    }
  }

  return result;
}

PositionIKResult KinematicsSolver::solve_position_step(
    const Eigen::VectorXd &current_q, const std::vector<TaskTarget> &targets,
    const PositionStepOptions &options) {

  PositionIKResult result;
  const double step_dt = (options.dt > 0.0) ? options.dt : dt_;
  const Eigen::VectorXd previous_applied_velocity = previous_dq_;

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

  const bool outermost_position_step_call = position_step_call_depth_ == 0;
  if (outermost_position_step_call) {
    last_post_step_collision_exact_distance_queries_ = 0;
    last_post_step_collision_motion_bound_culled_pairs_ = 0;
  }
  const bool owns_position_step_continuity =
      outermost_position_step_call ||
      (suppress_preferred_lock_step_retry_ &&
       position_step_call_depth_ == 1) ||
      (suppress_min_error_step_retry_ && position_step_call_depth_ <= 2);
  ScopedPositionStepCallDepth position_step_depth(position_step_call_depth_);
  std::optional<pinocchio::SE3> continuity_reference_pose;
  if (!options.continuity_reference_frame.empty()) {
    if (!robot_->has_frame(options.continuity_reference_frame)) {
      result.status = SolverStatus::kInvalidInput;
      result.status_message =
          "continuity_reference_frame not found in robot model";
      return result;
    }
    robot_->update_configuration(current_q);
    continuity_reference_pose =
        robot_->get_frame_pose(options.continuity_reference_frame);
  }
  if (outermost_position_step_call) {
    const bool had_previous_target_signature =
        last_position_step_target_signature_.has_value();
    PositionStepTargetSignature signature;
    signature.command_revision = options.continuity_command_revision;
    std::unordered_set<std::string> explicit_target_task_names;
    for (const auto &target : targets) {
      explicit_target_task_names.insert(target.task_name);
      signature.task_names.push_back(target.task_name);
      signature.target_poses.push_back(target.target_pose);
      if (continuity_reference_pose.has_value()) {
        signature.reference_target_poses.push_back(
            canonicalize_position_step_signature_pose(
                target.task_name, target.target_pose,
                continuity_reference_pose));
      }
      signature.gains.push_back(target.position_gain);
      signature.gains.push_back(target.orientation_gain);
      signature.gains.push_back(target.priority_position_tolerance);
      signature.gains.push_back(target.priority_orientation_tolerance);
      if (target.has_secondary_target_pose) {
        signature.task_names.push_back(target.task_name + "#secondary");
        signature.target_poses.push_back(target.secondary_target_pose);
        if (continuity_reference_pose.has_value()) {
          signature.reference_target_poses.push_back(
              canonicalize_position_step_signature_pose(
                  target.task_name, target.secondary_target_pose,
                  continuity_reference_pose));
        }
      }
    }
    signature.task_names.push_back("#continuity:" +
                                   options.continuity_reference_frame);
    signature.task_names.push_back("#torso:" +
                                   options.torso_constraint.frame_name);
    for (const auto &task : tasks_) {
      if (!task) {
        continue;
      }
      signature.task_names.push_back("#policy:" + task->getName());
      append_task_policy_signature(
          signature.gains, *task,
          explicit_target_task_names.find(task->getName()) ==
              explicit_target_task_names.end());
    }
    append_position_step_option_signature(signature.gains, options);
    update_position_step_target_motion_blocks(signature);
    position_step_target_geometry_moved_ =
        position_step_target_geometry_changed(signature);
    position_step_target_motion_observed_ =
        position_step_target_motion_observed_ ||
        (had_previous_target_signature &&
         position_step_target_geometry_moved_);
    if (update_position_step_target_signature(std::move(signature))) {
      capture_position_step_collision_command_floor(current_q);
    }
  }

  if (!options.preferred_locked_joint_indices.empty() &&
      !suppress_preferred_lock_step_retry_) {
    return solve_position_step_with_preferred_lock(current_q, targets, options);
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

  for (const auto &rt : resolved) {
    if (!suppress_min_error_step_retry_) {
      apply_position_step_primary_task_options(options, rt.task.get());
    }
  }

  int highest_active_target_priority = std::numeric_limits<int>::max();
  for (const auto &rt : resolved) {
    if (rt.task->isActive()) {
      highest_active_target_priority =
          std::min(highest_active_target_priority, rt.task->getPriority());
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
  // Keep the frozen set visible to collision computations that run after the
  // inner solve_velocity() clears pending_velocity_lock_indices_ (stall escape,
  // post-step validation), so the collision debug list and QP slots agree on
  // which pairs are controllable. Cleared on every exit path.
  position_step_locked_indices_ = step_locked_indices;
  struct ClearPositionStepLocks {
    KinematicsSolver *solver;
    ~ClearPositionStepLocks() {
      if (solver != nullptr) {
        solver->position_step_locked_indices_.clear();
      }
    }
  } clear_position_step_locks{this};
  (void)clear_position_step_locks;

  VelocitySolverResult last_vel_result;
  bool have_vel_result = false;
  Eigen::VectorXd first_tick_velocity;
  const int steps = std::max(1, options.max_steps);
  int steps_used = 0;
  Eigen::Matrix<double, 6, 1> vel;
  double prev_combined_error = std::numeric_limits<double>::infinity();
  double initial_primary_combined_error =
      std::numeric_limits<double>::quiet_NaN();
  std::vector<double> initial_target_merits;
  std::vector<double> initial_target_block_merits;
  int no_progress_count = 0;
  bool no_progress_exit = false;
  bool collision_violated_flag_mts = false;  // multi-target solve_position_step
  bool recovery_inside_collision_margin = false;
  bool configuration_step_limited = false;
  bool outer_acceleration_limit_applied = false;
  bool priority_candidate_rejected = false;

  int max_active_target_priority = std::numeric_limits<int>::min();
  for (const auto &rt : resolved) {
    if (rt.task->isActive()) {
      max_active_target_priority =
          std::max(max_active_target_priority, rt.task->getPriority());
    }
  }
  std::vector<int> protected_target_priorities;
  for (const auto &rt : resolved) {
    if (rt.task->isActive() &&
        rt.task->getPriority() < max_active_target_priority) {
      protected_target_priorities.push_back(rt.task->getPriority());
    }
  }
  std::sort(protected_target_priorities.begin(),
            protected_target_priorities.end());
  protected_target_priorities.erase(
      std::unique(protected_target_priorities.begin(),
                  protected_target_priorities.end()),
      protected_target_priorities.end());

  const auto build_priority_constraint_specs = [&](double step_dt_eff) {
    std::vector<PositionStepPriorityConstraintSpec> specs;
    if (!acceleration_limits_enabled_ ||
        acceleration_limits_.size() != robot_->nv()) {
      return specs;
    }
    specs.reserve(resolved.size());
    for (std::size_t index = 0; index < resolved.size(); ++index) {
      const auto &resolved_target = resolved[index];
      if (!resolved_target.task->isActive() ||
          !std::binary_search(protected_target_priorities.begin(),
                              protected_target_priorities.end(),
                              resolved_target.task->getPriority())) {
        continue;
      }
      const auto &target = targets[index];
      const bool protect_position =
          std::isfinite(target.priority_position_tolerance) &&
          target.priority_position_tolerance > 0.0;
      const bool protect_orientation =
          std::isfinite(target.priority_orientation_tolerance) &&
          target.priority_orientation_tolerance > 0.0;
      if (!protect_position && !protect_orientation) {
        continue;
      }
      specs.push_back({resolved_target.task,
                       protect_position ? target.priority_position_tolerance
                                        : -1.0,
                       protect_orientation
                           ? target.priority_orientation_tolerance
                           : -1.0,
                       step_dt_eff});
    }
    return specs;
  };

  const auto target_block_merits_at = [&](const Eigen::VectorXd &q_eval) {
    robot_->update_configuration(q_eval);
    std::vector<std::array<double, 2>> merits;
    merits.reserve(resolved.size());
    for (std::size_t index = 0; index < resolved.size(); ++index) {
      const auto &resolved_target = resolved[index];
      resolved_target.task->update(*robot_);
      TaskType merit_task_type = TaskType::FRAME_POSE;
      if (resolved_target.kind == PoseTaskKind::kFrame) {
        merit_task_type =
            static_cast<const FrameTask *>(resolved_target.task.get())
                ->getType();
      }
      merits.push_back(commanded_frame_block_merits(
          resolved_target.task->getError(), merit_task_type,
          targets[index].position_gain, targets[index].orientation_gain));
    }
    return merits;
  };

  const auto priority_candidate_acceptable =
      [&](const std::vector<std::array<double, 2>> &baseline_merits,
          const std::vector<std::array<double, 2>> &candidate_merits) {
        if (baseline_merits.size() != resolved.size() ||
            candidate_merits.size() != resolved.size()) {
          return false;
        }
        for (int priority : protected_target_priorities) {
          for (int block = 0; block < 2; ++block) {
            double acceptable_total = 0.0;
            double candidate_total = 0.0;
            bool saw_task = false;
            for (std::size_t index = 0; index < resolved.size(); ++index) {
              if (!resolved[index].task->isActive() ||
                  resolved[index].task->getPriority() != priority) {
                continue;
              }
              saw_task = true;
              const double baseline = baseline_merits[index][block];
              const double candidate = candidate_merits[index][block];
              const double configured_tolerance =
                  block == 0 ? targets[index].priority_position_tolerance
                             : targets[index].priority_orientation_tolerance;
              const bool has_configured_tolerance =
                  std::isfinite(configured_tolerance) &&
                  configured_tolerance > 0.0;
              const double protected_tolerance =
                  has_configured_tolerance
                      ? configured_tolerance
                      : kPositionStepSatisfiedMeritTolerance;
              if (!std::isfinite(baseline) || !std::isfinite(candidate)) {
                return false;
              }
              acceptable_total += std::max(baseline, protected_tolerance);
              candidate_total += candidate;
            }
            if (saw_task && candidate_total > acceptable_total + 1e-9) {
              return false;
            }
          }
        }
        return true;
      };

  const auto position_step_priority_scale =
      [&](const Eigen::VectorXd &q_before,
          const Eigen::VectorXd &q_after) {
        if (protected_target_priorities.empty()) {
          return 1.0;
        }
        const Eigen::VectorXd delta =
            pinocchio::difference(robot_->model(), q_before, q_after);
        if (!delta.allFinite() ||
            delta.squaredNorm() <= kCollisionEscapeNormEps) {
          robot_->update_configuration(q_after);
          return 1.0;
        }
        const auto baseline_merits = target_block_merits_at(q_before);
        const auto candidate_merits = target_block_merits_at(q_after);
        if (priority_candidate_acceptable(baseline_merits,
                                          candidate_merits)) {
          robot_->update_configuration(q_after);
          return 1.0;
        }

        double accepted_fraction = 0.0;
        double rejected_fraction = 1.0;
        for (int iteration = 0;
             iteration < kPositionStepPriorityBacktrackIterations;
             ++iteration) {
          const double fraction =
              0.5 * (accepted_fraction + rejected_fraction);
          Eigen::VectorXd candidate_q = pinocchio::integrate(
              robot_->model(), q_before, fraction * delta);
          const auto backoff_merits = target_block_merits_at(candidate_q);
          if (priority_candidate_acceptable(baseline_merits,
                                            backoff_merits)) {
            accepted_fraction = fraction;
          } else {
            rejected_fraction = fraction;
          }
        }
        robot_->update_configuration(q_after);
        return accepted_fraction;
      };

  const bool direct_priority_backtrack_is_safe =
      !acceleration_limits_enabled_ && !step_torso_constraint.has_value() &&
      !(collision_constraint_.has_value() &&
        collision_constraint_->enabled) &&
      !(com_constraint_.has_value() && com_constraint_->enabled) &&
      !(relative_pose_constraint_.has_value() &&
        relative_pose_constraint_->enabled) &&
      get_linear_velocity_constraint_rows() == 0 &&
      tight_frame_pose_constraints_.empty() &&
      tight_point_constraints_.empty() && !has_contact_frames();

  for (int step = 0; step < steps; ++step) {
    double combined_error = 0.0;
    double commanded_error = 0.0;
    std::vector<double> current_target_merits;
    current_target_merits.reserve(n_targets);
    std::vector<double> current_target_block_merits;
    current_target_block_merits.reserve(2 * n_targets);
    double max_position_error = 0.0;
    double primary_combined_error = std::numeric_limits<double>::quiet_NaN();
    std::vector<Eigen::VectorXd> target_velocities(n_targets);
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
        if (target.has_secondary_target_pose) {
          static_cast<AbsoluteFrameTask *>(rt.task.get())
              ->set_target_from_arm_targets(target.target_pose,
                                            target.secondary_target_pose);
        } else {
          static_cast<AbsoluteFrameTask *>(rt.task.get())
              ->setTargetPose(target.target_pose.block<3, 1>(0, 3),
                              target.target_pose.block<3, 3>(0, 0));
        }
        break;
      case PoseTaskKind::kRelative:
        static_cast<RelativeFrameTask *>(rt.task.get())
            ->setTargetPose(target.target_pose.block<3, 1>(0, 3),
                            target.target_pose.block<3, 3>(0, 0));
        break;
      }

      rt.task->update(*robot_);
      const Eigen::VectorXd &error = rt.task->getError();
      TaskType merit_task_type = TaskType::FRAME_POSE;
      if (rt.kind == PoseTaskKind::kFrame) {
        merit_task_type =
            static_cast<const FrameTask *>(rt.task.get())->getType();
      }
      const auto block_merits = commanded_frame_block_merits(
          error, merit_task_type, target.position_gain,
          target.orientation_gain);
      const double target_merit = block_merits[0] + block_merits[1];
      commanded_error += target_merit;
      current_target_merits.push_back(target_merit);
      current_target_block_merits.push_back(block_merits[0]);
      current_target_block_merits.push_back(block_merits[1]);
      if (error.size() == 3) {
        bool is_orientation_only = false;
        if (rt.kind == PoseTaskKind::kFrame) {
          const auto *ft = static_cast<const FrameTask *>(rt.task.get());
          is_orientation_only = ft->getType() == TaskType::FRAME_ORIENTATION;
        }
        Eigen::VectorXd v3 = (is_orientation_only ? target.orientation_gain
                                                  : target.position_gain) *
                             error.head<3>();
        if (is_orientation_only) {
          if (options.max_angular_speed > 0.0) {
            const double n = v3.norm();
            if (n > options.max_angular_speed) {
              v3 *= options.max_angular_speed / n;
            }
          }
        } else {
          if (options.max_linear_speed > 0.0) {
            const double n = v3.norm();
            if (n > options.max_linear_speed) {
              v3 *= options.max_linear_speed / n;
            }
          }
        }
        const double task_position_error = error.head<3>().norm();
        if (rt.task->isActive() &&
            rt.task->getPriority() == highest_active_target_priority) {
          primary_combined_error =
              std::isfinite(primary_combined_error)
                  ? std::max(primary_combined_error, task_position_error)
                  : task_position_error;
        }
        combined_error += task_position_error;
        if (!is_orientation_only) {
          max_position_error =
              std::max(max_position_error, task_position_error);
        }
        target_velocities[i] = v3;
      } else if (error.size() >= 6) {
        const double task_position_error = error.head<3>().norm();
        const double task_combined_error =
            task_position_error + error.tail<3>().norm();
        if (rt.task->isActive() &&
            rt.task->getPriority() == highest_active_target_priority) {
          primary_combined_error =
              std::isfinite(primary_combined_error)
                  ? std::max(primary_combined_error, task_combined_error)
                  : task_combined_error;
        }
        combined_error += task_combined_error;
        max_position_error = std::max(max_position_error, task_position_error);
        vel.head<3>() = target.position_gain * error.head<3>();
        vel.tail<3>() = target.orientation_gain * error.tail<3>();
        clamp_spatial_velocity_components(vel, options.max_linear_speed,
                                          options.max_angular_speed);
        target_velocities[i] = vel;
      } else {
        task_apply_failed = true;
        task_apply_error =
            "task '" + target.task_name + "' has invalid pose error dimension";
        break;
      }
    }

    if (task_apply_failed) {
      result.status = SolverStatus::kInvalidInput;
      result.status_message = task_apply_error;
      break;
    }
    if (step == 0) {
      initial_primary_combined_error = primary_combined_error;
      initial_target_merits = current_target_merits;
      initial_target_block_merits = current_target_block_merits;
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

    for (size_t i = 0; i < n_targets; ++i) {
      resolved[i].task->setPositionStepTargetVelocity(target_velocities[i]);
    }

    const double step_dt_eff = compute_adaptive_position_step_dt(
        options, step_dt, max_position_error,
        collision_constraint_.has_value() && collision_constraint_->enabled,
        last_constraint_min_recovery_margin_);
    const auto priority_constraint_specs =
        build_priority_constraint_specs(step_dt_eff);
    pending_velocity_lock_indices_ = step_locked_indices;
    pending_step_torso_constraint_ = step_torso_constraint;
    pending_step_validation_dt_ = step_dt_eff;
    pending_position_step_priority_constraints_ = priority_constraint_specs;
    pending_reuse_current_kinematics_ = true;
    const bool apply_setpoint_acceleration_limits =
        acceleration_limits_enabled_ && !position_step_target_geometry_moved_ &&
        step == 0;
    pending_position_step_acceleration_limits_ =
        apply_setpoint_acceleration_limits;
    VelocitySolverResult vel_out = solve_velocity(q, true);
    vel_out = retry_auto_task_layout_as_split_if_needed(
        q, std::move(vel_out), step_locked_indices, step_torso_constraint,
        step_dt_eff, priority_constraint_specs,
        apply_setpoint_acceleration_limits);
    const bool allow_backtrack =
        options.stall_recovery && vel_out.status != SolverStatus::kSuccess &&
        (vel_out.status == SolverStatus::kInfeasible ||
         vel_out.status == SolverStatus::kNumericalError ||
         vel_out.status == SolverStatus::kNoProgress) &&
        vel_out.joint_velocities.size() == robot_->nv() &&
        (!vel_out.joint_velocities.allFinite() ||
         vel_out.joint_velocities.norm() <= stall_config_.dq_stall_eps);
    if (allow_backtrack) {
      for (size_t i = 0; i < n_targets; ++i) {
        resolved[i].task->setPositionStepTargetVelocity(
            kPositionStepBacktrackGainScale * target_velocities[i]);
      }
      pending_velocity_lock_indices_ = step_locked_indices;
      pending_step_torso_constraint_ = step_torso_constraint;
      pending_step_validation_dt_ = step_dt_eff;
      pending_position_step_priority_constraints_ = priority_constraint_specs;
      pending_reuse_current_kinematics_ = true;
      pending_position_step_acceleration_limits_ =
          apply_setpoint_acceleration_limits;
      VelocitySolverResult retry = solve_velocity(q, true);
      if (retry.status == SolverStatus::kSuccess) {
        vel_out = std::move(retry);
      }
    }
    have_vel_result = true;
    last_vel_result = std::move(vel_out);
    ++steps_used;

    if (last_vel_result.status != SolverStatus::kSuccess &&
        last_vel_result.status != SolverStatus::kInfeasible &&
        last_vel_result.status != SolverStatus::kNumericalError) {
      break;
    }

    Eigen::VectorXd q_pre_step = q;
    bool step_collision_rejected = false;
    const bool adaptive_step_large = (step_dt_eff > step_dt * 1.01);
    const auto integrate_velocity_candidate =
        [&](VelocitySolverResult &velocity_result, double integration_dt) {
          if (!options.integration_zero_velocity_indices.empty() &&
              velocity_result.joint_velocities.size() == robot_->nv()) {
            apply_integration_velocity_mask(
                velocity_result.joint_velocities,
                options.integration_zero_velocity_indices);
          }
          if (options.limit_change_from_seed &&
              velocity_result.joint_velocities.size() == robot_->nv()) {
            Eigen::VectorXd corridor_lower = Eigen::VectorXd::Constant(
                robot_->nv(), -kUnboundedConstraintLimit);
            Eigen::VectorXd corridor_upper = Eigen::VectorXd::Constant(
                robot_->nv(), kUnboundedConstraintLimit);
            tighten_bounds_with_reference_corridor(
                corridor_lower, corridor_upper, q_reference, q_pre_step,
                vel_limits, step_dt, velocity_to_config_index, robot_->nv());
            clamp_joint_velocity_solution_in_place(
                velocity_result.joint_velocities, corridor_lower,
                corridor_upper, false, robot_->nv());
          }
          if (use_position_limits_ &&
              velocity_result.joint_velocities.size() == robot_->nv()) {
            auto [q_min, q_max] = robot_->get_joint_limits();
            if (elastic_band_config_.enabled &&
                elastic_band_state_.delta.size() == robot_->nv()) {
              expand_scalar_joint_limits_for_velocity_delta(
                  q_min, q_max, elastic_band_state_.delta,
                  velocity_to_config_index, robot_->nv());
            }
            clamp_joint_velocity_to_position_limits_for_integration(
                velocity_result.joint_velocities, q_pre_step, q_min, q_max,
                velocity_to_config_index, integration_dt, robot_->nv());
          }
          Eigen::VectorXd candidate = pinocchio::integrate(
              robot_->model(), q_pre_step,
              integration_dt * velocity_result.joint_velocities);
          if (use_position_limits_) {
            auto [q_min_true, q_max_true] = robot_->get_joint_limits();
            project_scalar_configuration_to_true_joint_limits(
                candidate, q_min_true, q_max_true, velocity_to_config_index,
                robot_->nv());
          }
          const bool candidate_configuration_limited =
              clamp_configuration_delta_from_reference(
                  robot_->model(), q_reference, candidate,
                  options.max_configuration_step_norm);
          configuration_step_limited =
              candidate_configuration_limited || configuration_step_limited;
          robot_->update_configuration(candidate);
          return candidate;
        };

    q = integrate_velocity_candidate(last_vel_result, step_dt_eff);
    double priority_scale = position_step_priority_scale(q_pre_step, q);
    if (priority_scale < 1.0) {
      bool accepted_priority_retry = false;
      if (direct_priority_backtrack_is_safe && priority_scale > 0.0) {
        const Eigen::VectorXd nominal_delta =
            pinocchio::difference(robot_->model(), q_pre_step, q);
        Eigen::VectorXd backtracked_q = pinocchio::integrate(
            robot_->model(), q_pre_step, priority_scale * nominal_delta);
        if (position_step_priority_scale(q_pre_step, backtracked_q) >= 1.0) {
          q = std::move(backtracked_q);
          last_vel_result.joint_velocities =
              pinocchio::difference(robot_->model(), q_pre_step, q) /
              std::max(step_dt_eff, 1e-9);
          last_vel_result.solution.assign(
              last_vel_result.joint_velocities.data(),
              last_vel_result.joint_velocities.data() +
                  last_vel_result.joint_velocities.size());
          accepted_priority_retry = true;
        }
      }
      double retry_dt =
          step_dt_eff * (priority_scale > 0.0 ? priority_scale : 0.5);
      for (int retry_index = 0;
           !direct_priority_backtrack_is_safe &&
           !accepted_priority_retry &&
           retry_index < kPositionStepPriorityRetryIterations;
           ++retry_index) {
        for (std::size_t index = 0; index < resolved.size(); ++index) {
          resolved[index].task->setPositionStepTargetVelocity(
              target_velocities[index]);
        }
        robot_->update_configuration(q_pre_step);
        const auto retry_priority_constraint_specs =
            build_priority_constraint_specs(retry_dt);
        pending_velocity_lock_indices_ = step_locked_indices;
        pending_step_torso_constraint_ = step_torso_constraint;
        pending_step_validation_dt_ = retry_dt;
        pending_position_step_priority_constraints_ =
            retry_priority_constraint_specs;
        pending_reuse_current_kinematics_ = true;
        pending_position_step_acceleration_limits_ =
            apply_setpoint_acceleration_limits;
        VelocitySolverResult retry = solve_velocity(q_pre_step, true);
        retry = retry_auto_task_layout_as_split_if_needed(
            q_pre_step, std::move(retry), step_locked_indices,
            step_torso_constraint, retry_dt, retry_priority_constraint_specs,
            apply_setpoint_acceleration_limits);
        const bool retry_has_candidate =
            (retry.status == SolverStatus::kSuccess ||
             retry.status == SolverStatus::kInfeasible ||
             retry.status == SolverStatus::kNumericalError) &&
            retry.joint_velocities.size() == robot_->nv() &&
            retry.joint_velocities.allFinite();
        if (!retry_has_candidate) {
          retry_dt *= 0.5;
          continue;
        }
        Eigen::VectorXd retry_q =
            integrate_velocity_candidate(retry, retry_dt);
        const double retry_scale =
            position_step_priority_scale(q_pre_step, retry_q);
        if (retry_scale >= 1.0) {
          last_vel_result = std::move(retry);
          q = std::move(retry_q);
          accepted_priority_retry = true;
          break;
        }
        retry_dt *= retry_scale > 0.0 ? retry_scale : 0.5;
      }
      if (!accepted_priority_retry) {
        q = q_pre_step;
        robot_->update_configuration(q);
        priority_candidate_rejected = true;
        if (last_vel_result.joint_velocities.size() == robot_->nv()) {
          last_vel_result.joint_velocities.setZero();
        }
        break;
      }
    }
    if (step == 0 &&
        last_vel_result.joint_velocities.size() == robot_->nv() &&
        last_vel_result.joint_velocities.allFinite()) {
      first_tick_velocity = last_vel_result.joint_velocities;
    }

    // Skip expensive post-step checks when integration produced no motion.
    const bool step_moved =
        (q - q_pre_step).squaredNorm() > kCollisionEscapeNormEps;

    auto eval_broadphase_expanded = [&](const Eigen::VectorXd &q_eval)
        -> std::optional<double> {
      evaluate_post_step_collision_recovery_margins(q_eval);
      return evaluate_post_step_collision_distance_from_current_results();
    };

    // Post-solve collision rejection: use min_distance as the violation
    // threshold (mirrors single-target overload fix).
    if (step_moved && collision_constraint_.has_value() &&
        collision_constraint_->enabled) {
      // Keep the legacy global guard for penetration diagnostics while the
      // recovery margin below enforces every pair's effective floor.
      const double violation_threshold = collision_constraint_->min_distance;
      const auto pre_dist_debug =
          evaluate_post_step_collision_distance_mutating(q_pre_step);
      const auto pre_recovery_margins =
          evaluate_post_step_collision_recovery_margins(q_pre_step);
      auto post_dist_debug =
          adaptive_step_large ? eval_broadphase_expanded(q)
                              : evaluate_post_step_collision_distance_mutating(q);
      const auto post_recovery_margins =
          evaluate_post_step_collision_recovery_margins(q);
      const double pre_dist =
          (pre_dist_debug.has_value() && std::isfinite(*pre_dist_debug))
              ? *pre_dist_debug
              : std::numeric_limits<double>::infinity();
      const double post_dist =
          (post_dist_debug.has_value() && std::isfinite(*post_dist_debug))
              ? *post_dist_debug
              : std::numeric_limits<double>::infinity();
      const bool recovery_floor_unacceptable =
          !collision_recovery_margins_acceptable(
              pre_recovery_margins, post_recovery_margins, 0.0);
      const bool recovery_floor_seed_was_safe =
          collision_recovery_margins_are_safe(pre_recovery_margins, 0.0);
      if (post_dist < violation_threshold ||
          recovery_floor_unacceptable) {
        bool seed_was_safe = (pre_dist >= violation_threshold);
        bool deepened =
            (pre_dist < violation_threshold &&
             post_dist <
                 pre_dist - kCollisionPenetrationWorsenTolerance);
        const bool hard_jump =
            std::isfinite(pre_dist) &&
            post_dist < kCollisionHardPenetrationRejectDistance &&
            post_dist < pre_dist - kCollisionHardWorsenTolerance;
        const bool pure_primary_min_error =
            !last_vel_result.task_modes_effective.empty() &&
            last_vel_result.task_modes_effective[0] == TaskSolveMode::kMinError &&
            (last_vel_result.task_used_fallback.empty() ||
             !last_vel_result.task_used_fallback[0]);
        const bool step_has_velocity_locks =
            !options.excluded_joint_indices.empty() ||
            !options.integration_zero_velocity_indices.empty();
        const bool enforce_material_penetration_guard =
            !pure_primary_min_error || step_has_velocity_locks;
        const bool post_penetrating =
            post_dist < kCollisionPenetrationDistanceThreshold;
        const bool crossed_into_penetration =
            enforce_material_penetration_guard && post_penetrating &&
            (!std::isfinite(pre_dist) ||
             pre_dist >= kCollisionPenetrationDistanceThreshold);
        const bool material_penetration_worsened =
            enforce_material_penetration_guard &&
            post_penetrating && std::isfinite(pre_dist) &&
            pre_dist < kCollisionPenetrationDistanceThreshold &&
            post_dist < pre_dist - kCollisionTolerance;
        if (seed_was_safe || deepened || hard_jump ||
            crossed_into_penetration || material_penetration_worsened ||
            recovery_floor_unacceptable) {
          double backoff_threshold;
          if (seed_was_safe) {
            backoff_threshold = violation_threshold;
          } else if (crossed_into_penetration) {
            backoff_threshold = kCollisionPenetrationDistanceThreshold;
          } else if (material_penetration_worsened) {
            backoff_threshold = pre_dist - kCollisionTolerance;
          } else if (hard_jump) {
            backoff_threshold = kCollisionPenetrationDistanceThreshold;
          } else {
            backoff_threshold = pre_dist - kCollisionPenetrationWorsenTolerance;
          }
          bool accepted_backoff = false;
          const Eigen::VectorXd dq_nominal =
              pinocchio::difference(robot_->model(), q_pre_step, q);
          for (double frac : kCollisionRejectionBackoffFractions) {
            Eigen::VectorXd q_backoff = pinocchio::integrate(
                robot_->model(), q_pre_step, frac * dq_nominal);
            robot_->update_configuration(q_backoff);
            auto backoff_dist_debug =
                adaptive_step_large
                    ? eval_broadphase_expanded(q_backoff)
                    : evaluate_post_step_collision_distance_mutating(q_backoff);
            const auto backoff_recovery_margins =
                evaluate_post_step_collision_recovery_margins(q_backoff);
            const bool backoff_recovery_acceptable =
                collision_recovery_margins_acceptable(
                    pre_recovery_margins, backoff_recovery_margins, 0.0);
            if (backoff_dist_debug.has_value() &&
                std::isfinite(*backoff_dist_debug) &&
                *backoff_dist_debug >= backoff_threshold &&
                backoff_recovery_acceptable &&
                position_step_priority_scale(q_pre_step, q_backoff) >=
                    1.0) {
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
            if (seed_was_safe || recovery_floor_seed_was_safe) {
              collision_violated_flag_mts = true;
              break;
            }
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
              auto esc_dist =
                  evaluate_post_step_collision_distance_mutating(q_candidate);
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
          auto esc_dist =
              evaluate_post_step_collision_distance_mutating(q_candidate);
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
    std::vector<int> desaturation_candidates = last_vel_result.saturated_joints;
    const bool saturated_zero_motion_plateau =
        combined_error > 1e-4 &&
        effective_step_dq_norm < stall_config_.dq_stall_eps &&
        !desaturation_candidates.empty() &&
        plateau_stall_steps >= kJointLimitDesaturationPlateauThreshold;
    const bool plateau_escape =
        (plateau_status || saturated_zero_motion_plateau) &&
        plateau_stall_steps >= kJointLimitDesaturationPlateauThreshold;
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
        auto curr_dist = evaluate_post_step_collision_distance_mutating(q);
        robot_->update_configuration(q_candidate);
        auto cand_dist =
            evaluate_post_step_collision_distance_mutating(q_candidate);
        const double desat_safe_threshold_mt =
            (collision_constraint_.has_value() && collision_constraint_->enabled)
            ? collision_constraint_->min_distance
            : kCollisionPenetrationDistanceThreshold;
        if (collision_recovery_candidate_acceptable(
                curr_dist, cand_dist, desat_safe_threshold_mt)) {
          q = q_candidate;
          result.stall_escape_count++;
          recovery_inside_collision_margin =
              recovery_inside_collision_margin ||
              (curr_dist.has_value() && std::isfinite(*curr_dist) &&
               *curr_dist < desat_safe_threshold_mt &&
               cand_dist.has_value() && std::isfinite(*cand_dist) &&
               *cand_dist < desat_safe_threshold_mt);
        } else {
          robot_->update_configuration(q);
        }
      }
    }
  }

  if (auto retry_result = attempt_min_error_position_step_retry(
          current_q, options, step_dt, have_vel_result, last_vel_result, q,
          [&resolved](std::vector<std::pair<Task *, TaskSolveMode>> *saved_modes) {
            return flip_primary_scale_tasks_to_min_error(resolved, saved_modes);
          },
          [&]() { return solve_position_step(current_q, targets, options); });
      retry_result.has_value()) {
    return retry_result.value();
  }

  if (position_step_target_geometry_moved_) {
    const Eigen::VectorXd terminal_q = q;
    robot_->update_configuration(current_q);
    bool all_target_blocks_commanded = true;
    double componentwise_prediction_weight = 1.0;
    for (size_t i = 0; i < n_targets; ++i) {
      const auto &target = targets[i];
      const auto &rt = resolved[i];
      rt.task->update(*robot_);
      const Eigen::VectorXd &current_error = rt.task->getError();
      TaskType command_task_type = TaskType::FRAME_POSE;
      if (rt.kind == PoseTaskKind::kFrame) {
        command_task_type =
            static_cast<const FrameTask *>(rt.task.get())->getType();
      }
      all_target_blocks_commanded =
          all_target_blocks_commanded &&
          all_frame_task_blocks_commanded(
              current_error, command_task_type, target.position_gain,
              target.orientation_gain);
      if (current_error.size() == 3) {
        bool is_orientation_only = false;
        if (rt.kind == PoseTaskKind::kFrame) {
          const auto *ft = static_cast<const FrameTask *>(rt.task.get());
          is_orientation_only =
              ft->getType() == TaskType::FRAME_ORIENTATION;
        }
        Eigen::VectorXd current_velocity =
            (is_orientation_only ? target.orientation_gain
                                 : target.position_gain) *
            current_error.head<3>();
        const double speed_limit = is_orientation_only
                                       ? options.max_angular_speed
                                       : options.max_linear_speed;
        if (speed_limit > 0.0 && current_velocity.norm() > speed_limit) {
          current_velocity *= speed_limit / current_velocity.norm();
        }
        rt.task->setPositionStepTargetVelocity(current_velocity);
      } else if (current_error.size() >= 6) {
        vel.head<3>() = target.position_gain * current_error.head<3>();
        vel.tail<3>() = target.orientation_gain * current_error.tail<3>();
        clamp_spatial_velocity_components(vel, options.max_linear_speed,
                                          options.max_angular_speed);
        rt.task->setPositionStepTargetVelocity(vel);
      }
      componentwise_prediction_weight =
          std::min(componentwise_prediction_weight,
                   position_step_task_terminal_prediction_weight(
                       *rt.task, current_error));
    }
    const auto componentwise_candidate =
        all_target_blocks_commanded && componentwise_prediction_weight > 0.0
            ? estimate_position_step_componentwise_outer_candidate(
                  current_q, previous_applied_velocity, options, step_dt,
                  terminal_q)
            : std::nullopt;
    const auto outer_priority_constraint_specs =
        build_priority_constraint_specs(step_dt);
    if (auto projected_result = apply_position_step_task_metric_projection(
            current_q, step_dt, first_tick_velocity, step_locked_indices,
            step_torso_constraint, outer_priority_constraint_specs, q);
        projected_result.has_value()) {
      last_vel_result = std::move(*projected_result);
      have_vel_result = true;
      if (outer_priority_constraint_specs.empty()) {
        outer_acceleration_limit_applied = true;
        if (componentwise_candidate.has_value() &&
            all_target_blocks_commanded &&
            componentwise_prediction_weight > 0.0) {
          const Eigen::VectorXd projected_q = q;
          const auto legacy_target_block_merits_at =
              [&](const Eigen::VectorXd &candidate_q) {
                robot_->update_configuration(candidate_q);
                std::vector<std::array<double, 2>> merits;
                merits.reserve(resolved.size());
                for (std::size_t index = 0; index < resolved.size(); ++index) {
                  const auto &resolved_target = resolved[index];
                  resolved_target.task->update(*robot_);
                  TaskType merit_task_type = TaskType::FRAME_POSE;
                  if (resolved_target.kind == PoseTaskKind::kFrame) {
                    merit_task_type =
                        static_cast<const FrameTask *>(resolved_target.task.get())
                            ->getType();
                  }
                  merits.push_back(commanded_frame_block_merits(
                      resolved_target.task->getError(), merit_task_type,
                      targets[index].position_gain,
                      targets[index].orientation_gain));
                }
                return merits;
              };
          const auto projected_merits = legacy_target_block_merits_at(q);
          const auto candidate_is_preferred =
              [&](const Eigen::VectorXd &candidate_q) {
                const auto candidate_merits =
                    legacy_target_block_merits_at(candidate_q);
                double projected_total = 0.0;
                double candidate_total = 0.0;
                bool all_targets_non_worsening =
                    projected_merits.size() == candidate_merits.size();
                for (std::size_t index = 0;
                     all_targets_non_worsening &&
                     index < projected_merits.size();
                     ++index) {
                  for (int block = 0; block < 2; ++block) {
                    all_targets_non_worsening =
                        all_targets_non_worsening &&
                        std::isfinite(projected_merits[index][block]) &&
                        std::isfinite(candidate_merits[index][block]) &&
                        candidate_merits[index][block] <=
                            projected_merits[index][block] + 1e-9;
                    projected_total += projected_merits[index][block];
                    candidate_total += candidate_merits[index][block];
                  }
                }
                return all_targets_non_worsening &&
                       std::isfinite(projected_total) &&
                       std::isfinite(candidate_total) &&
                       candidate_total + 1e-9 < projected_total;
              };
          Eigen::VectorXd safe_componentwise_candidate =
              *componentwise_candidate;
          if (componentwise_prediction_weight < 1.0) {
            const Eigen::VectorXd projected_delta = pinocchio::difference(
                robot_->model(), current_q, projected_q);
            const Eigen::VectorXd componentwise_delta = pinocchio::difference(
                robot_->model(), current_q, safe_componentwise_candidate);
            safe_componentwise_candidate = pinocchio::integrate(
                robot_->model(), current_q,
                projected_delta + componentwise_prediction_weight *
                                      (componentwise_delta - projected_delta));
          }
          if (candidate_is_preferred(safe_componentwise_candidate)) {
            apply_position_step_outer_acceleration_limit(
                current_q, previous_applied_velocity, options, step_dt,
                safe_componentwise_candidate);
            if (candidate_is_preferred(safe_componentwise_candidate)) {
              q = std::move(safe_componentwise_candidate);
              outer_acceleration_limit_applied = true;
            }
          }
          if (outer_acceleration_limit_applied) {
            last_vel_result.joint_velocities =
                pinocchio::difference(robot_->model(), current_q, q) / step_dt;
          }
          robot_->update_configuration(q);
        }
      } else {
        const Eigen::VectorXd projected_q = q;
        Eigen::VectorXd outer_validated_projected_q = projected_q;
        const bool projection_was_validated =
            apply_position_step_outer_acceleration_limit(
                current_q, previous_applied_velocity, options, step_dt,
                outer_validated_projected_q);
        const Eigen::VectorXd projection_adjustment = pinocchio::difference(
            robot_->model(), projected_q, outer_validated_projected_q);
        outer_acceleration_limit_applied =
            projection_was_validated && projection_adjustment.allFinite() &&
            projection_adjustment.norm() <=
                10.0 * constraint_tolerance_ * std::max(step_dt, 1e-9);
        robot_->update_configuration(projected_q);
        if (componentwise_candidate.has_value() &&
            all_target_blocks_commanded &&
            componentwise_prediction_weight > 0.0) {
          const auto current_merits = target_block_merits_at(current_q);
          const auto projected_merits = target_block_merits_at(q);
          const auto candidate_is_preferred =
              [&](const Eigen::VectorXd &candidate_q) {
                const auto candidate_merits = target_block_merits_at(candidate_q);
                if (!priority_candidate_acceptable(current_merits,
                                                    candidate_merits)) {
                  return false;
                }
                double projected_total = 0.0;
                double candidate_total = 0.0;
                bool lower_priority_merits_valid =
                    projected_merits.size() == candidate_merits.size();
                for (std::size_t index = 0;
                     lower_priority_merits_valid &&
                     index < projected_merits.size();
                     ++index) {
                  if (!resolved[index].task->isActive() ||
                      std::binary_search(
                          protected_target_priorities.begin(),
                          protected_target_priorities.end(),
                          resolved[index].task->getPriority())) {
                    continue;
                  }
                  for (int block = 0; block < 2; ++block) {
                    lower_priority_merits_valid =
                        lower_priority_merits_valid &&
                        std::isfinite(projected_merits[index][block]) &&
                        std::isfinite(candidate_merits[index][block]);
                    projected_total += projected_merits[index][block];
                    candidate_total += candidate_merits[index][block];
                  }
                }
                return lower_priority_merits_valid &&
                       std::isfinite(projected_total) &&
                       std::isfinite(candidate_total) &&
                       candidate_total + 1e-9 < projected_total;
              };
          Eigen::VectorXd safe_componentwise_candidate =
              *componentwise_candidate;
          if (componentwise_prediction_weight < 1.0) {
            const Eigen::VectorXd projected_delta = pinocchio::difference(
                robot_->model(), current_q, projected_q);
            const Eigen::VectorXd componentwise_delta = pinocchio::difference(
                robot_->model(), current_q, safe_componentwise_candidate);
            safe_componentwise_candidate = pinocchio::integrate(
                robot_->model(), current_q,
                projected_delta + componentwise_prediction_weight *
                                      (componentwise_delta - projected_delta));
          }
          if (candidate_is_preferred(safe_componentwise_candidate)) {
            apply_position_step_outer_acceleration_limit(
                current_q, previous_applied_velocity, options, step_dt,
                safe_componentwise_candidate);
            if (candidate_is_preferred(safe_componentwise_candidate)) {
              const Eigen::VectorXd componentwise_velocity =
                  pinocchio::difference(robot_->model(), current_q,
                                        safe_componentwise_candidate) /
                  std::max(step_dt, 1e-9);
              auto componentwise_projection =
                  apply_position_step_task_metric_projection(
                      current_q, step_dt, componentwise_velocity,
                      step_locked_indices, step_torso_constraint,
                      outer_priority_constraint_specs,
                      safe_componentwise_candidate);
              if (componentwise_projection.has_value() &&
                  componentwise_projection->joint_velocities.size() ==
                      robot_->nv() &&
                  componentwise_projection->joint_velocities.allFinite() &&
                  candidate_is_preferred(safe_componentwise_candidate)) {
                q = std::move(safe_componentwise_candidate);
                last_vel_result = std::move(*componentwise_projection);
                outer_acceleration_limit_applied = true;
              }
            }
          }
          if (outer_acceleration_limit_applied) {
            last_vel_result.joint_velocities =
                pinocchio::difference(robot_->model(), current_q, q) / step_dt;
          }
          robot_->update_configuration(q);
        }
      }
    }
  } else if (acceleration_limits_enabled_ &&
             first_tick_velocity.size() == robot_->nv() &&
             pinocchio::difference(robot_->model(), current_q, q)
                     .squaredNorm() > kCollisionEscapeNormEps) {
    q = pinocchio::integrate(robot_->model(), current_q,
                             step_dt * first_tick_velocity);
    if (use_position_limits_) {
      auto [q_min, q_max] = robot_->get_joint_limits();
      project_scalar_configuration_to_true_joint_limits(
          q, q_min, q_max, velocity_to_config_index, robot_->nv());
    }
    robot_->update_configuration(q);
  }

  const bool final_configuration_limited =
      clamp_configuration_delta_from_reference(
          robot_->model(), q_reference, q,
          options.max_configuration_step_norm);
  configuration_step_limited =
      final_configuration_limited || configuration_step_limited;
  if (final_configuration_limited) {
    robot_->update_configuration(q);
  }
  const bool final_outer_acceleration_limit_applied =
      apply_position_step_outer_acceleration_limit(
          current_q, previous_applied_velocity, options, step_dt, q);
  outer_acceleration_limit_applied =
      outer_acceleration_limit_applied ||
      final_outer_acceleration_limit_applied;
  double final_priority_scale = position_step_priority_scale(current_q, q);
  if (final_priority_scale < 1.0) {
    const Eigen::VectorXd terminal_delta =
        pinocchio::difference(robot_->model(), current_q, q);
    bool accepted_final_priority_retry = false;
    if (direct_priority_backtrack_is_safe && final_priority_scale > 0.0) {
      Eigen::VectorXd backtracked_q = pinocchio::integrate(
          robot_->model(), current_q,
          final_priority_scale * terminal_delta);
      if (position_step_priority_scale(current_q, backtracked_q) >= 1.0) {
        q = std::move(backtracked_q);
        last_vel_result.joint_velocities =
            pinocchio::difference(robot_->model(), current_q, q) /
            std::max(step_dt, 1e-9);
        last_vel_result.solution.assign(
            last_vel_result.joint_velocities.data(),
            last_vel_result.joint_velocities.data() +
                last_vel_result.joint_velocities.size());
        have_vel_result = true;
        accepted_final_priority_retry = true;
      }
    }
    double retry_fraction =
        final_priority_scale > 0.0 ? final_priority_scale : 0.5;
    for (int retry_index = 0;
         !direct_priority_backtrack_is_safe &&
         !accepted_final_priority_retry &&
         retry_index < kPositionStepPriorityRetryIterations;
         ++retry_index) {
      Eigen::VectorXd retry_q = pinocchio::integrate(
          robot_->model(), current_q, retry_fraction * terminal_delta);
      auto projected_result = apply_position_step_task_metric_projection(
          current_q, step_dt, Eigen::VectorXd(), step_locked_indices,
          step_torso_constraint, build_priority_constraint_specs(step_dt),
          retry_q);
      const bool retry_has_candidate =
          projected_result.has_value() &&
          projected_result->joint_velocities.size() == robot_->nv() &&
          projected_result->joint_velocities.allFinite();
      if (!retry_has_candidate) {
        retry_fraction *= 0.5;
        continue;
      }
      if (acceleration_limits_enabled_ &&
          !apply_position_step_outer_acceleration_limit(
              current_q, previous_applied_velocity, options, step_dt,
              retry_q)) {
        retry_fraction *= 0.5;
        continue;
      }

      bool collision_acceptable = true;
      if (collision_constraint_.has_value() &&
          collision_constraint_->enabled) {
        const auto current_dist =
            evaluate_post_step_collision_distance_mutating(current_q);
        const auto current_margins =
            evaluate_post_step_collision_recovery_margins(current_q);
        const auto retry_dist =
            evaluate_post_step_collision_distance_mutating(retry_q);
        const auto retry_margins =
            evaluate_post_step_collision_recovery_margins(retry_q);
        collision_acceptable =
            current_dist.has_value() && retry_dist.has_value() &&
            std::isfinite(*current_dist) && std::isfinite(*retry_dist) &&
            collision_recovery_margins_acceptable(
                current_margins, retry_margins, 0.0) &&
            ((*current_dist >= collision_constraint_->min_distance)
                 ? (*retry_dist >= collision_constraint_->min_distance -
                                      kCollisionTolerance)
                 : (*retry_dist >=
                    *current_dist - kCollisionPenetrationWorsenTolerance));
      }
      const double retry_priority_scale =
          position_step_priority_scale(current_q, retry_q);
      if (collision_acceptable && retry_priority_scale >= 1.0) {
        q = std::move(retry_q);
        projected_result->joint_velocities =
            pinocchio::difference(robot_->model(), current_q, q) /
            std::max(step_dt, 1e-9);
        last_vel_result = std::move(*projected_result);
        have_vel_result = true;
        outer_acceleration_limit_applied = true;
        accepted_final_priority_retry = true;
        break;
      }
      retry_fraction *=
          retry_priority_scale > 0.0 ? retry_priority_scale : 0.5;
    }
    if (!accepted_final_priority_retry) {
      q = current_q;
      priority_candidate_rejected = true;
    }
    robot_->update_configuration(q);
  }

  std::unordered_set<Task *> resolved_tasks;
  resolved_tasks.reserve(resolved.size());
  for (const auto &rt : resolved) {
    resolved_tasks.insert(rt.task.get());
    rt.task->clearPositionStepTargetVelocity();
  }
  for (auto &task : tasks_) {
    if (task && resolved_tasks.find(task.get()) == resolved_tasks.end()) {
      task->clearPositionStepTargetVelocity();
    }
  }

  result.q_solution = q;
  result.iterations_used = steps_used;

  const auto &primary = resolved.front();
  primary.task->update(*robot_);
  const Eigen::VectorXd &final_error = primary.task->getError();
  if (final_error.size() == 3) {
    result.position_error = final_error.head<3>().norm();
    result.orientation_error = 0.0;
  } else if (final_error.size() >= 6) {
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
    if (!runtime_config_.enable_auto_task_layout) {
      bool saw_pose = false;
      bool saw_split = false;
      for (const auto &rt : resolved) {
        if (rt.kind != PoseTaskKind::kFrame) {
          continue;
        }
        const auto *frame_task = static_cast<const FrameTask *>(rt.task.get());
        const TaskType type = frame_task->getType();
        saw_pose = saw_pose || type == TaskType::FRAME_POSE;
        saw_split = saw_split || type == TaskType::FRAME_POSITION ||
                                type == TaskType::FRAME_ORIENTATION;
      }
      result.active_task_layout =
          (saw_pose && !saw_split) ? TaskLayout::kMerged : TaskLayout::kSplit;
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
  if (collision_violated_flag_mts) {
    result.status = SolverStatus::kCollisionViolated;
    result.status_message =
        "solve_position_step (multi-target): no step could maintain collision "
        "min_distance; q_solution is the last safe configuration";
  }
  if (collision_constraint_.has_value() && collision_constraint_->enabled &&
      (result.stall_escape_count > 0 ||
       result.collision_rejection_count > 0 ||
       recovery_inside_collision_margin)) {
    auto final_dist = evaluate_min_collision_distance(q);
    if (final_dist.has_value() && std::isfinite(*final_dist) &&
        *final_dist < collision_constraint_->min_distance - kCollisionTolerance) {
      recovery_inside_collision_margin = true;
    }
  }
  if (recovery_inside_collision_margin &&
      result.status == SolverStatus::kSuccess) {
    result.status = SolverStatus::kNoProgress;
    result.status_message =
        "solve_position_step applied recovery motion inside collision margin";
  }
  if (!recovery_inside_collision_margin &&
      should_report_position_recovery_success(
          result, current_q,
          stall_config_.dq_stall_eps * std::max(step_dt, 1e-9))) {
    result.status = SolverStatus::kSuccess;
    result.status_message =
        "solve_position_step applied constraint recovery motion";
  }
  if (priority_candidate_rejected &&
      result.status != SolverStatus::kInvalidInput &&
      result.status != SolverStatus::kCollisionViolated) {
    const bool priority_hold =
        pinocchio::difference(robot_->model(), current_q, q).squaredNorm() <=
        kPositionStepMeritMotionThreshold;
    result.position_step_hold_active = priority_hold;
    result.status = priority_hold ? SolverStatus::kNoProgress
                                  : SolverStatus::kSuccess;
    result.status_message =
        priority_hold
            ? "solve_position_step held current configuration because no "
              "priority-safe candidate passed final constraint validation"
            : "solve_position_step decelerating before priority-safe hold";
  }
  double final_commanded_error = 0.0;
  double final_primary_combined_error =
      std::numeric_limits<double>::quiet_NaN();
  double final_primary_position_error = 0.0;
  double final_primary_orientation_error = 0.0;
  std::vector<double> final_target_merits;
  final_target_merits.reserve(resolved.size());
  std::vector<double> final_target_block_merits;
  final_target_block_merits.reserve(2 * resolved.size());
  for (std::size_t index = 0; index < resolved.size(); ++index) {
    const auto &resolved_target = resolved[index];
    resolved_target.task->update(*robot_);
    TaskType merit_task_type = TaskType::FRAME_POSE;
    if (resolved_target.kind == PoseTaskKind::kFrame) {
      merit_task_type = static_cast<const FrameTask *>(
                            resolved_target.task.get())
                            ->getType();
    }
    const auto block_merits = commanded_frame_block_merits(
        resolved_target.task->getError(), merit_task_type,
        targets[index].position_gain, targets[index].orientation_gain);
    const double target_merit = block_merits[0] + block_merits[1];
    final_commanded_error += target_merit;
    final_target_merits.push_back(target_merit);
    final_target_block_merits.push_back(block_merits[0]);
    final_target_block_merits.push_back(block_merits[1]);
    if (resolved_target.task->isActive() &&
        resolved_target.task->getPriority() ==
            highest_active_target_priority) {
      final_primary_combined_error =
          std::isfinite(final_primary_combined_error)
              ? std::max(final_primary_combined_error, target_merit)
              : target_merit;
      final_primary_position_error =
          std::max(final_primary_position_error, block_merits[0]);
      final_primary_orientation_error =
          std::max(final_primary_orientation_error, block_merits[1]);
    }
  }
  const double candidate_step_norm =
      result.q_solution.size() == current_q.size()
          ? pinocchio::difference(robot_->model(), current_q,
                                  result.q_solution)
                .norm()
          : std::numeric_limits<double>::quiet_NaN();
  int continuity_priority = highest_active_target_priority;
  std::vector<int> active_target_priorities;
  active_target_priorities.reserve(resolved.size());
  for (const auto &resolved_target : resolved) {
    if (resolved_target.task->isActive()) {
      active_target_priorities.push_back(resolved_target.task->getPriority());
    }
  }
  std::sort(active_target_priorities.begin(), active_target_priorities.end());
  active_target_priorities.erase(
      std::unique(active_target_priorities.begin(),
                  active_target_priorities.end()),
      active_target_priorities.end());
  for (int priority : active_target_priorities) {
    bool priority_unsatisfied = false;
    for (std::size_t index = 0; index < resolved.size(); ++index) {
      if (!resolved[index].task->isActive() ||
          resolved[index].task->getPriority() != priority) {
        continue;
      }
      priority_unsatisfied =
          priority_unsatisfied ||
          std::max(initial_target_merits[index], final_target_merits[index]) >
              kPositionStepSatisfiedMeritTolerance;
    }
    if (priority_unsatisfied) {
      continuity_priority = priority;
      break;
    }
  }
  double continuity_initial_merit = 0.0;
  double continuity_final_merit = 0.0;
  std::vector<double> continuity_initial_target_merits;
  std::vector<double> continuity_final_target_merits;
  std::vector<double> continuity_initial_block_merits;
  std::vector<double> continuity_final_block_merits;
  continuity_initial_block_merits.reserve(2 * resolved.size());
  continuity_final_block_merits.reserve(2 * resolved.size());
  for (std::size_t index = 0; index < resolved.size(); ++index) {
    if (!resolved[index].task->isActive() ||
        resolved[index].task->getPriority() != continuity_priority) {
      continue;
    }
    continuity_initial_merit += initial_target_merits[index];
    continuity_final_merit += final_target_merits[index];
    continuity_initial_target_merits.push_back(initial_target_merits[index]);
    continuity_final_target_merits.push_back(final_target_merits[index]);
    continuity_initial_block_merits.push_back(
        initial_target_block_merits[2 * index]);
    continuity_initial_block_merits.push_back(
        initial_target_block_merits[2 * index + 1]);
    continuity_final_block_merits.push_back(
        final_target_block_merits[2 * index]);
    continuity_final_block_merits.push_back(
        final_target_block_merits[2 * index + 1]);
  }
  const bool held_non_improving_step =
      !priority_candidate_rejected &&
      should_hold_position_step_for_continuity(
          result, current_q, continuity_initial_merit,
          continuity_final_merit, continuity_initial_target_merits,
          continuity_final_target_merits, continuity_initial_block_merits,
          continuity_final_block_merits, continuity_priority,
          candidate_step_norm,
          owns_position_step_continuity, collision_violated_flag_mts,
          step_torso_constraint.has_value() ||
              !build_priority_constraint_specs(step_dt).empty());
  if (held_non_improving_step) {
    const bool target_satisfied =
        continuity_initial_merit <= kPositionStepSatisfiedMeritTolerance;
    Eigen::VectorXd braking_q = current_q;
    const bool braking_to_hold = compute_position_step_continuity_brake(
        current_q, previous_applied_velocity, options, step_dt, braking_q);
    result.position_step_hold_active = !braking_to_hold;
    q = braking_to_hold ? std::move(braking_q) : current_q;
    robot_->update_configuration(q);
    result.q_solution = q;
    primary.task->update(*robot_);
    const Eigen::VectorXd &held_error = primary.task->getError();
    if (held_error.size() == 3) {
      result.position_error = held_error.head<3>().norm();
      result.orientation_error = 0.0;
    } else if (held_error.size() >= 6) {
      result.position_error = held_error.head<3>().norm();
      result.orientation_error = held_error.tail<3>().norm();
    }
    if (primary.kind == PoseTaskKind::kFrame) {
      result.achieved_pose = robot_->get_frame_pose(
          static_cast<FrameTask *>(primary.task.get())->getFrameName());
    } else {
      result.achieved_pose = targets.front().target_pose;
    }
    result.status = braking_to_hold
                        ? SolverStatus::kSuccess
                        : (target_satisfied ? SolverStatus::kSuccess
                                            : SolverStatus::kNoProgress);
    result.status_message =
        braking_to_hold
            ? "solve_position_step decelerating before continuity hold"
            : (target_satisfied
                   ? "solve_position_step held satisfied stationary target at "
                     "the current configuration"
                   : "solve_position_step held current configuration because "
                     "the nominal step did not reduce commanded task error");
    sync_position_result_applied_velocity(result, robot_->model(), current_q,
                                          step_dt);
    last_solution_dq_norm_ = result.joint_velocities.norm();
    if (!braking_to_hold && previous_dq_.size() == robot_->nv()) {
      previous_dq_.setZero();
    }
  }
  if (result.stall_escape_count > 0 || result.collision_rejection_count > 0 ||
      collision_violated_flag_mts || configuration_step_limited) {
    sync_position_result_applied_velocity(result, robot_->model(), current_q,
                                          step_dt);
  }
  if (!priority_candidate_rejected && !held_non_improving_step &&
      should_hold_soft_infeasible_position_step(
          result, current_q, initial_primary_combined_error,
          final_primary_combined_error, final_primary_position_error,
          final_primary_orientation_error,
          collision_violated_flag_mts, recovery_inside_collision_margin)) {
    Eigen::VectorXd braking_q = current_q;
    const bool braking_to_hold = compute_position_step_continuity_brake(
        current_q, previous_applied_velocity, options, step_dt, braking_q);
    result.position_step_hold_active = !braking_to_hold;
    q = braking_to_hold ? std::move(braking_q) : current_q;
    robot_->update_configuration(q);
    result.q_solution = q;
    primary.task->update(*robot_);
    const Eigen::VectorXd &held_error = primary.task->getError();
    if (held_error.size() == 3) {
      result.position_error = held_error.head<3>().norm();
      result.orientation_error = 0.0;
    } else if (held_error.size() >= 6) {
      result.position_error = held_error.head<3>().norm();
      result.orientation_error = held_error.tail<3>().norm();
    }
    if (primary.kind == PoseTaskKind::kFrame) {
      result.achieved_pose = robot_->get_frame_pose(
          static_cast<FrameTask *>(primary.task.get())->getFrameName());
    } else {
      result.achieved_pose = targets.front().target_pose;
    }
    result.status = braking_to_hold ? SolverStatus::kSuccess
                                    : SolverStatus::kNoProgress;
    result.status_message = braking_to_hold
                                ? "solve_position_step decelerating before "
                                  "soft-infeasible continuity hold"
                                : "solve_position_step held current configuration "
                                  "because the primary SCALE task was "
                                  "soft-infeasible";
    sync_position_result_applied_velocity(result, robot_->model(), current_q,
                                          step_dt);
    last_solution_dq_norm_ = result.joint_velocities.norm();
    if (!braking_to_hold && previous_dq_.size() == robot_->nv()) {
      previous_dq_.setZero();
    }
  }

  sync_position_result_applied_velocity(result, robot_->model(), current_q,
                                        step_dt);
  if (acceleration_limits_enabled_ &&
      result.joint_velocities.size() == robot_->nv()) {
    previous_dq_ = result.joint_velocities;
  }

  // In teleop-style loops (max_steps=1), stall handling must accumulate across
  // successive solve_position_step calls based on *applied* motion, not only
  // raw QP status.  This post-step update prevents success/no-progress
  // oscillations from resetting the consecutive stall counter.
  if (stall_handler_enabled()) {
    if (held_non_improving_step) {
      stall_state_.consecutive_stall_steps = 0;
    } else {
      const double applied_step_norm = (result.q_solution - current_q).norm();
      const double applied_step_eps =
          stall_config_.dq_stall_eps * std::max(step_dt, 1e-9);
      if (applied_step_norm < applied_step_eps) {
        stall_state_.consecutive_stall_steps++;
        stall_state_.total_stall_steps++;
      } else {
        stall_state_.consecutive_stall_steps = 0;
      }
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
