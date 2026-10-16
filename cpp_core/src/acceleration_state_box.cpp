#include "acceleration_state_box.hpp"
#include "acceleration_state_box_tolerance.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <utility>

namespace embodik::detail {
namespace {

constexpr double kFinalIntervalCanonicalizationTolerance = 1e-10;

double comparison_tolerance(double lhs, double rhs) {
  return state_box_comparison_tolerance(lhs, rhs);
}

bool bounds_are_finite(const ScalarBounds &bounds) {
  return std::isfinite(bounds.lower) && std::isfinite(bounds.upper);
}

bool active_bounds_are_ordered(const ScalarBounds &bounds) {
  return !bounds.lower_active || !bounds.upper_active ||
         bounds.lower <= bounds.upper;
}

bool input_is_finite(const StateBoxRowInput &input, double dt) {
  return std::isfinite(input.value) && std::isfinite(input.rate) &&
         std::isfinite(dt) && bounds_are_finite(input.state_bounds) &&
         bounds_are_finite(input.rate_bounds) &&
         bounds_are_finite(input.acceleration_bounds) &&
         std::isfinite(input.lower_braking_acceleration) &&
         std::isfinite(input.upper_braking_acceleration);
}

StateBoxRowResult failure(SolverStatus status, StateBoxFailure reason,
                          std::string message) {
  StateBoxRowResult result;
  result.status = status;
  result.failure = reason;
  result.message = std::move(message);
  return result;
}

bool tighten_lower(double candidate, StateBoxBoundCause cause,
                   StateBoxRowResult *result) {
  if (!std::isfinite(candidate)) {
    return false;
  }
  if (!result->lower_active) {
    result->lower_acceleration = candidate;
    result->lower_active = true;
    result->lower_causes = state_box_cause(cause);
    return true;
  }
  const double previous = result->lower_acceleration;
  const double tolerance = comparison_tolerance(candidate, previous);
  if (candidate > previous) {
    result->lower_acceleration = candidate;
    result->lower_causes = candidate <= previous + tolerance
                                ? result->lower_causes | state_box_cause(cause)
                                : state_box_cause(cause);
  } else if (std::abs(candidate - previous) <= tolerance) {
    result->lower_causes |= state_box_cause(cause);
  }
  return true;
}

bool tighten_upper(double candidate, StateBoxBoundCause cause,
                   StateBoxRowResult *result) {
  if (!std::isfinite(candidate)) {
    return false;
  }
  if (!result->upper_active) {
    result->upper_acceleration = candidate;
    result->upper_active = true;
    result->upper_causes = state_box_cause(cause);
    return true;
  }
  const double previous = result->upper_acceleration;
  const double tolerance = comparison_tolerance(candidate, previous);
  if (candidate < previous) {
    result->upper_acceleration = candidate;
    result->upper_causes = candidate >= previous - tolerance
                                ? result->upper_causes | state_box_cause(cause)
                                : state_box_cause(cause);
  } else if (std::abs(candidate - previous) <= tolerance) {
    result->upper_causes |= state_box_cause(cause);
  }
  return true;
}

bool braking_authority_is_consistent(const StateBoxRowInput &input) {
  if (input.state_bounds.lower_active) {
    if (!(input.lower_braking_acceleration > 0.0)) {
      return false;
    }
    if (input.acceleration_bounds.upper_active &&
        input.acceleration_bounds.upper +
                comparison_tolerance(input.acceleration_bounds.upper,
                                     input.lower_braking_acceleration) <
            input.lower_braking_acceleration) {
      return false;
    }
  }
  if (input.state_bounds.upper_active) {
    if (!(input.upper_braking_acceleration > 0.0)) {
      return false;
    }
    if (input.acceleration_bounds.lower_active &&
        -input.acceleration_bounds.lower +
                comparison_tolerance(-input.acceleration_bounds.lower,
                                     input.upper_braking_acceleration) <
            input.upper_braking_acceleration) {
      return false;
    }
  }
  return true;
}

bool continuous_acceleration_upper(double margin, double outward_rate,
                                   double dt, double *upper) {
  const double endpoint =
      2.0 * (margin - outward_rate * dt) / (dt * dt);
  if (!std::isfinite(endpoint)) {
    return false;
  }
  if (outward_rate <= 0.0 || 2.0 * margin >= outward_rate * dt) {
    *upper = endpoint;
    return true;
  }
  const double interior = -outward_rate * outward_rate / (2.0 * margin);
  if (!std::isfinite(interior)) {
    return false;
  }
  *upper = std::min(endpoint, interior);
  return true;
}

bool viability_acceleration_upper(double margin, double outward_rate,
                                  double braking_acceleration, double dt,
                                  double *upper) {
  const double dt_squared = dt * dt;
  const double linear =
      2.0 * outward_rate * dt + braking_acceleration * dt_squared;
  const double constant =
      outward_rate * outward_rate +
      2.0 * braking_acceleration * outward_rate * dt -
      2.0 * braking_acceleration * margin;
  const double reverse_rate_bound = -outward_rate / dt;
  if (!std::isfinite(linear) || !std::isfinite(constant) ||
      !std::isfinite(reverse_rate_bound)) {
    return false;
  }

  double discriminant = linear * linear - 4.0 * dt_squared * constant;
  const double tolerance =
      comparison_tolerance(linear * linear, 4.0 * dt_squared * constant);
  if (!std::isfinite(discriminant) || discriminant < -tolerance) {
    return false;
  }
  discriminant = std::max(0.0, discriminant);
  const double square_root = std::sqrt(discriminant);
  double upper_root = -linear / (2.0 * dt_squared);
  if (square_root > 0.0 || linear != 0.0) {
    const double stable_numerator =
        -0.5 * (linear + std::copysign(square_root, linear));
    const double first_root = stable_numerator / dt_squared;
    const double second_root =
        stable_numerator != 0.0 ? constant / stable_numerator : first_root;
    upper_root = std::max(first_root, second_root);
  }
  if (!std::isfinite(upper_root)) {
    return false;
  }
  *upper = std::max(reverse_rate_bound, upper_root);
  return std::isfinite(*upper);
}

double bounded_value(const StateBoxRowInput &input) {
  double value = input.value;
  if (input.state_bounds.lower_active) {
    value = std::max(value, input.state_bounds.lower);
  }
  if (input.state_bounds.upper_active) {
    value = std::min(value, input.state_bounds.upper);
  }
  return value;
}

bool is_outside_bound(double value, double bound) {
  return value > bound + comparison_tolerance(value, bound);
}

} // namespace

double canonicalize_state_box_acceleration_roundoff(double acceleration,
                                                   double lower,
                                                   double upper) {
  if (!std::isfinite(acceleration) || std::isnan(lower) ||
      std::isnan(upper) || lower > upper) {
    return acceleration;
  }
  const double face = acceleration < lower ? lower : upper;
  if ((acceleration >= lower && acceleration <= upper) ||
      !std::isfinite(face)) {
    return acceleration;
  }
  // The opposite face may be unbounded and must not inflate this allowance.
  const double tolerance =
      64.0 * std::numeric_limits<double>::epsilon() *
      std::max({1.0, std::abs(acceleration), std::abs(face)});
  return std::abs(acceleration - face) <= tolerance ? face : acceleration;
}

bool polish_state_box_acceleration(
    const Eigen::VectorXd &lower, const Eigen::VectorXd &upper,
    const Eigen::MatrixXd &hard_matrix, const Eigen::VectorXd &hard_lower,
    const Eigen::VectorXd &hard_upper,
    const std::vector<Eigen::MatrixXd> &objective_matrices,
    double box_tolerance, double normalized_hard_tolerance,
    double objective_tolerance, Eigen::VectorXd *acceleration, bool *changed) {
  if (changed != nullptr) {
    *changed = false;
  }
  if (acceleration == nullptr || acceleration->size() == 0 ||
      lower.size() != acceleration->size() || upper.size() != lower.size() ||
      !acceleration->allFinite() || lower.array().isNaN().any() ||
      upper.array().isNaN().any() ||
      (lower.array() > upper.array()).any() ||
      !std::isfinite(box_tolerance) || box_tolerance < 0.0 ||
      !std::isfinite(normalized_hard_tolerance) ||
      normalized_hard_tolerance < 0.0 ||
      !std::isfinite(objective_tolerance) || objective_tolerance < 0.0) {
    return false;
  }
  const double violation = std::max(
      0.0, std::max((lower - *acceleration).maxCoeff(),
                    (*acceleration - upper).maxCoeff()));
  if (violation == 0.0) {
    return true;
  }
  if (!std::isfinite(violation) || violation > box_tolerance ||
      hard_matrix.cols() != acceleration->size() ||
      hard_lower.size() != hard_matrix.rows() ||
      hard_upper.size() != hard_lower.size() || !hard_matrix.allFinite() ||
      hard_lower.array().isNaN().any() || hard_upper.array().isNaN().any() ||
      (hard_lower.array() > hard_upper.array()).any()) {
    return false;
  }
  const Eigen::VectorXd projected = acceleration->cwiseMax(lower).cwiseMin(upper);
  const Eigen::VectorXd physical_image = hard_matrix * projected;
  if (!projected.allFinite() || !physical_image.allFinite() ||
      (physical_image.array() < hard_lower.array() - box_tolerance).any() ||
      (physical_image.array() > hard_upper.array() + box_tolerance).any()) {
    return false;
  }
  // Match the generalized backend's original row normalization, including
  // unchanged zero rows; a physical-unit check alone loses its certificate.
  for (Eigen::Index row = 0; row < hard_matrix.rows(); ++row) {
    Eigen::RowVectorXd normalized = hard_matrix.row(row);
    const double row_norm = normalized.stableNorm();
    double row_lower = hard_lower(row);
    double row_upper = hard_upper(row);
    if (!std::isfinite(row_norm)) {
      return false;
    }
    if (row_norm > 0.0) {
      normalized /= row_norm;
      row_lower /= row_norm;
      row_upper /= row_norm;
    }
    const double value = normalized.dot(projected);
    if (!normalized.allFinite() || !std::isfinite(value) ||
        std::isnan(row_lower) || std::isnan(row_upper) ||
        value < row_lower - normalized_hard_tolerance ||
        value > row_upper + normalized_hard_tolerance) {
      return false;
    }
  }
  for (const auto &matrix : objective_matrices) {
    if (matrix.cols() != acceleration->size() || !matrix.allFinite()) {
      return false;
    }
    const Eigen::VectorXd original_image = matrix * *acceleration;
    const Eigen::VectorXd image = matrix * projected;
    const double original_norm = original_image.stableNorm();
    const double difference_norm = (image - original_image).stableNorm();
    const double tolerance = objective_tolerance * std::max(1.0, original_norm);
    if (!original_image.allFinite() || !image.allFinite() ||
        !std::isfinite(original_norm) || !std::isfinite(difference_norm) ||
        !std::isfinite(tolerance) || difference_norm > tolerance) {
      return false;
    }
  }
  *acceleration = projected;
  if (changed != nullptr) {
    *changed = true;
  }
  return true;
}

ReachableRecoveryRateResult compute_reachable_recovery_rate(
    double violation, double current_recovery_rate, double dt,
    double recovery_scale, double minimum_recovery_rate,
    double maximum_recovery_rate, double acceleration_authority,
    double boundary_epsilon, double authority_fraction) {
  if (!std::isfinite(violation) || !std::isfinite(current_recovery_rate) ||
      !std::isfinite(dt) || !std::isfinite(recovery_scale) ||
      !std::isfinite(minimum_recovery_rate) ||
      !std::isfinite(maximum_recovery_rate) ||
      !std::isfinite(acceleration_authority) ||
      !std::isfinite(boundary_epsilon) ||
      !std::isfinite(authority_fraction)) {
    return {SolverStatus::kNonFiniteInput,
            "recovery-rate inputs must be finite", 0.0};
  }
  if (violation < 0.0 || dt <= 0.0 || recovery_scale <= 0.0 ||
      minimum_recovery_rate <= 0.0 ||
      maximum_recovery_rate < minimum_recovery_rate ||
      acceleration_authority <= 0.0 || boundary_epsilon < 0.0 ||
      authority_fraction <= 0.0 || authority_fraction >= 1.0) {
    return {SolverStatus::kInvalidInput,
            "recovery-rate bounds or authority are invalid", 0.0};
  }
  const double requested_rate = std::min(
      maximum_recovery_rate,
      std::max(minimum_recovery_rate,
               recovery_scale *
                   std::max(0.0, violation - boundary_epsilon) / dt));
  const double reachable_rate =
      std::max(0.0, current_recovery_rate +
                        authority_fraction * acceleration_authority * dt);
  const double rate = std::min(requested_rate, reachable_rate);
  if (!std::isfinite(requested_rate) || !std::isfinite(reachable_rate) ||
      !std::isfinite(rate)) {
    return {SolverStatus::kNumericalError,
            "recovery-rate construction was non-finite", 0.0};
  }
  return {SolverStatus::kSuccess, {}, rate};
}

static StateBoxRowResult shape_state_box_row_nominal(
    const StateBoxRowInput &input, double dt) {
  if (!input_is_finite(input, dt)) {
    return failure(SolverStatus::kNonFiniteInput,
                   StateBoxFailure::kInvalidSpecification,
                   "state-box input must be finite");
  }
  if (dt <= 0.0 || !active_bounds_are_ordered(input.state_bounds) ||
      !active_bounds_are_ordered(input.rate_bounds) ||
      !active_bounds_are_ordered(input.acceleration_bounds) ||
      !braking_authority_is_consistent(input)) {
    return failure(SolverStatus::kInvalidInput,
                   StateBoxFailure::kInvalidSpecification,
                   "state-box bounds, dt, or braking authority are invalid");
  }

  StateBoxRowResult result;
  if (input.acceleration_bounds.lower_active) {
    result.lower_acceleration = input.acceleration_bounds.lower;
    result.lower_active = true;
    result.lower_causes = state_box_cause(StateBoxBoundCause::kAcceleration);
  }
  if (input.acceleration_bounds.upper_active) {
    result.upper_acceleration = input.acceleration_bounds.upper;
    result.upper_active = true;
    result.upper_causes = state_box_cause(StateBoxBoundCause::kAcceleration);
  }

  bool finite = true;
  // Fiore Eq. 11/12 shape state-aware rate bounds, then Eq. 14/15 convert the
  // compatible rate interval into an acceleration box. The endpoint,
  // continuous-path, and next-cycle viability rows conservatively intersect
  // that same box so accepted ddq remains compatible at the current tick.
  if (input.rate_bounds.lower_active) {
    finite &= tighten_lower((input.rate_bounds.lower - input.rate) / dt,
                            StateBoxBoundCause::kRate, &result);
  }
  if (input.rate_bounds.upper_active) {
    finite &= tighten_upper((input.rate_bounds.upper - input.rate) / dt,
                            StateBoxBoundCause::kRate, &result);
  }

  const double value = bounded_value(input);
  if (input.state_bounds.lower_active) {
    const double margin = value - input.state_bounds.lower;
    if (is_outside_bound(input.state_bounds.lower, input.value)) {
      return failure(SolverStatus::kInfeasible,
                     StateBoxFailure::kCurrentStateOutside,
                     "current state is below its lower bound");
    }
    if (margin <= 0.0 && input.rate < 0.0) {
      return failure(SolverStatus::kInfeasible,
                     StateBoxFailure::kOutwardRateAtBoundary,
                     "outward lower-bound rate at the boundary");
    }
    if (input.rate < 0.0 &&
        input.rate * input.rate >
            2.0 * input.lower_braking_acceleration * margin +
                comparison_tolerance(
                    input.rate * input.rate,
                    2.0 * input.lower_braking_acceleration * margin)) {
      return failure(SolverStatus::kInfeasible,
                     StateBoxFailure::kOutsideBrakingEnvelope,
                     "lower-bound braking envelope is already violated");
    }
    finite &= tighten_lower(
        (std::max((input.state_bounds.lower - value) / dt,
                  -std::sqrt(2.0 * input.lower_braking_acceleration *
                             margin)) -
         input.rate) /
            dt,
        StateBoxBoundCause::kStateRate, &result);
    finite &= tighten_lower((input.state_bounds.lower - value -
                             input.rate * dt) /
                                (0.5 * dt * dt),
                            StateBoxBoundCause::kStateEndpoint, &result);
    double continuous_upper = 0.0;
    finite &= continuous_acceleration_upper(
        margin, -input.rate, dt, &continuous_upper);
    finite &= tighten_lower(-continuous_upper,
                            StateBoxBoundCause::kContinuousPath, &result);
    double viability_upper = 0.0;
    finite &= viability_acceleration_upper(
        margin, -input.rate, input.lower_braking_acceleration, dt,
        &viability_upper);
    finite &= tighten_lower(-viability_upper,
                            StateBoxBoundCause::kNextStateViability, &result);
  }

  if (input.state_bounds.upper_active) {
    const double margin = input.state_bounds.upper - value;
    if (is_outside_bound(input.value, input.state_bounds.upper)) {
      return failure(SolverStatus::kInfeasible,
                     StateBoxFailure::kCurrentStateOutside,
                     "current state is above its upper bound");
    }
    if (margin <= 0.0 && input.rate > 0.0) {
      return failure(SolverStatus::kInfeasible,
                     StateBoxFailure::kOutwardRateAtBoundary,
                     "outward upper-bound rate at the boundary");
    }
    if (input.rate > 0.0 &&
        input.rate * input.rate >
            2.0 * input.upper_braking_acceleration * margin +
                comparison_tolerance(
                    input.rate * input.rate,
                    2.0 * input.upper_braking_acceleration * margin)) {
      return failure(SolverStatus::kInfeasible,
                     StateBoxFailure::kOutsideBrakingEnvelope,
                     "upper-bound braking envelope is already violated");
    }
    finite &= tighten_upper(
        (std::min((input.state_bounds.upper - value) / dt,
                  std::sqrt(2.0 * input.upper_braking_acceleration *
                            margin)) -
         input.rate) /
            dt,
        StateBoxBoundCause::kStateRate, &result);
    finite &= tighten_upper((input.state_bounds.upper - value -
                             input.rate * dt) /
                                (0.5 * dt * dt),
                            StateBoxBoundCause::kStateEndpoint, &result);
    double continuous_upper = 0.0;
    finite &= continuous_acceleration_upper(
        margin, input.rate, dt, &continuous_upper);
    finite &= tighten_upper(continuous_upper,
                            StateBoxBoundCause::kContinuousPath, &result);
    double viability_upper = 0.0;
    finite &= viability_acceleration_upper(
        margin, input.rate, input.upper_braking_acceleration, dt,
        &viability_upper);
    finite &= tighten_upper(viability_upper,
                            StateBoxBoundCause::kNextStateViability, &result);
  }

  if (!finite) {
    return failure(SolverStatus::kNumericalError,
                   StateBoxFailure::kNonFiniteDerivedValue,
                   "state-box derived acceleration bound was non-finite");
  }
  if (!result.lower_active || !result.upper_active) {
    return failure(SolverStatus::kInvalidInput,
                   StateBoxFailure::kInvalidSpecification,
                   "state-box row must produce both acceleration sides");
  }
  const double canonical_tolerance = std::max(
      kFinalIntervalCanonicalizationTolerance,
      comparison_tolerance(result.lower_acceleration,
                           result.upper_acceleration));
  if (result.lower_acceleration > result.upper_acceleration) {
    if (result.lower_acceleration - result.upper_acceleration <=
        canonical_tolerance) {
      const double midpoint =
          0.5 * (result.lower_acceleration + result.upper_acceleration);
      result.lower_acceleration = midpoint;
      result.upper_acceleration = midpoint;
      result.lower_causes |= result.upper_causes;
      result.upper_causes = result.lower_causes;
    } else {
      return failure(SolverStatus::kInfeasible,
                     StateBoxFailure::kEmptyInterval,
                     "state-box acceleration interval is empty");
    }
  }
  return result;
}


namespace {

bool strictly_within(double value, const ScalarBounds &bounds) {
  return std::isfinite(value) &&
         (!bounds.lower_active || value >= bounds.lower) &&
         (!bounds.upper_active || value <= bounds.upper);
}

// This is a numerical position-uncertainty policy, not exact-real feasibility.
// No dimensionless floor or opposite-face/range magnitude enters the budget.
bool position_within_arithmetic_budget(const StateBoxRowInput &input, double dt,
                                      double acceleration,
                                      double displacement, double point) {
  if (!std::isfinite(point) || !std::isfinite(displacement)) {
    return false;
  }
  const double velocity_term = std::abs(dt * input.rate);
  const double acceleration_term = std::abs(0.5 * dt * dt * acceleration);
  auto face_is_safe = [&](double face, bool lower) {
    const double scale = std::abs(input.value) + std::abs(face) + velocity_term +
                         acceleration_term + std::abs(displacement);
    const double budget = 64.0 * std::numeric_limits<double>::epsilon() * scale;
    const double excess = lower ? face - point : point - face;
    return std::isfinite(scale) && std::isfinite(budget) &&
           std::isfinite(excess) && excess <= budget;
  };
  return (!input.state_bounds.lower_active ||
          face_is_safe(input.state_bounds.lower, true)) &&
         (!input.state_bounds.upper_active ||
          face_is_safe(input.state_bounds.upper, false));
}

bool stopping_vertices_within_arithmetic_budget(const StateBoxRowInput &input,
                                               double dt) {
  if (input.rate == 0.0) {
    return strictly_within(input.value, input.state_bounds);
  }
  const bool toward_lower = input.rate < 0.0;
  if (!(toward_lower ? input.state_bounds.lower_active
                    : input.state_bounds.upper_active)) {
    return true;
  }
  if (!(toward_lower ? input.acceleration_bounds.upper_active
                    : input.acceleration_bounds.lower_active)) {
    return false;
  }
  const double acceleration = toward_lower ? input.acceleration_bounds.upper
                                           : input.acceleration_bounds.lower;
  auto stopping_vertex_is_safe = [&](double braking_acceleration) {
    const double stop_time = -input.rate / braking_acceleration;
    if (!std::isfinite(stop_time) || stop_time <= 0.0) {
      return false;
    }
    const double displacement =
        stop_time * input.rate +
        0.5 * stop_time * stop_time * braking_acceleration;
    return position_within_arithmetic_budget(
        input, dt, acceleration, displacement, input.value + displacement);
  };
  // A caller may promise less future braking authority than the physical cap.
  // Preserve that conservative policy as well as physical brake feasibility.
  const double declared_braking = toward_lower
                                      ? input.lower_braking_acceleration
                                      : -input.upper_braking_acceleration;
  return stopping_vertex_is_safe(acceleration) &&
         stopping_vertex_is_safe(declared_braking);
}

bool certify_numerical_braking_acceleration(
    const StateBoxRowInput &input, double dt, double acceleration,
    StateBoxRowResult *result) {
  // The original builder validates specification/finite inputs first. Repeat
  // these guards so the exception cannot bypass invalid authorities or shapes.
  if (!input_is_finite(input, dt) || dt <= 0.0 ||
      !active_bounds_are_ordered(input.state_bounds) ||
      !active_bounds_are_ordered(input.rate_bounds) ||
      !active_bounds_are_ordered(input.acceleration_bounds) ||
      !braking_authority_is_consistent(input) ||
      !strictly_within(input.value, input.state_bounds) ||
      !strictly_within(input.rate, input.rate_bounds) || input.rate == 0.0) {
    return false;
  }
  const bool toward_lower = input.rate < 0.0;
  if (!(toward_lower ? input.state_bounds.lower_active
                    : input.state_bounds.upper_active) ||
      !(toward_lower ? input.acceleration_bounds.upper_active
                    : input.acceleration_bounds.lower_active)) {
    return false;
  }
  if (!strictly_within(acceleration, input.acceleration_bounds)) {
    return false;
  }
  const double stop_time = -input.rate / acceleration;
  const double stopping_displacement =
      stop_time * input.rate + 0.5 * stop_time * stop_time * acceleration;
  const double next_rate = input.rate + dt * acceleration;
  const double next_value =
      input.value + (dt * input.rate + 0.5 * dt * dt * acceleration);
  if (!std::isfinite(stop_time) || stop_time <= 0.0 || stop_time > dt ||
      (toward_lower ? next_rate < 0.0 : next_rate > 0.0) ||
      !strictly_within(next_value, input.state_bounds) ||
      !strictly_within(next_rate, input.rate_bounds) ||
      !position_within_arithmetic_budget(
          input, dt, acceleration, stopping_displacement,
          input.value + stopping_displacement) ||
      !stopping_vertices_within_arithmetic_budget(input, dt)) {
    return false;
  }
  auto predicted = input;
  predicted.value = next_value;
  predicted.rate = next_rate;
  if (!stopping_vertices_within_arithmetic_budget(predicted, dt)) {
    return false;
  }
  // One nonrecursive nominal check makes re-entry independent of whether a
  // second numerical exception would be eligible on the following cycle.
  const auto predicted_box = shape_state_box_row_nominal(predicted, dt);
  if (predicted_box.status != SolverStatus::kSuccess ||
      (predicted.acceleration_bounds.lower_active &&
       predicted_box.upper_acceleration < predicted.acceleration_bounds.lower) ||
      (predicted.acceleration_bounds.upper_active &&
       predicted_box.lower_acceleration > predicted.acceleration_bounds.upper)) {
    return false;
  }
  result->status = SolverStatus::kSuccess;
  result->failure = StateBoxFailure::kNone;
  result->message = "numerical position-arithmetic braking singleton";
  result->lower_active = result->upper_active = true;
  result->lower_acceleration = result->upper_acceleration = acceleration;
  result->lower_causes = result->upper_causes =
      state_box_cause(StateBoxBoundCause::kAcceleration) |
      state_box_cause(StateBoxBoundCause::kContinuousPath) |
      state_box_cause(StateBoxBoundCause::kNextStateViability);
  return true;
}

bool numerical_braking_singleton(const StateBoxRowInput &input, double dt,
                                 StateBoxRowResult *result) {
  const double physical_acceleration = input.rate < 0.0
                                           ? input.acceleration_bounds.upper
                                           : input.acceleration_bounds.lower;
  // Try a scale-free, gentler single witness before the physical-cap fallback.
  // For q at a zero face, x = dt * abs(a) / abs(rate) >= 1/sqrt(e)
  // satisfies the same local stopping-position budget, e = 64 * epsilon.
  // The common certificate still checks the actual represented candidate and
  // all existing endpoint, rate, declared-braking, and next-box conditions.
  const double denominator =
      dt * std::sqrt(64.0 * std::numeric_limits<double>::epsilon());
  if (std::isfinite(input.rate) && input.rate != 0.0 &&
      std::isfinite(physical_acceleration) &&
      std::isfinite(denominator) && denominator > 0.0) {
    const double magnitude = std::abs(input.rate) / denominator;
    if (std::isfinite(magnitude) && magnitude > 0.0) {
      const double gentle_acceleration = std::copysign(
          std::min(magnitude, std::abs(physical_acceleration)),
          physical_acceleration);
      if (gentle_acceleration != physical_acceleration &&
          certify_numerical_braking_acceleration(
              input, dt, gentle_acceleration, result)) {
        return true;
      }
    }
  }
  // Overflow, underflow, or a failed gentle certificate preserves the
  // previously certified physical-cap witness. No interval is widened.
  return certify_numerical_braking_acceleration(
      input, dt, physical_acceleration, result);
}

} // namespace

StateBoxRowResult shape_state_box_row(const StateBoxRowInput &input, double dt) {
  auto result = shape_state_box_row_nominal(input, dt);
  const bool failed_boundary_or_envelope =
      result.status == SolverStatus::kInfeasible &&
      (result.failure == StateBoxFailure::kOutwardRateAtBoundary ||
       result.failure == StateBoxFailure::kOutsideBrakingEnvelope ||
       result.failure == StateBoxFailure::kEmptyInterval);
  const bool interval_misses_physical_box =
      result.status == SolverStatus::kSuccess &&
      ((input.acceleration_bounds.lower_active &&
        result.upper_acceleration < input.acceleration_bounds.lower) ||
       (input.acceleration_bounds.upper_active &&
        result.lower_acceleration > input.acceleration_bounds.upper));
  if (failed_boundary_or_envelope || interval_misses_physical_box) {
    StateBoxRowResult singleton;
    if (numerical_braking_singleton(input, dt, &singleton)) {
      return singleton;
    }
  }
  return result;
}

LinearizedStateBoxResult prepare_linearized_state_box(
    const LinearizedStateBoxInput &input, double dt,
    Eigen::Index variable_count) {
  LinearizedStateBoxResult result;
  if (input.source_id.empty()) {
    result.status = SolverStatus::kInvalidInput;
    result.message = "linearized state-box source_id must not be empty";
    return result;
  }
  const Eigen::Index row_count = input.coefficient_matrix.rows();
  if (row_count == 0 || input.coefficient_matrix.cols() != variable_count ||
      input.affine_bias.size() != row_count ||
      input.rows.size() != static_cast<std::size_t>(row_count)) {
    result.status = SolverStatus::kShapeMismatch;
    result.message = "linearized state-box dimensions are inconsistent";
    return result;
  }
  if (!input.coefficient_matrix.allFinite() || !input.affine_bias.allFinite()) {
    result.status = SolverStatus::kNonFiniteInput;
    result.message = "linearized state-box matrix and bias must be finite";
    return result;
  }

  Eigen::VectorXd lower(row_count);
  Eigen::VectorXd upper(row_count);
  std::vector<bool> lower_active(static_cast<std::size_t>(row_count));
  std::vector<bool> upper_active(static_cast<std::size_t>(row_count));
  std::vector<StateBoxRowResult> shaped_rows;
  shaped_rows.reserve(static_cast<std::size_t>(row_count));
  for (Eigen::Index row = 0; row < row_count; ++row) {
    auto shaped =
        shape_state_box_row(input.rows[static_cast<std::size_t>(row)], dt);
    if (shaped.status != SolverStatus::kSuccess) {
      result.status = shaped.status;
      result.message = "linearized state-box '" + input.source_id + "' row " +
                       std::to_string(row) + ": " + shaped.message;
      return result;
    }

    lower(row) = shaped.lower_acceleration;
    upper(row) = shaped.upper_acceleration;
    lower_active[static_cast<std::size_t>(row)] = shaped.lower_active;
    upper_active[static_cast<std::size_t>(row)] = shaped.upper_active;

    if (input.coefficient_matrix.row(row).stableNorm() == 0.0) {
      const double value = input.affine_bias(row);
      if ((shaped.lower_active &&
           value < shaped.lower_acceleration -
                       comparison_tolerance(value,
                                            shaped.lower_acceleration)) ||
          (shaped.upper_active &&
           value > shaped.upper_acceleration +
                       comparison_tolerance(value,
                                            shaped.upper_acceleration))) {
        result.status = SolverStatus::kInfeasible;
        result.message = "linearized state-box '" + input.source_id +
                         "' has an infeasible zero acceleration row";
        return result;
      }
    }

    shaped_rows.push_back(std::move(shaped));
  }

  result.physical_constraint.source_id = input.source_id;
  result.physical_constraint.coefficient_matrix = input.coefficient_matrix;
  result.physical_constraint.affine_bias = input.affine_bias;
  result.physical_constraint.lower_bounds = std::move(lower);
  result.physical_constraint.upper_bounds = std::move(upper);
  result.physical_constraint.lower_bound_active = std::move(lower_active);
  result.physical_constraint.upper_bound_active = std::move(upper_active);
  result.state_rows = input.rows;
  result.shaped_rows = std::move(shaped_rows);
  return result;
}

} // namespace embodik::detail
