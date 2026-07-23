#include "acceleration_state_box.hpp"

#include <algorithm>
#include <cmath>
#include <utility>

namespace embodik::detail {
namespace {

constexpr double kFinalIntervalCanonicalizationTolerance = 1e-10;

double comparison_tolerance(double lhs, double rhs) {
  return 1e-12 * (1.0 + std::max(std::abs(lhs), std::abs(rhs)));
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

StateBoxRowResult shape_state_box_row(const StateBoxRowInput &input,
                                      double dt) {
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

} // namespace embodik::detail
