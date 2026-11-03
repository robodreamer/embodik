#pragma once

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace embodik::detail {

// Construction proxy for a frozen affine distance, not a certificate for
// nonlinear geometry or simultaneously available multi-pair braking authority.
// For closing next rate w, budget this interval before the subsequent stop:
//   w*w/(2*A) <= slack + dt*(incoming_rate + w)/2.
// The ordinary velocity-command construction has a separate delay reserve.
inline double collision_acceleration_braking_speed(
    double slack, double incoming_rate, double acceleration, double dt) {
  if (!std::isfinite(slack) || !std::isfinite(incoming_rate) ||
      !std::isfinite(acceleration) || acceleration < 0.0 ||
      !std::isfinite(dt) || dt <= 0.0) {
    throw std::invalid_argument("invalid collision acceleration braking input");
  }
  if (slack <= 0.0 || acceleration == 0.0) return 0.0;
  const long double a = acceleration;
  const long double h = dt;
  const long double budget = 2.0L * slack + h * incoming_rate;
  if (!std::isfinite(budget))
    throw std::overflow_error("collision acceleration braking budget overflow");
  if (budget <= 0.0L) return 0.0;
  const long double c = a * h;
  const long double root = std::sqrt(c * c + 4.0L * a * budget);
  if (!std::isfinite(c) || !std::isfinite(root))
    throw std::overflow_error("collision acceleration braking root overflow");
  // Rationalized root avoids cancellation for small clearance budgets.
  return static_cast<double>(std::min(
      2.0L * a * budget / (root + c),
      static_cast<long double>(std::numeric_limits<double>::max())));
}

} // namespace embodik::detail
