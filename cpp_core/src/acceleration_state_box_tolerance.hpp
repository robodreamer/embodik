#pragma once

#include <algorithm>
#include <cmath>

namespace embodik::detail {

// Shared by incoming state-box checks and nonlinear endpoint state/braking
// checks. A looser endpoint certificate can publish a state that the next
// solve rejects before optimization starts.
inline double state_box_comparison_tolerance(double lhs, double rhs) {
  return 1e-12 * (1.0 + std::max(std::abs(lhs), std::abs(rhs)));
}

} // namespace embodik::detail
