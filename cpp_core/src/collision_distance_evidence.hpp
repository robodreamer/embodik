#pragma once

#include <cmath>
#include <limits>

namespace embodik::detail {

// Coal uses finite +/-max sentinels for missing or failed distance queries.
// They must never be interpreted as signed physical distances.
inline bool collision_distance_evidence_is_valid(double distance) {
  return std::isfinite(distance) &&
         distance != std::numeric_limits<double>::max() &&
         distance != -std::numeric_limits<double>::max();
}

// Disjoint enclosing bounds also rule out a finite distance below their
// conservative separation. Overlapping/unavailable bounds say nothing about
// signed penetration and must not reject ordinary contact evidence.
inline bool collision_distance_evidence_is_valid(
    double distance, double separation_lower_bound) {
  return collision_distance_evidence_is_valid(distance) &&
         (!std::isfinite(separation_lower_bound) ||
          separation_lower_bound <= 0.0 || distance >= separation_lower_bound);
}

} // namespace embodik::detail
