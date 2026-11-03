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

} // namespace embodik::detail
