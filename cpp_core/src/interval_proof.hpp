#pragma once

#include "bounded_interval.hpp"

#include <cmath>
#include <cstddef>
#include <string>
#include <vector>

namespace embodik::detail {

enum class IntervalProofStatus { kProved, kUnproved, kInvalidInput };
struct IntervalProofBudget {
  std::size_t nodes = 32;
  unsigned int depth = 24;
};
struct IntervalProofResult {
  IntervalProofStatus status = IntervalProofStatus::kUnproved;
  std::size_t visited_nodes = 0;
  std::size_t certified_intervals = 0;
  std::string message;
  bool proved() const { return status == IntervalProofStatus::kProved; }
};

// The callback owns the mathematical contract: each finite component must be a
// lower bound for a fixed original inequality residual over the WHOLE interval.
// This traversal adds no tolerance, changes no floor and never proves violation
// or infeasibility from a negative lower bound. It owns each callback result.
template<class LowerBounds>
IntervalProofResult prove_interval_inequalities(
    double dt, std::size_t inequality_count, LowerBounds &&lower_bounds,
    IntervalProofBudget budget = {}) {
  IntervalProofResult result;
  const ScopedIntervalEnvironment environment;
  if (!std::isfinite(dt) || dt <= 0.0 || inequality_count == 0 || budget.nodes == 0) {
    result.status = IntervalProofStatus::kInvalidInput;
    result.message = "positive finite horizon, inequality count and node budget required";
    return result;
  }
  struct Node { double lo, hi; unsigned int depth; };
  std::vector<Node> pending{{0.0, dt, 0}};
  while (!pending.empty()) {
    if (result.visited_nodes == budget.nodes) {
      result.message = "interval node budget exhausted";
      return result;
    }
    const auto node = pending.back();
    pending.pop_back();
    ++result.visited_nodes;
    std::vector<double> bounds;
    try {
      bounds = lower_bounds(node.lo, node.hi);
    } catch (const std::domain_error &error) {
      result.message = std::string("interval arithmetic unproved: ") + error.what();
      return result;
    } catch (const std::overflow_error &error) {
      result.message = std::string("interval arithmetic unproved: ") + error.what();
      return result;
    }
    if (bounds.size() != inequality_count) {
      result.status = IntervalProofStatus::kInvalidInput;
      result.message = "interval inequality count changed";
      return result;
    }
    bool proved = true;
    for (double value : bounds) {
      if (!std::isfinite(value)) {
        result.status = IntervalProofStatus::kInvalidInput;
        result.message = "finite interval inequality bounds required";
        return result;
      }
      proved = proved && value >= 0.0;
    }
    if (proved) {
      ++result.certified_intervals;
      continue;
    }
    const double midpoint = node.lo + 0.5 * (node.hi - node.lo);
    if (node.depth == budget.depth || midpoint <= node.lo || midpoint >= node.hi) {
      result.message = "interval unproved at depth or representable-time budget";
      return result;
    }
    pending.push_back({midpoint, node.hi, node.depth + 1});
    pending.push_back({node.lo, midpoint, node.depth + 1});
  }
  result.status = IntervalProofStatus::kProved;
  result.message = "all continuous intervals proved";
  return result;
}

} // namespace embodik::detail
