#include "interval_proof.hpp"

#include <gtest/gtest.h>
#include <limits>

namespace embodik::detail {
namespace {
TEST(IntervalProof, ProvesWholeHorizonWithoutChangingResiduals) {
  const auto result = prove_interval_inequalities(1.0, 2,
      [](double, double) { return std::vector<double>{0.0, 1e-300}; });
  EXPECT_TRUE(result.proved());
  EXPECT_EQ(result.visited_nodes, 1);
  EXPECT_EQ(result.certified_intervals, 1);
}
TEST(IntervalProof, CoversBothChildrenAfterSubdivision) {
  const auto result = prove_interval_inequalities(1.0, 1,
      [](double lo, double hi) { return std::vector<double>{hi - lo <= .5 ? 0.0 : -1.0}; });
  EXPECT_TRUE(result.proved());
  EXPECT_EQ(result.visited_nodes, 3);
  EXPECT_EQ(result.certified_intervals, 2);
}
TEST(IntervalProof, ExhaustionAndNegativeBoundsDoNotProveInfeasibility) {
  const auto assess = [](double lo, double hi) { return std::vector<double>{hi - lo <= .5 ? 0.0 : -1.0}; };
  const auto nodes = prove_interval_inequalities(1.0, 1, assess, {2, 24});
  EXPECT_EQ(nodes.status, IntervalProofStatus::kUnproved);
  EXPECT_EQ(nodes.visited_nodes, 2);
  EXPECT_EQ(nodes.certified_intervals, 1);
  const auto depth = prove_interval_inequalities(1.0, 1, assess, {32, 0});
  EXPECT_EQ(depth.status, IntervalProofStatus::kUnproved);
  EXPECT_EQ(depth.visited_nodes, 1);
}
TEST(IntervalProof, RejectsInvalidContractsAndNonfiniteBounds) {
  const auto zero = [](double, double) { return std::vector<double>{0.0}; };
  EXPECT_EQ(prove_interval_inequalities(0.0, 1, zero).status, IntervalProofStatus::kInvalidInput);
  EXPECT_EQ(prove_interval_inequalities(INFINITY, 1, zero).status, IntervalProofStatus::kInvalidInput);
  EXPECT_EQ(prove_interval_inequalities(1.0, 0, zero).status, IntervalProofStatus::kInvalidInput);
  EXPECT_EQ(prove_interval_inequalities(1.0, 1, zero, {0, 24}).status, IntervalProofStatus::kInvalidInput);
  EXPECT_EQ(prove_interval_inequalities(1.0, 2, zero).status, IntervalProofStatus::kInvalidInput);
  EXPECT_EQ(prove_interval_inequalities(1.0, 1,
      [](double, double) { return std::vector<double>{NAN}; }).status, IntervalProofStatus::kInvalidInput);
}
TEST(IntervalProof, ArithmeticFailureAndUnsplittableTimeRemainUnproved) {
  const auto failure = prove_interval_inequalities(1.0, 1,
      [](double, double) -> std::vector<double> { throw std::domain_error("enclosure overflow"); });
  EXPECT_EQ(failure.status, IntervalProofStatus::kUnproved);
  EXPECT_EQ(failure.visited_nodes, 1);
  const auto tiny = prove_interval_inequalities(std::numeric_limits<double>::denorm_min(), 1,
      [](double, double) { return std::vector<double>{-1.0}; });
  EXPECT_EQ(tiny.status, IntervalProofStatus::kUnproved);
  EXPECT_EQ(tiny.visited_nodes, 1);
}
TEST(IntervalProof, AChangingCallbackDimensionCannotCompleteAPartialProof) {
  const auto result = prove_interval_inequalities(1.0, 1,
      [](double lo, double hi) {
        if (lo == .5) return std::vector<double>{0.0, 0.0};
        return std::vector<double>{hi - lo <= .5 ? 0.0 : -1.0};
      });
  EXPECT_EQ(result.status, IntervalProofStatus::kInvalidInput);
  EXPECT_EQ(result.visited_nodes, 3);
  EXPECT_EQ(result.certified_intervals, 1);
}
} // namespace
} // namespace embodik::detail
