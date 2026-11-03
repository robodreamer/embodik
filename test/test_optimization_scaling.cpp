#include <embodik/optimization.hpp>

#include <Eigen/Dense>
#include <gtest/gtest.h>

namespace {
struct Projection {
  Eigen::Matrix2d J, A;
  Eigen::Vector2d target, lower, upper;
  Projection() {
    J << 2., 2., 1., 0.;
    target << 4., 5.;
    A << -2., -1., -1., 2.;
    lower.setConstant(-1.);
    upper.setConstant(1.);
  }
};

TEST(StandaloneOptimizationScaling, ResidualUnitsPreserveActiveFaceAndWarmStart) {
  const Projection p;
  for (const double scale : {1e-100, 1e-4, 1., 1e100}) {
    for (const bool warm : {false, true}) {
      embodik::SolveOptions options;
      if (warm) options.x0 = Eigen::Vector2d(.6, -.2);
      const auto result = embodik::solve_ls(scale*p.J, scale*p.target,
          p.A, p.lower, p.upper, 0., options);
      ASSERT_TRUE(result.success()) << result.message << " scale=" << scale;
      EXPECT_NEAR(result.x[0], .2, 1e-10);
      EXPECT_NEAR(result.x[1], .6, 1e-10);
      EXPECT_NEAR(result.objective/(scale*scale), 14.4, 1e-10);
      EXPECT_LE(result.normalized_violation, options.tolerance);
    }
  }
}

TEST(StandaloneOptimizationScaling, QuadraticUnitsPreserveActiveFaceAndWarmStart) {
  const Projection p;
  for (const double scale : {1e-200, 1e-8, 1., 1e200}) {
    for (const bool warm : {false, true}) {
      embodik::SolveOptions options;
      if (warm) options.x0 = Eigen::Vector2d(.6, -.2);
      const auto result = embodik::solve_qp(scale*(p.J.transpose()*p.J),
          -scale*(p.J.transpose()*p.target), p.A, p.lower, p.upper, options);
      ASSERT_TRUE(result.success()) << result.message << " scale=" << scale;
      EXPECT_NEAR(result.x[0], .2, 1e-10);
      EXPECT_NEAR(result.x[1], .6, 1e-10);
      EXPECT_NEAR(result.objective/scale, -6.1, 1e-10);
    }
  }
}

TEST(StandaloneOptimizationScaling, RidgeStaysInSquaredResidualUnits) {
  const Projection p;
  for (const double scale : {1e-100, 1e-4, 1., 1e100}) {
    const auto result = embodik::solve_ls(scale*p.J, scale*p.target, p.A,
        p.lower, p.upper, .5*scale*scale);
    ASSERT_TRUE(result.success()) << result.message << " scale=" << scale;
    EXPECT_NEAR(result.x[0], 4./15., 1e-10);
    EXPECT_NEAR(result.x[1], 7./15., 1e-10);
    EXPECT_NEAR(result.objective/(scale*scale), 869./60., 1e-10);
  }
}

TEST(StandaloneOptimizationScaling, UnrepresentableCoefficientsAreNotDropped) {
  Eigen::Matrix2d factors = Eigen::Matrix2d::Zero();
  factors(0, 0) = 1e200;
  factors(1, 1) = 1e-200;
  const Eigen::MatrixXd no_rows(0, 2);
  const Eigen::VectorXd no_bounds(0);
  for (const auto &result : {
      embodik::solve_ls(factors, Eigen::Vector2d::Zero(), no_rows, no_bounds, no_bounds),
      embodik::solve_qp(factors, Eigen::Vector2d::Zero(), no_rows, no_bounds, no_bounds)}) {
    EXPECT_EQ(result.status, embodik::SolveStatus::kNoProgress);
    EXPECT_EQ(result.x.size(), 0);
  }
  Eigen::MatrixXd one(1, 1), no_scalar_rows(0, 1);
  one(0, 0) = 1e100;
  const auto ridge = embodik::solve_ls(one, Eigen::VectorXd::Zero(1),
      no_scalar_rows, no_bounds, no_bounds, 1e-300);
  EXPECT_EQ(ridge.status, embodik::SolveStatus::kNoProgress);
  one(0, 0) = 1e-300;
  const auto target = embodik::solve_ls(one, Eigen::VectorXd::Constant(1, 1e100),
      no_scalar_rows, no_bounds, no_bounds);
  const auto linear = embodik::solve_qp(one, Eigen::VectorXd::Constant(1, 1e100),
      no_scalar_rows, no_bounds, no_bounds);
  EXPECT_EQ(target.status, embodik::SolveStatus::kNoProgress);
  EXPECT_EQ(linear.status, embodik::SolveStatus::kNoProgress);
}

TEST(StandaloneOptimizationScaling, RankAndCapabilityRulesRemainExplicit) {
  const Eigen::MatrixXd no_rows(0, 2);
  const Eigen::VectorXd no_bounds(0);
  Eigen::Matrix2d weak = Eigen::Matrix2d::Zero();
  weak.diagonal() << 1., 1e-20;
  Eigen::Vector2d target(0., 1.);
  const auto unresolved = embodik::solve_ls(weak, target, no_rows, no_bounds, no_bounds);
  EXPECT_EQ(unresolved.status, embodik::SolveStatus::kNoProgress);
  Eigen::Matrix2d dependent;
  dependent << 1., 1., 2., 2.;
  target << 1., 3.;
  const auto inconsistent = embodik::solve_ls(dependent, target, no_rows, no_bounds, no_bounds);
  EXPECT_EQ(inconsistent.status, embodik::SolveStatus::kNoProgress);
  const auto unsupported = embodik::solve_qp(Eigen::Matrix2d::Zero(),
      Eigen::Vector2d::Ones(), no_rows, no_bounds, no_bounds);
  EXPECT_EQ(unsupported.status, embodik::SolveStatus::kUnsupported);
}

TEST(StandaloneOptimizationScaling, ConstantRankPolicyAndEmptyResidualRemainExplicit) {
  const Eigen::MatrixXd no_rows(0, 2);
  const Eigen::VectorXd no_bounds(0);
  const auto constant = embodik::solve_ls(Eigen::Matrix2d::Zero(),
      Eigen::Vector2d::Ones(), no_rows, no_bounds, no_bounds);
  // Preserve the existing conservative nonzero-residual rank policy.
  EXPECT_EQ(constant.status, embodik::SolveStatus::kNoProgress);
  EXPECT_EQ(constant.x.size(), 0);
  const auto empty = embodik::solve_ls(Eigen::MatrixXd(0, 2),
      no_bounds, no_rows, no_bounds, no_bounds);
  ASSERT_TRUE(empty.success()) << empty.message;
  EXPECT_DOUBLE_EQ(empty.objective, 0.);
}
} // namespace
