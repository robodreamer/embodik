#include <embodik/optimization.hpp>

#include <Eigen/Dense>
#include <gtest/gtest.h>
#include <limits>

namespace {
const double inf = std::numeric_limits<double>::infinity();
const Eigen::MatrixXd no_rows(0, 1);
const Eigen::VectorXd no_bounds(0);

embodik::SoftBounds scalar_soft(double lower, double upper, double weight = 1.) {
  return {Eigen::MatrixXd::Ones(1, 1), Eigen::VectorXd::Constant(1, lower),
          Eigen::VectorXd::Constant(1, upper), Eigen::VectorXd::Constant(1, weight)};
}

embodik::SolveResult scalar(const embodik::SoftBounds &soft, double target,
                            double ridge = 0., const embodik::SolveOptions &options = {}) {
  return embodik::solve_ls(Eigen::MatrixXd::Ones(1, 1),
      Eigen::VectorXd::Constant(1, target), no_rows, no_bounds, no_bounds,
      soft, ridge, options);
}

TEST(StandaloneSoftBounds, InactiveIntervalsHaveNoMovementCost) {
  const auto result = scalar(scalar_soft(-1., 3., 9.), 2.);
  ASSERT_TRUE(result.success()) << result.message;
  ASSERT_EQ(result.x.size(), 1);
  EXPECT_NEAR(result.x(0), 2., 1e-12);
  EXPECT_NEAR(result.objective, 0., 1e-20);
  EXPECT_DOUBLE_EQ(result.max_violation, 0.);
}

TEST(StandaloneSoftBounds, WeightedOneSidedAndEqualityPenalties) {
  for (const auto &interval : {scalar_soft(0., inf, 3.), scalar_soft(-inf, -2., 3.),
                               scalar_soft(-2., -2., 3.)}) {
    const bool lower = interval.lower(0) == 0.;
    const auto result = scalar(interval, lower ? -2. : 0.);
    ASSERT_TRUE(result.success()) << result.message;
    EXPECT_NEAR(result.x(0), lower ? -.5 : -1.5, 1e-10);
    EXPECT_NEAR(result.objective, 1.5, 1e-10);
  }
}

TEST(StandaloneSoftBounds, RidgeAppliesOnlyToOriginalVariables) {
  const auto result = scalar(scalar_soft(1., inf), 0., 2.);
  ASSERT_TRUE(result.success()) << result.message;
  ASSERT_EQ(result.x.size(), 1);
  EXPECT_NEAR(result.x(0), .25, 1e-10);
  EXPECT_NEAR(result.objective, .375, 1e-10);
}

TEST(StandaloneSoftBounds, ConflictingSoftRowsDoNotMakeHardProblemInfeasible) {
  const embodik::SoftBounds soft{Eigen::MatrixXd::Ones(2, 1),
      (Eigen::Vector2d() << 2., -inf).finished(),
      (Eigen::Vector2d() << inf, -2.).finished(), Eigen::Vector2d::Ones()};
  const auto result = scalar(soft, 0.);
  ASSERT_TRUE(result.success()) << result.message;
  EXPECT_NEAR(result.x(0), 0., 1e-10);
  EXPECT_NEAR(result.objective, 4., 1e-10);
}

TEST(StandaloneSoftBounds, HardRowsTakePrecedenceAndMetricsExcludeSoftResiduals) {
  const Eigen::MatrixXd A = Eigen::MatrixXd::Ones(1, 1);
  const auto result = embodik::solve_ls(A, Eigen::VectorXd::Zero(1), A,
      Eigen::VectorXd::Constant(1, -inf), Eigen::VectorXd::Zero(1),
      scalar_soft(2., inf), 0., {});
  ASSERT_TRUE(result.success()) << result.message;
  EXPECT_NEAR(result.x(0), 0., 1e-10);
  EXPECT_NEAR(result.objective, 2., 1e-10);
  EXPECT_LE(result.max_violation, 1e-10);
  EXPECT_LE(result.normalized_violation, 1e-10);
}

TEST(StandaloneSoftBounds, ConstructedIndependentKktReference) {
  Eigen::MatrixXd S(4, 3), A(3, 3);
  S << 1.,0.,0., 0.,1.,0., 1.,-1.,0., 0.,0.,1.;
  A << 1.,1.,1., 1.,0.,0., 0.,0.,1.;
  Eigen::VectorXd sl(4), su(4), weights(4), lower(3), upper(3), target(3);
  sl << 1.,-inf,-.5,0.; su << inf,-.5,.5,1.; weights << 2.,4.,3.,5.;
  lower << 1.,.5,-inf; upper << 1.,inf,.75; target << .4,.075,1.725;
  embodik::SolveOptions options;
  options.tolerance = 1e-10;
  options.x0 = Eigen::Vector3d(.5,-.25,.75);
  const auto result = embodik::solve_ls(Eigen::Matrix3d::Identity(), target, A,
      lower, upper, embodik::SoftBounds{S,sl,su,weights}, .5, options);
  ASSERT_TRUE(result.success()) << result.message;
  ASSERT_EQ(result.x.size(), 3);
  EXPECT_TRUE(result.x.isApprox(Eigen::Vector3d(.5,-.25,.75), 1e-9));
  const Eigen::VectorXd value = S*result.x;
  const Eigen::VectorXd projected = value.cwiseMax(sl).cwiseMin(su);
  const Eigen::VectorXd difference = value-projected;
  const Eigen::VectorXd gradient = 1.5*result.x-target +
      S.transpose()*(weights.array()*difference.array()).matrix();
  // Equality multiplier .2, lower-bound multiplier -.3, upper multiplier .4.
  const Eigen::Vector3d dual(.2,-.3,.4);
  EXPECT_LE((gradient+A.transpose()*dual).lpNorm<Eigen::Infinity>(), 1e-9);
  const double objective = .5*(result.x-target).squaredNorm()+.25*result.x.squaredNorm()+
      .5*(weights.array()*difference.array().square()).sum();
  EXPECT_NEAR(result.objective, objective, 1e-12);
}

TEST(StandaloneSoftBounds, DisabledRowsPreserveBaseRankAndPrimalExactly) {
  Eigen::MatrixXd J(1, 2), S = Eigen::MatrixXd::Identity(2, 2);
  J << 1.,1.;
  const Eigen::MatrixXd A(0, 2);
  const auto base = embodik::solve_ls(J, Eigen::VectorXd::Ones(1), A, no_bounds, no_bounds);
  for (const auto &soft : {
      embodik::SoftBounds{S,Eigen::Vector2d::Zero(),Eigen::Vector2d::Ones(),Eigen::Vector2d::Zero()},
      embodik::SoftBounds{S,Eigen::Vector2d::Constant(-inf),Eigen::Vector2d::Constant(inf),Eigen::Vector2d::Ones()}}) {
    const auto result = embodik::solve_ls(J,Eigen::VectorXd::Ones(1),A,no_bounds,no_bounds,soft,0.,{});
    ASSERT_EQ(result.status, base.status);
    EXPECT_TRUE((result.x.array() == base.x.array()).all());
    EXPECT_DOUBLE_EQ(result.objective, base.objective);
    EXPECT_EQ(result.iterations, base.iterations);
  }
}

TEST(StandaloneSoftBounds, InvalidIntervalsAndWeightsHaveNoCandidate) {
  for (const auto &soft : {scalar_soft(1.,0.), scalar_soft(inf,inf),
      scalar_soft(-inf,-inf), scalar_soft(0.,inf,-1.), scalar_soft(0.,inf,inf),
      scalar_soft(std::numeric_limits<double>::quiet_NaN(),inf)}) {
    const auto result = scalar(soft,0.);
    EXPECT_EQ(result.status,embodik::SolveStatus::kInvalidInput);
    EXPECT_EQ(result.x.size(),0);
  }
}

TEST(StandaloneSoftBounds, InvalidShapesAndNonfiniteMatrixAreRejected) {
  auto soft = scalar_soft(0.,inf);
  soft.matrix.resize(1,2);
  EXPECT_EQ(scalar(soft,0.).status,embodik::SolveStatus::kInvalidInput);
  soft = scalar_soft(0.,inf); soft.weights = Eigen::Vector2d::Ones();
  EXPECT_EQ(scalar(soft,0.).status,embodik::SolveStatus::kInvalidInput);
  soft = scalar_soft(0.,inf); soft.matrix(0,0) = inf;
  EXPECT_EQ(scalar(soft,0.).status,embodik::SolveStatus::kInvalidInput);
}

TEST(StandaloneSoftBounds, EmptyResidualAndOriginalHardInfeasibilityRemainDistinct) {
  const auto soft = scalar_soft(1.,inf);
  const auto result = embodik::solve_ls(Eigen::MatrixXd(0,1), no_bounds,
      no_rows,no_bounds,no_bounds,soft,0.,{});
  ASSERT_TRUE(result.success()) << result.message;
  EXPECT_GE(result.x(0),1.-1e-10);
  EXPECT_NEAR(result.objective,0.,1e-18);
  const auto infeasible = embodik::solve_ls(Eigen::MatrixXd::Ones(1,1),
      Eigen::VectorXd::Zero(1),Eigen::MatrixXd::Zero(1,1),
      Eigen::VectorXd::Ones(1),Eigen::VectorXd::Ones(1),soft,0.,{});
  EXPECT_EQ(infeasible.status,embodik::SolveStatus::kInfeasible);
  EXPECT_EQ(infeasible.x.size(),0);
}

TEST(StandaloneSoftBounds, UnrepresentableWeightingAndExhaustedBudgetDoNotClaimSolved) {
  auto soft = scalar_soft(0.,inf,1e-300);
  soft.matrix(0,0) = 1e-300;
  const auto result = scalar(soft,1.);
  EXPECT_EQ(result.status,embodik::SolveStatus::kNoProgress);
  EXPECT_EQ(result.x.size(),0);
  embodik::SolveOptions options;
  options.max_iterations = 0;
  EXPECT_EQ(scalar(scalar_soft(0.,inf),1.,0.,options).status,embodik::SolveStatus::kInvalidInput);
}

TEST(StandaloneSoftBounds, ExistingBraceArgumentsStayUnambiguous) {
  const Eigen::MatrixXd J = Eigen::MatrixXd::Ones(1, 1);
  const Eigen::VectorXd target = Eigen::VectorXd::Zero(1);
  const auto six = embodik::solve_ls(J, target, no_rows, no_bounds, no_bounds, {});
  const auto seven = embodik::solve_ls(J, target, no_rows, no_bounds, no_bounds, {}, {});
  const auto eight = embodik::solve_ls(J, target, no_rows, no_bounds, no_bounds,
      scalar_soft(-1., 1.), 0., embodik::SolveOptions{});
  EXPECT_TRUE(six.success()) << six.message;
  EXPECT_TRUE(seven.success()) << seven.message;
  EXPECT_TRUE(eight.success()) << eight.message;
}
} // namespace
