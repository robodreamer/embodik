#include <embodik/ik_baseline.hpp>
#include <gtest/gtest.h>

#include <array>
#include <tuple>

namespace embodik::test {
namespace {
constexpr int kDecisionColumns = 2;
constexpr int kConstraintRows = 4;
constexpr double kTolerance = 1e-6;
constexpr double kRecoveryRate = 0.25;
constexpr double kBoxSpeed = 1.0;
constexpr double kPrimaryGoal = -1.0;
constexpr double kRecoveryUpperBound = 10.0 * kBoxSpeed;
constexpr unsigned int kMaximumIterations = 100;
constexpr std::array<double, kDecisionColumns> kPrimaryCoefficients{-0.4, -0.7};
constexpr std::array<double, kDecisionColumns> kSecondaryGoal{0.6, 0.2};
constexpr std::array<double, kDecisionColumns * kConstraintRows> kConstraintCoefficients{
    1.0, 0.0, 0.0, 1.0, -0.4, -1.0, -0.6, 0.4};

struct RecoveryProblem {
  Eigen::MatrixXd primary = Eigen::MatrixXd(1, kDecisionColumns);
  Eigen::VectorXd primary_goal = Eigen::VectorXd(1);
  Eigen::MatrixXd secondary = Eigen::MatrixXd::Identity(kDecisionColumns, kDecisionColumns);
  Eigen::VectorXd secondary_goal = Eigen::VectorXd(kDecisionColumns);
  Eigen::MatrixXd rows = Eigen::MatrixXd(kConstraintRows, kDecisionColumns);
  Eigen::VectorXd lower = Eigen::VectorXd(kConstraintRows);
  Eigen::VectorXd upper = Eigen::VectorXd(kConstraintRows);
  VelocitySolverConfig config;
  std::vector<ObjectiveSolveConfig> policies;

  RecoveryProblem() {
    primary = Eigen::Map<const Eigen::RowVector2d>(kPrimaryCoefficients.data());
    primary_goal.setConstant(kPrimaryGoal);
    secondary_goal = Eigen::Map<const Eigen::Vector2d>(kSecondaryGoal.data());
    rows = Eigen::Map<const Eigen::Matrix<double, kConstraintRows, kDecisionColumns, Eigen::RowMajor>>(kConstraintCoefficients.data());
    lower << -kBoxSpeed, -kBoxSpeed, kRecoveryRate, 0.0;
    upper << kBoxSpeed, kBoxSpeed, kRecoveryUpperBound, kRecoveryUpperBound;
    config.iteration_limit = kMaximumIterations;
    ObjectiveSolveConfig primary_policy;
    primary_policy.solve_mode = TaskSolveMode::kMinError;
    policies.push_back(primary_policy);
    ObjectiveSolveConfig secondary_policy = primary_policy;
    secondary_policy.priority = 1;
    policies.push_back(secondary_policy);
  }

  SolverResult legacy() const {
    return computeMultiObjectiveVelocitySolutionEigen(
        {primary_goal, secondary_goal}, {primary, secondary}, rows, lower, upper, config, policies);
  }

  SolverResult strict() const {
    return solveHierarchicalLinearSystemEigen(
        {primary_goal, secondary_goal},
        {Eigen::VectorXd::Zero(primary.rows()), Eigen::VectorXd::Zero(secondary.rows())},
        {primary, secondary}, rows, lower, upper, config, policies);
  }

  double maximum_violation(const SolverResult &result) const {
    const Eigen::Map<const Eigen::VectorXd> velocity(result.solution.data(), kDecisionColumns);
    const Eigen::VectorXd values = rows * velocity;
    return std::max((lower - values).maxCoeff(), (values - upper).maxCoeff());
  }
};

TEST(VelocityHardRowCertification, PublicVelocitySolveReturnsConstrainedMinErrorCommand) {
  const RecoveryProblem problem;
  const SolverResult result = problem.legacy();
  ASSERT_EQ(result.status, SolverStatus::kSuccess) << result.status_message;
  EXPECT_LE(problem.maximum_violation(result), kTolerance);
  const SolverResult reference = problem.strict();
  ASSERT_EQ(reference.status, SolverStatus::kSuccess);
  ASSERT_EQ(result.solution.size(), reference.solution.size());
  for (std::size_t index = 0; index < result.solution.size(); ++index) {
    EXPECT_NEAR(result.solution[index], reference.solution[index], kTolerance);
  }
}

TEST(VelocityHardRowCertification, GeneralizedPathObtainsConstrainedMinimumError) {
  const RecoveryProblem problem;
  const SolverResult result = problem.strict();
  ASSERT_EQ(result.status, SolverStatus::kSuccess);
  ASSERT_EQ(result.solution.size(), kDecisionColumns);
  EXPECT_LE(problem.maximum_violation(result), kTolerance);
  const Eigen::VectorXd expected = problem.rows.bottomRows(kDecisionColumns).fullPivLu().solve(
      problem.lower.tail(kDecisionColumns));
  const Eigen::Map<const Eigen::VectorXd> actual(result.solution.data(), kDecisionColumns);
  EXPECT_LE((actual - expected).norm(), kTolerance);
}
} // namespace
} // namespace embodik::test

namespace embodik::test {
namespace {
constexpr double kNearlyParallelCoupling = 5e-7;
constexpr double kActiveRankPrecision = 1e-10;
constexpr int kCompletionDecisionColumns = 21;
constexpr double kCompletionGoal = 1000.0;
constexpr double kDependentTaskRowScale = 2.0;
constexpr double kWellSeparatedCompletionCoupling = 1e-3;
using CompletionCase = std::tuple<TaskSolveMode, double, bool>;
class RedundantCompletionVelocity : public ::testing::TestWithParam<CompletionCase> {};

TEST_P(RedundantCompletionVelocity, NearParallelPinnedRowsKeepTheReachableScale) {
  const TaskSolveMode mode = std::get<0>(GetParam());
  const double coupling = std::get<1>(GetParam());
  const bool dependent_task_row = std::get<2>(GetParam());
  Eigen::MatrixXd objective = Eigen::MatrixXd::Ones(kDecisionColumns, kCompletionDecisionColumns);
  const double redundant_row_scale = dependent_task_row ? kDependentTaskRowScale : 0.0;
  objective.row(1) *= redundant_row_scale;
  Eigen::VectorXd goal(kDecisionColumns);
  goal << kCompletionGoal, redundant_row_scale * kCompletionGoal;
  Eigen::MatrixXd rows = Eigen::MatrixXd::Identity(kCompletionDecisionColumns, kCompletionDecisionColumns);
  rows.row(1).setZero();
  rows(1, 0) = 1.0;
  rows(1, 1) = coupling;
  const Eigen::VectorXd lower = Eigen::VectorXd::Zero(kCompletionDecisionColumns);
  Eigen::VectorXd upper = lower;
  upper(kCompletionDecisionColumns - 1) = kDependentTaskRowScale * kCompletionGoal;
  Eigen::VectorXd witness = lower;
  witness(kCompletionDecisionColumns - 1) = kCompletionGoal;
  ASSERT_TRUE(((rows * witness).array() >= lower.array()).all());
  ASSERT_TRUE(((rows * witness).array() <= upper.array()).all());
  ASSERT_LE((objective * witness - goal).norm(), kTolerance);
  VelocitySolverConfig config;
  config.epsilon = kTolerance;
  config.precision_threshold = kActiveRankPrecision;
  ObjectiveSolveConfig policy;
  policy.solve_mode = mode;
  policy.allow_min_error_fallback = false;
  const SolverResult result = computeMultiObjectiveVelocitySolutionEigen(
      {goal}, {objective}, rows, lower, upper, config, {policy});
  ASSERT_EQ(result.status, SolverStatus::kSuccess) << result.status_message;
  ASSERT_EQ(result.solution.size(), kCompletionDecisionColumns);
  const Eigen::Map<const Eigen::VectorXd> velocity(result.solution.data(), kCompletionDecisionColumns);
  // The witness above proves full-scale feasibility for both couplings. An
  // active-row rank loss must not turn that reachable command into a stall.
  const double expected_scale = 1.0;
  EXPECT_NEAR(result.task_scales.front(), expected_scale, kTolerance);
  EXPECT_LE((objective * velocity - expected_scale * goal).norm(), kTolerance);
  EXPECT_LE((velocity - expected_scale * witness).norm(), kTolerance);
  for (int row = 0; row < kCompletionDecisionColumns; ++row) {
    const double value = rows.row(row).dot(velocity);
    EXPECT_LE(std::max(lower(row) - value, value - upper(row)) / rows.row(row).norm(),
              kTolerance);
  }
}

INSTANTIATE_TEST_SUITE_P(ScaledPolicies, RedundantCompletionVelocity,
    ::testing::Combine(::testing::Values(TaskSolveMode::kScale, TaskSolveMode::kScaleElastic),
                       ::testing::Values(kNearlyParallelCoupling, kWellSeparatedCompletionCoupling),
                       ::testing::Bool()));
} // namespace
} // namespace embodik::test
