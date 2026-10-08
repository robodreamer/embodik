#include <embodik/ik_baseline.hpp>
#include <gtest/gtest.h>

#include <array>

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

TEST(VelocityHardRowCertification, PositiveRecoveryMinErrorMustReturnCertifiedCandidate) {
  const RecoveryProblem problem;
  RecoveryProblem certified = problem;
  certified.config.certify_explicit_task_modes = true;
  const SolverResult result = certified.legacy();
  ASSERT_EQ(result.status, SolverStatus::kSuccess);
  ASSERT_EQ(result.solution.size(), kDecisionColumns);
  EXPECT_LE(problem.maximum_violation(result), kTolerance);
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
class CertifiedVelocityModes : public ::testing::TestWithParam<TaskSolveMode> {};

TEST_P(CertifiedVelocityModes, FeasibleOriginalRowsAreCertifiedForEveryTaskMode) {
  RecoveryProblem problem;
  problem.config.certify_explicit_task_modes = true;
  problem.primary_goal.setConstant(2.0 * kRecoveryRate);
  problem.policies.front().solve_mode = GetParam();
  problem.policies.front().allow_min_error_fallback = false;
  const SolverResult result = problem.legacy();
  ASSERT_EQ(result.status, SolverStatus::kSuccess);
  EXPECT_LE(problem.maximum_violation(result), kTolerance);
}

TEST_P(CertifiedVelocityModes, InfeasibleOriginalRowsCannotReportSuccess) {
  RecoveryProblem problem;
  problem.config.certify_explicit_task_modes = true;
  problem.lower(kDecisionColumns) = 3.0 * kBoxSpeed;
  for (ObjectiveSolveConfig &policy : problem.policies) {
    policy.solve_mode = GetParam();
  }
  const SolverResult result = problem.legacy();
  EXPECT_EQ(result.status, SolverStatus::kInfeasible);
}

INSTANTIATE_TEST_SUITE_P(AllPolicies, CertifiedVelocityModes,
    ::testing::Values(TaskSolveMode::kMinError, TaskSolveMode::kScale,
                      TaskSolveMode::kScaleElastic));

TEST(VelocityHardRowCertification, ImplicitLegacyApiIsUnchangedWhenCertificationEnabled) {
  RecoveryProblem problem;
  problem.policies.clear();
  const SolverResult legacy = problem.legacy();
  problem.config.certify_explicit_task_modes = true;
  const SolverResult still_legacy = problem.legacy();
  EXPECT_EQ(still_legacy.status, legacy.status);
  EXPECT_EQ(still_legacy.solution, legacy.solution);
  EXPECT_EQ(still_legacy.task_scales, legacy.task_scales);
  EXPECT_EQ(still_legacy.task_errors, legacy.task_errors);
  EXPECT_EQ(still_legacy.task_modes_effective, legacy.task_modes_effective);
}

TEST(VelocityHardRowCertification, IncompatibleTaskEqualityReturnsCertifiedNoProgress) {
  RecoveryProblem problem;
  problem.config.certify_explicit_task_modes = true;
  problem.primary_goal.setConstant(2.0 * kRecoveryRate);
  for (ObjectiveSolveConfig &policy : problem.policies) {
    policy.solve_mode = TaskSolveMode::kScale;
    policy.allow_min_error_fallback = false;
  }
  const SolverResult result = problem.legacy();
  ASSERT_EQ(result.status, SolverStatus::kNoProgress);
  EXPECT_LE(problem.maximum_violation(result), kTolerance);
}

TEST(VelocityHardRowCertification, ExplicitLegacyDefaultRetainsPreviousBehavior) {
  const RecoveryProblem problem;
  const SolverResult legacy = problem.legacy();
  ASSERT_EQ(legacy.status, SolverStatus::kSuccess);
  EXPECT_GT(problem.maximum_violation(legacy), kTolerance);
}

} // namespace
} // namespace embodik::test
