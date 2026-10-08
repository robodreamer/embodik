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

namespace embodik::test {
namespace {
constexpr double kNearlyParallelCoupling = 5e-7;
constexpr double kNearlyParallelSpeed = 10.0;
constexpr double kActiveRankPrecision = 1e-10;
constexpr std::array<double, 3> kConstraintRowScales{1e-3, 1.0, 1e3};

class CertifiedNearlyParallelRows : public ::testing::TestWithParam<TaskSolveMode> {};

TEST_P(CertifiedNearlyParallelRows, FeasibleThinConeRetainsIndependentHardRows) {
  for (double row_scale : kConstraintRowScales) {
    for (bool duplicate_row : {false, true}) {
      const int constraint_rows = duplicate_row ? kConstraintRows + 1 : kConstraintRows;
      Eigen::MatrixXd rows = Eigen::MatrixXd::Zero(constraint_rows, kDecisionColumns);
      rows.topRows(kDecisionColumns).setIdentity();
      rows.row(kDecisionColumns) << row_scale, 0.0;
      rows.row(kDecisionColumns + 1) << 1.0 / row_scale,
          kNearlyParallelCoupling / row_scale;
      Eigen::VectorXd lower = Eigen::VectorXd::Constant(constraint_rows, -kNearlyParallelSpeed);
      Eigen::VectorXd upper = Eigen::VectorXd::Constant(constraint_rows, kNearlyParallelSpeed);
      lower(kDecisionColumns) *= row_scale;
      upper(kDecisionColumns) = 0.0;
      lower(kDecisionColumns + 1) = 0.0;
      upper(kDecisionColumns + 1) /= row_scale;
      if (duplicate_row) {
        rows.bottomRows(1) = rows.row(kDecisionColumns + 1);
        lower.tail(1) = lower.segment(kDecisionColumns + 1, 1);
        upper.tail(1) = upper.segment(kDecisionColumns + 1, 1);
      }
      const Eigen::VectorXd zero = Eigen::VectorXd::Zero(kDecisionColumns);
      ASSERT_TRUE(((rows * zero).array() >= lower.array()).all());
      ASSERT_TRUE(((rows * zero).array() <= upper.array()).all());
      Eigen::VectorXd goal = zero;
      goal(1) = -kNearlyParallelSpeed;
      const Eigen::MatrixXd objective = Eigen::MatrixXd::Identity(kDecisionColumns, kDecisionColumns);
      VelocitySolverConfig config;
      config.certify_explicit_task_modes = true;
      config.epsilon = kTolerance;
      config.precision_threshold = kActiveRankPrecision;
      config.iteration_limit = kMaximumIterations;
      ObjectiveSolveConfig policy;
      policy.solve_mode = GetParam();
      policy.allow_min_error_fallback = false;
      const SolverResult result = computeMultiObjectiveVelocitySolutionEigen(
          {goal}, {objective}, rows, lower, upper, config, {policy});
      SCOPED_TRACE(::testing::Message() << "row_scale=" << row_scale
          << " duplicate=" << duplicate_row);
      ASSERT_TRUE(result.status == SolverStatus::kSuccess ||
                  result.status == SolverStatus::kNoProgress) << result.status_message;
      ASSERT_EQ(result.solution.size(), kDecisionColumns);
      const Eigen::Map<const Eigen::VectorXd> velocity(result.solution.data(), kDecisionColumns);
      EXPECT_LE(velocity.norm(), kTolerance);
      for (int row = 0; row < constraint_rows; ++row) {
        const double row_norm = rows.row(row).norm();
        const double value = rows.row(row).dot(velocity);
        EXPECT_LE(std::max(lower(row) - value, value - upper(row)) / row_norm, kTolerance);
      }
      // The opposite request lies inside the thin cone and must retain motion.
      goal(1) = kNearlyParallelSpeed;
      const SolverResult inward_result = computeMultiObjectiveVelocitySolutionEigen(
          {goal}, {objective}, rows, lower, upper, config, {policy});
      ASSERT_EQ(inward_result.status, SolverStatus::kSuccess);
      const Eigen::Map<const Eigen::VectorXd> inward_velocity(
          inward_result.solution.data(), kDecisionColumns);
      EXPECT_GE(inward_velocity(1), kNearlyParallelSpeed / kDecisionColumns);
      for (int row = 0; row < constraint_rows; ++row) {
        const double value = rows.row(row).dot(inward_velocity);
        EXPECT_LE(std::max(lower(row) - value, value - upper(row)) / rows.row(row).norm(),
                  kTolerance);
      }
    }
  }
}

INSTANTIATE_TEST_SUITE_P(AllPolicies, CertifiedNearlyParallelRows,
    ::testing::Values(TaskSolveMode::kMinError, TaskSolveMode::kScale,
                      TaskSolveMode::kScaleElastic));
} // namespace
} // namespace embodik::test

namespace embodik::test {
namespace {
constexpr int kCompletionDecisionColumns = 21;
constexpr double kCompletionGoal = 1000.0;
constexpr double kDependentTaskRowScale = 2.0;
constexpr double kWellSeparatedCompletionCoupling = 1e-3;
using CompletionCase = std::tuple<TaskSolveMode, double, bool, bool>;
class CertifiedRedundantCompletion : public ::testing::TestWithParam<CompletionCase> {};

TEST_P(CertifiedRedundantCompletion, NearParallelPinnedRowsRetainReachableScaledTask) {
  const TaskSolveMode mode = std::get<0>(GetParam());
  const double coupling = std::get<1>(GetParam());
  const bool dependent_task_row = std::get<2>(GetParam());
  const bool certified = std::get<3>(GetParam());
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
  config.certify_explicit_task_modes = certified;
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
  // Freeze the existing default collapse for the narrow cone; the opt-in fix
  // must not alter uncertified execution. Well-separated defaults still move.
  const bool full_scale_expected = certified || coupling == kWellSeparatedCompletionCoupling;
  const double expected_scale = full_scale_expected ? 1.0 : 0.0;
  EXPECT_NEAR(result.task_scales.front(), expected_scale, kTolerance);
  EXPECT_LE((objective * velocity - expected_scale * goal).norm(), kTolerance);
  EXPECT_LE((velocity - expected_scale * witness).norm(), kTolerance);
  for (int row = 0; row < kCompletionDecisionColumns; ++row) {
    const double value = rows.row(row).dot(velocity);
    EXPECT_LE(std::max(lower(row) - value, value - upper(row)) / rows.row(row).norm(),
              kTolerance);
  }
}

INSTANTIATE_TEST_SUITE_P(ScaledPolicies, CertifiedRedundantCompletion,
    ::testing::Combine(::testing::Values(TaskSolveMode::kScale, TaskSolveMode::kScaleElastic),
                       ::testing::Values(kNearlyParallelCoupling, kWellSeparatedCompletionCoupling),
                       ::testing::Bool(), ::testing::Bool()));
} // namespace
} // namespace embodik::test
