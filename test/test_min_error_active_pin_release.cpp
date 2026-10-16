#include <embodik/ik_baseline.hpp>
#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <limits>
#include <tuple>
#include <vector>

namespace embodik::test {
namespace {

constexpr double kTolerance = 1e-6;

struct Problem {
  Eigen::MatrixXd task = Eigen::MatrixXd::Zero(6, 2);
  Eigen::VectorXd goal = Eigen::VectorXd::Zero(6);
  Eigen::MatrixXd constraints = Eigen::MatrixXd::Zero(6, 2);
  Eigen::VectorXd lower = Eigen::VectorXd(6);
  Eigen::VectorXd upper = Eigen::VectorXd(6);
  VelocitySolverConfig config;

  Problem() {
    task(0, 0) = 10.0;
    task(3, 1) = 10.0;
    goal(0) = -10.0;
    goal(3) = 10.0;
    constraints.topRows(2).setIdentity();
    constraints.row(2).setConstant(0.5);
    constraints.row(3).setConstant(-0.5);
    lower << 0.99, -1.0, -0.4, -0.4, -0.4, -0.4;
    upper << 1.0, -0.85, std::sqrt(2e-4), 0.4, 0.4, 0.4;
    config.regularization_config.epsilon = 0.1;
    config.regularization_config.regularization_factor = 0.1;
  }

  Eigen::Vector2d optimum() const {
    return Eigen::Vector2d(0.99, 2.0 * std::sqrt(2e-4) - 0.99);
  }
};

ObjectiveSolveConfig min_error() {
  ObjectiveSolveConfig result;
  result.solve_mode = TaskSolveMode::kMinError;
  result.allow_min_error_fallback = false;
  return result;
}

SolverResult solve(bool generalized,
                   const std::vector<Eigen::VectorXd> &goals,
                   const std::vector<Eigen::MatrixXd> &tasks,
                   const Eigen::MatrixXd &constraints,
                   const Eigen::VectorXd &lower,
                   const Eigen::VectorXd &upper,
                   const VelocitySolverConfig &config) {
  std::vector<ObjectiveSolveConfig> policies(goals.size(), min_error());
  for (std::size_t index = 0; index < policies.size(); ++index)
    policies[index].priority = static_cast<int>(index);
  if (!generalized) {
    return computeMultiObjectiveVelocitySolutionEigen(
        goals, tasks, constraints, lower, upper, config, policies);
  }
  std::vector<Eigen::VectorXd> biases;
  for (const auto &goal : goals) biases.push_back(Eigen::VectorXd::Zero(goal.size()));
  return solveHierarchicalLinearSystemEigen(
      goals, biases, tasks, constraints, lower, upper, config, policies);
}

class MinErrorActivePinRelease
    : public ::testing::TestWithParam<std::tuple<bool, int>> {};

TEST_P(MinErrorActivePinRelease, FindsTheConstrainedMinimumAfterConflictingTrialPins) {
  const auto [generalized, variant] = GetParam();
  Problem problem;
  if (variant != 0) {
    const std::array<int, 6> order{2, 5, 0, 4, 1, 3};
    const std::array<double, 6> scales{4.0, 2.0, 0.5, 8.0, 0.25, 2.0};
    const Eigen::MatrixXd original = problem.constraints;
    const Eigen::VectorXd lower = problem.lower;
    const Eigen::VectorXd upper = problem.upper;
    for (int row = 0; row < 6; ++row) {
      const double scale = variant == 1 ? 1.0 : scales[row];
      problem.constraints.row(row) = scale * original.row(order[row]);
      problem.lower(row) = scale * lower(order[row]);
      problem.upper(row) = scale * upper(order[row]);
    }
  }
  const Eigen::Vector2d expected = problem.optimum();
  ASSERT_LE(detail::ComputeMaxLinearConstraintViolation(
                problem.constraints, problem.lower, problem.upper, expected),
            kTolerance);
  const auto result = solve(generalized, {problem.goal}, {problem.task},
                            problem.constraints, problem.lower, problem.upper,
                            problem.config);
  ASSERT_EQ(result.status, SolverStatus::kSuccess)
      << result.status_message << " solution="
      << ::testing::PrintToString(result.solution);
  ASSERT_EQ(result.solution.size(), 2U);
  const Eigen::Map<const Eigen::Vector2d> velocity(result.solution.data());
  EXPECT_LE(detail::ComputeMaxLinearConstraintViolation(
                problem.constraints, problem.lower, problem.upper, velocity),
            kTolerance);
  EXPECT_LE((velocity - expected).norm(), kTolerance);
  ASSERT_EQ(result.task_scales.size(), 1U);
  EXPECT_EQ(result.task_scales.front(), -1.0);
  ASSERT_EQ(result.task_modes_effective.size(), 1U);
  EXPECT_EQ(result.task_modes_effective.front(), TaskSolveMode::kMinError);
}

INSTANTIATE_TEST_SUITE_P(
    LegacyAndGeneralized, MinErrorActivePinRelease,
    ::testing::Combine(::testing::Bool(), ::testing::Values(0, 1, 2)));

class MinErrorActivePinHierarchy
    : public ::testing::TestWithParam<std::tuple<bool, bool, bool>> {};

TEST_P(MinErrorActivePinHierarchy, RecoveryPreservesEarlierRealizedTaskImages) {
  const Problem problem;
  const auto [generalized, duplicate_image, stationary_primary] = GetParam();
  Eigen::MatrixXd constraints = Eigen::MatrixXd::Zero(7, 3);
  constraints.topRows(3).setIdentity();
  constraints.block(3, 0, 4, 2) = problem.constraints.bottomRows(4);
  Eigen::VectorXd lower(7), upper(7);
  lower << problem.lower.head(2), -1.0, problem.lower.tail(4);
  upper << problem.upper.head(2), 1.0, problem.upper.tail(4);
  Eigen::MatrixXd primary = Eigen::MatrixXd::Zero(duplicate_image ? 2 : 1, 3);
  primary(0, 2) = 1.0;
  const double primary_rate = stationary_primary ? 0.0 : 0.25;
  Eigen::VectorXd target = Eigen::VectorXd::Constant(primary.rows(), primary_rate);
  if (duplicate_image) {
    primary(1, 2) = 2.0;
    target(1) = 2.0 * primary_rate;
  }
  Eigen::MatrixXd secondary = Eigen::MatrixXd::Zero(6, 3);
  secondary.leftCols(2) = problem.task;
  const auto result = solve(generalized, {target, problem.goal},
                            {primary, secondary}, constraints, lower, upper,
                            problem.config);
  ASSERT_EQ(result.status, SolverStatus::kSuccess)
      << result.status_message << " solution="
      << ::testing::PrintToString(result.solution);
  ASSERT_EQ(result.solution.size(), 3U);
  const Eigen::Map<const Eigen::Vector3d> velocity(result.solution.data());
  EXPECT_LE(detail::ComputeMaxLinearConstraintViolation(
                constraints, lower, upper, velocity), kTolerance);
  EXPECT_NEAR(velocity(2), primary_rate, kTolerance);
  EXPECT_LE((primary * velocity - target).norm(), kTolerance);
  EXPECT_LE((velocity.head<2>() - problem.optimum()).norm(), kTolerance);
}

INSTANTIATE_TEST_SUITE_P(LegacyAndGeneralized, MinErrorActivePinHierarchy,
                        ::testing::Combine(::testing::Bool(), ::testing::Bool(),
                                           ::testing::Bool()));

TEST(MinErrorActivePinAffine, GeneralizedRecoveryRetainsTheUnscaledBias) {
  const Problem problem;
  Eigen::VectorXd bias = Eigen::VectorXd::Zero(problem.goal.size());
  bias(0) = 1.0;
  bias(3) = -2.0;
  const auto result = solveHierarchicalLinearSystemEigen(
      {problem.goal + bias}, {bias}, {problem.task}, problem.constraints,
      problem.lower, problem.upper, problem.config, {min_error()});
  ASSERT_EQ(result.status, SolverStatus::kSuccess)
      << result.status_message << " solution="
      << ::testing::PrintToString(result.solution);
  ASSERT_EQ(result.solution.size(), 2U);
  const Eigen::Map<const Eigen::Vector2d> velocity(result.solution.data());
  EXPECT_LE((velocity - problem.optimum()).norm(), kTolerance);
  EXPECT_LE(detail::ComputeMaxLinearConstraintViolation(
                problem.constraints, problem.lower, problem.upper, velocity),
            kTolerance);
  ASSERT_EQ(result.task_errors.size(), 1U);
  EXPECT_NEAR(result.task_errors.front(),
              (problem.task * velocity - problem.goal).norm(), kTolerance);
  EXPECT_NEAR(result.final_error, result.task_errors.front(), kTolerance);
}

class MixedScaleMinErrorRecovery : public ::testing::TestWithParam<int> {};

TEST_P(MixedScaleMinErrorRecovery, RetainsTheFeasibleDirectionalPrefix) {
  // At z=.99 the hard rows require x in[.99,1], y in[-1,-.85]
  // and .5*(x+y)<=sqrt(.0002). The whole hard set also contains zero.
  const int variant = GetParam();
  Eigen::MatrixXd C = Eigen::MatrixXd::Zero(6, 3);
  C.topRows(3).setIdentity();
  C.row(3) << 1.0, 0.0, -1.0;
  C.row(4) << 0.0, 1.0, 1.0;
  C.row(5) << 0.5, 0.5, 0.0;
  Eigen::VectorXd lower(6), upper(6);
  lower << -1.0, -1.0, 0.0, 0.0, -2.0, -0.4;
  upper << 1.0, 1.0, 0.99, 0.01, 0.14, std::sqrt(2e-4);
  if (variant == 1 || variant == 2) {
    const std::array<int, 6> order{5, 3, 1, 4, 2, 0};
    const auto original_C = C.eval();
    const auto original_lower = lower.eval(), original_upper = upper.eval();
    for (int row = 0; row < 6; ++row) {
      const double scale = variant == 2 ? (row % 2 ? 0.25 : 4.0) : 1.0;
      C.row(row) = scale * original_C.row(order[row]);
      lower(row) = scale * original_lower(order[row]);
      upper(row) = scale * original_upper(order[row]);
    }
  }
  Eigen::MatrixXd primary = Eigen::MatrixXd::Zero(1, 3);
  primary(0, 2) = 1.0;
  Eigen::MatrixXd secondary = Eigen::MatrixXd::Zero(2, 3);
  secondary.leftCols(2) = 10.0 * Eigen::Matrix2d::Identity();
  Eigen::VectorXd goal(1), secondary_goal(2);
  goal << 1.98;
  secondary_goal << -10.0, 10.0;
  ObjectiveSolveConfig scale;
  scale.solve_mode = TaskSolveMode::kScale;
  scale.allow_min_error_fallback = false;
  auto secondary_policy = min_error();
  secondary_policy.priority = 1;
  VelocitySolverConfig config;
  config.epsilon = 1e-10;
  config.precision_threshold = 1e-12;
  config.iteration_limit = 50;
  config.regularization_config.epsilon = 1e-10;
  config.regularization_config.regularization_factor = 0.0;
  if (variant == 3) {
    config.regularization_config.epsilon = 0.1;
    config.regularization_config.regularization_factor = 0.1;
  }
  const auto result = detail::SolveGeneralizedHierarchicalLinearSystemEigen(
      {goal, secondary_goal}, {Eigen::VectorXd::Zero(1), Eigen::VectorXd::Zero(2)},
      {primary, secondary}, C, lower, upper, config, {scale, secondary_policy},
      nullptr, 1e-8);
  ASSERT_EQ(result.status, SolverStatus::kSuccess) << result.status_message;
  ASSERT_EQ(result.solution.size(), 3U);
  const Eigen::Map<const Eigen::Vector3d> v(result.solution.data());
  EXPECT_LE(detail::ComputeMaxLinearConstraintViolation(C, lower, upper, v), 1e-9);
  EXPECT_NEAR(v(0), 0.99, 1e-8);
  EXPECT_NEAR(v(1), 2.0 * std::sqrt(2e-4) - 0.99, 1e-8);
  EXPECT_NEAR(v(2), 0.99, 1e-9);
  ASSERT_EQ(result.task_scales.size(), 2U);
  EXPECT_NEAR(result.task_scales[0], 0.5, 1e-9);
  EXPECT_EQ(result.task_scales[1], -1.0);
  EXPECT_EQ(result.task_modes_effective[0], TaskSolveMode::kScale);
  EXPECT_EQ(result.task_modes_effective[1], TaskSolveMode::kMinError);
  EXPECT_FALSE(result.task_used_fallback[0]);
  EXPECT_FALSE(result.task_used_fallback[1]);
}

INSTANTIATE_TEST_SUITE_P(RowOrderAndScaling, MixedScaleMinErrorRecovery,
                        ::testing::Values(0, 1, 2, 3));

TEST(MixedPrefixMinErrorRecovery, RetainsCompletedMinimumErrorAndDirectionalTasks) {
  // A separate held coordinate precedes the same constrained x/y problem.
  // Recovery must protect its achieved image as well as the z task's ray.
  Eigen::MatrixXd C = Eigen::MatrixXd::Zero(7, 4);
  C.topRows(4).setIdentity();
  C.row(4) << 1.0, 0.0, -1.0, 0.0;
  C.row(5) << 0.0, 1.0, 1.0, 0.0;
  C.row(6) << 0.5, 0.5, 0.0, 0.0;
  Eigen::VectorXd lower(7), upper(7);
  lower << -1.0, -1.0, 0.0, -1.0, 0.0, -2.0, -0.4;
  upper << 1.0, 1.0, 0.99, 1.0, 0.01, 0.14, std::sqrt(2e-4);
  Eigen::MatrixXd held = Eigen::MatrixXd::Zero(1, 4);
  held(0, 3) = 1.0;
  Eigen::MatrixXd directional = Eigen::MatrixXd::Zero(1, 4);
  directional(0, 2) = 1.0;
  Eigen::MatrixXd lower_task = Eigen::MatrixXd::Zero(2, 4);
  lower_task.leftCols(2) = 10.0 * Eigen::Matrix2d::Identity();
  Eigen::VectorXd held_goal(1), directional_goal(1), lower_goal(2);
  held_goal << 0.25;
  directional_goal << 1.98;
  lower_goal << -10.0, 10.0;
  auto held_policy = min_error();
  ObjectiveSolveConfig directional_policy;
  directional_policy.priority = 1;
  directional_policy.solve_mode = TaskSolveMode::kScale;
  directional_policy.allow_min_error_fallback = false;
  auto lower_policy = min_error();
  lower_policy.priority = 2;
  VelocitySolverConfig config;
  config.epsilon = 1e-10;
  config.precision_threshold = 1e-12;
  config.iteration_limit = 50;
  config.regularization_config.epsilon = 0.1;
  config.regularization_config.regularization_factor = 0.1;
  const auto result = detail::SolveGeneralizedHierarchicalLinearSystemEigen(
      {held_goal, directional_goal, lower_goal},
      {Eigen::VectorXd::Zero(1), Eigen::VectorXd::Zero(1), Eigen::VectorXd::Zero(2)},
      {held, directional, lower_task}, C, lower, upper, config,
      {held_policy, directional_policy, lower_policy}, nullptr, 1e-8);
  ASSERT_EQ(result.status, SolverStatus::kSuccess) << result.status_message;
  ASSERT_EQ(result.solution.size(), 4U);
  const Eigen::Map<const Eigen::Vector4d> v(result.solution.data());
  EXPECT_LE(detail::ComputeMaxLinearConstraintViolation(C, lower, upper, v), config.epsilon);
  EXPECT_NEAR(v(3), 0.25, 1e-9);
  EXPECT_NEAR(v(2), 0.99, 1e-9);
  EXPECT_NEAR(v(0), 0.99, 1e-8);
  EXPECT_NEAR(v(1), 2.0 * std::sqrt(2e-4) - 0.99, 1e-8);
  ASSERT_EQ(result.task_scales.size(), 3U);
  EXPECT_EQ(result.task_scales[0], -1.0);
  EXPECT_NEAR(result.task_scales[1], 0.5, 1e-9);
  EXPECT_EQ(result.task_scales[2], -1.0);
  EXPECT_EQ(result.task_modes_effective[0], TaskSolveMode::kMinError);
  EXPECT_EQ(result.task_modes_effective[1], TaskSolveMode::kScale);
  EXPECT_EQ(result.task_modes_effective[2], TaskSolveMode::kMinError);
  for (bool fallback : result.task_used_fallback) EXPECT_FALSE(fallback);
}

struct SvdFactorFixture {
  Eigen::ComputationInfo status = Eigen::Success;
  Eigen::VectorXd values = Eigen::VectorXd::Ones(2);
  Eigen::MatrixXd left = Eigen::MatrixXd::Identity(2, 2);
  Eigen::MatrixXd right = Eigen::MatrixXd::Identity(2, 2);
  bool request_left = true, request_right = true;
  Eigen::ComputationInfo info() const { return status; }
  const Eigen::VectorXd &singularValues() const { return values; }
  bool computeU() const { return request_left; }
  bool computeV() const { return request_right; }
  const Eigen::MatrixXd &matrixU() const { return left; }
  const Eigen::MatrixXd &matrixV() const { return right; }
};

TEST(RegularizedInverseFactorGuard, RejectsInvalidFactorsDespiteSuccessStatus) {
  SvdFactorFixture factors;
  EXPECT_TRUE(detail::HasUsableSvdFactors(factors));
  factors.left(0, 0) = std::numeric_limits<double>::quiet_NaN();
  EXPECT_FALSE(detail::HasUsableSvdFactors(factors));
  factors.left.setIdentity();
  factors.right(0, 0) = std::numeric_limits<double>::infinity();
  EXPECT_FALSE(detail::HasUsableSvdFactors(factors));
  factors.request_right = false;
  EXPECT_TRUE(detail::HasUsableSvdFactors(factors));
  factors.values(0) = std::numeric_limits<double>::quiet_NaN();
  EXPECT_FALSE(detail::HasUsableSvdFactors(factors));
  factors.values.setOnes();
  factors.status = Eigen::NoConvergence;
  EXPECT_FALSE(detail::HasUsableSvdFactors(factors));
}

TEST(RegularizedInverseFactorGuard, FiniteProjectorKeepsExactCutoffAndImages) {
  Eigen::VectorXd direction(19);
  for (Eigen::Index index = 0; index < direction.size(); ++index)
    direction(index) = static_cast<double>(index + 1);
  direction.normalize();
  const Eigen::MatrixXd projector = direction * direction.transpose();
  RegularizedInverseConfig config;
  config.epsilon = 1e-10;
  config.regularization_factor = 0.0;
  const Eigen::MatrixXd inverse =
      linalg::ComputeRegularizedInverse(config, projector, nullptr, true);
  ASSERT_TRUE(inverse.allFinite());
  EXPECT_LE((projector * inverse * projector - projector).norm(), 1e-12);
  EXPECT_LE((inverse - projector).norm(), 1e-12);
}

VelocitySolverConfig affine_seed_config() {
  VelocitySolverConfig config;
  config.epsilon = 1e-10;
  config.precision_threshold = 1e-12;
  config.regularization_config.epsilon = 1e-10;
  config.regularization_config.regularization_factor = 0.0;
  return config;
}

TEST(PhaseOnePinRelease, ReleasesWrongPinOnlyUnderObjectiveNullDrift) {
  // The Phase-I slack is positive. A task-null numerical direction must not
  // hide a negative multiplier that prevents reducing that slack.
  Eigen::MatrixXd active(3, 4);
  active << 1.0, 0.0, 0.0, -1.0,
            0.0, 1.0, 0.0, -1.0,
            0.6, 0.8, 0.0, -1.0;
  Eigen::MatrixXd hessian = Eigen::MatrixXd::Zero(4, 4);
  hessian(3, 3) = 1.0;
  Eigen::VectorXd point(4), gradient(4), direction(4), multipliers(3);
  point << 0.0, 0.0, 0.0, 1.0;
  gradient = hessian * point;
  direction << 0.0, 0.0, 0.001, 0.0;
  multipliers << 1.5, 2.0, -2.5;
  const double epsilon = 1e-12;
  const auto select = [&](const Eigen::VectorXd &step,
                          const Eigen::VectorXd &dual) {
    return detail::FindObjectiveNullPhasePinToRelease(
        hessian, gradient, active, step, dual, epsilon, epsilon,
        100.0 * epsilon);
  };
  ASSERT_EQ(select(direction, multipliers), 2);
  EXPECT_LE((active * direction).norm(), epsilon);
  EXPECT_LE((active * point + Eigen::VectorXd::Ones(3)).norm(), epsilon);
  Eigen::VectorXd descent(4);
  descent << -1.0, -1.0, 0.0, -1.0;
  EXPECT_LE((active.topRows(2) * descent).norm(), epsilon);
  EXPECT_NEAR((active * descent)(2), -0.4, epsilon);
  EXPECT_LT(gradient.dot(descent), 0.0);

  auto objective_motion = direction;
  objective_motion(3) = 0.001;
  EXPECT_EQ(select(objective_motion, multipliers), -1);
  auto nonstationary = multipliers;
  nonstationary(2) = -2.4;
  EXPECT_EQ(select(direction, nonstationary), -1);
  auto nonfinite = multipliers;
  nonfinite(2) = std::numeric_limits<double>::quiet_NaN();
  EXPECT_EQ(select(direction, nonfinite), -1);
  auto invalid_direction = direction;
  invalid_direction(2) = std::numeric_limits<double>::infinity();
  EXPECT_EQ(select(invalid_direction, multipliers), -1);
  direction(2) = epsilon;
  EXPECT_EQ(select(direction, multipliers), -1);

  // Binary-exact arithmetic guards the strict negative-multiplier cutoff.
  const double tau = std::ldexp(1.0, -40);
  point(3) = 4.0 * tau;
  gradient = hessian * point;
  direction(2) = 0.001;
  multipliers << 6.0 * tau, 8.0 * tau, -10.0 * tau;
  EXPECT_EQ(detail::FindObjectiveNullPhasePinToRelease(
                hessian, gradient, active, direction, multipliers,
                tau, tau, 100.0 * tau), -1);
  Eigen::MatrixXd no_pins(0, 4);
  Eigen::VectorXd no_multipliers(0);
  EXPECT_EQ(detail::FindObjectiveNullPhasePinToRelease(
                hessian, gradient, no_pins, direction, no_multipliers,
                tau, tau, 100.0 * tau), -1);
  gradient.setZero();
  multipliers.setZero();
  EXPECT_EQ(select(direction, multipliers), -1);
}

TEST(AffineRayFeasibleSeed, PhaseOneDoesNotPinInitialInteriorSlack) {
  Eigen::MatrixXd C = Eigen::MatrixXd::Zero(5, 3);
  C.topRows(3).setIdentity();
  C.row(3) << 0.0, 1.0, -1.0;
  C.row(4) << 1.0, 0.0, 0.0;
  Eigen::VectorXd lower(5), upper(5), witness;
  lower << -1.0, 0.0, 0.0, -0.2, 0.2;
  upper << 1.0, 1.0, 1.0, -0.2, 0.2;
  ASSERT_EQ(detail::FindHardConstraintFeasibleWitness(
                C, lower, upper, affine_seed_config(), &witness),
            detail::HardConstraintFeasibilityStatus::kFeasible);
  // The Phase-I initial point has strictly positive slack. Treating its
  // near-boundary rows as equalities retains that offset instead of finding
  // the actual affine equations x=.2, y-alpha=-.2.
  EXPECT_LE(detail::ComputeMaxLinearConstraintViolation(C, lower, upper,
                                                        witness), 1e-13);
}

TEST(PhaseOneFullRank, CertifiesCoupledContradictionsWithoutChangingBounds) {
  // Each pair of upper rows implies x <= 0, whereas the last row requires
  // x >= 1. No scalar row alone can detect the contradiction. The phase
  // optimum has three independent active rows in (x, y, slack).
  for (const double coupling : {1e-2, 1e-4, 1e-6}) {
    for (const bool reverse_rows : {false, true}) {
      Eigen::MatrixXd C(3, 2);
      C << 1.0, coupling, 1.0, -coupling, 1.0, 0.0;
      Eigen::VectorXd lower(3), upper(3), witness;
      lower << -1000.0, -1000.0, 1.0;
      upper << 0.0, 0.0, 1000.0;
      if (reverse_rows) {
        C = C.colwise().reverse().eval();
        lower.reverseInPlace();
        upper.reverseInPlace();
      }
      VelocitySolverConfig config;
      config.epsilon = 1e-10;
      for (const bool release_pins : {false, true}) {
        EXPECT_EQ(detail::FindHardConstraintFeasibleWitnessImpl(
                      C, lower, upper, config, &witness, release_pins),
                  detail::HardConstraintFeasibilityStatus::kProvenInfeasible)
            << "coupling=" << coupling << " reversed=" << reverse_rows;
      }
      // Relaxing the two coupled upper bounds admits the explicit point
      // (1, 0). The same phase search must return a valid original-row witness.
      if (reverse_rows) {
        upper.tail(2).setConstant(2.0);
      } else {
        upper.head(2).setConstant(2.0);
      }
      ASSERT_EQ(detail::FindHardConstraintFeasibleWitness(
                    C, lower, upper, config, &witness),
                detail::HardConstraintFeasibilityStatus::kFeasible);
      EXPECT_LE(detail::ComputeMaxLinearConstraintViolation(
                    C, lower, upper, witness), config.epsilon);
    }
  }
}

TEST(AffineRayFeasibleSeed, GeneralizedSolverRecoversTheOriginalAffineDirection) {
  // This public bound-release fixture needs an unnecessary trial pin released.
  // Its affine shift also makes the old zero-acceleration seed invalid.
  Eigen::MatrixXd J(2, 4);
  J << -0.25, -0.16, -0.17, 0.0,
       -0.93, -0.93, 0.0, -0.1;
  Eigen::VectorXd target(2), lower(4), upper(4);
  target << -0.04, -0.02;
  lower << -1.0, 0.0, -1.0, -1.0;
  upper << 0.0, 1.0, 1.0, 0.0;
  const Eigen::VectorXd bias = -0.25 * target;
  ObjectiveSolveConfig policy;
  policy.solve_mode = TaskSolveMode::kScale;
  policy.allow_min_error_fallback = false;
  const auto result = detail::SolveGeneralizedHierarchicalLinearSystemEigen(
      {target}, {bias}, {J}, Eigen::Matrix4d::Identity(), lower, upper,
      affine_seed_config(), {policy}, nullptr, 1e-8);
  ASSERT_EQ(result.status, SolverStatus::kSuccess) << result.status_message;
  ASSERT_EQ(result.solution.size(), 4U);
  const Eigen::Map<const Eigen::Vector4d> value(result.solution.data());
  EXPECT_NEAR(result.task_scales[0], 1.0, 1e-9);
  EXPECT_LE((J * value + bias - target).norm(), 1e-9);
  EXPECT_LE(detail::ComputeMaxLinearConstraintViolation(
                Eigen::Matrix4d::Identity(), lower, upper, value), 1e-9);
  EXPECT_EQ(result.task_modes_effective[0], TaskSolveMode::kScale);
  EXPECT_FALSE(result.task_used_fallback[0]);
}

TEST(AffineRayFeasibleSeed, FindsDirectionalMotionForEitherAffineBiasSign) {
  for (double bias_value : {-0.5, 0.5}) {
    const Eigen::MatrixXd J = Eigen::MatrixXd::Identity(1, 1);
    const Eigen::VectorXd target = Eigen::VectorXd::Ones(1);
    const Eigen::VectorXd bias = Eigen::VectorXd::Constant(1, bias_value);
    const Eigen::MatrixXd prefix(0, 1);
    const Eigen::VectorXd empty(0), lower = Eigen::VectorXd::Zero(1);
    const Eigen::VectorXd upper = Eigen::VectorXd::Ones(1);
    Eigen::VectorXd result = Eigen::VectorXd::Constant(1, 99.0);
    double scale = 0.0;
    ASSERT_TRUE(detail::RecoverScaleTaskWithFeasibleRaySeed(
        J, target, bias, prefix, empty, J, lower, upper, lower,
        affine_seed_config(), &result, &scale));
    EXPECT_NEAR(scale, bias_value > 0.0 ? 1.0 : 0.5, 1e-10);
    EXPECT_LE((J * result + bias - scale * target).norm(), 1e-12);
    EXPECT_LE(detail::ComputeMaxLinearConstraintViolation(J, lower, upper,
                                                          result), 1e-12);
  }
}

TEST(AffineRayFeasibleSeed, PreservesTheTrustedPrefixImage) {
  Eigen::MatrixXd J(1, 2), prefix(1, 2);
  J << 0.0, 1.0;
  prefix << 1.0, 0.0;
  const Eigen::VectorXd target = Eigen::VectorXd::Ones(1);
  const Eigen::VectorXd bias = Eigen::VectorXd::Constant(1, 0.2);
  const Eigen::VectorXd prefix_value = Eigen::VectorXd::Constant(1, 0.2);
  const Eigen::MatrixXd C = Eigen::MatrixXd::Identity(2, 2);
  Eigen::VectorXd lower(2), upper(2), seed(2), result(2);
  lower << -1.0, 0.0;
  upper << 1.0, 1.0;
  seed << 0.2, 0.0;
  double scale = 0.0;
  ASSERT_TRUE(detail::RecoverScaleTaskWithFeasibleRaySeed(
      J, target, bias, prefix, prefix_value, C, lower, upper, seed,
      affine_seed_config(), &result, &scale));
  EXPECT_NEAR(scale, 1.0, 1e-10);
  EXPECT_NEAR(result(0), 0.2, 1e-12);
  EXPECT_NEAR(result(1), 0.8, 1e-10);
}

TEST(AffineRayFeasibleSeed, ConstructsASeedAtTheRayValidationPrecision) {
  // A hard-set witness at the caller's tolerance need not satisfy the tighter
  // affine-ray check. In particular, zero is not an exact zero-scale seed.
  const Eigen::MatrixXd J = Eigen::MatrixXd::Identity(1, 1), prefix(0, 1);
  const Eigen::VectorXd target = Eigen::VectorXd::Ones(1), empty(0);
  const Eigen::VectorXd lower = Eigen::VectorXd::Constant(1, -2.0);
  const Eigen::VectorXd upper = Eigen::VectorXd::Constant(1, 2.0);
  auto config = affine_seed_config();
  config.epsilon = 1e-8;
  for (double bias_value : {-1e-9, 1e-9}) {
    const Eigen::VectorXd bias = Eigen::VectorXd::Constant(1, bias_value);
    Eigen::VectorXd result = Eigen::VectorXd::Constant(1, 99.0);
    double scale = 0.0;
    ASSERT_TRUE(detail::RecoverScaleTaskWithFeasibleRaySeed(
        J, target, bias, prefix, empty, J, lower, upper,
        Eigen::VectorXd::Zero(1), config, &result, &scale));
    EXPECT_NEAR(scale, 1.0, 1e-10);
    EXPECT_LE((J * result + bias - scale * target).norm(), 1e-12);
    EXPECT_LE(detail::ComputeMaxLinearConstraintViolation(J, lower, upper,
                                                          result), 1e-12);
  }
}

TEST(AffineRayFeasibleSeed, NearOutOfRangeScaleDoesNotBecomeAcceptedMotion) {
  const Eigen::MatrixXd J = Eigen::MatrixXd::Identity(1, 1), prefix(0, 1);
  const Eigen::VectorXd target = Eigen::VectorXd::Ones(1), empty(0);
  const Eigen::VectorXd bound = Eigen::VectorXd::Zero(1);
  auto config = affine_seed_config();
  config.epsilon = 1e-8;
  for (double bias_value : {-1e-9, 1.0 + 1e-9}) {
    const Eigen::VectorXd bias = Eigen::VectorXd::Constant(1, bias_value);
    Eigen::VectorXd result = Eigen::VectorXd::Constant(1, 99.0);
    double scale = 0.75;
    EXPECT_FALSE(detail::RecoverScaleTaskWithFeasibleRaySeed(
        J, target, bias, prefix, empty, J, bound, bound, bound,
        config, &result, &scale));
    EXPECT_DOUBLE_EQ(result(0), 99.0);
    EXPECT_DOUBLE_EQ(scale, 0.75);
  }
}

TEST(AffineRayFeasibleSeed, AuditsProjectedScaleWithoutChangingPhysicalWitness) {
  const Eigen::Vector2d phase_point(1.0, -1e-13);
  Eigen::VectorXd candidate = Eigen::VectorXd::Constant(1, 99.0);
  double scale = 0.75;
  auto validate = [](const Eigen::VectorXd &value, double alpha) {
    return alpha >= 0.0 && alpha <= 1.0 && value(0) >= 1.0 &&
           value(0) <= 2.0 && std::abs(value(0) - 1.0 - alpha) <= 1e-10;
  };
  EXPECT_FALSE(detail::AcceptAffineRayPhaseSeed(
      phase_point, 1, false, validate, &candidate, &scale));
  EXPECT_DOUBLE_EQ(candidate(0), 99.0);
  EXPECT_DOUBLE_EQ(scale, 0.75);
  ASSERT_TRUE(detail::AcceptAffineRayPhaseSeed(
      phase_point, 1, true, validate, &candidate, &scale));
  EXPECT_DOUBLE_EQ(candidate(0), phase_point(0));
  EXPECT_DOUBLE_EQ(scale, 0.0);
}

TEST(AffineRayFeasibleSeed, ProjectedScaleStillRequiresOriginalValidator) {
  for (const Eigen::Vector2d &point : {
           Eigen::Vector2d(0.0, -1e-13), Eigen::Vector2d(0.0, 1.0 + 1e-13)}) {
    Eigen::VectorXd candidate = Eigen::VectorXd::Constant(1, 99.0);
    double scale = 0.75;
    bool validator_called = false;
    auto reject = [&](const Eigen::VectorXd &value, double alpha) {
      validator_called = true;
      EXPECT_DOUBLE_EQ(value(0), point(0));
      EXPECT_GE(alpha, 0.0);
      EXPECT_LE(alpha, 1.0);
      return false;
    };
    EXPECT_FALSE(detail::AcceptAffineRayPhaseSeed(
        point, 1, true, reject, &candidate, &scale));
    EXPECT_TRUE(validator_called);
    EXPECT_DOUBLE_EQ(candidate(0), 99.0);
    EXPECT_DOUBLE_EQ(scale, 0.75);
  }
}

TEST(AffineRayFeasibleSeed, InvalidTrialScaleDoesNotBlockAFeasibleSmallerScale) {
  const Eigen::MatrixXd J = Eigen::MatrixXd::Identity(1, 1), prefix(0, 1);
  const Eigen::VectorXd target = Eigen::VectorXd::Ones(1);
  const Eigen::VectorXd bias = Eigen::VectorXd::Constant(1, 0.5), empty(0);
  const Eigen::VectorXd lower = Eigen::VectorXd::Zero(1);
  const Eigen::VectorXd upper = Eigen::VectorXd::Constant(1, 0.1);
  Eigen::VectorXd result = Eigen::VectorXd::Constant(1, 99.0);
  double scale = 1.0;
  ASSERT_TRUE(detail::RecoverScaleTaskWithFeasibleRaySeed(
      J, target, bias, prefix, empty, J, lower, upper, lower,
      affine_seed_config(), &result, &scale, 1.0));
  EXPECT_NEAR(scale, 0.6, 1e-10);
  EXPECT_NEAR(result(0), 0.1, 1e-10);
}

TEST(AffineRayFeasibleSeed, AcceptsAFeasiblePhaseOnePointAlreadyAtTheScaleCap) {
  const Eigen::MatrixXd J = Eigen::MatrixXd::Identity(1, 1), prefix(0, 1);
  const Eigen::VectorXd target = Eigen::VectorXd::Ones(1), empty(0);
  const Eigen::VectorXd bound = Eigen::VectorXd::Constant(1, 0.5);
  Eigen::VectorXd result = Eigen::VectorXd::Constant(1, 99.0);
  double scale = 0.0;
  ASSERT_TRUE(detail::RecoverScaleTaskWithFeasibleRaySeed(
      J, target, bound, prefix, empty, J, bound, bound,
      Eigen::VectorXd::Zero(1), affine_seed_config(), &result, &scale));
  EXPECT_NEAR(scale, 1.0, 1e-10);
  EXPECT_NEAR(result(0), 0.5, 1e-10);
}

TEST(AffineRayFeasibleSeed, InfeasibleHardRayOrPrefixLeavesOutputsUntouched) {
  for (int variant = 0; variant < 3; ++variant) {
    const Eigen::MatrixXd J = Eigen::MatrixXd::Identity(1, 1);
    const Eigen::VectorXd target = Eigen::VectorXd::Ones(1);
    Eigen::MatrixXd C = J, prefix(0, 1);
    Eigen::VectorXd lower = Eigen::VectorXd::Zero(1), upper = lower;
    Eigen::VectorXd bias = Eigen::VectorXd::Constant(1, 2.0), prefix_value(0);
    if (variant == 1) {
      C = Eigen::MatrixXd::Ones(2, 1);
      lower.resize(2); upper.resize(2);
      lower << 0.0, 1.0;
      upper = lower;
      bias.setConstant(0.5);
    } else if (variant == 2) {
      prefix = J;
      prefix_value = Eigen::VectorXd::Constant(1, 0.5);
      bias.setConstant(0.2);
    }
    Eigen::VectorXd result = Eigen::VectorXd::Constant(1, 99.0);
    double scale = 0.75;
    EXPECT_FALSE(detail::RecoverScaleTaskWithFeasibleRaySeed(
        J, target, bias, prefix, prefix_value, C, lower, upper,
        Eigen::VectorXd::Zero(1), affine_seed_config(), &result, &scale));
    EXPECT_DOUBLE_EQ(result(0), 99.0);
    EXPECT_DOUBLE_EQ(scale, 0.75);
  }
}

TEST(AffineRayFeasibleSeed, ValidOriginalSeedRetainsTheEstablishedSearch) {
  const Eigen::MatrixXd J = Eigen::MatrixXd::Identity(1, 1), prefix(0, 1);
  const Eigen::VectorXd target = Eigen::VectorXd::Ones(1), empty(0);
  const Eigen::VectorXd zero = Eigen::VectorXd::Zero(1), upper = target;
  Eigen::VectorXd original = zero, repaired = zero;
  double original_scale = 0.0, repaired_scale = 0.0;
  const bool expected = detail::RecoverZeroScaleTask(
      J, target, zero, prefix, J, zero, upper, zero, &original, &original_scale);
  const bool actual = detail::RecoverScaleTaskWithFeasibleRaySeed(
      J, target, zero, prefix, empty, J, zero, upper, zero,
      affine_seed_config(), &repaired, &repaired_scale);
  EXPECT_EQ(actual, expected);
  EXPECT_DOUBLE_EQ(repaired_scale, original_scale);
  EXPECT_DOUBLE_EQ(repaired(0), original(0));
}

TEST(AffineRayFeasibleSeed, ValidPhaseOneSeedRetainsTheEstablishedSearch) {
  const Eigen::MatrixXd J = Eigen::MatrixXd::Identity(1, 1), prefix(0, 1);
  const Eigen::VectorXd target = Eigen::VectorXd::Ones(1), empty(0);
  const Eigen::VectorXd bias = Eigen::VectorXd::Constant(1, 0.5);
  const Eigen::VectorXd lower = Eigen::VectorXd::Zero(1), upper = target;
  Eigen::MatrixXd augmented(3, 2);
  augmented << 1.0, 0.0, 1.0, -1.0, 0.0, 1.0;
  Eigen::VectorXd phase_lower(3), phase_upper(3), phase_point;
  phase_lower << 0.0, -0.5, 0.0;
  phase_upper << 1.0, -0.5, 1.0;
  const auto config = affine_seed_config();
  ASSERT_EQ(detail::FindHardConstraintFeasibleWitness(
                augmented, phase_lower, phase_upper, config, &phase_point),
            detail::HardConstraintFeasibilityStatus::kFeasible);
  ASSERT_EQ(phase_point.size(), 2);
  Eigen::VectorXd expected, actual;
  double expected_scale = phase_point(1), actual_scale = 0.0;
  ASSERT_TRUE(detail::RecoverZeroScaleTask(
      J, target, bias, prefix, J, lower, upper, phase_point.head(1),
      &expected, &expected_scale, expected_scale));
  ASSERT_TRUE(detail::RecoverScaleTaskWithFeasibleRaySeed(
      J, target, bias, prefix, empty, J, lower, upper, lower, config,
      &actual, &actual_scale));
  EXPECT_DOUBLE_EQ(actual_scale, expected_scale);
  EXPECT_DOUBLE_EQ(actual(0), expected(0));
}


class AuthorizedMinErrorFallbackRetry : public ::testing::TestWithParam<bool> {};

TEST_P(AuthorizedMinErrorFallbackRetry, FailedFallbackRetainsCallerAuthorization) {
  Problem problem;
  problem.config.epsilon = 1e-6;
  problem.config.precision_threshold = 1e-12;
  problem.config.iteration_limit = 50;
  ObjectiveSolveConfig policy;
  policy.solve_mode = TaskSolveMode::kScale;
  policy.allow_min_error_fallback = GetParam();
  const auto result = detail::SolveGeneralizedHierarchicalLinearSystemEigen(
      {problem.goal}, {Eigen::VectorXd::Zero(6)}, {problem.task},
      problem.constraints, problem.lower, problem.upper, problem.config,
      {policy}, nullptr, 1e-8);
  ASSERT_EQ(result.task_modes_effective.size(), 1U);
  ASSERT_EQ(result.task_used_fallback.size(), 1U);
  if (!GetParam()) {
    // A negative x ray cannot enter the original positive x interval.
    EXPECT_NE(result.status, SolverStatus::kSuccess);
    EXPECT_EQ(result.task_modes_effective[0], TaskSolveMode::kScale);
    EXPECT_FALSE(result.task_used_fallback[0]);
    return;
  }
  ASSERT_EQ(result.status, SolverStatus::kSuccess) << result.status_message;
  ASSERT_EQ(result.solution.size(), 2U);
  const Eigen::Map<const Eigen::Vector2d> value(result.solution.data());
  EXPECT_LE(detail::ComputeMaxLinearConstraintViolation(
                problem.constraints, problem.lower, problem.upper, value), 1e-8);
  EXPECT_LE((value - problem.optimum()).norm(), 1e-8);
  EXPECT_EQ(result.task_modes_effective[0], TaskSolveMode::kMinError);
  EXPECT_TRUE(result.task_used_fallback[0]);
  ASSERT_EQ(result.task_scales.size(), 1U);
  EXPECT_EQ(result.task_scales[0], -1.0);
}

INSTANTIATE_TEST_SUITE_P(AllowedAndDenied, AuthorizedMinErrorFallbackRetry,
                        ::testing::Bool());

TEST(AuthorizedMinErrorFallbackRetry, RetainsAnEarlierStrictScaleImage) {
  Problem problem;
  problem.config.epsilon = 1e-6;
  problem.config.precision_threshold = 1e-12;
  problem.config.iteration_limit = 50;
  Eigen::MatrixXd C = Eigen::MatrixXd::Zero(7, 3);
  C.topLeftCorner(6, 2) = problem.constraints;
  C(6, 2) = 1.0;
  Eigen::VectorXd lower(7), upper(7);
  lower << problem.lower, -1.0;
  upper << problem.upper, 1.0;
  Eigen::MatrixXd primary = Eigen::MatrixXd::Zero(1, 3);
  primary(0, 2) = 1.0;
  const Eigen::VectorXd primary_goal = Eigen::VectorXd::Constant(1, 0.2);
  Eigen::MatrixXd secondary = Eigen::MatrixXd::Zero(6, 3);
  secondary.leftCols(2) = problem.task;
  ObjectiveSolveConfig strict_scale, allowed_fallback;
  strict_scale.solve_mode = TaskSolveMode::kScale;
  strict_scale.allow_min_error_fallback = false;
  allowed_fallback.solve_mode = TaskSolveMode::kScale;
  allowed_fallback.allow_min_error_fallback = true;
  allowed_fallback.priority = 1;
  const auto result = detail::SolveGeneralizedHierarchicalLinearSystemEigen(
      {primary_goal, problem.goal},
      {Eigen::VectorXd::Zero(1), Eigen::VectorXd::Zero(6)},
      {primary, secondary}, C, lower, upper, problem.config,
      {strict_scale, allowed_fallback}, nullptr, 1e-8);
  ASSERT_EQ(result.status, SolverStatus::kSuccess) << result.status_message;
  ASSERT_EQ(result.solution.size(), 3U);
  const Eigen::Map<const Eigen::Vector3d> value(result.solution.data());
  EXPECT_LE(detail::ComputeMaxLinearConstraintViolation(C, lower, upper, value), 1e-8);
  EXPECT_LE((value.head<2>() - problem.optimum()).norm(), 1e-8);
  EXPECT_NEAR(value(2), primary_goal(0), 1e-12);
  ASSERT_EQ(result.task_scales.size(), 2U);
  EXPECT_NEAR(result.task_scales[0], 1.0, 1e-12);
  EXPECT_EQ(result.task_scales[1], -1.0);
  ASSERT_EQ(result.task_modes_effective.size(), 2U);
  EXPECT_EQ(result.task_modes_effective[0], TaskSolveMode::kScale);
  EXPECT_EQ(result.task_modes_effective[1], TaskSolveMode::kMinError);
  ASSERT_EQ(result.task_used_fallback.size(), 2U);
  EXPECT_FALSE(result.task_used_fallback[0]);
  EXPECT_TRUE(result.task_used_fallback[1]);
}


class MinErrorIllConditionedTrialPinRelease
    : public ::testing::TestWithParam<bool> {};

TEST_P(MinErrorIllConditionedTrialPinRelease,
       ReleasesNegativeDualBeforeRejectingAnIllConditionedWorkingSet) {
  // The last row represents an earlier realized task image. Starting at the
  // x upper/y lower corner first activates both pins; x must then be released.
  // Its near dependence with the protected row makes the full-pin dual solve
  // inaccurate, even though a feasible minimum exists without the x pin.
  const double offset = GetParam() ? 0.25 : 0.0;
  const double coupling = 1e-10;
  const Eigen::MatrixXd H = Eigen::MatrixXd::Identity(3, 3);
  Eigen::VectorXd g(3);
  g << 1.0 + offset, 0.0, 1.0;
  Eigen::MatrixXd C = Eigen::MatrixXd::Zero(4, 3);
  C.topRows(3).setIdentity();
  C.row(3) << 1.0, 1.0, coupling;
  Eigen::VectorXd lower(4), upper(4);
  lower << -2.0 + offset, 0.0, -2.0, offset;
  upper << offset, 2.0, 2.0, offset;
  Eigen::VectorXd seed = Eigen::VectorXd::Zero(3), solution;
  seed(0) = offset;
  VelocitySolverConfig config;
  config.epsilon = 1e-10;
  config.iteration_limit = 50;
  unsigned int iterations = 0;
  ASSERT_TRUE(detail::RecoverFeasibleWeightedQuadratic(
      H, g, C, lower, upper, config, &solution, &iterations, &seed));
  ASSERT_EQ(solution.size(), 3);
  EXPECT_LE(detail::ComputeMaxLinearConstraintViolation(
                C, lower, upper, solution), 1e-10);
  EXPECT_NEAR(C.row(3).dot(solution), offset, 1e-12);
  const double expected_z = (1.0 - coupling) / (1.0 + coupling * coupling);
  EXPECT_NEAR(solution(0), offset - coupling * expected_z, 1e-9);
  EXPECT_NEAR(solution(1), 0.0, 1e-9);
  EXPECT_NEAR(solution(2), expected_z, 1e-9);
  EXPECT_GT(iterations, 0U);
}

INSTANTIATE_TEST_SUITE_P(ZeroAndNonzeroProtectedImage,
                        MinErrorIllConditionedTrialPinRelease,
                        ::testing::Bool());

class MinErrorWeakQuadraticConvergence
    : public ::testing::TestWithParam<double> {};

TEST(MinErrorDirectPrimal, PreservesPrimaryImageUnderCoupledHardBound) {
  std::vector<Eigen::MatrixXd> matrices{Eigen::Matrix2d::Identity(),
                                      Eigen::Matrix2d::Identity()};
  std::vector<Eigen::VectorXd> targets{Eigen::Vector2d(3.0, 3.0),
                                      Eigen::Vector2d(-1.0, 1.0)};
  std::vector<Eigen::VectorXd> biases(2, Eigen::Vector2d::Zero());
  Eigen::MatrixXd C(3, 2);
  C << 1.0, 0.0, 0.0, 1.0, 1.0, 1.0;
  Eigen::Vector3d lower(-1.0, -1.0, -0.5), upper(1.0, 1.0, 0.5);
  VelocitySolverConfig config;
  config.epsilon = 1e-10;
  std::vector<ObjectiveSolveConfig> policies(2);
  policies[0].solve_mode = policies[1].solve_mode = TaskSolveMode::kMinError;
  policies[1].priority = 1;
  detail::HierarchicalLinearSolverExecutionOptions execution;
  execution.prefer_direct_min_error_qp = true;
  auto result = solveHierarchicalLinearSystemEigen(
      targets, biases, matrices, C, lower, upper, config, policies, nullptr, execution);
  ASSERT_EQ(result.status, SolverStatus::kSuccess) << result.status_message;
  EXPECT_EQ(result.status_message, "opt-in direct hard-constrained minimum-error QP");
  Eigen::Map<const Eigen::VectorXd> point(result.solution.data(), result.solution.size());
  EXPECT_TRUE(point.isApprox(Eigen::Vector2d(0.25, 0.25), 1e-8));
  EXPECT_LE(detail::ComputeMaxLinearConstraintViolation(C, lower, upper, point), config.epsilon);
  EXPECT_EQ(result.task_modes_effective,
            (std::vector<TaskSolveMode>{TaskSolveMode::kMinError, TaskSolveMode::kMinError}));
  EXPECT_EQ(result.task_used_fallback, (std::vector<bool>{false, false}));
}

TEST(MinErrorQuadraticConvergence, RecoversCertifiedTaskNullRoundoff) {
  // This rank-one least-squares problem starts on its task image. Tiny ridge
  // curvature amplifies cancellation in the remaining null direction, so a
  // step-norm-only stopping test can exhaust its budget despite convergence.
  const Eigen::Vector2d row(0.4, 0.68);
  const double target = 0.13;
  Eigen::Matrix2d H = row * row.transpose();
  H.diagonal().array() += 1e-12 * H.diagonal().maxCoeff();
  const Eigen::Vector2d g = row * target;
  const Eigen::Matrix2d C = Eigen::Matrix2d::Identity();
  const Eigen::Vector2d lower = Eigen::Vector2d::Constant(-1000.0);
  const Eigen::Vector2d upper = Eigen::Vector2d::Constant(1000.0);
  const Eigen::VectorXd seed =
      (Eigen::Vector2d() << 0.12, (target - row(0) * 0.12) / row(1)).finished();
  Eigen::VectorXd solution;
  unsigned int iterations = 0;
  VelocitySolverConfig config;
  config.epsilon = 1e-10;
  config.iteration_limit = 50;
  unsigned int legacy_iterations = 999U;
  Eigen::VectorXd legacy_solution;
  (void)detail::RecoverFeasibleWeightedQuadratic(
      H, g, C, lower, upper, config, &legacy_solution, &legacy_iterations, &seed);
  // Whether a platform converges or exhausts its budget, actual work must be
  // reported rather than leaving the caller's iteration count unchanged.
  EXPECT_GT(legacy_iterations, 0U);
  EXPECT_NE(legacy_iterations, 999U);
  ASSERT_TRUE(detail::RecoverFeasibleWeightedQuadratic(
      H, g, C, lower, upper, config, &solution, &iterations, &seed, true));
  ASSERT_EQ(solution.size(), 2);
  // The feasible seed already achieves the task: certify it immediately
  // instead of spending the iteration budget resolving task-null roundoff.
  EXPECT_EQ(iterations, 1U);
  EXPECT_NEAR(row.dot(solution), target, 1e-10);
  EXPECT_LE((H * solution - g).norm(), 1e-10);
  EXPECT_LE(detail::ComputeMaxLinearConstraintViolation(
                C, lower, upper, solution), config.epsilon);
}

TEST(MinErrorQuadraticConvergence, InvalidInitialWitnessReportsNoIterations) {
  Eigen::MatrixXd identity = Eigen::MatrixXd::Identity(1, 1);
  Eigen::VectorXd zero = Eigen::VectorXd::Zero(1);
  Eigen::VectorXd lower = Eigen::VectorXd::Constant(1, -1.0);
  Eigen::VectorXd upper = Eigen::VectorXd::Constant(1, 1.0);
  Eigen::VectorXd invalid_seed = Eigen::VectorXd::Constant(1, 2.0);
  Eigen::VectorXd solution = Eigen::VectorXd::Constant(1, 99.0);
  unsigned int iterations = 999U;
  EXPECT_FALSE(detail::RecoverFeasibleWeightedQuadratic(
      identity, zero, identity, lower, upper, VelocitySolverConfig{},
      &solution, &iterations, &invalid_seed));
  EXPECT_EQ(iterations, 0U);
  EXPECT_DOUBLE_EQ(solution(0), 99.0);
}

TEST_P(MinErrorWeakQuadraticConvergence,
       SmallStationarityResidualDoesNotSkipUsefulProgress) {
  // Scaling the complete objective must not turn its nonoptimal feasible seed
  // into a solution just because its absolute gradient is small.
  Eigen::MatrixXd H(1, 1), C(1, 1);
  H(0, 0) = GetParam();
  C(0, 0) = 1.0;
  const Eigen::VectorXd g = Eigen::VectorXd::Constant(1, 1000.0 * GetParam());
  const Eigen::VectorXd lower = Eigen::VectorXd::Constant(1, -2000.0);
  const Eigen::VectorXd upper = Eigen::VectorXd::Constant(1, 2000.0);
  const Eigen::VectorXd seed = Eigen::VectorXd::Zero(1);
  Eigen::VectorXd solution;
  unsigned int iterations = 0;
  VelocitySolverConfig config;
  config.epsilon = 1e-10;
  ASSERT_TRUE(detail::RecoverFeasibleWeightedQuadratic(
      H, g, C, lower, upper, config, &solution, &iterations, &seed, true));
  ASSERT_EQ(solution.size(), 1);
  EXPECT_NEAR(solution(0), 1000.0, 1e-9);
  EXPECT_GE(iterations, 2U);
}

INSTANTIATE_TEST_SUITE_P(ObjectiveScale,
                        MinErrorWeakQuadraticConvergence,
                        ::testing::Values(1e-12, 1e-24));

TEST(MinErrorQuadraticConvergence, NegativeDualStillReleasesTrialBound) {
  Eigen::Matrix2d H;
  H << 2.0, 1.5, 1.5, 2.0;
  Eigen::Vector2d g(1.0, 4.0), seed(-0.99, -0.99);
  const Eigen::Matrix2d C = Eigen::Matrix2d::Identity();
  const Eigen::Vector2d lower = Eigen::Vector2d::Constant(-1.0);
  const Eigen::Vector2d upper = Eigen::Vector2d::Constant(1.0);
  Eigen::VectorXd solution;
  const Eigen::VectorXd dynamic_seed = seed;
  unsigned int iterations = 0;
  VelocitySolverConfig config;
  config.epsilon = 1e-10;
  ASSERT_TRUE(detail::RecoverFeasibleWeightedQuadratic(
      H, g, C, lower, upper, config, &solution, &iterations, &dynamic_seed, true));
  ASSERT_EQ(solution.size(), 2);
  EXPECT_NEAR(solution(0), -0.25, 1e-10);
  EXPECT_NEAR(solution(1), 1.0, 1e-10);
  EXPECT_LE(detail::ComputeMaxLinearConstraintViolation(
                C, lower, upper, solution), config.epsilon);
}


TEST(ObjectiveMinErrorRetryEligibility, RepairsAuthorizedZeroErrorHardFailure) {
  ObjectiveSolveConfig objective;
  objective.solve_mode = TaskSolveMode::kScale;
  objective.allow_min_error_fallback = true;
  VelocitySolverConfig solver;
  detail::ObjectiveMinErrorRetryState state{
      TaskSolveMode::kMinError, true, true, -1.0, 0.0, false};
  EXPECT_TRUE(detail::ShouldRetryObjectiveAsMinError(objective, solver, state));
  state.error = 0.5 * solver.precision_threshold;
  EXPECT_TRUE(detail::ShouldRetryObjectiveAsMinError(objective, solver, state));
}

TEST(ObjectiveMinErrorRetryEligibility, PreservesAuthorizationAndModeGuards) {
  ObjectiveSolveConfig objective;
  objective.solve_mode = TaskSolveMode::kScale;
  objective.allow_min_error_fallback = true;
  VelocitySolverConfig solver;
  detail::ObjectiveMinErrorRetryState state{
      TaskSolveMode::kMinError, true, true, -1.0, 0.0, false};
  state.final_bound_failure = false;
  EXPECT_FALSE(detail::ShouldRetryObjectiveAsMinError(objective, solver, state));
  state.final_bound_failure = true;
  objective.allow_min_error_fallback = false;
  EXPECT_FALSE(detail::ShouldRetryObjectiveAsMinError(objective, solver, state));
  objective.allow_min_error_fallback = true;
  objective.solve_mode = TaskSolveMode::kMinError;
  EXPECT_FALSE(detail::ShouldRetryObjectiveAsMinError(objective, solver, state));
  objective.solve_mode = TaskSolveMode::kScale;
  state.used_fallback = false;
  EXPECT_FALSE(detail::ShouldRetryObjectiveAsMinError(objective, solver, state));
  state.used_fallback = true;
  state.effective_mode = TaskSolveMode::kScale;
  EXPECT_FALSE(detail::ShouldRetryObjectiveAsMinError(objective, solver, state));
}

TEST(ObjectiveMinErrorRetryEligibility, RetainsExistingScaleAndErrorGuards) {
  ObjectiveSolveConfig objective;
  objective.solve_mode = TaskSolveMode::kScale;
  objective.allow_min_error_fallback = true;
  VelocitySolverConfig solver;
  detail::ObjectiveMinErrorRetryState state{
      TaskSolveMode::kScale, false, false, 0.0, 1.0, false};
  EXPECT_TRUE(detail::ShouldRetryObjectiveAsMinError(objective, solver, state));
  state.scale = 1.0;
  EXPECT_FALSE(detail::ShouldRetryObjectiveAsMinError(objective, solver, state));
  state.failed_requested_equality = true;
  EXPECT_TRUE(detail::ShouldRetryObjectiveAsMinError(objective, solver, state));
  state.error = 0.0;
  EXPECT_FALSE(detail::ShouldRetryObjectiveAsMinError(objective, solver, state));
  state.error = 1.0;
  // A strict lower SCALE objective remains strict even after a failed equality.
  objective.allow_min_error_fallback = false;
  EXPECT_FALSE(detail::ShouldRetryObjectiveAsMinError(objective, solver, state));
}

}  // namespace
}  // namespace embodik::test
