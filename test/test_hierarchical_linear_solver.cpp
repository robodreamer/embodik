#include <embodik/ik_baseline.hpp>

#include <Eigen/Dense>
#include <gtest/gtest.h>

#include <cmath>
#include <initializer_list>
#include <limits>
#include <random>
#include <string>
#include <vector>

namespace embodik::test {
namespace {

constexpr double kTolerance = 1e-8;

Eigen::VectorXd vector(std::initializer_list<double> values) {
  Eigen::VectorXd result(static_cast<Eigen::Index>(values.size()));
  Eigen::Index index = 0;
  for (double value : values) {
    result(index++) = value;
  }
  return result;
}

Eigen::MatrixXd matrix(std::initializer_list<std::initializer_list<double>> rows) {
  const Eigen::Index row_count = static_cast<Eigen::Index>(rows.size());
  const Eigen::Index column_count =
      rows.size() == 0 ? 0 : static_cast<Eigen::Index>(rows.begin()->size());
  Eigen::MatrixXd result(row_count, column_count);
  Eigen::Index row_index = 0;
  for (const auto &row : rows) {
    EXPECT_EQ(static_cast<Eigen::Index>(row.size()), column_count);
    Eigen::Index column_index = 0;
    for (double value : row) {
      result(row_index, column_index++) = value;
    }
    ++row_index;
  }
  return result;
}

VelocitySolverConfig exact_config() {
  VelocitySolverConfig config;
  config.epsilon = 1e-10;
  config.precision_threshold = 1e-12;
  config.iteration_limit = 50;
  config.magnitude_limit = 1e10;
  config.stall_detection_count = 2;
  config.regularization_config.epsilon = 1e-10;
  config.regularization_config.regularization_factor = 0.0;
  return config;
}

Eigen::VectorXd solution_vector(const SolverResult &result) {
  if (result.solution.empty()) {
    return Eigen::VectorXd();
  }
  return Eigen::Map<const Eigen::VectorXd>(result.solution.data(),
                                           result.solution.size());
}

void expect_double_vectors_near(const std::vector<double> &actual,
                                const std::vector<double> &expected,
                                double tolerance = kTolerance) {
  ASSERT_EQ(actual.size(), expected.size());
  for (std::size_t index = 0; index < actual.size(); ++index) {
    EXPECT_NEAR(actual[index], expected[index], tolerance) << index;
  }
}

void expect_results_equal_except_timing(const SolverResult &actual,
                                        const SolverResult &expected) {
  EXPECT_EQ(actual.status, expected.status);
  expect_double_vectors_near(actual.solution, expected.solution);
  EXPECT_EQ(actual.iterations, expected.iterations);
  EXPECT_NEAR(actual.final_error, expected.final_error, kTolerance);
  expect_double_vectors_near(actual.task_scales, expected.task_scales);
  expect_double_vectors_near(actual.task_errors, expected.task_errors);
  EXPECT_EQ(actual.task_modes_effective, expected.task_modes_effective);
  EXPECT_EQ(actual.task_used_fallback, expected.task_used_fallback);
  EXPECT_EQ(actual.status_message, expected.status_message);
  if (std::isinf(actual.condition_number) ||
      std::isinf(expected.condition_number)) {
    EXPECT_EQ(std::isinf(actual.condition_number),
              std::isinf(expected.condition_number));
  } else {
    EXPECT_NEAR(actual.condition_number, expected.condition_number,
                kTolerance);
  }
}

SolverResult solve_canonical(
    const std::vector<Eigen::VectorXd> &scalable_targets_b_prime,
    const std::vector<Eigen::VectorXd> &affine_biases_b_double_prime,
    const std::vector<Eigen::MatrixXd> &objective_matrices,
    const Eigen::MatrixXd &constraint_matrix,
    const Eigen::VectorXd &lower_bounds,
    const Eigen::VectorXd &upper_bounds,
    const std::vector<ObjectiveSolveConfig> &objective_configs = {},
    const Eigen::VectorXd *softening_factors = nullptr) {
  return solveHierarchicalLinearSystemEigen(
      scalable_targets_b_prime, affine_biases_b_double_prime,
      objective_matrices, constraint_matrix, lower_bounds, upper_bounds,
      exact_config(), objective_configs, softening_factors);
}

TEST(HierarchicalLinearSolverTest,
     LegacyVelocityWrapperMatchesCanonicalZeroBiasResult) {
  const std::vector<Eigen::VectorXd> targets{vector({0.4}), vector({-0.2})};
  const std::vector<Eigen::VectorXd> zero_biases{vector({0.0}),
                                                 vector({0.0})};
  const std::vector<Eigen::MatrixXd> objectives{matrix({{1.0, 0.0}}),
                                                matrix({{0.0, 1.0}})};
  const Eigen::MatrixXd constraints = Eigen::Matrix2d::Identity();
  const Eigen::VectorXd lower = vector({-1.0, -1.0});
  const Eigen::VectorXd upper = vector({1.0, 1.0});
  const auto config = exact_config();

  const auto legacy = computeMultiObjectiveVelocitySolutionEigen(
      targets, objectives, constraints, lower, upper, config, {}, nullptr);
  const auto canonical = solveHierarchicalLinearSystemEigen(
      targets, zero_biases, objectives, constraints, lower, upper, config, {},
      nullptr);

  expect_results_equal_except_timing(canonical, legacy);
}

TEST(HierarchicalLinearSolverTest, OneObjectiveFastPathReturnsFullScale) {
  const auto result = solve_canonical(
      {vector({0.25, -0.5})}, {vector({0.0, 0.0})},
      {Eigen::Matrix2d::Identity()}, Eigen::Matrix2d::Identity(),
      vector({-1.0, -1.0}), vector({1.0, 1.0}));

  ASSERT_EQ(result.status, SolverStatus::kSuccess) << result.status_message;
  EXPECT_EQ(result.iterations, 1U);
  EXPECT_TRUE(solution_vector(result).isApprox(vector({0.25, -0.5}),
                                                kTolerance));
  ASSERT_EQ(result.task_scales.size(), 1U);
  EXPECT_NEAR(result.task_scales[0], 1.0, kTolerance);
  ASSERT_EQ(result.task_errors.size(), 1U);
  EXPECT_NEAR(result.task_errors[0], 0.0, kTolerance);
  EXPECT_EQ(result.task_modes_effective,
            std::vector<TaskSolveMode>{TaskSolveMode::kScale});
  EXPECT_EQ(result.task_used_fallback, std::vector<bool>{false});
}

TEST(HierarchicalLinearSolverTest,
     InconsistentTallScaleObjectiveIsNotReportedAsFullScaleSuccess) {
  const auto result = solve_canonical(
      {vector({1.0, -1.0})}, {Eigen::Vector2d::Zero()},
      {matrix({{1.0}, {1.0}})}, matrix({{1.0}}), vector({-10.0}),
      vector({10.0}));

  const bool false_full_scale_success =
      result.status == SolverStatus::kSuccess &&
      result.task_scales.size() == 1U &&
      std::abs(result.task_scales[0] - 1.0) <= kTolerance &&
      result.task_errors.size() == 1U && result.task_errors[0] > 1.0;
  EXPECT_FALSE(false_full_scale_success);
}

TEST(HierarchicalLinearSolverTest,
     DenseArbitraryConstraintRowLimitsTaskDirection) {
  const Eigen::MatrixXd constraints = matrix({{1.0, 1.0}, {1.0, -1.0}});
  const Eigen::VectorXd lower = vector({-10.0, -10.0});
  const Eigen::VectorXd upper = vector({0.9, 10.0});
  const auto result = solve_canonical(
      {vector({0.8, 0.4})}, {vector({0.0, 0.0})},
      {Eigen::Matrix2d::Identity()}, constraints, lower, upper);

  ASSERT_EQ(result.status, SolverStatus::kSuccess) << result.status_message;
  const Eigen::VectorXd solution = solution_vector(result);
  ASSERT_EQ(result.task_scales.size(), 1U);
  EXPECT_NEAR(result.task_scales[0], 0.75, kTolerance);
  EXPECT_TRUE(solution.isApprox(vector({0.6, 0.3}), kTolerance));
  const Eigen::VectorXd values = constraints * solution;
  EXPECT_TRUE((values.array() >= lower.array() - kTolerance).all());
  EXPECT_TRUE((values.array() <= upper.array() + kTolerance).all());
}

TEST(HierarchicalLinearSolverTest, BindingBoxConstraintReportsNonzeroScale) {
  const auto result = solve_canonical(
      {vector({2.0, 1.0})}, {vector({0.0, 0.0})},
      {Eigen::Matrix2d::Identity()}, Eigen::Matrix2d::Identity(),
      vector({-10.0, -10.0}), vector({1.0, 10.0}));

  ASSERT_EQ(result.status, SolverStatus::kSuccess) << result.status_message;
  ASSERT_EQ(result.task_scales.size(), 1U);
  EXPECT_NEAR(result.task_scales[0], 0.5, kTolerance);
  EXPECT_TRUE(solution_vector(result).isApprox(vector({1.0, 0.5}),
                                                kTolerance));
}

TEST(HierarchicalLinearSolverTest,
     FioreAffineBiasIsSubtractedButNeverTaskScaled) {
  const Eigen::VectorXd scalable_b_prime = vector({2.0, 1.0});
  const Eigen::VectorXd positive_b_double_prime = vector({0.5, -0.25});
  const auto result = solve_canonical(
      {scalable_b_prime}, {positive_b_double_prime},
      {Eigen::Matrix2d::Identity()}, Eigen::Matrix2d::Identity(),
      vector({-10.0, -10.0}), vector({0.5, 10.0}));

  ASSERT_EQ(result.status, SolverStatus::kSuccess) << result.status_message;
  ASSERT_EQ(result.task_scales.size(), 1U);
  const double scale = result.task_scales[0];
  EXPECT_NEAR(scale, 0.5, kTolerance);

  const Eigen::VectorXd solution = solution_vector(result);
  EXPECT_TRUE(solution.isApprox(vector({0.5, 0.75}), kTolerance));
  EXPECT_TRUE((solution + positive_b_double_prime)
                  .isApprox(scale * scalable_b_prime, kTolerance));
}

TEST(HierarchicalLinearSolverTest,
     LowerPriorityTaskUsesOnlyHigherPriorityNullSpace) {
  const auto result = solve_canonical(
      {vector({0.6}), vector({-0.2})}, {vector({0.0}), vector({0.0})},
      {matrix({{1.0, 0.0}}), matrix({{1.0, 1.0}})},
      Eigen::Matrix2d::Identity(), vector({-1.0, -1.0}),
      vector({1.0, 1.0}));

  ASSERT_EQ(result.status, SolverStatus::kSuccess) << result.status_message;
  const Eigen::VectorXd solution = solution_vector(result);
  EXPECT_NEAR(solution(0), 0.6, kTolerance);
  EXPECT_NEAR(solution.sum(), -0.2, kTolerance);
  ASSERT_EQ(result.task_scales.size(), 2U);
  EXPECT_NEAR(result.task_scales[0], 1.0, kTolerance);
  EXPECT_NEAR(result.task_scales[1], 1.0, kTolerance);
}

TEST(HierarchicalLinearSolverTest, MinErrorReturnsBoundedLeastResidualSolution) {
  ObjectiveSolveConfig min_error;
  min_error.solve_mode = TaskSolveMode::kMinError;
  min_error.allow_min_error_fallback = false;
  const auto result = solve_canonical(
      {vector({2.0, 1.0})}, {vector({0.0, 0.0})},
      {Eigen::Matrix2d::Identity()}, Eigen::Matrix2d::Identity(),
      vector({0.0, -10.0}), vector({0.0, 10.0}), {min_error});

  ASSERT_EQ(result.status, SolverStatus::kSuccess) << result.status_message;
  EXPECT_TRUE(solution_vector(result).isApprox(vector({0.0, 1.0}),
                                                kTolerance));
  EXPECT_EQ(result.task_modes_effective,
            std::vector<TaskSolveMode>{TaskSolveMode::kMinError});
  EXPECT_EQ(result.task_used_fallback, std::vector<bool>{false});
  ASSERT_EQ(result.task_scales.size(), 1U);
  EXPECT_NEAR(result.task_scales[0], -1.0, kTolerance);
  ASSERT_EQ(result.task_errors.size(), 1U);
  EXPECT_NEAR(result.task_errors[0], 2.0, kTolerance);
}

TEST(HierarchicalLinearSolverTest, ScaleModeCanFallBackToMinError) {
  ObjectiveSolveConfig scale_with_fallback;
  scale_with_fallback.solve_mode = TaskSolveMode::kScale;
  scale_with_fallback.allow_min_error_fallback = true;
  const auto result = solve_canonical(
      {vector({1.0, 1.0})}, {vector({0.0, 0.0})},
      {Eigen::Matrix2d::Identity()}, Eigen::Matrix2d::Identity(),
      vector({-1e-12, -1.0}), vector({1e-12, 1.0}),
      {scale_with_fallback});

  ASSERT_EQ(result.status, SolverStatus::kSuccess) << result.status_message;
  EXPECT_TRUE(solution_vector(result).isApprox(vector({0.0, 1.0}),
                                                kTolerance));
  EXPECT_EQ(result.task_modes_effective,
            std::vector<TaskSolveMode>{TaskSolveMode::kMinError});
  EXPECT_EQ(result.task_used_fallback, std::vector<bool>{true});
}

TEST(HierarchicalLinearSolverTest,
     AllOnesSofteningPointerPreservesNullPointerSolution) {
  const std::vector<Eigen::VectorXd> targets{vector({0.25})};
  const std::vector<Eigen::VectorXd> biases{vector({0.0})};
  const std::vector<Eigen::MatrixXd> objectives{matrix({{1.0}})};
  const Eigen::MatrixXd constraints = matrix({{1.0}});
  const Eigen::VectorXd lower = vector({-1.0});
  const Eigen::VectorXd upper = vector({1.0});

  ObjectiveSolveConfig scale_only;
  scale_only.solve_mode = TaskSolveMode::kScale;
  scale_only.allow_min_error_fallback = false;
  const auto strict = solve_canonical(targets, biases, objectives, constraints,
                                      lower, upper, {scale_only}, nullptr);

  const Eigen::VectorXd softening = vector({1.0});
  const auto softened = solve_canonical(targets, biases, objectives, constraints,
                                        lower, upper, {scale_only}, &softening);

  ASSERT_EQ(strict.status, SolverStatus::kSuccess) << strict.status_message;
  ASSERT_EQ(softened.status, SolverStatus::kSuccess)
      << softened.status_message;
  ASSERT_EQ(strict.task_scales.size(), 1U);
  ASSERT_EQ(softened.task_scales.size(), 1U);
  EXPECT_TRUE(solution_vector(softened).isApprox(solution_vector(strict),
                                                  kTolerance));
  EXPECT_NEAR(softened.task_scales[0], strict.task_scales[0], kTolerance);
  EXPECT_EQ(softened.task_modes_effective, strict.task_modes_effective);
  EXPECT_EQ(softened.task_used_fallback, strict.task_used_fallback);
  EXPECT_EQ(strict.iterations, 1U);
}

TEST(HierarchicalLinearSolverTest, InvalidDimensionsReturnSpecificStatuses) {
  const Eigen::MatrixXd constraints = Eigen::Matrix2d::Identity();
  const Eigen::VectorXd lower = vector({-1.0, -1.0});
  const Eigen::VectorXd upper = vector({1.0, 1.0});

  const auto wrong_bias_count = solve_canonical(
      {vector({0.0})}, {vector({0.0}), vector({0.0})},
      {matrix({{1.0, 0.0}})}, constraints, lower, upper);
  EXPECT_EQ(wrong_bias_count.status, SolverStatus::kShapeMismatch);
  EXPECT_FALSE(wrong_bias_count.status_message.empty());

  const auto missing_bias = solve_canonical(
      {vector({0.0})}, {}, {matrix({{1.0, 0.0}})}, constraints, lower,
      upper);
  EXPECT_EQ(missing_bias.status, SolverStatus::kShapeMismatch);
  EXPECT_FALSE(missing_bias.status_message.empty());

  const auto wrong_bias_rows = solve_canonical(
      {vector({0.0})}, {vector({0.0, 0.0})},
      {matrix({{1.0, 0.0}})}, constraints, lower, upper);
  EXPECT_EQ(wrong_bias_rows.status, SolverStatus::kShapeMismatch);
  EXPECT_FALSE(wrong_bias_rows.status_message.empty());

  const auto wrong_bounds = solve_canonical(
      {vector({0.0})}, {vector({0.0})}, {matrix({{1.0, 0.0}})},
      constraints, vector({-1.0}), upper);
  EXPECT_EQ(wrong_bounds.status, SolverStatus::kConstraintBoundsMismatch);
  EXPECT_FALSE(wrong_bounds.status_message.empty());
}

TEST(HierarchicalLinearSolverTest,
     NonFiniteAndReversedInputsAreRejectedBeforeSolving) {
  const Eigen::VectorXd zero = vector({0.0});
  const Eigen::MatrixXd identity = matrix({{1.0}});
  const Eigen::VectorXd lower = vector({-1.0});
  const Eigen::VectorXd upper = vector({1.0});

  Eigen::MatrixXd nonfinite_constraint = identity;
  nonfinite_constraint(0, 0) = std::numeric_limits<double>::quiet_NaN();
  EXPECT_EQ(solve_canonical({zero}, {zero}, {identity}, nonfinite_constraint,
                            lower, upper)
                .status,
            SolverStatus::kNonFiniteInput);

  Eigen::VectorXd nonfinite_target = zero;
  nonfinite_target(0) = std::numeric_limits<double>::infinity();
  const SolverResult nonfinite_target_result =
      solve_canonical({nonfinite_target}, {zero}, {identity}, identity, lower,
                      upper);
  EXPECT_EQ(nonfinite_target_result.status, SolverStatus::kNonFiniteInput);
  ASSERT_EQ(nonfinite_target_result.solution.size(), 1U);
  EXPECT_DOUBLE_EQ(nonfinite_target_result.solution.front(), 0.0);

  Eigen::VectorXd nonfinite_upper = upper;
  nonfinite_upper(0) = std::numeric_limits<double>::quiet_NaN();
  EXPECT_EQ(solve_canonical({zero}, {zero}, {identity}, identity, lower,
                            nonfinite_upper)
                .status,
            SolverStatus::kNonFiniteInput);

  EXPECT_EQ(solve_canonical({zero}, {zero}, {identity}, identity,
                            vector({1.0}), vector({-1.0}))
                .status,
            SolverStatus::kInvalidInput);
}

TEST(HierarchicalLinearSolverTest,
     FiniteInputsThatGenerateNonFiniteStateReturnNumericalError) {
  VelocitySolverConfig config = exact_config();
  config.epsilon = 0.0;
  config.regularization_config.epsilon = 0.0;
  config.regularization_config.regularization_factor = 0.0;

  const auto result = solveHierarchicalLinearSystemEigen(
      {vector({1.0})}, {vector({0.0})}, {matrix({{1e-309}})},
      matrix({{0.0}}), vector({0.0}), vector({0.0}), config);

  EXPECT_EQ(result.status, SolverStatus::kNumericalError);
  EXPECT_EQ(result.status_message,
            "non-finite values generated in solver state");
  ASSERT_EQ(result.solution.size(), 1U);
  EXPECT_EQ(result.solution[0], 0.0);
  ASSERT_EQ(result.task_scales.size(), 1U);
  EXPECT_EQ(result.task_scales[0], 0.0);
}

TEST(HierarchicalLinearSolverTest,
     FeasibleScaleRangeTracksBothEndpointsAndEmptyIntervals) {
  const auto positive =
      linalg::ComputeGeneralizedFeasibleScalingRange(0.4, 1.2, 2.0);
  EXPECT_NEAR(positive.second, 0.2, kTolerance);
  EXPECT_NEAR(positive.first, 0.6, kTolerance);

  const auto negative =
      linalg::ComputeGeneralizedFeasibleScalingRange(-1.2, -0.4, -2.0);
  EXPECT_NEAR(negative.second, 0.2, kTolerance);
  EXPECT_NEAR(negative.first, 0.6, kTolerance);

  const auto empty =
      linalg::ComputeGeneralizedFeasibleScalingRange(1.2, 2.0, 1.0);
  EXPECT_GT(empty.second, empty.first);
}

TEST(HierarchicalLinearSolverTest,
     ProvenInfeasibleHardSetReturnsLegacyHoldDiagnostics) {
  const auto result = solve_canonical(
      {vector({1.0})}, {vector({0.0})}, {matrix({{1.0, 0.0}})},
      matrix({{0.0, 0.0}}), vector({1.0}), vector({2.0}));

  ASSERT_EQ(result.status, SolverStatus::kInfeasible);
  EXPECT_TRUE(solution_vector(result).isZero(kTolerance));
  EXPECT_EQ(result.task_scales, std::vector<double>{0.0});
  EXPECT_EQ(result.task_modes_effective,
            std::vector<TaskSolveMode>{TaskSolveMode::kScale});
  EXPECT_EQ(result.task_used_fallback, std::vector<bool>{false});
}

TEST(HierarchicalLinearSolverTest,
     PhaseOneIsInvariantToDownwardConstraintRowScaling) {
  const auto unscaled = solve_canonical(
      {vector({1.0})}, {vector({0.0})}, {matrix({{1.0}})}, matrix({{1.0}}),
      vector({1.0}), vector({2.0}));
  const auto downscaled = solve_canonical(
      {vector({1.0})}, {vector({0.0})}, {matrix({{1.0}})},
      matrix({{1e-9}}), vector({1e-9}), vector({2e-9}));
  const auto deeply_downscaled = solve_canonical(
      {vector({1.0})}, {vector({0.0})}, {matrix({{1.0}})},
      matrix({{1e-12}}), vector({1e-12}), vector({2e-12}));

  ASSERT_EQ(unscaled.status, SolverStatus::kSuccess)
      << unscaled.status_message;
  ASSERT_EQ(downscaled.status, SolverStatus::kSuccess)
      << downscaled.status_message;
  ASSERT_EQ(deeply_downscaled.status, SolverStatus::kSuccess)
      << deeply_downscaled.status_message;
  EXPECT_TRUE(solution_vector(downscaled).isApprox(solution_vector(unscaled),
                                                     kTolerance));
  EXPECT_TRUE(solution_vector(deeply_downscaled)
                  .isApprox(solution_vector(unscaled), kTolerance));
}

TEST(HierarchicalLinearSolverTest,
     DenseContradictoryHardRowsAreCertifiedInfeasible) {
  const auto result = solve_canonical(
      {vector({0.0})}, {vector({0.0})}, {matrix({{1.0}})},
      matrix({{1.0}, {1.0}}), vector({1.0, -2.0}), vector({2.0, 0.0}));

  EXPECT_EQ(result.status, SolverStatus::kInfeasible)
      << result.status_message;
}

TEST(HierarchicalLinearSolverTest,
     WellConditionedCoupledContradictionProducesFarkasCertificate) {
  const auto result = solve_canonical(
      {vector({0.0, 0.0})}, {vector({0.0, 0.0})},
      {Eigen::Matrix2d::Identity()},
      matrix({{1.0, 0.0}, {0.0, 1.0}, {2.0, 0.5}, {0.5, 1.0}}),
      vector({-0.1, -0.1, -31.0, -11.0}),
      vector({0.1, 0.1, -29.0, -9.0}));

  EXPECT_EQ(result.status, SolverStatus::kInfeasible)
      << result.status_message;
}

TEST(HierarchicalLinearSolverTest,
     InfeasibleMinErrorAndAllOnesSofteningKeepCompleteDiagnostics) {
  ObjectiveSolveConfig min_error;
  min_error.solve_mode = TaskSolveMode::kMinError;
  const Eigen::VectorXd all_hard = vector({1.0});
  const auto result = solve_canonical(
      {vector({1.0})}, {vector({0.0})}, {matrix({{1.0}})},
      matrix({{0.0}}), vector({1.0}), vector({2.0}), {min_error},
      &all_hard);

  ASSERT_EQ(result.status, SolverStatus::kInfeasible)
      << result.status_message;
  EXPECT_EQ(result.task_scales, std::vector<double>{-1.0});
  EXPECT_EQ(result.task_modes_effective,
            std::vector<TaskSolveMode>{TaskSolveMode::kMinError});
  EXPECT_EQ(result.task_used_fallback, std::vector<bool>{false});
  ASSERT_EQ(result.task_errors.size(), 1U);
}

TEST(HierarchicalLinearSolverTest,
     MixedSofteningStillCertifiesImmutableHardRows) {
  const Eigen::VectorXd softening = vector({1.0, 2.0});
  const auto result = solve_canonical(
      {vector({0.0})}, {vector({0.0})}, {matrix({{1.0}})},
      matrix({{0.0}, {0.0}}), vector({1.0, 3.0}), vector({2.0, 4.0}),
      {}, &softening);

  EXPECT_EQ(result.status, SolverStatus::kInfeasible)
      << result.status_message;
}

TEST(HierarchicalLinearSolverTest,
     AnyPermittedSofteningExcludesThatRowFromHardPhaseOne) {
  const Eigen::VectorXd softening = vector({1.0 + 5e-11});
  const auto result = solve_canonical(
      {vector({0.0})}, {vector({0.0})}, {matrix({{1.0}})},
      matrix({{0.0}}), vector({2e-10}), vector({20.0 + 2e-10}), {},
      &softening);

  EXPECT_NE(result.status, SolverStatus::kInfeasible)
      << result.status_message;
}

TEST(HierarchicalLinearSolverTest,
     LegacyVelocityWrapperDoesNotUseGeneralizedPhaseOnePolicy) {
  const Eigen::MatrixXd impossible_row = matrix({{0.0}});
  const Eigen::VectorXd lower = vector({1.0});
  const Eigen::VectorXd upper = vector({2.0});
  const auto config = exact_config();
  const auto generalized = solveHierarchicalLinearSystemEigen(
      {vector({0.0})}, {vector({0.0})}, {matrix({{1.0}})}, impossible_row,
      lower, upper, config);
  const auto legacy = computeMultiObjectiveVelocitySolutionEigen(
      {vector({0.0})}, {matrix({{1.0}})}, impossible_row, lower, upper,
      config);

  EXPECT_EQ(generalized.status, SolverStatus::kInfeasible);
  EXPECT_NE(legacy.status, SolverStatus::kInfeasible);
}

TEST(HierarchicalLinearSolverTest,
     ZeroTaskKeepsZeroWhenHardIntervalContainsZero) {
  const auto result = solve_canonical(
      {vector({0.0})}, {vector({0.0})}, {matrix({{1.0}})},
      matrix({{1.0}}), vector({-0.2}), vector({0.3}));

  ASSERT_EQ(result.status, SolverStatus::kSuccess) << result.status_message;
  EXPECT_NEAR(solution_vector(result)(0), 0.0, kTolerance);
}

TEST(HierarchicalLinearSolverTest,
     ZeroTaskUsesNullSpaceForHardIntervalExcludingZero) {
  const Eigen::MatrixXd constraints = matrix({{0.0, 1.0}});
  const Eigen::VectorXd lower = vector({0.2});
  const Eigen::VectorXd upper = vector({0.5});
  const auto result = solve_canonical(
      {vector({0.0})}, {vector({0.0})}, {matrix({{1.0, 0.0}})},
      constraints, lower, upper);

  ASSERT_EQ(result.status, SolverStatus::kSuccess) << result.status_message;
  const Eigen::VectorXd solution = solution_vector(result);
  EXPECT_NEAR(solution(0), 0.0, kTolerance);
  const Eigen::VectorXd values = constraints * solution;
  EXPECT_TRUE((values.array() >= lower.array() - kTolerance).all());
  EXPECT_TRUE((values.array() <= upper.array() + kTolerance).all());
}

TEST(HierarchicalLinearSolverTest,
     ZeroTaskFindsFeasiblePointInCoupledHardSetExcludingZero) {
  const Eigen::MatrixXd constraints = matrix({
      {1.0, 0.0},
      {0.0, 1.0},
      {3.6574149524059965, 0.99120747620299832},
      {0.99120747620299832, 0.52500000000000002},
  });
  const Eigen::VectorXd lower = vector(
      {-10.0, -10.0, 2.1031069333272972, 2.2499807288824504});
  const Eigen::VectorXd upper = vector(
      {10.0, 10.0, 2.3031069333272972, 2.4499807288824504});
  ObjectiveSolveConfig config;
  config.solve_mode = TaskSolveMode::kMinError;

  const auto result = solve_canonical(
      {Eigen::Vector2d::Zero()}, {Eigen::Vector2d::Zero()},
      {Eigen::Matrix2d::Identity()}, constraints, lower, upper, {config});

  ASSERT_EQ(result.status, SolverStatus::kSuccess) << result.status_message;
  const Eigen::VectorXd solution = solution_vector(result);
  const Eigen::VectorXd values = constraints * solution;
  EXPECT_TRUE((values.array() >= lower.array() - kTolerance).all());
  EXPECT_TRUE((values.array() <= upper.array() + kTolerance).all());
  EXPECT_GT(solution.norm(), 0.1);
  EXPECT_EQ(result.task_modes_effective,
            std::vector<TaskSolveMode>{TaskSolveMode::kMinError});
  EXPECT_EQ(result.task_used_fallback, std::vector<bool>{false});
}

TEST(HierarchicalLinearSolverTest,
     TallRankDeficientTaskDampsOnlyAvailableLeftSingularVectors) {
  VelocitySolverConfig config = exact_config();
  config.regularization_config.regularization_factor = 1e-8;
  const Eigen::MatrixXd objective =
      matrix({{0.0, 1.0}, {0.0, 2.0}, {0.0, 3.0}});
  const auto result = solveHierarchicalLinearSystemEigen(
      {vector({2.0, 4.0, 6.0})}, {Eigen::Vector3d::Zero()}, {objective},
      Eigen::Matrix2d::Identity(), vector({-10.0, -10.0}),
      vector({10.0, 10.0}), config, {}, nullptr);

  ASSERT_EQ(result.status, SolverStatus::kSuccess) << result.status_message;
  const Eigen::VectorXd solution = solution_vector(result);
  ASSERT_EQ(solution.size(), 2);
  EXPECT_NEAR(solution(0), 0.0, kTolerance);
  EXPECT_NEAR(solution(1), 2.0, 1e-6);
}

TEST(HierarchicalLinearSolverTest,
     SquareRankDeficientExactInverseDropsSubthresholdDirection) {
  RegularizedInverseConfig config;
  config.epsilon = 1e-10;
  config.regularization_factor = 0.0;
  const Eigen::MatrixXd input = matrix({{1.0, 0.0}, {0.0, 1e-16}});

  const Eigen::VectorXd solved =
      linalg::ComputeRegularizedInverse(config, input, nullptr, true) *
      vector({2.0, 1.0});

  EXPECT_TRUE(solved.allFinite());
  EXPECT_NEAR(solved(0), 2.0, kTolerance);
  EXPECT_NEAR(solved(1), 0.0, kTolerance);
}

TEST(HierarchicalLinearSolverTest,
     ExactInverseKeepsDirectionImmediatelyAboveThreshold) {
  RegularizedInverseConfig config;
  config.epsilon = 1e-10;
  config.regularization_factor = 0.0;
  const Eigen::MatrixXd input = matrix({{1.0, 0.0}, {0.0, 2e-10}});
  const Eigen::VectorXd expected = vector({2.0, 3.0});

  const Eigen::VectorXd solved =
      linalg::ComputeRegularizedInverse(config, input, nullptr, true) * input *
      expected;

  EXPECT_TRUE(solved.allFinite());
  EXPECT_TRUE(solved.isApprox(expected, kTolerance));
}

TEST(HierarchicalLinearSolverTest,
     TallRankDeficientExactInverseDropsSubthresholdDirection) {
  RegularizedInverseConfig config;
  config.epsilon = 1e-10;
  config.regularization_factor = 0.0;
  const Eigen::MatrixXd input =
      matrix({{1.0, 0.0}, {0.0, 5e-11}, {0.0, 0.0}});

  const Eigen::VectorXd solved =
      linalg::ComputeRegularizedInverse(config, input, nullptr, true) *
      vector({2.0, 1.0, 0.0});

  EXPECT_TRUE(solved.allFinite());
  EXPECT_NEAR(solved(0), 2.0, kTolerance);
  EXPECT_NEAR(solved(1), 0.0, kTolerance);
}

TEST(HierarchicalLinearSolverTest,
     TallRegularizedInverseKeepsConsistentWellConditionedSystemsAccurate) {
  RegularizedInverseConfig config;
  config.epsilon = 1e-10;
  config.regularization_factor = 1e-8;
  std::mt19937 generator(20260710U);
  std::uniform_real_distribution<double> distribution(-1.0, 1.0);

  int checked = 0;
  for (int sample = 0; sample < 250; ++sample) {
    Eigen::MatrixXd objective(7, 3);
    Eigen::Vector3d state;
    for (Eigen::Index row = 0; row < objective.rows(); ++row) {
      for (Eigen::Index column = 0; column < objective.cols(); ++column) {
        objective(row, column) = distribution(generator);
      }
    }
    for (Eigen::Index index = 0; index < state.size(); ++index) {
      state(index) = distribution(generator);
    }

    const Eigen::JacobiSVD<Eigen::MatrixXd> oracle_svd(objective);
    const Eigen::VectorXd singular_values = oracle_svd.singularValues();
    if (singular_values.tail<1>()(0) <= 1e-6 ||
        singular_values(0) / singular_values.tail<1>()(0) >= 100.0) {
      continue;
    }

    const Eigen::VectorXd target = objective * state;
    const Eigen::VectorXd solved =
        linalg::ComputeRegularizedInverse(config, objective, nullptr, true) *
        target;
    const double relative_residual =
        (objective * solved - target).norm() / (1.0 + target.norm());
    EXPECT_LT(relative_residual, 1e-10) << "sample " << sample;
    ++checked;
  }
  EXPECT_GE(checked, 200);
}

TEST(HierarchicalLinearSolverTest,
     TallRegularizedInverseClampsNegativeGramDeterminantDamping) {
  RegularizedInverseConfig config;
  config.epsilon = 1e-10;
  config.regularization_factor = 1e-8;
  const Eigen::MatrixXd objective = matrix({
      {1.6422347038834264, 0.55605541495080257, 1.7836315571144192,
       0.70411263013121883},
      {-0.14165787321107681, 0.90107095479450539, -2.2127330600575981,
       -0.13657154488549372},
      {-0.36468263349502589, 0.10283553305449014, 3.3709820626724505,
       2.3370841867120364},
      {1.658375690018201, -0.72290964356831033, -0.44321479923159113,
       1.5746783461249312},
      {-2.2651306937214719, 0.054296377686794448, 1.0490920602038201,
       1.6868368138111436},
  });
  const Eigen::VectorXd state =
      vector({-0.10262385622501465, -2.0106302860894831,
              0.19453868188847578, 0.025977010039250688});
  const Eigen::VectorXd target = objective * state;

  const Eigen::VectorXd solved =
      linalg::ComputeRegularizedInverse(config, objective, nullptr, true) *
      target;
  const double relative_residual =
      (objective * solved - target).norm() / (1.0 + target.norm());
  EXPECT_LT(relative_residual, 1e-12);
  EXPECT_TRUE(solved.isApprox(state, 1e-12));
}

TEST(HierarchicalLinearSolverTest,
     PositiveDampingFallsBackWhenGramInverseIsUnreliable) {
  RegularizedInverseConfig config;
  config.epsilon = 1e-10;
  config.regularization_factor = 1e-8;
  const std::vector<Eigen::MatrixXd> objectives{
      matrix({
          {0.80451976416007231, 0.00092775982869943861},
          {-0.59392475890733309, -0.00068490465633183433},
      }),
      matrix({
          {0.84037051215174141, 0.54097772069664096},
          {-0.028146327856202677, -0.018118836714822183},
      }),
  };

  for (const Eigen::MatrixXd &objective : objectives) {
    const Eigen::MatrixXd regularized_inverse =
        linalg::ComputeRegularizedInverse(config, objective, nullptr, true);

    const Eigen::JacobiSVD<Eigen::MatrixXd> oracle_svd(
        objective, Eigen::ComputeFullU | Eigen::ComputeFullV);
    const Eigen::VectorXd singular_values = oracle_svd.singularValues();
    const Eigen::MatrixXd gram = objective * objective.transpose();
    const double threshold_squared = config.epsilon * config.epsilon;
    const double bounded_det_ratio =
        std::clamp(gram.determinant() / threshold_squared, 0.0, 1.0);
    const double global_regularization =
        gram.determinant() < threshold_squared
            ? (1.0 - bounded_det_ratio * bounded_det_ratio) *
                  threshold_squared
            : 0.0;
    Eigen::VectorXd filtered_inverse =
        Eigen::VectorXd::Zero(singular_values.size());
    for (Eigen::Index index = 0; index < singular_values.size(); ++index) {
      const double sigma = singular_values(index);
      const double normalized_sigma = std::min(1.0, sigma / config.epsilon);
      const double damping =
          config.regularization_factor *
          std::max(0.0, 1.0 - normalized_sigma * normalized_sigma);
      filtered_inverse(index) =
          sigma / (sigma * sigma + global_regularization + damping);
    }
    const Eigen::MatrixXd expected_inverse =
        oracle_svd.matrixV() * filtered_inverse.asDiagonal() *
        oracle_svd.matrixU().transpose();

    EXPECT_TRUE(regularized_inverse.allFinite());
    EXPECT_LT((regularized_inverse - expected_inverse).norm() /
                  (1.0 + expected_inverse.norm()),
              1e-12);
  }
}

TEST(HierarchicalLinearSolverTest,
     AffineBiasCreatesNonzeroBaseSolutionAtZeroTaskScale) {
  const Eigen::VectorXd scalable_b_prime = vector({0.0});
  const Eigen::VectorXd positive_b_double_prime = vector({-0.3});
  const auto result = solve_canonical(
      {scalable_b_prime}, {positive_b_double_prime}, {matrix({{1.0}})},
      matrix({{1.0}}), vector({0.2}), vector({0.5}));

  ASSERT_EQ(result.status, SolverStatus::kSuccess) << result.status_message;
  const Eigen::VectorXd solution = solution_vector(result);
  EXPECT_NEAR(solution(0), 0.3, kTolerance);
  EXPECT_TRUE((solution + positive_b_double_prime)
                  .isApprox(Eigen::VectorXd::Zero(1), kTolerance));
}

TEST(HierarchicalLinearSolverTest,
     FeasibilityWitnessDoesNotBiasObjectiveNullSpace) {
  const auto result = solve_canonical(
      {vector({1.0}), vector({0.0})}, {vector({0.0}), vector({0.0})},
      {matrix({{1.0, 0.0, 0.0}}), matrix({{0.0, 0.0, 1.0}})},
      matrix({{1.0, 1.0, 0.0}}), vector({1.0}), vector({10.0}));

  ASSERT_EQ(result.status, SolverStatus::kSuccess) << result.status_message;
  EXPECT_TRUE(solution_vector(result).isApprox(vector({1.0, 0.0, 0.0}),
                                                kTolerance));
  ASSERT_EQ(result.task_scales.size(), 2U);
  EXPECT_NEAR(result.task_scales[0], 1.0, kTolerance);
  EXPECT_NEAR(result.task_scales[1], 1.0, kTolerance);
}

TEST(HierarchicalLinearSolverTest,
     ScaleZeroAffineEqualityIsProtectedFromLowerPriorities) {
  const auto result = solve_canonical(
      {vector({1.0, 0.0}), vector({2.0})},
      {vector({0.0, 1.0}), vector({0.0})},
      {Eigen::Matrix2d::Identity(), matrix({{0.0, 1.0}})},
      matrix({{1.0, 0.0}}), vector({0.0}), vector({0.0}));

  EXPECT_EQ(result.status, SolverStatus::kNoProgress);
  ASSERT_EQ(result.task_scales.size(), 2U);
  EXPECT_NEAR(result.task_scales[0], 0.0, kTolerance);
  EXPECT_TRUE(solution_vector(result).isApprox(vector({0.0, -1.0}),
                                                kTolerance));
}

TEST(HierarchicalLinearSolverTest,
     InconclusiveFeasibilitySearchIsNotReportedAsInfeasible) {
  constexpr double theta = 1e-10;
  const Eigen::MatrixXd constraints =
      matrix({{1.0, 0.0}, {std::cos(theta), std::sin(theta)}});
  const auto result = solve_canonical(
      {vector({0.0, 0.0})}, {vector({0.0, 0.0})},
      {Eigen::Matrix2d::Identity()}, constraints, vector({0.0, 1.0}),
      vector({0.0, 1.0}));

  EXPECT_NE(result.status, SolverStatus::kInfeasible)
      << result.status_message;
}

TEST(HierarchicalLinearSolverTest,
     ConstraintRowPermutationAndPositiveScalingPreserveResult) {
  const std::vector<Eigen::VectorXd> targets{vector({2.0, 1.0})};
  const std::vector<Eigen::VectorXd> biases{vector({0.0, 0.0})};
  const std::vector<Eigen::MatrixXd> objectives{Eigen::Matrix2d::Identity()};

  const auto baseline = solve_canonical(
      targets, biases, objectives,
      matrix({{1.0, 1.0}, {1.0, 0.0}, {0.0, 1.0}}),
      vector({-10.0, -10.0, -10.0}), vector({1.5, 10.0, 10.0}));
  const auto transformed = solve_canonical(
      targets, biases, objectives,
      matrix({{0.0, 3.0}, {2.0, 2.0}, {4.0, 0.0}}),
      vector({-30.0, -20.0, -40.0}), vector({30.0, 3.0, 40.0}));

  ASSERT_EQ(baseline.status, SolverStatus::kSuccess)
      << baseline.status_message;
  ASSERT_EQ(transformed.status, SolverStatus::kSuccess)
      << transformed.status_message;
  EXPECT_TRUE(solution_vector(transformed).isApprox(solution_vector(baseline),
                                                     kTolerance));
  expect_double_vectors_near(transformed.task_scales, baseline.task_scales);
}

} // namespace
} // namespace embodik::test
