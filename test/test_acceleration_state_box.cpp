#include "acceleration_state_box.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <limits>

namespace embodik::detail {
namespace {

constexpr double kTolerance = 1e-12;

ScalarBounds active(double lower, double upper) {
  return {lower, upper, true, true};
}

StateBoxRowInput fixture(double sign = 1.0) {
  StateBoxRowInput input;
  input.value = sign * 0.04;
  input.rate = sign * 0.12;
  input.state_bounds = active(-0.05, 0.05);
  input.acceleration_bounds = active(-1.0, 1.0);
  input.lower_braking_acceleration = 1.0;
  input.upper_braking_acceleration = 1.0;
  return input;
}

TEST(AccelerationStateBoxTest, CompatibleBoxIsSignSymmetric) {
  constexpr double dt = 0.1;
  const auto upper = shape_state_box_row(fixture(), dt);
  const auto lower = shape_state_box_row(fixture(-1.0), dt);

  ASSERT_EQ(upper.status, SolverStatus::kSuccess) << upper.message;
  ASSERT_EQ(lower.status, SolverStatus::kSuccess) << lower.message;
  EXPECT_NEAR(upper.lower_acceleration, -1.0, kTolerance);
  EXPECT_NEAR(upper.upper_acceleration, -0.6753049234040402, kTolerance);
  EXPECT_NEAR(lower.lower_acceleration, 0.6753049234040402, kTolerance);
  EXPECT_NEAR(lower.upper_acceleration, 1.0, kTolerance);
  EXPECT_TRUE(has_state_box_cause(upper.upper_causes,
                                  StateBoxBoundCause::kNextStateViability));
}

TEST(AccelerationStateBoxTest,
     PositionRateShapingUsesTimestepBeforeAccelerationConversion) {
  StateBoxRowInput input;
  input.value = 0.9;
  input.rate = 0.0;
  input.state_bounds = active(-10.0, 1.0);
  input.acceleration_bounds = active(-100.0, 100.0);
  input.lower_braking_acceleration = 100.0;
  input.upper_braking_acceleration = 100.0;

  const auto shaped = shape_state_box_row(input, 0.2);

  ASSERT_EQ(shaped.status, SolverStatus::kSuccess) << shaped.message;
  EXPECT_NEAR(shaped.upper_acceleration, 2.5, kTolerance);
  EXPECT_TRUE(has_state_box_cause(shaped.upper_causes,
                                  StateBoxBoundCause::kStateRate));
}

TEST(AccelerationStateBoxTest, RejectsBoundaryAndBrakingViolations) {
  auto outside = fixture();
  outside.value = 0.06;
  auto outward = fixture();
  outward.value = 0.05;
  outward.rate = 0.01;
  auto outside_envelope = fixture();
  outside_envelope.rate = 1.0;

  EXPECT_EQ(shape_state_box_row(outside, 0.1).failure,
            StateBoxFailure::kCurrentStateOutside);
  EXPECT_EQ(shape_state_box_row(outward, 0.1).failure,
            StateBoxFailure::kOutwardRateAtBoundary);
  EXPECT_EQ(shape_state_box_row(outside_envelope, 0.1).failure,
            StateBoxFailure::kOutsideBrakingEnvelope);
}

TEST(AccelerationStateBoxTest, EmptyIntersectionAndOverflowFailClosed) {
  StateBoxRowInput empty;
  empty.value = 0.0;
  empty.rate = 1.0;
  empty.rate_bounds = {0.0, 0.0, false, true};
  empty.acceleration_bounds = active(-1.0, 1.0);

  StateBoxRowInput overflow;
  overflow.value = -1e308;
  overflow.rate = 1e308;
  overflow.state_bounds = active(-1e308, 1e308);
  overflow.acceleration_bounds = active(-1e308, 1e308);
  overflow.lower_braking_acceleration = 1e308;
  overflow.upper_braking_acceleration = 1e308;

  EXPECT_EQ(shape_state_box_row(empty, 0.1).failure,
            StateBoxFailure::kEmptyInterval);
  EXPECT_EQ(shape_state_box_row(overflow, 2.0).status,
            SolverStatus::kNumericalError);
}

TEST(AccelerationStateBoxTest,
     LinearizedPreparationPreservesMatrixBiasAndShapedRows) {
  LinearizedStateBoxInput input;
  input.source_id = "linearized_endpoint_rows";
  input.coefficient_matrix.resize(1, 2);
  input.coefficient_matrix << 2.0, -1.0;
  input.affine_bias = Eigen::VectorXd::Constant(1, 0.3);
  input.rows = {fixture()};

  const auto prepared = prepare_linearized_state_box(input, 0.1, 2);

  ASSERT_EQ(prepared.status, SolverStatus::kSuccess) << prepared.message;
  const auto &constraint = prepared.physical_constraint;
  EXPECT_EQ(constraint.source_id, input.source_id);
  EXPECT_TRUE(constraint.coefficient_matrix.isApprox(input.coefficient_matrix,
                                                     0.0));
  EXPECT_TRUE(constraint.affine_bias.isApprox(input.affine_bias, 0.0));
  ASSERT_EQ(prepared.state_rows.size(), 1U);
  ASSERT_EQ(prepared.shaped_rows.size(), 1U);
  EXPECT_EQ(prepared.state_rows[0].value, input.rows[0].value);
  EXPECT_EQ(prepared.state_rows[0].rate, input.rows[0].rate);
  EXPECT_NEAR(constraint.lower_bounds(0), -1.0, kTolerance);
  EXPECT_NEAR(constraint.upper_bounds(0), -0.6753049234040402, kTolerance);
  ASSERT_EQ(constraint.lower_bound_active.size(), 1U);
  ASSERT_EQ(constraint.upper_bound_active.size(), 1U);
  EXPECT_TRUE(constraint.lower_bound_active[0]);
  EXPECT_TRUE(constraint.upper_bound_active[0]);
}

TEST(AccelerationStateBoxTest,
     LinearizedPreparationRejectsShapeAndNonfiniteInputsAtomically) {
  LinearizedStateBoxInput wrong_shape;
  wrong_shape.source_id = "wrong_shape";
  wrong_shape.coefficient_matrix = Eigen::MatrixXd::Zero(2, 2);
  wrong_shape.affine_bias = Eigen::VectorXd::Zero(1);
  wrong_shape.rows = {fixture()};
  auto nonfinite = wrong_shape;
  nonfinite.coefficient_matrix.resize(1, 2);
  nonfinite.affine_bias(0) = std::numeric_limits<double>::quiet_NaN();

  const auto shape_result =
      prepare_linearized_state_box(wrong_shape, 0.1, 2);
  const auto nonfinite_result =
      prepare_linearized_state_box(nonfinite, 0.1, 2);

  EXPECT_EQ(shape_result.status, SolverStatus::kShapeMismatch);
  EXPECT_TRUE(shape_result.physical_constraint.coefficient_matrix.size() == 0);
  EXPECT_TRUE(shape_result.state_rows.empty());
  EXPECT_TRUE(shape_result.shaped_rows.empty());
  EXPECT_EQ(nonfinite_result.status, SolverStatus::kNonFiniteInput);
  EXPECT_TRUE(nonfinite_result.physical_constraint.coefficient_matrix.size() ==
              0);
  EXPECT_TRUE(nonfinite_result.state_rows.empty());
  EXPECT_TRUE(nonfinite_result.shaped_rows.empty());
}

TEST(AccelerationStateBoxTest,
     LinearizedPreparationEvaluatesZeroRowsInPhysicalCoordinates) {
  LinearizedStateBoxInput admissible;
  admissible.source_id = "constant_physical_acceleration";
  admissible.coefficient_matrix = Eigen::MatrixXd::Zero(1, 2);
  admissible.affine_bias = Eigen::VectorXd::Constant(1, -0.8);
  admissible.rows = {fixture()};
  auto infeasible = admissible;
  infeasible.affine_bias(0) = 0.0;

  const auto admissible_result =
      prepare_linearized_state_box(admissible, 0.1, 2);
  const auto infeasible_result =
      prepare_linearized_state_box(infeasible, 0.1, 2);

  ASSERT_EQ(admissible_result.status, SolverStatus::kSuccess)
      << admissible_result.message;
  EXPECT_TRUE(
      admissible_result.physical_constraint.coefficient_matrix.isZero(0.0));
  ASSERT_EQ(admissible_result.state_rows.size(), 1U);
  ASSERT_EQ(admissible_result.shaped_rows.size(), 1U);
  EXPECT_EQ(infeasible_result.status, SolverStatus::kInfeasible);
  EXPECT_TRUE(
      infeasible_result.physical_constraint.coefficient_matrix.size() == 0);
  EXPECT_TRUE(infeasible_result.state_rows.empty());
  EXPECT_TRUE(infeasible_result.shaped_rows.empty());
}

} // namespace
} // namespace embodik::detail
