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

} // namespace
} // namespace embodik::detail
