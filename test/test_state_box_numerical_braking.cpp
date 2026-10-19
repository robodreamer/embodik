#include "acceleration_state_box.hpp"

#include <embodik/ik_baseline.hpp>
#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <limits>

namespace embodik::detail {
namespace {

StateBoxRowInput boundary_input(double sign, double scale = 1.0) {
  StateBoxRowInput input;
  input.value = sign * scale;
  input.rate = sign * 1e-16 * scale;
  input.state_bounds = {-scale, scale, true, true};
  input.rate_bounds = {-scale, scale, true, true};
  input.acceleration_bounds = {-2.0 * scale, 2.0 * scale, true, true};
  input.lower_braking_acceleration = 2.0 * scale;
  input.upper_braking_acceleration = 2.0 * scale;
  return input;
}

void expect_certified_brake_and_reentry(const StateBoxRowInput &input, double dt) {
  const double original_value = input.value;
  const double original_rate = input.rate;
  const auto shaped = shape_state_box_row(input, dt);
  ASSERT_EQ(shaped.status, SolverStatus::kSuccess) << shaped.message;
  const double physical_acceleration = input.rate < 0.0
                                           ? input.acceleration_bounds.upper
                                           : input.acceleration_bounds.lower;
  const double acceleration = std::copysign(
      std::min(std::abs(physical_acceleration),
               std::abs(input.rate) /
                   (dt * std::sqrt(64.0 * std::numeric_limits<double>::epsilon()))),
      physical_acceleration);
  EXPECT_DOUBLE_EQ(shaped.lower_acceleration, acceleration);
  EXPECT_GE(acceleration, input.acceleration_bounds.lower);
  EXPECT_LE(acceleration, input.acceleration_bounds.upper);
  EXPECT_DOUBLE_EQ(shaped.upper_acceleration, acceleration);
  EXPECT_FALSE(has_state_box_cause(shaped.lower_causes,
                                  StateBoxBoundCause::kRate));
  EXPECT_FALSE(has_state_box_cause(shaped.upper_causes,
                                  StateBoxBoundCause::kRate));
  EXPECT_DOUBLE_EQ(input.value, original_value);
  EXPECT_DOUBLE_EQ(input.rate, original_rate);
  auto next = input;
  next.value = input.value +
               (dt * input.rate + 0.5 * dt * dt * acceleration);
  next.rate = input.rate + dt * acceleration;
  EXPECT_GE(next.value, input.state_bounds.lower);
  EXPECT_LE(next.value, input.state_bounds.upper);
  EXPECT_GE(next.rate, input.rate_bounds.lower);
  EXPECT_LE(next.rate, input.rate_bounds.upper);
  const auto next_box = shape_state_box_row(next, dt);
  ASSERT_EQ(next_box.status, SolverStatus::kSuccess) << next_box.message;
  EXPECT_LE(std::max(next_box.lower_acceleration,
                     input.acceleration_bounds.lower),
            std::min(next_box.upper_acceleration,
                     input.acceleration_bounds.upper));
}

TEST(StateBoxNumericalBraking, BoundaryUsesScaleFreeWitnessAcrossUnits) {
  for (double sign : {-1.0, 1.0}) {
    for (double scale : {1e-9, 1.0, 1e9}) {
      SCOPED_TRACE(::testing::Message() << sign << " " << scale);
      expect_certified_brake_and_reentry(boundary_input(sign, scale), 0.1);
    }
  }
}

TEST(StateBoxNumericalBraking, ZeroFaceCancellationUsesPositionUnits) {
  for (double sign : {-1.0, 1.0}) {
    auto input = boundary_input(sign);
    input.value = -sign * 1e-32;
    input.rate = sign * 3e-16;
    input.state_bounds = sign < 0.0 ? ScalarBounds{0.0, 2.0, true, true}
                                   : ScalarBounds{-2.0, 0.0, true, true};
    expect_certified_brake_and_reentry(input, 0.1);
  }
}

TEST(StateBoxNumericalBraking, AsymmetricCapsKeepWitnessWithinCorrectPhysicalFace) {
  for (double sign : {-1.0, 1.0}) {
    auto input = boundary_input(sign);
    input.acceleration_bounds = {-2.0, 4.0, true, true};
    input.lower_braking_acceleration = 4.0;
    expect_certified_brake_and_reentry(input, 0.1);
  }
}

TEST(StateBoxNumericalBraking, RestAndInwardMotionKeepOriginalIntervals) {
  for (double sign : {-1.0, 1.0}) {
    for (double rate : {0.0, -sign * 0.1}) {
      auto input = boundary_input(sign);
      input.rate = rate;
      const auto shaped = shape_state_box_row(input, 0.1);
      ASSERT_EQ(shaped.status, SolverStatus::kSuccess) << shaped.message;
      EXPECT_LE(shaped.lower_acceleration, 0.0);
      EXPECT_GE(shaped.upper_acceleration, 0.0);
      EXPECT_EQ(shaped.message, "");
    }
  }
}

TEST(StateBoxNumericalBraking, NearZeroPitchUsesGentleCertifiedWitness) {
  auto input = boundary_input(-1.0);
  input.value = 1e-33;
  input.rate = -2e-16;
  input.state_bounds = {0.0, 1.5, true, true};
  input.rate_bounds = {-2.0, 2.0, true, true};
  input.acceleration_bounds = {-15.0, 15.0, true, true};
  input.lower_braking_acceleration = 15.0;
  input.upper_braking_acceleration = 15.0;
  expect_certified_brake_and_reentry(input, 0.01);
  const auto shaped = shape_state_box_row(input, 0.01);
  EXPECT_LT(shaped.lower_acceleration, 1e-6);
}

TEST(StateBoxNumericalBraking, SmallerDeclaredBrakingStillCertifiesBothStates) {
  for (double sign : {-1.0, 1.0}) {
    auto input = boundary_input(sign);
    input.lower_braking_acceleration = 0.25;
    input.upper_braking_acceleration = 0.25;
    expect_certified_brake_and_reentry(input, 0.1);
  }
}

TEST(StateBoxNumericalBraking, GentleWitnessCannotExceedAsymmetricPhysicalCap) {
  auto input = boundary_input(1.0);
  input.acceleration_bounds = {-1e-10, 4.0, true, true};
  input.lower_braking_acceleration = 4.0;
  input.upper_braking_acceleration = 1e-10;
  expect_certified_brake_and_reentry(input, 0.1);
}

TEST(StateBoxNumericalBraking, UnderflowingGentleCandidateKeepsPhysicalFallback) {
  auto input = boundary_input(-1.0);
  input.value = 0.0;
  input.rate = -std::numeric_limits<double>::denorm_min();
  input.state_bounds = {0.0, 1e-200, true, true};
  input.rate_bounds = {-1.0, 1.0, true, true};
  input.acceleration_bounds = {-1e-300, 1e-300, true, true};
  input.lower_braking_acceleration = 1e-300;
  input.upper_braking_acceleration = 1e-300;
  const auto shaped = shape_state_box_row(input, 1e10);
  ASSERT_EQ(shaped.status, SolverStatus::kSuccess) << shaped.message;
  EXPECT_DOUBLE_EQ(shaped.lower_acceleration, 1e-300);
  EXPECT_DOUBLE_EQ(shaped.upper_acceleration, 1e-300);
}

TEST(StateBoxNumericalBraking, NonfiniteGentleArithmeticDoesNotAdmitInvalidState) {
  auto input = boundary_input(1.0);
  const auto shaped = shape_state_box_row(
      input, std::numeric_limits<double>::denorm_min());
  EXPECT_NE(shaped.status, SolverStatus::kSuccess);
}

TEST(StateBoxNumericalBraking, ConservativeDeclaredBrakingIsNotOverridden) {
  auto input = boundary_input(1.0);
  input.value = 0.999;
  input.rate = 0.1;
  input.acceleration_bounds = {-15.0, 15.0, true, true};
  input.lower_braking_acceleration = 15.0;
  input.upper_braking_acceleration = 0.1;
  EXPECT_NE(shape_state_box_row(input, 0.01).status, SolverStatus::kSuccess);
}

TEST(StateBoxNumericalBraking, DetectableVertexExcursionRemainsInfeasible) {
  for (double sign : {-1.0, 1.0}) {
    auto input = boundary_input(sign);
    input.rate = sign * 0.1;
    EXPECT_NE(shape_state_box_row(input, 0.1).status, SolverStatus::kSuccess);
  }
}

TEST(StateBoxNumericalBraking, InvalidAndLockedInputsRemainRejected) {
  for (double sign : {-1.0, 1.0}) {
    auto input = boundary_input(sign);
    input.rate_bounds = {0.0, 0.0, true, true};
    EXPECT_NE(shape_state_box_row(input, 0.1).status, SolverStatus::kSuccess);
    input = boundary_input(sign);
    input.acceleration_bounds = {0.0, 0.0, true, true};
    input.lower_braking_acceleration = 0.0;
    input.upper_braking_acceleration = 0.0;
    EXPECT_NE(shape_state_box_row(input, 0.1).status, SolverStatus::kSuccess);
    input = boundary_input(sign);
    input.value = sign * 1.01;
    EXPECT_NE(shape_state_box_row(input, 0.1).status, SolverStatus::kSuccess);
    input = boundary_input(sign);
    input.rate = std::numeric_limits<double>::quiet_NaN();
    EXPECT_EQ(shape_state_box_row(input, 0.1).status,
              SolverStatus::kNonFiniteInput);
  }
}

TEST(StateBoxNumericalBraking, StopMustFitTickAndOppositeEndpoint) {
  for (double sign : {-1.0, 1.0}) {
    auto input = boundary_input(sign);
    input.acceleration_bounds = {-1e-17, 1e-17, true, true};
    input.lower_braking_acceleration = 1e-17;
    input.upper_braking_acceleration = 1e-17;
    EXPECT_NE(shape_state_box_row(input, 0.1).status, SolverStatus::kSuccess);
    input = boundary_input(sign);
    input.rate_bounds = {-10.0, 10.0, true, true};
    EXPECT_NE(shape_state_box_row(input, 1e10).status, SolverStatus::kSuccess);
  }
}

TEST(StateBoxNumericalBraking, OppositeHugeFaceDoesNotInflateLocalBudget) {
  auto input = boundary_input(-1.0);
  input.value = 1e-32;
  input.rate = -0.1;
  input.state_bounds = {0.0, 1e300, true, true};
  EXPECT_NE(shape_state_box_row(input, 0.1).status, SolverStatus::kSuccess);
}

TEST(StateBoxNumericalBraking, ContradictoryZeroAccelerationRowStillFails) {
  const auto shaped = shape_state_box_row(boundary_input(1.0), 0.1);
  ASSERT_EQ(shaped.status, SolverStatus::kSuccess) << shaped.message;
  Eigen::MatrixXd constraints = Eigen::MatrixXd::Ones(2, 1);
  Eigen::VectorXd lower(2), upper(2);
  lower << shaped.lower_acceleration, 0.0;
  upper << shaped.upper_acceleration, 0.0;
  ObjectiveSolveConfig objective;
  objective.solve_mode = TaskSolveMode::kMinError;
  objective.allow_min_error_fallback = false;
  VelocitySolverConfig config;
  config.epsilon = 1e-10;
  config.precision_threshold = 1e-12;
  const auto result = solveHierarchicalLinearSystemEigen(
      {Eigen::VectorXd::Zero(1)}, {Eigen::VectorXd::Zero(1)},
      {Eigen::MatrixXd::Identity(1, 1)}, constraints, lower, upper,
      config, {objective});
  EXPECT_NE(result.status, SolverStatus::kSuccess);
}

} // namespace
} // namespace embodik::detail
