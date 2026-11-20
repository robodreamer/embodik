#include "collision_acceleration_braking.hpp"

#include <gtest/gtest.h>
#include <limits>

namespace embodik::test {

TEST(CollisionAccelerationBrakingTest, RejectsUnsafeCoastingAndAdmitsBraking) {
  const double speed = detail::collision_acceleration_braking_speed(
      0.01, -0.13, 1.0, 0.01);
  EXPECT_GE(speed, 0.131);
  const double next_slack = 0.01 + 0.005 * (-0.13 - 0.131);
  EXPECT_GT(next_slack - 0.131 * 0.131 / 2.0, 0.0);
  EXPECT_LT(detail::collision_acceleration_braking_speed(
      0.01, -0.135, 1.0, 0.01), 0.135);
}

TEST(CollisionAccelerationBrakingTest, ZeroAndExhaustedBudgetForbidApproach) {
  EXPECT_DOUBLE_EQ(detail::collision_acceleration_braking_speed(0., 0., 1., .01), 0.);
  EXPECT_DOUBLE_EQ(detail::collision_acceleration_braking_speed(.01, 0., 0., .01), 0.);
  EXPECT_DOUBLE_EQ(detail::collision_acceleration_braking_speed(.001, -1., 1., .01), 0.);
}

TEST(CollisionAccelerationBrakingTest, InvalidInputsAreRejected) {
  EXPECT_THROW(detail::collision_acceleration_braking_speed(.01, 0., 1., 0.), std::invalid_argument);
  EXPECT_THROW(detail::collision_acceleration_braking_speed(.01, 0., -1., .01), std::invalid_argument);
  EXPECT_THROW(detail::collision_acceleration_braking_speed(
      std::numeric_limits<double>::quiet_NaN(), 0., 1., .01), std::invalid_argument);
  EXPECT_THROW(detail::collision_acceleration_braking_speed(
      .01, std::numeric_limits<double>::infinity(), 1., .01), std::invalid_argument);
}

TEST(CollisionAccelerationBrakingTest, RationalizedRootRetainsSmallBudgets) {
  EXPECT_GT(detail::collision_acceleration_braking_speed(1e-20, 0., 1., .01), 0.);
  const double largest = std::numeric_limits<double>::max();
  // long double has platform-dependent range. Overflow must be reported,
  // never turned into an unbounded/NaN constraint on narrower platforms.
  try {
    EXPECT_TRUE(std::isfinite(detail::collision_acceleration_braking_speed(largest, 0., largest, 1.)));
  } catch (const std::overflow_error &) {
  }
}

} // namespace embodik::test
