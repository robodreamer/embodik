#include "velocity_collision_constraint_provider.hpp"
#include "sphere_pair_distance_lower_bound.hpp"

#include <embodik/kinematics_solver.hpp>
#include <embodik/robot_model.hpp>

#include <Eigen/Dense>
#include <cstdio>
#include <fstream>
#include <gtest/gtest.h>
#include <limits>

namespace embodik::test {
namespace {

class CollisionAccelerationCacheTest : public ::testing::Test {
protected:
  void SetUp() override {
    path_ = "/tmp/embodik_collision_acceleration_cache.urdf";
    std::ofstream file(path_);
    file << R"(<?xml version="1.0"?>
<robot name="collision_acceleration_cache">
  <link name="world"/>
  <link name="left">
    <collision><geometry><sphere radius="0.05"/></geometry></collision>
  </link>
  <joint name="left_slide" type="prismatic">
    <parent link="world"/><child link="left"/><axis xyz="1 0 0"/>
    <limit lower="-1" upper="1" effort="100" velocity="1"/>
  </joint>
  <link name="right">
    <collision><geometry><sphere radius="0.05"/></geometry></collision>
  </link>
  <joint name="right_slide" type="prismatic">
    <parent link="world"/><child link="right"/>
    <origin xyz="0.15 0 0"/><axis xyz="1 0 0"/>
    <limit lower="-1" upper="1" effort="100" velocity="1"/>
  </joint>
</robot>)";
    file.close();
    robot_ = std::make_shared<RobotModel>(path_, false);
    robot_->update_kinematics(Eigen::Vector2d::Zero(),
                             Eigen::Vector2d(0.02, -0.01));
    ASSERT_TRUE(robot_->has_collision_geometry());
  }

  void TearDown() override { std::remove(path_.c_str()); }

  KinematicsSolver solver(bool acceleration_enabled, double acceleration) {
    KinematicsSolver result(robot_);
    result.set_dt(0.01);
    result.configure_collision_constraint(0.005, {}, {}, true, 1);
    result.set_proximity_gated_collision_activation_enabled(false);
    result.enable_sphere_broadphase(false);
    result.enable_collision_pair_cache(true, 100, 0.10);
    result.set_collision_refinement_time_budget_us(0);
    result.set_acceleration_limits(Eigen::Vector2d::Constant(acceleration));
    result.enable_acceleration_limits(acceleration_enabled);
    return result;
  }

  std::optional<detail::VelocityCollisionConstraintLinearization>
  rows(KinematicsSolver &solver) {
    return detail::VelocityCollisionConstraintProvider(solver).linearize(0.01);
  }

  std::string path_;
  std::shared_ptr<RobotModel> robot_;
};

TEST_F(CollisionAccelerationCacheTest, EnablingLimitsRefreshesWarmedRows) {
  auto cached = solver(false, 1.0);
  const auto before = rows(cached);
  ASSERT_TRUE(before.has_value());
  ASSERT_EQ(before->coefficient_matrix.rows(), 1);
  ASSERT_EQ(before->coefficient_matrix.cols(), 2);
  cached.set_previous_joint_velocities(Eigen::Vector2d(0.2, -0.1));

  cached.enable_acceleration_limits(true);
  EXPECT_TRUE(cached.get_previous_joint_velocities().isZero(0.0));
  const auto after = rows(cached);
  auto fresh = solver(true, 1.0);
  const auto expected = rows(fresh);
  ASSERT_TRUE(after.has_value());
  ASSERT_TRUE(expected.has_value());
  EXPECT_GT(expected->lower_bounds(0), before->lower_bounds(0) + 0.1);
  EXPECT_TRUE(after->coefficient_matrix.isApprox(before->coefficient_matrix,
                                                1e-12));
  EXPECT_TRUE(after->lower_bounds.isApprox(expected->lower_bounds, 1e-12));
  EXPECT_TRUE(after->upper_bounds.isApprox(expected->upper_bounds, 1e-12));
}

TEST_F(CollisionAccelerationCacheTest,
       SphereSampleCertificatesMatchExactDistanceAtRecoveryAndFloorBoundaries) {
  const std::vector<std::pair<double, double>> cases{
      {0.0, 0.03}, {0.0, 0.045}, {0.0, 0.045 + 0.5e-8},
      {0.0, 0.045 + 2e-8}, {0.0, 0.06},
      {0.06, 0.055}, {0.06, 0.06}, {0.06, 0.065}};
  for (const auto &[seed_position, sample_position] : cases) {
    SCOPED_TRACE(seed_position);
    SCOPED_TRACE(sample_position);
    auto exact_solver = solver(false, 1.0);
    auto bounded_solver = solver(false, 1.0);
    bounded_solver.enable_sphere_broadphase(true);
    const Eigen::Vector2d seed(seed_position, 0.0);
    const std::vector<Eigen::VectorXd> samples{
        Eigen::Vector2d(sample_position, 0.0)};
    const auto exact = detail::VelocityCollisionConstraintProvider(exact_solver)
                           .validate_samples(seed, samples);
    const auto bounded = detail::VelocityCollisionConstraintProvider(bounded_solver)
                             .validate_samples(seed, samples);
    EXPECT_EQ(bounded.status, exact.status);
    EXPECT_EQ(bounded.acceptable, exact.acceptable);
    EXPECT_EQ(bounded.allowed_pair_count, exact.allowed_pair_count);
    EXPECT_EQ(bounded.failed_sample_index, exact.failed_sample_index);
    EXPECT_EQ(bounded.failed_pair_key, exact.failed_pair_key);
    if (bounded.acceptable) {
      EXPECT_EQ(bounded.sample_exact_distance_queries, 0u);
      EXPECT_EQ(bounded.conservative_bound_certified_pairs, 1u);
    } else {
      EXPECT_NEAR(bounded.failed_sample_distance,
                  exact.failed_sample_distance, 1e-12);
    }
  }
}

TEST(SpherePairDistanceLowerBound, IncludesRotatedOffsetsAndSignedOverlap) {
  BoundingSphere a{}, b{};
  a.center = Eigen::Vector3d(1.0, 0.0, 0.0);
  b.center = Eigen::Vector3d(0.0, 1.0, 0.0);
  a.radius = 0.2;
  b.radius = 0.3;
  Eigen::Matrix3d rotated;
  rotated << 0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0;
  EXPECT_DOUBLE_EQ(detail::sphere_pair_distance_lower_bound(
                       a, b, Eigen::Vector3d::Zero(), rotated,
                       Eigen::Vector3d(0.0, 2.0, 0.0), Eigen::Matrix3d::Identity()),
                   1.5);
  EXPECT_DOUBLE_EQ(detail::sphere_pair_distance_lower_bound(
                       a, b, Eigen::Vector3d::Zero(), rotated,
                       Eigen::Vector3d::Zero(), Eigen::Matrix3d::Identity()),
                   -0.5);
}

TEST_F(CollisionAccelerationCacheTest, DisablingLimitsRefreshesWarmedRows) {
  auto cached = solver(true, 1.0);
  const auto before = rows(cached);
  ASSERT_TRUE(before.has_value());
  cached.set_previous_joint_velocities(Eigen::Vector2d(0.2, -0.1));

  cached.enable_acceleration_limits(false);
  EXPECT_EQ(cached.get_previous_joint_velocities().size(), 0);
  const auto after = rows(cached);
  auto fresh = solver(false, 1.0);
  const auto expected = rows(fresh);
  ASSERT_TRUE(after.has_value());
  ASSERT_TRUE(expected.has_value());
  EXPECT_LT(expected->lower_bounds(0), before->lower_bounds(0) - 0.1);
  EXPECT_TRUE(after->lower_bounds.isApprox(expected->lower_bounds, 1e-12));
  EXPECT_TRUE(after->upper_bounds.isApprox(expected->upper_bounds, 1e-12));
}

TEST_F(CollisionAccelerationCacheTest, ChangingLimitsRefreshesRowsKeepsHistory) {
  auto cached = solver(true, 4.0);
  const auto before = rows(cached);
  ASSERT_TRUE(before.has_value());
  const Eigen::Vector2d previous(0.2, -0.1);
  cached.set_previous_joint_velocities(previous);

  cached.set_acceleration_limits(Eigen::Vector2d::Constant(0.25));
  EXPECT_TRUE(cached.get_previous_joint_velocities().isApprox(previous, 0.0));
  const auto after = rows(cached);
  auto fresh = solver(true, 0.25);
  const auto expected = rows(fresh);
  ASSERT_TRUE(after.has_value());
  ASSERT_TRUE(expected.has_value());
  EXPECT_GT(expected->lower_bounds(0), before->lower_bounds(0) + 0.1);
  EXPECT_TRUE(after->lower_bounds.isApprox(expected->lower_bounds, 1e-12));
  EXPECT_TRUE(after->upper_bounds.isApprox(expected->upper_bounds, 1e-12));
}

TEST_F(CollisionAccelerationCacheTest, InvalidLimitsKeepPolicyAndHistory) {
  auto cached = solver(true, 1.0);
  const auto before = rows(cached);
  ASSERT_TRUE(before.has_value());
  const Eigen::Vector2d previous(0.2, -0.1);
  cached.set_previous_joint_velocities(previous);

  EXPECT_THROW(cached.set_acceleration_limits(Eigen::VectorXd::Zero(1)),
               std::invalid_argument);
  EXPECT_THROW(cached.set_acceleration_limits(Eigen::Vector2d(1.0, -1.0)),
               std::invalid_argument);
  EXPECT_THROW(cached.set_acceleration_limits(Eigen::Vector2d(
                   1.0, std::numeric_limits<double>::infinity())),
               std::invalid_argument);
  EXPECT_TRUE(cached.get_acceleration_limits_enabled());
  EXPECT_TRUE(cached.get_previous_joint_velocities().isApprox(previous, 0.0));
  const auto after = rows(cached);
  ASSERT_TRUE(after.has_value());
  EXPECT_TRUE(after->lower_bounds.isApprox(before->lower_bounds, 1e-12));
  EXPECT_TRUE(after->upper_bounds.isApprox(before->upper_bounds, 1e-12));
}

} // namespace
} // namespace embodik::test
