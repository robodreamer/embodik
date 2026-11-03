#include "velocity_collision_constraint_provider.hpp"
#include "sphere_pair_distance_lower_bound.hpp"
#include "collision_distance_evidence.hpp"

#include <embodik/kinematics_solver.hpp>
#include <embodik/robot_model.hpp>

#include <Eigen/Dense>
#include <cstdio>
#include <fstream>
#include <gtest/gtest.h>
#include <limits>

namespace embodik::test {
namespace {

TEST(CollisionDistanceEvidenceTest,
     RejectsFailureSentinelsWithoutRejectingPenetration) {
  EXPECT_FALSE(detail::collision_distance_evidence_is_valid(
      std::numeric_limits<double>::max()));
  EXPECT_FALSE(detail::collision_distance_evidence_is_valid(
      -std::numeric_limits<double>::max()));
  EXPECT_FALSE(detail::collision_distance_evidence_is_valid(
      std::numeric_limits<double>::infinity()));
  EXPECT_FALSE(detail::collision_distance_evidence_is_valid(
      std::numeric_limits<double>::quiet_NaN()));
  EXPECT_TRUE(detail::collision_distance_evidence_is_valid(-0.02));
  EXPECT_TRUE(detail::collision_distance_evidence_is_valid(0.0));
  EXPECT_TRUE(detail::collision_distance_evidence_is_valid(0.19));
}

TEST(CurrentCollisionCatalogTest,
     SeparatedBoxCylinderRecoversFailedDefaultInitialization) {
  const std::string path = "/tmp/embodik_box_cylinder_distance_evidence.urdf";
  {
    std::ofstream file(path);
    file << R"(<robot name="distance_evidence">
      <link name="world"/><link name="box">
      <collision><geometry><box size="0.22 0.25 0.31"/></geometry></collision></link>
      <joint name="fixed" type="fixed"><parent link="world"/><child link="box"/></joint>
      <link name="cylinder"><collision><geometry>
      <cylinder radius="0.05" length="0.051"/></geometry></collision></link>
      <joint name="slide" type="prismatic"><parent link="world"/><child link="cylinder"/>
      <axis xyz="1 0 0"/><limit lower="-1" upper="1" velocity="1" effort="10"/></joint>
    </robot>)";
  }
  auto robot = std::make_shared<RobotModel>(path, false);
  auto *model = robot->collision_model();
  ASSERT_EQ(model->geometryObjects.size(), 2u);
  Eigen::Matrix3d box_rotation, cylinder_rotation;
  box_rotation << 0.9682954899372707, -0.24843278600670157, 0.026172409214466993,
      0.24871051772645333, 0.9685456401242328, -0.007900719497635636,
      -0.02338637507996422, 0.014159584502698691, 0.9996262219586532;
  cylinder_rotation << 0.03826340463173573, -0.15238085800505363, 0.9875808756651949,
      0.92464185353583, 0.38015281144892854, 0.022831614864103317,
      -0.3789107474799518, 0.9122849960736614, 0.15544365983553465;
  model->geometryObjects[0].placement = pinocchio::SE3(box_rotation,
      Eigen::Vector3d(0.2910784527836738, -0.09057192281790243, 1.0423211160451393));
  model->geometryObjects[1].placement = pinocchio::SE3(cylinder_rotation,
      Eigen::Vector3d(0.6539962804866386, -0.19549527170366934, 1.057170068713012));
  const Eigen::VectorXd q = Eigen::VectorXd::Zero(1);
  KinematicsSolver solver(robot);
  solver.configure_collision_constraint(0.005, {}, {}, true, 1);
  solver.enable_sphere_broadphase(false);
  const auto request_before = robot->collision_data()->distanceRequests[0];
  for (int repeat = 0; repeat < 3; ++repeat) {
    const auto distances = solver.evaluate_collision_pair_distances(q);
    ASSERT_TRUE(distances.has_value());
    ASSERT_EQ(distances->size(), 1u);
    EXPECT_NEAR(distances->front().distance, 0.19059572987372572, 1e-12);
    EXPECT_TRUE(robot->collision_data()->distanceRequests[0] == request_before);
  }
  // Startup anchoring must receive the recovered signed distance, never the
  // finite negative failure sentinel which would silently weaken the floor.
  solver.set_collision_trajectory_floor_anchor(q, 0.0);
  const auto anchored = solver.evaluate_collision_pair_distances(q);
  ASSERT_TRUE(anchored.has_value());
  ASSERT_TRUE(anchored->front().persistent_floor_m.has_value());
  EXPECT_DOUBLE_EQ(*anchored->front().persistent_floor_m, 0.005);
  const auto debug = solver.evaluate_collision_debug(q);
  ASSERT_TRUE(debug.has_value());
  EXPECT_NEAR(debug->distance, 0.19059572987372572, 1e-12);
  EXPECT_TRUE(debug->point_a_world.allFinite());
  EXPECT_TRUE(debug->point_b_world.allFinite());
  detail::VelocityCollisionConstraintProvider provider(solver);
  const auto sampled = provider.validate_samples(q, {q});
  EXPECT_TRUE(sampled.acceptable) << sampled.message;
  std::remove(path.c_str());
}

TEST(CurrentCollisionCatalogTest, RotatedCylinderRowsIgnoreUnrelatedQueryHistory) {
  const std::string path = "/tmp/embodik_current_catalog_cylinders.urdf";
  {
    std::ofstream file(path);
    file << R"(<robot name="current_catalog_cylinders">
      <link name="world"/><link name="fixed">
      <collision><origin rpy="0.2 0.3 0.4"/><geometry>
      <cylinder radius="0.1" length="0.4"/></geometry></collision></link>
      <joint name="fixed" type="fixed"><parent link="world"/><child link="fixed"/></joint>
      <link name="moving"><collision><origin rpy="0.7 0.2 0.8"/><geometry>
      <cylinder radius="0.12" length="0.35"/></geometry></collision></link>
      <joint name="slide" type="prismatic"><origin xyz="0.4 0.15 0.1"/>
      <parent link="world"/><child link="moving"/><axis xyz="1 0 0"/>
      <limit lower="-0.2" upper="0.5" velocity="1" effort="100"/></joint>
    </robot>)";
  }
  const Eigen::VectorXd q = Eigen::VectorXd::Zero(1);
  const Eigen::VectorXd dq = Eigen::VectorXd::Constant(1, 0.1);
  for (const bool all_nearest_points : {false, true}) {
    SCOPED_TRACE(all_nearest_points);
    auto robot = std::make_shared<RobotModel>(path, false);
    KinematicsSolver solver(robot);
    solver.configure_collision_constraint(0.001, {}, {}, all_nearest_points, 1);
    solver.set_proximity_gated_collision_activation_enabled(false);
    solver.enable_sphere_broadphase(false);
    solver.enable_collision_pair_cache(true, 100, 0.10);
    solver.set_collision_refinement_time_budget_us(0);
    solver.set_acceleration_limits(Eigen::VectorXd::Constant(1, 2.0));
    solver.enable_acceleration_limits(true);
    detail::VelocityCollisionConstraintProvider provider(solver);
    const auto query = [&]() {
      robot->update_kinematics(q, dq);
      solver.set_previous_joint_velocities(dq);
      return provider.linearize(0.01);
    };
    const auto initial = query();
    ASSERT_TRUE(initial.has_value());
    for (const double other : {-0.1, 0.4, 0.1, -0.05, 0.3, 0.0}) {
      solver.evaluate_min_collision_distance(Eigen::VectorXd::Constant(1, other));
      const auto actual = query();
      ASSERT_TRUE(actual.has_value());
      EXPECT_NEAR(actual->distance, initial->distance, 1e-12);
      EXPECT_TRUE(actual->coefficient_matrix.isApprox(initial->coefficient_matrix, 1e-12));
      EXPECT_TRUE(actual->lower_bounds.isApprox(initial->lower_bounds, 1e-12));
      EXPECT_TRUE(actual->upper_bounds.isApprox(initial->upper_bounds, 1e-12));
      EXPECT_TRUE(actual->point_a_world.isApprox(initial->point_a_world, 1e-12));
      EXPECT_TRUE(actual->point_b_world.isApprox(initial->point_b_world, 1e-12));
    }
  }
  std::remove(path.c_str());
}

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

TEST_F(CollisionAccelerationCacheTest,
       ExplicitTrajectoryFloorDoesNotSpendAdditionalSampleTolerance) {
  auto anchored = solver(false, 1.0);
  anchored.configure_collision_constraint(0.1, {}, {}, true, 1);
  const Eigen::Vector2d seed = Eigen::Vector2d::Zero();
  constexpr double tolerance = 1e-9;
  anchored.set_collision_trajectory_floor_anchor(seed, tolerance);
  const Eigen::Vector2d near(0.5 * tolerance, 0.0);
  const Eigen::Vector2d beyond(1.5 * tolerance, 0.0);
  detail::VelocityCollisionConstraintProvider provider(anchored);
  EXPECT_TRUE(provider.validate_samples(seed, {near}).acceptable);
  EXPECT_FALSE(provider.validate_samples(seed, {beyond}).acceptable);
  // The allowed first loss cannot become the next call's floor.
  EXPECT_FALSE(provider.validate_samples(near, {beyond}).acceptable);
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
