#include "velocity_collision_constraint_provider.hpp"
#include "sphere_pair_distance_lower_bound.hpp"
#include "collision_distance_evidence.hpp"
#include "kinematics_solver_internal.hpp"

#include <embodik/kinematics_solver.hpp>
#include <embodik/robot_model.hpp>
#include <embodik/acceleration_solver.hpp>
#include <coal/shape/geometric_shapes.h>

#include <Eigen/Dense>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <gtest/gtest.h>
#include <limits>
#include <memory>
#include <sstream>

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
       StartupRecoveryFloorsPreserveStateMarginsAndAccelerationPolicy) {
  for (const bool acceleration_enabled : {false, true}) {
    auto provider = solver(acceleration_enabled, 2.0);
    provider.configure_collision_constraint(0.06, {}, {}, true, 1);
    provider.set_collision_pair_min_distance("left", "right", 0.055, false);
    provider.set_non_worsening_collision_floor_enabled(true);
    const Eigen::Vector2d actual_q(0.02, -0.03), actual_dq(0.3, -0.2);
    const Eigen::Vector2d previous(0.11, -0.12);
    robot_->update_kinematics(actual_q, actual_dq);
    provider.set_previous_joint_velocities(previous);
    const auto before = provider.evaluate_collision_pair_distances(Eigen::Vector2d::Zero());
    ASSERT_TRUE(before.has_value());
    ASSERT_EQ(before->size(), 1u);
    ASSERT_FALSE(before->front().persistent_floor_m.has_value());
    const auto overrides = provider.get_collision_pair_min_distance_overrides();
    provider.initialize_collision_recovery_floors(Eigen::Vector2d::Zero());
    EXPECT_TRUE(robot_->get_current_configuration().isApprox(actual_q, 0.0));
    EXPECT_TRUE(robot_->get_current_velocity().isApprox(actual_dq, 0.0));
    EXPECT_EQ(provider.get_acceleration_limits_enabled(), acceleration_enabled);
    EXPECT_TRUE(provider.get_previous_joint_velocities().isApprox(previous, 0.0));
    EXPECT_DOUBLE_EQ(provider.get_collision_min_distance(), 0.06);
    EXPECT_DOUBLE_EQ(provider.get_collision_structural_floor(), 0.005);
    EXPECT_EQ(provider.get_collision_pair_min_distance_overrides(), overrides);
    for (const double excursion : {0.0, -0.08, 0.0}) {
      const auto after = provider.evaluate_collision_pair_distances(Eigen::Vector2d(excursion, 0.0));
      ASSERT_TRUE(after.has_value());
      ASSERT_TRUE(after->front().persistent_floor_m.has_value());
      EXPECT_DOUBLE_EQ(*after->front().persistent_floor_m,
                       std::min(before->front().distance, before->front().minimum_distance));
      EXPECT_DOUBLE_EQ(after->front().minimum_distance, 0.055);
    }
  }
}

TEST_F(CollisionAccelerationCacheTest,
       StartupRecoveryFloorsRejectMalformedLateAndUnsupportedInitialization) {
  auto provider = solver(false, 1.0);
  provider.configure_collision_constraint(0.06, {}, {}, true, 1);
  provider.set_non_worsening_collision_floor_enabled(true);
  const auto uninitialized = [&]() {
    const auto pair = provider.evaluate_collision_pair_distances(Eigen::Vector2d::Zero());
    ASSERT_TRUE(pair.has_value());
    EXPECT_FALSE(pair->front().persistent_floor_m.has_value());
  };
  EXPECT_THROW(provider.initialize_collision_recovery_floors(Eigen::VectorXd::Zero(1)), std::invalid_argument);
  EXPECT_THROW(provider.initialize_collision_recovery_floors(Eigen::Vector2d(NAN, 0.0)), std::invalid_argument);
  uninitialized();
  provider.set_collision_pair_min_distance("left", "right", 0.08, true);
  const auto pending = provider.get_collision_pair_min_distance_overrides();
  EXPECT_THROW(provider.initialize_collision_recovery_floors(Eigen::Vector2d::Zero()), std::invalid_argument);
  EXPECT_EQ(provider.get_collision_pair_min_distance_overrides(), pending);
  uninitialized();
  provider.clear_collision_pair_min_distance("left", "right");
  provider.set_collision_trajectory_floor_anchor(Eigen::Vector2d::Zero(), 0.0);
  EXPECT_THROW(provider.initialize_collision_recovery_floors(Eigen::Vector2d::Zero()), std::invalid_argument);
  provider.clear_collision_trajectory_floor_anchor();
  uninitialized();
  provider.initialize_collision_recovery_floors(Eigen::Vector2d::Zero());
  const auto initialized = provider.evaluate_collision_pair_distances(Eigen::Vector2d::Zero());
  ASSERT_TRUE(initialized.has_value());
  EXPECT_THROW(provider.initialize_collision_recovery_floors(Eigen::Vector2d(-0.05, 0.0)), std::invalid_argument);
  const auto unchanged = provider.evaluate_collision_pair_distances(Eigen::Vector2d::Zero());
  EXPECT_EQ(unchanged->front().persistent_floor_m, initialized->front().persistent_floor_m);
}

TEST_F(CollisionAccelerationCacheTest,
       StartupRecoveryFloorsRequireEnabledNonemptyRuntimeCatalog) {
  auto provider = solver(false, 1.0);
  EXPECT_THROW(provider.initialize_collision_recovery_floors(Eigen::Vector2d::Zero()), std::invalid_argument);
  provider.set_non_worsening_collision_floor_enabled(true);
  robot_->collision_data()->activeCollisionPairs[0] = false;
  EXPECT_THROW(provider.initialize_collision_recovery_floors(Eigen::Vector2d::Zero()), std::invalid_argument);
  robot_->collision_data()->activeCollisionPairs[0] = true;
  provider.initialize_collision_recovery_floors(Eigen::Vector2d::Zero());
  auto disabled = solver(false, 1.0);
  disabled.set_non_worsening_collision_floor_enabled(true);
  disabled.clear_collision_constraint();
  EXPECT_THROW(disabled.initialize_collision_recovery_floors(Eigen::Vector2d::Zero()), std::invalid_argument);
}

TEST_F(CollisionAccelerationCacheTest,
       StartupRecoveryFloorRefreshesWarmVelocityRowsAndCapsAboveNominalSeed) {
  auto provider = solver(false, 1.0);
  provider.configure_collision_constraint(0.06, {}, {}, true, 1);
  robot_->update_kinematics(Eigen::Vector2d::Zero(), Eigen::Vector2d::Zero());
  detail::VelocityCollisionConstraintProvider rows_provider(provider);
  const auto before = rows_provider.compute();
  ASSERT_TRUE(before.has_value());
  ASSERT_GT(before->lower_bounds(0), 0.0);
  provider.set_non_worsening_collision_floor_enabled(true);
  provider.initialize_collision_recovery_floors(Eigen::Vector2d::Zero());
  EXPECT_TRUE(provider.get_last_collision_debug_list().empty());
  const auto after = rows_provider.compute();
  ASSERT_TRUE(after.has_value());
  auto fresh = solver(false, 1.0);
  fresh.configure_collision_constraint(0.06, {}, {}, true, 1);
  fresh.set_non_worsening_collision_floor_enabled(true);
  fresh.initialize_collision_recovery_floors(Eigen::Vector2d::Zero());
  const auto expected = detail::VelocityCollisionConstraintProvider(fresh).compute();
  ASSERT_TRUE(expected.has_value());
  EXPECT_TRUE(after->jacobian.isApprox(expected->jacobian, 1e-12));
  EXPECT_TRUE(after->lower_bounds.isApprox(expected->lower_bounds, 1e-12));
  EXPECT_TRUE(after->upper_bounds.isApprox(expected->upper_bounds, 1e-12));
  EXPECT_LT(after->lower_bounds(0), before->lower_bounds(0)-1e-3);
  auto above = solver(false, 1.0);
  above.configure_collision_constraint(0.06, {}, {}, true, 1);
  above.set_non_worsening_collision_floor_enabled(true);
  above.initialize_collision_recovery_floors(Eigen::Vector2d(-0.05, 0.0));
  const auto pair = above.evaluate_collision_pair_distances(Eigen::Vector2d::Zero());
  ASSERT_TRUE(pair.has_value());
  EXPECT_DOUBLE_EQ(*pair->front().persistent_floor_m, 0.06);
}

TEST_F(CollisionAccelerationCacheTest,
       StartupRecoveryFloorRejectsLossAcceptedByStructuralAccelerationLift) {
  const Eigen::Vector2d q = Eigen::Vector2d::Zero();
  const Eigen::Vector2d dq = Eigen::Vector2d::Zero();
  const Eigen::Vector2d ddq(0.2, 0.0);
  const auto validate = [&](bool initialize) {
    auto provider = solver(false, 1.0);
    provider.configure_collision_constraint(0.06, {}, {}, true, 1);
    provider.set_non_worsening_collision_floor_enabled(true);
    if (initialize) provider.initialize_collision_recovery_floors(q);
    provider.set_previous_joint_velocities(dq);
    AccelerationSolver acceleration(robot_);
    AccelerationSolveOptions options;
    options.acceleration_limits_override = Eigen::Vector2d::Ones();
    VelocityCollisionLiftOptions lift;
    lift.validation_substeps = 4;
    return acceleration.validate_candidate_with_velocity_collision(provider, q, dq, 0.1, ddq, options, lift);
  };
  const auto structural = validate(false);
  ASSERT_EQ(structural.status, SolverStatus::kSuccess) << structural.status_message;
  EXPECT_TRUE(structural.acceleration_limits_applied);
  EXPECT_TRUE(structural.velocity_collision_lift_applied);
  EXPECT_TRUE(structural.collision_endpoint_validated);
  const auto startup = validate(true);
  EXPECT_NE(startup.status, SolverStatus::kSuccess);
  EXPECT_EQ(startup.joint_accelerations.size(), 0);
  // The rejected command would lose 1mm, well above native numerical tolerances.
  EXPECT_NEAR(structural.q_solution(0), 0.001, 1e-12);
}

TEST_F(CollisionAccelerationCacheTest,
       StartupRecoveryFloorRetainsFiniteNegativeSignedDistance) {
  auto provider = solver(false, 1.0);
  provider.configure_collision_constraint(0.06, {}, {}, true, 1);
  provider.set_non_worsening_collision_floor_enabled(true);
  const Eigen::Vector2d penetrating(0.08, 0.0);
  const auto before = provider.evaluate_collision_pair_distances(penetrating);
  ASSERT_TRUE(before.has_value());
  ASSERT_LT(before->front().distance, 0.0);
  provider.initialize_collision_recovery_floors(penetrating);
  const auto after = provider.evaluate_collision_pair_distances(penetrating);
  ASSERT_TRUE(after.has_value());
  EXPECT_DOUBLE_EQ(*after->front().persistent_floor_m, before->front().distance);
}

TEST_F(CollisionAccelerationCacheTest,
       StartupBoundaryRestAcceptsZeroPrescribedAcceleration) {
  const Eigen::Vector2d q = Eigen::Vector2d::Zero();
  const Eigen::Vector2d dq = Eigen::Vector2d::Zero();
  const Eigen::Vector2d ddq = Eigen::Vector2d::Zero();
  for (const bool acceleration_enabled : {false, true}) {
    auto provider = solver(acceleration_enabled, 1.0);
    provider.configure_collision_constraint(0.06, {}, {}, true, 1);
    provider.set_non_worsening_collision_floor_enabled(true);
    provider.set_collision_repulsion_deadband(0.0);
    provider.initialize_collision_recovery_floors(q);
    provider.set_previous_joint_velocities(dq);
    robot_->update_kinematics(q, dq);
    const auto pair = provider.evaluate_collision_pair_distances(q);
    ASSERT_TRUE(pair.has_value());
    const auto boundary = rows(provider);
    ASSERT_TRUE(boundary.has_value());
    ASSERT_EQ(boundary->coefficient_matrix.rows(), 1);
    const double floor = *pair->front().persistent_floor_m;
    std::ostringstream evidence;
    evidence << std::setprecision(17)
             << "provider_acceleration=" << acceleration_enabled
             << " startup_distance=" << pair->front().distance
             << " floor=" << floor << " row_distance=" << boundary->distance
             << " row_reserve=" << kCollisionTolerance
             << " row_lower=" << boundary->lower_bounds(0)
             << " dt=" << provider.get_dt();
    RecordProperty(acceleration_enabled ? "enabled_evidence" : "disabled_evidence",
                   evidence.str());
    SCOPED_TRACE(evidence.str());
    EXPECT_DOUBLE_EQ(boundary->distance, floor);
    EXPECT_LE(boundary->lower_bounds(0), 0.0);
    AccelerationSolver acceleration(robot_);
    AccelerationSolveOptions options;
    options.acceleration_limits_override = Eigen::Vector2d::Ones();
    VelocityCollisionLiftOptions lift;
    lift.validation_substeps = 4;
    const auto result = acceleration.validate_candidate_with_velocity_collision(
        provider, q, dq, 0.01, ddq, options, lift);
    EXPECT_EQ(result.status, SolverStatus::kSuccess) << result.status_message;
    if (result.status != SolverStatus::kSuccess) continue;
    EXPECT_TRUE(result.joint_accelerations.isZero(0.0));
    EXPECT_TRUE(result.joint_velocities_next.isZero(0.0));
    EXPECT_TRUE(result.q_solution.isApprox(q, 0.0));
    EXPECT_TRUE(result.acceleration_limits_applied);
    EXPECT_TRUE(result.velocity_collision_lift_applied);
    EXPECT_TRUE(result.collision_endpoint_validated);
    EXPECT_EQ(result.collision_validation_samples, 4u);
  }
}

TEST_F(CollisionAccelerationCacheTest,
       StartupBoundaryRestVelocityRowAllowsStationaryState) {
  auto provider = solver(false, 1.0);
  provider.configure_collision_constraint(0.06, {}, {}, true, 1);
  provider.set_non_worsening_collision_floor_enabled(true);
  provider.set_collision_repulsion_deadband(0.0);
  provider.enable_collision_pair_cache(false);
  const Eigen::Vector2d q = Eigen::Vector2d::Zero();
  provider.initialize_collision_recovery_floors(q);
  robot_->update_kinematics(q, Eigen::Vector2d::Zero());
  const auto pair = provider.evaluate_collision_pair_distances(q);
  const auto boundary = detail::VelocityCollisionConstraintProvider(provider).compute();
  ASSERT_TRUE(pair.has_value());
  ASSERT_TRUE(boundary.has_value());
  std::ostringstream evidence;
  evidence << std::setprecision(17) << "distance=" << boundary->distance
           << " floor=" << *pair->front().persistent_floor_m
           << " row_reserve=" << kCollisionTolerance
           << " row_lower=" << boundary->lower_bounds(0);
  RecordProperty("boundary_evidence", evidence.str());
  SCOPED_TRACE(evidence.str());
  EXPECT_DOUBLE_EQ(boundary->distance, *pair->front().persistent_floor_m);
  EXPECT_LE(boundary->lower_bounds(0), 0.0);
  EXPECT_GE(boundary->upper_bounds(0), 0.0);
}

TEST_F(CollisionAccelerationCacheTest,
       StartupBoundaryRestKeepsViolationRecoveryAndBufferedApproach) {
  auto provider = solver(false, 1.0);
  provider.configure_collision_constraint(0.06, {}, {}, true, 1);
  provider.set_non_worsening_collision_floor_enabled(true);
  provider.set_collision_repulsion_deadband(0.0);
  provider.initialize_collision_recovery_floors(Eigen::Vector2d::Zero());
  provider.set_previous_joint_velocities(Eigen::Vector2d::Zero());
  robot_->update_kinematics(Eigen::Vector2d(0.001, 0.0), Eigen::Vector2d::Zero());
  const auto violated = rows(provider);
  ASSERT_TRUE(violated.has_value());
  EXPECT_GT(violated->lower_bounds(0), 0.0);
  EXPECT_NEAR(violated->lower_bounds(0), 0.022, 1e-10);
  robot_->update_kinematics(Eigen::Vector2d(-0.002, 0.0), Eigen::Vector2d::Zero());
  const auto approaching = rows(provider);
  ASSERT_TRUE(approaching.has_value());
  EXPECT_NEAR(approaching->lower_bounds(0), -0.19, 1e-10);
}

TEST_F(CollisionAccelerationCacheTest,
       StartupBoundaryRestIsRelativeToDeclaredNegativeSignedFloor) {
  auto provider = solver(false, 1.0);
  provider.configure_collision_constraint(0.06, {}, {}, true, 1);
  provider.set_non_worsening_collision_floor_enabled(true);
  provider.set_collision_repulsion_deadband(0.0);
  const Eigen::Vector2d q(0.08, 0.0);
  provider.initialize_collision_recovery_floors(q);
  provider.set_previous_joint_velocities(Eigen::Vector2d::Zero());
  robot_->update_kinematics(q, Eigen::Vector2d::Zero());
  const auto pair = provider.evaluate_collision_pair_distances(q);
  const auto boundary = rows(provider);
  ASSERT_TRUE(pair.has_value());
  ASSERT_TRUE(boundary.has_value());
  ASSERT_LT(pair->front().distance, 0.0);
  EXPECT_DOUBLE_EQ(boundary->distance, *pair->front().persistent_floor_m);
  EXPECT_LE(boundary->lower_bounds(0), 0.0);
  // This retains a declared signed floor; it makes no nonpenetration claim.
  robot_->update_kinematics(Eigen::Vector2d(0.081, 0.0), Eigen::Vector2d::Zero());
  const auto violated = rows(provider);
  ASSERT_TRUE(violated.has_value());
  EXPECT_GT(violated->lower_bounds(0), 0.0);
  EXPECT_NEAR(violated->lower_bounds(0), 0.11, 1e-10);
}

TEST(CurrentCollisionCatalogTest,
     StartupFloorsUseRuntimeIntersectionAndRejectInvalidEvidenceAtomically) {
  const std::string path = "/tmp/embodik_startup_floor_catalog.urdf";
  {
    std::ofstream file(path);
    file << R"(<robot name="startup_floor_catalog"><link name="world"/>
      <link name="near"><collision><geometry><sphere radius="0.05"/></geometry></collision></link>
      <joint name="near_fixed" type="fixed"><parent link="world"/><child link="near"/></joint>
      <link name="far"><collision><geometry><sphere radius="0.05"/></geometry></collision></link>
      <joint name="far_fixed" type="fixed"><parent link="world"/><child link="far"/><origin xyz="0.6 0 0"/></joint>
      <link name="moving"><collision><geometry><sphere radius="0.05"/></geometry></collision></link>
      <joint name="slide" type="prismatic"><parent link="world"/><child link="moving"/>
      <origin xyz="0.2 0 0"/><axis xyz="1 0 0"/><limit lower="0" upper="1" effort="1" velocity="1"/></joint>
      </robot>)";
  }
  auto robot = std::make_shared<RobotModel>(path, false);
  KinematicsSolver provider(robot);
  provider.configure_collision_constraint(0.15, {}, {}, true, 1);
  provider.set_non_worsening_collision_floor_enabled(true);
  const auto allowed = provider.get_active_collision_pairs().size();
  ASSERT_GT(allowed, 1u);
  robot->collision_data()->activeCollisionPairs[0] = false;
  provider.initialize_collision_recovery_floors(Eigen::VectorXd::Zero(1));
  const auto initialized = provider.evaluate_collision_pair_distances(Eigen::VectorXd::Zero(1));
  ASSERT_TRUE(initialized.has_value());
  EXPECT_EQ(initialized->size(), allowed-1);
  for (const auto &pair : *initialized) EXPECT_TRUE(pair.persistent_floor_m.has_value());

  // A late invalid threshold must not commit floors for earlier valid records.
  robot->collision_data()->activeCollisionPairs[0] = true;
  KinematicsSolver atomic(robot);
  atomic.configure_collision_constraint(0.15, {}, {}, true, 1);
  atomic.set_non_worsening_collision_floor_enabled(true);
  atomic.set_collision_pair_min_distance("far", "moving", NAN, false);
  EXPECT_THROW(atomic.initialize_collision_recovery_floors(Eigen::VectorXd::Zero(1)), std::invalid_argument);
  const auto unchanged = atomic.evaluate_collision_pair_distances(Eigen::VectorXd::Zero(1));
  ASSERT_TRUE(unchanged.has_value());
  for (const auto &pair : *unchanged) EXPECT_FALSE(pair.persistent_floor_m.has_value());
  atomic.clear_collision_pair_min_distance("far", "moving");
  EXPECT_NO_THROW(atomic.initialize_collision_recovery_floors(Eigen::VectorXd::Zero(1)));
  std::remove(path.c_str());
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

TEST_F(CollisionAccelerationCacheTest,
       SeedBoundsPreserveExactFloorInitializationAndStructuralReturn) {
  auto exact = solver(false, 1.0);
  auto bounded = solver(false, 1.0);
  for (auto *provider : {&exact, &bounded}) {
    provider->configure_collision_constraint(0.06, {}, {}, true, 1);
    provider->set_non_worsening_collision_floor_enabled(true);
    provider->set_collision_structural_floor(0.005);
  }
  bounded.enable_sphere_broadphase(true);
  // First seed is below nominal clearance. It must initialize the structural
  // floor from exact evidence, which stays unchanged on an outward excursion.
  const std::vector<std::pair<double, double>> cases{
      {0.0, 0.02}, {0.001, 0.02}, {-0.03, 0.02}};
  for (std::size_t index = 0; index < cases.size(); ++index) {
    const auto &[seed_position, sample_position] = cases[index];
    const Eigen::Vector2d seed(seed_position, 0.0);
    const std::vector<Eigen::VectorXd> samples{Eigen::Vector2d(sample_position, 0.0)};
    const auto a = detail::VelocityCollisionConstraintProvider(exact).validate_samples(seed, samples);
    const auto b = detail::VelocityCollisionConstraintProvider(bounded).validate_samples(seed, samples);
    EXPECT_TRUE(a.acceptable);
    EXPECT_EQ(b.status, a.status);
    EXPECT_EQ(b.acceptable, a.acceptable);
    EXPECT_EQ(b.initial_exact_distance_queries, index == 0 ? 1u : 0u);
  }
}

TEST_F(CollisionAccelerationCacheTest,
       PendingOverrideUsesExactSeedAndPromotesAtTheSameClearance) {
  auto exact = solver(false, 1.0);
  auto bounded = solver(false, 1.0);
  bounded.enable_sphere_broadphase(true);
  for (auto *provider : {&exact, &bounded}) {
    provider->set_collision_pair_min_distance("left", "right", 0.06, true);
  }
  const std::vector<double> seeds{0.0, -0.02, -0.03};
  for (std::size_t index = 0; index < seeds.size(); ++index) {
    const Eigen::Vector2d seed(seeds[index], 0.0);
    const auto a = detail::VelocityCollisionConstraintProvider(exact).validate_samples(seed, {seed});
    const auto b = detail::VelocityCollisionConstraintProvider(bounded).validate_samples(seed, {seed});
    EXPECT_TRUE(a.acceptable);
    EXPECT_EQ(b.status, a.status);
    EXPECT_EQ(b.acceptable, a.acceptable);
    EXPECT_EQ(b.initial_exact_distance_queries, index < 2 ? 1u : 0u);
    EXPECT_EQ(bounded.get_collision_pair_min_distance_overrides(),
              exact.get_collision_pair_min_distance_overrides());
    EXPECT_EQ(bounded.get_collision_pair_min_distance_overrides().size(), 1u);
    if (index == 1) {
      const std::vector<Eigen::VectorXd> below_promoted_floor{Eigen::Vector2d::Zero()};
      const auto a_return = detail::VelocityCollisionConstraintProvider(exact)
                                .validate_samples(seed, below_promoted_floor);
      const auto b_return = detail::VelocityCollisionConstraintProvider(bounded)
                                .validate_samples(seed, below_promoted_floor);
      EXPECT_FALSE(a_return.acceptable);
      EXPECT_EQ(b_return.status, a_return.status);
      EXPECT_DOUBLE_EQ(b_return.failed_required_distance, 0.06);
    }
  }
}

TEST_F(CollisionAccelerationCacheTest, NearBoundarySeedsRetainExactEvidence) {
  auto bounded = solver(false, 1.0);
  bounded.enable_sphere_broadphase(true);
  const Eigen::Vector2d seed(0.044, 0.0);
  const auto result = detail::VelocityCollisionConstraintProvider(bounded).validate_samples(seed, {seed});
  EXPECT_TRUE(result.acceptable);
  EXPECT_EQ(result.initial_exact_distance_queries, 1u);
}

TEST_F(CollisionAccelerationCacheTest, SeedReserveBoundaryPreservesExactAcceptance) {
  for (double position : {0.0349, 0.035, 0.0351}) {
    SCOPED_TRACE(position);
    auto exact = solver(false, 1.0);
    auto bounded = solver(false, 1.0);
    bounded.enable_sphere_broadphase(true);
    const Eigen::Vector2d seed(position, 0.0);
    const auto a = detail::VelocityCollisionConstraintProvider(exact).validate_samples(seed, {seed});
    const auto b = detail::VelocityCollisionConstraintProvider(bounded).validate_samples(seed, {seed});
    EXPECT_TRUE(a.acceptable);
    EXPECT_EQ(b.status, a.status);
    EXPECT_EQ(b.acceptable, a.acceptable);
    if (position < 0.035) EXPECT_EQ(b.initial_exact_distance_queries, 0u);
    if (position > 0.035) EXPECT_EQ(b.initial_exact_distance_queries, 1u);
  }
}

TEST_F(CollisionAccelerationCacheTest, NegativeFloorsPreserveOverlapAndRecoveryDecisions) {
  for (const auto &[seed_position, sample_position] :
       std::vector<std::pair<double, double>>{
           {0.0, 0.065}, {0.0, 0.075}, {0.065, 0.065}, {0.065, 0.075},
           {0.08, 0.075}, {0.08, 0.085}}) {
    SCOPED_TRACE(seed_position);
    SCOPED_TRACE(sample_position);
    auto exact = solver(false, 1.0);
    auto bounded = solver(false, 1.0);
    for (auto *provider : {&exact, &bounded})
      provider->set_collision_pair_min_distance("left", "right", -0.02, false);
    bounded.enable_sphere_broadphase(true);
    const Eigen::Vector2d seed(seed_position, 0.0), sample(sample_position, 0.0);
    const auto a = detail::VelocityCollisionConstraintProvider(exact).validate_samples(seed, {sample});
    const auto b = detail::VelocityCollisionConstraintProvider(bounded).validate_samples(seed, {sample});
    EXPECT_EQ(b.status, a.status);
    EXPECT_EQ(b.acceptable, a.acceptable);
    EXPECT_EQ(b.acceptable, sample_position < 0.075 ||
                           (seed_position == 0.08 && sample_position == 0.075));
    EXPECT_EQ(b.initial_exact_distance_queries, seed_position == 0.0 ? 0u : 1u);
    if (!b.acceptable) {
      EXPECT_NEAR(b.failed_sample_distance, a.failed_sample_distance, 1e-12);
      EXPECT_NEAR(b.failed_required_distance, a.failed_required_distance, 1e-12);
    }
  }
}

TEST_F(CollisionAccelerationCacheTest, PlacementEditDisablesCachedBoundsAndRejectsUnsafeSample) {
  auto bounded = solver(false, 1.0);
  bounded.enable_sphere_broadphase(true);
  const Eigen::Vector2d seed = Eigen::Vector2d::Zero();
  detail::VelocityCollisionConstraintProvider provider(bounded);
  ASSERT_TRUE(provider.validate_samples(seed, {seed}).acceptable);
  auto *geometry = robot_->collision_model();
  geometry->geometryObjects[0].placement.translation().x() = 0.04;
  const auto result = provider.validate_samples(seed, {Eigen::Vector2d(0.02, 0.0)});
  EXPECT_EQ(result.status, SolverStatus::kCollisionViolated);
  EXPECT_FALSE(result.acceptable);
  EXPECT_FALSE(result.initial_state_certificate_reused);
  EXPECT_EQ(result.initial_exact_distance_queries, 1u);
  EXPECT_EQ(result.sample_exact_distance_queries, 1u);
  EXPECT_EQ(result.conservative_bound_certified_pairs, 0u);
  EXPECT_LT(result.failed_sample_distance, 0.005);
}

TEST_F(CollisionAccelerationCacheTest, ShapeEditDisablesCachedBoundsAndRejectsUnsafeSample) {
  auto bounded = solver(false, 1.0);
  bounded.enable_sphere_broadphase(true);
  const Eigen::Vector2d seed = Eigen::Vector2d::Zero();
  detail::VelocityCollisionConstraintProvider provider(bounded);
  ASSERT_TRUE(provider.validate_samples(seed, {seed}).acceptable);
  auto *geometry = robot_->collision_model();
  auto shape = std::dynamic_pointer_cast<coal::Sphere>(geometry->geometryObjects[0].geometry);
  ASSERT_TRUE(shape);
  shape->radius = 0.085;
  shape->computeLocalAABB();
  const auto result = provider.validate_samples(seed, {Eigen::Vector2d(0.02, 0.0)});
  EXPECT_EQ(result.status, SolverStatus::kCollisionViolated);
  EXPECT_FALSE(result.acceptable);
  EXPECT_FALSE(result.initial_state_certificate_reused);
  EXPECT_EQ(result.initial_exact_distance_queries, 1u);
  EXPECT_EQ(result.sample_exact_distance_queries, 1u);
  EXPECT_EQ(result.conservative_bound_certified_pairs, 0u);
  EXPECT_LT(result.failed_sample_distance, 0.005);
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
