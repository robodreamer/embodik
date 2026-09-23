#include "acceleration_analytic_collision.hpp"

#include <embodik/acceleration_solver.hpp>
#include <embodik/kinematics_solver.hpp>

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <limits>
#include <memory>
#include <string>
#include <vector>

namespace embodik::test {
namespace {

constexpr double kTolerance = 1e-9;

Eigen::VectorXd vector(std::initializer_list<double> values) {
  Eigen::VectorXd out(static_cast<Eigen::Index>(values.size()));
  Eigen::Index index = 0;
  for (double value : values) {
    out(index++) = value;
  }
  return out;
}

void expect_no_executable_outputs(const AccelerationSolverResult &result) {
  EXPECT_TRUE(result.solution.empty());
  EXPECT_EQ(result.joint_accelerations.size(), 0);
  EXPECT_EQ(result.joint_velocities_next.size(), 0);
  EXPECT_EQ(result.q_solution.size(), 0);
  EXPECT_EQ(result.predicted_torques.size(), 0);
}

bool same_unordered_pair(const CollisionGeometryPair &left,
                         const CollisionGeometryPair &right) {
  return (left.geometry_a == right.geometry_a &&
          left.geometry_b == right.geometry_b) ||
         (left.geometry_a == right.geometry_b &&
          left.geometry_b == right.geometry_a);
}

bool diagnostics_match_pair(
    const AccelerationAnalyticCollisionPairDiagnostics &diagnostics,
    const CollisionGeometryPair &pair) {
  return (diagnostics.object_a == pair.geometry_a &&
          diagnostics.object_b == pair.geometry_b) ||
         (diagnostics.object_a == pair.geometry_b &&
          diagnostics.object_b == pair.geometry_a);
}

class ScopedUrdf {
public:
  ScopedUrdf(std::string filename, const std::string &contents)
      : path_("/tmp/" + std::move(filename)) {
    std::ofstream file(path_);
    file << contents;
  }
  ~ScopedUrdf() { std::remove(path_.c_str()); }

  const std::string &path() const { return path_; }

private:
  std::string path_;
};

std::string single_prismatic_sphere_pair_urdf() {
  return R"(<?xml version="1.0"?>
<robot name="native_collision_single_prismatic_sphere_pair">
  <link name="world"/>
  <link name="obstacle">
    <collision name="obstacle_sphere">
      <geometry><sphere radius="0.05"/></geometry>
    </collision>
  </link>
  <joint name="obstacle_fixed" type="fixed">
    <parent link="world"/>
    <child link="obstacle"/>
  </joint>
  <link name="moving">
    <collision name="moving_sphere">
      <geometry><sphere radius="0.05"/></geometry>
    </collision>
  </link>
  <joint name="moving_slide" type="prismatic">
    <parent link="world"/>
    <child link="moving"/>
    <origin xyz="0.12 0 0"/>
    <axis xyz="1 0 0"/>
    <limit lower="-1.0" upper="1.0" effort="1000" velocity="100"/>
  </joint>
</robot>)";
}

std::string multi_prismatic_sphere_pair_urdf() {
  return R"(<?xml version="1.0"?>
<robot name="native_collision_multi_prismatic_sphere_pair">
  <link name="base">
    <collision name="left_sphere">
      <origin xyz="-0.14 0 0"/>
      <geometry><sphere radius="0.05"/></geometry>
    </collision>
    <collision name="right_sphere">
      <origin xyz="0.17 0 0"/>
      <geometry><sphere radius="0.05"/></geometry>
    </collision>
  </link>
  <link name="moving">
    <collision name="moving_sphere">
      <geometry><sphere radius="0.05"/></geometry>
    </collision>
  </link>
  <joint name="moving_slide" type="prismatic">
    <parent link="base"/>
    <child link="moving"/>
    <axis xyz="1 0 0"/>
    <limit lower="-1.0" upper="1.0" effort="1000" velocity="100"/>
  </joint>
</robot>)";
}

std::string moving_moving_prismatic_sphere_pair_urdf() {
  return R"(<?xml version="1.0"?>
<robot name="native_collision_moving_moving_prismatic_sphere_pair">
  <link name="world"/>
  <link name="moving_x">
    <collision name="moving_x_sphere">
      <geometry><sphere radius="0.03"/></geometry>
    </collision>
  </link>
  <joint name="moving_x_slide" type="prismatic">
    <parent link="world"/>
    <child link="moving_x"/>
    <axis xyz="1 0 0"/>
    <limit lower="-1.0" upper="1.0" effort="1000" velocity="100"/>
  </joint>
  <link name="moving_y">
    <collision name="moving_y_sphere">
      <geometry><sphere radius="0.03"/></geometry>
    </collision>
  </link>
  <joint name="moving_y_slide" type="prismatic">
    <parent link="world"/>
    <child link="moving_y"/>
    <origin xyz="0.20 0.20 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="-1.0" upper="1.0" effort="1000" velocity="100"/>
  </joint>
</robot>)";
}

std::string revolute_sphere_pair_urdf() {
  return R"(<?xml version="1.0"?>
<robot name="native_collision_revolute_sphere_pair">
  <link name="world"/>
  <link name="obstacle">
    <collision name="obstacle_sphere">
      <geometry><sphere radius="0.05"/></geometry>
    </collision>
  </link>
  <joint name="obstacle_fixed" type="fixed">
    <parent link="world"/>
    <child link="obstacle"/>
  </joint>
  <link name="moving">
    <collision name="moving_sphere">
      <origin xyz="0.12 0 0"/>
      <geometry><sphere radius="0.05"/></geometry>
    </collision>
  </link>
  <joint name="moving_hinge" type="revolute">
    <parent link="world"/>
    <child link="moving"/>
    <axis xyz="0 0 1"/>
    <limit lower="-1.0" upper="1.0" effort="1000" velocity="100"/>
  </joint>
</robot>)";
}

std::string non_sphere_collision_urdf() {
  return R"(<?xml version="1.0"?>
<robot name="native_collision_non_sphere_pair">
  <link name="world"/>
  <link name="obstacle">
    <collision name="obstacle_box">
      <geometry><box size="0.10 0.10 0.10"/></geometry>
    </collision>
  </link>
  <joint name="obstacle_fixed" type="fixed">
    <parent link="world"/>
    <child link="obstacle"/>
  </joint>
  <link name="moving">
    <collision name="moving_sphere">
      <geometry><sphere radius="0.05"/></geometry>
    </collision>
  </link>
  <joint name="moving_slide" type="prismatic">
    <parent link="world"/>
    <child link="moving"/>
    <origin xyz="0.12 0 0"/>
    <axis xyz="1 0 0"/>
    <limit lower="-1.0" upper="1.0" effort="1000" velocity="100"/>
  </joint>
</robot>)";
}

CollisionConstraintDefinition default_definition(double min_distance = 0.01) {
  CollisionConstraintDefinition definition;
  definition.min_distance = min_distance;
  return definition;
}

CollisionConstraintAccelerationPolicy default_policy() {
  CollisionConstraintAccelerationPolicy policy;
  policy.proximity_activation_enabled = false;
  policy.activation_margin = 0.05;
  policy.maximum_approach_rate = 1.0;
  policy.maximum_inward_acceleration = 10.0;
  policy.minimum_braking_acceleration = 10.0;
  policy.outside_policy =
      CollisionConstraintOutsidePolicy::kRecoverNonWorsening;
  return policy;
}

AccelerationSolveOptions options_with_limits(double limit) {
  AccelerationSolveOptions options;
  options.acceleration_limits_override = vector({limit});
  options.apply_position_limits = false;
  options.apply_velocity_limits = false;
  return options;
}

void set_posture_acceleration(AccelerationSolver &solver,
                              double desired_acceleration) {
  solver.add_posture_task("posture");
  AccelerationTaskReference reference;
  reference.desired_acceleration = vector({desired_acceleration});
  solver.set_task_reference("posture", reference);
}

class AccelerationSolverNativeCollisionTest : public ::testing::Test {
protected:
  void SetUp() override {
    urdf_ = std::make_unique<ScopedUrdf>(
        "embodik_acceleration_native_collision.urdf",
        single_prismatic_sphere_pair_urdf());
    robot_ = std::make_shared<RobotModel>(urdf_->path(), false);
    robot_->set_gravity(Eigen::Vector3d::Zero());
  }

  std::unique_ptr<ScopedUrdf> urdf_;
  std::shared_ptr<RobotModel> robot_;
};

TEST_F(AccelerationSolverNativeCollisionTest,
       CapabilitiesReportOnlyCertifiedPrismaticSphereNativeCollisionSupport) {
  const auto capabilities = AccelerationSolver::capabilities();
  EXPECT_TRUE(capabilities.supports_fixed_base_scalar_joints);
  EXPECT_TRUE(capabilities.supports_collision_constraints);
  EXPECT_TRUE(capabilities.supports_analytic_sphere_collision_constraints);
  EXPECT_FALSE(capabilities.supports_floating_base);
  EXPECT_FALSE(capabilities.supports_dynamic_contact);
}

TEST_F(AccelerationSolverNativeCollisionTest,
       ConfigureClearAndGettersExposeTheActiveNativeCollisionContract) {
  AccelerationSolver solver(robot_);
  auto definition = default_definition(0.02);
  const auto *geometry_model = robot_->collision_model();
  ASSERT_NE(geometry_model, nullptr);
  ASSERT_FALSE(geometry_model->collisionPairs.empty());
  const auto &catalog_pair = geometry_model->collisionPairs.front();
  CollisionGeometryPair configured_pair;
  configured_pair.geometry_a =
      geometry_model->geometryObjects[catalog_pair.first].name;
  configured_pair.geometry_b =
      geometry_model->geometryObjects[catalog_pair.second].name;
  definition.include_pairs.push_back(configured_pair);
  auto policy = default_policy();
  policy.outside_policy = CollisionConstraintOutsidePolicy::kReject;

  solver.configure_collision_constraint(definition, policy);

  EXPECT_TRUE(solver.has_collision_constraint());
  EXPECT_EQ(solver.get_active_collision_pairs().size(), 1U);
  const auto configured_definition =
      solver.get_collision_constraint_definition();
  ASSERT_TRUE(configured_definition.has_value());
  EXPECT_DOUBLE_EQ(configured_definition->min_distance, 0.02);
  const auto configured_policy = solver.get_collision_constraint_policy();
  ASSERT_TRUE(configured_policy.has_value());
  EXPECT_EQ(configured_policy->outside_policy,
            CollisionConstraintOutsidePolicy::kReject);

  auto invalid_definition = definition;
  invalid_definition.include_pairs.clear();
  invalid_definition.include_pairs.push_back(
      {"missing_geometry_a", "missing_geometry_b"});
  EXPECT_THROW(solver.configure_collision_constraint(invalid_definition,
                                                     default_policy()),
               std::invalid_argument);
  EXPECT_TRUE(solver.has_collision_constraint());
  EXPECT_EQ(solver.get_active_collision_pairs().size(), 1U);
  EXPECT_DOUBLE_EQ(solver.get_collision_min_distance(), 0.02);

  solver.clear_collision_constraint();

  EXPECT_FALSE(solver.has_collision_constraint());
  EXPECT_TRUE(solver.get_active_collision_pairs().empty());
}

TEST_F(AccelerationSolverNativeCollisionTest,
       HardRowsOverrideSoftTasksThatWouldAccelerateIntoTheCollisionFloor) {
  AccelerationSolver solver(robot_);
  solver.configure_collision_constraint(default_definition(0.02),
                                        default_policy());
  set_posture_acceleration(solver, -5.0);

  const auto result = solver.solve(vector({0.0}), vector({0.0}), 0.01,
                                   options_with_limits(100.0));

  ASSERT_EQ(result.status, SolverStatus::kSuccess) << result.status_message;
  EXPECT_TRUE(result.native_collision_constraint_applied);
  ASSERT_EQ(result.native_collision_diagnostics.size(), 1U);
  EXPECT_EQ(result.native_collision_diagnostics[0].regime,
            AccelerationCollisionRegime::kExactFloor);
  EXPECT_TRUE(result.native_collision_diagnostics[0].state_rate_shaping_active);
  EXPECT_GE(result.joint_accelerations(0), -kTolerance);
}

TEST_F(AccelerationSolverNativeCollisionTest,
       StrictInteriorContinuousPathIsCertifiedUnderConstantAcceleration) {
  AccelerationSolver solver(robot_);
  solver.configure_collision_constraint(default_definition(0.01),
                                        default_policy());
  set_posture_acceleration(solver, -0.2);

  const auto result = solver.solve(vector({0.05}), vector({-0.05}), 0.1,
                                   options_with_limits(10.0));

  ASSERT_EQ(result.status, SolverStatus::kSuccess) << result.status_message;
  EXPECT_TRUE(result.native_collision_constraint_applied);
  EXPECT_FALSE(result.velocity_collision_lift_applied);
  EXPECT_TRUE(result.collision_step_certified);
  ASSERT_EQ(result.native_collision_diagnostics.size(), 1U);
  EXPECT_EQ(result.native_collision_diagnostics[0].regime,
            AccelerationCollisionRegime::kStrictInterior);
  EXPECT_TRUE(result.native_collision_diagnostics[0].step_certified);
  EXPECT_GT(
      result.native_collision_diagnostics[0].lowest_path_signed_distance_lower_bound,
      result.native_collision_diagnostics[0].minimum_distance);
}

TEST_F(AccelerationSolverNativeCollisionTest,
       NativeCollisionAndJointStateBoxesShareOneAcceptedAcceleration) {
  AccelerationSolver solver(robot_);
  solver.configure_collision_constraint(default_definition(0.01),
                                        default_policy());
  set_posture_acceleration(solver, 5.0);
  AccelerationSolveOptions options;
  options.acceleration_limits_override = vector({100.0});

  const auto result =
      solver.solve(vector({0.50}), vector({0.0}), 0.1, options);

  ASSERT_EQ(result.status, SolverStatus::kSuccess) << result.status_message;
  EXPECT_TRUE(result.native_collision_constraint_applied);
  EXPECT_TRUE(result.collision_step_certified);
  ASSERT_EQ(result.joint_accelerations.size(), 1);
  ASSERT_EQ(result.joint_velocities_next.size(), 1);
  ASSERT_EQ(result.q_solution.size(), 1);
  EXPECT_LE(result.q_solution(0), 1.0 + kTolerance);
  EXPECT_LE(std::abs(result.joint_velocities_next(0)),
            robot_->get_velocity_limits()(0) + kTolerance);
}

TEST_F(AccelerationSolverNativeCollisionTest,
       DirectCertificateRejectsEndpointSafeInteriorCrossing) {
  auto policy = default_policy();
  policy.maximum_approach_rate = 1.0;
  policy.maximum_inward_acceleration = 1.0;
  policy.minimum_braking_acceleration = 1.0;
  const auto definition = default_definition(0.01);
  const auto compiled = detail::compile_analytic_collision_constraint(
      *robot_, definition, policy);
  ASSERT_TRUE(compiled.satisfied()) << compiled.message;
  detail::CollisionDifferentialScratch scratch(*robot_);
  const Eigen::VectorXd lower = vector({-10.0});
  const Eigen::VectorXd upper = vector({10.0});
  const Eigen::VectorXd q = vector({0.03});
  const Eigen::VectorXd dq = vector({-0.20});
  const auto prepared = detail::prepare_analytic_collision_constraint(
      *robot_, scratch, compiled, policy, q, dq, 1.0, lower, upper,
      "direct-current");
  ASSERT_TRUE(prepared.satisfied()) << prepared.message;
  const Eigen::VectorXd acceleration = vector({0.40});
  const Eigen::VectorXd q_next =
      robot_->integrate(q, dq + 0.5 * acceleration);
  const Eigen::VectorXd dq_next = dq + acceleration;

  const auto certificate = detail::certify_analytic_collision_step(
      *robot_, scratch, compiled, policy, prepared, q, dq, acceleration, 1.0,
      q_next, dq_next, lower, upper);

  EXPECT_EQ(certificate.status, SolverStatus::kCollisionViolated)
      << certificate.message;
  EXPECT_FALSE(certificate.step_certified);
}

TEST_F(AccelerationSolverNativeCollisionTest,
       ProximityGateUsesReachableJointBoxInwardAcceleration) {
  AccelerationSolver solver(robot_);
  auto policy = default_policy();
  policy.proximity_activation_enabled = true;
  policy.activation_margin = 0.0;
  policy.maximum_inward_acceleration = 0.01;
  policy.minimum_braking_acceleration = 1.0;
  solver.configure_collision_constraint(default_definition(0.01), policy);

  const auto result = solver.solve(vector({0.03}), vector({0.0}), 0.1,
                                   options_with_limits(10.0));

  ASSERT_EQ(result.status, SolverStatus::kSuccess) << result.status_message;
  ASSERT_EQ(result.native_collision_diagnostics.size(), 1U);
  EXPECT_TRUE(
      result.native_collision_diagnostics[0].state_rate_shaping_active);
  EXPECT_TRUE(result.collision_step_certified);
}

TEST(AccelerationSolverNativeCollisionMovingPairTest,
     MultiAxisMovingMovingSpherePathUsesOneContinuousCertificate) {
  ScopedUrdf urdf("embodik_native_collision_moving_moving.urdf",
                  moving_moving_prismatic_sphere_pair_urdf());
  auto robot = std::make_shared<RobotModel>(urdf.path(), false);
  AccelerationSolver solver(robot);
  auto policy = default_policy();
  policy.proximity_activation_enabled = false;
  solver.configure_collision_constraint(default_definition(0.05), policy);

  AccelerationSolveOptions options;
  options.acceleration_limits_override = vector({10.0, 10.0});
  options.apply_position_limits = false;
  options.apply_velocity_limits = false;
  const auto result = solver.solve(vector({0.0, 0.0}),
                                   vector({0.10, -0.10}), 0.1, options);

  ASSERT_EQ(result.status, SolverStatus::kSuccess) << result.status_message;
  EXPECT_TRUE(result.collision_endpoint_validated);
  EXPECT_TRUE(result.collision_step_certified);
  ASSERT_EQ(result.native_collision_diagnostics.size(), 1U);
  EXPECT_TRUE(result.native_collision_diagnostics[0].step_certified);
  EXPECT_GT(result.native_collision_path_visited_nodes, 0U);
  EXPECT_GT(result.native_collision_certified_intervals, 0U);
}

TEST(AccelerationSolverNativeCollisionMovingPairTest,
     ExactFloorTangentialMotionIsCertifiedWithoutFalseInwardRejection) {
  ScopedUrdf urdf("embodik_native_collision_tangential.urdf",
                  moving_moving_prismatic_sphere_pair_urdf());
  auto robot = std::make_shared<RobotModel>(urdf.path(), false);
  AccelerationSolver solver(robot);
  const double exact_distance = std::sqrt(0.08) - 0.06;
  auto policy = default_policy();
  policy.proximity_activation_enabled = false;
  solver.configure_collision_constraint(
      default_definition(exact_distance), policy);

  AccelerationSolveOptions options;
  options.acceleration_limits_override = vector({10.0, 10.0});
  options.apply_position_limits = false;
  options.apply_velocity_limits = false;
  const auto result = solver.solve(vector({0.0, 0.0}),
                                   vector({-0.10, -0.10}), 0.01, options);

  ASSERT_EQ(result.status, SolverStatus::kSuccess) << result.status_message;
  ASSERT_EQ(result.native_collision_diagnostics.size(), 1U);
  EXPECT_EQ(result.native_collision_diagnostics[0].regime,
            AccelerationCollisionRegime::kExactFloor);
  EXPECT_GE(
      result.native_collision_diagnostics[0]
          .lowest_path_signed_distance_lower_bound,
      result.native_collision_diagnostics[0].current_signed_distance);
  EXPECT_TRUE(result.collision_step_certified);
}

TEST_F(AccelerationSolverNativeCollisionTest,
       ExactFloorStationaryAndOutwardCommandsAreCertified) {
  AccelerationSolver stationary_solver(robot_);
  stationary_solver.configure_collision_constraint(default_definition(0.02),
                                                   default_policy());
  set_posture_acceleration(stationary_solver, 0.0);

  const auto stationary = stationary_solver.solve(
      vector({0.0}), vector({0.0}), 0.01, options_with_limits(10.0));

  ASSERT_EQ(stationary.status, SolverStatus::kSuccess)
      << stationary.status_message;
  EXPECT_TRUE(stationary.collision_step_certified);
  EXPECT_NEAR(stationary.joint_accelerations(0), 0.0, kTolerance);

  AccelerationSolver outward_solver(robot_);
  outward_solver.configure_collision_constraint(default_definition(0.02),
                                                default_policy());
  set_posture_acceleration(outward_solver, 0.0);

  const auto outward = outward_solver.solve(vector({0.0}), vector({0.2}),
                                            0.01, options_with_limits(10.0));

  ASSERT_EQ(outward.status, SolverStatus::kSuccess) << outward.status_message;
  EXPECT_TRUE(outward.collision_step_certified);
  EXPECT_EQ(outward.native_collision_diagnostics[0].regime,
            AccelerationCollisionRegime::kExactFloor);
}

TEST_F(AccelerationSolverNativeCollisionTest,
       BelowLimitRecoverNonWorseningMovesOutwardWhenReachable) {
  AccelerationSolver solver(robot_);
  auto policy = default_policy();
  policy.outside_policy =
      CollisionConstraintOutsidePolicy::kRecoverNonWorsening;
  solver.configure_collision_constraint(default_definition(0.02), policy);
  set_posture_acceleration(solver, 2.0);

  const auto result = solver.solve(vector({-0.01}), vector({0.0}), 0.1,
                                   options_with_limits(10.0));

  ASSERT_EQ(result.status, SolverStatus::kSuccess) << result.status_message;
  ASSERT_EQ(result.native_collision_diagnostics.size(), 1U);
  EXPECT_EQ(result.native_collision_diagnostics[0].regime,
            AccelerationCollisionRegime::kBelowLimitRecovery);
  EXPECT_TRUE(result.native_collision_diagnostics[0].step_certified);
  EXPECT_GT(result.q_solution(0), -0.01);
  EXPECT_GE(
      result.native_collision_diagnostics[0].lowest_path_signed_distance_lower_bound,
      result.native_collision_diagnostics[0].current_signed_distance);
}

TEST_F(AccelerationSolverNativeCollisionTest,
       BelowLimitRejectPolicyFailsClosedWithoutExecutableOutputs) {
  AccelerationSolver solver(robot_);
  auto policy = default_policy();
  policy.outside_policy = CollisionConstraintOutsidePolicy::kReject;
  solver.configure_collision_constraint(default_definition(0.02), policy);

  const auto result = solver.solve(vector({-0.01}), vector({0.0}), 0.1,
                                   options_with_limits(10.0));

  EXPECT_EQ(result.status, SolverStatus::kCollisionViolated);
  EXPECT_TRUE(result.native_collision_constraint_applied);
  expect_no_executable_outputs(result);
}

TEST(AccelerationSolverNativeCollisionCoupledPairTest,
     OpposingExactFloorPairsRequireAnInfeasibleCommonBrakingWitness) {
  ScopedUrdf urdf("embodik_native_collision_coupled_pairs.urdf",
                  multi_prismatic_sphere_pair_urdf());
  auto robot = std::make_shared<RobotModel>(urdf.path(), false);
  AccelerationSolver solver(robot);
  auto policy = default_policy();
  policy.minimum_braking_acceleration = 1.0;
  policy.maximum_inward_acceleration = 1.0;
  solver.configure_collision_constraint(default_definition(0.055), policy);

  const auto result = solver.solve(vector({0.015}), vector({0.0}), 0.01,
                                   options_with_limits(10.0));

  EXPECT_EQ(result.status, SolverStatus::kCollisionViolated)
      << result.status_message;
  EXPECT_TRUE(result.native_collision_constraint_applied);
  EXPECT_EQ(result.native_collision_diagnostics.size(), 2U);
  expect_no_executable_outputs(result);
}

TEST_F(AccelerationSolverNativeCollisionTest,
       IncludeExcludeAndPairMinimumDistancesSelectNativePairsDeterministically) {
  urdf_ = std::make_unique<ScopedUrdf>(
      "embodik_acceleration_native_collision_multi.urdf",
      multi_prismatic_sphere_pair_urdf());
  robot_ = std::make_shared<RobotModel>(urdf_->path(), false);

  AccelerationSolver all_solver(robot_);
  all_solver.configure_collision_constraint(default_definition(0.005),
                                            default_policy());
  const auto all_pairs = all_solver.get_active_collision_pairs();
  ASSERT_GE(all_pairs.size(), 2U);

  auto definition = default_definition(0.005);
  CollisionGeometryPair included_pair;
  included_pair.geometry_a = all_pairs.front().geometry_a;
  included_pair.geometry_b = all_pairs.front().geometry_b;
  definition.include_pairs.push_back(included_pair);
  CollisionGeometryPair excluded_pair;
  excluded_pair.geometry_a = all_pairs.back().geometry_a;
  excluded_pair.geometry_b = all_pairs.back().geometry_b;
  definition.exclude_pairs.push_back(excluded_pair);
  CollisionPairMinimumDistance pair_minimum_distance;
  pair_minimum_distance.pair = included_pair;
  pair_minimum_distance.min_distance = 0.03;
  definition.pair_minimum_distances.push_back(pair_minimum_distance);

  AccelerationSolver filtered_solver(robot_);
  filtered_solver.configure_collision_constraint(definition, default_policy());

  const auto filtered_pairs = filtered_solver.get_active_collision_pairs();
  ASSERT_EQ(filtered_pairs.size(), 1U);
  EXPECT_TRUE(same_unordered_pair(filtered_pairs.front(), all_pairs.front()));

  const auto result = filtered_solver.solve(vector({0.0}), vector({0.0}), 0.01,
                                            options_with_limits(10.0));

  ASSERT_EQ(result.status, SolverStatus::kSuccess) << result.status_message;
  ASSERT_EQ(result.native_collision_diagnostics.size(), 1U);
  EXPECT_TRUE(
      diagnostics_match_pair(result.native_collision_diagnostics[0],
                             all_pairs.front()));
  EXPECT_DOUBLE_EQ(result.native_collision_diagnostics[0].minimum_distance,
                   0.03);

  auto ignored_override_definition = default_definition(0.005);
  ignored_override_definition.include_pairs.push_back(all_pairs.front());
  ignored_override_definition.pair_minimum_distances.push_back(
      {all_pairs.back(), 0.04});
  AccelerationSolver rejected_solver(robot_);
  EXPECT_THROW(rejected_solver.configure_collision_constraint(
                   ignored_override_definition, default_policy()),
               std::invalid_argument);
  EXPECT_FALSE(rejected_solver.has_collision_constraint());
}

TEST_F(AccelerationSolverNativeCollisionTest,
       GeneralizedAllocationPreservesPhysicalNativeCollisionRows) {
  AccelerationSolver solver(robot_);
  solver.configure_collision_constraint(default_definition(0.02),
                                        default_policy());
  set_posture_acceleration(solver, -5.0);
  auto options = options_with_limits(10.0);
  GeneralizedAccelerationAllocation allocation;
  allocation.metric_diagonal = vector({4.0});
  allocation.reference_acceleration = vector({-5.0});
  options.generalized_acceleration_allocation = allocation;

  const auto result =
      solver.solve(vector({0.0}), vector({0.0}), 0.01, options);

  ASSERT_EQ(result.status, SolverStatus::kSuccess) << result.status_message;
  EXPECT_TRUE(result.allocation_diagnostics.applied);
  EXPECT_GE(result.joint_accelerations(0), -kTolerance);
  EXPECT_TRUE(result.collision_step_certified);
}

TEST_F(AccelerationSolverNativeCollisionTest,
       ConfiguredPairIgnoresMutableRobotActiveMask) {
  AccelerationSolver solver(robot_);
  solver.configure_collision_constraint(default_definition(0.01),
                                        default_policy());
  auto *collision_data = robot_->collision_data();
  ASSERT_NE(collision_data, nullptr);
  ASSERT_EQ(collision_data->activeCollisionPairs.size(), 1U);
  collision_data->activeCollisionPairs[0] = false;

  const auto result = solver.solve(vector({0.05}), vector({0.0}), 0.01,
                                   options_with_limits(10.0));

  ASSERT_EQ(result.status, SolverStatus::kSuccess) << result.status_message;
  EXPECT_EQ(result.native_collision_pair_evaluations, 3U);
  EXPECT_TRUE(result.collision_step_certified);
}

TEST(AccelerationSolverNativeCollisionTopologyTest,
     CatalogMutationAfterConfigurationFailsClosed) {
  ScopedUrdf urdf("embodik_native_collision_topology_mutation.urdf",
                  multi_prismatic_sphere_pair_urdf());
  auto robot = std::make_shared<RobotModel>(urdf.path(), false);
  AccelerationSolver solver(robot);
  solver.configure_collision_constraint(default_definition(0.005),
                                        default_policy());
  const auto configured_pairs = solver.get_active_collision_pairs();
  ASSERT_EQ(configured_pairs.size(), 2U);
  robot->apply_collision_exclusions(
      {{configured_pairs.front().geometry_a,
        configured_pairs.front().geometry_b}});

  AccelerationSolveOptions options;
  options.acceleration_limits_override = vector({10.0});
  options.apply_position_limits = false;
  options.apply_velocity_limits = false;
  const auto result =
      solver.solve(vector({0.0}), vector({0.0}), 0.01, options);

  EXPECT_NE(result.status, SolverStatus::kSuccess);
  EXPECT_TRUE(result.native_collision_constraint_applied);
  expect_no_executable_outputs(result);
}

TEST(AccelerationSolverNativeCollisionUnsupportedTest,
     UnsupportedRevoluteOrNonSphereGeometryIsRejectedAtomically) {
  ScopedUrdf revolute_urdf("embodik_native_collision_revolute.urdf",
                           revolute_sphere_pair_urdf());
  auto revolute_robot =
      std::make_shared<RobotModel>(revolute_urdf.path(), false);
  AccelerationSolver revolute_solver(revolute_robot);

  EXPECT_THROW(revolute_solver.configure_collision_constraint(
                   default_definition(0.01), default_policy()),
               std::invalid_argument);
  EXPECT_FALSE(revolute_solver.has_collision_constraint());
  EXPECT_TRUE(revolute_solver.get_active_collision_pairs().empty());

  ScopedUrdf box_urdf("embodik_native_collision_box.urdf",
                      non_sphere_collision_urdf());
  auto box_robot = std::make_shared<RobotModel>(box_urdf.path(), false);
  AccelerationSolver box_solver(box_robot);

  EXPECT_THROW(box_solver.configure_collision_constraint(
                   default_definition(0.01), default_policy()),
               std::invalid_argument);
  EXPECT_FALSE(box_solver.has_collision_constraint());
  EXPECT_TRUE(box_solver.get_active_collision_pairs().empty());
}

TEST_F(AccelerationSolverNativeCollisionTest,
       PredictedPathFailureFailsClosedAndClearsExecutableOutputs) {
  AccelerationSolver solver(robot_);
  auto policy = default_policy();
  policy.maximum_inward_acceleration = 0.01;
  policy.minimum_braking_acceleration = 0.01;
  solver.configure_collision_constraint(default_definition(0.02), policy);
  set_posture_acceleration(solver, -10.0);

  const auto result = solver.solve(vector({0.03}), vector({-1.0}), 0.1,
                                   options_with_limits(0.01));

  EXPECT_EQ(result.status, SolverStatus::kCollisionViolated)
      << result.status_message;
  EXPECT_FALSE(result.collision_endpoint_validated);
  EXPECT_FALSE(result.collision_step_certified);
  expect_no_executable_outputs(result);
}

TEST_F(AccelerationSolverNativeCollisionTest,
       UncertifiedHardFamilyCombinationIsRejectedBeforeSolve) {
  AccelerationSolver solver(robot_);
  solver.configure_collision_constraint(default_definition(0.01),
                                        default_policy());
  auto options = options_with_limits(10.0);
  AffineAccelerationConstraint affine;
  affine.source_id = "unvalidated_native_collision_companion";
  affine.coefficient_matrix = Eigen::MatrixXd::Identity(1, 1);
  affine.affine_bias = vector({0.0});
  affine.lower_bounds = vector({-1.0});
  affine.upper_bounds = vector({1.0});
  options.affine_constraints.push_back(affine);

  const auto result =
      solver.solve(vector({0.05}), vector({0.0}), 0.01, options);

  EXPECT_EQ(result.status, SolverStatus::kInvalidInput);
  EXPECT_TRUE(result.native_collision_constraint_applied);
  EXPECT_NE(result.status_message.find("predicted-state certificate"),
            std::string::npos);
  expect_no_executable_outputs(result);
}

TEST_F(AccelerationSolverNativeCollisionTest,
       SampledVelocityLiftAndNativeCertifiedCollisionModesAreNotConflated) {
  AccelerationSolver native_solver(robot_);
  native_solver.configure_collision_constraint(default_definition(0.01),
                                               default_policy());
  set_posture_acceleration(native_solver, 0.0);

  const auto native_result = native_solver.solve(vector({0.05}), vector({0.0}),
                                                 0.01,
                                                 options_with_limits(10.0));

  ASSERT_EQ(native_result.status, SolverStatus::kSuccess)
      << native_result.status_message;
  EXPECT_TRUE(native_result.native_collision_constraint_applied);
  EXPECT_FALSE(native_result.velocity_collision_lift_applied);
  EXPECT_TRUE(native_result.collision_step_certified);
  EXPECT_EQ(native_result.collision_validation_samples, 0U);

  KinematicsSolver sampled_collision(robot_);
  sampled_collision.configure_collision_constraint(0.01, {}, {}, true, 1);
  sampled_collision.set_proximity_gated_collision_activation_enabled(false);
  AccelerationSolver sampled_solver(robot_);

  const auto sampled_result = sampled_solver.solve_with_velocity_collision(
      sampled_collision, vector({0.05}), vector({0.0}), 0.01,
      options_with_limits(10.0));

  ASSERT_EQ(sampled_result.status, SolverStatus::kSuccess)
      << sampled_result.status_message;
  EXPECT_FALSE(sampled_result.native_collision_constraint_applied);
  EXPECT_TRUE(sampled_result.velocity_collision_lift_applied);
  EXPECT_FALSE(sampled_result.collision_step_certified);
  EXPECT_GE(sampled_result.collision_validation_samples, 1U);
  EXPECT_TRUE(sampled_result.native_collision_diagnostics.empty());
}

TEST_F(AccelerationSolverNativeCollisionTest,
       NonFinitePredictedStateFailureClearsAllExecutableOutputs) {
  AccelerationSolver solver(robot_);
  solver.configure_collision_constraint(default_definition(0.01),
                                        default_policy());
  set_posture_acceleration(solver, 0.0);
  Eigen::VectorXd q = vector({0.0});
  q(0) = std::numeric_limits<double>::quiet_NaN();

  const auto result =
      solver.solve(q, vector({0.0}), 0.01, options_with_limits(10.0));

  EXPECT_EQ(result.status, SolverStatus::kNonFiniteInput);
  EXPECT_FALSE(result.native_collision_constraint_applied);
  expect_no_executable_outputs(result);
}

} // namespace
} // namespace embodik::test
