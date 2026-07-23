#include <embodik/acceleration_solver.hpp>
#include <embodik/kinematics_solver.hpp>

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdio>
#include <fstream>
#include <limits>
#include <memory>
#include <string>

namespace embodik::test {
namespace {

Eigen::VectorXd vector(std::initializer_list<double> values) {
  Eigen::VectorXd out(static_cast<Eigen::Index>(values.size()));
  Eigen::Index index = 0;
  for (double value : values) {
    out(index++) = value;
  }
  return out;
}

void expect_no_motion_outputs(const AccelerationSolverResult &result) {
  EXPECT_TRUE(result.solution.empty());
  EXPECT_EQ(result.joint_accelerations.size(), 0);
  EXPECT_EQ(result.joint_velocities_next.size(), 0);
  EXPECT_EQ(result.q_solution.size(), 0);
  EXPECT_EQ(result.predicted_torques.size(), 0);
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

std::string single_pair_urdf(bool with_collision = true) {
  const std::string collision =
      with_collision ? R"(<collision><geometry><sphere radius="0.05"/></geometry></collision>)"
                     : "";
  return R"(<?xml version="1.0"?>
<robot name="acceleration_velocity_collision_lift_robot">
  <link name="world"/>
  <link name="obstacle">)" +
         collision + R"(</link>
  <joint name="obstacle_fixed" type="fixed">
    <parent link="world"/>
    <child link="obstacle"/>
  </joint>
  <link name="moving">)" +
         collision + R"(</link>
  <joint name="moving_slide" type="prismatic">
    <parent link="world"/>
    <child link="moving"/>
    <origin xyz="0.12 0 0"/>
    <axis xyz="1 0 0"/>
    <limit lower="-1.0" upper="1.0" effort="1000" velocity="100"/>
  </joint>
</robot>)";
}

std::string multi_pair_urdf() {
  return R"(<?xml version="1.0"?>
<robot name="acceleration_velocity_collision_lift_multi_robot">
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

AccelerationSolveOptions options_with_limits(double limit) {
  AccelerationSolveOptions options;
  options.acceleration_limits_override = vector({limit});
  options.apply_position_limits = false;
  options.apply_velocity_limits = false;
  return options;
}

VelocityCollisionLiftOptions lift_options(int substeps) {
  VelocityCollisionLiftOptions options;
  options.validation_substeps = substeps;
  return options;
}

class AccelerationVelocityCollisionLiftTest : public ::testing::Test {
protected:
  void SetUp() override {
    urdf_ = std::make_unique<ScopedUrdf>(
        "embodik_acceleration_velocity_collision_lift.urdf",
        single_pair_urdf());
    robot_ = std::make_shared<RobotModel>(urdf_->path(), false);
    robot_->set_gravity(Eigen::Vector3d::Zero());
  }

  KinematicsSolver collision_solver(double min_distance,
                                    bool proximity_gated = false) {
    KinematicsSolver solver(robot_);
    solver.set_dt(0.01);
    solver.configure_collision_constraint(min_distance, {}, {}, true, 1);
    solver.set_collision_tuning_mode(CollisionTuningMode::kPrecise);
    solver.set_proximity_gated_collision_activation_enabled(proximity_gated);
    if (proximity_gated) {
      solver.set_collision_constraint_activation_multiplier(5.0);
    }
    return solver;
  }

  std::unique_ptr<ScopedUrdf> urdf_;
  std::shared_ptr<RobotModel> robot_;
};

TEST_F(AccelerationVelocityCollisionLiftTest,
       CapabilitySeparatesLiftedFromNativeCollisionSupport) {
  const auto capabilities = AccelerationSolver::capabilities();
#ifdef PINOCCHIO_WITH_HPP_FCL
  EXPECT_TRUE(capabilities.supports_velocity_collision_lift);
#else
  EXPECT_FALSE(capabilities.supports_velocity_collision_lift);
#endif
  EXPECT_FALSE(capabilities.supports_collision_constraints);
}

TEST_F(AccelerationVelocityCollisionLiftTest, FlagsAndExactAccountingPass) {
  AccelerationSolver solver(robot_);
  auto collision = collision_solver(0.01, false);

  const auto result = solver.solve_with_velocity_collision(
      collision, vector({0.0}), vector({0.0}), 0.01,
      options_with_limits(10.0), lift_options(4));

  ASSERT_EQ(result.status, SolverStatus::kSuccess) << result.status_message;
  EXPECT_TRUE(result.velocity_collision_lift_applied);
  EXPECT_TRUE(result.collision_endpoint_validated);
  EXPECT_FALSE(result.collision_step_certified);
  EXPECT_EQ(result.collision_validation_samples, 4U);
  EXPECT_EQ(result.collision_validation_allowed_pairs, 1U);
  EXPECT_EQ(result.collision_validation_pairs_checked, 4U);
  EXPECT_EQ(result.collision_validation_exact_queries, 5U);
  EXPECT_EQ(result.collision_lift_pairs_considered, 1U);
  EXPECT_EQ(result.collision_lift_row_pairs, 1U);
  EXPECT_GE(result.collision_lift_row_exact_queries, 1U);
}

TEST_F(AccelerationVelocityCollisionLiftTest,
       DefaultOptionsValidateTheEndpointWithoutCertification) {
  AccelerationSolver solver(robot_);
  auto collision = collision_solver(0.01, false);

  const auto result = solver.solve_with_velocity_collision(
      collision, vector({0.0}), vector({0.0}), 0.01,
      options_with_limits(10.0));

  ASSERT_EQ(result.status, SolverStatus::kSuccess) << result.status_message;
  EXPECT_TRUE(result.velocity_collision_lift_applied);
  EXPECT_TRUE(result.collision_endpoint_validated);
  EXPECT_FALSE(result.collision_step_certified);
  EXPECT_EQ(result.collision_validation_samples, 1U);
  EXPECT_EQ(result.collision_validation_pairs_checked, 1U);
}

TEST_F(AccelerationVelocityCollisionLiftTest,
       NoActiveVelocityRowStillValidatesAllowedPairs) {
  AccelerationSolver solver(robot_);
  auto collision = collision_solver(0.005, true);

  const auto result = solver.solve_with_velocity_collision(
      collision, vector({0.2}), vector({0.0}), 0.01,
      options_with_limits(10.0), lift_options(3));

  ASSERT_EQ(result.status, SolverStatus::kSuccess) << result.status_message;
  EXPECT_TRUE(result.velocity_collision_lift_applied);
  EXPECT_TRUE(result.collision_endpoint_validated);
  EXPECT_EQ(result.collision_lift_pairs_considered, 1U);
  EXPECT_EQ(result.collision_lift_row_pairs, 0U);
  EXPECT_EQ(result.collision_validation_samples, 3U);
  EXPECT_EQ(result.collision_validation_allowed_pairs, 1U);
  EXPECT_EQ(result.collision_validation_exact_queries, 4U);
}

TEST_F(AccelerationVelocityCollisionLiftTest,
       DeferredOverridePromotesDuringNoRowValidation) {
  AccelerationSolver solver(robot_);
  auto collision = collision_solver(0.005, true);
  const auto active_pairs = collision.get_active_collision_pairs();
  ASSERT_EQ(active_pairs.size(), 1U);
  const Eigen::VectorXd q = vector({0.2});

  const auto warm = collision.solve_velocity(q, true);
  ASSERT_EQ(warm.status, SolverStatus::kSuccess) << warm.status_message;
  collision.set_collision_pair_min_distance("obstacle", "moving", 0.03, true);

  auto options = options_with_limits(100.0);
  AffineAccelerationConstraint forced_approach;
  forced_approach.source_id = "forced_deferred_floor_approach";
  forced_approach.coefficient_matrix = Eigen::MatrixXd::Ones(1, 1);
  forced_approach.affine_bias = vector({0.0});
  forced_approach.lower_bounds = vector({-40.0});
  forced_approach.upper_bounds = vector({-40.0});
  options.affine_constraints.push_back(std::move(forced_approach));

  const auto result = solver.solve_with_velocity_collision(
      collision, q, vector({0.0}), 0.1, options);

  EXPECT_EQ(result.status, SolverStatus::kCollisionViolated)
      << result.status_message;
  EXPECT_TRUE(result.velocity_collision_lift_applied);
  EXPECT_FALSE(result.collision_endpoint_validated);
  EXPECT_EQ(result.collision_lift_row_pairs, 0U);
  expect_no_motion_outputs(result);
}

TEST_F(AccelerationVelocityCollisionLiftTest,
       NoRowValidationPersistsNonWorseningFloorAcrossTicks) {
  AccelerationSolver solver(robot_);
  auto collision = collision_solver(0.03, true);
  collision.set_non_worsening_collision_floor_enabled(true);

  const auto seed = solver.solve_with_velocity_collision(
      collision, vector({0.2}), vector({0.0}), 0.1,
      options_with_limits(100.0));
  ASSERT_EQ(seed.status, SolverStatus::kSuccess) << seed.status_message;
  ASSERT_EQ(seed.collision_lift_row_pairs, 0U);

  ASSERT_TRUE(collision.set_collision_min_distance(0.05));
  auto hold_options = options_with_limits(100.0);
  AffineAccelerationConstraint zero_acceleration;
  zero_acceleration.source_id = "persistent_floor_zero_acceleration";
  zero_acceleration.coefficient_matrix = Eigen::MatrixXd::Ones(1, 1);
  zero_acceleration.affine_bias = vector({0.0});
  zero_acceleration.lower_bounds = vector({0.0});
  zero_acceleration.upper_bounds = vector({0.0});
  hold_options.affine_constraints.push_back(std::move(zero_acceleration));
  const auto worsened = solver.solve_with_velocity_collision(
      collision, vector({0.0}), vector({0.0}), 0.1,
      hold_options);

  EXPECT_EQ(worsened.status, SolverStatus::kInfeasible)
      << worsened.status_message;
  EXPECT_EQ(worsened.collision_lift_row_pairs, 1U);
  expect_no_motion_outputs(worsened);
}

TEST_F(AccelerationVelocityCollisionLiftTest,
       IncludeAndExcludeParityControlsValidatedPairs) {
  urdf_ = std::make_unique<ScopedUrdf>(
      "embodik_acceleration_velocity_collision_lift_multi.urdf",
      multi_pair_urdf());
  robot_ = std::make_shared<RobotModel>(urdf_->path(), false);
  AccelerationSolver solver(robot_);
  KinematicsSolver all(robot_);
  all.configure_collision_constraint(0.005, {}, {}, true, 2);
  const auto active_pairs = all.get_active_collision_pairs();
  ASSERT_GE(active_pairs.size(), 2U);

  KinematicsSolver include(robot_);
  include.configure_collision_constraint(0.005, {active_pairs.front()}, {},
                                         true, 2);
  include.set_collision_tuning_mode(CollisionTuningMode::kPrecise);
  include.set_proximity_gated_collision_activation_enabled(false);
  const auto include_result = solver.solve_with_velocity_collision(
      include, vector({0.0}), vector({0.0}), 0.01,
      options_with_limits(10.0), lift_options(2));
  ASSERT_EQ(include_result.status, SolverStatus::kSuccess)
      << include_result.status_message;
  EXPECT_EQ(include_result.collision_validation_allowed_pairs, 1U);
  EXPECT_EQ(include_result.collision_validation_exact_queries, 3U);

  KinematicsSolver exclude(robot_);
  exclude.configure_collision_constraint(0.005, {}, {active_pairs.front()},
                                         true, 2);
  exclude.set_collision_tuning_mode(CollisionTuningMode::kPrecise);
  exclude.set_proximity_gated_collision_activation_enabled(false);
  const auto exclude_result = solver.solve_with_velocity_collision(
      exclude, vector({0.0}), vector({0.0}), 0.01,
      options_with_limits(10.0), lift_options(2));
  ASSERT_EQ(exclude_result.status, SolverStatus::kSuccess)
      << exclude_result.status_message;
  EXPECT_EQ(exclude_result.collision_validation_allowed_pairs,
            active_pairs.size() - 1U);
}

TEST_F(AccelerationVelocityCollisionLiftTest,
       PerPairFloorAndInitiallyViolatedNonWorseningAreAccepted) {
  AccelerationSolver solver(robot_);
  auto collision = collision_solver(0.005, false);
  const auto active_pairs = collision.get_active_collision_pairs();
  ASSERT_EQ(active_pairs.size(), 1U);
  collision.set_collision_pair_min_distance("obstacle", "moving", 0.03,
                                            false);
  collision.set_non_worsening_collision_floor_enabled(true);

  const auto result = solver.solve_with_velocity_collision(
      collision, vector({0.0}), vector({0.0}), 0.01,
      options_with_limits(10.0), lift_options(2));

  ASSERT_EQ(result.status, SolverStatus::kSuccess) << result.status_message;
  EXPECT_TRUE(result.collision_endpoint_validated);
  EXPECT_EQ(result.collision_validation_allowed_pairs, 1U);
}

TEST_F(AccelerationVelocityCollisionLiftTest,
       UnsafeIntermediateSampleFailsClosedAndClearsTorqueOutputs) {
  AccelerationSolver solver(robot_);
  auto collision = collision_solver(0.005, true);
  auto options = options_with_limits(100.0);
  options.effort_constraints = EffortConstraintOptions{};
  options.effort_constraints->limits_override = vector({1e9});
  AffineAccelerationConstraint forced_turnaround;
  forced_turnaround.source_id = "forced_turnaround";
  forced_turnaround.coefficient_matrix = Eigen::MatrixXd::Ones(1, 1);
  forced_turnaround.affine_bias = vector({0.0});
  forced_turnaround.lower_bounds = vector({96.0});
  forced_turnaround.upper_bounds = vector({96.0});
  options.affine_constraints.push_back(std::move(forced_turnaround));

  const auto result = solver.solve_with_velocity_collision(
      collision, vector({0.04}), vector({-4.8}), 0.1, options,
      lift_options(8));

  EXPECT_EQ(result.status, SolverStatus::kCollisionViolated)
      << result.status_message;
  EXPECT_TRUE(result.velocity_collision_lift_applied);
  EXPECT_FALSE(result.collision_endpoint_validated);
  EXPECT_EQ(result.collision_validation_samples, 2U);
  EXPECT_EQ(result.collision_validation_pairs_checked, 2U);
  expect_no_motion_outputs(result);
}

TEST_F(AccelerationVelocityCollisionLiftTest,
       InvalidInputsAndSharedRobotMismatchFailClosed) {
  AccelerationSolver solver(robot_);
  auto collision = collision_solver(0.01, false);

  const auto bad_substeps = solver.solve_with_velocity_collision(
      collision, vector({0.0}), vector({0.0}), 0.01,
      options_with_limits(10.0), lift_options(0));
  EXPECT_EQ(bad_substeps.status, SolverStatus::kInvalidInput);
  EXPECT_FALSE(bad_substeps.velocity_collision_lift_applied);
  expect_no_motion_outputs(bad_substeps);

  const auto bad_dt = solver.solve_with_velocity_collision(
      collision, vector({0.0}), vector({0.0}), 0.0,
      options_with_limits(10.0), lift_options(1));
  EXPECT_EQ(bad_dt.status, SolverStatus::kInvalidInput);
  expect_no_motion_outputs(bad_dt);

  Eigen::VectorXd nonfinite_q = vector({0.0});
  nonfinite_q(0) = std::numeric_limits<double>::quiet_NaN();
  const auto nonfinite = solver.solve_with_velocity_collision(
      collision, nonfinite_q, vector({0.0}), 0.01,
      options_with_limits(10.0), lift_options(1));
  EXPECT_EQ(nonfinite.status, SolverStatus::kNonFiniteInput);
  expect_no_motion_outputs(nonfinite);

  auto other_urdf = std::make_unique<ScopedUrdf>(
      "embodik_acceleration_velocity_collision_lift_other.urdf",
      single_pair_urdf());
  auto other_robot = std::make_shared<RobotModel>(other_urdf->path(), false);
  KinematicsSolver other_collision(other_robot);
  other_collision.configure_collision_constraint(0.01, {}, {}, true, 1);
  const auto mismatch = solver.solve_with_velocity_collision(
      other_collision, vector({0.0}), vector({0.0}), 0.01,
      options_with_limits(10.0), lift_options(1));
  EXPECT_EQ(mismatch.status, SolverStatus::kInvalidInput);
  expect_no_motion_outputs(mismatch);
}

TEST_F(AccelerationVelocityCollisionLiftTest,
       MissingGeometryAndNoConfiguredPairsFailClosed) {
  auto no_collision_urdf = std::make_unique<ScopedUrdf>(
      "embodik_acceleration_velocity_collision_lift_no_collision.urdf",
      single_pair_urdf(false));
  auto no_collision_robot =
      std::make_shared<RobotModel>(no_collision_urdf->path(), false);
  AccelerationSolver no_collision_solver(no_collision_robot);
  KinematicsSolver no_collision_kinematics(no_collision_robot);
  const auto missing_geometry =
      no_collision_solver.solve_with_velocity_collision(
          no_collision_kinematics, vector({0.0}), vector({0.0}), 0.01,
          options_with_limits(10.0), lift_options(1));
  EXPECT_EQ(missing_geometry.status, SolverStatus::kInvalidInput);
  EXPECT_FALSE(missing_geometry.velocity_collision_lift_applied);
  expect_no_motion_outputs(missing_geometry);

  AccelerationSolver solver(robot_);
  KinematicsSolver unconfigured(robot_);
  const auto no_pairs = solver.solve_with_velocity_collision(
      unconfigured, vector({0.0}), vector({0.0}), 0.01,
      options_with_limits(10.0), lift_options(1));
  EXPECT_EQ(no_pairs.status, SolverStatus::kInvalidInput);
  EXPECT_FALSE(no_pairs.velocity_collision_lift_applied);
  expect_no_motion_outputs(no_pairs);
}

TEST_F(AccelerationVelocityCollisionLiftTest,
       ValidationUsesScratchAndPreservesRobotStateAtSolveInput) {
  AccelerationSolver solver(robot_);
  auto collision = collision_solver(0.005, true);
  solver.add_posture_task("move");
  AccelerationTaskReference reference;
  reference.desired_acceleration = vector({0.4});
  solver.set_task_reference("move", reference);

  const Eigen::VectorXd q = vector({0.2});
  const Eigen::VectorXd dq = vector({0.1});
  const auto result = solver.solve_with_velocity_collision(
      collision, q, dq, 0.01, options_with_limits(10.0), lift_options(4));

  ASSERT_EQ(result.status, SolverStatus::kSuccess) << result.status_message;
  EXPECT_TRUE(robot_->get_current_configuration().isApprox(q, 0.0));
  EXPECT_TRUE(robot_->get_current_velocity().isApprox(dq, 0.0));
}

} // namespace
} // namespace embodik::test
