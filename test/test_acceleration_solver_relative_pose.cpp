#include "acceleration_relative_pose_constraint.hpp"
#include "geometric_constraint_differential.hpp"
#include "so3_log_differential.hpp"

#include <embodik/acceleration_solver.hpp>

#include <gtest/gtest.h>

#include <Eigen/Geometry>
#include <cstdio>
#include <fstream>
#include <limits>
#include <memory>
#include <string>

namespace embodik::test {
namespace {

constexpr double kTolerance = 1e-7;

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

std::string two_link_urdf() {
  return R"(<?xml version="1.0"?>
<robot name="acceleration_relative_pose_test_robot">
  <link name="base_link"/>
  <link name="link1">
    <inertial>
      <origin xyz="0.5 0 0" rpy="0 0 0"/>
      <mass value="2.0"/>
      <inertia ixx="0.1" ixy="0" ixz="0" iyy="0.2" iyz="0" izz="0.1"/>
    </inertial>
  </link>
  <joint name="joint1" type="revolute">
    <parent link="base_link"/>
    <child link="link1"/>
    <origin xyz="0 0 0" rpy="0 0 0"/>
    <axis xyz="0 0 1"/>
    <limit lower="-6.4" upper="6.4" velocity="500.0" effort="100.0"/>
  </joint>
  <link name="tip">
    <inertial>
      <origin xyz="0.5 0 0" rpy="0 0 0"/>
      <mass value="1.0"/>
      <inertia ixx="0.05" ixy="0" ixz="0" iyy="0.1" iyz="0" izz="0.05"/>
    </inertial>
  </link>
  <joint name="joint2" type="revolute">
    <parent link="link1"/>
    <child link="tip"/>
    <origin xyz="1 0 0" rpy="0 0 0"/>
    <axis xyz="0 0 1"/>
    <limit lower="-6.4" upper="6.4" velocity="500.0" effort="100.0"/>
  </joint>
  <link name="tool"/>
  <joint name="tool_fixed" type="fixed">
    <parent link="tip"/>
    <child link="tool"/>
    <origin xyz="1 0 0" rpy="0 0 0"/>
  </joint>
</robot>)";
}

GeometricConstraintAccelerationPolicy policy6(double rate,
                                              double acceleration,
                                              double braking) {
  GeometricConstraintAccelerationPolicy policy;
  policy.rate_limits = Eigen::VectorXd::Constant(6, rate);
  policy.acceleration_limits = Eigen::VectorXd::Constant(6, acceleration);
  policy.lower_braking_accelerations =
      Eigen::VectorXd::Constant(6, braking);
  policy.upper_braking_accelerations =
      Eigen::VectorXd::Constant(6, braking);
  return policy;
}

RelativePoseAccelerationConstraint relative_pose(
    const std::string &source_id, const std::string &frame_a,
    const std::string &frame_b,
    const Eigen::Matrix<double, 6, 1> &lower,
    const Eigen::Matrix<double, 6, 1> &upper,
    const Eigen::Matrix<double, 6, 1> &mask,
    const GeometricConstraintAccelerationPolicy &policy) {
  RelativePoseAccelerationConstraint constraint;
  constraint.source_id = source_id;
  constraint.definition.frame_a = frame_a;
  constraint.definition.frame_b = frame_b;
  constraint.definition.lower_bounds = lower;
  constraint.definition.upper_bounds = upper;
  constraint.definition.axis_mask = mask;
  constraint.policy = policy;
  return constraint;
}

AccelerationSolveOptions options_with_limits(double limit) {
  AccelerationSolveOptions options;
  options.acceleration_limits_override = vector({limit, limit});
  options.apply_position_limits = false;
  options.apply_velocity_limits = false;
  return options;
}

detail::ScalarGeometricPathSegment so3_segment() {
  detail::ScalarGeometricPathSegment segment;
  segment.previous.tangent = Eigen::Vector2d::Zero();
  segment.current.tangent = Eigen::Vector2d::Zero();
  segment.previous.coordinates.value = Eigen::VectorXd::Zero(6);
  segment.current.coordinates.value = Eigen::VectorXd::Zero(6);
  segment.previous.coordinates.so3_rotation = Eigen::Matrix3d::Identity();
  segment.current.coordinates.so3_rotation = Eigen::Matrix3d::Identity();
  return segment;
}

class AccelerationSolverRelativePoseTest : public ::testing::Test {
protected:
  void SetUp() override {
    urdf_ = std::make_unique<ScopedUrdf>(
        "embodik_acceleration_relative_pose_test.urdf", two_link_urdf());
    robot_ = std::make_shared<RobotModel>(urdf_->path(), false);
    robot_->set_gravity(Eigen::Vector3d::Zero());
  }

  std::unique_ptr<ScopedUrdf> urdf_;
  std::shared_ptr<RobotModel> robot_;
};

TEST_F(AccelerationSolverRelativePoseTest, CapabilityFlagIsTruthful) {
  EXPECT_TRUE(AccelerationSolver::capabilities().supports_relative_pose_constraints);
}

TEST_F(AccelerationSolverRelativePoseTest,
       LoweredRowsUseExactRelativeTranslationAndSo3Differential) {
  const Eigen::VectorXd q = vector({0.35, -0.2});
  const Eigen::VectorXd dq = vector({0.7, -0.15});
  robot_->update_kinematics(q, dq);
  const auto exact =
      detail::evaluate_relative_pose_differential(*robot_, "link1", "tool");

  Eigen::Matrix<double, 6, 1> lower =
      Eigen::Matrix<double, 6, 1>::Constant(-10.0);
  Eigen::Matrix<double, 6, 1> upper =
      Eigen::Matrix<double, 6, 1>::Constant(10.0);
  Eigen::Matrix<double, 6, 1> mask = Eigen::Matrix<double, 6, 1>::Zero();
  mask(0) = 1.0;
  mask(5) = 1.0;

  const auto prepared = detail::prepare_relative_pose_constraint(
      relative_pose("relative_x_yaw", "link1", "tool", lower, upper, mask,
                    policy6(100.0, 100.0, 100.0)),
      *robot_, 0.01, Eigen::Vector2d::Constant(-1000.0),
      Eigen::Vector2d::Constant(1000.0));

  ASSERT_TRUE(prepared.satisfied()) << prepared.message;
  ASSERT_GT(exact.value.head<3>().norm(), 1e-3);
  ASSERT_GT(exact.value.tail<3>().norm(), 1e-3);
  const auto &physical =
      prepared.prepared->scalar.state_box.physical_constraint;
  ASSERT_EQ(physical.coefficient_matrix.rows(), 2);
  EXPECT_TRUE(physical.coefficient_matrix.row(0).isApprox(
      exact.jacobian.row(0), kTolerance));
  EXPECT_TRUE(physical.coefficient_matrix.row(1).isApprox(
      exact.jacobian.row(5), kTolerance));
  EXPECT_NEAR(physical.affine_bias(0), exact.affine_bias(0), kTolerance);
  EXPECT_NEAR(physical.affine_bias(1), exact.affine_bias(5), kTolerance);
}

TEST_F(AccelerationSolverRelativePoseTest,
       TranslationOnlyBypassesCurrentNearPiRelativeOrientation) {
  AccelerationSolver solver(robot_);
  const Eigen::VectorXd q = vector({0.0, detail::kSo3Pi - 5e-4});
  const Eigen::VectorXd dq = Eigen::Vector2d::Zero();
  robot_->update_kinematics(q, dq);
  const auto exact_translation =
      detail::evaluate_relative_pose_translation_differential(
          *robot_, "link1", "tool");

  Eigen::Matrix<double, 6, 1> lower =
      Eigen::Matrix<double, 6, 1>::Constant(-10.0);
  Eigen::Matrix<double, 6, 1> upper =
      Eigen::Matrix<double, 6, 1>::Constant(10.0);
  lower.head<3>() = exact_translation.value.array() - 0.1;
  upper.head<3>() = exact_translation.value.array() + 0.1;
  Eigen::Matrix<double, 6, 1> mask = Eigen::Matrix<double, 6, 1>::Zero();
  mask(1) = 1.0;

  auto options = options_with_limits(100.0);
  options.relative_pose_constraints.push_back(relative_pose(
      "translation_only", "link1", "tool", lower, upper, mask,
      policy6(100.0, 100.0, 1e-12)));

  const auto result = solver.solve(q, dq, 0.01, options);

  EXPECT_EQ(result.status, SolverStatus::kSuccess) << result.status_message;
}

TEST_F(AccelerationSolverRelativePoseTest,
       RotationActiveCurrentNearPiFailsAsInvalidInputAndClearsOutputs) {
  AccelerationSolver solver(robot_);
  Eigen::Matrix<double, 6, 1> lower =
      Eigen::Matrix<double, 6, 1>::Constant(-10.0);
  Eigen::Matrix<double, 6, 1> upper =
      Eigen::Matrix<double, 6, 1>::Constant(10.0);
  Eigen::Matrix<double, 6, 1> yaw_mask = Eigen::Matrix<double, 6, 1>::Zero();
  yaw_mask(5) = 1.0;
  auto options = options_with_limits(100.0);
  options.relative_pose_constraints.push_back(relative_pose(
      "yaw_near_pi", "link1", "tool", lower, upper, yaw_mask,
      policy6(100.0, 100.0, 100.0)));

  const auto result =
      solver.solve(vector({0.0, detail::kSo3Pi - 5e-4}),
                   Eigen::Vector2d::Zero(), 0.01, options);

  EXPECT_EQ(result.status, SolverStatus::kInvalidInput);
  EXPECT_NE(result.status_message.find("near-pi"), std::string::npos);
  expect_no_motion_outputs(result);
}

TEST_F(AccelerationSolverRelativePoseTest,
       RotationActiveBetweenSampleNearPiFailsAsNumericalAndClearsOutputs) {
  AccelerationSolver solver(robot_);
  Eigen::Matrix<double, 6, 1> lower =
      Eigen::Matrix<double, 6, 1>::Constant(-10.0);
  Eigen::Matrix<double, 6, 1> upper =
      Eigen::Matrix<double, 6, 1>::Constant(10.0);
  Eigen::Matrix<double, 6, 1> yaw_mask = Eigen::Matrix<double, 6, 1>::Zero();
  yaw_mask(5) = 1.0;
  auto options = options_with_limits(100.0);
  options.relative_pose_constraints.push_back(relative_pose(
      "yaw_segment_guard", "link1", "tool", lower, upper, yaw_mask,
      policy6(1000.0, 100.0, 100.0)));

  const auto result =
      solver.solve(vector({0.0, detail::kSo3Pi - 2e-3}), vector({0.0, 0.8}),
                   0.01, options);

  EXPECT_EQ(result.status, SolverStatus::kNumericalError)
      << result.status_message;
  EXPECT_NE(result.status_message.find("SO(3)"), std::string::npos);
  expect_no_motion_outputs(result);
}

TEST_F(AccelerationSolverRelativePoseTest,
       MultipleRecordsAsymmetricBoundsAndTwoMovingFramesSolve) {
  AccelerationSolver solver(robot_);
  solver.add_posture_task("posture");
  AccelerationTaskReference reference;
  reference.desired_acceleration = vector({0.0, 20.0});
  solver.set_task_reference("posture", reference);

  const Eigen::VectorXd q = vector({0.25, 0.0});
  const Eigen::VectorXd dq = Eigen::Vector2d::Zero();
  robot_->update_kinematics(q, dq);
  const auto exact =
      detail::evaluate_relative_pose_differential(*robot_, "link1", "tool");

  Eigen::Matrix<double, 6, 1> lower =
      Eigen::Matrix<double, 6, 1>::Constant(-10.0);
  Eigen::Matrix<double, 6, 1> upper =
      Eigen::Matrix<double, 6, 1>::Constant(10.0);
  lower(5) = exact.value(5) - 0.05;
  upper(5) = exact.value(5) + 0.0001;
  Eigen::Matrix<double, 6, 1> xy_mask = Eigen::Matrix<double, 6, 1>::Zero();
  xy_mask(0) = 1.0;
  xy_mask(1) = 1.0;
  Eigen::Matrix<double, 6, 1> yaw_mask = Eigen::Matrix<double, 6, 1>::Zero();
  yaw_mask(5) = 1.0;

  auto options = options_with_limits(100.0);
  options.relative_pose_constraints.push_back(relative_pose(
      "relative_xy", "link1", "tool", lower, upper, xy_mask,
      policy6(100.0, 100.0, 1e-12)));
  options.relative_pose_constraints.push_back(relative_pose(
      "relative_yaw_asymmetric", "link1", "tool", lower, upper, yaw_mask,
      policy6(100.0, 100.0, 1e-12)));

  const auto result = solver.solve(q, dq, 0.01, options);

  ASSERT_EQ(result.status, SolverStatus::kSuccess) << result.status_message;
  EXPECT_LE(result.joint_accelerations(1), 2.0 + 1e-6);
}

TEST_F(AccelerationSolverRelativePoseTest,
       CurrentAndPredictedSupportRejectionsClearOutputs) {
  AccelerationSolver solver(robot_);
  Eigen::Matrix<double, 6, 1> lower =
      Eigen::Matrix<double, 6, 1>::Constant(-10.0);
  Eigen::Matrix<double, 6, 1> upper =
      Eigen::Matrix<double, 6, 1>::Constant(10.0);
  Eigen::Matrix<double, 6, 1> z_mask = Eigen::Matrix<double, 6, 1>::Zero();
  z_mask(2) = 1.0;
  auto options = options_with_limits(100.0);
  options.relative_pose_constraints.push_back(relative_pose(
      "unsupported_z", "link1", "tool", lower, upper, z_mask,
      policy6(100.0, 100.0, 100.0)));

  auto result =
      solver.solve(vector({0.0, 0.0}), Eigen::Vector2d::Zero(), 0.01, options);
  EXPECT_EQ(result.status, SolverStatus::kInfeasible)
      << result.status_message;
  EXPECT_NE(result.status_message.find("current-state"), std::string::npos);
  expect_no_motion_outputs(result);

  AccelerationSolver predicted_solver(robot_);
  predicted_solver.add_posture_task("posture");
  const Eigen::VectorXd q = vector({0.2, -0.2});
  constexpr double kDt = 0.1;
  AccelerationTaskReference reference;
  reference.desired_acceleration = vector({0.0, -2.0 * q(1) / (kDt * kDt)});
  predicted_solver.set_task_reference("posture", reference);
  robot_->update_kinematics(q, Eigen::Vector2d::Zero());
  const auto exact =
      detail::evaluate_relative_pose_differential(*robot_, "link1", "tool");
  Eigen::Matrix<double, 6, 1> x_mask = Eigen::Matrix<double, 6, 1>::Zero();
  x_mask(0) = 1.0;
  auto predicted_options = options_with_limits(1e6);
  predicted_options.relative_pose_constraints.push_back(relative_pose(
      "predicted_singular_x", "link1", "tool",
      exact.value.array() - 0.1, exact.value.array() + 0.1, x_mask,
      policy6(100.0, 1e5, 1000.0)));

  result = predicted_solver.solve(q, Eigen::Vector2d::Zero(), kDt,
                                  predicted_options);
  EXPECT_EQ(result.status, SolverStatus::kNumericalError)
      << result.status_message;
  EXPECT_NE(result.status_message.find("predicted next-state"),
            std::string::npos);
  EXPECT_NE(result.status_message.find("braking support"), std::string::npos);
  expect_no_motion_outputs(result);
}

TEST_F(AccelerationSolverRelativePoseTest,
       ComposesWithAllocationEffortLocksTaskExclusionAndContact) {
  AccelerationSolver solver(robot_);
  auto task =
      solver.add_frame_task("tool_task", "tool", TaskType::FRAME_POSITION);
  task->set_excluded_joint_indices({0});
  task->setTargetPosition(Eigen::Vector3d(5.0, 0.0, 0.0));

  Eigen::Matrix<double, 6, 1> lower =
      Eigen::Matrix<double, 6, 1>::Constant(-10.0);
  Eigen::Matrix<double, 6, 1> upper =
      Eigen::Matrix<double, 6, 1>::Constant(10.0);
  lower(5) = 0.0;
  upper(5) = 0.0;
  Eigen::Matrix<double, 6, 1> yaw_mask = Eigen::Matrix<double, 6, 1>::Zero();
  yaw_mask(5) = 1.0;

  auto options = options_with_limits(100.0);
  GeneralizedAccelerationAllocation allocation;
  allocation.metric_diagonal = vector({2.0, 0.5});
  allocation.reference_acceleration = vector({4.0, -3.0});
  options.generalized_acceleration_allocation = allocation;
  options.effort_constraints = EffortConstraintOptions{};
  options.effort_constraints->limits_override = vector({1e9, 1e9});
  options.zero_acceleration_joint_indices = {0};
  options.contact_acceleration_constraints.push_back(
      ContactAccelerationConstraint{"link1_point_contact", "link1",
                                    ContactType::kPointContact});
  options.relative_pose_constraints.push_back(relative_pose(
      "relative_yaw_lock", "link1", "tool", lower, upper, yaw_mask,
      policy6(100.0, 100.0, 100.0)));

  const auto result = solver.solve(vector({0.0, 0.0}), Eigen::Vector2d::Zero(),
                                   0.01, options);

  ASSERT_EQ(result.status, SolverStatus::kSuccess) << result.status_message;
  EXPECT_TRUE(result.allocation_diagnostics.applied);
  EXPECT_EQ(result.predicted_torques.size(), robot_->nv());
  EXPECT_NEAR(result.joint_accelerations(0), 0.0, kTolerance);
  EXPECT_NEAR(result.joint_accelerations(1), 0.0, kTolerance);
}

TEST_F(AccelerationSolverRelativePoseTest,
       RejectsMalformedInputsAndDuplicateSourceIds) {
  AccelerationSolver solver(robot_);
  const Eigen::VectorXd q = vector({0.0, 0.0});
  const Eigen::VectorXd dq = Eigen::Vector2d::Zero();
  Eigen::Matrix<double, 6, 1> lower =
      Eigen::Matrix<double, 6, 1>::Constant(-1.0);
  Eigen::Matrix<double, 6, 1> upper =
      Eigen::Matrix<double, 6, 1>::Constant(1.0);
  Eigen::Matrix<double, 6, 1> mask = Eigen::Matrix<double, 6, 1>::Zero();
  mask(0) = 1.0;

  auto options = options_with_limits(100.0);
  options.relative_pose_constraints.push_back(relative_pose(
      "", "link1", "tool", lower, upper, mask, policy6(10.0, 10.0, 10.0)));
  auto result = solver.solve(q, dq, 0.01, options);
  EXPECT_EQ(result.status, SolverStatus::kInvalidInput);
  expect_no_motion_outputs(result);

  options = options_with_limits(100.0);
  options.relative_pose_constraints.push_back(relative_pose(
      "same_frames", "tool", "tool", lower, upper, mask,
      policy6(10.0, 10.0, 10.0)));
  result = solver.solve(q, dq, 0.01, options);
  EXPECT_EQ(result.status, SolverStatus::kInvalidInput);

  options = options_with_limits(100.0);
  options.relative_pose_constraints.push_back(relative_pose(
      "unknown", "link1", "missing", lower, upper, mask,
      policy6(10.0, 10.0, 10.0)));
  result = solver.solve(q, dq, 0.01, options);
  EXPECT_EQ(result.status, SolverStatus::kInvalidInput);

  auto bad_lower = lower;
  bad_lower(0) = std::numeric_limits<double>::quiet_NaN();
  options = options_with_limits(100.0);
  options.relative_pose_constraints.push_back(relative_pose(
      "nonfinite", "link1", "tool", bad_lower, upper, mask,
      policy6(10.0, 10.0, 10.0)));
  result = solver.solve(q, dq, 0.01, options);
  EXPECT_EQ(result.status, SolverStatus::kNonFiniteInput);

  auto bad_upper = upper;
  bad_upper(0) = -2.0;
  options = options_with_limits(100.0);
  options.relative_pose_constraints.push_back(relative_pose(
      "bad_bounds", "link1", "tool", lower, bad_upper, mask,
      policy6(10.0, 10.0, 10.0)));
  result = solver.solve(q, dq, 0.01, options);
  EXPECT_EQ(result.status, SolverStatus::kInvalidInput);

  auto bad_mask = mask;
  bad_mask(0) = 0.5;
  options = options_with_limits(100.0);
  options.relative_pose_constraints.push_back(relative_pose(
      "bad_mask", "link1", "tool", lower, upper, bad_mask,
      policy6(10.0, 10.0, 10.0)));
  result = solver.solve(q, dq, 0.01, options);
  EXPECT_EQ(result.status, SolverStatus::kInvalidInput);

  auto bad_policy = policy6(10.0, 10.0, 10.0);
  bad_policy.rate_limits(0) = std::numeric_limits<double>::quiet_NaN();
  options = options_with_limits(100.0);
  options.relative_pose_constraints.push_back(relative_pose(
      "bad_policy", "link1", "tool", lower, upper, mask, bad_policy));
  result = solver.solve(q, dq, 0.01, options);
  EXPECT_EQ(result.status, SolverStatus::kNonFiniteInput);

  options = options_with_limits(100.0);
  AffineAccelerationConstraint affine;
  affine.source_id = "dup";
  affine.coefficient_matrix = Eigen::MatrixXd::Identity(1, 2);
  affine.affine_bias = Eigen::VectorXd::Zero(1);
  affine.lower_bounds = vector({-1.0});
  affine.upper_bounds = vector({1.0});
  options.affine_constraints.push_back(affine);
  options.relative_pose_constraints.push_back(relative_pose(
      "dup", "link1", "tool", lower, upper, mask,
      policy6(10.0, 10.0, 10.0)));
  result = solver.solve(q, dq, 0.01, options);
  EXPECT_EQ(result.status, SolverStatus::kInvalidInput);
  expect_no_motion_outputs(result);
}

TEST_F(AccelerationSolverRelativePoseTest,
       RelativeSo3SegmentGuardMasksCommonUpstreamMotionOnly) {
  const auto mask =
      detail::relative_pose_angular_path_support_mask(*robot_, "link1", "tool");
  ASSERT_EQ(mask.size(), 2);
  EXPECT_EQ(mask(0), 0.0);
  EXPECT_EQ(mask(1), 1.0);

  auto common_motion = so3_segment();
  common_motion.current.tangent = vector({detail::kSo3Pi, 0.0});
  auto guard =
      detail::validate_so3_log_rotation_segment(common_motion, mask);
  EXPECT_TRUE(guard.satisfied()) << guard.message;

  auto exclusive_motion = so3_segment();
  exclusive_motion.current.tangent = vector({0.0, detail::kSo3Pi});
  guard = detail::validate_so3_log_rotation_segment(exclusive_motion, mask);
  EXPECT_EQ(guard.status, SolverStatus::kNumericalError);
  EXPECT_NE(guard.message.find("near-pi"), std::string::npos);
}

} // namespace
} // namespace embodik::test
